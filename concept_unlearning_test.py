"""
Van Gogh Concept Unlearning Test
Compare Traditional SAE vs MSAE for concept removal in diffusion models
"""
import torch
import pickle
import numpy as np
import logging
import os
from datetime import datetime
import argparse
from diffusers import StableDiffusionPipeline
from models.traditional_sae import TraditionalSAE
from models.diffusion_msae import DiffusionMSAE
from PIL import Image
import torchvision.transforms as transforms

def setup_logging(log_dir="log/concept_unlearning_log"):
    """Setup logging for unlearning experiments"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"unlearning_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Concept unlearning logging initialized. Log file: {log_file}")
    return log_file

class ConceptUnlearner:
    """Handle concept unlearning with SAE/MSAE models"""
    
    def __init__(self, diffusion_model_path, hook_point="unet.up_blocks.1.attentions.2"):
        self.hook_point = hook_point
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load diffusion model
        logging.info(f"Loading diffusion model from {diffusion_model_path}")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            diffusion_model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        ).to(self.device)
        
        self.sae_model = None
        self.msae_model = None
        self.concept_features = None
        self.hook_handle = None
        
    def load_sae_model(self, checkpoint_path):
        """Load traditional SAE model"""
        logging.info(f"Loading SAE model: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        config = checkpoint['config']
        
        self.sae_model = TraditionalSAE(
            input_dim=config['input_dim'],
            expansion_factor=config['expansion_factor'],
            k=config['k']
        )
        self.sae_model.load_state_dict(checkpoint['model_state_dict'])
        self.sae_model = self.sae_model.to(self.device).eval()
        logging.info(f"✅ SAE model loaded (k={config['k']})")
        
    def load_msae_model(self, checkpoint_path):
        """Load MSAE model"""
        logging.info(f"Loading MSAE model: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        config = checkpoint['config']
        
        self.msae_model = DiffusionMSAE(
            input_dim=config['input_dim'],
            expansion_factor=config['expansion_factor'],
            k_values=config['k_values']
        )
        self.msae_model.load_state_dict(checkpoint['model_state_dict'])
        self.msae_model = self.msae_model.to(self.device).eval()
        logging.info(f"✅ MSAE model loaded (k_values={config['k_values']})")
    
    def compute_concept_features(self, model_type="sae", granularity_level=-1, 
                                multiplier=-5.0, percentile=99.9):
        """Compute concept-specific features for Van Gogh style"""
        logging.info(f"Computing concept features for {model_type}")
        
        # Load activation data and separate by concept
        with open('activations_poc/activations_unet_up_blocks_1_attentions_2.pkl', 'rb') as f:
            activations = pickle.load(f)
        
        van_gogh_activations = []
        realistic_activations = []
        
        # Split by concept (first 250 are Van Gogh, next 250 are realistic)
        for i, activation in enumerate(activations[:100]):  # Use subset for speed
            if i < 50:  # Van Gogh
                for j in range(activation.shape[0]):
                    van_gogh_activations.append(activation[j])
            else:  # Realistic
                for j in range(activation.shape[0]):
                    realistic_activations.append(activation[j])
        
        van_gogh_batch = torch.stack(van_gogh_activations).to(self.device).float()
        realistic_batch = torch.stack(realistic_activations).to(self.device).float()
        
        with torch.no_grad():
            if model_type == "sae":
                # Get SAE features
                van_gogh_features, _ = self.sae_model(van_gogh_batch)
                realistic_features, _ = self.sae_model(realistic_batch)
            else:  # MSAE
                # Get MSAE features at specified granularity
                van_gogh_features_all, _ = self.msae_model(van_gogh_batch)
                realistic_features_all, _ = self.msae_model(realistic_batch)
                
                van_gogh_features = van_gogh_features_all[granularity_level]
                realistic_features = realistic_features_all[granularity_level]
        
        # Calculate feature importance
        van_gogh_mean = van_gogh_features.mean(dim=0)
        realistic_mean = realistic_features.mean(dim=0)
        feature_importance = van_gogh_mean - realistic_mean
        
        # Select top features based on percentile
        threshold = torch.quantile(feature_importance, percentile/100.0)
        selected_features = (feature_importance >= threshold).nonzero().flatten()
        
        logging.info(f"Selected {len(selected_features)} features for concept ablation")
        logging.info(f"Importance threshold: {threshold.item():.4f}")
        logging.info(f"Multiplier: {multiplier}")
        
        self.concept_features = {
            'indices': selected_features,
            'multiplier': multiplier,
            'model_type': model_type,
            'granularity_level': granularity_level
        }
        
        return selected_features
    
    def setup_intervention_hook(self):
        """Setup hook for real-time feature intervention"""
        
        def intervention_hook(module, input, output):
            if self.concept_features is None:
                return output
                
            # Handle tuple output
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Convert to float32 for processing
            original_dtype = hidden_states.dtype
            hidden_states = hidden_states.float()
            
            # Get model features
            with torch.no_grad():
                if self.concept_features['model_type'] == "sae":
                    features, _ = self.sae_model(hidden_states)
                else:  # MSAE
                    features_all, _ = self.msae_model(hidden_states)
                    features = features_all[self.concept_features['granularity_level']]
                
                # Apply intervention
                modified_features = features.clone()
                modified_features[:, self.concept_features['indices']] *= self.concept_features['multiplier']
                
                # Reconstruct
                if self.concept_features['model_type'] == "sae":
                    modified_hidden = self.sae_model.decode(modified_features, hidden_states.shape)
                else:  # MSAE
                    modified_hidden = self.msae_model.decode(modified_features, hidden_states.shape)
            
            # Convert back to original dtype
            modified_hidden = modified_hidden.to(original_dtype)
            
            if isinstance(output, tuple):
                return (modified_hidden,) + output[1:]
            else:
                return modified_hidden
        
        # Register hook
        module = self.pipe.unet
        for part in self.hook_point.split('.')[1:]:
            module = getattr(module, part)
        
        self.hook_handle = module.register_forward_hook(intervention_hook)
        logging.info(f"Intervention hook registered at: {self.hook_point}")
    
    def remove_hook(self):
        """Remove the intervention hook"""
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None
            logging.info("Intervention hook removed")
    
    def generate_comparison_images(self, prompts, save_dir, num_inference_steps=20):
        """Generate images with and without concept unlearning"""
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"{save_dir}/original", exist_ok=True)
        os.makedirs(f"{save_dir}/unlearned", exist_ok=True)
        
        results = []
        
        for i, prompt in enumerate(prompts):
            logging.info(f"Generating images for prompt {i+1}/{len(prompts)}: {prompt}")
            
            # Generate original image
            self.remove_hook()  # Ensure no intervention
            with torch.no_grad():
                original_image = self.pipe(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    generator=torch.Generator().manual_seed(42)  # Fixed seed for comparison
                ).images[0]
            
            # Generate unlearned image
            self.setup_intervention_hook()
            with torch.no_grad():
                unlearned_image = self.pipe(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    generator=torch.Generator().manual_seed(42)  # Same seed
                ).images[0]
            
            # Save images
            original_path = f"{save_dir}/original/prompt_{i:03d}.png"
            unlearned_path = f"{save_dir}/unlearned/prompt_{i:03d}.png"
            
            original_image.save(original_path)
            unlearned_image.save(unlearned_path)
            
            results.append({
                'prompt': prompt,
                'original_path': original_path,
                'unlearned_path': unlearned_path
            })
            
            logging.info(f"  Saved: {original_path} and {unlearned_path}")
        
        self.remove_hook()
        return results

def main():
    parser = argparse.ArgumentParser(description="Van Gogh concept unlearning test")
    parser.add_argument("--model_type", choices=["sae", "msae"], required=True,
                       help="Type of model to use for unlearning")
    parser.add_argument("--sae_k", type=int, default=32,
                       help="K value for SAE model (default: 32)")
    parser.add_argument("--msae_granularity", type=int, default=-1,
                       help="MSAE granularity level to use (-1 for finest, default: -1)")
    parser.add_argument("--expansion_factor", type=int, default=16,
                       help="Expansion factor (default: 16)")
    parser.add_argument("--multiplier", type=float, default=-5.0,
                       help="Feature ablation multiplier (default: -5.0)")
    parser.add_argument("--percentile", type=float, default=99.9,
                       help="Percentile for feature selection (default: 99.9)")
    parser.add_argument("--output_dir", type=str, default="concept_unlearning_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging()
    
    logging.info("=== VAN GOGH CONCEPT UNLEARNING TEST ===")
    logging.info(f"Model type: {args.model_type}")
    logging.info(f"Multiplier: {args.multiplier}")
    logging.info(f"Percentile: {args.percentile}")
    
    # Initialize unlearner
    unlearner = ConceptUnlearner("model_checkpoints/diffuser/style50")
    
    # Load appropriate model
    if args.model_type == "sae":
        checkpoint_path = f"sae-ckpts/traditional_sae/checkpoint_exp{args.expansion_factor}_k{args.sae_k}.pth"
        unlearner.load_sae_model(checkpoint_path)
        logging.info(f"Using SAE k={args.sae_k}")
    else:
        checkpoint_path = f"sae-ckpts/msae/checkpoint_exp{args.expansion_factor}_k16_32_64.pth"
        unlearner.load_msae_model(checkpoint_path)
        logging.info(f"Using MSAE granularity level {args.msae_granularity}")
    
    # Compute concept features
    unlearner.compute_concept_features(
        model_type=args.model_type,
        granularity_level=args.msae_granularity,
        multiplier=args.multiplier,
        percentile=args.percentile
    )
    
    # Test prompts
    test_prompts = [
        "A self-portrait in Van Gogh's style with thick impasto brushstrokes",
        "Van Gogh style sunflowers with swirling yellow petals",
        "A starry night sky painted in Van Gogh's distinctive swirling style", 
        "Van Gogh style wheat field with dramatic brushwork and movement",
        "A café scene painted in Van Gogh's expressive brushstroke technique",
        "Van Gogh style cypress trees with dark swirling forms",
        "A bedroom interior in Van Gogh's post-impressionist style",
        "Van Gogh style landscape with thick paint and vibrant colors",
        "A portrait of a peasant in Van Gogh's characteristic style",
        "Van Gogh style still life with bold brushstrokes and bright colors"
    ]
    
    # Generate comparison images
    results = unlearner.generate_comparison_images(
        test_prompts, 
        f"{args.output_dir}/{args.model_type}"
    )
    
    logging.info(f"✅ Concept unlearning test completed!")
    logging.info(f"�� Results saved to: {args.output_dir}/{args.model_type}")
    logging.info(f"📋 Log file: {log_file}")
    
    return results

if __name__ == "__main__":
    results = main()
