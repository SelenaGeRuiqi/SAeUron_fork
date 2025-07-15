#!/usr/bin/env python3
"""
Simplified activation collection for MSAE PoC
Collects activations from a single hook point with minimal data
"""
import torch
import os
from diffusers import StableDiffusionPipeline
import numpy as np
from tqdm import tqdm
import pickle

class ActivationCollector:
    def __init__(self, model_path, hook_name, device="cuda"):
        self.hook_name = hook_name
        self.device = device
        self.activations = []
        
        # Load the diffusion model
        print(f"Loading diffusion model from {model_path}")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)
        
        # Register hook
        self._register_hook()
    
    def _register_hook(self):
        """Register hook to collect activations"""
        def hook_fn(module, input, output):
            # Store activation (we'll only keep a subset for PoC)
            if len(self.activations) < 1000:  # Limit to 1000 samples for PoC
                # Handle tuple output from attention layers
                if isinstance(output, tuple):
                    activation = output[0]  # Take the first element (hidden states)
                else:
                    activation = output
                self.activations.append(activation.detach().cpu())
        
        # Navigate to the hook point
        module = self.pipe.unet
        for part in self.hook_name.split('.')[1:]:  # Skip 'unet'
            module = getattr(module, part)
        
        module.register_forward_hook(hook_fn)
        print(f"Hook registered at: {self.hook_name}")
    
    def collect_activations(self, prompts, num_inference_steps=20):
        """Collect activations from a list of prompts"""
        print(f"Collecting activations from {len(prompts)} prompts...")
        
        for prompt in tqdm(prompts):
            if len(self.activations) >= 1000:  # Stop at 1000 for PoC
                break
                
            # Generate image (but we only care about activations)
            with torch.no_grad():
                _ = self.pipe(
                    prompt, 
                    num_inference_steps=num_inference_steps,
                    output_type="latent"  # Faster since we don't need actual images
                )
        
        return self.activations

def main():
    # Configuration for PoC
    model_path = "model_checkpoints/diffuser/style50"
    hook_name = "unet.up_blocks.1.attentions.2"
    output_dir = "activations_poc"
    
    # prompts for PoC
    style_prompts = [
        # Van Gogh Style Prompts (10)
        "A self-portrait in Van Gogh's style with thick impasto brushstrokes",
        "Van Gogh style sunflowers with swirling yellow petals",
        "A starry night sky painted in Van Gogh's distinctive swirling style", 
        "Van Gogh style wheat field with dramatic brushwork and movement",
        "A café scene painted in Van Gogh's expressive brushstroke technique",
        "Van Gogh style cypress trees with dark swirling forms",
        "A bedroom interior in Van Gogh's post-impressionist style",
        "Van Gogh style landscape with thick paint and vibrant colors",
        "A portrait of a peasant in Van Gogh's characteristic style",
        "Van Gogh style still life with bold brushstrokes and bright colors",
        
        # Contrastive/Negative Prompts (10)
        "A photorealistic self-portrait with smooth, detailed brushwork",
        "Realistic sunflowers painted in classical realism style", 
        "A night sky painted in smooth, traditional landscape style",
        "A wheat field painted in realistic impressionist style",
        "A café scene in photorealistic contemporary style",
        "Realistic cypress trees painted in traditional landscape style",
        "A bedroom interior in clean, modern realistic style",
        "A realistic landscape with smooth brushwork and natural colors",
        "A portrait of a peasant in classical realism style",
        "A realistic still life with smooth brushwork and natural lighting"
    ] * 25
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect activations
    collector = ActivationCollector(model_path, hook_name)
    activations = collector.collect_activations(style_prompts)
    
    # Save activations
    output_path = os.path.join(output_dir, f"activations_{hook_name.replace('.', '_')}.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(activations, f)
    
    print(f"Collected {len(activations)} activation samples")
    print(f"Saved to: {output_path}")
    
    # Print some statistics
    if activations:
        sample_shape = activations[0].shape
        print(f"Activation shape: {sample_shape}")
        print(f"Total size: {len(activations) * np.prod(sample_shape) * 4 / 1024**2:.1f} MB")

if __name__ == "__main__":
    main()
