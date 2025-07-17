"""
Quantitative Evaluation of Van Gogh Concept Unlearning Results
Compare SAE vs MSAE performance across multiple metrics
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import logging
import os
from datetime import datetime
import argparse
import json
from pathlib import Path
import clip
from sklearn.metrics.pairwise import cosine_similarity
import cv2

def setup_logging(log_dir="log/evaluation_log"):
    """Setup logging for evaluation"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"evaluation_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Evaluation logging initialized. Log file: {log_file}")
    return log_file

class UnlearningEvaluator:
    """Comprehensive evaluation of concept unlearning results"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load CLIP model for semantic evaluation
        logging.info("Loading CLIP model for semantic evaluation...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Transform for image processing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Van Gogh style descriptors for CLIP evaluation
        self.van_gogh_descriptors = [
            "Van Gogh painting style",
            "thick impasto brushstrokes",
            "swirling paint texture",
            "post-impressionist style",
            "Van Gogh's distinctive brushwork",
            "expressive paint application"
        ]
        
        # Realistic style descriptors
        self.realistic_descriptors = [
            "photorealistic painting",
            "smooth brushwork",
            "classical realism style",
            "detailed realistic art",
            "traditional painting technique"
        ]
        
    def load_image_pairs(self, results_dir):
        """Load original and unlearned image pairs"""
        original_dir = Path(results_dir) / "original"
        unlearned_dir = Path(results_dir) / "unlearned"
        
        if not original_dir.exists() or not unlearned_dir.exists():
            logging.error(f"Results directory not found: {results_dir}")
            return []
        
        image_pairs = []
        original_files = sorted(original_dir.glob("*.png"))
        
        for original_file in original_files:
            unlearned_file = unlearned_dir / original_file.name
            if unlearned_file.exists():
                image_pairs.append({
                    'original': str(original_file),
                    'unlearned': str(unlearned_file),
                    'name': original_file.stem
                })
        
        logging.info(f"Loaded {len(image_pairs)} image pairs from {results_dir}")
        return image_pairs
    
    def calculate_clip_similarity(self, image_path, text_descriptors):
        """Calculate CLIP similarity between image and text descriptors"""
        image = Image.open(image_path).convert('RGB')
        image_preprocessed = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        
        # Encode text descriptors
        text_tokens = clip.tokenize(text_descriptors).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_preprocessed)
            text_features = self.clip_model.encode_text(text_tokens)
            
            # Calculate similarities
            similarities = torch.cosine_similarity(image_features, text_features, dim=1)
            
        return similarities.cpu().numpy()
    
    def calculate_style_removal_effectiveness(self, image_pair):
        """Calculate how effectively Van Gogh style was removed"""
        original_van_gogh_sim = self.calculate_clip_similarity(
            image_pair['original'], self.van_gogh_descriptors
        )
        unlearned_van_gogh_sim = self.calculate_clip_similarity(
            image_pair['unlearned'], self.van_gogh_descriptors
        )
        
        # Calculate reduction in Van Gogh similarity
        style_reduction = np.mean(original_van_gogh_sim - unlearned_van_gogh_sim)
        
        return {
            'original_van_gogh_sim': np.mean(original_van_gogh_sim),
            'unlearned_van_gogh_sim': np.mean(unlearned_van_gogh_sim),
            'style_reduction': style_reduction,
            'effectiveness_score': max(0, style_reduction)  # Positive values indicate successful removal
        }
    
    def calculate_content_preservation(self, image_pair):
        """Calculate how well content was preserved during unlearning"""
        # Load images
        original_img = Image.open(image_pair['original']).convert('RGB')
        unlearned_img = Image.open(image_pair['unlearned']).convert('RGB')
        
        # Convert to tensors
        original_tensor = self.transform(original_img).unsqueeze(0).to(self.device)
        unlearned_tensor = self.transform(unlearned_img).unsqueeze(0).to(self.device)
        
        # Calculate CLIP feature similarity (content preservation)
        with torch.no_grad():
            original_features = self.clip_model.encode_image(original_tensor)
            unlearned_features = self.clip_model.encode_image(unlearned_tensor)
            
            content_similarity = torch.cosine_similarity(
                original_features, unlearned_features, dim=1
            ).item()
        
        # Calculate pixel-level metrics
        original_np = np.array(original_img)
        unlearned_np = np.array(unlearned_img)
        
        # MSE and PSNR
        mse = np.mean((original_np - unlearned_np) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        # Structural similarity (simplified)
        ssim = self.calculate_ssim(original_np, unlearned_np)
        
        return {
            'content_similarity': content_similarity,
            'mse': mse,
            'psnr': psnr,
            'ssim': ssim
        }
    
    def calculate_ssim(self, img1, img2):
        """Calculate SSIM between two images"""
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        # Calculate SSIM
        from skimage.metrics import structural_similarity
        ssim_score = structural_similarity(gray1, gray2, data_range=255)
        
        return ssim_score
    
    def evaluate_model_results(self, results_dir, model_name):
        """Comprehensive evaluation of a single model's results"""
        logging.info(f"Evaluating {model_name} results from {results_dir}")
        
        image_pairs = self.load_image_pairs(results_dir)
        if not image_pairs:
            return None
        
        all_results = []
        
        for pair in image_pairs:
            logging.info(f"Evaluating {pair['name']}...")
            
            # Style removal effectiveness
            style_metrics = self.calculate_style_removal_effectiveness(pair)
            
            # Content preservation
            content_metrics = self.calculate_content_preservation(pair)
            
            # Combine results
            result = {
                'image_name': pair['name'],
                'model': model_name,
                **style_metrics,
                **content_metrics
            }
            
            all_results.append(result)
            
            logging.info(f"  Style reduction: {style_metrics['style_reduction']:.4f}")
            logging.info(f"  Content preservation: {content_metrics['content_similarity']:.4f}")
        
        # Calculate aggregate metrics
        aggregate_metrics = {
            'model': model_name,
            'num_images': len(all_results),
            'avg_style_reduction': np.mean([r['style_reduction'] for r in all_results]),
            'avg_effectiveness_score': np.mean([r['effectiveness_score'] for r in all_results]),
            'avg_content_similarity': np.mean([r['content_similarity'] for r in all_results]),
            'avg_psnr': np.mean([r['psnr'] for r in all_results]),
            'avg_ssim': np.mean([r['ssim'] for r in all_results]),
            'success_rate': np.mean([r['effectiveness_score'] > 0 for r in all_results])
        }
        
        logging.info(f"\n{model_name} Aggregate Results:")
        logging.info(f"  Average style reduction: {aggregate_metrics['avg_style_reduction']:.4f}")
        logging.info(f"  Average effectiveness score: {aggregate_metrics['avg_effectiveness_score']:.4f}")
        logging.info(f"  Average content similarity: {aggregate_metrics['avg_content_similarity']:.4f}")
        logging.info(f"  Average PSNR: {aggregate_metrics['avg_psnr']:.2f}")
        logging.info(f"  Average SSIM: {aggregate_metrics['avg_ssim']:.4f}")
        logging.info(f"  Success rate: {aggregate_metrics['success_rate']:.1%}")
        
        return all_results, aggregate_metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate concept unlearning results")
    parser.add_argument("--results_dir", type=str, default="concept_unlearning_results",
                       help="Directory containing results (default: concept_unlearning_results)")
    parser.add_argument("--models", nargs="+", default=["sae", "msae"],
                       help="Models to evaluate (default: [sae, msae])")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json",
                       help="Output file for results (default: evaluation_results.json)")
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging()
    
    logging.info("=== COMPREHENSIVE UNLEARNING EVALUATION ===")
    logging.info(f"Results directory: {args.results_dir}")
    logging.info(f"Models to evaluate: {args.models}")
    
    # Initialize evaluator
    evaluator = UnlearningEvaluator()
    
    # Evaluate each model
    all_detailed_results = []
    all_aggregate_results = []
    
    for model in args.models:
        model_results_dir = Path(args.results_dir) / model
        if not model_results_dir.exists():
            logging.warning(f"Results directory not found for {model}: {model_results_dir}")
            continue
            
        detailed_results, aggregate_results = evaluator.evaluate_model_results(
            str(model_results_dir), model
        )
        
        if detailed_results and aggregate_results:
            all_detailed_results.extend(detailed_results)
            all_aggregate_results.append(aggregate_results)
    
    # Compare models
    if len(all_aggregate_results) > 1:
        logging.info("\n" + "="*60)
        logging.info("MODEL COMPARISON")
        logging.info("="*60)
        
        # Sort by effectiveness score
        sorted_models = sorted(all_aggregate_results, 
                             key=lambda x: x['avg_effectiveness_score'], 
                             reverse=True)
        
        logging.info("🏆 Ranking by Style Removal Effectiveness:")
        for i, model in enumerate(sorted_models):
            logging.info(f"  {i+1}. {model['model']}: {model['avg_effectiveness_score']:.4f}")
        
        # Sort by content preservation
        sorted_by_content = sorted(all_aggregate_results,
                                 key=lambda x: x['avg_content_similarity'],
                                 reverse=True)
        
        logging.info("\n🏆 Ranking by Content Preservation:")
        for i, model in enumerate(sorted_by_content):
            logging.info(f"  {i+1}. {model['model']}: {model['avg_content_similarity']:.4f}")
        
        # Overall winner (balanced score)
        for model in all_aggregate_results:
            model['balanced_score'] = (
                0.6 * model['avg_effectiveness_score'] + 
                0.4 * model['avg_content_similarity']
            )
        
        sorted_balanced = sorted(all_aggregate_results,
                               key=lambda x: x['balanced_score'],
                               reverse=True)
        
        logging.info("\n🏆 Overall Winner (Balanced Score):")
        for i, model in enumerate(sorted_balanced):
            logging.info(f"  {i+1}. {model['model']}: {model['balanced_score']:.4f} "
                        f"(effectiveness: {model['avg_effectiveness_score']:.4f}, "
                        f"content: {model['avg_content_similarity']:.4f})")
        
        winner = sorted_balanced[0]
        logging.info(f"\n🎯 WINNER: {winner['model']} with balanced score {winner['balanced_score']:.4f}")
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        """Recursively convert numpy types to native Python types"""
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    # Save results with numpy type conversion
    output_data = {
        'detailed_results': convert_numpy_types(all_detailed_results),
        'aggregate_results': convert_numpy_types(all_aggregate_results),
        'evaluation_timestamp': datetime.now().isoformat(),
        'log_file': log_file
    }

    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logging.info(f"\n✅ Evaluation completed!")
    logging.info(f"📊 Results saved to: {args.output_file}")
    logging.info(f"📋 Detailed log: {log_file}")
    
    return all_detailed_results, all_aggregate_results

if __name__ == "__main__":
    detailed_results, aggregate_results = main()
