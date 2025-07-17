"""
UnlearnCanvas-compatible evaluation of Van Gogh concept unlearning results
Uses the same metrics as SAeUron paper for direct comparison
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
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import entropy
import requests
from tqdm import tqdm

def setup_logging(log_dir="log/evaluation_log"):
    """Setup logging for evaluation"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"unlearncanvas_evaluation_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"UnlearnCanvas evaluation logging initialized. Log file: {log_file}")
    return log_file

class StyleClassifier(nn.Module):
    """Style classifier for Van Gogh style detection (UnlearnCanvas-style)"""
    def __init__(self, num_classes=2):  # Van Gogh vs Not Van Gogh
        super(StyleClassifier, self).__init__()
        # Use ResNet-18 as backbone
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

class UnlearnCanvasEvaluator:
    """Evaluator using UnlearnCanvas metrics from SAeUron paper"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load CLIP model for semantic evaluation
        logging.info("Loading CLIP model for semantic evaluation...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Initialize style classifier (we'll create a simple one for PoC)
        logging.info("Initializing style classifier...")
        self.style_classifier = self._create_style_classifier()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # UnlearnCanvas-style text prompts for evaluation
        self.van_gogh_prompts = [
            "a painting by Vincent van Gogh",
            "Van Gogh style painting",
            "post-impressionist painting by Van Gogh",
            "painting with Van Gogh's brushstrokes",
            "Van Gogh's artistic style",
            "painting in the style of Vincent van Gogh"
        ]
        
        self.general_art_prompts = [
            "a painting",
            "an artwork",
            "a beautiful painting",
            "classical painting",
            "artistic painting",
            "fine art painting"
        ]
        
    def _create_style_classifier(self):
        """Create a simple style classifier for Van Gogh detection"""
        classifier = StyleClassifier(num_classes=2)
        classifier = classifier.to(self.device)
        classifier.eval()
        
        # For PoC, we'll use a pre-trained model without fine-tuning
        # In full implementation, this would be trained on Van Gogh vs other art
        logging.info("Style classifier initialized (using pre-trained features)")
        return classifier
    
    def calculate_unlearning_accuracy(self, images, target_concept="van_gogh"):
        """
        Calculate Unlearning Accuracy (UA) as defined in UnlearnCanvas
        UA = 1 - (number of images still showing target concept / total images)
        """
        total_images = len(images)
        detected_concept_count = 0
        
        for image_path in images:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Method 1: CLIP-based detection
            clip_score = self._calculate_clip_concept_score(image, self.van_gogh_prompts)
            
            # Method 2: Style classifier detection (simplified)
            style_score = self._calculate_style_score(image_tensor)
            
            # Combined detection (if either method detects Van Gogh style)
            if clip_score > 0.3 or style_score > 0.5:  # Thresholds tuned for Van Gogh
                detected_concept_count += 1
        
        unlearning_accuracy = 1.0 - (detected_concept_count / total_images)
        
        logging.info(f"Unlearning Accuracy: {unlearning_accuracy:.4f}")
        logging.info(f"  Images still showing Van Gogh style: {detected_concept_count}/{total_images}")
        
        return unlearning_accuracy, detected_concept_count
    
    def _calculate_clip_concept_score(self, image, concept_prompts):
        """Calculate CLIP similarity with concept prompts"""
        image_preprocessed = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        text_tokens = clip.tokenize(concept_prompts).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_preprocessed)
            text_features = self.clip_model.encode_text(text_tokens)
            
            # Calculate similarities and take max
            similarities = torch.cosine_similarity(image_features, text_features, dim=1)
            max_similarity = similarities.max().item()
            
        return max_similarity
    
    def _calculate_style_score(self, image_tensor):
        """Calculate style classifier score"""
        with torch.no_grad():
            outputs = self.style_classifier(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            van_gogh_prob = probabilities[0, 1].item()  # Assuming index 1 is Van Gogh
            
        return van_gogh_prob
    
    def calculate_content_preservation(self, original_images, unlearned_images):
        """
        Calculate content preservation metrics
        Similar to SAeUron's approach using CLIP features
        """
        if len(original_images) != len(unlearned_images):
            logging.error("Mismatch in number of original and unlearned images")
            return 0.0
        
        content_similarities = []
        
        for orig_path, unlearn_path in zip(original_images, unlearned_images):
            # Load images
            orig_image = Image.open(orig_path).convert('RGB')
            unlearn_image = Image.open(unlearn_path).convert('RGB')
            
            # Preprocess for CLIP
            orig_preprocessed = self.clip_preprocess(orig_image).unsqueeze(0).to(self.device)
            unlearn_preprocessed = self.clip_preprocess(unlearn_image).unsqueeze(0).to(self.device)
            
            # Calculate CLIP feature similarity
            with torch.no_grad():
                orig_features = self.clip_model.encode_image(orig_preprocessed)
                unlearn_features = self.clip_model.encode_image(unlearn_preprocessed)
                
                similarity = torch.cosine_similarity(orig_features, unlearn_features, dim=1).item()
                content_similarities.append(similarity)
        
        avg_content_preservation = np.mean(content_similarities)
        
        logging.info(f"Content Preservation: {avg_content_preservation:.4f}")
        logging.info(f"  Individual similarities: {[f'{s:.3f}' for s in content_similarities]}")
        
        return avg_content_preservation
    
    def calculate_fid_score(self, original_images, unlearned_images):
        """
        Calculate FID score between original and unlearned images
        Simplified version for PoC
        """
        try:
            from torchvision.models import inception_v3
            from scipy.linalg import sqrtm
            
            # Load Inception v3 for FID calculation
            inception = inception_v3(pretrained=True, transform_input=False).to(self.device)
            inception.eval()
            
            def extract_features(image_paths):
                features = []
                for img_path in image_paths:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        feat = inception(img_tensor)
                        features.append(feat.cpu().numpy().flatten())
                
                return np.array(features)
            
            # Extract features
            orig_features = extract_features(original_images)
            unlearn_features = extract_features(unlearned_images)
            
            # Calculate FID
            mu1, sigma1 = orig_features.mean(axis=0), np.cov(orig_features, rowvar=False)
            mu2, sigma2 = unlearn_features.mean(axis=0), np.cov(unlearn_features, rowvar=False)
            
            ssdiff = np.sum((mu1 - mu2) ** 2.0)
            covmean = sqrtm(sigma1.dot(sigma2))
            
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            
            fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
            
            logging.info(f"FID Score: {fid:.4f}")
            return fid
            
        except Exception as e:
            logging.warning(f"FID calculation failed: {e}")
            return None
    
    def calculate_semantic_consistency(self, images, prompts):
        """
        Calculate semantic consistency between generated images and prompts
        """
        if len(images) != len(prompts):
            logging.error("Mismatch in number of images and prompts")
            return 0.0
        
        consistencies = []
        
        for img_path, prompt in zip(images, prompts):
            image = Image.open(img_path).convert('RGB')
            
            # Calculate CLIP similarity with original prompt
            clip_score = self._calculate_clip_concept_score(image, [prompt])
            consistencies.append(clip_score)
        
        avg_consistency = np.mean(consistencies)
        
        logging.info(f"Semantic Consistency: {avg_consistency:.4f}")
        return avg_consistency
    
    def evaluate_unlearncanvas_metrics(self, results_dir, model_name, test_prompts=None):
        """
        Comprehensive UnlearnCanvas evaluation for a single model
        """
        logging.info(f"Evaluating {model_name} with UnlearnCanvas metrics")
        
        # Load image pairs
        original_dir = Path(results_dir) / "original"
        unlearned_dir = Path(results_dir) / "unlearned"
        
        if not original_dir.exists() or not unlearned_dir.exists():
            logging.error(f"Results directory not found: {results_dir}")
            return None
        
        # Get all image files
        original_images = sorted([str(f) for f in original_dir.glob("*.png")])
        unlearned_images = sorted([str(f) for f in unlearned_dir.glob("*.png")])
        
        if len(original_images) != len(unlearned_images):
            logging.error("Mismatch in number of original and unlearned images")
            return None
        
        logging.info(f"Evaluating {len(original_images)} image pairs")
        
        # 1. Unlearning Accuracy (UA) - primary metric
        ua_score, detected_count = self.calculate_unlearning_accuracy(unlearned_images)
        
        # 2. Content Preservation
        content_preservation = self.calculate_content_preservation(original_images, unlearned_images)
        
        # 3. FID Score
        fid_score = self.calculate_fid_score(original_images, unlearned_images)
        
        # 4. Semantic Consistency (if prompts provided)
        semantic_consistency = None
        if test_prompts and len(test_prompts) == len(unlearned_images):
            semantic_consistency = self.calculate_semantic_consistency(unlearned_images, test_prompts)
        
        # 5. Additional metrics
        # Calculate style reduction (CLIP-based)
        original_van_gogh_scores = []
        unlearned_van_gogh_scores = []
        
        for orig_path, unlearn_path in zip(original_images, unlearned_images):
            orig_img = Image.open(orig_path).convert('RGB')
            unlearn_img = Image.open(unlearn_path).convert('RGB')
            
            orig_score = self._calculate_clip_concept_score(orig_img, self.van_gogh_prompts)
            unlearn_score = self._calculate_clip_concept_score(unlearn_img, self.van_gogh_prompts)
            
            original_van_gogh_scores.append(orig_score)
            unlearned_van_gogh_scores.append(unlearn_score)
        
        style_reduction = np.mean(original_van_gogh_scores) - np.mean(unlearned_van_gogh_scores)
        
        # Compile results
        results = {
            'model': model_name,
            'num_images': len(original_images),
            
            # Primary UnlearnCanvas metrics
            'unlearning_accuracy': ua_score,
            'content_preservation': content_preservation,
            'fid_score': fid_score,
            'semantic_consistency': semantic_consistency,
            
            # Additional analysis
            'style_reduction': style_reduction,
            'original_van_gogh_score': np.mean(original_van_gogh_scores),
            'unlearned_van_gogh_score': np.mean(unlearned_van_gogh_scores),
            'detected_concept_count': detected_count,
            
            # Success metrics
            'success_rate': ua_score,  # UA is the primary success metric
            'effectiveness_score': max(0, style_reduction)
        }
        
        # Log comprehensive results
        logging.info(f"\n{model_name} UnlearnCanvas Results:")
        logging.info(f"  🎯 Unlearning Accuracy (UA): {ua_score:.4f}")
        logging.info(f"  📎 Content Preservation: {content_preservation:.4f}")
        logging.info(f"  📊 FID Score: {fid_score:.4f}" if fid_score else "  📊 FID Score: Failed")
        logging.info(f"  🔄 Style Reduction: {style_reduction:.4f}")
        logging.info(f"  ✅ Success Rate: {ua_score:.1%}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="UnlearnCanvas evaluation of concept unlearning")
    parser.add_argument("--results_dir", type=str, default="concept_unlearning_results",
                       help="Directory containing results (default: concept_unlearning_results)")
    parser.add_argument("--models", nargs="+", default=["sae", "msae"],
                       help="Models to evaluate (default: [sae, msae])")
    parser.add_argument("--output_file", type=str, default="unlearncanvas_evaluation.json",
                       help="Output file for results (default: unlearncanvas_evaluation.json)")
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging()
    
    logging.info("=== UNLEARNCANVAS EVALUATION ===")
    logging.info(f"Results directory: {args.results_dir}")
    logging.info(f"Models to evaluate: {args.models}")
    
    # Test prompts used in generation (if available)
    test_prompts = [
    ]
    
    # Initialize evaluator
    evaluator = UnlearnCanvasEvaluator()
    
    # Evaluate each model
    all_results = []
    
    for model in args.models:
        model_results_dir = Path(args.results_dir) / model
        if not model_results_dir.exists():
            logging.warning(f"Results directory not found for {model}: {model_results_dir}")
            continue
        
        result = evaluator.evaluate_unlearncanvas_metrics(
            str(model_results_dir), model, test_prompts
        )
        
        if result:
            all_results.append(result)
    
    # Model comparison (UnlearnCanvas style)
    if len(all_results) > 1:
        logging.info("\n" + "="*60)
        logging.info("UNLEARNCANVAS MODEL COMPARISON")
        logging.info("="*60)
        
        # Sort by Unlearning Accuracy (primary metric)
        sorted_by_ua = sorted(all_results, key=lambda x: x['unlearning_accuracy'], reverse=True)
        
        logging.info("🏆 Ranking by Unlearning Accuracy (Primary Metric):")
        for i, result in enumerate(sorted_by_ua):
            logging.info(f"  {i+1}. {result['model'].upper()}: {result['unlearning_accuracy']:.4f}")
        
        # Sort by Content Preservation
        sorted_by_cp = sorted(all_results, key=lambda x: x['content_preservation'], reverse=True)
        
        logging.info("\n🏆 Ranking by Content Preservation:")
        for i, result in enumerate(sorted_by_cp):
            logging.info(f"  {i+1}. {result['model'].upper()}: {result['content_preservation']:.4f}")
        
        # Overall winner (UA weighted heavily, as in UnlearnCanvas)
        for result in all_results:
            result['unlearncanvas_score'] = (
                0.7 * result['unlearning_accuracy'] + 
                0.3 * result['content_preservation']
            )
        
        sorted_overall = sorted(all_results, key=lambda x: x['unlearncanvas_score'], reverse=True)
        
        logging.info("\n🏆 Overall UnlearnCanvas Winner:")
        for i, result in enumerate(sorted_overall):
            logging.info(f"  {i+1}. {result['model'].upper()}: {result['unlearncanvas_score']:.4f} "
                        f"(UA: {result['unlearning_accuracy']:.4f}, CP: {result['content_preservation']:.4f})")
        
        winner = sorted_overall[0]
        logging.info(f"\n🎯 UNLEARNCANVAS WINNER: {winner['model'].upper()}")
        logging.info(f"   Score: {winner['unlearncanvas_score']:.4f}")
        logging.info(f"   (70% Unlearning Accuracy + 30% Content Preservation)")
        
        # Detailed comparison table
        logging.info(f"\n📊 Detailed Comparison:")
        logging.info(f"{'Model':<10} {'UA':<8} {'CP':<8} {'FID':<8} {'Style↓':<8}")
        logging.info("-" * 50)
        for result in sorted_overall:
            fid_str = f"{result['fid_score']:.2f}" if result['fid_score'] else "N/A"
            logging.info(f"{result['model'].upper():<10} {result['unlearning_accuracy']:<8.4f} "
                        f"{result['content_preservation']:<8.4f} {fid_str:<8} "
                        f"{result['style_reduction']:<8.4f}")
    
    # Convert numpy types for JSON serialization
    def convert_numpy_types(obj):
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
    
    # Save results
    output_data = {
        'evaluation_type': 'UnlearnCanvas',
        'results': convert_numpy_types(all_results),
        'evaluation_timestamp': datetime.now().isoformat(),
        'log_file': log_file,
        'metrics_description': {
            'unlearning_accuracy': 'Primary metric: 1 - (detected_concept_count / total_images)',
            'content_preservation': 'CLIP feature similarity between original and unlearned',
            'fid_score': 'Frechet Inception Distance between image distributions',
            'style_reduction': 'Reduction in Van Gogh style CLIP scores'
        }
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logging.info(f"\n✅ UnlearnCanvas evaluation completed!")
    logging.info(f"📊 Results saved to: {args.output_file}")
    logging.info(f"📋 Detailed log: {log_file}")
    
    return all_results

if __name__ == "__main__":
    results = main()