"""
Analyze concept separation capabilities of multiple SAE baselines vs MSAE
Test which model better isolates Van Gogh style features
Enhanced with logging and multiple baseline comparison
"""
import torch
import pickle
import numpy as np
import logging
import os
from datetime import datetime
import argparse
from models.traditional_sae import TraditionalSAE
from models.diffusion_msae import DiffusionMSAE

def setup_logging(log_dir="log/analyze_concept_features_log"):
    """Setup logging to both file and console"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for unique log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"concept_analysis_{timestamp}.log")
    
    # Setup logging format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    logging.info(f"Concept analysis logging initialized. Log file: {log_file}")
    return log_file

def load_model_checkpoint(model_path):
    """Load model checkpoint and return config"""
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        logging.info(f"Loaded checkpoint: {model_path}")
        logging.info(f"  Final loss: {checkpoint.get('final_loss', 'N/A')}")
        logging.info(f"  Parameters: {checkpoint.get('total_params', 'N/A'):,}")
        return checkpoint
    else:
        logging.warning(f"Checkpoint not found: {model_path}")
        return None

def load_sae_models(k_values=[16, 32, 64], expansion_factor=16):
    """Load multiple SAE models with different k values"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae_models = {}
    
    logging.info(f"Loading Traditional SAE models with k values: {k_values}")
    
    for k in k_values:
        checkpoint_path = f"sae-ckpts/traditional_sae/checkpoint_exp{expansion_factor}_k{k}.pth"
        checkpoint = load_model_checkpoint(checkpoint_path)
        
        if checkpoint:
            sae = TraditionalSAE(
                input_dim=1280,
                expansion_factor=expansion_factor,
                k=k
            )
            sae.load_state_dict(checkpoint['model_state_dict'])
            sae = sae.to(device).eval()
            sae_models[f"SAE_k{k}"] = sae
            logging.info(f"✅ SAE k={k} loaded successfully")
        else:
            logging.error(f"❌ Failed to load SAE k={k}")
    
    return sae_models

def load_msae_model(k_values=[16, 32, 64], expansion_factor=16):
    """Load MSAE model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    k_str = "_".join(map(str, k_values))
    checkpoint_path = f"sae-ckpts/msae/checkpoint_exp{expansion_factor}_k{k_str}.pth"
    
    logging.info(f"Loading MSAE model with k values: {k_values}")
    checkpoint = load_model_checkpoint(checkpoint_path)
    
    if checkpoint:
        msae = DiffusionMSAE(
            input_dim=1280,
            expansion_factor=expansion_factor,
            k_values=k_values
        )
        msae.load_state_dict(checkpoint['model_state_dict'])
        msae = msae.to(device).eval()
        logging.info(f"✅ MSAE loaded successfully")
        return msae
    else:
        logging.error(f"❌ Failed to load MSAE")
        return None

def prepare_concept_data(num_samples=100):
    """Prepare Van Gogh vs Realistic activation data"""
    logging.info("Preparing concept separation data...")
    
    # Load activations
    with open('activations_poc/activations_unet_up_blocks_1_attentions_2.pkl', 'rb') as f:
        activations = pickle.load(f)
    
    van_gogh_activations = []
    realistic_activations = []
    
    # Split by prompt type: first 250 activations are Van Gogh, next 250 are realistic
    for i, activation in enumerate(activations):
        if i < 250:  # Van Gogh prompts (each contains 2 samples)
            for j in range(activation.shape[0]):
                van_gogh_activations.append(activation[j])
        else:  # Realistic prompts
            for j in range(activation.shape[0]):
                realistic_activations.append(activation[j])
    
    # Sample for analysis
    van_gogh_batch = torch.stack(van_gogh_activations[:num_samples]).cuda().float()
    realistic_batch = torch.stack(realistic_activations[:num_samples]).cuda().float()
    
    logging.info(f"Concept data prepared:")
    logging.info(f"  Van Gogh samples: {van_gogh_batch.shape}")
    logging.info(f"  Realistic samples: {realistic_batch.shape}")
    
    return van_gogh_batch, realistic_batch

def analyze_feature_separation(van_gogh_features, realistic_features, model_name, k_value=None):
    """Analyze feature separation capability for a single feature set"""
    
    # Calculate mean activation for each feature
    van_gogh_mean = van_gogh_features.mean(dim=0)
    realistic_mean = realistic_features.mean(dim=0)
    
    # Calculate feature importance (absolute difference between groups)
    feature_importance = torch.abs(van_gogh_mean - realistic_mean)
    
    # Get top discriminative features
    top_features = torch.topk(feature_importance, k=32)
    
    # Calculate separation metrics
    feature_correlation = torch.corrcoef(torch.stack([van_gogh_mean, realistic_mean]))[0, 1]
    mean_separation = feature_importance.mean()
    max_separation = feature_importance.max()
    std_separation = feature_importance.std()
    
    # Calculate additional metrics
    # Jensen-Shannon divergence approximation
    van_gogh_prob = torch.softmax(van_gogh_mean, dim=0)
    realistic_prob = torch.softmax(realistic_mean, dim=0)
    js_divergence = 0.5 * (torch.sum(van_gogh_prob * torch.log(van_gogh_prob / ((van_gogh_prob + realistic_prob) / 2))) +
                           torch.sum(realistic_prob * torch.log(realistic_prob / ((van_gogh_prob + realistic_prob) / 2))))
    
    # Sparsity analysis
    van_gogh_sparsity = (van_gogh_features > 0).float().mean()
    realistic_sparsity = (realistic_features > 0).float().mean()
    sparsity_diff = torch.abs(van_gogh_sparsity - realistic_sparsity)
    
    model_display = f"{model_name}" + (f" (k={k_value})" if k_value else "")
    
    logging.info(f"\n{model_display} Analysis:")
    logging.info(f"  Feature correlation: {feature_correlation.item():.4f} (lower = better separation)")
    logging.info(f"  Mean feature separation: {mean_separation.item():.4f}")
    logging.info(f"  Max feature separation: {max_separation.item():.4f}")
    logging.info(f"  Std feature separation: {std_separation.item():.4f}")
    logging.info(f"  JS divergence: {js_divergence.item():.4f} (higher = better separation)")
    logging.info(f"  Van Gogh sparsity: {van_gogh_sparsity.item():.1%}")
    logging.info(f"  Realistic sparsity: {realistic_sparsity.item():.1%}")
    logging.info(f"  Sparsity difference: {sparsity_diff.item():.1%}")
    logging.info(f"  Top 32 features importance: {top_features.values[:32].mean().item():.4f}")
    
    return {
        'model_name': model_display,
        'feature_correlation': feature_correlation.item(),
        'mean_separation': mean_separation.item(),
        'max_separation': max_separation.item(),
        'std_separation': std_separation.item(),
        'js_divergence': js_divergence.item(),
        'van_gogh_sparsity': van_gogh_sparsity.item(),
        'realistic_sparsity': realistic_sparsity.item(),
        'sparsity_diff': sparsity_diff.item(),
        'top_features_importance': top_features.values[:32].mean().item(),
        'feature_importance': feature_importance,
        'top_features': top_features
    }

def comprehensive_concept_analysis(sae_k_values=[16, 32, 64], msae_k_values=[16, 32, 64], 
                                 expansion_factor=16, num_samples=100):
    """Comprehensive analysis comparing multiple SAE baselines with MSAE"""
    
    # Load models
    sae_models = load_sae_models(sae_k_values, expansion_factor)
    msae_model = load_msae_model(msae_k_values, expansion_factor)
    
    if not sae_models:
        logging.error("No SAE models loaded successfully!")
        return None
    
    if not msae_model:
        logging.error("MSAE model failed to load!")
        return None
    
    # Prepare data
    van_gogh_batch, realistic_batch = prepare_concept_data(num_samples)
    
    # Store all results
    all_results = []
    
    logging.info("\n" + "="*60)
    logging.info("COMPREHENSIVE CONCEPT SEPARATION ANALYSIS")
    logging.info("="*60)
    
    # Analyze all SAE models
    with torch.no_grad():
        for model_name, sae_model in sae_models.items():
            k_value = int(model_name.split('k')[1])
            
            # Get SAE features
            van_gogh_sae_features, _ = sae_model(van_gogh_batch)
            realistic_sae_features, _ = sae_model(realistic_batch)
            
            # Analyze separation
            sae_results = analyze_feature_separation(
                van_gogh_sae_features, realistic_sae_features, 
                "Traditional SAE", k_value
            )
            all_results.append(sae_results)
        
        # Analyze MSAE at all granularity levels
        van_gogh_msae_features, _ = msae_model(van_gogh_batch)
        realistic_msae_features, _ = msae_model(realistic_batch)
        
        for i, k_val in enumerate(msae_k_values):
            msae_results = analyze_feature_separation(
                van_gogh_msae_features[i], realistic_msae_features[i],
                f"MSAE Level {i}", k_val
            )
            all_results.append(msae_results)
    
    # Comprehensive comparison
    logging.info("\n" + "="*60)
    logging.info("COMPREHENSIVE COMPARISON SUMMARY")
    logging.info("="*60)
    
    # Sort by different metrics
    metrics_to_compare = [
        ('mean_separation', 'Mean Feature Separation', 'higher'),
        ('max_separation', 'Max Feature Separation', 'higher'),
        ('js_divergence', 'JS Divergence', 'higher'),
        ('feature_correlation', 'Feature Correlation', 'lower'),
        ('top_features_importance', 'Top Features Importance', 'higher')
    ]
    
    for metric, display_name, better in metrics_to_compare:
        logging.info(f"\n🏆 {display_name} Rankings ({'higher is better' if better == 'higher' else 'lower is better'}):")
        
        sorted_results = sorted(all_results, 
                              key=lambda x: x[metric], 
                              reverse=(better == 'higher'))
        
        for i, result in enumerate(sorted_results):
            logging.info(f"  {i+1}. {result['model_name']}: {result[metric]:.4f}")
    
    # Overall winner analysis
    logging.info("\n" + "="*60)
    logging.info("OVERALL WINNER ANALYSIS")
    logging.info("="*60)
    
    # Count wins for each model across metrics
    model_wins = {}
    for metric, _, better in metrics_to_compare:
        sorted_results = sorted(all_results, 
                              key=lambda x: x[metric], 
                              reverse=(better == 'higher'))
        winner = sorted_results[0]['model_name']
        model_wins[winner] = model_wins.get(winner, 0) + 1
    
    logging.info("Wins per model across all metrics:")
    for model, wins in sorted(model_wins.items(), key=lambda x: x[1], reverse=True):
        logging.info(f"  {model}: {wins}/5 metrics")
    
    overall_winner = max(model_wins.items(), key=lambda x: x[1])
    logging.info(f"\n🏆 Overall Winner: {overall_winner[0]} ({overall_winner[1]}/5 metrics)")
    
    # Best MSAE vs Best SAE comparison
    msae_results = [r for r in all_results if 'MSAE' in r['model_name']]
    sae_results = [r for r in all_results if 'SAE' in r['model_name'] and 'MSAE' not in r['model_name']]
    
    if msae_results and sae_results:
        best_msae = max(msae_results, key=lambda x: x['mean_separation'])
        best_sae = max(sae_results, key=lambda x: x['mean_separation'])
        
        logging.info(f"\n🔥 Best MSAE: {best_msae['model_name']} (separation: {best_msae['mean_separation']:.4f})")
        logging.info(f"🔥 Best SAE: {best_sae['model_name']} (separation: {best_sae['mean_separation']:.4f})")
        
        if best_msae['mean_separation'] > best_sae['mean_separation']:
            logging.info("🎯 MSAE's hierarchical approach wins overall!")
        else:
            logging.info("🎯 Traditional SAE approach wins overall!")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Comprehensive concept separation analysis")
    parser.add_argument("--sae_k_values", type=int, nargs="+", default=[16, 32, 64],
                       help="K values for SAE models to analyze (default: [16, 32, 64])")
    parser.add_argument("--msae_k_values", type=int, nargs="+", default=[16, 32, 64],
                       help="K values for MSAE hierarchy (default: [16, 32, 64])")
    parser.add_argument("--expansion_factor", type=int, default=16,
                       help="Expansion factor used in training (default: 16)")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples per concept for analysis (default: 100)")
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging()
    
    logging.info("=== COMPREHENSIVE CONCEPT SEPARATION ANALYSIS ===")
    logging.info(f"SAE k values to analyze: {args.sae_k_values}")
    logging.info(f"MSAE k values: {args.msae_k_values}")
    logging.info(f"Expansion factor: {args.expansion_factor}")
    logging.info(f"Samples per concept: {args.num_samples}")
    
    # Run comprehensive analysis
    results = comprehensive_concept_analysis(
        sae_k_values=args.sae_k_values,
        msae_k_values=args.msae_k_values,
        expansion_factor=args.expansion_factor,
        num_samples=args.num_samples
    )
    
    if results:
        logging.info(f"\n✅ Comprehensive analysis completed!")
        logging.info(f"📋 Detailed results logged to: {log_file}")
        logging.info(f"📊 Total models analyzed: {len(results)}")
    else:
        logging.error("❌ Analysis failed!")
    
    return results

if __name__ == "__main__":
    results = main()
