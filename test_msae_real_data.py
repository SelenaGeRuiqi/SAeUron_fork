"""
Test our DiffusionMSAE on real collected activation data
"""
import torch
import pickle
from models.diffusion_msae import DiffusionMSAE

def test_on_real_data():
    print("Testing DiffusionMSAE on real collected activations...")
    
    # Load our collected activations
    with open('activations_poc/activations_unet_up_blocks_1_attentions_2.pkl', 'rb') as f:
        activations = pickle.load(f)
    
    print(f"Loaded {len(activations)} activation samples")
    print(f"Sample activation shape: {activations[0].shape}")
    
    # Convert first few samples to a batch - handle the extra dimension
    batch_size = 4  # Small batch for testing

    # Each activation has shape [2, 1280, 16, 16], so we need to flatten the first dimension
    flattened_activations = []
    for activation in activations[:batch_size//2]:  # Take 2 activations to get 4 samples
        # Each activation contains 2 samples, so separate them
        for i in range(activation.shape[0]):
            flattened_activations.append(activation[i])

    batch_activations = torch.stack(flattened_activations)
    print(f"Test batch shape: {batch_activations.shape}")
    
    # Create and test MSAE
    msae = DiffusionMSAE(
        input_dim=1280, 
        expansion_factor=4,
        k_values=[16, 32]  # 2-level hierarchy for PoC
    )
    
    # Move to GPU if available and convert to float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    msae = msae.to(device)
    batch_activations = batch_activations.to(device).float()  # Convert to float32

    print(f"Running on device: {device}")
    print(f"Batch dtype: {batch_activations.dtype}")
    
    # Forward pass
    with torch.no_grad():
        features, reconstructions = msae(batch_activations)
        loss = msae.compute_loss(batch_activations, reconstructions)
    
    print("\n=== Results ===")
    print(f"Multi-granularity loss: {loss.item():.4f}")
    
    # Analyze sparsity at each level
    for i, feat in enumerate(features):
        active_features = (feat > 0).float().mean()
        print(f"Level {i} (k={msae.k_values[i]}): {active_features.item():.1%} features active")
    
    # Analyze reconstruction quality
    for i, recon in enumerate(reconstructions):
        mse = torch.nn.functional.mse_loss(recon, batch_activations)
        cosine_sim = torch.nn.functional.cosine_similarity(
            recon.flatten(1), batch_activations.flatten(1)
        ).mean()
        print(f"Level {i} reconstruction - MSE: {mse.item():.4f}, Cosine Sim: {cosine_sim.item():.4f}")
    
    print("\n✅ Real data test completed successfully!")
    print(f"MSAE can process {len(activations)} samples of shape {activations[0].shape}")
    
    return msae, activations

if __name__ == "__main__":
    msae, activations = test_on_real_data()
