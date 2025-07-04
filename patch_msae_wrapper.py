import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msae_wrapper import Sae

def create_msae_with_640d_projection():
    """Create MSAE wrapper with proper 640D projection layers"""
    
    # Load the base MSAE
    msae_path = "/data/selena/MSAE/2048_512_TopKReLU_64_UW_False_False_0.0_imagenet_ViT-B~32_train_image_1281167_512.pth"
    msae = Sae.load_from_disk(msae_path, device='cuda')
    
    # Add proper projection layers for 640D -> 512D -> 640D
    msae.input_projection = nn.Linear(640, 512, dtype=torch.float16, device='cuda')
    msae.output_projection = nn.Linear(512, 640, dtype=torch.float16, device='cuda')
    
    # Initialize projections with reasonable weights
    with torch.no_grad():
        # Xavier initialization
        nn.init.xavier_uniform_(msae.input_projection.weight)
        nn.init.zeros_(msae.input_projection.bias)
        nn.init.xavier_uniform_(msae.output_projection.weight)
        nn.init.zeros_(msae.output_projection.bias)
    
    print("✅ Added 640D projection layers to MSAE")
    return msae

def apply_msae_intervention_640d(activations, msae, multiplier=1.0):
    """Apply MSAE intervention to 640D activations"""
    
    original_shape = activations.shape
    batch_size = original_shape[0]
    
    # Flatten spatial dimensions: [B, C, H, W] -> [B*H*W, C]
    spatial_tokens = activations.permute(0, 2, 3, 1).reshape(-1, original_shape[1])
    
    with torch.no_grad():
        # Project 640D -> 512D
        projected_input = msae.input_projection(spatial_tokens)
        
        # Apply MSAE processing
        encoded = msae.encode(projected_input)
        
        if encoded.top_indices.numel() > 0:
            # Select features for intervention
            feature_indices = encoded.top_indices[0][:10]  # Top 10 features
            
            # Apply intervention
            manipulated = msae.manipulate_features(
                projected_input,
                feature_indices=feature_indices,
                multiplier=multiplier
            )
            
            # Project back 512D -> 640D
            projected_output = msae.output_projection(manipulated)
            
            # Reshape back to original format
            modified_activations = projected_output.reshape(
                original_shape[0], original_shape[2], original_shape[3], original_shape[1]
            ).permute(0, 3, 1, 2)
            
            return modified_activations, True  # Return success flag
    
    return activations, False  # Return original if failed

# Test the patched version
if __name__ == "__main__":
    print("Testing patched MSAE with 640D projection...")
    
    msae = create_msae_with_640d_projection()
    
    # Test with 640D input
    test_input = torch.randn(2, 640, 32, 32, device='cuda', dtype=torch.float16)
    print(f"Test input shape: {test_input.shape}")
    
    # Test intervention
    modified, success = apply_msae_intervention_640d(test_input, msae, multiplier=2.0)
    print(f"Intervention {'✅ SUCCESS' if success else '❌ FAILED'}")
    print(f"Output shape: {modified.shape}")
    print("🎯 Ready for concept control testing!")
