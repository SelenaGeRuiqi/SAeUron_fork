#!/usr/bin/env python3
"""
Minimal test for MSAE loading without SAeUron dependencies
"""

import torch
import torch.nn as nn
from typing import NamedTuple
import sys
from pathlib import Path

class SimpleEncoderOutput(NamedTuple):
    top_acts: torch.Tensor
    top_indices: torch.Tensor

class SimpleForwardOutput(NamedTuple):
    sae_out: torch.Tensor
    latent_acts: torch.Tensor

class SimpleMSAEModel(nn.Module):
    """
    Simplified MSAE model that loads directly from state dictionary
    """
    def __init__(self, state_dict, device):
        super().__init__()
        
        # Extract dimensions from state dict
        encoder_weight = state_dict['encoder']
        self.n_inputs = encoder_weight.shape[0]
        self.n_latents = encoder_weight.shape[1]
        
        print(f"Creating model with {self.n_inputs} inputs, {self.n_latents} latents")
        
        # Create parameters from state dict
        self.encoder = nn.Parameter(encoder_weight.clone().to(device))
        self.decoder = nn.Parameter(state_dict['decoder'].clone().to(device))
        self.pre_bias = nn.Parameter(state_dict['pre_bias'].clone().to(device))
        self.latent_bias = nn.Parameter(state_dict['latent_bias'].clone().to(device))
    
    def encode(self, x):
        """Encode input to latent space"""
        # Remove pre-bias and apply encoder
        x_centered = x - self.pre_bias
        latents = torch.matmul(x_centered, self.encoder) + self.latent_bias
        
        # Apply ReLU activation
        latents = torch.relu(latents)
        return latents
    
    def decode(self, latents):
        """Decode latents back to input space"""
        return torch.matmul(latents, self.decoder) + self.pre_bias
    
    def forward(self, x):
        """Full forward pass"""
        latents = self.encode(x)
        reconstruction = self.decode(latents)
        return reconstruction, latents

def test_checkpoint_loading():
    """Test basic checkpoint loading"""
    print("=== Testing Checkpoint Loading ===")
    
    msae_path = "/data/selena/MSAE/2048_512_TopKReLU_64_UW_False_False_0.0_imagenet_ViT-B~32_train_image_1281167_512.pth"
    
    if not Path(msae_path).exists():
        print(f"❌ Checkpoint not found: {msae_path}")
        return None
    
    print(f"Loading checkpoint: {msae_path}")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        checkpoint = torch.load(msae_path, map_location=device)
        print(f"✅ Checkpoint loaded successfully")
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Check model state dict
        model_state_dict = checkpoint['model']
        print(f"Model state dict keys: {list(model_state_dict.keys())}")
        
        # Check shapes
        encoder_shape = model_state_dict['encoder'].shape
        decoder_shape = model_state_dict['decoder'].shape
        print(f"Encoder shape: {encoder_shape}")
        print(f"Decoder shape: {decoder_shape}")
        
        return checkpoint, device
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_model_creation(checkpoint, device):
    """Test creating the MSAE model"""
    print("\n=== Testing Model Creation ===")
    
    try:
        model_state_dict = checkpoint['model']
        model = SimpleMSAEModel(model_state_dict, device)
        print(f"✅ Model created successfully")
        print(f"Model inputs: {model.n_inputs}")
        print(f"Model latents: {model.n_latents}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_model_forward(model, checkpoint, device):
    """Test forward pass through the model"""
    print("\n=== Testing Model Forward Pass ===")
    
    try:
        # Create dummy input
        batch_size = 2
        input_dim = model.n_inputs
        x = torch.randn(batch_size, input_dim, device=device)
        print(f"Input shape: {x.shape}")
        
        # Apply preprocessing
        mean_center = checkpoint['mean_center'].to(device)
        scaling_factor = checkpoint['scaling_factor']
        
        x_centered = x - mean_center
        x_scaled = x_centered * scaling_factor
        print(f"Preprocessed input shape: {x_scaled.shape}")
        
        # Forward pass
        with torch.no_grad():
            latents = model.encode(x_scaled)
            reconstruction = model.decode(latents)
            
        print(f"✅ Forward pass successful!")
        print(f"Latents shape: {latents.shape}")
        print(f"Reconstruction shape: {reconstruction.shape}")
        
        # Check reconstruction quality
        recon_error = torch.mean((x_scaled - reconstruction) ** 2)
        print(f"Reconstruction MSE: {recon_error.item():.6f}")
        
        # Check sparsity
        active_latents = torch.sum(latents > 0).item()
        total_latents = latents.numel()
        sparsity = 1.0 - (active_latents / total_latents)
        print(f"Sparsity: {sparsity:.3f} ({active_latents}/{total_latents} active)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_topk_selection(model, checkpoint, device):
    """Test TopK feature selection"""
    print("\n=== Testing TopK Feature Selection ===")
    
    try:
        # Create dummy input
        batch_size = 3
        input_dim = model.n_inputs
        x = torch.randn(batch_size, input_dim, device=device)
        
        # Apply preprocessing
        mean_center = checkpoint['mean_center'].to(device)
        scaling_factor = checkpoint['scaling_factor']
        x_prep = (x - mean_center) * scaling_factor
        
        # Get latents
        with torch.no_grad():
            latents = model.encode(x_prep)
            
        print(f"Latents shape: {latents.shape}")
        
        # Test different k values
        k_values = [16, 32, 64]
        
        for k in k_values:
            top_values, top_indices = torch.topk(latents, k=min(k, latents.shape[1]), dim=1)
            positive_mask = top_values > 0
            
            avg_positive = torch.sum(positive_mask).item() / batch_size
            print(f"k={k}: avg {avg_positive:.1f} positive activations per sample")
        
        print("✅ TopK selection test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error in TopK selection: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Minimal MSAE Test")
    print("=" * 40)
    
    # Test 1: Checkpoint loading
    result = test_checkpoint_loading()
    if result is None:
        print("\n❌ Checkpoint loading failed. Stopping tests.")
        return
    
    checkpoint, device = result
    
    # Test 2: Model creation
    model = test_model_creation(checkpoint, device)
    if model is None:
        print("\n❌ Model creation failed. Stopping tests.")
        return
    
    # Test 3: Forward pass
    if not test_model_forward(model, checkpoint, device):
        print("\n⚠️ Forward pass failed, but continuing...")
    
    # Test 4: TopK selection
    if not test_topk_selection(model, checkpoint, device):
        print("\n⚠️ TopK selection failed...")
    
    print("\n" + "=" * 40)
    print("🎉 Minimal testing completed!")
    print("\nIf all tests pass, the MSAE loading works correctly.")
    print("Next: Try the full SAeUron integration test.")

if __name__ == "__main__":
    main()