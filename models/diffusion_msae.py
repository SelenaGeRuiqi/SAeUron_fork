"""
Diffusion-adapted MSAE for concept unlearning PoC
Adapts MSAE architecture for U-Net attention activations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import math

class DiffusionMSAE(nn.Module):
    """
    Matryoshka Sparse Autoencoder adapted for diffusion model activations
    """
    def __init__(
        self,
        input_dim: int = 1280,  # U-Net attention dimension
        expansion_factor: int = 4,  # Simplified for PoC
        spatial_dims: Tuple[int, int] = (16, 16),  # Spatial dimensions
        k_values: List[int] = None,  # TopK values for different granularities
        device: str = "cuda"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.spatial_dims = spatial_dims
        self.hidden_dim = input_dim * expansion_factor
        self.device = device
        
        # Default k values for 2-level hierarchy (simplified for PoC)
        if k_values is None:
            self.k_values = [16, 32]  # Coarse to fine granularity
        else:
            self.k_values = k_values
        
        # Encoder and Decoder
        self.encoder = nn.Linear(input_dim, self.hidden_dim, bias=True)
        self.decoder = nn.Linear(self.hidden_dim, input_dim, bias=True)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights similar to original MSAE"""
        # Encoder: Xavier uniform
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        
        # Decoder: Transpose of encoder + normalization
        self.decoder.weight.data = self.encoder.weight.data.T
        nn.init.zeros_(self.decoder.bias)
        
        # Normalize decoder columns
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
    
    def topk_activation(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """Apply TopK activation with k active features"""
        # Get top k values and indices
        topk_values, topk_indices = torch.topk(x, k, dim=-1)
        
        # Create sparse tensor
        result = torch.zeros_like(x)
        result.scatter_(-1, topk_indices, topk_values)
        
        return result
    
    def encode_multi_granularity(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Encode input at multiple granularity levels
        Returns list of encoded features for each k value
        """
        # Flatten spatial dimensions: [batch, channels, h, w] -> [batch*h*w, channels]
        if len(x.shape) == 5:
            # Reshape [B, B2, C, H, W] to [B*B2, C, H, W]
            batch_size, batch2, channels, h, w = x.shape
            x = x.view(batch_size * batch2, channels, h, w)
        else:
            batch_size, channels, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(-1, channels)
        
        # Encode
        encoded = F.relu(self.encoder(x_flat))
        
        # Apply different TopK levels
        multi_granularity_features = []
        for k in self.k_values:
            sparse_features = self.topk_activation(encoded, k)
            multi_granularity_features.append(sparse_features)
        
        return multi_granularity_features
    
    def decode(self, encoded_features: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
        """Decode features back to original spatial format"""
        batch_size, channels, h, w = original_shape
        
        # Decode
        decoded_flat = self.decoder(encoded_features)
        
        # Reshape back to spatial format
        decoded = decoded_flat.view(batch_size, h, w, channels).permute(0, 3, 1, 2)
        
        return decoded
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass with multi-granularity reconstruction
        Returns: (multi_granularity_features, multi_granularity_reconstructions)
        """
        original_shape = x.shape
        
        # Encode at multiple granularities
        multi_features = self.encode_multi_granularity(x)
        
        # Decode each granularity level
        multi_reconstructions = []
        for features in multi_features:
            reconstruction = self.decode(features, original_shape)
            multi_reconstructions.append(reconstruction)
        
        return multi_features, multi_reconstructions
    
    def compute_loss(
        self, 
        x: torch.Tensor, 
        multi_reconstructions: List[torch.Tensor],
        weights: List[float] = None
    ) -> torch.Tensor:
        """
        Compute multi-granularity reconstruction loss
        """
        if weights is None:
            # Reverse weighting: emphasize sparser representations
            weights = [len(self.k_values) - i for i in range(len(self.k_values))]
            weights = [w / sum(weights) for w in weights]  # Normalize
        
        total_loss = 0.0
        for i, reconstruction in enumerate(multi_reconstructions):
            mse_loss = F.mse_loss(reconstruction, x)
            total_loss += weights[i] * mse_loss
        
        return total_loss

# Test function
def test_diffusion_msae():
    """Test our adapted MSAE"""
    print("Testing DiffusionMSAE...")
    
    # Create sample data matching our collected activations
    batch_size, channels, h, w = 2, 1280, 16, 16
    sample_data = torch.randn(batch_size, channels, h, w)
    
    # Create MSAE
    msae = DiffusionMSAE(input_dim=channels, expansion_factor=4)
    
    # Forward pass
    features, reconstructions = msae(sample_data)
    
    print(f"Input shape: {sample_data.shape}")
    print(f"Number of granularity levels: {len(features)}")
    for i, (feat, recon) in enumerate(zip(features, reconstructions)):
        print(f"Level {i}: features shape {feat.shape}, reconstruction shape {recon.shape}")
    
    # Test loss computation
    loss = msae.compute_loss(sample_data, reconstructions)
    print(f"Multi-granularity loss: {loss.item():.4f}")
    
    print("✅ DiffusionMSAE test completed successfully!")

if __name__ == "__main__":
    test_diffusion_msae()
