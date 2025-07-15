"""
Traditional TopK SAE implementation for comparison with MSAE
Based on SAeUron's architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TraditionalSAE(nn.Module):
    """
    Traditional Sparse Autoencoder with TopK activation
    Matches SAeUron's implementation for fair comparison
    """
    def __init__(
        self,
        input_dim: int = 1280,
        expansion_factor: int = 16,  # SAeUron uses 16x
        k: int = 32,  # SAeUron uses k=32
        spatial_dims: tuple = (16, 16),
        device: str = "cuda"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = input_dim * expansion_factor
        self.k = k
        self.spatial_dims = spatial_dims
        self.device = device
        
        # Encoder and Decoder
        self.encoder = nn.Linear(input_dim, self.hidden_dim, bias=True)
        self.decoder = nn.Linear(self.hidden_dim, input_dim, bias=True)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights following SAE best practices"""
        # Encoder: Xavier uniform
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        
        # Decoder: Transpose of encoder + normalization
        self.decoder.weight.data = self.encoder.weight.data.T
        nn.init.zeros_(self.decoder.bias)
        
        # Normalize decoder columns
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
    
    def topk_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply TopK activation"""
        topk_values, topk_indices = torch.topk(x, self.k, dim=-1)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk_indices, topk_values)
        return result
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input with TopK sparsity"""
        # Flatten spatial dimensions
        batch_size, channels, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(-1, channels)
        
        # Encode and apply TopK
        encoded = F.relu(self.encoder(x_flat))
        sparse_encoded = self.topk_activation(encoded)
        
        return sparse_encoded
    
    def decode(self, encoded: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
        """Decode back to original spatial format"""
        batch_size, channels, h, w = original_shape
        
        # Decode
        decoded_flat = self.decoder(encoded)
        
        # Reshape to spatial format
        decoded = decoded_flat.view(batch_size, h, w, channels).permute(0, 3, 1, 2)
        
        return decoded
    
    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass"""
        original_shape = x.shape
        
        # Encode
        encoded = self.encode(x)
        
        # Decode
        reconstructed = self.decode(encoded, original_shape)
        
        return encoded, reconstructed
    
    def compute_loss(self, x: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss"""
        return F.mse_loss(reconstructed, x)

# Test function
def test_traditional_sae():
    """Test our traditional SAE"""
    print("Testing TraditionalSAE...")
    
    # Test data
    batch_size, channels, h, w = 4, 1280, 16, 16
    sample_data = torch.randn(batch_size, channels, h, w)
    
    # Create SAE
    sae = TraditionalSAE(input_dim=channels, expansion_factor=16, k=32)
    
    # Forward pass
    encoded, reconstructed = sae(sample_data)
    loss = sae.compute_loss(sample_data, reconstructed)
    
    print(f"Input shape: {sample_data.shape}")
    print(f"Encoded shape: {encoded.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Reconstruction loss: {loss.item():.4f}")
    
    # Check sparsity
    sparsity = (encoded > 0).float().mean()
    expected_sparsity = sae.k / sae.hidden_dim
    print(f"Actual sparsity: {sparsity.item():.1%}")
    print(f"Expected sparsity: {expected_sparsity:.1%}")
    
    print("✅ TraditionalSAE test completed successfully!")

if __name__ == "__main__":
    test_traditional_sae()
