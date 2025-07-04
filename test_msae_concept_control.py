import torch
import torch.nn as nn
from typing import Optional, Tuple, NamedTuple
import numpy as np
from pathlib import Path

# Import SAeUron's required output types
from SAE.sae import EncoderOutput, ForwardOutput, SaeConfig

class SimpleMSAEModel(nn.Module):
    """
    Simplified MSAE model that loads directly from state dictionary
    without external dependencies
    """
    def __init__(self, state_dict, device):
        super().__init__()
        
        # Extract dimensions from state dict
        encoder_weight = state_dict['encoder']
        self.n_inputs = encoder_weight.shape[0]
        self.n_latents = encoder_weight.shape[1]
        
        # Create parameters from state dict
        self.encoder = nn.Parameter(encoder_weight.clone().to(device))
        self.decoder = nn.Parameter(state_dict['decoder'].clone().to(device))
        self.pre_bias = nn.Parameter(state_dict['pre_bias'].clone().to(device))
        self.latent_bias = nn.Parameter(state_dict['latent_bias'].clone().to(device))
        
        # Store activation frequency for reference
        if 'latents_activation_frequency' in state_dict:
            self.register_buffer('latents_activation_frequency', 
                               state_dict['latents_activation_frequency'].clone().to(device))
    
    def encode(self, x):
        """Encode input to latent space"""
        # Remove pre-bias and apply encoder
        x_centered = x - self.pre_bias
        latents = torch.matmul(x_centered, self.encoder) + self.latent_bias
        
        # Apply ReLU activation (simple version of TopK)
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

class MSAEWrapper(nn.Module):
    """
    Simplified MSAE wrapper for SAeUron compatibility
    Uses direct state dict loading without external dependencies
    """
    
    def __init__(self, 
                 msae_checkpoint_path: str,
                 target_dim: int = None,
                 device: str = "cuda",
                 k: int = 64):
        super().__init__()
        self.device = device
        self.msae_checkpoint_path = msae_checkpoint_path
        self.k = k  # TopK parameter for compatibility
        
        print(f"Loading MSAE from: {msae_checkpoint_path}")
        
        # Load MSAE checkpoint
        try:
            checkpoint = torch.load(msae_checkpoint_path, map_location=device)
            print("✅ Checkpoint loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")
        
        # Extract components from checkpoint
        model_state_dict = checkpoint['model']
        self.mean_center = checkpoint['mean_center'].to(device)
        self.scaling_factor = checkpoint['scaling_factor']
        self.target_norm = checkpoint['target_norm']
        
        print(f"✅ Extracted checkpoint components")
        
        # Create simplified MSAE model
        try:
            self.msae_model = SimpleMSAEModel(model_state_dict, device)
            print(f"✅ Created MSAE model")
        except Exception as e:
            raise RuntimeError(f"Failed to create MSAE model: {e}")
        
        # Get MSAE dimensions
        self.msae_input_dim = self.msae_model.n_inputs
        self.msae_latent_dim = self.msae_model.n_latents
        
        print(f"MSAE dimensions: {self.msae_input_dim} -> {self.msae_latent_dim}")
        
        # Set up dimension adaptation
        self.target_dim = target_dim if target_dim is not None else self.msae_input_dim
        
        if self.target_dim != self.msae_input_dim:
            print(f"Adding projection layers: {self.target_dim} <-> {self.msae_input_dim}")
            # Project diffusion activations to MSAE input space
            self.input_projection = nn.Linear(self.target_dim, self.msae_input_dim).to(device)
            # Project MSAE output back to diffusion space
            self.output_projection = nn.Linear(self.msae_input_dim, self.target_dim).to(device)
            self._init_projections()
        else:
            self.input_projection = None
            self.output_projection = None
            
        # Create config for compatibility
        self.cfg = SaeConfig(
            d_in=self.target_dim,
            d_sae=self.msae_latent_dim,
            k=k,
            auxk_alpha=0.0,
            dead_feature_threshold=0,
            normalize_activations="none"
        )
            
        self.eval()
        print("✅ MSAE wrapper initialization complete")
    
    def _init_projections(self):
        """Initialize projection layers to preserve information"""
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
    
    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MSAE preprocessing (projection, centering and scaling)"""
        # Project to MSAE input space if needed
        if self.input_projection is not None:
            x = self.input_projection(x)
            
        # Apply MSAE preprocessing (same as in MSAE training)
        x_centered = x - self.mean_center
        x_scaled = x_centered * self.scaling_factor
        return x_scaled
    
    def postprocess_output(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse MSAE preprocessing and project back if needed"""
        # Reverse MSAE preprocessing
        x_unscaled = x / self.scaling_factor
        x_uncentered = x_unscaled + self.mean_center
        
        # Project back to target space if needed
        if self.output_projection is not None:
            x_uncentered = self.output_projection(x_uncentered)
            
        return x_uncentered
    
    def encode(self, x: torch.Tensor) -> EncoderOutput:
        """
        Encode input through MSAE encoder
        Returns SAeUron-compatible EncoderOutput
        """
        # Handle different input shapes
        original_shape = x.shape
        if len(original_shape) > 2:
            # Flatten batch dimensions
            x = x.view(-1, original_shape[-1])
        
        x_prep = self.preprocess_input(x)
        
        # Get latent activations from MSAE
        z = self.msae_model.encode(x_prep)  # Shape: [batch, msae_latent_dim]
        
        # Apply TopK to match SAeUron's sparse behavior
        batch_size = z.shape[0]
        top_acts_list = []
        top_indices_list = []
        
        for i in range(batch_size):
            z_sample = z[i]
            # Get top-k values and indices
            k_actual = min(self.k, z_sample.shape[0])
            top_values, top_idx = torch.topk(z_sample, k=k_actual)
            # Only keep positive activations
            positive_mask = top_values > 0
            top_values = top_values[positive_mask]
            top_idx = top_idx[positive_mask]
            
            top_acts_list.append(top_values)
            top_indices_list.append(top_idx)
        
        # Pad to consistent length for batching
        max_len = max(len(acts) for acts in top_acts_list) if top_acts_list else 1
        if max_len == 0:
            max_len = 1
            
        padded_acts = torch.zeros(batch_size, max_len, device=self.device)
        padded_indices = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        
        for i, (acts, indices) in enumerate(zip(top_acts_list, top_indices_list)):
            if len(acts) > 0:
                padded_acts[i, :len(acts)] = acts
                padded_indices[i, :len(indices)] = indices
        
        # Reshape back to match original batch dimensions if needed
        if len(original_shape) > 2:
            batch_dims = original_shape[:-1]
            padded_acts = padded_acts.view(batch_dims + (max_len,))
            padded_indices = padded_indices.view(batch_dims + (max_len,))
        
        return EncoderOutput(top_acts=padded_acts, top_indices=padded_indices)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent activations back to input space"""
        original_shape = z.shape
        if len(original_shape) > 2:
            z = z.view(-1, original_shape[-1])
            
        x_recon = self.msae_model.decode(z)
        x_recon = self.postprocess_output(x_recon)
        
        if len(original_shape) > 2:
            output_dims = original_shape[:-1] + (x_recon.shape[-1],)
            x_recon = x_recon.view(output_dims)
            
        return x_recon
    
    def forward(self, x: torch.Tensor) -> ForwardOutput:
        """
        Forward pass through MSAE
        Returns SAeUron-compatible ForwardOutput
        """
        # Handle input shape
        original_shape = x.shape
        if len(original_shape) > 2:
            x = x.view(-1, original_shape[-1])
        
        # Get preprocessed input
        x_prep = self.preprocess_input(x)
        
        # Forward pass through MSAE
        z = self.msae_model.encode(x_prep)
        x_recon_prep = self.msae_model.decode(z)
        
        # Postprocess output
        x_recon = self.postprocess_output(x_recon_prep)
        
        # Reshape back if needed
        if len(original_shape) > 2:
            output_dims = original_shape[:-1] + (x_recon.shape[-1],)
            x_recon = x_recon.view(output_dims)
            z_dims = original_shape[:-1] + (z.shape[-1],)
            z = z.view(z_dims)
        
        return ForwardOutput(sae_out=x_recon, latent_acts=z)
    
    def get_feature_activations(self, x: torch.Tensor, 
                              percentile: float = 99.0) -> torch.Tensor:
        """
        Get high-activating features for concept manipulation
        """
        with torch.no_grad():
            original_shape = x.shape
            if len(original_shape) > 2:
                x = x.view(-1, original_shape[-1])
                
            x_prep = self.preprocess_input(x)
            z = self.msae_model.encode(x_prep)
            
            # Calculate threshold for top percentile
            z_flat = z.flatten()
            z_nonzero = z_flat[z_flat > 0]
            
            if len(z_nonzero) == 0:
                threshold = 0.0
            else:
                threshold = torch.quantile(z_nonzero, percentile / 100.0)
            
            # Return mask of activations above threshold
            mask = (z > threshold).float()
            
            # Reshape back if needed
            if len(original_shape) > 2:
                mask_dims = original_shape[:-1] + (mask.shape[-1],)
                mask = mask.view(mask_dims)
                
            return mask
    
    def manipulate_features(self, x: torch.Tensor, 
                          feature_indices: torch.Tensor,
                          multiplier: float = -1.0) -> torch.Tensor:
        """
        Manipulate specific features for concept editing
        """
        with torch.no_grad():
            original_shape = x.shape
            if len(original_shape) > 2:
                x = x.view(-1, original_shape[-1])
                
            # Get full latent representation
            x_prep = self.preprocess_input(x)
            z = self.msae_model.encode(x_prep)
            
            # Apply manipulation to selected features
            z_modified = z.clone()
            
            # Handle feature indices (boolean mask or indices)
            if feature_indices.dtype == torch.bool:
                # Ensure shapes match
                if feature_indices.shape != z.shape:
                    if len(feature_indices.shape) > 2:
                        feature_indices = feature_indices.view(-1, feature_indices.shape[-1])
                z_modified = z_modified * (1 - feature_indices) + z_modified * feature_indices * multiplier
            else:
                # Feature indices as integer indices
                z_modified[:, feature_indices] *= multiplier
            
            # Decode back to input space
            x_recon_prep = self.msae_model.decode(z_modified)
            x_modified = self.postprocess_output(x_recon_prep)
            
            # Reshape back if needed
            if len(original_shape) > 2:
                output_dims = original_shape[:-1] + (x_modified.shape[-1],)
                x_modified = x_modified.view(output_dims)
                
            return x_modified
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, target_dim: int = None, 
                           device: str = "cuda", **kwargs):
        """Load MSAE wrapper from checkpoint"""
        return cls(checkpoint_path, target_dim, device, **kwargs)
    
    @classmethod
    def load_from_hub(cls, repo_id: str, hookpoint: str = None, device: str = "cuda", **kwargs):
        """
        Compatibility method for SAeUron's load_from_hub interface
        """
        checkpoint_path = repo_id
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"MSAE checkpoint not found at: {checkpoint_path}")
        
        return cls.load_from_checkpoint(checkpoint_path, device=device, **kwargs)
    
    def get_config(self):
        """Return configuration dictionary for compatibility"""
        return {
            'msae_input_dim': self.msae_input_dim,
            'msae_latent_dim': self.msae_latent_dim,
            'target_dim': self.target_dim,
            'checkpoint_path': self.msae_checkpoint_path,
            'k': self.k
        }

# Main class alias for easy import
Sae = MSAEWrapper

# Factory function for SAeUron compatibility
def create_msae_for_saeuron(checkpoint_path: str, target_dim: int, device: str = "cuda"):
    """
    Create MSAE wrapper configured for SAeUron integration
    """
    return MSAEWrapper.load_from_checkpoint(checkpoint_path, target_dim, device)