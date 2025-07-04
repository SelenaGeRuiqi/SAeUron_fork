import torch
import torch.nn as nn
from typing import Optional, Tuple, NamedTuple
import numpy as np
from pathlib import Path

# Define the output types locally to avoid import issues
class EncoderOutput(NamedTuple):
    top_acts: torch.Tensor
    """Activations of the top-k latents."""
    top_indices: torch.Tensor
    """Indices of the top-k features."""

class ForwardOutput(NamedTuple):
    sae_out: torch.Tensor
    latent_acts: torch.Tensor
    """Activations of the top-k latents."""

# Simple config class for compatibility
class SimpleConfig:
    def __init__(self, d_in, d_sae, k, **kwargs):
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k
        for key, value in kwargs.items():
            setattr(self, key, value)

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
    MSAE wrapper for SAeUron compatibility
    Final version that works without import conflicts
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
        
        print(f"🔧 Loading MSAE from: {msae_checkpoint_path}")
        
        # Load MSAE checkpoint
        checkpoint = torch.load(msae_checkpoint_path, map_location=device)
        print(f"✅ Checkpoint loaded successfully")
        
        # Extract components from checkpoint
        model_state_dict = checkpoint['model']
        self.mean_center = checkpoint['mean_center'].to(device)
        self.scaling_factor = checkpoint['scaling_factor']
        self.target_norm = checkpoint['target_norm']
        
        # Create simplified MSAE model
        self.msae_model = SimpleMSAEModel(model_state_dict, device)
        print(f"✅ MSAE model created")
        
        # Get MSAE dimensions
        self.msae_input_dim = self.msae_model.n_inputs
        self.msae_latent_dim = self.msae_model.n_latents
        
        print(f"📐 MSAE dimensions: {self.msae_input_dim} -> {self.msae_latent_dim}")
        
        # Set up dimension adaptation
        self.target_dim = target_dim if target_dim is not None else self.msae_input_dim
        
        if self.target_dim != self.msae_input_dim:
            print(f"🔗 Adding projection layers: {self.target_dim} <-> {self.msae_input_dim}")
            # Project diffusion activations to MSAE input space
            self.input_projection = nn.Linear(self.target_dim, self.msae_input_dim).to(device)
            # Project MSAE output back to diffusion space
            self.output_projection = nn.Linear(self.msae_input_dim, self.target_dim).to(device)
            self._init_projections()
        else:
            self.input_projection = None
            self.output_projection = None
            
        # Create config for compatibility
        self.cfg = SimpleConfig(
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
        Compatible with SAeUron's feature selection approach
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
        Compatible with SAeUron's intervention approach
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
                # Ensure shapes match for boolean mask
                if feature_indices.shape != z.shape:
                    if len(feature_indices.shape) > 2:
                        feature_indices = feature_indices.view(-1, feature_indices.shape[-1])
                    # If still mismatched, reshape to match z
                    if feature_indices.shape != z.shape:
                        # Broadcast or reshape as needed
                        if feature_indices.shape[-1] == z.shape[-1]:
                            # Expand batch dimension if needed
                            if len(feature_indices.shape) == 1:
                                feature_indices = feature_indices.unsqueeze(0).expand(z.shape[0], -1)
                            elif feature_indices.shape[0] == 1 and z.shape[0] > 1:
                                feature_indices = feature_indices.expand(z.shape[0], -1)
                z_modified = z_modified * (~feature_indices).float() + z_modified * feature_indices.float() * multiplier
                
            elif feature_indices.dtype in [torch.float32, torch.float64]:
                # Convert float mask to boolean
                feature_mask = feature_indices > 0.5
                # Ensure proper shape matching
                if feature_mask.shape != z.shape:
                    if len(feature_mask.shape) > 2:
                        feature_mask = feature_mask.view(-1, feature_mask.shape[-1])
                    if feature_mask.shape != z.shape:
                        if feature_mask.shape[-1] == z.shape[-1]:
                            if len(feature_mask.shape) == 1:
                                feature_mask = feature_mask.unsqueeze(0).expand(z.shape[0], -1)
                            elif feature_mask.shape[0] == 1 and z.shape[0] > 1:
                                feature_mask = feature_mask.expand(z.shape[0], -1)
                z_modified = z_modified * (~feature_mask).float() + z_modified * feature_mask.float() * multiplier
                
            else:
                # Integer indices - convert to boolean mask
                if feature_indices.dtype in [torch.long, torch.int32, torch.int64]:
                    # Create boolean mask from indices
                    feature_mask = torch.zeros_like(z, dtype=torch.bool)
                    for b in range(z.shape[0]):
                        if len(feature_indices.shape) > 1:
                            indices = feature_indices[b]
                        else:
                            indices = feature_indices
                        # Filter valid indices
                        valid_indices = indices[indices < z.shape[-1]]
                        if len(valid_indices) > 0:
                            feature_mask[b, valid_indices] = True
                    z_modified = z_modified * (~feature_mask).float() + z_modified * feature_mask.float() * multiplier
                else:
                    raise ValueError(f"Unsupported feature_indices dtype: {feature_indices.dtype}")
            
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
        """Load MSAE wrapper from checkpoint - SAeUron compatibility method"""
        return cls(checkpoint_path, target_dim, device, **kwargs)
    
    @classmethod
    def load_from_hub(cls, repo_id: str, hookpoint: str = None, device: str = "cuda", **kwargs):
        """
        Compatibility method for SAeUron's load_from_hub interface
        For MSAE, we'll load from local checkpoint instead
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

    @classmethod
    def load_from_disk(cls, path: str, device: str = "cuda", **kwargs):
        """
        SAeUron compatibility method - maps to load_from_checkpoint
        """
        # For MSAE, we expect the path to point to our checkpoint file
        # If path is a directory, look for our checkpoint pattern
        from pathlib import Path
        path_obj = Path(path)
        
        if path_obj.is_dir():
            # Look for MSAE checkpoint in the directory
            msae_files = list(path_obj.glob("*.pth"))
            if msae_files:
                checkpoint_path = str(msae_files[0])  # Use first .pth file found
            else:
                raise FileNotFoundError(f"No .pth checkpoint found in {path}")
        else:
            checkpoint_path = str(path)
        
        return cls.load_from_checkpoint(checkpoint_path, device=device, **kwargs)

    @classmethod
    def load_many(cls, name: str, local: bool = True, layers: list = None, 
                device: str = "cuda", decoder: bool = True, pattern: str = None):
        """
        SAeUron compatibility method for loading multiple SAEs
        For MSAE, we'll just load one model but return it in the expected format
        """
        # For MSAE, we typically have one checkpoint, so we'll return a dict with one entry
        from pathlib import Path
        
        if local:
            path_obj = Path(name)
            if path_obj.is_file():
                # Single checkpoint file
                msae = cls.load_from_checkpoint(str(path_obj), device=device)
                return {"msae": msae}
            elif path_obj.is_dir():
                # Directory with checkpoint(s)
                msae_files = list(path_obj.glob("*.pth"))
                if msae_files:
                    results = {}
                    for i, checkpoint_path in enumerate(msae_files):
                        key = f"msae_{i}" if len(msae_files) > 1 else "msae"
                        results[key] = cls.load_from_checkpoint(str(checkpoint_path), device=device)
                    return results
                else:
                    raise FileNotFoundError(f"No .pth checkpoints found in {path_obj}")
        else:
            # Non-local loading (would need HuggingFace hub implementation)
            raise NotImplementedError("Non-local loading not implemented for MSAE")

    def save_to_disk(self, path: str):
        """
        SAeUron compatibility method for saving
        For MSAE, we'll save our wrapper config and point to original checkpoint
        """
        from pathlib import Path
        import json
        
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        
        # Save config for compatibility
        config = self.get_config()
        with open(path_obj / "msae_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"MSAE config saved to {path_obj}")
        print(f"Note: Original checkpoint is at {self.msae_checkpoint_path}")

    @property
    def device(self):
        """SAeUron compatibility - return device"""
        return self._device if hasattr(self, '_device') else self.msae_model.encoder.device

    @device.setter  
    def device(self, value):
        """SAeUron compatibility - set device"""
        self._device = value

# Main class alias for easy import
Sae = MSAEWrapper

# Factory function for SAeUron compatibility
def create_msae_for_saeuron(checkpoint_path: str, target_dim: int, device: str = "cuda"):
    """
    Create MSAE wrapper configured for SAeUron integration
    """
    return MSAEWrapper.load_from_checkpoint(checkpoint_path, target_dim, device)