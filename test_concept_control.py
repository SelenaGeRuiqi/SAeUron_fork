import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msae_wrapper import Sae
from SAE.hooked_sd_noised_pipeline import HookedStableDiffusionPipeline

class FinalMSAEController:
    def __init__(self, msae_path, device='cuda'):
        self.device = device
        
        # Load MSAE wrapper
        self.msae = Sae.load_from_disk(msae_path, device=device)
        
        # Create projection layers with consistent dtype
        self.proj_640_to_512 = nn.Linear(640, 512, dtype=torch.float16, device=device)
        self.proj_512_to_640 = nn.Linear(512, 640, dtype=torch.float16, device=device)
        
        # Initialize projections
        with torch.no_grad():
            nn.init.xavier_uniform_(self.proj_640_to_512.weight)
            nn.init.zeros_(self.proj_640_to_512.bias)
            nn.init.xavier_uniform_(self.proj_512_to_640.weight)
            nn.init.zeros_(self.proj_512_to_640.bias)
        
        print("✅ Final MSAE Controller ready")
    
    def apply_intervention(self, activations_640d, multiplier=1.0, num_features=5):
        """Apply MSAE intervention with proper dtype handling"""
        original_shape = activations_640d.shape
        
        # Ensure consistent dtype
        activations_640d = activations_640d.to(torch.float16)
        
        # Flatten spatial: [B, C, H, W] -> [B*H*W, C]
        spatial_tokens = activations_640d.permute(0, 2, 3, 1).reshape(-1, 640)
        
        with torch.no_grad():
            # Project 640D -> 512D (ensure dtype consistency)
            tokens_512d = self.proj_640_to_512(spatial_tokens.to(torch.float16))
            
            # Use wrapper's encode method
            encoded = self.msae.encode(tokens_512d)
            
            if hasattr(encoded, 'top_indices') and encoded.top_indices.numel() > 0:
                # Get feature indices for intervention
                feature_indices = encoded.top_indices[0][:num_features]
                
                # Apply intervention using wrapper's manipulation method
                manipulated_512d = self.msae.manipulate_features(
                    tokens_512d,
                    feature_indices=feature_indices,
                    multiplier=multiplier
                )
                
                # Project back to 640D
                manipulated_640d = self.proj_512_to_640(manipulated_512d.to(torch.float16))
                
                # Reshape to original format
                modified_activations = manipulated_640d.reshape(
                    original_shape[0], original_shape[2], original_shape[3], original_shape[1]
                ).permute(0, 3, 1, 2)
                
                # Match original dtype
                modified_activations = modified_activations.to(activations_640d.dtype)
                
                return modified_activations, True
        
        return activations_640d, False

def test_final_concept_control():
    device = "cuda"
    
    # Create controller
    msae_path = "/data/selena/MSAE/2048_512_TopKReLU_64_UW_False_False_0.0_imagenet_ViT-B~32_train_image_1281167_512.pth"
    controller = FinalMSAEController(msae_path, device)
    
    # Load pipeline
    pipe = HookedStableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        cache_dir='.cache'
    ).to(device)
    
    # Test different strengths
    test_configs = [
        (0.0, "baseline"),
        (-3.0, "suppressed"),
        (3.0, "enhanced")
    ]
    
    for multiplier, name in test_configs:
        print(f"\nGenerating {name} image (multiplier: {multiplier})...")
        
        intervention_count = 0
        
        def msae_hook(module, input, output):
            nonlocal intervention_count
            
            if multiplier == 0.0:
                return output
            
            try:
                activations = output[0] if isinstance(output, tuple) else output
                print(f"  Processing: {activations.shape}, dtype: {activations.dtype}")
                
                if activations.shape[1] == 640:
                    modified, success = controller.apply_intervention(
                        activations, multiplier=multiplier, num_features=8
                    )
                    
                    if success:
                        intervention_count += 1
                        print(f"  ✅ MSAE intervention #{intervention_count} applied!")
                        return (modified,) if isinstance(output, tuple) else modified
                        
            except Exception as e:
                print(f"  ⚠️ Hook error: {e}")
                import traceback
                traceback.print_exc()
            
            return output
        
        # Register hook and generate
        target_module = pipe.unet.down_blocks[1].attentions[1]
        hook_handle = target_module.register_forward_hook(msae_hook)
        
        try:
            with torch.no_grad():
                image = pipe(
                    "a red car in a city street",
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    generator=torch.Generator().manual_seed(42)
                ).images[0]
            
            filename = f"dtype_fixed_msae_{name}_mult{multiplier}.png"
            image.save(filename)
            print(f"  ✅ Saved: {filename}")
            print(f"  🎯 Total interventions applied: {intervention_count}")
            
        finally:
            hook_handle.remove()
    
    print(f"\n🎉 Dtype-fixed concept control test complete!")

if __name__ == "__main__":
    test_final_concept_control()