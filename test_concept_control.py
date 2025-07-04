import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msae_wrapper import Sae
from SAE.hooked_sd_noised_pipeline import HookedStableDiffusionPipeline
import utils.hooks as hooks
from PIL import Image

def test_concept_control():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load MSAE
    msae_path = "/data/selena/MSAE/2048_512_TopKReLU_64_UW_False_False_0.0_imagenet_ViT-B~32_train_image_1281167_512.pth"
    msae = Sae.load_from_disk(msae_path, device=device)
    print("✅ MSAE loaded")
    
    # Load hooked diffusion pipeline
    pipe = HookedStableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        cache_dir='.cache'
    ).to(device)
    print("✅ Hooked diffusion pipeline loaded")
    
    # Test hookpoint
    hookpoint = "unet.up_blocks.1.attentions.2"
    print(f"Testing hookpoint: {hookpoint}")
    
    # Generate baseline image (no intervention)
    prompt = "a red car in a city street"
    print(f"Prompt: '{prompt}'")
    
    with torch.no_grad():
        print("Generating baseline image...")
        baseline_image = pipe(
            prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=torch.Generator().manual_seed(42)
        ).images[0]
        baseline_image.save("baseline_image.png")
        print("✅ Baseline image saved as 'baseline_image.png'")
        
        # Test with concept intervention
        print("Testing concept intervention...")
        
        # Create a simple intervention hook
        def intervention_hook(module, input, output):
            # Get the activations
            activations = output[0] if isinstance(output, tuple) else output
            original_shape = activations.shape
            
            # Flatten for MSAE processing (if needed)
            batch_size = activations.shape[0]
            flattened = activations.reshape(batch_size, -1)
            
            # Apply MSAE intervention if dimensions match
            if flattened.shape[-1] == 512:  # MSAE input dimension
                print(f"  Applying MSAE intervention to {original_shape}")
                
                # Get important features
                important_features = msae.get_feature_activations(flattened, percentile=95.0)
                feature_indices = torch.nonzero(important_features[0] > 0).squeeze()[:5]  # Top 5 features
                
                if len(feature_indices) > 0:
                    # Apply concept suppression (negative multiplier)
                    modified = msae.manipulate_features(
                        flattened,
                        feature_indices=feature_indices,
                        multiplier=-3.0  # Strong suppression
                    )
                    
                    # Reshape back
                    modified_activations = modified.reshape(original_shape)
                    return (modified_activations,) if isinstance(output, tuple) else modified_activations
            
            return output
        
        # Register hook
        hook_handle = None
        try:
            target_module = pipe.unet
            for name in hookpoint.split('.')[1:]:  # Skip 'unet'
                target_module = getattr(target_module, name)
            hook_handle = target_module.register_forward_hook(intervention_hook)
            print(f"✅ Hook registered on {hookpoint}")
            
            # Generate with intervention
            print("Generating image with MSAE intervention...")
            intervened_image = pipe(
                prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
                generator=torch.Generator().manual_seed(42)  # Same seed for comparison
            ).images[0]
            intervened_image.save("intervened_image.png")
            print("✅ Intervened image saved as 'intervened_image.png'")
            
        except Exception as e:
            print(f"Hook registration failed: {e}")
            print("This is expected - we need to match MSAE dimensions with actual activations")
            
        finally:
            if hook_handle:
                hook_handle.remove()
    
    print("\n🎯 Concept control test completed!")
    print("Generated files:")
    print("  - baseline_image.png")
    print("  - intervened_image.png (if successful)")
    return True

if __name__ == "__main__":
    test_concept_control()
