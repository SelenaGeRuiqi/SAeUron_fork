import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msae_wrapper import Sae
from SAE.hooked_sd_noised_pipeline import HookedStableDiffusionPipeline

def diagnose_diffusion_activations():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load pipeline
    pipe = HookedStableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        cache_dir='.cache'
    ).to(device)
    
    # Test different hookpoints and capture their dimensions
    hookpoints_to_test = [
        "unet.up_blocks.1.attentions.2",
        "unet.up_blocks.0.attentions.0", 
        "unet.mid_block.attentions.0",
        "unet.down_blocks.1.attentions.1",
        "unet.down_blocks.2.attentions.1"
    ]
    
    activation_info = {}
    
    def create_hook(hookpoint_name):
        def hook_fn(module, input, output):
            activations = output[0] if isinstance(output, tuple) else output
            shape = activations.shape
            # Store info about this hookpoint
            activation_info[hookpoint_name] = {
                'shape': shape,
                'total_elements': activations.numel(),
                'last_dim': shape[-1] if len(shape) > 1 else shape[0]
            }
            print(f"{hookpoint_name}: {shape}")
        return hook_fn
    
    # Register hooks for all hookpoints
    handles = []
    for hookpoint in hookpoints_to_test:
        try:
            target_module = pipe.unet
            for name in hookpoint.split('.')[1:]:
                target_module = getattr(target_module, name)
            handle = target_module.register_forward_hook(create_hook(hookpoint))
            handles.append(handle)
            print(f"✅ Hook registered: {hookpoint}")
        except Exception as e:
            print(f"❌ Failed to register {hookpoint}: {e}")
    
    # Run inference to capture activations
    print("\nCapturing activations during inference...")
    with torch.no_grad():
        _ = pipe(
            "a simple test",
            num_inference_steps=5,  # Quick test
            guidance_scale=7.5,
            generator=torch.Generator().manual_seed(42)
        )
    
    # Clean up hooks
    for handle in handles:
        handle.remove()
    
    # Analyze results
    print(f"\n📊 ACTIVATION ANALYSIS:")
    print(f"MSAE expects: 512 dimensions")
    print("-" * 50)
    
    compatible_hookpoints = []
    for hookpoint, info in activation_info.items():
        shape = info['shape']
        last_dim = info['last_dim']
        compatible = "✅" if last_dim == 512 else "❌"
        print(f"{hookpoint}: {shape} | Last dim: {last_dim} {compatible}")
        
        if last_dim == 512:
            compatible_hookpoints.append(hookpoint)
    
    print(f"\n🎯 Compatible hookpoints for MSAE (512D):")
    for hp in compatible_hookpoints:
        print(f"  - {hp}")
    
    if not compatible_hookpoints:
        print("❌ No direct 512D matches found!")
        print("💡 Solutions:")
        print("  1. Use projection layers (current approach)")
        print("  2. Train MSAE on diffusion activations")
        print("  3. Find hookpoints with 512D activations")
    
    return compatible_hookpoints

if __name__ == "__main__":
    compatible = diagnose_diffusion_activations()
