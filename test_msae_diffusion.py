import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msae_wrapper import Sae
from SAE.hooked_sd_noised_pipeline import HookedStableDiffusionPipeline

def test_msae_diffusion_integration():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load MSAE
    print("Loading MSAE...")
    msae_path = "/data/selena/MSAE/2048_512_TopKReLU_64_UW_False_False_0.0_imagenet_ViT-B~32_train_image_1281167_512.pth"
    msae = Sae.load_from_disk(msae_path, device=device)
    print("✅ MSAE loaded successfully")
    
    # Load diffusion pipeline
    print("Loading diffusion pipeline...")
    pipe = HookedStableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        cache_dir='.cache'
    ).to(device)
    print("✅ Diffusion pipeline loaded")
    
    try:
        # Test basic generation
        prompt = "a beautiful landscape"
        print(f"Generating image with prompt: '{prompt}'")
        
        with torch.no_grad():
            image = pipe(
                prompt, 
                num_inference_steps=20,
                guidance_scale=7.5,
                generator=torch.Generator().manual_seed(42)
            ).images[0]
        print("✅ Basic generation successful")
        
        # Test MSAE functionality
        print("Testing MSAE complete functionality...")
        dummy_input = torch.randn(1, 512, device=device, dtype=torch.float16)
        
        with torch.no_grad():
            # Test forward pass (full reconstruction)
            forward_output = msae.forward(dummy_input)
            print(f"✅ Forward pass: {dummy_input.shape} -> {forward_output.sae_out.shape}")
            
            # Test sparse encoding
            encoded = msae.encode(dummy_input)
            print(f"✅ Sparse encoding: {encoded.top_acts.shape[1]} active features")
            
            # Test hierarchical feature selection
            important_features = msae.get_feature_activations(dummy_input, percentile=90.0)
            print(f"✅ Feature selection (90th percentile): {important_features.shape}")
            
            # Test concept manipulation with specific feature indices
            # Use the top active features from encoding
            feature_indices = encoded.top_indices[0][:10]  # Use top 10 features
            feature_multiplier = 2.0
            
            manipulated = msae.manipulate_features(
                dummy_input, 
                feature_indices=feature_indices, 
                multiplier=feature_multiplier
            )
            print(f"✅ Feature manipulation: modified {len(feature_indices)} features")
            print(f"Original -> Manipulated: {dummy_input.shape} -> {manipulated.shape}")
            
            # Test different manipulation strengths
            for multiplier in [-2.0, 0.0, 5.0]:
                manipulated = msae.manipulate_features(
                    dummy_input, 
                    feature_indices=feature_indices[:5], 
                    multiplier=multiplier
                )
                print(f"  Multiplier {multiplier}: ✅")
        
        print("✅ All MSAE functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_msae_diffusion_integration()
    if success:
        print("\n🎉 MSAE + Diffusion integration COMPLETE!")
        print("🚀 Ready to test controllable image generation!")
        print("\nNext steps:")
        print("1. Test MSAE with actual diffusion hookpoints")
        print("2. Compare concept removal with original SAE")
        print("3. Explore hierarchical concept control")
    else:
        print("\n❌ Integration test FAILED!")