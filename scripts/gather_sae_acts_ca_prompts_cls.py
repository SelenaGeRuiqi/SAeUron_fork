"""
Gather activations from a SAE for a given hookpoint and save them to a file.
"""

import os

import fire
import torch
from diffusers.utils.import_utils import is_xformers_available

from SAE.hooked_sd_noised_pipeline import HookedStableDiffusionPipeline
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msae_wrapper import Sae  # MSAE wrapper
from UnlearnCanvas_resources.const import class_available

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True
import pickle

import tqdm


def main(checkpoint_path, hookpoint, pipe_path, save_dir, steps=100, seed=188):
    cls_prompts_dict = {class_avail: [] for class_avail in class_available}
    for class_avail in class_available:
        with open(
            os.path.join(
                "UnlearnCanvas_resources/anchor_prompts/finetune_prompts",
                f"sd_prompt_{class_avail}.txt",
            ),
            "r",
        ) as prompt_file:
            prompts = prompt_file.readlines()
            prompt = [p.strip() for p in prompts]
            cls_prompts_dict[class_avail].extend(prompt)

    sae = Sae.load_from_disk(
        os.path.join(checkpoint_path, hookpoint), device="cuda"
    ).eval()

    sae = sae.to(dtype=torch.float16)
    sae.cfg.batch_topk = False
    sae.cfg.sample_topk = False

    pipe = HookedStableDiffusionPipeline.from_pretrained(
        pipe_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")
    if is_xformers_available():
        print("Enabling xFormers memory efficient attention")
        pipe.unet.enable_xformers_memory_efficient_attention()

    cls_latents_dict = {}

    progress_bar = tqdm.tqdm(list(cls_prompts_dict.keys()), total=len(cls_prompts_dict))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for class_avail in progress_bar:
        progress_bar.set_description(f"Processing class: {class_avail}")
        prompts = cls_prompts_dict[class_avail]
        _, acts_cache = pipe.run_with_cache(
            prompt=prompts,
            generator=generator,
            num_inference_steps=steps,
            save_input=False,
            save_output=True,
            positions_to_cache=[hookpoint],
            guidance_scale=9.0,
            output_type="latent",  # prevent decoding to pixel space
        )
        activations = acts_cache["output"][hookpoint].cpu()
        assert activations.shape[0] == len(prompts)
        assert activations.shape[1] == steps
        sae_latents = []
        with torch.no_grad():
            for i in range(len(prompts)):
                sae_in = activations[i].reshape(steps, -1, sae.d_in)
                top_acts, top_indices = sae.encode(sae_in.to(sae.device))
                sae_out = torch.zeros(
                    (top_acts.shape[0], sae.num_latents),
                    device=sae.device,
                    dtype=top_acts.dtype,
                ).scatter(-1, top_indices, top_acts)
                sae_out = sae_out.reshape(steps, -1, sae.num_latents).cpu()
                sae_latents.append(sae_out.mean(1).to(dtype=torch.float16))
        cls_latents_dict[class_avail] = torch.stack(sae_latents)

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"cls_latents_dict_{hookpoint}.pkl"), "wb") as f:
        pickle.dump(cls_latents_dict, f)
    print(f"Saved to {save_dir}/cls_latents_dict_{hookpoint}.pkl")


if __name__ == "__main__":
    fire.Fire(main)
