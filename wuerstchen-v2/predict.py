# BASED ON https://github.com/replicate/cog-sdxl/blob/main/predict.py

import os
from typing import List

import shutil
import torch
import numpy as np

from cog import BasePredictor, Input, Path

from diffusers import AutoPipelineForText2Image
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS

MODEL_CACHE = "/diffusers-cache"

class Predictor(BasePredictor):
    def setup(self):
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "warp-ai/wuerstchen", 
            torch_dtype=torch.float16, 
            use_safetensors=True,
            cache_dir=MODEL_CACHE
            )
        self.pipe.to("cuda")
        # self.pipe.enable_xformers_memory_efficient_attention()

        caption = "Anthropomorphic cat dressed as a fire fighter"
        images = self.pipe(
            caption, 
            width=1024,
            height=1536,
            prior_timesteps=DEFAULT_STAGE_C_TIMESTEPS,
            prior_guidance_scale=4.0,
            num_images_per_prompt=2,
        ).images
        print("Loading pipeline...")
        

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default=None,
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output.",
            default=None,
        ),
        # image: Path = Input(
        #     description="Input image for img2img or inpaint mode",
        #     default=None,
        # ),
        # mask: Path = Input(
        #     description="Input mask for inpaint mode. Black areas will be preserved, white areas will be inpainted.",
        #     default=None,
        # ),
        width: int = Input(
            description="Width of output image. Maximum size is 1536",
            choices=[512, 1024, 1536],
            default=1024,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1536",
            choices=[512, 1024, 1536],
            default=1024,
        ),
        # prompt_strength: float = Input(
        #     description="Prompt strength when using init image. 1.0 corresponds to full destruction of information in init image",
        #     default=0.8,
        # ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        
        prior_guidance_scale: float = Input(
            description="prior_guidance_scale", ge=1.0, le=20.0, default=4.0
        ),
        prior_num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=30
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=12
        ),
        decoder_guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=0.0, le=20.0, default=0.0
        ),
        # scheduler: str = Input(
        #     description="scheduler",
        #     choices=SCHEDULERS.keys(),
        #     default="K_EULER",
        # ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        # safety_checker: bool = Input(
        #     description="Safety checker. Disable to expose unfiltered results", default=True
        # ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        # if width * height > 786432:
        #     raise ValueError(
        #         "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
        #     )

        print(f"Prompt: {prompt}")
        generator = torch.Generator("cuda").manual_seed(seed)

        common_args = {
            "prompt": prompt if prompt is not None else None,
            "negative_prompt": negative_prompt if negative_prompt is not None else None,
            "width": width,
            "height": height,
            "num_images_per_prompt": num_outputs,
            "prior_guidance_scale": prior_guidance_scale,
            "prior_num_inference_steps": prior_num_inference_steps,
            "num_inference_steps": num_inference_steps,
            "decoder_guidance_scale": decoder_guidance_scale,
            "generator": generator,
        }

        output = self.pipe(**common_args)

        # if safety_checker:
        #     _, has_nsfw_content = self.run_safety_checker(output.images)
        
        output_paths = []
        for i, sample in enumerate(output.images):
            # if safety_checker and has_nsfw_content[i]:
            #     print(f"NSFW content detected in image {i}")
            #     continue
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        # if safety_checker and len(output_paths) == 0:
        #     raise Exception(
        #         f"NSFW content detected. Try running it again, or try a different prompt."
        #     )

        return output_paths
