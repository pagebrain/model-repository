# BASED ON https://github.com/replicate/cog-sdxl/blob/main/predict.py

import os
from typing import List

import shutil
import torch
import numpy as np

from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    HeunDiscreteScheduler,
)
from diffusers.utils import load_image
from transformers import CLIPImageProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)

class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)


SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KarrasDPM": KarrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
    "KLMS": LMSDiscreteScheduler,
}

EMBEDDINGS = [(x.split(".")[0], "/embeddings/"+x) for x in os.listdir("/embeddings/")]
EMBEDDING_TOKENS = [x[0] for x in EMBEDDINGS]
EMBEDDING_PATHS = [x[1] for x in EMBEDDINGS]

MODEL_ID = "pagebrain/realistic-vision-v5-1"
MODEL_CACHE = "diffusers-cache"
SAFETY_MODEL_ID = "CompVis/stable-diffusion-safety-checker"

class Predictor(BasePredictor):
    def load_image(self, path):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")

    def run_safety_checker(self, image):
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(
            "cuda"
        )
        np_image = [np.array(val) for val in image]
        image, has_nsfw_concept = self.safety_checker(
            images=np_image,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return image, has_nsfw_concept

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "/model/safety_checker",
            # cache_dir=MODEL_CACHE,
            local_files_only=True,
            torch_dtype=torch.float16,
        ).to("cuda")

        self.feature_extractor = CLIPImageProcessor.from_pretrained("/model/feature_extractor")

        print("Loading txt2img pipeline...")
        self.txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            "/model/",
            # cache_dir=MODEL_CACHE,
            local_files_only=True,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False
        )
        self.txt2img_pipe.load_textual_inversion(EMBEDDING_PATHS, token=EMBEDDING_TOKENS, local_files_only=True)
        self.txt2img_pipe.to("cuda")

        print("Loading img2img pipeline...")
        self.img2img_pipe = StableDiffusionImg2ImgPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            # text_encoder_2=self.txt2img_pipe.text_encoder_2,
            tokenizer=self.txt2img_pipe.tokenizer,
            # tokenizer_2=self.txt2img_pipe.tokenizer_2,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            feature_extractor=self.txt2img_pipe.feature_extractor,
            safety_checker=None,
            requires_safety_checker=False
        )
        self.img2img_pipe.to("cuda")

        print("Loading inpaint pipeline...")
        self.inpaint_pipe = StableDiffusionInpaintPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            # text_encoder_2=self.txt2img_pipe.text_encoder_2,
            tokenizer=self.txt2img_pipe.tokenizer,
            # tokenizer_2=self.txt2img_pipe.tokenizer_2,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            feature_extractor=self.txt2img_pipe.feature_extractor,
            safety_checker=None,
            requires_safety_checker=False
        )
        self.inpaint_pipe.to("cuda")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default=None,
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output. Supported embeddings: " + ", ".join(EMBEDDING_TOKENS),
            default=None,
        ),
        image: Path = Input(
            description="Input image for img2img or inpaint mode",
            default=None,
        ),
        mask: Path = Input(
            description="Input mask for inpaint mode. Black areas will be preserved, white areas will be inpainted.",
            default=None,
        ),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using init image. 1.0 corresponds to full destruction of information in init image",
            default=0.8,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=SCHEDULERS.keys(),
            default="K_EULER",
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        safety_checker: bool = Input(
            description="Safety checker. Disable to expose unfiltered results", default=True
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )
        
        kwargs = {}
        print(f"Prompt: {prompt}")
        if image and mask:
            print("inpainting mode")
            kwargs["image"] = self.load_image(image)
            kwargs["mask_image"] = self.load_image(mask)
            kwargs["strength"] = prompt_strength
            kwargs["width"] = width
            kwargs["height"] = height
            pipe = self.inpaint_pipe
        elif image:
            print("img2img mode")
            kwargs["image"] = self.load_image(image)
            kwargs["strength"] = prompt_strength
            pipe = self.img2img_pipe
        else:
            print("txt2img mode")
            kwargs["width"] = width
            kwargs["height"] = height
            pipe = self.txt2img_pipe

        # pipe.safety_checker = self.safety_checker if safety_checker else None
        pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)
        generator = torch.Generator("cuda").manual_seed(seed)

        common_args = {
            "prompt": [prompt] * num_outputs if prompt is not None else None,
            "negative_prompt": [negative_prompt] * num_outputs if negative_prompt is not None else None,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }

        output = pipe(**common_args, **kwargs)

        if safety_checker:
            _, has_nsfw_content = self.run_safety_checker(output.images)

        output_paths = []
        for i, sample in enumerate(output.images):
            if safety_checker and has_nsfw_content[i]:
                print(f"NSFW content detected in image {i}")
                continue
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        if safety_checker and len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths
