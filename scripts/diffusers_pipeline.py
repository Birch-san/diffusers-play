# make sure you're logged in with `huggingface-cli login`
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.models.autoencoder_kl import AutoencoderKL
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
import torch
from torch import Generator, compile
from PIL import Image
from typing import List

vae: AutoencoderKL = AutoencoderKL.from_pretrained('hakurei/waifu-diffusion', subfolder='vae', torch_dtype=torch.float16)

# scheduler args documented here:
# https://github.com/huggingface/diffusers/blob/0392eceba8d42b24fcecc56b2cc1f4582dbefcc4/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py#L83
scheduler: DPMSolverMultistepScheduler = DPMSolverMultistepScheduler.from_pretrained(
  'Birchlabs/wd-1-5-beta3-unofficial',
  subfolder='scheduler',
  # sde-dpmsolver++ is very new. if your diffusers version doesn't have it: use 'dpmsolver++' instead.
  algorithm_type='sde-dpmsolver++',
  solver_order=2,
  # solver_type='heun' may give a sharper image. Cheng Lu reckons midpoint is better.
  solver_type='midpoint',
  use_karras_sigmas=True,
)

# variant=None
# variant='ink'
# variant='mofu'
variant='radiance'
# variant='illusion'
pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
  'Birchlabs/wd-1-5-beta3-unofficial',
  torch_dtype=torch.float16,
  vae=vae,
  scheduler=scheduler,
  variant=variant,
)
pipe.to('cuda')
compile(pipe.unet, mode='reduce-overhead')

# WD1.5 was trained on area=896**2 and no side longer than 1152
sqrt_area=896
# note: pipeline requires width and height to be multiples of 8
height = 1024
width = sqrt_area**2//height

prompt = 'artoria pendragon (fate), reddizen, 1girl, best aesthetic, best quality, blue dress, full body, white shirt, blonde hair, looking at viewer, hair between eyes, floating hair, green eyes, blue ribbon, long sleeves, juliet sleeves, light smile, hair ribbon, outdoors, painting (medium), traditional media'
negative_prompt = 'lowres, bad anatomy, bad hands, missing fingers, extra fingers, blurry, mutation, deformed face, ugly, bad proportions, monster, cropped, worst quality, jpeg, bad posture, long body, long neck, jpeg artifacts, deleted, bad aesthetic, realistic, real life, instagram'

# pipeline invocation args documented here:
# https://github.com/huggingface/diffusers/blob/0392eceba8d42b24fcecc56b2cc1f4582dbefcc4/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#LL544C18-L544C18
out: StableDiffusionPipelineOutput = pipe.__call__(
  prompt,
  negative_prompt=negative_prompt,
  height=height,
  width=width,
  num_inference_steps=22,
  generator=Generator().manual_seed(1234)
)
images: List[Image.Image] = out.images
img, *_ = images

img.save('out_pipe/saber.png')