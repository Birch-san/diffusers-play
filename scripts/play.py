import os

# monkey-patch _randn to use CPU random before k-diffusion uses it
from helpers.brownian_tree_mps_fix import reassuring_message
from helpers.device import DeviceLiteral, get_device_type
from helpers.diffusers_denoiser import DiffusersSDDenoiser, DiffusersSD2Denoiser
from helpers.cfg_denoiser import CFGDenoiser
from helpers.log_intermediates import LogIntermediates, make_log_intermediates
from helpers.schedules import KarrasScheduleParams, KarrasScheduleTemplate, get_template_schedule
print(reassuring_message) # avoid "unused" import :P

import torch
from torch import Generator, Tensor, randn, no_grad
from diffusers.models import UNet2DConditionModel, AutoencoderKL
from k_diffusion.sampling import BrownianTreeNoiseSampler, get_sigmas_karras, sample_dpmpp_2m

from helpers.log_level import log_level
from helpers.schedule_params import get_alphas, get_alphas_cumprod, get_betas
from helpers.get_seed import get_seed
from helpers.latents_to_pils import LatentsToPils, make_latents_to_pils
from helpers.embed_text import ClipCheckpoint, ClipImplementation, Embed, get_embedder

from typing import List
from PIL import Image
import time

half = False
cfg_enabled = True

n_rand_seeds = 0
seeds = [
  2178792735,
  *[get_seed() for _ in range(n_rand_seeds)]
]

revision=None
torch_dtype=None
if half:
  revision='fp16'
  torch_dtype=torch.float16
device_type: DeviceLiteral = get_device_type()
device = torch.device(device_type)

model_name = (
  # 'CompVis/stable-diffusion-v1-4'
  # 'hakurei/waifu-diffusion'
  # 'runwayml/stable-diffusion-v1-5'
  'stabilityai/stable-diffusion-2'
)
is_sd2 = model_name == 'stabilityai/stable-diffusion-2'
unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
  model_name,
  subfolder='unet',
  revision=revision,
  torch_dtype=torch_dtype,
).to(device).eval()

alphas_cumprod: Tensor = get_alphas_cumprod(get_alphas(get_betas(device=device))).to(dtype=unet.dtype)
unet_k_wrapped = DiffusersSD2Denoiser(unet, alphas_cumprod) if is_sd2 else DiffusersSDDenoiser(unet, alphas_cumprod)
denoiser = CFGDenoiser(unet_k_wrapped, one_at_a_time=True)

# vae_model_name = 'hakurei/waifu-diffusion-v1-4' if model_name == 'hakurei/waifu-diffusion' else model_name
vae: AutoencoderKL = AutoencoderKL.from_pretrained(
  model_name,
  subfolder='vae',
  revision=revision,
  torch_dtype=torch_dtype,
).to(device).eval()
latents_to_pils: LatentsToPils = make_latents_to_pils(vae)

clip_impl = ClipImplementation.HF
clip_ckpt = ClipCheckpoint.LAION if is_sd2 else ClipCheckpoint.OpenAI
clip_subtract_hidden_state_layers = 1 if is_sd2 else 0
embed: Embed = get_embedder(
  impl=clip_impl,
  ckpt=clip_ckpt,
  subtract_hidden_state_layers=clip_subtract_hidden_state_layers,
  device=device,
  torch_dtype=torch_dtype
)

schedule_template = KarrasScheduleTemplate.Searching
schedule: KarrasScheduleParams = get_template_schedule(schedule_template, unet_k_wrapped)

steps, sigma_max, sigma_min, rho = schedule.steps, schedule.sigma_max, schedule.sigma_min, schedule.rho
sigmas: Tensor = get_sigmas_karras(
  n=steps,
  sigma_max=sigma_max,
  sigma_min=sigma_min,
  rho=rho,
  device=device,
).to(unet.dtype)

prompt='Emad Mostaque high-fiving Gordon Ramsay'
# prompt = 'artoria pendragon (fate), carnelian, 1girl, general content, upper body, white shirt, blonde hair, looking at viewer, medium breasts, hair between eyes, floating hair, green eyes, blue ribbon, long sleeves, light smile, hair ribbon, watercolor (medium), traditional media'
# prompt = "masterpiece character portrait of a blonde girl, full resolution, 4k, mizuryuu kei, akihiko. yoshida, Pixiv featured, baroque scenic, by artgerm, sylvain sarrailh, rossdraws, wlop, global illumination, vaporwave"

unprompts = [''] if cfg_enabled else []
prompts = [*unprompts, prompt]

sample_path='out'
intermediates_path='intermediates'
for path_ in [sample_path, intermediates_path]:
  os.makedirs(path_, exist_ok=True)
log_intermediates: LogIntermediates = make_log_intermediates(intermediates_path)

cond_scale = 7.5 if cfg_enabled else 1.
batch_size = 1
num_images_per_prompt = 1
width = 768 if is_sd2 else 512
height = width
latents_shape = (batch_size * num_images_per_prompt, unet.in_channels, height // 8, width // 8)
with no_grad():
  text_embeddings: Tensor = embed(prompts)
  chunked = text_embeddings.chunk(text_embeddings.size(0))
  if cfg_enabled:
    uc, c = chunked
  else:
    uc = None
    c, = chunked

  batch_tic = time.perf_counter()
  for seed in seeds:
    generator = Generator(device='cpu').manual_seed(seed)
    latents = randn(latents_shape, generator=generator, device='cpu', dtype=torch_dtype).to(device)

    tic = time.perf_counter()

    extra_args = {
      'cond': c,
      'uncond': uc,
      'cond_scale': cond_scale
    }
    noise_sampler = BrownianTreeNoiseSampler(
      latents,
      sigma_min=sigma_min,
      sigma_max=sigma_max,
      # there's no requirement that the noise sampler's seed be coupled to the init noise seed;
      # I'm just re-using it because it's a convenient arbitrary number
      seed=seed,
    )
    latents: Tensor = sample_dpmpp_2m(
      denoiser,
      latents * sigmas[0],
      sigmas,
      extra_args=extra_args,
      # noise_sampler=noise_sampler, # you can only pass noise sampler to ancestral samplers
      # callback=log_intermediates,
    )
    pil_images: List[Image.Image] = latents_to_pils(latents)
    print(f'generated {batch_size} images in {time.perf_counter()-tic} seconds')

    base_count = len(os.listdir(sample_path))
    for ix, image in enumerate(pil_images):
      image.save(os.path.join(sample_path, f"{base_count+ix:05}.{seed}.png"))

print(f'in total, generated {len(seeds)} batches of {num_images_per_prompt} images in {time.perf_counter()-batch_tic} seconds')