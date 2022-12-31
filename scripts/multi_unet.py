import os
import time

from helpers.device import DeviceLiteral, get_device_type
from helpers.diffusers_denoiser import DiffusersSDDenoiser, DiffusersSD2Denoiser
from helpers.log_intermediates import LogIntermediates, make_log_intermediates
from helpers.multi_unet_denoiser import MultiUnetCFGDenoiser, GetModelWeight, static_model_weight
from helpers.schedules import KarrasScheduleParams, KarrasScheduleTemplate, get_template_schedule
from helpers.schedule_params import get_alphas, get_alphas_cumprod, get_betas, quantize_to
from helpers.get_seed import get_seed
from helpers.latents_to_pils import LatentsToPils, LatentsToBCHW, make_latents_to_pils, make_latents_to_bchw
from helpers.embed_text_types import Embed
from helpers.embed_text import ClipCheckpoint, ClipImplementation, get_embedder
from k_diffusion.external import DiscreteSchedule
from k_diffusion.sampling import get_sigmas_karras, sample_dpmpp_2m

import torch
from torch import Generator, Tensor, randn, no_grad, zeros
from diffusers.models import UNet2DConditionModel, AutoencoderKL

from dataclasses import dataclass
from typing import List, Dict, Optional
from PIL import Image
from enum import Enum, auto

device_type: DeviceLiteral = get_device_type()
device = torch.device(device_type)

half = False

n_rand_seeds = 20
seeds = [
  # 2178792736,
  *[get_seed() for _ in range(n_rand_seeds)]
]

favourite_dtype: torch.dtype = torch.float16 if half else torch.float32

unet_dtype: torch.dtype = favourite_dtype
sampling_dtype: torch.dtype = favourite_dtype
vae_dtype: torch.dtype = favourite_dtype
clip_dtype: torch.dtype = favourite_dtype

@dataclass
class TextEncoderSpec:
  clip_impl: ClipImplementation
  clip_ckpt: ClipCheckpoint
  subtract_hidden_state_layers: int

@dataclass
class ModelSpec:
  name: str
  revision: Optional[str]
  torch_dtype: Optional[torch.dtype]
  encoder: TextEncoderSpec
  needs_vparam: bool

class ModelId(Enum):
  JPSD = auto()
  WD = auto()
  SD2_BASE = auto()
  SD2 = auto()

unet_revision: Optional[str] = 'fp16' if unet_dtype == torch.float16 else None

encoder = TextEncoderSpec(
  clip_impl=ClipImplementation.HF,
  clip_ckpt=ClipCheckpoint.OpenAI,
  subtract_hidden_state_layers=0
)
sd2_encoder = TextEncoderSpec(
  clip_impl=ClipImplementation.HF,
  clip_ckpt=ClipCheckpoint.LAION,
  subtract_hidden_state_layers=1
)
wd1_4_encoder = TextEncoderSpec(
  clip_impl=ClipImplementation.HF,
  clip_ckpt=ClipCheckpoint.Waifu,
  subtract_hidden_state_layers=1
)
jpsd = ModelSpec(
  name='rinna/japanese-stable-diffusion',
  revision=unet_revision,
  torch_dtype=unet_dtype,
  encoder=encoder,
  needs_vparam=False,
)
wd = ModelSpec(
  # specifications for WD1.4 (untested)
  name='hakurei/waifu-diffusion',
  # at the time of writing: no fp16 revision has been uploaded
  revision=None,
  torch_dtype=unet_dtype,
  encoder=wd1_4_encoder,
  needs_vparam=False,
)
sd2_base = ModelSpec(
  name='stabilityai/stable-diffusion-2-base',
  revision=unet_revision,
  torch_dtype=unet_dtype,
  encoder=sd2_encoder,
  needs_vparam=False,
)
sd2 = ModelSpec(
  name='stabilityai/stable-diffusion-2',
  revision=unet_revision,
  torch_dtype=unet_dtype,
  encoder=sd2_encoder,
  needs_vparam=True,
)

models: Dict[ModelId, ModelSpec] = {
  ModelId.JPSD: jpsd,
  ModelId.WD: wd,
  # ModelId.SD2_BASE: sd2_base,
  # ModelId.SD2: sd2,
}

# if you have limited VRAM then you might not want to transfer these to GPU eagerly.
# but Mac has unified memory so it doesn't make a difference.
unets: Dict[ModelId, UNet2DConditionModel] = {
  id: UNet2DConditionModel.from_pretrained(
    spec.name,
    subfolder='unet',
    revision=spec.revision,
    torch_dtype=spec.torch_dtype,
  ).to(device).eval() for id, spec in models.items()
}

alphas_cumprod: Tensor = get_alphas_cumprod(get_alphas(get_betas(device=device))).to(dtype=sampling_dtype)

denoisers: Dict[ModelId, DiffusersSDDenoiser] = {
  id: DiffusersSD2Denoiser(
    unet,
    alphas_cumprod,
    sampling_dtype
  ) if models[id].needs_vparam else DiffusersSDDenoiser(
    unet,
    alphas_cumprod,
    sampling_dtype
  ) for id, unet in unets.items()
}

denoiser = MultiUnetCFGDenoiser(denoisers)

# we can't do multi-VAE decoding, but not everybody finetunes the VAE
# a lot of them have a common ancestor in SD1.4's VAE
# probably best to pick one that matches one of your models
vae_revision: Optional[str] = 'fp16' if vae_dtype == torch.float16 else None
vae: AutoencoderKL = AutoencoderKL.from_pretrained(
  'hakurei/waifu-diffusion',
  subfolder='vae',
  revision=vae_revision,
  torch_dtype=vae_dtype,
).to(device).eval()
latents_to_bchw: LatentsToBCHW = make_latents_to_bchw(vae)
latents_to_pils: LatentsToPils = make_latents_to_pils(latents_to_bchw)

# this would benefit from a cache to understand when two models want the same embedder
embedders: Dict[ModelId, Embed] = {
  id: get_embedder(
    impl=spec.encoder.clip_impl,
    ckpt=spec.encoder.clip_ckpt,
    subtract_hidden_state_layers=spec.encoder.subtract_hidden_state_layers,
    device=device,
    torch_dtype=clip_dtype
  ) for id, spec in models.items()
}

schedule_template = KarrasScheduleTemplate.Mastering
# grab any of our k-diffusion wrapped denoisers; get_template_schedule() refers to its .sigmas property
# they should all be the same
unet_k_wrapped: DiscreteSchedule = denoisers[ModelId.WD]
schedule: KarrasScheduleParams = get_template_schedule(
  schedule_template,
  model_sigma_min=unet_k_wrapped.sigma_min,
  model_sigma_max=unet_k_wrapped.sigma_max,
  device=unet_k_wrapped.sigmas.device,
  dtype=unet_k_wrapped.sigmas.dtype,
)
steps, sigma_max, sigma_min, rho = schedule.steps, schedule.sigma_max, schedule.sigma_min, schedule.rho
sigmas: Tensor = get_sigmas_karras(
  n=steps,
  sigma_max=sigma_max,
  sigma_min=sigma_min,
  rho=rho,
  device=device,
).to(unet_dtype)
sigmas_quantized = torch.cat([
  quantize_to(sigmas[:-1], unet_k_wrapped.sigmas),
  zeros((1), device=sigmas.device, dtype=sigmas.dtype)
])
print(f"sigmas (quantized):\n{', '.join(['%.4f' % s.item() for s in sigmas_quantized])}")

prompts: Dict[ModelId, str] = {
  # ModelId.JPSD: '伏見稲荷大社のイラスト、copicで作った。',
  # nevermind it was trained on romaji
  ModelId.JPSD: 'fushimi inari taisha no torii no mae de hitori no onna no irasuto, copic de tsukutta, jouzu, kanpeki',
  # ModelId.WD: 'artoria pendragon (fate), carnelian, 1girl, general content, upper body, white shirt, blonde hair, looking at viewer, medium breasts, hair between eyes, floating hair, green eyes, blue ribbon, long sleeves, light smile, hair ribbon, watercolor (medium), traditional media',
  # ModelId.WD: 'carnelian, general content, still life, watercolor (medium), traditional media',
  ModelId.WD: 'kirisame marisa, carnelian, general content, one girl, full body, outdoors, light smile, looking at viewer, hair between eyes, floating hair, small breasts, touhou project, puffy short sleeves, witch hat, black dress, white shirt, buttons, yellow eyes, watercolor (medium), traditional media',
  # ModelId.SD2_BASE: 'an adorable teddy bear sitting on a bed. twilight. high quality. fluffy, wool.',
  # ModelId.SD2: 'an adorable teddy bear sitting on a bed. twilight. high quality. fluffy, wool.',
}
equal_weight: float = 1./len(models)
static_weight_getter: GetModelWeight = static_model_weight(equal_weight)

# initially (high-sigma denoising): only SD2 is used, and carves out the composition for the bear.
# once model starts denoising medium sigmas (below 2): we introduce waifu-diffusion with 60% weighting,
# to influence the fine details towards a watercolour style
cutoff: float = 4.
model_weights: Dict[ModelId, GetModelWeight] = {
  # ModelId.JPSD: static_weight_getter,
  # ModelId.WD: static_weight_getter,
  ModelId.JPSD: lambda sigma: 1. if sigma > cutoff else 0.,
  ModelId.WD: lambda sigma: 0. if sigma > cutoff else 1.,
  # ModelId.SD2_BASE: lambda sigma: 0.4 if sigma < cutoff else 1.
  # ModelId.SD2: static_weight_getter,
}

sample_path='out'
intermediates_path='intermediates'
for path_ in [sample_path, intermediates_path]:
  os.makedirs(path_, exist_ok=True)
log_intermediates: LogIntermediates = make_log_intermediates(intermediates_path)

num_images_per_prompt = 1
width = 512
height = width
latents_shape = (num_images_per_prompt, unet_k_wrapped.inner_model.in_channels, height // 8, width // 8)

with no_grad():
  unconds: Dict[ModelId, Tensor] = {}
  conds: Dict[ModelId, Tensor] = {}
  for id, embed in embedders.items():
    prompt: str = prompts[id]
    text_embeddings: Tensor = embed(['', prompt])
    chunked: Tensor = text_embeddings.chunk(text_embeddings.size(0))
    uc, c = chunked
    unconds[id] = uc
    conds[id] = c
  
  batch_tic = time.perf_counter()
  for seed in seeds:
    generator = Generator(device='cpu').manual_seed(seed)
    latents = randn(latents_shape, generator=generator, device='cpu', dtype=sampling_dtype).to(device)

    tic = time.perf_counter()

    extra_args = {
      'conds': conds,
      'unconds': unconds,
      'cond_scale': 7.5,
      'model_weights': model_weights,
    }
    latents: Tensor = sample_dpmpp_2m(
      denoiser,
      latents * sigmas[0],
      sigmas,
      extra_args=extra_args,
      # callback=log_intermediates,
    ).to(vae_dtype)
    pil_images: List[Image.Image] = latents_to_pils(latents)
    print(f'generated {num_images_per_prompt} images in {time.perf_counter()-tic} seconds')

    base_count = len(os.listdir(sample_path))
    for ix, image in enumerate(pil_images):
      image.save(os.path.join(sample_path, f"{base_count+ix:05}.{seed}.png"))

print(f'in total, generated {len(seeds)} batches of {num_images_per_prompt} images in {time.perf_counter()-batch_tic} seconds')