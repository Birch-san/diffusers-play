import os

# monkey-patch _randn to use CPU random before k-diffusion uses it
from helpers.brownian_tree_mps_fix import reassuring_message
from helpers.cumsum_mps_fix import reassuring_message as reassuring_message_2
from helpers.device import DeviceLiteral, get_device_type
from helpers.diffusers_denoiser import DiffusersSDDenoiser, DiffusersSD2Denoiser
from helpers.cfg_denoiser import Denoiser, DenoiserFactory
from helpers.log_intermediates import LogIntermediates, make_log_intermediates
from helpers.schedules import KarrasScheduleParams, KarrasScheduleTemplate, get_template_schedule
print(reassuring_message) # avoid "unused" import :P
print(reassuring_message_2)

import torch
from torch import Tensor, no_grad, zeros, FloatTensor
from diffusers.models import UNet2DConditionModel, AutoencoderKL
from k_diffusion.sampling import BrownianTreeNoiseSampler, get_sigmas_karras, sample_dpmpp_2m

from helpers.schedule_params import get_alphas, get_alphas_cumprod, get_betas, quantize_to
from helpers.get_seed import get_seed
from helpers.latents_to_pils import LatentsToPils, LatentsToBCHW, make_latents_to_pils, make_latents_to_bchw
from helpers.embed_text_types import Embed, EmbeddingAndMask
from helpers.embed_text import ClipCheckpoint, ClipImplementation, get_embedder
from helpers.model_db import get_model_needs, ModelNeeds
from helpers.inference_spec.sample_spec_2 import SampleSpec
from helpers.inference_spec.latent_spec import SeedSpec
from helpers.inference_spec.latents_shape import LatentsShape
from helpers.inference_spec.cond_spec import ConditionSpec, SingleCondition
from helpers.inference_spec.execution_plan_batcher_2 import ExecutionPlanBatcher, BatchSpecGeneric
from helpers.inference_spec.execution_plan import ExecutionPlan, make_execution_plan
from helpers.inference_spec.batch_latent_maker import BatchLatentMaker
from helpers.inference_spec.latent_maker import LatentMaker, MakeLatentsStrategy
from helpers.inference_spec.latent_maker_seed_strategy import SeedLatentMaker
from itertools import chain, repeat

from typing import List, Generator, Iterable
from PIL import Image
import time

half = True
cfg_enabled = True

revision=None
torch_dtype=None
if half:
  revision='fp16'
  torch_dtype=torch.float16
device_type: DeviceLiteral = get_device_type()
device = torch.device(device_type)

model_name = (
  # 'CompVis/stable-diffusion-v1-4'
  'hakurei/waifu-diffusion'
  # 'runwayml/stable-diffusion-v1-5'
  # 'stabilityai/stable-diffusion-2'
  # 'stabilityai/stable-diffusion-2-1'
  # 'stabilityai/stable-diffusion-2-base'
  # 'stabilityai/stable-diffusion-2-1-base'
)

model_needs: ModelNeeds = get_model_needs(model_name, torch.float32 if torch_dtype is None else torch_dtype)

is_768 = model_needs.is_768
needs_vparam = model_needs.needs_vparam
needs_penultimate_clip_hidden_state = model_needs.needs_penultimate_clip_hidden_state
upcast_attention = model_needs.needs_upcast_attention

# WD 1.4 hasn't uploaded an fp16 revision yet
revision = None if model_name == 'hakurei/waifu-diffusion' else revision
unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
  model_name,
  subfolder='unet',
  revision=revision,
  torch_dtype=torch_dtype,
  upcast_attention=upcast_attention,
).to(device).eval()

# sampling in higher-precision helps to converge more stably toward the "true" image (not necessarily better-looking though)
sampling_dtype: torch.dtype = torch.float32
# sampling_dtype: torch.dtype = torch_dtype
alphas_cumprod: Tensor = get_alphas_cumprod(get_alphas(get_betas(device=device))).to(dtype=sampling_dtype)
unet_k_wrapped = DiffusersSD2Denoiser(unet, alphas_cumprod, sampling_dtype) if needs_vparam else DiffusersSDDenoiser(unet, alphas_cumprod, sampling_dtype)
denoiser_factory = DenoiserFactory(unet_k_wrapped)

vae_dtype = torch_dtype
vae_revision = revision
# you can make VAE 32-bit but it looks the same to me and would be slightly slower + more disk space
# vae_dtype: torch.dtype = torch.float32
# vae_revision=None

vae: AutoencoderKL = AutoencoderKL.from_pretrained(
  model_name,
  subfolder='vae',
  revision=vae_revision,
  torch_dtype=vae_dtype,
).to(device).eval()
latents_to_bchw: LatentsToBCHW = make_latents_to_bchw(vae)
latents_to_pils: LatentsToPils = make_latents_to_pils(latents_to_bchw)

clip_impl = ClipImplementation.HF
clip_ckpt: ClipCheckpoint = model_needs.clip_ckpt
clip_subtract_hidden_state_layers = 1 if needs_penultimate_clip_hidden_state else 0
embed: Embed = get_embedder(
  impl=clip_impl,
  ckpt=clip_ckpt,
  subtract_hidden_state_layers=clip_subtract_hidden_state_layers,
  max_context_segments=model_needs.xattn_max_context_segments,
  device=device,
  torch_dtype=torch_dtype
)

schedule_template = KarrasScheduleTemplate.Mastering
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
).to(sampling_dtype)
sigmas_quantized = torch.cat([
  quantize_to(sigmas[:-1], unet_k_wrapped.sigmas),
  zeros((1), device=sigmas.device, dtype=sigmas.dtype)
])
print(f"sigmas (quantized):\n{', '.join(['%.4f' % s.item() for s in sigmas_quantized])}")

sample_path='out'
intermediates_path='intermediates'
for path_ in [sample_path, intermediates_path]:
  os.makedirs(path_, exist_ok=True)
log_intermediates: LogIntermediates = make_log_intermediates(intermediates_path)

cond_scale = 7.5 if cfg_enabled else 1.
if model_name == 'hakurei/waifu-diffusion':
  # WD1.4 was trained on area=640**2 and no side longer than 768
  height = 768
  width = 640**2//height
else:
  width = 768 if is_768 else 512
  height = width

latent_scale_factor = 8
latents_shape = LatentsShape(unet.in_channels, height // latent_scale_factor, width // latent_scale_factor)

latent_strategies: List[MakeLatentsStrategy] = [
  SeedLatentMaker(latents_shape, dtype=torch.float32, device=device).make_latents,
]

latent_maker = LatentMaker(
  strategies=latent_strategies,
)

batch_latent_maker = BatchLatentMaker(
  latent_maker.make_latents,
)

# TODO: make larger batches work
max_batch_size = 1
n_rand_seeds = 1
seeds: Iterable[int] = chain(
  (
    2178792736,
  ),
  (get_seed() for _ in range(n_rand_seeds))
)

prompt='artoria pendragon (fate), carnelian, 1girl, general content, upper body, white shirt, blonde hair, looking at viewer, medium breasts, hair between eyes, floating hair, green eyes, blue ribbon, long sleeves, light smile, hair ribbon, watercolor (medium), traditional media'
conditions: Iterable[ConditionSpec] = repeat(SingleCondition(cfg_scale=7.5, prompt=prompt))
sample_specs: Iterable[SampleSpec] = (SampleSpec(
  latent_spec=SeedSpec(seed),
  cond_spec=cond,
) for seed, cond in zip(seeds, conditions))

batcher = ExecutionPlanBatcher[SampleSpec, ExecutionPlan](
  max_batch_size=max_batch_size,
  make_execution_plan=make_execution_plan,
)
batch_generator: Generator[BatchSpecGeneric[ExecutionPlan], None, None] = batcher.generate(sample_specs)

consistent_batch_size=None

sum_of_batch_times=0
initial_batch_time=0
sample_count=0

with no_grad():
  batch_tic = time.perf_counter()
  for batch_ix, (plan, specs) in enumerate(batch_generator):
    seeds: List[int] = list(map(lambda spec: spec.latent_spec.seed, specs))
    latents: FloatTensor = batch_latent_maker.make_latents(map(lambda spec: spec.latent_spec, specs))
    embedding_and_mask: EmbeddingAndMask = embed(plan.prompts)
    embedding, mask = embedding_and_mask
    if plan.cfg_enabled:
      uc, c = embedding.split((1, embedding.size(0)-1))
      uc_mask, c_mask = mask.split((1, mask.size(0)-1))
      # SD was trained loads on an unmasked uc, so undo uc's masking
      uc_mask = (torch.arange(uc_mask.size(1), device=device) < 77).unsqueeze(0)
      mask = torch.cat([uc_mask, c_mask])
    else:
      uc, c = None, embedding
      uc_mask, c_mask = None, mask

    # TODO: support varied CFG scale within a batch
    denoiser: Denoiser = denoiser_factory(uncond=uc, cond=c, cond_scale=specs[0].cond_spec.cfg_scale, attention_mask=mask)
    noise_sampler = BrownianTreeNoiseSampler(
      latents,
      sigma_min=sigma_min,
      sigma_max=sigma_max,
      # there's no requirement that the noise sampler's seed be coupled to the init noise seed;
      # I'm just re-using it because it's a convenient arbitrary number
      seed=seeds[0],
    )

    tic = time.perf_counter()
    latents: Tensor = sample_dpmpp_2m(
      denoiser,
      latents * sigmas[0],
      sigmas,
      # noise_sampler=noise_sampler, # you can only pass noise sampler to ancestral samplers
      # callback=log_intermediates,
    ).to(vae_dtype)
    pil_images: List[Image.Image] = latents_to_pils(latents)

    sample_time=time.perf_counter()-tic
    sum_of_batch_times += sample_time
    sample_count += len(specs)
    if batch_ix == 0:
      # account for first sample separately because warmup can be an outlier
      initial_batch_time = sample_time
      # if all 
      consistent_batch_size = len(specs)
    else:
      consistent_batch_size = consistent_batch_size if len(specs) == consistent_batch_size else None

    base_count = len(os.listdir(sample_path))
    for ix, (seed, image) in enumerate(zip(seeds, pil_images)):
      image.save(os.path.join(sample_path, f"{base_count+ix:05}.{seed}.png"))

total_time=time.perf_counter()-batch_tic

batch_count = len(seeds)

perf_message = f'in total, generated {batch_count} batches'
if consistent_batch_size is not None:
  perf_message += f' of {consistent_batch_size} images'
perf_message += ' in (secs):\n'

perf_message += f'Embed + Unet + sampling + VAE + RGB-to-PIL + PIL-to-disk:\n  {total_time:.2f}'
perf_message += f' (avg {total_time/sample_count:.2f}/sample)'
if consistent_batch_size and batch_count>1:
  perf_message += f' (avg {total_time/batch_count:.2f}/batch)'
perf_message += '\n'

perf_message += f'Unet + sampling + VAE + RGB-to-PIL:\n  {sum_of_batch_times:.2f}'
perf_message += f' (avg {sum_of_batch_times/sample_count:.2f}/sample)'
if consistent_batch_size and batch_count>1:
  perf_message += f' (avg {sum_of_batch_times/batch_count:.2f}/batch)'
perf_message += '\n'

if batch_count>1:
  excl_warmup_time=sum_of_batch_times-initial_batch_time
  perf_message += f'Unet + sampling + VAE + RGB-to-PIL (excl. warmup batch):\n  {excl_warmup_time:.2f}'
  if consistent_batch_size:
    perf_message += f' (avg {excl_warmup_time/sample_count:.2f}/sample)'
  if batch_count>2:
    perf_message += f' (avg {excl_warmup_time/batch_count:.2f}/batch)'

print(perf_message)