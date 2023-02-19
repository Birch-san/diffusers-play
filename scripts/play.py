import os

# monkey-patch _randn to use CPU random before k-diffusion uses it
from helpers.brownian_tree_mps_fix import reassuring_message
from helpers.cumsum_mps_fix import reassuring_message as reassuring_message_2
from helpers.device import DeviceLiteral, get_device_type
from helpers.diffusers_denoiser import DiffusersSDDenoiser, DiffusersSD2Denoiser
from helpers.batch_denoiser import Denoiser, BatchDenoiserFactory
from helpers.log_intermediates import LogIntermediates, make_log_intermediates
from helpers.schedules import KarrasScheduleParams, KarrasScheduleTemplate, get_template_schedule
print(reassuring_message) # avoid "unused" import :P
print(reassuring_message_2)

import torch
from torch import Tensor, FloatTensor, BoolTensor, LongTensor, no_grad, zeros, tensor, arange
from diffusers.models import UNet2DConditionModel, AutoencoderKL
from diffusers.utils.import_utils import is_xformers_available
from k_diffusion.sampling import BrownianTreeNoiseSampler, get_sigmas_karras, sample_dpmpp_2m

from helpers.attention.mode import AttentionMode
from helpers.attention.multi_head_attention.to_mha import to_mha
from helpers.attention.set_chunked_attn import make_set_chunked_attn
from helpers.attention.tap_attn import TapAttn, tap_attn_to_tap_module
from helpers.attention.replace_attn import replace_attn_to_tap_module
from helpers.tap.tap_module import TapModule
from helpers.schedule_params import get_alphas, get_alphas_cumprod, get_betas, quantize_to
from helpers.get_seed import get_seed
from helpers.latents_to_pils import LatentsToPils, LatentsToBCHW, make_latents_to_pils, make_latents_to_bchw
from helpers.embed_text_types import Embed, EmbeddingAndMask
from helpers.embed_text import ClipCheckpoint, ClipImplementation, get_embedder
from helpers.model_db import get_model_needs, ModelNeeds
from helpers.inference_spec.sample_spec import SampleSpec
from helpers.inference_spec.latent_spec import SeedSpec
from helpers.inference_spec.latents_shape import LatentsShape
from helpers.inference_spec.cond_spec import ConditionSpec, SingleCondition, MultiCond, WeightedPrompt, CFG, Prompt
from helpers.inference_spec.execution_plan_batcher import ExecutionPlanBatcher, BatchSpecGeneric
from helpers.inference_spec.execution_plan import ExecutionPlan, make_execution_plan
from helpers.inference_spec.batch_latent_maker import BatchLatentMaker
from helpers.inference_spec.latent_maker import LatentMaker, MakeLatentsStrategy
from helpers.inference_spec.latent_maker_seed_strategy import SeedLatentMaker
from helpers.sample_interpolation.make_in_between import make_inbetween
from helpers.sample_interpolation.intersperse_linspace import intersperse_linspace
from itertools import chain, repeat

from typing import List, Generator, Iterable, Optional
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
  # 'hakurei/waifu-diffusion'
  'waifu-diffusion/wd-1-5-beta'
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

match model_name:
  # WD 1.4 and 1.5 haven't uploaded fp16 revision yet
  case 'hakurei/waifu-diffusion' | 'waifu-diffusion/wd-1-5-beta':
    revision = None
unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
  model_name,
  subfolder='unet',
  revision=revision,
  torch_dtype=torch_dtype,
  upcast_attention=upcast_attention,
).to(device).eval()

attn_mode = AttentionMode.TorchMultiheadAttention
match(attn_mode):
  case AttentionMode.Standard: pass
  case AttentionMode.Chunked:
    set_chunked_attn: TapAttn = make_set_chunked_attn(
      query_chunk_size = 1024,
      kv_chunk_size = None,
    )
    tap_module: TapModule = tap_attn_to_tap_module(set_chunked_attn)
    unet.apply(tap_module)
  case AttentionMode.TorchMultiheadAttention:
    tap_module: TapModule = replace_attn_to_tap_module(to_mha)
    unet.apply(tap_module)
  case AttentionMode.Xformers:
    assert is_xformers_available()
    unet.enable_xformers_memory_efficient_attention()

# sampling in higher-precision helps to converge more stably toward the "true" image (not necessarily better-looking though)
sampling_dtype: torch.dtype = torch.float32
# sampling_dtype: torch.dtype = torch_dtype
alphas_cumprod: Tensor = get_alphas_cumprod(get_alphas(get_betas(device=device))).to(dtype=sampling_dtype)
unet_k_wrapped = DiffusersSD2Denoiser(unet, alphas_cumprod, sampling_dtype) if needs_vparam else DiffusersSDDenoiser(unet, alphas_cumprod, sampling_dtype)
denoiser_factory = BatchDenoiserFactory(unet_k_wrapped)

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

schedule_template = KarrasScheduleTemplate.CudaMastering
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
match(model_name):
  case 'hakurei/waifu-diffusion':
    # WD1.4 was trained on area=640**2 and no side longer than 768
    height = 768
    width = 640**2//height
  case 'waifu-diffusion/wd-1-5-beta':
    # WD1.5 was trained on area=896**2 and no side longer than 1152
    sqrt_area=896
    # >1 = portrait
    # aspect_ratio = 1.357
    aspect_ratio = 1.15
    height = int(sqrt_area*aspect_ratio)
    width = sqrt_area**2//height
  case _:
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

max_batch_size = 8
n_rand_seeds = 0
seeds: Iterable[int] = chain(
  repeat(2178792736),
  (get_seed() for _ in range(n_rand_seeds))
)

uncond_prompt = Prompt(text='lowres, bad anatomy, bad hands, blurry, mutation, deformed face, ugly, bad proportions, monster, real life, realistic, instagram, worst quality, jpeg, bad posture, long body, long neck, jpeg artifacts, deleted, bad aesthetic')

cond_keyframes: List[SingleCondition|MultiCond] = [
  SingleCondition(
    cfg=CFG(scale=7.5, uncond_prompt=uncond_prompt),
    # flandre scarlet = 3 tokens
    # touhou = 2 tokens
    prompt=Prompt(text='beautiful, 1girl, flandre scarlet touhou, detailed hair, portrait, floating hair, waifu, anime, best aesthetic, best quality, ribbon, outdoors, good posture, marker (medium), colored pencil (medium), reddizen'),
  ),
  SingleCondition(
    cfg=CFG(scale=7.5, uncond_prompt=uncond_prompt),
    # kirisame marisa = 3 tokens
    # touhou = 2 tokens
    prompt=Prompt(text='beautiful, 1girl, kirisame marisa touhou, detailed hair, portrait, floating hair, waifu, anime, best aesthetic, best quality, ribbon, outdoors, good posture, marker (medium), colored pencil (medium), reddizen'),
  ),
  SingleCondition(
    cfg=CFG(scale=7.5, uncond_prompt=uncond_prompt),
    # hakurei reimu = 5 tokens
    prompt=Prompt(text='beautiful, 1girl, hakurei reimu, detailed hair, portrait, floating hair, waifu, anime, best aesthetic, best quality, ribbon, outdoors, good posture, marker (medium), colored pencil (medium), reddizen'),
  ),
  SingleCondition(
    cfg=CFG(scale=7.5, uncond_prompt=uncond_prompt),
    # konpaku youmu = 5 tokens
    prompt=Prompt(text='beautiful, 1girl, konpaku youmu, detailed hair, portrait, floating hair, waifu, anime, best aesthetic, best quality, ribbon, outdoors, good posture, marker (medium), colored pencil (medium), reddizen'),
  ),
  SingleCondition(
    cfg=CFG(scale=7.5, uncond_prompt=uncond_prompt),
    # flandre scarlet = 3 tokens
    # touhou = 2 tokens
    prompt=Prompt(text='beautiful, 1girl, flandre scarlet touhou, detailed hair, portrait, floating hair, waifu, anime, best aesthetic, best quality, ribbon, outdoors, good posture, marker (medium), colored pencil (medium), reddizen'),
  ),
  # MultiCond(
  #   cfg=CFG(scale=7.5, uncond_prompt=Prompt(text='')),
  #   weighted_cond_prompts=[WeightedPrompt(
  #     prompt=Prompt(text='man'),
  #     weight=0.5,
  #   ), WeightedPrompt(
  #     prompt=Prompt(text='bear'),
  #     weight=0.5,
  #   ), WeightedPrompt(
  #     prompt=Prompt(text='pig'),
  #     weight=0.5,
  #   )]
  # )
]

conditions: List[SingleCondition|MultiCond] = intersperse_linspace(
  keyframes=cond_keyframes,
  make_inbetween=make_inbetween,
  steps=8,
)

# prompt='artoria pendragon (fate), carnelian, 1girl, general content, upper body, white shirt, blonde hair, looking at viewer, medium breasts, hair between eyes, floating hair, green eyes, blue ribbon, long sleeves, light smile, hair ribbon, watercolor (medium), traditional media'
# conditions: Iterable[ConditionSpec] = repeat(SingleCondition(
#   cfg=CFG(scale=7.5, uncond_prompt=Prompt(text='')),
#   prompt=Prompt(text=prompt),
# ))
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
batch_count=0

with no_grad():
  batch_tic = time.perf_counter()
  for batch_ix, (plan, specs) in enumerate(batch_generator):
    # explicit type cast to help IDE infer type
    plan: ExecutionPlan = plan
    specs: List[SampleSpec] = specs

    batch_count += 1
    batch_sample_count = len(specs)
    seeds: List[Optional[int]] = list(map(lambda spec: spec.latent_spec.seed if isinstance(spec.latent_spec, SeedSpec) else None, specs))
    latents: FloatTensor = batch_latent_maker.make_latents(map(lambda spec: spec.latent_spec, specs))
    del specs
    embedding_and_mask: EmbeddingAndMask = embed(plan.prompt_texts_ordered)
    embedding, mask = embedding_and_mask
    del embedding_and_mask

    embed_instance_ixs_flat: List[int] = [ix for sample_ixs in plan.prompt_text_instance_ixs for ix in sample_ixs]
    # denormalize
    embedding: FloatTensor = embedding.index_select(0, tensor(embed_instance_ixs_flat, device=device))
    mask: BoolTensor = mask.index_select(0, tensor(embed_instance_ixs_flat, device=device))

    # per sample: quantity of conditions upon which it should be denoised
    conds_per_prompt: List[int] = [len(sample_ixs) for sample_ixs in plan.prompt_text_instance_ixs]
    conds_per_prompt_tensor: LongTensor = tensor(conds_per_prompt, dtype=torch.long, device=device)
    if plan.cfg is None:
      uncond_ixs = None
      cfg_scales = None
    else:
      first_cond_ix_per_prompt: LongTensor = conds_per_prompt_tensor.roll(1).index_put(
        indices=[torch.zeros([1], dtype=torch.long, device=device)],
        values=torch.zeros([1], dtype=torch.long, device=device),
      ).cumsum(0)
      uncond_ixs: LongTensor = first_cond_ix_per_prompt + tensor(plan.cfg.uncond_instance_ixs, dtype=torch.long, device=device)
      # SD was trained loads on unmasked empty-string uncond, so undo uc mask on any empty-string uncond
      empty_string_uncond_ixs: LongTensor = uncond_ixs.masked_select(tensor(plan.cfg.has_empty_string_uncond, dtype=torch.bool, device=device))
      unmasked_clip_segment: BoolTensor = arange(mask.size(1), device=device) < 77
      mask: BoolTensor = mask.index_put(
        indices=[empty_string_uncond_ixs],
        values=unmasked_clip_segment,
      )
      del empty_string_uncond_ixs, unmasked_clip_segment
      cfg_scales: FloatTensor = tensor(plan.cfg.scales, dtype=sampling_dtype, device=device)
    cond_weights: FloatTensor = tensor(plan.cond_weights, dtype=sampling_dtype, device=device)
    del plan
    
    match(attn_mode):
      # xformers attn_bias is only implemented for Triton + A100 GPU
      # https://github.com/facebookresearch/xformers/issues/576
      # chunked attention *can* be made to support masks, but I didn't implement it yet
      case AttentionMode.Xformers | AttentionMode.Chunked:
        mask = None

    denoiser: Denoiser = denoiser_factory(
      cross_attention_conds=embedding,
      cross_attention_mask=mask,
      conds_per_prompt=conds_per_prompt_tensor,
      cond_weights=cond_weights,
      uncond_ixs=uncond_ixs,
      cfg_scales=cfg_scales,
    )
    del embedding, mask, conds_per_prompt_tensor, cond_weights, uncond_ixs, cfg_scales

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
    del denoiser
    pil_images: List[Image.Image] = latents_to_pils(latents)
    del latents

    sample_time=time.perf_counter()-tic
    sum_of_batch_times += sample_time
    sample_count += batch_sample_count
    if batch_ix == 0:
      # account for first sample separately because warmup can be an outlier
      initial_batch_time = sample_time
      consistent_batch_size = batch_sample_count
    else:
      consistent_batch_size = consistent_batch_size if batch_sample_count == consistent_batch_size else None

    base_count = len(os.listdir(sample_path))
    for ix, (seed, image) in enumerate(zip(seeds, pil_images)):
      image.save(os.path.join(sample_path, f"{base_count+ix:05}.{seed}.png"))
    del pil_images
    if device.type == 'cuda':
      torch.cuda.empty_cache()

total_time=time.perf_counter()-batch_tic

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