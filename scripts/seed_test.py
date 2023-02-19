from helpers.inference_spec.sample_spec import SampleSpec
from helpers.inference_spec.latent_spec import SeedSpec
from helpers.inference_spec.latents_shape import LatentsShape
from helpers.inference_spec.cond_spec import SingleCondition, MultiCond, WeightedPrompt, CFG, Prompt
from helpers.inference_spec.execution_plan_batcher import ExecutionPlanBatcher, BatchSpecGeneric
from helpers.inference_spec.execution_plan import ExecutionPlan, make_execution_plan
from helpers.inference_spec.batch_latent_maker import BatchLatentMaker
from helpers.inference_spec.latent_maker import LatentMaker, MakeLatentsStrategy
from helpers.inference_spec.latent_maker_seed_strategy import SeedLatentMaker
from helpers.sample_interpolation.make_in_between import make_inbetween
from helpers.sample_interpolation.intersperse_linspace import intersperse_linspace
from helpers.get_seed import get_seed
from helpers.device import DeviceLiteral, get_device_type
from helpers.embed_text_types import Prompts, EmbeddingAndMask
import torch
from torch import FloatTensor, BoolTensor
from typing import Iterable, Generator, List, Optional
from itertools import chain, repeat

cond_keyframes: List[SingleCondition|MultiCond] = [SingleCondition(
  cfg=CFG(scale=7.5, uncond_prompt=Prompt(text='')),
  prompt=Prompt(text='hello'),
), MultiCond(
  cfg=CFG(scale=7.5, uncond_prompt=Prompt(text='')),
  weighted_cond_prompts=[WeightedPrompt(
    prompt=Prompt(text='man'),
    weight=0.5,
  ), WeightedPrompt(
    prompt=Prompt(text='bear'),
    weight=0.5,
  ), WeightedPrompt(
    prompt=Prompt(text='pig'),
    weight=0.5,
  )]
)]

cond_linspace: List[SingleCondition|MultiCond] = intersperse_linspace(
  keyframes=cond_keyframes,
  make_inbetween=make_inbetween,
  steps=3
)

device_type: DeviceLiteral = get_device_type()
device = torch.device(device_type)

def embed(prompts: Prompts) -> EmbeddingAndMask:
  batch_size=len(prompts)
  # token_length = 77
  # dims = 768
  token_length = 2
  dims = 3
  return EmbeddingAndMask(
    embedding=torch.arange(batch_size, dtype=torch.float32, device=device).unsqueeze(-1).unsqueeze(-1).expand(-1, token_length, dims),
    attn_mask=torch.ones(batch_size, token_length, dtype=torch.bool, device=device),
  )

height = 768
width = 640**2//height

unet_channels = 4
latent_scale_factor = 8

latents_shape = LatentsShape(unet_channels, height // latent_scale_factor, width // latent_scale_factor)

latent_strategies: List[MakeLatentsStrategy] = [
  SeedLatentMaker(latents_shape, dtype=torch.float32, device=device).make_latents,
]

latent_maker = LatentMaker(
  strategies=latent_strategies,
)

batch_latent_maker = BatchLatentMaker(
  latent_maker.make_latents,
)

n_rand_seeds=2
seeds: Iterable[int] = chain(
  repeat(1, 3),
  (
    2,
    3,
    4,
  ),
  (get_seed() for _ in range(n_rand_seeds))
)
prompt='artoria pendragon (fate), carnelian, 1girl, general content, upper body, white shirt, blonde hair, looking at viewer, medium breasts, hair between eyes, floating hair, green eyes, blue ribbon, long sleeves, light smile, hair ribbon, watercolor (medium), traditional media'
# conditions: Iterable[ConditionSpec] = repeat(SingleCondition(cfg_scale=7.5, prompt=prompt))
# conditions: Iterable[ConditionSpec] = MultiCond
sample_specs: Iterable[SampleSpec] = (SampleSpec(
  latent_spec=SeedSpec(seed),
  cond_spec=cond,
) for seed, cond in zip(seeds, cond_linspace))

batcher = ExecutionPlanBatcher[SampleSpec, ExecutionPlan](
  max_batch_size=3,
  make_execution_plan=make_execution_plan,
)
batch_generator: Generator[BatchSpecGeneric[ExecutionPlan], None, None] = batcher.generate(sample_specs)

for batch_ix, (plan, specs) in enumerate(batch_generator):
  # explicit type cast to help IDE infer type
  plan: ExecutionPlan = plan
  specs: List[SampleSpec] = specs
  seeds: List[Optional[int]] = list(map(lambda spec: spec.latent_spec.seed if isinstance(spec.latent_spec, SeedSpec) else None, specs))
  latents: FloatTensor = batch_latent_maker.make_latents(map(lambda spec: spec.latent_spec, specs))
  embedding_and_mask: EmbeddingAndMask = embed(plan.prompt_texts_ordered)
  embedding, mask = embedding_and_mask

  embed_instance_ixs_flat: List[int] = [ix for sample_ixs in plan.prompt_text_instance_ixs for ix in sample_ixs]

  # denormalize
  embedding: FloatTensor = embedding.index_select(0, torch.tensor(embed_instance_ixs_flat, device=device))
  mask: BoolTensor = mask.index_select(0, torch.tensor(embed_instance_ixs_flat, device=device))
