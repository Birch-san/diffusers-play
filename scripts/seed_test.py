from helpers.inference_spec.sample_spec import SampleSpec
from helpers.inference_spec.latent_spec import SeedSpec
from helpers.inference_spec.latents_shape import LatentsShape
from helpers.inference_spec.cond_spec import ConditionSpec, SingleCondition, MultiCond, WeightedPrompt
from helpers.inference_spec.execution_plan_batcher import ExecutionPlanBatcher, BatchSpecGeneric
from helpers.inference_spec.execution_plan import ExecutionPlan, make_execution_plan
from helpers.inference_spec.batch_latent_maker import BatchLatentMaker
from helpers.inference_spec.latent_maker import LatentMaker, MakeLatentsStrategy
from helpers.inference_spec.latent_maker_seed_strategy import SeedLatentMaker
from helpers.inference_spec.spec_dependence_checker import SpecDependenceChecker, CheckSpecDependenceStrategy
from helpers.inference_spec.feedback_spec_dependence_strategy import has_feedback_dependence
from helpers.sample_interpolation.in_between import MakeInbetween, InBetweenParams
from helpers.sample_interpolation.intersperse_linspace import intersperse_linspace
from helpers.get_seed import get_seed
from helpers.device import DeviceLiteral, get_device_type
from helpers.embed_text_types import Prompts, EmbeddingAndMask
import torch
from torch import FloatTensor
from typing import Iterable, Generator, List, Callable, Tuple, Protocol
from itertools import chain, repeat


# make_inbetween: MakeInbetween[SingleCondition|MultiCond, MultiCond] = lambda params: MultiCond(
#   cfg_scale=params.from_.cfg_scale,
#   weighted_prompts=[*params.from_.weighted_prompts, *params.to.weighted_prompts]
# )
class AdjustScale(Protocol):
  @staticmethod
  def __call__(scale_nominal: float, quotient: float) -> float: ...

WeightedPromptAndScaleAdjuster = Tuple[WeightedPrompt, AdjustScale]

def make_inbetween(params: InBetweenParams[SingleCondition|MultiCond]) -> MultiCond:
  assert params.from_.cfg_enabled == params.to.cfg_enabled
  cfg_scale_coeff = params.to.cfg_scale / params.from_.cfg_scale
  scale_from: AdjustScale = lambda scale_nominal, quotient: scale_nominal * (1-quotient)
  scale_to: AdjustScale = lambda scale_nominal, quotient: scale_nominal * quotient * cfg_scale_coeff
  prompts_and_scale_strategies: Iterable[WeightedPromptAndScaleAdjuster] = chain(
    zip(params.from_.weighted_prompts, repeat(scale_from)),
    zip(params.to.weighted_prompts, repeat(scale_to))
  )
  return MultiCond(
    cfg_scale=params.from_.cfg_scale,
    weighted_prompts=[
      WeightedPrompt(
        wp.prompt,
        scale(wp.weight, params.quotient)
      ) for wp, scale in prompts_and_scale_strategies
    ]
  )

cond_keyframes: List[SingleCondition|MultiCond] = [SingleCondition(
  cfg_scale=7.5,
  prompt='hello',
), MultiCond(
  cfg_scale=7.5,
  weighted_prompts=[WeightedPrompt(
    prompt='man',
    weight=0.5,
  ), WeightedPrompt(
    prompt='bear',
    weight=0.5,
  ), WeightedPrompt(
    prompt='pig',
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
  return EmbeddingAndMask(
    embedding=torch.ones(len(prompts), 77, 768, dtype=torch.float32, device=device),
    attn_mask=torch.ones(len(prompts), 77, dtype=torch.bool, device=device),
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

dependence_strategies: List[CheckSpecDependenceStrategy[SampleSpec]] = [
  has_feedback_dependence,
]
spec_dependence_checker=SpecDependenceChecker[SampleSpec](
  strategies=dependence_strategies,
)

batcher = ExecutionPlanBatcher[SampleSpec, ExecutionPlan](
  max_batch_size=3,
  make_execution_plan=make_execution_plan,
  depends_on_prev_sample=spec_dependence_checker.has_dependence,
)
batch_generator: Generator[BatchSpecGeneric[ExecutionPlan], None, None] = batcher.generate(sample_specs)

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
  pass