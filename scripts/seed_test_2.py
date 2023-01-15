from helpers.inference_spec.sample_spec_2 import SampleSpec
from helpers.inference_spec.latent_spec import SeedSpec
from helpers.inference_spec.latents_shape import LatentsShape
from helpers.inference_spec.cond_spec import ConditionSpec, SingleCondition
from helpers.inference_spec.execution_plan_batcher_2 import ExecutionPlanBatcher, BatchSpecGeneric
from helpers.inference_spec.execution_plan import ExecutionPlan, make_execution_plan
from helpers.inference_spec.batch_latent_maker import BatchLatentMaker
from helpers.inference_spec.latent_maker import LatentMaker, MakeLatentsStrategy
from helpers.inference_spec.latent_maker_seed_strategy import SeedLatentMaker
from helpers.get_seed import get_seed
from helpers.device import DeviceLiteral, get_device_type
from helpers.embed_text_types import Prompts, EmbeddingAndMask
import torch
from torch import FloatTensor
from typing import Iterable, Generator, Optional, List
from itertools import chain, repeat

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
conditions: Iterable[ConditionSpec] = repeat(SingleCondition(cfg_scale=7.5, prompt=prompt))
sample_specs: Iterable[SampleSpec] = (SampleSpec(
  latent_spec=SeedSpec(seed),
  cond_spec=cond,
) for seed, cond in zip(seeds, conditions))

batcher = ExecutionPlanBatcher[SampleSpec, ExecutionPlan](
  max_batch_size=3,
  make_execution_plan=make_execution_plan,
)
batch_generator: Generator[BatchSpecGeneric[ExecutionPlan], None, None] = batcher.generate(sample_specs)

carry: Optional[FloatTensor] = None
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