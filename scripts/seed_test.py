from dataclasses import dataclass
from helpers.inference_spec.sample_spec_batcher import SampleSpecBatcher, BatchSpecGeneric
from helpers.inference_spec.latents_from_seed import latents_from_seed_factory, MakeLatents, make_latent_batches, LatentsShape, GetSeedFromSpec
from helpers.inference_spec.map_spec_chunks import map_spec_chunks
from helpers.inference_spec.cond_spec import ConditionSpec, SingleCondition
from helpers.inference_spec.cond_batcher import MakeConds
from helpers.inference_spec.conds_from_prompts import make_cond_batches, conds_from_prompts_factory, GetPromptsFromSpec
from helpers.embed_text_types import Prompts, EmbeddingAndMask
from helpers.get_seed import get_seed
from helpers.device import DeviceLiteral, get_device_type
import torch
from typing import Iterable, Generator
from functools import partial
from itertools import chain, repeat

device_type: DeviceLiteral = get_device_type()
device = torch.device(device_type)

height = 768
width = 640**2//height

unet_channels = 4
latent_scale_factor = 8

latents_shape = LatentsShape(unet_channels, height // latent_scale_factor, width // latent_scale_factor)
make_latents: MakeLatents[int] = latents_from_seed_factory(latents_shape, dtype=torch.float32, device=device)

# mock embed function so we don't have to load CLIP
def embed(prompts: Prompts) -> EmbeddingAndMask:
  return EmbeddingAndMask(
    embedding=torch.ones(1, 77, 768, dtype=torch.float32, device=device),
    attn_mask=torch.ones(1, 77, 768, dtype=torch.bool, device=device),
  )
make_conds: MakeConds[Prompts] = conds_from_prompts_factory(embed)

@dataclass
class SampleSpec:
  seed: int
  cond_spec: ConditionSpec

def get_prompts(cond_spec: ConditionSpec) -> Prompts:
  if cond_spec.cfg_scale == 1.0:
    return cond_spec.get_prompts()
  return ['', cond_spec.get_prompts()]

get_seed_from_spec: GetSeedFromSpec = lambda spec: spec.seed
get_prompts_from_spec: GetPromptsFromSpec[SampleSpec] = lambda spec: get_prompts(spec.cond_spec)

sample_spec_batcher = SampleSpecBatcher(
  batch_size=3,
  make_latent_batches=lambda spec_chunks: make_latent_batches(make_latents, map_spec_chunks(get_seed_from_spec, spec_chunks)),
  make_cond_batches=lambda spec_chunks: make_cond_batches(make_conds, map_spec_chunks(get_prompts_from_spec, spec_chunks)),
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
  seed=seed,
  cond_spec=cond,
) for seed, cond in zip(seeds, conditions))
batch_spec_generator: Generator[BatchSpecGeneric[SampleSpec], None, None] = sample_spec_batcher.generate(sample_specs)

for spec_chunk, latents, conds in batch_spec_generator:
  # print(spec_chunk)
  # print(latents)
  # print(conds)
  pass