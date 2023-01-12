from dataclasses import dataclass
from helpers.inference_spec.sample_spec_batcher import SampleSpecBatcher, BatchSpecGeneric
from helpers.inference_spec.latents_from_seed import latents_from_seed_factory, MakeLatents, make_latent_batches, LatentsShape
from helpers.get_seed import get_seed
from helpers.device import DeviceLiteral, get_device_type
import torch
from typing import Iterable, Generator, List
from functools import partial

device_type: DeviceLiteral = get_device_type()
device = torch.device(device_type)

height = 768
width = 640**2//height

unet_channels = 4
latent_scale_factor = 8

latents_shape = LatentsShape(unet_channels, height // latent_scale_factor, width // latent_scale_factor)
make_latents: MakeLatents[int] = latents_from_seed_factory(latents_shape, dtype=torch.float32, device=device)

@dataclass
class SampleSpec:
  seed: int

sample_spec_batcher = SampleSpecBatcher(
  batch_size=3,
  make_latent_batches=partial(make_latent_batches, make_latents),
)
n_rand_seeds=2
seeds: List[int] = [
  *(1,)*3,
  2,
  3,
  4,
  *[get_seed() for _ in range(n_rand_seeds)]
]
sample_specs: Iterable[SampleSpec] = map(lambda seed: SampleSpec(seed=seed), seeds)
generator: Generator[BatchSpecGeneric[SampleSpec], None, None] = sample_spec_batcher.generate(sample_specs)

for spec_chunk, latents in generator:
  print(spec_chunk)
  print(latents)