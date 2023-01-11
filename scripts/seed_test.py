from dataclasses import dataclass
from helpers.inference_spec.batch_spec_factory import LatentBatcher, latents_from_seed_factory, MakeLatents, LatentsShape, SampleSpecBatcher, MakeLatentBatches, BatchSpecX
from helpers.get_seed import get_seed
from helpers.device import DeviceLiteral, get_device_type
import torch
from torch import FloatTensor
from typing import Iterable, Generator, List

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

def make_latent_batches(batch_size: int, specs: Iterable[SampleSpec]) -> Iterable[FloatTensor]:
  seeds: Iterable[int] = map(lambda spec: spec.seed, specs)
  batcher = LatentBatcher(
    batch_size=batch_size,
    specs=seeds,
    make_latents=make_latents,
  )
  generator: Generator[FloatTensor, None, None] = batcher.generate()
  return generator

sample_spec_batcher = SampleSpecBatcher(
  batch_size=3,
  make_latent_batches=make_latent_batches,
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
generator: Generator[BatchSpecX, None, None] = sample_spec_batcher.generate(sample_specs)

for latents, in generator:
  print(latents)