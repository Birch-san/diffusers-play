import torch
from torch import Generator as TorchGenerator, FloatTensor, randn
from typing import NamedTuple, Iterable, Tuple, Generator
from ..device import DeviceType
from .latent_batcher import MakeLatents, LatentBatcher, LatentBatcherOutput

class LatentsShape(NamedTuple):
  channels: int
  height: int
  width: int

def latents_from_seed_factory(
  shape: LatentsShape,
  dtype: torch.dtype = torch.float32,
  device: DeviceType = torch.device('cpu')
) -> MakeLatents[int]:
  generator = TorchGenerator(device='cpu')
  def make_latents(seed: int) -> FloatTensor:
    generator.manual_seed(seed)
    latents: FloatTensor = randn((1, *shape), generator=generator, device='cpu', dtype=dtype).to(device)
    return latents
  return make_latents

def make_latent_batches(
  make_latents: MakeLatents[int],
  seed_chunks: Iterable[Tuple[int, ...]],
) -> Iterable[LatentBatcherOutput]:
  batcher = LatentBatcher(
    make_latents=make_latents,
  )
  generator: Generator[LatentBatcherOutput, None, None] = batcher.generate(seed_chunks)
  return generator