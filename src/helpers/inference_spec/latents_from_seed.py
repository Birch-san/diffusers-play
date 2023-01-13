import torch
from torch import Generator as TorchGenerator, FloatTensor, randn
from typing import NamedTuple, Iterable, Tuple, Generator, TypeVar, Protocol, Generic
from ..device import DeviceType
from .latent_batcher import MakeLatents, LatentBatcher

SampleSpec = TypeVar('SampleSpec')

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

class GetSeedFromSpec(Protocol, Generic[SampleSpec]):
  @staticmethod
  def __call__(spec: SampleSpec) -> int: ...

def make_latent_batches(
  make_latents: MakeLatents[int],
  get_seed_from_spec: GetSeedFromSpec,
  spec_chunks: Iterable[Tuple[SampleSpec, ...]],
) -> Iterable[FloatTensor]:
  seed_chunks: Iterable[Tuple[int, ...]] = map(lambda chunk: tuple(map(get_seed_from_spec, chunk)), spec_chunks)
  batcher = LatentBatcher(
    make_latents=make_latents,
  )
  generator: Generator[FloatTensor, None, None] = batcher.generate(seed_chunks)
  return generator