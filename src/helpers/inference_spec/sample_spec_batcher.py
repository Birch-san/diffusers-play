from typing import Protocol, Generator, Iterable, NamedTuple, Generic, TypeVar, Tuple
from torch import FloatTensor
from itertools import tee
from ..iteration.chunk import chunk

SampleSpec = TypeVar('SampleSpec')

class MakeLatentBatches(Protocol, Generic[SampleSpec]):
  @staticmethod
  def __call__(spec_chunks: Iterable[Tuple[SampleSpec, ...]]) -> Iterable[FloatTensor]: ...

class BatchSpec(NamedTuple):
  spec_chunk: Tuple[SampleSpec, ...]
  latents: FloatTensor
class BatchSpecGeneric(BatchSpec, Generic[SampleSpec]): pass

class SampleSpecBatcher(Generic[SampleSpec]):
  make_latent_batches: MakeLatentBatches[SampleSpec]
  def __init__(
    self,
    batch_size: int,
    make_latent_batches: MakeLatentBatches[SampleSpec],
  ) -> None:
    self.batch_size = batch_size
    self.make_latent_batches = make_latent_batches
  
  def generate(
    self,
    specs: Iterable[SampleSpec],
  ) -> Generator[BatchSpecGeneric[SampleSpec], None, None]:
    spec_chunks: Iterable[Tuple[SampleSpec, ...]] = chunk(specs, self.batch_size)
    batcher_it, latent_it = tee(spec_chunks, 2)
    latent_batches: Iterable[FloatTensor] = self.make_latent_batches(latent_it)
    for spec_chunk, latents in zip(batcher_it, latent_batches):
      yield BatchSpecGeneric[SampleSpec](
        spec_chunk=spec_chunk,
        latents=latents,
      )