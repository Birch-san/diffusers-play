from typing import Protocol, Generator, Iterable, NamedTuple, Generic, TypeVar, Tuple, List
from torch import FloatTensor
from itertools import tee
from ..iteration.chunk import chunk
from ..embed_text_types import EmbeddingAndMask

SampleSpec = TypeVar('SampleSpec')

class MakeLatentBatches(Protocol, Generic[SampleSpec]):
  @staticmethod
  def __call__(spec_chunks: Iterable[Tuple[SampleSpec, ...]]) -> Iterable[FloatTensor]: ...

class MakeCondBatches(Protocol, Generic[SampleSpec]):
  @staticmethod
  def __call__(spec_chunks: Iterable[Tuple[SampleSpec, ...]]) -> Iterable[EmbeddingAndMask]: ...

class BatchSpec(NamedTuple):
  spec_chunk: Tuple[SampleSpec, ...]
  latents: FloatTensor
  conds: List[EmbeddingAndMask]
class BatchSpecGeneric(BatchSpec, Generic[SampleSpec]): pass

class SampleSpecBatcher(Generic[SampleSpec]):
  make_latent_batches: MakeLatentBatches[SampleSpec]
  make_cond_batches: MakeCondBatches[SampleSpec]
  def __init__(
    self,
    batch_size: int,
    make_latent_batches: MakeLatentBatches[SampleSpec],
    make_cond_batches: MakeCondBatches[SampleSpec],
  ) -> None:
    self.batch_size = batch_size
    self.make_latent_batches = make_latent_batches
    self.make_cond_batches = make_cond_batches
  
  def generate(
    self,
    specs: Iterable[SampleSpec],
  ) -> Generator[BatchSpecGeneric[SampleSpec], None, None]:
    spec_chunks: Iterable[Tuple[SampleSpec, ...]] = chunk(specs, self.batch_size)
    batcher_it, latent_it, cond_it = tee(spec_chunks, 3)
    latent_batches: Iterable[FloatTensor] = self.make_latent_batches(latent_it)
    cond_batches: Iterable[FloatTensor] = self.make_cond_batches(cond_it)
    for spec_chunk, latents, conds in zip(batcher_it, latent_batches, cond_batches):
      yield BatchSpecGeneric[SampleSpec](
        spec_chunk=spec_chunk,
        latents=latents,
        conds=conds,
      )