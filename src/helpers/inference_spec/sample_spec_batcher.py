from typing import Protocol, Generator, Iterable, NamedTuple, Generic, TypeVar, Tuple
from torch import FloatTensor
from itertools import tee
from ..iteration.chunk import chunk
from .latent_batcher import LatentBatcherOutput
from .cond_batcher import CondBatcherOutput
from .execution_plan_batcher import ExecutionPlanBatcherOutput

SampleSpec = TypeVar('SampleSpec')

class MakeLatentBatches(Protocol, Generic[SampleSpec]):
  @staticmethod
  def __call__(spec_chunks: Iterable[Tuple[SampleSpec, ...]]) -> Iterable[LatentBatcherOutput]: ...

class MakeCondBatches(Protocol, Generic[SampleSpec]):
  @staticmethod
  def __call__(spec_chunks: Iterable[Tuple[SampleSpec, ...]]) -> Iterable[CondBatcherOutput]: ...

class MakeExecutionPlanBatches(Protocol, Generic[SampleSpec]):
  @staticmethod
  def __call__(spec_chunks: Iterable[Tuple[SampleSpec, ...]]) -> Iterable[ExecutionPlanBatcherOutput]: ...

class BatchSpec(NamedTuple):
  spec_chunk: Tuple[SampleSpec, ...]
  latents: LatentBatcherOutput
  conds: CondBatcherOutput
  execution_plan: ExecutionPlanBatcherOutput
class BatchSpecGeneric(BatchSpec, Generic[SampleSpec]): pass

class SampleSpecBatcher(Generic[SampleSpec]):
  make_latent_batches: MakeLatentBatches[SampleSpec]
  make_cond_batches: MakeCondBatches[SampleSpec]
  make_execution_plan_batches: MakeExecutionPlanBatches[SampleSpec]
  def __init__(
    self,
    batch_size: int,
    make_latent_batches: MakeLatentBatches[SampleSpec],
    make_cond_batches: MakeCondBatches[SampleSpec],
    make_execution_plan_batches: MakeExecutionPlanBatches[SampleSpec],
  ) -> None:
    self.batch_size = batch_size
    self.make_latent_batches = make_latent_batches
    self.make_cond_batches = make_cond_batches
    self.make_execution_plan_batches = make_execution_plan_batches
  
  def generate(
    self,
    specs: Iterable[SampleSpec],
  ) -> Generator[BatchSpecGeneric[SampleSpec], None, None]:
    spec_chunks: Iterable[Tuple[SampleSpec, ...]] = chunk(specs, self.batch_size)
    batcher_it, latent_it, cond_it, ex_it = tee(spec_chunks, 4)
    latent_batches: Iterable[LatentBatcherOutput] = self.make_latent_batches(latent_it)
    cond_batches: Iterable[CondBatcherOutput] = self.make_cond_batches(cond_it)
    execution_plan_batches: Iterable[ExecutionPlanBatcherOutput] = self.make_execution_plan_batches(ex_it)
    for spec_chunk, latents, conds, ex in zip(batcher_it, latent_batches, cond_batches, execution_plan_batches):
      yield BatchSpecGeneric[SampleSpec](
        spec_chunk=spec_chunk,
        latents=latents,
        conds=conds,
        execution_plan=ex,
      )