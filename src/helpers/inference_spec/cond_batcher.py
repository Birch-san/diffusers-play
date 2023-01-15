from typing import TypeVar, Protocol, Generic, Iterable, Tuple, List, Generator, TypeAlias
from ..iteration.rle import run_length, RLEGeneric
from ..embed_text_types import EmbeddingAndMask

SampleSpec = TypeVar('SampleSpec')

CondBatcherOutput: TypeAlias = List[RLEGeneric[EmbeddingAndMask]]

class MakeConds(Protocol, Generic[SampleSpec]):
  @staticmethod
  def __call__(spec: SampleSpec) -> EmbeddingAndMask: ...

class CondBatcher(Generic[SampleSpec]):
  make_conds: MakeConds[SampleSpec]
  def __init__(
    self,
    make_conds: MakeConds[SampleSpec],
  ) -> None:
    self.make_conds = make_conds

  def generate(
    self,
    spec_chunks: Iterable[Tuple[SampleSpec, ...]],
  ) -> Generator[CondBatcherOutput, None, None]:
    for chnk in spec_chunks:
      rle_specs: List[RLEGeneric[SampleSpec]] = list(run_length.encode(chnk))
      embeds: List[RLEGeneric[EmbeddingAndMask]] = [
        RLEGeneric(self.make_conds(rle_spec.element), rle_spec.count) for rle_spec in rle_specs
      ]
      yield embeds