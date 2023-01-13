from typing import TypeVar, Protocol, Generic, Iterable, Tuple, List, Generator
from torch import FloatTensor
from torch import cat
from ..iteration.rle import run_length, RLEGeneric
from ..embed_text_types import EmbeddingAndMask

SampleSpec = TypeVar('SampleSpec')

class MakeConds(Protocol, Generic[SampleSpec]):
  @staticmethod
  def __call__(spec: SampleSpec) -> List[EmbeddingAndMask]: ...

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
  ) -> Generator[List[EmbeddingAndMask], None, None]:
    for chnk in spec_chunks:
      rle_specs: List[RLEGeneric[SampleSpec]] = list(run_length.encode(chnk))
      embeds: List[EmbeddingAndMask] = [
        self.make_conds(rle_spec.element) for rle_spec in rle_specs
      ]
      # TODO: we probably want to repeat_interleave() the embedding and mask along batch dim
      yield embeds