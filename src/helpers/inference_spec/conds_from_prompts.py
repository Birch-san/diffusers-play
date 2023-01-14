from typing import Iterable, Tuple, Generator, TypeVar
from .cond_batcher import MakeConds, CondBatcher
from ..embed_text_types import Embed, EmbeddingAndMask, Prompts
from .map_spec_chunks import MapSpec

SampleSpec = TypeVar('SampleSpec')

def conds_from_prompts_factory(
  embed: Embed,
) -> MakeConds[Prompts]:
  def make_conds(prompts: Prompts) -> EmbeddingAndMask:
    embedding_and_mask = embed(prompts)
    return embedding_and_mask
  return make_conds

class GetPromptsFromSpec(MapSpec[SampleSpec, Prompts]): pass

def make_cond_batches(
  make_conds: MakeConds[int],
  prompts_chunks: Iterable[Tuple[Prompts, ...]],
) -> Iterable[EmbeddingAndMask]:
  batcher = CondBatcher(
    make_conds=make_conds,
  )
  generator: Generator[EmbeddingAndMask, None, None] = batcher.generate(prompts_chunks)
  return generator