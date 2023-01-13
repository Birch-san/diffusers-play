from typing import Iterable, Tuple, Generator, TypeVar, Generic, Protocol
from .cond_batcher import MakeConds, CondBatcher
from ..embed_text_types import Embed, EmbeddingAndMask, Prompts

SampleSpec = TypeVar('SampleSpec')

def conds_from_prompts_factory(
  embed: Embed,
) -> MakeConds[Prompts]:
  def make_conds(prompts: Prompts) -> EmbeddingAndMask:
    embedding_and_mask = embed(prompts)
    return embedding_and_mask
  return make_conds

class GetPromptsFromSpec(Protocol, Generic[SampleSpec]):
  @staticmethod
  def __call__(spec: SampleSpec) -> Prompts: ...

def make_cond_batches(
  make_conds: MakeConds[int],
  get_prompts_from_spec: GetPromptsFromSpec,
  spec_chunks: Iterable[Tuple[SampleSpec, ...]],
) -> Iterable[EmbeddingAndMask]:
  seed_chunks: Iterable[Tuple[Prompts, ...]] = map(lambda chunk: tuple(map(get_prompts_from_spec, chunk)), spec_chunks)
  batcher = CondBatcher(
    make_conds=make_conds,
  )
  generator: Generator[EmbeddingAndMask, None, None] = batcher.generate(seed_chunks)
  return generator