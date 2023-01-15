from typing import Iterable, Tuple, Generator
from .cond_batcher import MakeConds, CondBatcher, CondBatcherOutput
from ..embed_text_types import Embed, EmbeddingAndMask, Prompts
from .cond_spec import ConditionSpec

def prompts_from_cond_spec(cond_spec: ConditionSpec) -> Prompts:
  if cond_spec.cfg_scale == 1.0:
    return cond_spec.get_prompts()
  return ['', cond_spec.get_prompts()]

def conds_from_prompts_factory(
  embed: Embed,
) -> MakeConds[Prompts]:
  def make_conds(prompts: Prompts) -> EmbeddingAndMask:
    embedding_and_mask = embed(prompts)
    return embedding_and_mask
  return make_conds

def make_cond_batches(
  make_conds: MakeConds[int],
  prompts_chunks: Iterable[Tuple[Prompts, ...]],
) -> Iterable[CondBatcherOutput]:
  batcher = CondBatcher(
    make_conds=make_conds,
  )
  generator: Generator[CondBatcherOutput, None, None] = batcher.generate(prompts_chunks)
  return generator