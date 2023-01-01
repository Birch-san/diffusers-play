from torch import BoolTensor, FloatTensor
from typing import Union, Iterable, Protocol, NamedTuple
from typing_extensions import TypeAlias

Prompts: TypeAlias = Union[str, Iterable[str]]

class EmbeddingAndMask(NamedTuple):
  embedding: FloatTensor
  attn_mask: BoolTensor

class Embed(Protocol):
  @staticmethod
  def __call__(prompt: Prompts) -> EmbeddingAndMask: ...