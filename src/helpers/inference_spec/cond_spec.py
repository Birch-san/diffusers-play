from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
from ..embed_text_types import Prompts


@dataclass
class WeightedPrompt:
  prompt: str
  weight: float

@dataclass
class ConditionSpec(ABC):
  cfg_scale: float
  @abstractmethod
  def get_prompts(self) -> Prompts: ...
@dataclass
class SingleCondition(ConditionSpec):
  prompt: str
  def get_prompts(self) -> Prompts:
    return self.prompt
@dataclass
class MultiCond(ConditionSpec):
  weighted_prompts: List[WeightedPrompt]
  def get_prompts(self) -> Prompts:
    return [weighted_prompt.prompt for weighted_prompt in self.weighted_prompts]
ConditionSpec.register(SingleCondition)
ConditionSpec.register(MultiCond)