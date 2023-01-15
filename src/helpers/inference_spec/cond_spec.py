from abc import ABC, abstractproperty
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
  @abstractproperty
  def prompts(self) -> Prompts: ...

@dataclass
class SingleCondition(ConditionSpec):
  prompt: str
  
  @property
  def prompts(self) -> Prompts:
    return self.prompt

@dataclass
class MultiCond(ConditionSpec):
  weighted_prompts: List[WeightedPrompt]

  @property
  def prompts(self) -> Prompts:
    return [weighted_prompt.prompt for weighted_prompt in self._weighted_prompts]

ConditionSpec.register(SingleCondition)
ConditionSpec.register(MultiCond)