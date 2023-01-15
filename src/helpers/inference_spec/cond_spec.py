from abc import ABC, abstractproperty
from dataclasses import dataclass
from typing import List

@dataclass
class WeightedPrompt:
  prompt: str
  weight: float

@dataclass
class ConditionSpec(ABC):
  cfg_scale: float
  @abstractproperty
  def cond_prompts(self) -> List[str]: ...
  @property
  def prompts(self) -> List[str]:
    if self.cfg_scale==1.:
      return self.cond_prompts
    return ['', *self.cond_prompts]
    

@dataclass
class SingleCondition(ConditionSpec):
  prompt: str
  
  @property
  def cond_prompts(self) -> List[str]:
    return [self.prompt]

@dataclass
class MultiCond(ConditionSpec):
  weighted_prompts: List[WeightedPrompt]

  @property
  def prompts(self) -> List[str]:
    return [weighted_prompt.prompt for weighted_prompt in self._weighted_prompts]

ConditionSpec.register(SingleCondition)
ConditionSpec.register(MultiCond)