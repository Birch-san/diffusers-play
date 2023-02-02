from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Protocol

@dataclass
class WeightedPrompt:
  prompt: str
  weight: float

class ConditionProto(Protocol):
  weighted_prompts: List[WeightedPrompt]
  prompts: List[str]

@dataclass
class ConditionSpec(ABC, ConditionProto):
  cfg_scale: float

  @property
  def cfg_enabled(self) -> bool:
    return self.cfg_scale > 1.0
  
  # @abstractmethod
  # @property
  # def cond_prompts(self) -> List[str]: ...
  # @property
  # def prompts(self) -> List[str]:
  #   if self.cfg_enabled:
  #     return self.cond_prompts
  #   return ['', *self.cond_prompts]

  # @property
  # @abstractmethod
  # def weighted_prompts(self) -> List[WeightedPrompt]: ...

  # @property
  # @abstractmethod
  # def prompts(self) -> List[str]: ...
    

@dataclass
class SingleCondition(ConditionSpec):
  prompt: str
  
  # @property
  # def cond_prompts(self) -> List[str]:
  #   return [self.prompt]

  @property
  def prompts(self) -> List[str]:
    return [self.prompt]

  @property
  def weighted_prompts(self) -> List[WeightedPrompt]:
    return [WeightedPrompt(
      prompt=self.prompt,
      weight=1.,
    )]


@dataclass
class MultiCond(ConditionSpec):
  weighted_prompts: List[WeightedPrompt]

  # @property
  # def weighted_prompts(self) -> List[WeightedPrompt]:
  #   return self._weighted_prompts

  @property
  def prompts(self) -> List[str]:
    return [weighted_prompt.prompt for weighted_prompt in self.weighted_prompts]

ConditionSpec.register(SingleCondition)
ConditionSpec.register(MultiCond)