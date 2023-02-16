from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Protocol, Optional


# intention is to eventually support boosting tokens
@dataclass
class Prompt:
  text: str = ''

@dataclass
class CFG:
  scale: float
  uncond_prompt: Prompt = field(default_factory=Prompt)

@dataclass
class WeightedPrompt:
  prompt: Prompt
  weight: float

class ConditionProto(Protocol):
  weighted_cond_prompts: List[WeightedPrompt]
  prompt_texts: List[str]

@dataclass
class ConditionSpec(ABC, ConditionProto):
  cfg: Optional[CFG]

  # @property
  # def cfg_enabled(self) -> bool:
  #   return self.cfg.scale > 1.0

  @property
  def uncond_prompt_texts(self) -> List[str]:
    return [] if self.cfg is None else [self.cfg.uncond_prompt.text]
  
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
  prompt: Prompt
  
  # @property
  # def cond_prompts(self) -> List[str]:
  #   return [self.prompt]

  @property
  def prompt_texts(self) -> List[str]:
    return [*self.uncond_prompt_texts, self.prompt.text]

  @property
  def weighted_cond_prompts(self) -> List[WeightedPrompt]:
    return [WeightedPrompt(
      prompt=self.prompt,
      weight=1.,
    )]


@dataclass
class MultiCond(ConditionSpec):
  weighted_cond_prompts: List[WeightedPrompt]

  # @property
  # def weighted_prompts(self) -> List[WeightedPrompt]:
  #   return self._weighted_prompts

  @property
  def prompt_texts(self) -> List[str]:
    return [*self.uncond_prompt_texts, *(weighted_prompt.prompt.text for weighted_prompt in self.weighted_cond_prompts)]

ConditionSpec.register(SingleCondition)
ConditionSpec.register(MultiCond)