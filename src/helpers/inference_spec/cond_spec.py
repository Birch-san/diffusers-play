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
  # we deliberately avoid implementing support for uncond to be a plurality of weighted prompts.
  # it's not impossible to do, but so obscure as to not be worth it.
  # CFG is *already* weird
  # "negative prompting" is weird on top of weird
  # "negative multi-prompting"? now that's a step too far
  # maybe you can get a similar (identical?) effect by creating a
  # ConditionProto#weighted_cond_prompts element with a negative weight
  uncond_prompt: Prompt = field(default_factory=Prompt)

@dataclass
class WeightedPrompt:
  prompt: Prompt
  weight: float

class ConditionProto(Protocol):
  weighted_cond_prompts: List[WeightedPrompt]
  cond_prompt_texts: List[str]
  cond_prompt_weights: List[float]
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

  @property
  def prompt_texts(self) -> List[str]:
    return [*self.uncond_prompt_texts, *self.cond_prompt_texts]
  
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
  def cond_prompt_texts(self) -> List[str]:
    return [self.prompt.text]

  @property
  def cond_prompt_weights(self) -> List[float]:
    return [1.]

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
  def cond_prompt_texts(self) -> List[str]:
    return [weighted_prompt.prompt.text for weighted_prompt in self.weighted_cond_prompts]

  @property
  def cond_prompt_weights(self) -> List[float]:
    return [weighted_prompt.weight for weighted_prompt in self.weighted_cond_prompts]

ConditionSpec.register(SingleCondition)
ConditionSpec.register(MultiCond)