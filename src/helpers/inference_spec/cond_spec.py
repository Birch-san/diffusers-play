from abc import ABC
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

@dataclass
class ConditionSpec(ABC, ConditionProto):
  cfg: Optional[CFG]

  @property
  def uncond_prompt_texts(self) -> List[str]:
    return [] if self.cfg is None else [self.cfg.uncond_prompt.text]


@dataclass
class SingleCondition(ConditionSpec):
  prompt: Prompt

  @property
  def weighted_cond_prompts(self) -> List[WeightedPrompt]:
    return [WeightedPrompt(
      prompt=self.prompt,
      weight=1.,
    )]


@dataclass
class MultiCond(ConditionSpec):
  weighted_cond_prompts: List[WeightedPrompt]

ConditionSpec.register(SingleCondition)
ConditionSpec.register(MultiCond)