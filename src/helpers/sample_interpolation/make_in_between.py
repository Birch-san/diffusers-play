from ..inference_spec.cond_spec import SingleCondition, MultiCond, WeightedPrompt, CFG, Prompt
from .in_between import InBetweenParams
from itertools import chain, repeat
from typing import Protocol, Tuple, Iterable

class AdjustScale(Protocol):
  @staticmethod
  def __call__(scale_nominal: float, quotient: float) -> float: ...

WeightedPromptAndScaleAdjuster = Tuple[WeightedPrompt, AdjustScale]

def make_inbetween(params: InBetweenParams[SingleCondition|MultiCond]) -> MultiCond:
  assert (params.from_.cfg is None) == (params.to.cfg is None)
  if params.from_.cfg is not None:
    assert params.from_.cfg.uncond_prompt.text == params.to.cfg.uncond_prompt.text
    from_scale: float = params.from_.cfg.scale
    to_scale: float = params.to.cfg.scale
    cfg = CFG(scale=from_scale, uncond_prompt=params.from_.cfg.uncond_prompt)
    cfg_scale_coeff = to_scale / from_scale
  else:
    cfg = None
    cfg_scale_coeff = 1.
  scale_from: AdjustScale = lambda scale_nominal, quotient: scale_nominal * (1-quotient)
  scale_to: AdjustScale = lambda scale_nominal, quotient: scale_nominal * quotient * cfg_scale_coeff
  prompts_and_scale_strategies: Iterable[WeightedPromptAndScaleAdjuster] = chain(
    zip(params.from_.weighted_cond_prompts, repeat(scale_from)),
    zip(params.to.weighted_cond_prompts, repeat(scale_to))
  )

  return MultiCond(
    cfg=cfg,
    weighted_cond_prompts=[
      WeightedPrompt(
        wp.prompt,
        scale(wp.weight, params.quotient)
      ) for wp, scale in prompts_and_scale_strategies
    ]
  )