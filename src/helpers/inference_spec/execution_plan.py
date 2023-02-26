from dataclasses import dataclass
from typing import Optional, Dict, List
from ..sample_interpolation.interp_strategy import InterpStrategy
from .cond_spec import InterPrompt
from .sample_spec import SampleSpec
from .latent_spec import Img2ImgSpec, FeedbackSpec
from .execution_plan_batcher import PlanMergeResultGeneric

@dataclass
class CfgState:
  # per sample: within that sample's prompt_text_instance_ixs element, the ix at which its uncond can be found
  uncond_instance_ixs: List[int]
  scales: List[float]

@dataclass
class CondInterp:
  prompt_text_instance_ix: int
  interp_strategy: InterpStrategy
  interp_quotient: InterpStrategy

@dataclass
class ExecutionPlan:
  start_sigma: Optional[float]
  # deduped list for telling CLIP what to embed
  prompt_texts_ordered: List[str]
  # accumulator state, for locating prompts we've already planned to embed
  _prompt_text_to_ix: Dict[str, int]
  # per sample: the indices (in prompt_texts_ordered) of embeddings on which it depends (includes uncond and cond)
  prompt_text_instance_ixs: List[List[int]]
  cond_weights: List[float]
  cfg: Optional[CfgState]
  cond_interps: List[List[Optional[CondInterp]]]

def make_execution_plan(acc: Optional[ExecutionPlan], spec: SampleSpec) -> PlanMergeResultGeneric[ExecutionPlan]:
  start_sigma: Optional[float] = spec.latent_spec.from_sigma if isinstance(spec.latent_spec, Img2ImgSpec) else None

  # FeedbackSpec has dependency on previous sample, so must only ever be first in a batch
  can_merge = acc is not None and (spec.cond_spec.cfg is None) == (acc.cfg is None) and start_sigma == acc.start_sigma and not(isinstance(spec.latent_spec, FeedbackSpec))

  if can_merge:
    prompt_text_to_ix: Dict[str, int] = acc._prompt_text_to_ix
    prompt_texts_ordered: List[str] = acc.prompt_texts_ordered
    prompt_text_instance_ixs: List[List[int]] = acc.prompt_text_instance_ixs
    cond_weights: List[float] = acc.cond_weights
    cfg: Optional[CfgState] = acc.cfg
    cond_interps: List[List[Optional[CondInterp]]] = acc.cond_interps
  else:
    prompt_text_to_ix: Dict[str, int] = {}
    prompt_texts_ordered: List[str] = []
    prompt_text_instance_ixs: List[List[int]] = []
    cond_weights: List[float] = []
    cfg: Optional[CfgState] = None if spec.cond_spec.cfg is None else CfgState(
      uncond_instance_ixs = [],
      scales = []
    )
    cond_interps: List[List[Optional[CondInterp]]] = []
  
  def register_prompt_text(prompt_text: str) -> None:
    if prompt_text not in prompt_text_to_ix:
      prompt_text_to_ix[prompt_text] = len(prompt_texts_ordered)
      prompt_texts_ordered.append(prompt_text)
    ix: int = prompt_text_to_ix[prompt_text]
    return ix
  
  sample_prompt_text_instance_ixs: List[int] = []
  sample_cond_interps: List[Optional[CondInterp]] = []
  if cfg is not None:
    cfg.scales.append(spec.cond_spec.cfg.scale)
    cfg.uncond_instance_ixs.append(len(sample_prompt_text_instance_ixs))
    prompt_text: str = spec.cond_spec.cfg.uncond_prompt.text
    prompt_ix: int = register_prompt_text(prompt_text)
    sample_prompt_text_instance_ixs.append(prompt_ix)
    assert not isinstance(spec.cond_spec.cfg.uncond_prompt, InterPrompt), "InterPrompt not implemented for uncond. you can achieve this by describing your CFG as a multi-cond instead https://twitter.com/Birchlabs/status/1627286152087478272"
    sample_cond_interps.append(None)

  for weighted_prompt in spec.cond_spec.weighted_cond_prompts:
    prompt_text: str = weighted_prompt.prompt.text
    prompt_ix: int = register_prompt_text(prompt_text)
    sample_prompt_text_instance_ixs.append(prompt_ix)
    cond_weights.append(weighted_prompt.weight)
    match weighted_prompt.prompt:
      case InterPrompt(start, end, quotient, strategy):
        secondary_prompt_ix: int = register_prompt_text(end.text)
        cond_interp = CondInterp(
          prompt_text_instance_ix=secondary_prompt_ix,
          interp_quotient=quotient,
          interp_strategy=strategy,
        )
        sample_cond_interps.append(cond_interp)
      case _:
        sample_cond_interps.append(None)
  prompt_text_instance_ixs.append(sample_prompt_text_instance_ixs)
  cond_interps.append(sample_cond_interps)
  
  plan = ExecutionPlan(
    start_sigma=start_sigma,
    prompt_texts_ordered=prompt_texts_ordered,
    _prompt_text_to_ix=prompt_text_to_ix,
    prompt_text_instance_ixs=prompt_text_instance_ixs,
    cond_weights=cond_weights,
    cfg=cfg,
    cond_interps=cond_interps,
  )

  return PlanMergeResultGeneric(
    plan=plan,
    merge_success=can_merge,
  )