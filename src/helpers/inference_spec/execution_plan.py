from dataclasses import dataclass
from typing import Optional, Dict, List
from .sample_spec import SampleSpec
from .latent_spec import Img2ImgSpec, FeedbackSpec
from .execution_plan_batcher import PlanMergeResultGeneric

@dataclass
class ExecutionPlan:
  start_sigma: Optional[float]
  prompt_texts_ordered: List[str]
  prompt_text_to_ix: Dict[str, int]
  prompt_text_instance_ixs: List[List[int]]
  cfg_enabled: bool
  cfgs: List[float]

def make_execution_plan(acc: Optional[ExecutionPlan], spec: SampleSpec) -> PlanMergeResultGeneric[ExecutionPlan]:
  start_sigma: Optional[float] = spec.latent_spec.from_sigma if isinstance(spec.latent_spec, Img2ImgSpec) else None
  cfg_enabled: bool = spec.cond_spec.cfg is not None

  # FeedbackSpec has dependency on previous sample, so must only ever be first in a batch
  can_merge = acc is not None and cfg_enabled == acc.cfg_enabled and start_sigma == acc.start_sigma and not(isinstance(spec.latent_spec, FeedbackSpec))

  if can_merge:
    prompt_text_to_ix: Dict[str, int] = acc.prompt_text_to_ix
    prompt_texts_ordered: List[str] = acc.prompt_texts_ordered
    prompt_text_instance_ixs: List[List[int]] = acc.prompt_text_instance_ixs
    cfgs: List[float] = acc.cfgs
  else:
    prompt_text_to_ix: Dict[str, int] = {}
    prompt_texts_ordered: List[str] = []
    prompt_text_instance_ixs: List[List[int]] = []
    cfgs: List[float] = []
  
  sample_prompt_text_instance_ixs: List[int] = []
  for prompt_text in spec.cond_spec.prompt_texts:
    if prompt_text not in prompt_text_to_ix:
      prompt_text_to_ix[prompt_text] = len(prompt_texts_ordered)
      prompt_texts_ordered.append(prompt_text)
    ix: int = prompt_text_to_ix[prompt_text]
    sample_prompt_text_instance_ixs.append(ix)
  prompt_text_instance_ixs.append(sample_prompt_text_instance_ixs)

  if cfg_enabled:
    cfgs.append(spec.cond_spec.cfg.scale)
  
  plan = ExecutionPlan(
    start_sigma=start_sigma,
    prompt_texts_ordered=prompt_texts_ordered,
    prompt_text_to_ix=prompt_text_to_ix,
    prompt_text_instance_ixs=prompt_text_instance_ixs,
    cfg_enabled=cfg_enabled,
    cfgs=cfgs,
  )

  return PlanMergeResultGeneric(
    plan=plan,
    merge_success=can_merge,
  )