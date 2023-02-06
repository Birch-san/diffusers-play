from dataclasses import dataclass
from typing import Optional, Dict, List
from .sample_spec import SampleSpec
from .latent_spec import Img2ImgSpec, FeedbackSpec
from .execution_plan_batcher import PlanMergeResultGeneric

@dataclass
class ExecutionPlan:
  start_sigma: Optional[float]
  prompts_ordered: List[str]
  prompt_to_ix: Dict[str, int]
  prompt_instance_ixs: List[List[int]]
  cfg_enabled: bool
  cfgs: List[float]

def make_execution_plan(acc: Optional[ExecutionPlan], spec: SampleSpec) -> PlanMergeResultGeneric[ExecutionPlan]:
  start_sigma: Optional[float] = spec.latent_spec.from_sigma if isinstance(spec.latent_spec, Img2ImgSpec) else None
  cfg_enabled: bool = spec.cond_spec.cfg_enabled

  # FeedbackSpec has dependency on previous sample, so must only ever be first in a batch
  can_merge = acc is not None and cfg_enabled == acc.cfg_enabled and start_sigma == acc.start_sigma and not(isinstance(spec.latent_spec, FeedbackSpec))

  if can_merge:
    prompt_to_ix: Dict[str, int] = acc.prompt_to_ix
    prompts_ordered: List[str] = acc.prompts_ordered
    prompt_instance_ixs: List[List[int]] = acc.prompt_instance_ixs
    cfgs: List[float] = acc.cfgs
  else:
    prompt_to_ix: Dict[str, int] = {}
    prompts_ordered: List[str] = []
    prompt_instance_ixs: List[List[int]] = []
    cfgs: List[float] = []
  
  sample_prompt_instance_ixs: List[int] = []
  for prompt in spec.cond_spec.prompts:
    if prompt not in prompt_to_ix:
      prompt_to_ix[prompt] = len(prompts_ordered)
      prompts_ordered.append(prompt)
    ix: int = prompt_to_ix[prompt]
    sample_prompt_instance_ixs.append(ix)
  prompt_instance_ixs.append(sample_prompt_instance_ixs)

  if cfg_enabled:
    cfgs.append(spec.cond_spec.cfg_scale)
  
  plan = ExecutionPlan(
    start_sigma=start_sigma,
    prompts_ordered=prompts_ordered,
    prompt_to_ix=prompt_to_ix,
    prompt_instance_ixs=prompt_instance_ixs,
    cfg_enabled=cfg_enabled,
    cfgs=cfgs,
  )

  return PlanMergeResultGeneric(
    plan=plan,
    merge_success=can_merge,
  )