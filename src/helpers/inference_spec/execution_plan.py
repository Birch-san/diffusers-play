from dataclasses import dataclass
from typing import Optional, Dict, List
from .sample_spec import SampleSpec
from .latent_spec import Img2ImgSpec, FeedbackSpec
from .execution_plan_batcher import PlanMergeResultGeneric

@dataclass
class CfgState:
  # per sample: within that sample's prompt_text_instance_ixs element, wthe ix at which its uncond can be found
  uncond_instance_ixs: List[int]
  has_empty_string_uncond: List[bool]
  scales: List[float]

@dataclass
class ExecutionPlan:
  start_sigma: Optional[float]
  # deduped list for telling CLIP what to embed
  prompt_texts_ordered: List[str]
  # accumulator state, for locating prompts we've already planned to embed
  _prompt_text_to_ix: Dict[str, int]
  # per sample: the indices (in prompt_texts_ordered) of embeddings on which it depends (includes uncond and cond)
  prompt_text_instance_ixs: List[List[int]]
  cfg: Optional[CfgState]
  

def make_execution_plan(acc: Optional[ExecutionPlan], spec: SampleSpec) -> PlanMergeResultGeneric[ExecutionPlan]:
  start_sigma: Optional[float] = spec.latent_spec.from_sigma if isinstance(spec.latent_spec, Img2ImgSpec) else None

  # FeedbackSpec has dependency on previous sample, so must only ever be first in a batch
  can_merge = acc is not None and (spec.cond_spec.cfg is None) == (acc.cfg is None) and start_sigma == acc.start_sigma and not(isinstance(spec.latent_spec, FeedbackSpec))

  if can_merge:
    prompt_text_to_ix: Dict[str, int] = acc._prompt_text_to_ix
    prompt_texts_ordered: List[str] = acc.prompt_texts_ordered
    prompt_text_instance_ixs: List[List[int]] = acc.prompt_text_instance_ixs
    cfg: Optional[CfgState] = acc.cfg
  else:
    prompt_text_to_ix: Dict[str, int] = {}
    prompt_texts_ordered: List[str] = []
    prompt_text_instance_ixs: List[List[int]] = []
    cfg: Optional[CfgState] = None if spec.cond_spec.cfg is None else CfgState(
      uncond_instance_ixs = [],
      has_empty_string_uncond = [],
      scales = []
    )
  
  def register_prompt_text(prompt_text: str) -> None:
    if prompt_text not in prompt_text_to_ix:
      prompt_text_to_ix[prompt_text] = len(prompt_texts_ordered)
      prompt_texts_ordered.append(prompt_text)
    ix: int = prompt_text_to_ix[prompt_text]
    sample_prompt_text_instance_ixs.append(ix)
  
  sample_prompt_text_instance_ixs: List[int] = []
  if cfg is not None:
    cfg.scales.append(spec.cond_spec.cfg.scale)
    cfg.uncond_instance_ixs.append(len(sample_prompt_text_instance_ixs))
    prompt_text: str = spec.cond_spec.cfg.uncond_prompt.text
    cfg.has_empty_string_uncond.append(prompt_text == '')
    register_prompt_text(prompt_text)
  for prompt_text in spec.cond_spec.cond_prompt_texts:
    register_prompt_text(prompt_text)
  prompt_text_instance_ixs.append(sample_prompt_text_instance_ixs)
  
  plan = ExecutionPlan(
    start_sigma=start_sigma,
    prompt_texts_ordered=prompt_texts_ordered,
    _prompt_text_to_ix=prompt_text_to_ix,
    prompt_text_instance_ixs=prompt_text_instance_ixs,
    cfg=cfg,
  )

  return PlanMergeResultGeneric(
    plan=plan,
    merge_success=can_merge,
  )