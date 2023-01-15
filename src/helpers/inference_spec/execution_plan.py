from dataclasses import dataclass
from ..embed_text_types import Prompts
from typing import Optional
from .sample_spec_2 import SampleSpec
from .latent_spec import Img2ImgSpec, FeedbackSpec

@dataclass
class ExecutionPlan:
  latents_from_prev_sample: bool
  start_sigma: Optional[float]
  prompts: Prompts
  cfg_enabled: bool

def make_execution_plan(spec: SampleSpec) -> ExecutionPlan:
  latents_from_prev_sample: bool = isinstance(spec.latent_spec, FeedbackSpec)
  start_sigma: Optional[float] = spec.latent_spec.from_sigma if isinstance(spec.latent_spec, Img2ImgSpec) else None
  prompts: Prompts = spec.cond_spec.prompts
  cfg_enabled: bool = spec.cond_spec.cfg_scale > 1.
  return ExecutionPlan(
    latents_from_prev_sample=latents_from_prev_sample,
    start_sigma=start_sigma,
    prompts=prompts,
    cfg_enabled=cfg_enabled,
  )