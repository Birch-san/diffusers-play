from dataclasses import dataclass
from .cond_spec import ConditionSpec
from .latent_spec import LatentSpec

@dataclass
class SampleSpec:
  latent_spec: LatentSpec
  cond_spec: ConditionSpec