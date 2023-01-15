from dataclasses import dataclass
from .cond_spec import ConditionSpec

@dataclass
class SampleSpec:
  seed: int
  cond_spec: ConditionSpec