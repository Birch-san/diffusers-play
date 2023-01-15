from .cond_spec import ConditionSpec
from dataclasses import dataclass

@dataclass
class ExecutionPlan:
  cond_spec: ConditionSpec