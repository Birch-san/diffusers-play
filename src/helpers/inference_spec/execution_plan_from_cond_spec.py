from dataclasses import dataclass
from typing import Iterable, Tuple, Generator
from .execution_plan_batcher import MakeExecutionPlan, ExecutionPlanBatcher, ExecutionPlanBatcherOutput
from .cond_spec import ConditionSpec
from .execution_plan import ExecutionPlan

def make_execution_plan_from_cond_spec(cond_spec: ConditionSpec) -> ExecutionPlan:
  return ExecutionPlan(
    cond_spec=cond_spec,
  )

def make_execution_plan_batches(
  make_execution_plan: MakeExecutionPlan[ConditionSpec],
  cond_spec_chunks: Iterable[Tuple[ConditionSpec, ...]],
) -> Iterable[ExecutionPlanBatcherOutput]:
  batcher = ExecutionPlanBatcher(
    make_execution_plan=make_execution_plan,
  )
  generator: Generator[ExecutionPlanBatcherOutput, None, None] = batcher.generate(cond_spec_chunks)
  return generator