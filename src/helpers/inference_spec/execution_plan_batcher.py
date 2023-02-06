from typing import Generic, TypeVar, Protocol, Generator, Iterable, List, NamedTuple, Optional

SampleSpec = TypeVar('SampleSpec')
ExecutionPlan = TypeVar('ExecutionPlan')

class PlanMergeResult(NamedTuple):
  plan: ExecutionPlan
  merge_success: bool
class PlanMergeResultGeneric(PlanMergeResult, Generic[ExecutionPlan]): pass

class MakeExecutionPlan(Protocol, Generic[SampleSpec, ExecutionPlan]):
  @staticmethod
  def __call__(acc: Optional[ExecutionPlan], spec: SampleSpec) -> PlanMergeResultGeneric[ExecutionPlan]: ...

class BatchSpec(NamedTuple):
  execution_plan: ExecutionPlan
  sample_specs: List[SampleSpec]
class BatchSpecGeneric(BatchSpec, Generic[ExecutionPlan]): pass

class ExecutionPlanBatcher(Generic[SampleSpec, ExecutionPlan]):
  max_batch_size: int
  make_execution_plan: MakeExecutionPlan[SampleSpec, ExecutionPlan]
  def __init__(
    self,
    max_batch_size: int,
    make_execution_plan: MakeExecutionPlan[SampleSpec, ExecutionPlan],
  ) -> None:
    self.max_batch_size = max_batch_size
    self.make_execution_plan = make_execution_plan
  
  def generate(self, specs: Iterable[SampleSpec]) -> Generator[BatchSpecGeneric[ExecutionPlan], None, None]:
    current_plan: Optional[ExecutionPlan] = None
    current_batch: List[SampleSpec] = []
    for spec in specs:
      result: PlanMergeResultGeneric[ExecutionPlan] = self.make_execution_plan(current_plan, spec)
      plan, merge_success = result
      if not merge_success and current_plan is not None:
        # flush currently-accumulated batch
        yield BatchSpecGeneric[ExecutionPlan](
          execution_plan=current_plan,
          sample_specs=current_batch,
        )
        current_batch.clear()
      current_plan = plan
      current_batch.append(spec)
      if len(current_batch) == self.max_batch_size:
        yield BatchSpecGeneric[ExecutionPlan](
          execution_plan=current_plan,
          sample_specs=current_batch,
        )
        current_batch.clear()
        current_plan = None
    if current_batch:
      assert current_plan is not None
      yield BatchSpecGeneric[ExecutionPlan](
        execution_plan=current_plan,
        sample_specs=current_batch,
      )