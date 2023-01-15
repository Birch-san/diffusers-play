from typing import Generic, TypeVar, Protocol, Generator, Iterable, List, NamedTuple, Optional

SampleSpec = TypeVar('SampleSpec')
ExecutionPlan = TypeVar('ExecutionPlan')

class MakeExecutionPlan(Protocol, Generic[SampleSpec, ExecutionPlan]):
  @staticmethod
  def __call__(spec: SampleSpec) -> ExecutionPlan: ...

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
      plan: ExecutionPlan = self.make_execution_plan(spec)
      if current_plan is None:
        current_plan = plan
      if plan == current_plan:
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