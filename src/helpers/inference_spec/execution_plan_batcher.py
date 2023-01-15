from typing import TypeVar, Protocol, Generic, Iterable, Tuple, List, Generator, TypeAlias
from .execution_plan import ExecutionPlan
from ..iteration.rle import run_length, RLEGeneric

SampleSpec = TypeVar('SampleSpec')

ExecutionPlanBatcherOutput: TypeAlias = RLEGeneric[ExecutionPlan]

class MakeExecutionPlan(Protocol, Generic[SampleSpec]):
  @staticmethod
  def __call__(spec: SampleSpec) -> ExecutionPlan: ...

class ExecutionPlanBatcher(Generic[SampleSpec]):
  make_execution_plan: MakeExecutionPlan[SampleSpec]
  def __init__(
    self,
    make_execution_plan: MakeExecutionPlan[SampleSpec],
  ) -> None:
    self.make_execution_plan = make_execution_plan

  def generate(
    self,
    spec_chunks: Iterable[Tuple[SampleSpec, ...]],
  ) -> Generator[ExecutionPlanBatcherOutput, None, None]:
    for chnk in spec_chunks:
      rle_specs: List[RLEGeneric[SampleSpec]] = list(run_length.encode(chnk))
      ex_plans: List[RLEGeneric[ExecutionPlan]] = [
        RLEGeneric(self.make_execution_plan(rle_spec.element), rle_spec.count) for rle_spec in rle_specs
      ]
      yield ex_plans