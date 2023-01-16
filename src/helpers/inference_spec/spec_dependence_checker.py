from typing import Protocol, Iterable, TypeVar, Generic

SampleSpec = TypeVar('SampleSpec')

class CheckSpecDependenceStrategy(Protocol, Generic[SampleSpec]):
  @staticmethod
  def __call__(spec: SampleSpec) -> bool: ...

class SpecDependenceChecker(Generic[SampleSpec]):
  strategies: Iterable[CheckSpecDependenceStrategy[SampleSpec]]
  def __init__(
    self,
    strategies: Iterable[CheckSpecDependenceStrategy[SampleSpec]],
  ) -> None:
    self.strategies = strategies

  def has_dependence(
    self,
    spec: SampleSpec,
  ) -> bool:
    for strategy in self.strategies:
      if strategy(spec):
        return True
    return False