from typing import TypeVar, TypeAlias, Generic, Callable
from dataclasses import dataclass

T = TypeVar('T')
U = TypeVar('U')

@dataclass
class InBetweenParams(Generic[T]):
  from_: T
  to: T
  quotient: float

MakeInbetween: TypeAlias = Callable[[InBetweenParams[T]], U]