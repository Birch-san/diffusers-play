from typing import TypeVar, TypeAlias, Generic, Callable, List, Optional, Iterable, Iterator, Tuple
from itertools import chain, pairwise, islice
from dataclasses import dataclass
import numpy as np
from enum import Enum, auto

T = TypeVar('T')

@dataclass
class InBetweenParams(Generic[T]):
  from_: T
  to: T
  step: float

MakeInbetween: TypeAlias = Callable[[InBetweenParams], T]