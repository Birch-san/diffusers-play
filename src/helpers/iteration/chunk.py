from typing import TypeVar, Iterable, Iterator, Tuple
from itertools import islice

T = TypeVar('T')

def chunk(it: Iterable[T], size: int) -> Iterable[Tuple[T, ...]]:
  it: Iterator[T] = iter(it)
  return iter(lambda: tuple(islice(it, size)), ())