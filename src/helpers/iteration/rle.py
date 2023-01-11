from typing import Generic, TypeVar, Iterable, NamedTuple, Generator
from itertools import count, groupby, chain, repeat
from collections import deque

# from Erik Rose's more-itertools, MIT-licensed
# https://pypi.org/project/more-itertools/
# https://more-itertools.readthedocs.io/en/latest/api.html#more_itertools.run_length
# https://more-itertools.readthedocs.io/en/latest/_modules/more_itertools/more.html#run_length
# typings added by Alex Birch

T = TypeVar('T')

class RunLengthEncoded(NamedTuple):
  element: T
  count: int
class RLEGeneric(RunLengthEncoded, Generic[T]): pass

def ilen(iterable: Iterable[T]) -> int:
  """Return the number of items in *iterable*.

    >>> ilen(x for x in range(1000000) if x % 3 == 0)
    333334

  This consumes the iterable, so handle with care.

  """
  # This approach was selected because benchmarks showed it's likely the
  # fastest of the known implementations at the time of writing.
  # See GitHub tracker: #236, #230.
  counter = count()
  deque(zip(iterable, counter), maxlen=0)
  return next(counter)

class run_length(Generic[T]):
  """
  :func:`run_length.encode` compresses an iterable with run-length encoding.
  It yields groups of repeated items with the count of how many times they
  were repeated:

      >>> uncompressed = 'abbcccdddd'
      >>> list(run_length.encode(uncompressed))
      [('a', 1), ('b', 2), ('c', 3), ('d', 4)]

  :func:`run_length.decode` decompresses an iterable that was previously
  compressed with run-length encoding. It yields the items of the
  decompressed iterable:

      >>> compressed = [('a', 1), ('b', 2), ('c', 3), ('d', 4)]
      >>> list(run_length.decode(compressed))
      ['a', 'b', 'b', 'c', 'c', 'c', 'd', 'd', 'd', 'd']

  """

  @staticmethod
  def encode(iterable: Iterable[T]) -> Generator[RLEGeneric[T], None, None]:
    for k, g in groupby(iterable):
      yield RLEGeneric(k, ilen(g))

  @staticmethod
  def decode(iterable: Iterable[T]) -> chain[T]:
    return chain.from_iterable(repeat(k, n) for k, n in iterable)