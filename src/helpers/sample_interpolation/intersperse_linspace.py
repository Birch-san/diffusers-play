from typing import List, Iterable
from itertools import chain, pairwise
import numpy as np
from .in_between import InBetweenParams, MakeInbetween, T, U

def intersperse_linspace(
  keyframes: List[T],
  make_inbetween: MakeInbetween[T, U],
  steps: int
) -> Iterable[T | U]:
  assert len(keyframes) > 1
  assert steps > 0
  return (
    *chain(
      *(
        (
          pair[0],
          *(
            make_inbetween(
              InBetweenParams[T](
                from_=pair[0],
                to=pair[1],
                quotient=step,
              )
            ) for step in np.linspace(
                start=1/steps,
                stop=1,
                num=steps-1,
                endpoint=False,
            )
          )
        ) for pair in pairwise(keyframes)
      )
    ),
    keyframes[-1]
  )