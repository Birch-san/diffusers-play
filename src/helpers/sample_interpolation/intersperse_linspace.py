from typing import TypeVar, List, Optional
from itertools import chain, pairwise
import numpy as np
from .in_between import InBetweenParams, MakeInbetween

T = TypeVar('T')

def intersperse_linspace(
  lst: List[T],
  make_inbetween: MakeInbetween[T],
  steps: Optional[int]
) -> List[T]:
  if steps is None:
    return lst
  return [
    *chain(
      *(
        (
          pair[0],
          *(
            make_inbetween(
              InBetweenParams(
                from_=pair[0],
                to=pair[1],
                step=step
              )
            ) for step in np.linspace(
                start=1/steps,
                stop=1,
                num=steps-1,
                endpoint=False
            )
          )
        ) for pair in pairwise(lst)
      )
    ),
    lst[-1]
  ]