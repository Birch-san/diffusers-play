from enum import Enum, auto

class InterpStrategy(Enum):
  CondDiff = auto()
  LatentSlerp = auto()
  LatentLerp = auto()