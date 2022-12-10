from enum import Enum, auto

class ClipImplementation(Enum):
  HF = auto()
  OpenCLIP = auto()
  # OpenAI CLIP and clip-anytorch not implemented

class ClipCheckpoint(Enum):
  OpenAI = auto()
  LAION = auto()