from typing import List
from dataclasses import dataclass
from abc import ABC

from .sample_spec import SampleSpec

@dataclass
class IdenticalSamplesBatchSpec():
  sample: SampleSpec

@dataclass
class VariedSamplesBatchSpec():
  samples: List[SampleSpec]

@dataclass
class BatchSpec(ABC): pass
BatchSpec.register(IdenticalSamplesBatchSpec)
BatchSpec.register(VariedSamplesBatchSpec)