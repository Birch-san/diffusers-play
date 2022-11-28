import numpy as np
from random import randint

uint32_iinfo = np.iinfo(np.uint32)
min, max = uint32_iinfo.min, uint32_iinfo.max

def get_seed() -> int:
  return randint(uint32_iinfo.min, uint32_iinfo.max)