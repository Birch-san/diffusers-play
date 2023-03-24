from enum import Enum, auto
from typing import Dict

class EncoderCkpt(Enum):
  SD1_5 = auto()

approx_encoder_ckpt_filenames: Dict[EncoderCkpt, str] = {
 EncoderCkpt.SD1_5: 'approx_encoder_sd1.5.pt',
}