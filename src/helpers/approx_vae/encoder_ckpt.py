from enum import Enum, auto
from typing import Dict

class EncoderCkpt(Enum):
  SD1_4 = auto()
  SD1_5 = auto()
  WD1_3 = auto()
  WD1_4 = auto()
  WD1_5 = auto()

approx_encoder_ckpt_filenames: Dict[EncoderCkpt, str] = {
 EncoderCkpt.SD1_4: 'encoder_sd1.4.pt',
 EncoderCkpt.SD1_5: 'encoder_sd1.5.pt',
 EncoderCkpt.WD1_3: 'encoder_wd1.3.pt',
 EncoderCkpt.WD1_4: 'encoder_wd1.4.pt',
 EncoderCkpt.WD1_5: 'encoder_wd1.5.pt',
}