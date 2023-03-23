from enum import Enum, auto
from typing import Dict

class DecoderCkpt(Enum):
  SD1_4 = auto()
  SD1_5 = auto()
  WD1_3 = auto()
  WD1_4 = auto()
  WD1_5 = auto()

approx_decoder_ckpt_filenames: Dict[DecoderCkpt, str] = {
 DecoderCkpt.SD1_4: 'approx_decoder_sd1.4.pt',
 DecoderCkpt.SD1_5: 'approx_decoder_sd1.5.pt',
 DecoderCkpt.WD1_3: 'approx_decoder_wd1.3.pt',
 DecoderCkpt.WD1_4: 'approx_decoder_wd1.4.pt',
 DecoderCkpt.WD1_5: 'approx_decoder_wd1.5.pt',
}