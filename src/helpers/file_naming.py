from typing import Optional
from .model_db import model_shortname

def get_sample_stem(
  base_count: int,
  ix_in_batch: int,
  seed: Optional[int],
  cfg: Optional[float],
  mimic: Optional[float],
  dynthresh_percentile: Optional[float],
  center_denoise_output: Optional[bool],
  half: bool,
  model_name: str,
  filename_qualifier: str,
) -> str:
  mim: str = '' if mimic is None else f'.m{mimic}'
  dynpct: str = '' if dynthresh_percentile is None else f'.p{dynthresh_percentile}'
  cen: str = '' if center_denoise_output is None else f'.c{center_denoise_output}'
  model: str = f'.{model_shortname[model_name]}' if model_name in model_shortname else ''
  depth: str = '' if half else '.fp32'
  cfg_str: str = '' if (cfg is None or cfg == 7.5) else f'.cfg{cfg:05.2f}'
  qual: str = '' if filename_qualifier is None else f'.{filename_qualifier}'
  return f"{base_count+ix_in_batch:05}.{seed}{cfg_str}{mim}{dynpct}{cen}{model}{depth}{qual}"