import torch
from typing import Union, Literal
from typing_extensions import TypeAlias

DeviceType: TypeAlias = Union[torch.device, str]
DeviceLiteral: TypeAlias = Literal['cuda', 'mps', 'cpu']

def get_device_type() -> DeviceLiteral:
  if(torch.cuda.is_available()):
    return 'cuda'
  if(torch.backends.mps.is_available()):
    return 'mps'
  return 'cpu'