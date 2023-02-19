import torch
from typing import Union, Literal
from typing_extensions import TypeAlias
from warnings import warn
from sys import platform

DeviceType: TypeAlias = Union[torch.device, str]
DeviceLiteral: TypeAlias = Literal['cuda', 'mps', 'cpu']

def get_device_type() -> DeviceLiteral:
  if(torch.cuda.is_available()):
    return 'cuda'
  if torch.cuda._is_compiled() and platform == 'linux':
    warn("""CUDA unavailable, even though PyTorch was compiled with CUDA support.
This can happen on Linux after waking from sleep:
https://discuss.pytorch.org/t/userwarning-cuda-initialization-cuda-unknown-error-this-may-be-due-to-an-incorrectly-set-up-environment-e-g-changing-env-variable-cuda-visible-devices-after-program-start-setting-the-available-devices-to-be-zero/129335/5
You could try:
sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm
""")
  if(torch.backends.mps.is_available()):
    return 'mps'
  warn('Falling back to CPU device -- GPU acceleration capability not detected via CUDA or MPS backends.')
  return 'cpu'