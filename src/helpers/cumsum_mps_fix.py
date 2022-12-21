# monkey-patch cumsum to fallback to CPU
# https://github.com/pytorch/pytorch/issues/89784
import torch
from torch import cumsum, Tensor
torch.cumsum = lambda input, *args, **kwargs: (
  cumsum(input.cpu() if input.device.type == 'mps' else input, *args, **kwargs).to(input.device)
)
orig_cumsum = torch.Tensor.cumsum
def patched_cumsum(self: Tensor, *args, **kwargs):
    return orig_cumsum(self.cpu() if self.device.type == 'mps' else self, *args, **kwargs).to(self.device)
torch.Tensor.cumsum = patched_cumsum

reassuring_message = "monkey-patched cumsum to fallback to CPU, for compatibility on MPS backend; see https://github.com/pytorch/pytorch/issues/89784"