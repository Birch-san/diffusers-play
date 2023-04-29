from torch.nn import Module, ModuleList, Conv2d, SiLU
from torch import Tensor
# import transformer_engine.pytorch as te

# 5x5 conv 3-> 128
# activation function
# 5x5 conv 128->4

class WackyEncoder(Module):
  in_conv: Conv2d
  out_conv: Conv2d
  def __init__(self) -> None:
    super().__init__()
    self.in_conv = Conv2d(3, 128, kernel_size=(5, 5), padding=(2, 2), bias=False)
    self.nonlin = SiLU()
    self.out_conv = Conv2d(128, 4, kernel_size=(5, 5), padding=(2, 2), bias=False)
  
  def forward(self, x: Tensor) -> Tensor:
    x: Tensor = self.in_conv(x)
    x: Tensor = self.nonlin(x)
    x: Tensor = self.out_conv(x)
    return x
