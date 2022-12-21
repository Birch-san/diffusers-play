from os import listdir, path, makedirs
from torch.nn import Linear
from torch import tensor, Tensor, inference_mode
from helpers.device import get_device_type, DeviceLiteral
import fnmatch
import torch
from PIL import Image
from PIL.Image import Resampling
from typing import List
import numpy as np
from torchvision.io import read_image, write_png
from torchvision.transforms.functional import resize, InterpolationMode
from torch.nn import L1Loss, Module
from torch.optim import SGD
from enum import Enum, auto
from torch.nn.functional import normalize

device_type: DeviceLiteral = get_device_type()
device = torch.device(device_type)

assets_dir = 'out_learn'
samples_dir=path.join(assets_dir, 'samples')
inputs_dir=path.join(assets_dir, 'pt')
test_inputs_dir=path.join(assets_dir, 'test_pt')
predictions_dir=path.join(assets_dir, 'test_pred')
science_dir=path.join(assets_dir, 'science')
weights_dir=path.join(assets_dir, 'weights')
for path_ in [weights_dir, predictions_dir, science_dir]:
  makedirs(path_, exist_ok=True)

weights_path = path.join(weights_dir, "model.pth")

class Decoder(Module):
  lin: Linear
  def __init__(self) -> None:
    super().__init__()
    self.lin = Linear(4, 3, False)
  
  def forward(self, input: Tensor) -> Tensor:
    output: Tensor = self.lin(input)
    return output

model = Decoder()
if path.exists(weights_path):
  model.load_state_dict(torch.load(weights_path))
model = model.to(device)

training_dtype = torch.float32

epochs = 1000

loss_fn = L1Loss()
optim = SGD(model.parameters(), lr=9e-1)

def train(epoch: int):
  train_input_paths: List[str] = fnmatch.filter(listdir(inputs_dir), f"{5*'[0-9]'}.*.pt")
  # there's so few of these we may as well keep them all resident in memory
  train_inputs: Tensor = torch.stack([torch.load(path.join(inputs_dir, pt), map_location=device).flatten(-2).transpose(-2,-1).to(training_dtype) for pt in train_input_paths])
  train_outputs: Tensor = resize(torch.stack([read_image(path.join(samples_dir, input.replace('pt', 'png'))).to(device=device) for input in train_input_paths]), [64, 64], InterpolationMode.BICUBIC).flatten(-2).transpose(-2,-1).to(training_dtype).contiguous()
  model.train()
  out: Tensor = model(train_inputs)
  loss: Tensor = loss_fn(out, train_outputs)
  optim.zero_grad()
  loss.backward()
  optim.step()
  if epoch % 100 == 0:
    print(f'epoch {epoch:04d}, loss: {loss.item():.2f}')

@inference_mode(True)
def test():
  test_input_paths: List[str] = fnmatch.filter(listdir(test_inputs_dir), f"{5*'[0-9]'}.*.pt")
  test_inputs: Tensor = torch.stack([torch.load(path.join(test_inputs_dir, pt), map_location=device).flatten(-2).transpose(-2,-1).to(training_dtype) for pt in test_input_paths])
  model.eval() # maybe inference mode does that for us
  predicts: Tensor = model(test_inputs)
  predicts: Tensor = predicts.transpose(2,1).unflatten(2, (64, -1)).contiguous()
  predicts: Tensor = predicts.round().clamp(0, 255).to(dtype=torch.uint8).cpu()
  for prediction, input_path in zip(predicts, test_input_paths):
    write_png(prediction, path.join(predictions_dir, input_path.replace('pt', 'png')))

class Mode(Enum):
  Train = auto()
  Test = auto()
  VisualizeLatents = auto()

mode = Mode.Test
match(mode):
  case Mode.Train:
    for epoch in range(epochs):
      train(epoch)
    torch.save(model.state_dict(), weights_path)
  case Mode.Test:
    test()
  case Mode.VisualizeLatents:
    input_path = '00004.3532916755.pt'
    sample: Tensor = torch.load(path.join(test_inputs_dir, input_path), map_location=device)
    for ix, channel in enumerate(sample):
      centered: Tensor = channel-(channel.max()+channel.min())/2
      norm: Tensor = centered / centered.abs().max()
      rescaled = (norm + 1)*(255/2)
      write_png(rescaled.unsqueeze(0).to(dtype=torch.uint8).cpu(), path.join(science_dir, input_path.replace('pt', f'.{ix}.png')))