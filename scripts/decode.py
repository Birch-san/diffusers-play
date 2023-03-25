from dataclasses import dataclass
from os import listdir, path, makedirs
from torch.nn import Linear, SiLU
from torch import tensor, Tensor, IntTensor, FloatTensor, inference_mode
from helpers.device import get_device_type, DeviceLiteral
import fnmatch
import torch
from PIL import Image
from PIL.Image import Resampling
from typing import List
import numpy as np
from torchvision.io import read_image, write_png
from torchvision.transforms.functional import resize, InterpolationMode
from torch.nn import MSELoss, L1Loss, Module
from torch.optim import AdamW
from enum import Enum, auto
from torch.nn.functional import normalize

int8_iinfo = torch.iinfo(torch.int8)
int8_range = int8_iinfo.max-int8_iinfo.min
int8_half_range = int8_range / 2

device_type: DeviceLiteral = get_device_type()
device = torch.device(device_type)

assets_dir = 'out_learn_wd1.3'
samples_dir=path.join(assets_dir, 'samples')
inputs_dir=path.join(assets_dir, 'pt')
test_inputs_dir=path.join(assets_dir, 'test_pt')
predictions_dir=path.join(assets_dir, 'test_pred3')
science_dir=path.join(assets_dir, 'science')
weights_dir=path.join(assets_dir, 'weights')
for path_ in [weights_dir, predictions_dir, science_dir]:
  makedirs(path_, exist_ok=True)

weights_path = path.join(weights_dir, "decoder3.pt")

class Decoder3(Module):
  lin: Linear
  def __init__(self, inner_dim: int) -> None:
    super().__init__()
    self.lin1 = Linear(4, inner_dim)
    self.nonlin = SiLU()
    self.lin2 = Linear(inner_dim, 3)
  
  def forward(self, sample: Tensor) -> Tensor:
    sample: Tensor = self.lin1(sample)
    sample: Tensor = self.nonlin(sample)
    sample: Tensor = self.lin2(sample)
    return sample

class Mode(Enum):
  Train = auto()
  Test = auto()
  VisualizeLatents = auto()
mode = Mode.Train
test_after_train=True
resume_training=False

model = Decoder3(inner_dim=12)
if path.exists(weights_path) and resume_training or mode is not Mode.Train:
  model.load_state_dict(torch.load(weights_path, weights_only=True))
model = model.to(device)

training_dtype = torch.float32

epochs = 3000

l2_loss = MSELoss(reduction='mean')
l1_loss = L1Loss()
optim = AdamW(model.parameters(), lr=5e-2)

@dataclass
class Dataset:
  train_inputs: Tensor
  train_outputs: Tensor

def get_data() -> Dataset:
  train_input_paths: List[str] = fnmatch.filter(listdir(inputs_dir), f"*.pt")
  # there's so few of these we may as well keep them all resident in memory
  train_inputs: Tensor = torch.stack([torch.load(path.join(inputs_dir, pt), map_location=device, weights_only=True).flatten(-2).transpose(-2,-1).to(training_dtype) for pt in train_input_paths])
  images: List[IntTensor] = [read_image(path.join(samples_dir, input.replace('pt', 'png'))).to(device=device) for input in train_input_paths]
  first, *_ = images
  _, height, width = first.shape
  vae_scale_factor = 8
  scaled_height = height//vae_scale_factor
  scaled_width = width//vae_scale_factor
  train_outputs: FloatTensor = resize(torch.stack(images), [scaled_height, scaled_width], InterpolationMode.BICUBIC).flatten(-2).transpose(-2,-1).to(training_dtype).contiguous()
  train_outputs = train_outputs-int8_half_range
  train_outputs = train_outputs/int8_half_range
  return Dataset(
    train_inputs=train_inputs,
    train_outputs=train_outputs,
  )

def loss_fn(input: FloatTensor, target: FloatTensor) -> FloatTensor:
  return l2_loss(input, target) + 0.1 * (input.abs().max() - 1).clamp(min=0)**2

def train(epoch: int, dataset: Dataset):
  model.train()
  out: Tensor = model(dataset.train_inputs)
  loss: Tensor = loss_fn(out, dataset.train_outputs)
  optim.zero_grad()
  loss.backward()
  optim.step()
  if epoch % 100 == 0:
    print(f'epoch {epoch:04d}, loss: {loss.item():.2f}, abs().max(): {out.abs().max().item():.2f}')

@inference_mode(True)
def test():
  test_input_paths: List[str] = fnmatch.filter(listdir(test_inputs_dir), f"*.pt")
  test_inputs: Tensor = torch.stack([torch.load(path.join(test_inputs_dir, pt), map_location=device, weights_only=True).flatten(-2).transpose(-2,-1).to(training_dtype) for pt in test_input_paths])
  model.eval() # maybe inference mode does that for us
  predicts: Tensor = model(test_inputs)
  latent_height: int = 64
  # latent_height: int = 128
  # latent_width: int = 97
  predicts: Tensor = predicts.transpose(2,1).unflatten(2, (latent_height, -1)).contiguous()
  predicts = predicts + 1
  predicts = predicts * int8_half_range
  predicts: Tensor = predicts.round().clamp(0, 255).to(dtype=torch.uint8).cpu()
  for prediction, input_path in zip(predicts, test_input_paths):
    write_png(prediction, path.join(predictions_dir, input_path.replace('pt', 'png')))

match(mode):
  case Mode.Train:
    dataset: Dataset = get_data()
    for epoch in range(epochs):
      train(epoch, dataset)
    del dataset
    torch.save(model.state_dict(), weights_path)
    if test_after_train:
      test()
  case Mode.Test:
    test()
  case Mode.VisualizeLatents:
    input_path = '00228.4209087706.cfg07.50.pt'
    sample: Tensor = torch.load(path.join(test_inputs_dir, input_path), map_location=device, weights_only=True)
    for ix, channel in enumerate(sample):
      centered: Tensor = channel-(channel.max()+channel.min())/2
      norm: Tensor = centered / centered.abs().max()
      rescaled = (norm + 1)*(255/2)
      write_png(rescaled.unsqueeze(0).to(dtype=torch.uint8).cpu(), path.join(science_dir, input_path.replace('pt', f'.{ix}.png')))