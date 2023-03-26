from dataclasses import dataclass
from os import listdir, path, makedirs
from torch.nn import Linear, Conv2d
from torch import Tensor, IntTensor, inference_mode, FloatTensor, sub
from helpers.device import get_device_type, DeviceLiteral
import fnmatch
import torch
from typing import List, NamedTuple
import numpy as np
from torchvision.io import read_image, write_png
from torchvision.transforms.functional import resize, InterpolationMode
from torch.nn import MSELoss, L1Loss, Module
from torch.optim import AdamW
from enum import Enum, auto
from torch.nn.functional import normalize

device_type: DeviceLiteral = get_device_type()
device = torch.device(device_type)

assets_dir = 'out_learn_sd1.5'
samples_dir=path.join(assets_dir, 'samples')
latents_dir=path.join(assets_dir, 'pt')
test_sample_inputs_dir=path.join(assets_dir, 'test_png')
predictions_dir=path.join(assets_dir, 'test_pred2')
science_dir=path.join(assets_dir, 'science')
weights_dir=path.join(assets_dir, 'weights')
for path_ in [weights_dir, predictions_dir, science_dir]:
  makedirs(path_, exist_ok=True)

weights_path = path.join(weights_dir, 'encoder_model2.pt')

class Encoder2(Module):
  # lin: Linear
  proj: Conv2d
  def __init__(self) -> None:
    super().__init__()
    # self.lin = Linear(3, 4, True)
    self.proj = Conv2d(3, 4, kernel_size=3, padding=1)
  
  def forward(self, input: Tensor) -> Tensor:
    output: Tensor = self.proj(input)
    return output

model = Encoder2()
if path.exists(weights_path):
  model.load_state_dict(torch.load(weights_path, weights_only=True))
model = model.to(device)

training_dtype = torch.float32

epochs = 1000

l2_loss = MSELoss()
l1_loss = L1Loss()
optim = AdamW(model.parameters(), lr=5e-2)

@dataclass
class Dataset:
  sample_inputs: Tensor
  latent_outputs: Tensor

def get_data() -> Dataset:
  sample_input_paths: List[str] = fnmatch.filter(listdir(samples_dir), f"*.png")
  # there's so few of these we may as well keep them all resident in memory
  latent_outputs: Tensor = torch.stack([torch.load(path.join(latents_dir, sample_path.replace('.png', '.pt')), map_location=device, weights_only=True).to(training_dtype) for sample_path in sample_input_paths])
  images: List[IntTensor] = [read_image(path.join(samples_dir, sample_path)).to(device=device) for sample_path in sample_input_paths]
  first, *_ = images
  _, height, width = first.shape
  vae_scale_factor = 8
  scaled_height = height//vae_scale_factor
  scaled_width = width//vae_scale_factor
  sample_inputs: Tensor = resize(torch.stack(images), [scaled_height, scaled_width], InterpolationMode.BICUBIC).to(training_dtype).contiguous()
  return Dataset(
    sample_inputs=sample_inputs,
    latent_outputs=latent_outputs,
  )

@dataclass
class LossBreakdown:
  l2: FloatTensor
  l1: FloatTensor
  range: FloatTensor
  l2_scaled: FloatTensor
  l1_scaled: FloatTensor
  range_scaled: FloatTensor

class LossComponents(NamedTuple):
  total_loss: FloatTensor
  breakdown: LossBreakdown

def describe_loss(loss_components: LossComponents) -> str:
  loss, b = loss_components
  unscaled_components = f'l2: {b.l2.abs().max().item():.2f}, l1: {b.l1.abs().max().item():.2f}, r: {b.range.abs().max().item():.2f}'
  scaled_components = f'l2: {b.l2_scaled.abs().max().item():.2f}, l1: {b.l1_scaled.abs().max().item():.2f}, r: {b.range_scaled.abs().max().item():.2f}'
  return f'loss: {loss.item():.2f}, [u] {unscaled_components} [s] {scaled_components}'

def loss_fn(input: FloatTensor, target: FloatTensor) -> LossComponents:
  l2 = l2_loss(input, target)
  l2_scaled = 1. * l2
  l1 = l1_loss(input, target)
  l1_scaled = 0.66 * l1
  range = sub(input.abs().max(dim=0).values, 1).clamp(min=0).mean()
  range_scaled = 0. * range
  breakdown = LossBreakdown(
    l2=l2,
    l1=l1,
    range=range,
    l2_scaled=l2_scaled,
    l1_scaled=l1_scaled,
    range_scaled=range_scaled,
  )
  total_loss: FloatTensor = l2_scaled + l1_scaled + range_scaled
  return LossComponents(
    total_loss=total_loss,
    breakdown=breakdown,
  )

def train(epoch: int, dataset: Dataset):
  model.train()
  out: Tensor = model(dataset.sample_inputs)
  loss_components = loss_fn(out, dataset.latent_outputs)
  loss, _ = loss_components
  optim.zero_grad()
  loss.backward()
  optim.step()
  if epoch % 100 == 0:
    loss_desc: str = describe_loss(loss_components)
    print(f'epoch {epoch:04d}, {loss_desc}')

@inference_mode(True)
def test(): pass
#   test_sample_input_paths: List[str] = fnmatch.filter(listdir(test_sample_inputs_dir), f"*.png")
#   test_inputs: Tensor = torch.stack([torch.load(path.join(test_sample_inputs_dir, pt), map_location=device, weights_only=True).flatten(-2).transpose(-2,-1).to(training_dtype) for sample_path in test_sample_input_paths])
#   model.eval() # maybe inference mode does that for us
#   predicts: Tensor = model(test_inputs)
#   latent_height: int = 64
#   # latent_height: int = 128
#   # latent_width: int = 97
#   predicts: Tensor = predicts.transpose(2,1).unflatten(2, (latent_height, -1)).contiguous()
#   predicts: Tensor = predicts.round().clamp(0, 255).to(dtype=torch.uint8).cpu()
#   for prediction, input_path in zip(predicts, test_input_paths):
#     write_latents(prediction, path.join(predictions_dir, input_path.replace('pt', 'png')))

class Mode(Enum):
  Train = auto()
  Test = auto()
  VisualizeLatents = auto()

mode = Mode.Train
match(mode):
  case Mode.Train:
    dataset: Dataset = get_data()
    for epoch in range(epochs):
      train(epoch, dataset)
    torch.save(model.state_dict(), weights_path)
  case Mode.Test:
    test()
  case Mode.VisualizeLatents:
    input_path = '00228.4209087706.cfg07.50.pt'
    sample: Tensor = torch.load(path.join(test_sample_inputs_dir, input_path), map_location=device, weights_only=True)
    for ix, channel in enumerate(sample):
      centered: Tensor = channel-(channel.max()+channel.min())/2
      norm: Tensor = centered / centered.abs().max()
      rescaled = (norm + 1)*(255/2)
      write_png(rescaled.unsqueeze(0).to(dtype=torch.uint8).cpu(), path.join(science_dir, input_path.replace('pt', f'.{ix}.png')))