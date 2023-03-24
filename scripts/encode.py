from dataclasses import dataclass
from os import listdir, path, makedirs
from torch.nn import Linear
from torch import tensor, Tensor, IntTensor, inference_mode
from helpers.device import get_device_type, DeviceLiteral
import fnmatch
import torch
from PIL import Image
from PIL.Image import Resampling
from typing import List
import numpy as np
from torchvision.io import read_image, write_png
from torchvision.transforms.functional import resize, InterpolationMode
from torch.nn import MSELoss, Module
from torch.optim import AdamW
from enum import Enum, auto
from torch.nn.functional import normalize

device_type: DeviceLiteral = get_device_type()
device = torch.device(device_type)

assets_dir = 'out_learn_sd1.5'
samples_dir=path.join(assets_dir, 'samples')
latents_dir=path.join(assets_dir, 'pt')
test_sample_inputs_dir=path.join(assets_dir, 'test_png')
predictions_dir=path.join(assets_dir, 'test_pred')
science_dir=path.join(assets_dir, 'science')
weights_dir=path.join(assets_dir, 'weights')
for path_ in [weights_dir, predictions_dir, science_dir]:
  makedirs(path_, exist_ok=True)

weights_path = path.join(weights_dir, 'encoder_model.pt')

class Encoder(Module):
  lin: Linear
  def __init__(self) -> None:
    super().__init__()
    self.lin = Linear(3, 4, True)
  
  def forward(self, input: Tensor) -> Tensor:
    output: Tensor = self.lin(input)
    return output

model = Encoder()
if path.exists(weights_path):
  model.load_state_dict(torch.load(weights_path, weights_only=True))
model = model.to(device)

training_dtype = torch.float32

epochs = 30000

loss_fn = MSELoss()
optim = AdamW(model.parameters(), lr=9e-1)

@dataclass
class Dataset:
  sample_inputs: Tensor
  latent_outputs: Tensor

def get_data() -> Dataset:
  sample_input_paths: List[str] = fnmatch.filter(listdir(samples_dir), f"*.png")
  # there's so few of these we may as well keep them all resident in memory
  latent_outputs: Tensor = torch.stack([torch.load(path.join(latents_dir, sample_path.replace('.png', '.pt')), map_location=device, weights_only=True).flatten(-2).transpose(-2,-1).to(training_dtype) for sample_path in sample_input_paths])
  images: List[IntTensor] = [read_image(path.join(samples_dir, sample_path)).to(device=device) for sample_path in sample_input_paths]
  first, *_ = images
  _, height, width = first.shape
  vae_scale_factor = 8
  scaled_height = height//vae_scale_factor
  scaled_width = width//vae_scale_factor
  sample_inputs: Tensor = resize(torch.stack(images), [scaled_height, scaled_width], InterpolationMode.BICUBIC).flatten(-2).transpose(-2,-1).to(training_dtype).contiguous()
  return Dataset(
    sample_inputs=sample_inputs,
    latent_outputs=latent_outputs,
  )

def train(epoch: int, dataset: Dataset):
  model.train()
  out: Tensor = model(dataset.sample_inputs)
  loss: Tensor = loss_fn(out, dataset.latent_outputs)
  optim.zero_grad()
  loss.backward()
  optim.step()
  if epoch % 100 == 0:
    print(f'epoch {epoch:04d}, loss: {loss.item():.2f}')

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