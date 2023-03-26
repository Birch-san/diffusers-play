from dataclasses import dataclass
from os import listdir, path, makedirs
from torch import Tensor, IntTensor, FloatTensor, inference_mode, sub
from helpers.device import get_device_type, DeviceLiteral
import fnmatch
import torch
from typing import List, NamedTuple
from torchvision.io import read_image, write_png
from torchvision.transforms.functional import resize, InterpolationMode
from torch.nn import MSELoss, L1Loss, Module, Conv2d
from torch.optim import AdamW
from enum import Enum, auto

int8_iinfo = torch.iinfo(torch.int8)
int8_range = int8_iinfo.max-int8_iinfo.min
int8_half_range = int8_range / 2

device_type: DeviceLiteral = get_device_type()
device = torch.device(device_type)

assets_dir = 'out_learn_sd1.5'
samples_dir=path.join(assets_dir, 'samples')
inputs_dir=path.join(assets_dir, 'pt')
test_inputs_dir=path.join(assets_dir, 'test_pt')
predictions_dir=path.join(assets_dir, 'test_pred4')
science_dir=path.join(assets_dir, 'science')
weights_dir=path.join(assets_dir, 'weights')
for path_ in [weights_dir, predictions_dir, science_dir]:
  makedirs(path_, exist_ok=True)

weights_path = path.join(weights_dir, "decoder4.pt")

class Decoder4(Module):
  proj: Conv2d
  def __init__(self) -> None:
    super().__init__()
    self.proj = Conv2d(4, 3, kernel_size=3, padding=1)

  def forward(self, sample: Tensor) -> Tensor:
    sample: Tensor = self.proj(sample)
    return sample

class Mode(Enum):
  Train = auto()
  Test = auto()
  VisualizeLatents = auto()
mode = Mode.Train
test_after_train=True
resume_training=False

model = Decoder4()
if path.exists(weights_path) and resume_training or mode is not Mode.Train:
  model.load_state_dict(torch.load(weights_path, weights_only=True))
model = model.to(device)

training_dtype = torch.float32

epochs = 300

l2_loss = MSELoss()
l1_loss = L1Loss()
optim = AdamW(model.parameters(), lr=5e-2)

@dataclass
class Dataset:
  train_inputs: Tensor
  train_outputs: Tensor

def get_data() -> Dataset:
  train_input_paths: List[str] = fnmatch.filter(listdir(inputs_dir), f"*.pt")
  # there's so few of these we may as well keep them all resident in memory
  train_inputs: Tensor = torch.stack([torch.load(path.join(inputs_dir, pt), map_location=device, weights_only=True).to(training_dtype) for pt in train_input_paths])
  images: List[IntTensor] = [read_image(path.join(samples_dir, input.replace('pt', 'png'))).to(device=device) for input in train_input_paths]
  first, *_ = images
  _, height, width = first.shape
  vae_scale_factor = 8
  latent_height = height//vae_scale_factor
  latent_width = width//vae_scale_factor
  train_outputs: FloatTensor = resize(torch.stack(images), [latent_height, latent_width], InterpolationMode.BICUBIC).to(training_dtype)
  train_outputs = train_outputs-int8_half_range
  train_outputs = train_outputs/int8_half_range
  return Dataset(
    train_inputs=train_inputs,
    train_outputs=train_outputs,
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
  # TODO: better summing of absmax loss
  # return l2_loss(input, target) + 0.05 * l1_loss(input, target) + 0.05 * (input.abs().max() - 1).clamp(min=0)**2
  # return l2_loss(input, target) #+ 0.025 * (input.abs().max() - 1).clamp(min=0)**2
  # return 0.9 * l2_loss(input, target) + 0.1 * l1_loss(input, target) #+ 0.025 * (input.abs().max() - 1).clamp(min=0)**2
  l2 = l2_loss(input, target)
  l2_scaled = 1. * l2
  l1 = l1_loss(input, target)
  l1_scaled = 0.33 * l1
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
  out: Tensor = model(dataset.train_inputs)
  loss_components = loss_fn(out, dataset.train_outputs)
  loss, _ = loss_components
  optim.zero_grad()
  loss.backward()
  optim.step()
  if epoch % 100 == 0:
    loss_desc: str = describe_loss(loss_components)
    print(f'epoch {epoch:04d}, {loss_desc}')

@inference_mode(True)
def test():
  test_input_paths: List[str] = fnmatch.filter(listdir(test_inputs_dir), f"*.pt")
  test_inputs: Tensor = torch.stack([torch.load(path.join(test_inputs_dir, pt), map_location=device, weights_only=True).to(training_dtype) for pt in test_input_paths])
  model.eval() # maybe inference mode does that for us
  predicts: Tensor = model(test_inputs)
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