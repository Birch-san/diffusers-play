from cv2 import Mat, cvtColor
import cv2
from dataclasses import dataclass
from enum import Enum, auto
import fnmatch
import torch
from os import listdir, makedirs
from os.path import join, exists
from torch import Tensor, IntTensor, FloatTensor, inference_mode, sub, zeros, from_numpy, load, save, stack
from torch.nn import Linear, SiLU, ModuleList, MSELoss, L1Loss, Module
from torch.optim import AdamW
from torchvision.io import write_png
from typing import List, NamedTuple, Callable, Tuple
from helpers.device import get_device_type, DeviceLiteral

int8_iinfo = torch.iinfo(torch.int8)
int8_range = int8_iinfo.max-int8_iinfo.min
int8_half_range = int8_range / 2

device_type: DeviceLiteral = get_device_type()
device = torch.device(device_type)

model_shortname = 'sd1.5'
assets_dir = f'out_learn_{model_shortname}'
samples_dir=join(assets_dir, 'samples')
processed_train_data_dir=join(assets_dir, 'processed_train_data')
processed_test_data_dir=join(assets_dir, 'processed_test_data')
latents_dir=join(assets_dir, 'pt')
test_latents_dir=join(assets_dir, 'test_pt')
test_samples_dir=join(assets_dir, 'test_png')
predictions_dir=join(assets_dir, 'test_pred5')
science_dir=join(assets_dir, 'science')
weights_dir=join(assets_dir, 'weights')
for path_ in [weights_dir, processed_train_data_dir, processed_test_data_dir, predictions_dir, science_dir]:
  makedirs(path_, exist_ok=True)

weights_path = join(weights_dir, f'approx_decoder5_{model_shortname}.pt')

class Decoder5(Module):
  in_proj: Linear
  hidden_layers: ModuleList
  out_proj: Linear
  def __init__(self, hidden_layer_count: int, inner_dim: int) -> None:
    super().__init__()
    self.in_proj = Linear(4, inner_dim)
    make_nonlin = SiLU
    self.nonlin = make_nonlin()
    self.hidden_layers = ModuleList([
      layer for layer in (Linear(inner_dim, inner_dim), make_nonlin()) for _ in range(hidden_layer_count)
    ])
    self.out_proj = Linear(inner_dim, 3)
  
  def forward(self, sample: Tensor) -> Tensor:
    sample: Tensor = self.in_proj(sample)
    sample: Tensor = self.nonlin(sample)
    for layer in self.hidden_layers:
      sample: Tensor = layer.forward(sample)
    sample: Tensor = self.out_proj(sample)
    return sample

class Mode(Enum):
  Train = auto()
  Test = auto()
  VisualizeLatents = auto()
mode = Mode.Train
test_after_train=True
resume_training=False

model = Decoder5(hidden_layer_count=1, inner_dim=12)
if exists(weights_path) and resume_training or mode is not Mode.Train:
  model.load_state_dict(load(weights_path, weights_only=True))
model = model.to(device)

training_dtype = torch.float32

# epochs = 3000
# epochs = 1000
epochs = 400

l2_loss = MSELoss()
l1_loss = L1Loss()
optim = AdamW(model.parameters(), lr=5e-2)

@dataclass
class Dataset:
  latents: FloatTensor
  samples: FloatTensor

GetFileNames = Callable[[], List[str]]

# there's so few of these we may as well keep them all resident in memory
def get_latents(in_dir: str, out_dir: str, get_latent_filenames: GetFileNames) -> FloatTensor:
  latents_path = join(out_dir, 'latents.pt')
  try:
    latents: FloatTensor = load(latents_path, map_location=device, weights_only=True)
    return latents
  except:
    latents: List[FloatTensor] = [load(join(in_dir, pt), map_location=device, weights_only=True) for pt in get_latent_filenames()]
    latents: FloatTensor = stack(latents)
    save(latents, latents_path)
    return latents
  

# there's so few of these we may as well keep them all resident in memory
def get_resized_samples(in_dir: str, out_dir: str, get_sample_filenames: GetFileNames) -> IntTensor:
  filename_stem = 'resized_samples'
  resized_path_png = join(out_dir, f'{filename_stem}.png')
  resized_path_pt = join(out_dir, f'{filename_stem}.pt')
  try:
    resized: IntTensor = load(resized_path_pt, map_location=device, weights_only=True)
    return resized
  except:
    # load/resize via cv2 instead of torchvision because cv2 has INTER_AREA downsample for best PSNR.
    # CUDA-accelerating this requires building opencv from source. instead we'll cache the result on-disk.
    sample_filenames: List[str] = get_sample_filenames()
    mats: List[Mat] = [cv2.imread(join(in_dir, sample_filename)) for sample_filename in sample_filenames]
    first, *_ = mats
    height, width, _ = first.shape
    vae_scale_factor = 8
    scaled_height = height//vae_scale_factor
    scaled_width = width//vae_scale_factor
    dsize: Tuple[int, int] = (scaled_height, scaled_width)
    resizeds: List[Mat] = [cv2.resize(mat, dsize, interpolation=cv2.INTER_AREA) for mat in mats]
    for img, filename in zip(resizeds, sample_filenames):
      cv2.imwrite(join(out_dir, filename), img)
    tensors: List[IntTensor] = [from_numpy(cvtColor(resized, cv2.COLOR_BGR2RGB)) for resized in resizeds]
    tensor: IntTensor = stack(tensors)
    save(tensor, resized_path_pt)
    write_png(tensor.permute(3, 0, 1, 2).flatten(1, end_dim=2), resized_path_png)
    return tensor.to(device)

def get_data() -> Dataset:
  get_latent_filenames: GetFileNames = lambda: fnmatch.filter(listdir(latents_dir), f"*.pt")
  get_sample_filenames: GetFileNames = lambda: [latent_path.replace('pt', 'png') for latent_path in get_latent_filenames()]

  latents: FloatTensor = get_latents(latents_dir, processed_train_data_dir, get_latent_filenames)
  # channels-last is good for linear layers
  latents = latents.permute(0, 2, 3, 1).contiguous()
  latents = latents.to(training_dtype)
  
  samples: IntTensor = get_resized_samples(samples_dir, processed_train_data_dir, get_sample_filenames)
  samples: FloatTensor = samples.to(training_dtype)
  samples = samples-int8_half_range
  samples = samples/int8_half_range
  return Dataset(
    latents=latents,
    samples=samples,
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

zero = zeros(1, device=device, dtype=training_dtype)

def loss_fn(input: FloatTensor, target: FloatTensor) -> LossComponents:
  # return l2_loss(input, target) + 0.05 * l1_loss(input, target) + 0.05 * (input.abs().max() - 1).clamp(min=0)**2
  # return l2_loss(input, target) #+ 0.025 * (input.abs().max() - 1).clamp(min=0)**2
  # return 0.9 * l2_loss(input, target) + 0.1 * l1_loss(input, target) #+ 0.025 * (input.abs().max() - 1).clamp(min=0)**2
  l2 = l2_loss(input, target)
  l2_scaled = 1. * l2
  l1 = l1_loss(input, target)
  l1_scaled = 0. * l1
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
  out: Tensor = model(dataset.latents)
  loss_components = loss_fn(out, dataset.samples)
  loss, _ = loss_components
  optim.zero_grad()
  loss.backward()
  optim.step()
  if epoch % 100 == 0:
    loss_desc: str = describe_loss(loss_components)
    print(f'epoch {epoch:04d}, {loss_desc}')

@inference_mode(True)
def test():
  get_latent_filenames: GetFileNames = lambda: fnmatch.filter(listdir(test_latents_dir), f"*.pt")
  get_sample_filenames: GetFileNames = lambda: [latent_path.replace('pt', 'png') for latent_path in get_latent_filenames()]

  latents: FloatTensor = get_latents(test_latents_dir, processed_test_data_dir, get_latent_filenames)
  # linear layers expect channels-last
  latents = latents.permute(0, 2, 3, 1).contiguous()
  latents = latents.to(training_dtype)

  true_samples: IntTensor = get_resized_samples(test_samples_dir, processed_test_data_dir, get_sample_filenames)
  true_samples: FloatTensor = true_samples.to(training_dtype)
  true_samples = true_samples-int8_half_range
  true_samples = true_samples/int8_half_range

  model.eval() # might be redundant due to inference mode but whatever

  predicts: Tensor = model.forward(latents)
  
  loss_components = loss_fn(predicts, true_samples)
  loss_desc: str = describe_loss(loss_components)
  print(f'validation {loss_desc}')

  # channels-first for saving
  predicts = predicts.permute(0, 3, 1, 2)

  predicts = predicts + 1
  predicts = predicts * int8_half_range
  predicts: Tensor = predicts.round().clamp(0, 255).to(dtype=torch.uint8).cpu()
  for prediction, sample_filename in zip(predicts, get_sample_filenames()):
    write_png(prediction, join(predictions_dir, sample_filename))

match(mode):
  case Mode.Train:
    dataset: Dataset = get_data()
    for epoch in range(epochs):
      train(epoch, dataset)
    del dataset
    save(model.state_dict(), weights_path)
    if test_after_train:
      test()
  case Mode.Test:
    test()
  case Mode.VisualizeLatents:
    input_path = '00228.4209087706.cfg07.50.pt'
    sample: Tensor = load(join(test_latents_dir, input_path), map_location=device, weights_only=True)
    for ix, channel in enumerate(sample):
      centered: Tensor = channel-(channel.max()+channel.min())/2
      norm: Tensor = centered / centered.abs().max()
      rescaled = (norm + 1)*(255/2)
      write_png(rescaled.unsqueeze(0).to(dtype=torch.uint8).cpu(), join(science_dir, input_path.replace('pt', f'.{ix}.png')))