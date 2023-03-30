from enum import Enum, auto
import fnmatch
import torch
from os import listdir, makedirs
from os.path import join, exists
from torch import IntTensor, FloatTensor, inference_mode, load, save
from torch.optim import AdamW
from helpers.device import get_device_type, DeviceLiteral
from helpers.approx_vae.encoder import Encoder
from helpers.approx_vae.dataset import get_data, Dataset
from helpers.approx_vae.get_file_names import GetFileNames
from helpers.approx_vae.get_latents import get_latents
from helpers.approx_vae.resize_samples import get_resized_samples
from helpers.approx_vae.int_info import int8_half_range
from helpers.approx_vae.loss import loss_fn, describe_loss, LossComponents

device_type: DeviceLiteral = get_device_type()
device = torch.device(device_type)

model_shortname = 'sd1.5'
repo_root='..'
assets_dir = f'out_learn_{model_shortname}'
samples_dir=join(assets_dir, 'samples')
processed_train_data_dir=join(assets_dir, 'processed_train_data')
processed_test_data_dir=join(assets_dir, 'processed_test_data')
latents_dir=join(assets_dir, 'pt')
test_latents_dir=join(assets_dir, 'test_pt')
test_samples_dir=join(assets_dir, 'test_png')
weights_dir=join(repo_root, 'approx_vae')
for path_ in [processed_train_data_dir, processed_test_data_dir]:
  makedirs(path_, exist_ok=True)

weights_path = join(weights_dir, f'encoder_{model_shortname}.pt')

class Mode(Enum):
  Train = auto()
  Test = auto()
mode = Mode.Train
test_after_train=True
resume_training=False

model = Encoder()
if exists(weights_path) and resume_training or mode is not Mode.Train:
  model.load_state_dict(load(weights_path, weights_only=True))
model = model.to(device)

training_dtype = torch.float32

# epochs = 3000
# epochs = 1000
epochs = 400

optim = AdamW(model.parameters(), lr=5e-2)

def train(epoch: int, dataset: Dataset):
  model.train()
  out: FloatTensor = model(dataset.samples)
  loss_components: LossComponents = loss_fn(out, dataset.latents)
  loss, _ = loss_components
  optim.zero_grad()
  loss.backward()
  optim.step()
  if epoch % 100 == 0:
    loss_desc: str = describe_loss(loss_components)
    print(f'epoch {epoch:04d}, {loss_desc}')

@inference_mode(True)
def test():
  get_sample_filenames: GetFileNames = lambda: fnmatch.filter(listdir(test_samples_dir), f"*.png")
  get_latent_filenames: GetFileNames = lambda: [sample_path.replace('png', 'pt') for sample_path in get_sample_filenames()]

  samples: IntTensor = get_resized_samples(
    in_dir=test_samples_dir,
    out_dir=processed_test_data_dir,
    get_sample_filenames=get_sample_filenames,
    device=device,
  )
  samples: FloatTensor = samples.to(training_dtype)
  samples = samples-int8_half_range
  samples = samples/int8_half_range

  true_latents: FloatTensor = get_latents(
    in_dir=test_latents_dir,
    out_dir=processed_test_data_dir,
    get_latent_filenames=get_latent_filenames,
    device=device
  )
  true_latents = true_latents.to(training_dtype)

  model.eval() # might be redundant due to inference mode but whatever

  predicted_latents: FloatTensor = model.forward(samples)
  # back to channels-first for comparison with true latents
  predicted_latents = predicted_latents.permute(0, 3, 1, 2)
  
  loss_components = loss_fn(predicted_latents, true_latents)
  loss_desc: str = describe_loss(loss_components)
  print(f'validation {loss_desc}')

match(mode):
  case Mode.Train:
    dataset: Dataset = get_data(
      latents_dir=latents_dir,
      processed_train_data_dir=processed_train_data_dir,
      samples_dir=samples_dir,
      dtype=training_dtype,
    )
    for epoch in range(epochs):
      train(epoch, dataset)
    del dataset
    save(model.state_dict(), weights_path)
    if test_after_train:
      test()
  case Mode.Test:
    test()