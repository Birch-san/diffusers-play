from enum import Enum, auto
import fnmatch
import torch
from os import listdir, makedirs
from os.path import join, dirname, exists
from torch import Tensor, IntTensor, FloatTensor, inference_mode, load, save
from torch.optim import AdamW
from torchvision.io import write_png
from helpers.device import get_device_type, DeviceLiteral
from helpers.approx_vae.decoder import Decoder
from helpers.approx_vae.dataset import get_data, Dataset
from helpers.approx_vae.get_file_names import GetFileNames
from helpers.approx_vae.get_latents import get_latents
from helpers.approx_vae.resize_samples import get_resized_samples
from helpers.approx_vae.int_info import int8_half_range
from helpers.approx_vae.loss import loss_fn, describe_loss, LossComponents
from helpers.approx_vae.visualize_latents import normalize_latents, norm_latents_to_rgb, collage_2by2

device_type: DeviceLiteral = get_device_type()
device = torch.device(device_type)

model_shortname = 'sd1.5'
repo_root=join(dirname(__file__), '..')
assets_dir = f'out_learn_{model_shortname}'
samples_dir=join(assets_dir, 'samples')
processed_train_data_dir=join(assets_dir, 'processed_train_data')
processed_test_data_dir=join(assets_dir, 'processed_test_data')
latents_dir=join(assets_dir, 'pt')
test_latents_dir=join(assets_dir, 'test_pt')
test_samples_dir=join(assets_dir, 'test_png')
predictions_dir=join(assets_dir, 'test_pred5')
science_dir=join(assets_dir, 'science')
weights_dir=join(repo_root, 'approx_vae')
for path_ in [processed_train_data_dir, processed_test_data_dir, predictions_dir, science_dir]:
  makedirs(path_, exist_ok=True)

weights_path = join(weights_dir, f'decoder_{model_shortname}.pt')

class Mode(Enum):
  Train = auto()
  Test = auto()
  VisualizeLatents = auto()
mode = Mode.Train
test_after_train=True
resume_training=False

model = Decoder()
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
  out: Tensor = model(dataset.latents)
  loss_components: LossComponents = loss_fn(out, dataset.samples)
  loss, _ = loss_components
  optim.zero_grad()
  loss.backward()
  optim.step()
  if epoch % 100 == 0:
    loss_desc: str = describe_loss(loss_components)
    print(f'epoch {epoch:04d}, {loss_desc}')

@inference_mode(True)
def test():
  get_latent_filenames: GetFileNames = lambda: sorted(fnmatch.filter(listdir(test_latents_dir), f"*.pt"), key=lambda fname: int(fname.split('.', 1)[0]))
  get_sample_filenames: GetFileNames = lambda: [latent_path.replace('pt', 'png') for latent_path in get_latent_filenames()]

  latents: FloatTensor = get_latents(
    test_latents_dir,
    processed_test_data_dir,
    get_latent_filenames,
    device=device,
  )
  # linear layers expect channels-last
  latents = latents.permute(0, 2, 3, 1).contiguous()
  latents = latents.to(training_dtype)

  true_samples: IntTensor = get_resized_samples(
    in_dir=test_samples_dir,
    out_dir=processed_test_data_dir,
    get_sample_filenames=get_sample_filenames,
    device=device,
  )
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
    dataset: Dataset = get_data(
      latents_dir=latents_dir,
      processed_train_data_dir=processed_train_data_dir,
      samples_dir=samples_dir,
      dtype=training_dtype,
      device=device,
    )
    for epoch in range(epochs):
      train(epoch, dataset)
    del dataset
    save(model.state_dict(), weights_path)
    if test_after_train:
      test()
  case Mode.Test:
    test()
  case Mode.VisualizeLatents:
    latent_filename = '00228.4209087706.cfg07.50.pt'
    input_path = join(test_latents_dir, latent_filename)
    latents: FloatTensor = load(input_path, map_location=device, weights_only=True)
    norm: FloatTensor = normalize_latents(latents)
    rgb: IntTensor = norm_latents_to_rgb(norm)
    for ix, channel in enumerate(rgb.split(1)):
      write_png(channel.cpu(), join(science_dir, latent_filename.replace('.pt', f'.{ix}.png')))
    collage: IntTensor = collage_2by2(rgb, keepdim=True)
    write_png(collage.cpu(), join(science_dir, latent_filename.replace('.pt', '.png')))