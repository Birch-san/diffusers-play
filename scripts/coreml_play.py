from argparse import Namespace, ArgumentParser
from python_coreml_stable_diffusion.torch2coreml import main, parser_spec
from python_coreml_stable_diffusion.unet import AttentionImplementations
from coremltools import ComputeUnit
half = True
model_revision = 'fp16' if half else 'a20c448ad20e797115c379fa2418c5ad64a4cd5c'
namespace = Namespace(
  half = half,
  check_output_correctness = True,
  chunk_unet = False,
  convert_text_encoder = False,
  convert_vae_encoder = False,
  convert_vae_decoder = False,
  convert_safety_checker = False,
  convert_unet = True,
  model_version = 'hakurei/waifu-diffusion',
  model_revision = model_revision,
  o = 'out_coreml',
  out_model_name_stem = 'wd13_mycoremltools_hfunet_gpu',
  attention_implementation = AttentionImplementations.ORIGINAL.name,
  latent_h = 512//8,
  latent_w = 512//8,
  compute_unit = ComputeUnit.CPU_AND_GPU.name,
  quantize_weights_to_8bits = False,
  bundle_resources_for_swift_cli = False,
  diffusers_unet = True,
)
if __name__ == "__main__":
  parser: ArgumentParser = parser_spec()
  args: Namespace = parser.parse_args(namespace=namespace)
  main(args)