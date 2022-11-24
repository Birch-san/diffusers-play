```bash
git submodule update --init --recursive
conda create -n diffnightly -c pytorch-nightly -c defaults python==3.10.6 pytorch
conda activate diffnightly
# and then pip install anything that you don't have, until the errors stop

# invoke play.py like so:
# the MPS flag is just for Mac.
# PYTHONPATH is used to import the src/helpers directory, and to prefer the local git submodules of diffusers and k-diffusion
# but you could use your pip-installed diffusers and k-diffusion if you prefer.
# note: Mac users should probably prefer my k-diffusion, as my fork includes fixes for Mac.
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src/diffusers/src:src/k-diffusion:src python scripts/play.py
```