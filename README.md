## Setup Python environment

### Mac

```bash
git submodule update --init --recursive
conda create -n p311 -c pytorch-nightly -c conda-forge -c defaults python==3.11
conda activate p311
pip3 install --upgrade --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

### Linux + CUDA

Install Python 3.11 + CUDA 11.8 + Nvidia drivers 525 + latest pytorch + torchvision + xformers like so:  
https://gist.github.com/Birch-san/8ec1f5073b117737cda86a70b01973ba

## Install dependencies

Having activated your conda env, install dependencies:

```bash
cd src/k-diffusion
pip install -r requirements.txt
cd ../..
cd src/diffusers
# strictly speaking it may be sufficient to just build rather than install, since we're gonna PYTHONPATH diffusers anyway
python setup.py install
cd ../..
# and everything else we missed
pip install transformers safetensors easing-functions
```

Treat yourself to ipython:

```bash
pip install ipython
```

CUDA-only:

```bash
pip install triton
```

## Run

Invoke play.py like so:

```bash
PYTHONPATH=src/diffusers/src:src/k-diffusion:src python scripts/play.py
```

Add `PYTORCH_ENABLE_MPS_FALLBACK=1` to this if you're using Mac.  
Mac users in particular should prefer my fork of k-diffusion (the local git submodule) because it has Mac-specific fixes.

And most of the time this repository points at a fork of diffusers, usually for customizing how attention works (e.g. adding support for cross-attention-only masks, or memory-efficient attention).