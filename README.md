```bash
git submodule update --init --recursive
conda create -n diffnightly -c pytorch-nightly -c defaults python==3.10.6 pytorch
conda activate diffnightly
# and then pip install anything that you don't have, until the errors stop
```