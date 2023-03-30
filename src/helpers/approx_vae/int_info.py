import torch

int8_iinfo = torch.iinfo(torch.int8)
int8_range = int8_iinfo.max-int8_iinfo.min
int8_half_range = int8_range / 2