import torch

from helpers.attention.exponly.fixtures import get_fixtures, Fixtures
from helpers.attention.exponly.reference import reference_attn
from helpers.attention.exponly.exponly import exponly_attn

self_attn = False
half = False
sd2 = True
device = torch.device('mps')

fixtures: Fixtures = get_fixtures(
  self_attn=self_attn,
  half=half,
  sd2=sd2,
  device=device,
)

reference_attn(fixtures, device)
exponly_attn(fixtures, device)