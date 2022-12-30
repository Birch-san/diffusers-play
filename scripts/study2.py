import torch
from torch import Tensor, no_grad
from torch.nn import Linear, MSELoss, Module, Sequential, ReLU
from torch.optim import AdamW
from helpers.device import get_device_type, DeviceLiteral
from enum import Enum, auto

device_type: DeviceLiteral = get_device_type()
device = torch.device(device_type)

class NN(Module):
	layers: Sequential
	def __init__(self) -> None:
		super().__init__()
		self.layers = Sequential(
			Linear(in_features=1, out_features=10, bias=True),
			ReLU(),
			Linear(in_features=10, out_features=1, bias=True),
			# ReLU(),
		)
	
	def forward(self, input: Tensor) -> Tensor:
		return self.layers(input)

training_dtype = torch.float32

model = NN()
model = model.to(device=device, dtype=training_dtype)

loss_fn = MSELoss()

def train(inputs: Tensor, targets: Tensor) -> Tensor:	
	outs: Tensor = model(inputs)
	loss: Tensor = loss_fn(outs, targets)
	return loss

def trainer() -> None:
	model.train()
	optim = AdamW(model.parameters(), lr=5e-3)
	epochs = 10000
	for epoch in range(epochs):
		# inputs = torch.randint(1, 12, (400, 1), device=device, dtype=training_dtype, requires_grad=True)
		inputs = torch.randn((400, 1), device=device, dtype=training_dtype, requires_grad=True)*12+1
		targets = inputs.detach() ** 2
		loss: Tensor = train(inputs, targets)

		optim.zero_grad()
		loss.backward()
		optim.step()

		if epoch % 100 == 0:
			print(f"epoch {epoch:4d}, loss: {loss.item():.2f} lr: {optim.param_groups[0]['lr']}")
			model.eval()
			test()
			model.train()

@no_grad()
def test():
	inputs = torch.randint(1, 12, (2, 1), device=device, dtype=training_dtype)
	targets = inputs ** 2
	outs: Tensor = model(inputs)
	loss: Tensor = loss_fn(outs, targets)
	print(f'test\n  inputs:\n{inputs}\n  outputs:\n{outs}\n  loss:\n{loss.item():04f}')

class Mode(Enum):
	Train = auto()

mode = Mode.Train

match mode:
	case Mode.Train:
		trainer()
	case _:
		raise f"No such mode {mode}"