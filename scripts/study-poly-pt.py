import torch
from torch import nn, Tensor, no_grad
from torch.optim import AdamW

class Model(nn.Module):
  layers: nn.Sequential
  def __init__(
    self,
    hidden_width=12
  ) -> None:
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(in_features=1, out_features=hidden_width, bias=True),
      nn.ReLU(),
      nn.Linear(in_features=hidden_width, out_features=1, bias=False),
    )
  def forward(self, x: Tensor) -> Tensor:
    outs = self.layers(x)
    return outs

device=torch.device('mps')
model = Model().to(device)

optim = AdamW(model.parameters(), lr=1e-3)
lfn = nn.MSELoss()

@no_grad()
def fn(x: Tensor) -> Tensor:
  return 2*x**2+3*x+4

for epoch in range(10000):
  x = torch.rand((800, 1), dtype=torch.float32, device=device, requires_grad=True)
  y = fn(x)
  optim.zero_grad()
  y_pred = model(x)
  loss: Tensor = lfn(y_pred, y)
  loss.backward()
  optim.step()

  if epoch % 100 == 0:
    print(f'epoch: {epoch} loss: {loss.item():04f}')
  if epoch % 500 == 0:
    with torch.no_grad():
      x = torch.rand((2, 1), dtype=torch.float32, device=device)
      y = fn(x)
      model = model.eval()
      model = model.train(True)
      y_pred = model(x)
      loss: Tensor = lfn(y_pred, y)
      print('\n'.join([f'  in: {in_.item():04f} out: {out.item():04f} pred: {pred.item():04f} loss: {loss.item():04f}' for in_, out, pred in zip(x, y, y_pred)]))