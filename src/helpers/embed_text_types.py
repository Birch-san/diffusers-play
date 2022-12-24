from torch import Tensor
from typing import Callable, Union, Iterable
from typing_extensions import TypeAlias

Prompts: TypeAlias = Union[str, Iterable[str]]
Embed: TypeAlias = Callable[[Prompts], Tensor]