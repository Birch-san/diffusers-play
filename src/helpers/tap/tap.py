from typing import Callable, TypeVar

T = TypeVar('T')
Tap = Callable[[T], None]