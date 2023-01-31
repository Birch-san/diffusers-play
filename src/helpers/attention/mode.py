from enum import Enum, auto

class AttentionMode(Enum):
    Standard = auto()
    # https://github.com/huggingface/diffusers/issues/1892
    Chunked = auto()
    TorchMultiheadAttention = auto()
    Xformers = auto()