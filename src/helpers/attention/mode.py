from enum import Enum, auto

class AttentionMode(Enum):
    # diffusers default (picks AttnProcessor2_0 if supported by pytorch, otherwise CrossAttnProcessor)
    Standard = auto()
    # CrossAttnProcessor via baddbmm(), bmm()
    Legacy = auto()
    # https://github.com/huggingface/diffusers/issues/1892
    # CrossAttnProcessor configured to chunk attention via torch.narrow()'d baddbmm(), bmm()s ("memory-efficient" in pure PyTorch)
    Chunked = auto()
    # replaces diffusers' CrossAttention layers with torch.nn.MultiheadAttention
    TorchMultiheadAttention = auto()
    # AttnProcessor2_0; computes attention probabilities via torch.nn.functional.scaled_dot_product_attention
    ScaledDPAttn = auto()
    # usual diffusers CrossAttention layer, CrossAttnProcessor via Xformers
    Xformers = auto()