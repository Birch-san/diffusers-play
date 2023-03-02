import torch
from transformers import CLIPTextModel, PreTrainedTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, TensorType
from torch import FloatTensor, LongTensor, BoolTensor, no_grad
from torch.nn import functional as F
from .embed_text_types import Embed, EmbeddingAndMask, Prompts
from typing import List, Iterable

def _get_segment_split_indices(segments: int, segment_max_length: int) -> List[int]:
  if segments == 1:
    return [segment_max_length]
  return [
    # retain first segment
    segment_max_length,
    *(1, segment_max_length-1)*(segments-1), # discard BOS embed from any remaining segments; it's a duplicate of BOS from first segment
  ]

def _without_token_embeddings_at_segment_seams(spliced: FloatTensor, split_indices: Iterable[int]) -> FloatTensor:
  return torch.cat(
    spliced.split(split_indices, dim=1)[::2],
    dim=1,
  )

# a *much* simpler implementation exists at an older commit here:
#   https://github.com/Birch-san/diffusers-play/blob/f79064dbf5d236921fe2f2ff8c4b5c1fd96b84c4/src/helpers/clip_embed_text.py
# or even simpler (if you don't need attention masks) here:
#   https://github.com/Birch-san/diffusers-play/blob/823f6484328721fac974ed4e1f3822a4c0509174/src/helpers/clip_embed_text.py
# most complexity here exists to support waifu-diffusion's triple-length captions:
#   - split a prompt into segments that fit into CLIP's context length (77)
#   - embed each segment individually
#   - join the embeddings back together, removing the extraneous (PAD/EOS, BOS) embedding at the seams between each segments
#   - we try to do this with minimal conditions, loops, special cases, mutation
#     - otherwise, the code becomes very branchy, like this:
#       https://github.com/waifu-diffusion/network-trainer/blob/7a2d189bf20203c3247e040b5def8c4c21fd166b/finetuner.py
def get_embedder(
  tokenizer: PreTrainedTokenizer,
  text_encoder: CLIPTextModel,
  subtract_hidden_state_layers = 0,
  max_context_segments = 1,
) -> Embed:
  def embed(prompts: Prompts) -> EmbeddingAndMask:
    max_len_incl_special=tokenizer.model_max_length
    max_len_excl_special=max_len_incl_special-2
    tokens: BatchEncoding = tokenizer(
      prompts,
      padding=PaddingStrategy.MAX_LENGTH,
      max_length=max_context_segments*max_len_excl_special,
      return_tensors=TensorType.PYTORCH,
      return_attention_mask=True,
      return_length=True,
      add_special_tokens=False,
    )
    # you can see how the words get split into tokens like so:
    # [tokenizer.convert_ids_to_tokens([tokenizer.bos_token_id, *tokens.input_ids[ix][:tokens.length[ix]], tokenizer.eos_token_id]) for ix in range(len(tokens.input_ids))]
    # tokenizer.convert_ids_to_tokens(tokenizer('hey').input_ids)
    device=text_encoder.device
    text_input_ids: LongTensor = tokens.input_ids.to(device)
    prompt_count = text_input_ids.size(0)
    token_lengths: LongTensor = tokens.length.to(device)
    bos_t = torch.full((1,), tokenizer.bos_token_id, device=device)
    eos_t = torch.full_like(bos_t, tokenizer.eos_token_id)
    # subtract 1 to convert lengths into the index occupied by that length.
    segments_needed = (token_lengths-1).clamp(min=0).to('cpu' if token_lengths.device.type == 'mps' else token_lengths.device).max().item()//max_len_excl_special+1
    text_input_ids = text_input_ids.narrow(1, 0, max_len_excl_special*segments_needed)
    attention_mask: BoolTensor = F.pad(
      tokens.attention_mask.to(dtype=torch.bool, device=device).narrow(1, 0, max_len_excl_special*segments_needed),
      # when we embed each segment individually: they'll each gain special tokens, so we need to extend our mask to include those.
      # we allocate a positive mask element for:
      # - BOS+EOS of first segment
      # - EOS of each subsequent segment
      #   - we don't need to allocate mask element for their BOS embeds, since we'll narrow those out of the tensor altogether.
      (2+(segments_needed-1),0),
      'constant',
      1,
    )
    special = F.pad(
      F.pad(
        text_input_ids.unflatten(
          1,
          (-1, max_len_excl_special)
        ),
        (1,0),
        'constant',
        tokenizer.bos_token_id,
      ),
      (0,1),
      'constant',
      tokenizer.pad_token_id,
    ).index_put(
      indices=[
        torch.arange(prompt_count, device=device).repeat_interleave(segments_needed),
        torch.arange(segments_needed, device=device).repeat(prompt_count),
        (token_lengths.unsqueeze(1).expand(-1, segments_needed) - torch.arange(segments_needed, device=device)*max_len_excl_special+1).clamp(1, max_len_incl_special-1).flatten()
      ],
      values=eos_t,
    ).flatten(end_dim=1)
    with no_grad():
      encoder_outputs: BaseModelOutputWithPooling = text_encoder.forward(
        special,
        output_hidden_states=subtract_hidden_state_layers != 0,
        return_dict=True,
      )
      text_embeddings: FloatTensor = encoder_outputs.last_hidden_state if subtract_hidden_state_layers == 0 else (
        text_encoder.text_model.final_layer_norm.forward(encoder_outputs.hidden_states[-1 - subtract_hidden_state_layers])
      )
      spliced = text_embeddings.unflatten(0, (tokens.input_ids.size(0), -1)).flatten(start_dim=1, end_dim=2)
      seamless = _without_token_embeddings_at_segment_seams(
        spliced,
        _get_segment_split_indices(segments_needed, max_len_incl_special)
      )
    return seamless, attention_mask
  return embed
