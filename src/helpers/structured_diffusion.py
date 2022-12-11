from dataclasses import dataclass
from functools import partial
import torch
from torch import Tensor, LongTensor, tensor
from nltk.tree import Tree
from typing import List, Optional, Protocol
import re

from .embed_text import Embed
from .tokenize_text import CountTokens
from .prompt_type import Prompts


def get_deepest_nps(tree: Tree) -> List[Tree]:
  found = []
  for subtree in tree:
    if isinstance(subtree, Tree):
      found.extend(get_deepest_nps(subtree))
  
  if not found and tree.label() == 'NP':
    found.append(tree)
  return found

def brace_comma_delimit(elems: List[str]) -> str:
  return '[%s]' % ', '.join(elems)

def align_np(embed: Tensor, np_embed: Tensor, np_start_ix: LongTensor, np_end_ix: LongTensor) -> Tensor:
  embed = embed.clone()
  embed[np_start_ix:np_end_ix] = np_embed[np_start_ix:np_end_ix]
  return embed

def align_nps(embed: Tensor, np_embeds: Tensor, np_start_ixs: LongTensor, np_end_ixs: LongTensor) -> Tensor:
  # make it cheaper for align_np to clone
  embed = embed.detach()
  align_np_ = partial(align_np, embed)
  return torch.stack([
    # I don't understand why this clone is necessary (each align_np() call modifies their own clone).
    # but without it: I found that the align_np() outputs had almost no token embeddings in common with the nominal embed.
    # maybe it's an MPS bug.
    embed.clone(),
    *map(lambda z: align_np_(*z), zip(np_embeds, np_start_ixs, np_end_ixs))
  ])

@dataclass
class IndexedNounPhrases():
  noun_phrases: List[str]
  start_ixs: LongTensor
  end_ixs: LongTensor

@dataclass
class StructuredEmbedding():
  embeds: Tensor
  uncond: Optional[Tensor]
  np_arities: List[int]

class StructuredEmbed(Protocol):
  def __call__(self, prompts: Prompts, gimme_uncond=False) -> StructuredEmbedding: ...

def get_structured_embedder(embed: Embed, count_tokens: CountTokens, device: torch.device = torch.device('cpu')) -> StructuredEmbed:
  import stanza
  from stanza.models.common.doc import Document, Sentence
  from stanza.models.constituency.parse_tree import Tree as ConstituencyTree
  stanza_batch_delimeter = '\n\n'
  nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency', device=device)

  def fit_noun_phrases_to_prompt(prompt: str, sentence: Sentence) -> IndexedNounPhrases:
    constituency: ConstituencyTree = sentence.constituency
    tree: Tree = Tree.fromstring(str(constituency))
    deepest_nps: List[Tree] = get_deepest_nps(tree)
    np_stanza_tokens: List[List[str]] = [np_.leaves() for np_ in deepest_nps]
    pattern: str = '^(.*)%s(.*)$' % '(.*)'.join(['(%s)' % '\s*'.join([re.escape(token) for token in tokens]) for tokens in np_stanza_tokens])
    matches = re.search(pattern, prompt)
    assert matches is not None, f"Failed to fit noun-phrases back onto original phrase. Used regex pattern: <{pattern}> to match tokens [{brace_comma_delimit([brace_comma_delimit(tokens) for tokens in np_stanza_tokens])}] to prompt <{prompt}>"
    match_groups = matches.groups()
    counts = count_tokens([prompt, *match_groups], device=device)
    counts_len, = counts.shape
    whole_count, part_counts = counts.split((1, counts_len-1))
    whole_count_item, part_counts_sum_item = whole_count.item(), part_counts.sum().item()
    assert whole_count_item == part_counts_sum_item, "Failed to fit noun-phrases back onto original phrase. Whole phrase has {whole_count_item} tokens, but parts added to {part_counts_sum_item} tokens."
    noun_phrase_capture_group_indices = [2*ix+1 for ix in range(0, len(np_stanza_tokens))]
    noun_phrases: List[str] = [match_groups[ix] for ix in noun_phrase_capture_group_indices]
    indices_tensor = tensor(noun_phrase_capture_group_indices, device=device)
    noun_phrase_token_counts: LongTensor = part_counts.index_select(0, indices_tensor)
    # cumsum is a no-op on MPS on some nightlies, including 1.14.0.dev20221105
    # https://github.com/pytorch/pytorch/issues/89784
    part_counts_cumsum: LongTensor = part_counts.cpu().cumsum(0).to(device) if device.type == 'mps' else part_counts.cumsum(0)
    # our token lengths were computed without encoding BOS token. increment by 1 in order to line up with how prompt will be tokenized downstream.
    noun_phrase_start_ixs: LongTensor = part_counts_cumsum.index_select(0, indices_tensor) + 1
    noun_phrase_end_ixs: LongTensor = noun_phrase_start_ixs + noun_phrase_token_counts + 1
    return IndexedNounPhrases(
      noun_phrases=noun_phrases,
      start_ixs=noun_phrase_start_ixs,
      end_ixs=noun_phrase_end_ixs,
    )

  def get_structured_embed(cond_prompts: Prompts, gimme_uncond=False) -> StructuredEmbedding:
    if isinstance(cond_prompts, str):
      cond_prompts: List[str] = [cond_prompts]

    for prompt in cond_prompts:
      assert not prompt.__contains__(stanza_batch_delimeter)

    prompt_batch: str = stanza_batch_delimeter.join(cond_prompts)
    doc: Document = nlp.process(prompt_batch)

    indexed_nps: List[IndexedNounPhrases] = [fit_noun_phrases_to_prompt(*z) for z in zip(cond_prompts, doc.sentences)]
    nps: List[List[str]] = [inp.noun_phrases for inp in indexed_nps]
    np_arities: List[int] = [len(nps) for nps in nps]
    np_start_ixs: List[LongTensor] = [inp.start_ixs for inp in indexed_nps]
    np_end_ixs: List[LongTensor] = [inp.end_ixs for inp in indexed_nps]
    nps_flattened: List[str] = [noun_phrase for nps in nps for noun_phrase in nps]
    uncond_prompts: List[str] = [''] if gimme_uncond else []
    prompts: List[str] = [*uncond_prompts, *cond_prompts]
    embeds: Tensor = embed([*prompts, *nps_flattened])
    embeds_nominal, *np_embeds = embeds.split((len(prompts), *np_arities))
    uncond_embed, cond_embeds = embeds_nominal.split((1, embeds_nominal.size(0)-1)) if gimme_uncond else (
      None,
      embeds_nominal
    )
    aligned_embeds: Tensor = torch.cat([align_nps(*e) for e in zip(cond_embeds, np_embeds, np_start_ixs, np_end_ixs)])
    return StructuredEmbedding(
      embeds=aligned_embeds,
      uncond=uncond_embed,
      np_arities=np_arities,
    )

  return get_structured_embed