from torch import Tensor, LongTensor
from nltk.tree import Tree
import numpy as np
from numpy.typing import NDArray
from typing import List, NamedTuple, Optional

from .embed_text import Embed
from .tokenize_text import CountTokens
from .prompt_type import Prompts

class Span(NamedTuple):
  start: int
  end: int

class IndexedNounPhrase(NamedTuple):
  noun_phrase: str
  span: Span

class AllNounPhrases(NamedTuple):
  all_nps: List[str]
  spans: List[Span]
  lowest_nps: List[IndexedNounPhrase]

def get_sub_nps(tree: Tree, left: int, right: int) -> List[IndexedNounPhrase]:
  leaves: List[str] = tree.leaves()
  n_leaves: int = len(leaves)
  if isinstance(tree, str) or n_leaves == 1:
    return []
  sub_nps: List[IndexedNounPhrase] = []
  n_subtree_leaves: List[int] = [len(t.leaves()) for t in tree]
  offset: NDArray = np.cumsum([0] + n_subtree_leaves)[:len(n_subtree_leaves)]
  assert right - left == n_leaves
  if tree.label() == 'NP' and n_leaves > 1:
    noun_phrase=" ".join(leaves)
    span = Span(
      start=int(left),
      end=int(right),
    )
    indexed_noun_phrase = IndexedNounPhrase(
      noun_phrase=noun_phrase,
      span=span
    )
    sub_nps.append(indexed_noun_phrase)
  for i, subtree in enumerate(tree):
    sub_nps += get_sub_nps(subtree, left=left+offset[i], right=left+offset[i]+n_subtree_leaves[i])
  return sub_nps

def get_all_nps(tree: Tree, full_sent: Optional[str]=None) -> AllNounPhrases:
  start: int = 0
  end: int = len(tree.leaves())

  all_nps: List[IndexedNounPhrase] = get_sub_nps(tree, left=start, right=end)
  lowest_nps: List[IndexedNounPhrase] = []
  for np in all_nps:
    _, span = np
    start, end = span
    lowest = True
    for np_ in all_nps:
      _, span_ = np_
      start_, end_ = span_
      if start_ >= start and end_ <= end:
        lowest = False
        break
    if lowest:
      lowest_nps.append(np)

  all_nps, spans = map(list, zip(*all_nps))
  if full_sent and full_sent not in all_nps:
    all_nps: List[str] = [full_sent] + all_nps
    spans: List[Span] = [Span(start, end)] + spans

  return AllNounPhrases(
    all_nps=all_nps,
    spans=spans,
    lowest_nps=lowest_nps
  )

def expand_sequence(seq, length, dim=1):
  seq = seq.transpose(0, dim)
  max_length = seq.size(0)
  n_repeat = (max_length - 2) // length
  repeat_size = (n_repeat,) + (1, ) * (len(seq.size()) -1)

  eos = seq[length+1, ...].clone()
  segment = seq[1:length+1, ...].repeat(*repeat_size)
  seq[1:len(segment)+1] = segment
  seq[len(segment)+1] = eos

  return seq.transpose(0, dim)

def align_sequence(main_seq, seq, span, eos_loc, dim=1, zero_out=False, replace_pad=False):
  seq = seq.transpose(0, dim)
  main_seq = main_seq.transpose(0, dim)
  start, end = span[0]+1, span[1]+1
  seg_length = end - start

  # TODO: use torch.where(main_seq)
  #       err torch.index_put
  # main_seq.index_put(, seq[1:1+seg_length])
  main_seq[start:end] = seq[1:1+seg_length]
  if zero_out:
    main_seq[1:start] = 0
    main_seq[end:eos_loc] = 0

  if replace_pad:
    pad_length = len(main_seq) - eos_loc
    main_seq[eos_loc:] = seq[1+seg_length:1+seg_length+pad_length]

  return main_seq.transpose(0, dim)

def get_structured_embedder(embed: Embed, count_tokens: CountTokens) -> Embed:
  import stanza
  from stanza.models.common.doc import Document, Sentence
  from stanza.models.constituency.parse_tree import Tree as ConstituencyTree
  stanza_batch_delimeter = '\n\n'
  nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')

  def parse_constituency(prompt: str, constituency: ConstituencyTree):
    mytree: Tree = Tree.fromstring(str(constituency))
    nps, spans, noun_chunk = get_all_nps(mytree, prompt)

  def get_structured_embed(prompts: Prompts) -> Tensor:
    if isinstance(prompts, str):
      prompts: List[str] = [prompts]
    for prompt in prompts:
      assert not prompt.__contains__(stanza_batch_delimeter)
    prompt_batch: str = stanza_batch_delimeter.join(prompts)
    doc: Document = nlp.process(prompt_batch)

    for prompt, sentence in zip(prompts, doc.sentences):
      sentence: Sentence = sentence
      constituency: ConstituencyTree = sentence.constituency
      parse_constituency(prompt, constituency)
  return get_structured_embed