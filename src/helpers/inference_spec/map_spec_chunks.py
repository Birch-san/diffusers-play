from typing import TypeVar, Generic, Protocol, Iterable, Tuple

SampleSpec = TypeVar('SampleSpec')
MappedSpec = TypeVar('MappedSpec')

class MapSpec(Protocol, Generic[SampleSpec, MappedSpec]):
  @staticmethod
  def __call__(spec: SampleSpec) -> MappedSpec: ...

def map_spec_chunks(
  map_spec: MapSpec[SampleSpec, MappedSpec],
  spec_chunks: Iterable[Tuple[int, ...]],
  ) -> Iterable[Tuple[MappedSpec, ...]]:
  return map(lambda chunk: tuple(map(map_spec, chunk)), spec_chunks)