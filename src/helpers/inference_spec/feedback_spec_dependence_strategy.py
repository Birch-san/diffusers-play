from .sample_spec import SampleSpec
from .latent_spec import FeedbackSpec

def has_feedback_dependence(spec: SampleSpec) -> bool:
  return isinstance(spec.latent_spec, FeedbackSpec)