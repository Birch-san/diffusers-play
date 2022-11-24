from transformers import logging

class log_level:
  orig_log_level: int
  log_level: int
  def __init__(self, log_level: int):
    self.log_level = log_level
    self.orig_log_level = logging.get_verbosity()
  def __enter__(self):
    logging.set_verbosity(self.log_level)
  def __exit__(self, exc_type, exc_value, exc_traceback):
    logging.set_verbosity(self.orig_log_level)