# https://stackoverflow.com/a/59987363
class PostInitMixin(object):
  def __post_init__(self):
    # just intercept the __post_init__ calls so they
    # aren't relayed to `object`
    pass