from .._plugins.tensor import tensor as _tensor
from ._facade import install

tensor = install(globals(), "tensor", _tensor)
