from .._plugins.gpu import gpu as _gpu
from ._facade import install

gpu = install(globals(), "gpu", _gpu)
