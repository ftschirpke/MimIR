from .._plugins.autodiff import autodiff as _autodiff
from ._facade import install

autodiff = install(globals(), "autodiff", _autodiff)
