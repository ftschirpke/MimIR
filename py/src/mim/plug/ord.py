from .._plugins.ord import ord as _ord
from ._facade import install

ord = install(globals(), "ord", _ord)
