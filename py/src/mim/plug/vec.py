from .._plugins.vec import vec as _vec
from ._facade import install

vec = install(globals(), "vec", _vec)
