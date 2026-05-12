from .._plugins.tuple import tuple as _tuple
from ._facade import install

tuple = install(globals(), "tuple", _tuple)
