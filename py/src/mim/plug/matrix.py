from .._plugins.matrix import matrix as _matrix
from ._facade import install

matrix = install(globals(), "matrix", _matrix)
