from .._plugins.compile import compile as _compile
from ._facade import install

compile = install(globals(), "compile", _compile)
