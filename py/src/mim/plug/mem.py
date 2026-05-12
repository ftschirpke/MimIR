from .._plugins.mem import mem as _mem
from ._facade import install

mem = install(globals(), "mem", _mem)
