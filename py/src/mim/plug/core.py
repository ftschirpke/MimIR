from .._plugins.core import core as _core
from ._facade import install

core = install(globals(), "core", _core)
