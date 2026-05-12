from .._plugins.refly import refly as _refly
from ._facade import install

refly = install(globals(), "refly", _refly)
