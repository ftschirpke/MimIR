from .._plugins.direct import direct as _direct
from ._facade import install

direct = install(globals(), "direct", _direct)
