from .._plugins.clos import clos as _clos
from ._facade import install

clos = install(globals(), "clos", _clos)
