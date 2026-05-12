from .._plugins.option import option as _option
from ._facade import install

option = install(globals(), "option", _option)
