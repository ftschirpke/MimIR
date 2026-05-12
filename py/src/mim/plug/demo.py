from .._plugins.demo import demo as _demo
from ._facade import install

demo = install(globals(), "demo", _demo)
