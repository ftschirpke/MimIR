from .._plugins.opt import opt as _opt
from ._facade import install

opt = install(globals(), "opt", _opt)
