from .._plugins.math import math as _math
from ._facade import install

math = install(globals(), "math", _math)
