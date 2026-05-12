from .._plugins.affine import affine as _affine
from ._facade import install

affine = install(globals(), "affine", _affine)
