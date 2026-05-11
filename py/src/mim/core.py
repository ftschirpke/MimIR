from ._enums.core import core as _core

core = _core


def __getattr__(name):
    return getattr(_core, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_core)))
