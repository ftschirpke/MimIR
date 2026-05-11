from enum import IntEnum

from ._mim import *
from ._mim import Def, Sym, World
from .callable import MimCallable
from .jit import JIT
from .plugin import MimPlugin


def call(self, *args) -> Def:
    callee = args[0]
    if isinstance(args[0], IntEnum):
        callee = self.annex_by_id(args[0].value)
    if isinstance(callee, str):
        callee = self.sym(callee)
        callee = self.annex(callee)
    if isinstance(callee, Sym):
        callee = self.annex(callee)
        if len(args) == 1:
            return callee

    if len(args) == 1:
        return callee

    if len(args) >= 3:
        if isinstance(args[1], list):
            return self.call(self.implicit_app(callee, args[1]), args[2::])
        else:
            return self.call(self.implicit_app(callee, [args[1]]), args[2::])
    else:
        if isinstance(args[1], list):
            return self.implicit_app(callee, args[1])
        else:
            if isinstance(args[1], tuple):
                tmp = list(args[1])
                return self.implicit_app(callee, tmp[0])
            return self.implicit_app(callee, [args[1]])
    raise TypeError("The given arguments dont match the expected types")


World.call = call

from . import plug

from .plug.regex import MimRegex, RegBuilder
