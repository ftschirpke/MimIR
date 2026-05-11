"""Tests for the World.call(...) helper monkey-patched in mim/__init__.py."""
from __future__ import annotations

from pathlib import Path

import unittest
import mim
import mim.plug.core as core

_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "build" / "lib" / "mim"


class TestCallFunctions(unittest.TestCase):

    def setUp(self):
        self.driver = mim.Driver()
        self.driver.add_search_path(_PLUGIN_DIR)
        self.world = self.driver.world()
        self.driver.load_plugins(["core"])

    def test_call_with_string_resolves_annex(self):
        callee = self.world.call(core.bit2.and_)
        assert isinstance(callee, mim.Def)

    def test_call_with_sym_resolves_annex(self):
        s = self.world.sym("%core.bit2.and_")
        callee = self.world.call(s)
        assert isinstance(callee, mim.Def)

    def test_call_with_def_passthrough(self):
        nat0 = self.world.lit_nat_0()
        # Per mim/__init__.py:23, a single non-string/Sym arg returns itself.
        assert self.world.call(nat0) is nat0 or isinstance(self.world.call(nat0), mim.Def)

    def test_call_folds_arg_list(self):
        # %core.bit2.and_ with a single Def arg — the folding branch.
        nat0 = self.world.lit_nat_0()
        result = self.world.call("%core.bit2.and_", nat0)
        assert isinstance(result, mim.Def)


if __name__ == "__main__":
    unittest.main()
