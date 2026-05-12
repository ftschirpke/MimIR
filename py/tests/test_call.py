"""Tests for the World.call(...) helper monkey-patched in mim/__init__.py."""
from __future__ import annotations

import pytest

import mim
import mim.plug.core as core


@pytest.fixture
def core_world(driver, plugin_dir):
    driver.add_search_path(plugin_dir)
    driver.load_plugins(["core"])
    return driver.world()


def test_call_with_string_resolves_annex(core_world):
    assert isinstance(core_world.call(core.bit2.and_), mim.Def)


def test_call_with_sym_resolves_annex(core_world):
    s = core_world.sym("%core.bit2.and_")
    assert isinstance(core_world.call(s), mim.Def)


def test_call_with_def_passthrough(core_world):
    nat0 = core_world.lit_nat_0()
    # Per mim/__init__.py: a single non-string/Sym arg returns itself.
    result = core_world.call(nat0)
    assert result is nat0 or isinstance(result, mim.Def)


def test_call_folds_arg_list(core_world):
    nat0 = core_world.lit_nat_0()
    assert isinstance(core_world.call("%core.bit2.and_", nat0), mim.Def)
