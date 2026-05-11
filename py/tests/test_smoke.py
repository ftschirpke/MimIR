"""Top-level binding registration smoke tests for mim.

Catches missing init_*(m) calls in py/bindings/py.cpp and missing
re-exports in mim/__init__.py.
"""
from __future__ import annotations

import importlib

import mim
import pytest

EXPECTED_FROM_BINDINGS = [
    "AST",
    "Debug",
    "Def",
    "Driver",
    "Error",
    "Flags",
    "Info",
    "Lam",
    "Level",
    "Lit",
    "Log",
    "MIM_Error",
    "Parser",
    "Pi",
    "Sym",
    "SymPool",
    "Verbose",
    "Warn",
    "World",
]

EXPECTED_FROM_WRAPPER = [
    "MimCallable",
    "MimPlugin",
    "MimRegex",
    "RegBuilder",
    "JIT",
    "call",
    "plug",
]

EXPECTED_PLUGIN_FACADES = [
    "affine",
    "autodiff",
    "clos",
    "compile",
    "core",
    "demo",
    "direct",
    "gpu",
    "math",
    "matrix",
    "mem",
    "opt",
    "option",
    "ord",
    "refly",
    "regex",
    "tensor",
    "tuple",
    "vec",
]


@pytest.mark.parametrize("name", EXPECTED_FROM_BINDINGS)
def test_binding_symbol_present(name):
    assert hasattr(mim, name), f"mim.{name} missing — check init_*() in py.cpp"


@pytest.mark.parametrize("name", EXPECTED_FROM_WRAPPER)
def test_wrapper_symbol_present(name):
    assert hasattr(mim, name), f"mim.{name} missing — check mim/__init__.py"


def test_log_levels_exported():
    for level_name in ("Error", "Warn", "Info", "Verbose", "Debug"):
        level = getattr(mim, level_name)
        assert isinstance(level, mim.Level)


def test_driver_constructs():
    d = mim.Driver()
    assert isinstance(d, mim.Driver)
    assert isinstance(d, mim.SymPool)


def test_world_reachable_via_driver(driver):
    w = driver.world()
    assert isinstance(w, mim.World)


def test_def_subclasses():
    assert issubclass(mim.Lit, mim.Def)
    assert issubclass(mim.Lam, mim.Def)
    assert issubclass(mim.Pi, mim.Def)


def test_mim_error_is_exception():
    assert issubclass(mim.MIM_Error, Exception)


def test_world_call_is_monkey_patched():
    """mim/__init__.py attaches a `call` method to World."""
    assert callable(getattr(mim.World, "call", None))


def test_plugin_facades_delegate_to_generated_plugins():
    assert mim.plug.core.bit2.and_ is not None
    assert mim.plug.regex.lit is not None


def test_generated_plugin_facades_cover_generated_plugins():
    for name in EXPECTED_PLUGIN_FACADES:
        facade = importlib.import_module(f"mim.plug.{name}")
        generated = importlib.import_module(f"mim._plugins.{name}")

        assert getattr(facade, name) is getattr(generated, name)
