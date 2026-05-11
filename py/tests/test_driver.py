"""Tests for py/bindings/driver.cpp bindings."""
from __future__ import annotations

from pathlib import Path

import mim
import pytest

_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "build" / "lib" / "mim"


def test_driver_default_construct():
    assert isinstance(mim.Driver(), mim.Driver)


def test_world_returns_world():
    d = mim.Driver()
    assert isinstance(d.world(), mim.World)
    assert isinstance(d.world(), mim.World)  # repeat call


def test_log_chains():
    d = mim.Driver()
    chained = d.log().set_stdout().set(mim.Level.Debug)
    assert chained is not None


def test_add_search_path_accepts_pathlib(tmp_path):
    mim.Driver().add_search_path(tmp_path)


def test_load_plugins_core_succeeds(driver):
    driver.add_search_path(_PLUGIN_DIR)
    driver.load_plugins(["core"])


def test_load_plugins_unknown_raises(driver, tmp_path):
    driver.add_search_path(tmp_path)
    with pytest.raises(Exception):
        driver.load_plugins(["this_plugin_does_not_exist_xyzzy"])


def test_sympool_inheritance():
    s = mim.Driver().sym("foo")
    assert s.str() == "foo"
