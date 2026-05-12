from __future__ import annotations

from pathlib import Path

import pytest

import mim


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "build" / "lib" / "mim"


@pytest.fixture
def plugin_dir() -> Path:
    if not PLUGIN_DIR.is_dir():
        pytest.skip(f"expected plugin directory at '{PLUGIN_DIR}'")
    return PLUGIN_DIR


@pytest.fixture
def driver() -> mim.Driver:
    return mim.Driver()


@pytest.fixture
def world(driver: mim.Driver):
    return driver.world()


@pytest.fixture
def regex_world(plugin_dir: Path):
    driver = mim.Driver()
    driver.add_search_path(plugin_dir)
    driver.load_plugins(["compile", "mem", "core", "math", "regex", "opt"])
    return driver.world()
