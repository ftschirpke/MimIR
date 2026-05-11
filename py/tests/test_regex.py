"""Regex plugin smoke + end-to-end via RegBuilder.jit()."""
from __future__ import annotations

import shutil
from pathlib import Path

import mim
import mim.regex as regex
import pytest


def test_regex_plugin_loads(regex_world):
    """The regex plugin loads and its top-level annex symbols resolve."""
    for sym_name in ("%regex.lit", "%regex.quant.star", "%regex.conj"):
        s = regex_world.sym(sym_name)
        assert s.str() == sym_name
        assert regex_world.annex(s) is not None


@pytest.mark.slow
@pytest.mark.needs_clang
def test_regbuilder_jit_match(tmp_path, monkeypatch):
    """End-to-end: build a tiny regex, JIT, run the matcher.

    Uses the canonical RegBuilder pattern from py/plugins/regex/tests/test.py.
    Runs in tmp_path so the generated .ll/.so don't pollute the repo.
    """
    if shutil.which("clang") is None:
        pytest.skip("clang not on PATH")

    monkeypatch.chdir(tmp_path)

    repo_root = Path(__file__).resolve().parents[2]
    plugin_dir = repo_root / "build" / "lib" / "mim"

    driver = mim.Driver()
    driver.add_search_path(plugin_dir)

    b = regex.RegBuilder(driver, "regex_test", mim.Level.Error)
    pattern = b.lit("a") + b.lit("b") + b.lit("c")
    library = pattern.jit()

    match_func = library["match_func"]
    assert match_func(b"abc") is True
    assert match_func(b"xyz") is False
