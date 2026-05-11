"""Tests for py/bindings/def.cpp bindings."""
from __future__ import annotations

import mim
import pytest


def test_def_world_round_trip(world):
    lit = world.lit_i8(0)
    assert isinstance(lit.world(), mim.World)


def test_set_returns_def_for_chaining(world):
    m = world.mut_con([world.type_bool()])
    assert isinstance(m.set("named"), mim.Def)


def test_dump_does_not_raise(world):
    world.lit_i8(0).dump()


def test_num_projs_matches_arity(world):
    m = world.mut_con([world.type_bool(), world.type_i8(), world.type_i32()])
    assert m.var().num_projs() == 3


def test_projs_returns_list_of_defs(world):
    m = world.mut_con([world.type_bool(), world.type_i8(), world.type_i32()])
    parts = m.var().projs(3)
    assert isinstance(parts, list)
    assert len(parts) == 3
    for p in parts:
        assert isinstance(p, mim.Def)


def test_proj_single_arg(world):
    m = world.mut_con([world.type_bool(), world.type_i8()])
    assert isinstance(m.var().proj(0), mim.Def)


def test_proj_two_args(world):
    m = world.mut_con([world.type_bool(), world.type_i8()])
    assert isinstance(m.var().proj(2, 0), mim.Def)


def test_driver_via_def(driver):
    d = driver.world().lit_i8(0).driver()
    assert d is not None


@pytest.mark.skip(
    reason="calling .var() on a projection currently segfaults; "
           "see pytest/regex.py:222 — convert to xfail-strict once the "
           "binding raises MIM_Error instead of crashing."
)
def test_var_on_projection_raises(world):
    m = world.mut_con([world.type_bool(), world.type_i8()])
    p = m.var().proj(0)
    with pytest.raises(mim.MIM_Error):
        p.var()
