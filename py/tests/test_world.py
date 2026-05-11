"""Tests for py/bindings/world.cpp bindings (no plugins required)."""
from __future__ import annotations

import mim


def test_lit_i8(world):
    assert isinstance(world.lit_i8(7), mim.Def)


def test_lit_nat_0(world):
    assert isinstance(world.lit_nat_0(), mim.Def)


def test_top_nat(world):
    assert isinstance(world.top_nat(), mim.Def)


def test_type_bool(world):
    assert isinstance(world.type_bool(), mim.Def)


def test_type_i8(world):
    assert isinstance(world.type_i8(), mim.Def)


def test_type_i32(world):
    assert isinstance(world.type_i32(), mim.Def)


def test_type_idx_from_def(world):
    assert isinstance(world.type_idx(world.top_nat()), mim.Def)


def test_type_idx_from_int(world):
    assert isinstance(world.type_idx(256), mim.Def)


def test_lit_with_explicit_type(world):
    idx_t = world.type_idx(world.top_nat())
    assert isinstance(world.lit(idx_t, 0), mim.Def)


def test_sym_roundtrip(world):
    s = world.sym("hello")
    assert isinstance(s, mim.Sym)
    assert s.str() == "hello"
    assert not s.empty()
    assert s.size() == len("hello")


def test_sym_empty(world):
    s = world.sym("")
    assert s.empty()
    assert s.size() == 0


def test_cn_accepts_def_list(world):
    assert isinstance(
        world.cn([world.type_bool(), world.type_i8()]),
        mim.Def,
    )


def test_arr(world):
    assert isinstance(world.arr(world.top_nat(), world.type_i8()), mim.Def)


def test_mut_con(world):
    assert isinstance(
        world.mut_con([world.type_bool(), world.type_i8()]),
        mim.Def,
    )
