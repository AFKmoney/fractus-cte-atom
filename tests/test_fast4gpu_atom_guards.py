"""Guards of scripts/fast4gpu_atom.py: no GPT-2 ids, no 50257 head, no rewind."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fast4gpu_atom", ROOT / "scripts" / "fast4gpu_atom.py")
fa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fa)

from fractus.continuous_engine import ContinuousThoughtEngine  # noqa: E402


def test_i16_atom_corpus_opens(tmp_path):
    p = tmp_path / "c.i16"
    np.arange(266, dtype=np.int16).tofile(p)
    arr, n = fa.open_corpus(str(p))
    assert n == 266 and int(arr.max()) == 265


def test_gpt2_ids_refused(tmp_path):
    p = tmp_path / "shard.npy"
    np.save(p, np.array([10, 300, 50256], dtype=np.int32))
    with pytest.raises(SystemExit, match="not an Atom stream"):
        fa.open_corpus(str(p))


def test_negative_ids_refused(tmp_path):
    p = tmp_path / "c.i16"
    np.array([1, -5, 3], dtype=np.int16).tofile(p)
    with pytest.raises(SystemExit):
        fa.open_corpus(str(p))


def test_50257_head_refused_not_sliced(tmp_path):
    eng = ContinuousThoughtEngine(vocab_size=50257, **fa.SCALES["smoke"])
    p = tmp_path / "x8.pt"
    torch.save({"model_state": eng.state_dict(), "config": {**fa.SCALES["smoke"], "vocab_size": 50257}}, p)
    with pytest.raises(SystemExit, match="rows"):
        fa.build_engine("smoke", str(p), log=lambda *a: None)


def test_fresh_applies_full_kuramoto_fix():
    eng, _, birth = fa.build_engine("smoke", None, log=lambda *a: None)
    assert birth["fresh"] and birth["kuramoto_fix"]["omega_scale"] == 4.0
    assert birth["kuramoto_fix"]["blocks"] == len(eng.blocks)


def test_atom_resume_is_strict_and_does_not_recompound_omega(tmp_path):
    eng, cfg, _ = fa.build_engine("smoke", None, log=lambda *a: None)
    p = tmp_path / "atom.pt"
    torch.save({"model_state": eng.state_dict(), "config": {**cfg, "vocab_size": 266}}, p)
    eng2, _, birth = fa.build_engine("smoke", str(p), log=lambda *a: None)
    assert not birth["fresh"]
    for a, b in zip(eng.blocks, eng2.blocks):
        assert torch.equal(a.kuramoto.omega, b.kuramoto.omega)


def test_resume_needs_start_token_and_never_zero(monkeypatch):
    monkeypatch.delenv("START_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        fa.resolve_start("some.pt")
    monkeypatch.setenv("START_TOKEN", "0")
    with pytest.raises(SystemExit, match="Never rewind"):
        fa.resolve_start("some.pt")
    monkeypatch.setenv("START_TOKEN", "4096")
    assert fa.resolve_start("some.pt") == 4096
    monkeypatch.delenv("START_TOKEN")
    assert fa.resolve_start(None) == 0
