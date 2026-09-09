"""Contracts for the 2026-09-09 `run_zworld_p0` seam (config= / target_fn=), V3-EXQ-1017.

The seam exists so a driver can run the SD-070 P0a recipe with a non-default objective (a
generic-only compression; a GOV-MATCHAUX-1 matched arbitrary-auxiliary control through the
proximity channel) WITHOUT re-implementing the warmup loop and without touching `ree_core/`.
The load-bearing property is BACKWARD BIT-IDENTITY: every pre-seam caller (config=None,
target_fn=None) must resolve to exactly the config and target it always did -- otherwise
every banked arm in the x734/737/1002/1008 family silently changes meaning.

ASCII-only (repo rule).
"""
import inspect
from pathlib import Path

import pytest

import experiments._lib.zworld_p0_warmup as zp0
from ree_core.latent.zworld_p0 import ZWorldP0Config

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_c1_legacy_construction_is_byte_identical():
    for dry in (False, True):
        for w in (0.0, 0.5):
            got = zp0.resolve_p0a_config(42, dry, w, None)
            legacy = (ZWorldP0Config(seed=42, batch_size=8, epochs=2, resource_field_weight=w)
                      if dry else ZWorldP0Config(seed=42, resource_field_weight=w))
            assert got == legacy, (dry, w, got, legacy)


def test_c2_explicit_config_is_seed_stamped_and_dry_shrunk_only():
    cfg = ZWorldP0Config(presence_weight=0.0, distance_weight=0.0, proximity_weight=0.5,
                         reconstruction_weight=10.0, seed=999)
    got = zp0.resolve_p0a_config(7, False, 0.0, cfg)
    assert got.seed == 7
    assert got.presence_weight == 0.0 and got.distance_weight == 0.0
    assert got.proximity_weight == 0.5 and got.reconstruction_weight == 10.0
    assert got.batch_size == cfg.batch_size and got.epochs == cfg.epochs
    dry = zp0.resolve_p0a_config(7, True, 0.0, cfg)
    assert dry.batch_size == 8 and dry.epochs == 2 and dry.presence_weight == 0.0
    # the caller's object is never mutated
    assert cfg.seed == 999


def test_c3_two_sources_for_one_weight_are_refused():
    with pytest.raises(ValueError):
        zp0.resolve_p0a_config(1, False, 0.5, ZWorldP0Config())


def test_c4_default_target_is_the_sd018_proximity_target():
    assert zp0.resolve_target_fn(None) is zp0.resource_prox_target

    def custom(obs):
        return 0.25
    assert zp0.resolve_target_fn(custom) is custom


def test_c5_signature_defaults_keep_every_existing_caller_valid():
    sig = inspect.signature(zp0.run_zworld_p0)
    assert sig.parameters["config"].default is None
    assert sig.parameters["target_fn"].default is None
    assert sig.parameters["resource_field_weight"].default == 0.0


def test_c6_sources_are_ascii():
    for rel in ("experiments/_lib/zworld_p0_warmup.py",
                "experiments/_lib/matched_aux_targets.py",
                "tests/contracts/test_zworld_p0_warmup_seam.py"):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        bad = [(i + 1, ch) for i, line in enumerate(text.splitlines())
               for ch in line if ord(ch) > 127]
        assert not bad, (rel, bad[:5])
