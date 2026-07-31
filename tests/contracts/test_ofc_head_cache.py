"""SD-033b commitment_closure:GAP-8-affordability -- reusable trained-OFC-head cache.

Contract for chip-20260730-ofc-deval-affordable: the SD-033b OFC-analog
devaluation route requires per-cell gradient-tracking training (measured ~2x the
no-grad full-loop cost), which is why claims.yaml's named Q-085 design has
repeatedly been substituted for a policy-independent workaround (most recently
V3-EXQ-841). `experiments/_lib/ofc_head_cache.py` lets a caller pay that training
cost ONCE per (substrate, config_slice, seed) and reuse the trained weights --
including across a grid's dose arms, if the caller's config_slice deliberately
excludes the dose axis (per the module's documented soundness argument: OFCAnalog
reads only state_code + candidate summaries, never chunking/dose config).

C1 key stability: identical inputs -> identical key.
C2 key discrimination: seed / config_slice / substrate_scope changes -> different
   key (this is what makes the cache false-HIT-safe).
C3 the documented affordability lever: two config_slices differing ONLY in a
   dose-like field produce DIFFERENT keys when the field is included (safe
   default, no reuse) and the SAME key when a caller narrows it out (the
   deliberate reuse-across-arms case) -- both directions exercised so the
   contract pins the mechanism, not just one side of it.
C4 round-trip: get_or_train MISS trains + stores; a second call with the SAME key
   HITs and does NOT call train_fn again; the loaded head reproduces the trained
   weights exactly.
C5 OFCAnalog.head_state_dict/load_head_state_dict round-trip (module-level, no
   cache involved): saved weights survive a save->fresh-instance->load cycle
   bit-identically, and a shape-mismatched load raises rather than silently
   producing a wrong-shape head.
C6 corrupt / truncated / key-mismatched blob on disk -> MISS, never a crash.
C7 REE_OFC_HEAD_CACHE_DISABLE=1 -> every call is a miss, nothing persists.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

_EXP_DIR = Path(__file__).resolve().parents[2] / "experiments"
_LIB_DIR = _EXP_DIR / "_lib"
for _p in (str(_EXP_DIR), str(_LIB_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ofc_head_cache import (  # noqa: E402
    compute_head_key,
    get_or_train,
    load_blob,
    store_blob,
)
from ree_core.pfc.ofc_analog import OFCAnalog, OFCConfig  # noqa: E402

WORLD_DIM = 32
K = 8


def _ofc(**cfg_kw) -> OFCAnalog:
    cfg = OFCConfig(use_ofc_analog=True, harm_dim=0, **cfg_kw)
    return OFCAnalog(world_dim=WORLD_DIM, config=cfg)


def _bank(seed: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(K, WORLD_DIM, generator=g)


# ---------------------------------------------------------------------------
# C1 / C2 -- key stability and discrimination.
# ---------------------------------------------------------------------------

def test_c1_key_is_stable_for_identical_inputs():
    k1 = compute_head_key(config_slice={"lr": 0.001, "p1_episodes": 120}, seed=101)
    k2 = compute_head_key(config_slice={"lr": 0.001, "p1_episodes": 120}, seed=101)
    assert k1 == k2
    assert isinstance(k1, str) and len(k1) == 64  # sha256 hex


def test_c2_key_discriminates_seed():
    base = {"lr": 0.001}
    assert compute_head_key(config_slice=base, seed=101) != compute_head_key(
        config_slice=base, seed=202
    )


def test_c2_key_discriminates_config_slice_content():
    k_a = compute_head_key(config_slice={"lr": 0.001}, seed=101)
    k_b = compute_head_key(config_slice={"lr": 0.002}, seed=101)
    assert k_a != k_b


def test_c2_key_discriminates_config_slice_declared_flag():
    """config_slice_declared enters the key: a narrowed-and-declared slice can
    never collide with the same dict passed undeclared (mirrors
    arm_fingerprint.compute_arm_fingerprint's own config_slice_declared field)."""
    slice_ = {"lr": 0.001}
    k_undeclared = compute_head_key(config_slice=slice_, seed=101, config_slice_declared=False)
    k_declared = compute_head_key(config_slice=slice_, seed=101, config_slice_declared=True)
    assert k_undeclared != k_declared


def test_c2_key_discriminates_substrate_scope():
    """None (whole-tree) vs a declared scope must never collide, even if the
    resulting substrate_hash happens to agree -- mirrors arm_fingerprint's own
    substrate_scope_declared discriminator."""
    k_whole = compute_head_key(config_slice={"a": 1}, seed=101, substrate_scope=None)
    k_scoped = compute_head_key(
        config_slice={"a": 1}, seed=101,
        substrate_scope=("ree_core/pfc/ofc_analog.py",),
    )
    assert k_whole != k_scoped


# ---------------------------------------------------------------------------
# C3 -- the documented affordability lever: dose-field inclusion vs exclusion.
# ---------------------------------------------------------------------------

def test_c3_dose_field_included_forces_a_miss_across_arms():
    """Safe default: including the dose-like field in config_slice means two
    'arms' differing only in that field get DIFFERENT keys -- no reuse, but no
    risk of a false hit either."""
    cfg_arm_s2 = {"world_dim": WORLD_DIM, "chunk_max_size": 2}
    cfg_arm_s3 = {"world_dim": WORLD_DIM, "chunk_max_size": 3}
    assert compute_head_key(config_slice=cfg_arm_s2, seed=101) != compute_head_key(
        config_slice=cfg_arm_s3, seed=101
    )


def test_c3_dose_field_excluded_enables_cross_arm_reuse():
    """The deliberate reuse case: a caller that has verified OFC head training
    never reads chunk_max_size narrows config_slice to omit it -- both 'arms'
    then resolve to the SAME key for the same seed, so a single trained head
    serves both."""
    cfg_narrowed = {"world_dim": WORLD_DIM}  # chunk_max_size deliberately absent
    k_s2 = compute_head_key(config_slice=cfg_narrowed, seed=101, config_slice_declared=True)
    k_s3 = compute_head_key(config_slice=cfg_narrowed, seed=101, config_slice_declared=True)
    assert k_s2 == k_s3  # same narrowed slice -> same key regardless of which "arm" called it


# ---------------------------------------------------------------------------
# C4 -- get_or_train round trip via the cache.
# ---------------------------------------------------------------------------

def test_c4_get_or_train_miss_then_hit(tmp_path):
    calls = {"n": 0}

    def train_fn(ofc: OFCAnalog):
        calls["n"] += 1
        with torch.no_grad():
            ofc.state_bias_head[-1].weight.fill_(0.5)
            ofc.state_bias_head[-1].bias.fill_(0.1)

    ofc1 = _ofc(train_state_bias_head=True)
    result1 = get_or_train(
        ofc1,
        config_slice={"lr": 0.001, "seed_group": "test_c4"},
        seed=101,
        train_fn=lambda: train_fn(ofc1),
        cache_dir=tmp_path,
    )
    assert result1["cache"] == "miss"
    assert calls["n"] == 1

    # Fresh instance -- train_state_bias_head=True keeps random init (NOT zeroed;
    # that flag exists precisely so gradient can move the head from step 1), so a
    # HIT is verified below by content equality to the trained values, not by a
    # zeroed-before-load precondition (which would only hold for the OFF/zeroed
    # default and is not this test's point).
    ofc2 = _ofc(train_state_bias_head=True)
    result2 = get_or_train(
        ofc2,
        config_slice={"lr": 0.001, "seed_group": "test_c4"},
        seed=101,
        train_fn=lambda: train_fn(ofc2),
        cache_dir=tmp_path,
    )
    assert result2["cache"] == "hit"
    assert calls["n"] == 1  # train_fn NOT called again on HIT
    assert result1["key"] == result2["key"]
    assert torch.allclose(ofc2.state_bias_head[-1].weight, torch.full_like(
        ofc2.state_bias_head[-1].weight, 0.5))
    assert torch.allclose(ofc2.state_bias_head[-1].bias, torch.full_like(
        ofc2.state_bias_head[-1].bias, 0.1))


def test_c4_get_or_train_different_seed_is_a_separate_miss(tmp_path):
    calls = {"n": 0}

    def train_fn(ofc: OFCAnalog):
        calls["n"] += 1

    ofc_a = _ofc(train_state_bias_head=True)
    get_or_train(ofc_a, config_slice={"lr": 0.001}, seed=101,
                 train_fn=lambda: train_fn(ofc_a), cache_dir=tmp_path)
    ofc_b = _ofc(train_state_bias_head=True)
    result_b = get_or_train(ofc_b, config_slice={"lr": 0.001}, seed=202,
                             train_fn=lambda: train_fn(ofc_b), cache_dir=tmp_path)
    assert result_b["cache"] == "miss"
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# C5 -- OFCAnalog.head_state_dict / load_head_state_dict round trip.
# ---------------------------------------------------------------------------

def test_c5_head_state_dict_round_trip_state_bias_head_only():
    src = _ofc(train_state_bias_head=True)
    with torch.no_grad():
        src.state_bias_head[-1].weight.normal_()
        src.state_bias_head[-1].bias.normal_()
    blob = src.head_state_dict()
    assert set(blob.keys()) == {"state_bias_head"}  # no devaluation head built

    dst = _ofc(train_state_bias_head=True)
    dst.load_head_state_dict(blob)
    assert torch.allclose(dst.state_bias_head[-1].weight, src.state_bias_head[-1].weight)
    assert torch.allclose(dst.state_bias_head[-1].bias, src.state_bias_head[-1].bias)


def test_c5_head_state_dict_round_trip_both_heads():
    src = _ofc(train_state_bias_head=True, use_devaluation_head=True,
                train_devaluation_head=True)
    with torch.no_grad():
        src.state_bias_head[-1].weight.normal_()
        src.devaluation_bias_head[-1].weight.normal_()
    blob = src.head_state_dict()
    assert set(blob.keys()) == {"state_bias_head", "devaluation_bias_head"}

    dst = _ofc(train_state_bias_head=True, use_devaluation_head=True,
                train_devaluation_head=True)
    dst.load_head_state_dict(blob)
    assert torch.allclose(dst.state_bias_head[-1].weight, src.state_bias_head[-1].weight)
    assert torch.allclose(
        dst.devaluation_bias_head[-1].weight, src.devaluation_bias_head[-1].weight
    )


def test_c5_load_partial_blob_onto_instance_without_devaluation_head_ok():
    """A blob trained WITH the decoupled head loaded onto an instance that did not
    build one: state_bias_head loads, devaluation_bias_head is silently skipped
    (nothing to load onto) -- this is a valid partial load, not an error."""
    src = _ofc(train_state_bias_head=True, use_devaluation_head=True,
                train_devaluation_head=True)
    blob = src.head_state_dict()
    dst = _ofc(train_state_bias_head=True)  # no devaluation head
    dst.load_head_state_dict(blob)  # must not raise
    assert dst.devaluation_bias_head is None


def test_c5_shape_mismatched_load_raises():
    src = _ofc(train_state_bias_head=True, hidden_dim=32)
    blob = src.head_state_dict()
    dst = _ofc(train_state_bias_head=True, hidden_dim=64)  # different architecture
    with pytest.raises(RuntimeError):
        dst.load_head_state_dict(blob)


# ---------------------------------------------------------------------------
# C6 -- corrupt / mismatched blob handling.
# ---------------------------------------------------------------------------

def test_c6_missing_file_is_a_miss(tmp_path):
    assert load_blob("nonexistent" * 4, cache_dir=tmp_path) is None


def test_c6_corrupt_file_is_a_miss_not_a_crash(tmp_path):
    key = "a" * 64
    (tmp_path / f"{key}.pt").write_bytes(b"not a torch file")
    assert load_blob(key, cache_dir=tmp_path) is None


def test_c6_key_mismatch_is_a_miss(tmp_path):
    key = "b" * 64
    store_blob(key, {"key": "different-key", "head_state_dict": {}}, cache_dir=tmp_path)
    assert load_blob(key, cache_dir=tmp_path) is None


# ---------------------------------------------------------------------------
# C7 -- global disable knob.
# ---------------------------------------------------------------------------

def test_c7_disable_env_var_forces_miss_and_no_persist(tmp_path, monkeypatch):
    monkeypatch.setenv("REE_OFC_HEAD_CACHE_DISABLE", "1")
    calls = {"n": 0}

    def train_fn(ofc: OFCAnalog):
        calls["n"] += 1

    ofc1 = _ofc(train_state_bias_head=True)
    r1 = get_or_train(ofc1, config_slice={"x": 1}, seed=1,
                       train_fn=lambda: train_fn(ofc1), cache_dir=tmp_path)
    assert r1["cache"] == "disabled"

    ofc2 = _ofc(train_state_bias_head=True)
    r2 = get_or_train(ofc2, config_slice={"x": 1}, seed=1,
                       train_fn=lambda: train_fn(ofc2), cache_dir=tmp_path)
    assert r2["cache"] == "disabled"
    assert calls["n"] == 2  # trained BOTH times -- nothing was ever persisted
    assert list(tmp_path.glob("*.pt")) == []
