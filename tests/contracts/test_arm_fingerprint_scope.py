"""Contracts for the GLOBAL arm_fingerprint substrate-scope surface (plan sec 11).

Generalises the maturation-curriculum prototype's dependency-scoped substrate hashing
to compute_arm_fingerprint / arm_cell: a multi-arm experiment may declare an
author-declared substrate scope, defaulting to scope=None (hash everything) exactly as
before -- strictly opt-in + false-miss-only.

Invariants asserted:
  (1) REGRESSION: substrate_scope=None is byte-identical to the pre-feature path -- same
      fingerprint whether the param is omitted or None; no scope keys enter the hash.
  (2) The declared scope is FOLDED INTO the hash (discriminator): a scoped fingerprint
      never collides with the whole-tree one, and two different scopes key differently --
      even when the hashed file CONTENT is identical (declared-whole-tree != undeclared).
  (3) An OUT-OF-scope substrate edit no longer busts the scoped fingerprint (HIT), while
      an IN-scope edit still refuses it -- the whole point of dependency scoping.
  (4) arm_cell threads substrate_scope identically to the low-level function.
  (5) The scope discriminator is recorded in the returned payload for audit.

Fast (no cell execution): reuses the maturation world scope as a known-good declared
closure and edits substrate files transiently at the hash layer.

Design plan: REE_assembly/evidence/planning/arm_reuse_fingerprint_plan.md sec 11.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
for _p in (str(REPO_ROOT), str(REPO_ROOT / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _lib import arm_fingerprint as afp  # noqa: E402
from _lib.arm_fingerprint import (  # noqa: E402
    _SUBSTRATE_GLOBS,
    arm_cell,
    compute_arm_fingerprint,
    compute_substrate_hash,
    reset_all_rng,
)
from _lib.baselines import maturation_curriculum as MC  # noqa: E402

WORLD = MC._WORLD_SUBSTRATE_SCOPE


def _fp(**kw):
    """compute_arm_fingerprint with fixed defaults; returns the payload dict."""
    base = dict(config_slice={"p0": 60}, seed=42, script_path=None, rng_fully_reset=True)
    base.update(kw)
    return compute_arm_fingerprint(**base)


def _append(rel, text=b"\n# arm-fingerprint-scope transient edit\n"):
    p = REPO_ROOT / rel
    orig = p.read_bytes()
    p.write_bytes(orig + text)
    return lambda: p.write_bytes(orig)


# ---- (1) regression: default path byte-unchanged -----------------------------

def test_default_scope_byte_identical():
    """substrate_scope omitted == substrate_scope=None: identical fingerprint + hash, and
    NO scope keys fold into the reuse-critical hash."""
    omitted = _fp()
    explicit_none = _fp(substrate_scope=None)
    assert omitted["arm_fingerprint"] == explicit_none["arm_fingerprint"]
    assert omitted["substrate_hash"] == explicit_none["substrate_hash"]
    # default path records the discriminator as undeclared (audit), does not scope
    assert omitted["substrate_scope_declared"] is False
    assert omitted["substrate_scope"] is None


# ---- (2) discriminator: scope folds into the key -----------------------------

def test_scoped_never_collides_with_default():
    """A scoped fingerprint differs from the whole-tree one (fewer files + discriminator)."""
    default = _fp()
    scoped = _fp(substrate_scope=WORLD)
    assert scoped["substrate_scope_declared"] is True
    assert list(scoped["substrate_scope"]) == list(WORLD)
    assert scoped["arm_fingerprint"] != default["arm_fingerprint"]
    # the scoped substrate_hash hashes strictly fewer files
    assert scoped["substrate_n_files"] < default["substrate_n_files"]


def test_declared_whole_tree_differs_from_undeclared():
    """Declaring the WHOLE globs as the scope hashes identical CONTENT to the default, yet
    keys differently -- the discriminator makes 'declared-all' a distinct reuse contract
    from 'undeclared' (mirrors config_slice_declared)."""
    default = _fp()
    declared_all = _fp(substrate_scope=tuple(_SUBSTRATE_GLOBS))
    # identical hashed content ...
    assert (compute_substrate_hash(scope=_SUBSTRATE_GLOBS)["substrate_hash"]
            == compute_substrate_hash()["substrate_hash"])
    # ... but different fingerprint (scope discriminator folded in)
    assert declared_all["arm_fingerprint"] != default["arm_fingerprint"]
    assert declared_all["substrate_scope_declared"] is True


def test_two_different_scopes_key_differently():
    world = _fp(substrate_scope=MC._WORLD_SUBSTRATE_SCOPE)
    harm = _fp(substrate_scope=MC._HARM_SUBSTRATE_SCOPE)
    assert world["arm_fingerprint"] != harm["arm_fingerprint"]


# ---- (3) the point of scoping: out-of-scope HIT, in-scope refuse -------------

def _next_process():
    """Cross the process boundary these two tests actually model.

    A mint and a later consumer are two PROCESSES. Since 2026-07-20 the substrate
    identity is resolved once per process and reused for every cell, so that a mid-run
    checkout move can no longer split one run's cells across two recorded identities
    (the executed-substrate fix -- failure_autopsy_V3-EXQ-778a). Editing a file inside a
    single process therefore does NOT move the fingerprint, by design. To exercise
    "consumer runs after the edit" we must start the consumer's snapshot fresh, which is
    exactly what a real second process does.
    """
    afp._reset_substrate_snapshot()


def test_out_of_scope_edit_hits():
    """An edit to a ree_core module NOT in the declared scope changes the DEFAULT
    fingerprint (old behaviour = bust) but leaves the SCOPED fingerprint unchanged (HIT)."""
    oof = "ree_core/sleep/bayesian_aggregator.py"
    assert oof not in set(WORLD)
    default_before = _fp()["arm_fingerprint"]
    scoped_before = _fp(substrate_scope=WORLD)["arm_fingerprint"]
    restore = _append(oof)
    try:
        _next_process()
        assert _fp()["arm_fingerprint"] != default_before          # default busts
        assert _fp(substrate_scope=WORLD)["arm_fingerprint"] == scoped_before  # scoped HITs
    finally:
        restore()
        _next_process()
    assert _fp(substrate_scope=WORLD)["arm_fingerprint"] == scoped_before


def test_in_scope_edit_refuses():
    """An edit INSIDE the declared scope changes the scoped fingerprint -- the cell refuses
    a stale reuse (the false-HIT guard)."""
    in_scope = "ree_core/latent/stack.py"
    assert in_scope in set(WORLD)
    scoped_before = _fp(substrate_scope=WORLD)["arm_fingerprint"]
    restore = _append(in_scope)
    try:
        _next_process()
        assert _fp(substrate_scope=WORLD)["arm_fingerprint"] != scoped_before
    finally:
        restore()
        _next_process()
    assert _fp(substrate_scope=WORLD)["arm_fingerprint"] == scoped_before


# ---- (4) arm_cell threads the scope identically ------------------------------

def test_arm_cell_threads_scope():
    cfg = {"p0": 60, "p1": 20}
    with arm_cell(43, config_slice=cfg, script_path=None, substrate_scope=WORLD) as cell:
        viacell = cell.stamp({})
    reset_all_rng(43)
    manual = compute_arm_fingerprint(config_slice=cfg, seed=43, script_path=None,
                                     rng_fully_reset=True, substrate_scope=WORLD)
    assert viacell["arm_fingerprint"] == manual["arm_fingerprint"]
    assert viacell["substrate_scope_declared"] is True
    assert list(viacell["substrate_scope"]) == list(WORLD)


# ---- (5) opt-in runtime guard 2 tripwire ------------------------------------

def test_runtime_guard_env_flag_catches_under_approximation(monkeypatch):
    """With REE_ARM_SCOPE_GUARD=1, emitting a fingerprint under a scope that is NOT
    data-closed (drops the regulators SITE_* leaf a scope file reads) raises loudly at
    emit time -- the cheap tripwire. Off by default so the normal path is unaffected."""
    bad = tuple(x for x in WORLD if "regulators" not in x)
    # default (flag off): NO guard, emits fine even though `bad` under-approximates
    ok = _fp(substrate_scope=bad)
    assert ok["substrate_scope_declared"] is True
    # flag on: the static guard fires
    monkeypatch.setenv("REE_ARM_SCOPE_GUARD", "1")
    raised = False
    try:
        _fp(substrate_scope=bad)
    except AssertionError:
        raised = True
    assert raised, "REE_ARM_SCOPE_GUARD=1 must reject a non-data-closed scope"
    # a data-closed scope still passes under the flag
    _fp(substrate_scope=WORLD)
