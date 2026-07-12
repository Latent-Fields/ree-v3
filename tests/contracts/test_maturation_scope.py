"""Contracts for dependency-scoped substrate hashing of the maturation-curriculum
frozen-prefix cache (arm_reuse_fingerprint_plan.md sec 11).

FAST invariants only (no cell execution): the regression that the global sec-9
compute_substrate_hash path is byte-unchanged; the per-leg declared-scope sizes; the
IN-CLOSURE-refuses / OUT-OF-CLOSURE-hits behaviour at the hash + key layer; and the
STATIC conservatism guard (guard 2 -- the scope is a data-closed fixpoint of existing
files). The expensive guards -- guard 1 (call-trace: every executed file is in scope)
and the cold-MISS/warm-HIT bit-identity harness -- run full frozen-prefix cells and
live in the scratch harness verify_scope_conservatism(run_once=...); this suite keeps
CI fast while still catching the false-HIT hazard that matters most: a future edit that
lets a scope file read a constant from an unscoped module (guard 2 fails loudly).

Design plan: REE_assembly/evidence/planning/arm_reuse_fingerprint_plan.md sec 11.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
for _p in (str(REPO_ROOT), str(REPO_ROOT / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _lib.arm_fingerprint import compute_substrate_hash, _SUBSTRATE_GLOBS  # noqa: E402
from _lib.baselines import maturation_curriculum as MC  # noqa: E402


def _transient_append(rel, text=b"\n# maturation-scope contract transient edit\n"):
    """Append bytes to a repo file; return a restore() that byte-restores it."""
    p = REPO_ROOT / rel
    orig = p.read_bytes()
    p.write_bytes(orig + text)
    return lambda: p.write_bytes(orig)


def test_c1_default_substrate_hash_unchanged():
    """scope=None (default) must byte-reproduce today's whole-tree hash -- the global
    sec-9 arm_fingerprint path is untouched, so existing fingerprints are unaffected."""
    d_default = compute_substrate_hash()
    assert d_default["scoped"] is False
    assert compute_substrate_hash(scope=None)["substrate_hash"] == d_default["substrate_hash"]
    # passing the default globs explicitly must reproduce the default (faithful generalization)
    assert compute_substrate_hash(scope=_SUBSTRATE_GLOBS)["substrate_hash"] == d_default["substrate_hash"]


def test_c2_declared_scope_sizes_and_subset():
    """The declared scope is far narrower than the whole tree, and HARM subset of WORLD."""
    world = set(MC._WORLD_SUBSTRATE_SCOPE)
    harm = set(MC._HARM_SUBSTRATE_SCOPE)
    assert len(world) == 24
    assert len(harm) == 19
    assert harm <= world
    full = compute_substrate_hash()["n_files"]
    assert compute_substrate_hash(scope=MC._WORLD_SUBSTRATE_SCOPE)["n_files"] == 24 < full
    assert compute_substrate_hash(scope=MC._HARM_SUBSTRATE_SCOPE)["n_files"] == 19 < full


def test_c3_out_of_closure_edit_hits():
    """An edit to a ree_core module NOT in the declared scope (a sleep module) changes
    the GLOBAL-glob hash (the old behaviour would bust the cache) but must NOT change the
    scoped hash or the prefix key -- i.e. it now HITS."""
    oof = "ree_core/sleep/bayesian_aggregator.py"
    assert oof not in set(MC._WORLD_SUBSTRATE_SCOPE)
    glob_before = compute_substrate_hash()["substrate_hash"]
    scoped_before = compute_substrate_hash(scope=MC._WORLD_SUBSTRATE_SCOPE)["substrate_hash"]
    key_before = MC._prefix_key("world", {"seed": 1, "onset": 4})
    restore = _transient_append(oof)
    try:
        assert compute_substrate_hash()["substrate_hash"] != glob_before  # global bust
        assert compute_substrate_hash(scope=MC._WORLD_SUBSTRATE_SCOPE)["substrate_hash"] == scoped_before
        assert MC._prefix_key("world", {"seed": 1, "onset": 4}) == key_before  # HIT preserved
    finally:
        restore()
    assert compute_substrate_hash()["substrate_hash"] == glob_before  # restored


def test_c4_in_closure_edit_refuses():
    """An edit to any file INSIDE the declared scope must change the scoped hash AND the
    prefix key -- the cache refuses the stale prefix. Checks both an encoder-path file
    (latent/stack, both legs) and the data-closure leaf (regulators)."""
    key_w = MC._prefix_key("world", {"seed": 1, "onset": 4})
    key_h = MC._prefix_key("harm", {"seed": 1, "onset": 4})

    restore = _transient_append("ree_core/latent/stack.py")
    try:
        assert MC._prefix_key("world", {"seed": 1, "onset": 4}) != key_w
        assert MC._prefix_key("harm", {"seed": 1, "onset": 4}) != key_h
    finally:
        restore()
    # key restored
    assert MC._prefix_key("world", {"seed": 1, "onset": 4}) == key_w

    # data-closure leaf: SITE_* constants (a (b) value-import channel) must be in-closure.
    leaf = "ree_core/regulators/simulation_mode_rule_gate.py"
    assert leaf in set(MC._WORLD_SUBSTRATE_SCOPE)
    restore = _transient_append(leaf)
    try:
        assert MC._prefix_key("world", {"seed": 1, "onset": 4}) != key_w
    finally:
        restore()


def test_c5_static_conservatism_guard():
    """Guard 2: each declared scope is a data-closed fixpoint of existing files. If a
    future edit adds a constant value-import from an unscoped module (a false-HIT hazard),
    _verify_scope_static raises loudly -- this test is that tripwire in CI."""
    MC._verify_scope_static("world")
    MC._verify_scope_static("harm")
    # the static closure adds nothing outside the declared scope (idempotent fixpoint)
    for leg in ("world", "harm"):
        scope = set(MC._LEG_SUBSTRATE_SCOPE[leg])
        assert MC._static_data_closure(scope) == scope


def test_c6_provenance_records_scope():
    """The scope discriminator is recorded for audit (mirrors config_slice_declared)."""
    prov = MC._scope_provenance("world")
    assert prov["substrate_scope_declared"] is True
    assert isinstance(prov["substrate_scope"], list) and len(prov["substrate_scope"]) == 24
    # unknown leg -> safe default (hash everything, undeclared)
    assert MC._scope_provenance("nonexistent")["substrate_scope_declared"] is False
