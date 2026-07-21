"""SD-077 contract: common-mode-invariant (centered) super-ordinal cue key.

The MECH-189 SuperOrdinalGoalMemory keys its anchors on raw z_world cosine. Under
SD-008 z_world under-differentiation the untrained encoder maps every context into
a ~0.99-cosine common-mode cone, so every contact REINFORCES one slot and
anchor_count saturates at 1 -- the V3-EXQ-669b R3 readiness-gate failure that left
MECH-329 / MECH-189 unmeasurable. SD-077 subtracts a slow EMA common-mode baseline
before every cue cosine (the SD-066 pattern).

C1 default OFF + bit-identity: super_ordinal_cue_centering defaults False; no
   baseline is allocated; a common-mode-dominated context stream saturates to a
   single anchor exactly as before (the reproduced 669b signature).
C2 centering separates: the SAME context stream with centering ON allocates more
   than one anchor and recovers a non-degenerate contextual-complexity range.
C3 threshold tuning cannot substitute: with centering OFF, NO
   (merge_similarity, complexity_threshold) setting recovers a mean complexity
   above 669b's pre-registered COMPLEXITY_MARGIN of 0.05, because
   complexity = 1 - best_cosine is bounded by the common-mode floor.
C4 lazy seed + MECH-094: the baseline is seeded from the first WAKING context and
   is never advanced by a simulation_mode context.
C5 keys are stored raw: a drifting baseline moves query and stored keys together,
   so a repeated context keeps matching itself.
C6 persistence: the baseline round-trips through state_dict/load_state_dict, a
   pre-SD-077 checkpoint (no "baseline" key) loads as unseeded, and
   reset_anchors() clears it.
"""
from __future__ import annotations

import torch

from ree_core.goal import GoalConfig, SuperOrdinalGoalMemory

# 669b's pre-registered C3 margin -- the number that makes C3 unsatisfiable
# on a raw-z_world key.
COMPLEXITY_MARGIN = 0.05

CONTEXT_DIM = 8


def _store(centering: bool, **kw) -> SuperOrdinalGoalMemory:
    cfg = GoalConfig(
        goal_dim=4,
        use_super_ordinal_goal_anchors=True,
        super_ordinal_n_slots=32,
        super_ordinal_cue_centering=centering,
        **kw,
    )
    return SuperOrdinalGoalMemory(
        cfg, context_dim=CONTEXT_DIM, device=torch.device("cpu")
    )


def _common_mode_stream(n: int = 60, spread: float = 0.05):
    """A context stream with the measured 669b geometry: a large shared offset
    plus a small context-carrying residual. Raw pairwise cosine ~0.99; the
    residual alone is fully separated."""
    g = torch.Generator().manual_seed(7)
    offset = torch.ones(CONTEXT_DIM) * 1.0
    out = []
    for _ in range(n):
        resid = torch.randn(CONTEXT_DIM, generator=g)
        resid = resid / resid.norm()
        out.append((offset + spread * resid).unsqueeze(0))
    return out


def _drive(store: SuperOrdinalGoalMemory, stream, salience: float = 1.0):
    """Feed the stream through the write path; return the complexity of each
    fired write."""
    goal = torch.ones(1, 4)
    comps = []
    for ctx in stream:
        n0 = store._n_writes
        store.write(ctx, goal, salience=salience)
        if store._n_writes > n0:
            comps.append(store._last_complexity)
    return comps


def _mean_non_bootstrap(comps):
    """669b scores the MEAN complexity over fired writes. The very first write
    lands on an empty store and bootstraps at 1.0 by definition; report the mean
    the way the experiment does (all writes) so the comparison is like-for-like."""
    return sum(comps) / len(comps) if comps else 0.0


# -- C1 default OFF + bit-identity --------------------------------------------

def test_c1_centering_defaults_off_and_saturates():
    assert GoalConfig().super_ordinal_cue_centering is False
    assert GoalConfig().super_ordinal_cue_baseline_alpha == 0.02

    store = _store(centering=False)
    assert store.centering is False
    assert store._baseline is None

    comps = _drive(store, _common_mode_stream())
    # The reproduced 669b R3 failure: one anchor, everything reinforced.
    assert store.n_occupied() == 1, store.n_occupied()
    assert store._n_allocate == 1
    assert store._n_reinforce > 1
    # No baseline was ever allocated on the OFF path.
    assert store._baseline is None
    assert store.get_state()["super_ordinal_baseline_norm"] == -1.0
    # And C3 is unreachable: mean complexity sits far below the margin.
    assert _mean_non_bootstrap(comps) < COMPLEXITY_MARGIN


# -- C2 centering separates ----------------------------------------------------

def test_c2_centering_recovers_anchor_diversity():
    stream = _common_mode_stream()
    off = _store(centering=False)
    on = _store(centering=True)

    comps_off = _drive(off, stream)
    comps_on = _drive(on, stream)

    assert off.n_occupied() == 1
    # R3 (max anchor_count >= 2) is the gate 669b failed. Centering clears it.
    assert on.n_occupied() >= 2, on.n_occupied()
    assert on.n_occupied() > off.n_occupied()
    # And the complexity statistic regains a usable range.
    assert _mean_non_bootstrap(comps_on) > _mean_non_bootstrap(comps_off)
    assert on.get_state()["super_ordinal_baseline_norm"] > 0.0
    assert on.get_state()["super_ordinal_cue_centering"] is True


# -- C3 threshold tuning is not a substitute -----------------------------------

def test_c3_no_threshold_setting_rescues_the_raw_key():
    """The load-bearing claim of the SD-077 problem statement: because
    complexity = 1 - best_cosine and the common-mode floor bounds best_cosine
    from below, NO threshold pair lifts the mean complexity over 669b's margin
    while centering is off."""
    stream = _common_mode_stream()
    for merge in (0.5, 0.8, 0.95, 0.99):
        for cthr in (0.01, 0.2, 0.3):
            store = _store(
                centering=False,
                super_ordinal_merge_similarity=merge,
                super_ordinal_complexity_threshold=cthr,
            )
            comps = _drive(store, stream)
            assert _mean_non_bootstrap(comps) < COMPLEXITY_MARGIN, (
                f"merge={merge} cthr={cthr} unexpectedly cleared the margin"
            )

    # Whereas centering does clear it, at the SAME thresholds.
    store = _store(
        centering=True,
        super_ordinal_merge_similarity=0.8,
        super_ordinal_complexity_threshold=0.2,
    )
    assert _mean_non_bootstrap(_drive(store, stream)) > COMPLEXITY_MARGIN


# -- C4 lazy seed + MECH-094 ---------------------------------------------------

def test_c4_lazy_seed_and_simulation_mode_does_not_move_baseline():
    store = _store(centering=True)
    ctx = torch.ones(1, CONTEXT_DIM) * 3.0

    # A simulation_mode context must not seed the baseline (MECH-094).
    store.write(ctx, torch.ones(1, 4), salience=1.0, simulation_mode=True)
    assert store._baseline is None
    store.observe(ctx, simulation_mode=True)
    assert store._baseline is None

    # The first WAKING context seeds it exactly (no zero-init transient).
    store.observe(ctx)
    assert store._baseline is not None
    assert torch.allclose(store._baseline, ctx.reshape(-1))

    # A later simulation context still does not move it.
    before = store._baseline.clone()
    store.observe(torch.ones(1, CONTEXT_DIM) * -9.0, simulation_mode=True)
    assert torch.allclose(store._baseline, before)

    # A waking context moves it by exactly baseline_alpha.
    nxt = torch.ones(1, CONTEXT_DIM) * 5.0
    store.observe(nxt)
    a = store.config.super_ordinal_cue_baseline_alpha
    assert torch.allclose(store._baseline, (1 - a) * before + a * nxt.reshape(-1))


# -- C5 keys stored raw, self-match survives baseline drift ---------------------

def test_c5_repeated_context_still_matches_itself_as_baseline_drifts():
    store = _store(centering=True)
    goal = torch.ones(1, 4)
    anchor_ctx = torch.ones(1, CONTEXT_DIM)
    anchor_ctx[0, 0] = 4.0
    store.write(anchor_ctx, goal, salience=1.0)
    assert store.n_occupied() == 1

    # Drift the baseline with many unrelated waking contexts.
    for ctx in _common_mode_stream(n=40):
        store.observe(ctx)

    # The stored key is raw, so the same context still retrieves its own anchor.
    got = store.retrieve(anchor_ctx)
    assert got is not None
    _value, match, slot = got
    assert slot == 0
    assert match > 0.5, match


# -- C6 persistence ------------------------------------------------------------

def test_c6_baseline_roundtrips_and_resets():
    store = _store(centering=True)
    for ctx in _common_mode_stream(n=10):
        store.write(ctx, torch.ones(1, 4), salience=1.0)
    assert store._baseline is not None

    sd = store.state_dict()
    assert sd["baseline"] is not None

    other = _store(centering=True)
    other.load_state_dict(sd)
    assert torch.allclose(other._baseline, store._baseline)
    assert other.n_occupied() == store.n_occupied()

    # A pre-SD-077 checkpoint has no "baseline" key -> loads as unseeded.
    legacy = {k: v for k, v in sd.items() if k != "baseline"}
    third = _store(centering=True)
    third.load_state_dict(legacy)
    assert third._baseline is None

    # reset_anchors() clears the cue geometry along with the bank.
    store.reset_anchors()
    assert store._baseline is None
    assert store.n_occupied() == 0
