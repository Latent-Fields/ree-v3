"""SD-079 contract: common-mode-invariant (centered) z_goal cue for goal_match.

SD-039's `Anchor.goal_match` is a raw cosine between a stored z_goal_snapshot and
the current z_goal. z_goal is an EMA attractor pulled toward z_world, so it
inherits -- and amplifies -- the SD-008 common-mode offset: measured on the
V3-EXQ-669b Stage-0 nursery, z_goal pairwise cosine min 0.9878 with
||mean(z_goal)|| / mean||z_goal|| = 0.9987. Every ABSOLUTE gate downstream is then
pinned: MECH-292's goal_match_floor excludes nothing, and MECH-339's outshining
gate `clip((outshine_pivot - goal_match) / outshine_pivot)` sits at exactly 0.0
for every anchor -- the composite cue's context channel is unconditionally dead
whenever it is enabled. SD-079 subtracts a slow EMA common-mode baseline before
the match (the SD-066 / SD-077 / SD-078 pattern).

C1 default OFF + bit-identity: goal_cue_centering defaults False; no baseline is
   allocated; goal_match is the raw cosine and the pool's match spread stays
   pinned inside the common-mode cone.
C2 centering separates: the SAME snapshots with centering ON recover a wide
   goal_match range.
C3 the downstream ABSOLUTE gates unpin: with centering OFF the MECH-339 outshining
   gate is 0.0 for every anchor and the MECH-292 floor excludes none; with
   centering ON both regain range.
C4 lazy seed + MECH-094: the baseline is seeded from the first WAKING cue and is
   never advanced by a simulation_mode cue.
C5 snapshots are stored RAW: an anchor still matches its own snapshot as the
   baseline drifts.
C6 the WRITE path advances the baseline, not only the read path. Advancing on
   reads alone lets the baseline be seeded from the query itself, which centers
   every snapshot against the query and drives every match to 0.0 (measured).
C7 alpha default is 0.05, NOT SD-066's 0.02 -- z_goal drifts, and 0.02 lags it.
   Only the OVER-tracking bound is asserted here; the lag half is measured on the
   real agent and is deliberately not fixture-asserted (see the test's scope note).
"""
from __future__ import annotations

import torch

from ree_core.hippocampal.anchor_set import (
    Anchor,
    AnchorGoalPayload,
    AnchorSet,
)
from ree_core.utils.config import AnchorSetConfig

GOAL_DIM = 8
N_ANCHORS = 24

OUTSHINE_PIVOT = 0.5      # GhostGoalBankConfig.outshine_pivot default
GOAL_MATCH_FLOOR = 0.05   # GhostGoalBankConfig.goal_match_floor default


def _drifting_common_mode_cues(n: int = N_ANCHORS + 1, dim: int = GOAL_DIM,
                               offset: float = 12.0, drift: float = 0.15,
                               seed: int = 0) -> torch.Tensor:
    """z_goal-shaped cue stream: a dominant shared direction that also DRIFTS
    (z_goal is an integrator), with diverse residuals underneath."""
    g = torch.Generator().manual_seed(seed)
    common = torch.ones(dim) * offset
    steps = torch.arange(n, dtype=torch.float32).unsqueeze(-1) * drift
    return common.unsqueeze(0) + steps + torch.randn(n, dim, generator=g)


def _anchor_set(centering: bool, alpha: float = 0.05) -> AnchorSet:
    cfg = AnchorSetConfig(
        use_sd039_anchor_payload=True,
        goal_cue_centering=centering,
        goal_cue_baseline_alpha=alpha,
    )
    return AnchorSet(cfg)


def _populate(aset: AnchorSet, cues: torch.Tensor) -> None:
    for i, cue in enumerate(cues[:-1]):
        aset.write_anchor(
            "fast", f"seg{i}", ("world",),
            z_world=torch.zeros(4),
            goal_payload=AnchorGoalPayload(
                z_goal_snapshot=cue.clone(), wanting_strength=0.5
            ),
        )


def _scores(aset: AnchorSet, query: torch.Tensor) -> torch.Tensor:
    pairs = aset.query_by_goal_match(query, threshold=-1.0)
    return torch.tensor([s for _, s in pairs])


def _outshine_gate(scores: torch.Tensor) -> torch.Tensor:
    return ((OUTSHINE_PIVOT - scores) / OUTSHINE_PIVOT).clamp(0.0, 1.0)


def test_c0_fixture_has_the_measured_z_goal_geometry():
    cues = _drifting_common_mode_cues()
    n = torch.nn.functional.normalize(cues, dim=-1)
    iu = torch.triu_indices(len(cues), len(cues), offset=1)
    v = (n @ n.t())[iu[0], iu[1]]
    assert v.min() > 0.95, f"fixture not common-mode dominated: {v.min()}"


def test_c1_centering_defaults_off_and_match_stays_pinned():
    assert AnchorSetConfig().goal_cue_centering is False

    cues = _drifting_common_mode_cues()
    aset = _anchor_set(centering=False)
    _populate(aset, cues)
    scores = _scores(aset, cues[-1])

    assert aset._goal_baseline is None, "OFF path must allocate no baseline"
    assert aset.goal_cue_baseline is None
    assert len(scores) == N_ANCHORS
    assert scores.min() > 0.9, "raw matches should sit inside the common-mode cone"
    assert (scores.max() - scores.min()) < 0.2, "raw spread unexpectedly wide"


def test_c2_centering_recovers_goal_match_range():
    cues = _drifting_common_mode_cues()

    raw = _anchor_set(centering=False)
    _populate(raw, cues)
    raw_scores = _scores(raw, cues[-1])
    raw_spread = float(raw_scores.max() - raw_scores.min())

    cen = _anchor_set(centering=True)
    _populate(cen, cues)
    cen_scores = _scores(cen, cues[-1])
    cen_spread = float(cen_scores.max() - cen_scores.min())

    assert cen.goal_cue_baseline is not None
    assert cen_spread > 5.0 * raw_spread, (
        f"centering did not widen goal_match range (raw {raw_spread:.4f} "
        f"-> centered {cen_spread:.4f})"
    )


def test_c3_downstream_absolute_gates_unpin():
    """The floor and the outshining gate are the two ABSOLUTE consumers of
    goal_match. Both are dead on a raw cue and both regain range centered."""
    cues = _drifting_common_mode_cues()

    raw = _anchor_set(centering=False)
    _populate(raw, cues)
    raw_scores = _scores(raw, cues[-1])
    assert int((raw_scores < GOAL_MATCH_FLOOR).sum()) == 0, \
        "MECH-292 floor should exclude nothing on a raw cue"
    assert float(_outshine_gate(raw_scores).max()) == 0.0, \
        "MECH-339 outshining gate should be identically 0 on a raw cue"

    cen = _anchor_set(centering=True)
    _populate(cen, cues)
    cen_scores = _scores(cen, cues[-1])
    assert float(_outshine_gate(cen_scores).max()) > 0.0, \
        "centered outshining gate still dead -- the context channel cannot fire"


def test_c4_lazy_seed_and_simulation_mode_does_not_move_baseline():
    cues = _drifting_common_mode_cues()
    aset = _anchor_set(centering=True)

    assert aset._goal_baseline is None
    aset.observe_goal_cue(cues[0], simulation_mode=False)
    assert torch.allclose(aset._goal_baseline, cues[0].reshape(-1))

    before = aset._goal_baseline.clone()
    aset.observe_goal_cue(cues[1] * 50.0, simulation_mode=True)
    assert torch.allclose(aset._goal_baseline, before), \
        "MECH-094: a simulation cue must not advance the baseline"

    aset.observe_goal_cue(cues[1], simulation_mode=False)
    assert not torch.allclose(aset._goal_baseline, before)


def test_c5_anchor_still_matches_its_own_raw_snapshot():
    cues = _drifting_common_mode_cues()
    aset = _anchor_set(centering=True)
    _populate(aset, cues)

    target = cues[3]
    scores = aset.query_by_goal_match(target, threshold=-1.0)
    best_anchor, best_score = scores[0]
    stored = best_anchor.goal_payload.z_goal_snapshot
    assert torch.allclose(stored, target), \
        "an anchor stopped being its own best match as the baseline drifted"
    assert best_score > 0.5


def test_c6_write_path_advances_the_baseline():
    """Read-only advancing seeds the baseline FROM THE QUERY, which zeroes every
    residual. The write path must advance it too."""
    cues = _drifting_common_mode_cues()
    aset = _anchor_set(centering=True)
    _populate(aset, cues)

    assert aset._goal_baseline is not None, \
        "write_anchor with a goal_payload must advance the baseline"
    # And the baseline must NOT simply be the query, which is what a read-only
    # advance would leave it as.
    scores = _scores(aset, cues[-1])
    assert float(scores.max()) > 0.0, \
        "all matches zero -- baseline was seeded from the query itself"


def test_c7_alpha_default_is_not_in_the_over_tracking_regime():
    """The default is 0.05, deliberately NOT SD-066/077/078's 0.02: z_goal is an
    integrator, so its common mode drifts and a baseline slow enough for a
    stationary cue lags it.

    SCOPE NOTE -- read before "strengthening" this test. The LAG half of that
    claim is NOT assertable here and is deliberately not asserted. It was
    established on the real agent (seeds 101/202/303: goal_match spread 0.0942 /
    0.1508 / 0.3319 at alpha=0.02 vs 0.9995+ at 0.05; see the SD-079 doc and the
    AnchorSetConfig comment). It does not reproduce on a synthetic fixture,
    because the real z_goal drift SATURATES (it is an EMA attractor) whereas any
    cheap fixture drifts linearly -- and under a linear drift a baseline seeded
    from an early cue sits far from the query, which manufactures a WIDE spread at
    0.02 and would invert the ordering. Calibrating a fixture until it reproduced
    the ordering would be fitting the test to the answer.

    What this fixture CAN show, and what is asserted, is the other bound: the
    default must not sit in the over-tracking regime, where the baseline chases
    the cue and erases genuine matches (measured on the agent: 14/20 and 18/20
    anchors driven below the floor at alpha 0.2 and 0.5)."""
    assert AnchorSetConfig().goal_cue_baseline_alpha == 0.05

    cues = _drifting_common_mode_cues()
    n_excluded = {}
    for alpha in (0.05, 0.1, 0.2):
        aset = _anchor_set(centering=True, alpha=alpha)
        _populate(aset, cues)
        s = _scores(aset, cues[-1])
        n_excluded[alpha] = int((s < GOAL_MATCH_FLOOR).sum())

    assert n_excluded[0.05] < n_excluded[0.1] < n_excluded[0.2], (
        "over-tracking should worsen monotonically with alpha, so the default is "
        f"a real choice on that axis: {n_excluded}"
    )
    assert n_excluded[0.05] < N_ANCHORS // 2, (
        f"default alpha already over-tracks: {n_excluded[0.05]}/{N_ANCHORS} "
        "anchors driven below the floor"
    )
