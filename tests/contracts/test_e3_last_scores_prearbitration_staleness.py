"""Contracts for chip-20260819-e3-last-scores-prearbitration-staleness.

THE DEFECT. `E3TrajectorySelector.select()` sets `self.last_scores` ONCE,
before any within-eligible narrowing mechanism (the modulatory shortlist /
F-eligibility-demotion lever, or ARC-110 cross-loop segregation) can override
the raw `scores.argmin()` winner. Those mechanisms are correctly
authoritative over the COMMITTED action (`selected_idx` reflects them), but
`last_scores` never does. Three `ree_core/agent.py` consumers read
`self.e3.last_scores`:

  1. MECH-342 maintenance-release decisiveness margin (agent.py ~5990)
  2. SD-061 stuck-state-detector margin (agent.py ~6265)
  3. dACC per-candidate payoff proxy (agent.py ~6773)

Consumers 1/2 both take `sorted(last_scores)[1] - sorted(last_scores)[0]` as
"the margin between the winner and the runner-up" -- an assumption that
silently breaks once a narrowing mechanism means the actual winner is NOT
`last_scores.argmin()`. Consumer 3 needs a payoff value for every candidate
(not just the winner), and no narrowing mechanism produces a re-scored value
for candidates outside the eligible set, so there is no well-defined
"post-arbitration" substitute for it -- it is deliberately left unchanged.

THE FIX. `E3TrajectorySelector` now also records `last_selected_idx` (the
full-space index `select()` actually committed to), stored UNCONDITIONALLY
like `last_raw_scores` so the write is output-neutral for any consumer that
never reads it. `E3TrajectorySelector.decisiveness_margin(arbitration_aware)`
computes the winner-vs-runner-up margin; with `arbitration_aware=False`
(legacy call sites' old inline behaviour) it is a blind
`sorted[1] - sorted[0]`; with `arbitration_aware=True` it anchors instead to
`last_selected_idx`: `(best score among the OTHER candidates) - (the
selected candidate's score)`. REEConfig.use_arbitration_aware_decisiveness_
margin (default False) gates which mode agent.py's two margin readers use.

Coverage:
  (A) default-off config flag + bit-identical-off across both call sites'
      consumers when no shortlist/arbitration mechanism ever fires (the
      degenerate regime is the ONLY one reachable with the flag off).
  (B) `decisiveness_margin` unit contract: None cases, legacy-vs-aware
      identity when selected == argmin, divergence when selected != argmin,
      defensive fallback when last_selected_idx is None/out-of-range.
  (C) END-TO-END reproduction of the staleness bug through a real
      `select()` call (modulatory top_k shortlist forces selected_idx away
      from the pre-arbitration argmin -- the same code path ARC-110 loop
      segregation funnels through at `if shortlist_idx is not None:
      selected_idx = int(shortlist_idx)`), proving `last_selected_idx`
      always equals the real `SelectionResult.selected_index`.
"""

from __future__ import annotations

import torch

from ree_core.predictors.e2_fast import Trajectory
from ree_core.predictors.e3_selector import E3Config, E3TrajectorySelector
from ree_core.utils.config import REEConfig


def _candidate(action_class: int, action_dim: int = 8) -> Trajectory:
    world_dim = 6
    horizon = 3
    states = [torch.zeros(1, world_dim) for _ in range(horizon + 1)]
    world_states = [torch.zeros(1, world_dim) for _ in range(horizon + 1)]
    actions = torch.zeros(1, horizon, action_dim)
    actions[:, 0, action_class % action_dim] = 1.0
    return Trajectory(states=states, actions=actions, world_states=world_states)


def _selector(**e3_kwargs) -> E3TrajectorySelector:
    sel = E3TrajectorySelector(E3Config(world_dim=6, hidden_dim=8, **e3_kwargs))
    sel._running_variance = 0.0  # committed path (deterministic argmin baseline)
    return sel


def _patch_raw(selector, candidates, raw_costs):
    """Force score_trajectory to return a known per-candidate raw cost."""
    raw_map = {id(c): torch.tensor([float(v)]) for c, v in zip(candidates, raw_costs)}
    selector.score_trajectory = lambda cand, **kw: raw_map[id(cand)]


# --------------------------------------------------------------------------- #
# (A) default-off config flag                                                 #
# --------------------------------------------------------------------------- #


def test_flag_defaults_false():
    cfg = REEConfig()
    assert cfg.use_arbitration_aware_decisiveness_margin is False


def test_last_selected_idx_is_none_before_any_selection():
    sel = _selector()
    assert sel.last_selected_idx is None


# --------------------------------------------------------------------------- #
# (B) decisiveness_margin unit contract                                       #
# --------------------------------------------------------------------------- #


def test_margin_none_when_no_last_scores():
    sel = _selector()
    assert sel.decisiveness_margin(arbitration_aware=False) is None
    assert sel.decisiveness_margin(arbitration_aware=True) is None


def test_margin_none_when_fewer_than_two_candidates():
    sel = _selector()
    sel.last_scores = torch.tensor([1.0])
    sel.last_selected_idx = 0
    assert sel.decisiveness_margin(arbitration_aware=False) is None
    assert sel.decisiveness_margin(arbitration_aware=True) is None


def test_legacy_mode_ignores_last_selected_idx():
    """Negative control: arbitration_aware=False must reproduce the exact old
    blind-sort call sites regardless of what last_selected_idx holds -- this
    is what makes the flag-off path bit-identical to pre-fix behaviour."""
    sel = _selector()
    sel.last_scores = torch.tensor([5.0, 1.0, 9.0, -3.0])
    for bogus_sel in (None, 0, 2, 3, -1, 99):
        sel.last_selected_idx = bogus_sel
        margin = sel.decisiveness_margin(arbitration_aware=False)
        assert margin == 4.0  # sorted = [-3, 1, 5, 9]; sorted[1]-sorted[0] = 4


def test_aware_mode_falls_back_to_legacy_when_selected_idx_missing():
    """A controlled state-machine probe that sets last_scores directly without
    going through select() leaves last_selected_idx at None (or stale/out of
    range) -- arbitration_aware=True must degrade to the legacy blind sort
    rather than crash or silently produce a nonsense margin."""
    sel = _selector()
    sel.last_scores = torch.tensor([5.0, 1.0, 9.0, -3.0])
    legacy = 4.0
    for bogus_sel in (None, -1, 4, 99):
        sel.last_selected_idx = bogus_sel
        assert sel.decisiveness_margin(arbitration_aware=True) == legacy


def test_aware_mode_matches_legacy_when_selected_is_the_argmin():
    """Identity case: when the committed candidate already IS the raw argmin
    (the only case reachable at all with every shortlist/arbitration flag
    off), arbitration_aware=True must equal arbitration_aware=False exactly."""
    sel = _selector()
    sel.last_scores = torch.tensor([5.0, 1.0, 9.0, -3.0])
    sel.last_selected_idx = 3  # -3.0 is the argmin
    assert sel.decisiveness_margin(arbitration_aware=False) == 4.0
    assert sel.decisiveness_margin(arbitration_aware=True) == 4.0


def test_aware_mode_diverges_when_selected_is_not_the_argmin():
    """The bug this chip fixes: when arbitration overrode the raw argmin, the
    legacy blind sort describes two candidates NEITHER of which was selected,
    while arbitration_aware anchors to the real winner."""
    sel = _selector()
    sel.last_scores = torch.tensor([5.0, 1.0, 9.0, -3.0])
    sel.last_selected_idx = 1  # NOT the argmin (that's index 3, score -3.0)

    legacy = sel.decisiveness_margin(arbitration_aware=False)
    aware = sel.decisiveness_margin(arbitration_aware=True)

    assert legacy == 4.0  # unchanged: still sorted[1]-sorted[0]
    # others = [5.0, 9.0, -3.0] -> min == -3.0; selected score == 1.0
    assert aware == -3.0 - 1.0
    assert aware != legacy


# --------------------------------------------------------------------------- #
# (C) end-to-end: a real select() call reproduces the staleness bug           #
# --------------------------------------------------------------------------- #


def test_end_to_end_shortlist_override_reproduces_and_fixes_staleness():
    """8 candidates. Raw (F) costs are [0..6, 100] -- candidate 7 is a clear F
    outlier. A large negative bias makes candidate 7's BIASED score the global
    minimum by a wide margin, but the top_k=3 shortlist gates ELIGIBILITY on
    raw F alone, so candidate 7 (never F-competitive) is never eligible and
    can never be selected -- the MECH-448 safety guarantee. Within the
    F-eligible set {0,1,2} (bias 0), the legacy within-eligible argmin picks
    index 0.

    This reproduces the exact staleness shape: last_scores.argmin() (index 7)
    is NOT the selected candidate (index 0) -- a blind top-2 sort of
    last_scores describes candidates {7, 0}, neither of which is "the winner
    and its runner-up" in any meaningful sense once you account for index 0
    actually being the commit.
    """
    sel = _selector(
        use_modulatory_shortlist_then_modulate=True,
        modulatory_shortlist_mode="top_k",
        modulatory_shortlist_k=3,
    )
    candidates = [_candidate(i) for i in range(8)]
    raw_costs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0]
    _patch_raw(sel, candidates, raw_costs)
    bias = torch.zeros(8)
    bias[7] = -1000.0  # candidate 7: biased score = 100 - 1000 = -900

    result = sel.select(candidates, temperature=1.0, score_bias=bias)

    # Safety: the F-outlier is never eligible, regardless of how attractive
    # its biased score looks.
    assert result.selected_index == 0

    # THE INVARIANT THIS CHIP ADDS: last_selected_idx always tracks the real
    # committed action, independent of which narrowing mechanism produced it.
    assert sel.last_selected_idx == result.selected_index

    # THE STALENESS PRECONDITION: last_scores' own argmin is a DIFFERENT
    # candidate than the one actually selected -- this is what makes the
    # legacy blind-sort margin meaningless here.
    assert int(sel.last_scores.argmin().item()) == 7
    assert int(sel.last_scores.argmin().item()) != result.selected_index

    legacy_margin = sel.decisiveness_margin(arbitration_aware=False)
    aware_margin = sel.decisiveness_margin(arbitration_aware=True)

    # Legacy: sorted last_scores = [-900, 0, 1, 2, 3, 4, 5, 6] -> margin 900.
    # This claims extreme decisiveness while the actually-selected candidate
    # (score 0.0) has a real next-best-eligible-competitor margin of only 1.0.
    assert legacy_margin == 900.0

    # Aware: anchored to the real winner (index 0, score 0.0) against the
    # best of everyone else (index 7, score -900.0, which the safety gate
    # correctly refused to select) -> a large NEGATIVE margin. This is the
    # CORRECT qualitative signal: the true F-and-bias-best candidate was
    # deliberately passed over, which downstream deficit computations
    # (CommitMaintenanceRelease._deficit_decisiveness /
    # StuckStateDetector._margin_deficit) both clip to their maximum-deficit
    # bound -- i.e. "very low decisiveness" -- rather than mishandling the
    # sign.
    assert aware_margin == -900.0
    assert aware_margin != legacy_margin


def test_end_to_end_bit_identical_off_when_no_narrowing_mechanism_active():
    """With every shortlist/arbitration flag at its default (off), select()
    can only ever land on scores.argmin() (the legacy fallback paths), so
    last_selected_idx == last_scores.argmin() on every call and the two
    margin modes agree exactly -- across a spread of random score landscapes,
    not just a hand-picked one."""
    for seed in range(10):
        torch.manual_seed(seed)
        sel = _selector()  # no shortlist / f_demotion / loop_segregation flags
        candidates = [_candidate(i) for i in range(6)]
        raw_costs = torch.randn(6).tolist()
        _patch_raw(sel, candidates, raw_costs)
        bias = torch.randn(6)

        result = sel.select(candidates, temperature=1.0, score_bias=bias)

        assert sel.last_selected_idx == result.selected_index
        assert sel.last_selected_idx == int(sel.last_scores.argmin().item())
        assert sel.decisiveness_margin(
            arbitration_aware=False
        ) == sel.decisiveness_margin(arbitration_aware=True)
