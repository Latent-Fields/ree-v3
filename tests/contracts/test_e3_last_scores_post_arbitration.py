"""Contracts for the E3-last-scores-pre-arbitration-staleness repair.

self.last_scores (e3_selector.select(), ~line 3239) is published from the
additive-authority `scores` field BEFORE the shortlist-then-modulate
(use_modulatory_shortlist_then_modulate / use_f_eligibility_demotion) /
loop-segregation (use_loop_segregation) arbitration runs -- but when either
lever is active, that arbitration IS authoritative over the committed action
(selected_idx = shortlist_idx = int(eligible_idx[local].item())). So three
downstream last_scores readers in agent.py (MECH-342 maintenance-release
decisiveness margin, SD-061 stuck-state-detector margin, dACC per-candidate
payoff proxy) could read a ranking the arbitration had already overridden,
and last_scores.argmin() need not match the committed candidate.

use_post_arbitration_last_scores (E3Config, no-op default) republishes the
eligible slice of last_scores after the arbitration decides its winner, via a
RANK-PRESERVING remap: the score VALUES already at the eligible positions are
kept (a permutation, not a rescale) but reassigned to eligible candidates in
the arbitration's own preference order.

These tests lock:
  (A) config default is a no-op; from_dims plumbing reaches config.e3.
  (B) unit-level remap invariants on _publish_post_arbitration_last_scores:
      rank-preserving permutation (multiset of values preserved), ineligible
      positions untouched, argmin lands on the arbitration's own winner, the
      no-op guards (<=1 eligible / last_scores is None), and no aliasing with
      the tensor last_scores previously pointed at (the autograd-safety
      property -- see the module docstring on the helper).
  (C) BYTE-IDENTICAL OFF: default (flag False) leaves last_scores exactly the
      pre-fix `scores` snapshot, with either lever active or not.
  (D) NO-OP even with the flag ON when neither lever ever sets shortlist_idx:
      last_scores is unaffected by the flag when the legacy single-arena path
      runs (this is what makes the flag safe to enable broadly).
  (E) MECHANISM, shortlist-then-modulate path: a scenario where the eligible
      band's tiny raw-F spread flips the naive combine away from the
      shortlist's pure-bias winner -- last_scores.argmin() mismatches
      selected_index OFF, matches it ON.
  (F) MECHANISM, loop-segregation path: same shape, using the per-loop
      normalised cross-loop arbitration (a case the additive combine cannot
      reproduce even in principle, since z-scoring strips F's raw-magnitude
      advantage -- the whole point of ARC-110).
  (G) MARGIN-STATISTIC INVARIANCE: because the remap is a permutation, the
      full-vector sorted last_scores (and hence the MECH-342 / SD-061 top-2
      gap those consumers read) is IDENTICAL whether the flag is on or off --
      the fix changes WHICH candidate a value belongs to, never the value
      distribution a margin/threshold consumer reads. This is the "why this
      cannot regress a calibrated margin threshold" contract.
  (H) dACC PAYOFF PROXY: `payoffs = -last_scores` is the one consumer that DOES
      change, because it is read PER-CANDIDATE (matched to `action_classes[i]`
      etc. in agent.py). OFF, payoffs.argmax() (best payoff) need not be the
      committed candidate; ON, it always is.
  (I) SAFETY: the remapped argmin is always inside eligible_idx even when the
      eligible set is a strict subset (some candidates excluded upstream) --
      the remap can never promote an excluded candidate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
    sel._running_variance = 0.0  # force committed path (deterministic argmin)
    return sel


def _patch_raw(selector, candidates, raw_costs):
    """Force score_trajectory to return a known per-candidate raw F cost."""
    raw_map = {id(c): torch.tensor([float(v)]) for c, v in zip(candidates, raw_costs)}
    selector.score_trajectory = lambda cand, **kw: raw_map[id(cand)]


# --------------------------------------------------------------------------- #
# (A) config default / from_dims plumbing                                     #
# --------------------------------------------------------------------------- #


def test_config_default_is_noop():
    cfg = E3Config(world_dim=6, hidden_dim=8)
    assert cfg.use_post_arbitration_last_scores is False


def test_from_dims_surfaces_flag_onto_e3():
    cfg = REEConfig.from_dims(
        body_obs_dim=10, world_obs_dim=10, action_dim=4, self_dim=16, world_dim=16,
        use_post_arbitration_last_scores=True,
    )
    assert cfg.e3.use_post_arbitration_last_scores is True
    cfg_off = REEConfig.from_dims(
        body_obs_dim=10, world_obs_dim=10, action_dim=4, self_dim=16, world_dim=16,
    )
    assert cfg_off.e3.use_post_arbitration_last_scores is False


# --------------------------------------------------------------------------- #
# (B) unit-level remap invariants                                             #
# --------------------------------------------------------------------------- #


class TestPublishHelperInvariants:
    def test_rank_preserving_permutation(self):
        sel = _selector()
        sel.last_scores = torch.tensor([0.5, 0.1, 0.9, 0.3])
        before_multiset = sorted(sel.last_scores.tolist())
        eligible_idx = torch.tensor([0, 1, 2, 3])
        arb_pref = torch.tensor([1.0, -5.0, 2.0, 0.0])  # candidate 1 is the arb winner
        sel._publish_post_arbitration_last_scores(eligible_idx, arb_pref)
        assert sorted(sel.last_scores.tolist()) == before_multiset, (
            "the remap must be a permutation -- the value multiset is preserved"
        )
        assert int(sel.last_scores.argmin().item()) == 1, (
            "last_scores.argmin() must land on the arbitration's own winner"
        )

    def test_ineligible_positions_untouched(self):
        sel = _selector()
        sel.last_scores = torch.tensor([0.5, 0.1, 0.9, 0.3, 7.0])
        eligible_idx = torch.tensor([0, 1, 3])  # candidate 2 and 4 excluded
        arb_pref = torch.tensor([2.0, 0.0, 1.0])  # local winner is eligible position 1 -> global 1
        sel._publish_post_arbitration_last_scores(eligible_idx, arb_pref)
        assert sel.last_scores[2].item() == pytest.approx(0.9), "excluded candidate 2 must be untouched"
        assert sel.last_scores[4].item() == pytest.approx(7.0), "excluded candidate 4 must be untouched"

    def test_noop_guards(self):
        sel = _selector()
        sel.last_scores = torch.tensor([0.5, 0.1])
        ref = sel.last_scores
        sel._publish_post_arbitration_last_scores(torch.tensor([0]), torch.tensor([1.0]))
        assert sel.last_scores is ref, "<=1 eligible candidate must be a strict no-op"

        sel2 = _selector()
        sel2.last_scores = None
        sel2._publish_post_arbitration_last_scores(
            torch.tensor([0, 1]), torch.tensor([1.0, 0.0])
        )
        assert sel2.last_scores is None, "last_scores is None must be a strict no-op"

    def test_no_aliasing_with_prior_last_scores_tensor(self):
        """The remap must reassign self.last_scores to a CLONE, never write through
        the tensor a caller may still be holding a reference to (that tensor shares
        storage with the live `scores` local select() keeps using for log_prob /
        SelectionResult.scores -- see the helper's docstring)."""
        sel = _selector()
        original = torch.tensor([0.5, 0.1, 0.9, 0.3])
        sel.last_scores = original
        eligible_idx = torch.tensor([0, 1, 2, 3])
        arb_pref = torch.tensor([1.0, -5.0, 2.0, 0.0])
        sel._publish_post_arbitration_last_scores(eligible_idx, arb_pref)
        assert sel.last_scores is not original
        assert torch.equal(original, torch.tensor([0.5, 0.1, 0.9, 0.3])), (
            "the tensor last_scores previously pointed at must be untouched"
        )


# --------------------------------------------------------------------------- #
# (C) / (D) byte-identical OFF, and no-op-with-flag-on-but-lever-inactive     #
# --------------------------------------------------------------------------- #


def test_byte_identical_off_legacy_path():
    """With neither lever active, last_scores is identical regardless of the flag
    (the publish call site is only reached inside the shortlist/loop-segregation
    branch, so this is a structural no-op, pinned here as a behavioural contract)."""
    raw = [0.1, 0.4, 0.2, 0.05]
    bias = torch.tensor([0.0, -0.2, 0.1, 0.0])
    for flag in (False, True):
        sel = _selector(use_post_arbitration_last_scores=flag)
        cands = [_candidate(i) for i in range(4)]
        _patch_raw(sel, cands, raw)
        r = sel.select(cands, temperature=1.0, score_bias=bias.clone())
        assert torch.equal(sel.last_scores, r.scores), (
            f"flag={flag}: legacy (no shortlist/loop-segregation) path must leave "
            f"last_scores identical to `scores`"
        )


def test_byte_identical_off_across_seeds_with_shortlist_active():
    """Flag OFF must be bit-identical to pre-fix behaviour (last_scores ==
    scores.detach()) across seeds, even with shortlist-then-modulate active --
    this locks that the new code path is inert unless explicitly opted in."""
    for seed in range(8):
        torch.manual_seed(seed)
        raw = torch.rand(5).tolist()
        bias = (torch.rand(5) - 0.5) * 2.0
        sel = _selector(
            use_modulatory_shortlist_then_modulate=True,
            use_post_arbitration_last_scores=False,
        )
        cands = [_candidate(i % 3) for i in range(5)]
        _patch_raw(sel, cands, raw)
        r = sel.select(cands, temperature=1.0, score_bias=bias.clone())
        assert torch.equal(sel.last_scores, r.scores), f"seed {seed}: OFF must match `scores`"


# --------------------------------------------------------------------------- #
# (E) mechanism: shortlist-then-modulate divergence                           #
# --------------------------------------------------------------------------- #


def _shortlist_scenario(flag: bool):
    """Eligible band where a small raw-F spread flips the naive additive combine
    away from the shortlist's pure-bias winner: F favours candidate 2 slightly,
    the near-tie margin still admits candidate 1 (raw right at the cutoff), and
    the modulatory bias decisively favours candidate 1 -- but only within the
    eligible-restricted `mod_eligible` argmin, not the raw+bias sum."""
    raw = [0.05, 0.6, 0.0, 0.02]
    bias = torch.tensor([0.0, -0.5, 0.0, 0.0])
    sel = _selector(
        use_modulatory_shortlist_then_modulate=True,
        modulatory_shortlist_margin=1.0,
        use_post_arbitration_last_scores=flag,
    )
    cands = [_candidate(i) for i in range(4)]
    _patch_raw(sel, cands, raw)
    r = sel.select(cands, temperature=1.0, score_bias=bias.clone())
    return sel, r


def test_shortlist_mechanism_mismatch_off_match_on():
    sel_off, r_off = _shortlist_scenario(flag=False)
    sel_on, r_on = _shortlist_scenario(flag=True)

    # The arbitration (shortlist-then-modulate) is authoritative either way.
    assert r_off.selected_index == r_on.selected_index == 1

    assert int(sel_off.last_scores.argmin().item()) != r_off.selected_index, (
        "OFF must reproduce the staleness: last_scores.argmin() disagrees with "
        "the actually-committed candidate"
    )
    assert int(sel_on.last_scores.argmin().item()) == r_on.selected_index, (
        "ON must repair it: last_scores.argmin() matches the committed candidate"
    )


# --------------------------------------------------------------------------- #
# (F) mechanism: loop-segregation divergence                                  #
# --------------------------------------------------------------------------- #


def _loop_segregation_scenario(flag: bool):
    """F (motor loop) strongly favours candidate 0 in raw magnitude; the limbic
    loop (ofc/liking/vigour) only weakly favours candidate 3 in raw magnitude --
    but per-loop zscore normalisation strips F's magnitude advantage (the ARC-110
    mechanism), and a boosted limbic spiral gain makes the cross-loop arbitration
    follow the limbic loop. The naive additive `scores` combine has no such
    normalisation and stays F-dominated."""
    raw = [-5.0, 0.5, 0.5, 0.5]
    limbic_val = torch.tensor([0.5, 0.5, 0.5, -0.01])
    sel = _selector(
        use_modulatory_shortlist_then_modulate=True,
        modulatory_shortlist_margin=1.0,
        use_loop_segregation=True,
        use_finer_channel_gating=True,
        loop_segregation_spiral_gain_limbic=3.0,
        use_post_arbitration_last_scores=flag,
    )
    cands = [_candidate(i) for i in range(4)]
    _patch_raw(sel, cands, raw)
    r = sel.select(
        cands,
        temperature=1.0,
        score_bias=limbic_val.clone() * 3,
        score_bias_channels={
            "ofc": limbic_val.clone(),
            "liking": limbic_val.clone(),
            "vigour": limbic_val.clone(),
        },
    )
    return sel, r


def test_loop_segregation_mechanism_mismatch_off_match_on():
    sel_off, r_off = _loop_segregation_scenario(flag=False)
    sel_on, r_on = _loop_segregation_scenario(flag=True)

    assert r_off.selected_index == r_on.selected_index == 3, (
        "the cross-loop arbitration (not the additive combine) must decide the "
        "committed candidate in both runs"
    )
    assert int(sel_off.last_scores.argmin().item()) != r_off.selected_index, (
        "OFF: the F-dominated additive combine disagrees with the actual "
        "(limbic-driven) commit"
    )
    assert int(sel_on.last_scores.argmin().item()) == r_on.selected_index, (
        "ON: last_scores tracks the arbitration's actual winner"
    )


# --------------------------------------------------------------------------- #
# (G) margin-statistic invariance (why this cannot regress MECH-342 / SD-061)  #
# --------------------------------------------------------------------------- #


def test_margin_statistic_is_invariant_to_the_flag():
    """A permutation cannot change a sorted-order statistic: the MECH-342 /
    SD-061 top-2 margin (sorted(last_scores)[1] - sorted(last_scores)[0]) is
    IDENTICAL on and off, in both mechanism scenarios above -- the fix only
    changes WHICH candidate a value is attached to."""
    for scenario in (_shortlist_scenario, _loop_segregation_scenario):
        sel_off, _ = scenario(flag=False)
        sel_on, _ = scenario(flag=True)
        sorted_off, _ = torch.sort(sel_off.last_scores)
        sorted_on, _ = torch.sort(sel_on.last_scores)
        assert torch.allclose(sorted_off, sorted_on), (
            f"{scenario.__name__}: sorted last_scores must be unaffected by the flag"
        )
        margin_off = float((sorted_off[1] - sorted_off[0]).item())
        margin_on = float((sorted_on[1] - sorted_on[0]).item())
        assert margin_off == margin_on, (
            f"{scenario.__name__}: MECH-342/SD-061 top-2 margin must be identical "
            f"regardless of the flag (off={margin_off}, on={margin_on})"
        )


# --------------------------------------------------------------------------- #
# (H) dACC per-candidate payoff proxy: the consumer that DOES change           #
# --------------------------------------------------------------------------- #


def test_dacc_payoff_proxy_tracks_committed_candidate_only_when_on():
    """Reproduces agent.py's dACC payoff proxy (payoffs = -last_scores when
    last_scores.numel() == K) for both mechanism scenarios: OFF, the best
    (max) payoff need not land on the committed candidate; ON, it always does."""
    for scenario in (_shortlist_scenario, _loop_segregation_scenario):
        sel_off, r_off = scenario(flag=False)
        sel_on, r_on = scenario(flag=True)

        K = 4  # both scenarios use 4 candidates
        assert sel_off.last_scores.numel() == K and sel_on.last_scores.numel() == K

        payoffs_off = -sel_off.last_scores.detach().float()
        payoffs_on = -sel_on.last_scores.detach().float()

        assert int(payoffs_off.argmax().item()) != r_off.selected_index, (
            f"{scenario.__name__}: OFF, the dACC payoff proxy's best candidate must "
            f"disagree with the committed one (the staleness this fix addresses)"
        )
        assert int(payoffs_on.argmax().item()) == r_on.selected_index, (
            f"{scenario.__name__}: ON, the dACC payoff proxy's best candidate must be "
            f"the committed one"
        )


# --------------------------------------------------------------------------- #
# (I) safety: remap never promotes an excluded candidate                      #
# --------------------------------------------------------------------------- #


def test_remap_stays_within_a_restricted_eligible_set():
    sel = _selector()
    sel.last_scores = torch.tensor([0.5, 0.1, 0.9, 0.3, -2.0])
    # Candidate 4 has the globally-lowest last_scores value but is EXCLUDED from
    # the eligible set -- the remap must never move that value onto an eligible
    # position in a way that makes an excluded candidate reachable, and must
    # never touch candidate 4's own slot.
    eligible_idx = torch.tensor([0, 1, 2])
    arb_pref = torch.tensor([1.0, 0.0, 2.0])  # local winner is eligible position 1 -> global 1
    sel._publish_post_arbitration_last_scores(eligible_idx, arb_pref)
    assert sel.last_scores[4].item() == pytest.approx(-2.0), "excluded candidate must be untouched"
    assert int(sel.last_scores[eligible_idx].argmin().item()) == 1, (
        "within the eligible slice, argmin must land on local position 1 (global "
        "candidate 1), matching arb_pref's own winner"
    )
