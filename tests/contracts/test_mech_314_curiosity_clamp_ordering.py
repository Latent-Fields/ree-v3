"""Contract tests for the MECH-314 curiosity clamp-ordering fix (2026-07-21).

THE DEFECT these pin against (traced 2026-07-21, session
affectionate-bose-38affb; write-up in REE_assembly e85ff72f74, node
infant_substrate:GAP-14 key governance_2026_07_21b):

    StructuredCuriosity.compute_score_bias accumulated three MECH-314
    sub-flavours into one `total` and then clamped the SUM to
    +/-curiosity_bias_scale. Two of the three (314b uncertainty, 314c
    learning progress) are UNIFORM broadcasts. Once
    w_unc*unc + w_lp*lp >= curiosity_bias_scale, EVERY element pinned to
    the rail, `total` became exactly uniform, and the per-candidate 314a
    novelty component -- the only part of curiosity that can change a
    selection -- was ANNIHILATED. A uniform vector is argmin-invariant at
    the E3 commit point (e3_selector.py:3113, and :3022 under
    use_f_eligibility_demotion), so the channel went behaviourally silent
    with no diagnostic saying so, and curiosity authority was NON-MONOTONE
    in the configured weights: raising them drove the system INTO
    saturation.

Contracts (S1-S5):

  S1: per-candidate ORDERING SURVIVES large uniform sub-signals. This is
      the pre-fix regression: with a 314b weight large enough to blow the
      whole clamp budget, the returned tensor was uniform (range 0) and
      carried no 314a ordering. It must now preserve both the range and
      the rank order of the 314a-only bias.

  S2: MONOTONE authority. Raising curiosity_novelty_weight must not
      REDUCE the argmin-relevant per-candidate range. Pre-fix, a joint
      weight sweep saturated and the range collapsed to 0 -- the
      V3-EXQ-667 signature (h_pos 4x byte-identical to 2x).

  S3: the clamp still BOUNDS curiosity's selection-relevant influence at
      curiosity_bias_scale (the original intent is preserved, not
      discarded), and the total magnitude at 2 * bias_scale.

  S4: the saturation diagnostic reports honestly -- 0 when unsaturated,
      and rising toward its (K-1)/K ceiling when the per-candidate
      deviation is genuinely railed -- so an experiment can gate
      curiosity non-vacuity at readiness rather than discovering a
      silent channel in a null result. The ceiling is (K-1)/K rather
      than 1.0 because the deviation is zero-mean: at least one
      candidate always sits inside the rail, which is precisely why the
      clamp can no longer flatten the vector completely.

  S5: unsaturated behaviour is unchanged (the decomposition reconstructs
      the plain sum when nothing clamps), and reset() clears the new
      diagnostics.

Assertions are on ORDERING / RANGE / diagnostics, never on an exact
committed action: torch.multinomial diverges across machine classes
(REE_Working/CLAUDE.md "Running the test suite").
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.policy import StructuredCuriosity, StructuredCuriosityConfig


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
class _MockE3:
    """Stand-in exposing _running_variance for the 314b uniform signal."""

    def __init__(self, running_variance: float = 1.0):
        self._running_variance = running_variance


class _MockResidue:
    """ResidueField stand-in with a single active RBF center at the origin.

    With one center at the origin the 314a novelty of a candidate is
    ||sig|| / mean(||sig||), so distinct candidate norms give a distinct,
    known per-candidate ordering.
    """

    def __init__(self, world_dim: int = 8):
        class _RBF:
            pass

        rbf = _RBF()
        rbf.centers = torch.zeros(2, world_dim)
        rbf.active_mask = torch.tensor([True, False])
        self.rbf_field = rbf


def _spread_summaries(K: int = 5, world_dim: int = 8) -> torch.Tensor:
    """Candidates at strictly increasing distance from the origin."""
    base = torch.zeros(K, world_dim)
    for i in range(K):
        base[i, 0] = 1.0 + 0.5 * i
    return base


def _module(**overrides) -> StructuredCuriosity:
    cfg = StructuredCuriosityConfig(use_structured_curiosity=True, **overrides)
    return StructuredCuriosity(cfg)


def _rank(t: torch.Tensor) -> list:
    return torch.argsort(t).tolist()


# ----------------------------------------------------------------------
# S1 -- per-candidate ordering survives large uniform sub-signals
# ----------------------------------------------------------------------
def test_s1_per_candidate_ordering_survives_saturating_uniform_terms():
    """THE regression. Pre-fix this returns a uniform tensor (range 0.0)."""
    world_dim = 8
    summaries = _spread_summaries(world_dim=world_dim)
    res = _MockResidue(world_dim=world_dim)
    common = dict(curiosity_novelty_weight=0.05, curiosity_bias_scale=0.1)

    # Reference: 314a alone. Carries the per-candidate ordering.
    mod_a = _module(
        use_curiosity_novelty=True,
        use_curiosity_uncertainty=False,
        use_curiosity_learning_progress=False,
        **common,
    )
    bias_a = mod_a.compute_score_bias(summaries, residue_field=res, e3=None)
    range_a = float(bias_a.max() - bias_a.min())
    assert range_a > 1e-4, "test setup: 314a must carry a per-candidate range"

    # 314a PLUS a uniform 314b large enough to blow the whole clamp budget
    # on its own (w_unc * unc = 5.0 * 2.0 = 10.0 >> bias_scale 0.1).
    mod_all = _module(
        use_curiosity_novelty=True,
        use_curiosity_uncertainty=True,
        use_curiosity_learning_progress=False,
        curiosity_uncertainty_weight=5.0,
        **common,
    )
    bias_all = mod_all.compute_score_bias(
        summaries, residue_field=res, e3=_MockE3(running_variance=2.0),
    )

    range_all = float(bias_all.max() - bias_all.min())
    assert range_all > 1e-6, (
        "per-candidate ranking annihilated by the uniform 314b broadcast: "
        f"bias={bias_all}"
    )
    assert _rank(bias_all) == _rank(bias_a), (
        "314a ordering not preserved under a saturating uniform term. "
        f"all={bias_all} (rank {_rank(bias_all)}), "
        f"a_only={bias_a} (rank {_rank(bias_a)})"
    )
    # The 314a range is small here, so it survives the deviation clamp
    # intact rather than merely surviving in sign.
    assert abs(range_all - range_a) < 1e-5


# ----------------------------------------------------------------------
# S2 -- monotone authority in the per-candidate weight
# ----------------------------------------------------------------------
def test_s2_authority_monotone_in_novelty_weight_under_uniform_load():
    """Raising the 314a weight must never SHRINK the argmin-relevant range.

    Pre-fix, a joint sweep drove the sum into the rail and the range
    collapsed to exactly 0 -- the V3-EXQ-667 "4x byte-identical to 2x"
    readiness failure.
    """
    world_dim = 8
    summaries = _spread_summaries(world_dim=world_dim)
    res = _MockResidue(world_dim=world_dim)
    e3 = _MockE3(running_variance=2.0)

    ranges = []
    for w in (0.01, 0.02, 0.04, 0.08):
        mod = _module(
            use_curiosity_novelty=True,
            use_curiosity_uncertainty=True,
            use_curiosity_learning_progress=False,
            curiosity_novelty_weight=w,
            curiosity_uncertainty_weight=5.0,   # saturating uniform load
            curiosity_bias_scale=0.1,
        )
        bias = mod.compute_score_bias(summaries, residue_field=res, e3=e3)
        ranges.append(float(bias.max() - bias.min()))

    assert all(r > 0.0 for r in ranges), f"range collapsed somewhere: {ranges}"
    for lo, hi in zip(ranges, ranges[1:]):
        assert hi >= lo - 1e-9, f"authority non-monotone in weight: {ranges}"
    assert ranges[-1] > ranges[0], f"authority did not grow at all: {ranges}"


# ----------------------------------------------------------------------
# S3 -- the clamp still bounds influence (original intent preserved)
# ----------------------------------------------------------------------
def test_s3_clamp_still_bounds_influence():
    """Curiosity must not dominate the score-bias chain at extreme weights."""
    world_dim = 8
    summaries = _spread_summaries(world_dim=world_dim)
    res = _MockResidue(world_dim=world_dim)
    scale = 0.1

    mod = _module(
        use_curiosity_novelty=True,
        use_curiosity_uncertainty=True,
        use_curiosity_learning_progress=False,
        curiosity_novelty_weight=50.0,      # enormous per-candidate signal
        curiosity_uncertainty_weight=50.0,  # enormous uniform signal
        curiosity_bias_scale=scale,
    )
    bias = mod.compute_score_bias(
        summaries, residue_field=res, e3=_MockE3(running_variance=2.0),
    )
    # Selection-relevant influence bounded at bias_scale (deviation is
    # clamped elementwise).
    dev = bias - bias.mean()
    assert float(dev.abs().max()) <= scale + 1e-6, (
        f"per-candidate deviation exceeded bias_scale: {dev}"
    )
    # Total magnitude bounded at 2 * bias_scale (deviation + offset).
    assert float(bias.abs().max()) <= 2.0 * scale + 1e-6, (
        f"total magnitude exceeded 2 * bias_scale: {bias}"
    )


# ----------------------------------------------------------------------
# S4 -- saturation diagnostic reports honestly
# ----------------------------------------------------------------------
def test_s4_saturation_diagnostic():
    world_dim = 8
    summaries = _spread_summaries(world_dim=world_dim)
    res = _MockResidue(world_dim=world_dim)

    # (a) Unsaturated: small per-candidate signal under a huge uniform
    #     load. The uniform term no longer eats the budget, so nothing
    #     rails and the diagnostic must read 0.
    mod_ok = _module(
        use_curiosity_novelty=True,
        use_curiosity_uncertainty=True,
        use_curiosity_learning_progress=False,
        curiosity_novelty_weight=0.05,
        curiosity_uncertainty_weight=5.0,
        curiosity_bias_scale=0.1,
    )
    mod_ok.compute_score_bias(
        summaries, residue_field=res, e3=_MockE3(running_variance=2.0),
    )
    assert mod_ok._last_clamp_saturated_frac == 0.0
    assert mod_ok._last_bias_range > 0.0
    st = mod_ok.get_state()
    assert st["last_clamp_saturated_frac"] == 0.0
    assert st["last_bias_range"] > 0.0

    # (b) Genuinely saturated: a per-candidate signal so large that the
    #     clamp compresses the ranking. The diagnostic must say so rather
    #     than hiding it.
    #
    #     The ceiling is (K-1)/K, NOT 1.0 -- the deviation is zero-mean by
    #     construction so at least one candidate always sits inside the
    #     rail. That property is the fix: unlike the pre-fix clamp on the
    #     offset-carrying sum, this clamp can never flatten the vector
    #     completely, so some ordering always survives.
    K = summaries.shape[0]
    mod_sat = _module(
        use_curiosity_novelty=True,
        use_curiosity_uncertainty=False,
        use_curiosity_learning_progress=False,
        curiosity_novelty_weight=50.0,
        curiosity_bias_scale=1e-4,
    )
    bias_sat = mod_sat.compute_score_bias(summaries, residue_field=res, e3=None)
    assert mod_sat._last_clamp_saturated_frac >= 0.5, (
        "compressed ranking must be reported as saturating, got "
        f"{mod_sat._last_clamp_saturated_frac}"
    )
    assert mod_sat._last_clamp_saturated_frac == float(K - 1) / float(K), (
        "zero-mean deviation should leave exactly one candidate off the "
        f"rail here, got {mod_sat._last_clamp_saturated_frac}"
    )
    # Even fully compressed, SOME ordering survives (not a flat vector).
    assert float(bias_sat.max() - bias_sat.min()) > 0.0


# ----------------------------------------------------------------------
# S5 -- unsaturated behaviour unchanged + reset clears diagnostics
# ----------------------------------------------------------------------
def test_s5_unsaturated_matches_plain_sum_and_reset_clears():
    """With a generous clamp, output == the plain un-clamped sum."""
    world_dim = 8
    summaries = _spread_summaries(world_dim=world_dim)
    res = _MockResidue(world_dim=world_dim)
    e3 = _MockE3(running_variance=0.5)
    common = dict(
        curiosity_novelty_weight=0.02,
        curiosity_uncertainty_weight=0.02,
        curiosity_bias_scale=10.0,  # nothing can clamp
    )

    mod_a = _module(
        use_curiosity_novelty=True,
        use_curiosity_uncertainty=False,
        use_curiosity_learning_progress=False,
        **common,
    )
    bias_a = mod_a.compute_score_bias(summaries, residue_field=res, e3=e3)

    mod_b = _module(
        use_curiosity_novelty=False,
        use_curiosity_uncertainty=True,
        use_curiosity_learning_progress=False,
        **common,
    )
    bias_b = mod_b.compute_score_bias(summaries, residue_field=res, e3=e3)

    mod_all = _module(
        use_curiosity_novelty=True,
        use_curiosity_uncertainty=True,
        use_curiosity_learning_progress=False,
        **common,
    )
    bias_all = mod_all.compute_score_bias(summaries, residue_field=res, e3=e3)

    assert torch.allclose(bias_all, bias_a + bias_b, atol=1e-6), (
        f"unsaturated path diverged from the plain sum: {bias_all} vs "
        f"{bias_a + bias_b}"
    )

    assert mod_all._last_bias_range > 0.0
    mod_all.reset()
    assert mod_all._last_clamp_saturated_frac == 0.0
    assert mod_all._last_bias_range == 0.0


# ----------------------------------------------------------------------
# MECH-094 gate is unaffected by the new branch
# ----------------------------------------------------------------------
def test_simulation_gate_still_zeroes_and_leaves_diagnostics_alone():
    world_dim = 8
    summaries = _spread_summaries(world_dim=world_dim)
    mod = _module(curiosity_bias_scale=0.1)
    bias = mod.compute_score_bias(
        summaries,
        residue_field=_MockResidue(world_dim=world_dim),
        e3=_MockE3(running_variance=5.0),
        simulation_mode=True,
    )
    assert (bias == 0).all()
    assert mod._last_clamp_saturated_frac == 0.0
    assert mod._last_bias_range == 0.0
