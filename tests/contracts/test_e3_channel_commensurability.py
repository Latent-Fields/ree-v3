"""f_dominance_conversion_ceiling (MECH-439): E3 channel-commensurability operator.

WHAT THIS PINS. `failure_autopsy_V3-EXQ-571c_2026-09-02` (confirmed; ratified by
/governance 2026-09-02, REE_assembly 0ade914d46) measured a within-tick
CROSS-CANDIDATE variance partition over `score_trajectory`'s additive channels
and found a single channel holding 0.98-0.99999 of the variance in 15 of 16
cells -- `residue_weighted` in all 8 residue-FED cells (F's share 4e-06 to
1.1e-05), `f`/`harm_weighted` in 7 of 8 residue-STARVED cells (0.994-0.9998).
The load-bearing observation is that EVERY competing channel cleared the 1e-12
ABSOLUTE variance floor and failed only the 1e-3 RELATIVE share floor: the
monopoly is a SCALE phenomenon, not dead channels. So "which channel holds
authority" was a restatement of units rather than a contest, and
`n_live_channels=1` red-gated all four arms of 571c before criteria evaluation.

The operator divides each declared channel's per-candidate term by an EMA of
that channel's own CROSS-CANDIDATE standard deviation before the additive sum.

READINESS TARGET pinned by `test_operator_meets_readiness_target`: >= 2 channels
simultaneously above a 1e-3 relative cross-candidate share -- verbatim the
condition 571c could not meet.

METHOD / anti-vacuity. Every assertion here is paired with its own negative
control: the monopoly is first REPRODUCED with the operator off (so a test that
silently stopped exercising the defect fails), then shown lifted with it on. The
quantities are variance shares and scale ratios computed from `float64` python
arithmetic over recorded channel terms -- no `multinomial` draw is involved, so
this carries no cross-machine class-contract hazard (CLAUDE.md "Running the test
suite"; memory `reference-cross-machine-class-contract-divergence`).

BIT-IDENTITY. `test_off_path_is_the_plain_additive_sum` pins the OFF arithmetic
against the sub-terms recomputed from the public sub-methods. The stronger check
-- full-`float64` equality of scores, selected indices and every decomp value
across 25 select() ticks against pristine `origin/main` -- was run at build time
(2026-09-07) and passed on all 1250 recorded values; it cannot live in a
contract because a contract cannot reach a prior revision.
"""

import math

import torch

from ree_core.utils.config import E3Config
from ree_core.predictors.e3_selector import (
    E3TrajectorySelector,
    _COMMENSURABILITY_CHANNELS,
)
from ree_core.predictors.e2_fast import Trajectory

WORLD_DIM = 8
HIDDEN_DIM = 16
N_CANDIDATES = 6

# The two floors are 571c's own, deliberately: the operator's floor and the
# instrument's liveness floor must agree or "commensurable" means two things.
MIN_LIVE_CHANNEL_SHARE = 1e-3


def _make_trajectory(seed: int) -> Trajectory:
    g = torch.Generator().manual_seed(seed)
    horizon = 4
    ws = torch.randn(1, horizon + 1, WORLD_DIM, generator=g)
    ss = torch.randn(1, horizon + 1, WORLD_DIM, generator=g)
    acts = torch.randn(1, horizon, 4, generator=g)
    return Trajectory(
        states=[ss[:, i, :] for i in range(horizon + 1)],
        actions=acts,
        world_states=[ws[:, i, :] for i in range(horizon + 1)],
    )


def _candidates() -> list:
    return [_make_trajectory(100 + i) for i in range(N_CANDIDATES)]


def _selector(commensurability: bool, decomp: bool = True, **cfg_kw) -> E3TrajectorySelector:
    torch.manual_seed(0)
    cfg = E3Config(world_dim=WORLD_DIM, hidden_dim=HIDDEN_DIM)
    cfg.use_e3_channel_commensurability = commensurability
    for k, v in cfg_kw.items():
        setattr(cfg, k, v)
    sel = E3TrajectorySelector(cfg)
    sel.e3_score_decomp_enabled = decomp
    return sel


def _drive(sel: E3TrajectorySelector, ticks: int) -> None:
    cands = _candidates()
    for _ in range(ticks):
        sel.select(cands)


def _xcand_shares(sel: E3TrajectorySelector) -> dict:
    """Within-tick cross-candidate variance share, 571c's routed partition."""
    per_cand = sel.last_score_decomp.get("per_candidate", [])
    assert len(per_cand) >= 2, "partition needs >= 2 candidates"
    variances = {}
    for name in _COMMENSURABILITY_CHANNELS:
        vals = [float(d.get(name, 0.0)) for d in per_cand]
        mu = sum(vals) / len(vals)
        variances[name] = sum((v - mu) ** 2 for v in vals) / len(vals)
    total = sum(variances.values())
    if total <= 0.0:
        return {name: 0.0 for name in _COMMENSURABILITY_CHANNELS}
    return {name: v / total for name, v in variances.items()}


def _n_live(shares: dict) -> int:
    return sum(1 for s in shares.values() if s >= MIN_LIVE_CHANNEL_SHARE)


# --------------------------------------------------------------------------- #
# 1. The defect, and the readiness target that fixes it                        #
# --------------------------------------------------------------------------- #

def test_monopoly_is_reproduced_with_the_operator_off():
    """NEGATIVE CONTROL for the whole file: without the operator, one channel
    monopolises and every competitor falls below the RELATIVE share floor --
    571c's finding. If this ever stops failing, the tests below are vacuous."""
    sel = _selector(commensurability=False)
    _drive(sel, ticks=40)
    shares = _xcand_shares(sel)

    assert _n_live(shares) == 1, (
        "expected the 571c monopoly (exactly 1 live channel) with the operator "
        "OFF, got shares=%r" % (shares,)
    )
    assert max(shares.values()) > 0.98, (
        "monopoly should hold >0.98 of cross-candidate variance, got %.6f"
        % max(shares.values())
    )


def test_operator_meets_readiness_target():
    """THE READINESS TARGET, verbatim from the autopsy: >= 2 channels
    simultaneously above a 1e-3 relative cross-candidate share."""
    sel = _selector(commensurability=True)
    _drive(sel, ticks=40)
    shares = _xcand_shares(sel)

    assert _n_live(shares) >= 2, (
        "readiness target unmet: expected >= 2 channels above a %g relative "
        "cross-candidate share, got %d. shares=%r"
        % (MIN_LIVE_CHANNEL_SHARE, _n_live(shares), shares)
    )
    assert max(shares.values()) < 0.98, (
        "a channel still monopolises with the operator on: top share %.6f"
        % max(shares.values())
    )


def test_competing_channel_share_rises_by_orders_of_magnitude():
    """The defect is a SCALE phenomenon, so the fix must move the suppressed
    channel's share by orders of magnitude, not marginally."""
    off = _xcand_shares(_drive_and_return(_selector(commensurability=False), 40))
    on = _xcand_shares(_drive_and_return(_selector(commensurability=True), 40))

    suppressed = min(off, key=lambda k: off[k] if off[k] > 0.0 else 1.0)
    assert off[suppressed] < MIN_LIVE_CHANNEL_SHARE
    assert on[suppressed] > off[suppressed] * 100.0, (
        "suppressed channel %s moved only %.3e -> %.3e"
        % (suppressed, off[suppressed], on[suppressed])
    )


def _drive_and_return(sel, ticks):
    _drive(sel, ticks)
    return sel


# --------------------------------------------------------------------------- #
# 2. Default-off / bit-identity                                                #
# --------------------------------------------------------------------------- #

def test_default_is_off():
    assert E3Config().use_e3_channel_commensurability is False


def test_off_path_touches_no_operator_state():
    """With the operator off there must be no EMA update and no exposure --
    the whole mechanism is inert, not merely neutral."""
    sel = _selector(commensurability=False)
    _drive(sel, ticks=10)

    assert sel._chan_scale_n == 0
    assert sel._chan_scale_ema == {}
    assert sel.last_channel_scale_estimates == {}
    assert sel._last_commensurability_raw == {}


def test_off_path_is_the_plain_additive_sum():
    """OFF must be the pre-operator arithmetic: f_weight*F + lambda*M +
    rho*Phi, with no scaling factor anywhere."""
    sel = _selector(commensurability=False, decomp=False)
    traj = _make_trajectory(7)

    score = float(sel.score_trajectory(traj).mean().item())
    f = float(sel.compute_reality_cost(traj).mean().item())
    m = float(sel.compute_harm_cost_fallback(traj).mean().item())
    phi = float(sel.compute_residue_cost(traj).mean().item())
    expected = (
        sel.config.f_weight * f
        + sel.config.lambda_ethical * m
        + sel.config.rho_residue * phi
    )
    # rel_tol is float32-scaled ON PURPOSE. The score accumulates in float32
    # tensors while `expected` sums float64 python floats, so the two differ at
    # float32 epsilon (~1.19e-7) by construction -- measured 3.3e-8 here. A
    # tighter tolerance would fail on arithmetic that is in fact correct. This
    # still pins what it is for: a spurious scaling factor on the OFF path
    # would move the score by ORDERS of magnitude, not by an ulp.
    assert math.isclose(score, expected, rel_tol=1e-6, abs_tol=1e-9), (
        "OFF score %.17g != plain additive sum %.17g" % (score, expected)
    )


# --------------------------------------------------------------------------- #
# 3. The three design properties that are easy to get wrong                    #
# --------------------------------------------------------------------------- #

def test_operator_is_not_gated_on_the_decomp_diagnostic():
    """The operator captures its own terms. If it reused
    _last_traj_components it would be silently INERT whenever
    e3_score_decomp_enabled is off -- a live selection-path behaviour must not
    depend on a diagnostic switch."""
    sel = _selector(commensurability=True, decomp=False)
    _drive(sel, ticks=40)

    assert sel._chan_scale_n == 40, (
        "scale estimate did not update with the decomp diagnostic OFF "
        "(n_updates=%d) -- the operator is gated on the diagnostic"
        % sel._chan_scale_n
    )
    assert sel.last_channel_scale_estimates.get("engaged") is True


def test_scale_estimate_tracks_raw_spread_not_normalised_spread():
    """ANTI-FEEDBACK-LOOP. The EMA is fed from RAW terms. Were it fed from
    post-normalisation terms it would chase its own tail and every scale would
    converge to ~1.0, making the estimate meaningless (and the exposure a lie).
    Pinned by comparing the ON selector's estimates against the raw
    cross-candidate spread measured independently with the operator OFF."""
    off = _selector(commensurability=False)
    _drive(off, ticks=40)
    raw_shares_src = off.last_score_decomp["per_candidate"]

    on = _selector(commensurability=True)
    _drive(on, ticks=40)
    scales = on.last_channel_scale_estimates["scales"]

    for name in ("f_weighted", "harm_weighted"):
        vals = [float(d.get(name, 0.0)) for d in raw_shares_src]
        mu = sum(vals) / len(vals)
        raw_sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))
        if raw_sd <= 0.0:
            continue
        ratio = scales[name] / raw_sd
        assert 0.5 < ratio < 2.0, (
            "channel %s scale estimate %.6e is not tracking the RAW spread "
            "%.6e (ratio %.3f) -- estimated from normalised terms?"
            % (name, scales[name], raw_sd, ratio)
        )
        # And the collapse signature the feedback loop would produce:
        assert not math.isclose(scales[name], 1.0, rel_tol=1e-6), (
            "channel %s scale collapsed to 1.0 -- self-referential normaliser"
            % name
        )


def test_warmup_gates_engagement():
    """Dividing by a one-sample spread estimate is unstable, so nothing is
    normalised until the estimate has seen warmup_ticks ticks."""
    sel = _selector(commensurability=True, e3_commensurability_warmup_ticks=25)
    _drive(sel, ticks=5)

    assert sel._chan_scale_n == 5
    assert sel.last_channel_scale_estimates["engaged"] is False
    for name in _COMMENSURABILITY_CHANNELS:
        assert sel._commensurability_scale(name) == 1.0, (
            "channel %s was normalised during warmup" % name
        )

    shares_warm = _xcand_shares(sel)
    assert _n_live(shares_warm) == 1, (
        "selection changed during warmup -- the operator engaged early"
    )


def test_dead_channel_keeps_unit_scale_and_no_nonfinite_leaks():
    """A structurally-dead channel (novelty_weighted is hardcoded 0.0;
    benefit_weighted is warmup-gated behind a method with no callers) has a
    near-zero spread. It must keep unit scale rather than be divided by it."""
    sel = _selector(commensurability=True)
    _drive(sel, ticks=40)

    scales = sel.last_channel_scale_estimates["scales"]
    for name in _COMMENSURABILITY_CHANNELS:
        if scales[name] <= sel.config.e3_commensurability_floor:
            assert sel._commensurability_scale(name) == 1.0, (
                "dead channel %s (scale %.3e) was not floored to unit scale"
                % (name, scales[name])
            )

    for d in sel.last_score_decomp["per_candidate"]:
        for name, val in d.items():
            assert math.isfinite(float(val)), (
                "non-finite value leaked into channel %s: %r" % (name, val)
            )
    assert math.isfinite(float(sel.last_scores.sum().item()))


def test_normalisation_is_rank_preserving_within_channel():
    """Each channel is divided by a POSITIVE constant within a tick, so the
    channel's own ordering over candidates is preserved -- the
    'rank-preserving renormalisation' the implementation_hint requires. Only
    the relative authority BETWEEN channels may move."""
    off = _selector(commensurability=False)
    _drive(off, ticks=40)
    on = _selector(commensurability=True)
    _drive(on, ticks=40)

    for name in ("f_weighted", "harm_weighted"):
        off_vals = [float(d[name]) for d in off.last_score_decomp["per_candidate"]]
        on_vals = [float(d[name]) for d in on.last_score_decomp["per_candidate"]]
        off_order = sorted(range(len(off_vals)), key=lambda i: off_vals[i])
        on_order = sorted(range(len(on_vals)), key=lambda i: on_vals[i])
        assert off_order == on_order, (
            "channel %s candidate ordering changed: %r -> %r"
            % (name, off_order, on_order)
        )


def test_scale_estimates_are_exposed_for_verification():
    """The autopsy requires the per-channel scale estimates be EXPOSED, so a
    successor can VERIFY commensurability was achieved rather than assume it."""
    sel = _selector(commensurability=True)
    _drive(sel, ticks=40)

    est = sel.last_channel_scale_estimates
    for key in ("scales", "n_updates", "engaged", "warmup_ticks", "floor",
                "ema_alpha", "channels"):
        assert key in est, "exposure missing %r" % key
    assert set(est["scales"]) == set(_COMMENSURABILITY_CHANNELS)
    assert est["n_updates"] == 40
    assert est["channels"] == list(_COMMENSURABILITY_CHANNELS)


def test_single_candidate_tick_contributes_no_spread():
    """One candidate carries no cross-candidate spread; that tick must
    contribute nothing rather than a spurious zero that would drag every
    estimate toward the floor."""
    sel = _selector(commensurability=True)
    for _ in range(10):
        sel.select([_make_trajectory(5)])

    assert sel._chan_scale_n == 0
    assert sel._chan_scale_ema == {}
