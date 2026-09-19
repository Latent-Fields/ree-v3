"""Contract tests for the mode-governance-engagement item (1) bounding
operator on the SD-032a SalienceCoordinator affinity inputs.

User decision 2026-09-19T04:18:29Z: replace the hard symmetric box clamp
`value = max(-cap, min(cap, value))` in SalienceCoordinator.tick() with a
PER-SIGNAL, SIGN-PRESERVING SATURATING SQUASH `cap * x / (sigma + |x|)`,
selected by a mode switch that keeps the box clamp reachable (so the landed
V3-EXQ-934 cap-sweep baseline stays reproducible).

Interface-level guarantees, independent of tuning:
  C1  DEFAULT IS BIT-IDENTICAL. affinity_input_cap=None is a no-op whatever
      the selector says; with a cap set, the selector's default is the legacy
      box clamp and reproduces its arithmetic exactly.
  C2  BOUNDED. The squash lands strictly INSIDE (-cap, +cap) for every finite
      input, including inputs orders of magnitude above the cap.
  C3  ODD / SIGN-PRESERVING. f(-x) == -f(x) and sign(f(x)) == sign(x).
  C4  MONOTONE. Strictly increasing across a sweep spanning the cap.
  C5  CONTINUOUS DERIVATIVE AT THE OLD CLAMP BOUNDARY. The squash's slope
      matches from the left and the right of x = +/-cap (and at x = 0), where
      the box clamp's slope jumps 1 -> 0. This discontinuity is what the
      substrate_queue entry names as the cause of the <= 1-grid-step crossing.
  C6  SIGMA IS REQUIRED AND UNDEFAULTED. Selecting the squash without an
      explicit positive sigma raises, at the operator AND through a live tick.
  C7  THE AT-CAP DEGENERACY IS GONE. Two large-but-DIFFERENT inputs produce
      IDENTICAL clamped logits (the V3-EXQ-935a signature: ext_margin_mean
      linear in cap at R^2 0.9996-0.9999) but DIFFERENT squashed ones.
  C8  REEConfig REACHABILITY. All three sites (dataclass field, from_dims
      signature, post-cls re-apply) carry both new knobs, and they reach the
      live coordinator -- the MECH-307 from_dims-swallows-unknown-kwargs trap.

Deliberately NOT asserted here: any particular sigma. sigma has no default and
no value is endorsed by the substrate_queue entry or by the commissioned lit
pull (targeted_review_salience_gain_normalisation); the values below are test
fixtures, not a production recommendation. The production default and the
commitment-term grading are items (2)/(3) of the queue entry and remain open.
"""

import pytest
import torch

from ree_core.agent import REEAgent
from ree_core.cingulate.salience_coordinator import (
    AFFINITY_BOUND_CLAMP,
    AFFINITY_BOUND_SQUASH,
    SalienceCoordinator,
    SalienceCoordinatorConfig,
    bound_affinity_input,
)
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

CAP = 2.0
SIGMA = 1.0  # test fixture ONLY -- see the module docstring.


def _coord(**kw):
    return SalienceCoordinator(SalienceCoordinatorConfig(**kw))


def _tick(coord, pe):
    """One tick with a single nonzero affinity input (dacc_pe)."""
    return coord.tick(
        dacc_bundle={"pe": pe, "foraging_value": 0.0, "choice_difficulty": 0.0},
        drive_level=0.0,
        is_offline=False,
    )


# -- C1 default is bit-identical ------------------------------------------


def test_c1_no_cap_is_a_noop_whatever_the_selector_says():
    base = _tick(_coord(), 17.0)["operating_mode"]
    for mode, sigma in (
        (AFFINITY_BOUND_CLAMP, None),
        (AFFINITY_BOUND_SQUASH, SIGMA),
    ):
        got = _tick(
            _coord(affinity_bound_mode=mode, affinity_squash_sigma=sigma), 17.0
        )["operating_mode"]
        assert got == base, "cap=None must stay a no-op under mode " + mode


def test_c1_selector_default_is_the_legacy_box_clamp():
    cfg = SalienceCoordinatorConfig()
    assert cfg.affinity_bound_mode == AFFINITY_BOUND_CLAMP
    assert cfg.affinity_squash_sigma is None

    capped = _tick(_coord(affinity_input_cap=CAP), 17.0)["operating_mode"]
    # The legacy clamp maps 17.0 -> CAP exactly, so a tick whose raw input IS
    # the cap must give the identical vector.
    at_cap = _tick(_coord(affinity_input_cap=CAP), CAP)["operating_mode"]
    assert capped == at_cap
    assert bound_affinity_input(17.0, CAP) == CAP
    assert bound_affinity_input(-17.0, CAP) == -CAP
    assert bound_affinity_input(0.5, CAP) == 0.5


# -- C2/C3/C4/C5 operator properties --------------------------------------


def test_c2_squash_is_bounded_strictly_inside_the_cap():
    for x in (0.1, 1.0, CAP, 10.0, 1e3, 1e9):
        y = bound_affinity_input(x, CAP, AFFINITY_BOUND_SQUASH, SIGMA)
        assert 0.0 < y < CAP, (x, y)
        assert -CAP < -y < 0.0
    # The clamp, by contrast, SITS ON the boundary -- that is the degeneracy.
    assert bound_affinity_input(1e9, CAP) == CAP


def test_c3_squash_is_odd_and_sign_preserving():
    for x in (1e-6, 0.25, 1.0, CAP, 7.5, 1e4):
        pos = bound_affinity_input(x, CAP, AFFINITY_BOUND_SQUASH, SIGMA)
        neg = bound_affinity_input(-x, CAP, AFFINITY_BOUND_SQUASH, SIGMA)
        assert pos == pytest.approx(-neg, abs=1e-15)
        assert pos > 0.0 and neg < 0.0
    assert bound_affinity_input(0.0, CAP, AFFINITY_BOUND_SQUASH, SIGMA) == 0.0


def test_c4_squash_is_strictly_monotone():
    xs = [-20.0 + 0.25 * i for i in range(161)]
    ys = [bound_affinity_input(x, CAP, AFFINITY_BOUND_SQUASH, SIGMA) for x in xs]
    for lo, hi in zip(ys, ys[1:]):
        assert hi > lo


def test_c5_derivative_is_continuous_across_the_old_clamp_boundary():
    h = 1e-6

    def slope(x):
        a = bound_affinity_input(x - h, CAP, AFFINITY_BOUND_SQUASH, SIGMA)
        b = bound_affinity_input(x + h, CAP, AFFINITY_BOUND_SQUASH, SIGMA)
        return (b - a) / (2.0 * h)

    for boundary in (CAP, -CAP, 0.0):
        left = slope(boundary - 10 * h)
        right = slope(boundary + 10 * h)
        assert left == pytest.approx(right, rel=1e-4), boundary
        assert left > 0.0
    # Closed form: d/dx [cap*x/(sigma+|x|)] = cap*sigma/(sigma+|x|)^2.
    assert slope(CAP) == pytest.approx(
        CAP * SIGMA / (SIGMA + CAP) ** 2, rel=1e-5
    )

    # The box clamp is what does NOT have this property: slope 1 below the
    # cap, slope 0 above it.
    def clamp_slope(x):
        a = bound_affinity_input(x - h, CAP)
        b = bound_affinity_input(x + h, CAP)
        return (b - a) / (2.0 * h)

    assert clamp_slope(CAP - 10 * h) == pytest.approx(1.0, rel=1e-4)
    assert clamp_slope(CAP + 10 * h) == pytest.approx(0.0, abs=1e-6)


# -- C6 sigma is required and undefaulted ---------------------------------


def test_c6_squash_without_sigma_raises_at_the_operator():
    with pytest.raises(ValueError, match="requires an explicit"):
        bound_affinity_input(1.0, CAP, AFFINITY_BOUND_SQUASH, None)
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="strictly positive"):
            bound_affinity_input(1.0, CAP, AFFINITY_BOUND_SQUASH, bad)
    with pytest.raises(ValueError, match="unknown affinity_bound_mode"):
        bound_affinity_input(1.0, CAP, "divisive_pool", SIGMA)


def test_c6_squash_without_sigma_raises_on_a_live_tick():
    coord = _coord(
        affinity_input_cap=CAP, affinity_bound_mode=AFFINITY_BOUND_SQUASH
    )
    with pytest.raises(ValueError, match="requires an explicit"):
        _tick(coord, 17.0)
    # ... and the SAME misconfiguration with no cap stays inert (C1).
    inert = _coord(affinity_bound_mode=AFFINITY_BOUND_SQUASH)
    _tick(inert, 17.0)


# -- C7 the at-cap degeneracy is gone --------------------------------------


def test_c7_clamp_collapses_distinct_large_inputs_and_squash_does_not():
    lo, hi = 16.0, 17.0  # the measured dacc_pe range through 464d/467d eval.

    clamp_lo = _tick(_coord(affinity_input_cap=CAP), lo)["operating_mode"]
    clamp_hi = _tick(_coord(affinity_input_cap=CAP), hi)["operating_mode"]
    assert clamp_lo == clamp_hi, "the V3-EXQ-935a at-cap degeneracy"

    def squashed(pe):
        return _tick(
            _coord(
                affinity_input_cap=CAP,
                affinity_bound_mode=AFFINITY_BOUND_SQUASH,
                affinity_squash_sigma=SIGMA,
            ),
            pe,
        )["operating_mode"]

    sq_lo, sq_hi = squashed(lo), squashed(hi)
    assert sq_lo != sq_hi, "squash must keep distinct large inputs distinct"
    # Graded in the right direction: more PE -> more internal_planning.
    assert sq_hi["internal_planning"] > sq_lo["internal_planning"]
    # And genuinely different from the clamp (liveness: the lever MOVES).
    assert sq_hi["internal_planning"] != clamp_hi["internal_planning"]


# -- C8 REEConfig reachability (the MECH-307 trap) -------------------------


def test_c8_from_dims_reaches_the_live_coordinator():
    env = CausalGridWorldV2(size=8, seed=11)

    def build(**kw):
        return REEConfig.from_dims(
            body_obs_dim=env.body_obs_dim,
            world_obs_dim=env.world_obs_dim,
            action_dim=4,
            use_salience_coordinator=True,
            **kw
        )

    default = build()
    assert default.salience_affinity_bound_mode == "clamp"
    assert default.salience_affinity_squash_sigma is None

    cfg = build(
        salience_affinity_input_cap=CAP,
        salience_affinity_bound_mode=AFFINITY_BOUND_SQUASH,
        salience_affinity_squash_sigma=SIGMA,
    )
    assert cfg.salience_affinity_bound_mode == AFFINITY_BOUND_SQUASH
    assert cfg.salience_affinity_squash_sigma == SIGMA

    torch.manual_seed(321)
    agent = REEAgent(cfg)
    assert agent.salience.config.affinity_bound_mode == AFFINITY_BOUND_SQUASH
    assert agent.salience.config.affinity_squash_sigma == SIGMA
    assert agent.salience.config.affinity_input_cap == CAP

    # An agent built with the defaults carries the legacy clamp.
    torch.manual_seed(321)
    base_agent = REEAgent(build(salience_affinity_input_cap=CAP))
    assert base_agent.salience.config.affinity_bound_mode == AFFINITY_BOUND_CLAMP
    assert base_agent.salience.config.affinity_squash_sigma is None
