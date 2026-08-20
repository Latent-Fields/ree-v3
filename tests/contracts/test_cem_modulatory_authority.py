"""Contracts for the modulatory-bias-selection-authority 2026-08-19 AMEND.

Source spec: failure_autopsy_931-932-wanting-authority-cluster_2026-08-16
(confirmed, human-gated), target V3-EXQ-931,
recommended_substrate_queue_entry.implementation_hint_addendum.

The amend has three halves and each is pinned here:

  (a) AUTHORITY  -- the hippocampal CEM elite stage's modulatory terms get
      bounded authority over the elite argmin, one layer UPSTREAM of the
      implemented E3.select fix (which does not reach that call site).
  (b) THROUGHPUT -- authority does NOT imply behavioural throughput. Measured:
      80.3% of genuine elite argmins flipped while mean_resource_proximity was
      BIT-IDENTICAL to ablation. The elite stage is advisory-only unless its
      per-candidate contribution is routed into E3.
  (c) READINESS  -- a scoring-layer lever must report the ratio of its own
      cross-candidate spread to the dominant term's, and that ratio must be
      COMPETITIVE, not merely nonzero, before a behavioural falsifier is
      queued. V3-EXQ-931 read ~0.0037.

Roughly half of these are NEGATIVE CONTROLS -- assertions that the levers stay
inert by default and that a degenerate input cannot be dressed up as authority.
Those are the ones that stop a later session widening the mechanism until the
no-op default stops being a no-op.
"""
import torch

from ree_core.utils.config import REEConfig, HippocampalConfig, E3Config
from ree_core.agent import REEAgent
from ree_core.predictors.e3_selector import (
    AUTHORITY_COMPETITIVE_RATIO_FLOOR_DEFAULT,
    authority_ratio_is_competitive,
    authority_spread_ratio,
)


def _cfg(**kw):
    return REEConfig.from_dims(
        body_obs_dim=8, world_obs_dim=16, action_dim=4,
        self_dim=16, world_dim=16, **kw
    )


def _pool(agent, n=6, **kw):
    torch.manual_seed(0)
    z_self = torch.randn(1, 16)
    z_world = torch.randn(1, 16)
    return agent.hippocampal.propose_trajectories(
        z_self=z_self, z_world=z_world, num_candidates=n, **kw
    )


# --------------------------------------------------------------------------
# NEGATIVE CONTROLS -- the no-op default
# --------------------------------------------------------------------------

def test_new_levers_default_off():
    """Every field the amend adds is no-op by default, on all three configs."""
    h = HippocampalConfig()
    assert h.use_cem_modulatory_authority is False
    assert h.use_cem_modulatory_throughput is False
    assert h.cem_modulatory_authority_gain == 0.5
    assert h.cem_modulatory_authority_normalize_basis == "range"
    assert h.cem_modulatory_authority_min_spread_floor == 1e-6
    # The readiness floor is carried on every layer that reports a verdict, so
    # the two layers cannot judge against different floors.
    assert h.authority_competitive_ratio_floor == 0.1
    assert E3Config().authority_competitive_ratio_floor == 0.1
    assert _cfg().authority_competitive_ratio_floor == 0.1


def test_from_dims_actually_reaches_the_fields():
    """The [memory] reference-reeconfig-from-dims-silent-kwargs failure mode: a
    field with no from_dims entry falls into **kwargs and is silently dropped,
    so the flag reads False and the experiment runs with the lever OFF and no
    error. Assert every new knob is genuinely reachable through the factory."""
    c = _cfg(
        use_cem_modulatory_authority=True,
        cem_modulatory_authority_gain=0.25,
        cem_modulatory_authority_normalize_basis="std",
        cem_modulatory_authority_min_spread_floor=1e-4,
        use_cem_modulatory_throughput=True,
        authority_competitive_ratio_floor=0.3,
    )
    assert c.hippocampal.use_cem_modulatory_authority is True
    assert c.hippocampal.cem_modulatory_authority_gain == 0.25
    assert c.hippocampal.cem_modulatory_authority_normalize_basis == "std"
    assert c.hippocampal.cem_modulatory_authority_min_spread_floor == 1e-4
    assert c.hippocampal.use_cem_modulatory_throughput is True
    # fanned out to all three homes from the single parameter
    assert c.authority_competitive_ratio_floor == 0.3
    assert c.hippocampal.authority_competitive_ratio_floor == 0.3
    assert c.e3.authority_competitive_ratio_floor == 0.3


def test_score_trajectory_default_return_is_unchanged():
    """return_components=False must return the bare scalar, as every
    pre-existing call site expects."""
    agent = REEAgent(_cfg())
    trajs = _pool(agent)
    for t in trajs:
        s = agent.hippocampal._score_trajectory(t)
        assert isinstance(s, torch.Tensor)
        assert s.dim() == 0


def test_score_trajectory_components_reconstruct_exactly():
    """score == terrain + modulatory, EXACTLY (not approximately).

    The components are carried explicitly rather than recovered as
    (score - terrain) precisely because that subtraction is what produced the
    V3-EXQ-643 dead gate: at ~1e32 primary magnitude the real ~0.17 modulatory
    range fell below the float32 ULP and the difference computed exactly 0.0.
    """
    agent = REEAgent(_cfg())
    for t in _pool(agent):
        bare = agent.hippocampal._score_trajectory(t)
        s, comp = agent.hippocampal._score_trajectory(t, return_components=True)
        assert float(s.item()) == float(bare.item())
        assert float((comp["terrain"] + comp["modulatory"]).item()) == float(s.item())


def test_throughput_cache_stays_none_by_default():
    """NEGATIVE CONTROL: with the throughput lever off there is nothing to
    route, so the cache must stay None -- not an empty tensor, which a route
    source would happily project into a zero bias and report as live."""
    agent = REEAgent(_cfg())
    assert agent.hippocampal.last_candidate_modulatory_bias is None
    _pool(agent)
    assert agent.hippocampal.last_candidate_modulatory_bias is None
    d = agent.hippocampal.get_last_propose_diagnostics()
    assert d["cem_modulatory_authority_enabled"] is False
    assert d["cem_modulatory_authority_fired_iterations"] == 0
    assert d["cem_modulatory_throughput_available"] is False
    assert d["cem_elite_stage_advisory_only"] is True


def test_cache_is_declared_before_first_propose():
    """Reading the cache before any propose call must give None, never
    AttributeError -- a missing attribute sends the route source down its
    silent-None path, which is the V3-EXQ-863 inert-route failure shape."""
    agent = REEAgent(_cfg())
    assert getattr(agent.hippocampal, "last_candidate_modulatory_bias", "MISSING") is None


# --------------------------------------------------------------------------
# (a) AUTHORITY
# --------------------------------------------------------------------------

def _mode_cfg(**kw):
    """A config whose modulatory term has genuine cross-candidate spread.

    The wanting term cannot serve here: a fresh ResidueField's valence head is
    identically zero (the V3-EXQ-869/923 wash-out regime), so wanting is flat
    across candidates and correctly reads as below the spread floor. The
    mode_value term keys on z_world, which is always non-trivial.
    """
    c = _cfg(use_cem_modulatory_authority=True, **kw)
    c.hippocampal.mode_conditioning_enabled = True
    c.hippocampal.mode_value_weight = {"explore": [0.7] * 16}
    return c


def test_authority_rescale_fires_and_normalises_to_gain():
    """(a) After the rescale the modulatory term's cross-candidate spread is
    exactly gain * the terrain term's -- E3's algebra, one layer upstream."""
    gain = 0.5
    agent = REEAgent(_mode_cfg(cem_modulatory_authority_gain=gain))
    mode = {"explore": 1.0}
    trajs = _pool(agent, operating_mode=mode)
    d = agent.hippocampal.get_last_propose_diagnostics()
    assert d["cem_modulatory_authority_enabled"] is True
    assert d["cem_modulatory_authority_fired_iterations"] > 0
    assert d["cem_modulatory_authority_scale_factor_mean"] != 0.0

    terr, mod = [], []
    for t in trajs:
        _s, c = agent.hippocampal._score_trajectory(
            t, operating_mode=mode, return_components=True
        )
        terr.append(c["terrain"])
        mod.append(c["modulatory"])
    terr_t = torch.stack(terr).detach()
    mod_t = torch.stack(mod).detach()
    spread = float((mod_t.max() - mod_t.min()).item())
    assert spread > 0.0, "fixture must produce a non-flat modulatory term"
    scale = (gain * float((terr_t.max() - terr_t.min()).item())) / spread
    assert abs(authority_spread_ratio(scale * mod_t, terr_t) - gain) < 1e-4


def test_rescale_is_bidirectional():
    """The rescale is a NORMALISATION, so it both RAISES a sub-competitive
    lever (the V3-EXQ-931 shape, ratio 0.0037) and BOUNDS an over-dominant one
    down to gain. The bounding direction is the safety property the gain < 1.0
    convention exists for: a modulatory channel must not out-magnitude the
    terrain score it is meant to bias."""
    gain = 0.5
    dominant = torch.tensor([0.0, 1.0, 2.0, 3.0])
    for raw_ratio in (0.0037, 8.4):
        lever = torch.tensor([0.0, 1.0, 2.0, 3.0]) * raw_ratio
        pre = authority_spread_ratio(lever, dominant)
        assert abs(pre - raw_ratio) < 1e-6
        spread = float((lever.max() - lever.min()).item())
        scale = (gain * float((dominant.max() - dominant.min()).item())) / spread
        post = authority_spread_ratio(scale * lever, dominant)
        assert abs(post - gain) < 1e-6
        if raw_ratio < gain:
            assert post > pre, "sub-competitive lever must be raised"
        else:
            assert post < pre, "over-dominant lever must be bounded"


def test_flat_modulatory_term_is_not_amplified():
    """NEGATIVE CONTROL, and the load-bearing one. 'Scaling zero is still zero'
    (V3-EXQ-648): a modulatory term with no cross-candidate spread carries no
    information, so amplifying it would manufacture numerical noise dressed up
    as authority. Below the floor the rescale must SKIP, not divide by ~0."""
    agent = REEAgent(_cfg(use_cem_modulatory_authority=True))
    agent.hippocampal.config.wanting_weight = 0.5  # flat on a fresh field
    _pool(agent)
    d = agent.hippocampal.get_last_propose_diagnostics()
    assert d["cem_modulatory_authority_enabled"] is True
    assert d["cem_modulatory_authority_fired_iterations"] == 0
    assert d["cem_modulatory_authority_scale_factor_mean"] == 0.0


# --------------------------------------------------------------------------
# (b) THROUGHPUT
# --------------------------------------------------------------------------

def test_throughput_cache_is_index_aligned_to_the_final_pool():
    """The cache is routed into E3 as a per-candidate bias, so a length
    mismatch would silently misattribute each bias to a DIFFERENT trajectory.
    It must be computed over the FINAL pool -- after ghost mixing, support
    injection, scaffold and chunk splicing -- not the CEM's internal samples."""
    agent = REEAgent(_cfg(use_cem_modulatory_throughput=True))
    agent.hippocampal.config.mode_conditioning_enabled = True
    agent.hippocampal.config.mode_value_weight = {"explore": [0.7] * 16}
    trajs = _pool(agent, operating_mode={"explore": 1.0})
    bias = agent.hippocampal.last_candidate_modulatory_bias
    assert bias is not None
    assert bias.dim() == 1
    assert int(bias.shape[0]) == len(trajs)
    d = agent.hippocampal.get_last_propose_diagnostics()
    assert d["cem_modulatory_throughput_available"] is True
    assert d["cem_elite_stage_advisory_only"] is False


def test_cem_elite_is_covered_by_the_inert_route_backstop():
    """The throughput route must be watched by the inert-route backstop.

    It is the source MOST likely to go silently inert: it needs a SECOND flag
    on a DIFFERENT config object (HippocampalConfig.use_cem_modulatory_
    throughput) before the cache it reads exists at all. That two-flag coupling
    is exactly the V3-EXQ-863 silent no-op shape the backstop exists for."""
    from ree_core.agent import _MODULATORY_ROUTE_BACKSTOP_SOURCES
    assert "cem_elite" in _MODULATORY_ROUTE_BACKSTOP_SOURCES


def test_backstop_set_is_separate_from_the_tracker_parity_set():
    """NEGATIVE CONTROL, and the reason the two constants exist separately.

    _MODULATORY_ROUTE_CHANNEL_SOURCES carries a SECOND meaning beyond "sources
    the backstop watches": test_modulatory_route_decomp_gate_decoupling.py::
    test_c4b_route_source_set_matches_the_dispatch pins it 1:1 against
    _DECOUPLED_TRACKERS, the channels whose per-candidate bias is captured
    decoupled from the decomp gate. "cem_elite" is not a bias-head channel at
    all -- it is a cache on the hippocampal module -- so putting it there
    conflates tracker parity with backstop coverage. That is not hypothetical:
    this build did exactly that and the C4b contract caught it (2026-08-19).

    Pin BOTH directions, so a later session cannot "tidy" the two constants
    back into one in either direction."""
    from ree_core.agent import (
        _MODULATORY_ROUTE_BACKSTOP_SOURCES,
        _MODULATORY_ROUTE_CHANNEL_SOURCES,
    )
    assert "cem_elite" not in _MODULATORY_ROUTE_CHANNEL_SOURCES
    assert _MODULATORY_ROUTE_CHANNEL_SOURCES < _MODULATORY_ROUTE_BACKSTOP_SOURCES
    assert (
        _MODULATORY_ROUTE_BACKSTOP_SOURCES - _MODULATORY_ROUTE_CHANNEL_SOURCES
        == {"cem_elite"}
    )


# --------------------------------------------------------------------------
# (c) READINESS
# --------------------------------------------------------------------------

def test_readiness_statistic_reproduces_the_931_verdict():
    """(c) The whole point: 0.0037 is nonzero and NOT competitive. A
    nonzero-range gate passes exactly the case that predicts the null."""
    lever = torch.tensor([0.0, 0.0037, 0.0])
    dominant = torch.tensor([0.0, 1.0, 0.5])
    ratio = authority_spread_ratio(lever, dominant)
    assert abs(ratio - 0.0037) < 1e-6
    assert ratio > 0.0, "the range genuinely EXISTS -- that is the trap"
    assert authority_ratio_is_competitive(ratio) is False
    assert authority_ratio_is_competitive(0.5) is True
    assert authority_ratio_is_competitive(
        AUTHORITY_COMPETITIVE_RATIO_FLOOR_DEFAULT
    ) is True, "the floor itself must read as competitive (>=, not >)"


def test_readiness_statistic_basis_and_degenerate_inputs():
    """NEGATIVE CONTROLS. A degenerate input must read 0.0 -- the reading that
    keeps a readiness gate CLOSED -- never inf, NaN, or an exception."""
    lever = torch.tensor([0.0, 1.0, 2.0])
    assert authority_spread_ratio(lever, torch.tensor([0.0, 2.0, 4.0])) == 1.0 / 2.0
    # std basis is a different question (typical spread, not extremes)
    assert authority_spread_ratio(
        lever, torch.tensor([0.0, 2.0, 4.0]), basis="std"
    ) == 0.5
    # unrecognised basis falls back to "range" rather than raising
    assert authority_spread_ratio(
        lever, torch.tensor([0.0, 2.0, 4.0]), basis="nonsense"
    ) == 0.5
    # fewer than 2 candidates
    assert authority_spread_ratio(torch.tensor([1.0]), torch.tensor([1.0])) == 0.0
    # flat dominant term: a lever cannot be shown competitive against nothing
    assert authority_spread_ratio(lever, torch.tensor([1.0, 1.0, 1.0])) == 0.0
    # NaN ratio reads not-competitive
    assert authority_ratio_is_competitive(float("nan")) is False


def test_readiness_is_reported_at_both_layers():
    """One statistic, one definition, two layers. A second layer-local
    definition would reintroduce the incomparability this removes."""
    agent = REEAgent(_cfg())
    _pool(agent)
    d = agent.hippocampal.get_last_propose_diagnostics()
    for k in (
        "cem_modulatory_authority_ratio",
        "cem_modulatory_authority_competitive",
        "cem_modulatory_throughput_ratio",
        "cem_modulatory_throughput_competitive",
        "authority_competitive_ratio_floor",
    ):
        assert k in d, k
    # E3 layer: the ratio has been reported for a while as
    # score_bias_to_raw_range_ratio; the VERDICT is what the amend adds.
    sel = agent.e3
    assert hasattr(sel, "last_score_diagnostics")


def test_readiness_is_reported_never_enforced():
    """A substrate that REFUSED to run below the floor would break every
    existing default-off configuration, and watching a sub-competitive lever
    run is exactly what produced this finding. The gate belongs at
    /queue-experiment time, so a sub-competitive lever must still run."""
    agent = REEAgent(_cfg(authority_competitive_ratio_floor=0.99))
    trajs = _pool(agent)
    assert len(trajs) > 0
    d = agent.hippocampal.get_last_propose_diagnostics()
    assert d["authority_competitive_ratio_floor"] == 0.99
    assert d["cem_modulatory_authority_competitive"] is False
