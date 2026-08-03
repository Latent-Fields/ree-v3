"""
SD-modulatory-channel-route-decomp-gate-fix -- substrate contract.

WHAT WAS BROKEN. `modulatory_channel_route_source` dispatch in
`REEAgent.select_action` identity-routes an already-computed per-candidate [K]
channel bias to E3 via `channel_route_bias`. It reads that bias out of the
`_bdc_*` trackers -- but four of them (`_bdc_lpfc`, `_bdc_gp`, `_bdc_m295`,
`_bdc_curiosity`) were only ASSIGNED under `if self.e3.e3_score_decomp_enabled:`,
an unrelated V3-EXQ-571 diagnostic flag that defaults False and that none of the
confirmed drivers set. So four of the six route sources were silently inert: the
route resolved to None, `channel_route_bias` stayed None, and the only trace was
a `modulatory_channel_route_range` of exactly 0.0 -- indistinguishable in a
manifest from a channel that is genuinely undifferentiated.

Confirmed by V3-EXQ-863 (autopsy
`failure_autopsy_V3-EXQ-863-route-decomp-gate_2026-08-02`):
`modulatory_channel_route_range_mean` and `modulatory_channel_route_active_frac`
both read 0.0 on a correctly-configured `lateral_pfc` arm whose underlying
per-candidate bias was independently measured nonzero and differentiated by
V3-EXQ-851's own `mean_lateral_pfc_bias_abs` readout.

WHAT THIS FILE PINS.
  C1 behaviour   -- lateral_pfc route + `e3_score_decomp_enabled=False` (the
                    common case, and the exact V3-EXQ-863 configuration) yields
                    a NONZERO route_range when the underlying bias is genuinely
                    differentiated. This is the assertion that fails on the
                    pre-fix substrate (verified: 0.0 / inactive on every tick).
  C2 decoupling  -- route_range is IDENTICAL with the diagnostic flag ON and
                    OFF. The sharpest pin: the selection path must not depend on
                    a diagnostic flag at all, in either direction.
  C3 OFF control -- `route_source="none"` is still inert (route_range 0.0,
                    inactive), so the default path is unchanged.
  C4 all four    -- a STRUCTURAL check that none of the four `_bdc_*` capture
                    sites is guarded by `e3_score_decomp_enabled`. C1/C2 can only
                    afford to boot one channel; this covers the other three (and
                    any future re-coupling) without four agent configurations.
  C5 exposure    -- the V3-EXQ-571 diagnostic PAYLOAD is unchanged: the decomp
                    dict is still written only when the flag is ON, and it still
                    carries the per-channel entries. Only the CAPTURE was
                    decoupled, not the exposure.
  C6 backstop    -- a configured channel route that keeps yielding nothing warns
                    at runtime, so any remaining coupling of this shape surfaces
                    during the run instead of in a post-hoc autopsy.
"""

import ast
import warnings
from pathlib import Path

import torch

from ree_core.agent import (
    _MODULATORY_ROUTE_CHANNEL_SOURCES,
    _ROUTE_SOURCE_EMPTY_WARN_TICKS,
    REEAgent,
)
from ree_core.utils.config import REEConfig

_AGENT_SRC = Path(__file__).resolve().parents[2] / "ree_core" / "agent.py"

# The four capture sites decoupled by this fix, keyed by the tracker they assign.
_DECOUPLED_TRACKERS = {
    "_bdc_lpfc": "_bdc_keep_lpfc",
    "_bdc_gp": "_bdc_keep_gp",
    "_bdc_m295": "_bdc_keep_m295",
    "_bdc_curiosity": "_bdc_keep_curiosity",
}

_FLOOR = 1e-6

# Env steps for the C6 backstop cases. The threshold counts E3 SELECT calls, not
# env steps, and a committed policy chunk covers several steps per select -- so
# reaching the raw threshold from a cold agent would cost ~40 env steps of
# contract time for a property that needs exactly one select. Those cases seed
# the counter to threshold-1 instead and take the last step here; the first tick
# always selects (nothing is committed yet), and each case carries an explicit
# non-vacuity assertion so a future change that stops selecting fails loudly
# rather than passing on an empty run.
_BACKSTOP_TICKS = 3


def _make_env(seed=7):
    from ree_core.environment.causal_grid_world import CausalGridWorldV2

    return CausalGridWorldV2(
        seed=seed, size=6, num_hazards=2, num_resources=2, use_proxy_fields=True
    )


def _build_agent(env, route_source, decomp_enabled=False, **overrides):
    """Minimal faithful stand-in for the V3-EXQ-863 lateral_pfc arm."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        self_dim=16,
        world_dim=16,
        use_lateral_pfc_analog=True,
        use_candidate_rule_field=True,
        use_modulatory_channel_routing=True,
        modulatory_channel_route_source=route_source,
        use_modulatory_selection_authority=True,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    # the authority + routing read both config surfaces
    cfg.e3.use_modulatory_channel_routing = True
    cfg.e3.use_modulatory_selection_authority = True
    agent = REEAgent(cfg)
    agent.reset()
    agent.e3.e3_score_decomp_enabled = bool(decomp_enabled)
    return agent


def _differentiate_lateral_pfc(agent):
    """Force a genuinely differentiated per-candidate lateral_pfc bias.

    An untrained rule_bias_head has its last Linear zeroed at init, so a
    contract-speed agent cannot be relied on to differentiate on its own. The
    property under test is the ROUTING of a differentiated bias, not whether an
    untrained head produces one -- V3-EXQ-851 already measured that it does.
    """
    agent.lateral_pfc.compute_bias = lambda cws: torch.linspace(
        -0.5, 0.5, cws.shape[0]
    )


def _run(agent, env, n=6, seed=7):
    """Tick the agent, returning (route_active, route_range) per tick."""
    torch.manual_seed(seed)
    _f, obs = env.reset()
    out = []
    for _ in range(n):
        body = obs["body_state"]
        world = obs["world_state"]
        if body.dim() == 1:
            body = body.unsqueeze(0)
        if world.dim() == 1:
            world = world.unsqueeze(0)
        with torch.no_grad():
            act = agent.act_with_split_obs(obs_body=body, obs_world=world)
        diag = getattr(agent.e3, "last_score_diagnostics", None) or {}
        out.append(
            (
                bool(diag.get("modulatory_channel_route_active", False)),
                float(diag.get("modulatory_channel_route_range", 0.0)),
            )
        )
        _f, _r, _d, _i, obs = env.step(int(act.reshape(-1).argmax().item()))
    return out


# ----------------------------------------------------------------------
# C1: the confirmed V3-EXQ-863 configuration now routes
# ----------------------------------------------------------------------
def test_c1_lateral_pfc_route_range_nonzero_with_decomp_disabled():
    env = _make_env()
    agent = _build_agent(env, "lateral_pfc", decomp_enabled=False)
    assert agent.e3.e3_score_decomp_enabled is False, (
        "this contract is about the DEFAULT diagnostic-flag-OFF case"
    )
    _differentiate_lateral_pfc(agent)
    ticks = _run(agent, env)

    assert all(rng > _FLOOR for _a, rng in ticks), (
        "lateral_pfc route_range read "
        f"{[r for _a, r in ticks]} with e3_score_decomp_enabled=False -- a route "
        "carrying a differentiated bias must not depend on a diagnostic flag "
        "(this is the V3-EXQ-863 failure signature: exactly 0.0 on every tick)"
    )
    assert all(active for active, _r in ticks), (
        "modulatory_channel_route_active must be True once the route carries "
        "range above the min-range floor"
    )


# ----------------------------------------------------------------------
# C2: the route no longer depends on the diagnostic flag, in either direction
# ----------------------------------------------------------------------
def test_c2_route_range_independent_of_decomp_flag():
    off = _make_env()
    agent_off = _build_agent(off, "lateral_pfc", decomp_enabled=False)
    _differentiate_lateral_pfc(agent_off)
    ticks_off = _run(agent_off, off)

    on = _make_env()
    agent_on = _build_agent(on, "lateral_pfc", decomp_enabled=True)
    _differentiate_lateral_pfc(agent_on)
    ticks_on = _run(agent_on, on)

    assert ticks_off == ticks_on, (
        f"route diagnostics differ with the decomp flag ON ({ticks_on}) vs OFF "
        f"({ticks_off}) -- e3_score_decomp_enabled is a DIAGNOSTIC flag and must "
        "not change the selection path"
    )


# ----------------------------------------------------------------------
# C3: the default route source stays inert (no accidental always-on routing)
# ----------------------------------------------------------------------
def test_c3_route_source_none_still_inert():
    env = _make_env()
    agent = _build_agent(env, "none", decomp_enabled=False)
    _differentiate_lateral_pfc(agent)
    ticks = _run(agent, env)

    assert all(not active and rng == 0.0 for active, rng in ticks), (
        f"route_source='none' must stay inert, got {ticks} -- decoupling the "
        "capture must not make routing fire when it was not configured"
    )


# ----------------------------------------------------------------------
# C4: NONE of the four capture sites is guarded by the diagnostic flag
# ----------------------------------------------------------------------
def test_c4_no_route_tracker_capture_is_decomp_gated():
    tree = ast.parse(_AGENT_SRC.read_text())
    found = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        guard = ast.dump(node.test)
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            for tgt in stmt.targets:
                if isinstance(tgt, ast.Name) and tgt.id in _DECOUPLED_TRACKERS:
                    found.setdefault(tgt.id, []).append(guard)

    missing = sorted(set(_DECOUPLED_TRACKERS) - set(found))
    assert not missing, (
        f"no guarded capture site found for {missing} -- if these were renamed "
        "or unguarded entirely, update this contract deliberately rather than "
        "letting it pass vacuously"
    )

    for tracker, guards in sorted(found.items()):
        for guard in guards:
            assert "e3_score_decomp_enabled" not in guard, (
                f"{tracker} capture is gated on e3_score_decomp_enabled again. "
                "That flag is a V3-EXQ-571 DIAGNOSTIC (default False), but this "
                "tracker is read by the modulatory_channel_route_source dispatch, "
                "which is a load-bearing SELECTION path -- gating it there makes "
                "the route silently inert (V3-EXQ-863)."
            )
            assert _DECOUPLED_TRACKERS[tracker] in guard, (
                f"{tracker} capture is no longer guarded by "
                f"{_DECOUPLED_TRACKERS[tracker]}, the predicate that ORs the "
                "diagnostic flag with 'routing selected this channel'"
            )


def test_c4b_route_source_set_matches_the_dispatch():
    """The backstop's source set must stay aligned with the tracker set."""
    assert _MODULATORY_ROUTE_CHANNEL_SOURCES == {
        "lateral_pfc",
        "gated_policy",
        "mech295",
        "curiosity",
    }, (
        "the four decoupled channel route sources. 'coherence' is deliberately "
        "excluded (never decomp-gated, and its None is a designed MECH-294 "
        "outcome); 'cand_world_summary' builds its own representation."
    )
    assert len(_MODULATORY_ROUTE_CHANNEL_SOURCES) == len(_DECOUPLED_TRACKERS)


# ----------------------------------------------------------------------
# C5: the V3-EXQ-571 diagnostic exposure is unchanged
# ----------------------------------------------------------------------
def test_c5_decomp_exposure_still_gated_on_the_flag():
    off = _make_env()
    agent_off = _build_agent(off, "lateral_pfc", decomp_enabled=False)
    _differentiate_lateral_pfc(agent_off)
    _run(agent_off, off, n=3)
    assert not agent_off._last_score_bias_decomp, (
        "the decomp diagnostic payload must stay EMPTY when the flag is OFF -- "
        "only the capture was decoupled, not the exposure"
    )

    on = _make_env()
    agent_on = _build_agent(on, "lateral_pfc", decomp_enabled=True)
    _differentiate_lateral_pfc(agent_on)
    _run(agent_on, on, n=3)
    decomp = agent_on._last_score_bias_decomp
    assert decomp, "the decomp payload must still be written when the flag is ON"
    for key in (
        "lateral_pfc",
        "gated_policy",
        "mech295_liking",
        "curiosity",
        "lateral_pfc_bias_range_mean",
    ):
        assert key in decomp, f"V3-EXQ-571 payload lost the '{key}' entry"
    assert decomp["lateral_pfc_bias_range_mean"] > _FLOOR, (
        "the differentiated lateral_pfc bias must still reach the diagnostic"
    )


# ----------------------------------------------------------------------
# C6: an inert configured route warns rather than failing silently
# ----------------------------------------------------------------------
def _route_warnings(agent, env, n):
    """Run and return only this backstop's warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run(agent, env, n=n)
    return [
        str(w.message)
        for w in caught
        if issubclass(w.category, UserWarning)
        and "modulatory_channel_route_source" in str(w.message)
    ]


def test_c6_persistently_empty_route_warns():
    env = _make_env()
    # "curiosity" route with the curiosity module absent -> the tracker can
    # never be populated, which is exactly the inert-route shape.
    agent = _build_agent(env, "curiosity", decomp_enabled=False)
    assert agent.curiosity is None, (
        "this control needs the curiosity channel absent so the route is starved"
    )
    # Seed to one below the threshold -- see _BACKSTOP_TICKS. The property under
    # test is that the threshold fires, not how many env steps it takes to reach.
    agent._route_source_empty_ticks = _ROUTE_SOURCE_EMPTY_WARN_TICKS - 1
    msgs = _route_warnings(agent, env, n=_BACKSTOP_TICKS)
    assert msgs, (
        "a configured channel route that never produces a bias must warn -- the "
        "V3-EXQ-863 failure cost an entire experiment precisely because it was "
        "silent"
    )
    assert "curiosity" in msgs[0], "the warning must name the inert route source"


def test_c6b_healthy_route_does_not_warn():
    env = _make_env()
    agent = _build_agent(env, "lateral_pfc", decomp_enabled=False)
    _differentiate_lateral_pfc(agent)
    # Pre-seeded ONE BELOW the threshold, so this is a real negative control: a
    # healthy route must RESET the counter, not merely fail to reach the floor.
    agent._route_source_empty_ticks = _ROUTE_SOURCE_EMPTY_WARN_TICKS - 1
    msgs = _route_warnings(agent, env, n=_BACKSTOP_TICKS)
    assert not msgs, (
        f"a working route must not warn (a backstop that fires on healthy "
        f"configurations gets ignored): {msgs}"
    )
    assert agent._route_source_empty_ticks == 0, (
        "a route that produced a bias must reset the empty-tick counter"
    )


def test_c6c_default_route_source_never_warns():
    """route_source='none' is not a channel source -- the backstop must ignore it."""
    env = _make_env()
    agent = _build_agent(env, "none", decomp_enabled=False)
    agent._route_source_empty_ticks = _ROUTE_SOURCE_EMPTY_WARN_TICKS - 1
    assert not _route_warnings(agent, env, n=_BACKSTOP_TICKS), (
        "the default (unrouted) configuration must never emit this warning"
    )
    # non-vacuity: the E3 select path really ran, and left the counter alone
    assert getattr(agent.e3, "last_score_diagnostics", None), (
        "vacuous -- E3 select never ran, so nothing was actually exercised"
    )
    assert agent._route_source_empty_ticks == _ROUTE_SOURCE_EMPTY_WARN_TICKS - 1, (
        "an unrouted configuration must not touch the backstop counter at all"
    )
