"""Contract tests: ARC-108 JOB-2 control-plane DRIVER pair (the dopaminergic
driver of the commit/maintain/de-commit machinery REE built but never gave its
neuromodulator). unified_dopamine_substrate_design_2026-06-22.md secs 3-6.

Two no-op-default pieces, both composing with MECH-090/342/SD-034 (gate +
operator + refractory kept as safety plumbing; no parallel module, ARC-106 G2):

  (c) rho_t MAINTENANCE RAMP -- a goal-proximity x value ramp that REPLACES the
      flat (unconditional) re-assert DRIVER of the natural-commit latch-hold so
      the hold self-limits past the proximity peak (the structural B6 / 460h-
      monolith fix), instead of running monolithically.
  (d) HABENULA negative-delta_t DE-COMMIT -- a NEW abort input to the SD-034
      ClosureOperator: a negative phasic RPE (the same signed delta_t = R_t -
      V-hat_t the ARC-108 JOB-1 slice computes) fires a content-driven de-commit,
      dissociable from the latch's refractory state.

Contracts (interface-level guarantees, NOT magnitude thresholds):

  C1  config defaults: all 6 JOB-2 flags default no-op; from_dims surfaces them;
      master OFF -> agent.rho_maintenance_ramp is None and (when a closure
      operator is built) its habenula abort is disabled -> bit-identical.
  C2  rho ramp UNIT peaks-then-declines: holds through the onset grace and while
      rho rises, self-limits once rho declines past release_margin * peak; also
      releases below hold_floor. A FLAT (non-declining) rho never self-limits.
  C3  rho ramp MECH-094: tick(simulation_mode=True) never self-limits and does
      not advance the peak.
  C4  habenula_tick UNIT: fires the SD-034 closure (beta released,
      n_habenula_aborts increments) on a negative delta_t below threshold while
      beta is elevated; no-op when the abort is disabled / beta not elevated /
      delta_t >= threshold / hypothesis_tag.
  C5  agent preconditions + wiring: use_rho_maintenance_ramp without the latch-
      hold raises; with it builds the ramp; use_habenula_decommit forwards onto
      the ClosureOperatorConfig.
  C6  rho ramp REPLACES the flat driver (LOAD-BEARING, agent-level): with the
      latch-hold armed and a DECLINING rho_t, the ramp ON self-limits the hold
      (disarms) where the flat hold (ramp OFF) keeps re-asserting against churn.
  C7  e3 delta_t reuse: with JOB-2 ON (JOB-1 OFF) post_action_update emits
      habenula_delta_t and advances the shared V-hat_t; with JOB-2 OFF it emits
      no habenula_delta_t (JOB-1 path bit-identical).
  C8  agent habenula de-commit end-to-end: a committed+elevated agent whose
      realised outcome is worse-than-expected (negative delta_t) fires the
      habenula abort, releasing beta and tearing down the committed program.
"""

from __future__ import annotations

import torch

from ree_core.agent import REEAgent
from ree_core.governance.closure_operator import (
    ClosureOperator,
    ClosureOperatorConfig,
)
from ree_core.heartbeat.beta_gate import BetaGate
from ree_core.policy.rho_maintenance_ramp import (
    RhoMaintenanceRamp,
    RhoMaintenanceRampConfig,
)
from ree_core.predictors.e2_fast import Trajectory
from ree_core.predictors.e3_selector import SelectionResult
from ree_core.utils.config import REEConfig

ACTION_DIM = 4
SELF_DIM = 8
WORLD_DIM = 8
BODY_OBS_DIM = 4
WORLD_OBS_DIM = 8
H = 3


def _one_hot(i: int) -> torch.Tensor:
    a = torch.zeros(1, H, ACTION_DIM)
    a[:, :, i] = 1.0
    return a


def _traj(i: int, off: float) -> Trajectory:
    states = [torch.full((1, SELF_DIM), off + 0.01 * k) for k in range(H + 1)]
    world = [torch.full((1, WORLD_DIM), off + 0.02 * k) for k in range(H + 1)]
    return Trajectory(states=states, actions=_one_hot(i), world_states=world)


def _candidates():
    return [_traj(1, 0.1), _traj(2, 0.2)]


class _Stub:
    def __init__(self, committed: bool = True):
        self.committed = committed
        self.scores = torch.tensor([0.0, 1.0, 2.0])

    def select(self, candidates, temperature: float = 1.0, **kw):
        return SelectionResult(
            selected_trajectory=candidates[0],
            selected_index=0,
            selected_action=candidates[0].actions[:, 0, :],
            scores=self.scores.clone(),
            precision=1.0,
            committed=self.committed,
            log_prob=torch.tensor(0.0),
            urgency=0.0,
        )


def _build_agent(**kwargs) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        use_sleep_loop=False,
        sws_enabled=False,
        rem_enabled=False,
        use_sleep_aggregation_cluster=False,
        **kwargs,
    )
    cfg.heartbeat.beta_gate_bistable = True
    agent = REEAgent(cfg)
    agent.reset()
    return agent


# ----------------------------------------------------------------------
# C1 config defaults + bit-identical OFF
# ----------------------------------------------------------------------
def test_c1_config_defaults_and_master_off():
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
    )
    assert cfg.use_rho_maintenance_ramp is False
    assert cfg.rho_hold_floor == 0.05
    assert cfg.rho_release_margin == 0.5
    assert cfg.rho_onset_grace_ticks == 3
    assert cfg.use_habenula_decommit is False
    assert cfg.habenula_decommit_delta_threshold == 0.0

    cfg2 = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        use_rho_maintenance_ramp=True,
        use_natural_commit_latch_hold=True,
        rho_release_margin=0.25,
        use_habenula_decommit=True,
        habenula_decommit_delta_threshold=-0.1,
    )
    assert cfg2.use_rho_maintenance_ramp is True
    assert cfg2.rho_release_margin == 0.25
    assert cfg2.use_habenula_decommit is True
    assert cfg2.habenula_decommit_delta_threshold == -0.1

    # Master OFF agent: ramp not built; closure-operator habenula abort disabled.
    off = _build_agent(use_closure_operator=True, use_lateral_pfc_analog=True)
    assert off.rho_maintenance_ramp is None
    assert off.closure_operator is not None
    assert off.closure_operator.config.habenula_abort_enabled is False


# ----------------------------------------------------------------------
# C2 rho ramp UNIT: peaks-then-declines self-limit
# ----------------------------------------------------------------------
def test_c2_rho_ramp_peaks_then_declines():
    ramp = RhoMaintenanceRamp(
        RhoMaintenanceRampConfig(
            use_rho_maintenance_ramp=True,
            hold_floor=0.05,
            release_margin=0.5,
            onset_grace_ticks=2,
        )
    )
    ramp.note_commit_entry()
    # Onset grace (ticks 1,2): never release even on a dip.
    assert ramp.tick(0.1) is False
    assert ramp.tick(0.2) is False
    # Rising past grace: peak climbs, no decline -> hold.
    assert ramp.tick(0.6) is False  # tick 3, peak 0.6
    assert ramp.tick(1.0) is False  # tick 4, peak 1.0
    assert ramp.tick(0.8) is False  # decline 0.2 < 0.5*1.0 -> hold
    # Decline past the proximity peak by >= margin*peak -> self-limit.
    assert ramp.tick(0.4) is True   # decline 0.6 >= 0.5*1.0 -> RELEASE
    assert ramp.is_active is False
    st = ramp.get_state()
    assert st["rho_n_releases"] == 1
    assert st["rho_peak"] == 1.0

    # A FLAT (non-declining) rho NEVER self-limits (the flat-hold monopoly the
    # ramp fixes; a flat hold has no decline term to cross the test).
    flat = RhoMaintenanceRamp(
        RhoMaintenanceRampConfig(
            use_rho_maintenance_ramp=True, onset_grace_ticks=1
        )
    )
    flat.note_commit_entry()
    for _ in range(30):
        assert flat.tick(0.7) is False
    assert flat.is_active is True
    assert flat.get_state()["rho_n_releases"] == 0

    # Below the floor -> release (no value left to maintain).
    floor = RhoMaintenanceRamp(
        RhoMaintenanceRampConfig(
            use_rho_maintenance_ramp=True, hold_floor=0.1, onset_grace_ticks=0
        )
    )
    floor.note_commit_entry()
    assert floor.tick(0.02) is True


# ----------------------------------------------------------------------
# C3 rho ramp MECH-094
# ----------------------------------------------------------------------
def test_c3_rho_ramp_simulation_no_op():
    ramp = RhoMaintenanceRamp(
        RhoMaintenanceRampConfig(
            use_rho_maintenance_ramp=True, onset_grace_ticks=0
        )
    )
    ramp.note_commit_entry()
    ramp.tick(1.0)  # establish a peak
    # A simulation tick with a deep dip must NOT self-limit nor move the peak.
    assert ramp.tick(0.0, simulation_mode=True) is False
    assert ramp.get_state()["rho_peak"] == 1.0
    assert ramp.get_state()["rho_n_simulation_skips"] == 1
    assert ramp.is_active is True


# ----------------------------------------------------------------------
# C4 habenula_tick UNIT
# ----------------------------------------------------------------------
def _operator(enabled: bool, threshold: float = 0.0) -> tuple:
    bg = BetaGate(initial_beta_elevated=False)
    op = ClosureOperator(
        config=ClosureOperatorConfig(
            use_closure_operator=True,
            habenula_abort_enabled=enabled,
            habenula_delta_threshold=threshold,
        ),
        beta_gate=bg,
    )
    return op, bg


def test_c4_habenula_tick_fires_and_no_ops():
    zw = torch.zeros(1, WORLD_DIM)

    # Fires: enabled + beta elevated + delta_t < threshold (0.0).
    op, bg = _operator(enabled=True)
    bg.elevate()
    ev = op.habenula_tick(delta_t=-0.5, z_world=zw, action_class=1)
    assert ev.fired is True
    assert ev.reason == "habenula"
    assert ev.beta_released is True
    assert bg.is_elevated is False  # the de-commit released the latch
    assert op.get_state()["n_habenula_aborts"] == 1

    # No-op: disabled.
    op2, bg2 = _operator(enabled=False)
    bg2.elevate()
    ev2 = op2.habenula_tick(delta_t=-0.5, z_world=zw)
    assert ev2.fired is False and ev2.reason == "skipped:habenula_disabled"
    assert bg2.is_elevated is True

    # No-op: beta not elevated (nothing to de-commit).
    op3, bg3 = _operator(enabled=True)
    ev3 = op3.habenula_tick(delta_t=-0.5, z_world=zw)
    assert ev3.fired is False and ev3.reason == "skipped:beta_not_elevated"

    # No-op: delta_t at/above threshold (outcome not worse-than-expected).
    op4, bg4 = _operator(enabled=True)
    bg4.elevate()
    ev4 = op4.habenula_tick(delta_t=0.2, z_world=zw)
    assert ev4.fired is False and ev4.reason.startswith("skipped:delta_above")
    assert bg4.is_elevated is True

    # No-op: MECH-094 hypothesis_tag (a replay outcome must not abort).
    op5, bg5 = _operator(enabled=True)
    bg5.elevate()
    ev5 = op5.habenula_tick(delta_t=-0.5, z_world=zw, hypothesis_tag=True)
    assert ev5.fired is False and ev5.reason == "skipped:hypothesis_tag"
    assert bg5.is_elevated is True

    # Threshold respected: delta between threshold and 0 fires when threshold>delta.
    op6, bg6 = _operator(enabled=True, threshold=-0.3)
    bg6.elevate()
    assert op6.habenula_tick(delta_t=-0.1, z_world=zw).fired is False  # -0.1 > -0.3
    assert op6.habenula_tick(delta_t=-0.4, z_world=zw).fired is True   # -0.4 < -0.3


# ----------------------------------------------------------------------
# C5 agent preconditions + wiring
# ----------------------------------------------------------------------
def test_c5_agent_preconditions_and_wiring():
    # rho ramp without the latch-hold -> loud ValueError.
    raised = False
    try:
        _build_agent(use_rho_maintenance_ramp=True)
    except ValueError as e:
        raised = "use_natural_commit_latch_hold" in str(e)
    assert raised, "rho ramp must require the latch-hold"

    # With the latch-hold -> ramp built.
    on = _build_agent(
        use_rho_maintenance_ramp=True, use_natural_commit_latch_hold=True
    )
    assert on.rho_maintenance_ramp is not None

    # Habenula flag forwarded onto the ClosureOperatorConfig.
    hab = _build_agent(
        use_closure_operator=True,
        use_lateral_pfc_analog=True,
        use_habenula_decommit=True,
        habenula_decommit_delta_threshold=-0.2,
    )
    assert hab.closure_operator.config.habenula_abort_enabled is True
    assert hab.closure_operator.config.habenula_delta_threshold == -0.2


# ----------------------------------------------------------------------
# C6 rho ramp REPLACES the flat driver (LOAD-BEARING)
# ----------------------------------------------------------------------
def _arm_hold(agent, cands):
    agent.e3.select = _Stub(committed=True).select
    agent.e3._running_variance = 0.0
    agent.e3.last_scores = torch.tensor([0.0, 1.0, 2.0])
    agent.e3._committed_trajectory = cands[0]
    agent.select_action(cands, {"e3_tick": True})


def test_c6_rho_ramp_replaces_flat_driver():
    cands = _candidates()

    # Ramp ON: arm the hold, then feed a DECLINING rho_t. The ramp self-limits
    # the hold (disarms) once rho falls past the proximity peak.
    on = _build_agent(
        use_rho_maintenance_ramp=True, use_natural_commit_latch_hold=True
    )
    _arm_hold(on, cands)
    assert on._ncl_hold_active is True
    # Scripted rise-then-fall rho_t (replaces the env-derived proximity x value).
    seq = [0.5, 1.0, 1.0, 0.9, 0.7, 0.3, 0.1]
    it = iter(seq)
    on._compute_rho_t = lambda: next(it, 0.0)
    stub_off = _Stub(committed=False)  # no re-commit; only the hold sustains
    disarmed_at = None
    for t in range(len(seq)):
        on.e3.select = stub_off.select
        on.e3.last_scores = stub_off.scores.clone()
        on.e3._committed_trajectory = cands[0]
        on.select_action(cands, {"e3_tick": True})
        if not on._ncl_hold_active and disarmed_at is None:
            disarmed_at = t
    assert disarmed_at is not None, "rho ramp must self-limit the hold on decline"
    assert on.rho_maintenance_ramp.get_state()["rho_n_releases"] >= 1

    # Ramp OFF (flat hold): same arm + same churn, the hold keeps re-asserting
    # (never self-limits) -- the monolithic-hold behaviour the ramp fixes.
    off = _build_agent(use_natural_commit_latch_hold=True)
    _arm_hold(off, cands)
    assert off._ncl_hold_active is True
    stub_off2 = _Stub(committed=False)
    for _ in range(len(seq)):
        off.beta_gate.release()  # churn drops the latch
        off.e3.select = stub_off2.select
        off.e3.last_scores = stub_off2.scores.clone()
        off.e3._committed_trajectory = cands[0]
        off.select_action(cands, {"e3_tick": True})
    assert off._ncl_hold_active is True, "flat hold must not self-limit"
    assert off._ncl_hold_reassert_count >= 1


# ----------------------------------------------------------------------
# C7 e3 delta_t reuse (JOB-1 path bit-identical with JOB-2 off)
# ----------------------------------------------------------------------
def _force_neg_delta(agent):
    # delta_t = R_t - V-hat_t. Push V-hat_t high so the realised outcome is
    # "worse than expected" (delta_t strongly negative) regardless of the
    # (nn.Module) valuation heads -- the habenula de-commit trigger.
    agent.e3._lcg_value_baseline = 100.0


def test_c7_e3_delta_t_reuse_and_job1_unchanged():
    cands = _candidates()

    # JOB-2 ON (JOB-1 OFF): post_action_update emits habenula_delta_t + advances
    # the shared V-hat_t even with no JOB-1 eligibility trace.
    on = _build_agent(use_habenula_decommit=True)
    _force_neg_delta(on)
    on.e3._last_selected_trajectory = cands[0]
    v0 = on.e3._lcg_value_baseline
    m = on.e3.post_action_update(actual_z_world=torch.zeros(1, WORLD_DIM),
                                 harm_occurred=False)
    assert "habenula_delta_t" in m
    assert float(m["habenula_delta_t"].item()) < 0.0  # worse-than-expected
    assert on.e3._lcg_value_baseline != v0  # V-hat advanced toward R_t

    # JOB-2 OFF: no habenula_delta_t emitted; the JOB-1 w_chan path is untouched
    # (no learned gating -> w_chan stays at init).
    off = _build_agent()
    off.e3._last_selected_trajectory = cands[0]
    w_before = off.e3.w_chan.clone()
    m2 = off.e3.post_action_update(actual_z_world=torch.zeros(1, WORLD_DIM),
                                   harm_occurred=False)
    assert "habenula_delta_t" not in m2
    assert torch.equal(off.e3.w_chan, w_before)


# ----------------------------------------------------------------------
# C8 agent habenula de-commit end-to-end
# ----------------------------------------------------------------------
def test_c8_agent_habenula_decommit_end_to_end():
    agent = _build_agent(
        use_closure_operator=True,
        use_lateral_pfc_analog=True,
        use_habenula_decommit=True,
    )
    cands = _candidates()
    # Populate _current_latent + a committed, elevated state.
    obs = torch.zeros(1, BODY_OBS_DIM + WORLD_OBS_DIM)
    agent.sense_flat(obs)
    _force_neg_delta(agent)
    agent.e3._committed_trajectory = cands[0]
    agent.e3._last_selected_trajectory = cands[0]
    agent._last_action = cands[0].actions[:, 0, :]
    agent.beta_gate.elevate()
    assert agent.beta_gate.is_elevated is True

    metrics = agent.update_residue(harm_signal=0.0)
    assert "habenula_decommit_fired" in metrics
    assert agent.closure_operator.get_state()["n_habenula_aborts"] == 1
    assert agent.beta_gate.is_elevated is False  # the de-commit released beta
    assert agent.e3._committed_trajectory is None  # program torn down
