"""Contract tests for the MECH-039 two-part veto readout.

User decision rec-20260925-3b215584 (MECH-039 option 1): the hard-veto /
interrupt channel is read from EXISTING producers, split into an INTERRUPT
part (coordinator-wired: SD-035 CeA, SD-037 override_signal) and a CONTROL
part (local: MECH-279 PAG freeze, ARC-108 habenula abort, MECH-449 safety
No-Go). Read-only, default OFF (config.use_mech039_veto_readout).

C1  default OFF -> flag False, readout empty after a real run.
C2  from_dims threads both knobs (from_dims silently swallows unknown kwargs).
C3  bit-identical: ON vs OFF emit identical action streams under matched
    seeds, with the veto producers + coordinator armed (pure telemetry).
C4  ON over a real run -> one record per select_action exit; E3 / coordinator
    tick counts are read from the live path, not supplied by the test.
C5  onset / latency / censoring accounting (the MECH-046 shared
    time-to-mode-switch arm): an interrupt onset followed two steps later by
    a coordinator switch records latency 2; an unresolved onset is censored
    at the episode boundary; a habenula abort is a CONTROL onset.
"""

import random

import numpy as np
import torch

from ree_core.agent import REEAgent
from ree_core.amygdala.cea import CeAOutput
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments._harness import StepHarness


PRODUCERS = dict(
    use_amygdala_analog=True,
    use_cea_analog=True,
    use_broadcast_override=True,
    use_salience_coordinator=True,
    use_pag_freeze_gate=True,
    use_closure_operator=True,
    use_lateral_pfc_analog=True,
    use_habenula_decommit=True,
)


def _mk_env(seed):
    return CausalGridWorldV2(size=8, num_hazards=4, num_resources=2, seed=seed)


def _dims(env):
    return dict(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
    )


def _run(cfg, steps=24, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env = _mk_env(seed)
    agent = REEAgent(cfg)
    results = StepHarness(agent, env, train_mode=True, seed=seed).run_episode(
        max_steps=steps
    )
    return agent, [int(r.action.argmax().item()) for r in results]


def test_c1_default_off_readout_empty():
    env = _mk_env(0)
    cfg = REEConfig.from_dims(**_dims(env), **PRODUCERS)
    assert cfg.use_mech039_veto_readout is False
    agent, _ = _run(cfg, steps=12)
    r = agent.get_veto_readout()
    assert r["last"] == {}
    assert r["summary"]["n_steps"] == 0


def test_c2_from_dims_threads_knobs():
    env = _mk_env(0)
    cfg = REEConfig.from_dims(
        **_dims(env),
        use_mech039_veto_readout=True,
        veto_readout_override_onset_threshold=0.37,
    )
    assert cfg.use_mech039_veto_readout is True
    assert abs(cfg.veto_readout_override_onset_threshold - 0.37) < 1e-12


def test_c3_on_vs_off_bit_identical_actions():
    env = _mk_env(0)
    off = REEConfig.from_dims(**_dims(env), **PRODUCERS)
    on = REEConfig.from_dims(
        **_dims(env), **PRODUCERS, use_mech039_veto_readout=True
    )
    _, a_off = _run(off, steps=24, seed=3)
    agent_on, a_on = _run(on, steps=24, seed=3)
    assert a_off == a_on
    # and the ON run actually recorded (guard against a vacuous identity)
    assert agent_on.get_veto_readout()["summary"]["n_steps"] > 0


def test_c4_on_records_every_select_exit_from_live_path():
    env = _mk_env(0)
    cfg = REEConfig.from_dims(
        **_dims(env), **PRODUCERS, use_mech039_veto_readout=True
    )
    agent, actions = _run(cfg, steps=24, seed=1)
    r = agent.get_veto_readout()
    s = r["summary"]
    assert s["n_steps"] == len(actions)
    assert s["n_e3_ticks"] >= 1
    assert 1 <= s["n_coord_ticks"] <= s["n_e3_ticks"]
    last = r["last"]
    for key in (
        "cea_mode_prior", "override_signal", "interrupt_active",
        "freeze_active", "habenula_abort_fired", "gng_safety_active",
        "control_active", "coord_current_mode", "channels",
    ):
        assert key in last
    assert last["site"] in ("e3", "between_e3", "e3_shortcircuit")
    assert last["coord_current_mode"] is not None


def _logic_agent():
    env = _mk_env(0)
    cfg = REEConfig.from_dims(
        **_dims(env),
        use_amygdala_analog=True,
        use_cea_analog=True,
        use_salience_coordinator=True,
        use_mech039_veto_readout=True,
    )
    agent = REEAgent(cfg)
    assert agent.cea is not None and agent.salience is not None
    return agent


def _tick(agent, *, cea=None, switch=False):
    agent._cea_last_output = cea if cea is not None else CeAOutput()
    agent._salience_last_tick = {
        "mode_switch_trigger": switch,
        "operating_mode": {"external_task": 1.0},
    }
    agent._record_veto_readout(e3_tick=True, site="e3")
    return agent.get_veto_readout()


def test_c5_onset_latency_and_censoring():
    agent = _logic_agent()
    _tick(agent)  # rest
    r = _tick(agent, cea=CeAOutput(mode_prior=0.3, urgency_fire=True))
    assert r["last"]["interrupt_onset"] is True
    assert r["summary"]["n_interrupt_onsets"] == 1
    assert r["summary"]["n_producer_onsets"]["cea"] == 1
    # still active: not a new onset, clock not restarted
    r = _tick(agent, cea=CeAOutput(mode_prior=0.3))
    assert r["last"]["interrupt_onset"] is False
    r = _tick(agent, cea=CeAOutput(mode_prior=0.3), switch=True)
    assert r["summary"]["switch_latency_steps"]["interrupt"] == [2]
    assert r["summary"]["switch_latency_coord_ticks"]["interrupt"] == [2]
    assert r["last"]["switch_latency_steps_interrupt"] == 2
    assert r["summary"]["n_mode_switches"] == 1

    # a switch with nothing pending is counted as unattributed
    r = _tick(agent, switch=True)
    assert r["summary"]["n_mode_switches_without_pending_onset"] == 1

    # habenula abort after select -> CONTROL onset, pending until a switch
    _tick(agent)
    agent._veto_readout_note_habenula()
    r = agent.get_veto_readout()
    assert r["last"]["habenula_abort_fired"] is True
    assert r["last"]["control_onset"] is True
    assert r["summary"]["n_control_onsets"] == 1
    assert r["summary"]["pending_onsets"]["control"] is not None

    # episode boundary right-censors the unresolved control onset
    agent.reset()
    r = agent.get_veto_readout()
    assert r["summary"]["n_censored_onsets"]["control"] == 1
    assert r["summary"]["pending_onsets"]["control"] is None
    # counts persist across the boundary; reset_veto_readout clears them
    assert r["summary"]["n_interrupt_onsets"] == 1
    agent.reset_veto_readout()
    assert agent.get_veto_readout()["summary"]["n_steps"] == 0
