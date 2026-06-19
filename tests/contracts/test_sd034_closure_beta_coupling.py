"""SD-034 commitment-closure-control-plane BETA-ENGAGEMENT amend contracts
(2026-06-17, failure_autopsy_V3-EXQ-460e).

Mechanism (a): couple the closure-plane installed commitment
(e3._committed_trajectory is not None) to bistable BetaGate elevation, so the
de-commit latch-occupancy DV is readable on every seed where a closure-plane
commitment forms -- decoupled from the fragile natural running_variance
crossing (result.committed) that fires on only 1/3 seeds on the 603n foraging
substrate (the commit-without-beta dissociation, total_beta_elevated=0).

The new flag use_closure_commit_beta_coupling defaults False -> the bistable
elevate block keys on result.committed alone -> bit-identical OFF.
"""

import torch

from ree_core.heartbeat.beta_gate import BetaGate
from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent


def _build_agent(coupling: bool, bistable: bool = True):
    cfg = REEConfig.from_dims(
        world_obs_dim=250,
        body_obs_dim=12,
        action_dim=4,
        use_lateral_pfc_analog=True,
        use_dacc=True,
        use_closure_operator=True,
        use_closure_commit_beta_coupling=coupling,
    )
    cfg.heartbeat.beta_gate_bistable = bistable
    return REEAgent(cfg)


def _force_dissociation(agent):
    """Wrap e3.select so result.committed is forced False (no natural commit)
    while a closure-plane commitment is active (e3._committed_trajectory set
    to a real candidate trajectory) -- the V3-EXQ-460e seed-42/43 scenario,
    made deterministic.
    """
    real_select = agent.e3.select

    def patched(candidates, *args, **kw):
        res = real_select(candidates, *args, **kw)
        res.committed = False
        if candidates:
            agent.e3._committed_trajectory = candidates[0]
        return res

    agent.e3.select = patched


# ---------------------------------------------------------------------------
# C1 -- BetaGate primitive: closure-coupled-elevation diagnostic counter
# ---------------------------------------------------------------------------
def test_c1_betagate_closure_coupled_counter():
    g = BetaGate()
    assert g.get_state()["sd034_n_closure_coupled_elevations"] == 0
    g.note_closure_coupled_elevation()
    g.note_closure_coupled_elevation()
    assert g.get_state()["sd034_n_closure_coupled_elevations"] == 2
    # the counter is pure diagnostic -- it does not change gate state
    assert not g.is_elevated
    g.reset()
    assert g.get_state()["sd034_n_closure_coupled_elevations"] == 0


# ---------------------------------------------------------------------------
# C2 -- config: default False + from_dims propagation
# ---------------------------------------------------------------------------
def test_c2_config_default_and_from_dims():
    cfg_default = REEConfig.from_dims(
        world_obs_dim=250, body_obs_dim=12, action_dim=4
    )
    assert cfg_default.use_closure_commit_beta_coupling is False
    cfg_on = REEConfig.from_dims(
        world_obs_dim=250, body_obs_dim=12, action_dim=4,
        use_closure_commit_beta_coupling=True,
    )
    assert cfg_on.use_closure_commit_beta_coupling is True


# ---------------------------------------------------------------------------
# C3 -- bit-identical OFF: default vs explicit-False produce the same action
#       stream + beta state over a multi-step run
# ---------------------------------------------------------------------------
def test_c3_bit_identical_off():
    obs_seq = [torch.full((262,), 0.01 * i) for i in range(12)]

    torch.manual_seed(7)
    a_default = _build_agent(coupling=False)
    acts_default = [a_default.act(o).clone() for o in obs_seq]

    torch.manual_seed(7)
    a_explicit = _build_agent(coupling=False)
    acts_explicit = [a_explicit.act(o).clone() for o in obs_seq]

    for ad, ae in zip(acts_default, acts_explicit):
        assert torch.equal(ad, ae), "coupling-OFF action stream must be deterministic"
    # the coupling counter never advances when the flag is off
    assert a_default.beta_gate.get_state()["sd034_n_closure_coupled_elevations"] == 0


# ---------------------------------------------------------------------------
# C4 -- coupling ON: a closure-plane commitment WITHOUT a natural commit
#       (result.committed False) elevates beta, and the diagnostic counter fires
# ---------------------------------------------------------------------------
def test_c4_coupling_on_elevates_on_closure_plane_commit():
    torch.manual_seed(11)
    a = _build_agent(coupling=True)
    a.beta_gate.release()
    _force_dissociation(a)
    for i in range(12):
        a.act(torch.full((262,), 0.01 * i))
        if a.beta_gate.is_elevated:
            break
    assert a.beta_gate.is_elevated, (
        "closure-plane commit must elevate beta under the coupling even when "
        "result.committed is False"
    )
    assert a.beta_gate.get_state()["sd034_n_closure_coupled_elevations"] >= 1


# ---------------------------------------------------------------------------
# C5 -- coupling OFF: the SAME forced dissociation does NOT elevate beta
#       (result.committed False + coupling off -> the natural-only path)
# ---------------------------------------------------------------------------
def test_c5_coupling_off_no_elevation_under_dissociation():
    torch.manual_seed(11)
    a = _build_agent(coupling=False)
    a.beta_gate.release()
    _force_dissociation(a)
    for i in range(12):
        a.act(torch.full((262,), 0.01 * i))
    assert not a.beta_gate.is_elevated, (
        "with the coupling OFF and result.committed forced False, beta must "
        "never elevate from a closure-plane commitment"
    )
    assert a.beta_gate.get_state()["sd034_n_closure_coupled_elevations"] == 0


# ---------------------------------------------------------------------------
# SD-034 460h refractory-INDEPENDENT coupling certifier
# (failure_autopsy_V3-EXQ-460g_2026-06-19). The 460f-prescribed
# sd034_n_closure_coupled_elevations counts only elevations that SURVIVE the
# refractory; the 460g committed-run-scaled de-commit lever pins apply_refractory
# at its cap, making elevate() a no-op for the rest of the post-closure episode,
# so the coupled counter collapsed 36 -> 0 on seed 42 even though the de-commit
# DID act (within-arm occupancy 0.333 -> 0.0). The new intent counter, counted
# BEFORE the elevate/refractory gate, certifies MECH-445 coupling engagement
# regardless of whether the magnitude lever (MECH-446) then blocks the elevate.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# C6 -- BetaGate primitive: commit-intent counter is a pure readout, advances
#       regardless of an active refractory window, get_state + reset
# ---------------------------------------------------------------------------
def test_c6_betagate_commit_intent_counter():
    g = BetaGate()
    assert g.get_state()["sd034_n_closure_commit_intent"] == 0
    g.note_closure_commit_intent()
    g.note_closure_commit_intent()
    assert g.get_state()["sd034_n_closure_commit_intent"] == 2
    # pure diagnostic -- does not change gate state
    assert not g.is_elevated
    # refractory-independent: the intent counter advances while a refractory
    # window (which no-ops elevate()) is active, and does not touch it
    g.apply_refractory(60)
    g.note_closure_commit_intent()
    assert g.get_state()["sd034_n_closure_commit_intent"] == 3
    assert g.refractory_remaining == 60
    g.reset()
    assert g.get_state()["sd034_n_closure_commit_intent"] == 0


# ---------------------------------------------------------------------------
# C7 -- the load-bearing 460h property: while the elevate gate is BLOCKED (the
#       460g latch/refractory suppression, here a held latch + no-op elevate),
#       the per-entry coupled-elevation counter stays suppressed but the
#       refractory-independent intent counter keeps certifying coupling per tick
# ---------------------------------------------------------------------------
def test_c7_commit_intent_certifies_while_elevate_gate_blocked():
    torch.manual_seed(11)
    a = _build_agent(coupling=True)
    _force_dissociation(a)
    # Deterministic stand-in for the 460g refractory/latch block that drove
    # sd034_n_closure_coupled_elevations 36 -> 0: hold the latch elevated and
    # make elevate()/release() no-ops, so the elevate if-block (guarded by
    # `not is_elevated`) is never entered and the gate can never re-fire.
    a.beta_gate._beta_elevated = True
    a.beta_gate.elevate = lambda: None
    a.beta_gate.release = lambda: None
    # run enough acts to span multiple E3 ticks (the bistable elevate block --
    # and the intent count -- runs on the E3 cadence of the multi-rate loop)
    for i in range(16):
        a.act(torch.full((262,), 0.01 * i))
    st = a.beta_gate.get_state()
    assert st["sd034_n_closure_coupled_elevations"] == 0, (
        "the per-entry coupled-elevation counter must stay suppressed while the "
        "elevate gate is blocked (the 460g binding fault)"
    )
    assert st["sd034_n_closure_commit_intent"] > 0, (
        "the refractory-independent intent counter must certify closure->beta "
        "coupling engagement even when the elevate gate is blocked and the "
        "coupled-elevation counter is suppressed (the 460h fix), got %d"
        % st["sd034_n_closure_commit_intent"]
    )


# ---------------------------------------------------------------------------
# C8 -- bit-identical OFF: with the coupling flag off the intent counter never
#       advances (_closure_commit_active stays False)
# ---------------------------------------------------------------------------
def test_c8_commit_intent_zero_when_coupling_off():
    torch.manual_seed(11)
    a = _build_agent(coupling=False)
    _force_dissociation(a)
    for i in range(12):
        a.act(torch.full((262,), 0.01 * i))
    assert a.beta_gate.get_state()["sd034_n_closure_commit_intent"] == 0, (
        "with the coupling flag OFF the closure-plane intent counter must never "
        "advance -> bit-identical to pre-460h"
    )
