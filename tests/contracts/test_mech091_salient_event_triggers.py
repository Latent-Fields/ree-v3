"""MECH-091 contract tests: the two previously-unwired salient-event triggers
for HeartbeatClock.phase_reset() (2026-08-16 wiring).

Background: MECH-091 names three salient events (claims.yaml; clock.py:186)
-- task completion, unexpected harm, commitment-boundary crossing. Before
this change only harm_signal < 0 (agent.py's update_residue()) called
phase_reset(); the other two were unwired for four months despite both
events already firing elsewhere in the substrate. See
evidence/planning/sd006_phase2_generation_brief.md for the governance
routing (2026-08-16, GFLAG-0037) that un-parked this claim to V3.

Contracts (interface-level guarantees, NOT magnitude thresholds):

  C1  commitment-boundary crossing (entry): a genuine not-elevated ->
      elevated transition via the natural readiness-conjunction admission
      path (select_action()) fires phase_reset() exactly once.
  C2  no over-triggering: repeated ticks while ALREADY elevated do not
      re-fire phase_reset() -- this is the load-bearing guard for the
      legacy-mode call site (agent.py, the MECH-342 pressure-reset block)
      that re-invokes beta_gate.elevate() on every committed tick.
  C3  task completion: a hippocampal-completion-driven BetaGate release
      (ARC-028 / MECH-105, completion_signal >= threshold while elevated)
      fires phase_reset().
  C4  no release, no reset: a completion signal below threshold does not
      release the gate and does not fire phase_reset() (negative control
      for C3).
  C5  the pre-existing harm trigger (update_residue(), harm_signal < 0)
      is unaffected by this change -- still fires phase_reset() exactly
      as before.
"""

import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
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


def _scores(margin: float) -> torch.Tensor:
    return torch.tensor([0.0, float(margin)])


class _Stub:
    def __init__(self):
        self.committed = True
        self.scores = _scores(0.5)

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


def _build_readiness_agent() -> REEAgent:
    """Agent configured so the natural readiness-conjunction admission path
    (agent.py select_action(), the MECH-342 pressure-reset call site) admits
    elevation on the first tick -- the same harness shape as
    test_mech_342_commit_maintenance_release.py's _build_agent, reused here
    because it is the established pattern for exercising this exact
    beta_gate.elevate() call site."""
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        use_mech090_readiness_conjunction=True,
        mech090_readiness_floor=0.3,
        commit_readiness_initial=1.0,
        use_sleep_loop=False,
        sws_enabled=False,
        rem_enabled=False,
        use_sleep_aggregation_cluster=False,
    )
    cfg.heartbeat.beta_gate_bistable = True
    cfg.heartbeat.use_commit_readiness_gate = True
    cfg.heartbeat.commit_readiness_floor = 0.05
    agent = REEAgent(cfg)
    agent.reset()
    return agent


def _spy_phase_reset(agent: REEAgent) -> list:
    calls = []
    orig = agent.clock.phase_reset

    def spy():
        calls.append(1)
        return orig()

    agent.clock.phase_reset = spy
    return calls


# ---------------------------------------------------------------- C1 / C2
def test_c1_c2_commitment_entry_fires_once_not_on_every_tick():
    agent = _build_readiness_agent()
    calls = _spy_phase_reset(agent)
    stub = _Stub()
    agent.e3.select = stub.select
    agent.e3._running_variance = 0.0
    cands = _candidates()

    assert not agent.beta_gate.is_elevated
    for t in range(5):
        agent.commit_readiness.notify_outcome(1.0)
        agent.e3.last_scores = _scores(0.5)
        agent.select_action(cands, {"e3_tick": True})
        assert agent.beta_gate.is_elevated
        # C2: fires on the transition tick (0) only, never again while
        # already elevated -- this is the guard against the legacy-mode
        # re-invoked elevate() call site over-triggering every tick.
        assert len(calls) == 1, f"tick {t}: expected exactly 1 call, got {len(calls)}"


# ---------------------------------------------------------------- C3 / C4
def _run_completion_probe(completion_signal: float):
    torch.manual_seed(1)
    env = CausalGridWorldV2(size=8, seed=1)
    _, obs = env.reset()
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=16,
        world_dim=16,
    )
    cfg.heartbeat.beta_gate_bistable = True
    agent = REEAgent(cfg)
    agent.reset()
    agent.beta_gate.elevate()
    agent.hippocampal.compute_completion_signal = lambda trajectories: completion_signal
    calls = _spy_phase_reset(agent)

    latent = agent.sense(obs["body_state"], obs["world_state"])
    ticks = agent.clock.advance()
    e1_prior = (
        agent._e1_tick(latent)
        if ticks.get("e1_tick")
        else torch.zeros(1, cfg.latent.world_dim)
    )
    agent.generate_trajectories(latent, e1_prior, ticks)
    return agent, calls


def test_c3_completion_release_fires_phase_reset():
    agent, calls = _run_completion_probe(0.99)
    assert not agent.beta_gate.is_elevated  # released
    assert len(calls) == 1


def test_c4_low_completion_no_release_no_reset():
    agent, calls = _run_completion_probe(0.0)
    assert agent.beta_gate.is_elevated  # still elevated -- no release
    assert len(calls) == 0


# ---------------------------------------------------------------- C5
def test_c5_harm_trigger_unaffected():
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
    )
    agent = REEAgent(cfg)
    agent.reset()  # agent.reset() sets _current_latent via latent_stack.init_state()
    calls = _spy_phase_reset(agent)
    agent.update_residue(harm_signal=-0.5, owned=True, hypothesis_tag=False)
    assert len(calls) == 1
    # MECH-094: hypothesis/counterfactual evaluation must not fire it.
    agent.update_residue(harm_signal=-0.5, owned=True, hypothesis_tag=True)
    assert len(calls) == 1
