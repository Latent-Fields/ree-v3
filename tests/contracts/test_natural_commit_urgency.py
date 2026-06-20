"""Contract tests: commit/release-DURATION lever -- graded natural-commit-
occupancy release (rung-6 of f_dominance_conversion_ceiling, the duration face
PARALLEL to the selection-face MECH-448).

Contracts (interface-level guarantees, NOT magnitude thresholds):

  C1  config defaults: use_natural_commit_urgency_release False; from_dims
      surfaces the knobs with no-op defaults; master OFF -> agent.
      natural_commit_urgency is None.
  C2  graded urgency, gap-scaled rate is LOAD-BEARING (BG-3 D1): an F-decisive
      entry (gap_norm=1) accrues release-urgency faster -> releases at fewer
      ticks than a near-tie entry (gap_norm=0); gap_entry_sensitivity=0 reduces
      to a flat fixed-rate timeout independent of gap (the contrasted control).
  C3  action-extent mode (Jin): fires when the executed action sequence
      completes, independent of urgency.
  C4  unarmed no-op: tick() returns False on a run not armed by a natural
      commit entry (a purely closure-coupled run is left to SD-034).
  C5  MECH-094: simulation_mode=True is a no-op (no accumulate, returns False).
  C6  config validation: urgency_rate<=0, release_bound<=0, cap<bound,
      gap_entry_sensitivity<0, onset_ticks<0 raise ValueError.
  C7  agent release wiring: a prolonged natural commit is RELEASED by the lever
      (beta drops, _committed_step_idx reset, e3._committed_trajectory cleared)
      and its occupancy is BOUNDED, where the OFF agent holds the latch.
  C8  agent bit-identical OFF: an ON-but-inert lever (both sub-modes off)
      produces an identical action stream to OFF over a real loop.
  C9  agent arm-site: the natural-elevation path arms the lever (is_armed True
      after a committed select_action tick) -- the note_commit_entry wiring.
  C10 safety: the lever only RELEASES (returns to fresh deliberation); it never
      elevates the latch and never forces an action.
"""

from __future__ import annotations

import pytest
import torch

from ree_core.agent import REEAgent
from ree_core.policy import (
    NaturalCommitUrgencyRelease,
    NaturalCommitUrgencyReleaseConfig,
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


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
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


def _scores(gap_norm: float) -> torch.Tensor:
    # REE lower-is-better. Build a 3-candidate score vector whose normalised
    # top-2 gap == gap_norm: winner 0.0, runner-up = gap_norm, worst = 1.0
    # (range = 1.0 so gap/range = gap_norm).
    return torch.tensor([0.0, float(gap_norm), 1.0])


class _Stub:
    """E3.select stub: committed=True with a chosen entry decisiveness."""

    def __init__(self, gap_norm: float = 1.0):
        self.committed = True
        self.scores = _scores(gap_norm)

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


def _reg(**overrides) -> NaturalCommitUrgencyRelease:
    cfg = NaturalCommitUrgencyReleaseConfig(
        use_natural_commit_urgency_release=True, **overrides
    )
    return NaturalCommitUrgencyRelease(cfg)


def _build_agent(**lever_kwargs) -> REEAgent:
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
        **lever_kwargs,
    )
    cfg.heartbeat.beta_gate_bistable = True
    agent = REEAgent(cfg)
    agent.reset()
    return agent


def _drive(agent, ticks: int):
    """Drive a natural commit through real select_action ticks; return the list
    of beta-elevated states (per tick, post-call)."""
    stub = _Stub(gap_norm=1.0)
    agent.e3.select = stub.select
    agent.e3._running_variance = 0.0
    cands = _candidates()
    elevated = []
    for _ in range(ticks):
        agent.e3.last_scores = stub.scores.clone()
        agent.e3._committed_trajectory = cands[0]
        agent.select_action(cands, {"e3_tick": True})
        elevated.append(bool(agent.beta_gate.is_elevated))
    return elevated


# ----------------------------------------------------------------------
# C1 config defaults + master OFF
# ----------------------------------------------------------------------
def test_c1_config_defaults_and_master_off():
    c = NaturalCommitUrgencyReleaseConfig()
    assert c.use_natural_commit_urgency_release is False
    assert c.urgency_mode is True
    assert c.action_extent_mode is True
    assert c.urgency_rate == 0.01
    assert c.release_bound == 1.0
    assert c.urgency_cap == 1.5
    assert c.gap_entry_sensitivity == 1.0
    assert c.onset_ticks == 0
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
    )
    assert cfg.use_natural_commit_urgency_release is False
    assert cfg.natural_commit_gap_entry_sensitivity == 1.0
    agent = _build_agent()  # master off (default)
    assert agent.natural_commit_urgency is None


# ----------------------------------------------------------------------
# C2 graded urgency, gap-scaled rate load-bearing
# ----------------------------------------------------------------------
def test_c2_gap_scaled_rate_load_bearing():
    # decisive entry (gap_norm=1) -> scale 2 -> 0.5/tick -> fires sooner.
    r = _reg(urgency_rate=0.25, gap_entry_sensitivity=1.0)
    r.note_commit_entry(1.0)
    decisive_fire = next(
        t for t in range(1, 20)
        if r.tick(committed_run_length=t, action_sequence_complete=False)
    )
    # near-tie entry (gap_norm=0) -> scale 1 -> 0.25/tick -> fires later.
    r2 = _reg(urgency_rate=0.25, gap_entry_sensitivity=1.0)
    r2.note_commit_entry(0.0)
    neartie_fire = next(
        t for t in range(1, 20)
        if r2.tick(committed_run_length=t, action_sequence_complete=False)
    )
    assert decisive_fire < neartie_fire, (
        f"gap-scaling not load-bearing: decisive {decisive_fire} "
        f">= near-tie {neartie_fire}"
    )
    # flat (gap_entry_sensitivity=0): release tick independent of gap.
    rflat_a = _reg(urgency_rate=0.25, gap_entry_sensitivity=0.0)
    rflat_a.note_commit_entry(1.0)
    flat_a = next(
        t for t in range(1, 20)
        if rflat_a.tick(committed_run_length=t, action_sequence_complete=False)
    )
    rflat_b = _reg(urgency_rate=0.25, gap_entry_sensitivity=0.0)
    rflat_b.note_commit_entry(0.0)
    flat_b = next(
        t for t in range(1, 20)
        if rflat_b.tick(committed_run_length=t, action_sequence_complete=False)
    )
    assert flat_a == flat_b, "flat (sensitivity=0) should be gap-independent"


# ----------------------------------------------------------------------
# C3 action-extent mode
# ----------------------------------------------------------------------
def test_c3_action_extent_mode():
    r = _reg(urgency_mode=False, action_extent_mode=True)
    r.note_commit_entry(0.5)
    # urgency off -> no urgency release even at high run length
    assert r.tick(committed_run_length=100, action_sequence_complete=False) is False
    # sequence complete -> fires
    assert r.tick(committed_run_length=3, action_sequence_complete=True) is True
    st = r.get_state()
    assert st["ncur_n_action_extent_releases"] == 1
    assert st["ncur_n_urgency_releases"] == 0


# ----------------------------------------------------------------------
# C4 unarmed no-op
# ----------------------------------------------------------------------
def test_c4_unarmed_no_op():
    r = _reg(urgency_rate=1.0, action_extent_mode=True)
    # never armed -> no-op even at sequence complete / high run length
    assert r.tick(committed_run_length=999, action_sequence_complete=True) is False
    assert r.get_state()["ncur_n_releases_total"] == 0


# ----------------------------------------------------------------------
# C5 MECH-094 simulation no-op
# ----------------------------------------------------------------------
def test_c5_simulation_no_op():
    r = _reg(urgency_rate=1.0)
    r.note_commit_entry(1.0)
    assert r.tick(
        committed_run_length=999,
        action_sequence_complete=True,
        simulation_mode=True,
    ) is False
    assert r.get_state()["ncur_n_simulation_skips"] == 1
    assert r.get_state()["ncur_n_releases_total"] == 0


# ----------------------------------------------------------------------
# C6 config validation
# ----------------------------------------------------------------------
def test_c6_config_validation():
    with pytest.raises(ValueError):
        NaturalCommitUrgencyRelease(
            NaturalCommitUrgencyReleaseConfig(urgency_rate=0.0)
        )
    with pytest.raises(ValueError):
        NaturalCommitUrgencyRelease(
            NaturalCommitUrgencyReleaseConfig(release_bound=0.0)
        )
    with pytest.raises(ValueError):
        NaturalCommitUrgencyRelease(
            NaturalCommitUrgencyReleaseConfig(release_bound=1.0, urgency_cap=0.5)
        )
    with pytest.raises(ValueError):
        NaturalCommitUrgencyRelease(
            NaturalCommitUrgencyReleaseConfig(gap_entry_sensitivity=-1.0)
        )
    with pytest.raises(ValueError):
        NaturalCommitUrgencyRelease(
            NaturalCommitUrgencyReleaseConfig(onset_ticks=-1)
        )


# ----------------------------------------------------------------------
# C7 agent release wiring + bounded occupancy
# ----------------------------------------------------------------------
def test_c7_agent_releases_prolonged_natural_commit():
    # OFF: nothing releases the natural commit -> the latch occupancy grows
    # monotonically across the run (the 460h ~2400-2600-step monopoly in
    # miniature). The always-committed stub holds the bistable latch.
    off = _build_agent()  # master off
    elevated_off = _drive(off, ticks=12)
    assert all(elevated_off[2:]), "OFF agent should hold the latch elevated"
    off_occupancy = off.beta_gate.committed_run_length
    assert off_occupancy >= 10, (
        f"OFF natural-commit occupancy should grow; got {off_occupancy}"
    )

    # ON (urgency mode, fast rate, action-extent off): the lever RELEASES the
    # prolonged natural commit. The always-committed stub re-commits next tick
    # (so post-tick is_elevated stays True), but each held run is now BOUNDED:
    # the lever fires and resets the occupancy instead of letting it grow.
    on = _build_agent(
        use_natural_commit_urgency_release=True,
        natural_commit_release_action_extent_mode=False,
        natural_commit_urgency_rate=0.5,
        natural_commit_urgency_release_bound=1.0,
        natural_commit_gap_entry_sensitivity=1.0,
    )
    _drive(on, ticks=12)
    st = on.natural_commit_urgency.get_state()
    assert st["ncur_n_urgency_releases"] >= 1, "ON lever should fire"
    # Latch occupancy shortens: each fire releases at a small run length, far
    # below the OFF sustained occupancy.
    assert st["ncur_last_occupancy_at_release"] < off_occupancy
    assert on.beta_gate.committed_run_length < off_occupancy


# ----------------------------------------------------------------------
# C8 agent bit-identical OFF (ON-inert == OFF)
# ----------------------------------------------------------------------
def test_c8_agent_bit_identical_off_vs_inert():
    off = _build_agent()
    inert = _build_agent(
        use_natural_commit_urgency_release=True,
        natural_commit_release_urgency_mode=False,
        natural_commit_release_action_extent_mode=False,
    )
    e_off = _drive(off, ticks=10)
    e_inert = _drive(inert, ticks=10)
    assert e_off == e_inert, (
        "ON-but-inert lever must not perturb the latch occupancy"
    )
    # The inert lever arms but never fires.
    assert inert.natural_commit_urgency.get_state()["ncur_n_releases_total"] == 0


# ----------------------------------------------------------------------
# C9 agent arm-site (note_commit_entry wiring)
# ----------------------------------------------------------------------
def test_c9_agent_arm_site():
    on = _build_agent(
        use_natural_commit_urgency_release=True,
        natural_commit_urgency_rate=1e-6,  # tiny -> won't fire during the probe
        natural_commit_release_action_extent_mode=False,
    )
    stub = _Stub(gap_norm=0.7)
    on.e3.select = stub.select
    on.e3._running_variance = 0.0
    cands = _candidates()
    on.e3.last_scores = stub.scores.clone()
    on.e3._committed_trajectory = cands[0]
    on.select_action(cands, {"e3_tick": True})
    assert on.natural_commit_urgency.is_armed is True
    # gap_norm captured from the entry scores (0.7 within tolerance).
    assert abs(
        on.natural_commit_urgency._gap_norm_at_entry - 0.7
    ) < 1e-5


# ----------------------------------------------------------------------
# C10 safety: lever only releases, never elevates / forces an action
# ----------------------------------------------------------------------
def test_c10_safety_release_only():
    r = _reg(urgency_rate=0.5, gap_entry_sensitivity=1.0)
    # The regulator returns only a bool (release-or-not); it has no path to
    # elevate the latch or emit an action. Confirm a fire is a clean release
    # signal and the accumulator resets (no carried-over forced state).
    r.note_commit_entry(1.0)
    fired = False
    for t in range(1, 10):
        if r.tick(committed_run_length=t, action_sequence_complete=False):
            fired = True
            break
    assert fired
    assert r.get_urgency() == 0.0  # reset on fire; no residual forcing state
    assert r.is_armed is False  # disarmed on fire -> back to fresh deliberation
