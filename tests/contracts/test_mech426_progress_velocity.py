"""Contract tests: SD-093 / MECH-426 (progress_velocity_maintenance).

A rate-of-progress (velocity) signal -- the temporal derivative of the
on-path goal_proximity() estimate -- modulates E3's commit-threshold
EFFORT/PERSISTENCE pressure (Carver & Scheier 1990 second-order "velocity"
control loop). Per the claim's own explicit caveat, this is modelled as
EFFORT-REGULATION (coasting), never as same-goal reinforcement: it must
never touch goal VALUE / trajectory score (compute_goal_score() /
goal_proximity() stay untouched).

Coverage:
  C1  config defaults: use_progress_velocity_effort_modulation False;
      GoalState.progress_velocity / progress_velocity_effort_modulation are
      0.0 by default.
  C2  flag-off bit-identical: record_progress() is a true no-op (history
      never populated, returns 0.0) regardless of how many times it is
      called or what z_world is passed.
  C3  velocity computed correctly (magnitude, not just sign) from a known
      monotonic-approach sequence and a known monotonic-recede sequence,
      against the documented (newest - oldest) / span rolling-window
      arithmetic.
  C4  effort-modulation direction (coasting model): positive velocity
      (approaching goal) -> NEGATIVE modulation (ease off); negative
      velocity (falling behind) -> POSITIVE modulation (boost effort).
      This is the sign the claim's notes explicitly warn a naive
      implementation gets backwards.
  C5  saturation cap (progress_velocity_effort_max) respected regardless of
      gain magnitude.
  C6  with_injection() (the z_goal_inject > 0 wrapper E3.select() actually
      receives) propagates the velocity state -- the injected view must
      report the SAME progress_velocity_effort_modulation as the source
      GoalState, not a fresh zero.
  C7  reset() clears the rolling window and cached velocity (per-episode
      state must not leak across episodes).
  C8  E3TrajectorySelector.select() wiring: the commit decision
      (committed = variance < effective_threshold) responds correctly to
      the modulation sign -- stalling RAISES effective_threshold (commits
      MORE readily at a fixed variance), coasting LOWERS it (commits LESS
      readily); the flag off leaves effective_threshold at baseline.
  C9  goal_proximity() / goal_distance() (goal VALUE) are byte-identical
      whether or not the flag is on -- the modulator never touches scoring.
"""

from __future__ import annotations

import torch

from ree_core.goal import GoalConfig, GoalState
from ree_core.predictors.e2_fast import Trajectory
from ree_core.predictors.e3_selector import E3Config, E3TrajectorySelector

GOAL_DIM = 4
WORLD_DIM = 6
HIDDEN = 8
HORIZON = 3
ACTION_DIM = 4


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _goal_state(**overrides) -> GoalState:
    cfg = GoalConfig(goal_dim=GOAL_DIM, z_goal_enabled=True, **overrides)
    gs = GoalState(cfg, torch.device("cpu"))
    # Seed z_goal so goal_proximity() has a real attractor to measure against.
    gs.update(torch.ones(1, GOAL_DIM), benefit_exposure=1.0, drive_level=1.0)
    assert gs.is_active()
    return gs


def _candidate(action_class: int = 0) -> Trajectory:
    states = [torch.zeros(1, WORLD_DIM) for _ in range(HORIZON + 1)]
    world_states = [torch.zeros(1, WORLD_DIM) for _ in range(HORIZON + 1)]
    actions = torch.zeros(1, HORIZON, ACTION_DIM)
    actions[:, 0, action_class % ACTION_DIM] = 1.0
    return Trajectory(states=states, actions=actions, world_states=world_states)


def _selector(**e3_kwargs) -> E3TrajectorySelector:
    return E3TrajectorySelector(
        E3Config(world_dim=WORLD_DIM, hidden_dim=HIDDEN, commitment_threshold=0.40, **e3_kwargs)
    )


# --------------------------------------------------------------------------- #
# C1 config defaults
# --------------------------------------------------------------------------- #
def test_c1_config_defaults():
    cfg = GoalConfig()
    assert cfg.use_progress_velocity_effort_modulation is False
    assert cfg.progress_velocity_window == 5
    assert cfg.progress_velocity_effort_gain == 1.0
    assert cfg.progress_velocity_effort_max == 0.3

    gs = _goal_state()  # flag off (default)
    assert gs.progress_velocity == 0.0
    assert gs.progress_velocity_effort_modulation == 0.0


# --------------------------------------------------------------------------- #
# C2 flag-off bit-identical / true no-op
# --------------------------------------------------------------------------- #
def test_c2_flag_off_is_true_noop():
    gs = _goal_state()  # use_progress_velocity_effort_modulation=False
    zg = gs._z_goal.clone()
    for alpha in [0.0, 0.2, 0.5, 0.8, 1.0]:
        v = gs.record_progress(zg * alpha)
        assert v == 0.0
    assert len(gs._progress_history) == 0, "history must stay empty when OFF"
    assert gs.progress_velocity == 0.0
    assert gs.progress_velocity_effort_modulation == 0.0


# --------------------------------------------------------------------------- #
# C3 velocity magnitude from a known sequence
# --------------------------------------------------------------------------- #
def test_c3_velocity_magnitude_approach_sequence():
    gs = _goal_state(use_progress_velocity_effort_modulation=True, progress_velocity_window=4)
    zg = gs._z_goal.clone()
    alphas = [0.0, 0.3, 0.6, 0.9]
    proximities = []
    vel = None
    for alpha in alphas:
        vel = gs.record_progress(zg * alpha)
        proximities.append(float(gs.goal_proximity(zg * alpha).item()))
    # Rolling window arithmetic: velocity = (newest - oldest) / (n - 1)
    expected = (proximities[-1] - proximities[0]) / (len(proximities) - 1)
    assert abs(vel - expected) < 1e-9
    assert vel > 0.0, "approaching the goal must yield positive velocity"
    assert gs.progress_velocity == vel


def test_c3_velocity_magnitude_recede_sequence():
    gs = _goal_state(use_progress_velocity_effort_modulation=True, progress_velocity_window=4)
    zg = gs._z_goal.clone()
    alphas = [0.9, 0.6, 0.3, 0.0]  # moving AWAY from goal
    proximities = []
    vel = None
    for alpha in alphas:
        vel = gs.record_progress(zg * alpha)
        proximities.append(float(gs.goal_proximity(zg * alpha).item()))
    expected = (proximities[-1] - proximities[0]) / (len(proximities) - 1)
    assert abs(vel - expected) < 1e-9
    assert vel < 0.0, "receding from the goal must yield negative velocity"


def test_c3_single_reading_velocity_is_zero():
    gs = _goal_state(use_progress_velocity_effort_modulation=True, progress_velocity_window=5)
    zg = gs._z_goal.clone()
    v = gs.record_progress(zg * 0.5)
    assert v == 0.0, "a single reading has no derivative -- must be 0.0, not NaN/error"


# --------------------------------------------------------------------------- #
# C4 effort-modulation direction (coasting model)
# --------------------------------------------------------------------------- #
def test_c4_positive_velocity_yields_negative_modulation():
    gs = _goal_state(use_progress_velocity_effort_modulation=True, progress_velocity_window=4)
    zg = gs._z_goal.clone()
    for alpha in [0.0, 0.3, 0.6, 0.9]:
        gs.record_progress(zg * alpha)
    assert gs.progress_velocity > 0.0
    mod = gs.progress_velocity_effort_modulation
    assert mod < 0.0, (
        "positive (approaching) velocity must yield NEGATIVE effort "
        "modulation (coasting/ease-off) -- a positive value here would be "
        "the theory-inverting bug the claim's notes warn against"
    )


def test_c4_negative_velocity_yields_positive_modulation():
    gs = _goal_state(use_progress_velocity_effort_modulation=True, progress_velocity_window=4)
    zg = gs._z_goal.clone()
    for alpha in [0.9, 0.6, 0.3, 0.0]:
        gs.record_progress(zg * alpha)
    assert gs.progress_velocity < 0.0
    mod = gs.progress_velocity_effort_modulation
    assert mod > 0.0, (
        "negative (receding/stalled) velocity must yield POSITIVE effort "
        "modulation (boost persistence)"
    )


def test_c4_zero_velocity_yields_zero_modulation():
    gs = _goal_state(use_progress_velocity_effort_modulation=True, progress_velocity_window=4)
    zg = gs._z_goal.clone()
    for _ in range(4):
        gs.record_progress(zg * 0.5)  # constant proximity -> zero derivative
    assert gs.progress_velocity == 0.0
    assert gs.progress_velocity_effort_modulation == 0.0


# --------------------------------------------------------------------------- #
# C5 saturation cap
# --------------------------------------------------------------------------- #
def test_c5_saturation_cap_respected():
    gs = _goal_state(
        use_progress_velocity_effort_modulation=True,
        progress_velocity_window=2,
        progress_velocity_effort_gain=1000.0,
        progress_velocity_effort_max=0.3,
    )
    gs.record_progress(torch.zeros(1, GOAL_DIM))
    gs.record_progress(gs._z_goal.clone())  # large jump -> large |velocity|
    mod = gs.progress_velocity_effort_modulation
    assert abs(mod) <= 0.3 + 1e-9

    gs2 = _goal_state(
        use_progress_velocity_effort_modulation=True,
        progress_velocity_window=2,
        progress_velocity_effort_gain=1000.0,
        progress_velocity_effort_max=0.3,
    )
    gs2.record_progress(gs2._z_goal.clone())
    gs2.record_progress(torch.zeros(1, GOAL_DIM))  # large negative jump
    mod2 = gs2.progress_velocity_effort_modulation
    assert abs(mod2) <= 0.3 + 1e-9


# --------------------------------------------------------------------------- #
# C6 with_injection() propagates velocity state
# --------------------------------------------------------------------------- #
def test_c6_with_injection_propagates_velocity():
    gs = _goal_state(use_progress_velocity_effort_modulation=True, progress_velocity_window=3)
    zg = gs._z_goal.clone()
    for alpha in [0.1, 0.5, 0.9]:
        gs.record_progress(zg * alpha)
    mod_before = gs.progress_velocity_effort_modulation
    assert mod_before != 0.0

    injected = gs.with_injection(0.5)
    assert injected.progress_velocity == gs.progress_velocity
    assert injected.progress_velocity_effort_modulation == mod_before


def test_c6_with_injection_default_zero_when_untouched():
    gs = _goal_state(use_progress_velocity_effort_modulation=True)
    injected = gs.with_injection(0.5)
    assert injected.progress_velocity == 0.0
    assert injected.progress_velocity_effort_modulation == 0.0


# --------------------------------------------------------------------------- #
# C7 reset() clears velocity state
# --------------------------------------------------------------------------- #
def test_c7_reset_clears_velocity_state():
    gs = _goal_state(use_progress_velocity_effort_modulation=True, progress_velocity_window=3)
    zg = gs._z_goal.clone()
    for alpha in [0.1, 0.5, 0.9]:
        gs.record_progress(zg * alpha)
    assert len(gs._progress_history) > 0
    assert gs.progress_velocity != 0.0

    gs.reset()
    assert len(gs._progress_history) == 0
    assert gs.progress_velocity == 0.0
    assert gs.progress_velocity_effort_modulation == 0.0


# --------------------------------------------------------------------------- #
# C8 E3TrajectorySelector.select() wiring
# --------------------------------------------------------------------------- #
def _goal_state_with_velocity(velocity: float) -> GoalState:
    """Directly seed _progress_velocity to test select()'s consumption of the
    modulation in isolation, without needing many record_progress() ticks."""
    gs = _goal_state(use_progress_velocity_effort_modulation=True)
    gs._progress_velocity = velocity
    return gs


def test_c8_no_goal_state_baseline_unaffected():
    sel = _selector()
    sel._running_variance = 0.35  # < 0.40 baseline threshold
    result = sel.select([_candidate(0), _candidate(1)], temperature=1.0, goal_state=None)
    assert result.committed is True


def test_c8_stalling_raises_threshold_commits_more_readily():
    # modulation = -gain*velocity; velocity=-0.2, gain=1.0 -> modulation=+0.2
    # -> effective_threshold = 0.40 * 1.2 = 0.48
    gs = _goal_state_with_velocity(-0.2)
    sel = _selector()
    sel._running_variance = 0.44  # between 0.40 (baseline) and 0.48 (raised)
    result = sel.select([_candidate(0), _candidate(1)], temperature=1.0, goal_state=gs)
    assert result.committed is True, (
        "stalled progress must RAISE effective_threshold enough to commit "
        "at a variance that would NOT commit at the baseline threshold"
    )


def test_c8_coasting_lowers_threshold_commits_less_readily():
    # velocity=+0.2 -> modulation=-0.2 -> effective_threshold = 0.40*0.8=0.32
    gs = _goal_state_with_velocity(0.2)
    sel = _selector()
    sel._running_variance = 0.35  # between 0.32 (lowered) and 0.40 (baseline)
    result = sel.select([_candidate(0), _candidate(1)], temperature=1.0, goal_state=gs)
    assert result.committed is False, (
        "progress ahead of pace (coasting) must LOWER effective_threshold "
        "enough to NOT commit at a variance that WOULD commit at baseline"
    )


def test_c8_flag_off_leaves_threshold_at_baseline():
    gs = _goal_state(use_progress_velocity_effort_modulation=False)
    gs._progress_velocity = -0.2  # would be a large modulation if flag were on
    sel = _selector()
    sel._running_variance = 0.44  # would only commit if threshold were raised
    result = sel.select([_candidate(0), _candidate(1)], temperature=1.0, goal_state=gs)
    assert result.committed is False, "flag off must leave effective_threshold untouched"


def test_c8_inactive_goal_state_is_inert():
    cfg = GoalConfig(goal_dim=GOAL_DIM, z_goal_enabled=True, use_progress_velocity_effort_modulation=True)
    gs = GoalState(cfg, torch.device("cpu"))  # never update()'d -- inactive
    assert not gs.is_active()
    sel = _selector()
    sel._running_variance = 0.44
    result = sel.select([_candidate(0), _candidate(1)], temperature=1.0, goal_state=gs)
    assert result.committed is False, "an inactive goal_state must not modulate the threshold"


# --------------------------------------------------------------------------- #
# C9 goal VALUE (compute_goal_score / goal_proximity) untouched
# --------------------------------------------------------------------------- #
def test_c9_goal_proximity_byte_identical_regardless_of_flag():
    gs_off = _goal_state(use_progress_velocity_effort_modulation=False)
    gs_on = _goal_state(use_progress_velocity_effort_modulation=True)
    # Force identical z_goal (both seeded identically in _goal_state()).
    z_world = torch.rand(1, GOAL_DIM)
    p_off = gs_off.goal_proximity(z_world)
    p_on = gs_on.goal_proximity(z_world)
    assert torch.equal(p_off, p_on)

    # Populate velocity history on the ON instance, then re-check -- the
    # modulator must never feed back into goal_proximity/goal_distance.
    zg = gs_on._z_goal.clone()
    for alpha in [0.1, 0.5, 0.9]:
        gs_on.record_progress(zg * alpha)
    p_on_after = gs_on.goal_proximity(z_world)
    assert torch.equal(p_off, p_on_after)
    assert torch.equal(gs_off.goal_distance(z_world), gs_on.goal_distance(z_world))
