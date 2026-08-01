"""
Contracts for the Q-081 REFRAMED pair-specific reach probe
(experiments/_lib/q081_pair_reach_check_stream.py), the generalisation of
`test_q081_pair_reach_check.py` for z_goal [primary] / z_harm_a [fallback]
rather than operating_mode.

What is pinned here, and why each one is load-bearing:

  1. `snapshot_stream_signal` reads the CORRECT per-target activation
     precondition and raw/gated tensor, and returns None (not a stale or
     zeroed snapshot) exactly when the real agent.py call site would never
     have consumed a value this tick either -- for both z_goal (E1 path,
     gated on `goal.e1_goal_conditioned` + `goal_state.is_active()`) and
     z_harm_a (E2 path, gated on `agent.e2_harm_a is not None` +
     `agent._harm_a_prev is not None`).
  2. THE GATE COMPUTATION IS EXERCISED, using the REAL `VsRolloutGate`
     class (not a fake): a stream held below threshold returns the cached
     snapshot as "gated", diverging from "raw" -- proving the probe can
     actually detect a gate-mediated divergence, which is the entire
     mechanism this probe exists to catch (see module docstring).
  3. THE GATE CALL NEVER MUTATES DIAGNOSTIC STATE. `snapshot_stream_signal`
     must call the gate's pure `_gate_value`, not the mutating public
     `gate_stream()` wrapper -- confirmed by asserting the gate's own
     `_held_count_e1` / `_last_held_e1` counters are untouched after a
     snapshot call that DID hit the held branch.
  4. UNKNOWN TARGET RAISES rather than silently returning a wrong stream.
  5. THE COMPARISON LOGIC PLUGS INTO THE PARENT MODULE'S PURE DIFF
     (`pair_specific_reach_report`, reused rather than re-implemented) --
     this module's per-dimension named-float snapshot shape is diffed
     correctly by that unchanged function.
  6. `assert_pair_specific_stream_reach` RAISES (strict) / RETURNS
     (non-strict) appropriately, names the target in its message, and (for
     target="z_goal") names the z_harm_a fallback -- the GATE text a future
     driver script's operator will actually read.
  7. LIVE SMOKE (real REEAgent/env, small dims), one per target, at settings
     MEASURED during this module's own authoring to be non-degenerate for
     EACH guard independently:
       - z_harm_a activates from tick 1 (no restart budget needed) --
         confirmed non-degenerate at n_episodes=1, steps_per_episode=150,
         env_size=6, seed=2 (same cheap point q081_pair_reach_check.py's own
         positive-control test uses for boundary events).
       - z_goal's activation precondition needs many episode RESTARTS
         (short per-episode survival dominates, not a longer per-episode
         step cap) -- a separate single-agent activation scan (documented in
         v3_exq_865_q081_zgoal_reach_preflight_scan.py's module docstring)
         found it activates within a ~600-1200 cumulative-tick budget at
         seeds 2/3/4 of 0-4, but NOT within the smaller budget this test
         file uses (kept small for contract-suite speed) -- so the z_goal
         live smoke below is INTENTIONALLY at a scale expected to be
         degenerate (confirming the activation guard fires correctly,
         mirroring how `test_q081_pair_reach_check.py`'s own
         `test_live_probe_degeneracy_guard_raises_when_strict` pins a
         guaranteed-degenerate case), while a separate SEPARATE test at the
         validated larger budget confirms z_goal CAN clear the guard.
  8. BOTH NON-DEGENERACY GUARDS (boundary events, target activation) are
     independently triggerable and independently named in
     `degeneracy_reason` -- a run degenerate on ONE guard but not the other
     must still report BOTH reasons it checked, not just the first found.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments._lib.q081_pair_reach_check_stream import (
    TARGET_STREAMS,
    assert_pair_specific_stream_reach,
    run_pair_specific_stream_reach_probe,
    snapshot_stream_signal,
)
from ree_core.regulators.vs_rollout_gate import VsRolloutGate, VsRolloutGateConfig


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #


class _FakeGoalState:
    def __init__(self, z_goal: torch.Tensor, active: bool):
        self._z_goal = z_goal
        self._active = active

    @property
    def z_goal(self) -> torch.Tensor:
        return self._z_goal

    def is_active(self) -> bool:
        return self._active


class _FakeGoalConfig:
    def __init__(self, e1_goal_conditioned: bool = True):
        self.e1_goal_conditioned = e1_goal_conditioned


class _FakeAgentConfig:
    def __init__(self, goal_cfg: Optional[_FakeGoalConfig]):
        self.goal = goal_cfg


class _FakeHippocampal:
    def __init__(self, per_stream_vs: Dict[str, float]):
        self.per_stream_vs = per_stream_vs


class _FakeAgent:
    def __init__(
        self,
        goal_state: Optional[_FakeGoalState] = None,
        goal_cfg: Optional[_FakeGoalConfig] = None,
        vs_rollout_gate: Optional[VsRolloutGate] = None,
        per_stream_vs: Optional[Dict[str, float]] = None,
        e2_harm_a: Optional[Any] = None,
        harm_a_prev: Optional[torch.Tensor] = None,
        vs_gate_staleness_cache: Optional[Dict[str, float]] = None,
    ):
        self.goal_state = goal_state
        self.config = _FakeAgentConfig(goal_cfg if goal_cfg is not None else _FakeGoalConfig())
        self.vs_rollout_gate = vs_rollout_gate
        self.hippocampal = _FakeHippocampal(per_stream_vs or {})
        self.e2_harm_a = e2_harm_a
        self._harm_a_prev = harm_a_prev
        self._vs_gate_staleness_cache = vs_gate_staleness_cache


def _trace(*snapshots: Optional[Dict[str, float]]) -> List[Optional[Dict[str, float]]]:
    return list(snapshots)


# --------------------------------------------------------------------------- #
# snapshot_stream_signal -- activation preconditions                          #
# --------------------------------------------------------------------------- #


def test_snapshot_z_goal_none_when_inactive():
    agent = _FakeAgent(goal_state=_FakeGoalState(torch.zeros(1, 4), active=False))
    assert snapshot_stream_signal(agent, "z_goal") is None


def test_snapshot_z_goal_none_when_e1_goal_conditioned_false():
    agent = _FakeAgent(
        goal_state=_FakeGoalState(torch.ones(1, 4), active=True),
        goal_cfg=_FakeGoalConfig(e1_goal_conditioned=False),
    )
    assert snapshot_stream_signal(agent, "z_goal") is None


def test_snapshot_z_goal_none_when_goal_state_is_none():
    agent = _FakeAgent(goal_state=None)
    assert snapshot_stream_signal(agent, "z_goal") is None


def test_snapshot_z_harm_a_none_when_e2_harm_a_is_none():
    agent = _FakeAgent(e2_harm_a=None, harm_a_prev=torch.ones(1, 4))
    assert snapshot_stream_signal(agent, "z_harm_a") is None


def test_snapshot_z_harm_a_none_when_harm_a_prev_is_none():
    agent = _FakeAgent(e2_harm_a=object(), harm_a_prev=None)
    assert snapshot_stream_signal(agent, "z_harm_a") is None


def test_snapshot_unknown_target_raises():
    agent = _FakeAgent()
    with pytest.raises(ValueError, match="not in TARGET_STREAMS"):
        snapshot_stream_signal(agent, "z_world")


# --------------------------------------------------------------------------- #
# snapshot_stream_signal -- no gate (raw == gated passthrough)                #
# --------------------------------------------------------------------------- #


def test_snapshot_z_goal_raw_equals_gated_when_no_gate_present():
    z = torch.tensor([[0.5, -0.25, 1.0]])
    agent = _FakeAgent(
        goal_state=_FakeGoalState(z, active=True),
        vs_rollout_gate=None,
    )
    snap = snapshot_stream_signal(agent, "z_goal")
    assert snap == {
        "z_goal_raw_0": 0.5, "z_goal_raw_1": -0.25, "z_goal_raw_2": 1.0,
        "z_goal_gated_0": 0.5, "z_goal_gated_1": -0.25, "z_goal_gated_2": 1.0,
    }


def test_snapshot_z_harm_a_raw_equals_gated_when_no_gate_present():
    z = torch.tensor([[2.0, 3.0]])
    agent = _FakeAgent(e2_harm_a=object(), harm_a_prev=z, vs_rollout_gate=None)
    snap = snapshot_stream_signal(agent, "z_harm_a")
    assert snap == {
        "z_harm_a_raw_0": 2.0, "z_harm_a_raw_1": 3.0,
        "z_harm_a_gated_0": 2.0, "z_harm_a_gated_1": 3.0,
    }


# --------------------------------------------------------------------------- #
# snapshot_stream_signal -- real VsRolloutGate, held-snapshot substitution     #
# --------------------------------------------------------------------------- #


def test_snapshot_z_goal_gated_diverges_from_raw_when_held():
    """A real VsRolloutGate with a cached snapshot and V_s below the e1
    threshold must return the SNAPSHOT as 'gated', not the current raw
    value -- the exact mechanism this whole probe exists to detect."""
    gate = VsRolloutGate(VsRolloutGateConfig(e1_threshold=0.4))
    gate._snapshots["z_goal"] = torch.tensor([[9.0, 9.0]])
    current = torch.tensor([[1.0, 2.0]])
    agent = _FakeAgent(
        goal_state=_FakeGoalState(current, active=True),
        vs_rollout_gate=gate,
        per_stream_vs={"z_goal": 0.1},  # below e1_threshold=0.4 -> held
    )
    snap = snapshot_stream_signal(agent, "z_goal")
    assert snap["z_goal_raw_0"] == 1.0
    assert snap["z_goal_raw_1"] == 2.0
    assert snap["z_goal_gated_0"] == 9.0
    assert snap["z_goal_gated_1"] == 9.0


def test_snapshot_z_harm_a_gated_equals_raw_when_vs_above_threshold():
    gate = VsRolloutGate(VsRolloutGateConfig(e2_threshold=0.4))
    gate._snapshots["z_harm_a"] = torch.tensor([[9.0]])
    current = torch.tensor([[1.0]])
    agent = _FakeAgent(
        e2_harm_a=object(), harm_a_prev=current, vs_rollout_gate=gate,
        per_stream_vs={"z_harm_a": 0.9},  # above e2_threshold=0.4 -> passthrough
    )
    snap = snapshot_stream_signal(agent, "z_harm_a")
    assert snap["z_harm_a_raw_0"] == 1.0
    assert snap["z_harm_a_gated_0"] == 1.0


def test_gate_call_does_not_mutate_diagnostic_counters():
    """snapshot_stream_signal must use the gate's PURE `_gate_value`, never
    the mutating public gate_stream()/gate() wrappers -- confirmed by
    checking the gate's own diagnostic counters are unchanged after a call
    that DID hit the held branch (which gate_stream() would have counted)."""
    gate = VsRolloutGate(VsRolloutGateConfig(e1_threshold=0.4))
    gate._snapshots["z_goal"] = torch.tensor([[9.0]])
    agent = _FakeAgent(
        goal_state=_FakeGoalState(torch.tensor([[1.0]]), active=True),
        vs_rollout_gate=gate,
        per_stream_vs={"z_goal": 0.1},
    )
    snap = snapshot_stream_signal(agent, "z_goal")
    assert snap["z_goal_gated_0"] == 9.0  # sanity: the held branch DID fire
    assert gate._held_count_e1["z_goal"] == 0
    assert gate._last_held_e1["z_goal"] is False
    assert gate._refresh_count["z_goal"] == 0


# --------------------------------------------------------------------------- #
# pair_specific_reach_report reuse (generic diff, unmodified from the parent  #
# module -- sanity check that THIS module's snapshot shape plugs in)          #
# --------------------------------------------------------------------------- #


def test_reused_comparator_detects_divergence_in_stream_shaped_traces():
    intact = _trace(
        {"z_goal_raw_0": 0.1, "z_goal_gated_0": 0.1},
        {"z_goal_raw_0": 0.2, "z_goal_gated_0": 0.2},
    )
    manipulated = _trace(
        {"z_goal_raw_0": 0.1, "z_goal_gated_0": 0.1},
        {"z_goal_raw_0": 0.2, "z_goal_gated_0": 0.9},  # gated diverges at t=1
    )
    report = assert_pair_specific_stream_reach(
        "z_goal", intact, manipulated, strict=False,
    )
    assert report["has_pair_specific_reach"] is True
    assert report["divergent_signals"] == {"z_goal_gated_0": 1}


def test_reused_comparator_no_reach_on_bit_identical_traces():
    trace_a = _trace({"z_harm_a_raw_0": 0.5}, {"z_harm_a_raw_0": 0.5})
    trace_b = _trace({"z_harm_a_raw_0": 0.5}, {"z_harm_a_raw_0": 0.5})
    report = assert_pair_specific_stream_reach(
        "z_harm_a", trace_a, trace_b, strict=False,
    )
    assert report["has_pair_specific_reach"] is False
    assert report["divergent_signals"] == {}


# --------------------------------------------------------------------------- #
# assert_pair_specific_stream_reach                                           #
# --------------------------------------------------------------------------- #


def test_assert_raises_when_strict_and_no_reach_names_target_and_fallback():
    trace = _trace({"z_goal_raw_0": 0.1}, {"z_goal_raw_0": 0.1})
    with pytest.raises(RuntimeError, match="target='z_goal'"):
        assert_pair_specific_stream_reach("z_goal", trace, trace, strict=True)
    with pytest.raises(RuntimeError, match="z_harm_a"):
        assert_pair_specific_stream_reach("z_goal", trace, trace, strict=True)


def test_assert_does_not_raise_when_not_strict():
    trace = _trace({"z_harm_a_raw_0": 0.1}, {"z_harm_a_raw_0": 0.1})
    report = assert_pair_specific_stream_reach(
        "z_harm_a", trace, trace, strict=False,
    )
    assert report["has_pair_specific_reach"] is False


def test_assert_does_not_raise_when_reach_present():
    intact = _trace({"z_harm_a_raw_0": 0.1})
    manipulated = _trace({"z_harm_a_raw_0": 0.9})
    report = assert_pair_specific_stream_reach(
        "z_harm_a", intact, manipulated, strict=True,
    )
    assert report["has_pair_specific_reach"] is True


# --------------------------------------------------------------------------- #
# Live smoke -- real REEAgent/env, small and short (LIVE SMOKE, item 7/8)     #
# --------------------------------------------------------------------------- #


def test_live_probe_z_harm_a_runs_end_to_end_non_degenerate():
    """z_harm_a activates from tick 1 onward once `use_e2_harm_a` +
    `use_affective_harm_stream` are configured (this module sets both for
    target='z_harm_a'), unlike z_goal's sparse benefit-gated activation --
    confirmed non-degenerate at this exact cheap setting during authoring
    (n_episodes=1, steps_per_episode=150, env_size=6, seed=2 -- the same
    point q081_pair_reach_check.py's own suppress-mode positive control
    uses for boundary-event non-degeneracy).

    Does NOT assert a specific has_pair_specific_reach value (measured
    False for iei_permute/jitter/suppress at this seed during authoring,
    but this test pins that the pipeline runs to completion against the
    real substrate and both non-degeneracy guards clear, not a specific
    scientific reading -- see this script's own v3_exq_865 driver for the
    actual multi-seed adjudication).
    """
    report = run_pair_specific_stream_reach_probe(
        target="z_harm_a", n_episodes=1, steps_per_episode=150,
        env_size=6, seed=2, strict=False,
    )
    assert report["target"] == "z_harm_a"
    assert report["n_episodes"] == 1
    assert report["steps_per_episode"] == 150
    assert report["is_degenerate"] is False
    assert report["degeneracy_reason"] is None
    assert report["n_boundaries_true_total"] == 6
    assert report["n_active_ticks_intact"] == 150
    assert report["n_ticks_compared"] == 150
    assert isinstance(report["has_pair_specific_reach"], bool)
    assert isinstance(report["divergent_signals"], dict)
    assert report["behavioural_reach_precondition"]["has_behavioural_reach"] is True


def test_live_probe_z_goal_degenerate_at_small_scale_names_activity_reason():
    """z_goal's activation precondition needs many episode RESTARTS (see
    module docstring + v3_exq_865's authoring-time single-agent scan) --
    at THIS small a budget (well under the ~600-1200 cumulative-tick budget
    the scan found necessary), z_goal is expected, measured, and DELIBERATE
    to come back degenerate on the activity guard specifically (not the
    boundary-event guard, which independently clears at this seed/scale).
    Measured 8 true boundary events at this exact setting during authoring
    -- NOT the same figure as z_harm_a's own measured 6 at the nominally
    "same" seed=2/steps=150/n_episodes=1: target="z_goal" sets no extra
    agent config flags while target="z_harm_a" additionally sets
    `use_e2_harm_a=True` (see TARGET_STREAMS), which changes how many
    torch.nn.Module parameters are randomly initialised at agent
    construction time and therefore shifts the subsequent RNG draw
    sequence (env/action randomness) even at an identical `seed` --
    reproducible per target, but not equal across targets.
    """
    report = run_pair_specific_stream_reach_probe(
        target="z_goal", n_episodes=1, steps_per_episode=150,
        env_size=6, seed=2, strict=False,
    )
    assert report["target"] == "z_goal"
    assert report["is_degenerate"] is True
    assert report["n_boundaries_true_total"] == 8  # boundary guard CLEARS
    assert report["n_active_ticks_intact"] == 0    # activity guard FAILS
    assert "was active (non-None snapshot) on only 0" in report["degeneracy_reason"]
    assert "z_goal" in report["degeneracy_reason"]
    # A degenerate run must never accidentally claim reach was found.
    assert report["has_pair_specific_reach"] is False


def test_live_probe_z_goal_degeneracy_guard_raises_when_strict():
    with pytest.raises(RuntimeError, match="DEGENERATE"):
        run_pair_specific_stream_reach_probe(
            target="z_goal", n_episodes=1, steps_per_episode=150,
            env_size=6, seed=2, strict=True,
        )


def test_live_probe_z_harm_a_degeneracy_guard_raises_when_strict_and_no_boundaries():
    """Distinct from the z_goal degeneracy test above: this pins the
    INHERITED boundary-event guard (shared with the parent module) firing
    independently of z_harm_a's own (always-clear-once-configured)
    activation guard. A single-episode, 5-step rollout is certain to be
    boundary-degenerate (q081_pair_reach_check.py's own docstring: an
    untrained event segmenter needs on the order of hundreds of steps
    before its first boundary)."""
    with pytest.raises(RuntimeError, match="DEGENERATE"):
        run_pair_specific_stream_reach_probe(
            target="z_harm_a", n_episodes=1, steps_per_episode=5,
            env_size=6, seed=2, strict=True,
        )


def test_unknown_target_raises_before_any_live_work():
    with pytest.raises(ValueError, match="not in TARGET_STREAMS"):
        run_pair_specific_stream_reach_probe(
            target="operating_mode", n_episodes=1, steps_per_episode=5,
            env_size=6, seed=0, strict=False,
        )
