#!/opt/local/bin/python3
"""V3-EXQ-516 -- MECH-302 suffering-derivative comparator agent-loop integration.

Claim: MECH-302 (relief.completion_event_reuses_goal_achievement_pipeline)
Substrate: SD-050 (SufferingDerivativeComparator, ree_core/comparator/)
Status: candidate (v3_pending). Substrate IMPLEMENTED 2026-05-04.

Why this experiment exists
--------------------------
MECH-302 asserts that a relief-completion event fires the same downstream
pipeline as goal-achievement (MECH-057a beta-gate-drop, MECH-094 VALENCE_LIKING
write). SD-050 provides the substrate: a non-trainable rolling-window descent
detector (SufferingDerivativeComparator) on z_harm_a.norm(). The detector fires
when a sustained drop from oldest to newest buffer entry exceeds drop_threshold,
provided the initial norm exceeds min_initial_norm (prevents false fires on an
already-quiet stream).

This is a SUBSTRATE READINESS DIAGNOSTIC, not a governance evidence experiment.
It verifies four integration properties:

ARM_0 (comparator OFF -- backward compat):
  - C0a: agent.suffering_comparator is None with default config.
  - C0b: _relief_completion_event never becomes True over N sense() calls.

ARM_1 (comparator ON, hazard env, event fires):
  - C1a: agent.suffering_comparator is not None.
  - C1b: When comparator buffer is pre-filled to trigger state, sense() with
    a low-norm obs_harm_a causes _relief_completion_event = True.
  - C1c: After select_action() consumes the event, _relief_completion_event
    is cleared (= False).

ARM_2 (comparator ON + valence_liking, valence write fires):
  - C2a: Same as C1b (event fires).
  - C2b: select_action() calls residue_field.update_valence() when
    valence_liking_enabled=True and event is set.

ARM_3 (comparator ON, flat signal -- no hazard input):
  - C3: No relief events fire over N steps when obs_harm_a = zeros throughout
    (z_harm_a stays below min_initial_norm).

PASS = C0a AND C0b AND C1a AND C1b AND C1c AND C2a AND C2b AND C3.

PASS = SD-050 substrate is integrated correctly and ready for behavioural
validation. Successor: discriminative-pair experiment using trained agent
with/without comparator to measure chronic-avoidance failure mode vs clean
episode completion.

FAIL on C0 -> backward-compat broken; check config default wiring.
FAIL on C1b -> sense() comparator tick path not firing; check agent.py wiring.
FAIL on C1c -> event not cleared in select_action(); check fire handler.
FAIL on C2b -> valence write path not reached; check select_action() condition
  (valence_liking_enabled gate + _current_latent.z_world non-None check).
FAIL on C3 -> false fires on flat signal; min_initial_norm guard broken.

experiment_purpose = "diagnostic" (substrate readiness, not governance evidence).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_516_mech302_suffering_derivative_integration.py
  /opt/local/bin/python3 experiments/v3_exq_516_mech302_suffering_derivative_integration.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_516_mech302_suffering_derivative_integration"
CLAIM_IDS = ["MECH-302"]
EXPERIMENT_PURPOSE = "diagnostic"

# Diagnostic comparator params.
# Very sensitive to fire reliably regardless of random encoder weight magnitude.
# This is correct for a substrate-integration diagnostic -- we are testing the
# wiring, not calibrating thresholds. Production defaults (drop=0.10, min=0.05)
# are the scientifically meaningful values; diagnostic uses drop=0.001 to
# confirm the pipeline fires when triggered.
DIAG_WINDOW_LENGTH = 3
DIAG_DROP_THRESHOLD = 0.001
DIAG_MIN_INITIAL_NORM = 0.0001

# Pre-fill norm used to put the buffer in trigger state.
# Any value well above DIAG_MIN_INITIAL_NORM suffices.
PREFILL_NORM = 0.50

N_FLAT_STEPS = 100  # ARM_0 (OFF) and ARM_3 (flat) runs

# Env dims (CausalGridWorldV2 defaults)
BODY_OBS_DIM = 12
WORLD_OBS_DIM = 250
HARM_OBS_A_DIM = 50
ACTION_DIM = 5


def make_config(
    comparator_on: bool,
    valence_liking: bool = False,
) -> REEConfig:
    return REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        use_affective_harm_stream=True,
        harm_obs_a_dim=HARM_OBS_A_DIM,
        use_suffering_derivative_comparator=comparator_on,
        suffering_window_length=DIAG_WINDOW_LENGTH,
        suffering_drop_threshold=DIAG_DROP_THRESHOLD,
        suffering_min_initial_norm=DIAG_MIN_INITIAL_NORM,
        valence_liking_enabled=valence_liking,
        relief_completion_weight=1.0,
    )


def make_env(num_hazards: int = 3, seed: int = 42) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        num_hazards=num_hazards,
        num_resources=5,
        seed=seed,
    )


def zero_obs_dict() -> Dict:
    """Minimal obs dict with all zeros -- used for ARM_0 flat-signal tests."""
    return {
        "body_state": torch.zeros(1, BODY_OBS_DIM),
        "world_state": torch.zeros(1, WORLD_OBS_DIM),
        "harm_obs_a": torch.zeros(1, HARM_OBS_A_DIM),
    }


def sense_step(agent: REEAgent, obs_dict: Dict) -> bool:
    """Call sense() and return the _relief_completion_event flag value."""
    body = obs_dict["body_state"]
    world = obs_dict["world_state"]
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    sense_kwargs = {"obs_body": body, "obs_world": world}
    obs_harm_a = obs_dict.get("harm_obs_a")
    if obs_harm_a is not None:
        if obs_harm_a.dim() == 1:
            obs_harm_a = obs_harm_a.unsqueeze(0)
        sense_kwargs["obs_harm_a"] = obs_harm_a
    with torch.no_grad():
        agent.sense(**sense_kwargs)
    return bool(agent._relief_completion_event)


def full_step(
    agent: REEAgent, env: CausalGridWorldV2, obs_dict: Dict
) -> tuple:
    """
    One full agent+env step (sense -> clock -> e1 -> trajectories -> select_action -> env.step).
    Returns (next_obs_dict, event_fired_in_sense, valence_writes_in_select_action, done).
    """
    body = obs_dict["body_state"]
    world = obs_dict["world_state"]
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    sense_kwargs = {"obs_body": body, "obs_world": world}
    obs_harm_a = obs_dict.get("harm_obs_a")
    if obs_harm_a is not None:
        if obs_harm_a.dim() == 1:
            obs_harm_a = obs_harm_a.unsqueeze(0)
        sense_kwargs["obs_harm_a"] = obs_harm_a

    # Count valence writes via monkey-patch (restored after select_action).
    valence_writes = [0]
    orig_uv = None
    if (
        hasattr(agent, "residue_field")
        and hasattr(agent.residue_field, "update_valence")
    ):
        orig_uv = agent.residue_field.update_valence

        def _patched_uv(*args, **kwargs):
            valence_writes[0] += 1
            return orig_uv(*args, **kwargs)

        agent.residue_field.update_valence = _patched_uv

    with torch.no_grad():
        _latent = agent.sense(**sense_kwargs)
        event_fired = bool(agent._relief_completion_event)
        ticks = agent.clock.advance()
        world_dim = agent.config.latent.world_dim
        e1_prior = (
            agent._e1_tick(_latent)
            if ticks.get("e1_tick", True)
            else torch.zeros(1, world_dim)
        )
        candidates = agent.generate_trajectories(_latent, e1_prior, ticks)
        action = agent.select_action(candidates, ticks)

    if orig_uv is not None:
        agent.residue_field.update_valence = orig_uv

    action_idx = int(action.argmax(dim=-1).item())
    _flat, harm_signal, done, _info, next_obs = env.step(action_idx)
    agent.update_residue(float(harm_signal))

    return next_obs, event_fired, valence_writes[0], bool(done)


def test_arm_0(verbose: bool) -> Dict:
    """ARM_0: comparator OFF -- backward compat."""
    results: Dict = {"arm": "ARM_0_off", "checks": {}}
    cfg = make_config(comparator_on=False)
    agent = REEAgent(cfg)

    # C0a: comparator attribute is None.
    c0a = agent.suffering_comparator is None
    results["checks"]["C0a_comparator_is_none"] = bool(c0a)

    # C0b: _relief_completion_event never fires over N sense() calls with zeros.
    agent.reset()
    n_events = 0
    obs = zero_obs_dict()
    for _ in range(N_FLAT_STEPS):
        fired = sense_step(agent, obs)
        if fired:
            n_events += 1
    c0b = n_events == 0
    results["checks"]["C0b_no_events_over_flat_steps"] = bool(c0b)
    results["n_events"] = n_events

    results["PASS"] = c0a and c0b
    if verbose:
        print(f"  ARM_0: C0a={c0a}, C0b={c0b} (n_events={n_events})")
    return results


def test_arm_1(verbose: bool) -> Dict:
    """ARM_1: comparator ON, event fires via pre-triggered buffer."""
    results: Dict = {"arm": "ARM_1_on_event_fires", "checks": {}}
    cfg = make_config(comparator_on=True)
    agent = REEAgent(cfg)
    env = make_env(num_hazards=3, seed=42)
    agent.reset()
    _flat, obs_dict = env.reset()

    # C1a: comparator attribute is not None.
    c1a = agent.suffering_comparator is not None
    results["checks"]["C1a_comparator_not_none"] = bool(c1a)

    if not c1a:
        results["PASS"] = False
        return results

    # C1b: pre-fill buffer to (window_length-1) high-norm values, then
    # call full_step with obs_harm_a=zeros.  The final tick in sense()
    # produces z_harm_a.norm() near 0.  Buffer completes:
    # [0.5, 0.5, ~0] -> initial=0.5 > min_initial_norm; drop=0.5 > threshold.
    # Override obs_dict harm_obs_a with zeros so the encoder sees a flat input.
    agent.suffering_comparator._norm_buffer = [PREFILL_NORM] * (DIAG_WINDOW_LENGTH - 1)
    obs_dict_zero_harm = dict(obs_dict)
    obs_dict_zero_harm["harm_obs_a"] = torch.zeros(HARM_OBS_A_DIM)

    next_obs, event_fired, _vw, done = full_step(agent, env, obs_dict_zero_harm)
    c1b = event_fired
    results["checks"]["C1b_event_fired_after_prefill"] = bool(c1b)

    # C1c: event cleared by select_action() (full_step already consumed it).
    c1c = not bool(agent._relief_completion_event)
    results["checks"]["C1c_event_cleared_by_select_action"] = bool(c1c)
    results["event_fired"] = bool(event_fired)

    results["PASS"] = c1a and c1b and c1c
    if verbose:
        print(f"  ARM_1: C1a={c1a}, C1b={c1b}, C1c={c1c}")
    return results


def test_arm_2(verbose: bool) -> Dict:
    """ARM_2: comparator ON + valence_liking, valence write fires on event."""
    results: Dict = {"arm": "ARM_2_valence_write", "checks": {}}
    cfg = make_config(comparator_on=True, valence_liking=True)
    agent = REEAgent(cfg)
    env = make_env(num_hazards=3, seed=42)
    agent.reset()
    _flat, obs_dict = env.reset()

    # C2a: event fires (same pre-fill trigger as ARM_1).
    agent.suffering_comparator._norm_buffer = [PREFILL_NORM] * (DIAG_WINDOW_LENGTH - 1)
    obs_dict_zero_harm = dict(obs_dict)
    obs_dict_zero_harm["harm_obs_a"] = torch.zeros(HARM_OBS_A_DIM)

    next_obs, event_fired, valence_writes, done = full_step(
        agent, env, obs_dict_zero_harm
    )
    c2a = event_fired
    results["checks"]["C2a_event_fired"] = bool(c2a)

    # C2b: update_valence was called during the same select_action() tick.
    c2b = valence_writes > 0
    results["checks"]["C2b_valence_write_on_event"] = bool(c2b)
    results["valence_writes"] = valence_writes
    results["event_fired"] = bool(event_fired)

    results["PASS"] = c2a and c2b
    if verbose:
        print(f"  ARM_2: C2a={c2a}, C2b={c2b} (valence_writes={valence_writes})")
    return results


def test_arm_3(verbose: bool) -> Dict:
    """ARM_3: comparator ON, flat zero signal -- no false fires."""
    results: Dict = {"arm": "ARM_3_flat_signal_no_fire", "checks": {}}
    cfg = make_config(comparator_on=True)
    agent = REEAgent(cfg)

    # Drive the agent with all-zeros obs_harm_a for N_FLAT_STEPS.
    # z_harm_a will be near-zero throughout; the buffer will fill with
    # near-zero norms; initial_norm < min_initial_norm -> no event.
    agent.reset()
    n_events = 0
    obs = zero_obs_dict()
    for _ in range(N_FLAT_STEPS):
        fired = sense_step(agent, obs)
        if fired:
            n_events += 1

    c3 = n_events == 0
    results["checks"]["C3_no_events_on_flat_signal"] = bool(c3)
    results["n_events"] = n_events

    results["PASS"] = c3
    if verbose:
        print(f"  ARM_3: C3={c3} (n_events={n_events})")
    return results


def run_all(dry_run: bool = False, verbose: bool = True) -> Dict:
    t0 = time.time()
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "_v3"

    if verbose:
        print(f"EXQ-516 agent-loop integration diagnostic (run_id={run_id})")
        if dry_run:
            print("  [dry-run mode]")

    arm0 = test_arm_0(verbose)
    arm1 = test_arm_1(verbose)
    arm2 = test_arm_2(verbose)
    arm3 = test_arm_3(verbose)

    all_checks: Dict[str, bool] = {}
    for arm in (arm0, arm1, arm2, arm3):
        all_checks.update(arm["checks"])

    overall_pass = all(all_checks.values())

    elapsed = time.time() - t0
    if verbose:
        status = "PASS" if overall_pass else "FAIL"
        print(f"  Overall: {status} ({elapsed:.1f}s)")
        for k, v in sorted(all_checks.items()):
            tick = "PASS" if v else "FAIL"
            print(f"    {tick}  {k}")

    result = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "status": "PASS" if overall_pass else "FAIL",
        "pass": overall_pass,
        "elapsed_s": round(elapsed, 2),
        "all_checks": all_checks,
        "arms": {
            "ARM_0_off": arm0,
            "ARM_1_on_event_fires": arm1,
            "ARM_2_valence_write": arm2,
            "ARM_3_flat_signal_no_fire": arm3,
        },
        "config": {
            "diag_window_length": DIAG_WINDOW_LENGTH,
            "diag_drop_threshold": DIAG_DROP_THRESHOLD,
            "diag_min_initial_norm": DIAG_MIN_INITIAL_NORM,
            "prefill_norm": PREFILL_NORM,
            "n_flat_steps": N_FLAT_STEPS,
        },
        "dry_run": dry_run,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    result = run_all(dry_run=args.dry_run, verbose=not args.quiet)

    if not args.dry_run:
        out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{EXPERIMENT_TYPE}_{result['run_id']}.json"
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"Result written to {out_path}")


if __name__ == "__main__":
    main()
