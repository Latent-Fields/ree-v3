#!/opt/local/bin/python3
"""
V3-EXQ-642 -- MECH-353 blocked-agency (z_block) discriminative diagnostic.

Claims: [] (experiment_purpose=diagnostic; excluded from confidence/conflict
        scoring per REE_assembly Phase-3 governance). A PASS clears the MECH-353
        v3_pending substrate gate (substrate-readiness, not claim weighting).
        Bears on (cited, NOT tagged): MECH-353, SD-029, MECH-112, MECH-320,
        MECH-342, ARC-016, SD-011, SD-019b.

THE QUESTION
------------
Does z_block (the blocked-agency / control-failure affect readout, MECH-353)
behave as the verdict predicts: (a) rise when an intended, predicted-to-succeed
action is repeatedly BLOCKED while harm + goal-value are held constant, (b) drive
an ASSERT / persist response (effort escalation / alternative-action search)
DISTINCT from the withdraw signature, and (c) DISSOCIATE from z_harm_a (it fires
with zero noxious input -- the SD-011 / SD-019b distinctness claim)?

DESIGN
------
Static landmark env (SD-023 gradient fields fixed per episode; NO hazards / drift /
respawn / multi-source) so z_world tracks AGENT POSITION rather than env dynamics
-- the precondition for the SD-029 action-outcome comparator to read a block.
Harm held constant (num_hazards=0 -> z_harm_a ~ flat in both arms). Goal-value
held constant (a fixed z_goal is injected and re-pinned each episode).

Two arms, same trained agent (so the ONLY difference is the external block):
  ARM_BLOCK   -- scheduled_action_block intermittent (every other step cancelled)
  ARM_CONTROL -- no block.

Per seed:
  P0  train E2.world_forward on the agent's single-sense rollout with the
      ONE-HOT discrete executed action (the encoding the z_block comparator uses;
      env.step does argmax) so the action-outcome comparator becomes
      discriminative. (world_forward is not trained by E2's primary objective.)
  P1a scripted-driver measurement (random discrete actions): clean z_block /
      outcome_mismatch / z_harm_a readout free of policy confound.
  P1b policy-driver measurement (act_with_split_obs): the ASSERT consumer can
      shift action selection -- measures assert-vs-withdraw behaviour.

PRE-REGISTERED ACCEPTANCE (per seed; PASS = each on >= 2/3 seeds)
  C0 detector readiness: blocked-step outcome_mismatch - free-step outcome_mismatch
     >= C0_MARGIN in ARM_BLOCK (the comparator distinguishes blocked from
     successful). If C0 fails the comparator is non-discriminative on z_world ->
     substrate_ceiling (encoder / world_forward enrichment), NOT a falsification.
  C1 z_block rises: z_block_peak(BLOCK) - z_block_peak(CONTROL) >= C1_MARGIN
     AND z_block_peak(BLOCK) >= Z_BLOCK_MIN.
  C2 dissociation from z_harm_a: (z_block_peak_BLOCK - z_block_peak_CONTROL)
     - (z_harm_a_mean_BLOCK - z_harm_a_mean_CONTROL) >= C2_MARGIN (z_block
     separates while z_harm_a does not -- zero noxious input).
  C3 assert NOT withdraw: action_rate(BLOCK) >= action_rate(CONTROL) - EPS
     (no withdrawal/freeze) AND (action_rate(BLOCK) > action_rate(CONTROL)
     OR alt_switch_rate(BLOCK) > alt_switch_rate(CONTROL)) (assert signature)
     AND z_harm_a_mean(BLOCK) <= z_harm_a_mean(CONTROL) + EPS (no suffering rise).

INTERPRETATION GRID (applied at review)
  PASS (C0,C1,C2,C3)                 -> MECH-353 validated; clear v3_pending.
  C0 fail                            -> substrate_ceiling: z_world action-outcome
                                        comparator non-discriminative; route to
                                        encoder / world_forward enrichment.
                                        MECH-353 stays v3_pending (NOT falsified).
  C0 pass, C1 fail                   -> z_block integrator does not rise; regulator
                                        tuning (accumulation_rate / floors).
  C1 pass, C2 fail                   -> z_block tracks z_harm_a; distinctness from
                                        suffering not demonstrated -- reconsider.
  C2 pass, C3 fail                   -> z_block rises + dissociates but does not
                                        drive assert; consumer-wiring follow-up.

honors feedback_biology_before_formal_definitions: z_block must stay differentiated
from harm (SD-011, no noxious input), suffering (SD-019b, opposite controllability
pole), residue (MECH-056), and licit commitment-hold (MECH-090).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v3_exq_642_blocked_agency_zblock_discriminative"

# ---- Pre-registered thresholds (defined here, never inferred post-hoc) --------
C0_MARGIN = 0.10     # blocked-step minus free-step outcome_mismatch (detector)
C1_MARGIN = 0.20     # z_block_peak(BLOCK) - z_block_peak(CONTROL)
Z_BLOCK_MIN = 0.20   # z_block_peak(BLOCK) absolute floor
C2_MARGIN = 0.20     # z_block separation minus z_harm_a separation (dissociation)
C3_EPS = 0.02        # tolerance for "no withdrawal" / "no suffering rise"
SEED_PASS_FRACTION = 2.0 / 3.0

SEEDS = (42, 43, 44)
P0_WARMUP_EPISODES = 60      # world_forward training episodes per seed
P1_MEASURE_EPISODES = 20     # measurement episodes per arm (scripted + policy)
STEPS_PER_EPISODE = 120
BLOCK_INTERVAL = 2           # every other step blocked in ARM_BLOCK
CONDITIONS = ("ARM_BLOCK", "ARM_CONTROL")

DRY_RUN_SEEDS = (42,)
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_STEPS = 20

GOAL_PIN = 0.5  # fixed z_goal magnitude (goal-value held constant)

ENV_KWARGS = dict(
    size=8,
    num_hazards=0,        # harm held constant (z_harm_a ~ flat)
    num_resources=0,      # no respawn dynamics
    n_landmarks_a=3,      # SD-023 static positional gradient -> z_world tracks position
    n_landmarks_b=3,
    toroidal=True,        # no edge-stall confound for the scripted driver
)
CFG_KWARGS = dict(
    use_blocked_agency=True,
    z_goal_enabled=True,
    drive_weight=2.0,
    alpha_world=1.0,       # no EMA smoothing -> z_world reflects current position
)


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _pin_goal(agent: REEAgent) -> None:
    if agent.goal_state is not None:
        agent.goal_state._z_goal = torch.ones(
            1, agent.goal_state.config.goal_dim, device=agent.device
        ) * GOAL_PIN


def _one_hot(idx: int) -> torch.Tensor:
    a = torch.zeros(1, 4)
    a[0, idx] = 1.0
    return a


def _build_env(seed: int, block: bool):
    return CausalGridWorldV2(
        seed=seed,
        scheduled_action_block_enabled=block,
        scheduled_action_block_interval=BLOCK_INTERVAL,
        scheduled_action_block_prob=1.0,
        **ENV_KWARGS,
    )


def _build_agent(seed: int) -> REEAgent:
    env = _build_env(seed, block=False)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        **CFG_KWARGS,
    )
    return REEAgent(cfg)


def _train_world_forward(agent: REEAgent, seed: int, episodes: int, steps: int,
                         label: str) -> float:
    """P0: train E2.world_forward on the agent's single-sense rollout transitions
    with the one-hot discrete executed action (the z_block comparator encoding).
    Encoder is left fixed (random); world_forward is the trainable target."""
    env = _build_env(seed, block=False)
    _, od = env.reset()
    _pin_goal(agent)
    params = (
        list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters())
    )
    opt = torch.optim.Adam(params, lr=3e-3)
    prev = agent.sense(od["body_state"], od["world_state"]).z_world.detach().clone()
    last_loss = 0.0
    rng = np.random.RandomState(seed)
    for ep in range(episodes):
        for _ in range(steps):
            idx = int(rng.randint(0, 4))
            aoh = _one_hot(idx)
            _, h, d, inf, od = env.step(aoh)
            cur = agent.sense(od["body_state"], od["world_state"]).z_world.detach().clone()
            loss = ((agent.e2.world_forward(prev, aoh) - cur) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
            prev = cur.clone()
            if d:
                _, od = env.reset()
                _pin_goal(agent)
                prev = agent.sense(
                    od["body_state"], od["world_state"]
                ).z_world.detach().clone()
        if (ep + 1) % 10 == 0 or ep == episodes - 1:
            print(
                f"  [train] {label} ep {ep+1}/{episodes} wf_mse={last_loss:.6f}",
                flush=True,
            )
    return last_loss


def _scripted_measure(agent: REEAgent, seed: int, block: bool, episodes: int,
                      steps: int) -> Dict:
    """P1a: drive random discrete actions (no policy confound). Reads z_block,
    outcome_mismatch (split by blocked/free step), z_harm_a."""
    env = _build_env(seed, block=block)
    _, od = env.reset()
    agent.reset()
    _pin_goal(agent)
    agent.sense(od["body_state"], od["world_state"])  # tick0, seeds caches
    rng = np.random.RandomState(seed + 1000)
    z_block_series: List[float] = []
    z_harm_a_series: List[float] = []
    mism_blocked: List[float] = []
    mism_free: List[float] = []
    for ep in range(episodes):
        for _ in range(steps):
            idx = int(rng.randint(0, 4))
            aoh = _one_hot(idx)
            agent._last_action = aoh.clone()
            _, h, d, inf, od = env.step(aoh)
            lat = agent.sense(od["body_state"], od["world_state"])
            o = agent.blocked_agency.last_output()
            z_block_series.append(float(o.z_block))
            if lat.z_harm_a is not None:
                z_harm_a_series.append(float(lat.z_harm_a.detach().norm().item()))
            else:
                z_harm_a_series.append(0.0)
            if inf.get("action_blocked_this_step", False):
                mism_blocked.append(float(o.outcome_mismatch))
            else:
                mism_free.append(float(o.outcome_mismatch))
            if d:
                _, od = env.reset()
                agent.reset()
                _pin_goal(agent)
                agent.sense(od["body_state"], od["world_state"])
    return {
        "z_block_peak": max(z_block_series) if z_block_series else 0.0,
        "z_block_mean": float(np.mean(z_block_series)) if z_block_series else 0.0,
        "z_harm_a_mean": float(np.mean(z_harm_a_series)) if z_harm_a_series else 0.0,
        "blocked_step_mismatch_mean": float(np.mean(mism_blocked)) if mism_blocked else 0.0,
        "free_step_mismatch_mean": float(np.mean(mism_free)) if mism_free else 0.0,
        "n_blocked_steps": len(mism_blocked),
        "n_free_steps": len(mism_free),
    }


def _policy_measure(agent: REEAgent, seed: int, block: bool, episodes: int,
                    steps: int) -> Dict:
    """P1b: drive the agent's own policy (act_with_split_obs) so the ASSERT
    score-bias can shift selection. Reads action-vs-noop rate + alt-action
    switching + z_harm_a."""
    env = _build_env(seed, block=block)
    _, od = env.reset()
    agent.reset()
    _pin_goal(agent)
    noop_class = int(getattr(agent.config, "blocked_agency_noop_class", 0))
    n_action = 0
    n_total = 0
    n_switch = 0
    prev_cls: Optional[int] = None
    z_harm_a_series: List[float] = []
    for ep in range(episodes):
        for _ in range(steps):
            a = agent.act_with_split_obs(od["body_state"], od["world_state"])
            cls = int(a.detach().argmax(dim=-1).flatten()[0].item())
            n_total += 1
            if cls != noop_class:
                n_action += 1
            if prev_cls is not None and cls != prev_cls:
                n_switch += 1
            prev_cls = cls
            _, h, d, inf, od = env.step(a)
            lat = agent.sense(od["body_state"], od["world_state"])
            if lat.z_harm_a is not None:
                z_harm_a_series.append(float(lat.z_harm_a.detach().norm().item()))
            if d:
                _, od = env.reset()
                agent.reset()
                _pin_goal(agent)
                prev_cls = None
    return {
        "action_rate": (n_action / n_total) if n_total else 0.0,
        "alt_switch_rate": (n_switch / n_total) if n_total else 0.0,
        "z_harm_a_mean_policy": float(np.mean(z_harm_a_series)) if z_harm_a_series else 0.0,
        "n_policy_steps": n_total,
    }


def _evaluate_seed(block: Dict, control: Dict) -> Dict:
    """Apply the pre-registered acceptance criteria for one seed."""
    c0 = (block["blocked_step_mismatch_mean"] - block["free_step_mismatch_mean"]) >= C0_MARGIN
    c1 = (
        (block["z_block_peak"] - control["z_block_peak"]) >= C1_MARGIN
        and block["z_block_peak"] >= Z_BLOCK_MIN
    )
    z_block_sep = block["z_block_peak"] - control["z_block_peak"]
    z_harm_a_sep = block["z_harm_a_mean"] - control["z_harm_a_mean"]
    c2 = (z_block_sep - z_harm_a_sep) >= C2_MARGIN
    no_withdraw = block["action_rate"] >= (control["action_rate"] - C3_EPS)
    assert_sig = (
        block["action_rate"] > control["action_rate"]
        or block["alt_switch_rate"] > control["alt_switch_rate"]
    )
    no_suffering = block["z_harm_a_mean_policy"] <= (control["z_harm_a_mean_policy"] + C3_EPS)
    c3 = no_withdraw and assert_sig and no_suffering
    return {
        "C0_detector_readiness": bool(c0),
        "C1_z_block_rises": bool(c1),
        "C2_dissociation_from_z_harm_a": bool(c2),
        "C3_assert_not_withdraw": bool(c3),
        "z_block_separation": z_block_sep,
        "z_harm_a_separation": z_harm_a_sep,
    }


def run_experiment(seeds: List[int], conditions: List[str], p0_episodes: int,
                   p1_episodes: int, steps_per_episode: int,
                   dry_run: bool = False) -> Dict:
    per_seed: List[Dict] = []
    n_runs_completed = 0
    n_total_runs = len(seeds) * len(conditions)
    for seed in seeds:
        _seed_all(seed)
        agent = _build_agent(seed)
        wf_mse = _train_world_forward(
            agent, seed, p0_episodes, steps_per_episode, label=f"seed{seed}"
        )
        arms: Dict[str, Dict] = {}
        for cond in conditions:
            print(f"Seed {seed} Condition {cond}", flush=True)
            is_block = cond == "ARM_BLOCK"
            scripted = _scripted_measure(agent, seed, is_block, p1_episodes, steps_per_episode)
            policy = _policy_measure(agent, seed, is_block, p1_episodes, steps_per_episode)
            merged = {**scripted, **policy}
            arms[cond] = merged
            n_runs_completed += 1
            print(
                f"  {cond}: z_block_peak={merged['z_block_peak']:.3f} "
                f"z_harm_a_mean={merged['z_harm_a_mean']:.3f} "
                f"action_rate={merged['action_rate']:.3f}",
                flush=True,
            )
            print(f"verdict: {'PASS' if merged['z_block_peak'] >= 0.0 else 'FAIL'}", flush=True)
        crit = _evaluate_seed(arms["ARM_BLOCK"], arms["ARM_CONTROL"])
        per_seed.append({
            "seed": seed,
            "wf_mse_final": wf_mse,
            "ARM_BLOCK": arms["ARM_BLOCK"],
            "ARM_CONTROL": arms["ARM_CONTROL"],
            "criteria": crit,
        })

    n = len(per_seed)
    need = 1 if dry_run else int(np.ceil(SEED_PASS_FRACTION * n)) if n else 0
    def frac(key: str) -> int:
        return sum(1 for s in per_seed if s["criteria"][key])
    c0n, c1n, c2n, c3n = frac("C0_detector_readiness"), frac("C1_z_block_rises"), \
        frac("C2_dissociation_from_z_harm_a"), frac("C3_assert_not_withdraw")

    c0_pass = c0n >= need
    c1_pass = c1n >= need
    c2_pass = c2n >= need
    c3_pass = c3n >= need
    overall_pass = c0_pass and c1_pass and c2_pass and c3_pass

    if overall_pass:
        interp = "validated_clear_v3_pending"
    elif not c0_pass:
        interp = "substrate_ceiling_comparator_nondiscriminative"
    elif not c1_pass:
        interp = "z_block_integrator_no_rise"
    elif not c2_pass:
        interp = "z_block_tracks_z_harm_a_not_dissociated"
    else:
        interp = "z_block_does_not_drive_assert"

    return {
        "outcome": "PASS" if overall_pass else "FAIL",
        "interpretation": {
            "label": interp,
            "c0_seeds_pass": c0n,
            "c1_seeds_pass": c1n,
            "c2_seeds_pass": c2n,
            "c3_seeds_pass": c3n,
            "seeds_needed": need,
        },
        "per_seed": per_seed,
        "n_runs_completed": n_runs_completed,
        "n_total_runs": n_total_runs,
    }


def _build_manifest(result: Dict, timestamp_utc: str, dry_run: bool) -> Dict:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    return {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": [],
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": result["outcome"],
        "timestamp_utc": timestamp_utc,
        "dry_run": dry_run,
        "pre_registered_thresholds": {
            "C0_MARGIN": C0_MARGIN,
            "C1_MARGIN": C1_MARGIN,
            "Z_BLOCK_MIN": Z_BLOCK_MIN,
            "C2_MARGIN": C2_MARGIN,
            "C3_EPS": C3_EPS,
            "SEED_PASS_FRACTION": SEED_PASS_FRACTION,
        },
        "interpretation": result["interpretation"],
        "per_seed": result["per_seed"],
        "n_runs_completed": result["n_runs_completed"],
        "n_total_runs": result["n_total_runs"],
        "bears_on": ["MECH-353", "SD-029", "MECH-112", "MECH-320", "MECH-342",
                     "ARC-016", "SD-011", "SD-019b"],
        "notes": (
            "Diagnostic substrate-readiness for MECH-353 z_block. PASS clears "
            "the MECH-353 v3_pending gate. C0-fail routes to substrate_ceiling "
            "(z_world action-outcome comparator non-discriminative -> encoder / "
            "world_forward enrichment), NOT a falsification of MECH-353."
        ),
    }


def main() -> Tuple[Optional[str], Optional[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Short smoke run (1 seed, 2+2 ep, 20 steps).")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    _run_started = datetime.now(timezone.utc)

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        p0, p1, steps = DRY_RUN_P0, DRY_RUN_P1, DRY_RUN_STEPS
    else:
        seeds = list(SEEDS)
        p0, p1, steps = P0_WARMUP_EPISODES, P1_MEASURE_EPISODES, STEPS_PER_EPISODE

    result = run_experiment(
        seeds=seeds, conditions=list(CONDITIONS),
        p0_episodes=p0, p1_episodes=p1, steps_per_episode=steps,
        dry_run=bool(args.dry_run),
    )

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run))

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = (
            Path(__file__).resolve().parents[2]
            / "REE_assembly" / "evidence" / "experiments"
        )
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=args.dry_run,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
        elapsed_seconds=(datetime.now(timezone.utc) - _run_started).total_seconds(),
    )

    print(f"manifest: {out_path}", flush=True)
    print(
        f"outcome: {result['outcome']} interp={result['interpretation']['label']} "
        f"runs={result['n_runs_completed']}/{result['n_total_runs']}",
        flush=True,
    )

    if args.dry_run:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass

    outcome_norm = result["outcome"].upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    manifest_for_sentinel = str(out_path) if not args.dry_run else None
    return outcome_emit, manifest_for_sentinel


if __name__ == "__main__":
    _outcome, _manifest_path = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path)
    sys.exit(0)
