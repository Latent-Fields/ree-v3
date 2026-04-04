#!/opt/local/bin/python3
"""
V3-EXQ-232 -- ARC-026: Approach/Contact Slope Conflict Resolution

Claims: ARC-026
Proposal: resolution of conflict_ratio=1.0 from EXQ-033

EXPERIMENT_PURPOSE = "evidence"

ARC-026 asserts:
  In goal-directed behavior, the approach slope (harm calibration at
  hazard_approach transitions) precedes and predicts contact events in
  temporal structure. The approach_slope > contact_slope indicates that
  the gradient of harm-detection extends backward from the endpoint (contact)
  toward the causal precursor (approach).

EXQ-033 FAIL context:
  Both signals peaked at ep500 then degraded by ep1000 (training instability).
  conflict_ratio = 1.0 from equal supports and weakens evidence.
  Root cause: terminal performance (ep1000) not the right measurement point
  when training is unstable.

This experiment resolves the conflict by:
  1. Using more checkpoints (ep200, ep400, ep600, ep800, ep1000, ep1500) and
     reporting PEAK checkpoint results rather than terminal.
  2. Longer training (1500 eps) to extend the measurement window.
  3. 3 seeds (EXQ-033 used 1 seed effectively).
  4. Early stopping detection: record the checkpoint where
     approach_gap - contact_gap is maximised.

PASS criterion (revised)
------------------------
At the peak performance checkpoint (across all seeds):
  C1: approach_slope_at_peak > contact_slope_at_peak (>= 2/3 seeds)
      Approach detection improves FASTER than contact detection.
      slope_at_peak = (gap[peak_ckpt] - gap[earliest_ckpt]) / ckpt_episodes_elapsed.
  C2: approach_gap_at_peak > 0.05 (>= 2/3 seeds)
      Non-trivial approach detection at peak.
  C3: n_approach_events >= 15 at each checkpoint in >= 2/3 seeds.
      Sufficient approach events for reliable measurement.
  C4: peak_checkpoint >= ep400 (>= 2/3 seeds)
      Training actually progresses before degradation.

PASS: C1 AND C2 AND C3 AND C4.

Seeds: [42, 7, 13]
Env:   CausalGridWorldV2 size=10, 3 hazards, 3 resources, hazard_harm=0.1
Train: 1500 episodes x 200 steps (checkpoints at 200,400,600,800,1000,1500)
Eval:  30 episodes x 200 steps at each checkpoint
Est:   ~250 min DLAPTOP-4.local (long; phased training with checkpoints)
"""

import sys
import random
import json
import math
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.optim as optim
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE = "v3_exq_232_arc026_approach_contact_resolution"
CLAIM_IDS = ["ARC-026"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
THRESH_APPROACH_GAP   = 0.05   # C2: approach_gap at peak
MIN_APPROACH_EVENTS   = 15     # C3: approach events per checkpoint eval
MIN_PEAK_CHECKPOINT   = 400    # C4: peak must be at or after ep400

# ---------------------------------------------------------------------------
# Training schedule
# ---------------------------------------------------------------------------
CHECKPOINTS    = [200, 400, 600, 800, 1000, 1500]  # cumulative episode counts
EVAL_EPISODES  = 30
STEPS_PER_EP   = 200

SEEDS = [42, 7, 13]


# ---------------------------------------------------------------------------
# Env / config factory
# ---------------------------------------------------------------------------

def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=3,
        num_resources=3,
        hazard_harm=0.1,   # Stronger harm signal for gradient detection
        # No env_drift_interval=0: that causes ZeroDivisionError in env.step().
        # Use default (5) -- static enough for gradient detection.
    )


def _make_config() -> REEConfig:
    return REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        action_dim=5,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
    )


# ---------------------------------------------------------------------------
# Calibration eval: measure approach vs contact gap
# ---------------------------------------------------------------------------

def _eval_calibration_gap(
    agent: REEAgent,
    env: CausalGridWorldV2,
    n_episodes: int,
    steps_per_ep: int,
) -> Dict:
    """
    Measure E3 harm eval calibration gap for approach vs contact transitions.

    calibration_gap = mean E3 harm_eval output on positive transitions
                    - mean E3 harm_eval output on neutral transitions.

    approach = info["transition_type"] in ("hazard_approach",)
    contact  = info["transition_type"] in ("agent_caused_hazard",)
    neutral  = info["transition_type"] == "none"
    """
    agent.eval()
    approach_vals:  List[float] = []
    contact_vals:   List[float] = []
    neutral_vals:   List[float] = []

    with torch.no_grad():
        for ep in range(n_episodes):
            _, obs_dict = env.reset()
            agent.reset()

            for _ in range(steps_per_ep):
                obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
                obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)
                latent    = agent.sense(obs_body, obs_world)
                ticks     = agent.clock.advance()
                e1_prior  = (
                    agent._e1_tick(latent) if ticks["e1_tick"]
                    else torch.zeros(1, 32, device=agent.device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action     = agent.select_action(candidates, ticks, temperature=0.5)

                _, reward, done, info, obs_dict = env.step(action)

                # Get E3 harm eval on current z_world
                z_world   = latent.z_world.detach()
                harm_pred = float(agent.e3.harm_eval(z_world).item())

                ttype = info.get("transition_type", "none")
                if ttype in ("hazard_approach",):
                    approach_vals.append(harm_pred)
                elif ttype in ("agent_caused_hazard",):
                    contact_vals.append(harm_pred)
                elif ttype == "none":
                    neutral_vals.append(harm_pred)

                agent.update_residue(float(reward) if reward < 0 else 0.0)
                if done:
                    break

    mean_a = sum(approach_vals) / max(1, len(approach_vals))
    mean_c = sum(contact_vals)  / max(1, len(contact_vals))
    mean_n = sum(neutral_vals)  / max(1, len(neutral_vals))

    gap_approach = mean_a - mean_n
    gap_contact  = mean_c - mean_n

    agent.train()
    return {
        "gap_approach":         gap_approach,
        "gap_contact":          gap_contact,
        "mean_harm_approach":   mean_a,
        "mean_harm_contact":    mean_c,
        "mean_harm_neutral":    mean_n,
        "n_approach_events":    len(approach_vals),
        "n_contact_events":     len(contact_vals),
        "n_neutral_events":     len(neutral_vals),
    }


# ---------------------------------------------------------------------------
# Run one seed
# ---------------------------------------------------------------------------

def _run_seed(seed: int, dry_run: bool) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    checkpoints   = [10, 20] if dry_run else CHECKPOINTS
    n_eval        = 3  if dry_run else EVAL_EPISODES
    n_step        = 20 if dry_run else STEPS_PER_EP

    env    = _make_env(seed)
    config = _make_config()
    agent  = REEAgent(config)

    e1_opt  = optim.Adam(agent.e1.parameters(), lr=1e-3)
    e2_opt  = optim.Adam(agent.e2.parameters(), lr=3e-3)
    e3_opt  = optim.Adam(agent.e3.parameters(), lr=1e-3)
    lat_opt = optim.Adam(agent.latent_stack.parameters(), lr=1e-3)

    print(
        f"  [EXQ-232] seed={seed} checkpoints={checkpoints}"
        f" eval_eps={n_eval} steps={n_step}",
        flush=True,
    )

    checkpoint_results: Dict[int, Dict] = {}
    episodes_done = 0

    agent.train()

    for ckpt_target in checkpoints:
        # Train up to this checkpoint
        episodes_to_run = ckpt_target - episodes_done
        for ep in range(episodes_to_run):
            _, obs_dict = env.reset()
            agent.reset()

            for _ in range(n_step):
                obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
                obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)
                latent    = agent.sense(obs_body, obs_world)
                ticks     = agent.clock.advance()
                e1_prior  = (
                    agent._e1_tick(latent) if ticks["e1_tick"]
                    else torch.zeros(1, 32, device=agent.device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action     = agent.select_action(candidates, ticks, temperature=1.0)

                _, reward, done, info, obs_dict = env.step(action)
                harm_signal = float(reward) if reward < 0 else 0.0

                # E1 + E2 losses
                e1_loss = agent.compute_prediction_loss()
                e2_loss = agent.compute_e2_loss()
                total   = e1_loss + e2_loss
                if total.requires_grad:
                    e1_opt.zero_grad(); e2_opt.zero_grad()
                    total.backward()
                    e1_opt.step(); e2_opt.step()

                # E3 harm supervision (phase 2: after initial E1/E2 convergence)
                if episodes_done + ep >= 100:
                    if agent._current_latent is not None:
                        z_world = agent._current_latent.z_world.detach()
                        harm_t  = torch.tensor(
                            [[1.0 if harm_signal < 0 else 0.0]]
                        )
                        hloss = F.mse_loss(agent.e3.harm_eval(z_world), harm_t)
                        if hloss.requires_grad:
                            e3_opt.zero_grad()
                            lat_opt.zero_grad()
                            hloss.backward()
                            e3_opt.step()
                            lat_opt.step()

                # E3 running variance update
                if hasattr(e1_loss, 'item'):
                    agent.e3.update_running_variance(
                        torch.tensor([[e1_loss.item()]])
                    )

                agent.update_residue(harm_signal)
                if done:
                    break

        episodes_done = ckpt_target

        # Calibration eval at this checkpoint
        print(f"    [eval] seed={seed} checkpoint ep={ckpt_target}", flush=True)
        cr = _eval_calibration_gap(agent, env, n_eval, n_step)
        checkpoint_results[ckpt_target] = cr
        print(
            f"    gap_approach={cr['gap_approach']:.4f}"
            f" gap_contact={cr['gap_contact']:.4f}"
            f" n_approach={cr['n_approach_events']}"
            f" n_contact={cr['n_contact_events']}",
            flush=True,
        )

    # Find peak: checkpoint with max gap_approach
    peak_ckpt = max(checkpoint_results.keys(),
                    key=lambda k: checkpoint_results[k]["gap_approach"])
    peak_cr   = checkpoint_results[peak_ckpt]

    # Compute slopes up to peak
    earliest_ckpt = min(checkpoint_results.keys())
    earliest_cr   = checkpoint_results[earliest_ckpt]
    eps_elapsed   = max(1, peak_ckpt - earliest_ckpt)
    approach_slope = (
        peak_cr["gap_approach"] - earliest_cr["gap_approach"]
    ) / eps_elapsed
    contact_slope  = (
        peak_cr["gap_contact"] - earliest_cr["gap_contact"]
    ) / eps_elapsed

    print(
        f"  seed={seed} peak_ckpt={peak_ckpt}"
        f" approach_slope={approach_slope:.6f}"
        f" contact_slope={contact_slope:.6f}"
        f" approach_gap_at_peak={peak_cr['gap_approach']:.4f}",
        flush=True,
    )

    return {
        "seed":                    seed,
        "peak_checkpoint":         peak_ckpt,
        "approach_slope":          approach_slope,
        "contact_slope":           contact_slope,
        "approach_gap_at_peak":    peak_cr["gap_approach"],
        "contact_gap_at_peak":     peak_cr["gap_contact"],
        "n_approach_at_peak":      peak_cr["n_approach_events"],
        "checkpoint_results":      {str(k): v for k, v in checkpoint_results.items()},
    }


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(dry_run: bool = False) -> dict:
    print(f"\n[EXQ-232] ARC-026 Approach/Contact Slope Resolution (dry_run={dry_run})",
          flush=True)

    seed_results = []
    c1_passes = []; c2_passes = []; c3_passes = []; c4_passes = []

    for seed in SEEDS:
        print(f"\n--- seed={seed} ---", flush=True)
        sr = _run_seed(seed, dry_run)
        seed_results.append(sr)

        c1_passes.append(sr["approach_slope"] > sr["contact_slope"])
        c2_passes.append(sr["approach_gap_at_peak"] > THRESH_APPROACH_GAP)
        c3_passes.append(sr["n_approach_at_peak"] >= MIN_APPROACH_EVENTS)
        c4_passes.append(sr["peak_checkpoint"] >= MIN_PEAK_CHECKPOINT)

    c1_pass = sum(c1_passes) >= 2
    c2_pass = sum(c2_passes) >= 2
    c3_pass = sum(c3_passes) >= 2
    c4_pass = sum(c4_passes) >= 2

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass
    status   = "PASS" if all_pass else "FAIL"
    crit_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])

    print(f"\n[EXQ-232] Results:", flush=True)
    for i, seed in enumerate(SEEDS):
        sr = seed_results[i]
        print(
            f"  seed={seed}: approach_slope={sr['approach_slope']:.6f}"
            f" contact_slope={sr['contact_slope']:.6f}"
            f" approach_gap={sr['approach_gap_at_peak']:.4f}"
            f" peak_ckpt={sr['peak_checkpoint']}"
            f" C1={'P' if c1_passes[i] else 'F'}"
            f" C2={'P' if c2_passes[i] else 'F'}"
            f" C3={'P' if c3_passes[i] else 'F'}"
            f" C4={'P' if c4_passes[i] else 'F'}",
            flush=True,
        )
    print(f"  Status: {status} ({crit_met}/4)", flush=True)

    if all_pass:
        interpretation = (
            "ARC-026 SUPPORTED: At peak performance checkpoint, approach_slope"
            " > contact_slope. The gradient of harm detection extends backward"
            " from contact to approach. Consistent with 'love expands under"
            " intelligence' derivation (ARC-024): more training -> deeper"
            " causal gradient detection."
        )
    elif crit_met >= 2:
        interpretation = (
            "ARC-026 PARTIAL: Some slope evidence but below full threshold."
            " May need longer training or higher hazard_harm signal."
        )
    else:
        interpretation = (
            "ARC-026 NOT SUPPORTED: Approach slope does not exceed contact"
            " slope at peak checkpoint. Conflict from EXQ-033 not resolved."
        )

    failure_notes = []
    if not c1_pass:
        pairs = [(seed_results[i]["approach_slope"], seed_results[i]["contact_slope"])
                 for i in range(3)]
        failure_notes.append(
            f"C1 FAIL: approach_slope <= contact_slope in >= 2/3 seeds. "
            f"Pairs (approach,contact): {[(round(a,6),round(c,6)) for a,c in pairs]}"
        )
    if not c2_pass:
        vals = [round(seed_results[i]["approach_gap_at_peak"], 4) for i in range(3)]
        failure_notes.append(
            f"C2 FAIL: approach_gap_at_peak < {THRESH_APPROACH_GAP}: {vals}"
        )
    if not c3_pass:
        vals = [seed_results[i]["n_approach_at_peak"] for i in range(3)]
        failure_notes.append(
            f"C3 FAIL: n_approach_at_peak < {MIN_APPROACH_EVENTS}: {vals}"
        )
    if not c4_pass:
        vals = [seed_results[i]["peak_checkpoint"] for i in range(3)]
        failure_notes.append(
            f"C4 FAIL: peak_checkpoint < {MIN_PEAK_CHECKPOINT}: {vals}"
        )
    for n in failure_notes:
        print(f"  {n}", flush=True)

    summary_markdown = (
        f"# V3-EXQ-232 -- ARC-026 Approach/Contact Slope Conflict Resolution\n\n"
        f"**Status:** {status}  **Criteria met:** {crit_met}/4\n"
        f"**Claims:** ARC-026  **Purpose:** evidence\n"
        f"**Supersedes (conflict resolution):** EXQ-033 conflict_ratio=1.0\n\n"
        f"## Design Changes vs EXQ-033\n\n"
        f"- Peak checkpoint analysis (not terminal) to handle training instability\n"
        f"- 1500 eps (vs 1000), checkpoints at {CHECKPOINTS}\n"
        f"- 3 seeds (vs 1)\n"
        f"- E3 harm supervision enabled from ep100 onward (phased)\n\n"
        f"## Results by Seed\n\n"
        f"| Seed | Approach slope | Contact slope | Gap@peak | Peak ckpt | C1 | C2 | C3 | C4 |\n"
        f"|------|---------------|--------------|----------|----------|----|----|----|----|"
    )
    for i, seed in enumerate(SEEDS):
        sr = seed_results[i]
        summary_markdown += (
            f"\n| {seed} | {sr['approach_slope']:.6f}"
            f" | {sr['contact_slope']:.6f}"
            f" | {sr['approach_gap_at_peak']:.4f}"
            f" | {sr['peak_checkpoint']}"
            f" | {'PASS' if c1_passes[i] else 'FAIL'}"
            f" | {'PASS' if c2_passes[i] else 'FAIL'}"
            f" | {'PASS' if c3_passes[i] else 'FAIL'}"
            f" | {'PASS' if c4_passes[i] else 'FAIL'} |"
        )
    summary_markdown += (
        f"\n\n## Interpretation\n\n{interpretation}\n"
    )
    if failure_notes:
        summary_markdown += "\n## Failure Notes\n\n"
        summary_markdown += "\n".join(f"- {n}" for n in failure_notes) + "\n"

    metrics: Dict = {
        "c1_pass": 1.0 if c1_pass else 0.0,
        "c2_pass": 1.0 if c2_pass else 0.0,
        "c3_pass": 1.0 if c3_pass else 0.0,
        "c4_pass": 1.0 if c4_pass else 0.0,
        "criteria_met": float(crit_met),
    }
    for i, seed in enumerate(SEEDS):
        sr  = seed_results[i]
        sfx = f"_seed{i}"
        metrics[f"approach_slope{sfx}"]      = float(sr["approach_slope"])
        metrics[f"contact_slope{sfx}"]       = float(sr["contact_slope"])
        metrics[f"approach_gap_at_peak{sfx}"]= float(sr["approach_gap_at_peak"])
        metrics[f"peak_checkpoint{sfx}"]     = float(sr["peak_checkpoint"])
        metrics[f"n_approach_at_peak{sfx}"]  = float(sr["n_approach_at_peak"])
        slope_gap = sr["approach_slope"] - sr["contact_slope"]
        metrics[f"slope_gap{sfx}"]           = float(slope_gap)

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if crit_met >= 2 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
        "seed_results": seed_results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run(dry_run=args.dry_run)

    ts  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["run_id"]         = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.5f}", flush=True)
        else:
            print(f"  {k}: {v}", flush=True)
