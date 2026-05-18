#!/opt/local/bin/python3
"""
V3-EXQ-454a -- ARC-016 adaptive commit_threshold via E2 PE variance (reef).

Supersedes: V3-EXQ-454 (non_contributory -- monostrategy on default 10x10 env
prevented behavioral diversity, making C2 regime sensitivity unmeasurable).
Reef enrichment (SD-054) breaks monostrategy by adding safe-zone attractors and
food-attracted hazard drift.

Claims: ARC-016

ARC-016 (active in claims.yaml, v3_pending=True) claims that the E3
commit_threshold should be precision-adaptive: commit_threshold(t) =
k * warmup_baseline_std(PE). High E3 prediction-error variance -> low
precision -> higher commit_threshold (committing is riskier when noisy).
Low PE variance -> high precision -> lower commit_threshold (committing
is cheap when the model is confident).

Substrate: e3.current_precision (= 1/running_variance) already exists.
What is missing is the feedback loop from current_precision back to the
commit_threshold itself. This experiment wires that loop via a script-
level monkey-patch on agent.e3.select (NO ree_core modification).

Conditions (3):
  STATIC_LOW:   commit_threshold=0.3 (default low, static)
  STATIC_HIGH:  commit_threshold=0.8 (default high, static)
  ADAPTIVE:     commit_threshold computed each tick from e3.running_variance
                via the ARC-016 scaling: threshold = k / sqrt(rv),
                with k calibrated during P1 from mean sqrt(rv).

Env: CausalGridWorldV2, num_hazards=3, size=10, hazard_harm=0.04,
proximity_harm_scale=0.12, num_resources=3. SD-018 resource proximity
head enabled so rewards are structured.

Phases:
  P0_EPS=30:  encoder warmup (identical across conditions)
  P1_EPS=50:  main training. In ADAPTIVE this is the commit_threshold
              calibration window (collect sqrt(rv) samples, set k).
              In STATIC_* arms, commit_threshold is fixed throughout.
  P2_EPS=30:  evaluation.

Seeds: [42, 7, 13]; steps/ep = 150.

Metrics (P2):
  commit_rate:                   fraction of E3 ticks that elevate beta_gate
  commit_survival:               mean duration of beta_gate elevations (ticks)
  pe_variance_mean:              mean running_variance observed in P2
  commit_rate_by_pe_regime:      {low_pe_var_commit, high_pe_var_commit}
                                 split at P2 rv median
  goal_reach_rate:               fraction of eps reaching a resource

Acceptance:
  C1 static_low_behavioural_signature:
      STATIC_LOW commit_rate > STATIC_HIGH commit_rate in >=2/3 seeds
      (sanity: low threshold commits more)
  C2 adaptive_regime_sensitivity:
      ADAPTIVE (low_pe_commit - high_pe_commit) >= 0.05 in >=2/3 seeds
      (threshold varies with precision state)
  C3 adaptive_not_degenerate:
      ADAPTIVE mean commit_rate in (0.1, 0.9) in >=2/3 seeds
  C4 adaptive_goal_performance (diagnostic):
      ADAPTIVE goal_reach_rate >= STATIC_LOW goal_reach_rate - 0.1
      in >=2/3 seeds

PASS = C1 AND C2 AND C3. C4 is diagnostic.

claim_ids: ["ARC-016"]
experiment_purpose: "evidence"
"""

import sys
import json
import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_454a_arc016_adaptive_commitment_reef"
CLAIM_IDS = ["ARC-016"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 7, 13]
STEPS_PER_EP = 150
P0_EPS = 30
P1_EPS = 50
P2_EPS = 30

CONDITIONS = ["STATIC_LOW", "STATIC_HIGH", "ADAPTIVE"]

STATIC_LOW_THRESH = 0.3
STATIC_HIGH_THRESH = 0.8
# Safety clamp for adaptive threshold.
ADAPTIVE_MIN = 0.05
ADAPTIVE_MAX = 2.0
# Base threshold k is scaled so the mean calibration threshold starts
# near STATIC_LOW (the "confident" operating point for an adaptive agent).
ADAPTIVE_BASE_THRESH = 0.3


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=3,
        num_resources=3,
        hazard_harm=0.04,
        proximity_harm_scale=0.12,
        # SD-054 reef enrichment -- breaks monostrategy on 10x10 grid
        reef_enabled=True,
        n_reef_patches=3,
        reef_patch_radius=2,
        hazard_food_attraction=0.7,
    )


def _make_agent(env: CausalGridWorldV2, condition: str) -> REEAgent:
    if condition == "STATIC_LOW":
        ct = STATIC_LOW_THRESH
    elif condition == "STATIC_HIGH":
        ct = STATIC_HIGH_THRESH
    else:
        # ADAPTIVE: seed config with the base; monkey-patch will override.
        ct = ADAPTIVE_BASE_THRESH

    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
    )
    cfg.e3.commitment_threshold = ct
    return REEAgent(cfg)


class AdaptiveThresholdController:
    """Precision-adaptive commit_threshold wrapper for ARC-016.

    Wraps the agent's E3 selector by monkey-patching e3.select. On each
    call:
      1. If in calibration mode, record sqrt(running_variance) samples.
      2. If calibrated, set config.commitment_threshold = k / sqrt(rv)
         clipped to [ADAPTIVE_MIN, ADAPTIVE_MAX].
      3. Delegate to the original e3.select.
    Calibration: call finalise_calibration() at end of P1 to compute
    k = base_threshold * mean(sqrt(rv_samples)), which sets the scale so
    that the typical post-P1 threshold equals ADAPTIVE_BASE_THRESH.
    """

    def __init__(self, agent: REEAgent, base_threshold: float = ADAPTIVE_BASE_THRESH):
        self.agent = agent
        self.e3 = agent.e3
        self.base_threshold = base_threshold
        self.k: Optional[float] = None
        self.calibrating = False
        self.rv_samples: List[float] = []
        self._orig_select = self.e3.select
        self._install()

    def _install(self):
        ctrl = self

        def patched_select(candidates, *args, **kwargs):
            rv = float(ctrl.e3._running_variance)
            if rv < 1e-6:
                rv = 1e-6
            # Record calibration samples.
            if ctrl.calibrating:
                ctrl.rv_samples.append(math.sqrt(rv))
            # Apply adaptive threshold once k is set.
            if ctrl.k is not None:
                adaptive_thresh = ctrl.k / math.sqrt(rv)
                adaptive_thresh = max(ADAPTIVE_MIN, min(ADAPTIVE_MAX, adaptive_thresh))
                ctrl.e3.config.commitment_threshold = adaptive_thresh
            return ctrl._orig_select(candidates, *args, **kwargs)

        self.e3.select = patched_select

    def start_calibration(self):
        self.calibrating = True
        self.rv_samples = []

    def finalise_calibration(self):
        self.calibrating = False
        if self.rv_samples:
            mean_sqrt_rv = sum(self.rv_samples) / len(self.rv_samples)
            # Target: adaptive_thresh ~ base_threshold at typical rv.
            # adaptive_thresh = k / sqrt(rv) => k = base_threshold * sqrt(rv_typ)
            self.k = self.base_threshold * mean_sqrt_rv
        else:
            # Fallback: use base_threshold as k.
            self.k = self.base_threshold


def _obs_tensors(obs_dict):
    body = obs_dict["body_state"].float().unsqueeze(0)
    world = obs_dict["world_state"].float().unsqueeze(0)
    harm = obs_dict["harm_obs"].float().unsqueeze(0) if "harm_obs" in obs_dict else None
    harm_a = obs_dict["harm_obs_a"].float().unsqueeze(0) if "harm_obs_a" in obs_dict else None
    harm_hist = obs_dict["harm_history"].float().unsqueeze(0) if "harm_history" in obs_dict else None
    return body, world, harm, harm_a, harm_hist


def _run_condition(seed: int, condition: str, verbose: bool = True) -> Dict:
    torch.manual_seed(seed)

    env = _make_env(seed)
    agent = _make_agent(env, condition)

    adaptive_ctrl: Optional[AdaptiveThresholdController] = None
    if condition == "ADAPTIVE":
        adaptive_ctrl = AdaptiveThresholdController(agent, base_threshold=ADAPTIVE_BASE_THRESH)

    total_eps = P0_EPS + P1_EPS + P2_EPS
    phase_boundaries = (P0_EPS, P0_EPS + P1_EPS)

    # P2 metric accumulators.
    p2_beta_elevated_ticks = 0
    p2_total_ticks = 0
    p2_elevation_runs: List[int] = []  # durations
    current_run_len = 0
    p2_rv_samples: List[float] = []
    p2_rv_at_tick: List[float] = []
    p2_commit_at_tick: List[int] = []
    goal_reach_ep_flags: List[int] = []

    calibration_started = False

    for ep_idx in range(total_eps):
        agent.reset()
        _obs, obs_dict = env.reset()

        phase_is_p1 = (ep_idx >= phase_boundaries[0]) and (ep_idx < phase_boundaries[1])
        phase_is_p2 = ep_idx >= phase_boundaries[1]

        # Begin calibration at P1 start for ADAPTIVE.
        if adaptive_ctrl is not None and phase_is_p1 and not calibration_started:
            adaptive_ctrl.start_calibration()
            calibration_started = True

        # Finalise calibration at P2 entry.
        if adaptive_ctrl is not None and phase_is_p2 and adaptive_ctrl.calibrating:
            adaptive_ctrl.finalise_calibration()

        if ep_idx % 10 == 0:
            phase = "P0" if ep_idx < P0_EPS else ("P1" if phase_is_p1 else "P2")
            print(
                f"[train] seed={seed} cond={condition} ep {ep_idx}/{total_eps} phase={phase}",
                flush=True,
            )

        ep_reached = 0

        for step in range(STEPS_PER_EP):
            body, world, harm, harm_a, harm_hist = _obs_tensors(obs_dict)
            latent = agent.sense(
                obs_body=body,
                obs_world=world,
                obs_harm=harm,
                obs_harm_a=harm_a,
                obs_harm_history=harm_hist,
            )
            ticks = agent.clock.advance()
            world_dim_local = latent.z_world.shape[-1]
            e1_prior = (
                agent._e1_tick(latent) if ticks["e1_tick"]
                else torch.zeros(1, world_dim_local, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)

            # Metric collection (P2 only).
            if phase_is_p2 and ticks.get("e3_tick", False):
                rv = float(agent.e3._running_variance)
                p2_rv_samples.append(rv)
                elevated = bool(agent.beta_gate.is_elevated)
                p2_total_ticks += 1
                if elevated:
                    p2_beta_elevated_ticks += 1
                    current_run_len += 1
                else:
                    if current_run_len > 0:
                        p2_elevation_runs.append(current_run_len)
                        current_run_len = 0
                p2_rv_at_tick.append(rv)
                p2_commit_at_tick.append(1 if elevated else 0)

            _obs, harm_signal, done, info, obs_dict = env.step(action)
            hs = float(harm_signal)
            # Positive harm_signal = benefit (resource collected in this env).
            if hs > 0:
                ep_reached = 1

            if done:
                break

        # Close any open elevation run at episode end.
        if phase_is_p2 and current_run_len > 0:
            p2_elevation_runs.append(current_run_len)
            current_run_len = 0

        if phase_is_p2:
            goal_reach_ep_flags.append(ep_reached)

    # Aggregate P2 metrics.
    commit_rate = p2_beta_elevated_ticks / p2_total_ticks if p2_total_ticks > 0 else 0.0
    commit_survival = (sum(p2_elevation_runs) / len(p2_elevation_runs)) if p2_elevation_runs else 0.0
    pe_variance_mean = (sum(p2_rv_samples) / len(p2_rv_samples)) if p2_rv_samples else 0.0

    # Split commit_rate by PE regime (rv low vs high at tick, split at median).
    if p2_rv_at_tick:
        sorted_rv = sorted(p2_rv_at_tick)
        median_rv = sorted_rv[len(sorted_rv) // 2]
        low_commits = 0
        low_total = 0
        high_commits = 0
        high_total = 0
        for rv, c in zip(p2_rv_at_tick, p2_commit_at_tick):
            if rv <= median_rv:
                low_total += 1
                low_commits += c
            else:
                high_total += 1
                high_commits += c
        low_rate = low_commits / low_total if low_total > 0 else 0.0
        high_rate = high_commits / high_total if high_total > 0 else 0.0
    else:
        median_rv = 0.0
        low_rate = 0.0
        high_rate = 0.0

    goal_reach_rate = (sum(goal_reach_ep_flags) / len(goal_reach_ep_flags)) if goal_reach_ep_flags else 0.0

    # Per-condition unit verdict (diagnostic).
    if condition == "ADAPTIVE":
        unit_pass = (0.1 < commit_rate < 0.9) and ((low_rate - high_rate) >= 0.05)
    elif condition == "STATIC_LOW":
        unit_pass = commit_rate > 0.0
    else:  # STATIC_HIGH
        unit_pass = True

    verdict = "PASS" if unit_pass else "FAIL"

    result = {
        "seed": seed,
        "condition": condition,
        "commit_rate": float(commit_rate),
        "commit_survival": float(commit_survival),
        "pe_variance_mean": float(pe_variance_mean),
        "commit_rate_by_pe_regime": {
            "low_pe_var_commit": float(low_rate),
            "high_pe_var_commit": float(high_rate),
            "median_rv_split": float(median_rv),
        },
        "goal_reach_rate": float(goal_reach_rate),
        "n_p2_e3_ticks": int(p2_total_ticks),
        "n_elevation_runs": len(p2_elevation_runs),
        "adaptive_k": (adaptive_ctrl.k if adaptive_ctrl is not None else None),
        "final_commitment_threshold": float(agent.e3.config.commitment_threshold),
    }

    if verbose:
        print(
            f"  [seed={seed} {condition}] "
            f"commit_rate={commit_rate:.3f} "
            f"survival={commit_survival:.2f} "
            f"rv_mean={pe_variance_mean:.4f} "
            f"low_rate={low_rate:.3f} high_rate={high_rate:.3f} "
            f"goal_reach={goal_reach_rate:.3f} "
            f"verdict: {verdict}",
            flush=True,
        )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.dry_run:
        print("Smoke: seed=42, cond=ADAPTIVE, P0=2/P1=2/P2=2 eps, 40 steps/ep")
        global P0_EPS, P1_EPS, P2_EPS, STEPS_PER_EP
        P0_EPS, P1_EPS, P2_EPS, STEPS_PER_EP = 2, 2, 2, 40
        print("Seed 42 Condition ADAPTIVE")
        r = _run_condition(seed=42, condition="ADAPTIVE", verbose=True)
        print(f"  ADAPTIVE: {r}")
        print("Smoke test PASSED")
        return

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        script_dir = Path(__file__).resolve().parents[1]
        out_dir = (
            script_dir.parent / "REE_assembly" / "evidence"
            / "experiments" / EXPERIMENT_TYPE
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for seed in SEEDS:
        print(f"Seed {seed}")
        for cond in CONDITIONS:
            print(f"Seed {seed} Condition {cond}")
            r = _run_condition(seed=seed, condition=cond)
            all_results.append(r)

    def by_cond(c):
        return [r for r in all_results if r["condition"] == c]

    static_low = by_cond("STATIC_LOW")
    static_high = by_cond("STATIC_HIGH")
    adaptive = by_cond("ADAPTIVE")

    # C1: STATIC_LOW commit_rate > STATIC_HIGH commit_rate in >=2/3 seeds.
    c1_wins = 0
    for sl, sh in zip(
        sorted(static_low, key=lambda r: r["seed"]),
        sorted(static_high, key=lambda r: r["seed"]),
    ):
        if sl["commit_rate"] > sh["commit_rate"]:
            c1_wins += 1
    c1 = c1_wins >= 2

    # C2: ADAPTIVE low_rate - high_rate >= 0.05 in >=2/3 seeds.
    c2_wins = 0
    for r in adaptive:
        low = r["commit_rate_by_pe_regime"]["low_pe_var_commit"]
        high = r["commit_rate_by_pe_regime"]["high_pe_var_commit"]
        if (low - high) >= 0.05:
            c2_wins += 1
    c2 = c2_wins >= 2

    # C3: ADAPTIVE commit_rate in (0.1, 0.9) in >=2/3 seeds.
    c3_wins = sum(1 for r in adaptive if 0.1 < r["commit_rate"] < 0.9)
    c3 = c3_wins >= 2

    # C4 diagnostic: ADAPTIVE goal_reach >= STATIC_LOW goal_reach - 0.1 in >=2/3.
    c4_wins = 0
    for ad, sl in zip(
        sorted(adaptive, key=lambda r: r["seed"]),
        sorted(static_low, key=lambda r: r["seed"]),
    ):
        if ad["goal_reach_rate"] >= (sl["goal_reach_rate"] - 0.1):
            c4_wins += 1
    c4 = c4_wins >= 2

    outcome = "PASS" if (c1 and c2 and c3) else "FAIL"

    summary = {
        "c1_static_low_behavioural_signature": {
            "wins": c1_wins,
            "pass": c1,
            "desc": "STATIC_LOW commit_rate > STATIC_HIGH commit_rate in >=2/3 seeds",
        },
        "c2_adaptive_regime_sensitivity": {
            "wins": c2_wins,
            "pass": c2,
            "desc": "ADAPTIVE (low_pe_commit - high_pe_commit) >= 0.05 in >=2/3 seeds",
        },
        "c3_adaptive_not_degenerate": {
            "wins": c3_wins,
            "pass": c3,
            "desc": "ADAPTIVE commit_rate in (0.1, 0.9) in >=2/3 seeds",
        },
        "c4_adaptive_goal_performance_diagnostic": {
            "wins": c4_wins,
            "pass": c4,
            "desc": "ADAPTIVE goal_reach_rate >= STATIC_LOW - 0.1 (diagnostic only)",
        },
    }

    print(f"\nOutcome: {outcome}")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    per_claim = {
        "ARC-016": "supports" if outcome == "PASS" else "weakens",
    }

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": outcome,
        "evidence_direction": "supports" if outcome == "PASS" else "weakens",
        "evidence_direction_per_claim": per_claim,
        "supersedes": "v3_exq_454_arc016_adaptive_commitment_threshold",
        "pass_criteria_summary": summary,
        "per_seed_results": all_results,
        "config": {
            "conditions": CONDITIONS,
            "seeds": SEEDS,
            "p0_eps": P0_EPS,
            "p1_eps": P1_EPS,
            "p2_eps": P2_EPS,
            "steps_per_ep": STEPS_PER_EP,
            "static_low_thresh": STATIC_LOW_THRESH,
            "static_high_thresh": STATIC_HIGH_THRESH,
            "adaptive_base_thresh": ADAPTIVE_BASE_THRESH,
            "adaptive_min": ADAPTIVE_MIN,
            "adaptive_max": ADAPTIVE_MAX,
        },
    }

    out_file = out_dir / f"{run_id}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Output written to: {out_file}")


if __name__ == "__main__":
    main()
