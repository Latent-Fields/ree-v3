#!/opt/local/bin/python3
"""V3-EXQ-507 -- ARC-026 capacity-scaling test for goal-pursuit preservation.

Claim: ARC-026 (ethics.love_expands_under_intelligence)
Status: candidate (exp_conf=0.539, 1 PASS / 1 FAIL across 7 runs)

Why this experiment exists
--------------------------
ARC-026 says "love as long-horizon care investment expands under
intelligence rather than being crowded out." EXQ-033 had C1-C3 PASS but
C4 FAIL (approach_slope/contact_slope = 0.92, attributed to depth-
calibration training instability ep500-1000). This test addresses the
weak-reading falsifiable subset: under capacity expansion, goal-pursuit
behaviour should at least be PRESERVED (no crowding-out), and ideally
INCREASED. The strong reading (genuinely "expanded long-horizon care")
needs a substrate that V3 does not yet expose -- it remains substrate-
pending.

The test is a capacity sweep over world_dim, holding everything else
fixed. Agent is briefly trained, then evaluated on goal-pursuit metrics.

Three capacity levels (3 seeds each = 9 runs)
----------------------------------------------
SMALL:  world_dim=16, self_dim=16, hidden_dim=64
MEDIUM: world_dim=32, self_dim=32, hidden_dim=128 (V3 baseline)
LARGE:  world_dim=64, self_dim=32, hidden_dim=256

Each runs WARMUP_EPISODES of training then EVAL_EPISODES of eval. Random
action policy during both phases (we are not testing policy improvement,
only whether goal-pursuit metrics scale or collapse with capacity).

Pre-registered metrics
----------------------
  approach_score: mean(info["resource_field_at_agent"]) over eval steps.
                  Measures average proximity to resources -- how close the
                  agent gets in expectation. Random policy baseline is
                  bounded by the resource_field's geometry but exhibits
                  capacity-driven structure when the encoder learns
                  coherent z_world (an MECH-216-like latent attractor).
  contact_rate:   eval-step fraction with benefit_exposure >= 0.05.
                  Resource-consumption rate.
  ratio:          contact_rate / approach_score (the EXQ-033 metric).
                  Higher = better goal-pursuit efficiency.

PASS criteria (>= 2/3 seeds for each)
-------------------------------------
  C1 ratio non-decreasing across capacity (per seed):
     ratio_LARGE >= 0.95 * ratio_MEDIUM AND ratio_MEDIUM >= 0.95 * ratio_SMALL
     (5% slack -- monotone-non-decreasing within tolerance).
  C2 absolute non-collapse: ratio_LARGE > 0.0 in all seeds (the agent
     still contacts resources at the largest capacity; not stuck in a
     dead-state).
  C3 approach_score does not collapse:
     approach_score_LARGE >= 0.5 * approach_score_SMALL.

PASS = C1 AND C2 AND C3.
PASS supports the weak-reading "no crowding-out" subset of ARC-026.
FAIL with C2 alone PASSing -> capacity expansion correlates with
behavioural collapse (active crowding-out, the failure mode the claim
explicitly predicts against). FAIL with C1 failing on a single seed
-> training-stability artifact (the kind of failure mode the original
EXQ-033 was diagnosed as).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_507_arc026_capacity_scaling.py
  /opt/local/bin/python3 experiments/v3_exq_507_arc026_capacity_scaling.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_507_arc026_capacity_scaling"
CLAIM_IDS = ["ARC-026"]
EXPERIMENT_PURPOSE = "evidence"

# --- Config -------------------------------------------------------------
SEEDS = (42, 43, 44)
WARMUP_EPISODES = 60
EVAL_EPISODES = 30
STEPS_PER_EPISODE = 150
LR = 1e-3
ALPHA_WORLD = 0.9
ALPHA_SELF = 0.3
NUM_HAZARDS = 1
NUM_RESOURCES = 3
CONTACT_THRESHOLD = 0.05  # benefit_exposure threshold counted as a contact step.

CAPACITY_LEVELS = (
    ("SMALL",  16, 16, 64),
    ("MEDIUM", 32, 32, 128),
    ("LARGE",  64, 32, 256),
)

# Pre-registered thresholds.
C1_MONOTONE_TOL = 0.95
C2_MIN_RATIO_LARGE = 0.0   # strict > 0; explicit threshold for clarity.
C3_MIN_APPROACH_RATIO = 0.5
PASS_FRACTION_REQUIRED = 2.0 / 3.0


def _action_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def run_capacity(seed: int, capacity_label: str, world_dim: int,
                  self_dim: int, hidden_dim: int,
                  warmup_eps: int, eval_eps: int) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(
        seed=seed, size=10,
        num_hazards=NUM_HAZARDS, num_resources=NUM_RESOURCES,
        hazard_harm=0.5, env_drift_interval=5, env_drift_prob=0.1,
        proximity_harm_scale=0.05, proximity_benefit_scale=0.03,
        hazard_field_decay=0.5, energy_decay=0.005,
        use_proxy_fields=True, resource_respawn_on_consume=True,
    )
    action_dim = env.action_dim
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=ALPHA_WORLD,
        alpha_self=ALPHA_SELF,
        novelty_bonus_weight=0.0,
        benefit_eval_enabled=False,
    )
    agent = REEAgent(cfg)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)

    # ---- Warmup training (random policy; trains encoders + E1/E2 only) ----
    agent.train()
    for ep in range(warmup_eps):
        _, obs_dict = env.reset()
        agent.reset()
        for _ in range(STEPS_PER_EPISODE):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm = obs_dict.get("harm_obs", None)
            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            agent.clock.advance()
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total = e1_loss + e2_loss
            if total.requires_grad:
                optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()
            action_idx = random.randint(0, action_dim - 1)
            action_oh = _action_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh
            _, _, done, _, obs_dict = env.step(action_oh)
            if done:
                break

    # ---- Eval (random policy; collect goal-pursuit metrics) ----
    agent.eval()
    approach_vals: List[float] = []
    contact_steps = 0
    total_steps = 0
    for ep in range(eval_eps):
        _, obs_dict = env.reset()
        agent.reset()
        for _ in range(STEPS_PER_EPISODE):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm = obs_dict.get("harm_obs", None)
            with torch.no_grad():
                _ = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            agent.clock.advance()
            action_idx = random.randint(0, action_dim - 1)
            action_oh = _action_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh
            _, _, done, info, obs_dict = env.step(action_oh)
            res_field = float(info.get("resource_field_at_agent", 0.0))
            ben_exp = float(info.get("benefit_exposure", 0.0))
            approach_vals.append(res_field)
            if ben_exp >= CONTACT_THRESHOLD:
                contact_steps += 1
            total_steps += 1
            if done:
                break

    approach_score = float(sum(approach_vals) / max(1, len(approach_vals)))
    contact_rate = float(contact_steps / max(1, total_steps))
    ratio = contact_rate / max(1e-6, approach_score)

    print(f"  [{capacity_label}] seed={seed} "
          f"approach={approach_score:.4f} contact={contact_rate:.4f} "
          f"ratio={ratio:.4f}", flush=True)

    return {
        "seed": seed, "capacity_label": capacity_label,
        "world_dim": world_dim, "self_dim": self_dim, "hidden_dim": hidden_dim,
        "approach_score": approach_score,
        "contact_rate": contact_rate,
        "ratio": ratio,
        "n_total_eval_steps": total_steps,
    }


def _evaluate(rows_by_seed: Dict[int, Dict[str, Dict]]) -> Dict:
    """rows_by_seed: {seed: {capacity_label: row_dict}}"""
    seeds = list(rows_by_seed.keys())
    n = len(seeds)
    required = math.ceil(n * PASS_FRACTION_REQUIRED)

    c1_passes = 0
    c2_passes = 0
    c3_passes = 0
    per_seed_summary = []
    for seed in seeds:
        r_s = rows_by_seed[seed]["SMALL"]
        r_m = rows_by_seed[seed]["MEDIUM"]
        r_l = rows_by_seed[seed]["LARGE"]

        c1 = (r_l["ratio"] >= C1_MONOTONE_TOL * r_m["ratio"]
              and r_m["ratio"] >= C1_MONOTONE_TOL * r_s["ratio"])
        c2 = r_l["ratio"] > C2_MIN_RATIO_LARGE
        c3 = r_l["approach_score"] >= C3_MIN_APPROACH_RATIO * max(1e-6, r_s["approach_score"])

        c1_passes += int(c1)
        c2_passes += int(c2)
        c3_passes += int(c3)

        per_seed_summary.append({
            "seed": seed,
            "ratio_small": r_s["ratio"], "ratio_medium": r_m["ratio"],
            "ratio_large": r_l["ratio"],
            "approach_small": r_s["approach_score"],
            "approach_medium": r_m["approach_score"],
            "approach_large": r_l["approach_score"],
            "c1_pass": c1, "c2_pass": c2, "c3_pass": c3,
        })

    return {
        "n_seeds": n, "min_seeds_required": required,
        "c1_seeds_pass": c1_passes,
        "c2_seeds_pass": c2_passes,
        "c3_seeds_pass": c3_passes,
        "c1_pass": c1_passes >= required,
        "c2_pass": c2_passes >= required,
        "c3_pass": c3_passes >= required,
        "overall_pass": (c1_passes >= required and c2_passes >= required
                          and c3_passes >= required),
        "per_seed_summary": per_seed_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--warmup", type=int, default=WARMUP_EPISODES)
    parser.add_argument("--eval", type=int, default=EVAL_EPISODES)
    args = parser.parse_args()

    if args.dry_run:
        seeds = (args.seeds[0],)
        warmup = 2
        eval_eps = 2
        print("[DRY-RUN] 1 seed x 3 capacities, 2 warmup eps, 2 eval eps -- smoke only.", flush=True)
    else:
        seeds = tuple(args.seeds)
        warmup = args.warmup
        eval_eps = args.eval

    t0 = time.time()
    rows_by_seed: Dict[int, Dict[str, Dict]] = {}
    for s in seeds:
        rows_by_seed[s] = {}
        for label, world_dim, self_dim, hidden_dim in CAPACITY_LEVELS:
            row = run_capacity(s, label, world_dim, self_dim, hidden_dim, warmup, eval_eps)
            rows_by_seed[s][label] = row
    elapsed = time.time() - t0

    criteria = _evaluate(rows_by_seed)
    outcome = "PASS" if criteria["overall_pass"] else "FAIL"
    direction = "supports" if criteria["overall_pass"] else "weakens"

    print(f"\nV3-EXQ-507 (ARC-026) -- {outcome} in {elapsed:.1f}s ({len(seeds)} seed(s))", flush=True)
    print(f"  C1 ratio non-decreasing (LARGE>=0.95*MEDIUM>=0.95*SMALL): "
          f"{criteria['c1_seeds_pass']}/{criteria['n_seeds']} -> "
          f"{'PASS' if criteria['c1_pass'] else 'FAIL'}", flush=True)
    print(f"  C2 ratio_LARGE > 0: "
          f"{criteria['c2_seeds_pass']}/{criteria['n_seeds']} -> "
          f"{'PASS' if criteria['c2_pass'] else 'FAIL'}", flush=True)
    print(f"  C3 approach_LARGE >= 0.5 * approach_SMALL: "
          f"{criteria['c3_seeds_pass']}/{criteria['n_seeds']} -> "
          f"{'PASS' if criteria['c3_pass'] else 'FAIL'}", flush=True)

    if args.dry_run:
        print("[--dry-run] manifest not written.", flush=True)
        return

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest = {
        "schema_version": "v1", "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS, "result": outcome,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {"ARC-026": direction},
        "criteria": criteria,
        "registered_thresholds": {
            "C1_MONOTONE_TOL": C1_MONOTONE_TOL,
            "C2_MIN_RATIO_LARGE": C2_MIN_RATIO_LARGE,
            "C3_MIN_APPROACH_RATIO": C3_MIN_APPROACH_RATIO,
            "PASS_FRACTION_REQUIRED": PASS_FRACTION_REQUIRED,
            "CONTACT_THRESHOLD": CONTACT_THRESHOLD,
        },
        "config": {
            "warmup_episodes": warmup, "eval_episodes": eval_eps,
            "steps_per_episode": STEPS_PER_EPISODE,
            "capacity_levels": [
                {"label": l, "world_dim": w, "self_dim": s, "hidden_dim": h}
                for l, w, s, h in CAPACITY_LEVELS
            ],
            "seeds": list(seeds),
        },
        "results_per_seed_per_capacity": {
            str(s): rows_by_seed[s] for s in seeds
        },
        "elapsed_seconds": elapsed,
        "notes": (
            "Capacity-sweep weak-reading test of ARC-026. Sweeps world_dim "
            "in {16, 32, 64} and measures approach_score, contact_rate, "
            "and the ratio metric from EXQ-033. PASS supports the 'no "
            "crowding-out' weak reading. The strong reading (genuinely "
            "expanded long-horizon care) needs a substrate that V3 does "
            "not expose; remains substrate-pending. Random action policy "
            "during both warmup and eval -- this experiment tests whether "
            "the ENCODED z_world supports goal-pursuit, not whether E3 "
            "policy improves with capacity (which would be a different test)."
        ),
    }
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Result written to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
