#!/opt/local/bin/python3
"""
V3-EXQ-229 -- MECH-128: E1 Goal-Conditioning Discriminative Pair

Claims: MECH-128
EXPERIMENT_PURPOSE = "evidence"

Scientific question: When E1 receives z_goal_latent as context (vs. zeros),
does E1 prediction error on goal-relevant transitions decrease more than on
goal-irrelevant transitions?

Context:
  EXQ-147a had mixed result. This experiment uses the same design but with:
  - Larger warmup (400 eps) to ensure z_goal seeding via SD-012 (drive_weight=2.0)
  - Stricter precondition check (z_goal_norm > 0.05 in all seeds required)
  - Cleaner zone-specific error tracking (100 eval eps)
  - BLOCKED NOTE: requires z_goal_latent > 0.05 as precondition

MECH-128 asserts:
  E1's LSTM hidden state, when conditioned on z_goal_latent, produces
  lower prediction error on goal-relevant transitions (near resource/benefit
  zones) than when goal context is withheld. Goal conditioning is
  zone-specific (not a general E1 prediction improvement).

Design:
  GOAL_CONDITIONED: E1 receives [z_self, z_world, z_goal_latent].
    config.e1.goal_dim = WORLD_DIM (goal_input_proj active in E1DeepPredictor).
    drive_weight=2.0, z_goal_enabled=True.
  GOAL_ABLATED: E1 receives [z_self, z_world] only -- goal context withheld.
    config.e1.goal_dim = 0 (goal_input_proj=None).
    z_goal still updated (drive active) but NOT passed to E1.

  3 seeds x 400 warmup + 100 eval eps x 200 steps.

  Key metric -- interaction:
    interaction = (err_neutral_ablated - err_neutral_conditioned)
                - (err_goal_ablated    - err_goal_conditioned)
  Positive interaction = goal conditioning selectively reduces E1 error
  on goal-relevant transitions (MECH-128 support).

Pre-registered PASS criteria:
  C1: interaction >= THRESH_INTERACTION in >= 2/3 seeds.
  C2: err_goal_conditioned < err_goal_ablated in >= 2/3 seeds.
  C3: |err_neutral_conditioned - err_neutral_ablated| < THRESH_NEUTRAL_DIFF
      in >= 2/3 seeds (goal-specificity).
  C4: z_goal_norm_conditioned > THRESH_GOAL_NORM in ALL seeds (SD-012 check).

PASS: C1 + C2 + C3 + C4
FAIL: C4 fails in any seed -> substrate_limitation
PARTIAL: C1 + C2 without C3 -> directional support, weaker specificity

evidence_direction_per_claim: MECH-128 only (single-claim experiment)
"""

import sys
import random
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE = "v3_exq_229_mech128_e1_goal_conditioning_pair"
CLAIM_IDS = ["MECH-128"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
THRESH_INTERACTION  = 0.005   # C1: interaction term
THRESH_NEUTRAL_DIFF = 0.03    # C3: neutral zone error must stay small
THRESH_GOAL_NORM    = 0.05    # C4: z_goal must be active (substrate check)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BODY_OBS_DIM  = 12
WORLD_OBS_DIM = 250
ACTION_DIM    = 5
WORLD_DIM     = 32
SELF_DIM      = 32

GOAL_ZONE_RADIUS = 3   # grid cells within nearest resource = "goal zone"
NEUTRAL_RADIUS   = 4   # cells from ALL resources AND hazards = "neutral"

WARMUP_EPISODES = 400
EVAL_EPISODES   = 100
STEPS_PER_EP    = 200

SEEDS = [42, 7, 123]


# ---------------------------------------------------------------------------
# Env / config factories
# ---------------------------------------------------------------------------

def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=2,
        num_resources=3,
        hazard_harm=0.02,
        env_drift_interval=10,
        env_drift_prob=0.2,
        resource_respawn_on_consume=True,
    )


def _make_config(goal_conditioned: bool) -> REEConfig:
    """
    Build REEConfig with or without E1 goal conditioning (MECH-128 toggle).
    drive_weight=2.0 (SD-012) amplifies benefit to seed z_goal.
    """
    config = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
        z_goal_enabled=True,
        e1_goal_conditioned=True,
        goal_weight=1.0,
        benefit_eval_enabled=True,
        benefit_weight=1.0,
        drive_weight=2.0,
    )
    # MECH-128 toggle: goal_dim=WORLD_DIM enables goal_input_proj in E1.
    # goal_dim=0 disables (goal context withheld from E1 LSTM).
    config.e1.goal_dim = WORLD_DIM if goal_conditioned else 0
    return config


# ---------------------------------------------------------------------------
# Run one condition + seed
# ---------------------------------------------------------------------------

def _run_condition(
    seed: int,
    condition: str,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    lr: float,
) -> Dict:
    goal_conditioned = (condition == "GOAL_CONDITIONED")

    torch.manual_seed(seed)
    random.seed(seed)

    env    = _make_env(seed)
    config = _make_config(goal_conditioned=goal_conditioned)
    agent  = REEAgent(config)

    e1_opt = optim.Adam(agent.e1.parameters(), lr=lr)
    e2_opt = optim.Adam(agent.e2.parameters(), lr=lr * 3)
    e3_opt = optim.Adam(
        list(agent.e3.parameters()) + list(agent.latent_stack.parameters()),
        lr=lr,
    )

    print(
        f"  [EXQ-229] {condition} seed={seed}"
        f" goal_conditioned={goal_conditioned}"
        f" e1_goal_dim={config.e1.goal_dim}"
        f" drive_weight={config.goal.drive_weight}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Warmup training
    # -----------------------------------------------------------------------
    agent.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_benefit = 0.0
        ep_harm    = 0.0

        for _ in range(steps_per_episode):
            obs_body  = torch.as_tensor(obs_dict["body_state"],  dtype=torch.float32)
            obs_world = torch.as_tensor(obs_dict["world_state"], dtype=torch.float32)

            latent = agent.sense(obs_body, obs_world)
            ticks  = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, WORLD_DIM, device=agent.device)
            )

            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action     = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = torch.zeros(1, ACTION_DIM)
                action[0, random.randint(0, ACTION_DIM - 1)] = 1.0

            _, reward, done, _, obs_dict = env.step(action)
            harm_signal    = float(reward) if float(reward) < 0 else 0.0
            benefit_signal = float(reward) if float(reward) > 0 else 0.0
            ep_harm    += abs(harm_signal)
            ep_benefit += benefit_signal

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total   = e1_loss + e2_loss
            if total.requires_grad:
                e1_opt.zero_grad()
                e2_opt.zero_grad()
                total.backward()
                e1_opt.step()
                e2_opt.step()

            # E3 harm supervision
            if agent._current_latent is not None:
                z_world_cur = agent._current_latent.z_world.detach()
                harm_target = torch.tensor(
                    [[1.0 if harm_signal < 0 else 0.0]], device=agent.device
                )
                harm_loss = F.mse_loss(agent.e3.harm_eval(z_world_cur), harm_target)
                if harm_loss.requires_grad:
                    e3_opt.zero_grad()
                    harm_loss.backward()
                    e3_opt.step()

            # z_goal update on benefit (drive_level=1.0 = fully depleted)
            if benefit_signal > 0 and agent.goal_state is not None:
                agent.update_z_goal(benefit_signal, drive_level=1.0)

            agent.update_residue(harm_signal)
            if done:
                break

        if (ep + 1) % 100 == 0:
            diag = agent.compute_goal_maintenance_diagnostic()
            print(
                f"    [train] {condition} seed={seed} ep {ep+1}/{warmup_episodes}"
                f" harm={ep_harm:.3f} benefit={ep_benefit:.3f}"
                f" goal_norm={diag['goal_norm']:.4f}",
                flush=True,
            )

    # -----------------------------------------------------------------------
    # Eval: collect E1 prediction error by zone
    # -----------------------------------------------------------------------
    agent.eval()

    err_goal_zone:    List[float] = []
    err_neutral_zone: List[float] = []
    goal_zone_steps   = 0
    neutral_steps     = 0
    total_steps_eval  = 0

    for _ in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = torch.as_tensor(obs_dict["body_state"],  dtype=torch.float32)
            obs_world = torch.as_tensor(obs_dict["world_state"], dtype=torch.float32)

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                ticks  = agent.clock.advance()
                if ticks.get("e1_tick", False):
                    agent._e1_tick(latent)

            e1_loss_val = agent.compute_prediction_loss()
            e1_err = float(e1_loss_val.item()) if hasattr(e1_loss_val, "item") else 0.0

            # Zone determination
            ax = int(getattr(env, "agent_x", 0))
            ay = int(getattr(env, "agent_y", 0))

            d_to_resource = float("inf")
            if hasattr(env, "resources") and env.resources:
                for rx, ry in env.resources:
                    d = abs(ax - int(rx)) + abs(ay - int(ry))
                    if d < d_to_resource:
                        d_to_resource = d

            d_to_hazard = float("inf")
            if hasattr(env, "hazards") and env.hazards:
                for hx, hy in env.hazards:
                    d = abs(ax - int(hx)) + abs(ay - int(hy))
                    if d < d_to_hazard:
                        d_to_hazard = d

            is_goal_zone = d_to_resource <= GOAL_ZONE_RADIUS
            is_neutral   = (
                d_to_resource > NEUTRAL_RADIUS
                and d_to_hazard > NEUTRAL_RADIUS
            )

            if is_goal_zone:
                err_goal_zone.append(e1_err)
                goal_zone_steps += 1
            elif is_neutral:
                err_neutral_zone.append(e1_err)
                neutral_steps += 1

            total_steps_eval += 1

            # Random action during eval
            action_idx = random.randint(0, ACTION_DIM - 1)
            action = torch.zeros(1, ACTION_DIM)
            action[0, action_idx] = 1.0
            _, reward, done, _, obs_dict = env.step(action)

            benefit_signal = float(reward) if float(reward) > 0 else 0.0
            if benefit_signal > 0 and agent.goal_state is not None:
                agent.update_z_goal(benefit_signal, drive_level=1.0)

            if done:
                break

    def _mean(lst: List[float]) -> float:
        return float(sum(lst) / max(1, len(lst)))

    mean_err_goal    = _mean(err_goal_zone)
    mean_err_neutral = _mean(err_neutral_zone)

    goal_norm_val = 0.0
    if agent.goal_state is not None:
        goal_norm_val = float(agent.goal_state.goal_norm())

    print(
        f"  [{condition}] seed={seed}"
        f" n_goal={goal_zone_steps} n_neutral={neutral_steps}"
        f" err_goal={mean_err_goal:.5f}"
        f" err_neutral={mean_err_neutral:.5f}"
        f" goal_norm={goal_norm_val:.4f}",
        flush=True,
    )

    return {
        "seed":             seed,
        "condition":        condition,
        "goal_conditioned": goal_conditioned,
        "mean_err_goal":    mean_err_goal,
        "mean_err_neutral": mean_err_neutral,
        "goal_zone_steps":  goal_zone_steps,
        "neutral_steps":    neutral_steps,
        "total_steps":      total_steps_eval,
        "goal_norm":        goal_norm_val,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ts      = int(time.time())
    root    = Path(__file__).resolve().parents[2]
    out_dir = root / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"

    warmup = 3   if args.dry_run else WARMUP_EPISODES
    n_eval = 3   if args.dry_run else EVAL_EPISODES
    steps  = 20  if args.dry_run else STEPS_PER_EP
    lr     = 1e-3

    print(
        f"[EXQ-229] MECH-128 E1 Goal-Conditioning Discriminative Pair (SD-012)",
        flush=True,
    )
    print(f"  dry_run={args.dry_run}", flush=True)

    all_results = []
    for seed in SEEDS:
        for condition in ["GOAL_CONDITIONED", "GOAL_ABLATED"]:
            res = _run_condition(
                seed=seed,
                condition=condition,
                warmup_episodes=warmup,
                eval_episodes=n_eval,
                steps_per_episode=steps,
                lr=lr,
            )
            all_results.append(res)

    # -----------------------------------------------------------------------
    # Per-seed interaction terms
    # -----------------------------------------------------------------------
    per_seed: Dict[int, Dict] = {}
    for r in all_results:
        s = r["seed"]
        if s not in per_seed:
            per_seed[s] = {}
        per_seed[s][r["condition"]] = r

    seed_metrics = []
    for seed in SEEDS:
        if seed not in per_seed:
            continue
        cond = per_seed[seed]
        if "GOAL_CONDITIONED" not in cond or "GOAL_ABLATED" not in cond:
            continue

        c_cond = cond["GOAL_CONDITIONED"]
        c_abl  = cond["GOAL_ABLATED"]

        interaction = (
            (c_abl["mean_err_neutral"] - c_cond["mean_err_neutral"])
            - (c_abl["mean_err_goal"]  - c_cond["mean_err_goal"])
        )
        neutral_diff = abs(c_cond["mean_err_neutral"] - c_abl["mean_err_neutral"])

        c1 = interaction >= THRESH_INTERACTION
        c2 = c_cond["mean_err_goal"] < c_abl["mean_err_goal"]
        c3 = neutral_diff < THRESH_NEUTRAL_DIFF
        c4 = c_cond["goal_norm"] > THRESH_GOAL_NORM

        print(
            f"  [EXQ-229] seed={seed}"
            f" interaction={interaction:.5f}"
            f" neutral_diff={neutral_diff:.5f}"
            f" goal_norm_cond={c_cond['goal_norm']:.4f}"
            f" C1={c1} C2={c2} C3={c3} C4={c4}",
            flush=True,
        )

        seed_metrics.append({
            "seed":           seed,
            "interaction":    interaction,
            "neutral_diff":   neutral_diff,
            "err_goal_cond":  c_cond["mean_err_goal"],
            "err_goal_abl":   c_abl["mean_err_goal"],
            "err_neut_cond":  c_cond["mean_err_neutral"],
            "err_neut_abl":   c_abl["mean_err_neutral"],
            "goal_norm_cond": c_cond["goal_norm"],
            "c1": c1,
            "c2": c2,
            "c3": c3,
            "c4": c4,
        })

    # -----------------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------------
    n_seeds  = len(seed_metrics)
    c1_count = sum(1 for m in seed_metrics if m["c1"])
    c2_count = sum(1 for m in seed_metrics if m["c2"])
    c3_count = sum(1 for m in seed_metrics if m["c3"])
    c4_count = sum(1 for m in seed_metrics if m["c4"])

    c1_pass = c1_count >= 2
    c2_pass = c2_count >= 2
    c3_pass = c3_count >= 2
    c4_pass = c4_count == n_seeds   # ALL seeds must show z_goal active

    if not c4_pass:
        outcome   = "FAIL"
        direction = "non_contributory"
        decision  = "substrate_limitation"
    elif c1_pass and c2_pass and c3_pass and c4_pass:
        outcome   = "PASS"
        direction = "supports"
        decision  = "retain_ree"
    elif c1_pass and c2_pass and not c3_pass:
        outcome   = "PARTIAL"
        direction = "mixed"
        decision  = "hybridize"
    else:
        outcome   = "FAIL"
        direction = "does_not_support"
        decision  = "retire_ree_claim"

    def _mean_sm(key: str) -> float:
        return sum(m[key] for m in seed_metrics) / max(1, len(seed_metrics))

    print(
        f"\n[EXQ-229] RESULT: {outcome}"
        f" interaction={_mean_sm('interaction'):.5f}"
        f" goal_norm={_mean_sm('goal_norm_cond'):.4f}"
        f" C1={c1_pass} C2={c2_pass} C3={c3_pass} C4={c4_pass}",
        flush=True,
    )

    if args.dry_run:
        print("\n[DRY-RUN] Smoke test complete. Not writing output.", flush=True)
        return

    manifest = {
        "run_id":                     f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type":            EXPERIMENT_TYPE,
        "architecture_epoch":         "ree_hybrid_guardrails_v1",
        "claim_ids":                  CLAIM_IDS,
        "experiment_purpose":         EXPERIMENT_PURPOSE,
        "outcome":                    outcome,
        "evidence_direction":         direction,
        "decision":                   decision,
        "timestamp":                  ts,
        "dry_run":                    args.dry_run,
        "seeds":                      SEEDS,
        "warmup_episodes":            warmup,
        "eval_episodes":              n_eval,
        "steps_per_episode":          steps,
        "world_dim":                  WORLD_DIM,
        "drive_weight":               2.0,
        "thresh_interaction":         THRESH_INTERACTION,
        "thresh_neutral_diff":        THRESH_NEUTRAL_DIFF,
        "thresh_goal_norm":           THRESH_GOAL_NORM,
        # Aggregate metrics
        "mean_interaction":           _mean_sm("interaction"),
        "mean_goal_norm_conditioned": _mean_sm("goal_norm_cond"),
        "mean_err_goal_conditioned":  _mean_sm("err_goal_cond"),
        "mean_err_goal_ablated":      _mean_sm("err_goal_abl"),
        # Criteria
        "c1_interaction_pass":        c1_pass,
        "c2_goal_direction_pass":     c2_pass,
        "c3_neutral_specificity_pass": c3_pass,
        "c4_goal_norm_pass":          c4_pass,
        "c1_count":                   c1_count,
        "c2_count":                   c2_count,
        "c3_count":                   c3_count,
        "c4_count":                   c4_count,
        "n_seeds":                    n_seeds,
        "seed_metrics":               seed_metrics,
        "all_condition_results":      all_results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[EXQ-229] Written: {out_path}", flush=True)


if __name__ == "__main__":
    main()
