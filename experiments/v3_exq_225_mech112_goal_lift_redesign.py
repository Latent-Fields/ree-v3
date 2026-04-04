#!/opt/local/bin/python3
"""
V3-EXQ-225 -- MECH-112: Goal-Lift Redesign with SD-012

Claims: MECH-112
EXPERIMENT_PURPOSE = "evidence"

Scientific question: With SD-012 (drive_weight=2.0) and proper harm-penalized
trajectory scoring, does an active z_goal (GOAL_PRESENT) produce genuine resource
lift without excess harm, compared to a goal-ablated control (GOAL_ABSENT)?

Context:
  EXQ-189 FAIL: z_goal seeded at 0.30 but harm_ratio=1.48 (goal signal increased
  risk-taking). EXQ-182a PASS (oracle: 11x lift with perfect signal). SD-012 now
  implemented (drive_weight=2.0). Key fix over EXQ-189: trajectory scoring must
  weight harm at least as much as goal to prevent harmful exploration.
  harm_weight=1.0, goal_weight=0.5 (harm penalised twice as strongly as goal rewarded).

Design:
  - GOAL_PRESENT: z_goal active (drive_weight=2.0), CEM-style trajectory scoring:
      score = harm_weight * harm_eval(z_next) - benefit_weight * benefit_eval(z_next)
                                               - goal_weight * goal_proximity(z_next)
    with harm_weight=1.0, benefit_weight=0.5, goal_weight=0.5.
  - GOAL_ABSENT: same architecture, z_goal=False, benefit_eval disabled,
    action selection: score = harm_eval(z_next) only.
  - 3 seeds x 300 warmup eps + 300 eval eps x 200 steps.
  - Warmup: 50% proximity-greedy (to seed z_goal).

Pre-registered PASS criteria (ALL required):
  C1: resource_lift >= 0.03
      (goal_present_resource_rate - goal_absent_resource_rate >= 0.03)
  C2: harm_ratio <= 1.3
      (goal_present_harm_rate / goal_absent_harm_rate <= 1.3)
  C3: avg_goal_norm >= 0.05
      (z_goal actually seeds -- non-trivial representation confirmed)

Decision:
  ALL PASS -> retain_ree (MECH-112 supported)
  C3 FAIL  -> substrate_limitation (z_goal not seeding)
  C1 FAIL, C3 PASS -> hybridize (goal seeds but no lift)
  C2 FAIL  -> retire_ree_claim (goal causes harmful exploration)
"""

import sys
import random
import json
import time
from pathlib import Path
from typing import Dict, List

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
EXPERIMENT_TYPE = "v3_exq_225_mech112_goal_lift_redesign"
CLAIM_IDS = ["MECH-112"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
RESOURCE_LIFT_THRESH = 0.03   # C1: GOAL_PRESENT resource_rate - GOAL_ABSENT >= this
HARM_RATIO_MAX       = 1.3    # C2: GOAL_PRESENT harm_rate / GOAL_ABSENT <= this
GOAL_NORM_THRESH     = 0.05   # C3: z_goal norm must exceed this

# ---------------------------------------------------------------------------
# Action-selection weights (harm penalised harder than goal rewarded)
# ---------------------------------------------------------------------------
HARM_WEIGHT    = 1.0
BENEFIT_WEIGHT = 0.5
GOAL_WEIGHT    = 0.5

# ---------------------------------------------------------------------------
# Episode settings
# ---------------------------------------------------------------------------
GRID_SIZE        = 8
WARMUP_EPISODES  = 300
EVAL_EPISODES    = 300
STEPS_PER_EP     = 200
SEEDS            = [42, 7, 13]
GREEDY_FRAC      = 0.5
MAX_BUF          = 4000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _manhattan_to_nearest_resource(env) -> int:
    if not env.resources:
        return 999
    ax, ay = env.agent_x, env.agent_y
    return min(abs(ax - int(r[0])) + abs(ay - int(r[1])) for r in env.resources)


def _greedy_toward_resource(env) -> int:
    ax, ay = env.agent_x, env.agent_y
    if not env.resources:
        return random.randint(0, env.action_dim - 1)
    best_d = float("inf")
    nearest = None
    for r in env.resources:
        rx, ry = int(r[0]), int(r[1])
        d = abs(ax - rx) + abs(ay - ry)
        if d < best_d:
            best_d = d
            nearest = (rx, ry)
    if nearest is None or best_d == 0:
        return random.randint(0, env.action_dim - 1)
    rx, ry = nearest
    dx, dy = rx - ax, ry - ay
    if abs(dx) >= abs(dy):
        return 1 if dx > 0 else 0
    else:
        return 3 if dy > 0 else 2


def _select_goal_present(agent: REEAgent, z_world: torch.Tensor, n_actions: int) -> int:
    """Score = harm_weight*harm - benefit_weight*benefit - goal_weight*goal_prox."""
    with torch.no_grad():
        best_idx = 0
        best_score = float("inf")
        for idx in range(n_actions):
            a_oh = _onehot(idx, n_actions, z_world.device)
            z_next = agent.e2.world_forward(z_world, a_oh)
            harm  = agent.e3.harm_eval(z_next).mean().item()
            bene  = agent.e3.benefit_eval(z_next).mean().item()
            gprox = agent.goal_state.goal_proximity(z_next).mean().item()
            score = HARM_WEIGHT * harm - BENEFIT_WEIGHT * bene - GOAL_WEIGHT * gprox
            if score < best_score:
                best_score = score
                best_idx   = idx
    return best_idx


def _select_goal_absent(agent: REEAgent, z_world: torch.Tensor, n_actions: int) -> int:
    """Score = harm_eval(z_next) only."""
    with torch.no_grad():
        best_idx = 0
        best_score = float("inf")
        for idx in range(n_actions):
            a_oh = _onehot(idx, n_actions, z_world.device)
            z_next = agent.e2.world_forward(z_world, a_oh)
            score  = agent.e3.harm_eval(z_next).mean().item()
            if score < best_score:
                best_score = score
                best_idx   = idx
    return best_idx


# ---------------------------------------------------------------------------
# Single seed run
# ---------------------------------------------------------------------------

def _run_seed(
    seed: int,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
) -> Dict:
    results = {}

    for condition in ["GOAL_PRESENT", "GOAL_ABSENT"]:
        torch.manual_seed(seed)
        random.seed(seed)

        goal_present = (condition == "GOAL_PRESENT")

        env = CausalGridWorldV2(
            seed=seed,
            size=GRID_SIZE,
            num_hazards=3,
            num_resources=4,
            hazard_harm=0.02,
            env_drift_interval=5,
            env_drift_prob=0.1,
            proximity_harm_scale=0.3,
            proximity_benefit_scale=0.18,
            proximity_approach_threshold=0.15,
            hazard_field_decay=0.5,
            energy_decay=0.005,
            use_proxy_fields=True,
            resource_respawn_on_consume=True,
        )
        n_actions = env.action_dim

        config = REEConfig.from_dims(
            body_obs_dim=env.body_obs_dim,
            world_obs_dim=env.world_obs_dim,
            action_dim=n_actions,
            self_dim=16,
            world_dim=32,
            alpha_world=0.9,
            alpha_self=0.3,
            reafference_action_dim=0,
            novelty_bonus_weight=0.0,
            benefit_eval_enabled=goal_present,
            benefit_weight=BENEFIT_WEIGHT if goal_present else 0.0,
            z_goal_enabled=goal_present,
            e1_goal_conditioned=goal_present,
            goal_weight=GOAL_WEIGHT if goal_present else 0.0,
            drive_weight=2.0 if goal_present else 0.0,
        )
        agent = REEAgent(config)

        std_params = [
            p for n, p in agent.named_parameters()
            if "harm_eval_head" not in n and "benefit_eval_head" not in n
        ]
        optimizer      = optim.Adam(std_params, lr=1e-3)
        harm_eval_opt  = optim.Adam(agent.e3.harm_eval_head.parameters(), lr=1e-4)
        benefit_eval_opt = (
            optim.Adam(agent.e3.benefit_eval_head.parameters(), lr=1e-3)
            if goal_present else None
        )

        harm_buf_pos: List[torch.Tensor] = []
        harm_buf_neg: List[torch.Tensor] = []
        benefit_buf_zw:  List[torch.Tensor] = []
        benefit_buf_lbl: List[float]        = []

        agent.train()

        # ---- WARMUP ----
        for ep in range(warmup_episodes):
            _, obs_dict = env.reset()
            agent.reset()

            for step_i in range(steps_per_episode):
                obs_body  = obs_dict["body_state"]
                obs_world = obs_dict["world_state"]
                obs_harm  = obs_dict.get("harm_obs", None)

                latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
                agent.clock.advance()
                z_world_curr = latent.z_world.detach()

                # Mixed warmup policy
                if goal_present and random.random() < GREEDY_FRAC:
                    action_idx = _greedy_toward_resource(env)
                else:
                    action_idx = random.randint(0, n_actions - 1)
                action_oh = _onehot(action_idx, n_actions, agent.device)
                agent._last_action = action_oh

                dist   = _manhattan_to_nearest_resource(env)
                is_near = 1.0 if dist <= 2 else 0.0

                _, harm_signal, done, info, obs_dict = env.step(action_oh)

                # benefit_exposure for z_goal
                b_exp = 0.0
                if obs_body.dim() == 1 and obs_body.shape[0] > 11:
                    b_exp = float(obs_body[11].item())
                elif obs_body.dim() > 1 and obs_body.shape[-1] > 11:
                    b_exp = float(obs_body[0, 11].item())

                # Standard agent training
                e1_loss = agent.compute_prediction_loss()
                e2_loss = agent.compute_e2_loss()
                total   = e1_loss + e2_loss
                if total.requires_grad:
                    optimizer.zero_grad()
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                    optimizer.step()

                # harm_eval training (stratified)
                if float(harm_signal) < 0:
                    harm_buf_pos.append(z_world_curr)
                    if len(harm_buf_pos) > MAX_BUF:
                        harm_buf_pos = harm_buf_pos[-MAX_BUF:]
                else:
                    harm_buf_neg.append(z_world_curr)
                    if len(harm_buf_neg) > MAX_BUF:
                        harm_buf_neg = harm_buf_neg[-MAX_BUF:]

                if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                    k_p = min(16, len(harm_buf_pos))
                    k_n = min(16, len(harm_buf_neg))
                    pi = torch.randperm(len(harm_buf_pos))[:k_p].tolist()
                    ni = torch.randperm(len(harm_buf_neg))[:k_n].tolist()
                    zw_b = torch.cat(
                        [harm_buf_pos[i] for i in pi] +
                        [harm_buf_neg[i] for i in ni], dim=0
                    )
                    tgt = torch.cat([
                        torch.ones(k_p, 1, device=agent.device),
                        torch.zeros(k_n, 1, device=agent.device),
                    ], dim=0)
                    pred = agent.e3.harm_eval(zw_b)
                    hloss = F.binary_cross_entropy(pred, tgt)
                    if hloss.requires_grad:
                        harm_eval_opt.zero_grad()
                        hloss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            agent.e3.harm_eval_head.parameters(), 0.5
                        )
                        harm_eval_opt.step()

                # benefit_eval training (GOAL_PRESENT only)
                if goal_present:
                    benefit_buf_zw.append(z_world_curr)
                    benefit_buf_lbl.append(is_near)
                    if len(benefit_buf_zw) > MAX_BUF:
                        benefit_buf_zw  = benefit_buf_zw[-MAX_BUF:]
                        benefit_buf_lbl = benefit_buf_lbl[-MAX_BUF:]

                    if len(benefit_buf_zw) >= 32 and step_i % 4 == 0:
                        k = min(32, len(benefit_buf_zw))
                        idxs = random.sample(range(len(benefit_buf_zw)), k)
                        zw_b = torch.cat([benefit_buf_zw[i] for i in idxs], dim=0)
                        lbl  = torch.tensor(
                            [benefit_buf_lbl[i] for i in idxs],
                            dtype=torch.float32
                        ).unsqueeze(1).to(agent.device)
                        pred_b = agent.e3.benefit_eval(zw_b)
                        bloss  = F.binary_cross_entropy(pred_b, lbl)
                        if bloss.requires_grad:
                            benefit_eval_opt.zero_grad()
                            bloss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                agent.e3.benefit_eval_head.parameters(), 0.5
                            )
                            benefit_eval_opt.step()
                            agent.e3.record_benefit_sample(k)

                    # z_goal update
                    agent.update_z_goal(b_exp)

                if done:
                    break

            if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
                diag = agent.compute_goal_maintenance_diagnostic()
                print(
                    f"  [warmup] seed={seed} cond={condition}"
                    f" ep {ep+1}/{warmup_episodes}"
                    f" harm_pos={len(harm_buf_pos)} harm_neg={len(harm_buf_neg)}"
                    f" goal_norm={diag['goal_norm']:.3f}",
                    flush=True,
                )

        # Goal norm at end of warmup
        diag_final  = agent.compute_goal_maintenance_diagnostic()
        goal_norm_f = float(diag_final["goal_norm"])

        # Check substrate limitation: z_goal must seed before eval is meaningful
        substrate_ok = (not goal_present) or (goal_norm_f >= GOAL_NORM_THRESH)

        # ---- EVAL ----
        agent.eval()

        resource_counts: List[int]  = []
        harm_rates:       List[float] = []

        for _ in range(eval_episodes):
            _, obs_dict = env.reset()
            agent.reset()
            ep_resources = 0
            ep_harm_sum  = 0.0
            ep_steps     = 0

            for _ in range(steps_per_episode):
                obs_body  = obs_dict["body_state"]
                obs_world = obs_dict["world_state"]
                obs_harm  = obs_dict.get("harm_obs", None)

                with torch.no_grad():
                    latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
                    agent.clock.advance()
                    z_world_curr = latent.z_world.detach()

                if goal_present:
                    action_idx = _select_goal_present(agent, z_world_curr, n_actions)
                else:
                    action_idx = _select_goal_absent(agent, z_world_curr, n_actions)

                action_oh = _onehot(action_idx, n_actions, agent.device)
                agent._last_action = action_oh

                _, harm_signal, done, info, obs_dict = env.step(action_oh)
                ttype = info.get("transition_type", "none")
                if ttype == "resource":
                    ep_resources += 1
                if float(harm_signal) < 0:
                    ep_harm_sum += abs(float(harm_signal))
                ep_steps += 1
                if done:
                    break

            resource_counts.append(1 if ep_resources >= 1 else 0)
            harm_rates.append(ep_harm_sum / max(1, ep_steps))

        resource_rate = float(sum(resource_counts)) / max(1, len(resource_counts))
        harm_rate     = float(sum(harm_rates)) / max(1, len(harm_rates))

        print(
            f"  [eval] seed={seed} cond={condition}"
            f" resource_rate={resource_rate:.3f}"
            f" harm_rate={harm_rate:.5f}"
            f" goal_norm={goal_norm_f:.3f}"
            f" substrate_ok={substrate_ok}",
            flush=True,
        )

        results[condition] = {
            "resource_rate": resource_rate,
            "harm_rate":     harm_rate,
            "goal_norm":     goal_norm_f,
            "substrate_ok":  substrate_ok,
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    warmup = 2       if args.dry_run else WARMUP_EPISODES
    n_eval = 2       if args.dry_run else EVAL_EPISODES
    steps  = 20      if args.dry_run else STEPS_PER_EP
    seeds  = [42]    if args.dry_run else SEEDS

    print(
        f"[V3-EXQ-225] MECH-112 Goal-Lift Redesign (SD-012)"
        f" dry_run={args.dry_run}"
        f" harm_weight={HARM_WEIGHT} benefit_weight={BENEFIT_WEIGHT}"
        f" goal_weight={GOAL_WEIGHT}",
        flush=True,
    )

    all_results = []
    for seed in seeds:
        print(f"\n[V3-EXQ-225] seed={seed}", flush=True)
        res = _run_seed(
            seed=seed,
            warmup_episodes=warmup,
            eval_episodes=n_eval,
            steps_per_episode=steps,
        )
        all_results.append({"seed": seed, "conditions": res})

    # ---- Aggregate ----
    def _avg(key: str, cond: str) -> float:
        return sum(r["conditions"][cond][key] for r in all_results) / max(1, len(all_results))

    gp_resource  = _avg("resource_rate", "GOAL_PRESENT")
    ga_resource  = _avg("resource_rate", "GOAL_ABSENT")
    gp_harm      = _avg("harm_rate",     "GOAL_PRESENT")
    ga_harm      = _avg("harm_rate",     "GOAL_ABSENT")
    avg_goal_norm = _avg("goal_norm",    "GOAL_PRESENT")

    resource_lift = gp_resource - ga_resource
    harm_ratio    = gp_harm / max(1e-9, ga_harm)

    substrate_ok_all = all(
        r["conditions"]["GOAL_PRESENT"]["substrate_ok"] for r in all_results
    )

    c1_pass = resource_lift >= RESOURCE_LIFT_THRESH
    c2_pass = harm_ratio   <= HARM_RATIO_MAX
    c3_pass = avg_goal_norm >= GOAL_NORM_THRESH

    all_pass     = c1_pass and c2_pass and c3_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass])

    if all_pass:
        outcome   = "PASS"
        direction = "supports"
        decision  = "retain_ree"
    elif not c3_pass:
        outcome   = "FAIL"
        direction = "non_contributory"
        decision  = "substrate_limitation"
    elif c3_pass and not c1_pass:
        outcome   = "FAIL"
        direction = "does_not_support"
        decision  = "hybridize"
    else:
        outcome   = "FAIL"
        direction = "weakens"
        decision  = "retire_ree_claim"

    print(f"\n[V3-EXQ-225] Results:", flush=True)
    print(
        f"  GOAL_PRESENT: resource_rate={gp_resource:.3f} harm_rate={gp_harm:.5f}",
        flush=True,
    )
    print(
        f"  GOAL_ABSENT:  resource_rate={ga_resource:.3f} harm_rate={ga_harm:.5f}",
        flush=True,
    )
    print(
        f"  resource_lift={resource_lift:.4f}  harm_ratio={harm_ratio:.3f}"
        f"  avg_goal_norm={avg_goal_norm:.4f}",
        flush=True,
    )
    print(
        f"  C1(lift>={RESOURCE_LIFT_THRESH}): {'PASS' if c1_pass else 'FAIL'}"
        f"  C2(ratio<={HARM_RATIO_MAX}): {'PASS' if c2_pass else 'FAIL'}"
        f"  C3(norm>={GOAL_NORM_THRESH}): {'PASS' if c3_pass else 'FAIL'}",
        flush=True,
    )
    print(f"  -> {outcome} ({criteria_met}/3) decision={decision}", flush=True)

    if args.dry_run:
        print("\n[DRY-RUN] Smoke test complete. Not writing output.", flush=True)
        return

    ts      = int(time.time())
    root    = Path(__file__).resolve().parents[2]
    out_dir = root / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"

    manifest = {
        "run_id":               f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type":      EXPERIMENT_TYPE,
        "architecture_epoch":   "ree_hybrid_guardrails_v1",
        "claim_ids":            CLAIM_IDS,
        "experiment_purpose":   EXPERIMENT_PURPOSE,
        "outcome":              outcome,
        "evidence_direction":   direction,
        "decision":             decision,
        "timestamp":            ts,
        "seeds":                seeds,
        "warmup_episodes":      warmup,
        "eval_episodes":        n_eval,
        "steps_per_episode":    steps,
        "harm_weight":          HARM_WEIGHT,
        "benefit_weight":       BENEFIT_WEIGHT,
        "goal_weight":          GOAL_WEIGHT,
        "resource_lift_thresh": RESOURCE_LIFT_THRESH,
        "harm_ratio_max":       HARM_RATIO_MAX,
        "goal_norm_thresh":     GOAL_NORM_THRESH,
        # Metrics
        "gp_resource_rate":     float(gp_resource),
        "ga_resource_rate":     float(ga_resource),
        "gp_harm_rate":         float(gp_harm),
        "ga_harm_rate":         float(ga_harm),
        "resource_lift":        float(resource_lift),
        "harm_ratio":           float(harm_ratio),
        "avg_goal_norm":        float(avg_goal_norm),
        "substrate_ok_all":     substrate_ok_all,
        # Criteria
        "c1_resource_lift_pass": c1_pass,
        "c2_harm_ratio_pass":    c2_pass,
        "c3_goal_norm_pass":     c3_pass,
        "criteria_met":          criteria_met,
        # Per-seed
        "per_seed_results": [
            {
                "seed":               r["seed"],
                "gp_resource_rate":   r["conditions"]["GOAL_PRESENT"]["resource_rate"],
                "ga_resource_rate":   r["conditions"]["GOAL_ABSENT"]["resource_rate"],
                "gp_harm_rate":       r["conditions"]["GOAL_PRESENT"]["harm_rate"],
                "ga_harm_rate":       r["conditions"]["GOAL_ABSENT"]["harm_rate"],
                "goal_norm":          r["conditions"]["GOAL_PRESENT"]["goal_norm"],
            }
            for r in all_results
        ],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[V3-EXQ-225] Written: {out_path}", flush=True)
    print(f"Status: {outcome}", flush=True)


if __name__ == "__main__":
    main()
