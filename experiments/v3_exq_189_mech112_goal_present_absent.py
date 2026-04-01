#!/opt/local/bin/python3
"""
V3-EXQ-189 -- MECH-112 Goal Present vs Absent (Discriminative Pair)

Claims: MECH-112

Scientific question: Does a learned goal representation (z_goal seeded,
benefit_eval enabled) produce genuine goal-directed behavior that
outperforms a goal-ablated matched control?

Dispatch mode: discriminative_pair
Priority: high

Design:
  - Condition A (GOAL_PRESENT): z_goal_enabled=True, benefit_eval_enabled=True,
    e1_goal_conditioned=True. Current best learned goal pathway active.
  - Condition B (GOAL_ABSENT): z_goal_enabled=False, benefit_eval_enabled=False.
    Goal pathway fully ablated. Matched architecture otherwise.
  - Both conditions use 50% greedy proximity navigation during warmup to
    ensure z_goal seeds (learned from EXQ-074e fix).
  - Eval phase: score-based action selection for GOAL_PRESENT (harm_eval
    minus benefit_eval minus goal_proximity), harm-only for GOAL_ABSENT.
  - 3 matched seeds: [42, 7, 13]
  - Warmup: 300 episodes (random + proximity-greedy). Trains E1/E2/E3 and
    harm_eval/benefit_eval heads. z_goal updated each step from benefit_exposure.
  - Eval: 100 episodes per condition per seed.
  - Grid: CausalGridWorldV2 size=8, 4 resources (respawning), 3 hazards.

Pre-registered PASS criteria (ALL required):
  C1: goal_present_resource_rate >= goal_absent_resource_rate + 0.03
      (GOAL_PRESENT collects more resources -- behavioral lift)
  C2: goal_present_mean_benefit >= goal_absent_mean_benefit * 1.15
      (GOAL_PRESENT accumulates 15% more benefit exposure)
  C3: goal_present_harm_rate <= goal_absent_harm_rate * 1.3
      (harm does not regress catastrophically -- at most 30% increase)
  C4: avg_goal_norm_final >= 0.05
      (z_goal must actually seed -- non-trivial representation)

Decision scoring:
  ALL PASS -> retain_ree (MECH-112 supported by learned signal)
  C4 FAIL -> retire_ree_claim (z_goal never seeds -- learned rep is broken)
  C1 or C2 FAIL, C4 PASS -> hybridize (goal seeds but no behavioral lift --
    learned representation is bottleneck, compare with EXP-0091 handcrafted)
  else -> inconclusive

Diagnostic relationship to EXP-0091 (handcrafted goal):
  If EXQ-189 FAILS and EXP-0091 PASSES -> learned representation is bottleneck
  If both FAIL -> selector/integration path is bottleneck
  If both PASS -> learned goal is sufficient
"""

import sys
import random
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_189_mech112_goal_present_absent"
CLAIM_IDS = ["MECH-112"]

GRID_SIZE = 8
GREEDY_FRAC = 0.5  # fraction of warmup steps that use proximity-following


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _manhattan_dist_to_nearest_resource(env) -> int:
    """Return manhattan distance from agent to nearest resource, or 999 if none."""
    if not env.resources:
        return 999
    ax, ay = env.agent_x, env.agent_y
    return min(abs(ax - int(r[0])) + abs(ay - int(r[1])) for r in env.resources)


def _greedy_action_toward_resource(env) -> int:
    """Return action index that reduces L1 distance to nearest resource.
    CausalGridWorldV2 actions: 0=up(-1,0), 1=down(+1,0), 2=left(0,-1),
                               3=right(0,+1), 4=stay(0,0).
    Falls back to random if no resources or already at resource.
    """
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
    dx = rx - ax
    dy = ry - ay
    if abs(dx) >= abs(dy):
        return 1 if dx > 0 else 0   # down or up
    else:
        return 3 if dy > 0 else 2   # right or left


def _compute_auc(scores: List[float], labels: List[float]) -> float:
    """Compute AUC-ROC from predicted scores and binary labels."""
    n = len(scores)
    if n < 4:
        return 0.5
    paired = list(zip(scores, labels))
    paired.sort(key=lambda x: -x[0])
    tp = 0.0
    fp = 0.0
    total_pos = sum(labels)
    total_neg = n - total_pos
    if total_pos < 1 or total_neg < 1:
        return 0.5
    auc = 0.0
    prev_fp = 0.0
    prev_tp = 0.0
    for s, lab in paired:
        if lab > 0.5:
            tp += 1.0
        else:
            fp += 1.0
        auc += 0.5 * (fp - prev_fp) * (tp + prev_tp)
        prev_fp = fp
        prev_tp = tp
    auc /= (total_pos * total_neg)
    return float(auc)


# ------------------------------------------------------------------ #
# Score-based action selection                                         #
# ------------------------------------------------------------------ #

def _select_action_goal_present(
    agent: REEAgent,
    z_world: torch.Tensor,
    num_actions: int,
    benefit_weight: float = 1.0,
    goal_weight: float = 1.0,
) -> int:
    """
    GOAL_PRESENT: action_score = harm_eval(z_next)
                                 - benefit_weight * benefit_eval(z_next)
                                 - goal_weight * goal_proximity(z_next).
    Pick action with lowest combined score (low harm, high benefit, close to goal).
    """
    with torch.no_grad():
        best_action = 0
        best_score = float("inf")
        for idx in range(num_actions):
            a_oh = _action_to_onehot(idx, num_actions, z_world.device)
            z_world_next = agent.e2.world_forward(z_world, a_oh)
            harm_score = agent.e3.harm_eval(z_world_next).mean().item()
            benefit_score = agent.e3.benefit_eval(z_world_next).mean().item()
            goal_prox = agent.goal_state.goal_proximity(z_world_next).mean().item()
            score = harm_score - benefit_weight * benefit_score - goal_weight * goal_prox
            if score < best_score:
                best_score = score
                best_action = idx
    return best_action


def _select_action_goal_absent(
    agent: REEAgent,
    z_world: torch.Tensor,
    num_actions: int,
) -> int:
    """
    GOAL_ABSENT: action_score = harm_eval(z_world_next).
    Pick action with lowest harm score (no goal or benefit signal).
    """
    with torch.no_grad():
        best_action = 0
        best_score = float("inf")
        for idx in range(num_actions):
            a_oh = _action_to_onehot(idx, num_actions, z_world.device)
            z_world_next = agent.e2.world_forward(z_world, a_oh)
            harm_score = agent.e3.harm_eval(z_world_next).mean().item()
            if harm_score < best_score:
                best_score = harm_score
                best_action = idx
    return best_action


# ------------------------------------------------------------------ #
# Make env + agent                                                     #
# ------------------------------------------------------------------ #

def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
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


def _make_agent(
    condition: str,
    env: CausalGridWorldV2,
    self_dim: int = 16,
    world_dim: int = 32,
) -> REEAgent:
    """Create agent with goal pathway enabled (GOAL_PRESENT) or disabled (GOAL_ABSENT)."""
    goal_present = (condition == "GOAL_PRESENT")
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=0,
        novelty_bonus_weight=0.0,
        # Goal pathway
        benefit_eval_enabled=goal_present,
        benefit_weight=1.0 if goal_present else 0.0,
        z_goal_enabled=goal_present,
        e1_goal_conditioned=goal_present,
        goal_weight=1.0 if goal_present else 0.0,
    )
    return REEAgent(config)


# ------------------------------------------------------------------ #
# Single-seed run                                                      #
# ------------------------------------------------------------------ #

def _run_single(
    seed: int,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    benefit_weight: float,
    goal_weight: float,
) -> Dict:
    """Run both conditions (GOAL_PRESENT, GOAL_ABSENT) for one seed."""

    conditions = ["GOAL_PRESENT", "GOAL_ABSENT"]
    seed_results: Dict[str, Dict] = {}

    for cond in conditions:
        torch.manual_seed(seed)
        random.seed(seed)

        env = _make_env(seed)
        agent = _make_agent(cond, env, self_dim, world_dim)
        action_dim = env.action_dim
        device = agent.device

        # Separate optimizer groups
        standard_params = [
            p for n, p in agent.named_parameters()
            if "harm_eval_head" not in n and "benefit_eval_head" not in n
        ]
        harm_eval_params = list(agent.e3.harm_eval_head.parameters())
        optimizer = optim.Adam(standard_params, lr=1e-3)
        harm_eval_opt = optim.Adam(harm_eval_params, lr=1e-4)

        benefit_eval_opt = None
        if cond == "GOAL_PRESENT":
            benefit_eval_params = list(agent.e3.benefit_eval_head.parameters())
            benefit_eval_opt = optim.Adam(benefit_eval_params, lr=1e-3)

        # Replay buffers for harm_eval
        harm_buf_pos: List[torch.Tensor] = []
        harm_buf_neg: List[torch.Tensor] = []
        # Replay buffers for benefit_eval
        benefit_buf_zw: List[torch.Tensor] = []
        benefit_buf_lbl: List[float] = []
        MAX_BUF = 4000

        # AUC evaluation buffer (last 50 warmup episodes)
        auc_eval_scores: List[float] = []
        auc_eval_labels: List[float] = []

        agent.train()

        # ---- WARMUP ----
        for ep in range(warmup_episodes):
            _, obs_dict = env.reset()
            agent.reset()
            collect_auc = (ep >= warmup_episodes - 50)

            for step_i in range(steps_per_episode):
                obs_body = obs_dict["body_state"]
                obs_world = obs_dict["world_state"]
                obs_harm = obs_dict.get("harm_obs", None)

                latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
                agent.clock.advance()
                z_world_curr = latent.z_world.detach()

                # Mixed warmup policy: proximity-greedy + random
                if random.random() < GREEDY_FRAC:
                    action_idx = _greedy_action_toward_resource(env)
                else:
                    action_idx = random.randint(0, action_dim - 1)
                action_oh = _action_to_onehot(action_idx, action_dim, device)
                agent._last_action = action_oh

                # Compute is_near_resource label BEFORE step
                dist = _manhattan_dist_to_nearest_resource(env)
                is_near = 1.0 if dist <= 2 else 0.0

                _, harm_signal, done, info, obs_dict = env.step(action_oh)

                # Get benefit_exposure from obs_body for z_goal update
                if obs_body.dim() == 1:
                    b_exp = float(obs_body[11].item()) if obs_body.shape[0] > 11 else 0.0
                else:
                    b_exp = float(obs_body[0, 11].item()) if obs_body.shape[-1] > 11 else 0.0

                # --- Standard agent training (E1 + E2) ---
                e1_loss = agent.compute_prediction_loss()
                e2_loss = agent.compute_e2_loss()
                total_loss = e1_loss + e2_loss
                if total_loss.requires_grad:
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                    optimizer.step()

                # --- harm_eval training (stratified) ---
                is_harm = float(harm_signal) < 0
                if is_harm:
                    harm_buf_pos.append(z_world_curr)
                    if len(harm_buf_pos) > MAX_BUF:
                        harm_buf_pos = harm_buf_pos[-MAX_BUF:]
                else:
                    harm_buf_neg.append(z_world_curr)
                    if len(harm_buf_neg) > MAX_BUF:
                        harm_buf_neg = harm_buf_neg[-MAX_BUF:]

                if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                    k_pos = min(16, len(harm_buf_pos))
                    k_neg = min(16, len(harm_buf_neg))
                    pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                    neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()
                    zw_pos = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
                    zw_neg = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)
                    zw_b = torch.cat([zw_pos, zw_neg], dim=0)
                    target = torch.cat([
                        torch.ones(k_pos, 1, device=device),
                        torch.zeros(k_neg, 1, device=device),
                    ], dim=0)
                    pred_harm = agent.e3.harm_eval(zw_b)
                    harm_loss = F.binary_cross_entropy(pred_harm, target)
                    if harm_loss.requires_grad:
                        harm_eval_opt.zero_grad()
                        harm_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            agent.e3.harm_eval_head.parameters(), 0.5,
                        )
                        harm_eval_opt.step()

                # --- benefit_eval training (BCE on is_near_resource) ---
                if cond == "GOAL_PRESENT":
                    benefit_buf_zw.append(z_world_curr)
                    benefit_buf_lbl.append(is_near)
                    if len(benefit_buf_zw) > MAX_BUF:
                        benefit_buf_zw = benefit_buf_zw[-MAX_BUF:]
                        benefit_buf_lbl = benefit_buf_lbl[-MAX_BUF:]

                    if len(benefit_buf_zw) >= 32 and step_i % 4 == 0:
                        k = min(32, len(benefit_buf_zw))
                        indices = random.sample(range(len(benefit_buf_zw)), k)
                        zw_batch = torch.cat(
                            [benefit_buf_zw[i] for i in indices], dim=0,
                        )
                        lbl_batch = torch.tensor(
                            [benefit_buf_lbl[i] for i in indices],
                            dtype=torch.float32,
                        ).unsqueeze(1).to(device)
                        pred_benefit = agent.e3.benefit_eval(zw_batch)
                        benefit_loss = F.binary_cross_entropy(pred_benefit, lbl_batch)
                        if benefit_loss.requires_grad:
                            benefit_eval_opt.zero_grad()
                            benefit_loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                agent.e3.benefit_eval_head.parameters(), 0.5,
                            )
                            benefit_eval_opt.step()
                            agent.e3.record_benefit_sample(k)

                    # Collect AUC eval data from last 50 warmup episodes
                    if collect_auc:
                        with torch.no_grad():
                            score = agent.e3.benefit_eval(z_world_curr).mean().item()
                        auc_eval_scores.append(score)
                        auc_eval_labels.append(is_near)

                # --- z_goal update (GOAL_PRESENT only) ---
                if cond == "GOAL_PRESENT":
                    agent.update_z_goal(b_exp)

                if done:
                    break

            if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
                diag = agent.compute_goal_maintenance_diagnostic()
                print(
                    f"  [warmup] seed={seed} cond={cond} ep {ep+1}/{warmup_episodes}"
                    f" harm_pos={len(harm_buf_pos)} harm_neg={len(harm_buf_neg)}"
                    f" goal_norm={diag['goal_norm']:.3f}"
                    f" goal_active={diag['is_active']}",
                    flush=True,
                )

        # Compute benefit_eval AUC (GOAL_PRESENT only)
        benefit_eval_auc = 0.5
        if cond == "GOAL_PRESENT" and auc_eval_scores:
            benefit_eval_auc = _compute_auc(auc_eval_scores, auc_eval_labels)
            print(
                f"  [warmup done] seed={seed} cond={cond}"
                f" benefit_eval_auc={benefit_eval_auc:.4f}"
                f" auc_n={len(auc_eval_scores)}",
                flush=True,
            )

        # Record goal_norm at end of warmup
        diag_final = agent.compute_goal_maintenance_diagnostic()
        goal_norm_final = diag_final["goal_norm"]
        goal_active_final = diag_final["is_active"]

        # ---- EVAL ----
        agent.eval()

        resource_counts: List[int] = []
        mean_benefits: List[float] = []
        harm_rates: List[float] = []

        for eval_ep in range(eval_episodes):
            _, obs_dict = env.reset()
            agent.reset()
            ep_resources = 0
            ep_benefit_sum = 0.0
            ep_harm_sum = 0.0
            ep_steps = 0

            for step_i in range(steps_per_episode):
                obs_body = obs_dict["body_state"]
                obs_world = obs_dict["world_state"]
                obs_harm = obs_dict.get("harm_obs", None)

                with torch.no_grad():
                    latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
                    agent.clock.advance()
                    z_world_curr = latent.z_world.detach()

                if cond == "GOAL_PRESENT":
                    action_idx = _select_action_goal_present(
                        agent, z_world_curr, action_dim,
                        benefit_weight=benefit_weight,
                        goal_weight=goal_weight,
                    )
                else:
                    action_idx = _select_action_goal_absent(
                        agent, z_world_curr, action_dim,
                    )

                action_oh = _action_to_onehot(action_idx, action_dim, device)
                agent._last_action = action_oh

                _, harm_signal, done, info, obs_dict = env.step(action_oh)
                ttype = info.get("transition_type", "none")

                if ttype == "resource":
                    ep_resources += 1

                # Accumulate benefit_exposure from obs
                if obs_body.dim() == 1:
                    b_exp = float(obs_body[11].item()) if obs_body.shape[0] > 11 else 0.0
                else:
                    b_exp = float(obs_body[0, 11].item()) if obs_body.shape[-1] > 11 else 0.0
                ep_benefit_sum += b_exp

                if float(harm_signal) < 0:
                    ep_harm_sum += abs(float(harm_signal))

                ep_steps += 1
                if done:
                    break

            resource_counts.append(ep_resources)
            mean_benefits.append(ep_benefit_sum / max(1, ep_steps))
            harm_rates.append(ep_harm_sum / max(1, ep_steps))

        resource_rate = float(sum(resource_counts)) / max(1, sum(
            max(1, 1) for _ in resource_counts
        ))
        # Compute as: fraction of eval episodes where at least 1 resource collected
        resource_collection_rate = float(
            sum(1 for c in resource_counts if c >= 1)
        ) / max(1, len(resource_counts))
        # Also compute mean resources per episode
        mean_resources_per_ep = float(sum(resource_counts)) / max(1, len(resource_counts))
        mean_benefit = float(sum(mean_benefits)) / max(1, len(mean_benefits))
        harm_rate = float(sum(harm_rates)) / max(1, len(harm_rates))

        print(
            f"  [eval] seed={seed} cond={cond}"
            f" resource_rate={resource_collection_rate:.3f}"
            f" mean_resources={mean_resources_per_ep:.2f}"
            f" mean_benefit={mean_benefit:.5f}"
            f" harm_rate={harm_rate:.5f}",
            flush=True,
        )

        seed_results[cond] = {
            "resource_collection_rate": resource_collection_rate,
            "mean_resources_per_ep": mean_resources_per_ep,
            "mean_benefit": mean_benefit,
            "harm_rate": harm_rate,
            "benefit_eval_auc": benefit_eval_auc,
            "goal_norm_final": goal_norm_final,
            "goal_active_final": goal_active_final,
        }

    return {
        "seed": seed,
        "conditions": seed_results,
    }


# ------------------------------------------------------------------ #
# Aggregation and output                                               #
# ------------------------------------------------------------------ #

def run(
    seeds: Tuple = (42, 7, 13),
    warmup_episodes: int = 300,
    eval_episodes: int = 100,
    steps_per_episode: int = 200,
    self_dim: int = 16,
    world_dim: int = 32,
    benefit_weight: float = 1.0,
    goal_weight: float = 1.0,
    **kwargs,
) -> dict:
    all_results: List[Dict] = []

    for seed in seeds:
        print(
            f"\n[V3-EXQ-189] seed={seed}"
            f" warmup={warmup_episodes} eval={eval_episodes}"
            f" steps={steps_per_episode}"
            f" benefit_weight={benefit_weight}"
            f" goal_weight={goal_weight}",
            flush=True,
        )
        r = _run_single(
            seed=seed,
            warmup_episodes=warmup_episodes,
            eval_episodes=eval_episodes,
            steps_per_episode=steps_per_episode,
            self_dim=self_dim,
            world_dim=world_dim,
            benefit_weight=benefit_weight,
            goal_weight=goal_weight,
        )
        all_results.append(r)

    # ---- Aggregate across seeds ----
    def _avg_cond(key: str, cond: str) -> float:
        vals = [r["conditions"][cond][key] for r in all_results]
        return float(sum(vals) / max(1, len(vals)))

    gp_resource_rate = _avg_cond("resource_collection_rate", "GOAL_PRESENT")
    gp_mean_resources = _avg_cond("mean_resources_per_ep", "GOAL_PRESENT")
    gp_mean_benefit = _avg_cond("mean_benefit", "GOAL_PRESENT")
    gp_harm_rate = _avg_cond("harm_rate", "GOAL_PRESENT")
    gp_benefit_auc = _avg_cond("benefit_eval_auc", "GOAL_PRESENT")
    gp_goal_norm = _avg_cond("goal_norm_final", "GOAL_PRESENT")

    ga_resource_rate = _avg_cond("resource_collection_rate", "GOAL_ABSENT")
    ga_mean_resources = _avg_cond("mean_resources_per_ep", "GOAL_ABSENT")
    ga_mean_benefit = _avg_cond("mean_benefit", "GOAL_ABSENT")
    ga_harm_rate = _avg_cond("harm_rate", "GOAL_ABSENT")

    # ---- PASS criteria ----
    resource_lift = gp_resource_rate - ga_resource_rate
    benefit_ratio = gp_mean_benefit / max(1e-8, ga_mean_benefit)
    harm_ratio = gp_harm_rate / max(1e-8, ga_harm_rate) if ga_harm_rate > 1e-8 else 0.0

    c1_pass = resource_lift >= 0.03
    c2_pass = benefit_ratio >= 1.15
    c3_pass = harm_ratio <= 1.3 if ga_harm_rate > 1e-8 else True
    c4_pass = gp_goal_norm >= 0.05

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    status = "PASS" if all_pass else "FAIL"

    # Decision scoring
    if all_pass:
        decision = "retain_ree"
    elif not c4_pass:
        decision = "retire_ree_claim"
    elif not (c1_pass and c2_pass):
        decision = "hybridize"
    else:
        decision = "inconclusive"

    print(f"\n[V3-EXQ-189] Final results:", flush=True)
    print(
        f"  GOAL_PRESENT: resource_rate={gp_resource_rate:.3f}"
        f" mean_resources={gp_mean_resources:.2f}"
        f" mean_benefit={gp_mean_benefit:.5f}"
        f" harm_rate={gp_harm_rate:.5f}",
        flush=True,
    )
    print(
        f"  GOAL_ABSENT:  resource_rate={ga_resource_rate:.3f}"
        f" mean_resources={ga_mean_resources:.2f}"
        f" mean_benefit={ga_mean_benefit:.5f}"
        f" harm_rate={ga_harm_rate:.5f}",
        flush=True,
    )
    print(
        f"  resource_lift={resource_lift:+.3f}"
        f" benefit_ratio={benefit_ratio:.2f}x"
        f" harm_ratio={harm_ratio:.2f}x"
        f" goal_norm={gp_goal_norm:.4f}"
        f" benefit_auc={gp_benefit_auc:.4f}",
        flush=True,
    )
    print(
        f"  decision={decision}  status={status} ({criteria_met}/4)",
        flush=True,
    )

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: resource_lift={resource_lift:+.3f} < 0.03."
            " GOAL_PRESENT does not collect more resources."
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: benefit_ratio={benefit_ratio:.2f}x < 1.15x."
            " GOAL_PRESENT does not accumulate more benefit."
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: harm_ratio={harm_ratio:.2f}x > 1.3x."
            " Goal pursuit causes catastrophic harm regression."
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: goal_norm={gp_goal_norm:.4f} < 0.05."
            " z_goal never seeds -- learned representation is broken."
        )
    for note in failure_notes:
        print(f"  {note}", flush=True)

    per_seed_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" gp_res={r['conditions']['GOAL_PRESENT']['resource_collection_rate']:.3f}"
        f" ga_res={r['conditions']['GOAL_ABSENT']['resource_collection_rate']:.3f}"
        f" gp_ben={r['conditions']['GOAL_PRESENT']['mean_benefit']:.5f}"
        f" ga_ben={r['conditions']['GOAL_ABSENT']['mean_benefit']:.5f}"
        f" goal_norm={r['conditions']['GOAL_PRESENT']['goal_norm_final']:.4f}"
        for r in all_results
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-189 -- MECH-112 Goal Present vs Absent\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-112\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**Dispatch mode:** discriminative_pair\n\n"
        f"## Design\n\n"
        f"Two-condition discriminative pair: GOAL_PRESENT (z_goal + benefit_eval active)"
        f" vs GOAL_ABSENT (both ablated). Matched seeds, identical architecture otherwise.\n"
        f"GOAL_PRESENT eval: harm - benefit - goal_proximity scoring.\n"
        f"GOAL_ABSENT eval: harm-only scoring.\n"
        f"Warmup: 50% proximity-greedy navigation to ensure z_goal seeds.\n\n"
        f"## Results\n\n"
        f"| Metric | GOAL_PRESENT | GOAL_ABSENT |\n"
        f"|---|---|---|\n"
        f"| resource_collection_rate | {gp_resource_rate:.3f} | {ga_resource_rate:.3f} |\n"
        f"| mean_resources_per_ep | {gp_mean_resources:.2f} | {ga_mean_resources:.2f} |\n"
        f"| mean_benefit | {gp_mean_benefit:.5f} | {ga_mean_benefit:.5f} |\n"
        f"| harm_rate | {gp_harm_rate:.5f} | {ga_harm_rate:.5f} |\n\n"
        f"**benefit_eval_auc (GOAL_PRESENT):** {gp_benefit_auc:.4f}\n"
        f"**goal_norm_final (GOAL_PRESENT avg):** {gp_goal_norm:.4f}\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: resource_lift >= 0.03 | {'PASS' if c1_pass else 'FAIL'}"
        f" | {resource_lift:+.3f} |\n"
        f"| C2: benefit_ratio >= 1.15x | {'PASS' if c2_pass else 'FAIL'}"
        f" | {benefit_ratio:.2f}x |\n"
        f"| C3: harm_ratio <= 1.3x | {'PASS' if c3_pass else 'FAIL'}"
        f" | {harm_ratio:.2f}x |\n"
        f"| C4: goal_norm >= 0.05 | {'PASS' if c4_pass else 'FAIL'}"
        f" | {gp_goal_norm:.4f} |\n\n"
        f"Criteria met: {criteria_met}/4 -> **{status}**\n\n"
        f"## Per-Seed\n\n{per_seed_rows}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "gp_resource_rate":        float(gp_resource_rate),
        "gp_mean_resources":       float(gp_mean_resources),
        "gp_mean_benefit":         float(gp_mean_benefit),
        "gp_harm_rate":            float(gp_harm_rate),
        "gp_benefit_eval_auc":     float(gp_benefit_auc),
        "gp_goal_norm_final":      float(gp_goal_norm),
        "ga_resource_rate":        float(ga_resource_rate),
        "ga_mean_resources":       float(ga_mean_resources),
        "ga_mean_benefit":         float(ga_mean_benefit),
        "ga_harm_rate":            float(ga_harm_rate),
        "resource_lift":           float(resource_lift),
        "benefit_ratio":           float(benefit_ratio),
        "harm_ratio":              float(harm_ratio),
        "benefit_weight":          float(benefit_weight),
        "goal_weight":             float(goal_weight),
        "n_seeds":                 float(len(seeds)),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    per_seed_results = []
    for r in all_results:
        per_seed_results.append({
            "seed": r["seed"],
            "gp_resource_rate": r["conditions"]["GOAL_PRESENT"]["resource_collection_rate"],
            "gp_mean_resources": r["conditions"]["GOAL_PRESENT"]["mean_resources_per_ep"],
            "gp_mean_benefit": r["conditions"]["GOAL_PRESENT"]["mean_benefit"],
            "gp_harm_rate": r["conditions"]["GOAL_PRESENT"]["harm_rate"],
            "gp_benefit_eval_auc": r["conditions"]["GOAL_PRESENT"]["benefit_eval_auc"],
            "gp_goal_norm_final": r["conditions"]["GOAL_PRESENT"]["goal_norm_final"],
            "ga_resource_rate": r["conditions"]["GOAL_ABSENT"]["resource_collection_rate"],
            "ga_mean_resources": r["conditions"]["GOAL_ABSENT"]["mean_resources_per_ep"],
            "ga_mean_benefit": r["conditions"]["GOAL_ABSENT"]["mean_benefit"],
            "ga_harm_rate": r["conditions"]["GOAL_ABSENT"]["harm_rate"],
        })

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "per_seed_results": per_seed_results,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if criteria_met >= 2 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
    }


# ------------------------------------------------------------------ #
# __main__: flat JSON output to REE_assembly/evidence/experiments/     #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",           type=int,   nargs="+", default=[42, 7, 13])
    parser.add_argument("--warmup",          type=int,   default=300)
    parser.add_argument("--eval-eps",        type=int,   default=100)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--benefit-weight",  type=float, default=1.0)
    parser.add_argument("--goal-weight",     type=float, default=1.0)
    parser.add_argument("--dry-run",         action="store_true",
                        help="Run 1 seed, 2 warmup eps, 2 eval eps for smoke test")
    args = parser.parse_args()

    if args.dry_run:
        args.seeds = [42]
        args.warmup = 2
        args.eval_eps = 2
        print("[DRY-RUN] 1 seed, 2 warmup, 2 eval", flush=True)

    result = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        benefit_weight=args.benefit_weight,
        goal_weight=args.goal_weight,
    )

    if args.dry_run:
        print("\n[DRY-RUN] Smoke test complete. Not writing output file.", flush=True)
        print(f"Status: {result['status']}", flush=True)
        for k, v in result["metrics"].items():
            print(f"  {k}: {v}", flush=True)
        sys.exit(0)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
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
        print(f"  {k}: {v}", flush=True)
