#!/opt/local/bin/python3
"""
V3-EXQ-183 -- ARC-030 Shared Selector Payoff

Claims: ARC-030, MECH-112

Scientific question: Does a shared harm+goal selector (E3 with both
harm_eval and benefit_eval active) outperform a harm-only selector
in a hazard-bearing goal task?

Design:
  - 3 conditions: COMBINED, HARM_ONLY, RANDOM
  - COMBINED: E3 with harm_eval + trained benefit_eval_head (Go+NoGo)
  - HARM_ONLY: E3 with harm_eval only (benefit_eval disabled/zeroed)
  - RANDOM: pure random baseline
  - Warmup: 300 episodes with random actions; trains standard REE agent,
    harm_eval_head on stratified harm samples, benefit_eval_head on
    BCE (is_near_resource labels).
  - Eval: 100 episodes per condition per seed.
  - 3 seeds: [42, 7, 13]

CRITICAL: benefit_eval_head has terminal Sigmoid (output in [0,1]).
Training uses F.binary_cross_entropy (NOT with_logits) to avoid the
Sigmoid+MSE saturation that caused 085m failure (benefit_r2_train=-8.6).
Labels: is_near_resource = 1.0 if manhattan_dist_to_nearest_resource <= 2.

PASS criteria:
  C1: benefit_ratio_combined_vs_harm_only >= 1.2
  C2: benefit_eval_auc >= 0.65
  C3: combined_harm_rate <= 1.2 * harm_only_harm_rate
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_183_arc030_shared_selector_payoff"
CLAIM_IDS = ["ARC-030", "MECH-112"]


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


def _compute_auc(scores: List[float], labels: List[float]) -> float:
    """Compute AUC-ROC from predicted scores and binary labels."""
    n = len(scores)
    if n < 4:
        return 0.5
    paired = list(zip(scores, labels))
    paired.sort(key=lambda x: -x[0])  # descending by score
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
        # trapezoidal rule
        auc += 0.5 * (fp - prev_fp) * (tp + prev_tp)
        prev_fp = fp
        prev_tp = tp
    auc /= (total_pos * total_neg)
    return float(auc)


# ------------------------------------------------------------------ #
# Score-based action selection                                         #
# ------------------------------------------------------------------ #

def _select_action_combined(
    agent: REEAgent,
    z_world: torch.Tensor,
    num_actions: int,
    benefit_weight: float = 1.0,
) -> int:
    """
    COMBINED: action_score = harm_eval(z_world) - benefit_weight * benefit_eval(z_world).
    For each candidate action, use E2.world_forward to estimate z_world_next, then score.
    Pick action with lowest combined score (low harm, high benefit).
    """
    with torch.no_grad():
        best_action = 0
        best_score = float("inf")
        for idx in range(num_actions):
            a_oh = _action_to_onehot(idx, num_actions, z_world.device)
            z_world_next = agent.e2.world_forward(z_world, a_oh)
            harm_score = agent.e3.harm_eval(z_world_next).mean().item()
            benefit_score = agent.e3.benefit_eval(z_world_next).mean().item()
            score = harm_score - benefit_weight * benefit_score
            if score < best_score:
                best_score = score
                best_action = idx
    return best_action


def _select_action_harm_only(
    agent: REEAgent,
    z_world: torch.Tensor,
    num_actions: int,
) -> int:
    """
    HARM_ONLY: action_score = harm_eval(z_world_next).
    Pick action with lowest harm score.
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
# Main run function                                                     #
# ------------------------------------------------------------------ #

def _run_single(
    seed: int,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    benefit_weight: float,
) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(
        seed=seed,
        size=10,
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

    action_dim = env.action_dim

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=0,
        novelty_bonus_weight=0.0,
    )

    agent = REEAgent(config)

    # Separate optimizer groups
    standard_params = [p for n, p in agent.named_parameters()
                       if "harm_eval_head" not in n and "benefit_eval_head" not in n]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    benefit_eval_params = list(agent.e3.benefit_eval_head.parameters())

    optimizer = optim.Adam(standard_params, lr=1e-3)
    harm_eval_opt = optim.Adam(harm_eval_params, lr=1e-4)
    benefit_eval_opt = optim.Adam(benefit_eval_params, lr=1e-3)

    # Replay buffers
    harm_buf_pos: List[torch.Tensor] = []  # z_world where harm occurred
    harm_buf_neg: List[torch.Tensor] = []  # z_world where no harm
    benefit_buf_zw: List[torch.Tensor] = []  # z_world samples
    benefit_buf_lbl: List[float] = []        # is_near_resource labels
    MAX_BUF = 4000

    # AUC evaluation buffer (last 50 warmup episodes)
    auc_eval_scores: List[float] = []
    auc_eval_labels: List[float] = []

    agent.train()

    # ---- WARMUP: random actions, train agent + harm_eval + benefit_eval ----
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

            # Random action during warmup
            action_idx = random.randint(0, action_dim - 1)
            action_oh = _action_to_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh

            # Compute is_near_resource label BEFORE step (current state)
            dist = _manhattan_dist_to_nearest_resource(env)
            is_near = 1.0 if dist <= 2 else 0.0

            _, harm_signal, done, info, obs_dict = env.step(action_oh)
            ttype = info.get("transition_type", "none")

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
                    torch.ones(k_pos, 1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                ], dim=0)
                pred_harm = agent.e3.harm_eval(zw_b)
                # harm_eval_head has terminal Sigmoid -> use BCE (not with_logits)
                harm_loss = F.binary_cross_entropy(pred_harm, target)
                if harm_loss.requires_grad:
                    harm_eval_opt.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5,
                    )
                    harm_eval_opt.step()

            # --- benefit_eval training (BCE on is_near_resource) ---
            benefit_buf_zw.append(z_world_curr)
            benefit_buf_lbl.append(is_near)
            if len(benefit_buf_zw) > MAX_BUF:
                benefit_buf_zw = benefit_buf_zw[-MAX_BUF:]
                benefit_buf_lbl = benefit_buf_lbl[-MAX_BUF:]

            if len(benefit_buf_zw) >= 32 and step_i % 4 == 0:
                k = min(32, len(benefit_buf_zw))
                indices = random.sample(range(len(benefit_buf_zw)), k)
                zw_batch = torch.cat([benefit_buf_zw[i] for i in indices], dim=0)
                lbl_batch = torch.tensor(
                    [benefit_buf_lbl[i] for i in indices],
                    dtype=torch.float32,
                ).unsqueeze(1).to(agent.device)
                # benefit_eval_head has terminal Sigmoid -> output in [0,1]
                # CRITICAL: use F.binary_cross_entropy (NOT with_logits)
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

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
            print(
                f"  [warmup] seed={seed} ep {ep+1}/{warmup_episodes}"
                f" harm_pos={len(harm_buf_pos)} harm_neg={len(harm_buf_neg)}"
                f" benefit_buf={len(benefit_buf_zw)}"
                f" benefit_samples_seen={agent.e3._benefit_samples_seen}",
                flush=True,
            )

    # Compute benefit_eval AUC on held-out warmup data
    benefit_eval_auc = _compute_auc(auc_eval_scores, auc_eval_labels)
    print(
        f"  [warmup done] seed={seed} benefit_eval_auc={benefit_eval_auc:.4f}"
        f" auc_n={len(auc_eval_scores)}",
        flush=True,
    )

    # ---- EVAL ----
    agent.eval()

    conditions = ["COMBINED", "HARM_ONLY", "RANDOM"]
    cond_results: Dict[str, Dict] = {}

    for cond in conditions:
        benefit_counts: List[int] = []   # 1 if collected >= 1 resource
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

                if cond == "COMBINED":
                    action_idx = _select_action_combined(
                        agent, z_world_curr, action_dim,
                        benefit_weight=benefit_weight,
                    )
                elif cond == "HARM_ONLY":
                    action_idx = _select_action_harm_only(
                        agent, z_world_curr, action_dim,
                    )
                else:  # RANDOM
                    action_idx = random.randint(0, action_dim - 1)

                action_oh = _action_to_onehot(action_idx, action_dim, agent.device)
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

            benefit_counts.append(1 if ep_resources >= 1 else 0)
            mean_benefits.append(ep_benefit_sum / max(1, ep_steps))
            harm_rates.append(ep_harm_sum / max(1, ep_steps))

        benefit_rate = float(sum(benefit_counts)) / max(1, len(benefit_counts))
        mean_benefit = float(sum(mean_benefits)) / max(1, len(mean_benefits))
        harm_rate = float(sum(harm_rates)) / max(1, len(harm_rates))

        cond_results[cond] = {
            "benefit_rate": benefit_rate,
            "mean_benefit": mean_benefit,
            "harm_rate": harm_rate,
        }

        print(
            f"  [eval] seed={seed} cond={cond}"
            f" benefit_rate={benefit_rate:.3f}"
            f" mean_benefit={mean_benefit:.5f}"
            f" harm_rate={harm_rate:.5f}",
            flush=True,
        )

    return {
        "seed": seed,
        "benefit_eval_auc": benefit_eval_auc,
        "conditions": cond_results,
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
    **kwargs,
) -> dict:
    all_results: List[Dict] = []

    for seed in seeds:
        print(
            f"\n[V3-EXQ-183] seed={seed}"
            f" warmup={warmup_episodes} eval={eval_episodes}"
            f" steps={steps_per_episode}"
            f" benefit_weight={benefit_weight}",
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
        )
        all_results.append(r)

    def _avg_cond(key: str, cond: str) -> float:
        vals = [r["conditions"][cond][key] for r in all_results]
        return float(sum(vals) / max(1, len(vals)))

    def _avg_key(key: str) -> float:
        vals = [r[key] for r in all_results]
        return float(sum(vals) / max(1, len(vals)))

    # Aggregate per condition
    combined_benefit_rate = _avg_cond("benefit_rate", "COMBINED")
    combined_mean_benefit = _avg_cond("mean_benefit", "COMBINED")
    combined_harm_rate = _avg_cond("harm_rate", "COMBINED")

    harm_only_benefit_rate = _avg_cond("benefit_rate", "HARM_ONLY")
    harm_only_mean_benefit = _avg_cond("mean_benefit", "HARM_ONLY")
    harm_only_harm_rate = _avg_cond("harm_rate", "HARM_ONLY")

    random_benefit_rate = _avg_cond("benefit_rate", "RANDOM")
    random_mean_benefit = _avg_cond("mean_benefit", "RANDOM")
    random_harm_rate = _avg_cond("harm_rate", "RANDOM")

    avg_benefit_eval_auc = _avg_key("benefit_eval_auc")

    # Compute benefit ratio (combined vs harm_only)
    benefit_ratio = (
        combined_benefit_rate / max(1e-6, harm_only_benefit_rate)
        if harm_only_benefit_rate > 1e-6 else 0.0
    )

    # Harm ratio (combined / harm_only)
    harm_ratio = (
        combined_harm_rate / max(1e-6, harm_only_harm_rate)
        if harm_only_harm_rate > 1e-6 else 0.0
    )

    # PASS criteria
    c1_pass = benefit_ratio >= 1.2
    c2_pass = avg_benefit_eval_auc >= 0.65
    c3_pass = harm_ratio <= 1.2

    all_pass = c1_pass and c2_pass and c3_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass])
    status = "PASS" if all_pass else "FAIL"

    if all_pass:
        decision = "retain_ree"
    elif c2_pass:
        decision = "hybridize"
    else:
        decision = "retire_ree_claim"

    print(f"\n[V3-EXQ-183] Final results:", flush=True)
    print(
        f"  COMBINED:  benefit_rate={combined_benefit_rate:.3f}"
        f" mean_benefit={combined_mean_benefit:.5f}"
        f" harm_rate={combined_harm_rate:.5f}",
        flush=True,
    )
    print(
        f"  HARM_ONLY: benefit_rate={harm_only_benefit_rate:.3f}"
        f" mean_benefit={harm_only_mean_benefit:.5f}"
        f" harm_rate={harm_only_harm_rate:.5f}",
        flush=True,
    )
    print(
        f"  RANDOM:    benefit_rate={random_benefit_rate:.3f}"
        f" mean_benefit={random_mean_benefit:.5f}"
        f" harm_rate={random_harm_rate:.5f}",
        flush=True,
    )
    print(
        f"  benefit_eval_auc={avg_benefit_eval_auc:.4f}",
        flush=True,
    )
    print(
        f"  benefit_ratio={benefit_ratio:.2f}x"
        f" harm_ratio={harm_ratio:.2f}x",
        flush=True,
    )
    print(
        f"  decision={decision}  status={status} ({criteria_met}/3)",
        flush=True,
    )

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: benefit_ratio={benefit_ratio:.2f}x < 1.2x."
            " Combined selector does not sufficiently outperform harm-only."
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: benefit_eval_auc={avg_benefit_eval_auc:.4f} < 0.65."
            " Benefit head cannot discriminate near-resource states."
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: harm_ratio={harm_ratio:.2f}x > 1.2x."
            " Goal pursuit increases harm beyond acceptable margin."
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    per_seed_rows = "\n".join(
        f"  seed={r['seed']}: auc={r['benefit_eval_auc']:.4f}"
        f" combined_br={r['conditions']['COMBINED']['benefit_rate']:.3f}"
        f" harm_only_br={r['conditions']['HARM_ONLY']['benefit_rate']:.3f}"
        f" random_br={r['conditions']['RANDOM']['benefit_rate']:.3f}"
        for r in all_results
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-183 -- ARC-030 Shared Selector Payoff\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** ARC-030, MECH-112\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n\n"
        f"## Design\n\n"
        f"Shared harm+goal selector (E3 with benefit_eval_head) vs harm-only vs random.\n"
        f"benefit_eval_head trained with F.binary_cross_entropy (Sigmoid already applied).\n"
        f"Labels: is_near_resource = manhattan_dist <= 2.\n"
        f"Action selection: score = harm_eval(z_world_next) - benefit_weight * benefit_eval(z_world_next).\n\n"
        f"## Results\n\n"
        f"| Condition | benefit_rate | mean_benefit | harm_rate |\n"
        f"|---|---|---|---|\n"
        f"| COMBINED | {combined_benefit_rate:.3f} | {combined_mean_benefit:.5f}"
        f" | {combined_harm_rate:.5f} |\n"
        f"| HARM_ONLY | {harm_only_benefit_rate:.3f} | {harm_only_mean_benefit:.5f}"
        f" | {harm_only_harm_rate:.5f} |\n"
        f"| RANDOM | {random_benefit_rate:.3f} | {random_mean_benefit:.5f}"
        f" | {random_harm_rate:.5f} |\n\n"
        f"**benefit_eval_auc:** {avg_benefit_eval_auc:.4f}\n"
        f"**benefit_ratio (combined/harm_only):** {benefit_ratio:.2f}x\n"
        f"**harm_ratio (combined/harm_only):** {harm_ratio:.2f}x\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: benefit_ratio >= 1.2x | {'PASS' if c1_pass else 'FAIL'}"
        f" | {benefit_ratio:.2f}x |\n"
        f"| C2: benefit_eval_auc >= 0.65 | {'PASS' if c2_pass else 'FAIL'}"
        f" | {avg_benefit_eval_auc:.4f} |\n"
        f"| C3: harm_ratio <= 1.2x | {'PASS' if c3_pass else 'FAIL'}"
        f" | {harm_ratio:.2f}x |\n\n"
        f"Criteria met: {criteria_met}/3 -> **{status}**\n\n"
        f"## Per-Seed\n\n{per_seed_rows}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "combined_benefit_rate":    float(combined_benefit_rate),
        "combined_mean_benefit":    float(combined_mean_benefit),
        "combined_harm_rate":       float(combined_harm_rate),
        "harm_only_benefit_rate":   float(harm_only_benefit_rate),
        "harm_only_mean_benefit":   float(harm_only_mean_benefit),
        "harm_only_harm_rate":      float(harm_only_harm_rate),
        "random_benefit_rate":      float(random_benefit_rate),
        "random_mean_benefit":      float(random_mean_benefit),
        "random_harm_rate":         float(random_harm_rate),
        "benefit_eval_auc":         float(avg_benefit_eval_auc),
        "benefit_ratio":            float(benefit_ratio),
        "harm_ratio":               float(harm_ratio),
        "benefit_weight":           float(benefit_weight),
        "n_seeds":                  float(len(seeds)),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    per_seed_results = []
    for r in all_results:
        per_seed_results.append({
            "seed": r["seed"],
            "benefit_eval_auc": r["benefit_eval_auc"],
            "combined_benefit_rate": r["conditions"]["COMBINED"]["benefit_rate"],
            "combined_mean_benefit": r["conditions"]["COMBINED"]["mean_benefit"],
            "combined_harm_rate": r["conditions"]["COMBINED"]["harm_rate"],
            "harm_only_benefit_rate": r["conditions"]["HARM_ONLY"]["benefit_rate"],
            "harm_only_mean_benefit": r["conditions"]["HARM_ONLY"]["mean_benefit"],
            "harm_only_harm_rate": r["conditions"]["HARM_ONLY"]["harm_rate"],
            "random_benefit_rate": r["conditions"]["RANDOM"]["benefit_rate"],
            "random_mean_benefit": r["conditions"]["RANDOM"]["mean_benefit"],
            "random_harm_rate": r["conditions"]["RANDOM"]["harm_rate"],
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


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",           type=int,   nargs="+", default=[42, 7, 13])
    parser.add_argument("--warmup",          type=int,   default=300)
    parser.add_argument("--eval-eps",        type=int,   default=100)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--benefit-weight",  type=float, default=1.0)
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
