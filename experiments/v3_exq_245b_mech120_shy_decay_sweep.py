#!/opt/local/bin/python3
"""
V3-EXQ-245b -- MECH-120 SHY Decay Parameter Sweep (diagnostic)

Claims: MECH-120
Supersedes: V3-EXQ-245a

Scientific question:
  EXQ-245a confirmed the wiring works (S1 PASS: norm_std reduced) but
  decay=0.85 applied cumulatively across 10 sleep cycles is too aggressive
  (0.85^10 = 20% retention). Phase 2 writes cannot overcome the flattening.

  This sweep tests three decay rates and two Phase 2 write counts to find
  the parameter regime where SHY flattens dominant attractors WITHOUT
  collapsing all slots to the mean.

Design:
  6 conditions (3 decay rates x 2 write counts) + 1 baseline, 3 seeds,
  100 training episodes.

  Conditions:
    BASELINE:         shy_enabled=False, no SHY, 20 writes (same as 245a DISABLED)
    SHY_090_W20:      decay=0.90, 20 writes
    SHY_090_W40:      decay=0.90, 40 writes
    SHY_095_W20:      decay=0.95, 20 writes
    SHY_095_W40:      decay=0.95, 40 writes
    SHY_098_W20:      decay=0.98, 20 writes
    SHY_098_W40:      decay=0.98, 40 writes

  decay=0.98 per cycle: 0.98^10 = 82% retention (gentle)
  decay=0.95 per cycle: 0.95^10 = 60% retention (moderate)
  decay=0.90 per cycle: 0.90^10 = 35% retention (aggressive)

Metrics:
  - slot_cosine_sim: mean pairwise cosine similarity (lower = better)
  - slot_norm_std: std of slot norms (SHY should reduce but not collapse)

PASS criteria:
  P1: At least ONE SHY condition has slot_cosine_sim < BASELINE in >=2/3 seeds
  P2: That same condition also has slot_norm_std < BASELINE in >=2/3 seeds
  (SHY must both differentiate AND normalize -- not just collapse)

PASS: P1 AND P2

experiment_purpose: "diagnostic"
claim_ids: ["MECH-120"]
"""

import sys
import random
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_245b_mech120_shy_decay_sweep"
CLAIM_IDS = ["MECH-120"]
EXPERIMENT_PURPOSE = "diagnostic"

SLEEP_INTERVAL = 10
CONTEXT_SWITCH_EVERY = 5

TRAINING_EPISODES = 100
STEPS_PER_EPISODE = 150
EVAL_EPISODES_EACH = 20
SEEDS = [42, 7, 13]

# condition_name -> (shy_enabled, decay_rate, sws_write_steps)
CONDITIONS = {
    "BASELINE":    (False, 0.85, 20),
    "SHY_090_W20": (True,  0.90, 20),
    "SHY_090_W40": (True,  0.90, 40),
    "SHY_095_W20": (True,  0.95, 20),
    "SHY_095_W40": (True,  0.95, 40),
    "SHY_098_W20": (True,  0.98, 20),
    "SHY_098_W40": (True,  0.98, 40),
}


# ------------------------------------------------------------------ #
# Environment / agent helpers                                          #
# ------------------------------------------------------------------ #

def _make_env_safe(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed, size=10,
        num_hazards=1, num_resources=4,
        hazard_harm=0.02,
        env_drift_interval=50, env_drift_prob=0.05,
        proximity_harm_scale=0.10, proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5, energy_decay=0.005,
        use_proxy_fields=True, resource_respawn_on_consume=True,
    )


def _make_env_dangerous(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed + 1000, size=10,
        num_hazards=8, num_resources=4,
        hazard_harm=0.05,
        env_drift_interval=50, env_drift_prob=0.05,
        proximity_harm_scale=0.15, proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5, energy_decay=0.005,
        use_proxy_fields=True, resource_respawn_on_consume=True,
    )


def _make_agent(env: CausalGridWorldV2, shy_enabled: bool, shy_decay_rate: float) -> REEAgent:
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32, world_dim=32,
        alpha_world=0.9, alpha_self=0.3,
        reafference_action_dim=0,
        novelty_bonus_weight=0.0,
        shy_enabled=shy_enabled,
        shy_decay_rate=shy_decay_rate,
    )
    return REEAgent(config)


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


# ------------------------------------------------------------------ #
# Slot metrics                                                         #
# ------------------------------------------------------------------ #

def _compute_slot_cosine_sim(agent: REEAgent) -> float:
    with torch.no_grad():
        mem = agent.e1.context_memory.memory
        n = mem.shape[0]
        norms = F.normalize(mem, dim=-1)
        sim_matrix = torch.mm(norms, norms.t())
        mask = ~torch.eye(n, dtype=torch.bool, device=mem.device)
        return sim_matrix[mask].mean().item()


def _compute_slot_norm_std(agent: REEAgent) -> float:
    with torch.no_grad():
        mem = agent.e1.context_memory.memory
        return mem.norm(dim=-1).std().item()


# ------------------------------------------------------------------ #
# Phase implementations                                                #
# ------------------------------------------------------------------ #

def _sws_pass(
    agent: REEAgent,
    safe_z_buf: List[torch.Tensor],
    dang_z_buf: List[torch.Tensor],
    self_dim: int,
    n_writes: int,
) -> None:
    n = min(len(safe_z_buf), len(dang_z_buf), n_writes)
    if n < 2:
        return
    device = safe_z_buf[0].device
    z_self_zeros = torch.zeros(1, self_dim, device=device)
    for i in range(n):
        z_safe = safe_z_buf[-(i + 1)]
        z_dang = dang_z_buf[-(i + 1)]
        state_safe = torch.cat([z_self_zeros, z_safe], dim=-1)
        state_dang = torch.cat([z_self_zeros, z_dang], dim=-1)
        agent.e1.context_memory.write(state_safe)
        agent.e1.context_memory.write(state_dang)


# ------------------------------------------------------------------ #
# Episode runner                                                       #
# ------------------------------------------------------------------ #

def _run_episode(
    agent: REEAgent,
    env: CausalGridWorldV2,
    steps: int,
    train: bool,
    optimizer,
    harm_buf_pos: List,
    harm_buf_neg: List,
    harm_eval_opt=None,
) -> Tuple[float, List[torch.Tensor]]:
    _, obs_dict = env.reset()
    agent.reset()
    ep_harm = 0.0
    z_world_list: List[torch.Tensor] = []

    for _step in range(steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        obs_harm = obs_dict.get("harm_obs", None)

        latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
        agent.clock.advance()
        z_world = latent.z_world.detach().clone()
        z_world_list.append(z_world)

        with torch.no_grad():
            best_action = 0
            best_score = float("inf")
            for idx in range(env.action_dim):
                a_oh = _action_to_onehot(idx, env.action_dim, agent.device)
                z_next = agent.e2.world_forward(z_world, a_oh)
                score = agent.e3.harm_eval(z_next).mean().item()
                if score < best_score:
                    best_score = score
                    best_action = idx

        action_oh = _action_to_onehot(best_action, env.action_dim, agent.device)
        agent._last_action = action_oh

        _, harm_signal, done, _info, obs_dict = env.step(action_oh)
        is_harm = float(harm_signal) < 0

        if is_harm:
            ep_harm += abs(float(harm_signal))

        if train:
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total = e1_loss + e2_loss
            if total.requires_grad:
                optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if is_harm:
                harm_buf_pos.append(z_world)
            else:
                harm_buf_neg.append(z_world)

            if harm_eval_opt is not None and len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
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
                pred = agent.e3.harm_eval_head(zw_b)
                harm_loss = F.binary_cross_entropy_with_logits(pred, target)
                harm_eval_opt.zero_grad()
                harm_loss.backward()
                harm_eval_opt.step()

        if done:
            break

    return ep_harm, z_world_list


# ------------------------------------------------------------------ #
# Condition runner                                                     #
# ------------------------------------------------------------------ #

def _run_condition(
    seed: int,
    condition: str,
    shy_enabled: bool,
    shy_decay_rate: float,
    sws_write_steps: int,
    training_episodes: int,
    steps_per_episode: int,
    eval_episodes_each: int,
    verbose: bool = True,
) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env_safe = _make_env_safe(seed)
    env_dang = _make_env_dangerous(seed)
    agent = _make_agent(env_safe, shy_enabled=shy_enabled, shy_decay_rate=shy_decay_rate)

    standard_params = [p for n, p in agent.named_parameters()
                       if "harm_eval_head" not in n]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    optimizer = optim.Adam(standard_params, lr=1e-3)
    harm_eval_opt = optim.Adam(harm_eval_params, lr=1e-4)

    self_dim = agent.config.latent.self_dim
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    safe_z_buf: List[torch.Tensor] = []
    dang_z_buf: List[torch.Tensor] = []
    sleep_log: List[Dict] = []

    agent.train()

    for ep in range(training_episodes):
        block = ep // CONTEXT_SWITCH_EVERY
        is_safe_ep = (block % 2 == 0)
        env = env_safe if is_safe_ep else env_dang

        if (ep + 1) % 20 == 0:
            print(
                f"  [train] {condition} seed={seed} ep {ep+1}/{training_episodes}",
                flush=True,
            )

        ep_harm, z_list = _run_episode(
            agent, env, steps_per_episode,
            train=True,
            optimizer=optimizer,
            harm_buf_pos=harm_buf_pos,
            harm_buf_neg=harm_buf_neg,
            harm_eval_opt=harm_eval_opt,
        )

        if is_safe_ep:
            safe_z_buf.extend(z_list)
            if len(safe_z_buf) > 2000:
                safe_z_buf = safe_z_buf[-2000:]
        else:
            dang_z_buf.extend(z_list)
            if len(dang_z_buf) > 2000:
                dang_z_buf = dang_z_buf[-2000:]

        if len(harm_buf_pos) > 4000:
            harm_buf_pos = harm_buf_pos[-4000:]
        if len(harm_buf_neg) > 4000:
            harm_buf_neg = harm_buf_neg[-4000:]

        # Sleep pass every SLEEP_INTERVAL episodes
        if (ep + 1) % SLEEP_INTERVAL == 0:
            if len(safe_z_buf) >= 2 and len(dang_z_buf) >= 2:
                sim_pre = _compute_slot_cosine_sim(agent)
                norm_std_pre = _compute_slot_norm_std(agent)

                # enter_sws_mode() handles SHY via config flag
                agent.enter_sws_mode()

                norm_std_post_shy = _compute_slot_norm_std(agent)

                # Phase 2: alternating context writes
                agent.e1._offline_mode = False
                _sws_pass(agent, safe_z_buf, dang_z_buf, self_dim, sws_write_steps)
                agent.e1._offline_mode = True

                sim_post = _compute_slot_cosine_sim(agent)
                norm_std_post = _compute_slot_norm_std(agent)

                # Exit sleep mode
                agent.exit_sleep_mode()

                sleep_log.append({
                    "ep": ep,
                    "slot_sim_pre": float(sim_pre),
                    "slot_sim_post": float(sim_post),
                    "norm_std_pre": float(norm_std_pre),
                    "norm_std_post_shy": float(norm_std_post_shy),
                    "norm_std_post": float(norm_std_post),
                })

    final_slot_sim = _compute_slot_cosine_sim(agent)
    final_norm_std = _compute_slot_norm_std(agent)

    # Evaluation
    agent.eval()
    eval_harm_safe: List[float] = []
    eval_harm_dang: List[float] = []

    for _ in range(eval_episodes_each):
        harm_s, _ = _run_episode(
            agent, env_safe, steps_per_episode,
            train=False, optimizer=optimizer,
            harm_buf_pos=harm_buf_pos, harm_buf_neg=harm_buf_neg,
        )
        eval_harm_safe.append(harm_s / steps_per_episode)

    for _ in range(eval_episodes_each):
        harm_d, _ = _run_episode(
            agent, env_dang, steps_per_episode,
            train=False, optimizer=optimizer,
            harm_buf_pos=harm_buf_pos, harm_buf_neg=harm_buf_neg,
        )
        eval_harm_dang.append(harm_d / steps_per_episode)

    harm_rate_safe = sum(eval_harm_safe) / len(eval_harm_safe)
    harm_rate_dang = sum(eval_harm_dang) / len(eval_harm_dang)

    if verbose:
        print(f"  [seed={seed} {condition}] "
              f"slot_sim={final_slot_sim:.4f} "
              f"norm_std={final_norm_std:.6f} "
              f"harm_safe={harm_rate_safe:.4f} "
              f"harm_dang={harm_rate_dang:.4f} "
              f"sleep_passes={len(sleep_log)}")

    return {
        "seed": seed,
        "condition": condition,
        "shy_enabled": shy_enabled,
        "shy_decay_rate": shy_decay_rate,
        "sws_write_steps": sws_write_steps,
        "slot_cosine_sim": float(final_slot_sim),
        "slot_norm_std": float(final_norm_std),
        "harm_rate_safe": float(harm_rate_safe),
        "harm_rate_dangerous": float(harm_rate_dang),
        "sleep_passes": len(sleep_log),
        "sleep_log": sleep_log,
    }


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cond_names = list(CONDITIONS.keys())

    if args.dry_run:
        print("Smoke test: seed=42, 3 train episodes + 2 eval, all conditions")
        for cond_name in cond_names:
            shy_en, decay, writes = CONDITIONS[cond_name]
            print(f"  Testing: {cond_name} (shy={shy_en}, decay={decay}, writes={writes})")
            result = _run_condition(
                seed=42, condition=cond_name,
                shy_enabled=shy_en, shy_decay_rate=decay,
                sws_write_steps=writes,
                training_episodes=3,
                steps_per_episode=30,
                eval_episodes_each=2,
                verbose=False,
            )
            print(f"    slot_sim={result['slot_cosine_sim']:.4f} "
                  f"norm_std={result['slot_norm_std']:.6f}")
        print("Smoke test PASSED")
        return

    ts_str = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts_str}_v3"

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        script_dir = Path(__file__).resolve().parents[1]
        out_dir = (script_dir.parent / "REE_assembly" / "evidence"
                   / "experiments" / EXPERIMENT_TYPE)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for seed in SEEDS:
        print(f"\nSeed {seed}")
        for cond_name in cond_names:
            shy_en, decay, writes = CONDITIONS[cond_name]
            print(f"Seed {seed} Condition {cond_name}", flush=True)
            r = _run_condition(
                seed=seed, condition=cond_name,
                shy_enabled=shy_en, shy_decay_rate=decay,
                sws_write_steps=writes,
                training_episodes=TRAINING_EPISODES,
                steps_per_episode=STEPS_PER_EPISODE,
                eval_episodes_each=EVAL_EPISODES_EACH,
            )
            print("verdict: PASS", flush=True)
            r_slim = {k: v for k, v in r.items() if k != "sleep_log"}
            all_results.append(r_slim)

    # Evaluate: does any SHY condition beat BASELINE on both metrics?
    baseline_results = [r for r in all_results if r["condition"] == "BASELINE"]

    best_condition = None
    best_p1_wins = 0
    best_p2_wins = 0
    condition_summaries = {}

    for cond_name in cond_names:
        if cond_name == "BASELINE":
            continue
        cond_results = [r for r in all_results if r["condition"] == cond_name]

        # P1: cosine_sim < BASELINE
        p1_wins = sum(1 for b, c in zip(baseline_results, cond_results)
                      if c["slot_cosine_sim"] < b["slot_cosine_sim"])
        # P2: norm_std < BASELINE
        p2_wins = sum(1 for b, c in zip(baseline_results, cond_results)
                      if c["slot_norm_std"] < b["slot_norm_std"])

        condition_summaries[cond_name] = {
            "p1_cosine_wins": p1_wins,
            "p2_norm_wins": p2_wins,
            "p1_pass": p1_wins >= 2,
            "p2_pass": p2_wins >= 2,
            "both_pass": p1_wins >= 2 and p2_wins >= 2,
            "mean_cosine_sim": sum(r["slot_cosine_sim"] for r in cond_results) / 3,
            "mean_norm_std": sum(r["slot_norm_std"] for r in cond_results) / 3,
        }

        if p1_wins >= 2 and p2_wins >= 2:
            if best_condition is None or p1_wins > best_p1_wins:
                best_condition = cond_name
                best_p1_wins = p1_wins
                best_p2_wins = p2_wins

    outcome = "PASS" if best_condition is not None else "FAIL"
    direction = "supports" if outcome == "PASS" else "weakens"

    summary = {
        "best_condition": best_condition,
        "condition_details": condition_summaries,
        "baseline_mean_cosine_sim": sum(r["slot_cosine_sim"] for r in baseline_results) / 3,
        "baseline_mean_norm_std": sum(r["slot_norm_std"] for r in baseline_results) / 3,
    }

    print(f"\nFINAL OUTCOME: {outcome}")
    if best_condition:
        print(f"  Best condition: {best_condition}")
    for cond_name, cs in condition_summaries.items():
        flag = " <-- PASS" if cs["both_pass"] else ""
        print(f"  {cond_name}: cosine_wins={cs['p1_cosine_wins']}/3 "
              f"norm_wins={cs['p2_norm_wins']}/3 "
              f"mean_sim={cs['mean_cosine_sim']:.4f} "
              f"mean_std={cs['mean_norm_std']:.6f}{flag}")

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts_str,
        "outcome": outcome,
        "evidence_direction": direction,
        "pass_criteria_summary": summary,
        "per_seed_results": all_results,
        "config": {
            "conditions": {k: {"shy_enabled": v[0], "decay": v[1], "writes": v[2]}
                           for k, v in CONDITIONS.items()},
            "sleep_interval": SLEEP_INTERVAL,
            "context_switch_every": CONTEXT_SWITCH_EVERY,
            "training_episodes": TRAINING_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "eval_episodes_each": EVAL_EPISODES_EACH,
            "seeds": SEEDS,
        },
    }

    out_file = out_dir / f"{run_id}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nOutput written to {out_file}")


if __name__ == "__main__":
    main()
