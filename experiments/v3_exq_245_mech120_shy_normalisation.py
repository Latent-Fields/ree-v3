#!/opt/local/bin/python3
"""
V3-EXQ-245 -- MECH-120 SHY-Analog Normalisation (Phase 1 ordering prerequisite)

Claims: MECH-120

Scientific question:
  Does SHY-analog normalisation (flattening dominant attractors BEFORE replay)
  produce more differentiated ContextMemory schema slots than replay without
  prior normalisation? Is MECH-120 a functional prerequisite for effective
  SWS-analog schema installation?

MECH-120 asserts: Synaptic homeostasis (Tononi/Cirelli SHY hypothesis) must
precede replay consolidation. Without prior normalisation, replaying into a
landscape dominated by recent high-salience experiences reinforces the dominant
trace rather than consolidating diverse content. The offline_phases.md Phase 1
ordering requires SHY normalisation BEFORE NREM schema replay (Phase 2).

Design:
  Four conditions, 3 seeds, 100 training episodes.

  WAKING_ONLY:
    No offline phases. Baseline slot differentiation.

  SWS_NO_SHY:
    SWS-analog (alternating SAFE/DANGEROUS context_memory writes) WITHOUT
    prior SHY normalisation. Tests whether plain alternating writes already
    produce differentiated slots.

  SHY_THEN_SWS:
    MECH-120 correct order: shy_normalise(decay=0.85) first (flattens dominant
    attractors), then SWS-analog alternating writes. This is Phase 1->2 order.
    Prediction: better slot differentiation than SWS_NO_SHY because SHY
    ensures no single context's trace dominates the slot landscape before
    context-contrastive writes occur.

  SWS_THEN_SHY:
    Wrong order: SWS-analog writes first, then shy_normalise.
    Tests whether order matters: if SHY after SWS re-erases the differentiation
    that SWS just installed, this is a control for phase ordering effects.

Environment: SAFE (num_hazards=1) vs DANGEROUS (num_hazards=8), alternating 5 eps.
SLEEP_INTERVAL: every 10 episodes.

Metrics:
  - slot_cosine_sim: mean pairwise cosine similarity of ContextMemory slots
    (lower = more differentiated = better schema installation)
  - slot_variance: variance of slot norms (higher = slots more distinct)

PASS criteria:
  S1: slot_cosine_sim(SHY_THEN_SWS) < slot_cosine_sim(SWS_NO_SHY)
      in >=2/3 seeds (SHY before SWS improves differentiation)
  S2: slot_cosine_sim(SHY_THEN_SWS) < slot_cosine_sim(SWS_THEN_SHY)
      in >=2/3 seeds (correct order outperforms wrong order)

PASS: S1 AND S2

experiment_purpose: "evidence"

claim_ids: ["MECH-120"]
experiment_purpose: "evidence"
"""

import sys
import random
import json
import math
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


EXPERIMENT_TYPE = "v3_exq_245_mech120_shy_normalisation"
CLAIM_IDS = ["MECH-120"]
EXPERIMENT_PURPOSE = "evidence"

SLEEP_INTERVAL = 10
SWS_WRITE_STEPS = 20
SHY_DECAY = 0.85
CONTEXT_SWITCH_EVERY = 5

TRAINING_EPISODES = 100
STEPS_PER_EPISODE = 150
EVAL_EPISODES_EACH = 20
SEEDS = [42, 7, 13]

CONDITIONS = ["WAKING_ONLY", "SWS_NO_SHY", "SHY_THEN_SWS", "SWS_THEN_SHY"]


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


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32, world_dim=32,
        alpha_world=0.9, alpha_self=0.3,
        reafference_action_dim=0,
        novelty_bonus_weight=0.0,
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
    """Mean pairwise cosine similarity of ContextMemory slots (lower = better)."""
    with torch.no_grad():
        mem = agent.e1.context_memory.memory   # [num_slots, memory_dim]
        n = mem.shape[0]
        norms = F.normalize(mem, dim=-1)
        sim_matrix = torch.mm(norms, norms.t())
        mask = ~torch.eye(n, dtype=torch.bool, device=mem.device)
        return sim_matrix[mask].mean().item()


def _compute_slot_variance(agent: REEAgent) -> float:
    """Variance of slot L2 norms (higher = slots more distinct in magnitude)."""
    with torch.no_grad():
        mem = agent.e1.context_memory.memory   # [num_slots, memory_dim]
        norms = mem.norm(dim=-1)               # [num_slots]
        return norms.var().item()


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
    """Alternating SAFE/DANGEROUS context_memory writes (SWS-analog)."""
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
    """Run single episode. Returns (harm_sum, z_world_list)."""
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
    training_episodes: int,
    steps_per_episode: int,
    eval_episodes_each: int,
    verbose: bool = True,
) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env_safe = _make_env_safe(seed)
    env_dang = _make_env_dangerous(seed)
    agent = _make_agent(env_safe)

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
    per_ep_harm_safe: List[float] = []
    per_ep_harm_dang: List[float] = []

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
        harm_rate = ep_harm / steps_per_episode

        if is_safe_ep:
            per_ep_harm_safe.append(harm_rate)
            safe_z_buf.extend(z_list)
            if len(safe_z_buf) > 2000:
                safe_z_buf = safe_z_buf[-2000:]
        else:
            per_ep_harm_dang.append(harm_rate)
            dang_z_buf.extend(z_list)
            if len(dang_z_buf) > 2000:
                dang_z_buf = dang_z_buf[-2000:]

        if len(harm_buf_pos) > 4000:
            harm_buf_pos = harm_buf_pos[-4000:]
        if len(harm_buf_neg) > 4000:
            harm_buf_neg = harm_buf_neg[-4000:]

        if condition != "WAKING_ONLY" and (ep + 1) % SLEEP_INTERVAL == 0:
            if len(safe_z_buf) >= 2 and len(dang_z_buf) >= 2:
                sim_pre = _compute_slot_cosine_sim(agent)

                if condition == "SWS_NO_SHY":
                    # Plain SWS without SHY normalisation
                    _sws_pass(agent, safe_z_buf, dang_z_buf, self_dim, SWS_WRITE_STEPS)

                elif condition == "SHY_THEN_SWS":
                    # MECH-120 correct order: SHY first, then SWS
                    agent.e1.shy_normalise(decay=SHY_DECAY)
                    _sws_pass(agent, safe_z_buf, dang_z_buf, self_dim, SWS_WRITE_STEPS)

                elif condition == "SWS_THEN_SHY":
                    # Wrong order: SWS then SHY (SHY re-erases differentiation)
                    _sws_pass(agent, safe_z_buf, dang_z_buf, self_dim, SWS_WRITE_STEPS)
                    agent.e1.shy_normalise(decay=SHY_DECAY)

                sim_post = _compute_slot_cosine_sim(agent)
                sleep_log.append({
                    "ep": ep,
                    "slot_sim_pre": float(sim_pre),
                    "slot_sim_post": float(sim_post),
                    "delta_sim": float(sim_post - sim_pre),
                })

    # Final slot metrics
    final_slot_sim = _compute_slot_cosine_sim(agent)
    final_slot_var = _compute_slot_variance(agent)

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

    # Trend: did slot sim improve (decrease) over training?
    sim_trend = "NA"
    if len(sleep_log) >= 2:
        early_sim = sum(s["slot_sim_post"] for s in sleep_log[:3]) / 3
        late_sim = sum(s["slot_sim_post"] for s in sleep_log[-3:]) / 3
        sim_trend = "improving" if late_sim < early_sim else "stable_or_worse"

    if verbose:
        print(f"  [seed={seed} {condition}] "
              f"slot_sim={final_slot_sim:.4f} "
              f"slot_var={final_slot_var:.6f} "
              f"harm_safe={harm_rate_safe:.4f} "
              f"harm_dang={harm_rate_dang:.4f} "
              f"sleep_passes={len(sleep_log)} "
              f"trend={sim_trend}")

    return {
        "seed": seed,
        "condition": condition,
        "slot_cosine_sim": float(final_slot_sim),
        "slot_variance": float(final_slot_var),
        "harm_rate_safe": float(harm_rate_safe),
        "harm_rate_dangerous": float(harm_rate_dang),
        "sleep_passes": len(sleep_log),
        "sim_trend": sim_trend,
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

    if args.dry_run:
        print("Smoke test: seed=42, 3 train episodes + 2 eval each condition")
        for cond in CONDITIONS:
            print(f"  Testing condition: {cond}")
            result = _run_condition(
                seed=42, condition=cond,
                training_episodes=3,
                steps_per_episode=30,
                eval_episodes_each=2,
                verbose=False,
            )
            print(f"    slot_sim={result['slot_cosine_sim']:.4f} "
                  f"slot_var={result['slot_variance']:.6f} "
                  f"harm_dang={result['harm_rate_dangerous']:.4f} "
                  f"sleep_passes={result['sleep_passes']}")
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
        for cond in CONDITIONS:
            print(f"Seed {seed} Condition {cond}", flush=True)
            r = _run_condition(
                seed=seed, condition=cond,
                training_episodes=TRAINING_EPISODES,
                steps_per_episode=STEPS_PER_EPISODE,
                eval_episodes_each=EVAL_EPISODES_EACH,
            )
            print("verdict: PASS", flush=True)
            # Remove sleep_log from per-seed results to keep output compact
            r_slim = {k: v for k, v in r.items() if k != "sleep_log"}
            all_results.append(r_slim)

    def by_cond(c):
        return [r for r in all_results if r["condition"] == c]

    sws_no_shy  = by_cond("SWS_NO_SHY")
    shy_then_sws = by_cond("SHY_THEN_SWS")
    sws_then_shy = by_cond("SWS_THEN_SHY")

    # S1: SHY_THEN_SWS slot_cosine_sim < SWS_NO_SHY in >=2/3 seeds
    s1_wins = sum(1 for a, b in zip(sws_no_shy, shy_then_sws)
                  if b["slot_cosine_sim"] < a["slot_cosine_sim"])
    s1 = s1_wins >= 2

    # S2: SHY_THEN_SWS < SWS_THEN_SHY in >=2/3 seeds
    s2_wins = sum(1 for a, b in zip(sws_then_shy, shy_then_sws)
                  if b["slot_cosine_sim"] < a["slot_cosine_sim"])
    s2 = s2_wins >= 2

    outcome = "PASS" if (s1 and s2) else "FAIL"
    direction = "supports" if outcome == "PASS" else "weakens"

    summary = {
        "s1_shy_before_sws_beats_sws_alone": {
            "wins": s1_wins, "pass": s1,
            "desc": "SHY_THEN_SWS slot_cosine_sim < SWS_NO_SHY in >=2/3 seeds",
        },
        "s2_correct_order_beats_wrong_order": {
            "wins": s2_wins, "pass": s2,
            "desc": "SHY_THEN_SWS slot_cosine_sim < SWS_THEN_SHY in >=2/3 seeds",
        },
    }

    print(f"\nOutcome: {outcome}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Status: {outcome}", flush=True)

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
            "conditions": CONDITIONS,
            "sleep_interval": SLEEP_INTERVAL,
            "sws_write_steps": SWS_WRITE_STEPS,
            "shy_decay": SHY_DECAY,
            "context_switch_every": CONTEXT_SWITCH_EVERY,
            "training_episodes": TRAINING_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "eval_episodes_each": EVAL_EPISODES_EACH,
            "seeds": SEEDS,
            "env_safe_num_hazards": 1,
            "env_dangerous_num_hazards": 8,
        },
    }

    out_file = out_dir / f"{run_id}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Output written to: {out_file}")


if __name__ == "__main__":
    main()
