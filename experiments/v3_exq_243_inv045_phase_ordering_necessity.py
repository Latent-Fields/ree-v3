#!/opt/local/bin/python3
"""
V3-EXQ-243 -- INV-045 Phase Ordering Necessity + MECH-123 Precision Recalibration

Claims: INV-045 (primary), MECH-123 (secondary)

Scientific question:
  Is the NREM->REM (SWS->REM) ordering computationally necessary, or can REM
  alone produce equivalent context discrimination? Does MECH-123 precision
  recalibration (REM posterior prior reset) add value on top of ordered phases?

INV-045 asserts: The NREM->REM sequence is NOT arbitrary. REM attribution replay
cannot fill context slots that do not yet exist (INV-044: schema-before-posterior
constraint). REM_ONLY and wrong-order REM_THEN_SWS must fail to match correct-
order SWS_THEN_REM on harm discrimination. The ordering is a computational
necessity, not a biological convention.

MECH-123 asserts: After SWS+REM consolidation, resetting E3's precision prior
(running_variance) to the episode's natural harm variance range improves
next-waking-cycle performance by starting with calibrated priors rather than
whatever noise-inflated variance was running at end-of-waking.

Design:
  Five conditions, each run with 3 seeds on TWO-CONTEXT task.

  Environment: same as EXQ-242 (SAFE: num_hazards=1 vs DANGEROUS: num_hazards=8)
  Alternation: 5 episodes SAFE, 5 episodes DANGEROUS, repeat.
  SLEEP_INTERVAL: every 10 episodes.

  Conditions:
    WAKING_ONLY:       No offline phases (baseline).
    REM_ONLY:          REM-analog after every interval, NO SWS first.
                       Tests whether attribution replay alone works without
                       prior schema slot formation (INV-045 violation).
    REM_THEN_SWS:      Wrong order: REM first, then SWS.
                       Tests whether reversed ordering also fails (INV-045).
    SWS_THEN_REM:      Correct order: SWS first, then REM (INV-045 prediction).
    SWS_THEN_REM_RECAL: Correct order + MECH-123: after SWS+REM, reset
                        agent.e3._running_variance to mean episode harm variance
                        measured during last N DANGEROUS episodes.

PASS criteria:
  INV-045 (primary):
    P1: harm_discrimination(REM_ONLY) <= harm_discrimination(WAKING_ONLY) + eps
        in >=2/3 seeds (REM without SWS does NOT improve over baseline)
    P2: harm_discrimination(REM_THEN_SWS) <= harm_discrimination(WAKING_ONLY) + eps
        in >=2/3 seeds (wrong order also fails)
    P3: harm_discrimination(SWS_THEN_REM) > harm_discrimination(WAKING_ONLY)
        in >=2/3 seeds (correct order succeeds)
    INV-045 PASSES if P1 AND P2 AND P3.

  MECH-123 (secondary):
    P4: harm_discrimination(SWS_THEN_REM_RECAL) >= harm_discrimination(SWS_THEN_REM)
        in >=2/3 seeds (precision recalibration does not hurt, and ideally helps)

PASS: INV-045 PASSES (P1+P2+P3). MECH-123 assessed independently via P4.

experiment_purpose: "evidence"
  INV-045 is a structural claim about ordering necessity -- this directly tests it.
  MECH-123 is secondary/exploratory in this experiment.

claim_ids: ["INV-045", "MECH-123"]
experiment_purpose: "evidence"
"""

import sys
import random
import json
import time
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


EXPERIMENT_TYPE = "v3_exq_243_inv045_phase_ordering_necessity"
CLAIM_IDS = ["INV-045", "MECH-123"]
EXPERIMENT_PURPOSE = "evidence"

SLEEP_INTERVAL = 10        # episodes between sleep phases
SWS_WRITE_STEPS = 20       # alternating writes per SWS pass
CONTEXT_SWITCH_EVERY = 5   # alternate context every N episodes

TRAINING_EPISODES = 150
STEPS_PER_EPISODE = 150
EVAL_EPISODES_EACH = 20    # eval episodes per context
MAX_HARM_BUF = 4000
SEEDS = [42, 7, 13]

CONDITIONS = ["WAKING_ONLY", "REM_ONLY", "REM_THEN_SWS", "SWS_THEN_REM", "SWS_THEN_REM_RECAL"]

# MECH-123: epsilon for ordering discrimination (REM_ONLY/wrong order must not exceed
# WAKING_ONLY + DISCRIMINATION_EPS on harm_discrimination)
DISCRIMINATION_EPS = 0.01


# ------------------------------------------------------------------ #
# Environment / agent helpers                                          #
# ------------------------------------------------------------------ #

def _make_env_safe(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=1,
        num_resources=4,
        hazard_harm=0.02,
        env_drift_interval=50,
        env_drift_prob=0.05,
        proximity_harm_scale=0.10,
        proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_env_dangerous(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed + 1000,
        size=10,
        num_hazards=8,
        num_resources=4,
        hazard_harm=0.05,
        env_drift_interval=50,
        env_drift_prob=0.05,
        proximity_harm_scale=0.15,
        proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=0,
        novelty_bonus_weight=0.0,
    )
    return REEAgent(config)


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _select_action_harm_avoid(agent: REEAgent, z_world: torch.Tensor,
                               num_actions: int) -> int:
    """Pick action with lowest predicted harm (E3 harm_eval on E2 world_forward)."""
    with torch.no_grad():
        best_action = 0
        best_score = float("inf")
        for idx in range(num_actions):
            a_oh = _action_to_onehot(idx, num_actions, z_world.device)
            z_world_next = agent.e2.world_forward(z_world, a_oh)
            score = agent.e3.harm_eval(z_world_next).mean().item()
            if score < best_score:
                best_score = score
                best_action = idx
    return best_action


# ------------------------------------------------------------------ #
# Sleep phase implementations                                          #
# ------------------------------------------------------------------ #

def _compute_slot_cosine_sim(agent: REEAgent) -> float:
    """Mean pairwise cosine similarity between ContextMemory slots."""
    with torch.no_grad():
        mem = agent.e1.context_memory.memory   # [num_slots, memory_dim]
        n = mem.shape[0]
        norms = F.normalize(mem, dim=-1)
        sim_matrix = torch.mm(norms, norms.t())
        mask = ~torch.eye(n, dtype=torch.bool, device=mem.device)
        return sim_matrix[mask].mean().item()


def _sws_analog_pass(
    agent: REEAgent,
    safe_z_buf: List[torch.Tensor],
    dang_z_buf: List[torch.Tensor],
    self_dim: int,
    sws_write_steps: int,
) -> Dict:
    """
    SWS-analog: schema slot differentiation (hip->cx direction).
    Interleaves SAFE/DANGEROUS z_world writes into ContextMemory.
    """
    n = min(len(safe_z_buf), len(dang_z_buf), sws_write_steps)
    if n < 2:
        return {"sws_writes": 0, "slot_cosine_sim_post_sws": _compute_slot_cosine_sim(agent)}

    device = safe_z_buf[0].device
    z_self_zeros = torch.zeros(1, self_dim, device=device)

    for i in range(n):
        z_safe = safe_z_buf[-(i + 1)]
        z_dang = dang_z_buf[-(i + 1)]
        state_safe = torch.cat([z_self_zeros, z_safe], dim=-1)
        state_dang = torch.cat([z_self_zeros, z_dang], dim=-1)
        agent.e1.context_memory.write(state_safe)
        agent.e1.context_memory.write(state_dang)

    slot_sim = _compute_slot_cosine_sim(agent)
    return {"sws_writes": 2 * n, "slot_cosine_sim_post_sws": slot_sim}


def _rem_analog_pass(
    agent: REEAgent,
    dang_harm_traj: List[Tuple[torch.Tensor, float]],
    safe_z_recent: List[torch.Tensor],
    harm_eval_opt,
) -> Dict:
    """
    REM-analog: context-stratified harm attribution replay (cx->hip direction).
    Trains E3.harm_eval on DANGEROUS harm events (temporal order) + SAFE false-
    positive suppression. Runs residue_field.integrate() to consolidate.
    Must run AFTER SWS for slot slots to exist (INV-045/MECH-166).
    """
    rem_loss_total = 0.0

    # DANGEROUS harm attribution (temporal order)
    if len(dang_harm_traj) >= 4:
        n = min(len(dang_harm_traj), 64)
        recent_dang = dang_harm_traj[-n:]
        zw_list = [zw for (zw, _) in recent_dang]
        lbl_list = [float(lbl) for (_, lbl) in recent_dang]
        zw_d = torch.cat(zw_list, dim=0)
        tgt_d = torch.tensor([[l] for l in lbl_list], dtype=torch.float,
                              device=agent.device)
        pred_d = agent.e3.harm_eval_head(zw_d)
        loss_d = F.binary_cross_entropy_with_logits(pred_d, tgt_d)
        harm_eval_opt.zero_grad()
        loss_d.backward()
        harm_eval_opt.step()
        rem_loss_total += loss_d.item()

    # SAFE false-positive suppression
    if len(safe_z_recent) >= 4:
        n_safe = min(len(safe_z_recent), 32)
        zw_s = torch.cat(safe_z_recent[-n_safe:], dim=0)
        tgt_s = torch.zeros(zw_s.shape[0], 1, device=agent.device)
        pred_s = agent.e3.harm_eval_head(zw_s)
        loss_s = F.binary_cross_entropy_with_logits(pred_s, tgt_s)
        harm_eval_opt.zero_grad()
        loss_s.backward()
        harm_eval_opt.step()
        rem_loss_total += loss_s.item()

    agent.residue_field.integrate(num_steps=5)

    return {"rem_loss": rem_loss_total}


def _mech123_precision_recalibration(
    agent: REEAgent,
    dang_harm_rates: List[float],
) -> Dict:
    """
    MECH-123: Precision prior reset after REM consolidation.

    Measures the natural variance of harm rates during the last N DANGEROUS
    episodes, then resets agent.e3._running_variance to this value. This
    recalibrates E3's commit threshold prior to the next waking cycle,
    so early evidence in the next episode is weighted by the episode-calibrated
    prior rather than the noise-inflated running variance from waking.

    Biological analog: REM aminergic suppression (z_beta) -> E1 runs
    unconstrained -> commitment_threshold and precision_ema_alpha recalibrated.
    """
    if len(dang_harm_rates) < 4:
        return {"mech123_skip": True, "variance_before": agent.e3._running_variance}

    n = min(len(dang_harm_rates), 20)
    recent = dang_harm_rates[-n:]
    # Natural harm variance during recent DANGEROUS episodes
    harm_mean = sum(recent) / len(recent)
    harm_var = sum((h - harm_mean) ** 2 for h in recent) / len(recent)
    # Clamp to [0.01, 0.50] to avoid resetting to degenerate values
    harm_var = max(0.01, min(0.50, harm_var))

    variance_before = agent.e3._running_variance
    agent.e3._running_variance = harm_var

    return {
        "mech123_skip": False,
        "variance_before": float(variance_before),
        "variance_after": float(harm_var),
        "harm_mean_dang": float(harm_mean),
        "n_episodes": n,
    }


# ------------------------------------------------------------------ #
# Episode runner                                                       #
# ------------------------------------------------------------------ #

def _run_episode(
    agent: REEAgent,
    env: CausalGridWorldV2,
    steps: int,
    train: bool,
    optimizer,
    harm_eval_opt,
    harm_buf_pos: List,
    harm_buf_neg: List,
) -> Tuple[float, List[torch.Tensor], List[Tuple[torch.Tensor, float]]]:
    """Run single episode. Returns (harm_sum, z_world_list, harm_events)."""
    _, obs_dict = env.reset()
    agent.reset()
    ep_harm = 0.0
    z_world_list: List[torch.Tensor] = []
    harm_events: List[Tuple[torch.Tensor, float]] = []

    for _step in range(steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        obs_harm = obs_dict.get("harm_obs", None)

        latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
        agent.clock.advance()
        z_world = latent.z_world.detach().clone()
        z_world_list.append(z_world)

        action_idx = _select_action_harm_avoid(agent, z_world, env.action_dim)
        action_oh = _action_to_onehot(action_idx, env.action_dim, agent.device)
        agent._last_action = action_oh

        _, harm_signal, done, _info, obs_dict = env.step(action_oh)
        is_harm = float(harm_signal) < 0

        harm_events.append((z_world, 1.0 if is_harm else 0.0))

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
                pred = agent.e3.harm_eval_head(zw_b)
                harm_loss = F.binary_cross_entropy_with_logits(pred, target)
                harm_eval_opt.zero_grad()
                harm_loss.backward()
                harm_eval_opt.step()

        if done:
            break

    return ep_harm, z_world_list, harm_events


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
    """Run one condition."""
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
    dang_harm_traj: List[Tuple[torch.Tensor, float]] = []
    dang_harm_rates: List[float] = []  # per-episode harm rates for MECH-123
    sleep_log: List[Dict] = []

    agent.train()
    per_ep_harm_safe: List[float] = []
    per_ep_harm_dang: List[float] = []

    for ep in range(training_episodes):
        block = ep // CONTEXT_SWITCH_EVERY
        is_safe_ep = (block % 2 == 0)
        env = env_safe if is_safe_ep else env_dang

        ep_harm, z_list, harm_events = _run_episode(
            agent, env, steps_per_episode,
            train=True,
            optimizer=optimizer,
            harm_eval_opt=harm_eval_opt,
            harm_buf_pos=harm_buf_pos,
            harm_buf_neg=harm_buf_neg,
        )
        harm_rate = ep_harm / steps_per_episode

        if is_safe_ep:
            per_ep_harm_safe.append(harm_rate)
            safe_z_buf.extend(z_list)
            if len(safe_z_buf) > 2000:
                safe_z_buf = safe_z_buf[-2000:]
        else:
            per_ep_harm_dang.append(harm_rate)
            dang_harm_rates.append(harm_rate)
            dang_z_buf.extend(z_list)
            if len(dang_z_buf) > 2000:
                dang_z_buf = dang_z_buf[-2000:]
            dang_harm_traj.extend(harm_events)
            if len(dang_harm_traj) > 5000:
                dang_harm_traj = dang_harm_traj[-5000:]

        if len(harm_buf_pos) > MAX_HARM_BUF:
            harm_buf_pos = harm_buf_pos[-MAX_HARM_BUF:]
        if len(harm_buf_neg) > MAX_HARM_BUF:
            harm_buf_neg = harm_buf_neg[-MAX_HARM_BUF:]

        # Sleep phases at interval boundary
        if condition != "WAKING_ONLY" and (ep + 1) % SLEEP_INTERVAL == 0:
            if len(safe_z_buf) >= 2 and len(dang_z_buf) >= 2:
                sws_m = {}
                rem_m = {}
                mech123_m = {}

                if condition == "REM_ONLY":
                    # REM without prior SWS -- INV-045 violation
                    rem_m = _rem_analog_pass(
                        agent, dang_harm_traj,
                        safe_z_buf[-100:], harm_eval_opt,
                    )

                elif condition == "REM_THEN_SWS":
                    # Wrong order: REM first, then SWS
                    rem_m = _rem_analog_pass(
                        agent, dang_harm_traj,
                        safe_z_buf[-100:], harm_eval_opt,
                    )
                    sws_m = _sws_analog_pass(
                        agent, safe_z_buf, dang_z_buf,
                        self_dim, SWS_WRITE_STEPS,
                    )

                elif condition == "SWS_THEN_REM":
                    # Correct order: SWS then REM
                    sws_m = _sws_analog_pass(
                        agent, safe_z_buf, dang_z_buf,
                        self_dim, SWS_WRITE_STEPS,
                    )
                    rem_m = _rem_analog_pass(
                        agent, dang_harm_traj,
                        safe_z_buf[-100:], harm_eval_opt,
                    )

                elif condition == "SWS_THEN_REM_RECAL":
                    # Correct order + MECH-123 precision recalibration
                    sws_m = _sws_analog_pass(
                        agent, safe_z_buf, dang_z_buf,
                        self_dim, SWS_WRITE_STEPS,
                    )
                    rem_m = _rem_analog_pass(
                        agent, dang_harm_traj,
                        safe_z_buf[-100:], harm_eval_opt,
                    )
                    mech123_m = _mech123_precision_recalibration(agent, dang_harm_rates)

                sleep_log.append({"ep": ep, **sws_m, **rem_m, **mech123_m})

    final_slot_sim = _compute_slot_cosine_sim(agent)

    # Evaluation
    agent.eval()
    eval_harm_safe: List[float] = []
    eval_harm_dang: List[float] = []
    eval_z_safe: List[torch.Tensor] = []
    eval_z_dang: List[torch.Tensor] = []

    for _ in range(eval_episodes_each):
        harm_s, z_s, _ = _run_episode(
            agent, env_safe, steps_per_episode,
            train=False, optimizer=optimizer,
            harm_eval_opt=harm_eval_opt,
            harm_buf_pos=harm_buf_pos,
            harm_buf_neg=harm_buf_neg,
        )
        eval_harm_safe.append(harm_s / steps_per_episode)
        eval_z_safe.extend(z_s)

    for _ in range(eval_episodes_each):
        harm_d, z_d, _ = _run_episode(
            agent, env_dang, steps_per_episode,
            train=False, optimizer=optimizer,
            harm_eval_opt=harm_eval_opt,
            harm_buf_pos=harm_buf_pos,
            harm_buf_neg=harm_buf_neg,
        )
        eval_harm_dang.append(harm_d / steps_per_episode)
        eval_z_dang.extend(z_d)

    with torch.no_grad():
        n_samp = min(len(eval_z_safe), len(eval_z_dang), 200)
        z_s_samp = torch.cat(eval_z_safe[:n_samp], dim=0)
        z_d_samp = torch.cat(eval_z_dang[:n_samp], dim=0)
        he_safe = agent.e3.harm_eval(z_s_samp).mean().item()
        he_dang = agent.e3.harm_eval(z_d_samp).mean().item()
        harm_discrimination = he_dang - he_safe

    harm_rate_safe = sum(eval_harm_safe) / len(eval_harm_safe)
    harm_rate_dang = sum(eval_harm_dang) / len(eval_harm_dang)

    if verbose:
        print(f"  [seed={seed} {condition}] "
              f"slot_sim={final_slot_sim:.4f} "
              f"harm_safe={harm_rate_safe:.4f} "
              f"harm_dang={harm_rate_dang:.4f} "
              f"discrim={harm_discrimination:.4f} "
              f"sleep_passes={len(sleep_log)}")

    return {
        "seed": seed,
        "condition": condition,
        "slot_cosine_sim": float(final_slot_sim),
        "harm_rate_safe": float(harm_rate_safe),
        "harm_rate_dangerous": float(harm_rate_dang),
        "harm_discrimination": float(harm_discrimination),
        "harm_eval_safe": float(he_safe),
        "harm_eval_dangerous": float(he_dang),
        "train_harm_dang_final": float(
            sum(per_ep_harm_dang[-10:]) / max(len(per_ep_harm_dang[-10:]), 1)
        ),
        "sleep_passes": len(sleep_log),
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
                  f"harm_safe={result['harm_rate_safe']:.4f} "
                  f"harm_dang={result['harm_rate_dangerous']:.4f} "
                  f"discrim={result['harm_discrimination']:.4f} "
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
            r = _run_condition(
                seed=seed, condition=cond,
                training_episodes=TRAINING_EPISODES,
                steps_per_episode=STEPS_PER_EPISODE,
                eval_episodes_each=EVAL_EPISODES_EACH,
            )
            all_results.append(r)

    def by_cond(c):
        return [r for r in all_results if r["condition"] == c]

    waking   = by_cond("WAKING_ONLY")
    rem_only = by_cond("REM_ONLY")
    rem_sws  = by_cond("REM_THEN_SWS")
    sws_rem  = by_cond("SWS_THEN_REM")
    sws_rem_r = by_cond("SWS_THEN_REM_RECAL")

    # P1: REM_ONLY does NOT beat WAKING_ONLY + eps on harm_discrimination
    p1_wins = sum(1 for w, r in zip(waking, rem_only)
                  if r["harm_discrimination"] <= w["harm_discrimination"] + DISCRIMINATION_EPS)
    p1 = p1_wins >= 2

    # P2: REM_THEN_SWS does NOT beat WAKING_ONLY + eps
    p2_wins = sum(1 for w, r in zip(waking, rem_sws)
                  if r["harm_discrimination"] <= w["harm_discrimination"] + DISCRIMINATION_EPS)
    p2 = p2_wins >= 2

    # P3: SWS_THEN_REM beats WAKING_ONLY
    p3_wins = sum(1 for w, r in zip(waking, sws_rem)
                  if r["harm_discrimination"] > w["harm_discrimination"])
    p3 = p3_wins >= 2

    # INV-045 overall
    inv045_pass = p1 and p2 and p3

    # P4: SWS_THEN_REM_RECAL >= SWS_THEN_REM (MECH-123 does not hurt)
    p4_wins = sum(1 for s, sr in zip(sws_rem, sws_rem_r)
                  if sr["harm_discrimination"] >= s["harm_discrimination"])
    p4 = p4_wins >= 2

    outcome = "PASS" if inv045_pass else "FAIL"

    # Determine per-claim direction
    # INV-045 requires P1+P2+P3 all true
    inv045_direction = "supports" if inv045_pass else "weakens"
    # MECH-123 is independent: passes if P4 true
    mech123_direction = "supports" if p4 else "weakens"
    # If inv045 fails but we can say something about the pattern, use mixed
    if not inv045_pass:
        if p3:  # SWS_THEN_REM beat waking but REM_ONLY or wrong-order didn't fail
            inv045_direction = "mixed"

    overall_direction = "supports" if inv045_pass else ("mixed" if p3 else "weakens")

    summary = {
        "p1_rem_only_no_benefit": {
            "wins": p1_wins, "pass": p1,
            "desc": "REM_ONLY harm_discrim <= WAKING_ONLY+eps in >=2/3 seeds",
        },
        "p2_wrong_order_no_benefit": {
            "wins": p2_wins, "pass": p2,
            "desc": "REM_THEN_SWS harm_discrim <= WAKING_ONLY+eps in >=2/3 seeds",
        },
        "p3_correct_order_benefits": {
            "wins": p3_wins, "pass": p3,
            "desc": "SWS_THEN_REM harm_discrim > WAKING_ONLY in >=2/3 seeds",
        },
        "p4_mech123_no_regression": {
            "wins": p4_wins, "pass": p4,
            "desc": "SWS_THEN_REM_RECAL harm_discrim >= SWS_THEN_REM in >=2/3 seeds",
        },
        "inv045_pass": inv045_pass,
    }

    print(f"\nOutcome: {outcome}")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts_str,
        "outcome": outcome,
        "evidence_direction": overall_direction,
        "evidence_direction_per_claim": {
            "INV-045": inv045_direction,
            "MECH-123": mech123_direction,
        },
        "pass_criteria_summary": summary,
        "per_seed_results": all_results,
        "config": {
            "conditions": CONDITIONS,
            "sleep_interval": SLEEP_INTERVAL,
            "sws_write_steps": SWS_WRITE_STEPS,
            "context_switch_every": CONTEXT_SWITCH_EVERY,
            "training_episodes": TRAINING_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "eval_episodes_each": EVAL_EPISODES_EACH,
            "seeds": SEEDS,
            "discrimination_eps": DISCRIMINATION_EPS,
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
