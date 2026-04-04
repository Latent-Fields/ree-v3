#!/opt/local/bin/python3
"""
V3-EXQ-242 -- SD-017 Minimal Sleep Phase Ablation

Claims: SD-017, ARC-045, MECH-166

Scientific question: Do SWS-analog (offline schema slot differentiation) and
REM-analog (context-specific attribution replay) phases produce measurably
better context discrimination and harm avoidance than waking-only training?

SD-017 asserts: A minimal sleep-phase infrastructure (SWS-analog + REM-analog)
is required for hippocampal attribution mapping to converge. Without periodic
offline phases, ContextMemory slots remain globally undifferentiated
(cosine_sim -> 1.0) regardless of online training duration.

ARC-045 asserts: Bidirectional hippocampus-cortex flow (hip->cx = SWS schema
installation; cx->hip = REM attribution filling) is necessary for context-
sensitive attribution.

MECH-166 asserts: Slot-formation (SWS) must temporally precede slot-filling
(REM). Concurrent online learning cannot achieve stable slot structure because
the prior is not yet stable when posterior inference begins (INV-044).

Design:
  Three conditions, each run with 5 seeds on a TWO-CONTEXT task:

  Environment:
    SAFE_CONTEXT:      CausalGridWorldV2(num_hazards=1, hazard_harm=0.02)
    DANGEROUS_CONTEXT: CausalGridWorldV2(num_hazards=8, hazard_harm=0.05)
    Alternation: 5 episodes SAFE, then 5 episodes DANGEROUS, repeat.

  WAKING_ONLY:
    Standard online training. ContextMemory.write() called each step.
    No offline consolidation phases.

  SWS_ONLY:
    After every SLEEP_INTERVAL=10 episodes:
      SWS-analog pass: interleave recent SAFE and DANGEROUS z_world writes
      to ContextMemory, alternating one write per context. This installs
      differentiated schema slots (hip->cx direction, MECH-166 slot-formation).
    Measure: slot_cosine_sim (lower = more differentiated).

  SWS_THEN_REM:
    After every SLEEP_INTERVAL=10 episodes:
      (1) SWS-analog pass (same as SWS_ONLY).
      (2) REM-analog pass: targeted harm_eval gradient updates on DANGEROUS
          context harm events (temporal order), plus residue_field.integrate().
          This fills the DANGEROUS context slot with harm attribution evidence
          (cx->hip direction, MECH-166 slot-filling after slot-formation).
    The temporal ordering in REM implements causal attribution: E3 learns
    that harm follows specific action sequences in DANGEROUS context.

Metrics (per seed, at end of training):
  - slot_cosine_sim: mean cosine similarity between ContextMemory slot pairs
    (lower = more differentiated schema)
  - harm_rate_dangerous: harm rate in DANGEROUS context at eval
  - harm_rate_safe: harm rate in SAFE context at eval
  - harm_discrimination: mean harm_eval(z_dangerous) - mean harm_eval(z_safe)
    (positive = correctly assigns higher harm score to dangerous context)

PASS criteria:
  C1: slot_cosine_sim(SWS_ONLY) < slot_cosine_sim(WAKING_ONLY) in >=3/5 seeds
      (SWS-analog differentiates schema slots -- direct slot-formation test)
  C2: harm_rate_dangerous(SWS_THEN_REM) < harm_rate_dangerous(WAKING_ONLY)*0.90
      in >=3/5 seeds (offline phases improve behavioral harm avoidance 10%+)
  C3: harm_rate_safe < 0.04 in all conditions in >=3/5 seeds
      (context specificity -- offline phases do not impair SAFE performance)
  C4: harm_discrimination(SWS_THEN_REM) > harm_discrimination(WAKING_ONLY)
      in >=3/5 seeds (harm evaluation becomes more context-specific)

PASS: C1 AND (C2 OR C4)
FAIL otherwise

EXPERIMENT_PURPOSE: "diagnostic"
  Rationale: SD-017 is not yet a first-class architectural component.
  SWS-analog and REM-analog are implemented via existing substrate hooks
  (context_memory.write, harm_eval_head gradient steps, residue_field.integrate).
  A PASS establishes that periodic offline schema-differentiation + attribution
  replay produces measurable benefit over waking-only training. A FAIL
  diagnoses whether deeper substrate changes are needed (e.g., explicit
  bidirectional ThetaBuffer mode, full E1 offline gradient updates).

claim_ids: ["SD-017", "ARC-045", "MECH-166"]
experiment_purpose: "diagnostic"
"""

import sys
import random
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_242_sd017_sleep_phase_ablation"
CLAIM_IDS = ["SD-017", "ARC-045", "MECH-166"]
EXPERIMENT_PURPOSE = "diagnostic"

SLEEP_INTERVAL = 10       # episodes between sleep phases
SWS_WRITE_STEPS = 20      # alternating writes per SWS pass
CONTEXT_SWITCH_EVERY = 5  # alternate context every N episodes

TRAINING_EPISODES = 200
STEPS_PER_EPISODE = 150
EVAL_EPISODES_EACH = 30   # eval episodes per context (30 SAFE + 30 DANGEROUS)
MAX_HARM_BUF = 4000
SEEDS = [42, 7, 13, 100, 200]

CONDITIONS = ["WAKING_ONLY", "SWS_ONLY", "SWS_THEN_REM"]


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
    """Mean pairwise cosine similarity between ContextMemory slots (lower = more differentiated)."""
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
    SWS-analog: offline schema slot differentiation (hip->cx direction).

    Interleaves SAFE and DANGEROUS z_world writes into ContextMemory,
    alternating one write from each context. The contrast forces different
    slots to encode SAFE vs DANGEROUS patterns (schema installation,
    MECH-166 slot-formation). This is purely direct memory slot writes --
    no gradient steps required.

    ContextMemory.write() selects the least-attended slot each call, so
    alternating writes distribute SAFE and DANGEROUS patterns across slots.
    """
    n = min(len(safe_z_buf), len(dang_z_buf), sws_write_steps)
    if n < 2:
        return {"sws_writes": 0, "slot_cosine_sim_post_sws": _compute_slot_cosine_sim(agent)}

    device = safe_z_buf[0].device
    z_self_zeros = torch.zeros(1, self_dim, device=device)

    for i in range(n):
        z_safe = safe_z_buf[-(i + 1)]   # [1, world_dim]
        z_dang = dang_z_buf[-(i + 1)]   # [1, world_dim]
        state_safe = torch.cat([z_self_zeros, z_safe], dim=-1)   # [1, total_dim]
        state_dang = torch.cat([z_self_zeros, z_dang], dim=-1)   # [1, total_dim]
        # Alternating writes: SAFE then DANGEROUS
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

    Trains E3.harm_eval on DANGEROUS context harm events in temporal order
    (causal attribution: E3 learns DANGEROUS context harm dynamics). Also
    penalises false positives from SAFE context (safe should score near 0).
    Then runs residue_field.integrate() to consolidate neural field.

    Must run AFTER SWS so that ContextMemory slots are already differentiated.
    The temporal ordering is the key property: unlike online stratified replay
    (which mixes timestamps), REM replays DANGEROUS trajectories in the order
    they occurred, preserving causal sequence (action -> harm outcome).
    """
    rem_loss_total = 0.0

    # DANGEROUS harm attribution (temporal order -- oldest first in slice)
    if len(dang_harm_traj) >= 4:
        n = min(len(dang_harm_traj), 64)
        recent_dang = dang_harm_traj[-n:]   # temporal order preserved
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

    # SAFE context false-positive suppression
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

    # Residue field integration: consolidate neural field from harm history
    agent.residue_field.integrate(num_steps=5)

    return {"rem_loss": rem_loss_total}


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
    """
    Run single episode. Returns (harm_sum, z_world_list, harm_events).

    harm_events: list of (z_world, label) tuples where label=1 if harm event.
    This is used to build the temporal harm trajectory for REM-analog.
    """
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

        # Record harm event for REM trajectory buffer (temporal order)
        harm_events.append((z_world, 1.0 if is_harm else 0.0))

        if is_harm:
            ep_harm += abs(float(harm_signal))

        if train:
            # E1 + E2 online training
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total = e1_loss + e2_loss
            if total.requires_grad:
                optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # harm_eval stratified training (online)
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
    """Run one condition (WAKING_ONLY, SWS_ONLY, or SWS_THEN_REM)."""
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

    # Harm buffers (shared across contexts for online training)
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []

    # Context-stratified z_world buffers for SWS writes
    safe_z_buf: List[torch.Tensor] = []
    dang_z_buf: List[torch.Tensor] = []

    # DANGEROUS context harm trajectory (temporal order) for REM-analog
    dang_harm_traj: List[Tuple[torch.Tensor, float]] = []

    # Training
    agent.train()
    per_ep_harm_safe: List[float] = []
    per_ep_harm_dang: List[float] = []
    sleep_log: List[Dict] = []

    for ep in range(training_episodes):
        # Determine context: alternate every CONTEXT_SWITCH_EVERY episodes
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

        # Update context-stratified buffers
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
            dang_harm_traj.extend(harm_events)
            if len(dang_harm_traj) > 5000:
                dang_harm_traj = dang_harm_traj[-5000:]

        # Trim online harm buffers
        if len(harm_buf_pos) > MAX_HARM_BUF:
            harm_buf_pos = harm_buf_pos[-MAX_HARM_BUF:]
        if len(harm_buf_neg) > MAX_HARM_BUF:
            harm_buf_neg = harm_buf_neg[-MAX_HARM_BUF:]

        # Sleep phases at interval boundary
        if condition != "WAKING_ONLY" and (ep + 1) % SLEEP_INTERVAL == 0:
            if len(safe_z_buf) >= 2 and len(dang_z_buf) >= 2:
                sws_m = _sws_analog_pass(
                    agent, safe_z_buf, dang_z_buf,
                    self_dim, SWS_WRITE_STEPS,
                )
                rem_m = {}
                if condition == "SWS_THEN_REM":
                    rem_m = _rem_analog_pass(
                        agent, dang_harm_traj,
                        safe_z_buf[-100:],
                        harm_eval_opt,
                    )
                sleep_log.append({"ep": ep, **sws_m, **rem_m})

    # Final slot cosine similarity
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

    # Harm discrimination: E3 harm_eval gap between contexts
    with torch.no_grad():
        n_samp = min(len(eval_z_safe), len(eval_z_dang), 200)
        z_s_samp = torch.cat(eval_z_safe[:n_samp], dim=0)
        z_d_samp = torch.cat(eval_z_dang[:n_samp], dim=0)
        he_safe = agent.e3.harm_eval(z_s_samp).mean().item()
        he_dang = agent.e3.harm_eval(z_d_samp).mean().item()
        harm_discrimination = he_dang - he_safe

    harm_rate_safe = sum(eval_harm_safe) / len(eval_harm_safe)
    harm_rate_dang = sum(eval_harm_dang) / len(eval_harm_dang)

    last_sws_sim = (sleep_log[-1].get("slot_cosine_sim_post_sws", final_slot_sim)
                   if sleep_log else final_slot_sim)

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
        "slot_cosine_sim_post_last_sws": float(last_sws_sim),
        "harm_rate_safe": float(harm_rate_safe),
        "harm_rate_dangerous": float(harm_rate_dang),
        "harm_discrimination": float(harm_discrimination),
        "harm_eval_safe": float(he_safe),
        "harm_eval_dangerous": float(he_dang),
        "train_harm_safe_final": float(sum(per_ep_harm_safe[-20:]) / max(len(per_ep_harm_safe[-20:]), 1)),
        "train_harm_dang_final": float(sum(per_ep_harm_dang[-20:]) / max(len(per_ep_harm_dang[-20:]), 1)),
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
        print("Smoke test: seed=42, 3 train episodes + 2 eval episodes per condition")
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

    # Full run
    ts = int(time.time())
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

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

    # Evaluate PASS criteria
    def by_cond(c):
        return [r for r in all_results if r["condition"] == c]

    waking = by_cond("WAKING_ONLY")
    sws    = by_cond("SWS_ONLY")
    sws_r  = by_cond("SWS_THEN_REM")

    # C1: SWS_ONLY slot_cosine_sim < WAKING_ONLY in >=3/5 seeds
    c1_wins = sum(1 for w, s in zip(waking, sws)
                  if s["slot_cosine_sim"] < w["slot_cosine_sim"])
    c1 = c1_wins >= 3

    # C2: SWS_THEN_REM harm_rate_dangerous < WAKING_ONLY * 0.90 in >=3/5 seeds
    c2_wins = sum(1 for w, s in zip(waking, sws_r)
                  if s["harm_rate_dangerous"] < w["harm_rate_dangerous"] * 0.90)
    c2 = c2_wins >= 3

    # C3: harm_rate_safe < 0.04 in all conditions in >=3/5 of all runs
    all_safe_ok = sum(1 for r in all_results if r["harm_rate_safe"] < 0.04)
    c3 = all_safe_ok >= (len(SEEDS) * len(CONDITIONS) * 3 // 5)

    # C4: harm_discrimination(SWS_THEN_REM) > harm_discrimination(WAKING_ONLY) in >=3/5
    c4_wins = sum(1 for w, s in zip(waking, sws_r)
                  if s["harm_discrimination"] > w["harm_discrimination"])
    c4 = c4_wins >= 3

    outcome = "PASS" if (c1 and (c2 or c4)) else "FAIL"

    summary = {
        "c1_slot_differentiation": {
            "wins": c1_wins, "pass": c1,
            "desc": "SWS_ONLY slot_cosine_sim < WAKING_ONLY in >=3/5 seeds",
        },
        "c2_harm_rate_benefit": {
            "wins": c2_wins, "pass": c2,
            "desc": "SWS_THEN_REM harm_dangerous < WAKING_ONLY*0.90 in >=3/5 seeds",
        },
        "c3_safe_specificity": {
            "count_ok": all_safe_ok, "pass": c3,
            "desc": "harm_rate_safe < 0.04 in >=60% of all condition-seed runs",
        },
        "c4_harm_discrimination": {
            "wins": c4_wins, "pass": c4,
            "desc": "SWS_THEN_REM harm_discrimination > WAKING_ONLY in >=3/5 seeds",
        },
    }

    print(f"\nOutcome: {outcome}")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "outcome": outcome,
        "evidence_direction": "supports" if outcome == "PASS" else "weakens",
        "evidence_direction_per_claim": {
            "SD-017": "supports" if c1 else "weakens",
            "ARC-045": "supports" if (c1 and c2) else ("mixed" if c1 else "weakens"),
            "MECH-166": "supports" if (c1 and (c2 or c4)) else ("mixed" if c1 else "weakens"),
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
