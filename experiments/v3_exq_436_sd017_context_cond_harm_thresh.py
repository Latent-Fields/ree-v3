#!/opt/local/bin/python3
"""
V3-EXQ-436 -- SD-017 Sleep Phase Ablation Redesign: Context-Conditioned Harm Threshold

Claims: SD-017, ARC-045, MECH-166

Supersedes: V3-EXQ-242 (diagnostic: slot differentiation worked 4/5 seeds but no
  behavioral path -- agent did not use context knowledge in action selection).

EXP-0106 proposal redesign (DR-6): Close the missing link between ContextMemory
slot readout and action selection by making harm_eval threshold context-conditioned.
Each step: (a) determine active ContextMemory slot via attention argmax,
(b) compute slot "danger score" from running co-occurrence with DANGEROUS episodes,
(c) modulate harm threshold in action selection: dangerous context -> lower threshold
(more cautious filtering) -> higher threshold (more exploratory).

Three conditions, 5 seeds, same env/alternation as EXQ-242:
  WAKING_ONLY: baseline, no sleep phases, no context-conditioned threshold.
  SWS_ONLY:    first-class run_sws_schema_pass() every SLEEP_INTERVAL eps.
  SWS_THEN_REM: run_sleep_cycle() (SWS then REM) every SLEEP_INTERVAL eps,
                PLUS context-conditioned harm threshold in action selection.

Env: CausalGridWorldV2, SAFE (hazards=1) / DANGEROUS (hazards=8), alternating
every CONTEXT_SWITCH_EVERY=5 episodes.

Action selection (all conditions):
  baseline: argmin over actions of E3.harm_eval(E2.world_forward(z_world, a))
  SWS_THEN_REM with context conditioning:
    threshold = base_thresh * (1 - context_beta * slot_danger_score)
    filter candidates whose predicted harm > threshold, pick lowest-harm remaining
    (fallback to unfiltered argmin if all candidates exceed threshold)

Acceptance checks (proposal C1 AND C2 required for PASS):
  C1: slot_cosine_sim(SWS_THEN_REM) < slot_cosine_sim(WAKING_ONLY) in >=3/5 seeds
      (slot differentiation reproduced -- regression check vs EXQ-242)
  C2: harm_rate_dangerous(SWS_THEN_REM) < harm_rate_dangerous(WAKING_ONLY)
      in >=3/5 seeds (context-conditioning produces behavioral benefit)
  C3 (secondary): harm_rate_safe < 0.05 in all conditions in >=3/5 seeds
      (context specificity preserved)
  C4 (secondary): slot_danger_score separation > 0.3 in >=3/5 seeds
      (SAFE and DANGEROUS episodes activate different slots, i.e. action
       pathway has a real signal to condition on)

PASS: C1 AND C2 (proposal pass criteria)
FAIL otherwise

EXPERIMENT_PURPOSE: "evidence"
  Rationale: Adds a concrete behavioral-path claim (C2) on top of the diagnostic
  slot-differentiation check (C1). If C1 passes but C2 fails -> slot
  differentiation does not produce behavioral benefit via threshold modulation.
  If C1 and C2 pass -> SD-017 supported end-to-end (slot formation + slot use).

claim_ids: ["SD-017", "ARC-045", "MECH-166"]
experiment_purpose: "evidence"
"""

import sys
import random
import json
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_436_sd017_context_cond_harm_thresh"
CLAIM_IDS = ["SD-017", "ARC-045", "MECH-166"]
EXPERIMENT_PURPOSE = "evidence"

# Pre-registered thresholds
BASE_HARM_THRESHOLD = 0.05       # filter actions whose predicted harm exceeds this
CONTEXT_BETA = 0.8                # danger-score modulation strength
SLOT_DANGER_EMA_ALPHA = 0.05      # slot_danger_score EMA update rate

SLEEP_INTERVAL = 10
CONTEXT_SWITCH_EVERY = 5
TRAINING_EPISODES = 200
STEPS_PER_EPISODE = 150
EVAL_EPISODES_EACH = 30
MAX_HARM_BUF = 4000
SEEDS = [42, 7, 13, 100, 200]

CONDITIONS = ["WAKING_ONLY", "SWS_ONLY", "SWS_THEN_REM"]


# ------------------------------------------------------------------ #
# Env / agent helpers                                                  #
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


def _make_agent(env: CausalGridWorldV2, sws_enabled: bool, rem_enabled: bool) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=0,
        novelty_bonus_weight=0.0,
        sws_enabled=sws_enabled,
        sws_consolidation_steps=8,
        sws_schema_weight=0.1,
        rem_enabled=rem_enabled,
        rem_attribution_steps=6,
    )
    return REEAgent(cfg)


def _action_onehot(a_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, a_idx] = 1.0
    return v


# ------------------------------------------------------------------ #
# Context-slot detection                                               #
# ------------------------------------------------------------------ #

def _active_slot_idx(agent: REEAgent, z_self: torch.Tensor,
                     z_world: torch.Tensor) -> int:
    """Determine which ContextMemory slot is most strongly activated by (z_self, z_world).

    Uses the same attention-query pipeline as ContextMemory.read(), but returns
    argmax over slots rather than a soft-mixed value. This is the active-context
    readout that conditions action selection.
    """
    with torch.no_grad():
        cm = agent.e1.context_memory
        state = torch.cat([z_self, z_world], dim=-1)  # [1, self+world]
        query = cm.query_proj(state)                   # [1, memory_dim]
        keys = cm.key_proj(cm.memory)                  # [num_slots, memory_dim]
        scores = torch.mm(query, keys.t()) / (cm.memory_dim ** 0.5)  # [1, num_slots]
        idx = int(scores.argmax(dim=-1).item())
    return idx


def _compute_slot_cosine_sim(agent: REEAgent) -> float:
    with torch.no_grad():
        mem = agent.e1.context_memory.memory
        n = mem.shape[0]
        normed = F.normalize(mem, dim=-1)
        sim = torch.mm(normed, normed.t())
        mask = ~torch.eye(n, dtype=torch.bool, device=mem.device)
        return float(sim[mask].mean().item())


# ------------------------------------------------------------------ #
# Action selection                                                     #
# ------------------------------------------------------------------ #

def _select_action_baseline(agent: REEAgent, z_world: torch.Tensor,
                             num_actions: int) -> Tuple[int, float]:
    """Argmin predicted harm over actions."""
    with torch.no_grad():
        best_a = 0
        best_h = float("inf")
        for a in range(num_actions):
            a_oh = _action_onehot(a, num_actions, z_world.device)
            zw_next = agent.e2.world_forward(z_world, a_oh)
            h = agent.e3.harm_eval(zw_next).mean().item()
            if h < best_h:
                best_h = h
                best_a = a
    return best_a, best_h


def _select_action_context_cond(agent: REEAgent, z_world: torch.Tensor,
                                 num_actions: int, slot_danger_score: float,
                                 base_thresh: float, context_beta: float
                                 ) -> Tuple[int, float, float]:
    """Context-conditioned harm threshold action selection.

    Effective threshold: base_thresh * (1 - context_beta * slot_danger_score).
    (Higher danger -> lower threshold -> more candidates filtered -> more cautious.)
    Filter candidates with predicted harm > threshold; among remaining pick argmin.
    If all candidates exceed threshold, fall back to unfiltered argmin (do SOMETHING).
    Returns (action_idx, chosen_harm, effective_threshold).
    """
    eff_thresh = base_thresh * max(0.1, 1.0 - context_beta * slot_danger_score)
    with torch.no_grad():
        harms = []
        for a in range(num_actions):
            a_oh = _action_onehot(a, num_actions, z_world.device)
            zw_next = agent.e2.world_forward(z_world, a_oh)
            h = agent.e3.harm_eval(zw_next).mean().item()
            harms.append(h)
        # Filter by threshold
        filtered = [(a, h) for a, h in enumerate(harms) if h <= eff_thresh]
        if filtered:
            best_a, best_h = min(filtered, key=lambda x: x[1])
        else:
            best_a = int(min(range(num_actions), key=lambda a: harms[a]))
            best_h = harms[best_a]
    return best_a, float(best_h), float(eff_thresh)


# ------------------------------------------------------------------ #
# Episode runner                                                       #
# ------------------------------------------------------------------ #

def _run_episode(
    agent: REEAgent,
    env: CausalGridWorldV2,
    steps: int,
    train: bool,
    is_dangerous_ep: bool,
    optimizer,
    harm_eval_opt,
    harm_buf_pos: List,
    harm_buf_neg: List,
    slot_danger_ema: List[float],
    use_context_cond: bool,
) -> Tuple[float, List[torch.Tensor], List[int]]:
    """Run single episode. Returns (harm_sum, z_world_list, slot_visits).
    Also updates slot_danger_ema in place if train=True.
    """
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    ep_harm = 0.0
    z_world_list: List[torch.Tensor] = []
    slot_visits: List[int] = []

    self_dim = agent.config.latent.self_dim

    for _step in range(steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        obs_harm = obs_dict.get("harm_obs", None)

        latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
        agent.clock.advance()
        z_self = latent.z_self.detach().clone()
        z_world = latent.z_world.detach().clone()
        z_world_list.append(z_world)

        # Active context slot (always compute for bookkeeping)
        slot_idx = _active_slot_idx(agent, z_self, z_world)
        slot_visits.append(slot_idx)

        if use_context_cond and not train:
            # Use slot_danger_score to condition action selection in eval
            # (score in [0,1] where 1 = strongly associated with DANGEROUS)
            danger = slot_danger_ema[slot_idx]
            action_idx, _, _ = _select_action_context_cond(
                agent, z_world, env.action_dim, danger,
                BASE_HARM_THRESHOLD, CONTEXT_BETA,
            )
        elif use_context_cond and train:
            # During training we also use the context-conditioned selection
            # so the agent's exposure matches its eval policy.
            danger = slot_danger_ema[slot_idx]
            action_idx, _, _ = _select_action_context_cond(
                agent, z_world, env.action_dim, danger,
                BASE_HARM_THRESHOLD, CONTEXT_BETA,
            )
        else:
            action_idx, _ = _select_action_baseline(agent, z_world, env.action_dim)

        action_oh = _action_onehot(action_idx, env.action_dim, agent.device)
        agent._last_action = action_oh

        _, harm_signal, done, _info, obs_dict = env.step(action_oh)
        is_harm = float(harm_signal) < 0
        if is_harm:
            ep_harm += abs(float(harm_signal))

        if train:
            # Update slot_danger_ema: target is 1.0 if dangerous episode, 0.0 if safe
            target = 1.0 if is_dangerous_ep else 0.0
            slot_danger_ema[slot_idx] = (
                (1.0 - SLOT_DANGER_EMA_ALPHA) * slot_danger_ema[slot_idx]
                + SLOT_DANGER_EMA_ALPHA * target
            )

            # E1 + E2 online training
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total = e1_loss + e2_loss
            if total.requires_grad:
                optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # harm_eval stratified training
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
                target_t = torch.cat([
                    torch.ones(k_pos, 1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                ], dim=0)
                pred = agent.e3.harm_eval_head(zw_b)
                h_loss = F.binary_cross_entropy_with_logits(pred, target_t)
                harm_eval_opt.zero_grad()
                h_loss.backward()
                harm_eval_opt.step()

        if done:
            break

    return ep_harm, z_world_list, slot_visits


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

    sws_en = condition in ("SWS_ONLY", "SWS_THEN_REM")
    rem_en = condition == "SWS_THEN_REM"
    use_context_cond = condition == "SWS_THEN_REM"   # DR-6 pathway only here

    env_safe = _make_env_safe(seed)
    env_dang = _make_env_dangerous(seed)
    agent = _make_agent(env_safe, sws_en, rem_en)

    standard_params = [p for n, p in agent.named_parameters()
                       if "harm_eval_head" not in n]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    optimizer = optim.Adam(standard_params, lr=1e-3)
    harm_eval_opt = optim.Adam(harm_eval_params, lr=1e-4)

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    num_slots = agent.e1.context_memory.num_slots
    slot_danger_ema: List[float] = [0.5] * num_slots

    per_ep_harm_safe: List[float] = []
    per_ep_harm_dang: List[float] = []
    slot_visit_safe_count: List[int] = [0] * num_slots
    slot_visit_dang_count: List[int] = [0] * num_slots
    sleep_passes = 0

    agent.train()
    for ep in range(training_episodes):
        block = ep // CONTEXT_SWITCH_EVERY
        is_safe_ep = (block % 2 == 0)
        env = env_safe if is_safe_ep else env_dang

        ep_harm, _z_list, slot_visits = _run_episode(
            agent, env, steps_per_episode,
            train=True,
            is_dangerous_ep=(not is_safe_ep),
            optimizer=optimizer,
            harm_eval_opt=harm_eval_opt,
            harm_buf_pos=harm_buf_pos,
            harm_buf_neg=harm_buf_neg,
            slot_danger_ema=slot_danger_ema,
            use_context_cond=use_context_cond,
        )
        harm_rate = ep_harm / steps_per_episode
        if is_safe_ep:
            per_ep_harm_safe.append(harm_rate)
            for s in slot_visits:
                slot_visit_safe_count[s] += 1
        else:
            per_ep_harm_dang.append(harm_rate)
            for s in slot_visits:
                slot_visit_dang_count[s] += 1

        # Trim online harm buffers
        if len(harm_buf_pos) > MAX_HARM_BUF:
            harm_buf_pos = harm_buf_pos[-MAX_HARM_BUF:]
        if len(harm_buf_neg) > MAX_HARM_BUF:
            harm_buf_neg = harm_buf_neg[-MAX_HARM_BUF:]

        # Sleep cycle
        if (sws_en or rem_en) and (ep + 1) % SLEEP_INTERVAL == 0 and ep > 0:
            if rem_en:
                _ = agent.run_sleep_cycle()
            else:
                _ = agent.run_sws_schema_pass()
            sleep_passes += 1

        if (ep + 1) % 50 == 0:
            print(f"  [train] label seed={seed} cond={condition} "
                  f"ep {ep+1}/{training_episodes} "
                  f"harm_safe_ema={(sum(per_ep_harm_safe[-10:])/max(len(per_ep_harm_safe[-10:]),1)):.4f} "
                  f"harm_dang_ema={(sum(per_ep_harm_dang[-10:])/max(len(per_ep_harm_dang[-10:]),1)):.4f}",
                  flush=True)

    # Slot danger separation signal: how differently SAFE and DANGEROUS visit slots
    safe_tot = float(sum(slot_visit_safe_count))
    dang_tot = float(sum(slot_visit_dang_count))
    if safe_tot > 0 and dang_tot > 0:
        safe_dist = [c / safe_tot for c in slot_visit_safe_count]
        dang_dist = [c / dang_tot for c in slot_visit_dang_count]
        # L1 distance between visit distributions, in [0, 2]
        slot_separation = float(sum(abs(s - d) for s, d in zip(safe_dist, dang_dist)))
    else:
        slot_separation = 0.0

    final_slot_sim = _compute_slot_cosine_sim(agent)

    # Evaluation
    agent.eval()
    eval_harm_safe: List[float] = []
    eval_harm_dang: List[float] = []
    eval_z_safe: List[torch.Tensor] = []
    eval_z_dang: List[torch.Tensor] = []

    for _ in range(eval_episodes_each):
        h_s, zs, _ = _run_episode(
            agent, env_safe, steps_per_episode,
            train=False, is_dangerous_ep=False,
            optimizer=optimizer, harm_eval_opt=harm_eval_opt,
            harm_buf_pos=harm_buf_pos, harm_buf_neg=harm_buf_neg,
            slot_danger_ema=slot_danger_ema,
            use_context_cond=use_context_cond,
        )
        eval_harm_safe.append(h_s / steps_per_episode)
        eval_z_safe.extend(zs)

    for _ in range(eval_episodes_each):
        h_d, zd, _ = _run_episode(
            agent, env_dang, steps_per_episode,
            train=False, is_dangerous_ep=True,
            optimizer=optimizer, harm_eval_opt=harm_eval_opt,
            harm_buf_pos=harm_buf_pos, harm_buf_neg=harm_buf_neg,
            slot_danger_ema=slot_danger_ema,
            use_context_cond=use_context_cond,
        )
        eval_harm_dang.append(h_d / steps_per_episode)
        eval_z_dang.extend(zd)

    with torch.no_grad():
        n_samp = min(len(eval_z_safe), len(eval_z_dang), 200)
        if n_samp > 0:
            zs_s = torch.cat(eval_z_safe[:n_samp], dim=0)
            zd_s = torch.cat(eval_z_dang[:n_samp], dim=0)
            he_safe = float(agent.e3.harm_eval(zs_s).mean().item())
            he_dang = float(agent.e3.harm_eval(zd_s).mean().item())
        else:
            he_safe = 0.0
            he_dang = 0.0
    harm_discrim = he_dang - he_safe

    harm_safe = sum(eval_harm_safe) / max(1, len(eval_harm_safe))
    harm_dang = sum(eval_harm_dang) / max(1, len(eval_harm_dang))

    if verbose:
        print(f"  [seed={seed} {condition}] "
              f"slot_sim={final_slot_sim:.4f} "
              f"slot_sep={slot_separation:.3f} "
              f"harm_safe={harm_safe:.4f} "
              f"harm_dang={harm_dang:.4f} "
              f"discrim={harm_discrim:.4f} "
              f"sleep_passes={sleep_passes}",
              flush=True)

    verdict = "PASS" if (harm_dang < 0.04 and harm_safe < 0.04) else "FAIL"
    print(f"verdict: {verdict}", flush=True)

    return {
        "seed": seed,
        "condition": condition,
        "slot_cosine_sim": float(final_slot_sim),
        "slot_separation": float(slot_separation),
        "harm_rate_safe": float(harm_safe),
        "harm_rate_dangerous": float(harm_dang),
        "harm_discrimination": float(harm_discrim),
        "harm_eval_safe": float(he_safe),
        "harm_eval_dangerous": float(he_dang),
        "slot_danger_ema": [float(x) for x in slot_danger_ema],
        "slot_visit_safe_count": slot_visit_safe_count,
        "slot_visit_dang_count": slot_visit_dang_count,
        "train_harm_safe_final": float(sum(per_ep_harm_safe[-20:]) / max(1, len(per_ep_harm_safe[-20:]))),
        "train_harm_dang_final": float(sum(per_ep_harm_dang[-20:]) / max(1, len(per_ep_harm_dang[-20:]))),
        "sleep_passes": sleep_passes,
    }


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.dry_run:
        print("Smoke test: seed=42, 3 train eps + 2 eval eps per condition")
        for cond in CONDITIONS:
            print(f"Seed 42 Condition {cond}")
            r = _run_condition(
                seed=42, condition=cond,
                training_episodes=3,
                steps_per_episode=30,
                eval_episodes_each=2,
                verbose=False,
            )
            print(f"  {cond}: slot_sim={r['slot_cosine_sim']:.4f} "
                  f"slot_sep={r['slot_separation']:.3f} "
                  f"harm_safe={r['harm_rate_safe']:.4f} "
                  f"harm_dang={r['harm_rate_dangerous']:.4f} "
                  f"sleep_passes={r['sleep_passes']}")
        print("Smoke test PASSED")
        return

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
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
        print(f"Seed {seed}")
        for cond in CONDITIONS:
            print(f"Seed {seed} Condition {cond}")
            r = _run_condition(
                seed=seed, condition=cond,
                training_episodes=TRAINING_EPISODES,
                steps_per_episode=STEPS_PER_EPISODE,
                eval_episodes_each=EVAL_EPISODES_EACH,
            )
            all_results.append(r)

    def by_cond(c):
        return [r for r in all_results if r["condition"] == c]

    waking = by_cond("WAKING_ONLY")
    sws = by_cond("SWS_ONLY")
    sws_r = by_cond("SWS_THEN_REM")

    # C1: slot_cosine_sim(SWS_THEN_REM) < slot_cosine_sim(WAKING_ONLY) >=3/5
    c1_wins = sum(1 for w, s in zip(waking, sws_r)
                  if s["slot_cosine_sim"] < w["slot_cosine_sim"])
    c1 = c1_wins >= 3

    # C2: harm_rate_dangerous(SWS_THEN_REM) < harm_rate_dangerous(WAKING_ONLY) >=3/5
    c2_wins = sum(1 for w, s in zip(waking, sws_r)
                  if s["harm_rate_dangerous"] < w["harm_rate_dangerous"])
    c2 = c2_wins >= 3

    # C3: harm_rate_safe < 0.05 in all conditions, >=3/5 of all seed-condition runs
    c3_count = sum(1 for r in all_results if r["harm_rate_safe"] < 0.05)
    c3 = c3_count >= (len(SEEDS) * len(CONDITIONS) * 3 // 5)

    # C4: slot_separation > 0.3 in >=3/5 SWS_THEN_REM seeds
    c4_wins = sum(1 for s in sws_r if s["slot_separation"] > 0.3)
    c4 = c4_wins >= 3

    outcome = "PASS" if (c1 and c2) else "FAIL"

    summary = {
        "c1_slot_differentiation_vs_waking": {
            "wins": c1_wins, "pass": c1,
            "desc": "slot_sim(SWS_THEN_REM) < slot_sim(WAKING_ONLY) in >=3/5 seeds",
        },
        "c2_behavioral_harm_reduction": {
            "wins": c2_wins, "pass": c2,
            "desc": "harm_dang(SWS_THEN_REM) < harm_dang(WAKING_ONLY) in >=3/5 seeds",
        },
        "c3_safe_specificity": {
            "count_ok": c3_count, "pass": c3,
            "desc": "harm_rate_safe < 0.05 in >=60% of runs",
        },
        "c4_slot_separation": {
            "wins": c4_wins, "pass": c4,
            "desc": "slot_separation > 0.3 in >=3/5 SWS_THEN_REM seeds",
        },
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
        "timestamp_utc": ts,
        "outcome": outcome,
        "evidence_direction": "supports" if outcome == "PASS" else "weakens",
        "evidence_direction_per_claim": {
            "SD-017": "supports" if (c1 and c2) else ("mixed" if c1 else "weakens"),
            "ARC-045": "supports" if (c1 and c2) else ("mixed" if c1 else "weakens"),
            "MECH-166": "supports" if c1 else "weakens",
        },
        "pass_criteria_summary": summary,
        "per_seed_results": all_results,
        "supersedes": "V3-EXQ-242",
        "config": {
            "conditions": CONDITIONS,
            "sleep_interval": SLEEP_INTERVAL,
            "context_switch_every": CONTEXT_SWITCH_EVERY,
            "training_episodes": TRAINING_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "eval_episodes_each": EVAL_EPISODES_EACH,
            "seeds": SEEDS,
            "base_harm_threshold": BASE_HARM_THRESHOLD,
            "context_beta": CONTEXT_BETA,
            "slot_danger_ema_alpha": SLOT_DANGER_EMA_ALPHA,
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
