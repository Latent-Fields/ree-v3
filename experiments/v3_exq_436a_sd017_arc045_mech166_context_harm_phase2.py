#!/opt/local/bin/python3
"""
V3-EXQ-436a -- SD-017 + ARC-045 + MECH-166 Context-Conditioned Harm Threshold:
SLEEP DRIVER: K=1 single-fire (SleepLoopManager, default cycle_every_k_episodes=1, fires every episode)
                Phase 2 substrate retest with per-seed distribution diagnostics.

SUPERSEDES: V3-EXQ-436. Original 436 ran on the pre-Phase-2 substrate; under the
SD-016 attention-uniformity confound + missing per-stream V_s + missing anchor sets,
the slot_cosine_sim / harm_rate / slot_separation signatures could not be measured
against a stable substrate baseline. This retest applies the validated Phase 2
substrate template (V3-EXQ-265a PASS 2026-05-09T20:12Z) to the same 3-condition x
5-seed design.

MECHANISMS UNDER TEST (multi-claim, claim_ids re-evaluated from scratch per
CLAUDE.md accuracy rule):
  SD-017  (sleep_phase.minimal_sleep_infrastructure_v3) -- slot dynamics differ
          between sleep arms (SWS_THEN_REM) and waking baseline (WAKING_ONLY)
          under the Phase 2 substrate.
  ARC-045 (cross-frequency bidirectional flow signature) -- SWS_THEN_REM
          produces a non-trivial slot_separation between SAFE and DANGEROUS
          context visits (the action-pathway signal that the context-
          conditioned harm threshold can condition on).
  MECH-166 (context-conditional harm-threshold) -- harm_rate_dangerous in
          SWS_THEN_REM differs from harm_rate_dangerous in WAKING_ONLY by more
          than tolerance, giving the behavioural footprint of context-
          conditioning. Tagged because the experiment directly measures it
          (the SWS_THEN_REM arm activates use_context_cond=True in
          _select_action_context_cond).

PHASE 2 SUBSTRATE TEMPLATE (validated by V3-EXQ-265a PASS, mechanically applied
across the GAP-2 successor cohort 265a / 500a / 503a / 418c / 436a):
  sd016_writepath_mode = "off"             # A2_div_only; div loss alone
  sd016_diversification_weight = 0.5        # matches EXQ-418e LAMBDA_DIVERSIFY
  use_per_stream_vs = True                  # MECH-269 Phase 1
  use_anchor_sets = True                    # MECH-269 Phase 2 (ii) dual-trace
  use_sd039_anchor_payload = True           # SD-039 substrate-side payload
  use_sleep_loop = True                     # SD-017 SleepLoopManager (sleep arms only)
The 5 substrate flags are CONSTANT across all 3 conditions; only sleep flags
(sws_enabled / rem_enabled / use_sleep_loop) toggle per-condition (sleep arms
get them ON; WAKING_ONLY gets them OFF). 265a's smoke confirmed Phase 2 stack
+ sleep OFF preserves backward compat; this experiment exercises the full stack.

DESIGN: 3 conditions x 5 seeds = 15 runs.
  WAKING_ONLY:  baseline -- no sleep, no context-cond threshold.
                Phase 2 substrate flags ON (constant across conditions).
  SWS_ONLY:     SWS schema pass every SLEEP_INTERVAL eps, no REM, no context-cond.
                Phase 2 substrate flags ON; use_sleep_loop=True; sws_enabled=True.
  SWS_THEN_REM: full sleep cycle (SWS then REM) every SLEEP_INTERVAL eps,
                PLUS context-conditioned harm threshold in action selection
                (DR-6 pathway). Phase 2 substrate flags ON; use_sleep_loop=True;
                sws_enabled=True; rem_enabled=True; use_context_cond=True.

KEY 436a-SPECIFIC REQUIREMENT: per-seed distribution diagnostics. The 265a result
showed informative cross-seed heterogeneity (seed 42 sleep ADDS diversity by
+0.090; seed 49 SATURATES near-tie at +0.007; seed 56 sleep COLLAPSES at -0.194).
Mean-only summaries would mask the seed-56-style collapse pattern in a 5-seed
cohort. The aggregated.per_seed block emits per-seed slot_cosine_sim,
slot_separation, and harm-rate metrics for each (condition, seed) cell PLUS the
SWS_THEN_REM-vs-WAKING_ONLY signed differences that drive the C1/C2/C4
acceptance. C4 |diff| tolerance tightened to 0.03 (vs 265a's 0.05) to retain
sensitivity to bimodal cross-seed distributions.

ACCEPTANCE CRITERIA (multi-claim, signed-difference shape per 265a template):
  C1 (SD-017): slot_cosine_sim signed |diff| between SWS_THEN_REM and
              WAKING_ONLY > C1_DIFF_TOLERANCE (0.03) in >= 2/5 seeds.
              Either direction informative (sleep adds OR collapses slot
              identity vs waking-only attractor). Replaces original 436's
              direction-only check slot_sim(SWS_THEN_REM) < slot_sim(WAKING_ONLY).
  C2 (MECH-166): harm_rate_dangerous signed |diff| between SWS_THEN_REM and
              WAKING_ONLY > C2_DIFF_TOLERANCE (0.005) in >= 2/5 seeds.
              Either direction informative (context-conditioning either
              reduces dangerous-context harm via cautious filtering OR raises
              it if the threshold modulation overshoots). Replaces original
              436's direction-only check harm_dang(SWS_THEN_REM) < harm_dang(WAKING_ONLY).
  C3 (baseline preservation): harm_rate_safe < 0.05 across all 15 runs in
              >= 9/15 (60%). Magnitude check, preserved from original 436.
  C4 (ARC-045): slot_separation in SWS_THEN_REM > 0.3 in >= 2/5 seeds. Magnitude
              check on the cross-frequency signature; preserved from original
              436 (slot_separation is in [0, 2]; > 0.3 = SAFE and DANGEROUS
              contexts visit measurably different slot subsets, i.e. the
              action pathway has a real signal to condition on).

PASS: C1 AND C2 (the multi-claim signed-difference behavioural test).
  C3 / C4 are secondary and reported but do not gate PASS/FAIL (they preserve
  the original 436 magnitude-check shape for cross-experiment continuity).

INTERPRETATION GRID (for the discussant reviewing this):
  C1 PASS, C2 PASS              -> Phase 2 substrate unblocks SD-017 + MECH-166
                                   end-to-end; ARC-045 magnitude preserved if
                                   C4 PASS. Multi-claim "supports" verdict for
                                   SD-017, MECH-166; ARC-045 "supports" iff C4.
                                   Roll sleep_substrate_plan.md GAP-2 forward
                                   per the owner-EXQ list.
  C1 PASS, C2 FAIL              -> Slot dynamics differ between sleep arms
                                   under Phase 2, but context-conditioning
                                   does NOT produce a behavioural harm-rate
                                   footprint vs waking baseline. SD-017 supports;
                                   MECH-166 weakens; ARC-045 supports iff C4.
                                   Likely candidates: slot_danger_score EMA too
                                   slow to track context switches (alpha=0.05);
                                   or context_beta=0.8 modulation strength
                                   insufficient to shift action selection given
                                   the harm-rate distribution; or the action
                                   pathway has a signal (C4 PASS) but it doesn't
                                   route to enough of the actual harm-bearing
                                   trajectories. Investigate via metric inspection
                                   first; route to /diagnose-errors only if no
                                   clear architectural reading emerges.
  C1 FAIL, C2 PASS              -> Behavioural harm-rate footprint differs but
                                   slot_cosine_sim does not. Possible: SWS+REM
                                   passes drive a behavioural shift via REM
                                   attribution / residue terrain rather than
                                   via slot identity. SD-017 weakens (slot
                                   metric was the primary signal); MECH-166
                                   supports; ARC-045 supports iff C4. The
                                   context-conditioned threshold may be doing
                                   work even when the slot-identity proxy is
                                   flat. Worth a metric redesign successor
                                   (436b) to find the right slot signature.
  C1 FAIL, C2 FAIL              -> Phase 2 substrate did NOT recover the
                                   436-cohort signal. Both slot dynamics and
                                   behavioural footprint are bit-equivalent
                                   between sleep arms and waking. SD-017,
                                   MECH-166 weaken. ARC-045 supports iff C4
                                   (the action pathway can still have signal
                                   even if context-conditioning produces no
                                   measurable change). Strongest signal that
                                   either the context-conditioned-threshold
                                   pathway needs structural redesign or the
                                   3-condition x 5-seed power is insufficient
                                   to detect the effect. Compare to 265a
                                   per-seed heterogeneity to assess whether
                                   the failure is methodological or architectural.
  C3 FAIL only                  -> Baseline regression: WAKING_ONLY safe-context
                                   harm rate exceeds 0.05 in >=40% of runs.
                                   Phase 2 substrate may be destabilising the
                                   baseline policy (encoder warmup interacting
                                   poorly with anchor_set V_s gating). Flag to
                                   the sleep-substrate cluster regardless of
                                   C1/C2 outcome; existing per-seed diagnostics
                                   should expose which arm carries the
                                   regression.

claim_ids: ["SD-017", "ARC-045", "MECH-166"]
experiment_purpose: "evidence"
"""

import sys
import random
import json
import time
import argparse
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
from experiment_protocol import emit_outcome


EXPERIMENT_TYPE = "v3_exq_436a_sd017_arc045_mech166_context_harm_phase2"
QUEUE_ID = "V3-EXQ-436a"
SUPERSEDES = "V3-EXQ-436"
CLAIM_IDS = ["SD-017", "ARC-045", "MECH-166"]
EXPERIMENT_PURPOSE = "evidence"

# Pre-registered thresholds
BASE_HARM_THRESHOLD = 0.05       # filter actions whose predicted harm exceeds this
CONTEXT_BETA = 0.8                # danger-score modulation strength
SLOT_DANGER_EMA_ALPHA = 0.05      # slot_danger_score EMA update rate

# Phase 2 substrate template (validated by V3-EXQ-265a PASS 2026-05-09T20:12Z).
SD016_DIVERSIFICATION_WEIGHT = 0.5  # matches EXQ-418e LAMBDA_DIVERSIFY / EXQ-265a

# Acceptance tolerances (signed-|diff| shape per 265a template).
C1_DIFF_TOLERANCE = 0.03   # tightened from 265a's 0.05 per 436a per-seed-sensitivity ask
C2_DIFF_TOLERANCE = 0.005  # harm rates are typically <0.05; calibrate from smoke if needed
C3_HARM_SAFE_THRESHOLD = 0.05
C4_SLOT_SEPARATION_THRESHOLD = 0.3
N_SEEDS_FOR_C1 = 2  # >= 2/5 seeds (per 265a 2/3 pattern, scaled to 5-seed cohort)
N_SEEDS_FOR_C2 = 2
N_SEEDS_FOR_C4 = 2

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


def _make_agent(env: CausalGridWorldV2, sws_enabled: bool, rem_enabled: bool,
                use_sleep_loop: bool) -> REEAgent:
    """Phase 2 substrate stack: SD-016 div loss + MECH-269 V_s + anchor sets +
    SD-039 payload, with SD-017 sleep machinery toggled by caller. Five
    substrate flags constant across all 3 conditions; only sleep flags vary.

    anchor_sets requires per_stream_vs ON (precondition raised in
    HippocampalModule). They are wired together here, matching 265a / 500a.
    """
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
        # Phase 2 substrate template (5 flags, mechanically applied; constant
        # across all 3 conditions in this experiment).
        sd016_writepath_mode="off",
        sd016_diversification_weight=SD016_DIVERSIFICATION_WEIGHT,
        use_per_stream_vs=True,
        use_anchor_sets=True,
        use_sd039_anchor_payload=True,
        # SD-017 sleep phases (toggle per condition).
        sws_enabled=sws_enabled,
        sws_consolidation_steps=8,
        sws_schema_weight=0.1,
        rem_enabled=rem_enabled,
        rem_attribution_steps=6,
        # SD-017 SleepLoopManager (Phase A scaffolding wraps run_sleep_cycle);
        # ON in sleep arms, OFF in WAKING_ONLY.
        use_sleep_loop=use_sleep_loop,
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
    """Determine which ContextMemory slot is most strongly activated by
    (z_self, z_world). Argmax over slots vs ContextMemory.read()'s soft mix.
    """
    with torch.no_grad():
        cm = agent.e1.context_memory
        state = torch.cat([z_self, z_world], dim=-1)
        query = cm.query_proj(state)
        keys = cm.key_proj(cm.memory)
        scores = torch.mm(query, keys.t()) / (cm.memory_dim ** 0.5)
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
    Higher danger -> lower threshold -> more candidates filtered -> more cautious.
    Filter candidates with predicted harm > threshold; pick argmin among remaining.
    Fall back to unfiltered argmin if all candidates exceed threshold.
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
    Updates slot_danger_ema in place when train=True.
    """
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    ep_harm = 0.0
    z_world_list: List[torch.Tensor] = []
    slot_visits: List[int] = []

    for _step in range(steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        obs_harm = obs_dict.get("harm_obs", None)

        latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
        agent.clock.advance()
        z_self = latent.z_self.detach().clone()
        z_world = latent.z_world.detach().clone()
        z_world_list.append(z_world)

        slot_idx = _active_slot_idx(agent, z_self, z_world)
        slot_visits.append(slot_idx)

        if use_context_cond:
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
            target = 1.0 if is_dangerous_ep else 0.0
            slot_danger_ema[slot_idx] = (
                (1.0 - SLOT_DANGER_EMA_ALPHA) * slot_danger_ema[slot_idx]
                + SLOT_DANGER_EMA_ALPHA * target
            )

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
    use_sleep_loop = sws_en or rem_en  # ON for any sleep arm; OFF for WAKING_ONLY
    use_context_cond = condition == "SWS_THEN_REM"   # DR-6 pathway only here

    env_safe = _make_env_safe(seed)
    env_dang = _make_env_dangerous(seed)
    agent = _make_agent(env_safe, sws_en, rem_en, use_sleep_loop)

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

        if len(harm_buf_pos) > MAX_HARM_BUF:
            harm_buf_pos = harm_buf_pos[-MAX_HARM_BUF:]
        if len(harm_buf_neg) > MAX_HARM_BUF:
            harm_buf_neg = harm_buf_neg[-MAX_HARM_BUF:]

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

    safe_tot = float(sum(slot_visit_safe_count))
    dang_tot = float(sum(slot_visit_dang_count))
    if safe_tot > 0 and dang_tot > 0:
        safe_dist = [c / safe_tot for c in slot_visit_safe_count]
        dang_dist = [c / dang_tot for c in slot_visit_dang_count]
        slot_separation = float(sum(abs(s - d) for s, d in zip(safe_dist, dang_dist)))
    else:
        slot_separation = 0.0

    final_slot_sim = _compute_slot_cosine_sim(agent)

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

    # Per-condition verdict (preserves original 436's harm-magnitude check for
    # progress-instrumentation / runner-ETA purposes; the experiment-level PASS
    # is computed at main()-time across all 15 runs via the C1/C2 acceptance).
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
        print("[DRY RUN] Phase 2 substrate + 3-condition smoke (seed=42, 3 train eps each)")
        smoke_ok = True
        smoke_results = []
        try:
            for cond in CONDITIONS:
                print(f"Seed 42 Condition {cond}", flush=True)
                r = _run_condition(
                    seed=42, condition=cond,
                    training_episodes=3,
                    steps_per_episode=30,
                    eval_episodes_each=2,
                    verbose=False,
                )
                smoke_results.append(r)
                print(f"  {cond}: slot_sim={r['slot_cosine_sim']:.4f} "
                      f"slot_sep={r['slot_separation']:.3f} "
                      f"harm_safe={r['harm_rate_safe']:.4f} "
                      f"harm_dang={r['harm_rate_dangerous']:.4f} "
                      f"sleep_passes={r['sleep_passes']}")

            # Phase 2 substrate wiring sanity: sleep arms must report sleep_passes > 0
            # at sufficient training episode count (smoke runs 3 eps so SLEEP_INTERVAL=10
            # never triggers; but the agent build must accept the 5 Phase 2 flags +
            # use_sleep_loop without raising). The successful return from _run_condition
            # is the wiring check. We additionally assert that all 3 conditions returned
            # the expected per-seed metric keys.
            required_keys = {"slot_cosine_sim", "slot_separation",
                             "harm_rate_safe", "harm_rate_dangerous"}
            for r in smoke_results:
                missing = required_keys - set(r.keys())
                if missing:
                    print(f"  [SMOKE] FAIL: condition {r['condition']} missing keys {missing}")
                    smoke_ok = False

            if smoke_ok:
                print("[DRY RUN] PASS - Phase 2 stack + 3 conditions wire correctly; "
                      "per-seed metrics populate as expected")
            else:
                print("[DRY RUN] FAIL - check above for missing metric keys")
        except Exception as exc:
            print(f"[DRY RUN] FAIL - exception during smoke: {exc!r}")
            smoke_ok = False

        emit_outcome(
            outcome="PASS" if smoke_ok else "FAIL",
            manifest_path=None,
        )
        sys.exit(0 if smoke_ok else 1)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        script_dir = Path(__file__).resolve().parents[1]
        out_dir = (script_dir.parent / "REE_assembly" / "evidence"
                   / "experiments")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"{QUEUE_ID} {EXPERIMENT_TYPE}", flush=True)

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

    elapsed = time.time() - t0

    def by_cond(c):
        return [r for r in all_results if r["condition"] == c]

    waking = by_cond("WAKING_ONLY")
    sws = by_cond("SWS_ONLY")
    sws_r = by_cond("SWS_THEN_REM")

    # Per-seed signed-difference diagnostics. Pairs WAKING_ONLY (baseline) with
    # SWS_THEN_REM (sleep + context-cond) at matched seed indices.
    per_seed_diff: Dict[str, Dict] = {}
    for w_r, s_r in zip(waking, sws_r):
        seed = w_r["seed"]
        slot_sim_diff = s_r["slot_cosine_sim"] - w_r["slot_cosine_sim"]
        harm_dang_diff = s_r["harm_rate_dangerous"] - w_r["harm_rate_dangerous"]
        harm_safe_diff = s_r["harm_rate_safe"] - w_r["harm_rate_safe"]
        slot_sep_diff = s_r["slot_separation"] - w_r["slot_separation"]
        per_seed_diff[str(seed)] = {
            "seed": seed,
            "waking_slot_cosine_sim": w_r["slot_cosine_sim"],
            "sws_then_rem_slot_cosine_sim": s_r["slot_cosine_sim"],
            "slot_cosine_sim_signed_diff": slot_sim_diff,
            "slot_cosine_sim_abs_diff": abs(slot_sim_diff),
            "slot_cosine_sim_passes_C1": abs(slot_sim_diff) > C1_DIFF_TOLERANCE,
            "waking_harm_rate_dangerous": w_r["harm_rate_dangerous"],
            "sws_then_rem_harm_rate_dangerous": s_r["harm_rate_dangerous"],
            "harm_rate_dangerous_signed_diff": harm_dang_diff,
            "harm_rate_dangerous_abs_diff": abs(harm_dang_diff),
            "harm_rate_dangerous_passes_C2": abs(harm_dang_diff) > C2_DIFF_TOLERANCE,
            "waking_harm_rate_safe": w_r["harm_rate_safe"],
            "sws_then_rem_harm_rate_safe": s_r["harm_rate_safe"],
            "harm_rate_safe_signed_diff": harm_safe_diff,
            "waking_slot_separation": w_r["slot_separation"],
            "sws_then_rem_slot_separation": s_r["slot_separation"],
            "slot_separation_signed_diff": slot_sep_diff,
            "sws_then_rem_slot_separation_passes_C4": s_r["slot_separation"] > C4_SLOT_SEPARATION_THRESHOLD,
        }

    # C1: SD-017 -- slot_cosine_sim signed |diff| > C1_DIFF_TOLERANCE in >= 2/5 seeds.
    c1_count = sum(1 for d in per_seed_diff.values() if d["slot_cosine_sim_passes_C1"])
    c1_pass = c1_count >= N_SEEDS_FOR_C1

    # C2: MECH-166 -- harm_rate_dangerous signed |diff| > C2_DIFF_TOLERANCE in >= 2/5 seeds.
    c2_count = sum(1 for d in per_seed_diff.values() if d["harm_rate_dangerous_passes_C2"])
    c2_pass = c2_count >= N_SEEDS_FOR_C2

    # C3: harm_rate_safe < 0.05 across all 15 runs in >= 9/15 (60%).
    c3_count = sum(1 for r in all_results if r["harm_rate_safe"] < C3_HARM_SAFE_THRESHOLD)
    c3_pass = c3_count >= (len(SEEDS) * len(CONDITIONS) * 3 // 5)

    # C4: ARC-045 -- slot_separation in SWS_THEN_REM > 0.3 in >= 2/5 seeds.
    c4_count = sum(1 for d in per_seed_diff.values() if d["sws_then_rem_slot_separation_passes_C4"])
    c4_pass = c4_count >= N_SEEDS_FOR_C4

    # PASS = C1 AND C2 (the multi-claim signed-difference behavioural test).
    # C3 / C4 are reported but do not gate PASS/FAIL.
    outcome = "PASS" if (c1_pass and c2_pass) else "FAIL"

    # Per-claim direction. SD-017 -> C1; MECH-166 -> C2; ARC-045 -> C4.
    def _direction(passed: bool) -> str:
        return "supports" if passed else "weakens"

    evidence_direction_per_claim = {
        "SD-017": _direction(c1_pass),
        "ARC-045": _direction(c4_pass),
        "MECH-166": _direction(c2_pass),
    }

    if all(v == "supports" for v in evidence_direction_per_claim.values()):
        evidence_direction = "supports"
    elif all(v == "weakens" for v in evidence_direction_per_claim.values()):
        evidence_direction = "weakens"
    else:
        evidence_direction = "mixed"

    summary = {
        "C1_sd017_slot_cosine_sim_signed_diff": {
            "tolerance": C1_DIFF_TOLERANCE,
            "n_seeds_required": N_SEEDS_FOR_C1,
            "n_seeds_passed": c1_count,
            "pass": c1_pass,
            "desc": ("SD-017: slot_cosine_sim signed |diff| between SWS_THEN_REM "
                     "and WAKING_ONLY > 0.03 in >= 2/5 seeds. Either direction "
                     "informative."),
        },
        "C2_mech166_harm_rate_dangerous_signed_diff": {
            "tolerance": C2_DIFF_TOLERANCE,
            "n_seeds_required": N_SEEDS_FOR_C2,
            "n_seeds_passed": c2_count,
            "pass": c2_pass,
            "desc": ("MECH-166: harm_rate_dangerous signed |diff| between "
                     "SWS_THEN_REM and WAKING_ONLY > 0.005 in >= 2/5 seeds. "
                     "Either direction informative."),
        },
        "C3_baseline_preservation_harm_safe": {
            "threshold": C3_HARM_SAFE_THRESHOLD,
            "n_runs_required": len(SEEDS) * len(CONDITIONS) * 3 // 5,
            "n_runs_passed": c3_count,
            "pass": c3_pass,
            "desc": ("Baseline preservation: harm_rate_safe < 0.05 in >= 60% "
                     "of all 15 runs. Magnitude check; secondary."),
        },
        "C4_arc045_slot_separation_threshold": {
            "threshold": C4_SLOT_SEPARATION_THRESHOLD,
            "n_seeds_required": N_SEEDS_FOR_C4,
            "n_seeds_passed": c4_count,
            "pass": c4_pass,
            "desc": ("ARC-045: slot_separation in SWS_THEN_REM > 0.3 in "
                     ">= 2/5 seeds. Cross-frequency signature; secondary."),
        },
    }

    print(f"\nOutcome: {outcome}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  Per-claim direction: {evidence_direction_per_claim}")

    output = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "supersedes": SUPERSEDES,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": outcome,
        "result": outcome,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": evidence_direction_per_claim,
        "pass_criteria_summary": summary,
        "aggregated": {
            "per_seed": per_seed_diff,
            "n_seeds_passed": {
                "C1": c1_count,
                "C2": c2_count,
                "C3_runs": c3_count,
                "C4": c4_count,
            },
        },
        "per_seed_results": all_results,
        "registered_thresholds": {
            "C1_DIFF_TOLERANCE": C1_DIFF_TOLERANCE,
            "C2_DIFF_TOLERANCE": C2_DIFF_TOLERANCE,
            "C3_HARM_SAFE_THRESHOLD": C3_HARM_SAFE_THRESHOLD,
            "C4_SLOT_SEPARATION_THRESHOLD": C4_SLOT_SEPARATION_THRESHOLD,
            "N_SEEDS_FOR_C1": N_SEEDS_FOR_C1,
            "N_SEEDS_FOR_C2": N_SEEDS_FOR_C2,
            "N_SEEDS_FOR_C4": N_SEEDS_FOR_C4,
            "BASE_HARM_THRESHOLD": BASE_HARM_THRESHOLD,
            "CONTEXT_BETA": CONTEXT_BETA,
            "SLOT_DANGER_EMA_ALPHA": SLOT_DANGER_EMA_ALPHA,
        },
        "config": {
            "conditions": CONDITIONS,
            "sleep_interval": SLEEP_INTERVAL,
            "context_switch_every": CONTEXT_SWITCH_EVERY,
            "training_episodes": TRAINING_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "eval_episodes_each": EVAL_EPISODES_EACH,
            "seeds": SEEDS,
            "env_safe_num_hazards": 1,
            "env_dangerous_num_hazards": 8,
            # Phase 2 substrate template flags (recorded for indexer audit).
            "sd016_writepath_mode": "off",
            "sd016_diversification_weight": SD016_DIVERSIFICATION_WEIGHT,
            "use_per_stream_vs": True,
            "use_anchor_sets": True,
            "use_sd039_anchor_payload": True,
            "use_sleep_loop_in_sleep_arms": True,
        },
        "elapsed_seconds": elapsed,
        "notes": (
            "Phase 2 retest of EXP-0106 SD-017 + ARC-045 + MECH-166 context-"
            "conditioned harm threshold. Successor to V3-EXQ-436 under the "
            "validated 5-flag substrate template (SD-016 div loss + MECH-269 "
            "per-stream V_s + anchor sets + SD-039 payload + SD-017 sleep "
            "loop). Multi-claim, signed-|diff| acceptance shape per the 265a "
            "template; per-seed distribution diagnostics emitted in "
            "aggregated.per_seed to expose seed-level heterogeneity."
        ),
    }

    out_file = out_dir / f"{run_id}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Output written to: {out_file}")
    print(f"Elapsed: {elapsed:.1f}s")

    return outcome, str(out_file)


if __name__ == "__main__":
    _outcome, _out_path = main()
    _outcome_clean = str(_outcome).upper() if _outcome in ("PASS", "FAIL") else "FAIL"
    emit_outcome(
        outcome=_outcome_clean,
        manifest_path=_out_path,
    )
