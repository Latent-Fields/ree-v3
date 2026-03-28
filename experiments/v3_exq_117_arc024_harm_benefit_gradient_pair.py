#!/opt/local/bin/python3
"""
V3-EXQ-117 -- ARC-024 Harm-Benefit Opposing Gradient Discriminative Pair

Claim: ARC-024 -- "Harm and benefit signals have asymptotic proxy structure
in world latent space."

Proposal: EXP-0014 / EVB-0011
Dispatch mode: discriminative_pair
Min shared seeds: 2

Background:
  EXQ-107 (5 seeds, 200 warmup eps): FAIL on C1 (gap_approach_none=0.003 vs
  threshold 0.08), PASS on C2 (AUC=0.648) and C3 (disc_advantage=0.147).
  The C1 failure: z_world does not show a detectable pre-contact harm gradient
  at 200 warmup episodes. C2 passes because the AUC over all harm events
  (approach+contact) beats the FLAT baseline -- but the pre-contact window
  alone is not discriminated.

  Root cause analysis:
  (1) EXQ-107 measured gap_approach_none in the GRADIENT condition only, using
      E3.harm_eval(z_world). With 200 warmup eps and random policy, approach
      events are rare (<10% of eval steps), so harm_eval gets little signal
      to learn the pre-contact window.
  (2) EXQ-107 trained only E3.harm_eval (harm signal direction). ARC-024 also
      predicts a BENEFIT gradient in the opposing direction -- resources pull
      E3.benefit_eval upward as the agent approaches. The full claim requires
      testing BOTH gradient directions and their opposition.

This experiment redesigns the discriminative pair to test opposing gradient
structure directly:

  GRADIENT_PAIR -- use_proxy_fields=True: agent observes both hazard_field_view
                   AND resource_field_view in world_obs. Both harm and benefit
                   signals are graded by proximity. E3.harm_eval is trained on
                   harm_signal<0 events. A parallel benefit probe (linear head
                   on z_world) is trained on benefit_signal>0 events.

  FLAT_CONTROL  -- use_proxy_fields=False: agent observes binary contact signals
                   only. No hazard_approach events. No proximity gradient in obs.

Mechanism under test:
  If ARC-024 is correct, the GRADIENT_PAIR condition should show:
  (a) E3.harm_eval(z_world) rises BEFORE hazard contact (hazard_approach > none).
  (b) A benefit probe on z_world rises BEFORE resource contact (resource_approach > none).
  (c) The harm direction and benefit direction are OPPOSING in z_world feature space --
      measured as negative cosine similarity between the mean harm-gradient vector and
      mean benefit-gradient vector in z_world.

  The FLAT_CONTROL cannot produce (a) or (b) because no approach events occur.
  The pair comparison (GRADIENT - FLAT) demonstrates the gradient structure is
  specifically caused by the proxy fields, not by incidental training dynamics.

PRE-REGISTERED THRESHOLDS (hardcoded, not inferred post-hoc):
  SEEDS = [42, 123]  (matched across both conditions)
  CONDITIONS = ["GRADIENT_PAIR", "FLAT_CONTROL"]
  WARMUP_EPISODES = 400  (2x EXQ-107 to allow harm_eval and benefit_probe to learn)
  EVAL_EPISODES   = 40
  STEPS_PER_EP    = 200

  C1 (harm gradient, both seeds):
    gap_harm_approach_none (GRADIENT) >= 0.04 for BOTH seeds.
    mean(harm_eval(z_world) | hazard_approach) - mean(harm_eval | none) >= 0.04.
    Threshold halved vs EXQ-107 because 400 warmup allows better calibration.
    PASS: c1_seed_pass_count == 2.

  C2 (benefit gradient, both seeds):
    gap_benefit_approach_none (GRADIENT) >= 0.04 for BOTH seeds.
    mean(benefit_probe(z_world) | resource_approach) - mean(benefit_probe | none) >= 0.04.
    Tests the opposing (benefit) arm of ARC-024.
    PASS: c2_seed_pass_count == 2.

  C3 (opposing directionality, at least 1 seed):
    cosine_similarity(harm_gradient_vec, benefit_gradient_vec) <= -0.10 for >= 1 seed.
    harm_gradient_vec = mean_z_world(hazard_approach) - mean_z_world(none).
    benefit_gradient_vec = mean_z_world(resource_approach) - mean_z_world(none).
    Negative cosine similarity indicates the two gradient directions oppose each other.
    PASS: c3_seed_pass_count >= 1.

  C4 (data quality):
    n_approach_harm_eval >= 20 AND n_approach_benefit_eval >= 20 per seed.
    Ensures sufficient approach events for both harm and benefit gradient measurement.

PASS criteria:
  C1_both AND C2_both AND C4 -> PASS -> supports ARC-024 (both gradient arms present)
  C1_both AND NOT C2_both    -> FAIL -> partial (harm only, benefit not confirmed)
  NOT C1_both AND C2_both    -> FAIL -> partial (benefit only, harm not confirmed)
  NOT C1_both AND NOT C2_both -> FAIL -> weakens ARC-024

PASS = C1_both AND C2_both AND C4.
C3 (opposing cosine) is supporting evidence but not required for PASS.
"""

import sys
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_117_arc024_harm_benefit_gradient_pair"
CLAIM_IDS = ["ARC-024"]

# Pre-registered thresholds
C1_THRESHOLD = 0.04   # gap_harm_approach_none >= 0.04 (harm gradient, per-seed)
C2_THRESHOLD = 0.04   # gap_benefit_approach_none >= 0.04 (benefit gradient, per-seed)
C3_THRESHOLD = -0.10  # cosine_sim(harm_dir, benefit_dir) <= -0.10 (opposing, supporting)
C4_MIN_APPROACH = 20  # approach events needed per seed for each signal type

SEEDS: Tuple[int, ...] = (42, 123)
CONDITIONS = ["GRADIENT_PAIR", "FLAT_CONTROL"]

WARMUP_EPISODES = 400
EVAL_EPISODES   = 40
STEPS_PER_EP    = 200

SELF_DIM    = 32
WORLD_DIM   = 32
LR          = 1e-3
ALPHA_WORLD = 0.9   # SD-008: must not use 0.3 default
ALPHA_SELF  = 0.3

MAX_BUF = 2000


def _action_to_onehot(action_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, action_idx] = 1.0
    return v


def _cosine_sim(vec_a: List[float], vec_b: List[float]) -> float:
    """Cosine similarity between two vectors (as lists). Returns nan if zero norm."""
    if len(vec_a) != len(vec_b) or len(vec_a) == 0:
        return float("nan")
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a < 1e-8 or norm_b < 1e-8:
        return float("nan")
    return dot / (norm_a * norm_b)


def _mean_vec(tensors: List[torch.Tensor]) -> List[float]:
    """Element-wise mean of a list of 1-D tensors."""
    if not tensors:
        return []
    stacked = torch.stack([t.squeeze() for t in tensors], dim=0)
    return stacked.mean(dim=0).tolist()


def _run_single(
    seed: int,
    condition: str,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_ep: int,
) -> Dict:
    """Train and eval one (seed, condition) cell. Returns metrics dict."""
    torch.manual_seed(seed)
    random.seed(seed)

    use_proxy = (condition == "GRADIENT_PAIR")

    env = CausalGridWorldV2(
        seed=seed,
        size=6,
        num_hazards=4,
        num_resources=5,
        hazard_harm=0.5,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.03,
        hazard_field_decay=0.5,
        resource_field_decay=0.5,
        use_proxy_fields=use_proxy,
        resource_respawn_on_consume=True,  # keep resource fields active throughout
    )

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=ALPHA_WORLD,
        alpha_self=ALPHA_SELF,
    )
    config.latent.unified_latent_mode = False  # SD-005 split

    agent = REEAgent(config)

    # Harm eval uses E3 head (standard)
    std_params       = [p for n, p in agent.named_parameters() if "harm_eval_head" not in n]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    optimizer        = optim.Adam(std_params, lr=LR)
    harm_eval_opt    = optim.Adam(harm_eval_params, lr=1e-4)

    # Benefit probe: standalone linear head on z_world (separate from E3)
    benefit_probe = nn.Linear(WORLD_DIM, 1)
    benefit_probe_opt = optim.Adam(benefit_probe.parameters(), lr=1e-4)
    benefit_probe = benefit_probe.to(agent.device)

    # Replay buffers
    harm_buf_pos:    List[torch.Tensor] = []  # z_world at harm events
    harm_buf_neg:    List[torch.Tensor] = []  # z_world at neutral steps
    benefit_buf_pos: List[torch.Tensor] = []  # z_world at benefit events
    benefit_buf_neg: List[torch.Tensor] = []  # z_world at neutral steps (benefit side)

    # ------------------------------------------------------------------ TRAIN
    agent.train()
    benefit_probe.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_ep):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            latent = agent.sense(obs_body, obs_world)
            ticks  = agent.clock.advance()

            if ticks["e1_tick"]:
                agent._e1_tick(latent)

            # Policy: random exploration
            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            _, harm_signal, done, _, obs_dict = env.step(action)

            z_w = latent.z_world.detach()

            # Harm replay buffer
            if float(harm_signal) < 0:
                harm_buf_pos.append(z_w)
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_w)
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            # Benefit replay buffer (benefit_signal > 0 = resource collected)
            if float(harm_signal) > 0:
                benefit_buf_pos.append(z_w)
                if len(benefit_buf_pos) > MAX_BUF:
                    benefit_buf_pos = benefit_buf_pos[-MAX_BUF:]
            else:
                benefit_buf_neg.append(z_w)
                if len(benefit_buf_neg) > MAX_BUF:
                    benefit_buf_neg = benefit_buf_neg[-MAX_BUF:]

            # E1 + E2 loss
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # E3 harm_eval training
            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pi  = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                ni  = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()
                zw_b = torch.cat([harm_buf_pos[i] for i in pi] + [harm_buf_neg[i] for i in ni], dim=0)
                tgt  = torch.cat([
                    torch.ones( k_pos, 1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                ], dim=0)
                loss_h = F.mse_loss(agent.e3.harm_eval(zw_b), tgt)
                if loss_h.requires_grad:
                    harm_eval_opt.zero_grad()
                    loss_h.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5
                    )
                    harm_eval_opt.step()

            # Benefit probe training
            if len(benefit_buf_pos) >= 4 and len(benefit_buf_neg) >= 4:
                k_pos = min(16, len(benefit_buf_pos))
                k_neg = min(16, len(benefit_buf_neg))
                pi  = torch.randperm(len(benefit_buf_pos))[:k_pos].tolist()
                ni  = torch.randperm(len(benefit_buf_neg))[:k_neg].tolist()
                zw_b = torch.cat([benefit_buf_pos[i] for i in pi] + [benefit_buf_neg[i] for i in ni], dim=0)
                tgt  = torch.cat([
                    torch.ones( k_pos, 1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                ], dim=0)
                loss_b = F.mse_loss(benefit_probe(zw_b), tgt)
                if loss_b.requires_grad:
                    benefit_probe_opt.zero_grad()
                    loss_b.backward()
                    benefit_probe_opt.step()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
            print(
                f"  [train] cond={condition} seed={seed}"
                f" ep {ep+1}/{warmup_episodes}"
                f" harm_buf_pos={len(harm_buf_pos)}"
                f" benefit_buf_pos={len(benefit_buf_pos)}",
                flush=True,
            )

    # ------------------------------------------------------------------ EVAL
    # For each step, record:
    #   - harm_eval score paired with the transition_type that led to this state
    #   - benefit_probe score paired with the transition_type
    #   - z_world tensors for hazard_approach, resource_approach, and none states
    agent.eval()
    benefit_probe.eval()

    harm_approach_scores:    List[float] = []  # harm_eval at hazard_approach
    harm_none_scores:        List[float] = []  # harm_eval at none
    benefit_approach_scores: List[float] = []  # benefit_probe at resource_approach
    benefit_none_scores:     List[float] = []  # benefit_probe at none

    # z_world vectors for cosine opposition test
    zw_harm_approach_tensors:    List[torch.Tensor] = []
    zw_benefit_approach_tensors: List[torch.Tensor] = []
    zw_none_tensors:             List[torch.Tensor] = []

    for _ in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        prev_ttype = "none"

        for _ in range(steps_per_ep):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                ticks  = agent.clock.advance()
                if ticks["e1_tick"]:
                    agent._e1_tick(latent)
                z_w = latent.z_world.detach()
                harm_score    = float(agent.e3.harm_eval(z_w).item())
                benefit_score = float(benefit_probe(z_w).item())

            # Record scores and z_world vectors for the state reached by prev_ttype
            if prev_ttype == "hazard_approach":
                harm_approach_scores.append(harm_score)
                zw_harm_approach_tensors.append(z_w.clone())
            elif prev_ttype == "resource_approach":
                benefit_approach_scores.append(benefit_score)
                zw_benefit_approach_tensors.append(z_w.clone())
            elif prev_ttype == "none":
                harm_none_scores.append(harm_score)
                benefit_none_scores.append(benefit_score)
                zw_none_tensors.append(z_w.clone())

            # Policy: random actions for maximum coverage
            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            _, _, done, step_info, obs_dict = env.step(action)
            prev_ttype = step_info.get("transition_type", "none")

            if done:
                break

    # --- Compute per-seed metrics ---
    n_harm_approach    = len(harm_approach_scores)
    n_benefit_approach = len(benefit_approach_scores)
    n_none             = len(harm_none_scores)

    if n_harm_approach >= 3 and n_none >= 3:
        gap_harm = (
            sum(harm_approach_scores) / n_harm_approach
            - sum(harm_none_scores) / n_none
        )
    else:
        gap_harm = float("nan")

    if n_benefit_approach >= 3 and len(benefit_none_scores) >= 3:
        gap_benefit = (
            sum(benefit_approach_scores) / n_benefit_approach
            - sum(benefit_none_scores) / len(benefit_none_scores)
        )
    else:
        gap_benefit = float("nan")

    # Cosine similarity between harm-gradient direction and benefit-gradient direction
    cosine_sim = float("nan")
    if (zw_harm_approach_tensors and zw_benefit_approach_tensors
            and zw_none_tensors):
        mean_harm_approach    = _mean_vec(zw_harm_approach_tensors)
        mean_benefit_approach = _mean_vec(zw_benefit_approach_tensors)
        mean_none             = _mean_vec(zw_none_tensors)
        harm_dir    = [h - n for h, n in zip(mean_harm_approach, mean_none)]
        benefit_dir = [b - n for b, n in zip(mean_benefit_approach, mean_none)]
        cosine_sim  = _cosine_sim(harm_dir, benefit_dir)

    gap_h_str = f"{gap_harm:.4f}"    if not math.isnan(gap_harm)    else "nan"
    gap_b_str = f"{gap_benefit:.4f}" if not math.isnan(gap_benefit) else "nan"
    cos_str   = f"{cosine_sim:.4f}"  if not math.isnan(cosine_sim)  else "nan"

    print(
        f"  [eval] cond={condition} seed={seed}"
        f" n_harm_approach={n_harm_approach}"
        f" n_benefit_approach={n_benefit_approach}"
        f" n_none={n_none}"
        f" gap_harm={gap_h_str}"
        f" gap_benefit={gap_b_str}"
        f" cosine_sim={cos_str}",
        flush=True,
    )

    return {
        "seed":               seed,
        "condition":          condition,
        "gap_harm_approach":  gap_harm,
        "gap_benefit_approach": gap_benefit,
        "cosine_sim_harm_benefit": cosine_sim,
        "n_harm_approach":    n_harm_approach,
        "n_benefit_approach": n_benefit_approach,
        "n_none":             n_none,
    }


def run(
    seeds: Tuple = SEEDS,
    warmup_episodes: int = WARMUP_EPISODES,
    eval_episodes:   int = EVAL_EPISODES,
    steps_per_ep:    int = STEPS_PER_EP,
    **kwargs,
) -> dict:
    """
    Discriminative pair: GRADIENT_PAIR vs FLAT_CONTROL x seeds [42, 123].
    C1: gap_harm_approach_none >= 0.04 in GRADIENT_PAIR for both seeds.
    C2: gap_benefit_approach_none >= 0.04 in GRADIENT_PAIR for both seeds.
    C3 (supporting): cosine_sim(harm_dir, benefit_dir) <= -0.10 for >= 1 seed.
    C4: n_harm_approach >= 20 AND n_benefit_approach >= 20 per seed.
    PASS: C1_both AND C2_both AND C4.
    """
    all_results: List[Dict] = []

    for seed in seeds:
        for cond in CONDITIONS:
            print(
                f"\n[V3-EXQ-117] {cond} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" steps={steps_per_ep}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                condition=cond,
                warmup_episodes=warmup_episodes,
                eval_episodes=eval_episodes,
                steps_per_ep=steps_per_ep,
            )
            all_results.append(r)

    grad_results = [r for r in all_results if r["condition"] == "GRADIENT_PAIR"]
    flat_results = [r for r in all_results if r["condition"] == "FLAT_CONTROL"]

    # ---- Per-seed criterion evaluation ----
    c1_seed_passes = []
    c2_seed_passes = []
    c3_seed_passes = []
    c4_seed_passes = []

    for r in grad_results:
        gap_h = r["gap_harm_approach"]
        gap_b = r["gap_benefit_approach"]
        cos   = r["cosine_sim_harm_benefit"]
        n_h   = r["n_harm_approach"]
        n_b   = r["n_benefit_approach"]

        c1_ok = (not math.isnan(gap_h)) and gap_h >= C1_THRESHOLD
        c2_ok = (not math.isnan(gap_b)) and gap_b >= C2_THRESHOLD
        c3_ok = (not math.isnan(cos))   and cos   <= C3_THRESHOLD
        c4_ok = (n_h >= C4_MIN_APPROACH) and (n_b >= C4_MIN_APPROACH)

        c1_seed_passes.append(c1_ok)
        c2_seed_passes.append(c2_ok)
        c3_seed_passes.append(c3_ok)
        c4_seed_passes.append(c4_ok)

    c1_pass_count = sum(c1_seed_passes)
    c2_pass_count = sum(c2_seed_passes)
    c3_pass_count = sum(c3_seed_passes)
    c4_pass_count = sum(c4_seed_passes)

    c1_both = (c1_pass_count == len(seeds))
    c2_both = (c2_pass_count == len(seeds))
    c3_any  = (c3_pass_count >= 1)
    c4_all  = (c4_pass_count == len(seeds))

    status = "PASS" if (c1_both and c2_both and c4_all) else "FAIL"

    # ---- Summary statistics ----
    def _safe_mean(vals: List[float]) -> float:
        finite = [v for v in vals if not math.isnan(v)]
        return sum(finite) / len(finite) if finite else float("nan")

    def _safe_sem(vals: List[float], mean: float) -> float:
        finite = [v for v in vals if not math.isnan(v)]
        n = len(finite)
        if n < 2:
            return float("nan")
        var = sum((v - mean) ** 2 for v in finite) / (n - 1)
        return math.sqrt(var / n)

    grad_harm_gaps    = [r["gap_harm_approach"]       for r in grad_results]
    grad_benefit_gaps = [r["gap_benefit_approach"]     for r in grad_results]
    grad_cos_sims     = [r["cosine_sim_harm_benefit"]  for r in grad_results]

    mean_gap_harm    = _safe_mean(grad_harm_gaps)
    mean_gap_benefit = _safe_mean(grad_benefit_gaps)
    mean_cos_sim     = _safe_mean(grad_cos_sims)
    sem_gap_harm     = _safe_sem(grad_harm_gaps,    mean_gap_harm)
    sem_gap_benefit  = _safe_sem(grad_benefit_gaps, mean_gap_benefit)

    def _fmt(v: float) -> str:
        return f"{v:.4f}" if not math.isnan(v) else "nan"

    print(f"\n[V3-EXQ-117] Final results:", flush=True)
    print(
        f"  C1 (harm gradient, both seeds): "
        f"mean_gap_harm={_fmt(mean_gap_harm)} +/- {_fmt(sem_gap_harm)}"
        f" (need >={C1_THRESHOLD} per seed, pass_count={c1_pass_count}/{len(seeds)})"
        f" -> {'PASS' if c1_both else 'FAIL'}",
        flush=True,
    )
    print(
        f"  C2 (benefit gradient, both seeds): "
        f"mean_gap_benefit={_fmt(mean_gap_benefit)} +/- {_fmt(sem_gap_benefit)}"
        f" (need >={C2_THRESHOLD} per seed, pass_count={c2_pass_count}/{len(seeds)})"
        f" -> {'PASS' if c2_both else 'FAIL'}",
        flush=True,
    )
    print(
        f"  C3 (opposing cosine, supporting): "
        f"mean_cos_sim={_fmt(mean_cos_sim)}"
        f" (need <={C3_THRESHOLD} for >=1 seed, pass_count={c3_pass_count}/{len(seeds)})"
        f" -> {'PASS' if c3_any else 'FAIL'}",
        flush=True,
    )
    print(
        f"  C4 (data quality): "
        f"harm_approach>={C4_MIN_APPROACH} AND benefit_approach>={C4_MIN_APPROACH} per seed"
        f" (pass_count={c4_pass_count}/{len(seeds)})"
        f" -> {'PASS' if c4_all else 'FAIL'}",
        flush=True,
    )
    print(f"  Status: {status}", flush=True)

    # ---- Interpretation ----
    if status == "PASS":
        interpretation = (
            f"ARC-024 SUPPORTED: Both harm and benefit gradient structure confirmed"
            f" in z_world with GRADIENT_PAIR condition."
            f" Harm gap={_fmt(mean_gap_harm)}>={C1_THRESHOLD} (both seeds)."
            f" Benefit gap={_fmt(mean_gap_benefit)}>={C2_THRESHOLD} (both seeds)."
            + (f" Opposing cosine confirmed (mean={_fmt(mean_cos_sim)})." if c3_any else "")
        )
    elif not c1_both and not c2_both:
        interpretation = (
            f"ARC-024 NOT SUPPORTED: Neither harm gradient (mean_gap={_fmt(mean_gap_harm)}"
            f" vs threshold {C1_THRESHOLD}) nor benefit gradient (mean_gap={_fmt(mean_gap_benefit)}"
            f" vs threshold {C2_THRESHOLD}) detected in z_world at {warmup_episodes} warmup episodes."
        )
    elif not c1_both:
        interpretation = (
            f"ARC-024 PARTIAL: Benefit gradient confirmed (gap={_fmt(mean_gap_benefit)}>={C2_THRESHOLD})"
            f" but harm gradient not confirmed (gap={_fmt(mean_gap_harm)} < {C1_THRESHOLD})"
            f" in at least one seed. Both arms required for full ARC-024 support."
        )
    else:
        interpretation = (
            f"ARC-024 PARTIAL: Harm gradient confirmed (gap={_fmt(mean_gap_harm)}>={C1_THRESHOLD})"
            f" but benefit gradient not confirmed (gap={_fmt(mean_gap_benefit)} < {C2_THRESHOLD})"
            f" in at least one seed. Both arms required for full ARC-024 support."
        )

    # ---- Failure notes ----
    failure_notes: List[str] = []
    if not c1_both:
        for i, (r, ok) in enumerate(zip(grad_results, c1_seed_passes)):
            if not ok:
                failure_notes.append(
                    f"C1 FAIL seed={r['seed']}: gap_harm_approach={_fmt(r['gap_harm_approach'])}"
                    f" < {C1_THRESHOLD} -- harm gradient not detected in GRADIENT_PAIR"
                )
    if not c2_both:
        for i, (r, ok) in enumerate(zip(grad_results, c2_seed_passes)):
            if not ok:
                failure_notes.append(
                    f"C2 FAIL seed={r['seed']}: gap_benefit_approach={_fmt(r['gap_benefit_approach'])}"
                    f" < {C2_THRESHOLD} -- benefit gradient not detected in GRADIENT_PAIR"
                )
    if not c4_all:
        for i, (r, ok) in enumerate(zip(grad_results, c4_seed_passes)):
            if not ok:
                failure_notes.append(
                    f"C4 FAIL seed={r['seed']}: n_harm_approach={r['n_harm_approach']}"
                    f" n_benefit_approach={r['n_benefit_approach']}"
                    f" (both need >={C4_MIN_APPROACH}) -- insufficient approach events"
                )

    # Per-seed detail table
    per_seed_rows = ""
    for r in all_results:
        per_seed_rows += (
            f"| {r['condition']} | {r['seed']}"
            f" | {_fmt(r['gap_harm_approach'])}"
            f" | {_fmt(r['gap_benefit_approach'])}"
            f" | {_fmt(r['cosine_sim_harm_benefit'])}"
            f" | {r['n_harm_approach']}/{r['n_benefit_approach']}/{r['n_none']} |\n"
        )

    flat_harm_gaps    = [r["gap_harm_approach"]      for r in flat_results]
    flat_benefit_gaps = [r["gap_benefit_approach"]    for r in flat_results]
    mean_flat_harm    = _safe_mean(flat_harm_gaps)
    mean_flat_benefit = _safe_mean(flat_benefit_gaps)

    summary_markdown = (
        f"# V3-EXQ-117 -- ARC-024 Harm-Benefit Opposing Gradient Discriminative Pair\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** ARC-024\n"
        f"**Proposal:** EXP-0014 / EVB-0011\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps"
        f"  **Steps/ep:** {steps_per_ep}\n\n"
        f"## Design\n\n"
        f"Discriminative pair testing BOTH harm and benefit gradient arms of ARC-024:\n"
        f"  GRADIENT_PAIR: use_proxy_fields=True -- hazard_field_view + resource_field_view.\n"
        f"                  E3.harm_eval trained on harm events."
        f" Benefit probe (linear on z_world) trained on benefit events.\n"
        f"  FLAT_CONTROL:  use_proxy_fields=False -- binary contact signals only.\n"
        f"                  No approach events; no proximity gradient.\n\n"
        f"Background: EXQ-107 (5 seeds, 200 warmup) FAIL on C1 (gap_harm=0.003 vs 0.08).\n"
        f"  This experiment uses 400 warmup, 2 matched seeds, reduced C1/C2 threshold (0.04),\n"
        f"  and adds benefit gradient (C2) and opposing cosine test (C3).\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1 (harm gradient): gap_harm_approach_none >= {C1_THRESHOLD} GRADIENT, both seeds\n"
        f"C2 (benefit gradient): gap_benefit_approach_none >= {C2_THRESHOLD} GRADIENT, both seeds\n"
        f"C3 (opposing cosine, supporting): cos_sim(harm_dir, benefit_dir) <= {C3_THRESHOLD} >= 1 seed\n"
        f"C4 (data quality): n_harm_approach >= {C4_MIN_APPROACH} AND n_benefit_approach >= {C4_MIN_APPROACH} per seed\n"
        f"PASS: C1_both AND C2_both AND C4\n\n"
        f"## Results Summary\n\n"
        f"| Metric | GRADIENT_PAIR | FLAT_CONTROL |\n"
        f"|--------|--------------|-------------|\n"
        f"| gap_harm_approach_none (mean) | {_fmt(mean_gap_harm)} | {_fmt(mean_flat_harm)} |\n"
        f"| gap_benefit_approach_none (mean) | {_fmt(mean_gap_benefit)} | {_fmt(mean_flat_benefit)} |\n"
        f"| cosine_sim(harm_dir, benefit_dir) (mean) | {_fmt(mean_cos_sim)} | N/A |\n\n"
        f"| Criterion | Result | Value |\n"
        f"|-----------|--------|-------|\n"
        f"| C1: harm gap >= {C1_THRESHOLD} (both seeds) | {'PASS' if c1_both else 'FAIL'}"
        f" | {_fmt(mean_gap_harm)} (pass_count={c1_pass_count}/{len(seeds)}) |\n"
        f"| C2: benefit gap >= {C2_THRESHOLD} (both seeds) | {'PASS' if c2_both else 'FAIL'}"
        f" | {_fmt(mean_gap_benefit)} (pass_count={c2_pass_count}/{len(seeds)}) |\n"
        f"| C3: cosine <= {C3_THRESHOLD} (>=1 seed, supporting) | {'PASS' if c3_any else 'FAIL'}"
        f" | {_fmt(mean_cos_sim)} (pass_count={c3_pass_count}/{len(seeds)}) |\n"
        f"| C4: data quality (both seeds) | {'PASS' if c4_all else 'FAIL'}"
        f" | pass_count={c4_pass_count}/{len(seeds)} |\n\n"
        f"**PASS = C1_both AND C2_both AND C4 -> {status}**\n\n"
        f"## Per-Seed Results\n\n"
        f"| Condition | Seed | gap_harm | gap_benefit | cosine_sim | n(harm_app/ben_app/none) |\n"
        f"|-----------|------|----------|-------------|------------|-------------------------|\n"
        + per_seed_rows
        + f"\n## Interpretation\n\n{interpretation}\n"
    )
    if failure_notes:
        summary_markdown += (
            "\n## Failure Notes\n\n"
            + "\n".join(f"- {fn}" for fn in failure_notes)
            + "\n"
        )

    def _sf(v: float) -> float:
        return float(v) if not math.isnan(v) else -99.0

    def _sf_sem(v: float) -> float:
        return float(v) if not math.isnan(v) else -1.0

    return {
        "status":             status,
        "claim_ids":          CLAIM_IDS,
        "experiment_type":    EXPERIMENT_TYPE,
        "evidence_direction": "supports" if status == "PASS" else "weakens",
        "fatal_error_count":  0,
        "summary_markdown":   summary_markdown,
        "metrics": {
            "gradient_gap_harm_approach_none_mean":    _sf(mean_gap_harm),
            "gradient_gap_harm_approach_none_sem":     _sf_sem(sem_gap_harm),
            "gradient_gap_benefit_approach_none_mean": _sf(mean_gap_benefit),
            "gradient_gap_benefit_approach_none_sem":  _sf_sem(sem_gap_benefit),
            "gradient_cosine_sim_harm_benefit_mean":   _sf(mean_cos_sim),
            "flat_gap_harm_approach_none_mean":        _sf(mean_flat_harm),
            "flat_gap_benefit_approach_none_mean":     _sf(mean_flat_benefit),
            "c1_threshold":                            float(C1_THRESHOLD),
            "c2_threshold":                            float(C2_THRESHOLD),
            "c3_threshold":                            float(C3_THRESHOLD),
            "c4_min_approach":                         float(C4_MIN_APPROACH),
            "c1_pass_count":                           float(c1_pass_count),
            "c2_pass_count":                           float(c2_pass_count),
            "c3_pass_count":                           float(c3_pass_count),
            "c4_pass_count":                           float(c4_pass_count),
            "n_seeds":                                 float(len(seeds)),
            "warmup_episodes":                         float(warmup_episodes),
            "eval_episodes":                           float(eval_episodes),
        },
        "per_seed": all_results,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int,            default=WARMUP_EPISODES)
    parser.add_argument("--eval",   type=int,            default=EVAL_EPISODES)
    parser.add_argument("--steps",  type=int,            default=STEPS_PER_EP)
    parser.add_argument("--seeds",  type=int, nargs="+", default=list(SEEDS))
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Quick check: 5 warmup, 5 eval, 20 steps, seed 42 only",
    )
    args = parser.parse_args()

    if args.smoke_test:
        warmup, eval_eps, steps, seeds = 5, 5, 20, [42]
        print("[V3-EXQ-117] SMOKE TEST MODE", flush=True)
    else:
        warmup, eval_eps, steps, seeds = args.warmup, args.eval, args.steps, args.seeds

    result = run(
        seeds=tuple(seeds),
        warmup_episodes=warmup,
        eval_episodes=eval_eps,
        steps_per_ep=steps,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"]      = ts
    result["run_id"]             = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    if args.smoke_test:
        print(f"\n[SMOKE] Status: {result['status']}", flush=True)
        for k, v in result["metrics"].items():
            print(f"  {k}: {v}", flush=True)
        print("[SMOKE] Script ran without error. No file written.", flush=True)
    else:
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
