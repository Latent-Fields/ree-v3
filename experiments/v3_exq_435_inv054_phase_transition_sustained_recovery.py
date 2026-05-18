#!/opt/local/bin/python3
"""
V3-EXQ-435 -- INV-054: Phase-Transition Recovery (Sustained-Crossing Criterion)

Claims: INV-054
EXPERIMENT_PURPOSE = "evidence"
Backlog: EXP-0097
Supersedes: V3-EXQ-278 (single-episode recovery crossing was too noisy)

=== SCIENTIFIC QUESTION ===

Does recovery from the computational depression attractor follow a phase-transition
pattern (delayed, threshold-crossing recovery) rather than graded/immediate recovery?

Specifically: after establishing depression attractor (z_goal_norm < 0.1 for 50+
consecutive episodes in LONG_HORIZON), does switching to LOW_HARM environment produce:
  - PHASE_TRANSITION: SUSTAINED recovery onset > 50 phase-2 episodes
                      (delayed, attractor escape)
  - GRADED:           SUSTAINED recovery onset <= 10 phase-2 episodes
                      (immediate, reward-signal account)

=== KEY DIFFERENCE vs V3-EXQ-278 ===

V3-EXQ-278 measured recovery_latency = first episode where z_goal_norm >= 0.3 (any
single crossing). Result: latency=1-3 in all seeds (instantaneous "recovery" that
was actually a one-step bounce driven by drive_weight=2.0 x benefit_exposure in the
fresh low-harm env, not a genuine attractor escape).

This iteration fixes the measurement: recovery is declared only when z_goal_norm
is SUSTAINED above RECOVERY_NORM_THRESH for RECOVERY_SUSTAIN_EPS consecutive
episodes. This matches the biological prediction: a single spike of goal
representation is not recovery from a depressive attractor; stable re-engagement
with the terrain is.

=== BIOLOGICAL GROUNDING ===

INV-054 maintenance loop predicts that the depressive state is self-maintaining:
z_goal absent -> no terrain exploration -> terrain stays collapsed.
Recovery requires accumulated terrain re-exposure to cross a threshold.

This is a unique prediction distinguishing INV-054 from simple reward-signal
insufficiency accounts. A reward-signal account predicts immediate/graded recovery
when the environment improves. The INV-054 attractor account predicts hysteresis:
the state persists even after the triggering environmental condition is removed,
until sufficient terrain re-exposure accumulates to cross a basin boundary.

Mathew, Manji & Charney (2008) on SSRI treatment response latency (2-4 weeks):
if recovery were graded, immediate partial response would be expected; the observed
delayed non-linear response supports an attractor/phase-transition account.

=== DESIGN ===

3 seeds (42, 43, 44).

Phase 1: LONG_HORIZON environment (same as EXQ-249/278 parameters).
  - Train until z_goal_norm < 0.1 sustained for 50 consecutive episodes,
    OR max 300 episodes.
  - Records: depression_established (bool), phase1_final_z_goal_norm.

Phase 2: Switch to LOW_HARM environment (num_hazards=0). Continue training for
  300 more episodes. Per-episode z_goal_norm is averaged over the last
  RECOVERY_AVG_WINDOW steps of each episode to reduce per-step noise.
  - recovery_onset = first episode e such that episodes [e .. e + SUSTAIN - 1]
    all have episode_avg_goal_norm >= RECOVERY_NORM_THRESH.

Per-seed recovery_type (based on recovery_onset):
  "phase_transition": onset > 50 (INV-054 attractor prediction)
  "graded":           onset <= 10 (immediate/graded recovery, contradicts INV-054)
  "partial":          10 < onset <= 50 (intermediate)
  "no_recovery":      no sustained crossing in phase 2

PASS per seed: depression_established AND recovery_onset > 50
PASS experiment: >= 2/3 seeds pass.

=== PRE-REGISTERED CRITERIA ===

PASS: >= 2/3 seeds with depression_established=True AND recovery_onset > 50.
FAIL: < 2/3 seeds meeting both criteria.

evidence_direction:
  - "supports"          if PASS
  - "does_not_support"  if >=2/3 depression established but graded recovery (onset <= 10)
  - "non_contributory"  if depression not established in >= 2/3 seeds (substrate limit)
  - "mixed"             otherwise (partial / no_recovery mixture)
"""

import sys
import random
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
EXPERIMENT_TYPE    = "v3_exq_435_inv054_phase_transition_sustained_recovery"
CLAIM_IDS          = ["INV-054"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
DEPRESSION_NORM_THRESH    = 0.1   # z_goal_norm below this = depression attractor
DEPRESSION_SUSTAIN_EPS    = 50    # consecutive episodes below thresh to confirm depression
RECOVERY_NORM_THRESH      = 0.3   # z_goal_norm above this = recovered
RECOVERY_SUSTAIN_EPS      = 10    # consecutive episodes above thresh to confirm recovery
RECOVERY_AVG_WINDOW       = 30    # avg over last N steps of an episode to smooth noise
RECOVERY_PHASE_TRANS_EPS  = 50    # onset > this = phase_transition (INV-054 prediction)
RECOVERY_GRADED_EPS       = 10    # onset <= this = graded (contradicts INV-054)
PASS_SEED_COUNT           = 2     # need >= this many seeds passing (out of 3)

# ---------------------------------------------------------------------------
# Grid and episode parameters
# ---------------------------------------------------------------------------
GRID_SIZE         = 8
STEPS_PER_EP      = 150
WORLD_DIM         = 32

# Phase 1: LONG_HORIZON (same as EXQ-249/278)
LH_N_RESOURCES    = 1
LH_N_HAZARDS      = 3
LH_HAZARD_HARM    = 0.02

# Phase 2: LOW_HARM (no hazards, same resources)
LH2_N_RESOURCES   = 1
LH2_N_HAZARDS     = 0
LH2_HAZARD_HARM   = 0.02

# Training parameters
PHASE1_MAX_EPS    = 300
PHASE2_EPS        = 300
SEEDS             = [42, 43, 44]

GREEDY_FRAC       = 0.4
MAX_BUF           = 4000
WF_BUF_MAX        = 2000
BATCH_SIZE        = 16

# Learning rates
LR_E1      = 1e-3
LR_E2_WF   = 1e-3
LR_HARM    = 1e-4
LR_BENEFIT = 1e-3

# SD-018: resource proximity supervision
LAMBDA_RESOURCE = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _greedy_toward_resource(env) -> int:
    """Greedy action: move toward nearest resource (Manhattan)."""
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
    return 3 if dy > 0 else 2


def _dist_to_nearest_resource(env) -> int:
    if not env.resources:
        return 999
    ax, ay = env.agent_x, env.agent_y
    return min(abs(ax - int(r[0])) + abs(ay - int(r[1])) for r in env.resources)


def _get_benefit_exposure(obs_body: torch.Tensor) -> float:
    flat = obs_body.flatten()
    if flat.shape[0] > 11:
        return float(flat[11].item())
    return 0.0


def _get_energy(obs_body: torch.Tensor) -> float:
    flat = obs_body.flatten()
    if flat.shape[0] > 3:
        return float(flat[3].item())
    return 1.0


def _update_z_goal(agent: REEAgent, obs_body: torch.Tensor) -> None:
    b_exp = _get_benefit_exposure(obs_body)
    energy = _get_energy(obs_body)
    drive_level = max(0.0, 1.0 - energy)
    agent.update_z_goal(b_exp, drive_level=drive_level)


def _get_goal_norm(agent: REEAgent) -> float:
    diag = agent.compute_goal_maintenance_diagnostic()
    return float(diag["goal_norm"])


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------

def _make_long_horizon_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_resources=LH_N_RESOURCES,
        num_hazards=LH_N_HAZARDS,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        hazard_harm=LH_HAZARD_HARM,
        proximity_harm_scale=0.3,
        proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        env_drift_interval=999,
        env_drift_prob=0.0,
    )


def _make_low_harm_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed + 1000,
        size=GRID_SIZE,
        num_resources=LH2_N_RESOURCES,
        num_hazards=LH2_N_HAZARDS,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        hazard_harm=LH2_HAZARD_HARM,
        proximity_harm_scale=0.3,
        proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        env_drift_interval=999,
        env_drift_prob=0.0,
    )


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _make_agent(env: CausalGridWorldV2, seed: int) -> REEAgent:
    torch.manual_seed(seed)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=16,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=0,
        novelty_bonus_weight=0.0,
        benefit_eval_enabled=True,
        benefit_weight=0.5,
        z_goal_enabled=True,
        e1_goal_conditioned=True,
        goal_weight=1.0,
        drive_weight=2.0,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
    )
    return REEAgent(config)


# ---------------------------------------------------------------------------
# Per-episode training step (returns per-step goal_norm trace over last window)
# ---------------------------------------------------------------------------

def _train_episode(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    e1_opt: optim.Optimizer,
    e2_wf_opt: optim.Optimizer,
    harm_opt: optim.Optimizer,
    benefit_opt: optim.Optimizer,
    wf_buf: List,
    harm_pos_buf: List,
    harm_neg_buf: List,
    ben_zw_buf: List,
    ben_lbl_buf: List,
    e1_params: List,
    e2_wf_params: List,
    record_per_step: bool = False,
) -> List[float]:
    device = agent.device
    n_act  = env.action_dim

    _, obs_dict = env.reset()
    agent.reset()

    z_world_prev: Optional[torch.Tensor] = None
    action_prev:  Optional[torch.Tensor] = None

    per_step_goal_norm: List[float] = []

    for step_i in range(STEPS_PER_EP):
        obs_body  = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        obs_harm  = obs_dict.get("harm_obs", None)

        latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
        ticks  = agent.clock.advance()

        if ticks.get("e1_tick", False):
            _ = agent._e1_tick(latent)

        z_world_curr = latent.z_world.detach()

        if z_world_prev is not None and action_prev is not None:
            wf_buf.append((z_world_prev, action_prev, z_world_curr))
            if len(wf_buf) > WF_BUF_MAX:
                del wf_buf[:-WF_BUF_MAX]

        if random.random() < GREEDY_FRAC:
            action_idx = _greedy_toward_resource(env)
        else:
            action_idx = random.randint(0, n_act - 1)
        action_oh = _onehot(action_idx, n_act, device)
        agent._last_action = action_oh

        dist    = _dist_to_nearest_resource(env)
        is_near = 1.0 if dist <= 2 else 0.0

        _, harm_signal, done, info, obs_dict = env.step(action_oh)

        _update_z_goal(agent, obs_dict["body_state"])

        if record_per_step:
            per_step_goal_norm.append(_get_goal_norm(agent))

        if len(agent._world_experience_buffer) >= 2:
            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                e1_opt.zero_grad()
                e1_loss.backward()
                torch.nn.utils.clip_grad_norm_(e1_params, 1.0)
                e1_opt.step()

        rfv = obs_dict.get("resource_field_view", None)
        if rfv is not None:
            rp_target = rfv[12].item()
            rp_loss = agent.compute_resource_proximity_loss(rp_target, latent)
            if rp_loss.requires_grad:
                e1_opt.zero_grad()
                (LAMBDA_RESOURCE * rp_loss).backward()
                torch.nn.utils.clip_grad_norm_(e1_params, 1.0)
                e1_opt.step()

        if len(wf_buf) >= BATCH_SIZE:
            idxs  = random.sample(range(len(wf_buf)), min(BATCH_SIZE, len(wf_buf)))
            zw_b  = torch.cat([wf_buf[i][0] for i in idxs]).to(device)
            a_b   = torch.cat([wf_buf[i][1] for i in idxs]).to(device)
            zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(device)
            wf_pred = agent.e2.world_forward(zw_b, a_b)
            wf_loss = F.mse_loss(wf_pred, zw1_b)
            if wf_loss.requires_grad:
                e2_wf_opt.zero_grad()
                wf_loss.backward()
                torch.nn.utils.clip_grad_norm_(e2_wf_params, 1.0)
                e2_wf_opt.step()
            with torch.no_grad():
                agent.e3.update_running_variance(
                    (wf_pred.detach() - zw1_b).detach()
                )

        if float(harm_signal) < 0:
            harm_pos_buf.append(z_world_curr)
            if len(harm_pos_buf) > MAX_BUF:
                del harm_pos_buf[:-MAX_BUF]
        else:
            harm_neg_buf.append(z_world_curr)
            if len(harm_neg_buf) > MAX_BUF:
                del harm_neg_buf[:-MAX_BUF]

        if len(harm_pos_buf) >= 4 and len(harm_neg_buf) >= 4:
            k_p = min(BATCH_SIZE // 2, len(harm_pos_buf))
            k_n = min(BATCH_SIZE // 2, len(harm_neg_buf))
            pi  = torch.randperm(len(harm_pos_buf))[:k_p].tolist()
            ni  = torch.randperm(len(harm_neg_buf))[:k_n].tolist()
            zw_b = torch.cat(
                [harm_pos_buf[i] for i in pi] + [harm_neg_buf[i] for i in ni],
                dim=0,
            )
            tgt = torch.cat([
                torch.ones(k_p,  1, device=device),
                torch.zeros(k_n, 1, device=device),
            ], dim=0)
            pred = agent.e3.harm_eval(zw_b)
            hloss = F.binary_cross_entropy(pred, tgt)
            if hloss.requires_grad:
                harm_opt.zero_grad()
                hloss.backward()
                torch.nn.utils.clip_grad_norm_(
                    agent.e3.harm_eval_head.parameters(), 0.5
                )
                harm_opt.step()

        ben_zw_buf.append(z_world_curr)
        ben_lbl_buf.append(is_near)
        if len(ben_zw_buf) > MAX_BUF:
            del ben_zw_buf[:-MAX_BUF]
            del ben_lbl_buf[:-MAX_BUF]

        if len(ben_zw_buf) >= 32 and step_i % 4 == 0:
            k    = min(32, len(ben_zw_buf))
            idxs = random.sample(range(len(ben_zw_buf)), k)
            zw_b = torch.cat([ben_zw_buf[i] for i in idxs], dim=0)
            lbl  = torch.tensor(
                [ben_lbl_buf[i] for i in idxs],
                dtype=torch.float32,
            ).unsqueeze(1).to(device)
            pred_b = agent.e3.benefit_eval(zw_b)
            bloss  = F.binary_cross_entropy(pred_b, lbl)
            if bloss.requires_grad:
                benefit_opt.zero_grad()
                bloss.backward()
                torch.nn.utils.clip_grad_norm_(
                    agent.e3.benefit_eval_head.parameters(), 0.5
                )
                benefit_opt.step()
                agent.e3.record_benefit_sample(k)

        z_world_prev = z_world_curr
        action_prev  = action_oh

        if done:
            break

    return per_step_goal_norm


# ---------------------------------------------------------------------------
# Per-seed run
# ---------------------------------------------------------------------------

def _run_seed(seed: int, phase1_max_eps: int, phase2_eps: int, total_eps: int) -> Dict:
    print(f"Seed {seed}", flush=True)

    env_lh = _make_long_horizon_env(seed)
    agent  = _make_agent(env_lh, seed)

    e1_params    = list(agent.e1.parameters())
    e2_wf_params = (
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters())
    )
    e1_opt      = optim.Adam(e1_params, lr=LR_E1)
    e2_wf_opt   = optim.Adam(e2_wf_params, lr=LR_E2_WF)
    harm_opt    = optim.Adam(agent.e3.harm_eval_head.parameters(), lr=LR_HARM)
    benefit_opt = optim.Adam(agent.e3.benefit_eval_head.parameters(), lr=LR_BENEFIT)

    wf_buf:       List = []
    harm_pos_buf: List = []
    harm_neg_buf: List = []
    ben_zw_buf:   List = []
    ben_lbl_buf:  List = []

    random.seed(seed)
    agent.train()

    # --- Phase 1 ---
    consecutive_depressed = 0
    depression_established = False
    phase1_final_norm = 0.0
    phase1_eps_run = 0

    for ep in range(phase1_max_eps):
        _train_episode(
            agent, env_lh, seed,
            e1_opt, e2_wf_opt, harm_opt, benefit_opt,
            wf_buf, harm_pos_buf, harm_neg_buf, ben_zw_buf, ben_lbl_buf,
            e1_params, e2_wf_params,
            record_per_step=False,
        )
        phase1_eps_run = ep + 1

        goal_norm = _get_goal_norm(agent)

        if goal_norm < DEPRESSION_NORM_THRESH:
            consecutive_depressed += 1
        else:
            consecutive_depressed = 0

        if (ep + 1) % 50 == 0 or (ep + 1) == phase1_max_eps or ep == 0:
            print(
                f"  [train] seed={seed} ep {ep+1}/{total_eps} "
                f"phase=1 goal_norm={goal_norm:.3f} "
                f"consec_dep={consecutive_depressed}",
                flush=True,
            )

        if consecutive_depressed >= DEPRESSION_SUSTAIN_EPS:
            depression_established = True
            phase1_final_norm = goal_norm
            break

    if not depression_established:
        phase1_final_norm = _get_goal_norm(agent)
        print(
            f"  [phase1 done] seed={seed} depression NOT established "
            f"after {phase1_max_eps} eps, final_norm={phase1_final_norm:.3f}",
            flush=True,
        )
    else:
        print(
            f"  [phase1 done] seed={seed} depression ESTABLISHED at "
            f"ep={phase1_eps_run}, final_norm={phase1_final_norm:.3f}",
            flush=True,
        )

    # --- Phase 2: switch to LOW_HARM ---
    env_lh2 = _make_low_harm_env(seed)

    episode_avg_goal_norms: List[float] = []  # averaged over last RECOVERY_AVG_WINDOW steps

    for ep in range(phase2_eps):
        per_step_trace = _train_episode(
            agent, env_lh2, seed,
            e1_opt, e2_wf_opt, harm_opt, benefit_opt,
            wf_buf, harm_pos_buf, harm_neg_buf, ben_zw_buf, ben_lbl_buf,
            e1_params, e2_wf_params,
            record_per_step=True,
        )

        # Smoothed metric: average over last RECOVERY_AVG_WINDOW steps of episode
        if len(per_step_trace) >= RECOVERY_AVG_WINDOW:
            ep_avg = sum(per_step_trace[-RECOVERY_AVG_WINDOW:]) / RECOVERY_AVG_WINDOW
        elif per_step_trace:
            ep_avg = sum(per_step_trace) / len(per_step_trace)
        else:
            ep_avg = _get_goal_norm(agent)
        episode_avg_goal_norms.append(ep_avg)

        if (ep + 1) % 50 == 0 or ep == 0 or (ep + 1) == phase2_eps:
            recent_above = 0
            for v in reversed(episode_avg_goal_norms):
                if v >= RECOVERY_NORM_THRESH:
                    recent_above += 1
                else:
                    break
            print(
                f"  [train] seed={seed} ep {phase1_eps_run+ep+1}/{total_eps} "
                f"phase=2 ep_avg_goal_norm={ep_avg:.3f} "
                f"recent_above={recent_above}",
                flush=True,
            )

    # --- Find sustained recovery onset ---
    # recovery_onset = smallest e such that episodes[e..e+SUSTAIN-1] all >= thresh
    recovery_onset = phase2_eps + 1  # sentinel
    for e in range(len(episode_avg_goal_norms) - RECOVERY_SUSTAIN_EPS + 1):
        window = episode_avg_goal_norms[e : e + RECOVERY_SUSTAIN_EPS]
        if all(v >= RECOVERY_NORM_THRESH for v in window):
            recovery_onset = e + 1  # 1-indexed
            break

    if recovery_onset > phase2_eps:
        recovery_type = "no_recovery"
    elif recovery_onset > RECOVERY_PHASE_TRANS_EPS:
        recovery_type = "phase_transition"
    elif recovery_onset > RECOVERY_GRADED_EPS:
        recovery_type = "partial"
    else:
        recovery_type = "graded"

    seed_passed = (
        depression_established
        and recovery_onset > RECOVERY_PHASE_TRANS_EPS
        and recovery_onset <= phase2_eps
    )

    print(
        f"  [phase2 done] seed={seed} "
        f"recovery_onset={recovery_onset if recovery_onset <= phase2_eps else 'none'} "
        f"recovery_type={recovery_type}",
        flush=True,
    )
    print(f"verdict: {'PASS' if seed_passed else 'FAIL'}", flush=True)

    return {
        "seed": seed,
        "depression_established": depression_established,
        "phase1_final_z_goal_norm": float(phase1_final_norm),
        "phase1_eps_run": int(phase1_eps_run),
        "recovery_onset": int(recovery_onset),
        "recovery_type": recovery_type,
        "seed_passed": seed_passed,
        "episode_avg_goal_norm_sampled": [
            float(episode_avg_goal_norms[i])
            for i in range(0, len(episode_avg_goal_norms), 10)
        ],
    }


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def _aggregate(per_seed_results: List[Dict]) -> Tuple[str, str]:
    seeds_passing  = sum(1 for r in per_seed_results if r["seed_passed"])
    dep_established = sum(1 for r in per_seed_results if r["depression_established"])
    graded_fast    = sum(
        1 for r in per_seed_results
        if r["depression_established"]
        and r["recovery_onset"] <= RECOVERY_GRADED_EPS
    )

    if seeds_passing >= PASS_SEED_COUNT:
        return "PASS", "supports"
    if dep_established < PASS_SEED_COUNT:
        return "FAIL", "non_contributory"
    if graded_fast >= PASS_SEED_COUNT:
        return "FAIL", "does_not_support"
    return "FAIL", "mixed"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    phase1_max = 5  if args.dry_run else PHASE1_MAX_EPS
    phase2_eps = 5  if args.dry_run else PHASE2_EPS
    seeds      = [42] if args.dry_run else SEEDS
    total_eps  = phase1_max + phase2_eps

    print(
        f"[V3-EXQ-435] INV-054: Phase-Transition Recovery (sustained-crossing criterion) "
        f"dry_run={args.dry_run}",
        flush=True,
    )
    print(
        f"  Phase1 max_eps={phase1_max} "
        f"(LONG_HORIZON: {LH_N_RESOURCES} res, {LH_N_HAZARDS} haz, harm={LH_HAZARD_HARM})",
        flush=True,
    )
    print(
        f"  Phase2 eps={phase2_eps} "
        f"(LOW_HARM: {LH2_N_RESOURCES} res, {LH2_N_HAZARDS} haz)",
        flush=True,
    )
    print(
        f"  Recovery: onset = first ep with {RECOVERY_SUSTAIN_EPS} consecutive "
        f"ep_avg >= {RECOVERY_NORM_THRESH} (avg over last {RECOVERY_AVG_WINDOW} steps)",
        flush=True,
    )
    print(
        f"  PASS criterion: >={PASS_SEED_COUNT}/{len(seeds)} seeds with "
        f"depression_established AND recovery_onset > {RECOVERY_PHASE_TRANS_EPS}",
        flush=True,
    )
    print(f"  seeds={seeds}", flush=True)

    per_seed_results: List[Dict] = []
    for seed in seeds:
        res = _run_seed(seed, phase1_max, phase2_eps, total_eps)
        per_seed_results.append(res)

    outcome, evidence_direction = _aggregate(per_seed_results)
    seeds_passing = sum(1 for r in per_seed_results if r["seed_passed"])

    print(f"\n[V3-EXQ-435] === Results ===", flush=True)
    for r in per_seed_results:
        print(
            f"  seed={r['seed']} "
            f"dep={r['depression_established']} "
            f"phase1_norm={r['phase1_final_z_goal_norm']:.3f} "
            f"recovery_onset={r['recovery_onset'] if r['recovery_onset'] <= PHASE2_EPS else 'none'} "
            f"type={r['recovery_type']} "
            f"passed={r['seed_passed']}",
            flush=True,
        )
    print(
        f"  seeds_passing={seeds_passing}/{len(per_seed_results)} "
        f"(need >={PASS_SEED_COUNT})",
        flush=True,
    )
    print(f"  -> {outcome} evidence_direction={evidence_direction}", flush=True)

    if args.dry_run:
        print("\n[DRY-RUN] Smoke test complete. Not writing output.", flush=True)
        return

    ts      = int(time.time())
    ts_str  = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    root    = Path(__file__).resolve().parents[2]
    out_dir = root / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"

    manifest = {
        "run_id":              f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type":     EXPERIMENT_TYPE,
        "architecture_epoch":  "ree_hybrid_guardrails_v1",
        "claim_ids":           CLAIM_IDS,
        "experiment_purpose":  EXPERIMENT_PURPOSE,
        "outcome":             outcome,
        "evidence_direction":  evidence_direction,
        "timestamp_utc":       ts_str,
        "seeds":               seeds,
        "backlog_id":          "EXP-0097",
        "supersedes":          "v3_exq_278_inv054_depression_recovery_phase_transition",
        # Parameters
        "phase1_max_episodes": PHASE1_MAX_EPS,
        "phase2_episodes":     PHASE2_EPS,
        "steps_per_ep":        STEPS_PER_EP,
        "world_dim":           WORLD_DIM,
        # Phase configs
        "lh_n_resources":      LH_N_RESOURCES,
        "lh_n_hazards":        LH_N_HAZARDS,
        "lh_hazard_harm":      LH_HAZARD_HARM,
        "lh2_n_resources":     LH2_N_RESOURCES,
        "lh2_n_hazards":       LH2_N_HAZARDS,
        # Thresholds (pre-registered)
        "depression_norm_thresh":   DEPRESSION_NORM_THRESH,
        "depression_sustain_eps":   DEPRESSION_SUSTAIN_EPS,
        "recovery_norm_thresh":     RECOVERY_NORM_THRESH,
        "recovery_sustain_eps":     RECOVERY_SUSTAIN_EPS,
        "recovery_avg_window":      RECOVERY_AVG_WINDOW,
        "recovery_phase_trans_eps": RECOVERY_PHASE_TRANS_EPS,
        "recovery_graded_eps":      RECOVERY_GRADED_EPS,
        "pass_seed_count":          PASS_SEED_COUNT,
        # Results
        "seeds_passing":       seeds_passing,
        "per_seed_results":    per_seed_results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[V3-EXQ-435] Written: {out_path}", flush=True)
    print(f"Status: {outcome}", flush=True)


if __name__ == "__main__":
    main()
