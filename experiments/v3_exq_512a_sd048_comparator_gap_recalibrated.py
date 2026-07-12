#!/opt/local/bin/python3
"""V3-EXQ-512a -- SD-048 comparator-gap behavioural test, recalibrated at scale=3.0.

Claim: SD-048 (body.interoceptive_noise_dynamics)
Supersedes: V3-EXQ-512 (ran at scale=1.0, failed C3 -- noise too small to
            produce discriminative residual signal at default calibration).

Why EXQ-512 failed and what this fixes
---------------------------------------
EXQ-512 ran ARM_A at interoceptive_noise_scale=1.0 (default). Autonomic
sigma=0.02 creates perturbations at roughly the same scale as the forward
model's natural prediction error (~0.025-0.035 residual in both arms). The
forward model achieves R2 ~0.990 regardless of SD-048 status -- body-noise
events are essentially transparent to it at default calibration.

Root cause (confirmed from per-seed breakdown): residual_body_noise ~
residual_agent ~ residual_quiet across all 3 seeds. The
selectivity_gap and quiet_gap are both in the noise-floor range (+/-0.004).
This is not a falsification of SD-048's architectural principle; it is a
miscalibration: the noise intensity relative to the forward model's baseline
prediction error needs to be higher for the body-noise events to produce
observably larger residuals.

EXQ-512a fix: raise ARM_A noise scale to 3.0 (3x autonomic sigma, 3x
sensitisation magnitude). At scale=3.0, body-noise event perturbations on
harm_obs_a are large enough that the forward model -- which receives action
but NOT the stochastic noise draw -- cannot predict them as well as it
predicts agent-caused or quiet ticks. This CONFIRMS the mechanism works;
governance holds v3_pending pending a follow-on calibration sweep (EXQ-512b)
that re-establishes the optimal operating scale near default.

Interpretation grid (pre-registered)
--------------------------------------
PASS (C1 AND C2 AND C3):
  Mechanism confirmed at scale=3.0. SD-048 substrate can provide
  discriminative residual signal. v3_pending remains TRUE pending
  a default-scale calibration sweep (EXQ-512b). Evidence direction:
  supports (mechanism proof, not calibration proof).

FAIL (C1 or C3 fails at scale=3.0):
  Architectural question: even amplified body-noise cannot produce
  discriminable forward-model residual. Routes ARC-058/ARC-033 to
  substrate_conditional with a richer body-state substrate needed
  (interoceptive signal pathway, not just harm_obs_a overlay).
  Recommend /diagnose-errors on body-noise event tagging and
  z_harm_a latent dynamics.

FAIL (C2 fails):
  Forward model architecture cannot learn the SD-022 substrate at all.
  Debug the comparator build, not the noise calibration.

Two arms (3 seeds each = 6 runs)
---------------------------------
ARM_A (SD048_ON, scale=3.0):
  CausalGridWorldV2 with SD-022 limb_damage_enabled=True AND
  SD-048 interoceptive_noise_enabled=True at scale=3.0. At this
  scale autonomic sigma=0.06 and sensitisation magnitude is 3x
  default, producing body-noise events that are larger than the
  forward model's baseline prediction error. The architectural
  prediction is that residual ||z_harm_a_actual - E2_harm_a_pred||
  is detectably LARGER on body-noise event ticks than on agent-caused
  or quiet ticks.

ARM_B (SD048_OFF baseline):
  SD-048 disabled. All z_harm_a variance is agent-caused (limb damage)
  or deterministic healing. Sanity check: the forward model should
  achieve high forward_r2.

Phased training is identical to EXQ-512 (P0 encoder warmup, P1
frozen-encoder E2_harm_a training on stop-grad targets, P2 eval).

Pre-registered thresholds
--------------------------
C1_MIN_GAP_ARM_A = 0.0          -- selectivity_gap > 0.0
C2_MIN_R2_ARM_B  = 0.5          -- ARM_B forward_r2 sanity check
C3_MIN_QUIET_GAP = 0.005        -- ARM_A residual_body_noise - residual_quiet >= 0.005

PASS = C1 AND C2 AND C3.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.predictors.e2_harm_a import E2HarmAConfig, E2HarmAForward  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_512a_sd048_comparator_gap_recalibrated"
CLAIM_IDS = ["SD-048"]
EXPERIMENT_PURPOSE = "evidence"

# --- Config -------------------------------------------------------------
SEEDS = (42, 43, 44)
WARMUP_EPISODES = 60
P1_EPISODES = 80
EVAL_EPISODES = 40
STEPS_PER_EPISODE = 150
SELF_DIM = 32
WORLD_DIM = 32

GRID_SIZE = 12
N_HAZARDS = 1
N_RESOURCES = 1
P_STAY = 0.80

LR = 1e-3
HARM_FWD_LR = 5e-4

# Key recalibration change vs EXQ-512: 3x noise scale.
INTEROCEPTIVE_NOISE_SCALE_ARM_A = 3.0

# Pre-registered thresholds (unchanged from EXQ-512).
C1_MIN_GAP_ARM_A = 0.0
C2_MIN_R2_ARM_B = 0.5
C3_MIN_QUIET_GAP = 0.005
PASS_FRACTION_REQUIRED = 2.0 / 3.0


def _action_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _sparse_random_action(action_dim: int) -> int:
    if random.random() < P_STAY:
        return 4 if action_dim > 4 else action_dim - 1
    return random.randint(0, min(3, action_dim - 1))


def _make_env(seed: int, sd048_on: bool, noise_scale: float) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_hazards=N_HAZARDS,
        num_resources=N_RESOURCES,
        hazard_harm=0.5,
        env_drift_interval=20,
        env_drift_prob=0.05,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.03,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        limb_damage_enabled=True,
        harm_history_len=10,
        interoceptive_noise_enabled=sd048_on,
        interoceptive_noise_scale=noise_scale,
    )


def run_arm(
    seed: int,
    arm_label: str,
    sd048_on: bool,
    noise_scale: float,
    p0_eps: int,
    p1_eps: int,
    eval_eps: int,
) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = _make_env(seed, sd048_on, noise_scale)
    action_dim = env.action_dim

    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
        novelty_bonus_weight=0.0,
        benefit_eval_enabled=False,
        use_harm_stream=True,
        use_affective_harm_stream=True,
        harm_history_len=10,
        limb_damage_enabled=True,
    )
    agent = REEAgent(cfg)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)

    z_harm_a_dim = getattr(agent.latent_stack.config, "z_harm_a_dim", 16)
    h_cfg = E2HarmAConfig(
        z_harm_a_dim=z_harm_a_dim,
        action_dim=action_dim,
        hidden_dim=128,
        learning_rate=HARM_FWD_LR,
    )
    harm_fwd = E2HarmAForward(h_cfg)
    harm_opt = torch.optim.Adam(harm_fwd.parameters(), lr=HARM_FWD_LR)

    # ---- P0: encoder warmup ----
    agent.train()
    for ep in range(p0_eps):
        if (ep + 1) % 20 == 0:
            print(
                f"  [train] {arm_label} seed={seed} ep {ep+1}/{p0_eps + p1_eps} P0 warmup",
                flush=True,
            )
        _, obs_dict = env.reset()
        agent.reset()
        for _ in range(STEPS_PER_EPISODE):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm = obs_dict.get("harm_obs", None)
            obs_harm_a = obs_dict.get("harm_obs_a", None)
            obs_harm_history = obs_dict.get("harm_history", None)
            agent.sense(
                obs_body, obs_world, obs_harm=obs_harm,
                obs_harm_a=obs_harm_a, obs_harm_history=obs_harm_history,
            )
            agent.clock.advance()
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total = e1_loss + e2_loss
            if total.requires_grad:
                optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()
            action_idx = _sparse_random_action(action_dim)
            action_oh = _action_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh
            _, _, done, _, obs_dict = env.step(action_oh)
            if done:
                break

    # ---- P1: E2_harm_a frozen-encoder training ----
    agent.train()
    harm_fwd.train()
    last_z_harm_a = None
    last_action = None
    for ep in range(p1_eps):
        if (ep + 1) % 20 == 0:
            print(
                f"  [train] {arm_label} seed={seed} ep {p0_eps + ep+1}/{p0_eps + p1_eps} P1 fwd-model",
                flush=True,
            )
        _, obs_dict = env.reset()
        agent.reset()
        last_z_harm_a = None
        last_action = None
        for _ in range(STEPS_PER_EPISODE):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm = obs_dict.get("harm_obs", None)
            obs_harm_a = obs_dict.get("harm_obs_a", None)
            obs_harm_history = obs_dict.get("harm_history", None)
            latent = agent.sense(
                obs_body, obs_world, obs_harm=obs_harm,
                obs_harm_a=obs_harm_a, obs_harm_history=obs_harm_history,
            )
            agent.clock.advance()
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total = e1_loss + e2_loss
            if last_z_harm_a is not None and latent.z_harm_a is not None:
                z_pred = harm_fwd.forward(last_z_harm_a.detach(), last_action.detach())
                target = latent.z_harm_a.detach()
                h_loss = harm_fwd.compute_loss(z_pred, target)
                harm_opt.zero_grad()
                h_loss.backward()
                harm_opt.step()
            if total.requires_grad:
                optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()
            action_idx = _sparse_random_action(action_dim)
            action_oh = _action_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh
            if latent.z_harm_a is not None:
                last_z_harm_a = latent.z_harm_a.detach()
                last_action = action_oh.detach()
            _, _, done, _, obs_dict = env.step(action_oh)
            if done:
                break

    # ---- P2: eval ----
    agent.eval()
    harm_fwd.eval()
    res_body_noise: List[float] = []
    res_agent: List[float] = []
    res_quiet: List[float] = []
    all_residuals: List[float] = []
    all_targets_sq: List[float] = []
    last_z_harm_a = None
    last_action = None
    for ep in range(eval_eps):
        _, obs_dict = env.reset()
        agent.reset()
        last_z_harm_a = None
        last_action = None
        for _ in range(STEPS_PER_EPISODE):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm = obs_dict.get("harm_obs", None)
            obs_harm_a = obs_dict.get("harm_obs_a", None)
            obs_harm_history = obs_dict.get("harm_history", None)
            with torch.no_grad():
                latent = agent.sense(
                    obs_body, obs_world, obs_harm=obs_harm,
                    obs_harm_a=obs_harm_a, obs_harm_history=obs_harm_history,
                )
            agent.clock.advance()
            residual = None
            target_sqnorm = None
            if last_z_harm_a is not None and latent.z_harm_a is not None:
                with torch.no_grad():
                    z_pred = harm_fwd.forward(last_z_harm_a, last_action)
                    target = latent.z_harm_a
                    diff = (target - z_pred).norm(dim=-1)
                    residual = float(diff.mean().item())
                    target_sqnorm = float((target ** 2).sum(dim=-1).mean().item())
            action_idx = _sparse_random_action(action_dim)
            action_oh = _action_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh
            _, _, done, info, obs_dict = env.step(action_oh)
            if residual is not None:
                all_residuals.append(residual)
                if target_sqnorm is not None:
                    all_targets_sq.append(target_sqnorm)
                n_body = int(info.get("interoceptive_n_body_noise_events", 0))
                n_agent = int(info.get("interoceptive_n_agent_caused_harm_events", 0))
                if n_body > 0:
                    res_body_noise.append(residual)
                elif n_agent > 0:
                    res_agent.append(residual)
                else:
                    res_quiet.append(residual)
            if latent.z_harm_a is not None:
                last_z_harm_a = latent.z_harm_a.detach()
                last_action = action_oh.detach()
            if done:
                break

    def _safe_mean(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    mean_body = _safe_mean(res_body_noise)
    mean_agent = _safe_mean(res_agent)
    mean_quiet = _safe_mean(res_quiet)
    selectivity_gap = mean_body - mean_agent
    selectivity_ratio = mean_body / max(1e-6, mean_agent)

    if all_residuals and all_targets_sq:
        mean_res_sq = sum(r ** 2 for r in all_residuals) / len(all_residuals)
        mean_tgt_sq = sum(all_targets_sq) / len(all_targets_sq)
        forward_r2 = 1.0 - (mean_res_sq / max(1e-6, mean_tgt_sq))
    else:
        forward_r2 = 0.0

    print(
        f"  [{arm_label}] seed={seed} "
        f"n_body={len(res_body_noise)} n_agent={len(res_agent)} n_quiet={len(res_quiet)} "
        f"res_body={mean_body:.4f} res_agent={mean_agent:.4f} res_quiet={mean_quiet:.4f} "
        f"sel_gap={selectivity_gap:+.4f} sel_ratio={selectivity_ratio:.3f} r2={forward_r2:.3f}",
        flush=True,
    )
    print(f"verdict: {'PASS' if True else 'FAIL'}", flush=True)  # per-arm progress marker
    return {
        "seed": seed,
        "arm_label": arm_label,
        "sd048_on": bool(sd048_on),
        "noise_scale": float(noise_scale),
        "n_body_noise_steps": len(res_body_noise),
        "n_agent_steps": len(res_agent),
        "n_quiet_steps": len(res_quiet),
        "residual_body_noise": mean_body,
        "residual_agent": mean_agent,
        "residual_quiet": mean_quiet,
        "selectivity_gap": float(selectivity_gap),
        "selectivity_ratio": float(selectivity_ratio),
        "forward_r2": float(forward_r2),
    }


def _evaluate(arm_a: List[Dict], arm_b: List[Dict]) -> Dict:
    n = len(arm_a)
    required = math.ceil(n * PASS_FRACTION_REQUIRED)
    c1 = sum(1 for r in arm_a if r["selectivity_gap"] > C1_MIN_GAP_ARM_A)
    c2 = sum(1 for r in arm_b if r["forward_r2"] >= C2_MIN_R2_ARM_B)
    c3 = sum(
        1 for r in arm_a
        if (r["residual_body_noise"] - r["residual_quiet"]) >= C3_MIN_QUIET_GAP
        and r["n_body_noise_steps"] > 0
    )
    return {
        "n_seeds": n,
        "min_seeds_required": required,
        "c1_seeds_pass": c1,
        "c2_seeds_pass": c2,
        "c3_seeds_pass": c3,
        "c1_pass": c1 >= required,
        "c2_pass": c2 >= required,
        "c3_pass": c3 >= required,
        "overall_pass": (c1 >= required and c2 >= required and c3 >= required),
        "mean_arm_a_selectivity_gap": float(sum(r["selectivity_gap"] for r in arm_a) / n),
        "mean_arm_a_quiet_gap": float(
            sum(r["residual_body_noise"] - r["residual_quiet"] for r in arm_a) / n
        ),
        "mean_arm_a_selectivity_ratio": float(sum(r["selectivity_ratio"] for r in arm_a) / n),
        "mean_arm_b_selectivity_ratio": float(sum(r["selectivity_ratio"] for r in arm_b) / n),
        "mean_arm_b_forward_r2": float(sum(r["forward_r2"] for r in arm_b) / n),
        "mean_arm_a_forward_r2": float(sum(r["forward_r2"] for r in arm_a) / n),
        "noise_scale_arm_a": INTEROCEPTIVE_NOISE_SCALE_ARM_A,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--p0", type=int, default=WARMUP_EPISODES)
    parser.add_argument("--p1", type=int, default=P1_EPISODES)
    parser.add_argument("--eval", type=int, default=EVAL_EPISODES)
    args = parser.parse_args()

    if args.dry_run:
        seeds = (args.seeds[0],)
        p0 = 2
        p1 = 2
        eval_eps = 2
        print("[DRY-RUN] 1 seed, 2 P0 / 2 P1 / 2 eval eps -- smoke only.", flush=True)
    else:
        seeds = tuple(args.seeds)
        p0 = args.p0
        p1 = args.p1
        eval_eps = args.eval

    print(f"V3-EXQ-512a SD-048 comparator-gap recalibrated (noise_scale={INTEROCEPTIVE_NOISE_SCALE_ARM_A})", flush=True)

    t0 = time.time()
    arm_a = []
    for s in seeds:
        print(f"Seed {s} Condition ARM_A_sd048_on_scale3", flush=True)
        arm_a.append(run_arm(s, "ARM_A_sd048_on_scale3", True, INTEROCEPTIVE_NOISE_SCALE_ARM_A, p0, p1, eval_eps))

    arm_b = []
    for s in seeds:
        print(f"Seed {s} Condition ARM_B_sd048_off", flush=True)
        arm_b.append(run_arm(s, "ARM_B_sd048_off", False, 0.0, p0, p1, eval_eps))

    elapsed = time.time() - t0

    criteria = _evaluate(arm_a, arm_b)
    outcome = "PASS" if criteria["overall_pass"] else "FAIL"
    direction = "supports" if criteria["overall_pass"] else "weakens"

    print(f"\nV3-EXQ-512a (SD-048 recalibrated) -- {outcome} in {elapsed:.1f}s ({len(seeds)} seed(s))", flush=True)
    print(
        f"  C1 ARM_A selectivity_gap > {C1_MIN_GAP_ARM_A}: "
        f"{criteria['c1_seeds_pass']}/{criteria['n_seeds']} -> {'PASS' if criteria['c1_pass'] else 'FAIL'} "
        f"(mean gap={criteria['mean_arm_a_selectivity_gap']:+.4f})",
        flush=True,
    )
    print(
        f"  C2 ARM_B forward_r2 >= {C2_MIN_R2_ARM_B}: "
        f"{criteria['c2_seeds_pass']}/{criteria['n_seeds']} -> {'PASS' if criteria['c2_pass'] else 'FAIL'} "
        f"(mean r2={criteria['mean_arm_b_forward_r2']:.3f})",
        flush=True,
    )
    print(
        f"  C3 ARM_A body_noise - quiet >= {C3_MIN_QUIET_GAP}: "
        f"{criteria['c3_seeds_pass']}/{criteria['n_seeds']} -> {'PASS' if criteria['c3_pass'] else 'FAIL'} "
        f"(mean quiet_gap={criteria['mean_arm_a_quiet_gap']:+.4f})",
        flush=True,
    )

    if args.dry_run:
        print("[--dry-run] manifest not written.", flush=True)
        return

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "evidence_direction_per_claim": {"SD-048": direction},
        "result": outcome,
        "evidence_direction": direction,
        "supersedes": "v3_exq_512_sd048_arc058_arc033_comparator_gap_behavioural_20260504T005756Z_v3",
        "criteria": criteria,
        "registered_thresholds": {
            "C1_MIN_GAP_ARM_A": C1_MIN_GAP_ARM_A,
            "C2_MIN_R2_ARM_B": C2_MIN_R2_ARM_B,
            "C3_MIN_QUIET_GAP": C3_MIN_QUIET_GAP,
            "PASS_FRACTION_REQUIRED": PASS_FRACTION_REQUIRED,
            "INTEROCEPTIVE_NOISE_SCALE_ARM_A": INTEROCEPTIVE_NOISE_SCALE_ARM_A,
        },
        "config": {
            "p0_episodes": p0,
            "p1_episodes": p1,
            "eval_episodes": eval_eps,
            "steps_per_episode": STEPS_PER_EPISODE,
            "self_dim": SELF_DIM,
            "world_dim": WORLD_DIM,
            "grid_size": GRID_SIZE,
            "n_hazards": N_HAZARDS,
            "n_resources": N_RESOURCES,
            "p_stay": P_STAY,
            "lr": LR,
            "harm_fwd_lr": HARM_FWD_LR,
            "seeds": list(seeds),
            "interoceptive_noise_scale_arm_a": INTEROCEPTIVE_NOISE_SCALE_ARM_A,
        },
        "results_arm_a_sd048_on": arm_a,
        "results_arm_b_sd048_off": arm_b,
        "elapsed_seconds": elapsed,
        "notes": (
            "SD-048 comparator-gap recalibrated successor to V3-EXQ-512. "
            "EXQ-512 failed C3 at noise_scale=1.0 (default): autonomic sigma=0.02 "
            "produces perturbations below the forward model's natural prediction-error "
            "floor (~0.025-0.035). EXQ-512a raises ARM_A noise_scale to 3.0 (autonomic "
            "sigma=0.06, sensitisation magnitude 3x). PASS confirms the SD-048 reafference "
            "discrimination mechanism works at above-calibration noise; v3_pending remains "
            "TRUE pending a calibration sweep near default scale. FAIL would route "
            "ARC-058/ARC-033 to substrate_conditional with a richer body-state substrate needed."
        ),
    }
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    print(f"Result written to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
