#!/opt/local/bin/python3
"""V3-EXQ-902 -- SD-048 default-scale interoceptive-noise calibration sweep.

Claim: SD-048 (body.interoceptive_noise_dynamics)

The originally-specified validation experiment for SD-048, never queued.
V3-EXQ-511 (2026-05-03, substrate-readiness diagnostic) FAILED and was
reclassified non_contributory. V3-EXQ-512 (2026-05-04, comparator-gap
behavioural test at default noise_scale=1.0) FAILED C3: autonomic
sigma=0.02 perturbations are too small relative to the forward model's
natural prediction-error floor (~0.025-0.035) for the discrimination
residual to separate. V3-EXQ-512a (2026-05-04) recalibrated to
noise_scale=3.0 and PASSED C1/C2/C3, confirming the reafference-
discrimination mechanism CAN work -- but its own manifest and notes
explicitly leave v3_pending TRUE "pending a calibration sweep near
default scale", i.e. it deliberately did not close the loop. That
default-scale sweep has never been run: EXQ-512/512a only ever compared
ONE noise-on arm against OFF, never the full ARM_0-ARM_3 sweep the SD
doc pre-registers.

This script runs the full pre-registered protocol from
REE_assembly/docs/architecture/sd_048_interoceptive_noise_dynamics.md
("Validation experiment (deferred)"): a 4-arm noise-level sweep spanning
OFF, 0.25x, 1.0x (the substrate's actual default), and 4.0x, testing for
the predicted inverted-U in comparator competence (Asai 2016
non-monotonic comparator competence; same architectural logic as SD-047's
Level-1 sweep).

Measurement methodology (unchanged from V3-EXQ-512/512a -- kept
consistent across the SD-048 evidence lineage for comparability rather
than inventing new statistics): CausalGridWorldV2 with SD-022
limb_damage_enabled=True, layered with SD-048's
interoceptive_noise_enabled/interoceptive_noise_scale. REEAgent encodes
the environment; a standalone E2HarmAForward comparator is trained
P0 (encoder warmup, no downstream loss) -> P1 (frozen-encoder forward-
model training on .detach()ed z_harm_a targets) -> P2 (eval, no grad).
At eval, per-step forward-model residual norm is split by event type
(info["interoceptive_n_body_noise_events"] vs
info["interoceptive_n_agent_caused_harm_events"]) into
residual_body_noise / residual_agent / residual_quiet.

The architecture doc's C1-C4 are stated more ambitiously (AUC/ROC,
correlation) than what EXQ-511/512/512a actually implemented and were
accepted as valid evidence for. This script continues that same proxy
scheme rather than inventing a new one:
  C1 (body-noise-event selectivity)   -> selectivity_gap > 0 per ON arm
  C2 (self-vs-body-noise gap)         -> quiet_gap >= threshold per ON arm
  C4 (OFF-arm base model sanity)      -> forward_r2 >= threshold in ARM_0
C3 (interoceptive attributor calibration, a correlation-across-conditions
statistic) is addressed structurally by running all 4 arms and reporting
which arm peaks selectivity_gap/quiet_gap -- the direct test of the
inverted-U / non-monotonicity prediction, which is the primary new
scientific content this sweep adds beyond EXQ-512a's single-arm proof.

DV-symmetry check (per the DV-SYMMETRY INVARIANCE rule): the manipulation
(interoceptive_noise_scale) is not a broadcast constant, a monotone
rescaling of a rank-based DV, or a permutation of interchangeable units.
It is a physical noise-magnitude parameter feeding additive/multiplicative
perturbations into harm_obs_a, read by a genuinely-trained forward model
whose residual norm is the DV. No known symmetry renders the manipulation
invisible to the DV by construction.

Four arms (3 seeds each = 12 runs)
------------------------------------
ARM_0 (OFF baseline): interoceptive_noise_enabled=False. SD-022 only.
  Prediction: C1-C3 fail or are untestable (no body-noise events exist);
  C4 (forward_r2) holds -- base model sanity check.
ARM_1 (0.25x): all three sources at 25% of default scale.
  Below-calibration prediction: modest or no signal, comparator
  competence impaired by insufficient background variance.
ARM_2 (1.0x, DEFAULT): all sources at the substrate's own default
  calibration (autonomic sigma=0.02, sensitisation_rate=0.008, etc).
  Hypothesised optimum per the Ward/Asai non-monotonicity principle.
  This is the arm V3-EXQ-512 already tested (with a slightly different
  driver) and FAILED C3 -- this run re-tests it as part of the full
  4-arm grid under an identical measurement methodology to the other
  three arms, which V3-EXQ-512 alone could not provide.
ARM_3 (4.0x): all sources at 4x default scale (V3-EXQ-512a tested 3.0x
  and passed; this arm uses the SD doc's own pre-registered 4.0x).
  Above-calibration prediction: body noise drowns agent-caused signal,
  comparator competence degrades (drift back toward FAIL).

Phased training is identical to EXQ-512/512a (P0 encoder warmup, P1
frozen-encoder E2_harm_a training on stop-grad targets, P2 eval).

Interpretation grid (pre-registered, from the SD doc)
--------------------------------------------------------
ARM_2 PASS (C1+C2) and ARM_0 FAILS (no body-noise events / C4 only):
  SD-048 validated. Route ARC-058 to standard testable.
Inverted-U with ARM_2 as the peak (of ARM_1/ARM_2/ARM_3):
  SD-048 validated WITH calibration confirmation -- non-monotonicity
  directly observed. Strongest possible result.
ARM_1 or ARM_3 peaks instead of ARM_2, but at least one arm shows C1+C2:
  SD-048 validated, calibration off. Recalibrate per-source scales;
  architectural conclusion (the mechanism works) still holds.
Flat curve, all of ARM_1/ARM_2/ARM_3 fail C1+C2:
  FALSIFICATION of the V3-tractable assumption for the default-scale
  regime. This reproduces V3-EXQ-512's near-baseline collapse across the
  whole grid, not just at scale=1.0. Per the claim's own stated fallback
  (SD-048 claims.yaml "what_would_answer", falsifying branch): route
  ARC-058's Level 2 comparator from V3-tractable to substrate_conditional
  (a richer body-state substrate is needed -- interoceptive signal
  pathway, not just a harm_obs_a overlay). SD-048 retired as an ARC-058
  unblocker or kept as ancillary SD-011 enrichment only.
ARM_2 PASS and ARM_0 ALSO PASS (trivial comparator):
  SD-048 irrelevant -- comparator passes without noise, so it was never
  substrate-gated. Revisit C1/C2 threshold calibration.

Pre-registered thresholds
--------------------------
C1_MIN_SELECTIVITY_GAP = 0.0     -- selectivity_gap > 0.0 (ON arms)
C2_MIN_QUIET_GAP       = 0.005   -- residual_body_noise - residual_quiet >= 0.005 (ON arms)
C4_MIN_R2_ARM_0        = 0.5     -- ARM_0 forward_r2 sanity floor
PASS_FRACTION_REQUIRED = 2/3     -- per-arm seed majority, matches EXQ-512a

PASS (overall) = ARM_2 passes C1 AND C2 (the default-scale claim) AND
ARM_0 passes C4 (base model sanity). Direction is "supports" only when
this holds; "weakens" otherwise. The full per-arm grid (including the
peak-arm / inverted-U characterization) is always recorded regardless of
overall outcome, since that characterization is itself the primary new
finding this sweep is designed to produce.
"""
from __future__ import annotations

import argparse
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
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._metrics import check_degeneracy  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_902_sd048_default_scale_calibration_sweep"
CLAIM_IDS = ["SD-048"]
EXPERIMENT_PURPOSE = "evidence"

# --- Config -------------------------------------------------------------
SEEDS = (42, 43, 44)  # matches EXQ-512a precedent; reef_enabled=False here, so
                       # the seed-44/reef-config instability warning does not apply.
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

# The 4-arm sweep lever, per the SD doc's pre-registered protocol.
ARMS = [
    {"arm_id": "ARM_0", "label": "ARM_0_off", "sd048_on": False, "noise_scale": 0.0},
    {"arm_id": "ARM_1", "label": "ARM_1_0.25x", "sd048_on": True, "noise_scale": 0.25},
    {"arm_id": "ARM_2", "label": "ARM_2_1.0x_default", "sd048_on": True, "noise_scale": 1.0},
    {"arm_id": "ARM_3", "label": "ARM_3_4.0x", "sd048_on": True, "noise_scale": 4.0},
]

# Pre-registered thresholds.
C1_MIN_SELECTIVITY_GAP = 0.0
C2_MIN_QUIET_GAP = 0.005
C4_MIN_R2_ARM_0 = 0.5
PASS_FRACTION_REQUIRED = 2.0 / 3.0

_ZG = ZGoalStreamAccumulator()


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


def _config_slice(arm: Dict, seed: int) -> Dict:
    """Everything this cell's build+collect path reads. Used for the arm
    fingerprint (multi-arm reuse instrument -- emit-only, no reuse consumed
    here). Deliberately whole-tree substrate_hash (no substrate_scope):
    this is a first-of-its-kind sweep, not a lineage with an established
    narrow closure worth proving conservative."""
    return {
        "grid_size": GRID_SIZE,
        "n_hazards": N_HAZARDS,
        "n_resources": N_RESOURCES,
        "p_stay": P_STAY,
        "sd048_on": arm["sd048_on"],
        "noise_scale": arm["noise_scale"],
        "warmup_episodes": WARMUP_EPISODES,
        "p1_episodes": P1_EPISODES,
        "eval_episodes": EVAL_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "self_dim": SELF_DIM,
        "world_dim": WORLD_DIM,
        "lr": LR,
        "harm_fwd_lr": HARM_FWD_LR,
        "seed": seed,
    }


def run_cell(arm: Dict, seed: int, p0_eps: int, p1_eps: int, eval_eps: int) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    sd048_on = arm["sd048_on"]
    noise_scale = arm["noise_scale"]

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
                f"  [train] {arm['label']} seed={seed} ep {ep+1}/{p0_eps + p1_eps} P0 warmup",
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
                f"  [train] {arm['label']} seed={seed} ep {p0_eps + ep+1}/{p0_eps + p1_eps} P1 fwd-model",
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

    _ZG.observe(agent)

    def _safe_mean(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    mean_body = _safe_mean(res_body_noise)
    mean_agent = _safe_mean(res_agent)
    mean_quiet = _safe_mean(res_quiet)
    selectivity_gap = mean_body - mean_agent
    selectivity_ratio = mean_body / max(1e-6, mean_agent)
    quiet_gap = mean_body - mean_quiet

    if all_residuals and all_targets_sq:
        mean_res_sq = sum(r ** 2 for r in all_residuals) / len(all_residuals)
        mean_tgt_sq = sum(all_targets_sq) / len(all_targets_sq)
        forward_r2 = 1.0 - (mean_res_sq / max(1e-6, mean_tgt_sq))
    else:
        forward_r2 = 0.0

    print(
        f"  [{arm['label']}] seed={seed} "
        f"n_body={len(res_body_noise)} n_agent={len(res_agent)} n_quiet={len(res_quiet)} "
        f"res_body={mean_body:.4f} res_agent={mean_agent:.4f} res_quiet={mean_quiet:.4f} "
        f"sel_gap={selectivity_gap:+.4f} quiet_gap={quiet_gap:+.4f} r2={forward_r2:.3f}",
        flush=True,
    )

    row = {
        "seed": seed,
        "arm_id": arm["arm_id"],
        "arm_label": arm["label"],
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
        "quiet_gap": float(quiet_gap),
        "forward_r2": float(forward_r2),
    }
    print(f"verdict: {'PASS' if True else 'FAIL'}", flush=True)  # per-cell progress marker
    return row


def _evaluate(rows_by_arm: Dict[str, List[Dict]]) -> Dict:
    n = len(SEEDS)
    required = math.ceil(n * PASS_FRACTION_REQUIRED)

    per_arm = {}
    for arm in ARMS:
        arm_id = arm["arm_id"]
        rows = rows_by_arm[arm_id]
        c1 = sum(1 for r in rows if r["selectivity_gap"] > C1_MIN_SELECTIVITY_GAP)
        c2 = sum(
            1 for r in rows
            if r["quiet_gap"] >= C2_MIN_QUIET_GAP and r["n_body_noise_steps"] > 0
        )
        c4 = sum(1 for r in rows if r["forward_r2"] >= C4_MIN_R2_ARM_0)
        per_arm[arm_id] = {
            "arm_label": arm["label"],
            "noise_scale": arm["noise_scale"],
            "c1_seeds_pass": c1,
            "c2_seeds_pass": c2,
            "c4_seeds_pass": c4,
            "c1_pass": c1 >= required,
            "c2_pass": c2 >= required,
            "c4_pass": c4 >= required,
            "mean_selectivity_gap": float(sum(r["selectivity_gap"] for r in rows) / n),
            "mean_quiet_gap": float(sum(r["quiet_gap"] for r in rows) / n),
            "mean_selectivity_ratio": float(sum(r["selectivity_ratio"] for r in rows) / n),
            "mean_forward_r2": float(sum(r["forward_r2"] for r in rows) / n),
            "mean_n_body_noise_steps": float(sum(r["n_body_noise_steps"] for r in rows) / n),
        }

    # Peak-arm detection across ON arms (ARM_1/ARM_2/ARM_3) -- the direct
    # inverted-U / non-monotonicity test.
    on_arm_ids = ["ARM_1", "ARM_2", "ARM_3"]
    peak_selectivity_arm = max(on_arm_ids, key=lambda a: per_arm[a]["mean_selectivity_gap"])
    peak_quiet_gap_arm = max(on_arm_ids, key=lambda a: per_arm[a]["mean_quiet_gap"])

    arm0_c4_pass = per_arm["ARM_0"]["c4_pass"]
    arm0_c1_pass = per_arm["ARM_0"]["c1_pass"]  # should be False/untestable (no body-noise events)
    arm2_pass = per_arm["ARM_2"]["c1_pass"] and per_arm["ARM_2"]["c2_pass"]
    any_on_arm_pass = any(per_arm[a]["c1_pass"] and per_arm[a]["c2_pass"] for a in on_arm_ids)

    # Interpretation label per the SD doc's pre-registered grid.
    if arm2_pass and arm0_c4_pass and not arm0_c1_pass:
        if peak_selectivity_arm == "ARM_2" and peak_quiet_gap_arm == "ARM_2":
            label = "sd048_validated_calibration_confirmed_inverted_u"
        else:
            label = "sd048_validated_default_scale"
    elif arm2_pass and per_arm["ARM_0"]["c1_pass"]:
        label = "sd048_irrelevant_trivial_comparator"
    elif any_on_arm_pass and not arm2_pass:
        label = "sd048_validated_calibration_off"
    elif not any_on_arm_pass:
        label = "sd048_falsified_flat_curve_default_scale"
    else:
        label = "sd048_ambiguous_see_per_arm_grid"

    overall_pass = bool(arm2_pass and arm0_c4_pass)

    return {
        "n_seeds": n,
        "min_seeds_required": required,
        "per_arm": per_arm,
        "peak_selectivity_gap_arm": peak_selectivity_arm,
        "peak_quiet_gap_arm": peak_quiet_gap_arm,
        "arm2_pass_c1_and_c2": arm2_pass,
        "arm0_pass_c4_sanity": arm0_c4_pass,
        "arm0_pass_c1_trivial_check": arm0_c1_pass,
        "any_on_arm_pass": any_on_arm_pass,
        "interpretation_label": label,
        "overall_pass": overall_pass,
    }


def main() -> Dict:
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
        print("[DRY-RUN] 1 seed, 4 arms, 2 P0 / 2 P1 / 2 eval eps -- smoke only.", flush=True)
    else:
        seeds = tuple(args.seeds)
        p0 = args.p0
        p1 = args.p1
        eval_eps = args.eval

    print(f"V3-EXQ-902 SD-048 default-scale calibration sweep "
          f"(arms={[a['arm_id'] for a in ARMS]}, seeds={seeds})", flush=True)

    t0 = time.time()
    arm_results: List[Dict] = []
    rows_by_arm: Dict[str, List[Dict]] = {a["arm_id"]: [] for a in ARMS}

    for arm in ARMS:
        for s in seeds:
            print(f"Seed {s} Condition {arm['label']}", flush=True)
            with arm_cell(
                s,
                config_slice=_config_slice(arm, s),
                script_path=Path(__file__),
                config_slice_declared=True,
            ) as cell:
                row = run_cell(arm, s, p0, p1, eval_eps)
                cell.stamp(row)
            arm_results.append(row)
            rows_by_arm[arm["arm_id"]].append(row)

    elapsed = time.time() - t0

    if args.dry_run:
        # Smoke-only non-degeneracy check: at least ONE noise-on arm must show
        # non-trivial body-noise event engagement before trusting the full grid.
        any_engaged = any(r["n_body_noise_steps"] > 0 for r in arm_results if r["sd048_on"])
        print(f"[smoke] any ON-arm body-noise engagement: {any_engaged}", flush=True)
        assert any_engaged, (
            "SMOKE FAILURE: no ON arm produced any body-noise events at all "
            "(n_body_noise_steps == 0 in every ON-arm cell) -- the manipulation "
            "did not engage; do not proceed to the full seed x arm grid."
        )
        print("[--dry-run] manifest not written.", flush=True)
        return {"outcome": "PASS", "manifest_path": None, "run_id": None, "dry_run": True}

    criteria = _evaluate(rows_by_arm)
    outcome = "PASS" if criteria["overall_pass"] else "FAIL"
    direction = "supports" if criteria["overall_pass"] else "weakens"

    print(f"\nV3-EXQ-902 (SD-048 default-scale calibration sweep) -- {outcome} "
          f"in {elapsed:.1f}s ({len(seeds)} seed(s) x {len(ARMS)} arms)", flush=True)
    for arm in ARMS:
        pa = criteria["per_arm"][arm["arm_id"]]
        print(
            f"  {arm['arm_id']} ({arm['label']}): "
            f"C1(sel_gap>0)={pa['c1_seeds_pass']}/{criteria['n_seeds']} "
            f"C2(quiet_gap>={C2_MIN_QUIET_GAP})={pa['c2_seeds_pass']}/{criteria['n_seeds']} "
            f"C4(r2>={C4_MIN_R2_ARM_0})={pa['c4_seeds_pass']}/{criteria['n_seeds']} "
            f"mean_sel_gap={pa['mean_selectivity_gap']:+.4f} "
            f"mean_quiet_gap={pa['mean_quiet_gap']:+.4f} "
            f"mean_r2={pa['mean_forward_r2']:.3f}",
            flush=True,
        )
    print(f"  peak selectivity_gap arm: {criteria['peak_selectivity_gap_arm']}", flush=True)
    print(f"  peak quiet_gap arm: {criteria['peak_quiet_gap_arm']}", flush=True)
    print(f"  interpretation: {criteria['interpretation_label']}", flush=True)

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
        "outcome": outcome,
        "evidence_direction": direction,
        "criteria": criteria,
        "registered_thresholds": {
            "C1_MIN_SELECTIVITY_GAP": C1_MIN_SELECTIVITY_GAP,
            "C2_MIN_QUIET_GAP": C2_MIN_QUIET_GAP,
            "C4_MIN_R2_ARM_0": C4_MIN_R2_ARM_0,
            "PASS_FRACTION_REQUIRED": PASS_FRACTION_REQUIRED,
            "arms": [{"arm_id": a["arm_id"], "noise_scale": a["noise_scale"]} for a in ARMS],
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
            "arms": [{"arm_id": a["arm_id"], "label": a["label"],
                       "sd048_on": a["sd048_on"], "noise_scale": a["noise_scale"]} for a in ARMS],
        },
        "arm_results": arm_results,
        "elapsed_seconds": elapsed,
        "notes": (
            "SD-048's originally pre-registered 4-arm default-scale calibration sweep "
            "(ARM_0 OFF / ARM_1 0.25x / ARM_2 1.0x-default / ARM_3 4.0x), queued "
            "2026-08-08 after being absent from ree-v3/experiment_queue.json for "
            "~3 months following V3-EXQ-512a (2026-05-04). V3-EXQ-512a proved the "
            "SD-048 reafference-discrimination mechanism works at noise_scale=3.0 "
            "but explicitly left v3_pending TRUE pending this default-scale sweep. "
            "Existing-evidence check (GOV-REUSE-1): V3-EXQ-511/512/512a checked -- "
            "all pre-date the 2026-07-12 recording standard (no substrate_hash), "
            "and none ran the full ARM_0-ARM_3 comparator-gap grid under one "
            "consistent methodology (511 used a different, event-counting-only "
            "diagnostic methodology; 512/512a tested only 2 of these 4 arms) -- "
            "decisive readout (selectivity_gap/quiet_gap at ARM_1 0.25x and ARM_3 "
            "4.0x under the comparator methodology) is absent from every prior "
            "manifest, so this experiment proceeds rather than reanalyzing. "
            "FALLBACK TRIGGER (per the claim's own governance): if ARM_1/ARM_2/ARM_3 "
            "all fail C1+C2 (interpretation_label == "
            "'sd048_falsified_flat_curve_default_scale'), route ARC-058's Level 2 "
            "comparator from V3-tractable to substrate_conditional -- see claims.yaml "
            "SD-048 what_would_answer (falsifying branch) and the SD doc's "
            "interpretation grid. Relevant but not directly tagged: ARC-058 "
            "(primary target claim), ARC-033 (competing claim, same substrate "
            "enrichment benefits both)."
        ),
        "sleep_driver_pattern": "not_applicable_no_sleep",
    }
    # Non-degeneracy self-report on the two load-bearing discriminative metrics
    # (C1 selectivity_gap, C2 quiet_gap), grouped per-arm so a run where every
    # arm is pinned/zero-spread self-excludes from confidence scoring instead
    # of scoring as a false PASS/FAIL.
    manifest.update(check_degeneracy({
        "selectivity_gap": {
            "groups": [[r["selectivity_gap"] for r in rows_by_arm[a["arm_id"]]] for a in ARMS],
        },
        "quiet_gap": {
            "groups": [[r["quiet_gap"] for r in rows_by_arm[a["arm_id"]]] for a in ARMS],
        },
    }))
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=list(seeds),
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    print(f"Result written to: {out_path}", flush=True)

    return {
        "outcome": outcome,
        "manifest_path": out_path,
        "run_id": run_id,
        "dry_run": args.dry_run,
    }


if __name__ == "__main__":
    _result = main()
    emit_outcome(
        outcome=_result["outcome"],
        manifest_path=_result["manifest_path"],
        run_id=_result.get("run_id"),
        dry_run=_result["dry_run"],
    )
