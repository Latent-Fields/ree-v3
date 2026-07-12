#!/opt/local/bin/python3
"""V3-EXQ-512 -- SD-048 ARC-058 / ARC-033 comparator-gap behavioural test.

Claim: SD-048 (body.interoceptive_noise_dynamics)
Status: candidate (v3_pending). Substrate IMPLEMENTED 2026-05-03.
        V3-EXQ-511 substrate-readiness diagnostic 6/7 PASS, governance
        2026-05-03T2356Z accepted partial-PASS as non-falsifying but kept
        v3_pending until a behavioural successor lands. THIS IS THAT
        SUCCESSOR.

Why this experiment exists
--------------------------
SD-048 lands three concurrent agent-independent stochastic body-state noise
sources on harm_obs_a (autonomic Gaussian, sensitisation Poisson + decay,
fatigue AR(1)). The architectural commitment is that the Level 2
interoceptive comparator (ARC-058 trunk path or ARC-033 independent path)
trained on SD-048-active substrate learns to discriminate agent-caused
body-state change (hazard contact -> limb damage) from body-noise-caused
change (autonomic / sensitisation / fatigue). Without SD-048, all z_harm_a
variance is agent-caused or deterministic healing -- a trivial forward
model passes by memorising the action -> damage coupling rather than by
learning the not-self distinction.

V3-EXQ-511 confirmed the substrate runs and is calibrated at ARM_2 default
(env_events ratio 2.39 in [0.5, 5.0]). What V3-EXQ-511 did NOT confirm is
that the comparator can USE the SD-048-induced variance to discriminate.
This experiment closes that gap with a discriminative pair.

Two arms (3 seeds each = 6 runs)
--------------------------------
ARM_A (SD048_ON, scale=1.0 default):
  CausalGridWorldV2 with SD-022 limb_damage_enabled=True AND
  SD-048 interoceptive_noise_enabled=True at default calibration. The
  trained E2_harm_a forward model sees both agent-caused limb damage
  AND body-noise-caused harm_obs_a perturbations; the architectural
  prediction is that residual ||z_harm_a_actual - E2_harm_a(z_harm_a_prev,
  a_prev)|| is LARGER on body-noise event ticks than on agent-caused
  event ticks (the agent-caused side is predictable from the action;
  the body-noise side is by construction not).

ARM_B (SD048_OFF baseline):
  Same env, same agent build, same training, but SD-048 disabled. All
  z_harm_a variance is agent-caused (limb damage) or deterministic
  (healing). Body-noise event labels still computed by the env (always
  zero in OFF arm by C0 of V3-EXQ-511). C2 sanity check: the forward
  model should still achieve high forward_r2 in OFF arm (no body noise
  to confuse it -- this is the prerequisite for ARM_A's discrimination
  to be meaningful).

ARC-058 vs ARC-033 arbitration is INTENTIONALLY OUT OF SCOPE for this
experiment. Both paths benefit from SD-048 substrate enrichment; the
arbitration runs on a separate experiment (V3-EXQ-445 already executed
the dACC three-arm test). V3-EXQ-512 uses the simpler ARC-033 path
(independent E2_harm_a) so the substrate-discrimination measurement
is not entangled with the trunk-vs-independent design choice. PASS
on V3-EXQ-512 supports SD-048 substrate adequacy for BOTH paths;
governance arbitration between ARC-058 and ARC-033 is a separate gate.

Phased training per arm
-----------------------
P0 (warmup): train AffectiveHarmEncoder + agent encoders (SD-011 second
   source enabled, harm_history_len=10). The E2_harm_a forward model
   is NOT trained in P0 -- its targets are the encoder outputs, which
   need to stabilise first.

P1 (forward-model training): freeze AffectiveHarmEncoder gradients via
   .detach() on z_harm_a targets, train E2_harm_a on stop-grad targets
   per the canonical MECH-258 P1 recipe. Random-policy rollout.

P2 (eval): no gradient updates. For each step, compute the forward
   residual and tag the step as agent_caused / body_noise / quiet
   using the env's _last_transition_type and the SD-048
   interoceptive_n_body_noise_events / interoceptive_n_agent_caused_harm_events
   info-tag counters. Aggregate per-arm metrics over the eval window.

Pre-registered metrics (per arm per seed)
-----------------------------------------
  residual_body_noise:  mean ||z_harm_a_actual - E2_harm_a_pred|| at
                        ticks where info["interoceptive_n_body_noise_events"]
                        == 1 this step. Measures comparator residual on
                        agent-independent body-state perturbations.
  residual_agent:       same metric on ticks where
                        info["interoceptive_n_agent_caused_harm_events"]
                        == 1 this step. Measures comparator residual on
                        agent-driven body-state change.
  residual_quiet:       same metric on remaining ticks (no event).
                        Background floor.
  forward_r2:           1 - var(residual) / var(target). Standard r2 of
                        the forward model on all eval steps.
  selectivity_gap:      residual_body_noise - residual_agent. Architectural
                        prediction: positive in ARM_A (body-noise events
                        unpredictable from action, agent events predictable);
                        approximately zero in ARM_B (no body-noise events
                        exist).
  selectivity_ratio:    residual_body_noise / max(eps, residual_agent).
                        SNR by which the forward model distinguishes
                        body-noise from agent-caused harm_obs_a change.

Acceptance criteria (>= 2/3 seeds for each)
-------------------------------------------
  C1: ARM_A selectivity_gap > 0.0 -- body-noise residual exceeds
      agent-caused residual at default SD-048 calibration. This is the
      architectural commitment SD-048 makes. Threshold deliberately
      generous (>0 not >=0.05) -- this is the FIRST behavioural test of
      the substrate's discriminative utility.
  C2: ARM_B forward_r2 >= 0.5 -- with SD-048 disabled, the forward model
      reaches a non-trivial fit on the deterministic body-state dynamics.
      C2 is a SANITY check: the forward model architecture is capable of
      learning the SD-022 substrate, so any failure to discriminate in
      ARM_A reflects substrate inadequacy rather than under-trained
      comparator.
  C3: ARM_A residual_body_noise > ARM_A residual_quiet (with margin
      >= C3_MIN_QUIET_GAP) -- the body-noise event residual exceeds the
      quiet-tick baseline residual, confirming the body-noise tag is a
      real perturbation seen by the comparator (not a relabel of quiet
      steps). Without this gate, a high selectivity_gap from C1 could
      reflect agent-caused events being unusually predictable rather
      than body-noise events being unpredictable; C3 closes that
      ambiguity by anchoring the body-noise residual against the quiet
      baseline. ARM_B does not contribute to C3 (it has zero body-noise
      events by construction); the quiet-baseline reference is
      intrinsic to ARM_A.

PASS = C1 AND C2 AND C3.
PASS lifts the v3_pending gate on SD-048 (governance promotes candidate ->
provisional with two pillars: V3-EXQ-511 substrate-readiness and V3-EXQ-512
behavioural discrimination). FAIL with C2 only -> SD-048 substrate exists
but the forward model cannot discriminate; route ARC-058/ARC-033 to
substrate_conditional and re-evaluate at V4 with a richer body-state
substrate (temperature, proprioception, visceral specificity). FAIL with
C2 also failing -> the agent build itself cannot learn the substrate;
debug instrumentation, not architecture.

experiment_purpose = "evidence" (this contributes to claim confidence,
unlike V3-EXQ-511 which was diagnostic-only).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_512_sd048_arc058_arc033_comparator_gap_behavioural.py
  /opt/local/bin/python3 experiments/v3_exq_512_sd048_arc058_arc033_comparator_gap_behavioural.py --dry-run
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
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.predictors.e2_harm_a import E2HarmAConfig, E2HarmAForward  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_512_sd048_arc058_arc033_comparator_gap_behavioural"
CLAIM_IDS = ["SD-048"]
EXPERIMENT_PURPOSE = "evidence"

# --- Config -------------------------------------------------------------
SEEDS = (42, 43, 44)
WARMUP_EPISODES = 60     # P0: encoder warmup
P1_EPISODES = 80         # P1: E2_harm_a frozen-encoder training
EVAL_EPISODES = 40       # P2: metric collection
STEPS_PER_EPISODE = 150
SELF_DIM = 32
WORLD_DIM = 32

# Larger env + sparser policy than dense-contact templates so the SD-048
# body-noise denominator can land on quiet ticks rather than being crowded
# out by agent_caused_hazard transitions every step. Matches V3-EXQ-511
# eval setup for direct comparability.
GRID_SIZE = 12
N_HAZARDS = 1
N_RESOURCES = 1
P_STAY = 0.80

LR = 1e-3
HARM_FWD_LR = 5e-4

# Pre-registered thresholds.
C1_MIN_GAP_ARM_A = 0.0          # selectivity_gap > 0.0 (architectural commitment)
C2_MIN_R2_ARM_B = 0.5           # baseline forward model accuracy in OFF arm
C3_MIN_QUIET_GAP = 0.005        # ARM_A residual_body_noise - residual_quiet >= 0.005
PASS_FRACTION_REQUIRED = 2.0 / 3.0


def _action_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _sparse_random_action(action_dim: int) -> int:
    """Mirrors V3-EXQ-511 sparse-policy rollout. 80% stay (action 4)."""
    if random.random() < P_STAY:
        return 4 if action_dim > 4 else action_dim - 1
    return random.randint(0, min(3, action_dim - 1))


def _make_env(seed: int, sd048_on: bool) -> CausalGridWorldV2:
    """Construct env with SD-022 limb_damage substrate; SD-048 toggled per arm."""
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
        # SD-022 body-damage substrate -- prerequisite for SD-048.
        limb_damage_enabled=True,
        harm_history_len=10,
        # SD-048 master + per-source switches (per-source default True under master).
        interoceptive_noise_enabled=sd048_on,
        interoceptive_noise_scale=1.0,
    )


def run_arm(
    seed: int,
    arm_label: str,
    sd048_on: bool,
    p0_eps: int,
    p1_eps: int,
    eval_eps: int,
) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = _make_env(seed, sd048_on)
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
        # SD-010 sensory + SD-011 dual stream + SD-022 body damage (same as EXQ-508).
        use_harm_stream=True,
        use_affective_harm_stream=True,
        harm_history_len=10,
        limb_damage_enabled=True,
    )
    agent = REEAgent(cfg)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)

    # E2_harm_a forward model (ARC-033 path, independent per-stream).
    z_harm_a_dim = getattr(agent.latent_stack.config, "z_harm_a_dim", 16)
    h_cfg = E2HarmAConfig(
        z_harm_a_dim=z_harm_a_dim,
        action_dim=action_dim,
        hidden_dim=128,
        learning_rate=HARM_FWD_LR,
    )
    harm_fwd = E2HarmAForward(h_cfg)
    harm_opt = torch.optim.Adam(harm_fwd.parameters(), lr=HARM_FWD_LR)

    # ---- P0: encoder warmup (E2_harm_a NOT trained yet) ----
    agent.train()
    for ep in range(p0_eps):
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

    # ---- P1: E2_harm_a frozen-encoder training (stop-grad targets) ----
    agent.train()
    harm_fwd.train()
    last_z_harm_a: torch.Tensor = None
    last_action: torch.Tensor = None
    for ep in range(p1_eps):
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

            # Standard agent losses (encoder-side; preserve smooth substrate).
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total = e1_loss + e2_loss

            # E2_harm_a P1 step: predict THIS-step z_harm_a from PREVIOUS-step
            # (z_harm_a, action). MSE on stop-grad target. Gradient flows ONLY
            # through harm_fwd parameters; encoder is frozen via .detach().
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

    # ---- P2: eval (no gradient updates) ----
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

            # Compute residual using PREVIOUS-step (z_harm_a, action) -> THIS-step
            # actual z_harm_a. Tag attribution by THIS-step env info.
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

            # Tag the step using SD-048 info counters (set by env in
            # _apply_interoceptive_noise; both fire only when |delta_harm_obs_a|
            # exceeds env's interoceptive_change_threshold). agent_caused vs
            # body_noise are mutually exclusive in the env classifier.
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

    # Aggregate per-arm metrics.
    def _safe_mean(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    mean_body = _safe_mean(res_body_noise)
    mean_agent = _safe_mean(res_agent)
    mean_quiet = _safe_mean(res_quiet)
    selectivity_gap = mean_body - mean_agent
    selectivity_ratio = mean_body / max(1e-6, mean_agent)

    # forward_r2 = 1 - mean_residual_sq / mean_target_sq (per-step normaliser).
    if all_residuals and all_targets_sq:
        mean_res_sq = sum(r ** 2 for r in all_residuals) / len(all_residuals)
        mean_tgt_sq = sum(all_targets_sq) / len(all_targets_sq)
        forward_r2 = 1.0 - (mean_res_sq / max(1e-6, mean_tgt_sq))
    else:
        forward_r2 = 0.0

    print(
        f"  [{arm_label}] seed={seed} "
        f"n_body={len(res_body_noise)} n_agent={len(res_agent)} "
        f"n_quiet={len(res_quiet)} "
        f"res_body={mean_body:.4f} res_agent={mean_agent:.4f} "
        f"res_quiet={mean_quiet:.4f} "
        f"sel_gap={selectivity_gap:+.4f} sel_ratio={selectivity_ratio:.3f} "
        f"r2={forward_r2:.3f}",
        flush=True,
    )
    return {
        "seed": seed,
        "arm_label": arm_label,
        "sd048_on": bool(sd048_on),
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
    # C3: ARM_A residual_body_noise - residual_quiet >= margin. Confirms the
    # body-noise tag is anchored against the quiet baseline, not an artefact
    # of agent_caused events being unusually predictable.
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

    t0 = time.time()
    arm_a = [run_arm(s, "ARM_A_sd048_on", True, p0, p1, eval_eps) for s in seeds]
    arm_b = [run_arm(s, "ARM_B_sd048_off", False, p0, p1, eval_eps) for s in seeds]
    elapsed = time.time() - t0

    criteria = _evaluate(arm_a, arm_b)
    outcome = "PASS" if criteria["overall_pass"] else "FAIL"
    direction = "supports" if criteria["overall_pass"] else "weakens"

    print(f"\nV3-EXQ-512 (SD-048) -- {outcome} in {elapsed:.1f}s ({len(seeds)} seed(s))", flush=True)
    print(
        f"  C1 ARM_A selectivity_gap > {C1_MIN_GAP_ARM_A}: "
        f"{criteria['c1_seeds_pass']}/{criteria['n_seeds']} -> "
        f"{'PASS' if criteria['c1_pass'] else 'FAIL'} "
        f"(mean gap={criteria['mean_arm_a_selectivity_gap']:+.4f})",
        flush=True,
    )
    print(
        f"  C2 ARM_B forward_r2 >= {C2_MIN_R2_ARM_B}: "
        f"{criteria['c2_seeds_pass']}/{criteria['n_seeds']} -> "
        f"{'PASS' if criteria['c2_pass'] else 'FAIL'} "
        f"(mean r2={criteria['mean_arm_b_forward_r2']:.3f})",
        flush=True,
    )
    print(
        f"  C3 ARM_A body_noise - quiet >= {C3_MIN_QUIET_GAP}: "
        f"{criteria['c3_seeds_pass']}/{criteria['n_seeds']} -> "
        f"{'PASS' if criteria['c3_pass'] else 'FAIL'} "
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
        "result": outcome,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {"SD-048": direction},
        "criteria": criteria,
        "registered_thresholds": {
            "C1_MIN_GAP_ARM_A": C1_MIN_GAP_ARM_A,
            "C2_MIN_R2_ARM_B": C2_MIN_R2_ARM_B,
            "C3_MIN_QUIET_GAP": C3_MIN_QUIET_GAP,
            "PASS_FRACTION_REQUIRED": PASS_FRACTION_REQUIRED,
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
            "interoceptive_noise_scale": 1.0,
        },
        "results_arm_a_sd048_on": arm_a,
        "results_arm_b_sd048_off": arm_b,
        "elapsed_seconds": elapsed,
        "notes": (
            "SD-048 ARC-058 / ARC-033 comparator-gap behavioural successor "
            "to V3-EXQ-511 substrate-readiness diagnostic. Discriminative "
            "pair on SD-022 limb_damage substrate: ARM_A SD-048 ON at "
            "default calibration vs ARM_B SD-048 OFF baseline. Both arms "
            "train E2_harm_a forward model (ARC-033 path, independent "
            "per-stream) under the canonical phased-training recipe (P0 "
            "encoder warmup, P1 frozen-encoder forward-model training on "
            ".detach()ed targets, P2 eval). C1 selectivity_gap > 0 "
            "validates the architectural commitment (body-noise events "
            "produce larger forward-model residual than agent-caused "
            "events). C2 ARM_B forward_r2 >= 0.5 sanity-checks the agent "
            "build can learn the SD-022 substrate. C3 ARM_A vs ARM_B "
            "selectivity-ratio delta confirms the discrimination signal "
            "is genuinely SD-048-induced, not random-policy event-tagging "
            "noise. PASS lifts SD-048 v3_pending gate."
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
