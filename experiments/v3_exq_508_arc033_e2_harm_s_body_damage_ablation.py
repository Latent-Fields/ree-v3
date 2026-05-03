#!/opt/local/bin/python3
"""V3-EXQ-508 -- ARC-033 E2_harm_s ablation on the SD-022 body-damage substrate.

Claim: ARC-033 (harm_stream.sensory_discriminative_forward_model)
Status: provisional (exp_conf=0.812 / 6 PASS / 8 FAIL across 34 runs).
        Recently weakened by ARC-058 lit (Wager 2013, Hofbauer 2001) via the
        somatic/affective dissociation argument. Substrate is in place
        (E2HarmSForward landed 2026-04-09; SD-022 limb_damage_enabled
        substrate landed 2026-04-09).

Why this experiment exists
--------------------------
ARC-033 asserts SD-003 counterfactual attribution requires a dedicated
forward model on the sensory-discriminative harm stream (E2_harm_s). The
ablation is a direct test: with the body-damage substrate enabled (SD-022,
the home turf where z_harm_s and z_harm_a are causally independent --
EXQ-241b confirmed r2_s_to_a=0.996 ceiling is structural, not a bug),
compare counterfactual attribution accuracy with vs without E2_harm_s.

If ARC-033 holds, removing E2_harm_s should drop attribution accuracy
significantly. If accuracy is unchanged, the comparator role can be
collapsed into the existing E2 world-stream forward model and ARC-033
narrows to "useful but not necessary."

Two arms (3 seeds each = 6 runs)
--------------------------------
ARM_A (E2_HARM_S_ON):
  use_e2_harm_s_forward=True. SD-013 interventional loss enabled
  (use_interventional=True, fraction=0.3, margin=0.1). After warmup,
  measures the counterfactual attribution gap on harm-event transitions.

ARM_B (E2_HARM_S_OFF, world-only ablation):
  use_e2_harm_s_forward=False. Same warmup + same eval pass uses the
  agent's existing world-stream E2 (via z_world counterfactual) for the
  attribution gap. The architectural prediction is that world-stream-only
  attribution collapses on z_harm_s-driven events because z_world is
  perpendicular to z_harm_s under SD-010.

Both arms run with limb_damage_enabled=True (SD-022 body-damage substrate)
so z_harm_s and z_harm_a are structurally independent. SD-011 second
source (harm_history) enabled. Phased training: warmup runs both encoders,
then E2_harm_s trains on frozen z_harm_s targets per the canonical P1
recipe (caller .detach()s the targets).

Pre-registered metrics (per arm)
--------------------------------
  cf_gap_on_harm_events:  mean ||cf(z, a_actual) - cf(z, a_cf)||
                          across eval-step pairs where harm_exposure
                          jumped this step (delta_harm > 0.05). Counterfactual
                          uses E2_harm_s (ARM_A) or z_world E2 (ARM_B).
  cf_gap_on_quiet_events: same metric on steps where harm_exposure
                          did NOT jump. Baseline / background.
  attribution_snr: cf_gap_on_harm_events / max(eps, cf_gap_on_quiet_events).
                   The SNR by which the comparator distinguishes
                   harm-arrival ticks from quiet ticks.

PASS criteria (>= 2/3 seeds)
----------------------------
  C1: ARM_A attribution_snr >= 1.5
  C2: ARM_B attribution_snr <= 1.2  (world-only ablation does NOT
                                      discriminate harm events)
  C3: ARM_A attribution_snr / ARM_B attribution_snr >= 1.5

PASS = C1 AND C2 AND C3.
PASS supports ARC-033 strong reading (E2_harm_s is necessary for
counterfactual attribution on the harm stream). FAIL with C1 PASSing
but C2 also PASSing -> world-stream attribution is not as collapsed as
SD-010 predicts; there's residual coupling that the ablation can use.
FAIL with C3 PASSing but C1/C2 individually marginal -> the relative
comparison still favours ARC-033 even if absolute SNRs are low.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_508_arc033_e2_harm_s_body_damage_ablation.py
  /opt/local/bin/python3 experiments/v3_exq_508_arc033_e2_harm_s_body_damage_ablation.py --dry-run
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
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.predictors.e2_harm_s import E2HarmSConfig, E2HarmSForward  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_508_arc033_e2_harm_s_body_damage_ablation"
CLAIM_IDS = ["ARC-033"]
EXPERIMENT_PURPOSE = "evidence"

# --- Config -------------------------------------------------------------
SEEDS = (42, 43, 44)
WARMUP_EPISODES = 100
EVAL_EPISODES = 50
STEPS_PER_EPISODE = 200
SELF_DIM = 32
WORLD_DIM = 32
HARM_DIM = 32
NUM_HAZARDS = 2
NUM_RESOURCES = 3
LR = 1e-3
HARM_FWD_LR = 5e-4
DELTA_HARM_THRESHOLD = 0.05  # threshold for "harm-event tick"

# Pre-registered thresholds.
C1_MIN_SNR_ARM_A = 1.5
C2_MAX_SNR_ARM_B = 1.2
C3_MIN_RELATIVE_SNR = 1.5
PASS_FRACTION_REQUIRED = 2.0 / 3.0


def _action_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _measure_cf_gap_arm_a(harm_fwd: E2HarmSForward, z_harm_s: torch.Tensor,
                            a_actual: torch.Tensor, action_dim: int,
                            gen: torch.Generator) -> float:
    """ARM_A counterfactual gap via E2_harm_s."""
    actual_idx = a_actual.argmax(dim=-1)
    cf_idx = (actual_idx + 1 + torch.randint(0, action_dim - 1, actual_idx.shape, generator=gen)) % action_dim
    a_cf = F.one_hot(cf_idx, num_classes=action_dim).float()
    with torch.no_grad():
        z_a = harm_fwd.forward(z_harm_s, a_actual)
        z_c = harm_fwd.counterfactual_forward(z_harm_s, a_cf)
    return float((z_a - z_c).norm(dim=-1).mean().item())


def _measure_cf_gap_arm_b(agent: REEAgent, z_world: torch.Tensor,
                            a_actual: torch.Tensor, action_dim: int,
                            gen: torch.Generator) -> float:
    """ARM_B counterfactual gap via the z_world E2 forward model.

    Note: this is the *ablated* path -- it deliberately uses z_world
    counterfactuals as a stand-in for harm-stream attribution. The
    architectural prediction is that this path collapses on
    harm-stream-driven events because z_world is perpendicular to
    z_harm_s under SD-010.
    """
    actual_idx = a_actual.argmax(dim=-1)
    cf_idx = (actual_idx + 1 + torch.randint(0, action_dim - 1, actual_idx.shape, generator=gen)) % action_dim
    a_cf = F.one_hot(cf_idx, num_classes=action_dim).float()
    if not hasattr(agent.e2, "world_forward"):
        return 0.0
    with torch.no_grad():
        z_a = agent.e2.world_forward(z_world, a_actual)
        z_c = agent.e2.world_forward(z_world, a_cf)
    return float((z_a - z_c).norm(dim=-1).mean().item())


def run_arm(seed: int, arm_label: str, use_e2_harm_s: bool,
            warmup_eps: int, eval_eps: int) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)

    env = CausalGridWorldV2(
        seed=seed, size=10,
        num_hazards=NUM_HAZARDS, num_resources=NUM_RESOURCES,
        hazard_harm=0.5, env_drift_interval=5, env_drift_prob=0.1,
        proximity_harm_scale=0.05, proximity_benefit_scale=0.03,
        hazard_field_decay=0.5, energy_decay=0.005,
        use_proxy_fields=True, resource_respawn_on_consume=True,
        limb_damage_enabled=True,
        harm_history_len=10,
    )
    action_dim = env.action_dim
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9, alpha_self=0.3,
        novelty_bonus_weight=0.0,
        benefit_eval_enabled=False,
        # SD-010 sensory + SD-011 dual stream + SD-022 body damage:
        use_harm_stream=True,
        use_affective_harm_stream=True,
        harm_history_len=10,
        limb_damage_enabled=True,
    )
    agent = REEAgent(cfg)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)

    harm_fwd = None
    harm_opt = None
    if use_e2_harm_s:
        # Standalone E2_harm_s with SD-013 interventional loss.
        # z_harm_dim must match the agent's z_harm latent dim (HarmEncoder).
        z_harm_dim = getattr(agent.latent_stack.config, "z_harm_dim", HARM_DIM)
        h_cfg = E2HarmSConfig(
            use_e2_harm_s_forward=True,
            z_harm_dim=z_harm_dim,
            action_dim=action_dim,
            hidden_dim=128,
            learning_rate=HARM_FWD_LR,
            use_interventional=True,
            interventional_fraction=0.3,
            interventional_margin=0.1,
        )
        harm_fwd = E2HarmSForward(h_cfg)
        harm_opt = torch.optim.Adam(harm_fwd.parameters(), lr=HARM_FWD_LR)

    # ---- Warmup ----
    agent.train()
    if harm_fwd is not None:
        harm_fwd.train()

    # FIFO of (z_harm_s, action) pairs from the previous step so we can
    # train E2_harm_s on stop-grad targets z_harm_s_next.
    last_z_harm: torch.Tensor = None
    last_action: torch.Tensor = None

    for ep in range(warmup_eps):
        _, obs_dict = env.reset()
        agent.reset()
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

            # Standard agent losses (encoder + E1 + E2).
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total = e1_loss + e2_loss

            # E2_harm_s training step (P1: stop-grad targets).
            if harm_fwd is not None and last_z_harm is not None:
                z_pred = harm_fwd.forward(last_z_harm.detach(), last_action.detach())
                target = latent.z_harm.detach() if latent.z_harm is not None else None
                if target is not None:
                    h_loss = harm_fwd.compute_loss(z_pred, target)
                    cf_idx = (last_action.argmax(dim=-1) + 1) % action_dim
                    a_cf = F.one_hot(cf_idx, num_classes=action_dim).float()
                    h_int = harm_fwd.compute_interventional_loss(
                        last_z_harm.detach(), last_action.detach(), a_cf
                    )
                    h_total = h_loss + h_int
                    harm_opt.zero_grad()
                    h_total.backward()
                    harm_opt.step()

            if total.requires_grad:
                optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            action_idx = random.randint(0, action_dim - 1)
            action_oh = _action_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh
            if latent.z_harm is not None:
                last_z_harm = latent.z_harm.detach()
                last_action = action_oh.detach()
            _, _, done, _, obs_dict = env.step(action_oh)
            if done:
                break

    # ---- Eval ----
    agent.eval()
    if harm_fwd is not None:
        harm_fwd.eval()

    cf_gaps_event: List[float] = []
    cf_gaps_quiet: List[float] = []
    last_harm_exposure = 0.0
    for ep in range(eval_eps):
        _, obs_dict = env.reset()
        agent.reset()
        last_harm_exposure = 0.0
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

            action_idx = random.randint(0, action_dim - 1)
            action_oh = _action_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh

            # Compute counterfactual gap BEFORE stepping the env.
            if use_e2_harm_s and harm_fwd is not None and latent.z_harm is not None:
                gap = _measure_cf_gap_arm_a(harm_fwd, latent.z_harm.detach(), action_oh, action_dim, gen)
            else:
                gap = _measure_cf_gap_arm_b(agent, latent.z_world.detach(), action_oh, action_dim, gen)

            _, _, done, info, obs_dict = env.step(action_oh)
            cur_harm_exposure = float(info.get("harm_exposure", 0.0))
            delta_harm = abs(cur_harm_exposure - last_harm_exposure)
            last_harm_exposure = cur_harm_exposure

            if delta_harm >= DELTA_HARM_THRESHOLD:
                cf_gaps_event.append(gap)
            else:
                cf_gaps_quiet.append(gap)

            if done:
                break

    mean_event = float(sum(cf_gaps_event) / max(1, len(cf_gaps_event)))
    mean_quiet = float(sum(cf_gaps_quiet) / max(1, len(cf_gaps_quiet)))
    snr = mean_event / max(1e-6, mean_quiet)

    print(f"  [{arm_label}] seed={seed} "
          f"n_event={len(cf_gaps_event)} n_quiet={len(cf_gaps_quiet)} "
          f"gap_event={mean_event:.4f} gap_quiet={mean_quiet:.4f} snr={snr:.3f}",
          flush=True)
    return {
        "seed": seed, "arm_label": arm_label,
        "use_e2_harm_s": bool(use_e2_harm_s),
        "n_event_steps": len(cf_gaps_event),
        "n_quiet_steps": len(cf_gaps_quiet),
        "cf_gap_on_harm_events": mean_event,
        "cf_gap_on_quiet_events": mean_quiet,
        "attribution_snr": snr,
    }


def _evaluate(arm_a: List[Dict], arm_b: List[Dict]) -> Dict:
    n = len(arm_a)
    required = math.ceil(n * PASS_FRACTION_REQUIRED)
    c1 = sum(1 for r in arm_a if r["attribution_snr"] >= C1_MIN_SNR_ARM_A)
    c2 = sum(1 for r in arm_b if r["attribution_snr"] <= C2_MAX_SNR_ARM_B)
    c3 = 0
    for ra, rb in zip(arm_a, arm_b):
        ratio = ra["attribution_snr"] / max(1e-6, rb["attribution_snr"])
        if ratio >= C3_MIN_RELATIVE_SNR:
            c3 += 1
    return {
        "n_seeds": n, "min_seeds_required": required,
        "c1_seeds_pass": c1, "c2_seeds_pass": c2, "c3_seeds_pass": c3,
        "c1_pass": c1 >= required, "c2_pass": c2 >= required,
        "c3_pass": c3 >= required,
        "overall_pass": (c1 >= required and c2 >= required and c3 >= required),
        "mean_arm_a_snr": float(sum(r["attribution_snr"] for r in arm_a) / n),
        "mean_arm_b_snr": float(sum(r["attribution_snr"] for r in arm_b) / n),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--warmup", type=int, default=WARMUP_EPISODES)
    parser.add_argument("--eval", type=int, default=EVAL_EPISODES)
    args = parser.parse_args()

    if args.dry_run:
        seeds = (args.seeds[0],)
        warmup = 2
        eval_eps = 2
        print("[DRY-RUN] 1 seed, 2 warmup eps, 2 eval eps -- smoke only.", flush=True)
    else:
        seeds = tuple(args.seeds)
        warmup = args.warmup
        eval_eps = args.eval

    t0 = time.time()
    arm_a = [run_arm(s, "ARM_A_e2_harm_s_on", True, warmup, eval_eps) for s in seeds]
    arm_b = [run_arm(s, "ARM_B_e2_harm_s_off", False, warmup, eval_eps) for s in seeds]
    elapsed = time.time() - t0

    criteria = _evaluate(arm_a, arm_b)
    outcome = "PASS" if criteria["overall_pass"] else "FAIL"
    direction = "supports" if criteria["overall_pass"] else "weakens"

    print(f"\nV3-EXQ-508 (ARC-033) -- {outcome} in {elapsed:.1f}s ({len(seeds)} seed(s))", flush=True)
    print(f"  C1 ARM_A snr >= {C1_MIN_SNR_ARM_A}: "
          f"{criteria['c1_seeds_pass']}/{criteria['n_seeds']} -> "
          f"{'PASS' if criteria['c1_pass'] else 'FAIL'}", flush=True)
    print(f"  C2 ARM_B snr <= {C2_MAX_SNR_ARM_B}: "
          f"{criteria['c2_seeds_pass']}/{criteria['n_seeds']} -> "
          f"{'PASS' if criteria['c2_pass'] else 'FAIL'}", flush=True)
    print(f"  C3 ARM_A/ARM_B snr ratio >= {C3_MIN_RELATIVE_SNR}: "
          f"{criteria['c3_seeds_pass']}/{criteria['n_seeds']} -> "
          f"{'PASS' if criteria['c3_pass'] else 'FAIL'}", flush=True)

    if args.dry_run:
        print("[--dry-run] manifest not written.", flush=True)
        return

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest = {
        "schema_version": "v1", "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS, "result": outcome,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {"ARC-033": direction},
        "criteria": criteria,
        "registered_thresholds": {
            "C1_MIN_SNR_ARM_A": C1_MIN_SNR_ARM_A,
            "C2_MAX_SNR_ARM_B": C2_MAX_SNR_ARM_B,
            "C3_MIN_RELATIVE_SNR": C3_MIN_RELATIVE_SNR,
            "PASS_FRACTION_REQUIRED": PASS_FRACTION_REQUIRED,
            "DELTA_HARM_THRESHOLD": DELTA_HARM_THRESHOLD,
        },
        "config": {
            "warmup_episodes": warmup, "eval_episodes": eval_eps,
            "steps_per_episode": STEPS_PER_EPISODE,
            "self_dim": SELF_DIM, "world_dim": WORLD_DIM, "harm_dim": HARM_DIM,
            "num_hazards": NUM_HAZARDS, "num_resources": NUM_RESOURCES,
            "lr": LR, "harm_fwd_lr": HARM_FWD_LR, "seeds": list(seeds),
        },
        "results_arm_a_e2_harm_s_on": arm_a,
        "results_arm_b_e2_harm_s_off": arm_b,
        "elapsed_seconds": elapsed,
        "notes": (
            "ARC-033 ablation on the SD-022 body-damage substrate. ARM_A "
            "trains a dedicated E2_harm_s with SD-013 interventional loss. "
            "ARM_B uses z_world E2 forward as the counterfactual stand-in. "
            "Attribution SNR = cf_gap_event / cf_gap_quiet. PASS supports "
            "ARC-033 strong reading. C1+C2 PASSing but C3 also tight at "
            "the threshold -> the absolute discrimination is real but the "
            "relative gap is small; lit-pull-driven concerns about somatic/"
            "affective dissociation (Wager 2013, Hofbauer 2001 -- ARC-058) "
            "may apply."
        ),
    }
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Result written to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
