"""
V3-EXQ-627: MECH-306 sustained_drive_trace -- evidence-purpose validation.

=== PURPOSE ===

MECH-306 (goal_seeding.sustained_drive_trace) is status=candidate_substrate_landed
with NO governance-weighting evidence: its only post-landing run, V3-EXQ-582a
(GAP-3 drive_floor sweep, PASS 2026-05-19), was experiment_purpose=diagnostic with
claim_ids=[] (non_contributory), and it predates MECH-306's registration
(2026-05-20). The mechanism is demonstrated but cannot move governance.

This experiment is the evidence-purpose, MECH-306-tagged confirming run that lets
governance move MECH-306 off candidate_substrate_landed. It re-uses the 582a
design, narrowed to a clean 2-arm ablation that isolates the load-bearing
sustained-drive-trace mechanism.

=== MECHANISM UNDER TEST ===

SD-012 amplifies z_goal seeding by (1 + drive_weight * drive_trace). The
instantaneous drive_level collapses to ~0.005 the step a resource is consumed
(energy resets toward 1.0), cancelling the amplification at exactly the
consummatory contact where seeding must fire (EXQ-536a / EXQ-582). The
sustained_drive_trace amendment keeps the multiplier elevated across the
consummatory pulse via GoalState._drive_trace, the EMA of the (floored)
drive_level (goal.py GoalState.update). V3-EXQ-582 showed the EMA alone
(Option 1) is insufficient when drive is low all episode; V3-EXQ-582a showed the
insatiability floor (Option 2, drive_floor) is the load-bearing knob. Operating
recommendation (goal_pipeline_plan.md GAP-3): drive_floor=0.9 with
drive_ema_alpha=1.0.

=== ARMS (2-arm ablation, sustained trace ON vs OFF) ===

ARM_SUSTAINED: drive_floor=0.9, drive_ema_alpha=1.0
    The 582a-validated sustained-trace config. _drive_trace stays >= 0.9 in
    steady state -> effective_benefit >= benefit_exposure * (1 + 2.0*0.9) = 2.8x
    -> seeding fires at contact.
ARM_OFF: drive_floor=0.0, drive_ema_alpha=1.0
    Instantaneous drive (no sustained trace). Reproduces the 536a/582 collapse:
    drive_trace ~ drive_level ~ 0.005 at contact -> multiplier ~ 1.0 -> seeding
    does not fire. The falsifier that confirms the trace is load-bearing.

=== ACCEPTANCE (pre-registered) ===

PASS (evidence: supports MECH-306) iff ALL of:
  A1  ARM_SUSTAINED: mean_effective_benefit_on_contact (pooled seeds) > 0.08
      (lift over ARM_OFF's ~0.03; near benefit_threshold 0.1)
  A2  ARM_SUSTAINED: >= 2/3 seeds have n_seedings_fired > 0
      (effective_benefit > benefit_threshold at >= 1 contact step)
  A3  ARM_SUSTAINED: mean_z_goal_active_fraction (pooled seeds) > 0.05
  A4  FALSIFIER: ARM_OFF has total n_seedings_fired == 0 across all seeds
      (confirms the collapse persists without the sustained trace)
Else FAIL (evidence: weakens / does_not_support; route to /failure-autopsy).

=== DIAGNOSTIC INTERPRETATION GRID ===

| Outcome                  | Reading                                          | Next action |
|--------------------------|--------------------------------------------------|-------------|
| A1-A4 all hold           | Sustained trace is the load-bearing mechanism.   | MECH-306 supported. Governance: candidate_substrate_landed -> candidate (clear v3_pending pending conflict ratio). |
| A1/A2 hold, A3 fails     | Seeding fires but z_goal does not sustain.       | Downstream (alpha_goal/decay_goal); not a MECH-306 failure. /failure-autopsy. |
| A1 holds, A2 fails       | benefit accrues but contacts too sparse.         | Monostrategy (agent not navigating to resources). Pair with MECH-269 V_s. |
| A4 fails                 | OFF arm spontaneously fires.                      | Regime drift vs 582a anchor. STOP -- diagnose before trusting any arm. |
| A1 fails (sustained arm) | Even drive_floor=0.9 insufficient on this env.    | Substrate regression vs 582a. /failure-autopsy on goal.py GoalState.update. |

architecture_epoch: "ree_hybrid_guardrails_v1"
"""

import json
import sys
import os
import time
import random
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome

EXPERIMENT_TYPE = "v3_exq_627_mech306_sustained_drive_trace_validation"
QUEUE_ID = "V3-EXQ-627"
CLAIM_IDS = ["MECH-306"]
EXPERIMENT_PURPOSE = "evidence"

# 2-arm ablation: (label, drive_floor). drive_ema_alpha=1.0 throughout (Option 1
# OFF) so the sustained trace is driven solely by the insatiability floor, the
# 582a-validated load-bearing knob.
ARMS = [
    ("ARM_SUSTAINED", 0.9),
    ("ARM_OFF", 0.0),
]
SUSTAINED_FLOOR = 0.9
OFF_FLOOR = 0.0
SEEDS = [0, 1, 2]

N_TRAIN_EPS = 30
N_EVAL_EPS = 20
N_STEPS = 200
GRID_SIZE = 12

# Pre-registered acceptance thresholds (mirror V3-EXQ-582a).
A1_EFF_BENEFIT_MIN = 0.08
A2_MIN_SEEDS = 2
A3_ZGOAL_ACTIVE_FRAC_MIN = 0.05

DRY_RUN = "--dry-run" in sys.argv
if DRY_RUN:
    N_TRAIN_EPS = 4
    N_EVAL_EPS = 3
    N_STEPS = 40


# ------------------------------------------------------------------ #

def _obs_tensors(obs_dict):
    body = obs_dict["body_state"].float().unsqueeze(0)
    world = obs_dict["world_state"].float().unsqueeze(0)
    return body, world


def _benefit_drive_energy(obs_dict):
    body_raw = obs_dict["body_state"]
    benefit = float(body_raw[11].item()) if body_raw.numel() > 11 else 0.0
    energy = float(body_raw[3].item()) if body_raw.numel() > 3 else 0.5
    drive = max(0.0, min(1.0, 1.0 - energy))
    return benefit, drive, energy


def _make_env(seed):
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_hazards=4,
        num_resources=4,
        hazard_harm=0.02,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        reef_enabled=False,
    )


def _make_agent(env, drive_floor):
    cfg = REEConfig.from_dims(
        world_obs_dim=env.world_obs_dim,
        body_obs_dim=env.body_obs_dim,
        action_dim=env.action_dim,
        use_resource_proximity_head=True,   # SD-018
        drive_weight=2.0,
        drive_ema_alpha=1.0,                # Option 1 OFF (isolate Option 2 floor)
        drive_floor=drive_floor,            # sustained-trace lever (swept)
        z_goal_enabled=True,
        benefit_eval_enabled=True,
        benefit_weight=2.0,
        use_mech295_liking_bridge=True,
        mech295_drive_to_liking_gain=1.0,
        mech295_liking_to_approach_cue_gain=0.5,
    )
    cfg.e3.goal_weight = 1.0  # EXQ-536a regime anchor
    return REEAgent(cfg)


def _stat(arr):
    if not arr:
        return None
    return {
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def _run_one(arm_label, drive_floor, seed):
    """One seed x arm run. Returns a per-run metrics dict."""
    print(f"Seed {seed} Condition {arm_label}", flush=True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    env = _make_env(seed)
    agent = _make_agent(env, drive_floor)
    optimizer = optim.Adam(agent.parameters(), lr=3e-4)
    benefit_eval_optimizer = optim.Adam(
        list(agent.e3.benefit_eval_head.parameters()), lr=1e-4
    )

    benefit_threshold = float(agent.config.goal.benefit_threshold)
    drive_weight = float(agent.config.goal.drive_weight)
    seeding_gain = float(agent.config.goal.z_goal_seeding_gain)
    world_dim = agent.config.latent.world_dim

    # ----------------- training (random actions, 536a/582a loss stack) ---- #
    agent.train()
    for ep in range(N_TRAIN_EPS):
        _, obs_dict = env.reset()
        agent.reset()
        if agent.goal_state is not None:
            agent.goal_state.reset()

        for _step in range(N_STEPS):
            body, world = _obs_tensors(obs_dict)
            latent = agent.sense(obs_body=body, obs_world=world)
            agent.clock.advance()

            action_int = random.randint(0, env.action_dim - 1)
            action_oh = torch.zeros(1, env.action_dim)
            action_oh[0, action_int] = 1.0
            agent._last_action = action_oh

            _, _harm, done, _info, obs_dict = env.step(action_oh)
            benefit, drive, _energy = _benefit_drive_energy(obs_dict)

            pred_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = pred_loss + e2_loss

            resource_field = obs_dict.get("resource_field_view", None)
            prox_target = (
                float(resource_field.max().item())
                if resource_field is not None else 0.0
            )
            prox_loss = agent.compute_resource_proximity_loss(prox_target, latent)
            total_loss = total_loss + prox_loss

            with torch.no_grad():
                z_world_det = latent.z_world.detach()
            benefit_pred = agent.e3.benefit_eval_head(z_world_det)
            prox_t = torch.tensor([[prox_target]], dtype=torch.float32)
            b_loss = torch.nn.functional.mse_loss(benefit_pred, prox_t)
            agent.e3.record_benefit_sample(1)

            agent.update_z_goal(benefit, drive)

            if b_loss.requires_grad:
                benefit_eval_optimizer.zero_grad()
                b_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(agent.e3.benefit_eval_head.parameters()), 0.5
                )
                benefit_eval_optimizer.step()

            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if done:
                break

        _print_every = max(1, N_TRAIN_EPS // 3)
        if (ep + 1) % _print_every == 0 or (ep + 1) == N_TRAIN_EPS:
            print(
                f"  [train] {arm_label} seed={seed} ep {ep + 1}/{N_TRAIN_EPS}",
                flush=True,
            )

    # ----------------- eval (instrumented) -------------------------- #
    agent.eval()

    eff_benefit_on_contact = []
    benefit_exp_on_contact = []
    trace_on_contact = []
    n_seedings_fired = 0
    n_steps = 0
    n_contacts = 0
    n_active = 0

    for ep in range(N_EVAL_EPS):
        _, obs_dict = env.reset()
        agent.reset()
        if agent.goal_state is not None:
            agent.goal_state.reset()

        for _step in range(N_STEPS):
            body, world = _obs_tensors(obs_dict)
            agent.sense(obs_body=body, obs_world=world)
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(agent._current_latent)
                if ticks["e1_tick"]
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(
                agent._current_latent, e1_prior, ticks
            )
            action_tensor = agent.select_action(candidates, ticks)
            action_int = int(action_tensor.argmax(dim=-1).item())
            action_oh = torch.zeros(1, env.action_dim)
            action_oh[0, action_int] = 1.0

            _, _harm, done, info, obs_dict = env.step(action_oh)
            benefit, drive, _energy = _benefit_drive_energy(obs_dict)
            ttype = info.get("transition_type", "none")
            is_contact = ttype in ("resource", "benefit_approach")

            agent.update_z_goal(benefit, drive)

            trace = float(agent.goal_state._drive_trace)
            eff_benefit = benefit * seeding_gain * (1.0 + drive_weight * trace)
            is_active = bool(agent.goal_state.is_active())

            n_steps += 1
            if is_active:
                n_active += 1
            if is_contact:
                n_contacts += 1
                eff_benefit_on_contact.append(eff_benefit)
                benefit_exp_on_contact.append(benefit)
                trace_on_contact.append(trace)
                if eff_benefit > benefit_threshold:
                    n_seedings_fired += 1

            if done:
                break

    metrics = {
        "arm": arm_label,
        "drive_floor": drive_floor,
        "seed": seed,
        "benefit_threshold": benefit_threshold,
        "drive_weight": drive_weight,
        "seeding_gain": seeding_gain,
        "n_steps": n_steps,
        "n_contacts": n_contacts,
        "n_seedings_fired": n_seedings_fired,
        "mean_effective_benefit_on_contact": (
            float(np.mean(eff_benefit_on_contact))
            if eff_benefit_on_contact else None
        ),
        "max_effective_benefit_on_contact": (
            float(np.max(eff_benefit_on_contact))
            if eff_benefit_on_contact else None
        ),
        "mean_benefit_exposure_on_contact": (
            float(np.mean(benefit_exp_on_contact))
            if benefit_exp_on_contact else None
        ),
        "mean_trace_on_contact": (
            float(np.mean(trace_on_contact)) if trace_on_contact else None
        ),
        "stat_eff_benefit_on_contact": _stat(eff_benefit_on_contact),
        "z_goal_active_fraction": n_active / max(1, n_steps),
    }

    run_ok = n_contacts > 0
    print(f"verdict: {'PASS' if run_ok else 'FAIL'}", flush=True)
    return metrics


def main():
    start_time = time.time()
    print("V3-EXQ-627 MECH-306 sustained_drive_trace validation", flush=True)
    print(
        f"DRY_RUN={DRY_RUN} ARMS={[a for a, _ in ARMS]} SEEDS={SEEDS} "
        f"N_TRAIN={N_TRAIN_EPS} N_EVAL={N_EVAL_EPS} N_STEPS={N_STEPS}",
        flush=True,
    )

    runs = []
    for arm_label, drive_floor in ARMS:
        for seed in SEEDS:
            runs.append(_run_one(arm_label, drive_floor, seed))

    # Aggregate per arm (pool seeds).
    by_arm = {}
    for arm_label, drive_floor in ARMS:
        arm = [r for r in runs if r["arm"] == arm_label]
        eff_vals = [
            r["mean_effective_benefit_on_contact"]
            for r in arm
            if r["mean_effective_benefit_on_contact"] is not None
        ]
        n_seedings_by_seed = [r["n_seedings_fired"] for r in arm]
        seeds_with_seeding = sum(1 for n in n_seedings_by_seed if n > 0)
        total_seedings = sum(n_seedings_by_seed)
        active_fracs = [r["z_goal_active_fraction"] for r in arm]
        contact_counts = [r["n_contacts"] for r in arm]
        benefit_exp_vals = [
            r["mean_benefit_exposure_on_contact"]
            for r in arm
            if r["mean_benefit_exposure_on_contact"] is not None
        ]
        by_arm[arm_label] = {
            "arm": arm_label,
            "drive_floor": drive_floor,
            "n_seeds": len(arm),
            "mean_effective_benefit_on_contact": (
                float(np.mean(eff_vals)) if eff_vals else None
            ),
            "seeds_with_seeding_fired": seeds_with_seeding,
            "total_seedings_fired": total_seedings,
            "mean_z_goal_active_fraction": float(np.mean(active_fracs)),
            "mean_n_contacts": float(np.mean(contact_counts)),
            "mean_benefit_exposure_on_contact": (
                float(np.mean(benefit_exp_vals)) if benefit_exp_vals else None
            ),
        }

    sustained = by_arm["ARM_SUSTAINED"]
    off = by_arm["ARM_OFF"]

    s_eff = sustained["mean_effective_benefit_on_contact"]
    a1 = s_eff is not None and s_eff > A1_EFF_BENEFIT_MIN
    a2 = sustained["seeds_with_seeding_fired"] >= A2_MIN_SEEDS
    a3 = sustained["mean_z_goal_active_fraction"] > A3_ZGOAL_ACTIVE_FRAC_MIN
    a4 = off["total_seedings_fired"] == 0

    passed = bool(a1 and a2 and a3 and a4)
    outcome = "PASS" if passed else "FAIL"
    evidence_direction = "supports" if passed else "does_not_support"

    acceptance = {
        "A1_mean_eff_benefit_on_contact_gt_0.08": {
            "value": s_eff, "threshold": A1_EFF_BENEFIT_MIN, "pass": a1,
            "arm": "ARM_SUSTAINED",
        },
        "A2_seeds_with_seeding_fired_ge_2": {
            "value": sustained["seeds_with_seeding_fired"],
            "threshold": A2_MIN_SEEDS, "pass": a2, "arm": "ARM_SUSTAINED",
        },
        "A3_z_goal_active_fraction_gt_0.05": {
            "value": sustained["mean_z_goal_active_fraction"],
            "threshold": A3_ZGOAL_ACTIVE_FRAC_MIN, "pass": a3,
            "arm": "ARM_SUSTAINED",
        },
        "A4_off_arm_zero_seedings": {
            "value": off["total_seedings_fired"],
            "threshold": 0, "pass": a4, "arm": "ARM_OFF",
            "off_arm_seeds_with_seeding": off["seeds_with_seeding_fired"],
        },
    }

    elapsed = time.time() - start_time
    print(f"\nElapsed: {elapsed:.1f}s", flush=True)
    print(
        f"mean_eff_benefit_on_contact by arm: "
        f"{[(a, by_arm[a]['mean_effective_benefit_on_contact']) for a, _ in ARMS]}",
        flush=True,
    )
    print(f"A1 eff_benefit@contact(SUSTAINED)={s_eff} pass={a1}", flush=True)
    print(f"A2 seeds_with_seeding(SUSTAINED)={sustained['seeds_with_seeding_fired']} pass={a2}", flush=True)
    print(f"A3 z_goal_active_frac(SUSTAINED)={sustained['mean_z_goal_active_fraction']:.4f} pass={a3}", flush=True)
    print(f"A4 off_arm_seedings={off['total_seedings_fired']} pass={a4}", flush=True)
    print(f"OUTCOME: {outcome}", flush=True)

    manifest = {
        "run_id": (
            f"{EXPERIMENT_TYPE}_"
            f"{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3"
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": evidence_direction,
        "evidence_direction_note": (
            "Evidence-purpose validation of MECH-306 (goal_seeding."
            "sustained_drive_trace). 2-arm ablation isolates the load-bearing "
            "drive_floor lever: ARM_SUSTAINED (drive_floor=0.9, drive_ema_alpha"
            "=1.0) vs ARM_OFF (drive_floor=0.0). PASS=supports if the sustained "
            "trace lifts effective_benefit at contact, fires seeding in >=2/3 "
            "seeds, activates z_goal, AND the OFF arm confirms the collapse "
            "persists without the trace. Lineage: V3-EXQ-582a PASS (diagnostic, "
            "claim_ids=[], predates MECH-306 registration 2026-05-20); this run "
            "provides the governance-weighting evidence 582a could not."
        ),
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "outcome": outcome,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "acceptance": acceptance,
        "by_arm": by_arm,
        "per_run": runs,
        "config": {
            "arms": [{"label": a, "drive_floor": f} for a, f in ARMS],
            "seeds": SEEDS,
            "drive_ema_alpha": 1.0,
            "drive_weight": 2.0,
            "n_train_eps": N_TRAIN_EPS,
            "n_eval_eps": N_EVAL_EPS,
            "n_steps": N_STEPS,
            "grid_size": GRID_SIZE,
            "dry_run": DRY_RUN,
        },
        "elapsed_seconds": elapsed,
        "generated_utc": datetime.utcnow().isoformat() + "Z",
    }

    if DRY_RUN:
        print("[DRY RUN] Not writing evidence.", flush=True)
        return outcome, None

    evidence_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments"
        / EXPERIMENT_TYPE
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = evidence_dir / f"{manifest['run_id']}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written: {manifest_path}", flush=True)
    return outcome, str(manifest_path)


if __name__ == "__main__":
    _outcome, _manifest_path = main()
    emit_outcome(outcome=_outcome, manifest_path=_manifest_path)
