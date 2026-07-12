"""
V3-EXQ-582a: GAP-3 Option 2 -- insatiability floor (drive_floor) sweep.

Escalation from V3-EXQ-582 FAIL (Option 1 EMA, all arms): diagnostic grid
row "No arm clears A1" -> escalate to Option 2 per sustained_drive_anticipatory_wanting.md.

=== ROOT CAUSE (from EXQ-582) ===

EXQ-582 showed all alpha arms (incl. 0.01) had drive_trace_at_contact ~0.0002-0.005
because drive_level stays near-zero throughout the episode: the agent is well-fed
and never builds genuine homeostatic deficit. EMA cannot help when the EMA INPUT
is consistently low. The trace was lower for slower alphas (cold-start from 0 in
200-step episodes) -- the inverse of the prediction -- confirming drive was low
all along, not just at the contact step.

Additionally: ALL contacts occurred before step 100 (post_warmup window had 0
contacts across all 12 runs). This experiment drops the warmup cut and measures
all contacts.

=== OPTION 2: INSATIABILITY FLOOR ===

Add drive_floor to GoalConfig (default 0.0 = bit-identical OFF). In update():
    drive_level_floored = max(drive_level, drive_floor)
    trace = (1 - alpha) * trace + alpha * drive_level_floored

With drive_ema_alpha=1.0 (Option 1 OFF), trace = drive_level_floored every step.
This guarantees effective_benefit >= benefit * (1 + drive_weight * drive_floor)
at every contact, regardless of the agent's satiation state.

Predicted minimum effective_benefit at first contact (benefit_exposure from
nociception_ema_alpha=0.1 x resource_benefit=0.3 = 0.03 at first contact):
    floor=0.0:  0.030 * (1 + 2.0*0.00) = 0.030   [536a/582 collapse]
    floor=0.3:  0.030 * (1 + 2.0*0.30) = 0.048   [below threshold]
    floor=0.6:  0.030 * (1 + 2.0*0.60) = 0.066   [below threshold]
    floor=0.9:  0.030 * (1 + 2.0*0.90) = 0.084   [near threshold; 2nd contact 0.16]
    floor=1.2:  0.030 * (1 + 2.0*1.20) = 0.102   [above threshold at first contact]
Seeding fires when effective_benefit > benefit_threshold (0.1). With respawn the
benefit_exposure accumulates across contacts; seeding should fire reliably by
the 2nd contact for floor >= 0.9.

=== SWEEP ===

drive_floor in {0.0, 0.3, 0.6, 0.9, 1.2} x 3 seeds.
  0.0 = OFF arm: must reproduce 582/536a collapse (no seedings fired).
  0.9 = first-PASS arm (lit-predicted, per doc analysis).
  1.2 = aggressive arm: effective_benefit > threshold even at first contact.
drive_ema_alpha = 1.0 (Option 1 OFF) throughout: testing Option 2 in isolation.

No POST_WARMUP_CUT: EXQ-582 had 0 post-warmup contacts; Option 2 applies from
step 0 and has no cold-start transient. All contacts measured.

=== ACCEPTANCE (pre-registered) ===

PASS iff ALL of:
  A1  first-PASS arm (floor=0.9): mean_effective_benefit_on_contact (all contacts,
      pooled seeds) > 0.08  [2.67x lift over OFF arm's 0.03; near benefit_threshold]
  A2  first-PASS arm: >= 2/3 seeds have n_seedings_fired > 0
      (effective_benefit > benefit_threshold at >= 1 contact step)
  A3  first-PASS arm: mean_z_goal_active_fraction (all steps, pooled seeds) > 0.05
  A4  FALSIFIER: OFF arm (floor=0.0) has n_seedings_fired == 0 across all seeds
      (confirms problem persists without floor; regime not spontaneously fixed)
Else FAIL.

=== DIAGNOSTIC INTERPRETATION GRID ===

| Outcome                           | Reading                                     | Next action |
|-----------------------------------|---------------------------------------------|-------------|
| A1-A4 all hold                    | Option 2 drive_floor resolves seeding.      | GAP-3 done. Register MECH-306 via governance. Mark goal_pipeline:GAP-3 done. |
| A1/A2 hold, A3 fails              | Seeding fires but z_goal does not sustain.  | Downstream bottleneck (alpha_goal too slow, or decay_goal too fast). Investigate GoalConfig.alpha_goal. Not an Option 2 failure. |
| A1 holds, A2 fails                | benefit_exposure accrues but contacts too   | Monostrategy: agent not navigating to resources. Check contact rate. Option 2 insufficient alone; pair with MECH-269 V_s fix. |
|                                   | sparse for consistent seeding.              | |
| A4 fails                          | OFF arm spontaneously fires.                | Regime drift from 536a/582; check env config vs anchor. STOP -- diagnose before trusting any arm. |
| No arm clears A1 (incl. 1.2)      | Even drive_floor=1.2 insufficient.          | Escalate to Option 3 (MECH-216 schema-driven anticipatory wanting) per sustained_drive_anticipatory_wanting.md. The benefit_exposure baseline may be too low; also consider lowering benefit_threshold. |

architecture_epoch: "ree_hybrid_guardrails_v1"
supersedes: "V3-EXQ-582"
"""

import json
import sys
import os
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_582a_gap3_drive_floor_sweep"
QUEUE_ID = "V3-EXQ-582a"
CLAIM_IDS = []  # diagnostic: substrate-readiness, no governance claim tags
EXPERIMENT_PURPOSE = "diagnostic"

FLOORS = [0.0, 0.3, 0.6, 0.9, 1.2]  # ascending; 0.0 = OFF arm
SEEDS = [0, 1, 2]
FIRST_PASS_FLOOR = 0.9
OFF_FLOOR = 0.0

N_TRAIN_EPS = 30
N_EVAL_EPS = 20
N_STEPS = 200
GRID_SIZE = 12

# Pre-registered acceptance thresholds.
A1_EFF_BENEFIT_MIN = 0.08  # mean effective_benefit at contact for first-PASS arm
A2_MIN_SEEDS = 2            # seeds with n_seedings_fired > 0
A3_ZGOAL_ACTIVE_FRAC_MIN = 0.05  # lowered from 582's 0.20 (harder regime)

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
        drive_ema_alpha=1.0,                # Option 1 OFF (testing Option 2 only)
        drive_floor=drive_floor,            # Option 2: swept variable
        z_goal_enabled=True,
        benefit_eval_enabled=True,
        benefit_weight=2.0,
        use_mech295_liking_bridge=True,
        mech295_drive_to_liking_gain=1.0,
        mech295_liking_to_approach_cue_gain=0.5,
    )
    cfg.e3.goal_weight = 1.0  # EXQ-536a regime anchor bug fix
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


def _run_one(drive_floor, seed):
    """One seed x floor run. Returns a per-run metrics dict."""
    print(f"Seed {seed} Condition floor={drive_floor}", flush=True)
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

    # ----------------- training (random actions, 536a loss stack) ---- #
    # Matches EXQ-582/EXQ-536a anchor: random-action warmup so world-model and
    # proximity head train on diverse transitions (not policy-biased samples).
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

            # Random action (random-walk warmup, matching 536a regime).
            action_int = random.randint(0, env.action_dim - 1)
            action_oh = torch.zeros(1, env.action_dim)
            action_oh[0, action_int] = 1.0
            agent._last_action = action_oh

            _, _harm, done, _info, obs_dict = env.step(action_oh)
            benefit, drive, _energy = _benefit_drive_energy(obs_dict)

            pred_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = pred_loss + e2_loss

            # SD-018 resource proximity loss (latent from sense(), gradient intact).
            resource_field = obs_dict.get("resource_field_view", None)
            prox_target = (
                float(resource_field.max().item())
                if resource_field is not None else 0.0
            )
            prox_loss = agent.compute_resource_proximity_loss(prox_target, latent)
            total_loss = total_loss + prox_loss

            # Benefit eval head (z_world detached, separate optimizer).
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
                f"  [train] floor={drive_floor} seed={seed} "
                f"ep {ep + 1}/{N_TRAIN_EPS}",
                flush=True,
            )

    # ----------------- eval (instrumented) -------------------------- #
    agent.eval()

    eff_benefit_on_contact = []   # effective_benefit at each contact step
    benefit_exp_on_contact = []   # raw benefit_exposure at each contact step
    trace_on_contact = []         # drive trace at each contact step
    n_seedings_fired = 0          # steps where effective_benefit > benefit_threshold
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

            # Ungated every tick -- drives the floor+EMA inside GoalState.update().
            agent.update_z_goal(benefit, drive)

            # Read the trace the substrate actually used (post-update).
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
            float(np.mean(trace_on_contact))
            if trace_on_contact else None
        ),
        "stat_eff_benefit_on_contact": _stat(eff_benefit_on_contact),
        "z_goal_active_fraction": n_active / max(1, n_steps),
    }

    # Per-run verdict for runner ETA counting.
    run_ok = n_contacts > 0
    print(f"verdict: {'PASS' if run_ok else 'FAIL'}", flush=True)
    return metrics


def main():
    start_time = time.time()
    print("V3-EXQ-582a GAP-3 Option 2 drive_floor sweep", flush=True)
    print(
        f"DRY_RUN={DRY_RUN} FLOORS={FLOORS} SEEDS={SEEDS} "
        f"N_TRAIN={N_TRAIN_EPS} N_EVAL={N_EVAL_EPS} N_STEPS={N_STEPS}",
        flush=True,
    )

    runs = []
    for floor in FLOORS:
        for seed in SEEDS:
            runs.append(_run_one(floor, seed))

    # Aggregate per floor (pool seeds).
    by_floor = {}
    for floor in FLOORS:
        arm = [r for r in runs if r["drive_floor"] == floor]
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
        by_floor[str(floor)] = {
            "drive_floor": floor,
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

    fp = by_floor[str(FIRST_PASS_FLOOR)]
    off = by_floor[str(OFF_FLOOR)]

    fp_eff = fp["mean_effective_benefit_on_contact"]
    a1 = fp_eff is not None and fp_eff > A1_EFF_BENEFIT_MIN
    a2 = fp["seeds_with_seeding_fired"] >= A2_MIN_SEEDS
    a3 = fp["mean_z_goal_active_fraction"] > A3_ZGOAL_ACTIVE_FRAC_MIN
    # A4: OFF arm has zero seedings in all seeds (problem confirmed without floor).
    a4 = off["total_seedings_fired"] == 0

    passed = bool(a1 and a2 and a3 and a4)
    outcome = "PASS" if passed else "FAIL"

    acceptance = {
        "A1_mean_eff_benefit_on_contact_gt_0.08": {
            "value": fp_eff, "threshold": A1_EFF_BENEFIT_MIN, "pass": a1,
            "floor_arm": FIRST_PASS_FLOOR,
        },
        "A2_seeds_with_seeding_fired_ge_2": {
            "value": fp["seeds_with_seeding_fired"],
            "threshold": A2_MIN_SEEDS, "pass": a2,
            "floor_arm": FIRST_PASS_FLOOR,
        },
        "A3_z_goal_active_fraction_gt_0.05": {
            "value": fp["mean_z_goal_active_fraction"],
            "threshold": A3_ZGOAL_ACTIVE_FRAC_MIN, "pass": a3,
            "floor_arm": FIRST_PASS_FLOOR,
        },
        "A4_off_arm_zero_seedings": {
            "value": off["total_seedings_fired"],
            "threshold": 0, "pass": a4,
            "floor_arm": OFF_FLOOR,
            "off_arm_seeds_with_seeding": off["seeds_with_seeding_fired"],
        },
    }

    elapsed = time.time() - start_time
    print(f"\nElapsed: {elapsed:.1f}s", flush=True)
    print(
        f"mean_eff_benefit_on_contact by floor: "
        f"{[(f, by_floor[str(f)]['mean_effective_benefit_on_contact']) for f in FLOORS]}",
        flush=True,
    )
    print(
        f"mean_benefit_exposure_on_contact by floor: "
        f"{[(f, by_floor[str(f)]['mean_benefit_exposure_on_contact']) for f in FLOORS]}",
        flush=True,
    )
    print(f"A1 eff_benefit@contact(floor=0.9)={fp_eff} pass={a1}", flush=True)
    print(f"A2 seeds_with_seeding(floor=0.9)={fp['seeds_with_seeding_fired']} pass={a2}", flush=True)
    print(f"A3 z_goal_active_frac(floor=0.9)={fp['mean_z_goal_active_fraction']:.4f} pass={a3}", flush=True)
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
        "evidence_direction": "non_contributory",
        "evidence_direction_note": (
            "Substrate-readiness validation for goal_pipeline:GAP-3 Option 2 "
            "(SD-012 insatiability floor). claim_ids=[]; excluded from governance "
            "scoring. PASS = drive_floor=0.9 arm lifts effective_benefit at "
            "contact, seeding fires in >=2/3 seeds, z_goal activates, and OFF "
            "arm confirms the problem persists without a floor. MECH-306 "
            "sustained_drive_trace registration is the governance follow-on "
            "gated on this result (PASS on either Option 1 or Option 2).",
        ),
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "supersedes": "V3-EXQ-582",
        "outcome": outcome,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "acceptance": acceptance,
        "by_floor": by_floor,
        "per_run": runs,
        "config": {
            "floors": FLOORS,
            "seeds": SEEDS,
            "first_pass_floor": FIRST_PASS_FLOOR,
            "drive_ema_alpha": 1.0,
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
    manifest_path = write_flat_manifest(
        manifest,
        evidence_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    print(f"Manifest written: {manifest_path}", flush=True)
    return outcome, str(manifest_path)


if __name__ == "__main__":
    _outcome, _manifest_path = main()
    emit_outcome(outcome=_outcome, manifest_path=_manifest_path)
