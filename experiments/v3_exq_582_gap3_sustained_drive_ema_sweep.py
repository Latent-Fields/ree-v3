#!/opt/local/bin/python3
"""V3-EXQ-582: goal_pipeline:GAP-3 Phase 3 Option 1 -- sustained-drive EMA
discriminative alpha sweep.

experiment_purpose: diagnostic
claim_ids:           []   (substrate-readiness validation; NOT governance
                           evidence. MECH-306 sustained_drive_trace
                           registration is the separate governance follow-on,
                           gated on THIS result.)

=== WHAT THIS TESTS ===

The SD-012 sustained-drive amendment (goal_pipeline:GAP-3, Option 1) landed
this session: GoalConfig.drive_ema_alpha (default 1.0 = OFF, bit-identical).
GoalState.update() now EMA-smooths drive_level into self._drive_trace before
the SD-012 multiplier:

    _drive_trace = (1 - drive_ema_alpha) * _drive_trace + drive_ema_alpha * drive
    effective_benefit = benefit * z_goal_seeding_gain
                        * (1 + drive_weight * _drive_trace)

EXQ-536a documented the failure this fixes (same regime as this script):
instantaneous drive_level collapses to ~0.005 the step a resource is consumed
(energy resets toward 1.0), so the multiplier ~1.0 and z_goal never seeds at
the exact contact events where seeding must fire (H_b_threshold_never_crossed,
mean drive on contact 0.005).

Swept variable: drive_ema_alpha in {0.01, 0.02, 0.2, 1.0} x 3 seeds.
  alpha=1.0  : OFF parity arm. Must reproduce the EXQ-536a collapse
               (drive trace at contact ~ instantaneous ~ low).
  alpha=0.02 : lit-anchored first-PASS arm (~35-step half-life; inside the
               30-60 step post-consummatory wanting window in
               evidence/literature/wanting_liking_sleep_consolidation_synthesis.md).
  alpha=0.2  : fast-end falsifier bracket (~3-step half-life).
  alpha=0.01 : slow-end bracket (~69-step half-life).

=== REGIME (anchored to EXQ-536a v3_exq_536a_goal_seeding_instrumentation) ===

Env + agent config identical to EXQ-536a (CausalGridWorldV2 12x12, 4 hazards,
4 resources, SD-018 resource-proximity head, drive_weight=2.0, z_goal_enabled,
benefit_eval, MECH-295 liking bridge), plus drive_ema_alpha=<arm>. Training is
light (random-action warmup + the same loss stack as 536a). During eval, every
step calls agent.update_z_goal(benefit, drive) UNGATED (the substrate-faithful
time-EMA semantics, matching the EXQ-514 family; this DIVERGES from 536a's
benefit>0.01 gate, which was a 536a-specific instrumentation choice -- the
drive trace must integrate every tick to be a continuous-time EMA). The
measured trace is read from agent.goal_state._drive_trace AFTER the update
(the value the substrate actually used), NOT recomputed with instantaneous
drive.

GoalState.reset() (called per eval episode, as in 536a) now also zeroes
_drive_trace (the Q2 cold-start is per-episode). The trace is zero-initialised,
so alpha < 1.0 carries a ~1/alpha-step cold-start transient that underestimates
drive early in an episode. To stop that accepted confound from masquerading as
an EMA failure, every metric is reported over TWO windows:
  - "all"          : all eval steps.
  - "post_warmup"  : eval steps with step index >= POST_WARMUP_CUT (=100, a
                     fixed, alpha-independent cut so the comparison across arms
                     is fair; ~3 half-lives for the headline alpha=0.02 arm).
Headline acceptance uses the post_warmup window. Caveat: at alpha=0.01 a
200-step episode gives only ~1.5 half-lives of warmup before the cut, so its
post_warmup trace is a lower bound -- acceptable because 0.01 is the slow-end
falsifier bracket, not the first-PASS arm.

=== ACCEPTANCE (pre-registered) ===

PASS iff ALL of:
  A1  mean_drive_trace_on_contact (post_warmup, alpha=0.02, pooled seeds) > 0.10
        (>= ~10x lift over EXQ-536a's 0.005)
  A2  >= 2 of 3 seeds have max_effective_benefit (post_warmup, alpha=0.02) > 0.10
        (benefit_threshold; the inner seeding gate becomes reachable)
  A3  z_goal_active_fraction (post_warmup, alpha=0.02, pooled seeds) > 0.20
  A4  FALSIFIER / monotone curve: mean_drive_trace_on_contact (post_warmup,
        pooled) is monotone non-increasing across alpha=[0.01,0.02,0.2,1.0]
        (eps 1e-3) AND the alpha=1.0 OFF arm does NOT clear A1 (< 0.10) --
        i.e. the EMA is demonstrably what made the difference.
Else FAIL.

This is diagnostic (claim_ids=[]), so it is excluded from governance
confidence/conflict scoring regardless of outcome. The PASS/FAIL is the
GAP-3-closure signal only.

=== DIAGNOSTIC INTERPRETATION GRID ===

| Outcome                                              | Reading                                          | Next action |
|------------------------------------------------------|--------------------------------------------------|-------------|
| A1-A4 all hold                                       | Option 1 sustained-drive EMA resolves the        | GAP-3 done. Route MECH-306 sustained_drive_trace registration to governance (gated on this). Mark goal_pipeline:GAP-3 node done, owner_exq=V3-EXQ-582. Unblocks GAP-4 prerequisite. |
|                                                      | EXQ-536a contact-collapse; alpha=0.02 lit value  |             |
|                                                      | is sufficient.                                   |             |
| A1-A3 hold, A4 non-monotone                          | alpha=0.02 seeds but the curve is not clean      | Investigate cold-start / measurement window before closing GAP-3; re-run with longer episodes or steady-state-only contact measurement. Do NOT register MECH-306 yet. |
|                                                      | (cold-start or measurement artifact).            |             |
| A1 holds, A2/A3 fail                                 | Drive trace lifts but z_goal still does not seed | Seeding bottleneck is downstream of the multiplier (alpha_goal pull / encoder / is_active threshold). Hand to a goal-seeding diagnostic, not Option 2. |
|                                                      | -> the multiplier was not the only blocker.      |             |
| No arm clears A1 (incl. 0.01)                        | Sustained EMA insufficient at any timescale      | Escalate to Option 2 (insatiability floor, drive_floor) per sustained_drive_anticipatory_wanting.md; queue the Option-2 falsifier. |
|                                                      | in this regime.                                  |             |
| alpha=1.0 arm does NOT reproduce 536a collapse       | Regime / instrumentation drift since 536a        | STOP -- diagnose regime drift before trusting any arm. The OFF arm is the anchor; if it does not match 536a the comparison is invalid. |
|                                                      | (the OFF anchor is broken).                      |             |

architecture_epoch: "ree_hybrid_guardrails_v1"
"""

import json
import sys
import os
import time
import math
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

EXPERIMENT_TYPE = "v3_exq_582_gap3_sustained_drive_ema_sweep"
QUEUE_ID = "V3-EXQ-582"
CLAIM_IDS = []  # diagnostic: substrate-readiness, no governance claim tags

ALPHAS = [0.01, 0.02, 0.2, 1.0]   # ascending; 1.0 = OFF parity arm
SEEDS = [0, 1, 2]
FIRST_PASS_ALPHA = 0.02
POST_WARMUP_CUT = 100             # fixed, alpha-independent eval-step cut

N_TRAIN_EPS = 30
N_EVAL_EPS = 20
N_STEPS = 200
GRID_SIZE = 12

# Pre-registered acceptance thresholds (constants, not derived post-hoc).
A1_DRIVE_TRACE_MIN = 0.10
A2_EFF_BENEFIT_MIN = 0.10
A2_MIN_SEEDS = 2
A3_ZGOAL_ACTIVE_FRAC_MIN = 0.20
A4_MONOTONE_EPS = 1e-3
A4_OFF_ARM_MAX = 0.10             # alpha=1.0 must stay below A1

DRY_RUN = "--dry-run" in sys.argv
if DRY_RUN:
    N_TRAIN_EPS = 4
    N_EVAL_EPS = 3
    N_STEPS = 40
    POST_WARMUP_CUT = 10


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


def _make_agent(env, drive_ema_alpha):
    cfg = REEConfig.from_dims(
        world_obs_dim=env.world_obs_dim,
        body_obs_dim=env.body_obs_dim,
        action_dim=env.action_dim,
        use_resource_proximity_head=True,   # SD-018
        drive_weight=2.0,
        drive_ema_alpha=drive_ema_alpha,    # SD-012 GAP-3 swept variable
        z_goal_enabled=True,
        benefit_eval_enabled=True,
        benefit_weight=2.0,
        use_mech295_liking_bridge=True,
        mech295_drive_to_liking_gain=1.0,
        mech295_liking_to_approach_cue_gain=0.5,
    )
    cfg.e3.goal_weight = 1.0  # EXQ-536 bug fix (carried in the anchor regime)
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


def _run_one(alpha, seed):
    """One seed x alpha run. Returns a per-run metrics dict."""
    print(f"Seed {seed} Condition alpha={alpha}", flush=True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    env = _make_env(seed)
    agent = _make_agent(env, alpha)
    optimizer = optim.Adam(agent.parameters(), lr=3e-4)
    benefit_eval_optimizer = optim.Adam(
        list(agent.e3.benefit_eval_head.parameters()), lr=1e-4
    )

    benefit_threshold = float(agent.config.goal.benefit_threshold)
    drive_weight = float(agent.config.goal.drive_weight)
    seeding_gain = float(agent.config.goal.z_goal_seeding_gain)

    # ----------------- training (random action, 536a loss stack) ------ #
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
            benefit, drive, _ = _benefit_drive_energy(obs_dict)

            pred_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = pred_loss + e2_loss

            resource_field = obs_dict.get("resource_field_view", None)
            prox_target = (
                float(resource_field.max().item())
                if resource_field is not None else 0.0
            )
            prox_loss = agent.compute_resource_proximity_loss(
                prox_target, latent
            )
            total_loss = total_loss + prox_loss

            with torch.no_grad():
                z_world_det = latent.z_world.detach()
            benefit_pred_train = agent.e3.benefit_eval_head(z_world_det)
            prox_t = torch.tensor([[prox_target]], dtype=torch.float32)
            b_loss = F.mse_loss(benefit_pred_train, prox_t)

            agent.e3.record_benefit_sample(1)
            # Substrate-faithful: integrate the drive EMA every tick
            # (ungated; diverges from 536a's benefit>0.01 gate by design).
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
                f"  [train] alpha={alpha} seed={seed} "
                f"ep {ep + 1}/{N_TRAIN_EPS}",
                flush=True,
            )

    # ----------------- eval (instrumented) ---------------------------- #
    agent.eval()
    world_dim = agent.config.latent.world_dim

    trace_on_contact_all = []
    trace_on_contact_pw = []
    eff_benefit_all = []
    eff_benefit_pw = []
    n_active_all = 0
    n_active_pw = 0
    n_steps_all = 0
    n_steps_pw = 0
    n_contacts_all = 0
    n_contacts_pw = 0
    max_eff_benefit_pw_per_ep = []

    for ep in range(N_EVAL_EPS):
        _, obs_dict = env.reset()
        agent.reset()
        if agent.goal_state is not None:
            agent.goal_state.reset()
        ep_max_eff_pw = 0.0

        for step_idx in range(N_STEPS):
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

            # Ungated, every tick -> drives the EMA inside GoalState.update.
            agent.update_z_goal(benefit, drive)

            # Read the trace the substrate actually used (post-update).
            trace = float(agent.goal_state._drive_trace)
            eff_benefit = (
                benefit * seeding_gain * (1.0 + drive_weight * trace)
            )
            is_active = bool(agent.goal_state.is_active())
            post_warmup = step_idx >= POST_WARMUP_CUT

            n_steps_all += 1
            eff_benefit_all.append(eff_benefit)
            if is_active:
                n_active_all += 1
            if is_contact:
                n_contacts_all += 1
                trace_on_contact_all.append(trace)
            if post_warmup:
                n_steps_pw += 1
                eff_benefit_pw.append(eff_benefit)
                if eff_benefit > ep_max_eff_pw:
                    ep_max_eff_pw = eff_benefit
                if is_active:
                    n_active_pw += 1
                if is_contact:
                    n_contacts_pw += 1
                    trace_on_contact_pw.append(trace)

            if done:
                break

        max_eff_benefit_pw_per_ep.append(ep_max_eff_pw)

    run_max_eff_pw = (
        float(np.max(max_eff_benefit_pw_per_ep))
        if max_eff_benefit_pw_per_ep else 0.0
    )
    metrics = {
        "alpha": alpha,
        "seed": seed,
        "benefit_threshold": benefit_threshold,
        "drive_weight": drive_weight,
        "seeding_gain": seeding_gain,
        "n_steps_all": n_steps_all,
        "n_steps_post_warmup": n_steps_pw,
        "n_contacts_all": n_contacts_all,
        "n_contacts_post_warmup": n_contacts_pw,
        "mean_drive_trace_on_contact_all": (
            float(np.mean(trace_on_contact_all))
            if trace_on_contact_all else None
        ),
        "mean_drive_trace_on_contact_post_warmup": (
            float(np.mean(trace_on_contact_pw))
            if trace_on_contact_pw else None
        ),
        "stat_trace_on_contact_post_warmup": _stat(trace_on_contact_pw),
        "max_effective_benefit_all": (
            float(np.max(eff_benefit_all)) if eff_benefit_all else 0.0
        ),
        "max_effective_benefit_post_warmup": run_max_eff_pw,
        "z_goal_active_fraction_all": n_active_all / max(1, n_steps_all),
        "z_goal_active_fraction_post_warmup": (
            n_active_pw / max(1, n_steps_pw)
        ),
    }

    # Per-run completion verdict (run-counting marker for the runner ETA;
    # the scientific PASS/FAIL is the cross-arm acceptance computed in main()).
    run_ok = (
        n_contacts_pw > 0
        and metrics["mean_drive_trace_on_contact_post_warmup"] is not None
    )
    print(f"verdict: {'PASS' if run_ok else 'FAIL'}", flush=True)
    return metrics


def main():
    start_time = time.time()
    print("V3-EXQ-582 GAP-3 sustained-drive EMA sweep", flush=True)
    print(
        f"DRY_RUN={DRY_RUN} ALPHAS={ALPHAS} SEEDS={SEEDS} "
        f"N_TRAIN={N_TRAIN_EPS} N_EVAL={N_EVAL_EPS} N_STEPS={N_STEPS} "
        f"POST_WARMUP_CUT={POST_WARMUP_CUT}",
        flush=True,
    )

    runs = []
    for alpha in ALPHAS:
        for seed in SEEDS:
            runs.append(_run_one(alpha, seed))

    # Aggregate per alpha (pool seeds).
    by_alpha = {}
    for alpha in ALPHAS:
        arm = [r for r in runs if r["alpha"] == alpha]
        traces_pw = [
            r["mean_drive_trace_on_contact_post_warmup"]
            for r in arm
            if r["mean_drive_trace_on_contact_post_warmup"] is not None
        ]
        active_fracs_pw = [
            r["z_goal_active_fraction_post_warmup"] for r in arm
        ]
        seeds_clearing_eff = sum(
            1 for r in arm
            if r["max_effective_benefit_post_warmup"] > A2_EFF_BENEFIT_MIN
        )
        by_alpha[str(alpha)] = {
            "alpha": alpha,
            "n_seeds": len(arm),
            "mean_drive_trace_on_contact_post_warmup": (
                float(np.mean(traces_pw)) if traces_pw else None
            ),
            "mean_z_goal_active_fraction_post_warmup": (
                float(np.mean(active_fracs_pw)) if active_fracs_pw else 0.0
            ),
            "seeds_clearing_eff_benefit": seeds_clearing_eff,
            "mean_drive_trace_on_contact_all": (
                float(np.mean([
                    r["mean_drive_trace_on_contact_all"] for r in arm
                    if r["mean_drive_trace_on_contact_all"] is not None
                ])) if any(
                    r["mean_drive_trace_on_contact_all"] is not None
                    for r in arm
                ) else None
            ),
        }

    fp = by_alpha[str(FIRST_PASS_ALPHA)]
    off = by_alpha[str(1.0)]
    fp_trace = fp["mean_drive_trace_on_contact_post_warmup"]
    off_trace = off["mean_drive_trace_on_contact_post_warmup"]

    a1 = fp_trace is not None and fp_trace > A1_DRIVE_TRACE_MIN
    a2 = fp["seeds_clearing_eff_benefit"] >= A2_MIN_SEEDS
    a3 = (
        fp["mean_z_goal_active_fraction_post_warmup"]
        > A3_ZGOAL_ACTIVE_FRAC_MIN
    )

    # A4: monotone non-increasing trace across ascending alpha + OFF arm low.
    curve = [
        by_alpha[str(a)]["mean_drive_trace_on_contact_post_warmup"]
        for a in ALPHAS
    ]
    monotone = all(
        (curve[i] is not None and curve[i + 1] is not None
         and curve[i] >= curve[i + 1] - A4_MONOTONE_EPS)
        for i in range(len(curve) - 1)
    )
    off_arm_low = off_trace is not None and off_trace < A4_OFF_ARM_MAX
    a4 = monotone and off_arm_low

    passed = bool(a1 and a2 and a3 and a4)
    outcome = "PASS" if passed else "FAIL"

    acceptance = {
        "A1_drive_trace_on_contact_gt_0.10": {
            "value": fp_trace, "threshold": A1_DRIVE_TRACE_MIN, "pass": a1,
        },
        "A2_seeds_eff_benefit_gt_threshold": {
            "value": fp["seeds_clearing_eff_benefit"],
            "threshold": A2_MIN_SEEDS, "pass": a2,
        },
        "A3_z_goal_active_fraction_gt_0.20": {
            "value": fp["mean_z_goal_active_fraction_post_warmup"],
            "threshold": A3_ZGOAL_ACTIVE_FRAC_MIN, "pass": a3,
        },
        "A4_monotone_curve_and_off_arm_low": {
            "curve_alpha_asc": curve, "monotone": monotone,
            "off_arm_trace": off_trace, "off_arm_low": off_arm_low,
            "pass": a4,
        },
    }

    elapsed = time.time() - start_time
    print(f"\nElapsed: {elapsed:.1f}s", flush=True)
    print(f"alpha curve (trace@contact, post_warmup, asc alpha): {curve}",
          flush=True)
    print(f"A1 drive_trace@contact(0.02)={fp_trace} pass={a1}", flush=True)
    print(f"A2 seeds_eff>thr(0.02)={fp['seeds_clearing_eff_benefit']} "
          f"pass={a2}", flush=True)
    print(f"A3 z_goal_active_frac(0.02)="
          f"{fp['mean_z_goal_active_fraction_post_warmup']:.4f} pass={a3}",
          flush=True)
    print(f"A4 monotone={monotone} off_arm_trace={off_trace} pass={a4}",
          flush=True)
    print(f"OUTCOME: {outcome}", flush=True)

    manifest = {
        "run_id": (
            f"{EXPERIMENT_TYPE}_"
            f"{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3"
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": "diagnostic",
        "evidence_direction": "non_contributory",
        "evidence_direction_note": (
            "Substrate-readiness validation for goal_pipeline:GAP-3 Phase 3 "
            "Option 1 (SD-012 sustained-drive EMA). claim_ids=[]; excluded "
            "from governance scoring. PASS = the lit-anchored alpha=0.02 arm "
            "lifts drive-at-contact past the EXQ-536a collapse, z_goal seeds, "
            "and the alpha-sweep curve is monotone with the OFF arm at the "
            "536a value. MECH-306 sustained_drive_trace registration is the "
            "governance follow-on gated on this result."
        ),
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "outcome": outcome,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "acceptance": acceptance,
        "by_alpha": by_alpha,
        "per_run": runs,
        "config": {
            "alphas": ALPHAS,
            "seeds": SEEDS,
            "first_pass_alpha": FIRST_PASS_ALPHA,
            "post_warmup_cut": POST_WARMUP_CUT,
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
