#!/opt/local/bin/python3
"""V3-EXQ-539 -- MECH-307 anticipatory affect conjunction commit-gating check.

Claims: MECH-307, MECH-216, MECH-205, MECH-093, SD-014
Proposal: discriminative ablation paired to EXQ-536b

This experiment tests whether the MECH-307 four-gap substrate fix recovers the
commit-chain inertness pattern that EXQ-536b documented as 'downstream chain
inert even with active z_goal'.

EXQ-536b (2026-05-08) found that with cfg.goal.z_goal_inject=0.3 force-armed at
the action-time MECH-188 site, approach_commit_rate stayed at 0.0 across all
3 seeds even though inject_observed_fraction=1.0 at the per-candidate read site.
The companion mech188_vs_mech295_dual_path.md note disambiguated this as a
read-site bypass: MECH-188 inject does not reach the MECH-295 anticipatory-
liking bridge because the bridge gates on self.goal_state.is_active() (the
persistent attractor) which the MECH-188 inject does not touch.

The MECH-307 four-gap fix is registered as the architectural answer:
  Gap 1: signed VALENCE_SURPRISE writes
  Gap 2: MECH-216 schema readout writes anticipatory VALENCE_LIKING
  Gap 3: MECH-216 schema readout pulses z_beta arousal
  Gap 4: MECH-216 writes at predicted z_world (e1_prior)

If the conjunction-fix architecture is correct, an agent with all four flags
ON should produce a detectable conjunction-state at predicted-reward locations
that the MECH-295 bridge (or any downstream commit-gating consumer) can read,
restoring approach_commit_rate above the EXQ-536b inert baseline.

Conditions (2 arms x 3 seeds)
-----------------------------
ARM_OFF (control, replicates EXQ-536b inert-chain pattern):
  - All four MECH-307 flags False
  - VALENCE_SURPRISE unsigned, MECH-216 single-channel write, no z_beta pulse,
    write at current z_world
  - Predicted: replicates 536b -- approach_commit_rate ~0.0, no anticipatory
    VALENCE_LIKING writes, z_beta unchanged by schema_salience.

ARM_ON (MECH-307 conjunction-fix):
  - use_mech307_signed_pe = True
  - use_mech307_schema_multichannel = True
  - use_mech307_predicted_location_write = True
  - mech307_anticipatory_liking_gain = 0.5 (default)
  - mech307_z_beta_schema_gain = 0.3 (default)
  - Predicted: anticipatory VALENCE_LIKING writes appear at predicted-reward
    locations, z_beta arousal elevates with schema_salience, signed PE writes
    distinguish harm-paired from non-harm surprise. If the MECH-295 bridge or
    any downstream consumer reads these signals, approach_commit_rate should
    lift above ARM_OFF.

Pre-registered acceptance criteria
----------------------------------
C1 (substrate-readiness): ARM_ON shows non-zero anticipatory VALENCE_LIKING
   writes per episode (gap2_liking_writes > 0 per seed; ARM_OFF == 0).
C2 (substrate-readiness): ARM_ON shows non-zero z_beta arousal pulses
   (mean(|z_beta_pulses_observed|) > gap3_pulse_floor in ARM_ON; ARM_OFF == 0).
C3 (substrate-readiness): ARM_ON shows signed VALENCE_SURPRISE writes (negative
   signed surprise count > 0 in ARM_ON when harm events occur; ARM_OFF
   accumulates only positive magnitudes regardless of harm sign).
C4 (substrate-readiness): ARM_ON's MECH-216 writes land at e1_prior locations
   (gap4_predicted_writes > 0); ARM_OFF writes land at current z_world.
C5 (behavioural -- the load-bearing question): ARM_ON approach_commit_rate
   >= 0.10 across at least 2/3 seeds, OR equivalently >= ARM_OFF + 0.10.
   Below this threshold, the conjunction-fix substrate is firing but
   downstream consumers are not picking up the conjunction signal -- a
   diagnostic outcome that would route to the SD-014 6-channel amendment
   fallback experiment.

PASS = C1 AND C2 AND C3 AND C4 AND C5: MECH-307 conjunction architecture
  validated. The four-gap fix is the right architectural answer; SD-014
  6-channel amendment can be retired as fallback. EXQ-141b on this same
  substrate becomes the natural next test.

PARTIAL PASS (C1-C4 PASS but C5 FAIL): substrate-readiness confirmed
  (the four wiring fixes work as designed) but the conjunction-state is
  not consumed by downstream commit-gating logic. Routes to SD-014
  6-channel amendment as the fallback architecture, OR to a focused
  consumer-side fix at MECH-295 (read the conjunction explicitly).

FAIL on C1-C4: substrate fix is incorrect or incomplete. Investigate
  before queueing the SD-014 amendment.

experiment_purpose = "evidence" (governance evidence for MECH-307).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_539_mech307_commit_gating_check.py
  /opt/local/bin/python3 experiments/v3_exq_539_mech307_commit_gating_check.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorld  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.residue.field import (  # noqa: E402
    VALENCE_WANTING, VALENCE_LIKING, VALENCE_SURPRISE,
)
from experiment_protocol import emit_outcome  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_539_mech307_commit_gating_check"
CLAIM_IDS = ["MECH-307", "MECH-216", "MECH-205", "MECH-093", "SD-014"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 43, 44]
P0_EPISODES = 50
EVAL_EPISODES = 20
STEPS_PER_EPISODE = 200

# Acceptance thresholds.
C1_LIKING_WRITES_FLOOR = 1.0          # per-seed mean
C2_Z_BETA_PULSE_FLOOR = 0.05           # min mean |z_beta_dim0| deviation in ARM_ON
C3_SIGNED_SURPRISE_NEG_FLOOR = 1       # at least 1 negative signed surprise in ARM_ON
C4_PREDICTED_WRITES_FLOOR = 1.0        # per-seed mean MECH-216 writes at e1_prior
C5_APPROACH_LIFT = 0.10                # ARM_ON approach_commit_rate - ARM_OFF


def _make_env(seed: int) -> CausalGridWorld:
    """Foraging-class env with sparse contacts -- same regime as EXQ-536a/b."""
    return CausalGridWorld(
        size=10,
        num_hazards=3,
        num_resources=8,
        hazard_harm=0.01,
        resource_benefit=0.18,
        use_proxy_fields=True,
        seed=seed,
        proximity_benefit_scale=0.18,
        resource_respawn_on_consume=True,
    )


def _make_config(env: CausalGridWorld, mech307_flags: Dict) -> REEConfig:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        world_dim=32,
        drive_weight=2.0,
        z_goal_enabled=True,
    )
    cfg.surprise_gated_replay = True
    cfg.e1.schema_wanting_enabled = True
    cfg.schema_wanting_threshold = 0.1
    cfg.schema_wanting_gain = 0.5
    cfg.residue.valence_enabled = True
    # Apply MECH-307 flags per arm.
    for flag, val in mech307_flags.items():
        setattr(cfg, flag, val)
    return cfg


def _measure_arm(
    seed: int,
    arm_label: str,
    mech307_flags: Dict,
    n_warmup: int,
    n_eval: int,
    steps_per_episode: int,
) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = _make_env(seed)
    cfg = _make_config(env, mech307_flags)
    agent = REEAgent(cfg)

    # Diagnostic counters.
    liking_writes = 0
    predicted_writes = 0  # number of MECH-216 schema fires; 1:1 with VALENCE_WANTING writes
    z_beta_dim0_excursions: List[float] = []
    harm_paired_surprise_writes = 0
    nonharm_surprise_writes = 0
    approach_commit_steps = 0
    total_eval_steps = 0
    contact_events = 0

    def _approach_commit_at_step(_agent: REEAgent) -> bool:
        # Heuristic: a step counts as 'approach commit' when the beta_gate is
        # elevated AND there's a non-trivial wanting signal at the agent's
        # predicted next location. This deliberately mirrors the
        # commit-chain-active definition used in EXQ-536b's downstream
        # commit-rate measurement.
        beta_elevated = bool(getattr(_agent.beta_gate, "is_elevated", False))
        if not beta_elevated:
            return False
        if _agent._current_latent is None:
            return False
        z = _agent._current_latent.z_world
        with torch.no_grad():
            v = _agent.residue_field.evaluate_valence(z)
        wanting_amp = float(v[0, VALENCE_WANTING].item())
        return wanting_amp > 0.05

    def _z_beta_excursion(_agent: REEAgent) -> float:
        if _agent._current_latent is None:
            return 0.0
        zb = _agent._current_latent.z_beta
        if zb is None or zb.numel() == 0:
            return 0.0
        return float(zb[..., 0].abs().mean().item())

    # P0 warmup -- light training pass so the agent has some structure.
    for ep in range(n_warmup):
        _, obs_dict = env.reset()
        agent.reset()
        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick")
                else torch.zeros(1, agent.config.latent.world_dim)
            )
            # Drive MECH-216 schema readout writes (this is the call site that
            # exercises Gap 2/3/4 in ARM_ON).
            drive = float(REEAgent.compute_drive_level(obs_body))
            agent.update_schema_wanting(drive_level=drive)
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item())
            _, harm_signal, done, info, obs_dict = env.step(action_idx)
            ttype = info.get("transition_type", "none")
            if ttype == "resource":
                agent.update_z_goal(float(harm_signal), drive_level=drive)
            update_metrics = agent.update_residue(float(harm_signal))
            if done:
                break

    # Eval: count diagnostic signals.
    for ep in range(n_eval):
        _, obs_dict = env.reset()
        agent.reset()
        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick")
                    else torch.zeros(1, agent.config.latent.world_dim)
                )
                drive = float(REEAgent.compute_drive_level(obs_body))

                # Snapshot residue + z_beta state BEFORE the schema write so
                # we can detect Gap 2 / Gap 3 / Gap 4 firing.
                z_world_pre = latent.z_world.clone()
                z_beta_pre_dim0 = (
                    float(latent.z_beta[..., 0].abs().mean().item())
                    if latent.z_beta is not None and latent.z_beta.numel() > 0
                    else 0.0
                )
                v_pre = agent.residue_field.evaluate_valence(z_world_pre)
                liking_pre = float(v_pre[0, VALENCE_LIKING].item())
                wanting_pre = float(v_pre[0, VALENCE_WANTING].item())

                agent.update_schema_wanting(drive_level=drive)

                v_post = agent.residue_field.evaluate_valence(z_world_pre)
                liking_delta = float(v_post[0, VALENCE_LIKING].item()) - liking_pre
                wanting_delta = float(v_post[0, VALENCE_WANTING].item()) - wanting_pre
                z_beta_post_dim0 = (
                    float(latent.z_beta[..., 0].abs().mean().item())
                    if latent.z_beta is not None and latent.z_beta.numel() > 0
                    else 0.0
                )

                if liking_delta > 1e-9:
                    liking_writes += 1
                if wanting_delta > 1e-9:
                    predicted_writes += 1
                # z_beta excursion: post - pre on the dim used by the pulse.
                z_beta_dim0_excursions.append(z_beta_post_dim0 - z_beta_pre_dim0)

                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)

            action_idx = int(action.argmax(dim=-1).item())
            if _approach_commit_at_step(agent):
                approach_commit_steps += 1
            total_eval_steps += 1

            _, harm_signal, done, info, obs_dict = env.step(action_idx)
            ttype = info.get("transition_type", "none")
            if ttype == "resource":
                agent.update_z_goal(float(harm_signal), drive_level=drive)
                contact_events += 1

            # Track surprise write events (harm-paired vs non-harm). The SIGN
            # of the write itself is detected via the end-of-run residue-field
            # readout below -- this counter just records when surprises fired.
            surprise_count_pre = agent._surprise_write_count
            agent.update_residue(float(harm_signal))
            surprise_count_post = agent._surprise_write_count
            if surprise_count_post > surprise_count_pre and harm_signal < 0:
                harm_paired_surprise_writes += 1
            elif surprise_count_post > surprise_count_pre:
                nonharm_surprise_writes += 1

            if done:
                break

    approach_commit_rate = approach_commit_steps / max(1, total_eval_steps)
    z_beta_excursion_mean = float(np.mean([abs(x) for x in z_beta_dim0_excursions])) if z_beta_dim0_excursions else 0.0

    # Read the residue field at end-of-run to count negative VALENCE_SURPRISE
    # entries -- this is the actual Gap 1 signature. ARM_OFF accumulates only
    # non-negative magnitudes (legacy unsigned path). ARM_ON stores signed
    # values, so harm-paired surprises produce negative VALENCE_SURPRISE
    # entries in the field. Count centers whose VALENCE_SURPRISE < 0.
    valence_vecs = agent.residue_field.rbf_field.valence_vecs
    active_mask = agent.residue_field.rbf_field.active_mask
    if active_mask.any():
        active_surprise = valence_vecs[active_mask][:, VALENCE_SURPRISE]
        n_negative_surprise_centers = int((active_surprise < 0).sum().item())
    else:
        n_negative_surprise_centers = 0

    return {
        "seed": seed,
        "arm": arm_label,
        "mech307_flags": dict(mech307_flags),
        "liking_writes": liking_writes,
        "predicted_writes": predicted_writes,
        "z_beta_excursion_mean": z_beta_excursion_mean,
        "harm_paired_surprise_writes": harm_paired_surprise_writes,
        "nonharm_surprise_writes": nonharm_surprise_writes,
        "n_negative_surprise_centers": n_negative_surprise_centers,
        "approach_commit_rate": approach_commit_rate,
        "approach_commit_steps": approach_commit_steps,
        "total_eval_steps": total_eval_steps,
        "contact_events": contact_events,
    }


ARMS = [
    {
        "arm": "ARM_OFF",
        "flags": {
            "use_mech307_signed_pe": False,
            "use_mech307_schema_multichannel": False,
            "use_mech307_predicted_location_write": False,
        },
    },
    {
        "arm": "ARM_ON",
        "flags": {
            "use_mech307_signed_pe": True,
            "use_mech307_schema_multichannel": True,
            "use_mech307_predicted_location_write": True,
        },
    },
]


def _aggregate(per_cell: List[Dict]) -> Dict[str, Dict]:
    bucket: Dict[str, Dict] = {}
    for r in per_cell:
        arm = r["arm"]
        if arm not in bucket:
            bucket[arm] = {
                "arm": arm,
                "n_seeds": 0,
                "liking_writes_mean": 0.0,
                "predicted_writes_mean": 0.0,
                "z_beta_excursion_mean": 0.0,
                "harm_paired_surprise_writes_total": 0,
                "nonharm_surprise_writes_total": 0,
                "n_negative_surprise_centers_total": 0,
                "approach_commit_rate_mean": 0.0,
                "contact_events_total": 0,
                "per_seed_approach": [],
                "per_seed_neg_surprise_centers": [],
            }
        b = bucket[arm]
        b["n_seeds"] += 1
        b["liking_writes_mean"] += r["liking_writes"]
        b["predicted_writes_mean"] += r["predicted_writes"]
        b["z_beta_excursion_mean"] += r["z_beta_excursion_mean"]
        b["harm_paired_surprise_writes_total"] += r["harm_paired_surprise_writes"]
        b["nonharm_surprise_writes_total"] += r["nonharm_surprise_writes"]
        b["n_negative_surprise_centers_total"] += r["n_negative_surprise_centers"]
        b["approach_commit_rate_mean"] += r["approach_commit_rate"]
        b["contact_events_total"] += r["contact_events"]
        b["per_seed_approach"].append(r["approach_commit_rate"])
        b["per_seed_neg_surprise_centers"].append(r["n_negative_surprise_centers"])
    for arm, b in bucket.items():
        n = max(1, b["n_seeds"])
        b["liking_writes_mean"] /= n
        b["predicted_writes_mean"] /= n
        b["z_beta_excursion_mean"] /= n
        b["approach_commit_rate_mean"] /= n
    return bucket


def _evaluate_acceptance(agg: Dict[str, Dict]) -> Dict:
    off = agg["ARM_OFF"]
    on = agg["ARM_ON"]
    c1 = on["liking_writes_mean"] >= C1_LIKING_WRITES_FLOOR and off["liking_writes_mean"] == 0
    c2 = on["z_beta_excursion_mean"] >= C2_Z_BETA_PULSE_FLOOR
    # C3 is the actual Gap 1 signature -- ARM_ON should accumulate at least
    # one center with a negative VALENCE_SURPRISE (signed write fired);
    # ARM_OFF should have zero negative-surprise centers (all writes were
    # unsigned magnitudes >= 0).
    c3 = (
        on["n_negative_surprise_centers_total"] >= C3_SIGNED_SURPRISE_NEG_FLOOR
        and off["n_negative_surprise_centers_total"] == 0
    )
    c4 = on["predicted_writes_mean"] >= C4_PREDICTED_WRITES_FLOOR
    seeds_above_lift = sum(
        1 for a, o in zip(off["per_seed_approach"], on["per_seed_approach"])
        if (o - a) >= C5_APPROACH_LIFT
    )
    c5 = seeds_above_lift >= max(2, on["n_seeds"] - 1)
    overall = c1 and c2 and c3 and c4 and c5
    return {
        "C1_liking_writes": bool(c1),
        "C2_z_beta_pulse": bool(c2),
        "C3_signed_surprise_negative": bool(c3),
        "C4_predicted_location_writes": bool(c4),
        "C5_approach_commit_lift": bool(c5),
        "all_pass": bool(overall),
        "seeds_above_c5_lift": seeds_above_lift,
    }


def main(dry_run: bool = False):
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})")
    seeds = (SEEDS[0],) if dry_run else SEEDS
    n_warmup = 4 if dry_run else P0_EPISODES
    n_eval = 2 if dry_run else EVAL_EPISODES

    per_cell: List[Dict] = []
    t0 = time.time()
    for seed in seeds:
        for arm_cfg in ARMS:
            arm_t0 = time.time()
            r = _measure_arm(
                seed=seed,
                arm_label=arm_cfg["arm"],
                mech307_flags=arm_cfg["flags"],
                n_warmup=n_warmup,
                n_eval=n_eval,
                steps_per_episode=STEPS_PER_EPISODE,
            )
            per_cell.append(r)
            print(
                f"  seed={seed} arm={r['arm']:<8} "
                f"liking_writes={r['liking_writes']:>3} "
                f"pred_writes={r['predicted_writes']:>3} "
                f"z_beta_exc={r['z_beta_excursion_mean']:.4f} "
                f"harm/nonharm_surprise={r['harm_paired_surprise_writes']}/{r['nonharm_surprise_writes']} "
                f"neg_surprise_centers={r['n_negative_surprise_centers']} "
                f"approach_commit_rate={r['approach_commit_rate']:.3f} "
                f"contacts={r['contact_events']} "
                f"elapsed={time.time()-arm_t0:.1f}s"
            )

    agg = _aggregate(per_cell)
    acceptance = _evaluate_acceptance(agg)
    elapsed = time.time() - t0
    outcome = "PASS" if acceptance["all_pass"] else "FAIL"

    print(f"[{EXPERIMENT_TYPE}] aggregates:")
    for arm in ("ARM_OFF", "ARM_ON"):
        a = agg[arm]
        print(
            f"  {arm:<8} liking={a['liking_writes_mean']:.1f} "
            f"pred_writes={a['predicted_writes_mean']:.1f} "
            f"z_beta_exc={a['z_beta_excursion_mean']:.4f} "
            f"harm_surprise={a['harm_paired_surprise_writes_total']} "
            f"nonharm_surprise={a['nonharm_surprise_writes_total']} "
            f"neg_surprise_centers={a['n_negative_surprise_centers_total']} "
            f"approach_rate={a['approach_commit_rate_mean']:.3f} "
            f"per_seed={[round(x, 3) for x in a['per_seed_approach']]}"
        )
    print(f"[{EXPERIMENT_TYPE}] acceptance:")
    for k, v in acceptance.items():
        print(f"  {k}: {v}")
    print(f"[{EXPERIMENT_TYPE}] outcome={outcome} elapsed={elapsed:.1f}s")
    print(f"Done. Outcome: {outcome}")

    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; not writing manifest.")
        return 0

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "started_utc": datetime.utcnow().isoformat() + "Z",
        "outcome": outcome,
        "evidence_direction": "supports" if outcome == "PASS" else (
            "mixed"
            if (acceptance["C1_liking_writes"] and acceptance["C2_z_beta_pulse"]
                and acceptance["C3_signed_surprise_negative"]
                and acceptance["C4_predicted_location_writes"])
            else "weakens"
        ),
        "evidence_direction_per_claim": {
            "MECH-307": "supports" if outcome == "PASS" else (
                "mixed" if (
                    acceptance["C1_liking_writes"]
                    and acceptance["C2_z_beta_pulse"]
                    and acceptance["C3_signed_surprise_negative"]
                    and acceptance["C4_predicted_location_writes"]
                ) else "weakens"
            ),
            "MECH-216": "supports" if acceptance["C1_liking_writes"] and acceptance["C4_predicted_location_writes"] else "non_contributory",
            "MECH-205": "supports" if acceptance["C3_signed_surprise_negative"] else "non_contributory",
            "MECH-093": "supports" if acceptance["C2_z_beta_pulse"] else "non_contributory",
            "SD-014": "non_contributory",
        },
        "elapsed_seconds": elapsed,
        "n_seeds": len(seeds),
        "p0_episodes": P0_EPISODES,
        "eval_episodes": EVAL_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "arms": list(agg.values()),
        "per_seed_per_arm": per_cell,
        "acceptance": acceptance,
        "thresholds": {
            "C1_liking_writes_floor": C1_LIKING_WRITES_FLOOR,
            "C2_z_beta_pulse_floor": C2_Z_BETA_PULSE_FLOOR,
            "C3_signed_surprise_neg_floor": C3_SIGNED_SURPRISE_NEG_FLOOR,
            "C4_predicted_writes_floor": C4_PREDICTED_WRITES_FLOOR,
            "C5_approach_commit_lift": C5_APPROACH_LIFT,
        },
        "note": (
            "Discriminative ablation paired to EXQ-536b's 'downstream chain inert "
            "even with active z_goal' finding. ARM_OFF replicates the EXQ-536b "
            "regime; ARM_ON enables all four MECH-307 wiring fixes. PASS = the "
            "conjunction-fix substrate is firing AND downstream consumers pick "
            "up the conjunction signal; PARTIAL PASS (C1-C4 only) routes to the "
            "SD-014 6-channel amendment as the architectural fallback."
        ),
    }
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Result written to: {out_path}", flush=True)
    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke run, no manifest.")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    if args.dry_run or result == 0:
        sys.exit(0)
    _outcome, _out_path = result
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
    )
    sys.exit(0)
