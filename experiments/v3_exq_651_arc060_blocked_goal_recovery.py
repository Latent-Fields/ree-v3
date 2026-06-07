"""V3-EXQ-651: ARC-060 Hybrid Field+Bank -- Blocked-Goal Recovery (behavioural).

experiment_purpose: evidence
backlog_id: EVB-0076  (proposal EXP-0112)
claim_ids: ['ARC-060']

CLAIM TESTED (ARC-060, architecture_hypothesis):
"A good agent needs both a continuous wanting landscape AND a discrete bank of
unresolved goal traces: field-level wanting supports smooth local guidance, while
ghost-goal traces preserve blocked or deferred goals whose local gradient has
collapsed." ARC-060 predicts: agents with the FIELD ALONE navigate well while a
gradient exists but PREMATURELY ABANDON a blocked/deferred goal once the local
gradient collapses; agents with BOTH layers RECOVER MORE FLEXIBLY after the
invalidation.

This is the architectural-level behavioural test that the sub-mechanism
validations (SD-039 V3-EXQ-494, MECH-292 V3-EXQ-496, MECH-293 V3-EXQ-497, all
PASS) explicitly defer to ARC-060. The proposal's why_now is
"missing_experimental_evidence / synthetic_signals_only / low_exp_conf": the
substrate is wired and unit-validated, but no experiment has tested the ARC-060
recovery prediction itself.

DESIGN: a matched OFF-vs-ON ablation over distinct seeds.
  FIELD_ONLY  (OFF): the shared MECH-269 anchor stack + continuous wanting field
                     (z_goal + GoalState.goal_proximity scoring in E3) WITHOUT
                     the discrete ghost-goal bank (use_mech292_ghost_bank=False,
                     use_mech293_ghost_probes=False, use_sd039_anchor_payload=False).
  FIELD_PLUS_BANK (ON): identical config PLUS the discrete bank
                     (use_sd039_anchor_payload + use_mech292_ghost_bank +
                     use_mech293_ghost_probes all True). The ONLY difference is
                     the discrete ghost-goal layer, so any Phase-C recovery
                     differential isolates ARC-060's discrete-bank contribution.

  Phase A (goal establishment): forced drive+benefit seed z_goal (mirrors
    V3-EXQ-496 _step_episode); a fast-scale boundary is forced every K ticks so
    SD-039 dual-trace anchors populate with goal payloads (ON arm) -- the proven
    494/496 population pattern.
  Phase B (invalidation / gradient collapse): the goal resource is removed and
    the proximity fields recomputed, so the LOCAL resource gradient near the
    agent collapses to flat. The persisted z_goal (the continuous field) and the
    inactive payloaded anchors (the discrete bank) both survive.
  Phase C (recovery measurement): stepping continues with NO further z_goal
    seeding and NO forced boundaries -- the goal is "computationally alive but
    locally gradient-less". Each tick we read GoalState.goal_proximity of the
    current z_world to the PERSISTED z_goal: the latent-space measure of whether
    the agent keeps returning toward the goal (persistence) or drifts away
    (premature abandonment).

PRE-REGISTERED CRITERIA (thresholds are script constants, not post-hoc):
  C1 (non-degeneracy precondition, ON arm): the discrete bank must be genuinely
     engaged -- total MECH-293 ghost candidates admitted over Phase C >= 1 AND
     z_goal active on >= GOAL_ACTIVE_FRAC_MIN of Phase-C ticks. If C1 is NOT met
     on a majority of seeds the run self-routes to non_contributory
     (substrate-not-engaged), NEVER to "weakens" -- a bank that never fired
     cannot falsify ARC-060.
  C2 (clean ablation): the FIELD_ONLY arm admits ZERO ghost candidates (bank
     absent). A non-zero count is a wiring error, surfaced loudly.
  C3 (ARC-060 recovery prediction): mean Phase-C goal_proximity-to-persisted-goal
     is higher for FIELD_PLUS_BANK than FIELD_ONLY by >= GOAL_PROX_MARGIN, on a
     majority of seeds. C3 holding (given C1) SUPPORTS ARC-060; C3 failing (given
     C1) WEAKENS it.

OUTCOME / evidence_direction:
  C1 not met (majority)         -> outcome FAIL, evidence_direction non_contributory.
  C1 met AND C3 met (majority)  -> outcome PASS, evidence_direction supports.
  C1 met AND C3 not met         -> outcome FAIL, evidence_direction weakens.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_651_arc060_blocked_goal_recovery.py [--dry-run]

Writes a flat JSON manifest to REE_assembly/evidence/experiments/.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import reset_all_rng, compute_arm_fingerprint

EVIDENCE_ROOT = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

EXPERIMENT_PURPOSE = "evidence"  # directly tests the ARC-060 recovery hypothesis

# ---------------------------------------------------------------------------
# Pre-registered constants (NOT derived from run statistics)
# ---------------------------------------------------------------------------
SEEDS = [42, 43, 44]
ARMS: List[Tuple[str, bool]] = [("FIELD_ONLY", False), ("FIELD_PLUS_BANK", True)]

PHASE_A_TICKS = 40           # goal establishment + anchor population
PHASE_C_TICKS = 40           # recovery measurement after gradient collapse
FORCE_BOUNDARY_EVERY = 8     # deterministic fast-scale boundaries (494/496 pattern)
FORCE_DRIVE = 0.8            # Phase-A forced drive (matches V3-EXQ-496)
FORCE_BENEFIT = 0.4          # Phase-A forced benefit (matches V3-EXQ-496)

GHOST_FIRE_MIN = 1           # C1: ON arm Phase-C total admitted ghost candidates
GOAL_ACTIVE_FRAC_MIN = 0.5   # C1: fraction of Phase-C ticks with z_goal active
GOAL_PROX_MARGIN = 0.01      # C3: ON mean goal_prox - OFF mean goal_prox >= margin

EPISODES_PER_RUN = PHASE_A_TICKS + PHASE_C_TICKS   # progress denominator M


def _majority(n: int) -> int:
    return (n // 2) + 1


# ---------------------------------------------------------------------------
# Construction (mirrors V3-EXQ-496 _make_env / _make_agent + ghost flags)
# ---------------------------------------------------------------------------
def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=8,
        num_resources=1,
        num_hazards=0,
        resource_benefit=0.5,
        resource_respawn_on_consume=False,   # so the goal can be invalidated
        proximity_benefit_scale=0.05,
        use_proxy_fields=True,
    )


def _make_agent(bank_on: bool, seed: int) -> REEAgent:
    """Shared MECH-269 anchor stack + continuous wanting field; ghost layer toggled."""
    torch.manual_seed(seed)
    env = _make_env(seed=seed)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        # MECH-269 anchor substrate (required for any anchor / payload to exist).
        use_per_stream_vs=True,
        use_event_segmenter=True,
        use_invalidation_trigger=True,
        use_anchor_sets=True,
        # Discrete ghost-goal layer (ARC-060 treatment) -- the ONLY arm difference.
        use_sd039_anchor_payload=bank_on,
        use_mech292_ghost_bank=bank_on,
        use_mech293_ghost_probes=bank_on,
        # Continuous wanting field (shared by both arms).
        drive_weight=2.0,
        use_resource_proximity_head=True,
    )
    cfg.goal.z_goal_enabled = True
    return REEAgent(cfg)


def _goal_resource_cell(env: CausalGridWorldV2) -> Optional[Tuple[int, int]]:
    if env.resources:
        r = env.resources[0]
        return (int(r[0]), int(r[1]))
    return None


# ---------------------------------------------------------------------------
# One seed x arm cell
# ---------------------------------------------------------------------------
def _run_cell(arm_label: str, bank_on: bool, seed: int, dry_run: bool) -> Dict[str, Any]:
    a_ticks = 16 if dry_run else PHASE_A_TICKS
    c_ticks = 8 if dry_run else PHASE_C_TICKS
    # Smaller boundary period in dry-run so the bank still populates at reduced
    # scale (lets the smoke verify the ON ghost-probe path, not just OFF).
    boundary_every = 4 if dry_run else FORCE_BOUNDARY_EVERY
    total_ticks = a_ticks + c_ticks

    reset_all_rng(seed)
    agent = _make_agent(bank_on=bank_on, seed=seed)
    env = _make_env(seed=seed)
    flat_obs, obs_dict = env.reset()
    agent.reset()

    print(f"Seed {seed} Condition {arm_label}", flush=True)

    # --- Phase A: establish goal + populate anchors -----------------------
    for t in range(a_ticks):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        latent = agent.sense(obs_body, obs_world)
        ticks = agent.clock.advance()
        if ticks.get("e1_tick", False):
            e1_prior = agent._e1_tick(latent)
        else:
            e1_prior = torch.zeros(1, agent.config.latent.world_dim)
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        # Forced drive+benefit seed z_goal (V3-EXQ-496 forced pattern).
        agent.update_z_goal(benefit_exposure=FORCE_BENEFIT, drive_level=FORCE_DRIVE)
        # Deterministic fast-scale boundary -> SD-039 dual-trace anchors populate.
        if (
            t > 0
            and (t % boundary_every) == 0
            and agent.hippocampal is not None
            and agent.hippocampal.event_segmenter is not None
            and agent.hippocampal.anchor_set is not None
        ):
            ev = agent.hippocampal.event_segmenter.force_boundary(
                "fast", reason=f"v3_exq_651_t{t}",
            )
            payload = agent.hippocampal.build_goal_payload(
                latent_state=latent,
                goal_state=agent.goal_state,
                residue_field=agent.residue_field,
                bla_output=agent._bla_last_output,
                current_step=int(agent._step_count),
                simulation_mode=False,
            )
            agent.hippocampal.tick_anchor_set(latent, [ev], goal_payload=payload)
        action = agent.select_action(candidates, ticks, temperature=1.0)
        if action is None:
            action = torch.zeros(1, env.action_dim)
            action[0, 0] = 1.0
            agent._last_action = action
        flat_obs, harm_signal, done, info, obs_dict = env.step(action)
        agent._step_count += 1
        if (t + 1) % 10 == 0:
            print(
                f"  [train] arc060 seed={seed} arm={arm_label} "
                f"ep {t + 1}/{total_ticks} phaseA",
                flush=True,
            )

    z_goal_active_after_a = bool(
        agent.goal_state is not None and agent.goal_state.is_active()
    )
    goal_norm_after_a = float(agent.goal_state.goal_norm()) if agent.goal_state else 0.0
    goal_cell = _goal_resource_cell(env)

    # Bank-state snapshot at end of Phase A (ON arm).
    bank_size_after_a = 0
    if bank_on and agent.hippocampal is not None and agent.goal_state is not None:
        try:
            zg = agent.goal_state.z_goal if agent.goal_state.is_active() else None
            bank = agent.hippocampal.rank_ghost_goals(zg)
            bank_size_after_a = len(bank)
        except Exception:
            bank_size_after_a = -1

    # --- Phase B: invalidate -> collapse the local gradient ---------------
    env.resources = []
    env._compute_proximity_fields()
    obs_dict = env._get_observation_dict()

    # --- Phase C: recovery measurement ------------------------------------
    goal_prox_series: List[float] = []
    n_goal_active_c = 0
    ghost_admitted_total = 0
    ghost_fire_ticks = 0
    for t in range(c_ticks):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        latent = agent.sense(obs_body, obs_world)
        ticks = agent.clock.advance()
        if ticks.get("e1_tick", False):
            e1_prior = agent._e1_tick(latent)
        else:
            e1_prior = torch.zeros(1, agent.config.latent.world_dim)
        # generate_trajectories -> _e3_tick threads current_z_goal -> ghost probes.
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)

        # Latent-space recovery signal: proximity of current z_world to the
        # PERSISTED z_goal (NOT re-seeded in Phase C). High = staying near the
        # goal scene (persistence); low = drifting away (abandonment).
        if agent.goal_state is not None and agent.goal_state.is_active():
            n_goal_active_c += 1
            zw = latent.z_world
            if zw.dim() == 1:
                zw = zw.unsqueeze(0)
            prox = float(agent.goal_state.goal_proximity(zw).mean().item())
            goal_prox_series.append(prox)

        # MECH-293 ghost-probe engagement (ON arm only fires).
        if agent.hippocampal is not None:
            diag = agent.hippocampal.get_last_propose_diagnostics()
            n_adm = int(diag.get("mech293_n_ghost_admitted", 0) or 0)
            if n_adm > 0:
                ghost_admitted_total += n_adm
                ghost_fire_ticks += 1

        action = agent.select_action(candidates, ticks, temperature=1.0)
        if action is None:
            action = torch.zeros(1, env.action_dim)
            action[0, 0] = 1.0
            agent._last_action = action
        flat_obs, harm_signal, done, info, obs_dict = env.step(action)
        agent._step_count += 1
        if (t + 1) % 10 == 0:
            print(
                f"  [train] arc060 seed={seed} arm={arm_label} "
                f"ep {a_ticks + t + 1}/{total_ticks} phaseC",
                flush=True,
            )

    mean_goal_prox_c = (
        sum(goal_prox_series) / len(goal_prox_series) if goal_prox_series else 0.0
    )
    # Persistence shape: late-third vs early-third mean (drop = abandonment).
    third = max(1, len(goal_prox_series) // 3)
    early_prox = (
        sum(goal_prox_series[:third]) / third if goal_prox_series else 0.0
    )
    late_prox = (
        sum(goal_prox_series[-third:]) / third if goal_prox_series else 0.0
    )
    goal_active_frac_c = n_goal_active_c / float(c_ticks)

    full_config = {
        "arm_label": arm_label,
        "bank_on": bank_on,
        "size": 8,
        "num_resources": 1,
        "num_hazards": 0,
        "resource_respawn_on_consume": False,
        "use_proxy_fields": True,
        "use_per_stream_vs": True,
        "use_event_segmenter": True,
        "use_invalidation_trigger": True,
        "use_anchor_sets": True,
        "use_sd039_anchor_payload": bank_on,
        "use_mech292_ghost_bank": bank_on,
        "use_mech293_ghost_probes": bank_on,
        "drive_weight": 2.0,
        "use_resource_proximity_head": True,
        "z_goal_enabled": True,
        "phase_a_ticks": a_ticks,
        "phase_c_ticks": c_ticks,
        "force_boundary_every": FORCE_BOUNDARY_EVERY,
        "force_drive": FORCE_DRIVE,
        "force_benefit": FORCE_BENEFIT,
    }
    row: Dict[str, Any] = {
        "arm_label": arm_label,
        "bank_on": bank_on,
        "seed": seed,
        "z_goal_active_after_phase_a": z_goal_active_after_a,
        "goal_norm_after_phase_a": goal_norm_after_a,
        "goal_resource_cell": list(goal_cell) if goal_cell else None,
        "bank_size_after_phase_a": bank_size_after_a,
        "mean_goal_prox_phase_c": mean_goal_prox_c,
        "early_goal_prox_phase_c": early_prox,
        "late_goal_prox_phase_c": late_prox,
        "goal_active_frac_phase_c": goal_active_frac_c,
        "ghost_admitted_total_phase_c": ghost_admitted_total,
        "ghost_fire_ticks_phase_c": ghost_fire_ticks,
        "n_goal_prox_samples": len(goal_prox_series),
    }
    # Phase-0 arm fingerprint (emit-only; required for multi-arm grids).
    row["arm_fingerprint"] = compute_arm_fingerprint(
        config_slice=full_config,
        seed=seed,
        script_path=Path(__file__),
        rng_fully_reset=True,
    )

    # Per-run cosmetic verdict (advances the runner's runs_done counter; the
    # scientific verdict is computed across arm pairs after all cells run).
    run_ok = (z_goal_active_after_a and goal_active_frac_c >= GOAL_ACTIVE_FRAC_MIN)
    print(f"verdict: {'PASS' if run_ok else 'FAIL'}", flush=True)
    return row


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(dry_run: bool = False) -> int:
    t0 = time.time()
    seeds = SEEDS[:1] if dry_run else SEEDS
    print(
        f"[v3_exq_651] ARC-060 blocked-goal recovery: "
        f"{len(seeds)} seed(s) x {len(ARMS)} arm(s)"
        f"{' (dry-run)' if dry_run else ''}",
        flush=True,
    )

    cells: Dict[Tuple[int, str], Dict[str, Any]] = {}
    arm_results: List[Dict[str, Any]] = []
    for seed in seeds:
        for arm_label, bank_on in ARMS:
            row = _run_cell(arm_label, bank_on, seed, dry_run)
            cells[(seed, arm_label)] = row
            arm_results.append(row)

    # --- Per-seed pairing + criteria --------------------------------------
    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        off = cells[(seed, "FIELD_ONLY")]
        on = cells[(seed, "FIELD_PLUS_BANK")]
        c1_ghost = on["ghost_admitted_total_phase_c"] >= GHOST_FIRE_MIN
        c1_goal_active = on["goal_active_frac_phase_c"] >= GOAL_ACTIVE_FRAC_MIN
        c1_met = bool(c1_ghost and c1_goal_active)
        c2_clean = off["ghost_admitted_total_phase_c"] == 0
        prox_delta = on["mean_goal_prox_phase_c"] - off["mean_goal_prox_phase_c"]
        c3_recovery = prox_delta >= GOAL_PROX_MARGIN
        per_seed.append({
            "seed": seed,
            "c1_precondition_met": c1_met,
            "c1_ghost_admitted_total": on["ghost_admitted_total_phase_c"],
            "c1_goal_active_frac": on["goal_active_frac_phase_c"],
            "c2_off_arm_clean": bool(c2_clean),
            "off_mean_goal_prox": off["mean_goal_prox_phase_c"],
            "on_mean_goal_prox": on["mean_goal_prox_phase_c"],
            "goal_prox_delta_on_minus_off": prox_delta,
            "c3_recovery_met": bool(c3_recovery),
        })

    n = len(seeds)
    maj = _majority(n)
    n_c1 = sum(1 for s in per_seed if s["c1_precondition_met"])
    n_c2_clean = sum(1 for s in per_seed if s["c2_off_arm_clean"])
    n_c3 = sum(1 for s in per_seed if s["c1_precondition_met"] and s["c3_recovery_met"])

    precondition_met = n_c1 >= maj
    if not precondition_met:
        outcome = "FAIL"
        evidence_direction = "non_contributory"
        evidence_note = (
            "ARC-060 recovery test inconclusive: the discrete ghost-goal bank "
            f"was engaged on only {n_c1}/{n} seeds (need >= {maj}). C1 "
            "non-degeneracy precondition (ghost candidates admitted >= "
            f"{GHOST_FIRE_MIN} AND z_goal active on >= {GOAL_ACTIVE_FRAC_MIN} of "
            "Phase-C ticks) was not met on a majority of seeds, so the run "
            "cannot adjudicate ARC-060 -- a bank that never fired cannot "
            "falsify the field+bank prediction. Routed to substrate-not-engaged."
        )
    elif n_c3 >= maj:
        outcome = "PASS"
        evidence_direction = "supports"
        evidence_note = (
            "ARC-060 supported: with the discrete ghost-goal bank engaged (C1 "
            f"met on {n_c1}/{n} seeds), FIELD_PLUS_BANK kept mean Phase-C "
            "goal-proximity-to-persisted-goal higher than FIELD_ONLY by >= "
            f"{GOAL_PROX_MARGIN} on {n_c3}/{n} seeds after the local gradient "
            "collapsed. The field-alone agent abandoned the blocked goal; the "
            "field+bank agent recovered toward it, matching ARC-060's hybrid "
            "prediction."
        )
    else:
        outcome = "FAIL"
        evidence_direction = "weakens"
        evidence_note = (
            "ARC-060 weakened: the discrete ghost-goal bank was genuinely "
            f"engaged (C1 met on {n_c1}/{n} seeds) but FIELD_PLUS_BANK did NOT "
            "improve Phase-C goal-proximity recovery over FIELD_ONLY by >= "
            f"{GOAL_PROX_MARGIN} on a majority of seeds (recovery margin met on "
            f"only {n_c3}/{n}). The discrete-bank layer added no measurable "
            "blocked-goal recovery advantage over the continuous field alone."
        )

    elapsed = time.time() - t0
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"v3_exq_651_arc060_blocked_goal_recovery_{ts}_v3"

    manifest: Dict[str, Any] = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": "v3_exq_651_arc060_blocked_goal_recovery",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "backlog_id": "EVB-0076",
        "claim_ids": ["ARC-060"],
        "claim_ids_tested": ["ARC-060"],
        "evidence_class": "experimental",
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": {"ARC-060": evidence_direction},
        "evidence_direction_note": evidence_note,
        "outcome": outcome,
        "elapsed_sec": elapsed,
        "seeds": seeds,
        "arms": [a[0] for a in ARMS],
        "pre_registered_thresholds": {
            "ghost_fire_min": GHOST_FIRE_MIN,
            "goal_active_frac_min": GOAL_ACTIVE_FRAC_MIN,
            "goal_prox_margin": GOAL_PROX_MARGIN,
            "majority_seeds": maj,
        },
        "criteria_summary": {
            "n_seeds": n,
            "n_c1_precondition_met": n_c1,
            "n_c2_off_arm_clean": n_c2_clean,
            "n_c3_recovery_met": n_c3,
            "precondition_met": precondition_met,
        },
        "per_seed": per_seed,
        "arm_results": arm_results,
        "dry_run": bool(dry_run),
    }

    for s in per_seed:
        print(
            f"  seed {s['seed']}: C1={s['c1_precondition_met']} "
            f"(ghost={s['c1_ghost_admitted_total']}, "
            f"goal_active={s['c1_goal_active_frac']:.2f}) "
            f"C2_clean={s['c2_off_arm_clean']} "
            f"OFF_prox={s['off_mean_goal_prox']:.4f} "
            f"ON_prox={s['on_mean_goal_prox']:.4f} "
            f"delta={s['goal_prox_delta_on_minus_off']:+.4f} "
            f"C3={s['c3_recovery_met']}",
            flush=True,
        )
    print(
        f"[v3_exq_651] outcome={outcome} direction={evidence_direction} "
        f"({elapsed:.1f}s)",
        flush=True,
    )

    out_path: Optional[Path] = None
    if not dry_run:
        out_dir = EVIDENCE_ROOT / "v3_exq_651_arc060_blocked_goal_recovery"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{run_id}.json"
        with open(out_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"Result written to: {out_path}", flush=True)

    return {"outcome": outcome, "manifest_path": out_path, "dry_run": bool(dry_run)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true",
        help="1 seed x 2 arms at reduced ticks; no manifest, no sentinel.",
    )
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)

    if not result["dry_run"]:
        _outcome_raw = str(result["outcome"]).upper()
        emit_outcome(
            outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
            manifest_path=result["manifest_path"],
        )
    sys.exit(0 if result["outcome"] == "PASS" else 1)
