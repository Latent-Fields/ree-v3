"""
V3-EXQ-667 -- Q-043 exploration-magnitude sweep on the InfantCurriculumScheduler
Phase 0->1 reachability probe (591c lineage).

PRIMARY ROUTING OUTPUT of failure_autopsy_V3-EXQ-591c_2026-06-11 (confirmed).
591c FAILed: arming the exploration-diversity stack (MECH-313 noise floor +
MECH-314 structured curiosity) at its LANDED-DEFAULT magnitudes
(noise_floor_alpha=0.1, curiosity_weight=0.05) did NOT rescue the worst-case
early-collapse seed (seed 46: h_pos_mean=0.0375, IDENTICAL to 591b with the stack
OFF). The autopsy diagnosis: ARC-065's exploration-diversity substrate is
load-bearing but UNDER-POWERED at default magnitudes for the collapse-prone seed
tail.

SLEEP DRIVER: K=never (SleepLoopManager instantiated with K > total episodes via
the diversity-armed builder; never fires during this readiness probe -- inherited
from the 591/591b/591c lineage).

QUESTION (Q-043, registered open_question)
------------------------------------------
Q-043 asks how the relative weights of MECH-313 (noise floor) and MECH-314
(structured curiosity) should be calibrated for V3. Its resolution path is a
parametric sweep of those weights. This experiment runs the focused 1-D
magnitude-scale form of that sweep, asking the autopsy's specific fork: does
scaling noise_floor_alpha + curiosity_weight ABOVE their landed defaults rescue
the worst-case Phase-0 collapse seed?

DESIGN
------
Joint magnitude-scale grid. Re-run the EXACT 591c reachability probe (same
InfantCurriculumScheduler arm, same 5 seeds 42-46, same 160 ep x 200 steps,
same landed H_POS_FRAC_OF_MAX=0.20 Phase 0->1 gate -- NOT changed) at four
magnitude scales S applied JOINTLY to both knobs:
    S=1.0 : noise_floor_alpha=0.1,  curiosity_weight=0.05  (reproduces 591c; anchor)
    S=2.0 : noise_floor_alpha=0.2,  curiosity_weight=0.10
    S=4.0 : noise_floor_alpha=0.4,  curiosity_weight=0.20
    S=8.0 : noise_floor_alpha=0.8,  curiosity_weight=0.40
curiosity_weight scales all three MECH-314 sub-flavour weights
(novelty/uncertainty/learning_progress) together (the "curiosity_weight" the
591c manifest names). Multi-arm (seed x scale) grid; per-cell RNG reset +
arm_fingerprint via experiments._lib.arm_fingerprint.arm_cell.

The Phase 0->1 threshold and the genuine-exploration definition are UNCHANGED
from 591c -- this experiment does NOT touch the gate (the seed-45 gate-
permissiveness brittleness is a SEPARATE ARC-046 K-of-N/EMA follow-on, explicitly
out of scope; lowering the threshold is NOT the route).

INTERPRETATION (the diagnose-first fork the autopsy set up)
-----------------------------------------------------------
- (a) If the worst-case collapse seed escapes Phase 0 with GENUINE exploration at
      some swept magnitude above default -> stronger default magnitudes rescue the
      collapse-prone tail. The fix is CONFIG-ONLY (raise the ARC-065 default
      magnitudes). Label config_magnitude_rescues_collapse_seed.
- (b) If the collapse seed stays collapsed at EVERY swept magnitude (incl 8x) ->
      passive noise+curiosity is an insufficient translation of actively-driven
      early-development exploration; an ACTIVE Phase-0 exploration-shaping substrate
      is required. Label magnitude_insufficient_active_phase0_shaping_required.

READINESS / NON-VACUITY (same-statistic gate, the V3-EXQ-643 lesson)
--------------------------------------------------------------------
The whole sweep is vacuous if the swept knobs do NOT actually modulate exploration.
POSITIVE CONTROL = the HEALTHY seeds (those that genuinely explored in the default
1x arm -- 591c had 42/43/44). The readiness statistic is the SAME quantity the
rescue criterion routes on (per-episode h_pos / pos_entropy): the RANGE, across the
magnitude arms, of the healthy-seed-averaged h_pos_mean must clear a floor. If even
the healthy seeds' exploration does not move as the magnitude is scaled 1x->8x, the
knobs are inert and the run self-routes substrate_not_ready_requeue (NEVER a
substrate verdict). A below-floor reading means "knobs not operative", never
"magnitude insufficient".

ASCII-only output.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from infant_curriculum import InfantCurriculumScheduler, H_POS_FRAC_OF_MAX  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

# Reuse the canonical 591 helpers + constants (DRY -- no copy-drift).
from v3_exq_591_isef005_curriculum_vs_flat_v3 import (  # noqa: E402
    _extract_obs,
    BODY_OBS_DIM,
    WORLD_OBS_DIM,
    GRID_SIZE,
    ACTION_DIM,
)

QUEUE_ID = "V3-EXQ-667"
EXPERIMENT_TYPE = "v3_exq_667_q043_exploration_magnitude_sweep"
CLAIM_IDS: List[str] = ["Q-043"]  # registered open_question; relative-weight calibration
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44, 45, 46]
N_EPISODES = 160          # faithful to V3-EXQ-591b / 591c
STEPS_PER_EPISODE = 200   # faithful to V3-EXQ-591b / 591c

# Landed default magnitudes (the 591c baseline anchor).
DEFAULT_NOISE_FLOOR_ALPHA = 0.1   # MECH-313
DEFAULT_CURIOSITY_WEIGHT = 0.05   # MECH-314 (all three sub-flavour weights)
# Joint magnitude scales applied to BOTH knobs above their defaults.
MAGNITUDE_SCALES = [1.0, 2.0, 4.0, 8.0]

H_MAX = math.log(GRID_SIZE ** 2)
PHASE_01_THRESHOLD = H_POS_FRAC_OF_MAX * H_MAX   # 0.20 * ln(144) ~= 0.994 (UNCHANGED gate)
# Genuine-exploration floors (UNCHANGED from 591c -- NOT a gate-permissiveness edit).
GENUINE_EXPLORATION_H_POS_MEAN_FLOOR = 0.20
GENUINE_EXPLORATION_MIN_CROSSINGS = 2
# Readiness floor: the swept magnitude must move healthy-seed exploration by at least
# this much (h_pos range across arms) for the rescue/collapse verdict to be trusted.
READINESS_RANGE_FLOOR = 0.05


def _arm_label(scale: float) -> str:
    return f"mag_{scale:g}x"


def _build_sweep_agent(*, noise_floor_alpha: float, curiosity_weight: float,
                       n_episodes: int) -> REEAgent:
    """Mirror of V3-EXQ-591c's _build_diversity_agent with the exploration-diversity
    stack armed (MECH-313 noise floor + MECH-314 curiosity ON) but at SWEPT
    magnitudes. Everything else bit-identical to the 591/591b/591c agent build."""
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        z_goal_enabled=True,
        drive_weight=2.0,
        novelty_bonus_weight=0.5,
        use_sleep_loop=True,
        sleep_loop_episodes_K=n_episodes + 1,   # K=never (> total episodes)
        # --- exploration-diversity stack at SWEPT magnitudes (the intervention) ---
        use_noise_floor=True,
        noise_floor_alpha=noise_floor_alpha,            # MECH-313 (default 0.1)
        use_structured_curiosity=True,
        curiosity_novelty_weight=curiosity_weight,      # MECH-314a (default 0.05)
        curiosity_uncertainty_weight=curiosity_weight,  # MECH-314b
        curiosity_learning_progress_weight=curiosity_weight,  # MECH-314c
        # use_support_preserving_cem defaults True (SP-CEM main-path; left implicit)
    )
    cfg.latent.alpha_world = 0.9
    cfg.sws_enabled = True
    cfg.rem_enabled = True
    return REEAgent(cfg)


def _run_cell(*, seed: int, scale: float, n_episodes: int,
              steps_per_episode: int) -> Dict[str, Any]:
    """One (seed x magnitude-scale) cell of the sweep. RNG reset + arm_fingerprint
    via arm_cell; training loop mirrors V3-EXQ-591c._run_seed exactly."""
    alpha = DEFAULT_NOISE_FLOOR_ALPHA * scale
    cw = DEFAULT_CURIOSITY_WEIGHT * scale
    label = _arm_label(scale)

    config_slice = {
        "experiment_type": EXPERIMENT_TYPE,
        "arm": label,
        "magnitude_scale": scale,
        "noise_floor_alpha": alpha,
        "curiosity_weight": cw,
        "use_noise_floor": True,
        "use_structured_curiosity": True,
        "alpha_world": 0.9,
        "grid_size": GRID_SIZE,
        "steps_per_episode": steps_per_episode,
        "n_episodes": n_episodes,
        "phase_0to1_threshold": round(PHASE_01_THRESHOLD, 6),
    }

    print(f"Seed {seed} Condition {label}", flush=True)

    with arm_cell(seed, config_slice=config_slice, script_path=Path(__file__)) as cell:
        # arm_cell.__enter__ has already done the complete RNG reset (torch+cuda+
        # numpy+random) keyed on `seed`, so the only difference across arms for one
        # seed is the swept magnitude config.
        agent = _build_sweep_agent(
            noise_floor_alpha=alpha, curiosity_weight=cw, n_episodes=n_episodes)
        sched = InfantCurriculumScheduler(grid_size=GRID_SIZE)

        h_pos_window: deque = deque(maxlen=100)  # rolling (informational only)
        per_ep_h_pos: List[float] = []
        phase_01_at: Optional[int] = None
        phase_12_at: Optional[int] = None

        for ep in range(n_episodes):
            sched.env_kwargs()  # Phase 0 -> all infant features OFF (side-effect free read)
            agent.config.e3.novelty_bonus_weight = float(
                sched.config_overrides().get("novelty_bonus_weight", 0.5))

            env = CausalGridWorldV2(
                size=GRID_SIZE,
                seed=seed * n_episodes + ep,
                resource_respawn_on_consume=True,
                pos_telemetry_enabled=True,
                traj_telemetry_enabled=True,
            )
            _flat, obs_dict = env.reset()
            ob, ow = _extract_obs(obs_dict)

            ep_h_pos = -1.0
            ep_benefit_contacts = 0

            for _step in range(steps_per_episode):
                with torch.no_grad():
                    action = agent.act_with_split_obs(obs_body=ob, obs_world=ow)
                ai = int(action.argmax().item()) % ACTION_DIM
                _o, harm_signal, done, info, obs_dict = env.step(ai)
                agent.update_residue(float(harm_signal))
                ob, ow = _extract_obs(obs_dict)
                benefit = float(ob[11].item()) if ob.shape[0] > 11 else 0.0
                energy = float(ob[3].item()) if ob.shape[0] > 3 else 0.5
                drive = max(0.0, min(1.0, 1.0 - energy))
                agent.update_z_goal(benefit_exposure=benefit, drive_level=drive)
                ep_h_pos = float(info.get("pos_entropy", -1.0))
                ep_benefit_contacts += int(
                    float(info.get("transient_benefit_contact_this_tick", 0.0)) > 0.0)
                if done:
                    _flat, obs_dict = env.reset()
                    ob, ow = _extract_obs(obs_dict)

            z_norm = agent.goal_state.goal_norm() if agent.goal_state is not None else 0.0
            cov = float(agent.residue_field.get_coverage_telemetry()["residue_coverage_pct"])

            per_ep_h_pos.append(ep_h_pos)
            h_pos_window.append(ep_h_pos)

            prev_phase = sched.current_phase
            sched.update(
                ep,
                h_pos=ep_h_pos if ep_h_pos >= 0.0 else None,
                z_goal_norm=z_norm,
                benefit_contacts=ep_benefit_contacts,
                residue_coverage_pct=cov,
            )
            if prev_phase == 0 and sched.current_phase >= 1 and phase_01_at is None:
                phase_01_at = ep
            if prev_phase <= 1 and sched.current_phase >= 2 and phase_12_at is None:
                phase_12_at = ep

            if (ep + 1) % 50 == 0 or (ep + 1) == n_episodes:
                print(
                    f"  [train] sweep seed={seed} {label} ep {ep + 1}/{n_episodes}"
                    f" phase={sched.current_phase} h_pos={ep_h_pos:.3f}"
                    f" z_goal={z_norm:.4f}",
                    flush=True,
                )

        valid = [h for h in per_ep_h_pos if h >= 0.0]
        eligible = [h for h in per_ep_h_pos[100:] if h >= 0.0]  # post Phase-0 ep_min
        n_eligible_ge_threshold = sum(1 for h in eligible if h >= PHASE_01_THRESHOLD)
        final_phase = sched.current_phase
        reached_phase1 = final_phase >= 1
        h_pos_mean = (sum(valid) / len(valid)) if valid else -1.0
        genuine_exploration = bool(
            reached_phase1
            and h_pos_mean >= GENUINE_EXPLORATION_H_POS_MEAN_FLOOR
            and n_eligible_ge_threshold >= GENUINE_EXPLORATION_MIN_CROSSINGS
        )
        print(f"verdict: {'PASS' if (reached_phase1 and genuine_exploration) else 'FAIL'}",
              flush=True)

        row: Dict[str, Any] = {
            "arm": label,
            "magnitude_scale": scale,
            "seed": seed,
            "noise_floor_alpha": round(alpha, 6),
            "curiosity_weight": round(cw, 6),
            "curriculum_final_phase": final_phase,
            "phase_0to1_advanced_at_episode": phase_01_at,
            "phase_1to2_advanced_at_episode": phase_12_at,
            "reached_phase1": reached_phase1,
            "genuine_exploration": genuine_exploration,
            "h_pos_min": round(min(valid), 4) if valid else -1.0,
            "h_pos_mean": round(h_pos_mean, 4) if valid else -1.0,
            "h_pos_max": round(max(valid), 4) if valid else -1.0,
            "h_pos_std": round((sum((h - h_pos_mean) ** 2 for h in valid) / len(valid)) ** 0.5, 4) if valid else -1.0,
            "n_eligible_episodes": len(eligible),
            "n_eligible_ge_threshold": n_eligible_ge_threshold,
        }
        cell.stamp(row)  # writes row["arm_fingerprint"]; one call discharges both obligations

    return row


def run_experiment(*, dry_run: bool = False) -> Dict[str, Any]:
    seeds = [42, 46] if dry_run else SEEDS
    scales = [1.0, 8.0] if dry_run else MAGNITUDE_SCALES
    n_episodes = 2 if dry_run else N_EPISODES

    print(
        f"V3-EXQ-667 Q-043 magnitude sweep: seeds={seeds} scales={scales}"
        f" n_episodes={n_episodes} steps={STEPS_PER_EPISODE}"
        f" threshold={PHASE_01_THRESHOLD:.4f} (default alpha={DEFAULT_NOISE_FLOOR_ALPHA}"
        f" cw={DEFAULT_CURIOSITY_WEIGHT})",
        flush=True,
    )

    arm_results: List[Dict[str, Any]] = []
    # results[(label, seed)] = row
    by_cell: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for scale in scales:
        for seed in seeds:
            row = _run_cell(seed=seed, scale=scale, n_episodes=n_episodes,
                            steps_per_episode=STEPS_PER_EPISODE)
            arm_results.append(row)
            by_cell[(_arm_label(scale), seed)] = row

    default_label = _arm_label(scales[0])  # the 1x anchor
    above_default_labels = [_arm_label(s) for s in scales if s > scales[0]]

    # Healthy seeds = genuine exploration at the DEFAULT (1x) arm (positive control).
    healthy_seeds = [
        s for s in seeds if by_cell[(default_label, s)]["genuine_exploration"]
    ]
    # Collapse-prone tail = NOT genuine at the default arm (591c: seeds 45, 46).
    collapse_seeds_default = [
        s for s in seeds if not by_cell[(default_label, s)]["genuine_exploration"]
    ]

    # --- READINESS (same statistic the rescue criterion routes on: per-episode h_pos) ---
    # Range, across magnitude arms, of the healthy-seed-averaged h_pos_mean. If the
    # knobs do not move healthy-seed exploration as magnitude scales, the sweep is vacuous.
    readiness_range = -1.0
    per_arm_healthy_mean: Dict[str, float] = {}
    if healthy_seeds:
        for scale in scales:
            lbl = _arm_label(scale)
            vals = [by_cell[(lbl, s)]["h_pos_mean"] for s in healthy_seeds
                    if by_cell[(lbl, s)]["h_pos_mean"] >= 0.0]
            if vals:
                per_arm_healthy_mean[lbl] = sum(vals) / len(vals)
        if len(per_arm_healthy_mean) >= 2:
            readiness_range = max(per_arm_healthy_mean.values()) - min(per_arm_healthy_mean.values())
    readiness_ok = bool(healthy_seeds) and readiness_range >= READINESS_RANGE_FLOOR

    # --- RESCUE: for each collapse seed, min scale (above default) at which it becomes
    # genuinely exploring. None if never rescued. ---
    rescue_scale_by_seed: Dict[int, Optional[float]] = {}
    for s in collapse_seeds_default:
        rescued_at: Optional[float] = None
        for scale in scales:
            if scale <= scales[0]:
                continue
            if by_cell[(_arm_label(scale), s)]["genuine_exploration"]:
                rescued_at = scale
                break
        rescue_scale_by_seed[s] = rescued_at
    all_collapse_rescued = (
        len(collapse_seeds_default) > 0
        and all(rescue_scale_by_seed[s] is not None for s in collapse_seeds_default)
    )

    # --- No-regression guard (informational): healthy seeds stay genuine at every scale. ---
    no_healthy_regression = all(
        by_cell[(_arm_label(scale), s)]["genuine_exploration"]
        for scale in scales for s in healthy_seeds
    ) if healthy_seeds else False

    # Non-degeneracy of the rescue criterion: the swept magnitude must actually move
    # exploration (readiness_ok) AND there must be a collapse seed to rescue.
    rescue_criterion_non_degenerate = bool(readiness_ok and len(collapse_seeds_default) >= 1)

    # --- self-route ---
    if not readiness_ok:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        ev_dir = "non_contributory"
    elif len(collapse_seeds_default) == 0:
        # 591c's worst-case collapse did not reproduce here (every seed already
        # genuine at default) -> the rescue fork is not adjudicable on this run.
        outcome = "FAIL"
        label = "collapse_not_reproduced_inconclusive"
        ev_dir = "non_contributory"
    elif all_collapse_rescued:
        # Stronger default magnitudes rescue the collapse-prone tail -> config-only fix.
        outcome = "PASS"
        label = "config_magnitude_rescues_collapse_seed"
        ev_dir = "supports"
    else:
        # Collapse persists at every swept magnitude -> active Phase-0 exploration-
        # shaping substrate required (passive noise+curiosity insufficient).
        outcome = "FAIL"
        label = "magnitude_insufficient_active_phase0_shaping_required"
        ev_dir = "does_not_support"

    return {
        "outcome": outcome,
        "label": label,
        "evidence_direction": ev_dir,
        "arm_results": arm_results,
        "seeds": list(seeds),
        "scales": list(scales),
        "healthy_seeds": healthy_seeds,
        "collapse_seeds_default": collapse_seeds_default,
        "rescue_scale_by_seed": {str(k): v for k, v in rescue_scale_by_seed.items()},
        "all_collapse_rescued": all_collapse_rescued,
        "no_healthy_regression": no_healthy_regression,
        "readiness_range": readiness_range,
        "readiness_ok": readiness_ok,
        "per_arm_healthy_mean": {k: round(v, 4) for k, v in per_arm_healthy_mean.items()},
        "rescue_criterion_non_degenerate": rescue_criterion_non_degenerate,
        "above_default_labels": above_default_labels,
    }


def main(*, dry_run: bool = False) -> Tuple[str, Path]:
    result = run_experiment(dry_run=dry_run)
    outcome = result["outcome"]

    run_id = f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_path = out_dir / f"{run_id}.json"

    seeds_used = result["seeds"]
    scales_used = result["scales"]

    manifest = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": result["evidence_direction"],
        "sleep_driver_pattern": "K=never (SleepLoopManager K > total episodes; never fires)",
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "dry_run": dry_run,
        "config": {
            "seeds": seeds_used,
            "magnitude_scales": scales_used,
            "default_noise_floor_alpha": DEFAULT_NOISE_FLOOR_ALPHA,
            "default_curiosity_weight": DEFAULT_CURIOSITY_WEIGHT,
            "swept_arms": [
                {
                    "arm": _arm_label(s),
                    "magnitude_scale": s,
                    "noise_floor_alpha": round(DEFAULT_NOISE_FLOOR_ALPHA * s, 6),
                    "curiosity_weight": round(DEFAULT_CURIOSITY_WEIGHT * s, 6),
                }
                for s in scales_used
            ],
            "n_episodes": N_EPISODES if not dry_run else 2,
            "steps_per_episode": STEPS_PER_EPISODE,
            "grid_size": GRID_SIZE,
            "h_pos_frac_of_max": H_POS_FRAC_OF_MAX,
            "phase_0to1_threshold": round(PHASE_01_THRESHOLD, 6),
            "genuine_exploration_h_pos_mean_floor": GENUINE_EXPLORATION_H_POS_MEAN_FLOOR,
            "genuine_exploration_min_crossings": GENUINE_EXPLORATION_MIN_CROSSINGS,
            "readiness_range_floor": READINESS_RANGE_FLOOR,
            "arm": "InfantCurriculumScheduler (experiments/infant_curriculum.py)",
            "diversity_stack": {
                "use_noise_floor": True,
                "use_structured_curiosity": True,
                "use_support_preserving_cem": "default-on (main-path since 2026-05-17)",
                "swept_knobs": "noise_floor_alpha + curiosity_weight scaled jointly above landed defaults",
            },
        },
        "acceptance_criteria": {
            "C_rescue_primary": (
                "load-bearing: every default-arm collapse seed (NOT genuinely exploring"
                " at the 1x default) reaches Phase 1 with GENUINE exploration"
                f" (h_pos_mean >= {GENUINE_EXPLORATION_H_POS_MEAN_FLOOR} AND"
                f" >= {GENUINE_EXPLORATION_MIN_CROSSINGS} eligible crossings) at some"
                " swept magnitude ABOVE default -> config-only fix. If it stays collapsed"
                " at every magnitude (incl 8x) -> active Phase-0 exploration-shaping"
                " substrate required."
            ),
            "C_readiness_nonvacuity": (
                "the swept magnitude must MOVE exploration on the healthy-seed positive"
                " control: range across magnitude arms of the healthy-seed-averaged"
                f" h_pos_mean >= {READINESS_RANGE_FLOOR} (SAME statistic the rescue"
                " criterion routes on). Below floor -> substrate_not_ready_requeue, NEVER"
                " a substrate verdict."
            ),
            "C_no_regression": (
                "informational: healthy seeds (genuine at default) stay genuine at every"
                " swept magnitude (stronger exploration must not break the healthy seeds)."
            ),
            "C_gate_unchanged": (
                "the Phase 0->1 threshold and genuine-exploration definition are UNCHANGED"
                " from 591c. The seed-45 gate-permissiveness brittleness is a SEPARATE"
                " ARC-046 K-of-N/EMA follow-on -- out of scope here; lowering the threshold"
                " is NOT the route."
            ),
        },
        "interpretation": {
            "label": result["label"],
            "preconditions": [
                {
                    "name": "healthy_seed_h_pos_range_across_arms_clears_floor",
                    "description": (
                        "Non-vacuity / same-statistic gate. On the healthy-seed positive"
                        " control, the swept knobs must actually move exploration (h_pos)."
                        " The measured quantity is the RANGE (max minus min) across the"
                        " swept-scale arms of the healthy-seed-averaged h_pos_mean -- the"
                        " SAME range statistic the rescue/collapse criterion routes on"
                        " (per-episode pos_entropy), NOT a mean-abs/norm proxy. If even the"
                        " healthy seeds' h_pos range across arms stays below the floor as"
                        " alpha/curiosity scale 1x->8x, the knobs are inert and the"
                        " collapse-persists verdict cannot be trusted."
                    ),
                    "measured": round(result["readiness_range"], 4),
                    "threshold": READINESS_RANGE_FLOOR,
                    "direction": "lower",
                    "control": (
                        "healthy seeds (genuine-exploration in the default 1x arm) -- the"
                        " positive control where stronger noise/curiosity should change"
                        " position entropy"
                    ),
                    "met": bool(result["readiness_ok"]),
                },
            ],
            "criteria_non_degenerate": {
                # The rescue criterion discriminates only if magnitude moves exploration
                # (readiness_ok) AND there is a collapse seed to rescue.
                "C_collapse_seed_rescued": bool(result["rescue_criterion_non_degenerate"]),
            },
            "criteria": [
                {
                    "name": "C_collapse_seed_rescued",
                    "load_bearing": True,
                    "passed": bool(result["all_collapse_rescued"]),
                },
                {
                    "name": "C_no_healthy_regression",
                    "load_bearing": False,
                    "passed": bool(result["no_healthy_regression"]),
                },
            ],
        },
        "metrics": {
            "healthy_seeds": result["healthy_seeds"],
            "collapse_seeds_default": result["collapse_seeds_default"],
            "rescue_scale_by_seed": result["rescue_scale_by_seed"],
            "all_collapse_rescued": result["all_collapse_rescued"],
            "no_healthy_regression": result["no_healthy_regression"],
            "readiness_range": round(result["readiness_range"], 4),
            "readiness_ok": result["readiness_ok"],
            "per_arm_healthy_mean": result["per_arm_healthy_mean"],
            "n_cells": len(result["arm_results"]),
        },
        "arm_results": result["arm_results"],
        "notes": (
            "PRIMARY routing output of failure_autopsy_V3-EXQ-591c_2026-06-11 (confirmed)."
            " Q-043 (registered open_question) magnitude-scale sweep on the 591c-lineage"
            " InfantCurriculumScheduler Phase 0->1 reachability probe. 591c FAILed: arming"
            " MECH-313 noise floor + MECH-314 curiosity at LANDED-DEFAULT magnitudes"
            " (noise_floor_alpha=0.1, curiosity_weight=0.05) did NOT rescue seed 46"
            " (h_pos_mean=0.0375, identical to 591b stack-OFF). This sweep scales both"
            " knobs jointly to 2x/4x/8x and asks: does any magnitude above default rescue"
            " the collapse-prone tail (config-only fix) or does it persist (active Phase-0"
            " exploration-shaping substrate required)? The 1x arm reproduces 591c as the"
            " anchor. The Phase 0->1 threshold + genuine-exploration definition are"
            " UNCHANGED -- the seed-45 gate-permissiveness brittleness is a SEPARATE ARC-046"
            " K-of-N/EMA follow-on, explicitly out of scope; lowering the threshold is NOT"
            " the route. A below-floor readiness range (knobs inert on the healthy-seed"
            " positive control) self-routes substrate_not_ready_requeue, never a substrate"
            " verdict."
        ),
    }

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Result written to: {out_path}", flush=True)
    else:
        print("[dry-run] manifest not written", flush=True)
        summary = {k: v for k, v in manifest.items() if k != "arm_results"}
        print(json.dumps(summary, indent=2), flush=True)

    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _outcome, _out_path = main(dry_run=args.dry_run)

    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
    )
