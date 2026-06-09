"""V3-EXQ-588c -- MECH-189 super-ordinal goal-anchor substrate-readiness validation.

infant_substrate:GAP-11 -- the V3-EXQ-588 successor with a NEW letter (do NOT
re-queue 588). 588 was reviewed non_contributory for MECH-189
(failure_autopsy_V3-EXQ-588_2026-05-19): it measured the within-episode GoalState
attractor (MECH-112 / DEV-NEED-006), NOT the child-phase ContextMemory
super-ordinal write path the claim describes -- which did not exist. The
SuperOrdinalGoalMemory substrate (ree_core/goal.py, use_super_ordinal_goal_anchors)
landed 2026-06-09; this run validates it.

DESIGN (decoupled from the foraging-contact ceiling that blocked 588, per the
autopsy): the child phase FORCES high-salience benefit each step (nursery
forced-feed), so super-ordinal anchor formation is guaranteed and the test
isolates the MECH-189 WRITE+READ substrate from the orthogonal contact-rate
problem.

  CHILD phase (write_enabled): N_child episodes, sense -> forced
    update_z_goal(high benefit, high drive) at each visited z_world context ->
    cross-episode super-ordinal anchors form.
  FREEZE: agent.set_super_ordinal_write_enabled(False).
  ADULT phase: N_adult episodes, fresh sub-floor z_goal (goal_state.reset()),
    sense -> update_z_goal(benefit_exposure=0, drive_level=0) -> READ-only
    seeding from the stored anchors in matching contexts (no benefit pulse, no
    new writes). Measure adult z_goal.norm().

  ARM_ON  : use_super_ordinal_goal_anchors=True  (anchor store active)
  ARM_OFF : use_super_ordinal_goal_anchors=False (no store -> adult z_goal ~ 0)

A trained z_world encoder is NOT required: z_world is a deterministic context
cue (same obs -> same z_world), so adult contexts recur and match child anchors.
The substrate readiness gate is anchor-formation + seeding-firing, not encoder
fidelity.

ACCEPTANCE -- substrate-readiness framing.
  C1 (LOAD-BEARING) DISCRIMINATION: ARM_ON adult median peak z_goal.norm()
     substantially exceeds the per-seed ARM_OFF baseline -- ARM_ON > DISCRIM_FLOOR
     (0.1) AND ARM_ON > ARM_OFF + DISCRIM_MARGIN (0.1) -- in >= 2/3 seeds. This is
     the substrate question: does the MECH-189 WRITE+READ path move adult z_goal
     where the no-store baseline gives ~ 0?
  ADVISORY (reported, NOT gating) DEV-NEED-006 governance gate: fraction of
     ARM_ON seeds whose adult median crosses 0.4 (the 588 / DEV-NEED-006
     threshold). On the untrained-encoder readiness harness the matured-z_goal
     anchor norm ceilings at ~ 0.37 (the forced-feed z_world-EMA asymptote
     ~ 0.9 * ||z_world||), so 0.4 is regime-bound -- a clean cross is a bonus, a
     near-miss with strong discrimination still validates the substrate and
     routes the absolute gate to a trained-encoder evidence successor (NOT a
     substrate failure).
READINESS / non-vacuity preconditions (else substrate_not_ready_requeue):
  - ARM_ON child phase forms anchors (n_occupied > 0).
  - ARM_ON adult seeding fires (n_seeds > 0).
  - ARM_ON adult seeding produces a non-zero z_goal (positive control,
    the same statistic class C1 routes on).
"""
from __future__ import annotations

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell

EXPERIMENT_TYPE = "v3_exq_588c_mech189_super_ordinal_seeding"
EXPERIMENT_PURPOSE = "diagnostic"

# Pre-registered thresholds (defined here, not inferred post-hoc).
DISCRIM_FLOOR = 0.1             # C1 (load-bearing): ARM_ON adult median must clear this
DISCRIM_MARGIN = 0.1           # C1: ARM_ON must exceed the per-seed ARM_OFF baseline by this
ADULT_ZGOAL_GATE = 0.4          # ADVISORY: the DEV-NEED-006 / 588 z_goal.norm governance gate
SEED_PASS_FRACTION = 2.0 / 3.0  # >= 2/3 seeds must pass C1
FORCED_BENEFIT = 0.5            # child forced-feed benefit (salience = 0.5*(1+2*0.9)=1.4 >> 0.5)
FORCED_DRIVE = 0.9


def _build_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        size=8,
        num_hazards=2,
        num_resources=3,
        use_proxy_fields=True,
        seed=seed,
    )


def _build_agent(env: CausalGridWorldV2, arm_on: bool) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        z_goal_enabled=True,
        drive_weight=2.0,
        alpha_world=0.9,  # SD-008
        use_super_ordinal_goal_anchors=arm_on,
        super_ordinal_salience_threshold=0.5,
        super_ordinal_complexity_mode="novelty",
        super_ordinal_complexity_threshold=0.2,
        super_ordinal_merge_similarity=0.8,
        super_ordinal_write_alpha=0.3,
        super_ordinal_seed_below_norm=ADULT_ZGOAL_GATE,
        super_ordinal_seed_match_threshold=0.3,
        super_ordinal_seed_strength=0.2,
    )
    return REEAgent(cfg)


def _step_world_dim(agent: REEAgent) -> int:
    return agent.config.latent.world_dim


def _child_episode(agent: REEAgent, env: CausalGridWorldV2, steps: int) -> None:
    """Forced-feed child episode: forms super-ordinal anchors at visited contexts."""
    _, obs_dict = env.reset()
    agent.reset()  # per-episode reset -- does NOT clear the super-ordinal store
    wd = _step_world_dim(agent)
    for _ in range(steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        latent = agent.sense(obs_body, obs_world)
        ticks = agent.clock.advance()
        e1_prior = (
            agent._e1_tick(latent) if ticks.get("e1_tick")
            else torch.zeros(1, wd, device=agent.device)
        )
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        action = agent.select_action(candidates, ticks)
        action_idx = int(action.argmax(dim=-1).item())
        # FORCED high-salience benefit -> anchor write at this z_world context.
        agent.update_z_goal(benefit_exposure=FORCED_BENEFIT, drive_level=FORCED_DRIVE)
        _, _harm, done, _, obs_dict = env.step(action_idx)
        if done:
            break


def _adult_episode(agent: REEAgent, env: CausalGridWorldV2, steps: int) -> float:
    """Adult episode with fresh sub-floor z_goal: measures READ-only seeding.
    Returns the per-episode peak z_goal.norm()."""
    _, obs_dict = env.reset()
    agent.reset()
    if agent.goal_state is not None:
        agent.goal_state.reset()  # fresh sub-floor z_goal each adult episode
    wd = _step_world_dim(agent)
    peak = 0.0
    for _ in range(steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick")
                else torch.zeros(1, wd, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item())
        # NO benefit pulse -> only the MECH-189 READ (super-ordinal seeding) can
        # lift z_goal. Writes are frozen.
        agent.update_z_goal(benefit_exposure=0.0, drive_level=0.0)
        cur = float(agent.goal_state.goal_norm())
        if cur > peak:
            peak = cur
        _, _harm, done, _, obs_dict = env.step(action_idx)
        if done:
            break
    return peak


def _run_seed_arm(arm_on: bool, seed: int, n_child: int, n_adult: int,
                  steps: int) -> Dict[str, Any]:
    arm_label = "ON" if arm_on else "OFF"
    print(f"Seed {seed} Condition {arm_label}", flush=True)
    full_config = {
        "arm_on": arm_on, "seed": seed, "n_child": n_child,
        "n_adult": n_adult, "steps": steps,
        "forced_benefit": FORCED_BENEFIT, "forced_drive": FORCED_DRIVE,
        "seed_below_norm": ADULT_ZGOAL_GATE,
    }
    with arm_cell(seed, config_slice=full_config, script_path=Path(__file__)) as cell:
        torch.manual_seed(seed)
        env = _build_env(seed)
        agent = _build_agent(env, arm_on)
        som = agent.super_ordinal_goal_memory

        total_eps = n_child + n_adult
        # CHILD phase (write_enabled True by default).
        for ep in range(n_child):
            _child_episode(agent, env, steps)
            if (ep + 1) % 2 == 0 or ep == n_child - 1:
                print(f"  [train] child seed={seed} arm={arm_label} "
                      f"ep {ep + 1}/{total_eps}", flush=True)
        n_occupied = som.n_occupied() if som is not None else 0

        # FREEZE writes for the adult measurement phase.
        agent.set_super_ordinal_write_enabled(False)

        # ADULT phase (READ-only seeding).
        adult_peaks: List[float] = []
        for ep in range(n_adult):
            adult_peaks.append(_adult_episode(agent, env, steps))
            print(f"  [train] adult seed={seed} arm={arm_label} "
                  f"ep {n_child + ep + 1}/{total_eps}", flush=True)
        n_seeds = (som._n_seeds if som is not None else 0)

        adult_peaks_sorted = sorted(adult_peaks)
        m = len(adult_peaks_sorted)
        adult_median = (
            adult_peaks_sorted[m // 2] if m % 2 == 1
            else 0.5 * (adult_peaks_sorted[m // 2 - 1] + adult_peaks_sorted[m // 2])
        ) if m else 0.0

        # Per-cell progress verdict (a rough signal; the load-bearing C1
        # discrimination is computed by pairing arms in run_experiment).
        cell_ok = adult_median > DISCRIM_FLOOR if arm_on else (adult_median < DISCRIM_FLOOR)
        print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)

        row = {
            "arm": arm_label,
            "arm_on": arm_on,
            "seed": seed,
            "child_n_occupied": int(n_occupied),
            "adult_n_seeds": int(n_seeds),
            "adult_peaks": [round(p, 6) for p in adult_peaks],
            "adult_median_zgoal_norm": round(float(adult_median), 6),
        }
        cell.stamp(row)
    return row


def run_experiment(n_child: int, n_adult: int, steps: int, seeds: List[int],
                   dry_run: bool) -> Dict[str, Any]:
    arm_results: List[Dict[str, Any]] = []
    for seed in seeds:
        for arm_on in (False, True):  # OFF baseline first, then ON
            arm_results.append(
                _run_seed_arm(arm_on, seed, n_child, n_adult, steps)
            )

    on_rows = [r for r in arm_results if r["arm_on"]]
    off_rows = [r for r in arm_results if not r["arm_on"]]
    off_by_seed = {r["seed"]: r["adult_median_zgoal_norm"] for r in off_rows}

    # Readiness / non-vacuity (positive control = ARM_ON).
    on_anchors_formed = all(r["child_n_occupied"] > 0 for r in on_rows)
    on_seeding_fired = all(r["adult_n_seeds"] > 0 for r in on_rows)
    on_mean = sum(r["adult_median_zgoal_norm"] for r in on_rows) / max(1, len(on_rows))
    on_seed_positive_control = on_mean  # same statistic class C1 routes on
    readiness_met = (
        on_anchors_formed and on_seeding_fired and on_seed_positive_control > 1e-3
    )

    # C1 (LOAD-BEARING) DISCRIMINATION, per seed: ARM_ON adult median clears the
    # floor AND exceeds the per-seed ARM_OFF baseline by the margin.
    def _seed_discriminates(on_row) -> bool:
        off_v = off_by_seed.get(on_row["seed"], 0.0)
        on_v = on_row["adult_median_zgoal_norm"]
        return (on_v > DISCRIM_FLOOR) and (on_v > off_v + DISCRIM_MARGIN)

    n_on_pass = sum(1 for r in on_rows if _seed_discriminates(r))
    frac_on_pass = n_on_pass / float(max(1, len(on_rows)))
    off_max_adult = max((r["adult_median_zgoal_norm"] for r in off_rows), default=0.0)
    c1_pass = frac_on_pass >= SEED_PASS_FRACTION

    # Non-degeneracy: ARM_ON and ARM_OFF adult z_goal must genuinely differ
    # (the seeding actually moved z_goal vs the no-store baseline).
    c1_non_degenerate = (on_mean - off_max_adult) > DISCRIM_MARGIN

    # ADVISORY: DEV-NEED-006 / 588 governance gate (0.4) crossing -- reported,
    # NOT gating (anchor-norm regime-bound on the untrained-encoder harness).
    n_on_cross_gate = sum(
        1 for r in on_rows if r["adult_median_zgoal_norm"] > ADULT_ZGOAL_GATE
    )
    frac_on_cross_gate = n_on_cross_gate / float(max(1, len(on_rows)))

    if not readiness_met:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        direction = "unknown"
    elif c1_pass:
        outcome = "PASS"
        label = "super_ordinal_seeding_validated"
        direction = "supports"
    else:
        outcome = "FAIL"
        label = "super_ordinal_seeding_insufficient"
        direction = "weakens"

    result: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": ["MECH-189"],
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes": "v3_exq_588_isef002_transient_benefit_zgoal_seeding",
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "outcome": outcome,
        "dry_run": dry_run,
        "config": {
            "n_child": n_child, "n_adult": n_adult, "steps": steps,
            "seeds": seeds, "forced_benefit": FORCED_BENEFIT,
            "forced_drive": FORCED_DRIVE,
            "discrim_floor": DISCRIM_FLOOR,
            "discrim_margin": DISCRIM_MARGIN,
            "devneed006_gate_advisory": ADULT_ZGOAL_GATE,
            "seed_pass_fraction": SEED_PASS_FRACTION,
        },
        "metrics": {
            "frac_on_pass_c1_discrimination": round(frac_on_pass, 4),
            "n_on_pass_c1_discrimination": n_on_pass,
            "on_mean_adult_zgoal_norm": round(on_mean, 6),
            "off_max_adult_zgoal_norm": round(off_max_adult, 6),
            "frac_on_cross_devneed006_gate_advisory": round(frac_on_cross_gate, 4),
            "n_on_cross_devneed006_gate_advisory": n_on_cross_gate,
            "devneed006_gate_threshold": ADULT_ZGOAL_GATE,
            "on_anchors_formed": on_anchors_formed,
            "on_seeding_fired": on_seeding_fired,
        },
        "arm_results": arm_results,
        "interpretation": {
            "label": label,
            "preconditions": [
                {
                    "name": "arm_on_child_anchors_formed",
                    "description": "ARM_ON child phase wrote >=1 super-ordinal anchor (n_occupied>0).",
                    "measured": int(min((r["child_n_occupied"] for r in on_rows), default=0)),
                    "threshold": 1,
                    "met": bool(on_anchors_formed),
                },
                {
                    "name": "arm_on_adult_seeding_fired",
                    "description": "ARM_ON adult phase fired the READ seeding path (n_seeds>0).",
                    "measured": int(min((r["adult_n_seeds"] for r in on_rows), default=0)),
                    "threshold": 1,
                    "met": bool(on_seeding_fired),
                },
                {
                    "name": "arm_on_adult_zgoal_positive_control",
                    "description": "Readiness: ARM_ON adult seeding produces a non-zero z_goal.norm "
                                   "(same statistic the load-bearing C1 routes on).",
                    "measured": round(float(on_seed_positive_control), 6),
                    "threshold": 1e-3,
                    "control": "ARM_ON adult episodes seed z_goal from childhood anchors with no benefit pulse",
                    "met": bool(on_seed_positive_control > 1e-3),
                },
            ],
            "criteria_non_degenerate": {
                "C1": bool(c1_non_degenerate),
            },
            "criteria": [
                {
                    "name": "C1_arm_on_adult_zgoal_discriminates_over_arm_off",
                    "load_bearing": True,
                    "passed": bool(c1_pass),
                },
                {
                    "name": "ADVISORY_arm_on_crosses_devneed006_gate_0p4",
                    "load_bearing": False,
                    "passed": bool(frac_on_cross_gate >= SEED_PASS_FRACTION),
                },
            ],
            "evidence_direction": direction,
        },
        "evidence_direction": direction,
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        n_child, n_adult, steps, seeds = 2, 2, 20, [42]
    else:
        n_child, n_adult, steps, seeds = 8, 6, 100, [42, 43, 44]

    result = run_experiment(n_child, n_adult, steps, seeds, args.dry_run)

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{result['run_id']}.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(f"outcome: {result['outcome']}", flush=True)
    print(f"label: {result['interpretation']['label']}", flush=True)
    print(f"C1_discrimination frac_on_pass: "
          f"{result['metrics']['frac_on_pass_c1_discrimination']}", flush=True)
    print(f"on_mean_adult_zgoal: {result['metrics']['on_mean_adult_zgoal_norm']} "
          f"off_max_adult_zgoal: {result['metrics']['off_max_adult_zgoal_norm']}", flush=True)
    print(f"ADVISORY devneed006_gate(0.4) frac_on_cross: "
          f"{result['metrics']['frac_on_cross_devneed006_gate_advisory']}", flush=True)
    print(f"wrote: {out_path}", flush=True)
    return result, out_path


if __name__ == "__main__":
    _result, _out_path = main()
    _outcome_raw = str(_result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
    )
