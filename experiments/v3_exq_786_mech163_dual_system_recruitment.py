"""
V3-EXQ-786 -- MECH-163 dual-system recruitment signature (WALL-INDEPENDENT).

Claims: MECH-163

WHAT MECH-163 ASSERTS. Two goal-directed systems run in parallel: a habit system
(SNc/dorsal-striatum, model-free, cached S-R, no multi-step rollout) sufficient for
approach in PRACTICED contexts, and a hippocampally-planned system (VTA/ventral-striatum
+ PFC, model-based, multi-step rollout) REQUIRED for (1) novel contexts, (2) long-horizon
benefit accumulation, and (3) prosocial planning.

SCOPE -- THIS EXPERIMENT TESTS LEG (1) ONLY. Legs (2) and (3) are OUT OF SCOPE and a PASS
here MUST NOT be read as confirming them:
  * Leg (2) long-horizon benefit accumulation is SUBSTRATE-BLOCKED by ARC-007 STRICT
    (Q-020, 2026-03-16): HippocampalModule generates VALUE-FLAT proposals. There is no
    hippocampal value computation in V3, so the substrate cannot express the very
    quantity that would distinguish the planned system on this leg.
  * Leg (3) prosocial planning has no V3 substrate at all (no social mind).
  * ARC-071 (policy.composition_via_repeated_grounding) -- which MECH-163's own
    claims.yaml note names as the missing planned->habit TRANSITION mechanism -- is NOT
    implemented in ree-v3 (zero hits in CLAUDE.md / ree_core). This experiment therefore
    tests the PRESENCE of differential recruitment, never the transfer dynamics.

WHY A RECRUITMENT DV AND NOT A PERFORMANCE DV (the load-bearing design decision).
The obvious test -- ablate planning, measure the task-performance drop in novel vs
familiar contexts -- is NOT VIABLE on this substrate and was rejected before authoring.
experiments/_lib/capability_eval.py records COMPETENCE_RESOURCE_FLOOR = 1.0 against a
measured all-ON foraging competence of 0.065 / 0.0 / 0.455 resources per episode, 0/3
seeds (failure_autopsy_V3-EXQ-719a_2026-07-08, substrate_ceiling / non_contributory). A
performance interaction measured there is FLOOR-PINNED: it would ask whether ablation
degrades an agent that already cannot act, and would return a vacuous or
non-contributory result. A second defect compounds it: HippocampalConfig.horizon sets the
terrain_prior output width (Linear(hidden_dim, action_object_dim * horizon)), so a
horizon ablation changes the network SHAPE and each arm needs its own from-scratch agent
-- adding a capacity confound on top of the floor.

So this experiment measures DIFFERENTIAL RECRUITMENT instead, which needs no task
competence and no second agent. MECH-163's core functional prediction is that the
model-based system is RECRUITED when the context is novel and the habit path suffices
when it is not. Within a SINGLE trained agent, each candidate trajectory carries
world_states [batch, horizon+1, world_dim], so the agent's OWN scorer can be applied at
two depths:
    full-horizon score  = residue_field.evaluate_trajectory(world_seq)          (all steps)
    first-step score    = residue_field.evaluate_trajectory(world_seq[:, :1, :]) (1 step)
(both lower = better; this mirrors HippocampalModule.score_trajectory, which sums
residue_field.evaluate_trajectory over the whole sequence). RECRUITMENT is then the rate
at which deep rollout REORDERS the candidate set relative to a myopic read of the same
machinery: recruitment = 1 - spearman(full_horizon_scores, first_step_scores).
0 = lookahead changes nothing (habit path suffices); higher = planning is doing work.

  MECH-163 leg (1) predicts: recruitment rate is HIGHER in novel than in familiar
  contexts.

This DV is wall-independent (no reward, no competence, no goal attainment), confound-free
on capacity (ONE agent per seed, one config, both conditions scored on the same weights),
and needs no substrate work.

EVIDENCE DIRECTIONS (both declared, per the diagnostic-descriptions rule):
  * SUPPORTS MECH-163 (leg 1): recruitment rate is reliably higher on novel layouts than
    on practiced ones -- deep rollout changes the chosen candidate more when the context
    is unfamiliar, which is the dual-system division of labour this leg asserts.
  * WEAKENS MECH-163 (leg 1): recruitment rate is equal or LOWER on novel layouts, on a
    substrate that passed both readiness preconditions. That is a genuine negative for
    leg (1): the planned system is engaged no more by novelty than by familiarity, so the
    two systems are not differentially recruited by context novelty as claimed.
  * NEITHER (self-routes substrate_not_ready_requeue, weighting NOTHING): a readiness
    precondition fails -- the familiarity manipulation did not separate the conditions,
    or the cross-candidate score RANGE is degenerate so the ranking (and hence any
    divergence) is arbitrary noise.

FAMILIAR VS NOVEL, MECHANICALLY. CausalGridWorldV2 seeds self._rng ONCE in __init__ and
reset() advances that stream, so layouts differ episode to episode within one env
instance. Re-INSTANTIATING the env at a fixed seed therefore reproduces an identical
first-reset layout every time. FAMILIAR = the layouts the agent practiced on (fixed env
seeds, re-instantiated); NOVEL = held-out env seeds never seen in the practice phase.
This requires ZERO substrate change.

READINESS (P0, both preconditions assert the SAME statistic their criterion routes on):
  * familiarity_separation -- FamiliarityTracker.query() familiarity must be HIGHER on
    familiar layouts than novel by FAMILIARITY_SEP_FLOOR. This is the manipulation check:
    without it, "novel" is a label, not a condition.
  * candidate_score_range -- the load-bearing criterion routes on a RANKING (argmin over
    candidates), which is meaningful only if candidates are separable. So the readiness
    check asserts the cross-candidate score RANGE clears a floor -- deliberately RANGE,
    not mean-abs / magnitude (the V3-EXQ-643 same-statistic lesson: a uniform per-tick
    offset has large magnitude and ~0 range, and a magnitude check would pass while the
    ranking is arbitrary).
Below-floor on either routes to substrate_not_ready_requeue -- NEVER to a substrate
verdict label and never to an evidence direction.

SLEEP: not used (no sleep flags set), so no SLEEP DRIVER line applies.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402

from _lib.arm_fingerprint import reset_all_rng  # noqa: E402
from _lib.goal_pipeline_tier1 import (  # noqa: E402
    ENV_FISHTANK_KWARGS,
    ArmSpec,
    build_config,
    warmup_train,
)
from _lib.robustness_bars import robust_by_sem  # noqa: E402
from _metrics import check_degeneracy  # noqa: E402
from pack_writer import write_flat_manifest  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.hippocampal.curiosity import FamiliarityTracker  # noqa: E402

EXPERIMENT_PURPOSE = "evidence"

EXPERIMENT_TYPE = "v3_exq_786_mech163_dual_system_recruitment"
CLAIM_IDS = ["MECH-163"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# ---------------------------------------------------------------------------
# Pre-registered constants (thresholds fixed here, never derived from the run)
# ---------------------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]

# Layout identity is the env seed; FAMILIAR are practiced, NOVEL are held out.
FAMILIAR_ENV_SEEDS = [1000, 1001, 1002]
NOVEL_ENV_SEEDS = [2000, 2001, 2002]

PRACTICE_EPISODES_PER_LAYOUT = 20          # per familiar layout
STEPS_PER_EPISODE = 120
PROBE_EPISODES_PER_LAYOUT = 4              # per layout, per condition
# Forward-only familiarity accumulation over the PRACTICED layouts. Scaled with the
# run (not hardcoded to 1): the instrument must actually encode the familiar set
# before it can separate it from held-out layouts.
ACCUM_EPISODES_PER_LAYOUT = 6
TOTAL_TRAINING_EPISODES = PRACTICE_EPISODES_PER_LAYOUT * len(FAMILIAR_ENV_SEEDS)

# Load-bearing bar: mean(novel - familiar) - k*SEM > MARGIN.
DIVERGENCE_MARGIN = 0.05
SEM_K = 1.0
SEM_MIN_N = 3

# Readiness floors.
FAMILIARITY_SEP_FLOOR = 0.05
CANDIDATE_SCORE_RANGE_FLOOR = 1e-6

# The canonical cohort env config build_config() is designed against. Do NOT
# hand-roll a subset here: build_config derives the harm-stream encoder input
# widths from these kwargs (harm_history_len, limb_damage_enabled -> harm_obs_a
# width), so a partial dict silently produces a config whose affective-harm
# encoder cannot consume the env's own observation (1x60 into a 17-wide Linear).
ENV_KWARGS: Dict[str, Any] = dict(ENV_FISHTANK_KWARGS)


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------
def _obs_field(obs_dict: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    val = obs_dict.get(key)
    if val is None:
        return None
    val = val.float()
    return val.unsqueeze(0) if val.dim() == 1 else val


def _spearman(a: List[float], b: List[float]) -> Optional[float]:
    """Spearman rank correlation, computed as Pearson over ranks (no scipy dep).

    Returns None when either vector is constant (rank variance 0), which is the
    degenerate case the readiness range-floor is there to exclude.
    """
    n = len(a)
    if n < 2 or len(b) != n:
        return None
    ra = np.argsort(np.argsort(np.asarray(a, dtype=float))).astype(float)
    rb = np.argsort(np.argsort(np.asarray(b, dtype=float))).astype(float)
    if float(np.std(ra)) == 0.0 or float(np.std(rb)) == 0.0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _depth_scores(agent: REEAgent, candidates: List[Any]) -> Tuple[List[float], List[float]]:
    """Score every candidate at FULL horizon and at FIRST STEP ONLY.

    Mirrors HippocampalModule.score_trajectory (which sums
    residue_field.evaluate_trajectory over the sequence). Lower = better.
    Returns ([], []) when any candidate lacks a world-state sequence.
    """
    full: List[float] = []
    first: List[float] = []
    with torch.no_grad():
        for traj in candidates:
            world_seq = traj.get_world_state_sequence()   # [batch, horizon+1, world_dim]
            if world_seq is None or world_seq.shape[1] < 2:
                return [], []
            full.append(float(agent.residue_field.evaluate_trajectory(world_seq)[0].item()))
            first.append(
                float(agent.residue_field.evaluate_trajectory(world_seq[:, :1, :])[0].item())
            )
    return full, first


def _familiarity(tracker: FamiliarityTracker, z_world: torch.Tensor) -> Optional[float]:
    """Query the EXPERIMENT-OWNED familiarity instrument.

    DELIBERATELY NOT agent.hippocampal.familiarity_tracker. That tracker is built
    only when HippocampalConfig.curiosity_weight > 0, and that same flag makes the
    CEM scorer itself novelty-sensitive (score -= curiosity_weight * novelty,
    SD-025). Turning it on to obtain a familiarity readout would make this
    experiment CIRCULAR: the scorer whose reordering is the DV would itself become
    a function of novelty, guaranteeing the predicted effect by construction. So
    the instrument is a standalone tracker, updated on real visits and queried at
    probe time, wired into NOTHING -- the agent's behaviour is bit-identical to
    curiosity_weight=0.0 (the master no-op) with or without it.
    """
    with torch.no_grad():
        fam = tracker.query(z_world)
    if fam is None or fam.numel() == 0:
        return None
    return float(fam.mean().item())


def _probe_layout(
    agent: REEAgent,
    tracker: FamiliarityTracker,
    env_seed: int,
    n_episodes: int,
    update_tracker: bool = False,
) -> Dict[str, Any]:
    """Run probe episodes on ONE layout, measuring recruitment per tick.

    RECRUITMENT = 1 - spearman(full_horizon_scores, first_step_scores) over the
    candidate set: how much multi-step lookahead REORDERS the candidates relative to
    a myopic read of the same machinery. 0 = deep rollout changes nothing (the habit
    path suffices); higher = planning is doing work.

    NOT a top-1 argmin disagreement rate. That measure SATURATES: with
    num_candidates=32 and continuous scores the two argmins almost always differ by
    chance, and the smoke test duly measured 1.0 in nearly every cell (delta pinned
    to ~0, one seed NEGATIVE). A saturated DV cannot express a between-condition
    difference, so the rank correlation -- bounded, unsaturated -- replaces it.

    No reward, no competence, forward-only.
    """
    ticks_scored = 0
    recruitments: List[float] = []
    score_ranges: List[float] = []
    familiarities: List[float] = []

    for _ep in range(n_episodes):
        # Fresh instance at a FIXED seed -> identical layout every time.
        env = CausalGridWorldV2(seed=env_seed, **ENV_KWARGS)
        _flat, obs_dict = env.reset()
        agent.reset()

        for _step in range(STEPS_PER_EPISODE):
            body = _obs_field(obs_dict, "body_state")
            world = _obs_field(obs_dict, "world_state")
            if body is None or world is None:
                break
            latent = agent.sense(
                obs_body=body,
                obs_world=world,
                obs_harm=_obs_field(obs_dict, "harm_obs"),
                obs_harm_a=_obs_field(obs_dict, "harm_obs_a"),
                obs_harm_history=_obs_field(obs_dict, "harm_history"),
            )
            ticks = agent.clock.advance()
            wdim = latent.z_world.shape[-1]
            e1_prior = (
                agent._e1_tick(latent)
                if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            # Only score on a real E3 tick: between ticks generate_trajectories
            # returns the CACHED candidate set (MECH-057a gate), so scoring it
            # again would re-record the same row and pseudo-replicate.
            if ticks.get("e3_tick", False) and candidates and len(candidates) >= 2:
                full, first = _depth_scores(agent, candidates)
                if full and first:
                    rng = float(max(full) - min(full))
                    score_ranges.append(rng)
                    if rng > CANDIDATE_SCORE_RANGE_FLOOR:
                        rho = _spearman(full, first)
                        if rho is not None:
                            ticks_scored += 1
                            recruitments.append(1.0 - rho)
                    fam = _familiarity(tracker, latent.z_world)
                    if fam is not None:
                        familiarities.append(fam)

            # Familiarity is advanced ONLY on the practice pass, and only on real
            # visited states -- never on CEM-internal rollout states (they write no
            # real memory; curiosity.py records that gating as the call-site's job).
            if update_tracker:
                with torch.no_grad():
                    tracker.update(latent.z_world.detach())

            action = agent.select_action(candidates, ticks)
            if action is None or not torch.isfinite(action).all():
                act_idx = int(np.random.randint(0, int(env.action_dim)))
            else:
                act_idx = int(action[0].argmax().item())

            _flat, _harm, done, info, obs_dict = env.step(act_idx)
            with torch.no_grad():
                agent.update_residue(
                    harm_signal=float(info.get("harm_signal", 0.0)) if isinstance(info, dict) else 0.0,
                    world_delta=None,
                    hypothesis_tag=False,
                    owned=True,
                )
            if done:
                break

    return {
        "env_seed": env_seed,
        "ticks_scored": ticks_scored,
        "recruitment_rate": float(np.mean(recruitments)) if recruitments else None,
        "recruitment_sd": float(np.std(recruitments)) if recruitments else None,
        "mean_score_range": float(np.mean(score_ranges)) if score_ranges else 0.0,
        "mean_familiarity": float(np.mean(familiarities)) if familiarities else None,
    }


def _run_seed(seed: int, practice_episodes: int, probe_episodes: int,
              accum_episodes: int) -> Dict[str, Any]:
    """Practice on the FAMILIAR layouts, then probe recruitment on both conditions."""
    print(f"Seed {seed} Condition practice", flush=True)
    reset_all_rng(seed)

    proto_env = CausalGridWorldV2(seed=FAMILIAR_ENV_SEEDS[0], **ENV_KWARGS)
    arm = ArmSpec(arm_id="mech163_recruitment", gap4_operating=False)
    cfg = build_config(proto_env, arm)          # from_dims path -> alpha_world=0.9 (SD-008)
    agent = REEAgent(cfg)

    # Experiment-owned familiarity instrument (see _familiarity docstring for why
    # this is NOT agent.hippocampal.familiarity_tracker). Wired into nothing.
    tracker = FamiliarityTracker(
        world_dim=int(cfg.hippocampal.world_dim),
        ema_alpha=float(cfg.hippocampal.familiarity_ema_alpha),
        bandwidth=float(cfg.hippocampal.familiarity_bandwidth),
    )

    total_eps = practice_episodes * len(FAMILIAR_ENV_SEEDS)
    for env_seed in FAMILIAR_ENV_SEEDS:
        env = CausalGridWorldV2(seed=env_seed, **ENV_KWARGS)
        warmup_train(
            agent,
            env,
            num_episodes=practice_episodes,
            steps_per_episode=STEPS_PER_EPISODE,
            label=f"seed{seed}_layout{env_seed}",
            progress_total_episodes=total_eps,
        )

    # Familiarity accumulation pass: forward-only over the PRACTICED layouts, so the
    # instrument encodes exactly "these are the contexts the agent has visited".
    # Runs after training so it reads the same encoder the probes will use.
    for env_seed in FAMILIAR_ENV_SEEDS:
        _probe_layout(agent, tracker, env_seed, accum_episodes, update_tracker=True)

    per_condition: Dict[str, Any] = {}
    for cond, env_seeds in (("familiar", FAMILIAR_ENV_SEEDS), ("novel", NOVEL_ENV_SEEDS)):
        print(f"Seed {seed} Condition {cond}", flush=True)
        rows = [_probe_layout(agent, tracker, es, probe_episodes) for es in env_seeds]
        rates = [r["recruitment_rate"] for r in rows if r["recruitment_rate"] is not None]
        fams = [r["mean_familiarity"] for r in rows if r["mean_familiarity"] is not None]
        per_condition[cond] = {
            "per_layout": rows,
            "recruitment_rate": float(np.mean(rates)) if rates else None,
            "mean_familiarity": float(np.mean(fams)) if fams else None,
            "mean_score_range": float(np.mean([r["mean_score_range"] for r in rows])),
            "ticks_scored": int(sum(r["ticks_scored"] for r in rows)),
        }

    fam_rate = per_condition["familiar"]["recruitment_rate"]
    nov_rate = per_condition["novel"]["recruitment_rate"]
    delta = (nov_rate - fam_rate) if (fam_rate is not None and nov_rate is not None) else None

    fam_f = per_condition["familiar"]["mean_familiarity"]
    nov_f = per_condition["novel"]["mean_familiarity"]
    fam_sep = (fam_f - nov_f) if (fam_f is not None and nov_f is not None) else None

    print(f"verdict: {'PASS' if (delta is not None and delta > DIVERGENCE_MARGIN) else 'FAIL'}", flush=True)

    return {
        "seed": seed,
        "conditions": per_condition,
        "recruitment_delta": delta,
        "familiarity_separation": fam_sep,
        "min_score_range": float(min(
            per_condition["familiar"]["mean_score_range"],
            per_condition["novel"]["mean_score_range"],
        )),
    }


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------
def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    seeds = SEEDS[:2] if dry_run else SEEDS
    practice = 2 if dry_run else PRACTICE_EPISODES_PER_LAYOUT
    probe = 1 if dry_run else PROBE_EPISODES_PER_LAYOUT
    accum = 2 if dry_run else ACCUM_EPISODES_PER_LAYOUT

    per_seed = [_run_seed(s, practice, probe, accum) for s in seeds]

    deltas = [r["recruitment_delta"] for r in per_seed if r["recruitment_delta"] is not None]
    fam_seps = [r["familiarity_separation"] for r in per_seed if r["familiarity_separation"] is not None]
    score_ranges = [r["min_score_range"] for r in per_seed]

    # --- P0 readiness preconditions (both assert the statistic their criterion routes on)
    mean_fam_sep = float(np.mean(fam_seps)) if fam_seps else 0.0
    mean_range = float(np.mean(score_ranges)) if score_ranges else 0.0
    preconditions = [
        {
            "name": "familiarity_separation",
            "description": (
                "FamiliarityTracker familiarity higher on practiced layouts than held-out "
                "ones -- the manipulation check that novel/familiar are real conditions"
            ),
            "measured": mean_fam_sep,
            "threshold": FAMILIARITY_SEP_FLOOR,
            "direction": "lower",   # FLOOR: met when measured >= threshold
            "control": "practiced vs held-out env seeds under an identical env config",
            "met": bool(mean_fam_sep >= FAMILIARITY_SEP_FLOOR),
        },
        {
            "name": "candidate_score_range_non_degenerate",
            "description": (
                "Cross-candidate RANGE of the full-horizon score (NOT magnitude): the "
                "load-bearing criterion routes on an argmin ranking, which is arbitrary "
                "noise unless candidates are separable"
            ),
            "measured": mean_range,
            "threshold": CANDIDATE_SCORE_RANGE_FLOOR,
            "direction": "lower",   # FLOOR: met when measured >= threshold
            "control": "candidate set on a real E3 tick",
            "met": bool(mean_range >= CANDIDATE_SCORE_RANGE_FLOOR),
        },
    ]
    ready = all(p["met"] for p in preconditions)

    # --- Load-bearing criterion: recruitment higher on novel, robust to its own noise
    bar = robust_by_sem(deltas, margin=DIVERGENCE_MARGIN, k=SEM_K, min_n=SEM_MIN_N)
    c1_passed = bool(ready and bar.get("passes", False))

    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "dry_run": bool(dry_run),
        "per_seed_results": per_seed,
        "recruitment_deltas": deltas,
        "robustness_bar": bar,
        "pre_registered": {
            "divergence_margin": DIVERGENCE_MARGIN,
            "sem_k": SEM_K,
            "sem_min_n": SEM_MIN_N,
            "familiarity_sep_floor": FAMILIARITY_SEP_FLOOR,
            "candidate_score_range_floor": CANDIDATE_SCORE_RANGE_FLOOR,
            "familiar_env_seeds": FAMILIAR_ENV_SEEDS,
            "novel_env_seeds": NOVEL_ENV_SEEDS,
        },
        "criteria": [
            {
                "name": "C1_recruitment_higher_on_novel",
                "load_bearing": True,
                "passed": c1_passed,
            }
        ],
        "scope_note": (
            "Tests MECH-163 leg (1) NOVEL-CONTEXT RECRUITMENT ONLY. Leg (2) long-horizon "
            "benefit accumulation is blocked by ARC-007 STRICT value-flat proposals; leg "
            "(3) prosocial planning has no V3 substrate; ARC-071 (planned->habit transfer) "
            "is unbuilt. A PASS does NOT confirm the full dual-system claim."
        ),
    }

    if not ready:
        # Substrate could not carry the measurement -- weight NOTHING.
        manifest["outcome"] = "FAIL"
        manifest["experiment_purpose"] = "diagnostic"
        manifest["evidence_direction"] = "non_contributory"
        manifest["interpretation"] = {
            "label": "substrate_not_ready_requeue",
            "preconditions": preconditions,
            "criteria_non_degenerate": {"C1_recruitment_higher_on_novel": False},
        }
    else:
        manifest["outcome"] = "PASS" if c1_passed else "FAIL"
        manifest["evidence_direction"] = "supports" if c1_passed else "weakens"
        manifest["interpretation"] = {
            "label": "recruitment_signature_present" if c1_passed else "no_differential_recruitment",
            "preconditions": preconditions,
            "criteria_non_degenerate": {
                "C1_recruitment_higher_on_novel": bool(len(deltas) >= SEM_MIN_N and mean_range > CANDIDATE_SCORE_RANGE_FLOOR)
            },
        }

    manifest.update(check_degeneracy({
        "recruitment_delta": deltas,
        "candidate_score_range": {"values": score_ranges, "floor": CANDIDATE_SCORE_RANGE_FLOOR},
    }))

    manifest["_full_config"] = {
        "env_kwargs": ENV_KWARGS,
        "steps_per_episode": STEPS_PER_EPISODE,
        "practice_episodes_per_layout": practice,
        "probe_episodes_per_layout": probe,
        "accum_episodes_per_layout": accum,
        "total_training_episodes": practice * len(FAMILIAR_ENV_SEEDS),
        "arm_id": "mech163_recruitment",
    }
    manifest["_seeds"] = seeds
    manifest["_started_at"] = t0
    return manifest


def _out_dir() -> Path:
    return (_ROOT.parent / "REE_assembly" / "evidence" / "experiments").resolve()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest = run_experiment(dry_run=args.dry_run)

    # Route through the sanctioned single writer: it stamps the Recording Standard
    # always-core (recording_schema / substrate_hash / machine / machine_class /
    # elapsed_seconds / config / seeds) and enforces the run_id/_v3 invariants.
    full_config = manifest.pop("_full_config")
    seeds_used = manifest.pop("_seeds")
    started_at = manifest.pop("_started_at")
    out_path = write_flat_manifest(
        manifest,
        _out_dir(),
        dry_run=args.dry_run,
        config=full_config,
        seeds=seeds_used,
        script_path=Path(__file__),
        started_at=started_at,
    )

    print(f"outcome: {manifest['outcome']}", flush=True)
    print(f"evidence_direction: {manifest.get('evidence_direction')}", flush=True)
    print(f"manifest: {out_path}", flush=True)

    _raw = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_raw if _raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(out_path),
        dry_run=args.dry_run,
    )
