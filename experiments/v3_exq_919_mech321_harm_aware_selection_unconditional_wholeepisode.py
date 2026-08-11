#!/opt/local/bin/python3
"""V3-EXQ-919 -- MECH-321 HARM-AWARE SELECTION, UNCONDITIONAL WHOLE-EPISODE
HARM RATE. Redesigned measurement for the same question four prior
generations (V3-EXQ-844/867/867a/867b) each failed to test decisively: does
SD-hazard-aware-policy-decomposition's harm-valence-weighted retiling
selection reduce realised TASK-LEVEL HARM relative to harm-blind selection,
at the same abort-mechanism-ON baseline?

NEW EXQ NUMBER, NOT 867c. Per CLAUDE.md's EXQ convention ("a new letter is
for 'the scientific question is unchanged but the implementation was
wrong'; a new number is for 'the mechanism under test changed, OR the
experimental design is substantially different'") and per the design doc's
own explicit recommendation (section 7), this changes THREE design axes at
once relative to 867b: the unit of comparison, the DV, and the sampling
strategy. `supersedes` is deliberately OMITTED -- 867b is not superseded, it
is a different measurement of the same question, and its screen-soundness
instrumentation finding retains standalone value.

Full design: evidence/planning/mech321_harm_aware_selection_measurement_
redesign_staged_2026-08-08.md (design work; nothing built there). This
script is the build/queue step per that doc's section 9 hand-off.

WHY THE PRIOR SCREEN-THEN-MATCH FAMILY IS RETIRED (design doc sections 2, 3)

867b's own manifest shows the screen-soundness "violation" it flagged is an
INSTRUMENTATION ARTIFACT, not evidence the manipulation perturbs
decomposition: the violation appears in the OFF arm, where the manipulation
is provably inert, and is explained by an RNG-reset asymmetry -- 867b's
screen cells called `_run_cell` directly with NO reset, while its
measurement cells entered through `arm_cell` (full RNG reset), so the two
phases measured DIFFERENT AGENTS. At the full measurement schedule the
matching actually held PERFECTLY (0 on_only, 0 off_only of 10 measured
seeds) -- the pool was never as exhausted as it looked, either.

The deeper defect, independent of the above: the manipulation is
UNCONDITIONALLY active (ON-arm `decomp_n_harm_bias_nonzero` was 270-1005 on
EVERY one of 867b's 10 measured seeds, including all six seeds used as
"neither-decompose" negative controls -- `action_sequences_identical: false`
on all six, so they were never actually controls). Conditioning the C1
statistic on the rare, downstream, mid-execution-decomposition event
discarded 6 of 10 experimental units on 867b's own run -- and discarded the
LOWER-variance ones (SD 0.053 vs 0.239 for the retained tier).

THE FIX, IN THREE PARTS (design doc section 5)

(1) NO SCREEN, NO TIERING. Every measured seed is a unit. There is no
    matched-pair machinery, so there is no matching assumption to verify or
    falsify -- disposing of the entire defect class in one move. EVERY cell
    in this driver -- all 40 measurement seeds x 2 arms, AND all 4 A-A
    control seeds x 2 replicates -- enters through `arm_cell(seed, ...)`.
    There is NO bare `_run_cell` call anywhere in this file (contrast 867b,
    whose now-deleted `_screen_pool` was the one bare-call site).
(2) UNCONDITIONAL WHOLE-EPISODE DV. Per seed, paired:
    `mean_harm_signal(ON) - mean_harm_signal(OFF)` over the WHOLE run, all
    ticks -- positive = ON less harmful. `_run_cell` already computes this
    (`mean_harm_signal`) as a whole-run mean; no new instrumentation is
    needed and no post-hoc divergence-tick windowing is in the gating path.
    Pre-registered secondary (reported, non-gating): the same delta
    restricted to fresh e3-selection ticks, over the WHOLE run (not
    windowed from any divergence tick) -- lets this run cross-read against
    844/867/867a/867b's fresh-tick convention.
(3) n=40 SEEDS, NO EXCLUSION. Fixed at authoring time as a literal tuple,
    taken in screen order from 867b's own 48-candidate pool (no screening
    performed -- this design does not need decomposition to occur at all).
    The guard is a HARD `n_seeds >= 40` on measured cells, which can never
    be softened by any observed quantity because no unit is EVER excluded.

A-A NULL CONTROL (design doc section 5.7) -- discharges the matching-
validity precondition BY CONSTRUCTION rather than by post-hoc verification,
and is the direct fix for 867b's actual defect. 4 additional seeds, each run
TWICE as ARM_SELECTION_OFF vs ARM_SELECTION_OFF: identical config slice,
identical seed, both entered through `arm_cell` (full RNG reset). Because
`arm_cell.__enter__` resets ALL RNG and the config slices are identical, the
two replicate cells MUST be bit-identical: action_sequence equal, delta
EXACTLY 0.0. A nonzero delta on ANY control seed means the measurement path
carries an uncontrolled source of variation and the run is VOID --
`non_degenerate: false`, no C1 reading emitted, label
`aa_control_uncontrolled_variation_run_void`.

ARMS, SCHEDULE, BAR -- ALL UNCHANGED FROM 867a/867b. `ARM_SELECTION_OFF` /
`ARM_SELECTION_ON`: hazard-tuned env overlay + abort mechanism ON + the
three affective/sensory-harm stream flags in BOTH arms (preconditions, not
the manipulation); only `decomposition_use_harm_aware_selection` differs,
at the SD doc's recommended defaults (unchanged constants below).
EPISODES=12, STEPS_PER_EPISODE=60 (unchanged -- a shorter schedule is
statistically superior per the design doc's variance decomposition, section
4, but would change how trained the agent is at measurement time and
confound comparability against all four prior generations; that tradeoff is
recorded as the FALLBACK, not taken here). EFFECT_SIZE_K_SIGMA=1.0,
REL_IMPROVEMENT_FLOOR=0.0, carried verbatim -- this redesign repairs the
MEASUREMENT, not the criterion.

READINESS (P0) -- STRENGTHENED to per-cell, not per-run. `harm_bias_engages`
must hold on EVERY ON-arm measurement cell (not just in aggregate -- 867b
satisfied this only in aggregate while individual cells varied 270-1005
fires) and `decomp_n_harm_bias_nonzero` must be EXACTLY 0 on every OFF-arm
cell (the DV-symmetry declaration below: harm-aware selection must be a
bit-identical no-op in the arm carrying no manipulation).

COVARIATES -- reported, NEVER gating: per-seed `decomp_n_decomposed_
midexec` (both arms), `decomp_n_harm_bias_nonzero`, `decomp_n_harm_
override_fires`, `max_z_harm_a_norm`, `multi_action_commits`, and a
both/on_only/off_only/neither tier label (purely descriptive here -- this
design's DV does not condition on it). Pre-registered secondary (design doc
section 5.8, reported, non-gating): Spearman rank correlation between
ON-arm engagement (`decomp_n_harm_bias_nonzero`) and the per-seed delta,
using the shared `experiments._lib.stats.spearman` helper (average-ranked,
guards against the constant-input degeneracy bug SD-081 fixed).

C2 (non-load-bearing, unchanged in spirit from 844/867/867a/867b): forward-
PE lower in ON, now computed on the same unconditional whole-run basis
(`fwd_pe_all_mean`, already produced by `_run_cell` per cell).

DV-SYMMETRY DECLARATION (per this skill's mandatory per-arm check). DV =
mean per-tick environment harm signal over the WHOLE run (a set-aggregate;
symmetry group = permutation of the ticks it averages over). The
manipulation is NOT invariant under that symmetry: harm-aware selection
applies a PER-LEAF penalty (`harm_bias`) plus a categorical per-leaf
override (`select_harm_aware_leaves`) across retiling candidates at
selection time -- it changes WHICH actions are taken (and hence which harm
values are observed at each tick), not a uniform relabelling of a fixed set
of ticks. Identical reasoning for both arms: the OFF arm carries no
manipulation at all, so its per-tick harm sequence is the harm-blind
baseline by construction, never a symmetry-fixed artifact of the DV itself.

GOV-REUSE-1. Decisive readout = unconditional whole-episode paired per-seed
`mean_harm_signal` delta over n=40 seeds. Checked via
`reanalysis_query.py query --readout mean_harm_signal --claim MECH-321`:
every one of the 6 prior MECH-321 manifests carrying this readout
(844/867/867a/867b + the two policy-decomposition diagnostics) has a
DISTINCT top-level `substrate_hash` -- no compatible group exists, so
nothing is derivable by reprocessing banked cells; each generation's harm
signal was measured against a substrate no other run shares. Also confirmed
`ree_core/**` carries 18 commits since 867b's 2026-08-04 run (git log
--since), so even a same-lineage arm-reuse attempt against 867b's OFF mint
would refuse on `substrate_hash` -- `try_reuse_cell` is therefore OMITTED
entirely rather than left in as dead code (867a and 867b made the identical
call for the identical reason against their own predecessors). This run's
own ARM_SELECTION_OFF measurement cells ARE minted reuse-eligible
(`include_driver_script_in_hash=False`) as the current canonical baseline
for this lineage, exactly as 867a/867b did for theirs.

RE-DERIVE BRAKE (Step 2.5b): does not apply. MECH-321 has zero autopsies
scored `substrate_ceiling` in this lineage -- all four priors scored
`non_contributory`/`measurement_test_design_defect` or `environment_
adequacy_defect`, categories the brake's own instrument-defect carve-out
excludes (the substrate itself has never ceilinged; every prior failure was
a measurement-instrument defect, which is exactly what this redesign fixes).

SLEEP DRIVER: not applicable -- no sleep phase entered in this run.

Z_GOAL: deliberately inert, carried over verbatim from 844/867/867a/867b
for the identical reason (`REEConfig.from_dims(z_goal_enabled=...)`
defaults False).
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.stats import spearman  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
import experiments._lib.baselines.sd084_midexec_reachability as baselines  # noqa: E402

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.policy import ChunkedPrimitive, ChunkState  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_919_mech321_harm_aware_selection_unconditional_wholeepisode"
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["MECH-321"]
QUEUE_ID = "V3-EXQ-919"
# supersedes deliberately OMITTED -- see module docstring "NEW EXQ NUMBER" section.

ARM_OFF = "ARM_SELECTION_OFF"
ARM_ON = "ARM_SELECTION_ON"
ARMS = (ARM_OFF, ARM_ON)

# --- Harm-aware-selection parameters. VERBATIM from 844/867/867a/867b -- the
# manipulation does not move in this redesign; only the measurement does. ---
HARM_BIAS_GAIN = 0.1
HARM_BIAS_SCALE = 0.1
HARM_THREAT_FLOOR = 0.1
HARM_THREAT_REF = 0.5
HARM_OVERRIDE_W_THRESHOLD = 0.9

# CONFIG_SLICE_DECLARATION_EXEMPT: validate_experiments.py's static resolver
# cannot trace `_config_slice(arm_id, episodes)`'s conditional dispatch to
# `_on_selection_config_slice()`, so it flags these five constants as
# "omitted." They are NOT omitted -- `_on_selection_config_slice()` declares
# each one under its REEConfig-mirror key, which is what `_arm_flags()` /
# `REEConfig.from_dims` actually reads downstream (identical reasoning to
# 867a's/867b's own exemption note).

# --- Pre-registered thresholds. Constants, never derived from this run's own
# statistics (mirrors 844/867/867a/867b convention). ---
PE_VARIANCE_FLOOR = 1e-12
PE_SANITY_CEIL = 1e6
HARM_BIAS_ENGAGE_FLOOR = 0.0     # ON-arm per-cell floor: must exceed this
HARM_BIAS_INERT_CEIL = 1.0      # OFF-arm per-cell ceiling: must stay below this
                                  # (integer count field -- < 1.0 means exactly 0)

# --- C1's bar. UNCHANGED from 844/867/867a/867b. Do not move these: this
# redesign repairs the SAMPLE and the DV, not the criterion. ---
EFFECT_SIZE_K_SIGMA = 1.0
REL_IMPROVEMENT_FLOOR = 0.0

# --- The seed-count floor. HARD, and unlike 867b's MIN_DECOMPOSING_SEEDS this
# one can never be softened by any observed quantity, because no unit is EVER
# excluded from this design (design doc section 5.1/5.6). ---
MIN_SEEDS = 40

# Pre-registered measurement-seed list -- the first 40 of 867b's own
# 48-candidate CANDIDATE_SEEDS pool, taken VERBATIM in the same order (which
# already deliberately excludes seed 44 -- CLAUDE.md's recurring per-seed
# early-episode-death instability). No screening is performed: this design's
# DV does not require decomposition to occur, so there is no reason to order
# by expected activity as 867b did for its screen.
MEASUREMENT_SEEDS: Tuple[int, ...] = (
    11, 23, 47, 71, 3, 29, 89, 97, 17, 53,
    5, 7, 13, 19, 31, 37, 41, 43, 59, 61,
    67, 73, 79, 83, 101, 103, 107, 109, 113, 127,
    2, 6, 8, 12, 14, 18, 22, 26, 33, 39,
)
assert len(MEASUREMENT_SEEDS) == MIN_SEEDS
assert len(set(MEASUREMENT_SEEDS)) == MIN_SEEDS  # no duplicates

# A-A null control seeds -- the next 4 of the same pool (867b's remaining,
# unscreened candidates), disjoint from MEASUREMENT_SEEDS.
AA_CONTROL_SEEDS: Tuple[int, ...] = (51, 57, 63, 69)
assert not (set(AA_CONTROL_SEEDS) & set(MEASUREMENT_SEEDS))

_ZGOAL = ZGoalStreamAccumulator()


# ---------------------------------------------------------------------------
# Config construction -- REUSED VERBATIM (in shape) from 867a/867b. Sound
# bodies; the defect those runs carried was in HOW _run_cell was CALLED from
# the (here nonexistent) screen, never in these functions.
# ---------------------------------------------------------------------------
def _off_selection_config_slice(episodes: int) -> Dict[str, Any]:
    """ARM_SELECTION_OFF's fingerprint config slice.

    `episodes` is a required argument (not a module constant) so a --dry-run
    cell's reduced schedule gets its own distinct fingerprint rather than
    colliding with a real-run cell's.
    """
    slice_: Dict[str, Any] = {
        "env": dict(baselines.HAZARD_TUNED_ENV_OVERLAY),
        "env_seeded_per_cell": baselines.ENV_SEEDED_PER_CELL,
        "schedule": {
            "episodes": int(episodes),
            "steps_per_episode": baselines.STEPS_PER_EPISODE,
        },
        "self_dim": baselines.SELF_DIM,
        "world_dim": baselines.WORLD_DIM,
        "seeded_chunk_sequence": list(baselines.SEEDED_CHUNK_SEQUENCE),
        "seeded_chunk_depth": baselines.SEEDED_CHUNK_DEPTH,
        "seeded_chunk_selection_weight": baselines.SEEDED_CHUNK_SELECTION_WEIGHT,
    }
    slice_.update(baselines.on_arm_flags())  # abort mechanism ON (both arms)
    slice_.update(baselines.HAZARD_TUNED_STREAM_FLAGS)
    return slice_


def _on_selection_config_slice(episodes: int) -> Dict[str, Any]:
    slice_ = _off_selection_config_slice(episodes)
    slice_.update({
        "decomposition_use_harm_aware_selection": True,
        "decomposition_harm_bias_gain": HARM_BIAS_GAIN,
        "decomposition_harm_bias_scale": HARM_BIAS_SCALE,
        "decomposition_harm_threat_floor": HARM_THREAT_FLOOR,
        "decomposition_harm_threat_ref": HARM_THREAT_REF,
        "decomposition_harm_override_w_threshold": HARM_OVERRIDE_W_THRESHOLD,
    })
    return slice_


def _arm_flags(arm_id: str) -> Dict[str, Any]:
    flags = dict(baselines.on_arm_flags())  # abort mechanism ON in BOTH arms
    flags.update(baselines.HAZARD_TUNED_STREAM_FLAGS)  # precondition, BOTH arms
    if arm_id == ARM_ON:
        flags.update({
            "decomposition_use_harm_aware_selection": True,
            "decomposition_harm_bias_gain": HARM_BIAS_GAIN,
            "decomposition_harm_bias_scale": HARM_BIAS_SCALE,
            "decomposition_harm_threat_floor": HARM_THREAT_FLOOR,
            "decomposition_harm_threat_ref": HARM_THREAT_REF,
            "decomposition_harm_override_w_threshold": HARM_OVERRIDE_W_THRESHOLD,
        })
    return flags


def _config_slice(arm_id: str, episodes: int) -> Dict[str, Any]:
    return (_off_selection_config_slice(episodes) if arm_id == ARM_OFF
            else _on_selection_config_slice(episodes))


def _build(seed: int, arm_id: str) -> Tuple[CausalGridWorldV2, REEAgent, Dict[str, Any]]:
    env = CausalGridWorldV2(**baselines.env_kwargs_hazard_tuned(seed))
    env.reset()
    flags = _arm_flags(arm_id)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=baselines.SELF_DIM,
        world_dim=baselines.WORLD_DIM,
        reafference_action_dim=env.action_dim,
        **flags,
    )
    agent = REEAgent(cfg)
    return env, agent, flags


def _register_chunk(agent: REEAgent) -> None:
    agent.policy_chunking.library.register(
        ChunkedPrimitive(
            sequence=baselines.SEEDED_CHUNK_SEQUENCE,
            depth=baselines.SEEDED_CHUNK_DEPTH,
            state=ChunkState.CRYSTALLISED,
            selection_weight=baselines.SEEDED_CHUNK_SELECTION_WEIGHT,
        )
    )


# ---------------------------------------------------------------------------
# One cell -- REUSED VERBATIM (body) from 867a/867b. Every field this design
# needs is already produced here: `mean_harm_signal` is the unconditional
# whole-run mean (the new primary DV input), `fwd_pe_all_mean` is the
# unconditional whole-run forward-PE mean (C2's input), and `per_tick_harm` +
# `e3_tick_flags` are what the fresh-only secondary DV is derived from below
# (no windowing/divergence-tick machinery is used anywhere in this file).
# ---------------------------------------------------------------------------
def _run_cell(seed: int, arm_id: str, episodes: int, steps: int,
              quiet: bool = False) -> Dict[str, Any]:
    env, agent, flags = _build(seed, arm_id)
    _register_chunk(agent)
    world_dim = agent.config.latent.world_dim

    n_ticks = 0
    multi_action_commits = 0
    actions: List[int] = []
    forward_pe_ticks: List[Optional[float]] = []
    harm_ticks: List[float] = []
    e3_tick_flags: List[bool] = []
    max_z_harm_a_norm = 0.0  # diagnostic only -- not itself gating

    if not quiet:
        print(f"Seed {seed} Condition {arm_id}", flush=True)

    for ep in range(episodes):
        _, obs = env.reset()
        agent.reset()
        if not agent.policy_chunking.library.all_chunks():
            _register_chunk(agent)

        for _ in range(steps):
            latent = agent.sense(
                obs["body_state"], obs["world_state"],
                obs_harm=obs.get("harm_obs"),
                obs_harm_a=obs.get("harm_obs_a"),
            )
            if getattr(latent, "z_harm_a", None) is not None:
                n = float(latent.z_harm_a.detach().norm(dim=-1).mean().item())
                if n > max_z_harm_a_norm:
                    max_z_harm_a_norm = n
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent)
                if ticks.get("e1_tick")
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            e3_tick_flags.append(bool(ticks.get("e3_tick")))
            body = obs["body_state"]
            agent.update_z_goal(
                benefit_exposure=0.0,
                drive_level=REEAgent.compute_drive_level(body),
            )
            action = agent.select_action(candidates, ticks)

            committed = agent.e3._committed_trajectory
            if committed is not None:
                meta = committed.metadata or {}
                seq_len = len(meta.get("chunk_sequence", ()))
                if seq_len > 1:
                    multi_action_commits += 1

            a_int = int(action.argmax(dim=-1).item())
            actions.append(a_int)
            _flat, harm, _done, _info, obs = env.step(a_int)
            harm_ticks.append(float(harm))
            metrics = agent.update_residue(harm)
            pe_raw = metrics.get("e3_prediction_error")
            if pe_raw is not None:
                pe = float(pe_raw.detach()) if torch.is_tensor(pe_raw) else float(pe_raw)
                forward_pe_ticks.append(pe if math.isfinite(pe) else None)
            else:
                forward_pe_ticks.append(None)
            n_ticks += 1

        if not quiet:
            print(
                f"  [train] rollout seed={seed} arm={arm_id} ep {ep + 1}/{episodes} "
                f"ticks={n_ticks} multi_commits={multi_action_commits}",
                flush=True,
            )

    _ZGOAL.observe(agent)

    state = agent.get_policy_decomposition_state()
    row: Dict[str, Any] = {
        "arm_id": arm_id,
        "seed": int(seed),
        "n_ticks": n_ticks,
        "episodes": episodes,
        "steps_per_episode": steps,
        "multi_action_commits": multi_action_commits,
        "decomp_n_evaluated_midexec": int(state.get("decomp_n_evaluated_midexec", 0)),
        "decomp_n_decomposed_midexec": int(state.get("decomp_n_decomposed_midexec", 0)),
        "decomp_n_evaluated_precommit": int(state.get("decomp_n_evaluated_precommit", 0)),
        "decomp_n_decomposed_precommit": int(state.get("decomp_n_decomposed_precommit", 0)),
        "decomp_n_marked_unreliable": int(state.get("decomp_n_marked_unreliable", 0)),
        "decomp_n_vs_trigger": int(state.get("decomp_n_vs_trigger", 0)),
        "decomp_n_boundary_fires": int(state.get("decomp_n_boundary_fires", 0)),
        "decomp_n_harm_bias_nonzero": int(state.get("decomp_n_harm_bias_nonzero", 0)),
        "decomp_n_harm_override_fires": int(state.get("decomp_n_harm_override_fires", 0)),
        "max_z_harm_a_norm": max_z_harm_a_norm,
        "action_sequence": actions,
        "per_tick_forward_pe": forward_pe_ticks,
        "per_tick_harm": harm_ticks,
        "e3_tick_flags": e3_tick_flags,
        "n_fresh_select": sum(e3_tick_flags),
        "n_latched": len(e3_tick_flags) - sum(e3_tick_flags),
        "fresh_select_yield": round(
            sum(e3_tick_flags) / max(1, len(e3_tick_flags)), 6),
        "fwd_pe_all_mean": (
            statistics.fmean(v for v in forward_pe_ticks if v is not None)
            if any(v is not None for v in forward_pe_ticks) else None),
        "fwd_pe_all_var": (
            statistics.pvariance(v for v in forward_pe_ticks if v is not None)
            if sum(1 for v in forward_pe_ticks if v is not None) > 1 else 0.0),
        # UNCONDITIONAL WHOLE-EPISODE MEAN -- this IS the new primary DV's
        # per-cell input; no windowing is applied anywhere in this script.
        "mean_harm_signal": (
            statistics.fmean(harm_ticks) if harm_ticks else 0.0),
        "arm_flags": dict(flags),
    }

    # This design's DV does not require decomposition to occur (unlike
    # 867b's cell_pass, which gated on multi_action_commits/midexec being
    # nonzero) -- so cell_pass is a pure completion check, not a signal of
    # decomposition activity.
    cell_pass = bool(n_ticks == episodes * steps)
    row["cell_pass"] = cell_pass
    if not quiet:
        print(f"verdict: {'PASS' if cell_pass else 'FAIL'}", flush=True)
    return row


# ---------------------------------------------------------------------------
# Reducers -- min/max over an arm's cells, tolerant of a missing key.
# ---------------------------------------------------------------------------
def _best_cell(rows: List[Dict[str, Any]], key: str) -> float:
    """Max over cells. Used where the precondition wants the BEST case."""
    return float(max((r[key] for r in rows), default=0.0))


def _worst_cell(rows: List[Dict[str, Any]], key: str) -> float:
    """Min over cells. Used where the precondition wants the WORST case."""
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(min(vals)) if vals else 0.0


def _worst_cell_max(rows: List[Dict[str, Any]], key: str) -> float:
    """Max over cells, tolerant of None. Used for an upper-bound sanity check."""
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(max(vals)) if vals else 0.0


def _mean_fresh_only(values: List[Optional[float]],
                     flags: List[bool]) -> Tuple[Optional[float], int]:
    """Whole-run mean restricted to fresh e3-selection ticks. NOT windowed --
    the pre-registered secondary DV (design doc section 5.2) is the same
    fresh-tick restriction 844/867/867a/867b used, but over the WHOLE run
    rather than from any divergence tick (there is no divergence tick concept
    in this design)."""
    picked = [float(v) for i, v in enumerate(values)
              if v is not None and i < len(flags) and flags[i]]
    return (statistics.fmean(picked) if picked else None), len(picked)


# ---------------------------------------------------------------------------
# Preconditions -- STRENGTHENED to per-cell (every cell of the arm), not
# per-run aggregate (design doc section 5.7 / this skill's readiness-gate
# convention: "the readiness precondition's measured MUST be the SAME
# statistic the load-bearing criterion routes on"). The load-bearing DV here
# is a per-seed, per-cell whole-episode mean, so the readiness gate checks
# EVERY cell, not just the best one.
# ---------------------------------------------------------------------------
def _arm_context(arm_id: str) -> Dict[str, Any]:
    return {"id": arm_id, "arm_id": arm_id, "harm_aware_on": arm_id == ARM_ON}


def _precondition_specs() -> List[PreconditionSpec]:
    return [
        PreconditionSpec(
            name="harm_bias_engages_every_cell",
            description=(
                "ARM_SELECTION_ON's Stage-1 graded harm bias engages on "
                "EVERY measurement cell (decomp_n_harm_bias_nonzero > 0), "
                "asserted per-cell rather than in aggregate -- 867b "
                "satisfied this only in aggregate while individual cells "
                "varied 270-1005 fires. EXISTENTIAL-PER-CELL, ON arm only."
            ),
            control="WORST (minimum) ON-arm measurement cell",
            threshold=HARM_BIAS_ENGAGE_FLOOR,
            direction="lower",
            kind="readiness",
            applies_to=lambda ctx: ctx["arm_id"] == ARM_ON,
        ),
        PreconditionSpec(
            name="harm_bias_inert_every_cell",
            description=(
                "ARM_SELECTION_OFF carries no manipulation at all: "
                "decomp_n_harm_bias_nonzero must be EXACTLY 0 on every OFF "
                "measurement cell -- the DV-symmetry declaration's "
                "requirement that harm-aware selection is a bit-identical "
                "no-op in the arm carrying no manipulation. "
                "EXISTENTIAL-PER-CELL, OFF arm only."
            ),
            control="WORST (maximum) OFF-arm measurement cell",
            threshold=HARM_BIAS_INERT_CEIL,
            direction="upper",
            kind="readiness",
            applies_to=lambda ctx: ctx["arm_id"] == ARM_OFF,
        ),
        PreconditionSpec(
            name="off_forward_pe_varies",
            description=(
                "OFF-arm positive control: committed-trajectory forward PE "
                "must have non-zero variance across the whole run. Mirrors "
                "816/839/844/867/867a/867b's identical precondition."
            ),
            control="WORST OFF-arm cell (min variance)",
            threshold=PE_VARIANCE_FLOOR,
            direction="lower",
            kind="readiness",
            applies_to=lambda ctx: ctx["arm_id"] == ARM_OFF,
        ),
        PreconditionSpec(
            name="off_forward_pe_bounded",
            description=(
                "OFF-arm positive control: forward PE must be bounded (no "
                "explosion / divergence in the online forward model)."
            ),
            control="WORST OFF-arm cell (max mean)",
            threshold=PE_SANITY_CEIL,
            direction="upper",
            kind="readiness",
            applies_to=lambda ctx: ctx["arm_id"] == ARM_OFF,
        ),
    ]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def _analyse(rows: List[Dict[str, Any]],
             aa_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    off_rows = [r for r in rows if r["arm_id"] == ARM_OFF]
    on_rows = [r for r in rows if r["arm_id"] == ARM_ON]
    off_by_seed = {r["seed"]: r for r in off_rows}
    on_by_seed = {r["seed"]: r for r in on_rows}
    seeds = sorted(set(off_by_seed) & set(on_by_seed))

    specs = _precondition_specs()
    arm_contexts = {a: _arm_context(a) for a in ARMS}

    off_pe_var_worst = _worst_cell(off_rows, "fwd_pe_all_var")
    off_pe_mean_worst = _worst_cell_max(off_rows, "fwd_pe_all_mean")

    arm_gates = []
    for arm_id, arm_rows in ((ARM_OFF, off_rows), (ARM_ON, on_rows)):
        measured: Dict[str, float] = {}
        if arm_id == ARM_ON:
            measured["harm_bias_engages_every_cell"] = _worst_cell(
                arm_rows, "decomp_n_harm_bias_nonzero")
        if arm_id == ARM_OFF:
            measured["harm_bias_inert_every_cell"] = _best_cell(
                arm_rows, "decomp_n_harm_bias_nonzero")
            measured["off_forward_pe_varies"] = off_pe_var_worst
            measured["off_forward_pe_bounded"] = off_pe_mean_worst
        gate = evaluate_arm_gate(arm_id, arm_contexts[arm_id], specs, measured=measured)
        arm_gates.append(gate)
    aggregate = aggregate_arm_gates(arm_gates)

    # --- A-A null control: discharges the matching-validity precondition BY
    # CONSTRUCTION. Any nonzero delta or non-identical action sequence on ANY
    # control seed voids the whole run -- see module docstring. ---
    aa_by_seed: Dict[int, List[Dict[str, Any]]] = {}
    for r in aa_rows:
        aa_by_seed.setdefault(int(r["seed"]), []).append(r)

    # Derive the control-seed set from what was ACTUALLY run (aa_rows), not
    # the module-level AA_CONTROL_SEEDS constant -- under --dry-run only a
    # subset of AA_CONTROL_SEEDS is run, and iterating the full constant here
    # would report the un-run seeds as missing replicates.
    aa_checks: List[Dict[str, Any]] = []
    for seed in sorted(aa_by_seed.keys()):
        reps = aa_by_seed.get(seed, [])
        if len(reps) != 2:
            aa_checks.append({
                "seed": seed, "delta": None,
                "action_sequences_identical": False, "bit_identical": False,
                "note": f"expected 2 replicate cells, found {len(reps)}",
            })
            continue
        r1, r2 = reps[0], reps[1]
        delta = r2["mean_harm_signal"] - r1["mean_harm_signal"]
        identical_actions = r1["action_sequence"] == r2["action_sequence"]
        aa_checks.append({
            "seed": seed,
            "delta": delta,
            "action_sequences_identical": identical_actions,
            "bit_identical": bool(identical_actions and delta == 0.0),
        })
    aa_control_ok = bool(aa_checks) and all(c["bit_identical"] for c in aa_checks)
    aa_abs_deltas = [abs(c["delta"]) for c in aa_checks if c["delta"] is not None]
    aa_control_max_abs_delta = max(aa_abs_deltas) if aa_abs_deltas else None

    aa_precondition_entries = [
        {
            "name": f"aa_control_bit_identical_seed_{c['seed']}",
            "kind": "readiness",
            "description": (
                "A-A null control (OFF vs OFF, same seed, both through "
                "arm_cell) must be bit-identical: delta EXACTLY 0.0, "
                "action sequences equal. Discharges the matching-validity "
                "precondition BY CONSTRUCTION -- the direct fix for 867b's "
                "RNG-reset-asymmetry defect."
            ),
            "control": f"A-A replicate pair, seed {c['seed']}",
            "measured": c["delta"] if c["delta"] is not None else float("nan"),
            "threshold_low": 0.0,
            "threshold_high": 0.0,
            "comparator_low": ">=",
            "comparator_high": "<=",
            "direction": "interval",
            "met": bool(c["bit_identical"]),
        }
        for c in aa_checks
    ]

    # --- Hard n>=MIN_SEEDS floor. Never softened -- see MIN_SEEDS docstring. ---
    n_seeds = len(seeds)
    enough_seeds = n_seeds >= MIN_SEEDS

    non_degenerate = bool(aggregate["non_degenerate"] and enough_seeds and aa_control_ok)

    # --- Covariate tiers -- purely descriptive. This DV does not condition on
    # decomposition ever occurring, so these are reported, never gating. ---
    both_decompose_seeds = [
        s for s in seeds
        if off_by_seed[s]["decomp_n_decomposed_midexec"] > 0
        and on_by_seed[s]["decomp_n_decomposed_midexec"] > 0]
    on_only_seeds = [
        s for s in seeds
        if on_by_seed[s]["decomp_n_decomposed_midexec"] > 0
        and off_by_seed[s]["decomp_n_decomposed_midexec"] == 0]
    off_only_seeds = [
        s for s in seeds
        if off_by_seed[s]["decomp_n_decomposed_midexec"] > 0
        and on_by_seed[s]["decomp_n_decomposed_midexec"] == 0]
    neither_decompose_seeds = [
        s for s in seeds if s not in both_decompose_seeds
        and s not in on_only_seeds and s not in off_only_seeds]

    # Defaults -- overwritten only in the successful (non_degenerate) branch.
    harm_deltas: List[float] = []
    harm_delta_mean = 0.0
    harm_delta_sd = 0.0
    harm_delta_se = 0.0
    rel_improvement = 0.0
    effect_size_ok = False
    rel_floor_ok = False
    c1_task_outcome_improves = False
    pe_delta_mean = 0.0
    pe_corroborates = False
    fresh_delta_mean: Optional[float] = None
    n_fresh_paired = 0
    engagement_outcome_rho: Optional[float] = None

    if not aggregate["non_degenerate"]:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        direction = "unknown"
        degeneracy_reason = aggregate["degeneracy_reason"]
    elif not aa_control_ok:
        label = "aa_control_uncontrolled_variation_run_void"
        outcome = "FAIL"
        direction = "unknown"
        degeneracy_reason = (
            "A-A null control (OFF vs OFF, same seed, both through "
            "arm_cell) produced a nonzero delta and/or non-identical action "
            "sequences on at least one control seed -- the measurement path "
            "carries an uncontrolled source of variation. Per the design's "
            "precondition (2), no C1 reading is emitted. "
            f"max_abs_aa_delta={aa_control_max_abs_delta}")
    elif not enough_seeds:
        label = "insufficient_measured_seed_count"
        outcome = "FAIL"
        direction = "unknown"
        degeneracy_reason = (
            f"only {n_seeds} measured seed(s) with both arms present, "
            f"below the pre-registered hard floor of {MIN_SEEDS}. This "
            "floor is never softened by any observed quantity -- no unit "
            "is ever excluded from this design.")
    else:
        degeneracy_reason = None
        harm_deltas = [
            on_by_seed[s]["mean_harm_signal"] - off_by_seed[s]["mean_harm_signal"]
            for s in seeds]
        harm_delta_mean = statistics.fmean(harm_deltas)
        harm_delta_sd = statistics.stdev(harm_deltas) if len(harm_deltas) > 1 else 0.0
        harm_delta_se = (
            harm_delta_sd / math.sqrt(len(harm_deltas)) if harm_deltas else 0.0)
        off_harm_ref = statistics.fmean(off_by_seed[s]["mean_harm_signal"] for s in seeds)
        rel_improvement = (
            (harm_delta_mean / abs(off_harm_ref)) if off_harm_ref not in (0.0, None) else 0.0)
        effect_size_ok = harm_delta_mean > EFFECT_SIZE_K_SIGMA * harm_delta_se
        rel_floor_ok = rel_improvement >= REL_IMPROVEMENT_FLOOR
        c1_task_outcome_improves = bool(effect_size_ok and rel_floor_ok)

        pe_deltas = [
            off_by_seed[s]["fwd_pe_all_mean"] - on_by_seed[s]["fwd_pe_all_mean"]
            for s in seeds
            if off_by_seed[s]["fwd_pe_all_mean"] is not None
            and on_by_seed[s]["fwd_pe_all_mean"] is not None]
        pe_delta_mean = statistics.fmean(pe_deltas) if pe_deltas else 0.0
        pe_corroborates = bool(pe_deltas) and pe_delta_mean > 0.0

        fresh_deltas: List[float] = []
        for s in seeds:
            off_fresh, _n_off = _mean_fresh_only(
                off_by_seed[s]["per_tick_harm"], off_by_seed[s]["e3_tick_flags"])
            on_fresh, _n_on = _mean_fresh_only(
                on_by_seed[s]["per_tick_harm"], on_by_seed[s]["e3_tick_flags"])
            if off_fresh is not None and on_fresh is not None:
                fresh_deltas.append(on_fresh - off_fresh)
        fresh_delta_mean = statistics.fmean(fresh_deltas) if fresh_deltas else None
        n_fresh_paired = len(fresh_deltas)

        on_bias_list = [float(on_by_seed[s]["decomp_n_harm_bias_nonzero"]) for s in seeds]
        engagement_outcome_rho = spearman(on_bias_list, harm_deltas)

        if c1_task_outcome_improves:
            label = "harm_aware_selection_reduces_task_harm_unconditional"
            direction = "supports"
        elif harm_delta_mean <= 0:
            label = "harm_aware_selection_does_not_reduce_task_harm_unconditional"
            direction = "weakens"
        else:
            label = "harm_aware_selection_task_effect_below_threshold_unconditional"
            direction = "mixed"
        outcome = "PASS" if c1_task_outcome_improves else "FAIL"

    non_degen_map = {
        "C1_TASK_OUTCOME_IMPROVES_UNCONDITIONAL": non_degenerate,
        "C2_FORWARD_PE_CORROBORATES": non_degenerate,
    }

    criteria = [
        {"name": "C1_TASK_OUTCOME_IMPROVES_UNCONDITIONAL", "load_bearing": True,
         "passed": bool(c1_task_outcome_improves),
         "measured": harm_delta_mean, "threshold": 0.0,
         "statement": (
             "Over ALL measured seeds (no screen, no tiering, no post-hoc "
             "selection), the unconditional whole-episode mean harm signal "
             "is LESS negative (less harmful) in ARM_SELECTION_ON than "
             "ARM_SELECTION_OFF, by an effect exceeding "
             f"{EFFECT_SIZE_K_SIGMA} x SE over >= {MIN_SEEDS} paired seeds. "
             "Bar unchanged from 844/867/867a/867b; the DV and the unit of "
             "comparison are what moved.")},
        {"name": "C2_FORWARD_PE_CORROBORATES", "load_bearing": False,
         "passed": bool(pe_corroborates),
         "measured": pe_delta_mean, "threshold": 0.0,
         "statement": (
             "Consistency check (non-load-bearing, carried over from 844's "
             "established finding): over the same seeds, forward-"
             "prediction error is also lower in ON than OFF, computed on "
             "the same unconditional whole-run basis.")},
    ]

    return {
        "outcome": outcome,
        "evidence_direction": direction,
        "interpretation_label": label,
        "criteria": criteria,
        "criteria_non_degenerate": non_degen_map,
        "preconditions": aggregate["adjudication_preconditions"] + aa_precondition_entries,
        "per_arm_gate": aggregate["per_arm_gate"],
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "aa_control": {
            "checks": aa_checks,
            "ok": aa_control_ok,
            "max_abs_delta": aa_control_max_abs_delta,
        },
        "seed_tiers_measured": {
            "both_decompose": both_decompose_seeds,
            "on_only_decompose": on_only_seeds,
            "off_only_decompose": off_only_seeds,
            "neither_decompose": neither_decompose_seeds,
        },
        "per_seed_harm_deltas": [
            {"seed": s, "harm_delta_on_minus_off": d}
            for s, d in zip(seeds, harm_deltas)
        ] if harm_deltas else [],
        "summary": {
            "n_seeds": n_seeds,
            "min_seeds_required": MIN_SEEDS,
            "enough_seeds": enough_seeds,
            "harm_delta_mean_unconditional": harm_delta_mean,
            "harm_delta_sd": harm_delta_sd,
            "harm_delta_se": harm_delta_se,
            "rel_improvement": rel_improvement,
            "effect_size_ok": effect_size_ok,
            "rel_floor_ok": rel_floor_ok,
            "pe_delta_mean_unconditional": pe_delta_mean,
            "fresh_only_secondary_delta_mean": fresh_delta_mean,
            "n_fresh_only_paired_seeds": n_fresh_paired,
            "engagement_outcome_spearman_rho": engagement_outcome_rho,
            "off_pe_var_worst": off_pe_var_worst,
            "off_pe_mean_worst": off_pe_mean_worst,
            "n_both_decompose_seeds": len(both_decompose_seeds),
            "n_on_only_seeds": len(on_only_seeds),
            "n_off_only_seeds": len(off_only_seeds),
            "n_neither_decompose_seeds": len(neither_decompose_seeds),
            "max_z_harm_a_norm_on_arm": _best_cell(on_rows, "max_z_harm_a_norm"),
            "aa_control_ok": aa_control_ok,
            "aa_control_max_abs_delta": aa_control_max_abs_delta,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> Tuple[Optional[str], Optional[str], bool]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        measurement_seeds = MEASUREMENT_SEEDS[:3]
        control_seeds = AA_CONTROL_SEEDS[:1]
        episodes = 2
        steps = 12
    else:
        measurement_seeds = MEASUREMENT_SEEDS
        control_seeds = AA_CONTROL_SEEDS
        episodes = baselines.EPISODES
        steps = baselines.STEPS_PER_EPISODE

    assert_no_structurally_unsatisfiable_gate(
        _precondition_specs(), [_arm_context(a) for a in ARMS])

    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    # --- Measurement cells. FIXED cell count = len(measurement_seeds) x 2.
    # EVERY cell enters through arm_cell -- there is no bare _run_cell call
    # anywhere in this file. ---
    rows: List[Dict[str, Any]] = []
    for seed in measurement_seeds:
        for arm_id in ARMS:
            with arm_cell(
                seed,
                config_slice=_config_slice(arm_id, episodes),
                script_path=Path(__file__),
                config_slice_declared=True,
                include_driver_script_in_hash=False,
            ) as cell:
                row = _run_cell(seed, arm_id, episodes, steps)
                cell.stamp(row)
            row["role"] = "measurement"
            rows.append(row)

    # --- A-A null control cells. Each control seed run TWICE as
    # ARM_SELECTION_OFF, both through arm_cell. Must be bit-identical. ---
    aa_rows: List[Dict[str, Any]] = []
    for seed in control_seeds:
        for replicate in (1, 2):
            with arm_cell(
                seed,
                config_slice=_config_slice(ARM_OFF, episodes),
                script_path=Path(__file__),
                config_slice_declared=True,
                include_driver_script_in_hash=False,
            ) as cell:
                row = _run_cell(seed, ARM_OFF, episodes, steps)
                cell.stamp(row)
            row["role"] = "aa_control"
            row["aa_replicate"] = replicate
            aa_rows.append(row)

    result = _analyse(rows, aa_rows)

    run_id = (f"{EXPERIMENT_TYPE}_"
              f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_v3")
    all_rows = rows + aa_rows
    cfg_record = {
        "arms": list(ARMS),
        "measurement_seeds": list(measurement_seeds),
        "aa_control_seeds": list(control_seeds),
        "episodes": episodes,
        "steps_per_episode": steps,
        "min_seeds_required": MIN_SEEDS,
        "self_dim": baselines.SELF_DIM,
        "world_dim": baselines.WORLD_DIM,
        "seeded_chunk_sequence": list(baselines.SEEDED_CHUNK_SEQUENCE),
        "decomposition_vs_threshold": baselines.DECOMPOSITION_VS_THRESHOLD,
        "decomposition_depth_cap": baselines.DECOMPOSITION_DEPTH_CAP,
        "hazard_tuned_stream_flags": dict(baselines.HAZARD_TUNED_STREAM_FLAGS),
        "harm_aware_selection_params": {
            "decomposition_harm_bias_gain": HARM_BIAS_GAIN,
            "decomposition_harm_bias_scale": HARM_BIAS_SCALE,
            "decomposition_harm_threat_floor": HARM_THREAT_FLOOR,
            "decomposition_harm_threat_ref": HARM_THREAT_REF,
            "decomposition_harm_override_w_threshold": HARM_OVERRIDE_W_THRESHOLD,
        },
        "arm_flags": {a: _arm_flags(a) for a in ARMS},
        "thresholds": {
            "PE_VARIANCE_FLOOR": PE_VARIANCE_FLOOR,
            "PE_SANITY_CEIL": PE_SANITY_CEIL,
            "HARM_BIAS_ENGAGE_FLOOR": HARM_BIAS_ENGAGE_FLOOR,
            "HARM_BIAS_INERT_CEIL": HARM_BIAS_INERT_CEIL,
            "EFFECT_SIZE_K_SIGMA": EFFECT_SIZE_K_SIGMA,
            "REL_IMPROVEMENT_FLOOR": REL_IMPROVEMENT_FLOOR,
            "MIN_SEEDS": MIN_SEEDS,
        },
    }

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "outcome": result["outcome"],
        "claim_ids": CLAIM_IDS,
        "bears_on": ["MECH-321", "SD-hazard-aware-policy-decomposition", "ARC-070", "ARC-071"],
        "evidence_direction": result["evidence_direction"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "per_arm_gate": result["per_arm_gate"],
        "aa_control": result["aa_control"],
        "seed_tiers_measured": result["seed_tiers_measured"],
        "interpretation": {
            "label": result["interpretation_label"],
            "preconditions": result["preconditions"],
            "criteria": result["criteria"],
            "criteria_non_degenerate": result["criteria_non_degenerate"],
            "preconditions_scope_note": result["per_arm_gate"].get(
                "preconditions_scope_note", ""),
        },
        "summary": result["summary"],
        "per_seed_harm_deltas": result["per_seed_harm_deltas"],
        "arm_results": all_rows,
        "per_seed_rows": all_rows,
        "custom_information": {
            "predecessor_runs_not_superseded": [
                "V3-EXQ-844", "V3-EXQ-867", "V3-EXQ-867a", "V3-EXQ-867b"],
            "design_doc": (
                "REE_assembly/evidence/planning/"
                "mech321_harm_aware_selection_measurement_redesign_"
                "staged_2026-08-08.md"),
            "substrate_doc": (
                "REE_assembly/docs/architecture/"
                "sd_hazard_aware_policy_decomposition.md"),
            "redesign_note": (
                "New EXQ number, not 867c -- three design axes changed at "
                "once relative to 867b (unit of comparison: every seed, no "
                "screen/tiering; DV: unconditional whole-episode mean_harm_"
                "signal delta, no divergence-tick windowing; sampling: n=40 "
                "pre-registered seeds, no exclusion). 867b's own manifest "
                "shows its screen-soundness falsification was an RNG-reset "
                "instrumentation artifact (screen cells bypassed arm_cell's "
                "reset; measurement cells did not), not evidence the "
                "manipulation perturbs decomposition -- the violation "
                "appeared in the OFF arm, which is provably inert. See "
                "design doc sections 2, 3, 7."),
            "aa_control_note": (
                "4 seeds x 2 replicates, ARM_SELECTION_OFF vs "
                "ARM_SELECTION_OFF, both through arm_cell (full RNG reset). "
                "Discharges the matching-validity precondition by "
                "CONSTRUCTION -- the direct fix for 867b's actual defect. "
                "Any nonzero delta voids the run (non_degenerate: false, no "
                "C1 reading)."),
            "dv_symmetry_declaration": (
                "DV = mean per-tick environment harm signal over the WHOLE "
                "run (a set-aggregate; symmetry group = permutation of the "
                "ticks it averages over). The manipulation is invariant "
                "under NEITHER a broadcast additive constant nor a "
                "permutation-preserving relabelling: harm-aware selection "
                "applies a PER-LEAF penalty (harm_bias) plus a categorical "
                "per-leaf override (select_harm_aware_leaves) across "
                "retiling candidates at selection time, changing WHICH "
                "actions are taken and hence which harm values are "
                "observed at each tick. Identical for both arms -- the OFF "
                "arm carries no manipulation at all, so its per-tick harm "
                "sequence is the harm-blind baseline by construction."),
            "gov_reuse_1_note": (
                "Decisive readout = unconditional whole-episode paired "
                "per-seed mean_harm_signal delta over n=40 seeds. Checked "
                "via reanalysis_query.py query --readout mean_harm_signal "
                "--claim MECH-321: every one of the 6 prior MECH-321 "
                "manifests carrying this readout (844/867/867a/867b + 2 "
                "policy-decomposition diagnostics) has a DISTINCT top-level "
                "substrate_hash -- no compatible group, nothing derivable "
                "by reprocessing. ree_core/** carries 18 commits since "
                "867b's 2026-08-04 run (git log --since), so even a "
                "same-lineage arm-reuse attempt against 867b's OFF mint "
                "would refuse on substrate_hash. Not recoverable -> run. "
                "try_reuse_cell OMITTED (would refuse on every cell); this "
                "run's own ARM_SELECTION_OFF cells ARE minted reuse-"
                "eligible (include_driver_script_in_hash=False) as the "
                "current canonical baseline for this lineage."),
            "re_derive_brake_note": (
                "Does not apply. All four prior MECH-321 autopsies scored "
                "non_contributory/measurement_test_design_defect or "
                "environment_adequacy_defect -- instrument-defect "
                "categories the brake's own carve-out excludes, since none "
                "owed a substrate build and the substrate has never "
                "ceilinged in this lineage."),
        },
    }

    out_path = write_flat_manifest(
        manifest, None, dry_run=args.dry_run, config=cfg_record,
        seeds=list(measurement_seeds) + list(control_seeds),
        script_path=Path(__file__),
        elapsed_seconds=round(time.perf_counter() - t0, 3),
        started_at=None,
        z_goal_stream_stats=_ZGOAL.stats(),
    )

    print(f"manifest: {out_path}", flush=True)
    s = result["summary"]
    print(
        f"outcome: {result['outcome']} label={result['interpretation_label']} "
        f"direction={result['evidence_direction']} "
        f"non_degenerate={result['non_degenerate']}", flush=True)
    print(
        f"  n_seeds={s['n_seeds']}/{s['min_seeds_required']} "
        f"harm_delta_mean={s['harm_delta_mean_unconditional']:.6g} "
        f"harm_delta_se={s['harm_delta_se']:.6g} "
        f"rel_improvement={s['rel_improvement']:.4f}", flush=True)
    print(
        f"  aa_control_ok={s['aa_control_ok']} "
        f"aa_control_max_abs_delta={s['aa_control_max_abs_delta']}", flush=True)
    print(
        f"  both_decompose={s['n_both_decompose_seeds']} "
        f"on_only={s['n_on_only_seeds']} off_only={s['n_off_only_seeds']} "
        f"neither={s['n_neither_decompose_seeds']} (covariate tiers, "
        "non-gating)", flush=True)
    print(
        f"  green_arms={result['per_arm_gate']['green_arms']} "
        f"red_arms={result['per_arm_gate']['red_arms']}", flush=True)
    for c in result["criteria"]:
        print(f"  {c['name']}: passed={c['passed']} measured={c['measured']} "
              f"load_bearing={c['load_bearing']}", flush=True)
    if result["degeneracy_reason"]:
        print(f"  degeneracy_reason: {result['degeneracy_reason']}", flush=True)
    print(f"started_utc: {started.strftime('%Y%m%dT%H%M%SZ')}", flush=True)

    if args.dry_run:
        # Smoke assertions: (a) the manipulation must be non-trivially
        # engaged on EVERY ON cell and inert on EVERY OFF cell -- far cheaper
        # to catch here than after the full grid; (b) the A-A control must
        # actually be bit-identical, or the smoke has already caught 867b's
        # defect class recurring.
        on_bias_min = min(
            (r["decomp_n_harm_bias_nonzero"] for r in rows if r["arm_id"] == ARM_ON),
            default=-1)
        off_bias_max = max(
            (r["decomp_n_harm_bias_nonzero"] for r in rows if r["arm_id"] == ARM_OFF),
            default=0)
        print(f"[smoke] on_arm_harm_bias_nonzero_min={on_bias_min} "
              f"off_arm_harm_bias_nonzero_max={off_bias_max} "
              f"aa_control_ok={result['aa_control']['ok']} "
              f"aa_control_checks={result['aa_control']['checks']}", flush=True)
        assert on_bias_min > 0, (
            "SMOKE FAIL: at least one ARM_SELECTION_ON cell never engaged "
            "the graded harm bias (decomp_n_harm_bias_nonzero == 0). This "
            "is 867's inert-manipulation failure recurring, now caught "
            "per-cell -- do not queue.")
        assert off_bias_max == 0, (
            "SMOKE FAIL: at least one ARM_SELECTION_OFF cell engaged the "
            "harm bias, which must be a bit-identical no-op in that arm.")
        assert result["aa_control"]["ok"], (
            "SMOKE FAIL: the A-A null control (OFF vs OFF, same seed) did "
            "not produce a bit-identical replicate pair. This is exactly "
            "867b's RNG-reset-asymmetry defect resurfacing -- do not queue "
            "until every cell provably enters through arm_cell with a full "
            "reset.")

    outcome_norm = str(result["outcome"]).upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    return outcome_emit, str(out_path), bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry_run = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path,
                     dry_run=_dry_run)
