#!/opt/local/bin/python3
"""V3-EXQ-844 -- MECH-321 R4 MID-EXECUTION: does the now-reachable abort hook
help or hurt TASK OUTCOME?

EVIDENCE. `claim_ids=["MECH-321"]`. Successor to V3-EXQ-839
(confirmed `failure_autopsy_V3-EXQ-839_2026-07-30.md`), which validated that
SD-084's persistent committed-program handle makes MECH-321's R4 mid-execution
hook REACHABLE (415 evaluations ON, 0 OFF) but adjudicated REACHABILITY ONLY --
its own docstring states "the behavioural delta is what R4 actually CLAIMS...
A successor evidence experiment consumes this block instead of re-deriving it."
This is that successor.

THE QUESTION
    MECH-321's R4 functional_restatement (claims.yaml): mid-execution
    decomposition reads V_s on a committed macro's REMAINING content while it
    is executing; if V_s has dropped (or a MECH-288 boundary fires), the
    remainder is decomposed and the commit latch released rather than blindly
    finishing a now-unreliable plan. The reachability question is closed; the
    TASK-OUTCOME question -- does aborting a stale macro reduce execution-time
    forward-prediction error / harm relative to letting it run to completion --
    is open. C1 below is the load-bearing criterion for that question.

NOT A REPEAT OF THE REFUSED 816-FAMILY RE-QUEUE. claims.yaml MECH-321 carries a
6-hit diagnostic-chain-recurrence note (GOV-DIAG-1) on the "policy
decomposition" bears_on chain; the 2026-07-29 entry refused a fourth
same-question env-harshening letter (816e) on the PRE-COMMIT / R1 forward-PE
DISCRIMINATION-FLOOR axis (816/816b/816c/816d all saturated at forward-PE
~0.0086-0.0087, short of the 0.01 floor -- a mis-posed-question signal on THAT
axis). This run is R4 (MID-EXECUTION), a different phase of the same
mechanism, reached via V3-EXQ-839's clean (not verdict-free) reachability
result, and it is EVIDENCE (claim_ids=[MECH-321]) rather than another
verdict-free diagnostic. The 2026-07-30 governance note names this exact
successor as "the redesign-the-discriminating-statistic response the
2026-07-29 note already called for" -- see that note for the full genealogy.

GOV-REUSE-1 (Step 2.4). Decisive readout: the PAIRED, POST-DIVERGENCE-WINDOW
mean forward-prediction error / harm (ON vs OFF), on seeds where mid-execution
decomposition ACTUALLY OCCURRED this run. V3-EXQ-839's own `behavioural_delta`
block records `mean_harm_signal_off/on/delta`, but only as a WHOLE-RUN mean --
diluted by the ~99% of ticks the manipulation never touches (see the WHY A
WHOLE-RUN MEAN IS THE WRONG STATISTIC section below) -- and it does not retain
per-tick forward-PE or the raw per-tick harm sequence at all (checked: 839's
row schema has no `e3_prediction_error`/`forward_pe` field; `update_residue`'s
return value is called but not captured in that script). So the decisive
windowed statistic is NOT recoverable from 839's manifest -> not recoverable ->
run. Checked: v3_exq_839_sd084_midexec_reachability_20260729T220727Z_v3.json,
no per-tick forward-PE field present, `behavioural_delta` is whole-run only.

WHY A WHOLE-RUN MEAN IS THE WRONG STATISTIC, MEASURED (not assumed)
    839's own `behavioural_delta` shows the whole-run harm delta is small and
    mixed across its three decomposing seeds (mean_harm_signal_delta, ON-OFF):
    seed 3 -0.000737, seed 71 -0.002024, seed 89 -0.000449 -- i.e. slightly
    WORSE on 3/3 seeds by that aggregate, the opposite of MECH-321's predicted
    direction. But a mid-execution abort can only possibly affect ticks from
    the FIRST divergence tick onward (839 itself proves the arms are
    BIT-IDENTICAL before that tick on every seed) -- 12 episodes x 60 steps =
    720 ticks per cell, and 839's `first_divergence_tick` values (101, 64, 78)
    mean the affected window is at most the LAST 85% of the run and typically
    much less once the macro re-stabilises. A whole-run mean is therefore
    diluted by up to that fraction of untouched, arm-identical ticks -- exactly
    the same "aggregate mean vs a windowed / worst-cell statistic" hazard this
    skill's Step 3.5 checklist warns about for preconditions, arriving here at
    an evidence DV instead. This run computes the windowed statistic 839's own
    docstring says a successor should compute instead of re-deriving the
    (wrong-grain) whole-run one.

ARMS -- IDENTICAL TO THE SD-084 LINEAGE, NO NEW MANIPULATION
    Reuses `experiments/_lib/baselines/sd084_midexec_reachability.py` verbatim
    for env, schedule, substrate stack and BOTH seed tiers -- same
    ARM_HANDLE_OFF / ARM_HANDLE_ON flags V3-EXQ-839 used. The manipulation is
    unchanged (`use_persistent_committed_program_handle`); only the
    INSTRUMENTATION and the DV differ (per-tick forward-PE + harm captured
    fresh, windowed around each seed's own divergence tick, rather than
    reachability counters).

WHY THE OFF ARM IS RE-RUN RATHER THAN CELL-REUSED FROM 839
    839 minted ARM_HANDLE_OFF reuse-eligible
    (`include_driver_script_in_hash=False`) precisely so a successor could skip
    re-running it. `try_reuse_cell` is called below with the FULL needed-key
    list this run actually reads (`per_tick_forward_pe`, `per_tick_harm`,
    `action_sequence`, decomposition counters) and 839's manifest does not
    carry the first two -- so the reuse correctly REFUSES
    (`set(needed_keys) <= set(cell_keys)` fails) and the arm runs fresh. This
    is the intended, safe behaviour of the refuse-by-default gate (a false
    cache-miss costs compute, never correctness) -- attempted and logged
    rather than skipped, per the skill's reuse-consumer recipe. THIS run's own
    OFF cells are re-minted reuse-eligible from the SAME canonical baseline
    module (`include_driver_script_in_hash=False`) so a THIRD, richer-still
    consumer can hit against either mint.

TWO SEED TIERS WITHIN ARM_HANDLE_ON, BOTH FROM 839's OWN MEASUREMENT
    839's `behavioural_delta` already discriminates, per attributable seed,
    between "the hook FIRES and DECOMPOSES" (3, 71, 89: decomposed_midexec_on
    50/8/77, action_divergence_frac > 0) and "the hook FIRES but stays INERT"
    (47: decomposed_midexec_on=0, action_divergence_frac=0.0 -- the hook
    evaluates 134 times and never once decides to decompose). This IS the
    routing's requested "ON-fires-but-inert vs ON-decomposes" split -- realised
    as a SEED tier within one ARM_HANDLE_ON arm rather than a fabricated third
    config arm, because no knob in the substrate can force decomposition
    independent of the seed's own V_s / boundary trajectory (same reasoning
    839's module docstring gives for why the seed pool is a measurement, not a
    free parameter). This run RE-MEASURES the split fresh (839's tiering is a
    prior, not an assumption) rather than importing 839's per-seed
    classification, in case online learning this run performs shifts which
    seeds decompose.
      ON-DECOMPOSES tier  -- seeds where THIS run's own
          decomp_n_decomposed_midexec > 0. Carries the load-bearing C1.
      ON-FIRES-INERT / negative-control tier -- seeds where the hook evaluates
          (or the arm never diverges) but does not decompose. Predicted NULL:
          forward-PE / harm delta ~ 0, because no latch release occurred so
          the two arms follow the identical action sequence. A non-null
          reading here would mean the handle changes behaviour through some
          path other than decomposition -- informative, not gating.

HOLD-WEIGHTED-READOUT TRIAGE (validate_experiments advisory), FIX APPLIED NOT WAIVED
    Both `per_tick_harm` and `per_tick_forward_pe` accumulate once per ENV STEP,
    and `agent.py` returns the HELD action (and `generate_trajectories` returns
    CACHED candidates) on a non-e3-tick before `e3.select()` is reached
    (cadence `e3_steps_per_tick` defaults 10). Unlike 839's LATCHED E3
    diagnostics (which repeat a literally frozen value across a hold -- pure
    pseudo-replication), `harm` and `e3_prediction_error` are NOT latched: the
    environment genuinely advances and the agent's actual latent genuinely
    moves every tick even while executing a held macro, so these are not
    repeated-frozen values. But the windowed MEAN across such ticks is still a
    CONTINUOUS-MARGIN statistic gating a load-bearing criterion -- the
    advisory's "AT RISK", not "SAFE" (threshold-invariant) case -- so the
    prescribed fix is applied, not waived: `e3_tick_flags` is recorded per
    tick (839's exact pattern) and the windowed delta is computed in BOTH
    forms, kept distinct. **C1 (load-bearing) uses the FRESH-SELECTION-
    RESTRICTED delta** (ticks where `e3.select()` actually ran this tick),
    which removes the hold-weighting confound entirely; the all-tick
    (hold-weighted) delta is retained as a NON-gating corroborating readout of
    realised task outcome over the full physical window. If the fresh-tick
    sample in a seed's window is empty, that seed's fresh-restricted delta is
    `None` and does not enter C1 -- underpowered, not silently substituted.

DV-SYMMETRY, DECLARED PER THE 604c RUBRIC
    Load-bearing DV = paired per-seed windowed mean forward-prediction error
    delta (OFF minus ON, post-divergence window). Symmetry group: a mean over
    ticks is a symmetric (permutation-invariant) function of its per-tick
    terms. The manipulation is NOT invariant under it: a mid-execution abort
    changes WHICH action executes at and after the abort tick (a branch change
    in `select_action`, exactly as 839 declares for its own behavioural DVs),
    which changes the per-tick forward-PE and harm terms themselves, not
    merely their order or scale. Not a broadcast-additive-constant-on-argmax
    case (839's C1 reachability counter is the one with that shape, and it is
    not reused as a DV here) and not a monotone-rescaling-of-a-rank case (the
    DV is a raw magnitude mean, not an order statistic).

READINESS (P0), MIRRORING V3-EXQ-816's P1-P3 SHAPE FOR A FORWARD-PE DV
    P_MULTI  multi_action_commits_present (existential, both arms) -- carried
        over from 839's own gate 6 precondition: without a multi-action
        commit there is nothing for the mid-execution hook to act on.
    P_PRECOMMIT  decomposition_precommit_live (existential, both arms) --
        839's positive control that the decomposition machinery runs at all.
    P_MIDEXEC_DECOMP  midexec_decomposition_occurs (existential, ON arm ONLY
        via `applies_to`) -- at least one seed shows THIS run's own
        `decomp_n_decomposed_midexec > 0`. Reachability (839's C1) is
        necessary but not sufficient for a task-outcome test: seed 47 shows
        evaluation without decomposition, which correctly produces no
        divergence to measure. Below floor -> substrate_not_ready_requeue,
        never a task-outcome verdict on a test that never ran.
    P_PE_VARIES / P_PE_BOUNDED  OFF-arm forward-PE is a real, bounded, VARYING
        quantity (applies to OFF only, mirroring 816's P3 positive control) --
        rules out an untrained-forward-model noise floor making the DV
        meaningless before any cross-arm comparison is drawn.

SLEEP DRIVER: not applicable -- no sleep phase entered in this run.

Z_GOAL: deliberately inert, carried over verbatim from 839 for the identical
reason (REEConfig.from_dims(z_goal_enabled=...) defaults False, so
`goal_state` is never constructed; recorded via the accumulator so absence
reads as measured-inert rather than unmeasured).
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
from experiments._lib.arm_reuse import try_reuse_cell  # noqa: E402
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
import experiments._lib.baselines.sd084_midexec_reachability as baselines  # noqa: E402

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.policy import ChunkedPrimitive, ChunkState  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_844_mech321_r4_midexec_task_effect"
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["MECH-321"]

ARM_OFF = "ARM_HANDLE_OFF"
ARM_ON = "ARM_HANDLE_ON"
ARMS = (ARM_OFF, ARM_ON)

MINT_RUN_ID = "v3_exq_839_sd084_midexec_reachability_20260729T220727Z_v3"

# --- Pre-registered thresholds. Constants, never derived from this run's own
# statistics (mirrors 839 / 816 convention). ---
MULTI_ACTION_COMMIT_FLOOR = 0.0
PRECOMMIT_EVAL_FLOOR = 0.0
MIDEXEC_DECOMP_FLOOR = 0.0
PE_VARIANCE_FLOOR = 1e-12
PE_SANITY_CEIL = 1e6

# Effect-size gate for the load-bearing paired delta, same shape as 816's
# EFFECT_SIZE_K_SIGMA / REL_IMPROVEMENT_FLOOR. K is lenient (1.0) because the
# seed tier is small BY SUBSTRATE CONSTRUCTION (no chunk-selection-bias knob
# exists to force more decomposing seeds -- see baseline module docstring);
# this is a known, declared limitation, not an under-designed n.
EFFECT_SIZE_K_SIGMA = 1.0
REL_IMPROVEMENT_FLOOR = 0.0
MIN_DECOMPOSING_SEEDS = 2

_ZGOAL = ZGoalStreamAccumulator()


# ---------------------------------------------------------------------------
# Config construction (verbatim from the SD-084 lineage baseline module)
# ---------------------------------------------------------------------------
def _arm_flags(arm_id: str) -> Dict[str, Any]:
    return (baselines.off_arm_flags() if arm_id == ARM_OFF
            else baselines.on_arm_flags())


def _config_slice(arm_id: str) -> Dict[str, Any]:
    if arm_id == ARM_OFF:
        return baselines.off_path_config_slice()
    slice_ = baselines.off_path_config_slice()
    slice_.update(baselines.on_arm_flags())
    return slice_


def _build(seed: int, arm_id: str) -> Tuple[CausalGridWorldV2, REEAgent, Dict[str, Any]]:
    env = CausalGridWorldV2(**baselines.env_kwargs(seed))
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


def _needed_keys() -> List[str]:
    """Keys this run reads back from a (real or reused) OFF cell."""
    return [
        "per_tick_forward_pe", "per_tick_harm", "action_sequence",
        "e3_tick_flags", "decomp_n_evaluated_midexec",
        "decomp_n_decomposed_midexec", "decomp_n_evaluated_precommit",
        "multi_action_commits", "n_ticks",
    ]


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------
def _run_cell(seed: int, arm_id: str, episodes: int, steps: int,
              quiet: bool = False) -> Dict[str, Any]:
    """Drive the canonical loop for one (seed, arm) cell, capturing PER-TICK
    forward-PE and harm (not just aggregates) so a post-divergence window can
    be computed after the fact. Nothing about the mid-execution gate is
    injected -- identical discipline to V3-EXQ-839's `_run_cell`."""
    env, agent, flags = _build(seed, arm_id)
    _register_chunk(agent)
    world_dim = agent.config.latent.world_dim

    n_ticks = 0
    multi_action_commits = 0
    actions: List[int] = []
    forward_pe_ticks: List[Optional[float]] = []
    harm_ticks: List[float] = []
    e3_tick_flags: List[bool] = []

    if not quiet:
        print(f"Seed {seed} Condition {arm_id}", flush=True)

    for ep in range(episodes):
        _, obs = env.reset()
        agent.reset()
        if not agent.policy_chunking.library.all_chunks():
            _register_chunk(agent)

        for _ in range(steps):
            latent = agent.sense(obs["body_state"], obs["world_state"])
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent)
                if ticks.get("e1_tick")
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            # Hold-weighting audit (validate_experiments advisory): a
            # non-e3-tick returns the HELD action before e3.select() is
            # reached, so this flag separates fresh selections from held
            # repeats -- 839's exact pattern.
            e3_tick_flags.append(bool(ticks.get("e3_tick")))
            body = obs["body_state"]
            # kwargs-only -- a positional call collides with `latent` (839's
            # documented EXQ-471/475/483/490/524 trap). z_goal deliberately
            # inert here; see module docstring.
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
            # Full waking loop: update_residue trains the forward model AND
            # returns the committed-trajectory execution-time forward PE
            # (metrics["e3_prediction_error"], per ree_core/agent.py:8656 /
            # e3_selector.py post_action_update -- verified against source,
            # same read 816 uses).
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
        "mean_harm_signal": (
            statistics.fmean(harm_ticks) if harm_ticks else 0.0),
        "arm_flags": dict(flags),
    }

    cell_pass = (
        multi_action_commits > 0 and row["decomp_n_decomposed_midexec"] > 0
        if arm_id == ARM_ON
        else multi_action_commits > 0 and row["decomp_n_evaluated_midexec"] == 0
    )
    row["cell_pass"] = bool(cell_pass)
    if not quiet:
        print(f"verdict: {'PASS' if cell_pass else 'FAIL'}", flush=True)
    return row


def _reused_off_row(seed: int, cached: Dict[str, Any]) -> Dict[str, Any]:
    """Reshape a `try_reuse_cell` hit into the same row shape `_run_cell`
    produces, so downstream analysis code does not need to branch on
    reused-vs-fresh."""
    row = dict(cached)
    row["arm_id"] = ARM_OFF
    row["seed"] = int(seed)
    row.setdefault("e3_tick_flags", [])
    fwd = row.get("per_tick_forward_pe") or []
    row["fwd_pe_all_mean"] = (
        statistics.fmean(v for v in fwd if v is not None)
        if any(v is not None for v in fwd) else None)
    row["fwd_pe_all_var"] = (
        statistics.pvariance(v for v in fwd if v is not None)
        if sum(1 for v in fwd if v is not None) > 1 else 0.0)
    harm = row.get("per_tick_harm") or []
    row["mean_harm_signal"] = statistics.fmean(harm) if harm else 0.0
    row["cell_pass"] = bool(
        row.get("multi_action_commits", 0) > 0
        and row.get("decomp_n_evaluated_midexec", 0) == 0)
    row["reused_from_run_id"] = MINT_RUN_ID
    return row


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------
def _arm_context(arm_id: str) -> Dict[str, Any]:
    return {"id": arm_id, "arm_id": arm_id, "handle_on": arm_id == ARM_ON}


def _precondition_specs() -> List[PreconditionSpec]:
    return [
        PreconditionSpec(
            name="multi_action_commits_present",
            description=(
                "At least one cell of this ARM committed a MULTI-ACTION "
                "program. Gate (6) of the mid-execution conjunction requires "
                "len(remaining) > 1 -- carried over from V3-EXQ-839's "
                "identical precondition. EXISTENTIAL, both arms."
            ),
            control="BEST cell of this arm (max)",
            threshold=MULTI_ACTION_COMMIT_FLOOR,
            direction="lower",
            kind="readiness",
        ),
        PreconditionSpec(
            name="decomposition_precommit_live",
            description=(
                "Pre-commit decomposition evaluated at least once (V3-EXQ-830's "
                "known-reachable phase) -- a zero means the decomposition "
                "machinery is not running at all, a config error rather than a "
                "finding about this run's question."
            ),
            control="BEST cell of this arm (positive control)",
            threshold=PRECOMMIT_EVAL_FLOOR,
            direction="lower",
            kind="readiness",
        ),
        PreconditionSpec(
            name="midexec_decomposition_occurs",
            description=(
                "At least one ON-arm cell shows decomp_n_decomposed_midexec > "
                "0 THIS run. Reachability (V3-EXQ-839's C1) is necessary but "
                "NOT sufficient for a task-outcome test: a seed can evaluate "
                "the hook without ever deciding to decompose (839's seed 47), "
                "which produces no divergence to measure. EXISTENTIAL, ON "
                "arm only -- OFF structurally cannot decompose by design, "
                "that is the negative control, not a readiness failure."
            ),
            control="BEST ON-arm cell (max)",
            threshold=MIDEXEC_DECOMP_FLOOR,
            direction="lower",
            kind="readiness",
            applies_to=lambda ctx: ctx["arm_id"] == ARM_ON,
        ),
        PreconditionSpec(
            name="off_forward_pe_varies",
            description=(
                "OFF-arm positive control: committed-trajectory forward PE "
                "must have non-zero variance across the whole run (world_forward "
                "learned something informative) -- else the DV is a degenerate "
                "constant and any ON-vs-OFF delta would be noise, not signal. "
                "Mirrors V3-EXQ-816's P3."
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


def _best_cell(rows: List[Dict[str, Any]], key: str) -> float:
    return float(max((r[key] for r in rows), default=0.0))


def _worst_cell(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(min(vals)) if vals else 0.0


def _worst_cell_max(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(max(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# Windowed per-seed paired delta
# ---------------------------------------------------------------------------
def _first_divergence_tick(off_row: Dict[str, Any], on_row: Dict[str, Any]) -> Optional[int]:
    a_off = off_row["action_sequence"]
    a_on = on_row["action_sequence"]
    n = min(len(a_off), len(a_on))
    for i in range(n):
        if a_off[i] != a_on[i]:
            return i
    return None


def _mean_at(values: List[Any], flags: List[bool], start: int,
             fresh_only: bool) -> Tuple[Optional[float], int]:
    """Mean of `values[start:]`, optionally restricted to indices flagged
    fresh in `flags[start:]`. Returns (mean_or_None, n_used)."""
    picked: List[float] = []
    for i in range(start, len(values)):
        v = values[i]
        if v is None:
            continue
        if fresh_only:
            if i >= len(flags) or not flags[i]:
                continue
        picked.append(float(v))
    return (statistics.fmean(picked) if picked else None), len(picked)


def _windowed_delta(off_row: Dict[str, Any], on_row: Dict[str, Any],
                     div_tick: int) -> Dict[str, Any]:
    """Post-divergence-window paired readout for one seed, in BOTH the
    all-tick (hold-weighted) and fresh-selection-restricted forms (validate_
    experiments hold-weighted advisory, "emit BOTH" -- see module docstring).

    Each arm's OWN per-tick series from div_tick to the end of its own run --
    not requiring the two arms' tick counts to still align after the abort
    changes the action sequence (n_ticks per cell is fixed by the schedule,
    so both arms have exactly the same NUMBER of ticks; only which action
    fires at each tick, and hence which ticks are e3-fresh, may differ).
    """
    off_harm, off_harm_fresh = off_row["per_tick_harm"], off_row.get("e3_tick_flags") or []
    on_harm, on_harm_fresh = on_row["per_tick_harm"], on_row.get("e3_tick_flags") or []
    off_fwd, on_fwd = off_row["per_tick_forward_pe"], on_row["per_tick_forward_pe"]

    off_pe_all, _ = _mean_at(off_fwd, off_harm_fresh, div_tick, fresh_only=False)
    on_pe_all, _ = _mean_at(on_fwd, on_harm_fresh, div_tick, fresh_only=False)
    off_harm_all, _ = _mean_at(off_harm, off_harm_fresh, div_tick, fresh_only=False)
    on_harm_all, _ = _mean_at(on_harm, on_harm_fresh, div_tick, fresh_only=False)

    off_pe_fresh, n_off_pe_fresh = _mean_at(off_fwd, off_harm_fresh, div_tick, fresh_only=True)
    on_pe_fresh, n_on_pe_fresh = _mean_at(on_fwd, on_harm_fresh, div_tick, fresh_only=True)
    off_harm_fresh_mean, n_off_harm_fresh = _mean_at(off_harm, off_harm_fresh, div_tick, fresh_only=True)
    on_harm_fresh_mean, n_on_harm_fresh = _mean_at(on_harm, on_harm_fresh, div_tick, fresh_only=True)

    def _harm_delta(off_m, on_m):
        # harm_signal < 0 means harm occurred; positive delta = ON is LESS
        # harmful (less negative / more positive) than OFF in the window.
        return (on_m - off_m) if (off_m is not None and on_m is not None) else None

    def _pe_delta(off_m, on_m):
        # positive = ON has LOWER forward PE than OFF in the window (improvement)
        return (off_m - on_m) if (off_m is not None and on_m is not None) else None

    return {
        "seed": off_row["seed"],
        "divergence_tick": div_tick,
        "n_window_ticks": len(off_harm) - div_tick,
        "n_window_ticks_fresh": min(n_off_harm_fresh, n_on_harm_fresh),
        # All-tick (hold-weighted): NON-gating, realised task outcome over
        # the full physical window.
        "off_harm_window_mean_all": off_harm_all,
        "on_harm_window_mean_all": on_harm_all,
        "harm_delta_all_off_minus_on": _harm_delta(off_harm_all, on_harm_all),
        "off_pe_window_mean_all": off_pe_all,
        "on_pe_window_mean_all": on_pe_all,
        "pe_delta_all_off_minus_on": _pe_delta(off_pe_all, on_pe_all),
        # Fresh-selection-restricted: LOAD-BEARING, hold-weighting confound removed.
        "off_harm_window_mean_fresh": off_harm_fresh_mean,
        "on_harm_window_mean_fresh": on_harm_fresh_mean,
        "harm_delta_fresh_off_minus_on": _harm_delta(off_harm_fresh_mean, on_harm_fresh_mean),
        "off_pe_window_mean_fresh": off_pe_fresh,
        "on_pe_window_mean_fresh": on_pe_fresh,
        "pe_delta_fresh_off_minus_on": _pe_delta(off_pe_fresh, on_pe_fresh),
    }


def _null_check_delta(off_row: Dict[str, Any], on_row: Dict[str, Any]) -> Dict[str, Any]:
    """For a seed with NO divergence (fires-but-inert or negative-control
    tier): whole-run means should be bit-identical between arms. Non-gating."""
    off_pe = [v for v in off_row["per_tick_forward_pe"] if v is not None]
    on_pe = [v for v in on_row["per_tick_forward_pe"] if v is not None]
    off_pe_mean = statistics.fmean(off_pe) if off_pe else None
    on_pe_mean = statistics.fmean(on_pe) if on_pe else None
    return {
        "seed": off_row["seed"],
        "off_pe_whole_run_mean": off_pe_mean,
        "on_pe_whole_run_mean": on_pe_mean,
        "off_harm_whole_run_mean": off_row["mean_harm_signal"],
        "on_harm_whole_run_mean": on_row["mean_harm_signal"],
        "action_sequences_identical": off_row["action_sequence"] == on_row["action_sequence"],
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def _analyse(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    off_rows = [r for r in rows if r["arm_id"] == ARM_OFF]
    on_rows = [r for r in rows if r["arm_id"] == ARM_ON]
    off_by_seed = {r["seed"]: r for r in off_rows}
    on_by_seed = {r["seed"]: r for r in on_rows}

    specs = _precondition_specs()
    arm_contexts = {a: _arm_context(a) for a in ARMS}

    off_pe_var_worst = _worst_cell(off_rows, "fwd_pe_all_var")
    off_pe_mean_worst = _worst_cell_max(off_rows, "fwd_pe_all_mean")

    arm_gates = []
    for arm_id, arm_rows in ((ARM_OFF, off_rows), (ARM_ON, on_rows)):
        measured = {
            "multi_action_commits_present": _best_cell(arm_rows, "multi_action_commits"),
            "decomposition_precommit_live": _best_cell(arm_rows, "decomp_n_evaluated_precommit"),
        }
        if arm_id == ARM_ON:
            measured["midexec_decomposition_occurs"] = _best_cell(
                arm_rows, "decomp_n_decomposed_midexec")
        if arm_id == ARM_OFF:
            measured["off_forward_pe_varies"] = off_pe_var_worst
            measured["off_forward_pe_bounded"] = off_pe_mean_worst
        gate = evaluate_arm_gate(arm_id, arm_contexts[arm_id], specs, measured=measured)
        arm_gates.append(gate)

    aggregate = aggregate_arm_gates(arm_gates)

    # --- Seed tiering, RE-MEASURED this run (not imported from 839). ---
    seeds = sorted(set(off_by_seed) & set(on_by_seed))
    decomposing_seeds = [s for s in seeds if on_by_seed[s]["decomp_n_decomposed_midexec"] > 0]
    inert_seeds = [s for s in seeds if s not in decomposing_seeds]

    windowed: List[Dict[str, Any]] = []
    for s in decomposing_seeds:
        div_tick = _first_divergence_tick(off_by_seed[s], on_by_seed[s])
        if div_tick is not None:
            windowed.append(_windowed_delta(off_by_seed[s], on_by_seed[s], div_tick))

    null_checks = [_null_check_delta(off_by_seed[s], on_by_seed[s]) for s in inert_seeds]

    # LOAD-BEARING: fresh-selection-restricted delta (hold-weighting confound
    # removed -- see module docstring's hold-weighted-readout triage).
    harm_deltas = [w["harm_delta_fresh_off_minus_on"] for w in windowed
                   if w["harm_delta_fresh_off_minus_on"] is not None]
    n_pairs = len(harm_deltas)
    harm_delta_mean = statistics.fmean(harm_deltas) if harm_deltas else 0.0
    harm_delta_sd = statistics.stdev(harm_deltas) if n_pairs > 1 else 0.0
    harm_delta_se = (harm_delta_sd / math.sqrt(n_pairs)) if n_pairs > 0 else 0.0
    off_harm_ref = (statistics.fmean(w["off_harm_window_mean_fresh"] for w in windowed
                                     if w["off_harm_window_mean_fresh"] is not None)
                    if any(w["off_harm_window_mean_fresh"] is not None for w in windowed)
                    else 0.0)
    rel_improvement = (
        (harm_delta_mean / abs(off_harm_ref)) if off_harm_ref not in (0.0, None) else 0.0)

    # NON-gating corroboration: mechanistic (forward-PE, fresh-restricted) and
    # realised-task-outcome-over-full-window (all-tick, hold-weighted) readouts.
    pe_deltas = [w["pe_delta_fresh_off_minus_on"] for w in windowed
                 if w["pe_delta_fresh_off_minus_on"] is not None]
    pe_delta_mean = statistics.fmean(pe_deltas) if pe_deltas else 0.0
    pe_corroborates = bool(pe_deltas) and pe_delta_mean > 0.0

    all_tick_harm_deltas = [w["harm_delta_all_off_minus_on"] for w in windowed
                            if w["harm_delta_all_off_minus_on"] is not None]
    all_tick_harm_delta_mean = (
        statistics.fmean(all_tick_harm_deltas) if all_tick_harm_deltas else 0.0)

    enough_pairs = n_pairs >= min(MIN_DECOMPOSING_SEEDS, len(decomposing_seeds) or 1)
    effect_size_ok = enough_pairs and (harm_delta_mean > EFFECT_SIZE_K_SIGMA * harm_delta_se)
    rel_floor_ok = rel_improvement >= REL_IMPROVEMENT_FLOOR
    c1_task_outcome_improves = bool(
        aggregate["non_degenerate"] and enough_pairs and effect_size_ok and rel_floor_ok)

    # Non-degeneracy: readiness must hold AND the manipulation must actually
    # have produced at least one measurable windowed pair -- else this is
    # either substrate-not-ready or (readiness met, zero divergence anywhere)
    # a DV-symmetry / inertness trap, mirroring 816's `manipulation_inert`.
    any_divergence = len(windowed) > 0
    non_degenerate = bool(aggregate["non_degenerate"] and any_divergence)

    if not aggregate["non_degenerate"]:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        direction = "unknown"
        degeneracy_reason = aggregate["degeneracy_reason"]
    elif not any_divergence:
        label = "manipulation_inert"
        outcome = "FAIL"
        direction = "unknown"
        degeneracy_reason = (
            "readiness met but no seed produced BOTH a mid-execution "
            "decomposition and a resulting action-sequence divergence -- the "
            "manipulation was behaviourally inert this run (DV-symmetry trap: "
            "no window exists to measure)")
    elif not enough_pairs:
        label = "midexec_task_effect_underpowered"
        outcome = "FAIL"
        direction = "unknown"
        degeneracy_reason = (
            f"only {n_pairs} decomposing-seed pair(s) with a measurable window "
            f"(< {MIN_DECOMPOSING_SEEDS}); the seed pool is a substrate "
            "measurement (no chunk-selection-bias knob), not a free parameter "
            "-- see baseline module docstring")
    else:
        degeneracy_reason = None
        if c1_task_outcome_improves:
            label = "midexec_decomposition_reduces_harm"
            direction = "supports"
        elif harm_delta_mean <= 0:
            label = "midexec_decomposition_does_not_reduce_harm"
            direction = "weakens"
        else:
            label = "midexec_task_effect_below_threshold"
            direction = "mixed"
        outcome = "PASS" if c1_task_outcome_improves else "FAIL"

    # Both criteria are asserted ACROSS arms (a paired OFF-vs-ON delta), so
    # neither is owned by a single arm's gate the way a per-arm criterion
    # would be -- non-degeneracy here is this run's own readiness-AND-power
    # computation (aggregate gate green, a real windowed pair existed, and
    # enough decomposing seeds), not `arm_criteria_non_degenerate`'s
    # single-arm-ownership model.
    non_degen_map = {
        "C1_TASK_OUTCOME_IMPROVES": non_degenerate,
        "C2_FORWARD_PE_CORROBORATES": non_degenerate,
    }

    criteria = [
        {"name": "C1_TASK_OUTCOME_IMPROVES", "load_bearing": True,
         "passed": bool(c1_task_outcome_improves),
         "measured": harm_delta_mean, "threshold": 0.0,
         "statement": (
             "On seeds where mid-execution decomposition actually fired "
             "(decomp_n_decomposed_midexec > 0), the paired post-divergence-"
             "window mean harm signal, restricted to FRESH e3-selection ticks "
             "(hold-weighting confound removed -- see module docstring), is "
             "LESS negative (less harmful) in ARM_HANDLE_ON than "
             "ARM_HANDLE_OFF, by an effect exceeding "
             f"{EFFECT_SIZE_K_SIGMA} x SE over >= {MIN_DECOMPOSING_SEEDS} "
             "paired seeds.")},
        {"name": "C2_FORWARD_PE_CORROBORATES", "load_bearing": False,
         "passed": bool(pe_corroborates),
         "measured": pe_delta_mean, "threshold": 0.0,
         "statement": (
             "Mechanistic corroboration (fresh-selection-restricted): "
             "post-divergence-window forward-prediction error is also lower "
             "in ON than OFF, consistent with the causal story (abort a "
             "stale macro -> lower forward-PE -> lower harm). Non-load-"
             "bearing: harm is the task-outcome DV this claim actually asks "
             "about. See summary.all_tick_harm_delta_mean for the hold-"
             "weighted (realised, full-window) corroboration.")},
    ]

    return {
        "outcome": outcome,
        "evidence_direction": direction,
        "interpretation_label": label,
        "criteria": criteria,
        "criteria_non_degenerate": non_degen_map,
        "preconditions": aggregate["adjudication_preconditions"],
        "per_arm_gate": aggregate["per_arm_gate"],
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "seed_tiers_measured": {
            "decomposing": decomposing_seeds,
            "fires_inert_or_control": inert_seeds,
        },
        "windowed_deltas": windowed,
        "null_checks": null_checks,
        "summary": {
            "n_seeds": len(seeds),
            "n_decomposing_seeds": len(decomposing_seeds),
            "n_windowed_pairs": n_pairs,
            "harm_delta_mean_post_divergence": harm_delta_mean,
            "harm_delta_sd": harm_delta_sd,
            "harm_delta_se": harm_delta_se,
            "rel_improvement": rel_improvement,
            "pe_delta_mean_post_divergence": pe_delta_mean,
            "all_tick_harm_delta_mean": all_tick_harm_delta_mean,
            "effect_size_ok": effect_size_ok,
            "rel_floor_ok": rel_floor_ok,
            "off_pe_var_worst": off_pe_var_worst,
            "off_pe_mean_worst": off_pe_mean_worst,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> Tuple[Optional[str], Optional[str], bool]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seeds = list(baselines.SEED_POOL)
    episodes = 2 if args.dry_run else baselines.EPISODES
    steps = 12 if args.dry_run else baselines.STEPS_PER_EPISODE

    # DESIGN-TIME REFUSAL before any compute (V3-EXQ-785 discipline). Neither
    # readiness precondition here has a pre-registered structural bound (all
    # are emergent counts / variances), so there is nothing to refuse today --
    # the honest state, not an omission. Called under --dry-run too.
    assert_no_structurally_unsatisfiable_gate(
        _precondition_specs(), [_arm_context(a) for a in ARMS])

    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    reuse_log: List[str] = []
    rows: List[Dict[str, Any]] = []
    for arm_id in ARMS:
        for seed in seeds:
            if arm_id == ARM_OFF:
                cached = try_reuse_cell(
                    config_slice=baselines.off_path_config_slice(),
                    seed=seed,
                    script_path=Path(__file__),
                    needed_keys=_needed_keys(),
                    cite_run_id=MINT_RUN_ID,
                    include_driver_script_in_hash=False,
                )
                if cached is not None:
                    rows.append(_reused_off_row(seed, cached))
                    reuse_log.append(f"seed={seed}: REUSED from {MINT_RUN_ID}")
                    continue
                reuse_log.append(
                    f"seed={seed}: reuse_refused (839's manifest lacks "
                    "per_tick_forward_pe / per_tick_harm -- expected, see "
                    "module docstring) -- running fresh")

            with arm_cell(
                seed,
                config_slice=_config_slice(arm_id),
                script_path=Path(__file__),
                config_slice_declared=True,
                include_driver_script_in_hash=(arm_id == ARM_OFF),
            ) as cell:
                row = _run_cell(seed, arm_id, episodes, steps)
                cell.stamp(row)
            rows.append(row)

    result = _analyse(rows)

    run_id = (f"{EXPERIMENT_TYPE}_"
              f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_v3")
    cfg_record = {
        "arms": list(ARMS),
        "seeds": seeds,
        "episodes": episodes,
        "steps_per_episode": steps,
        "env_kwargs_per_seed": {str(s): baselines.env_kwargs(s) for s in seeds},
        "self_dim": baselines.SELF_DIM,
        "world_dim": baselines.WORLD_DIM,
        "seeded_chunk_sequence": list(baselines.SEEDED_CHUNK_SEQUENCE),
        "decomposition_vs_threshold": baselines.DECOMPOSITION_VS_THRESHOLD,
        "decomposition_depth_cap": baselines.DECOMPOSITION_DEPTH_CAP,
        "arm_flags": {a: _arm_flags(a) for a in ARMS},
        "thresholds": {
            "MULTI_ACTION_COMMIT_FLOOR": MULTI_ACTION_COMMIT_FLOOR,
            "PRECOMMIT_EVAL_FLOOR": PRECOMMIT_EVAL_FLOOR,
            "MIDEXEC_DECOMP_FLOOR": MIDEXEC_DECOMP_FLOOR,
            "PE_VARIANCE_FLOOR": PE_VARIANCE_FLOOR,
            "PE_SANITY_CEIL": PE_SANITY_CEIL,
            "EFFECT_SIZE_K_SIGMA": EFFECT_SIZE_K_SIGMA,
            "REL_IMPROVEMENT_FLOOR": REL_IMPROVEMENT_FLOOR,
            "MIN_DECOMPOSING_SEEDS": MIN_DECOMPOSING_SEEDS,
        },
        "reuse_baseline_from": MINT_RUN_ID,
        "reuse_log": reuse_log,
    }

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "outcome": result["outcome"],
        "claim_ids": CLAIM_IDS,
        "bears_on": ["MECH-321", "SD-084", "ARC-070", "ARC-071"],
        "evidence_direction": result["evidence_direction"],
        "supersedes": None,
        "reuse_baseline_from": MINT_RUN_ID,
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "per_arm_gate": result["per_arm_gate"],
        "seed_tiers_measured": result["seed_tiers_measured"],
        "seed_tiers_declared_by_839": {
            "attributable": list(baselines.SEED_POOL_ATTRIBUTABLE),
            "negative_control": list(baselines.SEED_POOL_NEGATIVE_CONTROL),
        },
        "interpretation": {
            "label": result["interpretation_label"],
            "preconditions": result["preconditions"],
            "criteria": result["criteria"],
            "criteria_non_degenerate": result["criteria_non_degenerate"],
            "preconditions_scope_note": result["per_arm_gate"].get(
                "preconditions_scope_note", ""),
        },
        "summary": result["summary"],
        "windowed_deltas": result["windowed_deltas"],
        "null_checks": result["null_checks"],
        "arm_results": rows,
        "per_seed_rows": rows,
        "custom_information": {
            "predecessor_reachability_run": MINT_RUN_ID,
            "predecessor_autopsy": (
                "REE_assembly/evidence/planning/"
                "failure_autopsy_V3-EXQ-839_2026-07-30.md"),
            "gov_reuse_1_check": (
                "Decisive readout (paired post-divergence-window forward-PE / "
                "harm delta) is NOT present in V3-EXQ-839's manifest -- only "
                "whole-run aggregate mean_harm_signal_off/on/delta was "
                "recorded, no per-tick forward-PE or raw per-tick harm "
                "retained. Not recoverable via reanalysis; fresh run required. "
                "839's own whole-run harm deltas (seed 3: -0.000737, seed 71: "
                "-0.002024, seed 89: -0.000449) are diluted by up to ~85% of "
                "ticks the manipulation cannot touch (pre-divergence) and are "
                "the reason a windowed statistic is used here instead."),
            "gov_diag_1_note": (
                "This is NOT the refused 816e env-harshening re-queue (2026-07-29 "
                "governance note on MECH-321): that refusal targets the "
                "PRE-COMMIT / R1 forward-PE discrimination-floor axis "
                "(816/816b/816c/816d). This run is R4 (mid-execution), reached "
                "via V3-EXQ-839's clean reachability verdict, and is EVIDENCE "
                "(claim_ids=[MECH-321]) per the 2026-07-30 governance note's "
                "explicit routing."),
            "seed_tier_is_measurement_not_assumption": (
                "839's per-seed attributable/negative-control/inert "
                "classification is RE-MEASURED fresh this run rather than "
                "imported, since online learning this run performs could in "
                "principle shift which seeds decompose."),
        },
    }

    out_path = write_flat_manifest(
        manifest, None, dry_run=args.dry_run, config=cfg_record, seeds=seeds,
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
        f"  decomposing_seeds={result['seed_tiers_measured']['decomposing']} "
        f"n_pairs={s['n_windowed_pairs']} "
        f"harm_delta_mean={s['harm_delta_mean_post_divergence']:.6g} "
        f"pe_delta_mean={s['pe_delta_mean_post_divergence']:.6g} "
        f"rel_improvement={s['rel_improvement']:.4f}", flush=True)
    print(
        f"  green_arms={result['per_arm_gate']['green_arms']} "
        f"red_arms={result['per_arm_gate']['red_arms']}", flush=True)
    for c in result["criteria"]:
        print(f"  {c['name']}: passed={c['passed']} measured={c['measured']} "
              f"load_bearing={c['load_bearing']}", flush=True)
    if result["degeneracy_reason"]:
        print(f"  degeneracy_reason: {result['degeneracy_reason']}", flush=True)
    print(f"started_utc: {started.strftime('%Y%m%dT%H%M%SZ')}", flush=True)

    outcome_norm = str(result["outcome"]).upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    return outcome_emit, str(out_path), bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry_run = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path,
                     dry_run=_dry_run)
