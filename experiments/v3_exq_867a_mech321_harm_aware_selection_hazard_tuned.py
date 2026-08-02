#!/opt/local/bin/python3
"""V3-EXQ-867a -- MECH-321 HARM-AWARE SELECTION, HAZARD-DENSITY + AFFECTIVE-
STREAM REDESIGN. Same question as V3-EXQ-867 (does SD-hazard-aware-policy-
decomposition's harm-valence-weighted retiling selection improve TASK OUTCOME
over the abort-mechanism-alone baseline, at matched abort rates?), re-posed
after V3-EXQ-867's own P0 precondition (harm_bias_engages) FAILED on every
seed, both arms.

SUPERSEDES V3-EXQ-867 (environment/config-adequacy fix -- same question, no
change to the mechanism under test; per CLAUDE.md's EXQ lettering convention,
"the scientific question is unchanged but the implementation/config was not
adequate to test it" is exactly the bug-fix case, even though 867's own
verdict was `manipulation_inert_or_unmatched` / `non_contributory` rather
than a scored direction -- precedent: V3-EXQ-847a superseded a self-routed
`substrate_not_ready_requeue` 847 the same way).

WHY 867 FAILED, AND WHY THE CONFIRMED AUTOPSY'S OWN FIX RECOMMENDATION WAS
NOT SUFFICIENT (both established this session, 2026-08-02, by direct
measurement against the live substrate -- see
`experiments/_lib/baselines/sd084_midexec_reachability.py`'s
"HAZARD-TUNED + AFFECTIVE-STREAM overlay" block for the full derivation):

`failure_autopsy_V3-EXQ-867_2026-08-02` (confirmed) diagnosed 867's
`decomp_n_harm_bias_nonzero=0` / `decomp_n_harm_override_fires=0` on every
seed, both arms, as `environment_adequacy_defect`: "the run used the default
SD-084 midexec-reachability baseline env with no scheduled_external_hazard
tuning ... z_harm_a_norm apparently never cleared decomposition_harm_threat_
floor=0.1". The recommended fix was SD-029's scheduled_external_hazard
overlay exactly as validated in V3-EXQ-625c (interval=20, prob=0.7,
adjacent_only=True, hazard_harm 0.2, proximity_harm_scale 0.2).

Applying ONLY that overlay to this baseline (measured this session, seed 29,
full 12-episode/60-step schedule) left decomp_n_harm_bias_nonzero at 0,
because `z_harm_a` was structurally `None` throughout the ENTIRE 844/867
lineage -- `use_affective_harm_stream` (SD-011) defaults `False` and neither
844 nor 867 ever set it, nor ever passed the environment's `harm_obs_a`
channel into `agent.sense(...)` (867's `_run_cell` called
`agent.sense(obs["body_state"], obs["world_state"])`, no harm kwargs at
all). `hippocampal/module.py`'s `z_harm_a_norm = float(...) if (harm_aware
and z_harm_a is not None) else 0.0` (line ~983) therefore read 0.0 on every
tick regardless of environmental hazard density -- no amount of env tuning
alone could have moved this number.

Fixing that alone (measured this session, same seed, same schedule) still
left `decomp_n_harm_bias_nonzero=0`, this time because Stage 1's graded
`harm_bias()` ALSO requires its `harm_penalty` argument
(`_decomposition_harm_penalty` in hippocampal/module.py) to be positive, and
that is read from the residue field's RBF `VALENCE_HARM_DISCRIMINATIVE`
channel (SD-014) -- written ONLY when `valence_harm_enabled=True` AND
`z_harm` (the SEPARATE SD-010 sensory-discriminative channel, gated by
`use_harm_stream`, itself also unset in 844/867) is non-None
(`ree_core/agent.py` ~4497-4509). Direct instrumentation this session (a
monkeypatch around `_decomposition_harm_penalty` reading its raw, pre-clamp
return value) confirmed this read EXACTLY 0.0 on all 96 calls across a full
12-episode run with only the affective-stream fix + hazard overlay applied
-- not a small undertrained value, a hard structural zero from every RBF
center starting and staying at its zero-init default. Stage 2's categorical
override (`select_harm_aware_leaves`) needs only the THRESHOLD variable
(w = threat_scale(z_harm_a_norm)), not `harm_penalty`, so it DID fire in
that intermediate configuration (Stage 1 and Stage 2 are not redundant
checks of the same engagement).

THE THREE-FLAG FIX, CONFIRMED THIS SESSION (seed 11, full 12-episode/60-step
schedule, BOTH arms):
  1. `use_affective_harm_stream=True` + `agent.sense(..., obs_harm_a=
     obs["harm_obs_a"])`.
  2. `use_harm_stream=True` + `agent.sense(..., obs_harm=obs["harm_obs"])`.
  3. `valence_harm_enabled=True`.
  ...on top of the autopsy's own SD-029 hazard-density overlay (env kwargs
  unchanged from its recommendation). Result: OFF arm
  decomp_n_harm_bias_nonzero=0 (correct -- bit-identical no-op,
  decomposition_use_harm_aware_selection=False); ON arm
  decomp_n_harm_bias_nonzero=270, decomp_n_harm_override_fires=89, max
  z_harm_a_norm=0.852 (floor=0.1, ref=0.5 -- the graded ramp saturates for
  most of the run). BOTH arms independently decompose mid-execution on this
  seed (OFF decomp_n_decomposed_midexec=46, ON=40) -- a genuine
  both_decompose seed satisfying the matched-abort-pairing design
  requirement below.

Adding three new encoder modules (AffectiveHarmEncoder, HarmEncoder, plus
the valence-write branch) shifts the RNG draw sequence at agent
construction, so the seed tiers measured for the UNTUNED baseline
(SEED_POOL_ATTRIBUTABLE / SEED_POOL_NEGATIVE_CONTROL in the baseline module)
do NOT carry over -- this run's seed pool
(HAZARD_TUNED_SEED_POOL_ATTRIBUTABLE / _NEGATIVE_CONTROL) was RE-MEASURED
under the corrected config this session (see the baseline module for the
full 10-candidate scan table).

THE QUESTION (UNCHANGED FROM 867/844): with MECH-321's mid-execution abort
mechanism enabled in BOTH arms (844's own axis held fixed), does additionally
turning on harm-aware selection over the redecomposition's candidate
re-tilings (SD-hazard-aware-policy-decomposition) improve the SAME paired,
post-divergence-window, fresh-selection-restricted task-outcome (harm)
statistic 844 used to test the abort mechanism alone? C1 below is the
load-bearing criterion, reused verbatim from 844/867.

ARMS -- BOTH REPRODUCE THE ABORT MECHANISM ON, PLUS THE THREE STREAM FLAGS
(substrate-stack preconditions, identical in both arms, NOT the
manipulation); THE ONLY NEW MANIPULATION IS HARM-AWARE SELECTION
    ARM_SELECTION_OFF: hazard-tuned env + abort mechanism ON + the three
        stream flags, harm-aware-selection flag left at its off-by-default
        value. NOT bit-identical to 867's or 844's OFF/ON-mechanism cells
        (env and substrate stack both changed) -- this is a NEW baseline
        mint for this lineage, not a reuse of 844's ARM_HANDLE_ON.
    ARM_SELECTION_ON: identical to ARM_SELECTION_OFF plus
        `decomposition_use_harm_aware_selection=True` at the SD doc's
        recommended defaults (unchanged from 867):
        `decomposition_harm_bias_gain=0.1`, `decomposition_harm_bias_scale=
        0.1`, `decomposition_harm_threat_floor=0.1`,
        `decomposition_harm_threat_ref=0.5`,
        `decomposition_harm_override_w_threshold=0.9`.

NO ARM-REUSE THIS RUN (env + substrate stack both changed vs 844/867, so
867's `MINT_RUN_ID` cite would always refuse on fingerprint mismatch --
omitted rather than left in as dead code). Per GOV-REUSE-1/mint-as-you-go,
THIS run's own ARM_SELECTION_OFF cells are minted reuse-eligible
(`include_driver_script_in_hash=False`) as the canonical baseline for this
NEW hazard-tuned + affective-stream MECH-321 lineage, so a future successor
(867b or a sibling driver) can hit against them.

C1 STATISTIC -- REUSED VERBATIM FROM V3-EXQ-844/867, NOT RE-DERIVED. Same
`_windowed_delta` / `_first_divergence_tick` / `_mean_at` /
fresh-selection-restricted-vs-all-tick split / `EFFECT_SIZE_K_SIGMA=1.0` /
`REL_IMPROVEMENT_FLOOR=0.0` / `MIN_DECOMPOSING_SEEDS=2`. Load-bearing DV =
paired per-seed windowed mean harm delta (OFF minus ON restricted to fresh
e3-selection ticks), positive = ARM_SELECTION_ON is less harmful.

MATCHED ABORT RATES -- REQUIRED DESIGN ELEMENT, HANDLED TWO WAYS (unchanged
from 867): (a) covariate reporting of decomp_n_decomposed_midexec per seed,
non-gating; (b) load-bearing C1 restricted to `both_decompose_seeds` (both
arms independently decompose mid-execution this run).

SEED TIERING -- FOUR-WAY, RE-MEASURED THIS RUN (unchanged shape from 867,
new pool): both_decompose (load-bearing C1) / on_only_decompose /
off_only_decompose (both non-gating) / neither_decompose (negative control,
`_null_check_delta`).

HOLD-WEIGHTED-READOUT TRIAGE, DV-SYMMETRY DECLARATION, READINESS (P0) --
ALL UNCHANGED FROM 867/844 IN SHAPE, one addition: `harm_bias_engages` is
now genuinely satisfiable (confirmed this session; see above) rather than
structurally unreachable as it was for 867. A below-floor reading on THIS
corrected config would mean the substrate genuinely is not ready for a
FURTHER reason beyond the three already fixed -- self-routes
`substrate_not_ready_requeue`, never a C1 verdict on an inert treatment.

    P_MULTI  multi_action_commits_present (existential, both arms).
    P_PRECOMMIT  decomposition_precommit_live (existential, both arms).
    P_MIDEXEC_DECOMP  midexec_decomposition_occurs (existential, both arms).
    P_HARM_BIAS_ENGAGES  harm_bias_engages (existential, ARM_SELECTION_ON
        only) -- decomp_n_harm_bias_nonzero > 0. Now genuinely reachable
        (confirmed seed 11: 270 this run) rather than structurally
        unreachable as in 867.
    P_PE_VARIES / P_PE_BOUNDED  OFF-arm forward-PE positive control,
        unchanged from 867/844/816.

SLEEP DRIVER: not applicable -- no sleep phase entered in this run.

Z_GOAL: deliberately inert, carried over verbatim from 844/839/867 for the
identical reason (`REEConfig.from_dims(z_goal_enabled=...)` defaults False).
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
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
import experiments._lib.baselines.sd084_midexec_reachability as baselines  # noqa: E402

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.policy import ChunkedPrimitive, ChunkState  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_867a_mech321_harm_aware_selection_hazard_tuned"
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["MECH-321"]
QUEUE_ID = "V3-EXQ-867a"
SUPERSEDES: Optional[str] = "V3-EXQ-867"

ARM_OFF = "ARM_SELECTION_OFF"
ARM_ON = "ARM_SELECTION_ON"
ARMS = (ARM_OFF, ARM_ON)

# --- Harm-aware-selection parameters. Unchanged from 867 -- verified against
# the PolicyDecompositionConfig dataclass defaults AND the REEConfig.from_dims
# mirror defaults. No documented reason to deviate for this redesign. ---
HARM_BIAS_GAIN = 0.1
HARM_BIAS_SCALE = 0.1
HARM_THREAT_FLOOR = 0.1
HARM_THREAT_REF = 0.5
HARM_OVERRIDE_W_THRESHOLD = 0.9

# CONFIG_SLICE_DECLARATION_EXEMPT: validate_experiments.py's static resolver
# cannot trace `_config_slice(arm_id)`'s conditional dispatch to
# `_on_selection_config_slice()`, so it flags these five constants as
# "omitted." They are NOT omitted -- `_on_selection_config_slice()` declares
# each one under its REEConfig-mirror key, which is what `_arm_flags()`/
# `REEConfig.from_dims` actually reads downstream (identical reasoning to
# 867's own exemption note).

# --- Pre-registered thresholds. Constants, never derived from this run's own
# statistics (mirrors 844/839/867/816 convention). ---
MULTI_ACTION_COMMIT_FLOOR = 0.0
PRECOMMIT_EVAL_FLOOR = 0.0
MIDEXEC_DECOMP_FLOOR = 0.0
HARM_BIAS_ENGAGE_FLOOR = 0.0
PE_VARIANCE_FLOOR = 1e-12
PE_SANITY_CEIL = 1e6

EFFECT_SIZE_K_SIGMA = 1.0
REL_IMPROVEMENT_FLOOR = 0.0
MIN_DECOMPOSING_SEEDS = 2

_ZGOAL = ZGoalStreamAccumulator()


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------
def _off_selection_config_slice() -> Dict[str, Any]:
    """ARM_SELECTION_OFF's fingerprint config slice -- the canonical baseline
    for THIS hazard-tuned + affective-stream lineage (mint-as-you-go; no
    predecessor to reuse from, see module docstring "NO ARM-REUSE")."""
    slice_: Dict[str, Any] = {
        "env": dict(baselines.HAZARD_TUNED_ENV_OVERLAY),
        "env_seeded_per_cell": baselines.ENV_SEEDED_PER_CELL,
        "schedule": {
            "episodes": baselines.EPISODES,
            "steps_per_episode": baselines.STEPS_PER_EPISODE,
        },
        "self_dim": baselines.SELF_DIM,
        "world_dim": baselines.WORLD_DIM,
        "seeded_chunk_sequence": list(baselines.SEEDED_CHUNK_SEQUENCE),
        "seeded_chunk_depth": baselines.SEEDED_CHUNK_DEPTH,
        "seeded_chunk_selection_weight": baselines.SEEDED_CHUNK_SELECTION_WEIGHT,
    }
    slice_.update(baselines.on_arm_flags())  # abort mechanism ON
    slice_.update(baselines.HAZARD_TUNED_STREAM_FLAGS)
    return slice_


def _on_selection_config_slice() -> Dict[str, Any]:
    slice_ = _off_selection_config_slice()
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
    flags.update(baselines.HAZARD_TUNED_STREAM_FLAGS)  # substrate-stack precondition, BOTH arms
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


def _config_slice(arm_id: str) -> Dict[str, Any]:
    return _off_selection_config_slice() if arm_id == ARM_OFF else _on_selection_config_slice()


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
# One cell
# ---------------------------------------------------------------------------
def _run_cell(seed: int, arm_id: str, episodes: int, steps: int,
              quiet: bool = False) -> Dict[str, Any]:
    """Drive the canonical loop for one (seed, arm) cell. Identical discipline
    to 844/839/867's `_run_cell`, PLUS the fix this redesign is about: `obs_
    harm`/`obs_harm_a` are now read from the env's obs_dict and passed into
    `agent.sense(...)` every tick (867's version called `agent.sense(obs
    ["body_state"], obs["world_state"])` with NO harm channels at all -- the
    root cause `z_harm_a` was structurally None throughout that lineage)."""
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
        "mean_harm_signal": (
            statistics.fmean(harm_ticks) if harm_ticks else 0.0),
        "arm_flags": dict(flags),
    }

    cell_pass = bool(
        multi_action_commits > 0 and row["decomp_n_evaluated_midexec"] > 0)
    row["cell_pass"] = cell_pass
    if not quiet:
        print(f"verdict: {'PASS' if cell_pass else 'FAIL'}", flush=True)
    return row


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------
def _arm_context(arm_id: str) -> Dict[str, Any]:
    return {"id": arm_id, "arm_id": arm_id, "harm_aware_on": arm_id == ARM_ON}


def _precondition_specs() -> List[PreconditionSpec]:
    return [
        PreconditionSpec(
            name="multi_action_commits_present",
            description=(
                "At least one cell of this ARM committed a MULTI-ACTION "
                "program. Gate (6) of the mid-execution conjunction requires "
                "len(remaining) > 1 -- carried over from 839/844/867's "
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
                "Pre-commit decomposition evaluated at least once -- a zero "
                "means the decomposition machinery is not running at all."
            ),
            control="BEST cell of this arm (positive control)",
            threshold=PRECOMMIT_EVAL_FLOOR,
            direction="lower",
            kind="readiness",
        ),
        PreconditionSpec(
            name="midexec_decomposition_occurs",
            description=(
                "At least one cell of this ARM shows decomp_n_decomposed_"
                "midexec > 0 THIS run. Both arms have the abort mechanism "
                "enabled here, so this applies to BOTH arms."
            ),
            control="BEST cell of this arm (max)",
            threshold=MIDEXEC_DECOMP_FLOOR,
            direction="lower",
            kind="readiness",
        ),
        PreconditionSpec(
            name="harm_bias_engages",
            description=(
                "ARM_SELECTION_ON's Stage-1 graded harm bias actually "
                "engages at least once this run (decomp_n_harm_bias_nonzero "
                "> 0). Genuinely reachable in this redesign (unlike 867 -- "
                "see module docstring for the three-flag fix); a below-"
                "floor reading here would mean the substrate is not ready "
                "for a FURTHER reason -- substrate_not_ready_requeue, never "
                "a C1 verdict on an inert treatment. EXISTENTIAL, ON arm "
                "only."
            ),
            control="BEST ON-arm cell (max)",
            threshold=HARM_BIAS_ENGAGE_FLOOR,
            direction="lower",
            kind="readiness",
            applies_to=lambda ctx: ctx["arm_id"] == ARM_ON,
        ),
        PreconditionSpec(
            name="off_forward_pe_varies",
            description=(
                "OFF-arm positive control: committed-trajectory forward PE "
                "must have non-zero variance across the whole run. Mirrors "
                "816/839/844/867's P3."
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
# Windowed per-seed paired delta -- REUSED VERBATIM FROM V3-EXQ-844/867
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
        return (on_m - off_m) if (off_m is not None and on_m is not None) else None

    def _pe_delta(off_m, on_m):
        return (off_m - on_m) if (off_m is not None and on_m is not None) else None

    return {
        "seed": off_row["seed"],
        "divergence_tick": div_tick,
        "n_window_ticks": len(off_harm) - div_tick,
        "n_window_ticks_fresh": min(n_off_harm_fresh, n_on_harm_fresh),
        "off_harm_window_mean_all": off_harm_all,
        "on_harm_window_mean_all": on_harm_all,
        "harm_delta_all_off_minus_on": _harm_delta(off_harm_all, on_harm_all),
        "off_pe_window_mean_all": off_pe_all,
        "on_pe_window_mean_all": on_pe_all,
        "pe_delta_all_off_minus_on": _pe_delta(off_pe_all, on_pe_all),
        "off_harm_window_mean_fresh": off_harm_fresh_mean,
        "on_harm_window_mean_fresh": on_harm_fresh_mean,
        "harm_delta_fresh_off_minus_on": _harm_delta(off_harm_fresh_mean, on_harm_fresh_mean),
        "off_pe_window_mean_fresh": off_pe_fresh,
        "on_pe_window_mean_fresh": on_pe_fresh,
        "pe_delta_fresh_off_minus_on": _pe_delta(off_pe_fresh, on_pe_fresh),
    }


def _null_check_delta(off_row: Dict[str, Any], on_row: Dict[str, Any]) -> Dict[str, Any]:
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


def _abort_rate_covariate(off_row: Dict[str, Any], on_row: Dict[str, Any]) -> Dict[str, Any]:
    off_n = off_row["decomp_n_decomposed_midexec"]
    on_n = on_row["decomp_n_decomposed_midexec"]
    return {
        "seed": off_row["seed"],
        "off_decomp_n_decomposed_midexec": off_n,
        "on_decomp_n_decomposed_midexec": on_n,
        "abort_count_delta_on_minus_off": on_n - off_n,
        "on_harm_bias_nonzero_count": on_row.get("decomp_n_harm_bias_nonzero", 0),
        "on_harm_override_fires_count": on_row.get("decomp_n_harm_override_fires", 0),
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
            "midexec_decomposition_occurs": _best_cell(arm_rows, "decomp_n_decomposed_midexec"),
        }
        if arm_id == ARM_ON:
            measured["harm_bias_engages"] = _best_cell(arm_rows, "decomp_n_harm_bias_nonzero")
        if arm_id == ARM_OFF:
            measured["off_forward_pe_varies"] = off_pe_var_worst
            measured["off_forward_pe_bounded"] = off_pe_mean_worst
        gate = evaluate_arm_gate(arm_id, arm_contexts[arm_id], specs, measured=measured)
        arm_gates.append(gate)

    aggregate = aggregate_arm_gates(arm_gates)

    seeds = sorted(set(off_by_seed) & set(on_by_seed))
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

    windowed: List[Dict[str, Any]] = []
    for s in both_decompose_seeds:
        div_tick = _first_divergence_tick(off_by_seed[s], on_by_seed[s])
        if div_tick is not None:
            windowed.append(_windowed_delta(off_by_seed[s], on_by_seed[s], div_tick))

    unmatched_windowed: List[Dict[str, Any]] = []
    for s in on_only_seeds + off_only_seeds:
        div_tick = _first_divergence_tick(off_by_seed[s], on_by_seed[s])
        if div_tick is not None:
            w = _windowed_delta(off_by_seed[s], on_by_seed[s], div_tick)
            w["unmatched_tier"] = "on_only" if s in on_only_seeds else "off_only"
            unmatched_windowed.append(w)

    null_checks = [_null_check_delta(off_by_seed[s], on_by_seed[s]) for s in neither_decompose_seeds]
    abort_rate_covariate = [_abort_rate_covariate(off_by_seed[s], on_by_seed[s]) for s in seeds]

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

    pe_deltas = [w["pe_delta_fresh_off_minus_on"] for w in windowed
                 if w["pe_delta_fresh_off_minus_on"] is not None]
    pe_delta_mean = statistics.fmean(pe_deltas) if pe_deltas else 0.0
    pe_corroborates = bool(pe_deltas) and pe_delta_mean > 0.0

    all_tick_harm_deltas = [w["harm_delta_all_off_minus_on"] for w in windowed
                            if w["harm_delta_all_off_minus_on"] is not None]
    all_tick_harm_delta_mean = (
        statistics.fmean(all_tick_harm_deltas) if all_tick_harm_deltas else 0.0)

    enough_pairs = n_pairs >= min(MIN_DECOMPOSING_SEEDS, len(both_decompose_seeds) or 1)
    effect_size_ok = enough_pairs and (harm_delta_mean > EFFECT_SIZE_K_SIGMA * harm_delta_se)
    rel_floor_ok = rel_improvement >= REL_IMPROVEMENT_FLOOR
    c1_task_outcome_improves = bool(
        aggregate["non_degenerate"] and enough_pairs and effect_size_ok and rel_floor_ok)

    any_divergence = len(windowed) > 0
    non_degenerate = bool(aggregate["non_degenerate"] and any_divergence)

    if not aggregate["non_degenerate"]:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        direction = "unknown"
        degeneracy_reason = aggregate["degeneracy_reason"]
    elif not any_divergence:
        label = "manipulation_inert_or_unmatched"
        outcome = "FAIL"
        direction = "unknown"
        degeneracy_reason = (
            "readiness met but no seed showed BOTH arms decomposing mid-"
            "execution AND a resulting action-sequence divergence between "
            "them")
    elif not enough_pairs:
        label = "harm_aware_selection_task_effect_underpowered"
        outcome = "FAIL"
        direction = "unknown"
        degeneracy_reason = (
            f"only {n_pairs} matched-abort-pair(s) with a measurable window "
            f"(< {MIN_DECOMPOSING_SEEDS}); the seed pool is a substrate "
            "measurement (no chunk-selection-bias knob), not a free "
            "parameter -- see baseline module docstring")
    else:
        degeneracy_reason = None
        if c1_task_outcome_improves:
            label = "harm_aware_selection_reduces_task_harm"
            direction = "supports"
        elif harm_delta_mean <= 0:
            label = "harm_aware_selection_does_not_reduce_task_harm"
            direction = "weakens"
        else:
            label = "harm_aware_selection_task_effect_below_threshold"
            direction = "mixed"
        outcome = "PASS" if c1_task_outcome_improves else "FAIL"

    non_degen_map = {
        "C1_TASK_OUTCOME_IMPROVES": non_degenerate,
        "C2_FORWARD_PE_CORROBORATES": non_degenerate,
    }

    criteria = [
        {"name": "C1_TASK_OUTCOME_IMPROVES", "load_bearing": True,
         "passed": bool(c1_task_outcome_improves),
         "measured": harm_delta_mean, "threshold": 0.0,
         "statement": (
             "On seeds where mid-execution decomposition fired in BOTH arms "
             "(matched-abort-rate pairing), the paired post-divergence-"
             "window mean harm signal, restricted to FRESH e3-selection "
             "ticks, is LESS negative (less harmful) in ARM_SELECTION_ON "
             "than ARM_SELECTION_OFF, by an effect exceeding "
             f"{EFFECT_SIZE_K_SIGMA} x SE over >= {MIN_DECOMPOSING_SEEDS} "
             "paired seeds.")},
        {"name": "C2_FORWARD_PE_CORROBORATES", "load_bearing": False,
         "passed": bool(pe_corroborates),
         "measured": pe_delta_mean, "threshold": 0.0,
         "statement": (
             "Consistency check (non-load-bearing, carried over from 844's "
             "established finding): on the SAME matched-abort-pair window, "
             "forward-prediction error is also lower in ON than OFF.")},
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
            "both_decompose": both_decompose_seeds,
            "on_only_decompose": on_only_seeds,
            "off_only_decompose": off_only_seeds,
            "neither_decompose": neither_decompose_seeds,
        },
        "windowed_deltas": windowed,
        "unmatched_windowed_deltas": unmatched_windowed,
        "null_checks": null_checks,
        "abort_rate_covariate": abort_rate_covariate,
        "summary": {
            "n_seeds": len(seeds),
            "n_both_decompose_seeds": len(both_decompose_seeds),
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
            "abort_rate_matched_seed_count": len(both_decompose_seeds),
            "abort_rate_unmatched_seed_count": len(on_only_seeds) + len(off_only_seeds),
            "max_z_harm_a_norm_on_arm": _best_cell(on_rows, "max_z_harm_a_norm"),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> Tuple[Optional[str], Optional[str], bool]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seeds = list(baselines.HAZARD_TUNED_SEED_POOL)
    episodes = 2 if args.dry_run else baselines.EPISODES
    steps = 12 if args.dry_run else baselines.STEPS_PER_EPISODE

    assert_no_structurally_unsatisfiable_gate(
        _precondition_specs(), [_arm_context(a) for a in ARMS])

    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    rows: List[Dict[str, Any]] = []
    for arm_id in ARMS:
        for seed in seeds:
            with arm_cell(
                seed,
                config_slice=_config_slice(arm_id),
                script_path=Path(__file__),
                config_slice_declared=True,
                include_driver_script_in_hash=False,
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
        "env_kwargs_per_seed": {str(s): baselines.env_kwargs_hazard_tuned(s) for s in seeds},
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
            "MULTI_ACTION_COMMIT_FLOOR": MULTI_ACTION_COMMIT_FLOOR,
            "PRECOMMIT_EVAL_FLOOR": PRECOMMIT_EVAL_FLOOR,
            "MIDEXEC_DECOMP_FLOOR": MIDEXEC_DECOMP_FLOOR,
            "HARM_BIAS_ENGAGE_FLOOR": HARM_BIAS_ENGAGE_FLOOR,
            "PE_VARIANCE_FLOOR": PE_VARIANCE_FLOOR,
            "PE_SANITY_CEIL": PE_SANITY_CEIL,
            "EFFECT_SIZE_K_SIGMA": EFFECT_SIZE_K_SIGMA,
            "REL_IMPROVEMENT_FLOOR": REL_IMPROVEMENT_FLOOR,
            "MIN_DECOMPOSING_SEEDS": MIN_DECOMPOSING_SEEDS,
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
        "supersedes": SUPERSEDES,
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "per_arm_gate": result["per_arm_gate"],
        "seed_tiers_measured": result["seed_tiers_measured"],
        "seed_tiers_declared_by_baseline": {
            "attributable": list(baselines.HAZARD_TUNED_SEED_POOL_ATTRIBUTABLE),
            "negative_control": list(baselines.HAZARD_TUNED_SEED_POOL_NEGATIVE_CONTROL),
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
        "unmatched_windowed_deltas": result["unmatched_windowed_deltas"],
        "null_checks": result["null_checks"],
        "abort_rate_covariate": result["abort_rate_covariate"],
        "arm_results": rows,
        "per_seed_rows": rows,
        "custom_information": {
            "predecessor_run_supersedes": SUPERSEDES,
            "predecessor_autopsy": (
                "REE_assembly/evidence/planning/"
                "failure_autopsy_V3-EXQ-867_2026-08-02.md"),
            "substrate_doc": (
                "REE_assembly/docs/architecture/"
                "sd_hazard_aware_policy_decomposition.md"),
            "three_flag_fix_note": (
                "867's autopsy diagnosed ONLY the missing SD-029 hazard-"
                "density env overlay. This session's direct measurement "
                "found that overlay alone insufficient -- z_harm_a was "
                "structurally None (use_affective_harm_stream unset + "
                "obs_harm_a never wired into agent.sense()), and even after "
                "fixing that, harm_penalty read a hard 0.0 on all 96 calls "
                "in a full-schedule probe because the residue field's "
                "VALENCE_HARM_DISCRIMINATIVE channel is written only under "
                "valence_harm_enabled+use_harm_stream (both also unset). "
                "All three plus the env overlay are applied here; confirmed "
                "this session on seed 11 at full schedule: ON arm "
                "decomp_n_harm_bias_nonzero=270, max z_harm_a_norm=0.852, "
                "OFF arm correctly 0 (bit-identical no-op)."),
            "matched_abort_rates_design_note": (
                "Unchanged from 867: (a) per-seed decomp_n_decomposed_"
                "midexec covariate reported for every seed; (b) load-"
                "bearing C1 restricted to both_decompose_seeds."),
            "gov_reuse_1_note": (
                "No compatible prior mint exists for this env+substrate-"
                "stack combination (both changed vs 844/867) -- this run's "
                "own ARM_SELECTION_OFF cells are minted reuse-eligible as "
                "the canonical baseline for this new lineage."),
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
        f"  both_decompose_seeds={result['seed_tiers_measured']['both_decompose']} "
        f"n_pairs={s['n_windowed_pairs']} "
        f"harm_delta_mean={s['harm_delta_mean_post_divergence']:.6g} "
        f"pe_delta_mean={s['pe_delta_mean_post_divergence']:.6g} "
        f"rel_improvement={s['rel_improvement']:.4f} "
        f"max_z_harm_a_norm_on_arm={s['max_z_harm_a_norm_on_arm']:.4f}", flush=True)
    print(
        f"  on_only_seeds={result['seed_tiers_measured']['on_only_decompose']} "
        f"off_only_seeds={result['seed_tiers_measured']['off_only_decompose']} "
        f"(unmatched abort-rate tiers, non-gating)", flush=True)
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
