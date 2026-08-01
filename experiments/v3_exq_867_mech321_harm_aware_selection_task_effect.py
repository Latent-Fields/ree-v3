#!/opt/local/bin/python3
"""V3-EXQ-867 -- MECH-321 HARM-AWARE SELECTION: does SD-hazard-aware-policy-
decomposition's harm-valence-weighted retiling selection improve TASK OUTCOME
over the abort-mechanism-alone baseline, at matched abort rates?

EVIDENCE. `claim_ids=["MECH-321"]`. Successor to V3-EXQ-844 (confirmed
`failure_autopsy_V3-EXQ-844_2026-08-01.md`), which tested MECH-321's
mid-execution abort mechanism on the SAME C1 task-outcome (harm) statistic
used here: C2 (mechanistic corroboration -- abort lowers forward-PE) PASSED
(+0.000247); C1 (load-bearing -- abort reduces task harm) FAILED (-0.003262,
wrong direction). Code-verified root cause (844 autopsy): `_apply_policy_
decomposition` / `PolicyDecomposition.evaluate()`/`decompose_sequence()` read
no harm-valence signal and perform NO ranked selection among candidate
re-tilings at all -- every surviving leaf tile is additively recombined
blind, so the abort mechanism has nothing downstream biasing the replan
toward a lower-harm continuation.

SD-hazard-aware-policy-decomposition (implemented 2026-08-01, chip-
20260801-hazard-aware-decomp-build) closes exactly that gap: `PolicyDecomp-
osition.harm_bias()` (Stage 1, graded, always-active additive score_bias
term) + `.select_harm_aware_leaves()` (Stage 2, categorical, threshold-gated
pool-admission override restricting a withheld chunk's leaf tiles to the
single lowest-harm-penalty one). Both are bit-identical no-ops when
`decomposition_use_harm_aware_selection=False` (the pre-existing default;
confirmed by the SD's own module docstring + 32 pre-existing contracts
passing unmodified + 11 new C18-C24 contracts). Both claims.yaml (line
~42091-42096) and the SD doc's "What This SD Enables" section name this
exact follow-on: "a MECH-321 behavioural-effect validation experiment
measuring whether harm-aware selection improves V3-EXQ-844's C1 task-outcome
statistic, ON vs OFF, at matched abort rates." This is that experiment.

THE QUESTION
    With MECH-321's mid-execution abort mechanism enabled in BOTH arms (the
    844 axis is held fixed here, not re-tested), does additionally turning on
    harm-aware selection over the redecomposition's candidate re-tilings
    (SD-hazard-aware-policy-decomposition) improve the SAME paired,
    post-divergence-window, fresh-selection-restricted task-outcome (harm)
    statistic 844 used to test the abort mechanism alone? C1 below is the
    load-bearing criterion.

NOT A RE-TEST OF THE 816-FAMILY OR OF 844's OWN AXIS. claims.yaml's MECH-321
GOV-DIAG-1 recurrence note distinguishes the refused 816e pre-commit/R1
env-harshening re-pose from 844's mid-execution task-outcome axis. This run
is neither: it holds 844's own manipulation (`use_persistent_committed_
program_handle=True`, i.e. abort mechanism ON) FIXED IN BOTH ARMS and adds a
genuinely NEW manipulated variable on top of it (`decomposition_use_harm_
aware_selection`). Re-derive brake check (Step 2.5b, this session): 0
autopsies count a substrate_ceiling/non_contributory hit against MECH-321 at
this granularity -- brake does not fire.

ARMS -- BOTH REPRODUCE V3-EXQ-844's ARM_HANDLE_ON (abort mechanism ON);
THE ONLY NEW MANIPULATION IS HARM-AWARE SELECTION
    ARM_SELECTION_OFF: `experiments/_lib/baselines/sd084_midexec_
        reachability.py`'s `on_arm_flags()` (abort mechanism ON) with the new
        harm-aware-selection flag left at its off-by-default value (undeclared
        in the config slice, exactly mirroring how 844 -- which predates the
        flag's existence -- never declared it either). This cell is
        BIT-IDENTICAL to 844's own ARM_HANDLE_ON cells (same env, schedule,
        substrate stack, seeds, manipulation), so it is constructed to HIT
        against 844's own reuse-eligible ARM_HANDLE_ON mint (see REUSE below)
        rather than re-running compute 844 already produced.
    ARM_SELECTION_ON: identical to ARM_SELECTION_OFF plus
        `decomposition_use_harm_aware_selection=True` at the SD doc's
        recommended (and `PolicyDecompositionConfig` dataclass / `REEConfig.
        from_dims` default) parameter values -- confirmed by direct source
        read of `ree_core/agent.py:3963-3968,5939-5944,7178-7186` (REEConfig
        mirror fields, matching the dataclass defaults exactly, no
        REEConfig-vs-dataclass naming mismatch): `decomposition_harm_bias_
        gain=0.1`, `decomposition_harm_bias_scale=0.1`, `decomposition_harm_
        threat_floor=0.1`, `decomposition_harm_threat_ref=0.5`,
        `decomposition_harm_override_w_threshold=0.9`. No documented reason
        on record to deviate from these defaults for a first validation run.

REUSE (GOV-REUSE-1 / arm-reuse Phase 1): ARM_SELECTION_OFF cells are
    CONSTRUCTED from the SAME canonical lineage baseline module 844 itself
    used, with the SAME config slice 844's `_config_slice(ARM_ON)` computed
    (`off_path_config_slice()` updated by `on_arm_flags()`, no harm-aware-
    selection keys declared). 844's own ARM_HANDLE_ON cells were minted
    reuse-eligible (`include_driver_script_in_hash=False`; verified directly
    against the manifest: `arm_fingerprint.reuse_eligible=True` on every
    ARM_HANDLE_ON row) precisely so a bit-identical successor arm could skip
    re-running them -- this run is that successor. `try_reuse_cell` is called
    per seed with `cite_run_id=<844's run_id>` and the full needed-key list
    this run reads back (`per_tick_forward_pe`, `per_tick_harm`,
    `action_sequence`, `e3_tick_flags`, decomposition counters,
    `multi_action_commits`, `n_ticks`) -- all present in 844's manifest.
    Reuse decision (HIT vs REFUSE, e.g. on a machine-class mismatch between
    this run's worker and 844's) is logged per seed in `reuse_log` in the
    manifest config, exactly as 844 logged its own reuse attempt against 839.
    A refused cell runs fresh via the normal per-cell path and is ITSELF
    re-minted reuse-eligible (`include_driver_script_in_hash=False`,
    mint-as-you-go) so a future third consumer can hit against either mint.
    ARM_SELECTION_ON always runs fresh (it is this run's own new manipulation)
    and is likewise minted reuse-eligible.

C1 STATISTIC -- REUSED VERBATIM FROM V3-EXQ-844, NOT RE-DERIVED
    Per instruction, this run reuses 844's exact statistic-construction logic
    (`_windowed_delta`, `_first_divergence_tick`, `_mean_at`, the fresh-
    selection-restricted vs all-tick split, `EFFECT_SIZE_K_SIGMA=1.0`,
    `REL_IMPROVEMENT_FLOOR=0.0`, `MIN_DECOMPOSING_SEEDS=2`) applied to
    ARM_SELECTION_OFF vs ARM_SELECTION_ON in place of 844's ARM_HANDLE_OFF vs
    ARM_HANDLE_ON. Load-bearing DV = paired per-seed windowed mean harm delta
    (OFF minus ON restricted to fresh e3-selection ticks), positive =
    ARM_SELECTION_ON is less harmful.

MATCHED ABORT RATES -- REQUIRED DESIGN ELEMENT, HANDLED TWO WAYS
    Harm-aware selection could in principle change HOW OFTEN mid-execution
    decomposition/abort fires (not just what happens after it fires) --
    Stage 2's categorical override only restricts WHICH leaf tiles survive an
    ALREADY-DECOMPOSED chunk, and does not itself gate the V_s-threshold
    decompose/keep test in `decompose_sequence()` (code-verified: `_apply_
    policy_decomposition`'s existential trigger is unchanged by this SD), but
    the selected replan can still shift the agent's SUBSEQUENT trajectory and
    therefore its future V_s readings, so an INDIRECT abort-rate shift is a
    real possible confound, not a structural impossibility to rule out by
    code inspection alone.
    (a) COVARIATE REPORTING (non-gating): every seed's `decomp_n_decomposed_
        midexec` count is recorded per arm, and the per-seed delta
        (ON minus OFF) is reported in `summary.abort_rate_covariate` --
        exactly the "matched abort rates" check as an explicit, non-gating,
        auditable readout, mirroring 844's own C1/C2 load-bearing/non-load-
        bearing split style.
    (b) MATCHED PAIRING (structural): the load-bearing C1 windowed comparison
        is restricted to seeds where BOTH arms independently show `decomp_n_
        decomposed_midexec > 0` this run (`both_decompose_seeds`) -- i.e. an
        abort genuinely occurred in BOTH arms on that seed, so the measured
        delta reflects SELECTION QUALITY among candidates the abort already
        produced, not whether an abort happened at all. Seeds where only ONE
        arm decomposed (`on_only_seeds` / `off_only_seeds`) are reported
        SEPARATELY and non-gating -- informative about an abort-RATE effect,
        but not mixed into the load-bearing selection-QUALITY comparison.
        Seeds where NEITHER arm decomposed (`neither_decompose_seeds`) are
        the negative control, handled exactly as 844's `_null_check_delta`.

SEED TIERING -- FOUR-WAY, RE-MEASURED THIS RUN (not imported from 844)
    Because BOTH arms have the abort mechanism enabled here (unlike 844,
    where OFF structurally could not decompose), this run's tiering is
    four-way rather than 844's two-way (decomposing / fires-inert):
      both_decompose      -- carries load-bearing C1 (matched abort pairing).
      on_only_decompose    -- non-gating: ON aborted, OFF (same abort
          mechanism, no harm-aware selection) did not on this seed. Possible
          abort-rate-shift signature; reported, not gated.
      off_only_decompose   -- non-gating: OFF aborted, ON did not. Also a
          possible abort-rate-shift signature, opposite direction.
      neither_decompose    -- negative control: predicted NULL (whole-run
          means bit-identical between arms, since neither arm ever released
          the commit latch mid-execution on this seed and harm-aware
          selection cannot act on a decomposition that never happened).

HOLD-WEIGHTED-READOUT TRIAGE (validate_experiments advisory), FIX APPLIED,
CARRIED OVER FROM V3-EXQ-844 UNCHANGED
    Both `per_tick_harm` and `per_tick_forward_pe` accumulate once per ENV
    STEP, and `agent.py` returns the HELD action (and `generate_trajectories`
    returns CACHED candidates) on a non-e3-tick before `e3.select()` is
    reached (cadence `e3_steps_per_tick` defaults 10) -- this driver loop is
    unchanged from 844's, so the same hold-weighting confound applies
    identically. `e3_tick_flags` is recorded per tick; C1 (load-bearing) uses
    the FRESH-SELECTION-RESTRICTED delta (hold-weighting confound removed);
    the all-tick (hold-weighted) delta is retained as a NON-gating
    corroborating readout of realised task outcome over the full physical
    window. A seed's empty fresh-tick sample yields `None` and does not enter
    C1 -- underpowered, not silently substituted.

DV-SYMMETRY, DECLARED PER THE 604c RUBRIC
    Load-bearing DV = paired per-seed windowed mean harm delta (a symmetric
    mean over ticks). The manipulation under test here
    (`decomposition_use_harm_aware_selection`) is NOT invariant under this
    symmetry, and specifically is NOT a broadcast-additive-constant case: (1)
    Stage 1's `harm_bias(tile)` is `w(h) * harm_penalty(tile)`, a term that
    varies PER CANDIDATE TILE (each tile's own predicted world_states feed
    its own harm_penalty read) -- not a uniform constant added to every
    candidate, so it does not cancel in an argmax/softmax selection the way a
    broadcast term would. (2) Stage 2's categorical override directly changes
    candidate SET MEMBERSHIP (pool admission, not a score adjustment), which
    is definitionally not invariant under any selection rule. Both stages can
    therefore genuinely move which action is selected at and after a
    redecomposition tick, changing the per-tick harm and forward-PE terms
    themselves -- not merely their order or scale. Not a monotone-rescaling-
    of-a-rank case either (the DV is a raw magnitude mean, not an order
    statistic).

READINESS (P0)
    P_MULTI  multi_action_commits_present (existential, both arms) -- carried
        over from 839/844's identical precondition.
    P_PRECOMMIT  decomposition_precommit_live (existential, both arms).
    P_MIDEXEC_DECOMP  midexec_decomposition_occurs (existential, BOTH arms
        here -- unlike 844, where this only applied to the ON arm, since
        BOTH arms here have the abort mechanism enabled and should be able to
        decompose mid-execution).
    P_HARM_BIAS_ENGAGES  harm_bias_engages (existential, ARM_SELECTION_ON
        only) -- at least one ON-arm cell shows `decomp_n_harm_bias_
        nonzero > 0` this run. `use_harm_aware_selection=True` alone is not
        sufficient for the treatment to actually DO anything: `harm_bias()`
        is a no-op whenever `z_harm_a_norm` never rises above `harm_threat_
        floor` this run, in which case ARM_SELECTION_ON is behaviourally
        identical to ARM_SELECTION_OFF for a reason that has nothing to do
        with whether harm-aware selection HELPS -- a below-floor reading here
        must self-route `substrate_not_ready_requeue`, never a C1 verdict on
        a manipulation that never engaged.
    P_PE_VARIES / P_PE_BOUNDED  ARM_SELECTION_OFF forward-PE is a real,
        bounded, VARYING quantity (applies to OFF only, mirroring 816's P3 /
        844's identical precondition) -- rules out an untrained forward-model
        noise floor making the DV meaningless before any cross-arm comparison
        is drawn.

SLEEP DRIVER: not applicable -- no sleep phase entered in this run.

Z_GOAL: deliberately inert, carried over verbatim from 844/839 for the
identical reason (`REEConfig.from_dims(z_goal_enabled=...)` defaults False,
so `goal_state` is never constructed; recorded via the accumulator so absence
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

EXPERIMENT_TYPE = "v3_exq_867_mech321_harm_aware_selection_task_effect"
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["MECH-321"]

ARM_OFF = "ARM_SELECTION_OFF"
ARM_ON = "ARM_SELECTION_ON"
ARMS = (ARM_OFF, ARM_ON)

MINT_RUN_ID = "v3_exq_844_mech321_r4_midexec_task_effect_20260801T013315Z_v3"

# --- Harm-aware-selection parameters. Verified against the PolicyDecomposition-
# Config dataclass defaults AND the REEConfig.from_dims mirror defaults
# (ree_core/agent.py:3963-3968, 5939-5944) -- identical values, no naming
# mismatch. No documented reason on record to deviate for a first validation. ---
HARM_BIAS_GAIN = 0.1
HARM_BIAS_SCALE = 0.1
HARM_THREAT_FLOOR = 0.1
HARM_THREAT_REF = 0.5
HARM_OVERRIDE_W_THRESHOLD = 0.9

# CONFIG_SLICE_DECLARATION_EXEMPT: validate_experiments.py's static resolver
# cannot trace `_config_slice(arm_id)`'s conditional dispatch to
# `_on_selection_config_slice()`, so it flags these five constants as
# "omitted." They are NOT omitted -- `_on_selection_config_slice()` declares
# each one under its REEConfig-mirror key ("decomposition_harm_bias_gain" for
# HARM_BIAS_GAIN, etc.), which is what `_arm_flags()`/`REEConfig.from_dims`
# actually reads downstream, so a future consumer citing a different value
# for any of these five would necessarily declare a different value under
# that same key and correctly get a different fingerprint -- no false-HIT
# risk. (ARM_SELECTION_OFF, the only arm this script ever tries to REUSE
# INTO via try_reuse_cell, never reads these five at all -- see
# _off_selection_config_slice.)

# --- Pre-registered thresholds. Constants, never derived from this run's own
# statistics (mirrors 844 / 839 / 816 convention). ---
MULTI_ACTION_COMMIT_FLOOR = 0.0
PRECOMMIT_EVAL_FLOOR = 0.0
MIDEXEC_DECOMP_FLOOR = 0.0
HARM_BIAS_ENGAGE_FLOOR = 0.0
PE_VARIANCE_FLOOR = 1e-12
PE_SANITY_CEIL = 1e6

# Effect-size gate for the load-bearing paired delta -- same shape and same
# values as 844's, for the same reason (small seed tier by substrate
# construction; no chunk-selection-bias knob exists).
EFFECT_SIZE_K_SIGMA = 1.0
REL_IMPROVEMENT_FLOOR = 0.0
MIN_DECOMPOSING_SEEDS = 2

_ZGOAL = ZGoalStreamAccumulator()


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------
def _off_selection_config_slice() -> Dict[str, Any]:
    """ARM_SELECTION_OFF's fingerprint config slice: reproduces V3-EXQ-844's
    `_config_slice(ARM_HANDLE_ON)` EXACTLY (abort mechanism ON, no harm-
    aware-selection keys declared -- 844 predates the flag's existence, and
    the flag is a bit-identical no-op when absent/False), so this cell's
    fingerprint matches 844's own reuse-eligible ARM_HANDLE_ON mint."""
    slice_ = dict(baselines.off_path_config_slice())
    slice_.update(baselines.on_arm_flags())
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
    """Keys this run reads back from a (real or reused) OFF cell -- identical
    list to V3-EXQ-844's own `_needed_keys()`, since ARM_SELECTION_OFF reads
    exactly what 844's ARM_HANDLE_ON already recorded."""
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
    forward-PE and harm so a post-divergence window can be computed after the
    fact. Identical discipline to V3-EXQ-844/839's `_run_cell` -- nothing
    about the mid-execution gate or the harm-aware selection is injected;
    both are read out of the agent's own state after stepping."""
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
            # reached -- 844/839's exact pattern.
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
            # (metrics["e3_prediction_error"], same read 844/839/816 use).
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

    # Both arms have the abort mechanism enabled here, so cell_pass is a
    # uniform sanity check (multi-action program committed AND the
    # mid-execution hook was actually evaluated) rather than 844's
    # arm-conditional OFF-must-be-zero test.
    cell_pass = bool(
        multi_action_commits > 0 and row["decomp_n_evaluated_midexec"] > 0)
    row["cell_pass"] = cell_pass
    if not quiet:
        print(f"verdict: {'PASS' if cell_pass else 'FAIL'}", flush=True)
    return row


def _reused_off_row(seed: int, cached: Dict[str, Any]) -> Dict[str, Any]:
    """Reshape a `try_reuse_cell` hit (a cell from 844's ARM_HANDLE_ON) into
    the same row shape `_run_cell` produces, so downstream analysis code does
    not need to branch on reused-vs-fresh. Harm-aware-selection counters are
    trivially 0 -- the reused cell ran with the flag off (844 predates it)."""
    row = dict(cached)
    row["arm_id"] = ARM_OFF
    row["seed"] = int(seed)
    row.setdefault("e3_tick_flags", [])
    row["decomp_n_harm_bias_nonzero"] = 0
    row["decomp_n_harm_override_fires"] = 0
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
        and row.get("decomp_n_evaluated_midexec", 0) > 0)
    row["reused_from_run_id"] = MINT_RUN_ID
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
                "len(remaining) > 1 -- carried over from V3-EXQ-839/844's "
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
                "means the decomposition machinery is not running at all, a "
                "config error rather than a finding about this run's "
                "question."
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
                "midexec > 0 THIS run. Unlike V3-EXQ-844 (whose OFF arm "
                "structurally could not decompose), BOTH arms here have the "
                "abort mechanism enabled, so this applies to BOTH arms."
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
                "> 0). use_harm_aware_selection=True alone does not "
                "guarantee the treatment channel does anything: harm_bias() "
                "is a no-op whenever z_harm_a_norm never rises above "
                "harm_threat_floor this run. Below floor means the "
                "manipulation never engaged -- substrate_not_ready_requeue, "
                "never a C1 verdict on an inert treatment. EXISTENTIAL, ON "
                "arm only."
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
                "must have non-zero variance across the whole run (world_"
                "forward learned something informative) -- else the DV is a "
                "degenerate constant. Mirrors V3-EXQ-816/844's P3."
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
# Windowed per-seed paired delta -- REUSED VERBATIM FROM V3-EXQ-844
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
    all-tick (hold-weighted) and fresh-selection-restricted forms. Identical
    logic to V3-EXQ-844's `_windowed_delta`."""
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
    """For a seed with NO divergence in either arm (`neither_decompose`
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


def _abort_rate_covariate(off_row: Dict[str, Any], on_row: Dict[str, Any]) -> Dict[str, Any]:
    """MATCHED-ABORT-RATES covariate (design requirement (a)): per-seed
    decomp_n_decomposed_midexec counts and their delta, reported for EVERY
    seed regardless of tier -- non-gating, auditable alongside C1."""
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

    # --- Four-way seed tiering, RE-MEASURED this run. See MATCHED ABORT
    # RATES / SEED TIERING in the module docstring. ---
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

    # LOAD-BEARING C1 windowed pairing: restricted to both_decompose_seeds
    # (matched-abort-pairing, design requirement (b)) so the measured delta
    # reflects selection QUALITY, not whether an abort happened at all.
    windowed: List[Dict[str, Any]] = []
    for s in both_decompose_seeds:
        div_tick = _first_divergence_tick(off_by_seed[s], on_by_seed[s])
        if div_tick is not None:
            windowed.append(_windowed_delta(off_by_seed[s], on_by_seed[s], div_tick))

    # Non-gating, unmatched-abort-rate windowed deltas -- reported separately,
    # NEVER folded into the load-bearing C1 aggregate.
    unmatched_windowed: List[Dict[str, Any]] = []
    for s in on_only_seeds + off_only_seeds:
        div_tick = _first_divergence_tick(off_by_seed[s], on_by_seed[s])
        if div_tick is not None:
            w = _windowed_delta(off_by_seed[s], on_by_seed[s], div_tick)
            w["unmatched_tier"] = "on_only" if s in on_only_seeds else "off_only"
            unmatched_windowed.append(w)

    null_checks = [_null_check_delta(off_by_seed[s], on_by_seed[s]) for s in neither_decompose_seeds]
    abort_rate_covariate = [_abort_rate_covariate(off_by_seed[s], on_by_seed[s]) for s in seeds]

    # LOAD-BEARING: fresh-selection-restricted delta (hold-weighting confound
    # removed; matched-abort-pairing confound removed).
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

    # NON-gating corroboration: mechanistic (forward-PE, fresh-restricted,
    # matched-pairing) and realised-task-outcome-over-full-window (all-tick,
    # hold-weighted) readouts.
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
            "them -- either harm-aware selection never engaged on a seed "
            "where the abort mechanism also fired in BOTH arms (see "
            "on_only/off_only tiers for possible abort-rate asymmetry "
            "instead), or the selected leaf never differed from the default "
            "harm-blind pick")
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
             "(matched-abort-rate pairing -- both_decompose_seeds), the "
             "paired post-divergence-window mean harm signal, restricted to "
             "FRESH e3-selection ticks, is LESS negative (less harmful) in "
             "ARM_SELECTION_ON (harm-aware selection) than ARM_SELECTION_OFF, "
             f"by an effect exceeding {EFFECT_SIZE_K_SIGMA} x SE over >= "
             f"{MIN_DECOMPOSING_SEEDS} paired seeds.")},
        {"name": "C2_FORWARD_PE_CORROBORATES", "load_bearing": False,
         "passed": bool(pe_corroborates),
         "measured": pe_delta_mean, "threshold": 0.0,
         "statement": (
             "Consistency check (non-load-bearing, carried over from V3-EXQ-"
             "844's established finding): on the SAME matched-abort-pair "
             "window, forward-prediction error is also lower in ON than "
             "OFF, i.e. layering harm-aware selection on top of the abort "
             "mechanism does not undermine 844's own PE-reduction result. "
             "Not a novel claim about harm-aware selection's own "
             "mechanism -- harm is the task-outcome DV this claim actually "
             "asks about.")},
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
    # are emergent counts / variances), so there is nothing to refuse today.
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
                    config_slice=_off_selection_config_slice(),
                    seed=seed,
                    script_path=Path(__file__),
                    needed_keys=_needed_keys(),
                    cite_run_id=MINT_RUN_ID,
                    include_driver_script_in_hash=False,
                )
                if cached is not None:
                    rows.append(_reused_off_row(seed, cached))
                    reuse_log.append(f"seed={seed}: REUSED from {MINT_RUN_ID} (844 ARM_HANDLE_ON)")
                    continue
                reuse_log.append(
                    f"seed={seed}: reuse_refused (fingerprint mismatch -- likely "
                    "machine-class or substrate drift vs 844's mint) -- running fresh")

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
        "env_kwargs_per_seed": {str(s): baselines.env_kwargs(s) for s in seeds},
        "self_dim": baselines.SELF_DIM,
        "world_dim": baselines.WORLD_DIM,
        "seeded_chunk_sequence": list(baselines.SEEDED_CHUNK_SEQUENCE),
        "decomposition_vs_threshold": baselines.DECOMPOSITION_VS_THRESHOLD,
        "decomposition_depth_cap": baselines.DECOMPOSITION_DEPTH_CAP,
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
        "bears_on": ["MECH-321", "SD-hazard-aware-policy-decomposition", "ARC-070", "ARC-071"],
        "evidence_direction": result["evidence_direction"],
        "supersedes": None,
        "reuse_baseline_from": MINT_RUN_ID,
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "per_arm_gate": result["per_arm_gate"],
        "seed_tiers_measured": result["seed_tiers_measured"],
        "seed_tiers_declared_by_baseline": {
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
        "unmatched_windowed_deltas": result["unmatched_windowed_deltas"],
        "null_checks": result["null_checks"],
        "abort_rate_covariate": result["abort_rate_covariate"],
        "arm_results": rows,
        "per_seed_rows": rows,
        "custom_information": {
            "predecessor_task_outcome_run": MINT_RUN_ID,
            "predecessor_autopsy": (
                "REE_assembly/evidence/planning/"
                "failure_autopsy_V3-EXQ-844_2026-08-01.md"),
            "substrate_doc": (
                "REE_assembly/docs/architecture/"
                "sd_hazard_aware_policy_decomposition.md"),
            "gov_reuse_1_check": (
                "Decisive readout (paired matched-abort-pair windowed harm "
                "delta between harm-aware-selection ON/OFF) does not exist "
                "in ANY prior manifest -- the harm-aware-selection substrate "
                "was built 2026-08-01 and this is its first behavioural "
                "test. `reanalysis_query.py query --readout "
                "decomposition_harm_bias --claim MECH-321` confirmed 0 of 4 "
                "matched manifests carry it. Not recoverable via reanalysis "
                "-> proceed to author. ARM_SELECTION_OFF cells, however, ARE "
                "recoverable via arm-reuse from V3-EXQ-844's own reuse-"
                "eligible ARM_HANDLE_ON mint (see reuse_log)."),
            "matched_abort_rates_design_note": (
                "Handled two ways per the routing note: (a) per-seed "
                "decomp_n_decomposed_midexec counts and their ON-minus-OFF "
                "delta are recorded for every seed as a non-gating covariate "
                "(abort_rate_covariate); (b) the load-bearing C1 windowed "
                "comparison is restricted to both_decompose_seeds -- seeds "
                "where BOTH arms independently decomposed mid-execution -- "
                "so the measured harm delta reflects selection QUALITY among "
                "candidates the abort already produced, not whether an "
                "abort happened at all. on_only/off_only tiers (asymmetric "
                "abort rate) are reported separately in "
                "unmatched_windowed_deltas, non-gating."),
            "gov_diag_1_note": (
                "Not the refused 816e env-harshening re-queue (pre-commit/R1 "
                "axis) and not a re-test of 844's own abort-mechanism axis "
                "(held fixed, ON, in both arms here). This run adds a "
                "genuinely new manipulated variable -- harm-aware selection "
                "-- on top of 844's fixed manipulation. Re-derive brake "
                "check (Step 2.5b): 0 substrate_ceiling/non_contributory "
                "autopsy hits against MECH-321 at this granularity."),
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
        f"rel_improvement={s['rel_improvement']:.4f}", flush=True)
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
