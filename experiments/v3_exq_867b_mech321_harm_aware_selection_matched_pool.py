#!/opt/local/bin/python3
"""V3-EXQ-867b -- MECH-321 HARM-AWARE SELECTION, SCREENED MATCHED-ABORT SEED
POOL. Same question as V3-EXQ-867a/867/844 (does SD-hazard-aware-policy-
decomposition's harm-valence-weighted retiling selection improve TASK OUTCOME
over the abort-mechanism-alone baseline, at matched abort rates?), re-posed
after V3-EXQ-867a's load-bearing C1 came back SEVERELY UNDERPOWERED.

SUPERSEDES V3-EXQ-867a (measurement/sampling-design fix -- same question, same
substrate, same manipulation, SAME BAR; only the matched-pair sample size and
the power guard change). Per CLAUDE.md's EXQ lettering convention this is the
bug-fix case, not a redesign: nothing about the mechanism under test moves.

WHY 867a WAS UNDERPOWERED, AND THE TWO SEPARATE DEFECTS BEHIND IT
(`failure_autopsy_V3-EXQ-867a_2026-08-03`, confirmed; scored
`measurement_test_design_defect` / `non_contributory`)

867a was a genuine success at the substrate level: the SD-029 hazard-density
overlay plus the three-flag stream fix finally made the manipulation ENGAGE
(harm_bias_engages met; ON arm decomp_n_harm_bias_nonzero up to 972). But its
load-bearing C1 ran on TWO matched seeds:

    tier                seeds        harm_delta_fresh_off_minus_on
    both_decompose      11, 23       -0.0402 , +0.0125   <- C1 ran on THESE
    on_only_decompose   47           +0.1270             <- correctly EXCLUDED
    neither_decompose   89, 97       (negative control)

Two opposite-signed deltas netted -0.0139 against a 1xSE bar, and the run
self-routed `harm_aware_selection_does_not_reduce_task_harm` / `weakens`.
The autopsy declined that reading: the excluded unmatched seed's much larger
beneficial-direction effect (+0.127) is a material covariate against treating
n=2 as a confident null.

DEFECT 1 -- THE MATCHED TIER WAS SET BY CHANCE, NOT BY DESIGN CONTROL.
    `both_decompose` membership requires mid-execution decomposition to fire
    independently in BOTH arms on the same seed. 867a ran a fixed 5-seed pool
    and simply took whatever landed in that tier. Of the 5, two were declared
    negative controls (multi_action_commits == 0 in both arms, so they can
    NEVER be matched), one landed on_only -- leaving 2. The pool was chosen
    from a darwin-arm64 --seed-scan that measured the ON ARM ONLY, at a
    reduced schedule, on a DIFFERENT MACHINE CLASS from the cloud worker that
    actually ran it (ree-cloud-4, linux-x86_64). Nothing in that pipeline
    controls the matched count.

DEFECT 2 -- THE POWER GUARD WAS VACUOUS, WHICH IS WHY n=2 SCORED AS A
VERDICT AT ALL. 867a (and 844/867 before it) computed

        enough_pairs = n_pairs >= min(MIN_DECOMPOSING_SEEDS,
                                      len(both_decompose_seeds) or 1)

    The `min(...)` collapses the pre-registered floor onto the OBSERVED count
    whenever n_pairs == len(both_decompose_seeds) -- i.e. in the normal case.
    With MIN_DECOMPOSING_SEEDS=2 and both_decompose=2 it reads 2 >= min(2,2),
    which is True; it would ALSO have read True at n=1. So the guard could
    never fire, `harm_aware_selection_task_effect_underpowered` was
    unreachable, and the run took the scored `weakens` branch instead. This
    is the mechanism by which an n=2 sample became a claim-weakening verdict.

THE FIX, IN THREE PARTS

(1) A SCREENED, ENLARGED MATCHED POOL -- and the screen is SOUND, not a
    heuristic. `_run_cell` at K episodes is a bit-identical PREFIX of the same
    cell at N>K episodes (same RNG reset at cell entry via `arm_cell`, same
    per-cell env seed, nothing in the loop reads `episodes`), and
    `PolicyDecomposition.reset()` is explicitly NOT called on the agent's
    per-episode reset (see its docstring in policy/policy_decomposition.py),
    so `decomp_n_decomposed_midexec` accumulates monotonically across the
    whole cell. Therefore:

        screened both-arms-decompose at SCREEN_EPISODES
            ==> both-arms-decompose at the full EPISODES        (guaranteed)

    The implication runs one way only: the screen has NO false positives and
    may have false negatives (a seed that first decomposes in a later
    episode), which costs a candidate and never a false verdict -- the same
    discipline the baseline module's own --seed-scan already runs on.
    VERIFIED DIRECTLY (2026-08-03, darwin-arm64, seed 11, both arms, 2ep vs
    4ep x 12 steps): action_sequence prefix identical in both arms, and every
    decomposition counter monotone.

    This converts the matched-pair count from a chance outcome into a DESIGNED
    one, which is exactly what the autopsy asked for. Critically the screen
    runs IN THIS PROCESS, immediately before the measurement, so it
    characterises the pool on the SAME MACHINE CLASS that measures it --
    closing the darwin-scan/cloud-run gap 867a inherited.

(2) A HARD POWER FLOOR. `enough_pairs = n_pairs >= MIN_DECOMPOSING_SEEDS`,
    with MIN_DECOMPOSING_SEEDS raised 2 -> 6 and the `min(...)` softening
    DELETED. An underpowered run now self-routes
    `harm_aware_selection_task_effect_underpowered` AND stamps
    `non_degenerate: false`, so it is scoring-excluded rather than recorded as
    a `weakens` -- the V3-EXQ-514m class the recording standard names.

(3) THE BAR ITSELF IS UNCHANGED. EFFECT_SIZE_K_SIGMA=1.0,
    REL_IMPROVEMENT_FLOOR=0.0, the `_windowed_delta` /
    `_first_divergence_tick` / `_mean_at` machinery, the fresh-selection
    restriction, the arms, the env overlay, the three stream flags and the
    harm-aware-selection parameters are all carried over VERBATIM from
    867a/867/844. Only n and the guard move. Moving the bar as well would be
    changing the question, not repairing the measurement.

ARMS -- UNCHANGED FROM 867a. Both reproduce the abort mechanism ON plus the
three stream flags (substrate-stack preconditions, identical in both arms,
NOT the manipulation); the only manipulation is harm-aware selection.
    ARM_SELECTION_OFF: hazard-tuned env + abort mechanism ON + three stream
        flags; harm-aware-selection flag at its off-by-default value.
    ARM_SELECTION_ON: identical plus
        `decomposition_use_harm_aware_selection=True` at the SD doc's
        recommended defaults.

RUN SHAPE
    PHASE S (screen, quiet): each candidate seed in CANDIDATE_SEEDS is run in
        BOTH arms at SCREEN_EPISODES x STEPS_PER_EPISODE. Stops early once
        TARGET_MATCHED_SEEDS matched candidates AND MEASURE_CONTROL_SLOTS
        zero-activity candidates have been found. Emits no `verdict:` line and
        no `ep N/M` line, so the runner's progress contract sees only PHASE M.
    PHASE M (measure): exactly MEASURE_SEEDS seeds x 2 arms at the full
        EPISODES x STEPS_PER_EPISODE -- MEASURE_MATCHED_SLOTS drawn from the
        screened matched set (padded deterministically from the highest-
        activity unmatched candidates if the screen came up short) plus
        MEASURE_CONTROL_SLOTS zero-activity negative controls, which are
        retained from 867a's design so the run still DEMONSTRATES the
        attributability instrument rather than asserting it. The cell count is
        therefore FIXED at 20 regardless of screen yield, which is what keeps
        the runner's `seeds x conditions` contract honest under an
        early-exiting screen.

FINGERPRINT CORRECTNESS NOTE (non-obvious, load-bearing). The screen and the
measurement run the same (seed, arm) pairs at DIFFERENT episode counts, so the
config slice MUST carry the actual episode count -- `_config_slice` takes
`episodes` and writes it into `schedule.episodes`. Sharing one slice across
both phases would give a 4-episode screen cell and a 12-episode measurement
cell the SAME fingerprint while holding different content, which is a false-
reuse hazard for any future consumer of this mint.

NO ARM-REUSE THIS RUN. 867a minted its ARM_SELECTION_OFF cells reuse-eligible
and this run's OFF slice is built from the same canonical baseline module, so
a hit would be structurally possible -- but `ree_core/**` has moved since
867a's 2026-08-02 run, so the substrate_hash cannot match and `try_reuse_cell`
would refuse on every cell. The call is omitted rather than left in as dead
code (same decision, same reason, as 867a made about 867's mint). This run's
own ARM_SELECTION_OFF measurement cells are again minted reuse-ELIGIBLE
(`include_driver_script_in_hash=False`) as the canonical baseline for this
lineage.

GOV-REUSE-1. The decisive readout is the paired per-seed windowed harm delta
over >= MIN_DECOMPOSING_SEEDS matched seeds. Checked against the only two
recorded runs that carry it -- v3_exq_867a_...20260802T203309Z_v3 (n=2 matched)
and v3_exq_867_...20260802T031937Z_v3 (manipulation structurally inert, no
matched tier at all). Neither records, nor permits deriving, the readout at the
required n: additional matched seeds are new cells that were never run, not a
reprocessing of banked ones. Not recoverable -> run.

C1 STATISTIC -- REUSED VERBATIM FROM 844/867/867a. Load-bearing DV = paired
per-seed windowed mean harm delta (OFF minus ON, restricted to fresh
e3-selection ticks, from the first action-sequence divergence tick), positive
= ARM_SELECTION_ON is less harmful.

MATCHED ABORT RATES -- unchanged, handled two ways: (a) per-seed
decomp_n_decomposed_midexec covariate reported for every measured seed,
non-gating; (b) load-bearing C1 restricted to `both_decompose_seeds` measured
at the FULL schedule (the screen selects candidates; it never substitutes for
the real tier assignment, which is recomputed from the measurement cells).

READINESS (P0) -- unchanged in shape from 867a; `harm_bias_engages` re-
confirmed reachable on the current substrate by this session's probe (ON arm
decomp_n_harm_bias_nonzero=33 at 4 episodes, OFF arm 0).

SLEEP DRIVER: not applicable -- no sleep phase entered in this run.

Z_GOAL: deliberately inert, carried over verbatim from 844/867/867a for the
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

EXPERIMENT_TYPE = "v3_exq_867b_mech321_harm_aware_selection_matched_pool"
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["MECH-321"]
QUEUE_ID = "V3-EXQ-867b"
SUPERSEDES: Optional[str] = "V3-EXQ-867a"

ARM_OFF = "ARM_SELECTION_OFF"
ARM_ON = "ARM_SELECTION_ON"
ARMS = (ARM_OFF, ARM_ON)

# --- Harm-aware-selection parameters. VERBATIM from 867a/867 -- the
# manipulation does not move in this iteration. ---
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
# 867a's and 867's own exemption note).

# --- Pre-registered thresholds. Constants, never derived from this run's own
# statistics (mirrors 844/867/867a convention). ---
MULTI_ACTION_COMMIT_FLOOR = 0.0
PRECOMMIT_EVAL_FLOOR = 0.0
MIDEXEC_DECOMP_FLOOR = 0.0
HARM_BIAS_ENGAGE_FLOOR = 0.0
PE_VARIANCE_FLOOR = 1e-12
PE_SANITY_CEIL = 1e6

# --- C1's bar. UNCHANGED from 844/867/867a. Do not move these: this
# iteration repairs the SAMPLE, not the criterion. ---
EFFECT_SIZE_K_SIGMA = 1.0
REL_IMPROVEMENT_FLOOR = 0.0

# --- The power floor. Raised 2 -> 6, and applied HARD (see module docstring
# DEFECT 2: 867a's `min(MIN_DECOMPOSING_SEEDS, len(both_decompose_seeds) or 1)`
# collapsed the floor onto the observed count and could never fire). ---
MIN_DECOMPOSING_SEEDS = 6

# --- Screen / measurement design parameters. ---
SCREEN_EPISODES = 4          # matches the baseline module's own --seed-scan schedule
MEASURE_MATCHED_SLOTS = 8    # >= MIN_DECOMPOSING_SEEDS, with headroom
MEASURE_CONTROL_SLOTS = 2    # zero-activity negative controls (867a design element)
MEASURE_SEEDS = MEASURE_MATCHED_SLOTS + MEASURE_CONTROL_SLOTS   # FIXED cell count
TARGET_MATCHED_SEEDS = MEASURE_MATCHED_SLOTS

# Pre-registered candidate pool. The first ten are the seeds already
# characterised for this hazard-tuned lineage (baseline module's 2026-08-02
# --seed-scan table), ordered activity-first so the early exit is reached
# sooner; the remainder are fresh. Seed 44 is deliberately absent (CLAUDE.md:
# a recurring per-seed early-episode-death instability). Screened in this
# order, so the selected pool is a deterministic function of the run machine.
CANDIDATE_SEEDS: Tuple[int, ...] = (
    11, 23, 47, 71, 3, 29, 89, 97, 17, 53,
    5, 7, 13, 19, 31, 37, 41, 43, 59, 61,
    67, 73, 79, 83, 101, 103, 107, 109, 113, 127,
    2, 6, 8, 12, 14, 18, 22, 26, 33, 39,
    51, 57, 63, 69, 77, 81, 87, 93,
)

_ZGOAL = ZGoalStreamAccumulator()


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------
def _off_selection_config_slice(episodes: int) -> Dict[str, Any]:
    """ARM_SELECTION_OFF's fingerprint config slice.

    `episodes` is a REQUIRED argument, not a module constant: the screen and
    the measurement run the same (seed, arm) pairs at different schedules, and
    a slice that did not carry the episode count would give those two cells
    the same fingerprint while holding different content. See the module
    docstring's FINGERPRINT CORRECTNESS NOTE.
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
    slice_.update(baselines.on_arm_flags())  # abort mechanism ON
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
# One cell -- carried over from 867a, plus the extra decomposition-trigger
# diagnostics the 867a autopsy could not read (see `decomp_n_boundary_fires` /
# `decomp_n_vs_trigger` / `decomp_n_marked_unreliable`: these are what explain
# WHY a seed's midexec hit rate is what it is, and were absent from 867a's
# rows).
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
# PHASE S -- the screen
# ---------------------------------------------------------------------------
def _screen_pool(candidates: Tuple[int, ...], screen_episodes: int, steps: int,
                 target_matched: int, control_slots: int) -> Dict[str, Any]:
    """Screen candidate seeds for the both-arms-decompose tier.

    Sound by prefix-monotonicity (module docstring, part (1)): a candidate
    that decomposes mid-execution in BOTH arms within `screen_episodes` is
    GUARANTEED to do so at the full schedule, because the full run's first
    `screen_episodes` episodes are bit-identical and the counters accumulate.
    The converse does not hold, so a screen miss costs a candidate and never
    a false verdict.
    """
    screened: List[Dict[str, Any]] = []
    matched: List[int] = []
    controls: List[int] = []

    for i, seed in enumerate(candidates):
        off = _run_cell(seed, ARM_OFF, screen_episodes, steps, quiet=True)
        # SHORT-CIRCUIT. `matched` requires the OFF arm to decompose, so a
        # candidate whose OFF cell did not decompose can never be matched and
        # its ON cell buys nothing for the selection -- skip it. The ON cell IS
        # still run when the candidate could serve as a zero-activity negative
        # control and we still need one, because that tier is defined over
        # BOTH arms. No information about matchedness is lost; only covariate
        # richness on already-rejected candidates, which is recorded as
        # `screen_on_arm_skipped` rather than left implicit.
        need_on = bool(
            off["decomp_n_decomposed_midexec"] > 0
            or (off["multi_action_commits"] == 0 and len(controls) < control_slots))
        on = (_run_cell(seed, ARM_ON, screen_episodes, steps, quiet=True)
              if need_on else None)
        rec = {
            "seed": int(seed),
            "off_decomp_midexec": off["decomp_n_decomposed_midexec"],
            "on_decomp_midexec": (
                on["decomp_n_decomposed_midexec"] if on is not None else None),
            "off_multi_action_commits": off["multi_action_commits"],
            "on_multi_action_commits": (
                on["multi_action_commits"] if on is not None else None),
            "off_evaluated_midexec": off["decomp_n_evaluated_midexec"],
            "on_evaluated_midexec": (
                on["decomp_n_evaluated_midexec"] if on is not None else None),
            "on_harm_bias_nonzero": (
                on["decomp_n_harm_bias_nonzero"] if on is not None else None),
            "off_harm_bias_nonzero": off["decomp_n_harm_bias_nonzero"],
            "screen_on_arm_skipped": on is None,
        }
        rec["screen_matched"] = bool(
            on is not None
            and rec["off_decomp_midexec"] > 0 and rec["on_decomp_midexec"] > 0)
        rec["screen_zero_activity"] = bool(
            on is not None
            and rec["off_multi_action_commits"] == 0
            and rec["on_multi_action_commits"] == 0)
        screened.append(rec)
        if rec["screen_matched"]:
            matched.append(int(seed))
        elif rec["screen_zero_activity"]:
            controls.append(int(seed))

        print(
            f"  [screen] cand {i + 1}/{len(candidates)} seed={seed} "
            f"off_decomp={rec['off_decomp_midexec']} "
            f"on_decomp={rec['on_decomp_midexec']} "
            f"matched={len(matched)}/{target_matched} "
            f"controls={len(controls)}/{control_slots}",
            flush=True,
        )
        if len(matched) >= target_matched and len(controls) >= control_slots:
            break

    return {
        "screened": screened,
        "matched_seeds": matched,
        "control_seeds": controls,
        "n_candidates_screened": len(screened),
        "screen_episodes": screen_episodes,
        "screen_steps_per_episode": steps,
        "screen_exhausted_pool": len(screened) >= len(candidates),
    }


def _select_measurement_seeds(screen: Dict[str, Any], matched_slots: int,
                              control_slots: int) -> Dict[str, Any]:
    """Deterministically fill a FIXED number of measurement slots.

    Matched slots take screened matched seeds first (screen order), then pad
    from the highest-activity unmatched, non-control candidates so the cell
    count -- and therefore the runner's `seeds x conditions` contract -- is
    constant regardless of screen yield. Padded seeds are expected to land in
    the unmatched tiers and are reported as covariates, which is exactly the
    role 867a's excluded seed 47 played.
    """
    matched = list(screen["matched_seeds"])[:matched_slots]
    controls = list(screen["control_seeds"])[:control_slots]
    chosen = set(matched) | set(controls)

    def _both_arm_min(rec: Dict[str, Any], off_key: str, on_key: str) -> int:
        """min over the two arms, tolerating a short-circuited (skipped) ON cell."""
        off_v = int(rec[off_key])
        on_v = rec[on_key]
        return off_v if on_v is None else min(off_v, int(on_v))

    pad_pool = [
        r for r in screen["screened"]
        if r["seed"] not in chosen and not r["screen_zero_activity"]
    ]
    # Candidates whose ON cell was actually measured rank ahead of
    # short-circuited ones at equal activity: their tier is better evidenced.
    pad_pool.sort(key=lambda r: (
        1 if r["screen_on_arm_skipped"] else 0,
        -_both_arm_min(r, "off_decomp_midexec", "on_decomp_midexec"),
        -_both_arm_min(r, "off_multi_action_commits", "on_multi_action_commits"),
        r["seed"],
    ))
    matched_pad = [r["seed"] for r in pad_pool[: max(0, matched_slots - len(matched))]]
    chosen |= set(matched_pad)

    control_pad_pool = [
        r for r in screen["screened"]
        if r["seed"] not in chosen
    ]
    control_pad_pool.sort(key=lambda r: (
        _both_arm_min(r, "off_multi_action_commits", "on_multi_action_commits"),
        r["seed"],
    ))
    control_pad = [
        r["seed"] for r in control_pad_pool[: max(0, control_slots - len(controls))]
    ]

    seeds = matched + matched_pad + controls + control_pad
    return {
        "measurement_seeds": seeds,
        "from_screen_matched": matched,
        "matched_slot_padding": matched_pad,
        "from_screen_control": controls,
        "control_slot_padding": control_pad,
        "screen_yield_met": len(matched) >= matched_slots,
    }


# ---------------------------------------------------------------------------
# Preconditions -- unchanged in shape from 867a
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
                "len(remaining) > 1 -- carried over from 839/844/867/867a's "
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
                "> 0). Re-confirmed reachable on the current substrate by "
                "this session's 2026-08-03 probe (ON arm 33 at 4 episodes, "
                "OFF arm 0). A below-floor reading would mean the substrate "
                "is not ready for a FURTHER reason -- "
                "substrate_not_ready_requeue, never a C1 verdict on an inert "
                "treatment. EXISTENTIAL, ON arm only."
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
                "816/839/844/867/867a's P3."
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
# Windowed per-seed paired delta -- REUSED VERBATIM FROM V3-EXQ-844/867/867a
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
        "n_fresh_ticks_off": n_off_harm_fresh,
        "n_fresh_ticks_on": n_on_harm_fresh,
        "n_fresh_pe_ticks_off": n_off_pe_fresh,
        "n_fresh_pe_ticks_on": n_on_pe_fresh,
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
        "off_decomp_n_evaluated_midexec": off_row["decomp_n_evaluated_midexec"],
        "on_decomp_n_evaluated_midexec": on_row["decomp_n_evaluated_midexec"],
        "off_decomp_n_boundary_fires": off_row["decomp_n_boundary_fires"],
        "on_decomp_n_boundary_fires": on_row["decomp_n_boundary_fires"],
        "on_harm_bias_nonzero_count": on_row.get("decomp_n_harm_bias_nonzero", 0),
        "on_harm_override_fires_count": on_row.get("decomp_n_harm_override_fires", 0),
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def _analyse(rows: List[Dict[str, Any]], screen: Dict[str, Any],
             selection: Dict[str, Any]) -> Dict[str, Any]:
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

    # Tier assignment is recomputed from the MEASUREMENT cells. The screen
    # selects which seeds get measured; it never substitutes for this.
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

    # The screen's soundness claim, checked against the measurement rather
    # than asserted: every screen-matched seed that was measured MUST land in
    # the both_decompose tier (prefix-monotonicity). A violation would mean
    # the prefix assumption is false and is reported, not silently absorbed.
    screen_matched_measured = [
        s for s in selection["from_screen_matched"] if s in seeds]
    screen_monotonicity_violations = [
        s for s in screen_matched_measured if s not in both_decompose_seeds]

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

    null_checks = [_null_check_delta(off_by_seed[s], on_by_seed[s])
                   for s in neither_decompose_seeds]
    abort_rate_covariate = [_abort_rate_covariate(off_by_seed[s], on_by_seed[s])
                            for s in seeds]

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

    unmatched_harm_deltas = [w["harm_delta_fresh_off_minus_on"] for w in unmatched_windowed
                             if w["harm_delta_fresh_off_minus_on"] is not None]
    unmatched_harm_delta_mean = (
        statistics.fmean(unmatched_harm_deltas) if unmatched_harm_deltas else None)

    # HARD floor -- 867a's `min(MIN_DECOMPOSING_SEEDS, len(both_decompose_seeds)
    # or 1)` softening is deliberately DELETED. See module docstring DEFECT 2.
    enough_pairs = n_pairs >= MIN_DECOMPOSING_SEEDS
    effect_size_ok = enough_pairs and (harm_delta_mean > EFFECT_SIZE_K_SIGMA * harm_delta_se)
    rel_floor_ok = rel_improvement >= REL_IMPROVEMENT_FLOOR
    c1_task_outcome_improves = bool(
        aggregate["non_degenerate"] and enough_pairs and effect_size_ok and rel_floor_ok)

    any_divergence = len(windowed) > 0
    # An underpowered C1 is DEGENERATE, not a weak verdict: it must be
    # scoring-excluded rather than recorded as a direction (the mechanism by
    # which 867a's n=2 became a `weakens`).
    non_degenerate = bool(
        aggregate["non_degenerate"] and any_divergence and enough_pairs)

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
            f"only {n_pairs} matched-abort-pair(s) with a measurable window, "
            f"below the pre-registered hard floor of {MIN_DECOMPOSING_SEEDS}; "
            f"the screen returned {len(screen['matched_seeds'])} matched "
            f"candidate(s) from {screen['n_candidates_screened']} screened "
            "(pool "
            + ("EXHAUSTED" if screen["screen_exhausted_pool"] else "not exhausted")
            + "). Reported as non_contributory rather than a direction -- "
            "this is the guard 867a lacked")
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
             "paired seeds. Bar unchanged from 844/867/867a; only the "
             "required n moved (2 -> 6) and the floor is now applied hard.")},
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
        "screen_soundness_check": {
            "screen_matched_and_measured": screen_matched_measured,
            "monotonicity_violations": screen_monotonicity_violations,
            "held": not screen_monotonicity_violations,
            "note": (
                "Prefix-monotonicity predicts every screen-matched measured "
                "seed lands in both_decompose. A non-empty violations list "
                "falsifies that design assumption and must be adjudicated "
                "before the C1 reading is trusted."),
        },
        "windowed_deltas": windowed,
        "unmatched_windowed_deltas": unmatched_windowed,
        "null_checks": null_checks,
        "abort_rate_covariate": abort_rate_covariate,
        "summary": {
            "n_seeds": len(seeds),
            "n_both_decompose_seeds": len(both_decompose_seeds),
            "n_windowed_pairs": n_pairs,
            "min_decomposing_seeds_required": MIN_DECOMPOSING_SEEDS,
            "harm_delta_mean_post_divergence": harm_delta_mean,
            "harm_delta_sd": harm_delta_sd,
            "harm_delta_se": harm_delta_se,
            "rel_improvement": rel_improvement,
            "pe_delta_mean_post_divergence": pe_delta_mean,
            "all_tick_harm_delta_mean": all_tick_harm_delta_mean,
            "unmatched_harm_delta_mean": unmatched_harm_delta_mean,
            "n_unmatched_windowed_pairs": len(unmatched_harm_deltas),
            "effect_size_ok": effect_size_ok,
            "enough_pairs": enough_pairs,
            "rel_floor_ok": rel_floor_ok,
            "off_pe_var_worst": off_pe_var_worst,
            "off_pe_mean_worst": off_pe_mean_worst,
            "abort_rate_matched_seed_count": len(both_decompose_seeds),
            "abort_rate_unmatched_seed_count": len(on_only_seeds) + len(off_only_seeds),
            "max_z_harm_a_norm_on_arm": _best_cell(on_rows, "max_z_harm_a_norm"),
            "n_candidates_screened": screen["n_candidates_screened"],
            "n_screen_matched": len(screen["matched_seeds"]),
            "screen_yield_met": selection["screen_yield_met"],
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
        candidates = CANDIDATE_SEEDS[:4]
        screen_episodes = 1
        episodes = 2
        steps = 12
        matched_slots, control_slots = 1, 1
    else:
        candidates = CANDIDATE_SEEDS
        screen_episodes = SCREEN_EPISODES
        episodes = baselines.EPISODES
        steps = baselines.STEPS_PER_EPISODE
        matched_slots, control_slots = MEASURE_MATCHED_SLOTS, MEASURE_CONTROL_SLOTS

    assert_no_structurally_unsatisfiable_gate(
        _precondition_specs(), [_arm_context(a) for a in ARMS])

    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    # --- PHASE S: screen. Quiet w.r.t. the runner's progress patterns (no
    # `ep N/M`, no `verdict:`) so only PHASE M drives the progress bar. ---
    print(f"[phase] SCREEN: up to {len(candidates)} candidate seed(s) x 2 arms "
          f"at {screen_episodes} x {steps}", flush=True)
    screen = _screen_pool(candidates, screen_episodes, steps,
                          matched_slots, control_slots)
    selection = _select_measurement_seeds(screen, matched_slots, control_slots)
    measurement_seeds = selection["measurement_seeds"]
    print(f"[phase] SCREEN done: {len(screen['matched_seeds'])} matched from "
          f"{screen['n_candidates_screened']} screened; measuring "
          f"{measurement_seeds}", flush=True)

    # --- PHASE M: measure. FIXED cell count = len(measurement_seeds) x 2. ---
    rows: List[Dict[str, Any]] = []
    for arm_id in ARMS:
        for seed in measurement_seeds:
            with arm_cell(
                seed,
                config_slice=_config_slice(arm_id, episodes),
                script_path=Path(__file__),
                config_slice_declared=True,
                include_driver_script_in_hash=False,
            ) as cell:
                row = _run_cell(seed, arm_id, episodes, steps)
                cell.stamp(row)
            rows.append(row)

    result = _analyse(rows, screen, selection)

    run_id = (f"{EXPERIMENT_TYPE}_"
              f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_v3")
    cfg_record = {
        "arms": list(ARMS),
        "candidate_seeds": list(candidates),
        "measurement_seeds": list(measurement_seeds),
        "episodes": episodes,
        "steps_per_episode": steps,
        "screen_episodes": screen_episodes,
        "measure_matched_slots": matched_slots,
        "measure_control_slots": control_slots,
        "env_kwargs_per_seed": {
            str(s): baselines.env_kwargs_hazard_tuned(s) for s in measurement_seeds},
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
        "screen_results": screen,
        "measurement_seed_selection": selection,
        "screen_soundness_check": result["screen_soundness_check"],
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
                "failure_autopsy_V3-EXQ-867a_2026-08-03.md"),
            "substrate_doc": (
                "REE_assembly/docs/architecture/"
                "sd_hazard_aware_policy_decomposition.md"),
            "power_fix_note": (
                "867a's load-bearing C1 ran on 2 matched seeds with opposite-"
                "signed deltas (-0.040, +0.0125) and scored `weakens`. Two "
                "defects: (1) the matched tier was set by chance -- a fixed "
                "5-seed pool characterised on a DIFFERENT machine class, ON "
                "arm only; (2) the power guard was vacuous -- "
                "`n_pairs >= min(MIN_DECOMPOSING_SEEDS, "
                "len(both_decompose_seeds) or 1)` collapses onto the observed "
                "count, so the underpowered branch was unreachable. 867b "
                "screens an enlarged candidate pool IN-PROCESS on the run "
                "machine (sound by prefix-monotonicity) and applies a HARD "
                "floor of 6, with an underpowered result stamped "
                "non_degenerate:false so it is scoring-excluded rather than "
                "recorded as a direction. The BAR (1 x SE, rel floor 0) is "
                "unchanged."),
            "screen_soundness_note": (
                "_run_cell at K episodes is a bit-identical prefix of the "
                "same cell at N>K (arm_cell resets all RNG at cell entry, the "
                "env is seeded per cell, nothing in the loop reads `episodes`) "
                "and PolicyDecomposition.reset() is explicitly NOT called on "
                "the agent's per-episode reset, so decomp_n_decomposed_midexec "
                "accumulates monotonically. Screen-matched therefore IMPLIES "
                "full-schedule matched; the converse does not hold, so a "
                "screen miss costs a candidate and never a false verdict. "
                "Verified directly 2026-08-03 (darwin-arm64, seed 11, both "
                "arms, 2ep vs 4ep): prefix identical, all counters monotone. "
                "The manifest's screen_soundness_check re-tests this against "
                "the measurement rather than asserting it."),
            "dv_symmetry_declaration": (
                "DV = mean per-tick environment harm over the post-divergence "
                "window, restricted to fresh e3-selection ticks. Its symmetry "
                "group is permutation of the ticks it averages over (a set-"
                "aggregate) and any transform of the candidate scores that "
                "preserves the selection argmax (broadcast additive constants, "
                "monotone rescalings). The manipulation is invariant under "
                "NEITHER: harm-aware selection applies a PER-LEAF penalty "
                "(harm_bias) plus a categorical per-leaf override "
                "(select_harm_aware_leaves) across the retiling candidates, "
                "not a broadcast scalar, so it moves the argmax and hence the "
                "committed action sequence; and it acts at selection time, so "
                "it changes WHICH ticks occur rather than relabelling a fixed "
                "set. Confirmed empirically in 867a: both matched seeds "
                "diverged in action sequence (divergence_tick=60) with "
                "differing per-seed harm means. Identical for both arms of "
                "this run -- the OFF arm carries no manipulation at all."),
            "gov_reuse_1_note": (
                "Decisive readout = paired per-seed windowed harm delta over "
                ">= 6 matched seeds. Checked v3_exq_867a_..._20260802T203309Z_v3 "
                "(n=2 matched) and v3_exq_867_..._20260802T031937Z_v3 "
                "(manipulation structurally inert, no matched tier). Neither "
                "records nor permits deriving the readout at the required n -- "
                "additional matched seeds are cells that were never run. Not "
                "recoverable -> run. Arm-reuse against 867a's OFF mint would "
                "additionally refuse on substrate_hash (ree_core has moved "
                "since 2026-08-02), so no try_reuse_cell call is made."),
            "matched_abort_rates_design_note": (
                "Unchanged from 867/867a: (a) per-seed decomp_n_decomposed_"
                "midexec covariate reported for every measured seed; (b) "
                "load-bearing C1 restricted to both_decompose_seeds, with the "
                "tier recomputed from the MEASUREMENT cells (the screen "
                "selects which seeds are measured, it does not assign tiers)."),
        },
    }

    out_path = write_flat_manifest(
        manifest, None, dry_run=args.dry_run, config=cfg_record,
        seeds=measurement_seeds, script_path=Path(__file__),
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
        f"  screened={s['n_candidates_screened']} "
        f"screen_matched={s['n_screen_matched']} "
        f"screen_yield_met={s['screen_yield_met']}", flush=True)
    print(
        f"  both_decompose_seeds={result['seed_tiers_measured']['both_decompose']} "
        f"n_pairs={s['n_windowed_pairs']}/{s['min_decomposing_seeds_required']} "
        f"harm_delta_mean={s['harm_delta_mean_post_divergence']:.6g} "
        f"pe_delta_mean={s['pe_delta_mean_post_divergence']:.6g} "
        f"rel_improvement={s['rel_improvement']:.4f} "
        f"max_z_harm_a_norm_on_arm={s['max_z_harm_a_norm_on_arm']:.4f}", flush=True)
    print(
        f"  on_only_seeds={result['seed_tiers_measured']['on_only_decompose']} "
        f"off_only_seeds={result['seed_tiers_measured']['off_only_decompose']} "
        f"(unmatched abort-rate tiers, non-gating; mean delta="
        f"{s['unmatched_harm_delta_mean']})", flush=True)
    print(
        f"  screen_soundness_held={result['screen_soundness_check']['held']} "
        f"violations={result['screen_soundness_check']['monotonicity_violations']}",
        flush=True)
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
        # Smoke assertion: the decisive readout must be NON-TRIVIALLY ENGAGED
        # before this design is worth a multi-hour grid. A structural zero on
        # decomp_n_harm_bias_nonzero is exactly 867's failure, and is far
        # cheaper to catch here than after the full run.
        on_bias = max(
            (r["decomp_n_harm_bias_nonzero"] for r in rows if r["arm_id"] == ARM_ON),
            default=0)
        off_bias = max(
            (r["decomp_n_harm_bias_nonzero"] for r in rows if r["arm_id"] == ARM_OFF),
            default=0)
        print(f"[smoke] on_arm_harm_bias_nonzero_max={on_bias} "
              f"off_arm_harm_bias_nonzero_max={off_bias}", flush=True)
        assert on_bias > 0, (
            "SMOKE FAIL: ARM_SELECTION_ON never engaged the graded harm bias "
            "(decomp_n_harm_bias_nonzero == 0 on every cell). This is 867's "
            "inert-manipulation failure -- do not queue.")
        assert off_bias == 0, (
            "SMOKE FAIL: ARM_SELECTION_OFF engaged the harm bias, which must "
            "be a bit-identical no-op in that arm.")

    outcome_norm = str(result["outcome"]).upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    return outcome_emit, str(out_path), bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry_run = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path,
                     dry_run=_dry_run)
