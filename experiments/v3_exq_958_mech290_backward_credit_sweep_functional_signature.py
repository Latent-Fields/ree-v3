"""
V3-EXQ-958: MECH-290 backward trajectory credit sweep FUNCTIONAL-SIGNATURE
confirmer (wall-independent).

WHY THIS RUN EXISTS (GOV-CONFIRM-1-style). MECH-290 -- the hippocampal
backward temporal credit sweep (Foster & Wilson 2006 reverse-replay analog)
-- has a fully BUILT V3 substrate (ree_core/hippocampal/module.py
HippocampalModule.record_committed_trajectory / .backward_credit_sweep /
.reset_committed_trajectory, wired into ree_core/agent.py at commit-entry and
at BetaGate-release; REEConfig.use_backward_credit_sweep, IMPLEMENTED
2026-04-24 per ree-v3/CLAUDE.md) but has NEVER been experimentally tested:
zero entries in REE_assembly/evidence/experiments/claim_evidence.v1.json for
MECH-290, zero autopsies, zero substrate_queue.json failure_record entries.
The claim has sat "candidate/v3_pending" with no direct evidence since
registration (2026-04-24).

This experiment is deliberately WALL-INDEPENDENT: it does NOT ask whether any
particular environment/curriculum naturally drives goal-completion events
that trigger the sweep (that would confound the mechanism's own correctness
with an unrelated env-competence question, exactly the trap MECH-266's
cluster fell into). Instead it drives the built substrate DIRECTLY with
prescribed committed trajectories and outcome-quality values, and confirms
the real config -> HippocampalModule -> ResidueField wiring exhibits the
three functional signatures MECH-290's claim asserts (claims.yaml MECH-290
functional_restatement: terminal outcome propagates backward through
preceding states, discounted, consistent with reverse replay at reward
endpoints):

  S1  GEOMETRIC-DISCOUNT LAW. At trajectory completion, each preceding
      z_world waypoint receives VALENCE_WANTING credit = outcome_quality *
      gamma^(T-1-t), where T is trajectory length and t is the 0-indexed step
      (t=T-1, the endpoint, gets the full outcome_quality; earlier steps are
      geometrically discounted). Confirmed per-waypoint against an
      INDEPENDENT reference computation of the same closed-form law, read
      directly off each waypoint's own residue-field valence-vector entry
      (not a blended/aggregate readout -- see EMPIRICAL PROBE below for why
      this distinction matters). FALSIFIER: any per-waypoint credit that
      does not match the closed-form reference within numerical tolerance,
      or a credit curve that is not monotonically increasing toward the
      endpoint.

  S2  QUALITY GATE (retroactive reward assignment is conditional, not
      unconditional). A trajectory completed with outcome_quality below
      HippocampalConfig.backward_sweep_min_quality (default 0.6) receives
      NO credit at all (n_steps_swept == 0, no valence write at any
      waypoint) -- "low-quality completions do not deserve retroactive
      reward assignment" per the module's own docstring. The SAME committed
      trajectory swept again at a supra-threshold outcome_quality DOES
      receive credit. FALSIFIER: a sub-threshold sweep producing any
      nonzero credit, or a supra-threshold sweep on the same trajectory
      producing none.

  S3  GAMMA DOSE-RESPONSE (discount steepness is a real, tunable parameter,
      not a structural no-op). For a fixed outcome_quality and trajectory
      length, the ratio of the earliest waypoint's credit to the terminal
      waypoint's credit is exactly gamma^(T-1) -- swept across
      backward_sweep_gamma in {0.5, 0.8, 0.95}, this ratio must vary and be
      monotonically increasing in gamma (a higher gamma discounts less,
      so earlier waypoints keep a larger fraction of the terminal credit).
      FALSIFIER: a flat or non-monotone ratio across the gamma sweep (the
      dose-sweep saturation-fingerprint pattern -- see CLAUDE.md /
      queue-experiment Step 3.5 "Label / metric calibration").

BACKWARD-COMPAT / FLAG-GATING CHECK (P0, not a claim signature but a
correctness precondition every downstream consumer relies on):
use_backward_credit_sweep=False must make record_committed_trajectory() and
backward_credit_sweep() unconditional no-ops (buffer stays None, sweep
returns {}) -- confirmed once per run on a separately-built OFF agent.

EMPIRICAL PROBE (Step 2.5a, closes a doc-vs-runtime gap found while writing
this script -- NOT a script bug, a caller-discipline note worth recording).
Two things a naive driver gets wrong that this script gets right:
  (a) use_backward_credit_sweep (and the two backward_sweep_* knobs) are
      HippocampalConfig fields, not REEConfig fields. Setting
      `cfg.use_backward_credit_sweep = True` AFTER `REEConfig.from_dims(...)`
      returns silently sets an attribute nobody reads (agent.hippocampal.
      config.use_backward_credit_sweep stays False, no error). The three
      knobs MUST be passed as from_dims(...) kwargs, which is how from_dims
      threads them onto config.hippocampal at construction (config.py
      lines ~8929-8931). Verified live in a throwaway probe: the mis-set
      form silently no-ops the whole mechanism (buffer never populated,
      sweep always returns {}); the from_dims kwarg form is confirmed live
      to reproduce the exact closed-form credit values to ~3e-8 float
      tolerance.
  (b) ResidueField.update_valence() writes to the NEAREST ACTIVE RBF
      center to a query point, not to an exact per-waypoint slot -- so
      reading credit back via evaluate_valence(z_world) at each waypoint
      BLENDS contributions from every nearby active center (Gaussian RBF
      weighting) and does not reproduce the closed-form per-step law
      exactly. This script places waypoints far enough apart (probed and
      confirmed empirically) that each accumulate() call allocates its OWN
      center, and reads credit back via each waypoint's own known
      center_idx directly off rbf_field.valence_vecs -- the exact
      per-step ground truth, not a spatially-blended proxy.

claim_ids = [MECH-290] ONLY. This experiment does not exercise the upstream
completion-signal machinery (ARC-028 / MECH-105 BetaGate release) or the
CEM-proposal commit-entry hook in select_action() -- it drives
record_committed_trajectory() / backward_credit_sweep() directly, which is
the actual mechanism MECH-290 asserts. PASS => the built substrate realises
the geometric backward-credit-assignment transfer function MECH-290 claims,
exactly, over a distribution of seeded trajectory geometries and outcome
qualities (functional-signature support, wall-independent). FAIL => the
built substrate does not match the claimed mechanism (weakens).

EXPERIMENT_PURPOSE = evidence. No training, no gradient, no environment; a
controlled functional probe of the real (non-trainable, pure-arithmetic)
mechanism, mirroring the V3-EXQ-776 MECH-279 GOV-CONFIRM-1 template.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from _metrics import check_degeneracy, p0_readiness_gate, P0NotReady  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.predictors.e2_fast import Trajectory  # noqa: E402
from ree_core.residue.field import VALENCE_WANTING  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_958_mech290_backward_credit_sweep_functional_signature"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS = ["MECH-290"]

# ----------------------------------------------------------------------------
# Pre-registered constants (NOT derived from the run's own statistics)
# ----------------------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]

# Agent build dims (match the MECH-279 GOV-CONFIRM-1 template idiom).
BODY_OBS_DIM = 12
WORLD_OBS_DIM = 250
ACTION_DIM = 4
SELF_DIM = 8  # z_self placeholder dim; irrelevant to this mechanism

# Trajectory geometry.
TRAJ_LEN = 5  # T
# Waypoint spacing along one world-latent axis. Probed empirically: at this
# spacing each accumulate() call allocates its OWN RBF center (no cross-talk
# in the read-back), confirmed by exact match to the closed-form law to
# ~3e-8 tolerance in the empirical probe described in the module docstring.
SPACING_BASE = 50.0
SPACING_JITTER = 2.0  # per-seed jitter, stays >> RBF bandwidth
ACCUMULATE_MAGNITUDE = 0.001  # tiny harm_magnitude just to allocate a center

# Nominal (P0 / S1) mechanism parameters -- the REEConfig defaults.
NOMINAL_GAMMA = 0.9
NOMINAL_MIN_QUALITY = 0.6
NOMINAL_OUTCOME_QUALITY = 0.9  # supra-threshold positive control

# S1 numeric tolerance (float32 arithmetic; probe measured ~3e-8 max diff).
S1_TOLERANCE = 1e-4

# S2 quality-gate probe values.
S2_SUBTHRESHOLD_QUALITY = NOMINAL_MIN_QUALITY - 0.1  # 0.5, below gate
S2_SUPRATHRESHOLD_QUALITY = NOMINAL_MIN_QUALITY + 0.2  # 0.8, above gate

# S3 gamma dose sweep.
S3_GAMMAS = [0.5, 0.8, 0.95]
S3_OUTCOME_QUALITY = 0.9

# P0 positive control thresholds.
P0_MIN_STEPS_SWEPT = TRAJ_LEN
P0_MIN_TERMINAL_CREDIT = 0.5  # terminal credit == outcome_quality (0.9); floor with margin

# Progress-instrumentation denominator: T (S1 per-waypoint) + 2 (S2 checks)
# + len(S3_GAMMAS) (S3 per-gamma) printed per seed.
EPISODES_PER_RUN = TRAJ_LEN + 2 + len(S3_GAMMAS)


# ----------------------------------------------------------------------------
# Real-substrate agent construction
# ----------------------------------------------------------------------------
def _build_agent(
    use_sweep: bool = True,
    gamma: float = NOMINAL_GAMMA,
    min_quality: float = NOMINAL_MIN_QUALITY,
) -> REEAgent:
    """Build a REAL V3 agent with MECH-290 backward-credit-sweep enabled via
    the from_dims(...) kwargs (see EMPIRICAL PROBE in the module docstring
    for why this must NOT be set post-hoc as cfg.use_backward_credit_sweep)."""
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        use_backward_credit_sweep=use_sweep,
        backward_sweep_gamma=gamma,
        backward_sweep_min_quality=min_quality,
    )
    return REEAgent(cfg)


def _make_trajectory(
    agent: REEAgent, rng: np.random.Generator, length: int = TRAJ_LEN
) -> Tuple[Trajectory, List[torch.Tensor], List[int]]:
    """Build a committed trajectory of `length` well-separated z_world
    waypoints, pre-activating one residue center per waypoint via
    accumulate(). Returns (trajectory, world_states, center_idxs)."""
    world_dim = agent.residue_field.rbf_field.centers.shape[1]
    world_states: List[torch.Tensor] = []
    for t in range(length):
        z = torch.zeros(1, world_dim)
        jitter = float(rng.uniform(-SPACING_JITTER, SPACING_JITTER))
        z[0, 0] = float(t) * SPACING_BASE + jitter
        world_states.append(z)

    states = [torch.zeros(1, SELF_DIM) for _ in range(length)]
    actions = torch.zeros(1, length, ACTION_DIM)
    traj = Trajectory(
        states=states,
        actions=actions,
        world_states=world_states,
        action_objects=None,
        is_reverse=False,
        memory_strength=1.0,
        arousal_tag=0.0,
        hypothesis_tag=False,
        metadata=None,
    )

    center_idxs: List[int] = []
    for t in range(length):
        res = agent.residue_field.accumulate(
            world_states[t], harm_magnitude=ACCUMULATE_MAGNITUDE
        )
        center_idxs.append(int(res["center_idx"].item()))
    return traj, world_states, center_idxs


def _read_wanting(agent: REEAgent, center_idxs: List[int]) -> List[float]:
    """Read the exact per-center VALENCE_WANTING value (ground truth, not a
    spatially-blended evaluate_valence() readout -- see module docstring)."""
    return [
        float(agent.residue_field.rbf_field.valence_vecs[c, VALENCE_WANTING].item())
        for c in center_idxs
    ]


# ----------------------------------------------------------------------------
# Signature measurements (per seed)
# ----------------------------------------------------------------------------
def _measure_s1(seed: int, ep_base: int) -> Dict[str, Any]:
    """S1 geometric-discount law: each waypoint's credit == outcome_quality *
    gamma^(T-1-t), read off its own center, compared to an independent
    closed-form reference."""
    rng = np.random.default_rng((seed << 4) + 1)
    agent = _build_agent(use_sweep=True, gamma=NOMINAL_GAMMA, min_quality=NOMINAL_MIN_QUALITY)
    traj, world_states, center_idxs = _make_trajectory(agent, rng, TRAJ_LEN)

    before = _read_wanting(agent, center_idxs)
    agent.hippocampal.record_committed_trajectory(traj)
    result = agent.hippocampal.backward_credit_sweep(outcome_quality=NOMINAL_OUTCOME_QUALITY)
    after = _read_wanting(agent, center_idxs)
    delta = [a - b for a, b in zip(after, before)]

    T = TRAJ_LEN
    reference = [
        NOMINAL_OUTCOME_QUALITY * (NOMINAL_GAMMA ** (T - 1 - t)) for t in range(T)
    ]
    diffs = [abs(d - r) for d, r in zip(delta, reference)]
    all_match = all(diff <= S1_TOLERANCE for diff in diffs)
    monotone = all(delta[t + 1] >= delta[t] for t in range(T - 1))
    n_steps_swept = int(result.get("n_steps_swept", 0))

    for t in range(T):
        print(
            f"  [probe] S1 seed={seed} ep {ep_base + t + 1}/{EPISODES_PER_RUN} "
            f"t={t} credit={delta[t]:.6f} ref={reference[t]:.6f} "
            f"diff={diffs[t]:.2e}",
            flush=True,
        )

    s1 = bool(all_match and monotone and n_steps_swept == T)
    return {
        "S1_geometric_discount_law": s1,
        "s1_delta": delta,
        "s1_reference": reference,
        "s1_max_abs_diff": max(diffs),
        "s1_monotone_increasing": bool(monotone),
        "s1_n_steps_swept": n_steps_swept,
        "s1_result": result,
    }


def _measure_s2(seed: int, ep_base: int) -> Dict[str, Any]:
    """S2 quality gate: sub-threshold outcome_quality writes nothing; the
    SAME trajectory swept again supra-threshold DOES write."""
    rng = np.random.default_rng((seed << 4) + 2)
    agent = _build_agent(use_sweep=True, gamma=NOMINAL_GAMMA, min_quality=NOMINAL_MIN_QUALITY)
    traj, world_states, center_idxs = _make_trajectory(agent, rng, TRAJ_LEN)

    before = _read_wanting(agent, center_idxs)
    agent.hippocampal.record_committed_trajectory(traj)
    sub_result = agent.hippocampal.backward_credit_sweep(
        outcome_quality=S2_SUBTHRESHOLD_QUALITY
    )
    after_sub = _read_wanting(agent, center_idxs)
    sub_no_write = all(
        abs(a - b) <= S1_TOLERANCE for a, b in zip(after_sub, before)
    ) and int(sub_result.get("n_steps_swept", 0)) == 0
    print(
        f"  [probe] S2 seed={seed} ep {ep_base + 1}/{EPISODES_PER_RUN} "
        f"subthreshold quality={S2_SUBTHRESHOLD_QUALITY:.2f} "
        f"n_steps_swept={sub_result.get('n_steps_swept', 0)} no_write={sub_no_write}",
        flush=True,
    )

    agent.hippocampal.record_committed_trajectory(traj)
    supra_result = agent.hippocampal.backward_credit_sweep(
        outcome_quality=S2_SUPRATHRESHOLD_QUALITY
    )
    after_supra = _read_wanting(agent, center_idxs)
    supra_wrote = int(supra_result.get("n_steps_swept", 0)) == TRAJ_LEN and any(
        abs(a - b) > S1_TOLERANCE for a, b in zip(after_supra, after_sub)
    )
    print(
        f"  [probe] S2 seed={seed} ep {ep_base + 2}/{EPISODES_PER_RUN} "
        f"suprathreshold quality={S2_SUPRATHRESHOLD_QUALITY:.2f} "
        f"n_steps_swept={supra_result.get('n_steps_swept', 0)} wrote={supra_wrote}",
        flush=True,
    )

    s2 = bool(sub_no_write and supra_wrote)
    return {
        "S2_quality_gate": s2,
        "s2_sub_no_write": bool(sub_no_write),
        "s2_supra_wrote": bool(supra_wrote),
        "s2_sub_result": sub_result,
        "s2_supra_result": supra_result,
    }


def _measure_s3(seed: int, ep_base: int) -> Dict[str, Any]:
    """S3 gamma dose-response: earliest/terminal credit ratio == gamma^(T-1),
    strictly increasing across the ascending gamma sweep."""
    rng = np.random.default_rng((seed << 4) + 3)
    T = TRAJ_LEN
    ratios: List[float] = []
    per_gamma: List[Dict[str, Any]] = []
    for j, gamma in enumerate(S3_GAMMAS):
        agent = _build_agent(use_sweep=True, gamma=gamma, min_quality=NOMINAL_MIN_QUALITY)
        traj, world_states, center_idxs = _make_trajectory(agent, rng, T)
        before = _read_wanting(agent, center_idxs)
        agent.hippocampal.record_committed_trajectory(traj)
        agent.hippocampal.backward_credit_sweep(outcome_quality=S3_OUTCOME_QUALITY)
        after = _read_wanting(agent, center_idxs)
        delta = [a - b for a, b in zip(after, before)]
        ratio = delta[0] / delta[-1] if delta[-1] > 0 else float("nan")
        expected_ratio = gamma ** (T - 1)
        ratios.append(ratio)
        per_gamma.append(
            {
                "gamma": gamma,
                "earliest_credit": delta[0],
                "terminal_credit": delta[-1],
                "observed_ratio": ratio,
                "expected_ratio": expected_ratio,
                "ratio_match": abs(ratio - expected_ratio) <= S1_TOLERANCE,
            }
        )
        print(
            f"  [probe] S3 seed={seed} ep {ep_base + j + 1}/{EPISODES_PER_RUN} "
            f"gamma={gamma:.2f} ratio={ratio:.6f} expected={expected_ratio:.6f}",
            flush=True,
        )

    all_ratio_match = all(pg["ratio_match"] for pg in per_gamma)
    strictly_increasing = all(
        ratios[i] < ratios[i + 1] for i in range(len(ratios) - 1)
    )
    s3 = bool(all_ratio_match and strictly_increasing)
    return {
        "S3_gamma_dose_response": s3,
        "s3_ratios": ratios,
        "s3_per_gamma": per_gamma,
        "s3_all_ratio_match": bool(all_ratio_match),
        "s3_strictly_increasing": bool(strictly_increasing),
    }


def _p0_positive_control() -> Tuple[list, Dict[str, Any]]:
    """Confirm the built substrate fires under nominal config: >=T waypoints
    swept, terminal credit clears a floor. Also confirms backward-compat:
    flag OFF makes both methods unconditional no-ops. Raises P0NotReady if
    the ON path is inert/miswired."""
    rng = np.random.default_rng(999)
    agent = _build_agent(use_sweep=True, gamma=NOMINAL_GAMMA, min_quality=NOMINAL_MIN_QUALITY)
    traj, world_states, center_idxs = _make_trajectory(agent, rng, TRAJ_LEN)
    before = _read_wanting(agent, center_idxs)
    agent.hippocampal.record_committed_trajectory(traj)
    result = agent.hippocampal.backward_credit_sweep(outcome_quality=NOMINAL_OUTCOME_QUALITY)
    after = _read_wanting(agent, center_idxs)
    terminal_credit = after[-1] - before[-1]

    agent_off = _build_agent(use_sweep=False)
    traj_off, _, _ = _make_trajectory(agent_off, rng, TRAJ_LEN)
    agent_off.hippocampal.record_committed_trajectory(traj_off)
    buffer_stayed_none = agent_off.hippocampal._committed_trajectory_buffer is None
    off_result = agent_off.hippocampal.backward_credit_sweep(outcome_quality=NOMINAL_OUTCOME_QUALITY)
    off_is_noop = off_result == {}

    diag = {
        "p0_n_steps_swept": int(result.get("n_steps_swept", 0)),
        "p0_terminal_credit": terminal_credit,
        "p0_buffer_stayed_none_when_off": bool(buffer_stayed_none),
        "p0_sweep_noop_when_off": bool(off_is_noop),
    }
    # Backward-compat break is itself a readiness failure -- if the flag does
    # not gate cleanly, every OTHER experiment's OFF control is potentially
    # contaminated. Fold it into the SAME p0_readiness_gate(...) call (not a
    # post-hoc append) so an unmet check actually RAISES P0NotReady and
    # aborts before any S-series signature is read, rather than silently
    # passing through as an informational entry.
    preconditions = p0_readiness_gate(
        [
            {
                "name": "backward_sweep_fires_on_positive_control",
                "measured": float(result.get("n_steps_swept", 0)),
                "threshold": float(P0_MIN_STEPS_SWEPT),
                "direction": "lower",
                "control": f"trajectory of {TRAJ_LEN} well-separated waypoints, "
                f"outcome_quality={NOMINAL_OUTCOME_QUALITY} (supra-threshold)",
            },
            {
                "name": "terminal_waypoint_credit_clears_floor",
                "measured": float(terminal_credit),
                "threshold": float(P0_MIN_TERMINAL_CREDIT),
                "direction": "lower",
                "control": "terminal waypoint credit should equal outcome_quality exactly",
            },
            {
                "name": "flag_off_is_unconditional_noop",
                "description": "use_backward_credit_sweep=False must make both "
                "record_committed_trajectory and backward_credit_sweep no-ops",
                "measured": 1.0 if (buffer_stayed_none and off_is_noop) else 0.0,
                "threshold": 1.0,
                "direction": "lower",
                "control": "separately-built OFF agent, identical trajectory",
            },
        ]
    )
    return preconditions, diag


# ----------------------------------------------------------------------------
def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    global SEEDS
    t0 = time.perf_counter()
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    seeds = SEEDS[:2] if dry_run else SEEDS

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    full_config = {
        "experiment_type": EXPERIMENT_TYPE,
        "body_obs_dim": BODY_OBS_DIM,
        "world_obs_dim": WORLD_OBS_DIM,
        "action_dim": ACTION_DIM,
        "seeds": SEEDS,
        "traj_len": TRAJ_LEN,
        "spacing_base": SPACING_BASE,
        "spacing_jitter": SPACING_JITTER,
        "nominal_gamma": NOMINAL_GAMMA,
        "nominal_min_quality": NOMINAL_MIN_QUALITY,
        "nominal_outcome_quality": NOMINAL_OUTCOME_QUALITY,
        "s2_subthreshold_quality": S2_SUBTHRESHOLD_QUALITY,
        "s2_suprathreshold_quality": S2_SUPRATHRESHOLD_QUALITY,
        "s3_gammas": S3_GAMMAS,
        "s3_outcome_quality": S3_OUTCOME_QUALITY,
    }

    # ---- P0 positive control + abort gate ----
    try:
        preconditions, p0_diag = _p0_positive_control()
    except P0NotReady as e:
        manifest = {
            "run_id": run_id,
            "experiment_type": EXPERIMENT_TYPE,
            "architecture_epoch": "ree_hybrid_guardrails_v1",
            "claim_ids": CLAIM_IDS,
            "experiment_purpose": "diagnostic",
            "outcome": "FAIL",
            "timestamp_utc": timestamp,
            "non_degenerate": False,
            "degeneracy_reason": "P0 positive control inert: " + e.reason,
            "interpretation": {
                "label": "substrate_not_ready_requeue",
                "preconditions": e.preconditions,
            },
            "dry_run": dry_run,
        }
        out_path = write_flat_manifest(
            manifest, out_dir, dry_run=dry_run, config=full_config, seeds=SEEDS,
            script_path=Path(__file__), started_at=t0,
        )
        print(f"Manifest written: {out_path}", flush=True)
        print("Outcome: FAIL (substrate_not_ready_requeue)", flush=True)
        manifest["manifest_path"] = str(out_path)
        return manifest

    # ---- main measurement: three signatures per seed ----
    per_seed: List[Dict[str, Any]] = []
    s1_hits = s2_hits = s3_hits = 0
    for seed in seeds:
        print(f"Seed {seed} Condition functional_signature", flush=True)
        s1 = _measure_s1(seed, ep_base=0)
        s2 = _measure_s2(seed, ep_base=TRAJ_LEN)
        s3 = _measure_s3(seed, ep_base=TRAJ_LEN + 2)
        s1_hits += int(s1["S1_geometric_discount_law"])
        s2_hits += int(s2["S2_quality_gate"])
        s3_hits += int(s3["S3_gamma_dose_response"])
        row: Dict[str, Any] = {"seed": seed}
        row.update(s1)
        row.update(s2)
        row.update(s3)
        per_seed.append(row)
        seed_pass = (
            s1["S1_geometric_discount_law"]
            and s2["S2_quality_gate"]
            and s3["S3_gamma_dose_response"]
        )
        print(f"verdict: {'PASS' if seed_pass else 'FAIL'}", flush=True)

    n = len(seeds)
    S1 = s1_hits == n
    S2 = s2_hits == n
    S3 = s3_hits == n
    passed = bool(S1 and S2 and S3)
    outcome = "PASS" if passed else "FAIL"

    # ---- non-degeneracy net: the S1 credit curve and the S3 ratio sweep
    # must show genuine spread, not a pinned/degenerate readout. ----
    degen = check_degeneracy(
        {
            "s1_delta_over_waypoint_sweep": per_seed[0]["s1_delta"],
            "s3_ratio_over_gamma_sweep": per_seed[0]["s3_ratios"],
        }
    )

    criteria = [
        {"name": "S1_geometric_discount_law", "load_bearing": True, "passed": S1},
        {"name": "S2_quality_gate", "load_bearing": True, "passed": S2},
        {"name": "S3_gamma_dose_response", "load_bearing": True, "passed": S3},
    ]

    summary = {
        "S1_geometric_discount_law": S1,
        "S2_quality_gate": S2,
        "S3_gamma_dose_response": S3,
        "s1_seed_hits": s1_hits,
        "s2_seed_hits": s2_hits,
        "s3_seed_hits": s3_hits,
        "n_seeds": n,
        "s1_max_abs_diff_example": per_seed[0]["s1_max_abs_diff"],
        "s1_delta_example": per_seed[0]["s1_delta"],
        "s3_ratios_example": per_seed[0]["s3_ratios"],
    }

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": "supports" if passed else "weakens",
        "evidence_direction_note": (
            "Wall-independent functional-signature confirmation of the BUILT "
            "hippocampal backward-credit-sweep substrate (MECH-290): the real "
            "config -> HippocampalModule -> ResidueField wiring realises the "
            "geometric backward-credit-assignment transfer function "
            "(discount law, quality gate, gamma dose-response) exactly as "
            "MECH-290 asserts, over seeded trajectory-geometry distributions. "
            "This is substrate/functional support only; it does not claim any "
            "particular environment naturally drives goal-completion events "
            "that trigger the sweep (a separate, wall-dependent question)."
        ),
        "timestamp_utc": timestamp,
        "dry_run": dry_run,
        "p0_readiness": p0_diag,
        "interpretation": {
            "label": "backward_credit_sweep_functional_signature_confirmed"
            if passed
            else "backward_credit_sweep_functional_signature_not_observed",
            "preconditions": preconditions,
            "criteria": criteria,
            "criteria_non_degenerate": {
                c["name"]: bool(degen["non_degenerate"]) for c in criteria
            },
        },
        "acceptance_criteria": summary,
        "summary": summary,
        "per_seed": per_seed,
        "constants": {
            "SEEDS": SEEDS,
            "TRAJ_LEN": TRAJ_LEN,
            "NOMINAL_GAMMA": NOMINAL_GAMMA,
            "NOMINAL_MIN_QUALITY": NOMINAL_MIN_QUALITY,
            "NOMINAL_OUTCOME_QUALITY": NOMINAL_OUTCOME_QUALITY,
            "S2_SUBTHRESHOLD_QUALITY": S2_SUBTHRESHOLD_QUALITY,
            "S2_SUPRATHRESHOLD_QUALITY": S2_SUPRATHRESHOLD_QUALITY,
            "S3_GAMMAS": S3_GAMMAS,
            "S3_OUTCOME_QUALITY": S3_OUTCOME_QUALITY,
        },
    }
    manifest.update(degen)

    out_path = write_flat_manifest(
        manifest, out_dir, dry_run=dry_run, config=full_config, seeds=SEEDS,
        script_path=Path(__file__), started_at=t0,
    )
    print(f"Manifest written: {out_path}", flush=True)

    print(f"Outcome: {outcome}", flush=True)
    print(f"  S1(discount-law)={S1} S2(quality-gate)={S2} S3(gamma-dose)={S3}", flush=True)
    print(
        f"  s1_max_abs_diff={summary['s1_max_abs_diff_example']:.2e} "
        f"s3_ratios={summary['s3_ratios_example']}",
        flush=True,
    )
    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run_experiment(dry_run=args.dry_run)
    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(result.get("manifest_path", "/dev/null")),
        dry_run=args.dry_run,
    )
