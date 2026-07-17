"""
V3-EXQ-776: MECH-279 PAG freeze-gate FUNCTIONAL-SIGNATURE confirmer (wall-independent).

WHY THIS RUN EXISTS (GOV-CONFIRM-1). MECH-279 -- the PAG-analog committed-freeze
gate -- has strong literature backing (lit_conf ~0.90) and a BUILT V3 substrate
(ree_core/pag/freeze_gate.py, wired in agent.py; REEConfig.use_pag_freeze_gate)
but ZERO valid experimental evidence: the entire EXQ-471/475/483/490/524 cohort
that was meant to test it is contaminated + superseded (update_z_goal TypeError
swallowed; see claims.yaml MECH-279 evidence_quality_note), and every downstream
483d/483e attempt to see the gate act behaviourally FAILED for a purely
ENVIRONMENTAL reason: in the fishtank env, z_harm_a magnitudes never sustain
above the commit threshold, so pag_release_count == 0 across all runs and the
gate never fired (failure_autopsy_V3-EXQ-483d/483e). That is a WALL (an env /
consumer-input-threshold ceiling), not a statement about MECH-279's own machinery.

This experiment is deliberately WALL-INDEPENDENT: it does NOT ask whether any env
naturally drives the gate; it drives the gate DIRECTLY with prescribed z_harm_a(t)
trajectories and confirms the built substrate exhibits the three functional
signatures MECH-279's claim asserts. Those signatures are the claim's own
falsifiable content (claims.yaml MECH-279 functional_restatement):
    freeze_commit when z_harm_a * duration > theta_freeze;
    freeze_active sustained until z_harm_a < exit_threshold = theta_freeze * gaba_tone;
    lower gaba_tone => sustained freeze under mild harm (stuporous catatonia);
    higher gaba_tone => accelerated exit (the GABA-agonist/benzo clinical prediction).

SUBSTRATE FAITHFULNESS (why this is NOT the C4-C7 unit contract). The C4-C7
contracts in tests/contracts/test_mech_279_pag_freeze_gate.py hand-construct a
PAGFreezeGateConfig and check a single trajectory each. This experiment instead
builds a REAL V3 agent via REEConfig.from_dims(...) with use_pag_freeze_gate=True
+ use_gabaergic_decay=True, and drives agent.pag_freeze_gate exactly as the live
select_action() path does -- ticking with gaba_tone read off the REAL
agent.gabaergic_decay regulator (agent.py:7384-7398). So it confirms the whole
config -> gate-instantiation -> gaba_tone-coupling wiring delivers the claimed
transfer function, over a DISTRIBUTION of seeded input trajectories (not one
hand-picked schedule), with pre-registered analytic predictions. It is the
representation/functional-signature confirming DV the GOV-CONFIRM-1 remit asks
for; it does NOT claim the fishtank behaviourally expresses catatonia (that
remains the SD-037 483e substrate-ceiling, a separate wall).

THREE PRE-REGISTERED SIGNATURES (theta_freeze=2.0, duration_input_threshold=0.4
-- the REEConfig defaults, read back off the built gate for provenance):

  S1  COMMIT-LAW (entry is an INTEGRATOR, not a memoryless z-threshold). For a
      sustained constant drive z, the gate commits at exactly the tick where the
      running product z*duration first exceeds theta_freeze -- i.e. commit latency
      = the first d with (z*d) > theta_freeze, which DECREASES with z. A drive
      z <= duration_input_threshold never commits (the counter never accumulates).
      Confirmed by comparing the real gate's commit tick to an INDEPENDENT
      reimplementation of the z*duration>theta rule for each z, over a per-seed
      jittered sweep. FALSIFIER: duration-independent (instant) commit, or the
      sub-threshold drive committing.

  S2  COMMITTED-STATE HYSTERESIS (the load-bearing "committed state" vs "just no
      movement" falsifier). The entry boundary (integrator on z*duration) and the
      exit boundary (instantaneous z < theta*gaba_tone) are DISSOCIATED, so the
      gate is a committed state with its own exit criterion -- NOT a memoryless
      function of the current drive. Demonstrated at low tone (gaba_tone=0.15 =>
      exit_threshold=0.30) with a mild HOLD drive Z_hold in (0.30, 0.40): it sits
      ABOVE exit_threshold (so it sustains an already-committed freeze) yet BELOW
      duration_input_threshold=0.40 (so from REST it can never even start a
      freeze). Measured: a committed gate held at Z_hold stays frozen the whole
      probe window; a fresh gate driven with the IDENTICAL Z_hold never freezes.
      The memoryless null predicts these are equal (same drive -> same state); the
      observed persistence gap ~1.0 falsifies it. This directly instantiates the
      stuporous-catatonia reading: a mild sustained harm too weak to initiate a
      freeze nonetheless SUSTAINS one when GABAergic tone is low.

  S3  GABA-TONE EXIT CONTROL (benzo accelerates exit / low tone prolongs freeze).
      For one FIXED post-commit decaying drive z(t), the release tick is strictly
      ordered by gaba_tone: exit(tone=1.5) < exit(tone=1.0) < exit(tone=0.5),
      because exit_threshold = theta*gaba_tone rises with tone so the decaying
      drive crosses below it sooner. gaba_tone is taken from the REAL
      agent.gabaergic_decay regulator (config.gaba_tone -> regulator -> gate), so
      this confirms the SD-036-coupled exit-threshold path. FALSIFIER: no monotone
      tone->exit-latency relationship. The tone=0.5 arm's prolonged freeze is the
      catatonia pole; the tone=1.5 arm's fast exit is the GABA-agonist pole.

READINESS. Before reading any signature, a P0 positive-control confirms the built
gate actually fires (>=1 commit under a known-supra-threshold drive, >=1 release
when the drive is removed). If the gate is None (flag miswired) or inert, the run
self-routes substrate_not_ready_requeue (never a MECH-279 verdict).

claim_ids = [MECH-279] ONLY. PASS => the built PAG freeze-gate substrate realises
the committed-freeze transfer function and its gaba-modulated / catatonia
signatures exactly as MECH-279 asserts (functional-signature support,
wall-independent). FAIL => the built substrate does NOT match the claimed
mechanism (weakens).

EXPERIMENT_PURPOSE = evidence. No training, no env; a controlled functional probe
of the real gate over seeded input-trajectory distributions.
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from _metrics import check_degeneracy, p0_readiness_gate, P0NotReady  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_776_mech279_pag_freeze_gate_functional_signature"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS = ["MECH-279"]

# ----------------------------------------------------------------------------
# Pre-registered constants (NOT derived from the run's own statistics)
# ----------------------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]

# Agent build dims (match the MECH-279 contract test idiom).
BODY_OBS_DIM = 12
WORLD_OBS_DIM = 250
ACTION_DIM = 4

# Expected REEConfig defaults for the gate (read back off the built gate for
# provenance; the experiment does NOT set them -- it confirms the defaults).
EXPECTED_THETA_FREEZE = 2.0
EXPECTED_DURATION_INPUT_THRESHOLD = 0.4

# S1 commit-law sweep: sustained-drive levels with comfortably non-integer
# theta/z so a small jitter never lands on a float-equality boundary. Plus one
# sub-duration-threshold level that must NEVER commit.
S1_Z_LEVELS = [0.6, 0.9, 1.3, 1.7, 2.5]
S1_SUBTHRESHOLD_Z = 0.35          # < duration_input_threshold (0.40) -> never commits
S1_SUSTAIN_TICKS = 12             # long enough for the slowest level to commit
S1_JITTER = 0.02                  # per-seed uniform jitter on each z level

# S2 hysteresis / catatonia: low tone so exit_threshold is low.
S2_TONE = 0.15                    # exit_threshold = theta*tone = 0.30
S2_Z_RUNUP = 1.0                  # supra-duration-threshold run-up that commits
S2_Z_HOLD_BASE = 0.35             # in (exit_threshold 0.30, duration_thr 0.40)
S2_HOLD_TICKS = 8                 # probe window
S2_JITTER = 0.015                 # keep Z_hold safely inside (0.30, 0.40)
S2_PERSISTENCE_GAP_MIN = 0.9      # committed_frozen_frac - rest_frozen_frac must exceed

# S3 gaba-tone exit control: one fixed post-commit decaying drive across 3 tones.
S3_TONES = [0.5, 1.0, 1.5]        # exit_thresholds = 1.0, 2.0, 3.0
S3_Z0_BASE = 3.2                  # commits on tick 0 for every tone (3.2 > theta)
S3_DECAY_BASE = 0.2               # linear per-tick decrement
S3_MAX_TICKS = 40                 # cap; slowest arm (tone 0.5) exits well before
S3_JITTER_Z0 = 0.05
S3_JITTER_DECAY = 0.01

# P0 positive control: a drive at/above exit_threshold=theta*tone (=2.0 at tone
# 1.0) must commit AND stay frozen (so removing it produces a clean release). A
# z below exit_threshold would commit then immediately self-exit (1-tick freeze),
# so use z=2.5 (> exit_threshold) to demonstrate commit->hold->release-on-drop.
P0_CONTROL_TONE = 1.0
P0_CONTROL_Z = 2.5
P0_CONTROL_TICKS = 6
P0_MIN_COMMITS = 1
P0_MIN_RELEASES = 1

# Progress-instrumentation denominator: ep lines printed per seed (one per S1
# level + one per S3 tone). Must equal episodes_per_run in the queue entry.
EPISODES_PER_RUN = len(S1_Z_LEVELS) + len(S3_TONES)


# ----------------------------------------------------------------------------
# Real-substrate gate construction
# ----------------------------------------------------------------------------
def _build_gate(gaba_tone: float):
    """Build a REAL V3 agent with the PAG freeze-gate + GABAergic decay enabled,
    and return (gate, effective_gaba_tone, gate_cfg). The gate is instantiated by
    the real REEConfig -> REEAgent wiring, and effective_gaba_tone is read off the
    real agent.gabaergic_decay regulator -- exactly the value the live
    select_action() path passes to gate.tick() (agent.py:7384-7398)."""
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM, world_obs_dim=WORLD_OBS_DIM, action_dim=ACTION_DIM
    )
    cfg.use_pag_freeze_gate = True
    cfg.use_gabaergic_decay = True
    cfg.gaba_tone = float(gaba_tone)
    agent = REEAgent(cfg)
    gate = agent.pag_freeze_gate
    if gate is None:
        raise RuntimeError("use_pag_freeze_gate=True did not instantiate agent.pag_freeze_gate")
    eff_tone = float(agent.gabaergic_decay.gaba_tone)
    gate_cfg = {
        "theta_freeze": float(gate.config.theta_freeze),
        "duration_input_threshold": float(gate.config.duration_input_threshold),
        "min_freeze_duration": int(gate.config.min_freeze_duration),
        "max_freeze_duration": int(gate.config.max_freeze_duration),
        "requested_gaba_tone": float(gaba_tone),
        "effective_gaba_tone": eff_tone,
    }
    return gate, eff_tone, gate_cfg


def _tick(gate, z: float, tone: float) -> Any:
    """One gate tick, mirroring the live select_action call (override_signal=0.0 so
    the MECH-279 mechanism is isolated from SD-037 broadcast override)."""
    return gate.tick(z_harm_a_norm=float(z), gaba_tone=float(tone),
                     simulation_mode=False, override_signal=0.0)


# --- Independent reference reimplementation of the commit rule (for S1) --------
def _reference_commit_tick(z: float, theta: float, dur_thr: float, horizon: int) -> int:
    """Return the 0-indexed tick at which a sustained constant drive z first
    commits under the claim's rule (z*duration > theta, duration accumulates only
    while z > dur_thr), or -1 if it never commits within horizon. Uses the SAME
    float arithmetic as the gate so the comparison is bit-faithful."""
    duration = 0
    for t in range(horizon):
        if z > dur_thr:
            duration += 1
        else:
            duration = 0
        if (z * duration) > theta:
            return t
        # once committed the gate stops accumulating; for a sustained supra drive
        # the first crossing is the commit, which is all S1 measures.
    return -1


# ----------------------------------------------------------------------------
# Signature measurements (per seed)
# ----------------------------------------------------------------------------
def _measure_s1(seed: int) -> Dict[str, Any]:
    """S1 commit-law: observed gate commit tick == independent reference tick for
    every jittered sustained drive; sub-threshold drive never commits."""
    rng = np.random.default_rng((seed << 4) + 1)
    gate, tone, gate_cfg = _build_gate(P0_CONTROL_TONE)  # commit path is tone-independent
    theta = gate_cfg["theta_freeze"]
    dur_thr = gate_cfg["duration_input_threshold"]
    entries: List[Dict[str, Any]] = []
    all_match = True
    commit_ticks: List[int] = []
    for i, z_base in enumerate(S1_Z_LEVELS):
        z = float(z_base + rng.uniform(-S1_JITTER, S1_JITTER))
        gate.reset()
        obs_commit = -1
        for t in range(S1_SUSTAIN_TICKS):
            out = _tick(gate, z, tone)
            if out.freeze_commit:
                obs_commit = t
                break
        ref_commit = _reference_commit_tick(z, theta, dur_thr, S1_SUSTAIN_TICKS)
        match = (obs_commit == ref_commit) and (obs_commit >= 0)
        all_match = all_match and match
        commit_ticks.append(obs_commit)
        entries.append({"z": z, "observed_commit_tick": obs_commit,
                        "reference_commit_tick": ref_commit, "match": bool(match)})
        print(f"  [probe] S1 seed={seed} ep {i+1}/{EPISODES_PER_RUN} z={z:.3f} "
              f"commit@{obs_commit} ref@{ref_commit}", flush=True)

    # Sub-threshold drive must never commit.
    gate.reset()
    sub_committed = False
    for _ in range(S1_SUSTAIN_TICKS):
        out = _tick(gate, S1_SUBTHRESHOLD_Z, tone)
        if out.freeze_commit:
            sub_committed = True
            break

    # Monotone non-increasing commit latency in z (integrator signature).
    monotone = all(commit_ticks[i + 1] <= commit_ticks[i] for i in range(len(commit_ticks) - 1))
    s1 = bool(all_match and (not sub_committed) and monotone)
    return {"S1_commit_law": s1, "s1_all_match": bool(all_match),
            "s1_subthreshold_committed": bool(sub_committed),
            "s1_commit_latency_monotone": bool(monotone),
            "s1_commit_ticks": commit_ticks, "s1_entries": entries,
            "gate_cfg": gate_cfg}


def _measure_s2(seed: int) -> Dict[str, Any]:
    """S2 committed-state hysteresis: a committed gate held at a mild sub-initiation
    drive stays frozen; a fresh gate at the IDENTICAL drive never freezes."""
    rng = np.random.default_rng((seed << 4) + 2)
    gate, tone, gate_cfg = _build_gate(S2_TONE)
    z_hold = float(S2_Z_HOLD_BASE + rng.uniform(-S2_JITTER, S2_JITTER))
    exit_threshold = gate_cfg["theta_freeze"] * tone

    # Committed branch: run-up to commit, then hold at z_hold.
    gate.reset()
    committed = False
    for _ in range(S1_SUSTAIN_TICKS):
        out = _tick(gate, S2_Z_RUNUP, tone)
        if out.freeze_active:
            committed = True
            break
    committed_frozen = 0
    for _ in range(S2_HOLD_TICKS):
        out = _tick(gate, z_hold, tone)
        committed_frozen += int(out.freeze_active)
    committed_frozen_frac = committed_frozen / float(S2_HOLD_TICKS)

    # Rest branch: fresh gate, same z_hold from rest (never accumulates).
    gate.reset()
    rest_frozen = 0
    for _ in range(S2_HOLD_TICKS):
        out = _tick(gate, z_hold, tone)
        rest_frozen += int(out.freeze_active)
    rest_frozen_frac = rest_frozen / float(S2_HOLD_TICKS)

    persistence_gap = committed_frozen_frac - rest_frozen_frac
    s2 = bool(committed and persistence_gap >= S2_PERSISTENCE_GAP_MIN)
    return {"S2_committed_state_hysteresis": s2,
            "s2_z_hold": z_hold, "s2_exit_threshold": exit_threshold,
            "s2_committed_frozen_frac": committed_frozen_frac,
            "s2_rest_frozen_frac": rest_frozen_frac,
            "s2_persistence_gap": persistence_gap,
            "s2_runup_committed": bool(committed)}


def _measure_s3(seed: int) -> Dict[str, Any]:
    """S3 gaba-tone exit control: release tick strictly ordered by gaba_tone under
    one fixed post-commit decaying drive."""
    rng = np.random.default_rng((seed << 4) + 3)
    z0 = float(S3_Z0_BASE + rng.uniform(-S3_JITTER_Z0, S3_JITTER_Z0))
    decay = float(S3_DECAY_BASE + rng.uniform(-S3_JITTER_DECAY, S3_JITTER_DECAY))
    exit_ticks: List[int] = []
    per_tone: List[Dict[str, Any]] = []
    for j, req_tone in enumerate(S3_TONES):
        gate, tone, gate_cfg = _build_gate(req_tone)
        gate.reset()
        # Commit on tick 0 with z0 (> theta), then apply the decaying drive.
        _tick(gate, z0, tone)
        release_tick = -1
        for t in range(1, S3_MAX_TICKS):
            z = max(0.0, z0 - decay * t)
            out = _tick(gate, z, tone)
            if out.freeze_release:
                release_tick = t
                break
        exit_ticks.append(release_tick)
        per_tone.append({"requested_tone": req_tone, "effective_tone": tone,
                         "exit_threshold": gate_cfg["theta_freeze"] * tone,
                         "release_tick": release_tick})
        print(f"  [probe] S3 seed={seed} ep {len(S1_Z_LEVELS)+j+1}/{EPISODES_PER_RUN} "
              f"tone={tone:.2f} release@{release_tick}", flush=True)

    # Higher gaba_tone -> higher exit_threshold -> EARLIER exit. S3_TONES is
    # ascending, so the exit ticks must be STRICTLY DECREASING:
    # exit(0.5) > exit(1.0) > exit(1.5), i.e. exit(1.5) < exit(1.0) < exit(0.5).
    all_released = all(t >= 0 for t in exit_ticks)
    strict = all(exit_ticks[i] > exit_ticks[i + 1]
                 for i in range(len(exit_ticks) - 1))
    s3 = bool(all_released and strict)
    return {"S3_gaba_tone_exit_control": s3, "s3_z0": z0, "s3_decay": decay,
            "s3_exit_ticks_by_ascending_tone": exit_ticks,
            "s3_all_released": bool(all_released), "s3_strict_order": bool(strict),
            "s3_per_tone": per_tone}


def _p0_positive_control() -> Tuple[list, Dict[str, Any]]:
    """Confirm the built gate fires: >=1 commit under a supra-threshold drive and
    >=1 release when the drive is removed. Raises P0NotReady if inert/miswired."""
    gate, tone, gate_cfg = _build_gate(P0_CONTROL_TONE)
    gate.reset()
    commits = 0
    for _ in range(P0_CONTROL_TICKS):
        out = _tick(gate, P0_CONTROL_Z, tone)
        commits += int(out.freeze_commit)
    releases = 0
    for _ in range(P0_CONTROL_TICKS):
        out = _tick(gate, 0.0, tone)  # remove the drive -> must release
        releases += int(out.freeze_release)
    diag = {"p0_commits": commits, "p0_releases": releases,
            "p0_control_z": P0_CONTROL_Z, "p0_control_tone": tone,
            "gate_cfg": gate_cfg}
    preconditions = p0_readiness_gate([
        {"name": "pag_gate_commits_on_positive_control", "measured": float(commits),
         "threshold": float(P0_MIN_COMMITS), "direction": "lower",
         "control": f"z={P0_CONTROL_Z} sustained {P0_CONTROL_TICKS} ticks on the built gate"},
        {"name": "pag_gate_releases_on_drop", "measured": float(releases),
         "threshold": float(P0_MIN_RELEASES), "direction": "lower",
         "control": "drive removed (z=0) after commit -> gate must release"},
    ])
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
        "body_obs_dim": BODY_OBS_DIM, "world_obs_dim": WORLD_OBS_DIM,
        "action_dim": ACTION_DIM, "seeds": SEEDS,
        "expected_theta_freeze": EXPECTED_THETA_FREEZE,
        "expected_duration_input_threshold": EXPECTED_DURATION_INPUT_THRESHOLD,
        "s1_z_levels": S1_Z_LEVELS, "s1_subthreshold_z": S1_SUBTHRESHOLD_Z,
        "s2_tone": S2_TONE, "s2_z_hold_base": S2_Z_HOLD_BASE,
        "s2_z_runup": S2_Z_RUNUP, "s2_hold_ticks": S2_HOLD_TICKS,
        "s3_tones": S3_TONES, "s3_z0_base": S3_Z0_BASE, "s3_decay_base": S3_DECAY_BASE,
    }

    # ---- P0 positive control + abort gate ----
    try:
        preconditions, p0_diag = _p0_positive_control()
    except P0NotReady as e:
        manifest = {
            "run_id": run_id, "experiment_type": EXPERIMENT_TYPE,
            "architecture_epoch": "ree_hybrid_guardrails_v1",
            "claim_ids": CLAIM_IDS, "experiment_purpose": "diagnostic",
            "outcome": "FAIL", "timestamp_utc": timestamp,
            "non_degenerate": False,
            "degeneracy_reason": "P0 positive control inert: " + e.reason,
            "interpretation": {"label": "substrate_not_ready_requeue",
                               "preconditions": e.preconditions},
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
    gate_cfg_recorded = None
    for seed in seeds:
        print(f"Seed {seed} Condition functional_signature", flush=True)
        s1 = _measure_s1(seed)
        s2 = _measure_s2(seed)
        s3 = _measure_s3(seed)
        gate_cfg_recorded = s1["gate_cfg"]
        s1_hits += int(s1["S1_commit_law"])
        s2_hits += int(s2["S2_committed_state_hysteresis"])
        s3_hits += int(s3["S3_gaba_tone_exit_control"])
        row = {"seed": seed}
        row.update({k: v for k, v in s1.items() if k != "gate_cfg"})
        row.update(s2)
        row.update(s3)
        per_seed.append(row)
        seed_pass = s1["S1_commit_law"] and s2["S2_committed_state_hysteresis"] and s3["S3_gaba_tone_exit_control"]
        print(f"verdict: {'PASS' if seed_pass else 'FAIL'}", flush=True)

    n = len(seeds)
    # Exact functional laws: require unanimity across seeds.
    S1 = s1_hits == n
    S2 = s2_hits == n
    S3 = s3_hits == n
    passed = bool(S1 and S2 and S3)
    outcome = "PASS" if passed else "FAIL"

    # ---- non-degeneracy net (the gate must be genuinely exercised, and the
    # integrator/exit boundaries must produce SPREAD -- a memoryless or inert gate
    # would pin these). Fed the within-sweep spreads, NOT cross-seed values. ----
    degen = check_degeneracy({
        "s1_commit_ticks_over_z_sweep": per_seed[0]["s1_commit_ticks"],
        "s3_exit_ticks_over_tone_sweep": per_seed[0]["s3_exit_ticks_by_ascending_tone"],
    })

    criteria = [
        {"name": "S1_commit_law", "load_bearing": True, "passed": S1},
        {"name": "S2_committed_state_hysteresis", "load_bearing": True, "passed": S2},
        {"name": "S3_gaba_tone_exit_control", "load_bearing": True, "passed": S3},
    ]

    summary = {
        "S1_commit_law": S1, "S2_committed_state_hysteresis": S2,
        "S3_gaba_tone_exit_control": S3,
        "s1_seed_hits": s1_hits, "s2_seed_hits": s2_hits, "s3_seed_hits": s3_hits,
        "n_seeds": n,
        "gate_cfg": gate_cfg_recorded,
        "s2_mean_persistence_gap": float(np.mean([r["s2_persistence_gap"] for r in per_seed])),
        "s3_exit_ticks_example": per_seed[0]["s3_exit_ticks_by_ascending_tone"],
        "s1_commit_ticks_example": per_seed[0]["s1_commit_ticks"],
    }

    manifest: Dict[str, Any] = {
        "run_id": run_id, "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS, "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": "supports" if passed else "weakens",
        "evidence_direction_note": (
            "Wall-independent functional-signature confirmation of the BUILT PAG "
            "freeze-gate substrate (MECH-279): the real config->gate->gaba_tone "
            "wiring realises the committed-freeze transfer function (commit-law, "
            "committed-state hysteresis, gaba-tone-modulated exit / catatonia pole) "
            "over seeded input-trajectory distributions. This is substrate/functional "
            "support only; it does NOT claim the fishtank env behaviourally expresses "
            "the freeze (that remains the SD-037/483e consumer-input-threshold "
            "substrate-ceiling, a separate wall)."
        ),
        "timestamp_utc": timestamp, "dry_run": dry_run,
        "p0_readiness": p0_diag,
        "interpretation": {
            "label": "pag_freeze_gate_functional_signature_confirmed" if passed
                     else "pag_freeze_gate_functional_signature_not_observed",
            "preconditions": preconditions,
            "criteria": criteria,
            "criteria_non_degenerate": {c["name"]: bool(degen["non_degenerate"]) for c in criteria},
        },
        "acceptance_criteria": summary,
        "summary": summary,
        "per_seed": per_seed,
        "constants": {
            "SEEDS": SEEDS, "S1_Z_LEVELS": S1_Z_LEVELS,
            "S1_SUBTHRESHOLD_Z": S1_SUBTHRESHOLD_Z, "S2_TONE": S2_TONE,
            "S2_Z_HOLD_BASE": S2_Z_HOLD_BASE, "S2_PERSISTENCE_GAP_MIN": S2_PERSISTENCE_GAP_MIN,
            "S3_TONES": S3_TONES, "S3_Z0_BASE": S3_Z0_BASE, "S3_DECAY_BASE": S3_DECAY_BASE,
        },
    }
    manifest.update(degen)

    out_path = write_flat_manifest(
        manifest, out_dir, dry_run=dry_run, config=full_config, seeds=SEEDS,
        script_path=Path(__file__), started_at=t0,
    )
    print(f"Manifest written: {out_path}", flush=True)

    print(f"Outcome: {outcome}", flush=True)
    print(f"  S1(commit-law)={S1} S2(hysteresis)={S2} S3(gaba-exit)={S3}", flush=True)
    print(f"  s1_commit_ticks={summary['s1_commit_ticks_example']} "
          f"s3_exit_ticks={summary['s3_exit_ticks_example']} "
          f"s2_persistence_gap={summary['s2_mean_persistence_gap']:.3f}", flush=True)
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
