#!/opt/local/bin/python3
"""V3-EXQ-762: MECH-046 wall-INDEPENDENT functional-signature confirming test.

Confirms candidate/provisional claim MECH-046 -- "CeA analogue writes a mode
prior into the SalienceCoordinator from fast salience classification of z_harm_a"
-- which carries ZERO indexed EVIDENCE-class experiment (exp_conf 0.0,
genuine_exp_count 0) despite its substrate being built + IMPLEMENTED in ree-v3.
Surfaced by the GOV-CONFIRM-1 evidence-confirmer detector (2026-07-14) as a
confirmable-but-unconfirmed candidate with built substrate.

  Substrate under test:
    ree_core/amygdala/cea.py -- CeAAnalog.tick(z_harm_a) computes mode_prior,
    a pre-softmax additive log-odds bias, when |LowFreq(z_harm_a)| crosses
    fast_route_threshold; agent.py (line ~5601) injects it into the
    SalienceCoordinator via update_signal("cea_mode_prior", value), registered
    on affinity_weights["cea_mode_prior"] = {"external_task": 1.0}.

  GOV-REUSE-1 (checked): the ONLY prior CeA-mode-prior run, V3-EXQ-473
    (v3_exq_473_sd035_cea_mode_prior), is experiment_purpose=diagnostic
    (excluded from governance confidence -- which is WHY MECH-046 scores
    exp_conf 0.0) and tested PRESENCE + coordinator-write + bound at a SINGLE
    fixed threat point (|z_harm_a|=0.9 -> mode_prior 0.4) plus the |mp|<=cap
    bound. Its substrate_hash is None (pre-2026-07-12 standard, unverifiable
    compatibility). The DECISIVE readout for a confirmation -- mode_prior
    CONTEXT-CONDITIONED (a genuine range that VARIES with mode-relevant context,
    monotone in threat load), EVIDENCE-purpose, scored -- is absent. Not
    recoverable -> run.

WHY WALL-INDEPENDENT (the design contract):
  The V3 program is bottlenecked on the "competence wall" -- the integrated
  agent is not behaviourally competent enough to emit committed behaviour worth
  measuring (behavioral_diversity_isolation:GAP-I; V3-EXQ-752..756 attack it).
  ANY experiment with a committed-behaviour / mode-SWITCH DV is wall-bound. This
  experiment's DV is a FUNCTIONAL-SIGNATURE readout -- the cea_mode_prior scalar
  the CeAAnalog writes into a REAL SalienceCoordinator -- read out action-free,
  no agent policy, no training, no mode-switch behaviour. It passes or fails
  independent of the wall (precedent: V3-EXQ-757 PASSED confirming sibling
  claims MECH-288/287 the same way; V3-EXQ-455/447/448 COORD_ON functional DVs
  PASSED while the behavioural baseline was monostrategy-locked, failure_autopsy
  455a).

METHOD (representation/functional level; NO training, NO phased training):
  Instantiate the REAL CeAAnalog built exactly as agent.py builds it when
  use_cea_analog=True (CeAConfig mapped field-for-field from a REEConfig with
  use_amygdala_analog=True / use_cea_analog=True -- honouring the V3-EXQ-688
  vacuous-null ARMING CAVEAT: the CeA is explicitly ON and its z_harm_a input
  path is explicitly populated, never left to a default). Instantiate the REAL
  SalienceCoordinator and register the cea_mode_prior affinity weight exactly as
  agent.__init__ does. Drive the CeA action-free with controlled STATIONARY
  z_harm_a streams whose affective-valence load (|LowFreq(z_harm_a)|) is set per
  CONTEXT; each tick inject mode_prior into the coordinator and read it back.

CONDITIONS (cells = seed x context; 5 seeds x 4 contexts = 20 cells):
  Contexts are graded mode-relevant (harm/threat) states, targeting an ascending
  |LowFreq(z_harm_a)| = mean(|z_harm_a_i|):
    safe        target 0.20  (BELOW fast_route_threshold 0.5 -> mode_prior ~0)
    threat_low  target 0.65  (just above threshold)
    threat_mid  target 0.90
    threat_high target 1.20  (well above threshold, still below the log-odds cap)
  Expected mode_prior (thr 0.5, cap 0.8, gain 1.0; mode_prior = clip(lf-thr,0,cap)):
    safe ~0.0, threat_low ~0.15, threat_mid ~0.40, threat_high ~0.70.

DVs / PRE-REGISTERED PASS (thresholds are constants below, not post-hoc):
  C046_present : mean threat_high mode_prior >= PRESENT_MIN (a non-trivial bias
      is emitted -- the write is present).
  C046_varies (LOAD-BEARING): cross-context RANGE of per-context mean mode_prior
      >= RANGE_MIN -- the context-conditioned log-odds bias range (the heart of
      "writes a mode prior ... FROM fast salience classification of z_harm_a").
  C046_monotone: per-context mean mode_prior strictly increases across the
      ordered contexts (each step >= MONO_STEP_MIN) -- the bias tracks threat load.
  C046_rest_silent (specificity): mean safe-context mode_prior <= REST_MAX -- CeA
      does not bias the coordinator absent an above-threshold affective signal.
  C046_coordinator_write (LOAD-BEARING): the value read back from the REAL
      SalienceCoordinator (_input_signals["cea_mode_prior"]) equals the injected
      CeAOutput.mode_prior (max abs diff <= COORD_WRITE_TOL) for every cell, AND
      cea_mode_prior is a registered affinity source on the coordinator frame --
      the literal claim that the prior is WRITTEN INTO the SalienceCoordinator.
  C046_bounded: max |mode_prior| across all cells <= cap + BOUND_EPS -- CeA never
      over-rules cortex via the fast route.
  PASS = present AND varies AND monotone AND rest_silent AND coordinator_write
         AND bounded.

NON-VACUITY READINESS GATE (V3-EXQ-688 vacuous-null lesson): before scoring, a
  positive control measures -- with the SAME statistic the LOAD-BEARING
  C046_varies routes on (a cross-context RANGE, NOT a magnitude) -- that the
  constructed context sweep injects a NON-DEGENERATE, threshold-spanning
  fast-route INPUT:
    input_lowfreq_range_posctrl        = max-min of per-context |LowFreq(z_harm_a)|
        (range readiness for the range-gated criterion) >= INPUT_LOWFREQ_RANGE_FLOOR.
    input_max_lowfreq_supra_threshold  = max per-context |LowFreq(z_harm_a)|
        >= INPUT_MAX_LOWFREQ_FLOOR (> fast_route_threshold, so >=1 arm fires).
  If either is below floor the run self-routes interpretation.label
  "substrate_not_ready_requeue" -> outcome FAIL, evidence_direction "unknown",
  non_degenerate=False (non_contributory; NEVER a false weakens). A
  met-precondition criterion FAIL, by contrast, is a genuine WEAKENS.

Design doc: REE_assembly/docs/architecture/sd_035_amygdala_analog.md#mech-046
Claims:     MECH-046 (docs/claims/claims.yaml)
"""

from __future__ import annotations

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
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._metrics import p0_readiness_gate, P0NotReady  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from ree_core.amygdala import CeAAnalog, CeAConfig  # noqa: E402
from ree_core.cingulate.salience_coordinator import (  # noqa: E402
    SalienceCoordinator,
    SalienceCoordinatorConfig,
)
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_762_mech046_cea_mode_prior_context_conditioning"
QUEUE_ID = "V3-EXQ-762"
CLAIM_IDS = ["MECH-046"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES = None

# ------------------------------------------------------------------ #
# Design constants (pre-registered; NOT derived from run statistics) #
# ------------------------------------------------------------------ #
SEEDS = [0, 1, 2, 3, 4]
# Ordered ascending in threat load -> used directly for the monotonicity check.
CONDITIONS = ["safe", "threat_low", "threat_mid", "threat_high"]
CONTEXT_TARGETS = {
    "safe": 0.20,        # below fast_route_threshold (0.5) -> mode_prior ~0
    "threat_low": 0.65,  # just above threshold
    "threat_mid": 0.90,
    "threat_high": 1.20,  # well above threshold, still below the log-odds cap
}

TICKS_PER_RUN = 120       # ticks per cell == queue episodes_per_run denominator
PROGRESS_EVERY = 40
READ_WARMUP = 20          # skip the first ticks; read steady-state mode_prior after
Z_HARM_A_DIM = 8          # dimensionality of the driven affective stream
TICK_NOISE = 0.02         # per-tick per-element jitter (<< min target 0.20)

# --- MECH-046 acceptance thresholds ---
PRESENT_MIN = 0.30        # C046_present: mean threat_high mode_prior
RANGE_MIN = 0.30          # C046_varies (LOAD-BEARING): cross-context mode_prior range
MONO_STEP_MIN = 0.05      # C046_monotone: min per-step increase across ordered contexts
REST_MAX = 0.02           # C046_rest_silent: mean safe-context mode_prior
COORD_WRITE_TOL = 1e-9    # C046_coordinator_write: max |readback - injected|
BOUND_EPS = 1e-6          # C046_bounded slack over the log-odds cap

# --- non-vacuity readiness floors (measured on a positive control) ---
INPUT_LOWFREQ_RANGE_FLOOR = 0.50   # range statistic (matches range-gated C046_varies)
INPUT_MAX_LOWFREQ_FLOOR = 0.60     # > fast_route_threshold (0.5) so >=1 arm fires
READINESS_SEED = 91

EPS = 1e-9


# ------------------------------------------------------------------ #
# Substrate construction (identical to agent.py use_cea_analog=True) #
# ------------------------------------------------------------------ #
def _build_cea() -> CeAAnalog:
    """Build the REAL CeAAnalog from the ON-path config -- exactly the CeAConfig
    agent.__init__ maps from REEConfig when use_amygdala_analog + use_cea_analog
    are True (agent.py lines ~1724-1735). Honours the V3-EXQ-688 arming caveat:
    the module is explicitly ON, not defaulted."""
    cfg = REEConfig()
    cfg.use_amygdala_analog = True
    cfg.use_cea_analog = True
    cea_cfg = CeAConfig(
        fast_route_threshold=cfg.cea_fast_route_threshold,
        fast_route_input_is_lowfreq=cfg.cea_fast_route_input_is_lowfreq,
        mode_prior_log_odds_max=cfg.cea_mode_prior_log_odds_max,
        mode_prior_gain=cfg.cea_mode_prior_gain,
        pre_softmax_additive=cfg.cea_pre_softmax_additive,
        fast_prime_amplitude=cfg.cea_fast_prime_amplitude,
        fast_prime_decay_tau_steps=cfg.cea_fast_prime_decay_tau_steps,
        fast_prime_override_window_steps=cfg.cea_fast_prime_override_window_steps,
        cortical_confirmation_weight=cfg.cea_cortical_confirmation_weight,
    )
    return CeAAnalog(cea_cfg)


def _build_coordinator() -> SalienceCoordinator:
    """Build the REAL SalienceCoordinator and register the cea_mode_prior affinity
    weight exactly as agent.__init__ does (agent.py lines ~1743-1747), so the
    injected prior enters the coordinator's mode-affinity FRAME."""
    salience = SalienceCoordinator(SalienceCoordinatorConfig())
    salience.config.affinity_weights["cea_mode_prior"] = {"external_task": 1.0}
    salience.config.salience_weights["cea_fast_prime"] = 0.5
    return salience


# ------------------------------------------------------------------ #
# Controlled affective-stream construction                           #
# ------------------------------------------------------------------ #
def _z_harm_a_series(seed: int, target: float, ticks: int) -> np.ndarray:
    """Return a [ticks, Z_HARM_A_DIM] stationary z_harm_a stream whose per-tick
    low-frequency projection mean(|z_harm_a_i|) ~ target. Fixed random sign
    pattern per seed; small per-tick jitter (target >> jitter keeps signs stable,
    so mean(|.|) ~ target)."""
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=Z_HARM_A_DIM)
    base = target * signs
    noise = TICK_NOISE * rng.standard_normal((ticks, Z_HARM_A_DIM))
    return base[None, :] + noise


def _context_lowfreq(seed: int, target: float, ticks: int) -> float:
    """The fast-route INPUT statistic the CeA gate consumes: mean over ticks of
    mean(|z_harm_a_i|). Used by the readiness positive control."""
    series = _z_harm_a_series(seed, target, ticks)
    return float(np.mean(np.abs(series)))


# ------------------------------------------------------------------ #
# Per-cell runner                                                    #
# ------------------------------------------------------------------ #
def _run_context_cell(
    seed: int, context: str, ticks: int, warmup: int, progress_every: int
) -> Dict[str, Any]:
    """Drive the REAL CeAAnalog action-free over a stationary z_harm_a stream at
    this context's affective-load target; inject mode_prior into the REAL
    SalienceCoordinator each tick and read it back."""
    target = CONTEXT_TARGETS[context]
    series = _z_harm_a_series(seed, target, ticks)
    cea = _build_cea()
    cea.reset()
    salience = _build_coordinator()

    mode_priors: List[float] = []       # post-warmup steady-state readings
    low_freqs: List[float] = []
    readback_diffs: List[float] = []
    abs_mode_priors: List[float] = []
    n_fires = 0
    for t in range(ticks):
        z_harm_a = torch.tensor(series[t], dtype=torch.float32)
        out = cea.tick(z_harm_a=z_harm_a)
        mp = float(out.mode_prior)
        # Inject into the coordinator exactly as agent.py does, then read back the
        # value that landed in the SalienceCoordinator frame.
        salience.update_signal("cea_mode_prior", mp)
        readback = float(salience._input_signals["cea_mode_prior"])
        if out.urgency_fire:
            n_fires += 1
        abs_mode_priors.append(abs(mp))
        if t >= warmup:
            mode_priors.append(mp)
            low_freqs.append(float(out.low_freq_magnitude))
            readback_diffs.append(abs(readback - mp))
        if (t + 1) % progress_every == 0:
            print(f"  [train] ctx seed={seed} cond={context} ep {t + 1}/{ticks} "
                  f"mode_prior={mp:.3f} lf={out.low_freq_magnitude:.3f} "
                  f"fires={n_fires}", flush=True)

    in_frame = "cea_mode_prior" in salience.config.affinity_weights
    return {
        "condition": context,
        "seed": seed,
        "target_lowfreq": round(target, 4),
        "mean_mode_prior": round(float(np.mean(mode_priors)), 6) if mode_priors else 0.0,
        "peak_mode_prior": round(float(np.max(mode_priors)), 6) if mode_priors else 0.0,
        "mean_low_freq_magnitude": round(float(np.mean(low_freqs)), 6) if low_freqs else 0.0,
        "n_fires": int(n_fires),
        "max_readback_diff": float(np.max(readback_diffs)) if readback_diffs else 0.0,
        "max_abs_mode_prior": round(float(np.max(abs_mode_priors)), 6) if abs_mode_priors else 0.0,
        "cea_mode_prior_in_affinity_frame": bool(in_frame),
    }


# ------------------------------------------------------------------ #
# Readiness (non-vacuity) positive control                           #
# ------------------------------------------------------------------ #
def _readiness_controls(ticks: int) -> List[Dict[str, Any]]:
    """Measure -- on a positive-control context sweep -- that the constructed
    fast-route INPUT is non-degenerate and threshold-spanning, using the SAME
    statistic the LOAD-BEARING C046_varies routes on (a cross-context RANGE).
    Raises P0NotReady (self-route substrate_not_ready_requeue) if below floor."""
    lows = [_context_lowfreq(READINESS_SEED, CONTEXT_TARGETS[c], ticks) for c in CONDITIONS]
    lowfreq_range = float(max(lows) - min(lows))
    max_lowfreq = float(max(lows))
    return p0_readiness_gate([
        {"name": "input_lowfreq_range_posctrl", "measured": lowfreq_range,
         "threshold": INPUT_LOWFREQ_RANGE_FLOOR, "direction": "lower"},
        {"name": "input_max_lowfreq_supra_threshold_posctrl", "measured": max_lowfreq,
         "threshold": INPUT_MAX_LOWFREQ_FLOOR, "direction": "lower"},
    ])


# ------------------------------------------------------------------ #
# Aggregation + criteria                                             #
# ------------------------------------------------------------------ #
def _mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def _evaluate(rows: List[Dict[str, Any]], cap: float) -> Dict[str, Any]:
    # Per-context mean mode_prior (mean over seeds), in the ordered-context order.
    by_ctx = {c: [r["mean_mode_prior"] for r in rows if r["condition"] == c] for c in CONDITIONS}
    ctx_means = {c: _mean(by_ctx[c]) for c in CONDITIONS}
    ordered_means = [ctx_means[c] for c in CONDITIONS]

    mp_safe = ctx_means["safe"]
    mp_high = ctx_means["threat_high"]
    mp_range = float(max(ordered_means) - min(ordered_means))

    # Monotone: strictly increasing with margin across each ordered step.
    steps = [ordered_means[i + 1] - ordered_means[i] for i in range(len(ordered_means) - 1)]
    monotone = all(s >= MONO_STEP_MIN for s in steps)

    # Coordinator-write fidelity + frame membership across all cells.
    max_readback_diff = max((r["max_readback_diff"] for r in rows), default=0.0)
    all_in_frame = all(r["cea_mode_prior_in_affinity_frame"] for r in rows) if rows else False

    # Bound: no cell's |mode_prior| exceeds the log-odds cap.
    max_abs_mp = max((r["max_abs_mode_prior"] for r in rows), default=0.0)

    c046_present = mp_high >= PRESENT_MIN
    c046_varies = mp_range >= RANGE_MIN
    c046_monotone = bool(monotone)
    c046_rest_silent = mp_safe <= REST_MAX
    c046_coordinator_write = (max_readback_diff <= COORD_WRITE_TOL) and all_in_frame
    c046_bounded = max_abs_mp <= (cap + BOUND_EPS)

    supports = (c046_present and c046_varies and c046_monotone
                and c046_rest_silent and c046_coordinator_write and c046_bounded)

    return {
        "context_mean_mode_prior": {c: round(ctx_means[c], 6) for c in CONDITIONS},
        "mode_prior_context_range": round(mp_range, 6),
        "per_step_increases": [round(s, 6) for s in steps],
        "mean_safe_mode_prior": round(mp_safe, 6),
        "mean_threat_high_mode_prior": round(mp_high, 6),
        "max_readback_diff": max_readback_diff,
        "all_cells_in_affinity_frame": bool(all_in_frame),
        "max_abs_mode_prior": round(max_abs_mp, 6),
        "log_odds_cap": round(float(cap), 6),
        "C046_present": bool(c046_present),
        "C046_varies": bool(c046_varies),
        "C046_monotone": bool(c046_monotone),
        "C046_rest_silent": bool(c046_rest_silent),
        "C046_coordinator_write": bool(c046_coordinator_write),
        "C046_bounded": bool(c046_bounded),
        "MECH-046_supports": bool(supports),
    }


def _cell_verdict(row: Dict[str, Any], cap: float) -> bool:
    """Per-cell LOCAL verdict (progress display only; the scientific outcome is the
    across-seed/context aggregate in _evaluate). A cell is locally OK when the
    coordinator write is faithful, in-frame, and bounded, and the mode_prior
    matches the context direction (safe near-silent, threat present)."""
    if row["max_readback_diff"] > COORD_WRITE_TOL or not row["cea_mode_prior_in_affinity_frame"]:
        return False
    if row["max_abs_mode_prior"] > cap + BOUND_EPS:
        return False
    if row["condition"] == "safe":
        return row["mean_mode_prior"] <= REST_MAX
    return row["mean_mode_prior"] > REST_MAX


# ------------------------------------------------------------------ #
# Orchestration                                                      #
# ------------------------------------------------------------------ #
def _cell_config_slice(seed: int, context: str, ticks: int, warmup: int) -> Dict[str, Any]:
    return {
        "context": context, "seed": seed, "ticks": ticks, "read_warmup": warmup,
        "target_lowfreq": CONTEXT_TARGETS[context], "z_harm_a_dim": Z_HARM_A_DIM,
        "tick_noise": TICK_NOISE, "cea_config": "REEConfig() ON-path canonical defaults",
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        seeds = [0]
        ticks, warmup, progress_every = 40, 10, 20
    else:
        seeds = SEEDS
        ticks, warmup, progress_every = TICKS_PER_RUN, READ_WARMUP, PROGRESS_EVERY

    # --- non-vacuity readiness gate (self-routes on a degenerate positive control) ---
    preconditions = _readiness_controls(ticks)

    # Report the actual log-odds cap under test (for the bound criterion).
    cap = float(_build_cea().config.mode_prior_log_odds_max)

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for context in CONDITIONS:
            print(f"Seed {seed} Condition {context}", flush=True)
            # Cells are pure deterministic functions of (seed, context) -- fresh
            # CeAAnalog + SalienceCoordinator per cell, no shared mutable state, no
            # global RNG use (numpy default_rng is local) -- so they emit
            # reuse-ELIGIBLE by default (mint-as-you-go). There is no trained
            # OFF/baseline arm to reuse here (pure arithmetic substrate, no training).
            with arm_cell(
                seed,
                config_slice=_cell_config_slice(seed, context, ticks, warmup),
                script_path=Path(__file__),
            ) as cell:
                row = _run_context_cell(seed, context, ticks, warmup, progress_every)
                cell.stamp(row)
            print(f"verdict: {'PASS' if _cell_verdict(row, cap) else 'FAIL'}", flush=True)
            rows.append(row)

    criteria = _evaluate(rows, cap)
    supports = criteria["MECH-046_supports"]
    outcome = "PASS" if supports else "FAIL"
    evidence_direction = "supports" if supports else "weakens"
    return {
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": {"MECH-046": evidence_direction},
        "criteria": criteria,
        "arm_results": rows,
        "preconditions": preconditions,
        "log_odds_cap": cap,
        "substrate_ready": True,
    }


def _write_manifest(result: Dict[str, Any], started_at: float) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    c = result["criteria"]
    full_config = {
        "seeds": SEEDS,
        "conditions": CONDITIONS,
        "context_targets": CONTEXT_TARGETS,
        "ticks_per_run": TICKS_PER_RUN,
        "read_warmup": READ_WARMUP,
        "z_harm_a_dim": Z_HARM_A_DIM,
        "tick_noise": TICK_NOISE,
        "cea_config": "REEConfig() ON-path canonical defaults (use_amygdala_analog+use_cea_analog=True)",
        "coordinator_config": "SalienceCoordinatorConfig() canonical defaults + cea_mode_prior affinity weight",
        "log_odds_cap": result["log_odds_cap"],
        "thresholds": {
            "PRESENT_MIN": PRESENT_MIN,
            "RANGE_MIN": RANGE_MIN,
            "MONO_STEP_MIN": MONO_STEP_MIN,
            "REST_MAX": REST_MAX,
            "COORD_WRITE_TOL": COORD_WRITE_TOL,
            "BOUND_EPS": BOUND_EPS,
            "INPUT_LOWFREQ_RANGE_FLOOR": INPUT_LOWFREQ_RANGE_FLOOR,
            "INPUT_MAX_LOWFREQ_FLOOR": INPUT_MAX_LOWFREQ_FLOOR,
        },
    }
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "evidence_class": "exp:simulation",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "non_degenerate": result.get("non_degenerate", True),
        "degeneracy_reason": result.get("degeneracy_reason", ""),
        "timestamp_utc": timestamp,
        "seeds": SEEDS,
        "conditions": CONDITIONS,
        "thresholds": full_config["thresholds"],
        "criteria": result["criteria"],
        "arm_results": result["arm_results"],
        "interpretation": {
            "label": "functional_signature_measured",
            "preconditions": result.get("preconditions", []),
            "criteria_non_degenerate": {
                "C046_present": True, "C046_varies": True, "C046_monotone": True,
                "C046_rest_silent": True, "C046_coordinator_write": True,
                "C046_bounded": True,
            },
            "criteria": [
                {"name": "C046_varies", "load_bearing": True,
                 "passed": bool(c.get("C046_varies", False))},
                {"name": "C046_coordinator_write", "load_bearing": True,
                 "passed": bool(c.get("C046_coordinator_write", False))},
            ],
        },
        "deliverable_note": (
            "Wall-INDEPENDENT functional-signature confirmation of MECH-046 (CeA "
            "analogue writes a mode prior into the SalienceCoordinator), the first "
            "indexed EVIDENCE-class experiment for a claim that scored exp_conf 0.0 "
            "(the only prior CeA-mode-prior run, V3-EXQ-473, is diagnostic-purpose, "
            "non-scoring, and tested a single fixed threat point, not context "
            "variation). The REAL CeAAnalog (built on the use_cea_analog=True ON-path "
            "config) is driven action-free across graded harm/threat contexts; the "
            "cea_mode_prior scalar injected into a REAL SalienceCoordinator is read "
            "back as the DV. PASS requires the prior to be PRESENT under threat, VARY "
            "with context (load-bearing cross-context range) MONOTONICALLY in threat "
            "load, be near-SILENT in the safe context (specificity), be WRITTEN "
            "faithfully into the coordinator frame (load-bearing readback + affinity "
            "membership), and stay BOUNDED by the log-odds cap. A met-precondition "
            "criterion FAIL is a genuine WEAKENS; a below-floor positive control "
            "self-routes substrate_not_ready_requeue (non_contributory)."
        ),
        "wall_independence_note": (
            "DVs are functional-signature readouts of the representation-level "
            "substrate (a scalar written into the SalienceCoordinator), action-free "
            "and training-free with NO mode-switch behaviour, so the outcome is "
            "independent of the V3 competence wall "
            "(behavioral_diversity_isolation:GAP-I; V3-EXQ-752..756). Precedent: "
            "V3-EXQ-757 confirmed MECH-288/287 the same way; V3-EXQ-455/447/448 "
            "COORD_ON functional DVs PASSED while the behavioural baseline was "
            "monostrategy-locked (failure_autopsy 455a)."
        ),
        "gov_reuse_1_note": (
            "Checked V3-EXQ-473 (v3_exq_473_sd035_cea_mode_prior): diagnostic-purpose "
            "(non-scoring), substrate_hash None (pre-standard, unverifiable), tested "
            "presence+coordinator-write+bound at a SINGLE threat point (|z_harm_a|=0.9 "
            "-> mode_prior 0.4). Decisive readout (context-conditioned RANGE + "
            "monotonicity, evidence-purpose) absent -> not recoverable, run."
        ),
        "ethics_preflight": {
            "involves_negative_valence": False,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
        },
    }
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_path = write_flat_manifest(
        manifest, out_dir, config=full_config, seeds=SEEDS,
        script_path=Path(__file__), started_at=started_at,
    )
    print(f"Wrote manifest: {out_path}", flush=True)
    return out_path


def _write_requeue_manifest(preconditions: List[Dict[str, Any]], started_at: float) -> Path:
    """Non-vacuity self-route: the positive-control context sweep did not inject a
    non-degenerate / threshold-spanning fast-route input, so the CeA was never
    exercised on a signal that could produce a varying mode_prior. non_contributory,
    NEVER a false weakens."""
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    unmet = [p["name"] for p in preconditions if not p.get("met", False)]
    full_config = {"seeds": SEEDS, "conditions": CONDITIONS, "ticks_per_run": TICKS_PER_RUN}
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "evidence_class": "exp:simulation",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": "FAIL",
        "evidence_direction": "unknown",
        "evidence_direction_per_claim": {"MECH-046": "unknown"},
        "non_degenerate": False,
        "degeneracy_reason": (
            "positive-control fast-route input below floor (non-vacuity gate): "
            + ", ".join(unmet)),
        "timestamp_utc": timestamp,
        "seeds": SEEDS,
        "conditions": CONDITIONS,
        "interpretation": {
            "label": "substrate_not_ready_requeue",
            "preconditions": preconditions,
        },
        "deliverable_note": (
            "Non-vacuity self-route: the constructed positive-control context sweep did "
            "not inject a supra-threshold, threshold-spanning fast-route input, so the "
            "CeA mode_prior was never exercised on a signal that could vary. Routed "
            "non_contributory (substrate_not_ready_requeue), NOT a weakens -- re-queue "
            "with an adequate context envelope."
        ),
        "ethics_preflight": {
            "involves_negative_valence": False,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
        },
    }
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_path = write_flat_manifest(
        manifest, out_dir, config=full_config, seeds=SEEDS,
        script_path=Path(__file__), started_at=started_at,
    )
    print(f"Wrote requeue manifest: {out_path}", flush=True)
    return out_path


def main(dry_run: bool = False) -> Tuple[str, Any]:
    started_at = time.perf_counter()
    try:
        result = run_experiment(dry_run=dry_run)
    except P0NotReady as e:
        if dry_run:
            print(f"DRY_RUN: readiness self-route ({e.reason})", flush=True)
            return "FAIL", None
        out_path = _write_requeue_manifest(e.preconditions, started_at)
        print(f"OUTCOME: FAIL (substrate_not_ready_requeue: {e.reason})", flush=True)
        return "FAIL", out_path

    if dry_run:
        print(f"DRY_RUN complete: {len(result['arm_results'])} cells, pipeline OK", flush=True)
        return "PASS", None

    out_path = _write_manifest(result, started_at)
    c = result["criteria"]
    print("=== V3-EXQ-762 MECH-046 CeA mode_prior context-conditioning result (n=5) ===", flush=True)
    print(f"  context_mean_mode_prior={c['context_mean_mode_prior']}", flush=True)
    print(f"  MECH-046 supports={c['MECH-046_supports']} "
          f"(present={c['C046_present']} high={c['mean_threat_high_mode_prior']:.3f}>={PRESENT_MIN}; "
          f"varies={c['C046_varies']} range={c['mode_prior_context_range']:.3f}>={RANGE_MIN}; "
          f"monotone={c['C046_monotone']} steps={c['per_step_increases']}; "
          f"rest_silent={c['C046_rest_silent']} safe={c['mean_safe_mode_prior']:.4f}<={REST_MAX}; "
          f"coord_write={c['C046_coordinator_write']} max_diff={c['max_readback_diff']:.2e}; "
          f"bounded={c['C046_bounded']} max_abs={c['max_abs_mode_prior']:.3f}<={c['log_odds_cap']})",
          flush=True)
    print(f"  OUTCOME: {result['outcome']} (direction={result['evidence_direction']})", flush=True)

    _outcome_raw = str(result["outcome"]).upper()
    return (_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL"), out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    _outcome, _manifest_path = main(dry_run=args.dry_run)
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_manifest_path,
        dry_run=args.dry_run,
    )
