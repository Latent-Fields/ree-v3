"""V3-EXQ-779: MECH-063 sub-claim (ii) tonic-vs-phasic behavioural DISSOCIATION.

Claim:    MECH-063 (control plane retains orthogonal tonic/phasic axes rather than
          collapsing into one scalar) -- SUB-CLAIM (ii): each control axis carries
          BOTH a slow TONIC baseline AND a fast PHASIC event-burst as independent,
          independently-toggleable degrees of freedom on a comparable readout.
Purpose:  evidence (tests MECH-063 sub-claim ii directly).

WHY NOW: sub-claim (i) (orthogonal axes) was covered by V3-EXQ-777. Sub-claim (ii)
(tonic/phasic split) became behaviourally testable only after SD-069
(phasic_surprise_burst) landed the PHASIC complement to MECH-313 noise_floor (TONIC)
on the SAME E3 softmax-temperature channel, and after the 2026-07-17 SHARP-SURPRISE
SOURCE amend (REEConfig.phasic_burst_signal_source="instantaneous_pe") routed the
phasic event detector to e3.last_instantaneous_pe -- the RAW per-tick PE-MSE before
the running-variance EMA smoothing -- so the lever fires on REAL surprise events with
NO synthetic poke. (The default smoothed e3._running_variance decays monotonically for
an untrained forward model and fires 0 natural events; verified in the SD-069 landing.)

DESIGN: 2x2 factorial telemetry probe, NO gradient training, NO synthetic signals.
  Factor TONIC  (T) = use_noise_floor  in {OFF, ON}  (MECH-313 SUSTAINED every-tick
                      temperature lift; NF_ALPHA / NF_MIN_T)
  Factor PHASIC (P) = use_phasic_burst in {OFF, ON}  (SD-069 EVENT-LOCKED transient
                      temperature delta; PHASIC-ON sets signal_source="instantaneous_pe")
  4 arms x SEEDS seeds. Shared across ALL arms: use_control_vector_logging=True
  (read-only telemetry, bit-identical), hippocampal.use_action_class_scaffold_candidates
  =True, and a VOLATILITY-ENABLED env (CausalGridWorldV2 background_drift_enabled=True,
  n_drift_sources=3, drift_policy=random_walk) so genuine per-tick PE spikes exist for
  the phasic detector to fire on.

READOUT (per fresh E3 selection; candidates captured via a read-only wrapper on
agent.generate_trajectories aligned with agent.e3.last_precommit_probs):
  E = normalised Shannon entropy of last_precommit_probs (nats / ln K); softmax spread.
Each tick is classified by the phasic burst envelope (agent._last_control_vector
["phasic_burst"]["burst_level"]):
  EVENT-WINDOW tick : burst_level  > EVENT_LEVEL_FLOOR (the phasic delta is behaviourally
                      active this tick -- default temp_delta NEGATIVE = sharpening).
  QUIESCENT   tick : burst_level <= EVENT_LEVEL_FLOOR (no active phasic delta).
Note: burst_level != 0 stays true for many ticks after an event (geometric decay never
reaches 0 exactly), so a strict >0 test would mark ~every tick as event. EVENT_LEVEL_FLOOR
scopes the window to the ticks where temp_delta = temp_delta*burst_level is behaviourally
meaningful. PHASIC-OFF arms have NO event-window ticks (phasic_burst is None -> burst_level
== 0 every tick), so their transient is 0 by construction.

Per-arm readouts:
  S = E_quiescent_mean            -- the SUSTAINED baseline entropy (tonic lever moves this).
  R = E_event_mean - E_quiescent_mean  -- the EVENT-LOCKED TRANSIENT (phasic lever moves
      this; 0 for PHASIC-OFF arms, negative for PHASIC-ON since phasic sharpens).

AGGREGATION (per seed): the two main effects on each readout (averaged over the other
factor):
  dS_tonic  = mean_T1(S) - mean_T0(S)   (noise_floor ON-OFF on the sustained baseline)
  dS_phasic = mean_P1(S) - mean_P0(S)   (phasic ON-OFF on the sustained baseline -> ~0)
  dR_tonic  = mean_T1(R) - mean_T0(R)   (noise_floor ON-OFF on the transient -> ~0)
  dR_phasic = mean_P1(R) - mean_P0(R)   (phasic ON-OFF on the transient -> non-zero)

DOUBLE DISSOCIATION (pre-registered; supports MECH-063 sub-claim ii):
  C1 (TONIC owns the SUSTAINED baseline): |dS_tonic| >= SUSTAINED_MARGIN
       AND |dS_tonic| >= DOMINANCE_K * |dR_tonic|   (tonic moves sustained, not transient).
  C2 (PHASIC owns the EVENT-LOCKED TRANSIENT): |dR_phasic| >= TRANSIENT_MARGIN
       AND |dR_phasic| >= DOMINANCE_K * |dS_phasic|  (phasic moves transient, not baseline).
  Load-bearing verdict = C1 AND C2 on >= MIN_SEEDS seeds, robust (mean margin exceeds its
  own cross-seed SD). Honest note: the dS_phasic~0 leg is PARTLY structural (on quiescent
  ticks the PHASIC-ON effective temperature equals the PHASIC-OFF one by construction), so
  the genuinely falsifiable, load-bearing content is (a) tonic actually lifts the sustained
  baseline (dS_tonic non-trivial), (b) phasic actually produces an event-locked entropy
  transient (dR_phasic non-trivial), and (c) tonic does NOT fake a transient (dR_tonic~0).

P0 READINESS (self-routes substrate_not_ready_requeue -- NEVER a claim verdict):
  R1 phasic fires  : PHASIC-ON arms have >= MIN_EVENT_TICKS event-window ticks SUMMED
                     ACROSS EPISODES (the regulator n_events counter resets per episode via
                     agent.reset -> phasic_burst.reset, so we accumulate our own tick counts
                     during the rollout, NOT get_state at the end).
  R2 both partitions: PHASIC-ON arms also have >= MIN_QUIESCENT_TICKS quiescent ticks (so R
                     is computable).
  R3 tonic live    : TONIC-ON arms mean noise_floor_temp_lift >= TEMP_LIFT_FLOOR.
  R4 samples       : every cell has >= MIN_SELECTS fresh E3 selections.
  R5 headroom      : T0P0 baseline entropy in (E_SAT_LOW, E_SAT_HIGH) so both a tonic lift
                     (up) and a phasic sharpening (down) have room to move.
Below any precondition -> outcome FAIL, evidence_direction non_contributory,
non_degenerate False, label substrate_not_ready_requeue.

VERDICT (pre-registered constants; not derived post-hoc):
  readiness unmet                    -> FAIL / non_contributory (requeue).
  readiness met AND C1 AND C2 robust -> PASS / supports (tonic and phasic are independent
     tonic-baseline vs phasic-transient degrees of freedom on one readout -- MECH-063 ii).
  readiness met AND (C1 AND C2) unmet -> FAIL / weakens (the two levers do not dissociate
     into a sustained-baseline vs event-transient split on a comparable readout).

MECH-094: trains nothing, writes no memory, no replay -- N/A (waking select only).
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

_THIS = Path(__file__).resolve()
_REE_V3 = _THIS.parent.parent
if str(_REE_V3) not in sys.path:
    sys.path.insert(0, str(_REE_V3))

from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiments._harness import StepHarness  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_779_mech063_tonic_phasic_dissociation"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS: List[str] = ["MECH-063"]

# ---- Pre-registered constants (fixed before the run; not derived post-hoc) ----
SEEDS = [11, 17, 23, 29, 37]
MIN_SEEDS = 4                 # criteria must hold on >= this many of the 5 seeds
N_EPISODES = 3
STEPS_PER_EPISODE = 300
EPISODES_PER_RUN = N_EPISODES  # denominator for [train] progress prints

ENV_SIZE = 8
ENV_HAZARDS = 2
ENV_RESOURCES = 3
# Volatility so genuine per-tick PE spikes exist for the phasic detector.
ENV_DRIFT_SOURCES = 3
ENV_DRIFT_POLICY = "random_walk"

# TONIC axis (MECH-313 noise_floor). Base E3 temperature is 1.0 (StepHarness passes
# temperature=1.0); effective = max(1.0 + alpha, min_T) = 2.0 -> a sustained +1.0 lift.
NF_ALPHA = 1.0
NF_MIN_T = 2.0

# PHASIC axis (SD-069 phasic_surprise_burst). Sharp source; trigger tuned (with the P0
# gate as backstop) for a populated event-window AND quiescent partition.
PHASIC_SOURCE = "instantaneous_pe"
PHASIC_TRIGGER_RATIO = 1.2
PHASIC_EMA_DECAY = 0.1
PHASIC_TEMP_DELTA = -0.5        # NEGATIVE = phasic sharpening (LC-NE phasic gain increase)
PHASIC_DECAY = 0.5
PHASIC_TRIGGER_FLOOR = 1e-6
PHASIC_MIN_T = 0.1
# Burst-level above which the phasic temperature delta is behaviourally active
# (scopes the event window; below this the decaying envelope is treated as quiescent).
EVENT_LEVEL_FLOOR = 0.05

# Readiness thresholds.
MIN_SELECTS = 20              # R4: fresh E3 selections per cell
MIN_EVENT_TICKS = 10         # R1: event-window ticks in PHASIC-ON arms (summed / episodes)
MIN_QUIESCENT_TICKS = 10     # R2: quiescent ticks in PHASIC-ON arms
TEMP_LIFT_FLOOR = 0.5        # R3: noise_floor_temp_lift in TONIC-ON arms
E_SAT_LOW = 0.02             # R5: baseline entropy floor (headroom to sharpen down)
E_SAT_HIGH = 0.98            # R5: baseline entropy ceiling (headroom to lift up)

# Verdict thresholds (normalised entropy is in [0, 1]).
SUSTAINED_MARGIN = 0.05      # C1: min |tonic effect on sustained baseline entropy|
TRANSIENT_MARGIN = 0.02      # C2: min |phasic effect on event-locked transient|
DOMINANCE_K = 2.0            # each axis moves its OWN readout >= K x the cross-readout

ARMS = ["T0P0", "T1P0", "T0P1", "T1P1"]  # (noise_floor, phasic_burst)
_ARM_FLAGS = {
    "T0P0": (False, False),
    "T1P0": (True, False),
    "T0P1": (False, True),
    "T1P1": (True, True),
}


def _mk_env() -> CausalGridWorldV2:
    return CausalGridWorldV2(
        size=ENV_SIZE,
        num_hazards=ENV_HAZARDS,
        num_resources=ENV_RESOURCES,
        background_drift_enabled=True,
        n_drift_sources=ENV_DRIFT_SOURCES,
        drift_policy=ENV_DRIFT_POLICY,
    )


def _build_config(arm: str, env: CausalGridWorldV2) -> REEConfig:
    use_nf, use_pb = _ARM_FLAGS[arm]
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
    )
    # Shared substrate operating settings (identical across all four arms).
    cfg.use_control_vector_logging = True
    cfg.hippocampal.use_action_class_scaffold_candidates = True
    # TONIC axis.
    cfg.use_noise_floor = use_nf
    if use_nf:
        cfg.noise_floor_alpha = NF_ALPHA
        cfg.noise_floor_min_temperature = NF_MIN_T
    # PHASIC axis.
    cfg.use_phasic_burst = use_pb
    if use_pb:
        cfg.phasic_burst_signal_source = PHASIC_SOURCE
        cfg.phasic_burst_trigger_ratio = PHASIC_TRIGGER_RATIO
        cfg.phasic_burst_surprise_ema_decay = PHASIC_EMA_DECAY
        cfg.phasic_burst_temp_delta = PHASIC_TEMP_DELTA
        cfg.phasic_burst_decay = PHASIC_DECAY
        cfg.phasic_burst_trigger_floor = PHASIC_TRIGGER_FLOOR
        cfg.phasic_burst_min_temperature = PHASIC_MIN_T
    return cfg


def _config_slice(arm: str) -> Dict[str, Any]:
    """Fingerprint config slice: env + shared operating settings + this arm's
    control flags. Declares only what the cell's build+collect path reads."""
    use_nf, use_pb = _ARM_FLAGS[arm]
    sl: Dict[str, Any] = {
        "env_size": ENV_SIZE,
        "env_hazards": ENV_HAZARDS,
        "env_resources": ENV_RESOURCES,
        "env_drift_sources": ENV_DRIFT_SOURCES,
        "env_drift_policy": ENV_DRIFT_POLICY,
        "n_episodes": N_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "use_control_vector_logging": True,
        "use_action_class_scaffold_candidates": True,
        "use_noise_floor": use_nf,
        "use_phasic_burst": use_pb,
    }
    if use_nf:
        sl.update(noise_floor_alpha=NF_ALPHA, noise_floor_min_temperature=NF_MIN_T)
    if use_pb:
        sl.update(
            phasic_burst_signal_source=PHASIC_SOURCE,
            phasic_burst_trigger_ratio=PHASIC_TRIGGER_RATIO,
            phasic_burst_surprise_ema_decay=PHASIC_EMA_DECAY,
            phasic_burst_temp_delta=PHASIC_TEMP_DELTA,
            phasic_burst_decay=PHASIC_DECAY,
            phasic_burst_trigger_floor=PHASIC_TRIGGER_FLOOR,
            phasic_burst_min_temperature=PHASIC_MIN_T,
            event_level_floor=EVENT_LEVEL_FLOOR,
        )
    return sl


def _norm_entropy(probs: torch.Tensor) -> Optional[float]:
    """Normalised Shannon entropy (nats / ln K) of a candidate softmax."""
    p = probs.detach().reshape(-1).float()
    k = int(p.numel())
    if k < 2:
        return None
    p = p[p > 0]
    h = float(-(p * p.log()).sum().item())
    return h / math.log(k)


def _run_cell(arm: str, seed: int) -> Dict[str, Any]:
    """One (arm, seed) cell: telemetry rollout, no gradient training."""
    with arm_cell(
        seed,
        config_slice=_config_slice(arm),
        script_path=_THIS,
        config_slice_declared=True,
        include_driver_script_in_hash=False,  # cross-driver reusable mint
    ) as cell:
        env = _mk_env()
        cfg = _build_config(arm, env)
        agent = REEAgent(cfg)

        # Read-only wrapper to capture the candidate list E3 selects over, so a
        # fresh-selection is detected against agent.e3.last_precommit_probs.
        captured: Dict[str, Any] = {"cands": None}
        _orig_gen = agent.generate_trajectories

        def _gen_capture(*a: Any, **k: Any) -> Any:
            cands = _orig_gen(*a, **k)
            captured["cands"] = cands
            return cands

        agent.generate_trajectories = _gen_capture  # type: ignore[assignment]

        harness = StepHarness(agent, env, train_mode=True, seed=seed)

        e_event: List[float] = []       # entropy on event-window ticks
        e_quiescent: List[float] = []   # entropy on quiescent ticks
        templift_vals: List[float] = []
        burst_levels: List[float] = []
        n_selects = 0
        n_env_steps = 0
        prev_probs_id: Optional[int] = None

        for ep in range(N_EPISODES):
            _flat, obs_dict = env.reset()
            agent.reset()
            harness.reset()
            prev_probs_id = None
            for _step in range(STEPS_PER_EPISODE):
                r = harness.step(obs_dict)
                n_env_steps += 1
                probs = getattr(agent.e3, "last_precommit_probs", None)
                pid = id(probs) if probs is not None else None
                fresh = probs is not None and pid != prev_probs_id
                prev_probs_id = pid
                if fresh and int(probs.numel()) >= 2:
                    ent = _norm_entropy(probs)
                    if ent is not None:
                        n_selects += 1
                        cv = agent._last_control_vector or {}
                        pb = cv.get("phasic_burst", {}) or {}
                        gv = cv.get("G_vigor", {}) or {}
                        blevel = float(pb.get("burst_level", 0.0))
                        burst_levels.append(blevel)
                        templift_vals.append(
                            float(gv.get("noise_floor_temp_lift", 0.0))
                        )
                        if blevel > EVENT_LEVEL_FLOOR:
                            e_event.append(ent)
                        else:
                            e_quiescent.append(ent)
                obs_dict = r.next_obs_dict
                if r.done:
                    break
            print(
                f"  [train] {arm} seed={seed} ep {ep + 1}/{EPISODES_PER_RUN} "
                f"env_steps={n_env_steps} e3_selects={n_selects} "
                f"event_ticks={len(e_event)} quiescent_ticks={len(e_quiescent)}",
                flush=True,
            )

        def _mean(xs: List[float]) -> float:
            return float(statistics.fmean(xs)) if xs else 0.0

        s_quiescent = _mean(e_quiescent)     # SUSTAINED baseline
        s_event = _mean(e_event)
        # EVENT-LOCKED TRANSIENT: 0 when there are no event-window ticks
        # (PHASIC-OFF arms, by construction).
        transient = (s_event - s_quiescent) if e_event else 0.0

        row: Dict[str, Any] = {
            "arm": arm,
            "seed": seed,
            "use_noise_floor": _ARM_FLAGS[arm][0],
            "use_phasic_burst": _ARM_FLAGS[arm][1],
            "n_env_steps": n_env_steps,
            "n_e3_selects": n_selects,
            "n_event_ticks": len(e_event),
            "n_quiescent_ticks": len(e_quiescent),
            "S_sustained_entropy": s_quiescent,
            "E_event_entropy": s_event,
            "R_transient": transient,
            "noise_floor_temp_lift_mean": _mean(templift_vals),
            "burst_level_mean": _mean(burst_levels),
            "burst_level_max": float(max(burst_levels)) if burst_levels else 0.0,
            # Per-tick series retained for generous recording / reanalysis.
            "E_event_series": e_event,
            "E_quiescent_series": e_quiescent,
        }
        cell.stamp(row)
    return row


# ----------------------------------------------------------------------
# Aggregation: 2x2 double dissociation of the tonic/phasic control axes.
# ----------------------------------------------------------------------
def _pooled_std(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    return float(statistics.pstdev(vals))


def _seed_effects(rows_by_arm: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Main effects of TONIC and PHASIC on the sustained baseline (S) and the
    event-locked transient (R)."""
    def _S(arm: str) -> float:
        return float(rows_by_arm[arm]["S_sustained_entropy"])

    def _R(arm: str) -> float:
        return float(rows_by_arm[arm]["R_transient"])

    # TONIC = noise_floor ON - OFF, averaged over the PHASIC factor.
    dS_tonic = ((_S("T1P0") + _S("T1P1")) / 2.0) - ((_S("T0P0") + _S("T0P1")) / 2.0)
    dR_tonic = ((_R("T1P0") + _R("T1P1")) / 2.0) - ((_R("T0P0") + _R("T0P1")) / 2.0)
    # PHASIC = phasic ON - OFF, averaged over the TONIC factor.
    dS_phasic = ((_S("T0P1") + _S("T1P1")) / 2.0) - ((_S("T0P0") + _S("T1P0")) / 2.0)
    dR_phasic = ((_R("T0P1") + _R("T1P1")) / 2.0) - ((_R("T0P0") + _R("T1P0")) / 2.0)

    # C1: tonic owns the sustained baseline.
    c1 = (abs(dS_tonic) >= SUSTAINED_MARGIN) and (
        abs(dS_tonic) >= DOMINANCE_K * abs(dR_tonic)
    )
    # C2: phasic owns the event-locked transient.
    c2 = (abs(dR_phasic) >= TRANSIENT_MARGIN) and (
        abs(dR_phasic) >= DOMINANCE_K * abs(dS_phasic)
    )
    return {
        "dS_tonic": dS_tonic,
        "dR_tonic": dR_tonic,
        "dS_phasic": dS_phasic,
        "dR_phasic": dR_phasic,
        "C1_tonic_owns_sustained": bool(c1),
        "C2_phasic_owns_transient": bool(c2),
        "dissociation": bool(c1 and c2),
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    global N_EPISODES, STEPS_PER_EPISODE, EPISODES_PER_RUN
    seeds = SEEDS[:2] if dry_run else SEEDS
    if dry_run:
        N_EPISODES = 2
        STEPS_PER_EPISODE = 120
        EPISODES_PER_RUN = N_EPISODES

    t0 = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for arm in ARMS:
            print(f"Seed {seed} Condition {arm}", flush=True)
            row = _run_cell(arm, seed)
            rows.append(row)
            print(f"verdict: {'PASS' if row['n_e3_selects'] > 0 else 'FAIL'}", flush=True)

    # Per-seed double-dissociation analysis.
    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        by_arm = {r["arm"]: r for r in rows if r["seed"] == seed}
        if len(by_arm) != len(ARMS):
            continue
        eff = _seed_effects(by_arm)
        eff["seed"] = seed
        per_seed.append(eff)

    # ---- Readiness preconditions (self-route requeue if unmet) ----
    p1_rows = [r for r in rows if r["use_phasic_burst"]]   # PHASIC-ON arms
    t1_rows = [r for r in rows if r["use_noise_floor"]]    # TONIC-ON arms
    baseline_rows = [r for r in rows if r["arm"] == "T0P0"]

    r1_phasic_fires = bool(p1_rows) and all(
        r["n_event_ticks"] >= MIN_EVENT_TICKS for r in p1_rows
    )
    r2_both_partitions = bool(p1_rows) and all(
        r["n_quiescent_ticks"] >= MIN_QUIESCENT_TICKS for r in p1_rows
    )
    r3_tonic_live = bool(t1_rows) and all(
        r["noise_floor_temp_lift_mean"] >= TEMP_LIFT_FLOOR for r in t1_rows
    )
    r4_samples = bool(rows) and all(r["n_e3_selects"] >= MIN_SELECTS for r in rows)
    r5_headroom = bool(baseline_rows) and all(
        E_SAT_LOW < r["S_sustained_entropy"] < E_SAT_HIGH for r in baseline_rows
    )
    readiness_met = bool(
        r1_phasic_fires and r2_both_partitions and r3_tonic_live
        and r4_samples and r5_headroom
    )

    # ---- Load-bearing criterion: double dissociation (C1 AND C2) ----
    diss = [a["dissociation"] for a in per_seed]
    seeds_diss = sum(1 for d in diss if d)
    diss_seed_count = seeds_diss >= min(MIN_SEEDS, len(seeds))
    # Robustness: mean tonic-sustained and phasic-transient effects exceed their
    # own cross-seed SD (effect is not noise).
    dS_tonic_all = [a["dS_tonic"] for a in per_seed]
    dR_phasic_all = [a["dR_phasic"] for a in per_seed]
    mean_dS_tonic = statistics.fmean(dS_tonic_all) if dS_tonic_all else 0.0
    mean_dR_phasic = statistics.fmean(dR_phasic_all) if dR_phasic_all else 0.0
    robust = (
        abs(mean_dS_tonic) - _pooled_std(dS_tonic_all) > 0.0
        and abs(mean_dR_phasic) - _pooled_std(dR_phasic_all) > 0.0
    )
    dissociation = bool(diss_seed_count and robust)

    non_degenerate = bool(readiness_met and len(per_seed) >= 1)
    degeneracy_reason = None

    if not readiness_met:
        outcome = "FAIL"
        direction = "non_contributory"
        label = "substrate_not_ready_requeue"
        non_degenerate = False
        degeneracy_reason = "readiness precondition unmet (see interpretation.preconditions)"
    elif dissociation:
        outcome = "PASS"
        direction = "supports"
        label = "tonic_phasic_double_dissociation"
    else:
        outcome = "FAIL"
        direction = "weakens"
        label = "tonic_phasic_no_dissociation"

    interpretation = {
        "label": label,
        "preconditions": [
            {"name": "phasic_fires_real_events",
             "control": "PHASIC-ON arms: event-window ticks summed across episodes",
             "measured": (min(r["n_event_ticks"] for r in p1_rows) if p1_rows else 0),
             "threshold": MIN_EVENT_TICKS, "met": bool(r1_phasic_fires)},
            {"name": "both_partitions_populated",
             "control": "PHASIC-ON arms: quiescent ticks (so the transient is computable)",
             "measured": (min(r["n_quiescent_ticks"] for r in p1_rows) if p1_rows else 0),
             "threshold": MIN_QUIESCENT_TICKS, "met": bool(r2_both_partitions)},
            {"name": "tonic_axis_live",
             "control": "TONIC-ON arms: noise_floor_temp_lift",
             "measured": (statistics.fmean([r["noise_floor_temp_lift_mean"] for r in t1_rows]) if t1_rows else 0.0),
             "threshold": TEMP_LIFT_FLOOR, "met": bool(r3_tonic_live)},
            {"name": "sample_sufficiency",
             "control": "min fresh E3 selections over cells",
             "measured": (min(r["n_e3_selects"] for r in rows) if rows else 0),
             "threshold": MIN_SELECTS, "met": bool(r4_samples)},
            {"name": "baseline_entropy_headroom",
             "control": "T0P0 sustained entropy strictly inside (E_SAT_LOW, E_SAT_HIGH)",
             "measured": (statistics.fmean([r["S_sustained_entropy"] for r in baseline_rows]) if baseline_rows else 0.0),
             "threshold": E_SAT_HIGH, "direction": "upper", "met": bool(r5_headroom)},
        ],
        "criteria": [
            {"name": "double_dissociation_C1_and_C2", "load_bearing": True, "passed": bool(dissociation)},
        ],
        "criteria_non_degenerate": {
            # Non-degenerate iff the load-bearing effects carry real cross-seed
            # signal (not all-pinned): tonic-sustained OR phasic-transient effect
            # has non-zero spread across seeds, on >= MIN_SEEDS seeds.
            "double_dissociation_C1_and_C2": bool(
                len(per_seed) >= min(MIN_SEEDS, len(seeds))
                and (_pooled_std(dS_tonic_all) > 0.0 or _pooled_std(dR_phasic_all) > 0.0)
            ),
        },
        "summary": (
            "MECH-063 sub-claim (ii) tonic-vs-phasic dissociation on the E3 softmax "
            "temperature. The TONIC lever (MECH-313 noise_floor) moves the SUSTAINED "
            "(quiescent-tick) baseline entropy; the PHASIC lever (SD-069 phasic_surprise_"
            "burst, sharp instantaneous_pe source) moves an EVENT-LOCKED entropy TRANSIENT; "
            "cross-legs ~0. PASS = both hold (independent tonic-baseline vs phasic-transient "
            "degrees of freedom on one readout); FAIL/weakens = they do not dissociate; "
            "FAIL/non_contributory = a lever was not exercised (substrate_not_ready_requeue)."
        ),
    }

    ethics_preflight = {
        "involves_negative_valence": False,
        "involves_suffering_like_state": False,
        "involves_self_model": False,
        "involves_inescapability_or_helplessness": False,
        "involves_offline_replay_over_harm": False,
        "involves_social_mind_or_language": False,
        "involves_human_data_or_clinical_context": False,
        "decision": "allow",
    }

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest: Dict[str, Any] = {
        "schema_version": "v1",
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "outcome": outcome,
        "evidence_direction": direction,
        "dry_run": bool(dry_run),
        "non_degenerate": non_degenerate,
        "interpretation": interpretation,
        "ethics_preflight": ethics_preflight,
        "acceptance": {
            "readiness_met": readiness_met,
            "double_dissociation": dissociation,
            "diss_seed_count": seeds_diss,
            "min_seeds": min(MIN_SEEDS, len(seeds)),
            "mean_dS_tonic": mean_dS_tonic,
            "mean_dR_phasic": mean_dR_phasic,
            "sd_dS_tonic": _pooled_std(dS_tonic_all),
            "sd_dR_phasic": _pooled_std(dR_phasic_all),
            "robust": robust,
            "SUSTAINED_MARGIN": SUSTAINED_MARGIN,
            "TRANSIENT_MARGIN": TRANSIENT_MARGIN,
            "DOMINANCE_K": DOMINANCE_K,
        },
        "per_seed": per_seed,
        "arm_results": rows,
        "notes": (
            "MECH-063 sub-claim (ii) tonic/phasic split behavioural dissociation. TONIC "
            "= MECH-313 noise_floor (sustained temperature lift); PHASIC = SD-069 "
            "phasic_surprise_burst with the sharp instantaneous_pe source (raw per-tick "
            "PE-MSE before running-variance smoothing), so the lever fires on REAL surprise "
            "events with NO synthetic poke (the default smoothed source fires 0 natural "
            "events). Complements V3-EXQ-777 (sub-claim i, orthogonal axes). GOV-REUSE-1: "
            "the decisive readout (event-window-partitioned entropy transient under the "
            "sharp phasic source) requires the SD-069 sharp-source substrate landed "
            "2026-07-17 and is absent from all recorded manifests -> run. Re-derive brake: "
            "0 substrate_ceiling/non_contributory autopsies on MECH-063 -> not braked."
        ),
    }
    if non_degenerate is False and degeneracy_reason:
        manifest["degeneracy_reason"] = degeneracy_reason

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t_start = time.perf_counter()
    manifest = run_experiment(dry_run=args.dry_run)

    out_dir = _REE_V3.parent / "REE_assembly" / "evidence" / "experiments"
    full_config = {
        "seeds": SEEDS,
        "n_episodes": N_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "env": {
            "size": ENV_SIZE, "num_hazards": ENV_HAZARDS, "num_resources": ENV_RESOURCES,
            "background_drift_enabled": True, "n_drift_sources": ENV_DRIFT_SOURCES,
            "drift_policy": ENV_DRIFT_POLICY,
        },
        "tonic_noise_floor": {"alpha": NF_ALPHA, "min_temperature": NF_MIN_T},
        "phasic_burst": {
            "signal_source": PHASIC_SOURCE, "trigger_ratio": PHASIC_TRIGGER_RATIO,
            "surprise_ema_decay": PHASIC_EMA_DECAY, "temp_delta": PHASIC_TEMP_DELTA,
            "decay": PHASIC_DECAY, "trigger_floor": PHASIC_TRIGGER_FLOOR,
            "min_temperature": PHASIC_MIN_T, "event_level_floor": EVENT_LEVEL_FLOOR,
        },
        "thresholds": {
            "MIN_SEEDS": MIN_SEEDS, "MIN_SELECTS": MIN_SELECTS,
            "MIN_EVENT_TICKS": MIN_EVENT_TICKS, "MIN_QUIESCENT_TICKS": MIN_QUIESCENT_TICKS,
            "TEMP_LIFT_FLOOR": TEMP_LIFT_FLOOR, "E_SAT_LOW": E_SAT_LOW, "E_SAT_HIGH": E_SAT_HIGH,
            "SUSTAINED_MARGIN": SUSTAINED_MARGIN, "TRANSIENT_MARGIN": TRANSIENT_MARGIN,
            "DOMINANCE_K": DOMINANCE_K,
        },
    }

    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=args.dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=_THIS,
        started_at=t_start,
    )

    # NOTE: use a non-"verdict:" prefix here -- the runner counts "verdict:" lines
    # as completed runs, and the per-cell prints already emit exactly
    # seeds x conditions of them.
    print(f"FINAL_OUTCOME: {manifest['outcome']}", flush=True)
    print(f"  label: {manifest['interpretation']['label']}", flush=True)
    print(f"  readiness_met: {manifest['acceptance']['readiness_met']}", flush=True)
    print(f"  mean_dS_tonic: {manifest['acceptance']['mean_dS_tonic']:.3f} "
          f"(margin {SUSTAINED_MARGIN})", flush=True)
    print(f"  mean_dR_phasic: {manifest['acceptance']['mean_dR_phasic']:.3f} "
          f"(margin {TRANSIENT_MARGIN})", flush=True)
    print(f"Result written to: {out_path}", flush=True)

    emit_outcome(
        outcome=manifest["outcome"] if manifest["outcome"] in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(out_path),
        dry_run=args.dry_run,
    )
