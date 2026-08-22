#!/opt/local/bin/python3
"""
V3-EXQ-944 -- MECH-091: do salient events reset E3's cycle boundary, and does
that eliminate partial-integration artefacts straddling the event?

Claims: MECH-091 (mechanism_hypothesis, candidate, implementation_phase v3)
Chip:   chip-20260821-igw-234-mech091-confirm (IGW-20260821-234)

experiment_purpose: evidence

WHY THIS RUN EXISTS
-------------------
MECH-091 ("Salient events phase-reset the E3 heartbeat clock") has carried
literature confidence 0.82 and ~zero experimental evidence since 2026-03. It was
parked from 2026-05-08 on SD-006 phase 2 (async multi-rate execution). Governance
2026-08-16 (GFLAG-0037 SPLIT, cycle cranky-driscoll-126a36) found that deferral
MIS-SCOPED: phase_reset() is BUILT (ree_core/heartbeat/clock.py), WIRED, and was
measured as the DOMINANT driver of E3 tick cadence in a real 53,063-step rollout
(evidence/planning/diagnostic_arc071_e3_reselection_probe_2026-08-01.md). The
real gap was small and is now CLOSED: of the three salient events MECH-091 names
(task completion, unexpected harm, commitment-boundary crossing) only harm was
wired; ree-v3 origin/main 6293b2395248 (2026-08-17, substrate_queue
MECH091-SALIENT-EVENT-TRIGGER-WIRING, status implemented) wired the other two.

This is the experiment MECH-091's own what_would_answer asks for and that
EXQ-133 was not. EXQ-133 (2026-03-28, 2026-04-21 rerun) ran a HAND-ROLLED
synthetic counter and measured latent divergence (harm_eval_gap) as a PROXY; it
never touched MultiRateClock, and both runs were reclassified non_contributory --
explicitly NOT falsifications -- for the absent substrate. This run measures the
CYCLE-BOUNDARY DV the claim actually names, on the real substrate.

RE-DERIVE BRAKE (MOVE-3): RELEASED, not bypassed. MECH-091 counts 2 braking
autopsies, both targets of failure_autopsy_grandfathered-r5-batch01-mixed-findings
_2026-08-08 (the two EXQ-133 runs), both recommended_epistemic_category
"precondition_unmet", both routing "governance-note-only" with
recommended_substrate_queue_entry.action "none" and pending_retest_after_substrate
TRUE. The named upstream substrate is now built (6293b2395248), and this is a new
EXQ NUMBER carrying a different design (real-substrate cycle-boundary DV vs a
synthetic latent-divergence proxy), which the brake's own "not braked" clause
covers. Both conditions hold; either alone would suffice.

GOV-REUSE-1: not recoverable -> run. The decisive readout is the per-salient-event
E3 integration lag / straddle fraction. `reanalysis_query.py query --claim MECH-091`
returns 0 of 927 flat manifests; the only MECH-091 runs are the two EXQ-133 runs,
which (a) carry a different proxy DV, (b) predate substrate_hash entirely so no
compatibility key exists, and (c) predate the 2026-08-17 wiring.

THE DV, AND WHY IT IS NOT THE EXQ-131 STALENESS ARTIFACT
--------------------------------------------------------
GFLAG-0037 owed ONE cheap puzzle-grade check before this could be queued:
confirm the what_would_answer DV -- "no partial-integration artefacts straddling
a salient event" -- is observable on phase 1 and is not itself absorbed by the
EXQ-131 E3-output-freeze staleness artifact. It was performed twice, from two
directions, and PASSED both times:

  (i) The 6293b2395248 wiring commit's own pre-check: harm_eval() is a direct
      per-tick neural head, not the EMA-based var_harm_eval_on term EXQ-131
      found frozen.
  (ii) This script's own pre-queue probe (2026-08-22): the DV as operationalised
      HERE reads NOTHING from harm_eval or any EMA term. It is a pure clock-
      timing readout -- the sequence of `e3_tick` flags returned by
      `clock.advance()` versus the steps at which `clock.phase_reset()` was
      requested. An output-freeze artifact in a downstream evaluator cannot
      reach it.

Operationalisation. A "partial-integration artefact straddling a salient event"
is an E3 planning cycle whose interior CONTAINS the event: the cycle opened
before the event, so its selection was made without the event's information, yet
it governs behaviour for the post-event steps. Two readouts per cell:

  straddle_frac         fraction of salient events whose NEXT E3 tick is more
                        than 1 step away (i.e. a cycle straddles the event)
  mean_integration_lag  mean env steps from the event to the next E3 tick

PRE-QUEUE PROBE (2026-08-22, untrained agent, 6 ep x 72 steps, seeds 42/7/13;
throwaway, not under evidence/). This is the measurement that answers GFLAG-0037:

  arm            mean_lag            straddle_frac        e3_tick_rate
  ALIGNED        1.00 1.00 1.00      0.00 0.00 0.00       0.296 0.188 0.195
  RATE_MATCHED   3.86 5.89 6.60      0.75 0.89 1.00       0.247 0.139 0.150
  NO_RESET       6.30 7.24 7.00      1.00 1.00 1.00       0.109 0.111 0.111

So partial integration MEASURABLY OCCURS in the no-reset control (the condition
what_would_answer explicitly requires for the comparison to mean anything), and
the DV is non-degenerate. Queued rather than self-routed substrate_not_ready.

THE ARMS, AND WHY RATE_MATCHED IS NOT OPTIONAL
----------------------------------------------
  ALIGNED       production: phase_reset() executes at the salient event.
  RATE_MATCHED  the request is SUPPRESSED at the event and re-issued at
                event_step + U{K..2K}, clamped into the episode: the same
                number of extra E3 ticks, decoupled from the event. The clamp
                is load-bearing -- without it a reset requested late in an
                episode is scheduled past the episode end, never fires, and
                RATE_MATCHED drifts back toward NO_RESET so C3 would measure
                rate rather than phase (65-84 percent execution measured at
                an unclamped 3K window in the pre-queue probe).
  NO_RESET      phase_reset() never executes (what_would_answer's literal
                "no-reset control").

The probe row above is the reason RATE_MATCHED exists. ALIGNED does not only
re-align the E3 window, it also RAISES the E3 tick rate (0.19-0.30 vs 0.11).
An ALIGNED-vs-NO_RESET contrast alone therefore cannot separate "replans more
often" from "replans in phase with the event" -- and the latter is the whole of
MECH-091. RATE_MATCHED holds the rate roughly fixed and moves only the phase.

ISOLATION: one trained substrate per seed, three clock regimes. Each seed runs
ONE P0 prediction-loss warmup; the resulting weights are snapshotted and all
three arms are measured from that identical snapshot on a freshly-seeded env.
This removes the learning-trajectory confound, so the arms differ ONLY in
cycle-boundary alignment. Consequence, declared not hidden: the cells share
mutable state across arms, so every cell is stamped
`extra_ineligible_reasons=["shared_trained_snapshot_across_arms"]` and is NOT
reuse-eligible. That is the correctness guard doing its job, not a skipped mint.

DV-SYMMETRY DECLARATION (one line per arm; the 604c net)
--------------------------------------------------------
The DV, `straddle_frac`, is a function of the relative OFFSETS between the
salient-event step sequence and the E3-tick step sequence. Its symmetry group is
joint time-translation of both sequences (and, for the rate-only reading,
permutation of reset times holding their COUNT fixed).
  ALIGNED       the manipulation shifts the E3 phase counter to the event, i.e.
                it changes exactly the offset the DV reads -- not invariant.
                DECLARED WEAKNESS: for this arm the readout is close to an
                implementation identity (phase_reset() sets the next advance()
                to tick), so C2 below is deliberately NOT the finding. C3 is.
  RATE_MATCHED  invariant under the reset COUNT (it holds the count fixed by
                construction) and NOT invariant under reset TIMING, which is
                the offset the DV reads. This is the arm that carries the
                measurement, and its value is not fixed by any identity.
  NO_RESET      the manipulation removes the event-locked component of the tick
                sequence, leaving the periodic counter -- the offset
                distribution becomes uniform-ish over [1, K]; not invariant.

PRE-REGISTERED PRECONDITIONS (readiness-kind; below-floor self-routes
substrate_not_ready_requeue, never a substrate verdict)
  P1 min-over-cells n_salient_events >= 20
  P2 min-over-seeds straddle_frac(NO_RESET) >= 0.50  -- the no-reset control
     must actually EXHIBIT partial integration, else nothing is being compared
  P3 max-over-seeds |e3_rate(RATE_MATCHED) - e3_rate(ALIGNED)| / e3_rate(ALIGNED)
     <= 0.35 (direction upper) -- the rate match must hold, else C3 measures
     rate and not phase

PRE-REGISTERED ACCEPTANCE CRITERIA
  C1 manipulation check (NOT load-bearing): every seed has
     n_resets_executed(ALIGNED) > 0 and n_resets_executed(NO_RESET) == 0
  C2 load-bearing: straddle_frac(ALIGNED) <= 0.05 on every seed
  C3 load-bearing: straddle_frac(RATE_MATCHED) - straddle_frac(ALIGNED) >= 0.50
     on every seed  -- the non-tautological one; see the DV-symmetry block
  C4 corroborating (NOT load-bearing): mean_integration_lag(NO_RESET) -
     mean_integration_lag(ALIGNED) >= 3.0 steps on every seed

COMBINATION RULE: outcome PASS iff every precondition is met AND C1 AND C2 AND
C3. It is a plain AND -- recorded explicitly per the multi-criterion rule, so a
reader need not open this docstring to know the gate.

  PASS         -> evidence_direction "supports": salient events measurably reset
                  E3's cycle boundary and eliminate straddling cycles, and the
                  effect survives holding the replanning RATE fixed.
  C2/C3 fail   -> "weakens". This is what_would_answer's FALSIFYING signature:
                  salient events produce no measurable change in cycle-boundary
                  timing beyond noise. Distinct from the already-ruled-out
                  "no clock exists" reading that closed both EXQ-133 runs.
  P unmet      -> "non_contributory" / substrate_not_ready_requeue.

WHAT A PASS DOES AND DOES NOT ESTABLISH (read before weighting this run)
-----------------------------------------------------------------------
A PASS establishes that the salient-event triggers wired on 2026-08-17 DO reset
E3's cycle boundary on the real substrate, that the no-reset control genuinely
exhibits the straddling cycles MECH-091 predicts, and that the difference
survives holding the replanning RATE fixed. It is a STRUCTURAL/TIMING result.

It does NOT establish that the re-alignment improves behaviour. The behavioural
consequence is recorded per cell as `post_event_harm_mean` (harm accrued in the
K steps after a salient event) but is explicitly NON-GATING: the pre-queue probe
found it noisy and sign-inconsistent across seeds on an untrained agent, so
pre-registering it as a criterion would have manufactured a coin flip. A run
that PASSes here should be weighted as "the mechanism does what it says", not as
"the mechanism pays off". The payoff question is separate follow-on work.

A scale check at 72-step episodes (2026-08-22, 12 warmup / 10 measure episodes,
seeds 42 and 7) put every precondition and criterion comfortably inside its
threshold: rate deviation 0.014 / 0.055 against a 0.35 ceiling, C3 straddle
delta 0.972 / 0.956 against a 0.50 floor, C4 lag delta 7.24 / 6.94 against 3.0.
So a PASS is the EXPECTED outcome, and C2 in particular is close to an
implementation identity (declared above). The informative content is C3 and the
recorded coverage/behavioural diagnostics, not the pass/fail bit alone.

KNOWN LIMITATION -- TRIGGER COVERAGE IS 2 OF 3, MEASURED AND RECORDED
---------------------------------------------------------------------
`trigger_coverage` per cell counts phase_reset() requests by MECH-091 event
class, attributed at runtime from `ree_core/agent.py`'s own source (so it
survives line drift, and `assert_trigger_wiring()` fails loudly if a substrate
edit ever REMOVES a call site). In the 2026-08-22 probe on this env:
  harm          20-111 per cell  (fires abundantly)
  commit_entry  5-6 per cell     (~one per episode, via the readiness-conjunction
                                  admission site under beta_gate_bistable=True)
  completion    0 per cell       -- the ARC-028/MECH-105 hippocampal-completion
                                  release never fires here: the completion signal
                                  topped out at 0.20-0.39 against a release
                                  threshold of 0.75.
So this run exercises 2 of MECH-091's 3 named salient events. That is exactly the
under-testing the substrate entry's own severity_note anticipated and classified
`degrading`, not corrupting -- "a run simply exercises one trigger instead of
three, which is visible in the readout". It is visible here by construction:
`trigger_coverage` and `completion_signal_max` vs `completion_release_threshold`
are recorded on every cell, as a NON-GATING diagnostic. It is deliberately NOT an
`interpretation.preconditions[]` entry, because the indexer reads that list flat
and arm-blind and would flag the WHOLE run `precondition_unmet` on it, burying a
valid two-trigger result. A completion-trigger probe is separate follow-on work.

SUBSTRATE-PATH OVERLAP GATE (Step 2.5c), checked 2026-08-22
-----------------------------------------------------------
Two OPEN `corrupting` substrate_queue entries list `ree_core/agent.py`, which
this driver necessarily exercises. Neither defect is reachable here, and both
are held off by config rather than by assumption:
  mode-governance-engagement -- the defect is in the affinity-input clamp in
    ree_core/cingulate/salience_coordinator.py and in the per-seed existential
    gate in experiments/_lib/regime_occupancy_gate.py. This run imports neither
    and leaves use_salience_coordinator at its REEConfig default (False).
  MECH122-CONTENT-PACKAGING-SPINDLE-SELECTION -- scoped by its own author to
    agent.py::run_sws_schema_pass and mel_consumer.py::relative_novelty
    ("Paths are scoped to the two functions rather than whole files to bound the
    Step 2.5c gate"). This run has no sleep loop, so run_sws_schema_pass is
    never called.
mech203-valence-pool-admissibility overlaps too but carries no severity -> no
action per the gate.

SLEEP DRIVER: none (use_sleep_loop left at its REEConfig default, False).

ethics_preflight:
  involves_negative_valence: false
  involves_suffering_like_state: false
  involves_self_model: false
  involves_inescapability_or_helplessness: false
  involves_offline_replay_over_harm: false
  involves_social_mind_or_language: false
  involves_human_data_or_clinical_context: false
  decision: allow
"""

import argparse
import copy
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._metrics import check_degeneracy  # noqa: E402
from experiments._lib.baselines import mech091_phase_reset as base  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_944_mech091_salient_event_cycle_boundary"
QUEUE_ID = "V3-EXQ-944"
CLAIM_IDS: List[str] = ["MECH-091"]
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

SEEDS: List[int] = [42, 7, 13, 100, 200]
ARM_NAMES = list(base.ARM_NAMES)

WARMUP_EPISODES = base.WARMUP_EPISODES
MEASURE_EPISODES = base.MEASURE_EPISODES
STEPS_PER_EPISODE = base.STEPS_PER_EPISODE

# --- Pre-registered thresholds. Defined HERE, never inferred post-hoc. -------
P1_MIN_SALIENT_EVENTS = 20          # floor, per cell
P2_NO_RESET_STRADDLE_FLOOR = 0.50   # floor, per seed
P3_RATE_MATCH_TOLERANCE = 0.35      # CEILING on relative e3-rate deviation
C2_ALIGNED_STRADDLE_CEILING = 0.05
C3_STRADDLE_DELTA_FLOOR = 0.50
C4_LAG_DELTA_FLOOR = 3.0

COMBINATION_RULE = (
    "PASS iff (P1 AND P2 AND P3) AND C1 AND C2 AND C3. Plain AND. C4 is "
    "corroborating and does not gate. C1 is a manipulation check, not a finding."
)


def _fmt(x: Optional[float], nd: int = 4) -> Optional[float]:
    return None if x is None else round(float(x), nd)


def run(dry_run: bool = False):
    warmup_eps = 2 if dry_run else WARMUP_EPISODES
    measure_eps = 2 if dry_run else MEASURE_EPISODES
    steps = 20 if dry_run else STEPS_PER_EPISODE
    seeds = SEEDS[:2] if dry_run else SEEDS

    sites = base.resolve_trigger_sites()
    wiring = base.assert_trigger_wiring(sites)
    print("MECH-091 trigger wiring (call sites in agent.py): %s" % wiring, flush=True)

    zg = ZGoalStreamAccumulator()
    arm_results: List[Dict[str, Any]] = []
    warmup_lengths: Dict[str, List[int]] = {}
    last_agent = None

    for seed in seeds:
        # ONE warmup per seed, shared by all three arms (isolation -- see docstring).
        env = base.build_env(seed)
        agent = base.build_agent(env)
        torch.manual_seed(seed)
        lens = base.warmup(agent, env, seed, warmup_eps, steps,
                           progress_every=max(1, warmup_eps // 4))
        warmup_lengths[str(seed)] = lens
        snapshot = copy.deepcopy(agent.state_dict())

        for arm in ARM_NAMES:
            print("Seed %d Condition %s" % (seed, arm), flush=True)
            with arm_cell(
                seed,
                config_slice=base.arm_config_slice(arm),
                script_path=Path(__file__),
                config_slice_declared=True,
                # The three arms are measured from ONE per-seed trained snapshot
                # so they differ only in cycle-boundary alignment. That is shared
                # mutable cross-cell state, so the cell is never reuse-eligible.
                extra_ineligible_reasons=["shared_trained_snapshot_across_arms"],
            ) as cell:
                env_arm = base.build_env(seed)
                agent_arm = base.build_agent(env_arm)
                agent_arm.load_state_dict(snapshot)
                row = base.measure_cell(
                    agent_arm, env_arm, arm, seed, measure_eps, steps, sites,
                    zg=zg, progress_every=max(1, measure_eps // 3),
                )
                cell.stamp(row)
            last_agent = agent_arm
            # Per-cell verdict = this cell's own adequacy (did the decisive
            # readout engage at all). The RUN outcome is the cross-arm decision
            # below; a per-cell PASS is not a claim verdict.
            cell_ok = (row["n_salient_events"] >= (1 if dry_run else P1_MIN_SALIENT_EVENTS)
                       and row["straddle_frac"] is not None)
            row["cell_adequate"] = bool(cell_ok)
            arm_results.append(row)
            print("verdict: %s" % ("PASS" if cell_ok else "FAIL"), flush=True)

    by = {(r["arm"], r["seed"]): r for r in arm_results}

    def val(arm: str, seed: int, key: str):
        r = by.get((arm, seed))
        return None if r is None else r.get(key)

    # --- Preconditions (readiness-kind; numeric so the indexer recomputes met).
    ev_counts = [r["n_salient_events"] for r in arm_results]
    worst_ev = min(ev_counts) if ev_counts else 0
    worst_ev_cell = min(arm_results, key=lambda r: r["n_salient_events"]) if arm_results else None

    nr_straddles = [(s, val(base.ARM_NO_RESET, s, "straddle_frac")) for s in seeds]
    nr_defined = [(s, v) for s, v in nr_straddles if v is not None]
    worst_nr = min((v for _s, v in nr_defined), default=0.0)
    worst_nr_seed = min(nr_defined, key=lambda t: t[1])[0] if nr_defined else None

    rate_devs = []
    for s in seeds:
        a = val(base.ARM_ALIGNED, s, "e3_tick_rate")
        m = val(base.ARM_RATE_MATCHED, s, "e3_tick_rate")
        if a and m is not None and a > 0:
            rate_devs.append((s, abs(m - a) / a))
    worst_rate = max((d for _s, d in rate_devs), default=1.0)
    worst_rate_seed = max(rate_devs, key=lambda t: t[1])[0] if rate_devs else None

    preconditions = [
        {"name": "salient_events_per_cell_floor",
         "description": "min over cells of n_salient_events -- the DV is per-event, "
                        "so a cell below this measured nothing usable",
         "kind": "readiness",
         "control": "every cell of every arm; the harm trigger fires abundantly on "
                    "this hazard-carrying env by construction",
         "measured": float(worst_ev), "threshold": float(P1_MIN_SALIENT_EVENTS),
         "direction": "lower",
         "offending_cell": (None if worst_ev_cell is None
                            else "%s/seed%d" % (worst_ev_cell["arm"], worst_ev_cell["seed"])),
         "met": bool(worst_ev >= P1_MIN_SALIENT_EVENTS)},
        {"name": "no_reset_control_shows_partial_integration",
         "description": "min over seeds of straddle_frac(NO_RESET). what_would_answer "
                        "requires a no-reset control in which partial integration "
                        "MEASURABLY OCCURS; below this floor there is nothing to compare",
         "kind": "readiness",
         "control": "NO_RESET arm -- phase_reset() never executes, so the E3 tick is "
                    "the bare periodic counter and events land at arbitrary phase",
         "measured": _fmt(worst_nr), "threshold": float(P2_NO_RESET_STRADDLE_FLOOR),
         "direction": "lower",
         "offending_cell": (None if worst_nr_seed is None else "NO_RESET/seed%d" % worst_nr_seed),
         "met": bool(worst_nr >= P2_NO_RESET_STRADDLE_FLOOR)},
        {"name": "rate_match_holds",
         "description": "max over seeds of |e3_rate(RATE_MATCHED)-e3_rate(ALIGNED)|"
                        "/e3_rate(ALIGNED). C3 only separates PHASE from RATE while "
                        "the rates are matched",
         "kind": "readiness",
         "control": "RATE_MATCHED re-issues every suppressed reset, so its executed "
                    "count matches ALIGNED's up to end-of-episode truncation",
         "measured": _fmt(worst_rate), "threshold": float(P3_RATE_MATCH_TOLERANCE),
         "direction": "upper", "comparator": "<=",
         "offending_cell": (None if worst_rate_seed is None
                            else "RATE_MATCHED/seed%d" % worst_rate_seed),
         "met": bool(worst_rate <= P3_RATE_MATCH_TOLERANCE)},
    ]
    preconditions_met = all(p["met"] for p in preconditions)

    # --- Criteria.
    c1_seeds, c2_seeds, c3_seeds, c4_seeds = [], [], [], []
    per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        a_ex = val(base.ARM_ALIGNED, s, "n_resets_executed")
        n_ex = val(base.ARM_NO_RESET, s, "n_resets_executed")
        a_st = val(base.ARM_ALIGNED, s, "straddle_frac")
        m_st = val(base.ARM_RATE_MATCHED, s, "straddle_frac")
        n_st = val(base.ARM_NO_RESET, s, "straddle_frac")
        a_lag = val(base.ARM_ALIGNED, s, "mean_integration_lag")
        n_lag = val(base.ARM_NO_RESET, s, "mean_integration_lag")
        c1 = bool(a_ex is not None and a_ex > 0 and n_ex == 0)
        c2 = bool(a_st is not None and a_st <= C2_ALIGNED_STRADDLE_CEILING)
        c3 = bool(a_st is not None and m_st is not None
                  and (m_st - a_st) >= C3_STRADDLE_DELTA_FLOOR)
        c4 = bool(a_lag is not None and n_lag is not None
                  and (n_lag - a_lag) >= C4_LAG_DELTA_FLOOR)
        c1_seeds.append(c1); c2_seeds.append(c2); c3_seeds.append(c3); c4_seeds.append(c4)
        per_seed.append({
            "seed": s,
            "aligned_straddle_frac": _fmt(a_st),
            "rate_matched_straddle_frac": _fmt(m_st),
            "no_reset_straddle_frac": _fmt(n_st),
            "aligned_mean_lag": _fmt(a_lag),
            "no_reset_mean_lag": _fmt(n_lag),
            "aligned_e3_rate": _fmt(val(base.ARM_ALIGNED, s, "e3_tick_rate")),
            "rate_matched_e3_rate": _fmt(val(base.ARM_RATE_MATCHED, s, "e3_tick_rate")),
            "no_reset_e3_rate": _fmt(val(base.ARM_NO_RESET, s, "e3_tick_rate")),
            "straddle_delta_rate_matched_minus_aligned": (
                None if (a_st is None or m_st is None) else _fmt(m_st - a_st)),
            "c1": c1, "c2": c2, "c3": c3, "c4": c4,
        })

    c1_pass = all(c1_seeds); c2_pass = all(c2_seeds)
    c3_pass = all(c3_seeds); c4_pass = all(c4_seeds)
    overall_pass = bool(preconditions_met and c1_pass and c2_pass and c3_pass)

    criteria = [
        {"name": "C1_manipulation_check", "load_bearing": False, "passed": c1_pass,
         "description": "ALIGNED executes resets and NO_RESET executes none, every seed"},
        {"name": "C2_aligned_straddle_ceiling", "load_bearing": True, "passed": c2_pass,
         "description": "straddle_frac(ALIGNED) <= %.2f every seed" % C2_ALIGNED_STRADDLE_CEILING},
        {"name": "C3_rate_matched_straddle_delta", "load_bearing": True, "passed": c3_pass,
         "description": "straddle_frac(RATE_MATCHED) - straddle_frac(ALIGNED) >= %.2f "
                        "every seed (phase, not rate)" % C3_STRADDLE_DELTA_FLOOR},
        {"name": "C4_no_reset_lag_delta", "load_bearing": False, "passed": c4_pass,
         "description": "mean_lag(NO_RESET) - mean_lag(ALIGNED) >= %.1f steps every seed"
                        % C4_LAG_DELTA_FLOOR},
    ]

    # Degeneracy. The load-bearing criteria compare ACROSS ARMS, so the
    # collection whose spread decides degeneracy is the pooled cross-arm
    # readout: zero spread there means the manipulation moved nothing and the
    # criteria could not have discriminated. Do NOT hand check_degeneracy the
    # per-seed DELTAS instead -- a constant delta across seeds is a perfectly
    # CONSISTENT effect, and it reports as "zero spread", which would set
    # non_degenerate:false on the strongest possible result and drop it from
    # governance confidence scoring. Caught in the 2026-08-22 dry run, which
    # reported exactly that on a delta of 1.0 on every seed.
    aligned_st = [v for v in (val(base.ARM_ALIGNED, s, "straddle_frac") for s in seeds)
                  if v is not None]
    matched_st = [v for v in (val(base.ARM_RATE_MATCHED, s, "straddle_frac") for s in seeds)
                  if v is not None]
    no_reset_st = [v for v in (val(base.ARM_NO_RESET, s, "straddle_frac") for s in seeds)
                   if v is not None]
    aligned_lag = [v for v in (val(base.ARM_ALIGNED, s, "mean_integration_lag") for s in seeds)
                   if v is not None]
    no_reset_lag = [v for v in (val(base.ARM_NO_RESET, s, "mean_integration_lag") for s in seeds)
                    if v is not None]
    deltas = [m - a for a, m in zip(aligned_st, matched_st)]
    degeneracy = check_degeneracy({
        "straddle_frac_across_arms": {"values": aligned_st + matched_st + no_reset_st},
        "mean_integration_lag_across_arms": {"values": aligned_lag + no_reset_lag},
    })

    criteria_non_degenerate = {
        # C1 is vacuous if no arm ever requested a reset.
        "C1": bool(sum(r["n_salient_events"] for r in arm_results) > 0),
        # C2 is vacuous if ALIGNED recorded no event with a following tick.
        "C2": bool(all(val(base.ARM_ALIGNED, s, "n_events_with_following_tick") or 0
                       for s in seeds)),
        # C3 is vacuous if the two arms are bit-identical in straddle_frac
        # (no spread across the contrast the criterion reads).
        "C3": bool(len(set(_fmt(d) for d in deltas)) > 1 or any(d != 0 for d in deltas)),
        "C4": bool(all(val(base.ARM_NO_RESET, s, "n_events_with_following_tick") or 0
                       for s in seeds)),
    }

    if not preconditions_met:
        label = "substrate_not_ready_requeue"
        evidence_direction = "non_contributory"
    elif overall_pass:
        label = "mech091_cycle_boundary_reset_confirmed"
        evidence_direction = "supports"
    elif not (c2_pass and c3_pass):
        label = "mech091_cycle_boundary_reset_not_supported"
        evidence_direction = "weakens"
    else:
        label = "mech091_cycle_boundary_reset_mixed"
        evidence_direction = "mixed"

    # Trigger coverage: recorded, NON-GATING. Deliberately not a
    # preconditions[] entry -- see the KNOWN LIMITATION block in the docstring.
    coverage_total = {c: 0 for c in base.TRIGGER_CLASSES}
    for r in arm_results:
        for c in base.TRIGGER_CLASSES:
            coverage_total[c] += r["trigger_coverage"].get(c, 0)
    comp_maxes = [r["completion_signal_max"] for r in arm_results
                  if r["completion_signal_max"] is not None]
    thresholds = [r["completion_release_threshold"] for r in arm_results
                  if r["completion_release_threshold"] is not None]

    metrics = {
        "n_cells": len(arm_results),
        "preconditions_met": preconditions_met,
        "c1_pass": c1_pass, "c2_pass": c2_pass, "c3_pass": c3_pass, "c4_pass": c4_pass,
        "overall_pass": overall_pass,
        "worst_cell_salient_events": worst_ev,
        "worst_seed_no_reset_straddle_frac": _fmt(worst_nr),
        "worst_seed_rate_match_deviation": _fmt(worst_rate),
        "mean_straddle_frac_aligned": _fmt(
            sum(aligned_st) / len(aligned_st)) if aligned_st else None,
        "mean_straddle_frac_rate_matched": _fmt(
            sum(matched_st) / len(matched_st)) if matched_st else None,
        "mean_straddle_delta": _fmt(sum(deltas) / len(deltas)) if deltas else None,
    }

    diagnostics = {
        "mech091_trigger_wiring_call_sites": wiring,
        "mech091_trigger_coverage_total": coverage_total,
        "mech091_completion_trigger_fired": coverage_total.get("completion", 0) > 0,
        "completion_signal_max_over_cells": _fmt(max(comp_maxes)) if comp_maxes else None,
        "completion_release_threshold": (thresholds[0] if thresholds else None),
        "completion_coverage_note": (
            "NON-GATING. The ARC-028/MECH-105 hippocampal-completion release did not "
            "fire in the 2026-08-22 pre-queue probe on this env (signal max 0.20-0.39 "
            "vs release threshold 0.75), so this run is expected to exercise 2 of "
            "MECH-091's 3 named salient events. Recorded here rather than as a "
            "precondition so a valid two-trigger result is not buried under a "
            "whole-run precondition_unmet."),
        "warmup_episode_lengths_by_seed": warmup_lengths,
    }

    result: Dict[str, Any] = {
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "status": "PASS" if overall_pass else "FAIL",
        "outcome": "PASS" if overall_pass else "FAIL",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": evidence_direction,
        "sleep_driver_pattern": "N/A (no sleep loop)",
        "combination_rule": COMBINATION_RULE,
        "metrics": metrics,
        "per_seed_results": per_seed,
        "arm_results": arm_results,
        "diagnostics": diagnostics,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "criteria": criteria,
            "combination_rule": COMBINATION_RULE,
        },
        "fatal_error_count": 0,
    }
    result.update(degeneracy)
    return result, zg, last_agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result, zg, last_agent = run(dry_run=args.dry_run)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["timestamp_utc"] = ts
    result["run_id"] = "%s_%s_v3" % (EXPERIMENT_TYPE, ts)
    result["architecture_epoch"] = ARCHITECTURE_EPOCH

    full_config = {
        "arms": ARM_NAMES,
        "seeds": SEEDS,
        "env_kwargs": dict(base.ENV_KWARGS),
        "warmup_episodes": WARMUP_EPISODES,
        "measure_episodes": MEASURE_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "alpha_world": base.ALPHA_WORLD,
        "beta_gate_bistable": base.BETA_GATE_BISTABLE,
        "decouple_k_range": [base.DECOUPLE_MIN_K, base.DECOUPLE_MAX_K],
        "p1_min_salient_events": P1_MIN_SALIENT_EVENTS,
        "p2_no_reset_straddle_floor": P2_NO_RESET_STRADDLE_FLOOR,
        "p3_rate_match_tolerance": P3_RATE_MATCH_TOLERANCE,
        "c2_aligned_straddle_ceiling": C2_ALIGNED_STRADDLE_CEILING,
        "c3_straddle_delta_floor": C3_STRADDLE_DELTA_FLOOR,
        "c4_lag_delta_floor": C4_LAG_DELTA_FLOOR,
        "dry_run": bool(args.dry_run),
    }

    out_path = write_flat_manifest(
        result,
        dry_run=args.dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=__file__,
        started_at=t0,
        agent=last_agent,
        z_goal_stream_stats=zg.stats(),
    )

    print("\nResult written to: %s" % out_path, flush=True)
    print("Status: %s  label: %s" % (result["status"], result["interpretation"]["label"]),
          flush=True)
    print("Trigger coverage: %s" % result["diagnostics"]["mech091_trigger_coverage_total"],
          flush=True)

    if args.dry_run:
        # Smoke assertion (skill Step 3.5): the decisive readout must be
        # non-trivially engaged BEFORE committing to the full seed x arm grid.
        assert all(r["n_salient_events"] > 0 for r in result["arm_results"]), (
            "SMOKE FAIL: some cell recorded zero salient events -- the DV is "
            "per-event, so the full run would measure nothing.")
        assert any(r["straddle_frac"] not in (None,) for r in result["arm_results"]), (
            "SMOKE FAIL: straddle_frac undefined in every cell.")
        print("[smoke] decisive-readout engagement OK", flush=True)

    emit_outcome(
        outcome=result["status"] if result["status"] in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
