"""
V3-EXQ-910b -- MECH-489 (SD-099) defensive-orienting VALENCE-GATING retest, with the
decision_counts readout corrected to count AT THE OVERRIDE TICK ONLY.

Supersedes V3-EXQ-910a in the lettered-iteration sense (CLAUDE.md EXQ Versioning Policy:
the scientific question is unchanged; the instrument was wrong). V3-EXQ-910's
trigger-alignment finding and V3-EXQ-910a's confirmation of it are NOT re-litigated here
-- see SCOPE below.

SLEEP DRIVER: K=10 multi-fire (SleepLoopManager, fires every 10 episodes) -- inherited
unchanged from the reused 906b substrate/curriculum; sleep is not the mechanism under
test here.

WHY THIS RE-QUEUE EXISTS
------------------------
SD-ORIENTING-DECISION-SCALE had TWO independent defects. The SUBSTRATE half landed
2026-08-20 (ree-v3 775eb55): the post-override decision window now expires
unconditionally, so a "resume" decision no longer persists for the rest of the episode
and the countdown no longer freezes on ticks where the score-bias gate does not open
(pinned by tests/contracts/test_orienting_decision_window_expiry.py).

The DRIVER half is what this script fixes. The 910-lineage `_decision_alignment` read the
latched `agent._orienting_decision` once per ENV STEP, while the persistence window is
denominated in E3 TICKS. `select_action` returns the held action at
`ree_core/agent.py:6546` (`if not ticks["e3_tick"] and self._last_action is not None`)
BEFORE the Component 4/5 orienting block at line 8130 is ever reached, so on a non-E3
tick `agent._orienting_last_output` and `agent._orienting_decision` both retain the
PREVIOUS tick's values. At the default `heartbeat.e3_steps_per_tick = 10` a single
override therefore appears in the per-step log many times over.

MEASURED, not assumed (this session's Step 2.5a probe, 40 env steps, seed 7, ON arm):
`agent._orienting_last_output` read non-None on 40/40 env steps while only 4/40 ticks
were genuine E3 ticks -- a 10x replication. The same probe confirmed the per-step read
is fresh on exactly the ticks where the orienting block ran, plus the one first-step
case where `_last_action is None` makes the early return unreachable.

THE CONSEQUENCE THE PRIOR LINEAGE DID NOT DRAW: `orienting_override_fired` is read off
the SAME latched `_orienting_last_output`, so the 910-lineage `n_overrides` counter was
ITSELF replicated by the same factor -- it was never a trustworthy denominator. The
autopsy's observation that the substrate sets the decision "once, synchronously, in the
same `if _do_out.override_fired:` block" is true of the SUBSTRATE VARIABLE at the moment
it is written, and says nothing about a DRIVER that samples the latched log field once
per env step. That is why V3-EXQ-910's 206/42 = 4.90 is a ratio of two inflated numbers
(decision latched over the ~5-E3-tick bias window ~= 50 env steps; override latched over
one E3 tick ~= 10 env steps) rather than the ~5 the window length alone would give.

THE FIX (driver-side; no substrate change)
------------------------------------------
This script taps `REEAgent.select_action` with a NON-DESTRUCTIVE SENTINEL MARKER, the
same idiom as `experiments/_lib/fresh_select.py` (the 699/689d instrument repair) and for
the same documented reason -- a marker is provably inert where nulling an attribute may
not be:

  * before each `select_action`, stamp a private attribute onto the CURRENT
    `agent._orienting_last_output` object (if any);
  * after the call, the attribute is ABSENT iff the orienting block ran and replaced the
    object with a fresh one.

`DefensiveOrientingGate.tick()` constructs a NEW `DefensiveOrientingOutput` at every one
of its three return sites (`ree_core/pag/defensive_orienting.py:220/233/330`) and never
returns a cached instance, so the marker is an exact per-tick freshness signal. Nothing
in `ree_core` reads the private attribute, so the tap cannot perturb selection dynamics.

On a FRESH tick with `override_fired`, the driver increments `n_override_ticks` and
classifies `agent._orienting_decision` -- read synchronously, immediately after the call
that set it, in the same block that increments the override count. That yields EXACTLY
`n_override_ticks` classifications and is immune to the latching entirely.

Nulling `_orienting_last_output` would in fact ALSO have been safe here, and this was
verified rather than assumed: its only other reader is the motor-arrest block at
`agent.py:9239`, which lies inside the SAME `select_action` call downstream of the
assignment at 8196 (verified: no other `def` between 5839 and 9260), and the Step 2.5a
probe produced bit-identical action sequences with and without the clear. The sentinel is
used anyway because it needs no such proof.

PRE-REGISTERED READINESS ASSERTION (from the chip brief, machine-checked as criterion C1):
`sum(decision_counts) == n_override_ticks` EXACTLY -- not ~e3_steps_per_tick times it.
The legacy per-env-step readout is computed alongside and reported as
`diagnostics.legacy_per_env_step_readout`, with the realized inflation ratios, so the
defect and its magnitude are MEASURED by this run rather than inferred from the 910
aggregates.

SCOPE
-----
Scoped to the VALENCE-GATING sub-claim (MECH-489 Components 4/5) ONLY. The
trigger-alignment sub-claim (Components 1-3) already stands as fairly falsified per
failure_autopsy_V3-EXQ-910_2026-08-10 and its 910a reconfirmation, and is NOT re-tested:
the event-alignment and behaviour-coupling numbers are still computed on the identical
906b substrate and reported for context, but they are NOT pass criteria of this run.

POWER
-----
The corrected counter is ~e3_steps_per_tick times SMALLER than the legacy one, so the
prior runs' apparent n=21/42 corresponds to only ~2-4 genuine overrides -- far too few to
judge a three-class non-degeneracy. Overrides are produced only during the observational
EVAL phase, which is ~8% of a cell's steps (220 train episodes x 200 steps vs 8 segments
x <=500 steps), so lengthening EVAL is far cheaper per additional override than adding
seeds. This run therefore raises `EVAL_EPISODES` 8 -> 48 (a module-attribute swap on the
reused 906b module, restored via try/finally -- eval is ONE continuous multi-segment life
per 906b's CONTINUITY REDESIGN, so lengthening it is a clean extension, not a semantic
change) and uses 3 seeds. Expected genuine overrides ~19 pooled across ON-arm seeds,
against a pre-registered floor of 10. Below the floor the run self-routes
`substrate_not_ready_requeue` -- which would itself be the informative finding that
SD-099's override rate at its shipped defaults is too low to measure.

ARM STRUCTURE / DV SYMMETRY
---------------------------
ARM orienting_on is the ONLY scored arm. Its DV is the class histogram of the resolved
decision over override ticks. The symmetry group of that DV is (i) permutation/relabelling
of the three decision classes and (ii) any common monotone rescaling of the two z-scored
channels. The manipulation (enabling the gate) is not invariant under either: with the
gate off the DV has EMPTY SUPPORT -- there are no override ticks at all -- so the
manipulation changes the DV's domain, not merely its value.

ARM orienting_off is a STRUCTURAL NEGATIVE CONTROL, not a scored arm: with
`use_defensive_orienting=False` the agent has no `defensive_orienting` gate, so
`n_override_ticks == 0` is forced by construction. Per the two-dispositions rule in
/queue-experiment Step 3 it is scoped OUT of criterion scoring (disposition (b)) and its
zero is recorded as a diagnostic rather than dressed up as a criterion it passes
vacuously. It is retained because EVB-0610's dispatch contract requires a discriminative
pair on >=2 matched shared seeds, and because it is the control that makes the ON arm's
override count attributable to the gate rather than to the instrument.

KNOWN OPEN SUBSTRATE DEFECTS ON THIS DRIVER'S PATH (/queue-experiment Step 2.5c)
--------------------------------------------------------------------------------
Recorded rather than silently passed. None is on this run's DV path, and all are
arm-invariant in a single-variable ablation:
  * SD-ORIENTING-DECISION-SCALE (degrading, `agent.py::select_action`) -- this run IS its
    validation; that is the point, not a confound.
  * SD-E3-SCORER-COMPLETION (degrading, `e3_selector.py`) -- affects scoring completeness,
    identical in both arms.
  * mode-governance-engagement (corrupting, whole-file `agent.py` + `_lib/regime_occupancy_gate.py`)
    -- its severity_note scopes the live successor defect to the `regime_occupancy_gate.py`
    gate pattern, which this driver does not use.
  * MECH122-CONTENT-PACKAGING-SPINDLE-SELECTION (corrupting,
    `agent.py::run_sws_schema_pass`, `mel_consumer.py::relative_novelty`) -- its own
    severity_note scopes the risk to "any future experiment consuming relative_novelty".
    No criterion here reads any sleep/SWS/novelty quantity.
  * contextmemory-write-path-addressing-degeneracy (corrupting, `e1_deep.py`) -- affects
    E1 context-memory write addressing, identical in both arms and read by no criterion.

GOV-REUSE-1 (Step 2.4)
----------------------
Decisive readout: the count of resolved approach/withdraw/resume decisions per GENUINE
override tick. Checked `v3_exq_910_..._20260810T004433Z_v3.json` and
`v3_exq_910a_..._20260810T213616Z_v3.json` -- both record only the latched per-env-step
counter (the defect), neither retained a per-step episode log, and both predate the
2026-08-20 substrate amend. The readout is therefore neither recorded nor derivable by
reanalysis; a run is required.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments.pack_writer import write_flat_manifest
from experiments.v3_exq_906b_full_stack_observational_fishtank import (
    _make_config,
    run_seed,
)
import experiments.v3_exq_906b_full_stack_observational_fishtank as _fishtank906b
from ree_core.agent import REEAgent

EXPERIMENT_TYPE = "v3_exq_910b_mech489_orienting_decision_at_override_tick_retest"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS: List[str] = ["MECH-489"]
SUPERSEDES = "V3-EXQ-910a"

SEEDS = [0, 1, 2]
ARMS = ["orienting_off", "orienting_on"]
SCORED_ARM = "orienting_on"
CONTROL_ARM = "orienting_off"

# Observational-eval length for THIS run (906b default is 8). See POWER above: overrides
# are produced only in eval, which is ~8% of a cell's steps, so lengthening eval is far
# cheaper per additional override than adding seeds. Applied by module-attribute swap on
# the reused 906b module, restored via try/finally.
EVAL_EPISODES_OVERRIDE = 48

DECISION_CLASSES = ("approach", "withdraw", "resume")

# Pre-registered readiness floor for the DECISIVE criterion (C2), stated on the CORRECTED
# counter. 10 genuine override ticks is the minimum at which "is the resolved decision
# unconditionally one class" is a question about the mechanism rather than about small
# numbers. NOT comparable to V3-EXQ-910a's floor of 5, which was applied to the inflated
# per-env-step counter and so was ~10x weaker than it appeared.
MIN_OVERRIDE_TICKS = 10
# The tap must observe enough genuine orienting ticks for the freshness signal to mean
# anything at all.
MIN_FRESH_ORIENTING_TICKS = 100
# ...and at least one LATCHED tick, or the sentinel is reporting every tick as fresh --
# the one tap failure mode that is not self-catching (see _lib/fresh_select.py).
MIN_LATCHED_TICKS = 1

# Context-only (NOT pass criteria of this run -- the trigger-alignment sub-claim is not
# re-litigated here; see SCOPE). Kept so the manifest carries the same descriptive
# readouts as its predecessors.
GROUND_TRUTH_EVENT_FIELDS = [
    "limb_damage_injected", "external_hazard_injected", "world_rule_shift_occurred",
]
ALIGNMENT_WINDOW = 3
LEGACY_SURPRISE_SPIKE_THRESHOLD = 0.040

# Private sentinel attribute stamped on the CURRENT DefensiveOrientingOutput before every
# select_action. Namespaced per driver (mirrors _lib/fresh_select.marker_key_for) so two
# concurrently-instrumented drivers sharing one agent can never collide.
_STALE_MARKER = "_exq910b_orienting_stale_marker"


def _new_cell_counters() -> Dict[str, Any]:
    return {
        "n_select_calls": 0,
        "n_fresh_orienting_ticks": 0,
        "n_latched_ticks": 0,
        "n_trigger_ticks": 0,
        "n_override_ticks": 0,
        "decision_counts": {k: 0 for k in DECISION_CLASSES},
        "n_override_ticks_unclassified_decision": 0,
    }


class _OrientingTickTap:
    """Counts GENUINE orienting ticks by sentinel-marking the previous output object.

    Only counts while `measuring` is True, which `run()` sets exactly around 906b's
    `_observational_run` -- so the tap's scope matches the per-step episode log's scope
    and the corrected and legacy counters are directly comparable.
    """

    def __init__(self) -> None:
        self.measuring = False
        self.cell: Optional[Dict[str, Any]] = None

    def observe(self, agent: REEAgent, before: bool) -> None:
        if not self.measuring or self.cell is None:
            return
        if before:
            prev = getattr(agent, "_orienting_last_output", None)
            if prev is not None:
                try:
                    setattr(prev, _STALE_MARKER, True)
                except Exception:
                    # A slotted/frozen output would make the marker unusable; fail loud
                    # rather than silently reporting every tick as fresh.
                    raise RuntimeError(
                        "V3-EXQ-910b tap: cannot stamp the freshness sentinel on "
                        "DefensiveOrientingOutput -- the corrected counter would be "
                        "meaningless. Do not interpret this run."
                    )
            return
        c = self.cell
        c["n_select_calls"] += 1
        cur = getattr(agent, "_orienting_last_output", None)
        fresh = (cur is not None) and (not getattr(cur, _STALE_MARKER, False))
        if not fresh:
            c["n_latched_ticks"] += 1
            return
        c["n_fresh_orienting_ticks"] += 1
        if bool(getattr(cur, "trigger_fired", False)):
            c["n_trigger_ticks"] += 1
        if bool(getattr(cur, "override_fired", False)):
            # THE FIX: count the decision here -- at the override tick, synchronously,
            # in the same block that increments the override count.
            c["n_override_ticks"] += 1
            decision = getattr(agent, "_orienting_decision", None)
            if decision in c["decision_counts"]:
                c["decision_counts"][decision] += 1
            else:
                c["n_override_ticks_unclassified_decision"] += 1


_TAP = _OrientingTickTap()
_ZG = ZGoalStreamAccumulator()
_ORIG_SELECT_ACTION = REEAgent.select_action


def _tapped_select_action(self, *args, **kwargs):
    _TAP.observe(self, before=True)
    action = _ORIG_SELECT_ACTION(self, *args, **kwargs)
    _TAP.observe(self, before=False)
    return action


def _make_config_orienting_on(env):
    """ON arm: the OFF arm's exact config plus use_defensive_orienting=True at its shipped
    SD-099 defaults. Deliberately NOT re-tuned here."""
    cfg = _make_config(env)
    cfg.use_defensive_orienting = True
    return cfg


def _run_arm_seed(arm: str, seed: int, dry_run: bool = False) -> Dict[str, Any]:
    """Run one (arm, seed) cell with the tap armed only around the observational eval."""
    orig_make_config = _fishtank906b._make_config
    orig_observational_run = _fishtank906b._observational_run
    orig_eval_episodes = _fishtank906b.EVAL_EPISODES
    orig_select_action = REEAgent.select_action

    def _measured_observational_run(*a, **kw):
        _TAP.measuring = True
        try:
            return orig_observational_run(*a, **kw)
        finally:
            _TAP.measuring = False

    try:
        if arm == SCORED_ARM:
            _fishtank906b._make_config = _make_config_orienting_on
        if not dry_run:
            _fishtank906b.EVAL_EPISODES = EVAL_EPISODES_OVERRIDE
        _fishtank906b._observational_run = _measured_observational_run
        REEAgent.select_action = _tapped_select_action
        _TAP.cell = _new_cell_counters()
        result = run_seed(seed, dry_run=dry_run)
    finally:
        _fishtank906b._make_config = orig_make_config
        _fishtank906b._observational_run = orig_observational_run
        _fishtank906b.EVAL_EPISODES = orig_eval_episodes
        REEAgent.select_action = orig_select_action
        _TAP.measuring = False

    result["orienting_tick_counters"] = _TAP.cell
    _TAP.cell = None
    agent = result.pop("agent", None)
    if agent is not None:
        _ZG.observe(agent)
    return result


def _flatten_steps(episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [s for ep in episodes for s in ep.get("steps", [])]


def _pool_counters(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    pooled = _new_cell_counters()
    for c in cells:
        if not c:
            continue
        for k, v in c.items():
            if k == "decision_counts":
                for cls in DECISION_CLASSES:
                    pooled["decision_counts"][cls] += int(v.get(cls, 0))
            else:
                pooled[k] += int(v)
    return pooled


def _legacy_per_env_step_readout(all_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The EXACT 910/910a readout, reproduced so this run MEASURES the defect it fixes
    rather than inferring its magnitude from the prior manifests' aggregates."""
    counts = {k: 0 for k in DECISION_CLASSES}
    n_overrides = 0
    n_triggers = 0
    for s in all_steps:
        if s.get("orienting_override_fired"):
            n_overrides += 1
        if s.get("orienting_trigger_fired"):
            n_triggers += 1
        d = s.get("orienting_decision")
        if d in counts:
            counts[d] += 1
    return {
        "n_env_steps": len(all_steps),
        "n_overrides_latched": n_overrides,
        "n_triggers_latched": n_triggers,
        "decision_counts_latched": counts,
        "decision_counts_latched_sum": sum(counts.values()),
    }


def _event_trigger_alignment(all_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """CONTEXT ONLY (not a pass criterion here) -- see SCOPE."""
    per_event: Dict[str, Any] = {}
    for field in GROUND_TRUTH_EVENT_FIELDS:
        n_events = 0
        n_aligned = 0
        for i, s in enumerate(all_steps):
            if not s.get(field):
                continue
            n_events += 1
            if any(w.get("orienting_trigger_fired")
                   for w in all_steps[i:i + ALIGNMENT_WINDOW]):
                n_aligned += 1
        per_event[field] = {
            "n_events": n_events,
            "n_aligned": n_aligned,
            "alignment_rate": (n_aligned / n_events) if n_events > 0 else None,
        }
    return per_event


def _coupling_stats(episodes: List[Dict[str, Any]], spike_key_fn) -> Dict[str, Any]:
    """CONTEXT ONLY (not a pass criterion here) -- see SCOPE."""
    n_spike = n_spike_moved = n_spike_modechange = 0
    n_nospike = n_nospike_moved = n_nospike_modechange = 0
    for ep in episodes:
        steps = ep.get("steps", [])
        for i in range(len(steps) - 1):
            spiked = bool(spike_key_fn(steps[i]))
            nxt = steps[i + 1]
            moved = nxt.get("pos") != steps[i].get("pos")
            mode_changed = nxt.get("mode") != steps[i].get("mode")
            if spiked:
                n_spike += 1
                n_spike_moved += int(moved)
                n_spike_modechange += int(mode_changed)
            else:
                n_nospike += 1
                n_nospike_moved += int(moved)
                n_nospike_modechange += int(mode_changed)
    return {
        "n_spike": n_spike,
        "p_moved_given_spike": (n_spike_moved / n_spike) if n_spike else None,
        "p_modechange_given_spike": (n_spike_modechange / n_spike) if n_spike else None,
        "n_nospike": n_nospike,
        "p_moved_unconditional": (n_nospike_moved / n_nospike) if n_nospike else None,
        "p_modechange_unconditional": ((n_nospike_modechange / n_nospike)
                                       if n_nospike else None),
    }


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    return (numerator / denominator) if denominator else None


def run(seeds: Optional[List[int]] = None, dry_run: bool = False) -> Dict[str, Any]:
    if seeds is None:
        seeds = SEEDS if not dry_run else [0]
    arms = ARMS

    arm_results: List[Dict[str, Any]] = []
    per_arm_pooled_steps: Dict[str, List[Dict[str, Any]]] = {a: [] for a in arms}
    per_arm_episodes: Dict[str, List[Dict[str, Any]]] = {a: [] for a in arms}
    per_arm_cells: Dict[str, List[Dict[str, Any]]] = {a: [] for a in arms}

    total_runs = len(arms) * len(seeds)
    run_idx = 0
    for arm in arms:
        for seed in seeds:
            run_idx += 1
            print(f"Seed {seed} Condition {arm}", flush=True)
            with arm_cell(
                seed,
                config_slice={"arm": arm,
                              "use_defensive_orienting": arm == SCORED_ARM,
                              "eval_episodes": (EVAL_EPISODES_OVERRIDE if not dry_run
                                                else 2)},
                script_path=Path(__file__),
                config_slice_declared=True,
            ) as cell:
                seed_result = _run_arm_seed(arm, seed, dry_run=dry_run)
                episodes = seed_result.get("episodes", [])
                cell.stamp(seed_result)
            counters = seed_result.get("orienting_tick_counters") or _new_cell_counters()
            per_arm_episodes[arm].extend(episodes)
            per_arm_pooled_steps[arm].extend(_flatten_steps(episodes))
            per_arm_cells[arm].append(counters)
            arm_results.append({
                "arm": arm, "seed": seed,
                "chan_std": seed_result.get("chan_std", {}),
                "harm_trained": seed_result.get("harm_trained"),
                "limb_damage_events": seed_result.get("limb_damage_events"),
                "external_hazard_events": seed_result.get("external_hazard_events"),
                "world_rule_shift_events": seed_result.get("world_rule_shift_events"),
                "orienting_tick_counters": counters,
                "arm_fingerprint": seed_result.get("arm_fingerprint"),
            })
            print(f"  [orienting-tap] arm={arm} seed={seed} "
                  f"select_calls={counters['n_select_calls']} "
                  f"fresh={counters['n_fresh_orienting_ticks']} "
                  f"latched={counters['n_latched_ticks']} "
                  f"override_ticks={counters['n_override_ticks']} "
                  f"decisions={counters['decision_counts']}", flush=True)
            print(f"verdict: PASS seed={seed} arm={arm} "
                  f"(cell ran to completion -- see aggregate criteria below)", flush=True)
            print(f"  [progress] run {run_idx}/{total_runs} complete", flush=True)

    on_steps = per_arm_pooled_steps[SCORED_ARM]
    scored = _pool_counters(per_arm_cells[SCORED_ARM])
    control = _pool_counters(per_arm_cells[CONTROL_ARM])
    legacy = _legacy_per_env_step_readout(on_steps)

    dec_counts = scored["decision_counts"]
    dec_sum = sum(dec_counts.values())
    n_override_ticks = scored["n_override_ticks"]

    # --- Readiness preconditions. Computed on the SCORED arm only: the control arm has no
    # defensive_orienting gate, so an override-count precondition is structurally
    # unsatisfiable there (disposition (b) -- the ARM is scoped out of scoring, not the
    # precondition scoped out of the design). ---
    preconditions = [
        {
            "name": "fresh_orienting_ticks_sufficient",
            "description": (
                "Genuine (non-latched) orienting ticks observed by the sentinel tap on "
                "the ON arm. Below this floor the tap saw too little of the mechanism "
                "for any override-derived statistic to mean anything."
            ),
            "control": "ON arm, gate enabled at shipped SD-099 defaults",
            "measured": scored["n_fresh_orienting_ticks"],
            "threshold": MIN_FRESH_ORIENTING_TICKS,
            "direction": "lower",
            "met": scored["n_fresh_orienting_ticks"] >= MIN_FRESH_ORIENTING_TICKS,
        },
        {
            "name": "latch_separation_observed",
            "description": (
                "At least one LATCHED tick must be observed. Zero latched ticks would "
                "mean the freshness sentinel is reporting every env step as fresh -- the "
                "one tap failure mode that is not self-catching (see "
                "experiments/_lib/fresh_select.py), and the exact defect this run exists "
                "to remove. It would silently reproduce the 910-lineage inflation."
            ),
            "control": "ON arm; e3_steps_per_tick defaults to 10, so most ticks are held",
            "measured": scored["n_latched_ticks"],
            "threshold": MIN_LATCHED_TICKS,
            "direction": "lower",
            "met": scored["n_latched_ticks"] >= MIN_LATCHED_TICKS,
        },
        {
            "name": "pooled_override_ticks_sufficient",
            "description": (
                f"At least {MIN_OVERRIDE_TICKS} GENUINE Component-4/5 override ticks "
                "pooled across ON-arm seeds, counted at the override tick itself. This "
                "is the readiness gate for the decisive criterion. NOT comparable to "
                "V3-EXQ-910a's floor of 5, which was applied to the inflated "
                "per-env-step counter."
            ),
            "control": "ON arm, gate enabled; the control arm cannot fire an override",
            "measured": n_override_ticks,
            "threshold": MIN_OVERRIDE_TICKS,
            "direction": "lower",
            "met": n_override_ticks >= MIN_OVERRIDE_TICKS,
        },
    ]
    all_preconditions_met = all(p["met"] for p in preconditions)

    diagnostics = {
        "scored_arm": SCORED_ARM,
        "control_arm": CONTROL_ARM,
        "per_arm_gate": {
            "scored": [SCORED_ARM],
            "structurally_vacuous_acknowledged": [CONTROL_ARM],
            "scope_note": (
                "orienting_off has no defensive_orienting gate, so n_override_ticks == 0 "
                "is forced by construction. It is scoped OUT of criterion scoring "
                "(disposition (b)) and retained as a structural negative control; its "
                "zero is recorded below, not scored as a criterion it would pass "
                "vacuously. Preconditions above are ON-arm only for the same reason."
            ),
        },
        "orienting_tick_counters_on_arm": scored,
        "orienting_tick_counters_off_arm_control": control,
        "off_arm_control_zero_overrides": control["n_override_ticks"] == 0,
        "legacy_per_env_step_readout": legacy,
        "realized_inflation_ratio_overrides": _ratio(
            legacy["n_overrides_latched"], n_override_ticks),
        "realized_inflation_ratio_decisions": _ratio(
            legacy["decision_counts_latched_sum"], n_override_ticks),
        "realized_fresh_tick_fraction": _ratio(
            scored["n_fresh_orienting_ticks"], scored["n_select_calls"]),
        "eval_episodes_used": (EVAL_EPISODES_OVERRIDE if not dry_run else 2),
    }

    if not all_preconditions_met:
        interpretation = {
            "label": "substrate_not_ready_requeue",
            "preconditions": preconditions,
            "criteria_non_degenerate": {},
            "diagnostics": diagnostics,
        }
        outcome = "FAIL"
        criteria = []
        criteria_non_degenerate: Dict[str, bool] = {}
    else:
        c1_pass = (dec_sum == n_override_ticks
                   and scored["n_override_ticks_unclassified_decision"] == 0)
        c2_pass = sum(1 for v in dec_counts.values() if v > 0) >= 2
        criteria = [
            {
                "name": "C_decision_counts_bounded_by_override_ticks",
                "load_bearing": True,
                "passed": bool(c1_pass),
                "measured_decision_counts_sum": dec_sum,
                "measured_n_override_ticks": n_override_ticks,
                "n_unclassified": scored["n_override_ticks_unclassified_decision"],
                "note": (
                    "The DRIVER FIX's own acceptance test, pre-registered: the corrected "
                    "counter must yield EXACTLY one classification per genuine override "
                    "tick. The legacy per-env-step readout for this same run is at "
                    "diagnostics.legacy_per_env_step_readout, with the realized "
                    "inflation ratios."
                ),
            },
            {
                "name": "C_decision_alignment_non_degenerate",
                "load_bearing": True,
                "passed": bool(c2_pass),
                "decision_counts": dec_counts,
                "n_override_ticks": n_override_ticks,
                "note": (
                    "The decisive criterion for MECH-489's valence-gating sub-claim: "
                    "PASSES if the resolved decision is NOT unconditionally one class "
                    "(V3-EXQ-910's bug signature was 0 approach / 0 resume / 206 "
                    "withdraw, on the now-superseded inflated counter)."
                ),
            },
        ]
        criteria_non_degenerate = {
            # C1 would pass vacuously at 0 == 0.
            "C_decision_counts_bounded_by_override_ticks": n_override_ticks > 0,
            "C_decision_alignment_non_degenerate": n_override_ticks >= MIN_OVERRIDE_TICKS,
        }
        vacuous = any(
            c["passed"] and not criteria_non_degenerate.get(c["name"], True)
            for c in criteria
        )
        all_criteria_pass = all(c["passed"] for c in criteria)
        outcome = "PASS" if (all_criteria_pass and not vacuous) else "FAIL"
        if not c1_pass:
            label = "decision_counter_still_inflated"
        elif c2_pass:
            label = "orienting_valence_gating_non_degenerate"
        else:
            label = "orienting_valence_gating_collapsed_on_corrected_counter"
        interpretation = {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "combination_rule": (
                "AND over both load-bearing criteria (C1 instrument-correctness, C2 "
                "valence non-degeneracy); a vacuous pass on either forces FAIL."
            ),
            "diagnostics": diagnostics,
            "context_only_not_pass_criteria": {
                "note": (
                    "The trigger-alignment sub-claim (Components 1-3) is NOT re-tested by "
                    "this run -- see SCOPE. These are reported for continuity with "
                    "V3-EXQ-910/910a only, and are computed on the LATCHED per-step log "
                    "exactly as those runs did, so they remain directly comparable to "
                    "them and are NOT corrected."
                ),
                "event_trigger_alignment": _event_trigger_alignment(on_steps),
                "on_arm_coupling_latched": _coupling_stats(
                    per_arm_episodes[SCORED_ARM],
                    lambda s: bool(s.get("orienting_trigger_fired")),
                ),
                "off_arm_coupling_legacy_proxy": _coupling_stats(
                    per_arm_episodes[CONTROL_ARM],
                    lambda s: ((s.get("residue_surprise") or 0.0)
                               > LEGACY_SURPRISE_SPIKE_THRESHOLD),
                ),
            },
        }

    metrics: Dict[str, Any] = {
        "n_seeds": len(seeds),
        "arms": arms,
        "n_override_ticks": n_override_ticks,
        "decision_counts": dec_counts,
        "decision_counts_sum": dec_sum,
        "n_fresh_orienting_ticks": scored["n_fresh_orienting_ticks"],
        "n_latched_ticks": scored["n_latched_ticks"],
        "n_trigger_ticks": scored["n_trigger_ticks"],
        "legacy_n_overrides_latched": legacy["n_overrides_latched"],
        "legacy_decision_counts_latched_sum": legacy["decision_counts_latched_sum"],
        "realized_inflation_ratio_overrides":
            diagnostics["realized_inflation_ratio_overrides"],
        "realized_inflation_ratio_decisions":
            diagnostics["realized_inflation_ratio_decisions"],
        "off_arm_control_n_override_ticks": control["n_override_ticks"],
    }

    summary_markdown = (
        f"# V3-EXQ-910b -- MECH-489 (SD-099) valence-gating retest, decision counted at "
        f"the override tick\n\n"
        f"Seeds: {seeds}. Arms: {arms}. Scored arm: {SCORED_ARM} "
        f"(eval_episodes={diagnostics['eval_episodes_used']}).\n\n"
        f"## Corrected readout (at the override tick, synchronously)\n"
        f"- n_override_ticks: {n_override_ticks}\n"
        f"- decision_counts: {dec_counts} (sum {dec_sum})\n"
        f"- fresh orienting ticks: {scored['n_fresh_orienting_ticks']}; "
        f"latched: {scored['n_latched_ticks']}\n\n"
        f"## Legacy per-env-step readout (the defect this run fixes), same run\n"
        f"- n_overrides_latched: {legacy['n_overrides_latched']}\n"
        f"- decision_counts_latched_sum: {legacy['decision_counts_latched_sum']}\n"
        f"- realized inflation ratio (overrides): "
        f"{diagnostics['realized_inflation_ratio_overrides']}\n"
        f"- realized inflation ratio (decisions): "
        f"{diagnostics['realized_inflation_ratio_decisions']}\n\n"
        f"## Structural negative control ({CONTROL_ARM})\n"
        f"- n_override_ticks: {control['n_override_ticks']} (must be 0 by construction)\n\n"
        f"## Interpretation label: {interpretation['label']}\n"
        f"## Outcome: {outcome}\n"
    )

    return {
        "status": outcome, "outcome": outcome, "metrics": metrics,
        "summary_markdown": summary_markdown, "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": "supports" if outcome == "PASS" else "weakens",
        "experiment_type": EXPERIMENT_TYPE, "interpretation": interpretation,
        "arm_results": arm_results, "supersedes": SUPERSEDES,
        "seeds": seeds,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result = run(seeds=args.seeds, dry_run=args.dry_run)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["timestamp_utc"] = ts
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    out_dir = (Path(__file__).resolve().parents[2]
               / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = write_flat_manifest(
        result,
        out_dir.parent,
        dry_run=args.dry_run,
        config={"note": "see arm_results[].arm_fingerprint for per-cell config_slice",
                "eval_episodes_override": EVAL_EPISODES_OVERRIDE,
                "min_override_ticks": MIN_OVERRIDE_TICKS},
        seeds=result.get("seeds") or (args.seeds or SEEDS),
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    print(f"final_outcome: {result['outcome']}", flush=True)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    emit_outcome(outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
                 manifest_path=out_path,
                 dry_run=bool(args.dry_run))
