#!/opt/local/bin/python3
"""V3-EXQ-937 -- ARC-070 / MECH-321 PREDICTION-FAILURE SELECTIVITY, RATE-MATCHED
YOKED CONTROL, UNCONDITIONAL WHOLE-EPISODE DV.

THE QUESTION, RESTATED SO A RUN CAN REACH A VERDICT EITHER WAY

    Does decomposition placed at high-forward-prediction-error loci produce a
    better whole-episode outcome than the SAME AMOUNT of decomposition placed
    elsewhere?

ARC-070 asserts: when a chunked primitive's predicted outcome is unreliable,
re-segment it into finer primitives. The load-bearing word is WHEN. That is a
SELECTIVITY claim -- decomposition AT prediction-failure loci specifically is
better than the alternative -- NOT a claim that decomposition is good. So the
load-bearing contrast here is ARM_PE vs ARM_YOKED. ARM_OFF is retained as a
manipulation check, NOT as the scientific comparison.

Full design: REE_assembly/evidence/planning/govdiag1_repose_mech321_chain_
2026-08-12.md section 5. This script is the build/queue step per that doc's
section 8 hand-off (chip-20260814-mech321-pe-selectivity-repose). That document
promotes nothing and queues nothing; it re-poses the question and refuses
several re-queues (section 6, honoured below).

WHY A RE-POSE AND NOT ANOTHER LETTER OF 816 (design doc sections 1, 2)

Six prior runs on this chain (816b, 816c, 816d, 830 x2, 839) reached NO
verdict, and it was ONE failure, not five: every one died at a trigger-
OCCUPANCY gate, and in five of the six the load-bearing DV was itself
CONDITIONAL on that occupancy. A subset-conditional DV is UNDEFINED on the
empty subset, so "no data in the low-V_s subset" is equally consistent with
NO EFFECT (a real answer) and NO OCCASION (not an answer at all). Those two
worlds were aliased into one `non_contributory` reading every time. Because
occupancy was treated as a precondition to be HOPED FOR, the natural next move
after each non-reading was to escalate the environment: 816 -> 816b -> 816d
moved off_pe_mean_worst 0.0080 -> 0.008594 -> 0.008675 (under 0.0007 total,
each step within noise of the last, never reaching the 0.01 floor) with
low_vs_steps = 0 at EVERY dose.

Two design choices made that loop self-sustaining, and this driver fixes both:

  (1) THE TRIGGER THRESHOLD WAS ABSOLUTE, on a readout whose absolute scale is
      a property of how well-trained the agent is. A competent agent has low PE
      by definition, so the better the agent works the less the instrument can
      see -- there is no environment dose that fixes an instrument whose
      sensitivity is inversely coupled to the system's competence. FIXED HERE by
      the RANK-BASED trigger (see TRIGGER below).
  (2) THE READOUT WAS A PROXY ALREADY MEASURED AS SATURATED AND DECOUPLED from
      the construct it stands in for. V3-EXQ-816c measured region-V_s at
      region_vs_min_over_cells 0.9338, var_best 0.000275, total_low_vs_steps
      0 of 1654, and spearman(V_s, forward-PE) = 0.0832 against a coupling floor
      of 0.2, while forward-PE in the SAME cells was genuinely heterogeneous
      (pe_heterogeneous true, pe_var_best 8.64e-7 vs a 1e-9 floor) with
      vs_tracking_live true (NOT the degenerate constant-1.0 fallback). That leg
      (H-vs-proxy-saturation) is now CONFIRMED in
      hypothesis_space_registry.v1.json (REE_assembly a843ee6ebb, 2026-08-18).
      FIXED HERE by keying the trigger on forward-model PE, the construct the
      claim actually names.

GATING PRECONDITION, DISCHARGED. This driver was authored only after the
section 7 ledger resolutions landed: H-vs-proxy-saturation -> `confirmed`, and
H-env-underdrives-uncertainty ratified SUPERSEDED/moot (its registry `state`
stays the string `alive` ONLY because the schema's derived-state vocabulary has
no `superseded` member -- the basis field says so explicitly and says "READ
THIS LEG AS MOOT, NOT AS OPEN"). The re-pose therefore probes no formally-live
leg.

TRIGGER -- RANK-BASED WITHIN THE RUN, NOT ABSOLUTE (design doc section 5b)

Fire on the top-q% of rollout steps by forward-model PE, ranked WITHIN THE RUN,
with q = PE_QUANTILE_Q a pre-declared design constant. THIS IS THE SINGLE
CHANGE THAT RETIRES THE ENVIRONMENT LADDER rather than escalating it:
occupancy stops being an OUTCOME of the run and becomes a PARAMETER of the
design. A quantile trigger fires on the top q% whatever the absolute PE scale
is, so the "no occasion" branch is eliminated BY CONSTRUCTION and no
environment harshening is required. It also removes the perverse coupling in
(1) above -- a better-trained agent no longer becomes a less measurable one.

That the signal exists at usable resolution is MEASURED, not assumed. 816c
found pe_heterogeneous true in exactly the cells where V_s was flat, and a
direct probe on THIS config (2 episodes x 60 steps, seed 11, 2026-08-18) found
forward-PE spanning 0.004973 to 0.014042 with variance 2.91e-06 -- genuine
dynamic range. Only the absolute-floor framing (pe_elevated_floor = 0.01) ever
made it look unusable.

PE LAG = 1 TICK, AND THAT IS THE HONEST CAUSAL FORM. `e3_prediction_error` is
computed in `E3Selector.post_action_update` (ree_core/predictors/e3_selector.py
~3713: `prediction_error = actual_z_world - predicted_world`, reported as its
squared mean), which by construction runs AFTER the action is taken. So the
most recent PE available at DECISION time is the previous tick's. This driver
uses it deliberately rather than reaching for an unavailable same-tick value:
"decomposition placed at high-prediction-error loci" is implementable online
only as "placed at loci FOLLOWING high prediction error", and any same-tick
formulation would be an oracle the agent could not have.

CALIBRATION PREFIX. The first CALIBRATION_TICKS ticks of each run collect PE
without ever firing, so the within-run quantile has a population before it is
consulted. Thereafter the cut is recomputed each tick over ALL PE seen so far
(an expanding window, which tracks drift as the agent trains -- a fixed cut
estimated once would drift out of calibration exactly as the agent improves,
reintroducing defect (1) in slower form).

ARMS -- THE DISCRIMINATING COMPARISON (design doc section 5c)

  ARM_OFF    use_policy_decomposition=False. Structural zero, as in
             V3-EXQ-904's OFF arm. MANIPULATION CHECK ONLY.
  ARM_PE     decomposition ON; region_vs injected LOW (fire) at the top-q%
             lagged-forward-PE loci, HIGH (silent) elsewhere.
  ARM_YOKED  decomposition ON; region_vs injected LOW at the SAME NUMBER of
             loci PER EPISODE as ARM_PE realised FOR THAT SAME SEED, drawn
             uniformly at random over the episode's ticks (PE-uninformative)
             from a dedicated seeded RNG. Same depth cap, same everything else.

**THE LOAD-BEARING CONTRAST IS ARM_PE vs ARM_YOKED.** Rate-matching is what
isolates SELECTIVITY from DECOMPOSITION PER SE; comparing against OFF cannot
separate them, which is why every prior generation's OFF-vs-ON contrast could
not have answered ARC-070's actual question even had it reached a verdict.

None of the six chain runs had a rate-matched control. V3-EXQ-820's ARM_2 was
a different TRIGGER (R5 bottleneck) firing at its own uncontrolled rate -- a
different-mechanism comparison, not a rate control -- and its R1 side never
fired, so the contrast was vacuous in any case. This is a genuinely new
discriminator, not a re-run.

YOKING IS ON FORCED LOCI, AND THE REALISED RATE IS THEN CHECKED. ARM_YOKED is
given ARM_PE's per-episode forced-locus COUNT exactly, by construction. What it
cannot be given by construction is an identical REALISED decomposition count,
for two substrate reasons: the R1 trigger is an OR (`v_s < threshold OR
boundary.fired`, policy_decomposition.py ~535) so the live MECH-288 boundary
detector contributes fires in BOTH arms; and a forced locus only reaches
`evaluate()` on ticks where a chunk candidate is actually under consideration.
Both effects are present in both arms and neither is keyed to PE, so they are
background, not confound -- but the realised rate is therefore a MEASURED
quantity and is checked against RATE_MATCH_TOL rather than assumed. Probe
(seed 11, 2 x 60): V_s silent throughout gave 33 boundary fires -> 26 of 278
precommit evaluations decomposing, i.e. the boundary background is a real but
MINORITY contributor that does not swamp the forced manipulation (V_s forced at
every 5th tick moved decomp_n_vs_trigger 0 -> 13 and decomposed_precommit
26 -> 34; forced everywhere gave 62).

ARM_BOUNDARY IS DELIBERATELY NOT BUILT -- decision recorded, not deferred
silently. The design doc offers it as OPTIONAL, to make H-r1-r5-dissociable
answerable as a by-product. It is declined here because on THIS substrate the
MECH-288 boundary detector is ALREADY LIVE in every arm (26 decompositions from
boundary alone in the probe above), so a "rate-matched boundary arm" would have
to SUPPRESS the natural boundary path and re-place it -- a second, invasive
manipulation whose validity would need its own controls, inside a run whose
load-bearing contrast does not need it. Building it would also raise the grid
from 132 to 172 cells (~+10h) for a contrast the design doc itself does not
call load-bearing. INSTEAD: `decomp_n_boundary_fires` is recorded PER CELL in
every arm, so the boundary path's behaviour under PE-selected vs PE-
uninformative decomposition is in the manifest as descriptive data for whoever
next takes up H-r1-r5-dissociable. That leg is NOT claimed to be answered here.

DV AND POWER -- 919'S SHAPE, VERBATIM IN STRUCTURE (design doc section 5d)

UNCONDITIONAL whole-episode mean harm signal over ALL measured seeds: no
screen, no tiering, no post-hoc divergence-tick windowing, no exclusion. n >= 40
paired seeds as a PRE-REGISTERED HARD FLOOR (MIN_SEEDS), which can never be
softened by any observed quantity because no unit is ever excluded by this
design. Declared secondaries, REPORTED AND NON-GATING: mean per-episode return
(summed harm), mean done-events per episode (health-depletion / step-cap
terminations -- episode LENGTH is constant by construction, see below),
terminal health/energy, and forward-PE.

FIXED-LENGTH ROLLOUTS, AND WHY `done` IS NOT A BREAK. Every cell runs the full
`steps` ticks of every episode and does NOT break on the env's `done` flag
(`done = _health_depleted or _step_cap_reached`, causal_grid_world.py ~3123).
This is the 844/867/867a/867b/919 lineage convention -- 919 names it `_done`
and ignores it -- and it is load-bearing twice over: (a) `mean_harm_signal` is
a whole-episode harm RATE, so truncating at health depletion would silently
drop exactly the most harmful ticks and make the DV a survival-biased
estimate, and it would not be cross-readable against 919's; (b) it guarantees
every ARM_YOKED locus drawn from `range(steps)` actually occurs, so the
rate-matched count is delivered EXACTLY rather than under-delivered whenever
an episode happened to end early. Terminations are instead COUNTED and
reported as `mean_done_events_per_episode`.

This shape is not proposed on theory. V3-EXQ-919 ran it on this substrate and
reached a decisive reading where four prior generations had not (C1 measured
-0.0037281, non_degenerate true, n=40, A-A control max_abs_delta 0.0).

OCCUPANCY IS A MANIPULATION CHECK, NEVER A GATE (design doc section 5e)

Fires-per-episode is REPORTED per arm and per seed. Readiness is only the
trivial existential: ARM_PE forced fires > 0, ARM_OFF decomposition EXACTLY 0,
and the ARM-LEVEL realised rates matched within RATE_MATCH_TOL. The rate gate
is ARM-level because that is the design doc's literal quantity (section 5e,
`|rate(ARM_PE) - rate(ARM_YOKED)|`); gating a per-seed conjunction instead
would let ONE outlier seed void the whole run over sampling noise, which is
the wrong failure mode for a claim about the amount of decomposition each arm
carried IN AGGREGATE. Per-seed gaps and the count outside tolerance are
reported regardless, so an aggregate match hiding a bimodal per-seed
distribution stays visible. **There is NO
`vs_heterogeneity_low_vs_steps_present` gate and NO absolute `low_vs_steps >= N`
precondition anywhere in this file** -- that precondition IS the aliasing device
that produced the six-run chain, and carrying it forward would reproduce it.

PRE-DECLARED NULL (design doc section 5f). ARM_PE - ARM_YOKED whole-episode
harm delta <= 0 (within EFFECT_SIZE_K_SIGMA x SE over >= MIN_SEEDS paired
seeds) -> ARC-070's PREDICTION-FAILURE-SELECTIVITY LEG IS REFUTED AT THIS GRAIN.
That null is reachable regardless of what the environment does, which is the
entire point of the re-pose. BOTH DIRECTIONS ARE VERDICTS.

A-A NULL CONTROL, ON ARM_PE (a deliberate strengthening over 919, which ran its
control on the inert OFF arm). Each control seed is run THREE times as ARM_PE,
all through `arm_cell` (full RNG reset). ARM_PE exercises the entire NEW
instrumentation this design introduces -- the PE history, the expanding-window
quantile, and the injected-region_vs actuator -- so it is precisely the path
whose determinism needs proving; OFF-vs-OFF would have exercised none of it.
All three replicates must be mutually bit-identical (pairwise equal action
sequences, delta EXACTLY 0.0). Any nonzero delta on ANY control seed VOIDS the
run: non_degenerate false, no C1 reading, label
`aa_control_uncontrolled_variation_run_void`.

REFUSED, AND NOT BUILT (design doc section 6). No V3-EXQ-816e or any fourth
environment-axis escalation of the 816 design. No driver keyed on region-V_s as
the PREDICTION-FAILURE READOUT. No H-algorithm-axis probe as pre-registered
(lowering the absolute V_s threshold keeps the dead proxy). NOTE THE
DISTINCTION, because it is the one thing about this file most likely to be
misread: `region_vs` is used here ONLY as the substrate's trigger ACTUATOR --
`PolicyDecomposition.evaluate()` takes it as a caller-supplied float and
`HippocampalModule._region_vs()` is its sole source, so overriding that method
is how ANY trigger is placed at a chosen locus (V3-EXQ-904 established the
technique). The READOUT that decides WHERE to place it is forward-model PE. The
refusal is on region-V_s as the readout, and that refusal is honoured: this
driver never reads the substrate's own region_vs value, never gates on it, and
never reports it as evidence.

DV-SYMMETRY DECLARATION (mandatory per-arm, /queue-experiment Step 3). DV =
mean per-tick environment harm signal over the WHOLE run -- a set-aggregate
whose symmetry group is permutation of the ticks it averages over.
  ARM_PE     NOT invariant: forcing the R1 trigger at a locus re-tiles the
             committed chunk into finer primitives and releases the commit
             latch, changing WHICH actions are taken from that tick on, and
             hence which harm values are observed at each subsequent tick.
  ARM_YOKED  NOT invariant, by the identical mechanism at different loci.
  ARM_OFF    carries no manipulation, so its per-tick harm sequence is the
             no-decomposition baseline by construction, never a symmetry-fixed
             artifact of the DV itself.
  PE vs YOKED (the load-bearing delta) is NOT a broadcast additive constant, NOT
             a monotone rescaling, and NOT a permutation of interchangeable
             units: the two arms place the SAME NUMBER of decompositions at
             DIFFERENT loci, and since a decomposition changes the action
             sequence from its locus onward, the two harm sequences differ in
             content and not merely in labelling. The delta is therefore a
             measurement, not an arithmetic identity fixed before the run.

GOV-REUSE-1 (Step 2.4). Decisive readout = paired per-seed unconditional
whole-episode `mean_harm_signal` delta between ARM_PE and ARM_YOKED. Neither
arm has ever existed: no manifest in the corpus carries a rank-based
forward-PE-triggered decomposition arm or a rate-matched yoked decomposition
control (`grep -rlE "ARM_YOKED|ARM_PE\b" ree-v3/experiments/` is EMPTY as of
2026-08-18). The readout cannot be derived by reprocessing banked cells because
no banked cell was produced under this manipulation -- the manipulation IS the
question. Not recoverable -> run. `try_reuse_cell` is OMITTED rather than left
as dead code: there is no prior mint of an ARM_PE/ARM_YOKED cell to cite.
ARM_OFF cells ARE minted reuse-eligible (`include_driver_script_in_hash=False`)
as the canonical baseline for this lineage.

RE-DERIVE BRAKE (Step 2.5b). Does not fire. Measured 2026-08-18: ARC-070
counts 0 braking autopsies, MECH-321 counts 1 (failure_autopsy_V3-EXQ-867_
2026-08-02), both below the threshold of 2. Independently, this is a REDESIGN
testing a DIFFERENT mechanism under a NEW EXQ NUMBER (prediction-failure
SELECTIVITY, rate-matched), not a lettered iteration of a braked design.

SLEEP DRIVER: not applicable -- no sleep flags set, no sleep phase entered.

Z_GOAL: deliberately inert, carried over verbatim from the 844/867/919 lineage
for the identical reason (`REEConfig.from_dims(z_goal_enabled=...)` defaults
False). The stream stats are recorded regardless.
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from experiments._lib.stats import spearman  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
import experiments._lib.baselines.sd084_midexec_reachability as baselines  # noqa: E402

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.policy import ChunkedPrimitive, ChunkState  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_937_arc070_mech321_pe_selectivity_yoked_wholeepisode"
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["ARC-070", "MECH-321"]
QUEUE_ID = "V3-EXQ-937"
# `supersedes` deliberately OMITTED. The 816/830/839 chain is NOT superseded --
# it is re-posed. Those runs' instrumentation findings (816c's decoupling
# measurement above all) retain standalone value and are cited, not replaced.

ARM_OFF = "ARM_OFF"
ARM_PE = "ARM_PE"
ARM_YOKED = "ARM_YOKED"
ARMS = (ARM_OFF, ARM_PE, ARM_YOKED)

# --- Injected region_vs actuator levels. baselines.DECOMPOSITION_VS_THRESHOLD
# is 0.99 and the substrate's test is `region_vs < threshold`, so the sign
# convention is the opposite of the naive reading: a HIGH threshold means
# near-certain decomposition. VS_SILENT must therefore be >= 0.99, not merely
# "large". Probe-verified 2026-08-18: VS_SILENT gives decomp_n_vs_trigger == 0. ---
VS_FIRE = 0.1
VS_SILENT = 1.0

# --- Pre-declared design constants. Constants, never derived from this run's
# own statistics. ---
PE_QUANTILE_Q = 0.20          # fire on the top 20% of post-calibration ticks
CALIBRATION_TICKS = 60        # == one episode; collect PE, never fire
YOKED_RNG_STREAM_OFFSET = 90_001  # derives ARM_YOKED's locus RNG from the seed

# --- Pre-registered thresholds. ---
PE_VARIANCE_FLOOR = 1e-12     # positive control: the ranking signal has range
PE_SANITY_CEIL = 1e6          # positive control: forward model has not diverged
FORCED_FIRE_FLOOR = 0.0       # ARM_PE / ARM_YOKED per-cell: must exceed this
DECOMP_INERT_CEIL = 1.0       # ARM_OFF per-cell: integer count, < 1.0 means == 0
RATE_MATCH_TOL = 0.25         # |rate(PE) - rate(YOKED)| / mean(rate), per seed

# --- C1's bar. Carried verbatim from 919 (which carried it from 844/867/867a/
# 867b). Do NOT move these: this re-pose repairs the QUESTION and the CONTROL,
# not the criterion. ---
EFFECT_SIZE_K_SIGMA = 1.0
REL_IMPROVEMENT_FLOOR = 0.0

# --- Hard seed-count floor. Never softened by any observed quantity: no unit is
# ever excluded from this design (design doc section 5d). ---
MIN_SEEDS = 40

# Pre-registered measurement seeds -- the same 40 V3-EXQ-919 used, taken
# verbatim and in the same order, so this run's harm readings are directly
# cross-readable against 919's on the same substrate family. The list already
# excludes seed 44 (CLAUDE.md's recurring per-seed early-episode-death
# instability on this env family).
MEASUREMENT_SEEDS: Tuple[int, ...] = (
    11, 23, 47, 71, 3, 29, 89, 97, 17, 53,
    5, 7, 13, 19, 31, 37, 41, 43, 59, 61,
    67, 73, 79, 83, 101, 103, 107, 109, 113, 127,
    2, 6, 8, 12, 14, 18, 22, 26, 33, 39,
)
assert len(MEASUREMENT_SEEDS) == MIN_SEEDS
assert len(set(MEASUREMENT_SEEDS)) == MIN_SEEDS

# A-A null-control seeds, disjoint from the measurement set (919's control set).
AA_CONTROL_SEEDS: Tuple[int, ...] = (51, 57, 63, 69)
AA_REPLICATES = 3
assert not (set(AA_CONTROL_SEEDS) & set(MEASUREMENT_SEEDS))

_ZGOAL = ZGoalStreamAccumulator()


# ---------------------------------------------------------------------------
# The trigger controller -- the one genuinely new instrument in this driver.
# ---------------------------------------------------------------------------
def _quantile_cut(values: Sequence[float], q: float) -> float:
    """Upper-tail cut: the value at rank (1 - q) by nearest-rank, so that
    roughly the top `q` fraction of `values` compares >= the returned cut.

    Returns +inf on an empty population, which makes `fire` False -- the
    correct degenerate behaviour (never fire on no evidence)."""
    if not values:
        return float("inf")
    ordered = sorted(values)
    rank = int(math.ceil((1.0 - q) * len(ordered)))
    idx = min(len(ordered) - 1, max(0, rank - 1))
    return float(ordered[idx])


class _TriggerController:
    """Decides, ONCE PER AGENT TICK, whether the injected region_vs fires.

    Injection is via `HippocampalModule._region_vs`, which is that module's
    SOLE region_vs source (module.py ~840) and takes no arguments. It is called
    once per `_evaluate_decomposition_ticks` invocation, i.e. potentially
    several times within one agent tick (once per chunk candidate under
    consideration). The decision is therefore computed ONCE PER TICK by
    `decide_tick()` and merely READ by the override, so every candidate within
    a tick sees the same locus decision -- which is what makes "locus" mean an
    agent tick rather than a candidate evaluation.

    ARM_PE     fires when the most recent forward-PE (lag 1; see module
               docstring) is at or above the expanding-window (1 - q) quantile
               of all PE observed so far this run, after CALIBRATION_TICKS.
    ARM_YOKED  fires at a per-episode set of tick indices drawn uniformly at
               random (PE-uninformative) from a dedicated seeded RNG, sized to
               ARM_PE's realised per-episode forced-fire counts for the SAME
               seed.
    """

    def __init__(self, arm_id: str, seed: int,
                 yoked_counts_per_episode: Optional[Sequence[int]] = None,
                 calibration_ticks: int = CALIBRATION_TICKS):
        self.arm_id = arm_id
        self.seed = int(seed)
        self.calibration_ticks = int(calibration_ticks)
        self.pe_history: List[float] = []
        self.fire_now = False
        self.forced_loci: List[Tuple[int, int]] = []   # (episode, tick)
        self.forced_per_episode: List[int] = []
        self._episode_forced = 0
        self._yoked_counts = (
            list(yoked_counts_per_episode)
            if yoked_counts_per_episode is not None else None)
        # A dedicated, seed-derived RNG so ARM_YOKED's loci are deterministic
        # per (seed, cell) WITHOUT consuming the global RNG stream that
        # arm_cell resets -- drawing from the global stream would make the
        # yoked draw depend on how much randomness the agent happened to
        # consume first, which is neither reproducible nor PE-uninformative.
        self._rng = random.Random(self.seed + YOKED_RNG_STREAM_OFFSET)
        self._episode_yoked_ticks: set = set()

    # -- episode lifecycle ---------------------------------------------------
    def start_episode(self, ep: int, steps: int) -> None:
        if ep > 0:
            self.forced_per_episode.append(self._episode_forced)
        self._episode_forced = 0
        self._episode_yoked_ticks = set()
        if self.arm_id == ARM_YOKED and self._yoked_counts is not None:
            want = int(self._yoked_counts[ep]) if ep < len(self._yoked_counts) else 0
            want = max(0, min(want, steps))
            if want:
                self._episode_yoked_ticks = set(
                    self._rng.sample(range(steps), want))

    def end_run(self) -> None:
        self.forced_per_episode.append(self._episode_forced)

    # -- per-tick decision ---------------------------------------------------
    def decide_tick(self, ep: int, t: int) -> None:
        fire = False
        if self.arm_id == ARM_PE:
            if len(self.pe_history) >= self.calibration_ticks:
                cut = _quantile_cut(self.pe_history, PE_QUANTILE_Q)
                fire = bool(self.pe_history[-1] >= cut)
        elif self.arm_id == ARM_YOKED:
            fire = t in self._episode_yoked_ticks
        self.fire_now = fire
        if fire:
            self._episode_forced += 1
            self.forced_loci.append((ep, t))

    def observe_pe(self, pe: Optional[float]) -> None:
        if pe is not None and math.isfinite(pe):
            self.pe_history.append(float(pe))

    # -- the injected actuator ----------------------------------------------
    def region_vs(self) -> float:
        return VS_FIRE if self.fire_now else VS_SILENT


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------
def _decomposition_on_flags() -> Dict[str, Any]:
    """Substrate stack with MECH-321 live. Identical in ARM_PE and ARM_YOKED --
    the ONLY difference between those two arms is WHERE the controller places
    the forced trigger, which is exactly the selectivity contrast."""
    flags = dict(baselines.substrate_stack_flags())
    flags["use_persistent_committed_program_handle"] = True
    flags.update(baselines.HAZARD_TUNED_STREAM_FLAGS)
    return flags


def _arm_flags(arm_id: str) -> Dict[str, Any]:
    flags = _decomposition_on_flags()
    if arm_id == ARM_OFF:
        # Structural zero, as in V3-EXQ-904's OFF arm. The substrate's
        # early-return then makes the chunk-candidate splice bit-identical to
        # pre-MECH-321 behaviour.
        flags["use_policy_decomposition"] = False
    return flags


def _config_slice(arm_id: str, episodes: int,
                  calibration_ticks: int = CALIBRATION_TICKS) -> Dict[str, Any]:
    """Fingerprint config slice. Declares ONLY what the cell's computation
    reads. `episodes` is an argument, not a module constant, so a --dry-run
    cell's reduced schedule gets a DISTINCT fingerprint rather than colliding
    with a real-run cell's."""
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
    slice_.update(_arm_flags(arm_id))
    if arm_id in (ARM_PE, ARM_YOKED):
        # The trigger placement policy is part of this cell's closure.
        slice_.update({
            "trigger_policy": arm_id,
            "pe_quantile_q": PE_QUANTILE_Q,
            "calibration_ticks": int(calibration_ticks),
            "vs_fire": VS_FIRE,
            "vs_silent": VS_SILENT,
        })
    if arm_id == ARM_YOKED:
        slice_["yoked_rng_stream_offset"] = YOKED_RNG_STREAM_OFFSET
    return slice_


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
    """Ordinary ARC-071 usage, done IDENTICALLY in every arm. It is a
    PRECONDITION of the measurement, not the manipulation: without a
    multi-action chunk in the library there is nothing composite to decompose
    and the DV would be a structural zero for a reason unrelated to ARC-070."""
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
              yoked_counts: Optional[Sequence[int]] = None,
              calibration_ticks: int = CALIBRATION_TICKS,
              quiet: bool = False) -> Dict[str, Any]:
    env, agent, flags = _build(seed, arm_id)
    _register_chunk(agent)
    world_dim = agent.config.latent.world_dim

    ctrl = _TriggerController(arm_id, seed, yoked_counts_per_episode=yoked_counts,
                              calibration_ticks=calibration_ticks)
    if arm_id in (ARM_PE, ARM_YOKED):
        # `_region_vs` is HippocampalModule's SOLE region_vs source, and takes
        # no arguments -- the same injection point V3-EXQ-904 established.
        agent.hippocampal._region_vs = ctrl.region_vs

    n_ticks = 0
    multi_action_commits = 0
    actions: List[int] = []
    forward_pe_ticks: List[Optional[float]] = []
    harm_ticks: List[float] = []
    e3_tick_flags: List[bool] = []
    episode_returns: List[float] = []
    episode_lengths: List[int] = []
    episode_done_events: List[int] = []
    terminal_health: List[float] = []
    terminal_energy: List[float] = []
    done_causes: Dict[str, int] = {}
    max_z_harm_a_norm = 0.0   # diagnostic only, never gating

    if not quiet:
        print(f"Seed {seed} Condition {arm_id}", flush=True)

    for ep in range(episodes):
        _, obs = env.reset()
        agent.reset()
        if not agent.policy_chunking.library.all_chunks():
            _register_chunk(agent)
        ctrl.start_episode(ep, steps)
        ep_return = 0.0
        ep_steps = 0
        n_done_events = 0
        last_info: Dict[str, Any] = {}

        for t in range(steps):
            # The locus decision for THIS tick, taken before any candidate is
            # evaluated so every candidate within the tick sees the same value.
            ctrl.decide_tick(ep, t)

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
                if len(meta.get("chunk_sequence", ())) > 1:
                    multi_action_commits += 1

            a_int = int(action.argmax(dim=-1).item())
            actions.append(a_int)
            _flat, harm, done, info, obs = env.step(a_int)
            harm_ticks.append(float(harm))
            ep_return += float(harm)
            ep_steps += 1
            last_info = info or {}
            metrics = agent.update_residue(harm)
            pe_raw = metrics.get("e3_prediction_error")
            if pe_raw is not None:
                pe = float(pe_raw.detach()) if torch.is_tensor(pe_raw) else float(pe_raw)
                pe_val = pe if math.isfinite(pe) else None
            else:
                pe_val = None
            forward_pe_ticks.append(pe_val)
            # The trigger reads the PREVIOUS tick's PE (lag 1) -- this append is
            # what makes the next tick's decide_tick() causal rather than an
            # oracle. See module docstring "PE LAG = 1 TICK".
            ctrl.observe_pe(pe_val)
            n_ticks += 1
            if done:
                n_done_events += 1

        episode_returns.append(ep_return)
        episode_lengths.append(ep_steps)
        episode_done_events.append(n_done_events)
        terminal_health.append(float(last_info.get("health", 0.0)))
        terminal_energy.append(float(last_info.get("energy", 0.0)))
        cause = str(last_info.get("done_cause", "") or "none")
        done_causes[cause] = done_causes.get(cause, 0) + 1

        if not quiet:
            print(
                f"  [train] rollout seed={seed} arm={arm_id} ep {ep + 1}/{episodes} "
                f"ticks={n_ticks} forced={ctrl._episode_forced} "
                f"multi_commits={multi_action_commits}",
                flush=True,
            )

    ctrl.end_run()
    _ZGOAL.observe(agent)

    state = agent.get_policy_decomposition_state()
    decomposed_total = (int(state.get("decomp_n_decomposed_precommit", 0))
                        + int(state.get("decomp_n_decomposed_midexec", 0)))
    row: Dict[str, Any] = {
        "arm_id": arm_id,
        "seed": int(seed),
        "n_ticks": n_ticks,
        "episodes": episodes,
        "steps_per_episode": steps,
        "multi_action_commits": multi_action_commits,
        # --- decomposition activity (manipulation check, NEVER a gate) ---
        "decomp_n_evaluated_midexec": int(state.get("decomp_n_evaluated_midexec", 0)),
        "decomp_n_decomposed_midexec": int(state.get("decomp_n_decomposed_midexec", 0)),
        "decomp_n_evaluated_precommit": int(state.get("decomp_n_evaluated_precommit", 0)),
        "decomp_n_decomposed_precommit": int(state.get("decomp_n_decomposed_precommit", 0)),
        "decomp_n_decomposed_total": decomposed_total,
        "decomp_n_marked_unreliable": int(state.get("decomp_n_marked_unreliable", 0)),
        "decomp_n_vs_trigger": int(state.get("decomp_n_vs_trigger", 0)),
        # Recorded in EVERY arm as descriptive data for H-r1-r5-dissociable --
        # see the ARM_BOUNDARY decision in the module docstring. Not claimed to
        # answer that leg.
        "decomp_n_boundary_fires": int(state.get("decomp_n_boundary_fires", 0)),
        "decomp_n_harm_bias_nonzero": int(state.get("decomp_n_harm_bias_nonzero", 0)),
        "decomp_n_harm_override_fires": int(state.get("decomp_n_harm_override_fires", 0)),
        # --- trigger-controller instrumentation ---
        "forced_fires_total": len(ctrl.forced_loci),
        "forced_fires_per_episode": list(ctrl.forced_per_episode),
        "forced_loci": [list(x) for x in ctrl.forced_loci],
        "forced_fire_rate_per_episode": (
            len(ctrl.forced_loci) / max(1, len(ctrl.forced_per_episode))),
        "decomposed_rate_per_episode": decomposed_total / max(1, episodes),
        "pe_history_len": len(ctrl.pe_history),
        "pe_cut_final": (
            _quantile_cut(ctrl.pe_history, PE_QUANTILE_Q)
            if len(ctrl.pe_history) >= ctrl.calibration_ticks else None),
        "calibration_ticks": int(ctrl.calibration_ticks),
        # --- DV inputs ---
        "mean_harm_signal": (statistics.fmean(harm_ticks) if harm_ticks else 0.0),
        "mean_episode_return": (
            statistics.fmean(episode_returns) if episode_returns else 0.0),
        "mean_episode_length": (
            statistics.fmean(episode_lengths) if episode_lengths else 0.0),
        "mean_done_events_per_episode": (
            statistics.fmean(episode_done_events) if episode_done_events else 0.0),
        "mean_terminal_health": (
            statistics.fmean(terminal_health) if terminal_health else 0.0),
        "mean_terminal_energy": (
            statistics.fmean(terminal_energy) if terminal_energy else 0.0),
        "done_causes": dict(done_causes),
        "per_episode_returns": list(episode_returns),
        "per_episode_lengths": list(episode_lengths),
        "per_episode_done_events": list(episode_done_events),
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
            statistics.pvariance([v for v in forward_pe_ticks if v is not None])
            if sum(1 for v in forward_pe_ticks if v is not None) > 1 else 0.0),
        "arm_flags": dict(flags),
    }

    # Pure completion check. This design's DV does NOT require decomposition to
    # have occurred, so cell_pass is deliberately not a decomposition-activity
    # signal -- that would smuggle an occupancy gate back in.
    cell_pass = bool(n_ticks > 0)
    row["cell_pass"] = cell_pass
    if not quiet:
        print(f"verdict: {'PASS' if cell_pass else 'FAIL'}", flush=True)
    return row


# ---------------------------------------------------------------------------
# Reducers
# ---------------------------------------------------------------------------
def _worst_min(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(min(vals)) if vals else 0.0


def _worst_max(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(max(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# Preconditions -- the TRIVIAL EXISTENTIAL ONLY (design doc section 5e).
# Deliberately absent: any low_vs_steps floor, any vs_heterogeneity gate, any
# absolute forward-PE floor. Those are the aliasing devices that produced the
# six-run chain.
# ---------------------------------------------------------------------------
def _arm_context(arm_id: str) -> Dict[str, Any]:
    return {"id": arm_id, "arm_id": arm_id,
            "decomposition_on": arm_id in (ARM_PE, ARM_YOKED)}


def _precondition_specs() -> List[PreconditionSpec]:
    return [
        PreconditionSpec(
            name="forced_trigger_engages_every_cell",
            description=(
                "The injected trigger actually fires on EVERY measurement "
                "cell of a decomposition-ON arm (forced_fires_total > 0). "
                "This is the TRIVIAL EXISTENTIAL the design permits -- it "
                "asserts the instrument is connected, NOT that any "
                "particular occupancy level was reached. Per-cell, not "
                "aggregate."
            ),
            control="WORST (minimum) cell of the arm",
            threshold=FORCED_FIRE_FLOOR,
            direction="lower",
            kind="readiness",
            applies_to=lambda ctx: ctx["decomposition_on"],
        ),
        PreconditionSpec(
            name="decomposition_inert_every_cell",
            description=(
                "ARM_OFF carries no decomposition at all: "
                "decomp_n_decomposed_total must be EXACTLY 0 on every OFF "
                "cell. This is the structural zero the DV-symmetry "
                "declaration requires of the arm carrying no manipulation."
            ),
            control="WORST (maximum) ARM_OFF cell",
            threshold=DECOMP_INERT_CEIL,
            direction="upper",
            kind="readiness",
            applies_to=lambda ctx: ctx["arm_id"] == ARM_OFF,
        ),
        PreconditionSpec(
            name="forward_pe_varies",
            description=(
                "Positive control on the RANKING SIGNAL ITSELF, and the "
                "same statistic the trigger routes on: the forward-PE "
                "population the within-run quantile is computed over must "
                "have non-zero variance. A degenerate (constant) PE stream "
                "would make the top-q% selection arbitrary rather than "
                "PE-informative, which would make ARM_PE a second yoked arm "
                "and the load-bearing contrast vacuous."
            ),
            control="WORST cell of the arm (min variance)",
            threshold=PE_VARIANCE_FLOOR,
            direction="lower",
            kind="readiness",
        ),
        PreconditionSpec(
            name="forward_pe_bounded",
            description=(
                "Positive control: forward PE is bounded (the online forward "
                "model has not diverged). Mirrors 816/839/844/867/919's "
                "identical precondition."
            ),
            control="WORST cell of the arm (max mean)",
            threshold=PE_SANITY_CEIL,
            direction="upper",
            kind="readiness",
        ),
    ]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def _analyse(rows: List[Dict[str, Any]],
             aa_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_arm = {a: [r for r in rows if r["arm_id"] == a] for a in ARMS}
    by_seed = {a: {r["seed"]: r for r in by_arm[a]} for a in ARMS}
    # The load-bearing contrast needs BOTH decomposition arms present.
    seeds = sorted(set(by_seed[ARM_PE]) & set(by_seed[ARM_YOKED]))

    specs = _precondition_specs()
    arm_contexts = {a: _arm_context(a) for a in ARMS}

    arm_gates = []
    for arm_id in ARMS:
        arm_rows = by_arm[arm_id]
        measured: Dict[str, float] = {
            "forward_pe_varies": _worst_min(arm_rows, "fwd_pe_all_var"),
            "forward_pe_bounded": _worst_max(arm_rows, "fwd_pe_all_mean"),
        }
        if arm_id in (ARM_PE, ARM_YOKED):
            measured["forced_trigger_engages_every_cell"] = _worst_min(
                arm_rows, "forced_fires_total")
        if arm_id == ARM_OFF:
            measured["decomposition_inert_every_cell"] = _worst_max(
                arm_rows, "decomp_n_decomposed_total")
        arm_gates.append(
            evaluate_arm_gate(arm_id, arm_contexts[arm_id], specs, measured=measured))
    aggregate = aggregate_arm_gates(arm_gates)

    # --- RATE MATCH. The manipulation check that makes ARM_YOKED a rate
    # control rather than merely a second treatment. Reported per seed; the
    # aggregate worst case is the gate. Occupancy LEVEL is never gated -- only
    # the MATCH between the two arms is, which is what "same amount of
    # decomposition placed elsewhere" means. ---
    rate_rows: List[Dict[str, Any]] = []
    for s in seeds:
        rp = float(by_seed[ARM_PE][s]["decomposed_rate_per_episode"])
        ry = float(by_seed[ARM_YOKED][s]["decomposed_rate_per_episode"])
        denom = (rp + ry) / 2.0
        rel = abs(rp - ry) / denom if denom > 0 else 0.0
        rate_rows.append({
            "seed": s,
            "pe_forced_per_episode": by_seed[ARM_PE][s]["forced_fire_rate_per_episode"],
            "yoked_forced_per_episode": by_seed[ARM_YOKED][s]["forced_fire_rate_per_episode"],
            "pe_decomposed_per_episode": rp,
            "yoked_decomposed_per_episode": ry,
            "rel_rate_gap": rel,
            "within_tol": bool(rel <= RATE_MATCH_TOL),
        })
    worst_rate_gap = max((r["rel_rate_gap"] for r in rate_rows), default=0.0)
    worst_rate_seed = next(
        (r["seed"] for r in rate_rows if r["rel_rate_gap"] == worst_rate_gap), None)
    n_seeds_outside_tol = sum(1 for r in rate_rows if not r["within_tol"])
    # THE GATE IS ARM-LEVEL, and that is the design doc's literal quantity:
    # section 5e reads `|rate(ARM_PE) - rate(ARM_YOKED)|` -- a comparison of the
    # two ARMS' rates, not a per-seed conjunction. Gating on all() over seeds
    # instead would let ONE outlier seed void a ~30h run, which is both harsher
    # than the design asks and the wrong failure mode: the claim under test is
    # that the two arms carried the same AMOUNT of decomposition in aggregate,
    # and a single seed's sampling noise does not falsify that. Per-seed gaps
    # are REPORTED (rate_rows, plus n_seeds_outside_tol) so an aggregate match
    # concealing a wildly bimodal per-seed distribution is still visible to a
    # reader and to any later autopsy.
    arm_rate_pe = statistics.fmean(
        [r["pe_decomposed_per_episode"] for r in rate_rows]) if rate_rows else 0.0
    arm_rate_yoked = statistics.fmean(
        [r["yoked_decomposed_per_episode"] for r in rate_rows]) if rate_rows else 0.0
    arm_denom = (arm_rate_pe + arm_rate_yoked) / 2.0
    arm_rate_gap = (abs(arm_rate_pe - arm_rate_yoked) / arm_denom) if arm_denom > 0 else 0.0
    rate_match_ok = bool(rate_rows) and arm_rate_gap <= RATE_MATCH_TOL
    rate_precondition = {
        "name": "pe_yoked_realised_rate_matched",
        "kind": "readiness",
        "description": (
            "ARM_YOKED is given ARM_PE's per-episode forced-locus count "
            "EXACTLY by construction; the REALISED decomposition rate is a "
            "measured quantity (the R1 trigger is an OR, so the live MECH-288 "
            "boundary path contributes in both arms, and a forced locus only "
            "reaches evaluate() where a chunk candidate is under "
            "consideration). This checks the two arms genuinely carry the "
            "SAME AMOUNT of decomposition, which is what makes the contrast a "
            "selectivity test rather than a dose comparison."),
        "control": (
            "ARM-LEVEL mean realised decomposition rate, ARM_PE vs ARM_YOKED "
            f"over {len(rate_rows)} paired seed(s); per-seed gaps reported "
            f"separately (worst seed {worst_rate_seed} at {worst_rate_gap:.4f}, "
            f"{n_seeds_outside_tol} seed(s) outside tolerance)"),
        "measured": arm_rate_gap,
        "threshold": RATE_MATCH_TOL,
        "direction": "upper",
        "met": bool(rate_match_ok),
        "offending_cell": worst_rate_seed,
    }

    # --- A-A null control on ARM_PE: all replicates mutually bit-identical. ---
    aa_by_seed: Dict[int, List[Dict[str, Any]]] = {}
    for r in aa_rows:
        aa_by_seed.setdefault(int(r["seed"]), []).append(r)

    aa_checks: List[Dict[str, Any]] = []
    for seed in sorted(aa_by_seed.keys()):
        reps = aa_by_seed[seed]
        if len(reps) < 2:
            aa_checks.append({
                "seed": seed, "n_replicates": len(reps), "max_abs_delta": None,
                "action_sequences_identical": False, "bit_identical": False,
                "note": f"expected >= 2 replicate cells, found {len(reps)}",
            })
            continue
        base = reps[0]
        deltas = [r["mean_harm_signal"] - base["mean_harm_signal"] for r in reps[1:]]
        identical = all(r["action_sequence"] == base["action_sequence"] for r in reps[1:])
        max_abs = max((abs(d) for d in deltas), default=0.0)
        aa_checks.append({
            "seed": seed,
            "n_replicates": len(reps),
            "max_abs_delta": max_abs,
            "action_sequences_identical": bool(identical),
            "bit_identical": bool(identical and max_abs == 0.0),
        })
    aa_control_ok = bool(aa_checks) and all(c["bit_identical"] for c in aa_checks)
    aa_deltas = [c["max_abs_delta"] for c in aa_checks if c["max_abs_delta"] is not None]
    aa_max_abs_delta = max(aa_deltas) if aa_deltas else None

    aa_precondition_entries = [
        {
            "name": f"aa_control_bit_identical_seed_{c['seed']}",
            "kind": "readiness",
            "description": (
                "A-A null control on ARM_PE (same seed, all replicates through "
                "arm_cell with full RNG reset) must be bit-identical: max "
                "|delta| EXACTLY 0.0 and every action sequence equal. Run on "
                "ARM_PE rather than the inert OFF arm because ARM_PE is the "
                "arm that exercises the new PE-history + expanding-window-"
                "quantile + injected-actuator instrumentation, so it is the "
                "path whose determinism actually needs proving."),
            "control": f"A-A replicate set, seed {c['seed']}",
            "measured": (c["max_abs_delta"]
                         if c["max_abs_delta"] is not None else float("nan")),
            "threshold_low": 0.0,
            "threshold_high": 0.0,
            "comparator_low": ">=",
            "comparator_high": "<=",
            "direction": "interval",
            "met": bool(c["bit_identical"]),
        }
        for c in aa_checks
    ]

    n_seeds = len(seeds)
    enough_seeds = n_seeds >= MIN_SEEDS
    non_degenerate = bool(
        aggregate["non_degenerate"] and enough_seeds and aa_control_ok and rate_match_ok)

    # Defaults -- overwritten only in the successful branch.
    harm_deltas: List[float] = []
    harm_delta_mean = 0.0
    harm_delta_sd = 0.0
    harm_delta_se = 0.0
    rel_improvement = 0.0
    effect_size_ok = False
    rel_floor_ok = False
    c1_pe_selectivity = False
    pe_vs_off_mean = 0.0
    decomposition_per_se_helps = False
    fwd_pe_delta_mean = 0.0
    ret_delta_mean = 0.0
    len_delta_mean = 0.0
    engagement_outcome_rho: Optional[float] = None

    if not aggregate["non_degenerate"]:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        direction = "unknown"
        degeneracy_reason = aggregate["degeneracy_reason"]
    elif not rate_match_ok:
        label = "rate_match_failed_yoked_control_invalid"
        outcome = "FAIL"
        direction = "unknown"
        degeneracy_reason = (
            "ARM_PE and ARM_YOKED did not carry the same realised amount of "
            "decomposition: ARM-LEVEL relative rate gap "
            f"{arm_rate_gap:.4f} > tolerance {RATE_MATCH_TOL} "
            f"(ARM_PE {arm_rate_pe:.3f} vs ARM_YOKED {arm_rate_yoked:.3f} "
            f"decompositions/episode; worst single seed {worst_rate_seed} at "
            f"{worst_rate_gap:.4f}, {n_seeds_outside_tol} seed(s) outside "
            "tolerance). Without a rate match the contrast measures "
            "decomposition DOSE, not SELECTIVITY, so no C1 reading is "
            "emitted. This is an instrument finding, NOT evidence about "
            "ARC-070.")
    elif not aa_control_ok:
        label = "aa_control_uncontrolled_variation_run_void"
        outcome = "FAIL"
        direction = "unknown"
        degeneracy_reason = (
            "A-A null control on ARM_PE (same seed, replicate cells) produced "
            "a nonzero delta and/or non-identical action sequences on at "
            "least one control seed -- the measurement path carries an "
            "uncontrolled source of variation, so no C1 reading is emitted. "
            f"max_abs_aa_delta={aa_max_abs_delta}")
    elif not enough_seeds:
        label = "insufficient_measured_seed_count"
        outcome = "FAIL"
        direction = "unknown"
        degeneracy_reason = (
            f"only {n_seeds} seed(s) with BOTH decomposition arms present, "
            f"below the pre-registered hard floor of {MIN_SEEDS}. This floor "
            "is never softened by any observed quantity -- no unit is ever "
            "excluded from this design.")
    else:
        degeneracy_reason = None
        # THE LOAD-BEARING CONTRAST. Sign convention: harm is negative-is-worse
        # (env returns negative harm), so ARM_PE - ARM_YOKED > 0 means PE-placed
        # decomposition is LESS harmful, i.e. selectivity helps.
        harm_deltas = [
            by_seed[ARM_PE][s]["mean_harm_signal"]
            - by_seed[ARM_YOKED][s]["mean_harm_signal"]
            for s in seeds]
        harm_delta_mean = statistics.fmean(harm_deltas)
        harm_delta_sd = statistics.stdev(harm_deltas) if len(harm_deltas) > 1 else 0.0
        harm_delta_se = (
            harm_delta_sd / math.sqrt(len(harm_deltas)) if harm_deltas else 0.0)
        yoked_ref = statistics.fmean(
            by_seed[ARM_YOKED][s]["mean_harm_signal"] for s in seeds)
        rel_improvement = (
            (harm_delta_mean / abs(yoked_ref)) if yoked_ref not in (0.0, None) else 0.0)
        effect_size_ok = harm_delta_mean > EFFECT_SIZE_K_SIGMA * harm_delta_se
        rel_floor_ok = rel_improvement >= REL_IMPROVEMENT_FLOOR
        c1_pe_selectivity = bool(effect_size_ok and rel_floor_ok)

        # C2 -- decomposition PER SE, vs the structural zero. Non-load-bearing:
        # this is the comparison every prior generation made, retained only so
        # this run can be cross-read against them.
        off_seeds = [s for s in seeds if s in by_seed[ARM_OFF]]
        if off_seeds:
            pe_vs_off = [
                by_seed[ARM_PE][s]["mean_harm_signal"]
                - by_seed[ARM_OFF][s]["mean_harm_signal"] for s in off_seeds]
            pe_vs_off_mean = statistics.fmean(pe_vs_off)
            decomposition_per_se_helps = pe_vs_off_mean > 0.0

        # Declared secondaries -- reported, NEVER gating.
        pe_pairs = [
            (by_seed[ARM_PE][s]["fwd_pe_all_mean"],
             by_seed[ARM_YOKED][s]["fwd_pe_all_mean"]) for s in seeds]
        fwd_vals = [(y - p) for p, y in pe_pairs if p is not None and y is not None]
        fwd_pe_delta_mean = statistics.fmean(fwd_vals) if fwd_vals else 0.0
        ret_delta_mean = statistics.fmean([
            by_seed[ARM_PE][s]["mean_episode_return"]
            - by_seed[ARM_YOKED][s]["mean_episode_return"] for s in seeds])
        # Episode LENGTH is constant by construction (fixed-length rollouts),
        # so the informative termination secondary is the done-event RATE --
        # how often health depletion / step cap actually fired.
        len_delta_mean = statistics.fmean([
            by_seed[ARM_PE][s]["mean_done_events_per_episode"]
            - by_seed[ARM_YOKED][s]["mean_done_events_per_episode"] for s in seeds])

        engagement_outcome_rho = spearman(
            [float(by_seed[ARM_PE][s]["decomp_n_decomposed_total"]) for s in seeds],
            harm_deltas)

        if c1_pe_selectivity:
            label = "pe_selectivity_improves_whole_episode_outcome"
            direction = "supports"
        elif harm_delta_mean <= 0:
            # THE PRE-DECLARED NULL. This is a VERDICT, not a non-reading.
            label = "pe_selectivity_refuted_rate_matched_wholeepisode"
            direction = "weakens"
        else:
            label = "pe_selectivity_effect_below_threshold"
            direction = "mixed"
        outcome = "PASS" if c1_pe_selectivity else "FAIL"

    non_degen_map = {
        "C1_PE_SELECTIVITY_IMPROVES_OUTCOME": non_degenerate,
        "C2_DECOMPOSITION_PER_SE_VS_OFF": non_degenerate,
    }

    criteria = [
        {"name": "C1_PE_SELECTIVITY_IMPROVES_OUTCOME", "load_bearing": True,
         "passed": bool(c1_pe_selectivity),
         "measured": harm_delta_mean, "threshold": 0.0,
         "statement": (
             "LOAD-BEARING. Over ALL measured seeds (no screen, no tiering, "
             "no post-hoc selection, no occupancy gate), the unconditional "
             "whole-episode mean harm signal is LESS harmful under "
             "decomposition placed at top-"
             f"{int(PE_QUANTILE_Q * 100)}% forward-PE loci (ARM_PE) than "
             "under the SAME per-episode amount of decomposition placed at "
             "PE-uninformative loci (ARM_YOKED), by an effect exceeding "
             f"{EFFECT_SIZE_K_SIGMA} x SE over >= {MIN_SEEDS} paired seeds. "
             "PRE-DECLARED NULL: a delta <= 0 within that bound REFUTES "
             "ARC-070's prediction-failure-selectivity leg at this grain. "
             "Both directions are verdicts.")},
        {"name": "C2_DECOMPOSITION_PER_SE_VS_OFF", "load_bearing": False,
         "passed": bool(decomposition_per_se_helps),
         "measured": pe_vs_off_mean, "threshold": 0.0,
         "statement": (
             "NON-LOAD-BEARING context: ARM_PE vs the ARM_OFF structural "
             "zero -- decomposition PER SE, which is the comparison the six "
             "prior chain runs were posed on. Retained only for cross-"
             "reading; it cannot separate selectivity from decomposition "
             "per se, which is why it is not load-bearing here.")},
    ]

    combination_rule = (
        "PASS iff C1 alone passes (plain single-criterion gate, no OR/AND "
        "combination). C2 is contextual and never contributes to the outcome.")

    return {
        "outcome": outcome,
        "evidence_direction": direction,
        "interpretation_label": label,
        "criteria": criteria,
        "combination_rule": combination_rule,
        "criteria_non_degenerate": non_degen_map,
        "preconditions": (aggregate["adjudication_preconditions"]
                          + aa_precondition_entries + [rate_precondition]),
        "per_arm_gate": aggregate["per_arm_gate"],
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "aa_control": {
            "checks": aa_checks, "ok": aa_control_ok,
            "max_abs_delta": aa_max_abs_delta, "arm": ARM_PE,
            "replicates": AA_REPLICATES,
        },
        "rate_match": {
            "per_seed": rate_rows, "ok": rate_match_ok,
            "gate_level": "arm",
            "arm_rate_pe_per_episode": arm_rate_pe,
            "arm_rate_yoked_per_episode": arm_rate_yoked,
            "arm_rel_gap": arm_rate_gap,
            "worst_rel_gap": worst_rate_gap, "worst_seed": worst_rate_seed,
            "n_seeds_outside_tol": n_seeds_outside_tol,
            "tolerance": RATE_MATCH_TOL,
        },
        "per_seed_harm_deltas": [
            {"seed": s, "harm_delta_pe_minus_yoked": d}
            for s, d in zip(seeds, harm_deltas)] if harm_deltas else [],
        "summary": {
            "n_seeds": n_seeds,
            "min_seeds_required": MIN_SEEDS,
            "enough_seeds": enough_seeds,
            "harm_delta_pe_minus_yoked_mean": harm_delta_mean,
            "harm_delta_sd": harm_delta_sd,
            "harm_delta_se": harm_delta_se,
            "rel_improvement": rel_improvement,
            "effect_size_ok": effect_size_ok,
            "rel_floor_ok": rel_floor_ok,
            "pe_vs_off_harm_delta_mean": pe_vs_off_mean,
            "secondary_return_delta_mean": ret_delta_mean,
            "secondary_done_events_delta_mean": len_delta_mean,
            "secondary_fwd_pe_delta_yoked_minus_pe": fwd_pe_delta_mean,
            "engagement_outcome_spearman_rho": engagement_outcome_rho,
            "rate_match_ok": rate_match_ok,
            "rate_match_arm_rel_gap": arm_rate_gap,
            "rate_match_worst_seed_rel_gap": worst_rate_gap,
            "rate_match_n_seeds_outside_tol": n_seeds_outside_tol,
            "aa_control_ok": aa_control_ok,
            "aa_control_max_abs_delta": aa_max_abs_delta,
            "forced_fires_min_pe_arm": _worst_min(by_arm[ARM_PE], "forced_fires_total"),
            "forced_fires_min_yoked_arm": _worst_min(by_arm[ARM_YOKED], "forced_fires_total"),
            "decomposed_max_off_arm": _worst_max(by_arm[ARM_OFF], "decomp_n_decomposed_total"),
            "boundary_fires_mean_pe_arm": (
                statistics.fmean([r["decomp_n_boundary_fires"] for r in by_arm[ARM_PE]])
                if by_arm[ARM_PE] else 0.0),
            "boundary_fires_mean_yoked_arm": (
                statistics.fmean([r["decomp_n_boundary_fires"] for r in by_arm[ARM_YOKED]])
                if by_arm[ARM_YOKED] else 0.0),
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
        measurement_seeds = MEASUREMENT_SEEDS[:3]
        control_seeds = AA_CONTROL_SEEDS[:1]
        episodes = 3
        steps = 30
        calib = min(CALIBRATION_TICKS, steps)  # keep a real calibration prefix
    else:
        measurement_seeds = MEASUREMENT_SEEDS
        control_seeds = AA_CONTROL_SEEDS
        episodes = baselines.EPISODES
        steps = baselines.STEPS_PER_EPISODE
        calib = CALIBRATION_TICKS

    assert_no_structurally_unsatisfiable_gate(
        _precondition_specs(), [_arm_context(a) for a in ARMS])

    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    # --- Measurement cells. EVERY cell enters through arm_cell (full RNG
    # reset); there is no bare _run_cell call anywhere in this file.
    #
    # ORDER IS LOAD-BEARING: ARM_PE must run before ARM_YOKED for a given seed,
    # because ARM_YOKED's per-episode forced-locus counts ARE ARM_PE's realised
    # counts for that same seed. That is what "yoked" means -- the schedule is
    # taken from the treatment arm, seed by seed, not from a nominal constant. ---
    rows: List[Dict[str, Any]] = []
    for seed in measurement_seeds:
        pe_counts: Optional[List[int]] = None
        for arm_id in (ARM_OFF, ARM_PE, ARM_YOKED):
            with arm_cell(
                seed,
                config_slice=_config_slice(arm_id, episodes, calib),
                script_path=Path(__file__),
                config_slice_declared=True,
                include_driver_script_in_hash=False,
            ) as cell:
                row = _run_cell(
                    seed, arm_id, episodes, steps,
                    yoked_counts=(pe_counts if arm_id == ARM_YOKED else None),
                    calibration_ticks=calib)
                cell.stamp(row)
            row["role"] = "measurement"
            if arm_id == ARM_PE:
                pe_counts = list(row["forced_fires_per_episode"])
            if arm_id == ARM_YOKED:
                row["yoked_from_pe_counts"] = list(pe_counts or [])
            rows.append(row)

    # --- A-A null control on ARM_PE: each seed run AA_REPLICATES times, all
    # through arm_cell. All replicates must be mutually bit-identical. ---
    aa_rows: List[Dict[str, Any]] = []
    for seed in control_seeds:
        for replicate in range(1, AA_REPLICATES + 1):
            with arm_cell(
                seed,
                config_slice=_config_slice(ARM_PE, episodes, calib),
                script_path=Path(__file__),
                config_slice_declared=True,
                include_driver_script_in_hash=False,
            ) as cell:
                row = _run_cell(seed, ARM_PE, episodes, steps,
                                calibration_ticks=calib)
                cell.stamp(row)
            row["role"] = "aa_control"
            row["aa_replicate"] = replicate
            aa_rows.append(row)

    result = _analyse(rows, aa_rows)

    run_id = (f"{EXPERIMENT_TYPE}_"
              f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_v3")
    all_rows = rows + aa_rows
    cfg_record = {
        "arms": list(ARMS),
        "load_bearing_contrast": f"{ARM_PE} vs {ARM_YOKED}",
        "measurement_seeds": list(measurement_seeds),
        "aa_control_seeds": list(control_seeds),
        "aa_control_arm": ARM_PE,
        "aa_replicates": AA_REPLICATES,
        "episodes": episodes,
        "steps_per_episode": steps,
        "calibration_ticks": calib,
        "min_seeds_required": MIN_SEEDS,
        "self_dim": baselines.SELF_DIM,
        "world_dim": baselines.WORLD_DIM,
        "seeded_chunk_sequence": list(baselines.SEEDED_CHUNK_SEQUENCE),
        "decomposition_vs_threshold": baselines.DECOMPOSITION_VS_THRESHOLD,
        "decomposition_depth_cap": baselines.DECOMPOSITION_DEPTH_CAP,
        "hazard_tuned_env_overlay": dict(baselines.HAZARD_TUNED_ENV_OVERLAY),
        "hazard_tuned_stream_flags": dict(baselines.HAZARD_TUNED_STREAM_FLAGS),
        "trigger": {
            "readout": "e3_prediction_error (forward-model PE), lag 1 tick",
            "rule": "top-q% within-run, expanding-window quantile",
            "pe_quantile_q": PE_QUANTILE_Q,
            "calibration_ticks": calib,
            "vs_fire": VS_FIRE,
            "vs_silent": VS_SILENT,
            "yoked_rng_stream_offset": YOKED_RNG_STREAM_OFFSET,
        },
        "arm_flags": {a: _arm_flags(a) for a in ARMS},
        "thresholds": {
            "PE_VARIANCE_FLOOR": PE_VARIANCE_FLOOR,
            "PE_SANITY_CEIL": PE_SANITY_CEIL,
            "FORCED_FIRE_FLOOR": FORCED_FIRE_FLOOR,
            "DECOMP_INERT_CEIL": DECOMP_INERT_CEIL,
            "RATE_MATCH_TOL": RATE_MATCH_TOL,
            "EFFECT_SIZE_K_SIGMA": EFFECT_SIZE_K_SIGMA,
            "REL_IMPROVEMENT_FLOOR": REL_IMPROVEMENT_FLOOR,
            "MIN_SEEDS": MIN_SEEDS,
            "PE_QUANTILE_Q": PE_QUANTILE_Q,
        },
    }

    direction = result["evidence_direction"]
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "outcome": result["outcome"],
        "claim_ids": CLAIM_IDS,
        "bears_on": ["ARC-070", "MECH-321", "MECH-288", "ARC-069"],
        "evidence_direction": direction,
        # Both claims are tested by the SAME load-bearing contrast: ARC-070 is
        # the selectivity commitment and MECH-321 is its only child mechanism
        # (the trigger this driver places). C1 therefore routes identically for
        # both, and the per-claim map says so explicitly rather than leaving a
        # blanket direction to be read as if the two were independently tested.
        "evidence_direction_per_claim": {c: direction for c in CLAIM_IDS},
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "per_arm_gate": result["per_arm_gate"],
        "aa_control": result["aa_control"],
        "rate_match": result["rate_match"],
        "interpretation": {
            "label": result["interpretation_label"],
            "preconditions": result["preconditions"],
            "criteria": result["criteria"],
            "combination_rule": result["combination_rule"],
            "criteria_non_degenerate": result["criteria_non_degenerate"],
            "preconditions_scope_note": result["per_arm_gate"].get(
                "preconditions_scope_note", ""),
        },
        "summary": result["summary"],
        "per_seed_harm_deltas": result["per_seed_harm_deltas"],
        "arm_results": all_rows,
        "per_seed_rows": all_rows,
        "custom_information": {
            "design_doc": (
                "REE_assembly/evidence/planning/"
                "govdiag1_repose_mech321_chain_2026-08-12.md"),
            "repose_note": (
                "Re-pose of the ARC-070/MECH-321/MECH-288 chain, not a "
                "letter of it. The six prior runs (816b/816c/816d/830 x2/839) "
                "each died at a trigger-occupancy gate with a DV conditional "
                "on that occupancy, which aliases NO EFFECT and NO OCCASION "
                "into one non-verdict. Three changes fix that: (1) the "
                "trigger is RANK-BASED within the run (top-q% forward-PE), so "
                "occupancy is a design parameter rather than a run outcome "
                "and no environment escalation is needed; (2) the DV is "
                "UNCONDITIONAL whole-episode, so it is defined regardless of "
                "trigger activity; (3) the control is RATE-MATCHED (ARM_YOKED), "
                "which is what isolates SELECTIVITY -- ARC-070's actual "
                "content -- from decomposition per se. None of the six chain "
                "runs had a rate-matched control."),
            "predecessor_runs_not_superseded": [
                "V3-EXQ-816b", "V3-EXQ-816c", "V3-EXQ-816d",
                "V3-EXQ-830", "V3-EXQ-839", "V3-EXQ-904", "V3-EXQ-919"],
            "gating_precondition_discharged": (
                "Authored only after the design doc's section 7 ledger "
                "resolutions landed (REE_assembly a843ee6ebb, 2026-08-18, "
                "GFLAG-0038): H-vs-proxy-saturation -> confirmed; "
                "H-env-underdrives-uncertainty ratified SUPERSEDED/moot (its "
                "registry state string stays 'alive' only because the schema "
                "has no 'superseded' member -- the basis field says 'READ "
                "THIS LEG AS MOOT, NOT AS OPEN'). No formally-live leg is "
                "probed by this run."),
            "region_vs_is_actuator_not_readout": (
                "The design doc section 6 refuses any driver keyed on "
                "region-V_s as the PREDICTION-FAILURE READOUT. That refusal "
                "is honoured. region_vs appears here ONLY as the substrate's "
                "trigger ACTUATOR: PolicyDecomposition.evaluate() takes it as "
                "a caller-supplied float and HippocampalModule._region_vs() "
                "is its sole source, so overriding that method is how ANY "
                "trigger is placed at a chosen locus (technique established "
                "by V3-EXQ-904). The READOUT deciding WHERE to place it is "
                "forward-model PE. This driver never reads the substrate's "
                "own region_vs value, never gates on it, and never reports "
                "it as evidence. There is no low_vs_steps floor and no "
                "vs_heterogeneity gate anywhere in this file."),
            "arm_boundary_not_built": (
                "The optional 4th arm (ARM_BOUNDARY, MECH-288 boundary "
                "trigger rate-matched) is deliberately NOT built, and this is "
                "a decision rather than an omission. On this substrate the "
                "boundary detector is ALREADY LIVE in every arm (probe "
                "2026-08-18, seed 11: V_s silent throughout still gave 33 "
                "boundary fires -> 26 of 278 precommit evaluations "
                "decomposing), so a rate-matched boundary arm would have to "
                "SUPPRESS the natural boundary path and re-place it -- a "
                "second invasive manipulation needing its own controls, "
                "inside a run whose load-bearing contrast does not require "
                "it, and +40 cells (~+10h). INSTEAD decomp_n_boundary_fires "
                "is recorded per cell in every arm as descriptive data for "
                "whoever next takes up H-r1-r5-dissociable. That leg is NOT "
                "claimed to be answered here."),
            "pe_lag_note": (
                "The trigger reads the PREVIOUS tick's forward-PE (lag 1). "
                "e3_prediction_error is computed in "
                "E3Selector.post_action_update, i.e. AFTER the action, so the "
                "most recent PE available at decision time is the previous "
                "tick's. 'Decomposition at high-prediction-error loci' is "
                "implementable online only as 'at loci FOLLOWING high "
                "prediction error'; any same-tick formulation would be an "
                "oracle the agent could not have."),
            "dv_symmetry_declaration": (
                "DV = mean per-tick environment harm signal over the WHOLE "
                "run (set-aggregate; symmetry group = permutation of the "
                "ticks averaged over). ARM_PE and ARM_YOKED are both NOT "
                "invariant under it: forcing the R1 trigger at a locus "
                "re-tiles the committed chunk into finer primitives and "
                "releases the commit latch, changing WHICH actions are taken "
                "from that tick onward. ARM_OFF carries no manipulation, so "
                "its harm sequence is the no-decomposition baseline by "
                "construction. The load-bearing PE-minus-YOKED delta is "
                "NEITHER a broadcast additive constant, NOR a monotone "
                "rescaling, NOR a permutation of interchangeable units: the "
                "two arms place the SAME NUMBER of decompositions at "
                "DIFFERENT loci and the resulting harm sequences differ in "
                "content, not merely in labelling. The delta is a "
                "measurement, not an arithmetic identity fixed before the "
                "run."),
            "occupancy_is_a_manipulation_check": (
                "Fires-per-episode and realised decomposition rates are "
                "REPORTED per arm and per seed. The ONLY gates are the "
                "trivial existential (ARM_PE/ARM_YOKED forced fires > 0, "
                "ARM_OFF decomposition exactly 0) and the PE-vs-YOKED rate "
                "MATCH within RATE_MATCH_TOL. No absolute occupancy LEVEL is "
                "ever gated -- that precondition is the aliasing device that "
                "produced the six-run chain."),
            "gov_reuse_1_note": (
                "Decisive readout = paired per-seed unconditional "
                "whole-episode mean_harm_signal delta, ARM_PE minus "
                "ARM_YOKED. Neither arm has ever existed: no manifest in the "
                "corpus carries a rank-based forward-PE-triggered "
                "decomposition arm or a rate-matched yoked decomposition "
                "control (grep -rlE 'ARM_YOKED|ARM_PE\\b' over "
                "ree-v3/experiments/ is EMPTY, 2026-08-18). The readout "
                "cannot be derived by reprocessing banked cells because no "
                "banked cell was produced under this manipulation -- the "
                "manipulation IS the question. Not recoverable -> run. "
                "try_reuse_cell OMITTED (no prior ARM_PE/ARM_YOKED mint to "
                "cite); ARM_OFF cells ARE minted reuse-eligible "
                "(include_driver_script_in_hash=False)."),
            "re_derive_brake_note": (
                "Does not fire. Measured 2026-08-18: ARC-070 counts 0 "
                "braking autopsies, MECH-321 counts 1 "
                "(failure_autopsy_V3-EXQ-867_2026-08-02), both below the "
                "threshold of 2. Independently, this is a redesign of a "
                "DIFFERENT mechanism under a NEW EXQ NUMBER, not a lettered "
                "iteration of a braked design."),
            "substrate_defect_gate_note": (
                "Step 2.5c: one open substrate_queue entry carries "
                "substrate_paths overlapping this driver "
                "(mech203-valence-pool-admissibility, touching "
                "ree_core/agent.py::update_harm_salience and "
                "ree_core/latent/stack.py::HarmEncoder). Its severity is "
                "UNSET, so the gate does not block -- but HarmEncoder is "
                "upstream of this run's DV, so it is recorded here rather "
                "than silently passed over."),
        },
    }

    out_path = write_flat_manifest(
        manifest, None, dry_run=args.dry_run, config=cfg_record,
        seeds=list(measurement_seeds) + list(control_seeds),
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
        f"  n_seeds={s['n_seeds']}/{s['min_seeds_required']} "
        f"harm_delta_pe_minus_yoked={s['harm_delta_pe_minus_yoked_mean']:.6g} "
        f"se={s['harm_delta_se']:.6g} "
        f"rel_improvement={s['rel_improvement']:.4f}", flush=True)
    print(
        f"  rate_match_ok={s['rate_match_ok']} "
        f"arm_rel_gap={s['rate_match_arm_rel_gap']:.4f} "
        f"tol={RATE_MATCH_TOL}", flush=True)
    print(
        f"  aa_control_ok={s['aa_control_ok']} "
        f"aa_max_abs_delta={s['aa_control_max_abs_delta']}", flush=True)
    print(
        f"  forced_fires_min pe={s['forced_fires_min_pe_arm']} "
        f"yoked={s['forced_fires_min_yoked_arm']} "
        f"off_decomposed_max={s['decomposed_max_off_arm']} "
        f"(manipulation checks, non-gating levels)", flush=True)
    print(
        f"  boundary_fires_mean pe={s['boundary_fires_mean_pe_arm']:.2f} "
        f"yoked={s['boundary_fires_mean_yoked_arm']:.2f} "
        "(descriptive, H-r1-r5-dissociable)", flush=True)
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
        pe_rows = [r for r in rows if r["arm_id"] == ARM_PE]
        yk_rows = [r for r in rows if r["arm_id"] == ARM_YOKED]
        off_rows = [r for r in rows if r["arm_id"] == ARM_OFF]
        pe_forced_min = min((r["forced_fires_total"] for r in pe_rows), default=-1)
        yk_forced_min = min((r["forced_fires_total"] for r in yk_rows), default=-1)
        off_decomp_max = max((r["decomp_n_decomposed_total"] for r in off_rows), default=-1)
        pe_var_min = min((r["fwd_pe_all_var"] for r in pe_rows), default=0.0)
        # Does the DV actually MOVE between the two rate-matched arms? A
        # bit-identical PE-vs-YOKED pair would mean the placement of the
        # trigger has no behavioural consequence at all, which would make the
        # load-bearing contrast structurally vacuous -- far cheaper to catch
        # here than after 132 cells.
        pe_by_seed = {r["seed"]: r for r in pe_rows}
        yk_by_seed = {r["seed"]: r for r in yk_rows}
        shared = sorted(set(pe_by_seed) & set(yk_by_seed))
        differing = [s for s in shared
                     if pe_by_seed[s]["action_sequence"] != yk_by_seed[s]["action_sequence"]]
        print(f"[smoke] pe_forced_min={pe_forced_min} yoked_forced_min={yk_forced_min} "
              f"off_decomposed_max={off_decomp_max} pe_var_min={pe_var_min:.3e} "
              f"arms_differ_on={len(differing)}/{len(shared)} seeds "
              f"aa_ok={result['aa_control']['ok']} "
              f"aa_checks={result['aa_control']['checks']}", flush=True)
        print(f"[smoke] rate_match={result['rate_match']['per_seed']}", flush=True)
        assert pe_forced_min > 0, (
            "SMOKE FAIL: at least one ARM_PE cell never fired the injected "
            "trigger (forced_fires_total == 0). The top-q% quantile trigger "
            "is not connected -- do not queue.")
        assert yk_forced_min > 0, (
            "SMOKE FAIL: at least one ARM_YOKED cell never fired. The yoked "
            "schedule did not transfer from ARM_PE -- do not queue.")
        assert off_decomp_max == 0, (
            "SMOKE FAIL: an ARM_OFF cell decomposed. ARM_OFF must be a "
            "structural zero (use_policy_decomposition=False).")
        assert pe_var_min > PE_VARIANCE_FLOOR, (
            "SMOKE FAIL: the forward-PE ranking signal is degenerate (zero "
            "variance), so the top-q% selection would be arbitrary and "
            "ARM_PE would be a second yoked arm -- do not queue.")
        assert differing, (
            "SMOKE FAIL: ARM_PE and ARM_YOKED produced bit-identical action "
            "sequences on every shared seed. Placing the same amount of "
            "decomposition at different loci had NO behavioural consequence, "
            "so the load-bearing contrast is structurally vacuous -- do not "
            "queue.")
        assert result["aa_control"]["ok"], (
            "SMOKE FAIL: the A-A null control on ARM_PE did not produce "
            "bit-identical replicates. The new PE-history / quantile / "
            "actuator instrumentation carries uncontrolled variation -- do "
            "not queue.")

    outcome_norm = str(result["outcome"]).upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    return outcome_emit, str(out_path), bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry_run = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path,
                     dry_run=_dry_run)
