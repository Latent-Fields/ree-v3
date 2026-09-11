"""V3-EXQ-1025: MECH-349 -- CandidateRuleField CHURN / RETIREMENT leg (FALSIFYING(3)).

red-team: PASS 1 (opus-5; fable requested first, refused by a provider spend limit, re-spawned
once on the session model per /queue-experiment Step 4.5) -- CONTESTED. All findings dispositioned
under DESIGN HISTORY below; none dismissed without a source check.

=== THE QUESTION THIS RUN SETTLES, AND WHY IT IS OWED ===
MECH-349 FALSIFYING(3) says the claim is falsified if "the 666c signature reappears on the
SD-078-centered key with maintenance ON -- high crf_n_minted_total with crf_n_retired_total of
comparable magnitude and crf_max_pairwise_rule_dist at or near 0 ... it mints tokens, not slots."

No run has ever been able to test it. Under the stack MECH-349 names --
crf_mature_pool_dynamics=True + crf_availability_maintenance=True with the default
crf_maintenance_decay=0.0 -- retirement is STRUCTURALLY UNREACHABLE, so crf_n_retired_total is
pinned at 0 and FALSIFYING(3) cannot occur whatever the mechanism does. That structural fact is
recorded in governance flag GFLAG-0265 and is RE-MEASURED HERE from scratch by ARM_FROZEN (12
minted, 0 retired, every seed) rather than being taken on trust.

PROVENANCE, stated precisely because it is easy to get wrong: the id V3-EXQ-1024 was RESERVED on
2026-09-11 for MECH-349's owed GFLAG-0198 validation run and then RELEASED WITHOUT EVER BEING
QUEUED. Four successive designs were each killed by adversarial design review, no manifest was
written, no queue entry ever reached origin or the coordinator DB, and the driver was retired to
experiments/_scratch/mech349_crf_harness_NOT_QUEUEABLE.py. So there is NO V3-EXQ-1024 RUN and no
1024 manifest to cite; the harness conventions this driver reuses come from that retired script,
and every empirical figure below was measured by this driver. THIS RUN IS THE GFLAG-0265
RESIDUAL: it makes retirement genuinely reachable and then asks
whether churn is SELECTIVE (a structural creator that forgets what is gone and keeps what is
present) or INDISCRIMINATE (a token-minter -- the 666c treadmill).

=== THE LEVER, AND THE ONE THAT DOES NOT WORK ===
MECH-349's own what_would_answer (c) names three remedies; the first is "lower maintenance_floor
below the retire floor for a subset of rules". MEASURED 2026-09-11, THAT LEVER IS INERT ON ITS
OWN, and the reason is worth recording because it is invisible from the config alone:
_maybe_mint sets init_avail = max(tolerance_floor, maintenance_floor), so dropping
maintenance_floor from 0.45 to 0.02 only lowers a freshly minted rule to tolerance_floor (0.30)
-- still six times the 0.05 retire floor. Measured: retired=0, unchanged. Dropping
tolerance_floor TOO does produce retirement (93 of 100 mints), but degenerately: every rule then
mints BELOW the retire floor and dies on its first unprotected credit tick, which MANUFACTURES the
666c signature rather than testing for it. That configuration is rejected.

The lever used here is crf_maintenance_decay -- the config's own documented "optional slow
long-horizon multiplicative leak per tick that REPLACES the silence-driven decay under
maintenance (default 0.0 = pure hold -- the synaptic impression persists)". It is the ONLY knob in
credit() that touches a rule which is not currently eligible, and it is therefore the only one
that can reach a rule that has gone silent. It implements use-it-or-lose-it: an ACTIVE rule is
credited toward 1.0 (alpha 0.1, eligibility 1.0) which dominates a few-percent leak, while a
SILENT rule leaks monotonically toward the retire floor. Retirement becomes REACHABLE without
being FORCED -- the only regime in which FALSIFYING(3) is a live possibility rather than an
arithmetic outcome.

=== MEASURED DOSE-RESPONSE (seed 0, the ecology below; why the arms sit where they sit) ===
    maintenance_decay   minted  retired   retired_PERSISTENT  remint_PERSISTENT
        0.000 - 0.005       12        0                    0                  0   <- frozen
        0.010 - 0.280       12        6                    0                  0   <- SELECTIVE
        0.300               49       43                   13                 13   <- transition
        0.320              120      116                   45                 43   <- TREADMILL
        0.350 - 0.400      171      175                   94                 98   <- TREADMILL
The selective plateau spans a 28x range of the lever (0.01 to 0.28) and the treadmill transition
is at ~0.30, so ARM_LEAK_LO (0.02) and ARM_LEAK_HI (0.10) are interior points of a wide basin,
NOT a tuned knife-edge, and ARM_TREADMILL_POSCTL (0.35) clears the transition comfortably.
Verified on all 5 seeds: 12/0 at 0.00, 12 minted / 6 retired (all 6 extinct, 0 persistent, 0
persistent re-mints) at 0.02 and 0.10, and 171-172 minted / 165-166 retired with 94-95 persistent
re-mints at 0.35. Figures are post-fix -- see DESIGN HISTORY.

=== DESIGN HISTORY -- defects found and fixed BEFORE queueing ===
CHURN BOOKKEEPING UNDERCOUNTED BY 30% (found 2026-09-11, pre-queue). The first version attributed
mints and retirements by SET DIFFERENCE on live slot INDICES across a step. That silently loses
any slot retired and re-filled inside the SAME step() -- credit() retires before _maybe_mint()
runs, and _free_slot_index() returns the LOWEST free index, so the freed slot is usually the one
re-used immediately. Measured at maintenance_decay=0.35: 160 mints / 154 retirements counted
against the field's own 228 / 222. The bias UNDERSTATES churn, i.e. it is a false negative on C1
in the `supports` direction -- the worst direction for this run. Object id() cannot repair it:
CPython reuses the freed rule's address for its replacement, so ids compare EQUAL across a
same-step recycle (a first id()-based detector duly reported 0 recycles against a real 68-event
gap). FIX: rule instances are keyed (slot_index, minted_step), which is unique per instance and
immune to both hazards, plus a hard reconciliation assert against the field's own counters on
every cell. An independent adversarial pass then re-derived the same 47% undercount from
instrumented ground truth, confirming both the defect and the repair.

ADVERSARIAL DESIGN REVIEW (CONTESTED) -- four further defects, all fixed:
 (F1) C2 WAS INERT: it was ANALYTICALLY IMPLIED by (gate AND C1). The gate required every scored
   cell to retire >= 1; C1 required retired_persistent == 0; retirements partition into
   persistent + extinct; therefore retired_extinct >= 1, i.e. C2, always. `load_bearing_pass`
   reduced to `c1_pass` and the `crf_retirement_not_exercised` branch was unreachable.
 (F2) THE GATE CONSUMED THE SHORTFALL C2 EXISTS TO FLAG -- and this FIRED in the smoke test.
   `retirement_reachable` and C2 are the same measurement at adjacent thresholds, and the gate
   ran first, so an arm that retired nothing was evicted from scoring rather than failing C2.
   The smoke duly reported PASS/supports on one surviving arm. FIX for F1+F2: reachability is
   now gated on the POSITIVE CONTROL only; scored arms are never evicted on a criterion's own
   predicate. The previously-unreachable branch is now reachable and the same smoke correctly
   reports `crf_retirement_not_exercised` / unknown.
 (F3) `non_degenerate` COULD NEVER BE FALSE. It was `any arm green`, and ARM_FROZEN is green
   unconditionally (it is an anchor; both retirement preconditions are scoped out of it) -- so
   a run with no green SCORED arm still reached the indexer as non-degenerate with all
   preconditions met. FIX: gated on a scored arm being green, and when none is, the FULL per-arm
   precondition list is emitted so the indexer sees the unmet entries.
 (F5) MECH-349 NON-DEGENERACY (c) WAS HALF-IMPLEMENTED, and the missing conjunct failed on every
   cell: all 12 mints landed in ticks 24-35 of 840, so `mints_final_third` was 0 throughout while
   the docstring claimed (c) was covered. FIX: the ecology now staggers persistent-regime onsets
   so creation continues through the window, and the conjunct is a gated precondition. Measured
   after the fix: 2 late mints on every arm and seed.
 (F6, minor) the anchor could not fail and nothing would have noticed if it retired -> added
   `anchor_retires_nothing`; `_worst` returned NaN on an empty scored set, which is not valid
   JSON -> returns a -1 sentinel; `seeds_required` is denominated on realized n, recorded
   explicitly alongside the pre-registered value.
 NOT A DEFECT, recorded as a reading note: the dose-response was pre-measured at design time, so
 a `supports` here is the properly-tagged option-C filing MECH-349's own evidence_quality_note
 owes, not an independent discriminating test. See the seeds/power note below.

=== ARMS (one axis only: crf_maintenance_decay) ===
  ARM_FROZEN            decay 0.00  ANCHOR. The claim's named default stack. Retirement structurally
                                    unreachable; re-measures the GFLAG-0265 structural fact in
                                    this ecology. NOT SCORED.
  ARM_LEAK_LO           decay 0.02  SCORED.
  ARM_LEAK_HI           decay 0.10  SCORED.
  ARM_TREADMILL_POSCTL  decay 0.35  POSITIVE CONTROL. NOT SCORED. Its job is to prove the
                                    instrument can SEE the 666c treadmill at all. If it does not
                                    produce one, C1 is unfalsifiable and the run self-routes
                                    substrate_not_ready_requeue rather than reporting a pass.
The positive control differs from ARM_LEAK_HI on the LEVER ONLY. An earlier design starved the
persistent regimes by interleaving filler ticks instead; it was rejected because filler ticks
shift the SD-078 common-mode EMA, which would have confounded the control against the scored arms
on the very quantity (the centered context key) the claim is about.

=== THE ECOLOGY ===
Reuses the direct-drive harness conventions of the retired V3-EXQ-1024 design
(experiments/_scratch/mech349_crf_harness_NOT_QUEUEABLE.py) and both of the properties its
BLOCKING adversarial reviews forced -- do not re-derive these:
  * SETTLED COMMON-MODE BASELINE. BASELINE_WARMUP_TICKS of observe()-only ticks advance the
    SD-078 EMA without minting and without touching any recurrence counter. Without it the
    opening phase is effectively single-regime, cue-centering annihilates the regime, and the
    run manufactures a burst of MUTUALLY DUPLICATE rules that can supply an entire apparent
    effect. That defect made two earlier versions of 1024 unpublishable.
  * TAG-SPACE DUPLICATE DETECTION at a FIXED cosine. crf_max_pairwise_rule_dist is NOT a
    distinctness measurement -- it is a deterministic lookup on the live-rule SET off the fixed
    pinned_seed=6063 matrix, blind by construction to whether two rules encode the same context.
    It is recorded for continuity only and is a criterion for nothing.
Added here, and the reason the churn question is answerable at all:
  * TWO REGIME CLASSES. N_PERSISTENT regimes recur for the rest of the window once introduced;
    N_EXTINCT regimes recur from the start and then fall silent for good at EXTINCT_OFFSET_FRAC.
    A structural creator should retire the extinct ones and hold the persistent ones. A
    token-minter should churn both. Every mint and every retirement is attributed to its regime
    by slot provenance, and the attribution is asserted to PARTITION the retirements exactly.
  * STAGGERED ONSETS, so creation keeps happening. Persistent regularities are introduced
    progressively (PERSISTENT_ONSET_FRAC, the last two past the final-third boundary) rather
    than all being present from tick 0. This is what makes MECH-349 non-degeneracy (c)'s FIRST
    conjunct -- at least one mint in the final third -- satisfiable and therefore worth gating:
    an all-regimes-from-the-start ecology puts every mint in the opening ~4% of the window, which
    is the one-shot INITIALISER the claim warns turns an arm contrast into a dose contrast.

=== WHY THE LITERAL MECH-349 CHURN BAR IS RECORDED BUT NOT SCORED ===
Non-degeneracy (b) of the claim asks for crf_n_retired_total / crf_n_minted_total <= 0.25. In an
ecology that deliberately contains extinct regularities, that GLOBAL ratio is fixed by the ecology,
not by the mechanism: a perfectly selective creator retires exactly the extinct regimes, giving
N_EXTINCT / (N_PERSISTENT + N_EXTINCT) = 0.50 -- it would FAIL the bar for behaving CORRECTLY.
The bar was written for an ecology in which nothing goes extinct. So the global ratio is RECORDED
(with its ecology-determined expected value stated) and the scored criteria are per-regime-class
instead. This is a scope correction to the claim's own bar, and is reported as such.

=== PRE-REGISTERED CRITERIA (constants below; never derived from the run) ===
Scored over the SCORED ARMS ONLY (ARM_LEAK_LO, ARM_LEAK_HI).
C1 (LOAD-BEARING) -- PERSISTENT REGULARITIES HOLD THEIR SLOT. On >= C_SEEDS_REQUIRED of the
   seeds, every scored cell has ZERO retirements AND ZERO re-mints attributable to a regime that
   is still recurring. This is the direct complement of FALSIFYING(3): a re-mint of a regime
   whose regularity never went away IS the treadmill -- the slot was destroyed and re-created
   while the thing it encoded was still there.
C2 (LOAD-BEARING) -- EXTINCT REGULARITIES ARE ACTUALLY RETIRED. On >= C_SEEDS_REQUIRED seeds,
   every scored cell retires >= C2_MIN_EXTINCT_RETIRED extinct-regime rules. C2 is what stops C1
   passing vacuously: without it, ARM_FROZEN (which retires nothing at all) would satisfy C1
   perfectly. C1 and C2 together are the claim that churn is SELECTIVE.
C3 (RECORDED, NOT LOAD-BEARING) -- the live population is distinct in TAG space: no two live
   rules' centered context_tags reach DUPLICATE_COSINE. Demoted from load-bearing on measurement:
   it holds in EVERY arm including the treadmill positive control (max tag cosine 0.47-0.62,
   0 duplicate pairs everywhere), so it does not discriminate here and is scored
   non-degenerate:false. That dissociation is itself a finding -- see below.
PASS iff every applicable precondition is met on a scored arm AND C1 AND C2.

=== A SCOPE FINDING THIS RUN RECORDS ABOUT FALSIFYING(3) ITSELF ===
FALSIFYING(3) bundles THREE signatures as if they were one: (i) high minted, (ii) comparable
retired, (iii) pairwise distance at or near 0. The positive-control arm here produces (i) and (ii)
in full -- 228 minted / 222 retired -- while (iii) does NOT follow: re-minted rules land back on
the same well-separated regime directions, so tag-space distinctness stays high. The three
conjuncts therefore DISSOCIATE on the SD-078-centered key, and 666c's near-zero distance is a
property of its UNCENTERED key rather than an intrinsic consequence of churn. Recorded as a
diagnostic observation for governance; it is not scored and MECH-350 is NOT tagged.

=== NON-DEGENERACY PRECONDITIONS (MECH-349 what_would_answer (a)-(d)) ===
Regime-conditioned per arm via experiments/_lib/precondition_gate.py -- never AND'ed whole-run.
(a) mint fires -- every arm, >= PRECOND_A_MIN_MINTED mints and >= PRECOND_A_MIN_LIVE live rules.
(b) retirement is REACHABLE -- POSITIVE CONTROL ONLY, and deliberately NOT the scored arms.
    Gating a scored arm on `crf_n_retired_total >= 1` would evict it on C2's own predicate
    (`retired_extinct >= 1`) before C2 was computed, so a dose that retires nothing would be
    silently DROPPED instead of FAILING C2 -- and that narrowing always runs toward PASS.
    Whether a given dose retires is the scientific question, so it lives in a criterion, never
    in a gate. Scoped out of ARM_FROZEN too (structurally impossible there by design).
(b2) the instrument can SEE a treadmill -- scored arms, MEASURED ON THE POSITIVE CONTROL. Same
    statistic C1 routes on (persistent re-mints), taken on a known-positive control. SCOPED OUT
    of ARM_TREADMILL_POSCTL itself: a gate must not certify its own subject.
(c) BOTH conjuncts, every arm: live rules < n_slots (`pool_not_saturated`) AND >= 1 mint in the
    final third of the window (`mints_late_in_window`). The claim names the second explicitly and
    says that under maintenance with maintenance_floor > retire_floor its failure is the DEFAULT,
    so it is gated rather than assumed.
(d) the ARC-062 top-down seed leg is NOT UNDER TEST -- no arc062_seed is supplied at any call
    site. Reported as untested, never as a null.

=== DV-SYMMETRY INVARIANCE, PER ARM ===
The DVs are counts of retirement and re-mint events attributed to a regime class. Their symmetry
group is permutation of interchangeable units (counts are symmetric functions of their inputs).
The manipulation -- a per-tick multiplicative leak on availability -- is NOT invariant under it:
it acts on each rule through that rule's OWN silence interval, which differs systematically
between the persistent and extinct classes, and the retirement test is a threshold crossing rather
than a rank or a set-aggregate. Not argued only: MEASURED to move the DV from 0 to 6 to 222
retirements across the arms, on every seed.

=== SEEDS ARE A REPRODUCIBILITY GATE, NOT A POWER CALCULATION ===
Stated so the criteria cannot be over-read. Seeds vary only the common-mode offset and the
observation noise; the regime directions are seed-independent by construction (as in 1024). The
DVs are consequently near-deterministic -- identical across all 5 seeds on three of the four arms.
The seed requirement therefore certifies REPRODUCIBILITY; it is not evidence of statistical power,
and no criterion here should be read as a significance test. The discrimination this run rests on
is BETWEEN ARMS (0 vs 6 vs 222), where the separation is three orders of magnitude.

=== OPEN GOVERNANCE QUESTION THIS RUN DOES NOT SETTLE (GFLAG-0268) ===
GFLAG-0268 (contested_disposition, OPEN, raised 2026-09-11T17:37Z) asks whether MECH-349 is an
EMPIRICAL claim at all: its CREATE-face clauses (i) recurrence-gating and (ii) novelty-gating are
argued to restate CandidateRuleField._maybe_mint rather than to predict anything it could fail to
do, leaving only parameter sensitivity testable. THIS RUN DOES NOT ADDRESS THAT and must not be
read as answering it. It targets FALSIFYING(3), the CHURN face, which GFLAG-0268's tautology
argument does not reach: whether a rule whose regularity is still recurring keeps its slot is a
dynamical property of credit()'s leak-versus-refresh balance, not an arithmetic property of
_maybe_mint -- and it is demonstrably falsifiable here, since the positive-control arm fails C1
outright (94-95 persistent re-mints). The honest caveat is the other side of the same coin: what
C1 establishes is that the selective-forgetting basin CONTAINS the operating point, which is a
parameter fact of exactly the kind GFLAG-0268 says is all that remains. Governance should read
this run as evidence about the churn face only, and should settle GFLAG-0268 on its own terms.

SLEEP DRIVER: not applicable (no SleepLoopManager; this driver steps the CandidateRuleField
directly and never constructs an agent).
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.precondition_gate import (
    PreconditionSpec,
    aggregate_arm_gates,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.run_id import make_run_id
from ree_core.policy.candidate_rule_field import (
    CandidateRuleField,
    CandidateRuleFieldConfig,
)

EXPERIMENT_TYPE = "v3_exq_1025_mech349_crf_churn_retirement"
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["MECH-349"]
RELATED_EXQ = ["V3-EXQ-666c", "V3-EXQ-806"]   # NOT V3-EXQ-1024: reserved, never queued, no manifest

SEEDS = [0, 1, 2, 3, 4]

# --- ecology -----------------------------------------------------------------
CONTEXT_DIM = 16
N_PERSISTENT = 6          # regimes that, once introduced, recur for the rest of the window
N_EXTINCT = 6             # regimes that recur from the start and then stop forever
TOTAL_TICKS = 900
# STAGGERED ONSETS. MECH-349 non-degeneracy (c) requires >= 1 mint in the FINAL THIRD:
# a pool that fills in an early burst and then freezes has turned the asserted TRIGGERED
# creator into a one-shot INITIALISER, and its arm contrast is a dose contrast rather than
# a trigger contrast. An all-regimes-from-tick-0 ecology fails that conjunct by
# construction (measured: all 12 mints inside the first 35 of 840 ticks), so persistent
# regularities are INTRODUCED progressively instead and the conjunct is gated, not assumed.
PERSISTENT_ONSET_FRAC = [0.00, 0.00, 0.25, 0.50, 0.70, 0.85]
# The last TWO onsets sit past the final-third boundary (0.667), so the (c) conjunct has
# margin rather than resting on a single late mint.
DRY_RUN_ONSET_FRAC = [0.00, 0.00, 0.85]   # keeps a late onset at reduced scale
EXTINCT_OFFSET_FRAC = 0.45   # every extinct regime goes silent here, and stays silent
REGIME_MAGNITUDE = 2.0    # regime residual scale, far above OBS_NOISE
COMMON_MODE_SCALE = 3.0   # SD-008 geometry: every context sits in one narrow cone
OBS_NOISE = 0.005
BASELINE_WARMUP_TICKS = 300   # observe()-only; no minting, no recurrence counting
OUTCOME_SIGNAL = 0.1          # mild positive credit (the retired 1024 design used the same)

# --- substrate stack under test (the claim's named operating point) ----------
N_SLOTS = 64
RULE_DIM = 16

# --- the manipulation: crf_maintenance_decay (ONE axis) ----------------------
ARM_FROZEN = "ARM_FROZEN"
ARM_LEAK_LO = "ARM_LEAK_LO"
ARM_LEAK_HI = "ARM_LEAK_HI"
ARM_POSCTL = "ARM_TREADMILL_POSCTL"
ARM_DECAY: Dict[str, float] = {
    ARM_FROZEN: 0.0,
    ARM_LEAK_LO: 0.02,
    ARM_LEAK_HI: 0.10,
    ARM_POSCTL: 0.35,
}
ARMS: List[str] = [ARM_FROZEN, ARM_LEAK_LO, ARM_LEAK_HI, ARM_POSCTL]
SCORED_ARMS: List[str] = [ARM_LEAK_LO, ARM_LEAK_HI]

# --- pre-registered thresholds (constants, never derived from the run) -------
C_SEEDS_REQUIRED = 4              # of len(SEEDS); a REPRODUCIBILITY gate (see docstring)
C1_MAX_PERSISTENT_CHURN = 0       # retirements + re-mints on a still-recurring regime
C2_MIN_EXTINCT_RETIRED = 1        # a scored cell must actually forget something
DUPLICATE_COSINE = 0.95           # FIXED reference in TAG space
PRECOND_A_MIN_MINTED = 2
PRECOND_A_MIN_LIVE = 2
POSCTL_MIN_PERSISTENT_REMINT = 1  # the instrument must be able to SEE a treadmill
DIST_CONTINUITY_REF = 1.5         # recorded only -- structural identity, see docstring


def _finite(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return float(int(v))
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    return None


def _regime_direction(k: int) -> torch.Tensor:
    """Deterministic unit direction for regime k, independent of the run seed."""
    g = torch.Generator()
    g.manual_seed(1000 + int(k))
    v = torch.randn(CONTEXT_DIM, generator=g)
    return v / v.norm().clamp_min(1e-8)


def _build_config(*, maintenance_decay: float, n_slots: int) -> CandidateRuleFieldConfig:
    """The SD-078-centered, mature-pool + maintenance stack the claim names.

    Identical to the claim's named default stack except for maintenance_decay, the one
    manipulated knob. Every other field is left at its default on purpose, so the
    arms differ from 1024 and from each other on exactly one axis.
    """
    return CandidateRuleFieldConfig(
        n_slots=n_slots,
        rule_dim=RULE_DIM,
        cue_centering=True,
        mature_pool_dynamics=True,
        availability_maintenance=True,
        maintenance_decay=float(maintenance_decay),
    )


def config_slice(*, maintenance_decay: float, n_slots: int, total_ticks: int,
                 warmup: int, n_persistent: int, n_extinct: int,
                 onset_fracs: List[float]) -> Dict[str, Any]:
    """Everything the cell's computation reads. Declared for the arm fingerprint."""
    return {
        "context_dim": CONTEXT_DIM,
        "rule_dim": RULE_DIM,
        "n_slots": int(n_slots),
        "maintenance_decay": float(maintenance_decay),
        "cue_centering": True,
        "mature_pool_dynamics": True,
        "availability_maintenance": True,
        "n_persistent": int(n_persistent),
        "n_extinct": int(n_extinct),
        "total_ticks": int(total_ticks),
        "persistent_onset_frac": list(onset_fracs),
        "extinct_offset_frac": EXTINCT_OFFSET_FRAC,
        "regime_magnitude": REGIME_MAGNITUDE,
        "common_mode_scale": COMMON_MODE_SCALE,
        "obs_noise": OBS_NOISE,
        "baseline_warmup_ticks": int(warmup),
        "outcome_signal": OUTCOME_SIGNAL,
    }


def _regime_windows(total_ticks: int, n_persistent: int, n_extinct: int,
                    onset_fracs: Optional[List[float]] = None) -> List[Tuple[int, int]]:
    """(onset, offset) tick for every regime. offset is exclusive; total_ticks = never."""
    fracs = list(onset_fracs if onset_fracs is not None else PERSISTENT_ONSET_FRAC)
    if len(fracs) != n_persistent:
        raise ValueError("onset_fracs has %d entries for %d persistent regimes"
                         % (len(fracs), n_persistent))
    windows = [(int(round(f * total_ticks)), total_ticks) for f in fracs]
    off = int(round(EXTINCT_OFFSET_FRAC * total_ticks))
    windows.extend((0, off) for _ in range(n_extinct))
    return windows


def _build_sequence(total_ticks: int, n_persistent: int, n_extinct: int,
                    onset_fracs: Optional[List[float]] = None) -> List[int]:
    """Round-robin over the regimes ACTIVE at each tick, under staggered onsets.

    Round-robin rather than shuffled: the silence interval between two successive
    presentations of a regime is what the maintenance leak acts through, so it is held
    known (= the number of regimes active just then) instead of being a random variable
    that would smear the dose-response. Persistent regimes enter progressively, so new
    regularities keep arriving and minting continues into the final third of the window
    (MECH-349 non-degeneracy (c)); extinct regimes all fall silent at EXTINCT_OFFSET_FRAC.
    """
    windows = _regime_windows(total_ticks, n_persistent, n_extinct, onset_fracs)
    seq: List[int] = []
    cursor = 0
    for t in range(total_ticks):
        active = [k for k, (on, off) in enumerate(windows) if on <= t < off]
        if not active:
            active = [0]
        seq.append(active[cursor % len(active)])
        cursor += 1
    return seq


def _run_field(*, seed: int, maintenance_decay: float, n_slots: int,
               total_ticks: int, warmup: int,
               n_persistent: int, n_extinct: int, arm: str,
               onset_fracs: Optional[List[float]] = None) -> Dict[str, Any]:
    """Drive the CandidateRuleField over the two-phase ecology via the PUBLIC step().

    Mints and retirements are attributed to the regime being presented when the slot
    was created (slot provenance), which is what makes persistent-vs-extinct churn
    separable. A RE-MINT is a mint for a regime that has held a slot before.
    """
    n_all = n_persistent + n_extinct
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    common = torch.randn(CONTEXT_DIM, generator=gen) * COMMON_MODE_SCALE

    def context_for(k: int) -> torch.Tensor:
        return (common
                + _regime_direction(k) * REGIME_MAGNITUDE
                + torch.randn(CONTEXT_DIM, generator=gen) * OBS_NOISE)

    cfg = _build_config(maintenance_decay=maintenance_decay, n_slots=n_slots)
    crf = CandidateRuleField(context_dim=CONTEXT_DIM, config=cfg)

    # Baseline warmup -- observe() only: no mint, no recurrence increment.
    for i in range(warmup):
        crf.observe(context_for(i % n_all))

    sequence = _build_sequence(total_ticks, n_persistent, n_extinct, onset_fracs)
    total = len(sequence)

    slot_regime: Dict[Tuple[int, int], int] = {}
    minted_by: Dict[int, int] = {}
    retired_by: Dict[int, int] = {}
    remint_by: Dict[int, int] = {}
    seen_regimes = set()
    mint_ticks: List[int] = []

    # Rule instances are keyed (slot_index, minted_step), NOT by slot index alone and
    # NOT by object identity. Both of the obvious alternatives are WRONG here, measured:
    #   * SET DIFFERENCE ON SLOT INDICES undercounts. credit() retires BEFORE _maybe_mint()
    #     runs inside the same step() (candidate_rule_field.py step()), and
    #     _free_slot_index() returns the LOWEST free index -- so a slot freed this step is
    #     immediately re-filled this step and appears in both the before and after key sets,
    #     hiding one retirement AND one mint. Measured at maintenance_decay=0.35: 160/154
    #     counted against the field's own 228/222 -- a 30% undercount, and biased toward
    #     UNDERSTATING churn, i.e. a false negative on C1 in the `supports` direction.
    #   * OBJECT id() cannot repair it: CPython reuses the freed rule's address for its
    #     replacement, so the id compares EQUAL across a same-step recycle (that is exactly
    #     why a first id()-based detector reported 0 recycles against a real 68-event gap).
    # minted_step is assigned self._step at mint and step() increments _step first, so the
    # pair is unique per rule instance and survives both hazards. The reconciliation assert
    # below is what proves it, against the field's own authoritative counters.
    def _live_pairs() -> Dict[Tuple[int, int], Any]:
        return {(idx, int(r.minted_step)): r for idx, r in crf._rules.items()}

    prev_minted = int(crf.get_state()["crf_n_minted_total"])
    prev_retired = int(crf.get_state()["crf_n_retired_total"])
    counted_minted = 0
    counted_retired = 0

    for t, k in enumerate(sequence):
        if (t + 1) % 100 == 0 or (t + 1) == total:
            print("  [train] crf seed=%d ep %d/%d" % (seed, t + 1, total), flush=True)
        ctx = context_for(k)
        before = _live_pairs()
        # arc062_seed deliberately omitted -- precondition (d).
        crf.step(ctx, action_object_idx=int(k), outcome_signal=OUTCOME_SIGNAL)
        after = _live_pairs()
        for key in after.keys() - before.keys():
            slot_regime[key] = int(k)
            minted_by[k] = minted_by.get(k, 0) + 1
            counted_minted += 1
            mint_ticks.append(t)
            if k in seen_regimes:
                remint_by[k] = remint_by.get(k, 0) + 1
            seen_regimes.add(int(k))
        for key in before.keys() - after.keys():
            owner = slot_regime.get(key, -1)
            retired_by[owner] = retired_by.get(owner, 0) + 1
            counted_retired += 1

    # RECONCILIATION -- the instrument must account for every event the field itself
    # recorded. A silent undercount here would understate churn and bias C1 toward
    # `supports`, so this is a hard assert rather than a warning.
    st_chk = crf.get_state()
    exp_minted = int(st_chk["crf_n_minted_total"]) - prev_minted
    exp_retired = int(st_chk["crf_n_retired_total"]) - prev_retired
    if (counted_minted, counted_retired) != (exp_minted, exp_retired):
        raise AssertionError(
            "churn bookkeeping lost events (arm=%s seed=%d): attributed %d mints / %d "
            "retirements against the field's own %d / %d. Every mint and retirement must "
            "be attributable to a regime or the persistent-vs-extinct split is unsound."
            % (arm, seed, counted_minted, counted_retired, exp_minted, exp_retired))

    st = crf.get_state()
    minted = int(st["crf_n_minted_total"])
    retired = int(st["crf_n_retired_total"])
    live = int(st["crf_n_slots_minted"])

    # Duplicate detection in TAG space at a FIXED cosine. crf_max_pairwise_rule_dist is
    # computed over pinned rule_embeddings and is structurally blind to this.
    base = crf._baseline
    tags = [(r.context_tag.reshape(-1) - base) for r in crf._rules.values()]
    dup_pairs = 0
    max_tag_cos = -1.0
    for a, b in itertools.combinations(range(len(tags)), 2):
        x, y = tags[a], tags[b]
        c = float((x @ y) / (x.norm() * y.norm()).clamp_min(1e-8))
        max_tag_cos = max(max_tag_cos, c)
        if c >= DUPLICATE_COSINE:
            dup_pairs += 1

    def _sum_class(d: Dict[int, int], persistent: bool) -> int:
        if persistent:
            return sum(v for kk, v in d.items() if 0 <= kk < n_persistent)
        return sum(v for kk, v in d.items() if kk >= n_persistent)

    ret_p = _sum_class(retired_by, True)
    ret_e = _sum_class(retired_by, False)
    # The persistent/extinct split must PARTITION the retirements, not merely sample them:
    # an unattributable retirement (owner -1) would land in neither class and silently
    # shrink both. C2's meaning depends on this being exact.
    if ret_p + ret_e != retired:
        raise AssertionError(
            "retirement partition incomplete (arm=%s seed=%d): persistent %d + extinct %d "
            "!= total %d; some retirement had no regime provenance."
            % (arm, seed, ret_p, ret_e, retired))
    remint_p = _sum_class(remint_by, True)
    remint_e = _sum_class(remint_by, False)

    cut = int(total * 2 / 3)
    mints_final_third = sum(1 for tt in mint_ticks if tt >= cut)

    return {
        "arm": arm,
        "seed": int(seed),
        "maintenance_decay": float(maintenance_decay),
        "crf_n_minted_total": minted,
        "crf_n_retired_total": retired,
        "crf_live_rules_final": live,
        "crf_n_slots": int(n_slots),
        "crf_max_pairwise_rule_dist": float(st["crf_max_pairwise_rule_dist"]),
        "crf_frac_active": float(st["crf_frac_active"]),
        "global_churn_ratio": (retired / minted) if minted else 0.0,
        "retired_persistent": ret_p,
        "retired_extinct": ret_e,
        "remint_persistent": remint_p,
        "remint_extinct": remint_e,
        # The scored C1 quantity: churn attributable to a STILL-RECURRING regime.
        "persistent_churn": ret_p + remint_p,
        "minted_by_regime": {str(kk): v for kk, v in sorted(minted_by.items())},
        "retired_by_regime": {str(kk): v for kk, v in sorted(retired_by.items())},
        "duplicate_tag_pairs": dup_pairs,
        "max_tag_cosine": max_tag_cos,
        "live_over_slots": (live / n_slots) if n_slots else 0.0,
        "mints_final_third": mints_final_third,
        "total_ticks": total,
        "n_persistent": int(n_persistent),
        "n_extinct": int(n_extinct),
    }


def _run_cell(*, arm: str, seed: int, n_slots: int, total_ticks: int,
              warmup: int, n_persistent: int, n_extinct: int,
              onset_fracs: List[float]) -> Dict[str, Any]:
    """One (arm x seed) cell: full RNG reset on entry, fingerprint stamped on exit."""
    decay = ARM_DECAY[arm]
    slice_ = config_slice(maintenance_decay=decay, n_slots=n_slots,
                          total_ticks=total_ticks, onset_fracs=onset_fracs,
                          warmup=warmup, n_persistent=n_persistent, n_extinct=n_extinct)
    print("Seed %d Condition %s" % (seed, arm), flush=True)
    with arm_cell(
        seed,
        config_slice=slice_,
        script_path=Path(__file__),
        config_slice_declared=True,
        # The ecology is DRIVER-RESIDENT (phase schedule, regime split, _regime_direction
        # offsets), so the driver MUST be folded into the substrate hash: a cross-driver
        # reuse here would be a false-HIT, which corrupts a conclusion.
        include_driver_script_in_hash=True,
    ) as cell:
        row = _run_field(seed=seed, maintenance_decay=decay, n_slots=n_slots,
                         total_ticks=total_ticks, onset_fracs=onset_fracs,
                         warmup=warmup, n_persistent=n_persistent,
                         n_extinct=n_extinct, arm=arm)
        cell.stamp(row)
    print("verdict: %s" % ("PASS" if row["crf_n_minted_total"] >= PRECOND_A_MIN_MINTED
                           else "FAIL"), flush=True)
    return row


def _rows_for(rows: List[Dict[str, Any]], arm: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r["arm"] == arm]


def _by_seed(rows: List[Dict[str, Any]], arm: str, key: str) -> Dict[int, Any]:
    return {r["seed"]: r[key] for r in rows if r["arm"] == arm}


def _worst(rows: List[Dict[str, Any]], key: str, mode: str = "min") -> Tuple[float, str]:
    """Worst-case value + offending cell id.

    A precondition's measured value must be the SAME statistic its met tests. met here
    is a worst-case claim over the arm's cells, so the WORST cell is reported, never the
    mean (the indexer recomputes met from the reported number).
    """
    if not rows:
        # NOT nan: this value reaches the manifest, and json.dumps emits a bare NaN token
        # that is not valid JSON. -1 is out of range for every statistic used here and
        # reads unambiguously as "no cell", which is the state when no arm was scored.
        return (-1.0, "(no scored cell)")
    pick = min(rows, key=lambda r: r[key]) if mode == "min" else max(rows, key=lambda r: r[key])
    return (float(pick[key]), "%s/seed%d" % (pick["arm"], pick["seed"]))


def _preconditions() -> List[PreconditionSpec]:
    """Regime-conditioned preconditions. Never AND'ed whole-run."""
    scored_or_posctl = set(SCORED_ARMS) | {ARM_POSCTL}
    return [
        PreconditionSpec(
            name="mint_fires",
            description=("Worst cell in the arm mints >= %d rules (a run with 0-1 live rules "
                         "measures nothing). MECH-349 non-degeneracy (a)."
                         % PRECOND_A_MIN_MINTED),
            control="the arm's own worst cell; minting is not manipulated on any arm",
            threshold=float(PRECOND_A_MIN_MINTED - 1),   # floor: met when measured > threshold
            direction="lower",
            kind="readiness",
        ),
        PreconditionSpec(
            name="live_rules_present",
            description=("Worst cell in the arm ends with >= %d live rules."
                         % PRECOND_A_MIN_LIVE),
            control="the arm's own worst cell",
            threshold=float(PRECOND_A_MIN_LIVE - 1),
            direction="lower",
            kind="readiness",
        ),
        PreconditionSpec(
            name="pool_not_saturated",
            description=("Worst cell's live rules stay strictly below n_slots, so mints are "
                         "regularity-limited rather than slot-limited. MECH-349 (c)."),
            control="the arm's own worst (highest live/slots) cell",
            threshold=1.0,
            direction="upper",
            kind="readiness",
        ),
        PreconditionSpec(
            name="retirement_reachable",
            description=("Worst cell in the arm retires >= 1 rule. Without this, churn is "
                         "pinned at 0 by the substrate and FALSIFYING(3) cannot occur whatever "
                         "the mechanism does -- the exact blocker GFLAG-0265 records."),
            control="the arm's own worst cell under its maintenance_decay dose",
            threshold=0.0,
            direction="lower",
            kind="readiness",
            applies_to=lambda ctx: ctx["arm"] == ARM_POSCTL,
            applies_note=("SCOPED TO THE POSITIVE CONTROL ONLY, and this is load-bearing. "
                          "Applying it to a SCORED arm would evict that arm from scoring on "
                          "C2's OWN PREDICATE -- `crf_n_retired_total >= 1` and C2's "
                          "`retired_extinct >= 1` are the same measurement at adjacent "
                          "thresholds -- and the eviction runs FIRST, so a dose that retires "
                          "nothing would be silently dropped instead of FAILING C2, and the "
                          "narrowing is always toward PASS. Whether a given dose retires is "
                          "the SCIENTIFIC QUESTION, so it belongs in a criterion, never in a "
                          "gate. ARM_FROZEN is scoped out too: retirement is structurally "
                          "unreachable there BY DESIGN (that arm is the GFLAG-0265 anchor), "
                          "precondition-gate disposition (a)."),
            structural_max=lambda ctx: 0.0 if ctx["maintenance_decay"] == 0.0 else None,
        ),
        PreconditionSpec(
            name="mints_late_in_window",
            description=("Worst cell in the arm makes >= 1 mint in the FINAL THIRD of the "
                         "window. MECH-349 non-degeneracy (c), first conjunct: a pool that "
                         "fills in an early burst and then freezes has converted the asserted "
                         "TRIGGERED creator into a one-shot INITIALISER, and its arm contrast "
                         "is a dose contrast rather than a trigger contrast."),
            control=("the arm's own worst cell; persistent regimes are introduced on staggered "
                     "onsets (the last at 0.75 of the window) so late minting is reachable"),
            threshold=0.0,
            direction="lower",
            kind="readiness",
        ),
        PreconditionSpec(
            name="anchor_retires_nothing",
            description=("ARM_FROZEN retires 0 rules, re-measuring the GFLAG-0265 structural "
                         "fact that retirement is unreachable at maintenance_decay=0.0. If it "
                         "ever retires, the framing this whole run rests on is wrong."),
            control="the anchor arm itself; maintenance_decay=0.0 makes the leak a no-op",
            threshold=1.0,
            direction="upper",
            kind="readiness",
            applies_to=lambda ctx: ctx["arm"] == ARM_FROZEN,
            applies_note="Meaningful only for the anchor arm.",
        ),
        PreconditionSpec(
            name="treadmill_instrument_sensitive",
            description=("The positive control produces >= %d re-mints of a STILL-RECURRING "
                         "regime, i.e. the 666c treadmill is producible in this ecology. Without "
                         "it C1 could not have failed under any outcome and its pass means "
                         "nothing." % POSCTL_MIN_PERSISTENT_REMINT),
            control=("MEASURED ON ARM_TREADMILL_POSCTL (maintenance_decay=%.2f) -- a known-"
                     "positive control, and the SAME statistic C1 routes on (persistent "
                     "re-mints)." % ARM_DECAY[ARM_POSCTL]),
            threshold=float(POSCTL_MIN_PERSISTENT_REMINT - 1),
            direction="lower",
            kind="readiness",
            applies_to=lambda ctx: ctx["arm"] in SCORED_ARMS,
            applies_note=("Scoped out of ARM_TREADMILL_POSCTL itself: a gate must not certify "
                          "its own subject. Scoped out of ARM_FROZEN, which is not scored."),
        ),
    ]


def _arm_context(arm: str) -> Dict[str, Any]:
    return {"arm": arm, "maintenance_decay": ARM_DECAY[arm],
            "is_scored": arm in SCORED_ARMS}


def main(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    seeds = SEEDS[:1] if dry_run else list(SEEDS)
    n_slots = N_SLOTS
    n_persistent = 3 if dry_run else N_PERSISTENT
    n_extinct = 3 if dry_run else N_EXTINCT
    total_ticks = 300 if dry_run else TOTAL_TICKS
    onset_fracs = list(DRY_RUN_ONSET_FRAC if dry_run else PERSISTENT_ONSET_FRAC)
    warm = 60 if dry_run else BASELINE_WARMUP_TICKS

    specs = _preconditions()
    # Design-time proof: refuse the run BEFORE compute if any arm's gate is structurally
    # unsatisfiable and was not deliberately scoped out or acknowledged.
    assert_no_structurally_unsatisfiable_gate(specs, [_arm_context(a) for a in ARMS])

    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            rows.append(_run_cell(arm=arm, seed=seed, n_slots=n_slots,
                                  total_ticks=total_ticks, onset_fracs=onset_fracs,
                                  warmup=warm, n_persistent=n_persistent,
                                  n_extinct=n_extinct))

    # ---- per-arm gates (regime-conditioned; a red arm never vacates a green one) ----
    posctl_remint_by_seed = _by_seed(rows, ARM_POSCTL, "remint_persistent")
    arm_gates = []
    for arm in ARMS:
        arm_rows = _rows_for(rows, arm)
        measured: Dict[str, float] = {
            "mint_fires": _worst(arm_rows, "crf_n_minted_total", "min")[0],
            "live_rules_present": _worst(arm_rows, "crf_live_rules_final", "min")[0],
            "pool_not_saturated": _worst(arm_rows, "live_over_slots", "max")[0],
            "mints_late_in_window": _worst(arm_rows, "mints_final_third", "min")[0],
        }
        if arm == ARM_POSCTL:
            measured["retirement_reachable"] = _worst(arm_rows, "crf_n_retired_total", "min")[0]
        if arm == ARM_FROZEN:
            measured["anchor_retires_nothing"] = _worst(arm_rows, "crf_n_retired_total", "max")[0]
        if arm in SCORED_ARMS:
            # Worst (lowest) positive-control re-mint count over the seeds actually run.
            measured["treadmill_instrument_sensitive"] = float(
                min(posctl_remint_by_seed.values()) if posctl_remint_by_seed else 0.0)
        arm_gates.append(evaluate_arm_gate(arm, _arm_context(arm), specs, measured))

    agg = aggregate_arm_gates(arm_gates)
    gate_by_arm = {g["arm"]: g for g in arm_gates}
    scored_green = [a for a in SCORED_ARMS if gate_by_arm[a]["gate_green"]]
    # A verdict on MECH-349 needs at least one SCORED arm green. The anchor and the
    # positive control cannot supply one -- neither is a test of the claim.
    preconditions_met = bool(scored_green)

    # ---- criteria, over the GREEN SCORED arms only ----
    scored_rows = [r for r in rows if r["arm"] in scored_green]

    def _seeds_meeting(pred) -> int:
        ok = 0
        for s in seeds:
            cells = [r for r in scored_rows if r["seed"] == s]
            if cells and all(pred(r) for r in cells):
                ok += 1
        return ok

    # Denominated on the REALIZED seed count so --dry-run stays runnable, but the
    # pre-registered value is recorded alongside it: a silently truncated seed list would
    # otherwise degrade "4 of 5" to "1 of 1" with nothing in the manifest to show for it.
    seeds_required = min(C_SEEDS_REQUIRED, len(seeds))
    seeds_truncated = len(seeds) != len(SEEDS)
    c1_n = _seeds_meeting(lambda r: r["persistent_churn"] <= C1_MAX_PERSISTENT_CHURN)
    c2_n = _seeds_meeting(lambda r: r["retired_extinct"] >= C2_MIN_EXTINCT_RETIRED)
    c3_n = _seeds_meeting(lambda r: r["duplicate_tag_pairs"] == 0)
    c1_pass = c1_n >= seeds_required
    c2_pass = c2_n >= seeds_required
    c3_pass = c3_n >= seeds_required

    worst_pc, worst_pc_cell = _worst(scored_rows, "persistent_churn", "max")
    worst_er, worst_er_cell = _worst(scored_rows, "retired_extinct", "min")
    worst_dup, worst_dup_cell = _worst(scored_rows, "duplicate_tag_pairs", "max")

    criteria = [
        {
            "name": "C1_persistent_regimes_hold_their_slot",
            "load_bearing": True,
            "measured": float(c1_n),
            "threshold": float(seeds_required),
            "passed": bool(c1_pass),
            "detail_worst_cell": {
                "measured": worst_pc,
                "threshold": float(C1_MAX_PERSISTENT_CHURN),
                "offending_cell": worst_pc_cell,
                "statistic": "retired_persistent + remint_persistent",
            },
            "note": ("Direct complement of MECH-349 FALSIFYING(3): a re-mint of a regime whose "
                     "regularity is still recurring IS the treadmill."),
        },
        {
            "name": "C2_extinct_regimes_are_retired",
            "load_bearing": True,
            "measured": float(c2_n),
            "threshold": float(seeds_required),
            "passed": bool(c2_pass),
            "detail_worst_cell": {
                "measured": worst_er,
                "threshold": float(C2_MIN_EXTINCT_RETIRED),
                "offending_cell": worst_er_cell,
                "statistic": "retired_extinct",
            },
            "note": ("Stops C1 passing vacuously: ARM_FROZEN, which retires nothing at all, "
                     "would satisfy C1 perfectly."),
        },
        {
            "name": "C3_live_population_distinct_in_tag_space",
            "load_bearing": False,
            "measured": float(c3_n),
            "threshold": float(seeds_required),
            "passed": bool(c3_pass),
            "detail_worst_cell": {
                "measured": worst_dup,
                "threshold": 0.0,
                "offending_cell": worst_dup_cell,
                "statistic": "duplicate_tag_pairs at cosine >= %.2f" % DUPLICATE_COSINE,
            },
            "note": ("RECORDED, NOT LOAD-BEARING. It holds in every arm including the treadmill "
                     "positive control, so it does not discriminate here."),
        },
    ]
    combination_rule = ("PASS iff at least one SCORED arm's precondition gate is green AND C1 AND "
                        "C2. C3 is recorded, not load-bearing (see its note). ARM_FROZEN and "
                        "ARM_TREADMILL_POSCTL are never scored.")
    load_bearing_pass = bool(c1_pass and c2_pass)
    outcome = "PASS" if (preconditions_met and load_bearing_pass) else "FAIL"

    # ---- non-degeneracy, per criterion ----
    posctl_rows = _rows_for(rows, ARM_POSCTL)
    posctl_treadmill = max((r["remint_persistent"] for r in posctl_rows), default=0)
    posctl_passes_c3 = (all(r["duplicate_tag_pairs"] == 0 for r in posctl_rows)
                        if posctl_rows else False)
    # Reachability is demonstrated on the CONTROL, never on the scored rows themselves:
    # `any(scored retired > 0)` would be the same predicate C2 tests, so it could not
    # witness C2's own degeneracy.
    posctl_retired_any = any(r["crf_n_retired_total"] > 0 for r in posctl_rows)
    min_live = min((r["crf_live_rules_final"] for r in scored_rows), default=0)
    criteria_non_degenerate = {
        # C1 can only discriminate if the treadmill it forbids is producible at all.
        "C1_persistent_regimes_hold_their_slot":
            bool(posctl_treadmill >= POSCTL_MIN_PERSISTENT_REMINT),
        # C2 can only discriminate if retirement is reachable on the scored arms.
        "C2_extinct_regimes_are_retired": bool(posctl_retired_any),
        # C3 is degenerate exactly when the positive control also passes it (it does).
        "C3_live_population_distinct_in_tag_space":
            bool(min_live >= 2 and not posctl_passes_c3),
    }

    # ---- verdict routing ----
    expected_global_ratio = n_extinct / float(n_persistent + n_extinct)
    if not preconditions_met:
        label = "substrate_not_ready_requeue"
        reds = ", ".join("%s(%s)" % (a, ",".join(gate_by_arm[a]["failed_preconditions"]))
                         for a in SCORED_ARMS if not gate_by_arm[a]["gate_green"])
        summary = ("No SCORED arm cleared its precondition gate (%s), so no MECH-349 verdict is "
                   "admissible. Re-queue at a maintenance_decay dose where retirement is "
                   "reachable and the positive control produces a treadmill." % reds)
        direction = "unknown"
    elif not criteria_non_degenerate["C1_persistent_regimes_hold_their_slot"]:
        label = "substrate_not_ready_requeue"
        summary = ("The positive control did not produce a treadmill (persistent re-mints=%d), so "
                   "C1 could not have failed under any outcome and its pass is uninformative. "
                   "Instrument not ready; re-queue at a higher positive-control dose."
                   % posctl_treadmill)
        direction = "unknown"
    elif load_bearing_pass:
        label = "crf_churn_selective_not_treadmill"
        summary = ("With retirement made reachable, churn is SELECTIVE: extinct regularities are "
                   "retired while still-recurring ones keep their slots (zero persistent "
                   "retirements or re-mints), and the same substrate DOES produce the 666c "
                   "treadmill at a higher dose. MECH-349 FALSIFYING(3) does not occur.")
        direction = "supports"
    elif not c1_pass:
        label = "crf_666c_treadmill_reappears"
        summary = ("C1 failed: still-recurring regularities lost and re-created their slots "
                   "(worst cell %s, persistent churn %g). This IS MECH-349 FALSIFYING(3) -- the "
                   "CREATE face mints tokens, not slots, once retirement is reachable."
                   % (worst_pc_cell, worst_pc))
        direction = "weakens"
    else:
        # C2 failed with C1 passing: the scored arms forgot nothing, so the selective-
        # forgetting contrast was never exercised. Nothing about the claim was tested.
        label = "crf_retirement_not_exercised"
        summary = ("C2 failed while C1 passed: the scored arms retired no extinct-regime rule, so "
                   "the selective-forgetting contrast was never exercised. This is an instrument "
                   "outcome, not evidence about MECH-349.")
        direction = "unknown"

    _readout_raw: Dict[str, Any] = {
        "c1_passed": c1_pass,
        "c1_seeds_meeting": c1_n,
        "c2_passed": c2_pass,
        "c2_seeds_meeting": c2_n,
        "c3_passed": c3_pass,
        "c3_seeds_meeting": c3_n,
        "seeds_required": seeds_required,
        "seeds_required_preregistered": C_SEEDS_REQUIRED,
        "seeds_preregistered": len(SEEDS),
        "seeds_truncated": seeds_truncated,
        "frozen_mints_final_third": _worst(_rows_for(rows, ARM_FROZEN), "mints_final_third", "min")[0],
        "worst_persistent_churn_scored": worst_pc,
        "worst_extinct_retired_scored": worst_er,
        "worst_duplicate_tag_pairs_scored": worst_dup,
        "duplicate_cosine_reference": DUPLICATE_COSINE,
        "posctl_persistent_remints": posctl_treadmill,
        "frozen_retired_total": sum(r["crf_n_retired_total"] for r in _rows_for(rows, ARM_FROZEN)),
        "leak_lo_retired_total": sum(r["crf_n_retired_total"] for r in _rows_for(rows, ARM_LEAK_LO)),
        "leak_hi_retired_total": sum(r["crf_n_retired_total"] for r in _rows_for(rows, ARM_LEAK_HI)),
        "posctl_retired_total": sum(r["crf_n_retired_total"] for r in posctl_rows),
        "posctl_minted_total": sum(r["crf_n_minted_total"] for r in posctl_rows),
        # RECORDED, NOT SCORED -- ecology-determined; see the docstring.
        "global_churn_ratio_worst_scored": _worst(scored_rows, "global_churn_ratio", "max")[0],
        "global_churn_ratio_expected_if_perfectly_selective": expected_global_ratio,
        "mech349_literal_churn_bar": 0.25,
        "preconditions_met": preconditions_met,
        "load_bearing_pass": load_bearing_pass,
        "n_seeds": len(seeds),
        "n_scored_arms_green": len(scored_green),
    }
    readout = {k: v for k, v in ((k, _finite(v)) for k, v in _readout_raw.items())
               if v is not None}

    full_config = {
        "context_dim": CONTEXT_DIM,
        "n_persistent": n_persistent,
        "n_extinct": n_extinct,
        "total_ticks": total_ticks,
        "persistent_onset_frac": list(onset_fracs),
        "extinct_offset_frac": EXTINCT_OFFSET_FRAC,
        "baseline_warmup_ticks": warm,
        "regime_magnitude": REGIME_MAGNITUDE,
        "common_mode_scale": COMMON_MODE_SCALE,
        "obs_noise": OBS_NOISE,
        "n_slots": n_slots,
        "rule_dim": RULE_DIM,
        "arm_decay": dict(ARM_DECAY),
        "scored_arms": list(SCORED_ARMS),
        "duplicate_cosine": DUPLICATE_COSINE,
        "outcome_signal": OUTCOME_SIGNAL,
        "cue_centering": True,
        "mature_pool_dynamics": True,
        "availability_maintenance": True,
        "seed_from_arc062": False,
        "seeds": seeds,
    }

    manifest: Dict[str, Any] = {
        "run_id": make_run_id(EXPERIMENT_TYPE),
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "related_exq": RELATED_EXQ,
        "outcome": outcome,
        "evidence_direction": direction,
        "sleep_driver_pattern": "not_applicable_no_sleep_loop",
        "readout": readout,
        "criteria": criteria,
        "combination_rule": combination_rule,
        "per_arm_gate": agg["per_arm_gate"],
        # ARM_FROZEN is green unconditionally (it is an anchor, and both retirement
        # preconditions are scoped out of it), so `any arm green` alone can NEVER be False
        # -- including on the branch where this script itself says no verdict is admissible.
        # The indexer reads this field as authoritative, so it is gated on a SCORED arm.
        "non_degenerate": bool(agg["non_degenerate"] and preconditions_met),
        "degeneracy_reason": (agg["degeneracy_reason"] if preconditions_met else
                              ("No SCORED arm cleared its gate; the anchor and positive-control "
                               "arms are green but neither is a test of MECH-349. "
                               + str(agg["degeneracy_reason"] or ""))),
        "interpretation": {
            "label": label,
            "summary": summary,
            # Green-arms-only on a partial run (precondition_gate's rule: a red arm's entry
            # would re-vacate the green arm at adjudication time). But when NO scored arm is
            # green there is no green arm to protect, and a green-only list would show the
            # indexer "all preconditions met" on a run that concluded nothing -- so the full
            # per-arm list is emitted instead.
            "preconditions": (agg["adjudication_preconditions"] if preconditions_met
                              else [pc for g in arm_gates for pc in g["preconditions"]]),
            "criteria_non_degenerate": criteria_non_degenerate,
            "scope_finding_falsifying3_conjuncts_dissociate": {
                "statement": ("MECH-349 FALSIFYING(3) bundles three signatures: high minted, "
                              "comparable retired, and pairwise distance at or near 0. The "
                              "positive control here produces the first two in full while the "
                              "third does NOT follow -- re-minted rules land back on the same "
                              "well-separated regime directions, so tag-space distinctness stays "
                              "high."),
                "posctl_minted": sum(r["crf_n_minted_total"] for r in posctl_rows),
                "posctl_retired": sum(r["crf_n_retired_total"] for r in posctl_rows),
                "posctl_duplicate_tag_pairs": sum(r["duplicate_tag_pairs"] for r in posctl_rows),
                "implication": ("666c's near-zero distance is a property of its UNCENTERED "
                                "context key, not an intrinsic consequence of churn. Recorded "
                                "for governance; not scored, and MECH-350 is NOT tagged."),
            },
            "lever_note": {
                "inert_lever": ("crf_maintenance_floor below crf_mature_retire_floor is INERT on "
                                "its own: _maybe_mint sets init_avail = max(tolerance_floor, "
                                "maintenance_floor), so a rule still mints at tolerance_floor "
                                "(0.30), six times the 0.05 retire floor."),
                "rejected_lever": ("Lowering tolerance_floor too DOES retire (93/100) but "
                                   "degenerately -- every rule mints below the retire floor and "
                                   "dies on its first unprotected credit tick, MANUFACTURING the "
                                   "666c signature instead of testing for it."),
                "lever_used": ("crf_maintenance_decay -- the only knob in credit() that touches a "
                               "rule carrying no eligibility, and therefore the only one that can "
                               "reach a rule that has gone silent."),
            },
            "structural_facts_not_measurements": {
                "distinctness_is_an_identity": (
                    "crf_max_pairwise_rule_dist is a deterministic lookup on the live-rule SET "
                    "off the pinned_seed=6063 matrix. Recorded for continuity with V3-EXQ-806's "
                    "1.598-1.711 only; a criterion for nothing here, and MECH-350 is NOT tagged."),
                "global_churn_ratio_is_ecology_determined": (
                    "MECH-349 non-degeneracy (b) asks for retired/minted <= 0.25. In an ecology "
                    "containing extinct regularities that ratio is fixed by the ecology: a "
                    "perfectly SELECTIVE creator retires exactly the extinct regimes, giving "
                    "%.2f -- it would FAIL the bar for behaving correctly. Recorded, not scored; "
                    "the scored criteria are per-regime-class instead." % expected_global_ratio),
            },
            "seeds_are_reproducibility_not_power": (
                "Seeds vary only the common-mode offset and observation noise; regime directions "
                "are seed-independent, so the DVs are near-deterministic. The seed requirement "
                "certifies reproducibility and is NOT a significance test. The discrimination "
                "rests on the BETWEEN-ARM separation."),
            "arc062_seed_leg": (
                "NOT UNDER TEST. No arc062_seed is supplied at any call site, so MECH-349 "
                "non-degeneracy (d) is reported as untested, never as a null."),
        },
        "claim_scope_note": (
            "MECH-350/351/352 are NOT tagged. The only quantity bearing on MECH-350 here is a "
            "structural identity under this configuration, and neither conflict nor credit is "
            "manipulated. Per-cell diagnostics ARE recorded so a future targeted run need not "
            "re-derive them. A null on any downstream committed-action or behavioural DV would be "
            "NOT FALSIFYING for MECH-349 and none is measured here."
        ),
        "arm_results": rows,
        "cell_summary": {
            arm: {
                "minted_by_seed": _by_seed(rows, arm, "crf_n_minted_total"),
                "retired_by_seed": _by_seed(rows, arm, "crf_n_retired_total"),
                "retired_persistent_by_seed": _by_seed(rows, arm, "retired_persistent"),
                "retired_extinct_by_seed": _by_seed(rows, arm, "retired_extinct"),
                "remint_persistent_by_seed": _by_seed(rows, arm, "remint_persistent"),
                "duplicate_pairs_by_seed": _by_seed(rows, arm, "duplicate_tag_pairs"),
            }
            for arm in ARMS
        },
    }

    out_path = write_flat_manifest(
        manifest, dry_run=dry_run, config=full_config, seeds=seeds,
        script_path=Path(__file__), started_at=t0,
    )
    manifest["_out_path"] = out_path
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-1025 MECH-349 CandidateRuleField churn / retirement leg")
    parser.add_argument("--dry-run", action="store_true",
                        help="1 seed, 3+3 regimes, short phases; manifest relocated out of evidence/")
    args = parser.parse_args()

    result = main(dry_run=args.dry_run)
    out_path = result["_out_path"]

    print()
    print("=== V3-EXQ-1025 MECH-349 CRF churn / retirement leg ===")
    print("label:   %s" % result["interpretation"]["label"])
    print("outcome: %s" % result["outcome"])
    print("summary: %s" % result["interpretation"]["summary"])
    print("--- per-arm precondition gate ---")
    _pag = result["per_arm_gate"]
    _failed = _pag.get("failed_preconditions_by_arm") or {}
    for arm in ARMS:
        _green = arm in (_pag.get("green_arms") or [])
        print("  %-22s green=%-5s scored=%-5s failed=%s"
              % (arm, _green, arm in SCORED_ARMS,
                 ",".join(_failed.get(arm) or []) or "-"))
    print("--- preconditions (green scored arms) ---")
    for p in result["interpretation"]["preconditions"]:
        print("  %-58s measured=%-10.4f thr=%-8.3f met=%s"
              % (p["name"], p["measured"], p["threshold"], p["met"]))
    print("--- criteria ---")
    for c in result["criteria"]:
        tag = "[LOAD-BEARING]" if c["load_bearing"] else "[recorded]    "
        d = c["detail_worst_cell"]
        print("  %-44s %s seeds=%g/%g passed=%s"
              % (c["name"], tag, c["measured"], c["threshold"], c["passed"]))
        print("        worst cell %-22s %s = %g (thr %g)"
              % (d["offending_cell"], d["statistic"], d["measured"], d["threshold"]))
    print("--- churn by arm (minted/retired : persistent/extinct retired) ---")
    for arm in ARMS:
        cs = result["cell_summary"][arm]
        vals = " ".join(
            "s%d=%s/%s:%s/%s" % (s, cs["minted_by_seed"][s], cs["retired_by_seed"][s],
                                 cs["retired_persistent_by_seed"][s],
                                 cs["retired_extinct_by_seed"][s])
            for s in sorted(cs["minted_by_seed"]))
        print("  %-22s (decay %.2f) %s" % (arm, ARM_DECAY[arm], vals))
    print("manifest: %s" % out_path)
    print("overall_outcome: %s" % result["outcome"])

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
