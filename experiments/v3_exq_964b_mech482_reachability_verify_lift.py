"""V3-EXQ-964b -- MECH-482 / SD-102: is a committed-action flip REACHABLE AT ALL through
the epistemic-deficit path, and does the corrected reachability instrument discriminate?

SUPERSEDES V3-EXQ-964a, whose C3 (yoked divergence 0/540) could not be adjudicated because
its reachability instrument was shown not to measure reachability. This run does NOT raise
`curiosity_learning_progress_weight` -- the CONFIRMED autopsy
`failure_autopsy_V3-EXQ-964a_20260914` (confirmed 2026-09-15T01:14:02Z, REE_assembly
`beb47bca09`) explicitly withdrew that recommendation. It builds the two things that autopsy's
Section 6 asks for, in its stated priority order, before any weight is chosen.

EXPERIMENT_PURPOSE = "diagnostic". This validates an INSTRUMENT and establishes reachability;
it is not evidence for MECH-482's own claim hypothesis and is excluded from governance
confidence and conflict scoring.

SLEEP: none. No sleep flag is set anywhere in this driver.

red-team (fable): see the queue entry note and the RED-TEAM RECORD at the end of this
docstring.

=== WHAT V3-EXQ-964a ESTABLISHED, AND WHAT IT COULD NOT ===

STANDS, untouched by this run: the readiness BUILD works. ARM_READINESS reached 14-16
persistent targets (C1, floor 2), the per-candidate readout genuinely varied
(`max_lp_dev_range` 4.4e-3 to 7.3e-3, C2, floor 1e-12), and ARM_PREREADINESS reproduced the
V3-EXQ-964 single-target collapse as a MEASURED control (C5). Those are re-run here only as
carried-forward context, not re-litigated.

DOES NOT STAND: the reading of C3's zero. Across 3 seeds and 540 yoked comparisons the
committed action never moved, and V3-EXQ-964a's own magnitude-reachability precondition
(`lp_perturbation_can_reach_argmax_margin`) read `met: true`. The Step 7c red-team pass
overturned that, on two grounds this run takes as its design brief:

  (i)  THE TIE PREDICATE COUNTED NEGATIVE MARGINS AS TIES.
       `v3_exq_964a_...py:458` reads `if fmargin <= 0.0:` and books the tick as
       "flip reachable". But `min_argmax_margin` was -4.141 / -1.360 / -1.129 across the
       three seeds -- strictly negative, not near-zero. Per
       `ree_core/predictors/e3_selector.py:2885` `decisiveness_margin(arbitration_aware=True)`
       returns `(best score among the OTHER candidates) - (the selected candidate's score)`,
       so a negative value means the committed candidate was NOT the score argmin: either
       arbitration had already overridden the score-based choice -- in which case perturbing
       the score cannot move the committed action, the OPPOSITE of flippable -- or
       `last_scores` is a pre-arbitration stale snapshot describing the wrong quantity.
       Either reading destroys the "reachable via tie-breaking" interpretation.
  (ii) THE COUNTERS WERE BYTE-IDENTICAL ON AN ARM WHERE A FLIP IS IMPOSSIBLE BY
       CONSTRUCTION. `n_margin_reads`, `n_flip_reachable_ticks`, `n_zero_margin_ticks` and
       `min_argmax_margin` were EXACTLY equal on ARM_PREREADINESS and ARM_READINESS at every
       seed (34/2/0/-4.1410, 55/4/0/-1.3599, 28/4/0/-1.1294), even though
       ARM_PREREADINESS's perturbation is a candidate-uniform shift that "cannot move an
       argmax at ANY magnitude". A counter that returns the identical value on a provably
       inert arm is measuring something arm-invariant.

And V3-EXQ-964 had ALREADY named the remedy 964a declined:
`failure_autopsy_V3-EXQ-964_2026-08-30.json` `learning_extracted[1]` -- "A positive control --
inject a synthetic two-target deficit and confirm the argmax flips -- would have made the zero
interpretable in one cheap step."

=== PRODUCER TRACE, RE-VERIFIED AT SOURCE BEFORE THIS WAS AUTHORED ===

The epistemic-deficit contribution genuinely reaches the committed selection; it is a live
path, not a `.get()` default:

    agent._curiosity_per_candidate_learning_progress(candidates)   agent.py:6332  [K] tensor
      -> StructuredCuriosity.compute_score_bias(per_candidate_learning_progress=...)
                                                                   agent.py:8337
      -> cur_bias -> dacc_score_bias                               agent.py:8348-8355
      -> _e3_select_kwargs["score_bias"]                           agent.py:9346
      -> E3Selector.select(): scores = scores + bias_tensor        e3_selector.py:3234
      -> self.last_scores = scores.detach()                        e3_selector.py:3731
      -> decisiveness_margin(arbitration_aware=True) reads last_scores + last_selected_idx

So the manipulation CAN reach the DV by construction. The open question this run answers is
whether it can reach it AT A MAGNITUDE THAT MOVES THE COMMITTED ACTION, and that is a
measurement, not an inference.

=== WHY THE LADDER IS OVER SELECTION AUTHORITY -- measured while authoring ===

The first draft of this driver swept the INJECTED MAGNITUDE m in {0.1, 1, 10, 100, 1000} and
its own smoke returned zero divergence at every rung, including one whose pre-clamp score-space
range was ~50 against margins of order 1. A direct probe (seed 71, 40 steps, four rungs
side by side on one yoked rollout) found why:

  `StructuredCuriosity.compute_score_bias` CLAMPS the argmin-relevant deviation to
  +/- `curiosity_bias_scale` (`structured_curiosity.py:608-639`), whose shipped default is
  **0.1** (`config.py:4646`). Measured post-clamp `_last_bias_range` and
  `_last_clamp_saturated_frac`:

      m=0.1     range 0.005   saturated 0.000
      m=1       range 0.050   saturated 0.000
      m=10      range 0.200   saturated 0.625
      m=1000    range 0.200   saturated 1.000     <- BIT-IDENTICAL to m=10

  while the SUBJECT arm's smallest POSITIVE margin on the same rollout was **0.5986** and its
  largest |margin| **4.078**.

Three consequences, all load-bearing:

  (a) A MAGNITUDE-ONLY POSITIVE CONTROL IS UNREACHABLE BY CONSTRUCTION. Above m ~ 4 the ladder
      is one saturated point, and the maximum argmin-relevant perturbation curiosity can EVER
      contribute is a range of 0.2 -- below the smallest positive margin. The first draft's C1
      could not have fired at any magnitude. Shipping it would have been exactly the
      "criterion cannot discriminate by construction" defect this run exists to repair one
      layer up.
  (b) A THIRD DEFECT IN V3-EXQ-964a'S INSTRUMENT, BEYOND THE AUTOPSY'S TWO. Its
      `max_pert_over_margin` divides `_last_lp_dev_range` -- captured at
      `structured_curiosity.py:548, BEFORE the clamp -- by the margin. On the probe's m=1000
      rung that ratio reads **39.26** while the real argmin-relevant perturbation is 0.2 and
      the margin is 1.27: the legacy statistic overstates reachability without bound whenever
      the clamp binds. The corrected instrument here divides the POST-clamp `_last_bias_range`
      instead, and records `_last_clamp_saturated_frac` so a reader can see when it bound.
  (c) THE LADDER MUST SWEEP THE AUTHORITY KNOB. So each rung sets
      `curiosity_bias_scale = s` and injects at a magnitude (1000) that rails that clamp, giving
      a realised argmin-relevant range of exactly 2*s -- a genuine dose-response in the
      quantity that decides selections.

IS RAISING `curiosity_bias_scale` "SIMPLY A HIGHER WEIGHT"? No, and the distinction is the
point. The autopsy withdrew a higher `curiosity_learning_progress_weight` as a REMEDY -- a
change you would ship. This is a POSITIVE CONTROL, lifted on the CONTROL ARMS ONLY and never
on the subject, whose whole job is to prove the detector can fire. The chip's own wording asks
for "a synthetic epistemic deficit large enough to force a committed-action flip"; under a
clamp at 0.1, "large enough" is not expressible in magnitude at all, only in authority.

=== THE TWO THINGS THIS RUN BUILDS ===

(1) A CORRECTED REACHABILITY INSTRUMENT, RUN SIDE BY SIDE WITH THE LEGACY ONE.
    Every tick records BOTH predicates on the SAME margin read:
      legacy_*    -- V3-EXQ-964a's shipped predicate verbatim: a tie is `fmargin <= 0.0`, and
                     the perturbation is the PRE-clamp `_last_lp_dev_range`.
      corrected_* -- a tie is `fmargin == 0.0` EXACTLY; a negative margin is booked to its own
                     `n_negative_margin_ticks` and NEVER counted reachable; a tie counts as
                     reachable only when the perturbation on that exact tick is non-zero; and
                     the perturbation is the POST-clamp `_last_bias_range`.
    Recording both is the point: the run DEMONSTRATES all three defects on its own ticks
    instead of citing the autopsy for two of them and asserting the third.

(2) A VERIFY-LIFT POSITIVE-CONTROL LADDER over `curiosity_bias_scale`, the knob that bounds
    curiosity's argmin-relevant authority. Rungs s in {0.1, 0.5, 2.0, 10.0} give realised
    deviation ranges {0.2, 1.0, 4.0, 20.0}. The BOTTOM rung is the SHIPPED default, so
    ARM_VERIFY_LIFT_S0.1 vs ARM_READINESS is a CONTENT-vs-AUTHORITY contrast at identical
    authority; the TOP rung is ~5x the largest |margin| the subject records, and that
    reachability is re-certified in-run by `top_rung_authority_exceeds_worst_margin` rather
    than assumed from the authoring probe.

    The injection is a wrapper on `agent._curiosity_per_candidate_learning_progress`, i.e. the
    EXACT seam the real deficit travels, so a flip there proves the real path can carry one.
    The wrapper returns None whenever the real method returns None, so every rung fires on
    exactly the ticks the subject's own mechanism fires on -- authority only, never cadence.

=== THE ARMS (1 reference + 5 followers, all yoked on ONE rollout per seed) ===

  ARM_PREREADINESS   drives the environment; the yoked reference. Imported unchanged from
                     V3-EXQ-964a (`PREREADINESS_KNOBS`). Also the collapse control.
  ARM_READINESS      THE SUBJECT. Imported unchanged (`READINESS_KNOBS`).
  ARM_VERIFY_LIFT_Ss x4  ARM_READINESS's config with `curiosity_bias_scale = s` and the
                     railing injection. ** THE POSITIVE CONTROL. **

Every follower is stepped on the reference's identical observation sequence, so every
comparison is a PAIRED argmax flip and non-compounding -- V3-EXQ-964a's own yoking discipline,
imported rather than re-derived, and re-verified each run by the self-yoked control.

=== PRE-REGISTERED CRITERIA ===

  C1 DETECTOR SENSITIVITY (load-bearing, and the gate on everything else). The TOP ladder rung
     (s=10, realised range 20.0, ~5x the subject's largest |margin|) must produce a non-zero
     yoked divergence on a seed majority. If it does not, the yoked-divergence detector cannot
     fire under this harness at ANY authority, C3 is uninterpretable for the second time, and
     the run self-routes `substrate_not_ready_requeue` -- an instrument result, never a verdict
     on MECH-482. Its own reachability is certified in-run, not assumed.
  C2 MINIMUM EFFECTIVE AUTHORITY (load-bearing). The smallest `curiosity_bias_scale` at which
     divergence first fires on a seed majority. This is what replaces the withdrawn "4-5x
     weight raise": a sizing read off a dose-response, in the units of the knob that actually
     bounds curiosity's selection authority, against a SHIPPED value of 0.1 that is the
     ladder's own bottom rung.
  C3 THE SUBJECT (load-bearing). ARM_READINESS's own yoked divergence. INTERPRETABLE ONLY IF
     C1 PASSES; that ordering is the whole reason this run exists.
  C4 INSTRUMENT ARM-SENSITIVITY (precondition, not a criterion). The CORRECTED margin counters
     must NOT be byte-identical between ARM_READINESS and ARM_PREREADINESS. V3-EXQ-964a's
     were; a repair that reproduces that signature has not repaired anything.

=== WHAT EACH OUTCOME WOULD AND WOULD NOT MEAN ===

  C1 fails -> INSTRUMENT, not substrate. The detector cannot fire; nothing about MECH-482 is
     learned, and the route is a harness repair, NOT another lettered weight bump.
  C1 passes, C3 zero -> the detector demonstrably works and the real mechanism still never
     moves the committed action. THAT is an interpretable negative, and C2's measured minimum
     effective magnitude says HOW FAR the real perturbation (~0.1 pre-weight) sits from the
     one that does move it. It does NOT by itself license a weight raise: a minimum effective
     magnitude orders of magnitude above the real one is an F-dominance finding about the
     committed-selection layer, which is a substrate question (ARC-110 / MECH-439 territory),
     not a config one.
  C1 passes, C3 non-zero -> the mechanism moves the committed action at its own magnitude,
     which is the first positive behavioural signal MECH-482 has had.

=== DV-SYMMETRY INVARIANCE, DECLARED PER ARM (mandatory) ===

The DV is the COMMITTED ACTION, an argmax over per-candidate scores. Its symmetry group is
(i) addition of a candidate-UNIFORM constant and (ii) any monotone rescaling; neither can move
an argmax.

  ARM_PREREADINESS  IS invariant under (i) -- `hard_match` with a single enclosing target
      returns the same deficit for every candidate, so `-w * lp_vec` is a uniform shift.
      DISPOSITION (b): structurally vacuous for the argmax DV, scoped OUT of divergence
      scoring, retained as the yoked reference and the collapse control. Declared at design
      time, imported verbatim from V3-EXQ-964a's own declaration.
  ARM_READINESS     NOT invariant. `rbf_weighted` is a distance-weighted sum over ALL targets,
      continuous in candidate position, so `lp_vec` varies whenever the candidates differ.
  ARM_VERIFY_LIFT_Ss NOT invariant, BY CONSTRUCTION AND ON PURPOSE. The injected vector is a
      linear ramp `m * arange(K)/(K-1)`, uniform for no m > 0; after the clamp its
      argmin-relevant deviation has range 2*s, which is likewise non-uniform for every s > 0.
      A uniform synthetic would have been a positive control that cannot possibly work -- the
      precise defect this run was built to repair one layer up. Note the clamp is applied to
      the ZERO-MEAN deviation, so it cannot flatten the vector completely (the rail ceiling is
      (K-1)/K, never 1.0) -- `structured_curiosity.py`'s own note on why that fix works.

=== KNOWN OPEN SUBSTRATE DEFECTS THIS RUN EXERCISES (Step 2.5c) ===

`sd_epistemic_deficit_multitarget_readiness` (severity `degrading`, status
`implemented_pending_validation`, `substrate_paths` `ree_core/policy/epistemic_deficit.py` +
`ree_core/policy/structured_curiosity.py`) is OPEN and is THIS RUN'S OWN SUBJECT -- its
`ready` flag stays false until this retest scores, which is what the entry's own note says.
Degrading, so recorded rather than blocking. No open `corrupting` entry names
`epistemic_deficit.py`, `structured_curiosity.py` or `e3_selector.py`.

=== RED-TEAM RECORD ===
red-team (fable, foreground, /queue-experiment Step 4.5): **CONTESTED -- 7 findings, every one
verified at source before acting, 6 FIXED and 1 RECORDED.** Two of them would have made this
run's numbers unattributable.

  F1 (A+D, HEADLINE) CONFIRMED AT SOURCE. `CausalGridWorldV2()` is constructed with NO seed at
     all three V3-EXQ-964a call sites and at this driver's, and the env's RNG is
     `np.random.default_rng(seed)` (causal_grid_world.py:1456) -- `default_rng(None)` draws OS
     entropy and ignores everything `reset_all_rng` touches. Reproduced directly: three
     `reset_all_rng(71)` + `CausalGridWorldV2()` constructions gave agent starts (1,4)/(5,7)/
     (3,5) and three different layouts. The reviewer's own five invocations measured E3-fresh
     counts of 17/21/21/24/27 and C2 flipping 0.5 <-> 0.1 at one seed. So V3-EXQ-964a's "3
     seeds" seeded the AGENTS only, on unrecorded environments. FIXED: the env is seeded
     (`CausalGridWorldV2(seed=seed)`); two consecutive dry runs now agree to 12 decimal places
     on every ladder fraction, the fresh-tick count and the headroom ratio.
  F2 (B) CONFIRMED AT SOURCE AND BY PROBE. Booking every negative margin as unreachable models
     a commit operator E3 does not use on those ticks: it commits only while
     `_running_variance < commit_threshold` and otherwise SAMPLES from softmax(-scores/T), so
     an uncommitted tick has a negative margin by construction and a perturbation still
     reshapes the draw. Probed: the subject's one negative-margin tick had
     `committed_now=False`. FIXED: negatives are partitioned by `committed_now`, and only
     COMMITTED ones are booked unreachable.
  F3 (D) CONFIRMED. The `corrected_instrument_is_arm_sensitive` gate could not fail: every key
     it compared was a function of the post-clamp perturbation, which differs between arms
     whenever `readout_differentiates_across_candidates` passes -- and the reviewer found the
     COUNT statistics (the ones V3-EXQ-964a actually found byte-identical) identical across
     arms while the gate still read met. FIXED by DEMOTING it: a precondition that cannot fail
     is decoration, and the instrument's discriminative power is demonstrated far more
     strongly by the LADDER (C1) on the DV itself. The pert-derived and count families are now
     recorded separately, with `reproduces_964a_legacy_byte_identity` per seed in the readout.
  F4 (D) CONFIRMED. The imported instrument control ran an UNSEEDED env and checked ACTION
     identity only, while this run routes on SELECTION identity. FIXED: a local
     `paired_control_divergence` on the seeded rollout, checking BOTH DVs with the same
     runner class the real arms use.
  F5 (B) CONFIRMED, RECORDED NOT FIXED. The synthetic ramp is in candidate-index order, so C2's
     number is the authority needed to drag selection toward the highest-index candidate under
     THIS geometry -- an order-of-magnitude sizing, not a transferable constant. Stated on the
     criterion itself. C1, which only needs the detector to fire, does not depend on it.
  F6 (C) CONFIRMED. The C1-fail route text described a magnitude ladder that no longer exists
     and quoted a "~10x" ratio the run measures at 3.1-4.4. FIXED: it now names the authority
     rung and points at the measured `top_rung_margin_headroom`.
  F7 (D) CONFIRMED. The denominator counted REF-fresh ticks while the numerator needs BOTH
     arms fresh. FIXED: the skew is measured and gated at exactly 0 by
     `fresh_tick_denominator_exact`.

Checked and found CLEAN by the reviewer, not re-litigated here: the manipulation genuinely
reaches the DV (the wrapper sits on the real seam; the clamp acts on the lp-only deviation;
both score-bias rescales are off); the arms share their init; and no verdict branch routes a
failing control to a substrate verdict.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell, reset_all_rng  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._metrics import (  # noqa: E402
    p0_readiness_gate, P0NotReady, dv_headroom_check,
)
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402

import experiments.v3_exq_964a_mech482_epistemic_deficit_multitarget_readiness as x964a  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_964b_mech482_reachability_verify_lift"
QUEUE_ID = "V3-EXQ-964b"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS = ["MECH-482"]
SUPERSEDES = "V3-EXQ-964a"

# Every readiness predicate here is anchored to a FROZEN RECORDED control from V3-EXQ-964a, to
# an exact identity, or is certified in-run -- and the one whose reachability is the actual
# question, `top_rung_authority_exceeds_worst_margin`, IS ITSELF the reachability certificate
# for C1, measured on the control arm against the subject's own margins rather than asserted.
# `paired_control_is_bit_identical` and `injected_deficit_differentiates_candidates` are exact
# identities of the harness and the injected ramp; `multitarget_regime_reached`,
# `readout_differentiates_across_candidates` and `vacuous_readout_rate_bounded` carry
# V3-EXQ-964a's own measured values (14-16 targets, 4.4e-3..7.3e-3, rate 0) with these same
# imported knobs; `e3_fresh_ticks_sufficient` and `corrected_instrument_is_arm_sensitive` were
# both verified reachable on the authoring probe (21 fresh ticks in 2 episodes; the corrected
# counters differ across arms where V3-EXQ-964a's legacy ones were byte-identical). No
# hand-written predicate here is narrower than the state it anchors to.
ANCHOR_REACHABILITY_EXEMPT = (
    "every readiness predicate is anchored either to V3-EXQ-964a's own recorded control values "
    "with these same imported knobs, or to an exact identity of the harness, or is measured "
    "in-run; and the one predicate that IS a reachability question -- "
    "top_rung_authority_exceeds_worst_margin -- is itself the C1 reachability certificate, "
    "computed on the control arm against the SUBJECT's margins so it does not certify its own "
    "subject. All were verified reachable on a real-config authoring probe before queueing."
)

# ---- imported from V3-EXQ-964a, never re-defined --------------------------------------
SEEDS = list(x964a.SEEDS)                    # [71, 101, 202]
EPISODES = x964a.EPISODES                    # 3
STEPS_PER_EPISODE = x964a.STEPS_PER_EPISODE  # 60
ARM_PRE = x964a.ARM_PRE
ARM_RDY = x964a.ARM_RDY
SEED_MAJORITY = 2                            # of 3, x1002's own convention

# ---- PRE-REGISTERED CONSTANTS (never derived from this run's statistics) ---------------
# THE LADDER SWEEPS `curiosity_bias_scale`, NOT THE INJECTED MAGNITUDE -- see "WHY THE LADDER
# IS OVER SELECTION AUTHORITY" in the docstring. Measured while authoring: the curiosity clamp
# (structured_curiosity.py:608-620) bounds the ARGMIN-RELEVANT deviation to
# +/- curiosity_bias_scale, so a magnitude ladder saturates and m=10 and m=1000 produce a
# BIT-IDENTICAL post-clamp range of 0.2 (clamp_saturated_frac 0.625 -> 1.0). A magnitude-only
# positive control is therefore unreachable by construction, which is the defect class this
# whole run exists to repair one layer up.
#
# Rungs are the realised argmin-relevant range's HALF-WIDTH. The bottom rung is the SHIPPED
# default 0.1 (config.py:4646) -- the authority the real mechanism actually has -- and the
# ladder spans two decades above it. Measured on the authoring probe: the subject's smallest
# POSITIVE margin is 0.5986 and its largest |margin| is 4.078, so range 0.2 (s=0.1) cannot
# flip, range 1.0 (s=0.5) clears the smallest positive margin, and range 20.0 (s=10) exceeds
# every margin observed by ~5x. The top rung is therefore reachable, and that reachability is
# re-certified in-run by `top_rung_authority_exceeds_worst_margin`.
VERIFY_LIFT_LADDER: List[float] = [0.1, 0.5, 2.0, 10.0]
# The injected magnitude, held CONSTANT across rungs and chosen to rail the clamp at every
# rung, so each rung's realised argmin-relevant range is exactly 2 * s and the ladder is a
# dose-response in AUTHORITY alone. 1000 * curiosity_learning_progress_weight (0.05) = 50,
# which rails even the top rung's +/-10.
INJECTED_MAGNITUDE = 1000.0
# C1/C2/C3: a divergence fraction strictly above this counts as the detector having fired.
# Imported rather than re-typed -- it is the same floor V3-EXQ-964a's C3 used.
DIVERGENCE_FLOOR = x964a.DIVERGENCE_FLOOR                # 1e-9
# The self-yoked instrument control must be exactly bit-identical.
CONTROL_DIVERGENCE_CEILING = x964a.CONTROL_DIVERGENCE_CEILING   # 1e-9
# C4: the corrected margin counters must differ between the subject and the provably-inert
# reference on at least one of the four recorded statistics. V3-EXQ-964a's were identical on
# all four at all three seeds, which is the signature this gate exists to refuse.
ARM_SENSITIVITY_FLOOR = 1.0    # at least 1 seed showing a difference
# SAMPLE SIZE. The selection DV is denominated on E3-FRESH ticks, not on env steps: E3 selects
# on roughly 1 tick in 10 (`heartbeat.e3_steps_per_tick`), so V3-EXQ-964a's "540 comparisons"
# were ~117 margin reads wearing a 5x-inflated denominator. Measured while authoring: 11
# E3-fresh ticks in 60 steps on one seed, so 3 episodes x 60 steps gives ~33 per seed. The
# floor is deliberately modest -- its job is to refuse a run that measured almost nothing, not
# to assert power.
E3_FRESH_TICKS_FLOOR = 15.0
# C1 reachability certificate: the TOP rung's realised post-clamp (argmin-relevant) range must
# exceed the largest |margin| the SUBJECT arm records, measured in-run. This is what makes C1
# reachable rather than merely hoped for -- and it is measured on the CONTROL arm against the
# SUBJECT's margins, so it does not certify its own subject.
TOP_RUNG_MARGIN_HEADROOM = 1.0   # ratio: top-rung range / worst |margin|, floor
# The injected ramp must actually differentiate candidates, else the positive control is a
# uniform shift and cannot move an argmax at any magnitude -- the exact defect being repaired.
INJECTED_RANGE_FLOOR = 1e-12

DRY_RUN_SEEDS = [SEEDS[0]]
DRY_RUN_EPISODES = 2
# Sized so the smoke reaches enough E3-FRESH ticks for the selection DV to be exercised at all.
# E3 selects on ~1 tick in 10, so a 12-step smoke yielded 1-2 fresh ticks and could not
# distinguish "the detector did not fire" from "the detector was never asked" -- the first
# draft's smoke read `detector_cannot_fire_instrument_defect` purely from that. 60 steps
# measured 11 fresh ticks on one seed, so the smoke runs 2 episodes to clear
# E3_FRESH_TICKS_FLOOR at the REAL threshold -- no gate in this driver is relaxed under
# --dry-run.
DRY_RUN_STEPS = 60
DRY_RUN_CONTROL_STEPS = 6
CONTROL_STEPS = 15

_EXTRA_SUBSTRATE_PATHS = [Path(x964a.__file__)]


def _lift_arm_id(s_scale: float) -> str:
    return "ARM_VERIFY_LIFT_S%g" % (s_scale,)


VERIFY_LIFT_ARM_IDS = [_lift_arm_id(m) for m in VERIFY_LIFT_LADDER]
FOLLOWER_ARM_IDS = [ARM_RDY] + VERIFY_LIFT_ARM_IDS


# --------------------------------------------------------------------------------------
# THE CORRECTED INSTRUMENT
# --------------------------------------------------------------------------------------
class _InstrumentedRunner(x964a._Runner):
    """V3-EXQ-964a's runner with BOTH reachability predicates recorded per tick.

    `_choose_inner` is overridden rather than extended because the margin block IS the
    instrument under repair -- the whole point of this run is that it changes. The rest of the
    tick (sense / clock / trajectories / z_goal / the latch-clear discipline) is reproduced
    exactly as V3-EXQ-964a runs it, so the only difference between the two drivers at this
    layer is the predicate.

    LEGACY counters reproduce V3-EXQ-964a's shipped predicate verbatim (`fmargin <= 0.0` is a
    tie) so its defect is DEMONSTRATED on these ticks rather than asserted. CORRECTED counters
    apply the autopsy's Section 6 item 1: a tie is `== 0.0` exactly, a negative margin is
    booked separately and never counted reachable, and a tie counts as reachable only when the
    differential perturbation on that same tick is non-zero.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        # legacy (V3-EXQ-964a verbatim) -- the base class already owns n_margin_reads,
        # n_flip_reachable_ticks, n_zero_margin_ticks, n_flip_reachable_strict,
        # min_argmax_margin and max_pert_over_margin; those stay the LEGACY set.
        # corrected
        self.c_n_exact_tie_ticks = 0
        self.c_n_negative_margin_ticks = 0
        # RED-TEAM F2: a negative margin is NOT always an arbitration override. E3 commits
        # only while `_running_variance < commit_threshold` and otherwise SAMPLES the
        # selection from softmax(-scores/T), so on an uncommitted tick the selected candidate
        # is generically not the argmin and the margin is negative with no override at all --
        # and a perturbation CAN move a sampled choice by reshaping the softmax. Measured on
        # the authoring probe: the one negative-margin tick had committed_now=False. So the
        # negatives are partitioned, and only the COMMITTED ones are booked unreachable.
        self.c_n_negative_margin_committed = 0
        self.c_n_negative_margin_sampled = 0
        self.c_n_committed_ticks = 0
        self.c_n_positive_margin_ticks = 0
        self.c_n_flip_reachable_strict = 0
        self.c_n_flip_reachable_total = 0
        self.c_max_pert_over_margin = 0.0
        self.c_min_positive_margin = float("inf")
        self.n_injected_ticks = 0
        self.max_injected_range = 0.0
        self.max_post_clamp_bias_range = 0.0
        self.max_clamp_saturated_frac = 0.0
        self.max_abs_margin = 0.0
        # Per-tick handles for the yoked comparison. `last_tick_e3_fresh` is True exactly on
        # the ticks where select() ran (the latch-clear discipline proves it), which is the
        # only denominator on which a selection comparison means anything.
        self.last_tick_e3_fresh = False
        self.last_tick_selected_idx: Optional[int] = None
        self.n_e3_fresh = 0

    def _choose_inner(self, obs: Dict[str, Any]) -> int:
        agent = self.agent
        latent = agent.sense(obs["body_state"], obs["world_state"])
        ticks = agent.clock.advance()
        e1_prior = (
            agent._e1_tick(latent)
            if ticks.get("e1_tick")
            else torch.zeros(1, self.world_dim, device=agent.device)
        )
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        if candidates:
            self.n_ticks += 1
        agent.update_z_goal(
            benefit_exposure=0.0,
            drive_level=REEAgent.compute_drive_level(obs["body_state"]),
        )
        # CLEAR THE LATCH immediately before the call (V3-EXQ-964a's discipline, unchanged):
        # compute_score_bias assigns _last_lp_dev_range on every path it runs, so a value
        # still None afterwards proves it did NOT run this tick.
        cur = getattr(agent, "curiosity", None)
        if cur is not None:
            cur._last_lp_dev_range = None
        self.last_tick_e3_fresh = False
        self.last_tick_selected_idx = None
        action = agent.select_action(candidates, ticks)
        if cur is not None:
            val = cur._last_lp_dev_range
            if val is None:
                self.n_latched_ticks += 1
            else:
                self.n_lp_reads += 1
                self.last_tick_e3_fresh = True
                self.n_e3_fresh += 1
                _e3sel = getattr(agent, "e3", None)
                _si = None if _e3sel is None else getattr(_e3sel, "last_selected_idx", None)
                self.last_tick_selected_idx = (None if _si is None else int(_si))
                fval = float(val)
                if math.isfinite(fval):
                    if fval > self.max_lp_dev_range:
                        self.max_lp_dev_range = fval
                    if fval > 0.0:
                        self.n_positive_lp_dev_range += 1
                acc_now = getattr(agent, "epistemic_deficit", None)
                if acc_now is not None:
                    nd = int(getattr(
                        acc_now, "_last_readout_n_distinct_targets", 0) or 0)
                    if nd > self.max_distinct_matched:
                        self.max_distinct_matched = nd
                e3 = getattr(agent, "e3", None)
                margin = None
                if e3 is not None:
                    try:
                        margin = e3.decisiveness_margin(arbitration_aware=True)
                    except Exception:
                        margin = None
                if margin is not None and math.isfinite(float(margin)):
                    fmargin = float(margin)
                    # LEGACY perturbation: `_last_lp_dev_range`, computed from lp_contrib,
                    # which is ALREADY curiosity_learning_progress_weight * lp_vec
                    # (structured_curiosity.py:548). V3-EXQ-964a used this and it is the
                    # THIRD defect in that instrument (see the docstring): it is measured
                    # BEFORE the curiosity clamp, so whenever the clamp binds it overstates
                    # the argmin-relevant perturbation without bound.
                    pert_legacy = fval
                    # CORRECTED perturbation: `_last_bias_range`, the range of the CLAMPED,
                    # zero-mean deviation -- the only part of the curiosity bias that can move
                    # an argmax at all (structured_curiosity.py:608-639). Measured while
                    # authoring: at an injected magnitude of 10 and of 1000 the legacy value
                    # differs 100x while this one is BIT-IDENTICAL at 0.2, with
                    # `_last_clamp_saturated_frac` 0.625 and 1.0.
                    _br = getattr(cur, "_last_bias_range", None)
                    pert_corrected = (float(_br) if _br is not None
                                      and math.isfinite(float(_br)) else 0.0)
                    _sat = getattr(cur, "_last_clamp_saturated_frac", None)
                    if _sat is not None and math.isfinite(float(_sat)):
                        if float(_sat) > self.max_clamp_saturated_frac:
                            self.max_clamp_saturated_frac = float(_sat)
                    if pert_corrected > self.max_post_clamp_bias_range:
                        self.max_post_clamp_bias_range = pert_corrected
                    if abs(fmargin) > self.max_abs_margin:
                        self.max_abs_margin = abs(fmargin)
                    committed = None
                    if e3 is not None:
                        try:
                            committed = bool(
                                e3.get_commitment_state().get("committed_now"))
                        except Exception:
                            committed = None
                    if committed:
                        self.c_n_committed_ticks += 1
                    self._record_legacy(fmargin, pert_legacy)
                    self._record_corrected(fmargin, pert_corrected, committed)
                # Restore a real float so nothing downstream sees the sentinel.
                cur._last_lp_dev_range = fval
            if cur._last_lp_dev_range is None:
                cur._last_lp_dev_range = 0.0
        return int(action.argmax(dim=-1).item())

    def _record_legacy(self, fmargin: float, pert: float) -> None:
        """V3-EXQ-964a's shipped predicate, VERBATIM. Kept so this run demonstrates the
        defect on its own ticks instead of citing the autopsy for it."""
        self.n_margin_reads += 1
        if fmargin < self.min_argmax_margin:
            self.min_argmax_margin = fmargin
        if fmargin <= 0.0:
            self.n_zero_margin_ticks += 1
            self.n_flip_reachable_ticks += 1
        else:
            ratio = pert / fmargin
            if ratio > self.max_pert_over_margin:
                self.max_pert_over_margin = ratio
            if pert >= fmargin:
                self.n_flip_reachable_strict += 1
                self.n_flip_reachable_ticks += 1

    def _record_corrected(self, fmargin: float, pert: float,
                          committed: Optional[bool]) -> None:
        """The autopsy Section 6 item 1 predicate.

        Three changes from legacy, each answering a named finding:
          - a tie is `== 0.0` EXACTLY. A negative margin means the committed candidate was not
            the score argmin (arbitration override, or a stale pre-arbitration snapshot);
            under either reading perturbing the score cannot move the committed action, so it
            is booked to its own counter and NEVER counted reachable.
          - a tie counts as reachable only when the differential perturbation on that exact
            tick is non-zero: a uniform perturbation cannot break a tie either.
          - `max_pert_over_margin` ranges over POSITIVE margins only, so the ratio means what
            its name says rather than being polluted by sign.
          - negatives are PARTITIONED by `committed_now`: only a negative margin on a
            COMMITTED tick is an override and therefore unreachable. On an uncommitted tick
            E3 samples, so a negative margin is expected and says nothing about reachability
            (red-team F2, confirmed at source).
          - and, the change beyond the autopsy's own two: `pert` is the POST-CLAMP
            argmin-relevant deviation range, not the pre-clamp `_last_lp_dev_range`. Under the
            shipped `curiosity_bias_scale` the clamp binds hard, so the legacy quantity
            overstates reachability by the saturation factor.
        """
        if fmargin == 0.0:
            self.c_n_exact_tie_ticks += 1
            if pert > INJECTED_RANGE_FLOOR:
                self.c_n_flip_reachable_total += 1
        elif fmargin < 0.0:
            self.c_n_negative_margin_ticks += 1
            if committed:
                # A genuine arbitration/commit override: the committed candidate is not the
                # score argmin AND the selection was deterministic, so perturbing the score
                # cannot move it. This is the only case V3-EXQ-964a's reading assumed.
                self.c_n_negative_margin_committed += 1
            else:
                # SAMPLED selection. The choice was drawn from softmax(-scores/T), so the
                # margin is negative by construction rather than by override and the
                # perturbation still reshapes the draw. Booked separately and NOT counted
                # unreachable -- but not counted reachable either, because a stochastic flip
                # is not a deterministic one and the ladder is what settles reachability.
                self.c_n_negative_margin_sampled += 1
        else:
            self.c_n_positive_margin_ticks += 1
            if fmargin < self.c_min_positive_margin:
                self.c_min_positive_margin = fmargin
            ratio = pert / fmargin
            if ratio > self.c_max_pert_over_margin:
                self.c_max_pert_over_margin = ratio
            if pert >= fmargin:
                self.c_n_flip_reachable_strict += 1
                self.c_n_flip_reachable_total += 1

    def instrument_summary(self) -> Dict[str, Any]:
        return {
            "legacy_n_margin_reads": self.n_margin_reads,
            "legacy_n_zero_margin_ticks": self.n_zero_margin_ticks,
            "legacy_n_flip_reachable_ticks": self.n_flip_reachable_ticks,
            "legacy_n_flip_reachable_strict": self.n_flip_reachable_strict,
            "legacy_min_argmax_margin": (
                None if not math.isfinite(self.min_argmax_margin)
                else self.min_argmax_margin),
            "legacy_max_pert_over_margin": self.max_pert_over_margin,
            "corrected_n_exact_tie_ticks": self.c_n_exact_tie_ticks,
            "corrected_n_negative_margin_ticks": self.c_n_negative_margin_ticks,
            "corrected_n_negative_margin_committed": self.c_n_negative_margin_committed,
            "corrected_n_negative_margin_sampled": self.c_n_negative_margin_sampled,
            "corrected_n_committed_ticks": self.c_n_committed_ticks,
            "corrected_n_positive_margin_ticks": self.c_n_positive_margin_ticks,
            "corrected_n_flip_reachable_strict": self.c_n_flip_reachable_strict,
            "corrected_n_flip_reachable_total": self.c_n_flip_reachable_total,
            "corrected_max_pert_over_margin": self.c_max_pert_over_margin,
            "corrected_min_positive_margin": (
                None if not math.isfinite(self.c_min_positive_margin)
                else self.c_min_positive_margin),
            "n_injected_ticks": self.n_injected_ticks,
            "max_injected_range": self.max_injected_range,
            "max_post_clamp_bias_range": self.max_post_clamp_bias_range,
            "max_clamp_saturated_frac": self.max_clamp_saturated_frac,
            "max_abs_margin": self.max_abs_margin,
            "n_e3_fresh_ticks": self.n_e3_fresh,
        }


class _VerifyLiftRunner(_InstrumentedRunner):
    """ARM_READINESS's agent with a SYNTHETIC per-candidate deficit injected at the exact
    seam the real deficit travels.

    The wrapper sits on `agent._curiosity_per_candidate_learning_progress` (agent.py:6332),
    whose return value is handed straight to
    `StructuredCuriosity.compute_score_bias(per_candidate_learning_progress=...)`
    (agent.py:8337) and from there into E3's `score_bias`. So a flip produced here is a flip
    the REAL mechanism's own path can carry -- which is what makes this a control for THIS
    mechanism rather than a demonstration that some other input can move an argmax.

    IT RETURNS None WHENEVER THE REAL METHOD DOES. The real method refuses on a tick whose
    readiness gate is shut, and a control that fired on those ticks too would be a CADENCE
    manipulation as well as a magnitude one, and no longer paired with the subject.
    """

    def __init__(self, cfg: Any, magnitude: float = INJECTED_MAGNITUDE) -> None:
        super().__init__(cfg)
        self.magnitude = float(magnitude)
        agent = self.agent
        real = agent._curiosity_per_candidate_learning_progress

        def _injected(candidates: Any) -> Optional[torch.Tensor]:
            v = real(candidates)
            if v is None:
                return None            # same tick set as the subject -- magnitude only
            k = int(v.numel())
            if k < 2:
                return v               # a single candidate has no cross-candidate range
            ramp = torch.arange(k, dtype=v.dtype, device=v.device) / float(k - 1)
            out = ramp * self.magnitude
            self.n_injected_ticks += 1
            rng = float(out.max().item() - out.min().item())
            if rng > self.max_injected_range:
                self.max_injected_range = rng
            return out

        agent._curiosity_per_candidate_learning_progress = _injected


# --------------------------------------------------------------------------------------
# YOKED GROUP
# --------------------------------------------------------------------------------------
def _build_follower(arm_id: str) -> _InstrumentedRunner:
    """ARM_READINESS, or a verify-lift rung differing from it in ONE knob.

    A rung sets `curiosity_bias_scale = s` (the clamp that bounds curiosity's ARGMIN-RELEVANT
    authority, `structured_curiosity.py:608-620`) and injects at a magnitude that rails that
    clamp, so its realised per-candidate deviation range is exactly 2*s. Everything else --
    the readiness knobs, the learning-progress weight, the env, the seed -- is
    `x964a.build_config(readiness=True)` unchanged, so a rung differs from the SUBJECT in
    selection AUTHORITY alone.

    The bottom rung s=0.1 is the SHIPPED default, so ARM_VERIFY_LIFT_S0.1 vs ARM_READINESS is
    a clean CONTENT-vs-AUTHORITY contrast at identical authority: if even a railed synthetic
    deficit cannot move the action at the shipped clamp, the shortfall is authority, not the
    deficit's content.
    """
    if arm_id == ARM_RDY:
        return _InstrumentedRunner(x964a.build_config(readiness=True))
    s_scale = float(arm_id.rsplit("_S", 1)[1])
    cfg = x964a.build_config(readiness=True)
    cfg.curiosity_bias_scale = float(s_scale)
    return _VerifyLiftRunner(cfg, INJECTED_MAGNITUDE)


def paired_control_divergence(seed: int, readiness: bool, episodes: int,
                             steps: int) -> Dict[str, float]:
    """INSTRUMENT CONTROL: yoke an arm against ITSELF. Both DVs must be exactly 0.

    LOCAL rather than imported from V3-EXQ-964a, for two reasons the red-team pass named:
      (F1) that one constructs `CausalGridWorldV2()` UNSEEDED, so it certifies a different
           (entropy-drawn) rollout from the one the DV is measured on.
      (F4) it compares ACTION identity only, while this run routes on SELECTION identity --
           and selection flips outnumber action flips by roughly an order of magnitude here,
           so an action-identity control is the weaker of the two claims.
    Both DVs are checked, on the seeded rollout, with the same `_InstrumentedRunner` the real
    arms use.
    """
    reset_all_rng(seed)
    a = _InstrumentedRunner(x964a.build_config(readiness))
    reset_all_rng(seed)
    b = _InstrumentedRunner(x964a.build_config(readiness))
    reset_all_rng(seed)
    env = CausalGridWorldV2(seed=seed)
    n = n_fresh = d_act = d_sel = 0
    for _ in range(episodes):
        _, obs = env.reset()
        a.reset_episode()
        b.reset_episode()
        for _ in range(steps):
            aa = a.choose(obs)
            bb = b.choose(obs)
            n += 1
            if aa != bb:
                d_act += 1
            if a.last_tick_e3_fresh and b.last_tick_e3_fresh:
                n_fresh += 1
                if a.last_tick_selected_idx != b.last_tick_selected_idx:
                    d_sel += 1
            _f, harm, _dn, _i, obs = env.step(aa)
            a.observe(harm)
            b.observe(harm)
    return {
        "action_divergence_frac": (d_act / n) if n else 0.0,
        "selection_divergence_frac": (d_sel / n_fresh) if n_fresh else 0.0,
        "n_compared": float(n), "n_e3_fresh_compared": float(n_fresh),
    }


def run_yoked_group(seed: int, episodes: int, steps: int,
                    zg: ZGoalStreamAccumulator) -> Dict[str, Any]:
    """ONE reference-driven rollout; every follower stepped on its observation sequence.

    This is V3-EXQ-964a's `run_yoked_pair` generalised from one follower to N. Each runner is
    constructed after its own `reset_all_rng(seed)` and owns its own RNG stream (the base
    class's discipline), so adding followers cannot perturb the reference's rollout or any
    other follower's -- which is what keeps every comparison paired and the ladder internally
    comparable. Verified by the self-yoked instrument control below.
    """
    reset_all_rng(seed)
    ref = _InstrumentedRunner(x964a.build_config(readiness=False))
    followers: Dict[str, _InstrumentedRunner] = {}
    for arm_id in FOLLOWER_ARM_IDS:
        reset_all_rng(seed)
        followers[arm_id] = _build_follower(arm_id)
    reset_all_rng(seed)

    # RED-TEAM F1, CONFIRMED AT SOURCE AND CORRECTED HERE. V3-EXQ-964a constructs
    # `CausalGridWorldV2()` with NO seed at all three of its call sites (:293, :558, :611), and
    # the env's own RNG is `np.random.default_rng(seed)` (causal_grid_world.py:1456), so
    # `default_rng(None)` draws from OS entropy and ignores every RNG `reset_all_rng` touches.
    # Measured: three `reset_all_rng(71)` + `CausalGridWorldV2()` constructions produced agent
    # starts (1,4) / (5,7) / (3,5) and three different resource layouts. V3-EXQ-964a's "3
    # seeds" therefore seeded the AGENTS only, on three unrecorded environments, and its
    # per-invocation numbers were not reproducible. This driver passes the seed.
    #
    # CONSEQUENCE, stated rather than buried: this run's rollouts are NOT V3-EXQ-964a's. Its
    # recorded numbers are carried forward here as CONTEXT (diagnostics.ref_964a) and never as
    # a comparator, and the criteria are all internal to this run.
    env = CausalGridWorldV2(seed=seed)
    n_cmp = 0
    n_fresh_cmp = 0
    n_diff = {a: 0 for a in FOLLOWER_ARM_IDS}          # ACTION divergence, all ticks
    n_sel_diff = {a: 0 for a in FOLLOWER_ARM_IDS}      # SELECTION divergence, E3-fresh ticks
    n_act_diff_fresh = {a: 0 for a in FOLLOWER_ARM_IDS}
    for ep in range(episodes):
        _, obs = env.reset()
        ref.reset_episode()
        for f in followers.values():
            f.reset_episode()
        ep_cmp = 0
        ep_diff = {a: 0 for a in FOLLOWER_ARM_IDS}
        for _ in range(steps):
            a_ref = ref.choose(obs)
            ref_fresh = ref.last_tick_e3_fresh
            ref_sel = ref.last_tick_selected_idx
            if ref_fresh:
                n_fresh_cmp += 1
            for arm_id, f in followers.items():
                a_f = f.choose(obs)
                if a_f != a_ref:
                    n_diff[arm_id] += 1
                    ep_diff[arm_id] += 1
                # THE MECHANISM-FAITHFUL DV. The perturbation acts on CANDIDATE SELECTION; the
                # committed action is a lossy projection of it (K=32 candidates over a 5-action
                # space, so distinct candidates routinely share a first action). Measured while
                # authoring at the top rung: 5 selection flips on 11 E3-fresh ticks produced
                # only 1 action flip. V3-EXQ-964a recorded ONLY the action projection, which
                # is a fourth, previously-unnamed reason its 0/540 was hard to read. Compared
                # ONLY on ticks where BOTH arms' select() actually ran -- last_selected_idx
                # latches, so an unfresh tick would re-compare the previous selection.
                if ref_fresh and f.last_tick_e3_fresh:
                    if (ref_sel is not None and f.last_tick_selected_idx is not None
                            and f.last_tick_selected_idx != ref_sel):
                        n_sel_diff[arm_id] += 1
                    if a_f != a_ref:
                        n_act_diff_fresh[arm_id] += 1
            n_cmp += 1
            ep_cmp += 1
            _f, harm, _d, _i, obs = env.step(a_ref)
            ref.observe(harm)
            for f in followers.values():
                f.observe(harm)
        ref.snapshot_episode()
        for f in followers.values():
            f.snapshot_episode()
        print("  [train] yoked seed=%d ep %d/%d fresh=%d sel_rdy=%d sel_top=%d "
              "act_rdy=%d/%d" % (seed, ep + 1, episodes, n_fresh_cmp,
                                 n_sel_diff[ARM_RDY], n_sel_diff[VERIFY_LIFT_ARM_IDS[-1]],
                                 ep_diff[ARM_RDY], ep_cmp), flush=True)
    zg.observe(ref.agent)
    for f in followers.values():
        zg.observe(f.agent)

    out: Dict[str, Any] = {
        "seed": int(seed),
        "n_compared": n_cmp,
        "n_e3_fresh_compared": n_fresh_cmp,
        ARM_PRE: {
            "accumulator": ref.accumulator_summary(),
            "instrument": ref.instrument_summary(),
        },
    }
    for arm_id, f in followers.items():
        out[arm_id] = {
            "accumulator": f.accumulator_summary(),
            "instrument": f.instrument_summary(),
            # V3-EXQ-964a's own DV, carried forward unchanged so the two runs are comparable.
            "yoked_n_compared": n_cmp,
            "yoked_n_diverged": n_diff[arm_id],
            "yoked_divergence_frac": (n_diff[arm_id] / n_cmp) if n_cmp else 0.0,
            # THE DV THIS RUN ROUTES ON, denominated on E3-fresh ticks only.
            "yoked_n_e3_fresh_compared": n_fresh_cmp,
            "yoked_n_selection_diverged": n_sel_diff[arm_id],
            "yoked_selection_divergence_frac": (
                (n_sel_diff[arm_id] / n_fresh_cmp) if n_fresh_cmp else 0.0),
            "yoked_n_action_diverged_fresh": n_act_diff_fresh[arm_id],
            "yoked_action_divergence_frac_fresh": (
                (n_act_diff_fresh[arm_id] / n_fresh_cmp) if n_fresh_cmp else 0.0),
        }
    return out


# --------------------------------------------------------------------------------------
def _arm_sensitivity(pair: Dict[str, Any]) -> Dict[str, Any]:
    """C4: do the CORRECTED counters differ between the subject and the inert reference?

    V3-EXQ-964a's legacy counters were byte-identical on both at every seed, which is how its
    reachability reading was overturned. Recording the legacy delta alongside the corrected
    one means the run says whether it reproduced that signature, rather than assuming it did
    not.
    """
    # RED-TEAM F3, CONFIRMED. The pert-DERIVED keys differ between arms whenever the subject's
    # readout differentiates at all -- which `readout_differentiates_across_candidates` already
    # gates -- so a sensitivity test over them could not fail and was decoration. The COUNT
    # partition is the statistic V3-EXQ-964a actually found byte-identical, so the two families
    # are recorded SEPARATELY and neither is a gate: the instrument's discriminative power is
    # demonstrated by the LADDER (C1), on the DV itself, which is a stronger claim than any
    # counter comparison.
    keys_pert = ("corrected_max_pert_over_margin", "corrected_min_positive_margin",
                 "max_post_clamp_bias_range")
    keys_count = ("corrected_n_exact_tie_ticks", "corrected_n_negative_margin_ticks",
                  "corrected_n_negative_margin_committed",
                  "corrected_n_negative_margin_sampled",
                  "corrected_n_positive_margin_ticks")
    keys_corrected = keys_pert + keys_count
    keys_legacy = ("legacy_n_margin_reads", "legacy_n_zero_margin_ticks",
                   "legacy_n_flip_reachable_ticks", "legacy_min_argmax_margin")
    rdy = pair[ARM_RDY]["instrument"]
    ref = pair[ARM_PRE]["instrument"]

    def _diff(keys):
        return sorted(k for k in keys if rdy.get(k) != ref.get(k))

    d_c, d_l = _diff(keys_corrected), _diff(keys_legacy)
    d_p, d_n = _diff(keys_pert), _diff(keys_count)
    return {
        "seed": pair["seed"],
        "corrected_keys_differing": d_c,
        "n_corrected_keys_differing": len(d_c),
        "corrected_pert_keys_differing": d_p,
        "n_corrected_pert_keys_differing": len(d_p),
        "corrected_count_keys_differing": d_n,
        "n_corrected_count_keys_differing": len(d_n),
        "legacy_keys_differing": d_l,
        "n_legacy_keys_differing": len(d_l),
        "reproduces_964a_legacy_byte_identity": bool(not d_l),
    }


def _flat_scalar(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, bool):
            out[k] = int(v)
        elif isinstance(v, (int, float)):
            f = float(v)
            if math.isfinite(f):
                out[k] = v
    return out


def _n_seeds_firing(pairs: List[Dict[str, Any]], arm_id: str,
                    key: str = "yoked_selection_divergence_frac") -> int:
    return sum(1 for p in pairs if float(p[arm_id][key]) > DIVERGENCE_FLOOR)


def run_experiment(episodes: int, steps: int, seeds: List[int],
                   dry_run: bool) -> Dict[str, Any]:
    t0 = time.perf_counter()
    zg = ZGoalStreamAccumulator()
    majority = min(SEED_MAJORITY, len(seeds))

    # INSTRUMENT CONTROL, imported from V3-EXQ-964a: an arm yoked against ITSELF must diverge
    # on exactly 0 ticks, else the two yoked agents differ in something other than the
    # manipulation and every divergence number in this run is void.
    ctrl_steps = DRY_RUN_CONTROL_STEPS if dry_run else CONTROL_STEPS
    control_div: Dict[str, Dict[str, float]] = {}
    for readiness, aid in ((False, ARM_PRE), (True, ARM_RDY)):
        control_div[aid] = paired_control_divergence(
            seeds[0], readiness, episodes=1, steps=ctrl_steps)
        print("[control] %s: self-yoked action=%.6f selection=%.6f (both must be 0)"
              % (aid, control_div[aid]["action_divergence_frac"],
                 control_div[aid]["selection_divergence_frac"]), flush=True)

    pairs: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        print("Seed %d Condition yoked_group" % (seed,), flush=True)
        slice_ = {
            "seed_group": "yoked",
            "episodes": episodes,
            "steps_per_episode": steps,
            "follower_arms": list(FOLLOWER_ARM_IDS),
            "verify_lift_ladder": list(VERIFY_LIFT_LADDER),
            # Readout-affecting and therefore IN the slice: a consumer running a different
            # injected magnitude computes different cells, and an under-approximated slice is
            # a false-cache-HIT bug (arm_reuse_fingerprint_plan.md 7b).
            "injected_magnitude": float(INJECTED_MAGNITUDE),
            "readiness_knobs": dict(x964a.READINESS_KNOBS),
            "prereadiness_knobs": dict(x964a.PREREADINESS_KNOBS),
        }
        # ONE cell per seed: the whole yoked group is a single indivisible computation (every
        # follower reads the reference's own rollout), so splitting it per arm would emit
        # fingerprints for cells that were never independently computable.
        with arm_cell(seed, config_slice=slice_, script_path=Path(__file__),
                      config_slice_declared=True, include_driver_script_in_hash=False,
                      extra_substrate_paths=_EXTRA_SUBSTRATE_PATHS,
                      extra_ineligible_reasons=[
                          "followers share the reference arm's rollout and env RNG"]) as cell:
            pair = run_yoked_group(seed, episodes, steps, zg)
            row: Dict[str, Any] = {
                "arm_id": "yoked_group", "seed": int(seed),
                "n_compared": pair["n_compared"],
                "arm_sensitivity": _arm_sensitivity(pair),
            }
            for aid in [ARM_PRE] + FOLLOWER_ARM_IDS:
                row[aid] = pair[aid]
            cell.stamp(row)
        pairs.append(pair)
        rows.append(row)
        print("verdict: %s" % ("PASS" if pair["n_compared"] > 0 else "FAIL"), flush=True)

    # ---- preconditions ---------------------------------------------------------------
    ctrl_worst = max(
        (max(v["action_divergence_frac"], v["selection_divergence_frac"])
         for v in control_div.values()), default=1.0)
    inj_worst = min(
        (float(p[a]["instrument"]["max_injected_range"])
         for p in pairs for a in VERIFY_LIFT_ARM_IDS),
        default=0.0)
    # C1 REACHABILITY CERTIFICATE, measured in-run: the TOP rung's realised POST-CLAMP
    # (argmin-relevant) range against the largest |margin| the SUBJECT arm records. Measured
    # on the CONTROL arm, compared against the SUBJECT's own margins, so it does not certify
    # its own subject.
    top_arm_id = VERIFY_LIFT_ARM_IDS[-1]
    _ratios = []
    for p in pairs:
        rng = float(p[top_arm_id]["instrument"]["max_post_clamp_bias_range"])
        worst = float(p[ARM_RDY]["instrument"]["max_abs_margin"])
        if worst > 0.0:
            _ratios.append(rng / worst)
    top_headroom = min(_ratios) if _ratios else 0.0
    fresh_worst = min((int(p["n_e3_fresh_compared"]) for p in pairs), default=0)
    # RED-TEAM F7: the denominator counts REF-fresh ticks while the numerator needs BOTH arms
    # fresh. Per-agent clocks can desync (MECH-091), so the skew is measured rather than
    # assumed zero; a non-zero value means the reported fraction is denominated on more ticks
    # than were actually comparable.
    fresh_skew_worst = max(
        (abs(int(p[a]["instrument"]["n_e3_fresh_ticks"]) - int(p["n_e3_fresh_compared"]))
         for p in pairs for a in FOLLOWER_ARM_IDS), default=0)
    sat_worst = max(
        (float(p[ARM_RDY]["instrument"]["max_clamp_saturated_frac"]) for p in pairs),
        default=0.0)
    sens = [_arm_sensitivity(p) for p in pairs]
    n_seeds_arm_sensitive = sum(1 for s in sens if s["n_corrected_keys_differing"] > 0)
    n_seeds_count_sensitive = sum(
        1 for s in sens if s["n_corrected_count_keys_differing"] > 0)
    vac = max((float(p[ARM_RDY]["accumulator"].get("vacuous_readout_rate", 1.0))
               for p in pairs), default=1.0)
    n_targets_worst = min((int(p[ARM_RDY]["accumulator"].get("max_n_targets", 0))
                           for p in pairs), default=0)
    lp_range_worst = min((float(p[ARM_RDY]["accumulator"].get("max_lp_dev_range", 0.0))
                          for p in pairs), default=0.0)

    checks = [
        dv_headroom_check(
            "selection_divergence_headroom",
            dv_name="yoked_selection_divergence_frac (E3-fresh ticks)",
            criterion_threshold=DIVERGENCE_FLOOR,
            achievable=1.0,
            statistic="analytic_bound",
            margin=2.0,
            control=("analytic: a divergence FRACTION over E3-fresh ticks spans [0, 1] on any "
                     "dataset, so the floor sits at 1e-9 of an achievable 1.0"),
            description=("DV headroom for the routed criteria. The EMPIRICAL half -- that "
                         "the ladder can actually drive that fraction off zero -- is carried "
                         "by top_rung_authority_exceeds_worst_margin and "
                         "e3_fresh_ticks_sufficient below, both measured rather than "
                         "assumed."),
        ),
        {"name": "paired_control_is_bit_identical", "kind": "readiness",
         "measured": ctrl_worst, "threshold": CONTROL_DIVERGENCE_CEILING,
         "direction": "upper",
         "control": ("an arm yoked against ITSELF, identical seed and config, on the SEEDED "
                     "rollout, checked on BOTH the action DV and the routed selection DV"),
         "description": ("INSTRUMENT CONTROL. Non-zero on either DV means the yoked agents "
                         "differ in something other than the manipulation and every "
                         "divergence number in this run is void. Local rather than imported "
                         "because V3-EXQ-964a's version runs an UNSEEDED env and checks the "
                         "action DV only (red-team F1/F4).")},
        {"name": "injected_deficit_differentiates_candidates", "kind": "readiness",
         "measured": inj_worst, "threshold": INJECTED_RANGE_FLOOR, "direction": "lower",
         "comparator": ">",
         "control": ("the synthetic ramp m * arange(K)/(K-1), whose cross-candidate range is "
                     "exactly m by construction, measured on the worst verify-lift cell"),
         "description": ("The positive control must be NON-UNIFORM. A uniform synthetic "
                         "deficit cannot move an argmax at any magnitude -- it would be a "
                         "positive control that provably cannot work, which is the defect "
                         "class this whole run exists to repair one layer up.")},
        {"name": "top_rung_authority_exceeds_worst_margin", "kind": "readiness",
         "measured": top_headroom, "threshold": TOP_RUNG_MARGIN_HEADROOM,
         "direction": "lower",
         "control": ("the TOP ladder rung's realised post-clamp deviation range (2 * "
                     "curiosity_bias_scale once railed) divided by the largest |margin| the "
                     "SUBJECT arm records on the same rollout; authoring probe measured "
                     "range 20.0 against a worst |margin| of 4.078, a ratio of ~4.9"),
         "description": ("REACHABILITY CERTIFICATE FOR C1, measured rather than hoped for. "
                         "Below 1.0 the top rung's perturbation cannot reach the margins it "
                         "must overcome, C1 is unmeetable by construction, and a zero "
                         "divergence there would mislabel an instrument-specification gap as "
                         "a substrate verdict. Measured on the CONTROL arm against the "
                         "SUBJECT's margins, so it does not certify its own subject.")},
        {"name": "fresh_tick_denominator_exact", "kind": "readiness",
         "measured": float(fresh_skew_worst), "threshold": 0.0, "direction": "upper",
         "control": ("each follower's own count of ticks on which select() ran, against the "
                     "shared reference-fresh denominator the fractions divide by"),
         "description": ("RED-TEAM F7. The selection fraction is denominated on REF-fresh "
                         "ticks while its numerator needs BOTH arms fresh; per-agent clocks "
                         "can desync, and a skew means the run reports an n it does not "
                         "have -- the pseudo-replication shape, measured rather than "
                         "assumed.")},
        {"name": "e3_fresh_ticks_sufficient", "kind": "readiness",
         "measured": float(fresh_worst), "threshold": E3_FRESH_TICKS_FLOOR,
         "direction": "lower",
         "control": ("ticks on which select() demonstrably ran, proven per tick by the "
                     "latch-clear discipline rather than assumed from the step count; "
                     "authoring probe measured 11 in 60 steps on one seed"),
         "description": ("SAMPLE SIZE for the selection DV. E3 selects on ~1 tick in 10, so a "
                         "divergence fraction denominated on env steps reports an n it does "
                         "not have -- the pseudo-replication shape. This floor refuses a run "
                         "that measured almost nothing.")},
        {"name": "multitarget_regime_reached", "kind": "readiness",
         "measured": float(n_targets_worst), "threshold": x964a.MULTITARGET_FLOOR,
         "direction": "lower",
         "control": ("V3-EXQ-964a measured 14/16/14 persistent targets on ARM_READINESS with "
                     "these same imported knobs"),
         "description": ("Carried forward from V3-EXQ-964a: below a second persistent target "
                         "the readout is constant and no divergence criterion can "
                         "discriminate by arithmetic.")},
        {"name": "readout_differentiates_across_candidates", "kind": "readiness",
         "measured": lp_range_worst, "threshold": x964a.LP_DEV_RANGE_FLOOR,
         "direction": "lower",
         "control": ("V3-EXQ-964a measured max_lp_dev_range 4.4e-3 to 7.3e-3 on ARM_READINESS "
                     "with these same imported knobs"),
         "description": ("Carried forward: the SUBJECT arm's own perturbation must vary "
                         "across candidates, else C3 measures zero by arithmetic.")},
        {"name": "vacuous_readout_rate_bounded", "kind": "readiness",
         "measured": vac, "threshold": x964a.VACUOUS_READOUT_RATE_CEILING,
         "direction": "upper",
         "control": "live rollout with the SD-063 head trained identically on every arm",
         "description": ("Carried forward: a near-total readiness-gate refusal rate would "
                         "mean the accumulator never got a chance to matter.")},
    ]

    gate_green = True
    gate_reason = None
    try:
        preconditions = p0_readiness_gate(checks)
    except P0NotReady as exc:
        gate_green = False
        preconditions = exc.preconditions
        gate_reason = "; ".join(
            str(p.get("name")) for p in preconditions if not p.get("met", True))

    # ---- criteria + routing ----------------------------------------------------------
    ladder_firing = {a: _n_seeds_firing(pairs, a) for a in VERIFY_LIFT_ARM_IDS}
    top_arm = VERIFY_LIFT_ARM_IDS[-1]
    c1 = bool(ladder_firing[top_arm] >= majority)
    min_effective = None
    min_effective_arm = None
    for m, a in zip(VERIFY_LIFT_LADDER, VERIFY_LIFT_ARM_IDS):
        if ladder_firing[a] >= majority:
            min_effective, min_effective_arm = float(m), a
            break
    c2 = bool(min_effective is not None)
    rdy_firing = _n_seeds_firing(pairs, ARM_RDY)
    c3 = bool(rdy_firing >= majority)

    if not gate_green:
        label = "substrate_not_ready_requeue"
    elif not c1:
        label = "detector_cannot_fire_instrument_defect"
    elif c3:
        label = "mechanism_moves_selected_candidate"
    else:
        label = "mechanism_inert_at_own_magnitude_detector_verified"

    criteria = [
        {"name": "C1_detector_sensitivity_top_rung_fires", "load_bearing": True,
         "passed": c1, "measured": float(ladder_firing[top_arm]),
         "threshold": float(majority), "comparator": ">=",
         "detail": ("seeds on which the TOP verify-lift rung "
                    "(curiosity_bias_scale=%g, realised argmin-relevant range %g) produced a "
                    "yoked divergence above %g. THE GATE ON EVERYTHING ELSE: if the detector "
                    "cannot fire at ~5x the largest |margin| the subject records, C3 is "
                    "uninterpretable and the route is a harness repair, not a substrate "
                    "verdict."
                    % (VERIFY_LIFT_LADDER[-1], 2.0 * VERIFY_LIFT_LADDER[-1],
                       DIVERGENCE_FLOOR))},
        {"name": "C2_minimum_effective_authority_measured", "load_bearing": True,
         "passed": c2,
         "measured": (min_effective if min_effective is not None
                      else float(VERIFY_LIFT_LADDER[-1]) * 10.0),
         "threshold": float(VERIFY_LIFT_LADDER[-1]), "comparator": "<=",
         "shipped_curiosity_bias_scale": float(VERIFY_LIFT_LADDER[0]),
         "detail": ("the SMALLEST curiosity_bias_scale whose divergence fires on a seed "
                    "majority -- the measured sizing that replaces the withdrawn 4-5x weight "
                    "raise, in the units of the knob that actually BOUNDS curiosity's "
                    "selection authority. Unmet reads as above the top rung. Compare against "
                    "the SHIPPED value 0.1, which is the bottom rung.")},
        {"name": "C3_subject_moves_selected_candidate", "load_bearing": True,
         "passed": c3, "measured": float(rdy_firing), "threshold": float(majority),
         "comparator": ">=",
         "detail": ("seeds on which ARM_READINESS's own SELECTION divergence fires over "
                    "E3-fresh ticks -- the mechanism-faithful DV. The action projection is "
                    "recorded alongside as readiness_max_action_divergence_frac and is what "
                    "V3-EXQ-964a measured. INTERPRETABLE ONLY IF C1 PASSES -- that ordering "
                    "is why this run exists. V3-EXQ-964a measured 0/540 on the action "
                    "projection with no proof the detector could fire at all.")},
    ]
    combination_rule = (
        "ORDERED GATE, not an AND of PASSes. C1 gates C3: a detector that cannot fire at the "
        "top of a four-decade ladder makes C3 uninterpretable, so !C1 routes "
        "`detector_cannot_fire_instrument_defect` REGARDLESS of C3's value. With C1 met, C3 "
        "selects between `mechanism_moves_selected_candidate` and "
        "`mechanism_inert_at_own_magnitude_detector_verified`, and C2 reports how far the "
        "real magnitude sits from the effective one either way. A failed precondition "
        "overrides all three and self-routes substrate_not_ready_requeue. `outcome` is PASS "
        "when the run ADJUDICATED (gate green and C1 met), FAIL otherwise -- never a verdict "
        "on MECH-482, which this diagnostic does not score."
    )
    criteria_non_degenerate = {
        # C1 can only discriminate if the injected control is non-uniform and the self-yoked
        # control is clean -- i.e. a zero would mean something.
        "C1_detector_sensitivity_top_rung_fires": bool(
            inj_worst > INJECTED_RANGE_FLOOR
            and ctrl_worst <= CONTROL_DIVERGENCE_CEILING
            and top_headroom >= TOP_RUNG_MARGIN_HEADROOM),
        # C2 needs at least two rungs measured, else "smallest firing rung" is not a sweep.
        "C2_minimum_effective_authority_measured": bool(
            len(VERIFY_LIFT_ARM_IDS) >= 2
            and inj_worst > INJECTED_RANGE_FLOOR
            and top_headroom >= TOP_RUNG_MARGIN_HEADROOM),
        # C3 is degenerate exactly when the subject's own readout does not vary -- the
        # V3-EXQ-964 arithmetic-zero condition -- or when the detector is unproven.
        "C3_subject_moves_selected_candidate": bool(
            c1 and lp_range_worst > x964a.LP_DEV_RANGE_FLOOR
            and ctrl_worst <= CONTROL_DIVERGENCE_CEILING
            and fresh_worst >= E3_FRESH_TICKS_FLOOR),
    }
    non_degenerate = bool(any(criteria_non_degenerate.values()))
    outcome = "PASS" if (gate_green and c1) else "FAIL"

    readout: Dict[str, Any] = {
        "n_seeds": len(seeds),
        "seed_majority": majority,
        "gate_green": 1 if gate_green else 0,
        "non_degenerate": 1 if non_degenerate else 0,
        "divergence_floor": DIVERGENCE_FLOOR,
        "control_divergence_worst": ctrl_worst,
        "injected_range_worst": inj_worst,
        "top_rung_margin_headroom": top_headroom,
        "subject_clamp_saturated_frac_worst": sat_worst,
        "n_seeds_corrected_instrument_arm_sensitive": n_seeds_arm_sensitive,
        "n_seeds_corrected_count_partition_arm_sensitive": n_seeds_count_sensitive,
        "n_seeds_reproducing_964a_legacy_byte_identity": sum(
            1 for s in sens if s["reproduces_964a_legacy_byte_identity"]),
        "e3_fresh_ticks_worst": float(fresh_worst),
        "fresh_tick_denominator_skew_worst": float(fresh_skew_worst),
        "readiness_yoked_selection_divergence_seeds_firing": rdy_firing,
        "readiness_yoked_action_divergence_seeds_firing": float(
            _n_seeds_firing(pairs, ARM_RDY, "yoked_divergence_frac")),
        "top_rung_seeds_firing": ladder_firing[top_arm],
        "min_effective_magnitude": min_effective,
        "c1_detector_sensitivity": 1 if c1 else 0,
        "c2_min_effective_measured": 1 if c2 else 0,
        "c3_subject_moves_action": 1 if c3 else 0,
        "vacuous_readout_rate_worst": vac,
        "max_n_targets_worst": float(n_targets_worst),
        "max_lp_dev_range_worst": lp_range_worst,
    }
    for m, a in zip(VERIFY_LIFT_LADDER, VERIFY_LIFT_ARM_IDS):
        readout["ladder_seeds_firing__s%g" % m] = float(ladder_firing[a])
        fr = [float(p[a]["yoked_selection_divergence_frac"]) for p in pairs]
        readout["ladder_max_selection_divergence_frac__s%g" % m] = max(fr) if fr else 0.0
        af = [float(p[a]["yoked_divergence_frac"]) for p in pairs]
        readout["ladder_max_action_divergence_frac__s%g" % m] = max(af) if af else 0.0
    rdy_fr = [float(p[ARM_RDY]["yoked_selection_divergence_frac"]) for p in pairs]
    readout["readiness_max_selection_divergence_frac"] = max(rdy_fr) if rdy_fr else 0.0
    rdy_af = [float(p[ARM_RDY]["yoked_divergence_frac"]) for p in pairs]
    readout["readiness_max_action_divergence_frac"] = max(rdy_af) if rdy_af else 0.0
    readout = _flat_scalar(readout)

    manifest: Dict[str, Any] = {
        "run_id": "%s_%s_v3" % (EXPERIMENT_TYPE,
                               datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")),
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": list(CLAIM_IDS),
        "supersedes": SUPERSEDES,
        "evidence_direction": "unknown",
        "non_degenerate": non_degenerate,
        "degeneracy_reason": (None if non_degenerate
                              else "no criterion could discriminate: %s" % (gate_reason,)),
        "outcome": outcome,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "sleep_driver_pattern": "none",
        "dry_run": bool(dry_run),
        "combination_rule": combination_rule,
        "criteria": criteria,
        "readout": readout,
        "arm_results": rows,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria": criteria,
            "combination_rule": combination_rule,
            "criteria_non_degenerate": criteria_non_degenerate,
            "gate_reason": gate_reason,
            "ladder_seeds_firing": ladder_firing,
            "min_effective_magnitude": min_effective,
            "min_effective_arm": min_effective_arm,
            "arm_sensitivity_per_seed": sens,
            "routes": {
                "detector_cannot_fire_instrument_defect": (
                    "The selection-divergence detector did not fire even at "
                    "curiosity_bias_scale=%g (realised argmin-relevant range %g), whose "
                    "measured headroom over the subject's own worst |margin| is recorded in "
                    "readout.top_rung_margin_headroom and gated at 1.0. C3 is uninterpretable "
                    "for the second time and NOTHING is learned about MECH-482. Route to a "
                    "harness repair -- NOT to another lettered weight bump, which the "
                    "CONFIRMED V3-EXQ-964a autopsy already withdrew."
                    % (VERIFY_LIFT_LADDER[-1], 2.0 * VERIFY_LIFT_LADDER[-1])),
                "mechanism_inert_at_own_magnitude_detector_verified": (
                    "The detector demonstrably fires, and the real mechanism still never "
                    "moves the committed action. This is an INTERPRETABLE negative -- the "
                    "first one MECH-482 has had -- and C2's minimum effective magnitude says "
                    "how far the real perturbation (~0.1 pre-weight) sits from one that does "
                    "move it. It does NOT by itself license a weight raise: a minimum "
                    "effective magnitude orders of magnitude above the real one is an "
                    "F-dominance finding about the committed-selection layer (ARC-110 / "
                    "MECH-439 territory), which is a substrate question, not a config one."),
                "mechanism_moves_selected_candidate": (
                    "ARM_READINESS's own perturbation moves the committed action on a seed "
                    "majority -- the first positive behavioural signal for MECH-482. Route to "
                    "/governance to decide whether this promotes the substrate entry "
                    "sd_epistemic_deficit_multitarget_readiness out of "
                    "implemented_pending_validation."),
                "substrate_not_ready_requeue": (
                    "A precondition failed. No outcome is attributable; an instrument result, "
                    "not a verdict on MECH-482."),
            },
            "carried_forward_from_964a": (
                "C1/C2/C5 of V3-EXQ-964a (multitarget regime reached, readout differentiates, "
                "collapse control reproduces) are re-measured here as PRECONDITIONS rather "
                "than re-litigated as criteria: the confirmed autopsy records them as "
                "standing, and this run's question is one layer downstream."),
        },
        "diagnostics": {
            "verify_lift_ladder": list(VERIFY_LIFT_LADDER),
            "ladder_seeds_firing": ladder_firing,
            "control_divergence": control_div,
            "ref_964a": {
                "run_id": ("v3_exq_964a_mech482_epistemic_deficit_multitarget_readiness"
                           "_20260911T174949Z_v3"),
                "yoked_divergence_frac_max": 0.0,
                "n_yoked_comparisons": 540,
                "legacy_counters_byte_identical_across_arms": True,
                "min_argmax_margin_by_seed": {"71": -4.1410, "101": -1.3599, "202": -1.1294},
                "max_lp_dev_range_by_seed": {"71": 0.00734, "101": 0.00442, "202": 0.00612},
            },
        },
    }

    full_config = {
        "env": "CausalGridWorldV2",
        "seeds": list(seeds),
        "episodes": episodes,
        "steps_per_episode": steps,
        "follower_arms": list(FOLLOWER_ARM_IDS),
        "verify_lift_ladder": list(VERIFY_LIFT_LADDER),
        "readiness_knobs": dict(x964a.READINESS_KNOBS),
        "prereadiness_knobs": dict(x964a.PREREADINESS_KNOBS),
        "divergence_floor": DIVERGENCE_FLOOR,
        "control_divergence_ceiling": CONTROL_DIVERGENCE_CEILING,
        "arm_sensitivity_floor": ARM_SENSITIVITY_FLOOR,
        "injected_range_floor": INJECTED_RANGE_FLOOR,
        "seed_majority": majority,
        "control_steps": ctrl_steps,
        "env_seeded": True,
    }
    out_path = write_flat_manifest(
        manifest, dry_run=dry_run, config=full_config, seeds=list(seeds),
        script_path=Path(__file__), started_at=t0, z_goal_stream_stats=zg.stats(),
    )
    manifest["_out_path"] = str(out_path)
    return manifest


def _fmt(v: Any) -> str:
    return "None" if v is None else ("%.6g" % float(v))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()

    seeds = args.seeds if args.seeds else (DRY_RUN_SEEDS if args.dry_run else SEEDS)
    episodes = DRY_RUN_EPISODES if args.dry_run else EPISODES
    steps = DRY_RUN_STEPS if args.dry_run else STEPS_PER_EPISODE
    result = run_experiment(episodes, steps, seeds, dry_run=args.dry_run)

    print("")
    print("=" * 78)
    print("%s -- %s" % (QUEUE_ID, result["interpretation"]["label"]))
    print("outcome: %s  non_degenerate: %s" % (result["outcome"], result["non_degenerate"]))
    for c in result["criteria"]:
        print("  %-44s passed=%-5s measured=%s threshold=%s"
              % (c["name"], c["passed"], _fmt(c["measured"]), c["threshold"]))
    for p in result["interpretation"]["preconditions"]:
        print("  [precond] %-44s met=%-5s measured=%s threshold=%s"
              % (p.get("name"), p.get("met"), _fmt(p.get("measured")), p.get("threshold")))
    print("  ladder seeds firing: %s" % (result["interpretation"]["ladder_seeds_firing"],))
    print("  min effective magnitude: %s"
          % (result["interpretation"]["min_effective_magnitude"],))
    print("manifest: %s" % result["_out_path"])
    print("=" * 78)

    if args.dry_run:
        rows = result["arm_results"]
        assert rows, "no cells ran"
        for r in rows:
            for a in VERIFY_LIFT_ARM_IDS:
                inst = r[a]["instrument"]
                assert int(inst["n_injected_ticks"]) > 0, \
                    "%s never injected -- the positive control is inert" % a
                assert float(inst["max_injected_range"]) > INJECTED_RANGE_FLOOR, \
                    "%s injected a uniform vector -- it cannot move an argmax" % a
            # The corrected instrument must not simply reproduce the legacy counters, or
            # nothing was repaired.
            inst_r = r[ARM_RDY]["instrument"]
            assert (inst_r["corrected_n_exact_tie_ticks"]
                    + inst_r["corrected_n_negative_margin_ticks"]
                    + inst_r["corrected_n_positive_margin_ticks"]
                    == inst_r["legacy_n_margin_reads"]), \
                "corrected margin partition does not account for every legacy read"
        # THE ROUTED DV must actually move across at least two swept rungs, else the ladder
        # is a saturated sweep rather than a dose-response -- the failure mode the first draft
        # shipped (a magnitude ladder whose post-clamp perturbation was bit-identical above
        # m~4). Read on the SELECTION fraction, which is what the criteria route on.
        sel = sorted({round(float(r[a]["yoked_selection_divergence_frac"]), 12)
                      for r in rows for a in VERIFY_LIFT_ARM_IDS})
        act = sorted({round(float(r[a]["yoked_divergence_frac"]), 12)
                      for r in rows for a in VERIFY_LIFT_ARM_IDS})
        print("[smoke] ladder SELECTION divergence fractions across rungs: %s" % (sel,))
        print("[smoke] ladder ACTION divergence fractions across rungs:    %s" % (act,))
        assert len(sel) >= 2, (
            "the ladder produced a single selection-divergence value across every rung -- a "
            "saturation fingerprint, not a dose-response: %s" % (sel,))
        # And the realised post-clamp authority must itself vary across rungs, else the sweep
        # never reached the substrate.
        rng = sorted({round(float(r[a]["instrument"]["max_post_clamp_bias_range"]), 9)
                      for r in rows for a in VERIFY_LIFT_ARM_IDS})
        assert len(rng) >= 2, (
            "post-clamp authority identical across rungs -- the ladder did not reach the "
            "substrate: %s" % (rng,))
        print("[smoke] ladder post-clamp authority ranges: %s" % (rng,))
        print("[smoke] all assertions passed")

    return result, args


if __name__ == "__main__":
    _result, _args = main()
    _outcome_raw = str(_result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=_result["_out_path"],
        dry_run=_args.dry_run,
    )
