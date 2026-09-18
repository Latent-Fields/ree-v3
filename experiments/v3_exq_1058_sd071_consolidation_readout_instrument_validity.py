"""
V3-EXQ-1058: SD-071 -- the SD-068 nrem and sws consolidation readouts are
CONTENT-CONTINGENT INSTRUMENTS. The non-diagnostic (evidence) run SD-071 needs.
SLEEP DRIVER: manual-cycle-loop (the SD-068 harness drives enter_sws_mode /
run_sws_schema_pass / the offline replay path directly per phase readout; no
SleepLoopManager scheduling).

RED-TEAM (Step 4.5): see the RED-TEAM line at the end of this docstring.

WHAT THIS MEASURES, AND WHY IT IS AN `evidence` RUN
---------------------------------------------------
SD-071 is an INSTRUMENT-VALIDITY claim, not a claim about sleep biology. Verbatim from
its `what_would_answer`, this is the whole of what it still needs:

    "REMAINING CONDITION FOR PROMOTION: one NON-DIAGNOSTIC run tagging this claim.
     Both supporting runs are experiment_purpose=diagnostic and therefore do not
     weight governance confidence."

and the whole of what would refute it:

    "REFUTED if a re-run at n>=8 puts ceiling_inside_ci95 true on either leg, or if
     the sws leg's content-scale ladder collapses below the 0.01 floor (which would
     reopen the scale-invariance caveat that C3 was pre-registered to discharge)."

Both criteria below are that text, transcribed. Nothing here is re-designed: the two
supporting measurements already exist (V3-EXQ-778c, V3-EXQ-778g) and were adjudicated;
what they could not be is NON-diagnostic, because the harness forbade it.

THE CARVE-OUT THIS RUN EXERCISES -- and the two conditions it must satisfy
-------------------------------------------------------------------------
Until 2026-09-18 the SD-068 harness carried an UNCONDITIONAL contract -- "Any run built
on it MUST be EXPERIMENT_PURPOSE=diagnostic" -- under the heading "Prerequisite caveat
-- MECH-121 hold (respected, not lifted)". SD-071's promotion condition and that
contract were in direct conflict, so SD-071 could never be promoted. Raised as
GFLAG-0319 (contested_disposition, 2026-09-17); REFUSED at the /queue-experiment
STOP-GATE by science-20260918-sd071-consolidation-readouts; decided by the USER
2026-09-18T19:47:16Z through the Orchestrator decision lane
(orchestrate-20260918-1840-cloud4, real AskUserQuestion): OPTION A, GRANT THE NARROW
CARVE-OUT. Amendment landed and verified on origin at both caveat sites --
REE_assembly d202c2d73c (docs/architecture/sd_068_consolidation_lesion_harness.md) and
ree-v3 6371c9fc8a (experiments/_lib/consolidation_lesion_harness.py). GFLAG-0319
resolved (REE_assembly 876429a2d4); EXP-1178/EVB-1657 lifted back to `proposed`
(REE_assembly 6d0427b432).

The carve-out is an ENUMERATION, not a category. A run on this harness may be
EXPERIMENT_PURPOSE="evidence" if and only if BOTH hold:

  (i)  its `claim_ids` are SOLELY instrument-validity claims about the harness's own
       readouts -- the enumerated set is {SD-071} and nothing else;
  (ii) it tags NEITHER MECH-120 NOR MECH-121.

THIS RUN SATISFIES BOTH BY CONSTRUCTION, and that is asserted in code (see
`_assert_carveout_conditions`, called before any compute): CLAIM_IDS == ["SD-071"]
exactly. Not "SD-071 plus the staging claims for context" -- 778c/778g tagged
["SD-068", "MECH-168", "INV-047", "MECH-169"], and repeating that here would break
condition (i). The staging claims are named in `notes` as provenance, which is not a
tag. MECH-121's status is untouched by this run under every outcome; condition (ii) is
what makes that automatic rather than a further obligation.

WHAT IS DELIBERATELY *NOT* CLAIMED HERE. A pass says the two readouts are
content-contingent instruments -- they respond to the AMOUNT of planted content and not
merely to perturbation magnitude. It says nothing about consolidation BEHAVIOUR, about
the staging ORDER those readouts were used to establish (SD-068 / MECH-168 / INV-047 /
MECH-169 own that, and this run does not weight them), and nothing about MECH-120 or
MECH-121.

THE MEASUREMENT
---------------
Per seed, the identical sigma sweep is run twice on the identical substrate, seeds,
warm-up and RNG streams, differing ONLY in `content_scale` (1.0 INJECTED vs 0.0 NULL).
The delivered perturbation is held numerically identical in both arms (each readout
references its noise scale to the UNSCALED content), so the null arm is Bar et al.
2020's "same odour delivered, no prior pairing" rather than "weaker odour". Both error
series are expressed in the INJECTED arm's common units and least-squares fitted
against sigma:

    null_slope_ratio_<leg> = |null sigma-slope| / |injected sigma-slope|

0.0 == fully content-contingent (inert on noise); 1.0 == fully confounded (responds to
sigma identically with and without content). Harness: `run_null_content_control`
(:1603), `null_slope_ratio` (:98, :675).

  nrem leg: `nrem_transfer_fidelity`   (harness:523, called at :1039)
  sws  leg: `_sws_pattern_completion`  (harness:400, called at :379)

The sws readout is the REBUILT one -- the prior SNR readout was an analytic identity
(778c: ratio 1.0000, sd 2.7e-8 on 8/8 seeds, because `_shy` is affine so the content
term differentiates away) and was replaced by a cosine retrieval margin against the
injected prototypes. SD-071 is the claim that the replacement, and the nrem leg
alongside it, are genuine instruments.

RECORDED MAGNITUDES TO BEAT (the two diagnostic runs SD-071 rests on)
  nrem: null_slope_ratio 0.1445, CI95 [0.1438, 0.1451], 0/8 seeds confounded  (778c)
  sws : null_slope_ratio 0.1495, sd 0.0218, CI95 [0.1344, 0.1646],
        ceiling_inside_ci95 FALSE, 8/8 seeds                                   (778g)
  sws C3 content-scale ladder: slope spread 0.1108 against the 0.01 floor      (778g)
These are context for reading the result, NOT thresholds -- the substrate has moved
since (harness commits 76508144, b42f69ff and the 6371c9fc8a docstring amendment), so
the substrate_hash differs and this is a fresh measurement, not a reproduction.

ACCEPTANCE (pre-registered -- transcribed from SD-071's own what_would_answer)
-----------------------------------------------------------------------------
  C1 (LOAD-BEARING): `ceiling_inside_ci95` is FALSE on BOTH legs (nrem AND sws).
     The ceiling is the harness's own NULL_SLOPE_RATIO_CEILING (0.25). The CI95 is over
     the per-seed ratio distribution at n>=8, computed by the harness's shared
     `subgroup_ratio_stats` (:1446) -- the same helper 778g used, so the interval is
     constructed identically. A ceiling INSIDE the interval means the verdict is
     unresolved at this n; SD-071 says that outcome REFUTES the claim.
  C2 (readiness / positive control, NOT a scientific criterion): on BOTH legs, the
     ratio's DENOMINATOR -- |injected sigma-slope|, the same statistic C1 routes on --
     clears INJECTED_SLOPE_FLOOR. A below-floor denominator means the sweep never
     damaged the readout, so the ratio is 0/0 and the control cannot discriminate.
     Below floor self-routes `substrate_not_ready_requeue`, NEVER a substrate verdict.
  C3 (LOAD-BEARING, sws leg only): the sws content-scale ladder SPREAD across the
     content_scale>0 rungs exceeds LADDER_SPREAD_FLOOR = 0.01. This is the 0.01 floor
     SD-071 names. It is the anti-artifact check: the rebuilt sws readout is
     cosine-based and therefore scale-invariant, so its null arm is flat in sigma
     PARTLY BY CONSTRUCTION -- a low null ratio is partly implied by the readout's form
     and needs an independent content-TRACKING check. C3 asks whether the sigma-response
     VARIES with content amplitude, not merely whether it switches on.
     C3 IS SWS-ONLY BY THE CLAIM'S OWN WORDING ("the sws leg's content-scale ladder").
     The nrem leg carries no ladder here, and this run does not invent one for it.
     READ THE RED-TEAM SECTION BELOW BEFORE INTERPRETING A C3 PASS. C3's ladder was
     confirmed (bit-exactly, 8/8 seeds against 778g's recorded ladder_error_series) to
     be a SIGMA-REPARAMETERISATION of the injected arm's own damage curve, so what it
     actually tests is that that curve is GRADED rather than step-shaped. It stays
     failable -- and therefore a real criterion -- but it does NOT independently
     discharge the scale-invariance caveat that this paragraph, SD-071's falsifier and
     778g's docstring all say it discharges.

TWO CHECKS THAT ARE MEASURED AND RECORDED BUT DELIBERATELY NOT GATED. Both would be
criteria SD-071 does not pre-register, and adding a gate the claim did not name would
change what this run can conclude:
  * C3a, the ladder SIGNAL-RATIO leg (content-bearing rungs respond several-fold more
    strongly in sigma than the content-free rung). 778g gated this at 3.0x as a
    diagnostic; SD-071's falsifier names only the 0.01 spread floor. Recorded as a
    non-gating precondition so a weak signal ratio is visible to a reader and to any
    later autopsy without silently re-scoping the claim.
  * `ci95_high_below_ceiling_<leg>` -- WHICH SIDE of the ceiling the interval fell on.
    This matters because `ceiling_inside_ci95 == False` is TWO-WAY AMBIGUOUS: it is
    False both when the interval sits entirely BELOW the ceiling (content-contingent,
    the reading SD-071 asserts) and when it sits entirely ABOVE it (fully confounded,
    which REFUTES SD-071). C1 as pre-registered cannot separate those. So rather than
    add an unsanctioned criterion, this run records the side explicitly and uses it in
    (a) the self-route label, (b) `evidence_direction`, and (c) the non-degeneracy net:
    an above-ceiling interval sets `non_degenerate: false`, which keeps a confounded
    leg from being scored as SUPPORT for SD-071 on a technically-passing C1.

NON-DEGENERACY NET (an `evidence` run, so top-level `non_degenerate`, per CLAUDE.md /
/queue-experiment Step 3). `non_degenerate` is set FALSE, with a `degeneracy_reason`,
when any of these holds on a gated leg -- each is the ABSENCE of a measurement, not a
reading:
  * the null series is a saturated CONSTANT across the sigma grid
    (`null_series_degenerate_<leg>` from the harness): its slope is exactly 0, so its
    ratio is exactly 0.0, which would otherwise clear the ceiling as a clean pass;
  * the null control is unavailable on that leg (`null_control_available_<leg>`);
  * C2's denominator is below floor (the ratio is 0/0);
  * the CI95 sits entirely ABOVE the ceiling (the confounded reading of a False C1).

DV-SYMMETRY INVARIANCE (Step 3.5, one line per arm, as required)
----------------------------------------------------------------
There is ONE arm pair per leg, contrasted within seed: INJECTED (content_scale=1.0) vs
NULL (content_scale=0.0).
  * nrem leg. DV = the sigma-slope of the nrem transfer-fidelity error, expressed in the
    injected arm's units. Symmetry group of a least-squares SLOPE in sigma: additive
    constants in the error series (a slope annihilates them) and any reordering of the
    sigma grid points. The manipulation is NOT invariant under either: `content_scale`
    changes the store the readout is scored against, which changes the error at each
    sigma DIFFERENTLY (the perturbation is referenced to the unscaled content, so it is
    held numerically identical across arms while the thing it damages is not) -- so the
    arms differ in the sigma-DEPENDENCE, not by a broadcast offset. A uniform offset
    would cancel; this does not.
  * sws leg. DV = the sigma-slope of the cosine retrieval margin, referenced to the
    injected arm's own undamaged margin. Symmetry group: additive constants (as above)
    AND -- specifically for a cosine -- positive RESCALINGS of the store, under which
    cosine is exactly invariant. This is the real hazard here and it is why C3 exists:
    at content_scale=0 the store is `0 + sigma*noise`, so sigma cancels out of the
    cosine entirely and the null arm is flat BY CONSTRUCTION. The manipulation is NOT
    invariant under the rescaling symmetry once content is present, because the store is
    then `content + sigma*noise` and the ratio of correct-to-incorrect similarity does
    carry `content`; but the null arm's flatness is partly arithmetic. C3's ladder is
    the independent test that the response tracks content AMOUNT, which a pure
    scale-invariance artifact cannot produce.

READINESS-ANCHOR REACHABILITY. Every declared readiness precondition is replayed at
setup, through THE SHIPPED PREDICATE, against the frozen per-seed values of the
completed V3-EXQ-778g run -- the known-healthy positive control for exactly these gates
(it recorded both legs' full `null_control` per seed). A precondition its own healthy
reference cannot pass is a guaranteed false negative that would mislabel an
instrument-specification gap as a substrate verdict (the confirmed V3-EXQ-778d defect).

WHAT A FAIL MEANS -- each outcome is informative, none is a broken run:
  C2 below floor       -> instrument not exercised; requeue at an adequate sweep.
                          NOT a claim verdict.
  C1 fail (ceiling inside CI on either leg) -> SD-071 REFUTED at this n, in exactly the
                          terms it pre-registered.
  C1 False-but-above-ceiling on a leg -> that leg is CONFOUNDED; SD-071 weakened, and
                          the run is marked non_degenerate:false so it does not score
                          as support.
  C3 fail (spread <= 0.01) -> the sws null-control pass is a scale-invariance artifact;
                          SD-071 REFUTED on its second pre-registered falsifier.

MECH-094: this harness runs weight/state operations offline; harness:155-157 records
that it produces no hypothesis-tagged residue/anchor/memory writes beyond what the
phase ops already do, and does not simulate-then-commit. No new MECH-094 surface.

ETHICS PREFLIGHT (Step 2.6): all involvement flags false, decision allow. No negative
valence drive, no MECH-219 accumulator, no self-model, no inescapability, no offline
replay over harm content, no social/language layer, no human or clinical data. SENT-0:
V3 is not claimed sentient; this is pre-ethical instrumentation on injected content.

GOV-REUSE-1 (Step 2.4): the decisive readout is `ceiling_inside_ci95` on the per-seed
`null_slope_ratio` of BOTH legs, from a run with experiment_purpose != diagnostic and
claim_ids == [SD-071]. Checked `reanalysis_query.py query --readout null_slope_ratio
--claim SD-071` (0 matches) and grepped the manifest corpus by hand: the readout IS
recorded, on 778b/778c/778g, but (a) ZERO manifests tag SD-071 at all -- all three tag
["SD-068","MECH-168","INV-047","MECH-169"] -- and (b) all are experiment_purpose=
diagnostic, which by governance rule does not weight confidence. The missing artifact is
therefore a NON-DIAGNOSTIC MEASUREMENT TAGGING SD-071, which is not a number that can be
derived post-hoc from diagnostic runs: no reanalysis of a diagnostic manifest can make it
non-diagnostic. Not recoverable -> ran. Additionally the substrate_hash has changed since
both runs (harness 76508144, b42f69ff, 6371c9fc8a), so their values would be
INCOMPATIBLE for reuse regardless.

RE-DERIVE BRAKE (Step 2.5b): 0 autopsy targets in the 523-artifact corpus tag SD-071.
Brake not tripped.

Design + validity model: REE_assembly/docs/architecture/sd_068_consolidation_lesion_harness.md
Claim: SD-071 (REE_assembly/docs/claims/claims.yaml); proposal EXP-1178 / EVB-1657.
Decision record: GFLAG-0319 (resolved), REE_assembly evidence/planning/governance_flags.v1.json.

RED-TEAM (Step 4.5): CONTESTED (model fable, 2026-09-18). Two findings, both verified
against the source and both DISPOSED IN WRITING rather than designed around. Also
recorded in the queue entry `note` and, in full, in the manifest `interpretation` block.

  FINDING 1 -- C3's ladder is a SIGMA-REPARAMETERISATION of the injected arm, so it
  cannot discharge the scale-invariance caveat it is advertised as discharging.
  CONFIRMED BIT-EXACTLY: damage is referenced to the UNSCALED content, so under a
  scale-invariant cosine readout error(content_scale=cs, sigma=s) ==
  error(content_scale=1.0, sigma=s/cs). Checked against V3-EXQ-778g's recorded
  `ladder_error_series` on the doubling sigma grid: rung0.5[i] == rung1.0[i+1] and
  rung0.25[i] == rung1.0[i+2] for every i>=1, on 8/8 seeds. What C3 therefore ACTUALLY
  tests is that the injected damage curve is GRADED rather than step-shaped -- a real,
  failable property (a saturating curve gives spread ~0 and C3 fails), so C3 is not
  vacuous and the run still answers its own question. What it CANNOT do is independently
  establish that the sws readout tracks content AMOUNT, which is how SD-071's falsifier
  and 778g's docstring both present it. Same root cause, recorded alongside: the sws
  NULL-arm error series is {0, c, c, c, c} (n_distinct 2, 8/8 seeds in 778g), flat above
  sigma=0 because the cosine annihilates the sigma factor on a pure-noise store, so the
  sws null SLOPE is driven entirely by the sigma=0 discontinuity.
  DISPOSITION: RECORDED, NOT DESIGNED AROUND. C1 and C3 are transcribed verbatim from
  SD-071's own pre-registered falsifier. Re-scoping either would change what this run
  measures, which is a governance decision and not this run's to make -- and the red-team
  verdict was CONTESTED, not BLOCKING, precisely because both criteria remain failable.
  Raised for governance as a claim-level finding on SD-071 so that a PASS here is
  weighted knowing what C3 does and does not establish.

  FINDING 2 -- a CONFOUNDED leg could be recorded as a clean pass. Because SD-071's C1
  is stated as `ceiling_inside_ci95 == false`, a leg whose CI95 sits entirely ABOVE the
  ceiling passes C1 while being fully confounded, which REFUTES the claim. An earlier
  draft of this file then fell through to evidence_direction "mixed", contradicting this
  docstring. DISPOSITION: FIXED -- that case now routes evidence_direction to "weakens",
  ahead of the pass cases. `outcome` is deliberately LEFT as the pre-registered rule
  computes it (changing the pass rule would change a criterion the claim itself fixes),
  so outcome=PASS with direction=weakens is diagnostic of exactly this case; it is named
  in `interpretation.outcome_vs_direction_note` and carries non_degenerate=false, which
  excludes it from confidence scoring. ONE VARIANT FURTHER DOWN, from the same reviewer's
  detail and fixed the same way: a SATURATED-CONSTANT null series (ratio exactly 0.0)
  would give CI [0,0], hence ci95_high_below_ceiling True, hence direction "supports" on
  a non-measurement. `supports` now REQUIRES non_degenerate, and every
  nothing-was-measured state (saturated null, unavailable null control, uncomputable CI)
  routes to "unknown" -- while a CONFOUNDED leg still routes to "weakens", which is why
  that branch is tested first.

  Two further facts the reviewer surfaced, recorded rather than acted on: C2's requeue
  branch is effectively unreachable (its 1e-6 floor is 4-5 orders below where C1 already
  fails), so a reduced-sensitivity outcome routes to `weakens`, never `unknown`; and the
  nrem DV is computed through the MECH-121-cluster consolidation pass
  (ree_core/sleep/cross_module_consolidation.py, Adam at :162), so the nrem leg's
  content-contingency is not wholly operator-agnostic. Neither breaches the carve-out --
  its conditions are about TAGGING, and this run tags neither MECH-120 nor MECH-121 and
  cannot move MECH-121's status under any outcome. Both are in `interpretation`.

  Reviewer family 1 (does the manipulation reach the DV): no finding -- the `cached=`
  ladder reuse is exact and both arms rebuild agent and generator per cell.
"""

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib import consolidation_lesion_harness as H  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments._lib.readiness_anchor import assert_anchor_reachable  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_1058_sd071_consolidation_readout_instrument_validity"
QUEUE_ID = "V3-EXQ-1058"
# NOT a supersession. 778c and 778g both stand -- their findings are what SD-071 rests
# on. This run supplies the one thing they structurally could not: a NON-diagnostic
# measurement tagging SD-071. Marking either superseded would erase the evidence.
SUPERSEDES = None

# --- THE CARVE-OUT CONDITIONS, IN CODE ---------------------------------------------
# Condition (i): claim_ids are SOLELY instrument-validity claims -- enumerated {SD-071}.
# Condition (ii): NEITHER MECH-120 NOR MECH-121 tagged.
# Asserted before any compute by _assert_carveout_conditions(); a future edit that adds
# a "context" tag here fails loudly instead of silently voiding the carve-out.
CLAIM_IDS: List[str] = ["SD-071"]
CARVEOUT_ENUMERATED_CLAIMS = frozenset({"SD-071"})
CARVEOUT_FORBIDDEN_CLAIMS = frozenset({"MECH-120", "MECH-121"})
EXPERIMENT_PURPOSE = "evidence"
SLEEP_DRIVER_PATTERN = "manual-cycle-loop"

# The V3-EXQ-778a / 778c / 778g 8-seed set, reused EXACTLY so this run's per-seed ratios
# are a within-seed comparison against the recorded diagnostic distribution rather than
# two independent samples. n = 8 satisfies SD-071's stated "n>=8".
SEEDS = [42, 7, 123, 2024, 99, 7777, 314, 1000]
SIGMAS = [0.0, 0.25, 0.5, 1.0, 2.0]
WARM_STEPS = 40
ARMS = ["INJECTED", "NULL"]
# sws content-scale ladder rungs. 0.0 and 1.0 are ALREADY computed by the main sweep
# (they are the null and injected arms) and sws_only_integrity_at_sigma reproduces those
# cells exactly, so only the intermediate rungs cost extra compute.
LADDER_SCALES = [0.0, 0.25, 0.5, 1.0]
LADDER_EXTRA_SCALES = [0.25, 0.5]

# --- PRE-REGISTERED THRESHOLDS (constants; never derived from this run's own stats) ---
# C1: the harness's own ceiling, so this run's criterion is the same bar 778c/778g used.
NULL_SLOPE_RATIO_CEILING = H.NULL_SLOPE_RATIO_CEILING            # 0.25
# C2 readiness: gates the literal DENOMINATOR of null_slope_ratio -- the same statistic
# C1 routes on. A divide-by-almost-zero tripwire belongs orders below the working range
# (recorded reference min |injected| is 0.0967 nrem / 0.3232 sws), and `met: true` here
# says only "the ratio was computable", never "the substrate was ready" in any stronger
# sense. Largely redundant with the harness's own NULL_MIN_INJECTED_SLOPE = 1e-9
# (harness:1409), which reports the phase UNAVAILABLE rather than scoring it.
INJECTED_SLOPE_FLOOR = 1e-6
# C3: the 0.01 floor SD-071 names verbatim. Absolute, on the SIGNED range
# max(pos) - min(pos) across the content_scale>0 rungs (no abs()).
LADDER_SPREAD_FLOOR = 0.01
# RECORDED, NOT GATED -- 778g's diagnostic signal-ratio bar. See the docstring section
# "TWO CHECKS THAT ARE MEASURED AND RECORDED BUT DELIBERATELY NOT GATED".
LADDER_SIGNAL_RATIO = 3.0

# BOTH legs are gated by C1. This is the difference from 778g, which scoped C1 to sws
# alone because the rem leg was known degenerate and would have failed it regardless.
# SD-071's falsifier says "either leg", and the two legs it names are these.
GATED_PHASES = ("nrem", "sws")
# The rem leg is measured and reported as CONTEXT only. It is known degenerate at both
# clamp rails (778c: exactly 0.0 on 5/8 seeds off a saturated constant, off-scale
# 1801-9143 on 3/8) and is owned by the GOV-FANOUT-1 portfolio V3-EXQ-778d/e/f. SD-071
# does not claim it, so gating it here would fail the run on a leg the claim excludes.
CONTEXT_PHASES = ("rem",)
# C3's ladder is sws-only, by SD-071's own wording ("the sws leg's content-scale ladder").
LADDER_PHASE = "sws"


def _assert_carveout_conditions() -> Dict[str, Any]:
    """Refuse to run unless the harness carve-out's two conditions hold.

    The carve-out (user decision 2026-09-18T19:47:16Z; REE_assembly d202c2d73c, ree-v3
    6371c9fc8a) permits EXPERIMENT_PURPOSE="evidence" on this harness ONLY for a run
    whose claim_ids are solely instrument-validity claims -- enumerated {SD-071} -- and
    that tags neither MECH-120 nor MECH-121. Every other run stays diagnostic.

    This is asserted rather than commented because the failure is SILENT otherwise: a
    later edit adding a "context" tag (the shape 778c/778g use, and the natural thing to
    copy) would void the carve-out while the run still reported itself as evidence.
    Raises AssertionError -> non-zero exit -> the runner classifies ERROR, which is the
    correct loud failure.
    """
    tagged = set(CLAIM_IDS)
    extra = sorted(tagged - CARVEOUT_ENUMERATED_CLAIMS)
    forbidden = sorted(tagged & CARVEOUT_FORBIDDEN_CLAIMS)
    if EXPERIMENT_PURPOSE != "diagnostic":
        assert not extra, (
            "SD-068 harness carve-out condition (i) VIOLATED: EXPERIMENT_PURPOSE="
            f"{EXPERIMENT_PURPOSE!r} with claim_ids {sorted(tagged)}, which include "
            f"{extra} outside the enumerated instrument-validity set "
            f"{sorted(CARVEOUT_ENUMERATED_CLAIMS)}. The carve-out is an ENUMERATION, "
            "not a category: either drop the extra tag(s), or set "
            'EXPERIMENT_PURPOSE="diagnostic", or get the enumeration widened by a '
            "fresh governance decision. See "
            "REE_assembly/docs/architecture/sd_068_consolidation_lesion_harness.md "
            "'Prerequisite caveat -- MECH-121 hold'."
        )
        assert not forbidden, (
            "SD-068 harness carve-out condition (ii) VIOLATED: a non-diagnostic run on "
            f"this harness must tag NEITHER MECH-120 NOR MECH-121, but tags {forbidden}."
        )
    print(
        "  [guard] carve-out conditions hold: purpose="
        f"{EXPERIMENT_PURPOSE} claim_ids={CLAIM_IDS} "
        "(enumerated instrument-validity set only; MECH-120/MECH-121 absent)",
        flush=True,
    )
    return {
        "purpose": EXPERIMENT_PURPOSE,
        "claim_ids": list(CLAIM_IDS),
        "enumerated_set": sorted(CARVEOUT_ENUMERATED_CLAIMS),
        "forbidden_absent": sorted(CARVEOUT_FORBIDDEN_CLAIMS),
        "condition_i_claim_ids_enumerated_only": True,
        "condition_ii_no_mech120_mech121": True,
        "decision": "user 2026-09-18T19:47:16Z, orchestrate-20260918-1840-cloud4, GFLAG-0319",
        "amendment_commits": {"REE_assembly": "d202c2d73c", "ree-v3": "6371c9fc8a"},
    }


def _fmt(v: Any) -> str:
    """ASCII-safe float rendering that keeps UNAVAILABLE legible."""
    if v is None:
        return "n/a"
    if v == H.UNAVAILABLE or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return f"{float(v):.6f}"
    return str(v)


def _finite(v: Any) -> bool:
    return (
        isinstance(v, (int, float))
        and not isinstance(v, bool)
        and v != H.UNAVAILABLE
        and not math.isnan(float(v))
    )


def _num(v: Any) -> Any:
    """Flat-readout encoder: drop non-finite, coerce bool to int (indexer contract)."""
    if isinstance(v, bool):
        return int(v)
    return float(v) if _finite(v) else None


# --- THE SHIPPED PRECONDITION PREDICATES -------------------------------------------
# Each declared readiness precondition is a PER-SEED boolean aggregated with `all(...)`
# over seeds, so the faithful re-expression for `assert_anchor_reachable` is:
# score_fn = the per-seed predicate, threshold = 1.0.
#
# Factored to MODULE LEVEL precisely so the live scoring path in `_score_seed` and the
# setup-time reachability guards run THE SAME CALLABLE. Scoring a guard with a
# re-implementation would defeat its purpose -- the defect class being guarded against
# IS a mis-specified predicate (SD-068 REM fanout autopsy, Learning 1).
#
# A `cell` is one seed's recorded values:
#   {"injected_slope_nrem": float, "injected_slope_sws": float,
#    "ladder_slopes": {content_scale: sigma_slope}}
# `ladder_slopes` keys may be float or str (the manifest round-trips them as str).


def _ladder_by_scale(cell: Dict[str, Any]) -> Dict[float, float]:
    """Float-keyed view of a cell's ladder, so recorded (str-keyed) cells score too."""
    return {float(k): v for k, v in (cell.get("ladder_slopes") or {}).items()}


def _zero_rung_slope(cell: Dict[str, Any]) -> float:
    """The content_scale=0 rung's sigma-slope (nan if that rung is absent)."""
    return _ladder_by_scale(cell).get(0.0, float("nan"))


def _positive_rung_slopes(cell: Dict[str, Any]) -> List[float]:
    """The finite content_scale>0 rung slopes, in LADDER_SCALES order."""
    ladder = _ladder_by_scale(cell)
    return [ladder[c] for c in LADDER_SCALES if c > 0.0 and _finite(ladder.get(c))]


def _injected_slope_supra_floor_nrem(cell: Dict[str, Any]) -> bool:
    """C2 readiness for the nrem leg, per seed: that leg's ratio DENOMINATOR clears the floor.

    Precondition `injected_arm_nrem_sigma_slope_supra_floor`. FLOOR-shaped, INCLUSIVE.
    """
    inj = cell.get("injected_slope_nrem", H.UNAVAILABLE)
    return bool(_finite(inj) and abs(float(inj)) >= INJECTED_SLOPE_FLOOR)


def _injected_slope_supra_floor_sws(cell: Dict[str, Any]) -> bool:
    """C2 readiness for the sws leg, per seed. FLOOR-shaped, INCLUSIVE.

    Precondition `injected_arm_sws_sigma_slope_supra_floor`. Declared SEPARATELY from
    the nrem one on purpose: C1 gates BOTH legs, so each leg's denominator is its own
    recomputable (measured, threshold) pair. One combined entry could not reproduce
    `met` from a single statistic.
    """
    inj = cell.get("injected_slope_sws", H.UNAVAILABLE)
    return bool(_finite(inj) and abs(float(inj)) >= INJECTED_SLOPE_FLOOR)


def _ladder_slope_spread_supra_floor(cell: Dict[str, Any]) -> bool:
    """C3, per seed: the sws ladder response VARIES with content amplitude.

    Precondition `ladder_content_slope_spread_supra_floor`. This is the 0.01 floor
    SD-071 names. Separates "tracks content" from "detects a non-empty store", which is
    what the cosine readout's scale-invariance makes necessary. FLOOR-shaped, STRICT
    (`>`) -- comparator preserved exactly as 778g shipped it.
    """
    pos = _positive_rung_slopes(cell)
    return bool(pos) and (max(pos) - min(pos)) > LADDER_SPREAD_FLOOR


def _ladder_signal_ratio_supra_floor(cell: Dict[str, Any]) -> bool:
    """RECORDED, NOT GATED: the ladder signal-ratio leg. FLOOR-shaped, STRICT (`>`).

    Precondition `ladder_content_signal_ratio_supra_floor`. Relative, not absolute: the
    zero-content rung carries a small nonzero slope from the sigma=0 store-is-exactly-
    zero discontinuity, so an absolute floor here would be met by that artifact alone.
    778g gated this at 3.0x as a diagnostic; SD-071's falsifier names only the 0.01
    spread floor, so this run records it without gating on it.
    """
    pos = _positive_rung_slopes(cell)
    zero_slope = _zero_rung_slope(cell)
    return bool(pos) and _finite(zero_slope) and min(abs(v) for v in pos) > (
        LADDER_SIGNAL_RATIO * max(abs(zero_slope), 1e-12)
    )


# The KNOWN-HEALTHY POSITIVE CONTROL for every precondition above, frozen as literals.
# Per-seed recorded values of the completed V3-EXQ-778g run
# `v3_exq_sd068_sws_content_scored_readout_diagnostic_20260718T130139Z_v3` (outcome
# PASS, label `sws_readout_content_contingent_validated`, 8/8 seeds). That run is the
# established-health control for exactly these gates: same harness, same seeds, same
# sigma grid, same ladder, and it recorded BOTH legs' full `null_control` per seed --
# which is what lets it anchor the nrem denominator as well as the sws one. Frozen as
# literals so the guards need zero compute and cannot drift with the substrate.
#   injected_slope_nrem -> arm_results[i].null_control.injected_slope_nrem
#   injected_slope_sws  -> arm_results[i].null_control.injected_slope_sws
#   ladder_slopes       -> arm_results[i].ladder_slopes
_REFERENCE_778G_HEALTHY: List[Dict[str, Any]] = [
    {"seed": 42,
     "injected_slope_nrem": 0.09671970039873905,
     "injected_slope_sws": 0.3245777508559374,
     "ladder_slopes": {0.0: 0.031980300752911715, 0.25: 0.4487783819465478,
                       0.5: 0.4301830271776864, 1.0: 0.3245777508559374}},
    {"seed": 7,
     "injected_slope_nrem": 0.09867377313414635,
     "injected_slope_sws": 0.343413371789214,
     "ladder_slopes": {0.0: 0.0489227172889514, 0.25: 0.47966099185402555,
                       0.5: 0.4588255786464148, 1.0: 0.343413371789214}},
    {"seed": 123,
     "injected_slope_nrem": 0.09667174055435042,
     "injected_slope_sws": 0.35778482921581956,
     "ladder_slopes": {0.0: 0.046219069304061125, 0.25: 0.4686332760975656,
                       0.5: 0.45804626607801335, 1.0: 0.35778482921581956}},
    {"seed": 2024,
     "injected_slope_nrem": 0.0967143029598984,
     "injected_slope_sws": 0.3231758770786943,
     "ladder_slopes": {0.0: 0.034861253030248916, 0.25: 0.45336767073688355,
                       0.5: 0.43151976762780286, 1.0: 0.3231758770786943}},
    {"seed": 99,
     "injected_slope_nrem": 0.09865533298549493,
     "injected_slope_sws": 0.3374661845218423,
     "ladder_slopes": {0.0: 0.04022558314609341, 0.25: 0.46361007502789997,
                       0.5: 0.4458609185345739, 1.0: 0.3374661845218423}},
    {"seed": 7777,
     "injected_slope_nrem": 0.09687813173177665,
     "injected_slope_sws": 0.3404493347147325,
     "ladder_slopes": {0.0: 0.042277684759551445, 0.25: 0.4655831592527425,
                       0.5: 0.4479344040397212, 1.0: 0.3404493347147325}},
    {"seed": 314,
     "injected_slope_nrem": 0.0974091445154601,
     "injected_slope_sws": 0.34553358192374584,
     "ladder_slopes": {0.0: 0.04762528844294138, 0.25: 0.47052610168852027,
                       0.5: 0.4524028295265678, 1.0: 0.34553358192374584}},
    {"seed": 1000,
     "injected_slope_nrem": 0.09708835577665026,
     "injected_slope_sws": 0.34239332795877564,
     "ladder_slopes": {0.0: 0.05012439275524231, 0.25: 0.47684803545852555,
                       0.5: 0.4536756876553808, 1.0: 0.34239332795877564}},
]
_REFERENCE_SOURCE = (
    "V3-EXQ-778g completed run "
    "v3_exq_sd068_sws_content_scored_readout_diagnostic_20260718T130139Z_v3 "
    "(outcome PASS, sws_readout_content_contingent_validated, 8/8 seeds; per-seed "
    "null_control.injected_slope_nrem / injected_slope_sws + ladder_slopes recorded "
    "in arm_results)"
)
# Every precondition is a per-seed boolean aggregated with `all(...)` over seeds, so the
# reachability threshold is the FRACTION 1.0 -- every reference cell must score.
ANCHOR_ALL_SEEDS_FRAC = 1.0


def _ladder_slope(
    *, seed: int, content_scale: float, sigmas: List[float], warm: int,
    cached: Dict[float, Dict[str, Dict[str, float]]] = None,
) -> Tuple[float, List[float]]:
    """Sigma-slope of the sws completion error at one content_scale.

    Uses the INJECTED arm's own undamaged margin at this rung as the denominator, so each
    rung is expressed in its own units and the rungs are directly comparable as
    fractions-of-own-discriminability-lost. `cached` supplies already-computed sws rows
    (the main sweep's null / injected arms) so rungs 0.0 and 1.0 cost nothing.
    """
    margins: List[float] = []
    m_clean = None
    for s in sigmas:
        if cached is not None and s in cached:
            row = cached[s]["sws"]
        else:
            row = H.sws_only_integrity_at_sigma(
                seed=seed, sigma=s, warm_steps=warm, content_scale=content_scale
            )["sws"]
        margins.append(float(row.get("sws_completion_margin", float("nan"))))
        if m_clean is None:
            m_clean = float(row.get("sws_completion_margin_clean", 0.0))

    if m_clean is None or abs(m_clean) <= 1e-9:
        # content_scale = 0 -> no injected discriminability to lose. The error series is
        # referenced to the margin itself, which is ~0 at every sigma, so the slope is
        # ~0. That is the C3 zero-rung property, measured rather than assumed.
        errs = [(-m) for m in margins]
    else:
        errs = [1.0 - (m / m_clean) for m in margins]

    xs = [s for s, e in zip(sigmas, errs) if not math.isnan(e)]
    ys = [e for e in errs if not math.isnan(e)]
    slope = float(H._lin_slope(xs, ys)) if len(ys) >= 2 else float("nan")
    return slope, errs


def _score_seed(
    control: Dict[str, float], ladder: Dict[float, float]
) -> Dict[str, Any]:
    """Score one seed. C1 is COHORT-level (a CI over seeds) so it is NOT decided here.

    What IS per-seed: the two legs' ratios and denominators (C1's inputs), C2 readiness
    on each leg, C3's ladder legs, and each leg's harness-reported degeneracy flags.
    """
    cell: Dict[str, Any] = {"ladder_slopes": dict(ladder)}
    legs: Dict[str, Any] = {}
    for leg in GATED_PHASES:
        inj = control.get(f"injected_slope_{leg}", H.UNAVAILABLE)
        cell[f"injected_slope_{leg}"] = inj
        legs[leg] = {
            "null_slope_ratio": control.get(f"null_slope_ratio_{leg}", H.UNAVAILABLE),
            "injected_slope": inj,
            "null_slope": control.get(f"null_slope_{leg}", H.UNAVAILABLE),
            # Harness-reported. A null series that is a saturated CONSTANT has slope
            # exactly 0 and therefore ratio exactly 0.0, which would clear the ceiling
            # as a clean pass -- it is the ABSENCE of a measurement, not a reading.
            "null_series_degenerate": bool(
                float(control.get(f"null_series_degenerate_{leg}", 0.0) or 0.0) >= 1.0
            ),
            "null_control_available": bool(
                float(control.get(f"null_control_available_{leg}", 0.0) or 0.0) >= 1.0
            ),
            "content_contingent": bool(
                float(control.get(f"content_contingent_{leg}", 0.0) or 0.0) >= 1.0
            ),
        }

    c2_nrem = _injected_slope_supra_floor_nrem(cell)
    c2_sws = _injected_slope_supra_floor_sws(cell)
    c3_spread = _ladder_slope_spread_supra_floor(cell)
    c3_signal = _ladder_signal_ratio_supra_floor(cell)   # recorded, not gated

    return {
        "legs": legs,
        "ladder_slopes": {str(k): v for k, v in ladder.items()},
        "zero_rung_slope": _zero_rung_slope(cell),
        "C2_nrem_denominator_supra_floor": c2_nrem,
        "C2_sws_denominator_supra_floor": c2_sws,
        "C2_ratio_interpretable": bool(c2_nrem and c2_sws),
        "C3_ladder_spread_tracks_content": c3_spread,
        "ladder_signal_ratio_supra_floor_recorded_not_gated": c3_signal,
        "context_null_slope_ratio": {
            p: control.get(f"null_slope_ratio_{p}", H.UNAVAILABLE)
            for p in CONTEXT_PHASES
        },
    }


def run_experiment(*, dry_run: bool = False) -> Dict[str, Any]:
    seeds = [SEEDS[0]] if dry_run else SEEDS
    warm = 8 if dry_run else WARM_STEPS
    sigmas = [0.0, 0.5, 2.0] if dry_run else SIGMAS
    extra_scales = [0.5] if dry_run else LADDER_EXTRA_SCALES
    ladder_scales = [0.0] + extra_scales + [1.0]

    print(
        "V3-EXQ-1058: SD-071 consolidation-readout instrument validity (EVIDENCE run)",
        flush=True,
    )
    print(
        f"  seeds={seeds} sigmas={sigmas} warm_steps={warm} arms={ARMS} "
        f"gated_legs={list(GATED_PHASES)} ladder={ladder_scales} dry_run={dry_run}",
        flush=True,
    )

    # ---- CARVE-OUT GUARD (setup-time, BEFORE any compute) ---------------------------
    carveout = _assert_carveout_conditions()

    # ---- READINESS-ANCHOR REACHABILITY GUARDS (setup-time, BEFORE any compute) -------
    # Every declared readiness precondition is replayed through THE SHIPPED PREDICATE
    # against the frozen known-healthy V3-EXQ-778g reference. A precondition its own
    # healthy control cannot pass is a guaranteed false negative: it would report
    # met=false on every run and mislabel an instrument-specification gap as a substrate
    # verdict (the confirmed V3-EXQ-778d defect; experiments/_lib/readiness_anchor.py).
    # Raises AnchorUnreachable (an AssertionError) -> non-zero exit -> runner ERROR,
    # which is the correct loud failure. Runs on dry-run too: the reference is frozen,
    # so the guards are dry-run-invariant and the smoke exercises them.
    anchor_guards: Dict[str, Any] = {}
    for _anchor_name, _score_fn in (
        ("injected_arm_nrem_sigma_slope_supra_floor", _injected_slope_supra_floor_nrem),
        ("injected_arm_sws_sigma_slope_supra_floor", _injected_slope_supra_floor_sws),
        ("ladder_content_slope_spread_supra_floor", _ladder_slope_spread_supra_floor),
        ("ladder_content_signal_ratio_supra_floor", _ladder_signal_ratio_supra_floor),
    ):
        _g = assert_anchor_reachable(
            anchor_name=_anchor_name,
            reference_cells=_REFERENCE_778G_HEALTHY,
            score_fn=_score_fn,
            threshold=ANCHOR_ALL_SEEDS_FRAC,
            reference_source=_REFERENCE_SOURCE,
            # margin_cells=0 is KNOWN AND INTENDED (readiness_anchor rule 4). The
            # shipped aggregation is `all(...)` over seeds, so the gate is already the
            # maximum expressible fraction (1.0) and no cell-level headroom above it can
            # be expressed. The headroom that matters is PER-CELL and it is large: the
            # reference's tightest cells sit at injected slope 0.0967 (nrem) and 0.3232
            # (sws) against a 1e-6 floor, ladder spread 0.1108 against 0.01, and signal
            # ratio 6.83 against 3.0. None is a thin-margin pass.
        )
        anchor_guards[_anchor_name] = _g
        print(
            f"  [guard] anchor reachable: {_anchor_name} -- the known-healthy "
            f"V3-EXQ-778g reference scores {_g['n_reference_scored_true']}/"
            f"{_g['n_reference_cells']} = {_g['reference_score']:.3f} under the shipped "
            f"predicate (gate {ANCHOR_ALL_SEEDS_FRAC:.2f})",
            flush=True,
        )

    config_slice = {
        "sigmas": sigmas,
        "warm_steps": warm,
        "arms": list(ARMS),
        "ladder_scales": ladder_scales,
        "null_slope_ratio_ceiling": NULL_SLOPE_RATIO_CEILING,
        "injected_slope_floor": INJECTED_SLOPE_FLOOR,
        "ladder_spread_floor": LADDER_SPREAD_FLOOR,
        "ladder_signal_ratio_recorded_not_gated": LADDER_SIGNAL_RATIO,
        "gated_phases": list(GATED_PHASES),
        "context_phases": list(CONTEXT_PHASES),
        "ladder_phase": LADDER_PHASE,
        "shy_decay_rate": 0.85,
        "body_obs_dim": H.BODY_OBS_DIM,
        "world_obs_dim": H.WORLD_OBS_DIM,
        "action_dim": H.ACTION_DIM,
        "harm_obs_dim": H.HARM_OBS_DIM,
    }

    arm_results: List[Dict[str, Any]] = []
    seed_scores: List[Dict[str, Any]] = []
    total_eps = len(sigmas)

    for seed in seeds:
        print(f"Seed {seed} Condition SD071_INSTRUMENT_VALIDITY", flush=True)
        with arm_cell(
            seed,
            config_slice=config_slice,
            script_path=Path(__file__),
            config_slice_declared=True,
        ) as cell:
            inj_pr: Dict[float, Dict[str, Dict[str, float]]] = {}
            null_pr: Dict[float, Dict[str, Dict[str, float]]] = {}
            for i, s in enumerate(sigmas):
                inj_pr[s] = H.phase_integrity_at_sigma(
                    seed=seed, sigma=s, warm_steps=warm, content_scale=1.0
                )
                null_pr[s] = H.phase_integrity_at_sigma(
                    seed=seed, sigma=s, warm_steps=warm, content_scale=0.0
                )
                print(
                    f"  [train] sd071_validity seed={seed} ep {i + 1}/{total_eps} "
                    f"sigma={s}",
                    flush=True,
                )

            control = H.run_null_content_control(
                seed=seed,
                sigmas=list(sigmas),
                warm_steps=warm,
                injected_pr_by_sigma=inj_pr,
                null_pr_by_sigma=null_pr,
            )

            # C3 ladder (sws leg only). Rungs 0.0 and 1.0 REUSE the main sweep's cells
            # (identical RNG stream by construction -- sws_only_integrity_at_sigma
            # reproduces phase_integrity_at_sigma's sws cell exactly), so only the
            # intermediate rungs cost extra compute.
            ladder: Dict[float, float] = {}
            ladder_series: Dict[str, List[float]] = {}
            for cs in ladder_scales:
                cache = None
                if cs == 0.0:
                    cache = null_pr
                elif cs == 1.0:
                    cache = inj_pr
                slope, errs = _ladder_slope(
                    seed=seed, content_scale=cs, sigmas=list(sigmas),
                    warm=warm, cached=cache,
                )
                ladder[cs] = slope
                ladder_series[str(cs)] = errs
                print(
                    f"  [ladder] seed={seed} content_scale={cs} "
                    f"sigma_slope={_fmt(slope)}",
                    flush=True,
                )

            score = _score_seed(control, ladder)
            row: Dict[str, Any] = {
                "seed": seed,
                "arm": "SD071_INSTRUMENT_VALIDITY",
                "arms_compared": list(ARMS),
                "sigmas": list(sigmas),
                "null_control": control,
                "ladder_scales": ladder_scales,
                "ladder_slopes": score["ladder_slopes"],
                "ladder_error_series": ladder_series,
                "zero_rung_slope": score["zero_rung_slope"],
                # Per-leg, flattened so a reader (and subgroup_ratio_stats below) can
                # key on the leg without walking the nested null_control block.
                "null_slope_ratio_nrem": score["legs"]["nrem"]["null_slope_ratio"],
                "null_slope_ratio_sws": score["legs"]["sws"]["null_slope_ratio"],
                "injected_slope_nrem": score["legs"]["nrem"]["injected_slope"],
                "injected_slope_sws": score["legs"]["sws"]["injected_slope"],
                "null_slope_nrem": score["legs"]["nrem"]["null_slope"],
                "null_slope_sws": score["legs"]["sws"]["null_slope"],
                "legs": score["legs"],
                "context_null_slope_ratio": score["context_null_slope_ratio"],
                # BOTH arms' full per-sigma internals recorded, per the Experimental
                # Recording Standard (the OFF/NULL arm as richly as the INJECTED one).
                "integrity_injected": {str(s): inj_pr[s] for s in sigmas},
                "integrity_null": {str(s): null_pr[s] for s in sigmas},
                "C2_nrem_denominator_supra_floor":
                    score["C2_nrem_denominator_supra_floor"],
                "C2_sws_denominator_supra_floor":
                    score["C2_sws_denominator_supra_floor"],
                "C2_ratio_interpretable": score["C2_ratio_interpretable"],
                "C3_ladder_spread_tracks_content":
                    score["C3_ladder_spread_tracks_content"],
                "ladder_signal_ratio_supra_floor_recorded_not_gated":
                    score["ladder_signal_ratio_supra_floor_recorded_not_gated"],
            }
            cell.stamp(row)

        arm_results.append(row)
        seed_scores.append(score)

        print(
            "  null_slope_ratio (GATED legs, ceiling "
            f"{NULL_SLOPE_RATIO_CEILING}): "
            f"nrem={_fmt(score['legs']['nrem']['null_slope_ratio'])} "
            f"sws={_fmt(score['legs']['sws']['null_slope_ratio'])} | context: "
            + " ".join(
                f"{p}={_fmt(score['context_null_slope_ratio'][p])}"
                for p in CONTEXT_PHASES
            ),
            flush=True,
        )
        print(
            f"  C2={score['C2_ratio_interpretable']} "
            f"(nrem={score['C2_nrem_denominator_supra_floor']} "
            f"sws={score['C2_sws_denominator_supra_floor']}) "
            f"C3_spread={score['C3_ladder_spread_tracks_content']} "
            f"[recorded-not-gated signal_ratio="
            f"{score['ladder_signal_ratio_supra_floor_recorded_not_gated']}]",
            flush=True,
        )
        # A per-seed verdict line is required by the runner's progress parser. C1 is a
        # COHORT criterion (a CI over seeds), so what a single seed can carry is C2+C3
        # only -- named explicitly so the line is not misread as a C1 verdict.
        _seed_ok = bool(
            score["C2_ratio_interpretable"]
            and score["C3_ladder_spread_tracks_content"]
        )
        print(f"verdict: {'PASS' if _seed_ok else 'FAIL'}  (seed-level C2+C3 only; "
              "C1 is cohort-level)", flush=True)

    n = len(seed_scores)
    # SD-071's falsifier is stated at n>=8 ('a re-run at n>=8'), and `need_seeds` below
    # reports the REALIZED n. Assert the INTENDED floor so a future edit that shrinks
    # SEEDS cannot silently produce a result the claim's own criterion does not cover
    # (red-team finding F4(d); cannot fire as shipped -- len(SEEDS) == 8).
    assert dry_run or n >= 8, (
        "SD-071's pre-registered criterion is stated at n>=8; this run has n=%d. "
        "Restore SEEDS to at least 8 seeds, or the result does not answer the claim."
        % n
    )

    # ---- C1: COHORT-LEVEL, per gated leg -------------------------------------------
    # `subgroup_ratio_stats` is the harness's own shared helper (:1446) -- the same one
    # 778g used -- so the interval is constructed identically and the V3-EXQ-778h
    # subgroup defect (summary stats pooled over seeds the criterion excludes) cannot
    # recur. The subgroup predicate here is the C2 readiness one FOR THAT LEG: a seed
    # whose denominator never cleared the floor has a 0/0 ratio that is an artifact, not
    # a reading. The exclusion is EMITTED (subgroup_n / excluded_seeds), never silent.
    leg_stats: Dict[str, Any] = {}
    for leg in GATED_PHASES:
        eligible = (
            _injected_slope_supra_floor_nrem if leg == "nrem"
            else _injected_slope_supra_floor_sws
        )
        st = H.subgroup_ratio_stats(
            arm_results,
            eligible=eligible,
            value=lambda r, _leg=leg: r.get(f"null_slope_ratio_{_leg}", H.UNAVAILABLE),
            ceiling=NULL_SLOPE_RATIO_CEILING,
        )
        # WHICH SIDE of the ceiling the interval fell on. `ceiling_inside_ci95 == False`
        # is two-way ambiguous (below = content-contingent, above = confounded) and C1
        # as SD-071 pre-registers it cannot separate them. Recorded, not gated; used by
        # the self-route, the direction, and the non-degeneracy net.
        st["ci95_high_below_ceiling"] = bool(
            _finite(st.get("ci95_high"))
            and float(st["ci95_high"]) < NULL_SLOPE_RATIO_CEILING
        )
        st["ci95_low_above_ceiling"] = bool(
            _finite(st.get("ci95_low"))
            and float(st["ci95_low"]) > NULL_SLOPE_RATIO_CEILING
        )
        st["per_seed_null_slope_ratio"] = [
            s["legs"][leg]["null_slope_ratio"] for s in seed_scores
        ]
        st["per_seed_injected_slope"] = [
            s["legs"][leg]["injected_slope"] for s in seed_scores
        ]
        st["per_seed_null_slope"] = [s["legs"][leg]["null_slope"] for s in seed_scores]
        st["n_seeds_content_contingent"] = sum(
            1 for s in seed_scores if s["legs"][leg]["content_contingent"]
        )
        st["n_seeds_null_series_degenerate"] = sum(
            1 for s in seed_scores if s["legs"][leg]["null_series_degenerate"]
        )
        st["n_seeds_null_control_unavailable"] = sum(
            1 for s in seed_scores if not s["legs"][leg]["null_control_available"]
        )
        leg_stats[leg] = st

    # C1 as pre-registered, verbatim: ceiling_inside_ci95 FALSE on BOTH legs.
    c1_per_leg = {
        leg: (not bool(leg_stats[leg]["ceiling_inside_ci95"])) for leg in GATED_PHASES
    }
    c1_all = all(c1_per_leg.values())

    # C2 readiness, aggregated per leg with all(...) over seeds.
    c2_nrem_all = all(s["C2_nrem_denominator_supra_floor"] for s in seed_scores)
    c2_sws_all = all(s["C2_sws_denominator_supra_floor"] for s in seed_scores)
    readiness_ok = bool(c2_nrem_all and c2_sws_all)

    # C3 (sws ladder spread vs the 0.01 floor), aggregated with all(...) over seeds.
    c3_all = all(s["C3_ladder_spread_tracks_content"] for s in seed_scores)
    # Recorded, not gated.
    c3_signal_all = all(
        s["ladder_signal_ratio_supra_floor_recorded_not_gated"] for s in seed_scores
    )

    overall_pass = bool(readiness_ok and c1_all and c3_all)

    # ---- worst-cell statistics for the recomputable preconditions -------------------
    # `measured` must be the SAME statistic `met` tests, and `met` is an all(...) claim,
    # so report the WORST CELL, never a mean.
    def _worst_min_abs(key: str) -> Tuple[float, Any]:
        worst, who = None, None
        for s in seed_scores:
            v = s["legs"][key]["injected_slope"] if key in GATED_PHASES else None
            v = abs(float(v)) if _finite(v) else 0.0
            if worst is None or v < worst:
                worst, who = v, s["legs"][key]
        return (worst if worst is not None else 0.0), who

    min_inj_nrem, _ = _worst_min_abs("nrem")
    min_inj_sws, _ = _worst_min_abs("sws")

    ladder_spreads: List[float] = []
    ladder_signal_ratios: List[float] = []
    for s in seed_scores:
        pos = [
            v for k, v in s["ladder_slopes"].items()
            if float(k) > 0.0 and _finite(v)
        ]
        zero_slope = s["zero_rung_slope"]
        # A seed with NO content-bearing rung fails both legs by construction
        # (`bool(pos)` guards each), so it contributes a 0.0 worst case rather than
        # being skipped -- otherwise the min over seeds could clear a floor the shipped
        # predicate did not.
        ladder_spreads.append(max(pos) - min(pos) if pos else 0.0)
        ladder_signal_ratios.append(
            min(abs(v) for v in pos) / max(abs(float(zero_slope)), 1e-12)
            if pos and _finite(zero_slope) else 0.0
        )
    worst_spread = min(ladder_spreads) if ladder_spreads else 0.0
    worst_signal = min(ladder_signal_ratios) if ladder_signal_ratios else 0.0
    worst_spread_seed = (
        seed_scores[ladder_spreads.index(worst_spread)]["legs"]["sws"]
        if ladder_spreads else None
    )

    # ---- NON-DEGENERACY NET (evidence run -> top-level non_degenerate) --------------
    degeneracy_reasons: List[str] = []
    for leg in GATED_PHASES:
        st = leg_stats[leg]
        if st["n_seeds_null_series_degenerate"] > 0:
            degeneracy_reasons.append(
                f"{leg}: null series is a saturated CONSTANT on "
                f"{st['n_seeds_null_series_degenerate']}/{n} seed(s) -- slope exactly 0, "
                "so ratio 0.0 clears the ceiling as the ABSENCE of a measurement"
            )
        if st["n_seeds_null_control_unavailable"] > 0:
            degeneracy_reasons.append(
                f"{leg}: null control UNAVAILABLE on "
                f"{st['n_seeds_null_control_unavailable']}/{n} seed(s)"
            )
        if st["ci95_low_above_ceiling"]:
            degeneracy_reasons.append(
                f"{leg}: CI95 [{_fmt(st['ci95_low'])}, {_fmt(st['ci95_high'])}] sits "
                f"entirely ABOVE the {NULL_SLOPE_RATIO_CEILING} ceiling -- the leg is "
                "CONFOUNDED. ceiling_inside_ci95 is False here for the opposite reason "
                "SD-071 asserts, so a C1 'pass' on this leg must not score as support"
            )
        if st["subgroup_n"] < 2:
            degeneracy_reasons.append(
                f"{leg}: subgroup_n={st['subgroup_n']} -- no CI95 is computable at n<2"
            )
    if not readiness_ok:
        degeneracy_reasons.append(
            "C2: an injected-arm sigma-slope is below floor, so the ratio is 0/0 and "
            "the null control never discriminated"
        )
    if not ladder_spreads:
        degeneracy_reasons.append("C3: the sws ladder produced no usable content>0 slopes")
    non_degenerate = not degeneracy_reasons

    # ---- SELF-ROUTE ----------------------------------------------------------------
    # Readiness dominates: a below-floor denominator means the control never
    # discriminated, which is a requeue, NEVER a substrate verdict.
    if not readiness_ok:
        label = "substrate_not_ready_requeue"
    elif any(leg_stats[leg]["ci95_low_above_ceiling"] for leg in GATED_PHASES):
        label = "consolidation_readout_leg_confounded"
    elif not c1_all:
        unresolved = [leg for leg in GATED_PHASES if not c1_per_leg[leg]]
        label = "ceiling_inside_ci95_verdict_unresolved_" + "_".join(unresolved)
    elif not c3_all:
        # C1 passed but the sws ladder did not -- exactly the scale-invariance artifact
        # C3 is pre-registered to catch.
        label = "sws_null_pass_is_scale_invariance_artifact"
    else:
        label = "consolidation_readouts_content_contingent_instruments_validated"

    # ---- EVIDENCE DIRECTION --------------------------------------------------------
    # SUPPORTS only when C1 passes for the reason SD-071 asserts (interval BELOW the
    # ceiling on both legs) and C3 holds. A confounded leg WEAKENS. An un-exercised
    # instrument is UNKNOWN, not weak evidence.
    #
    # THE CONFOUNDED-LEG BRANCH IS EXPLICIT AND COMES FIRST among the pass cases
    # (red-team finding 2, CONTESTED, confirmed against this file's own logic). A leg
    # whose CI95 sits entirely ABOVE the ceiling leaves `ceiling_inside_ci95` False, so
    # C1 as SD-071 pre-registers it PASSES and `overall_pass` is True -- while the
    # substantive reading is that the leg is fully CONFOUNDED, which refutes SD-071.
    # An earlier draft fell through to `mixed` there, contradicting this file's own
    # docstring. `outcome` is deliberately left as the pre-registered rule computes it
    # (changing the pass rule would change a criterion the claim itself fixes, which is
    # not this run's to do); the honest reading is carried by `evidence_direction`,
    # the self-route label `consolidation_readout_leg_confounded`, and
    # `non_degenerate: false`, which excludes the run from confidence scoring. So a
    # confounded leg can produce `outcome: PASS` -- that combination is DIAGNOSTIC of
    # this exact case and is spelled out in `outcome_vs_direction_note` below.
    if not readiness_ok:
        direction = "unknown"
    elif any(leg_stats[leg]["ci95_low_above_ceiling"] for leg in GATED_PHASES):
        # Kept AHEAD of the non-degeneracy branch below on purpose: a confounded leg also
        # sets non_degenerate False, and `weakens` -- not `unknown` -- is its right reading.
        direction = "weakens"
    elif not non_degenerate:
        # NOTHING WAS MEASURED, so no direction is warranted -- not even a weak one.
        # Reaches here for a saturated-CONSTANT null series (ratio exactly 0.0, which
        # would otherwise give CI [0,0], `ci95_high_below_ceiling` True and therefore
        # `supports` on a non-measurement -- the same overstatement as red-team F3, one
        # variant further down), an unavailable null control, or an uncomputable CI
        # (subgroup_n < 2, e.g. under --dry-run). `supports` now REQUIRES
        # non_degenerate; the absence of a measurement can never read as evidence for
        # the claim.
        direction = "unknown"
    elif overall_pass and all(
        leg_stats[leg]["ci95_high_below_ceiling"] for leg in GATED_PHASES
    ):
        direction = "supports"
    elif overall_pass:
        # C1 passed and no leg is provably above the ceiling, but at least one interval
        # is not provably BELOW it either (e.g. the CI is uncomputable at subgroup_n<2):
        # the pre-registered criterion is met while its substantive reading is unsettled.
        direction = "mixed"
    else:
        direction = "weakens"
    per_claim = {"SD-071": direction}

    # ---- FLAT SCALAR READOUT (the machine-readable verdict projection) --------------
    readout: Dict[str, Any] = {
        "n_seeds": n,
        "overall_pass": _num(overall_pass),
        "C1_ceiling_outside_ci95_both_legs": _num(c1_all),
        "C2_ratio_interpretable": _num(readiness_ok),
        "C3_sws_ladder_spread_tracks_content": _num(c3_all),
        "non_degenerate": _num(non_degenerate),
        "worst_sws_ladder_spread": _num(worst_spread),
        "ladder_spread_floor": _num(LADDER_SPREAD_FLOOR),
        "worst_sws_ladder_signal_ratio_recorded_not_gated": _num(worst_signal),
        "min_injected_slope_nrem": _num(min_inj_nrem),
        "min_injected_slope_sws": _num(min_inj_sws),
        "injected_slope_floor": _num(INJECTED_SLOPE_FLOOR),
        "null_slope_ratio_ceiling": _num(NULL_SLOPE_RATIO_CEILING),
        "ladder_signal_ratio_recorded_not_gated": _num(LADDER_SIGNAL_RATIO),
    }
    for leg in GATED_PHASES:
        st = leg_stats[leg]
        readout[f"mean_null_slope_ratio_{leg}"] = _num(st["mean"])
        readout[f"sd_null_slope_ratio_{leg}"] = _num(st["sd"])
        readout[f"ci95_low_{leg}"] = _num(st["ci95_low"])
        readout[f"ci95_high_{leg}"] = _num(st["ci95_high"])
        readout[f"ceiling_inside_ci95_{leg}"] = _num(st["ceiling_inside_ci95"])
        readout[f"ci95_high_below_ceiling_{leg}"] = _num(st["ci95_high_below_ceiling"])
        readout[f"ci95_low_above_ceiling_{leg}"] = _num(st["ci95_low_above_ceiling"])
        readout[f"subgroup_n_{leg}"] = _num(st["subgroup_n"])
        readout[f"n_seeds_content_contingent_{leg}"] = _num(
            st["n_seeds_content_contingent"]
        )
        readout[f"n_seeds_null_series_degenerate_{leg}"] = _num(
            st["n_seeds_null_series_degenerate"]
        )
    readout = {k: v for k, v in readout.items() if v is not None}

    interpretation = {
        "label": label,
        "combination_rule": (
            "overall_pass = C2 (readiness, BOTH legs' denominators supra-floor, "
            "all seeds) AND C1 (ceiling_inside_ci95 FALSE on BOTH gated legs, "
            "cohort-level CI95 over the C2-eligible subgroup) AND C3 (sws ladder "
            "spread > 0.01, all seeds). A plain AND of three criteria; C1 is the only "
            "cohort-level one. The recorded-not-gated checks "
            "(ladder_content_signal_ratio_supra_floor, ci95_high_below_ceiling_<leg>) "
            "are NOT in this rule -- they inform the self-route label, the "
            "evidence_direction, and the non-degeneracy net only."
        ),
        "preconditions": [
            {
                "name": "injected_arm_nrem_sigma_slope_supra_floor",
                "description": (
                    "C2 for the nrem leg. The ratio's DENOMINATOR -- the same statistic "
                    "C1 routes on -- measured on the known-damaged injected arm (the "
                    "positive control). If the sigma sweep never moved the nrem "
                    "readout, the ratio is 0/0 and the control cannot discriminate."
                ),
                "control": "injected arm (content_scale=1.0) across the full sigma grid",
                "measured": float(min_inj_nrem),
                "threshold": float(INJECTED_SLOPE_FLOOR),
                "comparator": ">=",
                "direction": "lower",
                "met": bool(c2_nrem_all),
            },
            {
                "name": "injected_arm_sws_sigma_slope_supra_floor",
                "description": (
                    "C2 for the sws leg. Declared separately from the nrem entry "
                    "because C1 gates BOTH legs, so each leg's denominator needs its "
                    "own recomputable (measured, threshold) pair -- one combined entry "
                    "could not reproduce `met` from a single statistic."
                ),
                "control": "injected arm (content_scale=1.0) across the full sigma grid",
                "measured": float(min_inj_sws),
                "threshold": float(INJECTED_SLOPE_FLOOR),
                "comparator": ">=",
                "direction": "lower",
                "met": bool(c2_sws_all),
            },
            {
                "name": "ladder_content_slope_spread_supra_floor",
                "description": (
                    "C3, the 0.01 floor SD-071 names. Guards the scale-invariance "
                    "artifact: the rebuilt sws readout is cosine-based and therefore "
                    "flat in sigma without content BY CONSTRUCTION, so the null ratio "
                    "alone is partly implied by the readout's form. If the sigma-slope "
                    "does not VARY with content amplitude, the readout is not tracking "
                    "content and a low null ratio means nothing. WORST CELL reported, "
                    "matching the all(...) aggregation of `met`."
                ),
                "control": "sws content_scale ladder on the injected path",
                "measured": float(worst_spread),
                "threshold": float(LADDER_SPREAD_FLOOR),
                "comparator": ">",
                "direction": "lower",
                "met": bool(c3_all),
                "offending_cell": (
                    {"seed_index": ladder_spreads.index(worst_spread)}
                    if ladder_spreads else None
                ),
            },
            {
                "name": "ladder_content_signal_ratio_supra_floor",
                "description": (
                    "RECORDED, NOT GATED. The content-bearing rungs' several-fold "
                    "stronger sigma-response than the content-free rung. 778g gated "
                    "this at 3.0x as a diagnostic; SD-071's falsifier names only the "
                    "0.01 spread floor, so gating it here would add a criterion the "
                    "claim does not pre-register. Reported so a weak signal ratio is "
                    "visible without silently re-scoping the claim. Relative, not "
                    "absolute: the zero-content rung carries a small nonzero slope from "
                    "the sigma=0 store-is-exactly-zero discontinuity."
                ),
                "control": "sws content_scale ladder on the injected path",
                "measured": float(worst_signal),
                "threshold": float(LADDER_SIGNAL_RATIO),
                "comparator": ">",
                "direction": "lower",
                "met": bool(c3_signal_all),
                "gating": False,
            },
        ],
        "criteria_non_degenerate": {
            # C1 is degenerate if a leg's CI is uncomputable, its null series saturated,
            # or its interval sits entirely above the ceiling (the confounded reading).
            "C1": bool(non_degenerate),
            "C2": bool(_finite(min_inj_nrem) and _finite(min_inj_sws)),
            "C3": bool(ladder_spreads),
        },
        "criteria": [
            {
                "name": "C1_ceiling_outside_ci95_both_legs",
                "load_bearing": True,
                "passed": bool(c1_all),
                "measured": {
                    leg: {
                        "ceiling_inside_ci95": bool(
                            leg_stats[leg]["ceiling_inside_ci95"]
                        ),
                        "ci95_low": leg_stats[leg]["ci95_low"],
                        "ci95_high": leg_stats[leg]["ci95_high"],
                        "mean": leg_stats[leg]["mean"],
                    }
                    for leg in GATED_PHASES
                },
                "threshold": float(NULL_SLOPE_RATIO_CEILING),
                "requirement": (
                    "ceiling_inside_ci95 == false on BOTH gated legs "
                    "(SD-071 what_would_answer, verbatim)"
                ),
            },
            {
                "name": "C2_ratio_interpretable",
                "load_bearing": False,
                "passed": bool(readiness_ok),
                "measured": float(min(min_inj_nrem, min_inj_sws)),
                "threshold": float(INJECTED_SLOPE_FLOOR),
            },
            {
                "name": "C3_sws_ladder_spread_tracks_content",
                "load_bearing": True,
                "passed": bool(c3_all),
                "measured": float(worst_spread),
                "threshold": float(LADDER_SPREAD_FLOOR),
            },
        ],
        # Proof, recorded in the shipped artifact, that each declared readiness
        # precondition is reachable by its own known-healthy reference under the SHIPPED
        # predicate.
        "anchor_reachability_guards": anchor_guards,
        "harness_carveout_compliance": carveout,
        "gated_phases": list(GATED_PHASES),
        "context_phases_not_gated": list(CONTEXT_PHASES),
        "ladder_phase": LADDER_PHASE,
        "rem_leg_owner": "V3-EXQ-778d/e/f (GOV-FANOUT-1 portfolio)",
        # --- RED-TEAM FINDING 1, CONFIRMED AND RECORDED (not designed around) --------
        # This is the single most important caveat on reading a C3 pass, and it is
        # recorded in the artifact rather than left in a session log.
        "c3_ladder_is_a_sigma_reparameterisation": {
            "finding": (
                "The content-scale ladder is a SIGMA-REPARAMETERISATION of the injected "
                "arm's own damage curve, not an independent content-tracking probe. "
                "Damage is referenced to the UNSCALED content (diffuse_perturb "
                "rms_ref=...), so it is held numerically fixed while content is scaled "
                "by cs; under a SCALE-INVARIANT cosine readout, scaling content by cs at "
                "damage d is arithmetically the same as scaling damage by 1/cs at full "
                "content. Hence error(content_scale=cs, sigma=s) == "
                "error(content_scale=1.0, sigma=s/cs)."
            ),
            "verified": (
                "Confirmed BIT-EXACTLY on 8/8 seeds against the recorded V3-EXQ-778g "
                "ladder_error_series: with the doubling sigma grid [0.0, 0.25, 0.5, "
                "1.0, 2.0], rung0.5[i] == rung1.0[i+1] and rung0.25[i] == rung1.0[i+2] "
                "for every i>=1 (the sigma=0 point is 0 in every rung by construction)."
            ),
            "what_c3_therefore_ACTUALLY_tests": (
                "That the injected damage curve is GRADED over the sigma grid rather "
                "than step-shaped. That is a real and failable property -- a saturating "
                "or step-shaped curve gives spread ~0 and C3 fails -- so C3 is NOT "
                "vacuous and the run still answers its own question."
            ),
            "what_c3_CANNOT_do": (
                "Independently discharge the scale-invariance caveat. SD-071's "
                "what_would_answer and the V3-EXQ-778g docstring both present C3 as the "
                "check that closes that caveat; it cannot, because it is GENERATED by "
                "the same invariance. A C3 pass should not be read as evidence that the "
                "sws readout tracks content AMOUNT independently of its own damage "
                "response."
            ),
            "related_sws_null_arm_shape": (
                "Same root cause, recorded for the same reason: the sws NULL-arm error "
                "series is {0, c, c, c, c} (n_distinct 2 on 8/8 seeds in 778g) -- flat "
                "above sigma=0 because the cosine annihilates the sigma factor when the "
                "store is pure noise. The sws null SLOPE is therefore driven entirely by "
                "the sigma=0 discontinuity, so a low sws null_slope_ratio is partly "
                "implied by the readout's form. The harness's own "
                "NULL_MIN_NULL_SERIES_DISTINCT = 2 is what lets this series count as "
                "non-degenerate."
            ),
            "disposition": (
                "RECORDED, NOT DESIGNED AROUND. C1 and C3 are transcribed verbatim from "
                "SD-071's own pre-registered falsifier; re-scoping either to close this "
                "would change what the run measures, which is a governance decision and "
                "not this run's to make. Raised for governance as a claim-level finding "
                "on SD-071 (governance_flag.py, flag_type evidence_discrepancy) so a "
                "PASS here is weighted knowing what C3 does and does not establish."
            ),
            "red_team": "Step 4.5 adversarial design review, model fable, verdict CONTESTED",
        },
        "outcome_vs_direction_note": (
            "`outcome` is the pre-registered rule (C2 AND C1 AND C3) and nothing else. "
            "Because SD-071's C1 is stated as `ceiling_inside_ci95 == false`, a leg whose "
            "CI95 sits entirely ABOVE the ceiling PASSES C1 while being fully CONFOUNDED. "
            "That case is recorded as outcome=PASS with evidence_direction=weakens, label "
            "consolidation_readout_leg_confounded, and non_degenerate=false (so it is "
            "excluded from confidence scoring). outcome=PASS with direction=weakens is "
            "therefore DIAGNOSTIC of exactly that case and must not be read as support."
        ),
        "c2_requeue_branch_reachability_note": (
            "Recorded honestly: C2's 1e-6 floor sits 4-5 orders of magnitude below where "
            "C1 actually fails (recorded injected slopes ~0.097 nrem, ~0.32 sws), and the "
            "harness independently reports the phase UNAVAILABLE at "
            "NULL_MIN_INJECTED_SLOPE = 1e-9. So the substrate_not_ready_requeue branch is "
            "effectively unreachable, and a reduced-sensitivity outcome routes to "
            "`weakens` rather than `unknown`. C2 is kept because it is the correct shape "
            "for a denominator guard and because `met` is recomputable; it should be read "
            "as 'the ratio was computable', never as 'the substrate was ready'."
        ),
        "nrem_leg_mech121_bearing_note": (
            "Recorded because the harness's carve-out rationale is stronger than the "
            "source supports for this leg. The nrem DV is the output of "
            "ree_core/sleep/cross_module_consolidation.py CrossModuleConsolidator, whose "
            "own module header names it the MECH-121 consolidation cluster and which steps "
            "with torch.optim.Adam (:162) and skips exactly-zero-loss modules (:173-177). "
            "Adam's per-parameter step normalisation is non-linear, and a LINEAR "
            "consolidator would make the nrem injected and null error series differ by a "
            "sigma-independent constant -- i.e. ratio 1.0 by arithmetic, the same affine "
            "cancellation that retired the old sws SNR readout. So the nrem leg's "
            "content-contingency is not wholly independent of that operator's update "
            "rule. THIS DOES NOT BREACH THE CARVE-OUT: its conditions are about TAGGING, "
            "and this run tags neither MECH-120 nor MECH-121 and cannot move MECH-121's "
            "status under any outcome. What it does mean is that a reader should not take "
            "an SD-071 pass as fully operator-agnostic for the nrem leg."
        ),
        "ci95_side_of_ceiling_note": (
            "ceiling_inside_ci95 == false is TWO-WAY AMBIGUOUS: false below the ceiling "
            "is the content-contingent reading SD-071 asserts, false above it is the "
            "CONFOUNDED reading that refutes it. C1 as pre-registered cannot separate "
            "them, so the side is recorded per leg (ci95_high_below_ceiling / "
            "ci95_low_above_ceiling) and used by the self-route label, the "
            "evidence_direction and the non-degeneracy net rather than by a criterion "
            "the claim did not name."
        ),
    }

    context_summary = {
        p: {
            "per_seed_null_slope_ratio": [
                s["context_null_slope_ratio"][p] for s in seed_scores
            ],
            "gated": False,
            "note": (
                "known degenerate at both clamp rails (778c: exactly 0.0 on 5/8 seeds "
                "off a saturated constant, off-scale 1801-9143 on 3/8); owned by the "
                "GOV-FANOUT-1 portfolio 778d/e/f. SD-071 does not claim this leg."
            ),
        }
        for p in CONTEXT_PHASES
    }

    print("", flush=True)
    print(
        f"overall {'PASS' if overall_pass else 'FAIL'}  (n={n} seeds; "
        f"C1={c1_all} C2={readiness_ok} C3={c3_all}; non_degenerate={non_degenerate})",
        flush=True,
    )
    for leg in GATED_PHASES:
        st = leg_stats[leg]
        print(
            f"  {leg:>4} (GATED): mean null_slope_ratio={_fmt(st['mean'])} "
            f"sd={_fmt(st['sd'])} ci95=[{_fmt(st['ci95_low'])}, "
            f"{_fmt(st['ci95_high'])}] subgroup_n={st['subgroup_n']} "
            f"content_contingent_seeds={st['n_seeds_content_contingent']}/{n}"
            + ("  [CEILING INSIDE CI -- SD-071 REFUTED at this n]"
               if st["ceiling_inside_ci95"] else "")
            + ("  [CI ENTIRELY ABOVE CEILING -- leg CONFOUNDED]"
               if st["ci95_low_above_ceiling"] else ""),
            flush=True,
        )
    print(
        f"  sws C3 ladder: worst spread={_fmt(worst_spread)} "
        f"(floor {LADDER_SPREAD_FLOOR}) | recorded-not-gated worst signal ratio="
        f"{_fmt(worst_signal)} (778g gated this at {LADDER_SIGNAL_RATIO})",
        flush=True,
    )
    print(
        "  recorded diagnostic references (NOT thresholds): nrem 0.1445 "
        "CI95 [0.1438, 0.1451] (778c); sws 0.1495 sd 0.0218 CI95 [0.1344, 0.1646], "
        "C3 spread 0.1108 (778g)",
        flush=True,
    )
    for p in CONTEXT_PHASES:
        print(f"  {p:>4} (context, NOT gated): recorded only", flush=True)
    if degeneracy_reasons:
        for r in degeneracy_reasons:
            print(f"  [degenerate] {r}", flush=True)
    print(f"self-route label: {label}", flush=True)
    print(f"evidence_direction: {direction}", flush=True)

    return {
        "outcome": "PASS" if overall_pass else "FAIL",
        "evidence_direction": direction,
        "evidence_direction_per_claim": per_claim,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": "; ".join(degeneracy_reasons) if degeneracy_reasons else "",
        "interpretation": interpretation,
        "arm_results": arm_results,
        "readout": readout,
        "leg_summary": leg_stats,
        "context_summary": context_summary,
        "c1_per_leg": c1_per_leg,
        "n_seeds_pass": sum(
            1 for s in seed_scores
            if s["C2_ratio_interpretable"] and s["C3_ladder_spread_tracks_content"]
        ),
        "need_seeds": n,
        "config": config_slice,
        "seeds": seeds,
        "worst_spread_seed": worst_spread_seed,
    }


def main(*, dry_run: bool = False) -> Tuple[str, Path]:
    import time

    t0 = time.perf_counter()
    result = run_experiment(dry_run=dry_run)
    outcome = result["outcome"]

    run_id = f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3"
    out_dir = (
        REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": result["evidence_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "interpretation": result["interpretation"],
        "sleep_driver_pattern": SLEEP_DRIVER_PATTERN,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "dry_run": dry_run,
        "n_seeds_pass": result["n_seeds_pass"],
        "need_seeds": result["need_seeds"],
        "readout": result["readout"],
        "leg_summary": result["leg_summary"],
        "context_summary": result["context_summary"],
        "c1_per_leg": result["c1_per_leg"],
        "acceptance_criteria": {
            "C1_ceiling_outside_ci95_both_legs": (
                "ceiling_inside_ci95 is FALSE on BOTH gated legs (nrem AND sws), where "
                f"the ceiling is {NULL_SLOPE_RATIO_CEILING} and the CI95 is over the "
                "per-seed null_slope_ratio distribution at n>=8, computed by the "
                "harness's shared subgroup_ratio_stats over the C2-eligible subgroup. "
                "LOAD-BEARING. Transcribed from SD-071's what_would_answer: 'REFUTED "
                "if a re-run at n>=8 puts ceiling_inside_ci95 true on either leg'."
            ),
            "C2_ratio_interpretable": (
                f"|injected sigma-slope| >= {INJECTED_SLOPE_FLOOR} on BOTH legs, all "
                "seeds (readiness, NOT a scientific criterion; asserts the ratio's "
                "denominator -- the same statistic C1 routes on). Below floor -> "
                "substrate_not_ready_requeue, never a substrate verdict."
            ),
            "C3_sws_ladder_spread_tracks_content": (
                "sws content-scale ladder: the SIGNED spread max(pos)-min(pos) across "
                f"the content_scale>0 rung slopes exceeds {LADDER_SPREAD_FLOOR}, all "
                "seeds. LOAD-BEARING anti-artifact. Transcribed from SD-071's "
                "what_would_answer: 'REFUTED ... if the sws leg's content-scale ladder "
                "collapses below the 0.01 floor'. sws-only by the claim's own wording."
            ),
            "RECORDED_NOT_GATED": (
                "ladder_content_signal_ratio_supra_floor (778g's 3.0x diagnostic bar) "
                "and ci95_high_below_ceiling_<leg> (which side of the ceiling the "
                "interval fell on) are measured and recorded but NOT part of the pass "
                "rule -- SD-071 does not pre-register either, and adding a gate the "
                "claim did not name would change what this run can conclude. They "
                "inform the self-route label, the evidence_direction and the "
                "non-degeneracy net."
            ),
        },
        "arm_results": result["arm_results"],
        "notes": (
            "SD-071 EVIDENCE run: the SD-068 nrem (nrem_transfer_fidelity, harness:523 "
            "called :1039) and sws (_sws_pattern_completion, harness:400 called :379) "
            "consolidation readouts are CONTENT-CONTINGENT INSTRUMENTS. This supplies "
            "the one thing SD-071 still needed and could not previously obtain: a "
            "NON-DIAGNOSTIC run tagging the claim. "
            "THE CARVE-OUT THIS EXERCISES: until 2026-09-18 the harness carried an "
            "unconditional 'Any run built on it MUST be EXPERIMENT_PURPOSE=diagnostic' "
            "contract under 'Prerequisite caveat -- MECH-121 hold', so SD-071's own "
            "promotion condition was unsatisfiable. Raised as GFLAG-0319 "
            "(contested_disposition, 2026-09-17), refused at the /queue-experiment "
            "STOP-GATE, and DECIDED BY THE USER 2026-09-18T19:47:16Z via the "
            "Orchestrator decision lane (orchestrate-20260918-1840-cloud4, real "
            "AskUserQuestion): OPTION A, grant the narrow carve-out. Amendment landed "
            "and verified on origin at both caveat sites -- REE_assembly d202c2d73c and "
            "ree-v3 6371c9fc8a; GFLAG-0319 resolved (876429a2d4); EXP-1178/EVB-1657 "
            "lifted back to proposed (6d0427b432). The carve-out is an ENUMERATION, not "
            "a category: non-diagnostic is permitted only when claim_ids are solely "
            "instrument-validity claims (enumerated {SD-071}) AND neither MECH-120 nor "
            "MECH-121 is tagged. Both conditions are ASSERTED IN CODE before any "
            "compute (_assert_carveout_conditions) -- claim_ids is exactly ['SD-071'], "
            "deliberately NOT 778c/778g's ['SD-068','MECH-168','INV-047','MECH-169'], "
            "because repeating that 'context' tagging would void condition (i). SD-068 "
            "/ MECH-168 / INV-047 / MECH-169 are named here as PROVENANCE, which is not "
            "a tag; this run does not weight them. MECH-121's status is untouched under "
            "every outcome. MECH-170 does NOT meet the carve-out condition and was NOT "
            "added to the enumeration (it is a behavioural recovery-order prediction "
            "that depends_on MECH-120 and MECH-121 directly); carried forward as "
            "decision chip chip-20260918-mech170-carveout-scope. "
            "C1 gates BOTH legs (778g scoped its C1 to sws alone because the rem leg "
            "was known degenerate and would have failed a three-phase C1 regardless); "
            "the rem leg is measured and reported as CONTEXT only and stays owned by "
            "the GOV-FANOUT-1 portfolio 778d/e/f. C3's ladder is sws-only, by the "
            "claim's own wording. "
            "GOV-REUSE-1: the decisive readout is ceiling_inside_ci95 on both legs' "
            "per-seed null_slope_ratio from a non-diagnostic run tagging SD-071. "
            "reanalysis_query.py --readout null_slope_ratio --claim SD-071 returned 0 "
            "matches; a hand grep of the manifest corpus finds the readout on "
            "778b/778c/778g, but ZERO manifests tag SD-071 at all (all three tag the "
            "SD-068 staging set) and all three are experiment_purpose=diagnostic, which "
            "does not weight confidence. No reanalysis of a diagnostic manifest can "
            "make it non-diagnostic, so the missing artifact is not derivable post-hoc "
            "-> ran. The substrate_hash has also moved since both runs (harness "
            "76508144, b42f69ff, 6371c9fc8a), so their cells would be INCOMPATIBLE for "
            "reuse regardless. "
            "Re-derive brake: 0 of 523 autopsy artifacts tag SD-071; not tripped. "
            "Experiment-layer only; zero ree_core change; no substrate_queue entry."
        ),
    }

    stamp_recording_core(
        manifest,
        config=result["config"],
        seeds=result["seeds"],
        script_path=Path(__file__),
        started_at=t0,
    )

    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=dry_run,
        config=result["config"],
        seeds=result["seeds"],
        script_path=Path(__file__),
        started_at=t0,
        json_default=str,
    )
    if dry_run:
        print("[dry-run] manifest relocated out of evidence/ by emit_outcome", flush=True)
    else:
        print(f"Result written to: {out_path}", flush=True)
    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _outcome, _out_path = main(dry_run=args.dry_run)

    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        dry_run=args.dry_run,
    )
    sys.exit(0)
