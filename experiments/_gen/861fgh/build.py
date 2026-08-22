"""Generate the V3-EXQ-861f/861g/861h drivers from the V3-EXQ-861e predecessor.

Every patch is an EXACT-STRING replacement against 861e and asserts its anchor,
so a silent drift in the predecessor fails loudly here instead of producing a
half-patched driver.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from legs import LEGS, QID, AUTOPSY, RUN_861E, RUN_861C  # noqa: E402

V3 = Path("/Users/dgolden/REE_Working/ree-v3")
SRC = V3 / "experiments/v3_exq_861e_inv050_mech180_calibration_power_raised_replication.py"
base = SRC.read_text()


def rep(s, old, new, count=1, tag=""):
    assert old in s, f"ANCHOR MISSING [{tag}]: {old[:110]!r}"
    assert s.count(old) == count, f"ANCHOR COUNT {s.count(old)} != {count} [{tag}]"
    return s.replace(old, new)


# --------------------------------------------------------------------------
# Shared docstring blocks
# --------------------------------------------------------------------------
COMMON_TAIL = f'''
================================================================================
WHAT IS UNCHANGED FROM V3-EXQ-861e (deliberately -- this is an ISOLATION leg)
================================================================================
Env, ARMS, SEEDS [7, 271, 883], agent config, C1/C2 formulae, R1/R2/R3
readiness, MEAS_CYCLES, thresholds, the interpretation grid and the scored-DV
set are all byte-identical to V3-EXQ-861e. Each leg of this portfolio varies
EXACTLY ONE thing, so a difference in the readout is attributable.

MECH-122 content-selection stays OFF (USE_MECH122_SPINDLE_CONTENT_SELECTION =
False), as in 861b/861c/861e.

SLEEP DRIVER: manual-cycle-loop (force_cycle() called once per cycle in a
dedicated MEAS_CYCLES wake-sleep loop) -- unchanged from 861/861a/845/861b/
861c/861e.

================================================================================
SUBSTRATE PIN (experiments/_lib/substrate_pin.py)
================================================================================
ree_core/ is executed from a PINNED historical commit, extracted read-only with
`git archive` into a scratch dir placed first on sys.path. experiments/_lib/**,
experiment_protocol and pack_writer still come from the LIVE checkout. The pin
is proven by TWO fatal checks before any science runs -- a structural path check
on ree_core.__file__, and a behavioural source-marker check
(ree_core.predictors.e3_selector.authority_spread_ratio, which exists at
17befb8c and does NOT exist at f810969). A pin that cannot be proven raises
SubstratePinError, exits non-zero, and the runner classifies it ERROR. Running
a leg that cannot say which substrate it executed is the exact verdict-aliasing
failure this portfolio exists to avoid, so there is no degraded mode.

Per-cell arm_fingerprint uses repo_root=<pin dir> and substrate_scope=
("ree_core/**/*.py",), so the recorded substrate_hash describes the code that
ACTUALLY RAN. That scope is deliberately narrower than the default globs, so
every pinned cell is stamped reuse-INELIGIBLE
(substrate_pinned_to_historical_commit) and must never be minted as a baseline.

Note the pin also makes this leg independent of trunk drift while it sits in
the queue -- a substrate change landing between queue-time and run-time cannot
silently move the comparison.

================================================================================
EXPERIMENT_PURPOSE = "diagnostic" -- and why the directions are "unknown"
================================================================================
This is an instrument-isolation probe, not evidence for or against INV-050 /
MECH-180. It is tagged with both claim_ids because that is the lineage it
adjudicates, but evidence_direction is pinned non_contributory and
evidence_direction_per_claim is {{"INV-050": "unknown", "MECH-180": "unknown"}}
REGARDLESS of how the replicated C1/C2 grid comes out. The grid IS still
computed and recorded in full (under interpretation + replication_readout), so
it is directly comparable to 861c/861e -- it just does not vote. Diagnostics are
excluded from governance confidence/conflict scoring by construction; this
pinning makes that explicit rather than relying on it.

PROMOTES/DEMOTES NOTHING BY ITSELF: /governance applies the verdict.

================================================================================
STANDING SUBSTRATE-DEFECT DISCLOSURE (queue-experiment Step 2.5c)
================================================================================
`contextmemory-write-path-addressing-degeneracy` (substrate_queue, severity
CORRUPTING, status implemented_pending_validation, substrate_paths
[ree_core/predictors/e1_deep.py]) OVERLAPS this driver -- and it is not
hypothetical here. Measured directly from the recorded manifests of both
predecessors, seed 271 is write-address LOCKED in BOTH runs, on EVERY arm:

  861c (f810969, n=5)   seed 271 per_cycle_n_touched_slots ~ [1,1,1,1,2,1]
  861e (17befb8c, n=10) seed 271 per_cycle_n_touched_slots ~ [1,1,1,1,1,1]
                        (ARM_3_HIGH_ON: 6/6 cycles insufficient, new_div 0.0000)
  seeds 7 and 883 rotate normally in both runs (4-14 slots touched).

Seed 271 is the seed the whole H1-vs-H3 discrimination rests on. Two
consequences, both pre-registered here:

 (a) The lock is a CONSTANT across 861c and 861e, so it cannot be the cause of
     the DIFFERENCE between them. H1 vs H3 remains well-posed.
 (b) The lock still means seed 271's readouts are produced by an agent with a
     1-slot context bank, which is a documented corrupting condition. That is
     why V3-EXQ-861h exists: it repeats this protocol with the already-built
     non-degenerate write selection ON, so the portfolio measures the defect
     instead of inheriting it. 861f/861g deliberately keep the LEGACY argmin
     write path, because changing it would destroy the single-variable
     isolation that is their entire point.

Every cell records per_cycle_n_touched_slots and
n_cycles_insufficient_touched_slots, and the manifest carries a
contextmemory_write_lock_audit block, so any later autopsy can read the lock
state per seed without re-deriving it.

Also disclosed, unchanged from 861e's own Step 2.5c adjudication:
 - MECH122-CONTENT-PACKAGING-SPINDLE-SELECTION (corrupting): inert here --
   agent.run_sws_schema_pass gates the whole relative_novelty() consumption
   behind use_mech122_spindle_content_selection, False in this run.
 - mode-governance-engagement (corrupting): unrelated subsystem; this driver
   never imports salience_coordinator.py or regime_occupancy_gate.
 - SD-MECH267-CEM-SELECTION-FIX / SD-MECH303-THRESHOLD-SOURCING /
   SD-SLEEP-ENTRY-PRESSURE / SD-E3-SCORER-COMPLETION / SD-ORIENTING-DECISION-
   SCALE (degrading): arm-symmetric, noted not blocking, as in 861e.

================================================================================
RE-DERIVE BRAKE (queue-experiment Step 2.5b) -- examined, RELEASED
================================================================================
The literal run-keyed counter reads INV-050 = 7 and MECH-180 = 5, i.e. over the
threshold of 2. It is released on three independent grounds:

 1. The skill's own "Not braked" clause: this is a `diagnostic` whose purpose is
    to discriminate WHY the reading came out as it did -- not a lettered
    re-derive of the same claim at the same granularity.
 2. GOV-FANOUT-1 makes a diverse discrimination PORTFOLIO the prescribed route
    for a braked lineage carrying a fanout_recommendation, which is exactly what
    {AUTOPSY} emitted.
 3. That autopsy's own producer half stamped re_derive_brake.fired = false,
    refused_requeue = false, route_to = "queue-experiment", and its Step 8
    interactive gate recorded verdict "adopt_redteam_portfolio".

The counter's hits are almost all `standard` / `non_contributory` targets that
count only through the non_contributory-direction proxy the skill itself warns
inverts the brake's purpose for instrument defects. 861e's category note says it
outright: "Stamping substrate_ceiling here would fire the re-derive brake and
forbid the H1/H3 isolation the data now need."

================================================================================
GOV-REUSE-1 (Step 2.4) -- not recoverable, must run
================================================================================
Decisive readout: ARM_3_HIGH_ON mean_mel and mean_duration_factor for seed 271,
each against that cell's own mel_reference, under this leg's single varied
condition. No recorded manifest carries it: 861c has (f810969, n=5, no reseed,
legacy write), 861e has (17befb8c, n=10, no reseed, legacy write), and every
cell in this portfolio is a condition neither ran. Not recoverable -> run.
'''


def preamble(leg):
    return f'''"""
{leg["title"]}

GOV-FANOUT-1 discrimination leg. Portfolio question (hypothesis_space_registry
qid): {QID}
Confirmed source autopsy: {AUTOPSY}
(status confirmed, Step 7c CONTESTED -> Step 8 interactive gate
"adopt_redteam_portfolio"; frozen hypothesis set H1 + H3, H2 a labelled
follow-on, count 2).

Hypothesis under test: {leg["qid_hyp"]}   Design axis: {leg["axis"]}
Substrate pinned to: {leg["pin_ref_short"]}
Predecessors compared against: {RUN_861E} (861e) and {RUN_861C} (861c).
'''


H1_BODY = f'''
================================================================================
WHAT 861e LEFT OPEN
================================================================================
V3-EXQ-861e raised CALIB_DRAWS 5 -> 10 to repair 861c's under-powered C2. The
instrument repair WORKED (R3 passed 3/3, SEM adequate) -- and the answer moved
the wrong way: seed 271's ARM_3_HIGH_ON mean MEL fell from 2.8999705e-05
(861c, ABOVE its own reference 2.3900e-05, duration factor 1.215) to
2.2980323e-05 (861e, BELOW its own reference 2.5982e-05, factor 0.884). The
confirmed cluster autopsy's projection that "n=10 flips seed 271 to PASS" is
therefore REFUTED at its premise: the numerator moved, not just the denominator.

{AUTOPSY} withdrew "same seeds isolate the calibration change" as a
ceteris-paribus premise, because the extra calibration draws are extra frozen
wake steps that consume torch policy RNG via select_action BEFORE a measurement
loop that never reseeds. Calibration runs train=False (E2 frozen), so the extra
draws are not extra world-model training -- but they do displace the RNG stream
the measurement phase then draws from.

================================================================================
H1 -- WHAT THIS LEG DOES
================================================================================
IT VARIES EXACTLY ONE THING vs 861e: torch and numpy are reseeded, with a fixed
derived seed, immediately before the measurement phase begins (before the
measurement env is constructed, so env construction is inside the isolated
stream). The measurement-phase RNG stream is then a function of (seed) alone and
is INDEPENDENT of how many calibration draws preceded it. That is what "the
intervention is isolated" has to mean here, and 861c/861e/861 never had it.

MEAS_RESEED_OFFSET is a distinct constant (not 0), so the measurement stream is
a fresh stream rather than a replay of the convergence stream.

Note the reseed also makes the measurement stream identical ACROSS ARMS within a
seed. The arms still differ in world_rule_shift_interval, so the environments
differ; what is removed is arm-to-arm RNG drift. For the C1 dose-response
comparison that is a strictly PAIRED design and is an improvement, but it is a
declared change from 861c/861e and is recorded as such.

================================================================================
IN-RUN CONTROL -- why this leg does not have to trust a cross-run comparison
================================================================================
After the 5x3 reseeded grid, this leg re-runs ARM_3_HIGH_ON for all three seeds
with the reseed DISABLED, at the same CALIB_DRAWS=10, on the same pinned
substrate, in the same process, on the same machine. Cells are independent
(every cell resets all RNG at entry), so order does not matter.

That unreseeded control is a same-machine, same-substrate replica of 861e's own
decisive cell. It gives this leg two things a cross-run comparison cannot:

 (i) H1 is decided WITHIN the run (reseeded vs unreseeded, everything else
     held), so a machine difference between this box and ree-worker-1 cannot
     masquerade as the effect; and
 (ii) comparing the unreseeded control against 861e's RECORDED value for the
      same cell MEASURES that machine delta directly, which is the part of H3
      the autopsy could not otherwise separate. (Both runs report
      machine_class linux-x86_64-py3.10-torch2.12.0+cpu, so any difference is
      sub-machine-class.)

================================================================================
DECLARED NULL (pre-registered)
================================================================================
DECISIVE COMPARISON: seed 271, ARM_3_HIGH_ON, reseeded vs unreseeded, each cell
scored against its OWN mel_reference (mean_duration_factor is exactly that
ratio, clamped).

 - H1 SUPPORTED if the reseeded cell's factor rises above 1.0 while the
   unreseeded control stays below (reproducing 861e). The 861e collapse was an
   intervention-isolation defect, not a producer failure, and the whole
   861c-vs-861e comparison must be re-read.
 - H1 NOT SUPPORTED if both cells sit below 1.0 (the collapse survives RNG
   isolation). This is an INFORMATIVE null: it removes measurement-phase RNG
   displacement from the live hypothesis set and hands the question to H3 (and
   then H2). It does NOT mean seed 271 is "like seed 7" -- 861c HIGH-graded
   seed 271 at n=5, whereas seed 7 was already below its own reference then.
 - UNINFORMATIVE (non_contributory) if the unreseeded control does NOT
   reproduce 861e's collapse. Then this box does not reproduce 861e at all,
   the leg has no baseline to isolate against, and the reading is a machine
   delta rather than an answer about H1.

WHAT A NULL HERE DOES NOT MEAN: it is not evidence about INV-050 or MECH-180.
Neither claim's status, confidence, or v3_pending may move on this leg.
'''

H3_BODY = f'''
================================================================================
WHAT 861e LEFT OPEN
================================================================================
861c and 861e are not only a CALIB_DRAWS contrast. They executed DIFFERENT
substrates on DIFFERENT boxes:

  861c: substrate_commit f810969, substrate_hash 5eaa59f5..., ree-cloud-4, 3.5h
  861e: substrate_commit 17befb8c, substrate_hash d1f4bdae..., ree-worker-1, 10.3h

and seed 271's ARM_3_HIGH_ON mean MEL went from 2.8999705e-05 (above its own
reference; factor 1.215) to 2.2980323e-05 (below it; factor 0.884). The
confirmed autopsy calls that a live H3, distinct from the within-run
substrate_stable_across_run flag (which is 798a's reuse-safety pattern, not a
mid-run code change).

The f810969 -> 17befb8c delta is real and NOT merely a default-off knob
addition: 1296 inserted lines across 8 ree_core files, including UNCONDITIONAL
changes in agent.py (extra clock.phase_reset() sites, the cem_elite modulatory
route backstop, orienting-decision tick decrements) and the whole
ContextMemory write-selection machinery in e1_deep.py. So "the substrate
changed under the comparison" is a hypothesis with a concrete mechanism behind
it, not a formality.

================================================================================
H3 -- WHAT THIS LEG DOES
================================================================================
IT VARIES EXACTLY ONE THING vs 861e: ree_core is pinned to f810969, 861c's own
substrate. CALIB_DRAWS stays at 10, R3 stays on, no reseed, legacy write path --
i.e. the 861e protocol, run on the old code. That fills the missing cell of the
2x2 the lineage has been reasoning across:

                     CALIB_DRAWS=5        CALIB_DRAWS=10
    f810969          861c (271 HIGH)      THIS LEG (primary grid)
    17befb8c         -- (that is H2) --   861e (271 collapsed)

Verified at authoring time on the pinned tree: all 43 REEConfig.from_dims
kwargs this driver passes exist at f810969, so none is silently swallowed
(cf. reference-reeconfig-from-dims-silent-kwargs, where a knob absent from
from_dims is dropped without error). The ContextMemory write-selection knobs do
NOT exist at f810969, which is itself informative: 861c could not have had the
repair, so both predecessors necessarily ran the legacy write path.

================================================================================
IN-RUN CONTROL -- a POSITIVE control that validates the pin behaviourally
================================================================================
After the 5x3 grid at CALIB_DRAWS=10, this leg re-runs ARM_3_HIGH_ON for all
three seeds at CALIB_DRAWS=5 -- i.e. 861c's exact decisive condition -- on the
same pinned substrate, in the same process, on the same machine.

This control does double duty:

 (i) It is a behavioural verification of the pin that no static check can give.
     861c recorded seed 271 ARM_3_HIGH_ON mean_mel 2.8999705e-05 against
     reference 2.3900e-05 (factor 1.215). If the pin is faithful, this control
     should land near that; if it lands near 861e's 2.2980e-05 instead, the pin
     did not take (or the machine delta dominates) and the primary grid must
     not be read as an H3 answer.
 (ii) It isolates the CALIB_DRAWS change ON THE OLD SUBSTRATE, same box --
      which is the contrast 861c-vs-861e was supposed to be but was not.

================================================================================
DECLARED NULL (pre-registered)
================================================================================
DECISIVE COMPARISON: seed 271, ARM_3_HIGH_ON, primary (f810969, n=10) against
861e's recorded cell (17befb8c, n=10), and against this run's own n=5 control.

 - H3 SUPPORTED if seed 271 stays HIGH-graded here (factor above 1.0) at n=10 on
   f810969, while 861e collapsed at n=10 on 17befb8c. The collapse is then a
   substrate (or machine) delta, and the 861c/861e comparison is confounded by
   code drift rather than by calibration power.
 - H3 NOT SUPPORTED if seed 271 collapses here too. The old substrate does not
   rescue it, so the collapse tracks CALIB_DRAWS or measurement RNG (H1) or the
   seed itself (H2) -- and the in-run n=5 control discriminates further: if n=5
   reproduces 861c's 1.215 while n=10 collapses on the SAME substrate and box,
   that is a clean, machine-free demonstration that CALIB_DRAWS alone moves the
   readout, which is H1's mechanism confirmed from the other side.
 - UNINFORMATIVE (non_contributory) if the n=5 positive control does NOT
   approximately reproduce 861c. Then the pin or the machine is dominating and
   the primary grid answers nothing about H3.

WHAT A NULL HERE DOES NOT MEAN: it is not evidence about INV-050 or MECH-180,
and it is not a substrate ceiling. Neither claim's status, confidence, or
v3_pending may move on this leg.
'''

H4_BODY = f'''
================================================================================
WHY THIS LEG EXISTS -- it was NOT in the autopsy's portfolio
================================================================================
{AUTOPSY} froze H1 + H3 (count 2), with H2 a labelled follow-on. This leg is
NOT a fourth frozen hypothesis about the 861c -> 861e difference. It is a
CONTROL on whether the decisive seed's measurement is trustworthy at all, added
because two mandatory queue-experiment checks fired at queue time and neither
could be discharged any other way:

 - Step 2.5b(iv), the adversarial pre-queue design audit, requires closing the
   worst coverage / verdict-aliasing gap before queuing.
 - Step 2.5c, the substrate-path overlap gate, fires CORRUPTING on
   `contextmemory-write-path-addressing-degeneracy`
   (substrate_paths [ree_core/predictors/e1_deep.py]) and blocks by default.

The finding that connects them was measured, not assumed. Reading
per_cycle_n_touched_slots straight out of both predecessors' manifests:

  861c seed 271, every arm: 1-3 slots touched per cycle, 4-5 of 6 cycles
       flagged insufficient.
  861e seed 271, every arm: 1-2 slots touched; ARM_3_HIGH_ON is 1,1,1,1,1,1
       with mean_sws_new_slot_diversity exactly 0.0000.
  861c/861e seeds 7 and 883: 3-14 slots touched, 0 cycles insufficient.

That is precisely the signature the substrate_queue entry describes -- "a
deterministic single-slot fixed point under a low-variance query stream", which
at V3-EXQ-436e/436f produced 1-of-16 occupancy while write() returned normally
and thousands of calls were logged. Seed 271 -- the ONE seed the entire H1-vs-H3
discrimination rests on -- has been running with a 1-slot context bank in BOTH
predecessor runs.

Because the lock is a CONSTANT across 861c and 861e, it cannot explain the
DIFFERENCE between them, so H1 vs H3 stays well-posed and 861f/861g must keep
the legacy write path to preserve their isolation. But a constant corrupting
condition on the decisive seed is exactly what makes a small, real difference
between two runs uninterpretable: with a near-constant context bank, E1's
contextual contribution to prediction is degenerate, so measured MEL
(e3.post_action_update -> prediction_error) can be unusually sensitive to any
perturbation. This leg asks whether that is what is going on.

================================================================================
WHAT THIS LEG DOES
================================================================================
IT VARIES EXACTLY ONE THING vs 861e: substrate pinned to 861e's own 17befb8c,
CALIB_DRAWS=10, no reseed, same everything -- with
contextmemory_write_selection switched from the default "argmin" (legacy,
degenerate) to "refractory" (contextmemory_write_refractory_k=2).

"refractory" was chosen over "usage_balancing" deliberately: V3-EXQ-943
validated BOTH (BIAS 16/16 occupied 5/5 seeds; REFRACTORY 5/5 seeds >= 2
occupied, locking seeds at exactly k+1=3), but refractory's non-degeneracy is
STRUCTURAL (a k-step write refractory period cannot re-select the same slot)
rather than counter-driven, so it is the smaller and more predictable
perturbation to a run whose readout is a prediction error.

Both knobs exist at 17befb8c and are absent at f810969 -- which is why this leg
pins 17befb8c and why the equivalent control cannot be run on 861c's substrate.

================================================================================
IN-RUN CONTROL
================================================================================
After the 5x3 refractory grid, this leg re-runs ARM_3_HIGH_ON for all three
seeds with the LEGACY argmin write path, everything else held, in the same
process on the same machine. So the write-path contrast is decided WITHIN the
run and the legacy cells double as a same-box replica of 861e's decisive cell.

================================================================================
DECLARED NULL (pre-registered)
================================================================================
DECISIVE COMPARISONS: (a) seed 271 ARM_3_HIGH_ON factor, refractory vs legacy,
in-run; (b) n_cycles_insufficient_touched_slots for seed 271, refractory vs
legacy -- the repair must actually fire.

 - CONTROL FAILS (the reading that matters most) if the refractory arm restores
   seed 271's HIGH grading (factor above 1.0) while the in-run legacy control
   reproduces the collapse. The decisive seed's readout then depends on a known
   CORRUPTING substrate defect, and the H1/H3 verdicts from 861f/861g must be
   read as conditional on it -- a fourth hypothesis would owe registration
   against qid {QID}, and that is a governance call, not this leg's.
 - CONTROL PASSES if seed 271's factor is materially unchanged under the
   repair. The write-address lock is then not load-bearing for measured MEL,
   861f/861g's readings stand on their own, and the corrupting overlap is
   discharged empirically rather than by argument.
 - UNINFORMATIVE (non_contributory) if the refractory arm does not actually
   reduce n_cycles_insufficient_touched_slots on seed 271 -- the repair did not
   engage, so nothing was controlled. This is checked as a pre-registered
   readiness precondition (write_repair_engaged_on_decisive_seed), not
   discovered post hoc.

WHAT ANY OUTCOME HERE DOES NOT MEAN: it is not evidence about INV-050 or
MECH-180, and it does NOT validate or close
`contextmemory-write-path-addressing-degeneracy` (that entry's own validation
is V3-EXQ-943's occupancy criterion, adjudicated separately). Neither claim's
status, confidence, or v3_pending may move on this leg.
'''

BODIES = {"861f": H1_BODY, "861g": H3_BODY, "861h": H4_BODY}


# --------------------------------------------------------------------------
# Code patches
# --------------------------------------------------------------------------
PIN_IMPORT = '''sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- SUBSTRATE PIN -- MUST run before the first `import ree_core` -----------
# A driver that imports ree_core first silently gets the live checkout and the
# pin becomes a no-op, which is exactly the verdict-aliasing failure this leg
# exists to avoid. pin_ree_core() raises if ree_core is already in sys.modules,
# and verify_pin() raises if the pin cannot be PROVEN structurally AND
# behaviourally. Both are fatal on purpose -- see the docstring.
from experiments._lib.substrate_pin import (            # noqa: E402
    pin_ree_core, verify_pin, pin_fingerprint_kwargs, pin_manifest_block,
)

SUBSTRATE_PIN_REF = "__PIN_REF__"
# Source marker that DIFFERS across the pinned/live boundary: added by
# ree-v3 commit 17befb8c ("modulatory-bias-selection-authority AMEND"), so it
# is present at 17befb8c and absent at f810969. A structural path check alone
# cannot catch a stale cache dir holding the wrong ref's content; this can.
SUBSTRATE_PIN_MARKER_MODULE = "ree_core.predictors.e3_selector"
SUBSTRATE_PIN_MARKER_ATTR = "authority_spread_ratio"
SUBSTRATE_PIN_MARKER_EXPECTED_PRESENT = __MARKER__

_PIN = pin_ree_core(SUBSTRATE_PIN_REF)
verify_pin(
    _PIN,
    marker_module=SUBSTRATE_PIN_MARKER_MODULE,
    marker_attr=SUBSTRATE_PIN_MARKER_ATTR,
    marker_expected_present=SUBSTRATE_PIN_MARKER_EXPECTED_PRESENT,
)
# repo_root=<pin dir> + substrate_scope=ree_core/**  -> the recorded
# substrate_hash describes the code that ACTUALLY RAN, and every pinned cell is
# stamped reuse-INELIGIBLE.
_PIN_FP_KWARGS = pin_fingerprint_kwargs(_PIN)
'''

ANCHOR_PIN = 'sys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n'

VARIANT_CONSTS = '''
# -- GOV-FANOUT-1 leg identity (see docstring) ------------------------------
FANOUT_QID = "__QID__"
FANOUT_HYPOTHESIS = "__HYP__"
FANOUT_AXIS = "__AXIS__"
FANOUT_SOURCE_AUTOPSY = "__AUTOPSY__"
PRIMARY_VARIANT = "__VAR_A__"
CONTROL_VARIANT = "__VAR_B__"
# The decisive cell of the whole 861 fan-out. Both the in-run control set and
# every declared-null comparison are scoped to it.
DECISIVE_ARM_ID = "ARM_3_HIGH_ON"
DECISIVE_SEED = 271
# Recorded predecessor values for the decisive cell, read from the manifests at
# authoring time and pinned here so the comparison is pre-registered rather than
# recomputed post hoc.
PREDECESSOR_DECISIVE = {
    "861c": {"run_id": "__RUN_861C__", "substrate_commit": "f810969",
             "calib_draws": 5, "mean_mel": 2.9000e-05, "mel_reference": 2.3900e-05,
             "mean_duration_factor": 1.215, "machine": "ree-cloud-4"},
    "861e": {"run_id": "__RUN_861E__", "substrate_commit": "17befb8c",
             "calib_draws": 10, "mean_mel": 2.2980323189226863e-05,
             "mel_reference": 2.5982e-05, "mean_duration_factor": 0.884,
             "machine": "ree-worker-1"},
}
# Seed 271 was write-address LOCKED in BOTH predecessors -- measured, not
# assumed. See the docstring's Step 2.5c disclosure.
PREDECESSOR_WRITE_LOCK = {
    "861c": {7: False, 271: True, 883: False},
    "861e": {7: False, 271: True, 883: False},
}
# A cycle counts as write-address locked when fewer than 2 slots moved, which is
# the same >=2-occupied-slots floor substrate_queue registered for
# contextmemory-write-path-addressing-degeneracy at V3-EXQ-436f/943.
WRITE_LOCK_INSUFFICIENT_FRAC = 0.5
'''

MEAS_RESEED_CONST = '''
# H1: fixed derived offset for the measurement-phase reseed. NOT 0 -- a reseed
# with the bare cell seed would replay the convergence stream rather than start
# an independent one.
MEAS_RESEED_OFFSET = 1000003
'''

RESEED_BLOCK = '''    if reseed_before_measurement:
        # H1: make the measurement-phase RNG stream a function of `seed` alone,
        # independent of how many calibration draws ran before it. Placed BEFORE
        # _make_env so env construction is inside the isolated stream too.
        torch.manual_seed(seed + MEAS_RESEED_OFFSET)
        np.random.seed(seed + MEAS_RESEED_OFFSET)
    meas_env = _make_env(seed, arm["interval"])'''


def build(key):
    leg = LEGS[key]
    s = base

    # (1) docstring
    end = s.index('"""', 3)
    s = preamble(leg) + BODIES[key] + COMMON_TAIL + s[end:]

    # (2) substrate pin, before the first ree_core import
    s = rep(s, ANCHOR_PIN,
            PIN_IMPORT.replace("__PIN_REF__", leg["pin_ref"])
                      .replace("__MARKER__", str(bool(leg["marker_present"]))),
            tag="pin-import")

    # (3) identity constants
    s = rep(s,
            'EXPERIMENT_TYPE = "v3_exq_861e_inv050_mech180_calibration_power_raised_replication"\n'
            'QUEUE_ID = "V3-EXQ-861e"\n'
            'CLAIM_IDS = ["INV-050", "MECH-180"]\n'
            'EXPERIMENT_PURPOSE = "evidence"\n',
            f'EXPERIMENT_TYPE = "{leg["exp_type"]}"\n'
            f'QUEUE_ID = "{leg["queue_id"]}"\n'
            'CLAIM_IDS = ["INV-050", "MECH-180"]\n'
            '# diagnostic, NOT evidence: an instrument-isolation leg. Excluded from\n'
            '# governance confidence/conflict scoring; directions pinned "unknown".\n'
            'EXPERIMENT_PURPOSE = "diagnostic"\n'
            + VARIANT_CONSTS.replace("__QID__", QID)
                            .replace("__HYP__", leg["qid_hyp"])
                            .replace("__AXIS__", leg["axis"])
                            .replace("__AUTOPSY__", AUTOPSY)
                            .replace("__VAR_A__", leg["variant_a"])
                            .replace("__VAR_B__", leg["variant_b"])
                            .replace("__RUN_861C__", RUN_861C)
                            .replace("__RUN_861E__", RUN_861E),
            tag="identity")

    # (4) compares_against -> 861e's run
    s = rep(s, 'COMPARES_AGAINST_RUN_ID = (', 'COMPARES_AGAINST_RUN_ID = (', tag="cmp-anchor")

    # (5) MEAS_RESEED_OFFSET constant (H1 only, but harmless to define once)
    if key == "861f":
        s = rep(s, '\nMEAS_CYCLES = 6\n', MEAS_RESEED_CONST + '\nMEAS_CYCLES = 6\n',
                tag="reseed-const")
    return s, leg


# --------------------------------------------------------------------------
# Part 3 -- cell variants, in-run control set, discrimination readout
# --------------------------------------------------------------------------
CELL_SIG_OLD = '''def _run_cell(seed: int, arm: Dict[str, Any], steps: int, conv_eps: int,
              meas_cycles: int, calib_draws: int,
              calib_eps_per_draw: int) -> Dict[str, Any]:
    """One (seed, arm) cell. Cell logic IDENTICAL to V3-EXQ-861c EXCEPT
    calib_draws defaults to 10 (was 5) via the module constant passed in."""'''

CELL_SIG_NEW = '''def _run_cell(seed: int, arm: Dict[str, Any], steps: int, conv_eps: int,
              meas_cycles: int, calib_draws: int,
              calib_eps_per_draw: int, *,
              variant: str = PRIMARY_VARIANT,
              reseed_before_measurement: bool = False,
              write_selection: str = "argmin") -> Dict[str, Any]:
    """One (seed, arm, variant) cell.

    Cell logic is byte-identical to V3-EXQ-861e except for the ONE knob this
    leg varies, which is carried by the keyword-only arguments above and
    recorded on the returned row as `variant`. Every cell still resets all RNG
    at entry, so cells are independent of each other and of their order --
    which is what makes the in-run control set a valid same-machine,
    same-substrate replica rather than a sequence effect."""'''

RESEED_ANCHOR = '    meas_env = _make_env(seed, arm["interval"])'

ROW_EXTRA = '''    return {
        "arm_id": arm_id,
        "variant": variant,
        "reseed_before_measurement": bool(reseed_before_measurement),
        "contextmemory_write_selection": str(write_selection),
        "calib_draws_this_cell": int(calib_draws),'''

ROW_ANCHOR = '''    return {
        "arm_id": arm_id,'''

MAKE_AGENT_H4_OLD = '''def _make_agent(env: CausalGridWorldV2, mel_on: bool, mel_reference: float) -> REEAgent:
    """UNCHANGED from 861c -- byte-identical config."""
    cfg = REEConfig.from_dims('''
MAKE_AGENT_H4_NEW = '''def _make_agent(env: CausalGridWorldV2, mel_on: bool, mel_reference: float,
                write_selection: str = "argmin") -> REEAgent:
    """Byte-identical to 861c/861e EXCEPT contextmemory_write_selection, the one
    knob this control leg varies. "argmin" reproduces 861e exactly (it is the
    substrate default); "refractory" engages the already-built non-degenerate
    write addressing validated at V3-EXQ-943. Both knobs exist at 17befb8c and
    neither exists at f810969, which is why this control pins 17befb8c."""
    cfg = REEConfig.from_dims(
        contextmemory_write_selection=str(write_selection),
        contextmemory_write_refractory_k=CONTEXTMEMORY_WRITE_REFRACTORY_K,'''

AGENT_CALL_OLD = '    agent = _make_agent(stable_env, mel_on=mel_on, mel_reference=0.0)'
AGENT_CALL_NEW = ('    agent = _make_agent(stable_env, mel_on=mel_on, mel_reference=0.0,\n'
                  '                        write_selection=write_selection)')

# ---- in-run control set --------------------------------------------------
PRIMARY_LOOP_ANCHOR = '''            with arm_cell(seed, config_slice=full_config,
                          script_path=Path(__file__)) as cell:
                row = _run_cell(seed, arm, steps, conv_eps, meas_cycles,
                                calib_draws, calib_eps_per_draw)
                cell.stamp(row)
            arm_results.append(row)
'''

PRIMARY_LOOP_NEW = '''            full_config["variant"] = PRIMARY_VARIANT
            full_config.update(_variant_config(PRIMARY_VARIANT, calib_draws))
            with arm_cell(seed, config_slice=full_config,
                          script_path=Path(__file__),
                          **_PIN_FP_KWARGS) as cell:
                row = _run_cell(seed, arm, steps, conv_eps, meas_cycles,
                                calib_draws, calib_eps_per_draw,
                                variant=PRIMARY_VARIANT,
                                **_variant_cell_kwargs(PRIMARY_VARIANT, calib_draws))
                cell.stamp(row)
            arm_results.append(row)

    # -- IN-RUN CONTROL SET: the decisive arm only, re-run under the CONTROL
    # variant. Same process, same machine, same pinned substrate; cells reset
    # all RNG at entry so order is irrelevant. This is what lets the declared
    # null be decided WITHIN this run instead of against a cross-machine
    # comparison -- see the docstring's "IN-RUN CONTROL" section.
    control_arm = next((a for a in arms if a["arm_id"] == DECISIVE_ARM_ID), None)
    if control_arm is not None:
        for seed in seeds:
            c_draws = _control_calib_draws(calib_draws)
            full_config = _base_config(control_arm, conv_eps, c_draws,
                                       calib_eps_per_draw, meas_cycles, steps)
            full_config["variant"] = CONTROL_VARIANT
            full_config.update(_variant_config(CONTROL_VARIANT, c_draws))
            with arm_cell(seed, config_slice=full_config,
                          script_path=Path(__file__),
                          **_PIN_FP_KWARGS) as cell:
                row = _run_cell(seed, control_arm, steps, conv_eps, meas_cycles,
                                c_draws, calib_eps_per_draw,
                                variant=CONTROL_VARIANT,
                                **_variant_cell_kwargs(CONTROL_VARIANT, c_draws))
                cell.stamp(row)
            arm_results.append(row)
'''

CONFIG_ANCHOR = '''            full_config = {
                "env_base": ENV_BASE,
                "arm": arm,'''

BASE_CONFIG_FN = '''def _base_config(arm: Dict[str, Any], conv_eps: int, calib_draws: int,
                 calib_eps_per_draw: int, meas_cycles: int, steps: int) -> Dict[str, Any]:
    """Config slice for one cell. Factored out of run_experiment (861e had it
    inline) so the in-run control set builds the SAME slice as the primary grid
    and the two fingerprints differ only by the varied knob."""
    return {
        "env_base": ENV_BASE,
        "arm": arm,
        "world_rule_shift_depth": WORLD_RULE_SHIFT_DEPTH,
        "conv_episodes": conv_eps,
        "calib_draws": calib_draws,
        "calib_episodes_per_draw": calib_eps_per_draw,
        "k_calib_margin": K_CALIB_MARGIN,
        "max_calib_rel_sd_of_mean": MAX_CALIB_REL_SD_OF_MEAN,
        "meas_cycles": meas_cycles,
        "steps_per_episode": steps,
        "sws_steps": SWS_CONSOLIDATION_STEPS,
        "rem_steps": REM_ATTRIBUTION_STEPS,
        "mel_gain": MEL_GAIN,
        "factor_min": FACTOR_MIN,
        "factor_max": FACTOR_MAX,
        "mel_relative_floor": MEL_RELATIVE_FLOOR,
        "touched_slot_l2_eps": TOUCHED_SLOT_L2_EPS,
        "use_mech122_spindle_content_selection": USE_MECH122_SPINDLE_CONTENT_SELECTION,
        "mech122_spindle_selection_gain": MECH122_SPINDLE_SELECTION_GAIN,
        "substrate_pin_ref": SUBSTRATE_PIN_REF,
    }


'''


VARIANT_HELPERS = {
    "861f": '''def _variant_config(variant: str, calib_draws: int) -> Dict[str, Any]:
    """The ONE knob this leg varies, folded into the cell's config slice so the
    primary and control fingerprints are distinguishable."""
    return {"reseed_before_measurement": variant == PRIMARY_VARIANT,
            "meas_reseed_offset": MEAS_RESEED_OFFSET}


def _variant_cell_kwargs(variant: str, calib_draws: int) -> Dict[str, Any]:
    return {"reseed_before_measurement": variant == PRIMARY_VARIANT}


def _control_calib_draws(calib_draws: int) -> int:
    """Unchanged: this leg's control varies the reseed, not calibration power."""
    return calib_draws


''',
    "861g": '''def _variant_config(variant: str, calib_draws: int) -> Dict[str, Any]:
    """The ONE knob varied between primary and in-run control here is
    CALIB_DRAWS, which _base_config already carries -- so nothing extra."""
    return {}


def _variant_cell_kwargs(variant: str, calib_draws: int) -> Dict[str, Any]:
    return {}


def _control_calib_draws(calib_draws: int) -> int:
    """Half the primary draw count. At the production CALIB_DRAWS=10 this is
    exactly 5, i.e. V3-EXQ-861c's own decisive condition, which is what makes
    the control a POSITIVE control for the pin. Expressed as a ratio so the
    --dry-run smoke exercises the same code path at its own scale."""
    return max(1, calib_draws // 2)


''',
    "861h": '''def _write_selection_for(variant: str) -> str:
    return "refractory" if variant == PRIMARY_VARIANT else "argmin"


def _variant_config(variant: str, calib_draws: int) -> Dict[str, Any]:
    return {"contextmemory_write_selection": _write_selection_for(variant),
            "contextmemory_write_refractory_k": CONTEXTMEMORY_WRITE_REFRACTORY_K}


def _variant_cell_kwargs(variant: str, calib_draws: int) -> Dict[str, Any]:
    return {"write_selection": _write_selection_for(variant)}


def _control_calib_draws(calib_draws: int) -> int:
    """Unchanged: this leg's control varies the write path, not calibration."""
    return calib_draws


''',
}

DISCRIM_COMMON = '''# -- Pre-registered decision thresholds for this leg's declared null ---------
# A cell is "HIGH-graded" iff its mean measured MEL sits above its OWN
# calibrated reference. mean_duration_factor IS that ratio (clamped to
# [FACTOR_MIN, FACTOR_MAX]), so > 1.0 is exactly the autopsy's own criterion --
# not a threshold invented here.
FACTOR_GRADED_FLOOR = 1.0
# 861c's recorded decisive factor was 1.215; the pin positive control is
# required to land within ~10% below it. Pre-registered from that recorded
# value, NOT tuned against this run.
PIN_CONTROL_MIN_FACTOR = 1.10


def _cell(rows: List[Dict[str, Any]], seed: int, variant: str,
          arm_id: str = DECISIVE_ARM_ID) -> Optional[Dict[str, Any]]:
    return next((r for r in rows
                 if r["seed"] == seed and r["arm_id"] == arm_id
                 and r.get("variant") == variant), None)


def _insufficient_frac(row: Optional[Dict[str, Any]]) -> Optional[float]:
    """Fraction of measurement cycles in which fewer than 2 ContextMemory slots
    moved -- the write-address lock signature of substrate_queue
    `contextmemory-write-path-addressing-degeneracy`."""
    if not row:
        return None
    n = float(row.get("n_cycles_insufficient_touched_slots", 0) or 0)
    d = float(row.get("meas_cycles", 0) or 0)
    return (n / d) if d > 0 else None


def _cell_summary(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    return {
        "arm_id": row["arm_id"], "seed": row["seed"], "variant": row.get("variant"),
        "mean_mel": row.get("mean_mel"),
        "mel_reference": row.get("mel_reference"),
        "mean_duration_factor": row.get("mean_duration_factor"),
        "high_graded": bool(float(row.get("mean_duration_factor", 0.0))
                            > FACTOR_GRADED_FLOOR),
        "calib_draws_this_cell": row.get("calib_draws_this_cell"),
        "mel_reference_calib_rel_sd_of_mean": row.get("mel_reference_calib_rel_sd_of_mean"),
        "per_cycle_n_touched_slots": row.get("per_cycle_n_touched_slots"),
        "n_cycles_insufficient_touched_slots": row.get("n_cycles_insufficient_touched_slots"),
        "write_address_locked": (
            (_insufficient_frac(row) or 0.0) >= WRITE_LOCK_INSUFFICIENT_FRAC),
        "contextmemory_write_selection": row.get("contextmemory_write_selection"),
        "reseed_before_measurement": row.get("reseed_before_measurement"),
    }


def _write_lock_audit(arm_results: List[Dict[str, Any]],
                      seeds: List[int]) -> Dict[str, Any]:
    """Per-seed ContextMemory write-address lock state, recorded so a later
    autopsy never has to re-derive it from per_cycle_n_touched_slots.
    Disclosure context: queue-experiment Step 2.5c, substrate_queue entry
    `contextmemory-write-path-addressing-degeneracy` (severity corrupting)."""
    out: Dict[str, Any] = {
        "sd_id": "contextmemory-write-path-addressing-degeneracy",
        "severity": "corrupting",
        "substrate_paths": ["ree_core/predictors/e1_deep.py"],
        "lock_criterion": ("fraction of measurement cycles with < 2 touched "
                           "slots >= %.2f" % WRITE_LOCK_INSUFFICIENT_FRAC),
        "predecessor_lock_state_measured_from_manifests": PREDECESSOR_WRITE_LOCK,
        "per_seed": {},
    }
    for s in seeds:
        per_variant = {}
        for v in (PRIMARY_VARIANT, CONTROL_VARIANT):
            rows = [r for r in arm_results
                    if r["seed"] == s and r.get("variant") == v]
            if not rows:
                continue
            fracs = [f for f in (_insufficient_frac(r) for r in rows)
                     if f is not None]
            per_variant[v] = {
                "n_cells": len(rows),
                "mean_insufficient_frac": (float(sum(fracs) / len(fracs))
                                           if fracs else None),
                "locked_cells": sum(1 for f in fracs
                                    if f >= WRITE_LOCK_INSUFFICIENT_FRAC),
                "decisive_arm_insufficient_frac": _insufficient_frac(
                    _cell(arm_results, s, v)),
            }
        out["per_seed"][str(s)] = per_variant
    return out


'''


DISCRIM_LEG = {
    "861f": '''def _discrimination(arm_results: List[Dict[str, Any]],
                    seeds: List[int]) -> Dict[str, Any]:
    """H1 verdict on the decisive cell, decided WITHIN this run.

    reseeded vs unreseeded, seed 271 ARM_3_HIGH_ON, everything else held. See
    the docstring's DECLARED NULL -- these branches are that text in code, and
    nothing here is computed from a threshold fitted to this run."""
    p = _cell(arm_results, DECISIVE_SEED, PRIMARY_VARIANT)
    c = _cell(arm_results, DECISIVE_SEED, CONTROL_VARIANT)
    ps, cs = _cell_summary(p), _cell_summary(c)
    pf = float(p.get("mean_duration_factor", 0.0)) if p else None
    cf = float(c.get("mean_duration_factor", 0.0)) if c else None

    control_reproduces_861e = (cf is not None and cf <= FACTOR_GRADED_FLOOR)
    reseeded_restores = (pf is not None and pf > FACTOR_GRADED_FLOOR)

    if not control_reproduces_861e:
        label = "uninformative_control_did_not_reproduce_861e_collapse"
        supported = None
    elif reseeded_restores:
        label = "h1_supported_intervention_isolation_defect"
        supported = True
    else:
        label = "h1_not_supported_collapse_survives_rng_isolation"
        supported = False

    machine_delta = None
    if cf is not None:
        machine_delta = {
            "note": ("in-run unreseeded control vs 861e's RECORDED decisive cell "
                     "-- same substrate, same protocol, different box. This is a "
                     "direct read on the machine half of H3, which the autopsy "
                     "could not otherwise separate. Both runs report "
                     "machine_class linux-x86_64-py3.10-torch2.12.0+cpu, so any "
                     "difference is sub-machine-class."),
            "control_factor": cf,
            "recorded_861e_factor": PREDECESSOR_DECISIVE["861e"]["mean_duration_factor"],
            "abs_delta": abs(cf - PREDECESSOR_DECISIVE["861e"]["mean_duration_factor"]),
        }
    return {
        "verdict_label": label,
        "hypothesis_supported": supported,
        "primary_cell": ps, "control_cell": cs,
        "readings": {
            "control_reproduces_861e_collapse": bool(control_reproduces_861e),
            "reseeded_restores_high_grading": bool(reseeded_restores),
        },
        "machine_delta_readout": machine_delta,
    }


''',
    "861g": '''def _discrimination(arm_results: List[Dict[str, Any]],
                    seeds: List[int]) -> Dict[str, Any]:
    """H3 verdict on the decisive cell.

    Primary = f810969 at CALIB_DRAWS=10 (the missing 2x2 cell). In-run control =
    f810969 at CALIB_DRAWS=5, i.e. 861c's own decisive condition, which doubles
    as the POSITIVE control that verifies the pin behaviourally. See the
    docstring's DECLARED NULL."""
    p = _cell(arm_results, DECISIVE_SEED, PRIMARY_VARIANT)
    c = _cell(arm_results, DECISIVE_SEED, CONTROL_VARIANT)
    ps, cs = _cell_summary(p), _cell_summary(c)
    pf = float(p.get("mean_duration_factor", 0.0)) if p else None
    cf = float(c.get("mean_duration_factor", 0.0)) if c else None

    # The pin is verified structurally + behaviourally at import time; this is
    # the third, empirical check: does the OLD substrate at 861c's own
    # CALIB_DRAWS reproduce 861c's own decisive number?
    pin_control_ok = (cf is not None and cf >= PIN_CONTROL_MIN_FACTOR)
    primary_high_graded = (pf is not None and pf > FACTOR_GRADED_FLOOR)

    if not pin_control_ok:
        label = "uninformative_pin_positive_control_did_not_reproduce_861c"
        supported = None
    elif primary_high_graded:
        label = "h3_supported_old_substrate_retains_high_grading_at_n10"
        supported = True
    else:
        label = "h3_not_supported_collapse_present_on_old_substrate_at_n10"
        supported = False

    # Machine-free read on CALIB_DRAWS: n=10 vs n=5 on ONE substrate, ONE box.
    calib_only = None
    if pf is not None and cf is not None:
        calib_only = {
            "note": ("n=10 vs n=5 on the SAME pinned substrate and the SAME box. "
                     "861c-vs-861e could not isolate this because substrate and "
                     "machine moved with CALIB_DRAWS."),
            "factor_n10": pf, "factor_n5": cf, "delta": pf - cf,
            "calib_draws_alone_moves_readout": bool(
                (cf > FACTOR_GRADED_FLOOR) != (pf > FACTOR_GRADED_FLOOR)),
        }
    return {
        "verdict_label": label,
        "hypothesis_supported": supported,
        "primary_cell": ps, "control_cell": cs,
        "readings": {
            "pin_positive_control_reproduces_861c": bool(pin_control_ok),
            "pin_control_min_factor": PIN_CONTROL_MIN_FACTOR,
            "recorded_861c_factor": PREDECESSOR_DECISIVE["861c"]["mean_duration_factor"],
            "primary_high_graded_at_n10_on_f810969": bool(primary_high_graded),
        },
        "calibration_only_readout": calib_only,
    }


''',
    "861h": '''def _discrimination(arm_results: List[Dict[str, Any]],
                    seeds: List[int]) -> Dict[str, Any]:
    """Substrate-defect control verdict.

    refractory vs legacy argmin write addressing, seed 271 ARM_3_HIGH_ON,
    everything else held, decided WITHIN this run. The FIRST thing checked is
    whether the repair engaged at all -- a control that did not control is
    uninformative, and that is a pre-registered readiness precondition, not a
    post-hoc excuse. See the docstring's DECLARED NULL."""
    p = _cell(arm_results, DECISIVE_SEED, PRIMARY_VARIANT)
    c = _cell(arm_results, DECISIVE_SEED, CONTROL_VARIANT)
    ps, cs = _cell_summary(p), _cell_summary(c)
    pf = float(p.get("mean_duration_factor", 0.0)) if p else None
    cf = float(c.get("mean_duration_factor", 0.0)) if c else None
    p_lock = _insufficient_frac(p)
    c_lock = _insufficient_frac(c)

    repair_engaged = (p_lock is not None and c_lock is not None
                      and p_lock < WRITE_LOCK_INSUFFICIENT_FRAC
                      and c_lock >= WRITE_LOCK_INSUFFICIENT_FRAC)
    repaired_restores = (pf is not None and pf > FACTOR_GRADED_FLOOR)
    legacy_reproduces = (cf is not None and cf <= FACTOR_GRADED_FLOOR)

    if not repair_engaged:
        label = "uninformative_write_repair_did_not_engage_on_decisive_seed"
        control_passes = None
    elif repaired_restores and legacy_reproduces:
        label = "control_FAILS_decisive_readout_depends_on_write_address_lock"
        control_passes = False
    else:
        label = "control_PASSES_write_address_lock_not_load_bearing_for_mel"
        control_passes = True

    return {
        "verdict_label": label,
        "control_passes": control_passes,
        "primary_cell": ps, "control_cell": cs,
        "readings": {
            "repair_engaged_on_decisive_seed": bool(repair_engaged),
            "refractory_insufficient_frac": p_lock,
            "legacy_insufficient_frac": c_lock,
            "refractory_restores_high_grading": bool(repaired_restores),
            "legacy_reproduces_861e_collapse": bool(legacy_reproduces),
        },
        "governance_note": (
            "If control_passes is False, 861f/861g's H1/H3 readings are "
            "conditional on a known CORRUPTING substrate defect and a fourth "
            "hypothesis owes registration against qid " + FANOUT_QID + ". That "
            "is a governance call (the frozen set is H1+H3, count 2, adopted at "
            "an interactive gate) -- this leg reports it, it does not make it."),
    }


''',
}


# --- variant-scoped filters: readiness / C1 / C2 must read the PRIMARY grid
#     only, never the in-run control cells. ---------------------------------
FILTER_PATCHES = [
    ('''        on_eco_cells = [r for r in arm_results
                        if r["seed"] == seed and r["mel_on"]]''',
     '''        on_eco_cells = [r for r in arm_results
                        if r["seed"] == seed and r["mel_on"]
                        and r.get("variant") == PRIMARY_VARIANT]''',
     "readiness-filter"),
    ('''        on_eco = {r["arm_id"]: r for r in arm_results
                  if r["seed"] == seed and r["mel_on"]}''',
     '''        on_eco = {r["arm_id"]: r for r in arm_results
                  if r["seed"] == seed and r["mel_on"]
                  and r.get("variant") == PRIMARY_VARIANT}''',
     "c1-filter"),
    ('''        off_cell = next((r for r in arm_results
                         if r["seed"] == seed and r["arm_id"] == "ARM_4_HIGH_OFF"), None)''',
     '''        off_cell = next((r for r in arm_results
                         if r["seed"] == seed and r["arm_id"] == "ARM_4_HIGH_OFF"
                         and r.get("variant") == PRIMARY_VARIANT), None)''',
     "off-filter"),
    ('''                float(next((r for r in arm_results
                            if r["seed"] == s and r["arm_id"] == "ARM_3_HIGH_ON"),
                           {}).get("mel_reference_calib_rel_sd_of_mean", 0.0))''',
     '''                float(next((r for r in arm_results
                            if r["seed"] == s and r["arm_id"] == "ARM_3_HIGH_ON"
                            and r.get("variant") == PRIMARY_VARIANT),
                           {}).get("mel_reference_calib_rel_sd_of_mean", 0.0))''',
     "calib-filter"),
]

INTERP_ANCHOR = '''    criteria = [
        {"name": "C1a_sws_power_monotone_in_measured_mel", "load_bearing": True,'''

INTERP_INJECT = '''    # -- Leg-specific: fan-out identity, the in-run discrimination, and the
    #    ContextMemory write-lock audit. Injected into the 861e interpretation
    #    block rather than replacing it, so the replicated C1/C2 grid stays
    #    directly comparable to 861c/861e.
    discrimination = _discrimination(arm_results, seeds)
    write_lock_audit = _write_lock_audit(arm_results, seeds)
    interpretation["fanout"] = {
        "qid": FANOUT_QID,
        "hypothesis": FANOUT_HYPOTHESIS,
        "axis": FANOUT_AXIS,
        "source_autopsy": FANOUT_SOURCE_AUTOPSY,
        "primary_variant": PRIMARY_VARIANT,
        "control_variant": CONTROL_VARIANT,
        "portfolio": ["V3-EXQ-861f (H1, measurement)",
                      "V3-EXQ-861g (H3, algorithm)",
                      "V3-EXQ-861h (substrate-defect control, representation)"],
        "frozen_set_note": ("The registry's frozen set for this qid is H1 + H3 "
                            "(count 2), H2 a labelled follow-on. V3-EXQ-861h is a "
                            "CONTROL added by the queue-experiment Step 2.5b(iv) "
                            "design audit and the Step 2.5c corrupting-overlap "
                            "gate, NOT a fourth frozen hypothesis."),
    }
    interpretation["discrimination"] = discrimination
    interpretation["contextmemory_write_lock_audit"] = write_lock_audit
    interpretation["preconditions"].extend(_leg_preconditions(discrimination))
    interpretation["criteria_non_degenerate"].update({
        "in_run_control_set_present": bool(discrimination.get("control_cell")),
        "primary_and_control_cells_differ": bool(
            discrimination.get("primary_cell") and discrimination.get("control_cell")
            and discrimination["primary_cell"].get("mean_duration_factor")
            != discrimination["control_cell"].get("mean_duration_factor")),
        "substrate_pin_verified": bool(_PIN.get("verified")),
    })
    interpretation["diagnostic_scope_note"] = (
        "EXPERIMENT_PURPOSE is diagnostic. The replicated C1/C2/readiness grid "
        "below is recorded for comparability with 861c/861e and does NOT vote on "
        "INV-050 or MECH-180; evidence_direction_per_claim is pinned unknown "
        "regardless of how it comes out.")

''' + INTERP_ANCHOR

LEG_PRECONDITIONS = {
    "861f": '''def _leg_preconditions(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The unstated assumption behind every H1 branch: the in-run unreseeded
    control must actually reproduce 861e's collapse. If it does not, this box
    is not reproducing 861e at all and there is nothing to isolate against --
    a precondition, not a post-hoc excuse (queue-experiment P0 readiness-assert)."""
    cs = d.get("control_cell") or {}
    return [{
        "name": "in_run_unreseeded_control_reproduces_861e_collapse",
        "kind": "readiness",
        "description": ("the unreseeded ARM_3_HIGH_ON cell on seed 271 must sit "
                        "at or below its own mel_reference (factor <= 1.0), as "
                        "861e recorded (0.884), before a reseeded-vs-unreseeded "
                        "contrast can mean anything."),
        "measured": cs.get("mean_duration_factor"),
        "threshold": FACTOR_GRADED_FLOOR,
        "direction": "upper",
        "control": ("in-run cell, same process/machine/substrate as the primary "
                    "grid; 861e recorded %.3f for this cell"
                    % PREDECESSOR_DECISIVE["861e"]["mean_duration_factor"]),
        "met": bool(d["readings"]["control_reproduces_861e_collapse"]),
    }]


''',
    "861g": '''def _leg_preconditions(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The unstated assumption behind every H3 branch: the pin must reproduce
    861c's own decisive number at 861c's own CALIB_DRAWS. Structural and
    behavioural pin checks run at import; this is the empirical one, and it is
    a precondition rather than a finding (queue-experiment P0 readiness-assert)."""
    cs = d.get("control_cell") or {}
    return [{
        "name": "pin_positive_control_reproduces_861c_decisive_cell",
        "kind": "readiness",
        "description": ("the f810969-pinned ARM_3_HIGH_ON cell on seed 271 at "
                        "CALIB_DRAWS=5 must land at or above PIN_CONTROL_MIN_FACTOR, "
                        "i.e. near 861c's recorded 1.215, before the n=10 primary "
                        "grid can be read as an H3 answer."),
        "measured": cs.get("mean_duration_factor"),
        "threshold": PIN_CONTROL_MIN_FACTOR,
        "direction": "lower",
        "control": ("861c's own recorded decisive value (%.3f) on the same "
                    "substrate commit; threshold pre-registered at ~10%% below "
                    "it, not fitted to this run"
                    % PREDECESSOR_DECISIVE["861c"]["mean_duration_factor"]),
        "met": bool(d["readings"]["pin_positive_control_reproduces_861c"]),
    }]


''',
    "861h": '''def _leg_preconditions(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The unstated assumption behind every branch here: the write repair must
    actually FIRE on the decisive seed. A control that did not control cannot
    discharge a corrupting-overlap gate (queue-experiment P0 readiness-assert)."""
    r = d["readings"]
    return [{
        "name": "write_repair_engaged_on_decisive_seed",
        "kind": "readiness",
        "description": ("on seed 271 ARM_3_HIGH_ON, the refractory arm's fraction "
                        "of measurement cycles with < 2 touched ContextMemory "
                        "slots must fall below WRITE_LOCK_INSUFFICIENT_FRAC while "
                        "the legacy argmin control stays at or above it. 861e "
                        "recorded 6/6 insufficient cycles on this exact cell."),
        "measured": r.get("refractory_insufficient_frac"),
        "threshold": WRITE_LOCK_INSUFFICIENT_FRAC,
        "direction": "upper",
        "control": ("in-run legacy argmin cell, same process/machine/substrate; "
                    "measured legacy fraction %s"
                    % (r.get("legacy_insufficient_frac"),)),
        "met": bool(r.get("repair_engaged_on_decisive_seed")),
    }]


''',
}


RETURN_OLD = '''    return {
        "outcome": outcome,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {
            "INV-050": inv050_direction,
            "MECH-180": mech180_direction,
        },
        "interpretation": interpretation,'''

RETURN_NEW = '''    # EXPERIMENT_PURPOSE is diagnostic: this leg answers a question about the
    # INSTRUMENT, so it must not vote on either claim no matter how the
    # replicated grid came out. The computed grid directions are preserved
    # verbatim under replication_readout for comparison with 861c/861e.
    return {
        "outcome": outcome,
        "evidence_direction": "non_contributory",
        "evidence_direction_per_claim": {
            "INV-050": "unknown",
            "MECH-180": "unknown",
        },
        "evidence_direction_note": (
            "Pinned non_contributory / unknown by construction: this is a "
            "GOV-FANOUT-1 instrument-isolation leg (" + FANOUT_HYPOTHESIS + ", "
            + FANOUT_AXIS + " axis) for qid " + FANOUT_QID + ", not evidence "
            "for or against INV-050 or MECH-180. Neither claim's status, "
            "confidence or v3_pending may move on it."),
        "replication_readout": {
            "note": ("What the unmodified 861e verdict machinery computed on this "
                     "leg's PRIMARY grid, recorded for comparability with "
                     "861c/861e. Does NOT vote -- see evidence_direction_note."),
            "grid_evidence_direction": direction,
            "grid_evidence_direction_per_claim": {
                "INV-050": inv050_direction,
                "MECH-180": mech180_direction,
            },
        },
        "discrimination": discrimination,
        "contextmemory_write_lock_audit": write_lock_audit,
        "substrate_pin": pin_manifest_block(_PIN),
        "interpretation": interpretation,'''

MANIFEST_ANCHOR = '        "timestamp_utc": ts,\n        "seeds": SEEDS,\n'
MANIFEST_NEW = ('        "timestamp_utc": ts,\n        "seeds": SEEDS,\n'
                '        "substrate_pin": pin_manifest_block(_PIN),\n'
                '        "fanout_qid": FANOUT_QID,\n'
                '        "fanout_hypothesis": FANOUT_HYPOTHESIS,\n'
                '        "fanout_axis": FANOUT_AXIS,\n'
                '        "fanout_source_autopsy": FANOUT_SOURCE_AUTOPSY,\n'
                '        "discrimination": result["discrimination"],\n'
                '        "contextmemory_write_lock_audit":\n'
                '            result["contextmemory_write_lock_audit"],\n'
                '        "replication_readout": result["replication_readout"],\n'
                '        "evidence_direction_note": result["evidence_direction_note"],\n')

MAIN_PRINT_ANCHOR = '''    print(f"manifest: {out_path}", flush=True)'''
MAIN_PRINT_NEW = '''    _d = result["discrimination"]
    print(f"fanout: qid={FANOUT_QID} hypothesis={FANOUT_HYPOTHESIS} "
          f"axis={FANOUT_AXIS}", flush=True)
    print(f"substrate_pin: ref={SUBSTRATE_PIN_REF[:10]} verified={_PIN['verified']} "
          f"dir={_PIN['pin_dir']}", flush=True)
    print(f"discrimination: {_d['verdict_label']}", flush=True)
    print(f"  primary({PRIMARY_VARIANT})={_d['primary_cell']} ", flush=True)
    print(f"  control({CONTROL_VARIANT})={_d['control_cell']}", flush=True)
    print(f"  readings={_d['readings']}", flush=True)
    print(f"manifest: {out_path}", flush=True)'''

# The smoke must exercise the in-run control set too -- a control path that is
# never run in --dry-run is a path whose first execution is the 5-hour run.
DRYRUN_ANCHOR = '''        smoke_ids = {"ARM_0_NONE_ON", "ARM_3_HIGH_ON", "ARM_4_HIGH_OFF"}'''
DRYRUN_NEW = '''        # DECISIVE_ARM_ID must stay in the smoke arm set: the in-run control
        # set is scoped to it, so dropping it would silently skip the control
        # path and _discrimination() entirely.
        smoke_ids = {"ARM_0_NONE_ON", DECISIVE_ARM_ID, "ARM_4_HIGH_OFF"}'''

REFRACTORY_K = '''
# MECH/substrate-control knob for V3-EXQ-861h. k=2 is the substrate default and
# the value V3-EXQ-943 validated (locking seeds settle at exactly k+1 occupied
# slots). Not tuned here.
CONTEXTMEMORY_WRITE_REFRACTORY_K = 2
'''


def build_full(key):
    s, leg = build(key)

    # cell signature + variant plumbing
    s = rep(s, CELL_SIG_OLD, CELL_SIG_NEW, tag="cell-sig")
    s = rep(s, ROW_ANCHOR, ROW_EXTRA, tag="row-extra")
    if key == "861f":
        s = rep(s, RESEED_ANCHOR, RESEED_BLOCK, tag="reseed")
    if key == "861h":
        s = rep(s, '\nMEAS_CYCLES = 6\n', REFRACTORY_K + '\nMEAS_CYCLES = 6\n',
                tag="refractory-k")
        s = rep(s, MAKE_AGENT_H4_OLD, MAKE_AGENT_H4_NEW, tag="make-agent")
        s = rep(s, AGENT_CALL_OLD, AGENT_CALL_NEW, tag="agent-call")

    # helpers + discrimination, inserted just above _run_cell
    helpers = (BASE_CONFIG_FN + VARIANT_HELPERS[key] + DISCRIM_COMMON
               + DISCRIM_LEG[key] + LEG_PRECONDITIONS[key])
    s = rep(s, "def _run_cell(", helpers + "def _run_cell(", tag="helpers")

    # run_experiment: config slice via _base_config + control set
    s = rep(s, CONFIG_ANCHOR + '''
                "world_rule_shift_depth": WORLD_RULE_SHIFT_DEPTH,
                "conv_episodes": conv_eps,
                "calib_draws": calib_draws,
                "calib_episodes_per_draw": calib_eps_per_draw,
                "k_calib_margin": K_CALIB_MARGIN,
                "max_calib_rel_sd_of_mean": MAX_CALIB_REL_SD_OF_MEAN,
                "meas_cycles": meas_cycles,
                "steps_per_episode": steps,
                "sws_steps": SWS_CONSOLIDATION_STEPS,
                "rem_steps": REM_ATTRIBUTION_STEPS,
                "mel_gain": MEL_GAIN,
                "factor_min": FACTOR_MIN,
                "factor_max": FACTOR_MAX,
                "mel_relative_floor": MEL_RELATIVE_FLOOR,
                "touched_slot_l2_eps": TOUCHED_SLOT_L2_EPS,
                "use_mech122_spindle_content_selection": USE_MECH122_SPINDLE_CONTENT_SELECTION,
                "mech122_spindle_selection_gain": MECH122_SPINDLE_SELECTION_GAIN,
            }
''',
            '''            full_config = _base_config(arm, conv_eps, calib_draws,
                                       calib_eps_per_draw, meas_cycles, steps)
''', tag="base-config")
    s = rep(s, PRIMARY_LOOP_ANCHOR, PRIMARY_LOOP_NEW, tag="control-set")

    for old, new, tag in FILTER_PATCHES:
        s = rep(s, old, new, tag=tag)

    s = rep(s, INTERP_ANCHOR, INTERP_INJECT, tag="interp")
    s = rep(s, RETURN_OLD, RETURN_NEW, tag="return")
    s = rep(s, MANIFEST_ANCHOR, MANIFEST_NEW, tag="manifest")
    s = rep(s, MAIN_PRINT_ANCHOR, MAIN_PRINT_NEW, tag="main-print")
    s = rep(s, DRYRUN_ANCHOR, DRYRUN_NEW, tag="dryrun")

    # compares_against -> 861e's actual run_id
    s = rep(s, 'COMPARES_AGAINST_RUN_ID = (\n', 'COMPARES_AGAINST_RUN_ID = (\n', tag="cmp2")

    out = V3 / "experiments" / (leg["exp_type"] + ".py")
    out.write_text(s)
    bad = [(i + 1, ln) for i, ln in enumerate(s.splitlines())
           if any(ord(c) > 127 for c in ln)]
    print(f"{leg['queue_id']}: {out.name}  lines={len(s.splitlines())} "
          f"non_ascii={bad[:3]}")
    return out


for k in ("861f", "861g", "861h"):
    build_full(k)
print("generated")
