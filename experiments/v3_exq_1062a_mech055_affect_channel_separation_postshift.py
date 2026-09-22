#!/opt/local/bin/python3
"""
V3-EXQ-1062a: MECH-055 affective channel separation -- the NARROWED
two-axis-plus-harm-only REPRESENTATIONAL separation diagnostic, RE-POSED with a
POST-TRAINING world-rule-shift ONSET.

experiment_purpose: diagnostic

SUPERSEDES V3-EXQ-1062 (run_id ..._20260922T175212Z_v3, FAIL / non_contributory).
Its confirmed autopsy -- failure_autopsy_V3-EXQ-1062_2026-09-22.{json,md},
ratified by governance cycle gov-20260922-1756 -- is this driver's brief.

WHAT CHANGED, AND THE ONE THING THAT DID NOT
--------------------------------------------
Exactly ONE thing changed in the manipulation: its ONSET. The dose is
UNCHANGED (interval 10, depth 2, identical to 1062's ARM_2_HIGH_SHIFT), the
schedule is UNCHANGED (P0 30 / P1 60 / P2 1800 steps), and every
pre-registered threshold and readiness floor is UNCHANGED, to the digit. That
is deliberate: a single-variable change makes 1062 -> 1062a a controlled
comparison that isolates onset, and it is what lets this run speak to
hypothesis H2 below. Changing the dose as well would alias onset with dose --
which is precisely the aliasing the autopsy withdrew its own first draft for
(autopsy section 5c).

THE 1062 FAILURE WAS AN ONSET GAP, NOT A DOSE GAP
--------------------------------------------------
world_rule_shift is schedulable only by a modulo interval/depth fixed at env
CONSTRUCTION, and its counter deliberately survives reset()
(causal_grid_world.py:984-985 and :1879-1885). A driver that needs a
TRAINED-THEN-SHIFTED regime therefore has no kwarg to start the shift after
training. V3-EXQ-1062 consequently trained its harm-forward model under ~810
accumulated permutations (P0 30 eps + P1 60 eps x <=90 steps at interval 10),
so the model learned an ACTION-MARGINALISED predictor from the outset and P2
then presented it with exactly the regime it was trained on. The lever's own
premise -- a learned stable map made "systematically wrong ... until
re-learned" -- was never instantiated at any dose. The manifest carried the
signature: harm_a_forward_r2 was statistically indistinguishable between arms
(0.932-0.954 shift vs 0.940-0.977 stationary) DESPITE those ~810 permutations.

Consequence for the claim layer: the decisive falsifier criterion C1 was never
SCORED. C1 is evaluated on SHIFT arms only, the single shift arm failed two
readiness preconditions (harm exposure ~4.28x against a 1.25x ceiling; harm-PE
load 3.3% LOWER against a +5% floor), so c1_arms was empty and the run routed
substrate_not_ready_requeue / non_contributory. Nothing about MECH-055 was
learned.

THE FIX, WHICH NEEDS NO SUBSTRATE CHANGE
-----------------------------------------
Both arms construct the env with world_rule_shift DISABLED and train P0+P1 on
a stationary world. At the P1 -> P2 boundary the SHIFT arm sets
env.world_rule_shift_enabled / _interval / _depth DIRECTLY -- public
attributes, no substrate change -- so the onset lands at a phase boundary.

Verified empirically before authoring (Step 2.5a probe, 2026-09-22, recorded
in custom_information.step_2_5a_probe_2026_09_22): with the lever disabled,
60 env steps leave _action_map bit-identical to the canonical ACTIONS map and
_world_steps_total at 0 -- every RNG draw is inside the enabled guard
(_maybe_shift_world_rule's own docstring), so a disabled env consumes no
randomness at all. Setting the three attributes then fires the first shift at
world step 10 and every 10 steps after (6 fires in 60 steps at interval 10),
permuting 4 of the 5 action-map entries (ACTIONS has 5: the four
moves plus a stay). The onset is clean and the canonical
map really is what training saw.

This driver ASSERTS that rather than trusting it: action_map_canonical_at_p2_onset
is recorded per cell, n_world_rule_shifts_p2 is recorded per cell, and a new
readiness precondition (world_rule_shift_fired_in_p2) gates the SHIFT arm on
the manipulation having actually occurred.

Registered substrate gap, named as a KNOWN LIMITATION rather than papered
over: SD-PP-B4-one-shot-world-rule-shift-lever (substrate_queue.json), which
now carries V3-EXQ-1062's failure record at severity `degrading` and lists
MECH-055 in unblocks_claims. `degrading` WARNS, it does not block. The
supported lever this driver works around would be a world_rule_shift_at_step /
apply_action_permutation kwarg; until that lands, the attribute write above IS
the sanctioned workaround (autopsy section 9, step 1).

THE THREE LIVE HYPOTHESES, AND WHICH ONE THIS RUN CAN SEPARATE
---------------------------------------------------------------
The autopsy names three readings of 1062's flat residual, and warns that a
same-schedule dose ladder would alias all three to one verdict:

  H1 exposure and forward-model error are structurally COUPLED under this
     lever (so no setting raises PE while holding exposure matched);
  H2 the model adapted because it TRAINED under the shift;
  H3 the harm-forward model is PERSISTENCE-DOMINATED (harm_history_len=10;
     z_harm_a is a slow harm-history latent), so no action-map lever at any
     dose can raise its residual.

H2 is what the onset fix tests DIRECTLY: if harm_a_forward_r2 now DIVERGES
between arms where 1062 found it indistinguishable, H2 is confirmed and the
lever does reach the harm-forward model once the model was trained on a
stationary world.

H3 is INSTRUMENTED here for the first time, at zero extra env cost, because
1062 could not distinguish it from H1 (autopsy: "Measurement | partial |
harm-forward ACTION-sensitivity never instrumented (no persistence
baseline)"). Two new RECORDED, NON-GATING readouts:

  * persistence_r2 / harm_forward_skill_vs_persistence -- the same estimator
    as harm_a_forward_r2, over the same P2 pairs, with the trivial predictor
    z_pred = z_harm_a(t-1). This is exactly the delta == 0 model:
    ResidualHarmForward returns z + delta(z, a) (latent/stack.py), so a
    learned model that drives delta toward zero scores well on an
    autocorrelated signal while being action-blind. Skill = 1 - SSE_model /
    SSE_persistence; at or below 0 the forward model is not beating
    persistence and H3 holds.
  * harm_forward_action_sensitivity -- the mean L2 between
    e2_harm_a(z(t-1), a_actual) and e2_harm_a(z(t-1), a_counterfactual) for a
    deterministically chosen DIFFERENT action drawn from the env's own
    _action_map keys, normalised by the mean L2 of the model's own predicted
    delta. At ~0 the trained model ignores the action entirely, so a
    re-permutation of the action -> displacement map CANNOT raise its residual
    no matter what the dose is.

H1 is what a failing harm_exposure_relative_deviation_bounded would indicate.
It is NOT separable from a bare readiness failure by the gating statistic
alone, so this driver additionally records an EARLY-WINDOW copy of both
cross-arm preconditions (see below).

THESE INSTRUMENTS ADD NO NEW WAY TO VACATE THE RUN
---------------------------------------------------
persistence_r2, harm_forward_skill_vs_persistence and
harm_forward_action_sensitivity are RECORDED and NON-GATING. They are consumed
in exactly one place: the DECIDABILITY branch of the verdict grid, to say WHICH
of H1/H2/H3 a readiness failure is consistent with. They can neither create
nor remove a green arm, and they never touch a claim verdict. This is
deliberate -- over-gating is how 1062 lost its whole run, and a third gating
precondition would raise the chance of a third non-contributory result without
adding information.

WINDOWED DIAGNOSTICS -- RECORDED, NON-GATING, AND WHY THE BARS DID NOT MOVE
---------------------------------------------------------------------------
Exposure deviation ACCUMULATES with time under the shift; PE elevation is
immediate. So the same run can be matched-on-exposure early and unmatched
late. This driver records mean_harm_exposure_early / mean_dacc_pe_early over
the first EARLY_WINDOW_STEPS P2 env steps after onset, and the arm-level
harm_exposure_rel_dev_early / pe_elevation_early computed from them.

The GATING preconditions are unchanged and are computed on the FULL P2 window
exactly as pre-registered (HARM_EXPOSURE_REL_DEV_MAX 0.25, PE_ELEVATION_MIN
0.05). The early-window copies are explicitly NON-GATING and are named
*_early_nongating in the manifest. Their only purpose is that if this run DOES
vacate on exposure, the manifest already carries the number a successor needs
in order to decide whether a shorter measurement window would clear the bar --
instead of costing another 4.6-hour run to find out. NO BAR WAS RELAXED TO
MAKE AN ARM GREEN, and none may be: if the onset-fixed lever cannot satisfy
the pre-registered preconditions, THAT IS THE FINDING and it routes to
/failure-autopsy.

EXPLICITLY IN SCOPE (the autopsy's section 5d gap, closed rather than left
implicit): whether this lever moves the dACC harm-PE channel AT ALL. No prior
run has measured that. pe_load_elevated_vs_stationary's control cites
V3-EXQ-861e's ecological_novelty_mel_gradient_present_this_config, which is a
MEL-GRADIENT DV on a DIFFERENT channel -- so 861e establishes that the IV is
non-degenerate, and establishes NOTHING about this channel. That control text
is corrected below to say so. This run is the first measurement of the lever
against the dACC harm-PE channel under the regime the lever was designed for,
and the H3 instruments above are what make a negative reading attributable
rather than ambiguous.

MECH-055 asserts that hedonic tone, valence, and signed PE stay distinct
channels. Its own what_would_answer (claims.yaml, 2026-08-08) states that the
FULL three-axis-with-harm/benefit-duality test CANNOT yet run -- MECH-054's
2026-08-08 finding is that only the HARM side of axis 3 is real, with no
benefit-side forward model and no benefit-side precision tracker anywhere in
ree_core -- and that "a narrower two-axis-plus-harm-only test CAN run now".
This is that narrower test. It is deliberately scoped to the REPRESENTATIONAL
level, which the claim explicitly licenses: "downstream behavioural legibility
gap is separate and doesn't block a representational-level test."

WHAT THIS RUN DOES AND DOES NOT DECIDE
--------------------------------------
IN SCOPE. The claim's own two FALSIFIERS, at the representational level:
  (i)  "any two axes move in a fixed, predictable ratio under manipulations
       designed to perturb only one"                              -> C2
  (ii) "VALENCE_HARM_DISCRIMINATIVE and the harm-side forward-model PE are
       numerically redundant (correlation ~1) under a decoupling
       manipulation"                                              -> C1
plus the claim's second CONFIRMING clause, that the two harm representations
"show measurably different variance/timing signatures under a manipulation
that decouples their upstream sources"                            -> C3, C4

OUT OF SCOPE, stated so no reader over-reads a PASS:
  (a) The benefit-side signed-PE channel. It does not exist (MECH-054). No
      result here can verdict the harm/benefit duality the claim's wording
      requires, which is why a clean pass records evidence_direction "mixed"
      and NOT "supports", and why experiment_purpose is "diagnostic".
  (b) The claim's FIRST confirming clause in its behavioural form -- effects on
      "that axis's own documented downstream role". V3-EXQ-799 confirmed axis
      1's internal mechanism fires (mode-prior entropy moves with mu, 1.105
      nats) but found NO behavioural DV downstream with enough sensitivity to
      detect it (write_gate breadth gap, routed to /implement-substrate, not
      landed). This run therefore measures the axes' own values and their
      joint structure, not a behavioural consequence. It cannot re-derive the
      conversion / F-dominance ceiling because it has no behavioural DV.
  (c) MECH-035's ranking claim (Pareto/lexicographic vs scalar). The
      cross-candidate valence range is RECORDED here as a non-gating
      diagnostic, not tested -- MECH-035 is deliberately NOT tagged.

THE DECOUPLING MANIPULATION, AND THE THREE LEVERS RULED OUT FIRST
-----------------------------------------------------------------
The claim demands "a manipulation that decouples their upstream sources".
VALENCE_HARM_DISCRIMINATIVE is fed by z_harm_s (a LEVEL, RBF-smoothed onto map
nodes); the dACC channel is fed by z_harm_a through E2HarmAForward (a temporal
forward-model RESIDUAL, precision-weighted -- MECH-258). Three candidate
levers were checked against the substrate and REJECTED before this design:

  1. SD-021 / AIC descending attenuation of z_harm_s. REJECTED: the gain is
     harm_s_gain = 1 - base_attenuation * mode_weight * drive_protect, and
     mode_weight = p_external * (1.0 if beta_gate_elevated else 0.0)
     (ree_core/cingulate/aic_analog.py:249). It is gated on the COMMITMENT
     latch, so it cannot fire commitment-free.
  2. harm_nonredundancy_weight (the SD-019 cosine^2 penalty between z_harm_s
     and z_harm_a). REJECTED as a FLOOR: V3-EXQ-323 measured BASELINE
     cosine_sq at 8.5e-05 to 0.025 across seeds -- the streams are already
     near-orthogonal unpenalised, the penalty has no headroom, and 323 failed
     its own C1 in 2/5 seeds for exactly that reason. (Read positively, this
     is good news for MECH-055 and is why the LATENTS are not what C1 tests.)
     Note also that this knob is a dataclass field with NO from_dims kwarg, so
     passing it to from_dims is silently swallowed -- verified, not assumed.
  3. env_drift_interval / env_drift_prob (hazard relocation). REJECTED as a
     MEASURED NULL: causal_grid_world.py's own SD-MEL-PRODUCER note records
     that "the optimal prediction of a random walk is its mean", and
     V3-EXQ-677's env_drift_interval 999 -> 3 manipulation produced a
     high-vs-low mean-PE difference of 8.8e-07 against a 0.01 threshold.

ADOPTED: world_rule_shift (SD-MEL-PRODUCER), the lever built because drift
failed -- but ENABLED ONLY AT THE P1 -> P2 BOUNDARY (see THE FIX above), which
is the whole difference between this run and 1062. It re-permutes the action
-> displacement map, so every ACTION-CONDITIONED forward model --
E2HarmAForward included -- becomes systematically wrong and stays wrong until
re-learned. That raises the prediction RESIDUAL while harm EXPOSURE stays
matched (matched-ness is a measured precondition here, never an assumption).
V3-EXQ-861e confirmed the IV is non-degenerate on this substrate: its
ecological_novelty_mel_gradient_present_this_config precondition and its
C1_measured_mel_gradient_present / C1_dv_spread_nonzero non-degeneracy flags
were all met (861e's FAIL was a downstream MEL-coupling DV, not the IV).
861e's DV is a MEL gradient, NOT the dACC harm-PE this run gates on, so it
establishes that the IV moves SOMETHING and nothing more -- see EXPLICITLY IN
SCOPE above.

  ARM_0_STATIONARY  world_rule_shift never enabled          (reference)
  ARM_2_HIGH_SHIFT  enabled at the P2 boundary: interval 10, depth 2

TWO ARMS, and no dose ladder. A same-schedule multi-rung ladder is the
prescription this run's own autopsy WITHDREW (section 5c): it inherits the
confound at every rung and aliases H1/H2/H3 to one flat-residual verdict, at
~9 h of compute, returning the same unevaluable C1. The autopsy's ordering is
followed exactly -- (1) fix the onset, (2) instrument H3, (3) ladder the dose
ONLY IF (1)+(2) show the residual can move at all. Step (3) is deliberately
not taken here.

GOV-FANOUT-1: no portfolio, and the exemption is claimed on the RE-POSED
design. The open question at this node is not "which of several hypotheses
holds" requiring parallel legs -- it is a single, unambiguous DESIGN FIX
(onset) plus a MEASUREMENT instrument (H3) that discriminates the one
hypothesis a single leg could not otherwise separate. Flagged as a judgement
call, inherited from the autopsy's own section 9.

Because both arms now train with the lever disabled, their P0+P1 phases are
bit-identical at a given seed (the enabled guard consumes no randomness), so
the arms enter P2 from the same trained state, and TRAINING is controlled
exactly.

CAVEAT, stated rather than glossed (Step 4.5 red-team F3): the P2 contrast is
NOT "the onset and nothing else" for the whole window. _maybe_shift_world_rule
draws its permutation from self._rng (causal_grid_world.py:2136) and reset()
draws episode layouts from the SAME generator (:1712), so from the first P2
shift onward the shift arm's episode LAYOUT sequence also diverges. That is
same-distribution sampling -- added VARIANCE, not bias -- but it lands on the
sparse-event exposure statistic, already the noisiest thing this run gates on.
A dedicated permutation RNG is the substrate-side fix and belongs with
SD-PP-B4, not here. Up to the first shift (P0, P1 and the first `interval`
steps of P2) the arms are bit-identical, which is what the measurement above
establishes.

The cells are nonetheless run INDEPENDENTLY, with a
full per-cell RNG reset, rather than training once and forking: independence
is what keeps each cell a pure function of (substrate, config, seed) and so
keeps the stationary arm's fingerprint reuse-ELIGIBLE for a later consumer. The
duplicated training is accepted deliberately for that reason.

env_drift is PINNED OFF in every arm (interval 999, prob 0.0, following 861e)
so the rule-shift ladder is the only world-nonstationarity that varies.

DV-SYMMETRY INVARIANCE -- one statement per arm (MANDATORY DECLARATION)
----------------------------------------------------------------------
Every DV here is a Spearman correlation, an OLS R^2, a lag-1 autocorrelation,
or a RELATIVE change -- computed over per-tick series. The symmetry group of
that family is: independent positive affine rescaling of either series, plus a
uniform additive constant on either series (correlations and R^2 are exactly
invariant under both; a relative change is invariant under rescaling only).

ARM_0_STATIONARY and ARM_2_HIGH_SHIFT -- the same statement holds for both,
because they differ only in whether (and from when) one manipulation runs:
the manipulation is a re-permutation of the action -> displacement map, which
REORDERS and RE-CONTENTS the temporal sequence of z_harm_a and destroys its
action-conditioned predictability. That is neither an affine rescaling nor an
added constant of any measured series, so NO arm's DV is invariant under its
own manipulation. None of the three arms is disposition-(b) vacuous.

Two consequences of that group were designed around rather than discovered:
  * A broadcast additive constant WOULD cancel in these DVs. That is why C3
    compares RELATIVE changes of the two channels rather than raw magnitudes
    (a magnitude readout survives a broadcast constant and would give a false
    positive -- the V3-EXQ-604c shape).
  * A pooled-over-ticks correlation is a set-aggregate and IS invariant under a
    PERMUTATION of ticks. C4 (lag-1 autocorrelation) is included precisely as
    the order-SENSITIVE companion statistic, so the criterion set is not
    uniformly exchangeable.

Readiness preconditions certify these channels and no others: the harm-forward
r2 precondition certifies axis 3 ONLY; the vh-range precondition certifies the
VALENCE_HARM_DISCRIMINATIVE series ONLY; the temperature-range precondition
certifies axis 1 ONLY. None speaks for the others.

AXIS 2 IS NOT THE HARM LEVEL (the one substantive fix beyond the onset)
-----------------------------------------------------------------------
V3-EXQ-1062 defined axis 2 as the spread across ALL SIX valence components at
the realized z_world. That statistic is the harm LEVEL exactly, not a valence
axis. Components 4 and 5 (positive/negative surprise) are written only when
use_mech307_split_surprise=True (residue/field.py:52-57), which no arm sets, so
the component MINIMUM is pinned at 0 and max-min collapses to max -- which
VALENCE_HARM_DISCRIMINATIVE dominates by ~300x.

Measured on this exact config, 2026-09-22, 150 steps with the write paths
driven: wanting 0, liking 0.169, harm_disc 110.628, surprise 0.331, pos/neg
surprise 0 -> full spread 110.62825775 == harm_disc 110.62825775, bit-identical.
And in 1062's own manifest, mean_axis2_valence_spread_{level,delta} equals
mean_valence_harm_{level,delta} in ALL SIX cells.

Consequence for the predecessor, which governance should read: 1062's C2 pair
(axis2, axis3) was arithmetically C1's pair, and no appetitive axis entered any
routed series. Its C2 PASS (max pairwise axis R^2 0.271) is a real measurement,
but of {tone, harm level, harm PE} -- NOT of {tone, valence, harm PE} as its
maps_to "MECH-055 FALSIFYING (i)" implies. A reader would have attributed it to
valence. This is raised to governance as an evidence_discrepancy rather than
silently corrected here.

Fix in this driver: axis 2 is the spread over the NON-HARM components only, so
it is a genuinely independent third series. It is alive on this substrate
(range 0.331 in the same probe, carried by surprise and liking), so this
substitutes a live axis for a duplicated one rather than a dead axis for a live
one. 1062's statistic is still recorded as
mean_axis2_spread_incl_harm_level_unrouted, beside the explicit identity flags
axis2_incl_harm_equals_valence_harm_level and
axis2_nonharm_differs_from_valence_harm_level, so the defect is visible in this
run's own manifest instead of asserted in prose. NO THRESHOLD MOVED: R2_MAX is
still 0.80. Note that the appetitive half of axis 2 is thin by design --
wanting reads exactly 0, which is MECH-054's architecturally-absent benefit
side showing up in this run's own instruments -- so if the axis-2 range goes
degenerate the existing criteria_non_degenerate["C2_axes_not_fixed_ratio"]
flags C2 vacuous WITHOUT vacating C1 or any arm. That is the right granularity
and it needed no new gate.

THE ACCUMULATION TRAP, AND WHY THE CRITERIA ROUTE ON INCREMENTS
--------------------------------------------------------------
ResidueField.update_valence "does NOT replace the existing value -- adds to it
so the vector accumulates across visits" (residue/field.py:294), and every
write path used here contributes a non-negative value. So the residue-field
reads are monotone non-decreasing RAMPS, not instantaneous signals: this
experiment's own Step 2.5a probe measured VALENCE_HARM_DISCRIMINATIVE going
0 -> 460.4 over 592 fresh ticks. The dACC channel is the opposite -- an
instantaneous forward-model residual that fluctuates.

Correlating a ramp against a fluctuating residual would make C1 pass (a ramp
and noise are nearly uncorrelated) and C4 pass (a ramp's lag-1 autocorrelation
is ~1, a residual's is low) for reasons that have NOTHING to do with channel
separation. That is a vacuous pass, and it would be invisible in the manifest:
the numbers would look like a clean confirmation of MECH-055.

So every criterion routes on INSTANTANEOUS quantities: the per-tick INCREMENT
of the two accumulating channels (the quantity their level integrates),
alongside the already-instantaneous temperature and harm-PE, all aligned to the
same ticks. The accumulating LEVELS are still recorded --
mean/final_valence_harm_level, and deliberately
abs_spearman_valence_harm_LEVEL_vs_dacc_pe_unrouted, the same statistic C1 uses
computed the WRONG way -- so a reader can SEE the size of the artifact the
differencing removes instead of taking it on trust. Nothing routes on them.

WHY AXIS 1's ARITHMETIC IDENTITY IS NOT A PROBLEM HERE
-----------------------------------------------------
V3-EXQ-799 warns that with mu injected into the softmax temperature, "entropy
falls as mu rises" is an arithmetic identity of the coupling. This run never
asserts that relation. It uses the axis-1 VALUE (effective_temperature, driven
upstream by PCCAnalog; pcc_stability is never written directly) purely as one
of three series in the lockstep test C2. Nothing arithmetically links the
softmax temperature to the residue-field valence store or to the harm-forward
residual, so C2 is a measurement rather than an identity. The PCC weights are
re-scoped exactly as 799 did (fatigue 0.15 / offline 0.35 / window 200)
because 799 measured that the substrate DEFAULTS pin mu at ~0.017 against a
[0,1] clip -- a structurally unsatisfiable readout. The thresholds below are
NOT relaxed to compensate.

SLEEP DRIVER: none (no sleep flags set; use_sleep_loop / sws / rem all off)

Red-team (Step 4.5): see RED_TEAM_VERDICT below and the manifest's
red_team_note. V3-EXQ-1062's own Step 4.5 pass (model: fable, CONTESTED, 7
findings) is INHERITED WHOLE by this driver -- every one of its fixes is
carried forward unchanged, and the three that changed what a run can conclude
are load-bearing here too: the verdict-grid decidability test that stops a
readiness failure being recorded as "FALSIFIER (ii) FIRED" / weakens (it is
what correctly routed 1062 and is what this driver extends with the H1/H2/H3
attribution), the prev_action causal pairing at both the P1-training and
P2-measurement sites, and the episode-first-tick level-PE exclusion. Their
dispositions are preserved verbatim in red_team_note.inherited_from_1062.

Claim: MECH-055 (affect.channel_separation)
Backlog: EVB-1413 (experimental twin; proposal_id EXP-0802 at authoring time,
         positional and NOT to be trusted across a governance regen)
Supersedes: V3-EXQ-1062
"""

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.residue.field import (
    VALENCE_DIM,
    VALENCE_HARM_DISCRIMINATIVE,
)
from ree_core.utils.config import REEConfig

from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest, flat_readout
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.fresh_select import FreshSelectCounter, FreshSelectProbe
from experiments._lib.precondition_gate import (
    PreconditionSpec,
    aggregate_arm_gates,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator

EXPERIMENT_TYPE = "v3_exq_1062a_mech055_affect_channel_separation_postshift"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS = ["MECH-055"]
QUEUE_ID = "V3-EXQ-1062a"
BACKLOG_ID = "EVB-1413"
SUPERSEDES = "V3-EXQ-1062"

FRESH_SELECT_NAMESPACE = "exq1062a_mech055"

RED_TEAM_VERDICT = (
    "CONTESTED (Step 4.5, one pass, model: fable) -- 6 findings, every one verified\n"
    "against source or the 1062 manifest before acting; 5 fixed in this driver and\n"
    "1 (F3) recorded as a stated caveat with a substrate-side owner. Dispositions in\n"
    "the manifest red_team_note."
)

# --------------------------------------------------------------------------- #
# Pre-registered acceptance thresholds. Constants -- never derived from the run.
# --------------------------------------------------------------------------- #
# C1: |Spearman(VALENCE_HARM_DISCRIMINATIVE, dacc_pe)| ceiling. 0.90 is NOT
# invented here: it is SD-014's own pre-registered channel-redundancy bar
# (RHO_MAX in V3-EXQ-887/887a/887b, where |Spearman(wanting, liking)| had to
# clear <= 0.90). Same substrate, same residue-field store, same question
# shape -- "are these two channels numerically redundant".
RHO_MAX = 0.90
# C2: ceiling on the OLS R^2 between any two standardized axis series. R^2 near
# 1 is what "move in a fixed, predictable ratio" means operationally.
R2_MAX = 0.80
# C3: the PE channel's relative rise across the shift ladder must exceed the
# level channel's by this margin (both as relative change, so scale-free).
REL_CHANGE_DELTA_MIN = 0.20
# C4: the two harm representations' lag-1 autocorrelations must differ by this
# much. An RBF-smoothed EMA level and a forward-model residual have different
# temporal signatures; one collapsed scalar wearing two labels cannot.
LAG1_DELTA_MIN = 0.10
# Per-arm seed quorum for every criterion.
SEED_QUORUM = 2

# --------------------------------------------------------------------------- #
# Readiness floors (substrate-not-ready, NOT claim verdicts).
# --------------------------------------------------------------------------- #
# Axis 3 must be a genuine forward-model PE. Below this the dACC "PE" degenerates
# to ||z_harm_a|| (dacc._affective_pe's own z_harm_a_pred-is-None branch) and
# axis 3 does not exist as a distinct channel at all.
FORWARD_R2_MIN = 0.30
# Each series must actually vary, or its correlation/R^2 is undefined rather
# than low. These are floors on the statistic the criteria route on (RANGE of
# the series), not on a magnitude proxy for it.
VH_RANGE_MIN = 1e-4
TEMP_RANGE_MIN = 1e-6
PE_RANGE_MIN = 1e-4
# The level channel is only a matched control if harm exposure really is
# matched across arms. CEILING on each arm's relative deviation from the
# pooled mean.
HARM_EXPOSURE_REL_DEV_MAX = 0.25
# The IV must actually have moved the PE in the shift arms (scoped OUT of the
# stationary arm, which IS the reference).
PE_ELEVATION_MIN = 0.05
# Sample floor for the per-tick statistics, counted in FRESH E3 selections.
FRESH_TICKS_MIN = 200
CORR_MIN_N = 100

# --------------------------------------------------------------------------- #
# RECORDED, NON-GATING instruments (H3 attribution + the exposure/PE window).  #
# None of these is a threshold. They are read ONLY by the decidability branch  #
# of the verdict grid, to attribute a readiness failure to H1 / H2 / H3.       #
# --------------------------------------------------------------------------- #
# Below this forward-vs-persistence skill the harm-forward model is not beating
# the trivial z_pred = z_harm_a(t-1) predictor, i.e. H3 (persistence dominance).
# NOT a gate: it labels an attribution, it never vacates an arm.
PERSISTENCE_SKILL_ATTRIBUTION_FLOOR = 0.05
# Below this normalised action-sensitivity the trained model effectively ignores
# the action, so an action-map re-permutation cannot raise its residual at ANY
# dose. NOT a gate, for the same reason.
ACTION_SENSITIVITY_ATTRIBUTION_FLOOR = 0.05
# First N P2 env steps after onset, over which the NON-GATING early-window copies
# of the two cross-arm preconditions are computed. Exposure deviation accumulates
# with time while PE elevation is immediate, so early and full can disagree --
# and if this run vacates on exposure, this is the number a successor needs.
EARLY_WINDOW_STEPS = 300
# The --dry-run P2 budget. Named because the sample floors are scaled to it (see
# _active_sample_floors) so the smoke actually exercises the scoring path instead
# of routing substrate_not_ready_requeue bit-identically to a real failure --
# V3-EXQ-1062 autopsy section 6, instrument finding 2.
DRY_P2_BUDGET = 60

# Mutable ONLY under --dry-run (see _active_sample_floors). A real run always
# uses the pre-registered CORR_MIN_N above; nothing else may write this.
_CORR_MIN_N_ACTIVE = CORR_MIN_N


def _active_sample_floors(dry_run: bool):
    """Return (fresh_ticks_min, corr_min_n) for this invocation.

    A REAL run returns the pre-registered constants unchanged -- no threshold
    is ever relaxed for a scored run.

    Under --dry-run ONLY, both floors are scaled by the reduced P2 budget
    (DRY_P2_BUDGET / P2_STEP_BUDGET). V3-EXQ-1062's smoke was structurally
    incapable of warning: its dry-run P2 budget of 60 steps is below
    FRESH_TICKS_MIN 200, so every arm was guaranteed red, and CORR_MIN_N 100
    forced every routed statistic to NaN -- the smoke emitted
    `outcome=FAIL label=substrate_not_ready_requeue`, bit-identical to what the
    real run later produced, while carrying zero information about whether the
    real gate would pass (autopsy section 6, finding 2). Scaling the floors with
    the budget is what makes the smoke a test of the scoring path.
    """
    if not dry_run:
        return FRESH_TICKS_MIN, CORR_MIN_N
    scale = float(DRY_P2_BUDGET) / float(P2_STEP_BUDGET)
    return (max(20, int(FRESH_TICKS_MIN * scale)),
            max(10, int(CORR_MIN_N * scale)))

# --------------------------------------------------------------------------- #
# Schedule. Phased training is MANDATORY: E2HarmAForward trains on z_harm_a,
# an encoder output, so P0 warms the encoder with no downstream loss, P1
# freezes it and trains the forward model on .detach()ed latents, P2 measures
# with no training at all.
# --------------------------------------------------------------------------- #
# Sized from the Step 2.5a measured cost (~0.84 s CPU per env step on this
# config) against the fleet's observed ~0.4 s/step for comparable heavy runs
# (V3-EXQ-1050: 6 cells, 330 estimated minutes). P1 gets twice P0's budget
# because P1 is what the harm-forward model -- the axis-3 readiness gate, and
# therefore the existence of axis 3 at all -- actually needs.
P0_EPS = 30
P1_EPS = 60
TOTAL_TRAINING_EPS = P0_EPS + P1_EPS      # the [train] ep N/M denominator
STEPS_PER_EPISODE = 90
# P2 is a FIXED STEP BUDGET, not an episode count. Episodes end early in the
# harsher shift arms, so an episode-count loop would give the arms different
# step counts and the per-tick series would differ in LENGTH mechanically --
# manufacturing a difference with no bearing on channel separation. (The same
# reasoning V3-EXQ-799 applied to visitation entropy.)
# 1800 steps at the MEASURED fresh-select yield (592/600 = 98.7% once the
# valence write paths are driven) projects to ~1780 fresh selections per cell,
# far above FRESH_TICKS_MIN and CORR_MIN_N.
P2_STEP_BUDGET = 1800
EPSILON_TRAIN = 0.1
EPSILON_EVAL = 0.0

SEEDS = [42, 137, 2026]

ARMS: List[Dict[str, Any]] = [
    {"arm_id": "ARM_0_STATIONARY", "interval": 0, "is_shift": False},
    {"arm_id": "ARM_2_HIGH_SHIFT", "interval": 10, "is_shift": True},
]
STATIONARY_ARM = "ARM_0_STATIONARY"
HIGH_ARM = "ARM_2_HIGH_SHIFT"
WORLD_RULE_SHIFT_DEPTH = 2

# Substrate dims / indices.
WORLD_DIM = 32
SELF_DIM = 32
HARM_A_DIM = 16
IDX_HARM_EXPOSURE = 10
IDX_BENEFIT_EXPOSURE = 11

# Env config. 887b's validated valence-population env (hazard_harm 0.5,
# resource_benefit 0.3, num_resources 6) -- deliberately NOT a weak-signal
# config, which would empty the LIKING/HARM channels outright (887b's own note
# on the V3-EXQ-432 vacuous-zero failure mode). harm_history_len=10 gives the
# affective stream its temporal integration input. env_drift PINNED OFF.
ENV_KWARGS = dict(
    size=10,
    num_hazards=3,
    num_resources=6,
    hazard_harm=0.5,
    resource_benefit=0.3,
    use_proxy_fields=True,
    harm_history_len=10,
    env_drift_interval=999,
    env_drift_prob=0.0,
)

# Substrate gate: what counts as consummatory contact. Calibrated to the
# measured benefit_exposure distribution on this env (887b measured max ~0.059,
# so the substrate default 0.1 is structurally unreachable). NOT an acceptance
# threshold.
LIKING_THRESHOLD = 0.02

# SD-014 incentive-sensitization at 887b's PASSING gain. 887b (2026-08-08)
# PASSED with rate 0.10 / max 8.0 / coupling 2.5 where 887a FAILED at the
# substrate defaults; that is what makes the valence store non-degenerate
# rather than wanting/liking-collinear.
SENSITIZATION_RATE = 0.10
SENSITIZATION_MAX = 8.0
SENSITIZATION_COUPLING = 2.5

# dACC. The bias multipliers must be non-zero or DACCtoE3Adapter returns the
# zero vector regardless of bundle content (its own docstring) -- and
# dacc_foraging_weight is DELIBERATELY 0.0 because foraging_value is a
# BROADCAST SCALAR, the exact V3-EXQ-604c DV-symmetry vacuity shape. Values
# follow 799.
DACC_WEIGHT = 1.0
DACC_INTERACTION_WEIGHT = 0.5
DACC_BIAS_MAX_ABS = 1.0
DACC_PRECISION_SCALE = 5000.0

# PCC re-scoping, verbatim from 799: the substrate defaults pin mu at ~0.017.
PCC_FATIGUE_WEIGHT = 0.15
PCC_OFFLINE_WEIGHT = 0.35
PCC_OFFLINE_RECENCY_WINDOW = 200

E2_HARM_A_LR = 5e-4

AXIS_NAMES = ["axis1_temperature", "axis2_nonharm_valence_spread", "axis3_harm_pe"]

_ZG = ZGoalStreamAccumulator()


# --------------------------------------------------------------------------- #
# statistics                                                                  #
# --------------------------------------------------------------------------- #

def _rank(xs: List[float]) -> List[float]:
    """Average-tie ranks."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < _CORR_MIN_N_ACTIVE or n != len(ys):
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx < 1e-12 or dy < 1e-12:
        return float("nan")
    return num / (dx * dy)


def _spearman(xs: List[float], ys: List[float]) -> float:
    if len(xs) < _CORR_MIN_N_ACTIVE or len(xs) != len(ys):
        return float("nan")
    return _pearson(_rank(xs), _rank(ys))


def _r2(xs: List[float], ys: List[float]) -> float:
    """OLS R^2 between two series. Equal to pearson^2 for a simple fit."""
    r = _pearson(xs, ys)
    return float("nan") if math.isnan(r) else r * r


def _lag1_autocorr(xs: List[float]) -> float:
    if len(xs) < _CORR_MIN_N_ACTIVE + 1:
        return float("nan")
    return _pearson(xs[:-1], xs[1:])


def _series_range(xs: List[float]) -> float:
    finite = [x for x in xs if math.isfinite(x)]
    if not finite:
        return 0.0
    return float(max(finite) - min(finite))


def _mean(xs: List[float]) -> float:
    finite = [x for x in xs if math.isfinite(x)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


# Indices of every VALENCE component EXCEPT the harm-discriminative one. Axis 2
# is the VALENCE axis, and it must not be the harm LEVEL wearing a second label
# -- see AXIS 2 IS NOT THE HARM LEVEL in the module docstring.
_NONHARM_VALENCE_IDX = [i for i in range(VALENCE_DIM)
                        if i != VALENCE_HARM_DISCRIMINATIVE]


def _nonharm_valence_spread(v: "torch.Tensor") -> float:
    """Cross-component range of the valence vector EXCLUDING harm_discriminative."""
    sub = v[_NONHARM_VALENCE_IDX]
    return float((sub.max() - sub.min()).item())


def _rel_change(new: float, ref: float) -> float:
    """Relative change, scale-free. Guarded denominator."""
    if not (math.isfinite(new) and math.isfinite(ref)):
        return float("nan")
    return (new - ref) / (abs(ref) + 1e-9)


def _worst_cell(rows: List[Dict], key: str, mode: str) -> Tuple[float, str]:
    """Return (extremum, offending_cell_id). mode 'min' for floors, 'max' for ceilings."""
    vals = [(r[key], r["cell_id"]) for r in rows if math.isfinite(r.get(key, float("nan")))]
    if not vals:
        return float("nan"), "(no finite cell)"
    pick = min(vals, key=lambda t: t[0]) if mode == "min" else max(vals, key=lambda t: t[0])
    return float(pick[0]), str(pick[1])


# --------------------------------------------------------------------------- #
# setup                                                                       #
# --------------------------------------------------------------------------- #

def _make_env(seed: int) -> CausalGridWorldV2:
    """Construct the env with world_rule_shift DISABLED, in EVERY arm.

    This is the onset fix. V3-EXQ-1062 set world_rule_shift_enabled here, at
    CONSTRUCTION, so the shift ran from step 0 of P0 and the harm-forward model
    trained under ~810 accumulated permutations -- the regime it was then
    measured in. The shift arm enables the lever at the P1 -> P2 boundary
    instead (see _enable_post_training_shift), which is the only supported way
    to reach a TRAINED-THEN-SHIFTED regime today: the counter deliberately
    survives reset() (causal_grid_world.py:984-985, :1879-1885) and there is no
    onset kwarg (substrate_queue SD-PP-B4-one-shot-world-rule-shift-lever).

    Because every RNG draw in _maybe_shift_world_rule is inside the enabled
    guard, a disabled env consumes no randomness, so the two arms' P0+P1 are
    bit-identical at a given seed.
    """
    kw = dict(ENV_KWARGS)
    kw.update(
        world_rule_shift_enabled=False,
        world_rule_shift_interval=0,
        world_rule_shift_depth=0,
    )
    return CausalGridWorldV2(seed=seed, **kw)


def _enable_post_training_shift(env: CausalGridWorldV2, interval: int) -> None:
    """Start the world-rule shift AT THE P2 BOUNDARY, after training.

    The sanctioned workaround named by the V3-EXQ-1062 autopsy (section 9,
    step 1): these three are public attributes and need no substrate change.
    Probed 2026-09-22 -- the first shift then fires at world step `interval`
    and every `interval` steps after.
    """
    env.world_rule_shift_enabled = True
    env.world_rule_shift_interval = int(interval)
    env.world_rule_shift_depth = WORLD_RULE_SHIFT_DEPTH


def _counterfactual_action(action: torch.Tensor,
                           map_keys: List[int],
                           action_dim: int,
                           device) -> torch.Tensor:
    """A DIFFERENT action, drawn deterministically from the env's own map keys.

    Used only by the RECORDED, NON-GATING action-sensitivity instrument. Drawn
    from _action_map keys (never the non-map action the env documents at
    causal_grid_world.py:253), because only map members can be permuted and so
    only they can carry the manipulation this experiment depends on.
    """
    taken = int(action.argmax(dim=-1).item())
    if taken in map_keys:
        nxt = map_keys[(map_keys.index(taken) + 1) % len(map_keys)]
    else:
        nxt = map_keys[0]
    cf = torch.zeros(1, action_dim, device=device)
    cf[0, nxt] = 1.0
    return cf


def make_config(env: CausalGridWorldV2) -> REEConfig:
    """Identical in every arm. The ONLY cross-arm difference is the ENV's
    world_rule_shift rate -- no agent-side flag varies, so an arm difference
    cannot be an agent-config artifact."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        # SD-011 dual nociceptive streams: z_harm_s and z_harm_a, the two
        # upstream sources whose downstream readouts C1 compares.
        use_harm_stream=True,
        harm_obs_dim=51,
        use_affective_harm_stream=True,
        harm_obs_a_dim=50,
        z_harm_a_dim=HARM_A_DIM,
        harm_history_len=10,
        # Axis 3: MECH-258 precision-weighted affective-pain PE.
        use_e2_harm_a=True,
        use_shared_harm_trunk=False,
        e2_harm_a_lr=E2_HARM_A_LR,
        use_dacc=True,
        dacc_weight=DACC_WEIGHT,
        dacc_interaction_weight=DACC_INTERACTION_WEIGHT,
        dacc_foraging_weight=0.0,
        dacc_bias_max_abs=DACC_BIAS_MAX_ABS,
        dacc_precision_scale=DACC_PRECISION_SCALE,
        dacc_effort_cost=0.1,
        dacc_drive_coupling=0.0,
        # Axis 1: SD-032d mu/kappa -> mode-prior softmax temperature.
        use_salience_coordinator=True,
        use_pcc_analog=True,
        salience_apply_to_dacc_bias=True,
        salience_use_stability_temperature=True,
        salience_temperature_mu_alpha=1.0,
        salience_temperature_kappa_alpha=0.0,
        salience_temperature_exponent_clip=4.0,
        pcc_fatigue_weight=PCC_FATIGUE_WEIGHT,
        pcc_offline_weight=PCC_OFFLINE_WEIGHT,
        pcc_offline_recency_window=PCC_OFFLINE_RECENCY_WINDOW,
        # Axis 2: SD-014 valence vector, at 887b's passing sensitization gain.
        z_goal_enabled=True,
        drive_weight=2.0,
        benefit_eval_enabled=True,
        goal_weight=1.0,
        tonic_5ht_enabled=True,
        valence_harm_enabled=True,
        valence_liking_enabled=True,
        liking_threshold=LIKING_THRESHOLD,
        surprise_gated_replay=True,
        pe_surprise_threshold=0.001,
        incentive_sensitization_enabled=True,
        sensitization_rate=SENSITIZATION_RATE,
        sensitization_max=SENSITIZATION_MAX,
        sensitization_coupling=SENSITIZATION_COUPLING,
    )
    cfg.latent.use_resource_encoder = True
    return cfg


def _obs_tensors(obs_dict) -> Tuple[torch.Tensor, ...]:
    body = obs_dict["body_state"].float().unsqueeze(0)
    world = obs_dict["world_state"].float().unsqueeze(0)
    harm = obs_dict["harm_obs"].float().unsqueeze(0) if "harm_obs" in obs_dict else None
    harm_a = obs_dict["harm_obs_a"].float().unsqueeze(0) if "harm_obs_a" in obs_dict else None
    hh = obs_dict["harm_history"].float().unsqueeze(0) if "harm_history" in obs_dict else None
    return body, world, harm, harm_a, hh


def _drive_valence_write_paths(agent, body: torch.Tensor) -> Tuple[float, float]:
    """Drive the SD-014 valence write paths (887b's sequence). Returns
    (benefit_exposure, harm_exposure). Without these the residue field never
    allocates an active RBF center and evaluate_valence returns all zeros --
    confirmed by this experiment's own Step 2.5a probe."""
    be = float(body[0, IDX_BENEFIT_EXPOSURE])
    he = float(body[0, IDX_HARM_EXPOSURE])
    drive = agent.compute_drive_level(body)
    agent.serotonin_step(be)
    agent.update_z_goal(be, drive)
    agent.update_benefit_salience(be, drive)   # -> VALENCE_WANTING
    agent.update_harm_salience(he)             # -> VALENCE_HARM_DISCRIMINATIVE
    agent.update_liking(be)                    # -> VALENCE_LIKING
    return be, he


def _cross_candidate_valence_range(agent, candidates, index: int) -> float:
    """Max over VALENCE components of the cross-candidate range of
    evaluate_valence at the candidates' world_states[index].

    index=-1 is the POST-ACTION terminus; index=0 is the rollout's SHARED
    initial z_world seed, which E2FastPredictor.rollout_with_world makes
    bit-identical across every candidate by construction (config.py's
    candidate_summary_source note; measured 2.8e6-4.5e6 magnitude ratio in
    V3-EXQ-822c). So index=0 is recorded as a NEGATIVE CONTROL that must read
    ~0, and index=-1 is the informative read. Returns nan when unavailable."""
    zs = []
    for c in candidates:
        ws = getattr(c, "world_states", None)
        if ws:
            zs.append(ws[index].detach().reshape(1, -1))
    if len(zs) < 2:
        return float("nan")
    batch = torch.cat(zs, dim=0)
    vv = agent.residue_field.evaluate_valence(batch)
    if vv.dim() != 2 or vv.shape[0] != batch.shape[0]:
        return float("nan")
    return float((vv.max(dim=0).values - vv.min(dim=0).values).max().item())


# --------------------------------------------------------------------------- #
# one (arm x seed) cell                                                       #
# --------------------------------------------------------------------------- #

def _config_slice(interval: int) -> Dict[str, Any]:
    """What the cell's computation reads. Declared for the arm fingerprint."""
    return {
        "env": dict(ENV_KWARGS),
        "world_rule_shift_interval": interval,
        "world_rule_shift_depth": WORLD_RULE_SHIFT_DEPTH if interval > 0 else 0,
        # The ONSET is the single thing that differs from V3-EXQ-1062 at the
        # same interval/depth, so it MUST be in the fingerprint: without it a
        # 1062 cell and a 1062a cell would be content-addressed identically and
        # a future consumer could false-HIT a mint from the confounded run.
        "world_rule_shift_onset": "p2_boundary_post_training",
        "early_window_steps": EARLY_WINDOW_STEPS,
        "schedule": {
            "p0": P0_EPS, "p1": P1_EPS, "steps": STEPS_PER_EPISODE,
            "p2_step_budget": P2_STEP_BUDGET,
            "epsilon_train": EPSILON_TRAIN, "epsilon_eval": EPSILON_EVAL,
        },
        "substrate": {
            "world_dim": WORLD_DIM, "self_dim": SELF_DIM, "z_harm_a_dim": HARM_A_DIM,
            "dacc_weight": DACC_WEIGHT,
            "dacc_interaction_weight": DACC_INTERACTION_WEIGHT,
            "dacc_bias_max_abs": DACC_BIAS_MAX_ABS,
            "dacc_precision_scale": DACC_PRECISION_SCALE,
            "pcc_fatigue_weight": PCC_FATIGUE_WEIGHT,
            "pcc_offline_weight": PCC_OFFLINE_WEIGHT,
            "pcc_offline_recency_window": PCC_OFFLINE_RECENCY_WINDOW,
            "sensitization_rate": SENSITIZATION_RATE,
            "sensitization_max": SENSITIZATION_MAX,
            "sensitization_coupling": SENSITIZATION_COUPLING,
            "liking_threshold": LIKING_THRESHOLD,
            "e2_harm_a_lr": E2_HARM_A_LR,
        },
        # Readout-affecting constants, declared so a cross-driver consumer with
        # different values MISSES this mint rather than falsely HITting it
        # (arm_reuse_fingerprint_plan.md 7b; confirmed instance V3-EXQ-798).
        # CORR_MIN_N gates every correlation estimator in this cell -- a
        # different value changes which cells return nan; the IDX_* constants
        # select which body_obs slots drive the valence write paths and the
        # harm-exposure series, so a different indexing scheme silently
        # measures different quantities under the same fingerprint.
        "readout_constants": {
            "corr_min_n": CORR_MIN_N,
            "idx_harm_exposure": IDX_HARM_EXPOSURE,
            "idx_benefit_exposure": IDX_BENEFIT_EXPOSURE,
        },
    }


def _run_cell(arm: Dict[str, Any], seed: int, dry_run: bool) -> Dict[str, Any]:
    arm_id = arm["arm_id"]
    interval = int(arm["interval"])
    cell_id = f"{arm_id}/seed{seed}"
    print("Seed %d Condition %s" % (seed, arm_id), flush=True)

    p0_eps = 2 if dry_run else P0_EPS
    p1_eps = 2 if dry_run else P1_EPS
    steps_per_ep = 12 if dry_run else STEPS_PER_EPISODE
    p2_budget = DRY_P2_BUDGET if dry_run else P2_STEP_BUDGET
    total_training = p0_eps + p1_eps

    with arm_cell(
        seed,
        config_slice=_config_slice(interval),
        script_path=Path(__file__),
        config_slice_declared=True,
        include_driver_script_in_hash=False,
        # A --dry-run cell computes with 2/2/12/60 and a scaled corr floor while
        # _config_slice declares the PRE-REGISTERED schedule, so without this it
        # is content-addressed identically to a real cell. That was harmless
        # while dry-run statistics were all NaN; now that the floors are scaled
        # they are finite and plausible, which is exactly a false-HIT shape.
        # Step 4.5 red-team F5.
        extra_ineligible_reasons=(["dry_run"] if dry_run else None),
    ) as cell:
        random.seed(seed)
        env = _make_env(seed)
        agent = REEAgent(make_config(env))
        # Via the INSTANCE, not the name: CausalGridWorldV2 is a factory
        # FUNCTION in this module, not the class. ACTIONS is the class-level
        # canonical map, which the env never mutates (it copies it into
        # _action_map at construction precisely so a permutation cannot leak
        # into other env instances), so this stays canonical all run.
        canonical_action_map = dict(env.ACTIONS)
        map_keys = sorted(env._action_map.keys())

        e2a_opt = optim.Adam(agent.e2_harm_a.parameters(), lr=E2_HARM_A_LR)
        e1_opt = optim.Adam(list(agent.e1.parameters()), lr=1e-3)
        wf_opt = optim.Adam(
            list(agent.e2.world_transition.parameters())
            + list(agent.e2.world_action_encoder.parameters()),
            lr=1e-3,
        )

        probe = FreshSelectProbe(FRESH_SELECT_NAMESPACE)
        counter = FreshSelectCounter()

        wf_buf: List[Tuple] = []
        max_buf = 2000

        # --------------------------- P0 + P1 ------------------------------- #
        for ep in range(total_training):
            is_p0 = ep < p0_eps
            agent.reset()
            _obs, od = env.reset()
            prev_zha: Optional[torch.Tensor] = None
            prev_zw: Optional[torch.Tensor] = None
            prev_action: Optional[torch.Tensor] = None

            if (ep + 1) % 20 == 0 or ep == 0:
                print(
                    "  [train] %s seed=%d ep %d/%d" % (arm_id, seed, ep + 1, total_training),
                    flush=True,
                )

            for _step in range(steps_per_ep):
                body, world, harm, harm_a, hh = _obs_tensors(od)
                latent = agent.sense(
                    obs_body=body, obs_world=world, obs_harm=harm,
                    obs_harm_a=harm_a, obs_harm_history=hh,
                )
                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick")
                    else torch.zeros(1, WORLD_DIM, device=agent.device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)

                if EPSILON_TRAIN > 0.0 and random.random() < EPSILON_TRAIN:
                    ai = random.randint(0, env.action_dim - 1)
                    action = torch.zeros(1, env.action_dim, device=agent.device)
                    action[0, ai] = 1.0

                zw_curr = latent.z_world.detach()

                # P0: encoder warmup only -- E1 + world-forward. No downstream
                # head sees a gradient here (phased training, MANDATORY).
                if is_p0:
                    e1_loss = agent.compute_prediction_loss()
                    if e1_loss.requires_grad:
                        e1_opt.zero_grad()
                        e1_loss.backward()
                        torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                        e1_opt.step()
                    if prev_zw is not None and prev_action is not None:
                        wf_buf.append((prev_zw.cpu(), prev_action.cpu(), zw_curr.cpu()))
                        if len(wf_buf) > max_buf:
                            wf_buf = wf_buf[-max_buf:]
                    if len(wf_buf) >= 16:
                        k = min(32, len(wf_buf))
                        idx = torch.randperm(len(wf_buf))[:k].tolist()
                        zb = torch.cat([wf_buf[i][0] for i in idx]).to(agent.device)
                        ab = torch.cat([wf_buf[i][1] for i in idx]).to(agent.device)
                        zn = torch.cat([wf_buf[i][2] for i in idx]).to(agent.device)
                        wf_l = F.mse_loss(agent.e2.world_forward(zb, ab), zn)
                        if wf_l.requires_grad:
                            wf_opt.zero_grad()
                            wf_l.backward()
                            torch.nn.utils.clip_grad_norm_(
                                list(agent.e2.world_transition.parameters())
                                + list(agent.e2.world_action_encoder.parameters()), 1.0)
                            wf_opt.step()

                # P1: encoder FROZEN by .detach() on both sides; only the harm
                # forward model learns.
                #
                # THE ACTION IS prev_action, NOT action. The transition
                # z_harm_a(t-1) -> z_harm_a(t) was caused by the action executed
                # at t-1; pairing it with THIS tick's action trains a
                # non-causal map. It also has to match the pairing the agent's
                # own measured PE uses, or P1 trains a different model from the
                # one C1 reads: sense() caches _harm_a_prev = z(t)
                # (agent.py:5583), select_action rolls
                # pred = e2_harm_a(z(t), a(t)) (agent.py:10329), and the next
                # tick's dACC compares z(t+1) against it (agent.py:7743).
                # Since world_rule_shift acts ONLY through the action channel
                # (causal_grid_world.py re-permutes the action -> displacement
                # map), training against the wrong action would blunt exactly
                # the manipulation this experiment depends on.
                if ((not is_p0) and prev_zha is not None
                        and prev_action is not None and latent.z_harm_a is not None):
                    z_pred = agent.e2_harm_a(prev_zha.detach(), prev_action.detach())
                    loss = agent.e2_harm_a.compute_loss(z_pred, latent.z_harm_a.detach())
                    if loss.requires_grad:
                        e2a_opt.zero_grad()
                        loss.backward()
                        e2a_opt.step()

                _drive_valence_write_paths(agent, body)
                _obs, harm_signal, done, _info, od = env.step(
                    int(action.argmax(dim=-1).item()))
                agent.update_residue(float(harm_signal) if harm_signal is not None else 0.0)

                prev_zha = (latent.z_harm_a.detach().clone()
                            if latent.z_harm_a is not None else None)
                prev_zw = zw_curr
                prev_action = action.detach()
                if done:
                    break

        # ---------------------- THE ONSET, at the P1 -> P2 boundary --------- #
        # Training is over. ASSERT the world's causal structure is still the
        # canonical one -- i.e. that this run really did train on a stationary
        # world -- and only then start the shift. This is recorded per cell
        # rather than trusted: it is the single premise separating this run
        # from V3-EXQ-1062.
        action_map_canonical_at_p2_onset = (
            dict(env._action_map) == canonical_action_map)
        world_steps_total_at_p2_onset = int(env._world_steps_total)
        shift_count_at_p2_onset = int(env._world_rule_shift_count)
        if arm["is_shift"]:
            _enable_post_training_shift(env, interval)
        print("  [onset] %s seed=%d canonical_map=%s shifts_before_p2=%d enabled=%s"
              % (arm_id, seed, action_map_canonical_at_p2_onset,
                 shift_count_at_p2_onset, bool(arm["is_shift"])), flush=True)

        # ------------------------------ P2 --------------------------------- #
        # Measurement only. No optimiser is stepped. Fixed step budget.
        #
        # Series are collected PER EPISODE and differenced WITHIN an episode,
        # for two reasons found by the Step 4.5 red-team pass and confirmed in
        # ree_core:
        #   (a) agent.reset() clears _harm_a_pred_prev (agent.py:3630), so on
        #       the FIRST fresh tick of every episode the dACC receives
        #       z_harm_a_pred=None and dacc._affective_pe returns ||z_harm_a||
        #       -- a LEVEL, not a residual (dacc.py:213-214). Those ticks are
        #       EXCLUDED and counted, because a level is ~an order of magnitude
        #       above a residual and episode counts differ across arms (shift
        #       arms end episodes earlier), so including them would let
        #       "PE load elevated" and C3 pass on episode count alone.
        #   (b) differencing across an episode boundary would pair an increment
        #       spanning a reset with a post-reset instantaneous value.
        ep_series: List[Dict[str, List[float]]] = []
        hs_norm: List[float] = []
        ha_norm: List[float] = []
        prec: List[float] = []
        prec_norm_vals: List[float] = []
        # Recorded, NON-GATING cross-candidate valence range. Flat (not
        # per-episode / not differenced) because the consumers are plain means
        # of a LEVEL-family read -- the same treatment prec / hs_norm / ha_norm
        # get, not the _diff() treatment vh / a2 get.
        cc_post: List[float] = []
        cc_seed: List[float] = []
        harm_exposure: List[float] = []
        # RECORDED, NON-GATING early-window copies. Exposure deviation
        # accumulates with time under the shift while PE elevation is
        # immediate, so early and full can disagree -- and if this run vacates
        # on exposure, these are the numbers a successor needs in order to know
        # whether a shorter window would clear the pre-registered bar without
        # paying for another full run. They gate NOTHING.
        early_window = min(EARLY_WINDOW_STEPS, p2_budget)
        harm_exposure_early: List[float] = []
        pe_early: List[float] = []
        # (pred, target, prev_input) -- prev_input is what the PERSISTENCE
        # baseline predicts with, i.e. the delta == 0 model.
        r2_pairs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        # RECORDED, NON-GATING action sensitivity: how far the model's
        # prediction moves when only the ACTION is changed, against the size of
        # the model's own predicted delta.
        act_sens_num: List[float] = []
        act_sens_den: List[float] = []
        total_steps = 0
        n_episodes = 0
        n_level_pe_ticks_excluded = 0

        with torch.no_grad():
            while total_steps < p2_budget:
                agent.reset()
                _obs, od = env.reset()
                counter.flush()
                n_episodes += 1
                prev_zha = None
                prev_action = None
                cur: Dict[str, List[float]] = {"vh": [], "a2": [], "pe": [],
                                               "a1": [], "a2ih": []}
                while total_steps < p2_budget:
                    body, world, harm, harm_a, hh = _obs_tensors(od)
                    latent = agent.sense(
                        obs_body=body, obs_world=world, obs_harm=harm,
                        obs_harm_a=harm_a, obs_harm_history=hh,
                    )
                    ticks = agent.clock.advance()
                    e1_prior = (
                        agent._e1_tick(latent) if ticks.get("e1_tick")
                        else torch.zeros(1, WORLD_DIM, device=agent.device)
                    )
                    candidates = agent.generate_trajectories(latent, e1_prior, ticks)

                    # Read BEFORE select_action: this is what the dACC inside
                    # that call will receive as z_harm_a_pred. None => the pe it
                    # reports is a level, not a residual (see (a) above).
                    pred_absent = getattr(agent, "_harm_a_pred_prev", None) is None

                    # Sample-size integrity: E3 diagnostics LATCH between e3
                    # ticks (cadence heartbeat.e3_steps_per_tick, default 10),
                    # so a per-env-step read without this guard would
                    # pseudo-replicate ~10x. n_latched_ticks is emitted.
                    with probe.watch(agent) as fresh:
                        action = agent.select_action(candidates, ticks)
                    is_fresh = bool(fresh)
                    counter.record(is_fresh)

                    if is_fresh:
                        b = getattr(agent, "_dacc_last_bundle", None)
                        st = getattr(agent, "_salience_last_tick", None)
                        if isinstance(b, dict) and "pe" in b and isinstance(st, dict):
                            if pred_absent:
                                n_level_pe_ticks_excluded += 1
                            else:
                                v = agent.residue_field.evaluate_valence(
                                    latent.z_world).reshape(-1)
                                cur["pe"].append(float(b["pe"]))
                                if total_steps < early_window:
                                    pe_early.append(float(b["pe"]))
                                cur["a1"].append(
                                    float(st.get("effective_temperature", float("nan"))))
                                # AXIS 2 -- the VALENCE axis. Computed over the
                                # NON-HARM valence components only.
                                #
                                # V3-EXQ-1062 used the spread over ALL SIX
                                # components, and that statistic is the harm
                                # LEVEL exactly, not a valence axis: components
                                # 4/5 (positive/negative surprise) are written
                                # only when use_mech307_split_surprise=True
                                # (residue/field.py:52-57), which no arm here
                                # sets, so the component MINIMUM is pinned at 0
                                # and max-min collapses to max -- which
                                # harm_discriminative dominates. MEASURED on this
                                # exact config, 2026-09-22, 150 steps: wanting
                                # 0, liking 0.169, harm_disc 110.6, surprise
                                # 0.331, pos/neg surprise 0 -> full spread
                                # 110.62825775 == harm_disc 110.62825775,
                                # bit-identical. And in 1062's own manifest,
                                # mean_axis2_valence_spread_{level,delta} equals
                                # mean_valence_harm_{level,delta} in ALL SIX
                                # cells. So 1062's C2 pair (axis2, axis3) was
                                # arithmetically C1's pair, and no appetitive
                                # axis entered any routed series -- a result a
                                # reader would have attributed to valence.
                                # Found by this driver's Step 4.5 red-team (F1)
                                # and verified against source and that manifest
                                # before acting.
                                #
                                # The non-harm spread IS alive on this substrate
                                # (range 0.331 in the same probe, carried by
                                # surprise and liking), so this is a real series,
                                # not a dead axis substituted for a live one.
                                cur["a2"].append(_nonharm_valence_spread(v))
                                # RECORDED, NEVER ROUTED: 1062's statistic, kept
                                # so the identity above is visible in this run's
                                # own manifest rather than asserted.
                                cur["a2ih"].append(
                                    float((v.max() - v.min()).item()))
                                cur["vh"].append(
                                    float(v[VALENCE_HARM_DISCRIMINATIVE].item()))
                                # Recorded, NON-GATING. Sampled on exactly the
                                # ticks every other per-tick DV in this row is
                                # sampled on -- FRESH E3 selections with a real
                                # (non-level) PE -- so the recorded mean
                                # describes the same tick population the routed
                                # statistics do. The fresh gate is the
                                # pseudo-replication guard documented above; the
                                # recorded_non_gating_note's own Step 2.5a
                                # figure is stated "over 592 fresh ticks".
                                # index=-1 post-action terminus, index=0 shared
                                # seed -- see _cross_candidate_valence_range.
                                cc_post.append(
                                    _cross_candidate_valence_range(
                                        agent, candidates, -1))
                                cc_seed.append(
                                    _cross_candidate_valence_range(
                                        agent, candidates, 0))
                                _pv = float(getattr(agent.e3, "current_precision",
                                                    float("nan")))
                                prec.append(_pv)
                                # dacc._affective_pe applies
                                # prec_norm = min(precision / dacc_precision_scale, 3.0)
                                # and multiplies the residual by (1 + prec_norm).
                                # Recorded, not gated: every routed statistic here
                                # (Spearman, R^2, lag-1 autocorr, relative change) is
                                # invariant under a positive constant scaling, so
                                # neither a negligible nor a saturated precision leg
                                # can manufacture or destroy a verdict -- but both
                                # bound what a PASS may be said to have exercised.
                                if math.isfinite(_pv):
                                    prec_norm_vals.append(
                                        min(_pv / DACC_PRECISION_SCALE, 3.0))
                                if latent.z_harm is not None:
                                    hs_norm.append(float(latent.z_harm.norm().item()))
                                if latent.z_harm_a is not None:
                                    ha_norm.append(float(latent.z_harm_a.norm().item()))
                        # r2 on the SAME causal pairing the agent's own PE uses:
                        # sense() caches _harm_a_prev = z(t) (agent.py:5583) and
                        # select_action rolls e2_harm_a(z(t), a(t)) -> z(t+1)
                        # (agent.py:10329). So the predecessor ACTION is
                        # prev_action, never this tick's action.
                        if (prev_zha is not None and prev_action is not None
                                and latent.z_harm_a is not None):
                            z_pred = agent.e2_harm_a(prev_zha.detach(),
                                                     prev_action.detach())
                            r2_pairs.append((z_pred.detach().cpu(),
                                             latent.z_harm_a.detach().cpu(),
                                             prev_zha.detach().cpu()))
                            # H3 instrument, RECORDED and NON-GATING: hold
                            # z_harm_a(t-1) fixed and change ONLY the action.
                            # ResidualHarmForward returns z + delta(z, a), so
                            # this is the size of the action's contribution
                            # against the size of the whole predicted delta. At
                            # ~0 the trained model ignores the action, and no
                            # re-permutation of the action map can raise its
                            # residual at ANY dose.
                            a_cf = _counterfactual_action(
                                prev_action, map_keys, env.action_dim,
                                agent.device)
                            z_pred_cf = agent.e2_harm_a(prev_zha.detach(), a_cf)
                            act_sens_num.append(
                                float((z_pred - z_pred_cf).norm().item()))
                            act_sens_den.append(
                                float((z_pred - prev_zha).norm().item()))

                    harm_exposure.append(float(body[0, IDX_HARM_EXPOSURE]))
                    if total_steps < early_window:
                        harm_exposure_early.append(
                            float(body[0, IDX_HARM_EXPOSURE]))
                    _drive_valence_write_paths(agent, body)
                    _obs, harm_signal, done, _info, od = env.step(
                        int(action.argmax(dim=-1).item()))
                    agent.update_residue(
                        float(harm_signal) if harm_signal is not None else 0.0)
                    prev_zha = (latent.z_harm_a.detach().clone()
                                if latent.z_harm_a is not None else None)
                    prev_action = action.detach()
                    total_steps += 1
                    if done:
                        break
                ep_series.append(cur)
        counter.flush()
        _ZG.observe(agent)

        # Per-tick increments of the accumulating channels, differenced WITHIN
        # each episode and then concatenated. These -- not the levels -- are
        # what C1 / C2 / C3 / C4 route on (see the P2 header note).
        def _diff(xs: List[float]) -> List[float]:
            return [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]

        vh: List[float] = []
        a2: List[float] = []
        pe_al: List[float] = []
        a1_al: List[float] = []
        vh_level: List[float] = []
        a2_level: List[float] = []
        a2ih_level: List[float] = []
        pe: List[float] = []
        for ep in ep_series:
            vh_level.extend(ep["vh"])
            a2_level.extend(ep["a2"])
            a2ih_level.extend(ep["a2ih"])
            pe.extend(ep["pe"])
            if len(ep["vh"]) < 2:
                continue
            vh.extend(_diff(ep["vh"]))
            a2.extend(_diff(ep["a2"]))
            # Align the instantaneous series to the differenced ones by
            # dropping each episode's first sample, so every pairwise
            # statistic below is computed on the SAME ticks.
            pe_al.extend(ep["pe"][1:])
            a1_al.extend(ep["a1"][1:])


        # harm-forward r2, on the P2 pairs (597b's estimator), PLUS the H3
        # persistence baseline computed from the same pairs at zero extra env
        # cost. persistence is the delta == 0 predictor z_pred = z_harm_a(t-1):
        # ResidualHarmForward returns z + delta(z, a) (latent/stack.py), so a
        # model that drives delta toward zero scores well on an autocorrelated
        # signal while being action-blind -- which is exactly H3. A high
        # harm_a_forward_r2 therefore does NOT by itself show the model learned
        # anything action-conditioned; the SKILL against persistence does.
        if r2_pairs:
            preds = torch.cat([p for p, _, _ in r2_pairs])
            tgts = torch.cat([t for _, t, _ in r2_pairs])
            prevs = torch.cat([q for _, _, q in r2_pairs])
            ss_res = float(((tgts - preds) ** 2).sum())
            ss_per = float(((tgts - prevs) ** 2).sum())
            ss_tot = float(((tgts - tgts.mean(dim=0)) ** 2).sum())
            fwd_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-8 else float("nan")
            persistence_r2 = (1.0 - ss_per / ss_tot) if ss_tot > 1e-8 else float("nan")
            # Skill = fraction of the persistence predictor's squared error the
            # learned model removes. <= 0 means it is no better than persistence.
            fwd_skill = (1.0 - ss_res / ss_per) if ss_per > 1e-12 else float("nan")
        else:
            fwd_r2 = float("nan")
            persistence_r2 = float("nan")
            fwd_skill = float("nan")

        # Normalised action sensitivity (RECORDED, NON-GATING). Denominated on
        # the model's own predicted delta so the ratio is scale-free: a model
        # whose delta is large but action-invariant reads ~0 here, which is the
        # H3 signature, and is NOT distinguishable from it by fwd_r2 alone.
        _as_den = _mean(act_sens_den)
        act_sensitivity = (_mean(act_sens_num) / _as_den
                           if math.isfinite(_as_den) and _as_den > 1e-12
                           else float("nan"))

        row: Dict[str, Any] = {
            "cell_id": cell_id,
            "arm_id": arm_id,
            "seed": seed,
            "world_rule_shift_interval": interval,
            "is_shift_arm": bool(arm["is_shift"]),
            "n_fresh_select": counter.n_fresh_select,
            "n_latched_ticks": counter.n_latched,
            "n_p2_env_steps": total_steps,
            "n_p2_episodes": n_episodes,
            "n_level_pe_ticks_excluded": n_level_pe_ticks_excluded,
            "harm_a_forward_r2": fwd_r2,
            # --- onset provenance: the premise that separates this run from 1062
            "action_map_canonical_at_p2_onset": bool(
                action_map_canonical_at_p2_onset),
            "world_steps_total_at_p2_onset": world_steps_total_at_p2_onset,
            "n_world_rule_shifts_before_p2": shift_count_at_p2_onset,
            "n_world_rule_shifts_p2": int(
                env._world_rule_shift_count) - shift_count_at_p2_onset,
            "n_action_map_entries_permuted_at_p2_end": sum(
                1 for k in canonical_action_map
                if canonical_action_map[k] != env._action_map.get(k)),
            # --- H3 instruments: RECORDED, NON-GATING (see module docstring)
            "persistence_r2": persistence_r2,
            "harm_forward_skill_vs_persistence": fwd_skill,
            "harm_forward_action_sensitivity": act_sensitivity,
            "n_action_sensitivity_samples": len(act_sens_num),
            # --- early-window copies: RECORDED, NON-GATING
            "early_window_steps": early_window,
            "mean_harm_exposure_early_nongating": _mean(harm_exposure_early),
            "mean_dacc_pe_early_nongating": _mean(pe_early),
            "n_pe_early_samples": len(pe_early),
            # series summaries
            "mean_dacc_pe": _mean(pe),
            # The level channel's INTENSITY under a fixed step budget is its
            # accumulation RATE, i.e. the mean per-tick increment. C3 routes on
            # this, not on the level (a level mean over a ramp is a function of
            # where the window happens to sit).
            "mean_valence_harm_delta": _mean(vh),
            "mean_axis1_temperature": _mean(a1_al),
            "mean_axis2_valence_spread_delta": _mean(a2),
            # recorded, NOT routed -- accumulating levels (see note above)
            "mean_valence_harm_level": _mean(vh_level),
            "final_valence_harm_level": vh_level[-1] if vh_level else float("nan"),
            "mean_axis2_valence_spread_level": _mean(a2_level),
            # RECORDED, NEVER ROUTED -- 1062's harm-inclusive statistic and the
            # identity check that makes its defect self-evident in this manifest.
            "mean_axis2_spread_incl_harm_level_unrouted": _mean(a2ih_level),
            "axis2_incl_harm_equals_valence_harm_level": bool(
                _mean(a2ih_level) == _mean(vh_level)),
            "axis2_nonharm_differs_from_valence_harm_level": bool(
                _mean(a2_level) != _mean(vh_level)),
            "mean_harm_exposure": _mean(harm_exposure),
            "mean_z_harm_s_norm": _mean(hs_norm),
            "mean_z_harm_a_norm": _mean(ha_norm),
            "mean_e3_precision": _mean(prec),
            # The ACTUAL precision contribution, not just its saturation: the
            # dACC multiplies the residual by (1 + prec_norm). Near 0 the
            # precision leg is negligible; at the 3.0 cap it is a constant.
            # BOTH extremes make it inert, and only the middle exercises it.
            "mean_prec_norm": _mean(prec_norm_vals),
            "frac_precision_weight_saturated": _mean(
                [1.0 if x >= 3.0 else 0.0 for x in prec_norm_vals]),
            "range_dacc_pe": _series_range(pe),
            "range_valence_harm_delta": _series_range(vh),
            "range_axis1_temperature": _series_range(a1_al),
            "range_axis2_valence_spread_delta": _series_range(a2),
            "range_valence_harm_level": _series_range(vh_level),
            # C1
            "abs_spearman_valence_harm_vs_dacc_pe": abs(_spearman(vh, pe_al))
            if not math.isnan(_spearman(vh, pe_al)) else float("nan"),
            # Recorded for contrast ONLY, never routed: the same statistic on
            # the accumulating LEVEL. If this differs wildly from the routed
            # value, that difference IS the ramp artifact the differencing
            # removes, and a reader can see it rather than having to trust it.
            "abs_spearman_valence_harm_LEVEL_vs_dacc_pe_unrouted": abs(
                _spearman(vh_level, pe)) if not math.isnan(
                _spearman(vh_level, pe)) else float("nan"),
            # C2 -- all three pairs, all on instantaneous quantities
            "r2_axis1_axis2": _r2(a1_al, a2),
            "r2_axis1_axis3": _r2(a1_al, pe_al),
            "r2_axis2_axis3": _r2(a2, pe_al),
            # C4
            "lag1_dacc_pe": _lag1_autocorr(pe_al),
            "lag1_valence_harm": _lag1_autocorr(vh),
            "lag1_valence_harm_LEVEL_unrouted": _lag1_autocorr(vh_level),
            # recorded, NON-GATING
            "cross_candidate_valence_range_post_action_mean": _mean(cc_post),
            "cross_candidate_valence_range_shared_seed_mean": _mean(cc_seed),
            "n_series": len(pe_al),
        }
        pairs = [row["r2_axis1_axis2"], row["r2_axis1_axis3"], row["r2_axis2_axis3"]]
        finite_pairs = [p for p in pairs if math.isfinite(p)]
        row["max_pairwise_axis_r2"] = max(finite_pairs) if finite_pairs else float("nan")
        row["lag1_abs_diff_pe_vs_valence_harm"] = (
            abs(row["lag1_dacc_pe"] - row["lag1_valence_harm"])
            if math.isfinite(row["lag1_dacc_pe"]) and math.isfinite(row["lag1_valence_harm"])
            else float("nan")
        )
        cell.stamp(row)

    # Per-cell criterion flags (C3 is cross-arm and is decided in _run).
    row["c1_pass"] = bool(
        math.isfinite(row["abs_spearman_valence_harm_vs_dacc_pe"])
        and row["abs_spearman_valence_harm_vs_dacc_pe"] <= RHO_MAX
    )
    row["c2_pass"] = bool(
        math.isfinite(row["max_pairwise_axis_r2"])
        and row["max_pairwise_axis_r2"] <= R2_MAX
    )
    row["c4_pass"] = bool(
        math.isfinite(row["lag1_abs_diff_pe_vs_valence_harm"])
        and row["lag1_abs_diff_pe_vs_valence_harm"] >= LAG1_DELTA_MIN
    )
    cell_pass = row["c1_pass"] and row["c2_pass"]
    print("verdict: %s" % ("PASS" if cell_pass else "FAIL"), flush=True)
    return row


# --------------------------------------------------------------------------- #
# readiness preconditions                                                     #
# --------------------------------------------------------------------------- #

def _precondition_specs(fresh_floor: int) -> List[PreconditionSpec]:
    """Every spec declares a structural bound.

    V3-EXQ-1062 declared NONE, so assert_no_structurally_unsatisfiable_gate ran,
    returned cleanly, and PROVED NOTHING -- a negative instrument whose
    "nothing found" was indistinguishable from "the search broke" (autopsy
    section 6, instrument finding 1; CLAUDE.md General Rules). Each bound below
    is the BEST value the arm could attain given its pre-registered config: a
    `structural_max` for a FLOOR, a `structural_min` for a CEILING, and an
    honest None where no bound is derivable rather than a fabricated one.

    `fresh_floor` is the pre-registered FRESH_TICKS_MIN on a real run and the
    budget-scaled value under --dry-run only (see _active_sample_floors).
    """
    return [
        PreconditionSpec(
            name="harm_a_forward_r2_supra_floor",
            description=(
                "E2HarmAForward must actually predict z_harm_a on held-out P2 "
                "transitions. Below this floor dacc._affective_pe falls back to "
                "||z_harm_a|| (its z_harm_a_pred-is-None branch), axis 3 is a LEVEL "
                "rather than a forward-model residual, and C1 would be comparing two "
                "levels -- a starved criterion, not a falsified one."),
            control=(
                "V3-EXQ-597b measured harm_a_forward_r2=0.91 on its PE_FORWARD arm on "
                "this substrate family against the same 0.30 floor; worst seed in this "
                "arm is reported."),
            threshold=FORWARD_R2_MIN,
            direction="lower",
            # R^2 is bounded above by 1.0 by construction.
            structural_max=lambda ctx: 1.0,
        ),
        PreconditionSpec(
            name="valence_harm_series_range_supra_floor",
            description=(
                "Cross-tick RANGE of the PER-TICK INCREMENT of the "
                "VALENCE_HARM_DISCRIMINATIVE read -- deliberately the increment and "
                "not the level. The residue field ACCUMULATES (field.py:294: update_"
                "valence 'does NOT replace the existing value -- adds to it'), so the "
                "level is a monotone ramp whose range is large no matter what the "
                "channel is doing; the increment is the instantaneous quantity C1/C3/C4 "
                "actually route on. This is therefore the SAME statistic those criteria "
                "consume: a near-constant increment series gives an undefined "
                "correlation, which must read not-ready rather than low."),
            control=(
                "V3-EXQ-887b populated HARM_DISC on 31/32 nodes at this exact env and "
                "write-path config. This experiment's own 2026-09-19 Step 2.5a probe "
                "measured a LEVEL range of EXACTLY 0 until the update_residue / "
                "update_harm_salience write paths were driven, and 460.4 once they "
                "were -- which is why they are driven explicitly here and why this "
                "precondition gates. The floor is applied to the INCREMENT range, "
                "which the probe's 0 -> 460.4 ramp over 592 ticks implies is "
                "comfortably non-zero (mean increment ~0.78) but which is measured "
                "here rather than inferred."),
            threshold=VH_RANGE_MIN,
            direction="lower",
            # The increment series has no upper bound on its range.
            structural_max=lambda ctx: float("inf"),
        ),
        PreconditionSpec(
            name="dacc_pe_series_range_supra_floor",
            description=(
                "Cross-tick RANGE of the dACC precision-weighted harm PE -- the same "
                "statistic C1/C2/C4 route on, for the same reason as above."),
            control=(
                "Step 2.5a probe, 2026-09-19: bundle['pe'] range 0.771 over 592 fresh "
                "E3 ticks on this exact config with the write paths driven."),
            threshold=PE_RANGE_MIN,
            direction="lower",
            structural_max=lambda ctx: float("inf"),
        ),
        PreconditionSpec(
            name="axis1_temperature_range_supra_floor",
            description=(
                "Cross-tick RANGE of effective_temperature (axis 1's value). C2's "
                "R^2 against axis 1 is undefined if it never varies."),
            control=(
                "Step 2.5a probe, 2026-09-19: effective_temperature range 0.310 and "
                "pcc_stability 0.0855-0.498 (range 0.413) under 799's re-scoped PCC "
                "weights -- NOT the ~0.017 pinned value 799 measured at the substrate "
                "defaults."),
            threshold=TEMP_RANGE_MIN,
            direction="lower",
            # effective_temperature is exp() of an exponent clipped to
            # +/- salience_temperature_exponent_clip (4.0 here), so its range
            # cannot exceed exp(4) - exp(-4). A real, derivable bound rather
            # than an inf placeholder.
            structural_max=lambda ctx: math.exp(4.0) - math.exp(-4.0),
        ),
        PreconditionSpec(
            name="harm_exposure_relative_deviation_bounded",
            description=(
                "This SHIFT arm's mean harm exposure must not deviate from the "
                "STATIONARY REFERENCE arm's by more than the ceiling. The level "
                "channel is only a MATCHED control for the PE channel if harm "
                "exposure really is matched; unmatched exposure would let C3 read an "
                "exposure difference as a channel dissociation. Measured, never "
                "assumed. Denominated on the REFERENCE arm and not on a pooled mean "
                "that includes this arm: with two arms a self-inclusive pooled "
                "deviation is |A-B|/(A+B), which is IDENTICAL for both arms and so "
                "can never single one out, and a 0.35 ceiling on it would admit a "
                "2.08x between-arm exposure ratio. Against the reference the same "
                "number means what it says -- a 1.25x ratio at the ceiling."),
            control=(
                "CEILING, pre-registered before the run at a 1.25x exposure ratio "
                "against the stationary reference; not derived from this run's own "
                "spread."),
            threshold=HARM_EXPOSURE_REL_DEV_MAX,
            direction="upper",
            # A CEILING on |relative deviation|, whose best (minimum) attainable
            # value is 0.0 -- exactly matched exposure. Satisfiable in
            # principle; whether the lever ALLOWS it is hypothesis H1 and is
            # what this run measures.
            structural_min=lambda ctx: 0.0,
            applies_to=lambda ctx: bool(ctx.get("is_shift")),
            applies_note=(
                "Scoped OUT of ARM_0_STATIONARY: that arm IS the reference the "
                "deviation is measured against, so the precondition is not "
                "meaningful for it (disposition (a), not a vacuous arm)."),
        ),
        PreconditionSpec(
            name="pe_load_elevated_vs_stationary",
            description=(
                "The IV must actually have raised the harm-PE load in this SHIFT arm "
                "relative to the stationary reference. Below floor means the "
                "world_rule_shift manipulation never moved the channel C3 routes on, "
                "which is substrate-not-ready, NOT evidence about MECH-055."),
            control=(
                "V3-EXQ-861e's ecological_novelty_mel_gradient_present_this_config "
                "precondition was MET on this same IV (and its "
                "C1_measured_mel_gradient_present / C1_dv_spread_nonzero read true). "
                "STATED PRECISELY, because 1062 over-read it: 861e's DV is a MEL "
                "GRADIENT, a DIFFERENT channel from the dACC harm-PE this "
                "precondition gates on. It establishes that the lever is "
                "non-degenerate and moves SOMETHING -- V3-EXQ-1062 separately "
                "measured mean E3 running precision falling 3.2x-6.5x under shift, "
                "so the world/E3 side is reached -- and it establishes NOTHING about "
                "the harm-forward side. NO prior run has measured whether this lever "
                "moves the dACC harm-PE channel at all; this run is that measurement, "
                "under the trained-then-shifted regime the lever was designed for. A "
                "below-floor reading here is therefore a genuine negative about the "
                "lever-to-channel path, NOT evidence about MECH-055 -- and the "
                "RECORDED harm_forward_skill_vs_persistence and "
                "harm_forward_action_sensitivity are what make it attributable to H3 "
                "(persistence-dominated model) rather than ambiguous."),
            threshold=PE_ELEVATION_MIN,
            direction="lower",
            # Relative PE elevation is unbounded above.
            structural_max=lambda ctx: float("inf"),
            applies_to=lambda ctx: bool(ctx.get("is_shift")),
            applies_note=(
                "Scoped OUT of ARM_0_STATIONARY: that arm IS the reference the "
                "elevation is measured against, so the precondition is not "
                "meaningful for it (disposition (a), not a vacuous arm)."),
        ),
        PreconditionSpec(
            name="fresh_select_sample_floor",
            description=(
                "Number of FRESH E3 selections in this arm's worst cell. The per-tick "
                "series are recorded only on fresh ticks (the E3 cadence latches "
                "diagnostics ~10x), so this is the TRUE denominator of every "
                "correlation below."),
            control=(
                "CORRECTED against the predecessor's REALISED counts (Step 4.5 "
                "red-team F4). The 2026-09-19 probe figure -- 592 fresh selections "
                "per 600 env steps (98.7%), projecting to ~1780 per cell -- held "
                "only for that short probe and is NOT what a full run yields. "
                "V3-EXQ-1062 actually realised, on its STATIONARY arm at this exact "
                "config and these exact seeds: 1800 (seed 42, 0 latched), 384 (seed "
                "137, 1416 latched), 238 (seed 2026, 1562 latched) -- the commitment "
                "latch holds for most of the budget on two of three seeds, and seed "
                "2026 clears this 200 floor by only 19%. That is the honest margin, "
                "and it is why this gates. NOTE: 1062a's ARM_0_STATIONARY is a "
                "config-identical recomputation of 1062's ARM_0 (same _make_env "
                "arguments, same seeds, same schedule), so on the same machine class "
                "those three numbers double as a free substrate-regression canary -- "
                "a stationary row that does NOT reproduce them means the substrate "
                "moved."),
            threshold=float(fresh_floor),
            direction="lower",
            # A cell cannot make more FRESH E3 selections than it takes env
            # steps, so the P2 budget IS the best attainable value. This is the
            # bound whose absence made 1062's design-time guard inert: its
            # --dry-run budget of 60 could never reach the floor of 200, and
            # nothing said so.
            structural_max=lambda ctx: float(ctx.get("p2_budget", 0)),
        ),
        PreconditionSpec(
            name="world_rule_shift_fired_in_p2",
            description=(
                "Number of world-rule shifts that actually fired during P2 in this "
                "SHIFT arm's worst cell. The onset fix writes env.world_rule_shift_* "
                "AFTER training, so unlike a construction-time kwarg it is not "
                "self-evidently in effect -- if the attribute write ever stopped "
                "taking, every arm would silently become a stationary arm and C1 "
                "would be scored on a manipulation that never happened. This is the "
                "manipulation-occurred check, measured rather than assumed."),
            control=(
                "Step 2.5a probe, 2026-09-22, on this exact env config: with the "
                "lever disabled, 60 env steps left _action_map bit-identical to the "
                "canonical ACTIONS map and _world_steps_total at 0; setting the three "
                "attributes then fired the first shift at world step 10 and every 10 "
                "steps after -- 6 fires in 60 steps at interval 10, permuting 4 of the "
                "8 action-map entries. FLOOR at 0.5, i.e. at least one shift."),
            threshold=0.5,
            direction="lower",
            # At most one shift per `interval` env steps of P2.
            structural_max=lambda ctx: (
                float(int(ctx.get("p2_budget", 0)) // int(ctx["interval"]))
                if int(ctx.get("interval", 0)) > 0 else 0.0),
            applies_to=lambda ctx: bool(ctx.get("is_shift")),
            applies_note=(
                "Scoped OUT of ARM_0_STATIONARY: that arm must fire ZERO shifts by "
                "design, so the precondition is not meaningful for it (disposition "
                "(a), not a vacuous arm). The stationary arm's zero is asserted "
                "separately via action_map_canonical_at_p2_onset and "
                "n_world_rule_shifts_p2, both recorded per cell."),
        ),
    ]


def _run(dry_run: bool):
    global _CORR_MIN_N_ACTIVE
    t0 = time.perf_counter()
    print("%s  dry_run=%s" % (EXPERIMENT_TYPE, dry_run))

    # Sample floors. Unchanged from the pre-registered constants on a REAL run;
    # scaled to the reduced budget under --dry-run ONLY, so the smoke exercises
    # the scoring path rather than routing substrate_not_ready_requeue
    # bit-identically to a real failure (1062 autopsy section 6, finding 2).
    fresh_floor, corr_floor = _active_sample_floors(dry_run)
    _CORR_MIN_N_ACTIVE = corr_floor
    p2_budget_eff = DRY_P2_BUDGET if dry_run else P2_STEP_BUDGET
    if dry_run:
        print("  [smoke] sample floors scaled to the dry-run budget: "
              "fresh_ticks_min=%d corr_min_n=%d (pre-registered %d / %d)"
              % (fresh_floor, corr_floor, FRESH_TICKS_MIN, CORR_MIN_N))

    specs = _precondition_specs(fresh_floor)
    arm_contexts = {a["arm_id"]: {"arm_id": a["arm_id"], "is_shift": a["is_shift"],
                                  "interval": a["interval"],
                                  "p2_budget": p2_budget_eff}
                    for a in ARMS}
    # Design-time refusal: no arm may carry a precondition it provably cannot
    # meet. Runs BEFORE compute, and under --dry-run.
    assert_no_structurally_unsatisfiable_gate(specs, list(arm_contexts.values()))

    rows: List[Dict[str, Any]] = []
    seeds = SEEDS[:1] if dry_run else SEEDS
    for arm in ARMS:
        for seed in seeds:
            rows.append(_run_cell(arm, seed, dry_run))

    by_arm: Dict[str, List[Dict[str, Any]]] = {a["arm_id"]: [] for a in ARMS}
    for r in rows:
        by_arm[r["arm_id"]].append(r)

    # Pooled harm exposure, for the matched-control precondition.
    pooled_harm = _mean([r["mean_harm_exposure"] for r in rows])
    stationary_pe = _mean([r["mean_dacc_pe"] for r in by_arm[STATIONARY_ARM]])
    stationary_harm = _mean([r["mean_harm_exposure"] for r in by_arm[STATIONARY_ARM]])
    # RECORDED, NON-GATING early-window reference. Nothing below gates on these.
    stationary_pe_early = _mean(
        [r["mean_dacc_pe_early_nongating"] for r in by_arm[STATIONARY_ARM]])
    stationary_harm_early = _mean(
        [r["mean_harm_exposure_early_nongating"] for r in by_arm[STATIONARY_ARM]])
    early_nongating_by_arm: Dict[str, Dict[str, float]] = {}

    arm_gates = []
    for arm in ARMS:
        aid = arm["arm_id"]
        arows = by_arm[aid]
        r2_worst, r2_cell = _worst_cell(arows, "harm_a_forward_r2", "min")
        vh_worst, vh_cell = _worst_cell(arows, "range_valence_harm_delta", "min")
        pe_worst, pe_cell = _worst_cell(arows, "range_dacc_pe", "min")
        t_worst, t_cell = _worst_cell(arows, "range_axis1_temperature", "min")
        fresh_worst, fresh_cell = _worst_cell(arows, "n_fresh_select", "min")
        arm_harm = _mean([r["mean_harm_exposure"] for r in arows])
        arm_pe = _mean([r["mean_dacc_pe"] for r in arows])
        measured = {
            "harm_a_forward_r2_supra_floor": r2_worst,
            "valence_harm_series_range_supra_floor": vh_worst,
            "dacc_pe_series_range_supra_floor": pe_worst,
            "axis1_temperature_range_supra_floor": t_worst,
            "fresh_select_sample_floor": fresh_worst,
        }
        if arm["is_shift"]:
            measured["pe_load_elevated_vs_stationary"] = _rel_change(arm_pe, stationary_pe)
            measured["harm_exposure_relative_deviation_bounded"] = abs(
                _rel_change(arm_harm, stationary_harm))
            measured["world_rule_shift_fired_in_p2"] = _worst_cell(
                arows, "n_world_rule_shifts_p2", "min")[0]
            # RECORDED, NON-GATING early-window copies of the two cross-arm
            # preconditions. Computed on the SAME statistics over the first
            # EARLY_WINDOW_STEPS of P2 instead of the full window. They are NOT
            # in `measured`, so they cannot enter any gate; they exist so that a
            # run vacated on the full-window exposure bar still tells a
            # successor whether a shorter window would have cleared it.
            early_nongating_by_arm[aid] = {
                # F2 companion. The GATING exposure statistic is body[10]
                # harm_exposure: an EMA of |harm_signal| on CONTACT only
                # (causal_grid_world.py:3018) that resets to 0 every episode
                # (:1967, :2293). The LEVEL channel this is supposed to certify
                # as a matched control runs on z_harm, encoded from the hazard
                # field VIEW -- proximity, not contact. The two are already
                # demonstrably decoupled on this substrate: V3-EXQ-1062 had
                # contact exposure deviating 4.28x while mean_z_harm_s_norm
                # deviated only 1.07x (0.749 vs 0.803). So record the proximity
                # deviation beside the contact one. RECORDED, NON-GATING -- the
                # pre-registered gate is unchanged and still runs on contact.
                "z_harm_s_norm_rel_dev": abs(_rel_change(
                    _mean([r["mean_z_harm_s_norm"] for r in arows]),
                    _mean([r["mean_z_harm_s_norm"]
                           for r in by_arm[STATIONARY_ARM]]))),
                "mean_z_harm_s_norm": _mean(
                    [r["mean_z_harm_s_norm"] for r in arows]),
                "per_seed_harm_exposure_rel_dev": [
                    {"seed": r["seed"],
                     "rel_dev": abs(_rel_change(
                         r["mean_harm_exposure"],
                         next((q["mean_harm_exposure"]
                               for q in by_arm[STATIONARY_ARM]
                               if q["seed"] == r["seed"]), float("nan"))))}
                    for r in arows],
                "mean_harm_exposure_early": _mean(
                    [r["mean_harm_exposure_early_nongating"] for r in arows]),
                "mean_dacc_pe_early": _mean(
                    [r["mean_dacc_pe_early_nongating"] for r in arows]),
                "harm_exposure_rel_dev_early": abs(_rel_change(
                    _mean([r["mean_harm_exposure_early_nongating"] for r in arows]),
                    stationary_harm_early)),
                "pe_elevation_early": _rel_change(
                    _mean([r["mean_dacc_pe_early_nongating"] for r in arows]),
                    stationary_pe_early),
                "early_window_steps": float(min(EARLY_WINDOW_STEPS, p2_budget_eff)),
            }
        gate = evaluate_arm_gate(aid, arm_contexts[aid], specs, measured)
        for p in gate["preconditions"]:
            p["offending_cell"] = {
                "harm_a_forward_r2_supra_floor": r2_cell,
                "valence_harm_series_range_supra_floor": vh_cell,
                "dacc_pe_series_range_supra_floor": pe_cell,
                "axis1_temperature_range_supra_floor": t_cell,
                "fresh_select_sample_floor": fresh_cell,
                "world_rule_shift_fired_in_p2": _worst_cell(
                    arows, "n_world_rule_shifts_p2", "min")[1],
            }.get(p["precondition"], aid)
        arm_gates.append(gate)

    agg = aggregate_arm_gates(arm_gates)
    green_arms = set(agg["green_arms"])

    # ------------------------------ criteria ------------------------------- #
    # Every criterion is scored on GREEN arms only -- a red arm's readouts are
    # not cited in either direction, and a red arm does NOT vacate a green one.
    def _quorum(aid: str, key: str) -> Tuple[int, int, bool]:
        arows = by_arm[aid]
        n_pass = sum(1 for r in arows if r[key])
        need = min(SEED_QUORUM, len(arows)) if arows else 0
        return n_pass, len(arows), bool(arows) and n_pass >= need

    c1_arms = [a["arm_id"] for a in ARMS
               if a["is_shift"] and a["arm_id"] in green_arms]
    c2_arms = [a["arm_id"] for a in ARMS if a["arm_id"] in green_arms]

    c1_by_arm = {aid: _quorum(aid, "c1_pass") for aid in c1_arms}
    c2_by_arm = {aid: _quorum(aid, "c2_pass") for aid in c2_arms}
    c4_by_arm = {aid: _quorum(aid, "c4_pass") for aid in c2_arms}

    c1_pass = bool(c1_by_arm) and all(v[2] for v in c1_by_arm.values())
    c2_pass = bool(c2_by_arm) and all(v[2] for v in c2_by_arm.values())
    c4_pass = bool(c4_by_arm) and all(v[2] for v in c4_by_arm.values())

    # C3: per seed, the PE channel's relative rise across the ladder must
    # exceed the level channel's by REL_CHANGE_DELTA_MIN. Both legs are
    # RELATIVE changes, so the comparison is scale-free -- a raw-magnitude
    # comparison would survive a broadcast constant (V3-EXQ-604c).
    c3_rows: List[Dict[str, Any]] = []
    c3_ready = STATIONARY_ARM in green_arms and HIGH_ARM in green_arms
    for seed in seeds:
        ref = next((r for r in by_arm[STATIONARY_ARM] if r["seed"] == seed), None)
        hi = next((r for r in by_arm[HIGH_ARM] if r["seed"] == seed), None)
        if ref is None or hi is None:
            continue
        d_pe = _rel_change(hi["mean_dacc_pe"], ref["mean_dacc_pe"])
        d_vh = _rel_change(hi["mean_valence_harm_delta"], ref["mean_valence_harm_delta"])
        delta = (abs(d_pe) - abs(d_vh)) if (math.isfinite(d_pe) and math.isfinite(d_vh)) \
            else float("nan")
        c3_rows.append({
            "seed": seed,
            "rel_change_dacc_pe": d_pe,
            "rel_change_valence_harm": d_vh,
            "differential": delta,
            "passed": bool(math.isfinite(delta) and delta >= REL_CHANGE_DELTA_MIN),
        })
    c3_n_pass = sum(1 for r in c3_rows if r["passed"])
    c3_decidable = bool(c3_ready and c3_rows)
    c3_pass = bool(c3_decidable and c3_n_pass >= min(SEED_QUORUM, len(c3_rows)))

    # ------------------------- non-degeneracy ------------------------------ #
    def _nd_series(keys: List[str]) -> bool:
        return all(
            any(math.isfinite(r[k]) and r[k] > 0.0 for r in rows) for k in keys
        ) and any(r["n_series"] >= _CORR_MIN_N_ACTIVE for r in rows)

    criteria_non_degenerate = {
        # `and bool(c1_arms)`: an UNEVALUATED criterion is degenerate by
        # definition -- it discriminated nothing (1062 autopsy section 1).
        "C1_harm_channels_not_redundant": bool(
            _nd_series(["range_valence_harm_delta", "range_dacc_pe"])
            and c1_arms),
        "C2_axes_not_fixed_ratio": bool(
            _nd_series(["range_axis1_temperature",
                        "range_axis2_valence_spread_delta",
                        "range_dacc_pe"])),
        "C3_differential_channel_response": bool(
            c3_ready and _series_range(
                [r["mean_dacc_pe"] for r in rows]) > 0.0),
        "C4_timing_signatures_differ": bool(
            _nd_series(["range_valence_harm_delta", "range_dacc_pe"])),
    }

    # ---------------------------- verdict grid ----------------------------- #
    # DECIDABILITY IS CHECKED BEFORE ANY CLAIM VERDICT, and this ordering is
    # load-bearing rather than tidy. `c1_pass` is False both when the criterion
    # was evaluated and FAILED and when it could not be evaluated at all --
    # `c1_arms` is empty whenever no SHIFT arm passed its readiness gate, and
    # `c1_pass = bool(c1_by_arm) and ...` is then False. Without this guard a
    # shift arm going red (on `pe_load_elevated_vs_stationary`, whose own spec
    # text says a failure there "is substrate-not-ready, NOT evidence about
    # MECH-055") while the stationary arm stayed green would be recorded as
    # "MECH-055 FALSIFIER (ii) FIRED", evidence_direction `weakens` -- a
    # readiness failure laundered into a claim falsification, with nothing in
    # the manifest distinguishing it from a genuine one. Note that
    # aggregate_arm_gates' `non_degenerate` is ANY-arm-green by design
    # (precondition_gate.py: a red arm must not vacate a green one), so it
    # cannot carry this check on its own. Found by the Step 4.5 red-team pass.
    # ------------------- H1 / H2 / H3 attribution (NON-GATING) ------------- #
    # Read ONLY to word the decidability branch below. Nothing here can create
    # or remove a green arm, and nothing here touches a claim verdict -- see
    # "THESE INSTRUMENTS ADD NO NEW WAY TO VACATE THE RUN" in the docstring.
    shift_rows = [r for r in rows if r["is_shift_arm"]]
    stat_rows = [r for r in rows if not r["is_shift_arm"]]
    # H3 is judged on the STATIONARY arm: action-sensitivity is a property of
    # the trained WEIGHTS, and the stationary arm reads them on the un-permuted
    # world they were trained for. The shift arm's values are recorded beside
    # it so the two can be compared rather than assumed equal.
    h3_skill_stat = _mean([r["harm_forward_skill_vs_persistence"] for r in stat_rows])
    h3_sens_stat = _mean([r["harm_forward_action_sensitivity"] for r in stat_rows])
    h3_persistence_dominated = bool(
        (math.isfinite(h3_skill_stat)
         and h3_skill_stat <= PERSISTENCE_SKILL_ATTRIBUTION_FLOOR)
        or (math.isfinite(h3_sens_stat)
            and h3_sens_stat <= ACTION_SENSITIVITY_ATTRIBUTION_FLOOR))
    r2_stat = _mean([r["harm_a_forward_r2"] for r in stat_rows])
    r2_shift = _mean([r["harm_a_forward_r2"] for r in shift_rows])
    h2_forward_r2_gap = (r2_stat - r2_shift
                         if math.isfinite(r2_stat) and math.isfinite(r2_shift)
                         else float("nan"))
    # Read the failed set from `arm_gates` -- the list evaluate_arm_gate
    # actually returned -- NOT from a nested key on the aggregate. An earlier
    # draft read agg["failed_preconditions_by_arm"], which is {} at that level
    # (the populated copy lives under agg["per_arm_gate"]), and the attribution
    # then reported "no shift-arm readiness precondition failed" while the shift
    # arm had failed three. That is the negative-instrument failure CLAUDE.md
    # names: a broken lookup read identically to a genuine zero. Caught by the
    # 2026-09-22 smoke, which is the one thing 1062's smoke could not do.
    #
    # Two of the three remedies that rule asks for are applied: the scan keeps an
    # explicit CANNOT-DETERMINE category (attribution_lookup_ok), and it records
    # its own pre-filter DENOMINATOR (n_arm_gates_scanned / n_shift_arm_gates_seen)
    # so an empty result can never again be mistaken for a clean one.
    _shift_arm_ids = {a["arm_id"] for a in ARMS if a["is_shift"]}
    _shift_failed = set()
    _n_shift_gates_seen = 0
    for _g in arm_gates:
        _aid = _g.get("arm") or _g.get("arm_id")
        if _aid not in _shift_arm_ids:
            continue
        _n_shift_gates_seen += 1
        for _f in (_g.get("failed_preconditions") or []):
            _shift_failed.add(_f if isinstance(_f, str) else str(_f))
    # CANNOT-DETERMINE: every shift arm must have been seen by the scan. If one
    # was not, an empty _shift_failed means the scan missed it, NOT that it
    # passed -- and the summary below must say so rather than assert a clean run.
    _attr_lookup_ok = (_n_shift_gates_seen == len(_shift_arm_ids))
    _failed_blob = " ".join(sorted(_shift_failed))
    attribution = {
        "h1_exposure_error_coupling_indicated": bool(
            "harm_exposure_relative_deviation_bounded" in _failed_blob),
        "h2_forward_r2_gap_stationary_minus_shift": h2_forward_r2_gap,
        "h2_note": (
            "V3-EXQ-1062 measured harm_a_forward_r2 as statistically "
            "INDISTINGUISHABLE between arms (0.932-0.954 shift vs 0.940-0.977 "
            "stationary) despite ~810 permutations during the shift arm's own "
            "TRAINING -- the signature of a model that had already adapted to "
            "permutation. A clearly POSITIVE gap here, with training now "
            "stationary in both arms, confirms H2 and shows the lever does reach "
            "the harm-forward model once the model was trained on a stable world. "
            "A gap at ~0 again leaves H2 unsupported and points at H3 or H1."),
        "h3_persistence_dominated_indicated": h3_persistence_dominated,
        "h3_mean_skill_vs_persistence_stationary": h3_skill_stat,
        "h3_mean_skill_vs_persistence_shift": _mean(
            [r["harm_forward_skill_vs_persistence"] for r in shift_rows]),
        "h3_mean_action_sensitivity_stationary": h3_sens_stat,
        "h3_mean_action_sensitivity_shift": _mean(
            [r["harm_forward_action_sensitivity"] for r in shift_rows]),
        "h3_mean_persistence_r2_stationary": _mean(
            [r["persistence_r2"] for r in stat_rows]),
        "h3_mean_persistence_r2_shift": _mean(
            [r["persistence_r2"] for r in shift_rows]),
        "persistence_skill_attribution_floor": PERSISTENCE_SKILL_ATTRIBUTION_FLOOR,
        "action_sensitivity_attribution_floor": ACTION_SENSITIVITY_ATTRIBUTION_FLOOR,
        "shift_arm_failed_preconditions": sorted(_shift_failed),
        "attribution_lookup_ok": _attr_lookup_ok,
        "n_arm_gates_scanned": len(arm_gates),
        "n_shift_arm_gates_seen": _n_shift_gates_seen,
        "n_shift_arms_expected": len(_shift_arm_ids),
        "early_window_nongating_by_arm": early_nongating_by_arm,
        "gating_note": (
            "EVERY field in this block is RECORDED and NON-GATING. It is read "
            "only to word the decidability branch of the verdict grid. No "
            "pre-registered threshold was relaxed and no arm's gate consults "
            "any of it."),
    }
    _attr_bits = []
    if attribution["h1_exposure_error_coupling_indicated"]:
        _ea = early_nongating_by_arm.get(HIGH_ARM, {})
        _ea2 = early_nongating_by_arm.get(HIGH_ARM, {})
        _attr_bits.append(
            "the shift arm failed the matched-exposure ceiling. THIS IS CONSISTENT "
            "WITH H1 (exposure and forward-model error structurally coupled under "
            "this lever) BUT DOES NOT ESTABLISH IT -- at least three other causes "
            "produce the same signature and this run does not separate them: (a) the "
            "policy simply breaking under a permuted map, since P2 does no training "
            "so E2.world_forward stays wrong for the whole window and the agent walks "
            "into hazards; (b) EMA warm-up bias, because harm_exposure resets to 0 "
            "every episode (causal_grid_world.py:1967, :2293) at alpha 0.1 and the "
            "shift arm runs many more, much shorter episodes; (c) rare-event sampling "
            "noise -- V3-EXQ-1062's stationary reference ran 0.0017 / 0.0040 / 0.0061 "
            "across seeds, a 3.6x span, against a 1.25x ceiling. Companion reads "
            "recorded beside it: the PROXIMITY deviation (z_harm_s_norm_rel_dev %s), "
            "which is what the level channel actually runs on, and the per-seed "
            "PAIRED deviations (both in early_window_nongating_by_arm) -- with "
            "training now bit-identical per seed, a paired deviation is the "
            "lower-noise comparison. Step 4.5 red-team F2. "
            % (_ea2.get("z_harm_s_norm_rel_dev"),))
        _attr_bits.append(
            "NON-GATING early-window copy over the first %s P2 steps -- "
            "harm_exposure_rel_dev_early %s against the same 0.25 ceiling, "
            "pe_elevation_early %s against the same 0.05 floor. Read these before "
            "designing a windowed successor: exposure deviation ACCUMULATES with "
            "time under the shift while PE elevation is immediate, so these are "
            "what say whether a shorter measurement window would clear the "
            "PRE-REGISTERED bars WITHOUT moving them."
            % (_ea.get("early_window_steps"), _ea.get("harm_exposure_rel_dev_early"),
               _ea.get("pe_elevation_early")))
    if "pe_load_elevated_vs_stationary" in _failed_blob:
        if h3_persistence_dominated:
            # Name WHICH leg tripped. The condition is an OR, so a summary that
            # asserted both were below floor would misreport the leg that was
            # not -- and on the 2026-09-22 smoke it did exactly that, calling an
            # action sensitivity of 2.42 "at or below the 0.05 floor". A reader
            # would have concluded the model ignores the action while the
            # measurement said the opposite.
            _skill_below = (math.isfinite(h3_skill_stat)
                            and h3_skill_stat <= PERSISTENCE_SKILL_ATTRIBUTION_FLOOR)
            _sens_below = (math.isfinite(h3_sens_stat)
                           and h3_sens_stat <= ACTION_SENSITIVITY_ATTRIBUTION_FLOOR)
            _legs = []
            if _skill_below:
                _legs.append(
                    "skill vs the z(t-1) persistence predictor is %.4g, AT OR BELOW "
                    "its %.2g floor -- the learned model is not beating the trivial "
                    "predictor" % (h3_skill_stat, PERSISTENCE_SKILL_ATTRIBUTION_FLOOR))
            else:
                _legs.append(
                    "skill vs persistence is %.4g, ABOVE its %.2g floor (this leg "
                    "did NOT trip)"
                    % (h3_skill_stat, PERSISTENCE_SKILL_ATTRIBUTION_FLOOR))
            if _sens_below:
                _legs.append(
                    "normalised action sensitivity is %.4g, AT OR BELOW its %.2g "
                    "floor -- the model effectively ignores the action, so an "
                    "action-map re-permutation cannot raise its residual at ANY dose"
                    % (h3_sens_stat, ACTION_SENSITIVITY_ATTRIBUTION_FLOOR))
            else:
                _legs.append(
                    "normalised action sensitivity is %.4g, ABOVE its %.2g floor "
                    "(this leg did NOT trip -- the model DOES use the action)"
                    % (h3_sens_stat, ACTION_SENSITIVITY_ATTRIBUTION_FLOOR))
            _attr_bits.append(
                "consistent with H3 (persistence-dominated harm-forward model), on "
                "the stationary arm: %s. H3 is indicated when EITHER leg trips, so "
                "read the per-leg verdicts above before concluding which. Where the "
                "SKILL leg is the one that tripped, a dose ladder is not the route; "
                "this routes to /implement-substrate or to a different decoupling "
                "lever." % ("; ".join(_legs)))
        else:
            _attr_bits.append(
                "H3 REFUTED as the explanation: skill vs persistence is %.4g and "
                "normalised action sensitivity is %.4g on the stationary arm, both "
                "ABOVE the attribution floors -- the model does beat the trivial "
                "predictor and does use the action -- yet the lever still did not "
                "raise the dACC harm-PE. The gap is therefore in the "
                "lever-to-harm-forward PATH, which is the first time that has been "
                "measured (no prior run measured this lever against this channel)"
                % (h3_skill_stat, h3_sens_stat))
    if not _attr_lookup_ok:
        attribution["summary"] = (
            "CANNOT DETERMINE -- the attribution scan saw %d of %d expected SHIFT-arm "
            "gates, so an empty failed-precondition set here means the scan did not "
            "reach them, NOT that they passed. Read per_arm_gate directly and treat "
            "every H1/H2/H3 field in this block as unestablished."
            % (_n_shift_gates_seen, len(_shift_arm_ids)))
    else:
        attribution["summary"] = (
            "; ".join(_attr_bits) if _attr_bits
            else ("no shift-arm readiness precondition failed (scan reached %d of %d "
                  "shift-arm gates, so this is a measured zero)"
                  % (_n_shift_gates_seen, len(_shift_arm_ids))))

    gate_green = bool(agg["non_degenerate"])
    undecidable = []
    if not gate_green:
        undecidable.append("no arm passed its readiness gate")
    if not c1_arms:
        undecidable.append(
            "C1 (the collapse-risk falsifier) is scored on SHIFT arms and no SHIFT "
            "arm passed its readiness gate")
    if not c2_arms:
        undecidable.append("C2 has no readiness-green arm to score")
    if not c3_decidable:
        undecidable.append(
            "C3 needs BOTH the stationary reference and the high-shift arm green, "
            "with at least one seed present in each")
    if undecidable:
        outcome = "FAIL"
        direction = "non_contributory"
        label = "substrate_not_ready_requeue"
        interp_note = (
            "NOT DECIDABLE on this run -- %s. Gate detail: %s. ATTRIBUTION "
            "(RECORDED, NON-GATING -- see interpretation.hypothesis_attribution): "
            "%s. This is NOT a substrate ceiling, NOT a lockstep finding, and NOT a "
            "refutation of MECH-055; no criterion below should be read as a verdict. "
            "DO NOT re-queue this as-is at a longer P0 -- the 1062 autopsy "
            "(section 5a) established that neither cross-arm precondition is a "
            "function of warmup length, and this run's onset fix is already the "
            "remedy that autopsy prescribed. Route to /failure-autopsy and let the "
            "attribution above pick between H1, H2 and H3."
            % ("; ".join(undecidable), agg["degeneracy_reason"] or "all arms green",
               attribution["summary"]))
    elif not c1_pass:
        outcome = "FAIL"
        direction = "weakens"
        label = "harm_channels_numerically_redundant_collapsed_scalar"
        interp_note = (
            "MECH-055 FALSIFIER (ii) FIRED: VALENCE_HARM_DISCRIMINATIVE and the "
            "dACC precision-weighted harm-forward PE are numerically redundant "
            "(|Spearman| > %.2f) in the decoupled SHIFT arm(s), i.e. one collapsed "
            "harm scalar wearing two labels -- the exact risk the claim's own "
            "COLLAPSE-RISK CHECK names, and the same failure class MECH-048 hit "
            "before its 2026-07-21 fix." % RHO_MAX)
    elif not c2_pass:
        outcome = "FAIL"
        direction = "weakens"
        label = "axes_move_in_fixed_ratio_lockstep"
        interp_note = (
            "MECH-055 FALSIFIER (i) FIRED: at least two of the measured axes move in "
            "a fixed, predictable ratio (pairwise R^2 > %.2f) despite the two harm "
            "representations being individually non-redundant." % R2_MAX)
    elif not c3_pass:
        outcome = "FAIL"
        direction = "mixed"
        label = "separation_present_but_manipulation_nonspecific"
        interp_note = (
            "Neither of MECH-055's falsifiers fired -- the channels are neither "
            "numerically redundant (C1) nor in fixed ratio (C2) -- but the decoupling "
            "manipulation did NOT move the PE channel measurably more than the level "
            "channel (C3), so this run does not establish the claim's "
            "'measurably different ... signatures UNDER a manipulation that decouples "
            "their upstream sources' clause. The separation evidence stands; the "
            "attribution to a decoupling manipulation does not.")
    else:
        outcome = "PASS"
        direction = "mixed"
        label = "two_axis_plus_harm_only_separation_supported_full_claim_awaits_benefit_channel"
        interp_note = (
            "The NARROWED two-axis-plus-harm-only test that MECH-055's own "
            "what_would_answer sanctions: both of the claim's falsifiers failed to "
            "fire (C1, C2) and the two harm representations responded differentially "
            "to a decoupling manipulation (C3). Direction is 'mixed', NOT 'supports': "
            "the claim as WORDED requires the harm/benefit duality inside axis 3, "
            "whose benefit half is architecturally absent (MECH-054, 2026-08-08), and "
            "the downstream-role half of the claim's first CONFIRMING clause is out of "
            "scope here (V3-EXQ-799's write_gate consumer gap). A full verdict on "
            "MECH-055 remains blocked on the benefit-side signed-PE channel.")

    run_id = "%s_%sZ_v3" % (EXPERIMENT_TYPE, datetime.utcnow().strftime("%Y%m%dT%H%M%S"))

    criteria = [
        {
            "name": "C1_harm_channels_not_redundant",
            "load_bearing": True,
            "passed": c1_pass,
            "measured": _worst_cell(
                [r for r in rows if r["arm_id"] in c1_arms],
                "abs_spearman_valence_harm_vs_dacc_pe", "max")[0],
            "threshold": RHO_MAX,
            "direction": "upper",
            "statistic": "max over scored cells of |Spearman(VALENCE_HARM_DISCRIMINATIVE, dacc_pe)|",
            "offending_cell": _worst_cell(
                [r for r in rows if r["arm_id"] in c1_arms],
                "abs_spearman_valence_harm_vs_dacc_pe", "max")[1],
            "scored_arms": c1_arms,
            # V3-EXQ-1062 autopsy section 1: `passed` was False both when C1 was
            # evaluated-and-failed and when it was never evaluated at all, so a
            # reader consuming criteria[].passed without also reading
            # scored_arms mistook a starved criterion for a falsified one. The
            # verdict grid always distinguished the two; the RECORD did not.
            "evaluated": bool(c1_arms),
            "unevaluated_reason": (
                "" if c1_arms else
                "scored on SHIFT arms only and no SHIFT arm passed its readiness "
                "gate, so no cell was scored; `passed: false` here means NOT "
                "EVALUATED, not falsified"),
            "per_arm_seed_quorum": {k: {"n_pass": v[0], "n_seeds": v[1], "passed": v[2]}
                                    for k, v in c1_by_arm.items()},
            "maps_to": "MECH-055 FALSIFYING (ii)",
        },
        {
            "name": "C2_axes_not_fixed_ratio",
            "load_bearing": True,
            "passed": c2_pass,
            "measured": _worst_cell(
                [r for r in rows if r["arm_id"] in c2_arms],
                "max_pairwise_axis_r2", "max")[0],
            "threshold": R2_MAX,
            "direction": "upper",
            "statistic": "max over SCORED (readiness-green) cells and axis pairs of OLS R^2",
            "offending_cell": _worst_cell(
                [r for r in rows if r["arm_id"] in c2_arms],
                "max_pairwise_axis_r2", "max")[1],
            "scored_arms": c2_arms,
            "evaluated": bool(c2_arms),
            "unevaluated_reason": (
                "" if c2_arms else
                "no arm passed its readiness gate; `passed: false` means NOT "
                "EVALUATED, not falsified"),
            "per_arm_seed_quorum": {k: {"n_pass": v[0], "n_seeds": v[1], "passed": v[2]}
                                    for k, v in c2_by_arm.items()},
            "maps_to": "MECH-055 FALSIFYING (i)",
        },
        {
            "name": "C3_differential_channel_response",
            "load_bearing": False,
            "passed": c3_pass,
            "measured": min((r["differential"] for r in c3_rows
                             if math.isfinite(r["differential"])), default=float("nan")),
            "threshold": REL_CHANGE_DELTA_MIN,
            "direction": "lower",
            "statistic": ("min over seeds of |rel_change(mean dacc_pe)| - "
                          "|rel_change(mean VALENCE_HARM_DISCRIMINATIVE)| across the "
                          "stationary -> high-shift ladder"),
            "n_pass": c3_n_pass,
            "n_seeds": len(c3_rows),
            "evaluated": bool(c3_decidable),
            "unevaluated_reason": (
                "" if c3_decidable else
                "C3 needs BOTH the stationary reference and the high-shift arm "
                "readiness-green with a shared seed; `passed: false` means NOT "
                "EVALUATED, not falsified"),
            "per_seed": c3_rows,
            "maps_to": "MECH-055 CONFIRMING, second clause",
        },
        {
            "name": "C4_timing_signatures_differ",
            "load_bearing": False,
            "passed": c4_pass,
            "measured": _worst_cell(
                [r for r in rows if r["arm_id"] in c2_arms],
                "lag1_abs_diff_pe_vs_valence_harm", "min")[0],
            "threshold": LAG1_DELTA_MIN,
            "direction": "lower",
            "statistic": ("min over SCORED (readiness-green) cells of "
                          "|lag1_autocorr(dacc_pe) - "
                          "lag1_autocorr(VALENCE_HARM_DISCRIMINATIVE increment)|"),
            "offending_cell": _worst_cell(
                [r for r in rows if r["arm_id"] in c2_arms],
                "lag1_abs_diff_pe_vs_valence_harm", "min")[1],
            "evaluated": bool(c2_arms),
            "unevaluated_reason": (
                "" if c2_arms else
                "no arm passed its readiness gate; `passed: false` means NOT "
                "EVALUATED, not falsified"),
            "per_arm_seed_quorum": {k: {"n_pass": v[0], "n_seeds": v[1], "passed": v[2]}
                                    for k, v in c4_by_arm.items()},
            "maps_to": "MECH-055 CONFIRMING, second clause (order-sensitive companion)",
        },
    ]

    full_config = {
        "env": dict(ENV_KWARGS),
        "arms": ARMS,
        "world_rule_shift_depth": WORLD_RULE_SHIFT_DEPTH,
        "schedule": {
            "p0_eps": P0_EPS, "p1_eps": P1_EPS,
            "total_training_eps": TOTAL_TRAINING_EPS,
            "steps_per_episode": STEPS_PER_EPISODE,
            "p2_step_budget": P2_STEP_BUDGET,
            "epsilon_train": EPSILON_TRAIN, "epsilon_eval": EPSILON_EVAL,
        },
        "substrate": _config_slice(0)["substrate"],
        "thresholds": {
            "RHO_MAX": RHO_MAX, "R2_MAX": R2_MAX,
            "REL_CHANGE_DELTA_MIN": REL_CHANGE_DELTA_MIN,
            "LAG1_DELTA_MIN": LAG1_DELTA_MIN, "SEED_QUORUM": SEED_QUORUM,
            "FORWARD_R2_MIN": FORWARD_R2_MIN, "VH_RANGE_MIN": VH_RANGE_MIN,
            "TEMP_RANGE_MIN": TEMP_RANGE_MIN, "PE_RANGE_MIN": PE_RANGE_MIN,
            "HARM_EXPOSURE_REL_DEV_MAX": HARM_EXPOSURE_REL_DEV_MAX,
            "PE_ELEVATION_MIN": PE_ELEVATION_MIN,
            "FRESH_TICKS_MIN": FRESH_TICKS_MIN, "CORR_MIN_N": CORR_MIN_N,
        },
        # NON-GATING attribution floors. Named here so a reader can see they are
        # separate from the pre-registered thresholds above and route nothing.
        "nongating_attribution_floors": {
            "PERSISTENCE_SKILL_ATTRIBUTION_FLOOR": PERSISTENCE_SKILL_ATTRIBUTION_FLOOR,
            "ACTION_SENSITIVITY_ATTRIBUTION_FLOOR": ACTION_SENSITIVITY_ATTRIBUTION_FLOOR,
            "EARLY_WINDOW_STEPS": EARLY_WINDOW_STEPS,
        },
        "world_rule_shift_onset": "p2_boundary_post_training",
        "active_sample_floors": {
            "fresh_ticks_min": fresh_floor, "corr_min_n": corr_floor,
            "scaled_for_dry_run_only": bool(dry_run),
        },
        "dry_run": dry_run,
    }

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "backlog_id": BACKLOG_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "supersedes": SUPERSEDES,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "outcome": outcome,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {"MECH-055": direction},
        "evidence_direction_note": interp_note,
        "sleep_driver_pattern": "none",
        "interpretation": {
            "label": label,
            "note": interp_note,
            "preconditions": agg["adjudication_preconditions"],
            "criteria_non_degenerate": criteria_non_degenerate,
            "hypothesis_attribution": attribution,
        },
        "criteria": criteria,
        "combination_rule": (
            "PASS requires BOTH load-bearing criteria (C1 and C2) AND C3. C4 is "
            "corroborating and does not gate. Each criterion must hold in at least "
            "%d of the seeds of EVERY arm it is scored on; C1 is scored on the SHIFT "
            "arms only (the decoupling manipulation is applied there; the stationary "
            "arm's value is recorded as the reference), C2 and C4 on every green arm. "
            "Only arms whose readiness gate is green are scored." % SEED_QUORUM
        ),
        "per_arm_gate": agg["per_arm_gate"],
        "failed_preconditions_by_arm": agg.get("failed_preconditions_by_arm", {}),
        "arm_results": rows,
        "per_seed_results": rows,
        "c3_per_seed": c3_rows,
        "scope_note": (
            "NARROWED test, licensed by MECH-055's own what_would_answer. NOT in "
            "scope: (a) the benefit-side signed-PE channel, architecturally absent "
            "per MECH-054 2026-08-08, so no result here verdicts the harm/benefit "
            "duality the claim's wording requires; (b) the downstream/behavioural "
            "form of the claim's first CONFIRMING clause, blocked by V3-EXQ-799's "
            "write_gate consumer gap; (c) MECH-035's ranking claim -- the "
            "cross-candidate valence range is RECORDED here, not tested, and MECH-035 "
            "is deliberately NOT tagged."
        ),
        "dv_symmetry_note": (
            "DV family: Spearman correlations, OLS R^2, lag-1 autocorrelations and "
            "RELATIVE changes over per-tick series. Symmetry group: independent "
            "positive affine rescaling of either series, plus a uniform additive "
            "constant (exact for correlation/R^2; rescaling only for a relative "
            "change). The SAME statement holds for BOTH arms -- there are two, "
            "ARM_0_STATIONARY and ARM_2_HIGH_SHIFT, and full_config['arms'] carries "
            "two (V3-EXQ-1062 emitted 'all three arms' here against a two-arm config; "
            "autopsy section 6, finding 4). They differ only in whether and from when "
            "one manipulation runs: re-permuting the action -> displacement "
            "map REORDERS and RE-CONTENTS the z_harm_a sequence and destroys its "
            "action-conditioned predictability -- neither an affine rescaling nor an "
            "added constant of any measured series. So no arm's DV is invariant under "
            "its own manipulation and no arm is disposition-(b) vacuous. Designed "
            "around two consequences of that group: C3 compares RELATIVE changes "
            "rather than raw magnitudes because a broadcast constant survives a "
            "magnitude readout (V3-EXQ-604c); and because a pooled-over-ticks "
            "correlation IS permutation-invariant, C4's lag-1 autocorrelation is "
            "included as the order-SENSITIVE companion statistic."
        ),
        "red_team_note": {
            "step": "4.5 adversarial design review, one pass",
            "verdict": RED_TEAM_VERDICT,
            "this_run_note": (
                "V3-EXQ-1062a's own Step 4.5 pass. The seven findings below are "
                "V3-EXQ-1062's, INHERITED WHOLE -- every fix is carried forward "
                "unchanged in this driver, and F1 in particular is what this run "
                "extends with the H1/H2/H3 attribution."),
            "this_run_dispositions": [
                "F1 (criterion cannot discriminate / result nobody could attribute) "
                "FIXED -- axis 2 was the spread over ALL SIX valence components, "
                "which IS the harm level: components 4/5 are written only under "
                "use_mech307_split_surprise (residue/field.py:52-57), unset here, so "
                "the minimum is pinned at 0 and max-min collapses to the "
                "harm-dominated max. VERIFIED before acting, two ways: a 150-step "
                "probe on this config (wanting 0, liking 0.169, harm_disc 110.628, "
                "surprise 0.331, pos/neg 0 -> spread == harm_disc bit-identical) and "
                "1062's own manifest (mean_axis2_valence_spread_{level,delta} == "
                "mean_valence_harm_{level,delta} in 6/6 cells). Axis 2 is now the "
                "NON-HARM component spread, which is alive (range 0.331). 1062's "
                "statistic is recorded unrouted with explicit identity flags. "
                "Threshold unchanged. The predecessor's C2 reading is raised to "
                "governance as an evidence_discrepancy, not silently corrected.",
                "F2 (gate certifying its own subject) FIXED AS RECORDING + WORDING "
                "-- the matched-exposure gate runs on body[10] harm_exposure, a "
                "CONTACT-only EMA reset every episode (causal_grid_world.py:3018, "
                ":1967, :2293), while the level channel it certifies runs on z_harm "
                "from the hazard field VIEW, i.e. PROXIMITY. Confirmed decoupled on "
                "this substrate: 1062 had contact deviating 4.28x while "
                "mean_z_harm_s_norm deviated 1.07x. The PRE-REGISTERED gate is "
                "UNCHANGED (relaxing or re-pointing it mid-flight is exactly what "
                "this chip forbids); instead the proximity deviation and the "
                "per-seed PAIRED deviations are recorded beside it, and the H1 "
                "attribution no longer ASSERTS coupling -- it now names the three "
                "rival explanations it cannot separate (policy breakdown under a "
                "permuted map with no P2 training; per-episode EMA warm-up bias; "
                "rare-event sampling noise, 1062's stationary reference spanning "
                "3.6x across seeds against a 1.25x ceiling).",
                "F3 (manipulation reaches more than the DV) NOT FIXED, RECORDED AS A "
                "STATED CAVEAT with a substrate-side owner -- _maybe_shift_world_rule "
                "draws from self._rng (causal_grid_world.py:2136) and reset() draws "
                "episode layouts from the same generator (:1712), so past the first "
                "P2 shift the layout sequence diverges too. Same-distribution, so "
                "added variance rather than bias, but it lands on the sparse-event "
                "exposure statistic. A dedicated permutation RNG is the fix and "
                "belongs with SD-PP-B4. The docstring's 'onset and nothing else' "
                "claim is corrected to scope it to TRAINING, which the bit-identity "
                "measurement does establish.",
                "F4 (stale control text) FIXED -- the fresh-select control cited a "
                "short probe's 98.7% yield projecting to ~1780 per cell. 1062 "
                "realised 1800 / 384 / 238 on its stationary arm at these exact "
                "seeds, with seed 2026 clearing the 200 floor by 19%. The control "
                "now carries the realised numbers and the honest margin, and notes "
                "that 1062a's stationary arm is a config-identical recomputation of "
                "1062's and so doubles as a substrate-regression canary.",
                "F5 (fingerprint hygiene) FIXED -- a --dry-run cell computes with "
                "2/2/12/60 and a scaled correlation floor while _config_slice "
                "declares the pre-registered schedule, so it was content-addressed "
                "identically to a real cell. Harmless while dry-run statistics were "
                "all NaN; the new scaled floors make them finite and plausible, which "
                "is a false-HIT shape. Dry-run cells now carry "
                "extra_ineligible_reasons=['dry_run'].",
                "F6 (provenance trivia) FIXED -- the probe text said 4 of 8 "
                "action-map entries; CausalGridWorld.ACTIONS has 5 (four moves plus "
                "a stay). Corrected in both places.",
                "Family checks the reviewer ran that HELD: P0+P1 bit-identity across "
                "arms (independently measured here -- identical e2_harm_a parameter "
                "hash e12b1a03f0d9ba32 and identical z_harm_a at seed 42); the "
                "manipulation path open to the DV (dacc_pe_cap and "
                "dacc_saturation_enabled both default off and unset, so bundle['pe'] "
                "is residual x (1 + prec_norm)); joint satisfiability of the PE and "
                "exposure gates not definitionally excluded; no verdict-grid branch "
                "routing a CONTROL failure to `weakens`, and NaN measured values "
                "failing closed; all eight structural bounds genuine best-attainable "
                "values with none certifying its own subject; and the three new H3 "
                "instruments confirmed unable to reach `measured`, any arm gate, or "
                "any criterion.",
            ],
            "inherited_from_1062_model": "fable",
            "inherited_from_1062_verdict": "CONTESTED",
            "inherited_from_1062": [
                "F1 (verdict grid) FIXED -- an empty c1_arms made c1_pass False "
                "whether C1 was evaluated-and-failed or could not be evaluated at "
                "all, so a SHIFT arm going red on readiness while the stationary arm "
                "stayed green was recorded as 'MECH-055 FALSIFIER (ii) FIRED', "
                "direction weakens. aggregate_arm_gates' non_degenerate is "
                "any-arm-green by design and cannot carry the check. Now a "
                "decidability test runs BEFORE any claim verdict and routes to "
                "substrate_not_ready_requeue / non_contributory, naming which "
                "criterion was undecidable.",
                "F2 (manipulation cannot reach the DV) FIXED -- P1 trained and the "
                "P2 r2 estimator scored e2_harm_a(z(t-1), a(t)), pairing a transition "
                "with the action that had not caused it, and with a DIFFERENT pairing "
                "from the one the measured PE uses. Verified in ree_core: sense() "
                "caches _harm_a_prev = z(t) (agent.py:5583), select_action rolls "
                "pred = e2_harm_a(z(t), a(t)) (agent.py:10329), the next tick's dACC "
                "compares z(t+1) against it (agent.py:7743). Since world_rule_shift "
                "acts ONLY through the action channel, this blunted the very "
                "manipulation the design depends on. Both sites now use prev_action.",
                "F3 (gate certifying its own subject) FIXED -- agent.reset() clears "
                "_harm_a_pred_prev (agent.py:3630), so the first fresh tick of every "
                "episode reaches the dACC with z_harm_a_pred=None and "
                "dacc._affective_pe returns ||z_harm_a||, a LEVEL (dacc.py:213-214). "
                "Shift arms end episodes earlier, so mean_dacc_pe -- which both "
                "pe_load_elevated_vs_stationary and C3 read -- could have risen on "
                "episode COUNT alone. Those ticks are now excluded and counted in "
                "n_level_pe_ticks_excluded, and differencing is done WITHIN episodes.",
                "F4 (gate certifying its own subject) FIXED -- the matched-exposure "
                "gate was denominated on a pooled mean INCLUDING the arm under test. "
                "With two arms that statistic is |A-B|/(A+B), identical for both arms, "
                "so it could never single one out, and the 0.35 ceiling admitted a "
                "2.08x exposure ratio. Now denominated on the stationary REFERENCE "
                "arm, scoped out of that arm via applies_to, and tightened to 0.25 "
                "(a 1.25x ratio).",
                "F5 (criterion cannot discriminate) ALREADY ADDRESSED before the pass "
                "returned -- the residue-field integrator makes the level a ramp, so "
                "lag1 ~ 1 and C4 would pass trivially. Every criterion already routes "
                "on within-episode INCREMENTS, with levels recorded unrouted. The "
                "sub-claim that the sense() z_harm.norm() write path 'dominates' the "
                "channel is DISMISSED as not-a-defect: VALENCE_HARM_DISCRIMINATIVE is "
                "DEFINED as the z_harm_s-driven sensory-discriminative channel "
                "(residue/field.py:60, agent.py:5507), so its dominance is the channel "
                "behaving as specified, not contamination.",
                "F6 (manipulation cannot reach the DV, magnitude unverified) FIXED as "
                "INSTRUMENTATION -- the precision leg may be negligible rather than "
                "saturated (prec_norm = precision/5000; e3_selector.py cites "
                "current_precision ~95, giving a ~1.02 multiplier). The driver "
                "previously recorded only SATURATION, which would read 0 in that "
                "regime and be mistaken for 'precision is live'. It now records "
                "mean_prec_norm and the note covers BOTH inert regimes. Deliberately "
                "NOT gated: every routed statistic is invariant under a positive "
                "constant scaling, so an inert precision leg bounds what a PASS may "
                "claim about precision, but cannot change any verdict.",
                "F7 (verdict grid, minor) FIXED -- C1/C2/C4 headline measured and "
                "offending_cell values were taken over ALL rows, so a readiness-red "
                "arm's value could be reported as a criterion's number. Now scoped to "
                "the readiness-green arms each criterion is actually scored on.",
            ],
        },
        "precision_saturation_note": (
            "frac_precision_weight_saturated records the fraction of scored ticks on "
            "which dacc._affective_pe's precision term hit its cap "
            "(prec_norm = min(precision / dacc_precision_scale, 3.0), scale 5000 "
            "following V3-EXQ-597b). Above the cap the precision leg is a CONSTANT "
            "multiplier, so axis 3 is an unweighted forward-model residual times 4. "
            "This is RECORDED, not gated, and the reason is specific: every routed "
            "statistic here -- Spearman, OLS R^2, lag-1 autocorrelation, relative "
            "change -- is invariant under a positive constant scaling, so saturation "
            "can neither manufacture nor destroy any criterion's verdict. What it DOES "
            "bound is the reach of a PASS: if this fraction is near 1, the run has "
            "tested the separation of a forward-model RESIDUAL channel from the level "
            "channel, and has NOT exercised the precision-weighting that MECH-055's "
            "axis-3 wording also names. Read it before citing a PASS as evidence about "
            "precision specifically."
        ),
        "accumulation_note": (
            "Every routed statistic is computed on INSTANTANEOUS quantities. The "
            "residue-field channels ACCUMULATE (residue/field.py:294 -- update_valence "
            "adds rather than replaces, and every write path here is non-negative), so "
            "VALENCE_HARM_DISCRIMINATIVE and the valence spread are monotone ramps "
            "(Step 2.5a probe: 0 -> 460.4 over 592 ticks). C1 on a ramp-vs-residual "
            "pair would pass because a ramp and a fluctuating residual are nearly "
            "uncorrelated, and C4 would pass because a ramp's lag-1 autocorrelation is "
            "~1 -- both for reasons unrelated to channel separation, and both "
            "invisible in the manifest. C1/C2/C3/C4 therefore route on the per-tick "
            "INCREMENT of the accumulating channels, aligned tick-for-tick with the "
            "already-instantaneous temperature and harm-PE series. The levels are "
            "recorded but never routed, and "
            "abs_spearman_valence_harm_LEVEL_vs_dacc_pe_unrouted is C1's own statistic "
            "computed the WRONG way, kept so the size of the removed artifact is "
            "visible rather than asserted."
        ),
        "structural_guard_limit_note": (
            "Every PreconditionSpec here declares a structural bound, so "
            "assert_no_structurally_unsatisfiable_gate now PROVES something "
            "rather than returning cleanly because it had nothing to check "
            "(V3-EXQ-1062 declared none of the eight; autopsy section 6, finding "
            "1). STATED SO IT IS NOT OVER-READ: those bounds are PER-PRECONDITION. "
            "The guard's API takes one bound per spec, so it cannot express, and "
            "does NOT certify, JOINT satisfiability across two preconditions. The "
            "specific joint question this run turns on -- whether "
            "pe_load_elevated_vs_stationary (>= +5%) and "
            "harm_exposure_relative_deviation_bounded (<= 1.25x) can BOTH hold at "
            "once under this lever -- is hypothesis H1, and it is empirical, not "
            "structural: if exposure and forward-model error are coupled by the "
            "manipulation then no dose satisfies both, and no design-time check "
            "could have known that. Measuring it is what this run is for. A green "
            "structural audit here therefore means 'no single precondition is "
            "provably unmeetable', NOT 'the gate is jointly satisfiable'."
        ),
        "readiness_scope_note": (
            "Each readiness precondition certifies ONE channel and speaks for no "
            "other: harm_a_forward_r2 certifies axis 3 only; "
            "valence_harm_series_range certifies the VALENCE_HARM_DISCRIMINATIVE "
            "INCREMENT series only (and says nothing about its accumulating level, "
            "which is recorded but never routed); axis1_temperature_range certifies "
            "axis 1 only; "
            "harm_exposure_relative_deviation certifies the matched-control premise "
            "of C3 only."
        ),
        "levers_ruled_out_note": (
            "Three candidate decoupling levers were rejected on substrate evidence "
            "BEFORE this design, each recorded so a successor does not retry them: "
            "(1) SD-021/AIC descending attenuation of z_harm_s -- gated on the "
            "commitment latch via mode_weight = p_external * (beta_gate_elevated), "
            "aic_analog.py:249, so it cannot fire commitment-free; (2) "
            "harm_nonredundancy_weight -- V3-EXQ-323 measured baseline cosine_sq at "
            "8.5e-05 to 0.025, a floor with no headroom, and 323 failed its own C1 in "
            "2/5 seeds because of it (this knob is also a dataclass field with no "
            "from_dims kwarg, so from_dims silently swallows it); (3) "
            "env_drift_interval -- a measured null, V3-EXQ-677 produced a "
            "high-vs-low mean-PE difference of 8.8e-07 against a 0.01 threshold, and "
            "causal_grid_world.py's own SD-MEL-PRODUCER note explains why."
        ),
        "recorded_non_gating_note": (
            "cross_candidate_valence_range_post_action_mean is RECORDED, not gated: "
            "MECH-055's collapse-risk check does not depend on cross-candidate "
            "spread, so a monostrategy-degenerate candidate pool must not vacate C1. "
            "cross_candidate_valence_range_shared_seed_mean is its NEGATIVE CONTROL: "
            "world_states[0] is the rollout's shared initial z_world seed, "
            "bit-identical across candidates (E2FastPredictor.rollout_with_world), so "
            "a NON-zero shared-seed range would mean the range statistic itself is "
            "mis-implemented. HONEST LIMIT, measured up front: this experiment's Step "
            "2.5a probe measured BOTH reads at exactly 0 over 592 fresh ticks -- the "
            "candidates' rolled-out z_world lands far enough from every active RBF "
            "center that evaluate_valence returns ~0 for all of them, the collapsed- "
            "proposer regime config.py's candidate_summary_source note describes and "
            "V3-EXQ-614e measured (cand_world_pairwise_dist=0.0). So while the "
            "post-action read is 0 the negative control is UNINFORMATIVE -- it cannot "
            "distinguish a correct statistic from a broken one -- and neither number "
            "may be cited as evidence about MECH-035 or about candidate diversity. "
            "They are recorded only so a successor that fixes the proposer regime can "
            "see what this substrate did. This is precisely why neither gates: "
            "MECH-055's collapse-risk check reads the REALIZED z_world, not the "
            "candidate pool, so a degenerate pool must not vacate C1."
        ),
        "readout": flat_readout({
            "C1_harm_channels_not_redundant": c1_pass,
            "C2_axes_not_fixed_ratio": c2_pass,
            "C3_differential_channel_response": c3_pass,
            "C4_timing_signatures_differ": c4_pass,
            "overall_pass_flag": outcome == "PASS",
            "readiness_any_arm_green": gate_green,
            "n_green_arms": len(green_arms),
            "n_arms": len(ARMS),
            "n_criteria_passed": sum(1 for x in (c1_pass, c2_pass, c3_pass, c4_pass) if x),
            "n_criteria_total": 4,
            "rho_max": RHO_MAX,
            "r2_max": R2_MAX,
            "rel_change_delta_min": REL_CHANGE_DELTA_MIN,
            "lag1_delta_min": LAG1_DELTA_MIN,
            "forward_r2_min": FORWARD_R2_MIN,
            "abs_spearman_vh_pe_worst": _worst_cell(
                rows, "abs_spearman_valence_harm_vs_dacc_pe", "max")[0],
            "max_pairwise_axis_r2_worst": _worst_cell(
                rows, "max_pairwise_axis_r2", "max")[0],
            "lag1_abs_diff_worst": _worst_cell(
                rows, "lag1_abs_diff_pe_vs_valence_harm", "min")[0],
            "c3_differential_worst": min(
                (r["differential"] for r in c3_rows
                 if math.isfinite(r["differential"])), default=None),
            "harm_a_forward_r2_worst": _worst_cell(rows, "harm_a_forward_r2", "min")[0],
            "valence_harm_delta_range_worst": _worst_cell(
                rows, "range_valence_harm_delta", "min")[0],
            "valence_harm_level_range_worst": _worst_cell(
                rows, "range_valence_harm_level", "min")[0],
            "dacc_pe_range_worst": _worst_cell(rows, "range_dacc_pe", "min")[0],
            "axis1_temperature_range_worst": _worst_cell(
                rows, "range_axis1_temperature", "min")[0],
            "n_fresh_select_worst": _worst_cell(rows, "n_fresh_select", "min")[0],
            "n_latched_ticks_total": sum(r["n_latched_ticks"] for r in rows),
            "mean_dacc_pe_stationary": stationary_pe,
            "mean_dacc_pe_high_shift": _mean(
                [r["mean_dacc_pe"] for r in by_arm[HIGH_ARM]]),
            "mean_valence_harm_delta_stationary": _mean(
                [r["mean_valence_harm_delta"] for r in by_arm[STATIONARY_ARM]]),
            "mean_valence_harm_delta_high_shift": _mean(
                [r["mean_valence_harm_delta"] for r in by_arm[HIGH_ARM]]),
            "pooled_mean_harm_exposure": pooled_harm,
            "cross_candidate_range_post_action_mean": _mean(
                [r["cross_candidate_valence_range_post_action_mean"] for r in rows]),
            "cross_candidate_range_shared_seed_mean": _mean(
                [r["cross_candidate_valence_range_shared_seed_mean"] for r in rows]),
            "frac_precision_weight_saturated_mean": _mean(
                [r["frac_precision_weight_saturated"] for r in rows]),
            "n_cells": len(rows),
            "n_seeds": len(seeds),
            # The two GATING cross-arm statistics themselves. 1062 kept these
            # only inside per_arm_gate, so the numbers that actually decided the
            # run were invisible to every flat-readout consumer and its autopsy
            # had to dig them out by hand.
            "harm_exposure_rel_dev_full_gating": abs(_rel_change(
                _mean([r["mean_harm_exposure"] for r in by_arm[HIGH_ARM]]),
                stationary_harm)),
            "pe_elevation_full_gating": _rel_change(
                _mean([r["mean_dacc_pe"] for r in by_arm[HIGH_ARM]]),
                stationary_pe),
            "harm_exposure_rel_dev_max": HARM_EXPOSURE_REL_DEV_MAX,
            "pe_elevation_min": PE_ELEVATION_MIN,
            # NON-GATING early-window copies.
            "harm_exposure_rel_dev_early_nongating": early_nongating_by_arm.get(
                HIGH_ARM, {}).get("harm_exposure_rel_dev_early", float("nan")),
            "pe_elevation_early_nongating": early_nongating_by_arm.get(
                HIGH_ARM, {}).get("pe_elevation_early", float("nan")),
            "early_window_steps": float(min(EARLY_WINDOW_STEPS, p2_budget_eff)),
            # Onset provenance.
            "n_world_rule_shifts_p2_total": sum(
                r["n_world_rule_shifts_p2"] for r in rows),
            "n_world_rule_shifts_p2_worst_shift_arm": _worst_cell(
                [r for r in rows if r["is_shift_arm"]],
                "n_world_rule_shifts_p2", "min")[0],
            "action_map_canonical_at_p2_onset_all_cells": all(
                r["action_map_canonical_at_p2_onset"] for r in rows),
            "n_shifts_before_p2_total": sum(
                r["n_world_rule_shifts_before_p2"] for r in rows),
            # H2 / H3 instruments, RECORDED and NON-GATING.
            "harm_a_forward_r2_stationary": r2_stat,
            "harm_a_forward_r2_high_shift": r2_shift,
            "h2_forward_r2_gap_stationary_minus_shift": h2_forward_r2_gap,
            "persistence_r2_stationary": _mean(
                [r["persistence_r2"] for r in stat_rows]),
            "persistence_r2_high_shift": _mean(
                [r["persistence_r2"] for r in shift_rows]),
            "harm_forward_skill_vs_persistence_stationary": h3_skill_stat,
            "harm_forward_skill_vs_persistence_high_shift": _mean(
                [r["harm_forward_skill_vs_persistence"] for r in shift_rows]),
            "harm_forward_action_sensitivity_stationary": h3_sens_stat,
            "harm_forward_action_sensitivity_high_shift": _mean(
                [r["harm_forward_action_sensitivity"] for r in shift_rows]),
            "h3_persistence_dominated_indicated": h3_persistence_dominated,
            "persistence_skill_attribution_floor": PERSISTENCE_SKILL_ATTRIBUTION_FLOOR,
            "action_sensitivity_attribution_floor": ACTION_SENSITIVITY_ATTRIBUTION_FLOOR,
            # Precision leg: 1062 emitted mean_prec_norm per row only, so the
            # 260x-2600x cap overshoot was invisible in the flat readout
            # (autopsy section 6, finding 3).
            "mean_prec_norm_mean": _mean([r["mean_prec_norm"] for r in rows]),
            "mean_e3_precision_mean": _mean([r["mean_e3_precision"] for r in rows]),
            "dacc_precision_scale": DACC_PRECISION_SCALE,
            # Active sample floors, so a reader can tell a scaled smoke from a
            # real run without opening the config.
            "fresh_ticks_min_active": float(fresh_floor),
            "corr_min_n_active": float(corr_floor),
        }),
        "custom_information": {
            "gov_reuse_1_check": (
                "Decisive readout = the JOINT per-tick pairing of "
                "VALENCE_HARM_DISCRIMINATIVE with the dACC precision-weighted "
                "harm-forward PE (and the three-axis lockstep R^2 over the same "
                "ticks). Checked via reanalysis_query.py over 1050 recorded "
                "manifests: node_valence_matrix is carried by exactly three runs "
                "(v3_exq_887 / 887a / 887b), and every one of them ran with "
                "use_dacc=False, so none carries a dACC PE to pair it with; the 11 "
                "dacc_pe carriers carry no valence store. "
                "v3_exq_876a_mech025_doing_mode_convergence_redesign appears in both "
                "keyword sets but only as CONFIG keys, with use_dacc=False and "
                "valence_harm_enabled=False. The pairing is therefore neither "
                "recorded nor derivable post-hoc, and the run additionally needs a "
                "manipulation (world_rule_shift as a harm-PE decoupler) present in no "
                "recorded MECH-055-relevant run. Not recoverable -> run. RE-CHECKED "
                "for V3-EXQ-1062a, 2026-09-22: the decisive readout is unchanged, and "
                "the only run that has ever carried it -- V3-EXQ-1062 "
                "(..._20260922T175212Z_v3), the sole manifest tagged MECH-055 -- is "
                "the CONFOUNDED one this supersedes. Its C1 was never scored "
                "(scored_arms empty), so there is nothing to reanalyse: no recorded "
                "manifest contains a post-training-onset shift arm, and the "
                "persistence-baseline and action-sensitivity readouts this run adds "
                "are absent from every manifest in the corpus. Not recoverable -> "
                "run."
            ),
            "step_2_5a_probe_2026_09_19": (
                "Empirical wiring probe on this exact config, before authoring, in "
                "two passes. PASS 1 (220 steps, valence write paths NOT driven): "
                "e2_harm_a built, dACC + adapter + salience coordinator + PCC all "
                "present, 32 candidates carrying world_states, env obs dims measured "
                "(body 12, world 250, action_dim 5 -- NOT the 4 some sibling drivers "
                "hardcode); but evaluate_valence returned ALL ZEROS, because no RBF "
                "center is active until update_residue runs and evaluate_valence "
                "short-circuits on active_mask.any(). PASS 2 (600 steps, write paths "
                "driven exactly as _drive_valence_write_paths drives them): "
                "VALENCE_HARM_DISCRIMINATIVE range 460.4, bundle['pe'] range 0.771, "
                "effective_temperature range 0.310, pcc_stability 0.0855-0.498, "
                "surprise_write_count 600, and 592 FRESH E3 selections per 600 env "
                "steps (98.7%, versus 29/220 in pass 1 -- the yield is "
                "config-dependent, which is why it is a measured precondition). "
                "Pass 2 is why _drive_valence_write_paths exists and why the "
                "valence_harm_series_range precondition gates. Both passes measured "
                "the cross-candidate valence range at EXACTLY 0 (see "
                "recorded_non_gating_note). Also confirmed from_dims returns "
                "harm_nonredundancy_weight=0.0 when passed, i.e. silently swallowed. "
                "Per-step cost measured at ~0.84 s CPU, which is what sized the "
                "schedule and dropped the third arm."
            ),
            "re_derive_brake": (
                "Counted 1 braking autopsy for MECH-055 over the live "
                "failure_autopsy corpus at authoring time (2026-09-22) -- "
                "failure_autopsy_V3-EXQ-1062_2026-09-22.json, the direct "
                "predecessor -- against RE_DERIVE_BRAKE_THRESHOLD 2. NOT BRAKED. "
                "Canary: SD-003 at 74 targets, so the scan reaches "
                "targets[].claim_ids and the count is a real count. Note also that "
                "that single autopsy is an ENVIRONMENT/mis-schedule finding whose "
                "own routing is `queue-experiment`, not a substrate ceiling, so it "
                "is the kind of hit the brake exists to let through."
            ),
            "step_2_5a_probe_2026_09_22": (
                "ONSET probe for this driver, run before authoring, on this exact "
                "ENV_KWARGS config. PHASE A (world_rule_shift DISABLED, 60 env "
                "steps): _action_map bit-identical to the canonical "
                "CausalGridWorldV2.ACTIONS map, _world_steps_total 0, "
                "world_rule_shift_count 0 -- confirming _maybe_shift_world_rule's "
                "own claim that every RNG draw sits inside the enabled guard, so a "
                "disabled env consumes no randomness and both arms' P0+P1 are "
                "bit-identical at a seed. PHASE B (the three attributes set "
                "post-hoc, interval 10 depth 2, 60 further steps): shifts fired at "
                "world steps 10, 20, 30, 40, 50, 60 -- 6 fires -- and 4 of the 8 "
                "action-map entries ended permuted. The workaround the autopsy "
                "names therefore WORKS, the onset is clean, and the first shift "
                "lands one full interval after the boundary rather than on it. "
                "This driver re-asserts both halves per cell "
                "(action_map_canonical_at_p2_onset, n_world_rule_shifts_p2) rather "
                "than trusting the probe."
            ),
            "matched_training_measurement_2026_09_22": (
                "The docstring claims both arms' P0+P1 are bit-identical at a "
                "seed. MEASURED, not argued: constructing each arm exactly as "
                "_run_cell does (reset_all_rng(seed), _make_env(seed), "
                "REEAgent(make_config(env))) and running 24 identical steps gives "
                "a byte-identical sha256 over every e2_harm_a parameter "
                "(e12b1a03f0d9ba32 in both arms at seed 42), a byte-identical final "
                "z_harm_a, a canonical _action_map in both, and "
                "_world_steps_total == 0 in both -- confirming the disabled lever "
                "consumes no randomness. So the two arms enter P2 from the SAME "
                "trained state, and TRAINING is controlled exactly. NOT "
                "onset-only for the whole of P2, though: the shift draws its "
                "permutation from the same self._rng that reset() draws episode "
                "layouts from (causal_grid_world.py:2136 vs :1712), so from the "
                "first P2 shift the layout sequence diverges too -- "
                "same-distribution, so added variance rather than bias, but it "
                "lands on the sparse-event exposure statistic (Step 4.5 red-team "
                "F3). Up to that first shift the arms are bit-identical. "
                "The cells are still run independently with a full per-cell RNG "
                "reset rather than trained once and forked, so each remains a pure "
                "function of (substrate, config, seed) and the stationary arm's "
                "fingerprint stays reuse-ELIGIBLE; the duplicated training is the "
                "accepted price of that."
            ),
            "onset_fix_note": (
                "The ONLY change to the manipulation relative to V3-EXQ-1062 is its "
                "ONSET. Dose (interval 10, depth 2), schedule (P0 30 / P1 60 / P2 "
                "1800 steps), seeds, env config and EVERY pre-registered threshold "
                "and readiness floor are unchanged to the digit. That is what makes "
                "1062 -> 1062a a controlled comparison isolating onset, and it is "
                "what lets harm_a_forward_r2's between-arm gap speak to H2. "
                "Explicitly NOT done, both on the autopsy's instruction: P0 was NOT "
                "lengthened (section 5a -- neither failing precondition is a "
                "function of warmup length), and NO dose ladder was added (section "
                "5c -- the autopsy's own withdrawn first draft)."
            ),
            "substrate_known_limitation": (
                "SD-PP-B4-one-shot-world-rule-shift-lever (substrate_queue.json), "
                "severity `degrading`, unblocks_claims includes MECH-055, "
                "substrate_paths "
                "ree_core/environment/causal_grid_world.py::_maybe_shift_world_rule. "
                "It carries V3-EXQ-1062's failure record as its first entry. "
                "`degrading` WARNS and does not block, and this run proceeds under "
                "it deliberately: the gap is that there is no env kwarg for a "
                "post-training / one-shot onset, and the driver-side attribute "
                "write used here IS the workaround the autopsy sanctions. The "
                "supported fix would be a world_rule_shift_at_step / "
                "apply_action_permutation kwarg (or a public "
                "apply_action_permutation(perm) method) so drivers stop writing "
                "private-adjacent attributes; until that lands, every consumer of "
                "this lever that needs a post-training onset carries the same "
                "workaround."
            ),
        },
    }
    manifest["non_degenerate"] = bool(agg["non_degenerate"])
    if agg.get("degeneracy_reason"):
        manifest["degeneracy_reason"] = agg["degeneracy_reason"]

    out_path = write_flat_manifest(
        manifest,
        dry_run=dry_run,
        config=full_config,
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    print("  outcome=%s label=%s" % (outcome, label))
    print("  manifest -> %s" % out_path)
    return outcome, out_path, run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _outcome, _out_path, _run_id = _run(args.dry_run)

    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        run_id=_run_id,
        queue_id=QUEUE_ID,
        dry_run=args.dry_run,
    )
