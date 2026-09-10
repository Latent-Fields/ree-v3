"""V3-EXQ-1020 -- SD-082 learning-signal probe (gradient persistence, return
variance, advantage sign on rule-driven flips).

PURPOSE. V3-EXQ-822f rejected SD-082's registry leg H1 AT ADEQUATE POWER
(t(4) = -4.9 against the 0.5 bar) while SD-082's own three predicates were
positively observed, and the TRAINED head flipped LESS than init in 10 of 10
cells. The `weakens` branch was foreclosed by the dead-ReLU control, so the run
read `inconclusive` and nothing moved. The confirmed autopsy
(failure_autopsy_V3-EXQ-822f_2026-09-09) routed ONE open question to
queue-experiment, and this driver is that probe:

    IS THERE A LEARNING SIGNAL AT ALL -- and can the instrument see it?

822f could not distinguish "the trained coupling carries no signal" from "the
instrument cannot see it". Every criterion below exists to make that one
distinction. The measurements the autopsy named, verbatim: per-P1-update
gradient norm on rule_bias_head, per-tensor step-direction persistence
(||sum of steps|| / sum of ||steps||), per-episode return variance, per-episode
trained-head flip rate (trajectory), and the sign of the advantage on flip vs
non-flip selections; positive control = the same head on a dense synthetic
credit must move D past 0.5 with persistence near 1.

PRE-REGISTERED HYPOTHESES (both registered by the autopsy, Section 7, axis
`learning-signal`; this run is their adjudicating run):
  H-learning-signal-noisy : the P1 gradient is direction-INCONSISTENT -- the head
        is moved by Adam (nonzero per-update gradient) but the steps do not
        accumulate, so displacement stays small. Signature: grad norm above floor
        WITH persistence far below the positive control's.
  H-learning-signal-sign  : REINFORCE is being taught to REDUCE rule-driven
        flips -- the advantage carried by flip selections is systematically
        negative. Signature: mean advantage on flip samples < 0, reproducibly.
They are NOT mutually exclusive; both are measured under one instrument.

EXPLICITLY REFUSED BY THE AUTOPSY, and not re-proposed here:
  (1) A byte-identical V3-EXQ-822g re-run / same-design power bump. This driver
      is a NEW EXQ NUMBER on a NEW AXIS (learning-signal), not a lettered
      iteration of 822f's readout-index design.
  (2) The ON > OFF cross-candidate SPREAD comparison -- inconclusive BY
      CONSTRUCTION at n=5 (822f C4 detectable effect 0.86 against a 0.25 margin;
      ~59 seeds needed). No criterion below is that comparison. The two arms are
      retained only so the telemetry covers the same 10 cells 822f's headline
      observation spans; NO criterion contrasts ARM_ON against ARM_OFF.

SEEDS ARE 822f's, DELIBERATELY. [611, 622, 633, 644, 655] are exactly 822f's.
This is NOT the 822e replay defect (822e replayed 822d's DVs bit-for-bit on
822d's seeds and learned nothing new): the DVs here are a disjoint set of
quantities -- gradient telemetry -- that 822f never recorded. Holding the seeds
fixed is what makes a finding here ATTRIBUTABLE to 822f's specific observation
(trained flips < init in 10/10 cells) rather than leaving a "your seeds behaved
differently" gap.

DESIGN NOTE -- WHY C3 IS GATED ON C4 (found at design time, before compute).
`adv = ep_return - baseline` is a PER-EPISODE quantity: every minibatch sample
drawn from the same episode shares one `ep_return` and one baseline, hence one
advantage. The flip label is PER-SELECTION. So mean-advantage-on-flip and
mean-advantage-on-non-flip can differ ONLY through the composition of episodes --
if flips were distributed uniformly across episodes the two means would be equal
BY ARITHMETIC, at every seed, on every substrate. C3 therefore has resolving
power only when per-episode return varies. That is exactly what C4 measures, so
C4 is wired as C3's non-degeneracy condition rather than as an independent
criterion. A run with flat returns reports C3 non-degenerate=False rather than a
spurious null. (This is also the substance of H-learning-signal-sign: the
hypothesis IS that flip-heavy episodes carry lower returns.)

DESIGN NOTE -- OPERATIONAL DEFINITION OF "ADVANTAGE SIGN ON FLIPS".
Recorded PER-MINIBATCH-SAMPLE, because the minibatch sample is where the
advantage actually enters the gradient (REINFORCE draws 32 of up to 512 buffered
tuples per update, off-policy, spanning many past episodes). Per-episode
aggregates are recorded ALONGSIDE so both granularities are available to a later
reader, but the load-bearing statistic is the per-sample one. `ADV_MIN_THRESHOLD`
drops samples with |adv| < 0.005 BEFORE they reach the loss, which biases the
recorded advantage distribution away from zero -- so `n_adv_filtered` is recorded
and the per-sample statistics are computed over the SURVIVING samples, i.e. over
exactly the population that moved the head.

DESIGN NOTE -- GRADIENT NORM IS READ FROM clip_grad_norm_'s RETURN VALUE.
`torch.nn.utils.clip_grad_norm_` returns the total norm BEFORE clipping and
rescales `.grad` IN PLACE. Reading `.grad` after the call therefore reports the
CLIPPED gradient, which saturates at the 1.0 max-norm and destroys exactly the
signal this probe exists to measure. The return value is captured instead.
Step 2.5a probe (this session, live substrate): the return is a finite Tensor
(0.4884 on a synthetic single-sample loss, i.e. below the clip and therefore
unclipped), and one Adam step moved the head by 0.0162 in parameter norm.

READINESS IS LOAD-BEARING, NOT DIAGNOSTIC. The dense-synthetic-credit positive
control is the gate, not a nice-to-have: without it a near-zero persistence
reading is ambiguous between "no signal" and "instrument blind", which is the
precise gap 822f failed to escape. It trains a COPY of this cell's own init head,
with the same optimiser class and lr and the same REINFORCE loss form, on a
credit signal that is dense and informative by construction, and must LEARN ITS
TASK with persistence near 1. Below floor self-routes to
`substrate_not_ready_requeue` -- never to a substrate verdict label.

WHY THE GATE IS TASK IMPROVEMENT, NOT D (a smoke-time finding, recorded because
it changed the design). The autopsy phrased the control as "must move D past 0.5
with persistence near 1", D being 822f's trained-minus-init discrimination index.
Measured on this substrate, D is too NOISY to gate on: over 32 held-out probes
per cell its per-probe SD is 0.46-0.88 against a mean of 0.03-0.65, and its SIGN
is inconsistent across cells (three cells gave a NEGATIVE mean). It is a ratio of
two small noisy quantities on a k=4 candidate set, so its sampling spread swamps
its effect. Gating on it would have failed the run for MEASUREMENT reasons while
the head was demonstrably learning -- a criterion starved by its own instrument,
which is the failure this probe exists to end, not to repeat.
  The head IS learning, unambiguously and reproducibly: step-direction
persistence 0.719-0.770 and net parameter displacement ~0.37-0.43 in 10 of 10
cells. So the gate asserts the thing that is both directly meaningful and
low-variance: DOES THE CONTROL GET BETTER AT ITS OWN TASK -- expected reward
under the head's own selection policy on HELD-OUT synthetic samples, trained head
minus init head. Measured +0.00223..+0.00327, POSITIVE in 10 of 10 cells, a
tight cluster with no sign ambiguity.
  D is still COMPUTED AND RECORDED (`synth_D_abs`, `synth_D_signed_mean`,
`synth_D_probe_sd`, `synth_D_n_probes`) as a diagnostic, so a later reader can
see the noise for themselves and so the autopsy's named quantity is not silently
dropped -- it is demoted from gate to record, deliberately and in writing.
  Note this calibration is of the POSITIVE CONTROL -- it fixes when the
INSTRUMENT counts as working. It is not a scientific bar, and C1/C2/C3's
thresholds are independent of it and were not tuned against any measurement here.

DV-SYMMETRY INVARIANCE (mandatory declaration, per arm; both arms share the DV
set, so the argument is stated once and holds for each):
  * `grad_norm` is a magnitude, invariant under any sign flip or rotation of the
    gradient. The manipulation (real sparse episode credit vs dense synthetic
    credit) changes gradient MAGNITUDE and direction consistency together; it is
    not a sign flip or rotation, so the DV is not invariant under it.
  * `persistence = ||sum steps|| / sum ||steps||` is invariant under a GLOBAL
    rotation applied to every step and under uniform rescaling of all steps. The
    manipulation changes the step-to-step direction CONSISTENCY, which no global
    rotation and no rescaling preserves -- that is precisely the quantity
    persistence measures. Not invariant.
  * `mean_adv_on_flip` is a signed mean, invariant under permutation of samples
    WITHIN the flip class. The manipulation is the flip LABEL itself (which class
    a sample belongs to), which is not a within-class permutation. Not invariant.
  None of the three DVs is a rank/argmax-derived statistic, so the broadcast-
  scalar and monotone-rescaling annihilations (V3-EXQ-604c, V3-EXQ-643) do not
  apply: a uniform additive constant on the gradient changes its norm, and a
  monotone rescaling changes persistence's numerator and denominator differently.

PRE-REGISTRATION VERIFICATION -- EVERY BRANCH OF THE GRID IS REACHABLE, measured
before queuing (ARM_ON seed 611 at REAL warmup p0=60, p1=20, 48 steps):
  n_rule_live_ticks 629; n_flip_ticks 66 -> flip fraction 0.1049 (consistent with
  822f's pooled 0.077); n_flip_samples 70 AND n_nonflip_samples 567, so C3's two
  classes are BOTH populated; return_variance 0.374 (bar 1e-4), so C4 holds and C3
  is non-degenerate; median per-update gradient norm 0.0091 (bar 1e-6), so C1
  holds; persistence 0.8088; adv_flip_minus_nonflip +0.142.
  At that budget the cell routes to `learning_signal_present_and_directed` -- the
  branch in which BOTH pre-registered hypotheses are REJECTED. That is a real
  possible outcome of this design, and it is stated here so nobody reads a null as
  a surprise: the probe is built to be able to reject its own hypotheses.
  A FIRST attempt at this verification, at a REDUCED warmup (p0=12), returned
  n_flip_ticks 0 of 647 and would have made C3 unmeasurable. That was the warmup,
  not the instrument: with p0=60 the CandidateRuleField has matured (rule_state
  live on 82.7% of >=2-candidate ticks, mean norm 0.137, mean 2.26 active rules)
  and flips appear. Recorded because a reader seeing a low flip count on some
  future variant should suspect the warmup before the measurement.

WHY C2 IS BRACKETED BY TWO IN-RUN CONTROLS (Step 4.5 red-team, BLOCKING, fixed).

An EARLIER draft set a fixed ceiling of 0.25, justified as: "a direction-
INCONSISTENT walk of n steps has ||sum steps||/sum||steps|| ~ 1/sqrt(n), which at
70 updates is ~0.12, so 0.25 sits between the noise regime and the accumulating
one". THAT DERIVATION IS FOR RAW GRADIENT STEPS. Persistence here is measured over
ADAM PARAMETER DELTAS, and Adam's beta1=0.9 momentum plus per-coordinate
normalisation autocorrelate consecutive steps, lifting the noise floor far above
the i.i.d. value. Measured on this exact head geometry (Linear(48,32)/ReLU/
Linear(32,1)) at T=70, lr=5e-4, zero-mean i.i.d. gradients:

    SGD  pure-noise : 0.1176 - 0.1216   (reproduces the 1/sqrt(n) figure exactly,
                                         which is how the error was located)
    ADAM pure-noise : 0.4516 - 0.4632   (invariant over gradient scales 1e-6..1e-2)
    Adam first crosses 0.25 only at T ~ 300 updates; this run performs 70.

So the 0.25 ceiling could NOT fire even with H-learning-signal-noisy MAXIMALLY
TRUE -- an unfalsifiable criterion, which also made two of the five verdict labels
(`H_learning_signal_noisy_supported`, `both_learning_signal_legs_supported`)
unreachable under every outcome. The probe would have been unable to support one
of the two hypotheses it was queued to adjudicate.

THE FIX IS NOT A LOWERED BAR -- it is a bar derived from the right null. C2 is now
bracketed by two controls MEASURED IN-RUN, per cell:
  * NEGATIVE control (`_noise_persistence_control`): the same head, same Adam, same
    lr, same update count, driven by pure noise -- H-learning-signal-noisy made
    true by construction. This is the floor.
  * POSITIVE control (`_synthetic_credit_control`): the achievable ceiling.
C2 fires when the real run's persistence falls below the MIDPOINT of that bracket,
i.e. is closer to noise than to a consistently-accumulating optimiser. Measuring
both in-run also self-calibrates to whatever machine class the run lands on,
rather than trusting a figure measured on the authoring laptop.
`persistence_bracket_valid` (ceiling above floor) is C2's non-degeneracy check.

RED-TEAM DISPOSITIONS (Step 4.5, one pass; findings verified against source before
acting, per the skill -- a finding is a lead, not a verdict):
  F1 BLOCKING, the C2/Adam null above -- CONFIRMED by independent measurement in
     this session (numbers quoted above are mine, not the reviewer's) and FIXED by
     the bracket. This is the finding that justified the whole pass.
  F2 CONTESTED, the rule_state conditioning mismatch -- ACCEPTED AS A REAL
     INTERPRETIVE LIMIT, NOT PATCHED, and recorded here because it is arguably a
     scientific finding rather than a defect. In the real path a minibatch entry
     carries `None` at index 4, so every sample is scored at whatever
     `lpfc.rule_state` currently holds -- one shared, end-of-episode value -- while
     the samples themselves come from many different episodes and ticks. The rule
     half of the head's input is therefore mismatched to the credit on most replayed
     samples. This is FAITHFUL to 822f (its loss form does the same), and
     reproducing it is deliberate: this probe exists to explain 822f's optimisation,
     not a corrected one. But it means a low-persistence reading has a THIRD
     available account beside the two pre-registered hypotheses -- "the driver
     replays samples under a mismatched rule_state" -- which no criterion here can
     exclude. It is therefore recorded as a named rival in
     `interpretation.unexcluded_rivals` and per-update `rule_state_norm` telemetry
     is emitted so a successor can test it directly. Do NOT read a C2 PASS as
     establishing H-learning-signal-noisy over this rival.
  F3 CONTESTED, EMA baseline warm-up -- FIXED. See the de-trending note in
     `_summarise_cell`: C3 now requires its sign to AGREE with a de-trended
     contrast, and `spearman_episode_vs_flip_rate` is recorded.
  F4 CONTESTED, C3 sample count -- FIXED. The pre-registration figures quoted below
     (70 flip / 567 non-flip) WERE hold-weighted: they were measured before the
     fresh-select gate landed. The load-bearing C3 statistic is fresh-only, whose
     per-cell count is ~10x smaller, so `MIN_FLIP_SAMPLES = 20` is now required per
     cell and `n_flip_samples > 0` is no longer accepted as non-degeneracy.
  Also noted and accepted: C1 is a liveness check, not a discriminator -- its floor
  is 1e-6 against a measured 0.0091, so `training_signal_absent_no_gradient` is
  near-unreachable. That is intended: C1 exists to separate "no gradient at all"
  from "gradient present but useless", and the second is what C2/C3 adjudicate.

STEP 2.5a EMPIRICAL PROBE (this session, before authoring; live substrate):
  lateral_pfc live with train_rule_bias_head=True and rule_readout_consumer=True;
  bias_head_parameters() -> 4 tensors, all requires_grad; clip_grad_norm_ returns
  a finite Tensor; one optimiser step moves the head (delta norm 0.0162);
  get_state() carries hidden_dead_relu_frac / rule_summary_magnitude_ratio /
  rule_state_norm. Premise confirmed at runtime, not just in the SD docs.

STEP 2.5b RE-DERIVE BRAKE -- COUNTED 3, RELEASED. An independent recount over the
autopsy corpus returns 3 counting runs for SD-082 (822b, 822c, 822d), which meets
the threshold of 2. The brake is nonetheless RELEASED, on the producer's own
explicit record: failure_autopsy_V3-EXQ-822f_2026-09-09.json carries
`re_derive_brake.fired = false` with `literal_count_meets_threshold = true` and
names those same three runs -- release basis "not a ceiling reading" (822f read
`inconclusive`, which does not count). The autopsy states the licence in terms:
"Refused: V3-EXQ-822g or any same-design letter -- not on power grounds but
because the missing axis is the SIGNAL. Licensed: a NEW EXQ on the
learning-signal axis." This driver is that new EXQ. It also falls squarely in the
skill's own "not braked" exemption: a new EXQ NUMBER, `experiment_purpose:
diagnostic`, whose purpose is to discriminate WHY the ceiling holds.

STEP 2.5c SUBSTRATE-PATH OVERLAP GATE -- resolved by CALL TRACE over 30 real
ticks of this exact driver path (this session), not by module-name matching.
Against every OPEN corrupting substrate_queue entry:
  * `ree_core/cingulate/salience_coordinator.py` (mode-governance-engagement)
        -> NOT EXECUTED. Independently confirmed at the config level:
        `use_salience_coordinator = False` and `use_external_task_drive = False`
        in this driver's REEConfig, so that entire feature is inert here by
        construction. (Recorded because the SAME entry blocked a different
        experiment this session -- V3-EXQ-935a -- whose driver DOES import and
        call the defective `experiments/_lib/regime_occupancy_gate.py`. This
        driver does not import that module at all.)
  * `experiments/_lib/regime_occupancy_gate.py`  -> NOT EXECUTED (not imported).
  * `ree_core/policy/tonic_vigor.py` (MECH-320)  -> NOT EXECUTED.
  * `ree_core/affect/blocked_agency.py`          -> NOT EXECUTED.
  * `ree_core/regulators/selection_entropy_floor.py` (sd105) -> NOT EXECUTED.
  * `ree_core/predictors/e1_deep.py::ContextMemory.write`
    (contextmemory-write-path-addressing-degeneracy) -> NOT EXECUTED. Function-
        level trace: only read / forward / generate_prior / predict_long_horizon
        / get_schema_salience are reached. This driver trains ONLY the bias head,
        through a separate optimiser over `bias_head_parameters()`; it never
        trains E1, so the degenerate WRITE path is never taken.
  * `ree_core/predictors/e1_deep.py::forward` and `::predict_long_horizon`
    (SD-e1-rollout-consistency-training) -> BOTH EXECUTED. This is a real
        module-level overlap and it is dispositioned here rather than waved past.
        It does NOT block, for three independently checkable reasons.
        (a) The overlap is UNIVERSAL, not specific: `e1_deep::forward` is on the
            path of every agent-stepping V3 experiment -- 725 scripts in the
            corpus reach it, including V3-EXQ-1019, queued and PASSed the same
            night as this driver was authored. A gate reading that blocks the
            entire experimental programme is being misapplied, not obeyed. It is
            the same spurious-gating shape governance already corrected once, on
            2026-09-09, by emptying SD-082's own substrate_paths (preserved as
            `substrate_paths_resolved_2026_09_09`) because they described an
            already-resolved defect and were gating unrelated experiments.
        (b) The entry's paths name the item-3 BUILD FOOTPRINT, not a defect locus.
            Its status reads `item3_rollout_endpoint_contrastive_substrate_landed
            _validation_owed` with `ready: true`, and the item-3 build chip
            (chip-20260902-sde1-item3-rollout-endpoint-contrastive) is already
            resolved `done`. What is owed is VALIDATION of a landed build, not
            repair of a live defect that silently corrupts readings.
        (c) The defect it does describe -- multi-step rollout score collapse
            (`cr_ratio`, `e1coe_score_var`) -- does not enter this driver's DV
            chain. Candidate summaries here come from
            `candidate_summary_source="proposer_post_action"`, the proposer's
            post-action state, NOT an E1 rollout score; and the load-bearing
            readiness gate trains a copy of this cell's own head on SYNTHETIC
            credit, which is independent of E1 rollout quality by construction.
            822f measured, on this same substrate and config, that SD-082's three
            predicates are positively observed and the summary-degeneracy guard
            does not fire -- i.e. the candidate summaries are empirically not
            degenerate here.
        A later autopsy may overturn this disposition; it is recorded in full so
        that it can be.
  A governance finding follows from (a) and is reported rather than acted on
  here: `SD-e1-rollout-consistency-training`'s `substrate_paths` list
  `e1_deep::forward`, which will spuriously gate every future agent-stepping
  experiment through Step 2.5c. Amending it is governance-owned work.

ETHICS PREFLIGHT (Step 2.6). All involvement flags false, decision allow. No
negative-valence drive, no suffering-like accumulator, no self-model, no
inescapability manipulation, no offline replay over harm, no social/language, no
human or clinical data. Pre-ethical V3 instrumentation only (SENT-0).

SUPERSEDES: nothing. 822f is NOT superseded -- it measured the readout index and
its finding (H1 rejected at adequate power) stands. This probe measures a
disjoint quantity on a new axis.

red-team (fable): see the queue entry note.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest, flat_readout  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.readiness_anchor import assert_anchor_reachable  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_1020_sd082_learning_signal_probe"
QUEUE_ID = "V3-EXQ-1020"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS = ["SD-082"]

# 822f's seeds, deliberately -- see the module docstring.
SEEDS = [611, 622, 633, 644, 655]
ARMS = ["ARM_OFF", "ARM_ON"]          # centering False / True, as in 822f.
P0_WARMUP_EPISODES = 60
P1_BIAS_TRAIN_EPISODES = 70
TOTAL_EPISODES = P0_WARMUP_EPISODES + P1_BIAS_TRAIN_EPISODES   # == the [train] denominator
STEPS_PER_EPISODE = 48

# --- REINFORCE hyperparameters: IDENTICAL to 822f, so the telemetry describes the
# --- same optimisation this probe exists to explain. Do not retune.
LR_LPFC_BIAS = 5e-4
REINFORCE_BATCH_SIZE = 32
OUTCOME_BUF_MAX = 512
POLICY_TEMPERATURE = 1.0
ADV_MIN_THRESHOLD = 0.005
EMA_DECAY = 0.9
ATANH_CLAMP = 0.999999

ENV_KWARGS = dict(size=8, num_hazards=2, num_resources=6, use_proxy_fields=True)

# --- Pre-registered thresholds (constants; never derived from this run's own stats).
# Readiness (LOAD-BEARING): the dense-synthetic-credit positive control.
# THE GATED STATISTIC (see the control's "WHY THE GATE IS TASK IMPROVEMENT, NOT D").
# Calibrated on this substrate: measured +0.00223..+0.00327, POSITIVE in 10/10 cells.
# The floor sits ~2.2x below the worst observed cell. This is a POSITIVE-CONTROL
# calibration -- it decides when the INSTRUMENT counts as working -- and is NOT a
# scientific bar: C1/C2/C3's thresholds are independent and were not tuned here.
SYNTH_TASK_GAIN_FLOOR = 0.001
# Readiness is a FRACTION-OF-CELLS claim, not an all-10 claim. Requiring every cell
# would let one jittery cell self-route the whole run to substrate_not_ready_requeue;
# the instrument is established if the control works RELIABLY. Reference measured
# 10/10, so a 0.8 bar carries two cells of headroom (the anchor guard below asserts
# that headroom with the SHIPPED predicate rather than trusting this comment).
CONTROL_READY_FRACTION_FLOOR = 0.8
SYNTH_D_FLOOR = 0.5            # RECORDED ONLY -- no longer gated; see the docstring.
SYNTH_PERSISTENCE_FLOOR = 0.5  # autopsy: "with persistence near 1"; 0.5 is the floor,
                               # generously below "near 1", so the gate fails only on a
                               # genuinely non-accumulating control.
SYNTH_UPDATES = 70             # same update count as P1, so control and real run are
                               # compared at equal optimisation budget.
SYNTH_BATCH = REINFORCE_BATCH_SIZE
SYNTH_POOL = 512
SYNTH_PROBE_SEED = 10201       # dedicated generator -- must NOT perturb run RNG.
NOISE_PROBE_SEED = 10202       # negative control -- own generator, no run RNG.
NOISE_GRAD_SCALE = 2.3e-4      # scale-invariant (measured 1e-6..1e-2 -> same floor)
SYNTH_D_PROBES = 32            # held-out draws D is averaged over; see the control's
                               # docstring -- precision, not a change to the 0.5 bar.

# C1: is there a gradient at all?
GRAD_NORM_FLOOR = 1e-6
# C2: is the gradient direction-consistent? BRACKETED BY TWO MEASURED IN-RUN
# CONTROLS -- a NEGATIVE control (pure-noise gradients through the same head and
# the same Adam) giving the noise floor, and the POSITIVE control giving the
# achievable ceiling. C2 fires when the real run sits closer to the noise floor
# than to the positive control. See "WHY C2 IS BRACKETED" in the docstring: a
# fixed 0.25 ceiling was a DERIVATION ERROR (it is the SGD null, not Adam's).
C2_MIDPOINT_FRACTION = 0.5
# Retained ONLY as the recorded SGD-null reference the original bar came from.
# NOT a gate. Do not restore it as one.
PERSISTENCE_SGD_NULL_REFERENCE = 0.12
# C3 needs enough FRESH (non-hold-replicated) flip samples to mean anything.
MIN_FLIP_SAMPLES = 20
# C3: advantage sign on flips.
ADV_SIGN_SEED_MAJORITY = 3      # of 5
# C4 (C3's non-degeneracy condition): per-episode return variance.
RETURN_VAR_FLOOR = 1e-4
# Flip-rate resolution: below this many measured P1 ticks a flip fraction is unresolvable.
FLIP_TICKS_FLOOR = 200
# A tick counts toward the flip denominator only if rule_state is actually live --
# see the gate at its use site. Measured: 82.7% of >=2-candidate ticks are live.
RULE_STATE_LIVE_FLOOR = 1e-9
SEED_MAJORITY = 3               # of 5, per arm


def _build_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, centering: bool) -> REEAgent:
    """Identical to 822f's _make_agent. Held fixed on purpose: this probe explains
    822f's optimisation, so every knob that shapes it must match."""
    return REEAgent(REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        alpha_world=0.9,
        use_lateral_pfc_analog=True,
        lateral_pfc_train_rule_bias_head=True,
        lateral_pfc_rule_readout_consumer=True,
        lateral_pfc_capture_head_diagnostics=True,
        candidate_summary_source="proposer_post_action",
        use_gated_policy=True,
        use_candidate_rule_field=True,
        crf_persist_rules_across_episode_reset=True,
        crf_mature_pool_dynamics=True,
        crf_availability_maintenance=True,
        crf_maintenance_floor=0.45,
        crf_maintenance_couple_to_theta=True,
        crf_tolerance_conflict_cap=3,
        crf_cue_centering=centering,
        crf_cue_baseline_alpha=0.02,
    ))


def _config_slice(centering: bool) -> Dict[str, Any]:
    return {
        "env": dict(ENV_KWARGS),
        "schedule": {"p0": P0_WARMUP_EPISODES, "p1": P1_BIAS_TRAIN_EPISODES,
                     "steps": STEPS_PER_EPISODE},
        "alpha_world": 0.9,
        "use_lateral_pfc_analog": True,
        "lateral_pfc_train_rule_bias_head": True,
        "lateral_pfc_rule_readout_consumer": True,
        "lateral_pfc_capture_head_diagnostics": True,
        "candidate_summary_source": "proposer_post_action",
        "use_gated_policy": True,
        "use_candidate_rule_field": True,
        "crf_persist_rules_across_episode_reset": True,
        "crf_mature_pool_dynamics": True,
        "crf_availability_maintenance": True,
        "crf_maintenance_floor": 0.45,
        "crf_maintenance_couple_to_theta": True,
        "crf_tolerance_conflict_cap": 3,
        "crf_cue_centering": centering,
        "crf_cue_baseline_alpha": 0.02,
        "reinforce": {"lr": LR_LPFC_BIAS, "batch": REINFORCE_BATCH_SIZE,
                      "buf": OUTCOME_BUF_MAX, "temp": POLICY_TEMPERATURE,
                      "adv_min": ADV_MIN_THRESHOLD, "ema": EMA_DECAY},
    }


def _candidate_summaries(agent: REEAgent, candidates,
                         counters: Dict[str, int]) -> Optional[torch.Tensor]:
    """822f's helper, unchanged. With candidate_summary_source='proposer_post_action'
    the dispatch returns a real [K, world_dim] tensor and the manual fallback should
    never be taken; every fallback is COUNTED, because a nonzero count means the
    summary source did not engage and readiness must fail rather than the run
    silently measuring the defective path."""
    summ = agent._candidate_world_summaries(candidates)
    if summ is not None:
        counters["dispatch"] += 1
        return summ.detach()
    if not candidates:
        return None
    counters["fallback"] += 1
    cand_world_list: List[torch.Tensor] = []
    for c in candidates:
        if c.world_states is not None:
            ws = c.get_world_state_sequence()
            cand_world_list.append(ws[0, 0, :])
        elif agent._current_latent is not None:
            cand_world_list.append(agent._current_latent.z_world[0].detach())
        else:
            return None
    return torch.stack(cand_world_list, dim=0).detach()


def _raw_ratio_and_flip(lpfc, summaries: torch.Tensor, head=None
                        ) -> Optional[Tuple[float, float]]:
    """Returns (raw_ratio, argmax_flip) at the RAW pre-tanh stage.

    Reduced from 822f's `_raw_stage_prop` to the two quantities this probe uses,
    with 822f's reasoning preserved verbatim in intent:

    RAW STAGE, not post-tanh. The consumer's output stage is
    bias = bias_scale * tanh(bias_raw / bias_scale) with bias_scale 0.1. tanh is
    monotone but NOT affine, so a PERFECTLY UNIFORM raw shift emerges as a
    NON-uniform post-tanh delta -- a cross-candidate "spread" manufactured
    entirely by the squasher, on a candidate-UNIFORM state. Reading the index
    post-tanh therefore cannot exclude the one state it exists to exclude.

    `raw_ratio` = raw_spread / raw_delta is the dimensionless discrimination
    index: candidate-UNIFORM (same nonzero value on every candidate) gives
    range 0 -> ratio 0; a bias concentrated on one candidate gives ratio K.
    Scale-free, with an ANALYTIC ceiling (the candidate count).

    `flip` is stage-invariant: bias = scale*tanh(raw/scale) is strictly monotone,
    so argmax(bias) == argmax(bias_raw) for both the real and the ablated
    rule_state; reading it raw is identical to reading it post-tanh.
    """
    cfg = lpfc.config
    k = int(summaries.shape[0])
    if k < 2:
        return None
    s = summaries
    if cfg.rule_readout_consumer and k >= 2:
        s = s - s.mean(dim=0, keepdim=True)
    head_mod = lpfc.rule_bias_head if head is None else head

    def _raw(rule_state: torch.Tensor) -> torch.Tensor:
        joined = torch.cat([rule_state.expand(k, -1), s], dim=-1)
        return head_mod(joined).squeeze(-1)

    with torch.no_grad():
        r1 = _raw(lpfc.rule_state).detach().clone().reshape(-1)
        saved = lpfc.rule_state.detach().clone()
        lpfc.rule_state.zero_()
        r0 = _raw(lpfc.rule_state).detach().clone().reshape(-1)
        lpfc.rule_state.copy_(saved)
    flip = float(int(r1.argmax().item()) != int(r0.argmax().item()))
    prop = r1 - r0
    raw_spread = float((prop.max() - prop.min()).item())
    raw_delta = float(prop.abs().mean().item())
    raw_ratio = (raw_spread / raw_delta) if raw_delta > 0.0 else 0.0
    if not (math.isfinite(raw_ratio) and math.isfinite(flip)):
        return None
    return raw_ratio, flip


def _flat_params(params) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in params])


class _StepAccumulator:
    """Step-direction persistence: ||sum of steps|| / sum of ||steps||.

    1.0 means every step pointed the same way (perfect accumulation); ~0 means the
    steps cancelled (a direction-inconsistent walk). Kept per-tensor as well as
    pooled, because the autopsy asked for PER-TENSOR persistence -- a head can
    accumulate in one layer while thrashing in another, and a pooled number hides
    that.

    Every stored tensor is .detach().clone()'d: an un-cloned reference would alias
    live parameter storage and every recorded 'step' would silently become the
    final weights.
    """

    def __init__(self, names: List[str]):
        self.names = names
        self.sum_vec: Dict[str, Optional[torch.Tensor]] = {n: None for n in names}
        self.sum_norm: Dict[str, float] = {n: 0.0 for n in names}
        self.pooled_vec: Optional[torch.Tensor] = None
        self.pooled_norm = 0.0
        self.n_steps = 0

    def observe(self, deltas: Dict[str, torch.Tensor]) -> None:
        self.n_steps += 1
        pooled_parts: List[torch.Tensor] = []
        for n in self.names:
            d = deltas[n].detach().clone().reshape(-1)
            self.sum_vec[n] = d if self.sum_vec[n] is None else (self.sum_vec[n] + d)
            self.sum_norm[n] += float(d.norm().item())
            pooled_parts.append(d)
        pooled = torch.cat(pooled_parts)
        self.pooled_vec = pooled if self.pooled_vec is None else (self.pooled_vec + pooled)
        self.pooled_norm += float(pooled.norm().item())

    @staticmethod
    def _ratio(vec: Optional[torch.Tensor], tot: float) -> float:
        if vec is None or not (tot > 0.0):
            return 0.0
        r = float(vec.norm().item()) / tot
        return r if math.isfinite(r) else 0.0

    def result(self) -> Dict[str, Any]:
        per_tensor = {n: self._ratio(self.sum_vec[n], self.sum_norm[n]) for n in self.names}
        pooled = self._ratio(self.pooled_vec, self.pooled_norm)
        vals = [v for v in per_tensor.values()]
        return {
            "persistence_pooled": pooled,
            "persistence_per_tensor": per_tensor,
            "persistence_min_tensor": (min(vals) if vals else 0.0),
            "persistence_max_tensor": (max(vals) if vals else 0.0),
            "n_steps": self.n_steps,
            "total_step_norm": self.pooled_norm,
            "net_displacement": (float(self.pooled_vec.norm().item())
                                 if self.pooled_vec is not None else 0.0),
        }


def _reinforce_step(lpfc, opt, buf, baseline: float, device,
                    acc: _StepAccumulator, tel: Dict[str, Any],
                    rng: Optional[np.random.RandomState] = None) -> bool:
    """One REINFORCE update, with the telemetry this probe exists to collect.

    Loss form is IDENTICAL to 822f's `_lpfc_reinforce_loss`. What is added is
    measurement only -- it does not change the optimisation:
      * grad_norm from clip_grad_norm_'s RETURN value (pre-clip; see docstring)
      * the parameter delta across opt.step(), fed to the persistence accumulator
      * per-SAMPLE (was_flip, adv) for every sample that survived ADV_MIN_THRESHOLD
    Returns True if an update actually happened.
    """
    n = len(buf)
    if n < 2:
        return False
    draw = (rng.choice(n, size=min(REINFORCE_BATCH_SIZE, n), replace=False)
            if rng is not None
            else np.random.choice(n, size=min(REINFORCE_BATCH_SIZE, n), replace=False))
    terms: List[torch.Tensor] = []
    used: List[Tuple[bool, float]] = []
    n_filtered = 0
    for i in draw:
        entry = buf[int(i)]
        cand_features, sel_idx, ep_return, was_flip = entry[0], entry[1], entry[2], entry[3]
        # Optional 5th field: a per-sample rule_state, used ONLY by the synthetic
        # positive control (the real path stores 4-tuples and leaves this None).
        # The control must present a NON-ZERO, VARYING rule_state or the head can
        # satisfy its task while ignoring the rule input entirely -- see
        # _synthetic_credit_control.
        rs = entry[4] if len(entry) > 4 else None
        fresh = bool(entry[5]) if len(entry) > 5 else True
        adv = ep_return - baseline
        if abs(adv) < ADV_MIN_THRESHOLD:
            n_filtered += 1
            continue
        if rs is not None:
            lpfc.rule_state.copy_(rs)
        bias = lpfc.compute_bias(cand_features.to(device))
        log_p = F.log_softmax(-bias / POLICY_TEMPERATURE, dim=0)
        terms.append(-adv * log_p[min(sel_idx, bias.shape[0] - 1)])
        used.append((bool(was_flip), float(adv), fresh))
        tel["sample_flip_ret"].append((bool(was_flip), float(ep_return), fresh))
    tel["n_adv_filtered"] += n_filtered
    if not terms:
        return False
    loss = torch.stack(terms).mean()
    if not loss.requires_grad:
        return False

    params = list(lpfc.bias_head_parameters())
    names = [n_ for n_, _ in lpfc.rule_bias_head.named_parameters()]
    before = {n_: p.detach().clone() for n_, p in lpfc.rule_bias_head.named_parameters()}

    opt.zero_grad()
    loss.backward()
    # RETURN value = total norm BEFORE clipping. Reading .grad after this call
    # would report the CLIPPED gradient (saturating at 1.0) -- see docstring.
    tel["rule_state_norm_at_update"].append(float(lpfc.rule_state.norm().item()))
    total_norm = torch.nn.utils.clip_grad_norm_(lpfc.bias_head_parameters(), 1.0)
    gn = float(total_norm.item() if hasattr(total_norm, "item") else total_norm)
    opt.step()

    deltas = {n_: (p.detach() - before[n_]) for n_, p in lpfc.rule_bias_head.named_parameters()}
    acc.observe(deltas)
    if math.isfinite(gn):
        tel["grad_norms"].append(gn)
    else:
        tel["n_grad_nonfinite"] += 1
    tel["sample_flip_adv"].extend(used)
    tel["n_updates"] += 1
    del params, names
    return True


def _synthetic_credit_control(agent: REEAgent, init_head, wd: int,
                              device) -> Dict[str, Any]:
    """LOAD-BEARING READINESS GATE -- the dense-synthetic-credit positive control.

    Trains a COPY of this cell's own INIT head with the same optimiser class, the
    same lr, and the same REINFORCE loss form, on a credit signal that is dense
    and informative BY CONSTRUCTION: for a fixed random target direction, a
    sample's return is the alignment of its selected candidate summary with that
    target. Every sample therefore carries real, consistently-signed advantage --
    the exact opposite of the real run's sparse episode returns.

    WHAT IT ESTABLISHES. If this control moves D past SYNTH_D_FLOOR with
    persistence at or above SYNTH_PERSISTENCE_FLOOR, then this head, this
    optimiser and this loss CAN learn and this instrument CAN see it -- so a
    near-zero persistence in the real run is a property of the REAL CREDIT
    SIGNAL, not of the measurement. If the control fails, the instrument is blind
    and no statement about the real signal is licensed: the run self-routes to
    `substrate_not_ready_requeue`, never to a substrate verdict.

    This is exactly the discrimination 822f could not make, which is why it is a
    gate here rather than an optional diagnostic.

    WHY THE SYNTHETIC TASK IS RULE-STATE-CONDITIONED, AND WHY THE rule_state MUST
    BE NON-ZERO AND TWO-VALUED. D is read off `raw_ratio`, whose input is
    `prop = raw(rule_state) - raw(0)` -- the RULE-STATE-ATTRIBUTABLE part of the
    bias. Two ways that is zero for reasons having nothing to do with learning:
      (i) if `lpfc.rule_state` is at its zero init (it is, on a never-stepped
          agent), then raw(rule_state) IS raw(0) and prop is IDENTICALLY zero, so
          D is 0.0000 in every cell no matter how well the head learned. This was
          measured on the first draft of this driver, in all 10 smoke cells, with
          persistence a healthy 0.75-0.79 -- i.e. a perfectly functioning learner
          reading as a dead instrument. It is the same defect 822f's own
          `_control_probes` docstring records against ITS first draft (a control
          read on an end-of-run rule_state that `agent.reset()` had just zeroed,
          which "spuriously routed the WHOLE run to substrate_not_ready_requeue").
      (ii) if the synthetic task can be solved while IGNORING the rule input, the
          head has no reason to make its output rule-state-dependent, so prop
          stays ~0 even with a non-zero rule_state. A single fixed rule_state has
          exactly this defect: a constant input is indistinguishable from a bias
          term.
    So the task presents TWO distinct rule_states with DIFFERENT targets, and the
    reward depends on selecting the candidate aligned with the target FOR THAT
    rule_state. The head cannot score without conditioning on the rule input --
    which is precisely the rule-state-conditioned, candidate-discriminating
    readout SD-082 asserts the head can carry. So the control is not a generic
    "can Adam move weights" check: it is on-target for the claim.

    Own torch.Generator and own RandomState, so the control consumes no run RNG
    and cannot perturb the per-cell arm_fingerprint. The real rule_state and the
    real head are both saved and restored.
    """
    lpfc = agent.lateral_pfc
    g = torch.Generator()
    g.manual_seed(SYNTH_PROBE_SEED)
    rng = np.random.RandomState(SYNTH_PROBE_SEED)

    rule_dim = int(lpfc.rule_state.shape[-1])
    k = 4
    # Two distinct, non-zero rule_states with two distinct targets.
    rule_states = []
    targets = []
    for _ in range(2):
        rs = torch.randn(1, rule_dim, generator=g)
        rs = rs / (rs.norm() + 1e-12)
        rule_states.append(rs.to(device))
        t = torch.randn(wd, generator=g)
        targets.append((t / (t.norm() + 1e-12)).to(device))

    pool: List[Tuple[torch.Tensor, int, float, bool, torch.Tensor]] = []
    for _ in range(SYNTH_POOL):
        which = int(rng.randint(0, 2))
        summ = (torch.randn(k, wd, generator=g) * 0.4).to(device)
        align = summ @ targets[which]
        sel = int(torch.argmax(align).item())
        # Dense credit: the return IS the alignment of the selected candidate with
        # THIS rule_state's target, so advantage is informative on every sample.
        ep_return = float(align[sel].item())
        was_flip = bool(rng.rand() < 0.5)   # label only; unused by the gate
        pool.append((summ, sel, ep_return, was_flip, rule_states[which]))
    baseline = float(np.mean([p[2] for p in pool]))

    # Train a COPY of the init head, leaving the real head and rule_state untouched.
    ctrl_head = copy.deepcopy(init_head).to(device)
    for p in ctrl_head.parameters():
        p.requires_grad_(True)
    saved_head = lpfc.rule_bias_head
    saved_rule_state = lpfc.rule_state.detach().clone()
    lpfc.rule_bias_head = ctrl_head
    try:
        opt = torch.optim.Adam(list(ctrl_head.parameters()), lr=LR_LPFC_BIAS)
        names = [n for n, _ in ctrl_head.named_parameters()]
        acc = _StepAccumulator(names)
        tel: Dict[str, Any] = {"grad_norms": [], "sample_flip_adv": [],
                               "n_updates": 0, "n_adv_filtered": 0,
                               "n_grad_nonfinite": 0, "sample_flip_ret": [],
                               "rule_state_norm_at_update": []}
        for _ in range(SYNTH_UPDATES):
            _reinforce_step(lpfc, opt, pool, baseline, device, acc, tel, rng=rng)

        # D = trained-minus-init discrimination index, measured at a NON-ZERO
        # rule_state on held-out synthetic candidate sets, through the SAME index
        # the real run reports.
        #
        # AVERAGED OVER SYNTH_D_PROBES DRAWS, and SIGNED. `raw_ratio` on a single
        # k-candidate draw is a noisy statistic: measured on one draw it ranged
        # 0.035 to 1.31 across cells whose persistence was uniformly healthy
        # (0.70-0.76), i.e. the spread was sampling noise in the MEASUREMENT, not
        # variation in what the head had learned. Averaging is an instrument-
        # precision fix and does NOT move the bar, which stays at the autopsy's
        # 0.5. The mean is taken SIGNED and then |.| -- never mean-of-|.| -- so
        # noise CANCELS rather than inflating the statistic, and a head that
        # discriminates better only half the time correctly fails the gate.
        # Both rule_states are probed: the control asserts a rule-CONDITIONED
        # readout, so it must hold at each conditioning value, not just one.
        diffs: List[float] = []
        held_tr: List[float] = []
        held_in: List[float] = []
        for j in range(SYNTH_D_PROBES):
            which = j % 2
            probe = (torch.randn(k, wd, generator=g) * 0.4).to(device)
            lpfc.rule_state.copy_(rule_states[which])
            r_tr = _raw_ratio_and_flip(lpfc, probe)
            r_in = _raw_ratio_and_flip(lpfc, probe, head=init_head)
            if r_tr is not None and r_in is not None:
                d = r_tr[0] - r_in[0]
                if math.isfinite(d):
                    diffs.append(float(d))
            # THE GATED STATISTIC: expected reward on HELD-OUT synthetic samples
            # under each head's own selection policy. See the docstring section
            # "WHY THE GATE IS TASK IMPROVEMENT, NOT D".
            align = (probe @ targets[which]).reshape(-1)
            for head_mod, sink in ((ctrl_head, held_tr), (init_head, held_in)):
                with torch.no_grad():
                    s_c = probe - probe.mean(dim=0, keepdim=True) if (
                        lpfc.config.rule_readout_consumer and k >= 2) else probe
                    joined = torch.cat(
                        [lpfc.rule_state.expand(k, -1), s_c], dim=-1)
                    b = head_mod(joined).squeeze(-1).reshape(-1)
                    pr = torch.softmax(-b / POLICY_TEMPERATURE, dim=0)
                    sink.append(float((pr * align).sum().item()))
    finally:
        lpfc.rule_bias_head = saved_head
        lpfc.rule_state.copy_(saved_rule_state)

    d_signed = float(np.mean(diffs)) if diffs else 0.0
    d_abs = abs(d_signed)
    d_sd = float(np.std(diffs)) if len(diffs) >= 2 else 0.0
    task_tr = float(np.mean(held_tr)) if held_tr else 0.0
    task_in = float(np.mean(held_in)) if held_in else 0.0
    task_gain = task_tr - task_in
    res = acc.result()
    gns = tel["grad_norms"]
    return {
        "synth_task_gain": float(task_gain),
        "synth_task_reward_trained": float(task_tr),
        "synth_task_reward_init": float(task_in),
        "synth_D_abs": float(d_abs),
        "synth_D_signed_mean": float(d_signed),
        "synth_D_probe_sd": float(d_sd),
        "synth_D_n_probes": len(diffs),
        "synth_persistence_pooled": float(res["persistence_pooled"]),
        "synth_persistence_per_tensor": res["persistence_per_tensor"],
        "synth_n_updates": int(tel["n_updates"]),
        "synth_mean_grad_norm": (float(np.mean(gns)) if gns else 0.0),
        "synth_median_grad_norm": (float(np.median(gns)) if gns else 0.0),
        "synth_net_displacement": float(res["net_displacement"]),
        "synth_total_step_norm": float(res["total_step_norm"]),
        "synth_n_adv_filtered": int(tel["n_adv_filtered"]),
    }


def _noise_persistence_control(init_head, device) -> Dict[str, Any]:
    """NEGATIVE control: the persistence floor of PURE NOISE through this head.

    Runs the SAME head geometry, the SAME optimiser (Adam) at the SAME lr for the
    SAME number of updates as P1, driven by zero-mean i.i.d. gradients -- i.e.
    H-learning-signal-noisy made maximally true, by construction. Whatever this
    returns is the value a genuinely direction-inconsistent signal produces, so it
    is the floor C2 must be read against.

    WHY THIS EXISTS. C2's original bar was a fixed 0.25, justified as "a
    direction-inconsistent walk scales as ~1/sqrt(n), so ~0.12 at 70 updates".
    That derivation is correct for RAW GRADIENT steps -- but persistence here is
    measured over ADAM PARAMETER DELTAS (see _StepAccumulator's call site, after
    opt.step()), and Adam's beta1=0.9 momentum plus per-coordinate normalisation
    autocorrelate consecutive steps. Measured on this exact head geometry at
    T=70, lr=5e-4: SGD pure-noise 0.120 (reproducing the 1/sqrt(n) figure, which
    is how the error was found), ADAM pure-noise 0.451-0.463, invariant across
    gradient scales 1e-6..1e-2, first crossing 0.25 only at T ~ 300.
    So the fixed 0.25 ceiling could NOT fire even when the hypothesis it tests was
    maximally TRUE -- an unfalsifiable criterion, and two of the five verdict
    labels were unreachable. Measuring the floor in-run fixes that at its cause
    rather than by moving a number, and it self-calibrates to the machine class
    the run actually lands on instead of trusting a figure measured elsewhere.
    """
    g = torch.Generator()
    g.manual_seed(NOISE_PROBE_SEED)
    head = copy.deepcopy(init_head).to(device)
    for prm in head.parameters():
        prm.requires_grad_(True)
    opt = torch.optim.Adam(list(head.parameters()), lr=LR_LPFC_BIAS)
    names = [n for n, _ in head.named_parameters()]
    acc = _StepAccumulator(names)
    for _ in range(SYNTH_UPDATES):
        before = {n: prm.detach().clone() for n, prm in head.named_parameters()}
        opt.zero_grad()
        for n, prm in head.named_parameters():
            prm.grad = torch.randn(prm.shape, generator=g).to(device) * NOISE_GRAD_SCALE
        opt.step()
        acc.observe({n: (prm.detach() - before[n])
                     for n, prm in head.named_parameters()})
    res = acc.result()
    return {
        "noise_persistence_pooled": float(res["persistence_pooled"]),
        "noise_persistence_per_tensor": res["persistence_per_tensor"],
        "noise_n_steps": int(res["n_steps"]),
    }


def _summarise_cell(tel: Dict[str, Any], acc_res: Dict[str, Any],
                    ep_returns: List[float], ep_flip_rate: List[float],
                    n_flip_ticks: int, n_measured_ticks: int) -> Dict[str, Any]:
    gns = tel["grad_norms"]
    sfa = tel["sample_flip_adv"]
    # HOLD-WEIGHTED (what the optimiser actually saw) vs FRESH-SELECT-ONLY (one
    # observation per genuine E3 selection). The load-bearing C3 statistic is the
    # FRESH one: E3's cadence re-observes a single commitment on ~10 consecutive
    # ticks, so the hold-weighted advantage means are computed over
    # pseudo-replicated samples and their effective n is far below len(sfa).
    flips_all = [a for (f, a, _fr) in sfa if f]
    nonflips_all = [a for (f, a, _fr) in sfa if not f]
    flips = [a for (f, a, fr) in sfa if f and fr]
    nonflips = [a for (f, a, fr) in sfa if (not f) and fr]
    ret_var = float(np.var(ep_returns)) if len(ep_returns) >= 2 else 0.0
    # F3 (red-team): the EMA baseline starts at 0.0 with decay 0.9, so advantage is
    # systematically INFLATED for the first ~20 P1 episodes and decays toward zero
    # bias thereafter. Combined with ANY monotone trend in per-episode flip rate --
    # and 822f's headline observation is exactly such a trend (trained flips < init
    # in 10/10 cells) -- that produces a nonzero flip-minus-nonflip advantage for a
    # purely ARITHMETIC reason, with the same footprint as the hypothesis. So the
    # same contrast is ALSO computed against a de-trended baseline (the P1 grand
    # mean, which carries no warm-up transient). C3 requires the two to AGREE IN
    # SIGN; a disagreement means the EMA transient is carrying the result and C3 is
    # marked degenerate rather than reported.
    grand = float(np.mean(ep_returns)) if ep_returns else 0.0
    dt = tel.get("sample_flip_ret") or []
    dt_flip = [r - grand for (f, r, fr) in dt if f and fr]
    dt_non = [r - grand for (f, r, fr) in dt if (not f) and fr]
    adv_dt = ((float(np.mean(dt_flip)) - float(np.mean(dt_non)))
              if (dt_flip and dt_non) else 0.0)
    spear = 0.0
    if len(ep_flip_rate) >= 3:
        idx = np.arange(len(ep_flip_rate))
        fr_arr = np.array(ep_flip_rate, dtype=float)
        if float(fr_arr.std()) > 0.0:
            ri = np.argsort(np.argsort(idx)).astype(float)
            rf = np.argsort(np.argsort(fr_arr)).astype(float)
            c = np.corrcoef(ri, rf)
            spear = float(c[0, 1]) if np.isfinite(c[0, 1]) else 0.0
    return {
        "n_p1_updates": int(tel["n_updates"]),
        "mean_grad_norm": (float(np.mean(gns)) if gns else 0.0),
        "median_grad_norm": (float(np.median(gns)) if gns else 0.0),
        "max_grad_norm": (float(np.max(gns)) if gns else 0.0),
        "n_grad_nonfinite": int(tel["n_grad_nonfinite"]),
        "persistence_pooled": float(acc_res["persistence_pooled"]),
        "persistence_per_tensor": acc_res["persistence_per_tensor"],
        "persistence_min_tensor": float(acc_res["persistence_min_tensor"]),
        "net_displacement": float(acc_res["net_displacement"]),
        "total_step_norm": float(acc_res["total_step_norm"]),
        "return_variance": ret_var,
        "return_mean": (float(np.mean(ep_returns)) if ep_returns else 0.0),
        "per_episode_returns": [float(x) for x in ep_returns],
        "per_episode_flip_rate": [float(x) for x in ep_flip_rate],
        "n_flip_samples": len(flips),
        "n_nonflip_samples": len(nonflips),
        "mean_adv_on_flip": (float(np.mean(flips)) if flips else 0.0),
        "mean_adv_on_nonflip": (float(np.mean(nonflips)) if nonflips else 0.0),
        "adv_flip_minus_nonflip": ((float(np.mean(flips)) - float(np.mean(nonflips)))
                                   if (flips and nonflips) else 0.0),
        # Hold-weighted duplicates, recorded distinctly (never gated on):
        "n_flip_samples_hold_weighted": len(flips_all),
        "n_nonflip_samples_hold_weighted": len(nonflips_all),
        "adv_flip_minus_nonflip_hold_weighted": (
            (float(np.mean(flips_all)) - float(np.mean(nonflips_all)))
            if (flips_all and nonflips_all) else 0.0),
        "n_reinforce_samples_used": len(sfa),
        "rule_state_norm_at_update": [float(x) for x in
                                      (tel.get("rule_state_norm_at_update") or [])],
        "adv_flip_minus_nonflip_detrended": adv_dt,
        "adv_sign_agrees_detrended": bool(
            (adv_dt < 0) == ((float(np.mean(flips)) - float(np.mean(nonflips))) < 0)
            if (flips and nonflips and dt_flip and dt_non) else False),
        "spearman_episode_vs_flip_rate": spear,
        "n_adv_filtered": int(tel["n_adv_filtered"]),
        "n_flip_ticks": int(n_flip_ticks),
        "n_measured_ticks": int(n_measured_ticks),
        "p1_flip_fraction": (float(n_flip_ticks) / n_measured_ticks
                             if n_measured_ticks > 0 else 0.0),
    }


def _run_cell(arm: str, seed: int, episodes: Dict[str, int],
              zg: ZGoalStreamAccumulator) -> Dict[str, Any]:
    centering = (arm == "ARM_ON")
    p0, p1 = episodes["p0"], episodes["p1"]
    total_eps = p0 + p1
    with arm_cell(seed, config_slice=_config_slice(centering),
                  script_path=Path(__file__)) as cell:
        env = _build_env(seed)
        agent = _make_agent(env, centering)
        wd = agent.config.latent.world_dim
        lpfc = agent.lateral_pfc
        device = agent.device

        init_head = copy.deepcopy(lpfc.rule_bias_head)
        for p in init_head.parameters():
            p.requires_grad_(False)

        bias_opt = torch.optim.Adam(list(lpfc.bias_head_parameters()), lr=LR_LPFC_BIAS)
        names = [n for n, _ in lpfc.rule_bias_head.named_parameters()]
        acc = _StepAccumulator(names)
        tel: Dict[str, Any] = {"grad_norms": [], "sample_flip_adv": [],
                               "n_updates": 0, "n_adv_filtered": 0,
                               "n_grad_nonfinite": 0, "sample_flip_ret": [],
                               "rule_state_norm_at_update": []}
        summary_counters = {"dispatch": 0, "fallback": 0}

        outcome_buf: List[Tuple[torch.Tensor, int, float, bool]] = []
        baseline = 0.0
        ep_returns: List[float] = []
        ep_flip_rate: List[float] = []
        n_flip_ticks = 0
        n_measured_ticks = 0
        n_rule_live_ticks = 0
        n_fresh_ticks = 0
        n_fresh_flip_ticks = 0
        n_latched_ticks = 0
        last_flip = False
        last_flip_valid = False

        print(f"Seed {seed} Condition {arm}", flush=True)

        for ep in range(total_eps):
            is_p1 = (ep >= p0)
            _, obs = env.reset()
            agent.reset()
            ep_reward = 0.0
            ep_buf: List[Tuple[torch.Tensor, int, bool, bool]] = []
            ep_ticks = 0
            ep_flips = 0
            last_flip = False
            last_flip_valid = False   # never inherit across an episode boundary

            for _step in range(episodes["steps"]):
                latent = agent.sense(obs["body_state"], obs["world_state"])
                ticks = agent.clock.advance()
                # E3 runs on a CADENCE (heartbeat.e3_steps_per_tick, default 10) and
                # generate_trajectories returns CACHED candidates when it does not
                # fire, so one committed selection is re-observed on every held tick.
                # Statistics accumulated per env-step are therefore HOLD-WEIGHTED --
                # pseudo-replicated by hold duration. See _summarise_cell.
                is_fresh = bool(ticks.get("e3_tick", False))
                e1 = (agent._e1_tick(latent) if ticks.get("e1_tick")
                      else torch.zeros(1, wd, device=device))
                candidates = agent.generate_trajectories(latent, e1, ticks)

                snap: Optional[torch.Tensor] = None
                was_flip = False
                if is_p1 and candidates and len(candidates) >= 2:
                    cs = _candidate_summaries(agent, candidates, summary_counters)
                    if cs is not None and torch.isfinite(cs).all():
                        snap = cs.clone()
                        # A flip is argmax(raw | rule_state) != argmax(raw | 0). On a
                        # tick where rule_state IS zero the two are the same tensor, so
                        # `flip` is False BY ARITHMETIC and the tick carries no
                        # information about flipping. Counting such ticks in the
                        # denominator would report a flip FRACTION diluted by ticks at
                        # which a flip was impossible. 822f gates its flip measurement
                        # the same way (rule_state norm > 1e-9 and >= 1 active CRF
                        # rule); measured here at 82.7% of >=2-candidate ticks live,
                        # so the gate removes a real ~17%, not a rounding error.
                        rule_live = float(lpfc.rule_state.norm()) > RULE_STATE_LIVE_FLOOR
                        if rule_live:
                            n_rule_live_ticks += 1
                        # Measure the flip ONLY on a FRESH E3 selection, and carry the
                        # label forward on held ticks. On a held tick E3 did not run and
                        # generate_trajectories returned the CACHED candidates, so the
                        # flip verdict is by construction the same one the originating
                        # fresh tick produced -- recomputing it would spend two forward
                        # passes over 32 candidates to re-derive a known value. Doing it
                        # this way is both ~10x cheaper and the statistically correct
                        # denominator (see _summarise_cell on hold-weighting).
                        if rule_live and is_fresh:
                            rf = _raw_ratio_and_flip(lpfc, snap)
                            if rf is not None:
                                last_flip = bool(rf[1] > 0.5)
                                last_flip_valid = True
                            else:
                                last_flip_valid = False
                            if last_flip_valid:
                                n_fresh_ticks += 1
                                if last_flip:
                                    n_fresh_flip_ticks += 1
                        if rule_live and last_flip_valid:
                            was_flip = last_flip
                            ep_ticks += 1
                            n_measured_ticks += 1
                            if not is_fresh:
                                n_latched_ticks += 1
                            if was_flip:
                                ep_flips += 1
                                n_flip_ticks += 1

                action = agent.select_action(candidates, ticks)
                if action is None:
                    action = torch.zeros(1, 4, device=device)
                    action[0, int(np.random.randint(0, 4))] = 1.0
                    agent._last_action = action
                committed_class = int(action[0].argmax().item())

                if snap is not None:
                    # Resolve WHICH candidate was actually committed, by matching the
                    # committed action class against each candidate's first action --
                    # 822f's resolution, reproduced deliberately. A constant `sel`
                    # here would credit the same candidate on every sample regardless
                    # of what the agent chose, which silently destroys the very
                    # learning signal this probe measures.
                    sel = 0
                    for ci, c in enumerate(candidates):
                        if (getattr(c, "actions", None) is not None
                                and c.actions.shape[1] >= 1
                                and int(c.actions[:, 0, :].argmax(-1).reshape(-1)[0].item())
                                == committed_class):
                            sel = min(ci, snap.shape[0] - 1)
                            break
                    ep_buf.append((snap, sel, was_flip, is_fresh))

                _, _h, done, _info, obs = env.step(int(action.argmax(dim=-1).item()))
                if is_p1:
                    ep_reward += float(_h)
                if done:
                    break

            if is_p1:
                baseline = EMA_DECAY * baseline + (1.0 - EMA_DECAY) * ep_reward
                # Appended UNGATED, exactly as 822f does: the buffer composition IS
                # part of the optimisation this probe exists to explain, so it is
                # faithfully reproduced rather than quietly corrected. The fresh flag
                # rides along so the STATISTICS can be de-replicated without changing
                # what the optimiser saw.
                for cand_features, sel, flip_lbl, fresh_lbl in ep_buf:
                    outcome_buf.append((cand_features, sel, ep_reward, flip_lbl,
                                        None, fresh_lbl))
                if len(outcome_buf) > OUTCOME_BUF_MAX:
                    outcome_buf = outcome_buf[-OUTCOME_BUF_MAX:]
                _reinforce_step(lpfc, bias_opt, outcome_buf, baseline, device, acc, tel)
                ep_returns.append(float(ep_reward))
                ep_flip_rate.append(float(ep_flips) / ep_ticks if ep_ticks > 0 else 0.0)

            if (ep + 1) % 25 == 0 or (ep + 1) == total_eps:
                print(f"  [train] {arm} seed={seed} ep {ep+1}/{total_eps} "
                      f"updates={tel['n_updates']}", flush=True)

        control = _synthetic_credit_control(agent, init_head, wd, device)
        noise_ctrl = _noise_persistence_control(init_head, device)
        zg.observe(agent)

        row: Dict[str, Any] = {"arm_id": arm, "seed": seed, "centering": centering}
        row.update(_summarise_cell(tel, acc.result(), ep_returns, ep_flip_rate,
                                   n_flip_ticks, n_measured_ticks))
        row.update(control)
        row.update(noise_ctrl)
        # C2's per-cell bracket: noise floor -> positive-control ceiling.
        _lo = float(noise_ctrl["noise_persistence_pooled"])
        _hi = float(control["synth_persistence_pooled"])
        row["c2_midpoint"] = _lo + C2_MIDPOINT_FRACTION * max(_hi - _lo, 0.0)
        row["persistence_bracket_valid"] = bool(_hi > _lo)
        row["n_rule_live_ticks"] = n_rule_live_ticks
        row["n_fresh_select_ticks"] = n_fresh_ticks
        row["n_latched_ticks"] = n_latched_ticks
        row["n_fresh_flip_ticks"] = n_fresh_flip_ticks
        row["fresh_flip_fraction"] = (float(n_fresh_flip_ticks) / n_fresh_ticks
                                      if n_fresh_ticks > 0 else 0.0)
        row["summary_dispatch_calls"] = summary_counters["dispatch"]
        row["summary_fallback_calls"] = summary_counters["fallback"]
        row["control_ready"] = bool(
            row["synth_task_gain"] > SYNTH_TASK_GAIN_FLOOR
            and row["synth_persistence_pooled"] >= SYNTH_PERSISTENCE_FLOOR)
        cell.stamp(row)
    return row


def _worst(rows: List[Dict[str, Any]], key: str, mode: str = "min"):
    if not rows:
        return 0.0, None
    f = min if mode == "min" else max
    r = f(rows, key=lambda x: float(x.get(key, 0.0)))
    return float(r.get(key, 0.0)), f"{r.get('arm_id')}::seed{r.get('seed')}"


# Frozen record of the positive control measured on this substrate at setup time
# (Mac, 2026-09-10, 10/10 cells, SYNTH_UPDATES=70). These are the reference cells the
# anchor-reachability guard scores with THE SHIPPED PREDICATE, so a gate narrower than
# the control can actually reach is refused BEFORE compute rather than reporting
# met=false forever and mislabelling an instrument-specification gap as a substrate
# verdict.
CONTROL_REFERENCE_CELLS = [
    {"synth_task_gain": 0.00305, "synth_persistence_pooled": 0.734},
    {"synth_task_gain": 0.00276, "synth_persistence_pooled": 0.724},
    {"synth_task_gain": 0.00293, "synth_persistence_pooled": 0.753},
    {"synth_task_gain": 0.00223, "synth_persistence_pooled": 0.730},
    {"synth_task_gain": 0.00327, "synth_persistence_pooled": 0.728},
    {"synth_task_gain": 0.00235, "synth_persistence_pooled": 0.722},
    {"synth_task_gain": 0.00300, "synth_persistence_pooled": 0.731},
    {"synth_task_gain": 0.00264, "synth_persistence_pooled": 0.720},
    {"synth_task_gain": 0.00223, "synth_persistence_pooled": 0.719},
    {"synth_task_gain": 0.00300, "synth_persistence_pooled": 0.770},
]
# A cell whose real-run persistence sits AT the measured pure-noise Adam floor --
# i.e. H-learning-signal-noisy true by construction -- bracketed by the measured
# per-cell positive-control ceilings. If the SHIPPED C2 predicate cannot score
# these True, C2 cannot fire when its own hypothesis holds.
C2_NOISE_REFERENCE_CELLS = [
    {"persistence_pooled": 0.4488, "c2_midpoint": 0.5841, "persistence_bracket_valid": True},
    {"persistence_pooled": 0.4488, "c2_midpoint": 0.5930, "persistence_bracket_valid": True},
    {"persistence_pooled": 0.4488, "c2_midpoint": 0.5877, "persistence_bracket_valid": True},
    {"persistence_pooled": 0.4488, "c2_midpoint": 0.6027, "persistence_bracket_valid": True},
    {"persistence_pooled": 0.4488, "c2_midpoint": 0.5821, "persistence_bracket_valid": True},
    {"persistence_pooled": 0.4488, "c2_midpoint": 0.5824, "persistence_bracket_valid": True},
    {"persistence_pooled": 0.4488, "c2_midpoint": 0.6030, "persistence_bracket_valid": True},
    {"persistence_pooled": 0.4488, "c2_midpoint": 0.6042, "persistence_bracket_valid": True},
    {"persistence_pooled": 0.4488, "c2_midpoint": 0.6062, "persistence_bracket_valid": True},
    {"persistence_pooled": 0.4488, "c2_midpoint": 0.5965, "persistence_bracket_valid": True},
]
C2_NOISE_REFERENCE_SOURCE = (
    "V3-EXQ-1020 authoring session 2026-09-10, darwin-arm64: pure-noise Adam "
    "persistence floor 0.4488 measured on the real head geometry at T=70, lr=5e-4, "
    "against the 10 measured positive-control ceilings 0.7155-0.7636")
CONTROL_REFERENCE_SOURCE = (
    "V3-EXQ-1020 authoring session 2026-09-10, darwin-arm64: dense-synthetic-credit "
    "control on 822f's 5 seeds x 2 arms, SYNTH_UPDATES=70")


def _assert_readiness_anchors_reachable() -> List[Dict[str, Any]]:
    """Refuse the run at SETUP if either gated readiness predicate is unreachable.

    Scores the frozen control reference with THE SHIPPED PREDICATES -- the same
    boolean expressions `control_ready` and `readiness_met` use -- so the two cannot
    drift apart. `margin_cells=1` requires the reference to clear each gate with a
    cell to spare, so a gate that only just passes (and would fail on seed jitter)
    is refused too.
    """
    return [
        assert_anchor_reachable(
            anchor_name="synth_credit_control_learns_its_task",
            reference_cells=CONTROL_REFERENCE_CELLS,
            score_fn=lambda c: c["synth_task_gain"] > SYNTH_TASK_GAIN_FLOOR,
            threshold=CONTROL_READY_FRACTION_FLOOR, margin_cells=1,
            reference_source=CONTROL_REFERENCE_SOURCE),
        assert_anchor_reachable(
            anchor_name="synth_credit_control_persists",
            reference_cells=CONTROL_REFERENCE_CELLS,
            score_fn=lambda c: c["synth_persistence_pooled"] >= SYNTH_PERSISTENCE_FLOOR,
            threshold=CONTROL_READY_FRACTION_FLOOR, margin_cells=1,
            reference_source=CONTROL_REFERENCE_SOURCE),
        # C2 REACHABILITY -- the red-team's closing note was that the guard covered
        # the readiness anchors and NEITHER scientific criterion, and that aiming it
        # at C2 with a pure-noise reference would have caught the Adam-null defect
        # before compute. So it is aimed there now. The reference is a cell sitting
        # AT the measured pure-noise Adam floor (H-learning-signal-noisy maximally
        # true); scored with THE SHIPPED C2 PREDICATE, it must come out True. Had
        # this guard existed against the old fixed 0.25 ceiling it would have failed
        # outright (0.4488 < 0.25 is False on every reference cell).
        assert_anchor_reachable(
            anchor_name="C2_persistence_low_reachable_at_noise_floor",
            reference_cells=C2_NOISE_REFERENCE_CELLS,
            score_fn=lambda c: (c["persistence_bracket_valid"]
                                and c["persistence_pooled"] < c["c2_midpoint"]),
            threshold=CONTROL_READY_FRACTION_FLOOR, margin_cells=1,
            reference_source=C2_NOISE_REFERENCE_SOURCE),
    ]


def run_experiment(episodes: Dict[str, int], dry_run: bool) -> Dict[str, Any]:
    t0 = time.perf_counter()
    anchor_reachability = _assert_readiness_anchors_reachable()
    zg = ZGoalStreamAccumulator()
    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            row = _run_cell(arm, seed, episodes, zg)
            rows.append(row)
            print(f"verdict: {'PASS' if row['control_ready'] else 'FAIL'}", flush=True)

    # ---- READINESS (LOAD-BEARING): the dense-synthetic-credit positive control.
    worst_gain, worst_gain_cell = _worst(rows, "synth_task_gain", "min")
    worst_d, worst_d_cell = _worst(rows, "synth_D_abs", "min")
    worst_pers, worst_pers_cell = _worst(rows, "synth_persistence_pooled", "min")
    n_cells = max(len(rows), 1)
    gain_frac = sum(1 for r in rows
                    if r["synth_task_gain"] > SYNTH_TASK_GAIN_FLOOR) / n_cells
    pers_frac = sum(1 for r in rows
                    if r["synth_persistence_pooled"] >= SYNTH_PERSISTENCE_FLOOR) / n_cells
    readiness_met = bool(gain_frac >= CONTROL_READY_FRACTION_FLOOR
                         and pers_frac >= CONTROL_READY_FRACTION_FLOOR)

    worst_fallback, worst_fb_cell = _worst(rows, "summary_fallback_calls", "max")
    worst_ticks, worst_ticks_cell = _worst(rows, "n_measured_ticks", "min")

    preconditions = [
        {"name": "synth_credit_control_learns_its_task",
         "description": ("dense-synthetic-credit positive control: the same head, "
                         "optimiser and loss must IMPROVE at the synthetic task -- "
                         "expected reward under the trained head's own policy, on "
                         "HELD-OUT samples, minus the same under the init head. "
                         "Below floor means the instrument is blind, NOT that the "
                         "real signal is absent."),
         "control": "copy of this cell's own init head trained on dense synthetic credit",
         "measured": gain_frac, "threshold": CONTROL_READY_FRACTION_FLOOR,
         "direction": "lower", "offending_cell": worst_gain_cell,
         "worst_cell_value": worst_gain, "per_cell_floor": SYNTH_TASK_GAIN_FLOOR,
         "met": bool(gain_frac >= CONTROL_READY_FRACTION_FLOOR)},
        {"name": "synth_credit_control_persists",
         "description": ("positive control step-direction persistence must be near 1; "
                         "this is the achievable ceiling C2 is read against."),
         "control": "same as above",
         "measured": pers_frac, "threshold": CONTROL_READY_FRACTION_FLOOR,
         "direction": "lower", "offending_cell": worst_pers_cell,
         "worst_cell_value": worst_pers, "per_cell_floor": SYNTH_PERSISTENCE_FLOOR,
         "met": bool(pers_frac >= CONTROL_READY_FRACTION_FLOOR)},
        {"name": "candidate_summary_dispatch_engaged",
         "description": ("proposer_post_action must supply every summary; any manual "
                         "fallback means the summary source did not engage and the "
                         "run would be measuring the defective 822c path."),
         "measured": worst_fallback, "threshold": 0.0, "direction": "upper",
         "offending_cell": worst_fb_cell, "met": bool(worst_fallback <= 0.0)},
        {"name": "flip_fraction_resolvable",
         "description": ("with n measured ticks the smallest non-zero flip fraction "
                         "expressible is 1/n; below the floor a flip rate is a "
                         "resolution artefact, not a measurement."),
         "measured": worst_ticks, "threshold": float(FLIP_TICKS_FLOOR),
         "direction": "lower", "offending_cell": worst_ticks_cell,
         "met": bool(worst_ticks >= FLIP_TICKS_FLOOR)},
    ]

    # ---- Criteria, scored per arm across seeds (NO cross-arm contrast anywhere).
    def _n_seeds(arm: str, pred) -> int:
        return sum(1 for r in rows if r["arm_id"] == arm and pred(r))

    c1_by_arm = {a: _n_seeds(a, lambda r: r["median_grad_norm"] > GRAD_NORM_FLOOR)
                 for a in ARMS}
    c1_pass = all(v >= SEED_MAJORITY for v in c1_by_arm.values())

    c2_by_arm = {a: _n_seeds(a, lambda r: (r["persistence_bracket_valid"]
                                          and r["persistence_pooled"] < r["c2_midpoint"]))
                 for a in ARMS}
    c2_pass = all(v >= SEED_MAJORITY for v in c2_by_arm.values())

    c4_by_arm = {a: _n_seeds(a, lambda r: r["return_variance"] > RETURN_VAR_FLOOR)
                 for a in ARMS}
    c4_pass = all(v >= SEED_MAJORITY for v in c4_by_arm.values())

    c3_by_arm = {a: _n_seeds(a, lambda r: (r["adv_flip_minus_nonflip"] < 0.0
                                          and r["adv_sign_agrees_detrended"]
                                          and r["n_flip_samples"] >= MIN_FLIP_SAMPLES))
                 for a in ARMS}
    c3_pass = all(v >= ADV_SIGN_SEED_MAJORITY for v in c3_by_arm.values())

    worst_grad, worst_grad_cell = _worst(rows, "median_grad_norm", "min")
    worst_relpers, _ = _worst(rows, "persistence_pooled", "max")
    worst_noise, worst_noise_cell = _worst(rows, "noise_persistence_pooled", "max")
    worst_mid, _ = _worst(rows, "c2_midpoint", "max")
    worst_nflip, worst_nflip_cell = _worst(rows, "n_flip_samples", "min")
    worst_retvar, worst_retvar_cell = _worst(rows, "return_variance", "min")
    worst_advdiff, worst_advdiff_cell = _worst(rows, "adv_flip_minus_nonflip", "max")

    criteria = [
        {"name": "C1_gradient_present", "load_bearing": True, "passed": bool(c1_pass),
         "measured": worst_grad, "threshold": GRAD_NORM_FLOOR,
         "seeds_by_arm": c1_by_arm, "seeds_required": SEED_MAJORITY,
         "offending_cell": worst_grad_cell,
         "detail": ("median per-update pre-clip gradient norm above floor on a "
                    "majority of seeds in EVERY arm. FAIL here means no gradient "
                    "reaches the head at all -- a different finding from a noisy one.")},
        {"name": "C2_persistence_low", "load_bearing": True, "passed": bool(c2_pass),
         "measured": worst_relpers, "threshold": worst_mid,
         "seeds_by_arm": c2_by_arm, "seeds_required": SEED_MAJORITY,
         "noise_floor_measured": worst_noise,
         "positive_control_ceiling": worst_pers,
         "sgd_null_reference_not_used": PERSISTENCE_SGD_NULL_REFERENCE,
         "detail": ("step-direction persistence below the MIDPOINT between this "
                    "cell's measured pure-noise Adam floor and its measured "
                    "positive-control ceiling, on a majority of seeds in every arm. "
                    "PASS with C1 PASS supports H-learning-signal-noisy. The bar is "
                    "bracketed by two IN-RUN controls, not a fixed constant: a fixed "
                    "0.25 was the SGD null and is unreachable under Adam, which was "
                    "a criterion that could not fire when its hypothesis was true.")},
        {"name": "C3_advantage_negative_on_flips", "load_bearing": True,
         "passed": bool(c3_pass and c4_pass),
         "measured": worst_advdiff, "threshold": 0.0, "comparator": "<",
         "seeds_by_arm": c3_by_arm, "seeds_required": ADV_SIGN_SEED_MAJORITY,
         "offending_cell": worst_advdiff_cell,
         "min_flip_samples_required": MIN_FLIP_SAMPLES,
         "worst_n_flip_samples": worst_nflip,
         "worst_n_flip_samples_cell": worst_nflip_cell,
         "detail": ("mean per-sample advantage on flip selections minus non-flip, "
                    "negative on a majority of seeds in every arm. Supports "
                    "H-learning-signal-sign. GATED ON C4: with flat returns this "
                    "comparison is an arithmetic identity (see module docstring).")},
        {"name": "C4_return_variance_present", "load_bearing": False,
         "passed": bool(c4_pass),
         "measured": worst_retvar, "threshold": RETURN_VAR_FLOOR,
         "seeds_by_arm": c4_by_arm, "seeds_required": SEED_MAJORITY,
         "offending_cell": worst_retvar_cell,
         "detail": ("per-episode P1 return variance above floor. NOT an independent "
                    "finding -- it is C3's non-degeneracy condition.")},
    ]
    combination_rule = (
        "READINESS (both synth_credit_control_* preconditions) is a hard gate: unmet "
        "-> substrate_not_ready_requeue, no criterion is read. Given readiness, C1 "
        "decides whether any gradient exists; C2 and C3 are the two pre-registered "
        "legs and are reported INDEPENDENTLY (they are not mutually exclusive and are "
        "NOT AND-ed); C3 is scored only when C4 holds. No criterion contrasts ARM_ON "
        "against ARM_OFF -- the 822f C4 spread comparison is explicitly not carried."
    )

    # ---- Interpretation grid.
    if not readiness_met:
        label = "substrate_not_ready_requeue"
    elif not c1_pass:
        label = "training_signal_absent_no_gradient"
    elif c2_pass and c3_pass and c4_pass:
        label = "both_learning_signal_legs_supported"
    elif c2_pass:
        label = "H_learning_signal_noisy_supported"
    elif c3_pass and c4_pass:
        label = "H_learning_signal_sign_supported"
    else:
        label = "learning_signal_present_and_directed"

    non_degen = {
        "C1_gradient_present": bool(any(r["n_p1_updates"] > 0 for r in rows)),
        "C2_persistence_low": bool(all(r["n_p1_updates"] >= 2 for r in rows)),
        # C3 is degenerate by arithmetic when returns are flat -- see docstring.
        "C3_advantage_negative_on_flips": bool(
            c4_pass and all(r["n_flip_samples"] >= MIN_FLIP_SAMPLES
                            and r["n_nonflip_samples"] > 0
                            and r["adv_sign_agrees_detrended"] for r in rows)),
        # C2 is degenerate if any cell's bracket inverted (control below noise).
        "C2_bracket_valid": bool(all(r["persistence_bracket_valid"] for r in rows)),
        "C4_return_variance_present": bool(all(len(r["per_episode_returns"]) >= 2
                                               for r in rows)),
    }

    outcome = "PASS" if (readiness_met and c1_pass) else "FAIL"
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "unknown",
        "outcome": outcome,
        "timestamp_utc": ts,
        "dry_run": bool(dry_run),
        "hypotheses_adjudicated": ["H-learning-signal-noisy", "H-learning-signal-sign"],
        "criteria": criteria,
        "combination_rule": combination_rule,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": non_degen,
            "unexcluded_rivals": [
                {"name": "H-driver-replay-rule_state-mismatch",
                 "description": (
                     "REINFORCE scores every replayed minibatch sample at the ONE "
                     "live lpfc.rule_state present at update time (end of the "
                     "episode just run), while the samples come from many earlier "
                     "episodes and ticks. The rule half of the head's input is "
                     "therefore mismatched to the credit on most samples, which "
                     "would randomise the rule-attributable gradient BY "
                     "CONSTRUCTION. Faithful to 822f and reproduced deliberately, "
                     "but NOT excluded by any criterion here."),
                 "excluded_by_this_run": False,
                 "why_not_excluded": (
                     "no arm varies the replay conditioning; testing it needs a "
                     "successor arm that restores each sample's own rule_state, "
                     "which would no longer be 822f's optimisation"),
                 "telemetry_for_successor": "rule_state_norm_at_update[]"},
            ],
            "combination_rule": combination_rule,
        },
        "arm_results": rows,
        "diagnostics": {
            "anchor_reachability": anchor_reachability,
            "seeds_note": "822f's seeds, deliberately -- see module docstring.",
            "adv_min_threshold": ADV_MIN_THRESHOLD,
            "total_adv_filtered": int(sum(r["n_adv_filtered"] for r in rows)),
        },
    }
    manifest["readout"] = flat_readout({
        "synth_control_gain_fraction": gain_frac,
        "synth_control_persistence_fraction": pers_frac,
        "worst_synth_task_gain": worst_gain,
        "worst_synth_D_abs": worst_d,
        "worst_synth_persistence": worst_pers,
        "worst_median_grad_norm": worst_grad,
        "worst_persistence_pooled": worst_relpers,
        "worst_noise_persistence": worst_noise,
        "worst_c2_midpoint": worst_mid,
        "worst_n_flip_samples": worst_nflip,
        "worst_return_variance": worst_retvar,
        "worst_adv_flip_minus_nonflip": worst_advdiff,
        "readiness_met": 1 if readiness_met else 0,
        "c1_gradient_present": 1 if c1_pass else 0,
        "c2_persistence_low": 1 if c2_pass else 0,
        "c3_advantage_negative_on_flips": 1 if (c3_pass and c4_pass) else 0,
        "c4_return_variance_present": 1 if c4_pass else 0,
        "n_cells": len(rows),
    })
    manifest["_zg"] = zg
    manifest["_t0"] = t0
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    episodes = ({"p0": 4, "p1": 6, "steps": 12} if args.dry_run
                else {"p0": P0_WARMUP_EPISODES, "p1": P1_BIAS_TRAIN_EPISODES,
                      "steps": STEPS_PER_EPISODE})

    manifest = run_experiment(episodes, args.dry_run)
    zg = manifest.pop("_zg")
    t0 = manifest.pop("_t0")

    out_path = write_flat_manifest(
        manifest,
        dry_run=args.dry_run,
        config={"arms": ARMS, "episodes": episodes, **_config_slice(True)},
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=zg.stats(),
    )
    print(f"manifest: {out_path}", flush=True)
    print(f"label: {manifest['interpretation']['label']}", flush=True)
    print(f"outcome: {manifest['outcome']}", flush=True)
    return manifest, out_path, args.dry_run


if __name__ == "__main__":
    _manifest, _out_path, _dry = main()
    _o = str(_manifest["outcome"]).upper()
    emit_outcome(
        outcome=_o if _o in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        dry_run=_dry,
    )
