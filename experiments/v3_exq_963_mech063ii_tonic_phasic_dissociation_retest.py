"""V3-EXQ-963: MECH-063 sub-claim (ii) tonic-vs-phasic behavioural DISSOCIATION -- RETEST.

Claim:    MECH-063 (control plane retains orthogonal tonic/phasic axes rather than
          collapsing into one scalar) -- SUB-CLAIM (ii): each control axis carries
          BOTH a slow TONIC baseline AND a fast PHASIC event-burst as independent,
          independently-toggleable degrees of freedom on a comparable readout.
Purpose:  evidence (tests MECH-063 sub-claim ii directly).

NEW-NUMBER RETEST, NOT A 779-LETTER (read this before touching the 779 lineage again).

V3-EXQ-779b self-routed `sample_starvation_requeue`: seed 23's episodes average ~6.9
steps, shorter than the phasic surprise-EMA's ~10-tick convergence time constant, so
its baseline never settles within one episode and the R1 precondition
(phasic_fires_real_events) measured 6 event ticks against threshold 10 -- regardless
of step budget (779b already raised seed 23's exposure 835 -> 2400 env steps and the
count did not move). `failure_autopsy_V3-EXQ-779b_2026-07-19.json` fired the
re-derive brake (count 3 on MECH-063: 777, 779, 779a/779b's shared cluster + 779b
itself) and explicitly REFUSED a V3-EXQ-779c: "no further lettered iteration of the
tonic/phasic dissociation probe against the current phasic regulator -- build
sd_phasic_ema_episode_continuity first." SD-075's own design doc restates the refusal
verbatim in its "Retest" section. This experiment is therefore a NEW EXQ NUMBER, not
a lettered continuation, per that refusal.

THE BRAKE IS RELEASED. SD-075 (sd_phasic_ema_episode_continuity, IMPLEMENTED
2026-07-19, ree-v3 4a5139838b) is now built and shows IMPLEMENTED in ree-v3/CLAUDE.md.
Per this repo's re-derive-brake protocol (CLAUDE.md /queue-experiment Step 2.5b): once
the substrate a brake names is built, the brake releases and a new-number retest is
meaningful again.

WHAT SD-075 ADDS (REE_assembly docs/architecture/sd_075_phasic_ema_episode_continuity.md
-- read it for full semantics). Two REEConfig fields on PhasicSurpriseBurstConfig:
  phasic_burst_baseline_continuity: "reset" (default, SD-069 shipping behaviour,
      clears the surprise-EMA at every episode boundary) | "carry" (preserves the
      EMA baseline across episode boundaries; only the envelope/delta/per-episode
      diagnostics still clear).
  phasic_burst_warmup_ticks: 0 OFF | -1 DERIVE = ceil(3 / surprise_ema_decay) = 30
      LIFETIME ticks | positive verbatim -- a convergence gate. ACCOUNTING ONLY: it
      does not suppress the burst (still fires, still perturbs the softmax
      temperature); it only splits get_state()'s event/tick counters into a
      pre-warmup and a converged partition (n_events_converged, n_converged_ticks,
      n_events_prewarmup, lifetime_ticks). Both fields default no-op; every existing
      SD-069 consumer is bit-identical.

THE RETEST-DESIGN QUESTION SD-075 LEFT OPEN, AND HOW IT WAS ANSWERED. SD-075's own
live smoke used an UNTRAINED agent and found n_events_converged = 0 in BOTH
continuity modes at 25 x 7-step episodes on seed 23 -- every event an untrained agent
produces lands inside the first 30 lifetime ticks, before the gate ever opens. The
build's author left an untested hypothesis: does agent TRAINING (SD-074 probe_warmup)
let the converged count clear MIN_EVENT_TICKS=10? V3-EXQ-952 (2026-08-28, diagnostic,
PASS, CLAIM_IDS=[] deliberately -- claim-free substrate-readiness evidence, NOT
MECH-063/SD-069 evidence itself) answered yes: warmup_episodes=40 (SD-074
probe_warmup) + carry continuity + warmup_ticks=-1 clears MIN_EVENT_TICKS=10 on
n_events_converged on ALL THREE of 779b's starvation-category seeds (11, 23, 29):
min 12 events converged across the warmed cells vs untrained control range [0,3].
Confirmed by failure_autopsy_V3-EXQ-952_2026-08-29.json with red-team verification
(control-exposure attack fails; rate-normalized direction holds; threshold inherited
from 779b, not fitted). Governance applied the consequence: REE_assembly cdd772b0dd
marked SD-075's 779b failure_record `resolved` and flipped `ready: true`.

THIS EXPERIMENT is the retest 952 evidence-supports: the SAME 2x2 factorial design as
779b/779a/779 (unchanged science, unchanged thresholds), with every cell's agent now
warmed via SD-074 probe_warmup (num_episodes=40, HELD FIXED across the whole grid --
warmup exposure is not itself a swept factor, only TONIC x PHASIC remain swept,
matching the original claim design) and PHASIC-ON cells configured with SD-075 carry
continuity + warmup_ticks=-1 (DERIVE=30).

ISOLATION CAVEAT (do not read this run as isolating carry vs warmup). The confirmed
V3-EXQ-952 autopsy is explicit that carry mode and warmup=40 training are JOINT
conditions in its own design -- a recorded red-team finding is that the regulator's
30-tick convergence gate is LIFETIME-denominated and does not re-arm per episode
under "reset" mode, so carry's MARGINAL contribution over reset was never isolated by
952. This retest inherits both conditions jointly, exactly as 952's own recommended
design specifies, and is NOT designed to attribute the rescue to one or the other.
That attribution is not needed for THIS retest's own reading: the double-dissociation
verdict (C1 AND C2) only needs the phasic axis to be adequately EXPOSED so R1 can
clear -- it does not need to know which SD-075 leg did the exposing. If that
attribution is ever wanted, the cheap confirmer 952's autopsy names is a 3-cell
reset+warmup arm; this experiment does not add it, because nothing in MECH-063 sub-
claim (ii) depends on it.

DESIGN CHOICE: WHERE THE CONVERGENCE GATE DOES AND DOES NOT APPLY (a genuine
retest-design decision the SD-075 doc leaves open; stated explicitly per its own
"choosing is a retest-design decision" framing, not hidden as an implementation
detail):
  - The R1 READINESS PRECONDITION ("phasic_fires_real_events") reads
    `n_events_converged` from the regulator's own get_state() at the end of each
    PHASIC-ON cell, against MIN_EVENT_TICKS=10 -- UNCHANGED threshold, per SD-075's
    explicit instruction that a consumer must read the converged count rather than a
    raw one. This IS the "declare the cell uninformative rather than reporting a
    near-zero count" instruction, operationalised: a starved cell (779b's seed-23
    failure shape) now fails a STRICTLY STRONGER bar (>=10 events that fired AFTER
    the baseline had genuinely settled, not just >=10 events of any provenance), so
    the self-route to sample_starvation_requeue still fires exactly when the design
    doc says it should.
  - The SCIENTIFIC READOUT (S_sustained_entropy, R_transient, and therefore C1/C2)
    is computed over the FULL tick stream (converged + prewarmup), UNCHANGED from
    779b/779a. Reasoning: SD-075's own framing is explicit that the convergence gate
    is "accounting only... does not suppress the burst" -- a prewarmup event tick is
    behaviourally identical to a converged one (the burst genuinely fires, genuinely
    perturbs the softmax temperature). What is uncertain pre-convergence is only
    whether that firing represents a genuinely surprise-triggered event versus an
    artifact of an unsettled baseline -- a question about CAUSAL ATTRIBUTION of the
    firing, not about whether the entropy effect it produced was real. R_transient
    measures "does phasic-ON produce an entropy transient on event ticks, at all" --
    a question the full tick stream answers correctly and is what the ORIGINAL
    779a/779b design (already shown to clear C1/C2 with real margin) validated.
    Restricting R_transient to converged-only ticks would also introduce an
    arm-asymmetric sampling window (PHASIC-OFF arms have no convergence concept at
    all -- MECH-313 noise_floor's tonic lift is state-independent and live from tick
    1, with no analogous ramp-up), which would bias the PHASIC-ON vs PHASIC-OFF
    comparison in an undocumented way for no SD-075-mandated reason. So: readiness
    reads the converged-only count; the claim-bearing entropy readout does not.

REACHABILITY, CHECKED BEFORE QUEUEING (not merely arithmetically, empirically):
  R1 (MIN_EVENT_TICKS=10 on n_events_converged): V3-EXQ-952 measured min 12 across
      ALL THREE of 779b's hardest starvation-category seeds (11, 23, 29) at this
      exact warmup=40 + carry + warmup_ticks=-1 configuration -- comfortably above
      the bar. Seeds 17 and 37 are new to this specific warmup-rescue measurement
      (952 tested only 11/23/29), but neither was ever R1-constrained in 779b (their
      episodes ran the full 300-step length, unlike seed 23's ~6.9-step episodes),
      so their reachability is expected to be at least as good as seed 23's -- if
      either nonetheless fails to clear, the run self-routes sample_starvation_
      requeue with the offending cell named, exactly as 779b did, rather than
      silently banking a starved cell.
  C1/C2 (SUSTAINED_MARGIN=0.05, TRANSIENT_MARGIN=0.02, DOMINANCE_K=2.0,
      MIN_SEEDS=4): thresholds and computation UNCHANGED from 779a, whose real run
      already cleared both non-degenerately and robustly (mean_dS_tonic +0.265 vs
      margin 0.05; mean_dR_phasic -0.048 vs margin 0.02; 4/5 seeds dissociating;
      robust). This retest changes only R1's readout and cell exposure (warmup +
      carry), not the C1/C2 computation or thresholds, so their joint reachability
      is empirically established by 779a's own manifest, not merely checked on
      paper.

SUBSTRATE-PATH OVERLAP GATE (skill Step 2.5c). This driver imports ree_core/agent.py,
ree_core/utils/config.py, ree_core/environment/causal_grid_world.py, and (via
experiments._lib.probe_warmup.warm_agent's E1/E2 forward-model training) exercises
ree_core/predictors/e1_deep.py. Open `corrupting`/`degrading` substrate_queue
entries checked:
  `mode-governance-engagement` (corrupting; salience_coordinator.py, config.py,
    agent.py) -- NOT exercised: no mode-governance / external-task-switching knob
    is enabled anywhere in `_build_config`; this is a single foraging env with no
    mode axis. Same reasoning as V3-EXQ-952's own overlap note.
  `SD-082` (corrupting; lateral_pfc_analog.py::compute_bias, agent.py) -- NOT
    exercised: concerns SD-078 rule-pool-to-action-bias propagation via the
    lateral-PFC bias head; this driver enables no rule-pool / lateral-PFC bias
    config.
  `contextmemory-write-path-addressing-degeneracy` (corrupting; e1_deep.py::
    ContextMemory.write, agent.py::compute_prediction_loss) -- IS potentially in
    path via E1's forward-model training during warm_agent (the phasic axis's
    signal_source is instantaneous_pe, sourced from e3.last_instantaneous_pe, itself
    downstream of E1's prediction quality). NOT blocking, for the same reason
    V3-EXQ-952's own docstring gave: it applies IDENTICALLY across every cell in
    this grid (all 4 arms x 5 seeds share the same warm_agent recipe) and does not
    interact with the swept factors (TONIC, PHASIC) or with the specific comparison
    C1/C2 test (a difference-of-arms readout). At worst it biases absolute PE/
    entropy levels toward the null uniformly, never toward a false-positive
    dissociation. This retest also directly inherits V3-EXQ-952's own confirmed,
    red-teamed result under the identical warm_agent + carry + instantaneous_pe
    configuration, so whatever residual influence this defect has is already
    priced into that confirmed finding.
  `mech357-freeze-incompatible-pressure-mechanism` / `SD-MECH303-THRESHOLD-SOURCING`
    (degrading; causal_grid_world.py, config.py, agent.py) -- concern hazard-pursuit
    pressure and contextual-safety-threshold subsystems this driver's config does
    not touch beyond defaults (num_hazards=2, no safety-threshold override); noted
    per the skill's degrading-severity handling, not blocking.
  `mech203-valence-pool-admissibility` (severity unset; agent.py::
    update_harm_salience/update_benefit_salience) -- unset severity, no action per
    the skill's own gate.

DESIGN (unchanged from 779/779a/779b except warmup+carry, see above): 2x2 factorial
telemetry probe, NO gradient training during the READ phase, NO synthetic signals.
  Factor TONIC  (T) = use_noise_floor  in {OFF, ON}  (MECH-313 SUSTAINED every-tick
                      temperature lift; NF_ALPHA / NF_MIN_T)
  Factor PHASIC (P) = use_phasic_burst in {OFF, ON}  (SD-069 EVENT-LOCKED transient
                      temperature delta; PHASIC-ON sets signal_source="instantaneous_pe",
                      baseline_continuity="carry", warmup_ticks=-1)
  4 arms x SEEDS seeds. Shared across ALL arms: use_control_vector_logging=True
  (read-only telemetry, bit-identical), hippocampal.use_action_class_scaffold_candidates
  =True, a VOLATILITY-ENABLED env (CausalGridWorldV2 background_drift_enabled=True,
  n_drift_sources=3, drift_policy=random_walk), and SD-074 probe_warmup
  (num_episodes=WARMUP_EPISODES=40) applied BEFORE the read rollout.

READOUT (per fresh E3 selection; candidates captured via a read-only wrapper on
agent.generate_trajectories aligned with agent.e3.last_precommit_probs):
  E = normalised Shannon entropy of last_precommit_probs (nats / ln K); softmax spread.
Each tick is classified by the phasic burst envelope (agent._last_control_vector
["phasic_burst"]["burst_level"]):
  EVENT-WINDOW tick : burst_level  > EVENT_LEVEL_FLOOR (the phasic delta is behaviourally
                      active this tick -- default temp_delta NEGATIVE = sharpening).
  QUIESCENT   tick : burst_level <= EVENT_LEVEL_FLOOR (no active phasic delta).
PHASIC-OFF arms have NO event-window ticks (phasic_burst is None -> burst_level == 0 every
tick), so their transient is 0 by construction.

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

DOUBLE DISSOCIATION (pre-registered; supports MECH-063 sub-claim ii; UNCHANGED from
779/779a/779b):
  C1 (TONIC owns the SUSTAINED baseline): |dS_tonic| >= SUSTAINED_MARGIN
       AND |dS_tonic| >= DOMINANCE_K * |dR_tonic|   (tonic moves sustained, not transient).
  C2 (PHASIC owns the EVENT-LOCKED TRANSIENT): |dR_phasic| >= TRANSIENT_MARGIN
       AND |dR_phasic| >= DOMINANCE_K * |dS_phasic|  (phasic moves transient, not baseline).
  Load-bearing verdict = C1 AND C2 on >= MIN_SEEDS seeds, robust (mean margin exceeds its
  own cross-seed SD).

P0 READINESS (self-routes a REQUEUE label -- NEVER a claim verdict):
  SAMPLE-kind preconditions (unmet -> `sample_starvation_requeue`):
    R1 phasic fires (converged): PHASIC-ON cells have >= MIN_EVENT_TICKS
                       `n_events_converged` (regulator get_state(), SD-075) -- SEE
                       "DESIGN CHOICE" ABOVE for why this reads the converged count
                       while R_transient does not.
    R2 both partitions: PHASIC-ON cells also have >= MIN_QUIESCENT_TICKS quiescent
                       ticks (full stream, unchanged from 779b -- so R is computable).
    R4 samples       : every cell has >= MIN_SELECTS fresh E3 selections (full stream).
  CAPABILITY-kind preconditions (unmet -> `substrate_not_ready_requeue`):
    R3 tonic live    : TONIC-ON cells mean noise_floor_temp_lift >= TEMP_LIFT_FLOOR.
    R5 headroom      : T0P0 baseline entropy in (E_SAT_LOW, E_SAT_HIGH) so both a tonic
                       lift (up) and a phasic sharpening (down) have room to move.
Below any precondition -> outcome FAIL, evidence_direction non_contributory,
non_degenerate False.

VERDICT (pre-registered constants; not derived post-hoc):
  readiness unmet                    -> FAIL / non_contributory (requeue).
  readiness met AND C1 AND C2 robust -> PASS / supports (tonic and phasic are independent
     tonic-baseline vs phasic-transient degrees of freedom on one readout -- MECH-063 ii).
  readiness met AND (C1 AND C2) unmet -> FAIL / weakens (the two levers do not dissociate
     into a sustained-baseline vs event-transient split on a comparable readout).

MECH-094: warm_agent's warmup phase trains a downstream forward-model/action-value
landscape (SD-074's own P0-style de-saturation protocol); the READ phase is waking
select only (StepHarness train_mode=True as in 779b, matching its original design --
the entropy readout is computed from committed E3 selections during ordinary
training-mode stepping, not eval). No sleep, no replay -- N/A for phased-training
applicability beyond warm_agent's own scope.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

_THIS = Path(__file__).resolve()
_REE_V3 = _THIS.parent.parent
if str(_REE_V3) not in sys.path:
    sys.path.insert(0, str(_REE_V3))

from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.regulators.phasic_surprise_burst import (  # noqa: E402
    PhasicSurpriseBurst,
    PhasicSurpriseBurstConfig,
)
from experiments._harness import StepHarness  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.probe_warmup import WarmupRecipe, warm_agent  # noqa: E402
from experiments._lib.sample_driven_rollout import (  # noqa: E402
    SELF_ROUTE_SAMPLE_STARVATION,
    RolloutBudget,
    RolloutOutcome,
    TickContext,
    run_cell_until_samples,
    starvation_selfroute,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.entropy_headroom import per_arm_headroom  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_963_mech063ii_tonic_phasic_dissociation_retest"
EXPERIMENT_PURPOSE = "evidence"
# SD-069 (phasic_surprise_burst) is the substrate this probe directly exercises --
# see the module docstring's PHASIC factor / readiness checks (R1-R5), which test
# SD-069's own firing behaviour (now SD-075-corrected), not just MECH-063's downstream
# dissociation.
CLAIM_IDS: List[str] = ["MECH-063", "SD-069"]

_ZG = ZGoalStreamAccumulator()

# ---- Pre-registered constants (fixed before the run; not derived post-hoc) ----
SEEDS = [11, 17, 23, 29, 37]   # UNCHANGED from 779/779a/779b
MIN_SEEDS = 4                 # criteria must hold on >= this many of the 5 seeds

# ---- SD-074 warmup exposure (952's confirmed rescue configuration; HELD FIXED
# across the whole grid -- not a swept factor. Only TONIC and PHASIC are swept. ----
WARMUP_EPISODES = 40
WARMUP_STEPS_PER_EPISODE = 300   # matches the read phase's own STEPS_PER_EPISODE

# ---- SD-075 fields on PHASIC-ON arms only (952's confirmed rescue configuration) ----
PHASIC_BASELINE_CONTINUITY = "carry"
PHASIC_WARMUP_TICKS = -1          # DERIVE = ceil(3 / PHASIC_EMA_DECAY) = 30

# ---- Sample-driven stopping (unchanged from 779b) ----
TARGET_SELECTS = 800          # fresh E3 selections, EVERY cell (equalises the S estimate
                              # across PHASIC-ON and PHASIC-OFF arms)
TARGET_EVENT_TICKS = 30       # event-window ticks, PHASIC-ON cells only (full stream)
TARGET_QUIESCENT_TICKS = 200  # quiescent ticks, PHASIC-ON cells only
MAX_ENV_STEPS_PER_CELL = 2400
# Step-denominated (779b fix, unchanged): every episode consumes >= 1 env step, so an
# episode cap equal to the step cap can never bind first.
MAX_EPISODES_PER_CELL = MAX_ENV_STEPS_PER_CELL
STEPS_PER_EPISODE = 300

PROGRESS_DENOM_ENV_STEPS = MAX_ENV_STEPS_PER_CELL  # == queue entry episodes_per_run
PROGRESS_EVERY_ENV_STEPS = 100                     # ~25 progress lines per cell

ENV_SIZE = 8
ENV_HAZARDS = 2
ENV_RESOURCES = 3
ENV_DRIFT_SOURCES = 3
ENV_DRIFT_POLICY = "random_walk"

# TONIC axis (MECH-313 noise_floor). Base E3 temperature is 1.0 (StepHarness passes
# temperature=1.0); effective = max(1.0 + alpha, min_T) = 2.0 -> a sustained +1.0 lift.
NF_ALPHA = 1.0
NF_MIN_T = 2.0

# PHASIC axis (SD-069 phasic_surprise_burst). Unchanged trigger/decay constants from
# 779/779a/779b -- only baseline_continuity and warmup_ticks are new (SD-075).
PHASIC_SOURCE = "instantaneous_pe"
PHASIC_TRIGGER_RATIO = 1.2
PHASIC_EMA_DECAY = 0.1
PHASIC_TEMP_DELTA = -0.5        # NEGATIVE = phasic sharpening (LC-NE phasic gain increase)
PHASIC_DECAY = 0.5
PHASIC_TRIGGER_FLOOR = 1e-6
PHASIC_MIN_T = 0.1
EVENT_LEVEL_FLOOR = 0.05

# Readiness thresholds (UNCHANGED from 779/779a/779b -- same pre-registered bar).
MIN_SELECTS = 20              # R4: fresh E3 selections per cell
MIN_EVENT_TICKS = 10          # R1: n_events_converged in PHASIC-ON cells (952-confirmed)
MIN_QUIESCENT_TICKS = 10      # R2: quiescent ticks in PHASIC-ON cells (full stream)
TEMP_LIFT_FLOOR = 0.5         # R3: noise_floor_temp_lift in TONIC-ON cells
E_SAT_LOW = 0.02              # R5: baseline entropy floor (headroom to sharpen down)
E_SAT_HIGH = 0.98             # R5: baseline entropy ceiling (headroom to lift up)

# Verdict thresholds (normalised entropy is in [0, 1]). UNCHANGED from 779a, whose
# real run already cleared both (see module docstring "REACHABILITY").
SUSTAINED_MARGIN = 0.05      # C1: min |tonic effect on sustained baseline entropy|
TRANSIENT_MARGIN = 0.02      # C2: min |phasic effect on event-locked transient|
DOMINANCE_K = 2.0            # each axis moves its OWN readout >= K x the cross-readout

assert TARGET_SELECTS >= MIN_SELECTS
assert TARGET_EVENT_TICKS >= MIN_EVENT_TICKS
assert TARGET_QUIESCENT_TICKS >= MIN_QUIESCENT_TICKS
assert MAX_EPISODES_PER_CELL >= MAX_ENV_STEPS_PER_CELL, (
    "MAX_EPISODES_PER_CELL must be step-denominated (>= MAX_ENV_STEPS_PER_CELL) -- see "
    "V3-EXQ-779b fix 1, unchanged here"
)

ARMS = ["T0P0", "T1P0", "T0P1", "T1P1"]  # (noise_floor, phasic_burst)
_ARM_FLAGS = {
    "T0P0": (False, False),
    "T1P0": (True, False),
    "T0P1": (False, True),
    "T1P1": (True, True),
}

LABEL_SAMPLE_STARVED = SELF_ROUTE_SAMPLE_STARVATION
LABEL_SUBSTRATE_NOT_READY = "substrate_not_ready_requeue"

# No driver-owned env seed knob (unlike 779b's opt-in _ENV_SEED_BASE): this retest
# does not need bitwise reproducibility across processes, and 779b's own default
# (None = OS entropy) is preserved by simply not building that knob at all -- every
# env construction here takes seed=None, identical in effect to 779b run with its
# default.


def _mk_env() -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=None,
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
        # SD-075 (this retest's whole reason to exist): carry the surprise-EMA
        # baseline across episode boundaries + gate event counting on convergence.
        cfg.phasic_burst_baseline_continuity = PHASIC_BASELINE_CONTINUITY
        cfg.phasic_burst_warmup_ticks = PHASIC_WARMUP_TICKS
    return cfg


def _config_slice(arm: str) -> Dict[str, Any]:
    """Fingerprint config slice: env + shared operating settings + this arm's
    control flags + the sample-driven stopping rule + the warmup exposure."""
    use_nf, use_pb = _ARM_FLAGS[arm]
    sl: Dict[str, Any] = {
        "env_size": ENV_SIZE,
        "env_hazards": ENV_HAZARDS,
        "env_resources": ENV_RESOURCES,
        "env_drift_sources": ENV_DRIFT_SOURCES,
        "env_drift_policy": ENV_DRIFT_POLICY,
        "stopping_rule": "sample_driven_v1",
        "target_selects": TARGET_SELECTS,
        "target_event_ticks": TARGET_EVENT_TICKS,
        "target_quiescent_ticks": TARGET_QUIESCENT_TICKS,
        "max_env_steps_per_cell": MAX_ENV_STEPS_PER_CELL,
        "max_episodes_per_cell": MAX_EPISODES_PER_CELL,
        "steps_per_episode": STEPS_PER_EPISODE,
        "use_control_vector_logging": True,
        "use_action_class_scaffold_candidates": True,
        "use_noise_floor": use_nf,
        "use_phasic_burst": use_pb,
        "warmup_episodes": WARMUP_EPISODES,
        "warmup_steps_per_episode": WARMUP_STEPS_PER_EPISODE,
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
            phasic_burst_baseline_continuity=PHASIC_BASELINE_CONTINUITY,
            phasic_burst_warmup_ticks=PHASIC_WARMUP_TICKS,
        )
    return sl


def _norm_entropy(probs: torch.Tensor) -> Optional[float]:
    p = probs.detach().reshape(-1).float()
    k = int(p.numel())
    if k < 2:
        return None
    p = p[p > 0]
    h = float(-(p * p.log()).sum().item())
    return h / math.log(k)


def _mean(xs: List[float]) -> float:
    return float(statistics.fmean(xs)) if xs else 0.0


def _median(xs: List[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def _budget(arm: str) -> RolloutBudget:
    floors: Dict[str, int] = {"selections": TARGET_SELECTS}
    if _ARM_FLAGS[arm][1]:  # PHASIC-ON
        floors["event_ticks"] = TARGET_EVENT_TICKS
        floors["quiescent_ticks"] = TARGET_QUIESCENT_TICKS
    return RolloutBudget(
        sample_floors=floors,
        max_env_steps=MAX_ENV_STEPS_PER_CELL,
        steps_per_episode=STEPS_PER_EPISODE,
        max_episodes=MAX_EPISODES_PER_CELL,
    )


def _fresh_regulator(agent: REEAgent) -> None:
    """Reinstall a zero-lifetime regulator after warmup so the READ phase's
    convergence-gate accounting starts at tick zero, not contaminated by ticks the
    warmup phase already burned (V3-EXQ-952's own pattern, reused verbatim)."""
    agent.phasic_burst = PhasicSurpriseBurst(
        config=PhasicSurpriseBurstConfig(
            enabled=True,
            surprise_ema_decay=PHASIC_EMA_DECAY,
            trigger_ratio=PHASIC_TRIGGER_RATIO,
            trigger_floor=PHASIC_TRIGGER_FLOOR,
            temp_delta=PHASIC_TEMP_DELTA,
            decay=PHASIC_DECAY,
            min_temperature=PHASIC_MIN_T,
            baseline_continuity=PHASIC_BASELINE_CONTINUITY,
            warmup_ticks=PHASIC_WARMUP_TICKS,
        )
    )


def _run_cell(arm: str, seed: int) -> Tuple[Dict[str, Any], RolloutOutcome]:
    """One (arm, seed) cell: SD-074-warmed agent, then a sample-driven telemetry
    rollout (no gradient training in the read phase itself beyond what StepHarness's
    train_mode=True ordinarily does -- unchanged from 779b)."""
    use_nf, use_pb = _ARM_FLAGS[arm]
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

        # SD-074 warmup (952's confirmed rescue exposure), HELD FIXED across the
        # whole 2x2 grid -- not a swept factor.
        recipe = WarmupRecipe(
            num_episodes=WARMUP_EPISODES, steps_per_episode=WARMUP_STEPS_PER_EPISODE,
        )
        env_kwargs = {
            "size": ENV_SIZE, "num_hazards": ENV_HAZARDS,
            "num_resources": ENV_RESOURCES, "background_drift_enabled": True,
            "n_drift_sources": ENV_DRIFT_SOURCES, "drift_policy": ENV_DRIFT_POLICY,
        }
        print(
            "  [warmup] %s seed=%d warmup_episodes=%d starting"
            % (arm, seed, WARMUP_EPISODES), flush=True,
        )
        warm_out = warm_agent(
            agent, env, seed=seed, recipe=recipe, env_kwargs=env_kwargs,
            label="v3_exq_963 %s seed=%d" % (arm, seed), measure=False,
        )
        if use_pb:
            # SD-075: reinstall a zero-lifetime regulator so the read phase's
            # convergence accounting is not contaminated by warmup-phase ticks.
            _fresh_regulator(agent)

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

        e_event: List[float] = []       # entropy on event-window ticks (full stream)
        e_quiescent: List[float] = []   # entropy on quiescent ticks (full stream)
        templift_vals: List[float] = []
        burst_levels: List[float] = []
        surprise_vals: List[float] = []
        surprise_ema_vals: List[float] = []
        surprise_over_ema: List[float] = []
        episode_lengths: List[int] = []

        def _observe(ctx: TickContext) -> Optional[Dict[str, int]]:
            if ctx.step_in_episode == 0:
                episode_lengths.append(0)
            episode_lengths[-1] += 1

            if (
                ctx.n_env_steps == 1
                or ctx.n_env_steps % PROGRESS_EVERY_ENV_STEPS == 0
            ):
                print(
                    f"  [train] {arm} seed={seed} "
                    f"ep {ctx.n_env_steps}/{PROGRESS_DENOM_ENV_STEPS} env-steps "
                    f"(episode {ctx.episode_index + 1}) "
                    f"e3_selects={len(e_event) + len(e_quiescent)}/{TARGET_SELECTS} "
                    f"event_ticks={len(e_event)} quiescent_ticks={len(e_quiescent)}",
                    flush=True,
                )

            probs = ctx.probs
            if not (ctx.fresh and probs is not None and int(probs.numel()) >= 2):
                return None
            ent = _norm_entropy(probs)
            if ent is None:
                return None

            cv = agent._last_control_vector or {}
            pb = cv.get("phasic_burst", {}) or {}
            gv = cv.get("G_vigor", {}) or {}
            blevel = float(pb.get("burst_level", 0.0))
            burst_levels.append(blevel)
            templift_vals.append(float(gv.get("noise_floor_temp_lift", 0.0)))
            reg = getattr(agent, "phasic_burst", None)
            if reg is not None:
                st = reg.get_state()
                s_t = float(st.get("last_surprise", 0.0))
                s_ema = float(st.get("surprise_ema", 0.0))
                surprise_vals.append(s_t)
                surprise_ema_vals.append(s_ema)
                eff = max(s_ema, PHASIC_TRIGGER_FLOOR)
                surprise_over_ema.append(s_t / eff if eff > 0.0 else 0.0)

            if blevel > EVENT_LEVEL_FLOOR:
                e_event.append(ent)
                return {"selections": 1, "event_ticks": 1}
            e_quiescent.append(ent)
            return {"selections": 1, "quiescent_ticks": 1}

        outcome: RolloutOutcome = run_cell_until_samples(
            env=env,
            agent=agent,
            harness=harness,
            budget=_budget(arm),
            observe=_observe,
            progress_label=f"{arm} seed={seed}",
        )
        n_selects = len(e_event) + len(e_quiescent)

        s_quiescent = _mean(e_quiescent)
        s_event = _mean(e_event)
        transient = (s_event - s_quiescent) if e_event else 0.0

        # SD-075 lifetime accounting, read AFTER the rollout, for R1 ONLY (see
        # module docstring "DESIGN CHOICE" -- R_transient/S above stay full-stream).
        n_events_converged = 0
        n_converged_ticks = 0
        n_events_prewarmup = 0
        lifetime_ticks = 0
        warmup_ticks_resolved = 0
        baseline_continuity_resolved = "n/a"
        if use_pb:
            reg = getattr(agent, "phasic_burst", None)
            if reg is not None:
                st = reg.get_state()
                n_events_converged = int(st.get("n_events_converged", 0))
                n_converged_ticks = int(st.get("n_converged_ticks", 0))
                n_events_prewarmup = int(st.get("n_events_prewarmup", 0))
                lifetime_ticks = int(st.get("lifetime_ticks", 0))
                warmup_ticks_resolved = int(st.get("warmup_ticks", 0))
                baseline_continuity_resolved = str(st.get("baseline_continuity", "n/a"))

        row: Dict[str, Any] = {
            "arm": arm,
            "seed": seed,
            "use_noise_floor": use_nf,
            "use_phasic_burst": use_pb,
            "warmup_episodes": WARMUP_EPISODES,
            "warmup_cache_hit": bool(warm_out.cache_hit),
            "episode_lengths": episode_lengths,
            "n_e3_selects": n_selects,
            "n_event_ticks": len(e_event),
            "n_quiescent_ticks": len(e_quiescent),
            "S_sustained_entropy": s_quiescent,
            "E_event_entropy": s_event,
            "R_transient": transient,
            "noise_floor_temp_lift_mean": _mean(templift_vals),
            "burst_level_mean": _mean(burst_levels),
            "burst_level_max": float(max(burst_levels)) if burst_levels else 0.0,
            "surprise_mean": _mean(surprise_vals),
            "surprise_median": _median(surprise_vals),
            "surprise_ema_mean": _mean(surprise_ema_vals),
            "surprise_ema_median": _median(surprise_ema_vals),
            "surprise_over_ema_mean": _mean(surprise_over_ema),
            "surprise_over_ema_median": _median(surprise_over_ema),
            "trigger_ratio": PHASIC_TRIGGER_RATIO,
            "event_rate": (len(e_event) / n_selects) if n_selects else 0.0,
            # SD-075 lifetime accounting (PHASIC-ON cells only; zeros on PHASIC-OFF).
            "n_events_converged": n_events_converged,
            "n_converged_ticks": n_converged_ticks,
            "n_events_prewarmup": n_events_prewarmup,
            "lifetime_ticks": lifetime_ticks,
            "warmup_ticks_resolved": warmup_ticks_resolved,
            "baseline_continuity_resolved": baseline_continuity_resolved,
            "E_event_series": e_event,
            "E_quiescent_series": e_quiescent,
            "surprise_series": surprise_vals,
            "surprise_ema_series": surprise_ema_vals,
        }
        row.update(outcome.as_manifest_fields())
        cell.stamp(row)
        _ZG.observe(agent)
    return row, outcome


def _pooled_std(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    return float(statistics.pstdev(vals))


def _seed_effects(rows_by_arm: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    def _S(arm: str) -> float:
        return float(rows_by_arm[arm]["S_sustained_entropy"])

    def _R(arm: str) -> float:
        return float(rows_by_arm[arm]["R_transient"])

    dS_tonic = ((_S("T1P0") + _S("T1P1")) / 2.0) - ((_S("T0P0") + _S("T0P1")) / 2.0)
    dR_tonic = ((_R("T1P0") + _R("T1P1")) / 2.0) - ((_R("T0P0") + _R("T0P1")) / 2.0)
    dS_phasic = ((_S("T0P1") + _S("T1P1")) / 2.0) - ((_S("T0P0") + _S("T1P0")) / 2.0)
    dR_phasic = ((_R("T0P1") + _R("T1P1")) / 2.0) - ((_R("T0P0") + _R("T1P0")) / 2.0)

    c1 = (abs(dS_tonic) >= SUSTAINED_MARGIN) and (
        abs(dS_tonic) >= DOMINANCE_K * abs(dR_tonic)
    )
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


def _worst_cell(
    rows: List[Dict[str, Any]], key: str
) -> Tuple[float, Optional[Dict[str, Any]]]:
    if not rows:
        return 0.0, None
    worst = min(rows, key=lambda r: r[key])
    return float(worst[key]), {
        "seed": worst["seed"],
        "arm": worst["arm"],
        "value": worst[key],
        "n_env_steps": worst["n_env_steps"],
        "n_episodes": worst["n_episodes"],
        "stop_reason": worst["rollout_stop_reason"],
        "sample_floors_met": worst["rollout_floors_met"],
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    global TARGET_SELECTS, TARGET_EVENT_TICKS, TARGET_QUIESCENT_TICKS
    global MAX_ENV_STEPS_PER_CELL, MAX_EPISODES_PER_CELL
    global PROGRESS_DENOM_ENV_STEPS, PROGRESS_EVERY_ENV_STEPS
    global STEPS_PER_EPISODE, WARMUP_EPISODES
    seeds = SEEDS[:2] if dry_run else SEEDS
    if dry_run:
        TARGET_SELECTS = 40
        TARGET_EVENT_TICKS = 3
        TARGET_QUIESCENT_TICKS = 10
        MAX_ENV_STEPS_PER_CELL = 120
        MAX_EPISODES_PER_CELL = MAX_ENV_STEPS_PER_CELL
        STEPS_PER_EPISODE = 40
        PROGRESS_DENOM_ENV_STEPS = MAX_ENV_STEPS_PER_CELL
        PROGRESS_EVERY_ENV_STEPS = 20
        WARMUP_EPISODES = 2  # smoke-only override; real run uses 40

    rows: List[Dict[str, Any]] = []
    rollout_cells: List[Dict[str, Any]] = []
    for seed in seeds:
        for arm in ARMS:
            print(f"Seed {seed} Condition {arm}", flush=True)
            row, rollout = _run_cell(arm, seed)
            rows.append(row)
            rollout_cells.append({"arm": arm, "seed": seed, "outcome": rollout})
            print(f"verdict: {'PASS' if row['n_e3_selects'] > 0 else 'FAIL'}", flush=True)

    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        by_arm = {r["arm"]: r for r in rows if r["seed"] == seed}
        if len(by_arm) != len(ARMS):
            continue
        eff = _seed_effects(by_arm)
        eff["seed"] = seed
        per_seed.append(eff)

    p1_rows = [r for r in rows if r["use_phasic_burst"]]   # PHASIC-ON arms
    t1_rows = [r for r in rows if r["use_noise_floor"]]    # TONIC-ON arms
    baseline_rows = [r for r in rows if r["arm"] == "T0P0"]

    # R1: SD-075-updated -- min-across-PHASIC-ON-cells of n_events_converged.
    r1_val, r1_cell = _worst_cell(p1_rows, "n_events_converged")
    # R2 unchanged from 779b: full-stream quiescent ticks.
    r2_val, r2_cell = _worst_cell(p1_rows, "n_quiescent_ticks")
    r4_val, r4_cell = _worst_cell(rows, "n_e3_selects")

    # R3/R5 fixed (validate_experiments.py precondition_recomputability finding, also
    # flagged against 779b itself): `met` is a worst-case (all()) claim, so `measured`
    # must report the WORST cell, not the mean -- a single out-of-band cell whose
    # deviation is masked by an in-band mean would otherwise recompute as MET.
    r3_val, r3_cell = _worst_cell(t1_rows, "noise_floor_temp_lift_mean") if t1_rows else (0.0, None)
    r1_phasic_fires = bool(p1_rows) and r1_val >= MIN_EVENT_TICKS
    r2_both_partitions = bool(p1_rows) and r2_val >= MIN_QUIESCENT_TICKS
    r3_tonic_live = bool(t1_rows) and r3_val >= TEMP_LIFT_FLOOR
    r4_samples = bool(rows) and r4_val >= MIN_SELECTS
    # R5 stays scoped to the BASELINE partition only (per the validator's own fix
    # recipe, matching V3-EXQ-779b's autopsy-derived guidance): a saturating TREATMENT
    # arm is not a substrate-readiness failure -- widening this gate to all arms would
    # mislabel a manipulation that exceeded the readout's dynamic range as "substrate
    # not ready". Within that scope, report the WORST baseline cell (max distance from
    # either bound) rather than the mean, so `met` recomputes correctly.
    r5_worst_dist = None
    r5_worst_cell: Optional[Dict[str, Any]] = None
    for r in baseline_rows:
        s_val = float(r["S_sustained_entropy"])
        dist = min(s_val - E_SAT_LOW, E_SAT_HIGH - s_val)
        if r5_worst_dist is None or dist < r5_worst_dist:
            r5_worst_dist = dist
            r5_worst_cell = {"seed": r["seed"], "arm": r["arm"], "value": s_val}
    r5_val = (r5_worst_cell["value"] if r5_worst_cell else 0.0)
    r5_headroom = bool(baseline_rows) and (r5_worst_dist is not None and r5_worst_dist > 0.0)
    # Non-gating per-arm headroom diagnostic (V3-EXQ-779b autopsy section 7, Learning
    # 1): the manipulation is what pushes a readout toward a bound, so this reports
    # EVERY arm's headroom -- including the tonic-ON / phasic-ON treatment arms the
    # baseline-only R5 gate above cannot see -- without gating on it. Emitted on every
    # outcome, PASS included.
    entropy_headroom_diagnostic = per_arm_headroom(
        rows, value_key="S_sustained_entropy", low=E_SAT_LOW, high=E_SAT_HIGH,
    )
    sample_unmet = [
        name for name, ok in (
            ("phasic_fires_real_events_converged", r1_phasic_fires),
            ("both_partitions_populated", r2_both_partitions),
            ("sample_sufficiency", r4_samples),
        ) if not ok
    ]
    capability_unmet = [
        name for name, ok in (
            ("tonic_axis_live", r3_tonic_live),
            ("baseline_entropy_headroom", r5_headroom),
        ) if not ok
    ]
    readiness_met = not sample_unmet and not capability_unmet

    diss = [a["dissociation"] for a in per_seed]
    seeds_diss = sum(1 for d in diss if d)
    diss_seed_count = seeds_diss >= min(MIN_SEEDS, len(seeds))
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

    sampling_shortfall = starvation_selfroute(rollout_cells)

    if not readiness_met:
        outcome = "FAIL"
        direction = "non_contributory"
        if capability_unmet:
            label = LABEL_SUBSTRATE_NOT_READY
            degeneracy_reason = (
                "substrate capability precondition unmet: "
                + ", ".join(capability_unmet)
            )
        else:
            label = LABEL_SAMPLE_STARVED
            offender = r1_cell if not r1_phasic_fires else (
                r2_cell if not r2_both_partitions else r4_cell
            )
            degeneracy_reason = (
                "sample-count precondition unmet (substrate present and active): "
                + ", ".join(sample_unmet)
                + (f"; offending cell seed={offender['seed']} arm={offender['arm']} "
                   f"n_env_steps={offender['n_env_steps']} "
                   f"stop_reason={offender['stop_reason']}" if offender else "")
            )
        non_degenerate = False
    elif dissociation:
        outcome = "PASS"
        direction = "supports"
        label = "tonic_phasic_double_dissociation"
    else:
        outcome = "FAIL"
        direction = "weakens"
        label = "tonic_phasic_no_dissociation"

    evidence_direction_per_claim = {"MECH-063": direction, "SD-069": direction}

    interpretation = {
        "label": label,
        "sampling_shortfall": sampling_shortfall,
        "readiness_unmet_sample_kind": sample_unmet,
        "readiness_unmet_capability_kind": capability_unmet,
        "preconditions": [
            {"name": "phasic_fires_real_events_converged",
             "kind": "sample",
             "control": "PHASIC-ON cells: SD-075 n_events_converged (min across cells) "
                        "-- the regulator's own lifetime post-convergence event count, "
                        "not the raw full-stream event-tick count 779b used",
             "measured": r1_val,
             "threshold": MIN_EVENT_TICKS, "direction": "lower", "met": bool(r1_phasic_fires),
             "offending_cell": r1_cell},
            {"name": "both_partitions_populated",
             "kind": "sample",
             "control": "PHASIC-ON cells: quiescent ticks, full stream (so the transient "
                        "R_transient is computable; min across cells)",
             "measured": r2_val,
             "threshold": MIN_QUIESCENT_TICKS, "direction": "lower", "met": bool(r2_both_partitions),
             "offending_cell": r2_cell},
            {"name": "tonic_axis_live",
             "kind": "capability",
             "control": "TONIC-ON cells: noise_floor_temp_lift (worst cell, not mean -- "
                        "met is a worst-case all() claim)",
             "measured": r3_val,
             "threshold": TEMP_LIFT_FLOOR, "direction": "lower", "met": bool(r3_tonic_live),
             "offending_cell": r3_cell},
            {"name": "sample_sufficiency",
             "kind": "sample",
             "control": "min fresh E3 selections over cells (full stream)",
             "measured": r4_val,
             "threshold": MIN_SELECTS, "direction": "lower", "met": bool(r4_samples),
             "offending_cell": r4_cell},
            {"name": "baseline_entropy_headroom",
             "kind": "capability",
             "control": "T0P0 sustained entropy strictly inside (E_SAT_LOW, E_SAT_HIGH); "
                        "worst baseline cell (max distance from either bound), not mean; "
                        "scoped to baseline ON PURPOSE -- see entropy_headroom_per_arm "
                        "diagnostic for the treatment arms this gate cannot see",
             "measured": r5_val,
             "threshold_low": E_SAT_LOW, "threshold_high": E_SAT_HIGH,
             "comparator_low": ">", "comparator_high": "<",
             "direction": "interval", "met": bool(r5_headroom),
             "offending_cell": r5_worst_cell},
        ],
        "criteria": [
            {"name": "double_dissociation_C1_and_C2", "load_bearing": True, "passed": bool(dissociation)},
        ],
        "criteria_non_degenerate": {
            "double_dissociation_C1_and_C2": bool(
                len(per_seed) >= min(MIN_SEEDS, len(seeds))
                and (_pooled_std(dS_tonic_all) > 0.0 or _pooled_std(dR_phasic_all) > 0.0)
            ),
        },
        "summary": (
            "MECH-063 sub-claim (ii) tonic-vs-phasic dissociation on the E3 softmax "
            "temperature -- RETEST of V3-EXQ-779b/779a/779 against the SD-075-corrected "
            "phasic regulator (carry continuity + convergence gate) with every cell's "
            "agent SD-074-warmed (num_episodes=40, 952's confirmed rescue exposure, held "
            "fixed across the whole grid). The TONIC lever (MECH-313 noise_floor) moves "
            "the SUSTAINED (quiescent-tick) baseline entropy; the PHASIC lever (SD-069 "
            "phasic_surprise_burst, sharp instantaneous_pe source) moves an EVENT-LOCKED "
            "entropy TRANSIENT; cross-legs ~0. PASS = both hold (independent tonic-"
            "baseline vs phasic-transient degrees of freedom on one readout); FAIL/"
            "weakens = they do not dissociate; FAIL/non_contributory = the run could not "
            "test the claim. R1 (phasic_fires_real_events_converged) reads SD-075's "
            "n_events_converged rather than 779b's raw event-tick count, per SD-075's "
            "design doc; R_transient/S_sustained_entropy remain full-stream, matching "
            "779a's already-validated computation (see module docstring 'DESIGN "
            "CHOICE'). NOT an isolation of carry-mode vs warmup-training credit -- "
            "both are inherited jointly from V3-EXQ-952's confirmed design (see module "
            "docstring 'ISOLATION CAVEAT'); this retest's own reading does not depend on "
            "that attribution."
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
        "evidence_direction_per_claim": evidence_direction_per_claim,
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
        "sampling_summary": {
            "stopping_rule": "sample_driven_v1",
            "target_selects": TARGET_SELECTS,
            "target_event_ticks": TARGET_EVENT_TICKS,
            "target_quiescent_ticks": TARGET_QUIESCENT_TICKS,
            "max_env_steps_per_cell": MAX_ENV_STEPS_PER_CELL,
            "max_episodes_per_cell": MAX_EPISODES_PER_CELL,
            "cells_floors_met": sum(1 for r in rows if r["rollout_floors_met"]),
            "n_cells": len(rows),
            "stop_reasons": {
                reason: sum(1 for r in rows if r["rollout_stop_reason"] == reason)
                for reason in sorted({r["rollout_stop_reason"] for r in rows})
            },
            "min_n_e3_selects": (min(r["n_e3_selects"] for r in rows) if rows else 0),
            "max_n_e3_selects": (max(r["n_e3_selects"] for r in rows) if rows else 0),
            "min_n_event_ticks_phasic_on": (
                min(r["n_event_ticks"] for r in p1_rows) if p1_rows else 0
            ),
            "min_n_events_converged_phasic_on": (
                min(r["n_events_converged"] for r in p1_rows) if p1_rows else 0
            ),
            "total_env_steps": sum(r["n_env_steps"] for r in rows),
        },
        "per_seed": per_seed,
        "arm_results": rows,
        "diagnostics": {
            # Non-gating (V3-EXQ-779b autopsy section 7, Learning 1): reported on
            # every outcome, PASS included, precisely because it must NOT gate.
            "entropy_headroom_per_arm": entropy_headroom_diagnostic,
        },
        "notes": (
            "MECH-063 sub-claim (ii) tonic/phasic split behavioural dissociation -- NEW-"
            "NUMBER RETEST of V3-EXQ-779b/779a/779, NOT a lettered continuation (the "
            "confirmed failure_autopsy_V3-EXQ-779b_2026-07-19.json re-derive brake "
            "explicitly refuses a V3-EXQ-779c). Brake released: SD-075 "
            "(sd_phasic_ema_episode_continuity) is now IMPLEMENTED (ree-v3 4a5139838b), "
            "and V3-EXQ-952 (2026-08-28, diagnostic, claim-free) confirmed that SD-074 "
            "warmup (num_episodes=40) + SD-075 carry continuity + warmup_ticks=-1 clears "
            "MIN_EVENT_TICKS=10 on n_events_converged for all three of 779b's starvation-"
            "category seeds (min 12 vs untrained control range [0,3]). This retest "
            "extends that confirmed configuration to the full original 779/779a/779b 2x2 "
            "factorial x 5-seed design, changing ONLY: (1) every cell's agent is now "
            "SD-074-warmed (40 episodes, fixed across the grid, not swept); (2) PHASIC-ON "
            "arms carry SD-075 baseline_continuity='carry' + warmup_ticks=-1; (3) the R1 "
            "readiness precondition reads n_events_converged instead of the raw event-"
            "tick count. C1/C2 computation, thresholds, aggregation, and the S/R readout "
            "itself are UNCHANGED from 779a, whose real run already cleared C1 AND C2 "
            "non-degenerately and robustly (mean_dS_tonic +0.265 vs margin 0.05; "
            "mean_dR_phasic -0.048 vs margin 0.02; 4/5 seeds; robust) and was withheld "
            "only on the now-fixed R1 precondition. ISOLATION CAVEAT: this run does not "
            "isolate carry-mode's marginal contribution over reset (V3-EXQ-952's own "
            "red-team finding); both SD-075 legs and the warmup exposure are inherited "
            "jointly, which the claim-level dissociation verdict does not need "
            "disentangled. Seeds 17 and 37 are new to the warmup-rescue measurement "
            "specifically (952 tested only 11/23/29) but were never R1-constrained in "
            "779b (their episodes ran the full 300-step length); if either fails to "
            "clear regardless, the run self-routes sample_starvation_requeue with the "
            "offending cell named, per the unchanged readiness machinery."
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
        "stopping_rule": "sample_driven_v1",
        "target_selects": TARGET_SELECTS,
        "target_event_ticks": TARGET_EVENT_TICKS,
        "target_quiescent_ticks": TARGET_QUIESCENT_TICKS,
        "max_env_steps_per_cell": MAX_ENV_STEPS_PER_CELL,
        "max_episodes_per_cell": MAX_EPISODES_PER_CELL,
        "steps_per_episode": STEPS_PER_EPISODE,
        "warmup_episodes": WARMUP_EPISODES,
        "warmup_steps_per_episode": WARMUP_STEPS_PER_EPISODE,
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
            "baseline_continuity": PHASIC_BASELINE_CONTINUITY,
            "warmup_ticks": PHASIC_WARMUP_TICKS,
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
        z_goal_stream_stats=_ZG.stats(),
    )

    print(f"FINAL_OUTCOME: {manifest['outcome']}", flush=True)
    print(f"  label: {manifest['interpretation']['label']}", flush=True)
    print(f"  readiness_met: {manifest['acceptance']['readiness_met']}", flush=True)
    _samp = manifest["sampling_summary"]
    print(f"  cells_floors_met: {_samp['cells_floors_met']}/{_samp['n_cells']} "
          f"stop_reasons={_samp['stop_reasons']}", flush=True)
    print(f"  min_selects={_samp['min_n_e3_selects']} "
          f"min_events_converged_P1={_samp['min_n_events_converged_phasic_on']}", flush=True)
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
