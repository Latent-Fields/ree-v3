"""
V3-EXQ-901 -- INV-051 MEL-dose Goldilocks rigidity sweep: does model RIGIDITY
trace a genuine U-shape (elevated at BOTH extremes, lower at mid-dose) across
a >=3-level graded Model Error Load (MEL) sweep, using the REAL SD-MEL-
PRODUCER->SD-MEL-CONSUMER->SD-017 sleep-loop pathway?

SLEEP DRIVER: manual-cycle-loop (agent.sleep_loop.force_cycle() called once
per cycle in a dedicated MEAS_CYCLES wake-sleep loop, per arm). The MEL
consumer engages ONLY through the SleepLoopManager path (force_cycle), never
via agent.run_sleep_cycle() directly (see SD-MEL-CONSUMER driver note,
ree-v3/CLAUDE.md).

CLAIM UNDER TEST (INV-051): "There exists an optimal range of daily Model
Error Load (MEL): insufficient MEL (extreme monotony, isolation,
institutional under-stimulation) produces progressive model rigidity via
under-stimulation of the learning drive even when sleep architecture is
intact; excessive MEL (acute trauma, overwhelming novelty, extreme
responsibility) produces overload insomnia and incomplete update even with
extended sleep opportunity."

================================================================================
WHY THIS RUN CLOSES THE NON-DEGENERACY GAP (claims.yaml INV-051
what_would_answer, digested 2026-08-08)
================================================================================
INV-051's own what_would_answer names a precise gap: SD-MEL-PRODUCER's
world_rule_shift_interval/world_rule_shift_depth knobs already support a
>=3-level graded MEL-dose variable, but no prior run (V3-EXQ-845/861/861a)
has used more than a high-vs-low CONTRAST -- and none of those runs measured
a RIGIDITY DV (they measure consolidation AMOUNT -- sws_n_writes /
rem_n_rollouts / sws_slot_diversity as a dose-response, monotone-in-MEL
target -- MECH-180's question, not INV-051's U-shape question). This run is
the first to (a) sweep >=3 graded MEL-dose levels spanning under-stimulation
through overload, and (b) score a pre-registered RIGIDITY DV (post-training
behavioural policy entropy on a fixed novel probe) for a U-shape, not a
monotone trend.

GOV-REUSE-1 CHECK (Step 2.4, this session): decisive readout = per-seed
rigidity DV (probe-battery policy entropy) scored across a >=3-level MEL-dose
sweep, testing for a U-shape rather than a monotone trend. Searched
REE_assembly/evidence/experiments for any manifest carrying claim_ids
INV-051: zero matches (grep, 2026-08-08 -- only aggregator/index files
mention the string, never a run manifest). No compatible substrate_hash
carries this combination (785/861/861a use MECH-180's amount DVs, not a
behavioural-flexibility readout, and none sweep more than 2 MEL levels
ecologically). Not recoverable by reanalysis -> proceeds to a new run.

SUBSTRATE READINESS (Step 2.5/2.5a, this session): all three named pieces are
built and IMPLEMENTED/VALIDATED (ree-v3/CLAUDE.md "SD-017", "SD-MEL-CONSUMER",
"SD-MEL-PRODUCER" sections; docs/architecture/sd_mel_producer.md Status:
VALIDATED via V3-EXQ-798a, confirmed failure_autopsy_V3-EXQ-798a_2026-07-30).
Step 2.5a one-tick probe (this session, throwaway scratch script) empirically
confirmed: CausalGridWorldV2(world_rule_shift_enabled=True, interval=10,
depth in {2,3}) constructs and steps cleanly, info carries
world_rule_shift_occurred/count/steps_since_world_rule_shift as documented;
depth=3 produces MORE distinct re-permuted action-maps than depth=2 over the
same 20 shifts (19/20 vs 13/20 distinct), mechanically supporting depth as a
graded magnitude knob for the OVERLOAD arm; REEAgent(use_mel_consumer=True,
use_sleep_loop=True, sws_enabled=True, rem_enabled=True) constructs with
agent.mel_consumer / agent.sleep_loop both live; one waking step +
agent.sleep_loop.force_cycle(agent) returns the exact keys this driver reads
(sws_n_writes, rem_n_rollouts, sws_slot_diversity, mel_duration_factor,
mel_mean), and mel_duration_factor was observed reaching the FACTOR_MAX=3.0
ceiling in that quick probe -- confirming the consumer's saturation clamp is
empirically reachable, the mechanism this run's OVERLOAD arm leans on to
operationalise "MEL exceeding consumer clearance capacity". No doc-vs-runtime
drift found; proceeds per Step 2.5a.

RE-DERIVE BRAKE (Step 2.5b, this session): zero prior failure-autopsy targets
tag INV-051 at all (grep across REE_assembly/evidence/planning/failure_autopsy_*.json
-> 0 hits), so the brake does not apply (nothing to release or hold).

SUBSTRATE-PATH OVERLAP GATE (Step 2.5c, this session): substrate_queue.json
has zero OPEN entries carrying a recorded substrate_paths field at all, so no
overlap check is possible or needed -- nothing to overlap with.

ETHICS PREFLIGHT (Step 2.6): see ethics_preflight block in the manifest.
All-false / decision=allow (no live self-model, no autobiographical memory,
harm/residue streams are pre-ethical instrumentation per SENT-0).

================================================================================
DESIGN -- ARM LADDER (>=3 graded MEL-dose levels, non-degeneracy precondition)
================================================================================
Five ON arms span the claim's own two failure modes plus a mid/optimal
region, using SD-MEL-PRODUCER's validated interval ladder (V3-EXQ-798a,
depth=2: 0/60/25/10) for the four lower/mid rungs, extended by ONE further
rung (same validated interval=10, depth=3 -- an empirically-probed magnitude
extension, not a new untested knob) for the upper-bound OVERLOAD arm:

  ARM_0_MONOTONY_ON  interval=0,  depth=0  -- lower-bound: extreme monotony,
                                               near-zero non-stationarity.
  ARM_1_LOW_ON       interval=60, depth=2
  ARM_2_OPTIMAL_ON   interval=25, depth=2  -- expected mid/optimal dose
  ARM_3_HIGH_ON      interval=10, depth=2
  ARM_4_OVERLOAD_ON  interval=10, depth=3  -- upper-bound: same shift RATE as
                                               HIGH but doubled-and-a-half
                                               transposition MAGNITUDE per
                                               shift (798a's ladder varied
                                               only rate; this run is the
                                               first to vary depth), leaning
                                               on the MEL-consumer's
                                               FACTOR_MAX clamp (empirically
                                               reachable per the 2.5a probe)
                                               to operationalise "MEL
                                               exceeding consumer clearance
                                               capacity".
  ARM_5_OVERLOAD_OFF interval=10, depth=3, mel_on=False -- non-load-bearing
                                               control (mirrors V3-EXQ-845's
                                               C2 pattern): does the MEL
                                               CONSUMER's adaptive scaling
                                               matter, or would raw
                                               environmental novelty alone
                                               produce the same rigidity
                                               reading regardless of whether
                                               sleep adapts to it?

The 5 ON arms are sorted PER SEED by MEASURED mean_mel (not nominal arm
label) before the U-shape test is scored -- exactly 798a/845's convention --
so the test is honest about which arm is actually LOW/MID/HIGH in THIS run's
measured dose, not merely by design intent.

FALSIFYING CONDITION (c) -- "opportunity vs content" control, BUILT INTO THE
DESIGN rather than a separate ablation arm: agent.sleep_loop.force_cycle()
fires MEAS_CYCLES times in EVERY arm regardless of measured MEL (the
SleepLoopManager's cadence here is driven by the DRIVER's wake-sleep loop,
not by MEL), and mel_duration_factor is clamped to [FACTOR_MIN=0.5,
FACTOR_MAX=3.0] -- it can scale CONTENT (how many SWS writes / REM rollouts
happen per cycle) but never drops sleep OPPORTUNITY to zero (a cycle always
fires, SWS+REM phases always both run). So sleep ARCHITECTURE and
OPPORTUNITY are identical across every ON arm by construction; the ONLY
thing that varies with MEL-dose is sleep CONTENT (how much gets processed).
R3 below asserts and records this (n_cycles_fired == MEAS_CYCLES + factor
never below FACTOR_MIN, across every ON arm/seed) so an apparent low-MEL
rigidity reading cannot be explained by "less sleep happened" -- it
structurally could not have.

================================================================================
RIGIDITY DV (pre-registered, load-bearing)
================================================================================
PRIMARY: post-measurement behavioural policy entropy on a FIXED, NOVEL,
held-out probe (option "action_bias_div/policy entropy" from INV-051's
what_would_answer). After the arm's measurement cycles complete, the trained
agent (encoder + E2 forward model as shaped by that arm's training; no
further training during the probe) is run for PROBE_EPISODES episodes on a
STATIONARY (world_rule_shift OFF) probe environment seeded DIFFERENTLY from
the arm's own training env (probe_seed = seed + PROBE_SEED_OFFSET) so the
resource/hazard layout is genuinely unseen -- this isolates what the agent's
TRAINING under that arm's MEL-dose taught it about producing varied,
adaptive behaviour, from any further environmental perturbation during the
probe itself. Every selected action (argmax over the 5-way ACTIONS map) is
tallied; Shannon entropy H = -sum(p*log(p)) is computed over the resulting
distribution, and

    rigidity_index = 1 - H / H_max,   H_max = ln(5)

clipped to [0, 1]. rigidity_index = 0 means the agent used all 5 actions
uniformly (maximally flexible); rigidity_index -> 1 means the agent
collapsed onto a small stereotyped repertoire ("habitual attractors",
literally the claim's own lower-bound-failure language).

SECONDARY / non-load-bearing / informational: mean_sws_slot_diversity (the
SAME SD-017 SWS-pass readout V3-EXQ-845/677/718/718a already use, mean
pairwise cosine DISTANCE across ContextMemory schema-attractor slots after
each measurement cycle -- a representation-level differentiation reading,
reported alongside the behavioural DV but not gated on, since it answers a
related-but-distinct question about what got consolidated rather than how
the agent then behaves).

DV-SYMMETRY INVARIANCE declaration (604c check, Step 3.5 MANDATORY): the
manipulation here is which MEL-dose ARM an agent's ~78-episode P0 + wake-
sleep TRAINING window was run under -- i.e. it changes the LEARNED WEIGHTS of
agent.e2 (world-forward) via the recon-only training loop, and changes the
CONTENT of SD-017's SWS/REM consolidation passes via the MEL-consumer's
duration-factor scaling. rigidity_index is computed AFTER that training, from
agent.select_action's live output on a fixed probe battery -- a genuinely
different LEARNED candidate-scoring function per arm, not a tick-level
transform applied to one shared, already-fixed scoring function. It is
therefore NOT a broadcast additive constant across candidates (there is no
shared candidate-scoring function the arms merely offset), NOT a monotone
rescaling of a shared rank order (each arm's E2 weights differ, so the
argmax itself can differ, not just its score), and NOT a permutation of
interchangeable units (the probe's action space and layout are IDENTICAL
across arms for a given seed -- only the trained policy differs). None of
the three documented invariance classes apply; the measured delta reflects
genuine representational/behavioural differences from differential training,
not an arithmetic identity fixed before the run.

================================================================================
PRE-REGISTERED ACCEPTANCE (evidence; claim_ids=["INV-051"])
================================================================================
READINESS (per seed, over the 5 ecological ON arms):
  R1 world-model trained: frozen-probe conv_rel_drop >= MIN_REL_CONV_DROP.
  R2 MEL-dose sweep non-degenerate IN THIS CONFIG: measured mean_mel is
     graded across the 5 ON arms (max >= min * (1+MIN_MEL_SPREAD)) -- without
     this, "dose" is not actually varying and no U-shape claim is testable.
  R3 sleep OPPORTUNITY uniform across dose (the "content not opportunity"
     control, see above): every ON arm/seed fired exactly MEAS_CYCLES sleep
     cycles AND never saw mel_duration_factor fall below FACTOR_MIN.
  Below-floor on R1/R2/R3 on >= SEED_PASS_FRAC of seeds routes to
  substrate_not_ready_requeue, NEVER an INV-051 verdict.

C1 (LOAD-BEARING -- the U-shape test). Per seed, sort the 5 ON arms by
MEASURED mean_mel ascending; LOW = index 0, MID = index 2 (median of 5),
HIGH = index 4.
  delta_low[seed]  = rigidity_index[LOW]  - rigidity_index[MID]
  delta_high[seed] = rigidity_index[HIGH] - rigidity_index[MID]
  margin_low  = max(RIGIDITY_SD_MULT * pstdev(delta_low across seeds),  RIGIDITY_MIN_EFFECT_FLOOR)
  margin_high = max(RIGIDITY_SD_MULT * pstdev(delta_high across seeds), RIGIDITY_MIN_EFFECT_FLOOR)
  (feedback_effect_size_pass_gate_margin convention: scale noise on the SD of
  the DELTA, never the SD of the baseline level, plus an absolute floor so a
  trivially-small-but-consistent delta cannot clear an SD-collapsed margin --
  see V3-EXQ-680b SUPERADD_MIN_EFFECT_FLOOR, same pattern, SD_MULT=2.0 here too.)
  clears_low[seed]  = delta_low[seed]  > margin_low
  clears_high[seed] = delta_high[seed] > margin_high
  COMBINATION RULE (explicit, per Step 3.5 multi-criterion requirement): C1
  is CONJUNCTIVE across the two sides -- both the low-dose AND high-dose
  extreme must independently clear their own margin against mid-dose, each
  on >= SEED_PASS_FRAC of seeds. C1 = (frac_seeds(clears_low) >= SEED_PASS_FRAC)
  AND (frac_seeds(clears_high) >= SEED_PASS_FRAC). This is the U-shape itself:
  a single elevated extreme (only low OR only high) is NOT sufficient --
  INV-051 predicts BOTH failure modes.

FALSIFYING MONOTONICITY CHECK (non-load-bearing, informs routing): per seed,
  is rigidity_index monotone (non-decreasing or non-increasing, +MONO_TOL)
  across the 5 ON arms sorted by measured mean_mel? Monotone on
  >= SEED_PASS_FRAC of seeds is one of INV-051's own pre-registered
  falsifying conditions (strictly increasing/decreasing rigidity, not
  U-shaped).

C2 (control, non-load-bearing -- mirrors V3-EXQ-845's C2 pattern): does the
  MEL CONSUMER's adaptive scaling matter for the overload-arm rigidity
  reading, or does raw environmental novelty alone (consumer OFF, fixed
  baseline duration factor) produce the same reading? Reported as
  rigidity_index[ARM_4_OVERLOAD_ON] vs rigidity_index[ARM_5_OVERLOAD_OFF];
  recorded, NOT gated (informational corroboration only).

INTERPRETATION GRID:
  readiness unmet (R1/R2/R3, >=1/3 seeds)      -> substrate_not_ready_requeue      (non_contributory, FAIL)
  C1 pass (both extremes clear, >=2/3 seeds each) -> mel_dose_rigidity_ushape_confirmed (PASS, supports)
  C1 fail AND monotonic on >=2/3 seeds          -> mel_dose_rigidity_monotonic_falsifies_ushape (FAIL, weakens)
  C1 fail AND flat (neither extreme clears, not monotonic) -> mel_dose_rigidity_flat_falsifies_ushape (FAIL, weakens)
  C1 fail, one extreme clears, not monotonic    -> mel_dose_rigidity_partial_ushape (FAIL, mixed)

OUT OF SCOPE (per INV-051's own what_would_answer, digested 2026-08-08 --
not required for a verdict on the core U-shape and not built here):
  (a) MECH-171's multi-year clinical vicious-cycle escalation (already
      out_of_domain; V3-EXQ-673 FAILed degenerate on its own test).
  (b) MECH-178's noradrenergic-hyperarousal coupling -- no NA/cortisol/LC
      substrate exists in ree_core. A PASS here supports "incomplete update
      at high MEL" but says nothing about WHY (reduced capacity vs
      demand-exceeds-fixed-ceiling); that mechanistic decomposition is a
      separate, still-substrate-gated question.

Also note for the record (not required, mentioned for auditability): a
STALE (2026-07-08) parked proposal EXP-0376 / backlog_id EVB-0111 exists in
REE_assembly/evidence/planning/experiment_proposals.v1.json for INV-051. Its
release_condition partially assumed a non-converging graded-MEL environment
did not yet exist (it now does -- SD-MEL-PRODUCER) and separately named the
MECH-178 NA plane as a release condition, which this run's freshly-digested
what_would_answer explicitly puts out of scope. Reconciling EXP-0376 itself
is left to a separate governance pass; this run does not edit it.
"""

import sys
import math
import time
import random
import argparse
import statistics
from collections import deque
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_901_inv051_mel_dose_rigidity_sweep"
QUEUE_ID = "V3-EXQ-901"
CLAIM_IDS = ["INV-051"]
EXPERIMENT_PURPOSE = "evidence"
EVIDENCE_DIR = Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments"

# z_goal_enabled=True is inherited verbatim from the V3-EXQ-845/718a/798a lineage
# for architecture parity, but no code path here calls update_z_goal -- the
# stream is inert, identically, in every arm (goal_weight/e1_goal_conditioned
# are structural config, not exercised by this driver). Wiring it live would
# activate the E3 goal term + E1 goal-conditioning + the SD-024 benefit-
# attractor producer, changing action selection and hence the rigidity
# readout this experiment routes on. The knob is arm-symmetric (identical in
# every arm), so the inertness cannot bias the cross-arm comparison.
DEAD_Z_GOAL_STREAM_EXEMPT = (
    "inherited verbatim from V3-EXQ-845/718a/798a for architecture parity; "
    "wiring update_z_goal would activate the E3 goal term, E1 conditioning, "
    "and the SD-024 benefit-attractor producer, confounding the rigidity "
    "readout this experiment routes on. Knob is arm-symmetric."
)

# -- Design parameters -------------------------------------------------------
SEEDS = [42, 123, 456]
CONV_EPISODES = 60            # P0 world-model convergence on the STABLE base env
STEPS_PER_EPISODE = 90
PROBE_BATTERY_SIZE = 64       # FIXED held-out battery for the convergence readiness probe
CALIB_EPISODES = 3            # stable-base MEL reference-calibration wake pass
MEAS_CYCLES = 6               # wake-sleep cycles per arm
WAKE_EPISODES_PER_CYCLE = 2   # wake episodes per cycle (populate buffers + MEL)
PROBE_EPISODES = 6            # rigidity probe: episodes on the fixed novel probe env.
                               # Doubled from an initial 3 because the load-bearing
                               # entropy tally is gated on fresh E3 selections only
                               # (ticks["e3_tick"], cadence default 10) -- most probe
                               # TICKS are held-action repeats, not fresh decisions, so
                               # more ticks are needed for a robust per-cell sample of
                               # genuine selections (see _run_rigidity_probe docstring).
PROBE_SEED_OFFSET = 900001    # probe env seed = cell seed + this (distinct layout)
# Progress denominator M (per cell): P0 + calibration + measurement wake + probe.
EPISODES_PER_RUN = (CONV_EPISODES + CALIB_EPISODES
                    + MEAS_CYCLES * WAKE_EPISODES_PER_CYCLE
                    + PROBE_EPISODES)

# Sleep pass base durations (same base as the 677/718/718a/845 lineage).
SWS_CONSOLIDATION_STEPS = 5
REM_ATTRIBUTION_STEPS = 10

# MEL consumer config (identical to the validated V3-EXQ-718a/845 test-bed).
MEL_GAIN = 1.0
FACTOR_MIN = 0.5
FACTOR_MAX = 3.0
MEL_RELATIVE_FLOOR = 1e-6     # relative floor only guards mel/ref against ref ~ 0

# E2 world-forward online training (recon-only; SD-056 auxiliary OFF at train time).
SD056_WEIGHT = 0.05
E2_LR = 1e-3
CONTRASTIVE_BATCH_K = 8
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0
TRANSITION_BUFFER_MAX = 256

# -- Thresholds (pre-registered constants, NOT derived from run stats) -------
MIN_REL_CONV_DROP = 0.10      # R1: per-seed frozen-probe PE drops at least 10%
SEED_PASS_FRAC = 2.0 / 3.0    # R / C1: at least 2/3 of seeds
MIN_MEL_SPREAD = 0.15         # R2: measured mean_mel[max] at least 15% above [min]
MONO_TOL = 0.05               # falsifying-monotonicity slack (relative to rigidity[min])
RIGIDITY_SD_MULT = 2.0        # C1: margin = max(SD_MULT * pstdev(delta), floor)
RIGIDITY_MIN_EFFECT_FLOOR = 0.05  # C1: absolute floor in rigidity_index units [0,1]
MIN_FRESH_SELECTIONS_PER_CELL = 15  # R4: minimum fresh e3_tick selections behind the
                                     # per-cell rigidity_index estimate (sample-size
                                     # integrity floor for the hold-weighted-readout fix)

# -- Environment base (identical to the V3-EXQ-798a/845/718a/701c lineage) ---
ENV_BASE: Dict[str, Any] = dict(
    size=12,
    num_hazards=4,
    num_resources=5,
    hazard_harm=0.05,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=10,
    use_proxy_fields=True,
)

# The stable base carries NO hazard drift, so the only non-stationarity in the
# graded arms is the SD-MEL-PRODUCER rule shift itself.
STABLE_DRIFT = dict(env_drift_interval=999, env_drift_prob=0.0)

# arm_id, level (nominal ordering, NOT what scoring sorts by), world-rule-shift
# interval/depth (798a's validated ladder for the first four; depth=3 at the
# validated interval=10 for the OVERLOAD arm -- see Step 2.5a probe note in
# the docstring), consumer on/off.
ARMS: List[Dict[str, Any]] = [
    {"arm_id": "ARM_0_MONOTONY_ON", "level": 0, "interval": 0,  "depth": 0, "mel_on": True},
    {"arm_id": "ARM_1_LOW_ON",      "level": 1, "interval": 60, "depth": 2, "mel_on": True},
    {"arm_id": "ARM_2_OPTIMAL_ON",  "level": 2, "interval": 25, "depth": 2, "mel_on": True},
    {"arm_id": "ARM_3_HIGH_ON",     "level": 3, "interval": 10, "depth": 2, "mel_on": True},
    {"arm_id": "ARM_4_OVERLOAD_ON", "level": 4, "interval": 10, "depth": 3, "mel_on": True},
    {"arm_id": "ARM_5_OVERLOAD_OFF","level": 4, "interval": 10, "depth": 3, "mel_on": False},
]
# The 5 ecological ON arms (readiness / C1 scored over these, sorted by
# MEASURED mean_mel per seed -- ARM_5_OVERLOAD_OFF is diagnostic-only, C2).
ON_ARMS = ["ARM_0_MONOTONY_ON", "ARM_1_LOW_ON", "ARM_2_OPTIMAL_ON",
           "ARM_3_HIGH_ON", "ARM_4_OVERLOAD_ON"]

ACTION_DIM_NOMINAL = 5   # CausalGridWorldV2 base ACTIONS map (no CONSUME action here)


def _make_env(seed: int, interval: int, depth: int) -> CausalGridWorldV2:
    kw = dict(ENV_BASE)
    kw.update(STABLE_DRIFT)
    kw.update(
        world_rule_shift_enabled=(interval > 0),
        world_rule_shift_interval=interval,
        world_rule_shift_depth=depth if interval > 0 else 0,
    )
    return CausalGridWorldV2(seed=seed, **kw)


def _make_agent(env: CausalGridWorldV2, mel_on: bool, mel_reference: float) -> REEAgent:
    """Converged-base agent (recon-only e2 training; encoder frozen) + SD-017
    SWS/REM passes + the SleepLoopManager. When mel_on, SD-MEL-CONSUMER is
    enabled with a FIXED reference set-point. Config is byte-identical to the
    validated V3-EXQ-718a/845 recipe (same architecture; only the arm's
    world_rule_shift knobs differ)."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        use_harm_stream=True,
        z_harm_dim=32,
        use_affective_harm_stream=True,
        z_harm_a_dim=16,
        harm_history_len=10,
        z_goal_enabled=True,
        goal_weight=0.5,
        drive_weight=2.0,
        e1_goal_conditioned=True,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        benefit_eval_enabled=True,
        benefit_weight=1.0,
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=2.0,
        surprise_gated_replay=True,
        # SD-017 sleep passes + SleepLoopManager (no aggregation cluster needed).
        use_sleep_loop=True,
        sleep_loop_episodes_K=10**9,   # never auto-fire; we drive via force_cycle
        sws_enabled=True,
        sws_consolidation_steps=SWS_CONSOLIDATION_STEPS,
        rem_enabled=True,
        rem_attribution_steps=REM_ATTRIBUTION_STEPS,
        # SD-MEL-CONSUMER (GAP-5b) -- fixed reference set-point.
        use_mel_consumer=bool(mel_on),
        mel_gain=MEL_GAIN,
        mel_reference=float(mel_reference),
        mel_reference_mode="fixed",
        mel_duration_factor_min=FACTOR_MIN,
        mel_duration_factor_max=FACTOR_MAX,
        mel_relative_floor=MEL_RELATIVE_FLOOR,
        mel_scale_sws=True,
        mel_scale_rem=True,
        use_mel_entry=False,
    )
    return REEAgent(cfg)


def _obs(d: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    h = d.get(key)
    if h is None:
        return None
    return h.float().unsqueeze(0) if h.dim() == 1 else h.float()


def _sense_latent(agent: REEAgent, obs_dict: Dict[str, Any]):
    body = obs_dict["body_state"].float()
    world = obs_dict["world_state"].float()
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    return agent.sense(
        obs_body=body, obs_world=world,
        obs_harm=_obs(obs_dict, "harm_obs"),
        obs_harm_a=_obs(obs_dict, "harm_obs_a"),
        obs_harm_history=_obs(obs_dict, "harm_history"),
    )


def _select_or_fallback(agent: REEAgent, env: CausalGridWorldV2,
                        latent, ticks) -> torch.Tensor:
    wdim = latent.z_world.shape[-1]
    e1_prior = (agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device))
    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
    action = agent.select_action(candidates, ticks)
    if action is None:
        idx = int(np.random.randint(0, env.action_dim))
        action = torch.zeros(1, env.action_dim, device=agent.device)
        action[0, idx] = 1.0
        agent._last_action = action
    return action


def _sample_class_diverse_batch(
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    k: int, rng: random.Random,
) -> Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    if len(buffer) < MIN_BUFFER_BEFORE_TRAIN:
        return None
    pool = list(buffer)
    rng.shuffle(pool)
    seen: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for tup in pool:
        cls = int(tup[1].argmax().item())
        if cls not in seen:
            seen[cls] = tup
        if len(seen) >= k:
            break
    if len(seen) < MIN_CLASSES_FOR_TRAIN:
        return None
    samples = list(seen.values())
    picked = {id(s) for s in samples}
    for tup in pool:
        if len(samples) >= k:
            break
        if id(tup) in picked:
            continue
        samples.append(tup)
        picked.add(id(tup))
    return samples


def _e2_train_step(
    agent: REEAgent,
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    optimiser: torch.optim.Optimizer, rng: random.Random,
) -> Optional[float]:
    """One recon-only P0 world-forward training step (reconstruction MSE)."""
    batch = _sample_class_diverse_batch(buffer, CONTRASTIVE_BATCH_K, rng)
    if batch is None:
        return None
    z0_K = torch.stack([t[0] for t in batch]).to(agent.device)
    actions_K = torch.stack([t[1] for t in batch]).to(agent.device)
    z1_K = torch.stack([t[2] for t in batch]).to(agent.device)
    optimiser.zero_grad(set_to_none=True)
    z1_pred = agent.e2.world_forward(z0_K, actions_K)
    recon = F.mse_loss(z1_pred, z1_K)
    recon_val = float(recon.detach().item())
    if not math.isfinite(recon_val):
        return recon_val
    recon.backward()
    torch.nn.utils.clip_grad_norm_(agent.e2.parameters(), max_norm=MAX_GRAD_NORM)
    optimiser.step()
    return recon_val


def _waking_step(
    agent: REEAgent, env: CausalGridWorldV2, obs_dict: Dict[str, Any],
    train: bool, buffer: Optional[Deque],
    e2_opt: Optional[torch.optim.Optimizer], sample_rng: Optional[random.Random],
    pending_capture_ref: List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
) -> Tuple[Dict[str, Any], bool]:
    """One waking step. Always calls agent.update_residue() (hypothesis_tag=False)
    so the MEL consumer accumulates per-step e3 prediction error. Returns
    (next_obs_dict, done)."""
    latent = _sense_latent(agent, obs_dict)

    if train and buffer is not None:
        pend = pending_capture_ref[0]
        if pend is not None:
            z0_prev, a_prev = pend
            z1_obs = latent.z_world.detach().reshape(-1).clone()
            if (torch.isfinite(z0_prev).all() and torch.isfinite(a_prev).all()
                    and torch.isfinite(z1_obs).all()):
                buffer.append((z0_prev, a_prev, z1_obs))
            pending_capture_ref[0] = None

    ticks = agent.clock.advance()
    action = _select_or_fallback(agent, env, latent, ticks)
    if not torch.isfinite(action).all():
        return obs_dict, True

    if train and buffer is not None and torch.isfinite(latent.z_world).all():
        pending_capture_ref[0] = (
            latent.z_world.detach().reshape(-1).clone(),
            action.detach().reshape(-1).clone(),
        )
        if e2_opt is not None and sample_rng is not None:
            _e2_train_step(agent, buffer, e2_opt, sample_rng)

    _, harm_signal, done, info, next_obs_dict = env.step(action)
    with torch.no_grad():
        agent.update_residue(
            harm_signal=float(harm_signal), world_delta=None,
            hypothesis_tag=False, owned=True,
        )
    return next_obs_dict, bool(done)


def _run_wake_window(
    agent: REEAgent, env: CausalGridWorldV2, n_episodes: int, steps: int,
    train: bool, buffer: Optional[Deque],
    e2_opt: Optional[torch.optim.Optimizer], sample_rng: Optional[random.Random],
    ep_offset: int, arm_id: str, seed: int,
) -> None:
    """Run n_episodes of waking on env. During P0 (train=True) trains e2 recon-only.
    During measurement (train=False) just drives the agent + accumulates MEL + warms
    the agent's experience buffers."""
    pending_capture_ref: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None]
    for ep in range(n_episodes):
        glob_ep = ep_offset + ep
        if (glob_ep % 10 == 0) or (glob_ep == EPISODES_PER_RUN - 1):
            print(f"  [train] {arm_id} seed={seed} ep {glob_ep+1}/{EPISODES_PER_RUN}",
                  flush=True)
        _, obs_dict = env.reset()
        agent.reset()
        agent.e1.reset_hidden_state()
        pending_capture_ref[0] = None
        for _step in range(steps):
            obs_dict, done = _waking_step(
                agent, env, obs_dict, train, buffer, e2_opt, sample_rng,
                pending_capture_ref,
            )
            if done:
                break


# -- FROZEN held-out probe battery (the V3-EXQ-701b/c convergence instrument) --
def _sample_probe_battery(
    agent: REEAgent, seed: int, n_transitions: int, steps: int,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    env = _make_env(seed, interval=0, depth=0)
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    act_rng = random.Random(seed + 9973)
    battery: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    prev: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    guard = 0
    max_guard = max(steps, 1) * 8
    while len(battery) < n_transitions and guard < max_guard:
        guard += 1
        latent = _sense_latent(agent, obs_dict)
        if not torch.isfinite(latent.z_world).all():
            break
        z_now = latent.z_world.detach().reshape(1, -1).clone()
        if prev is not None:
            z0, a = prev
            battery.append((z0, a, z_now))
        idx = act_rng.randrange(env.action_dim)
        action = torch.zeros(1, env.action_dim, device=agent.device)
        action[0, idx] = 1.0
        _, _, done, _, obs_dict = env.step(action)
        prev = (z_now, action)
        if done:
            _, obs_dict = env.reset()
            agent.reset()
            agent.e1.reset_hidden_state()
            prev = None
    return battery


def _frozen_probe_pe(
    agent: REEAgent, battery: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> float:
    """Mean one-step world_forward reconstruction error over the FIXED battery.
    The world-model convergence metric (readiness only)."""
    if not battery:
        return 0.0
    errs: List[float] = []
    with torch.no_grad():
        for z0, a, z1 in battery:
            pred = agent.e2.world_forward(z0.to(agent.device), a.to(agent.device))
            err = float((pred - z1.to(agent.device)).pow(2).mean().item())
            if math.isfinite(err):
                errs.append(err)
    return float(np.mean(errs)) if errs else 0.0


def _entropy_and_rigidity(counts: List[int], n_total: int,
                          action_dim: int) -> Dict[str, Any]:
    if n_total == 0:
        entropy_max = math.log(action_dim) if action_dim > 0 else 0.0
        return {"rigidity_index": 1.0, "policy_entropy": 0.0,
                "policy_entropy_max": entropy_max, "n_actions_total": 0,
                "action_counts": list(counts), "degenerate_no_actions": True}
    probs = [c / n_total for c in counts if c > 0]
    entropy = -sum(p * math.log(p) for p in probs)
    entropy_max = math.log(action_dim) if action_dim > 0 else 0.0
    rigidity_index = 1.0 - (entropy / entropy_max) if entropy_max > 0 else 1.0
    rigidity_index = float(min(1.0, max(0.0, rigidity_index)))
    return {
        "rigidity_index": rigidity_index,
        "policy_entropy": float(entropy),
        "policy_entropy_max": float(entropy_max),
        "n_actions_total": n_total,
        "action_counts": list(counts),
        "degenerate_no_actions": False,
    }


def _run_rigidity_probe(agent: REEAgent, seed: int, n_episodes: int,
                        steps: int) -> Dict[str, Any]:
    """RIGIDITY DV: run the trained agent (no further training) on a FIXED,
    NOVEL probe env (seed distinct from the arm's training env, world_rule_shift
    OFF) and record the Shannon entropy of the resulting action distribution.
    rigidity_index = 1 - H/H_max (higher = more behaviourally stereotyped).

    HOLD-WEIGHTED E3 READOUT GUARD (validate_experiments.py finding, this
    session): agent.py returns the HELD action on `not ticks["e3_tick"]`
    BEFORE e3.select() is reached (E3 cadence default 10, MECH-093 arousal-
    varying 5-20) -- so a per-TICK action tally is weighted by hold DURATION,
    not by decision count, and the validator flags this as DISQUALIFYING for
    a distribution-shape statistic (entropy) precisely because it is what
    this DV is. If cadence itself differs systematically across MEL-dose
    arms (plausible -- MECH-093 arousal could differ with environmental
    novelty), a hold-weighted entropy would confound "the agent chose
    differently" with "the agent's E3 cadence differed", which is not what
    INV-051 asks. FIX (per experiments/v3_exq_785a's pattern): gate the
    PRIMARY (load-bearing) tally on `ticks["e3_tick"]` -- only count a tick
    where a fresh e3.select() genuinely fired, so entropy is computed over
    decisions, not ticks. The hold-weighted (as-EXECUTED) distribution is
    also recorded, clearly labelled, as a non-load-bearing diagnostic only.
    """
    probe_env = _make_env(seed + PROBE_SEED_OFFSET, interval=0, depth=0)
    fresh_counts = [0] * probe_env.action_dim
    executed_counts = [0] * probe_env.action_dim
    n_fresh = 0
    n_executed = 0
    n_latched = 0
    _, obs_dict = probe_env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    for ep in range(n_episodes):
        _, obs_dict = probe_env.reset()
        agent.reset()
        agent.e1.reset_hidden_state()
        for _step in range(steps):
            latent = _sense_latent(agent, obs_dict)
            if not torch.isfinite(latent.z_world).all():
                break
            ticks = agent.clock.advance()
            fresh_select = bool(ticks.get("e3_tick", False))
            action = _select_or_fallback(agent, probe_env, latent, ticks)
            if not torch.isfinite(action).all():
                break
            idx = int(action.argmax(dim=-1).item())
            if 0 <= idx < len(executed_counts):
                executed_counts[idx] += 1
                n_executed += 1
                if fresh_select:
                    fresh_counts[idx] += 1
                    n_fresh += 1
                else:
                    n_latched += 1
            _, _, done, _, obs_dict = probe_env.step(action)
            if done:
                break

    fresh = _entropy_and_rigidity(fresh_counts, n_fresh, probe_env.action_dim)
    executed = _entropy_and_rigidity(executed_counts, n_executed, probe_env.action_dim)
    fresh_select_yield = (n_fresh / n_executed) if n_executed > 0 else 0.0

    out = dict(fresh)  # PRIMARY / load-bearing fields at top level (backward-compatible keys)
    out["n_fresh_select"] = n_fresh
    out["n_latched"] = n_latched
    out["n_executed_total"] = n_executed
    out["fresh_select_yield"] = float(fresh_select_yield)
    out["executed_rigidity_index"] = executed["rigidity_index"]      # hold-weighted, diagnostic only
    out["executed_policy_entropy"] = executed["policy_entropy"]      # hold-weighted, diagnostic only
    out["executed_action_counts"] = executed["action_counts"]
    return out


# z_goal-stream liveness, pooled across the run's per-cell agents.
_ZG = ZGoalStreamAccumulator()


def _run_cell(seed: int, arm: Dict[str, Any], steps: int, conv_eps: int,
              meas_cycles: int) -> Dict[str, Any]:
    """One (seed, arm) cell: build agent, converge P0 recon-only on the stable
    base, calibrate the MEL reference to the stable-base measurement-modality
    MEL, run the wake-sleep measurement cycles on the arm's SD-MEL-PRODUCER
    env, then run the fixed novel rigidity probe."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    arm_id = arm["arm_id"]
    mel_on = bool(arm["mel_on"])
    print(f"Seed {seed} Condition {arm_id}", flush=True)

    # ONE agent per cell. Encoder FROZEN (only agent.e2 trains). Build with
    # mel_reference=0.0, then fix it to the calibrated stable-base MEL AFTER
    # P0 (mode="fixed", so the >0 reference is used and never auto-relocked).
    stable_env = _make_env(seed, interval=0, depth=0)
    agent = _make_agent(stable_env, mel_on=mel_on, mel_reference=0.0)
    battery = _sample_probe_battery(agent, seed, PROBE_BATTERY_SIZE, steps)
    probe_pe_init = _frozen_probe_pe(agent, battery)

    buffer: Deque = deque(maxlen=TRANSITION_BUFFER_MAX)
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_LR)
    sample_rng = random.Random(seed + 4242)

    _run_wake_window(
        agent, stable_env, conv_eps, steps, train=True, buffer=buffer,
        e2_opt=e2_opt, sample_rng=sample_rng, ep_offset=0, arm_id=arm_id, seed=seed,
    )

    probe_pe_final = _frozen_probe_pe(agent, battery)
    conv_rel_drop = (((probe_pe_init - probe_pe_final) / probe_pe_init)
                     if probe_pe_init > 1e-12 else 0.0)

    # Reference calibration: a stable-base wake pass measured through the SAME
    # path as the measurement MEL (agent.update_residue -> e3 PE -> consumer
    # accumulator). MONOTONY arm (stable env, no shift) then yields factor
    # ~1.0; higher-shift arms scale above it.
    if agent.mel_consumer is not None:
        agent.mel_consumer.reset()   # clear P0 accumulation
    _run_wake_window(
        agent, stable_env, CALIB_EPISODES, steps, train=False, buffer=None,
        e2_opt=None, sample_rng=None, ep_offset=conv_eps, arm_id=arm_id, seed=seed,
    )
    if mel_on and agent.mel_consumer is not None:
        base_ref = float(agent.mel_consumer.current_mel())
        if not (base_ref > 0.0):     # degenerate fallback (no PE accumulated)
            base_ref = float(probe_pe_final)
        agent.config.mel_reference = base_ref
        agent.mel_consumer.config.mel_reference = base_ref
        agent.mel_consumer.reset()   # clean slate for the first measurement cycle
    else:
        base_ref = float(probe_pe_final)   # OFF arm: reference unused

    # Measurement: MEAS_CYCLES wake-sleep cycles on the arm's SD-MEL-PRODUCER
    # env (world_rule_shift is the ONLY non-stationarity; env_drift stays at
    # the STABLE_DRIFT sentinel in every arm).
    meas_env = _make_env(seed, arm["interval"], arm["depth"])
    cum_sws = 0.0
    cum_rem = 0.0
    per_cycle_sws: List[float] = []
    per_cycle_rem: List[float] = []
    per_cycle_diversity: List[float] = []
    per_cycle_factor: List[float] = []
    factors: List[float] = []
    mels: List[float] = []
    n_cycles_fired = 0
    ep_off = conv_eps + CALIB_EPISODES
    for _cyc in range(meas_cycles):
        _run_wake_window(
            agent, meas_env, WAKE_EPISODES_PER_CYCLE, steps, train=False,
            buffer=None, e2_opt=None, sample_rng=None, ep_offset=ep_off,
            arm_id=arm_id, seed=seed,
        )
        ep_off += WAKE_EPISODES_PER_CYCLE
        m = agent.sleep_loop.force_cycle(agent)
        if m is not None:
            n_cycles_fired += 1
        m = m or {}
        sws = float(m.get("sws_n_writes", 0.0))
        rem = float(m.get("rem_n_rollouts", 0.0))
        diversity = float(m.get("sws_slot_diversity", 0.0))
        factor = float(m.get("mel_duration_factor", 1.0))
        cum_sws += sws
        cum_rem += rem
        per_cycle_sws.append(sws)
        per_cycle_rem.append(rem)
        per_cycle_diversity.append(diversity)
        per_cycle_factor.append(factor)
        if mel_on:
            factors.append(factor)
            mels.append(float(m.get("mel_mean", 0.0)))

    mean_diversity = float(np.mean(per_cycle_diversity)) if per_cycle_diversity else 0.0
    mean_factor = float(np.mean(factors)) if factors else 1.0
    mean_mel = float(np.mean(mels)) if mels else 0.0
    min_factor_seen = float(min(per_cycle_factor)) if per_cycle_factor else 1.0

    # RIGIDITY DV: fixed novel probe, no further training.
    rigidity = _run_rigidity_probe(agent, seed, PROBE_EPISODES, steps)

    _ZG.observe(agent)

    print(f"    {arm_id} seed={seed}: conv_drop={conv_rel_drop:.3f} "
          f"ref={base_ref:.3e} mel={mean_mel:.3e} factor={mean_factor:.3f} "
          f"rigidity={rigidity['rigidity_index']:.4f} "
          f"cycles_fired={n_cycles_fired}/{meas_cycles} min_factor={min_factor_seen:.3f}",
          flush=True)
    print(f"verdict: {'PASS' if conv_rel_drop >= MIN_REL_CONV_DROP else 'FAIL'}",
          flush=True)

    return {
        "arm_id": arm_id,
        "level": arm["level"],
        "mel_on": mel_on,
        "world_rule_shift_interval": arm["interval"],
        "world_rule_shift_depth": arm["depth"],
        "seed": seed,
        "conv_rel_drop": conv_rel_drop,
        "probe_pe_init": probe_pe_init,
        "probe_pe_final": probe_pe_final,
        "mel_reference": base_ref,
        "mean_mel": mean_mel,
        "mean_duration_factor": mean_factor,
        "min_duration_factor": min_factor_seen,
        "cumulative_sws_writes": cum_sws,
        "cumulative_rem_rollouts": cum_rem,
        "mean_sws_slot_diversity": mean_diversity,
        "per_cycle_sws": per_cycle_sws,
        "per_cycle_rem": per_cycle_rem,
        "per_cycle_diversity": per_cycle_diversity,
        "per_cycle_factor": per_cycle_factor,
        "per_cycle_mel": mels,
        "n_cycles_fired": n_cycles_fired,
        "meas_cycles": meas_cycles,
        "rigidity_index": rigidity["rigidity_index"],
        "policy_entropy": rigidity["policy_entropy"],
        "policy_entropy_max": rigidity["policy_entropy_max"],
        "rigidity_n_actions_total": rigidity["n_actions_total"],
        "rigidity_action_counts": rigidity["action_counts"],
        "rigidity_degenerate_no_actions": rigidity["degenerate_no_actions"],
        # Sample-size integrity (hold-weighted E3 readout guard): the
        # rigidity_index above is computed ONLY from fresh e3_tick selections
        # (n_fresh_select of them); these fields make the true denominator
        # auditable and carry the hold-weighted (as-executed) variant as a
        # separate, non-load-bearing diagnostic.
        "rigidity_n_fresh_select": rigidity["n_fresh_select"],
        "rigidity_n_latched": rigidity["n_latched"],
        "rigidity_n_executed_total": rigidity["n_executed_total"],
        "rigidity_fresh_select_yield": rigidity["fresh_select_yield"],
        "rigidity_executed_index_diagnostic_only": rigidity["executed_rigidity_index"],
        "rigidity_executed_entropy_diagnostic_only": rigidity["executed_policy_entropy"],
    }


def _pstdev_or_zero(xs: List[float]) -> float:
    return float(statistics.pstdev(xs)) if len(xs) >= 2 else 0.0


def _seed_readiness(on_cells: List[Dict[str, Any]], meas_cycles: int) -> Dict[str, Any]:
    """R1 (world-model trained) AND R2 (MEL-dose sweep non-degenerate) AND
    R3 (sleep opportunity uniform across dose -- the content-not-opportunity
    control) AND R4 (adequate fresh-selection sample behind rigidity_index)
    for one seed's 5 ON arms."""
    if len(on_cells) != len(ON_ARMS):
        return {"r1_ok": False, "r2_ok": False, "r3_ok": False, "r4_ok": False, "ready": False}
    r1_ok = all(r["conv_rel_drop"] >= MIN_REL_CONV_DROP for r in on_cells)
    arms_sorted = sorted(on_cells, key=lambda r: r["mean_mel"])
    mels = [r["mean_mel"] for r in arms_sorted]
    r2_ok = mels[0] > 0 and mels[-1] >= mels[0] * (1 + MIN_MEL_SPREAD)
    r3_ok = all(r["n_cycles_fired"] == meas_cycles for r in on_cells) and \
        all(r["min_duration_factor"] >= FACTOR_MIN - 1e-9 for r in on_cells)
    r4_ok = all(r["rigidity_n_fresh_select"] >= MIN_FRESH_SELECTIONS_PER_CELL
               for r in on_cells)
    return {"r1_ok": bool(r1_ok), "r2_ok": bool(r2_ok), "r3_ok": bool(r3_ok),
            "r4_ok": bool(r4_ok),
            "ready": bool(r1_ok and r2_ok and r3_ok and r4_ok)}


def _seed_ushape(on_by_arm: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Sort the 5 ON arms by MEASURED mean_mel; LOW=idx0, MID=idx2, HIGH=idx4.
    Returns per-seed delta_low / delta_high and the monotonicity check.
    Margins are computed OUTSIDE this function (need the full cross-seed set)."""
    arms_sorted = sorted(on_by_arm.values(), key=lambda r: r["mean_mel"])
    arm_order = [r["arm_id"] for r in arms_sorted]
    mels = [r["mean_mel"] for r in arms_sorted]
    rigidity = [r["rigidity_index"] for r in arms_sorted]
    low, mid, high = rigidity[0], rigidity[2], rigidity[4]
    delta_low = low - mid
    delta_high = high - mid
    tol = MONO_TOL * max(min(rigidity), 1e-9)
    non_decreasing = all(rigidity[i] <= rigidity[i + 1] + tol for i in range(len(rigidity) - 1))
    non_increasing = all(rigidity[i] >= rigidity[i + 1] - tol for i in range(len(rigidity) - 1))
    monotone = bool(non_decreasing or non_increasing)
    return {
        "arm_order_by_measured_mel": arm_order,
        "mel_by_measured_order": mels,
        "rigidity_by_measured_order": rigidity,
        "rigidity_low": low, "rigidity_mid": mid, "rigidity_high": high,
        "delta_low": delta_low, "delta_high": delta_high,
        "monotone": monotone,
        "non_decreasing": bool(non_decreasing), "non_increasing": bool(non_increasing),
    }


def run_experiment(steps: int, conv_eps: int, meas_cycles: int,
                   seeds: List[int], arms: Optional[List[Dict[str, Any]]] = None,
                   ) -> Dict[str, Any]:
    arms = arms if arms is not None else ARMS
    arm_results: List[Dict[str, Any]] = []
    for seed in seeds:
        for arm in arms:
            full_config = {
                "env_base": ENV_BASE,
                "arm": arm,
                "conv_episodes": conv_eps,
                "meas_cycles": meas_cycles,
                "steps_per_episode": steps,
                "sws_steps": SWS_CONSOLIDATION_STEPS,
                "rem_steps": REM_ATTRIBUTION_STEPS,
                "mel_gain": MEL_GAIN,
                "factor_min": FACTOR_MIN,
                "factor_max": FACTOR_MAX,
                "mel_relative_floor": MEL_RELATIVE_FLOOR,
                "probe_episodes": PROBE_EPISODES,
                "probe_seed_offset": PROBE_SEED_OFFSET,
            }
            with arm_cell(seed, config_slice=full_config,
                          script_path=Path(__file__)) as cell:
                row = _run_cell(seed, arm, steps, conv_eps, meas_cycles)
                cell.stamp(row)
            arm_results.append(row)

    # -- Readiness --
    seed_ready: Dict[int, bool] = {}
    seed_readiness_detail: Dict[int, Dict[str, Any]] = {}
    for seed in seeds:
        on_cells = [r for r in arm_results if r["seed"] == seed and r["arm_id"] in ON_ARMS]
        rd = _seed_readiness(on_cells, meas_cycles)
        seed_readiness_detail[seed] = rd
        seed_ready[seed] = rd["ready"]
    readiness_frac = sum(seed_ready.values()) / max(1, len(seeds))
    r1_frac = sum(1 for s in seeds if seed_readiness_detail[s]["r1_ok"]) / max(1, len(seeds))
    r2_frac = sum(1 for s in seeds if seed_readiness_detail[s]["r2_ok"]) / max(1, len(seeds))
    r3_frac = sum(1 for s in seeds if seed_readiness_detail[s]["r3_ok"]) / max(1, len(seeds))
    r4_frac = sum(1 for s in seeds if seed_readiness_detail[s]["r4_ok"]) / max(1, len(seeds))
    readiness_ok = readiness_frac >= SEED_PASS_FRAC

    # -- Per-seed U-shape geometry (ready seeds only for scoring; all seeds recorded) --
    per_seed_ushape: Dict[int, Optional[Dict[str, Any]]] = {}
    for seed in seeds:
        on_by_arm = {r["arm_id"]: r for r in arm_results
                    if r["seed"] == seed and r["arm_id"] in ON_ARMS}
        if len(on_by_arm) == len(ON_ARMS):
            per_seed_ushape[seed] = _seed_ushape(on_by_arm)
        else:
            per_seed_ushape[seed] = None

    ready_seeds = [s for s in seeds if seed_ready[s]]
    deltas_low = [per_seed_ushape[s]["delta_low"] for s in ready_seeds if per_seed_ushape[s]]
    deltas_high = [per_seed_ushape[s]["delta_high"] for s in ready_seeds if per_seed_ushape[s]]
    margin_low = max(RIGIDITY_SD_MULT * _pstdev_or_zero(deltas_low), RIGIDITY_MIN_EFFECT_FLOOR)
    margin_high = max(RIGIDITY_SD_MULT * _pstdev_or_zero(deltas_high), RIGIDITY_MIN_EFFECT_FLOOR)

    clears_low_seeds = [s for s in ready_seeds
                        if per_seed_ushape[s] and per_seed_ushape[s]["delta_low"] > margin_low]
    clears_high_seeds = [s for s in ready_seeds
                         if per_seed_ushape[s] and per_seed_ushape[s]["delta_high"] > margin_high]
    monotone_seeds = [s for s in ready_seeds
                      if per_seed_ushape[s] and per_seed_ushape[s]["monotone"]]

    n_ready = max(1, len(ready_seeds))
    clears_low_frac = len(clears_low_seeds) / n_ready if ready_seeds else 0.0
    clears_high_frac = len(clears_high_seeds) / n_ready if ready_seeds else 0.0
    monotone_frac = len(monotone_seeds) / n_ready if ready_seeds else 0.0

    c1_low_pass = clears_low_frac >= SEED_PASS_FRAC
    c1_high_pass = clears_high_frac >= SEED_PASS_FRAC
    c1_pass = bool(c1_low_pass and c1_high_pass)   # COMBINATION RULE: conjunctive, both sides.
    falsifying_monotonic = monotone_frac >= SEED_PASS_FRAC

    # -- C2 control (non-load-bearing): consumer ON vs OFF at matched OVERLOAD novelty --
    c2_by_seed: Dict[int, Optional[Dict[str, Any]]] = {}
    for seed in seeds:
        on_cell = next((r for r in arm_results if r["seed"] == seed
                        and r["arm_id"] == "ARM_4_OVERLOAD_ON"), None)
        off_cell = next((r for r in arm_results if r["seed"] == seed
                         and r["arm_id"] == "ARM_5_OVERLOAD_OFF"), None)
        if on_cell and off_cell:
            c2_by_seed[seed] = {
                "rigidity_on": on_cell["rigidity_index"],
                "rigidity_off": off_cell["rigidity_index"],
                "consumer_reduces_rigidity": bool(
                    on_cell["rigidity_index"] <= off_cell["rigidity_index"]),
            }
        else:
            c2_by_seed[seed] = None
    c2_hits = sum(1 for v in c2_by_seed.values() if v and v["consumer_reduces_rigidity"])
    c2_evaluable = sum(1 for v in c2_by_seed.values() if v)
    c2_frac = (c2_hits / c2_evaluable) if c2_evaluable else 0.0

    # -- Self-route --
    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        direction = "non_contributory"
    elif c1_pass:
        label = "mel_dose_rigidity_ushape_confirmed"
        outcome = "PASS"
        direction = "supports"
    elif falsifying_monotonic:
        label = "mel_dose_rigidity_monotonic_falsifies_ushape"
        outcome = "FAIL"
        direction = "weakens"
    elif not c1_low_pass and not c1_high_pass:
        label = "mel_dose_rigidity_flat_falsifies_ushape"
        outcome = "FAIL"
        direction = "weakens"
    else:
        label = "mel_dose_rigidity_partial_ushape"
        outcome = "FAIL"
        direction = "mixed"

    c1_gradient_present = readiness_ok and bool(ready_seeds)
    ready_rigidity_values = [r["rigidity_index"] for r in arm_results
                             if r["arm_id"] in ON_ARMS and seed_ready.get(r["seed"], False)]
    c1_dv_spread_nonzero = (
        len(set(round(v, 4) for v in ready_rigidity_values)) > 1
    ) if ready_rigidity_values else False

    interpretation = {
        "label": label,
        "combination_rule": (
            "C1 (load-bearing) is CONJUNCTIVE: clears_low (>=SEED_PASS_FRAC "
            "seeds with delta_low > margin_low) AND clears_high (>=SEED_PASS_FRAC "
            "seeds with delta_high > margin_high). Neither side alone suffices -- "
            "a single elevated extreme is a partial/mixed finding "
            "(mel_dose_rigidity_partial_ushape), not a confirmed U-shape."
        ),
        "preconditions": [
            {
                "name": "world_forward_converged_frozen_probe",
                "description": "recon-only P0 converges on the fixed frozen-probe "
                               "battery (conv_rel_drop >= MIN_REL_CONV_DROP) on the "
                               "5 ON arms, on >= 2/3 seeds (R1).",
                "measured": r1_frac, "threshold": SEED_PASS_FRAC,
                "direction": "lower", "met": bool(r1_frac >= SEED_PASS_FRAC),
            },
            {
                "name": "mel_dose_sweep_gradient_present_this_config",
                "description": "measured mean_mel is non-degenerately graded "
                               "(max >= min*(1+MIN_MEL_SPREAD)) across the 5 ON "
                               "arms, on >= 2/3 seeds -- without this, dose is "
                               "not actually varying (R2).",
                "measured": r2_frac, "threshold": SEED_PASS_FRAC,
                "direction": "lower", "met": bool(r2_frac >= SEED_PASS_FRAC),
            },
            {
                "name": "sleep_opportunity_uniform_across_dose",
                "description": "every ON arm/seed fired exactly meas_cycles sleep "
                               "cycles AND mel_duration_factor never fell below "
                               "FACTOR_MIN -- the content-not-opportunity control "
                               "for INV-051's falsifying condition (c): rules out "
                               "'less sleep happened' as the explanation for any "
                               "low-MEL rigidity reading, since sleep opportunity "
                               "is structurally identical across every dose arm (R3).",
                "measured": r3_frac, "threshold": SEED_PASS_FRAC,
                "direction": "lower", "met": bool(r3_frac >= SEED_PASS_FRAC),
            },
            {
                "name": "rigidity_fresh_selection_sample_adequate",
                "description": "every ON arm/seed's rigidity_index rests on "
                               ">= MIN_FRESH_SELECTIONS_PER_CELL fresh e3_tick "
                               "selections (not held/latched repeats) -- the "
                               "sample-size-integrity floor added after the "
                               "hold-weighted-E3-readout validator finding "
                               "(this session): the primary tally is gated on "
                               "ticks['e3_tick'] to avoid an E3-cadence "
                               "confound, which shrinks the effective n per "
                               "cell and needs its own floor (R4).",
                "measured": r4_frac, "threshold": SEED_PASS_FRAC,
                "direction": "lower", "met": bool(r4_frac >= SEED_PASS_FRAC),
            },
        ],
        "criteria_non_degenerate": {
            "C1_measured_mel_gradient_present": bool(c1_gradient_present),
            "C1_rigidity_spread_nonzero": bool(c1_dv_spread_nonzero),
            "C2_off_control_present": bool(c2_evaluable > 0),
        },
        "margin_low": margin_low,
        "margin_high": margin_high,
        "clears_low_frac": clears_low_frac,
        "clears_high_frac": clears_high_frac,
        "monotone_frac": monotone_frac,
        "per_seed_ushape": {str(s): per_seed_ushape[s] for s in seeds},
        "c2_by_seed": {str(s): c2_by_seed[s] for s in seeds},
        "c2_frac_consumer_reduces_rigidity": c2_frac,
    }
    criteria = [
        {"name": "C1_low_extreme_clears_margin_vs_mid", "load_bearing": True,
         "passed": bool(c1_low_pass)},
        {"name": "C1_high_extreme_clears_margin_vs_mid", "load_bearing": True,
         "passed": bool(c1_high_pass)},
        {"name": "falsifying_monotonic_in_dose", "load_bearing": False,
         "passed": bool(falsifying_monotonic)},
        {"name": "C2_consumer_reduces_overload_rigidity_vs_off", "load_bearing": False,
         "passed": bool(c2_frac >= SEED_PASS_FRAC) if c2_evaluable else False},
    ]

    return {
        "outcome": outcome,
        "evidence_direction": direction,
        "interpretation": interpretation,
        "criteria": criteria,
        "readiness_ok": readiness_ok,
        "readiness_frac": readiness_frac,
        "r1_frac": r1_frac, "r2_frac": r2_frac, "r3_frac": r3_frac, "r4_frac": r4_frac,
        "c1_pass": c1_pass,
        "c1_low_pass": c1_low_pass, "c1_high_pass": c1_high_pass,
        "clears_low_frac": clears_low_frac, "clears_high_frac": clears_high_frac,
        "falsifying_monotonic": falsifying_monotonic, "monotone_frac": monotone_frac,
        "per_seed": [
            {"seed": s, "ready": seed_ready[s], "readiness_detail": seed_readiness_detail[s],
             "ushape": per_seed_ushape[s], "c2": c2_by_seed[s]}
            for s in seeds
        ],
        "arm_results": arm_results,
        "thresholds": {
            "MIN_REL_CONV_DROP": MIN_REL_CONV_DROP,
            "SEED_PASS_FRAC": SEED_PASS_FRAC,
            "MIN_MEL_SPREAD": MIN_MEL_SPREAD,
            "MONO_TOL": MONO_TOL,
            "RIGIDITY_SD_MULT": RIGIDITY_SD_MULT,
            "RIGIDITY_MIN_EFFECT_FLOOR": RIGIDITY_MIN_EFFECT_FLOOR,
            "MEL_GAIN": MEL_GAIN,
            "FACTOR_MIN": FACTOR_MIN,
            "FACTOR_MAX": FACTOR_MAX,
            "MEL_RELATIVE_FLOOR": MEL_RELATIVE_FLOOR,
        },
    }


def write_manifest(result: Dict[str, Any], *, dry_run: bool = False,
                   started_at: Optional[float] = None) -> str:
    ts = dt.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    full_config = {
        "env_base": ENV_BASE,
        "arms": ARMS,
        "conv_episodes": CONV_EPISODES,
        "calib_episodes": CALIB_EPISODES,
        "meas_cycles": MEAS_CYCLES,
        "wake_episodes_per_cycle": WAKE_EPISODES_PER_CYCLE,
        "probe_episodes": PROBE_EPISODES,
        "probe_seed_offset": PROBE_SEED_OFFSET,
        "steps_per_episode": STEPS_PER_EPISODE,
        "sws_steps": SWS_CONSOLIDATION_STEPS,
        "rem_steps": REM_ATTRIBUTION_STEPS,
        "mel_gain": MEL_GAIN,
        "factor_min": FACTOR_MIN,
        "factor_max": FACTOR_MAX,
        "mel_relative_floor": MEL_RELATIVE_FLOOR,
    }
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "sleep_driver_pattern": "manual-cycle-loop (force_cycle() once per cycle in a "
                                "MEAS_CYCLES wake-sleep loop; MEL consumer engages via "
                                "SleepLoopManager, novelty knob is SD-MEL-PRODUCER "
                                "world_rule_shift interval+depth)",
        "timestamp_utc": ts,
        "seeds": SEEDS,
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "interpretation": result["interpretation"],
        "criteria": result["criteria"],
        "readiness_ok": result["readiness_ok"],
        "readiness_frac": result["readiness_frac"],
        "r1_frac": result["r1_frac"], "r2_frac": result["r2_frac"], "r3_frac": result["r3_frac"], "r4_frac": result["r4_frac"],
        "c1_pass": result["c1_pass"],
        "c1_low_pass": result["c1_low_pass"], "c1_high_pass": result["c1_high_pass"],
        "clears_low_frac": result["clears_low_frac"], "clears_high_frac": result["clears_high_frac"],
        "falsifying_monotonic": result["falsifying_monotonic"],
        "monotone_frac": result["monotone_frac"],
        "per_seed": result["per_seed"],
        "arm_results": result["arm_results"],
        "thresholds": result["thresholds"],
        "ethics_preflight": {
            "involves_negative_valence": False,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
        },
    }
    out_path = write_flat_manifest(
        manifest,
        EVIDENCE_DIR,
        dry_run=dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=Path(__file__),
        z_goal_stream_stats=_ZG.stats(),
        started_at=started_at,
    )
    return str(out_path)


def main() -> None:
    t0 = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="1 seed, tiny convergence + measurement (smoke)")
    args = ap.parse_args()

    if args.dry_run:
        steps = 12
        conv_eps = 4
        meas_cycles = 3
        seeds = [42]
        # Smoke subset: exercise every distinct code path (2 ecological ON
        # arms spanning low/high MEL + calibration, plus the OFF control)
        # with 3 agent builds instead of 6. PROBE_EPISODES stays at its full
        # value (3) -- cheap at smoke's steps=12, and keeping it fixed avoids
        # a module-global mutation for a negligible time saving.
        smoke_ids = {"ARM_0_MONOTONY_ON", "ARM_4_OVERLOAD_ON", "ARM_5_OVERLOAD_OFF"}
        arms = [a for a in ARMS if a["arm_id"] in smoke_ids]
    else:
        steps = STEPS_PER_EPISODE
        conv_eps = CONV_EPISODES
        meas_cycles = MEAS_CYCLES
        seeds = SEEDS
        arms = ARMS

    result = run_experiment(steps, conv_eps, meas_cycles, seeds, arms)
    out_path = write_manifest(result, dry_run=bool(args.dry_run), started_at=t0)
    print(f"outcome: {result['outcome']}", flush=True)
    print(f"label: {result['interpretation']['label']}", flush=True)
    print(f"readiness_frac={result['readiness_frac']:.2f} "
          f"(r1={result['r1_frac']:.2f} r2={result['r2_frac']:.2f} r3={result['r3_frac']:.2f} r4={result['r4_frac']:.2f}) "
          f"clears_low={result['clears_low_frac']:.2f} clears_high={result['clears_high_frac']:.2f} "
          f"monotone={result['monotone_frac']:.2f}", flush=True)
    # [smoke] DV-variation-across-arms assertion (Step 3.5 mandatory for dose
    # sweeps): confirm rigidity_index actually varies across at least two
    # swept ON arms, not just that each arm ran (V3-EXQ-794/845/864 saturation
    # class). Print, and hard-assert under --dry-run.
    rigidity_by_arm = {}
    for r in result["arm_results"]:
        rigidity_by_arm.setdefault(r["arm_id"], []).append(round(r["rigidity_index"], 6))
    distinct_on_arm_values = {a: v for a, v in rigidity_by_arm.items() if a in ON_ARMS}
    all_values = [v for vs in distinct_on_arm_values.values() for v in vs]
    n_distinct = len(set(all_values))
    print(f"[smoke] rigidity_index by ON arm: {distinct_on_arm_values}", flush=True)
    print(f"[smoke] n_distinct_rigidity_values_across_on_arms={n_distinct}", flush=True)
    if args.dry_run:
        assert n_distinct >= 2, (
            "DV-variation-across-arms assertion FAILED: rigidity_index is "
            "identical (to 6dp) across every swept ON arm in the smoke -- this "
            "is the V3-EXQ-794/845/864 saturation signature (a clamp/floor "
            "absorbing the manipulation), not evidence of a flat effect. "
            "Investigate before queuing a full run."
        )
    print(f"manifest: {out_path}", flush=True)
    return result, out_path, args.dry_run


if __name__ == "__main__":
    _result, _out_path, _dry_run = main()
    _outcome_raw = str(_result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        dry_run=_dry_run,
    )
