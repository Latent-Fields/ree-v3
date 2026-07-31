"""
V3-EXQ-845 -- MECH-180 ecological end-to-end: does REAL graded environmental
novelty (SD-MEL-PRODUCER world-rule-shift, VALIDATED V3-EXQ-798a) drive graded
waking MEL which then drives graded offline-consolidation activity (SD-MEL-
CONSUMER, VALIDATED V3-EXQ-718a via injection), measured as SWS power, sleep
spindle density, and hippocampal replay rate?

SLEEP DRIVER: manual-cycle-loop (agent.sleep_loop.force_cycle() called once per
cycle in a dedicated MEAS_CYCLES wake-sleep loop). The MEL consumer engages
ONLY through the SleepLoopManager path (force_cycle), exactly as in
V3-EXQ-718a; a driver calling agent.run_sleep_cycle() directly would bypass it.

CLAIM UNDER TEST (MECH-180): "Novel environments and high-MEL learning
episodes adaptively upregulate the learning drive component of sleep (INV-050
third drive), producing measurable increases in slow-wave activity power,
sleep spindle density, and hippocampal replay rate proportional to
novelty/prediction-error load encountered during the preceding wake period."

================================================================================
WHY THIS RUN IS THE MISSING LINK (governance context, 2026-07-30/31)
================================================================================
MECH-180 decomposes into two links, and this is the FIRST run to test them
TOGETHER, ECOLOGICALLY:
  link (i)  real graded novelty -> graded above-reference waking MEL
  link (ii) MEL -> graded offline-phase consolidation
Three prior runs (V3-EXQ-677, 718, 718a) tried link (i) via the CausalGridWorldV2
env_drift knob and failed to demonstrate it: 677 measured a 8.8e-07 PE gradient
against a 0.01 threshold (no gradient at all); 718's C1 novelty-LABEL
monotonicity was 0/3 seeds; 718a's ecological measured MEL was noise-level
(~1e-5) and SCRAMBLED vs novelty level (conv_rel_drop ~0.98 -- the world model
fully converged, so drift added sampling noise, not learning load). All three
autopsies (failure_autopsy_V3-EXQ-677_2026-06-14, _718_2026-07-07,
_718a_2026-07-08) classify this as a measurement_gap / environment-producer
gap, NOT a substrate ceiling and NOT a falsification -- and count 3 against the
re-derive brake for MECH-180 (verified via the Step 2.5b brake-count script
this session, threshold >=2).

The producer gap is now CLOSED: SD-MEL-PRODUCER (ree-v3 CausalGridWorld;
docs/architecture/sd_mel_producer.md; VALIDATED via V3-EXQ-798a, confirmed
failure_autopsy_V3-EXQ-798a_2026-07-30) periodically re-permutes the
action->displacement map instead of merely moving hazards, so the world
becomes genuinely non-converging and re-learnable between shifts. 798a
demonstrated (diagnostic, consumer ABSENT by design, claim_ids=[]): anchor
decay 0.283 vs floor 0.15 (the elevated PE IS reducible -- genuine learning
load), matched noise-control decay 0.0042 vs ceiling 0.10 (an un-learnable
elevation does NOT decay), noise-control worst-seed fairness ratio 1.25x >=
1.0 (the control was at least as elevated, so the comparison is fair). So the
novelty->MEL link is now validated as a TEST-BED property, separately from any
claim.

Link (ii), MEL->cadence, is separately BUILT + PROVEN: V3-EXQ-718a's
MEL-injection positive control showed graded injected MEL [0.6..2.5]*base_ref
producing exact-monotone graded DV (cumulative_sws_writes + rem_rollouts:
[9,13,18,24,30,38], all seeds) via the real SleepLoopManager/force_cycle path.
But 718a's ECOLOGICAL arms used the broken env_drift knob for novelty, so its
own C1 (measured-MEL-gradient -> DV monotone) never got a fair ecological test
either -- 718a's ecological MEL was itself the noise-level, scrambled signal
718's autopsy also found.

So: the novelty->MEL link is validated (798a, as a test-bed property, no
sleep), the MEL->consolidation link is validated (718a, via injection,
decoupled from real novelty), but NO run has ever driven the validated
producer INTO the validated consumer end-to-end. That is what this experiment
does, using SD-MEL-PRODUCER (not env_drift) as this run's novelty knob for the
first time in a sleep-bearing driver.

GOV-REUSE-1 CHECK (Step 2.4, this session): the decisive readout -- SWS/replay
DVs monotone in ecologically-measured MEL when driven by the world-rule-shift
knob through the live SleepLoopManager -- is not recorded in any existing
manifest. V3-EXQ-798a never runs sleep (consumer absent by design, see its own
docstring: "the validation experiment for this SD runs WITH THE CONSUMER
ABSENT"). V3-EXQ-718/718a's sleep-bearing ecological arms use env_drift, not
world_rule_shift. No compatible substrate_hash carries this combination. Not
recoverable by reanalysis -> proceeds to a new run.

RE-DERIVE BRAKE: RELEASED. MECH-180 has 3 counted substrate_ceiling-class
autopsies (677, 718, 718a; >= threshold 2), but the named upstream substrate
SD-MEL-PRODUCER is now IMPLEMENTED/VALIDATED (ree-v3/CLAUDE.md line ~3637;
docs/architecture/sd_mel_producer.md Status: VALIDATED). Per the brake-release
rule this is the correct next test of a now-built substrate, not a same-ceiling
re-derive.

================================================================================
V3 PROXY MAPPING -- the claim's three DVs, honestly operationalized
================================================================================
V3 has no literal oscillatory EEG signal. Per the "biology before formal
definitions" convention (instantiate the mechanism, not the metaphor), the
three claimed readouts map onto substrate quantities that SD-017's own SWS/REM
passes already compute (ree_core/agent.py run_sws_schema_pass /
run_rem_attribution_pass), scaled every cycle by the REAL MELConsumer duration
factor (ree_core/sleep/phase_manager.py:357-365) -- none of these are
manufactured for this experiment:

  "slow-wave activity power"   -> sws_n_writes: the number of prototype
    z_world schema-consolidation writes to ContextMemory during the SWS pass
    this cycle. This is the SAME DV the SD-MEL-CONSUMER / V3-EXQ-677/718/718a
    lineage already uses as the established V3 stand-in for "how much
    consolidation work SWS is doing" -- an amount/dose reading, the natural
    analog of oscillatory power (more/deeper SWS => more schema writes).
    Bounded by min(sws_consolidation_steps_scaled, n_buf); n_buf is capped at
    1000 by agent.py:4862 and this run's P0 (60 episodes x 90 steps) fills it
    to cap well before any measurement cycle, so the bound is NEVER the
    limiting factor here (max scaled steps = 5*FACTOR_MAX=15 << 1000) --
    sws_n_writes tracks the MEL-driven duration factor cleanly, not a buffer
    artifact.
  "hippocampal replay rate"    -> rem_n_rollouts: the number of hippocampal
    forward + reverse replay rollouts (agent.hippocampal.replay /
    diverse_replay, called from run_rem_attribution_pass) executed during the
    REM pass this cycle. This is literally hippocampal replay, by name and by
    call site, and it is the OTHER established DV in the 677/718/718a
    lineage -- not a new proxy.
  "sleep spindle density"      -> sws_slot_diversity: mean pairwise cosine
    DISTANCE across ContextMemory's schema-attractor slots after the SWS pass
    (run_sws_schema_pass, agent.py ~10132-10150). This is a genuinely DISTINCT
    statistic from sws_n_writes -- a geometric/differentiation property of
    WHAT got consolidated, not a count of HOW MUCH. It is not redundant with
    sws_n_writes: the write count is bounded by the (buffer-unconstrained)
    scaled step budget, while slot diversity is an emergent property of which
    z_world prototypes were sampled and how they geometrically relate in
    ContextMemory -- it can in principle saturate, plateau, or fail to track
    write count. We do NOT reuse MECH-122's Phase-3 ThetaBuffer
    "spindle-coordination" proxy here: that is a SEPARATE claim's mechanism
    (bidirectional consolidation-transfer, un-modulated by MEL in the current
    substrate) and conflating the two would misattribute MECH-122's evidence
    to MECH-180. sws_slot_diversity is chosen because it is (a) already
    computed by the SWS pass this run drives regardless, at zero extra cost,
    and (b) the substrate's own literal "how differentiated is what SWS just
    installed" readout -- the closest honest analog of spindle-mediated
    consolidation SPECIFICITY without inventing a new mechanism.

DV-SYMMETRY (604c check): the manipulation is world_rule_shift re-permuting
the action->displacement map (validated non-invariant for the waking PE DV in
798a). Here the causal chain is: shift rate -> measured waking MEL (via
agent.update_residue -> e3 prediction_error -> mel_consumer accumulator) ->
mel_consumer.duration_factor() -> agent.config.sws_consolidation_steps /
rem_attribution_steps (REAL config values consumed by run_sws_schema_pass /
run_rem_attribution_pass to decide how much work to do) -> sws_n_writes /
rem_n_rollouts (bounded by that scaled config, not a buffer floor here) and
sws_slot_diversity (an emergent geometric readout of WHICH z_worlds got
sampled and written, not a direct function of the shift permutation or of the
duration factor). None of these three DVs is a broadcast-invariant, a
monotone-rescaling-invariant rank statistic, or a permutation-invariant
set-aggregate of the manipulation -- each genuinely depends on the measured
MEL through the real consumer path, not on an arithmetic identity of the
shift itself.

PRE-REGISTERED ACCEPTANCE (evidence; claim_ids=["MECH-180"])
--------------------------------------------------------------------------
READINESS (per seed, over the four ecological ON arms):
  R1 world-model trained: frozen-probe conv_rel_drop >= MIN_REL_CONV_DROP.
  R2 ecological novelty->MEL link holds IN THIS CONFIG: measured mean_mel is
     non-degenerately graded (mel[HIGH] >= mel[NONE] * (1+MIN_MEL_SPREAD)).
     R2 re-confirms 798a's link under THIS run's agent config (z_goal/benefit
     machinery + live sleep loop, absent from 798a's diagnostic setup) -- a
     cheap admissibility check, not a re-validation of SD-MEL-PRODUCER itself.
     Below-floor on either R1 or R2 on >= SEED_PASS_FRAC of seeds routes to
     substrate_not_ready_requeue, NEVER a MECH-180 verdict.

C1 (LOAD-BEARING, conjunctive over three sub-DVs; each on the 4 ON arms sorted
    ascending by MEASURED mean_mel, per seed):
  C1a SWS power:      cumulative_sws_writes monotone non-decreasing (+ tol)
                       and relative spread >= MIN_REL_DV_SPREAD.
  C1b spindle density: mean sws_slot_diversity monotone non-decreasing (+ tol)
                       and relative spread >= MIN_REL_DV_SPREAD.
  C1c replay rate:     cumulative_rem_rollouts monotone non-decreasing (+ tol)
                       and relative spread >= MIN_REL_DV_SPREAD.
  C1 = C1a AND C1b AND C1c, each on >= SEED_PASS_FRAC of seeds -- conjunctive
  because MECH-180 predicts all three increase together; partial support is
  recorded distinctly (see interpretation grid), not folded into a blanket
  pass/fail.

C2 (control, non-load-bearing sanity check -- reproduces the 677/718a check):
  pinned: the matched-novelty consumer-OFF arm (ARM_4_HIGH_OFF, same
    world_rule_shift as ARM_3_HIGH) shows near-zero per-cycle variance in
    sws_n_writes and rem_n_rollouts (these are directly step-count-bounded by
    a factor pinned at 1.0 when the consumer is off, so pinning is expected by
    construction -- unlike slot_diversity, an emergent geometric quantity not
    directly parameter-bounded, which is reported as a diagnostic only and is
    NOT gated for pinning).
  on_gt_off: ARM_3_HIGH (consumer ON) exceeds ARM_4_HIGH_OFF on all three DVs
    -- confirms the CONSUMER, not raw environmental stochasticity, drives the
    elevation.

INTERPRETATION GRID:
  readiness unmet                  -> substrate_not_ready_requeue   (non_contributory)
  C2 fail (control not pinned/gt)  -> mel_control_degenerate         (non_contributory)
  C2 pass, C1 all three pass       -> mel_producer_consumer_ecological_consolidation_dose_response (PASS, supports)
  C2 pass, C1 zero of three pass   -> consumer_capable_ecological_novelty_not_graded (non_contributory)
  C2 pass, C1 one-or-two pass      -> mel_producer_consumer_partial_dv_support (mixed)

PROMOTES/DEMOTES MECH-180 directly on the full-PASS branch (evidence_direction
"supports"); all other branches leave MECH-180 v3_pending / unresolved with an
honest partial-or-null reading -- this experiment does not itself close the
claim on anything short of C1a+C1b+C1c all clearing.
"""

import sys
import math
import time
import random
import argparse
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

EXPERIMENT_TYPE = "v3_exq_845_mech180_ecological_novelty_sleep_consolidation_dose_response"
QUEUE_ID = "V3-EXQ-845"
CLAIM_IDS = ["MECH-180"]
EXPERIMENT_PURPOSE = "evidence"

# z_goal_enabled=True is inherited verbatim from the V3-EXQ-718a/798a lineage for
# architecture parity, but no code path here calls update_z_goal -- the stream is
# inert, identically, in every arm (goal_weight/e1_goal_conditioned are structural
# config, not exercised by this driver). Wiring it live would activate the E3 goal
# term + E1 goal-conditioning and the SD-024 benefit-attractor producer, changing
# action selection and hence the trajectories the MEL/DV readouts are computed
# over -- a behaviour change that would make this run non-comparable to the
# lineage it validates against. The knob is arm-symmetric, so the inertness cannot
# bias the ON-vs-OFF or cross-arm comparisons this experiment routes on.
DEAD_Z_GOAL_STREAM_EXEMPT = (
    "inherited verbatim from V3-EXQ-718a/798a for architecture parity; wiring "
    "update_z_goal would activate the E3 goal term, E1 conditioning, and the "
    "SD-024 benefit-attractor producer, confounding the MEL/DV comparison this "
    "experiment routes on. Knob is arm-symmetric (identical in every arm)."
)

# -- Design parameters -------------------------------------------------------
SEEDS = [42, 123, 456]
CONV_EPISODES = 60            # P0 world-model convergence on the STABLE base env
STEPS_PER_EPISODE = 90
PROBE_BATTERY_SIZE = 64       # FIXED held-out probe battery (frozen-encoder)
CALIB_EPISODES = 3            # stable-base MEL reference-calibration wake pass
MEAS_CYCLES = 6               # wake-sleep cycles per arm
WAKE_EPISODES_PER_CYCLE = 2   # wake episodes per cycle (populate buffers + MEL)
# Progress denominator M (per cell): P0 + calibration + measurement wake episodes.
EPISODES_PER_RUN = (CONV_EPISODES + CALIB_EPISODES
                    + MEAS_CYCLES * WAKE_EPISODES_PER_CYCLE)

# Sleep pass base durations (the scheduler-pinned counts V3-EXQ-677 measured;
# same base as V3-EXQ-718/718a for lineage comparability).
SWS_CONSOLIDATION_STEPS = 5
REM_ATTRIBUTION_STEPS = 10

# MEL consumer config (identical to the validated V3-EXQ-718a test-bed).
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
SEED_PASS_FRAC = 2.0 / 3.0    # R / C1 / C2: at least 2/3 of seeds
MIN_MEL_SPREAD = 0.15         # R2: measured mean_mel[max] at least 15% above [min]
MIN_REL_DV_SPREAD = 0.15      # C1: DV[max] at least 15% above DV[min]
MONO_TOL = 0.05               # C1: monotonicity slack (relative to DV[min])
PINNED_ABS_VAR_ATOL = 1e-6    # C2: OFF-arm per-cycle count variance ceiling

# -- Environment base (identical to V3-EXQ-798a / 718a / 701c lineage) -------
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
# graded arms is the SD-MEL-PRODUCER rule shift itself (per its own docstring).
STABLE_DRIFT = dict(env_drift_interval=999, env_drift_prob=0.0)
WORLD_RULE_SHIFT_DEPTH = 2    # matches V3-EXQ-798a's validated ladder

# arm_id, novelty level (world_rule_shift_interval; 798a's validated ladder),
# consumer on/off.
ARMS: List[Dict[str, Any]] = [
    {"arm_id": "ARM_0_NONE_ON",  "level": 0, "interval": 0,  "mel_on": True},
    {"arm_id": "ARM_1_LOW_ON",   "level": 1, "interval": 60, "mel_on": True},
    {"arm_id": "ARM_2_MED_ON",   "level": 2, "interval": 25, "mel_on": True},
    {"arm_id": "ARM_3_HIGH_ON",  "level": 3, "interval": 10, "mel_on": True},
    {"arm_id": "ARM_4_HIGH_OFF", "level": 3, "interval": 10, "mel_on": False},
]
# The 4 ecological ON arms (C1 is scored over these, sorted by MEASURED mean_mel).
ON_ECO_ARMS = ["ARM_0_NONE_ON", "ARM_1_LOW_ON", "ARM_2_MED_ON", "ARM_3_HIGH_ON"]


def _make_env(seed: int, interval: int) -> CausalGridWorldV2:
    kw = dict(ENV_BASE)
    kw.update(STABLE_DRIFT)
    kw.update(
        world_rule_shift_enabled=(interval > 0),
        world_rule_shift_interval=interval,
        world_rule_shift_depth=WORLD_RULE_SHIFT_DEPTH if interval > 0 else 0,
    )
    return CausalGridWorldV2(seed=seed, **kw)


def _make_agent(env: CausalGridWorldV2, mel_on: bool, mel_reference: float) -> REEAgent:
    """Converged-base agent (recon-only e2 training; encoder frozen) + SD-017
    SWS/REM passes + the SleepLoopManager. When mel_on, the SD-MEL-CONSUMER is
    enabled with a FIXED reference set-point. Config is byte-identical to the
    validated V3-EXQ-718a recipe (same architecture; only the env's novelty
    knob differs -- world_rule_shift here vs env_drift there)."""
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
    env = _make_env(seed, interval=0)
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


# z_goal-stream liveness, pooled across the run's per-cell agents.
_ZG = ZGoalStreamAccumulator()


def _run_cell(seed: int, arm: Dict[str, Any], steps: int, conv_eps: int,
              meas_cycles: int) -> Dict[str, Any]:
    """One (seed, arm) cell: build agent, converge P0 recon-only on the stable
    base, calibrate the MEL reference to the stable-base measurement-modality
    MEL, then run the wake-sleep measurement cycles on the arm's
    SD-MEL-PRODUCER env (world_rule_shift, not env_drift)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    arm_id = arm["arm_id"]
    mel_on = bool(arm["mel_on"])
    print(f"Seed {seed} Condition {arm_id}", flush=True)

    # ONE agent per cell. Encoder FROZEN (only agent.e2 trains). Build with
    # mel_reference=0.0, then fix it to the calibrated stable-base MEL AFTER
    # P0 (mode="fixed", so the >0 reference is used and never auto-relocked).
    stable_env = _make_env(seed, interval=0)
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
    # accumulator). NONE arm (stable env, no shift) then yields factor ~1.0;
    # higher-shift-rate arms scale above it.
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
    meas_env = _make_env(seed, arm["interval"])
    cum_sws = 0.0
    cum_rem = 0.0
    per_cycle_sws: List[float] = []
    per_cycle_rem: List[float] = []
    per_cycle_diversity: List[float] = []
    factors: List[float] = []
    mels: List[float] = []
    ep_off = conv_eps + CALIB_EPISODES
    for _cyc in range(meas_cycles):
        _run_wake_window(
            agent, meas_env, WAKE_EPISODES_PER_CYCLE, steps, train=False,
            buffer=None, e2_opt=None, sample_rng=None, ep_offset=ep_off,
            arm_id=arm_id, seed=seed,
        )
        ep_off += WAKE_EPISODES_PER_CYCLE
        m = agent.sleep_loop.force_cycle(agent)
        sws = float(m.get("sws_n_writes", 0.0))
        rem = float(m.get("rem_n_rollouts", 0.0))
        diversity = float(m.get("sws_slot_diversity", 0.0))
        cum_sws += sws
        cum_rem += rem
        per_cycle_sws.append(sws)
        per_cycle_rem.append(rem)
        per_cycle_diversity.append(diversity)
        if mel_on:
            factors.append(float(m.get("mel_duration_factor", 1.0)))
            mels.append(float(m.get("mel_mean", 0.0)))

    mean_diversity = float(np.mean(per_cycle_diversity)) if per_cycle_diversity else 0.0
    sws_count_var = float(np.var(per_cycle_sws))
    rem_count_var = float(np.var(per_cycle_rem))
    diversity_var = float(np.var(per_cycle_diversity))
    mean_factor = float(np.mean(factors)) if factors else 1.0
    mean_mel = float(np.mean(mels)) if mels else 0.0

    _ZG.observe(agent)

    print(f"    {arm_id} seed={seed}: conv_drop={conv_rel_drop:.3f} "
          f"ref={base_ref:.3e} mel={mean_mel:.3e} factor={mean_factor:.3f} "
          f"cum_sws={cum_sws:.0f} mean_div={mean_diversity:.4f} cum_rem={cum_rem:.0f}",
          flush=True)
    print(f"verdict: {'PASS' if conv_rel_drop >= MIN_REL_CONV_DROP else 'FAIL'}",
          flush=True)

    return {
        "arm_id": arm_id,
        "level": arm["level"],
        "mel_on": mel_on,
        "world_rule_shift_interval": arm["interval"],
        "seed": seed,
        "conv_rel_drop": conv_rel_drop,
        "probe_pe_init": probe_pe_init,
        "probe_pe_final": probe_pe_final,
        "mel_reference": base_ref,
        "mean_mel": mean_mel,
        "mean_duration_factor": mean_factor,
        "cumulative_sws_writes": cum_sws,
        "cumulative_rem_rollouts": cum_rem,
        "mean_sws_slot_diversity": mean_diversity,
        "per_cycle_sws": per_cycle_sws,
        "per_cycle_rem": per_cycle_rem,
        "per_cycle_diversity": per_cycle_diversity,
        "per_cycle_mel": mels,
        "per_cycle_factor": factors,
        "sws_count_variance": sws_count_var,
        "rem_count_variance": rem_count_var,
        "diversity_variance": diversity_var,
        "meas_cycles": meas_cycles,
    }


def _dv_dose_response(values: List[float]) -> Dict[str, Any]:
    """Given a DV in ascending-measured-MEL order, test monotone non-decreasing
    (+ tol) and relative spread. Shared by C1a/C1b/C1c."""
    if len(values) < 2:
        return {"monotone_ok": False, "spread_ok": False, "pass_": False}
    lo = values[0]
    tol = MONO_TOL * max(lo, 1e-9)
    monotone_ok = all(values[i] <= values[i + 1] + tol for i in range(len(values) - 1))
    spread_ok = lo > 0 and values[-1] >= lo * (1 + MIN_REL_DV_SPREAD)
    return {"monotone_ok": bool(monotone_ok), "spread_ok": bool(spread_ok),
            "pass_": bool(monotone_ok and spread_ok)}


def _seed_readiness(on_eco_cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """R1 (world-model trained) AND R2 (ecological novelty->MEL link holds in
    THIS run's config) for one seed's 4 ecological ON arms."""
    if len(on_eco_cells) != len(ON_ECO_ARMS):
        return {"r1_ok": False, "r2_ok": False, "ready": False}
    r1_ok = all(r["conv_rel_drop"] >= MIN_REL_CONV_DROP for r in on_eco_cells)
    arms_sorted = sorted(on_eco_cells, key=lambda r: r["mean_mel"])
    mels = [r["mean_mel"] for r in arms_sorted]
    r2_ok = mels[0] > 0 and mels[-1] >= mels[0] * (1 + MIN_MEL_SPREAD)
    return {"r1_ok": bool(r1_ok), "r2_ok": bool(r2_ok), "ready": bool(r1_ok and r2_ok)}


def _seed_c1(on_eco_by_arm: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """C1a/b/c per seed on the 4 ecological ON arms, sorted ascending by
    MEASURED mean_mel."""
    arms_sorted = sorted(on_eco_by_arm.values(), key=lambda r: r["mean_mel"])
    arm_order = [r["arm_id"] for r in arms_sorted]
    mels = [r["mean_mel"] for r in arms_sorted]
    sws = [r["cumulative_sws_writes"] for r in arms_sorted]
    div = [r["mean_sws_slot_diversity"] for r in arms_sorted]
    rem = [r["cumulative_rem_rollouts"] for r in arms_sorted]
    c1a = _dv_dose_response(sws)
    c1b = _dv_dose_response(div)
    c1c = _dv_dose_response(rem)
    n_pass = sum(1 for c in (c1a, c1b, c1c) if c["pass_"])
    return {
        "arm_order_by_measured_mel": arm_order,
        "mel_by_measured_order": mels,
        "c1a_sws_power": {**c1a, "values": sws},
        "c1b_spindle_density": {**c1b, "values": div},
        "c1c_replay_rate": {**c1c, "values": rem},
        "c1_n_sub_pass": n_pass,
        "c1_all_pass": bool(n_pass == 3),
        "c1_zero_pass": bool(n_pass == 0),
    }


def _seed_c2(on_eco_by_arm: Dict[str, Dict[str, Any]],
             off_cell: Dict[str, Any]) -> Dict[str, Any]:
    """C2 per seed: OFF control's step-bounded counts pinned (near-zero
    variance; slot_diversity is reported but NOT gated -- see docstring) AND
    ON-HIGH exceeds OFF-HIGH on all three DVs."""
    pinned_ok = (off_cell["sws_count_variance"] <= PINNED_ABS_VAR_ATOL
                 and off_cell["rem_count_variance"] <= PINNED_ABS_VAR_ATOL)
    on_high = on_eco_by_arm["ARM_3_HIGH_ON"]
    on_gt_off = (
        on_high["cumulative_sws_writes"] > off_cell["cumulative_sws_writes"]
        and on_high["mean_sws_slot_diversity"] > off_cell["mean_sws_slot_diversity"]
        and on_high["cumulative_rem_rollouts"] > off_cell["cumulative_rem_rollouts"]
    )
    return {
        "pinned_ok": bool(pinned_ok),
        "on_gt_off": bool(on_gt_off),
        "c2_pass": bool(pinned_ok and on_gt_off),
        "off_diversity_variance_reported_only": off_cell["diversity_variance"],
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
                "world_rule_shift_depth": WORLD_RULE_SHIFT_DEPTH,
                "conv_episodes": conv_eps,
                "meas_cycles": meas_cycles,
                "steps_per_episode": steps,
                "sws_steps": SWS_CONSOLIDATION_STEPS,
                "rem_steps": REM_ATTRIBUTION_STEPS,
                "mel_gain": MEL_GAIN,
                "factor_min": FACTOR_MIN,
                "factor_max": FACTOR_MAX,
                "mel_relative_floor": MEL_RELATIVE_FLOOR,
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
        on_eco_cells = [r for r in arm_results
                        if r["seed"] == seed and r["mel_on"]]
        rd = _seed_readiness(on_eco_cells)
        seed_readiness_detail[seed] = rd
        seed_ready[seed] = rd["ready"]
    readiness_frac = sum(seed_ready.values()) / max(1, len(seeds))
    r1_frac = sum(1 for s in seeds if seed_readiness_detail[s]["r1_ok"]) / max(1, len(seeds))
    r2_frac = sum(1 for s in seeds if seed_readiness_detail[s]["r2_ok"]) / max(1, len(seeds))
    readiness_ok = readiness_frac >= SEED_PASS_FRAC

    # -- C1 / C2 per seed, on ready seeds only --
    c1_seed_pass = 0
    c2_seed_pass = 0
    per_dv_seed_pass = {"sws_power": 0, "spindle_density": 0, "replay_rate": 0}
    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        on_eco = {r["arm_id"]: r for r in arm_results
                  if r["seed"] == seed and r["mel_on"]}
        off_cell = next((r for r in arm_results
                         if r["seed"] == seed and r["arm_id"] == "ARM_4_HIGH_OFF"), None)
        rec: Dict[str, Any] = {
            "seed": seed, "ready": seed_ready[seed],
            "readiness_detail": seed_readiness_detail[seed],
        }
        if seed_ready[seed] and len(on_eco) == len(ON_ECO_ARMS) and off_cell:
            c1 = _seed_c1(on_eco)
            c2 = _seed_c2(on_eco, off_cell)
            rec.update({"c1": c1, "c2": c2})
            if c1["c1a_sws_power"]["pass_"]:
                per_dv_seed_pass["sws_power"] += 1
            if c1["c1b_spindle_density"]["pass_"]:
                per_dv_seed_pass["spindle_density"] += 1
            if c1["c1c_replay_rate"]["pass_"]:
                per_dv_seed_pass["replay_rate"] += 1
            if c1["c1_all_pass"]:
                c1_seed_pass += 1
            if c2["c2_pass"]:
                c2_seed_pass += 1
        else:
            rec.update({"c1": None, "c2": None, "skipped": "not_ready_or_missing_arms"})
        per_seed.append(rec)

    n_seeds = max(1, len(seeds))
    c1_frac = c1_seed_pass / n_seeds
    c2_frac = c2_seed_pass / n_seeds
    c1_all_pass = c1_frac >= SEED_PASS_FRAC
    c2_pass = c2_frac >= SEED_PASS_FRAC
    per_dv_frac = {k: v / n_seeds for k, v in per_dv_seed_pass.items()}
    per_dv_pass = {k: (v >= SEED_PASS_FRAC) for k, v in per_dv_frac.items()}
    n_dv_pass = sum(1 for v in per_dv_pass.values() if v)

    # -- Self-route --
    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        direction = "non_contributory"
    elif not c2_pass:
        label = "mel_control_degenerate"
        outcome = "FAIL"
        direction = "non_contributory"
    elif c1_all_pass:
        label = "mel_producer_consumer_ecological_consolidation_dose_response"
        outcome = "PASS"
        direction = "supports"
    elif n_dv_pass == 0:
        label = "consumer_capable_ecological_novelty_not_graded"
        outcome = "FAIL"
        direction = "non_contributory"
    else:
        label = "mel_producer_consumer_partial_dv_support"
        outcome = "FAIL"
        direction = "mixed"

    ready_seeds = [s for s in seeds if seed_ready[s]]
    c1_gradient_present = readiness_ok and bool(ready_seeds)
    ready_on_dvs_sws = [r["cumulative_sws_writes"] for r in arm_results
                        if r["mel_on"] and seed_ready.get(r["seed"], False)]
    c1_dv_spread_nonzero = (
        len(set(round(v, 3) for v in ready_on_dvs_sws)) > 1
    ) if ready_on_dvs_sws else False
    off_cells = [r for r in arm_results if not r["mel_on"]]
    c2_off_present = len(off_cells) > 0

    interpretation = {
        "label": label,
        "preconditions": [
            {
                "name": "world_forward_converged_frozen_probe",
                "description": "recon-only P0 converges on the fixed frozen-probe "
                               "battery (conv_rel_drop >= MIN_REL_CONV_DROP) on the "
                               "ecological ON arms, on >= 2/3 seeds -- the world "
                               "model must be trained for MEL to live at a "
                               "meaningful scale (R1).",
                "measured": r1_frac,
                "threshold": SEED_PASS_FRAC,
                "direction": "lower",
                "met": bool(r1_frac >= SEED_PASS_FRAC),
            },
            {
                "name": "ecological_novelty_mel_gradient_present_this_config",
                "description": "measured mean_mel is non-degenerately graded "
                               "(mel[HIGH] >= mel[NONE]*(1+MIN_MEL_SPREAD)) across "
                               "the 4 ON arms, on >= 2/3 seeds -- re-confirms the "
                               "V3-EXQ-798a novelty->MEL link under THIS run's "
                               "full agent config (z_goal/benefit machinery + live "
                               "sleep loop, absent from 798a's diagnostic setup). "
                               "Below-floor => the ecological link does not survive "
                               "in this config (not-ready), NOT a MECH-180 "
                               "falsification (R2).",
                "measured": r2_frac,
                "threshold": SEED_PASS_FRAC,
                "direction": "lower",
                "met": bool(r2_frac >= SEED_PASS_FRAC),
            },
        ],
        "criteria_non_degenerate": {
            "C1_measured_mel_gradient_present": bool(c1_gradient_present),
            "C1_dv_spread_nonzero": bool(c1_dv_spread_nonzero),
            "C2_off_control_present": bool(c2_off_present),
        },
        "per_dv_seed_pass_fraction": per_dv_frac,
        "per_dv_pass": per_dv_pass,
    }
    criteria = [
        {"name": "C1a_sws_power_monotone_in_measured_mel", "load_bearing": True,
         "passed": bool(per_dv_pass["sws_power"])},
        {"name": "C1b_spindle_density_monotone_in_measured_mel", "load_bearing": True,
         "passed": bool(per_dv_pass["spindle_density"])},
        {"name": "C1c_replay_rate_monotone_in_measured_mel", "load_bearing": True,
         "passed": bool(per_dv_pass["replay_rate"])},
        {"name": "C2_consumer_not_env_causes_variation", "load_bearing": False,
         "passed": bool(c2_pass)},
    ]

    return {
        "outcome": outcome,
        "evidence_direction": direction,
        "interpretation": interpretation,
        "criteria": criteria,
        "readiness_ok": readiness_ok,
        "readiness_frac": readiness_frac,
        "r1_frac": r1_frac,
        "r2_frac": r2_frac,
        "c1_all_pass": c1_all_pass, "c1_frac": c1_frac,
        "c2_pass": c2_pass, "c2_frac": c2_frac,
        "per_dv_pass": per_dv_pass,
        "per_dv_frac": per_dv_frac,
        "per_seed": per_seed,
        "arm_results": arm_results,
        "thresholds": {
            "MIN_REL_CONV_DROP": MIN_REL_CONV_DROP,
            "SEED_PASS_FRAC": SEED_PASS_FRAC,
            "MIN_MEL_SPREAD": MIN_MEL_SPREAD,
            "MIN_REL_DV_SPREAD": MIN_REL_DV_SPREAD,
            "MONO_TOL": MONO_TOL,
            "PINNED_ABS_VAR_ATOL": PINNED_ABS_VAR_ATOL,
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
    out_dir = (Path(__file__).resolve().parents[2]
               / "REE_assembly" / "evidence" / "experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    full_config = {
        "env_base": ENV_BASE,
        "arms": ARMS,
        "world_rule_shift_depth": WORLD_RULE_SHIFT_DEPTH,
        "conv_episodes": CONV_EPISODES,
        "calib_episodes": CALIB_EPISODES,
        "meas_cycles": MEAS_CYCLES,
        "wake_episodes_per_cycle": WAKE_EPISODES_PER_CYCLE,
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
                                "world_rule_shift, not env_drift)",
        "timestamp_utc": ts,
        "seeds": SEEDS,
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "interpretation": result["interpretation"],
        "criteria": result["criteria"],
        "readiness_ok": result["readiness_ok"],
        "readiness_frac": result["readiness_frac"],
        "r1_frac": result["r1_frac"],
        "r2_frac": result["r2_frac"],
        "c1_all_pass": result["c1_all_pass"], "c1_frac": result["c1_frac"],
        "c2_pass": result["c2_pass"], "c2_frac": result["c2_frac"],
        "per_dv_pass": result["per_dv_pass"],
        "per_dv_frac": result["per_dv_frac"],
        "per_seed": result["per_seed"],
        "arm_results": result["arm_results"],
        "thresholds": result["thresholds"],
    }
    out_path = write_flat_manifest(
        manifest,
        out_dir,
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
        # Smoke subset: exercise every distinct code path (ecological ON +
        # calibration, plus OFF pinning) with 3 agent builds instead of 5.
        smoke_ids = {"ARM_0_NONE_ON", "ARM_3_HIGH_ON", "ARM_4_HIGH_OFF"}
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
          f"(r1={result['r1_frac']:.2f} r2={result['r2_frac']:.2f}) "
          f"c1_frac={result['c1_frac']:.2f} c2_frac={result['c2_frac']:.2f} "
          f"per_dv={result['per_dv_frac']}", flush=True)
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
