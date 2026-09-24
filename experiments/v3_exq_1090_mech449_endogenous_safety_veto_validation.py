"""
V3-EXQ-1090 -- MECH-449 ENDOGENOUS safety-veto producer: substrate-readiness validation.

SLEEP DRIVER: N/A (no sleep loop; scaffolded_sd054_onboarding is a waking
goal-pipeline onboarding scheduler).

RED-TEAM (fable, 2026-09-24): CONTESTED. F1 fixed (absolute 0.30 envelope floor admitted
all 32 candidates -> F-blind eval; now the channel-adaptive mean-relative floor). F2 partly
fixed (predicted-state headroom diagnostics recorded; does_not_fire label now defers to
them; the rest dismissed: precondition range is over real states, the veto's excursion over
predicted states, which drift more -- not equal by construction). F3 dismissed
(pre-registered, orchestrator-confirmed; frac_positive recorded). F4 fixed (unscored
control / never-armed gate -> substrate_not_ready_requeue).

experiment_purpose: diagnostic (substrate validation -- excluded from confidence scoring)
claim_ids: [MECH-449] (tagged for discoverability of the build; non_contributory throughout)
chip: chip-20260918-mech449-endogenous-safety-veto-producer
unblocks (release test only): REE_assembly EVB-1409 / EXP-0796 (MECH-049) release_condition (b)

WHAT IS BEING VALIDATED
  The MECH-449 / ARC-107 Go/No-Go eligibility gate (e3_selector._go_nogo_eligibility_gate)
  has a fail-open-IMMUNE safety No-Go axis that removes a candidate from the F-built
  eligible set. Until 2026-09-24 nothing in ree_core produced that axis: only
  hand-constructed synthetic banks (`safety[u] = 0.9`, V3-EXQ-689g/926/926a/936).
  The new default-OFF producer E3Config.use_gng_endogenous_safety (ree_core/agent.py
  REEAgent._endogenous_gng_safety) derives it from the harm VALUATION pathway:
    h_k    = mean over candidate k's predicted z_world states of E3.harm_eval_head
    z_k    = (h_k - mu) / max(sd, gng_safety_sd_floor)   [per-agent running EMA scale,
                                                           read BEFORE this tick's update]
    veto   : z_k >= gng_safety_z_threshold (+2 SD), mapped onto the gate's 0.5 floor.
  CALIBRATION PROVENANCE: orchestrator decision "DECIDED Q-MECH449 -> A"
  (orchestrate-20260924-0808, 2026-09-24T11:03:55Z, .scratch/orch-20260924/QUESTIONS.md):
  the pre-flight showed raw sigmoid harm (spread 0.045 mean / 0.133 max, V3-EXQ-603k)
  either never or always crosses the absolute 0.5 floor, and per-bank min-max
  normalisation fires every tick BY CONSTRUCTION (release test (b) could not fail); so
  calibrate against a per-seed running harm scale, veto at > +2 SD, and pair release
  test (b) with three specificity checks + a per-seed harm-range precondition.

DESIGN
  Training regime = V3-EXQ-603q ARM_BASE_IA_ONLY exactly (as re-run fresh by
  V3-EXQ-866b on 2026-08-03: harm_eval_range 0.470 / 0.295 / 0.115, prox_corr ~0.93,
  3/3 seeds discriminative -- the strongest harm-pathway regime on record; 603k's
  weaker regime was discriminative on only 1/3 seeds). Stage0 -> Stage0b -> P0 ->
  Stage-H -> P1 (P2 omitted: the eval phase below replaces it). Two arms, 5 seeds each:
    ARM_HARM_ON           scaffold_train_harm_pathway=True  (603q base; the release arm)
    ARM_HARM_OFF_CONTROL  scaffold_train_harm_pathway=False (harm head never optimised ->
                          flat landscape; the specificity control)
  After P1 each cell measures harm discriminativeness (the scheduler's own read-only
  _measure_harm_discriminativeness probe) BEFORE arming, then ARMS the gate's full chain
  on the trained agent -- use_go_nogo_constitution + use_f_eligibility_demotion with the
  channel-adaptive mean-relative floor (the MECH-448 envelope builds the eligible set:
  above-average F-merit candidates) + the endogenous producer; the live
  modulatory accumulator comes from the 603q base's own score_bias channels
  (instrumental-avoidance / liking bridge), verified live by the authoring probe --
  and runs N_EVAL_EPISODES frozen-policy episodes in the Stage-H hazard env. Per fresh
  E3 selection tick (latch cleared before every select_action) it records the producer's
  per-candidate harm / z / fired mask, the gate's in-eligible-set safety No-Go count,
  and a GROUND-TRUTH counterfactual for each distinct candidate first action: a deepcopy
  of the env is stepped once and the true hazard proximity at the destination
  (harm_obs centre = the SD-010/SD-018 supervision label) and the realised one-step harm
  are read. The ground truth is independent of the harm head the veto reads.

PRE-REGISTERED CRITERIA (constants below; per seed, then aggregated)
  PRECONDITION (per ARM_HARM_ON seed): post-training harm_eval_range >=
    HARM_RANGE_PRECONDITION (0.02). A seed failing it is precondition_unmet and is
    NOT SCORED (an undiscriminative head would read "veto never fires" for a reason
    unrelated to the producer).
  READINESS: >= MIN_OK_SEEDS (3) ARM_HARM_ON seeds must meet the precondition. Fewer ->
    outcome FAIL, label substrate_not_ready_requeue, non_contributory (NOT a verdict on
    the producer). Seed count 5 chosen so that at 866b's 3/3 rate this is near-certain,
    and at 603k's pessimistic 1/3 rate the shortfall routes to requeue, not FAIL-verdict.
  C1 release (LOAD-BEARING): n_safety_nogo_applied > 0 over the eval (the EVB-1409
    release_condition (b) statistic: safety-vetoed candidates INSIDE the F-built
    eligible set on ticks where the gate ran).
  C2 rate: candidate-level fire rate n_signal_fired / n_candidates_scored <=
    VETO_RATE_CEILING (0.25) -- "well below 100%".
  C3 specificity-harm: over ticks with >= 1 vetoed AND >= 1 non-vetoed candidate,
    mean[ gt_proximity(vetoed) - gt_proximity(non-vetoed) ] > 0, needing >=
    MIN_QUALIFYING_TICKS (5) such ticks on the seed (fewer -> C3 unevaluable on that
    seed, counted as not passed and flagged degenerate).
  C1/C2/C3 each must hold on >= ceil(2/3 * n_ok) of the precondition-met ON seeds.
  C4 specificity-flat (ARM_HARM_OFF_CONTROL, no precondition): fire rate <=
    OFF_FIRE_RATE_CEILING (0.01) on EVERY control seed.
  PASS iff readiness AND C1 AND C2 AND C3 AND C4 (plain AND; combination_rule recorded).

ROUTING (diagnostic; evidence_direction non_contributory on every branch)
  readiness unmet                  -> FAIL substrate_not_ready_requeue
  C1 fails                         -> FAIL endogenous_veto_does_not_fire (producer inert on a
                                      discriminative landscape at +2 SD; calibration finding)
  C1+C2+C4 ok, C3 unmeasurable     -> FAIL endogenous_veto_fires_candidate_specificity_unmeasured
                                      (< ceil(2/3 n_ok) seeds had >= 5 mixed ticks)
  C1 ok, C2/C3/C4 fails            -> FAIL endogenous_veto_nonspecific
  all                              -> PASS endogenous_veto_fires_specifically (EVB-1409 (b) met)
  A PASS clears release_condition (b) only; EVB-1409's re-check is /governance's call.

DV-SYMMETRY (per arm)
  ARM_HARM_ON: DVs are (a) a count of candidates whose harm z-score against a RUNNING
    scale crosses +2 SD and (b) a within-tick paired difference against an external
    ground truth. Training the harm pathway changes the per-candidate harm VECTOR, not a
    broadcast constant. A uniform per-tick offset is NOT invisible to (a) -- the scale is
    running, not per-tick, so a tick that is uniformly more harmful fires every candidate;
    that is exactly why C2 bounds the rate and C3 is within-tick paired (a uniform offset
    cancels in C3 and cannot manufacture it). Not invariant.
  ARM_HARM_OFF_CONTROL: same DVs. The untrained head is NOT flat in candidate space
    (dry run: candidate harm drifts 0.463 -> 0.484 across ticks, running raw sd
    ~0.005-0.007), so C4 genuinely tests whether the 0.01 sd floor (veto needs >= mean
    + 0.02) removes the pure running-z tail; would_fire_without_sd_floor is recorded so a
    reader can see what the floor removed. Not invariant.
  SD FLOOR PROVENANCE: orchestrator decision Q-MECH449-FLOOR -> A (2026-09-24): floor =
    HARM_RANGE_PRECONDITION / Z_THRESHOLD = 0.01, derived from two ratified constants.
    Numeric operationalisations of the three specificity checks (C2 0.25, C4 0.01 on every
    control seed, C3 ground-truth paired with >= 5 mixed ticks, >= 2/3 of ok seeds)
    CONFIRMED by the orchestrator in the same decision.

MACHINE-CLASS NOTE: action selection uses multinomial sampling in places; run on a cloud
worker (machine_affinity any). machine_class is recorded.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _build_env,
    _hazard_proximity_target,
    _measure_harm_discriminativeness,
    _sense_with_optional_harm,
    stage_plan,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_1090_mech449_endogenous_safety_veto_validation"
QUEUE_ID = "V3-EXQ-1090"
CLAIM_IDS: List[str] = ["MECH-449"]
EXPERIMENT_PURPOSE = "diagnostic"
ANCHOR_REACHABILITY_EXEMPT = (
    "readiness preconditions are seed COUNTS of a structural measurement (post-training "
    "harm_eval_range >= 0.02, reached by 3/3 seeds 0.115-0.470 in the identical 603q regime "
    "on V3-EXQ-866b) and a reached-the-eval-phase count -- reachable by construction on any "
    "healthy run, not a signature-reproduction anchor narrower than the state it gates."
)

# Seed 44 substituted by 45 (reef-config early-death instability; queue-experiment 3.5).
SEEDS_ON = [42, 43, 45, 46, 47]
SEEDS_OFF = [42, 43, 45, 46, 47]

# ============================================================================
# 603q ARM_BASE_IA_ONLY CONFIG CONSTANTS (copied from V3-EXQ-866b, which copied 603q)
# ============================================================================
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
P1_BUDGET = 50
TRAIN_STEPS = 200
P1_HOLD_FRACTION = 0.3
P0_NUM_HAZARDS = 1
P2_HFA_GUARD = 0.3
P1_REEF_SPAWN_HOLD_FRACTION = 0.4

HAZARD_STAGE_BUDGET = 40
HAZARD_STAGE_NUM_HAZARDS = 6
HAZARD_STAGE_NUM_RESOURCES = 2
HAZARD_STAGE_HFA = 0.0
HAZARD_STAGE_PROXIMITY_HARM = 0.10
HAZARD_STAGE_SURVIVAL_GATE_STEPS = 75
HAZARD_STAGE_STABILITY_WINDOW = 10

SEED_GAIN = 1.5
SEED_BENEFIT_THRESHOLD = 0.02
SEED_DRIVE_FLOOR = 0.9
N_RESOURCE_TYPES = 3
CUE_RECALL_GAIN = 0.2

AVOIDANCE_SCAFFOLD_FLOOR_START = 0.8
AVOIDANCE_SCAFFOLD_FLOOR_END = 0.0
AVOIDANCE_THREAT_REF = 0.35
PAG_THETA_FREEZE = 0.8
PAG_DURATION_INPUT_THRESHOLD = 0.2

ESCAPE_THREAT_FLOOR = 0.1
ESCAPE_THREAT_REF = 0.35
ESCAPE_APPROACH_GAIN = 0.1
ESCAPE_BIAS_SCALE = 0.1

HARM_PATHWAY_LR = 1e-3
HARM_PATHWAY_ENCODER_LR = 3e-4
HARM_PATHWAY_WARMUP_STEPS = 250
ESCAPE_SAFETY_SIGNAL_THRESHOLD = 0.5

# ============================================================================
# EVAL PHASE + ENDOGENOUS PRODUCER (the manipulation under validation)
# ============================================================================
N_EVAL_EPISODES = 10
EVAL_STEPS = TRAIN_STEPS
# Producer knobs = the ree_core defaults (restated here only to record them; the
# driver passes them explicitly so a later default change cannot silently move them).
GNG_SAFETY_Z_THRESHOLD = 2.0
GNG_SAFETY_EMA_DECAY = 0.999
GNG_SAFETY_SD_FLOOR = 0.01  # = HARM_RANGE_PRECONDITION / Z (Q-MECH449-FLOOR -> A)
GNG_SAFETY_WARMUP_SAMPLES = 200
GNG_SAFETY_FLOOR = 0.5  # the gate's existing default floor (unchanged)

# ============================================================================
# PRE-REGISTERED THRESHOLDS (constants; never derived from this run)
# ============================================================================
HARM_RANGE_PRECONDITION = 0.02
MIN_OK_SEEDS = 3
MIN_SEED_FRACTION = 2.0 / 3.0
VETO_RATE_CEILING = 0.25
MIN_QUALIFYING_TICKS = 5
OFF_FIRE_RATE_CEILING = 0.01
TICK_RECORD_CAP = 600  # per cell; per-tick rows beyond this are summarised only

ARMS = [
    {"label": "ARM_HARM_ON", "train_harm": True, "seeds": SEEDS_ON},
    {"label": "ARM_HARM_OFF_CONTROL", "train_harm": False, "seeds": SEEDS_OFF},
]


def _make_scaffold_cfg(dry_run: bool, train_harm: bool) -> ScaffoldedSD054OnboardingConfig:
    """603q _make_scaffold_cfg for the base arm, with scaffold_train_harm_pathway per arm."""
    if dry_run:
        stage0, stage0b, p0, hazard, p1, steps = 2, 2, 5, 5, 5, 30
    else:
        stage0, stage0b, p0, hazard, p1, steps = (
            STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_STAGE_BUDGET,
            P1_BUDGET, TRAIN_STEPS,
        )
    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_stage0_enabled=True,
        scaffold_stage0_episode_budget=stage0,
        scaffold_p0_episode_budget=p0,
        scaffold_p1_episode_budget=p1,
        scaffold_p2_episode_budget=1,
        scaffold_steps_per_episode=steps,
        scaffold_p0_num_hazards=P0_NUM_HAZARDS,
        scaffold_p1_anneal_hold_fraction=P1_HOLD_FRACTION,
        scaffold_p2_hazard_food_attraction_guard=P2_HFA_GUARD,
        scaffold_developmental_window_enabled=True,
        scaffold_stage0b_enabled=True,
        scaffold_stage0b_episode_budget=stage0b,
        scaffold_stage0b_retention_gate=0.75,
        scaffold_contact_gated_goal_updates=True,
        scaffold_z_goal_seeding_gain=SEED_GAIN,
        scaffold_benefit_threshold=SEED_BENEFIT_THRESHOLD,
        scaffold_drive_floor=SEED_DRIVE_FLOOR,
        scaffold_auto_reconcile_gating_to_seeding=True,
        scaffold_p1_reef_spawn_hold_fraction=P1_REEF_SPAWN_HOLD_FRACTION,
        scaffold_cue_recall_bridge_enabled=True,
        scaffold_cue_n_resource_types=N_RESOURCE_TYPES,
        scaffold_stage0_bind_incentive_token=True,
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=hazard,
        scaffold_hazard_stage_num_hazards=HAZARD_STAGE_NUM_HAZARDS,
        scaffold_hazard_stage_num_resources=HAZARD_STAGE_NUM_RESOURCES,
        scaffold_hazard_stage_hazard_food_attraction=HAZARD_STAGE_HFA,
        scaffold_hazard_stage_proximity_harm_scale=HAZARD_STAGE_PROXIMITY_HARM,
        scaffold_hazard_stage_spawn_in_reef_half=False,
        scaffold_hazard_stage_survival_gate_steps=HAZARD_STAGE_SURVIVAL_GATE_STEPS,
        scaffold_hazard_stage_stability_window=HAZARD_STAGE_STABILITY_WINDOW,
        scaffold_avoidance_driver_enabled=True,
        scaffold_avoidance_scaffold_floor_start=AVOIDANCE_SCAFFOLD_FLOOR_START,
        scaffold_avoidance_scaffold_floor_end=AVOIDANCE_SCAFFOLD_FLOOR_END,
        scaffold_feed_harm_stream=True,
        scaffold_train_harm_pathway=bool(train_harm),
        scaffold_harm_pathway_lr=HARM_PATHWAY_LR,
        scaffold_harm_pathway_in_p0=True,
        scaffold_harm_pathway_encoder_lr=HARM_PATHWAY_ENCODER_LR,
        scaffold_harm_pathway_warmup_steps=HARM_PATHWAY_WARMUP_STEPS,
    )
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env) -> REEConfig:
    """603q _make_config for the base arm (bridge OFF). The gate + producer are OFF
    during training (no-op defaults) and armed only for the eval phase."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        use_harm_stream=True,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_obs_a_dim=HARM_OBS_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        use_e2_harm_s_forward=True,
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        z_goal_enabled=True,
        drive_weight=DRIVE_WEIGHT,
        use_mech295_liking_bridge=True,
        use_mech307_conjunction=True,
        use_incentive_token_bank=True,
        use_cue_recall=True,
        cue_recall_gain=CUE_RECALL_GAIN,
        e2_action_contrastive_enabled=True,
        use_pag_freeze_gate=True,
        pag_theta_freeze=PAG_THETA_FREEZE,
        pag_duration_input_threshold=PAG_DURATION_INPUT_THRESHOLD,
        use_instrumental_avoidance=True,
        avoidance_threat_ref=AVOIDANCE_THREAT_REF,
        use_escape_affordance_bridge=False,
        use_escape_relief_credit=False,
        use_escape_safety_credit=False,
        escape_threat_floor=ESCAPE_THREAT_FLOOR,
        escape_threat_ref=ESCAPE_THREAT_REF,
        escape_approach_gain=ESCAPE_APPROACH_GAIN,
        escape_bias_scale=ESCAPE_BIAS_SCALE,
        escape_use_trained_safety_signal=True,
        escape_safety_signal_threshold=ESCAPE_SAFETY_SIGNAL_THRESHOLD,
        use_contextual_safety_terrain=True,
        use_conditioned_safety_store=True,
        use_suffering_derivative_comparator=True,
    )
    cfg.latent.use_resource_encoder = True
    return cfg


def _arm_gate(agent: REEAgent, dry_run: bool = False) -> Dict[str, Any]:
    """Arm the MECH-449 gate's full chain + the endogenous producer on a TRAINED agent.
    E3 reads these via getattr(self.config, ...) at select time; agent.config.e3 IS
    the selector's config object (verified by the authoring probe: gate active on
    every fresh select after flipping)."""
    e3c = agent.config.e3
    assert agent.e3.config is e3c, "E3 selector does not share agent.config.e3"
    e3c.use_go_nogo_constitution = True
    e3c.use_f_eligibility_demotion = True
    # RED-TEAM F1 (fixed): at the absolute share floor 0.30 over K=32 candidates no
    # candidate holds 30% of the F-merit, so _f_eligibility_envelope returns ALL
    # candidates (dry run: envelope 32/32 on 13/13 ticks) and the within-eligible
    # pick is modulatory-only -- the eval would be F-blind and "inside the F-built
    # eligible set" vacuous. The repo's own channel-adaptive amend (MECH-448 AMEND,
    # mean-relative floor, mean_factor 1.0) keeps the above-average-merit half, so F
    # still bounds eligibility and the gate's in-set count is not the fire count.
    e3c.use_f_eligibility_adaptive_floor = True
    e3c.f_eligibility_adaptive_mean_factor = 1.0
    e3c.use_gng_endogenous_safety = True
    e3c.gng_safety_z_threshold = GNG_SAFETY_Z_THRESHOLD
    e3c.gng_safety_ema_decay = GNG_SAFETY_EMA_DECAY
    e3c.gng_safety_sd_floor = GNG_SAFETY_SD_FLOOR
    # Dry-run only: a short warmup so the smoke exercises the ARMED branch (z, fire
    # mask, ground-truth pairing); the real run uses the pre-registered 200.
    e3c.gng_safety_warmup_samples = 32 if dry_run else GNG_SAFETY_WARMUP_SAMPLES
    e3c.gng_safety_floor = GNG_SAFETY_FLOOR
    agent.reset_gng_safety_state()
    return {
        "use_go_nogo_constitution": True,
        "use_f_eligibility_demotion": True,
        "use_f_eligibility_adaptive_floor": True,
        "f_eligibility_adaptive_mean_factor": 1.0,
        "use_gng_endogenous_safety": True,
        "gng_safety_z_threshold": GNG_SAFETY_Z_THRESHOLD,
        "gng_safety_ema_decay": GNG_SAFETY_EMA_DECAY,
        "gng_safety_sd_floor": GNG_SAFETY_SD_FLOOR,
        "gng_safety_warmup_samples": int(e3c.gng_safety_warmup_samples),
        "gng_safety_floor": GNG_SAFETY_FLOOR,
        "gng_protect_min_eligible": int(getattr(e3c, "gng_protect_min_eligible", -1)),
    }


def _counterfactual_ground_truth(env, actions: List[int]) -> Dict[int, Dict[str, Optional[float]]]:
    """For each distinct first action: step a deepcopy of the env once and read the TRUE
    hazard proximity at the destination + the realised one-step harm. Never touches the
    live env. None values when the copy fails (recorded, C3 then unevaluable)."""
    out: Dict[int, Dict[str, Optional[float]]] = {}
    for a in sorted(set(actions)):
        try:
            env_c = copy.deepcopy(env)
            _, h, _done, _, obs_c = env_c.step(int(a))
            out[a] = {
                "prox": _hazard_proximity_target(obs_c),
                "harm": float(max(0.0, -float(h))),
            }
        except Exception as exc:  # recorded, not swallowed silently
            out[a] = {"prox": None, "harm": None, "error": type(exc).__name__}
    return out


def _eval_phase(scaffold_cfg, agent: REEAgent, device, n_episodes: int, steps: int,
                done_eps: int, total_eps: int, label: str, seed: int) -> Dict[str, Any]:
    """Frozen-policy eval in the Stage-H hazard env with the gate + producer armed."""
    was_training = agent.training
    agent.eval()
    world_dim = agent.config.latent.world_dim
    tick_rows: List[Dict[str, Any]] = []
    n_select = 0
    n_latched = 0
    n_gate_ticks_seen = 0
    n_selected_vetoed = 0
    n_selected_vetoed_all_fired = 0
    n_cf_errors = 0
    paired_diffs: List[float] = []
    paired_diffs_harm: List[float] = []
    within_tick_ranges: List[float] = []
    tick_harm_means: List[float] = []
    max_excursion = None
    envelope_sizes: List[int] = []
    n_env_all_admit = 0
    n_would_fire_no_floor = 0
    n_cand_no_floor = 0
    ep_lengths: List[int] = []
    realised_harm_total = 0.0
    tick_idx = 0
    cf_probe: Dict[int, Dict[str, Optional[float]]] = {}
    for ep in range(n_episodes):
        env = _build_env(scaffold_cfg, phase="hazard")
        _, obs = env.reset()
        agent.reset()
        ep_len = 0
        if ep == 0:
            # Instrument self-check: the ground-truth counterfactual must work on this
            # env class (deepcopy + one step) or C3 is unmeasurable by construction.
            cf_probe = _counterfactual_ground_truth(env, list(range(int(env.action_dim))))
        for t in range(steps):
            ob = obs["body_state"].to(device)
            ow = obs["world_state"].to(device)
            with torch.no_grad():
                lat = _sense_with_optional_harm(agent, ob, ow, obs, device,
                                                scaffold_cfg.scaffold_feed_harm_stream)
                ticks = agent.clock.advance()
                e1p = (agent._e1_tick(lat) if ticks.get("e1_tick")
                       else torch.zeros(1, world_dim, device=device))
                cand = agent.generate_trajectories(lat, e1p, ticks)
                n_scored_before = int(agent.gng_safety_diagnostics()["n_ticks_scored"])
                agent.e3.last_score_diagnostics = None  # clear the latch
                act = agent.select_action(cand, ticks)
            diag = agent.e3.last_score_diagnostics
            act_idx = int(act.argmax(dim=-1).item())
            if diag is None:
                n_latched += 1
            else:
                n_select += 1
                gd = agent.gng_safety_diagnostics()
                scored_now = int(gd["n_ticks_scored"]) > n_scored_before
                gate_ran = bool(diag.get("go_nogo_constitution_active", False))
                if gate_ran:
                    n_gate_ticks_seen += 1
                last = gd.get("last") if scored_now else None
                if last is not None and len(last["harm"]) == len(cand):
                    harm = last["harm"]
                    fired = last["fired"]
                    within_tick_ranges.append(float(max(harm) - min(harm)))
                    tick_harm_means.append(float(np.mean(harm)))
                    if last["armed"]:
                        exc = float(max(harm) - float(last["running_mean_pre"]))
                        max_excursion = exc if max_excursion is None else max(max_excursion, exc)
                    if gate_ran and diag.get("go_nogo_envelope_size") is not None:
                        envelope_sizes.append(int(diag.get("go_nogo_envelope_size")))
                        n_env_all_admit += int(int(diag.get("go_nogo_envelope_size")) >= len(cand))
                    raw_sd = float(last.get("running_sd_raw_pre", 0.0))
                    if last["armed"] and raw_sd > 1e-12:
                        mu = float(last["running_mean_pre"])
                        for hv in harm:
                            n_cand_no_floor += 1
                            if (hv - mu) / raw_sd >= GNG_SAFETY_Z_THRESHOLD:
                                n_would_fire_no_floor += 1
                    first_actions = [int(c.actions[0, 0].argmax().item()) for c in cand]
                    sel_res = getattr(agent, "_last_e3_selection_result", None)
                    sel_idx = getattr(sel_res, "selected_index", None) if sel_res is not None else None
                    sel_vetoed = None
                    if sel_idx is not None and 0 <= int(sel_idx) < len(fired):
                        sel_vetoed = bool(fired[int(sel_idx)])
                        if sel_vetoed and gate_ran:
                            n_selected_vetoed += 1
                            if all(fired):
                                n_selected_vetoed_all_fired += 1
                    n_fired = int(sum(1 for f in fired if f))
                    gt = None
                    diff = None
                    diff_h = None
                    if 0 < n_fired < len(fired):
                        gt = _counterfactual_ground_truth(env, first_actions)
                        if any(v.get("prox") is None for v in gt.values()):
                            n_cf_errors += 1
                        else:
                            pv = [gt[a]["prox"] for a, f in zip(first_actions, fired) if f]
                            pn = [gt[a]["prox"] for a, f in zip(first_actions, fired) if not f]
                            hv_ = [gt[a]["harm"] for a, f in zip(first_actions, fired) if f]
                            hn_ = [gt[a]["harm"] for a, f in zip(first_actions, fired) if not f]
                            diff = float(np.mean(pv) - np.mean(pn))
                            diff_h = float(np.mean(hv_) - np.mean(hn_))
                            paired_diffs.append(diff)
                            paired_diffs_harm.append(diff_h)
                    if len(tick_rows) < TICK_RECORD_CAP:
                        tick_rows.append({
                            "tick": tick_idx, "ep": ep, "step": t, "k": len(cand),
                            "armed": bool(last["armed"]), "gate_ran": gate_ran,
                            "n_fired": n_fired,
                            "n_safety_nogo_in_eligible": int(diag.get("go_nogo_n_safety_nogo", 0)),
                            "envelope_size": diag.get("go_nogo_envelope_size"),
                            "harm_min": float(min(harm)), "harm_max": float(max(harm)),
                            "harm_mean": float(np.mean(harm)),
                            "running_mean_pre": float(last["running_mean_pre"]),
                            "running_sd_pre": float(last["running_sd_pre"]),
                            "running_sd_raw_pre": raw_sd,
                            "z_max": float(max(last["z"])),
                            "selected_index": None if sel_idx is None else int(sel_idx),
                            "selected_vetoed": sel_vetoed,
                            "executed_action": act_idx,
                            "gt_diff_prox": diff, "gt_diff_harm": diff_h,
                        })
                    tick_idx += 1
            _, h, done, _, obs = env.step(act_idx)
            realised_harm_total += float(max(0.0, -float(h)))
            ep_len = t + 1
            if done:
                break
        ep_lengths.append(ep_len)
        print(f"  [train] eval {label} seed={seed} ep {done_eps + ep + 1}/{total_eps}"
              f" len={ep_len} fired_total={agent.gng_safety_diagnostics()['n_signal_fired']}"
              f" applied_total={agent.gng_safety_diagnostics()['n_safety_nogo_applied']}",
              flush=True)
    if was_training:
        agent.train()
    gd = agent.gng_safety_diagnostics()
    gd.pop("last", None)
    n_cand = int(gd["n_candidates_scored"])
    fire_rate = (float(gd["n_signal_fired"]) / n_cand) if n_cand > 0 else None
    return {
        "counterfactual_probe_ok": bool(all(v.get("prox") is not None for v in cf_probe.values())),
        "counterfactual_probe": {str(k): v for k, v in cf_probe.items()},
        "producer_diagnostics": gd,
        "fire_rate": fire_rate,
        "n_select_ticks": n_select,
        "n_latched_ticks": n_latched,
        "n_gate_active_ticks_seen": n_gate_ticks_seen,
        "n_selected_vetoed_while_gate_ran": n_selected_vetoed,
        "n_selected_vetoed_all_fired": n_selected_vetoed_all_fired,
        "n_counterfactual_errors": n_cf_errors,
        "n_qualifying_ticks": len(paired_diffs),
        "gt_paired_diff_prox_mean": float(np.mean(paired_diffs)) if paired_diffs else None,
        "gt_paired_diff_harm_mean": float(np.mean(paired_diffs_harm)) if paired_diffs_harm else None,
        "gt_paired_diff_prox_frac_positive": (
            float(np.mean([d > 0 for d in paired_diffs])) if paired_diffs else None),
        "within_tick_harm_range_median": (
            float(np.median(within_tick_ranges)) if within_tick_ranges else None),
        "within_tick_harm_range_frac_above_2x_floor": (
            float(np.mean([r > 2.0 * GNG_SAFETY_SD_FLOOR for r in within_tick_ranges]))
            if within_tick_ranges else None),
        "predicted_state_harm_across_tick_range": (
            float(max(tick_harm_means) - min(tick_harm_means)) if tick_harm_means else None),
        "predicted_state_max_excursion_above_running_mean": max_excursion,
        "min_veto_excursion": float(GNG_SAFETY_Z_THRESHOLD * GNG_SAFETY_SD_FLOOR),
        "gate_envelope_size_median": (float(np.median(envelope_sizes)) if envelope_sizes else None),
        "gate_envelope_frac_all_admit": (
            float(n_env_all_admit) / len(envelope_sizes) if envelope_sizes else None),
        "would_fire_without_sd_floor_rate": (
            float(n_would_fire_no_floor) / n_cand_no_floor if n_cand_no_floor > 0 else None),
        "eval_episode_lengths": ep_lengths,
        "eval_realised_harm_total": realised_harm_total,
        "tick_rows": tick_rows,
        "tick_rows_truncated": bool(tick_idx > len(tick_rows)),
    }


def _config_slice(arm: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
    return {
        "arm": arm["label"],
        "scaffold_train_harm_pathway": bool(arm["train_harm"]),
        "base": "603q ARM_BASE_IA_ONLY (via V3-EXQ-866b constants)",
        "budgets": [STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_STAGE_BUDGET,
                    P1_BUDGET, TRAIN_STEPS],
        "hazard_stage": [HAZARD_STAGE_NUM_HAZARDS, HAZARD_STAGE_NUM_RESOURCES,
                         HAZARD_STAGE_HFA, HAZARD_STAGE_PROXIMITY_HARM],
        "harm_pathway": [HARM_PATHWAY_LR, HARM_PATHWAY_ENCODER_LR, HARM_PATHWAY_WARMUP_STEPS],
        "eval": [N_EVAL_EPISODES, EVAL_STEPS],
        "producer": [GNG_SAFETY_Z_THRESHOLD, GNG_SAFETY_EMA_DECAY, GNG_SAFETY_SD_FLOOR,
                     GNG_SAFETY_WARMUP_SAMPLES, GNG_SAFETY_FLOOR],
        "gate_chain": ["use_go_nogo_constitution", "use_f_eligibility_demotion",
                       "use_f_eligibility_adaptive_floor"],
        "dry_run": bool(dry_run),
    }


def _total_eps(dry_run: bool) -> int:
    if dry_run:
        return 2 + 2 + 5 + 5 + 5 + 2
    return (STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_STAGE_BUDGET
            + P1_BUDGET + N_EVAL_EPISODES)


def _aborted(arm, seed, stage, reason) -> Dict[str, Any]:
    return {"arm": arm["label"], "seed": seed, "aborted_at": stage, "abort_reason": reason,
            "harm_eval_range_post_training": 0.0, "precondition_met": False,
            "eval": None}


def _run_cell(arm: Dict[str, Any], seed: int, dry_run: bool, total_eps: int,
              zg_acc: ZGoalStreamAccumulator) -> Dict[str, Any]:
    with arm_cell(seed, config_slice=_config_slice(arm, dry_run), script_path=Path(__file__),
                  config_slice_declared=True) as cell:
        scfg = _make_scaffold_cfg(dry_run, arm["train_harm"])
        device = torch.device("cpu")
        probe_env = _build_env(scfg, "p2")
        probe_env.reset()
        agent = REEAgent(_make_config(probe_env)).to(device)
        sched = ScaffoldedSD054OnboardingScheduler(scfg)
        label = arm["label"]
        print(f"Seed {seed} Condition {label}", flush=True)

        s0 = sched.run_stage0_nursery(agent, device)
        done = s0.n_episodes
        print(f"  [train] stage0 {label} seed={seed} ep {done}/{total_eps}"
              f" z_goal_peak={s0.z_goal_norm_peak:.4f}", flush=True)
        if s0.aborted:
            print(f"verdict: FAIL seed={seed} arm={label} aborted_at=stage0", flush=True)
            rec = _aborted(arm, seed, "stage0", s0.abort_reason)
            zg_acc.observe(agent)
            cell.stamp(rec)
            return rec
        s0b = sched.run_stage0b_consolidation(agent, device,
                                              stage0_baseline_norm=s0.z_goal_norm_peak)
        done += s0b.n_episodes
        if s0b.aborted:
            print(f"verdict: FAIL seed={seed} arm={label} aborted_at=stage0b", flush=True)
            rec = _aborted(arm, seed, "stage0b", s0b.abort_reason)
            zg_acc.observe(agent)
            cell.stamp(rec)
            return rec
        p0 = sched.run_p0(agent, device)
        done += p0.n_episodes
        print(f"  [train] p0 {label} seed={seed} ep {done}/{total_eps}"
              f" mean_len={p0.mean_episode_length:.1f}", flush=True)
        if p0.aborted:
            print(f"verdict: FAIL seed={seed} arm={label} aborted_at=p0", flush=True)
            rec = _aborted(arm, seed, "p0", p0.abort_reason)
            zg_acc.observe(agent)
            cell.stamp(rec)
            return rec
        hz = sched.run_hazard_avoidance(agent, device)
        done += hz.n_episodes
        hd_stage_h = dict(hz.harm_discriminativeness or {})
        print(f"  [train] hazard {label} seed={seed} ep {done}/{total_eps}"
              f" mean_len={hz.mean_episode_length:.1f}"
              f" harm_range_stageH={float(hd_stage_h.get('harm_eval_range', 0.0)):.4f}",
              flush=True)
        if hz.aborted:
            print(f"verdict: FAIL seed={seed} arm={label} aborted_at=hazard", flush=True)
            rec = _aborted(arm, seed, "hazard", hz.abort_reason)
            rec["harm_discriminativeness_stage_h"] = hd_stage_h
            zg_acc.observe(agent)
            cell.stamp(rec)
            return rec
        p1 = sched.run_p1(agent, device)
        done += p1.n_episodes
        print(f"  [train] p1 {label} seed={seed} ep {done}/{total_eps}", flush=True)

        # PRECONDITION: post-training discriminativeness, measured BEFORE arming, with
        # the scheduler's own read-only probe (the 866b-comparable statistic).
        hd_post = _measure_harm_discriminativeness(scfg, agent, device)
        harm_range_post = float(hd_post.get("harm_eval_range", 0.0))
        precondition_met = bool(harm_range_post >= HARM_RANGE_PRECONDITION)

        gate_cfg = _arm_gate(agent, dry_run=dry_run)
        n_eval = 2 if dry_run else N_EVAL_EPISODES
        ev = _eval_phase(scfg, agent, device, n_eval, 30 if dry_run else EVAL_STEPS,
                         done, total_eps, label, seed)
        done += n_eval

        pd = ev["producer_diagnostics"]
        c1 = bool(int(pd["n_safety_nogo_applied"]) > 0)
        c2 = bool(ev["fire_rate"] is not None and ev["fire_rate"] <= VETO_RATE_CEILING)
        c3_evaluable = bool(ev["n_qualifying_ticks"] >= MIN_QUALIFYING_TICKS)
        c3 = bool(c3_evaluable and (ev["gt_paired_diff_prox_mean"] or 0.0) > 0.0)
        c4 = bool(ev["fire_rate"] is not None and ev["fire_rate"] <= OFF_FIRE_RATE_CEILING)
        if arm["train_harm"]:
            seed_pass = bool(precondition_met and c1 and c2 and c3)
        else:
            seed_pass = c4
        print(f"verdict: {'PASS' if seed_pass else 'FAIL'} seed={seed} arm={label}"
              f" harm_range_post={harm_range_post:.4f} precond={precondition_met}"
              f" applied={pd['n_safety_nogo_applied']} fire_rate={ev['fire_rate']}"
              f" qual_ticks={ev['n_qualifying_ticks']}"
              f" gt_diff={ev['gt_paired_diff_prox_mean']}", flush=True)
        rec = {
            "arm": label, "seed": seed, "aborted_at": None, "abort_reason": "",
            "train_harm": bool(arm["train_harm"]),
            "stage0_z_goal_norm_peak": float(s0.z_goal_norm_peak),
            "p0_mean_episode_length": float(p0.mean_episode_length),
            "hazard_stage_mean_episode_length": float(hz.mean_episode_length),
            "hazard_stage_episode_lengths": list(hz.episode_lengths or []),
            "harm_pathway_n_train_steps": int((hz.harm_pathway_diag or {}).get("n_train_steps", 0)),
            "harm_discriminativeness_stage_h": hd_stage_h,
            "harm_discriminativeness_post_training": hd_post,
            "harm_eval_range_post_training": harm_range_post,
            "precondition_met": precondition_met,
            "gate_config": gate_cfg,
            "eval": ev,
            "c1_release_applied_gt0": c1,
            "c2_fire_rate_below_ceiling": c2,
            "c3_evaluable": c3_evaluable,
            "c3_vetoed_more_ground_truth_harm": c3,
            "c4_off_fire_rate_near_zero": c4,
            "seed_pass": seed_pass,
        }
        zg_acc.observe(agent)
        cell.stamp(rec)
        return rec


def _need(n: int) -> int:
    return int(math.ceil(MIN_SEED_FRACTION * n - 1e-9))


def _flat(v: Any) -> Any:
    if isinstance(v, bool):
        return int(v)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    total_eps = _total_eps(dry_run)
    zg_acc = ZGoalStreamAccumulator()
    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        seeds = arm["seeds"][:1] if dry_run else arm["seeds"]
        for s in seeds:
            rows.append(_run_cell(arm, s, dry_run, total_eps, zg_acc))

    on_rows = [r for r in rows if r["arm"] == "ARM_HARM_ON"]
    off_rows = [r for r in rows if r["arm"] == "ARM_HARM_OFF_CONTROL"]
    ok_rows = [r for r in on_rows if r.get("precondition_met")]
    n_ok = len(ok_rows)
    need = _need(n_ok) if n_ok else 0
    min_ok = 1 if dry_run else MIN_OK_SEEDS
    readiness = bool(n_ok >= min_ok)

    n_c1 = sum(1 for r in ok_rows if r.get("c1_release_applied_gt0"))
    n_c2 = sum(1 for r in ok_rows if r.get("c2_fire_rate_below_ceiling"))
    n_c3 = sum(1 for r in ok_rows if r.get("c3_vetoed_more_ground_truth_harm"))
    n_c3_eval = sum(1 for r in ok_rows if r.get("c3_evaluable"))
    off_scored = [r for r in off_rows if r.get("eval") is not None]
    n_c4 = sum(1 for r in off_scored if r.get("c4_off_fire_rate_near_zero"))
    c1 = bool(readiness and n_c1 >= need)
    c2 = bool(readiness and n_c2 >= need)
    c3 = bool(readiness and n_c3 >= need)
    c4 = bool(len(off_scored) > 0 and n_c4 == len(off_scored))

    gate_never_ran = bool(ok_rows) and all(
        int(r["eval"]["producer_diagnostics"]["n_gate_active_ticks"]) == 0 for r in ok_rows)
    if not readiness or not off_scored or gate_never_ran:
        # RED-TEAM F4 (fixed): an unscored control or a gate that never armed is an
        # unmeasured state, not a verdict on the producer.
        outcome, label = "FAIL", "substrate_not_ready_requeue"
    elif not c1:
        outcome, label = "FAIL", "endogenous_veto_does_not_fire"
    elif n_c3_eval < need and c2 and c4:
        # C3 could not be MEASURED (too few ticks mixing vetoed and non-vetoed
        # candidates) -- e.g. the veto fires on whole ticks because predicted world
        # states barely differ across candidates. Not a specificity failure.
        outcome, label = "FAIL", "endogenous_veto_fires_candidate_specificity_unmeasured"
    elif not (c2 and c3 and c4):
        outcome, label = "FAIL", "endogenous_veto_nonspecific"
    else:
        outcome, label = "PASS", "endogenous_veto_fires_specifically"

    def _vals(rs, key):
        return [((r.get("eval") or {}).get(key)) for r in rs]

    on_fire_rates = _vals(on_rows, "fire_rate")
    off_fire_rates = _vals(off_rows, "fire_rate")
    worst_on_rate = max([x for x in _vals(ok_rows, "fire_rate") if x is not None], default=None)
    worst_off_rate = max([x for x in off_fire_rates if x is not None], default=None)

    preconditions = [
        {
            "name": "harm_on_seeds_meeting_harm_range_precondition",
            "kind": "readiness",
            "description": "Count of ARM_HARM_ON seeds whose post-training harm_eval_range "
                           "(scheduler _measure_harm_discriminativeness, before arming) >= "
                           f"{HARM_RANGE_PRECONDITION}. Seeds below it are precondition_unmet and "
                           "unscored. Fewer than MIN_OK_SEEDS -> substrate_not_ready_requeue.",
            "control": "603q/866b regime: 3/3 seeds 0.115-0.470 on 2026-08-03.",
            "measured": float(n_ok),
            "threshold": float(min_ok),
            "direction": "lower",
            "met": readiness,
        },
        {
            "name": "harm_off_control_scored",
            "kind": "readiness",
            "description": "At least one ARM_HARM_OFF_CONTROL seed reached the eval phase "
                           "(else C4 is unmeasured).",
            "control": "same curriculum, harm pathway untrained.",
            "measured": float(len(off_scored)),
            "threshold": 1.0,
            "direction": "lower",
            "met": bool(len(off_scored) >= 1),
        },
    ]
    criteria = [
        {"name": "C1_release_safety_nogo_applied_gt0", "load_bearing": True,
         "measured": float(n_c1), "threshold": float(need), "n_scored_seeds": n_ok,
         "per_seed_statistic": "n_safety_nogo_applied > 0", "passed": c1},
        {"name": "C2_candidate_fire_rate_below_ceiling", "load_bearing": True,
         "measured": float(n_c2), "threshold": float(need), "n_scored_seeds": n_ok,
         "rate_ceiling": VETO_RATE_CEILING, "worst_scored_seed_rate": worst_on_rate,
         "passed": c2},
        {"name": "C3_vetoed_carry_more_ground_truth_hazard", "load_bearing": True,
         "measured": float(n_c3), "threshold": float(need), "n_scored_seeds": n_ok,
         "n_seeds_evaluable": n_c3_eval, "min_qualifying_ticks": MIN_QUALIFYING_TICKS,
         "passed": c3},
        {"name": "C4_harm_off_control_fire_rate_near_zero", "load_bearing": True,
         "measured": float(n_c4), "threshold": float(len(off_scored)),
         "rate_ceiling": OFF_FIRE_RATE_CEILING, "worst_control_seed_rate": worst_off_rate,
         "passed": c4},
    ]
    criteria_non_degenerate = {
        "C1": bool(readiness),
        "C2": bool(readiness and any(r["eval"]["producer_diagnostics"]["n_candidates_scored"] > 0
                                     for r in ok_rows)),
        "C3": bool(n_c3_eval >= 2),
        "C4": bool(len(off_scored) >= 1),
    }
    readout = {
        "n_on_seeds_precondition_met": n_ok,
        "n_on_seeds": len(on_rows),
        "n_c1_seeds": n_c1, "n_c2_seeds": n_c2, "n_c3_seeds": n_c3, "n_c4_seeds": n_c4,
        "seeds_needed": need,
        "readiness_met": int(readiness),
        "c1_pass": int(c1), "c2_pass": int(c2), "c3_pass": int(c3), "c4_pass": int(c4),
        "worst_scored_on_fire_rate": _flat(worst_on_rate),
        "worst_off_fire_rate": _flat(worst_off_rate),
        "on_total_safety_nogo_applied": int(sum(
            (r.get("eval") or {}).get("producer_diagnostics", {}).get("n_safety_nogo_applied", 0)
            for r in ok_rows)),
    }
    readout = {k: v for k, v in readout.items() if v is not None}
    print(f"[{EXPERIMENT_TYPE}] n_ok={n_ok}/{len(on_rows)} need={need} C1={n_c1} C2={n_c2}"
          f" C3={n_c3} C4={n_c4}/{len(off_scored)} -> {outcome} {label}", flush=True)

    return {
        "outcome": outcome,
        "evidence_direction": "non_contributory",
        "evidence_direction_per_claim": {cid: "non_contributory" for cid in CLAIM_IDS},
        "readout": readout,
        "per_seed_on_fire_rate": on_fire_rates,
        "per_seed_off_fire_rate": off_fire_rates,
        "per_seed_on_harm_range_post": [r.get("harm_eval_range_post_training") for r in on_rows],
        "per_seed_off_harm_range_post": [r.get("harm_eval_range_post_training") for r in off_rows],
        "per_seed_on_safety_nogo_applied": [
            ((r.get("eval") or {}).get("producer_diagnostics") or {}).get("n_safety_nogo_applied")
            for r in on_rows],
        "per_seed_on_gt_paired_diff_prox": _vals(on_rows, "gt_paired_diff_prox_mean"),
        "per_seed_on_would_fire_without_floor": _vals(on_rows, "would_fire_without_sd_floor_rate"),
        "per_seed_off_would_fire_without_floor": _vals(off_rows, "would_fire_without_sd_floor_rate"),
        "per_seed_on_within_tick_range_median": _vals(on_rows, "within_tick_harm_range_median"),
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "criteria": criteria,
            "combination_rule": "PASS iff readiness AND C1 AND C2 AND C3 AND C4; C1-C3 each on "
                                ">= ceil(2/3 * n_precondition_met_ON_seeds); C4 on every "
                                "scored control seed.",
            "grid": {
                "substrate_not_ready_requeue": "< MIN_OK_SEEDS ON seeds with a discriminative "
                                               "harm head, OR no control seed scored, OR the "
                                               "gate never armed on any scored seed: not a "
                                               "verdict on the producer.",
                "endogenous_veto_does_not_fire": "precondition-met head (real-state range), gate "
                                                 "armed, but the +2 SD running-scale veto never "
                                                 "removes an eligible candidate. READ WITH "
                                                 "predicted_state_max_excursion_above_running_mean "
                                                 "vs min_veto_excursion (red-team F2): below it, "
                                                 "the predicted-state landscape the veto reads "
                                                 "never offered a 0.02 excursion (instrument "
                                                 "headroom), above it the calibration is too "
                                                 "conservative.",
                "endogenous_veto_fires_candidate_specificity_unmeasured": "C1/C2/C4 hold but "
                    "fewer than the needed seeds had >= MIN_QUALIFYING_TICKS mixed ticks, so C3 "
                    "was not measured (veto fires on whole ticks); read "
                    "per_seed_on_within_tick_range_median.",
                "endogenous_veto_nonspecific": "fires, but too often, or not on the "
                                               "ground-truth-more-hazardous candidates, or on a "
                                               "flat landscape.",
                "endogenous_veto_fires_specifically": "EVB-1409 release_condition (b) met.",
            },
        },
        "per_seed": rows,
        "_zg": zg_acc.stats(),
        "_t0": t0,
    }


def main(dry_run: bool = False) -> Dict[str, Any]:
    result = run_experiment(dry_run=dry_run)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    zg = result.pop("_zg", None)
    t0 = result.pop("_t0", None)
    full_config = {
        "base": "V3-EXQ-603q ARM_BASE_IA_ONLY (constants copied via V3-EXQ-866b)",
        "budgets": {"stage0": STAGE0_BUDGET, "stage0b": STAGE0B_BUDGET, "p0": P0_BUDGET,
                    "hazard": HAZARD_STAGE_BUDGET, "p1": P1_BUDGET,
                    "steps_per_episode": TRAIN_STEPS, "eval_episodes": N_EVAL_EPISODES,
                    "eval_steps": EVAL_STEPS},
        "hazard_stage": {"num_hazards": HAZARD_STAGE_NUM_HAZARDS,
                         "num_resources": HAZARD_STAGE_NUM_RESOURCES,
                         "hazard_food_attraction": HAZARD_STAGE_HFA,
                         "proximity_harm_scale": HAZARD_STAGE_PROXIMITY_HARM},
        "harm_pathway": {"lr": HARM_PATHWAY_LR, "encoder_lr": HARM_PATHWAY_ENCODER_LR,
                         "warmup_steps": HARM_PATHWAY_WARMUP_STEPS},
        "producer": {"z_threshold": GNG_SAFETY_Z_THRESHOLD, "ema_decay": GNG_SAFETY_EMA_DECAY,
                     "sd_floor": GNG_SAFETY_SD_FLOOR,
                     "warmup_samples": GNG_SAFETY_WARMUP_SAMPLES,
                     "gate_safety_floor": GNG_SAFETY_FLOOR},
        "thresholds": {"harm_range_precondition": HARM_RANGE_PRECONDITION,
                       "min_ok_seeds": MIN_OK_SEEDS, "min_seed_fraction": MIN_SEED_FRACTION,
                       "veto_rate_ceiling": VETO_RATE_CEILING,
                       "min_qualifying_ticks": MIN_QUALIFYING_TICKS,
                       "off_fire_rate_ceiling": OFF_FIRE_RATE_CEILING},
        "arms": [{"label": a["label"], "train_harm": a["train_harm"], "seeds": a["seeds"]}
                 for a in ARMS],
    }
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "sleep_driver_pattern": "N/A (waking onboarding scheduler; no sleep loop)",
        "substrate": "MECH-449 endogenous safety producer (E3Config.use_gng_endogenous_safety, "
                     "2026-09-24) on the 603q ARM_BASE_IA_ONLY trained agent",
        "calibration_provenance": "orchestrator decision DECIDED Q-MECH449 -> A "
                                  "(orchestrate-20260924-0808, 2026-09-24T11:03:55Z): per-seed "
                                  "running harm scale, veto at > +2 SD, three specificity checks, "
                                  "per-seed harm_range >= 0.02 precondition.",
        "unblocks_release_test": "REE_assembly EVB-1409 / EXP-0796 release_condition (b)",
        "stage_plan": stage_plan(),
        "arm_results": [
            {"arm": r["arm"], "seed": r["seed"], "arm_fingerprint": r.get("arm_fingerprint"),
             "precondition_met": r.get("precondition_met"), "seed_pass": r.get("seed_pass")}
            for r in result["per_seed"]
        ],
    }
    manifest.update(result)
    out_path = write_flat_manifest(
        manifest, out_dir, dry_run=dry_run, config=full_config,
        seeds=sorted(set(SEEDS_ON) | set(SEEDS_OFF)), script_path=Path(__file__),
        started_at=t0, z_goal_stream_stats=zg,
    )
    print(f"[{EXPERIMENT_TYPE}] manifest -> {out_path}", flush=True)
    print(f"Done. Outcome: {result['outcome']} label: {result['interpretation']['label']}",
          flush=True)
    return {"outcome": result["outcome"], "manifest_path": str(out_path)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    _res = main(dry_run=args.dry_run)
    _outcome_raw = str(_res["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=_res["manifest_path"],
        dry_run=bool(args.dry_run),
    )
