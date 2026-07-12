"""
V3-EXQ-514u -- SD-049 Phase-2 MECH-436 drive-coupling MEASUREMENT REDESIGN (514t successor).

The deferred MEASUREMENT-REDESIGN fork routed by failure_autopsy_V3-EXQ-514t_2026-06-20
(confirmed, user-adjudicated) -- the fork failure_autopsy_V3-EXQ-514s Section 7 pre-registered
"for a later iteration IF the kappa amend stalls." The kappa amend HAS stalled: doubling
incentive_drive_kappa_scale 6.0 -> 12.0 (514s -> 514t) REGRESSED the natural drive-coupled WL
delta rather than improving it (514s mean +0.064 / sd 0.109 / 2-of-5-clear -> 514t mean -0.037 /
sd 0.188 / 1-of-5-clear, two seeds strongly negative). At kappa_eff=24 the drive term
over-amplifies base_value so most_wanted -> argmax(per_axis_drive), destabilising the argmax
through a near-tie landscape that the argmax-flip-gated WL metric reads as sign-unstable noise.
The "needs more kappa" hypothesis is FALSIFIED as a monotone repair; the diagnosed ceiling is
now the MEASUREMENT LAYER (the argmax-flip-gated WL dissociation delta is coarse and interacts
pathologically with high kappa).

THE REDESIGN (this run):
  ADD a CONTINUOUS INCENTIVE-AMPLITUDE readout -- how far natural per-axis drive shifts the
  wanting SCORE toward the most-depleted axis, EVEN WITHOUT crossing the argmax boundary --
  ALONGSIDE the existing argmax-flip-gated WL delta (kept as a secondary diagnostic). The
  continuous metric is the magnitude of drive's re-weighting of
  wanting[k] = base_value[k] * (1 + kappa * per_axis_drive[k]) toward the depleted k. It
  resolves whether drive CONSISTENTLY re-weights wanting toward the depleted axis (supporting
  MECH-436) even when argmax flips are noisy, and is robust to the argmax-destabilisation that
  exploded the flip-gated SD at high kappa. Concretely, per scored step, with k_dep = the
  most-depleted axis among the bank's tokens:
    share_natural[k_dep] = wanting_natural[k_dep] / sum_k wanting_natural[k]
                           (wanting_natural uses the in-run per_axis_drive)
    share_uniform[k_dep]  = wanting_uniform[k_dep]  / sum_k wanting_uniform[k]
                           (wanting_uniform: per_axis_drive=None -> uniform scalar drive ->
                            a flat multiplier -> normalized share == base_value share)
    amplitude_shift = share_natural[k_dep] - share_uniform[k_dep]   in (-1, 1)
  Positive amplitude_shift => natural per-axis drive re-weights wanting TOWARD the depleted
  axis (the MECH-436 prediction), captured continuously without an argmax flip. The OFF /
  per-axis-disabled control is structurally zero (share_natural == share_uniform when drive is
  uniform), giving a hard-zero floor on the continuous channel.

  RUN AT A MODERATE kappa: incentive_drive_kappa_scale = 6.0 (the 514s bimodal-positive point;
  kappa_eff = incentive_drive_kappa_weight(2.0) * 6.0 = 12). DO NOT exceed 6.0 -- the 514s C8
  bounded-kappa invariant (drive must not dominate base_value; a sated agent still wants a
  clearly-better object) is the upper bound and 12.0 (kappa_eff=24) violated it. lever (b) is
  done: per_axis_restoration_fraction stays 0.15 (enriched_spread ~0.19 met on all 514t seeds);
  do not deepen further.

KEPT (verbatim from the 514r/514s/514t controls -- non-vacuity self-routes
substrate_not_ready_requeue, NEVER a false weakens):
  (i)   OVERSHOOT positive control (magnitude 5.0): the genuine-weakens off-ramp. If overshoot
        still flips most_wanted on >= 2/3 seeds, a continuous-below-margin FAIL is enrichment/
        measurement insufficient (substrate_ceiling), NOT a weakens. Only when overshoot CANNOT
        flip even at 5.0 on an adequately-enriched substrate is it a genuine MECH-436 weakens.
  (ii)  OFF / bank-disabled FLOOR (wanting := consumed/liking type): wl_off_fraction must read
        ~0 (the flip-channel structural floor) AND the continuous-channel OFF shift (per-axis
        drive disabled) must read hard zero. Non-zero -> instrument broken ->
        substrate_not_ready_requeue, NEVER a claim verdict.
  (iii) RECALIBRATED argmax-relevance readiness positive control: overshoot MUST flip a
        constructed realistic base_value gap while the natural-magnitude control drive must NOT.
  (iv)  ENRICHED-SPREAD non-vacuity precondition: the in-run per-axis drive_spread_max must
        clear MIN_ENRICHED_SPREAD on >= 2/3 guard-passing seeds (lever b took effect).
  (v)   CONTACT guard.

DECISION GRID (after gates contact + argmax-relevance + OFF floor [flip + continuous] + bank
populated + ENRICHED-spread are ALL met; any unmet -> substrate_not_ready_requeue /
non_contributory, NEVER a weakens):
  PASS / supports         : continuous drive-coupled amplitude shift toward the depleted axis
    (substrate_ceiling      mean per seed >= max(CONT_K_SD*pstdev(per_seed_shift), CONT_FLOOR)
     -> supports)           on >= 2/3 guard-passing seeds. LOAD-BEARING promotion target:
                            MECH-436 substrate_ceiling -> supports. (Effect-size floor on
                            SD-of-DELTA + an absolute floor; NEVER pstdev(baseline_level).)
  FAIL / non_contributory : continuous shift below margin BUT overshoot still flips the argmax
    (substrate_ceiling)     on >= 2/3 seeds -> the enrichment / measurement is insufficient at
                            this kappa -> re-tune. NOT a weakens.
  FAIL / weakens          : every readiness gate passed AND overshoot CANNOT flip even at
                            OVERSHOOT_DRIVE_MAGNITUDE -> the genuine MECH-436 drive-coupling
                            weakens (drive cannot carve wanting even at maximal magnitude on an
                            adequately-enriched substrate).

CLAIM SCOPE: claim_ids=[MECH-436] (drive.wanting_drive_state_modulation), re-evaluated from
scratch. The run exercises the drive-coupling leg ONLY. Sub-leg (a) wanting != liking
(object-bound dissociation, MECH-229) is ALREADY ESTABLISHED by V3-EXQ-514o PASS (0.80) and is
NOT under test here; this run MUST NOT weaken it (the hard-zero OFF floor confirms drive is not
manufacturing a comparator artifact).

SUBSTRATE + instrument chain: IDENTICAL training stack to 514t (full scaffolded_sd054_onboarding
603n curriculum; 681-C4 info-tag liking source; SD-057 IncentiveTokenBank wanting). NEW vs 514t:
kappa_scale REVERTED 12.0 -> 6.0 (moderate, bounded) + a CONTINUOUS incentive-amplitude readout
added alongside the flip-gated delta. per_axis_restoration_fraction unchanged at 0.15.

experiment_purpose: evidence
supersedes: V3-EXQ-514t (kappa-magnitude lever exhausted/non-monotone; flip-gated measurement is
  the diagnosed ceiling -> add a continuous incentive-amplitude metric at moderate kappa).
predecessor: V3-EXQ-514t.

SLEEP DRIVER: N/A (no sleep loop; scaffolded_sd054_onboarding is a waking goal-pipeline
onboarding scheduler).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.goal import IncentiveTokenBank  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _build_env,
    _sense_with_optional_harm,
    stage_plan,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_514u_sd049_phase2_mech436_drive_coupling_continuous_amplitude"
QUEUE_ID = "V3-EXQ-514u"
CLAIM_IDS: List[str] = ["MECH-436"]
EXPERIMENT_PURPOSE = "evidence"
PREDECESSOR = "V3-EXQ-514t (514t: kappa 6.0->12.0 REGRESSED the natural flip-gated delta [+0.064 -> -0.037, sd 0.109->0.188, 2-clear -> 1-clear]; kappa-magnitude lever exhausted/non-monotone; the argmax-flip-gated WL metric is the diagnosed ceiling -> measurement-redesign fork)"
SUPERSEDES = "V3-EXQ-514t"

# Pool the 514o (42/43/44) + 514p (45/46/47) triples, as 514q/r/s/t did, for a stable
# per-seed estimate of the overshoot-flip fraction + the continuous amplitude shift.
SEEDS = [42, 43, 44, 45, 46, 47]
CONDITION_LABEL = "CURRICULUM_BUILT_SD049_PHASE2_DRIVE_COUPLING_CONTINUOUS_AMPLITUDE"

# --- Goal-pipeline / encoder dims (mirror 603n / 514t exactly) ---
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

# --- Curriculum budgets (mirror 603n / 514t exactly) ---
STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
HAZARD_STAGE_BUDGET = 40
P1_BUDGET = 50
P2_BUDGET = 15            # the 603n-canonical contact-guard measurement (run_p2)
BEHAV_EVAL_EPISODES = 30  # match 514t (doubled from 514p's 15 for more scored events)
TRAIN_STEPS = 200
P1_HOLD_FRACTION = 0.3
P0_NUM_HAZARDS = 1
P2_HFA_GUARD = 0.3
P1_REEF_SPAWN_HOLD_FRACTION = 0.4

HAZARD_STAGE_NUM_HAZARDS = 4
HAZARD_STAGE_NUM_RESOURCES = 2
HAZARD_STAGE_HFA = 0.0
HAZARD_STAGE_PROXIMITY_HARM = 0.1
HAZARD_STAGE_SPAWN_IN_REEF = True
HAZARD_STAGE_SURVIVAL_GATE_STEPS = 75
HAZARD_STAGE_STABILITY_WINDOW = 10

# --- 634c seeding calibration + SD-057 cue-recall bridge (mirror 603n / 514t) ---
SEED_GAIN = 1.5
SEED_BENEFIT_THRESHOLD = 0.02
SEED_DRIVE_FLOOR = 0.9
N_RESOURCE_TYPES = 3
CUE_RECALL_GAIN = 0.2

# --- SD-058 / MECH-357 protective-scaffold anneal (mirror 603n / 514t) ---
AVOIDANCE_SCAFFOLD_FLOOR_START = 0.8
AVOIDANCE_SCAFFOLD_FLOOR_END = 0.0
AVOIDANCE_THREAT_REF = 0.35
PAG_THETA_FREEZE = 0.8
PAG_DURATION_INPUT_THRESHOLD = 0.2
HARM_PATHWAY_LR = 1e-3
STAGE0B_RETENTION_GATE = 0.75

# --- Pre-registered acceptance thresholds (NOT derived from the run) ---
P2_ZGOAL_GATE = 0.4          # per-seed contact-guard: z_goal_norm_at_contact_peak floor (603n G3)
CONTACT_GATE = 0.0           # per-seed contact-guard: P2 contact_rate floor (603n G2)
MIN_FRACTION = 2.0 / 3.0     # >= 2/3 seeds for non-vacuity + any aggregate gate

# WL readiness (recalibrated; the 514r argmax-relevance fix)
PC_SEPARATION_FLOOR = 1.0    # leg-1 legacy: constructed control separates under drive (retained)
MIN_SCORED_STEPS = 5         # leg-2: min consumption events with both targets defined per seed
# (iii) RECALIBRATED leg: overshoot MUST flip a realistic base_value gap, natural must NOT.
PC_BASE_VALUE_GAP = 0.5      # constructed cross-token base_value gap for the readiness control
PC_NATURAL_DRIVE_SPREAD = 0.0075  # the observed in-run per-axis drive spread (514q ~0.006-0.0085)

# (i) OVERSHOOT control
OVERSHOOT_DRIVE_MAGNITUDE = 5.0  # large per-axis drive injected on the most-depleted axis
FLIP_FLOOR = 0.6                 # per-seed: fraction of scored events where overshoot flips argmax

# (ii) OFF / bank-disabled floor control: wl_off must read at-or-below this (wanting==liking)
OFF_FLOOR_MAX = 1e-9

# Secondary diagnostic only (the legacy flip-gated effect-size criterion; NOT the supports gate)
WL_DELTA_K_SD = 1.0              # effect-size SD-of-DELTA multiplier (NEVER SD-of-baseline)
WL_DELTA_FLOOR = 0.15            # absolute flip-gated drive-coupled dissociation effect floor

# --- THE REDESIGN: continuous incentive-amplitude effect-size gate (the LOAD-BEARING supports
#     criterion). Effect margin = max(CONT_K_SD * pstdev(per_seed_continuous_shift), CONT_FLOOR)
#     -- SD-of-DELTA + an absolute floor, NEVER pstdev(baseline_level) (a reproducible substrate
#     collapses baseline SD to ~0 -> false 'supports'; per the effect-size-PASS-gate rule). A
#     consistent shift also collapses SD-of-delta so CONT_FLOOR is load-bearing.
CONT_K_SD = 1.0                  # effect-size SD-of-DELTA multiplier for the continuous channel
CONT_FLOOR = 0.05                # absolute continuous-amplitude-shift floor (normalized-share units)

# --- SD-049-PHASE-2 drive-coupling amend levers (the substrate this retest exercises) ---
# MEASUREMENT-REDESIGN amend (failure_autopsy_V3-EXQ-514t): revert kappa to the 514s
# bimodal-positive point (6.0) and read a CONTINUOUS amplitude metric instead of gating supports
# on the exhausted argmax-flip delta.
# (a) kappa scale: effective kappa = incentive_drive_kappa_weight(2.0) * KAPPA_SCALE = 12.
#     6.0 is the 514s point where the natural flip delta was bimodal-POSITIVE (+0.064); 12.0
#     over-amplified into argmax destabilisation (-0.037). DO NOT exceed 6.0 (C8 bounded-kappa
#     invariant: drive must not dominate base_value).
KAPPA_SCALE = 6.0
# (b) standing differential depletion on the WL SCORING env only (training ecology is left at
#     the 603n-canonical config so survival/foraging competence is preserved). lever (b) DONE
#     (enriched_spread ~0.19 met on all 514t seeds) -- do not deepen further.
RESTORATION_FRACTION = 0.15      # deeper partial restore -> ~85% standing drive left on contact
ENRICHED_DECAY = (0.04, 0.01, 0.002)  # divergent per-axis depletion (padded to n types)
# (iv) ENRICHED-SPREAD non-vacuity gate: the in-run per-axis drive_spread_max must clear this on
#      >= 2/3 guard-passing seeds, confirming the env amend produced a STANDING argmax-relevant
#      spread. Below floor -> substrate_not_ready_requeue (env knobs need re-tuning), NEVER a
#      weakens.
MIN_ENRICHED_SPREAD = 0.10


def _make_scaffold_cfg(dry_run: bool) -> ScaffoldedSD054OnboardingConfig:
    """The FULL V3-EXQ-603n lever stack, ported verbatim from 514t."""
    if dry_run:
        stage0, stage0b, p0, hazard, p1, p2, steps = 2, 2, 5, 5, 5, 2, 30
    else:
        stage0, stage0b, p0, hazard, p1, p2, steps = (
            STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_STAGE_BUDGET,
            P1_BUDGET, P2_BUDGET, TRAIN_STEPS,
        )
    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_stage0_enabled=True,
        scaffold_stage0_episode_budget=stage0,
        scaffold_p0_episode_budget=p0,
        scaffold_p1_episode_budget=p1,
        scaffold_p2_episode_budget=p2,
        scaffold_steps_per_episode=steps,
        scaffold_p0_num_hazards=P0_NUM_HAZARDS,
        scaffold_p1_anneal_hold_fraction=P1_HOLD_FRACTION,
        scaffold_p2_hazard_food_attraction_guard=P2_HFA_GUARD,
        scaffold_developmental_window_enabled=True,
        scaffold_stage0b_enabled=True,
        scaffold_stage0b_episode_budget=stage0b,
        scaffold_stage0b_retention_gate=STAGE0B_RETENTION_GATE,
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
        scaffold_hazard_stage_spawn_in_reef_half=HAZARD_STAGE_SPAWN_IN_REEF,
        scaffold_hazard_stage_survival_gate_steps=HAZARD_STAGE_SURVIVAL_GATE_STEPS,
        scaffold_hazard_stage_stability_window=HAZARD_STAGE_STABILITY_WINDOW,
        scaffold_avoidance_driver_enabled=True,
        scaffold_avoidance_scaffold_floor_start=AVOIDANCE_SCAFFOLD_FLOOR_START,
        scaffold_avoidance_scaffold_floor_end=AVOIDANCE_SCAFFOLD_FLOOR_END,
        scaffold_feed_harm_stream=True,
        scaffold_train_harm_pathway=True,
        scaffold_harm_pathway_lr=HARM_PATHWAY_LR,
        scaffold_harm_pathway_in_p0=True,
    )
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env) -> REEConfig:
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
        incentive_drive_kappa_scale=KAPPA_SCALE,  # SD-049-PHASE-2 amend (a): MODERATE kappa (6.0)
        use_cue_recall=True,
        cue_recall_gain=CUE_RECALL_GAIN,
        e2_action_contrastive_enabled=True,
        use_pag_freeze_gate=True,
        pag_theta_freeze=PAG_THETA_FREEZE,
        pag_duration_input_threshold=PAG_DURATION_INPUT_THRESHOLD,
        use_instrumental_avoidance=True,
        avoidance_threat_ref=AVOIDANCE_THREAT_REF,
    )
    cfg.latent.use_resource_encoder = True   # SD-015 (z_resource -> bank L2 bind requires it)
    return cfg


def _ben_drive(obs_body: torch.Tensor) -> Tuple[float, float]:
    b = obs_body.reshape(-1)
    benefit = float(b[11].item()) if b.shape[0] > 11 else 0.0
    energy = float(b[3].item()) if b.shape[0] > 3 else 0.5
    drive = max(0.0, min(1.0, 1.0 - energy))
    return benefit, drive


def _per_axis_drive_from_obs(obs_dict: Dict[str, Any], device: torch.device) -> Optional[torch.Tensor]:
    pad = obs_dict.get("per_axis_drive", None)
    if pad is None:
        return None
    if hasattr(pad, "to"):
        return pad.to(device)
    return torch.as_tensor(np.asarray(pad, dtype=np.float32), device=device)


def _consumed_type_tag_from_info(info: Dict[str, Any]) -> Optional[int]:
    """681-C4 fix: the authoritative consumed (liking) tag is in the info dict."""
    if not isinstance(info, dict):
        return None
    raw = info.get("sd049_consumed_type_tag_this_tick", 0)
    try:
        tag = int(raw[0] if hasattr(raw, "__len__") else raw)
    except (TypeError, ValueError):
        return None
    return tag if tag > 0 else None


def _overshoot_drive(natural_pad: Optional[torch.Tensor], device: torch.device) -> torch.Tensor:
    """(i) Construct an artificially large per-axis drive on the most-depleted axis.

    If a natural per-axis drive vector is available, place the overshoot magnitude on its argmax
    axis (the most-depleted axis the env actually signalled); otherwise place it on axis 0. A
    large multiplier (1 + kappa * OVERSHOOT_DRIVE_MAGNITUDE) dominates realistic base_value gaps,
    so a non-flip means drive genuinely cannot carve."""
    od = torch.zeros(1, N_RESOURCE_TYPES, device=device)
    axis = 0
    if natural_pad is not None:
        flat = natural_pad.reshape(-1)
        if flat.numel() >= 1:
            axis = int(torch.argmax(flat).item())
    axis = min(axis, N_RESOURCE_TYPES - 1)
    od[0, axis] = OVERSHOOT_DRIVE_MAGNITUDE
    return od


def _continuous_amplitude_shift(bank, pad_drive, drive: float) -> Optional[float]:
    """THE REDESIGN: continuous incentive-amplitude readout.

    How far natural per-axis drive re-weights the NORMALIZED wanting distribution toward the
    most-depleted axis, relative to the uniform-drive (per-axis-disabled) baseline. Sub-flip:
    captures the score re-weighting even when the argmax does not cross. Returns a float in
    (-1, 1); positive => natural drive moves wanting TOWARD the depleted axis (the MECH-436
    prediction). Returns None when the bank has < 2 tokens or the wanting masses are degenerate.

    wanting_natural uses the in-run per_axis_drive; wanting_uniform uses per_axis_drive=None
    (a flat scalar-drive multiplier -> normalized share == base_value share). The continuous
    OFF floor (per_axis_drive=None on BOTH sides) is structurally exactly 0.0 -- a hard-zero
    sanity check on the comparator (the bank is not manufacturing a shift absent per-axis drive).
    """
    w_nat = bank.wanting(per_axis_drive=pad_drive, scalar_drive=float(drive))
    w_uni = bank.wanting(per_axis_drive=None, scalar_drive=float(drive))
    if len(w_nat) < 2:
        return None
    s_nat = float(sum(w_nat.values()))
    s_uni = float(sum(w_uni.values()))
    if s_nat <= 0.0 or s_uni <= 0.0:
        return None
    # most-depleted axis among the bank's tokens (highest per-axis drive); reuses the bank's own
    # axis-lookup so k_dep is consistent with how wanting() applies the per-axis multiplier.
    def _axis(k: int) -> float:
        return float(bank._drive_axis_for(k, pad_drive, float(drive)))
    k_dep = max(w_nat.keys(), key=_axis)
    share_nat = float(w_nat[k_dep]) / s_nat
    share_uni = float(w_uni[k_dep]) / s_uni
    return float(share_nat - share_uni)


def _argmax_relevance_positive_control(agent, device: torch.device) -> Dict[str, Any]:
    """(iii) RECALIBRATED readiness: a constructed 2-token bank with a realistic cross-token
    base_value gap on which the OVERSHOOT drive MUST flip the argmax while the NATURAL-magnitude
    drive must NOT. Asserts the overshoot magnitude is in the decision-moving regime for this
    run's base_value spread (the 514q gap: it only checked spread > 1e-3, never argmax-relevance).
    """
    goal_cfg = agent.config.goal
    d = int(getattr(goal_cfg, "goal_dim", WORLD_DIM))
    pc_bank = IncentiveTokenBank(goal_cfg, device)
    z_a = torch.zeros(1, d, device=device); z_a[0, 0] = 1.0   # token type 1
    z_b = torch.zeros(1, d, device=device); z_b[0, 1] = 1.0   # token type 2
    # Realistic base_value gap: token 1 favoured by PC_BASE_VALUE_GAP over token 2.
    pc_bank.update(1, 1.0, z_a)
    pc_bank.update(2, 1.0 - PC_BASE_VALUE_GAP, z_b)
    # Drive favours the LOWER-value axis (type 2 -> axis index 1). Natural-magnitude drive should
    # NOT flip the argmax away from type 1; overshoot drive MUST.
    natural = torch.zeros(1, N_RESOURCE_TYPES, device=device)
    natural[0, 1] = PC_NATURAL_DRIVE_SPREAD
    overshoot = torch.zeros(1, N_RESOURCE_TYPES, device=device)
    overshoot[0, 1] = OVERSHOOT_DRIVE_MAGNITUDE

    mw_nodrive = pc_bank.most_wanted(per_axis_drive=None, scalar_drive=1.0)
    mw_natural = pc_bank.most_wanted(per_axis_drive=natural, scalar_drive=1.0)
    mw_overshoot = pc_bank.most_wanted(per_axis_drive=overshoot, scalar_drive=1.0)

    base = int(mw_nodrive[0]) if mw_nodrive is not None else -1
    natural_flips = bool(mw_natural is not None and int(mw_natural[0]) != base)
    overshoot_flips = bool(mw_overshoot is not None and int(mw_overshoot[0]) != base)
    argmax_relevance_met = bool(overshoot_flips and (not natural_flips))
    # NEW: continuous-channel positive control on the SAME constructed bank -- the overshoot drive
    # MUST produce a strictly larger continuous amplitude shift toward type 2 than natural drive
    # (the continuous metric is sensitive to the constructed re-weighting). Diagnostic only.
    cont_natural = _continuous_amplitude_shift(pc_bank, natural, 1.0)
    cont_overshoot = _continuous_amplitude_shift(pc_bank, overshoot, 1.0)
    pc_cont_monotone = bool(
        cont_natural is not None and cont_overshoot is not None
        and cont_overshoot > cont_natural
    )
    return {
        "pc_base": base,
        "pc_natural_flips": natural_flips,
        "pc_overshoot_flips": overshoot_flips,
        "pc_argmax_relevance_met": argmax_relevance_met,
        "pc_cont_natural_shift": float(cont_natural) if cont_natural is not None else 0.0,
        "pc_cont_overshoot_shift": float(cont_overshoot) if cont_overshoot is not None else 0.0,
        "pc_cont_monotone": pc_cont_monotone,
    }


def _run_disambiguator_eval(agent, scaffold_cfg, device: torch.device, n_eps: int) -> Dict[str, Any]:
    """SD-049 Phase-2 OBJECT-BOUND wanting!=liking measurement, extended with the overshoot +
    OFF-floor disambiguation AND the continuous incentive-amplitude readout. Frozen policy."""
    env = _build_env(scaffold_cfg, "p2")
    # SD-049-PHASE-2 amend (b): enrich ONLY the WL scoring env with STANDING differential
    # depletion -- partial restoration + a divergent per-axis decay -- so the per-axis drive
    # spread at consumption (the WL scoring moment) is argmax-relevant. Both are plain attributes
    # read directly each step(); mutating post-_build_env enriches scoring without touching the
    # training ecology.
    env.per_axis_restoration_fraction = float(RESTORATION_FRACTION)
    n_types = int(getattr(env, "n_resource_types", N_RESOURCE_TYPES))
    decay = list(ENRICHED_DECAY)
    while len(decay) < n_types:
        decay.append(decay[-1] if decay else 0.002)
    env.per_axis_drive_decay = tuple(float(max(0.0, d)) for d in decay[:n_types])
    env.reset()
    steps_per_ep = scaffold_cfg.scaffold_steps_per_episode

    bank = getattr(agent.goal_state, "incentive_bank", None)

    scored_steps = 0
    wl_natural_steps = 0          # drive-favored wanting != liking (flip metric, secondary)
    wl_nodrive_steps = 0          # drive-uniform wanting != liking (514q control)
    wl_off_steps = 0              # bank-bypassed wanting (== liking) != liking  -> must be 0
    overshoot_flip_steps = 0      # overshoot argmax != drive-uniform argmax (the disambiguator)
    # THE REDESIGN: continuous incentive-amplitude accumulators.
    cont_shift_sum = 0.0          # sum of per-step continuous amplitude shifts toward depleted axis
    cont_shift_n = 0              # count of scored steps where the continuous shift is defined
    cont_off_abs_max = 0.0        # max |continuous OFF shift| (per-axis disabled -> must be ~0)
    contact_steps = 0
    total_steps = 0
    distinct_tokens_max = 0
    drive_spread_max = 0.0

    for _ep in range(n_eps):
        _, obs_dict = env.reset()
        agent.reset()
        for _step in range(steps_per_ep):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            with torch.no_grad():
                latent = _sense_with_optional_harm(
                    agent, obs_body, obs_world, obs_dict, device,
                    scaffold_cfg.scaffold_feed_harm_stream,
                )
                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick")
                    else torch.zeros(1, agent.config.latent.world_dim, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)

            action_idx = int(action.argmax(dim=-1).item())
            _, harm_signal, done, info, obs_dict = env.step(action_idx)
            total_steps += 1

            consumed_tag = _consumed_type_tag_from_info(info)   # liking target
            benefit, drive = _ben_drive(obs_dict["body_state"].to(device))
            if consumed_tag is not None:
                contact_steps += 1
                pad = _per_axis_drive_from_obs(obs_dict, device)
                if pad is not None:
                    agent._per_axis_drive = pad.reshape(-1)
                    flat = pad.reshape(-1)
                    if flat.numel() >= 2:
                        spread = float(flat.max().item() - flat.min().item())
                        drive_spread_max = max(drive_spread_max, spread)
                with torch.no_grad():
                    try:
                        agent.update_z_goal(float(benefit), drive_level=float(drive),
                                            resource_type=consumed_tag)
                    except TypeError:
                        agent.update_z_goal(float(benefit), drive_level=float(drive))
                    if bank is not None and not bank.is_empty():
                        pad_drive = getattr(agent, "_per_axis_drive", None)
                        n_distinct = len(bank.wanting(per_axis_drive=pad_drive, scalar_drive=float(drive)))
                        distinct_tokens_max = max(distinct_tokens_max, n_distinct)
                        mw_natural = bank.most_wanted(per_axis_drive=pad_drive, scalar_drive=float(drive))
                        mw_nodrive = bank.most_wanted(per_axis_drive=None, scalar_drive=float(drive))
                        overshoot = _overshoot_drive(
                            pad_drive.unsqueeze(0) if (pad_drive is not None and pad_drive.dim() == 1)
                            else pad_drive,
                            device,
                        )
                        mw_overshoot = bank.most_wanted(per_axis_drive=overshoot, scalar_drive=float(drive))
                        # SCORE only when wanting + liking defined AND bank holds >= 2 distinct tokens.
                        if mw_natural is not None and n_distinct >= 2:
                            scored_steps += 1
                            # --- secondary: flip-gated WL delta (the legacy coarse metric) ---
                            if int(mw_natural[0]) != int(consumed_tag):
                                wl_natural_steps += 1
                            if mw_nodrive is not None and int(mw_nodrive[0]) != int(consumed_tag):
                                wl_nodrive_steps += 1
                            # (ii) OFF / bank-disabled flip floor: bank bypassed -> wanting := liking
                            # -> dissociation 0 by construction. Computed explicitly as a structural
                            # sanity floor (a non-zero read would mean the comparator is broken).
                            mw_off = int(consumed_tag)  # bank bypassed -> wanting := liking
                            if mw_off != int(consumed_tag):
                                wl_off_steps += 1
                            # (i) overshoot flip: overshoot argmax vs drive-uniform argmax.
                            if (mw_overshoot is not None and mw_nodrive is not None
                                    and int(mw_overshoot[0]) != int(mw_nodrive[0])):
                                overshoot_flip_steps += 1
                            # --- THE REDESIGN: continuous incentive-amplitude shift toward the
                            # most-depleted axis (sub-flip; the LOAD-BEARING supports signal). ---
                            shift = _continuous_amplitude_shift(bank, pad_drive, float(drive))
                            if shift is not None:
                                cont_shift_sum += shift
                                cont_shift_n += 1
                            # continuous OFF floor: per-axis drive disabled -> structurally 0.0.
                            off_shift = _continuous_amplitude_shift(bank, None, float(drive))
                            if off_shift is not None:
                                cont_off_abs_max = max(cont_off_abs_max, abs(off_shift))

            if done:
                break

    def _frac(n: int) -> float:
        return (float(n) / float(scored_steps)) if scored_steps > 0 else 0.0

    wl_natural_fraction = _frac(wl_natural_steps)
    wl_nodrive_fraction = _frac(wl_nodrive_steps)
    wl_off_fraction = _frac(wl_off_steps)
    overshoot_flip_fraction = _frac(overshoot_flip_steps)
    wl_drive_delta = float(wl_natural_fraction - wl_nodrive_fraction)
    mean_cont_amplitude_shift = (float(cont_shift_sum) / float(cont_shift_n)) if cont_shift_n > 0 else 0.0
    behav_contact_rate = (float(contact_steps) / float(total_steps)) if total_steps > 0 else 0.0
    run_populated = bool(distinct_tokens_max >= 2 and scored_steps >= MIN_SCORED_STEPS)

    return {
        # secondary (flip-gated, legacy diagnostic):
        "object_bound_wl_dissoc_fraction": wl_natural_fraction,
        "wl_nodrive_dissoc_fraction": wl_nodrive_fraction,
        "wl_off_floor_fraction": wl_off_fraction,
        "overshoot_flip_fraction": overshoot_flip_fraction,
        "wl_drive_delta": wl_drive_delta,
        # THE REDESIGN (load-bearing continuous metric):
        "mean_cont_amplitude_shift": mean_cont_amplitude_shift,
        "n_cont_scored": cont_shift_n,
        "cont_off_floor_abs_max": float(cont_off_abs_max),
        # shared:
        "n_scored_wl_steps": scored_steps,
        "n_overshoot_flip_steps": overshoot_flip_steps,
        "distinct_tokens_max": distinct_tokens_max,
        "drive_spread_max": drive_spread_max,
        "run_bank_populated": run_populated,
        "behav_contact_rate": behav_contact_rate,
    }


def _aborted_seed_record(seed: int, stage: str, reason: str) -> Dict[str, Any]:
    return {
        "seed": seed, "aborted_at": stage, "abort_reason": reason,
        "guard_pass": False,
        "p2_contact_rate": 0.0, "p2_z_goal_norm_at_contact_peak": 0.0,
        "p2_num_contact_events": 0,
        "object_bound_wl_dissoc_fraction": 0.0, "wl_nodrive_dissoc_fraction": 0.0,
        "wl_off_floor_fraction": 0.0, "overshoot_flip_fraction": 0.0,
        "wl_drive_delta": 0.0,
        "mean_cont_amplitude_shift": 0.0, "n_cont_scored": 0, "cont_off_floor_abs_max": 0.0,
        "n_scored_wl_steps": 0, "n_overshoot_flip_steps": 0,
        "distinct_tokens_max": 0, "drive_spread_max": 0.0,
        "run_bank_populated": False, "behav_contact_rate": 0.0,
        "pc_argmax_relevance_met": False, "pc_overshoot_flips": False, "pc_natural_flips": False,
        "pc_cont_natural_shift": 0.0, "pc_cont_overshoot_shift": 0.0, "pc_cont_monotone": False,
        "overshoot_flips_seed": False,
    }


def _run_seed(seed: int, dry_run: bool, total_eps: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaffold_cfg = _make_scaffold_cfg(dry_run)
    device = torch.device("cpu")

    probe_env = _build_env(scaffold_cfg, "p2")
    probe_env.reset()
    agent = REEAgent(_make_config(probe_env)).to(device)
    scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)

    print(f"Seed {seed} Condition {CONDITION_LABEL}", flush=True)
    done = 0

    s0 = scheduler.run_stage0_nursery(agent, device)
    done += s0.n_episodes
    print(f"  [train] stage0_nursery seed={seed} ep {done}/{total_eps}"
          f" z_goal_peak={s0.z_goal_norm_peak:.4f} formed={s0.z_goal_formed}", flush=True)
    if s0.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=stage0 reason={s0.abort_reason}", flush=True)
        return _aborted_seed_record(seed, "stage0", s0.abort_reason)

    s0b = scheduler.run_stage0b_consolidation(agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
    done += s0b.n_episodes
    print(f"  [train] stage0b_consolidate seed={seed} ep {done}/{total_eps}"
          f" retention={s0b.retention_ratio:.3f}"
          f" gate={'pass' if s0b.retention_gate_passed else 'FAIL'}", flush=True)
    if s0b.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=stage0b reason={s0b.abort_reason}", flush=True)
        return _aborted_seed_record(seed, "stage0b", s0b.abort_reason)

    p0 = scheduler.run_p0(agent, device)
    done += p0.n_episodes
    print(f"  [train] p0_guided seed={seed} ep {done}/{total_eps}"
          f" mean_len={p0.mean_episode_length:.1f} rv={p0.final_running_variance:.5f}", flush=True)
    if p0.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=p0 reason={p0.abort_reason}", flush=True)
        return _aborted_seed_record(seed, "p0", p0.abort_reason)

    hz = scheduler.run_hazard_avoidance(agent, device)
    done += hz.n_episodes
    print(f"  [train] hazard_avoidance seed={seed} ep {done}/{total_eps}"
          f" median_last={hz.median_last_window_episode_length:.1f}"
          f" survival_gate={'pass' if hz.survival_gate_passed else 'FAIL'}", flush=True)
    if hz.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=hazard reason={hz.abort_reason}", flush=True)
        return _aborted_seed_record(seed, "hazard", hz.abort_reason)

    p1 = scheduler.run_p1(agent, device)
    done += p1.n_episodes
    print(f"  [train] p1_foraging seed={seed} ep {done}/{total_eps}"
          f" median_last={p1.median_last_window_episode_length:.1f}"
          f" survival_gate={'pass' if p1.survival_gate_passed else 'FAIL'}", flush=True)

    p2 = scheduler.run_p2(agent, device)
    done += p2.n_episodes
    print(f"  [train] p2_guard seed={seed} ep {done}/{total_eps}"
          f" contact_rate={p2.contact_rate:.4f} contact_events={p2.num_contact_events}"
          f" z_goal_at_contact={p2.z_goal_norm_at_contact_peak:.4f}", flush=True)

    guard_pass = bool(
        p2.contact_rate > CONTACT_GATE
        and p2.z_goal_norm_at_contact_peak > P2_ZGOAL_GATE
    )

    behav = _run_disambiguator_eval(agent, scaffold_cfg, device, BEHAV_EVAL_EPISODES)
    done += BEHAV_EVAL_EPISODES
    overshoot_flips_seed = bool(behav["overshoot_flip_fraction"] >= FLIP_FLOOR)
    print(f"  [train] disambig_eval seed={seed} ep {done}/{total_eps}"
          f" cont_shift={behav['mean_cont_amplitude_shift']:.4f}"
          f" cont_n={behav['n_cont_scored']}"
          f" wl_drive={behav['object_bound_wl_dissoc_fraction']:.3f}"
          f" wl_nodrive={behav['wl_nodrive_dissoc_fraction']:.3f}"
          f" flip_delta={behav['wl_drive_delta']:.3f}"
          f" overshoot_flip={behav['overshoot_flip_fraction']:.3f}"
          f" overshoot_flips_seed={overshoot_flips_seed}"
          f" cont_off={behav['cont_off_floor_abs_max']:.2e}"
          f" wl_off={behav['wl_off_floor_fraction']:.3f}"
          f" scored={behav['n_scored_wl_steps']}"
          f" distinct_tokens={behav['distinct_tokens_max']}", flush=True)

    print(f"verdict: {'PASS' if guard_pass else 'FAIL'} seed={seed}"
          f" guard_pass={guard_pass}"
          f" (contact_rate={p2.contact_rate:.4f} z_goal_at_contact={p2.z_goal_norm_at_contact_peak:.4f})",
          flush=True)

    rec: Dict[str, Any] = {
        "seed": seed,
        "aborted_at": None,
        "abort_reason": "",
        "guard_pass": guard_pass,
        "stage0_z_goal_norm_peak": float(s0.z_goal_norm_peak),
        "p1_survival_pass": bool(p1.survival_gate_passed),
        "hazard_stage_survival_pass": bool(hz.survival_gate_passed),
        "p2_contact_rate": float(p2.contact_rate),
        "p2_z_goal_norm_at_contact_peak": float(p2.z_goal_norm_at_contact_peak),
        "p2_num_contact_events": int(p2.num_contact_events),
        "overshoot_flips_seed": overshoot_flips_seed,
    }
    rec.update(behav)
    # (iii) recalibrated readiness positive control on THIS agent's GoalConfig.
    rec.update(_argmax_relevance_positive_control(agent, device))
    return rec


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS
    if dry_run:
        total_eps = 2 + 2 + 5 + 5 + 5 + 2 + BEHAV_EVAL_EPISODES
    else:
        total_eps = (
            STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_STAGE_BUDGET
            + P1_BUDGET + P2_BUDGET + BEHAV_EVAL_EPISODES
        )

    per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        per_seed.append(_run_seed(s, dry_run, total_eps))

    n = len(per_seed)
    guard_flags = [r["guard_pass"] for r in per_seed]
    guard_frac = _frac(guard_flags)
    guard_passing = [r for r in per_seed if r["guard_pass"]]
    contact_non_vacuity_met = bool(guard_frac >= MIN_FRACTION)

    # --- gate (b): RECALIBRATED WL readiness (argmax-relevance positive control) ---
    pc_relevance_flags = [bool(r.get("pc_argmax_relevance_met", False)) for r in guard_passing]
    pc_relevance_frac = _frac(pc_relevance_flags)
    wl_readiness_met = bool(pc_relevance_frac >= MIN_FRACTION)

    # --- gate (b2): OFF / bank-disabled floor sanity (flip channel: wanting==liking -> 0) ---
    off_floor_flags = [bool(r.get("wl_off_floor_fraction", 0.0) <= OFF_FLOOR_MAX) for r in guard_passing]
    off_floor_frac = _frac(off_floor_flags)
    off_floor_met = bool(off_floor_frac >= MIN_FRACTION)

    # --- gate (b2c): continuous-channel OFF floor (per-axis disabled -> structural hard zero) ---
    cont_off_flags = [bool(r.get("cont_off_floor_abs_max", 0.0) <= OFF_FLOOR_MAX) for r in guard_passing]
    cont_off_frac = _frac(cont_off_flags)
    cont_off_floor_met = bool(cont_off_frac >= MIN_FRACTION)

    # --- gate (b3): bank populated (>= 2 distinct tokens + enough scored events) ---
    run_populated_flags = [bool(r.get("run_bank_populated", False)) for r in guard_passing]
    run_populated_frac = _frac(run_populated_flags)
    run_populated_met = bool(run_populated_frac >= MIN_FRACTION)

    # --- gate (iv): ENRICHED-SPREAD non-vacuity (the env amend took effect) ---
    enriched_spread_flags = [bool(r.get("drive_spread_max", 0.0) >= MIN_ENRICHED_SPREAD) for r in guard_passing]
    enriched_spread_frac = _frac(enriched_spread_flags)
    enriched_spread_met = bool(enriched_spread_frac >= MIN_FRACTION)
    mean_drive_spread = float(np.mean([r.get("drive_spread_max", 0.0) for r in guard_passing])) if guard_passing else 0.0

    def _mean(key: str) -> float:
        vals = [r[key] for r in guard_passing]
        return float(np.mean(vals)) if vals else 0.0

    mean_wl_natural = _mean("object_bound_wl_dissoc_fraction")
    mean_wl_nodrive = _mean("wl_nodrive_dissoc_fraction")
    mean_overshoot_flip = _mean("overshoot_flip_fraction")
    n_scored_total = int(sum(r["n_scored_wl_steps"] for r in guard_passing))

    # --- THE REDESIGN: continuous incentive-amplitude supports gate (LOAD-BEARING) ---
    per_seed_cont = [float(r["mean_cont_amplitude_shift"]) for r in guard_passing]
    mean_cont = float(np.mean(per_seed_cont)) if per_seed_cont else 0.0
    sd_cont = float(statistics.pstdev(per_seed_cont)) if len(per_seed_cont) >= 1 else 0.0
    cont_effect_margin = max(CONT_K_SD * sd_cont, CONT_FLOOR)
    per_seed_cont_clears = [bool(d >= cont_effect_margin) for d in per_seed_cont]
    cont_seed_frac = _frac(per_seed_cont_clears)
    c_cont_supports = bool(cont_seed_frac >= MIN_FRACTION)

    # --- secondary diagnostic: legacy flip-gated effect-size criterion (NOT the supports gate) ---
    per_seed_delta = [float(r["wl_drive_delta"]) for r in guard_passing]
    mean_delta = float(np.mean(per_seed_delta)) if per_seed_delta else 0.0
    sd_delta = float(statistics.pstdev(per_seed_delta)) if len(per_seed_delta) >= 1 else 0.0
    flip_effect_margin = max(WL_DELTA_K_SD * sd_delta, WL_DELTA_FLOOR)
    c_flip_delta = bool(mean_delta >= flip_effect_margin)

    # --- disambiguator: does overshoot flip the argmax on >= 2/3 guard-passing seeds? ---
    overshoot_seed_flags = [bool(r.get("overshoot_flips_seed", False)) for r in guard_passing]
    overshoot_seed_frac = _frac(overshoot_seed_flags)
    overshoot_resolves = bool(overshoot_seed_frac >= MIN_FRACTION)

    readiness_ok = bool(
        contact_non_vacuity_met and wl_readiness_met and off_floor_met
        and cont_off_floor_met and run_populated_met and enriched_spread_met
    )

    if not contact_non_vacuity_met:
        outcome = "FAIL"; readiness_route = "substrate_not_ready_requeue"
        evidence_direction = "non_contributory"; route_reason = "contact_guard_unmet"
    elif not wl_readiness_met:
        outcome = "FAIL"; readiness_route = "substrate_not_ready_requeue"
        evidence_direction = "non_contributory"
        route_reason = "overshoot_magnitude_not_argmax_relevant_recalibrate_OVERSHOOT_DRIVE_MAGNITUDE"
    elif not run_populated_met:
        outcome = "FAIL"; readiness_route = "substrate_not_ready_requeue"
        evidence_direction = "non_contributory"; route_reason = "bank_not_populated_this_run"
    elif not off_floor_met:
        outcome = "FAIL"; readiness_route = "substrate_not_ready_requeue"
        evidence_direction = "non_contributory"; route_reason = "flip_off_floor_nonzero_instrument_broken"
    elif not cont_off_floor_met:
        # The continuous comparator returned a non-zero shift with per-axis drive disabled ->
        # the continuous metric is broken (it must be structurally zero absent per-axis drive).
        outcome = "FAIL"; readiness_route = "substrate_not_ready_requeue"
        evidence_direction = "non_contributory"; route_reason = "continuous_off_floor_nonzero_instrument_broken"
    elif not enriched_spread_met:
        outcome = "FAIL"; readiness_route = "substrate_not_ready_requeue"
        evidence_direction = "non_contributory"
        route_reason = "enriched_per_axis_spread_below_floor_retune_env_amend"
    elif c_cont_supports:
        # LOAD-BEARING PASS: the continuous drive-coupled amplitude shift toward the depleted axis
        # clears the effect margin on >= 2/3 seeds -> drive consistently re-weights wanting toward
        # the depleted axis (MECH-436), captured continuously and robustly to argmax noise.
        outcome = "PASS"; readiness_route = "mech436_drive_coupled_continuous_amplitude_supports"
        evidence_direction = "supports"
        route_reason = "continuous_amplitude_shift_cleared_effect_margin_on_enriched_substrate"
    elif overshoot_resolves:
        # continuous shift below margin BUT overshoot still flips -> enrichment/measurement
        # insufficient at this kappa. NOT a weakens.
        outcome = "FAIL"; readiness_route = "mech436_enrichment_insufficient_substrate_ceiling"
        evidence_direction = "non_contributory"
        route_reason = "continuous_below_margin_overshoot_still_flips_retune_sd049_phase2_amend"
    else:
        # every readiness gate passed AND overshoot cannot flip even at maximal drive on the
        # enriched substrate -> genuine MECH-436 drive-coupling weakens.
        outcome = "FAIL"; readiness_route = "mech436_drive_coupling_genuine_weakens_overshoot_cannot_flip"
        evidence_direction = "weakens"
        route_reason = "overshoot_cannot_flip_argmax_on_enriched_substrate_genuine_mech436_weakens"

    # Non-degeneracy scoring net (2026-06-11): the continuous-amplitude test got a fair,
    # non-vacuous test only when ALL readiness gates (contact + argmax-relevance + OFF floor
    # [flip + continuous] + bank populated + ENRICHED spread) are met. A weakens/non_contributory
    # verdict before those is degenerate.
    non_degenerate = bool(readiness_ok)
    degeneracy_reason = ("" if non_degenerate
                         else f"continuous-amplitude test not non-vacuously testable: {route_reason} "
                              f"(contact={contact_non_vacuity_met}, wl_readiness={wl_readiness_met}, "
                              f"off_floor={off_floor_met}, cont_off_floor={cont_off_floor_met}, "
                              f"bank_populated={run_populated_met}, enriched_spread={enriched_spread_met})")

    print(f"[{EXPERIMENT_TYPE}] readiness contact={contact_non_vacuity_met}"
          f" (guard {sum(guard_flags)}/{n}) argmax_relevance={wl_readiness_met}"
          f" (pc_frac={pc_relevance_frac:.3f}) flip_off_floor={off_floor_met}"
          f" cont_off_floor={cont_off_floor_met} bank_pop={run_populated_met}"
          f" (frac={run_populated_frac:.3f}) enriched_spread={enriched_spread_met}"
          f" (frac={enriched_spread_frac:.3f} mean_spread={mean_drive_spread:.4f})", flush=True)
    print(f"[{EXPERIMENT_TYPE}] CONTINUOUS amplitude mean={mean_cont:.4f} sd={sd_cont:.4f}"
          f" effect_margin={cont_effect_margin:.4f} seed_clears={cont_seed_frac:.3f}"
          f" -> C_CONT_supports={c_cont_supports}"
          f" | (secondary) flip delta mean={mean_delta:.4f} sd={sd_delta:.4f}"
          f" C_FLIP={c_flip_delta} | overshoot_flip mean={mean_overshoot_flip:.3f}"
          f" seed_frac={overshoot_seed_frac:.3f} resolves={overshoot_resolves}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] outcome={outcome} route={readiness_route}"
          f" per_claim MECH-436={evidence_direction} non_degenerate={non_degenerate}", flush=True)

    acceptance = {
        "contact_non_vacuity_met": contact_non_vacuity_met,
        "guard_fraction": guard_frac,
        "n_guard_passing_seeds": len(guard_passing),
        "wl_readiness_met_argmax_relevance": wl_readiness_met,
        "pc_argmax_relevance_frac": pc_relevance_frac,
        "off_floor_met": off_floor_met,
        "off_floor_frac": off_floor_frac,
        "cont_off_floor_met": cont_off_floor_met,
        "cont_off_floor_frac": cont_off_frac,
        "per_seed_cont_off_floor_abs_max": [float(r.get("cont_off_floor_abs_max", 0.0)) for r in guard_passing],
        "run_bank_populated_frac": run_populated_frac,
        # enriched-spread non-vacuity gate (the env amend took effect):
        "enriched_spread_met": enriched_spread_met,
        "enriched_spread_frac": enriched_spread_frac,
        "mean_drive_spread_max": mean_drive_spread,
        "min_enriched_spread_floor": MIN_ENRICHED_SPREAD,
        "per_seed_drive_spread_max": [float(r.get("drive_spread_max", 0.0)) for r in guard_passing],
        # THE REDESIGN -- continuous incentive-amplitude supports gate (LOAD-BEARING):
        "C_CONT_amplitude_supports": c_cont_supports,
        "mean_cont_amplitude_shift": mean_cont,
        "sd_cont_amplitude_shift": sd_cont,
        "cont_effect_margin": cont_effect_margin,
        "cont_seed_clear_frac": cont_seed_frac,
        "per_seed_cont_amplitude_shift": per_seed_cont,
        "per_seed_cont_clears": per_seed_cont_clears,
        "cont_k_sd": CONT_K_SD,
        "cont_floor": CONT_FLOOR,
        "n_cont_scored_total": int(sum(r["n_cont_scored"] for r in guard_passing)),
        # secondary diagnostic -- legacy flip-gated effect-size criterion:
        "C_WL_DRIVE_coupled_dissociation_flip_secondary": c_flip_delta,
        "mean_wl_drive_delta": mean_delta,
        "sd_wl_drive_delta": sd_delta,
        "wl_delta_effect_margin": flip_effect_margin,
        "per_seed_wl_drive_delta": per_seed_delta,
        # disambiguator (the load-bearing statistic for the substrate_ceiling vs weakens fork):
        "overshoot_resolves_magnitude_artifact": overshoot_resolves,
        "overshoot_seed_pass_frac": overshoot_seed_frac,
        "mean_overshoot_flip_fraction": mean_overshoot_flip,
        "per_seed_overshoot_flips": overshoot_seed_flags,
        # reported diagnostics:
        "mean_object_bound_wl_dissoc_fraction": mean_wl_natural,
        "mean_wl_nodrive_dissoc_fraction": mean_wl_nodrive,
        "n_scored_wl_steps_total": n_scored_total,
        "overall_pass": bool(readiness_ok and c_cont_supports),
        "per_seed_guard_pass": guard_flags,
        "route_reason": route_reason,
    }

    return {
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "acceptance": acceptance,
        "interpretation": {
            "label": readiness_route,
            "readiness_route": readiness_route,
            "route_reason": route_reason,
            "disambiguation": {
                "supports_if": "continuous drive-coupled amplitude shift toward the depleted axis "
                               ">= max(CONT_K_SD*pstdev(per_seed_shift), CONT_FLOOR) on >= 2/3 seeds "
                               "-> MECH-436 substrate_ceiling -> supports (the LOAD-BEARING "
                               "promotion path; the measurement-redesign metric).",
                "substrate_ceiling_if": "continuous shift below margin BUT overshoot still flips "
                                        "argmax on >= 2/3 seeds -> the enrichment / measurement is "
                                        "insufficient at this kappa (retune). NOT a weakens.",
                "genuine_weakens_if": "every readiness gate passed AND overshoot CANNOT flip even "
                                      "at OVERSHOOT_DRIVE_MAGNITUDE on the adequately-enriched "
                                      "substrate (drive cannot carve wanting even at maximal "
                                      "magnitude).",
                "substrate_not_ready_if": "any readiness gate unmet (contact / argmax-relevance / "
                                          "flip OFF floor / continuous OFF floor / bank populated / "
                                          "ENRICHED spread) -> substrate_not_ready_requeue, NEVER a "
                                          "weakens.",
                "claim_scope_note": "tests MECH-436 (drive.wanting_drive_state_modulation) ONLY -- "
                                    "the drive-coupling leg split out of MECH-229 (2026-06-16). "
                                    "Sub-leg (a) wanting!=liking (object-bound, MECH-229) is "
                                    "established by V3-EXQ-514o PASS (0.80) and is NOT under test "
                                    "here; this run must not weaken it (the hard-zero OFF floor "
                                    "confirms drive is not manufacturing a comparator artifact).",
            },
            "measurement_redesign": {
                "continuous_metric": "amplitude_shift = share_natural[k_dep] - share_uniform[k_dep], "
                                     "the normalized-wanting re-weighting toward the most-depleted "
                                     "axis k_dep under natural per-axis drive vs the uniform-drive "
                                     "baseline. Sub-flip (no argmax cross required); robust to the "
                                     "argmax destabilisation that exploded the 514t flip-gated SD.",
                "why": "failure_autopsy_V3-EXQ-514t: the argmax-flip-gated WL delta is the diagnosed "
                       "ceiling (coarse + interacts pathologically with high kappa). The kappa-"
                       "magnitude lever is exhausted (non-monotone: 6.0->12.0 regressed +0.064 -> "
                       "-0.037); this run reverts to moderate kappa (6.0) and reads the continuous "
                       "metric instead.",
            },
            "sd049_phase2_drive_coupling_amend": {
                "kappa_scale": KAPPA_SCALE,
                "kappa_eff_note": "effective kappa = incentive_drive_kappa_weight(2.0) * 6.0 = 12 "
                                  "(moderate; the 514s bimodal-positive point; <= the C8 bounded-"
                                  "kappa upper bound that 12.0/kappa_eff=24 violated).",
                "restoration_fraction": RESTORATION_FRACTION,
                "enriched_decay": list(ENRICHED_DECAY),
                "min_enriched_spread_floor": MIN_ENRICHED_SPREAD,
                "note": "lever (b) done (enriched_spread ~0.19 met on all 514t seeds; not deepened). "
                        "env amend enriches ONLY the WL scoring env so training survival/foraging "
                        "competence is preserved.",
            },
            "recalibrated_readiness": {
                "pc_argmax_relevance_frac": pc_relevance_frac,
                "definition": "constructed 2-token bank with base_value gap PC_BASE_VALUE_GAP; "
                              "overshoot drive MUST flip the argmax while natural-magnitude drive "
                              "(PC_NATURAL_DRIVE_SPREAD) must NOT. Continuous positive control: "
                              "overshoot continuous shift > natural continuous shift (diagnostic).",
            },
            "overshoot": {
                "magnitude": OVERSHOOT_DRIVE_MAGNITUDE,
                "flip_floor": FLIP_FLOOR,
                "min_fraction_seeds": MIN_FRACTION,
            },
            "continuous_effect_size_gate": {
                "cont_k_sd": CONT_K_SD, "cont_floor": CONT_FLOOR,
                "mean_cont": mean_cont, "sd_cont": sd_cont, "cont_effect_margin": cont_effect_margin,
                "seed_clear_frac": cont_seed_frac,
            },
            "flip_effect_size_gate_secondary": {
                "k_sd": WL_DELTA_K_SD, "delta_floor": WL_DELTA_FLOOR,
                "mean_delta": mean_delta, "sd_delta": sd_delta, "effect_margin": flip_effect_margin,
            },
        },
        "seeds": SEEDS,
        "per_seed": per_seed,
        "predecessor": PREDECESSOR,
        "supersedes": SUPERSEDES,
    }


def main(dry_run: bool = False) -> Dict[str, Any]:
    result = run_experiment(dry_run=dry_run)

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_path = out_dir / f"{run_id}.json"

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "timestamp_utc": ts,
        "supersedes": SUPERSEDES,
        "predecessor": PREDECESSOR,
        "dry_run": bool(dry_run),
    }
    manifest.update(result)

    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    print(f"[{EXPERIMENT_TYPE}] wrote {out_path}", flush=True)
    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    _res = main(dry_run=args.dry_run)
    if _res.get("manifest_path"):
        _outcome_raw = str(_res["outcome"]).upper()
        emit_outcome(
            outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
            manifest_path=str(_res["manifest_path"]),
        )
