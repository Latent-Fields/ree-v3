"""
V3-EXQ-1067 (MECH-266 / SD-032a): squash-vs-clamp affinity BOUNDING-OPERATOR cap
sweep for external_task mode-occupancy. DIAGNOSTIC.

!!! BLOCKED -- DO NOT QUEUE AS IT STANDS (2026-09-19) !!!
RED-TEAM (/queue-experiment Step 4.5, model fable): **BLOCKING**, confirmed
against source. This script was authored, passed validate_experiments --strict
(38/38) and every pre-flight gate, and was then REFUSED before queueing. It is
committed for resumability only.

THE DEFECT, in one paragraph: the bounding operator is applied to EVERY
affinity_weights signal (salience_coordinator.py:616-636), and
`external_task_drive` is one of them (agent.py:2760-2765, weight 3.0). At
sigma = cap the squash is NOT an identity on sub-cap signals -- it returns
0.800x at 0.25*cap, 0.667x at 0.5*cap and 0.500x AT the cap. Since the open
boolean commitment latch (agent.py:7870) pins that engagement signal at exactly
1.0, swapping clamp->squash changes the external_task affinity logit by -0.96 to
-1.50, while the dacc_pe degeneracy fix this run exists to test changes its logit
by only -0.034 to -0.173 -- an 8x to 28x confound, in the direction of LESS
external_task occupancy. Both a PASS and the pre-registered NULL are therefore
unattributable: each is equally explained by the drive gain being cut roughly in
half. The recorded `et_drive_saturated_frac` telemetry cannot separate them
because it samples the signal PRE-bound.

This also falsifies the inference in substrate_queue.json
`mode-governance-engagement`
implementation_log.squash_sigma_decision_2026_09_19.why_sigma_equals_cap
("every SUB-cap signal passes through precisely as the legacy box clamp passes
it"): slope 1 at the ORIGIN is true, identity over [0, cap] is not.

Two further CONFIRMED (lower-severity) defects, both in this file:
  - `operator_manipulation_landed` (the 1e-6 paired-margin guard) is vacuous:
    the cells are not RNG-paired, and banked 934 data shows same-operator cells
    already differing by 1.7e-3.
  - `bound_mode` is nested OUTERMOST in _run_seed, so all clamp cells precede all
    squash cells on one shared, never-rebuilt, stateful `dual_env`.

Full analysis, the four options, and the recommendation:
  REE_assembly/evidence/planning/sd032a_squash_vs_clamp_sweep_blocked_staged_20260919.md
Decision chip: chip-20260919-sd032a-squash-confounded-gain-cut

SLEEP DRIVER: N/A (waking goal-pipeline onboarding scheduler; no sleep loop).

RED-TEAM (Step 4.5): see the queue entry note for the verdict + model.

WHY THIS RUN EXISTS
-------------------
`mode-governance-engagement` (REE_assembly evidence/planning/substrate_queue.json)
is three coupled items. ITEM (1) -- the bounding operator -- COMPLETED on
2026-09-19: `SalienceCoordinator.tick()`'s hard box clamp now sits behind a mode
selector (ree-v3 956e213f81), and sigma defaults to the configured cap
(524ef52484). The entry's own words: "ITEM (1) IS NOW COMPLETE, operator AND its
sigma rule. Nothing further is owed on the bounding operator itself."

That entry then names THIS run as the validation it unblocks, verbatim:

    "NOW UNBLOCKED ... the cap/operator sweep in the V3-EXQ-934 / 935a lineage --
     a successor to experiments/v3_exq_934_mech266_cap_sweep_mode_occupancy.py
     that sweeps the cap with salience_affinity_bound_mode='squash' against the
     clamp arm, to test whether a bounded-but-GRADED input admits the mixed
     regime the bang-bang clamp could not. It must import
     experiments/_lib/regime_occupancy_gate.py rather than re-deriving min()."

THE PROBLEM THE SQUASH IS BUILT TO FIX (measured, not asserted)
---------------------------------------------------------------
The box clamp is DEGENERATE at the cap: two different large dacc_pe values map to
the identical bounded value, so the affinity logit cannot distinguish them. A
pre-authoring probe against the REAL SalienceCoordinator (sigma = cap default)
measured, at every cap this run sweeps:

  cap    clamp bound pe=16 / pe=17 (separation)   squash bound pe=16 / pe=17 (separation)
  0.75   0.750000000000 / 0.750000000000 (0.0)    0.716417910448 / 0.718309859155 (0.00189)
  1.00   1.000000000000 / 1.000000000000 (0.0)    0.941176470588 / 0.944444444444 (0.00327)
  1.25   1.250000000000 / 1.250000000000 (0.0)    1.159420289855 / 1.164383561644 (0.00496)
  1.50   1.500000000000 / 1.500000000000 (0.0)    1.371428571429 / 1.378378378378 (0.00695)
  1.75   1.750000000000 / 1.750000000000 (0.0)    1.577464788732 / 1.586666666667 (0.00920)

i.e. the clamp separation is EXACTLY ZERO at every swept cap and the squash
separation is not. On the live coordinator the same override moves the whole soft
mode vector (operating_mode['external_task'] differs between operators by up to
0.045 at cap 0.75). V3-EXQ-935a independently recorded the clamp-era signature of
this degeneracy: ext_margin_mean LINEAR in cap at R^2 0.9996-0.9999.

WHY sigma = cap (do NOT "improve" it)
-------------------------------------
sigma sets the SMALL-SIGNAL GAIN as well as where saturation bites: the derivative
at the origin is cap/sigma. At sigma == cap that slope is exactly 1 (measured
0.99999987-0.99999994 at the five swept caps), so every SUB-cap signal passes
through precisely as the legacy box clamp passed it and the ONLY behavioural
change is at the top end, where the clamp was degenerate. That is what lets this
sweep attribute an effect to the OPERATOR change rather than to a simultaneous
gain change. sigma is therefore left at the landed default (None -> cap) and is
NOT swept; an explicit sigma is what a CALIBRATION sweep varies, and an ADAPTIVE
sigma was considered and explicitly declined on the record ("IF IT IS EVER WANTED
IT IS A NEW substrate_queue ITEM").

DESIGN
------
Cross the OPERATOR with the CAP on SHARED seeds, at EVAL time, on clones of ONE
trained curriculum agent per seed. `SalienceCoordinator.tick()` reads
`self.config.affinity_input_cap`, `.affinity_bound_mode` and
`.affinity_squash_sigma` LIVE at every tick, so overriding them on a clone changes
arbitration with no retraining -- the same train-once/sweep-on-clones pattern 934
and 467e use, and empirically confirmed before authoring.

  BOUND_MODES = ["clamp", "squash"]   -- the MANIPULATION. "clamp" is the
      V3-EXQ-934 BASELINE and must reproduce it: the entry's own build constraint
      is "Keep the existing clamp reachable behind a mode selector so the
      V3-EXQ-934 baseline stays reproducible", and the operator contract file's C1
      pins bit-identity of the default path.
  CAP_SWEEP = [0.75, 1.0, 1.25, 1.5, 1.75]  -- 934's band, INHERITED unchanged.
      Transferability caveat, stated rather than papered over: this band was
      chosen from a CLAMP-era synthetic probe. sigma = cap fixes the small-signal
      gain at exactly 1, which is the property that makes it arguably
      transferable, but that is an argument, not a registration. A different band
      would be a new design decision and is not taken here.
  ARM_LABELS = ["ARM_SYMMETRIC", "ARM_ASYM_STICKY_TASK"]  -- 934's two rail arms,
      unchanged. The occupancy bar is defined on ARM_SYMMETRIC (primary); the
      sticky arm deliberately pushes toward saturation and is context only.
  SEEDS = [42, 43, 44]  -- 934's seeds, kept DELIBERATELY. The usual
      substitute-seed-44-on-reef caution is overridden here for a recorded reason:
      the clamp arm's whole job is to reproduce 934's banked cells, which requires
      the identical seeds, and 934's seed 44 guard-PASSED on this exact curriculum
      with the highest contact rate of the three (0.3064, z_goal 0.441).

Training is held at AFFINITY_INPUT_CAP_TRAIN = 2.0 under the CLAMP (934's / 464e's
construction) so the trained substrate is bit-comparable to the banked reference
and BOTH eval operators are applied to the same trained agent. use_closure_operator
OFF (closure injects a confounding mode-switch signal).

Cells = BOUND_MODE x CAP x ARM per seed = 2 x 5 x 2 = 20 (934 had 10).

DEPENDENT VARIABLE (LOAD-BEARING) -- OCCUPANCY
----------------------------------------------
`fraction_in_external_task` (discrete committed-mode occupancy), read through
`experiments/_lib/regime_occupancy_gate.evaluate_regime_occupancy_gate`.

THE BAR IS TRANSCRIBED, NOT INVENTED -- from this entry's own failure_record
target for V3-EXQ-934: "A COMMON cap value yielding per-arm occupancy in
(0.1, 0.9) on >= 2/3 seeds on ARM_SYMMETRIC, with a mixed band at least 2 grid
steps wide", which the gate module implements as: GRADED iff a run of
>= min_adjacent (2) CONSECUTIVE swept values each reads mixed on
>= min_seed_fraction (2/3) of the seeds measured at that value.

CALL SHAPE (the trap that cost 934 its routing): the gate is called ONCE per
(bound_mode, arm) over ALL (seed, cap) cells, with `seed` AND `sweep_value`
populated -- never per-seed with booleans counted afterwards. The entry: "Do NOT
call per-seed and count booleans afterwards -- that is precisely the 934 shape
whose collapse of 'which cap' produced the false routing." 934's own
`_regime_for_arm` is that defective shape (it builds OccupancyCells with neither
seed nor sweep_value) and is deliberately NOT inherited.

LOAD-BEARING CRITERION: the SQUASH x ARM_SYMMETRIC gate returns graded == True.
The clamp x ARM_SYMMETRIC gate is the recorded 934 BASELINE and is reported, NOT
load-bearing.

TELEMETRY, RECORDED WITH NO THRESHOLD (user decision 2026-09-19T22:12:50Z)
--------------------------------------------------------------------------
The continuous pre-argmax margin is RECORDED as telemetry and scored against NO
threshold. This is deliberate and is what makes a third occupancy failure
ATTRIBUTABLE rather than merely disappointing:

  ext_margin_mean/p10/p50/p90/max -- the continuous operating_mode['external_task'].
  margin_cap_linearity_r2 -- R^2 of a least-squares fit of ext_margin_mean vs cap,
      per (seed, arm, bound_mode). 935a recorded 0.9996-0.9999 under the CLAMP;
      whether the squash degrades it is REPORTED, never gated.
  operator margin deltas -- paired squash-minus-clamp ext_margin_mean per
      (seed, cap, arm).
  COMMITMENT-LATCH ATTRIBUTION (entry item (2), still OPEN):
      et_drive_saturated_frac -- fraction of eval steps where the injected
          external_task_drive engagement is EXACTLY 1.0. agent.py:7870 computes
          _et_commit = commit_w * float(beta_gate.is_elevated), a BOOLEAN latch,
          then :7881 clips _et_commit + _et_prox into [0,1] -- so at the default
          commit_weight 1.0 ANY beta elevation saturates engagement to exactly 1.0
          regardless of goal proximity. This is a cap- AND operator-INDEPENDENT
          discreteness source.
      dacc_pe_mean / dacc_pe_abs_max / dacc_pe_over_cap_frac -- whether the
          operator BITES at all (it acts only where |signal| is comparable to or
          above the cap). external_task_drive is already bounded to [0,1], so at
          any cap >= 1.0 BOTH operators are no-ops on it (934's own
          failure_record); the squash bites on dacc_pe (~16-17), i.e. on how hard
          internal_planning dominates.

WHAT A NULL MEANS (pre-registered, so a third failure is not a surprise)
------------------------------------------------------------------------
The entry predicts in writing that a graded operator ALONE may not be enough:
"item (2) commitment-term grading -- agent.py's _et_commit = commit_w *
float(beta_gate.is_elevated) remains a boolean latch and a cap-INDEPENDENT
discreteness source, so a graded bounding operator alone does NOT make the
register graded end-to-end". So a squash-not-graded read, WITH the operator
manipulation demonstrably landing on the continuous margin, is an INFORMATIVE
null: it isolates item (2) (and any other cap-independent source) as the residual
discreteness, rather than leaving the operator under suspicion. That is
pre-registered here as the point, not read out of the manifest afterwards.

PER-CLAIM DIRECTION: diagnostic, EXCLUDED from governance confidence/conflict
scoring. MECH-266 non_contributory (this establishes the measurement precondition
for the over-binding test; it does not itself measure over-binding). SD-032a
supports if the graded operator admits the mixed regime, weakens if it provably
does not while demonstrably landing (favouring a residual structural/latch
source), else non_contributory.

THIS PROMOTES NOTHING. The entry's governance line for this lineage is "PROMOTES
NOTHING. claims.yaml untouched; MECH-266 / SD-032a pending_retest_after_substrate
holds". It validates ITEM (1) ONLY; items (2) (commitment-term grading) and (3)
(production default) remain OPEN, and the production defaults
(salience_affinity_input_cap = None, use_external_task_drive = False) are NOT
flipped here -- that IS item (3).

claim_ids: MECH-266, SD-032a.
experiment_purpose: diagnostic
predecessor: V3-EXQ-934 (successor; NOT a supersede -- 934 measured the CLAMP
             across the same band and that reading stands).
"""

from __future__ import annotations

import argparse
import copy
import json
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
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.cingulate.salience_coordinator import (  # noqa: E402
    AFFINITY_BOUND_CLAMP,
    AFFINITY_BOUND_SQUASH,
)
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _derive_env_seed,
    _sd049_kwargs,
    _sense_with_optional_harm,
    stage_plan,
)
from experiments.pack_writer import write_flat_manifest, flat_readout  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.regime_occupancy_gate import (  # noqa: E402
    OccupancyCell,
    evaluate_regime_occupancy_gate,
)

EXPERIMENT_TYPE = "v3_exq_1067_mech266_squash_vs_clamp_cap_sweep"
QUEUE_ID = "V3-EXQ-1067"
CLAIM_IDS: List[str] = ["MECH-266", "SD-032a"]
EXPERIMENT_PURPOSE = "diagnostic"
PREDECESSOR = "V3-EXQ-934 (successor; NOT a supersede)"

# Same exemption 934 carries, for the same two readiness anchors. They are
# ordinary UPSTREAM gates -- "did the curriculum train" and "does the external_task
# drive engage at all" -- NOT the "positive control reproduces a known degenerate
# signature" pattern the V3-EXQ-778d readiness-anchor reachability guard targets.
# Reachable BY CONSTRUCTION: 934 cleared the 603n contact guard on this exact
# curriculum on 3/3 seeds, and the pre-authoring probe measured the continuous
# margin at 0.33-0.50 across the swept band under BOTH operators, ~7-10x the 0.05
# floor. The 778d failure mode (an unmeetable predicate that mislabels an
# instrument gap as a substrate verdict) does not apply: a margin-PASS with
# saturated occupancy under BOTH operators routes an informative,
# pre-registered null that names the residual discreteness source, never a
# starved false-falsification.
ANCHOR_REACHABILITY_EXEMPT = (
    "Readiness anchors are ordinary upstream gates (curriculum-trained; drive-engages), "
    "not degeneracy-reproduction anchors; thresholds reachable by construction (934 "
    "cleared the contact guard 3/3 on this curriculum; the 0.05 margin floor cleared "
    "~7-10x by the pre-authoring probe under BOTH operators). A margin-pass with "
    "saturated occupancy routes a VALID pre-registered null isolating the "
    "cap-independent commitment latch, never a starved false-falsification."
)

SEEDS = [42, 43, 44]
CONDITION_LABEL = "CURRICULUM_BUILT_SQUASH_VS_CLAMP_CAP_SWEEP"

_ZG = ZGoalStreamAccumulator()

MODE_NAMES = [
    "external_task",
    "internal_planning",
    "internal_replay",
    "offline_consolidation",
]
STICKY_MODE = "external_task"
STICKY_EXIT = 0.05
LOOSE_EXIT = 0.90

# THE MANIPULATION. "clamp" is the V3-EXQ-934 baseline arm.
BOUND_MODES: List[str] = [AFFINITY_BOUND_CLAMP, AFFINITY_BOUND_SQUASH]
# sigma stays at the landed DEFAULT (None -> sigma = cap). NOT a swept knob here.
AFFINITY_SQUASH_SIGMA: Optional[float] = None

# 934's cap band, inherited unchanged (see the docstring's transferability caveat).
CAP_SWEEP: List[float] = [0.75, 1.0, 1.25, 1.5, 1.75]
# Training-time cap AND operator: 464e/934's construction, so the trained substrate
# is comparable to the banked reference and both eval operators share one agent.
AFFINITY_INPUT_CAP_TRAIN = 2.0
AFFINITY_BOUND_MODE_TRAIN = AFFINITY_BOUND_CLAMP

# The mixed band, transcribed from the entry's own failure_record target.
OCCUPANCY_FLOOR = 0.10
OCCUPANCY_CEILING = 0.90
# Gate parameters -- the module defaults, named here so they are pre-registered in
# the manifest rather than implicit.
GATE_MIN_SEED_FRACTION = 2.0 / 3.0
GATE_MIN_ADJACENT = 2
GATE_MIN_SEEDS = 2

# G-margin readiness floor (934's value).
MARGIN_FLOOR = 0.05

# Bit-level identity epsilon for the two "did the manipulation land at all"
# non-degeneracy checks. This is a float-equality tolerance, NOT a contrast
# threshold on the DV -- no threshold is registered for the squash-vs-clamp
# contrast anywhere, by decision.
MANIPULATION_EPS = 1e-6

ARM_LABELS = ["ARM_SYMMETRIC", "ARM_ASYM_STICKY_TASK"]
PRIMARY_ARM = "ARM_SYMMETRIC"

WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
HAZARD_STAGE_BUDGET = 40
P1_BUDGET = 50
P2_BUDGET = 15
MODE_EVAL_EPISODES = 15
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
HARM_PATHWAY_LR = 1e-3
STAGE0B_RETENTION_GATE = 0.75

P2_ZGOAL_GATE = 0.4
CONTACT_GATE = 0.0
MIN_FRACTION = 2.0 / 3.0


def _make_scaffold_cfg(dry_run: bool,
                       env_seed: Optional[int] = None) -> ScaffoldedSD054OnboardingConfig:
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
        scaffold_env_seed=env_seed,
    )
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env) -> REEConfig:
    """603n-validated foraging substrate + SalienceCoordinator + dACC + LateralPFC +
    bistable. use_closure_operator OFF (closure would inject a confounding
    mode-switch signal). TRAINING holds salience_affinity_input_cap at
    AFFINITY_INPUT_CAP_TRAIN (= 2.0, 464e/934's value) under the CLAMP, so the
    trained substrate is bit-comparable to the banked 934 reference; the EVAL cap
    AND operator are swept per cell on clones."""
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
        use_dacc=True,
        use_salience_coordinator=True,
        use_lateral_pfc_analog=True,
        use_closure_operator=False,
        use_external_task_drive=True,
        external_task_drive_affinity_weight=3.0,
        external_task_drive_salience_weight=2.0,
        external_task_drive_commit_weight=1.0,
        external_task_drive_proximity_weight=1.0,
        salience_affinity_input_cap=AFFINITY_INPUT_CAP_TRAIN,
        salience_affinity_bound_mode=AFFINITY_BOUND_MODE_TRAIN,
        salience_affinity_squash_sigma=AFFINITY_SQUASH_SIGMA,
    )
    cfg.latent.use_resource_encoder = True
    cfg.heartbeat.beta_gate_bistable = True
    return cfg


def _build_dual_cue_env(scaffold_cfg: ScaffoldedSD054OnboardingConfig,
                        seed: Optional[int] = None) -> CausalGridWorldV2:
    """P2-config foraging env WITH the GAP-3 dual_cue primitive (competing goals),
    identical to 934/464e."""
    p2_hfa = (
        scaffold_cfg.scaffold_p2_hazard_food_attraction_guard
        if scaffold_cfg.scaffold_p2_hazard_food_attraction_guard >= 0.0
        else scaffold_cfg.scaffold_p2_hazard_food_attraction
    )
    return CausalGridWorldV2(
        seed=seed,
        size=scaffold_cfg.scaffold_env_size,
        num_hazards=scaffold_cfg.scaffold_p2_num_hazards,
        num_resources=scaffold_cfg.scaffold_p2_num_resources,
        hazard_food_attraction=p2_hfa,
        proximity_harm_scale=scaffold_cfg.scaffold_p2_proximity_harm_scale,
        limb_damage_enabled=True,
        reef_enabled=True,
        reef_bipartite_layout=True,
        reef_bipartite_axis=scaffold_cfg.scaffold_reef_bipartite_axis,
        reef_bipartite_agent_band_radius=scaffold_cfg.scaffold_reef_bipartite_agent_band_radius,
        reef_bipartite_agent_spawn_in_reef_half=False,
        dual_cue_enabled=True,
        dual_cue_min_active_ticks=10,
        dual_cue_replace_on_early_consume=False,
        dual_cue_type_tags=(1, 2),
        **_sd049_kwargs(scaffold_cfg),
    )


def _clone_for_arm(trained_agent: REEAgent, device: torch.device) -> REEAgent:
    """Clone the SAME trained weights into a fresh agent (rails, eval cap and eval
    bounding operator applied by the caller). INHERITED FROM V3-EXQ-934, NOT from
    464d/467d: GoalState is a plain Python object, invisible to state_dict(), so a
    weights-only clone dropped its z_goal attractor and
    external_task_drive_require_goal_active=True then hard-gated engagement to
    exactly 0.0 for the whole eval."""
    cfg = copy.deepcopy(trained_agent.config)
    agent = REEAgent(cfg).to(device)
    state = {k: v.detach().clone() for k, v in trained_agent.state_dict().items()}
    try:
        agent.load_state_dict(state)
    except RuntimeError:
        agent.load_state_dict(state, strict=False)
    agent.e3._running_variance = float(trained_agent.e3._running_variance)
    if trained_agent.goal_state is not None and agent.goal_state is not None:
        agent.goal_state.load_state_dict(trained_agent.goal_state.state_dict())
    return agent


def _apply_symmetric(coord) -> None:
    coord.config.enter_thresholds = {}
    coord.config.exit_thresholds = {}


def _apply_asymmetric_sticky_task(coord) -> None:
    coord.config.enter_thresholds = {}
    coord.config.exit_thresholds = {}
    for mode in MODE_NAMES:
        coord.set_exit_threshold(mode, STICKY_EXIT if mode == STICKY_MODE else LOOSE_EXIT)


def _apply_rails(coord, arm_label: str) -> None:
    if arm_label == "ARM_SYMMETRIC":
        _apply_symmetric(coord)
    elif arm_label == "ARM_ASYM_STICKY_TASK":
        _apply_asymmetric_sticky_task(coord)
    else:
        raise ValueError(f"unknown arm {arm_label}")


def _quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def _r2_vs_cap(caps: List[float], values: List[float]) -> Optional[float]:
    """R^2 of a least-squares line of `values` on `caps`. Telemetry only -- NO
    threshold is registered against it anywhere. Returns None when undefined
    (< 3 points, or a degenerate/constant axis), so an absent value reads as
    unmeasured rather than as 0.0."""
    if len(caps) < 3 or len(caps) != len(values):
        return None
    x = np.asarray(caps, dtype=float)
    y = np.asarray(values, dtype=float)
    if float(np.ptp(x)) <= 0.0:
        return None
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 0.0:
        # A perfectly constant y is the clamp's degenerate signature, not a fit.
        return None
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_res = float((resid ** 2).sum())
    return float(round(1.0 - ss_res / ss_tot, 6))


def _eval_cell(
    agent: REEAgent,
    env: CausalGridWorldV2,
    cap: float,
    bound_mode: str,
    arm_label: str,
    scaffold_cfg: ScaffoldedSD054OnboardingConfig,
    device: torch.device,
    n_eps: int,
    steps_per_ep: int,
) -> Dict[str, Any]:
    """Frozen-policy eval for ONE (bound_mode, cap, arm) cell. Rails must already be
    applied by the caller; this sets the EVAL-time cap AND bounding operator on the
    coordinator config (all three read live at tick()). Instruments the discrete
    occupancy DV, the continuous pre-argmax margin, a MODE-CONDITIONED dwell, and
    the commitment-latch / signal-magnitude attribution telemetry."""
    agent.eval()
    world_dim = agent.config.latent.world_dim
    coord = agent.salience
    # EVAL-time override -- THIS IS THE SWEEP. All three are read live at tick().
    coord.config.affinity_input_cap = float(cap)
    coord.config.affinity_bound_mode = str(bound_mode)
    coord.config.affinity_squash_sigma = AFFINITY_SQUASH_SIGMA
    feed_harm = scaffold_cfg.scaffold_feed_harm_stream

    coord_ticks_start = int(coord.diagnostics.get("n_ticks", 0))

    mode_step_counts = {m: 0 for m in MODE_NAMES}
    other_mode_steps = 0
    total_switches = 0
    total_steps = 0
    ext_margins: List[float] = []
    ext_run_lengths: List[int] = []
    all_run_lengths: List[int] = []
    # Attribution telemetry (recorded, never gated).
    et_drive_values: List[float] = []
    dacc_pe_values: List[float] = []

    with torch.no_grad():
        for _ep in range(n_eps):
            _, obs_dict = env.reset()
            agent.reset()
            prev_mode = coord.current_mode
            current_run = 1
            ext_run = 1 if prev_mode == STICKY_MODE else 0

            for _ in range(steps_per_ep):
                obs_body = obs_dict["body_state"].to(device)
                obs_world = obs_dict["world_state"].to(device)
                latent = _sense_with_optional_harm(
                    agent, obs_body, obs_world, obs_dict, device, feed_harm
                )

                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick")
                    else torch.zeros(1, world_dim, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)
                action_idx = int(action.argmax(dim=-1).item())

                # Continuous pre-argmax margin (telemetry; no threshold).
                ext_margins.append(float(coord.operating_mode.get(STICKY_MODE, 0.0)))
                # Commitment-latch + signal-magnitude attribution (telemetry).
                # Read AFTER select_action so the values are the ones this tick's
                # arbitration actually consumed.
                et_drive_values.append(
                    float(coord._input_signals.get("external_task_drive", 0.0))
                )
                dacc_pe_values.append(float(coord._input_signals.get("dacc_pe", 0.0)))

                cur_mode = coord.current_mode
                if cur_mode in mode_step_counts:
                    mode_step_counts[cur_mode] += 1
                else:
                    other_mode_steps += 1

                if cur_mode != prev_mode:
                    all_run_lengths.append(current_run)
                    if prev_mode == STICKY_MODE and ext_run > 0:
                        ext_run_lengths.append(ext_run)
                    total_switches += 1
                    current_run = 1
                    ext_run = 1 if cur_mode == STICKY_MODE else 0
                    prev_mode = cur_mode
                else:
                    current_run += 1
                    if cur_mode == STICKY_MODE:
                        ext_run += 1

                total_steps += 1
                _, _harm, done, _info, obs_dict = env.step(action_idx)
                if done:
                    all_run_lengths.append(current_run)
                    if cur_mode == STICKY_MODE and ext_run > 0:
                        ext_run_lengths.append(ext_run)
                    current_run = 0
                    ext_run = 0
                    break

            if current_run > 0:
                all_run_lengths.append(current_run)
                if prev_mode == STICKY_MODE and ext_run > 0:
                    ext_run_lengths.append(ext_run)

    frac_task = mode_step_counts[STICKY_MODE] / total_steps if total_steps else 0.0
    mean_dwell = (
        float(sum(all_run_lengths)) / len(all_run_lengths)
        if all_run_lengths else float(steps_per_ep)
    )
    ext_dwell_mean = (
        float(sum(ext_run_lengths)) / len(ext_run_lengths)
        if ext_run_lengths else 0.0
    )
    margins_sorted = sorted(ext_margins)
    margin_mean = float(sum(ext_margins) / len(ext_margins)) if ext_margins else 0.0
    coord_ticks = int(coord.diagnostics.get("n_ticks", 0)) - coord_ticks_start

    n_et = len(et_drive_values)
    # Item (2) attribution: engagement pinned at EXACTLY 1.0 is the boolean
    # commitment latch saturating (agent.py:7870/:7881).
    et_saturated = sum(1 for v in et_drive_values if v >= 1.0)
    et_zero = sum(1 for v in et_drive_values if v <= 0.0)
    n_pe = len(dacc_pe_values)
    # Does the bounding operator BITE at all on this signal at this cap?
    pe_over_cap = sum(1 for v in dacc_pe_values if abs(v) > float(cap))

    return {
        "cap": float(cap),
        "bound_mode": str(bound_mode),
        "arm": arm_label,
        "cell_label": f"{bound_mode}|cap={cap}|{arm_label}",
        "fraction_in_external_task": round(frac_task, 4),
        # --- continuous margin telemetry (recorded, NO threshold) ---
        "ext_margin_mean": round(margin_mean, 6),
        "ext_margin_p10": round(_quantile(margins_sorted, 0.10), 6),
        "ext_margin_p50": round(_quantile(margins_sorted, 0.50), 6),
        "ext_margin_p90": round(_quantile(margins_sorted, 0.90), 6),
        "ext_margin_max": round(margins_sorted[-1], 6) if margins_sorted else 0.0,
        # --- commitment-latch attribution telemetry (entry item (2)) ---
        "et_drive_mean": round(float(sum(et_drive_values) / n_et), 6) if n_et else 0.0,
        "et_drive_saturated_frac": round(et_saturated / n_et, 4) if n_et else 0.0,
        "et_drive_zero_frac": round(et_zero / n_et, 4) if n_et else 0.0,
        # --- does the operator bite on the signal it acts on? ---
        "dacc_pe_mean": round(float(sum(dacc_pe_values) / n_pe), 4) if n_pe else 0.0,
        "dacc_pe_abs_max": round(max((abs(v) for v in dacc_pe_values), default=0.0), 4),
        "dacc_pe_over_cap_frac": round(pe_over_cap / n_pe, 4) if n_pe else 0.0,
        # --- switching / dwell ---
        "n_switches": total_switches,
        "mean_dwell": round(mean_dwell, 3),
        "ext_dwell_mean": round(ext_dwell_mean, 3),
        "n_ext_runs": len(ext_run_lengths),
        "mode_step_counts": mode_step_counts,
        "other_mode_steps": other_mode_steps,
        "total_steps": total_steps,
        "coord_n_ticks": coord_ticks,
        "n_episodes": n_eps,
    }


def _aborted_seed_record(seed: int, stage: str, reason: str) -> Dict[str, Any]:
    return {
        "seed": seed, "aborted_at": stage, "abort_reason": reason,
        "guard_pass": False,
        "p2_contact_rate": 0.0, "p2_z_goal_norm_at_contact_peak": 0.0,
        "p2_num_contact_events": 0,
        "cells": [],
        "max_margin_mean": 0.0,
        "margin_engaged": False,
    }


def _run_seed(seed: int, dry_run: bool, total_eps: int,
              env_seed_base: Optional[int] = None) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    seed_env_base = None if env_seed_base is None else int(env_seed_base) + int(seed)
    scaffold_cfg = _make_scaffold_cfg(dry_run, env_seed=seed_env_base)
    device = torch.device("cpu")
    steps_per_ep = scaffold_cfg.scaffold_steps_per_episode
    eval_eps = 2 if dry_run else MODE_EVAL_EPISODES
    caps = CAP_SWEEP[:2] if dry_run else CAP_SWEEP

    probe_env = _build_dual_cue_env(
        scaffold_cfg, seed=_derive_env_seed(seed_env_base, stream=2, idx=0)
    )
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

    s0b = scheduler.run_stage0b_consolidation(
        agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
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
    _ZG.observe(agent)

    dual_env = _build_dual_cue_env(
        scaffold_cfg, seed=_derive_env_seed(seed_env_base, stream=2, idx=1)
    )
    dual_env.reset()

    # Sweep BOUND_MODE x CAP x ARM on clones of the SAME trained agent.
    cells: List[Dict[str, Any]] = []
    for bound_mode in BOUND_MODES:
        for cap in caps:
            for arm_label in ARM_LABELS:
                agent_cell = _clone_for_arm(agent, device)
                _apply_rails(agent_cell.salience, arm_label)
                cell = _eval_cell(
                    agent_cell, dual_env, cap, bound_mode, arm_label,
                    scaffold_cfg, device, eval_eps, steps_per_ep,
                )
                cell["seed"] = int(seed)
                cells.append(cell)
                done += eval_eps
                _ZG.observe(agent_cell)
                print(f"  [eval] seed={seed} mode={bound_mode} cap={cap} {arm_label}"
                      f" occ={cell['fraction_in_external_task']}"
                      f" margin_mean={cell['ext_margin_mean']}"
                      f" et_sat={cell['et_drive_saturated_frac']}"
                      f" pe_over_cap={cell['dacc_pe_over_cap_frac']}"
                      f" switches={cell['n_switches']}", flush=True)

    max_margin_mean = max((float(c["ext_margin_mean"]) for c in cells), default=0.0)
    margin_engaged = bool(max_margin_mean > MARGIN_FLOOR)

    print(f"  [seed] seed={seed} max_margin={max_margin_mean:.4f}"
          f" margin_engaged={margin_engaged} n_cells={len(cells)}", flush=True)
    print(f"verdict: {'PASS' if (guard_pass and margin_engaged) else 'FAIL'}"
          f" seed={seed} guard_pass={guard_pass} margin_engaged={margin_engaged}"
          f" (contact_rate={p2.contact_rate:.4f}"
          f" z_goal_at_contact={p2.z_goal_norm_at_contact_peak:.4f})", flush=True)

    return {
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
        "cells": cells,
        "max_margin_mean": round(max_margin_mean, 6),
        "margin_engaged": margin_engaged,
    }


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


def _gate_for(cells: List[Dict[str, Any]], bound_mode: str,
              arm_label: str) -> Dict[str, Any]:
    """Run the regime-conditioned occupancy gate ONCE over ALL (seed, cap) cells of
    one (bound_mode, arm) slice.

    THIS CALL SHAPE IS THE POINT. Both `seed` and `sweep_value` are populated, so
    the gate can evaluate the entry's actual bar -- >= 2 CONSECUTIVE cap values
    each mixed on >= 2/3 of the seeds measured at that value. V3-EXQ-934 called the
    gate PER SEED with neither field set and counted booleans afterwards, which
    discarded WHICH cap was mixed and produced the false `graded` routing on two
    seeds whose mixed bands were disjoint singletons at opposite ends of the sweep.
    The module FAILS CLOSED to `underdetermined` if the metadata cannot support the
    bar, so this is enforced, not merely intended."""
    slice_cells = [c for c in cells
                   if c["bound_mode"] == bound_mode and c["arm"] == arm_label]
    occ_cells = [
        OccupancyCell(
            label=f"cap={c['cap']}",
            fraction=float(c["fraction_in_external_task"]),
            seed=int(c["seed"]),
            sweep_value=float(c["cap"]),
        )
        for c in slice_cells
    ]
    gate = evaluate_regime_occupancy_gate(
        occ_cells, mode_label=STICKY_MODE,
        floor=OCCUPANCY_FLOOR, ceiling=OCCUPANCY_CEILING,
        min_seed_fraction=GATE_MIN_SEED_FRACTION,
        min_adjacent=GATE_MIN_ADJACENT,
        min_seeds=GATE_MIN_SEEDS,
    )
    gate["bound_mode"] = bound_mode
    gate["arm"] = arm_label
    gate["n_cells"] = len(occ_cells)
    return gate


def run_experiment(dry_run: bool = False,
                   env_seed_base: Optional[int] = None) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run}, "
          f"env_seed_base={env_seed_base})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS
    caps = CAP_SWEEP[:2] if dry_run else CAP_SWEEP
    n_cells = len(BOUND_MODES) * len(caps) * len(ARM_LABELS)
    if dry_run:
        total_eps = 2 + 2 + 5 + 5 + 5 + 2 + n_cells * 2
    else:
        total_eps = (
            STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_STAGE_BUDGET
            + P1_BUDGET + P2_BUDGET + n_cells * MODE_EVAL_EPISODES
        )

    per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        per_seed.append(_run_seed(s, dry_run, total_eps, env_seed_base=env_seed_base))

    n = len(per_seed)
    guard_flags = [r["guard_pass"] for r in per_seed]
    guard_frac = _frac(guard_flags)
    guard_passing = [r for r in per_seed if r["guard_pass"]]
    contact_non_vacuity_met = bool(guard_frac >= MIN_FRACTION)

    margin_flags = [bool(r.get("margin_engaged", False)) for r in guard_passing]
    margin_frac = _frac(margin_flags)
    margin_ready_met = bool(margin_frac >= MIN_FRACTION)

    # All cells from guard-passing seeds -- the gate is called over these.
    all_cells: List[Dict[str, Any]] = [c for r in guard_passing for c in r.get("cells", [])]

    # ONE gate call per (bound_mode, arm) over ALL (seed, cap) cells.
    gates: Dict[str, Dict[str, Any]] = {}
    for bound_mode in BOUND_MODES:
        for arm_label in ARM_LABELS:
            gates[f"{bound_mode}|{arm_label}"] = _gate_for(all_cells, bound_mode, arm_label)

    squash_primary = gates[f"{AFFINITY_BOUND_SQUASH}|{PRIMARY_ARM}"]
    clamp_primary = gates[f"{AFFINITY_BOUND_CLAMP}|{PRIMARY_ARM}"]

    # THE LOAD-BEARING CRITERION (user decision 2026-09-19T22:12:50Z): occupancy,
    # via the transcribed bar, on the SQUASH x ARM_SYMMETRIC slice.
    squash_graded = bool(squash_primary.get("graded", False))
    # Reported, NOT load-bearing: the V3-EXQ-934 clamp baseline reproduction.
    clamp_graded = bool(clamp_primary.get("graded", False))

    # --- NON-DEGENERACY (bit-level identity checks, NOT contrast thresholds) ---
    # (1) Did the OPERATOR manipulation land at all? Pair cells by (seed, cap, arm)
    #     and ask whether ANY pair's continuous margin differs. If the two operators
    #     are bit-identical everywhere, a not-graded read says nothing about the
    #     operator and is an instrument concern, not a finding.
    by_key: Dict[Any, Dict[str, float]] = {}
    for c in all_cells:
        key = (c["seed"], c["cap"], c["arm"])
        by_key.setdefault(key, {})[c["bound_mode"]] = float(c["ext_margin_mean"])
    operator_margin_deltas: List[Dict[str, Any]] = []
    for key, vals in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        if AFFINITY_BOUND_CLAMP in vals and AFFINITY_BOUND_SQUASH in vals:
            delta = vals[AFFINITY_BOUND_SQUASH] - vals[AFFINITY_BOUND_CLAMP]
            operator_margin_deltas.append({
                "seed": key[0], "cap": key[1], "arm": key[2],
                "clamp_margin_mean": round(vals[AFFINITY_BOUND_CLAMP], 6),
                "squash_margin_mean": round(vals[AFFINITY_BOUND_SQUASH], 6),
                "delta": round(delta, 6),
            })
    max_abs_operator_delta = max(
        (abs(d["delta"]) for d in operator_margin_deltas), default=0.0)
    operator_manipulation_landed = bool(max_abs_operator_delta > MANIPULATION_EPS)

    # Same question on the discrete DV -- reported separately so a reader can see
    # which level the operator reached.
    occ_by_key: Dict[Any, Dict[str, float]] = {}
    for c in all_cells:
        occ_by_key.setdefault((c["seed"], c["cap"], c["arm"]), {})[c["bound_mode"]] = \
            float(c["fraction_in_external_task"])
    max_abs_operator_occ_delta = max(
        (abs(v[AFFINITY_BOUND_SQUASH] - v[AFFINITY_BOUND_CLAMP])
         for v in occ_by_key.values()
         if AFFINITY_BOUND_CLAMP in v and AFFINITY_BOUND_SQUASH in v),
        default=0.0)
    operator_moves_occupancy = bool(max_abs_operator_occ_delta > MANIPULATION_EPS)

    # (2) Did the CAP manipulation land at all (934's check, per operator)?
    def _varies(key: str, bound_mode: str) -> bool:
        for r in guard_passing:
            for arm_label in ARM_LABELS:
                vals = [float(c[key]) for c in r.get("cells", [])
                        if c["arm"] == arm_label and c["bound_mode"] == bound_mode]
                if len(vals) >= 2 and (max(vals) - min(vals)) > MANIPULATION_EPS:
                    return True
        return False

    occupancy_varies = any(_varies("fraction_in_external_task", m) for m in BOUND_MODES)
    margin_varies = any(_varies("ext_margin_mean", m) for m in BOUND_MODES)
    cap_manipulation_landed = bool(occupancy_varies or margin_varies)

    # --- TELEMETRY: margin-vs-cap linearity per (seed, arm, bound_mode). NO
    #     threshold is registered against this anywhere; it is recorded so the
    #     935a clamp-era R^2 0.9996-0.9999 has a like-for-like successor value.
    linearity_rows: List[Dict[str, Any]] = []
    for r in guard_passing:
        for bound_mode in BOUND_MODES:
            for arm_label in ARM_LABELS:
                sel = sorted(
                    [c for c in r.get("cells", [])
                     if c["bound_mode"] == bound_mode and c["arm"] == arm_label],
                    key=lambda c: c["cap"])
                r2 = _r2_vs_cap([float(c["cap"]) for c in sel],
                                [float(c["ext_margin_mean"]) for c in sel])
                linearity_rows.append({
                    "seed": r["seed"], "bound_mode": bound_mode, "arm": arm_label,
                    "margin_cap_linearity_r2": r2,
                    "n_points": len(sel),
                })

    def _mean_r2(bound_mode: str) -> Optional[float]:
        vals = [row["margin_cap_linearity_r2"] for row in linearity_rows
                if row["bound_mode"] == bound_mode
                and row["arm"] == PRIMARY_ARM
                and row["margin_cap_linearity_r2"] is not None]
        return round(float(sum(vals) / len(vals)), 6) if vals else None

    clamp_r2_primary = _mean_r2(AFFINITY_BOUND_CLAMP)
    squash_r2_primary = _mean_r2(AFFINITY_BOUND_SQUASH)

    # --- TELEMETRY: commitment-latch attribution (entry item (2), still OPEN) ---
    et_sat_vals = [float(c["et_drive_saturated_frac"]) for c in all_cells]
    et_sat_mean = round(float(sum(et_sat_vals) / len(et_sat_vals)), 4) if et_sat_vals else 0.0
    et_sat_max = round(max(et_sat_vals), 4) if et_sat_vals else 0.0
    pe_over_vals = [float(c["dacc_pe_over_cap_frac"]) for c in all_cells]
    pe_over_mean = round(float(sum(pe_over_vals) / len(pe_over_vals)), 4) if pe_over_vals else 0.0
    pe_abs_max = round(max((float(c["dacc_pe_abs_max"]) for c in all_cells), default=0.0), 4)

    # --- ROUTING ---
    if not contact_non_vacuity_met:
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        route_reason = "contact_guard_unmet"
    elif not margin_ready_met:
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        route_reason = "external_task_drive_not_engaging"
    elif not operator_manipulation_landed:
        # The two operators produced bit-identical continuous margins everywhere.
        # The pre-authoring probe showed the override lands, so this is an
        # instrument concern to verify, NOT a conclusion about gradedness.
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        route_reason = "operator_manipulation_inert_verify_instrument"
    elif squash_graded:
        outcome = "PASS"
        readiness_route = "squash_operator_admits_mixed_regime"
        route_reason = "graded_regime_reachable_under_squash_on_symmetric_arm"
    elif clamp_graded:
        # Baseline divergence: the clamp arm graded where V3-EXQ-934 recorded no
        # common cap. Surfaced explicitly rather than absorbed -- the contrast is
        # not interpretable until this is explained.
        outcome = "FAIL"
        readiness_route = "clamp_baseline_diverges_from_934"
        route_reason = "clamp_arm_graded_contrast_not_interpretable"
    else:
        # THE PRE-REGISTERED NULL. Both operators saturate while the operator
        # manipulation demonstrably landed on the continuous margin -> the residual
        # discreteness is cap- AND operator-INDEPENDENT, which is exactly what the
        # substrate_queue entry predicts for item (2)'s boolean commitment latch.
        outcome = "FAIL"
        readiness_route = "graded_operator_insufficient_residual_cap_independent_discreteness"
        route_reason = "both_operators_saturated_margin_responds_isolates_commitment_latch"

    if squash_graded:
        sd032a_dir = "supports"
    elif (margin_ready_met and contact_non_vacuity_met
          and operator_manipulation_landed and not clamp_graded):
        sd032a_dir = "weakens"
    else:
        sd032a_dir = "non_contributory"
    direction_map = {
        "MECH-266": "non_contributory",
        "SD-032a": sd032a_dir,
    }
    overall_direction = "non_contributory"

    squash_band = squash_primary.get("reproducible_band")
    clamp_band = clamp_primary.get("reproducible_band")

    print(f"[{EXPERIMENT_TYPE}] contact_ready={contact_non_vacuity_met}"
          f" (guard {sum(guard_flags)}/{n}) margin_ready={margin_ready_met}"
          f" (frac={margin_frac:.3f})", flush=True)
    print(f"[{EXPERIMENT_TYPE}] SQUASH/{PRIMARY_ARM}"
          f" shape={squash_primary.get('regime_shape')}"
          f" graded={squash_graded} run={squash_primary.get('longest_adjacent_run')}"
          f" band={squash_band}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] CLAMP/{PRIMARY_ARM} (934 baseline)"
          f" shape={clamp_primary.get('regime_shape')}"
          f" graded={clamp_graded} run={clamp_primary.get('longest_adjacent_run')}"
          f" band={clamp_band}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] operator_landed={operator_manipulation_landed}"
          f" max_abs_margin_delta={max_abs_operator_delta:.6f}"
          f" moves_occupancy={operator_moves_occupancy}"
          f" r2_clamp={clamp_r2_primary} r2_squash={squash_r2_primary}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] latch_telemetry et_saturated_frac"
          f" mean={et_sat_mean} max={et_sat_max}"
          f" dacc_pe_over_cap_frac_mean={pe_over_mean} dacc_pe_abs_max={pe_abs_max}",
          flush=True)
    print(f"[{EXPERIMENT_TYPE}] -> outcome={outcome} route={readiness_route}", flush=True)
    for cid in CLAIM_IDS:
        print(f"[{EXPERIMENT_TYPE}] per_claim {cid}={direction_map[cid]}", flush=True)

    acceptance = {
        "contact_non_vacuity_met": contact_non_vacuity_met,
        "guard_fraction": guard_frac,
        "n_guard_passing_seeds": len(guard_passing),
        "margin_ready_met": margin_ready_met,
        "margin_ready_fraction": margin_frac,
        "squash_symmetric_graded": squash_graded,
        "clamp_symmetric_graded_baseline": clamp_graded,
        "squash_reproducible_band": squash_band,
        "clamp_reproducible_band": clamp_band,
        "operator_manipulation_landed": operator_manipulation_landed,
        "max_abs_operator_margin_delta": round(max_abs_operator_delta, 6),
        "operator_moves_occupancy": operator_moves_occupancy,
        "max_abs_operator_occupancy_delta": round(max_abs_operator_occ_delta, 6),
        "cap_manipulation_landed": cap_manipulation_landed,
        "occupancy_varies_across_caps": occupancy_varies,
        "margin_varies_across_caps": margin_varies,
        "route_reason": route_reason,
        "per_seed_guard_pass": guard_flags,
    }

    preconditions = [
        {
            "name": "foraging_contact_guard",
            "kind": "readiness",
            "description": "603n G2+G3 contact guard on >= 2/3 seeds. A curriculum "
                           "that never became foraging-competent makes every "
                           "occupancy reading meaningless.",
            "control": "fraction of seeds with P2 contact_rate > 0 AND "
                       "z_goal_norm_at_contact_peak > 0.4. Cleared 3/3 by "
                       "V3-EXQ-934 on this exact curriculum.",
            "measured": round(guard_frac, 4),
            "threshold": MIN_FRACTION,
            "direction": "lower",
            "met": contact_non_vacuity_met,
        },
        {
            "name": "external_task_drive_engages",
            "kind": "readiness",
            "description": "the external_task drive must ENGAGE at SOME swept "
                           "(operator, cap) cell -- per-seed max-over-cells of the "
                           "CONTINUOUS operating_mode['external_task'] margin > "
                           "MARGIN_FLOOR -- on >= 2/3 guard-passing seeds. This is "
                           "the readiness form of the SAME arbitration signal the "
                           "occupancy DV routes on: if the margin is ~0 in every "
                           "cell the drive is not producing the signal (a "
                           "substrate/wiring failure, e.g. a goal_state drop), "
                           "which must self-route substrate_not_ready_requeue and "
                           "NOT be read as gradedness evidence either way.",
            "control": "fraction of guard-passing seeds whose best cell's "
                       "ext_margin_mean clears MARGIN_FLOOR. The pre-authoring "
                       "probe measured 0.33-0.50 across the swept band under BOTH "
                       "operators, ~7-10x this floor.",
            "measured": round(margin_frac, 4),
            "threshold": MIN_FRACTION,
            "direction": "lower",
            "met": margin_ready_met,
        },
    ]

    criteria = [
        {
            "name": "H_squash_symmetric_arm_graded_regime_reachable",
            "load_bearing": True,
            "passed": squash_graded,
            "description": "SQUASH x ARM_SYMMETRIC: >= 2 CONSECUTIVE swept cap "
                           "values each with occupancy in (0.1, 0.9) on >= 2/3 of "
                           "the seeds measured at that value. Bar transcribed from "
                           "the mode-governance-engagement failure_record target "
                           "for V3-EXQ-934; evaluated by "
                           "experiments/_lib/regime_occupancy_gate.py, not "
                           "re-derived here.",
            "measured": int(squash_primary.get("longest_adjacent_run", 0)),
            "threshold": GATE_MIN_ADJACENT,
            "direction": "lower",
        },
        {
            "name": "clamp_symmetric_arm_baseline_934_reproduction",
            "load_bearing": False,
            "passed": not clamp_graded,
            "description": "REPORTED, NOT LOAD-BEARING. V3-EXQ-934 recorded no "
                           "common cap on the clamp; a graded clamp arm here would "
                           "be a baseline divergence that makes the contrast "
                           "uninterpretable, and is routed as such.",
            "measured": int(clamp_primary.get("longest_adjacent_run", 0)),
            "threshold": GATE_MIN_ADJACENT,
            "direction": "upper",
        },
    ]
    combination_rule = (
        "PASS iff the single load-bearing criterion "
        "H_squash_symmetric_arm_graded_regime_reachable passes, AND both readiness "
        "preconditions are met, AND the operator manipulation demonstrably landed "
        "on the continuous margin. The clamp baseline criterion is REPORTED and "
        "never contributes to PASS; it only redirects the FAIL route when the "
        "baseline itself diverges from V3-EXQ-934."
    )

    crit_non_degenerate = bool(
        contact_non_vacuity_met and margin_ready_met and operator_manipulation_landed
    )

    readout = flat_readout({
        "H_squash_symmetric_arm_graded_regime_reachable": squash_graded,
        "n_criteria_passed": sum(1 for c in criteria if c["passed"]),
        "n_criteria_total": len(criteria),
        "n_load_bearing_criteria": sum(1 for c in criteria if c.get("load_bearing")),
        "criteria_non_degenerate_flag": crit_non_degenerate,
        # readiness
        "min_fraction": MIN_FRACTION,
        "contact_non_vacuity_met": contact_non_vacuity_met,
        "guard_fraction": guard_frac,
        "n_guard_passing_seeds": len(guard_passing),
        "margin_ready_met": margin_ready_met,
        "margin_ready_fraction": margin_frac,
        "margin_floor": MARGIN_FLOOR,
        # the load-bearing DV, and the reported baseline
        "squash_symmetric_graded": squash_graded,
        "squash_longest_adjacent_run": int(squash_primary.get("longest_adjacent_run", 0)),
        "clamp_symmetric_graded": clamp_graded,
        "clamp_longest_adjacent_run": int(clamp_primary.get("longest_adjacent_run", 0)),
        "gate_min_adjacent": GATE_MIN_ADJACENT,
        "gate_min_seed_fraction": GATE_MIN_SEED_FRACTION,
        "gate_min_seeds": GATE_MIN_SEEDS,
        "occupancy_floor": OCCUPANCY_FLOOR,
        "occupancy_ceiling": OCCUPANCY_CEILING,
        # non-degeneracy (bit-level identity, not contrast thresholds)
        "operator_manipulation_landed": operator_manipulation_landed,
        "max_abs_operator_margin_delta": round(max_abs_operator_delta, 6),
        "operator_moves_occupancy": operator_moves_occupancy,
        "max_abs_operator_occupancy_delta": round(max_abs_operator_occ_delta, 6),
        "cap_manipulation_landed": cap_manipulation_landed,
        "occupancy_varies_across_caps": occupancy_varies,
        "margin_varies_across_caps": margin_varies,
        # continuous-margin telemetry -- RECORDED, NO THRESHOLD
        "margin_cap_linearity_r2_clamp_symmetric": clamp_r2_primary,
        "margin_cap_linearity_r2_squash_symmetric": squash_r2_primary,
        # commitment-latch attribution telemetry (entry item (2), OPEN)
        "et_drive_saturated_frac_mean": et_sat_mean,
        "et_drive_saturated_frac_max": et_sat_max,
        "dacc_pe_over_cap_frac_mean": pe_over_mean,
        "dacc_pe_abs_max": pe_abs_max,
        # sweep shape
        "n_seeds": n,
        "n_caps_swept": len(CAP_SWEEP),
        "cap_sweep_min": min(CAP_SWEEP),
        "cap_sweep_max": max(CAP_SWEEP),
        "n_bound_modes": len(BOUND_MODES),
        "n_preconditions_met": sum(1 for pc in preconditions if pc["met"]),
        "n_preconditions_total": len(preconditions),
    })

    return {
        "outcome": outcome,
        "evidence_direction": overall_direction,
        "evidence_direction_per_claim": direction_map,
        "readout": readout,
        "acceptance": acceptance,
        "occupancy_gates": gates,
        "operator_margin_deltas": operator_margin_deltas,
        "margin_cap_linearity": linearity_rows,
        "interpretation": {
            "label": readiness_route,
            "readiness_route": readiness_route,
            "route_reason": route_reason,
            "hypothesis": "A bounded-but-GRADED affinity input (the saturating "
                          "squash, sigma = cap) admits the mixed external_task "
                          "occupancy regime that the bang-bang box clamp could not, "
                          "at some cap in the swept band.",
            "null": "No >= 2 CONSECUTIVE cap values yield ARM_SYMMETRIC occupancy "
                    "in (0.1, 0.9) on >= 2/3 seeds under the squash -- i.e. the "
                    "graded operator alone does not make the mode register graded.",
            "null_meaning": "A null here is INFORMATIVE and pre-registered as such. "
                            "The mode-governance-engagement entry predicts it in "
                            "writing: item (2)'s _et_commit = commit_w * "
                            "float(beta_gate.is_elevated) is a boolean latch and a "
                            "cap- AND operator-INDEPENDENT discreteness source, so "
                            "'a graded bounding operator alone does NOT make the "
                            "register graded end-to-end'. With the operator "
                            "manipulation demonstrably landing on the continuous "
                            "margin (operator_manipulation_landed), a null "
                            "ISOLATES that residual source rather than leaving the "
                            "operator under suspicion. It does NOT license any "
                            "claim promotion or demotion.",
            "combination_rule": combination_rule,
            "preconditions": preconditions,
            "criteria": criteria,
            "criteria_non_degenerate": {
                "H_squash_symmetric_arm_graded_regime_reachable": crit_non_degenerate,
                "clamp_symmetric_arm_baseline_934_reproduction": crit_non_degenerate,
            },
            "occupancy_gate": {
                "definition": "ONE evaluate_regime_occupancy_gate call per "
                              "(bound_mode, arm) over ALL (seed, cap) cells, with "
                              "seed AND sweep_value populated. GRADED iff >= "
                              "min_adjacent (2) CONSECUTIVE swept cap values each "
                              "read mixed on >= min_seed_fraction (2/3) of the "
                              "seeds measured at that value. The bar is TRANSCRIBED "
                              "from mode-governance-engagement's own failure_record "
                              "target for V3-EXQ-934 and implemented in "
                              "experiments/_lib/regime_occupancy_gate.py -- never "
                              "re-derived, and never a min-across-the-sweep "
                              "statistic. V3-EXQ-934's per-seed call shape (no "
                              "seed, no sweep_value, booleans counted afterwards) "
                              "is deliberately NOT inherited: it is what produced "
                              "that run's false 'graded' routing.",
                "load_bearing_slice": f"{AFFINITY_BOUND_SQUASH}|{PRIMARY_ARM}",
                "baseline_slice": f"{AFFINITY_BOUND_CLAMP}|{PRIMARY_ARM}",
                "occupancy_floor": OCCUPANCY_FLOOR,
                "occupancy_ceiling": OCCUPANCY_CEILING,
                "min_seed_fraction": GATE_MIN_SEED_FRACTION,
                "min_adjacent": GATE_MIN_ADJACENT,
                "min_seeds": GATE_MIN_SEEDS,
                "margin_floor": MARGIN_FLOOR,
                "cap_sweep": CAP_SWEEP,
                "bound_modes": BOUND_MODES,
                "primary_arm": PRIMARY_ARM,
            },
            "continuous_margin_telemetry": {
                "status": "RECORDED WITH NO THRESHOLD -- deliberate (user decision "
                          "2026-09-19T22:12:50Z). Occupancy is the load-bearing "
                          "criterion; the continuous margin is recorded so that a "
                          "null is ATTRIBUTABLE to the cap-independent commitment "
                          "latch rather than to the operator. No numeric threshold "
                          "for the squash-vs-clamp contrast is registered anywhere, "
                          "and none is invented here.",
                "clamp_era_reference": "V3-EXQ-935a recorded ext_margin_mean LINEAR "
                                       "in cap at R^2 0.9996-0.9999 under the clamp "
                                       "(the at-cap degeneracy signature).",
                "margin_cap_linearity_r2_clamp_symmetric": clamp_r2_primary,
                "margin_cap_linearity_r2_squash_symmetric": squash_r2_primary,
            },
            "commitment_latch_attribution": {
                "status": "RECORDED WITH NO THRESHOLD. mode-governance-engagement "
                          "item (2) is OPEN and untouched by this run.",
                "mechanism": "ree_core/agent.py:7870 _et_commit = _et_commit_w * "
                             "(1.0 if self.beta_gate.is_elevated else 0.0); :7881 "
                             "_et_engagement = max(0.0, min(1.0, _et_commit + "
                             "_et_prox)). At the default "
                             "external_task_drive_commit_weight = 1.0 ANY beta "
                             "elevation saturates engagement to exactly 1.0 "
                             "regardless of goal proximity.",
                "et_drive_saturated_frac_mean": et_sat_mean,
                "et_drive_saturated_frac_max": et_sat_max,
                "operator_bite_note": "external_task_drive is already bounded to "
                                      "[0,1], so at any cap >= 1.0 BOTH operators "
                                      "are no-ops on it (V3-EXQ-934 failure_record). "
                                      "The operator bites on dacc_pe; "
                                      "dacc_pe_over_cap_frac records how often.",
                "dacc_pe_over_cap_frac_mean": pe_over_mean,
                "dacc_pe_abs_max": pe_abs_max,
            },
            "contact_guard": {
                "definition": "per-seed P2 contact_rate > 0 AND "
                              "z_goal_norm_at_contact_peak > 0.4; < 2/3 seeds -> "
                              "substrate_not_ready_requeue.",
                "min_fraction": MIN_FRACTION,
                "p2_zgoal_gate": P2_ZGOAL_GATE,
                "contact_gate": CONTACT_GATE,
            },
            "scope": "Validates mode-governance-engagement ITEM (1) (the bounding "
                     "operator) ONLY. Items (2) (commitment-term grading) and (3) "
                     "(production default) remain OPEN and are untouched: "
                     "salience_affinity_input_cap and use_external_task_drive are "
                     "set explicitly by this driver and the PRODUCTION defaults are "
                     "NOT flipped. PROMOTES NOTHING -- MECH-266 / SD-032a "
                     "pending_retest_after_substrate holds, and the Stage-H "
                     "nav-competence dependency gating v3_exq_464d / v3_exq_467d is "
                     "untouched.",
        },
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
        "per_seed": per_seed,
    }


def main(dry_run: bool = False,
         env_seed_base: Optional[int] = None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    result = run_experiment(dry_run=dry_run, env_seed_base=env_seed_base)
    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; manifest not written.", flush=True)
        return {"outcome": result["outcome"], "manifest_path": None}

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)

    full_config = {
        "bound_modes": BOUND_MODES,
        "affinity_squash_sigma": AFFINITY_SQUASH_SIGMA,
        "affinity_squash_sigma_rule": "None MEANS sigma = affinity_input_cap (landed "
                                      "default, user decision 2026-09-19T09:45Z). "
                                      "Slope at the origin is then exactly 1, so "
                                      "every SUB-cap signal passes through as the "
                                      "legacy clamp passed it and the only "
                                      "behavioural change is at the top end.",
        "cap_sweep": CAP_SWEEP,
        "affinity_input_cap_train": AFFINITY_INPUT_CAP_TRAIN,
        "affinity_bound_mode_train": AFFINITY_BOUND_MODE_TRAIN,
        "occupancy_floor": OCCUPANCY_FLOOR,
        "occupancy_ceiling": OCCUPANCY_CEILING,
        "gate_min_seed_fraction": GATE_MIN_SEED_FRACTION,
        "gate_min_adjacent": GATE_MIN_ADJACENT,
        "gate_min_seeds": GATE_MIN_SEEDS,
        "margin_floor": MARGIN_FLOOR,
        "manipulation_eps": MANIPULATION_EPS,
        "arms": ARM_LABELS,
        "primary_arm": PRIMARY_ARM,
        "sticky_exit": STICKY_EXIT,
        "loose_exit": LOOSE_EXIT,
        "mode_eval_episodes_per_cell": MODE_EVAL_EPISODES,
        "train_steps": TRAIN_STEPS,
        "seeds": SEEDS,
        "min_fraction": MIN_FRACTION,
        "p2_zgoal_gate": P2_ZGOAL_GATE,
        "contact_gate": CONTACT_GATE,
        "scaffold_curriculum": {
            "stage0_budget": STAGE0_BUDGET, "stage0b_budget": STAGE0B_BUDGET,
            "p0_budget": P0_BUDGET, "hazard_stage_budget": HAZARD_STAGE_BUDGET,
            "p1_budget": P1_BUDGET, "p2_budget": P2_BUDGET,
            "n_resource_types": N_RESOURCE_TYPES,
            "config_basis": "V3-EXQ-603n",
        },
    }

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "env_seed_base": env_seed_base,
        "timestamp_utc": timestamp,
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "sleep_driver_pattern": "N/A (waking goal-pipeline onboarding scheduler; no sleep loop)",
        "substrate": "scaffolded_sd054_onboarding (full curriculum; 603n config) + "
                     "SalienceCoordinator (SD-032a) + mode-governance-engagement "
                     "external_task drive (use_external_task_drive=True) + GAP-3 "
                     "dual_cue competing-goal env + goal_state clone fix + "
                     "salience_affinity_input_cap (trained at 2.0 under the CLAMP; "
                     "EVAL cap swept in [0.75,1.0,1.25,1.5,1.75] CROSSED with "
                     "salience_affinity_bound_mode in [clamp, squash] at the landed "
                     "default sigma = cap, on clones). use_closure_operator OFF.",
        "condition": CONDITION_LABEL,
        "predecessor": PREDECESSOR,
        "validates": "mode-governance-engagement ITEM (1) (bounding operator) ONLY; "
                     "items (2) commitment-term grading and (3) production default "
                     "remain OPEN.",
        "method_note": "Crosses the affinity BOUNDING OPERATOR (clamp vs squash) "
                       "with the cap at EVAL time on clones of a single trained "
                       "curriculum agent per seed -- cap, mode and sigma are all "
                       "read live at SalienceCoordinator.tick(), so no retraining "
                       "is needed (the same train-once/sweep-on-clones pattern 934 "
                       "and 467e use; confirmed by a pre-authoring one-tick probe "
                       "against the real coordinator). Training is held at cap 2.0 "
                       "under the CLAMP (464e/934 construction) so the clamp arm "
                       "reproduces the banked V3-EXQ-934 baseline and both eval "
                       "operators share one trained agent. Non-vacuity via "
                       "experiments/_lib/regime_occupancy_gate.py, called ONCE per "
                       "(bound_mode, arm) over ALL (seed, cap) cells with seed AND "
                       "sweep_value populated -- never per-seed with booleans "
                       "counted afterwards (the V3-EXQ-934 shape that produced its "
                       "false routing), and never min-across-the-sweep.",
        "pre_registered_thresholds": {
            "cap_sweep": CAP_SWEEP,
            "bound_modes": BOUND_MODES,
            "affinity_squash_sigma": AFFINITY_SQUASH_SIGMA,
            "affinity_input_cap_train": AFFINITY_INPUT_CAP_TRAIN,
            "occupancy_floor": OCCUPANCY_FLOOR,
            "occupancy_ceiling": OCCUPANCY_CEILING,
            "gate_min_seed_fraction": GATE_MIN_SEED_FRACTION,
            "gate_min_adjacent": GATE_MIN_ADJACENT,
            "gate_min_seeds": GATE_MIN_SEEDS,
            "margin_floor": MARGIN_FLOOR,
            "min_fraction": MIN_FRACTION,
            "p2_zgoal_gate": P2_ZGOAL_GATE,
            "contact_gate": CONTACT_GATE,
            "sticky_exit": STICKY_EXIT,
            "loose_exit": LOOSE_EXIT,
            "manipulation_eps": MANIPULATION_EPS,
            "continuous_margin_threshold": None,
        },
        "anchor_reachability_exempt": ANCHOR_REACHABILITY_EXEMPT,
        "config": full_config,
        "stage_plan": stage_plan(),
    }
    manifest.update(result)
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=full_config,
        seeds=SEEDS,
        script_path=Path(__file__),
        z_goal_stream_stats=_ZG.stats(),
        started_at=t0,
    )
    print(f"[{EXPERIMENT_TYPE}] manifest -> {out_path}", flush=True)
    print(f"Done. Outcome: {result['outcome']}", flush=True)
    return {"outcome": result["outcome"], "manifest_path": str(out_path)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--env-seed", type=int, default=None,
        help="Opt-in env-seed base. Omitted (the default) reproduces V3-EXQ-934's "
             "OS-entropy env seeding. Set it and every env this run builds is "
             "deterministically seeded. A pinned run is NOT comparable to a landed "
             "one.",
    )
    args = ap.parse_args()
    _res = main(dry_run=args.dry_run, env_seed_base=args.env_seed)
    _outcome_raw = str(_res["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=_res.get("manifest_path"),
        dry_run=bool(args.dry_run),
    )
