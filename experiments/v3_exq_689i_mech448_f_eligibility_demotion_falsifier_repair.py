"""V3-EXQ-689i: MECH-448 (ARC-107) rank-preserving F->eligibility demotion
falsifier -- INSTRUMENT REPAIR of V3-EXQ-689d.

SAME SCIENTIFIC QUESTION as 689d (does the graded, rank-preserving DN
eligibility envelope convert modulatory-channel range into COMMITTED action
diversity?). The 689d PASS was withdrawn on 2026-07-20 by
`REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-689d_2026-07-20.{md,json}`
(REE_assembly 2e6cc2569b) -- `non_contributory` /
`measurement_test_design_defect`, MECH-448 reverted provisional -> candidate,
`genuine_exp_count` 1 -> 0. This is the sanctioned 785 -> 785a / 708 -> 708a
shape: an alphabetic suffix repairing a BROKEN INSTRUMENT, not a re-test of the
same claim against the same ceiling. The re-derive brake is recorded NOT FIRED
for exactly that reason (autopsy sec 10, user-ratified at the Step 8 gate): the
recommended category is `measurement_test_design_defect`, NOT `substrate_ceiling`,
and the substrate is demonstrably built.

WHAT WAS WRONG, AND WHAT THIS SCRIPT DOES ABOUT IT. The entire 689d finding rested
on `C_PRIMARY` (committed-action-class entropy strict-above BOTH collapsed
controls), which was invalid on THREE INDEPENDENT grounds. Fixing only the DV
would leave defects 2 and 3 intact and reproduce a differently-broken PASS, so
all three are repaired here, plus the power deficit and the recording debt.

 1. HOLD-WEIGHTED DV -> FRESH-SELECTION GATE. 689d accumulated
    `selected_class_counts` per ENV STEP from the `select_action` return value
    (`:598`; `pool_class_counts` likewise at `:535`), but `agent.py:5430` returns
    the HELD action on `not ticks["e3_tick"]` BEFORE `e3.select()` is reached,
    and `generate_trajectories` (`agent.py:4812`) returns CACHED candidates on the
    same condition. Both histograms were therefore weighted by HOLD DURATION
    (E3 cadence `heartbeat.e3_steps_per_tick` defaults to 10). A class-histogram
    entropy is a DISTRIBUTION-SHAPE statistic, which the
    `hold_weighted_e3_readout_corpus_sweep_2026-07-20` triage classes
    DISQUALIFYING -- replication reweights the very distribution the statistic
    measures. NOTE the widely-assumed cause is wrong: the skip is NOT
    `beta_gate.is_elevated` (that only chooses step-vs-hold WITHIN an
    already-skipped tick); it is the E3 cadence, so "commitment was effectively
    disabled" would NOT have exculpated the run.

    REPAIR: every per-selection quantity -- the committed class, the proposer
    pool classes, `cand_world_pairwise_dist`, and ALL `last_score_diagnostics`
    reads (route-range, shortlist, and the f_eligibility demotion readouts) -- is
    recorded ONLY on a tick where E3 genuinely re-selected. `n_fresh_select`,
    `n_latched`, `fresh_select_yield` and `replication_factor` are emitted per
    cell so the true denominator is auditable without reconstructing the cadence.
    The hold-weighted occupancy entropy is ALSO emitted, kept distinct, so the
    size and sign of the defect is measurable rather than merely asserted.

    MECHANISM -- SENTINEL KEY, NOT `= None` (following
    `experiments/v3_exq_699b_pcomp_demotion_x_gonogo_fresh_select.py`, ree-v3
    2d3a249e01, in preference to the 785a `= None` reference the autopsy cites).
    Nulling `_last_selected_trajectory` CHANGES SUBSTRATE BEHAVIOUR:
    `post_action_update` (e3_selector.py:3224) falls back to it when
    `_committed_trajectory` is None (the ARC-016 deadlock fix) and runs on EVERY
    step via `update_residue` (agent.py:8006), so nulling it silently skips
    running-variance updates and the prediction_error / dynamic_precision metrics
    on non-E3 ticks. That would perturb the selection dynamics under test and make
    this a DIFFERENT experiment rather than a repaired instrument. Nulling
    `last_score_diagnostics` is None-safe but also not inert (agent.py:9660 would
    fall back to authority defaults). Instead a private sentinel key is stamped
    into the diagnostics dict before every `select_action`: `e3_selector.select()`
    reassigns that dict WHOLESALE at `:2452` (verified on this branch: exactly one
    assignment site, a dict literal, with no early return before it), so the key
    survives IFF `select()` did not run. Its sole ree_core reader
    (agent.py:9660) uses `.get()` with defaults and nothing iterates its keys, so
    the marker is fully INERT.

 2. VACUOUS MATCHED-NOISE CONTROL -> FACTOR B, AND MADE GATING. In 689d,
    `ARM_MATCHED_NOISE` (`MATCHED_ENTROPY_TEMPERATURE = 2.5`, `:226`) was
    BIT-IDENTICAL to `ARM_PROPOSER_CTRL` on every recorded metric on all three
    seeds -- including `n_p1_ticks` (387 / 3616 / 224 exactly), i.e. identical
    trajectories: they were the same arm. `temperature` is declared per-arm and
    folded into `arm_fingerprint`, but it is INERT on the committed path, which
    resolves by deterministic argmin -- the substrate says so itself at
    e3_selector.py:2685 ("a temperature lift raises THIS but not the
    committed-class entropy"). The negative control was UNMEETABLE BY
    CONSTRUCTION. The pre-registered guard `matched_noise_verified_lifting: false`
    fired, was recorded honestly, and the run PASSED ANYWAY, because a criterion
    of the form `strict above BOTH X and Y` degrades SILENTLY to `strict above X`
    when `X == Y`.

    REPAIR (a): the noise arm is instantiated on a path where temperature is
    genuinely LIVE. `use_gap_scaled_commit_temperature=True` (Factor B, MECH-439)
    routes the within-shortlist committed pick through `_gap_scaled_commit_pick`
    (e3_selector.py:1302), which is a real `torch.multinomial` sample rather than
    an argmin -- so stochasticity now reaches the COMMITTED class, which is the
    DV. This is verified INDEPENDENT of Factor A: `_conflict_gap_norm` is computed
    unconditionally whenever there are >= 2 candidates (e3_selector.py:2771), so
    no conflict-graded shortlist is required. Factor B appears ONLY in the noise
    arm and ONLY as a NOISE GENERATOR -- it is not a mechanism under test here,
    and the MECH-439 conflict-grade levers remain OFF on all three other arms
    (this remains the constitutional lever, not the parametric family
    689/689a/689c adjudicated exhausted by failure_autopsy_V3-EXQ-689a_2026-06-20).
    REPAIR (b): `C_CONTROL_DISTINCT` -- a HARD assertion that no two control arms
    produce identical committed-class count vectors on the same seed. Any
    identical pair invalidates the run (`control_arms_not_distinct_invalid`).
    REPAIR (c): `matched_noise_verified_lifting` is now GATING. A non-lifting
    noise control self-routes `matched_noise_control_unmeetable`
    (`non_contributory`) and BLOCKS the PASS instead of informing it.

 3. INTRA-RUN SUBSTRATE DIVERGENCE -> PER-CELL HASH CARDINALITY ASSERT. NEW defect
    class, found only because `arm_fingerprint` carries a per-cell
    `substrate_hash`. In 689d, `ARM_ON` seeds 43 and 44 ran on
    `fc6d17ce5fa323c4...` while every control arm AND `ARM_ON` seed 42 ran on
    `19b4073c41b90202...` -- `ree_core` was edited on the Mac WHILE THE RUN WAS IN
    FLIGHT on 2026-06-21. The split maps onto the finding exactly: the only seed
    whose treatment arm shared a substrate with its own controls (42) is the seed
    that FAILS C_PRIMARY, and BOTH surviving seeds compared a treatment arm on one
    build against controls on another. `C_PRIMARY` therefore had ZERO
    validly-controlled seeds -- sufficient on its own to withdraw the finding, and
    it survives any DV repair.

    REPAIR: `C_SUBSTRATE_INVARIANT` -- the set of per-cell `substrate_hash` values
    across ALL cells must have cardinality exactly 1, else the run FAILS with
    `intra_run_substrate_divergence_invalid` (`non_contributory`). Additionally
    this experiment is PINNED to a cloud worker (`machine_affinity: ree-cloud-3`)
    running a committed checkout, rather than `any` -- 689d's queue note said
    "Pinned ree-cloud-3" but its `machine_affinity` was in fact `any`, which is
    exactly how the Mac claimed it and ran it against a live, edited checkout.
    NOTE a top-level-only `substrate_hash` would have recorded ONE hash for the
    run and hidden this entirely; provenance granularity must follow the UNIT OF
    COMPARISON, not the unit of execution.

 4. POWER -> FIXED FRESH-SELECTION BUDGET PER CELL. At cadence ~10, ARM_ON seed
    44's 238 env steps were only ~24 genuine selections, and seed 42's 510 were
    ~51 -- against a 0.187-nat deciding margin. An entropy over ~24 draws across 5
    classes is not stable at that resolution. Worse, arm exposure was grossly
    ASYMMETRIC (within seed 42: CTRL 387 / OFF 2715 / ON 510, a 7.0x spread):
    equal env steps do NOT imply equal selections.

    REPAIR: P1 runs until the cell has collected `N_FRESH_SELECT_TARGET = 200`
    GENUINE selections (or exhausts `P1_EPISODE_CAP`), so the histogram N is
    equalised ACROSS ARMS BY CONSTRUCTION rather than modelled after the fact.
    The target is declared up front and is a GATING readiness precondition
    (`fresh_e3_selection_sufficiency_all_cells`) -- a cell short of it self-routes
    `substrate_not_ready_requeue`, never a verdict. Seeds 42/43/44/45 (4, up from
    3) with `MIN_SEEDS_FOR_PASS = 3` (up from 2 of 3), so a single lucky seed can
    no longer carry the finding -- which is how 689d passed.

 5. RECORDING DEBT -> ALWAYS-CORE STAMPED. The 689d manifest was missing
    `recording_schema`, top-level `substrate_hash`, `machine_class` and
    `elapsed_seconds`. Stamped here via
    `experiments/_lib/manifest_core.stamp_recording_core(...)` (called AFTER
    `arm_results` is assembled so `substrate_hash` HOISTS from the per-cell
    fingerprints rather than being recomputed and mismatching), and again through
    `pack_writer.write_flat_manifest` (additive, no-op-safe).

WHAT IS DELIBERATELY UNCHANGED. The autopsy adjudicated three criteria as
SURVIVING on threshold-invariance, and their predicates are carried over VERBATIM
-- only the instrument feeding them is repaired:
  * `C_READINESS`  -- `EXCLUDED_COUNT_FLOOR = 0.0`, strict `>`. Replication cannot
    manufacture a positive from an all-zero record.
  * `C_RANK_PRESERVING` -- measured exactly 1.0; a saturated fraction has nowhere
    to move under reweighting.
  * `C_SAFETY` -- reads realized per-env-step harm from `env.step`. The ENV STEP
    IS the correct sampling unit for a realized-harm rate, so harm accumulation is
    deliberately NOT fresh-gated here; hold-weighting is the intended denominator,
    not a defect. (This is the one readout that must stay per-env-step.)
The env, the conversion constant, the arm contrast and the MECH-448 lever config
are otherwise identical to 689d.

THE ARM CONTRAST (matched seeds):
  ARM_PROPOSER_CTRL  proposer source, demotion OFF, top_k k=3, T=1.0
    (collapsed-channel baseline -- the no-conversion-reaches floor).
  ARM_MATCHED_NOISE  proposer source, demotion OFF, top_k k=3, T=1.0 +
    Factor B gap-scaled stochastic commit (alpha=1.0) -- the REPAIRED
    noise-as-diversity negative control, now reachable on the committed path.
  ARM_OFF            e2wf source, demotion OFF, top_k k=3, T=1.0
    (the GAP-A conversion baseline = the hard env-conditional top-k F-prefix the
    569i lever tested).
  ARM_ON (PRIMARY)   e2wf source, demotion ON, graded DN envelope (floor 0.30).

ACCEPTANCE (evidence, claim_ids=[MECH-448]). Verdict chain, in order -- the three
NEW gates precede everything, so a broken instrument can never reach a verdict:
  C_SUBSTRATE_INVARIANT (NEW, hard) -> intra_run_substrate_divergence_invalid
  C_CONTROL_DISTINCT    (NEW, hard) -> control_arms_not_distinct_invalid
  C_FRESH_SUFFICIENT    (NEW, readiness) -> substrate_not_ready_requeue
  C_READINESS           (unchanged) -> substrate_not_ready_requeue
  C_NOISE_LIFTS         (NEW gating; was informational) -> matched_noise_control_unmeetable
  C_RANK_PRESERVING     (unchanged) -> rank_alteration_not_prefix_diagnose
  C_PRIMARY             (unchanged predicate, repaired DV, Miller-Madow estimator) -> conversion_ceiling_persists_despite_demotion
  C_SAFETY              (unchanged) -> demotion_disinhibits_harmful_classes (the ONLY weakens)
  else -> demotion_converts_committed_diversity (supports)

C_PRIMARY ESTIMATOR (residual correction, 2026-07-20). C_PRIMARY gates on
`selected_action_class_entropy_mm` -- the MILLER-MADOW bias-corrected Shannon
entropy, H_MM = H_plugin + (K_obs - 1)/(2N) -- NOT on the plug-in value. The
uncorrected plug-in estimator is still emitted per cell as
`selected_action_class_entropy`, with `miller_madow_correction_nats` beside it
and a `summary.miller_madow_audit` block recording the per-cell corrections plus
the counterfactual verdict under the uncorrected estimator
(`estimator_changes_verdict`).

This aligns 689i with its LANDED sibling V3-EXQ-699c, which independently adopted
the same two fixes for the same DV class in the same 699/689 defect lineage.
699c's split governs: the FIXED-N stopping rule is "the load-bearing fix" (the
differential bias is zero BY CONSTRUCTION rather than by correction), and
Miller-Madow is "belt and braces" for the residual. 689i ALREADY CARRIED the
load-bearing half -- every cell banks exactly N_FRESH_SELECT_TARGET = 200 genuine
selections and P1 stops at the target -- so this closes only the residual
K_obs-driven term. It is a CONSISTENCY FIX, NOT a defect repair: at N=200 the
correction is (K_obs-1)/400, i.e. 0.0025 nats at K_obs=2 up to 0.0100 at K_obs=5,
so the worst plausible arm differential is ~0.0075 nats -- ~4% of the 689d
deciding margin (0.187 nats) and 2.5% of C3_SELECTED_ENTROPY_FLOOR. Real, and
pointing in the direction C_PRIMARY tests, but not decisive alone.
C3_SELECTED_ENTROPY_FLOOR is DELIBERATELY NOT re-tuned to compensate, and the
C_READINESS / C_RANK_PRESERVING / C_SAFETY predicates are untouched -- the
autopsy adjudicated all three as surviving on threshold-invariance, and absorbing
a <=0.01-nat correction into a threshold would quietly undo that adjudication.
`occupancy_class_entropy` and `proposer_pool_class_entropy` stay PLUG-IN: the
former must remain byte-comparable with 689d's recorded hold-weighted DV, which
is the entire reason it is retained.

claim_ids=[MECH-448] only; ARC-107/MECH-447/MECH-449/MECH-439/Q-078 untouched.
MECH-448 stays candidate; this run PROMOTES NOTHING (governance applies). A PASS
restores the embodied env-loop pathway that the 689d withdrawal removed -- after
that withdrawal MECH-448 has NO genuine experimental entry at all (689h is
`purpose: diagnostic`, tests `demotion_x_gonogo_additive` rather than the
standalone conversion assertion, and shares the synthetic-bank pathway with 689g).

Usage:
  /opt/local/bin/python3 experiments/v3_exq_689i_mech448_f_eligibility_demotion_falsifier_repair.py --dry-run
"""

import argparse
import math
import random
import sys
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import (  # noqa: E402
    compute_arm_fingerprint,
    reset_all_rng,
)
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_689i_mech448_f_eligibility_demotion_falsifier_repair"
QUEUE_ID = "V3-EXQ-689i"
# Instrument repair of the withdrawn 689d run (failure_autopsy_V3-EXQ-689d_2026-07-20).
SUPERSEDES: Optional[str] = "v3_exq_689d_mech448_f_eligibility_demotion_falsifier"
CLAIM_IDS: List[str] = ["MECH-448"]
EXPERIMENT_PURPOSE = "evidence"

# --- validate_experiments lint exemptions -------------------------------------
# Both lints detect the EXACT defect this experiment exists to repair, and both
# fire here as FALSE POSITIVES: they pattern-match on a literal
# `agent.e3.<attr> = None` clear immediately preceding select_action(), and this
# script deliberately uses a substrate-inert SENTINEL KEY instead (docstring item
# 1, following 699b). The remedy each lint prescribes is implemented in full --
# accumulation is fresh-gated, n_fresh_select / n_latched / fresh_select_yield /
# replication_factor are emitted per cell, and the hold-weighted quantity is
# emitted TOO and kept distinct (occupancy_class_entropy vs
# selected_action_class_entropy). These markers must NOT be read as "689i still
# carries the 689d defect" -- the opposite is true. Re-audit if the freshness
# mechanism ever changes.
_FRESH_SELECT_EXEMPT_REASON = (
    "Freshness is enforced via a substrate-inert SENTINEL KEY stamped into "
    "agent.e3.last_score_diagnostics before every select_action(), not via a "
    "`= None` clear: e3_selector.select() reassigns that dict wholesale at :2452, "
    "so the key survives iff select() did not run. A `= None` clear of "
    "_last_selected_trajectory would suppress running-variance updates on non-E3 "
    "ticks (e3_selector.py:3224 via agent.py:8006) and perturb the selection "
    "dynamics under test. All per-selection accumulation is fresh-gated; "
    "n_fresh_select / n_latched / fresh_select_yield / replication_factor are "
    "emitted per cell; both the per-commitment and the hold-weighted occupancy "
    "entropies are emitted and kept distinct. Realized harm (C_SAFETY) is "
    "deliberately NOT fresh-gated -- the env step is the correct denominator for "
    "a harm RATE. See failure_autopsy_V3-EXQ-689d_2026-07-20.json."
)
E3_DIAGNOSTICS_STALENESS_EXEMPT = _FRESH_SELECT_EXEMPT_REASON
E3_HOLD_WEIGHTED_READOUT_EXEMPT = _FRESH_SELECT_EXEMPT_REASON

# Private key stamped into agent.e3.last_score_diagnostics before every
# select_action(). e3_selector.select() reassigns that dict wholesale
# (e3_selector.py:2452 -- verified on this branch as the ONLY assignment site, a
# dict literal, with no early return before it), so the key survives IFF select()
# did not run -- an exact, substrate-inert per-tick freshness signal. Nothing in
# ree_core iterates the dict (its sole reader, agent.py:9660, uses .get() with
# defaults), so the extra key is inert. Namespaced to this experiment so two
# concurrently-instrumented drivers can never collide. Never emitted.
_STALE_MARKER_KEY = "_exq689i_stale_marker"

SEEDS = [42, 43, 44, 45]          # 4 seeds (689d had 3) -- power repair
P0_WARMUP_EPISODES = 60           # SD-056 contrastive warmup (569i/689a/689d proven budget)
P1_EPISODE_CAP = 100              # P1 stops EARLY once the fresh-select target is met
N_FRESH_SELECT_TARGET = 200       # genuine E3 selections banked per cell (declared up front)
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1_CAP = 3
DRY_RUN_STEPS = 30
DRY_RUN_FRESH_TARGET = 5

# Acceptance thresholds (pre-registered).
ROUTE_RANGE_FLOOR = 0.01          # C_READINESS (a) -- UNCHANGED from 689d
C1_PAIRWISE_DIST_FLOOR = 0.03     # C_READINESS (b) -- UNCHANGED from 689d
EXCLUDED_COUNT_FLOOR = 0.0        # C_READINESS (c) -- UNCHANGED (strict >, threshold-invariant)
DEMOTION_ACTIVE_FRAC_FLOOR = 0.8  # C_READINESS (c) -- UNCHANGED from 689d
RANK_PRESERVING_FRAC_REQUIRED = 1.0  # C_RANK_PRESERVING -- UNCHANGED (saturated at 1.0)
C3_SELECTED_ENTROPY_FLOOR = 0.3   # C_PRIMARY absolute floor -- UNCHANGED from 689d
PROPOSER_CEILING_FRAC = 0.6       # informational -- UNCHANGED from 689d
SAFETY_HARM_TOL = 0.02            # C_SAFETY -- UNCHANGED from 689d
MIN_SEEDS_FOR_PASS = 3            # of 4 (689d was 2 of 3) -- power repair
MIN_FRESH_SELECT_PER_CELL = N_FRESH_SELECT_TARGET  # gating sufficiency floor

# Factor B (MECH-439) used ONLY in the repaired noise arm, ONLY as a noise
# generator on the committed path. NOT a mechanism under test.
NOISE_ARM_COMMIT_ENTROPY_ALPHA = 1.0   # gap_scaled_commit_entropy_alpha (substrate default)
NOISE_ARM_BASE_TEMPERATURE = 1.0

# MECH-448 lever config (ARC-107) -- UNCHANGED from 689d.
F_ELIGIBILITY_ENVELOPE_FLOOR = 0.30
F_ELIGIBILITY_DN_SIGMA = 0.0

# Shared shortlist / conversion constant (ON all arms) -- UNCHANGED from 689d.
MODULATORY_SHORTLIST_K = 3
MODULATORY_SHORTLIST_MODE = "top_k"
MODULATORY_AUTHORITY_GAIN = 2.0
MODULATORY_AUTHORITY_NORMALIZE_BASIS = "std"
MODULATORY_ROUTE_MIN_RANGE_FLOOR = 1e-6

# SD-056 online contrastive training -- UNCHANGED from 689d.
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
E2_TRAIN_EVERY_K_TICKS = 1
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0

# Behavioural-diversity env: SD-054 reef-bipartite hazard layout -- UNCHANGED from 689d.
ENV_KWARGS: Dict[str, Any] = dict(
    size=12,
    num_hazards=4,
    num_resources=5,
    hazard_harm=0.05,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=10,
    reef_enabled=True,
    n_reef_patches=3,
    reef_patch_radius=2,
    hazard_food_attraction=0.7,
    reef_bipartite_layout=True,
    reef_bipartite_axis="horizontal",
    reef_bipartite_agent_band_radius=1,
)

# Arm ids.
PRIMARY_ARM = "ARM_ON"
OFF_ARM = "ARM_OFF"
PROPOSER_CTRL_ARM = "ARM_PROPOSER_CTRL"
MATCHED_NOISE_ARM = "ARM_MATCHED_NOISE"
# The three non-treatment arms, for the C_CONTROL_DISTINCT pairwise assertion.
CONTROL_ARMS = [PROPOSER_CTRL_ARM, MATCHED_NOISE_ARM, OFF_ARM]

ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": PROPOSER_CTRL_ARM,
        "label": "proposer_collapsed_channel_baseline_control",
        "candidate_summary_source": "proposer",
        "temperature": 1.0,
        "use_f_eligibility_demotion": False,
        "use_gap_scaled_commit_temperature": False,
    },
    {
        # REPAIRED (defect 2): the 689d version differed from ARM_PROPOSER_CTRL
        # only by `temperature`, which is INERT on the deterministic-argmin
        # committed path -- the two arms were bit-identical. Factor B routes the
        # committed pick through a real torch.multinomial sample
        # (_gap_scaled_commit_pick), so stochasticity now reaches the DV.
        "arm_id": MATCHED_NOISE_ARM,
        "label": "proposer_gap_scaled_stochastic_commit_noise_negative_control",
        "candidate_summary_source": "proposer",
        "temperature": NOISE_ARM_BASE_TEMPERATURE,
        "use_f_eligibility_demotion": False,
        "use_gap_scaled_commit_temperature": True,
    },
    {
        "arm_id": OFF_ARM,
        "label": "e2wf_hard_topk_eligibility_demotion_off_gapa_baseline",
        "candidate_summary_source": "e2_world_forward",
        "temperature": 1.0,
        "use_f_eligibility_demotion": False,
        "use_gap_scaled_commit_temperature": False,
    },
    {
        "arm_id": PRIMARY_ARM,
        "label": "e2wf_graded_dn_envelope_f_eligibility_demotion_on",
        "candidate_summary_source": "e2_world_forward",
        "temperature": 1.0,
        "use_f_eligibility_demotion": True,
        "use_gap_scaled_commit_temperature": False,
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """GAP-A-ready conversion stack, identical to 689d except that the noise arm
    additionally enables Factor B (gap-scaled stochastic commit) so that its
    stochasticity actually reaches the COMMITTED class -- the defect-2 repair. All
    other MECH-439 conflict-grade levers stay OFF on every arm."""
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
        # ARC-065 SP-CEM (Layer A)
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # SHARED E3-side bias channels (-> _modulatory_accum non-None)
        use_lateral_pfc_analog=True,
        use_mech295_liking_bridge=True,
        # Other policy-layer regulators + CRF stack OFF
        use_structured_curiosity=False,
        use_e3_score_diversity=False,
        use_noise_floor=False,
        use_tonic_vigor=False,
        use_dacc=False,
        use_ofc_analog=False,
        use_gated_policy=False,
        use_candidate_rule_field=False,
        # SD-056 substrate trained online on every arm
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=2.0,
        # ARC-065 GAP-A divergent eligible set
        candidate_summary_source=str(arm["candidate_summary_source"]),
        # Shared route-range routing + authority + shortlist-then-modulate
        use_modulatory_channel_routing=True,
        modulatory_channel_route_source="cand_world_summary",
        modulatory_channel_route_weight=1.0,
        modulatory_channel_route_min_range_floor=MODULATORY_ROUTE_MIN_RANGE_FLOOR,
        use_modulatory_selection_authority=True,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
        modulatory_authority_normalize_basis=MODULATORY_AUTHORITY_NORMALIZE_BASIS,
        use_modulatory_shortlist_then_modulate=True,
        modulatory_shortlist_mode=MODULATORY_SHORTLIST_MODE,
        modulatory_shortlist_k=MODULATORY_SHORTLIST_K,
        # MECH-439 Factor A OFF on every arm.
        modulatory_shortlist_conflict_graded=False,
        # MECH-439 Factor B: ON only in the repaired noise control (defect-2 fix).
        use_gap_scaled_commit_temperature=bool(arm["use_gap_scaled_commit_temperature"]),
        gap_scaled_commit_entropy_alpha=NOISE_ARM_COMMIT_ENTROPY_ALPHA,
        # --- MECH-448 (ARC-107) rank-preserving F->eligibility demotion ---
        use_f_eligibility_demotion=bool(arm["use_f_eligibility_demotion"]),
        f_eligibility_envelope_floor=F_ELIGIBILITY_ENVELOPE_FLOOR,
        f_eligibility_dn_sigma=F_ELIGIBILITY_DN_SIGMA,
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def _first_actions_K(candidates) -> torch.Tensor:
    rows = []
    for traj in candidates:
        rows.append(traj.actions[:, 0, :].detach().reshape(-1))
    return torch.stack(rows, dim=0)


def _entropy_from_counts(counts: Dict[int, int]) -> float:
    """Plug-in (maximum-likelihood) Shannon entropy in nats.

    Retained UNCHANGED so `occupancy_class_entropy` stays byte-comparable with
    689d's recorded DV (it is the named secondary that MEASURES the defect
    magnitude -- changing its estimator would destroy that comparison).
    Downward-biased by ~(K-1)/(2N); see `_entropy_miller_madow`.
    """
    n = sum(counts.values())
    if n <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / n
        h -= p * math.log(p)
    return float(h)


def _entropy_miller_madow(counts: Dict[int, int]) -> float:
    """Miller-Madow bias-corrected Shannon entropy in nats (C_PRIMARY DV).

    H_MM = H_plugin + (K_obs - 1) / (2N), where K_obs is the number of classes
    actually OBSERVED (non-zero count). This removes the leading term of the
    plug-in estimator's downward bias.

    Note K_obs, not the alphabet size: the support is unknown here (an arm may
    genuinely never commit to a class), and K_obs is the standard conservative
    stand-in. It UNDER-corrects when a class exists but was never sampled, which
    is the safe direction -- it cannot manufacture entropy that was not observed.

    PORTED VERBATIM from the LANDED sibling
    `v3_exq_699c_pcomp_demotion_x_gonogo_fixed_n.py`, which adopted the same two
    fixes for the same DV class in the same 699/689 defect lineage. 699c's split
    is the governing one here: the FIXED-N stopping rule is "the load-bearing
    fix" (it makes the differential bias zero BY CONSTRUCTION), and Miller-Madow
    is "belt and braces" for the residual. 689i ALREADY HAS the load-bearing
    half -- every cell banks exactly N_FRESH_SELECT_TARGET = 200 genuine
    selections and P1 stops at the target -- so N is equal across cells by
    construction and this correction removes only the remaining K_obs-driven
    term. At N = 200 that residual is (K_obs - 1)/400: 0.0025 nats at K_obs=2
    rising to 0.0100 at K_obs=5, so the worst plausible arm differential (a
    K_obs=5 noise arm against a K_obs=2 ON arm, the occupancy split seen in the
    689i dry-run smoke) is ~0.0075 nats -- about 4% of the 689d deciding margin
    (0.187 nats) and 2.5% of C3_SELECTED_ENTROPY_FLOOR. Real, and pointing in
    the direction C_PRIMARY tests, but NOT decisive on its own. The uncorrected
    value is emitted alongside so the correction is auditable per cell.
    """
    n = sum(counts.values())
    if n <= 0:
        return 0.0
    k_obs = sum(1 for c in counts.values() if c > 0)
    return float(_entropy_from_counts(counts) + (k_obs - 1) / (2.0 * n))


def _obs(d: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    h = d.get(key)
    if h is None:
        return None
    return h.float().unsqueeze(0) if h.dim() == 1 else h.float()


def _sample_class_diverse_batch(
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    k: int,
    rng: random.Random,
) -> Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    if len(buffer) < MIN_BUFFER_BEFORE_TRAIN:
        return None
    pool = list(buffer)
    rng.shuffle(pool)
    seen_classes: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for tup in pool:
        cls = int(tup[1].argmax().item())
        if cls not in seen_classes:
            seen_classes[cls] = tup
        if len(seen_classes) >= k:
            break
    if len(seen_classes) < MIN_CLASSES_FOR_TRAIN:
        return None
    samples = list(seen_classes.values())
    picked_ids = {id(s) for s in samples}
    for tup in pool:
        if len(samples) >= k:
            break
        if id(tup) in picked_ids:
            continue
        samples.append(tup)
        picked_ids.add(id(tup))
    return samples


def _e2_contrastive_step(
    agent: REEAgent,
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    optimiser: torch.optim.Optimizer,
    rng: random.Random,
) -> Optional[float]:
    batch = _sample_class_diverse_batch(buffer, CONTRASTIVE_BATCH_K, rng)
    if batch is None:
        return None
    z0_K = torch.stack([t[0] for t in batch]).to(agent.device)
    actions_K = torch.stack([t[1] for t in batch]).to(agent.device)
    z1_K = torch.stack([t[2] for t in batch]).to(agent.device)
    optimiser.zero_grad(set_to_none=True)
    loss = agent.e2.world_forward_contrastive_loss(
        z_world_0=z0_K,
        actions=actions_K,
        z_world_1_targets=z1_K,
        simulation_mode=False,
    )
    if not torch.is_tensor(loss):
        return None
    loss_val = float(loss.detach().item())
    if not math.isfinite(loss_val):
        return loss_val
    if not loss.requires_grad or loss_val == 0.0:
        return loss_val
    weighted = SD056_WEIGHT * loss
    weighted.backward()
    torch.nn.utils.clip_grad_norm_(agent.e2.parameters(), max_norm=MAX_GRAD_NORM)
    optimiser.step()
    return loss_val


# ---------------------------------------------------------------------------
# Per-(seed, arm) runner
# ---------------------------------------------------------------------------

def _run_seed_arm(
    arm: Dict[str, Any],
    seed: int,
    p0_episodes: int,
    p1_episode_cap: int,
    steps_per_episode: int,
    fresh_target: int,
) -> Dict[str, Any]:
    reset_all_rng(seed)

    env = _make_env(seed)
    agent = _make_agent(env, arm)
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)

    transition_buffer: Deque[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)

    arm_temperature = float(arm["temperature"])
    total_train_eps = p0_episodes + p1_episode_cap

    pairwise_dists: List[float] = []
    route_ranges: List[float] = []
    route_range_max = 0.0
    authority_active_ticks = 0
    shortlist_sizes: List[float] = []
    shortlist_active_ticks = 0
    shortlist_mode_seen: Optional[str] = None
    demotion_active_ticks = 0
    envelope_sizes: List[float] = []
    excluded_counts: List[float] = []
    winner_neq_f_argmin_ticks = 0
    rank_preserving_active_ticks = 0
    # Factor B telemetry (noise arm only) -- confirms the sampling step actually fired.
    gap_scaled_commit_active_ticks = 0

    # C_SAFETY: realised harm exposure over P1. DELIBERATELY per-ENV-STEP (NOT
    # fresh-gated) -- the env step is the correct denominator for a harm RATE.
    harm_p1_abs_sum = 0.0
    harm_p1_ticks = 0

    # PRIMARY DV: committed-class counts on GENUINE selections only (defect-1 fix).
    selected_class_counts: Counter = Counter()
    # 689d's DV retained VERBATIM as a named SECONDARY: per-ENV-STEP occupancy,
    # weighting each commitment by its hold duration. Emitting both makes the size
    # and sign of the defect measurable rather than merely asserted.
    occupancy_class_counts: Counter = Counter()
    # Proposer ceiling reference, also fresh-gated (689d accumulated this per env
    # step at :535 off CACHED candidates -- the same defect).
    pool_class_counts: Counter = Counter()

    n_fresh_select = 0
    n_latched = 0
    n_p1_ticks = 0
    n_contrastive_steps = 0
    hold_durations: List[int] = []
    _cur_hold = 0
    p1_episodes_run = 0
    error_note: Optional[str] = None
    target_met = False

    for ep in range(total_train_eps):
        is_p1 = ep >= p0_episodes
        phase_label = "P1" if is_p1 else "P0"
        if is_p1:
            p1_episodes_run += 1

        _, obs_dict = env.reset()
        agent.reset()

        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        pending_capture: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        tick_in_ep = 0

        for _step in range(steps_per_episode):
            body = obs_dict["body_state"].float()
            world = obs_dict["world_state"].float()
            if body.dim() == 1:
                body = body.unsqueeze(0)
            if world.dim() == 1:
                world = world.unsqueeze(0)

            latent = agent.sense(
                obs_body=body, obs_world=world,
                obs_harm=_obs(obs_dict, "harm_obs"),
                obs_harm_a=_obs(obs_dict, "harm_obs_a"),
                obs_harm_history=_obs(obs_dict, "harm_history"),
            )

            if pending_capture is not None:
                z0_prev, a_prev = pending_capture
                z1_obs = latent.z_world.detach().reshape(-1).clone()
                if (
                    torch.isfinite(z0_prev).all()
                    and torch.isfinite(a_prev).all()
                    and torch.isfinite(z1_obs).all()
                ):
                    transition_buffer.append((z0_prev, a_prev, z1_obs))
                pending_capture = None

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(
                    z_self_prev, action_prev, latent.z_self.detach()
                )

            ticks = agent.clock.advance()
            wdim = latent.z_world.shape[-1]
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            # Pool-side quantities are computed BEFORE select_action (they read the
            # candidate list), but freshness is only knowable AFTER it. So BUFFER
            # them here and commit only if the tick turns out to be a genuine
            # selection -- generate_trajectories returns CACHED candidates on a
            # non-E3 tick (agent.py:4812), which is the :535 half of defect 1.
            pending_pool_classes: Optional[List[int]] = None
            pending_pdist: Optional[float] = None
            if is_p1 and candidates and len(candidates) >= 2:
                actions_K = _first_actions_K(candidates).to(agent.device)
                z0 = latent.z_world.detach()
                with torch.no_grad():
                    pdist = float(agent.e2.cand_world_pairwise_dist(z0, actions_K).item())
                if math.isfinite(pdist):
                    pending_pdist = pdist
                pending_pool_classes = [
                    int(c) for c in actions_K.argmax(dim=-1).reshape(-1).tolist()
                ]

            if agent.goal_state is not None:
                try:
                    energy = float(body[0, 3].item())
                except Exception:
                    energy = 1.0
                drive_level = max(0.0, 1.0 - energy)
                agent.update_z_goal(benefit_exposure=0.0, drive_level=drive_level)

            # --- FRESHNESS MARKER (the 689d instrument repair) ------------------
            # See the module docstring item 1 for why a sentinel key rather than a
            # `= None` clear: nulling _last_selected_trajectory changes substrate
            # behaviour via post_action_update, which would make this a different
            # experiment rather than a repaired instrument.
            _diag_prev = agent.e3.last_score_diagnostics
            if isinstance(_diag_prev, dict):
                _diag_prev[_STALE_MARKER_KEY] = True

            action = agent.select_action(
                candidates, ticks, temperature=arm_temperature
            )

            _diag_now = agent.e3.last_score_diagnostics
            fresh_select = (
                isinstance(_diag_now, dict)
                and _STALE_MARKER_KEY not in _diag_now
            )

            # MECH-448 readouts -- read ONLY on a genuine selection. The `else {}`
            # makes every downstream diag.get(...) fresh-gated without an `if`
            # around each one.
            if is_p1:
                diag = _diag_now if (fresh_select and isinstance(_diag_now, dict)) else {}
                if fresh_select:
                    rr = float(diag.get("modulatory_channel_route_range", 0.0))
                    if math.isfinite(rr):
                        route_ranges.append(rr)
                        route_range_max = max(route_range_max, rr)
                    if bool(diag.get("modulatory_authority_active", False)):
                        authority_active_ticks += 1
                    if bool(diag.get("modulatory_shortlist_active", False)):
                        shortlist_active_ticks += 1
                        sl_size = float(diag.get("modulatory_shortlist_size", 0))
                        if math.isfinite(sl_size):
                            shortlist_sizes.append(sl_size)
                        if shortlist_mode_seen is None:
                            shortlist_mode_seen = str(
                                diag.get("modulatory_shortlist_mode", "")
                            )
                    if bool(diag.get("gap_scaled_commit_active", False)):
                        gap_scaled_commit_active_ticks += 1
                    if bool(diag.get("f_eligibility_demotion_active", False)):
                        demotion_active_ticks += 1
                        env_size = float(diag.get("f_eligibility_envelope_size", -1))
                        if math.isfinite(env_size) and env_size >= 0:
                            envelope_sizes.append(env_size)
                        excl = float(diag.get("f_eligibility_excluded_count", -1))
                        if math.isfinite(excl) and excl >= 0:
                            excluded_counts.append(excl)
                        if bool(diag.get("f_eligibility_winner_neq_f_argmin", False)):
                            winner_neq_f_argmin_ticks += 1
                        if bool(diag.get("f_eligibility_rank_preserving", True)):
                            rank_preserving_active_ticks += 1

            if action is None:
                idx = int(np.random.randint(0, env.action_dim))
                action = torch.zeros(1, env.action_dim, device=agent.device)
                action[0, idx] = 1.0
                agent._last_action = action
            if not torch.isfinite(action).all():
                if error_note is None:
                    error_note = (
                        f"non-finite action at arm={arm['arm_id']} seed={seed} "
                        f"phase={phase_label} ep={ep} step={_step}"
                    )
                break

            if is_p1:
                committed_class = int(action[0].argmax().item())
                n_p1_ticks += 1

                # 689d's DV, retained verbatim as a NAMED SECONDARY (hold-weighted).
                occupancy_class_counts[committed_class] += 1

                # PRIMARY DV: per-COMMITMENT, fresh-gated.
                if fresh_select:
                    n_fresh_select += 1
                    selected_class_counts[committed_class] += 1
                    if pending_pool_classes is not None:
                        for cls in pending_pool_classes:
                            pool_class_counts[cls] += 1
                    if pending_pdist is not None:
                        pairwise_dists.append(pending_pdist)
                    if _cur_hold > 0:
                        hold_durations.append(_cur_hold)
                    _cur_hold = 1
                else:
                    n_latched += 1
                    if _cur_hold > 0:
                        _cur_hold += 1

            if (
                torch.isfinite(latent.z_world).all()
                and torch.isfinite(action).all()
            ):
                pending_capture = (
                    latent.z_world.detach().reshape(-1).clone(),
                    action.detach().reshape(-1).clone(),
                )

            if tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0:
                loss_val = _e2_contrastive_step(
                    agent=agent, buffer=transition_buffer,
                    optimiser=e2_opt, rng=sample_rng,
                )
                if loss_val is not None and math.isfinite(loss_val) and is_p1:
                    n_contrastive_steps += 1

            _, harm_signal, done, info, next_obs_dict = env.step(action)
            # C_SAFETY: realised harm exposure (P1 only). NOT fresh-gated by design.
            if is_p1:
                hv = abs(float(harm_signal))
                if math.isfinite(hv):
                    harm_p1_abs_sum += hv
                    harm_p1_ticks += 1
            with torch.no_grad():
                agent.update_residue(
                    harm_signal=float(harm_signal),
                    world_delta=None,
                    hypothesis_tag=False,
                    owned=True,
                )

            z_self_prev = latent.z_self.detach()
            action_prev = action
            obs_dict = next_obs_dict
            tick_in_ep += 1
            if done:
                break

        # Flush any hold open at the episode boundary -- agent.reset() clears the
        # commitment latch, so a hold cannot span episodes.
        if _cur_hold > 0:
            hold_durations.append(_cur_hold)
            _cur_hold = 0

        # ep == 0 is included so the runner establishes episodes_per_run from the
        # FIRST episode (and so a short --dry-run still exercises this parse).
        if ep == 0 or (ep + 1) % 10 == 0 or (ep + 1) == total_train_eps:
            print(
                f"  [train] arm={arm['arm_id']} seed={seed} phase={phase_label} "
                f"ep {ep + 1}/{total_train_eps} fresh={n_fresh_select}/{fresh_target}",
                flush=True,
            )

        if error_note is not None:
            break

        # POWER REPAIR: stop P1 as soon as the cell has banked the declared
        # fresh-selection budget. This equalises the histogram N across arms BY
        # CONSTRUCTION rather than modelling the 689d 7.0x exposure asymmetry
        # after the fact.
        if is_p1 and n_fresh_select >= fresh_target:
            target_met = True
            print(
                f"  [p1-done] arm={arm['arm_id']} seed={seed} "
                f"fresh_select target {fresh_target} met after {p1_episodes_run} P1 episodes",
                flush=True,
            )
            break

    def _mean(xs: List[float], default: float = 0.0) -> float:
        return float(sum(xs) / len(xs)) if xs else default

    # C_PRIMARY gates on the Miller-Madow value; the plug-in value is emitted
    # beside it so the size of the correction is auditable per cell. The two
    # SECONDARY readouts (occupancy, proposer pool) stay PLUG-IN deliberately:
    # occupancy_class_entropy must remain byte-comparable with 689d's recorded
    # hold-weighted DV, which is the whole point of retaining it.
    selected_action_entropy = _entropy_from_counts(dict(selected_class_counts))
    selected_action_entropy_mm = _entropy_miller_madow(dict(selected_class_counts))
    occupancy_entropy = _entropy_from_counts(dict(occupancy_class_counts))
    proposer_pool_entropy = _entropy_from_counts(dict(pool_class_counts))
    demotion_active_frac = (
        float(demotion_active_ticks) / float(n_fresh_select) if n_fresh_select > 0 else 0.0
    )
    rank_preserving_frac = (
        float(rank_preserving_active_ticks) / float(demotion_active_ticks)
        if demotion_active_ticks > 0
        else (1.0 if not bool(arm["use_f_eligibility_demotion"]) else 0.0)
    )
    harm_per_tick_mean = (
        harm_p1_abs_sum / float(harm_p1_ticks) if harm_p1_ticks > 0 else 0.0
    )
    fresh_select_yield = (
        float(n_fresh_select) / float(n_p1_ticks) if n_p1_ticks > 0 else 0.0
    )
    replication_factor = (
        float(n_p1_ticks) / float(n_fresh_select) if n_fresh_select > 0 else 0.0
    )

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": int(seed),
        "candidate_summary_source": arm["candidate_summary_source"],
        "temperature": arm_temperature,
        "use_f_eligibility_demotion": bool(arm["use_f_eligibility_demotion"]),
        "use_gap_scaled_commit_temperature": bool(arm["use_gap_scaled_commit_temperature"]),
        "n_p1_ticks": int(n_p1_ticks),
        "p1_episodes_run": int(p1_episodes_run),
        "n_contrastive_steps": int(n_contrastive_steps),
        "error_note": error_note,
        # --- FRESHNESS TELEMETRY (defect-1 repair; the field whose absence made
        # the 689d denominator unrecoverable) ---
        "n_fresh_select": int(n_fresh_select),
        "n_latched": int(n_latched),
        "fresh_select_yield": round(fresh_select_yield, 6),
        "replication_factor": round(replication_factor, 6),
        "fresh_select_target": int(fresh_target),
        "fresh_select_target_met": bool(target_met or n_fresh_select >= fresh_target),
        "hold_duration_mean": round(_mean([float(h) for h in hold_durations]), 6),
        "hold_duration_max": int(max(hold_durations)) if hold_durations else 0,
        # C_READINESS (a) -- fresh-gated.
        "modulatory_channel_route_range_mean": round(_mean(route_ranges), 6),
        "modulatory_channel_route_range_max": round(route_range_max, 6),
        "modulatory_authority_active_ticks": int(authority_active_ticks),
        # Shortlist diagnostic -- fresh-gated.
        "modulatory_shortlist_size_mean": round(_mean(shortlist_sizes), 6),
        "modulatory_shortlist_active_ticks": int(shortlist_active_ticks),
        "modulatory_shortlist_mode": shortlist_mode_seen or "",
        # Factor B telemetry (noise arm) -- the sampling step actually fired.
        "gap_scaled_commit_active_ticks": int(gap_scaled_commit_active_ticks),
        "gap_scaled_commit_active_frac": round(
            float(gap_scaled_commit_active_ticks) / float(n_fresh_select)
            if n_fresh_select > 0 else 0.0, 6
        ),
        # C_READINESS (b) -- fresh-gated.
        "cand_world_pairwise_dist_mean": round(_mean(pairwise_dists), 6),
        # C_READINESS (c) / MECH-448 non-degeneracy -- fresh-gated.
        "f_eligibility_demotion_active_ticks": int(demotion_active_ticks),
        "f_eligibility_demotion_active_frac": round(demotion_active_frac, 6),
        "f_eligibility_envelope_size_mean": round(_mean(envelope_sizes), 6),
        "f_eligibility_excluded_count_mean": round(_mean(excluded_counts), 6),
        "f_eligibility_winner_neq_f_argmin_ticks": int(winner_neq_f_argmin_ticks),
        # C_RANK_PRESERVING.
        "f_eligibility_rank_preserving_frac": round(rank_preserving_frac, 6),
        # --- C_PRIMARY DV: per-COMMITMENT (REPAIRED), Miller-Madow corrected ---
        # C_PRIMARY GATES ON `selected_action_class_entropy_mm`. The uncorrected
        # plug-in value is retained under its original key so the correction is
        # auditable and the pre-correction comparison remains recoverable.
        "selected_action_class_entropy_mm": round(selected_action_entropy_mm, 6),
        "selected_action_class_entropy": round(selected_action_entropy, 6),
        "miller_madow_correction_nats": round(
            selected_action_entropy_mm - selected_action_entropy, 6
        ),
        "selected_class_counts": dict(sorted(selected_class_counts.items())),
        "selected_classes_n_unique": int(len(selected_class_counts)),
        # --- 689d's hold-weighted DV, retained as a NAMED SECONDARY so the
        # magnitude and sign of the defect is measurable ---
        "occupancy_class_entropy": round(occupancy_entropy, 6),
        "occupancy_class_counts": dict(sorted(occupancy_class_counts.items())),
        "occupancy_minus_fresh_entropy_delta": round(
            occupancy_entropy - selected_action_entropy, 6
        ),
        # Proposer ceiling reference -- fresh-gated.
        "proposer_pool_class_entropy": round(proposer_pool_entropy, 6),
        "proposer_pool_classes_n_unique": int(len(pool_class_counts)),
        # C_SAFETY (per-env-step by design).
        "harm_per_p1_tick_mean": round(harm_per_tick_mean, 6),
        "harm_p1_ticks": int(harm_p1_ticks),
    }


# ---------------------------------------------------------------------------
# Cross-arm evaluation
# ---------------------------------------------------------------------------

def _arm_rows(rows: List[Dict[str, Any]], arm_id: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("arm_id") == arm_id]


def _n_seeds(rows: List[Dict[str, Any]], predicate) -> int:
    return sum(1 for r in rows if predicate(r))


def _mean_key(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [float(r.get(key, 0.0)) for r in rows]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _substrate_hashes(arm_results: List[Dict[str, Any]]) -> List[str]:
    out = []
    for r in arm_results:
        fp = r.get("arm_fingerprint") or {}
        h = fp.get("substrate_hash")
        if isinstance(h, str) and h:
            out.append(h)
    return out


def _identical_control_pairs(arm_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """C_CONTROL_DISTINCT (defect-2 repair). Returns every (seed, armA, armB)
    control pair whose committed-class COUNT VECTOR is identical. In 689d
    ARM_MATCHED_NOISE and ARM_PROPOSER_CTRL were identical on every metric on all
    three seeds -- they were the same arm -- and `strict above BOTH` silently
    degraded to `strict above ONE`. Any hit here invalidates the run."""
    by_seed: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for r in arm_results:
        by_seed.setdefault(int(r["seed"]), {})[str(r["arm_id"])] = r
    hits: List[Dict[str, Any]] = []
    for seed, arms in sorted(by_seed.items()):
        for i in range(len(CONTROL_ARMS)):
            for j in range(i + 1, len(CONTROL_ARMS)):
                a_id, b_id = CONTROL_ARMS[i], CONTROL_ARMS[j]
                ra, rb = arms.get(a_id), arms.get(b_id)
                if ra is None or rb is None:
                    continue
                if ra.get("selected_class_counts") == rb.get("selected_class_counts"):
                    hits.append({
                        "seed": seed,
                        "arm_a": a_id,
                        "arm_b": b_id,
                        "class_counts": ra.get("selected_class_counts"),
                    })
    return hits


def _evaluate(arm_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    proposer = _arm_rows(arm_results, PROPOSER_CTRL_ARM)
    noise = _arm_rows(arm_results, MATCHED_NOISE_ARM)
    off = _arm_rows(arm_results, OFF_ARM)
    on = _arm_rows(arm_results, PRIMARY_ARM)

    proposer_by_seed = {r["seed"]: r for r in proposer}
    noise_by_seed = {r["seed"]: r for r in noise}
    off_by_seed = {r["seed"]: r for r in off}

    RDIST = "modulatory_channel_route_range_mean"
    PDIST = "cand_world_pairwise_dist_mean"
    # C_PRIMARY reads the Miller-Madow-corrected DV. SENT_PLUGIN is the
    # uncorrected estimator, retained for the audit block only -- NOT gated on.
    SENT = "selected_action_class_entropy_mm"
    SENT_PLUGIN = "selected_action_class_entropy"
    HARM = "harm_per_p1_tick_mean"

    # === NEW GATE 1: C_SUBSTRATE_INVARIANT (defect-3 repair) ==================
    hashes = _substrate_hashes(arm_results)
    distinct_hashes = sorted(set(hashes))
    substrate_invariant = bool(len(distinct_hashes) == 1)

    # === NEW GATE 2: C_CONTROL_DISTINCT (defect-2 repair) =====================
    identical_pairs = _identical_control_pairs(arm_results)
    control_distinct = bool(len(identical_pairs) == 0)

    # === NEW GATE 3: C_FRESH_SUFFICIENT (defect-1 + power repair) =============
    all_rows = list(arm_results)
    cells_short = [
        {
            "arm_id": r["arm_id"], "seed": r["seed"],
            "n_fresh_select": int(r.get("n_fresh_select", 0)),
        }
        for r in all_rows
        if int(r.get("n_fresh_select", 0)) < MIN_FRESH_SELECT_PER_CELL
    ]
    min_fresh = min([int(r.get("n_fresh_select", 0)) for r in all_rows] or [0])
    fresh_sufficient = bool(len(cells_short) == 0)

    # === C_READINESS (predicates UNCHANGED from 689d) =========================
    on_route_mean = _mean_key(on, RDIST)
    route_seeds_ok = _n_seeds(on, lambda r: float(r.get(RDIST, 0.0)) > ROUTE_RANGE_FLOOR)
    route_ready = bool(route_seeds_ok >= MIN_SEEDS_FOR_PASS)

    on_pdist_mean = _mean_key(on, PDIST)
    c1_seeds_ok = _n_seeds(on, lambda r: float(r.get(PDIST, 0.0)) > C1_PAIRWISE_DIST_FLOOR)
    c1_pass = bool(c1_seeds_ok >= MIN_SEEDS_FOR_PASS)

    def _on_non_degenerate(r: Dict[str, Any]) -> bool:
        return bool(
            float(r.get("f_eligibility_demotion_active_frac", 0.0)) >= DEMOTION_ACTIVE_FRAC_FLOOR
            and float(r.get("f_eligibility_excluded_count_mean", 0.0)) > EXCLUDED_COUNT_FLOOR
        )
    non_degen_seeds_ok = _n_seeds(on, _on_non_degenerate)
    non_degeneracy_ready = bool(non_degen_seeds_ok >= MIN_SEEDS_FOR_PASS)

    readiness_ok = bool(route_ready and c1_pass and non_degeneracy_ready)

    # === C_NOISE_LIFTS -- NOW GATING (defect-2 repair) ========================
    # In 689d this was informational: it fired `false` and the run passed anyway.
    def _noise_lifts(rn: Dict[str, Any]) -> bool:
        rp = proposer_by_seed.get(rn["seed"])
        if rp is None:
            return False
        return float(rn.get(SENT, 0.0)) > float(rp.get(SENT, 0.0)) + 1e-6
    noise_lift_seeds = _n_seeds(noise, _noise_lifts)
    matched_noise_verified_lifting = bool(noise_lift_seeds >= MIN_SEEDS_FOR_PASS)

    # === C_RANK_PRESERVING (predicate UNCHANGED from 689d) ===================
    rank_seeds_ok = _n_seeds(
        on, lambda r: float(r.get("f_eligibility_rank_preserving_frac", 0.0)) >= RANK_PRESERVING_FRAC_REQUIRED
    )
    rank_preserving_pass = bool(rank_seeds_ok >= MIN_SEEDS_FOR_PASS)

    # === C_PRIMARY (predicate UNCHANGED; the DV underneath is repaired) ======
    def _strict_above_pair(r1: Dict[str, Any], by_seed_a, by_seed_b) -> bool:
        ra = by_seed_a.get(r1["seed"])
        rb = by_seed_b.get(r1["seed"])
        if ra is None or rb is None:
            return False
        e1 = float(r1.get(SENT, 0.0))
        return e1 > float(ra.get(SENT, 0.0)) and e1 > float(rb.get(SENT, 0.0))

    primary_seeds_ok = _n_seeds(
        on, lambda r: _strict_above_pair(r, proposer_by_seed, noise_by_seed)
    )
    on_sel_mean = _mean_key(on, SENT)
    primary_floor_ok = bool(on_sel_mean > C3_SELECTED_ENTROPY_FLOOR)
    primary_pass = bool(primary_seeds_ok >= MIN_SEEDS_FOR_PASS and primary_floor_ok)

    # Counterfactual: the SAME predicate evaluated on the UNCORRECTED plug-in
    # estimator. Recorded so a reader can see directly whether the Miller-Madow
    # correction changed the verdict on THIS run, rather than having to infer it
    # from the per-cell correction sizes. Expected to be identical -- with N
    # equalised at N_FRESH_SELECT_TARGET the residual is ~0.0025-0.0100 nats
    # against a 0.187-nat deciding margin -- and a DIFFERENCE here is a finding
    # worth flagging in review, not a silent improvement.
    def _strict_above_pair_plugin(r1: Dict[str, Any], by_seed_a, by_seed_b) -> bool:
        ra = by_seed_a.get(r1["seed"])
        rb = by_seed_b.get(r1["seed"])
        if ra is None or rb is None:
            return False
        e1 = float(r1.get(SENT_PLUGIN, 0.0))
        return e1 > float(ra.get(SENT_PLUGIN, 0.0)) and e1 > float(rb.get(SENT_PLUGIN, 0.0))

    primary_seeds_ok_plugin = _n_seeds(
        on, lambda r: _strict_above_pair_plugin(r, proposer_by_seed, noise_by_seed)
    )
    on_sel_mean_plugin = _mean_key(on, SENT_PLUGIN)
    primary_pass_plugin = bool(
        primary_seeds_ok_plugin >= MIN_SEEDS_FOR_PASS
        and on_sel_mean_plugin > C3_SELECTED_ENTROPY_FLOOR
    )
    estimator_changes_verdict = bool(primary_pass != primary_pass_plugin)

    # === C_SAFETY (predicate UNCHANGED from 689d) ============================
    def _safe_vs_off(r_on: Dict[str, Any]) -> bool:
        r_off = off_by_seed.get(r_on["seed"])
        if r_off is None:
            return False
        return float(r_on.get(HARM, 0.0)) <= float(r_off.get(HARM, 0.0)) + SAFETY_HARM_TOL
    safety_seeds_ok = _n_seeds(on, _safe_vs_off)
    safety_pass = bool(safety_seeds_ok >= MIN_SEEDS_FOR_PASS)

    # Informational secondary: graded DN envelope vs hard top-k F-prefix.
    def _on_above_off(r_on: Dict[str, Any]) -> bool:
        r_off = off_by_seed.get(r_on["seed"])
        if r_off is None:
            return False
        return float(r_on.get(SENT, 0.0)) > float(r_off.get(SENT, 0.0))
    on_above_off_seeds = _n_seeds(on, _on_above_off)

    # Instrument-defect magnitude: how far the 689d hold-weighted DV would have
    # been from the repaired one on this run.
    occupancy_delta_by_arm = {
        a: round(_mean_key(rows, "occupancy_minus_fresh_entropy_delta"), 6)
        for a, rows in (
            (PROPOSER_CTRL_ARM, proposer), (MATCHED_NOISE_ARM, noise),
            (OFF_ARM, off), (PRIMARY_ARM, on),
        )
    }

    all_arms = [proposer, noise, off, on]
    non_degenerate = bool(
        all(len(a) > 0 for a in all_arms)
        and all(int(r.get("n_fresh_select", 0)) > 0 for a in all_arms for r in a)
        and substrate_invariant
        and control_distinct
    )
    degeneracy_reason = ""
    if not substrate_invariant:
        degeneracy_reason = (
            f"intra-run substrate divergence: {len(distinct_hashes)} distinct "
            f"substrate_hash values across cells ({', '.join(h[:12] for h in distinct_hashes)}) "
            "-- the arms are not mutually controlled"
        )
    elif not control_distinct:
        degeneracy_reason = (
            f"{len(identical_pairs)} control-arm pair(s) produced identical committed-class "
            "count vectors -- a conjunctive 'strict above BOTH' criterion would silently "
            "degrade to 'strict above ONE'"
        )

    # === VERDICT CHAIN ========================================================
    # The three NEW gates precede everything, so a broken instrument can never
    # reach a scientific verdict -- the 689d failure mode.
    if not substrate_invariant:
        label = "intra_run_substrate_divergence_invalid"
        overall_pass = False
        evidence_direction = "non_contributory"
    elif not control_distinct:
        label = "control_arms_not_distinct_invalid"
        overall_pass = False
        evidence_direction = "non_contributory"
    elif not fresh_sufficient:
        label = "substrate_not_ready_requeue"
        overall_pass = False
        evidence_direction = "non_contributory"
    elif not readiness_ok:
        label = "substrate_not_ready_requeue"
        overall_pass = False
        evidence_direction = "non_contributory"
    elif not matched_noise_verified_lifting:
        label = "matched_noise_control_unmeetable"
        overall_pass = False
        evidence_direction = "non_contributory"
    elif not rank_preserving_pass:
        label = "rank_alteration_not_prefix_diagnose"
        overall_pass = False
        evidence_direction = "non_contributory"
    elif not primary_pass:
        label = "conversion_ceiling_persists_despite_demotion"
        overall_pass = False
        evidence_direction = "non_contributory"
    elif not safety_pass:
        label = "demotion_disinhibits_harmful_classes"
        overall_pass = False
        evidence_direction = "weakens"
    else:
        label = "demotion_converts_committed_diversity"
        overall_pass = True
        evidence_direction = "supports"

    return {
        "c_substrate_invariant": {
            "n_distinct_substrate_hashes": int(len(distinct_hashes)),
            "distinct_substrate_hashes": distinct_hashes,
            "n_cells": int(len(arm_results)),
            "n_cells_with_hash": int(len(hashes)),
            "c_substrate_invariant_pass": substrate_invariant,
            "note": (
                "NEW GATE (defect 3, failure_autopsy_V3-EXQ-689d sec 4). Every cell "
                "must have run on ONE substrate build. In 689d ARM_ON seeds 43/44 ran "
                "on fc6d17ce... while every control and ARM_ON seed 42 ran on "
                "19b4073c... (ree_core edited on the Mac mid-run), so C_PRIMARY had "
                "ZERO validly-controlled seeds. Cardinality > 1 => the arms are not "
                "mutually controlled => the run is invalid, independent of every other "
                "criterion. A top-level-only substrate_hash would have HIDDEN this."
            ),
        },
        "c_control_distinct": {
            "identical_control_pairs": identical_pairs,
            "n_identical_control_pairs": int(len(identical_pairs)),
            "control_arms_checked": CONTROL_ARMS,
            "c_control_distinct_pass": control_distinct,
            "note": (
                "NEW GATE (defect 2, autopsy sec 3 + Learning 1). No two control arms "
                "may produce identical committed-class count vectors on the same seed. "
                "In 689d ARM_MATCHED_NOISE was bit-identical to ARM_PROPOSER_CTRL on "
                "every metric on all three seeds -- temperature is inert on the "
                "deterministic-argmin committed path -- so 'strict above BOTH X and Y' "
                "degraded SILENTLY to 'strict above X'. A conjunctive criterion over "
                "multiple controls REQUIRES an explicit control-distinctness assertion."
            ),
        },
        "c_fresh_sufficient": {
            "fresh_select_target": int(N_FRESH_SELECT_TARGET),
            "min_fresh_select_per_cell": int(MIN_FRESH_SELECT_PER_CELL),
            "observed_min_n_fresh_select": int(min_fresh),
            "cells_below_floor": cells_short,
            "n_cells_below_floor": int(len(cells_short)),
            "mean_fresh_select_yield": round(_mean_key(all_rows, "fresh_select_yield"), 6),
            "mean_replication_factor": round(_mean_key(all_rows, "replication_factor"), 6),
            "c_fresh_sufficient_pass": fresh_sufficient,
            "note": (
                "NEW GATE (defect 1 + power, autopsy sec 2 + routing item 4). EFFECTIVE "
                "N, not row count, is the sample size for a shape statistic: at cadence "
                "~10, 689d's ARM_ON seed 44 had 238 env steps = ~24 genuine selections "
                "against a 0.187-nat deciding margin. Every cell must bank "
                "N_FRESH_SELECT_TARGET genuine E3 selections; P1 stops early once met, "
                "which also equalises N across arms and removes the 689d 7.0x exposure "
                "asymmetry BY CONSTRUCTION. Short of the floor self-routes "
                "substrate_not_ready_requeue, NEVER a verdict."
            ),
        },
        "readiness": {
            "route_range_floor": ROUTE_RANGE_FLOOR,
            "on_route_range_mean": round(on_route_mean, 6),
            "on_seeds_route_above_floor": int(route_seeds_ok),
            "route_ready": route_ready,
            "c1_pairwise_dist_floor": C1_PAIRWISE_DIST_FLOOR,
            "on_pairwise_dist_mean": round(on_pdist_mean, 6),
            "on_seeds_e2_divergent": int(c1_seeds_ok),
            "c1_pass": c1_pass,
            "non_degeneracy": {
                "demotion_active_frac_floor": DEMOTION_ACTIVE_FRAC_FLOOR,
                "excluded_count_floor": EXCLUDED_COUNT_FLOOR,
                "on_demotion_active_frac_mean": round(_mean_key(on, "f_eligibility_demotion_active_frac"), 6),
                "on_excluded_count_mean": round(_mean_key(on, "f_eligibility_excluded_count_mean"), 6),
                "on_envelope_size_mean": round(_mean_key(on, "f_eligibility_envelope_size_mean"), 6),
                "on_seeds_non_degenerate": int(non_degen_seeds_ok),
                "non_degeneracy_ready": non_degeneracy_ready,
                "note": (
                    "PREDICATE UNCHANGED from 689d (EXCLUDED_COUNT_FLOOR=0.0, strict >) "
                    "-- adjudicated SURVIVING on threshold-invariance by the autopsy "
                    "(sec 5). Only the instrument is repaired: the demotion diagnostics "
                    "are now read on genuine selections, and the active_frac denominator "
                    "is n_fresh_select rather than n_p1_ticks."
                ),
            },
            "min_seeds_required": MIN_SEEDS_FOR_PASS,
            "readiness_ok": readiness_ok,
        },
        "c_rank_preserving": {
            "on_seeds_rank_preserving": int(rank_seeds_ok),
            "on_rank_preserving_frac_mean": round(_mean_key(on, "f_eligibility_rank_preserving_frac"), 6),
            "rank_preserving_frac_required": RANK_PRESERVING_FRAC_REQUIRED,
            "c_rank_preserving_pass": rank_preserving_pass,
            "note": (
                "PREDICATE UNCHANGED from 689d -- adjudicated SURVIVING (measured "
                "exactly 1.0; a saturated fraction has nowhere to move under "
                "reweighting). ARM_ON eligible set is an F-rank PREFIX on every active "
                "tick. Fail = the demotion ALTERED the F-rank (global F-flatten) -> "
                "rank_alteration_not_prefix_diagnose, an impl/design fault routed to "
                "/diagnose-errors, NOT a weakens."
            ),
        },
        "c_primary": {
            "primary_dv_key": SENT,
            "uncorrected_dv_key": SENT_PLUGIN,
            "estimator": "miller_madow",
            "on_seeds_strict_above_both_collapsed_controls": int(primary_seeds_ok),
            "on_selected_entropy_mean": round(on_sel_mean, 6),
            "selected_entropy_floor": C3_SELECTED_ENTROPY_FLOOR,
            "primary_floor_ok": primary_floor_ok,
            "c_primary_pass": primary_pass,
            "estimator_note": (
                "C_PRIMARY GATES ON THE MILLER-MADOW-CORRECTED ESTIMATOR "
                "(selected_action_class_entropy_mm). The uncorrected plug-in value "
                "is emitted per cell as selected_action_class_entropy and the "
                "counterfactual verdict under it is recorded in "
                "summary.miller_madow_audit. C3_SELECTED_ENTROPY_FLOOR is "
                "DELIBERATELY UNCHANGED at its 689d value -- the correction is "
                "<=0.01 nats at N=200 and re-tuning the floor to absorb it would "
                "silently re-tune a threshold the autopsy adjudicated as SURVIVING "
                "on threshold-invariance."
            ),
            "note": (
                "THE REPAIRED CRITERION. Predicate UNCHANGED from 689d (ARM_ON "
                "committed-class entropy STRICTLY ABOVE BOTH ARM_PROPOSER_CTRL and "
                "ARM_MATCHED_NOISE on the same seed, plus an absolute floor), but all "
                "three grounds of its 689d invalidity are now closed: the DV counts "
                "GENUINE selections (not hold-weighted env steps), the noise control is "
                "reachable and its lift is GATING, and cell substrate invariance is "
                "asserted. Fail = no lift -> conversion_ceiling_persists_despite_demotion "
                "(MECH-449 / V4 off-ramp; NOT a falsification)."
            ),
        },
        "miller_madow_audit": {
            "primary_dv_key": SENT,
            "uncorrected_dv_key": SENT_PLUGIN,
            "correction_nats_by_cell": {
                f"{r['arm_id']}|{r['seed']}": float(r.get("miller_madow_correction_nats", 0.0))
                for r in all_rows
            },
            "max_correction_nats": float(
                max([r.get("miller_madow_correction_nats", 0.0) for r in all_rows] or [0.0])
            ),
            "max_within_seed_correction_spread_nats": float(max([
                max([r.get("miller_madow_correction_nats", 0.0)
                     for r in all_rows if r["seed"] == sd])
                - min([r.get("miller_madow_correction_nats", 0.0)
                       for r in all_rows if r["seed"] == sd])
                for sd in {r["seed"] for r in all_rows}
            ] or [0.0])),
            "fresh_select_target": int(N_FRESH_SELECT_TARGET),
            "selected_entropy_floor": C3_SELECTED_ENTROPY_FLOOR,
            # Counterfactual verdict under the uncorrected estimator.
            "plugin_on_seeds_strict_above_both": int(primary_seeds_ok_plugin),
            "plugin_on_selected_entropy_mean": round(on_sel_mean_plugin, 6),
            "plugin_c_primary_pass": primary_pass_plugin,
            "estimator_changes_verdict": estimator_changes_verdict,
            "note": (
                "max_within_seed_correction_spread_nats is the quantity that matters: "
                "it is the residual arm-dependent bias the correction removes, and it "
                "must be read against the deciding margin. This is the SECOND of the "
                "two fixes the LANDED sibling V3-EXQ-699c adopted for this DV class in "
                "the same 699/689 defect lineage. 699c's own docstring calls the "
                "FIXED-N stopping rule 'the load-bearing fix' (it makes the differential "
                "bias zero BY CONSTRUCTION) and Miller-Madow 'belt and braces' for the "
                "residual. 689i ALREADY CARRIED the load-bearing half -- every cell "
                "banks exactly N_FRESH_SELECT_TARGET genuine selections and P1 stops at "
                "the target -- so N is equal across cells and only the K_obs-driven term "
                "remained. At N=200 that residual is (K_obs-1)/400: 0.0025 nats at "
                "K_obs=2 up to 0.0100 at K_obs=5, so the worst plausible arm "
                "differential (K_obs=5 noise vs K_obs=2 ON, the occupancy split seen in "
                "the 689i dry-run smoke) is ~0.0075 nats -- ~4% of the 689d deciding "
                "margin (0.187 nats) and 2.5% of C3_SELECTED_ENTROPY_FLOOR. So this is "
                "a RESIDUAL CORRECTION, not a defect repair: it is real and points in "
                "the direction C_PRIMARY tests, but it is not close to decisive alone. "
                "estimator_changes_verdict is the direct check -- if TRUE on a real run, "
                "the verdict rests on a <=0.01-nat correction and MUST be flagged in "
                "review rather than read as a clean PASS."
            ),
        },
        "c_safety": {
            "on_seeds_safe_vs_off": int(safety_seeds_ok),
            "on_harm_per_tick_mean": round(_mean_key(on, HARM), 6),
            "off_harm_per_tick_mean": round(_mean_key(off, HARM), 6),
            "safety_harm_tol": SAFETY_HARM_TOL,
            "c_safety_pass": safety_pass,
            "note": (
                "PREDICATE UNCHANGED from 689d -- adjudicated SURVIVING because it reads "
                "realized per-env-step harm from env.step, and the ENV STEP IS the "
                "correct sampling unit for a realized-harm RATE. Harm accumulation is "
                "therefore DELIBERATELY NOT fresh-gated here: hold-weighting is the "
                "intended denominator, not a defect. Fail = the lift admits harmful "
                "classes -> demotion_disinhibits_harmful_classes (the ONLY weakens)."
            ),
        },
        "matched_noise_gating": {
            "matched_noise_lift_seeds_over_proposer": int(noise_lift_seeds),
            "matched_noise_verified_lifting": matched_noise_verified_lifting,
            "min_seeds_required": MIN_SEEDS_FOR_PASS,
            "noise_arm_gap_scaled_commit_active_frac_mean": round(
                _mean_key(noise, "gap_scaled_commit_active_frac"), 6),
            "note": (
                "NOW GATING (was informational in 689d, where it fired `false` and the "
                "run PASSED anyway -- autopsy Learning 1: 'a pre-registered guard that "
                "does not GATE is not a guard'). The noise control must VERIFIABLY LIFT "
                "committed entropy over the collapsed proposer baseline, else beating it "
                "is not a real bar and the run self-routes "
                "matched_noise_control_unmeetable (non_contributory). The arm is now "
                "instantiated on Factor B's gap-scaled stochastic commit, where the "
                "sampling step is LIVE on the committed path; "
                "gap_scaled_commit_active_frac confirms it actually fired."
            ),
        },
        "instrument_defect_magnitude": {
            "occupancy_minus_fresh_entropy_delta_by_arm": occupancy_delta_by_arm,
            "occupancy_entropy_by_arm_mean": {
                PROPOSER_CTRL_ARM: round(_mean_key(proposer, "occupancy_class_entropy"), 6),
                MATCHED_NOISE_ARM: round(_mean_key(noise, "occupancy_class_entropy"), 6),
                OFF_ARM: round(_mean_key(off, "occupancy_class_entropy"), 6),
                PRIMARY_ARM: round(_mean_key(on, "occupancy_class_entropy"), 6),
            },
            "mean_replication_factor": round(_mean_key(all_rows, "replication_factor"), 6),
            "note": (
                "INFORMATIONAL (NON-GATING). 689d's hold-weighted DV is retained here "
                "verbatim as occupancy_class_entropy alongside the repaired "
                "per-commitment DV, so the SIZE and SIGN of the instrument defect is "
                "measured on this substrate rather than merely asserted. A large "
                "arm-varying delta is direct evidence that the 689d readout was "
                "reweighted by hold duration."
            ),
        },
        "proposer_ceiling_reference": {
            "proposer_ctrl_pool_entropy_mean": round(_mean_key(proposer, "proposer_pool_class_entropy"), 6),
            "on_pool_entropy_mean": round(_mean_key(on, "proposer_pool_class_entropy"), 6),
            "on_selected_entropy_mean": round(on_sel_mean, 6),
            "proposer_ceiling_frac_target": PROPOSER_CEILING_FRAC,
            "on_reaches_proposer_ceiling": bool(
                _mean_key(proposer, "proposer_pool_class_entropy") > 1e-6
                and on_sel_mean >= PROPOSER_CEILING_FRAC * _mean_key(proposer, "proposer_pool_class_entropy")
            ),
            "note": (
                "INFORMATIONAL (NON-GATING). The proposer ceiling = the pooled candidate "
                "first-action class entropy the SP-CEM proposer OFFERED, now also "
                "fresh-gated (689d accumulated it per env step at :535 off CACHED "
                "candidates -- the same defect as the primary DV)."
            ),
        },
        "demotion_vs_hard_topk_secondary": {
            "on_seeds_above_off": int(on_above_off_seeds),
            "on_selected_entropy_mean": round(on_sel_mean, 6),
            "off_selected_entropy_mean": round(_mean_key(off, SENT), 6),
            "note": (
                "SECONDARY (informational, NON-GATING): does the graded DN envelope beat "
                "the HARD env-conditional top-k F-prefix (the 569i lever)? C_PRIMARY "
                "gates on the proposer ceiling per the design-note s4 acceptance, NOT "
                "on this."
            ),
        },
        "selected_action_entropy_per_arm_mean": {
            PROPOSER_CTRL_ARM: round(_mean_key(proposer, SENT), 6),
            MATCHED_NOISE_ARM: round(_mean_key(noise, SENT), 6),
            OFF_ARM: round(_mean_key(off, SENT), 6),
            PRIMARY_ARM: round(on_sel_mean, 6),
        },
        "fresh_select_by_arm_mean": {
            PROPOSER_CTRL_ARM: round(_mean_key(proposer, "n_fresh_select"), 2),
            MATCHED_NOISE_ARM: round(_mean_key(noise, "n_fresh_select"), 2),
            OFF_ARM: round(_mean_key(off, "n_fresh_select"), 2),
            PRIMARY_ARM: round(_mean_key(on, "n_fresh_select"), 2),
        },
        "route_range_per_arm_mean": {
            PROPOSER_CTRL_ARM: round(_mean_key(proposer, RDIST), 6),
            MATCHED_NOISE_ARM: round(_mean_key(noise, RDIST), 6),
            OFF_ARM: round(_mean_key(off, RDIST), 6),
            PRIMARY_ARM: round(on_route_mean, 6),
        },
        "harm_per_tick_per_arm_mean": {
            PROPOSER_CTRL_ARM: round(_mean_key(proposer, HARM), 6),
            MATCHED_NOISE_ARM: round(_mean_key(noise, HARM), 6),
            OFF_ARM: round(_mean_key(off, HARM), 6),
            PRIMARY_ARM: round(_mean_key(on, HARM), 6),
        },
        "label": label,
        "evidence_direction": evidence_direction,
        "overall_pass": overall_pass,
        "preconditions": [
            {
                "name": "cells_share_one_substrate_hash",
                "kind": "readiness",
                "description": (
                    "The set of per-cell arm_fingerprint.substrate_hash values across "
                    "ALL cells has cardinality exactly 1 -- every arm and seed ran on "
                    "the SAME substrate build, so the ON-vs-control contrast is a "
                    "controlled comparison. The 689d defect-3 gate."
                ),
                "control": "arm_fingerprint.substrate_hash stamped per (arm, seed) cell",
                "measured": int(len(distinct_hashes)),
                "threshold": 1,
                "direction": "upper",
                "comparator": "<=",
                "met": substrate_invariant,
            },
            {
                "name": "control_arms_produce_distinct_class_histograms",
                "kind": "readiness",
                "description": (
                    "No two control arms produce an identical committed-class count "
                    "vector on the same seed. Guards the silent degradation of the "
                    "conjunctive C_PRIMARY criterion ('strict above BOTH') to a single "
                    "bar when two controls coincide. The 689d defect-2 gate."
                ),
                "control": (
                    "pairwise comparison of selected_class_counts across "
                    "ARM_PROPOSER_CTRL / ARM_MATCHED_NOISE / ARM_OFF, per seed"
                ),
                "measured": int(len(identical_pairs)),
                "threshold": 0,
                "direction": "upper",
                "comparator": "<=",
                "met": control_distinct,
            },
            {
                "name": "fresh_e3_selection_sufficiency_all_cells",
                "kind": "readiness",
                "description": (
                    "Every cell banked at least MIN_FRESH_SELECT_PER_CELL GENUINE E3 "
                    "selections, measured by the substrate-inert sentinel key. Effective "
                    "N, not env-step count, is the sample size for a class-histogram "
                    "entropy. The 689d defect-1 / power gate."
                ),
                "control": (
                    "sentinel-key freshness marker on agent.e3.last_score_diagnostics, "
                    "which e3_selector.select() reassigns wholesale at :2452"
                ),
                "measured": int(min_fresh),
                "threshold": int(MIN_FRESH_SELECT_PER_CELL),
                "direction": "lower",
                "comparator": ">=",
                "observed_mean_fresh_select_yield": round(
                    _mean_key(all_rows, "fresh_select_yield"), 6),
                "observed_mean_replication_factor": round(
                    _mean_key(all_rows, "replication_factor"), 6),
                "met": fresh_sufficient,
            },
            {
                "name": "matched_noise_control_verifiably_lifts",
                "kind": "readiness",
                "description": (
                    "The noise negative control raises committed-class entropy over the "
                    "collapsed proposer baseline on >= MIN_SEEDS_FOR_PASS seeds, so "
                    "'ARM_ON beats it' is a REAL bar. Instantiated on Factor B's "
                    "gap-scaled stochastic commit, where the sampling step is live on "
                    "the committed path. GATING here; informational in 689d, where it "
                    "fired false and the run passed anyway."
                ),
                "control": (
                    "ARM_MATCHED_NOISE (use_gap_scaled_commit_temperature=True) vs "
                    "ARM_PROPOSER_CTRL, same seed"
                ),
                "measured": int(noise_lift_seeds),
                "threshold": int(MIN_SEEDS_FOR_PASS),
                "direction": "lower",
                "comparator": ">=",
                "met": matched_noise_verified_lifting,
            },
            {
                "name": "on_modulatory_channel_route_range_supra_floor",
                "kind": "readiness",
                "description": (
                    "ARM_ON IN-ARM RAW cross-candidate RANGE of the modulatory bias "
                    "routed into the E3 selection authority "
                    "(modulatory_channel_route_range), read on GENUINE selections. SAME "
                    "range statistic the route-range substrate gates on. Below floor => "
                    "routing not wired / e2 under-trained => substrate_not_ready_requeue."
                ),
                "control": (
                    "ARM_ON: candidate_summary_source=e2_world_forward, route-range "
                    "routing + shortlist-then-modulate + f_eligibility demotion ON"
                ),
                "measured": round(on_route_mean, 6),
                "threshold": ROUTE_RANGE_FLOOR,
                "direction": "lower",
                "met": route_ready,
            },
            {
                "name": "on_e2_world_forward_prediction_spread_supra_floor",
                "kind": "readiness",
                "description": (
                    "ARM_ON e2.world_forward(z0, a_i) per-candidate prediction spread "
                    "(cand_world_pairwise_dist) clears the floor -- SD-056 trained the "
                    "action-conditional divergence the eligible set needs. RANGE "
                    "statistic, read on genuine selections. Below floor => SD-056 "
                    "under-trained => substrate_not_ready_requeue."
                ),
                "control": "ARM_ON: agent.e2.cand_world_pairwise_dist on SP-CEM candidates",
                "measured": round(on_pdist_mean, 6),
                "threshold": C1_PAIRWISE_DIST_FLOOR,
                "direction": "lower",
                "met": c1_pass,
            },
            {
                "name": "on_f_eligibility_envelope_excludes_on_divergent_pool",
                "kind": "readiness",
                "description": (
                    "ARM_ON f_eligibility demotion is active on >= "
                    "DEMOTION_ACTIVE_FRAC_FLOOR of GENUINE selections AND the envelope "
                    "ACTUALLY EXCLUDES (mean f_eligibility_excluded_count > floor) on "
                    "the divergent e2_world_forward pool. An all-admit envelope "
                    "(excluded_count==0, a flat-F pool) is vacuous => "
                    "substrate_not_ready_requeue. Threshold UNCHANGED from 689d "
                    "(0.0, strict >) -- adjudicated threshold-invariant."
                ),
                "control": (
                    "ARM_ON: use_f_eligibility_demotion=True over the divergent pool; "
                    "f_eligibility_excluded_count read on genuine selections"
                ),
                "measured": round(_mean_key(on, "f_eligibility_excluded_count_mean"), 6),
                "threshold": EXCLUDED_COUNT_FLOOR,
                "direction": "lower",
                "met": non_degeneracy_ready,
            },
        ],
        "criteria": [
            {"name": "C_SUBSTRATE_INVARIANT_cells_share_one_build", "load_bearing": True,
             "passed": substrate_invariant},
            {"name": "C_CONTROL_DISTINCT_no_identical_control_histograms", "load_bearing": True,
             "passed": control_distinct},
            {"name": "C_FRESH_SUFFICIENT_effective_n_meets_target", "load_bearing": True,
             "passed": fresh_sufficient},
            {"name": "C_READINESS_e2_divergent_envelope_excludes", "load_bearing": True,
             "passed": readiness_ok},
            {"name": "C_NOISE_LIFTS_matched_noise_verifiably_lifting", "load_bearing": True,
             "passed": matched_noise_verified_lifting},
            {"name": "C_RANK_PRESERVING_eligible_set_is_F_rank_prefix", "load_bearing": True,
             "passed": rank_preserving_pass},
            {"name": "C_PRIMARY_on_selected_entropy_strict_above_both_collapsed_controls",
             "load_bearing": True, "passed": primary_pass},
            {"name": "C_SAFETY_on_harm_not_above_off", "load_bearing": True,
             "passed": safety_pass},
        ],
        "criteria_non_degenerate": {
            "C_SUBSTRATE_INVARIANT": non_degenerate,
            "C_CONTROL_DISTINCT": non_degenerate,
            "C_FRESH_SUFFICIENT": non_degenerate,
            "C_READINESS": non_degenerate,
            "C_NOISE_LIFTS": non_degenerate,
            "C_RANK_PRESERVING": non_degenerate,
            "C_PRIMARY": non_degenerate,
            "C_SAFETY": non_degenerate,
        },
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    p0 = DRY_RUN_P0 if dry_run else P0_WARMUP_EPISODES
    p1_cap = DRY_RUN_P1_CAP if dry_run else P1_EPISODE_CAP
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE
    fresh_target = DRY_RUN_FRESH_TARGET if dry_run else N_FRESH_SELECT_TARGET

    full_config: Dict[str, Any] = {
        "seeds": seeds,
        "p0_warmup_episodes": p0,
        "p1_episode_cap": p1_cap,
        "steps_per_episode": steps,
        "n_fresh_select_target": fresh_target,
        "env_kwargs": ENV_KWARGS,
        "arms": [
            {k: a[k] for k in (
                "arm_id", "label", "candidate_summary_source", "temperature",
                "use_f_eligibility_demotion", "use_gap_scaled_commit_temperature",
            )}
            for a in ARMS
        ],
        "sd056_weight": SD056_WEIGHT,
        "noise_arm_commit_entropy_alpha": NOISE_ARM_COMMIT_ENTROPY_ALPHA,
        "f_eligibility_config": {
            "f_eligibility_envelope_floor": F_ELIGIBILITY_ENVELOPE_FLOOR,
            "f_eligibility_dn_sigma": F_ELIGIBILITY_DN_SIGMA,
        },
        "conversion_constant": {
            "use_modulatory_channel_routing": True,
            "modulatory_channel_route_source": "cand_world_summary",
            "use_modulatory_selection_authority": True,
            "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
            "modulatory_authority_normalize_basis": MODULATORY_AUTHORITY_NORMALIZE_BASIS,
            "use_modulatory_shortlist_then_modulate": True,
            "modulatory_shortlist_mode": MODULATORY_SHORTLIST_MODE,
            "modulatory_shortlist_k": MODULATORY_SHORTLIST_K,
        },
        "thresholds": {
            "route_range_floor": ROUTE_RANGE_FLOOR,
            "c1_pairwise_dist_floor": C1_PAIRWISE_DIST_FLOOR,
            "excluded_count_floor": EXCLUDED_COUNT_FLOOR,
            "demotion_active_frac_floor": DEMOTION_ACTIVE_FRAC_FLOOR,
            "rank_preserving_frac_required": RANK_PRESERVING_FRAC_REQUIRED,
            "c3_selected_entropy_floor": C3_SELECTED_ENTROPY_FLOOR,
            "safety_harm_tol": SAFETY_HARM_TOL,
            "min_seeds_for_pass": MIN_SEEDS_FOR_PASS,
            "min_fresh_select_per_cell": MIN_FRESH_SELECT_PER_CELL,
        },
    }

    arm_results: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm['arm_id']}", flush=True)
            cell = _run_seed_arm(arm, seed, p0, p1_cap, steps, fresh_target)
            cell["arm_fingerprint"] = compute_arm_fingerprint(
                config_slice={
                    "arm": {
                        k: arm[k]
                        for k in (
                            "arm_id", "candidate_summary_source", "temperature",
                            "use_f_eligibility_demotion",
                            "use_gap_scaled_commit_temperature",
                        )
                    },
                    "env_kwargs": ENV_KWARGS,
                    "sd056_weight": SD056_WEIGHT,
                    "noise_arm_commit_entropy_alpha": NOISE_ARM_COMMIT_ENTROPY_ALPHA,
                    "f_eligibility_config": {
                        "f_eligibility_envelope_floor": F_ELIGIBILITY_ENVELOPE_FLOOR,
                        "f_eligibility_dn_sigma": F_ELIGIBILITY_DN_SIGMA,
                    },
                    "conversion_constant": {
                        "use_modulatory_channel_routing": True,
                        "modulatory_channel_route_source": "cand_world_summary",
                        "use_modulatory_selection_authority": True,
                        "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
                        "modulatory_authority_normalize_basis": MODULATORY_AUTHORITY_NORMALIZE_BASIS,
                        "use_modulatory_shortlist_then_modulate": True,
                        "modulatory_shortlist_mode": MODULATORY_SHORTLIST_MODE,
                        "modulatory_shortlist_k": MODULATORY_SHORTLIST_K,
                    },
                    "p0_episodes": p0, "p1_episode_cap": p1_cap,
                    "steps_per_episode": steps,
                    "n_fresh_select_target": fresh_target,
                },
                seed=seed,
                script_path=Path(__file__),
                rng_fully_reset=True,
                config_slice_declared=True,
                extra_ineligible_reasons=["online_e2_training_stateful_per_cell"],
            )
            arm_results.append(cell)
            passed = cell.get("error_note") is None
            print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)

    summary = _evaluate(arm_results)
    outcome = "PASS" if summary["overall_pass"] else "FAIL"
    evidence_direction = summary["evidence_direction"]

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    manifest: Dict[str, Any] = {
        "schema_version": "v1",
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": outcome,
        "result": outcome,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "supersedes": SUPERSEDES,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": {"MECH-448": evidence_direction},
        "non_degenerate": summary.get("non_degenerate", True),
        "degeneracy_reason": summary.get("degeneracy_reason", ""),
        "c_primary_estimator": {
            "primary_dv_key": "selected_action_class_entropy_mm",
            "uncorrected_dv_key": "selected_action_class_entropy",
            "estimator": "miller_madow",
            "formula": "H_MM = H_plugin + (K_obs - 1) / (2N)",
            "note": (
                "C_PRIMARY gates on the MILLER-MADOW-corrected estimator. Ported from "
                "the landed sibling V3-EXQ-699c (same DV class, same 699/689 defect "
                "lineage). RESIDUAL correction only: 689i already carries 699c's "
                "load-bearing FIXED-N fix (every cell banks N_FRESH_SELECT_TARGET=200 "
                "genuine selections), so N is equal across cells by construction and "
                "only the K_obs-driven term remained -- (K_obs-1)/400, i.e. 0.0025 to "
                "0.0100 nats, worst plausible arm differential ~0.0075 nats against a "
                "0.187-nat deciding margin. C3_SELECTED_ENTROPY_FLOOR is NOT re-tuned "
                "to compensate; C_READINESS / C_RANK_PRESERVING / C_SAFETY predicates "
                "are untouched. occupancy_class_entropy stays PLUG-IN to remain "
                "byte-comparable with 689d's recorded hold-weighted DV. See "
                "summary.miller_madow_audit for per-cell corrections and the "
                "counterfactual verdict under the uncorrected estimator."
            ),
        },
        "evidence_direction_note": (
            "MECH-448 (ARC-107) rank-preserving F->eligibility demotion falsifier -- "
            "INSTRUMENT REPAIR of V3-EXQ-689d, whose PASS was withdrawn 2026-07-20 by "
            "failure_autopsy_V3-EXQ-689d_2026-07-20 (non_contributory / "
            "measurement_test_design_defect; MECH-448 reverted provisional -> "
            "candidate, genuine_exp_count 1 -> 0). SAME QUESTION, repaired instrument "
            "-- the sanctioned 785 -> 785a / 708 -> 708a shape; the re-derive brake is "
            "recorded NOT FIRED (autopsy sec 10, user-ratified) because the category is "
            "measurement_test_design_defect, NOT substrate_ceiling, and the substrate is "
            "demonstrably built. THREE INDEPENDENT DEFECTS CLOSED, all of which had to "
            "be fixed together because defects 2 and 3 survive a DV repair: "
            "(1) HOLD-WEIGHTED DV -> every per-selection quantity (committed class, "
            "proposer pool, pairwise dist, all E3 diagnostics) is gated on a GENUINE E3 "
            "selection via a substrate-inert sentinel key (699b mechanism, chosen over "
            "the 785a `= None` clear because nulling _last_selected_trajectory perturbs "
            "post_action_update); n_fresh_select / n_latched / fresh_select_yield / "
            "replication_factor emitted per cell; the 689d hold-weighted DV retained as "
            "a named secondary (occupancy_class_entropy) so the defect magnitude is "
            "measured. (2) VACUOUS CONTROL -> ARM_MATCHED_NOISE was bit-identical to "
            "ARM_PROPOSER_CTRL (temperature is inert on the deterministic-argmin "
            "committed path); it is re-instantiated on Factor B gap-scaled stochastic "
            "commit where the sampling step is LIVE on the committed path, a hard "
            "control-distinctness assertion blocks any identical control pair, and "
            "matched_noise_verified_lifting is now GATING rather than informational. "
            "(3) INTRA-RUN SUBSTRATE DIVERGENCE -> per-cell substrate_hash cardinality "
            "must be exactly 1 or the run is invalid; pinned to a cloud worker rather "
            "than the Mac's live checkout. Plus POWER: 4 seeds (was 3), 3-of-4 to pass "
            "(was 2-of-3), and P1 runs until each cell banks 200 GENUINE selections, "
            "which also removes the 689d 7.0x arm-exposure asymmetry by construction. "
            "The predicates of C_READINESS (0.0 floor, strict >), C_RANK_PRESERVING "
            "(saturated 1.0) and C_SAFETY (realized per-env-step harm -- deliberately "
            "NOT fresh-gated, the env step being the correct denominator for a harm "
            "rate) are carried over UNCHANGED, per the autopsy's separate adjudication "
            "that all three survive on threshold-invariance. PASS "
            "(demotion_converts_committed_diversity) restores the embodied env-loop "
            "pathway that the 689d withdrawal removed -- 689h survives but is "
            "purpose:diagnostic, tests demotion_x_gonogo_additive rather than the "
            "standalone conversion assertion, and shares the synthetic-bank pathway "
            "with 689g. Only a C_SAFETY fail WEAKENS. claim_ids=[MECH-448] only; "
            "ARC-107/MECH-447/MECH-449/MECH-439/Q-078 untouched; MECH-448 stays "
            "candidate (PROMOTES NOTHING; governance applies)."
        ),
        "interpretation": {
            "label": summary["label"],
            "preconditions": summary["preconditions"],
            "criteria": summary["criteria"],
            "criteria_non_degenerate": summary["criteria_non_degenerate"],
            "routing": {
                "demotion_converts_committed_diversity": "PASS -> MECH-448 toward supports on a REPAIRED instrument; restores the embodied env-loop pathway (genuine_exp_count 0 -> 1); ARC-107 gains its first validly-measured lever",
                "intra_run_substrate_divergence_invalid": "cells did not share one substrate build -- the arms are not mutually controlled; the run is INVALID independent of every other criterion; re-run at a pinned commit on a cloud worker; NOT a weakens",
                "control_arms_not_distinct_invalid": "two control arms produced identical committed-class histograms, so the conjunctive C_PRIMARY criterion degenerates to a single bar (the 689d defect-2 signature); repair the control before any verdict is read; NOT a weakens",
                "substrate_not_ready_requeue": "insufficient GENUINE E3 selections per cell, OR routing/SD-056 under-trained, OR the pool is not divergent (all-admit envelope, excluded_count==0); re-queue; do NOT weaken MECH-448",
                "matched_noise_control_unmeetable": "the noise negative control did not verifiably lift committed entropy over the collapsed proposer baseline, so 'ARM_ON beats it' is not a real bar -- the 689d defect-2 failure mode, now GATING; repair the control; NOT a weakens",
                "rank_alteration_not_prefix_diagnose": "the eligible set is NOT an F-rank prefix (the envelope altered the F-rank) -> implementation/design fault -> /diagnose-errors; NOT a weakens",
                "conversion_ceiling_persists_despite_demotion": "OFF-RAMP -> the MECH-449 Go/No-Go eligibility constitution (double-gated) / the V4 directions; NOT a falsification of MECH-448",
                "demotion_disinhibits_harmful_classes": "WEAKENED -- the committed-entropy lift came from globally flattening F / admitting harmful classes (C_SAFETY fail), the design-note weaken condition",
            },
        },
        "dry_run": bool(dry_run),
        "config": full_config,
        "acceptance_criteria": {
            "C_SUBSTRATE_INVARIANT_cells_share_one_build": summary["c_substrate_invariant"]["c_substrate_invariant_pass"],
            "C_CONTROL_DISTINCT_no_identical_control_histograms": summary["c_control_distinct"]["c_control_distinct_pass"],
            "C_FRESH_SUFFICIENT_effective_n_meets_target": summary["c_fresh_sufficient"]["c_fresh_sufficient_pass"],
            "readiness_route_range_ready": summary["readiness"]["route_ready"],
            "readiness_e2_divergent_ready": summary["readiness"]["c1_pass"],
            "readiness_envelope_non_degenerate": summary["readiness"]["non_degeneracy"]["non_degeneracy_ready"],
            "readiness_substrate_ready": summary["readiness"]["readiness_ok"],
            "C_NOISE_LIFTS_matched_noise_verified_lifting": summary["matched_noise_gating"]["matched_noise_verified_lifting"],
            "C_RANK_PRESERVING_eligible_set_is_F_rank_prefix": summary["c_rank_preserving"]["c_rank_preserving_pass"],
            "C_PRIMARY_on_strict_above_both_collapsed_controls": summary["c_primary"]["c_primary_pass"],
            "C_SAFETY_on_harm_not_above_off": summary["c_safety"]["c_safety_pass"],
            "overall_pass": summary["overall_pass"],
        },
        "summary": summary,
        "arm_results": arm_results,
    }

    # Always-core stamp AFTER arm_results is assembled, so top-level
    # substrate_hash HOISTS from the per-cell fingerprints rather than being
    # recomputed (and mismatching). No-op-safe additive merge; write_flat_manifest
    # stamps again harmlessly.
    stamp_recording_core(
        manifest,
        config=full_config,
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t0,
    )

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=bool(dry_run),
        config=full_config,
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t0,
    )
    print(f"Manifest written: {out_path}", flush=True)
    print(f"Result written to: {out_path}", flush=True)

    print(
        f"Outcome: {outcome} (label={summary['label']}, "
        f"evidence_direction={evidence_direction})",
        flush=True,
    )
    for k, v in manifest["acceptance_criteria"].items():
        print(f"  {k}: {v}", flush=True)
    print(
        "  distinct_substrate_hashes: "
        f"{summary['c_substrate_invariant']['n_distinct_substrate_hashes']}",
        flush=True,
    )
    print(
        "  min_n_fresh_select: "
        f"{summary['c_fresh_sufficient']['observed_min_n_fresh_select']} "
        f"(target {MIN_FRESH_SELECT_PER_CELL})",
        flush=True,
    )

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "V3-EXQ-689i MECH-448 (ARC-107) rank-preserving F->eligibility demotion "
            "falsifier -- instrument repair of V3-EXQ-689d"
        )
    )
    parser.add_argument("--dry-run", action="store_true", help="Short smoke run.")
    args = parser.parse_args()

    result = run_experiment(dry_run=args.dry_run)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    _outcome = _outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL"
    emit_outcome(
        outcome=_outcome,
        manifest_path=str(result.get("manifest_path", Path("/dev/null"))),
        dry_run=args.dry_run,
    )
