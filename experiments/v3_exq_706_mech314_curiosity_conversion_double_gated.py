"""V3-EXQ-706: MECH-314 structured-curiosity conversion to committed-action-class
diversity DOUBLE-GATED -- the MECH-448/ARC-107 rank-preserving F->eligibility
demotion AND the BUILT MECH-449/ARC-107 Go/No-Go eligibility constitution BOTH ON.

SUPERSEDES V3-EXQ-705b. NEW number (NOT a 705 letter): this is the brake-EXEMPT
DOUBLE-GATED redesign routed by the confirmed, user-gated
failure_autopsy_V3-EXQ-705b_2026-06-25. 705b was the FIRST FAIR test of MECH-314
curiosity-conversion under a genuinely-excluding demotion (adaptive floor excluded
14.45 candidates vs 705's all-admit 0.0; 590c saturation-arm confound fixed; GAP-A
pool divergent ~0.107) and committed-class entropy STILL did not lift over the
demotion-ON F-only + matched-noise controls -> conversion_ceiling_persists_despite_
demotion. The re-derive brake FIRED (6th MECH-314 ceiling autopsy) and REFUSES a
705c demotion-ONLY same-substrate letter. The brake EXEMPTION is a DIFFERENT
substrate condition: compose the second, already-built eligibility gate.

=== THE SCIENTIFIC CHANGE (vs 705b) ===

705b ran with use_go_nogo_constitution OFF (only MECH-448 demotion + adaptive floor
active). 706 turns the BUILT MECH-449 Go/No-Go eligibility constitution ON
(use_go_nogo_constitution=True; e3_selector._go_nogo_eligibility_gate, landed
2026-06-21, falsifier V3-EXQ-689g PASS 3/3 conversion_rate 1.0; 689h showed
demotion x Go/No-Go compose ADDITIVELY, no interference). The autopsy diagnosis:
rank-preserving demotion is order-preserving over F and CANNOT, on its own, convert
the curiosity channel to committed authority -- the missing downstream dependency is
the ACTIVE No-Go (the 485k->MECH-449 OFC-devaluation shape: demotion-alone converts
a passive discrimination signature but cannot express the active No-Go).

=== WHY A FLAG-FLIP ALONE WOULD BE VACUOUS (substrate-readiness finding) ===

The Go/No-Go gate (_go_nogo_eligibility_gate) only SUPPRESSES candidates when it is
FED per-candidate No-Go signals (safety / staleness / perseveration / viability).
In the live agent.select_action path the gate sources signals from (a) the dACC
suppression bundle (MECH-260 reuse) -- but 705b/706 run use_dacc=False (curiosity is
the SOLE modulatory channel) so there is NO dACC bundle -- or (b) operator injection
via agent.set_injected_go_nogo_signals(). With dACC OFF and no injection, flipping
use_go_nogo_constitution=True is a STRUCTURAL NO-OP: the gate runs but suppresses
nothing, so the double-gated arm would be BIT-IDENTICAL to demotion-only 705b and
reproduce the no-lift vacuously (a silent vacuity masquerading as the terminal
off-ramp). This is the same trap the 705 all-admit confound burned a letter on.

=== THE ACTIVE No-Go (what makes the second gate genuinely fire) ===

706 injects an ACTIVE staleness/perseveration No-Go: each tick, the candidates whose
first-action class was COMMITTED within the last RECENCY_WINDOW ticks of the current
episode are flagged stale (staleness=STALENESS_NOGO_LEVEL >= gng_staleness_floor),
so the Go/No-Go gate drops them from the eligible set (subject to the fail-open
gng_protect_min_eligible=1 survivor guard). This is the canonical perseveration No-Go
axis named in 689g (the MECH-260 anti-recency suppression, generalised into an
eligibility-access gate by ARC-106 G2). It breaks the F-monostrategy lock and creates
within-eligible ROOM; whether MECH-314 curiosity CONVERTS that room into committed-
action-class diversity is exactly the test. CRITICAL: the No-Go is an ELIGIBILITY
gate, NOT a modulatory bias channel -- curiosity remains the SOLE modulatory channel
arbitrating the within-eligible argmin, so the 705b "curiosity sole modulatory
channel" design is preserved.

The SAME staleness-injection RULE runs on ALL THREE arms (a property of the double-
gated substrate condition, constant across arms). The contrast remains channel
PRESENCE: ARM_FONLY (curiosity=0) absorbs the staleness-gate-alone diversity baseline
(its within-eligible argmin is the F tie-break among non-stale survivors); ARM_CURIOSITY
must lift committed-class entropy OVER that double-gated F-only control to demonstrate
curiosity-specific conversion. (The realised per-tick suppression differs across arms
because each arm's own commit history drives its own staleness -- a faithful active
No-Go reacting to each arm's perseveration, exactly like a real anti-perseveration
gate.)

=== ARMS (3 arms x 3 seeds; ALL demotion ON + adaptive floor ON + Go/No-Go ON) ===

  ARM_CURIOSITY (PRIMARY) curiosity_novelty_weight=0.25, T=1.0
    The treatment: the MECH-314 novelty channel arbitrates the committed action
    within the doubly-gated eligibility set (F-demotion envelope minus the active
    staleness No-Go). 0.25 is the NON-saturation weight (590c range 0.0160 there;
    the clamp-saturation regime w=1.0 is NOT used -- 590c confounded-precondition fix).
  ARM_FONLY               curiosity_novelty_weight=0.0, T=1.0
    Double-gated F-only control: curiosity present (lever fires) but contributes ~0,
    so the within-eligible argmin is the F tie-break among non-stale survivors. The
    primary comparator -- it carries the staleness-gate-alone baseline diversity.
  ARM_NOISE               curiosity_novelty_weight=0.0, T=2.5
    Matched-noise control: identical to ARM_FONLY but a flat-hot uncommitted softmax
    temperature, so any committed-diversity it shows is SAMPLING noise. ARM_CURIOSITY
    must beat THIS (lawful curiosity access, not noise-as-diversity).

ALL arms share the 705b/648a/590c GAP-A-ready conversion stack verbatim: SP-CEM
Layer A (action-divergent pool) + V_s stack + SD-056 online contrastive
(e2.world_forward action-conditional divergence; rollout output-norm clamp) +
MECH-314 visitation-buffer novelty (curiosity_novelty_source="visitation",
first-action-onehot auto-augmentation) + curiosity_candidate_source="e2_world_forward"
(GAP-A divergent pool) + the modulatory-bias-selection-authority stack + MECH-448
demotion ON + use_f_eligibility_adaptive_floor ON. MECH-314 curiosity is the SOLE
modulatory channel (dacc / lateral_pfc / ofc / mech295 / tonic_vigor / noise_floor /
e3_score_diversity all OFF). MECH-439 conflict-grade levers OFF. Harm-free env
(num_hazards=0) -> visitation novelty source; SD-054 reef-bipartite GAP-A layout.

=== HARDENED READINESS (5 legs; all read at ARM_CURIOSITY = the non-saturation arm) ===

  Leg A (GAP-A pool divergent): cand_world_pairwise_dist_mean > CAND_DIST_FLOOR.
  Leg B (curiosity channel live; non-saturation arm): curiosity_bias_range_mean
    (pre-clamp cross-candidate RANGE, the SAME statistic the within-envelope argmin
    consumes) > BIAS_RANGE_FLOOR.
  Leg C (MECH-448 demotion non-degeneracy): f_eligibility_demotion_active_frac >=
    DEMOTION_ACTIVE_FRAC_FLOOR AND f_eligibility_excluded_count_mean >
    EXCLUDED_COUNT_FLOOR (the envelope actually narrows -- 705->705b hardening).
  Leg D (643a finite guard): max cand_world_pairwise_dist finite and < ceil.
  Leg E (NEW -- MECH-449 Go/No-Go gate non-degeneracy): go_nogo_n_soft_applied_mean
    > GNG_SOFT_APPLIED_FLOOR on >= MIN_SEEDS seeds. If the active No-Go never
    suppresses (inert gate -> double-gated == demotion-only), the run self-routes
    substrate_not_ready_requeue, NEVER a false reading. This is the non-vacuity gate
    that the flag-flip-alone vacuity finding above mandates.

The verdict resolver checks `not readiness_ok` FIRST, so the committed_class_entropy
LIFT criterion is scored ONLY after ALL FIVE legs are met. A residual all-admit
(legC) OR an inert Go/No-Go gate (legE) self-routes substrate_not_ready_requeue.

=== ACCEPTANCE (pre-registered; claim_ids=[MECH-314], experiment_purpose=evidence) ===

PASS (supports MECH-314) = READINESS met AND C_PRIMARY:
  ARM_CURIOSITY committed_class_entropy - ARM_FONLY (paired per seed) >=
  ENTROPY_LIFT_MARGIN on >= MIN_SEEDS seeds AND ARM_CURIOSITY strictly above
  ARM_NOISE (paired per seed) on >= MIN_SEEDS seeds AND ARM_CURIOSITY mean >
  ENTROPY_FLOOR. The MECH-314 per-candidate novelty channel, with F demoted out of
  the committed argmin AND the active Go/No-Go opening within-eligible room, converts
  to committed-action diversity.

Interpretation grid:
| outcome                                              | label                                              | evidence_direction | next |
|------------------------------------------------------|----------------------------------------------------|--------------------|------|
| any readiness leg below floor / non-finite           | substrate_not_ready_requeue                        | non_contributory   | pool/curiosity-range/demotion-excluded/Go-No-Go-inert below floor -> re-queue; NOT a weakens |
| readiness met + C_PRIMARY (lift over F-only + above noise) | mech314_curiosity_converts_under_double_gating | supports           | MECH-314 toward supports; the double gate lifts the conversion ceiling for curiosity |
| readiness met + no lift                              | conversion_ceiling_persists_despite_double_gating  | non_contributory   | PRE-REGISTERED TERMINAL off-ramp -> V4 ARC-110 loop-segregation; NO more V3 letters; NOT a falsification of MECH-314 |

PRE-REGISTERED TERMINAL: a readiness-MET no-lift here is the END of the V3 MECH-314
curiosity-conversion lineage. The two built selection-face eligibility gates
(MECH-448 demotion + MECH-449 Go/No-Go) are BOTH composed; if curiosity still does
not convert, the binding constraint is not at the eligibility-construction face and
escalation is V4 ARC-110 loop-segregation (existing v4_loop_segregation substrate_
queue entry). No further V3 letters on this lineage.

C_SAFETY (informational, NON-GATING): ARM_CURIOSITY realised harm-per-tick vs
ARM_FONLY is reported but does not gate (harm-free env keeps harm ~0; the Go/No-Go
safety axis is not exercised here -- staleness is the active axis).

architecture_epoch: "ree_hybrid_guardrails_v1"

Smoke:
  /opt/local/bin/python3 experiments/v3_exq_706_mech314_curiosity_conversion_double_gated.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, deque
from datetime import datetime, timezone
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
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_706_mech314_curiosity_conversion_double_gated"
QUEUE_ID = "V3-EXQ-706"
CLAIM_IDS: List[str] = ["MECH-314"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES: Optional[str] = "V3-EXQ-705b"  # brake-exempt double-gated redesign

SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 60            # SD-056 contrastive warmup (matches 705b / 648a)
P1_MEASUREMENT_EPISODES = 30       # committed-diversity measurement window
STEPS_PER_EPISODE = 200
MEASURE_AFTER_TICK = 20            # within-episode warmup before reading bias range

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_STEPS = 30
DRY_RUN_MEASURE_AFTER_TICK = 2

# Fixed curiosity weight at the treatment arm (NON-saturation; 590c range 0.0160).
# NOT swept -- the contrast is channel PRESENCE (0.25 vs 0.0), not two scales.
CURIOSITY_WEIGHT_ON = 0.25
MATCHED_NOISE_TEMPERATURE = 2.5

# --- Active Go/No-Go staleness/perseveration No-Go (706 addition) ---
# Each tick, candidates whose committed first-action class fell within the last
# RECENCY_WINDOW commits of the current episode are flagged stale at this level
# (>= the default gng_staleness_floor=0.5), so the BUILT MECH-449 gate drops them
# from the eligible set (fail-open keeps >= gng_protect_min_eligible=1 survivor).
USE_GO_NOGO_CONSTITUTION = True
STALENESS_NOGO_LEVEL = 0.9
RECENCY_WINDOW = 1                 # suppress the single most-recently-committed class

# Pre-registered thresholds.
BIAS_RANGE_FLOOR = 1.0e-4     # readiness leg B: curiosity per-candidate RANGE at the non-saturation arm
CAND_DIST_FLOOR = 0.02        # readiness leg A: e2.world_forward action-divergence (GAP-A pool)
EXCLUDED_COUNT_FLOOR = 0.0    # readiness leg C: mean f_eligibility_excluded_count strictly above this
DEMOTION_ACTIVE_FRAC_FLOOR = 0.8   # readiness leg C: fraction of ARM_CURIOSITY P1 ticks demotion-active
GNG_SOFT_APPLIED_FLOOR = 0.0  # readiness leg E: mean go_nogo_n_soft_applied strictly above this
MAGNITUDE_CEIL = 1.0e6        # readiness leg D: rolled-out z_world finite guard (643a)
ENTROPY_LIFT_MARGIN = 0.05    # C_PRIMARY: ARM_CURIOSITY committed_class_entropy - ARM_FONLY, per seed
ENTROPY_FLOOR = 0.3           # C_PRIMARY: ARM_CURIOSITY committed_class_entropy absolute floor
MIN_SEEDS_FOR_PASS = 2        # of 3

# SD-056 online contrastive training (mirror 705b / 648a / 689d harness).
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
E2_TRAIN_EVERY_K_TICKS = 1
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0

# Curiosity clamp / visitation buffer held FIXED (705b/648a/590c values).
CURIOSITY_BIAS_SCALE = 0.5
VISITATION_BUFFER_LEN = 256

# MECH-448 (ARC-107) demotion lever config (ON every arm; channel-adaptive floor
# per the 705b fix so the demotion ACTUALLY excludes on the near-tie-F pool).
USE_F_ELIGIBILITY_ADAPTIVE_FLOOR = True
F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR = 1.0   # >= 1.0 => excludes on any non-uniform field
F_ELIGIBILITY_ENVELOPE_FLOOR = 0.30        # inert under the adaptive floor (705 legacy)
F_ELIGIBILITY_DN_SIGMA = 0.0

# Modulatory-bias-selection-authority + shortlist stack (matches 705b).
MODULATORY_AUTHORITY_GAIN = 1.0
SHORTLIST_K = 3

# HARM-FREE env with SD-054 reef-bipartite GAP-A layout (divergent candidate pool).
ENV_KWARGS: Dict[str, Any] = dict(
    size=12,
    num_hazards=0,
    num_resources=5,
    hazard_harm=0.0,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.0,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=10,
    reef_enabled=True,
    n_reef_patches=3,
    reef_patch_radius=2,
    reef_bipartite_layout=True,
    reef_bipartite_axis="horizontal",
    reef_bipartite_agent_band_radius=1,
)

PRIMARY_ARM = "ARM_CURIOSITY"
FONLY_ARM = "ARM_FONLY"
NOISE_ARM = "ARM_NOISE"

ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": PRIMARY_ARM,
        "label": "mech314_curiosity_double_gated_treatment",
        "curiosity_novelty_weight": CURIOSITY_WEIGHT_ON,
        "temperature": 1.0,
        "is_control": False,
    },
    {
        "arm_id": FONLY_ARM,
        "label": "double_gated_f_only_control_curiosity_zero",
        "curiosity_novelty_weight": 0.0,
        "temperature": 1.0,
        "is_control": True,
    },
    {
        "arm_id": NOISE_ARM,
        "label": "matched_noise_flat_hot_temperature_curiosity_zero",
        "curiosity_novelty_weight": 0.0,
        "temperature": MATCHED_NOISE_TEMPERATURE,
        "is_control": True,
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """705b-verbatim substrate (MECH-314 curiosity the SOLE modulatory channel +
    MECH-448/ARC-107 demotion + adaptive floor on every arm) PLUS the BUILT
    MECH-449/ARC-107 Go/No-Go eligibility constitution turned ON
    (use_go_nogo_constitution=True). The Go/No-Go gate runs on the demotion-built
    eligible set BEFORE the within-eligible argmin; it suppresses candidates on an
    axis ORTHOGONAL to F-rank when fed per-candidate No-Go signals (here the active
    staleness/perseveration axis injected per tick in _run_seed_arm). The swept axis
    is channel PRESENCE (curiosity_novelty_weight 0.25 vs 0.0)."""
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
        # ARC-065 SP-CEM (Layer A) -- main-path action-divergent pool
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # MECH-314 curiosity is the SOLE modulatory channel -- all other bias channels OFF
        use_e3_score_diversity=False,
        use_noise_floor=False,
        use_tonic_vigor=False,
        use_dacc=False,
        use_lateral_pfc_analog=False,
        use_ofc_analog=False,
        use_mech295_liking_bridge=False,
        use_gated_policy=False,
        use_candidate_rule_field=False,
        # V_s substrate (main-path default; identical across arms)
        use_per_stream_vs=True,
        use_per_region_vs=True,
        use_event_segmenter=True,
        use_invalidation_trigger=True,
        use_anchor_sets=True,
        # SD-056 substrate present + trained online on every arm
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=2.0,
        # MECH-314 structured curiosity -- novelty sub-flavour ON (per-candidate channel)
        use_structured_curiosity=True,
        use_curiosity_novelty=True,
        use_curiosity_uncertainty=False,
        use_curiosity_learning_progress=False,
        curiosity_bias_scale=CURIOSITY_BIAS_SCALE,
        curiosity_novelty_weight=float(arm["curiosity_novelty_weight"]),
        curiosity_novelty_source="visitation",
        curiosity_visitation_buffer_len=VISITATION_BUFFER_LEN,
        curiosity_use_first_action_onehot=True,
        curiosity_first_action_augmentation_policy="auto",
        # GAP-A divergent pool: curiosity consumes the SD-056-divergent e2.world_forward(z0,a_i).
        curiosity_candidate_source="e2_world_forward",
        # Modulatory-bias-selection-authority + shortlist stack (CONSTANT across arms).
        use_modulatory_selection_authority=True,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
        use_modulatory_shortlist_then_modulate=True,
        modulatory_shortlist_mode="top_k",
        modulatory_shortlist_k=SHORTLIST_K,
        # MECH-439 conflict-grade levers OFF (this is the constitutional gate family).
        modulatory_shortlist_conflict_graded=False,
        use_gap_scaled_commit_temperature=False,
        # --- MECH-448 (ARC-107): rank-preserving F->eligibility demotion ON (every arm) ---
        use_f_eligibility_demotion=True,
        use_f_eligibility_adaptive_floor=USE_F_ELIGIBILITY_ADAPTIVE_FLOOR,
        f_eligibility_adaptive_mean_factor=F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR,
        f_eligibility_envelope_floor=F_ELIGIBILITY_ENVELOPE_FLOOR,
        f_eligibility_dn_sigma=F_ELIGIBILITY_DN_SIGMA,
        # --- MECH-449 (ARC-107): Go/No-Go eligibility constitution ON (every arm) ---
        # THE 705b->706 CHANGE. Built+validated (689g PASS 3/3). Gate suppresses only
        # when fed No-Go signals -> the active staleness axis is injected per tick in
        # _run_seed_arm; readiness leg E asserts it genuinely fires.
        use_go_nogo_constitution=USE_GO_NOGO_CONSTITUTION,
    )
    agent = REEAgent(cfg)
    # Per-channel score-bias decomposition so select_action records the per-candidate
    # curiosity bias range (the readiness leg-B statistic).
    agent.e3.e3_score_decomp_enabled = True
    return agent


# ---------------------------------------------------------------------------
# SD-056 online contrastive helpers (from 705b / 648a / 689d)
# ---------------------------------------------------------------------------

def _first_actions_K(candidates) -> torch.Tensor:
    rows = []
    for traj in candidates:
        rows.append(traj.actions[:, 0, :].detach().reshape(-1))
    return torch.stack(rows, dim=0)


def _candidate_first_action_classes(candidates) -> List[int]:
    """First-action class (argmax of the first-action one-hot) per candidate, in
    the SAME order the candidates are passed to select_action -> aligns the injected
    staleness signal to the gate's per-candidate raw_scores indexing."""
    classes: List[int] = []
    for traj in candidates:
        classes.append(int(traj.actions[:, 0, :].detach().reshape(-1).argmax().item()))
    return classes


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


def _entropy_from_counts(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for n in counts.values():
        if n <= 0:
            continue
        p = n / total
        ent -= p * math.log(p)
    return float(ent)


def _build_staleness_signal(
    candidate_classes: List[int],
    recent_classes: Deque[int],
) -> torch.Tensor:
    """Per-candidate staleness No-Go: STALENESS_NOGO_LEVEL for any candidate whose
    first-action class was committed within the last RECENCY_WINDOW commits of this
    episode (>= gng_staleness_floor so the MECH-449 gate drops it), else 0.0. Length
    == number of candidates so the gate's numel()==n alignment check passes."""
    recent_set = set(recent_classes)
    vals = [
        STALENESS_NOGO_LEVEL if cls in recent_set else 0.0
        for cls in candidate_classes
    ]
    return torch.tensor(vals, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Per-(seed, arm) runner
# ---------------------------------------------------------------------------

def _run_seed_arm(
    arm: Dict[str, Any],
    seed: int,
    p0_episodes: int,
    p1_episodes: int,
    steps_per_episode: int,
    measure_after_tick: int,
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
    total_train_eps = p0_episodes + p1_episodes

    # PRIMARY DV: pooled committed first-action classes over the P1 window.
    committed_class_counts: Counter = Counter()
    # Readiness instrumentation (P1, past the within-episode warmup window).
    curiosity_range_vals: List[float] = []
    pairwise_dists: List[float] = []
    pairwise_dist_max_seen = 0.0
    # MECH-448 demotion non-degeneracy readouts.
    demotion_active_ticks = 0
    envelope_sizes: List[float] = []
    excluded_counts: List[float] = []
    winner_neq_f_argmin_ticks = 0
    rank_preserving_active_ticks = 0
    # MECH-449 Go/No-Go gate non-degeneracy readouts (706 addition).
    gng_active_ticks = 0
    gng_soft_applied_vals: List[float] = []
    gng_safety_nogo_vals: List[float] = []
    gng_envelope_sizes: List[float] = []
    # C_SAFETY (informational).
    harm_p1_abs_sum = 0.0
    harm_p1_ticks = 0

    n_p1_ticks = 0
    n_p1_ticks_past_window = 0
    n_contrastive_steps = 0
    error_note: Optional[str] = None

    for ep in range(total_train_eps):
        is_p1 = ep >= p0_episodes
        phase_label = "P1" if is_p1 else "P0"

        _, obs_dict = env.reset()
        agent.reset()
        # Anti-perseveration recency is per-episode (a fresh foraging bout).
        recent_committed: Deque[int] = deque(maxlen=RECENCY_WINDOW)
        agent.set_injected_go_nogo_signals(None)

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

            past_window = is_p1 and tick_in_ep >= measure_after_tick
            if past_window and candidates and len(candidates) >= 2:
                actions_K = _first_actions_K(candidates).to(agent.device)
                z0 = latent.z_world.detach()
                with torch.no_grad():
                    dist = float(
                        agent.e2.cand_world_pairwise_dist(z0, actions_K).item()
                    )
                if math.isfinite(dist):
                    pairwise_dists.append(dist)
                    pairwise_dist_max_seen = max(pairwise_dist_max_seen, dist)

            if agent.goal_state is not None:
                try:
                    energy = float(body[0, 3].item())
                except Exception:
                    energy = 1.0
                drive_level = max(0.0, 1.0 - energy)
                agent.update_z_goal(benefit_exposure=0.0, drive_level=drive_level)

            # --- ACTIVE Go/No-Go staleness No-Go injection (706) ---
            # Flag candidates whose first-action class was committed within the last
            # RECENCY_WINDOW commits of this episode, so the BUILT MECH-449 gate drops
            # them from the eligible set. Identical RULE on every arm.
            if candidates:
                cand_classes = _candidate_first_action_classes(candidates)
                staleness = _build_staleness_signal(cand_classes, recent_committed)
                agent.set_injected_go_nogo_signals({"staleness": staleness})
            else:
                agent.set_injected_go_nogo_signals(None)

            action = agent.select_action(
                candidates, ticks, temperature=arm_temperature
            )

            # Per-candidate curiosity bias RANGE (readiness leg B) + committed-action
            # class (PRIMARY DV) + MECH-448 demotion diagnostics + MECH-449 Go/No-Go
            # diagnostics (readiness leg E), captured AFTER select_action.
            if past_window:
                decomp = getattr(agent, "_last_score_bias_decomp", {}) or {}
                crange = float(decomp.get("curiosity_bias_range_mean", 0.0))
                if math.isfinite(crange):
                    curiosity_range_vals.append(crange)
                diag = agent.e3.last_score_diagnostics
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
                if bool(diag.get("go_nogo_constitution_active", False)):
                    gng_active_ticks += 1
                    soft = float(diag.get("go_nogo_n_soft_applied", 0))
                    if math.isfinite(soft) and soft >= 0:
                        gng_soft_applied_vals.append(soft)
                    safe = float(diag.get("go_nogo_n_safety_nogo", 0))
                    if math.isfinite(safe) and safe >= 0:
                        gng_safety_nogo_vals.append(safe)
                    gsz = float(diag.get("go_nogo_envelope_size", -1))
                    if math.isfinite(gsz) and gsz >= 0:
                        gng_envelope_sizes.append(gsz)
                n_p1_ticks_past_window += 1

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

            committed_class = int(action.argmax().item())
            if past_window:
                committed_class_counts.update([committed_class])
            # Update the per-episode anti-perseveration recency window.
            recent_committed.append(committed_class)

            if is_p1:
                n_p1_ticks += 1

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
            if past_window:
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

        if (ep + 1) % 10 == 0 or (ep + 1) == total_train_eps:
            print(
                f"  [train] arm={arm['arm_id']} seed={seed} phase={phase_label} "
                f"ep {ep + 1}/{total_train_eps}",
                flush=True,
            )

    def _mean(xs: List[float], default: float = 0.0) -> float:
        return float(sum(xs) / len(xs)) if xs else default

    demotion_active_frac = (
        float(demotion_active_ticks) / float(n_p1_ticks_past_window)
        if n_p1_ticks_past_window > 0 else 0.0
    )
    rank_preserving_frac = (
        float(rank_preserving_active_ticks) / float(demotion_active_ticks)
        if demotion_active_ticks > 0 else 0.0
    )
    gng_active_frac = (
        float(gng_active_ticks) / float(n_p1_ticks_past_window)
        if n_p1_ticks_past_window > 0 else 0.0
    )
    harm_per_tick_mean = (
        harm_p1_abs_sum / float(harm_p1_ticks) if harm_p1_ticks > 0 else 0.0
    )

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "curiosity_novelty_weight": float(arm["curiosity_novelty_weight"]),
        "temperature": arm_temperature,
        "is_control": bool(arm["is_control"]),
        "seed": int(seed),
        "n_p1_ticks": int(n_p1_ticks),
        "n_p1_ticks_past_window": int(n_p1_ticks_past_window),
        "n_contrastive_steps": int(n_contrastive_steps),
        "error_note": error_note,
        # PRIMARY DV.
        "committed_class_entropy": round(_entropy_from_counts(committed_class_counts), 6),
        "n_committed_classes": int(len(committed_class_counts)),
        "committed_class_counts": dict(sorted(committed_class_counts.items())),
        # Readiness leg B: per-candidate curiosity bias range (pre-clamp; non-saturation arm).
        "curiosity_bias_range_mean": round(_mean(curiosity_range_vals), 8),
        # Readiness leg A/D input.
        "cand_world_pairwise_dist_mean": round(_mean(pairwise_dists), 6),
        "cand_world_pairwise_dist_max": round(pairwise_dist_max_seen, 6),
        # Readiness leg C: MECH-448 demotion non-degeneracy.
        "f_eligibility_demotion_active_ticks": int(demotion_active_ticks),
        "f_eligibility_demotion_active_frac": round(demotion_active_frac, 6),
        "f_eligibility_envelope_size_mean": round(_mean(envelope_sizes), 6),
        "f_eligibility_excluded_count_mean": round(_mean(excluded_counts), 6),
        "f_eligibility_winner_neq_f_argmin_ticks": int(winner_neq_f_argmin_ticks),
        "f_eligibility_rank_preserving_frac": round(rank_preserving_frac, 6),
        # Readiness leg E: MECH-449 Go/No-Go gate non-degeneracy.
        "go_nogo_active_ticks": int(gng_active_ticks),
        "go_nogo_active_frac": round(gng_active_frac, 6),
        "go_nogo_n_soft_applied_mean": round(_mean(gng_soft_applied_vals), 6),
        "go_nogo_n_safety_nogo_mean": round(_mean(gng_safety_nogo_vals), 6),
        "go_nogo_envelope_size_mean": round(_mean(gng_envelope_sizes), 6),
        # C_SAFETY (informational).
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


def _evaluate(arm_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    on = _arm_rows(arm_results, PRIMARY_ARM)
    fonly = _arm_rows(arm_results, FONLY_ARM)
    noise = _arm_rows(arm_results, NOISE_ARM)

    fonly_by_seed = {r["seed"]: r for r in fonly}
    noise_by_seed = {r["seed"]: r for r in noise}

    SENT = "committed_class_entropy"
    RANGE = "curiosity_bias_range_mean"
    PDIST = "cand_world_pairwise_dist_mean"
    GNG = "go_nogo_n_soft_applied_mean"

    # --- READINESS (all legs read at ARM_CURIOSITY = the non-saturation arm) ---
    # Leg A: GAP-A pool divergent (e2.world_forward action-divergence non-vacuity).
    on_cand_dist_mean = _mean_key(on, PDIST)
    legA_seeds = _n_seeds(on, lambda r: float(r.get(PDIST, 0.0)) > CAND_DIST_FLOOR)
    legA_ok = bool(legA_seeds >= MIN_SEEDS_FOR_PASS)

    # Leg B: curiosity per-candidate range at the non-saturation arm.
    on_curiosity_range_mean = _mean_key(on, RANGE)
    legB_seeds = _n_seeds(on, lambda r: float(r.get(RANGE, 0.0)) > BIAS_RANGE_FLOOR)
    legB_ok = bool(legB_seeds >= MIN_SEEDS_FOR_PASS)

    # Leg C: MECH-448 demotion non-degeneracy (active + actually excludes).
    def _dem_non_degen(r: Dict[str, Any]) -> bool:
        return bool(
            float(r.get("f_eligibility_demotion_active_frac", 0.0)) >= DEMOTION_ACTIVE_FRAC_FLOOR
            and float(r.get("f_eligibility_excluded_count_mean", 0.0)) > EXCLUDED_COUNT_FLOOR
        )
    legC_seeds = _n_seeds(on, _dem_non_degen)
    legC_ok = bool(legC_seeds >= MIN_SEEDS_FOR_PASS)

    # Leg D: finite guard.
    max_pairwise = max(
        [float(r.get("cand_world_pairwise_dist_max", 0.0)) for r in arm_results] or [0.0]
    )
    legD_ok = bool(math.isfinite(max_pairwise) and max_pairwise < MAGNITUDE_CEIL)

    # Leg E (706): MECH-449 Go/No-Go gate non-degeneracy -- the active No-Go genuinely
    # suppresses (go_nogo_n_soft_applied_mean > floor) so the double-gating is NOT an
    # inert no-op (== demotion-only 705b). An inert gate self-routes
    # substrate_not_ready_requeue, NEVER a false reading.
    on_gng_soft_applied_mean = _mean_key(on, GNG)
    legE_seeds = _n_seeds(on, lambda r: float(r.get(GNG, 0.0)) > GNG_SOFT_APPLIED_FLOOR)
    legE_ok = bool(legE_seeds >= MIN_SEEDS_FOR_PASS)

    readiness_ok = bool(legA_ok and legB_ok and legC_ok and legD_ok and legE_ok)

    # --- C_PRIMARY: lift over F-only (paired) + strict-above matched-noise + floor ---
    def _lift_over_fonly(r_on: Dict[str, Any]) -> bool:
        r_f = fonly_by_seed.get(r_on["seed"])
        if r_f is None:
            return False
        return float(r_on.get(SENT, 0.0)) - float(r_f.get(SENT, 0.0)) >= ENTROPY_LIFT_MARGIN

    def _above_noise(r_on: Dict[str, Any]) -> bool:
        r_n = noise_by_seed.get(r_on["seed"])
        if r_n is None:
            return False
        return float(r_on.get(SENT, 0.0)) > float(r_n.get(SENT, 0.0))

    lift_seeds = _n_seeds(on, _lift_over_fonly)
    above_noise_seeds = _n_seeds(on, _above_noise)
    on_sent_mean = _mean_key(on, SENT)
    primary_floor_ok = bool(on_sent_mean > ENTROPY_FLOOR)
    primary_pass = bool(
        lift_seeds >= MIN_SEEDS_FOR_PASS
        and above_noise_seeds >= MIN_SEEDS_FOR_PASS
        and primary_floor_ok
    )

    # VERDICT resolver: readiness -> C_PRIMARY. No weakens path (a no-lift under the
    # double gate is a conversion-ceiling persistence, NOT evidence against MECH-314).
    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        overall_pass = False
        evidence_direction = "non_contributory"
    elif primary_pass:
        label = "mech314_curiosity_converts_under_double_gating"
        overall_pass = True
        evidence_direction = "supports"
    else:
        label = "conversion_ceiling_persists_despite_double_gating"
        overall_pass = False
        evidence_direction = "non_contributory"

    per_arm_entropy = {
        PRIMARY_ARM: round(on_sent_mean, 6),
        FONLY_ARM: round(_mean_key(fonly, SENT), 6),
        NOISE_ARM: round(_mean_key(noise, SENT), 6),
    }

    return {
        "readiness": {
            "legA_cand_dist_floor": CAND_DIST_FLOOR,
            "on_cand_world_pairwise_dist_mean": round(on_cand_dist_mean, 6),
            "legA_seeds_above_floor": int(legA_seeds),
            "legA_ok": legA_ok,
            "legB_bias_range_floor": BIAS_RANGE_FLOOR,
            "on_curiosity_bias_range_mean": round(on_curiosity_range_mean, 8),
            "legB_seeds_above_floor": int(legB_seeds),
            "legB_ok": legB_ok,
            "legB_note": (
                "CONFOUNDED-PRECONDITION FIX (from 705b): curiosity_bias_range read "
                "at the NON-saturation arm ARM_CURIOSITY (w=0.25), the SAME range "
                "statistic the within-envelope argmin consumes."
            ),
            "legC_demotion_active_frac_floor": DEMOTION_ACTIVE_FRAC_FLOOR,
            "legC_excluded_count_floor": EXCLUDED_COUNT_FLOOR,
            "on_demotion_active_frac_mean": round(_mean_key(on, "f_eligibility_demotion_active_frac"), 6),
            "on_excluded_count_mean": round(_mean_key(on, "f_eligibility_excluded_count_mean"), 6),
            "on_envelope_size_mean": round(_mean_key(on, "f_eligibility_envelope_size_mean"), 6),
            "legC_seeds_non_degenerate": int(legC_seeds),
            "legC_ok": legC_ok,
            "legD_magnitude_ceil": MAGNITUDE_CEIL,
            "max_pairwise_dist_observed": round(max_pairwise, 6),
            "legD_ok": legD_ok,
            "legE_gng_soft_applied_floor": GNG_SOFT_APPLIED_FLOOR,
            "on_go_nogo_n_soft_applied_mean": round(on_gng_soft_applied_mean, 6),
            "on_go_nogo_active_frac_mean": round(_mean_key(on, "go_nogo_active_frac"), 6),
            "on_go_nogo_envelope_size_mean": round(_mean_key(on, "go_nogo_envelope_size_mean"), 6),
            "legE_seeds_non_degenerate": int(legE_seeds),
            "legE_ok": legE_ok,
            "legE_note": (
                "MECH-449 Go/No-Go gate non-degeneracy: the active staleness No-Go "
                "genuinely suppresses (go_nogo_n_soft_applied_mean > floor). An inert "
                "gate (== demotion-only 705b) self-routes substrate_not_ready_requeue, "
                "NEVER a false ceiling -- the 705->705b all-admit trap analog for the "
                "second gate."
            ),
            "min_seeds_required": MIN_SEEDS_FOR_PASS,
            "readiness_ok": readiness_ok,
        },
        "c_primary": {
            "entropy_lift_margin": ENTROPY_LIFT_MARGIN,
            "on_minus_fonly_lift_seeds": int(lift_seeds),
            "on_above_noise_seeds": int(above_noise_seeds),
            "on_committed_class_entropy_mean": round(on_sent_mean, 6),
            "entropy_floor": ENTROPY_FLOOR,
            "primary_floor_ok": primary_floor_ok,
            "c_primary_pass": primary_pass,
            "note": (
                "GATE: ARM_CURIOSITY committed_class_entropy lifts over the "
                "DOUBLE-GATED F-only control (ARM_FONLY) by >= margin per seed AND "
                "is strictly above the matched-noise control (ARM_NOISE) per seed, "
                "on >= MIN_SEEDS seeds, AND clears an absolute floor. Fail = "
                "conversion_ceiling_persists_despite_double_gating (non_contributory; "
                "PRE-REGISTERED TERMINAL -> V4 ARC-110 loop-segregation), NOT a "
                "MECH-314 weakens."
            ),
        },
        "per_arm_committed_class_entropy_mean": per_arm_entropy,
        "curiosity_bias_range_per_arm_mean": {
            PRIMARY_ARM: round(on_curiosity_range_mean, 8),
            FONLY_ARM: round(_mean_key(fonly, RANGE), 8),
            NOISE_ARM: round(_mean_key(noise, RANGE), 8),
        },
        "go_nogo_soft_applied_per_arm_mean": {
            PRIMARY_ARM: round(on_gng_soft_applied_mean, 6),
            FONLY_ARM: round(_mean_key(fonly, GNG), 6),
            NOISE_ARM: round(_mean_key(noise, GNG), 6),
        },
        "harm_per_tick_per_arm_mean": {
            PRIMARY_ARM: round(_mean_key(on, "harm_per_p1_tick_mean"), 6),
            FONLY_ARM: round(_mean_key(fonly, "harm_per_p1_tick_mean"), 6),
            NOISE_ARM: round(_mean_key(noise, "harm_per_p1_tick_mean"), 6),
        },
        "rank_preserving_frac_on_mean": round(_mean_key(on, "f_eligibility_rank_preserving_frac"), 6),
        "label": label,
        "evidence_direction": evidence_direction,
        "overall_pass": overall_pass,
        # Readiness preconditions (same-statistic discipline; verdict-class self-route).
        "preconditions": [
            {
                "name": "gapA_e2_world_forward_action_divergence_non_vacuity",
                "kind": "readiness",
                "description": (
                    "ARM_CURIOSITY e2.world_forward(z0,a_i) candidate predictions stay "
                    "action-divergent (cand_world_pairwise_dist > floor) -- the "
                    "curiosity_candidate_source='e2_world_forward' / GAP-A pool "
                    "precondition. A collapsed pool starves the per-candidate channel."
                ),
                "control": "ARM_CURIOSITY (w=0.25), cand_world_pairwise_dist_mean",
                "measured": round(on_cand_dist_mean, 6),
                "threshold": CAND_DIST_FLOOR,
                "met": legA_ok,
            },
            {
                "name": "curiosity_bias_range_supra_floor_at_non_saturation_arm",
                "kind": "readiness",
                "description": (
                    "ARM_CURIOSITY per-candidate curiosity_bias_range (pre-clamp "
                    "cross-candidate RANGE -- the SAME range statistic the "
                    "within-envelope argmin consumes) clears the floor at the "
                    "NON-saturation weight (w=0.25). 590c confounded-precondition fix."
                ),
                "control": "ARM_CURIOSITY (w=0.25, non-saturation), curiosity_bias_range_mean",
                "measured": round(on_curiosity_range_mean, 8),
                "threshold": BIAS_RANGE_FLOOR,
                "met": legB_ok,
            },
            {
                "name": "f_eligibility_demotion_non_degeneracy",
                "kind": "readiness",
                "description": (
                    "MECH-448 demotion is ACTIVE on >= floor of ARM_CURIOSITY P1 ticks "
                    "AND the envelope ACTUALLY excludes on the divergent pool (mean "
                    "f_eligibility_excluded_count > floor). excluded_count==0 = all-admit "
                    "(flat-F) = vacuous self-route (705->705b hardening)."
                ),
                "control": "ARM_CURIOSITY, f_eligibility_demotion_active_frac + excluded_count",
                "measured": round(_mean_key(on, "f_eligibility_excluded_count_mean"), 6),
                "threshold": EXCLUDED_COUNT_FLOOR,
                "met": legC_ok,
            },
            {
                "name": "rolled_out_zworld_magnitude_bounded",
                "kind": "readiness",
                "description": (
                    "Rolled-out z_world spread stayed finite and below the 643a "
                    "explosion ceiling (SD-056 online numerical stability; rollout "
                    "clamp ON)."
                ),
                "control": "max cand_world_pairwise_dist across all arms",
                "measured": round(max_pairwise, 6),
                "threshold": MAGNITUDE_CEIL,
                "direction": "upper",
                "met": legD_ok,
            },
            {
                "name": "go_nogo_gate_non_degeneracy",
                "kind": "readiness",
                "description": (
                    "MECH-449 Go/No-Go gate genuinely fires: the active staleness "
                    "No-Go suppresses >= 1 candidate on average (go_nogo_n_soft_applied "
                    "_mean > floor) at ARM_CURIOSITY on >= MIN_SEEDS seeds. An inert "
                    "gate (no suppression) makes the double-gated arm identical to "
                    "demotion-only 705b -> vacuous; self-route substrate_not_ready_"
                    "requeue, NOT a ceiling."
                ),
                "control": "ARM_CURIOSITY, go_nogo_n_soft_applied_mean",
                "measured": round(on_gng_soft_applied_mean, 6),
                "threshold": GNG_SOFT_APPLIED_FLOOR,
                "met": legE_ok,
            },
        ],
        "criteria": [
            {
                "name": "mech314_committed_diversity_lift_over_double_gated_f_only_and_noise",
                "load_bearing": True,
                "passed": primary_pass,
            },
        ],
        "criteria_non_degenerate": {
            "mech314_committed_diversity_lift_over_double_gated_f_only_and_noise": readiness_ok,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    p0 = DRY_RUN_P0 if dry_run else P0_WARMUP_EPISODES
    p1 = DRY_RUN_P1 if dry_run else P1_MEASUREMENT_EPISODES
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE
    measure_after = DRY_RUN_MEASURE_AFTER_TICK if dry_run else MEASURE_AFTER_TICK

    arm_results: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm['arm_id']}", flush=True)
            cell = _run_seed_arm(arm, seed, p0, p1, steps, measure_after)
            cell["arm_fingerprint"] = compute_arm_fingerprint(
                config_slice={
                    "arm": {
                        k: arm[k]
                        for k in ("arm_id", "curiosity_novelty_weight", "temperature", "is_control")
                    },
                    "env_kwargs": ENV_KWARGS,
                    "sd056_weight": SD056_WEIGHT,
                    "curiosity_bias_scale": CURIOSITY_BIAS_SCALE,
                    "visitation_buffer_len": VISITATION_BUFFER_LEN,
                    "curiosity_candidate_source": "e2_world_forward",
                    "use_f_eligibility_demotion": True,
                    "use_f_eligibility_adaptive_floor": USE_F_ELIGIBILITY_ADAPTIVE_FLOOR,
                    "f_eligibility_adaptive_mean_factor": F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR,
                    "f_eligibility_envelope_floor": F_ELIGIBILITY_ENVELOPE_FLOOR,
                    "f_eligibility_dn_sigma": F_ELIGIBILITY_DN_SIGMA,
                    "use_go_nogo_constitution": USE_GO_NOGO_CONSTITUTION,
                    "staleness_nogo_level": STALENESS_NOGO_LEVEL,
                    "recency_window": RECENCY_WINDOW,
                    "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
                    "modulatory_shortlist_k": SHORTLIST_K,
                    "p0_episodes": p0, "p1_episodes": p1, "steps_per_episode": steps,
                },
                seed=seed,
                script_path=Path(__file__),
                rng_fully_reset=True,
                extra_ineligible_reasons=["online_e2_training_stateful_per_cell"],
            )
            arm_results.append(cell)
            passed = cell.get("error_note") is None
            print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)

    summary = _evaluate(arm_results)
    outcome = "PASS" if summary["overall_pass"] else "FAIL"
    evidence_direction = summary["evidence_direction"]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    # Non-degeneracy net (evidence run): the load-bearing committed_class_entropy
    # must carry real cross-arm spread. If readiness failed (pool/curiosity-range/
    # demotion-excluded/Go-No-Go-inert below floor) the lift criterion could never
    # fire -> scoring-excluded.
    non_degenerate = bool(summary["readiness"]["readiness_ok"])

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
        "supersedes": "v3_exq_705b_mech314_curiosity_conversion_demotion_adaptive_floor",
        "evidence_direction": evidence_direction,
        "evidence_direction_note": (
            "V3-EXQ-706 SUPERSEDES V3-EXQ-705b (brake-EXEMPT DOUBLE-GATED redesign, "
            "routed by the confirmed failure_autopsy_V3-EXQ-705b_2026-06-25). NEW "
            "number (major redesign: adds the second composed eligibility gate -- the "
            "BUILT MECH-449/ARC-107 Go/No-Go constitution -- plus an active staleness/"
            "perseveration No-Go injection so the gate genuinely fires; the re-derive "
            "brake (6th MECH-314 ceiling autopsy) REFUSED a 705c demotion-ONLY "
            "same-substrate letter, and the double-gated condition is the different-"
            "substrate forward path the autopsy named as brake-EXEMPT). 705b was the "
            "FIRST FAIR demotion test (adaptive floor excluded 14.45 vs 705 all-admit "
            "0.0; 590c saturation-arm confound fixed; GAP-A pool divergent ~0.107) and "
            "committed-class entropy STILL did not lift over demotion-ON F-only + "
            "matched-noise -> conversion_ceiling_persists_despite_demotion. The autopsy "
            "diagnosis: rank-preserving demotion alone cannot express the active No-Go "
            "the conversion requires (the 485k->MECH-449 OFC-devaluation shape). 706 "
            "composes BOTH gates: use_f_eligibility_demotion=True AND "
            "use_go_nogo_constitution=True on every arm, with an active staleness "
            "No-Go suppressing recently-committed action classes from the eligible set "
            "(689h showed demotion x Go/No-Go compose ADDITIVELY). SUBSTRATE-READINESS "
            "FINDING (caught at code review): a flag-flip alone is VACUOUS -- with "
            "use_dacc=False and no signal injection the Go/No-Go gate suppresses "
            "nothing (bit-identical to demotion-only 705b), so 706 injects the active "
            "staleness axis AND adds readiness leg E (go_nogo_n_soft_applied>0 on "
            ">=2/3 seeds) so an inert gate self-routes substrate_not_ready_requeue, "
            "NEVER a false reading. The Go/No-Go is an ELIGIBILITY gate, not a "
            "modulatory bias channel -- MECH-314 curiosity remains the SOLE modulatory "
            "channel arbitrating the within-eligible argmin. The same staleness RULE "
            "runs on ALL arms; ARM_FONLY absorbs the gate-alone diversity baseline so "
            "the PASS criterion (ARM_CURIOSITY lift over the DOUBLE-GATED F-only "
            "control by >= margin on >= 2/3 seeds AND strict-above matched-noise) "
            "isolates curiosity-specific conversion. A readiness-MET no-lift is the "
            "PRE-REGISTERED TERMINAL off-ramp: BOTH selection-face eligibility gates "
            "are composed, so the binding constraint is not at the eligibility-"
            "construction face -> escalate to V4 ARC-110 loop-segregation (existing "
            "v4_loop_segregation substrate_queue entry); NO more V3 letters on this "
            "lineage. claim_ids=[MECH-314]; non_contributory off-ramp is NOT a MECH-314 "
            "weakens (the ceiling is architectural). PROMOTES NOTHING."
        ),
        "interpretation": {
            "label": summary["label"],
            "preconditions": summary["preconditions"],
            "criteria": summary["criteria"],
            "criteria_non_degenerate": summary["criteria_non_degenerate"],
            "routing": {
                "substrate_not_ready_requeue": (
                    "GAP-A pool not divergent / curiosity range below floor at the "
                    "non-saturation arm / demotion all-admit / Go-No-Go gate inert "
                    "(no suppression) -- re-queue; do NOT weaken MECH-314"
                ),
                "mech314_curiosity_converts_under_double_gating": (
                    "PASS (supports MECH-314); the demotion + active Go/No-Go double "
                    "gate lifts the conversion ceiling for the curiosity channel"
                ),
                "conversion_ceiling_persists_despite_double_gating": (
                    "PRE-REGISTERED TERMINAL off-ramp: readiness MET (both gates "
                    "genuinely active) but no lift over the double-gated F-only + "
                    "matched-noise controls -> the binding constraint is not at the "
                    "selection-face eligibility-construction; escalate to V4 ARC-110 "
                    "loop-segregation, NO more V3 letters. non_contributory; NOT a "
                    "falsification of MECH-314"
                ),
            },
        },
        "non_degenerate": non_degenerate,
        "degeneracy_reason": (
            None if non_degenerate else
            "readiness below floor (GAP-A pool / curiosity per-candidate range / "
            "demotion excluded_count / Go-No-Go gate inert) -- the "
            "committed_class_entropy lift criterion could not fire; scoring-excluded"
        ),
        "dry_run": bool(dry_run),
        "config": {
            "seeds": seeds,
            "p0_warmup_episodes": p0,
            "p1_measurement_episodes": p1,
            "steps_per_episode": steps,
            "measure_after_tick": measure_after,
            "arms": [
                {k: a[k] for k in ("arm_id", "label", "curiosity_novelty_weight", "temperature", "is_control")}
                for a in ARMS
            ],
            "curiosity_weight_on": CURIOSITY_WEIGHT_ON,
            "matched_noise_temperature": MATCHED_NOISE_TEMPERATURE,
            "use_f_eligibility_demotion": True,
            "use_f_eligibility_adaptive_floor": USE_F_ELIGIBILITY_ADAPTIVE_FLOOR,
            "f_eligibility_adaptive_mean_factor": F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR,
            "f_eligibility_envelope_floor": F_ELIGIBILITY_ENVELOPE_FLOOR,
            "f_eligibility_dn_sigma": F_ELIGIBILITY_DN_SIGMA,
            "use_go_nogo_constitution": USE_GO_NOGO_CONSTITUTION,
            "staleness_nogo_level": STALENESS_NOGO_LEVEL,
            "recency_window": RECENCY_WINDOW,
            "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
            "modulatory_shortlist_k": SHORTLIST_K,
            "env_kwargs": ENV_KWARGS,
            "sd056_weight": SD056_WEIGHT,
            "curiosity_bias_scale": CURIOSITY_BIAS_SCALE,
            "visitation_buffer_len": VISITATION_BUFFER_LEN,
            "thresholds": {
                "cand_dist_floor": CAND_DIST_FLOOR,
                "bias_range_floor": BIAS_RANGE_FLOOR,
                "excluded_count_floor": EXCLUDED_COUNT_FLOOR,
                "demotion_active_frac_floor": DEMOTION_ACTIVE_FRAC_FLOOR,
                "gng_soft_applied_floor": GNG_SOFT_APPLIED_FLOOR,
                "magnitude_ceil": MAGNITUDE_CEIL,
                "entropy_lift_margin": ENTROPY_LIFT_MARGIN,
                "entropy_floor": ENTROPY_FLOOR,
                "min_seeds_for_pass": MIN_SEEDS_FOR_PASS,
            },
        },
        "acceptance_criteria": {
            "readiness_substrate_ready": summary["readiness"]["readiness_ok"],
            "mech314_committed_diversity_lift": summary["c_primary"]["c_primary_pass"],
            "overall_pass": summary["overall_pass"],
        },
        "summary": summary,
        "arm_results": arm_results,
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    print(f"Manifest written: {out_path}", flush=True)

    print(f"Outcome: {outcome} (label={summary['label']})", flush=True)
    for k, v in manifest["acceptance_criteria"].items():
        print(f"  {k}: {v}", flush=True)
    print(
        f"  committed_class_entropy per arm: {summary['per_arm_committed_class_entropy_mean']}",
        flush=True,
    )
    print(
        f"  go_nogo_n_soft_applied per arm: {summary['go_nogo_soft_applied_per_arm_mean']}",
        flush=True,
    )

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run_experiment(dry_run=args.dry_run)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    _outcome = _outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL"
    emit_outcome(
        outcome=_outcome,
        manifest_path=str(result.get("manifest_path", Path("/dev/null"))),
        dry_run=args.dry_run,
    )
    sys.exit(0)
