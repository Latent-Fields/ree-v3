"""V3-EXQ-689j: MECH-448 (ARC-107) Factor B (MECH-439) gap-scaled stochastic
commit NOISE-CONTROL instrument follow-up -- TARGETED repower + diagnostics of
V3-EXQ-689i's C_NOISE_LIFTS gate only. NOT a re-test of C_PRIMARY.

PREDECESSOR AND SCOPE. V3-EXQ-689i
(run_id v3_exq_689i_mech448_f_eligibility_demotion_falsifier_repair_20260722T162850Z_v3)
is the confirmed repaired-instrument successor to 689d for MECH-448
(REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-689i_2026-07-24.{md,json},
status confirmed, user-adjudicated "gate defect, science upheld" 2026-07-24). 6 of
8 load-bearing criteria passed cleanly, INCLUDING C_PRIMARY (the mechanism's own
discrimination test: ARM_ON committed-class entropy strictly above both collapsed
controls). This script does NOT touch that finding -- ARM_OFF and ARM_ON
(PRIMARY_ARM) are DROPPED ENTIRELY. The one gap 689i left was C_NOISE_LIFTS
(matched_noise_control_verifiably_lifts): the Factor B gap-scaled stochastic
commit temperature control -- its FIRST exercise in this exact form -- only
verifiably lifted committed-class entropy (selected_action_class_entropy_mm,
Miller-Madow corrected) over ARM_PROPOSER_CTRL on 1 of 4 seeds against a >=3/4
bar. 689i's own autopsy recommended a TARGETED follow-up on this instrument
specifically, not a full re-letter of the falsifier -- that is what this script
is. claim_ids=[MECH-448] is retained for audit traceability, but this run is
EXPERIMENT_PURPOSE="diagnostic": with no ARM_ON present it cannot itself test the
MECH-448 mechanism, so it is excluded from governance confidence/conflict
scoring and reports evidence_direction="non_contributory" regardless of outcome.
It exists to resolve (or further characterize) 689i's ONE blocked gate.

WHY A REPOWER ALONE MIGHT NOT BE ENOUGH -- AND WHY IT IS STILL THE RIGHT FIRST
MOVE. 689i's per-seed selected_action_class_entropy_mm delta (ARM_MATCHED_NOISE
minus ARM_PROPOSER_CTRL) was: seed42 -0.038, seed43 -0.032 (both arms
near-collapsed to one class), seed44 -0.035, seed45 +0.089. THREE of four seeds
show a CONSISTENT SMALL NEGATIVE delta clustered within ~0.006 nats of each
other, with only seed45 lifting (and by a much larger margin). This is a
DIFFERENT signature from "true small positive effect, underpowered": a genuinely
underpowered small-positive-mean effect would show noisy signs scattered around
a positive mean, not three tightly-clustered negatives plus one large positive
outlier. So there is a live possibility that more seeds under the UNCHANGED
design converge on a small negative/null effect rather than crossing the >=3/4
bar. Repowering first (rather than jumping straight to a config change) is still
correct: (a) it is the cheap, non-destructive first move the autopsy itself
recommended; (b) a mechanistic explanation is offered below (route (b)) but is
NOT altered here -- MODIFYING alpha/base_temperature would be a design change
requiring its own justification and would not be "the same instrument, more
power"; (c) this script ALSO records the diagnostics that would explain a
persistent null (see next section), so a null result at n=12 is informative
rather than another opaque under-powered read, and directly motivates a
concrete follow-on config change if warranted.

MECHANISTIC HYPOTHESIS FOR THE WEAK/NEGATIVE LIFT (recorded, not fixed, here).
With `use_modulatory_shortlist_then_modulate=True` (on for both retained arms,
unchanged from 689i), Factor B's `_gap_scaled_commit_pick`
(ree_core/predictors/e3_selector.py:1440, call site ~3078-3106) samples via
softmax ONLY over the top-`modulatory_shortlist_k`=3 shortlisted candidates (by
modulatory-adjusted cost), at `T_eff = base_temperature(1.0) +
alpha(1.0)*(1 - gap_norm)`, i.e. T_eff in [1.0, 2.0]. It never samples over the
full candidate/class pool. Two candidate explanations for a weak/negative lift:
(i) gap_norm saturates near 1.0 on most ticks, pinning T_eff near its 1.0 floor
-- still nonzero, but the softest end of the intended range is never reached;
(ii) the top-3 shortlist is itself usually concentrated on very few action
classes (a further-refined, more homogeneous subset of an already-concentrated
candidate pool), so softening the pick within it cannot diversify committed-
class entropy much even at high T_eff. 689i recorded `gap_scaled_commit_active_
ticks`/`_frac` (confirms Factor B FIRED) but NOT the actual `gap_scaled_commit_
gap_norm` / `gap_scaled_commit_temperature_eff` values `_gap_scaled_commit_pick`
already stamps into `last_score_diagnostics` per tick (e3_selector.py:1463-1467)
-- so there was no way to tell (i) from (ii) from the 689i manifest alone.

WHAT THIS SCRIPT ADDS (instrumentation only -- no ree_core changes, no
mechanism/parameter changes on either retained arm):
 1. Per fresh (genuine) selection where `gap_scaled_commit_active` fires, reads
    and accumulates `gap_scaled_commit_gap_norm` and
    `gap_scaled_commit_temperature_eff` from `last_score_diagnostics` (already
    computed by the substrate; this is a pure additional READ, using the exact
    same fresh-selection sentinel-key gating as every other MECH-448 readout in
    689i -- no substrate code is touched). Emits per-cell mean/std/min/max.
 2. A SCOPE-LIMITED proxy for shortlist class homogeneity: the top-`k`=3
    shortlist's own candidate composition is internal to `select()` and is NOT
    exposed to the experiment driver without a ree_core change, which is out of
    scope for an experiment script (that would be `/implement-substrate`
    territory, and risks perturbing the substrate under test). Instead this
    script computes `pool_dominant_class_share` -- the dominant class's share of
    the FULL proposer candidate pool (`pool_class_counts`, already recorded in
    689i as `proposer_pool_class_entropy` / `_classes_n_unique`) -- as an upper
    bound / proxy: if the full pool is already concentrated on one class, the
    further-refined top-3 shortlist cannot be more diverse than that. This is
    explicitly a PROXY, not the literal shortlist composition, and is reported
    as such.
 3. Seeds: 42-45 (689i's own 4, retained for direct comparability) plus 8 new
    seeds 46-53, for SEEDS=12 total. MIN_SEEDS_FOR_PASS scales proportionally:
    9 of 12 (unchanged 75% bar, up from 689i's 3 of 4).
 4. ARMS: ARM_PROPOSER_CTRL and ARM_MATCHED_NOISE ONLY -- byte-identical
    per-arm config to 689i (candidate_summary_source, temperature,
    use_f_eligibility_demotion, use_gap_scaled_commit_temperature,
    gap_scaled_commit_entropy_alpha). ARM_OFF and PRIMARY_ARM (ARM_ON) are
    REMOVED -- C_PRIMARY, C_READINESS, C_RANK_PRESERVING and C_SAFETY all read
    from ARM_ON/ARM_OFF rows in 689i and are therefore NOT EVALUATED here; they
    are 689i's business, already adjudicated, and this script cannot re-derive
    them without those arms. This halves the arm count, which is what makes the
    3x seed increase affordable at roughly comparable total compute to 689i's
    16-cell run (24 cells here vs 16 there, both at the SAME per-cell budget:
    N_FRESH_SELECT_TARGET=200 fresh selections, P0=60, P1_CAP=100,
    STEPS_PER_EPISODE=200 -- all UNCHANGED from 689i).

WHAT IS DELIBERATELY UNCHANGED (byte-identical to 689i for the two retained
arms): env config (ENV_KWARGS), agent config (all REEConfig kwargs), MECH-439/
MECH-448 lever settings, the Miller-Madow C_PRIMARY-class estimator applied here
to the SAME `selected_action_class_entropy_mm` DV, N_FRESH_SELECT_TARGET,
C_SUBSTRATE_INVARIANT / C_CONTROL_DISTINCT / C_FRESH_SUFFICIENT gates (the three
689d-defect-repair gates that must hold before any noise-control verdict is
meaningful), the sentinel-key fresh-selection-gating discipline (`_STALE_MARKER_
KEY`, following 689i/699b -- `last_score_diagnostics` and
`_last_selected_trajectory` are never nulled), and single-cloud-worker pinning
(`machine_affinity`) so C_SUBSTRATE_INVARIANT is meaningful over a
multi-hour run.

ACCEPTANCE (diagnostic; claim_ids=[MECH-448], PROMOTES/DEMOTES NOTHING).
Verdict chain -- the three repair gates precede the noise-control read, exactly
as in 689i, so a broken instrument can never produce a noise-control verdict:
  C_SUBSTRATE_INVARIANT (hard)  -> intra_run_substrate_divergence_invalid
  C_CONTROL_DISTINCT    (hard)  -> control_arms_not_distinct_invalid
  C_FRESH_SUFFICIENT    (readiness) -> substrate_not_ready_requeue
  C_NOISE_LIFTS_REPOWERED (load-bearing) ->
      matched_noise_control_repowered_lifting_confirmed (noise DOES verifiably
        lift at n=12 -- 689i's 1/4 reading was small-N noise; 689i's already-
        recorded C_PRIMARY finding is corroborated as resting on a valid
        instrument)
      matched_noise_control_repowered_still_unmeetable (noise still does not
        verifiably lift at n=12 -- the weak/negative effect is real at this
        power, not small-N noise; route to a design review of Factor B's
        shortlist-scoped sampling per the mechanistic hypothesis above, e.g. a
        higher `gap_scaled_commit_entropy_alpha` / `base_temperature`, or
        widening the softmax beyond the top-k shortlist -- NOT evidence against
        MECH-448, whose C_PRIMARY finding in 689i did not depend on this
        control's magnitude, only on it being NON-VACUOUS, which
        C_CONTROL_DISTINCT already separately guarantees)
evidence_direction is "non_contributory" on EVERY branch (diagnostic purpose,
no ARM_ON present -- this run cannot itself move MECH-448's evidence).

Usage:
  /opt/local/bin/python3 experiments/v3_exq_689j_mech448_factor_b_noise_control_repower.py --dry-run
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
from experiments._lib.readiness_anchor import (  # noqa: E402
    assert_anchor_reachable,
    AnchorUnreachable,
)
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_689j_mech448_factor_b_noise_control_repower"
QUEUE_ID = "V3-EXQ-689j"
# Companion diagnostic to 689i -- does NOT supersede it (689i's C_PRIMARY-based
# supports finding stands and is not re-tested here).
SUPERSEDES: Optional[str] = None
REPOWERS_RUN_ID = "v3_exq_689i_mech448_f_eligibility_demotion_falsifier_repair_20260722T162850Z_v3"
CLAIM_IDS: List[str] = ["MECH-448"]
EXPERIMENT_PURPOSE = "diagnostic"

# --- validate_experiments lint exemptions -------------------------------------
# Same exemption rationale as 689i (this script reuses its fresh-selection
# sentinel-key mechanism verbatim, unchanged). Both lints pattern-match on a
# literal `agent.e3.<attr> = None` clear preceding select_action(), and this
# script deliberately uses a substrate-inert SENTINEL KEY instead.
_FRESH_SELECT_EXEMPT_REASON = (
    "Freshness is enforced via a substrate-inert SENTINEL KEY stamped into "
    "agent.e3.last_score_diagnostics before every select_action(), not via a "
    "`= None` clear -- identical mechanism to V3-EXQ-689i, reused verbatim here. "
    "e3_selector.select() reassigns that dict wholesale, so the key survives "
    "iff select() did not run. All per-selection accumulation (including the "
    "NEW gap_norm/temperature_eff/pool-homogeneity diagnostics this script "
    "adds) is fresh-gated on the same marker. See "
    "failure_autopsy_V3-EXQ-689i_2026-07-24.json and "
    "failure_autopsy_V3-EXQ-689d_2026-07-20.json."
)
E3_DIAGNOSTICS_STALENESS_EXEMPT = _FRESH_SELECT_EXEMPT_REASON
E3_HOLD_WEIGHTED_READOUT_EXEMPT = _FRESH_SELECT_EXEMPT_REASON

# Private key stamped into agent.e3.last_score_diagnostics before every
# select_action(). Namespaced to THIS experiment so two concurrently-
# instrumented drivers can never collide (689i used its own namespaced key).
_STALE_MARKER_KEY = "_exq689j_stale_marker"

# 689i's own 4 seeds, retained for direct comparability, plus 8 new seeds.
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]   # 12 seeds (689i had 4)
P0_WARMUP_EPISODES = 60           # UNCHANGED from 689i (SD-056 proven budget)
P1_EPISODE_CAP = 100              # UNCHANGED from 689i
N_FRESH_SELECT_TARGET = 200       # UNCHANGED from 689i
STEPS_PER_EPISODE = 200           # UNCHANGED from 689i

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1_CAP = 3
DRY_RUN_STEPS = 30
DRY_RUN_FRESH_TARGET = 5

MIN_FRESH_SELECT_PER_CELL = N_FRESH_SELECT_TARGET  # gating sufficiency floor
# Proportional to 689i's 3-of-4 (75%): 9 of 12.
MIN_SEEDS_FOR_PASS = 9

# Factor B (MECH-439) -- UNCHANGED from 689i. Live ONLY on ARM_MATCHED_NOISE.
NOISE_ARM_COMMIT_ENTROPY_ALPHA = 1.0
NOISE_ARM_BASE_TEMPERATURE = 1.0

# MECH-448 lever config -- retained only for config-slice/fingerprint parity
# with 689i's arm definitions; use_f_eligibility_demotion is False on both
# retained arms, so these values are inert here (no demotion arm remains).
F_ELIGIBILITY_ENVELOPE_FLOOR = 0.30
F_ELIGIBILITY_DN_SIGMA = 0.0

# Shared shortlist / conversion constant (ON both retained arms) -- UNCHANGED.
MODULATORY_SHORTLIST_K = 3
MODULATORY_SHORTLIST_MODE = "top_k"
MODULATORY_AUTHORITY_GAIN = 2.0
MODULATORY_AUTHORITY_NORMALIZE_BASIS = "std"
MODULATORY_ROUTE_MIN_RANGE_FLOOR = 1e-6

# SD-056 online contrastive training -- UNCHANGED from 689i.
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
E2_TRAIN_EVERY_K_TICKS = 1
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0

# Behavioural-diversity env -- UNCHANGED from 689i.
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

# Arm ids -- ARM_OFF / PRIMARY_ARM (ARM_ON) DROPPED. Only the two controls
# whose relationship this script exists to repower remain.
PROPOSER_CTRL_ARM = "ARM_PROPOSER_CTRL"
MATCHED_NOISE_ARM = "ARM_MATCHED_NOISE"
CONTROL_ARMS = [PROPOSER_CTRL_ARM, MATCHED_NOISE_ARM]

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
        "arm_id": MATCHED_NOISE_ARM,
        "label": "proposer_gap_scaled_stochastic_commit_noise_negative_control",
        "candidate_summary_source": "proposer",
        "temperature": NOISE_ARM_BASE_TEMPERATURE,
        "use_f_eligibility_demotion": False,
        "use_gap_scaled_commit_temperature": True,
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """Identical conversion stack to 689i, restricted to the two retained
    arms. use_f_eligibility_demotion is False on both -- the F-eligibility
    lever is retained in config only for slice/fingerprint parity with 689i's
    arm definitions."""
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
        # SHARED E3-side bias channels
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
        # MECH-439 Factor B: ON only on ARM_MATCHED_NOISE.
        use_gap_scaled_commit_temperature=bool(arm["use_gap_scaled_commit_temperature"]),
        gap_scaled_commit_entropy_alpha=NOISE_ARM_COMMIT_ENTROPY_ALPHA,
        # MECH-448 (ARC-107) -- inert on both retained arms (False here).
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
    """Plug-in (maximum-likelihood) Shannon entropy in nats."""
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
    """Miller-Madow bias-corrected Shannon entropy in nats.

    H_MM = H_plugin + (K_obs - 1) / (2N). PORTED VERBATIM from 689i (which
    itself ported from the landed sibling v3_exq_699c), so the DV here is the
    SAME estimator gating this same quantity in 689i -- necessary for the
    seed42-45 cells to be directly comparable across the two runs.
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


def _stats(xs: List[float]) -> Dict[str, Any]:
    """Summary stats for a per-tick diagnostic stream (gap_norm / T_eff)."""
    if not xs:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    return {
        "mean": round(float(mean), 6),
        "std": round(float(math.sqrt(var)), 6),
        "min": round(float(min(xs)), 6),
        "max": round(float(max(xs)), 6),
        "n": int(n),
    }


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

    # C_PRIMARY-class DV (repowered here for the noise-control comparison only).
    selected_class_counts: Counter = Counter()
    pool_class_counts: Counter = Counter()

    # NEW (689j): Factor B mechanistic diagnostics -- gap_norm / T_eff, read
    # from last_score_diagnostics on the SAME fresh-gated tick as everything
    # else. Only ever populated on ARM_MATCHED_NOISE (gap_scaled_commit_active
    # is never True on ARM_PROPOSER_CTRL).
    gap_norms: List[float] = []
    commit_temps_eff: List[float] = []
    gap_scaled_commit_active_ticks = 0

    n_fresh_select = 0
    n_latched = 0
    n_p1_ticks = 0
    n_contrastive_steps = 0
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

            pending_pool_classes: Optional[List[int]] = None
            if is_p1 and candidates and len(candidates) >= 2:
                actions_K = _first_actions_K(candidates).to(agent.device)
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

            # --- FRESHNESS MARKER (identical mechanism to 689i/699b) -----------
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

            if is_p1:
                diag = _diag_now if (fresh_select and isinstance(_diag_now, dict)) else {}
                if fresh_select and bool(diag.get("gap_scaled_commit_active", False)):
                    gap_scaled_commit_active_ticks += 1
                    gn = float(diag.get("gap_scaled_commit_gap_norm", float("nan")))
                    if math.isfinite(gn):
                        gap_norms.append(gn)
                    teff = float(diag.get("gap_scaled_commit_temperature_eff", float("nan")))
                    if math.isfinite(teff):
                        commit_temps_eff.append(teff)

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

                if fresh_select:
                    n_fresh_select += 1
                    selected_class_counts[committed_class] += 1
                    if pending_pool_classes is not None:
                        for cls in pending_pool_classes:
                            pool_class_counts[cls] += 1
                else:
                    n_latched += 1

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

        if ep == 0 or (ep + 1) % 10 == 0 or (ep + 1) == total_train_eps:
            print(
                f"  [train] arm={arm['arm_id']} seed={seed} phase={phase_label} "
                f"ep {ep + 1}/{total_train_eps} fresh={n_fresh_select}/{fresh_target}",
                flush=True,
            )

        if error_note is not None:
            break

        if is_p1 and n_fresh_select >= fresh_target:
            target_met = True
            print(
                f"  [p1-done] arm={arm['arm_id']} seed={seed} "
                f"fresh_select target {fresh_target} met after {p1_episodes_run} P1 episodes",
                flush=True,
            )
            break

    selected_action_entropy = _entropy_from_counts(dict(selected_class_counts))
    selected_action_entropy_mm = _entropy_miller_madow(dict(selected_class_counts))
    proposer_pool_entropy = _entropy_from_counts(dict(pool_class_counts))
    pool_total = sum(pool_class_counts.values())
    pool_dominant_class_share = (
        float(max(pool_class_counts.values())) / float(pool_total)
        if pool_total > 0 else 0.0
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
        "n_fresh_select": int(n_fresh_select),
        "n_latched": int(n_latched),
        "fresh_select_yield": round(fresh_select_yield, 6),
        "replication_factor": round(replication_factor, 6),
        "fresh_select_target": int(fresh_target),
        "fresh_select_target_met": bool(target_met or n_fresh_select >= fresh_target),
        # --- C_PRIMARY-class DV, repowered here for the noise comparison only ---
        "selected_action_class_entropy_mm": round(selected_action_entropy_mm, 6),
        "selected_action_class_entropy": round(selected_action_entropy, 6),
        "miller_madow_correction_nats": round(
            selected_action_entropy_mm - selected_action_entropy, 6
        ),
        "selected_class_counts": dict(sorted(selected_class_counts.items())),
        "selected_classes_n_unique": int(len(selected_class_counts)),
        "proposer_pool_class_entropy": round(proposer_pool_entropy, 6),
        "proposer_pool_classes_n_unique": int(len(pool_class_counts)),
        # --- NEW (689j): Factor B mechanistic diagnostics ---
        "gap_scaled_commit_active_ticks": int(gap_scaled_commit_active_ticks),
        "gap_scaled_commit_active_frac": round(
            float(gap_scaled_commit_active_ticks) / float(n_fresh_select)
            if n_fresh_select > 0 else 0.0, 6
        ),
        "gap_scaled_commit_gap_norm_stats": _stats(gap_norms),
        "gap_scaled_commit_temperature_eff_stats": _stats(commit_temps_eff),
        "pool_dominant_class_share": round(pool_dominant_class_share, 6),
    }


# ---------------------------------------------------------------------------
# Shared readiness predicates -- THE SHIPPED CHECKS. Named at module level so
# the setup-time reachability guards (see _run_setup_anchor_guards) score their
# frozen references with the EXACT SAME callables _evaluate() scores the live
# run with -- a re-implementation for the guard would defeat the guard's whole
# purpose (readiness_anchor.py docstring rule 1).
# ---------------------------------------------------------------------------

def _hash_matches_reference(cell_hash: str, reference_hash: str) -> bool:
    """Per-cell half of C_SUBSTRATE_INVARIANT: does this cell's substrate_hash
    equal a fixed reference hash. `len(set(all_hashes)) == 1` (used in
    _evaluate) is exactly 'every cell's hash equals any one of them' -- so
    scoring every cell against the first observed hash is the same check,
    decomposed per-cell for the reachability guard."""
    return cell_hash == reference_hash


def _control_pair_distinct(pair: Tuple[Dict[str, int], Dict[str, int]]) -> bool:
    """Per-seed half of C_CONTROL_DISTINCT: do the two arms' committed-class
    count dicts differ on this seed."""
    a, b = pair
    return a != b


def _fresh_select_clears_floor(n_fresh_select: int) -> bool:
    """Per-cell half of C_FRESH_SUFFICIENT."""
    return int(n_fresh_select) >= MIN_FRESH_SELECT_PER_CELL


def _noise_lift_clears_epsilon(delta: float) -> bool:
    """Per-seed half of C_NOISE_LIFTS_REPOWERED: does the noise arm's
    Miller-Madow entropy exceed the proposer control's by more than the
    epsilon margin on this seed."""
    return float(delta) > 1e-6


# Frozen reference fixtures -- recorded values from V3-EXQ-689i's own landed
# run (run_id v3_exq_689i_mech448_f_eligibility_demotion_falsifier_repair_
# 20260722T162850Z_v3), which already cleared C_SUBSTRATE_INVARIANT,
# C_CONTROL_DISTINCT and C_FRESH_SUFFICIENT. Frozen as literals per
# readiness_anchor.py rule 2 -- no compute, cannot drift with the substrate.
_REF_689I_SUBSTRATE_HASH = (
    "3c7dfea1b222ecdf54558c087f226f9e79680c871124f11c4299850ad454425d"
)
_REF_689I_CELL_HASHES = [_REF_689I_SUBSTRATE_HASH] * 16  # 4 arms x 4 seeds, all identical

# (proposer_selected_class_counts, noise_selected_class_counts) per seed 42-45.
_REF_689I_CONTROL_PAIRS: List[Tuple[Dict[str, int], Dict[str, int]]] = [
    ({0: 39, 1: 80, 2: 26, 3: 55, 4: 8}, {0: 51, 1: 84, 2: 27, 3: 50, 4: 4}),   # seed 42
    ({1: 1, 2: 217}, {2: 202}),                                                 # seed 43
    ({0: 85, 2: 17, 4: 104}, {0: 42, 2: 29, 4: 133}),                           # seed 44
    ({0: 3, 1: 6, 3: 199, 4: 2}, {0: 1, 1: 14, 3: 194, 4: 3}),                  # seed 45
]

# n_fresh_select per cell, ARM_PROPOSER_CTRL + ARM_MATCHED_NOISE, seeds 42-45.
_REF_689I_FRESH_SELECT_COUNTS = [208, 218, 206, 210, 216, 202, 204, 212]

# SYNTHETIC reference for C_NOISE_LIFTS_REPOWERED -- 689i's own 4-seed reading
# is precisely the underpowered case under adjudication, so it cannot serve as
# a "known-lifting" positive control. This fixture instead proves the COUNTING
# LOGIC is not narrower than intended: 9 deltas clearly above the epsilon
# margin and 3 clearly below, the exact boundary shape MIN_SEEDS_FOR_PASS=9-of-
# 12 needs to clear. It is a logic-soundness check on _noise_lift_clears_
# epsilon + the fraction threshold, NOT an empirical claim that the true effect
# exists -- documented per readiness_anchor.py's guidance on synthetic
# fixtures for population-fraction gates with no available real positive
# control.
_REF_SYNTHETIC_NOISE_LIFT_DELTAS = [0.05] * 9 + [-0.03] * 3


def _run_setup_anchor_guards() -> List[Dict[str, Any]]:
    """Reachability guards (readiness_anchor.py) for the four readiness-kind
    preconditions this script declares, run BEFORE the expensive multi-hour
    arm/seed loop. Raises AnchorUnreachable (refusing to run) if any predicate
    cannot score its own frozen reference above its gate -- catching a
    mis-specified predicate at design-audit time rather than after the run."""
    payloads: List[Dict[str, Any]] = []

    payloads.append(assert_anchor_reachable(
        anchor_name="cells_share_one_substrate_hash",
        reference_cells=_REF_689I_CELL_HASHES,
        score_fn=lambda h: _hash_matches_reference(h, _REF_689I_SUBSTRATE_HASH),
        threshold=1.0,
        reference_source=(
            "V3-EXQ-689i run_id v3_exq_689i_mech448_f_eligibility_demotion_"
            "falsifier_repair_20260722T162850Z_v3 -- all 16 cells recorded "
            "identical substrate_hash"
        ),
    ))
    payloads.append(assert_anchor_reachable(
        anchor_name="control_arms_produce_distinct_class_histograms",
        reference_cells=_REF_689I_CONTROL_PAIRS,
        score_fn=_control_pair_distinct,
        threshold=1.0,
        reference_source=(
            "V3-EXQ-689i ARM_PROPOSER_CTRL vs ARM_MATCHED_NOISE selected_class_"
            "counts, seeds 42-45 -- all 4 pairs recorded distinct"
        ),
    ))
    payloads.append(assert_anchor_reachable(
        anchor_name="fresh_e3_selection_sufficiency_all_cells",
        reference_cells=_REF_689I_FRESH_SELECT_COUNTS,
        score_fn=_fresh_select_clears_floor,
        threshold=1.0,
        reference_source=(
            "V3-EXQ-689i ARM_PROPOSER_CTRL + ARM_MATCHED_NOISE n_fresh_select, "
            "seeds 42-45 -- min observed 202 >= floor 200"
        ),
    ))
    payloads.append(assert_anchor_reachable(
        anchor_name="matched_noise_control_verifiably_lifts_repowered",
        reference_cells=_REF_SYNTHETIC_NOISE_LIFT_DELTAS,
        score_fn=_noise_lift_clears_epsilon,
        threshold=float(MIN_SEEDS_FOR_PASS) / float(len(SEEDS)),
        reference_source=(
            "SYNTHETIC logic-soundness fixture (9 deltas at +0.05, 3 at -0.03) "
            "-- 689i's own 4-seed reading is the underpowered case under "
            "adjudication and cannot serve as a known-lifting positive "
            "control; this proves the counting/threshold logic itself is not "
            "narrower than the 9-of-12 boundary it must clear"
        ),
    ))
    return payloads


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
    """C_CONTROL_DISTINCT, trimmed to the two retained arms. Any seed where
    ARM_PROPOSER_CTRL and ARM_MATCHED_NOISE produce an identical committed-
    class count vector reproduces the 689d defect-2 signature (temperature
    inert on a deterministic path) and invalidates the run."""
    by_seed: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for r in arm_results:
        by_seed.setdefault(int(r["seed"]), {})[str(r["arm_id"])] = r
    hits: List[Dict[str, Any]] = []
    for seed, arms in sorted(by_seed.items()):
        ra, rb = arms.get(CONTROL_ARMS[0]), arms.get(CONTROL_ARMS[1])
        if ra is None or rb is None:
            continue
        if not _control_pair_distinct(
            (ra.get("selected_class_counts"), rb.get("selected_class_counts"))
        ):
            hits.append({
                "seed": seed,
                "arm_a": CONTROL_ARMS[0],
                "arm_b": CONTROL_ARMS[1],
                "class_counts": ra.get("selected_class_counts"),
            })
    return hits


def _evaluate(arm_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    proposer = _arm_rows(arm_results, PROPOSER_CTRL_ARM)
    noise = _arm_rows(arm_results, MATCHED_NOISE_ARM)
    proposer_by_seed = {r["seed"]: r for r in proposer}

    SENT = "selected_action_class_entropy_mm"

    # === C_SUBSTRATE_INVARIANT (carried over from 689i) =======================
    hashes = _substrate_hashes(arm_results)
    distinct_hashes = sorted(set(hashes))
    # Equivalent to len(distinct_hashes) == 1, expressed via the SAME per-cell
    # predicate the setup-time reachability guard scores its frozen reference
    # with: every cell's hash equals a fixed reference (the first observed).
    substrate_invariant = bool(
        hashes and all(_hash_matches_reference(h, hashes[0]) for h in hashes)
    )

    # === C_CONTROL_DISTINCT (carried over from 689i, 2-arm form) ==============
    identical_pairs = _identical_control_pairs(arm_results)
    control_distinct = bool(len(identical_pairs) == 0)

    # === C_FRESH_SUFFICIENT (carried over from 689i) ===========================
    all_rows = list(arm_results)
    cells_short = [
        {
            "arm_id": r["arm_id"], "seed": r["seed"],
            "n_fresh_select": int(r.get("n_fresh_select", 0)),
        }
        for r in all_rows
        if not _fresh_select_clears_floor(int(r.get("n_fresh_select", 0)))
    ]
    min_fresh = min([int(r.get("n_fresh_select", 0)) for r in all_rows] or [0])
    fresh_sufficient = bool(len(cells_short) == 0)

    # === C_NOISE_LIFTS_REPOWERED (the sole load-bearing criterion here) =======
    def _noise_lifts(rn: Dict[str, Any]) -> bool:
        rp = proposer_by_seed.get(rn["seed"])
        if rp is None:
            return False
        return _noise_lift_clears_epsilon(
            float(rn.get(SENT, 0.0)) - float(rp.get(SENT, 0.0))
        )
    noise_lift_seeds = _n_seeds(noise, _noise_lifts)
    matched_noise_verified_lifting = bool(noise_lift_seeds >= MIN_SEEDS_FOR_PASS)

    # Per-seed deltas, recorded generously for direct comparison against 689i's
    # own 4-seed reading (seed42-45 overlap exactly).
    per_seed_delta = []
    for r in noise:
        rp = proposer_by_seed.get(r["seed"])
        if rp is None:
            continue
        per_seed_delta.append({
            "seed": r["seed"],
            "noise_entropy_mm": float(r.get(SENT, 0.0)),
            "proposer_entropy_mm": float(rp.get(SENT, 0.0)),
            "delta": round(float(r.get(SENT, 0.0)) - float(rp.get(SENT, 0.0)), 6),
            "lifts": bool(_noise_lifts(r)),
        })
    per_seed_delta.sort(key=lambda d: d["seed"])

    all_arms = [proposer, noise]
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
            "-- the two arms are not mutually controlled"
        )
    elif not control_distinct:
        degeneracy_reason = (
            f"{len(identical_pairs)} seed(s) where ARM_MATCHED_NOISE and "
            "ARM_PROPOSER_CTRL produced identical committed-class count vectors "
            "-- the 689d defect-2 signature (temperature inert on this path)"
        )

    # === VERDICT CHAIN ==========================================================
    if not substrate_invariant:
        label = "intra_run_substrate_divergence_invalid"
        overall_pass = False
    elif not control_distinct:
        label = "control_arms_not_distinct_invalid"
        overall_pass = False
    elif not fresh_sufficient:
        label = "substrate_not_ready_requeue"
        overall_pass = False
    elif matched_noise_verified_lifting:
        label = "matched_noise_control_repowered_lifting_confirmed"
        overall_pass = True
    else:
        label = "matched_noise_control_repowered_still_unmeetable"
        overall_pass = False

    # Diagnostic purpose, no ARM_ON present -> this run never moves MECH-448's
    # evidence directly, on either branch.
    evidence_direction = "non_contributory"

    # NEW (689j): Factor B mechanistic diagnostics, non-gating.
    noise_gap_norm_means = [
        float(r.get("gap_scaled_commit_gap_norm_stats", {}).get("mean", 0.0))
        for r in noise
    ]
    noise_temp_eff_means = [
        float(r.get("gap_scaled_commit_temperature_eff_stats", {}).get("mean", 0.0))
        for r in noise
    ]
    proposer_pool_dominant_shares = [
        float(r.get("pool_dominant_class_share", 0.0)) for r in proposer
    ]
    noise_pool_dominant_shares = [
        float(r.get("pool_dominant_class_share", 0.0)) for r in noise
    ]

    factor_b_diagnostics = {
        "noise_arm_gap_norm_mean_across_seeds": round(
            float(sum(noise_gap_norm_means) / len(noise_gap_norm_means))
            if noise_gap_norm_means else 0.0, 6
        ),
        "noise_arm_temperature_eff_mean_across_seeds": round(
            float(sum(noise_temp_eff_means) / len(noise_temp_eff_means))
            if noise_temp_eff_means else 0.0, 6
        ),
        "noise_arm_gap_scaled_commit_active_frac_mean": round(
            _mean_key(noise, "gap_scaled_commit_active_frac"), 6
        ),
        "pool_dominant_class_share_mean": {
            PROPOSER_CTRL_ARM: round(
                float(sum(proposer_pool_dominant_shares) / len(proposer_pool_dominant_shares))
                if proposer_pool_dominant_shares else 0.0, 6
            ),
            MATCHED_NOISE_ARM: round(
                float(sum(noise_pool_dominant_shares) / len(noise_pool_dominant_shares))
                if noise_pool_dominant_shares else 0.0, 6
            ),
        },
        "per_seed_gap_norm_and_temp_eff": [
            {
                "seed": r["seed"],
                "gap_norm_stats": r.get("gap_scaled_commit_gap_norm_stats"),
                "temperature_eff_stats": r.get("gap_scaled_commit_temperature_eff_stats"),
                "pool_dominant_class_share": r.get("pool_dominant_class_share"),
            }
            for r in noise
        ],
        "note": (
            "NON-GATING mechanistic diagnostics added by 689j to explain, not "
            "adjudicate, the noise-control's lift magnitude. gap_norm near 1.0 "
            "(temperature_eff near its 1.0 floor) means Factor B rarely reaches "
            "the softer end of its intended [1.0, 2.0] range. A high "
            "pool_dominant_class_share means the full candidate pool -- an "
            "UPPER BOUND on the top-3 shortlist's own diversity, which is not "
            "directly observable without a ree_core change (out of scope for "
            "this experiment script) -- is already concentrated on one class, "
            "so no amount of softmax temperature within that shortlist can "
            "diversify the committed class much. Read alongside "
            "matched_noise_control_repowered_still_unmeetable (if that is the "
            "outcome) to decide which explanation, if either, dominates."
        ),
    }

    return {
        "c_substrate_invariant": {
            "n_distinct_substrate_hashes": int(len(distinct_hashes)),
            "distinct_substrate_hashes": distinct_hashes,
            "n_cells": int(len(arm_results)),
            "n_cells_with_hash": int(len(hashes)),
            "c_substrate_invariant_pass": substrate_invariant,
        },
        "c_control_distinct": {
            "identical_control_pairs": identical_pairs,
            "n_identical_control_pairs": int(len(identical_pairs)),
            "control_arms_checked": CONTROL_ARMS,
            "c_control_distinct_pass": control_distinct,
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
        },
        "c_noise_lifts_repowered": {
            "primary_dv_key": SENT,
            "noise_lift_seeds": int(noise_lift_seeds),
            "min_seeds_required": int(MIN_SEEDS_FOR_PASS),
            "n_seeds_total": int(len(SEEDS)),
            "matched_noise_verified_lifting": matched_noise_verified_lifting,
            "per_seed_delta": per_seed_delta,
            "note": (
                "REPOWERED version of 689i's C_NOISE_LIFTS gate (the only 689i "
                "criterion this script re-evaluates). Same predicate: "
                "selected_action_class_entropy_mm (Miller-Madow corrected) on "
                "ARM_MATCHED_NOISE strictly above ARM_PROPOSER_CTRL, same seed, "
                "by more than 1e-6, on >= MIN_SEEDS_FOR_PASS of the seeds run "
                "(scaled proportionally: 9 of 12, vs 689i's 3 of 4). Does NOT "
                "gate or re-evaluate C_PRIMARY / C_READINESS / C_RANK_PRESERVING "
                "/ C_SAFETY -- those require ARM_ON/ARM_OFF, dropped here."
            ),
        },
        "factor_b_diagnostics": factor_b_diagnostics,
        "label": label,
        "evidence_direction": evidence_direction,
        "overall_pass": overall_pass,
        "preconditions": [
            {
                "name": "cells_share_one_substrate_hash",
                "kind": "readiness",
                "description": (
                    "The set of per-cell arm_fingerprint.substrate_hash values "
                    "across ALL cells has cardinality exactly 1 (carried over "
                    "from 689i's C_SUBSTRATE_INVARIANT defect-3 repair)."
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
                    "ARM_PROPOSER_CTRL and ARM_MATCHED_NOISE do not produce an "
                    "identical committed-class count vector on the same seed "
                    "(carried over from 689i's C_CONTROL_DISTINCT defect-2 repair, "
                    "trimmed to the two retained arms)."
                ),
                "control": (
                    "pairwise comparison of selected_class_counts across "
                    "ARM_PROPOSER_CTRL / ARM_MATCHED_NOISE, per seed"
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
                    "Every cell banked at least MIN_FRESH_SELECT_PER_CELL "
                    "GENUINE E3 selections (carried over from 689i's "
                    "C_FRESH_SUFFICIENT defect-1/power repair, UNCHANGED target)."
                ),
                "control": (
                    "sentinel-key freshness marker on "
                    "agent.e3.last_score_diagnostics, which e3_selector.select() "
                    "reassigns wholesale"
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
                "name": "matched_noise_control_verifiably_lifts_repowered",
                "kind": "readiness",
                "description": (
                    "REPOWERED read (12 seeds vs 689i's 4) of whether the noise "
                    "negative control raises committed-class entropy over the "
                    "collapsed proposer baseline on >= MIN_SEEDS_FOR_PASS seeds. "
                    "689i's per-seed deltas (-0.038, -0.032, -0.035, +0.089) "
                    "showed a consistent small-negative cluster on 3/4 seeds, "
                    "not scatter around a positive mean -- this is the read of "
                    "whether that pattern persists at n=12."
                ),
                "control": (
                    "ARM_MATCHED_NOISE (use_gap_scaled_commit_temperature=True) "
                    "vs ARM_PROPOSER_CTRL, same seed"
                ),
                "measured": int(noise_lift_seeds),
                "threshold": int(MIN_SEEDS_FOR_PASS),
                "direction": "lower",
                "comparator": ">=",
                "met": matched_noise_verified_lifting,
            },
        ],
        "criteria": [
            {"name": "C_SUBSTRATE_INVARIANT_cells_share_one_build", "load_bearing": True,
             "passed": substrate_invariant},
            {"name": "C_CONTROL_DISTINCT_no_identical_control_histograms", "load_bearing": True,
             "passed": control_distinct},
            {"name": "C_FRESH_SUFFICIENT_effective_n_meets_target", "load_bearing": True,
             "passed": fresh_sufficient},
            {"name": "C_NOISE_LIFTS_REPOWERED_matched_noise_verifiably_lifts",
             "load_bearing": True, "passed": matched_noise_verified_lifting},
        ],
        "criteria_non_degenerate": {
            "C_SUBSTRATE_INVARIANT": non_degenerate,
            "C_CONTROL_DISTINCT": non_degenerate,
            "C_FRESH_SUFFICIENT": non_degenerate,
            "C_NOISE_LIFTS_REPOWERED": non_degenerate,
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
            "min_seeds_for_pass": MIN_SEEDS_FOR_PASS,
            "min_fresh_select_per_cell": MIN_FRESH_SELECT_PER_CELL,
        },
        "repowers_run_id": REPOWERS_RUN_ID,
    }

    # Setup-time readiness-anchor reachability guards -- BEFORE the expensive
    # multi-hour arm/seed loop. Raises AnchorUnreachable (refuses to run) if
    # any of the four readiness predicates cannot score its own frozen
    # reference above its gate.
    anchor_guard_payloads = _run_setup_anchor_guards()
    print(
        f"Readiness-anchor reachability guards: {len(anchor_guard_payloads)} "
        "checked, all reachable.",
        flush=True,
    )

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
        "repowers_run_id": REPOWERS_RUN_ID,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": {"MECH-448": evidence_direction},
        "non_degenerate": summary.get("non_degenerate", True),
        "degeneracy_reason": summary.get("degeneracy_reason", ""),
        "evidence_direction_note": (
            "TARGETED instrument follow-up to V3-EXQ-689i's ONE blocked gate "
            "(C_NOISE_LIFTS) -- NOT a re-test of MECH-448's C_PRIMARY finding, "
            "which stands as recorded in 689i (confirmed autopsy, "
            "user-adjudicated 'gate defect, science upheld' 2026-07-24). ARM_OFF "
            "and ARM_ON are absent from this run by design, so this diagnostic "
            "cannot itself move MECH-448's evidence -- evidence_direction is "
            "non_contributory regardless of outcome. If "
            "matched_noise_control_repowered_lifting_confirmed: 689i's 1/4-seed "
            "reading was small-N noise, and 689i's C_PRIMARY-based supports "
            "finding is corroborated as resting on a valid (non-vacuous, "
            "genuinely lifting) noise control. If "
            "matched_noise_control_repowered_still_unmeetable: the weak/negative "
            "effect persists at n=12 and is likely real at this power -- route to "
            "a design review of Factor B's shortlist-scoped gap-scaled sampling "
            "(see factor_b_diagnostics for gap_norm/temperature_eff/pool-"
            "homogeneity readouts that bear on which mechanistic explanation "
            "dominates), NOT a weakening of MECH-448 (C_CONTROL_DISTINCT already "
            "separately guarantees the control is non-vacuous in 689i, "
            "independent of this control's magnitude)."
        ),
        "interpretation": {
            "label": summary["label"],
            "preconditions": summary["preconditions"],
            "criteria": summary["criteria"],
            "criteria_non_degenerate": summary["criteria_non_degenerate"],
            "routing": {
                "matched_noise_control_repowered_lifting_confirmed": "689i's C_NOISE_LIFTS 1/4 reading was small-N noise; the noise control verifiably lifts at n=12 -- corroborates 689i's C_PRIMARY finding as resting on a valid instrument. No governance action beyond noting the corroboration in 689i's evidence_quality_note.",
                "matched_noise_control_repowered_still_unmeetable": "the weak/negative effect on the noise control persists at n=12 -- route to a design review of Factor B (gap_scaled_commit_entropy_alpha / base_temperature / shortlist-scoped sampling per factor_b_diagnostics), NOT a weakening of MECH-448 (which does not depend on this control's magnitude, only its non-vacuity, already separately guaranteed by C_CONTROL_DISTINCT).",
                "intra_run_substrate_divergence_invalid": "cells did not share one substrate build -- the two arms are not mutually controlled; re-run at a pinned commit on a cloud worker.",
                "control_arms_not_distinct_invalid": "the two arms produced identical committed-class histograms on some seed -- reproduces the 689d defect-2 signature; investigate before any noise-control read.",
                "substrate_not_ready_requeue": "insufficient GENUINE E3 selections in some cell; re-queue.",
            },
        },
        "dry_run": bool(dry_run),
        "config": full_config,
        "acceptance_criteria": {
            "C_SUBSTRATE_INVARIANT_cells_share_one_build": summary["c_substrate_invariant"]["c_substrate_invariant_pass"],
            "C_CONTROL_DISTINCT_no_identical_control_histograms": summary["c_control_distinct"]["c_control_distinct_pass"],
            "C_FRESH_SUFFICIENT_effective_n_meets_target": summary["c_fresh_sufficient"]["c_fresh_sufficient_pass"],
            "C_NOISE_LIFTS_REPOWERED_matched_noise_verified_lifting": summary["c_noise_lifts_repowered"]["matched_noise_verified_lifting"],
            "overall_pass": summary["overall_pass"],
        },
        "summary": summary,
        "arm_results": arm_results,
        "readiness_anchor_guards": anchor_guard_payloads,
    }

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
        "  noise_lift_seeds: "
        f"{summary['c_noise_lifts_repowered']['noise_lift_seeds']} "
        f"/ {len(seeds)} (need >= {MIN_SEEDS_FOR_PASS})",
        flush=True,
    )

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "V3-EXQ-689j MECH-448 (ARC-107) Factor B noise-control instrument "
            "repower -- targeted follow-up to V3-EXQ-689i's C_NOISE_LIFTS gate"
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
