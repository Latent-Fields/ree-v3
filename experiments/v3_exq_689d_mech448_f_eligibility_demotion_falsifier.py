"""V3-EXQ-689d: MECH-448 (ARC-107) rank-preserving F->eligibility demotion
FALSIFIER -- the 689a-successor. NEW question (the LEAD lever of the ARC-107
basal-ganglia E3-selector constitution), NOT a re-queue of the conflict-grade
near-tie parametric family (689/689a/689c, adjudicated exhausted by the
user-ratified failure_autopsy_V3-EXQ-689a_2026-06-20 Step-8 decision).

WHAT MECH-448 DOES (e3_selector.py: _f_eligibility_envelope + the "f_demotion"
mode of the shortlist-then-modulate block). The conversion ceiling is that F
(the primary harm/goal score) monopolises ~88-89%% of E3 committed-selection
variance (V3-EXQ-571) -- every diversity channel reaches the E3 accumulator but
cannot move the F-dominated committed argmin. MECH-448 is constitutional, not
parametric: F decides who is ELIGIBLE to compete, not who wins. F is renormalised
against the COMPETING FIELD by a divisive-normalisation analog with an ABSOLUTE
share floor:
    merit[i] = clamp(raw_scores.max() - raw_scores[i], min=0)   # best (lowest F) = highest merit
    pooled   = f_eligibility_dn_sigma + merit.sum()
    elig[i]  = merit[i] / pooled                                # share of the competing field
    eligible = { i : elig[i] >= f_eligibility_envelope_floor }  # ABSOLUTE share floor
The absolute share floor is load-bearing: a decisive F-winner commands most of
the merit share so others fall below the floor (NARROW envelope); a near-tie
spreads the share (WIDE envelope) -- the BG hyperdirect conflict-grade emerging
from the field structure, NOT a hard top-k count (which is env-conditional: 569i
top-k works only on the reef-bipartite guarantee and thinly cleared 2/3). elig is
monotone in -F, so the eligible set is an F-RANK PREFIX (rank-preserving). Within
the eligible set the EXISTING _modulatory_accum arbitration picks the committed
action (argmin committed / softmax uncommitted) -- F REMOVED from the final argmin.

THE ARM CONTRAST (matched seeds; ARM_OFF vs ARM_ON + a VERIFIED-LIFTING
matched-noise control):
  ARM_PROPOSER_CTRL  proposer source, demotion OFF, top_k k=3, T=1.0
    (collapsed-channel baseline -- the no-conversion-reaches floor; the
    committed-entropy bar ARM_ON must clear ["reach the proposer ceiling"]).
  ARM_MATCHED_NOISE  proposer source, demotion OFF, top_k k=3, T=2.5
    (flat hot softmax over the COLLAPSED proposer channel; the noise-as-diversity
    negative control -- VERIFIED-LIFTING: entropy from sampling noise, NOT lawful
    access. ARM_ON must beat THIS, so a lift cannot be "added stochasticity").
  ARM_OFF            e2wf source, demotion OFF, top_k k=3, T=1.0
    (the GAP-A conversion baseline = the HARD env-conditional top-k eligibility
    set the 569i lever tested; the within-eligible _modulatory_accum arbitration
    is identical -- the ONLY difference vs ARM_ON is the eligibility-set
    construction: hard top-k F-prefix vs graded DN-share-floor envelope).
  ARM_ON (PRIMARY)   e2wf source, demotion ON (use_f_eligibility_demotion=True),
    graded DN envelope (floor 0.30), T=1.0.

ALL four arms share the GAP-A-ready conversion constant: SD-056 online-trained
e2.world_forward (rollout-norm clamp ON, 643a lesson) + route-range routing +
shortlist-then-modulate + SP-CEM Layer A + shared lateral_pfc/mech295 bias
channels (so _modulatory_accum is non-None -- the MECH-448 non-vacuity
precondition). The two proposer arms keep the COLLAPSED proposer source; the two
e2wf arms use ARC-065 GAP-A candidate_summary_source=e2_world_forward (the
DIVERGENT eligible set the envelope must actually exclude over). The conflict-grade
levers (Factor A / Factor B, MECH-439) are OFF on every arm -- this experiment is
the constitutional lever, not the parametric family. CRF stack off.

NON-VACUITY PRECONDITION (load-bearing): MECH-448 fires only when a modulatory
channel is active AND the candidate F pool is DIVERGENT. f_eligibility_excluded_count
> 0 across arms on the divergent pool -- a flat-F pool gives a wide all-admit
envelope (excluded_count == 0), which is a vacuous self-route. Below floor
self-routes substrate_not_ready_requeue, NEVER a false weakens.

ACCEPTANCE (evidence, claim_ids=[MECH-448]; arc_107 design note section 4):
  READINESS (load-bearing non-vacuity; below any -> substrate_not_ready_requeue,
  NEVER a weakens -- RANGE / non-degeneracy statistics):
    (a) IN-ARM ROUTE-RANGE: ARM_ON modulatory_channel_route_range mean (read LIVE
        at the select tick) > ROUTE_RANGE_FLOOR on >= MIN_SEEDS_FOR_PASS seeds.
    (b) E2-DIVERGENCE: ARM_ON cand_world_pairwise_dist > C1_PAIRWISE_DIST_FLOOR on
        >= MIN_SEEDS_FOR_PASS seeds (SD-056 trained the action-conditional spread).
    (c) NON-DEGENERACY: ARM_ON f_eligibility_demotion_active on (nearly) all P1
        ticks AND mean f_eligibility_excluded_count > EXCLUDED_COUNT_FLOOR (the
        envelope ACTUALLY excluded on the divergent pool, not all-admit) on
        >= MIN_SEEDS_FOR_PASS seeds.
  C_RANK_PRESERVING (LOAD-BEARING -- "order preserved on the numerators"): ARM_ON
    f_eligibility_rank_preserving fraction == 1.0 (every eligible cost <= every
    excluded cost, tie-robust) on >= MIN_SEEDS_FOR_PASS seeds. F still RANKS within
    the eligible set; the demotion is rank-preserving, not a global F-flatten.
  C_PRIMARY (gate -- "committed-class entropy reaches the proposer ceiling"):
    ARM_ON selected_action_class_entropy STRICTLY ABOVE BOTH ARM_PROPOSER_CTRL AND
    ARM_MATCHED_NOISE on the SAME seed, on >= MIN_SEEDS_FOR_PASS seeds, AND ARM_ON
    mean > C3_SELECTED_ENTROPY_FLOOR (no-lift-at-all off-ramp). Beating the
    collapsed baseline = reaches the proposer ceiling; beating matched-noise =
    NOT noise-as-diversity.
  C_SAFETY (LOAD-BEARING -- "no harmful action class globally disinhibited"):
    ARM_ON mean realised harm-per-P1-tick <= ARM_OFF mean harm-per-P1-tick +
    SAFETY_HARM_TOL on the SAME seed, on >= MIN_SEEDS_FOR_PASS seeds. The envelope
    keeps clearly-harmful (near-zero-merit) candidates EXCLUDED, so demoting F does
    NOT raise hazard exposure over the hard top-k baseline.
  PASS = READINESS AND C_RANK_PRESERVING AND C_PRIMARY AND C_SAFETY ->
    demotion_converts_committed_diversity; evidence_direction=supports.

Interpretation grid:
| outcome                                                  | label                                          | evidence_direction | next                                                                                            |
|----------------------------------------------------------|------------------------------------------------|--------------------|-------------------------------------------------------------------------------------------------|
| READINESS + rank-preserving + C_PRIMARY + C_SAFETY       | demotion_converts_committed_diversity          | supports           | MECH-448 toward supports; ARC-107 gains its first validated lever                               |
| route/e2-div below floor OR excluded_count==0 OR demotion not active | substrate_not_ready_requeue        | non_contributory   | routing/SD-056 under-trained OR pool not divergent (all-admit envelope); re-queue; NOT a weakens |
| READINESS met, rank_preserving fraction < 1.0            | rank_alteration_not_prefix_diagnose            | non_contributory   | the eligible set is NOT an F-rank prefix -> impl/design fault -> /diagnose-errors; NOT a weakens |
| READINESS + rank met, C_PRIMARY fail (no lift)           | conversion_ceiling_persists_despite_demotion   | non_contributory   | OFF-RAMP -> MECH-449 Go/No-Go constitution (double-gated) / V4 directions; NOT a falsification   |
| READINESS + C_PRIMARY met, C_SAFETY fail (harm up)       | demotion_disinhibits_harmful_classes           | weakens            | the lift admits harmful classes / global F-flatten -> the design-note WEAKENED condition         |

claim_ids=[MECH-448] only; ARC-107/MECH-447/MECH-449/MECH-439/Q-078 untouched.
MECH-448 stays candidate; a PASS moves it toward supports, a preconditions-met
no-lift routes to the MECH-449 follow-on (double-gated) / V4, and only a safety
fail (admits harmful classes) weakens.

Usage:
  /opt/local/bin/python3 experiments/v3_exq_689d_mech448_f_eligibility_demotion_falsifier.py --dry-run
"""

import argparse
import json
import math
import random
import sys
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
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_689d_mech448_f_eligibility_demotion_falsifier"
QUEUE_ID = "V3-EXQ-689d"
SUPERSEDES: Optional[str] = None  # NEW question (constitutional lever), not a fix of 689a
CLAIM_IDS: List[str] = ["MECH-448"]  # MECH-448's FIRST falsifier (the ARC-107 lead lever)
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 60           # SD-056 contrastive warmup (569i/689a proven budget)
P1_MEASUREMENT_EPISODES = 20
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_STEPS = 30

# Acceptance thresholds (pre-registered).
ROUTE_RANGE_FLOOR = 0.01          # READINESS (a): IN-ARM modulatory_channel_route_range
C1_PAIRWISE_DIST_FLOOR = 0.03     # READINESS (b): ARM_ON e2.world_forward prediction spread
EXCLUDED_COUNT_FLOOR = 0.0        # READINESS (c): mean f_eligibility_excluded_count strictly above this
DEMOTION_ACTIVE_FRAC_FLOOR = 0.8  # READINESS (c): fraction of ARM_ON P1 ticks with demotion active
RANK_PRESERVING_FRAC_REQUIRED = 1.0  # C_RANK_PRESERVING: every active tick rank-preserving
C3_SELECTED_ENTROPY_FLOOR = 0.3   # C_PRIMARY: ARM_ON selected-action class entropy floor
PROPOSER_CEILING_FRAC = 0.6       # informational: ARM_ON committed entropy as a fraction of the offered proposer ceiling
SAFETY_HARM_TOL = 0.02            # C_SAFETY: ARM_ON harm-per-tick may exceed ARM_OFF by at most this
MATCHED_ENTROPY_TEMPERATURE = 2.5
MIN_SEEDS_FOR_PASS = 2            # of 3

# MECH-448 lever config (ARC-107).
F_ELIGIBILITY_ENVELOPE_FLOOR = 0.30   # absolute DN-share floor (substrate default)
F_ELIGIBILITY_DN_SIGMA = 0.0          # DN semi-saturation (substrate default; >0 narrows)

# Shared shortlist / conversion constant (ON all arms; the within-eligible arbitration).
MODULATORY_SHORTLIST_K = 3
MODULATORY_SHORTLIST_MODE = "top_k"
MODULATORY_AUTHORITY_GAIN = 2.0
MODULATORY_AUTHORITY_NORMALIZE_BASIS = "std"
MODULATORY_ROUTE_MIN_RANGE_FLOOR = 1e-6

# SD-056 online contrastive training (mirror 569i/689a harness).
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
E2_TRAIN_EVERY_K_TICKS = 1
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0

# Behavioural-diversity env: SD-054 reef-bipartite hazard layout (matches 569i/689a exactly).
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
PRIMARY_ARM = "ARM_ON"             # use_f_eligibility_demotion=True (graded DN envelope)
OFF_ARM = "ARM_OFF"                # demotion OFF, hard top_k F-prefix (the GAP-A baseline)
PROPOSER_CTRL_ARM = "ARM_PROPOSER_CTRL"
MATCHED_NOISE_ARM = "ARM_MATCHED_NOISE"

# Per-arm fields: candidate_summary_source / temperature / f_eligibility_demotion toggle.
ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": PROPOSER_CTRL_ARM,
        "label": "proposer_collapsed_channel_baseline_control",
        "candidate_summary_source": "proposer",
        "temperature": 1.0,
        "use_f_eligibility_demotion": False,
    },
    {
        "arm_id": MATCHED_NOISE_ARM,
        "label": "proposer_matched_entropy_flat_temperature_negative_control",
        "candidate_summary_source": "proposer",
        "temperature": MATCHED_ENTROPY_TEMPERATURE,
        "use_f_eligibility_demotion": False,
    },
    {
        "arm_id": OFF_ARM,
        "label": "e2wf_hard_topk_eligibility_demotion_off_gapa_baseline",
        "candidate_summary_source": "e2_world_forward",
        "temperature": 1.0,
        "use_f_eligibility_demotion": False,
    },
    {
        "arm_id": PRIMARY_ARM,
        "label": "e2wf_graded_dn_envelope_f_eligibility_demotion_on",
        "candidate_summary_source": "e2_world_forward",
        "temperature": 1.0,
        "use_f_eligibility_demotion": True,
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """GAP-A-ready conversion stack (SP-CEM + SD-056 online + ARC-065 GAP-A
    candidate_summary_source + route-range routing + shortlist-then-modulate +
    shared lateral_pfc/mech295 bias channels), with the MECH-448 f_eligibility
    demotion lever toggled per arm. The MECH-439 conflict-grade levers are OFF on
    every arm (this is the constitutional lever, not the parametric family). With
    use_f_eligibility_demotion=False the shortlist block runs in top_k mode (the
    hard env-conditional F-prefix = the 569i baseline); with it True the shortlist
    block takes the f_demotion branch (the graded DN-share-floor envelope). Both
    keep the within-eligible _modulatory_accum arbitration (F removed from the
    final argmin) -- the ONLY axis is the eligibility-set construction."""
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
        # SHARED E3-side bias channels (consume cand_world_summaries -> set _modulatory_accum)
        use_lateral_pfc_analog=True,
        use_mech295_liking_bridge=True,
        # Other policy-layer regulators + CRF stack OFF (the eligibility lever is the axis)
        use_structured_curiosity=False,
        use_e3_score_diversity=False,
        use_noise_floor=False,
        use_tonic_vigor=False,
        use_dacc=False,
        use_ofc_analog=False,
        use_gated_policy=False,
        use_candidate_rule_field=False,
        # SD-056 substrate trained online on every arm (e2.world_forward divergence)
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=2.0,
        # ARC-065 GAP-A: divergent eligible set (the non-vacuity precondition for the e2wf arms)
        candidate_summary_source=str(arm["candidate_summary_source"]),
        # Shared route-range routing + authority + shortlist-then-modulate conversion constant
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
        # MECH-439 conflict-grade levers OFF on every arm (not the family under test).
        modulatory_shortlist_conflict_graded=False,
        use_gap_scaled_commit_temperature=False,
        # --- MECH-448 (ARC-107): rank-preserving F->eligibility demotion (per arm) ---
        #   ON  -> shortlist_mode resolves to "f_demotion" (graded DN envelope).
        #   OFF -> top_k F-prefix (the 569i hard-shortlist baseline). bit-identical OFF.
        use_f_eligibility_demotion=bool(arm["use_f_eligibility_demotion"]),
        f_eligibility_envelope_floor=F_ELIGIBILITY_ENVELOPE_FLOOR,
        f_eligibility_dn_sigma=F_ELIGIBILITY_DN_SIGMA,
    )
    agent = REEAgent(cfg)
    return agent


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def _first_actions_K(candidates) -> torch.Tensor:
    rows = []
    for traj in candidates:
        rows.append(traj.actions[:, 0, :].detach().reshape(-1))
    return torch.stack(rows, dim=0)


def _entropy_from_counts(counts: Dict[int, int]) -> float:
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
    p1_episodes: int,
    steps_per_episode: int,
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

    pairwise_dists: List[float] = []
    route_ranges: List[float] = []
    route_range_max = 0.0
    authority_active_ticks = 0
    shortlist_sizes: List[float] = []
    shortlist_active_ticks = 0
    shortlist_mode_seen: Optional[str] = None
    # MECH-448 f_eligibility demotion non-vacuity readouts (ARM_ON only fires them).
    demotion_active_ticks = 0
    envelope_sizes: List[float] = []
    excluded_counts: List[float] = []
    winner_neq_f_argmin_ticks = 0
    rank_preserving_active_ticks = 0    # of the demotion-active ticks, how many were rank-preserving
    # C_SAFETY: realised harm exposure over P1.
    harm_p1_abs_sum = 0.0
    harm_p1_ticks = 0

    selected_class_counts: Counter = Counter()
    # Proposer ceiling reference: the candidate-pool first-action classes the
    # SP-CEM proposer OFFERED over P1 (the diversity available BEFORE F-selection).
    pool_class_counts: Counter = Counter()
    n_p1_ticks = 0
    n_contrastive_steps = 0
    error_note: Optional[str] = None

    for ep in range(total_train_eps):
        is_p1 = ep >= p0_episodes
        phase_label = "P1" if is_p1 else "P0"

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

            if is_p1 and candidates and len(candidates) >= 2:
                actions_K = _first_actions_K(candidates).to(agent.device)
                z0 = latent.z_world.detach()
                with torch.no_grad():
                    pdist = float(agent.e2.cand_world_pairwise_dist(z0, actions_K).item())
                if math.isfinite(pdist):
                    pairwise_dists.append(pdist)
                # Proposer ceiling reference: classes the candidate pool offered.
                for cls in actions_K.argmax(dim=-1).reshape(-1).tolist():
                    pool_class_counts[int(cls)] += 1

            if agent.goal_state is not None:
                try:
                    energy = float(body[0, 3].item())
                except Exception:
                    energy = 1.0
                drive_level = max(0.0, 1.0 - energy)
                agent.update_z_goal(benefit_exposure=0.0, drive_level=drive_level)

            action = agent.select_action(
                candidates, ticks, temperature=arm_temperature
            )

            # MECH-448 readouts: read LIVE diagnostics at the select tick.
            if is_p1:
                diag = agent.e3.last_score_diagnostics
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
                # f_eligibility demotion diagnostics (set only when the f_demotion
                # branch fires -> ARM_ON; collect excluded_count etc. only then).
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

            # C_PRIMARY behavioural DV: committed first-action class.
            if is_p1:
                committed_class = int(action[0].argmax().item())
                selected_class_counts[committed_class] += 1
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
            # C_SAFETY: realised harm exposure (P1 only).
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

        if (ep + 1) % 10 == 0 or (ep + 1) == total_train_eps:
            print(
                f"  [train] arm={arm['arm_id']} seed={seed} phase={phase_label} "
                f"ep {ep + 1}/{total_train_eps}",
                flush=True,
            )

    def _mean(xs: List[float], default: float = 0.0) -> float:
        return float(sum(xs) / len(xs)) if xs else default

    selected_action_entropy = _entropy_from_counts(dict(selected_class_counts))
    proposer_pool_entropy = _entropy_from_counts(dict(pool_class_counts))
    demotion_active_frac = (
        float(demotion_active_ticks) / float(n_p1_ticks) if n_p1_ticks > 0 else 0.0
    )
    rank_preserving_frac = (
        float(rank_preserving_active_ticks) / float(demotion_active_ticks)
        if demotion_active_ticks > 0 else (1.0 if not bool(arm["use_f_eligibility_demotion"]) else 0.0)
    )
    harm_per_tick_mean = (
        harm_p1_abs_sum / float(harm_p1_ticks) if harm_p1_ticks > 0 else 0.0
    )

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": int(seed),
        "candidate_summary_source": arm["candidate_summary_source"],
        "temperature": arm_temperature,
        "use_f_eligibility_demotion": bool(arm["use_f_eligibility_demotion"]),
        "n_p1_ticks": int(n_p1_ticks),
        "n_contrastive_steps": int(n_contrastive_steps),
        "error_note": error_note,
        # READINESS (a) / IN-ARM route-range.
        "modulatory_channel_route_range_mean": round(_mean(route_ranges), 6),
        "modulatory_channel_route_range_max": round(route_range_max, 6),
        "modulatory_authority_active_ticks": int(authority_active_ticks),
        # Shortlist diagnostic.
        "modulatory_shortlist_size_mean": round(_mean(shortlist_sizes), 6),
        "modulatory_shortlist_active_ticks": int(shortlist_active_ticks),
        "modulatory_shortlist_mode": shortlist_mode_seen or "",
        # READINESS (b) / e2.world_forward prediction spread.
        "cand_world_pairwise_dist_mean": round(_mean(pairwise_dists), 6),
        # READINESS (c) / MECH-448 non-degeneracy (ARM_ON only).
        "f_eligibility_demotion_active_ticks": int(demotion_active_ticks),
        "f_eligibility_demotion_active_frac": round(demotion_active_frac, 6),
        "f_eligibility_envelope_size_mean": round(_mean(envelope_sizes), 6),
        "f_eligibility_excluded_count_mean": round(_mean(excluded_counts), 6),
        "f_eligibility_winner_neq_f_argmin_ticks": int(winner_neq_f_argmin_ticks),
        # C_RANK_PRESERVING.
        "f_eligibility_rank_preserving_frac": round(rank_preserving_frac, 6),
        # C_PRIMARY behavioural DV.
        "selected_action_class_entropy": round(selected_action_entropy, 6),
        "selected_class_counts": dict(sorted(selected_class_counts.items())),
        "selected_classes_n_unique": int(len(selected_class_counts)),
        # Proposer ceiling reference (informational): the pooled candidate
        # first-action class entropy = the diversity the proposer OFFERED over P1.
        "proposer_pool_class_entropy": round(proposer_pool_entropy, 6),
        "proposer_pool_classes_n_unique": int(len(pool_class_counts)),
        # C_SAFETY.
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
    proposer = _arm_rows(arm_results, PROPOSER_CTRL_ARM)
    noise = _arm_rows(arm_results, MATCHED_NOISE_ARM)
    off = _arm_rows(arm_results, OFF_ARM)
    on = _arm_rows(arm_results, PRIMARY_ARM)

    proposer_by_seed = {r["seed"]: r for r in proposer}
    noise_by_seed = {r["seed"]: r for r in noise}
    off_by_seed = {r["seed"]: r for r in off}

    RDIST = "modulatory_channel_route_range_mean"
    PDIST = "cand_world_pairwise_dist_mean"
    SENT = "selected_action_class_entropy"
    HARM = "harm_per_p1_tick_mean"

    # READINESS (a): IN-ARM route-range gate on the PRIMARY (ON) cell.
    on_route_mean = _mean_key(on, RDIST)
    route_seeds_ok = _n_seeds(on, lambda r: float(r.get(RDIST, 0.0)) > ROUTE_RANGE_FLOOR)
    route_ready = bool(route_seeds_ok >= MIN_SEEDS_FOR_PASS)

    # READINESS (b): e2.world_forward prediction spread on the ON cell.
    on_pdist_mean = _mean_key(on, PDIST)
    c1_seeds_ok = _n_seeds(on, lambda r: float(r.get(PDIST, 0.0)) > C1_PAIRWISE_DIST_FLOOR)
    c1_pass = bool(c1_seeds_ok >= MIN_SEEDS_FOR_PASS)

    # READINESS (c): MECH-448 non-degeneracy on the ON cell. The envelope is ACTIVE
    # on (nearly) all P1 ticks AND it ACTUALLY excludes on the divergent pool
    # (excluded_count > floor). A flat-F pool gives an all-admit envelope
    # (excluded_count == 0) -> vacuous -> substrate_not_ready_requeue.
    def _on_non_degenerate(r: Dict[str, Any]) -> bool:
        return bool(
            float(r.get("f_eligibility_demotion_active_frac", 0.0)) >= DEMOTION_ACTIVE_FRAC_FLOOR
            and float(r.get("f_eligibility_excluded_count_mean", 0.0)) > EXCLUDED_COUNT_FLOOR
        )
    non_degen_seeds_ok = _n_seeds(on, _on_non_degenerate)
    non_degeneracy_ready = bool(non_degen_seeds_ok >= MIN_SEEDS_FOR_PASS)

    readiness_ok = bool(route_ready and c1_pass and non_degeneracy_ready)

    # C_RANK_PRESERVING (load-bearing): ARM_ON eligible set is an F-rank prefix on
    # every active tick (frac == 1.0). F still RANKS within-eligible.
    rank_seeds_ok = _n_seeds(
        on, lambda r: float(r.get("f_eligibility_rank_preserving_frac", 0.0)) >= RANK_PRESERVING_FRAC_REQUIRED
    )
    rank_preserving_pass = bool(rank_seeds_ok >= MIN_SEEDS_FOR_PASS)

    # C_PRIMARY (gate): ARM_ON selected-action entropy STRICTLY ABOVE both the
    # collapsed-proposer baseline (reaches the proposer ceiling) AND the
    # matched-noise control (NOT noise-as-diversity), same seed.
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

    # C_SAFETY (load-bearing): ARM_ON realised harm-per-tick <= ARM_OFF + tolerance,
    # same seed. The envelope keeps clearly-harmful candidates excluded, so demoting
    # F does NOT disinhibit harmful action classes over the hard top-k baseline.
    def _safe_vs_off(r_on: Dict[str, Any]) -> bool:
        r_off = off_by_seed.get(r_on["seed"])
        if r_off is None:
            return False
        return float(r_on.get(HARM, 0.0)) <= float(r_off.get(HARM, 0.0)) + SAFETY_HARM_TOL
    safety_seeds_ok = _n_seeds(on, _safe_vs_off)
    safety_pass = bool(safety_seeds_ok >= MIN_SEEDS_FOR_PASS)

    # VERIFIED-LIFTING matched-noise diagnostic: the noise control DID raise
    # committed entropy over the collapsed proposer baseline (so beating it is a
    # real bar, not a vacuous one -- the 684 readiness lesson).
    def _noise_lifts(rn: Dict[str, Any]) -> bool:
        rp = proposer_by_seed.get(rn["seed"])
        if rp is None:
            return False
        return float(rn.get(SENT, 0.0)) > float(rp.get(SENT, 0.0)) + 1e-6
    noise_lift_seeds = _n_seeds(noise, _noise_lifts)
    matched_noise_verified_lifting = bool(noise_lift_seeds >= MIN_SEEDS_FOR_PASS)

    # ON-vs-OFF demotion-specific delta (SECONDARY, informational): does the graded
    # DN envelope beat the hard top-k F-prefix? (the MECH-448-vs-569i reading)
    def _on_above_off(r_on: Dict[str, Any]) -> bool:
        r_off = off_by_seed.get(r_on["seed"])
        if r_off is None:
            return False
        return float(r_on.get(SENT, 0.0)) > float(r_off.get(SENT, 0.0))
    on_above_off_seeds = _n_seeds(on, _on_above_off)

    # Non-degeneracy: every measured arm produced P1 ticks.
    all_arms = [proposer, noise, off, on]
    non_degenerate = bool(
        all(len(a) > 0 for a in all_arms)
        and all(int(r.get("n_p1_ticks", 0)) > 0 for a in all_arms for r in a)
    )

    # VERDICT resolver: readiness -> rank-preserving -> C_PRIMARY -> C_SAFETY.
    if not readiness_ok:
        label = "substrate_not_ready_requeue"
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
                    "MECH-448 NON-DEGENERACY: the envelope is active on >= "
                    "DEMOTION_ACTIVE_FRAC_FLOOR of ON P1 ticks AND actually EXCLUDES "
                    "(mean excluded_count > floor) on the divergent e2_world_forward "
                    "pool. excluded_count==0 = all-admit (flat-F) = vacuous self-route."
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
                "LOAD-BEARING (order preserved on the numerators): ARM_ON eligible "
                "set is an F-rank PREFIX on every active tick (every eligible cost "
                "<= every excluded cost). Fail = the demotion ALTERED the F-rank "
                "(global F-flatten) -> rank_alteration_not_prefix_diagnose, an "
                "impl/design fault routed to /diagnose-errors, NOT a weakens."
            ),
        },
        "c_primary": {
            "on_seeds_strict_above_both_collapsed_controls": int(primary_seeds_ok),
            "on_selected_entropy_mean": round(on_sel_mean, 6),
            "selected_entropy_floor": C3_SELECTED_ENTROPY_FLOOR,
            "primary_floor_ok": primary_floor_ok,
            "c_primary_pass": primary_pass,
            "note": (
                "GATE (reaches the proposer ceiling + NOT noise-as-diversity): "
                "ARM_ON committed-action entropy STRICTLY ABOVE BOTH "
                "ARM_PROPOSER_CTRL (collapsed channel floor) AND ARM_MATCHED_NOISE "
                "(flat-hot sampling-noise control) on the same seed. Fail = no lift "
                "-> conversion_ceiling_persists_despite_demotion (MECH-449 / V4 "
                "off-ramp; NOT a falsification)."
            ),
        },
        "c_safety": {
            "on_seeds_safe_vs_off": int(safety_seeds_ok),
            "on_harm_per_tick_mean": round(_mean_key(on, HARM), 6),
            "off_harm_per_tick_mean": round(_mean_key(off, HARM), 6),
            "safety_harm_tol": SAFETY_HARM_TOL,
            "c_safety_pass": safety_pass,
            "note": (
                "LOAD-BEARING (no global disinhibition of harmful classes): ARM_ON "
                "realised harm-per-P1-tick <= ARM_OFF + tolerance on the same seed. "
                "The envelope keeps near-zero-merit (clearly-harmful) candidates "
                "EXCLUDED. Fail = the lift admits harmful classes -> "
                "demotion_disinhibits_harmful_classes (the design-note WEAKENED "
                "condition)."
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
                "INFORMATIONAL (NON-GATING): the proposer ceiling = the pooled "
                "candidate first-action class entropy the SP-CEM proposer OFFERED "
                "over P1 (diversity available before F-selection). 'reaches the "
                "proposer ceiling' = ARM_ON committed entropy >= "
                "PROPOSER_CEILING_FRAC of it. C_PRIMARY gates on the strict-above "
                "-both-collapsed-controls + absolute floor (the 569/689 lineage "
                "operationalisation); this readout shows how close the committed "
                "diversity got to the offered ceiling."
            ),
        },
        "matched_noise_verified_lifting": {
            "matched_noise_lift_seeds_over_proposer": int(noise_lift_seeds),
            "matched_noise_verified_lifting": matched_noise_verified_lifting,
            "note": (
                "The matched-noise control MUST be VERIFIED-LIFTING (it raises "
                "committed entropy over the collapsed proposer baseline via sampling "
                "noise) so that ARM_ON beating it is a real bar. Informational: a "
                "non-lifting noise control makes the C_PRIMARY noise comparison "
                "vacuous (the proposer baseline becomes the binding bar)."
            ),
        },
        "demotion_vs_hard_topk_secondary": {
            "on_seeds_above_off": int(on_above_off_seeds),
            "on_selected_entropy_mean": round(on_sel_mean, 6),
            "off_selected_entropy_mean": round(_mean_key(off, SENT), 6),
            "note": (
                "SECONDARY (informational, NON-GATING): does the graded DN envelope "
                "(ARM_ON) beat the HARD env-conditional top-k F-prefix (ARM_OFF, the "
                "569i lever)? The MECH-448-vs-569i reading -- what the env-general "
                "graded envelope adds over the hard count. C_PRIMARY gates on the "
                "proposer ceiling per the design-note s4 acceptance, NOT on this."
            ),
        },
        "selected_action_entropy_per_arm_mean": {
            PROPOSER_CTRL_ARM: round(_mean_key(proposer, SENT), 6),
            MATCHED_NOISE_ARM: round(_mean_key(noise, SENT), 6),
            OFF_ARM: round(_mean_key(off, SENT), 6),
            PRIMARY_ARM: round(on_sel_mean, 6),
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
                "name": "on_modulatory_channel_route_range_supra_floor",
                "kind": "readiness",
                "description": (
                    "ARM_ON IN-ARM RAW cross-candidate RANGE of the modulatory bias "
                    "routed into the E3 selection authority "
                    "(modulatory_channel_route_range, read LIVE at the select tick) "
                    "clears the floor. SAME range statistic the route-range substrate "
                    "gates on. Below floor => routing not wired / e2 under-trained / "
                    "_modulatory_accum absent => substrate_not_ready_requeue."
                ),
                "control": (
                    "ARM_ON: candidate_summary_source=e2_world_forward, route-range "
                    "routing + shortlist-then-modulate + f_eligibility demotion ON"
                ),
                "measured": round(on_route_mean, 6),
                "threshold": ROUTE_RANGE_FLOOR,
                "met": route_ready,
            },
            {
                "name": "on_e2_world_forward_prediction_spread_supra_floor",
                "kind": "readiness",
                "description": (
                    "ARM_ON e2.world_forward(z0, a_i) per-candidate prediction spread "
                    "(cand_world_pairwise_dist) clears the floor -- SD-056 trained the "
                    "action-conditional divergence the eligible set needs. RANGE "
                    "statistic. Below floor => SD-056 under-trained => "
                    "substrate_not_ready_requeue."
                ),
                "control": "ARM_ON: agent.e2.cand_world_pairwise_dist on SP-CEM candidates",
                "measured": round(on_pdist_mean, 6),
                "threshold": C1_PAIRWISE_DIST_FLOOR,
                "met": c1_pass,
            },
            {
                "name": "on_f_eligibility_envelope_excludes_on_divergent_pool",
                "kind": "readiness",
                "description": (
                    "ARM_ON f_eligibility demotion is active on >= "
                    "DEMOTION_ACTIVE_FRAC_FLOOR of P1 ticks AND the envelope ACTUALLY "
                    "EXCLUDES (mean f_eligibility_excluded_count > floor) on the "
                    "divergent e2_world_forward pool -- the MECH-448 NON-DEGENERACY "
                    "signal. An all-admit envelope (excluded_count==0, a flat-F pool) "
                    "is vacuous => substrate_not_ready_requeue. Count statistic "
                    "(seeds meeting both)."
                ),
                "control": (
                    "ARM_ON: use_f_eligibility_demotion=True over the divergent pool; "
                    "f_eligibility_excluded_count read live"
                ),
                "measured": round(_mean_key(on, "f_eligibility_excluded_count_mean"), 6),
                "threshold": EXCLUDED_COUNT_FLOOR,
                "met": non_degeneracy_ready,
            },
        ],
        "criteria": [
            {"name": "C_READINESS_e2_divergent_envelope_excludes", "load_bearing": True, "passed": readiness_ok},
            {"name": "C_RANK_PRESERVING_eligible_set_is_F_rank_prefix", "load_bearing": True, "passed": rank_preserving_pass},
            {"name": "C_PRIMARY_on_selected_entropy_strict_above_both_collapsed_controls",
             "load_bearing": True, "passed": primary_pass},
            {"name": "C_SAFETY_on_harm_not_above_off", "load_bearing": True, "passed": safety_pass},
        ],
        "criteria_non_degenerate": {
            "C_READINESS": non_degenerate,
            "C_RANK_PRESERVING": non_degenerate,
            "C_PRIMARY": non_degenerate,
            "C_SAFETY": non_degenerate,
        },
        "non_degenerate": non_degenerate,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    p0 = DRY_RUN_P0 if dry_run else P0_WARMUP_EPISODES
    p1 = DRY_RUN_P1 if dry_run else P1_MEASUREMENT_EPISODES
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE

    arm_results: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm['arm_id']}", flush=True)
            cell = _run_seed_arm(arm, seed, p0, p1, steps)
            cell["arm_fingerprint"] = compute_arm_fingerprint(
                config_slice={
                    "arm": {
                        k: arm[k]
                        for k in (
                            "arm_id", "candidate_summary_source", "temperature",
                            "use_f_eligibility_demotion",
                        )
                    },
                    "env_kwargs": ENV_KWARGS,
                    "sd056_weight": SD056_WEIGHT,
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
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": {"MECH-448": evidence_direction},
        "non_degenerate": summary.get("non_degenerate", True),
        "evidence_direction_note": (
            "MECH-448 (ARC-107) rank-preserving F->eligibility demotion falsifier -- "
            "the 689a-successor (the constitutional LEAD lever, NOT a re-queue of the "
            "conflict-grade near-tie parametric family adjudicated exhausted by the "
            "user-ratified failure_autopsy_V3-EXQ-689a_2026-06-20 Step-8). F decides "
            "ELIGIBILITY only via a graded, rank-preserving divisive-normalisation "
            "envelope (absolute share floor); the existing within-eligible "
            "_modulatory_accum arbitration picks the committed action with F REMOVED "
            "from the final argmin. ARM_OFF (hard env-conditional top-k F-prefix = the "
            "569i lever) vs ARM_ON (graded DN envelope) on matched seeds, + "
            "ARM_PROPOSER_CTRL (collapsed-channel floor) + ARM_MATCHED_NOISE "
            "(VERIFIED-LIFTING flat-hot sampling-noise control). PASS "
            "(label=demotion_converts_committed_diversity) = READINESS (IN-ARM "
            "route-range + e2-divergence + NON-DEGENERACY: the envelope ACTUALLY "
            "excludes [excluded_count>0] on the divergent e2_world_forward pool) AND "
            "C_RANK_PRESERVING (eligible set is an F-rank prefix -- order preserved on "
            "the numerators) AND C_PRIMARY (committed entropy strict-above BOTH "
            "collapsed controls = reaches the proposer ceiling + NOT noise-as-diversity) "
            "AND C_SAFETY (no harmful-class disinhibition vs the hard top-k baseline). "
            "Route/e2-div below floor OR all-admit envelope (excluded_count==0) "
            "self-routes substrate_not_ready_requeue (non_contributory, NEVER a "
            "weakens). Readiness met + rank fraction < 1.0 => "
            "rank_alteration_not_prefix_diagnose (impl fault -> /diagnose-errors). "
            "Readiness + rank met, no lift => conversion_ceiling_persists_despite_demotion "
            "(MECH-449 Go/No-Go follow-on [double-gated] / V4 off-ramp; NOT a "
            "falsification). Only a C_SAFETY fail (the lift admits harmful classes / "
            "global F-flatten) WEAKENS. claim_ids=[MECH-448] only; "
            "ARC-107/MECH-447/MECH-449/MECH-439/Q-078 untouched; MECH-448 stays candidate."
        ),
        "interpretation": {
            "label": summary["label"],
            "preconditions": summary["preconditions"],
            "criteria": summary["criteria"],
            "criteria_non_degenerate": summary["criteria_non_degenerate"],
            "routing": {
                "demotion_converts_committed_diversity": "PASS -> MECH-448 toward supports; ARC-107 gains its first validated lever; F->eligibility demotion moves committed behaviour, rank-preserving, no harmful disinhibition",
                "substrate_not_ready_requeue": "routing/SD-056 under-trained OR the pool is not divergent (all-admit envelope, excluded_count==0); re-queue; do NOT weaken MECH-448",
                "rank_alteration_not_prefix_diagnose": "the eligible set is NOT an F-rank prefix (the envelope altered the F-rank) -> implementation/design fault -> /diagnose-errors; NOT a weakens",
                "conversion_ceiling_persists_despite_demotion": "OFF-RAMP -> the MECH-449 Go/No-Go eligibility constitution (double-gated on this falsifier showing the demotion lever alone is insufficient) / the V4 directions; NOT a falsification of MECH-448",
                "demotion_disinhibits_harmful_classes": "WEAKENED -- the committed-entropy lift came from globally flattening F / admitting harmful classes (C_SAFETY fail), the design-note weaken condition",
            },
        },
        "dry_run": bool(dry_run),
        "config": {
            "seeds": seeds,
            "p0_warmup_episodes": p0,
            "p1_measurement_episodes": p1,
            "steps_per_episode": steps,
            "env_kwargs": ENV_KWARGS,
            "arms": [
                {k: a[k] for k in (
                    "arm_id", "label", "candidate_summary_source", "temperature",
                    "use_f_eligibility_demotion",
                )}
                for a in ARMS
            ],
            "sd056_weight": SD056_WEIGHT,
            "matched_entropy_temperature": MATCHED_ENTROPY_TEMPERATURE,
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
            },
        },
        "acceptance_criteria": {
            "readiness_route_range_ready": summary["readiness"]["route_ready"],
            "readiness_e2_divergent_ready": summary["readiness"]["c1_pass"],
            "readiness_envelope_non_degenerate": summary["readiness"]["non_degeneracy"]["non_degeneracy_ready"],
            "readiness_substrate_ready": summary["readiness"]["readiness_ok"],
            "matched_noise_verified_lifting": summary["matched_noise_verified_lifting"]["matched_noise_verified_lifting"],
            "C_RANK_PRESERVING_eligible_set_is_F_rank_prefix": summary["c_rank_preserving"]["c_rank_preserving_pass"],
            "C_PRIMARY_on_strict_above_both_collapsed_controls": summary["c_primary"]["c_primary_pass"],
            "C_SAFETY_on_harm_not_above_off": summary["c_safety"]["c_safety_pass"],
            "overall_pass": summary["overall_pass"],
        },
        "summary": summary,
        "arm_results": arm_results,
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"

    if not dry_run:
        out_path = write_flat_manifest(
            manifest,
            out_dir,
            dry_run=False,
            config=manifest.get("config"),
            seeds=SEEDS,
            script_path=Path(__file__),
        )
        print(f"Manifest written: {out_path}", flush=True)
    else:
        out_path = Path("/dev/null")
        print("Dry run -- manifest not written.", flush=True)

    print(f"Outcome: {outcome} (label={summary['label']}, evidence_direction={evidence_direction})", flush=True)
    for k, v in manifest["acceptance_criteria"].items():
        print(f"  {k}: {v}", flush=True)

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-689d MECH-448 (ARC-107) rank-preserving F->eligibility demotion falsifier (689a-successor)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Short smoke run.")
    args = parser.parse_args()

    result = run_experiment(dry_run=args.dry_run)

    if args.dry_run:
        sys.exit(0)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    _outcome = _outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL"
    emit_outcome(
        outcome=_outcome,
        manifest_path=str(result.get("manifest_path", Path("/dev/null"))),
    )
