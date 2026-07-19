#!/opt/local/bin/python3
"""V3-EXQ-590c -- MECH-314 per-candidate novelty Goldilocks under top-k shortlist.

Supersedes V3-EXQ-590b (FAIL/non_contributory 2026-06-11). The 590b routing
table records "re-queue as V3-EXQ-590c at higher P0 budget..."; this is that
re-issue, but with the SWEPT KNOB and the SELECTION-AUTHORITY STACK both
re-posed per the F-dominance campaign. claim_ids=[MECH-314, DEV-NEED-003],
experiment_purpose=evidence. FIRST behavioural-evidence run for the MECH-314
parent novelty channel reaching COMMITTED action under the 569i top-k path
(the 590/590a lineage tested MECH-111 broadcast novelty; 590b tested the
MECH-314a authority-gain lift -- no inherited claim_ids; re-evaluated from
scratch).

=== WHY THE SWEPT KNOB IS curiosity_novelty_weight, AND WHY THAT IS NOT THE
    590a/590b DEGENERATE NULL ===

590b swept modulatory_authority_gain BECAUSE, under the ADDITIVE
modulatory-selection-authority path, the authority RESCALES the combined
modulatory contribution so its range == gain * raw_score_range REGARDLESS of
curiosity_novelty_weight -- the weight is washed out and sweeping it reproduces
the 590a byte-identical-arms null (590b docstring, "WHY THE SWEPT KNOB IS
modulatory_authority_gain").

THE TOP-K SHORTLIST CONVERSION PATH BREAKS THAT WASH-OUT. With
use_modulatory_shortlist_then_modulate=True + modulatory_shortlist_mode="top_k"
+ modulatory_shortlist_k=3 (the 569i conversion lever; CONVERSION amend
2026-06-15, TOP-K amend 2026-06-16), the primary score F filters the candidate
pool to the k=3 F-best, and the committed action is the ARGMIN of the modulatory
accumulator (_modulatory_accum) WITHIN that set -- NOT an authority-rescaled add.
The curiosity bias is the SOLE contributor to _modulatory_accum here, and it is
clamped to +/-curiosity_bias_scale BEFORE composition. So curiosity_novelty_weight
is genuinely load-bearing on the within-shortlist argmin:

  weight too LOW  -> per-candidate curiosity range is tiny -> argmin within the
                     top-3 is near-arbitrary / F-rank-tracking -> low committed
                     diversity.
  weight JUST RIGHT -> the per-candidate novelty range cleanly separates the
                     top-3 -> the committed class rotates with novelty ->
                     committed-action-class entropy lifts.
  weight too HIGH -> the curiosity bias saturates the +/-bias_scale clamp for
                     ~all candidates -> flat again -> committed diversity falls
                     back toward the F-only baseline (the clamp-saturation arm of
                     the inverted-U).

That is a genuine Goldilocks on curiosity_novelty_weight -- the 590a/590b
degeneracy does not apply because the shortlist argmin reads the curiosity
channel's pre-rescale per-candidate magnitude, not an authority-renormalized
range. The w=0.0 control is F-only committed selection (curiosity bias zero ->
within-top-k argmin tracks F rank).

=== DV: COMMITTED-ACTION-CLASS ENTROPY (NOT h_pos exploration lift) ===

The 590/590b DV was env pos_entropy (H_pos, position-visitation exploration).
590c routes on the F-dominance-campaign DV: committed-action-class entropy --
the Shannon entropy over the agent's emitted (committed) first-action classes,
pooled over the P1 measurement window. This is the statistic MECH-439 / 569i /
ARC-065 GAP-A measure (does the modulatory channel reach COMMITTED action under
F-dominance), and the one the non-vacuity guard routes on. H_pos is retained as
an informational secondary only.

=== SUBSTRATE (identical across arms except curiosity_novelty_weight) ===

648a-validated config: SP-CEM main path (action-divergent pool) + V_s stack +
SD-056 online contrastive (e2.world_forward action-conditional divergence;
rollout output-norm clamp ON per the 643a numerical-stability lesson) + MECH-314
visitation-buffer novelty (curiosity_novelty_source="visitation",
first-action-onehot auto-augmentation) + curiosity_candidate_source=
"e2_world_forward". MECH-314 novelty is the SOLE modulatory channel (dacc /
lateral_pfc / ofc / mech295 / tonic_vigor / noise_floor / e3_score_diversity all
OFF). The GAP-A / modulatory-bias-selection-authority stack is held CONSTANT
across arms: use_modulatory_selection_authority=True,
modulatory_authority_gain=1.0, use_modulatory_shortlist_then_modulate=True,
modulatory_shortlist_mode="top_k", modulatory_shortlist_k=3.

Harm-free env (num_hazards=0): the residue field stays empty, which is why the
visitation novelty source is used; SP-CEM + resources give action-divergent
candidates for SD-056 to keep e2.world_forward divergent.

=== ARMS (4 curiosity_novelty_weights x 3 seeds) ===

  ARM_W000  curiosity_novelty_weight=0.00  (curiosity bias zero; F-only committed
                                            selection within the top-3; control)
  ARM_W005  curiosity_novelty_weight=0.05  (the landed default)
  ARM_W025  curiosity_novelty_weight=0.25  (the 648a/590b fixed value)
  ARM_W100  curiosity_novelty_weight=1.00  (clamp-saturation regime)

=== MANDATORY NON-VACUITY SELF-ROUTE GATE (the 569i conversion is ENV-CONDITIONAL;
    V3-EXQ-625e autopsy 2026-06-20) ===

A 590c re-issue could re-derive the MECH-439 F-dominance conversion ceiling on
the infant env (committed-selection diversity may NOT survive). If it does not
survive, the run self-routes substrate_not_ready_requeue (non_contributory),
NEVER a MECH-314 / DEV-NEED-003 weakens. Three load-bearing legs + a finite
guard:

  Leg A (e2-divergence non-vacuity, the curiosity_candidate_source precondition):
    cand_world_pairwise_dist_mean at the highest-weight arm > CAND_DIST_FLOOR on
    >= MIN_SEEDS. The e2.world_forward predictions the novelty consumes must be
    action-divergent (the ARC-065 GAP-A / 648a precondition; an under-trained e2
    collapses the channel).
  Leg B (channel live; the 643a guard): the highest-weight arm's
    curiosity_bias_range_mean (pre-clamp/pre-rescale cross-candidate RANGE) >
    BIAS_RANGE_FLOOR on >= MIN_SEEDS. Scaling zero is still zero.
  Leg C (the knob moves the routed readout; SAME statistic the Goldilocks routes
    on): RANGE across the curiosity_novelty_weight arms of the healthy-seed-mean
    committed_class_entropy >= ENTROPY_RANGE_FLOOR. If committed diversity is flat
    across every weight, the channel does not reach committed action under
    F-dominance (the conversion ceiling persists) -> substrate_not_ready_requeue,
    NOT a verdict.
  Leg D (finite guard, 643a): max cand_world_pairwise_dist finite and < ceil
    (SD-056 online numerical stability).

=== ACCEPTANCE (pre-registered) ===

PASS (supports MECH-314 + DEV-NEED-003) = readiness met AND a Goldilocks weight
  is identified: the best non-zero-weight arm's mean committed_class_entropy
  exceeds the w=0 control by >= ENTROPY_LIFT_MARGIN on >= MIN_SEEDS of 3 seeds
  (paired per seed). The per-candidate MECH-314 novelty channel, given top-k
  selection access, lifts COMMITTED action diversity over F-only selection.
  nonmonotone=True (interior peak > both neighbours) reports the clamp-saturation
  ceiling.
FAIL (does_not_support) = readiness met (the knob moves committed_class_entropy
  and the channel is live) BUT no non-zero weight beats the control by the margin
  on a majority of seeds -- curiosity-under-top-k does not convert to committed
  diversity here.
substrate_not_ready_requeue (non_contributory) = any readiness leg below floor /
  non-finite. Most importantly the conversion-ceiling case (Leg C flat): the
  diversity does NOT survive F-dominance -> re-queue, do NOT weaken.

=== INTERPRETATION GRID ===

| Outcome                                       | label                                   | next action |
|-----------------------------------------------|-----------------------------------------|-------------|
| any readiness leg below floor / non-finite OR Leg-C flat | substrate_not_ready_requeue   | conversion ceiling persists / e2 collapsed -- re-queue as 590d on a GAP-A-readier substrate; do NOT weaken MECH-314 / DEV-NEED-003 |
| readiness OK + Goldilocks lift found          | mech314_committed_diversity_goldilocks_identified | PASS (supports MECH-314 + DEV-NEED-003); adopt best weight for the curiosity channel |
| readiness OK + no lift over control           | mech314_no_committed_diversity_benefit  | FAIL (does_not_support); /failure-autopsy the conversion reading |

architecture_epoch: "ree_hybrid_guardrails_v1"

Smoke:
  /opt/local/bin/python3 experiments/v3_exq_590c_mech314_novelty_goldilocks.py --dry-run
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


EXPERIMENT_TYPE = "v3_exq_590c_mech314_novelty_goldilocks"
QUEUE_ID = "V3-EXQ-590c"
CLAIM_IDS: List[str] = ["MECH-314", "DEV-NEED-003"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES_QUEUE_ID = "V3-EXQ-590b"
SUPERSEDES_RUN = "v3_exq_590b_mech314a_novelty_goldilocks_20260611T211806Z_v3"

SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 60            # SD-056 contrastive warmup (matches V3-EXQ-648a / 590b)
P1_MEASUREMENT_EPISODES = 30       # committed-diversity measurement window
STEPS_PER_EPISODE = 200
MEASURE_AFTER_TICK = 20            # within-episode warmup before reading bias range

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_STEPS = 30
DRY_RUN_MEASURE_AFTER_TICK = 2

# Swept knob: MECH-314 per-candidate novelty weight (the within-top-k argmin lever).
CURIOSITY_WEIGHTS: List[float] = [0.0, 0.05, 0.25, 1.0]

# Selection-authority stack held CONSTANT across arms (569i top-k conversion path).
MODULATORY_AUTHORITY_GAIN = 1.0
SHORTLIST_K = 3

# Pre-registered thresholds.
BIAS_RANGE_FLOOR = 1.0e-4     # readiness leg B: curiosity bias non-zero per-candidate RANGE
CAND_DIST_FLOOR = 0.02        # readiness leg A: e2.world_forward action-divergence non-vacuity
ENTROPY_RANGE_FLOOR = 0.05    # readiness leg C: committed_class_entropy RANGE across weight arms
MAGNITUDE_CEIL = 1.0e6        # readiness leg D: rolled-out z_world finite guard (643a)
ENTROPY_LIFT_MARGIN = 0.05    # PASS: best non-zero weight committed_class_entropy - control, per seed
MIN_SEEDS_FOR_PASS = 2        # of 3

# SD-056 online contrastive training (mirror V3-EXQ-648a / 590b harness).
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
E2_TRAIN_EVERY_K_TICKS = 1
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0

# Curiosity clamp / visitation buffer held FIXED (648a values); the swept knob is the
# novelty WEIGHT (set per-arm in _make_agent), NOT the clamp.
CURIOSITY_BIAS_SCALE = 0.5
VISITATION_BUFFER_LEN = 256

# HARM-FREE env (num_hazards=0): residue field stays empty (visitation source is the
# point); SP-CEM + resources still give action-divergent candidates for SD-056.
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
)


def _arm_id(weight: float) -> str:
    return f"ARM_W{int(round(weight * 100)):03d}"


ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": _arm_id(w),
        "curiosity_novelty_weight": w,
        "is_control": (w == 0.0),
    }
    for w in CURIOSITY_WEIGHTS
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """648a-validated substrate; MECH-314 curiosity is the SOLE modulatory channel.

    The GAP-A / modulatory-bias-selection-authority stack is held CONSTANT across
    arms (use_modulatory_selection_authority=True, modulatory_authority_gain=1.0,
    use_modulatory_shortlist_then_modulate=True, top-k k=3 from 569i). The swept
    knob is curiosity_novelty_weight -- under the top-k shortlist, the committed
    action is the within-top-3 argmin of the curiosity-fed _modulatory_accum, so
    the (pre-clamp) novelty weight is load-bearing (NOT washed out by the
    authority rescale, which the shortlist takes precedence over).

    SD-056 is trained online on every arm (the e2.world_forward divergence the
    curiosity novelty consumes); the rollout output-norm clamp is ON per the 643a
    numerical-stability lesson (online SD-056 can explode rolled-out z_world).
    """
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
        # ARC-065 SP-CEM (Layer A) -- main-path default (action-divergent pool)
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
        # THE SWEPT KNOB.
        curiosity_novelty_weight=float(arm["curiosity_novelty_weight"]),
        curiosity_novelty_source="visitation",
        curiosity_visitation_buffer_len=VISITATION_BUFFER_LEN,
        curiosity_use_first_action_onehot=True,
        curiosity_first_action_augmentation_policy="auto",
        # V3-EXQ-648a fix: novelty consumes the SD-056-divergent e2.world_forward(z0,a_i).
        curiosity_candidate_source="e2_world_forward",
        # GAP-A / modulatory-bias-selection-authority stack -- CONSTANT across arms.
        use_modulatory_selection_authority=True,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
        # 569i conversion lever: F filters to the top-k, the curiosity channel
        # arbitrates the within-set committed argmin (NOT an authority rescale).
        use_modulatory_shortlist_then_modulate=True,
        modulatory_shortlist_mode="top_k",
        modulatory_shortlist_k=SHORTLIST_K,
    )
    agent = REEAgent(cfg)
    # Per-channel score-bias decomposition so select_action records the
    # per-candidate curiosity bias range (the readiness leg-B statistic).
    agent.e3.e3_score_decomp_enabled = True
    return agent


# ---------------------------------------------------------------------------
# SD-056 online contrastive helpers (from V3-EXQ-648a / 604a / 569d / 590b)
# ---------------------------------------------------------------------------

def _first_actions_K(candidates) -> torch.Tensor:
    rows = []
    for traj in candidates:
        rows.append(traj.actions[:, 0, :].detach().reshape(-1))
    return torch.stack(rows, dim=0)


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


def _action_class_entropy(counts: Counter) -> float:
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
    # Full RNG reset at cell entry -> arm_fingerprint is order-independent.
    reset_all_rng(seed)

    env = _make_env(seed)
    agent = _make_agent(env, arm)
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)

    transition_buffer: Deque[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)

    total_train_eps = p0_episodes + p1_episodes

    # P1 accumulators.
    ep_h_pos: List[float] = []                 # per-P1-episode position entropy (secondary)
    action_class_counts: Counter = Counter()   # PRIMARY DV: pooled committed-action classes
    curiosity_range_vals: List[float] = []
    pairwise_dists: List[float] = []
    pairwise_dist_max_seen = 0.0
    n_p0_ticks = 0
    n_p1_ticks = 0
    n_p1_ticks_past_window = 0
    n_contrastive_steps = 0
    n_h_pos_fallback = 0                        # episodes where pos_entropy was absent
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
        last_info: Dict[str, Any] = {}
        ep_action_counts: Counter = Counter()

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

            action = agent.select_action(candidates, ticks)
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

            # Per-candidate curiosity bias RANGE (readiness leg B) + committed-action
            # class (PRIMARY DV) captured AFTER select_action (its single curiosity
            # call set the decomposition).
            if past_window:
                decomp = getattr(agent, "_last_score_bias_decomp", {}) or {}
                crange = float(decomp.get("curiosity_bias_range_mean", 0.0))
                if math.isfinite(crange):
                    curiosity_range_vals.append(crange)
                ep_action_counts.update([int(action.argmax().item())])
                n_p1_ticks_past_window += 1

            if is_p1:
                n_p1_ticks += 1
            else:
                n_p0_ticks += 1

            if tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0:
                loss_val = _e2_contrastive_step(
                    agent=agent, buffer=transition_buffer,
                    optimiser=e2_opt, rng=sample_rng,
                )
                if loss_val is not None and math.isfinite(loss_val) and is_p1:
                    n_contrastive_steps += 1

            if (
                torch.isfinite(latent.z_world).all()
                and torch.isfinite(action).all()
            ):
                pending_capture = (
                    latent.z_world.detach().reshape(-1).clone(),
                    action.detach().reshape(-1).clone(),
                )

            _, harm_signal, done, info, next_obs_dict = env.step(action)
            last_info = info if isinstance(info, dict) else {}

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

        # End-of-episode readouts (P1 only).
        if is_p1:
            # PRIMARY: pool the committed-action classes over the measurement window.
            action_class_counts.update(ep_action_counts)
            # Secondary (informational only): env position-visitation entropy.
            h_pos_raw = last_info.get("pos_entropy", None)
            if h_pos_raw is None:
                h_pos_raw = obs_dict.get("pos_entropy", None) if isinstance(obs_dict, dict) else None
            if h_pos_raw is None or not math.isfinite(float(h_pos_raw)) or float(h_pos_raw) < 0.0:
                h_pos_val = _action_class_entropy(ep_action_counts)
                n_h_pos_fallback += 1
            else:
                h_pos_val = float(h_pos_raw)
            ep_h_pos.append(h_pos_val)

        if (ep + 1) % 10 == 0 or (ep + 1) == total_train_eps:
            print(
                f"  [train] arm={arm['arm_id']} seed={seed} phase={phase_label} "
                f"ep {ep + 1}/{total_train_eps}",
                flush=True,
            )

    def _mean(xs: List[float], default: float = 0.0) -> float:
        return float(sum(xs) / len(xs)) if xs else default

    return {
        "arm_id": arm["arm_id"],
        "curiosity_novelty_weight": float(arm["curiosity_novelty_weight"]),
        "is_control": bool(arm["is_control"]),
        "seed": int(seed),
        "n_p0_ticks": int(n_p0_ticks),
        "n_p1_ticks": int(n_p1_ticks),
        "n_p1_ticks_past_window": int(n_p1_ticks_past_window),
        "n_p1_episodes": int(len(ep_h_pos)),
        "n_contrastive_steps": int(n_contrastive_steps),
        "n_h_pos_fallback_episodes": int(n_h_pos_fallback),
        "error_note": error_note,
        # PRIMARY DV: committed-action-class entropy (pooled over the P1 window).
        "committed_class_entropy": round(_action_class_entropy(action_class_counts), 6),
        "n_committed_classes": int(len(action_class_counts)),
        # Secondary (informational only): exploration position entropy.
        "h_pos_mean": round(_mean(ep_h_pos), 6),
        "ep_h_pos_last5": [round(x, 4) for x in ep_h_pos[-5:]],
        # Readiness leg B: per-candidate curiosity bias range (pre-clamp).
        "curiosity_bias_range_mean": round(_mean(curiosity_range_vals), 8),
        # Readiness leg A/D input.
        "cand_world_pairwise_dist_mean": round(_mean(pairwise_dists), 6),
        "cand_world_pairwise_dist_max": round(pairwise_dist_max_seen, 6),
    }


# ---------------------------------------------------------------------------
# Cross-arm evaluation
# ---------------------------------------------------------------------------

def _arm_rows(rows: List[Dict[str, Any]], arm_id: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("arm_id") == arm_id]


def _n_seeds_satisfying(rows: List[Dict[str, Any]], predicate) -> int:
    return sum(1 for r in rows if predicate(r))


def _mean_key(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [float(r.get(key, 0.0)) for r in rows]
    return float(sum(vals) / len(vals)) if vals else 0.0

def _kth_best(values: List[float], k: int, threshold: float,
              *, upper: bool = False, strict: bool = True) -> float:
    """The k-th BEST per-seed value -- k-th LARGEST for a floor, k-th SMALLEST
    for a ceiling.

    Reported in a precondition's `measured` INSTEAD of a mean/min/max so that a
    COUNT-OF-SEEDS predicate ("holds on >= k of n seeds") is exactly
    reproducible from the entry's own (measured, threshold) pair:
    `kth_largest > floor` IS "at least k seeds cleared the floor". A mean is not
    a function of that count at all, `min` is strictly harsher than the shipped
    predicate (it demands all n) and `max` strictly looser (it demands only 1),
    so each makes the indexer's AUTHORITATIVE recompute in
    build_experiment_indexes._precondition_unmet disagree with the author's
    `met` -- the 2026-06-07 V3-EXQ-648a/649 mis-scoring shape.

    Fewer than k values (the dry run ships 1 seed against k=2) means the
    predicate CANNOT hold, so return the value that recomputes as UNMET under
    this bound's own strictness rather than a real observation.
    """
    vals = sorted((float(v) for v in values), reverse=not upper)
    if k <= 0 or len(vals) < k:
        t = float(threshold)
        if strict:
            return t  # a strict bound reads measured == threshold as UNMET
        eps = abs(t) * 1e-9 + 1e-9
        return t + eps if upper else t - eps
    return vals[k - 1]


def _evaluate(arm_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_id = {a["arm_id"]: _arm_rows(arm_results, a["arm_id"]) for a in ARMS}
    control_arm = next(a for a in ARMS if a["is_control"])
    nonzero_arms = [a for a in ARMS if not a["is_control"]]
    control_rows = by_id[control_arm["arm_id"]]
    highest_weight_arm = max(ARMS, key=lambda a: a["curiosity_novelty_weight"])
    highest_weight_rows = by_id[highest_weight_arm["arm_id"]]

    # --- READINESS leg A: e2.world_forward action-divergence non-vacuity ---
    cand_dist_seeds_ok = _n_seeds_satisfying(
        highest_weight_rows,
        lambda r: float(r.get("cand_world_pairwise_dist_mean", 0.0)) > CAND_DIST_FLOOR,
    )
    highest_weight_cand_dist_mean = _mean_key(
        highest_weight_rows, "cand_world_pairwise_dist_mean"
    )
    # Statistic leg A's PRECONDITION reports (the mean stays as a diagnostic):
    # the MIN_SEEDS_FOR_PASS-th largest per-seed value, so `measured > floor`
    # reproduces `cand_dist_seeds_ok >= MIN_SEEDS_FOR_PASS` exactly.
    highest_weight_cand_dist_kth = _kth_best(
        [float(r.get("cand_world_pairwise_dist_mean", 0.0)) for r in highest_weight_rows],
        MIN_SEEDS_FOR_PASS, CAND_DIST_FLOOR,
    )

    # --- READINESS leg B: channel live (curiosity bias non-zero per-candidate range) ---
    bias_range_seeds_ok = _n_seeds_satisfying(
        highest_weight_rows,
        lambda r: float(r.get("curiosity_bias_range_mean", 0.0)) > BIAS_RANGE_FLOOR,
    )
    highest_weight_bias_range_mean = _mean_key(
        highest_weight_rows, "curiosity_bias_range_mean"
    )
    # Same k-th-best treatment for leg B's precondition.
    highest_weight_bias_range_kth = _kth_best(
        [float(r.get("curiosity_bias_range_mean", 0.0)) for r in highest_weight_rows],
        MIN_SEEDS_FOR_PASS, BIAS_RANGE_FLOOR,
    )

    # --- READINESS leg C (SAME statistic the Goldilocks routes on): committed_class_entropy
    # RANGE across the weight arms (healthy-seed-mean per arm) >= floor ---
    per_arm_entropy_mean = {
        a["arm_id"]: round(_mean_key(by_id[a["arm_id"]], "committed_class_entropy"), 6)
        for a in ARMS
    }
    entropy_arm_means = list(per_arm_entropy_mean.values())
    entropy_range_across_arms = (
        float(max(entropy_arm_means) - min(entropy_arm_means)) if entropy_arm_means else 0.0
    )
    entropy_range_ok = bool(entropy_range_across_arms >= ENTROPY_RANGE_FLOOR)
    # Statistic leg C's PRECONDITION reports. NaN ONLY (an infinite range still
    # satisfies `>= floor` in both the shipped predicate and the recompute, so it
    # needs no special case): a NaN range fails EVERY comparison, so
    # `entropy_range_ok` is False while the indexer's recompute
    # (`unmet == measured < threshold`) would read NaN as MET. Report a value just
    # below the floor instead -- the sentinel that recomputes as UNMET under this
    # inclusive `>=` bound.
    entropy_range_measured = (
        ENTROPY_RANGE_FLOOR - (abs(ENTROPY_RANGE_FLOOR) * 1e-9 + 1e-9)
        if math.isnan(entropy_range_across_arms) else entropy_range_across_arms
    )

    # --- READINESS leg D: finite / explosion guard ---
    max_pairwise = max(
        [float(r.get("cand_world_pairwise_dist_max", 0.0)) for r in arm_results] or [0.0]
    )
    magnitude_ok = bool(math.isfinite(max_pairwise) and max_pairwise < MAGNITUDE_CEIL)
    # Statistic the ceiling PRECONDITION reports. A NaN spread fails EVERY
    # comparison, so reporting it raw would make the indexer's recompute read the
    # entry as MET while the isfinite leg says otherwise; the ceiling itself is
    # the sentinel that recomputes as UNMET under the strict `<` bound.
    magnitude_measured = max_pairwise if math.isfinite(max_pairwise) else MAGNITUDE_CEIL

    readiness_ok = bool(
        cand_dist_seeds_ok >= MIN_SEEDS_FOR_PASS
        and bias_range_seeds_ok >= MIN_SEEDS_FOR_PASS
        and entropy_range_ok
        and magnitude_ok
    )

    # --- Goldilocks: best non-zero weight arm by mean committed_class_entropy;
    # per-seed lift over control ---
    control_by_seed = {
        int(r["seed"]): float(r.get("committed_class_entropy", 0.0)) for r in control_rows
    }

    def _seeds_lift_over_control(rows: List[Dict[str, Any]]) -> int:
        n = 0
        for r in rows:
            s = int(r["seed"])
            ctrl = control_by_seed.get(s, None)
            if ctrl is None:
                continue
            if float(r.get("committed_class_entropy", 0.0)) - ctrl >= ENTROPY_LIFT_MARGIN:
                n += 1
        return n

    nonzero_arm_lift = {
        a["arm_id"]: {
            "curiosity_novelty_weight": a["curiosity_novelty_weight"],
            "committed_class_entropy": round(
                _mean_key(by_id[a["arm_id"]], "committed_class_entropy"), 6
            ),
            "seeds_lift_over_control": int(_seeds_lift_over_control(by_id[a["arm_id"]])),
        }
        for a in nonzero_arms
    }
    # Goldilocks = highest mean-committed_class_entropy non-zero arm.
    best_arm_id = max(
        nonzero_arm_lift.keys(),
        key=lambda aid: nonzero_arm_lift[aid]["committed_class_entropy"],
    )
    best_weight = float(nonzero_arm_lift[best_arm_id]["curiosity_novelty_weight"])
    best_seeds_lift = int(nonzero_arm_lift[best_arm_id]["seeds_lift_over_control"])
    goldilocks_lift_ok = bool(best_seeds_lift >= MIN_SEEDS_FOR_PASS)

    # Inverted-U / clamp-saturation detection over the FULL weight axis (control + non-zero).
    ordered_ids = [a["arm_id"] for a in ARMS]
    ordered_ent = [per_arm_entropy_mean[aid] for aid in ordered_ids]
    peak_idx = ordered_ent.index(max(ordered_ent)) if ordered_ent else 0
    nonmonotone = bool(0 < peak_idx < len(ordered_ent) - 1)

    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        overall_pass = False
    elif goldilocks_lift_ok:
        label = "mech314_committed_diversity_goldilocks_identified"
        overall_pass = True
    else:
        label = "mech314_no_committed_diversity_benefit"
        overall_pass = False

    return {
        "readiness": {
            "legA_cand_dist_floor": CAND_DIST_FLOOR,
            "highest_weight_arm": highest_weight_arm["arm_id"],
            "highest_weight_cand_dist_mean": round(highest_weight_cand_dist_mean, 6),
            "legA_seeds_above_floor": int(cand_dist_seeds_ok),
            "legB_bias_range_floor": BIAS_RANGE_FLOOR,
            "highest_weight_bias_range_mean": round(highest_weight_bias_range_mean, 8),
            "legB_seeds_above_floor": int(bias_range_seeds_ok),
            "legC_entropy_range_floor": ENTROPY_RANGE_FLOOR,
            "committed_class_entropy_range_across_arms": round(entropy_range_across_arms, 6),
            "legC_entropy_range_ok": entropy_range_ok,
            "legD_magnitude_ceil": MAGNITUDE_CEIL,
            "max_pairwise_dist_observed": round(max_pairwise, 6),
            "legD_magnitude_ok": magnitude_ok,
            "min_seeds_required": MIN_SEEDS_FOR_PASS,
            "readiness_ok": readiness_ok,
        },
        "goldilocks": {
            "control_arm": control_arm["arm_id"],
            "control_committed_class_entropy": round(
                _mean_key(control_rows, "committed_class_entropy"), 6
            ),
            "best_arm": best_arm_id,
            "best_weight": best_weight,
            "best_arm_committed_class_entropy": round(
                nonzero_arm_lift[best_arm_id]["committed_class_entropy"], 6
            ),
            "best_arm_seeds_lift_over_control": best_seeds_lift,
            "entropy_lift_margin": ENTROPY_LIFT_MARGIN,
            "goldilocks_lift_ok": goldilocks_lift_ok,
            "nonmonotone_inverted_u": nonmonotone,
            "per_arm_committed_class_entropy_mean": per_arm_entropy_mean,
            "per_nonzero_arm_lift": nonzero_arm_lift,
        },
        "h_pos_per_arm_mean": {
            aid: round(_mean_key(rows, "h_pos_mean"), 6)
            for aid, rows in by_id.items()
        },
        "curiosity_bias_range_per_arm_mean": {
            aid: round(_mean_key(rows, "curiosity_bias_range_mean"), 8)
            for aid, rows in by_id.items()
        },
        "label": label,
        "overall_pass": overall_pass,
        # Readiness preconditions (same-statistic discipline; can self-route not-ready).
        "preconditions": [
            {
                "name": "e2_world_forward_action_divergence_non_vacuity",
                "kind": "readiness",
                "description": (
                    "At the highest-weight arm the e2.world_forward(z0,a_i) candidate "
                    "predictions stay action-divergent (cand_world_pairwise_dist > floor) "
                    "-- the curiosity_candidate_source='e2_world_forward' / ARC-065 GAP-A "
                    "precondition. An under-trained / collapsed e2 starves the per-candidate "
                    "novelty channel."
                ),
                "control": "highest curiosity_novelty_weight arm, cand_world_pairwise_dist_mean",
                                # NOT rounded: rounding can cross the bound and break the
                # round-trip (round(1e-9, 6) == 0.0 against a 0.0 floor reads as
                # UNMET while the shipped strict `>` on the raw value says met).
                # The rounded means below are diagnostics only, never the bound.
                "measured": float(highest_weight_cand_dist_kth),
                "threshold": CAND_DIST_FLOOR,
                # FLOOR-shaped, STRICTLY so: the per-seed predicate is
                # `cand_world_pairwise_dist_mean > CAND_DIST_FLOOR` and `met`
                # counts seeds ("holds on >= MIN_SEEDS_FOR_PASS of 3"), so
                # `measured` is the MIN_SEEDS_FOR_PASS-th LARGEST per-seed value
                # rather than the mean it used to report -- a mean is not a
                # function of that seed count, so the indexer's recompute could
                # not reproduce `met`.
                "comparator": ">",
                "direction": "lower",
                "mean_across_seeds": round(highest_weight_cand_dist_mean, 6),
                "seeds_above_floor": int(cand_dist_seeds_ok),
                "min_seeds_required": MIN_SEEDS_FOR_PASS,
                "met": bool(cand_dist_seeds_ok >= MIN_SEEDS_FOR_PASS),
            },
            {
                "name": "curiosity_bias_range_supra_floor",
                "kind": "readiness",
                "description": (
                    "At the highest-weight arm the per-candidate curiosity_bias_range "
                    "(pre-clamp cross-candidate RANGE) clears the floor -- the MECH-314 "
                    "channel carries a real per-candidate pattern the top-k argmin can "
                    "arbitrate on (the 643a 'scaling zero is still zero' guard)."
                ),
                "control": "highest curiosity_novelty_weight arm, curiosity_candidate_source=e2_world_forward",
                                # NOT rounded: rounding can cross the bound and break the
                # round-trip (round(1e-9, 6) == 0.0 against a 0.0 floor reads as
                # UNMET while the shipped strict `>` on the raw value says met).
                # The rounded means below are diagnostics only, never the bound.
                "measured": float(highest_weight_bias_range_kth),
                "threshold": BIAS_RANGE_FLOOR,
                # FLOOR-shaped, STRICTLY so: per-seed predicate is
                # `curiosity_bias_range_mean > BIAS_RANGE_FLOOR`, `met` counts
                # seeds, so `measured` is the k-th LARGEST per-seed value (same
                # aggregation fix as the entry above).
                "comparator": ">",
                "direction": "lower",
                "mean_across_seeds": round(highest_weight_bias_range_mean, 8),
                "seeds_above_floor": int(bias_range_seeds_ok),
                "min_seeds_required": MIN_SEEDS_FOR_PASS,
                "met": bool(bias_range_seeds_ok >= MIN_SEEDS_FOR_PASS),
            },
            {
                "name": "committed_class_entropy_range_across_weight_arms_supra_floor",
                "kind": "readiness",
                "description": (
                    "SAME-statistic gate (V3-EXQ-643 lesson; conversion-ceiling guard): "
                    "the RANGE across the swept curiosity_novelty_weight arms of the "
                    "healthy-seed-mean committed_class_entropy -- the EXACT statistic the "
                    "Goldilocks decision routes on -- clears a floor. If committed diversity "
                    "is flat across every weight, the MECH-314 channel does not reach "
                    "committed action under MECH-439 F-dominance (the 569i conversion is "
                    "ENV-CONDITIONAL; V3-EXQ-625e) -> substrate_not_ready_requeue, NOT a "
                    "MECH-314 / DEV-NEED-003 weakens."
                ),
                "control": "RANGE of per-arm mean committed_class_entropy across weights {0,0.05,0.25,1.0}",
                                # NOT rounded: rounding can cross the bound and break the
                # round-trip (round(1e-9, 6) == 0.0 against a 0.0 floor reads as
                # UNMET while the shipped strict `>` on the raw value says met).
                # The rounded means below are diagnostics only, never the bound.
                "measured": float(entropy_range_measured),
                "threshold": ENTROPY_RANGE_FLOOR,
                # FLOOR-shaped and INCLUSIVE: the predicate is
                # `entropy_range_across_arms >= ENTROPY_RANGE_FLOOR`, a single
                # cross-arm scalar (no seed count to aggregate). Declared rather
                # than left to the default so the strictness is explicit.
                "comparator": ">=",
                "direction": "lower",
                "met": entropy_range_ok,
            },
            {
                "name": "rolled_out_zworld_magnitude_bounded",
                "kind": "readiness",
                "description": (
                    "Rolled-out z_world spread stayed finite and below the 643a explosion "
                    "ceiling (SD-056 online training numerical stability; rollout clamp ON)."
                ),
                "control": "max cand_world_pairwise_dist across all arms",
                                # NOT rounded: rounding can cross the bound and break the
                # round-trip (round(1e-9, 6) == 0.0 against a 0.0 floor reads as
                # UNMET while the shipped strict `>` on the raw value says met).
                # The rounded means below are diagnostics only, never the bound.
                "measured": float(magnitude_measured),
                "threshold": MAGNITUDE_CEIL,
                # CEILING-shaped, and STRICTLY so: the predicate is
                # `math.isfinite(max_pairwise) and max_pairwise < MAGNITUDE_CEIL`.
                # `direction` alone was not enough -- without a comparator the
                # indexer reads an upper bound as INCLUSIVE, so a run sitting
                # exactly on the ceiling would recompute as met while the shipped
                # strict `<` says otherwise.
                "comparator": "<",
                "direction": "upper",
                "met": magnitude_ok,
            },
        ],
        "criteria": [
            {
                "name": "mech314_committed_diversity_lift_over_control",
                "load_bearing": True,
                "passed": goldilocks_lift_ok,
            },
        ],
        "criteria_non_degenerate": {
            "mech314_committed_diversity_lift_over_control": entropy_range_ok,
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
                        for k in ("arm_id", "curiosity_novelty_weight", "is_control")
                    },
                    "env_kwargs": ENV_KWARGS,
                    "sd056_weight": SD056_WEIGHT,
                    "curiosity_bias_scale": CURIOSITY_BIAS_SCALE,
                    "visitation_buffer_len": VISITATION_BUFFER_LEN,
                    "curiosity_candidate_source": "e2_world_forward",
                    "use_modulatory_selection_authority": True,
                    "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
                    "use_modulatory_shortlist_then_modulate": True,
                    "modulatory_shortlist_mode": "top_k",
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

    if summary["label"] == "substrate_not_ready_requeue":
        evidence_direction = "non_contributory"
        per_claim = {c: "non_contributory" for c in CLAIM_IDS}
    elif summary["overall_pass"]:
        evidence_direction = "supports"
        per_claim = {c: "supports" for c in CLAIM_IDS}
    else:
        evidence_direction = "does_not_support"
        per_claim = {c: "does_not_support" for c in CLAIM_IDS}

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
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
        "supersedes": SUPERSEDES_RUN,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": per_claim,
        "evidence_direction_note": (
            "MECH-314 per-candidate novelty Goldilocks under the 569i top-k shortlist "
            "conversion path on the 648a-validated substrate (curiosity_candidate_source="
            "'e2_world_forward' + SD-056 online + rollout clamp + SP-CEM + V_s) with "
            "MECH-314 curiosity as the SOLE modulatory channel. The GAP-A / "
            "modulatory-bias-selection-authority stack is CONSTANT across arms "
            "(use_modulatory_selection_authority=True, modulatory_authority_gain=1.0, "
            "use_modulatory_shortlist_then_modulate=True, top-k k=3). The swept knob is "
            "curiosity_novelty_weight -- under the top-k shortlist the committed action is "
            "the within-top-3 argmin of the curiosity-fed _modulatory_accum, so the "
            "pre-clamp novelty weight is load-bearing (NOT washed out by the authority "
            "rescale, unlike the 590a/590b additive path). DV = committed-action-class "
            "entropy. PASS (supports MECH-314 + DEV-NEED-003) = a Goldilocks weight lifts "
            "committed_class_entropy over the w=0 control by >= margin on >=2/3 seeds. The "
            "MANDATORY non-vacuity gate: if committed-selection diversity does NOT survive "
            "MECH-439 F-dominance (committed_class_entropy flat across weights, OR the "
            "curiosity bias range / e2 divergence below floor -- the SAME statistic the "
            "Goldilocks routes on), the run self-routes substrate_not_ready_requeue "
            "(non_contributory), NOT a weakens. claim_ids re-evaluated from scratch "
            "(MECH-314 parent novelty channel + DEV-NEED-003; NOT 590b's MECH-314a). "
            "supersedes V3-EXQ-590b (FAIL/non_contributory 2026-06-11)."
        ),
        "interpretation": {
            "label": summary["label"],
            "preconditions": summary["preconditions"],
            "criteria": summary["criteria"],
            "criteria_non_degenerate": summary["criteria_non_degenerate"],
            "routing": {
                "substrate_not_ready_requeue": (
                    "conversion ceiling persists / e2 collapsed -- re-queue as 590d on a "
                    "GAP-A-readier substrate; do NOT weaken MECH-314 / DEV-NEED-003"
                ),
                "mech314_committed_diversity_goldilocks_identified": (
                    "PASS (supports MECH-314 + DEV-NEED-003); adopt best curiosity_novelty_weight"
                ),
                "mech314_no_committed_diversity_benefit": (
                    "FAIL (does_not_support); /failure-autopsy the conversion reading"
                ),
            },
        },
        "dry_run": bool(dry_run),
        "config": {
            "seeds": seeds,
            "p0_warmup_episodes": p0,
            "p1_measurement_episodes": p1,
            "steps_per_episode": steps,
            "measure_after_tick": measure_after,
            "swept_knob": "curiosity_novelty_weight",
            "curiosity_novelty_weights": CURIOSITY_WEIGHTS,
            "modulatory_authority_gain_fixed": MODULATORY_AUTHORITY_GAIN,
            "modulatory_shortlist_mode": "top_k",
            "modulatory_shortlist_k": SHORTLIST_K,
            "env_kwargs": ENV_KWARGS,
            "sd056_weight": SD056_WEIGHT,
            "curiosity_bias_scale": CURIOSITY_BIAS_SCALE,
            "visitation_buffer_len": VISITATION_BUFFER_LEN,
            "thresholds": {
                "cand_dist_floor": CAND_DIST_FLOOR,
                "bias_range_floor": BIAS_RANGE_FLOOR,
                "entropy_range_floor": ENTROPY_RANGE_FLOOR,
                "magnitude_ceil": MAGNITUDE_CEIL,
                "entropy_lift_margin": ENTROPY_LIFT_MARGIN,
                "min_seeds_for_pass": MIN_SEEDS_FOR_PASS,
            },
        },
        "acceptance_criteria": {
            "readiness_substrate_ready": summary["readiness"]["readiness_ok"],
            "mech314_committed_diversity_lift_over_control": summary["goldilocks"]["goldilocks_lift_ok"],
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
        print("Dry run -- manifest written then relocated by emit_outcome.", flush=True)
        with open(out_path, "w") as fh:
            json.dump(manifest, fh, indent=2)

    print(f"Outcome: {outcome} (label={summary['label']})", flush=True)
    for k, v in manifest["acceptance_criteria"].items():
        print(f"  {k}: {v}", flush=True)
    print(
        f"  goldilocks: best_arm={summary['goldilocks']['best_arm']} "
        f"weight={summary['goldilocks']['best_weight']} "
        f"committed_class_entropy {summary['goldilocks']['best_arm_committed_class_entropy']} "
        f"vs control {summary['goldilocks']['control_committed_class_entropy']} "
        f"(nonmonotone={summary['goldilocks']['nonmonotone_inverted_u']})",
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
