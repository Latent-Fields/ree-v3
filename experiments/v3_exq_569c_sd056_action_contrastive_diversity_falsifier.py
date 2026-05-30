#!/opt/local/bin/python3
"""V3-EXQ-569c -- SD-056 action-contrastive matched-entropy FP-2 falsifier.

V3-EXQ-569c supersedes V3-EXQ-569b which never actually executed: cloud-4
claimed it at 2026-05-29T21:11:24Z, then ~65 s later the cloud-4
ree-runner.service was systemctl-restarted (to pick up the new
PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE=1 env var). systemd's cgroup-kill
sent SIGTERM to the experiment subprocess; Python's default handler
exited 143; no sentinel was written; the runner's ERROR-result branch
removed the queue entry and POST /queue/remove'd it to the coordinator.
No manifest, no governance evidence. 569c is a bit-identical re-run of
569b with a fresh queue_id so any worker that has 569b in its persisted
runner_status completed list cannot silently skip it.

V3-EXQ-569b supersedes V3-EXQ-569a (crashed twice with
"probability tensor contains either inf, nan or element < 0" at
e3_selector.py:870 inside torch.multinomial). Root cause was the
self-anchored contrastive training step at _e2_contrastive_step (569a
line 379-380): targets were derived from E2's OWN current forward pass
(targets = e2.world_forward(z0, actions).detach()), making the InfoNCE
objective self-referential. The targets followed the predictions every
step, gradients pushed preds away from prior-step targets, weights
drifted unboundedly, world_forward overflowed to Inf, scores went Inf,
softmax produced NaN, and torch.multinomial raised. ARM_3 (highest
weight) would have crashed fastest; cloud-2 hit ARM_1_W001 and
DLAPTOP-4 hit ARM_2_W005 -- both partway through P0/P1, consistent
with progressive divergence.

569b fix: anchor the contrastive targets in observed environment
dynamics. A rolling buffer captures (z_world_0, action_taken,
z_world_1_observed) triples each env tick. The contrastive step
samples K diverse triples (first-action-class-spread preferred) and
runs world_forward_contrastive_loss against the observed next-states.
This matches the contract-test pattern (test_sd_056_e2_action_contrastive
constructs targets as z_world_0 + per_action_shift) and the SD-056
design memo's intent: the targets must be independent of the weights
being trained. Belt-and-braces: skip the step if the loss is non-finite,
clip E2 gradients to norm 1.0 before optimiser.step().

Claims:    [ARC-065, MECH-341]
           ARC-065 directly (SP-CEM child / A_only falsifier; R1.b unlock).
           MECH-341 indirectly via the "E3 aggregation collapses upstream
           variance" interpretation cell -- the C1+!C3 outcome weakens
           MECH-341's non-load-bearing reading by demonstrating that
           per-candidate z_world variance is present yet behavioural
           diversity does not lift.
Supersedes: V3-EXQ-569 (non_contributory 2026-05-16, bias_fraction=0 collapse).
Plan-of-record: REE_assembly/evidence/planning/behavioral_diversity_isolation_plan.md
Predecessor diagnostic: REE_assembly/evidence/planning/v3_exq_571_root_cause_2026-05-25.md
SD-056 design memo: REE_assembly/evidence/planning/e2_action_divergence_substrate_design.md

Purpose
-------
GAP-A (Theory 1 / Layer A; behavioural diversity isolation) load-bearing
falsifier. SD-056 (e2.action_conditional_divergence_contrastive) landed
ree-v3 main 041a974 (2026-05-29); V3-EXQ-613 substrate-readiness PASS
(cand_world_pairwise_dist baseline 0.136 -> trained 0.286). V3-EXQ-613
shows the contrastive task is LEARNABLE in isolation; this experiment
tests whether the lifted per-candidate z_world variance PROPAGATES
through hippocampal scoring + E3 aggregation into observable action-class
diversity, OR whether the collapse merely migrates downstream.

Design: A_only matched-entropy falsifier. ARC-065 SP-CEM ON (main-path
default; V3-EXQ-567 baseline). MECH-341 (Layer B) OFF -- single-layer
attribution per behavioral_diversity_isolation_plan.md R1.b. MECH-313
(Layer C) OFF. MECH-269 (Layer D) at main-path default.

Arms (5)
--------
  ARM_0_OFF              SD-056 master OFF (baseline reproducing
                         V3-EXQ-569 non_contributory at the new env).
  ARM_1_W001             SD-056 ON, e2_action_contrastive_weight=0.01
                         (V3-EXQ-613 default-ON weight).
  ARM_2_W005             SD-056 ON, weight=0.05
                         (V3-EXQ-613 mid-tier on_weight).
  ARM_3_W020             SD-056 ON, weight=0.20
                         (stronger contrastive pressure).
  ARM_4_MATCHED_NOISE    SD-056 OFF + select_action softmax temperature
                         lifted to MATCHED_ENTROPY_TEMPERATURE=2.5 (per
                         V3-EXQ-569 precedent). FP-2 random-noise control.

Online E2 contrastive training (ARM_1/2/3 only)
-----------------------------------------------
SD-056's effect surfaces only when world_forward_contrastive_loss is
actually backpropped through E2 weights. We run online SGD on E2 params
each P0+P1 tick using OBSERVED env transitions as the contrastive
anchors. A per-arm ring buffer (length BUFFER_MAX) captures
(z_world_0_pre_step, action_taken_one_hot, z_world_1_post_step_observed)
triples on every env step. When the buffer holds >= CONTRASTIVE_BATCH_K
distinct first-action classes, the training step samples K class-diverse
triples and computes L_contrast = e2.world_forward_contrastive_loss(
z_world_0_batch, actions_batch, z_world_1_observed_batch). Caller
multiplies by arm-specific e2_action_contrastive_weight, clips E2
gradients to MAX_GRAD_NORM, and steps a dedicated Adam optimiser over
e2.parameters() ONLY (no joint training with E1 / E3 -- isolation per
the phased-training pattern). On ARM_0_OFF / ARM_4_MATCHED_NOISE the
opt step is skipped so the arm is a frozen-init baseline.

Acceptance criteria (pre-registered)
------------------------------------
  C1 (substrate operative): mean cand_world_pairwise_dist > 0.05 across
     >= MIN_SEEDS_PER_ARM_FOR_PASS / SEEDS in EACH of ARM_1, ARM_2, ARM_3.
     Confirms SD-056 is producing per-candidate z_world variance in
     behavioural runtime (not just on the 613 training-only diagnostic).
  C2 (R1.b unlock): in at least one of ARM_1/2/3,
     candidate_first_action_entropy > 0.3 AND > ARM_4 (matched-noise).
     Strict-above-matched-noise: structural beats random.
  C3 (selected-entropy lift): in ARM_2 OR ARM_3,
     selected_action_class_entropy > 0.3 (improves over the V3-EXQ-569
     monostrategy regime).

Overall PASS = C1 majority + (C2 OR C3) in at least one weight arm.

Interpretation grid (diagnostic block)
--------------------------------------
  All arms ~ ARM_0 -> SD-056 not propagating; substrate operative on its
    own loss but signal absorbed before reaching action selection. Route:
    diagnose hippocampal trajectory scorer where cand_world_summaries
    feed in. NOT another substrate edit.
  C1 holds, C2 / C3 fail strictly above MATCHED_NOISE -> per-candidate
    z_world variance present but E3 aggregation collapses it. Implicates
    GAP-B / MECH-341 as the load-bearing layer (consistent with EXQ-571
    finding F dominates 88-89% of E3 variance). Route: re-run with
    MECH-341 ON as A+B combined arm.
  One+ weight arm clears C1 + C2 + C3 strictly above MATCHED_NOISE ->
    R1.b fires, GAP-A unblocked, advance ARC-065 SP-CEM child toward
    provisional promotion. Apply R1.b in next /governance cycle.
  Strong weight (ARM_3) degrades core task performance (harm avoidance,
    survival) -> SD-056 weight regime upper bound found; document as
    Goldilocks constraint.
  C1 fails (substrate not operative in behavioural runtime even though
    V3-EXQ-613 PASSed in training-only setting) -> SD-056 main-path
    wiring gap; route to /diagnose-errors on the contrastive loss
    callsite under episode runtime.

Phases
------
P0 (30 ep, instrumentation OFF, training ON for ON arms): warmup. Allow
    V_s / event_segmenter / residue field to develop a realistic regime
    AND let E2 contrastive training move cand_world_pairwise_dist off
    the random-init baseline.
P1 (20 ep, instrumentation ON, training continues for ON arms):
    measurement window. Per-tick cand_world_pairwise_dist, pre-E3 class
    counts, post-E3 selected class, E3 score stats.

Budget: 5 arms x 3 seeds x 50 ep x 200 steps = 150k steps total.
~150 min on Mac (DLAPTOP-4.local @ ~14 steps/sec).

Smoke
-----
  /opt/local/bin/python3 experiments/v3_exq_569c_sd056_action_contrastive_diversity_falsifier.py --dry-run
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
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_569c_sd056_action_contrastive_diversity_falsifier"
QUEUE_ID = "V3-EXQ-569c"
CLAIM_IDS: List[str] = ["ARC-065", "MECH-341"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES = "V3-EXQ-569b"

SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 30
P1_MEASUREMENT_EPISODES = 20
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_STEPS = 30

# Acceptance thresholds (pre-registered)
C1_PAIRWISE_DIST_FLOOR = 0.05            # SD-056 substrate-operative gate
C2_FIRST_ACTION_ENTROPY_FLOOR = 0.3      # candidate pool diversity threshold
C3_SELECTED_ENTROPY_FLOOR = 0.3          # downstream selection diversity
MIN_SEEDS_PER_ARM_FOR_PASS = 2           # of 3
MATCHED_ENTROPY_TEMPERATURE = 2.5        # ARM_4 FP-2 control (per V3-EXQ-569)
E2_CONTRASTIVE_LR = 1e-3                 # Adam LR for online E2 training
E2_TRAIN_EVERY_K_TICKS = 1               # SGD step cadence
CONTRASTIVE_BATCH_K = 8                  # InfoNCE batch size from buffer
TRANSITION_BUFFER_MAX = 256              # rolling buffer length per arm
MIN_BUFFER_BEFORE_TRAIN = 16             # min entries before contrastive step fires
MIN_CLASSES_FOR_TRAIN = 2                # min distinct first-action classes in batch
MAX_GRAD_NORM = 1.0                      # E2 gradient clip (norm) for stability

# ENV identical to V3-EXQ-611b / 611 so manifest-comparability holds with the
# 611 ARM_0_ALL_OFF baseline already on origin/master.
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


# ---------------------------------------------------------------------------
# Arm definitions
# ---------------------------------------------------------------------------

ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": "ARM_0_OFF",
        "label": "sd056_master_off_baseline",
        "sd056_enabled": False,
        "sd056_weight": 0.0,
        "temperature": 1.0,
        "train_e2": False,
    },
    {
        "arm_id": "ARM_1_W001",
        "label": "sd056_on_weight_0p01",
        "sd056_enabled": True,
        "sd056_weight": 0.01,
        "temperature": 1.0,
        "train_e2": True,
    },
    {
        "arm_id": "ARM_2_W005",
        "label": "sd056_on_weight_0p05",
        "sd056_enabled": True,
        "sd056_weight": 0.05,
        "temperature": 1.0,
        "train_e2": True,
    },
    {
        "arm_id": "ARM_3_W020",
        "label": "sd056_on_weight_0p20",
        "sd056_enabled": True,
        "sd056_weight": 0.20,
        "temperature": 1.0,
        "train_e2": True,
    },
    {
        "arm_id": "ARM_4_MATCHED_NOISE",
        "label": "sd056_off_matched_entropy_temperature_control",
        "sd056_enabled": False,
        "sd056_weight": 0.0,
        "temperature": MATCHED_ENTROPY_TEMPERATURE,
        "train_e2": False,
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """Main-path SP-CEM + V_s + SD-054 stack with SD-056 arm overrides.

    MECH-341 (Layer B) is deliberately OFF (use_e3_score_diversity=False).
    MECH-313 (Layer C) is deliberately OFF (use_noise_floor=False).
    SD-056 is the single varying axis across ARM_0..ARM_3; ARM_4 uses
    increased softmax temperature at action selection (not a config flag).
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
        # ARC-065 SP-CEM (Layer A) -- main-path default
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # Layer B / C deliberately OFF (single-layer A-only falsifier)
        use_e3_score_diversity=False,
        use_noise_floor=False,
        # V_s substrate (main-path default; identical across arms)
        use_per_stream_vs=True,
        use_per_region_vs=True,
        use_event_segmenter=True,
        use_invalidation_trigger=True,
        use_anchor_sets=True,
        # SD-056 (the single varying axis)
        e2_action_contrastive_enabled=bool(arm["sd056_enabled"]),
        e2_action_contrastive_weight=float(arm["sd056_weight"]),
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# Per-tick measurement helpers
# ---------------------------------------------------------------------------

def _trajectory_first_action_class(traj) -> int:
    return int(
        traj.actions[:, 0, :].argmax(dim=-1).detach().reshape(-1)[0].item()
    )


def _first_actions_K(candidates) -> torch.Tensor:
    """Stack candidate first-step actions into [K, action_dim]."""
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


def _per_class_score_stats(
    candidates: List, last_scores: Optional[torch.Tensor]
) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    """Return (selected_class, top2_class_gap, score_std). None on insufficient data."""
    if (
        not candidates
        or len(candidates) < 2
        or last_scores is None
        or last_scores.numel() != len(candidates)
    ):
        return None, None, None
    scores_t = last_scores.detach().reshape(-1).float()
    per_class_scores: Dict[int, List[float]] = {}
    classes_per_cand: List[int] = []
    for i, traj in enumerate(candidates):
        cls = _trajectory_first_action_class(traj)
        classes_per_cand.append(cls)
        per_class_scores.setdefault(cls, []).append(float(scores_t[i].item()))
    sel_idx = int(scores_t.argmin().item())
    selected_class = int(classes_per_cand[sel_idx])
    class_means = [sum(v) / len(v) for v in per_class_scores.values()]
    sorted_means = sorted(class_means)
    top2_gap = float(sorted_means[1] - sorted_means[0]) if len(sorted_means) >= 2 else None
    score_std = float(scores_t.std(unbiased=False).item()) if scores_t.numel() > 1 else 0.0
    return selected_class, top2_gap, score_std


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
    """Pick K buffer entries spreading across first-action classes.

    First pass takes one entry per distinct first-action class (preserves
    discriminability). Second pass fills the remaining slots from the rest
    of the buffer. Returns None if fewer than MIN_CLASSES_FOR_TRAIN distinct
    classes are available -- a degenerate single-class batch has no
    informative negatives.
    """
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
    # Second pass: fill remaining slots from pool entries not yet picked.
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
    arm_weight: float,
    optimiser: torch.optim.Optimizer,
    rng: random.Random,
) -> Optional[float]:
    """Run one SGD step on E2 with the SD-056 contrastive loss.

    Targets are anchored in OBSERVED env transitions sampled from the
    rolling buffer. Returns the unweighted L_contrast scalar (or None if
    the step was skipped because the buffer was too sparse, the batch
    had insufficient class diversity, the loss came back non-finite, or
    the loss had no grad). Belt-and-braces: non-finite losses skip the
    backward+step entirely; gradients are clipped to MAX_GRAD_NORM before
    optimiser.step() so a momentary outlier transition cannot blow up
    weights the way 569a's self-anchored objective did.
    """
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
        # Skip without stepping; do not corrupt E2 weights with NaN/Inf.
        return loss_val
    if not loss.requires_grad or loss_val == 0.0:
        return loss_val
    weighted = float(arm_weight) * loss
    weighted.backward()
    torch.nn.utils.clip_grad_norm_(
        agent.e2.parameters(), max_norm=MAX_GRAD_NORM
    )
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
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = _make_env(seed)
    agent = _make_agent(env, arm)

    # E2-only optimiser for SD-056 online contrastive training.
    e2_opt: Optional[torch.optim.Optimizer] = None
    if bool(arm["train_e2"]) and bool(arm["sd056_enabled"]):
        e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)

    # Rolling buffer of OBSERVED env transitions for the SD-056
    # contrastive loss (569b fix -- replaces 569a's self-anchored target
    # form, which let weights diverge to NaN).
    transition_buffer: Deque[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)
    n_buffer_appends = 0
    n_contrastive_skipped_nonfinite = 0
    n_contrastive_skipped_sparse = 0

    arm_temperature = float(arm["temperature"])
    total_train_eps = p0_episodes + p1_episodes

    # P1 metric accumulators.
    pairwise_dists: List[float] = []
    candidate_first_action_counts: Counter = Counter()
    candidate_unique_per_tick: List[float] = []
    candidate_entropy_per_tick: List[float] = []
    selected_class_counts: Counter = Counter()
    top2_gaps: List[float] = []
    score_stds: List[float] = []
    contrastive_loss_values: List[float] = []

    # P0+P1 training-side counters.
    n_p0_ticks = 0
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
        # Pending (z_world_0, action_taken) from the previous tick; the
        # NEXT tick's latent.z_world is the observed z_world_1 for that
        # transition. Cleared on episode reset so cross-episode boundaries
        # don't pollute the buffer with non-causal (z0, a, z1) triples.
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

            # 569b fix: if last tick captured a (z_world_0, action) pending
            # the post-step observation, this tick's latent.z_world IS the
            # observed z_world_1. Append the completed (z0, a, z1) triple
            # to the contrastive-training buffer. Done BEFORE training so
            # the new sample is available in the same iteration.
            if pending_capture is not None:
                z0_prev, a_prev = pending_capture
                z1_obs = latent.z_world.detach().reshape(-1).clone()
                if (
                    torch.isfinite(z0_prev).all()
                    and torch.isfinite(a_prev).all()
                    and torch.isfinite(z1_obs).all()
                ):
                    transition_buffer.append((z0_prev, a_prev, z1_obs))
                    n_buffer_appends += 1
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

            # P1 instrumentation: pre-E3 candidate pool stats + SD-056 substrate
            # readiness (cand_world_pairwise_dist live on the runtime E2).
            pre_e3_classes: Optional[List[int]] = None
            if is_p1 and candidates:
                pre_e3_classes = [_trajectory_first_action_class(t) for t in candidates]
                candidate_first_action_counts.update(pre_e3_classes)
                candidate_unique_per_tick.append(float(len(set(pre_e3_classes))))
                cnt: Counter = Counter(pre_e3_classes)
                candidate_entropy_per_tick.append(_entropy_from_counts(dict(cnt)))
                if len(candidates) >= 2:
                    actions_K = _first_actions_K(candidates).to(agent.device)
                    z0 = latent.z_world.detach()
                    with torch.no_grad():
                        dist = float(
                            agent.e2.cand_world_pairwise_dist(z0, actions_K).item()
                        )
                    pairwise_dists.append(dist)

            # Drive z_goal (matches 611b protocol).
            if agent.goal_state is not None:
                benefit_exposure = float(obs_dict.get("benefit_exposure", 0.0)) if hasattr(
                    obs_dict.get("benefit_exposure", 0.0), "__float__"
                ) else 0.0
                try:
                    energy = float(body[0, 3].item())
                except Exception:
                    energy = 1.0
                drive_level = max(0.0, 1.0 - energy)
                agent.update_z_goal(
                    benefit_exposure=benefit_exposure,
                    drive_level=drive_level,
                )

            # Action selection with arm temperature.
            action = agent.select_action(
                candidates, ticks, temperature=arm_temperature
            )
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

            # P1 instrumentation: selected-class accounting + E3 score stats.
            if is_p1:
                last_scores = getattr(agent.e3, "last_scores", None)
                sel_class, top2_gap, score_std = _per_class_score_stats(
                    candidates, last_scores
                )
                committed_class = int(action[0].argmax().item())
                selected_class_counts[committed_class] += 1
                if top2_gap is not None:
                    top2_gaps.append(top2_gap)
                if score_std is not None:
                    score_stds.append(score_std)
                n_p1_ticks += 1
            else:
                n_p0_ticks += 1

            # SD-056 online contrastive training step (both P0 and P1).
            # 569b: anchored on OBSERVED transitions sampled from the
            # rolling buffer, NOT on E2's self-anchored predictions.
            if e2_opt is not None and tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0:
                loss_val = _e2_contrastive_step(
                    agent=agent,
                    buffer=transition_buffer,
                    arm_weight=float(arm["sd056_weight"]),
                    optimiser=e2_opt,
                    rng=sample_rng,
                )
                if loss_val is None:
                    n_contrastive_skipped_sparse += 1
                elif not math.isfinite(loss_val):
                    n_contrastive_skipped_nonfinite += 1
                elif is_p1:
                    contrastive_loss_values.append(loss_val)
                    n_contrastive_steps += 1

            # 569b fix: stash the pre-step latent + executed action so the
            # NEXT tick can append the observed transition triple to the
            # buffer. Capture is conditional on finiteness; a non-finite
            # latent / action means we've already lost the stream, and the
            # surrounding code's break on non-finite action will fire
            # below.
            if (
                e2_opt is not None
                and torch.isfinite(latent.z_world).all()
                and torch.isfinite(action).all()
            ):
                pending_capture = (
                    latent.z_world.detach().reshape(-1).clone(),
                    action.detach().reshape(-1).clone(),
                )

            # Env step.
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

        if (ep + 1) % 10 == 0 or (ep + 1) == total_train_eps:
            print(
                f"  [train] arm={arm['arm_id']} seed={seed} phase={phase_label} "
                f"ep {ep + 1}/{total_train_eps}",
                flush=True,
            )

    # Aggregate P1 metrics.
    def _mean(xs: List[float], default: float = 0.0) -> float:
        return float(sum(xs) / len(xs)) if xs else default

    def _maxx(xs: List[float], default: float = 0.0) -> float:
        return float(max(xs)) if xs else default

    def _minx(xs: List[float], default: float = 0.0) -> float:
        return float(min(xs)) if xs else default

    candidate_first_action_entropy_mean = _mean(candidate_entropy_per_tick)
    candidate_unique_mean = _mean(candidate_unique_per_tick)
    selected_action_entropy = _entropy_from_counts(dict(selected_class_counts))
    trajectory_class_count_mean = candidate_unique_mean  # alias for plan-of-record naming
    selected_n_unique = int(len(selected_class_counts))

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": int(seed),
        "n_p0_ticks": int(n_p0_ticks),
        "n_p1_ticks": int(n_p1_ticks),
        "n_contrastive_steps": int(n_contrastive_steps),
        "n_buffer_appends": int(n_buffer_appends),
        "n_contrastive_skipped_sparse": int(n_contrastive_skipped_sparse),
        "n_contrastive_skipped_nonfinite": int(n_contrastive_skipped_nonfinite),
        "error_note": error_note,
        # SD-056 substrate-operative metrics (C1 input)
        "cand_world_pairwise_dist_mean": round(_mean(pairwise_dists), 6),
        "cand_world_pairwise_dist_max": round(_maxx(pairwise_dists), 6),
        "cand_world_pairwise_dist_min": round(_minx(pairwise_dists), 6),
        # Pre-E3 candidate pool diversity (C2 input)
        "candidate_first_action_entropy_mean": round(candidate_first_action_entropy_mean, 6),
        "candidate_unique_first_action_classes_mean": round(candidate_unique_mean, 6),
        "trajectory_class_count_mean": round(trajectory_class_count_mean, 6),
        "candidate_first_action_counts": dict(sorted(candidate_first_action_counts.items())),
        # Post-E3 selection diversity (C3 input)
        "selected_action_class_entropy": round(selected_action_entropy, 6),
        "selected_class_counts": dict(sorted(selected_class_counts.items())),
        "selected_classes_n_unique": selected_n_unique,
        # E3 score diagnostics (for second-layer interpretation cell)
        "e3_top2_class_gap_mean": round(_mean(top2_gaps), 6),
        "e3_score_std_mean": round(_mean(score_stds), 6),
        # SD-056 training-side telemetry
        "contrastive_loss_mean": round(_mean(contrastive_loss_values), 6),
        "contrastive_loss_min": round(_minx(contrastive_loss_values), 6),
        "contrastive_loss_max": round(_maxx(contrastive_loss_values), 6),
    }


# ---------------------------------------------------------------------------
# Cross-arm evaluation
# ---------------------------------------------------------------------------

def _arm_rows(rows: List[Dict[str, Any]], arm_id: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("arm_id") == arm_id]


def _n_seeds_above(rows: List[Dict[str, Any]], key: str, floor: float) -> int:
    return sum(1 for r in rows if float(r.get(key, 0.0)) > floor)


def _mean_key(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [float(r.get(key, 0.0)) for r in rows]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _evaluate(arm_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    arm0 = _arm_rows(arm_results, "ARM_0_OFF")
    arm1 = _arm_rows(arm_results, "ARM_1_W001")
    arm2 = _arm_rows(arm_results, "ARM_2_W005")
    arm3 = _arm_rows(arm_results, "ARM_3_W020")
    arm4 = _arm_rows(arm_results, "ARM_4_MATCHED_NOISE")

    # C1: pairwise_dist > floor in majority of seeds in EACH of ARM_1/2/3
    arm1_c1 = _n_seeds_above(arm1, "cand_world_pairwise_dist_mean", C1_PAIRWISE_DIST_FLOOR)
    arm2_c1 = _n_seeds_above(arm2, "cand_world_pairwise_dist_mean", C1_PAIRWISE_DIST_FLOOR)
    arm3_c1 = _n_seeds_above(arm3, "cand_world_pairwise_dist_mean", C1_PAIRWISE_DIST_FLOOR)
    c1_pass = (
        arm1_c1 >= MIN_SEEDS_PER_ARM_FOR_PASS
        and arm2_c1 >= MIN_SEEDS_PER_ARM_FOR_PASS
        and arm3_c1 >= MIN_SEEDS_PER_ARM_FOR_PASS
    )

    # C2: candidate first-action entropy > floor AND strictly > ARM_4 in at least one weight arm
    arm4_cand_entropy_mean = _mean_key(arm4, "candidate_first_action_entropy_mean")
    c2_arm_passes: List[str] = []
    for label, rows in [("ARM_1_W001", arm1), ("ARM_2_W005", arm2), ("ARM_3_W020", arm3)]:
        mean_cand_entropy = _mean_key(rows, "candidate_first_action_entropy_mean")
        if mean_cand_entropy > C2_FIRST_ACTION_ENTROPY_FLOOR and mean_cand_entropy > arm4_cand_entropy_mean:
            c2_arm_passes.append(label)
    c2_pass = len(c2_arm_passes) >= 1

    # C3: selected_action_entropy > floor in ARM_2 or ARM_3
    arm2_sel_entropy = _mean_key(arm2, "selected_action_class_entropy")
    arm3_sel_entropy = _mean_key(arm3, "selected_action_class_entropy")
    c3_pass = bool(
        arm2_sel_entropy > C3_SELECTED_ENTROPY_FLOOR
        or arm3_sel_entropy > C3_SELECTED_ENTROPY_FLOOR
    )

    overall_pass = bool(c1_pass and (c2_pass or c3_pass))

    return {
        # C1 detail
        "c1_floor": C1_PAIRWISE_DIST_FLOOR,
        "c1_arm1_n_seeds_above": int(arm1_c1),
        "c1_arm2_n_seeds_above": int(arm2_c1),
        "c1_arm3_n_seeds_above": int(arm3_c1),
        "c1_min_seeds_required": MIN_SEEDS_PER_ARM_FOR_PASS,
        "c1_pass": bool(c1_pass),
        # C2 detail
        "c2_floor": C2_FIRST_ACTION_ENTROPY_FLOOR,
        "c2_arm4_cand_entropy_mean": round(arm4_cand_entropy_mean, 6),
        "c2_arm1_cand_entropy_mean": round(_mean_key(arm1, "candidate_first_action_entropy_mean"), 6),
        "c2_arm2_cand_entropy_mean": round(_mean_key(arm2, "candidate_first_action_entropy_mean"), 6),
        "c2_arm3_cand_entropy_mean": round(_mean_key(arm3, "candidate_first_action_entropy_mean"), 6),
        "c2_arms_passed": c2_arm_passes,
        "c2_pass": bool(c2_pass),
        # C3 detail
        "c3_floor": C3_SELECTED_ENTROPY_FLOOR,
        "c3_arm0_selected_entropy_mean": round(_mean_key(arm0, "selected_action_class_entropy"), 6),
        "c3_arm1_selected_entropy_mean": round(_mean_key(arm1, "selected_action_class_entropy"), 6),
        "c3_arm2_selected_entropy_mean": round(arm2_sel_entropy, 6),
        "c3_arm3_selected_entropy_mean": round(arm3_sel_entropy, 6),
        "c3_arm4_selected_entropy_mean": round(_mean_key(arm4, "selected_action_class_entropy"), 6),
        "c3_pass": bool(c3_pass),
        # Pairwise dist means per arm (interpretation aid)
        "pairwise_dist_arm0_mean": round(_mean_key(arm0, "cand_world_pairwise_dist_mean"), 6),
        "pairwise_dist_arm1_mean": round(_mean_key(arm1, "cand_world_pairwise_dist_mean"), 6),
        "pairwise_dist_arm2_mean": round(_mean_key(arm2, "cand_world_pairwise_dist_mean"), 6),
        "pairwise_dist_arm3_mean": round(_mean_key(arm3, "cand_world_pairwise_dist_mean"), 6),
        "pairwise_dist_arm4_mean": round(_mean_key(arm4, "cand_world_pairwise_dist_mean"), 6),
        # E3 score stats per arm
        "e3_top2_gap_arm0_mean": round(_mean_key(arm0, "e3_top2_class_gap_mean"), 6),
        "e3_top2_gap_arm2_mean": round(_mean_key(arm2, "e3_top2_class_gap_mean"), 6),
        "e3_top2_gap_arm3_mean": round(_mean_key(arm3, "e3_top2_class_gap_mean"), 6),
        # Overall
        "overall_pass": overall_pass,
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
            arm_results.append(cell)
            passed = cell.get("error_note") is None
            print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)

    summary = _evaluate(arm_results)
    outcome = "PASS" if summary["overall_pass"] else "FAIL"
    # Per-claim direction is the same for both tagged claims in this multi-claim
    # experiment: a PASS supports ARC-065 (R1.b unlock) AND weakens MECH-341's
    # non-load-bearing reading; a FAIL leaves ARC-065 unchanged at the
    # behavioural diversity question and supports the GAP-B / MECH-341
    # load-bearing reading per the interpretation grid in the docstring.
    per_claim_direction = (
        {"ARC-065": "supports", "MECH-341": "weakens"}
        if outcome == "PASS"
        else {"ARC-065": "weakens", "MECH-341": "supports"}
    )

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
        "supersedes": SUPERSEDES,
        "evidence_direction": "mixed" if outcome == "FAIL" else "supports",
        "evidence_direction_per_claim": per_claim_direction,
        "evidence_direction_note": (
            "SD-056 matched-entropy FP-2 falsifier for GAP-A (behavioural "
            "diversity isolation, Theory 1 / Layer A). PASS = SD-056 lifts "
            "per-candidate z_world variance AND that variance propagates "
            "through hippocampal scoring + E3 aggregation into observable "
            "action-class diversity strictly above the matched-entropy random-"
            "noise control. FAIL with C1 holding but C2/C3 failing = per-"
            "candidate variance present but E3 aggregation collapses it; "
            "implicates GAP-B / MECH-341 as load-bearing (consistent with "
            "EXQ-571 finding F dominates 88-89% of E3 variance). FAIL with C1 "
            "failing = SD-056 main-path wiring gap; route to /diagnose-errors."
        ),
        "dry_run": bool(dry_run),
        "config": {
            "seeds": seeds,
            "p0_warmup_episodes": p0,
            "p1_measurement_episodes": p1,
            "steps_per_episode": steps,
            "env_kwargs": ENV_KWARGS,
            "arms": [{"arm_id": a["arm_id"], "label": a["label"],
                       "sd056_enabled": a["sd056_enabled"],
                       "sd056_weight": a["sd056_weight"],
                       "temperature": a["temperature"],
                       "train_e2": a["train_e2"]} for a in ARMS],
            "c1_pairwise_dist_floor": C1_PAIRWISE_DIST_FLOOR,
            "c2_first_action_entropy_floor": C2_FIRST_ACTION_ENTROPY_FLOOR,
            "c3_selected_entropy_floor": C3_SELECTED_ENTROPY_FLOOR,
            "min_seeds_per_arm_for_pass": MIN_SEEDS_PER_ARM_FOR_PASS,
            "matched_entropy_temperature": MATCHED_ENTROPY_TEMPERATURE,
            "e2_contrastive_lr": E2_CONTRASTIVE_LR,
            "e2_train_every_k_ticks": E2_TRAIN_EVERY_K_TICKS,
        },
        "acceptance_criteria": {
            "C1_substrate_operative": summary["c1_pass"],
            "C2_r1b_unlock_first_action_entropy_above_matched_noise": summary["c2_pass"],
            "C3_selected_action_entropy_lift": summary["c3_pass"],
            "overall_pass": summary["overall_pass"],
        },
        "summary": summary,
        "arm_results": arm_results,
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"

    if not dry_run:
        with open(out_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"Manifest written: {out_path}", flush=True)
    else:
        out_path = Path("/dev/null")
        print("Dry run -- manifest not written.", flush=True)

    print(f"Outcome: {outcome}", flush=True)
    for k, v in manifest["acceptance_criteria"].items():
        print(f"  {k}: {v}", flush=True)

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-569b SD-056 matched-entropy FP-2 falsifier (GAP-A)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Short smoke run; no manifest written.",
    )
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
    sys.exit(0)
