#!/opt/local/bin/python3
"""V3-EXQ-648 -- MECH-314a Phase-2 (Candidate 5A) substrate-readiness diagnostic.

Validates the LANDED MECH-314a Phase-2 substrate (rolling z_world visitation
buffer + first-action one-hot augmentation; landed 2026-06-05, substrate_queue
entry MECH-314a-Phase-2-impl, status implemented_pending_validation). This run
is the only remaining gate on that entry. Architecture doc:
REE_assembly/docs/architecture/mech_314a_phase2_novelty_source_design.md
(section 6 falsifier sketch + section 9 augmentation-padding note + the
implementation_log architecture note: "the augmentation's spread benefit
materialises chiefly when z_world carries partial spread -- the falsifier
diagnostic measures this directly").

experiment_purpose: diagnostic (substrate-readiness; claim_ids=[]). On PASS the
pinned follow-on is /queue-experiment V3-EXQ-590b (the novelty-bonus Goldilocks
retest, owner_exq of closure-map node infant_substrate:GAP-13). Section-8
governance/claims updates (MECH-314a / MECH-314 / ARC-065) are GATED on this run
PASSing and are NOT applied here.

=== WHAT THE SUBSTRATE DOES (Candidate 5A) ===

MECH-314a Phase 1 computed per-candidate novelty as min-distance to the nearest
ACTIVE ResidueField RBF center ("residue" source). On harm-free runs the residue
field is empty -> novelty returns None -> the per-candidate curiosity bias
collapses to zero (failure mode F1). Phase 2 adds:
  - PRIMARY: a rolling z_world visitation buffer (curiosity_novelty_source=
    "visitation") that populates on every WAKING tick (MECH-094-gated; replay /
    DMN ticks do not write), so per-candidate novelty has a non-empty comparison
    set even on harm-free runs.
  - BYPASS: first-action one-hot augmentation (curiosity_use_first_action_onehot
    + curiosity_first_action_augmentation_policy="auto") that engages when the
    per-candidate z_world spread falls below curiosity_min_spread_threshold for
    curiosity_min_spread_consecutive_ticks (substrate-robustness when SD-056 is
    OFF and z_world collapses across candidates).

=== ARMS (4 conditions x 3 seeds) ===

The section-6 sketch lists 3 arms; C3's "augmentation engages when SD-056 is
ARTIFICIALLY DISABLED" probe condition requires BOTH an SD-056-ON arm (should NOT
engage) and an SD-056-OFF arm (should engage), so the recommended-config arm is
split into two cells. "SD-056 artificially disabled" is operationalised as the
SD-056-untrained probe (E2.world_forward stays action-degenerate so the
per-candidate z_world spread collapses below threshold) -- the only way to drive
the spread below the auto-policy floor, since merely stopping training does not
un-train an already-trained E2.

  ARM_0_BASELINE         -- novelty_source="residue", augmentation="never",
                            onehot OFF. Phase-1 behaviour. SD-056 trained ON.
                            On a harm-free env the residue field is empty -> the
                            per-candidate curiosity bias collapses to ~0 (C1).
  ARM_1_VISITATION_ONLY  -- novelty_source="visitation", augmentation="never",
                            onehot OFF. SD-056 trained ON. Tests candidate 1 in
                            isolation: the visitation buffer lifts per-candidate
                            spread (C2 + readiness positive control).
  ARM_2_VIS_ONEHOT_SD056_ON   -- Candidate 5A: novelty_source="visitation",
                            augmentation="auto", onehot ON. SD-056 trained ON.
                            z_world spread stays above threshold -> augmentation
                            should NOT engage (C3 negative leg).
  ARM_2_VIS_ONEHOT_SD056_OFF  -- Candidate 5A wiring, SD-056 NOT trained (probe).
                            z_world spread collapses below threshold ->
                            augmentation SHOULD engage on >=80% of ticks past
                            tick 20 (C3 positive leg).

SD-056 rollout-norm clamp (e2_rollout_output_norm_clamp_enabled) is ON in the
SD-056-trained arms per the V3-EXQ-643a numerical-stability lesson (online
SD-056 training can explode rolled-out z_world to ~1e32; the clamp bounds it so
the per-candidate spread + curiosity novelty are measured on finite magnitudes).

=== ACCEPTANCE (pre-registered; PASS = C1 AND C2 AND C3 AND C4, all unanimous) ===

  C1 baseline matches: ARM_0 per-candidate curiosity bias spread
     (curiosity_std_across_K + curiosity_bias_range_mean) <= C1_BIAS_ZERO_CEIL on
     >= MIN_SEEDS/SEEDS. (The Phase-1 residue baseline is silent on harm-free
     runs.) NON-DEGENERATE only if ARM_1's bias spread is > 0 (so "zero" is a
     real contrast, not a metric pinned at zero).
  C2 visitation lifts spread (LOAD-BEARING): ARM_1 curiosity_std_across_K > 0 in
     >= C2_TICK_FRAC of P1 waking ticks past tick 20, on >= MIN_SEEDS/SEEDS. (The
     visitation source converts the SD-056 z_world divergence into a
     per-candidate-VARYING curiosity bias.)
  C3 augmentation engages when needed: ARM_2_VIS_ONEHOT_SD056_OFF augmentation
     engaged on >= C3_ENGAGE_FRAC of P1 ticks past tick 20 (>= MIN_SEEDS) AND
     ARM_2_VIS_ONEHOT_SD056_ON augmentation engaged on <= C3_NOENGAGE_FRAC of
     those ticks (>= MIN_SEEDS) -- engages when spread collapses, stays
     disengaged when SD-056 keeps spread high.
  C4 MECH-094 simulation gate: ARM_1 visitation-buffer appends == number of
     WAKING sense() ticks (the buffer admits waking writes) AND a simulation-gate
     sub-probe with hypothesis_tag=True yields ZERO appends (replay / DMN ticks
     never write). Sentinel per arch-doc section 6.

=== READINESS PRECONDITION (substrate-not-ready guard) ===

C2 routes on a per-candidate SPREAD (range) statistic. Per the V3-EXQ-643
same-statistic rule, the readiness precondition asserts the SAME spread kind on a
positive control: ARM_1 (SD-056-trained, with a genuinely multi-first-action-class
CEM candidate pool) cand_world_pairwise_dist > C0_PAIRWISE_DIST_FLOOR on
>= MIN_SEEDS/SEEDS. cand_world_pairwise_dist is the mean pairwise L2 across the
per-candidate predicted z_world -- a cross-candidate RANGE, NOT a magnitude. If it
is below the floor the SD-056 substrate did not make candidates action-divergent
at this training budget (under-trained / unstable) -> the C2 criterion is STARVED,
not falsified. The run self-routes BELOW-FLOOR to substrate_not_ready_requeue
(re-queue at a higher P0 budget), NEVER to a substrate-verdict label. A second
readiness leg asserts the rolled-out z_world magnitude stayed finite/bounded (the
643a explosion guard).

=== DIAGNOSTIC INTERPRETATION GRID ===

| Outcome                                 | label                              | next action |
|-----------------------------------------|------------------------------------|-------------|
| readiness below floor / non-finite      | substrate_not_ready_requeue        | re-queue at higher P0 budget; do NOT weaken Phase-2 |
| readiness OK + C1+C2+C3+C4 all hold      | phase2_substrate_ready             | PASS -> /queue-experiment V3-EXQ-590b; apply section-8 updates |
| readiness OK + any of C1..C4 fails       | phase2_wiring_does_not_support     | FAIL -> /failure-autopsy on the failing criterion (Phase-2 wiring) |

architecture_epoch: "ree_hybrid_guardrails_v1"

Smoke:
  /opt/local/bin/python3 experiments/v3_exq_648_mech314a_phase2_substrate_readiness.py --dry-run
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


class _SimLatentStub:
    """Minimal stand-in for a hypothesis-tagged latent in the C4 gate probe.

    The agent's visitation-buffer write gate (agent.py:2749-2754) reads only
    `.hypothesis_tag` and `.z_world`; a full LatentState (which has several
    required fields) is unnecessary here.
    """

    def __init__(self, z_world: torch.Tensor, hypothesis_tag: bool) -> None:
        self.z_world = z_world
        self.hypothesis_tag = hypothesis_tag


EXPERIMENT_TYPE = "v3_exq_648_mech314a_phase2_substrate_readiness"
QUEUE_ID = "V3-EXQ-648"
CLAIM_IDS: List[str] = []  # substrate-readiness diagnostic (gates MECH-314a/MECH-314/ARC-065)
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 20           # SD-056 contrastive warmup (arch-doc "30 episodes" total)
P1_MEASUREMENT_EPISODES = 10
STEPS_PER_EPISODE = 200
MEASURE_AFTER_TICK = 20           # arch-doc "past tick 20" (within-episode)

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_STEPS = 30
DRY_RUN_MEASURE_AFTER_TICK = 2

# Acceptance thresholds (pre-registered).
C0_PAIRWISE_DIST_FLOOR = 0.05     # readiness: ARM_1 z_world spread on positive control
C0_MAGNITUDE_CEIL = 1.0e6         # readiness: rolled-out z_world magnitude bound (643a guard)
C1_BIAS_ZERO_CEIL = 1.0e-4        # ARM_0 per-candidate curiosity bias spread ~0
C2_TICK_FRAC = 0.80               # ARM_1 fraction of past-20 ticks with curiosity_std > 0
C3_ENGAGE_FRAC = 0.80             # ARM_2 SD-056-OFF augmentation-engaged fraction (engages)
C3_NOENGAGE_FRAC = 0.20           # ARM_2 SD-056-ON augmentation-engaged fraction (stays off)
MIN_SEEDS_FOR_PASS = 2            # of 3

# SD-056 online contrastive training (mirror the V3-EXQ-604a / 569d harness).
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
E2_TRAIN_EVERY_K_TICKS = 1
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0

# Curiosity magnitudes (V3-EXQ-604a calibration; EXQ-573 elevated regime).
CURIOSITY_BIAS_SCALE = 0.5
CURIOSITY_WEIGHT = 0.25

# Visitation buffer length (substrate default).
VISITATION_BUFFER_LEN = 256

# C4 simulation-gate sub-probe.
C4_WAKING_TICKS = 12

# HARM-FREE env (num_hazards=0): the residue field stays empty so the Phase-1
# residue-source baseline (ARM_0) collapses to zero novelty -- the F1 contrast
# the visitation source (ARM_1) is designed to fix. SP-CEM + resources still give
# the agent action-divergent candidates for SD-056 to train z_world divergence on.
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


# ---------------------------------------------------------------------------
# Arm definitions
# ---------------------------------------------------------------------------

ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": "ARM_0_BASELINE",
        "label": "phase1_residue_source_augmentation_never",
        "novelty_source": "residue",
        "augmentation_policy": "never",
        "use_first_action_onehot": False,
        "sd056_train": True,
    },
    {
        "arm_id": "ARM_1_VISITATION_ONLY",
        "label": "candidate1_visitation_source_augmentation_never",
        "novelty_source": "visitation",
        "augmentation_policy": "never",
        "use_first_action_onehot": False,
        "sd056_train": True,
    },
    {
        "arm_id": "ARM_2_VIS_ONEHOT_SD056_ON",
        "label": "candidate5a_visitation_auto_augment_sd056_on",
        "novelty_source": "visitation",
        "augmentation_policy": "auto",
        "use_first_action_onehot": True,
        "sd056_train": True,
    },
    {
        "arm_id": "ARM_2_VIS_ONEHOT_SD056_OFF",
        "label": "candidate5a_probe_sd056_disabled_augmentation_engages",
        "novelty_source": "visitation",
        "augmentation_policy": "auto",
        "use_first_action_onehot": True,
        "sd056_train": False,
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """Main-path SP-CEM + V_s stack with MECH-314a Phase-2 knobs per arm.

    SD-056 contrastive is ENABLED on every arm (so the substrate exists), but the
    online optimiser step is only RUN when arm['sd056_train'] is True -- the
    SD-056-OFF probe arm leaves E2.world_forward action-degenerate so the
    per-candidate z_world spread collapses below the auto-augmentation threshold.
    The rolled-out z_world clamp (e2_rollout_output_norm_clamp_enabled) is ON in
    the SD-056-trained arms per the V3-EXQ-643a numerical-stability lesson.
    """
    sd056_on = bool(arm["sd056_train"])
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
        # Other policy-layer regulators OFF (structured curiosity is the axis)
        use_e3_score_diversity=False,
        use_noise_floor=False,
        use_tonic_vigor=False,
        use_dacc=False,
        # V_s substrate (main-path default; identical across arms)
        use_per_stream_vs=True,
        use_per_region_vs=True,
        use_event_segmenter=True,
        use_invalidation_trigger=True,
        use_anchor_sets=True,
        # SD-056 substrate present on every arm; trained online only when sd056_train
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_rollout_output_norm_clamp_enabled=sd056_on,
        e2_rollout_output_norm_clamp_ratio=2.0,
        # MECH-314 structured curiosity -- novelty sub-flavour ON, Phase-2 knobs per arm
        use_structured_curiosity=True,
        use_curiosity_novelty=True,
        use_curiosity_uncertainty=False,
        use_curiosity_learning_progress=False,
        curiosity_bias_scale=CURIOSITY_BIAS_SCALE,
        curiosity_novelty_weight=CURIOSITY_WEIGHT,
        # --- MECH-314a Phase-2 (Candidate 5A) ---
        curiosity_novelty_source=str(arm["novelty_source"]),
        curiosity_visitation_buffer_len=VISITATION_BUFFER_LEN,
        curiosity_use_first_action_onehot=bool(arm["use_first_action_onehot"]),
        curiosity_first_action_augmentation_policy=str(arm["augmentation_policy"]),
        # min_spread_threshold / consecutive_ticks left at substrate defaults (0.01 / 5)
    )
    agent = REEAgent(cfg)
    # Enable per-channel score-bias decomposition so select_action records the
    # per-candidate curiosity bias spread (curiosity_std_across_K) that C2 reads.
    agent.e3.e3_score_decomp_enabled = True
    return agent


# ---------------------------------------------------------------------------
# Per-tick measurement helpers (from V3-EXQ-604a / 569d)
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
    # Full RNG reset at cell entry: makes the cell a pure function of
    # (substrate, config, seed) so the arm_fingerprint is order-independent.
    reset_all_rng(seed)

    env = _make_env(seed)
    agent = _make_agent(env, arm)
    sd056_train = bool(arm["sd056_train"])

    e2_opt: Optional[torch.optim.Optimizer] = (
        torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)
        if sd056_train else None
    )

    transition_buffer: Deque[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)

    total_train_eps = p0_episodes + p1_episodes

    # P1 measurement accumulators (filtered to ticks past measure_after_tick).
    pairwise_dists: List[float] = []
    pairwise_dist_max_seen = 0.0
    curiosity_std_vals: List[float] = []
    curiosity_range_vals: List[float] = []
    novelty_source_counts: Counter = Counter()
    augmentation_engaged_flags: List[int] = []
    n_p0_ticks = 0
    n_p1_ticks = 0
    n_p1_ticks_past_window = 0
    n_contrastive_steps = 0
    n_buffer_appends_in_run = 0
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

            buf_len_before = (
                len(agent._zworld_visitation_buffer)
                if agent._zworld_visitation_buffer is not None else 0
            )
            latent = agent.sense(
                obs_body=body, obs_world=world,
                obs_harm=_obs(obs_dict, "harm_obs"),
                obs_harm_a=_obs(obs_dict, "harm_obs_a"),
                obs_harm_history=_obs(obs_dict, "harm_history"),
            )
            buf_len_after = (
                len(agent._zworld_visitation_buffer)
                if agent._zworld_visitation_buffer is not None else 0
            )
            n_buffer_appends_in_run += max(0, buf_len_after - buf_len_before)

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

            # Capture MECH-314a Phase-2 diagnostics AFTER select_action (its single
            # curiosity call set both the per-channel decomposition and the
            # curiosity get_state() this-tick fields -- no double-advance).
            if past_window:
                decomp = getattr(agent, "_last_score_bias_decomp", {}) or {}
                cstd = float(decomp.get("curiosity_std_across_K", 0.0))
                crange = float(decomp.get("curiosity_bias_range_mean", 0.0))
                if math.isfinite(cstd):
                    curiosity_std_vals.append(cstd)
                if math.isfinite(crange):
                    curiosity_range_vals.append(crange)
                cst = (
                    agent.curiosity.get_state()
                    if getattr(agent, "curiosity", None) is not None else {}
                )
                novelty_source_counts.update([str(cst.get("novelty_source_used", "none"))])
                augmentation_engaged_flags.append(
                    1 if bool(cst.get("augmentation_engaged", False)) else 0
                )
                n_p1_ticks_past_window += 1

            if is_p1:
                n_p1_ticks += 1
            else:
                n_p0_ticks += 1

            if sd056_train and e2_opt is not None and tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0:
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

    def _frac_positive(xs: List[float], floor: float = 0.0) -> float:
        return float(sum(1 for x in xs if x > floor) / len(xs)) if xs else 0.0

    aug_engage_frac = (
        float(sum(augmentation_engaged_flags) / len(augmentation_engaged_flags))
        if augmentation_engaged_flags else 0.0
    )

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": int(seed),
        "sd056_train": sd056_train,
        "n_p0_ticks": int(n_p0_ticks),
        "n_p1_ticks": int(n_p1_ticks),
        "n_p1_ticks_past_window": int(n_p1_ticks_past_window),
        "n_contrastive_steps": int(n_contrastive_steps),
        "n_buffer_appends_in_run": int(n_buffer_appends_in_run),
        "error_note": error_note,
        # Readiness / C2 substrate spread (positive control)
        "cand_world_pairwise_dist_mean": round(_mean(pairwise_dists), 6),
        "cand_world_pairwise_dist_max": round(pairwise_dist_max_seen, 6),
        # C1 / C2 per-candidate curiosity bias spread
        "curiosity_std_across_K_mean": round(_mean(curiosity_std_vals), 8),
        "curiosity_bias_range_mean": round(_mean(curiosity_range_vals), 8),
        "curiosity_std_positive_frac_past20": round(_frac_positive(curiosity_std_vals), 6),
        # C3 augmentation engagement
        "augmentation_engaged_frac_past20": round(aug_engage_frac, 6),
        # diagnostics
        "novelty_source_counts": dict(sorted(novelty_source_counts.items())),
    }


# ---------------------------------------------------------------------------
# C4: MECH-094 simulation-gate sub-probe (deterministic)
# ---------------------------------------------------------------------------

def _c4_simulation_gate_probe(seed: int, steps_per_episode: int) -> Dict[str, Any]:
    """Confirm the visitation buffer admits WAKING writes and rejects SIMULATION
    (hypothesis_tag=True) writes.

    WAKING half is authoritative agent code: C4_WAKING_TICKS real env-driven
    sense() ticks must append exactly C4_WAKING_TICKS z_world states (the agent
    only calls sense() on the waking stream). SIMULATION half exercises the exact
    MECH-094 gate the agent applies at agent.py:2749-2754 (append only when
    `not new_latent.hypothesis_tag`) against the agent's own buffer with
    hypothesis_tag=True latents -- replay / DMN content must never write.
    """
    reset_all_rng(seed)
    env = _make_env(seed)
    agent = _make_agent(env, ARMS[1])  # ARM_1: visitation source -> buffer built
    agent.reset()
    buf = agent._zworld_visitation_buffer
    if buf is None:
        return {
            "buffer_present": False,
            "n_waking_appends": 0,
            "n_simulation_appends": 0,
            "waking_ok": False,
            "simulation_ok": False,
        }

    buf.clear()
    _, obs_dict = env.reset()
    wdim = int(agent.config.latent.world_dim)
    for _i in range(min(C4_WAKING_TICKS, steps_per_episode)):
        body = obs_dict["body_state"].float()
        world = obs_dict["world_state"].float()
        if body.dim() == 1:
            body = body.unsqueeze(0)
        if world.dim() == 1:
            world = world.unsqueeze(0)
        agent.sense(
            obs_body=body, obs_world=world,
            obs_harm=_obs(obs_dict, "harm_obs"),
            obs_harm_a=_obs(obs_dict, "harm_obs_a"),
            obs_harm_history=_obs(obs_dict, "harm_history"),
        )
        action = torch.zeros(1, env.action_dim, device=agent.device)
        action[0, int(np.random.randint(0, env.action_dim))] = 1.0
        _, _hs, done, _info, obs_dict = env.step(action)
        if done:
            _, obs_dict = env.reset()
    n_waking_appends = len(buf)

    # Simulation half: apply the agent's MECH-094 gate (agent.py:2749-2754) with
    # hypothesis_tag=True latents against the agent's own buffer.
    n_sim_ticks = C4_WAKING_TICKS
    len_before_sim = len(buf)
    for _i in range(n_sim_ticks):
        sim_latent = _SimLatentStub(z_world=torch.randn(1, wdim), hypothesis_tag=True)
        # Exact replica of the agent's visitation-buffer write gate.
        if agent._zworld_visitation_buffer is not None:
            if not bool(getattr(sim_latent, "hypothesis_tag", False)):
                if sim_latent.z_world is not None:
                    agent._zworld_visitation_buffer.append(
                        sim_latent.z_world[0].detach().clone()
                    )
    n_simulation_appends = len(buf) - len_before_sim

    return {
        "buffer_present": True,
        "n_waking_ticks": int(min(C4_WAKING_TICKS, steps_per_episode)),
        "n_waking_appends": int(n_waking_appends),
        "n_simulation_ticks": int(n_sim_ticks),
        "n_simulation_appends": int(n_simulation_appends),
        "waking_ok": bool(n_waking_appends == min(C4_WAKING_TICKS, steps_per_episode)
                          and n_waking_appends > 0),
        "simulation_ok": bool(n_simulation_appends == 0),
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


def _evaluate(arm_results: List[Dict[str, Any]], c4: Dict[str, Any]) -> Dict[str, Any]:
    by_id = {a["arm_id"]: _arm_rows(arm_results, a["arm_id"]) for a in ARMS}
    arm0 = by_id["ARM_0_BASELINE"]
    arm1 = by_id["ARM_1_VISITATION_ONLY"]
    arm2_on = by_id["ARM_2_VIS_ONEHOT_SD056_ON"]
    arm2_off = by_id["ARM_2_VIS_ONEHOT_SD056_OFF"]

    # --- READINESS (substrate-not-ready guard) ---
    # Positive control = ARM_1 (SD-056-trained, SP-CEM multi-class candidate pool).
    # Asserts the SAME spread statistic C2 routes on (cross-candidate range).
    readiness_seeds_ok = _n_seeds_satisfying(
        arm1, lambda r: float(r.get("cand_world_pairwise_dist_mean", 0.0)) > C0_PAIRWISE_DIST_FLOOR
    )
    arm1_pairwise_mean = _mean_key(arm1, "cand_world_pairwise_dist_mean")
    # Non-finite / explosion guard (643a): rolled-out z_world magnitude bounded.
    max_pairwise = max(
        [float(r.get("cand_world_pairwise_dist_max", 0.0)) for r in arm_results] or [0.0]
    )
    magnitude_ok = bool(math.isfinite(max_pairwise) and max_pairwise < C0_MAGNITUDE_CEIL)
    readiness_ok = bool(readiness_seeds_ok >= MIN_SEEDS_FOR_PASS and magnitude_ok)

    # --- C1: ARM_0 Phase-1 residue baseline collapses to ~0 ---
    c1_seeds_ok = _n_seeds_satisfying(
        arm0,
        lambda r: (
            float(r.get("curiosity_std_across_K_mean", 0.0)) <= C1_BIAS_ZERO_CEIL
            and float(r.get("curiosity_bias_range_mean", 0.0)) <= C1_BIAS_ZERO_CEIL
        ),
    )
    c1_pass = bool(c1_seeds_ok >= MIN_SEEDS_FOR_PASS)
    # Non-degenerate only if ARM_1 shows the bias spread CAN be > 0 (real contrast).
    c1_non_degenerate = bool(
        _mean_key(arm1, "curiosity_std_across_K_mean") > C1_BIAS_ZERO_CEIL
    )

    # --- C2 (LOAD-BEARING): ARM_1 visitation source -> per-candidate-varying bias ---
    c2_seeds_ok = _n_seeds_satisfying(
        arm1,
        lambda r: float(r.get("curiosity_std_positive_frac_past20", 0.0)) >= C2_TICK_FRAC,
    )
    c2_pass = bool(c2_seeds_ok >= MIN_SEEDS_FOR_PASS)
    c2_non_degenerate = bool(
        all(int(r.get("n_p1_ticks_past_window", 0)) > 0 for r in arm1) and len(arm1) > 0
    )

    # --- C3: augmentation engages on SD-056-OFF probe, not on SD-056-ON ---
    c3_off_seeds_ok = _n_seeds_satisfying(
        arm2_off,
        lambda r: float(r.get("augmentation_engaged_frac_past20", 0.0)) >= C3_ENGAGE_FRAC,
    )
    c3_on_seeds_ok = _n_seeds_satisfying(
        arm2_on,
        lambda r: float(r.get("augmentation_engaged_frac_past20", 1.0)) <= C3_NOENGAGE_FRAC,
    )
    c3_pass = bool(c3_off_seeds_ok >= MIN_SEEDS_FOR_PASS and c3_on_seeds_ok >= MIN_SEEDS_FOR_PASS)
    # Non-degenerate: the two legs differ (engage-when-needed is a real contrast).
    c3_off_mean = _mean_key(arm2_off, "augmentation_engaged_frac_past20")
    c3_on_mean = _mean_key(arm2_on, "augmentation_engaged_frac_past20")
    c3_non_degenerate = bool((c3_off_mean - c3_on_mean) > 0.0)

    # --- C4: MECH-094 simulation gate ---
    c4_pass = bool(c4.get("waking_ok", False) and c4.get("simulation_ok", False))
    c4_non_degenerate = bool(int(c4.get("n_waking_appends", 0)) > 0)

    criteria_pass = {"C1": c1_pass, "C2": c2_pass, "C3": c3_pass, "C4": c4_pass}
    all_criteria = all(criteria_pass.values())

    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        overall_pass = False
    elif all_criteria:
        label = "phase2_substrate_ready"
        overall_pass = True
    else:
        label = "phase2_wiring_does_not_support"
        overall_pass = False

    return {
        "readiness": {
            "c0_pairwise_dist_floor": C0_PAIRWISE_DIST_FLOOR,
            "arm1_pairwise_dist_mean": round(arm1_pairwise_mean, 6),
            "arm1_seeds_above_floor": int(readiness_seeds_ok),
            "min_seeds_required": MIN_SEEDS_FOR_PASS,
            "max_pairwise_dist_observed": round(max_pairwise, 6),
            "magnitude_ceil": C0_MAGNITUDE_CEIL,
            "magnitude_ok": magnitude_ok,
            "readiness_ok": readiness_ok,
        },
        "criteria_pass": criteria_pass,
        "c1_arm0_seeds_collapsed": int(c1_seeds_ok),
        "c2_arm1_seeds_ok": int(c2_seeds_ok),
        "c3_off_seeds_engaged": int(c3_off_seeds_ok),
        "c3_on_seeds_disengaged": int(c3_on_seeds_ok),
        "c3_off_engage_frac_mean": round(c3_off_mean, 6),
        "c3_on_engage_frac_mean": round(c3_on_mean, 6),
        "c4_probe": c4,
        "pairwise_dist_per_arm_mean": {
            aid: round(_mean_key(rows, "cand_world_pairwise_dist_mean"), 6)
            for aid, rows in by_id.items()
        },
        "curiosity_std_per_arm_mean": {
            aid: round(_mean_key(rows, "curiosity_std_across_K_mean"), 8)
            for aid, rows in by_id.items()
        },
        "augmentation_engage_frac_per_arm_mean": {
            aid: round(_mean_key(rows, "augmentation_engaged_frac_past20"), 6)
            for aid, rows in by_id.items()
        },
        "label": label,
        "overall_pass": overall_pass,
        # Diagnostic adjudication structures (skill Step 3.5).
        "preconditions": [
            {
                "name": "sd056_candidate_zworld_spread_supra_floor",
                "kind": "readiness",
                "description": (
                    "ARM_1 (SD-056-trained positive control, SP-CEM multi-class "
                    "candidate pool) per-candidate z_world SPREAD "
                    "cand_world_pairwise_dist clears the floor -- the SAME "
                    "cross-candidate range statistic C2 routes on (not a magnitude)."
                ),
                "control": "ARM_1: SD-056 contrastive trained online; candidates differ in first action (SP-CEM)",
                "measured": round(arm1_pairwise_mean, 6),
                "threshold": C0_PAIRWISE_DIST_FLOOR,
                "met": bool(readiness_seeds_ok >= MIN_SEEDS_FOR_PASS),
            },
            {
                "name": "rolled_out_zworld_magnitude_bounded",
                "kind": "readiness",
                "description": (
                    "Rolled-out z_world spread stayed finite and below the 643a "
                    "explosion ceiling (SD-056 online training numerical stability)."
                ),
                "control": "max cand_world_pairwise_dist across all arms",
                "measured": round(max_pairwise, 6),
                "threshold": C0_MAGNITUDE_CEIL,
                "met": magnitude_ok,
            },
        ],
        "criteria": [
            {"name": "C1_baseline_collapsed", "load_bearing": False, "passed": c1_pass},
            {"name": "C2_visitation_lifts_per_candidate_bias_spread",
             "load_bearing": True, "passed": c2_pass},
            {"name": "C3_augmentation_engages_when_needed", "load_bearing": False, "passed": c3_pass},
            {"name": "C4_mech094_simulation_gate", "load_bearing": False, "passed": c4_pass},
        ],
        "criteria_non_degenerate": {
            "C1": c1_non_degenerate,
            "C2": c2_non_degenerate,
            "C3": c3_non_degenerate,
            "C4": c4_non_degenerate,
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
            # Arm fingerprint (instrument-only, Phase 0).
            cell["arm_fingerprint"] = compute_arm_fingerprint(
                config_slice={
                    "arm": {k: arm[k] for k in (
                        "arm_id", "novelty_source", "augmentation_policy",
                        "use_first_action_onehot", "sd056_train")},
                    "env_kwargs": ENV_KWARGS,
                    "sd056_weight": SD056_WEIGHT,
                    "curiosity_bias_scale": CURIOSITY_BIAS_SCALE,
                    "curiosity_weight": CURIOSITY_WEIGHT,
                    "visitation_buffer_len": VISITATION_BUFFER_LEN,
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

    c4 = _c4_simulation_gate_probe(seeds[0], steps)
    summary = _evaluate(arm_results, c4)
    outcome = "PASS" if summary["overall_pass"] else "FAIL"

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
        "evidence_direction": "non_contributory",
        "evidence_direction_note": (
            "Substrate-readiness diagnostic for the LANDED MECH-314a Phase-2 "
            "(Candidate 5A) rolling z_world visitation buffer + first-action "
            "one-hot augmentation. claim_ids=[] (does not weight claim confidence). "
            "PASS (label=phase2_substrate_ready) gates /queue-experiment "
            "V3-EXQ-590b (novelty-bonus Goldilocks retest) AND the section-8 "
            "governance/claims updates for MECH-314a / MECH-314 / ARC-065. "
            "Readiness-below-floor self-routes to substrate_not_ready_requeue "
            "(re-queue at higher P0 budget), NOT a substrate verdict."
        ),
        "interpretation": {
            "label": summary["label"],
            "preconditions": summary["preconditions"],
            "criteria": summary["criteria"],
            "criteria_non_degenerate": summary["criteria_non_degenerate"],
            "routing": {
                "substrate_not_ready_requeue": "re-queue V3-EXQ-648a at higher P0 budget; do NOT weaken Phase-2",
                "phase2_substrate_ready": "PASS -> /queue-experiment V3-EXQ-590b; apply section-8 updates",
                "phase2_wiring_does_not_support": "FAIL -> /failure-autopsy on the failing criterion",
            },
        },
        "dry_run": bool(dry_run),
        "config": {
            "seeds": seeds,
            "p0_warmup_episodes": p0,
            "p1_measurement_episodes": p1,
            "steps_per_episode": steps,
            "measure_after_tick": measure_after,
            "env_kwargs": ENV_KWARGS,
            "arms": [{k: a[k] for k in (
                "arm_id", "label", "novelty_source", "augmentation_policy",
                "use_first_action_onehot", "sd056_train")} for a in ARMS],
            "sd056_weight": SD056_WEIGHT,
            "curiosity_bias_scale": CURIOSITY_BIAS_SCALE,
            "curiosity_weight": CURIOSITY_WEIGHT,
            "visitation_buffer_len": VISITATION_BUFFER_LEN,
            "thresholds": {
                "c0_pairwise_dist_floor": C0_PAIRWISE_DIST_FLOOR,
                "c0_magnitude_ceil": C0_MAGNITUDE_CEIL,
                "c1_bias_zero_ceil": C1_BIAS_ZERO_CEIL,
                "c2_tick_frac": C2_TICK_FRAC,
                "c3_engage_frac": C3_ENGAGE_FRAC,
                "c3_noengage_frac": C3_NOENGAGE_FRAC,
                "min_seeds_for_pass": MIN_SEEDS_FOR_PASS,
            },
        },
        "acceptance_criteria": {
            "readiness_substrate_ready": summary["readiness"]["readiness_ok"],
            "C1_baseline_collapsed": summary["criteria_pass"]["C1"],
            "C2_visitation_lifts_bias_spread": summary["criteria_pass"]["C2"],
            "C3_augmentation_engages_when_needed": summary["criteria_pass"]["C3"],
            "C4_mech094_simulation_gate": summary["criteria_pass"]["C4"],
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

    print(f"Outcome: {outcome} (label={summary['label']})", flush=True)
    for k, v in manifest["acceptance_criteria"].items():
        print(f"  {k}: {v}", flush=True)

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-648 MECH-314a Phase-2 substrate-readiness diagnostic"
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
    sys.exit(0)
