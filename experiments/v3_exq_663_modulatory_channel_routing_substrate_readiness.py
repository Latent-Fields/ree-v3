"""V3-EXQ-663: modulatory-bias-selection-authority route-range AMEND
substrate-readiness diagnostic (P0 routed-range gate).

Routed by failure_autopsy_569f-661-654a_2026-06-10 (confirmed; user-adjudicated).
Validates the route-range AMEND landed 2026-06-10 (E3Config.use_modulatory_channel_routing
+ REEConfig.modulatory_channel_route_source + project_channel_range in e3_selector.py).

THE GAP (569f/661/654a, one structural property): a channel whose REPRESENTATION
carries genuine cross-candidate range (569f consumed world-summary spread 0.196;
654a minted rule_state; 661 coherence) still does NOT move committed action, because
that range is flattened by the consuming bias head before it reaches the modulatory
accumulator the authority rescales -- so the authority has nothing to amplify (569f
selected-action entropy bit-identical 0.549141 across e2wf / proposer / matched-noise).
V3-EXQ-643 established "no range -> no authority"; this cluster extends it one link:
the channel range must be ROUTED into the per-candidate modulatory bias the authority
rescales, not merely exist in the representation.

THE FIX UNDER TEST: project_channel_range folds a parameter-free, range-preserving
projection of the channel-under-test's per-candidate representation into the
modulatory accumulator BEFORE the authority's range computation. The P0 readiness
diagnostic modulatory_channel_route_range exposes the routed bias's RAW cross-candidate
range (pre-normalise, pre-rescale) so a retest can assert the modulatory bias ITSELF
carries cross-candidate range derived from the channel under test before any behavioural
falsifier is scored.

DESIGN: 2-arm ablation, matched seeds. BOTH arms run the SHARED bias channels
(lateral_pfc + mech295) ON, the modulatory selection authority ON (gain=0.5),
candidate_summary_source=e2_world_forward, and SD-056 online contrastive (the e2
world-forward divergence the world-summary channel range depends on; rollout-norm
clamp ON per the 643a stability lesson). The ONLY swept axis is
use_modulatory_channel_routing.
  ARM_0_NO_ROUTE   use_modulatory_channel_routing=False  (current behaviour; 569f washout)
  ARM_1_ROUTE_ON   use_modulatory_channel_routing=True, source="cand_world_summary"

ACCEPTANCE (substrate-readiness, claim_ids=[] -- does NOT weight claim confidence):
  READINESS (load-bearing non-vacuity, RANGE statistic): ARM_1
    modulatory_channel_route_range mean > C0_ROUTE_FLOOR on >= MIN_SEEDS_FOR_PASS
    seeds, finite + below the 643a explosion ceiling. This is the P0 gate the autopsy
    demands -- the routed bias carries the channel's cross-candidate range. Below floor
    (under-trained e2 / collapsed candidate pool / amend not wired) self-routes
    substrate_not_ready_requeue, NEVER a substrate verdict.
  C1 PRIMARY (load-bearing, SAME range statistic as readiness): ARM_1
    modulatory_channel_route_active on >= MIN_SEEDS_FOR_PASS seeds AND ARM_0 route
    range ~0 / inactive (the off-arm reproduces the no-routing state). The routed
    range reaches the modulatory accumulator the authority rescales.
  C2 SECONDARY (behavioural reach, NOT load-bearing): committed-action class
    distribution differs ARM_1 vs ARM_0 per matched seed (TV > C2_TV_FLOOR) on
    >= MIN_SEEDS_FOR_PASS seeds -- the routed range now moves the committed argmax,
    breaking the 569f bit-identical-entropy washout. Behavioural movement is
    env/seed/near-tie-dependent, so this corroborates rather than gates (the per-claim
    behavioural EVIDENCE retests are separate follow-on sessions).
  PASS = READINESS AND C1. (C2 reported + required for the strongest reading but does
    not gate the substrate-readiness verdict.)

PASS (label=route_range_substrate_ready) unblocks the per-claim behavioural retests of
ARC-065 / MECH-294 / ARC-062 / MECH-309 / MECH-341 (each a SEPARATE /queue-experiment
session). Those claims stay candidate / v3_pending / pending_retest_after_substrate;
not weakened.

Interpretation grid:
| outcome                              | label                        | next                                                  |
|--------------------------------------|------------------------------|-------------------------------------------------------|
| READINESS+C1 (+C2)                   | route_range_substrate_ready  | /queue-experiment per-claim behavioural retests       |
| readiness leg below floor/non-finite | substrate_not_ready_requeue  | re-queue as 663a at higher P0 (or fix e2/SD-056); do NOT weaken |
| readiness ok but C1 fails            | route_range_inert            | /failure-autopsy on the routing wiring                |

SLEEP DRIVER: K=never (no sleep; waking action-selection diagnostic).

Usage:
  /opt/local/bin/python3 experiments/v3_exq_663_modulatory_channel_routing_substrate_readiness.py --dry-run
"""

import argparse
import json
import math
import random
import sys
from collections import deque
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

EXPERIMENT_TYPE = "v3_exq_663_modulatory_channel_routing_substrate_readiness"
QUEUE_ID = "V3-EXQ-663"
CLAIM_IDS: List[str] = []  # substrate-readiness diagnostic (gates per-claim behavioural retests)
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 60           # SD-056 contrastive warmup (V3-EXQ-649/648a proven budget)
P1_MEASUREMENT_EPISODES = 20
STEPS_PER_EPISODE = 200
MEASURE_AFTER_TICK = 20

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_STEPS = 30
DRY_RUN_MEASURE_AFTER_TICK = 2

# Acceptance thresholds (pre-registered).
C0_ROUTE_FLOOR = 0.01             # readiness: ARM_1 routed-bias cross-candidate RANGE floor
C0_MAGNITUDE_CEIL = 1.0e6         # readiness: routed range bounded (643a explosion guard)
C1_OFF_INACTIVE_CEIL = 1e-9       # C1: ARM_0 routed range ~0 (routing off -> diagnostic stays 0.0)
C2_TV_FLOOR = 0.02               # C2 (secondary): committed-class distribution TV ARM_1 vs ARM_0
MIN_SEEDS_FOR_PASS = 2            # of 3

# SD-056 online contrastive training (mirror V3-EXQ-649 harness).
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
E2_TRAIN_EVERY_K_TICKS = 1
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0

# HARM-FREE env: SP-CEM + resources give action-divergent candidates for SD-056 to
# train z_world divergence on; no hazards needed for the routing-readiness test.
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

ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": "ARM_0_NO_ROUTE",
        "label": "channel_routing_off_569f_washout_baseline",
        "use_modulatory_channel_routing": False,
    },
    {
        "arm_id": "ARM_1_ROUTE_ON",
        "label": "channel_routing_on_cand_world_summary",
        "use_modulatory_channel_routing": True,
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """Main-path SP-CEM stack with the SHARED E3-side bias channels (lateral_pfc +
    mech295) ON, the modulatory selection authority ON (gain=0.5),
    candidate_summary_source=e2_world_forward, and SD-056 contrastive trained online
    (the e2.world_forward divergence the routed world-summary channel range depends
    on; rollout-norm clamp ON per the 643a stability lesson). The ONLY swept axis is
    use_modulatory_channel_routing."""
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
        # SHARED E3-side bias channels (consume cand_world_summaries) ON in BOTH arms
        use_lateral_pfc_analog=True,
        use_mech295_liking_bridge=True,
        # Other policy-layer regulators OFF (channel routing is the swept axis)
        use_structured_curiosity=False,
        use_e3_score_diversity=False,
        use_noise_floor=False,
        use_tonic_vigor=False,
        use_dacc=False,
        use_ofc_analog=False,
        use_gated_policy=False,
        # SD-056 substrate trained online on every arm (e2.world_forward divergence)
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=2.0,
        # ARC-065 GAP-A shared channel source (the world-summary representation routed)
        candidate_summary_source="e2_world_forward",
        # modulatory-bias-selection-authority (the gate the routed range reaches)
        use_modulatory_selection_authority=True,
        modulatory_authority_gain=0.5,
        # --- route-range AMEND: the swept axis ---
        use_modulatory_channel_routing=bool(arm["use_modulatory_channel_routing"]),
        modulatory_channel_route_source="cand_world_summary",
        modulatory_channel_route_min_range_floor=1e-6,
        modulatory_channel_route_weight=1.0,
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs(d: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    h = d.get(key)
    if h is None:
        return None
    return h.float().unsqueeze(0) if h.dim() == 1 else h.float()


def _entropy_from_counts(counts: Dict[int, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p)
    return ent


def _tv_distance(counts_a: Dict[int, int], counts_b: Dict[int, int]) -> float:
    """Total-variation distance between two committed-class histograms (0.5 * L1 of
    the normalised distributions over the union of classes)."""
    ta = sum(counts_a.values())
    tb = sum(counts_b.values())
    if ta <= 0 or tb <= 0:
        return 0.0
    classes = set(counts_a) | set(counts_b)
    tv = 0.0
    for cls in classes:
        pa = counts_a.get(cls, 0) / ta
        pb = counts_b.get(cls, 0) / tb
        tv += abs(pa - pb)
    return 0.5 * tv


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
    reset_all_rng(seed)

    env = _make_env(seed)
    agent = _make_agent(env, arm)
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)

    transition_buffer: Deque[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)

    total_train_eps = p0_episodes + p1_episodes

    route_ranges: List[float] = []
    route_range_max = 0.0
    route_active_ticks = 0
    committed_class_counts: Dict[int, int] = {}
    n_p1_ticks_past_window = 0
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

            # --- route-range readout (P0 gate) + committed-action readout ---
            past_window = is_p1 and tick_in_ep >= measure_after_tick
            if past_window and candidates and len(candidates) >= 2:
                diag = agent.e3.last_score_diagnostics or {}
                rr = diag.get("modulatory_channel_route_range")
                if rr is not None and math.isfinite(float(rr)):
                    route_ranges.append(float(rr))
                    route_range_max = max(route_range_max, float(rr))
                if bool(diag.get("modulatory_channel_route_active", False)):
                    route_active_ticks += 1
                cls = int(action.argmax(dim=-1).item())
                committed_class_counts[cls] = committed_class_counts.get(cls, 0) + 1
                n_p1_ticks_past_window += 1

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

        if (ep + 1) % 10 == 0 or (ep + 1) == total_train_eps:
            print(
                f"  [train] arm={arm['arm_id']} seed={seed} phase={phase_label} "
                f"ep {ep + 1}/{total_train_eps}",
                flush=True,
            )

    def _mean(xs: List[float], default: float = 0.0) -> float:
        return float(sum(xs) / len(xs)) if xs else default

    route_active_frac = (
        float(route_active_ticks) / float(n_p1_ticks_past_window)
        if n_p1_ticks_past_window > 0 else 0.0
    )

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": int(seed),
        "use_modulatory_channel_routing": bool(arm["use_modulatory_channel_routing"]),
        "n_p1_ticks_past_window": int(n_p1_ticks_past_window),
        "n_contrastive_steps": int(n_contrastive_steps),
        "error_note": error_note,
        # P0 gate: RAW cross-candidate range of the routed channel bias (pre-rescale).
        "route_range_mean": round(_mean(route_ranges), 6),
        "route_range_max": round(route_range_max, 6),
        "route_active_frac": round(route_active_frac, 4),
        # Committed-action readout (the 569f selected-action axis).
        "committed_class_counts": {str(k): int(v) for k, v in committed_class_counts.items()},
        "committed_class_entropy": round(_entropy_from_counts(committed_class_counts), 6),
        "n_committed_classes": int(len(committed_class_counts)),
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


def _counts(r: Dict[str, Any]) -> Dict[int, int]:
    return {int(k): int(v) for k, v in (r.get("committed_class_counts") or {}).items()}


def _evaluate(arm_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    arm0 = _arm_rows(arm_results, "ARM_0_NO_ROUTE")
    arm1 = _arm_rows(arm_results, "ARM_1_ROUTE_ON")
    arm0_by_seed = {r["seed"]: r for r in arm0}

    # READINESS (load-bearing non-vacuity, RANGE statistic): ARM_1 routed range > floor.
    readiness_seeds_ok = _n_seeds(
        arm1, lambda r: float(r.get("route_range_mean", 0.0)) > C0_ROUTE_FLOOR
    )
    arm1_route_mean = _mean_key(arm1, "route_range_mean")
    max_route = max(
        [float(r.get("route_range_max", 0.0)) for r in arm_results] or [0.0]
    )
    magnitude_ok = bool(math.isfinite(max_route) and max_route < C0_MAGNITUDE_CEIL)
    readiness_ok = bool(readiness_seeds_ok >= MIN_SEEDS_FOR_PASS and magnitude_ok)

    # C1 PRIMARY (load-bearing, SAME range statistic): ARM_1 routing active on >=N seeds
    # AND ARM_0 routed range ~0 (routing off). The routed range reaches the accumulator.
    c1_on_active_seeds = _n_seeds(arm1, lambda r: float(r.get("route_active_frac", 0.0)) > 0.5)
    c1_off_inactive_seeds = _n_seeds(
        arm0, lambda r: float(r.get("route_range_mean", 0.0)) <= C1_OFF_INACTIVE_CEIL
    )
    c1_pass = bool(
        c1_on_active_seeds >= MIN_SEEDS_FOR_PASS
        and c1_off_inactive_seeds >= MIN_SEEDS_FOR_PASS
    )
    c1_non_degenerate = bool(arm1_route_mean > C0_ROUTE_FLOOR)  # real routed range exists

    # C2 SECONDARY (behavioural reach, NOT load-bearing): committed-class TV per seed.
    def _c2(r1: Dict[str, Any]) -> bool:
        r0 = arm0_by_seed.get(r1["seed"])
        if r0 is None:
            return False
        return _tv_distance(_counts(r1), _counts(r0)) > C2_TV_FLOOR
    c2_seeds_ok = _n_seeds(arm1, _c2)
    c2_pass = bool(c2_seeds_ok >= MIN_SEEDS_FOR_PASS)
    # TV per seed for the manifest.
    tv_per_seed = {}
    for r1 in arm1:
        r0 = arm0_by_seed.get(r1["seed"])
        if r0 is not None:
            tv_per_seed[str(r1["seed"])] = round(_tv_distance(_counts(r1), _counts(r0)), 4)
    c2_non_degenerate = bool(
        len(arm1) > 0 and all(int(r.get("n_p1_ticks_past_window", 0)) > 0 for r in arm1)
        and len(arm0) > 0 and all(int(r.get("n_p1_ticks_past_window", 0)) > 0 for r in arm0)
    )

    criteria_pass = {"C1": c1_pass, "C2": c2_pass}
    # PASS gated on READINESS + C1 (the substrate's P0 range gate). C2 corroborates.
    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        overall_pass = False
    elif c1_pass:
        label = "route_range_substrate_ready"
        overall_pass = True
    else:
        label = "route_range_inert"
        overall_pass = False

    return {
        "readiness": {
            "c0_route_floor": C0_ROUTE_FLOOR,
            "arm1_route_range_mean": round(arm1_route_mean, 6),
            "arm1_seeds_above_floor": int(readiness_seeds_ok),
            "min_seeds_required": MIN_SEEDS_FOR_PASS,
            "max_route_range_observed": round(max_route, 6),
            "magnitude_ceil": C0_MAGNITUDE_CEIL,
            "magnitude_ok": magnitude_ok,
            "readiness_ok": readiness_ok,
        },
        "criteria_pass": criteria_pass,
        "c1_arm1_seeds_route_active": int(c1_on_active_seeds),
        "c1_arm0_seeds_route_inactive": int(c1_off_inactive_seeds),
        "c2_arm1_seeds_committed_tv_above_floor": int(c2_seeds_ok),
        "c2_committed_tv_per_seed": tv_per_seed,
        "route_range_per_arm_mean": {
            "ARM_0_NO_ROUTE": round(_mean_key(arm0, "route_range_mean"), 6),
            "ARM_1_ROUTE_ON": round(_mean_key(arm1, "route_range_mean"), 6),
        },
        "route_active_frac_per_arm_mean": {
            "ARM_0_NO_ROUTE": round(_mean_key(arm0, "route_active_frac"), 4),
            "ARM_1_ROUTE_ON": round(_mean_key(arm1, "route_active_frac"), 4),
        },
        "committed_class_entropy_per_arm_mean": {
            "ARM_0_NO_ROUTE": round(_mean_key(arm0, "committed_class_entropy"), 6),
            "ARM_1_ROUTE_ON": round(_mean_key(arm1, "committed_class_entropy"), 6),
        },
        "label": label,
        "overall_pass": overall_pass,
        # Diagnostic adjudication structures (skill Step 3.5).
        "preconditions": [
            {
                "name": "arm1_routed_bias_range_supra_floor",
                "kind": "readiness",
                "description": (
                    "ARM_1 (routing ON, cand_world_summary source) routed-bias "
                    "cross-candidate RANGE (modulatory_channel_route_range -- the "
                    "RAW range the P0 gate keys on, pre-normalise/pre-rescale) clears "
                    "the floor. This is a RANGE statistic, the SAME one C1 routes on, "
                    "NOT a magnitude. Below floor => under-trained e2 / collapsed "
                    "candidate pool / amend not wired => substrate_not_ready_requeue, "
                    "never a substrate verdict."
                ),
                "control": "ARM_1: SD-056 contrastive trained online; SP-CEM multi-class candidates; candidate_summary_source=e2_world_forward; routing ON",
                "measured": round(arm1_route_mean, 6),
                "threshold": C0_ROUTE_FLOOR,
                "met": bool(readiness_seeds_ok >= MIN_SEEDS_FOR_PASS),
            },
            {
                "name": "routed_range_bounded",
                "kind": "readiness",
                "description": (
                    "Routed-bias range stayed finite and below the 643a explosion "
                    "ceiling (SD-056 online training numerical stability; rollout-norm "
                    "clamp ON)."
                ),
                "control": "max route_range across all arms",
                "measured": round(max_route, 6),
                "threshold": C0_MAGNITUDE_CEIL,
                "direction": "upper",
                "met": magnitude_ok,
            },
        ],
        "criteria": [
            {"name": "C1_routed_range_reaches_accumulator_on_active_off_inactive",
             "load_bearing": True, "passed": c1_pass},
            {"name": "C2_committed_class_distribution_moves_on_vs_off",
             "load_bearing": False, "passed": c2_pass},
        ],
        "criteria_non_degenerate": {"C1": c1_non_degenerate, "C2": c2_non_degenerate},
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
                    "arm": {k: arm[k] for k in ("arm_id", "use_modulatory_channel_routing")},
                    "env_kwargs": ENV_KWARGS,
                    "sd056_weight": SD056_WEIGHT,
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
            "modulatory-bias-selection-authority route-range AMEND substrate-readiness "
            "diagnostic (E3Config.use_modulatory_channel_routing + project_channel_range; "
            "landed 2026-06-10). Routed by failure_autopsy_569f-661-654a_2026-06-10. "
            "claim_ids=[] (does NOT weight claim confidence). PASS "
            "(label=route_range_substrate_ready) confirms the P0 routed-range gate: the "
            "channel-under-test's cross-candidate range is now routed into the modulatory "
            "bias the authority rescales (the 569f/661/654a gap). It UNBLOCKS the per-claim "
            "behavioural retests of ARC-065 / MECH-294 / ARC-062 / MECH-309 / MECH-341 "
            "(each a SEPARATE /queue-experiment session). Readiness-below-floor self-routes "
            "substrate_not_ready_requeue, NOT a substrate verdict. Those claims stay "
            "candidate / v3_pending / pending_retest_after_substrate; not weakened."
        ),
        "interpretation": {
            "label": summary["label"],
            "preconditions": summary["preconditions"],
            "criteria": summary["criteria"],
            "criteria_non_degenerate": summary["criteria_non_degenerate"],
            "routing": {
                "route_range_substrate_ready": "PASS -> /queue-experiment per-claim behavioural retests (ARC-065/MECH-294/ARC-062/MECH-309/MECH-341), each separate",
                "substrate_not_ready_requeue": "re-queue as V3-EXQ-663a at higher P0 budget (or fix e2/SD-056 wiring); do NOT weaken",
                "route_range_inert": "FAIL -> /failure-autopsy on the routing wiring",
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
            "arms": [{k: a[k] for k in ("arm_id", "label", "use_modulatory_channel_routing")} for a in ARMS],
            "sd056_weight": SD056_WEIGHT,
            "modulatory_channel_route_source": "cand_world_summary",
            "modulatory_authority_gain": 0.5,
            "thresholds": {
                "c0_route_floor": C0_ROUTE_FLOOR,
                "c0_magnitude_ceil": C0_MAGNITUDE_CEIL,
                "c1_off_inactive_ceil": C1_OFF_INACTIVE_CEIL,
                "c2_tv_floor": C2_TV_FLOOR,
                "min_seeds_for_pass": MIN_SEEDS_FOR_PASS,
            },
        },
        "acceptance_criteria": {
            "readiness_substrate_ready": summary["readiness"]["readiness_ok"],
            "C1_routed_range_reaches_accumulator": summary["criteria_pass"]["C1"],
            "C2_committed_distribution_moves": summary["criteria_pass"]["C2"],
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

    print(f"Outcome: {outcome} (label={summary['label']})", flush=True)
    for k, v in manifest["acceptance_criteria"].items():
        print(f"  {k}: {v}", flush=True)

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-663 modulatory-bias-selection-authority route-range substrate-readiness diagnostic"
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
