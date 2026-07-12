"""V3-EXQ-569f: ARC-065 GAP-A R1.b FP-2 matched-entropy action-contrastive
behavioural falsifier on the V3-EXQ-649-validated shared-channel stack.

behavioral_diversity_isolation:GAP-A. Successor to the 569 matched-entropy
FP-2 lineage (frontier 569e). supersedes V3-EXQ-569d -- the prior FP-2 falsifier
(PASS 2026-05-31) ran PRE-649, when the SHARED cand_world_summaries consumed by
the E3-side bias channels was built from the COLLAPSED proposer first-step
z_world (trajectory.world_states[:,0,:]), not from the SD-056-trained
action-conditional e2.world_forward predictions. The 614e autopsy relocated the
committed-class-diversity bottleneck to that consumption channel (GAP-A); the fix
(REEConfig.candidate_summary_source="e2_world_forward", ree-v3 2026-06-07) was
substrate-readiness VALIDATED by V3-EXQ-649 PASS (ARM_1 consumed-summary spread
0.090 >= 0.05 floor; e2_world_forward >> proposer).

This experiment is the behavioural FP-2 falsifier that CONSUMES that readiness:
does routing the shared cand_world_summaries through e2.world_forward (the 649
fix) translate the preserved per-candidate spread into SELECTED-ACTION diversity
STRICTLY ABOVE both the collapsed-proposer baseline and a matched-entropy
softmax-temperature noise control (the FP-2 / R1.b decision rule)?

DESIGN: 3-arm single-variable design, matched seeds. SP-CEM (Layer A) on; the
SHARED E3-side bias channels (lateral_pfc + mech295) ON; SD-056 online
contrastive trained on EVERY arm (the e2.world_forward divergence the GAP-A fix
consumes depends on it) with the rollout-norm clamp ON (643a float32-cancellation
lesson -- DVs do NOT depend on e3_top2_class_gap, which 569d saw explode to
~1e34 without the clamp). Each arm changes ONE thing relative to ARM_0:
  ARM_0_PROPOSER       candidate_summary_source="proposer",        temperature=1.0  (pre-649 consumption baseline)
  ARM_1_E2WF           candidate_summary_source="e2_world_forward", temperature=1.0  (the 649 GAP-A fix; under test)
  ARM_2_MATCHED_NOISE  candidate_summary_source="proposer",        temperature=2.5  (FP-2 matched-entropy noise control)

ARM_1 vs ARM_0 isolates the CONSUMPTION channel at matched temperature. ARM_1 vs
ARM_2 isolates structured-vs-noise: a higher-temperature proposer arm produces
selection entropy from pure softmax noise; the falsifier requires the structured
e2_world_forward channel to exceed it.

ACCEPTANCE (evidence, claim_ids=[ARC-065]; plan decision rule R1.a/R1.b):
  READINESS (load-bearing non-vacuity): ARM_1 consumed_summary_pairwise_dist mean
    > C0_SPREAD_FLOOR on >= MIN_SEEDS_FOR_PASS seeds, finite + below the 643a
    explosion ceiling. Below floor (under-trained e2 / amend not wired) self-routes
    substrate_not_ready_requeue -> evidence_direction non_contributory, NEVER a
    weakens.
  C1 substrate-operative (load-bearing non-vacuity): ARM_1 cand_world_pairwise_dist
    (e2.world_forward prediction spread) > C1_PAIRWISE_DIST_FLOOR on
    >= MIN_SEEDS_FOR_PASS seeds -- confirms SD-056 trained the action-conditional
    divergence the consumption channel re-sources.
  C_R1B PRIMARY (load-bearing): ARM_1 selected_action_class_entropy STRICTLY ABOVE
    BOTH ARM_2_MATCHED_NOISE AND ARM_0_PROPOSER on the SAME seed, on
    >= MIN_SEEDS_FOR_PASS seeds, AND ARM_1 mean > C3_SELECTED_ENTROPY_FLOOR.
  PASS = READINESS AND C1 AND C_R1B -> R1.b fires; ARC-065 GAP-A theory-1
    (SP-CEM child, shared-channel consumption) is a real diversity contributor;
    evidence_direction=supports.

Interpretation grid (plan R1.a/R1.b):
| outcome                                   | label                          | evidence_direction | next                                            |
|-------------------------------------------|--------------------------------|--------------------|-------------------------------------------------|
| READINESS+C1+C_R1B                        | r1b_diversity_above_matched_noise | supports        | R1.b cleared; GAP-A theory-1 confirmed          |
| readiness below floor / non-finite        | substrate_not_ready_requeue    | non_contributory   | re-queue as 569g at higher P0; do NOT weaken    |
| readiness+C1 ok, C_R1B fail (no strict lift) | r1a_entropy_only_artefact   | weakens            | R1.a: theory-1 not load-bearing alone; substrate revisit |
| readiness ok, C1 fail (e2 not divergent)  | substrate_not_ready_requeue    | non_contributory   | SD-056 under-trained; re-queue at higher P0     |

claim_ids=[ARC-065] ONLY. The MECH-341 committed-class GAP-B retest is a separate
concurrent experiment (V3-EXQ-660); tagging MECH-341 here would contaminate its
evidence record (claim_ids-accuracy rule: err toward fewer tags).

Usage:
  /opt/local/bin/python3 experiments/v3_exq_569f_gapa_e2wf_matched_entropy_falsifier.py --dry-run
"""

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

EXPERIMENT_TYPE = "v3_exq_569f_gapa_e2wf_matched_entropy_falsifier"
QUEUE_ID = "V3-EXQ-569f"
CLAIM_IDS: List[str] = ["ARC-065"]  # GAP-A theory-1 (SP-CEM child / shared-channel consumption)
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 60           # SD-056 contrastive warmup (V3-EXQ-649/648a proven budget)
P1_MEASUREMENT_EPISODES = 20
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_STEPS = 30

# Acceptance thresholds (pre-registered).
C0_SPREAD_FLOOR = 0.05            # readiness: ARM_1 consumed-summary pairwise spread (the 649 statistic)
C0_MAGNITUDE_CEIL = 1.0e6         # readiness: spreads finite + bounded (643a guard)
C1_PAIRWISE_DIST_FLOOR = 0.03     # C1: ARM_1 e2.world_forward prediction spread (SD-056 trained)
C3_SELECTED_ENTROPY_FLOOR = 0.3   # C_R1B: ARM_1 selected-action class entropy floor
MATCHED_ENTROPY_TEMPERATURE = 2.5
MIN_SEEDS_FOR_PASS = 2            # of 3

# SD-056 online contrastive training (mirror V3-EXQ-649 / 648a harness).
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
E2_TRAIN_EVERY_K_TICKS = 1
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0

# Behavioural-diversity env: SD-054 reef-bipartite hazard layout (matches the
# 569d FP-2 falsifier env exactly -- the env where the monostrategy phenotype
# was measured; reef-vs-forage forces categorically opposite first actions).
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

ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": "ARM_0_PROPOSER",
        "label": "shared_summary_source_proposer_pre649_baseline",
        "candidate_summary_source": "proposer",
        "temperature": 1.0,
    },
    {
        "arm_id": "ARM_1_E2WF",
        "label": "shared_summary_source_e2_world_forward_gapa_fix",
        "candidate_summary_source": "e2_world_forward",
        "temperature": 1.0,
    },
    {
        "arm_id": "ARM_2_MATCHED_NOISE",
        "label": "proposer_matched_entropy_temperature_control",
        "candidate_summary_source": "proposer",
        "temperature": MATCHED_ENTROPY_TEMPERATURE,
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """Main-path SP-CEM stack with the SHARED E3-side bias channels (lateral_pfc +
    mech295) ON and candidate_summary_source set per arm. SD-056 contrastive is
    ENABLED on every arm (the e2.world_forward divergence the GAP-A fix consumes
    depends on it) with the rollout-norm clamp ON (643a stability lesson)."""
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
        # SHARED E3-side bias channels under test (consume cand_world_summaries)
        use_lateral_pfc_analog=True,
        use_mech295_liking_bridge=True,
        # Other policy-layer regulators OFF (candidate_summary_source is the axis)
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
        # --- ARC-065 GAP-A: the swept axis ---
        candidate_summary_source=str(arm["candidate_summary_source"]),
    )
    agent = REEAgent(cfg)
    return agent


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def _trajectory_first_action_class(traj) -> int:
    return int(
        traj.actions[:, 0, :].argmax(dim=-1).detach().reshape(-1)[0].item()
    )


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


def _consumed_summaries(agent: REEAgent, candidates) -> Optional[torch.Tensor]:
    """The per-candidate cand_world_summaries the SHARED bias channels actually
    consume this arm: agent._candidate_world_summaries (e2.world_forward source)
    when candidate_summary_source='e2_world_forward'; else the proposer first-step
    z_world (trajectory.world_states[:,0,:]) -- the exact pre/post-GAP-A contrast."""
    summ = agent._candidate_world_summaries(candidates)
    if summ is not None:
        return summ.detach()
    rows: List[torch.Tensor] = []
    for c in candidates:
        if c.world_states is not None:
            rows.append(c.get_world_state_sequence()[0, 0, :].detach())
        elif agent._current_latent is not None:
            rows.append(agent._current_latent.z_world[0].detach())
        else:
            return None
    if not rows:
        return None
    return torch.stack(rows, dim=0)


def _mean_pairwise_l2(summ: torch.Tensor) -> float:
    """Mean pairwise L2 over the K rows = cand_world_pairwise_dist (=0.0000 under
    monostrategy / collapsed-proposer consumption)."""
    summ = summ.detach()
    K = summ.shape[0]
    if K < 2:
        return 0.0
    total = 0.0
    n = 0
    for i in range(K):
        for j in range(i + 1, K):
            total += float(torch.linalg.vector_norm(summ[i] - summ[j]))
            n += 1
    return total / max(n, 1)


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

    consumed_dists: List[float] = []
    consumed_dist_max = 0.0
    pairwise_dists: List[float] = []
    candidate_unique_per_tick: List[float] = []
    selected_class_counts: Counter = Counter()
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
                # Candidate-pool first-action diversity (substrate-naive readout).
                pre_e3_classes = [_trajectory_first_action_class(t) for t in candidates]
                candidate_unique_per_tick.append(float(len(set(pre_e3_classes))))
                # C1 substrate-operative: e2.world_forward prediction spread.
                actions_K = _first_actions_K(candidates).to(agent.device)
                z0 = latent.z_world.detach()
                with torch.no_grad():
                    pdist = float(agent.e2.cand_world_pairwise_dist(z0, actions_K).item())
                if math.isfinite(pdist):
                    pairwise_dists.append(pdist)
                # READINESS / GAP-A: per-candidate spread of the CONSUMED summaries.
                consumed = _consumed_summaries(agent, candidates)
                if consumed is not None and torch.isfinite(consumed).all():
                    cdist = _mean_pairwise_l2(consumed)
                    if math.isfinite(cdist):
                        consumed_dists.append(cdist)
                        consumed_dist_max = max(consumed_dist_max, cdist)

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

            # C_R1B behavioural DV: committed first-action class diversity.
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

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": int(seed),
        "candidate_summary_source": arm["candidate_summary_source"],
        "temperature": arm_temperature,
        "n_p1_ticks": int(n_p1_ticks),
        "n_contrastive_steps": int(n_contrastive_steps),
        "error_note": error_note,
        # READINESS / GAP-A statistic: per-candidate spread of the CONSUMED summaries.
        "consumed_summary_pairwise_dist_mean": round(_mean(consumed_dists), 6),
        "consumed_summary_pairwise_dist_max": round(consumed_dist_max, 6),
        # C1 substrate-operative: e2.world_forward prediction spread.
        "cand_world_pairwise_dist_mean": round(_mean(pairwise_dists), 6),
        # Candidate-pool first-action diversity (interpretation aid).
        "candidate_unique_first_action_classes_mean": round(_mean(candidate_unique_per_tick), 6),
        "trajectory_class_count_mean": round(_mean(candidate_unique_per_tick), 6),
        # C_R1B behavioural DV.
        "selected_action_class_entropy": round(selected_action_entropy, 6),
        "selected_class_counts": dict(sorted(selected_class_counts.items())),
        "selected_classes_n_unique": int(len(selected_class_counts)),
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
    arm0 = _arm_rows(arm_results, "ARM_0_PROPOSER")
    arm1 = _arm_rows(arm_results, "ARM_1_E2WF")
    arm2 = _arm_rows(arm_results, "ARM_2_MATCHED_NOISE")
    arm0_by_seed = {r["seed"]: r for r in arm0}
    arm2_by_seed = {r["seed"]: r for r in arm2}

    CDIST = "consumed_summary_pairwise_dist_mean"
    PDIST = "cand_world_pairwise_dist_mean"
    SENT = "selected_action_class_entropy"

    # READINESS (load-bearing non-vacuity): ARM_1 consumed spread > floor, bounded.
    arm1_consumed_mean = _mean_key(arm1, CDIST)
    readiness_seeds_ok = _n_seeds(arm1, lambda r: float(r.get(CDIST, 0.0)) > C0_SPREAD_FLOOR)
    max_dist = max(
        [float(r.get("consumed_summary_pairwise_dist_max", 0.0)) for r in arm_results] or [0.0]
    )
    magnitude_ok = bool(math.isfinite(max_dist) and max_dist < C0_MAGNITUDE_CEIL)
    readiness_ok = bool(readiness_seeds_ok >= MIN_SEEDS_FOR_PASS and magnitude_ok)

    # C1 substrate-operative (load-bearing non-vacuity): ARM_1 e2.world_forward
    # prediction spread > floor (SD-056 trained the action-conditional divergence).
    arm1_pdist_mean = _mean_key(arm1, PDIST)
    c1_seeds_ok = _n_seeds(arm1, lambda r: float(r.get(PDIST, 0.0)) > C1_PAIRWISE_DIST_FLOOR)
    c1_pass = bool(c1_seeds_ok >= MIN_SEEDS_FOR_PASS)

    # C_R1B PRIMARY (load-bearing, R1.b): ARM_1 selected-action class entropy
    # STRICTLY ABOVE both ARM_2_MATCHED_NOISE and ARM_0_PROPOSER on the same seed.
    def _r1b(r1: Dict[str, Any]) -> bool:
        r0 = arm0_by_seed.get(r1["seed"])
        r2 = arm2_by_seed.get(r1["seed"])
        if r0 is None or r2 is None:
            return False
        e1 = float(r1.get(SENT, 0.0))
        return e1 > float(r0.get(SENT, 0.0)) and e1 > float(r2.get(SENT, 0.0))
    r1b_seeds_ok = _n_seeds(arm1, _r1b)
    arm1_sel_mean = _mean_key(arm1, SENT)
    r1b_floor_ok = bool(arm1_sel_mean > C3_SELECTED_ENTROPY_FLOOR)
    r1b_pass = bool(r1b_seeds_ok >= MIN_SEEDS_FOR_PASS and r1b_floor_ok)

    # Non-degeneracy: every measured arm produced P1 ticks.
    all_arms = [arm0, arm1, arm2]
    non_degenerate = bool(
        all(len(a) > 0 for a in all_arms)
        and all(int(r.get("n_p1_ticks", 0)) > 0 for a in all_arms for r in a)
    )

    if not readiness_ok or not c1_pass:
        label = "substrate_not_ready_requeue"
        overall_pass = False
        evidence_direction = "non_contributory"
    elif r1b_pass:
        label = "r1b_diversity_above_matched_noise"
        overall_pass = True
        evidence_direction = "supports"
    else:
        label = "r1a_entropy_only_artefact"
        overall_pass = False
        evidence_direction = "weakens"

    return {
        "readiness": {
            "c0_spread_floor": C0_SPREAD_FLOOR,
            "arm1_consumed_spread_mean": round(arm1_consumed_mean, 6),
            "arm1_seeds_above_floor": int(readiness_seeds_ok),
            "min_seeds_required": MIN_SEEDS_FOR_PASS,
            "max_consumed_spread_observed": round(max_dist, 6),
            "magnitude_ceil": C0_MAGNITUDE_CEIL,
            "magnitude_ok": magnitude_ok,
            "readiness_ok": readiness_ok,
        },
        "criteria_pass": {"C1": c1_pass, "C_R1B": r1b_pass},
        "c1_arm1_seeds_e2_divergent": int(c1_seeds_ok),
        "c1_arm1_pairwise_dist_mean": round(arm1_pdist_mean, 6),
        "r1b_arm1_seeds_strict_above_both": int(r1b_seeds_ok),
        "r1b_floor": C3_SELECTED_ENTROPY_FLOOR,
        "r1b_floor_ok": r1b_floor_ok,
        "consumed_spread_per_arm_mean": {
            "ARM_0_PROPOSER": round(_mean_key(arm0, CDIST), 6),
            "ARM_1_E2WF": round(_mean_key(arm1, CDIST), 6),
            "ARM_2_MATCHED_NOISE": round(_mean_key(arm2, CDIST), 6),
        },
        "selected_action_entropy_per_arm_mean": {
            "ARM_0_PROPOSER": round(_mean_key(arm0, SENT), 6),
            "ARM_1_E2WF": round(arm1_sel_mean, 6),
            "ARM_2_MATCHED_NOISE": round(_mean_key(arm2, SENT), 6),
        },
        "trajectory_class_count_per_arm_mean": {
            "ARM_0_PROPOSER": round(_mean_key(arm0, "trajectory_class_count_mean"), 6),
            "ARM_1_E2WF": round(_mean_key(arm1, "trajectory_class_count_mean"), 6),
            "ARM_2_MATCHED_NOISE": round(_mean_key(arm2, "trajectory_class_count_mean"), 6),
        },
        "label": label,
        "evidence_direction": evidence_direction,
        "overall_pass": overall_pass,
        # Diagnostic-adjudication structures (the readiness self-route is falsifiable).
        "preconditions": [
            {
                "name": "arm1_consumed_summary_spread_supra_floor",
                "kind": "readiness",
                "description": (
                    "ARM_1 (e2_world_forward source) per-candidate SPREAD of the "
                    "CONSUMED cand_world_summaries clears the floor -- a "
                    "cross-candidate RANGE statistic (the 649 readiness statistic), "
                    "NOT a magnitude. Below floor => under-trained e2 / amend not "
                    "wired => substrate_not_ready_requeue, never a weakens."
                ),
                "control": "ARM_1: SD-056 contrastive trained online; SP-CEM multi-class candidates; candidate_summary_source=e2_world_forward",
                "measured": round(arm1_consumed_mean, 6),
                "threshold": C0_SPREAD_FLOOR,
                "met": bool(readiness_seeds_ok >= MIN_SEEDS_FOR_PASS),
            },
            {
                "name": "arm1_e2_world_forward_prediction_spread_supra_floor",
                "kind": "readiness",
                "description": (
                    "ARM_1 e2.world_forward(z0, a_i) per-candidate prediction spread "
                    "(cand_world_pairwise_dist) clears the floor -- confirms SD-056 "
                    "trained the action-conditional divergence the consumption channel "
                    "re-sources. Below floor => SD-056 under-trained => "
                    "substrate_not_ready_requeue."
                ),
                "control": "ARM_1: agent.e2.cand_world_pairwise_dist on SP-CEM candidates",
                "measured": round(arm1_pdist_mean, 6),
                "threshold": C1_PAIRWISE_DIST_FLOOR,
                "met": c1_pass,
            },
            {
                "name": "consumed_spread_bounded",
                "kind": "readiness",
                "description": (
                    "Consumed-summary spread stayed finite and below the 643a "
                    "explosion ceiling (SD-056 online numerical stability; "
                    "rollout-norm clamp ON). Upper-bound check: measured << threshold "
                    "means PASS."
                ),
                "control": "max consumed_summary_pairwise_dist across all arms",
                "measured": round(max_dist, 6),
                "threshold": C0_MAGNITUDE_CEIL,
                "direction": "upper",
                "met": magnitude_ok,
            },
        ],
        "criteria": [
            {"name": "C1_arm1_e2_world_forward_divergent", "load_bearing": True, "passed": c1_pass},
            {"name": "C_R1B_selected_entropy_strict_above_matched_noise_and_proposer",
             "load_bearing": True, "passed": r1b_pass},
        ],
        "criteria_non_degenerate": {"C1": non_degenerate, "C_R1B": non_degenerate},
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
                    "arm": {k: arm[k] for k in ("arm_id", "candidate_summary_source", "temperature")},
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
    evidence_direction = summary["evidence_direction"]

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
        "supersedes": "V3-EXQ-569d",
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": {"ARC-065": evidence_direction},
        "evidence_direction_note": (
            "ARC-065 GAP-A R1.b FP-2 matched-entropy behavioural falsifier on the "
            "V3-EXQ-649-validated shared-channel stack (candidate_summary_source="
            "e2_world_forward). supersedes V3-EXQ-569d (pre-649 collapsed-proposer "
            "consumption). PASS (label=r1b_diversity_above_matched_noise) = ARM_1 "
            "selected-action entropy strictly above BOTH the matched-entropy noise "
            "control (temperature 2.5) and the proposer baseline on >=2/3 seeds with "
            "readiness + C1 met => R1.b fires; ARC-065 GAP-A theory-1 is a real "
            "diversity contributor (supports). Readiness-below-floor OR C1 fail "
            "(SD-056 under-trained / amend not wired) self-routes "
            "substrate_not_ready_requeue => non_contributory, NOT a weakens. "
            "Readiness+C1 met but no strict lift => r1a_entropy_only_artefact "
            "(weakens; theory-1 not load-bearing alone -> substrate revisit). "
            "claim_ids=[ARC-065] only; MECH-341 committed-class GAP-B retest is the "
            "separate concurrent V3-EXQ-660."
        ),
        "interpretation": {
            "label": summary["label"],
            "preconditions": summary["preconditions"],
            "criteria": summary["criteria"],
            "criteria_non_degenerate": summary["criteria_non_degenerate"],
            "routing": {
                "r1b_diversity_above_matched_noise": "PASS -> R1.b cleared; GAP-A theory-1 (ARC-065) confirmed as a real diversity contributor",
                "substrate_not_ready_requeue": "re-queue as V3-EXQ-569g at higher P0 budget (or fix SD-056 wiring); do NOT weaken",
                "r1a_entropy_only_artefact": "FAIL -> R1.a: theory-1 not load-bearing alone; /failure-autopsy + substrate revisit",
            },
        },
        "dry_run": bool(dry_run),
        "config": {
            "seeds": seeds,
            "p0_warmup_episodes": p0,
            "p1_measurement_episodes": p1,
            "steps_per_episode": steps,
            "env_kwargs": ENV_KWARGS,
            "arms": [{k: a[k] for k in ("arm_id", "label", "candidate_summary_source", "temperature")} for a in ARMS],
            "sd056_weight": SD056_WEIGHT,
            "matched_entropy_temperature": MATCHED_ENTROPY_TEMPERATURE,
            "thresholds": {
                "c0_spread_floor": C0_SPREAD_FLOOR,
                "c0_magnitude_ceil": C0_MAGNITUDE_CEIL,
                "c1_pairwise_dist_floor": C1_PAIRWISE_DIST_FLOOR,
                "c3_selected_entropy_floor": C3_SELECTED_ENTROPY_FLOOR,
                "min_seeds_for_pass": MIN_SEEDS_FOR_PASS,
            },
        },
        "acceptance_criteria": {
            "readiness_substrate_ready": summary["readiness"]["readiness_ok"],
            "C1_arm1_e2_world_forward_divergent": summary["criteria_pass"]["C1"],
            "C_R1B_selected_entropy_strict_above_matched_noise": summary["criteria_pass"]["C_R1B"],
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
        description="V3-EXQ-569f ARC-065 GAP-A R1.b FP-2 matched-entropy falsifier (649 e2_world_forward stack)"
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
