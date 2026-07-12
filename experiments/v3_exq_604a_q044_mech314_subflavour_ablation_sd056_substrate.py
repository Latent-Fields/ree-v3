#!/opt/local/bin/python3
"""V3-EXQ-604a -- MECH-314 structured-curiosity validation on the SD-056 substrate.

Supersedes V3-EXQ-604 (Q-044 sub-flavour ablation; FAIL non_contributory /
substrate_ceiling 2026-05-21). 604's governance note: "All four arms identical
selected_action_entropy (0.244051) and monomodal collapse (seed 42
unique_actions=1) -- MECH-314 sub-flavour ablations exert no discriminable
signal. Do not weaken Q-044/MECH-314; RETEST AFTER structured-curiosity
substrate carries in."

ROOT CAUSE of the 604 collapse (per the MECH-314a autopsy + the SD-056 landing
notes): 604 ran the agent forward with NO gradient training, so E2.world_forward
mapped every candidate first-action to a near-identical z_world (the action
contribution was never fitted off random init). MECH-314a's per-candidate
novelty bonus reads candidate z_world; with z_world degenerate across candidates,
the bonus is uniform and exerts no selection signal -- and 314b/314c are
broadcast scalars by design, so the whole cluster looked inert. SP-CEM (which 604
used) diversifies action-OBJECTS at the proposal stage but the untrained E2
collapses them in z_world before scoring.

THE SUBSTRATE THAT "CARRIES IT IN": SD-056 (e2.action_conditional_divergence_
contrastive, landed 2026-05-29). When E2.world_forward is trained with the
InfoNCE action-contrastive loss, different first-actions produce divergent
z_world, so the candidate pool is no longer z_world-degenerate and the
per-candidate curiosity bonus has real inputs to act on.

This experiment re-runs the MECH-314 validation on the SD-056-trained substrate,
using the online E2-contrastive training harness validated in V3-EXQ-569d, and
adds a C0 substrate-readiness guard (cand_world_pairwise_dist > floor) so a null
result is INTERPRETABLE (collapse-confounded vs genuine no-effect) rather than
silently re-hitting the 604 ceiling.

=== ARMS (5) ===

All arms run SD-056 ON (e2_action_contrastive trained online, weight 0.05 --
the 569d ARM_2 operative weight) so candidates are action-divergent. The varying
axis is structured curiosity:
  ARM_OFF              -- use_structured_curiosity=False (MECH-314 parent anchor)
  ARM_ALL_ON           -- novelty + uncertainty + LP all ON
  ARM_NOVELTY_OFF      -- MECH-314a OFF (314b + 314c ON)
  ARM_UNCERTAINTY_OFF  -- MECH-314b OFF
  ARM_LP_OFF           -- MECH-314c OFF

=== ACCEPTANCE (pre-registered) ===

  C0 (substrate-readiness GUARD): mean cand_world_pairwise_dist > 0.03 in
     >= MIN_SEEDS_PER_ARM_FOR_PASS / SEEDS in EACH arm. Confirms SD-056 made
     candidates action-divergent (the precondition 604 lacked). If C0 FAILS the
     run is non_contributory (collapse persists) -> /diagnose-errors on the
     contrastive training, NOT a MECH-314 falsification.
  C1 (MECH-314 PARENT effect): ARM_ALL_ON selected_action_class_entropy differs
     from ARM_OFF by > DISTINCT_MARGIN (0.03). The structured curiosity bonus
     changes selection behaviour -> supports MECH-314 parent.
  C2 (Q-044 sub-flavour discriminability; informative): at least one sub-flavour
     ablation arm differs from ARM_ALL_ON by > DISTINCT_MARGIN. Identifies which
     sub-flavours are behaviourally load-bearing at the selection level. By the
     MECH-314 Phase-1 honest-scoping caveat (314b/314c are broadcast scalars
     that shift all candidates equally and do NOT change selection ordering),
     MECH-314a (novelty, per-candidate) is the expected discriminator.

Overall PASS (evidence supports MECH-314) = C0 AND C1.
C2 is reported and feeds evidence_direction_per_claim for the sub-flavours but
is NOT required for overall PASS (a parent-effect with only 314a load-bearing
is a valid, expected outcome -- it tells governance 314b/314c are not
selection-level load-bearing at Phase 1, which is the honest substrate reading).

=== DIAGNOSTIC INTERPRETATION GRID ===

| Outcome                       | Reading                                            | Next action |
|-------------------------------|----------------------------------------------------|-------------|
| C0 + C1 + C2 all hold         | Curiosity bonus is behaviourally operative AND >=1 | MECH-314 + sub-flavours supported. Governance: candidate_substrate_landed -> candidate. Q-044: which sub-flavours load-bearing recorded. |
|                               | sub-flavour discriminates.                         | |
| C0 + C1, C2 fails             | Parent effect real; only the broadcast scalars     | MECH-314 parent supported; 314a vs 314b/c independence not separable at selection level (expected per Phase-1 caveat). 314b/314c -> mixed. |
|                               | were ablated (314a still on in those arms).        | |
| C0 holds, C1 fails            | Candidates divergent but curiosity ON==OFF.        | MECH-314 weakens on this env (bonus magnitude too small vs score gap). /failure-autopsy; consider curiosity_bias_scale sweep. |
| C0 fails                      | SD-056 did not make candidates divergent at runtime| INVALID for MECH-314. Route to /diagnose-errors on contrastive training (cf. 569d C1-fail cell). Do NOT weaken MECH-314. |

architecture_epoch: "ree_hybrid_guardrails_v1"

Smoke:
  /opt/local/bin/python3 experiments/v3_exq_604a_q044_mech314_subflavour_ablation_sd056_substrate.py --dry-run
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
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_604a_q044_mech314_subflavour_ablation_sd056_substrate"
QUEUE_ID = "V3-EXQ-604a"
CLAIM_IDS: List[str] = ["Q-044", "MECH-314", "MECH-314a", "MECH-314b", "MECH-314c"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES = "V3-EXQ-604"

SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 60          # mirror 569d extended warmup (let E2 contrastive train)
P1_MEASUREMENT_EPISODES = 20
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_STEPS = 30

# Acceptance thresholds (pre-registered).
C0_PAIRWISE_DIST_FLOOR = 0.03            # substrate-operative floor (per 569d calibration)
DISTINCT_MARGIN = 0.03                   # selected-entropy delta for C1 / C2 (604's margin)
MIN_SEEDS_PER_ARM_FOR_PASS = 2           # of 3

# SD-056 online contrastive training (mirror 569d harness).
SD056_WEIGHT = 0.05                      # 569d ARM_2 operative weight
E2_CONTRASTIVE_LR = 1e-3
E2_TRAIN_EVERY_K_TICKS = 1
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0

# Curiosity magnitudes (604 calibration; EXQ-573 elevated regime).
CURIOSITY_BIAS_SCALE = 0.5
CURIOSITY_WEIGHT = 0.25

# ENV identical to V3-EXQ-604 / 569d / 611b so manifest-comparability holds.
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
        "arm_id": "ARM_OFF",
        "label": "structured_curiosity_off_parent_anchor",
        "use_structured_curiosity": False,
        "use_curiosity_novelty": False,
        "use_curiosity_uncertainty": False,
        "use_curiosity_learning_progress": False,
    },
    {
        "arm_id": "ARM_ALL_ON",
        "label": "novelty_uncertainty_lp_all_on",
        "use_structured_curiosity": True,
        "use_curiosity_novelty": True,
        "use_curiosity_uncertainty": True,
        "use_curiosity_learning_progress": True,
    },
    {
        "arm_id": "ARM_NOVELTY_OFF",
        "label": "mech314a_off",
        "use_structured_curiosity": True,
        "use_curiosity_novelty": False,
        "use_curiosity_uncertainty": True,
        "use_curiosity_learning_progress": True,
    },
    {
        "arm_id": "ARM_UNCERTAINTY_OFF",
        "label": "mech314b_off",
        "use_structured_curiosity": True,
        "use_curiosity_novelty": True,
        "use_curiosity_uncertainty": False,
        "use_curiosity_learning_progress": True,
    },
    {
        "arm_id": "ARM_LP_OFF",
        "label": "mech314c_off",
        "use_structured_curiosity": True,
        "use_curiosity_novelty": True,
        "use_curiosity_uncertainty": True,
        "use_curiosity_learning_progress": False,
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """Main-path SP-CEM + V_s + SD-054 stack, SD-056 ON (uniform across arms),
    structured curiosity per-arm sub-flavour overrides.

    SD-056 is uniform ON so candidates are action-divergent in EVERY arm (the
    precondition 604 lacked). MECH-341 (Layer B) and MECH-313 (Layer C) are OFF
    so structured curiosity is the single varying selection-bias axis.
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
        # ARC-065 SP-CEM (Layer A) -- main-path default (also in 604)
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # Layer B / C OFF (structured curiosity is the single varying axis)
        use_e3_score_diversity=False,
        use_noise_floor=False,
        use_dacc=False,
        # V_s substrate (main-path default; identical across arms)
        use_per_stream_vs=True,
        use_per_region_vs=True,
        use_event_segmenter=True,
        use_invalidation_trigger=True,
        use_anchor_sets=True,
        # SD-056 -- uniform ON across all arms (the substrate that "carries it in")
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        # MECH-314 structured curiosity -- the varying axis
        use_structured_curiosity=bool(arm["use_structured_curiosity"]),
        use_curiosity_novelty=bool(arm["use_curiosity_novelty"]),
        use_curiosity_uncertainty=bool(arm["use_curiosity_uncertainty"]),
        use_curiosity_learning_progress=bool(arm["use_curiosity_learning_progress"]),
        curiosity_bias_scale=CURIOSITY_BIAS_SCALE,
        curiosity_novelty_weight=CURIOSITY_WEIGHT,
        curiosity_uncertainty_weight=CURIOSITY_WEIGHT,
        curiosity_learning_progress_weight=CURIOSITY_WEIGHT,
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# Per-tick measurement helpers (verbatim from V3-EXQ-569d)
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
    arm_weight: float,
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
    weighted = float(arm_weight) * loss
    weighted.backward()
    torch.nn.utils.clip_grad_norm_(agent.e2.parameters(), max_norm=MAX_GRAD_NORM)
    optimiser.step()
    return loss_val


# ---------------------------------------------------------------------------
# Per-(seed, arm) runner (adapted from 569d; SD-056 trained on every arm,
# selected-action-class entropy is the MECH-314 / Q-044 measurement)
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

    # SD-056 is ON in every arm -> every arm trains E2 online.
    e2_opt: torch.optim.Optimizer = torch.optim.Adam(
        agent.e2.parameters(), lr=E2_CONTRASTIVE_LR
    )

    transition_buffer: Deque[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)
    n_buffer_appends = 0
    n_contrastive_skipped_nonfinite = 0
    n_contrastive_skipped_sparse = 0

    total_train_eps = p0_episodes + p1_episodes

    pairwise_dists: List[float] = []
    candidate_first_action_counts: Counter = Counter()
    candidate_unique_per_tick: List[float] = []
    candidate_entropy_per_tick: List[float] = []
    selected_class_counts: Counter = Counter()
    contrastive_loss_values: List[float] = []
    curiosity_bias_abs_means: List[float] = []

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
                    if math.isfinite(dist):
                        pairwise_dists.append(dist)
                # Curiosity bias magnitude diagnostic (substrate-firing evidence).
                if getattr(agent, "curiosity", None) is not None:
                    st = agent.curiosity.get_state()
                    bias_abs = st.get("last_bias_abs_mean")
                    if bias_abs is not None and math.isfinite(float(bias_abs)):
                        curiosity_bias_abs_means.append(float(bias_abs))

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

            if is_p1:
                committed_class = int(action[0].argmax().item())
                selected_class_counts[committed_class] += 1
                n_p1_ticks += 1
            else:
                n_p0_ticks += 1

            if tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0:
                loss_val = _e2_contrastive_step(
                    agent=agent,
                    buffer=transition_buffer,
                    arm_weight=SD056_WEIGHT,
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

    def _maxx(xs: List[float], default: float = 0.0) -> float:
        return float(max(xs)) if xs else default

    def _minx(xs: List[float], default: float = 0.0) -> float:
        return float(min(xs)) if xs else default

    selected_action_entropy = _entropy_from_counts(dict(selected_class_counts))

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
        "cand_world_pairwise_dist_mean": round(_mean(pairwise_dists), 6),
        "cand_world_pairwise_dist_max": round(_maxx(pairwise_dists), 6),
        "cand_world_pairwise_dist_min": round(_minx(pairwise_dists), 6),
        "candidate_first_action_entropy_mean": round(_mean(candidate_entropy_per_tick), 6),
        "candidate_unique_first_action_classes_mean": round(_mean(candidate_unique_per_tick), 6),
        "candidate_first_action_counts": dict(sorted(candidate_first_action_counts.items())),
        "selected_action_class_entropy": round(selected_action_entropy, 6),
        "selected_class_counts": dict(sorted(selected_class_counts.items())),
        "selected_classes_n_unique": int(len(selected_class_counts)),
        "curiosity_bias_abs_mean": round(_mean(curiosity_bias_abs_means), 6),
        "contrastive_loss_mean": round(_mean(contrastive_loss_values), 6),
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
    by_id = {a["arm_id"]: _arm_rows(arm_results, a["arm_id"]) for a in ARMS}

    # C0: substrate-readiness -- pairwise_dist > floor in majority of seeds in EACH arm.
    c0_per_arm = {
        aid: _n_seeds_above(rows, "cand_world_pairwise_dist_mean", C0_PAIRWISE_DIST_FLOOR)
        for aid, rows in by_id.items()
    }
    c0_pass = all(n >= MIN_SEEDS_PER_ARM_FOR_PASS for n in c0_per_arm.values())

    e_off = _mean_key(by_id["ARM_OFF"], "selected_action_class_entropy")
    e_all = _mean_key(by_id["ARM_ALL_ON"], "selected_action_class_entropy")
    e_nov = _mean_key(by_id["ARM_NOVELTY_OFF"], "selected_action_class_entropy")
    e_unc = _mean_key(by_id["ARM_UNCERTAINTY_OFF"], "selected_action_class_entropy")
    e_lp = _mean_key(by_id["ARM_LP_OFF"], "selected_action_class_entropy")

    # C1: MECH-314 parent effect -- curiosity ON differs from OFF.
    c1_delta = abs(e_all - e_off)
    c1_pass = c1_delta > DISTINCT_MARGIN

    # C2 (informative): at least one sub-flavour ablation differs from ALL_ON.
    sub_deltas = {
        "ARM_NOVELTY_OFF": abs(e_nov - e_all),
        "ARM_UNCERTAINTY_OFF": abs(e_unc - e_all),
        "ARM_LP_OFF": abs(e_lp - e_all),
    }
    c2_arms_passed = [a for a, d in sub_deltas.items() if d > DISTINCT_MARGIN]
    c2_pass = len(c2_arms_passed) >= 1

    overall_pass = bool(c0_pass and c1_pass)

    return {
        "c0_floor": C0_PAIRWISE_DIST_FLOOR,
        "c0_per_arm_n_seeds_above": c0_per_arm,
        "c0_min_seeds_required": MIN_SEEDS_PER_ARM_FOR_PASS,
        "c0_pass": bool(c0_pass),
        "selected_entropy_ARM_OFF": round(e_off, 6),
        "selected_entropy_ARM_ALL_ON": round(e_all, 6),
        "selected_entropy_ARM_NOVELTY_OFF": round(e_nov, 6),
        "selected_entropy_ARM_UNCERTAINTY_OFF": round(e_unc, 6),
        "selected_entropy_ARM_LP_OFF": round(e_lp, 6),
        "c1_distinct_margin": DISTINCT_MARGIN,
        "c1_parent_delta_on_vs_off": round(c1_delta, 6),
        "c1_pass": bool(c1_pass),
        "c2_sub_flavour_deltas_vs_all_on": {k: round(v, 6) for k, v in sub_deltas.items()},
        "c2_arms_passed": c2_arms_passed,
        "c2_pass": bool(c2_pass),
        "pairwise_dist_per_arm_mean": {
            aid: round(_mean_key(rows, "cand_world_pairwise_dist_mean"), 6)
            for aid, rows in by_id.items()
        },
        "curiosity_bias_abs_mean_per_arm": {
            aid: round(_mean_key(rows, "curiosity_bias_abs_mean"), 6)
            for aid, rows in by_id.items()
        },
        "overall_pass": overall_pass,
    }


def _evidence_direction_per_claim(summary: Dict[str, Any]) -> Dict[str, str]:
    parent = "supports" if summary["overall_pass"] else (
        "does_not_support" if summary["c0_pass"] else "non_contributory"
    )
    # Sub-flavours: supports if their ablation discriminably moved entropy.
    def _sub(arm_id: str) -> str:
        if not summary["c0_pass"]:
            return "non_contributory"
        return "supports" if arm_id in summary["c2_arms_passed"] else "mixed"
    return {
        "Q-044": "supports" if summary["c2_pass"] and summary["c0_pass"] else (
            "mixed" if summary["c0_pass"] else "non_contributory"
        ),
        "MECH-314": parent,
        "MECH-314a": _sub("ARM_NOVELTY_OFF"),
        "MECH-314b": _sub("ARM_UNCERTAINTY_OFF"),
        "MECH-314c": _sub("ARM_LP_OFF"),
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
    edpc = _evidence_direction_per_claim(summary)

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
        "evidence_direction": edpc["MECH-314"],
        "evidence_direction_per_claim": edpc,
        "evidence_direction_note": (
            "MECH-314 structured-curiosity validation on the SD-056-trained "
            "(non-degenerate) substrate. Supersedes V3-EXQ-604, which FAILed "
            "non_contributory/substrate_ceiling (all arms identical, monomodal "
            "collapse; governance directed 'retest after structured-curiosity "
            "substrate carries in'). 604 ran the agent forward with NO E2 training "
            "so candidate z_world was action-degenerate and the per-candidate "
            "novelty bonus (314a) had no signal. This run trains E2 online with "
            "the SD-056 action-contrastive loss (569d harness, weight 0.05) so "
            "candidates are action-divergent (C0 guard: cand_world_pairwise_dist "
            "> 0.03 in every arm), then ablates structured curiosity. C0+C1 PASS "
            "= curiosity ON differs from OFF -> MECH-314 parent supported. C2 "
            "identifies which sub-flavours are selection-level load-bearing "
            "(314a/novelty expected; 314b/c are broadcast scalars by design). "
            "C0 FAIL = collapse persists -> non_contributory, /diagnose-errors "
            "on contrastive training, NOT a MECH-314 falsification."
        ),
        "dry_run": bool(dry_run),
        "config": {
            "seeds": seeds,
            "p0_warmup_episodes": p0,
            "p1_measurement_episodes": p1,
            "steps_per_episode": steps,
            "env_kwargs": ENV_KWARGS,
            "arms": [{"arm_id": a["arm_id"], "label": a["label"],
                       "use_structured_curiosity": a["use_structured_curiosity"],
                       "use_curiosity_novelty": a["use_curiosity_novelty"],
                       "use_curiosity_uncertainty": a["use_curiosity_uncertainty"],
                       "use_curiosity_learning_progress": a["use_curiosity_learning_progress"]}
                      for a in ARMS],
            "sd056_weight": SD056_WEIGHT,
            "curiosity_bias_scale": CURIOSITY_BIAS_SCALE,
            "curiosity_weight": CURIOSITY_WEIGHT,
            "c0_pairwise_dist_floor": C0_PAIRWISE_DIST_FLOOR,
            "distinct_margin": DISTINCT_MARGIN,
            "min_seeds_per_arm_for_pass": MIN_SEEDS_PER_ARM_FOR_PASS,
            "e2_contrastive_lr": E2_CONTRASTIVE_LR,
        },
        "acceptance_criteria": {
            "C0_substrate_readiness_candidates_divergent": summary["c0_pass"],
            "C1_mech314_parent_effect_on_vs_off": summary["c1_pass"],
            "C2_subflavour_discriminability": summary["c2_pass"],
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

    print(f"Outcome: {outcome}", flush=True)
    for k, v in manifest["acceptance_criteria"].items():
        print(f"  {k}: {v}", flush=True)

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-604a MECH-314 curiosity validation on SD-056 substrate"
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
