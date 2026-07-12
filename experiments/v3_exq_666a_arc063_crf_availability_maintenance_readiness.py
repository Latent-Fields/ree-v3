"""V3-EXQ-666a: ARC-063 CRF availability-maintenance substrate-readiness diagnostic.

Purpose: diagnostic (claim_ids=[]). Supersedes V3-EXQ-666. Validates the 2026-06-11
crf-availability-maintenance amend (activity-silent maintenance trace +
maintained-pool readout, landed ree-v3 main 1d04e51) routed by
failure_autopsy_V3-EXQ-666_2026-06-11 + the targeted_review_arc_063_crf_rule_cell_persistence
B-leaning lit verdict. Does NOT weight governance and does NOT validate/weaken
MECH-309/ARC-062/ARC-063 -- those stay candidate / substrate_ceiling / v3_pending /
pending_retest_after_substrate. The ARC-062 GAP-B BEHAVIOURAL re-run (committed-class
entropy falsifier on the field-populated substrate) is a SEPARATE later
/queue-experiment session GATED on this readiness PASS.

THE GAP THIS CLOSES. V3-EXQ-666 decomposed the readiness gate into its two
sub-properties and found a differentiation<->persistence TENSION: the 654b mature-pool
amend delivers DIFFERENTIATION only via crf_context_from_e2_world_forward (ARM_2:
10-16 distinct rules, crf_max_pairwise_rule_dist 1.71) but WORSENS PERSISTENCE --
once each rule matches only a narrow context slice, its match-triggered availability
EMA never accumulates above theta between sparse matches and decays in the gaps, so
crf_frac_active collapses to 0.016 (worse than the undifferentiated legacy 0.125);
0/3 readiness cells in every arm. The B-leaning lit verdict (Mongillo 2008 / Stokes
2015 / Lundqvist 2018) says the unselected differentiated POOL must be held
activity-silently (NOT kept firing), and that crf_frac_active -- an INSTANTANEOUS
active fraction -- is the WRONG readiness readout for a sparsely-matched pool. The
amend (1) REMOVES the silence-driven availability decay under
crf_availability_maintenance (a minted differentiated rule HOLDS availability across
context-absent ticks; only the exception/interference path erodes it) and (2)
re-states the readiness readout on the MAINTAINED pool. This run confirms the
maintenance HOLDS a differentiated >=2-rule reactivatable pool at behavioural runtime
where the no-maintenance arm lets it erode out of the reactivatable set.

DESIGN (3 arms x 3 seeds). ALL arms run the SAME 654b/666 matched stack
(use_candidate_rule_field=True + use_lateral_pfc_analog=True UN-ZEROED trainable head +
crf_persist_rules_across_episode_reset=True + SD-056 e2 contrastive trained ONLINE so
e2.world_forward is action-divergent). The swept variables are the three flags:
  ARM_0_OFF                       : legacy CRF (no mature / no e2ctx / no maintenance)
                                    -- positive control reproducing the 654b/666 churn.
  ARM_1_MATURE_E2CTX              : mature + e2ctx, maintenance OFF
                                    -- reproduces V3-EXQ-666 ARM_2: DIFFERENTIATES
                                       (crf_max_pairwise_rule_dist high) but PERSISTENCE
                                       collapses (the maintained pool does NOT hold).
  ARM_2_MATURE_E2CTX_MAINTENANCE  : mature + e2ctx + maintenance (floor 0.45, decay 0.0)
                                    -- the fix: the differentiated pool is held silently.

CRF-READINESS GATE (load-bearing, RE-STATED on the MAINTAINED pool, NOT crf_frac_active
-- the load-bearing change vs V3-EXQ-666 per the B-leaning lit verdict):
  gate_cleared = crf_maintained_pairwise_dist > MAINT_DIST_FLOOR (0.1) AND
                 crf_n_maintained_reactivatable >= MAINT_N_FLOOR (2)
  i.e. >=2 DIFFERENTIATED rules are SIMULTANEOUSLY maintained-and-reactivatable.
crf_frac_active + crf_max_pairwise_rule_dist are kept as SECONDARY/diagnostic readouts
(expect ARM_1 and ARM_2 both LOW crf_frac_active -- confirming it is the wrong readout --
but only ARM_2 clears the maintained-pool gate).

INTERPRETATION GRID (the self-route is a hypothesis -- adjudicate before it drives any
governance action):
  - non_vacuity unmet (a cell minted <2, OR the e2ctx arms did not DIFFERENTIATE:
    crf_max_pairwise_rule_dist <= DIFF_FLOOR -- the pool the maintenance gate measures
    never formed) -> substrate_not_ready_requeue (NOT a fix verdict). Same statistic as
    the gate (pairwise rule distance), so a starved test never masquerades as a verdict.
  - ARM_2 clears the maintained-pool gate on >=2/3 AND ARM_1 does NOT (reproduces the
    666 persistence collapse) -> PASS: activity-silent maintenance HOLDS the
    differentiated pool the no-maintenance dynamics could not.
  - ARM_2 does NOT clear on >=2/3 (with differentiation real) -> FAIL: maintenance
    insufficient at behavioural runtime -> /failure-autopsy (NOT a requeue).
  - ARM_1 ALSO clears the maintained-pool gate -> e2ctx differentiation alone sufficed
    for persistence (maintenance not load-bearing): the readiness PASS cannot be
    attributed to the maintenance -> FAIL, flagged (criteria_non_degenerate). Revisit.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_666a_arc063_crf_availability_maintenance_readiness.py [--dry-run]

Writes a flat JSON manifest to REE_assembly/evidence/experiments/.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_PURPOSE = "diagnostic"
SUPERSEDES = "v3_exq_666_arc063_crf_mature_pool_readiness"

SEEDS = [42, 43, 44]
N_EPISODES = 100
STEPS_PER_EPISODE = 40
DRY_N_EPISODES = 3
DRY_STEPS_PER_EPISODE = 10

# CRF-readiness gate floors -- the load-bearing acceptance, RE-STATED on the
# MAINTAINED pool (NOT crf_frac_active) per the B-leaning lit verdict.
MAINT_DIST_FLOOR = 0.1    # crf_maintained_pairwise_dist; pinned dirs ~1.4 when >=2
#                           maintained-reactivatable rules differ, 0.0 when <=1.
MAINT_N_FLOOR = 2         # crf_n_maintained_reactivatable; >=2 simultaneously held.
DIFF_FLOOR = 0.1          # crf_max_pairwise_rule_dist; the e2ctx differentiation the
#                           maintenance gate presupposes (non-vacuity, same statistic).
MIN_MINTED_FLOOR = 2      # every cell must mint >=2 rules (non-vacuity).
SEED_MAJORITY = 2         # >=2/3 seeds.

# SD-056 online e2 training (mirror V3-EXQ-666 / 654b / 649 harness).
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0

# Maintenance knobs (the amend defaults; set explicitly when maintenance ON).
MAINTENANCE_FLOOR = 0.45
MAINTENANCE_DECAY = 0.0

# IDENTICAL env to V3-EXQ-666 / 654b / 614e so the readiness regime matches the
# GAP-B falsifier.
ENV_KWARGS = dict(
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
        "arm_id": "ARM_0_OFF",
        "label": "legacy_crf_no_mature_no_e2ctx_no_maintenance",
        "crf_mature_pool_dynamics": False,
        "crf_context_from_e2_world_forward": False,
        "crf_availability_maintenance": False,
    },
    {
        "arm_id": "ARM_1_MATURE_E2CTX",
        "label": "mature_e2ctx_no_maintenance_differentiates_but_persistence_collapses",
        "crf_mature_pool_dynamics": True,
        "crf_context_from_e2_world_forward": True,
        "crf_availability_maintenance": False,
    },
    {
        "arm_id": "ARM_2_MATURE_E2CTX_MAINTENANCE",
        "label": "mature_e2ctx_activity_silent_maintenance_the_fix",
        "crf_mature_pool_dynamics": True,
        "crf_context_from_e2_world_forward": True,
        "crf_availability_maintenance": True,
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """654b/666 matched stack; the ONLY swept variables are the three CRF amend flags.

    All arms build the CandidateRuleField (use_candidate_rule_field=True) with the
    LateralPFCAnalog bias head un-zeroed + trainable, crf_persist on, and the
    SD-056 / GAP-A / MECH-341 stack matched to the GAP-B falsifier.
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
        # Layer A: SP-CEM (candidate-pool first-action-class diversity).
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # GAP-A (V3-EXQ-649): shared per-candidate signal from e2.world_forward.
        candidate_summary_source="e2_world_forward",
        # modulatory-bias-selection-authority (V3-EXQ-643a float32 fix).
        use_modulatory_selection_authority=True,
        modulatory_authority_gain=0.5,
        # MECH-341 (stratified across-class; within-class temperature default).
        use_e3_score_diversity=True,
        use_e3_diversity_entropy_bonus=True,
        use_e3_diversity_stratified_select=True,
        e3_diversity_entropy_bias_scale=1.0,
        e3_diversity_stratified_within_class_temperature=None,
        # MECH-313 noise floor.
        use_noise_floor=True,
        noise_floor_alpha=0.1,
        # V_s minimal stack.
        use_per_stream_vs=True,
        use_vs_rollout_gating=True,
        vs_gate_snapshot_refresh_threshold=0.5,
        vs_gate_e1_threshold=0.4,
        # ARC-062 GatedPolicy + SD-033a LateralPFCAnalog (un-zeroed + trainable head).
        use_gated_policy=True,
        use_lateral_pfc_analog=True,
        lateral_pfc_train_rule_bias_head=True,
        # SD-056 (e2 action-conditional divergence; trained online below).
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_action_contrastive_multistep_enabled=True,
        e2_action_contrastive_horizon=5,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=2.0,
        # The 654a maturity fix (matched across arms).
        use_candidate_rule_field=True,
        crf_persist_rules_across_episode_reset=True,
        # --- The ONLY swept variables: the mature + e2ctx + maintenance flags ---
        crf_mature_pool_dynamics=bool(arm["crf_mature_pool_dynamics"]),
        crf_context_from_e2_world_forward=bool(
            arm["crf_context_from_e2_world_forward"]
        ),
        crf_availability_maintenance=bool(arm["crf_availability_maintenance"]),
        crf_maintenance_floor=MAINTENANCE_FLOOR,
        crf_maintenance_decay=MAINTENANCE_DECAY,
    )
    return REEAgent(cfg)


def _obs_harm(obs_dict):
    h = obs_dict.get("harm_obs")
    return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_a(obs_dict):
    h = obs_dict.get("harm_obs_a")
    return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_history(obs_dict):
    h = obs_dict.get("harm_history")
    return h.float().unsqueeze(0) if h is not None else None


# ---------------------------------------------------------------------------
# SD-056 online e2 training (mirror V3-EXQ-666)
# ---------------------------------------------------------------------------
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
        z_world_0=z0_K, actions=actions_K, z_world_1_targets=z1_K,
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
# One (arm, seed) cell: drive the agent + read the matured CRF pool
# ---------------------------------------------------------------------------
def _run_cell(arm: Dict[str, Any], seed: int, n_episodes: int,
              steps_per_episode: int) -> Dict[str, Any]:
    env = _make_env(seed)
    agent = _make_agent(env, arm)
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)
    sample_rng = random.Random(seed * 7919 + 13)
    transition_buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = (
        deque(maxlen=TRANSITION_BUFFER_MAX)
    )
    n_contrastive_steps = 0
    error_note: Optional[str] = None

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()  # crf_persist=True -> the field's reset() is a no-op
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
                obs_harm=_obs_harm(obs_dict),
                obs_harm_a=_obs_harm_a(obs_dict),
                obs_harm_history=_obs_harm_history(obs_dict),
            )

            # SD-056 transition capture (z0, a) this tick -> z1 next tick.
            if pending_capture is not None:
                z0_prev, a_prev = pending_capture
                z1_obs = latent.z_world.detach().reshape(-1).clone()
                if (torch.isfinite(z0_prev).all() and torch.isfinite(a_prev).all()
                        and torch.isfinite(z1_obs).all()):
                    transition_buffer.append((z0_prev, a_prev, z1_obs))
                pending_capture = None

            ticks = agent.clock.advance()
            wdim = latent.z_world.shape[-1]
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
            if action is None:
                idx = int(np.random.randint(0, env.action_dim))
                action = torch.zeros(1, env.action_dim, device=agent.device)
                action[0, idx] = 1.0
                agent._last_action = action
            if not torch.isfinite(action).all():
                error_note = (
                    f"non-finite action arm={arm['arm_id']} seed={seed} "
                    f"ep={ep} step={_step}"
                )
                break

            # Capture (z0, a) for the next-tick SD-056 transition.
            if torch.isfinite(latent.z_world).all() and torch.isfinite(action).all():
                pending_capture = (
                    latent.z_world.detach().reshape(-1).clone(),
                    action.detach().reshape(-1).clone(),
                )

            # SD-056 e2 online contrastive training (every tick).
            loss_val = _e2_contrastive_step(
                agent=agent, buffer=transition_buffer,
                optimiser=e2_opt, rng=sample_rng,
            )
            if loss_val is not None and math.isfinite(loss_val):
                n_contrastive_steps += 1

            _, _harm_signal, done, info, obs_dict = env.step(action)
            with torch.no_grad():
                agent.update_residue(
                    harm_signal=float(_harm_signal), world_delta=None,
                    hypothesis_tag=False, owned=True,
                )
            if agent.goal_state is not None:
                benefit_exposure = float(info.get("benefit_exposure", 0.0))
                energy = float(body[0, 3].item())
                agent.update_z_goal(
                    benefit_exposure=benefit_exposure,
                    drive_level=max(0.0, 1.0 - energy),
                )
            tick_in_ep += 1
            if done:
                break

        if (ep + 1) % 20 == 0 or (ep + 1) == n_episodes:
            print(
                f"  [train] cell {arm['arm_id']} seed={seed} "
                f"ep {ep + 1}/{n_episodes}",
                flush=True,
            )

    st = agent.candidate_rule_field.get_state()
    # PRIMARY (load-bearing) -- the maintained-pool readout.
    maint_n = int(st.get("crf_n_maintained_reactivatable", 0))
    maint_dist = float(st.get("crf_maintained_pairwise_dist", 0.0))
    frac_maintained = float(st.get("crf_frac_maintained", 0.0))
    # SECONDARY / diagnostic readouts.
    frac_active = float(st.get("crf_frac_active", 0.0))
    max_dist = float(st.get("crf_max_pairwise_rule_dist", 0.0))
    n_minted = int(st.get("crf_n_minted_total", 0))
    n_slots = int(st.get("crf_n_slots_minted", 0))

    gate_cleared = (maint_dist > MAINT_DIST_FLOOR) and (maint_n >= MAINT_N_FLOOR)
    differentiated = max_dist > DIFF_FLOOR  # the pool the maintenance gate measures
    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": seed,
        # primary (maintained-pool) readout
        "crf_n_maintained_reactivatable": maint_n,
        "crf_maintained_pairwise_dist": maint_dist,
        "crf_frac_maintained": frac_maintained,
        "crf_maintained_reactivation_threshold": float(
            st.get("crf_maintained_reactivation_threshold", 0.0)
        ),
        # secondary / diagnostic
        "crf_frac_active": frac_active,
        "crf_max_pairwise_rule_dist": max_dist,
        "crf_n_minted_total": n_minted,
        "crf_n_slots_minted": n_slots,
        "crf_n_retired_total": int(st.get("crf_n_retired_total", 0)),
        "n_contrastive_steps": n_contrastive_steps,
        # gates
        "gate_cleared": bool(gate_cleared),
        "differentiated": bool(differentiated),
        "minted_ge2": bool(n_minted >= MIN_MINTED_FLOOR),
        "error_note": error_note,
    }


def _arm_rows(rows: List[Dict[str, Any]], arm_id: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r["arm_id"] == arm_id]


def main(dry_run: bool = False):
    t0 = time.time()
    n_episodes = DRY_N_EPISODES if dry_run else N_EPISODES
    steps_per_episode = DRY_STEPS_PER_EPISODE if dry_run else STEPS_PER_EPISODE

    rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        for arm in ARMS:
            print(f"Seed {seed} Condition {arm['arm_id']}", flush=True)
            config_slice = {
                "arm_id": arm["arm_id"],
                "crf_mature_pool_dynamics": arm["crf_mature_pool_dynamics"],
                "crf_context_from_e2_world_forward":
                    arm["crf_context_from_e2_world_forward"],
                "crf_availability_maintenance": arm["crf_availability_maintenance"],
                "maintenance_floor": MAINTENANCE_FLOOR,
                "maintenance_decay": MAINTENANCE_DECAY,
                "env_kwargs": ENV_KWARGS,
                "n_episodes": n_episodes,
                "steps_per_episode": steps_per_episode,
            }
            with arm_cell(seed, config_slice=config_slice,
                          script_path=Path(__file__)) as cell:
                row = _run_cell(arm, seed, n_episodes, steps_per_episode)
                cell.stamp(row)
            rows.append(row)
            print(f"verdict: {'PASS' if row['gate_cleared'] else 'FAIL'}", flush=True)

    # --- Aggregate per arm ---
    arm_summary: Dict[str, Dict[str, Any]] = {}
    for arm in ARMS:
        ar = _arm_rows(rows, arm["arm_id"])
        n_cleared = sum(1 for r in ar if r["gate_cleared"])
        n_diff = sum(1 for r in ar if r["differentiated"])
        n_minted = sum(1 for r in ar if r["minted_ge2"])
        arm_summary[arm["arm_id"]] = {
            "n_seeds": len(ar),
            "n_gate_cleared": n_cleared,
            "n_differentiated": n_diff,
            "n_minted_ge2": n_minted,
            "min_n_minted": min((r["crf_n_minted_total"] for r in ar), default=0),
            "mean_n_maintained": (
                sum(r["crf_n_maintained_reactivatable"] for r in ar) / len(ar)
                if ar else 0.0
            ),
            "mean_maintained_pairwise_dist": (
                sum(r["crf_maintained_pairwise_dist"] for r in ar) / len(ar)
                if ar else 0.0
            ),
            "mean_frac_active": (
                sum(r["crf_frac_active"] for r in ar) / len(ar) if ar else 0.0
            ),
            "mean_max_pairwise_rule_dist": (
                sum(r["crf_max_pairwise_rule_dist"] for r in ar) / len(ar)
                if ar else 0.0
            ),
        }

    # --- Gates ---
    # NON-VACUITY: every cell mints >=2 AND the e2ctx arms (ARM_1 + ARM_2) DIFFERENTIATE
    # (crf_max_pairwise_rule_dist > DIFF_FLOOR on >=2/3 seeds) -- the pool the
    # maintenance gate measures must actually form. SAME statistic (pairwise rule
    # distance) as the load-bearing gate, so a starved test self-routes to
    # substrate_not_ready_requeue rather than masquerading as a verdict.
    min_minted_all = min(r["crf_n_minted_total"] for r in rows)
    min_diff_e2ctx = min(
        r["crf_max_pairwise_rule_dist"] for r in rows
        if r["arm_id"] in ("ARM_1_MATURE_E2CTX", "ARM_2_MATURE_E2CTX_MAINTENANCE")
    )
    minted_met = min_minted_all >= MIN_MINTED_FLOOR
    differentiation_met = (
        arm_summary["ARM_1_MATURE_E2CTX"]["n_differentiated"] >= SEED_MAJORITY
        and arm_summary["ARM_2_MATURE_E2CTX_MAINTENANCE"]["n_differentiated"]
        >= SEED_MAJORITY
    )
    non_vacuity_met = minted_met and differentiation_met
    # LOAD-BEARING: ARM_2 (maintenance) clears the MAINTAINED-pool gate on >=2/3 seeds.
    readiness_met = (
        arm_summary["ARM_2_MATURE_E2CTX_MAINTENANCE"]["n_gate_cleared"]
        >= SEED_MAJORITY
    )
    # DISCRIMINATION (same statistic): ARM_1 (no maintenance) must NOT clear the
    # maintained-pool gate on a majority (reproduces the 666 persistence collapse).
    discrimination_met = (
        arm_summary["ARM_1_MATURE_E2CTX"]["n_gate_cleared"] < SEED_MAJORITY
    )
    any_error = any(r["error_note"] for r in rows)

    if not non_vacuity_met:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        route = (
            "re-queue at an adequate P0 (a cell minted <2 OR the e2ctx arms did not "
            "differentiate -- the maintained-pool test was starved, NOT a fix verdict)"
        )
    elif readiness_met and discrimination_met:
        outcome = "PASS"
        label = "crf_availability_maintenance_holds_differentiated_pool"
        route = (
            "PASS -> governance flips substrate_queue crf-availability-maintenance "
            "ready=True; queue the 654c GAP-B behavioural re-run (committed-class "
            "entropy falsifier) on the field-populated maintained substrate"
        )
    elif not readiness_met:
        outcome = "FAIL"
        label = "maintenance_insufficient_at_behavioural_runtime"
        route = "/failure-autopsy (maintenance did not hold the differentiated pool)"
    else:  # readiness_met but not discrimination_met
        outcome = "FAIL"
        label = "e2ctx_differentiation_alone_maintains_arm1_also_cleared"
        route = (
            "/failure-autopsy (ARM_1 without maintenance ALSO cleared the "
            "maintained-pool gate -> the readiness PASS cannot be attributed to "
            "the activity-silent maintenance; e2ctx differentiation alone persisted)"
        )
    if any_error:
        outcome = "FAIL"

    exp_type = "v3_exq_666a_arc063_crf_availability_maintenance_readiness"
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{exp_type}_{ts}_v3"  # run_id must END in _v3 (V3 tagging rule)
    elapsed = time.time() - t0

    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": exp_type,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": [],
        "outcome": outcome,
        "evidence_direction": "non_contributory",
        "supersedes": SUPERSEDES,
        "metrics": {
            "arm_summary": arm_summary,
            "readiness_met_arm2_maintenance": readiness_met,
            "discrimination_met_arm1_no_maintenance_does_not_hold": discrimination_met,
            "non_vacuity_met": non_vacuity_met,
            "minted_met_all_cells_ge2": minted_met,
            "differentiation_met_e2ctx_arms": differentiation_met,
            "min_n_minted_all_cells": min_minted_all,
            "min_max_pairwise_dist_e2ctx_arms": min_diff_e2ctx,
            "maint_dist_floor": MAINT_DIST_FLOOR,
            "maint_n_floor": MAINT_N_FLOOR,
            "diff_floor": DIFF_FLOOR,
            "min_minted_floor": MIN_MINTED_FLOOR,
            "seed_majority": SEED_MAJORITY,
        },
        "arm_results": rows,
        "interpretation": {
            "label": label,
            "route": route,
            "preconditions": [
                {
                    "name": "field_minted_ge2_all_cells",
                    "kind": "readiness",
                    "description": (
                        "every cell must mint >=2 rules so a below-gate reading is "
                        "maintenance-insufficiency, not field-not-engaged. Below "
                        "floor -> substrate_not_ready_requeue."
                    ),
                    "control": "all (arm x seed) cells driven over the 654b hazard env",
                    "measured": min_minted_all,
                    "threshold": MIN_MINTED_FLOOR,
                    "direction": "lower",
                    "met": bool(minted_met),
                },
                {
                    "name": "e2ctx_pool_differentiated_supra_floor",
                    "kind": "readiness",
                    "description": (
                        "the maintenance gate routes on crf_maintained_pairwise_dist "
                        "(a pairwise rule distance over the MAINTAINED subset). Its "
                        "precondition asserts the SAME statistic on the FULL pool "
                        "(crf_max_pairwise_rule_dist > floor) on the e2ctx arms -- the "
                        "differentiated pool the maintenance is supposed to HOLD must "
                        "first exist. Below floor (e2ctx did not differentiate) -> "
                        "substrate_not_ready_requeue, NOT a maintenance verdict "
                        "(same-statistic V3-EXQ-643 lesson)."
                    ),
                    "control": (
                        "min crf_max_pairwise_rule_dist over ARM_1+ARM_2 cells "
                        "(e2ctx differentiation source ON, SD-056-trained e2)"
                    ),
                    "measured": min_diff_e2ctx,
                    "threshold": DIFF_FLOOR,
                    "direction": "lower",
                    "met": bool(differentiation_met),
                },
            ],
            "criteria_non_degenerate": {
                "maintained_pool_gate_discriminates_vs_arm1_no_maintenance": bool(
                    discrimination_met
                ),
                "e2ctx_pool_differentiated": bool(differentiation_met),
                "field_minted_ge2_all_cells": bool(minted_met),
            },
            "criteria": [
                {
                    "name": "ARM_2_MAINTENANCE_clears_maintained_pool_gate",
                    "load_bearing": True,
                    "passed": bool(readiness_met),
                },
                {
                    "name": "ARM_1_NO_MAINTENANCE_does_not_hold_maintained_pool",
                    "load_bearing": False,
                    "passed": bool(discrimination_met),
                },
            ],
        },
        "elapsed_seconds": elapsed,
        "notes": (
            "ARC-063 crf-availability-maintenance substrate-readiness diagnostic "
            "(claim_ids=[], non_contributory -- does NOT weight governance). "
            "Supersedes V3-EXQ-666. Validates the 2026-06-11 amend "
            "(failure_autopsy_V3-EXQ-666 + targeted_review_arc_063_crf_rule_cell_persistence "
            "B-leaning verdict): activity-silent maintenance (silence no longer erodes "
            "availability) + the maintained-pool readout that REPLACES crf_frac_active "
            "as the readiness criterion. The gate is RE-STATED on the maintained pool "
            "(crf_maintained_pairwise_dist > 0.1 AND crf_n_maintained_reactivatable "
            ">= 2): ARM_2 (maintenance) must clear it where ARM_1 (mature+e2ctx, no "
            "maintenance) reproduces the 666 persistence collapse (differentiates but "
            "the maintained pool does not hold; crf_frac_active is LOW in both -- the "
            "averaged-activity artefact the verdict identified). PASS clears the "
            "substrate_queue ready gate before the 654c GAP-B behavioural re-run is "
            "scored; the GAP-B falsifier (MECH-309 / ARC-062) is the "
            "governance-weighting successor, queued separately. MECH-309/ARC-062/ARC-063 "
            "stay candidate / substrate_ceiling / v3_pending / pending_retest_after_substrate."
        ),
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

    print(f"verdict: {manifest['outcome']}")
    print(f"  label: {label}")
    for aid, summ in arm_summary.items():
        print(
            f"  {aid}: gate_cleared={summ['n_gate_cleared']}/{summ['n_seeds']} "
            f"diff={summ['n_differentiated']}/{summ['n_seeds']} "
            f"mean_n_maint={summ['mean_n_maintained']:.2f} "
            f"mean_maint_dist={summ['mean_maintained_pairwise_dist']:.3f} "
            f"mean_frac_active={summ['mean_frac_active']:.3f} "
            f"mean_full_dist={summ['mean_max_pairwise_rule_dist']:.3f}",
            flush=True,
        )
    print(f"Result written to: {out_path}", flush=True)
    return manifest["outcome"], str(out_path)


if __name__ == "__main__":
    _dry = "--dry-run" in sys.argv
    _outcome, _path = main(dry_run=_dry)
    emit_outcome(outcome=_outcome, manifest_path=_path)
