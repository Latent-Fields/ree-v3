"""V3-EXQ-666: ARC-063 CandidateRuleField mature-pool substrate-readiness diagnostic.

Purpose: diagnostic (claim_ids=[]). Validates the 2026-06-11 ARC-063 amend
(mature-pool gate/credit/retire dynamics, landed ree-v3 main 7e2e0ef) routed by
failure_autopsy_V3-EXQ-654b_2026-06-11. Does NOT weight governance and does NOT
validate/weaken MECH-309/ARC-062/ARC-063 -- those stay candidate / substrate_ceiling
/ v3_pending / pending_retest_after_substrate. The ARC-062 GAP-B BEHAVIOURAL
re-run (committed-class entropy falsifier on the field-populated substrate) is a
SEPARATE later /queue-experiment session GATED on this readiness PASS.

THE GAP THIS CLOSES. Across V3-EXQ-654 (per-episode wipe), 654a (crf_persist), and
654b (crf_persist + 240 ep) the CandidateRuleField never matured: crf_frac_active
pinned ~0.12-0.14 and crf_max_pairwise_rule_dist EXACTLY 0.0 on every ARM_ON cell
despite 452-1014 cumulative mints (the pool churns -- mint -> brief life -> retire
-> re-mint -- and never holds >=2 rules present). The budget reading is exhausted;
the blocker is the GATE/CREDIT/RETIRE dynamics. The amend recalibrates them behind
crf_mature_pool_dynamics (conflict-gate theta<1 for >=2 matched rules; lower decay;
absolute retire floor; asymmetric negative credit; mint-youth protection; decoupled
mint-block threshold) and adds an optional e2.world_forward CRF context source
(crf_context_from_e2_world_forward, mirrors ARC-065 GAP-A). Unit smoke + contracts
C13/C14 proved the mechanism (legacy n_present=1/dist=0.0/READY=False -> mature
n_present=2/dist=1.49/READY=True); this run confirms it ENGAGES and matures the pool
at behavioural runtime under the matched 654b/649/SD-056/MECH-341 stack.

DESIGN (3 arms x 3 seeds). ALL arms run the SAME 654b matched stack with
use_candidate_rule_field=True + use_lateral_pfc_analog=True (UN-ZEROED bias head) +
crf_persist_rules_across_episode_reset=True + SD-056 e2 contrastive trained ONLINE
(so e2.world_forward is action-divergent and the regime reproduces the GAP-B
falsifier). The ONLY swept variables are the two new amend flags:
  ARM_0_OFF          : mature OFF, e2-context OFF  (reproduces the 654b churn ->
                       EXPECTED crf_max_pairwise_rule_dist ~0.0 AND frac_active<0.30)
  ARM_1_MATURE       : mature ON,  e2-context OFF  (the primary fix)
  ARM_2_MATURE_E2CTX : mature ON,  e2-context ON   (mature + e2.world_forward context)

CRF-READINESS GATE (load-bearing, V3-EXQ-639 style):
  gate_cleared = crf_max_pairwise_rule_dist > DIST_FLOOR (0.1) AND
                 crf_frac_active >= FRAC_FLOOR (0.30)

INTERPRETATION GRID (the self-route is a hypothesis -- adjudicate before it drives
any governance action):
  - non_vacuity unmet (a gate-relevant cell minted 0 rules) -> the field never
    engaged -> substrate_not_ready_requeue (NOT a fix verdict).
  - ARM_1_MATURE clears the gate on >=2/3 AND ARM_0_OFF does NOT (reproduces the
    654b signature) -> PASS: the mature dynamics mature a differentiated persistent
    pool the legacy dynamics could not.
  - ARM_1_MATURE does NOT clear on >=2/3 (with minting) -> FAIL: mature dynamics
    insufficient at behavioural runtime -> /failure-autopsy (NOT a requeue).
  - ARM_0_OFF ALSO clears the gate -> env non-discriminating (too easy / encoder
    spread too high): the readiness PASS cannot be attributed to the fix -> FAIL,
    flagged (criteria_non_degenerate). Do NOT silently pass.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_666_arc063_crf_mature_pool_readiness.py [--dry-run]

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

SEEDS = [42, 43, 44]
N_EPISODES = 100
STEPS_PER_EPISODE = 40
DRY_N_EPISODES = 3
DRY_STEPS_PER_EPISODE = 10

# CRF-readiness gate floors (the load-bearing acceptance).
DIST_FLOOR = 0.1          # crf_max_pairwise_rule_dist; pinned directions ~1.4 when
#                           >=2 present, exactly 0.0 when <=1 present.
FRAC_FLOOR = 0.30         # crf_frac_active; the autopsy's 0.30 maturation floor.
SEED_MAJORITY = 2         # >=2/3 seeds.

# SD-056 online e2 training (mirror V3-EXQ-654b / 649 harness).
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0

# IDENTICAL env to V3-EXQ-654b / 614e (SD-054 reef + hazard_food_attraction +
# bipartite layout) so the readiness regime matches the GAP-B falsifier.
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
        "label": "legacy_crf_dynamics_raw_zworld_context",
        "crf_mature_pool_dynamics": False,
        "crf_context_from_e2_world_forward": False,
    },
    {
        "arm_id": "ARM_1_MATURE",
        "label": "mature_pool_dynamics_raw_zworld_context",
        "crf_mature_pool_dynamics": True,
        "crf_context_from_e2_world_forward": False,
    },
    {
        "arm_id": "ARM_2_MATURE_E2CTX",
        "label": "mature_pool_dynamics_e2_world_forward_context",
        "crf_mature_pool_dynamics": True,
        "crf_context_from_e2_world_forward": True,
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """654b matched stack; the ONLY swept variables are the two new amend flags.

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
        # --- The ONLY swept variables: the 654b amend flags ---
        crf_mature_pool_dynamics=bool(arm["crf_mature_pool_dynamics"]),
        crf_context_from_e2_world_forward=bool(
            arm["crf_context_from_e2_world_forward"]
        ),
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
# SD-056 online e2 training (mirror V3-EXQ-654b)
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
    frac_active = float(st.get("crf_frac_active", 0.0))
    max_dist = float(st.get("crf_max_pairwise_rule_dist", 0.0))
    n_minted = int(st.get("crf_n_minted_total", 0))
    n_slots = int(st.get("crf_n_slots_minted", 0))
    gate_cleared = (max_dist > DIST_FLOOR) and (frac_active >= FRAC_FLOOR)
    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": seed,
        "crf_frac_active": frac_active,
        "crf_max_pairwise_rule_dist": max_dist,
        "crf_n_minted_total": n_minted,
        "crf_n_slots_minted": n_slots,
        "crf_n_retired_total": int(st.get("crf_n_retired_total", 0)),
        "n_contrastive_steps": n_contrastive_steps,
        "gate_cleared": bool(gate_cleared),
        "minted": bool(n_minted > 0),
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
        n_minted = sum(1 for r in ar if r["minted"])
        arm_summary[arm["arm_id"]] = {
            "n_seeds": len(ar),
            "n_gate_cleared": n_cleared,
            "n_minted": n_minted,
            "min_n_minted": min((r["crf_n_minted_total"] for r in ar), default=0),
            "mean_frac_active": (
                sum(r["crf_frac_active"] for r in ar) / len(ar) if ar else 0.0
            ),
            "mean_max_pairwise_rule_dist": (
                sum(r["crf_max_pairwise_rule_dist"] for r in ar) / len(ar)
                if ar else 0.0
            ),
        }

    # --- Gates ---
    # Non-vacuity: every gate-relevant cell (ALL arms) must mint, else the field
    # never engaged -> substrate_not_ready_requeue (NOT a fix verdict).
    min_minted_all = min(r["crf_n_minted_total"] for r in rows)
    non_vacuity_met = min_minted_all > 0
    # Load-bearing: ARM_1_MATURE clears the gate on >=2/3 seeds.
    readiness_met = arm_summary["ARM_1_MATURE"]["n_gate_cleared"] >= SEED_MAJORITY
    # Discrimination: ARM_0_OFF must NOT clear on a majority (reproduces 654b).
    discrimination_met = (
        arm_summary["ARM_0_OFF"]["n_gate_cleared"] < SEED_MAJORITY
    )
    # Bonus (informational): the e2-context arm.
    e2ctx_met = arm_summary["ARM_2_MATURE_E2CTX"]["n_gate_cleared"] >= SEED_MAJORITY
    any_error = any(r["error_note"] for r in rows)

    if not non_vacuity_met:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        route = "re-queue at an adequate P0 (the field never minted)"
    elif readiness_met and discrimination_met:
        outcome = "PASS"
        label = "crf_mature_pool_dynamics_matures_differentiated_pool"
        route = (
            "PASS -> queue the 654c GAP-B behavioural re-run (committed-class "
            "entropy falsifier) on the field-populated mature substrate"
        )
    elif not readiness_met:
        outcome = "FAIL"
        label = "mature_dynamics_insufficient_at_behavioural_runtime"
        route = "/failure-autopsy (mature dynamics did not mature the pool)"
    else:  # readiness_met but not discrimination_met
        outcome = "FAIL"
        label = "env_non_discriminating_arm0_off_also_matured"
        route = (
            "/failure-autopsy (ARM_0_OFF also cleared the gate -> the readiness "
            "PASS cannot be attributed to the mature dynamics; env too easy / "
            "encoder spread too high to reproduce the 654b collapsed-z_world regime)"
        )
    if any_error:
        outcome = "FAIL"

    exp_type = "v3_exq_666_arc063_crf_mature_pool_readiness"
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
        "supersedes": None,
        "metrics": {
            "arm_summary": arm_summary,
            "readiness_met_arm1_mature": readiness_met,
            "discrimination_met_arm0_off_does_not_mature": discrimination_met,
            "e2ctx_met_arm2": e2ctx_met,
            "non_vacuity_met_all_cells_minted": non_vacuity_met,
            "min_n_minted_all_cells": min_minted_all,
            "dist_floor": DIST_FLOOR,
            "frac_floor": FRAC_FLOOR,
            "seed_majority": SEED_MAJORITY,
        },
        "arm_results": rows,
        "interpretation": {
            "label": label,
            "route": route,
            "preconditions": [
                {
                    "name": "field_minted_all_cells",
                    "kind": "readiness",
                    "description": (
                        "every gate-relevant cell must mint >=1 rule so a "
                        "below-gate reading is fix-insufficiency, not "
                        "field-not-engaged. Below floor -> substrate_not_ready_requeue."
                    ),
                    "control": "all (arm x seed) cells driven over the 654b hazard env",
                    "measured": min_minted_all,
                    "threshold": 1,
                    "direction": "lower",
                    "met": bool(non_vacuity_met),
                },
            ],
            "criteria_non_degenerate": {
                "readiness_gate_discriminates_vs_arm0_off": bool(discrimination_met),
                "field_engaged_all_cells": bool(non_vacuity_met),
            },
            "criteria": [
                {
                    "name": "ARM_1_MATURE_clears_readiness_gate",
                    "load_bearing": True,
                    "passed": bool(readiness_met),
                },
                {
                    "name": "ARM_0_OFF_reproduces_654b_signature",
                    "load_bearing": False,
                    "passed": bool(discrimination_met),
                },
                {
                    "name": "ARM_2_MATURE_E2CTX_clears_gate",
                    "load_bearing": False,
                    "passed": bool(e2ctx_met),
                },
            ],
        },
        "elapsed_seconds": elapsed,
        "notes": (
            "ARC-063 mature-pool substrate-readiness diagnostic (claim_ids=[], "
            "non_contributory -- does NOT weight governance). Validates the "
            "2026-06-11 amend (failure_autopsy_V3-EXQ-654b_2026-06-11): "
            "crf_mature_pool_dynamics (conflict-gate + retire-churn + mint-block + "
            "mint-youth protection) + crf_context_from_e2_world_forward. The "
            "CRF-readiness gate (crf_max_pairwise_rule_dist > 0.1 AND "
            "crf_frac_active >= 0.30) on ARM_1_MATURE vs the legacy ARM_0_OFF "
            "654b signature (dist ~0.0 / frac_active < 0.30). PASS clears the gate "
            "before the 654c GAP-B behavioural re-run is scored; the GAP-B "
            "falsifier (MECH-309 / ARC-062) is the governance-weighting successor, "
            "queued separately. MECH-309/ARC-062/ARC-063 stay candidate / "
            "substrate_ceiling / v3_pending / pending_retest_after_substrate."
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
            f"minted={summ['n_minted']}/{summ['n_seeds']} "
            f"mean_frac_active={summ['mean_frac_active']:.3f} "
            f"mean_dist={summ['mean_max_pairwise_rule_dist']:.3f}",
            flush=True,
        )
    print(f"Result written to: {out_path}", flush=True)
    return manifest["outcome"], str(out_path)


if __name__ == "__main__":
    _dry = "--dry-run" in sys.argv
    _outcome, _path = main(dry_run=_dry)
    emit_outcome(outcome=_outcome, manifest_path=_path)
