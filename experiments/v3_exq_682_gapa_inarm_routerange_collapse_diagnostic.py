"""V3-EXQ-682: GAP-A in-arm route-range collapse diagnostic.

CLAIM-FREE diagnostic that locates WHY V3-EXQ-569g applied
modulatory_channel_route_range = 0.0 at the committed-selection site in its
falsifier arms despite the readiness probe certifying ARM_1 = 0.18.

This is NOT the 569g falsifier (which measured selected-action entropy). This
script instruments the CONVERSION CHAIN -- at the LIVE select tick, per arm, it
captures the upstream quantities that feed the routed modulatory bias and
attributes the applied 0.0 to exactly one of four causes:

  cause-iii: routing not wired in the arm agent (config off / wrong source), OR
             cand_world_summaries returned None (e2_world_forward source inactive).
  cause-i:   the live in-arm cand_world_summaries collapse to ~0 cross-candidate
             spread at the select tick (despite the e2_world_forward source).
  cause-ii:  summaries carry spread but project_channel_range flattens it (the
             range-preserving SVD projection collapses).
  cause-iv:  everything upstream OK but the e3 selector still records 0 (deeper
             wiring -- routing folds but the recorded route_range is below floor).

DESIGN: SAME 3-arm stack as V3-EXQ-569g (route-range substrate + selection
authority ON in every arm; SD-056 contrastive ON; only candidate_summary_source
differs). The agent-build / P0 warmup / arms / arm_cell fingerprint / output
pack scaffolding are reused from 569g verbatim. The MEASUREMENT phase replaces
the entropy readout with the conversion-chain instrumentation.

  ARM_0_PROPOSER       candidate_summary_source="proposer"          temperature=1.0
  ARM_1_E2WF           candidate_summary_source="e2_world_forward"  temperature=1.0  <- under investigation
  ARM_2_MATCHED_NOISE  candidate_summary_source="proposer"          temperature=2.5

All arms: use_modulatory_channel_routing=True,
modulatory_channel_route_source="cand_world_summary",
use_modulatory_selection_authority=True, modulatory_authority_gain=0.5,
SD-056 contrastive ON.

INSTRUMENTATION (per arm, last ~MEASURE_TICKS_PER_SEED select ticks):
  At each select tick, BEFORE agent.select_action:
    1. cfg_routing_on / cfg_route_source  (CAUSE-iii config check)
    2. summaries = agent._candidate_world_summaries(candidates)  (None for
       proposer arms by design); summary_pairwise_dist = mean pairwise L2
       (CAUSE-i live-collapse check); summaries_is_none.
    3. projected = project_channel_range(summaries); projected_range =
       max-min (CAUSE-ii projection check).
    5. first_action_class_count = distinct argmax-first-action classes across K.
  AFTER agent.select_action:
    4. applied_route_range / applied_route_active from
       agent.e3.last_score_diagnostics  (confirms the 569g symptom).

ATTRIBUTION (verdict, focused on ARM_1):
  pre-registered thresholds SUMMARY_SPREAD_FLOOR=0.05 / PROJECTED_RANGE_FLOOR=0.01
  / APPLIED_ROUTE_FLOOR=0.01. cause_label routed off the ARM_1 aggregates.

outcome="PASS" means the diagnostic ran and attributed a cause (collected enough
ticks, produced a non-measurement-failure cause_label). It is NOT a substrate
verdict; it does NOT self-route to substrate_ceiling. Too few ticks -> "FAIL"
with label "diagnostic_insufficient_ticks".

Usage:
  /opt/local/bin/python3 experiments/v3_exq_682_gapa_inarm_routerange_collapse_diagnostic.py --dry-run
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
from ree_core.predictors.e3_selector import project_channel_range  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_682_gapa_inarm_routerange_collapse_diagnostic"
QUEUE_ID = "V3-EXQ-682"
CLAIM_IDS: List[str] = []  # CLAIM-FREE diagnostic
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43]
P0_WARMUP_EPISODES = 60           # SD-056 contrastive warmup (mirror 569g budget)
P1_MEASUREMENT_EPISODES = 10      # short measurement window
STEPS_PER_EPISODE = 200
MEASURE_TICKS_PER_SEED = 60       # cap captured select ticks per (arm, seed)

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_STEPS = 30
DRY_RUN_MEASURE_TICKS = 20

# Pre-registered attribution thresholds.
SUMMARY_SPREAD_FLOOR = 0.05       # the 649/569g consumed-spread floor
PROJECTED_RANGE_FLOOR = 0.01      # the 662 route_range floor
APPLIED_ROUTE_FLOOR = 0.01
SUMMARIES_NONE_FRAC_FLOOR = 0.5   # > this -> e2_world_forward source inactive
MIN_TICKS_FOR_VALID = 10          # below this -> diagnostic_insufficient_ticks

# SD-056 online contrastive training (mirror V3-EXQ-569g harness).
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
E2_TRAIN_EVERY_K_TICKS = 1
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0

# V3-EXQ-662-validated route-range substrate (ON all arms; mirror 569g).
MODULATORY_AUTHORITY_GAIN = 0.5
MODULATORY_ROUTE_MIN_RANGE_FLOOR = 1e-6
MATCHED_ENTROPY_TEMPERATURE = 2.5

# Behavioural-diversity env: SD-054 reef-bipartite hazard layout (mirror 569g).
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
        "label": "routerange_summary_source_proposer_collapsed_baseline",
        "candidate_summary_source": "proposer",
        "temperature": 1.0,
    },
    {
        "arm_id": "ARM_1_E2WF",
        "label": "routerange_summary_source_e2_world_forward_under_investigation",
        "candidate_summary_source": "e2_world_forward",
        "temperature": 1.0,
    },
    {
        "arm_id": "ARM_2_MATCHED_NOISE",
        "label": "routerange_proposer_matched_entropy_temperature_control",
        "candidate_summary_source": "proposer",
        "temperature": MATCHED_ENTROPY_TEMPERATURE,
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """Main-path SP-CEM stack with the SHARED E3-side bias channels (lateral_pfc +
    mech295) ON and candidate_summary_source set per arm. SD-056 contrastive is
    ENABLED on every arm with the rollout-norm clamp ON. The V3-EXQ-662-validated
    route-range substrate is ON for EVERY arm. Mirrors V3-EXQ-569g _make_agent."""
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
        # --- V3-EXQ-662-validated route-range substrate (ON all arms) ---
        use_modulatory_channel_routing=True,
        modulatory_channel_route_source="cand_world_summary",
        modulatory_channel_route_weight=1.0,
        modulatory_channel_route_min_range_floor=MODULATORY_ROUTE_MIN_RANGE_FLOOR,
        use_modulatory_selection_authority=True,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
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


def _mean_pairwise_l2(summ: torch.Tensor) -> float:
    """Mean pairwise L2 over the K rows of a [K, D] tensor."""
    summ = summ.detach()
    if summ.dim() == 1:
        return 0.0
    k = summ.shape[0]
    if k < 2:
        return 0.0
    total = 0.0
    n = 0
    for i in range(k):
        for j in range(i + 1, k):
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
# Per-(seed, arm) runner -- the conversion-chain instrumentation
# ---------------------------------------------------------------------------

def _run_seed_arm(
    arm: Dict[str, Any],
    seed: int,
    p0_episodes: int,
    p1_episodes: int,
    steps_per_episode: int,
    measure_ticks_cap: int,
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

    # CAUSE-iii config check (read once; stable across the run).
    cfg_routing_on = bool(getattr(agent.config, "use_modulatory_channel_routing", False))
    cfg_route_source = str(getattr(agent.config, "modulatory_channel_route_source", "none"))

    # Per-tick captured quantities (P1 only).
    summary_pairwise_dists: List[float] = []
    projected_ranges: List[float] = []
    applied_route_ranges: List[float] = []
    applied_route_actives: List[float] = []
    first_action_class_counts: List[float] = []
    n_summaries_none = 0
    n_measure_ticks = 0
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

            do_measure = (
                is_p1
                and candidates is not None
                and len(candidates) >= 2
                and n_measure_ticks < measure_ticks_cap
            )

            # CONVERSION-CHAIN CAPTURE -- BEFORE select_action.
            cap_summ_dist: Optional[float] = None
            cap_proj_range: Optional[float] = None
            cap_summ_none: Optional[bool] = None
            cap_first_action_classes: Optional[int] = None
            if do_measure:
                # (5) distinct argmax-first-action classes across K candidates.
                pre_classes = [_trajectory_first_action_class(t) for t in candidates]
                cap_first_action_classes = int(len(set(pre_classes)))
                # (2) the LIVE in-arm cand_world_summaries tensor.
                summaries = agent._candidate_world_summaries(candidates)
                if summaries is None:
                    cap_summ_none = True
                else:
                    cap_summ_none = False
                    if torch.isfinite(summaries).all():
                        cap_summ_dist = _mean_pairwise_l2(summaries)
                        # (3) the range-preserving projection of the summaries.
                        projected = project_channel_range(summaries)
                        if torch.isfinite(projected).all() and projected.numel() > 0:
                            cap_proj_range = float(
                                projected.max().item() - projected.min().item()
                            )

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

            # CONVERSION-CHAIN CAPTURE -- AFTER select_action (applied route range).
            if do_measure:
                rr = float(
                    agent.e3.last_score_diagnostics.get(
                        "modulatory_channel_route_range", 0.0
                    )
                )
                ra = bool(
                    agent.e3.last_score_diagnostics.get(
                        "modulatory_channel_route_active", False
                    )
                )
                if math.isfinite(rr):
                    applied_route_ranges.append(rr)
                    applied_route_actives.append(1.0 if ra else 0.0)
                    if cap_summ_none:
                        n_summaries_none += 1
                    if cap_summ_dist is not None and math.isfinite(cap_summ_dist):
                        summary_pairwise_dists.append(cap_summ_dist)
                    if cap_proj_range is not None and math.isfinite(cap_proj_range):
                        projected_ranges.append(cap_proj_range)
                    if cap_first_action_classes is not None:
                        first_action_class_counts.append(float(cap_first_action_classes))
                    n_measure_ticks += 1

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

    summaries_none_frac = (
        float(n_summaries_none) / float(n_measure_ticks)
        if n_measure_ticks > 0 else 0.0
    )

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": int(seed),
        "candidate_summary_source": arm["candidate_summary_source"],
        "temperature": arm_temperature,
        "n_measure_ticks": int(n_measure_ticks),
        "n_contrastive_steps": int(n_contrastive_steps),
        "error_note": error_note,
        # CAUSE-iii config check.
        "cfg_routing_on": cfg_routing_on,
        "cfg_route_source": cfg_route_source,
        # CAUSE-i: live in-arm cand_world_summaries spread.
        "summary_pairwise_dist_mean": round(_mean(summary_pairwise_dists), 6),
        "summaries_none_frac": round(summaries_none_frac, 6),
        # CAUSE-ii: range-preserving projection output range.
        "projected_range_mean": round(_mean(projected_ranges), 6),
        # The 569g symptom: applied route range recorded by the e3 selector.
        "applied_route_range_mean": round(_mean(applied_route_ranges), 6),
        "applied_route_active_frac": round(_mean(applied_route_actives), 6),
        # Monostrategy indicator.
        "first_action_class_count_mean": round(_mean(first_action_class_counts), 6),
    }


# ---------------------------------------------------------------------------
# Attribution (the diagnostic verdict -- focused on ARM_1)
# ---------------------------------------------------------------------------

def _arm_rows(rows: List[Dict[str, Any]], arm_id: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("arm_id") == arm_id]


def _mean_key(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [float(r.get(key, 0.0)) for r in rows]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _sum_key_int(rows: List[Dict[str, Any]], key: str) -> int:
    return int(sum(int(r.get(key, 0)) for r in rows))


def _classify_arm1(arm1: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """Attribute the ARM_1 applied 0.0 to exactly one cause."""
    n_ticks = _sum_key_int(arm1, "n_measure_ticks")
    if n_ticks < MIN_TICKS_FOR_VALID:
        return "diagnostic_insufficient_ticks", {"n_measure_ticks": n_ticks}

    cfg_routing_on = all(bool(r.get("cfg_routing_on", False)) for r in arm1)
    cfg_route_source = arm1[0].get("cfg_route_source", "none") if arm1 else "none"
    summaries_none_frac = _mean_key(arm1, "summaries_none_frac")
    summary_pairwise_dist_mean = _mean_key(arm1, "summary_pairwise_dist_mean")
    projected_range_mean = _mean_key(arm1, "projected_range_mean")
    applied_route_range_mean = _mean_key(arm1, "applied_route_range_mean")

    detail = {
        "cfg_routing_on": cfg_routing_on,
        "cfg_route_source": cfg_route_source,
        "summaries_none_frac": round(summaries_none_frac, 6),
        "summary_pairwise_dist_mean": round(summary_pairwise_dist_mean, 6),
        "projected_range_mean": round(projected_range_mean, 6),
        "applied_route_range_mean": round(applied_route_range_mean, 6),
        "n_measure_ticks": n_ticks,
    }

    if (not cfg_routing_on) or (cfg_route_source != "cand_world_summary"):
        return "cause_iii_routing_not_wired_in_arm", detail
    if summaries_none_frac > SUMMARIES_NONE_FRAC_FLOOR:
        return "cause_iii_summaries_none_e2wf_source_inactive", detail
    if summary_pairwise_dist_mean < SUMMARY_SPREAD_FLOOR:
        return "cause_i_live_summary_recollapse", detail
    if projected_range_mean < PROJECTED_RANGE_FLOOR:
        return "cause_ii_project_channel_range_collapse", detail
    if applied_route_range_mean < APPLIED_ROUTE_FLOOR:
        return "cause_iv_applied_zero_despite_upstream_ok", detail
    return "no_collapse_reproduced", detail


def _evaluate(arm_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    arm0 = _arm_rows(arm_results, "ARM_0_PROPOSER")
    arm1 = _arm_rows(arm_results, "ARM_1_E2WF")
    arm2 = _arm_rows(arm_results, "ARM_2_MATCHED_NOISE")

    cause_label, cause_detail = _classify_arm1(arm1)

    arm1_n_ticks = _sum_key_int(arm1, "n_measure_ticks")
    diagnostic_valid = (
        arm1_n_ticks >= MIN_TICKS_FOR_VALID
        and cause_label != "diagnostic_insufficient_ticks"
    )
    overall_pass = bool(diagnostic_valid)

    arm1_summary_spread = _mean_key(arm1, "summary_pairwise_dist_mean")
    arm1_active_frac = 1.0 - _mean_key(arm1, "summaries_none_frac")
    arm1_projected = _mean_key(arm1, "projected_range_mean")
    arm1_applied = _mean_key(arm1, "applied_route_range_mean")

    arms_distinct_sources = (
        len(
            {
                r.get("candidate_summary_source")
                for r in arm_results
                if r.get("arm_id") in ("ARM_0_PROPOSER", "ARM_1_E2WF")
            }
        ) >= 1
        and arm1
        and all(r.get("candidate_summary_source") == "e2_world_forward" for r in arm1)
        and arm0
        and all(r.get("candidate_summary_source") == "proposer" for r in arm0)
    )

    preconditions = [
        {
            "name": "arm1_summaries_supra_floor_spread",
            "kind": "readiness",
            "description": (
                "ARM_1 live in-arm cand_world_summaries cross-candidate spread "
                "clears the floor (the route input must carry range). FLOOR "
                "precondition: met when measured >= threshold."
            ),
            "measured": round(arm1_summary_spread, 6),
            "threshold": SUMMARY_SPREAD_FLOOR,
            "met": bool(arm1_summary_spread >= SUMMARY_SPREAD_FLOOR),
        },
        {
            "name": "arm1_e2wf_source_active",
            "kind": "readiness",
            "description": (
                "ARM_1 _candidate_world_summaries returned non-None "
                "(e2_world_forward source active). FLOOR precondition: met when "
                "1 - summaries_none_frac >= threshold."
            ),
            "measured": round(arm1_active_frac, 6),
            "threshold": SUMMARIES_NONE_FRAC_FLOOR,
            "met": bool(arm1_active_frac >= SUMMARIES_NONE_FRAC_FLOOR),
        },
    ]

    criteria_non_degenerate = {
        "arm1_ticks_collected": bool(arm1_n_ticks >= MIN_TICKS_FOR_VALID),
        "arms_distinct_summary_source": bool(arms_distinct_sources),
    }

    return {
        "label": cause_label,
        "cause_label": cause_label,
        "cause_detail": cause_detail,
        "overall_pass": overall_pass,
        "diagnostic_valid": diagnostic_valid,
        "preconditions": preconditions,
        "criteria_non_degenerate": criteria_non_degenerate,
        "arm1_aggregates": {
            "summary_pairwise_dist_mean": round(arm1_summary_spread, 6),
            "summaries_none_frac": round(_mean_key(arm1, "summaries_none_frac"), 6),
            "projected_range_mean": round(arm1_projected, 6),
            "applied_route_range_mean": round(arm1_applied, 6),
            "applied_route_active_frac": round(_mean_key(arm1, "applied_route_active_frac"), 6),
            "first_action_class_count_mean": round(_mean_key(arm1, "first_action_class_count_mean"), 6),
            "n_measure_ticks": arm1_n_ticks,
        },
        "per_arm_summary_pairwise_dist_mean": {
            "ARM_0_PROPOSER": round(_mean_key(arm0, "summary_pairwise_dist_mean"), 6),
            "ARM_1_E2WF": round(_mean_key(arm1, "summary_pairwise_dist_mean"), 6),
            "ARM_2_MATCHED_NOISE": round(_mean_key(arm2, "summary_pairwise_dist_mean"), 6),
        },
        "per_arm_applied_route_range_mean": {
            "ARM_0_PROPOSER": round(_mean_key(arm0, "applied_route_range_mean"), 6),
            "ARM_1_E2WF": round(_mean_key(arm1, "applied_route_range_mean"), 6),
            "ARM_2_MATCHED_NOISE": round(_mean_key(arm2, "applied_route_range_mean"), 6),
        },
        "per_arm_summaries_none_frac": {
            "ARM_0_PROPOSER": round(_mean_key(arm0, "summaries_none_frac"), 6),
            "ARM_1_E2WF": round(_mean_key(arm1, "summaries_none_frac"), 6),
            "ARM_2_MATCHED_NOISE": round(_mean_key(arm2, "summaries_none_frac"), 6),
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
    measure_cap = DRY_RUN_MEASURE_TICKS if dry_run else MEASURE_TICKS_PER_SEED

    arm_results: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm['arm_id']}", flush=True)
            cell = _run_seed_arm(arm, seed, p0, p1, steps, measure_cap)
            cell["arm_fingerprint"] = compute_arm_fingerprint(
                config_slice={
                    "arm": {k: arm[k] for k in ("arm_id", "candidate_summary_source", "temperature")},
                    "env_kwargs": ENV_KWARGS,
                    "sd056_weight": SD056_WEIGHT,
                    "route_range_substrate": {
                        "use_modulatory_channel_routing": True,
                        "modulatory_channel_route_source": "cand_world_summary",
                        "use_modulatory_selection_authority": True,
                        "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
                    },
                    "p0_episodes": p0, "p1_episodes": p1, "steps_per_episode": steps,
                    "measure_ticks_cap": measure_cap,
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
        "evidence_direction": "diagnostic",
        "evidence_direction_note": (
            "CLAIM-FREE diagnostic locating WHY V3-EXQ-569g applied "
            "modulatory_channel_route_range=0.0 at the committed-selection site in "
            "its falsifier arms despite the readiness probe certifying ARM_1=0.18. "
            "Instruments the conversion chain (config -> live cand_world_summaries "
            "spread -> project_channel_range output -> applied e3 route_range) at "
            "the LIVE select tick per arm and attributes the applied 0.0 to exactly "
            "one of four causes: cause_iii (routing not wired / e2_world_forward "
            "source returned None), cause_i (live in-arm summary re-collapse), "
            "cause_ii (project_channel_range flattens the spread), cause_iv "
            "(everything upstream OK but e3 records 0). outcome=PASS means the "
            "diagnostic ran and attributed a cause (>= MIN_TICKS_FOR_VALID ticks, "
            "non-measurement-failure label); it is NOT a substrate verdict and does "
            "NOT self-route to substrate_ceiling. Too few ticks -> FAIL with label "
            "diagnostic_insufficient_ticks."
        ),
        "interpretation": {
            "label": summary["label"],
            "cause_label": summary["cause_label"],
            "cause_detail": summary["cause_detail"],
            "preconditions": summary["preconditions"],
            "criteria_non_degenerate": summary["criteria_non_degenerate"],
            "routing": {
                "cause_iii_routing_not_wired_in_arm": "routing config off / wrong source in the arm agent -- re-check _make_agent wiring",
                "cause_iii_summaries_none_e2wf_source_inactive": "e2_world_forward source returned None (e2 / current latent unavailable) -- check _candidate_world_summaries preconditions",
                "cause_i_live_summary_recollapse": "DOMINANT HYPOTHESIS: in-arm cand_world_summaries collapse at the live tick despite e2_world_forward source -- e2 under-trained or monostrategy at select",
                "cause_ii_project_channel_range_collapse": "summaries carry spread but the SVD projection flattens it -- re-examine project_channel_range",
                "cause_iv_applied_zero_despite_upstream_ok": "upstream OK but e3 selector records 0 -- deeper wiring (route fold / range floor) in e3.select",
                "no_collapse_reproduced": "the 569g symptom did NOT reproduce -- re-examine 569g harness vs this one",
                "diagnostic_insufficient_ticks": "too few measurement ticks collected -- raise P1 budget or measure cap",
            },
        },
        "dry_run": bool(dry_run),
        "config": {
            "seeds": seeds,
            "p0_warmup_episodes": p0,
            "p1_measurement_episodes": p1,
            "steps_per_episode": steps,
            "measure_ticks_per_seed": measure_cap,
            "env_kwargs": ENV_KWARGS,
            "arms": [{k: a[k] for k in ("arm_id", "label", "candidate_summary_source", "temperature")} for a in ARMS],
            "sd056_weight": SD056_WEIGHT,
            "matched_entropy_temperature": MATCHED_ENTROPY_TEMPERATURE,
            "route_range_substrate": {
                "use_modulatory_channel_routing": True,
                "modulatory_channel_route_source": "cand_world_summary",
                "modulatory_channel_route_weight": 1.0,
                "modulatory_channel_route_min_range_floor": MODULATORY_ROUTE_MIN_RANGE_FLOOR,
                "use_modulatory_selection_authority": True,
                "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
            },
            "thresholds": {
                "summary_spread_floor": SUMMARY_SPREAD_FLOOR,
                "projected_range_floor": PROJECTED_RANGE_FLOOR,
                "applied_route_floor": APPLIED_ROUTE_FLOOR,
                "summaries_none_frac_floor": SUMMARIES_NONE_FRAC_FLOOR,
                "min_ticks_for_valid": MIN_TICKS_FOR_VALID,
            },
        },
        "acceptance_criteria": {
            "diagnostic_valid": summary["diagnostic_valid"],
            "cause_label": summary["cause_label"],
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

    print(
        f"Outcome: {outcome} (cause_label={summary['cause_label']})",
        flush=True,
    )
    for k, v in manifest["acceptance_criteria"].items():
        print(f"  {k}: {v}", flush=True)
    a1 = summary["arm1_aggregates"]
    print(
        "  ARM_1: summary_pairwise_dist_mean="
        f"{a1['summary_pairwise_dist_mean']} projected_range_mean="
        f"{a1['projected_range_mean']} applied_route_range_mean="
        f"{a1['applied_route_range_mean']} summaries_none_frac="
        f"{a1['summaries_none_frac']}",
        flush=True,
    )

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-682 GAP-A in-arm route-range collapse diagnostic"
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
