"""
Shared helpers for goal_pipeline:GAP-4 Tier-1 StepHarness retest cohort.

Operating substrate (post GAP-3 / V3-EXQ-582a):
  drive_floor=0.9, drive_ema_alpha=1.0 (Option 1 OFF),
  REEConfig.goal_stream() bundle (MECH-307 + MECH-295 + schema wanting),
  post-540e relaxed MECH-295 activation floors, use_dacc=True.

2026-05-29 rebuild (post V3-EXQ-490g-cohort autopsy Fork A):
  - cfg.use_dacc=True is now UNCONDITIONAL in build_config (was nested in the
    gap4_operating=True branch only). Closes the 483c primary diagnosis:
    agent.dacc=None across all 12 runs because callers used gap4_operating
    arms but downstream config wiring left use_dacc unset under the
    REEConfig.goal_stream + arm.extra_config composition path. Every cohort
    experiment now gets dACC instantiation without per-script opt-in.
  - evaluate_tier1_cohort C3_lift_vs_baseline metric DEFAULT switched from
    approach_commit_rate (saturates at 1.0 in OFF_OFF baseline under
    drive_floor=0.9 + goal_stream + reef -- no headroom for lift per 483c
    OFF_OFF=ON_OFF=1.0 byte-identical observation) to goal_norm_peak delta
    (substrate-side, cross-claim-comparable; 483c/524a manifests show
    range 0.09-0.36 under realistic substrate firing).
  - SD-037-specific scripts can override via c3_lift_metric=
    "override_signal_nonzero_steps" at the evaluate_tier1_cohort call site
    -- measures the primary PAG override pathway directly. eval_tier1 now
    populates this metric automatically (broadcast_override.override_signal
    > 1e-3 step count).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from _harness import StepHarness, StepHooks
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.residue.field import VALENCE_WANTING
from ree_core.utils.config import REEConfig

DRIVE_FLOOR_OPERATING = 0.9
DRIVE_EMA_ALPHA_OFF = 1.0
SEEDS_DEFAULT = [42, 7, 19]
WARMUP_EPISODES_DEFAULT = 50
EVAL_EPISODES_DEFAULT = 10
STEPS_PER_EPISODE_DEFAULT = 200

WORLD_DIM = 32
SELF_DIM = 32
HARM_DIM = 32
HARM_A_DIM = 16
HARM_HISTORY_LEN = 10

APPROACH_WANTING_THRESH = 0.05
TIER1_CUE_FIRES_MIN = 1
TIER1_DACC_BIAS_MIN = 1
TIER1_APPROACH_COMMIT_MIN = 1
TIER1_GOAL_ACTIVE_FRAC_MIN = 0.05
TIER1_SEEDS_PASS_MIN = 2
# Default C3_lift_vs_baseline metric is goal_norm_peak delta -- substrate-side,
# cross-claim-comparable. approach_commit_rate saturates at 1.0 in OFF_OFF
# baseline under drive_floor=0.9 + goal_stream + reef (per 483c/524a autopsies),
# so it has no headroom. Per-script callers can override c3_lift_metric to
# "override_signal_nonzero_steps" (SD-037-specific) at the call site.
TIER1_GOAL_NORM_PEAK_DELTA = 0.01
TIER1_OVERRIDE_SIGNAL_DELTA = 1
DEFAULT_C3_LIFT_METRIC = "goal_norm_peak_delta"

ENV_FISHTANK_KWARGS: Dict[str, Any] = dict(
    size=10,
    num_hazards=3,
    num_resources=5,
    hazard_harm=0.05,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    use_proxy_fields=True,
    toroidal=False,
    harm_history_len=10,
    limb_damage_enabled=True,
    damage_increment=0.15,
    failure_prob_scale=0.3,
    heal_rate=0.002,
    n_landmarks_b=2,
)

ENV_REEF_KWARGS: Dict[str, Any] = dict(
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
)

WF_BUF_MAX = 2000
HARM_EVAL_BUF_MAX = 2000
BATCH_SIZE = 32
LR_E1 = 1e-4
LR_E2_WF = 3e-4
LR_E3_HARM = 1e-3
LR_ENC_AUX = 5e-4


@dataclass
class ArmSpec:
    arm_id: str
    gap4_operating: bool = False
    use_gabaergic_decay: bool = False
    use_pag_freeze_gate: bool = False
    use_broadcast_override: bool = False
    extra_config: Dict[str, Any] = field(default_factory=dict)


def make_env(seed: int, env_kwargs: Optional[Dict[str, Any]] = None) -> CausalGridWorldV2:
    kw = dict(env_kwargs or ENV_FISHTANK_KWARGS)
    return CausalGridWorldV2(seed=seed, **kw)


def build_config(env: CausalGridWorldV2, arm: ArmSpec) -> REEConfig:
    if arm.gap4_operating:
        cfg = REEConfig.goal_stream(
            body_obs_dim=env.body_obs_dim,
            world_obs_dim=env.world_obs_dim,
            action_dim=env.action_dim,
            alpha_world=0.9,
            world_dim=WORLD_DIM,
            self_dim=SELF_DIM,
            drive_weight=2.0,
            goal_weight=0.5,
            benefit_threshold=0.1,
            use_mech307=True,
            use_consumer_conjunction_read=True,
            use_resource_encoder=True,
            drive_floor=DRIVE_FLOOR_OPERATING,
            drive_ema_alpha=DRIVE_EMA_ALPHA_OFF,
        )
        cfg.mech295_min_drive_to_fire = 0.01
        cfg.mech295_min_z_goal_norm_to_fire = 0.005
        cfg.mech295_drive_to_liking_gain = 2.0
        cfg.mech295_liking_to_approach_cue_gain = 0.5
        cfg.use_e2_harm_a = True
        cfg.residue.valence_enabled = True
    else:
        cfg = REEConfig.from_dims(
            body_obs_dim=env.body_obs_dim,
            world_obs_dim=env.world_obs_dim,
            action_dim=env.action_dim,
            alpha_world=0.9,
            world_dim=WORLD_DIM,
            self_dim=SELF_DIM,
            use_harm_stream=True,
            z_harm_dim=HARM_DIM,
            use_affective_harm_stream=True,
            z_harm_a_dim=HARM_A_DIM,
            harm_history_len=HARM_HISTORY_LEN,
            use_resource_proximity_head=True,
            resource_proximity_weight=0.5,
            benefit_eval_enabled=True,
            benefit_weight=1.0,
            z_goal_enabled=True,
            goal_weight=0.5,
            drive_weight=2.0,
            drive_floor=0.0,
            drive_ema_alpha=DRIVE_EMA_ALPHA_OFF,
            limb_damage_enabled=bool(
                env_kwargs_or_default(env).get("limb_damage_enabled", False)
            ),
            damage_increment=float(env_kwargs_or_default(env).get("damage_increment", 0.15)),
            failure_prob_scale=float(
                env_kwargs_or_default(env).get("failure_prob_scale", 0.3)
            ),
            heal_rate=float(env_kwargs_or_default(env).get("heal_rate", 0.002)),
        )
        cfg.e3.goal_weight = float(cfg.goal.goal_weight)
        cfg.residue.valence_enabled = True

    cfg.e3.commitment_threshold = 0.5
    cfg.heartbeat.beta_gate_bistable = True
    cfg.harm_descending_mod_enabled = True
    cfg.descending_attenuation_factor = 0.5
    # use_dacc is the GAP-4 cohort default per the 2026-05-29 V3-EXQ-490g-cohort
    # autopsy Fork A library rebuild. Closes the 483c primary diagnosis (agent.dacc
    # is None -> C2_dacc_bias=0 unconditionally). Applies to both gap4_operating
    # branches so every cohort experiment gets dACC instantiation without per-script
    # opt-in. arm.extra_config can still override (e.g. {"use_dacc": False} for an
    # ablation arm).
    cfg.use_dacc = True
    cfg.use_gabaergic_decay = bool(arm.use_gabaergic_decay)
    cfg.use_pag_freeze_gate = bool(arm.use_pag_freeze_gate)
    cfg.use_broadcast_override = bool(arm.use_broadcast_override)

    for key, val in arm.extra_config.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)

    return cfg


def env_kwargs_or_default(env: CausalGridWorldV2) -> Dict[str, Any]:
    return getattr(env, "_exq_env_kwargs", ENV_FISHTANK_KWARGS)


def _approach_commit(agent: REEAgent) -> bool:
    if not bool(getattr(agent.beta_gate, "is_elevated", False)):
        return False
    if agent._current_latent is None:
        return False
    z = agent._current_latent.z_world
    with torch.no_grad():
        v = agent.residue_field.evaluate_valence(z)
    wanting_amp = float(v[0, VALENCE_WANTING].item())
    return wanting_amp > APPROACH_WANTING_THRESH


def _dacc_bias_norm(agent: REEAgent) -> float:
    if agent.dacc is None:
        return 0.0
    bundle = getattr(agent.dacc, "_last_bundle", None)
    if bundle is None:
        return 0.0
    sb = bundle.get("mode_ev")
    if sb is None:
        sb = bundle.get("harm_interaction")
    if sb is None:
        return 0.0
    try:
        return float(torch.as_tensor(sb).norm().item())
    except Exception:
        return 0.0


def _override_signal_value(agent: REEAgent) -> float:
    """SD-037 BroadcastOverrideRegulator override_signal readout; 0.0 in OFF arms."""
    bo = getattr(agent, "broadcast_override", None)
    if bo is None:
        return 0.0
    return float(getattr(bo, "override_signal", 0.0))


def _entropy(counts: Dict[int, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0.0:
            h -= p * math.log(p)
    return float(h)


def warmup_train(
    agent: REEAgent,
    env: CausalGridWorldV2,
    *,
    num_episodes: int,
    steps_per_episode: int,
    label: str,
    progress_total_episodes: Optional[int] = None,
) -> Dict[str, float]:
    progress_denom = progress_total_episodes or num_episodes
    device = agent.device
    e1_optimizer = optim.Adam(agent.e1.parameters(), lr=LR_E1)
    e2_wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters()),
        lr=LR_E2_WF,
    )
    harm_eval_optimizer = optim.Adam(agent.e3.harm_eval_head.parameters(), lr=LR_E3_HARM)
    aux_params = list(agent.latent_stack.parameters())
    aux_optimizer = optim.Adam(aux_params, lr=LR_ENC_AUX)

    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_eval_buf: List[Tuple[torch.Tensor, torch.Tensor]] = []
    harness = StepHarness(agent, env, train_mode=True)

    agent.train()
    for ep in range(num_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        harness.reset()
        z_world_prev = None
        action_prev = None

        for _ in range(steps_per_episode):
            result = harness.step(obs_dict)
            latent = result.latent
            z_world_curr = latent.z_world.detach()

            rv = result.next_obs_dict.get("resource_field_view")
            if isinstance(rv, torch.Tensor):
                prox_t = float(rv.max().item())
            else:
                prox_t = float(np.max(rv)) if rv is not None else 0.0

            aux_terms: List[torch.Tensor] = []
            prox_target_t = torch.tensor([[prox_t]], device=device)
            prox_loss = agent.compute_resource_proximity_loss(prox_target_t, latent)
            if prox_loss is not None and prox_loss.requires_grad:
                aux_terms.append(prox_loss)
            if aux_terms:
                aux_loss = sum(aux_terms)
                aux_optimizer.zero_grad()
                aux_loss.backward(retain_graph=False)
                torch.nn.utils.clip_grad_norm_(aux_params, 1.0)
                aux_optimizer.step()

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > WF_BUF_MAX:
                    wf_buf = wf_buf[-WF_BUF_MAX:]

            harm_target = abs(float(result.harm_signal)) if float(result.harm_signal) < 0 else 0.0
            harm_eval_buf.append((z_world_curr.cpu(), torch.tensor([harm_target])))
            if len(harm_eval_buf) > HARM_EVAL_BUF_MAX:
                harm_eval_buf = harm_eval_buf[-HARM_EVAL_BUF_MAX:]

            if len(wf_buf) >= BATCH_SIZE:
                idxs = torch.randperm(len(wf_buf))[:BATCH_SIZE].tolist()
                zw_b = torch.cat([wf_buf[i][0] for i in idxs]).to(device)
                a_b = torch.cat([wf_buf[i][1] for i in idxs]).to(device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(device)
                wf_pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(wf_pred, zw1_b)
                if wf_loss.requires_grad:
                    e2_wf_optimizer.zero_grad()
                    wf_loss.backward()
                    e2_wf_optimizer.step()
                with torch.no_grad():
                    agent.e3.update_running_variance((wf_pred.detach() - zw1_b).detach())

            if len(harm_eval_buf) >= BATCH_SIZE:
                idxs = torch.randperm(len(harm_eval_buf))[:BATCH_SIZE].tolist()
                zw_b = torch.cat([harm_eval_buf[i][0] for i in idxs]).to(device)
                ht_b = torch.cat([harm_eval_buf[i][1] for i in idxs]).to(device)
                hp = agent.e3.harm_eval(zw_b)
                he_loss = F.mse_loss(hp.squeeze(), ht_b.squeeze())
                if he_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    he_loss.backward()
                    harm_eval_optimizer.step()

            if len(agent._world_experience_buffer) >= 2:
                e1_loss = agent.compute_prediction_loss()
                if e1_loss.requires_grad:
                    e1_optimizer.zero_grad()
                    e1_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                    e1_optimizer.step()

            z_world_prev = z_world_curr
            action_prev = result.action.detach()
            obs_dict = result.next_obs_dict
            if result.done:
                break

        if (ep + 1) % 10 == 0 or ep + 1 == num_episodes:
            print(
                f"  [train] {label} ep {ep + 1}/{progress_denom}",
                flush=True,
            )

    return {"warmup_episodes": float(num_episodes)}


def eval_tier1(
    agent: REEAgent,
    env: CausalGridWorldV2,
    *,
    num_episodes: int,
    steps_per_episode: int,
    seed: int,
    arm_label: str,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "arm": arm_label,
        "seed": int(seed),
        "approach_commit_steps": 0,
        "total_eval_steps": 0,
        "dacc_bias_nonzero_steps": 0,
        "override_signal_nonzero_steps": 0,
        "bridge_cue_fires": 0,
        "bridge_write_fires": 0,
        "goal_active_steps": 0,
        "resource_contacts": 0,
        "action_counts": {},
    }

    def on_post_step(*, agent, latent, action, obs_dict, ticks, step, **kwargs) -> None:
        metrics["total_eval_steps"] += 1
        if _approach_commit(agent):
            metrics["approach_commit_steps"] += 1
        if _dacc_bias_norm(agent) > 1e-6:
            metrics["dacc_bias_nonzero_steps"] += 1
        if _override_signal_value(agent) > 1e-3:
            metrics["override_signal_nonzero_steps"] += 1
        if agent.goal_state is not None and agent.goal_state.is_active():
            metrics["goal_active_steps"] += 1
        br = getattr(agent, "mech295_bridge", None)
        if br is not None:
            metrics["bridge_cue_fires"] = int(getattr(br, "_n_cue_fires", 0))
            metrics["bridge_write_fires"] = int(getattr(br, "_n_write_fires", 0))

    hooks = StepHooks(on_post_step=on_post_step)
    harness = StepHarness(agent, env, train_mode=False, hooks=hooks, seed=seed)
    agent.eval()

    for ep in range(num_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        harness.reset()
        if getattr(agent, "mech295_bridge", None) is not None:
            agent.mech295_bridge._n_cue_fires = 0
            agent.mech295_bridge._n_write_fires = 0

        for _ in range(steps_per_episode):
            result = harness.step(obs_dict)
            ttype = result.info.get("transition_type", "none")
            if ttype == "resource":
                metrics["resource_contacts"] += 1
            aidx = int(result.action.argmax(dim=-1).item())
            ac = metrics["action_counts"]
            ac[aidx] = ac.get(aidx, 0) + 1
            obs_dict = result.next_obs_dict
            if result.done:
                break

    total = max(1, int(metrics["total_eval_steps"]))
    metrics["approach_commit_rate"] = float(metrics["approach_commit_steps"]) / total
    metrics["goal_active_fraction"] = float(metrics["goal_active_steps"]) / total
    metrics["action_entropy"] = _entropy(metrics["action_counts"])
    metrics["action_counts"] = {str(k): int(v) for k, v in metrics["action_counts"].items()}
    if agent.goal_state is not None:
        metrics["goal_norm_peak"] = float(getattr(agent.goal_state, "_goal_norm_peak", 0.0))
    else:
        metrics["goal_norm_peak"] = 0.0
    return metrics


def tier1_seed_pass(metrics: Dict[str, Any]) -> Dict[str, bool]:
    return {
        "C1_cue_fires": int(metrics.get("bridge_cue_fires", 0)) >= TIER1_CUE_FIRES_MIN,
        "C2_dacc_bias": int(metrics.get("dacc_bias_nonzero_steps", 0)) >= TIER1_DACC_BIAS_MIN,
        "C3_approach_commit": int(metrics.get("approach_commit_steps", 0)) >= TIER1_APPROACH_COMMIT_MIN,
        "C4_goal_active": float(metrics.get("goal_active_fraction", 0.0)) >= TIER1_GOAL_ACTIVE_FRAC_MIN,
    }


def _c3_lift_compare(
    gap4_row: Dict[str, Any],
    base_row: Dict[str, Any],
    metric: str,
) -> bool:
    """Per-seed C3 lift predicate. Caller chooses the metric per call site.

    Supported metrics:
      "goal_norm_peak_delta" (default): substrate-side, cross-claim-comparable.
          PASS when gap4.goal_norm_peak > base.goal_norm_peak + TIER1_GOAL_NORM_PEAK_DELTA.
          Has headroom because GAP-4 operating substrate amplifies seeding
          (range 0.09-0.36 observed on 483c/524a per the 2026-05-29 cluster
          autopsy) while OFF_OFF lacks the bridge / amplification stack.
      "override_signal_nonzero_steps": SD-037-specific.
          PASS when gap4.override_signal_nonzero_steps > base.override_signal_nonzero_steps
          + TIER1_OVERRIDE_SIGNAL_DELTA. OFF arms have broadcast_override=None
          so signal is 0 by construction; cleanly discriminative for SD-037
          ON arms.
      "approach_commit_rate": LEGACY. Saturates at 1.0 in OFF_OFF baseline
          under drive_floor=0.9 + goal_stream + reef (per 483c/524a autopsies).
          No headroom for lift. Retained only for back-compat / debugging.
    """
    if metric == "goal_norm_peak_delta":
        return float(gap4_row.get("goal_norm_peak", 0.0)) > (
            float(base_row.get("goal_norm_peak", 0.0)) + TIER1_GOAL_NORM_PEAK_DELTA
        )
    if metric == "override_signal_nonzero_steps":
        return int(gap4_row.get("override_signal_nonzero_steps", 0)) > (
            int(base_row.get("override_signal_nonzero_steps", 0)) + TIER1_OVERRIDE_SIGNAL_DELTA
        )
    if metric == "approach_commit_rate":
        return float(gap4_row.get("approach_commit_rate", 0)) > float(
            base_row.get("approach_commit_rate", 0)
        )
    raise ValueError(
        "Unknown c3_lift_metric '{}'. Supported: goal_norm_peak_delta, "
        "override_signal_nonzero_steps, approach_commit_rate.".format(metric)
    )


def evaluate_tier1_cohort(
    rows: List[Dict[str, Any]],
    *,
    gap4_arm_id: str,
    baseline_arm_id: Optional[str] = None,
    c3_lift_metric: str = DEFAULT_C3_LIFT_METRIC,
) -> Dict[str, Any]:
    """PASS when gap4 arm clears C1-C4 in >= TIER1_SEEDS_PASS_MIN seeds and beats baseline on C3 if set.

    c3_lift_metric default is "goal_norm_peak_delta" (substrate-side,
    cross-claim-comparable; chosen post-2026-05-29 V3-EXQ-490g-cohort autopsy
    after approach_commit_rate was shown to ceiling-saturate). SD-037-specific
    scripts should pass c3_lift_metric="override_signal_nonzero_steps" to
    measure the primary PAG override pathway directly. See _c3_lift_compare.
    """
    gap4_rows = [r for r in rows if r.get("arm") == gap4_arm_id]
    base_rows = [r for r in rows if r.get("arm") == baseline_arm_id] if baseline_arm_id else []

    per_seed = [tier1_seed_pass(r) for r in gap4_rows]
    c1 = sum(1 for p in per_seed if p["C1_cue_fires"]) >= TIER1_SEEDS_PASS_MIN
    c2 = sum(1 for p in per_seed if p["C2_dacc_bias"]) >= TIER1_SEEDS_PASS_MIN
    c3_direct = sum(1 for p in per_seed if p["C3_approach_commit"]) >= TIER1_SEEDS_PASS_MIN
    c4 = sum(1 for p in per_seed if p["C4_goal_active"]) >= TIER1_SEEDS_PASS_MIN

    c3_lift = True
    lifts = 0
    if baseline_arm_id and base_rows:
        for g in gap4_rows:
            seed = g.get("seed")
            b = next((x for x in base_rows if x.get("seed") == seed), None)
            if b is None:
                continue
            if _c3_lift_compare(g, b, c3_lift_metric):
                lifts += 1
        c3_lift = lifts >= TIER1_SEEDS_PASS_MIN

    passed = bool(c1 and c2 and c3_direct and c4 and c3_lift)
    return {
        "pass": passed,
        "C1_cue_fires": c1,
        "C2_dacc_bias": c2,
        "C3_approach_commit": c3_direct,
        "C3_lift_vs_baseline": c3_lift,
        "C3_lift_count": lifts,
        "C3_lift_metric": c3_lift_metric,
        "C4_goal_active": c4,
        "gap4_arm_id": gap4_arm_id,
        "baseline_arm_id": baseline_arm_id,
    }


def run_seed_arm(
    seed: int,
    arm: ArmSpec,
    *,
    env_kwargs: Optional[Dict[str, Any]] = None,
    warmup_episodes: int = WARMUP_EPISODES_DEFAULT,
    eval_episodes: int = EVAL_EPISODES_DEFAULT,
    steps_per_episode: int = STEPS_PER_EPISODE_DEFAULT,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env = make_env(seed, env_kwargs)
    env._exq_env_kwargs = dict(env_kwargs or ENV_FISHTANK_KWARGS)
    cfg = build_config(env, arm)
    agent = REEAgent(cfg)
    label = f"seed={seed} arm={arm.arm_id}"
    print(f"Seed {seed} Condition {arm.arm_id}", flush=True)
    total_episodes = warmup_episodes + eval_episodes
    warmup_train(
        agent,
        env,
        num_episodes=warmup_episodes,
        steps_per_episode=steps_per_episode,
        label=label,
        progress_total_episodes=total_episodes,
    )
    for ep in range(eval_episodes):
        if (ep + 1) == eval_episodes:
            print(
                f"  [train] {label} ep {warmup_episodes + ep + 1}/{total_episodes}",
                flush=True,
            )
    metrics = eval_tier1(
        agent,
        env,
        num_episodes=eval_episodes,
        steps_per_episode=steps_per_episode,
        seed=seed,
        arm_label=arm.arm_id,
    )
    checks = tier1_seed_pass(metrics)
    passed = all(checks.values())
    print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)
    metrics["tier1_checks"] = checks
    metrics["seed_pass"] = passed
    return metrics
