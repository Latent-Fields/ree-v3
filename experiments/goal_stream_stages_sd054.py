"""
goal_stream_stages_sd054.py -- four-stage goal-stream curriculum on SD-054 reef env.

Companion harness for V3-EXQ-622 (staged decomposition of V3-EXQ-621/621a).
Unlike scaffolded_sd054_onboarding (P0 frozen goal pipeline), this curriculum
explicitly enables z_goal + per-step update_z_goal() and Stage 0 "goal feeding"
(drive_floor + transient benefit patches + rich resources).

Stages (sequential within each seed; agent weights carry forward):
  S0  goal-only / minimal hazard -- prove z_goal can become non-zero
  S1  mild hazards, hazard_food_attraction=0 -- prove goal persists under risk
  S2  hazards + food attraction, slow anneal -- wanting/liking vs harm avoidance
  S3  full SD-054 target env -- arbitration and commitment under conflict
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


def _lerp(start: float, end: float, t: float) -> float:
    t = max(0.0, min(1.0, float(t)))
    return float(start + (end - start) * t)


def _benefit_and_drive(obs_body: torch.Tensor) -> Tuple[float, float]:
    benefit = float(obs_body[11].item()) if obs_body.shape[0] > 11 else 0.0
    energy = float(obs_body[3].item()) if obs_body.shape[0] > 3 else 0.5
    drive = max(0.0, min(1.0, 1.0 - energy))
    return benefit, drive


def _set_goal_pipeline_frozen(agent, frozen: bool) -> None:
    if frozen:
        agent.config.use_mech295_liking_bridge = False
        agent.config.use_mech307_conjunction = False
    else:
        agent.config.use_mech295_liking_bridge = True
        agent.config.use_mech307_conjunction = True


def _set_bridge_anneal(agent, cfg: "GoalStreamStagesConfig", anneal_t: float) -> None:
    drive_floor = _lerp(
        cfg.s2_anneal_mech295_min_drive_to_fire_max,
        cfg.s2_anneal_mech295_min_drive_to_fire_min,
        anneal_t,
    )
    z_beta = _lerp(
        cfg.s2_anneal_mech307_z_beta_threshold_max,
        cfg.s2_anneal_mech307_z_beta_threshold_min,
        anneal_t,
    )
    bridge = getattr(agent, "mech295_bridge", None)
    if bridge is not None:
        bridge.config.min_drive_to_fire = float(drive_floor)
        bridge.config.mech307_conjunction_z_beta_threshold = float(z_beta)
    agent.config.mech295_min_drive_to_fire = float(drive_floor)
    agent.config.mech307_conjunction_z_beta_threshold = float(z_beta)


def _build_env(cfg: "GoalStreamStagesConfig", stage: str, anneal_t: float = 0.0):
    from ree_core.environment.causal_grid_world import CausalGridWorldV2

    common = dict(
        size=cfg.env_size,
        limb_damage_enabled=True,
        reef_enabled=True,
        reef_bipartite_layout=True,
        reef_bipartite_axis=cfg.reef_bipartite_axis,
        reef_bipartite_agent_band_radius=cfg.reef_bipartite_agent_band_radius,
        resource_respawn_on_consume=True,
    )
    if stage == "S0":
        return CausalGridWorldV2(
            **common,
            num_hazards=cfg.s0_num_hazards,
            num_resources=cfg.s0_num_resources,
            hazard_food_attraction=0.0,
            proximity_harm_scale=cfg.s0_proximity_harm_scale,
            reef_bipartite_agent_spawn_in_reef_half=True,
            transient_benefit_enabled=cfg.s0_transient_benefit_enabled,
            transient_benefit_multiplier=cfg.s0_transient_benefit_multiplier,
            transient_benefit_prob=cfg.s0_transient_benefit_prob,
        )
    if stage == "S1":
        return CausalGridWorldV2(
            **common,
            num_hazards=cfg.s1_num_hazards,
            num_resources=cfg.s1_num_resources,
            hazard_food_attraction=0.0,
            proximity_harm_scale=cfg.s1_proximity_harm_scale,
            reef_bipartite_agent_spawn_in_reef_half=False,
            transient_benefit_enabled=False,
        )
    if stage == "S2":
        hfa = _lerp(
            cfg.s2_anneal_hazard_food_attraction_min,
            cfg.s2_anneal_hazard_food_attraction_max,
            anneal_t,
        )
        phs = _lerp(
            cfg.s2_anneal_proximity_harm_scale_min,
            cfg.s2_anneal_proximity_harm_scale_max,
            anneal_t,
        )
        return CausalGridWorldV2(
            **common,
            num_hazards=cfg.s2_num_hazards,
            num_resources=cfg.s2_num_resources,
            hazard_food_attraction=hfa,
            proximity_harm_scale=phs,
            reef_bipartite_agent_spawn_in_reef_half=False,
            transient_benefit_enabled=False,
        )
    if stage == "S3":
        return CausalGridWorldV2(
            **common,
            num_hazards=cfg.s3_num_hazards,
            num_resources=cfg.s3_num_resources,
            hazard_food_attraction=cfg.s3_hazard_food_attraction,
            proximity_harm_scale=cfg.s3_proximity_harm_scale,
            reef_bipartite_agent_spawn_in_reef_half=False,
            transient_benefit_enabled=False,
        )
    raise ValueError(f"unknown stage: {stage!r}")


@dataclass
class GoalStreamStagesConfig:
    """Knobs for the four-stage goal-stream curriculum."""

    env_size: int = 12
    reef_bipartite_axis: str = "horizontal"
    reef_bipartite_agent_band_radius: int = 1
    steps_per_episode: int = 200

    s0_episode_budget: int = 40
    s1_episode_budget: int = 30
    s2_episode_budget: int = 30
    s3_episode_budget: int = 30

    # S0: goal feed / minimal hazard
    s0_num_hazards: int = 0
    s0_num_resources: int = 6
    s0_proximity_harm_scale: float = 0.02
    s0_transient_benefit_enabled: bool = True
    s0_transient_benefit_multiplier: float = 2.5
    s0_transient_benefit_prob: float = 0.03
    s0_drive_floor: float = 0.9

    # S1: mild hazard, no food attraction
    s1_num_hazards: int = 2
    s1_num_resources: int = 4
    s1_proximity_harm_scale: float = 0.05
    s1_drive_floor_start: float = 0.9
    s1_drive_floor_end: float = 0.2

    # S2: hazard-food attraction anneal + bridge gates
    s2_num_hazards: int = 4
    s2_num_resources: int = 5
    s2_anneal_hazard_food_attraction_min: float = 0.0
    s2_anneal_hazard_food_attraction_max: float = 0.7
    s2_anneal_proximity_harm_scale_min: float = 0.05
    s2_anneal_proximity_harm_scale_max: float = 0.1
    s2_anneal_mech295_min_drive_to_fire_max: float = 1.0
    s2_anneal_mech295_min_drive_to_fire_min: float = 0.01
    s2_anneal_mech307_z_beta_threshold_max: float = 0.6
    s2_anneal_mech307_z_beta_threshold_min: float = 0.3
    s2_drive_floor_end: float = 0.0

    # S3: full target (603b / 621 P2)
    s3_num_hazards: int = 4
    s3_num_resources: int = 5
    s3_hazard_food_attraction: float = 0.7
    s3_proximity_harm_scale: float = 0.1

    # Training
    lr_e1: float = 1e-4
    lr_e2_wf: float = 1e-3
    batch_size: int = 32
    wf_buf_max: int = 2000

    # Measurement windows
    measure_last_n_episodes: int = 5

    # Pre-registered per-stage acceptance (V3-EXQ-622)
    s0_z_goal_peak_min: float = 0.1
    s1_z_goal_median_min: float = 0.05
    s1_median_ep_len_min: float = 40.0
    s2_z_goal_median_min: float = 0.04
    s2_median_ep_len_min: float = 50.0
    s2_bridge_per_ep_min: float = 0.5
    s3_median_ep_len_min: float = 60.0
    s3_approach_commit_min: float = 0.01
    s3_bridge_per_ep_min: float = 1.0
    s3_dacc_per_ep_min: float = 0.5


@dataclass
class StageMetrics:
    stage: str
    n_episodes: int
    z_goal_norm_peak_max: float
    z_goal_norm_median_last_window: float
    mean_episode_length: float
    median_episode_length_last_window: float
    approach_commit_rate: float
    bridge_cue_fires_per_ep_mean: float
    dacc_bias_nonzero_per_ep_mean: float
    mean_harm_exposure: float
    stage_pass: bool
    stage_pass_reason: str = ""
    episode_lengths: List[int] = field(default_factory=list)
    z_goal_norm_per_episode: List[float] = field(default_factory=list)


class GoalStreamStagesRunner:
    """Runs S0 -> S1 -> S2 training, then S3 frozen-policy measurement."""

    def __init__(self, cfg: GoalStreamStagesConfig):
        self.cfg = cfg

    def run_all(self, agent, device: torch.device) -> List[StageMetrics]:
        _set_goal_pipeline_frozen(agent, frozen=False)
        if agent.goal_state is not None:
            agent.config.goal.drive_floor = float(self.cfg.s0_drive_floor)

        out: List[StageMetrics] = []
        out.append(self._run_training_stage(agent, device, "S0"))
        if agent.goal_state is not None:
            agent.config.goal.drive_floor = float(self.cfg.s1_drive_floor_start)
        out.append(self._run_training_stage(agent, device, "S1"))
        if agent.goal_state is not None:
            agent.config.goal.drive_floor = float(self.cfg.s2_drive_floor_end)
        out.append(self._run_training_stage(agent, device, "S2"))
        out.append(self._run_eval_stage(agent, device, "S3"))
        return out

    def _run_training_stage(
        self, agent, device: torch.device, stage: str
    ) -> StageMetrics:
        budget = {
            "S0": self.cfg.s0_episode_budget,
            "S1": self.cfg.s1_episode_budget,
            "S2": self.cfg.s2_episode_budget,
        }[stage]
        agent.train()
        world_dim = agent.config.latent.world_dim
        e1_opt = optim.Adam(list(agent.e1.parameters()), lr=self.cfg.lr_e1)
        wf_opt = optim.Adam(
            list(agent.e2.world_transition.parameters())
            + list(agent.e2.world_action_encoder.parameters()),
            lr=self.cfg.lr_e2_wf,
        )
        wf_buf: Deque = deque(maxlen=self.cfg.wf_buf_max)

        ep_lengths: List[int] = []
        z_goal_peaks: List[float] = []
        harm_vals: List[float] = []
        bridge_per_ep: List[float] = []
        n_eps = max(1, budget)

        for ep in range(n_eps):
            anneal_t = ep / max(1, n_eps - 1) if n_eps > 1 else 1.0
            if stage == "S1" and agent.goal_state is not None:
                agent.config.goal.drive_floor = _lerp(
                    self.cfg.s1_drive_floor_start,
                    self.cfg.s1_drive_floor_end,
                    anneal_t,
                )
            if stage == "S2":
                _set_bridge_anneal(agent, self.cfg, anneal_t)
                env = _build_env(self.cfg, "S2", anneal_t=anneal_t)
            else:
                env = _build_env(self.cfg, stage)

            ep_len, z_peak, mean_harm, bridge_fires = self._train_episode(
                agent, env, device, e1_opt, wf_opt, wf_buf, world_dim
            )
            ep_lengths.append(ep_len)
            z_goal_peaks.append(z_peak)
            harm_vals.append(mean_harm)
            bridge_per_ep.append(float(bridge_fires))

        return self._aggregate_stage_metrics(
            stage, ep_lengths, z_goal_peaks, harm_vals, bridge_per_ep
        )

    def _run_eval_stage(self, agent, device: torch.device, stage: str) -> StageMetrics:
        env = _build_env(self.cfg, "S3")
        agent.eval()
        _set_bridge_anneal(agent, self.cfg, 1.0)

        ep_lengths: List[int] = []
        z_goal_peaks: List[float] = []
        approach_steps = 0
        total_steps = 0
        bridge_total = 0
        dacc_total = 0
        harm_vals: List[float] = []

        bridge = getattr(agent, "mech295_bridge", None)
        bridge_base = int(getattr(bridge, "_n_cue_fires", 0)) if bridge else 0
        dacc = getattr(agent, "dacc", None)

        for ep in range(self.cfg.s3_episode_budget):
            ep_m = self._eval_episode(agent, env, device)
            ep_lengths.append(int(ep_m["episode_length"]))
            z_goal_peaks.append(float(ep_m["z_goal_norm_peak"]))
            approach_steps += int(ep_m["approach_commit_steps"])
            total_steps += int(ep_m["episode_length"])
            bridge_total += int(ep_m["bridge_cue_fires"])
            dacc_total += int(ep_m["dacc_bias_nonzero_steps"])
            harm_vals.append(float(ep_m["mean_harm_exposure"]))

        if bridge is not None:
            bridge_total = int(getattr(bridge, "_n_cue_fires", 0)) - bridge_base

        peak_max = float(max(z_goal_peaks)) if z_goal_peaks else 0.0
        w = min(self.cfg.measure_last_n_episodes, len(ep_lengths))
        last_lens = ep_lengths[-w:] if w else []
        last_z = z_goal_peaks[-w:] if w else []
        median_last = float(np.median(last_lens)) if last_lens else 0.0
        median_z = float(np.median(last_z)) if last_z else 0.0
        mean_len = float(np.mean(ep_lengths)) if ep_lengths else 0.0
        mean_harm = float(np.mean(harm_vals)) if harm_vals else 0.0
        n_eps = max(1, len(ep_lengths))
        approach_rate = (
            float(approach_steps) / float(total_steps) if total_steps else 0.0
        )
        bridge_per_ep = float(bridge_total) / float(n_eps)
        dacc_per_ep = float(dacc_total) / float(n_eps)
        passed, reason = self._stage_acceptance(
            stage,
            peak_max,
            median_z,
            median_last,
            mean_len,
            approach_rate,
            bridge_per_ep,
            dacc_per_ep,
        )
        return StageMetrics(
            stage=stage,
            n_episodes=len(ep_lengths),
            z_goal_norm_peak_max=peak_max,
            z_goal_norm_median_last_window=median_z,
            mean_episode_length=mean_len,
            median_episode_length_last_window=median_last,
            approach_commit_rate=approach_rate,
            bridge_cue_fires_per_ep_mean=bridge_per_ep,
            dacc_bias_nonzero_per_ep_mean=dacc_per_ep,
            mean_harm_exposure=mean_harm,
            stage_pass=passed,
            stage_pass_reason=reason,
            episode_lengths=list(ep_lengths),
            z_goal_norm_per_episode=list(z_goal_peaks),
        )

    def _aggregate_stage_metrics(
        self,
        stage: str,
        ep_lengths: List[int],
        z_goal_peaks: List[float],
        harm_vals: List[float],
        bridge_per_episode: Optional[List[float]] = None,
    ) -> StageMetrics:
        w = min(self.cfg.measure_last_n_episodes, len(ep_lengths))
        last_lens = ep_lengths[-w:] if w else []
        last_z = z_goal_peaks[-w:] if w else []
        median_last = float(np.median(last_lens)) if last_lens else 0.0
        median_z = float(np.median(last_z)) if last_z else 0.0
        peak_max = float(max(z_goal_peaks)) if z_goal_peaks else 0.0
        mean_len = float(np.mean(ep_lengths)) if ep_lengths else 0.0
        mean_harm = float(np.mean(harm_vals)) if harm_vals else 0.0

        bridge_mean = (
            float(np.mean(bridge_per_episode))
            if bridge_per_episode
            else 0.0
        )
        passed, reason = self._stage_acceptance(
            stage,
            peak_max,
            median_z,
            median_last,
            mean_len,
            0.0,
            bridge_mean,
            0.0,
        )
        return StageMetrics(
            stage=stage,
            n_episodes=len(ep_lengths),
            z_goal_norm_peak_max=peak_max,
            z_goal_norm_median_last_window=median_z,
            mean_episode_length=mean_len,
            median_episode_length_last_window=median_last,
            approach_commit_rate=0.0,
            bridge_cue_fires_per_ep_mean=bridge_mean,
            dacc_bias_nonzero_per_ep_mean=0.0,
            mean_harm_exposure=mean_harm,
            stage_pass=passed,
            stage_pass_reason=reason,
            episode_lengths=list(ep_lengths),
            z_goal_norm_per_episode=list(z_goal_peaks),
        )

    def _stage_acceptance(
        self,
        stage: str,
        z_peak_max: float,
        z_median_last: float,
        median_ep_len: float,
        mean_ep_len: float,
        approach_rate: float,
        bridge_per_ep: float,
        dacc_per_ep: float,
    ) -> Tuple[bool, str]:
        c = self.cfg
        if stage == "S0":
            ok = z_peak_max >= c.s0_z_goal_peak_min
            return ok, f"S0 z_goal_peak>={c.s0_z_goal_peak_min} (got {z_peak_max:.4f})"
        if stage == "S1":
            ok = (
                z_median_last >= c.s1_z_goal_median_min
                and median_ep_len >= c.s1_median_ep_len_min
            )
            return ok, (
                f"S1 z_median>={c.s1_z_goal_median_min} & median_len>="
                f"{c.s1_median_ep_len_min} (z={z_median_last:.4f} len={median_ep_len:.1f})"
            )
        if stage == "S2":
            ok = (
                z_median_last >= c.s2_z_goal_median_min
                and median_ep_len >= c.s2_median_ep_len_min
                and bridge_per_ep >= c.s2_bridge_per_ep_min
            )
            return ok, f"S2 persistence+bridge (z={z_median_last:.4f} len={median_ep_len:.1f})"
        if stage == "S3":
            ok = (
                median_ep_len >= c.s3_median_ep_len_min
                and (
                    approach_rate >= c.s3_approach_commit_min
                    or bridge_per_ep >= c.s3_bridge_per_ep_min
                    or dacc_per_ep >= c.s3_dacc_per_ep_min
                )
            )
            return ok, f"S3 arbitration (len={median_ep_len:.1f} approach={approach_rate:.4f})"
        return False, "unknown stage"

    def _train_episode(
        self,
        agent,
        env,
        device: torch.device,
        e1_opt,
        wf_opt,
        wf_buf: Deque,
        world_dim: int,
    ) -> Tuple[int, float, float, int]:
        _, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        z_goal_peak = 0.0
        harm_sum = 0.0
        harm_n = 0
        bridge = getattr(agent, "mech295_bridge", None)
        bridge_base = int(getattr(bridge, "_n_cue_fires", 0)) if bridge else 0

        for step in range(self.cfg.steps_per_episode):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            latent = agent.sense(obs_body, obs_world)
            z_world_curr = latent.z_world.detach()

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append(
                    (z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu())
                )

            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent)
                if ticks.get("e1_tick")
                else torch.zeros(1, world_dim, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item())

            e1_opt.zero_grad()
            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                e1_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(agent.e1.parameters()), 1.0)
                e1_opt.step()

            if len(wf_buf) >= self.cfg.batch_size:
                k = min(self.cfg.batch_size, len(wf_buf))
                idxs = torch.randperm(len(wf_buf))[:k].tolist()
                zw_b = torch.cat([wf_buf[i][0] for i in idxs]).to(device)
                a_b = torch.cat([wf_buf[i][1] for i in idxs]).to(device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(device)
                wf_pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(wf_pred, zw1_b)
                if wf_loss.requires_grad:
                    wf_opt.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters())
                        + list(agent.e2.world_action_encoder.parameters()),
                        1.0,
                    )
                    wf_opt.step()
                with torch.no_grad():
                    agent.e3.update_running_variance(
                        (wf_pred.detach() - zw1_b).detach()
                    )

            z_world_prev = z_world_curr
            action_prev = action.detach()
            _, harm_signal, done, _, obs_dict = env.step(action_idx)
            harm_sum += float(harm_signal)
            harm_n += 1

            obs_body = obs_dict["body_state"].to(device)
            benefit, drive = _benefit_and_drive(obs_body)
            agent.update_z_goal(benefit_exposure=benefit, drive_level=drive)

            if agent.goal_state is not None:
                try:
                    cur = float(agent.goal_state.goal_norm())
                except TypeError:
                    cur = float(agent.goal_state.goal_norm)
                if cur > z_goal_peak:
                    z_goal_peak = cur

            if done:
                mean_harm = harm_sum / float(max(1, harm_n))
                bridge_fires = (
                    int(getattr(bridge, "_n_cue_fires", 0)) - bridge_base if bridge else 0
                )
                return step + 1, z_goal_peak, mean_harm, bridge_fires

        mean_harm = harm_sum / float(max(1, harm_n))
        bridge_fires = (
            int(getattr(bridge, "_n_cue_fires", 0)) - bridge_base if bridge else 0
        )
        return self.cfg.steps_per_episode, z_goal_peak, mean_harm, bridge_fires

    def _eval_episode(self, agent, env, device: torch.device) -> Dict[str, Any]:
        _, obs_dict = env.reset()
        agent.reset()
        world_dim = agent.config.latent.world_dim
        z_goal_peak = 0.0
        approach_commit_steps = 0
        dacc_bias_nonzero = 0
        harm_sum = 0.0
        harm_n = 0
        bridge_base = 0
        bridge = getattr(agent, "mech295_bridge", None)
        if bridge is not None:
            bridge_base = int(getattr(bridge, "_n_cue_fires", 0))
        dacc = getattr(agent, "dacc", None)

        for step in range(self.cfg.steps_per_episode):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent)
                    if ticks.get("e1_tick")
                    else torch.zeros(1, world_dim, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)

            benefit, drive = _benefit_and_drive(obs_body)
            agent.update_z_goal(benefit_exposure=benefit, drive_level=drive)

            if agent.goal_state is not None:
                try:
                    cur = float(agent.goal_state.goal_norm())
                except TypeError:
                    cur = float(agent.goal_state.goal_norm)
                if cur > z_goal_peak:
                    z_goal_peak = cur

            bg = getattr(agent, "beta_gate", None)
            if bg is not None and getattr(bg, "is_elevated", False):
                approach_commit_steps += 1

            if dacc is not None:
                bundle = getattr(dacc, "_last_bundle", None)
                if bundle is not None:
                    sb = bundle.get("mode_ev") or bundle.get("harm_interaction")
                    if sb is not None:
                        try:
                            if float(torch.as_tensor(sb).norm().item()) > 1e-6:
                                dacc_bias_nonzero += 1
                        except Exception:
                            pass

            action_idx = int(action.argmax(dim=-1).item())
            _, harm_signal, done, _, obs_dict = env.step(action_idx)
            harm_sum += float(harm_signal)
            harm_n += 1
            ep_len = step + 1
            if done:
                break
        else:
            ep_len = self.cfg.steps_per_episode

        bridge_final = int(getattr(bridge, "_n_cue_fires", 0)) if bridge else 0
        return {
            "episode_length": ep_len,
            "z_goal_norm_peak": z_goal_peak,
            "approach_commit_steps": approach_commit_steps,
            "bridge_cue_fires": bridge_final - bridge_base,
            "dacc_bias_nonzero_steps": dacc_bias_nonzero,
            "mean_harm_exposure": harm_sum / float(max(1, harm_n)),
        }


__all__ = [
    "GoalStreamStagesConfig",
    "GoalStreamStagesRunner",
    "StageMetrics",
]
