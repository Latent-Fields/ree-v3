"""
V3-EXQ-626a: goal-pipeline developmental-window diagnostic (4-arm) -- HARNESS-FIXED.

Lettered iteration of V3-EXQ-626. Scientific question UNCHANGED (which axis --
drive-floor anneal, hazard introduction, or writer-freeze rescue -- is
load-bearing for z_goal collapse). The implementation was wrong: per
failure_autopsy_V3-EXQ-626_2026-06-01.md, the 626 bespoke episode loop
(`_run_episode`) never called `agent.update_z_goal(...)`, so `GoalState.update()`
-- the only function that seeds z_goal toward a benefit target -- was never
reached. z_goal stayed at its torch.zeros(1, goal_dim) initialisation for every
step of every arm. The 626 manifest (ARM_A z_goal_norm_median 0.0 across all
phases/seeds; C2/C3 vacuously "true" because 0 < ceiling) is fully explained by
this single missing call. 626 is therefore a Class-1 harness/measurement failure,
not a substrate formation regression (622 S0 PASS + 582a refute regression).

HARNESS FIX (the only material change vs 626)
---------------------------------------------
1. Per-step goal feed: `_run_episode` now reads benefit_exposure from
   body_state[11] and drive from energy (body_state[3]) via `_benefit_and_drive`
   -- the exact pattern the V3-EXQ-622 shared runner
   (`experiments/goal_stream_stages_sd054.py`) uses -- and calls
   `agent.update_z_goal(benefit_exposure=benefit, drive_level=drive)` every step
   in BOTH the training (P0/P1) and frozen-eval (P2) paths, before measuring
   z_goal_norm. Without this the pipeline is never driven.
2. P0 positive-control gate (NEW, adjudicating bit): ARM_A P0
   z_goal_norm_peak >= 0.1 on >= 2/3 seeds. This is the Stage-0-style positive
   control 626 lacked. It makes the harness bug structurally visible:
   - If P0 positive control PASSES (z_goal forms when fed), the "626 formation
     failure" is confirmed a harness bug; the real open question is the
     object-binding gap (514k) -> ladder Stage 1+.
   - If it still FAILS even with update_z_goal driven, formation is genuinely
     broken (would contradict 622 S0) -> /failure-autopsy on GoalState.update.
   Axis criteria (C2/C3/C4) are NOT trusted unless the positive control passes
   (they are differential criteria; with a uniformly-zero z_goal they are vacuous,
   exactly the 626 trap).

Everything else (4-arm grid, phase budgets, env build, dACC config, C1/C2/C3/C4/C5
acceptance code) is carried over from 626 unchanged so the dissociation question
is asked on a harness that can actually form z_goal.

Arms (unchanged from 626)
-------------------------
ARM_A_FORMATION_ONLY        regression guard on 622 S0 PASS. drive_floor=0.9
                            constant P0+P1+P2, HFA=0.0 constant, goal pipeline ON.
ARM_B_DRIVE_ANNEAL_ONLY     drive_floor 0.9 -> 0.2 anneal during P1, no hazard.
ARM_C_HAZARD_ONLY           drive_floor=0.9 constant, HFA 0.0 -> 0.7 anneal in P1.
ARM_D_WRITER_FROZEN_DURING_TRANSITION   freeze use_mech295 + use_mech307 during
                            P1 while annealing BOTH drive_floor and HFA; unfreeze
                            at P2 start.

3 seeds [42, 43, 44]. P0(~30 ep) + P1(~30 ep) + P2(~30 ep). 200 steps/ep.

Pre-registered acceptance criteria
----------------------------------
P0 (positive control, NEW): ARM_A P0 z_goal_norm_peak >= 0.1 on >= 2/3 seeds.
   FAIL = formation broken even when fed -> /failure-autopsy on GoalState.update.
   Required for the axis criteria (C2/C3/C4) to be trusted.
C1 (formation regression guard): ARM_A z_goal_median_last_window >= 0.05 on
   >= 2/3 seeds.
C2 (drive-axis isolation): ARM_B z_goal_median_last_window < 0.05 on >= 2/3
   seeds AND ARM_B z_goal_norm at P1 end < 0.5 * ARM_A.
C3 (hazard-axis isolation): ARM_C z_goal_median_last_window < 0.05 on >= 2/3
   seeds.
C4 (writer-freeze rescue): ARM_D z_goal_median_last_window AT P2 START >=
   0.5 * ARM_A.
C5 (consumer readout under non-trivial z_goal): in the arm with highest z_goal
   during P2 (ARM_A or ARM_D), dacc_bias_nonzero_steps_per_episode_mean >= 1.0
   on >= 2/3 seeds.

overall_pass = P0 positive control AND C1 AND (C2 OR C3 OR C4) AND C5.

claim_ids = []  (substrate-readiness / harness diagnostic; NOT load-bearing
evidence weighting per CLAIM_IDS Accuracy Rule). experiment_purpose = "diagnostic".
supersedes = V3-EXQ-626 (queue) / v3_exq_626_goal_pipeline_developmental_window_diagnostic
(manifest experiment_type).

References:
- failure_autopsy_V3-EXQ-626_2026-06-01.md
- goal_stream_repair_diagnostic_ladder_2026-06-01.md (Deliverables E/F/G; F1 harness fix)
- goal_pipeline_developmental_window_diagnostic_memo_2026-06-01.md  (original 626 design)
- failure_autopsy_V3-EXQ-622_2026-06-01.md  (622 S0 positive control)
- experiments/goal_stream_stages_sd054.py    (canonical update_z_goal-driven runner)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

from experiment_protocol import emit_outcome

# -- Header constants --------------------------------------------------------
EXPERIMENT_TYPE = "v3_exq_626a_goal_pipeline_developmental_window_diagnostic"
QUEUE_ID = "V3-EXQ-626a"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"
SUPERSEDES = "v3_exq_626_goal_pipeline_developmental_window_diagnostic"
SEEDS = [42, 43, 44]

# -- Pre-registered thresholds (memo Section 3 + ladder F1 positive control) -
P0_POSITIVE_CONTROL_PEAK_FLOOR = 0.1   # ARM_A P0 z_goal_norm_peak floor (Stage-0 A0.2)
C1_Z_GOAL_MEDIAN_FLOOR = 0.05          # ARM_A regression guard floor
C2_ARM_B_Z_GOAL_MEDIAN_CEILING = 0.05  # ARM_B z_goal_median must collapse
C2_ARM_B_VS_A_P1_END_RATIO = 0.5       # ARM_B z_goal at P1 end < 0.5 * ARM_A
C3_ARM_C_Z_GOAL_MEDIAN_CEILING = 0.05  # ARM_C z_goal_median must collapse
C4_ARM_D_VS_A_P2_START_RATIO = 0.5     # ARM_D >= 0.5 * ARM_A at P2 start
C5_DACC_NONZERO_PER_EP_FLOOR = 1.0     # dacc_bias_nonzero >= 1.0/ep
C5_HIGH_Z_GOAL_THRESHOLD = 0.05        # z_goal_norm > 0.05 at commit tick

PASS_SEED_FRACTION = 2.0 / 3.0  # >= 2/3 seeds clears the per-criterion gate
LAST_WINDOW_EPISODES = 10        # window for P1 final 10 ep + P2 medians

# -- Phase budgets -----------------------------------------------------------
P0_EPISODES_DEFAULT = 30
P1_EPISODES_DEFAULT = 30
P2_EPISODES_DEFAULT = 30
STEPS_PER_EPISODE_DEFAULT = 200

# -- Common training rates (mirrors scaffolded_sd054_onboarding) -------------
LR_E1 = 1e-4
LR_E2_WF = 1e-3
BATCH_SIZE = 32
WF_BUF_MAX = 2000

# -- Env / agent dims --------------------------------------------------------
SELF_DIM = 32
WORLD_DIM = 32
HARM_DIM = 16
HARM_A_DIM = 7  # SD-022 limb damage path
HARM_HISTORY_LEN = 10

# -- dACC sub-weights (conservative defaults; same shape as V3-EXQ-490c) -----
DACC_WEIGHT = 0.5
DACC_INTERACTION_WEIGHT = 0.5


# ---------------------------------------------------------------------------
# Arm definitions
# ---------------------------------------------------------------------------


@dataclass
class ArmConfig:
    """Per-arm configuration for the 4-arm dissociation grid (memo Section 3)."""

    name: str
    # Drive-floor anneal axis: (P0_value, P1_start_value, P1_end_value, P2_value)
    drive_floor_p0: float
    drive_floor_p1_start: float
    drive_floor_p1_end: float
    drive_floor_p2: float
    # Hazard-axis anneal: same shape
    hfa_p0: float
    hfa_p1_start: float
    hfa_p1_end: float
    hfa_p2: float
    # ARM_D only: freeze use_mech295 + use_mech307 during P1
    freeze_writer_during_p1: bool


def arm_configs() -> List[ArmConfig]:
    """4-arm grid per memo Section 3 table."""
    return [
        # ARM_A: drive_floor=0.9 + HFA=0.0 throughout (regression guard).
        ArmConfig(
            name="ARM_A_FORMATION_ONLY",
            drive_floor_p0=0.9, drive_floor_p1_start=0.9,
            drive_floor_p1_end=0.9, drive_floor_p2=0.9,
            hfa_p0=0.0, hfa_p1_start=0.0, hfa_p1_end=0.0, hfa_p2=0.0,
            freeze_writer_during_p1=False,
        ),
        # ARM_B: drive_floor 0.9 -> 0.2 anneal, NO hazard.
        ArmConfig(
            name="ARM_B_DRIVE_ANNEAL_ONLY",
            drive_floor_p0=0.9, drive_floor_p1_start=0.9,
            drive_floor_p1_end=0.2, drive_floor_p2=0.2,
            hfa_p0=0.0, hfa_p1_start=0.0, hfa_p1_end=0.0, hfa_p2=0.0,
            freeze_writer_during_p1=False,
        ),
        # ARM_C: constant drive_floor=0.9, HFA 0.0 -> 0.7 anneal.
        ArmConfig(
            name="ARM_C_HAZARD_ONLY",
            drive_floor_p0=0.9, drive_floor_p1_start=0.9,
            drive_floor_p1_end=0.9, drive_floor_p2=0.9,
            hfa_p0=0.0, hfa_p1_start=0.0, hfa_p1_end=0.7, hfa_p2=0.7,
            freeze_writer_during_p1=False,
        ),
        # ARM_D: writer-freeze during P1; both drive AND hazard anneal.
        ArmConfig(
            name="ARM_D_WRITER_FROZEN_DURING_TRANSITION",
            drive_floor_p0=0.9, drive_floor_p1_start=0.9,
            drive_floor_p1_end=0.2, drive_floor_p2=0.2,
            hfa_p0=0.0, hfa_p1_start=0.0, hfa_p1_end=0.7, hfa_p2=0.7,
            freeze_writer_during_p1=True,
        ),
    ]


# ---------------------------------------------------------------------------
# Phase / arm helpers
# ---------------------------------------------------------------------------


def _lerp(start: float, end: float, t: float) -> float:
    """Linear interpolation; t clamped to [0, 1]."""
    t = max(0.0, min(1.0, float(t)))
    return float(start + (end - start) * t)


def _benefit_and_drive(obs_body: torch.Tensor) -> Tuple[float, float]:
    """
    Read the goal-pipeline feed signals from the body observation, identical to
    the V3-EXQ-622 shared runner (goal_stream_stages_sd054._benefit_and_drive):
    body_state[11] = benefit_exposure (proxy mode), body_state[3] = energy ->
    drive = 1 - energy clamped to [0, 1] (SD-012).
    """
    benefit = float(obs_body[11].item()) if obs_body.shape[0] > 11 else 0.0
    energy = float(obs_body[3].item()) if obs_body.shape[0] > 3 else 0.5
    drive = max(0.0, min(1.0, 1.0 - energy))
    return benefit, drive


def _build_env(
    size: int,
    num_hazards: int,
    num_resources: int,
    hazard_food_attraction: float,
    proximity_harm_scale: float,
    spawn_in_reef_half: bool,
    seed: int,
) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        size=size,
        num_hazards=num_hazards,
        num_resources=num_resources,
        hazard_food_attraction=hazard_food_attraction,
        proximity_harm_scale=proximity_harm_scale,
        limb_damage_enabled=True,
        reef_enabled=True,
        reef_bipartite_layout=True,
        reef_bipartite_axis="horizontal",
        reef_bipartite_agent_band_radius=1,
        reef_bipartite_agent_spawn_in_reef_half=spawn_in_reef_half,
        seed=seed,
    )


def _set_writer_frozen(agent: REEAgent, frozen: bool) -> None:
    """
    Freeze or unfreeze the goal-pipeline writer (MECH-295 bridge + MECH-307
    conjunction). The bridge reads its own config per-call so a runtime
    mutation takes effect immediately.
    """
    target = not frozen
    agent.config.use_mech295_liking_bridge = bool(target)
    agent.config.use_mech307_conjunction = bool(target)


def _set_drive_floor(agent: REEAgent, drive_floor: float) -> None:
    """Set GoalConfig.drive_floor on the agent live."""
    agent.config.goal.drive_floor = float(drive_floor)


def build_agent(seed: int, device: torch.device, world_obs_dim: int) -> REEAgent:
    """
    Build agent with use_dacc=True + goal pipeline ON. Initial drive_floor
    is overridden per phase by _set_drive_floor.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    config = REEConfig.from_dims(
        body_obs_dim=17,                  # SD-022 limb damage extends to 17
        world_obs_dim=world_obs_dim,
        action_dim=5,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        harm_dim=HARM_DIM,
        alpha_world=0.9,
        reafference_action_dim=5,
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
        drive_ema_alpha=1.0,
        drive_floor=0.9,                  # overridden per phase
        e1_goal_conditioned=True,
        # SD-022 limb damage env path.
        limb_damage_enabled=True,
        damage_increment=0.15,
        failure_prob_scale=0.3,
        heal_rate=0.002,
        # SD-032b dACC: enabled on ALL arms. dacc_weight + dacc_interaction
        # are non-zero so the bundle contributes nonzero score_bias values
        # and the C5 dacc_bias_nonzero metric is observable.
        use_dacc=True,
        dacc_weight=DACC_WEIGHT,
        dacc_interaction_weight=DACC_INTERACTION_WEIGHT,
        use_e2_harm_a=True,
        # MECH-295 + MECH-307 default ON; ARM_D toggles via runtime mutation.
        use_mech295_liking_bridge=True,
        use_mech307_conjunction=True,
    )
    config.e3.commitment_threshold = 0.5
    config.heartbeat.beta_gate_bistable = True
    agent = REEAgent(config).to(device)
    return agent


# ---------------------------------------------------------------------------
# Episode loops
# ---------------------------------------------------------------------------


def _train_step_e1_e2(
    agent: REEAgent,
    e1_opt,
    wf_opt,
    wf_buf: Deque,
    device: torch.device,
) -> None:
    """E1 prediction + E2 world-forward updates; updates E3 running variance."""
    e1_opt.zero_grad()
    e1_loss = agent.compute_prediction_loss()
    if e1_loss.requires_grad:
        e1_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(agent.e1.parameters()), 1.0)
        e1_opt.step()

    if len(wf_buf) >= BATCH_SIZE:
        k = min(BATCH_SIZE, len(wf_buf))
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
            agent.e3.update_running_variance((wf_pred.detach() - zw1_b).detach())


def _measure_step_metrics(agent: REEAgent) -> Dict[str, float]:
    """
    Read per-tick goal-pipeline + dACC + commit telemetry from the agent
    after select_action + update_z_goal.
    """
    metrics: Dict[str, float] = {
        "z_goal_norm": 0.0,
        "beta_elevated": 0.0,
        "dacc_bias_nonzero": 0.0,
    }
    goal_state = getattr(agent, "goal_state", None)
    if goal_state is not None and hasattr(goal_state, "goal_norm"):
        try:
            metrics["z_goal_norm"] = float(goal_state.goal_norm())
        except TypeError:
            metrics["z_goal_norm"] = float(goal_state.goal_norm)
    beta_gate = getattr(agent, "beta_gate", None)
    if beta_gate is not None and getattr(beta_gate, "is_elevated", False):
        metrics["beta_elevated"] = 1.0
    dacc = getattr(agent, "dacc", None)
    if dacc is not None:
        bundle = getattr(dacc, "_last_bundle", None)
        if bundle is not None:
            sb = bundle.get("mode_ev")
            if sb is None:
                sb = bundle.get("harm_interaction")
            if sb is not None:
                try:
                    if float(torch.as_tensor(sb).norm().item()) > 1e-6:
                        metrics["dacc_bias_nonzero"] = 1.0
                except Exception:
                    pass
    return metrics


def _bridge_cue_fires(agent: REEAgent) -> int:
    bridge = getattr(agent, "mech295_bridge", None)
    if bridge is None:
        return 0
    return int(getattr(bridge, "_n_cue_fires", 0))


def _run_episode(
    agent: REEAgent,
    env: CausalGridWorldV2,
    device: torch.device,
    steps_per_episode: int,
    train: bool,
    e1_opt=None,
    wf_opt=None,
    wf_buf: Optional[Deque] = None,
    world_dim: int = WORLD_DIM,
) -> Dict[str, Any]:
    """
    One episode. Returns per-episode summary metrics.

    train=True trains E1 + E2 each tick; train=False is frozen-policy eval.

    HARNESS FIX vs 626: after action selection, the goal pipeline is fed via
    agent.update_z_goal(benefit_exposure, drive_level) using body_state[11] /
    body_state[3] (the V3-EXQ-622 runner pattern), in BOTH train and eval paths.
    z_goal_norm is then measured AFTER the feed so the metric reflects the update.
    """
    _, obs_dict = env.reset()
    agent.reset()

    z_goal_peak = 0.0
    z_goal_sum = 0.0
    z_goal_steps_above = 0
    beta_steps = 0
    dacc_nonzero_steps = 0
    approach_commit_steps = 0
    approach_commit_high_z_goal_steps = 0
    ep_len = 0

    bridge_baseline = _bridge_cue_fires(agent)

    z_world_prev: Optional[torch.Tensor] = None
    action_prev: Optional[torch.Tensor] = None

    if train:
        agent.train()
    else:
        agent.eval()

    for step in range(steps_per_episode):
        obs_body = obs_dict["body_state"].to(device)
        obs_world = obs_dict["world_state"].to(device)

        if train:
            latent = agent.sense(obs_body, obs_world)
        else:
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
        z_world_curr = latent.z_world.detach()

        if train and z_world_prev is not None and action_prev is not None and wf_buf is not None:
            wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))

        ticks = agent.clock.advance()
        if train:
            e1_prior = (
                agent._e1_tick(latent)
                if ticks.get("e1_tick")
                else torch.zeros(1, world_dim, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
        else:
            with torch.no_grad():
                e1_prior = (
                    agent._e1_tick(latent)
                    if ticks.get("e1_tick")
                    else torch.zeros(1, world_dim, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)

        # HARNESS FIX (the line 626 dropped): drive the goal pipeline every step.
        # update_z_goal seeds z_goal from the current latent (z_resource if
        # available, else z_world), gated by benefit_exposure * drive. Called
        # outside any no_grad block in both train and eval, matching the 622
        # shared runner.
        benefit, drive = _benefit_and_drive(obs_body)
        agent.update_z_goal(benefit_exposure=benefit, drive_level=drive)

        per_tick = _measure_step_metrics(agent)
        z_goal_now = per_tick["z_goal_norm"]
        z_goal_sum += z_goal_now
        if z_goal_now > z_goal_peak:
            z_goal_peak = z_goal_now
        if z_goal_now > C5_HIGH_Z_GOAL_THRESHOLD:
            z_goal_steps_above += 1
        if per_tick["beta_elevated"] > 0.5:
            beta_steps += 1
            approach_commit_steps += 1
            if z_goal_now > C5_HIGH_Z_GOAL_THRESHOLD:
                approach_commit_high_z_goal_steps += 1
        if per_tick["dacc_bias_nonzero"] > 0.5:
            dacc_nonzero_steps += 1

        if train and e1_opt is not None and wf_opt is not None and wf_buf is not None:
            _train_step_e1_e2(agent, e1_opt, wf_opt, wf_buf, device)

        z_world_prev = z_world_curr
        action_prev = action.detach()
        action_idx = int(action.argmax(dim=-1).item())
        _, _harm_signal, done, _, obs_dict = env.step(action_idx)
        ep_len = step + 1
        if done:
            break

    bridge_final = _bridge_cue_fires(agent)
    return {
        "episode_length": ep_len,
        "z_goal_norm_peak": z_goal_peak,
        "z_goal_norm_mean": float(z_goal_sum / max(1, ep_len)),
        "z_goal_steps_above_0p05": z_goal_steps_above,
        "beta_elevated_steps": beta_steps,
        "dacc_bias_nonzero_steps": dacc_nonzero_steps,
        "approach_commit_steps": approach_commit_steps,
        "approach_commit_at_high_z_goal_steps": approach_commit_high_z_goal_steps,
        "bridge_cue_fires": bridge_final - bridge_baseline,
    }


# ---------------------------------------------------------------------------
# Arm runner: P0 + P1 + P2 for one seed x arm cell
# ---------------------------------------------------------------------------


@dataclass
class PhaseSummary:
    phase: str
    n_episodes: int
    z_goal_norm_peak_max: float
    z_goal_norm_median: float
    z_goal_norm_median_last_window: float
    bridge_cue_fires_per_episode_mean: float
    dacc_bias_nonzero_steps_per_episode_mean: float
    approach_commit_rate: float
    approach_commit_at_high_z_goal_rate: float
    mean_episode_length: float
    per_episode_peaks: List[float] = field(default_factory=list)


def _summarise_phase(phase: str, episodes: List[Dict[str, Any]]) -> PhaseSummary:
    n = max(1, len(episodes))
    peaks = [float(e["z_goal_norm_peak"]) for e in episodes]
    means = [float(e["z_goal_norm_mean"]) for e in episodes]
    bridges = sum(int(e["bridge_cue_fires"]) for e in episodes)
    dacc = sum(int(e["dacc_bias_nonzero_steps"]) for e in episodes)
    approach = sum(int(e["approach_commit_steps"]) for e in episodes)
    approach_hi = sum(int(e["approach_commit_at_high_z_goal_steps"]) for e in episodes)
    ep_lens = [int(e["episode_length"]) for e in episodes]
    total_steps = max(1, sum(ep_lens))

    last_window = means[-LAST_WINDOW_EPISODES:] if means else [0.0]
    median_last_window = float(np.median(last_window)) if last_window else 0.0
    median_all = float(np.median(means)) if means else 0.0
    return PhaseSummary(
        phase=phase,
        n_episodes=len(episodes),
        z_goal_norm_peak_max=float(max(peaks)) if peaks else 0.0,
        z_goal_norm_median=median_all,
        z_goal_norm_median_last_window=median_last_window,
        bridge_cue_fires_per_episode_mean=float(bridges) / float(n),
        dacc_bias_nonzero_steps_per_episode_mean=float(dacc) / float(n),
        approach_commit_rate=float(approach) / float(total_steps),
        approach_commit_at_high_z_goal_rate=(
            float(approach_hi) / float(approach) if approach > 0 else 0.0
        ),
        mean_episode_length=float(sum(ep_lens)) / float(n),
        per_episode_peaks=peaks,
    )


def run_seed_arm(
    seed: int,
    arm: ArmConfig,
    device: torch.device,
    world_obs_dim: int,
    p0_eps: int,
    p1_eps: int,
    p2_eps: int,
    steps_per_episode: int,
    total_training_eps: int,
) -> Dict[str, Any]:
    """Run all three phases for one seed x arm cell. Returns flat dict."""
    print(f"Seed {seed} Condition {arm.name}", flush=True)

    agent = build_agent(seed, device, world_obs_dim)
    world_dim = agent.config.latent.world_dim

    e1_opt = optim.Adam(list(agent.e1.parameters()), lr=LR_E1)
    wf_opt = optim.Adam(
        list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters()),
        lr=LR_E2_WF,
    )
    wf_buf: Deque = deque(maxlen=WF_BUF_MAX)
    ep_done = 0

    # -- P0: formation env. Goal pipeline ON across all arms (including ARM_D).
    _set_writer_frozen(agent, frozen=False)
    _set_drive_floor(agent, arm.drive_floor_p0)
    env_p0 = _build_env(
        size=12, num_hazards=2, num_resources=3,
        hazard_food_attraction=arm.hfa_p0,
        proximity_harm_scale=0.05,
        spawn_in_reef_half=True,
        seed=seed,
    )
    p0_episodes: List[Dict[str, Any]] = []
    for ep in range(p0_eps):
        ep_metrics = _run_episode(
            agent, env_p0, device, steps_per_episode,
            train=True, e1_opt=e1_opt, wf_opt=wf_opt, wf_buf=wf_buf,
            world_dim=world_dim,
        )
        p0_episodes.append(ep_metrics)
        ep_done += 1
        if (ep_done % 5 == 0) or ep_done == p0_eps:
            print(
                f"  [train] seed={seed} arm={arm.name} ep {ep_done}/{total_training_eps} "
                f"phase=P0 z_goal_peak={ep_metrics['z_goal_norm_peak']:.4f}",
                flush=True,
            )
    p0_summary = _summarise_phase("P0", p0_episodes)

    # -- P1: anneal phase. ARM_D freezes writer; others keep writer ON.
    if arm.freeze_writer_during_p1:
        _set_writer_frozen(agent, frozen=True)
    else:
        _set_writer_frozen(agent, frozen=False)

    p1_episodes: List[Dict[str, Any]] = []
    n_p1 = max(1, p1_eps)
    for ep in range(n_p1):
        t = ep / max(1, n_p1 - 1) if n_p1 > 1 else 1.0
        drive_floor_now = _lerp(arm.drive_floor_p1_start, arm.drive_floor_p1_end, t)
        hfa_now = _lerp(arm.hfa_p1_start, arm.hfa_p1_end, t)
        _set_drive_floor(agent, drive_floor_now)
        env_p1 = _build_env(
            size=12, num_hazards=4, num_resources=5,
            hazard_food_attraction=hfa_now,
            proximity_harm_scale=_lerp(0.05, 0.1, t),
            spawn_in_reef_half=False,
            seed=seed,
        )
        ep_metrics = _run_episode(
            agent, env_p1, device, steps_per_episode,
            train=True, e1_opt=e1_opt, wf_opt=wf_opt, wf_buf=wf_buf,
            world_dim=world_dim,
        )
        p1_episodes.append(ep_metrics)
        ep_done += 1
        if (ep_done % 5 == 0) or ep == n_p1 - 1:
            print(
                f"  [train] seed={seed} arm={arm.name} ep {ep_done}/{total_training_eps} "
                f"phase=P1 drive_floor={drive_floor_now:.3f} hfa={hfa_now:.3f} "
                f"z_goal_peak={ep_metrics['z_goal_norm_peak']:.4f}",
                flush=True,
            )
    p1_summary = _summarise_phase("P1", p1_episodes)

    # ARM_D writer-freeze unfreeze AT P2 START; measurement begins from P2 ep 0.
    if arm.freeze_writer_during_p1:
        _set_writer_frozen(agent, frozen=False)

    # -- P2: measurement (policy frozen at target env config).
    _set_drive_floor(agent, arm.drive_floor_p2)
    env_p2 = _build_env(
        size=12, num_hazards=4, num_resources=5,
        hazard_food_attraction=arm.hfa_p2,
        proximity_harm_scale=0.1,
        spawn_in_reef_half=False,
        seed=seed,
    )
    p2_episodes: List[Dict[str, Any]] = []
    for ep in range(p2_eps):
        ep_metrics = _run_episode(
            agent, env_p2, device, steps_per_episode,
            train=False, world_dim=world_dim,
        )
        p2_episodes.append(ep_metrics)
    p2_summary = _summarise_phase("P2", p2_episodes)

    # P2 start window: first LAST_WINDOW_EPISODES means for C4 measurement.
    p2_start_means = [float(e["z_goal_norm_mean"]) for e in p2_episodes[:LAST_WINDOW_EPISODES]]
    p2_start_median = float(np.median(p2_start_means)) if p2_start_means else 0.0
    p1_end_means = [float(e["z_goal_norm_mean"]) for e in p1_episodes[-LAST_WINDOW_EPISODES:]]
    p1_end_median = float(np.median(p1_end_means)) if p1_end_means else 0.0

    cell = {
        "seed": seed,
        "arm": arm.name,
        "phases": {
            "P0": p0_summary.__dict__,
            "P1": p1_summary.__dict__,
            "P2": p2_summary.__dict__,
        },
        "p1_end_z_goal_norm_median": p1_end_median,
        "p2_start_z_goal_norm_median": p2_start_median,
    }
    print(
        f"verdict: seed={seed} arm={arm.name} "
        f"P0_peak={p0_summary.z_goal_norm_peak_max:.4f} "
        f"P1_med_last={p1_summary.z_goal_norm_median_last_window:.4f} "
        f"P2_med_last={p2_summary.z_goal_norm_median_last_window:.4f}",
        flush=True,
    )
    return cell


# ---------------------------------------------------------------------------
# Acceptance evaluation
# ---------------------------------------------------------------------------


def _frac_seeds_above(values: List[float], floor: float) -> float:
    if not values:
        return 0.0
    return float(sum(1 for v in values if v >= floor)) / float(len(values))


def _frac_seeds_below(values: List[float], ceiling: float) -> float:
    if not values:
        return 0.0
    return float(sum(1 for v in values if v < ceiling)) / float(len(values))


def evaluate_acceptance(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate P0 positive control + C1..C5 per memo Section 3. Returns flat dict."""
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for c in cells:
        by_arm.setdefault(c["arm"], []).append(c)

    def arm_p1_p2_window_medians(arm: str) -> List[float]:
        out: List[float] = []
        for c in by_arm.get(arm, []):
            p1_last = float(c["phases"]["P1"]["z_goal_norm_median_last_window"])
            p2_last = float(c["phases"]["P2"]["z_goal_norm_median_last_window"])
            out.append(float(np.median([p1_last, p2_last])))
        return out

    def arm_p0_peaks(arm: str) -> List[float]:
        return [
            float(c["phases"]["P0"]["z_goal_norm_peak_max"])
            for c in by_arm.get(arm, [])
        ]

    def arm_p1_end_medians(arm: str) -> List[float]:
        return [float(c["p1_end_z_goal_norm_median"]) for c in by_arm.get(arm, [])]

    def arm_p2_start_medians(arm: str) -> List[float]:
        return [float(c["p2_start_z_goal_norm_median"]) for c in by_arm.get(arm, [])]

    def arm_p2_dacc_per_ep(arm: str) -> List[float]:
        return [
            float(c["phases"]["P2"]["dacc_bias_nonzero_steps_per_episode_mean"])
            for c in by_arm.get(arm, [])
        ]

    def arm_p2_z_goal_median(arm: str) -> float:
        vals = [
            float(c["phases"]["P2"]["z_goal_norm_median_last_window"])
            for c in by_arm.get(arm, [])
        ]
        return float(np.median(vals)) if vals else 0.0

    a_window = arm_p1_p2_window_medians("ARM_A_FORMATION_ONLY")
    b_window = arm_p1_p2_window_medians("ARM_B_DRIVE_ANNEAL_ONLY")
    c_window = arm_p1_p2_window_medians("ARM_C_HAZARD_ONLY")
    d_p2_start = arm_p2_start_medians("ARM_D_WRITER_FROZEN_DURING_TRANSITION")
    a_p2_start = arm_p2_start_medians("ARM_A_FORMATION_ONLY")
    a_p1_end = arm_p1_end_medians("ARM_A_FORMATION_ONLY")
    b_p1_end = arm_p1_end_medians("ARM_B_DRIVE_ANNEAL_ONLY")

    # P0 positive control (the adjudicating bit): ARM_A P0 z_goal_norm_peak >=
    # floor on >= 2/3 seeds. If this fails, formation is broken even when fed,
    # and the axis criteria (C2/C3/C4) are NOT trustworthy (they were vacuous in
    # 626 exactly because z_goal was uniformly zero).
    a_p0_peaks = arm_p0_peaks("ARM_A_FORMATION_ONLY")
    p0_frac = _frac_seeds_above(a_p0_peaks, P0_POSITIVE_CONTROL_PEAK_FLOOR)
    p0_pass = p0_frac >= PASS_SEED_FRACTION

    # C1: ARM_A median >= floor on >= 2/3 seeds.
    c1_frac = _frac_seeds_above(a_window, C1_Z_GOAL_MEDIAN_FLOOR)
    c1_pass = c1_frac >= PASS_SEED_FRACTION

    # C2: ARM_B median < ceiling on >= 2/3 seeds AND B P1-end < 0.5 * A P1-end (medians).
    c2_frac = _frac_seeds_below(b_window, C2_ARM_B_Z_GOAL_MEDIAN_CEILING)
    a_p1_end_median = float(np.median(a_p1_end)) if a_p1_end else 0.0
    b_p1_end_median = float(np.median(b_p1_end)) if b_p1_end else 0.0
    c2_ratio_ok = (
        a_p1_end_median <= 1e-9
        or b_p1_end_median < C2_ARM_B_VS_A_P1_END_RATIO * a_p1_end_median
    )
    c2_pass = (c2_frac >= PASS_SEED_FRACTION) and c2_ratio_ok

    # C3: ARM_C median < ceiling on >= 2/3 seeds.
    c3_frac = _frac_seeds_below(c_window, C3_ARM_C_Z_GOAL_MEDIAN_CEILING)
    c3_pass = c3_frac >= PASS_SEED_FRACTION

    # C4: ARM_D p2_start_median >= 0.5 * ARM_A p2_start_median (medians).
    a_p2_start_median = float(np.median(a_p2_start)) if a_p2_start else 0.0
    d_p2_start_median = float(np.median(d_p2_start)) if d_p2_start else 0.0
    c4_pass = (
        a_p2_start_median > 1e-9
        and d_p2_start_median >= C4_ARM_D_VS_A_P2_START_RATIO * a_p2_start_median
    )

    # C5: highest-z_goal arm during P2 (ARM_A vs ARM_D) has dacc_nonzero >= floor on >= 2/3 seeds.
    a_med = arm_p2_z_goal_median("ARM_A_FORMATION_ONLY")
    d_med = arm_p2_z_goal_median("ARM_D_WRITER_FROZEN_DURING_TRANSITION")
    highest_arm = "ARM_A_FORMATION_ONLY" if a_med >= d_med else "ARM_D_WRITER_FROZEN_DURING_TRANSITION"
    dacc_values = arm_p2_dacc_per_ep(highest_arm)
    c5_frac = _frac_seeds_above(dacc_values, C5_DACC_NONZERO_PER_EP_FLOOR)
    c5_pass = c5_frac >= PASS_SEED_FRACTION

    # Axis criteria are only trustworthy if the positive control fired.
    axis_criteria_trusted = bool(p0_pass)

    return {
        "P0_positive_control": {
            "pass": p0_pass, "frac_seeds_clearing": p0_frac,
            "arm_a_p0_z_goal_peak_per_seed": a_p0_peaks,
            "floor": P0_POSITIVE_CONTROL_PEAK_FLOOR,
            "note": (
                "ARM_A P0 z_goal_norm_peak; adjudicates harness-bug (PASS -> 626 "
                "formation failure was the missing update_z_goal call) vs genuine "
                "formation regression (FAIL -> contradicts 622 S0)."
            ),
        },
        "axis_criteria_trusted": axis_criteria_trusted,
        "C1_arm_a_formation_regression_guard": {
            "pass": c1_pass, "frac_seeds_clearing": c1_frac,
            "arm_a_window_medians": a_window,
            "floor": C1_Z_GOAL_MEDIAN_FLOOR,
        },
        "C2_drive_axis_isolation": {
            "pass": c2_pass, "frac_seeds_clearing": c2_frac,
            "arm_b_window_medians": b_window,
            "arm_a_p1_end_median": a_p1_end_median,
            "arm_b_p1_end_median": b_p1_end_median,
            "ratio_ok": c2_ratio_ok,
            "ceiling": C2_ARM_B_Z_GOAL_MEDIAN_CEILING,
            "ratio_threshold": C2_ARM_B_VS_A_P1_END_RATIO,
            "trusted": axis_criteria_trusted,
        },
        "C3_hazard_axis_isolation": {
            "pass": c3_pass, "frac_seeds_clearing": c3_frac,
            "arm_c_window_medians": c_window,
            "ceiling": C3_ARM_C_Z_GOAL_MEDIAN_CEILING,
            "trusted": axis_criteria_trusted,
        },
        "C4_writer_freeze_rescue": {
            "pass": c4_pass,
            "arm_a_p2_start_median": a_p2_start_median,
            "arm_d_p2_start_median": d_p2_start_median,
            "ratio_threshold": C4_ARM_D_VS_A_P2_START_RATIO,
            "trusted": axis_criteria_trusted,
        },
        "C5_consumer_readout": {
            "pass": c5_pass, "frac_seeds_clearing": c5_frac,
            "highest_z_goal_arm": highest_arm,
            "dacc_per_episode_mean_per_seed": dacc_values,
            "floor": C5_DACC_NONZERO_PER_EP_FLOOR,
            "high_z_goal_threshold": C5_HIGH_Z_GOAL_THRESHOLD,
        },
        "overall_pass": bool(
            p0_pass and c1_pass and (c2_pass or c3_pass or c4_pass) and c5_pass
        ),
    }


# ---------------------------------------------------------------------------
# Manifest + main
# ---------------------------------------------------------------------------


def emit_manifest(
    cells: List[Dict[str, Any]],
    acceptance: Dict[str, Any],
    out_dir: Path,
    dry_run: bool,
    cfg_phases: Dict[str, int],
) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    outcome = "PASS" if acceptance["overall_pass"] else "FAIL"
    manifest = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes": SUPERSEDES,
        "outcome": outcome,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "timestamp_utc": ts,
        "dry_run": dry_run,
        "related_queue_ids": ["V3-EXQ-621a", "V3-EXQ-622", "V3-EXQ-626"],
        "design_doc": (
            "REE_assembly/evidence/planning/"
            "goal_stream_repair_diagnostic_ladder_2026-06-01.md"
        ),
        "design_memo": (
            "REE_assembly/evidence/planning/"
            "goal_pipeline_developmental_window_diagnostic_memo_2026-06-01.md"
        ),
        "autopsy": (
            "REE_assembly/evidence/planning/"
            "failure_autopsy_V3-EXQ-626_2026-06-01.md"
        ),
        "harness_fix_note": (
            "626 omitted agent.update_z_goal(); z_goal stayed at zero-init. 626a "
            "feeds the pipeline every step (body_state[11] benefit, 1-energy drive) "
            "in train+eval and adds a P0 positive control on ARM_A formation."
        ),
        "acceptance": acceptance,
        "cells": cells,
        "seeds": SEEDS,
        "phase_budget": cfg_phases,
        "thresholds": {
            "P0_POSITIVE_CONTROL_PEAK_FLOOR": P0_POSITIVE_CONTROL_PEAK_FLOOR,
            "C1_Z_GOAL_MEDIAN_FLOOR": C1_Z_GOAL_MEDIAN_FLOOR,
            "C2_ARM_B_Z_GOAL_MEDIAN_CEILING": C2_ARM_B_Z_GOAL_MEDIAN_CEILING,
            "C2_ARM_B_VS_A_P1_END_RATIO": C2_ARM_B_VS_A_P1_END_RATIO,
            "C3_ARM_C_Z_GOAL_MEDIAN_CEILING": C3_ARM_C_Z_GOAL_MEDIAN_CEILING,
            "C4_ARM_D_VS_A_P2_START_RATIO": C4_ARM_D_VS_A_P2_START_RATIO,
            "C5_DACC_NONZERO_PER_EP_FLOOR": C5_DACC_NONZERO_PER_EP_FLOOR,
            "C5_HIGH_Z_GOAL_THRESHOLD": C5_HIGH_Z_GOAL_THRESHOLD,
            "PASS_SEED_FRACTION": PASS_SEED_FRACTION,
            "LAST_WINDOW_EPISODES": LAST_WINDOW_EPISODES,
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest written: {out_path}")
    return out_path


def main(args: argparse.Namespace) -> Tuple[str, Optional[str]]:
    device = torch.device("cpu")
    if args.dry_run:
        p0_eps, p1_eps, p2_eps, steps_per_ep = 2, 2, 2, 20
    else:
        p0_eps = P0_EPISODES_DEFAULT
        p1_eps = P1_EPISODES_DEFAULT
        p2_eps = P2_EPISODES_DEFAULT
        steps_per_ep = STEPS_PER_EPISODE_DEFAULT
    total_training_eps = p0_eps + p1_eps

    # Probe to read world_obs_dim under the reef + limb-damage env config.
    probe = _build_env(
        size=12, num_hazards=2, num_resources=3,
        hazard_food_attraction=0.0, proximity_harm_scale=0.05,
        spawn_in_reef_half=True, seed=0,
    )
    probe.reset()
    world_obs_dim = probe.world_obs_dim

    arms = arm_configs()
    cells: List[Dict[str, Any]] = []
    for arm in arms:
        for seed in SEEDS:
            cell = run_seed_arm(
                seed=seed, arm=arm, device=device,
                world_obs_dim=world_obs_dim,
                p0_eps=p0_eps, p1_eps=p1_eps, p2_eps=p2_eps,
                steps_per_episode=steps_per_ep,
                total_training_eps=total_training_eps,
            )
            cells.append(cell)

    acceptance = evaluate_acceptance(cells)
    outcome = "PASS" if acceptance["overall_pass"] else "FAIL"
    print(
        f"Overall: {outcome} | "
        f"P0_posctrl={acceptance['P0_positive_control']['pass']} "
        f"C1={acceptance['C1_arm_a_formation_regression_guard']['pass']} "
        f"C2={acceptance['C2_drive_axis_isolation']['pass']} "
        f"C3={acceptance['C3_hazard_axis_isolation']['pass']} "
        f"C4={acceptance['C4_writer_freeze_rescue']['pass']} "
        f"C5={acceptance['C5_consumer_readout']['pass']} "
        f"axis_trusted={acceptance['axis_criteria_trusted']}",
        flush=True,
    )

    if args.dry_run:
        print(f"verdict: dry_run overall={outcome}", flush=True)
        return outcome, None

    out_dir = Path(args.output_dir)
    cfg_phases = {
        "p0_eps": p0_eps, "p1_eps": p1_eps, "p2_eps": p2_eps,
        "steps_per_episode": steps_per_ep,
    }
    out_path = emit_manifest(cells, acceptance, out_dir, dry_run=False, cfg_phases=cfg_phases)
    return outcome, str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(
            REPO_ROOT.parent
            / "REE_assembly"
            / "evidence"
            / "experiments"
            / EXPERIMENT_TYPE
        ),
    )
    args = parser.parse_args()
    outcome, manifest_path = main(args)
    if not args.dry_run and manifest_path:
        emit_outcome(outcome=outcome, manifest_path=manifest_path)
    sys.exit(0 if outcome == "PASS" else 1)
