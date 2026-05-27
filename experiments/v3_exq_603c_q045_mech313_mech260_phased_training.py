#!/opt/local/bin/python3
"""V3-EXQ-603c -- Q-045 MECH-313 vs MECH-260 four-arm ablation with P0/P1 phased training.

Supersedes V3-EXQ-603b (measurement_gap, third in the 603 chain).

Routing source: failure_autopsy_V3-EXQ-603b_2026-05-25, Section 9 "Repair Pathway".
User-confirmed routing 2026-05-25T09:1?Z: ADD P0/P1 TRAINING PHASE; keep SD-054
env literally faithful to Q-045 spec; redesign the policy-training regime, NOT
the env.

Why this iteration exists (autopsy summary)
-------------------------------------------
V3-EXQ-603 / 603a / 603b are three consecutive measurement_gaps. The substrate
is fine across the chain (mech260_operative_all_seeds=true on every cell). The
failure shape is convergent: 2/3 seeds (42, 44) terminate at ~12-16 step/ep on
every arm despite progressively-extended budgets (Fix A STEPS_PER_EPISODE
200->500) and lowered contact-harm (Fix B hazard_harm 0.05->0.02). The
underlying cause is that the gated_policy is at RANDOM INIT when measurement
begins; on seeds where the random-init action proposals walk into the
hazard-food-attraction corridor the agent dies in ~14 steps regardless of
contact-harm tuning. SD-054 bipartite + reef + hazard_food_attraction=0.7 +
proximity_harm_scale=0.1 jointly demand a trained policy.

Fix C (the only fix in 603c): add P0 + P1 training phase per (arm, seed) cell
so the gated_policy is NOT random-init when P2 measurement starts.

Fix D (pre-measurement seed-stability gate): at the end of P1, check median
episode length over the last 10 P1 training episodes >= 75 (the FIFO_WARMUP_STEPS
threshold from 603b). Abort the cell with a clean diagnostic if not -- avoid
burning P2 budget when the policy still cannot survive.

Fix E (preserved from 603b): keep STEPS_PER_EPISODE=500, FIFO_WARMUP_STEPS=75,
4-arm structure, evidence_direction_per_claim vs ARM_0 baseline, and the
obs_harm_a wiring through agent.sense() so dACC.record_action fires on every
tick.

If Fix C also produces 1/3-survivor manifests (most cells abort at P0 or
fail Fix D gate), the 603 chain halts and Q-045 routes to
epistemic_category=substrate_conditional pending a V4 substrate that makes
untrained-inference survival on SD-054 enrichment a clean prerequisite.
This is the falsifiable substrate-conditional branch of the autopsy.

Arms
----
ARM_0_both_off   -- control (expected: no MECH-313 lift, no MECH-260 lift)
ARM_1_mech313    -- use_noise_floor only
ARM_2_mech260    -- use_dacc only (MECH-260 pathway)
ARM_3_both_on    -- MECH-313 + MECH-260

Pre-registered PASS (Q-045 mutually load-bearing, unchanged from 603b)
---------------------------------------------------------------------
C1: ARM_3 entropy > ARM_0 + ENTROPY_MARGIN (0.05)
C2: ARM_3 entropy > max(ARM_1, ARM_2) + ENTROPY_MARGIN (mutually load-bearing)
C3: ARM_1 entropy > ARM_0 AND ARM_2 entropy > ARM_0 (each-alone-beats-off)

PASS if C2. FAIL otherwise (still records per-arm directions).

Phased-training architecture
----------------------------
Each of 12 (arm, seed) cells runs:
    P0: 100 ep x 200 steps/ep on EASY env (reef OFF, no bipartite,
        no hazard_food_attraction, num_hazards=2). E1 prediction loss
        + E2 world-forward loss. Mid-probe abort at 60% if
        running_variance >= effective_threshold (R1 escalation).
    P1: 50 ep x 200 steps/ep on TARGET env (full 603b config:
        reef + bipartite-horizontal + hazard_food_attraction=0.7).
        Continue E1+E2 training. Fix D gate at end of P1.
    P2: 30 ep x 500 steps/ep on TARGET env, frozen policy, action-class
        entropy + dACC diagnostics + FIFO temporal gate (same as 603b).

Training loop threads obs_harm_a / obs_harm_history through agent.sense() so
SD-011 z_harm_a and the dACC suppression FIFO are both exercised during P0/P1
(parallel to 603b Fix 1).

claim_ids: Q-045, MECH-313, MECH-260
experiment_purpose: evidence

Smoke
-----
    /opt/local/bin/python3 experiments/v3_exq_603c_q045_mech313_mech260_phased_training.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_603c_q045_mech313_mech260_phased_training"
QUEUE_ID = "V3-EXQ-603c"
CLAIM_IDS = ["Q-045", "MECH-313", "MECH-260"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 43, 44]

# P2 measurement (unchanged from 603b)
EVAL_EPISODES = 30
STEPS_PER_EPISODE = 500
FIFO_WARMUP_STEPS = 75

# P0/P1 training (autopsy Section 9 bounds: P0 50-100, P1 30-50)
P0_BUDGET = 100
P1_BUDGET = 50
TRAIN_STEPS_PER_EPISODE = 200
TRAIN_BATCH_SIZE = 32
TRAIN_WF_BUF_MAX = 2000
TRAIN_LR_E1 = 1e-4
TRAIN_LR_E2_WF = 1e-3
P0_PROBE_INTERVAL = 20
P0_MID_PROBE_FRAC = 0.60
P1_STABILITY_WINDOW = 10
P1_SURVIVAL_GATE_STEPS = FIFO_WARMUP_STEPS  # Fix D: median ep length >= 75

DRY_RUN_SEEDS = [42]
DRY_RUN_P0_BUDGET = 4
DRY_RUN_P1_BUDGET = 3
DRY_RUN_P2_EPISODES = 2
DRY_RUN_STEPS = 30

ENTROPY_MARGIN = 0.05
DACC_SUPPRESSION_WEIGHT = 0.5
DACC_SUPPRESSION_MEMORY = 8
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_HISTORY_LEN = 10

# Target (P1 + P2) env: full 603b SD-054 enrichment config -- unchanged.
TARGET_ENV_KWARGS = dict(
    size=12,
    num_hazards=4,
    num_resources=5,
    hazard_harm=0.02,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=HARM_HISTORY_LEN,
    reef_enabled=True,
    n_reef_patches=3,
    reef_patch_radius=2,
    hazard_food_attraction=0.7,
    reef_bipartite_layout=True,
    reef_bipartite_axis="horizontal",
    reef_bipartite_agent_band_radius=1,
)

# Easy (P0) env: same world_obs_dim as TARGET env (reef_enabled=True so the
# +25 reef_field_view channel is present and the encoder shape matches), but
# bipartite_layout OFF and hazard_food_attraction=0.0 so the agent does not
# face the SD-054 path-dependent corridor that killed seeds 42/44 in 603a/b.
# Smaller env + fewer hazards + lower proximity_harm_scale so navigation is
# learnable in ~100 episodes.
EASY_ENV_KWARGS = dict(
    size=12,
    num_hazards=2,
    num_resources=3,
    hazard_harm=0.05,
    env_drift_interval=10,
    env_drift_prob=0.05,
    proximity_harm_scale=0.05,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=HARM_HISTORY_LEN,
    reef_enabled=True,
    n_reef_patches=3,
    reef_patch_radius=2,
    hazard_food_attraction=0.0,
)

ARMS: List[Dict[str, Any]] = [
    {
        "arm": "ARM_0_both_off",
        "use_noise_floor": False,
        "noise_floor_alpha": 0.1,
        "use_dacc": False,
    },
    {
        "arm": "ARM_1_mech313_only",
        "use_noise_floor": True,
        "noise_floor_alpha": 0.5,
        "use_dacc": False,
    },
    {
        "arm": "ARM_2_mech260_only",
        "use_noise_floor": False,
        "noise_floor_alpha": 0.1,
        "use_dacc": True,
    },
    {
        "arm": "ARM_3_both_on",
        "use_noise_floor": True,
        "noise_floor_alpha": 0.5,
        "use_dacc": True,
    },
]


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------

def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0.0:
            h -= p * math.log(p)
    return float(h)


def _make_env(seed: int, kwargs: Dict[str, Any]) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **kwargs)


def _make_config(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEConfig:
    return REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_obs_a_dim=50,
        harm_history_len=HARM_HISTORY_LEN,
        use_gated_policy=True,
        gated_policy_use_first_action_onehot=True,
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        use_noise_floor=arm["use_noise_floor"],
        noise_floor_alpha=arm["noise_floor_alpha"],
        use_dacc=arm["use_dacc"],
        dacc_weight=(1.0 if arm["use_dacc"] else 0.0),
        dacc_suppression_weight=(
            DACC_SUPPRESSION_WEIGHT if arm["use_dacc"] else 0.0
        ),
        dacc_suppression_memory=DACC_SUPPRESSION_MEMORY,
        use_structured_curiosity=False,
    )


def _obs_harm_a(obs_dict: Dict[str, Any]) -> Optional[torch.Tensor]:
    ha = obs_dict.get("harm_obs_a")
    if ha is None:
        return None
    t = ha.float()
    return t.unsqueeze(0) if t.dim() == 1 else t


def _obs_harm_history(obs_dict: Dict[str, Any]) -> Optional[torch.Tensor]:
    hh = obs_dict.get("harm_history")
    if hh is None:
        return None
    t = hh.float()
    return t.unsqueeze(0) if t.dim() == 1 else t


def _select_action_with_harm(
    agent: REEAgent,
    obs_body: torch.Tensor,
    obs_world: torch.Tensor,
    obs_harm_a: Optional[torch.Tensor],
    obs_harm_history: Optional[torch.Tensor],
):
    """Sense+generate+select_action path that threads obs_harm_a so z_harm_a
    is populated and dACC.record_action fires. Returns (action, latent_state).
    """
    latent_state = agent.sense(
        obs_body,
        obs_world,
        obs_harm_a=obs_harm_a,
        obs_harm_history=obs_harm_history,
    )
    ticks = agent.clock.advance()
    if ticks.get("e1_tick", False):
        e1_prior = agent._e1_tick(latent_state)
    else:
        e1_prior = torch.zeros(1, WORLD_DIM, device=agent.device)
    candidates = agent.generate_trajectories(latent_state, e1_prior, ticks)
    action = agent.select_action(candidates, ticks, temperature=1.0)
    if ticks.get("e3_quiescent", False):
        agent._do_replay(latent_state)
    agent._step_count += 1
    return action, latent_state


def _dacc_diagnostics(agent: REEAgent) -> Dict[str, Any]:
    if agent.dacc is None:
        return {
            "dacc_forward_calls": 0,
            "dacc_history_len": 0,
            "dacc_max_suppression": 0.0,
        }
    hist_len = len(agent.dacc._action_history)
    max_sup = 0.0
    bundle = getattr(agent, "_dacc_last_bundle", None)
    if bundle is not None and "suppression" in bundle:
        sup = bundle["suppression"]
        if isinstance(sup, torch.Tensor) and sup.numel() > 0:
            max_sup = float(sup.max().item())
    return {
        "dacc_forward_calls": int(agent.dacc._n_forward_calls),
        "dacc_history_len": hist_len,
        "dacc_max_suppression": round(max_sup, 6),
    }


# ---------------------------------------------------------------------------
# P0 / P1 training loop
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    """Result of one P0 or P1 training phase."""
    phase: str               # "P0" or "P1"
    n_episodes: int
    converged: bool
    aborted: bool
    abort_reason: str
    final_rv: float
    final_median_ep_steps: float  # median over last stability_window episodes
    probe_log: List[Dict] = field(default_factory=list)
    ep_steps_log: List[int] = field(default_factory=list)


def _train_one_episode(
    agent: REEAgent,
    env: CausalGridWorldV2,
    e1_opt: optim.Optimizer,
    wf_opt: optim.Optimizer,
    wf_buf: Deque,
    batch_size: int,
    steps_per_episode: int,
) -> Tuple[float, int]:
    """Run one training episode threading obs_harm_a / obs_harm_history.

    Mirrors committed_mode_curriculum._one_episode_train but uses
    _select_action_with_harm so SD-011 z_harm_a and dACC.record_action fire.

    Returns (final_running_variance, ep_steps_executed).
    """
    _, obs_dict = env.reset()
    agent.reset()

    z_world_prev: Optional[torch.Tensor] = None
    action_prev: Optional[torch.Tensor] = None
    ep_steps = 0

    for _ in range(steps_per_episode):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        if obs_body.dim() == 1:
            obs_body = obs_body.unsqueeze(0)
        if obs_world.dim() == 1:
            obs_world = obs_world.unsqueeze(0)

        action, latent_state = _select_action_with_harm(
            agent,
            obs_body,
            obs_world,
            _obs_harm_a(obs_dict),
            _obs_harm_history(obs_dict),
        )
        z_world_curr = latent_state.z_world.detach()

        if z_world_prev is not None and action_prev is not None:
            wf_buf.append(
                (z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu())
            )

        # E1 prediction loss
        e1_opt.zero_grad()
        e1_loss = agent.compute_prediction_loss()
        if e1_loss.requires_grad:
            e1_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(agent.e1.parameters()), 1.0)
            e1_opt.step()

        # E2 world-forward loss -- drives running_variance toward convergence
        if len(wf_buf) >= batch_size:
            k = min(batch_size, len(wf_buf))
            idxs = torch.randperm(len(wf_buf))[:k].tolist()
            zw_b = torch.cat([wf_buf[i][0] for i in idxs])
            a_b = torch.cat([wf_buf[i][1] for i in idxs])
            zw1_b = torch.cat([wf_buf[i][2] for i in idxs])
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

        if action is None:
            idx = random.randint(0, env.action_dim - 1)
            action_onehot = torch.zeros(1, env.action_dim, device=agent.device)
            action_onehot[0, idx] = 1.0
        else:
            action_onehot = action

        _flat, _harm, done, _info, obs_dict = env.step(action_onehot)
        ep_steps += 1
        if done:
            break

    return float(agent.e3._running_variance), ep_steps


def _run_train_phase(
    agent: REEAgent,
    env: CausalGridWorldV2,
    phase: str,
    budget: int,
    steps_per_episode: int,
    arm_label: str,
    seed: int,
    disable_mid_probe_abort: bool = False,
) -> TrainResult:
    """Run P0 or P1 training to budget. Probes running_variance + episode length.

    P0 abort: at mid_probe_frac if running_variance has not crossed the
    commit_threshold (R1 substrate mis-calibration).
    P1 abort: at end of phase if median ep length over last stability_window
    episodes < FIFO_WARMUP_STEPS (Fix D pre-measurement gate).
    """
    agent.train()
    e1_opt = optim.Adam(list(agent.e1.parameters()), lr=TRAIN_LR_E1)
    wf_opt = optim.Adam(
        list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters()),
        lr=TRAIN_LR_E2_WF,
    )
    wf_buf: Deque = deque(maxlen=TRAIN_WF_BUF_MAX)

    ep_steps_window: Deque[int] = deque(maxlen=P1_STABILITY_WINDOW)
    probe_log: List[Dict] = []
    ep_steps_log: List[int] = []
    aborted = False
    abort_reason = ""
    converged = False

    commit_threshold = float(agent.e3.commit_threshold)
    mid_probe_episode = math.ceil(P0_MID_PROBE_FRAC * budget)

    for ep in range(budget):
        rv, ep_steps = _train_one_episode(
            agent, env, e1_opt, wf_opt, wf_buf,
            TRAIN_BATCH_SIZE, steps_per_episode,
        )
        ep_steps_window.append(ep_steps)
        ep_steps_log.append(int(ep_steps))

        # Loop-print every probe interval so the runner can advance its bar.
        if (ep + 1) % P0_PROBE_INTERVAL == 0 or ep == budget - 1:
            median_ep_steps = float(np.median(list(ep_steps_window)))
            probe = {
                "episode": ep + 1,
                "running_variance": rv,
                "commit_threshold": commit_threshold,
                "rv_below_threshold": rv < commit_threshold,
                "median_ep_steps_window": median_ep_steps,
            }
            probe_log.append(probe)
            print(
                f"  [{phase} probe] arm={arm_label} seed={seed} ep {ep + 1}/{budget}"
                f"  rv={rv:.5f} thr={commit_threshold:.4f}"
                f"  median_ep_steps={median_ep_steps:.1f}",
                flush=True,
            )

            # P0 mid-probe abort (R1 escalation per autopsy section 9)
            if (
                phase == "P0"
                and not disable_mid_probe_abort
                and (ep + 1) >= mid_probe_episode
                and rv >= commit_threshold
            ):
                print(
                    f"  [{phase} ABORT] arm={arm_label} seed={seed} rv={rv:.5f}"
                    f" >= commit_threshold={commit_threshold:.4f}"
                    f" at ep={ep + 1} ({int(P0_MID_PROBE_FRAC * 100)}% of"
                    f" budget). R1 escalation -- substrate mis-calibration.",
                    flush=True,
                )
                aborted = True
                abort_reason = "p0_rv_not_converging"
                break

        # Standard per-episode loop print expected by the runner.
        if (ep + 1) % 10 == 0 or (ep + 1) == budget:
            print(
                f"  [train] {phase} arm={arm_label} seed={seed} ep {ep + 1}/{budget}",
                flush=True,
            )

    final_rv = float(agent.e3._running_variance)
    final_median = (
        float(np.median(list(ep_steps_window))) if ep_steps_window else 0.0
    )
    converged = (final_rv < commit_threshold) and not aborted

    # P1 Fix D gate: end-of-P1 median episode length must clear FIFO warmup.
    if (
        phase == "P1"
        and not aborted
        and not disable_mid_probe_abort  # dry-run reuses this flag
        and len(ep_steps_window) >= 1
        and final_median < P1_SURVIVAL_GATE_STEPS
    ):
        print(
            f"  [{phase} ABORT-FIXD] arm={arm_label} seed={seed}"
            f" median_ep_steps={final_median:.1f} <"
            f" survival_gate={P1_SURVIVAL_GATE_STEPS} at end of P1."
            f" Skip P2 measurement.",
            flush=True,
        )
        aborted = True
        abort_reason = "p1_survival_gate_failed"

    return TrainResult(
        phase=phase,
        n_episodes=len(ep_steps_log),
        converged=converged,
        aborted=aborted,
        abort_reason=abort_reason,
        final_rv=final_rv,
        final_median_ep_steps=final_median,
        probe_log=probe_log,
        ep_steps_log=ep_steps_log,
    )


# ---------------------------------------------------------------------------
# P2 measurement (frozen-policy entropy + dACC diagnostics)
# ---------------------------------------------------------------------------

def _run_p2_measurement(
    agent: REEAgent,
    env: CausalGridWorldV2,
    arm: Dict[str, Any],
    seed: int,
    episodes: int,
    steps_per_episode: int,
    fifo_warmup_steps: int,
) -> Dict[str, Any]:
    """Frozen-policy P2 measurement. Mirrors 603b._run_arm_seed."""
    agent.eval()
    action_counts: Counter = Counter()
    reef_steps = 0
    total_steps = 0
    measured_steps = 0
    reef_cells = set()
    max_dacc_history = 0
    max_dacc_forward = 0
    max_dacc_suppression = 0.0

    with torch.no_grad():
        for ep in range(episodes):
            _, obs_dict = env.reset()
            agent.reset()
            if ep == 0:
                reef_cells = set(getattr(env, "_reef_cells", set()))

            for step_in_ep in range(steps_per_episode):
                obs_body = obs_dict["body_state"]
                obs_world = obs_dict["world_state"]
                if obs_body.dim() == 1:
                    obs_body = obs_body.unsqueeze(0)
                if obs_world.dim() == 1:
                    obs_world = obs_world.unsqueeze(0)

                action, _ = _select_action_with_harm(
                    agent,
                    obs_body,
                    obs_world,
                    _obs_harm_a(obs_dict),
                    _obs_harm_history(obs_dict),
                )

                if arm["use_dacc"] and agent.dacc is not None:
                    diag = _dacc_diagnostics(agent)
                    max_dacc_history = max(max_dacc_history, diag["dacc_history_len"])
                    max_dacc_forward = max(max_dacc_forward, diag["dacc_forward_calls"])
                    max_dacc_suppression = max(
                        max_dacc_suppression, diag["dacc_max_suppression"]
                    )

                if action is None:
                    idx = random.randint(0, env.action_dim - 1)
                    action_onehot = torch.zeros(1, env.action_dim, device=agent.device)
                    action_onehot[0, idx] = 1.0
                else:
                    action_onehot = action
                    idx = int(action.argmax(dim=-1).item())

                if step_in_ep >= fifo_warmup_steps:
                    action_counts[idx] += 1
                    measured_steps += 1

                pos = (int(env.agent_x), int(env.agent_y))
                if pos in reef_cells:
                    reef_steps += 1

                _flat, _harm, done, _info, obs_dict = env.step(action_onehot)
                total_steps += 1
                if done:
                    break

            if (ep + 1) % 10 == 0 or (ep + 1) == episodes:
                print(
                    f"  [P2 train] arm={arm['arm']} seed={seed} ep {ep + 1}/{episodes}",
                    flush=True,
                )

    entropy = _entropy(action_counts)
    reef_fraction = reef_steps / max(total_steps, 1)
    steps_per_ep_measured = measured_steps / max(episodes, 1)
    fifo_gate_ok = (
        fifo_warmup_steps >= 2 * DACC_SUPPRESSION_MEMORY
        and steps_per_ep_measured >= (steps_per_episode - fifo_warmup_steps) * 0.9
    )

    out: Dict[str, Any] = {
        "arm": arm["arm"],
        "seed": seed,
        "selected_action_entropy": round(entropy, 6),
        "reef_fraction": round(reef_fraction, 6),
        "total_steps": int(total_steps),
        "measured_steps": int(measured_steps),
        "fifo_warmup_steps": int(fifo_warmup_steps),
        "steps_per_episode_measured_mean": round(steps_per_ep_measured, 2),
        "fifo_temporal_gate_ok": bool(fifo_gate_ok),
        "unique_actions": len(action_counts),
        "p2_run": True,
    }
    if arm["use_dacc"]:
        out.update(
            {
                "dacc_history_len_max": max_dacc_history,
                "dacc_forward_calls_max": max_dacc_forward,
                "dacc_max_suppression": round(max_dacc_suppression, 6),
                "mech260_operative": bool(
                    max_dacc_forward > 0 and max_dacc_history > 0
                ),
            }
        )
    return out


def _empty_cell_row(
    arm: Dict[str, Any],
    seed: int,
    abort_phase: str,
    abort_reason: str,
) -> Dict[str, Any]:
    """Row stub for cells that aborted at P0 or failed Fix D gate."""
    row: Dict[str, Any] = {
        "arm": arm["arm"],
        "seed": seed,
        "selected_action_entropy": 0.0,
        "reef_fraction": 0.0,
        "total_steps": 0,
        "measured_steps": 0,
        "fifo_warmup_steps": int(FIFO_WARMUP_STEPS),
        "steps_per_episode_measured_mean": 0.0,
        "fifo_temporal_gate_ok": False,
        "unique_actions": 0,
        "p2_run": False,
        "abort_phase": abort_phase,
        "abort_reason": abort_reason,
    }
    if arm["use_dacc"]:
        row.update(
            {
                "dacc_history_len_max": 0,
                "dacc_forward_calls_max": 0,
                "dacc_max_suppression": 0.0,
                "mech260_operative": False,
            }
        )
    return row


# ---------------------------------------------------------------------------
# Per-cell pipeline (P0 -> P1 -> P2)
# ---------------------------------------------------------------------------

def _run_arm_seed(
    arm: Dict[str, Any],
    seed: int,
    p0_budget: int,
    p1_budget: int,
    p2_episodes: int,
    p2_steps_per_episode: int,
    train_steps_per_episode: int,
    disable_mid_probe_abort: bool = False,
) -> Dict[str, Any]:
    """P0 -> P1 -> P2 pipeline for one (arm, seed) cell."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    target_env = _make_env(seed, TARGET_ENV_KWARGS)
    cfg = _make_config(target_env, arm)
    agent = REEAgent(cfg)

    cell_diagnostics: Dict[str, Any] = {}

    # P0: warmup on easy env
    easy_env = _make_env(seed, EASY_ENV_KWARGS)
    p0 = _run_train_phase(
        agent, easy_env, "P0",
        budget=p0_budget,
        steps_per_episode=train_steps_per_episode,
        arm_label=arm["arm"], seed=seed,
        disable_mid_probe_abort=disable_mid_probe_abort,
    )
    cell_diagnostics["p0"] = {
        "n_episodes": p0.n_episodes,
        "converged": p0.converged,
        "aborted": p0.aborted,
        "abort_reason": p0.abort_reason,
        "final_rv": round(p0.final_rv, 6),
        "final_median_ep_steps": round(p0.final_median_ep_steps, 2),
        "probe_log_len": len(p0.probe_log),
    }
    if p0.aborted:
        print(
            f"verdict: FAIL arm={arm['arm']} seed={seed} aborted_at=P0"
            f" reason={p0.abort_reason}",
            flush=True,
        )
        row = _empty_cell_row(arm, seed, "P0", p0.abort_reason)
        row["cell_diagnostics"] = cell_diagnostics
        return row

    # P1: consolidation on target env
    p1 = _run_train_phase(
        agent, target_env, "P1",
        budget=p1_budget,
        steps_per_episode=train_steps_per_episode,
        arm_label=arm["arm"], seed=seed,
        disable_mid_probe_abort=disable_mid_probe_abort,
    )
    cell_diagnostics["p1"] = {
        "n_episodes": p1.n_episodes,
        "converged": p1.converged,
        "aborted": p1.aborted,
        "abort_reason": p1.abort_reason,
        "final_rv": round(p1.final_rv, 6),
        "final_median_ep_steps": round(p1.final_median_ep_steps, 2),
        "p1_survival_gate_steps": int(P1_SURVIVAL_GATE_STEPS),
        "probe_log_len": len(p1.probe_log),
    }
    if p1.aborted:
        print(
            f"verdict: FAIL arm={arm['arm']} seed={seed} aborted_at=P1"
            f" reason={p1.abort_reason}",
            flush=True,
        )
        row = _empty_cell_row(arm, seed, "P1", p1.abort_reason)
        row["cell_diagnostics"] = cell_diagnostics
        return row

    # P2: frozen measurement on target env (rebuild fresh target env so the
    # episode RNG sequence is the same as 603a/603b for the measurement loop)
    p2_env = _make_env(seed, TARGET_ENV_KWARGS)
    row = _run_p2_measurement(
        agent, p2_env, arm, seed,
        episodes=p2_episodes,
        steps_per_episode=p2_steps_per_episode,
        fifo_warmup_steps=min(FIFO_WARMUP_STEPS, max(0, p2_steps_per_episode - 1)),
    )
    row["cell_diagnostics"] = cell_diagnostics
    print(f"verdict: PASS arm={arm['arm']} seed={seed}", flush=True)
    return row


# ---------------------------------------------------------------------------
# Per-claim direction
# ---------------------------------------------------------------------------

def _evaluate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)

    def mean_entropy(arm_name: str) -> float:
        cells = by_arm.get(arm_name, [])
        if not cells:
            return 0.0
        vals = [r["selected_action_entropy"] for r in cells if r.get("p2_run", False)]
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))

    e0 = mean_entropy("ARM_0_both_off")
    e1 = mean_entropy("ARM_1_mech313_only")
    e2 = mean_entropy("ARM_2_mech260_only")
    e3 = mean_entropy("ARM_3_both_on")

    c1 = e3 > e0 + ENTROPY_MARGIN
    c2 = e3 > max(e1, e2) + ENTROPY_MARGIN
    c3 = (e1 > e0) and (e2 > e0)

    mech260_rows = [
        r for r in rows
        if r["arm"] in ("ARM_2_mech260_only", "ARM_3_both_on") and r.get("p2_run", False)
    ]
    mech260_operative_all = bool(mech260_rows) and all(
        r.get("mech260_operative", False) for r in mech260_rows
    )

    p2_rows = [r for r in rows if r.get("p2_run", False)]
    fifo_ok_all = bool(p2_rows) and all(
        r.get("fifo_temporal_gate_ok", False) for r in p2_rows
    )

    p2_cell_count = len(p2_rows)
    aborted_cells = len(rows) - p2_cell_count

    return {
        "entropy_ARM_0": round(e0, 6),
        "entropy_ARM_1": round(e1, 6),
        "entropy_ARM_2": round(e2, 6),
        "entropy_ARM_3": round(e3, 6),
        "c1_both_beats_off": c1,
        "c2_mutually_load_bearing": c2,
        "c3_each_alone_beats_off": c3,
        "mech260_operative_all_seeds": mech260_operative_all,
        "fifo_temporal_gate_ok_all": fifo_ok_all,
        "p2_cell_count": p2_cell_count,
        "aborted_cells": aborted_cells,
        "overall_pass": bool(c2 and p2_cell_count == len(rows)),
    }


def _evidence_direction_per_claim(summary: Dict[str, Any]) -> Dict[str, str]:
    """Per-claim direction vs ARM_0 baseline. Mirrors 603b semantics.

    If P2 cell count is below half the matrix, route all three claims to
    non_contributory -- the design ran but the substrate-conditional branch
    of the autopsy fired and the data is structurally underpowered.
    """
    total_cells = 12  # 4 arms * 3 seeds
    if summary["p2_cell_count"] < total_cells // 2:
        return {
            "Q-045": "non_contributory",
            "MECH-313": "non_contributory",
            "MECH-260": "non_contributory",
        }

    e0 = summary["entropy_ARM_0"]
    e1 = summary["entropy_ARM_1"]
    e2 = summary["entropy_ARM_2"]

    if summary["c2_mutually_load_bearing"]:
        q045 = "supports"
        m313 = "supports" if e1 > e0 + ENTROPY_MARGIN else "mixed"
        m260 = "supports" if e2 > e0 + ENTROPY_MARGIN else "mixed"
    elif summary["c3_each_alone_beats_off"]:
        q045 = "mixed"
        m313 = "supports" if e1 > e0 + ENTROPY_MARGIN else "mixed"
        m260 = "supports" if e2 > e0 + ENTROPY_MARGIN else "mixed"
    elif summary["entropy_ARM_3"] > e0:
        q045 = "mixed"
        m313 = "mixed" if e1 > e0 else "weakens"
        m260 = "mixed" if e2 > e0 else "weakens"
    else:
        q045 = "weakens"
        m313 = "weakens"
        m260 = "weakens"

    if not summary.get("mech260_operative_all_seeds", True):
        m260 = "unknown"

    return {"Q-045": q045, "MECH-313": m313, "MECH-260": m260}


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    p0_budget = DRY_RUN_P0_BUDGET if dry_run else P0_BUDGET
    p1_budget = DRY_RUN_P1_BUDGET if dry_run else P1_BUDGET
    p2_episodes = DRY_RUN_P2_EPISODES if dry_run else EVAL_EPISODES
    train_steps = DRY_RUN_STEPS if dry_run else TRAIN_STEPS_PER_EPISODE
    p2_steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE

    rows: List[Dict[str, Any]] = []
    arms_to_run = ARMS[:1] if dry_run else ARMS
    for arm in arms_to_run:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm['arm']}", flush=True)
            cell = _run_arm_seed(
                arm, seed,
                p0_budget=p0_budget,
                p1_budget=p1_budget,
                p2_episodes=p2_episodes,
                p2_steps_per_episode=p2_steps,
                train_steps_per_episode=train_steps,
                disable_mid_probe_abort=dry_run,
            )
            rows.append(cell)

    summary = _evaluate(rows)
    outcome = "PASS" if summary["overall_pass"] else "FAIL"
    edpc = _evidence_direction_per_claim(summary)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": outcome,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": edpc["Q-045"],
        "evidence_direction_per_claim": edpc,
        "supersedes": "V3-EXQ-603b",
        "dry_run": dry_run,
        "acceptance_criteria": summary,
        "summary": summary,
        "arm_results": rows,
        "fifo_warmup_steps": int(min(FIFO_WARMUP_STEPS, max(0, p2_steps - 1))),
        "steps_per_episode_p2": int(p2_steps),
        "p0_budget": int(p0_budget),
        "p1_budget": int(p1_budget),
        "train_steps_per_episode": int(train_steps),
        "p1_survival_gate_steps": int(P1_SURVIVAL_GATE_STEPS),
        "fixes_applied": [
            "Fix C (new in 603c, autopsy section 9): P0 warmup on easy env"
            f" ({p0_budget} ep) + P1 consolidation on target env ({p1_budget} ep)"
            " before P2 measurement. gated_policy is no longer at random init.",
            f"Fix D (new in 603c, autopsy section 9): pre-P2 survival gate --"
            f" abort cell at end of P1 if median ep length over last"
            f" {P1_STABILITY_WINDOW} episodes < {P1_SURVIVAL_GATE_STEPS}.",
            "Fix 1 (from 603a, retained): sense() threads obs_harm_a +"
            " obs_harm_history so z_harm_a is populated and dACC.record_action"
            " fires every tick.",
            f"Fix 2 (from 603a, retained): FIFO_WARMUP_STEPS={FIFO_WARMUP_STEPS}"
            " before P2 entropy measurement.",
            "Fix 3 (from 603a, retained): evidence_direction_per_claim vs ARM_0"
            " baseline.",
            f"Fix A (from 603b, retained): P2 STEPS_PER_EPISODE={STEPS_PER_EPISODE}.",
            "Fix B (from 603b, retained): hazard_harm=0.02 on target env."
            " Latent harm signals (z_harm_s / z_harm_a) decoupled and unaffected.",
        ],
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = out_dir / f"{run_id}.json"
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"Manifest written: {out_path}", flush=True)
    else:
        out_path = Path("/dev/null")
        print("Dry run -- manifest not written.", flush=True)

    print(f"Outcome: {outcome}", flush=True)
    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run_experiment(dry_run=args.dry_run)
    if args.dry_run:
        sys.exit(0)
    emit_outcome(
        outcome=str(result.get("outcome", "FAIL")),
        manifest_path=str(result.get("manifest_path", "/dev/null")),
    )
