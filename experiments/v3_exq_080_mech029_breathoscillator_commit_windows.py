"""
V3-EXQ-080 -- MECH-029 BreathOscillator Uncommitted Windows (EXP-0089)

Claims: MECH-029

MECH-029 asserts that the default/reflective mode creates genuine uncommitted
processing windows -- states where the agent has lower commitment and higher
openness to revising its world model. The BreathOscillator (MECH-108) is used
to cyclically FORCE periodic uncommitted windows within a single trained agent,
allowing within-agent comparison of sweep-phase vs inter-sweep statistics.

Design (EXP-0089):
  One trained agent, two evaluation runs:
    Condition A (oscillator active):
      BreathOscillator (breath_period=50, sweep_duration=5, sweep_amplitude=0.25)
      fires in clock.py. During sweep-phase steps, the effective commit_threshold
      is reduced: effective_threshold = base_threshold * (1 - sweep_amplitude).
      This LOWERS the variance bar, pushing the agent into uncommitted mode when
      running_variance > effective_threshold (typically true for trained agents,
      since running_variance ~0.33 > effective_threshold 0.30 = 0.40 * 0.75).
    Condition B (oscillator inactive, control):
      Same trained agent, same phase schedule labelled identically, no threshold
      change. Tests that the sweep/inter-sweep z_world difference is absent
      without the oscillator mechanism.

  Note on e3_steps_per_tick:
    Default e3 rate (10 steps/tick) with breath_period=50, sweep_duration=5
    would never fire an e3_tick during the sweep window (ticks at 40, 50;
    sweep at steps 45-49). We use e3_steps_per_tick=5, which fires an e3_tick
    at step 45 (the start of each sweep window). This ensures the commitment
    decision changes during each sweep phase.

Pre-registered thresholds:
  C1: z_world_var_sweep > z_world_var_inter * 1.05
      (sweep phase shows higher z_world variability -- uncommitted exploration)
  C2: committed_rate_sweep < committed_rate_inter * 0.90
      (mechanism confirmation: agent commits less during sweep phases in A)
  C3: n_sweep_steps_A >= 20
      (sufficient sweep observations)
  C4: harm_pred_std_inter > 0.01
      (E3 harm prediction is not degenerate outside sweep windows)
  C5: no fatal errors

Background: EXQ-065 was design-inconclusive (always committed). The
BreathOscillator provides a controlled forcing mechanism that directly
approximates MECH-108 respiratory sweep clock within the current substrate.
Parameters (breath_period, sweep_amplitude) should be updated if LIT-0094
returns physiologically-grounded values.
"""

import sys
import random
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.optim as optim
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.heartbeat.clock import MultiRateClock
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_080_mech029_breathoscillator_commit_windows"
CLAIM_IDS = ["MECH-029"]

BODY_OBS_DIM = 10    # use_proxy_fields=False: position(2)+health+energy+footprint+harm_proxy(5)
WORLD_OBS_DIM = 200  # use_proxy_fields=False: local_view(5x5x7=175) + contamination(25)
ACTION_DIM = 4

# BreathOscillator parameters (pre-registered)
BREATH_PERIOD = 50       # total cycle length: 45 inter-sweep + 5 sweep steps
SWEEP_DURATION = 5       # sweep phase lasts 5 steps per cycle
SWEEP_AMPLITUDE = 0.25   # fractional threshold reduction: effective = base * 0.75
E3_RATE = 5              # e3_steps_per_tick; 5 ensures e3_tick fires at step 45 (sweep start)

# Pre-registered PASS thresholds
C1_RATIO = 1.05   # z_world_var[sweep] / z_world_var[inter] must exceed this
C2_RATIO = 0.90   # committed_rate[sweep] must be < this * committed_rate[inter]
C3_MIN_SWEEP_STEPS = 20
C4_MIN_HARM_STD = 0.01


def _make_env(seed: int) -> CausalGridWorld:
    return CausalGridWorld(
        size=10,
        num_resources=5,
        num_hazards=3,
        use_proxy_fields=False,
        seed=seed,
    )


def _make_clock(oscillator_active: bool) -> MultiRateClock:
    """Create a MultiRateClock with or without BreathOscillator."""
    return MultiRateClock(
        e1_steps_per_tick=1,
        e2_steps_per_tick=3,
        e3_steps_per_tick=E3_RATE,
        theta_buffer_size=10,
        breath_period=BREATH_PERIOD if oscillator_active else 0,
        sweep_duration=SWEEP_DURATION,
        sweep_amplitude=SWEEP_AMPLITUDE,
    )


def _train(
    agent: REEAgent,
    env: CausalGridWorld,
    episodes: int,
    steps_per_episode: int,
    world_dim: int,
    lr: float,
    seed: int,
    label: str,
) -> None:
    """Basic training loop (harm supervision on E3, E2 motor-sensory, E1 prediction)."""
    e1_opt = optim.Adam(agent.e1.parameters(), lr=lr)
    e3_opt = optim.Adam(
        list(agent.e3.parameters()) + list(agent.latent_stack.parameters()), lr=lr
    )
    e2_opt = optim.Adam(agent.e2.parameters(), lr=lr * 3)

    agent.train()
    for ep in range(episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_harm = 0.0

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"].detach().clone().float()
            obs_world = obs_dict["world_state"].detach().clone().float()

            latent = agent.sense(obs_body, obs_world)
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent)
                if ticks["e1_tick"]
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks, temperature=1.0)

            _, reward, done, _, obs_dict = env.step(action)
            harm_signal = float(reward) if reward < 0 else 0.0
            ep_harm += abs(harm_signal)

            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                e1_opt.zero_grad()
                e1_loss.backward()
                e1_opt.step()

            e2_loss = agent.compute_e2_loss()
            if e2_loss.requires_grad:
                e2_opt.zero_grad()
                e2_loss.backward()
                e2_opt.step()

            if agent._current_latent is not None:
                z_world = agent._current_latent.z_world.detach()
                harm_target = torch.tensor(
                    [[1.0 if harm_signal < 0 else 0.0]], device=agent.device
                )
                harm_loss = F.mse_loss(agent.e3.harm_eval(z_world), harm_target)
                e3_opt.zero_grad()
                harm_loss.backward()
                e3_opt.step()

            agent.update_residue(harm_signal)
            if done:
                break

        if (ep + 1) % 25 == 0:
            rv = agent.e3._running_variance
            print(
                f"  [train/{label}] ep {ep+1}/{episodes}"
                f" harm={ep_harm:.3f} rv={rv:.4f}",
                flush=True,
            )


def _eval_condition(
    agent: REEAgent,
    env: CausalGridWorld,
    oscillator_active: bool,
    eval_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    base_threshold: float,
    label: str,
) -> Dict:
    """
    Evaluate agent for eval_episodes with or without oscillator.

    Replaces agent.clock in place; restores after eval.
    During sweep steps (oscillator_active=True):
      agent.e3.config.commitment_threshold is temporarily lowered.
    """
    # Install appropriate clock
    agent.clock = _make_clock(oscillator_active)

    sweep_z_worlds: List[torch.Tensor] = []
    inter_z_worlds: List[torch.Tensor] = []
    sweep_committed: List[bool] = []
    inter_committed: List[bool] = []
    harm_preds_inter: List[float] = []
    n_sweep_steps = 0

    effective_sweep_threshold = base_threshold * (1.0 - SWEEP_AMPLITUDE)

    agent.eval()
    with torch.no_grad():
        for ep_idx in range(eval_episodes):
            _, obs_dict = env.reset()
            agent.reset()

            for _ in range(steps_per_episode):
                obs_body = obs_dict["body_state"].detach().clone().float()
                obs_world = obs_dict["world_state"].detach().clone().float()

                latent = agent.sense(obs_body, obs_world)
                ticks = agent.clock.advance()
                sweep = agent.clock.sweep_active

                # MECH-108: apply reduced threshold during sweep (oscillator_active only)
                if oscillator_active and sweep:
                    agent.e3.config.commitment_threshold = effective_sweep_threshold
                else:
                    agent.e3.config.commitment_threshold = base_threshold

                e1_prior = (
                    agent._e1_tick(latent)
                    if ticks["e1_tick"]
                    else torch.zeros(1, world_dim, device=agent.device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks, temperature=1.0)

                # Record z_world and commitment state (instantaneous check)
                z_w = latent.z_world.detach().squeeze(0)
                rv = agent.e3._running_variance
                is_committed = bool(rv < agent.e3.config.commitment_threshold)

                if sweep:
                    sweep_z_worlds.append(z_w)
                    sweep_committed.append(is_committed)
                    n_sweep_steps += 1
                else:
                    inter_z_worlds.append(z_w)
                    inter_committed.append(is_committed)
                    # Collect harm predictions for C4
                    h = agent.e3.harm_eval(z_w.unsqueeze(0)).item()
                    harm_preds_inter.append(h)

                _, _, done, _, obs_dict = env.step(action)
                if done:
                    break

    # Restore threshold
    agent.e3.config.commitment_threshold = base_threshold

    # Compute pooled z_world variance per phase
    def _pool_var(pool: List[torch.Tensor]) -> float:
        if len(pool) < 2:
            return 0.0
        stacked = torch.stack(pool)  # [N, world_dim]
        return float(stacked.var(dim=0).mean().item())

    z_world_var_sweep = _pool_var(sweep_z_worlds)
    z_world_var_inter = _pool_var(inter_z_worlds)

    committed_rate_sweep = (
        sum(sweep_committed) / max(1, len(sweep_committed))
    )
    committed_rate_inter = (
        sum(inter_committed) / max(1, len(inter_committed))
    )

    harm_pred_std_inter = 0.0
    if len(harm_preds_inter) > 1:
        vals = torch.tensor(harm_preds_inter)
        harm_pred_std_inter = float(vals.std().item())

    print(
        f"  [eval/{label}] sweep_steps={n_sweep_steps}"
        f" z_world_var[sweep]={z_world_var_sweep:.5f}"
        f" z_world_var[inter]={z_world_var_inter:.5f}"
        f" committed_rate[sweep]={committed_rate_sweep:.3f}"
        f" committed_rate[inter]={committed_rate_inter:.3f}"
        f" harm_pred_std_inter={harm_pred_std_inter:.4f}",
        flush=True,
    )

    return {
        "label": label,
        "oscillator_active": oscillator_active,
        "n_sweep_steps": n_sweep_steps,
        "n_inter_steps": len(inter_z_worlds),
        "z_world_var_sweep": z_world_var_sweep,
        "z_world_var_inter": z_world_var_inter,
        "z_world_var_ratio": (
            z_world_var_sweep / max(z_world_var_inter, 1e-8)
        ),
        "committed_rate_sweep": committed_rate_sweep,
        "committed_rate_inter": committed_rate_inter,
        "harm_pred_std_inter": harm_pred_std_inter,
    }


def run(
    seed: int = 42,
    train_episodes: int = 100,
    eval_episodes: int = 2,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    alpha_world: float = 0.9,
    lr: float = 1e-3,
    smoke: bool = False,
    smoke_out_dir: str = "",
    **kwargs,
) -> dict:
    """
    MECH-029 BreathOscillator uncommitted-window test.

    Trains one agent (with oscillator active during training).
    Then evaluates twice: once with oscillator (condition A) and once without (B).
    Compares sweep-phase vs inter-sweep z_world variance within condition A.
    """
    if smoke:
        train_episodes = 3
        eval_episodes = 1
        steps_per_episode = 100
        world_dim = 8
        self_dim = 8

    print(
        f"\n[EXQ-080] MECH-029 BreathOscillator Uncommitted Windows",
        flush=True,
    )
    print(
        f"  seed={seed} train_eps={train_episodes}"
        f" eval_eps={eval_episodes} steps={steps_per_episode}"
        f" breath_period={BREATH_PERIOD} sweep_dur={SWEEP_DURATION}"
        f" sweep_amp={SWEEP_AMPLITUDE} e3_rate={E3_RATE}",
        flush=True,
    )

    torch.manual_seed(seed)
    random.seed(seed)

    env = _make_env(seed)
    config = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=0.3,
    )
    agent = REEAgent(config)

    # Install breath-enabled clock for training (oscillator active during training)
    agent.clock = _make_clock(oscillator_active=True)

    base_threshold = agent.e3.config.commitment_threshold  # 0.40

    print(
        f"  base_commit_threshold={base_threshold:.3f}"
        f"  effective_sweep_threshold={base_threshold * (1 - SWEEP_AMPLITUDE):.3f}"
        f"  precision_init={agent.e3._running_variance:.3f}",
        flush=True,
    )

    # TRAIN
    print(f"\n[EXQ-080] Training ({train_episodes} eps)...", flush=True)
    _train(
        agent=agent,
        env=env,
        episodes=train_episodes,
        steps_per_episode=steps_per_episode,
        world_dim=world_dim,
        lr=lr,
        seed=seed,
        label="train",
    )
    print(
        f"  [post-train] running_variance={agent.e3._running_variance:.4f}"
        f"  (threshold={base_threshold:.3f},"
        f" sweep_threshold={base_threshold*(1-SWEEP_AMPLITUDE):.3f})",
        flush=True,
    )

    # EVAL condition A: oscillator active
    print(f"\n[EXQ-080] Eval condition A (oscillator active)...", flush=True)
    result_A = _eval_condition(
        agent=agent,
        env=env,
        oscillator_active=True,
        eval_episodes=eval_episodes,
        steps_per_episode=steps_per_episode,
        world_dim=world_dim,
        base_threshold=base_threshold,
        label="A_oscillator",
    )

    # EVAL condition B: oscillator inactive (same agent, control)
    print(f"\n[EXQ-080] Eval condition B (oscillator inactive, control)...", flush=True)
    result_B = _eval_condition(
        agent=agent,
        env=env,
        oscillator_active=False,
        eval_episodes=eval_episodes,
        steps_per_episode=steps_per_episode,
        world_dim=world_dim,
        base_threshold=base_threshold,
        label="B_control",
    )

    # --- Criteria evaluation (condition A) ---
    c1_pass = result_A["z_world_var_ratio"] > C1_RATIO
    c2_pass = (
        result_A["committed_rate_sweep"]
        < result_A["committed_rate_inter"] * C2_RATIO
    )
    c3_pass = result_A["n_sweep_steps"] >= C3_MIN_SWEEP_STEPS
    c4_pass = result_A["harm_pred_std_inter"] > C4_MIN_HARM_STD
    c5_pass = True  # no fatal errors (would propagate as exception)

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])
    status = "PASS" if all_pass else "FAIL"

    print(f"\n[EXQ-080] Results:", flush=True)
    print(
        f"  Cond A z_world_var: sweep={result_A['z_world_var_sweep']:.5f}"
        f" inter={result_A['z_world_var_inter']:.5f}"
        f" ratio={result_A['z_world_var_ratio']:.4f}",
        flush=True,
    )
    print(
        f"  Cond A committed_rate: sweep={result_A['committed_rate_sweep']:.3f}"
        f" inter={result_A['committed_rate_inter']:.3f}",
        flush=True,
    )
    print(
        f"  Cond B z_world_var_ratio={result_B['z_world_var_ratio']:.4f}"
        f" (control, expect ~1.0)",
        flush=True,
    )
    print(
        f"  n_sweep_A={result_A['n_sweep_steps']}"
        f"  harm_pred_std_inter={result_A['harm_pred_std_inter']:.4f}",
        flush=True,
    )
    print(f"  Status: {status} ({criteria_met}/5)", flush=True)

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: z_world_var_ratio={result_A['z_world_var_ratio']:.4f}"
            f" <= {C1_RATIO}."
            f" Possible causes: sweep threshold ({base_threshold*(1-SWEEP_AMPLITUDE):.3f})"
            f" above running_variance ({agent.e3._running_variance:.4f})"
            f" -- agent still committed during sweep phases."
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: committed_rate_sweep={result_A['committed_rate_sweep']:.3f}"
            f" >= {C2_RATIO} * inter={result_A['committed_rate_inter']:.3f}."
            f" Agent remained committed during sweep -- variance too low for threshold reduction."
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: n_sweep_steps={result_A['n_sweep_steps']} < {C3_MIN_SWEEP_STEPS}."
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: harm_pred_std_inter={result_A['harm_pred_std_inter']:.5f}"
            f" <= {C4_MIN_HARM_STD}. E3 harm head appears collapsed (constant output)."
        )
    for n in failure_notes:
        print(f"  {n}", flush=True)

    if all_pass:
        interpretation = (
            "MECH-029 SUPPORTED: BreathOscillator-forced uncommitted windows show"
            " higher z_world variability than inter-sweep committed intervals (C1)"
            " and lower committed rate (C2), confirming that periodic uncommitted"
            " windows produce measurably distinct processing. This validates the"
            " MECH-029 claim that uncommitted mode represents genuine variability"
            " in world-model engagement, not merely lower action precision."
            " LIT-0094 should determine whether breath_period=50 matches"
            " physiologically-grounded respiratory coupling frequencies."
        )
    elif criteria_met >= 3:
        interpretation = (
            "MECH-029 PARTIAL: Some uncommitted-window signal present but below"
            " threshold. Suggest increasing sweep_amplitude or reducing agent"
            " training depth to preserve higher running_variance."
        )
    else:
        interpretation = (
            "MECH-029 NOT SUPPORTED by this design: BreathOscillator could not"
            " reliably disrupt the committed state. Likely cause: agent's"
            " running_variance has fallen below sweep_threshold, making the"
            " mechanism ineffective. Consider using a stronger sweep_amplitude"
            " or a different oscillator-forcing mechanism (e.g., direct variance"
            " injection rather than threshold reduction)."
        )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-080 -- MECH-029 BreathOscillator Uncommitted Windows\n\n"
        f"**Status:** {status}\n**Claims:** MECH-029\n"
        f"**Seed:** {seed}  **Train:** {train_episodes} eps  **Eval:** {eval_episodes} eps"
        f"  **Steps/ep:** {steps_per_episode}\n"
        f"**BreathOscillator:** period={BREATH_PERIOD}, sweep_dur={SWEEP_DURATION},"
        f" sweep_amp={SWEEP_AMPLITUDE}, e3_rate={E3_RATE}\n\n"
        f"## Results\n\n"
        f"| Metric | Condition A (oscillator) | Condition B (control) |\n"
        f"|---|---|---|\n"
        f"| z_world_var[sweep] | {result_A['z_world_var_sweep']:.5f} | {result_B['z_world_var_sweep']:.5f} |\n"
        f"| z_world_var[inter] | {result_A['z_world_var_inter']:.5f} | {result_B['z_world_var_inter']:.5f} |\n"
        f"| z_world_var_ratio | {result_A['z_world_var_ratio']:.4f} | {result_B['z_world_var_ratio']:.4f} |\n"
        f"| committed_rate[sweep] | {result_A['committed_rate_sweep']:.3f} | {result_B['committed_rate_sweep']:.3f} |\n"
        f"| committed_rate[inter] | {result_A['committed_rate_inter']:.3f} | {result_B['committed_rate_inter']:.3f} |\n"
        f"| n_sweep_steps | {result_A['n_sweep_steps']} | {result_B['n_sweep_steps']} |\n"
        f"| harm_pred_std_inter | {result_A['harm_pred_std_inter']:.4f} | {result_B['harm_pred_std_inter']:.4f} |\n\n"
        f"**Post-train running_variance: {agent.e3._running_variance:.4f}**"
        f" (base_threshold={base_threshold:.3f},"
        f" sweep_threshold={base_threshold*(1-SWEEP_AMPLITUDE):.3f})\n\n"
        f"## PASS Criteria\n\n| Criterion | Threshold | Result |\n|---|---|---|\n"
        f"| C1: z_world_var ratio > {C1_RATIO} | {C1_RATIO:.2f} | {'PASS' if c1_pass else 'FAIL'} |\n"
        f"| C2: committed_rate[sweep] < inter*{C2_RATIO} | {C2_RATIO:.2f} | {'PASS' if c2_pass else 'FAIL'} |\n"
        f"| C3: n_sweep_steps >= {C3_MIN_SWEEP_STEPS} | {C3_MIN_SWEEP_STEPS} | {'PASS' if c3_pass else 'FAIL'} |\n"
        f"| C4: harm_pred_std_inter > {C4_MIN_HARM_STD} | {C4_MIN_HARM_STD} | {'PASS' if c4_pass else 'FAIL'} |\n"
        f"| C5: no fatal errors | -- | {'PASS' if c5_pass else 'FAIL'} |\n\n"
        f"Criteria met: {criteria_met}/5 -> **{status}**\n\n"
        f"## Interpretation\n\n{interpretation}\n{failure_section}\n"
    )

    metrics = {
        "z_world_var_sweep_A": float(result_A["z_world_var_sweep"]),
        "z_world_var_inter_A": float(result_A["z_world_var_inter"]),
        "z_world_var_ratio_A": float(result_A["z_world_var_ratio"]),
        "z_world_var_ratio_B": float(result_B["z_world_var_ratio"]),
        "committed_rate_sweep_A": float(result_A["committed_rate_sweep"]),
        "committed_rate_inter_A": float(result_A["committed_rate_inter"]),
        "committed_rate_sweep_B": float(result_B["committed_rate_sweep"]),
        "committed_rate_inter_B": float(result_B["committed_rate_inter"]),
        "harm_pred_std_inter_A": float(result_A["harm_pred_std_inter"]),
        "n_sweep_steps_A": float(result_A["n_sweep_steps"]),
        "n_sweep_steps_B": float(result_B["n_sweep_steps"]),
        "post_train_running_variance": float(agent.e3._running_variance),
        "base_commit_threshold": float(base_threshold),
        "sweep_threshold": float(base_threshold * (1.0 - SWEEP_AMPLITUDE)),
        "breath_period": float(BREATH_PERIOD),
        "sweep_duration": float(SWEEP_DURATION),
        "sweep_amplitude": float(SWEEP_AMPLITUDE),
        "e3_steps_per_tick": float(E3_RATE),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "crit5_pass": 1.0 if c5_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if criteria_met >= 3 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--train-episodes", type=int,   default=100)
    parser.add_argument("--eval-episodes",  type=int,   default=2)
    parser.add_argument("--steps",          type=int,   default=200)
    parser.add_argument("--world-dim",      type=int,   default=32)
    parser.add_argument("--alpha-world",    type=float, default=0.9)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke test (3 eps, world_dim=8, no JSON output to REE_assembly)",
    )
    parser.add_argument(
        "--smoke-out-dir",
        type=str,
        default="",
        help="Output dir for smoke test JSON (default: /tmp/exq080_smoke)",
    )
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        steps_per_episode=args.steps,
        self_dim=args.world_dim,
        world_dim=args.world_dim,
        alpha_world=args.alpha_world,
        lr=args.lr,
        smoke=args.smoke,
        smoke_out_dir=args.smoke_out_dir,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    if args.smoke:
        smoke_dir_str = args.smoke_out_dir or "/tmp/exq080_smoke"
        out_dir = Path(smoke_dir_str)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[SMOKE] Writing to {out_dir}", flush=True)
    else:
        out_dir = (
            Path(__file__).resolve().parents[2]
            / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
        )
        out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}", flush=True)
