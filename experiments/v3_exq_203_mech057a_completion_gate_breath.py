#!/opt/local/bin/python3
"""
V3-EXQ-203 -- MECH-057a Completion Gate with BreathOscillator

Claim: MECH-057a
Supersedes: V3-EXQ-139

MECH-057a asserts: "Committed action sequences suppress new precision updates until
execution completes."

Key substrate fix vs EXQ-139: BreathOscillator (MECH-108) creates periodic
uncommitted windows so we can actually compare committed vs uncommitted states.
Without it, trained agents tend to converge to permanently committed, making the
comparison degenerate. The breath cycle (breath_period=50, sweep_duration=10,
sweep_amplitude=0.30) periodically lowers the effective commit_threshold, forcing
the agent into uncommitted mode for 10/50 steps (~20% of the time).

Discriminative pair
--------------------
GATE_ON (MECH-057a architecture):
  - BetaGate active: elevated while committed, released when uncommitted
  - running_variance update SUPPRESSED during committed steps
  - Accumulated prediction errors flushed as a single batch at completion
    boundary (transition to uncommitted state)
  - BreathOscillator enabled: periodic uncommitted windows

GATE_ABLATED (continuous update, ablation):
  - BetaGate disabled: after each act(), force beta_gate.release()
  - running_variance updated every step regardless of commitment state
  - No batching at sequence boundary -- continuous precision updating
  - BreathOscillator enabled (same config): matching env dynamics

Both conditions train the same way (400 eps warmup), then eval for 50 eps.
The discriminative manipulation is whether the beta gate + precision suppression
are active during eval.

Pre-registered acceptance criteria (need 4/5):
  C1: harm_rate_on <= harm_rate_ablated * 0.90 (both seeds)
      Gate-ON reduces harm by >=10% relative to ablation.
  C2: per-seed absolute gap >= 0.005 (both seeds)
      Absolute harm rate reduction >= 0.5pp per seed.
  C3: n_harm_contact_min >= 20 per cell (both seeds, both conditions)
      Sufficient harm contacts in eval for reliable rate estimation.
  C4: n_committed_steps >= 30 in GATE_ON eval per seed
      Enough committed-state steps to confirm commitment dynamics are present.
  C5: no fatal errors across all cells.

Decision scoring:
  PASS (4+ of 5): retain_ree -- completion-gated precision suppression reduces harm
  C1 fails, C3+C4 pass: retire_ree_claim -- no harm benefit from completion gating
  C4 fails: hold -- agent not reaching committed state

If this experiment PASSes, which claim does that support and why?
  MECH-057a: completion-gated precision suppression (updating only at sequence
  boundary) produces lower harm rates than continuous per-step updating. The
  BreathOscillator provides natural uncommitted windows that make the comparison
  non-degenerate.

Seeds: [42, 123] (matched -- same env per seed across conditions)
Env:   CausalGridWorldV2, size=6, 4 hazards, nav_bias=0.45
Warmup: 400 eps x 200 steps
Eval:  50 eps x 200 steps (no training, fixed weights)
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_203_mech057a_completion_gate_breath"
CLAIM_IDS = ["MECH-057a"]

# Pre-registered thresholds
THRESH_C1_HARM_RATIO = 0.90    # gate_on harm_rate <= ablated * 0.90 (10% relative reduction)
THRESH_C2_ABS_GAP    = 0.005   # absolute harm_rate gap >= 0.005 per seed
THRESH_C3_N_HARM     = 20      # n_harm_contact_min >= 20 per cell
THRESH_C4_N_COMMITTED = 30     # n_committed_steps_total >= 30 in eval (GATE_ON, per seed)
PASS_THRESHOLD       = 4       # need 4/5 criteria


def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _hazard_approach_action(env: CausalGridWorldV2, n_actions: int) -> int:
    """Return action index biased toward nearest hazard gradient (nav_bias helper)."""
    obs_dict = env._get_observation_dict()
    world_state = obs_dict.get("world_state", None)
    if world_state is None or not getattr(env, "use_proxy_fields", False):
        return random.randint(0, n_actions - 1)
    # world_state[225:250] = hazard_field_view (5x5 flattened)
    try:
        field_view = world_state[225:250].numpy().reshape(5, 5)
    except Exception:
        return random.randint(0, n_actions - 1)
    # Agent at center (2,2); actions: 0=up(-1,0), 1=down(+1,0), 2=left(0,-1), 3=right(0,+1)
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    vals = []
    for dr, dc in deltas:
        r, c = 2 + dr, 2 + dc
        if 0 <= r < 5 and 0 <= c < 5:
            vals.append(float(field_view[r, c]))
        else:
            vals.append(-1.0)
    return int(np.argmax(vals))


def _run_cell(
    seed: int,
    gate_on: bool,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    lr: float,
    alpha_world: float,
    alpha_self: float,
    harm_scale: float,
    proximity_harm_scale: float,
    nav_bias: float,
    breath_period: int,
    breath_sweep_amplitude: float,
    breath_sweep_duration: int,
    dry_run: bool,
) -> Dict:
    """
    Run one (seed, condition) cell and return harm rate + commitment metrics.

    gate_on=True:  GATE_ON -- precision updates suppressed during commitment,
                   flushed as batch at completion boundary. BetaGate active.
                   BreathOscillator creates periodic uncommitted windows.
    gate_on=False: GATE_ABLATED -- continuous per-step precision updates,
                   BetaGate forced released after each step. BreathOscillator
                   still active (matching env dynamics).
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    cond_label = "GATE_ON" if gate_on else "GATE_ABLATED"

    env = CausalGridWorldV2(
        seed=seed,
        size=6,
        num_hazards=4,
        num_resources=3,
        hazard_harm=harm_scale,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=proximity_harm_scale,
        proximity_benefit_scale=proximity_harm_scale * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        use_proxy_fields=True,
    )
    n_actions = env.action_dim
    world_obs_dim = env.world_obs_dim
    body_obs_dim = env.body_obs_dim

    config = REEConfig.from_dims(
        body_obs_dim=body_obs_dim,
        world_obs_dim=world_obs_dim,
        action_dim=n_actions,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=0,
        use_event_classifier=True,   # SD-009
    )
    # BreathOscillator config (MECH-108)
    config.heartbeat.breath_period = breath_period
    config.heartbeat.breath_sweep_amplitude = breath_sweep_amplitude
    config.heartbeat.breath_sweep_duration = breath_sweep_duration

    agent = REEAgent(config)

    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n
    ]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    optimizer           = optim.Adam(standard_params, lr=lr)
    harm_eval_optimizer = optim.Adam(harm_eval_params, lr=1e-4)

    # Harm-eval replay buffer
    harm_buf_zw:     List[torch.Tensor] = []
    harm_buf_labels: List[float]        = []
    MAX_BUF = 2000

    # For GATE_ON: accumulate prediction errors during committed phases;
    # flush to running_variance at completion boundary.
    pending_errors: List[float] = []
    was_committed_prev = False

    if dry_run:
        actual_warmup = min(3, warmup_episodes)
        actual_eval   = min(2, eval_episodes)
    else:
        actual_warmup = warmup_episodes
        actual_eval   = eval_episodes

    train_harm_count = 0
    train_step_count = 0

    # ------------------------------------------------------------------ TRAIN
    agent.train()

    for ep in range(actual_warmup):
        _, obs_dict = env.reset()
        agent.reset()
        pending_errors_ep: List[float] = []
        was_committed_ep = False

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_curr = latent.z_world.detach()

            # Commitment state: check running_variance vs effective threshold.
            # During BreathOscillator sweep, effective threshold is lowered.
            sweep_reduction = (
                agent.clock.sweep_amplitude if agent.clock.sweep_active else 0.0
            )
            effective_threshold = (
                agent.e3.commit_threshold * (1.0 - sweep_reduction)
            )
            is_committed = (
                agent.e3._running_variance < effective_threshold
            )

            if gate_on:
                # GATE_ON: elevate/release BetaGate based on commitment
                if is_committed:
                    agent.beta_gate.elevate()
                else:
                    agent.beta_gate.release()
                    # Flush accumulated errors at completion boundary
                    if was_committed_ep and pending_errors_ep:
                        mean_err = sum(pending_errors_ep) / len(pending_errors_ep)
                        agent.e3._running_variance = (
                            (1 - agent.e3._ema_alpha) * agent.e3._running_variance
                            + agent.e3._ema_alpha * mean_err
                        )
                        pending_errors_ep = []
                was_committed_ep = is_committed
            else:
                # GATE_ABLATED: BetaGate always released, no gating
                agent.beta_gate.release()

            # Action selection
            if random.random() < nav_bias:
                action_idx = _hazard_approach_action(env, n_actions)
            else:
                action_idx = random.randint(0, n_actions - 1)
            action = _action_to_onehot(action_idx, n_actions, agent.device)
            agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)
            harm_sig_float = float(harm_signal)
            is_harm = harm_sig_float < 0
            train_step_count += 1
            if is_harm:
                train_harm_count += 1

            # E1 + E2 training step
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # Precision update (running_variance)
            with torch.no_grad():
                if latent.z_world is not None and z_world_curr is not None:
                    pred_err = float(
                        F.mse_loss(latent.z_world.detach(), z_world_curr).item()
                    )
                else:
                    pred_err = 0.0

            if gate_on and is_committed:
                # Suppress variance update; accumulate instead
                pending_errors_ep.append(pred_err)
            else:
                # Continuous update (ABLATED, or GATE_ON when uncommitted)
                agent.e3._running_variance = (
                    (1 - agent.e3._ema_alpha) * agent.e3._running_variance
                    + agent.e3._ema_alpha * pred_err
                )

            # Harm eval training
            harm_buf_zw.append(z_world_curr)
            harm_buf_labels.append(1.0 if is_harm else 0.0)
            if len(harm_buf_zw) > MAX_BUF:
                harm_buf_zw     = harm_buf_zw[-MAX_BUF:]
                harm_buf_labels = harm_buf_labels[-MAX_BUF:]

            n_pos = sum(1 for l in harm_buf_labels if l > 0.5)
            n_neg = sum(1 for l in harm_buf_labels if l <= 0.5)
            if n_pos >= 4 and n_neg >= 4:
                pos_idx = [i for i, l in enumerate(harm_buf_labels) if l > 0.5]
                neg_idx = [i for i, l in enumerate(harm_buf_labels) if l <= 0.5]
                k = min(8, min(len(pos_idx), len(neg_idx)))
                sel = random.sample(pos_idx, k) + random.sample(neg_idx, k)
                zw_b = torch.cat([harm_buf_zw[i] for i in sel], dim=0)
                lb_b = torch.tensor(
                    [harm_buf_labels[i] for i in sel],
                    dtype=torch.float32,
                ).unsqueeze(1)
                pred_he = agent.e3.harm_eval(zw_b)
                loss_he = F.mse_loss(pred_he, lb_b)
                if loss_he.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    loss_he.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5
                    )
                    harm_eval_optimizer.step()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == actual_warmup - 1:
            print(
                f"  [train] seed={seed} cond={cond_label}"
                f" ep {ep+1}/{actual_warmup}"
                f" train_harm={train_harm_count} train_steps={train_step_count}"
                f" running_var={agent.e3._running_variance:.4f}"
                f" committed={'yes' if is_committed else 'no'}"
                f" sweep_active={agent.clock.sweep_active}",
                flush=True,
            )

    # ------------------------------------------------------------------ EVAL
    agent.eval()

    eval_harm_contacts = 0
    eval_total_steps   = 0
    n_committed_steps  = 0
    n_fatal            = 0

    # Track committed-sequence lengths
    seq_start_step = None
    seq_lengths: List[int] = []
    step_in_eval  = 0

    for _ in range(actual_eval):
        _, obs_dict = env.reset()
        agent.reset()
        pending_errors_ep = []
        was_committed_ep  = False

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                z_world_curr = latent.z_world.detach()

            # Commitment with BreathOscillator sweep
            sweep_reduction = (
                agent.clock.sweep_amplitude if agent.clock.sweep_active else 0.0
            )
            effective_threshold = (
                agent.e3.commit_threshold * (1.0 - sweep_reduction)
            )
            is_committed = (
                agent.e3._running_variance < effective_threshold
            )

            if gate_on:
                if is_committed:
                    agent.beta_gate.elevate()
                    n_committed_steps += 1
                    if seq_start_step is None:
                        seq_start_step = step_in_eval
                else:
                    agent.beta_gate.release()
                    if was_committed_ep and pending_errors_ep:
                        mean_err = sum(pending_errors_ep) / len(pending_errors_ep)
                        agent.e3._running_variance = (
                            (1 - agent.e3._ema_alpha) * agent.e3._running_variance
                            + agent.e3._ema_alpha * mean_err
                        )
                        pending_errors_ep = []
                    if seq_start_step is not None and not is_committed:
                        seq_lengths.append(step_in_eval - seq_start_step)
                        seq_start_step = None
                was_committed_ep = is_committed
            else:
                # GATE_ABLATED: force release after each step
                agent.beta_gate.release()

            action_idx = random.randint(0, n_actions - 1)
            action = _action_to_onehot(action_idx, n_actions, agent.device)
            agent._last_action = action

            try:
                _, harm_signal, done, info, obs_dict = env.step(action)
            except Exception:
                n_fatal += 1
                break

            harm_sig_float = float(harm_signal)
            is_harm = harm_sig_float < 0

            eval_total_steps += 1
            step_in_eval     += 1
            if is_harm:
                eval_harm_contacts += 1

            # Precision update in eval
            with torch.no_grad():
                if latent.z_world is not None and z_world_curr is not None:
                    pred_err = float(
                        F.mse_loss(latent.z_world.detach(), z_world_curr).item()
                    )
                else:
                    pred_err = 0.0

            if gate_on and is_committed:
                pending_errors_ep.append(pred_err)
            else:
                agent.e3._running_variance = (
                    (1 - agent.e3._ema_alpha) * agent.e3._running_variance
                    + agent.e3._ema_alpha * pred_err
                )

            if done:
                break

    harm_rate = float(eval_harm_contacts) / max(1, eval_total_steps)
    mean_seq_len = (
        float(sum(seq_lengths) / max(1, len(seq_lengths)))
        if seq_lengths else 0.0
    )

    print(
        f"  [eval] seed={seed} cond={cond_label}"
        f" harm_rate={harm_rate:.4f}"
        f" harm_contacts={eval_harm_contacts}/{eval_total_steps}"
        f" n_committed_steps={n_committed_steps}"
        f" n_seq={len(seq_lengths)} mean_seq_len={mean_seq_len:.1f}"
        f" n_fatal={n_fatal}",
        flush=True,
    )

    return {
        "seed":                  seed,
        "condition":             cond_label,
        "gate_on":               gate_on,
        "harm_rate":             float(harm_rate),
        "harm_contacts":         int(eval_harm_contacts),
        "total_eval_steps":      int(eval_total_steps),
        "n_committed_steps":     int(n_committed_steps),
        "n_committed_sequences": int(len(seq_lengths)),
        "mean_committed_seq_len": float(mean_seq_len),
        "n_fatal":               int(n_fatal),
        "train_harm_count":      int(train_harm_count),
    }


def run(
    seeds: Tuple = (42, 123),
    warmup_episodes: int = 400,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_harm_scale: float = 0.05,
    nav_bias: float = 0.45,
    breath_period: int = 50,
    breath_sweep_amplitude: float = 0.30,
    breath_sweep_duration: int = 10,
    dry_run: bool = False,
    **kwargs,
) -> dict:
    """
    Discriminative pair: GATE_ON (precision updates suppressed during commitment,
    flushed at completion boundary -- MECH-057a architecture + BreathOscillator)
    vs GATE_ABLATED (continuous per-step precision updates, beta gate forced
    released).

    Tests MECH-057a: committed action sequences suppress precision updates until
    execution completes; completion is the principal policy-propagation event.
    BreathOscillator (MECH-108) ensures periodic uncommitted windows exist.
    """
    results_on:      List[Dict] = []
    results_ablated: List[Dict] = []

    for seed in seeds:
        for gate_on in [True, False]:
            label = "GATE_ON" if gate_on else "GATE_ABLATED"
            print(
                f"\n[V3-EXQ-203] {label} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" nav_bias={nav_bias}"
                f" breath_period={breath_period}"
                f" sweep_amp={breath_sweep_amplitude}"
                f" sweep_dur={breath_sweep_duration}",
                flush=True,
            )
            r = _run_cell(
                seed=seed,
                gate_on=gate_on,
                warmup_episodes=warmup_episodes,
                eval_episodes=eval_episodes,
                steps_per_episode=steps_per_episode,
                self_dim=self_dim,
                world_dim=world_dim,
                lr=lr,
                alpha_world=alpha_world,
                alpha_self=alpha_self,
                harm_scale=harm_scale,
                proximity_harm_scale=proximity_harm_scale,
                nav_bias=nav_bias,
                breath_period=breath_period,
                breath_sweep_amplitude=breath_sweep_amplitude,
                breath_sweep_duration=breath_sweep_duration,
                dry_run=dry_run,
            )
            if gate_on:
                results_on.append(r)
            else:
                results_ablated.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    harm_rate_on      = _avg(results_on,      "harm_rate")
    harm_rate_ablated = _avg(results_ablated, "harm_rate")
    delta_harm_rate   = harm_rate_ablated - harm_rate_on   # positive = gate_on better

    n_harm_min = min(r["harm_contacts"] for r in results_on + results_ablated)
    n_committed_min = min(r["n_committed_steps"] for r in results_on)

    # Per-seed harm rate ratio and absolute gap
    per_seed_ratios: List[float] = []
    per_seed_gaps:   List[float] = []
    for r_on in results_on:
        matching = [r for r in results_ablated if r["seed"] == r_on["seed"]]
        if matching:
            abl_rate = matching[0]["harm_rate"]
            on_rate  = r_on["harm_rate"]
            ratio = on_rate / max(abl_rate, 1e-9)
            gap   = abl_rate - on_rate
            per_seed_ratios.append(ratio)
            per_seed_gaps.append(gap)

    # Pre-registered PASS criteria
    # C1: harm_rate_gate_on <= harm_rate_ablated * 0.90 (both seeds)
    c1_pass = all(r <= THRESH_C1_HARM_RATIO for r in per_seed_ratios)
    # C2: per-seed absolute gap >= THRESH_C2_ABS_GAP (both seeds)
    c2_pass = len(per_seed_gaps) > 0 and all(g >= THRESH_C2_ABS_GAP for g in per_seed_gaps)
    # C3: n_harm_contact_min >= THRESH_C3_N_HARM (both seeds, both conditions)
    c3_pass = n_harm_min >= THRESH_C3_N_HARM
    # C4: n_committed_steps_total >= THRESH_C4_N_COMMITTED per seed (GATE_ON)
    c4_pass = n_committed_min >= THRESH_C4_N_COMMITTED
    # C5: no fatal errors
    c5_pass = all(r["n_fatal"] == 0 for r in results_on + results_ablated)

    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])
    all_pass     = criteria_met >= PASS_THRESHOLD
    status       = "PASS" if all_pass else "FAIL"

    if all_pass:
        decision = "retain_ree"
    elif c3_pass and c4_pass and not c1_pass:
        decision = "retire_ree_claim"
    elif not c4_pass:
        decision = "hold"
    else:
        decision = "retire_ree_claim"

    print(f"\n[V3-EXQ-203] Results:", flush=True)
    print(
        f"  harm_rate_on={harm_rate_on:.4f}"
        f" harm_rate_ablated={harm_rate_ablated:.4f}"
        f" delta={delta_harm_rate:+.4f}",
        flush=True,
    )
    print(
        f"  per_seed_ratios={[round(r, 4) for r in per_seed_ratios]}"
        f" per_seed_gaps={[round(g, 4) for g in per_seed_gaps]}",
        flush=True,
    )
    print(
        f"  n_harm_min={n_harm_min}"
        f" n_committed_min={n_committed_min}"
        f"  decision={decision}  status={status} ({criteria_met}/5)",
        flush=True,
    )

    failure_notes: List[str] = []
    if not c1_pass:
        failing_seeds = [
            r_on["seed"]
            for r_on, ratio in zip(results_on, per_seed_ratios)
            if ratio > THRESH_C1_HARM_RATIO
        ]
        failure_notes.append(
            f"C1 FAIL: harm_rate ratio > {THRESH_C1_HARM_RATIO} in seeds {failing_seeds}"
            " -- completion-gated precision suppression does not reduce harm rate by >=10%"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: per-seed gaps {[round(g, 4) for g in per_seed_gaps]}"
            f" < {THRESH_C2_ABS_GAP}"
            " -- absolute harm rate difference below threshold"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: n_harm_min={n_harm_min} < {THRESH_C3_N_HARM}"
            " -- insufficient harm contacts for reliable rate estimation"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: n_committed_min={n_committed_min} < {THRESH_C4_N_COMMITTED}"
            " -- insufficient committed steps in GATE_ON eval; agent may not be reaching"
            " committed state. Consider increasing warmup or lowering commit_threshold."
        )
    if not c5_pass:
        fatal_by_cell = {r["condition"] + "_" + str(r["seed"]): r["n_fatal"]
                         for r in results_on + results_ablated if r["n_fatal"] > 0}
        failure_notes.append(f"C5 FAIL: fatal errors detected -- {fatal_by_cell}")

    for note in failure_notes:
        print(f"  {note}", flush=True)

    if all_pass:
        interpretation = (
            "MECH-057a SUPPORTED: completion-gated precision suppression (GATE_ON)"
            f" produced lower harm_rate ({harm_rate_on:.4f}) than continuous updating"
            f" (GATE_ABLATED: {harm_rate_ablated:.4f}), delta={delta_harm_rate:+.4f}"
            f" across {len(seeds)} seeds. Per-seed ratios"
            f" {[round(r, 4) for r in per_seed_ratios]} meet"
            f" threshold {THRESH_C1_HARM_RATIO}. BreathOscillator"
            f" (period={breath_period}, sweep_amp={breath_sweep_amplitude},"
            f" sweep_dur={breath_sweep_duration}) provided periodic uncommitted"
            " windows enabling non-degenerate comparison. Suppressing precision"
            " updates during committed sequences and flushing at completion boundary"
            " reduces mid-sequence noise, improving harm avoidance -- consistent"
            " with MECH-057a's assertion that committed sequences suppress precision"
            " updates until execution completes."
        )
    elif c3_pass and c4_pass and not c1_pass:
        interpretation = (
            "MECH-057a NOT supported at V3 proxy: GATE_ON harm_rate"
            f" ({harm_rate_on:.4f}) not significantly lower than GATE_ABLATED"
            f" ({harm_rate_ablated:.4f}). Per-seed ratios"
            f" {[round(r, 4) for r in per_seed_ratios]} do not meet"
            f" threshold {THRESH_C1_HARM_RATIO}. Completion-gated precision suppression"
            " does not produce detectable harm reduction at this training scale."
            " Possible: the proxy (manual running_variance manipulation) does not"
            " faithfully model the biological completion-boundary mechanism; or the"
            " effect requires full ARC-023 + SD-006 substrate to manifest."
        )
    elif not c4_pass:
        interpretation = (
            f"Inconclusive: agent reached committed state insufficiently often"
            f" (n_committed_min={n_committed_min} < {THRESH_C4_N_COMMITTED})."
            " The completion gate mechanism cannot be tested if the agent does not"
            " enter committed sequences during eval. Consider increasing warmup"
            " or adjusting breath_sweep_amplitude to widen uncommitted windows."
        )
    else:
        interpretation = (
            f"MECH-057a NOT supported: criteria_met={criteria_met}/5."
            f" harm_rate_on={harm_rate_on:.4f},"
            f" harm_rate_ablated={harm_rate_ablated:.4f}."
        )

    per_on_rows = "\n".join(
        f"  seed={r['seed']}: harm_rate={r['harm_rate']:.4f}"
        f" contacts={r['harm_contacts']}/{r['total_eval_steps']}"
        f" n_committed={r['n_committed_steps']}"
        f" n_seq={r['n_committed_sequences']}"
        f" mean_seq_len={r['mean_committed_seq_len']:.1f}"
        for r in results_on
    )
    per_abl_rows = "\n".join(
        f"  seed={r['seed']}: harm_rate={r['harm_rate']:.4f}"
        f" contacts={r['harm_contacts']}/{r['total_eval_steps']}"
        for r in results_ablated
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-203 -- MECH-057a Completion Gate with BreathOscillator\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-057a\n"
        f"**Supersedes:** V3-EXQ-139\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**Conditions:** GATE_ON vs GATE_ABLATED\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps\n"
        f"**Env:** CausalGridWorldV2 size=6, 4 hazards, nav_bias={nav_bias}\n"
        f"**BreathOscillator:** period={breath_period},"
        f" sweep_amp={breath_sweep_amplitude},"
        f" sweep_dur={breath_sweep_duration}\n\n"
        f"## Design\n\n"
        f"MECH-057a asserts committed action sequences suppress precision updates until"
        f" execution completes. Completion events are principal policy-propagation"
        f" opportunities (beta drops, E3 state propagates to action selection).\n\n"
        f"Key improvement over EXQ-139: BreathOscillator (MECH-108) creates periodic"
        f" uncommitted windows (sweep_duration={breath_sweep_duration} steps every"
        f" {breath_period} steps), preventing degenerate always-committed states.\n\n"
        f"GATE_ON: running_variance update suppressed while committed;"
        f" accumulated errors flushed as batch at completion boundary (uncommitted"
        f" transition). BetaGate active (elevates when committed, releases at"
        f" completion). BreathOscillator-driven uncommitted windows.\n\n"
        f"GATE_ABLATED: continuous per-step running_variance updates;"
        f" BetaGate disabled (always released). BreathOscillator still active"
        f" (matching env dynamics).\n\n"
        f"## Pre-Registered Thresholds (need {PASS_THRESHOLD}/5)\n\n"
        f"C1: harm_rate_on <= harm_rate_ablated * {THRESH_C1_HARM_RATIO} (both seeds)\n"
        f"C2: per-seed absolute gap >= {THRESH_C2_ABS_GAP} (both seeds)\n"
        f"C3: n_harm_contact_min >= {THRESH_C3_N_HARM} per cell\n"
        f"C4: n_committed_steps_min >= {THRESH_C4_N_COMMITTED} in eval (GATE_ON, per seed)\n"
        f"C5: no fatal errors\n\n"
        f"## Results\n\n"
        f"| Condition | harm_rate (avg) | harm_contacts |\n"
        f"|-----------|----------------|---------------|\n"
        f"| GATE_ON      | {harm_rate_on:.4f}"
        f" | {sum(r['harm_contacts'] for r in results_on)} |\n"
        f"| GATE_ABLATED | {harm_rate_ablated:.4f}"
        f" | {sum(r['harm_contacts'] for r in results_ablated)} |\n\n"
        f"**delta harm_rate (ABLATED - ON): {delta_harm_rate:+.4f}**\n\n"
        f"Per-seed ratios (ON/ABLATED): {[round(r, 4) for r in per_seed_ratios]}\n"
        f"Per-seed absolute gaps (ABLATED-ON): {[round(g, 4) for g in per_seed_gaps]}\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: harm_rate ratio <= {THRESH_C1_HARM_RATIO} (both seeds)"
        f" | {'PASS' if c1_pass else 'FAIL'}"
        f" | {[round(r, 4) for r in per_seed_ratios]} |\n"
        f"| C2: per-seed gap >= {THRESH_C2_ABS_GAP} (both seeds)"
        f" | {'PASS' if c2_pass else 'FAIL'}"
        f" | {[round(g, 4) for g in per_seed_gaps]} |\n"
        f"| C3: n_harm_min >= {THRESH_C3_N_HARM}"
        f" | {'PASS' if c3_pass else 'FAIL'}"
        f" | {n_harm_min} |\n"
        f"| C4: n_committed_min >= {THRESH_C4_N_COMMITTED} (GATE_ON)"
        f" | {'PASS' if c4_pass else 'FAIL'}"
        f" | {n_committed_min} |\n"
        f"| C5: no fatal errors | {'PASS' if c5_pass else 'FAIL'} | -- |\n\n"
        f"Criteria met: {criteria_met}/5 -> **{status}** (need {PASS_THRESHOLD}/5)\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n\n"
        f"## Per-Seed\n\n"
        f"GATE_ON:\n{per_on_rows}\n\n"
        f"GATE_ABLATED:\n{per_abl_rows}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "harm_rate_gate_on":           float(harm_rate_on),
        "harm_rate_ablated":           float(harm_rate_ablated),
        "delta_harm_rate":             float(delta_harm_rate),
        "n_harm_min":                  float(n_harm_min),
        "n_committed_min":             float(n_committed_min),
        "n_seeds":                     float(len(seeds)),
        "nav_bias":                    float(nav_bias),
        "alpha_world":                 float(alpha_world),
        "breath_period":               float(breath_period),
        "breath_sweep_amplitude":      float(breath_sweep_amplitude),
        "breath_sweep_duration":       float(breath_sweep_duration),
        "per_seed_ratio_min":          float(min(per_seed_ratios)) if per_seed_ratios else 1.0,
        "per_seed_ratio_max":          float(max(per_seed_ratios)) if per_seed_ratios else 1.0,
        "per_seed_gap_min":            float(min(per_seed_gaps)) if per_seed_gaps else 0.0,
        "per_seed_gap_max":            float(max(per_seed_gaps)) if per_seed_gaps else 0.0,
        "mean_committed_seq_len_on":   float(_avg(results_on, "mean_committed_seq_len")),
        "crit1_pass":                  1.0 if c1_pass else 0.0,
        "crit2_pass":                  1.0 if c2_pass else 0.0,
        "crit3_pass":                  1.0 if c3_pass else 0.0,
        "crit4_pass":                  1.0 if c4_pass else 0.0,
        "crit5_pass":                  1.0 if c5_pass else 0.0,
        "criteria_met":                float(criteria_met),
    }

    return {
        "status":           status,
        "metrics":          metrics,
        "summary_markdown": summary_markdown,
        "claim_ids":        CLAIM_IDS,
        "supersedes":       "V3-EXQ-139",
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if criteria_met >= 3 else "weakens")
        ),
        "experiment_type":  EXPERIMENT_TYPE,
        "fatal_error_count": sum(r["n_fatal"] for r in results_on + results_ablated),
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",           type=int, nargs="+", default=[42, 123])
    parser.add_argument("--warmup",          type=int,   default=400)
    parser.add_argument("--eval-eps",        type=int,   default=50)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
    parser.add_argument("--nav-bias",        type=float, default=0.45)
    parser.add_argument("--breath-period",   type=int,   default=50)
    parser.add_argument("--sweep-amp",       type=float, default=0.30)
    parser.add_argument("--sweep-dur",       type=int,   default=10)
    parser.add_argument("--dry-run",         action="store_true",
                        help="Run 3 warmup + 2 eval episodes per cell to check for errors.")
    args = parser.parse_args()

    result = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        harm_scale=args.harm_scale,
        proximity_harm_scale=args.proximity_scale,
        nav_bias=args.nav_bias,
        breath_period=args.breath_period,
        breath_sweep_amplitude=args.sweep_amp,
        breath_sweep_duration=args.sweep_dur,
        dry_run=args.dry_run,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"]    = ts
    result["claim"]            = CLAIM_IDS[0]
    result["verdict"]          = result["status"]
    result["run_id"]           = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["registered_thresholds"] = {
        "C1_harm_rate_ratio":  THRESH_C1_HARM_RATIO,
        "C2_abs_gap":          THRESH_C2_ABS_GAP,
        "C3_n_harm_min":       THRESH_C3_N_HARM,
        "C4_n_committed_min":  THRESH_C4_N_COMMITTED,
        "pass_threshold":      PASS_THRESHOLD,
    }
    result["seeds"]          = list(args.seeds)
    result["conditions"]     = ["GATE_ON", "GATE_ABLATED"]
    result["dispatch_mode"]  = "discriminative_pair"
    result["evidence_class"] = "discriminative_pair"
    result["claim_ids_tested"] = CLAIM_IDS

    if args.dry_run:
        print("\n[dry-run] Skipping file output.", flush=True)
        sys.exit(0)

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
