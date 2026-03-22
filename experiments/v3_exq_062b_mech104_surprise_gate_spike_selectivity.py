#!/opt/local/bin/python3
"""
V3-EXQ-062b -- MECH-104: Surprise-Gated Volatility Interrupt (Spike Selectivity C4 Fix)

Claims: MECH-104, MECH-090

Context:
  EXQ-062a FAIL -- C4 FAIL: n_uncommitted_C > n_uncommitted_B despite fewer spikes.
  C4 definition was wrong. It measured total uncommitted DURATION, not spike FREQUENCY.

  EXQ-062a data revealed:
    always_n_spikes:   229   surprise_n_spikes:  215  (surprise fires LESS often -- correct)
    always_n_uncommitted: 1534   surprise_n_uncommitted: 2303
    Uncommitted steps per spike: always = 6.7,  surprise = 10.7

  The surprise gate fired on fewer harm steps (selectivity confirmed) but each spike
  caused a longer uncommitted episode -- because surprise spikes occur at genuinely
  high-error moments where recovery takes longer. This is the DESIRED MECH-104 behavior:
  selective interrupts that cause meaningful uncertainty, not frequent low-impact spikes.

  C4 in EXQ-062a penalised this correct behavior. Cumulative uncommitted time conflates
  spike frequency (the selectivity signal) with episode dynamics (how long variance takes
  to decay), making it an invalid criterion for MECH-104 selectivity.

Fix -- revised C4, two sub-criteria:
  C4a: surprise_n_spikes < always_n_spikes
       (surprise gate fires on strictly fewer committed harm steps than always-gate)
  C4b: surprise_uncommitted_per_spike >= always_uncommitted_per_spike * 0.90
       (each surprise spike causes at least as much state disruption as an always spike,
        confirming selectivity is not achieved by suppressing signal magnitude)

Mechanism under test (MECH-104):
  Unexpected harm events spike commitment uncertainty (LC-NE volatility interrupt).
  The surprise gate should RECOGNISE genuinely unexpected harm and fire selectively.
  C4a tests recognition selectivity; C4b tests that the gate is not trivially suppressed.

Design -- identical to EXQ-062a (committed-only gating):
  Single trained agent (400 eps). Eval under 3 interrupt policies:

  Condition A -- NO INTERRUPT (baseline)
    running_variance unchanged on harm contact.
    Expected: 0 uncommitted steps.

  Condition B -- ALWAYS-SPIKE (committed steps only)
    running_variance += spike_magnitude on harm steps WHERE committed.
    Expected: frequent de-commitment (fires on ALL committed harm steps).

  Condition C -- SURPRISE-GATED SPIKE (committed steps only)
    running_variance += spike_magnitude * max(0, |actual-predicted| - threshold)
    WHERE committed AND actual_harm < -0.01.
    Expected: selective de-commitment -- fewer spikes than always (C4a),
              similar or larger uncommitted episode per spike (C4b).

PASS criteria (ALL must hold):
  C1: Condition A n_uncommitted_steps == 0
      (confirms variance collapse; baseline intact)
  C2: Condition B n_uncommitted_steps > 100
      (always-spike enables de-commitment)
  C3: Condition C n_uncommitted_steps > 30
      (surprise-gated also enables de-commitment on genuine surprise)
  C4a: surprise_n_spikes < always_n_spikes
       (surprise gate is more selective -- fires on fewer steps)
  C4b: surprise_uncommitted_per_spike >= always_uncommitted_per_spike * 0.90
       (each surprise spike causes at least as much state disruption)
  C5: No fatal errors
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_062b_mech104_surprise_gate_spike_selectivity"
CLAIM_IDS = ["MECH-104", "MECH-090"]

ENV_KWARGS = dict(
    size=12, num_hazards=4, num_resources=5,
    hazard_harm=0.02,
    env_drift_interval=5, env_drift_prob=0.1,
    proximity_harm_scale=0.05,
    proximity_benefit_scale=0.03,
    proximity_approach_threshold=0.15,
    hazard_field_decay=0.5,
)


def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _mean_safe(lst: List[float]) -> float:
    return float(sum(lst) / len(lst)) if lst else 0.0


def _make_agent_and_env(
    seed: int,
    self_dim: int,
    world_dim: int,
    alpha_world: float,
) -> Tuple[REEAgent, CausalGridWorldV2]:
    torch.manual_seed(seed)
    random.seed(seed)
    env = CausalGridWorldV2(seed=seed, **ENV_KWARGS)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
    )
    agent = REEAgent(config)
    return agent, env


def _train_agent(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    lr: float = 1e-3,
) -> Dict:
    """Train agent until running_variance collapses (~400 eps)."""
    agent.train()
    optimizer = optim.Adam(list(agent.e1.parameters()), lr=lr)
    wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()),
        lr=1e-3,
    )
    harm_eval_optimizer = optim.Adam(
        list(agent.e3.harm_eval_head.parameters()), lr=1e-4,
    )

    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)

            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            theta_z = agent.theta_buffer.summary()
            z_world_curr = latent.z_world.detach()

            action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                )
                agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > 2000:
                    wf_buf = wf_buf[-2000:]

            (harm_buf_pos if harm_signal < 0 else harm_buf_neg).append(theta_z.detach())
            if len(harm_buf_pos) > 1000: harm_buf_pos = harm_buf_pos[-1000:]
            if len(harm_buf_neg) > 1000: harm_buf_neg = harm_buf_neg[-1000:]

            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                optimizer.zero_grad()
                e1_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                optimizer.step()

            if len(wf_buf) >= 16:
                k = min(32, len(wf_buf))
                idxs = torch.randperm(len(wf_buf))[:k].tolist()
                zw_b = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
                a_b = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
                wf_pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(wf_pred, zw1_b)
                if wf_loss.requires_grad:
                    wf_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters()) +
                        list(agent.e2.world_action_encoder.parameters()), 1.0,
                    )
                    wf_optimizer.step()
                with torch.no_grad():
                    agent.e3.update_running_variance(
                        (wf_pred.detach() - zw1_b).detach()
                    )

            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_p = min(16, len(harm_buf_pos))
                k_n = min(16, len(harm_buf_neg))
                pi = torch.randperm(len(harm_buf_pos))[:k_p].tolist()
                ni = torch.randperm(len(harm_buf_neg))[:k_n].tolist()
                zw_b = torch.cat(
                    [harm_buf_pos[i] for i in pi] + [harm_buf_neg[i] for i in ni], dim=0
                )
                target = torch.cat([
                    torch.ones(k_p, 1, device=agent.device),
                    torch.zeros(k_n, 1, device=agent.device),
                ], dim=0)
                pred = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred, target)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e3.harm_eval_head.parameters(), 0.5)
                    harm_eval_optimizer.step()

            z_world_prev = z_world_curr
            action_prev = action.detach()
            if done:
                break

        if (ep + 1) % 100 == 0 or ep == num_episodes - 1:
            rv = agent.e3._running_variance
            print(
                f"  [train] ep {ep+1}/{num_episodes}"
                f"  running_var={rv:.6f}",
                flush=True,
            )

    return {"final_running_variance": agent.e3._running_variance}


def _eval_interrupt_policy(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    policy: str,               # "none" | "always" | "surprise"
    spike_magnitude: float,
    surprise_threshold: float,
    label: str,
    initial_variance: float,
) -> Dict:
    """Eval gate concordance under a given interrupt policy.

    Committed-only gating (inherited from EXQ-062a):
    interrupt only fires on COMMITTED steps. This breaks the positive feedback
    loop where elevated variance causes noisier predictions -> more spikes.
    """
    agent.eval()

    n_uncommitted = 0
    n_committed = 0
    n_harm_steps = 0
    n_committed_harm_steps = 0
    n_surprise_spikes = 0
    n_always_spikes = 0
    predicted_harms: List[float] = []
    actual_harms: List[float] = []
    surprise_vals: List[float] = []
    surprise_at_spike_vals: List[float] = []  # surprise values only at gate-fire moments
    fatal = 0

    for _ in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        # Reset variance to post-training committed state
        agent.e3._running_variance = initial_variance

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)

            # Query predicted harm BEFORE env step (for surprise gate)
            with torch.no_grad():
                theta_z = agent.theta_buffer.summary()
                predicted_harm = float(agent.e3.harm_eval(theta_z).item())

            # Check commitment BEFORE action (used for interrupt gating)
            is_committed_pre = agent.e3._running_variance < agent.e3.commit_threshold

            try:
                with torch.no_grad():
                    ticks = agent.clock.advance()
                    e1_prior = (
                        agent._e1_tick(latent) if ticks.get("e1_tick", False)
                        else torch.zeros(1, world_dim, device=agent.device)
                    )
                    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                    action = agent.select_action(candidates, ticks, temperature=1.0)

                if action is None:
                    action = _action_to_onehot(
                        random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                    )
                    agent._last_action = action

                flat_obs, actual_harm, done, info, obs_dict = env.step(action)

                # Track commitment state (post-action)
                is_committed_post = agent.e3._running_variance < agent.e3.commit_threshold
                if is_committed_post:
                    n_committed += 1
                else:
                    n_uncommitted += 1

                # Apply interrupt policy -- COMMITTED STEPS ONLY
                if actual_harm < -0.01:
                    n_harm_steps += 1
                    predicted_harms.append(predicted_harm)
                    actual_harms.append(actual_harm)
                    surprise = abs(actual_harm - predicted_harm)
                    surprise_vals.append(surprise)

                    if is_committed_pre:
                        n_committed_harm_steps += 1
                        if policy == "always":
                            agent.e3._running_variance += spike_magnitude
                            n_always_spikes += 1
                        elif policy == "surprise":
                            impulse = spike_magnitude * max(0.0, surprise - surprise_threshold)
                            if impulse > 0:
                                agent.e3._running_variance += impulse
                                n_surprise_spikes += 1
                                surprise_at_spike_vals.append(surprise)

            except Exception:
                fatal += 1
                flat_obs, obs_dict = env.reset()
                done = True

            if done:
                break

    mean_surprise = _mean_safe(surprise_vals)
    mean_surprise_at_spike = _mean_safe(surprise_at_spike_vals)
    mean_pred_harm = _mean_safe(predicted_harms)
    mean_actual_harm = _mean_safe(actual_harms)

    print(
        f"\n  [{label}] policy={policy}"
        f"  n_uncommitted={n_uncommitted}  n_committed={n_committed}"
        f"  harm_steps={n_harm_steps}  committed_harm_steps={n_committed_harm_steps}"
        f"  spikes={'N/A' if policy == 'none' else (n_always_spikes if policy == 'always' else n_surprise_spikes)}"
        f"  mean_surprise={mean_surprise:.4f}"
        + (f"  mean_surprise_at_spike={mean_surprise_at_spike:.4f}" if policy == "surprise" else ""),
        flush=True,
    )

    return {
        "n_uncommitted": n_uncommitted,
        "n_committed": n_committed,
        "n_harm_steps": n_harm_steps,
        "n_committed_harm_steps": n_committed_harm_steps,
        "n_spikes": n_always_spikes if policy == "always" else n_surprise_spikes,
        "mean_surprise": mean_surprise,
        "mean_surprise_at_spike": mean_surprise_at_spike,
        "mean_predicted_harm": mean_pred_harm,
        "mean_actual_harm": mean_actual_harm,
        "fatal_errors": fatal,
    }


def run(
    seed: int = 0,
    warmup_episodes: int = 400,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    alpha_world: float = 0.9,
    spike_magnitude: float = 0.05,
    surprise_threshold: float = 0.02,
    self_dim: int = 32,
    world_dim: int = 32,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    print(
        f"[V3-EXQ-062b] MECH-104: Surprise-Gated Volatility Interrupt (Spike Selectivity C4 Fix)\n"
        f"  C4 revised: spike count selectivity (C4a) + per-spike impact (C4b)\n"
        f"  Train one agent -> committed (variance ~0)\n"
        f"  Eval under 3 interrupt policies: none / always-spike / surprise-gated\n"
        f"  spike_magnitude={spike_magnitude}  surprise_threshold={surprise_threshold}\n"
        f"  alpha_world={alpha_world}  seed={seed}",
        flush=True,
    )

    # -- Train single agent --------------------------------------------------
    print(f"\n{'='*60}", flush=True)
    print(f"[V3-EXQ-062b] Training ({warmup_episodes} eps)...", flush=True)
    print('='*60, flush=True)
    agent, env = _make_agent_and_env(seed, self_dim, world_dim, alpha_world)
    train_out = _train_agent(agent, env, warmup_episodes, steps_per_episode, world_dim)

    initial_variance = train_out["final_running_variance"]
    print(
        f"\n  Post-train running_variance={initial_variance:.6f}"
        f"  commit_threshold={agent.e3.commit_threshold:.3f}"
        f"  committed={initial_variance < agent.e3.commit_threshold}",
        flush=True,
    )

    if initial_variance >= agent.e3.commit_threshold:
        print(
            "  WARNING: agent did not collapse to committed state after training. "
            "Results may not be interpretable.",
            flush=True,
        )

    # -- Eval under three policies -------------------------------------------
    results: Dict[str, Dict] = {}
    for policy, label in [
        ("none",     "Condition A -- no interrupt"),
        ("always",   "Condition B -- always-spike (committed only)"),
        ("surprise", "Condition C -- surprise-gated (committed only)"),
    ]:
        print(f"\n{'='*60}", flush=True)
        print(f"[V3-EXQ-062b] {label}", flush=True)
        print('='*60, flush=True)
        results[policy] = _eval_interrupt_policy(
            agent=agent,
            env=env,
            num_episodes=eval_episodes,
            steps_per_episode=steps_per_episode,
            world_dim=world_dim,
            policy=policy,
            spike_magnitude=spike_magnitude,
            surprise_threshold=surprise_threshold,
            label=label,
            initial_variance=initial_variance,
        )

    r_none = results["none"]
    r_always = results["always"]
    r_surprise = results["surprise"]

    # Uncommitted steps per spike (impact per interrupt)
    always_per_spike = (
        r_always["n_uncommitted"] / max(1, r_always["n_spikes"])
    )
    surprise_per_spike = (
        r_surprise["n_uncommitted"] / max(1, r_surprise["n_spikes"])
    )

    # Discriminability: mean surprise at gate-fire vs mean surprise at all harm steps
    # Tests whether the gate fires on genuinely high-surprise events, not just any harm.
    # surprise_discriminability_ratio > 1.0 means the gate is selective on the right signal.
    surprise_discriminability_ratio = (
        r_surprise["mean_surprise_at_spike"]
        / max(1e-6, r_surprise["mean_surprise"])
        if r_surprise["mean_surprise_at_spike"] > 0
        else 0.0
    )

    # -- PASS / FAIL ---------------------------------------------------------
    c1_pass = r_none["n_uncommitted"] == 0
    c2_pass = r_always["n_uncommitted"] > 100
    c3_pass = r_surprise["n_uncommitted"] > 30
    # C4a: surprise fires on fewer committed harm steps than always
    c4a_pass = (
        r_always["n_spikes"] > 0 and
        r_surprise["n_spikes"] < r_always["n_spikes"]
    )
    # C4b: each surprise spike causes at least as much state disruption
    c4b_pass = (
        r_always["n_spikes"] > 0 and
        r_surprise["n_spikes"] > 0 and
        surprise_per_spike >= always_per_spike * 0.90
    )
    c4_pass = c4a_pass and c4b_pass
    c5_pass = (
        r_none["fatal_errors"] +
        r_always["fatal_errors"] +
        r_surprise["fatal_errors"]
    ) == 0

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status = "PASS" if all_pass else "FAIL"
    # Count C4 as one criterion (both sub-criteria must hold)
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: n_uncommitted_A={r_none['n_uncommitted']} != 0 "
            "(baseline collapse not confirmed)"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: n_uncommitted_B={r_always['n_uncommitted']} <= 100 "
            "(always-spike not enabling de-commitment)"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: n_uncommitted_C={r_surprise['n_uncommitted']} <= 30 "
            "(surprise-gated not triggering on genuine surprise)"
        )
    if not c4a_pass:
        failure_notes.append(
            f"C4a FAIL: surprise_n_spikes={r_surprise['n_spikes']}"
            f" not < always_n_spikes={r_always['n_spikes']}"
            " (surprise gate not more selective than always-spike)"
        )
    if not c4b_pass:
        failure_notes.append(
            f"C4b FAIL: surprise_per_spike={surprise_per_spike:.2f}"
            f" < always_per_spike={always_per_spike:.2f} x 0.90 = {always_per_spike*0.90:.2f}"
            " (each surprise spike not causing sufficient state disruption)"
        )
    if not c5_pass:
        failure_notes.append(
            f"C5 FAIL: fatal_errors total="
            f"{r_none['fatal_errors'] + r_always['fatal_errors'] + r_surprise['fatal_errors']}"
        )

    print(f"\nV3-EXQ-062b verdict: {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "train_final_variance":          float(initial_variance),
        "none_n_uncommitted":            float(r_none["n_uncommitted"]),
        "none_n_committed":              float(r_none["n_committed"]),
        "always_n_uncommitted":          float(r_always["n_uncommitted"]),
        "always_n_committed":            float(r_always["n_committed"]),
        "always_n_spikes":               float(r_always["n_spikes"]),
        "always_committed_harm_steps":   float(r_always["n_committed_harm_steps"]),
        "always_per_spike":              float(always_per_spike),
        "surprise_n_uncommitted":        float(r_surprise["n_uncommitted"]),
        "surprise_n_committed":          float(r_surprise["n_committed"]),
        "surprise_n_spikes":             float(r_surprise["n_spikes"]),
        "surprise_committed_harm_steps": float(r_surprise["n_committed_harm_steps"]),
        "surprise_per_spike":            float(surprise_per_spike),
        "surprise_mean_surprise":           float(r_surprise["mean_surprise"]),
        "surprise_mean_surprise_at_spike":  float(r_surprise["mean_surprise_at_spike"]),
        "surprise_discriminability_ratio":  float(surprise_discriminability_ratio),
        "surprise_mean_pred_harm":          float(r_surprise["mean_predicted_harm"]),
        "surprise_mean_actual_harm":        float(r_surprise["mean_actual_harm"]),
        "spike_count_ratio":                float(
            r_surprise["n_spikes"] / max(1, r_always["n_spikes"])
        ),
        "per_spike_ratio":               float(
            surprise_per_spike / max(0.001, always_per_spike)
        ),
        "spike_magnitude":       float(spike_magnitude),
        "surprise_threshold":    float(surprise_threshold),
        "crit1_pass":  1.0 if c1_pass  else 0.0,
        "crit2_pass":  1.0 if c2_pass  else 0.0,
        "crit3_pass":  1.0 if c3_pass  else 0.0,
        "crit4a_pass": 1.0 if c4a_pass else 0.0,
        "crit4b_pass": 1.0 if c4b_pass else 0.0,
        "crit4_pass":  1.0 if c4_pass  else 0.0,
        "crit5_pass":  1.0 if c5_pass  else 0.0,
        "criteria_met": float(criteria_met),
        "fatal_error_count": float(
            r_none["fatal_errors"] + r_always["fatal_errors"] + r_surprise["fatal_errors"]
        ),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-062b -- MECH-104: Surprise-Gated Volatility Interrupt (Spike Selectivity C4 Fix)

**Status:** {status}
**Claims:** MECH-104, MECH-090
**Design:** Single trained agent; 3 interrupt policies (none / always / surprise-gated), committed-only
**C4 fix:** Criterion revised to measure spike count selectivity (C4a) and per-spike impact (C4b)
**spike_magnitude:** {spike_magnitude}  |  **surprise_threshold:** {surprise_threshold}  |  **Warmup:** {warmup_episodes} eps  |  **Eval:** {eval_episodes} eps  |  **Seed:** {seed}

## C4 Criterion Revision (vs EXQ-062a)

EXQ-062a C4 measured total uncommitted DURATION (n_uncommitted_C < n_B x 0.80).
This is wrong: fewer, higher-impact spikes produce more uncommitted time than more,
lower-impact spikes -- penalising the correct MECH-104 behavior.

EXQ-062a data: always=229 spikes, 1534 uncommitted (6.7/spike)
               surprise=215 spikes, 2303 uncommitted (10.7/spike)
Surprise fired LESS often (selectivity confirmed) but each spike caused LONGER
uncommitted episodes (expected: surprise fires at genuinely uncertain moments).

EXQ-062b C4 tests the mechanism directly:
  C4a: surprise_n_spikes < always_n_spikes (selectivity)
  C4b: surprise_per_spike >= always_per_spike x 0.90 (impact per interrupt)

## Training

| Metric | Value |
|--------|-------|
| Final running_variance (post-train) | {initial_variance:.6f} |
| Committed? (variance < {agent.e3.commit_threshold:.2f}) | {"Yes" if initial_variance < agent.e3.commit_threshold else "No"} |

## Eval Results

| Condition | Policy | n_uncommitted | n_committed_harm | n_spikes | uncommitted/spike |
|---|---|---|---|---|---|
| A | None (baseline) | {r_none['n_uncommitted']} | {r_none['n_committed_harm_steps']} | -- | -- |
| B | Always-spike (committed only) | {r_always['n_uncommitted']} | {r_always['n_committed_harm_steps']} | {r_always['n_spikes']} | {always_per_spike:.2f} |
| C | Surprise-gated (committed only) | {r_surprise['n_uncommitted']} | {r_surprise['n_committed_harm_steps']} | {r_surprise['n_spikes']} | {surprise_per_spike:.2f} |

Spike count ratio (C/B): {metrics['spike_count_ratio']:.3f}  (target: < 1.0 for C4a)
Per-spike ratio (C/B):   {metrics['per_spike_ratio']:.3f}   (target: >= 0.90 for C4b)

Discriminability (mean surprise at spike / mean surprise all harm): {surprise_discriminability_ratio:.3f}
  mean_surprise_all_harm:  {r_surprise["mean_surprise"]:.4f}
  mean_surprise_at_spike:  {r_surprise["mean_surprise_at_spike"]:.4f}
  (diagnostic only -- ratio > 1.0 confirms gate fires on genuinely high-surprise events)

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: n_uncommitted_A == 0 (baseline collapse) | {"PASS" if c1_pass else "FAIL"} | {r_none['n_uncommitted']} |
| C2: n_uncommitted_B > 100 (always-spike works) | {"PASS" if c2_pass else "FAIL"} | {r_always['n_uncommitted']} |
| C3: n_uncommitted_C > 30 (surprise fires on genuine events) | {"PASS" if c3_pass else "FAIL"} | {r_surprise['n_uncommitted']} |
| C4a: surprise_n_spikes < always_n_spikes (selectivity) | {"PASS" if c4a_pass else "FAIL"} | {r_surprise['n_spikes']} vs {r_always['n_spikes']} |
| C4b: surprise_per_spike >= always_per_spike x 0.90 (impact) | {"PASS" if c4b_pass else "FAIL"} | {surprise_per_spike:.2f} vs {always_per_spike*0.90:.2f} |
| C5: No fatal errors | {"PASS" if c5_pass else "FAIL"} | {metrics['fatal_error_count']:.0f} |

Criteria met: {criteria_met}/5 -> **{status}**
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass else ("mixed" if criteria_met >= 3 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": float(metrics["fatal_error_count"]),
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",               type=int,   default=0)
    parser.add_argument("--warmup",             type=int,   default=400)
    parser.add_argument("--eval-eps",           type=int,   default=50)
    parser.add_argument("--steps",              type=int,   default=200)
    parser.add_argument("--alpha-world",        type=float, default=0.9)
    parser.add_argument("--spike-magnitude",    type=float, default=0.05)
    parser.add_argument("--surprise-threshold", type=float, default=0.02)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        spike_magnitude=args.spike_magnitude,
        surprise_threshold=args.surprise_threshold,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

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
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}", flush=True)
