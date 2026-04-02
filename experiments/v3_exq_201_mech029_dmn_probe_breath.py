#!/opt/local/bin/python3
"""
V3-EXQ-201 -- MECH-029: DMN/Reflective Mode Moral Evaluation (BreathOscillator)

Claims: MECH-029
Supersedes: V3-EXQ-080

Motivation:
  MECH-029: when the agent is in uncommitted / reflective state, it engages in
  internal simulation and reflective processing (the REE "default mode").
  EXQ-065 was design-inconclusive (always committed). EXQ-080 introduced the
  BreathOscillator to force periodic uncommitted windows but used CausalGridWorld
  (not V2) and only 2 eval episodes with no multi-seed aggregation.

  This experiment supersedes EXQ-080 with:
  - CausalGridWorldV2 (proxy fields for harm stream)
  - Stronger breath parameters (sweep_amplitude=0.30, sweep_duration=10)
  - 500 training + 50 eval episodes per seed
  - 2 seeds aggregated
  - nav_bias=0.45 to ensure harm exposure diversity
  - SD-008 alpha_world=0.9, SD-009 event classifier enabled

Design:
  For each seed:
    1. Train agent for 500 episodes (breath oscillator active during training).
    2. Eval for 50 episodes. At each eval step, record:
       - Whether the step is committed or uncommitted
       - z_world vector
       - E3 harm_eval output (harm prediction scalar)
    3. Compute per-seed metrics:
       - z_world variance for committed vs uncommitted steps
       - harm_eval std for committed vs uncommitted steps
       - harm_eval std overall (collapse check)
       - step counts per mode

Pre-registered PASS criteria (need 4/5 across seed aggregate):
  C1: uncommitted_step_count >= 50 per seed (enough reflective-mode samples)
  C2: harm_pred_std > 0.01 (E3 harm head not collapsed)
  C3: z_world_var_uncommitted > z_world_var_committed * 1.05
      (reflective mode has higher internal variability)
  C4: harm_eval_std_uncommitted > harm_eval_std_committed
      (E3 explores more harm scenarios when uncommitted)
  C5: no fatal errors
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
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_201_mech029_dmn_probe_breath"
CLAIM_IDS = ["MECH-029"]

# Environment
ENV_SIZE = 6
N_HAZARDS = 4
NAV_BIAS = 0.45

# BreathOscillator parameters
BREATH_PERIOD = 50
SWEEP_AMPLITUDE = 0.30
SWEEP_DURATION = 10
E3_RATE = 5  # e3_steps_per_tick=5 so e3_tick fires during sweep window

# Training / eval
TRAIN_EPISODES = 500
EVAL_EPISODES = 50
STEPS_PER_EPISODE = 200
SEEDS = [42, 123]

# Config
WORLD_DIM = 32
SELF_DIM = 32
ALPHA_WORLD = 0.9
LR = 1e-3


def _action_to_onehot(action_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, action_idx] = 1.0
    return v


def _mean_safe(lst: List[float]) -> float:
    return float(sum(lst) / len(lst)) if lst else 0.0


def _std_safe(lst: List[float]) -> float:
    if len(lst) < 2:
        return 0.0
    t = torch.tensor(lst)
    return float(t.std().item())


def _tensor_variance(tensors: List[torch.Tensor]) -> float:
    """Mean per-dimension variance across a list of [1, d] tensors."""
    if len(tensors) < 2:
        return 0.0
    stacked = torch.cat(tensors, dim=0)  # [n, d]
    return float(stacked.var(dim=0).mean().item())


def _hazard_approach_action(env: CausalGridWorld, n_actions: int) -> int:
    """Return action index that moves agent toward nearest hazard."""
    ax, ay = env.agent_x, env.agent_y
    best_dist = float("inf")
    hx, hy = ax, ay
    for i in range(env.size):
        for j in range(env.size):
            if env.grid[i, j] == env.ENTITY_TYPES.get("hazard", 2):
                d = abs(i - ax) + abs(j - ay)
                if d < best_dist:
                    best_dist = d
                    hx, hy = i, j
    # actions: 0=up, 1=down, 2=left, 3=right
    dx, dy = hx - ax, hy - ay
    if abs(dx) >= abs(dy):
        return 1 if dx > 0 else 0
    else:
        return 3 if dy > 0 else 2


def _make_env(seed: int) -> CausalGridWorld:
    return CausalGridWorld(
        size=ENV_SIZE,
        num_hazards=N_HAZARDS,
        num_resources=3,
        use_proxy_fields=True,
        seed=seed,
        hazard_harm=0.02,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.03,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )


def _train(
    agent: REEAgent,
    env: CausalGridWorld,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    nav_bias: float,
    seed: int,
) -> Dict:
    """Training loop: E1 + E2 world_forward + E3 harm_eval."""
    agent.train()

    e1_opt = optim.Adam(agent.e1.parameters(), lr=LR)
    wf_opt = optim.Adam(
        list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters()),
        lr=LR,
    )
    harm_eval_opt = optim.Adam(
        list(agent.e3.harm_eval_head.parameters()), lr=1e-4,
    )

    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    total_harm = 0

    for ep in range(num_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        z_self_prev: Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            theta_z = agent.theta_buffer.summary()
            z_world_curr = latent.z_world.detach()

            # Nav-biased action selection
            if random.random() < nav_bias:
                a_idx = _hazard_approach_action(env, env.action_dim)
                action = _action_to_onehot(a_idx, env.action_dim, agent.device)
                agent._last_action = action
            else:
                action = agent.select_action(candidates, ticks, temperature=1.0)
                if action is None:
                    action = _action_to_onehot(
                        random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                    )
                    agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)

            # E2 world_forward training
            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > 2000:
                    wf_buf = wf_buf[-2000:]

                if len(wf_buf) >= 16:
                    k = min(32, len(wf_buf))
                    idxs = torch.randperm(len(wf_buf))[:k].tolist()
                    zw_b = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
                    a_b = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
                    zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
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

            # Harm buffer for E3 harm_eval training
            if harm_signal < 0:
                total_harm += 1
                harm_buf_pos.append(theta_z.detach())
                if len(harm_buf_pos) > 1000:
                    harm_buf_pos = harm_buf_pos[-1000:]
            else:
                harm_buf_neg.append(theta_z.detach())
                if len(harm_buf_neg) > 1000:
                    harm_buf_neg = harm_buf_neg[-1000:]

            # E1 prediction loss
            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                e1_opt.zero_grad()
                e1_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                e1_opt.step()

            # E3 harm_eval training
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
                    harm_eval_opt.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e3.harm_eval_head.parameters(), 0.5)
                    harm_eval_opt.step()

            z_world_prev = z_world_curr
            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            if done:
                break

        if (ep + 1) % 100 == 0 or ep == num_episodes - 1:
            rv = agent.e3._running_variance
            print(
                f"  [train/seed={seed}] ep {ep+1}/{num_episodes}"
                f"  harm={total_harm}  running_var={rv:.6f}",
                flush=True,
            )

    return {
        "total_harm": total_harm,
        "final_running_variance": agent.e3._running_variance,
    }


def _eval_dmn(
    agent: REEAgent,
    env: CausalGridWorld,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    seed: int,
) -> Dict:
    """
    Eval: collect z_world and E3 harm_eval for committed vs uncommitted steps.

    Commitment state is determined by the agent's select_action path --
    the BreathOscillator creates periodic uncommitted windows via sweep_threshold_reduction
    applied inside E3.select(). We read commitment from the agent after each step.
    """
    agent.eval()
    agent.beta_gate.reset()

    z_worlds_committed: List[torch.Tensor] = []
    z_worlds_uncommitted: List[torch.Tensor] = []
    harm_preds_committed: List[float] = []
    harm_preds_uncommitted: List[float] = []
    all_harm_preds: List[float] = []
    fatal = 0

    for ep_idx in range(num_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            try:
                with torch.no_grad():
                    latent = agent.sense(obs_body, obs_world)
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

                    # Determine commitment from running_variance vs effective threshold
                    # During sweep: effective_threshold = base * (1 - sweep_amplitude)
                    sweep_active = agent.clock.sweep_active
                    base_threshold = agent.e3.commit_threshold
                    if sweep_active:
                        eff_threshold = base_threshold * (1.0 - SWEEP_AMPLITUDE)
                    else:
                        eff_threshold = base_threshold
                    rv = agent.e3._running_variance
                    is_committed = rv < eff_threshold

                    z_w = latent.z_world.detach().cpu()
                    harm_pred = float(agent.e3.harm_eval(latent.z_world).item())
                    all_harm_preds.append(harm_pred)

                    if is_committed:
                        z_worlds_committed.append(z_w)
                        harm_preds_committed.append(harm_pred)
                    else:
                        z_worlds_uncommitted.append(z_w)
                        harm_preds_uncommitted.append(harm_pred)

            except Exception:
                fatal += 1
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                )
                agent._last_action = action

            _, _, done, _, obs_dict = env.step(action)
            if done:
                break

    committed_z_world_var = _tensor_variance(z_worlds_committed)
    uncommitted_z_world_var = _tensor_variance(z_worlds_uncommitted)
    harm_eval_std_committed = _std_safe(harm_preds_committed)
    harm_eval_std_uncommitted = _std_safe(harm_preds_uncommitted)
    harm_pred_std = _std_safe(all_harm_preds)

    print(
        f"  [eval/seed={seed}]"
        f"  committed_steps={len(z_worlds_committed)}"
        f"  uncommitted_steps={len(z_worlds_uncommitted)}"
        f"\n    committed_z_world_var={committed_z_world_var:.6f}"
        f"  uncommitted_z_world_var={uncommitted_z_world_var:.6f}"
        f"\n    harm_eval_std_committed={harm_eval_std_committed:.6f}"
        f"  harm_eval_std_uncommitted={harm_eval_std_uncommitted:.6f}"
        f"\n    harm_pred_std={harm_pred_std:.4f}"
        f"  fatal={fatal}",
        flush=True,
    )

    return {
        "committed_step_count": len(z_worlds_committed),
        "uncommitted_step_count": len(z_worlds_uncommitted),
        "committed_z_world_var": committed_z_world_var,
        "uncommitted_z_world_var": uncommitted_z_world_var,
        "harm_eval_std_committed": harm_eval_std_committed,
        "harm_eval_std_uncommitted": harm_eval_std_uncommitted,
        "harm_pred_std": harm_pred_std,
        "fatal_errors": fatal,
    }


def run(
    seeds: List[int] = None,
    train_episodes: int = TRAIN_EPISODES,
    eval_episodes: int = EVAL_EPISODES,
    steps_per_episode: int = STEPS_PER_EPISODE,
    alpha_world: float = ALPHA_WORLD,
    nav_bias: float = NAV_BIAS,
    smoke: bool = False,
    smoke_out_dir: str = "",
    **kwargs,
) -> dict:
    if seeds is None:
        seeds = list(SEEDS)

    if smoke:
        train_episodes = 5
        eval_episodes = 2
        steps_per_episode = 100
        seeds = [42]

    world_dim = WORLD_DIM
    self_dim = SELF_DIM

    print(
        f"\n[V3-EXQ-201] MECH-029: DMN/Reflective Mode Probe (BreathOscillator)\n"
        f"  seeds={seeds}  train={train_episodes}  eval={eval_episodes}"
        f"  steps={steps_per_episode}\n"
        f"  alpha_world={alpha_world}  nav_bias={nav_bias}\n"
        f"  breath_period={BREATH_PERIOD}  sweep_amp={SWEEP_AMPLITUDE}"
        f"  sweep_dur={SWEEP_DURATION}  e3_rate={E3_RATE}",
        flush=True,
    )

    # Use a reference env to get obs dims
    ref_env = _make_env(0)
    body_obs_dim = ref_env.body_obs_dim
    world_obs_dim = ref_env.world_obs_dim
    action_dim = ref_env.action_dim

    all_seed_results: List[Dict] = []

    for seed in seeds:
        print(f"\n{'='*60}", flush=True)
        print(f"[V3-EXQ-201] Seed {seed}", flush=True)
        print(f"{'='*60}", flush=True)

        torch.manual_seed(seed)
        random.seed(seed)

        env = _make_env(seed)
        config = REEConfig.from_dims(
            body_obs_dim=body_obs_dim,
            world_obs_dim=world_obs_dim,
            action_dim=action_dim,
            self_dim=self_dim,
            world_dim=world_dim,
            alpha_world=alpha_world,
            alpha_self=0.3,
            reafference_action_dim=action_dim,
            use_event_classifier=True,
        )
        # BreathOscillator config
        config.heartbeat.breath_period = BREATH_PERIOD
        config.heartbeat.breath_sweep_amplitude = SWEEP_AMPLITUDE
        config.heartbeat.breath_sweep_duration = SWEEP_DURATION
        config.heartbeat.e3_steps_per_tick = E3_RATE

        torch.manual_seed(seed)
        random.seed(seed)
        agent = REEAgent(config)

        base_threshold = agent.e3.commit_threshold
        eff_sweep_threshold = base_threshold * (1.0 - SWEEP_AMPLITUDE)
        print(
            f"  base_commit_threshold={base_threshold:.3f}"
            f"  effective_sweep_threshold={eff_sweep_threshold:.3f}"
            f"  initial_running_var={agent.e3._running_variance:.3f}",
            flush=True,
        )

        # Train
        print(f"\n[V3-EXQ-201] Training seed={seed} ({train_episodes} eps)...", flush=True)
        train_out = _train(
            agent, env, train_episodes, steps_per_episode, world_dim, nav_bias, seed,
        )

        print(
            f"  [post-train/seed={seed}]"
            f" running_var={agent.e3._running_variance:.4f}"
            f" harm_count={train_out['total_harm']}",
            flush=True,
        )

        # Eval
        print(f"\n[V3-EXQ-201] Eval seed={seed} ({eval_episodes} eps)...", flush=True)
        eval_out = _eval_dmn(agent, env, eval_episodes, steps_per_episode, world_dim, seed)

        all_seed_results.append(eval_out)

    # ---- Aggregate across seeds ----
    agg_committed_count = min(r["committed_step_count"] for r in all_seed_results)
    agg_uncommitted_count = min(r["uncommitted_step_count"] for r in all_seed_results)
    agg_harm_pred_std = _mean_safe([r["harm_pred_std"] for r in all_seed_results])
    agg_committed_z_var = _mean_safe([r["committed_z_world_var"] for r in all_seed_results])
    agg_uncommitted_z_var = _mean_safe([r["uncommitted_z_world_var"] for r in all_seed_results])
    agg_harm_std_committed = _mean_safe([r["harm_eval_std_committed"] for r in all_seed_results])
    agg_harm_std_uncommitted = _mean_safe([r["harm_eval_std_uncommitted"] for r in all_seed_results])
    agg_fatal = sum(r["fatal_errors"] for r in all_seed_results)

    # ---- PASS criteria ----
    c1_pass = agg_uncommitted_count >= 50
    c2_pass = agg_harm_pred_std > 0.01
    c3_pass = agg_uncommitted_z_var > agg_committed_z_var * 1.05
    c4_pass = agg_harm_std_uncommitted > agg_harm_std_committed
    c5_pass = agg_fatal == 0

    criteria_results = [c1_pass, c2_pass, c3_pass, c4_pass, c5_pass]
    criteria_met = sum(criteria_results)
    all_pass = criteria_met >= 4
    status = "PASS" if all_pass else "FAIL"

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: min(uncommitted_step_count)={agg_uncommitted_count} < 50"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: mean(harm_pred_std)={agg_harm_pred_std:.6f} <= 0.01"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: uncommitted_z_world_var={agg_uncommitted_z_var:.6f}"
            f" <= committed*1.05={agg_committed_z_var * 1.05:.6f}"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: harm_eval_std_uncommitted={agg_harm_std_uncommitted:.6f}"
            f" <= harm_eval_std_committed={agg_harm_std_committed:.6f}"
        )
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: fatal_errors={agg_fatal}")

    print(f"\n{'='*60}", flush=True)
    print(f"[V3-EXQ-201] Aggregate Results ({len(seeds)} seeds)", flush=True)
    print(f"{'='*60}", flush=True)
    print(
        f"  min(uncommitted_steps)={agg_uncommitted_count}"
        f"  min(committed_steps)={agg_committed_count}",
        flush=True,
    )
    print(
        f"  mean(committed_z_world_var)={agg_committed_z_var:.6f}"
        f"  mean(uncommitted_z_world_var)={agg_uncommitted_z_var:.6f}"
        f"  ratio={agg_uncommitted_z_var / max(agg_committed_z_var, 1e-8):.4f}",
        flush=True,
    )
    print(
        f"  mean(harm_eval_std_committed)={agg_harm_std_committed:.6f}"
        f"  mean(harm_eval_std_uncommitted)={agg_harm_std_uncommitted:.6f}",
        flush=True,
    )
    print(
        f"  mean(harm_pred_std)={agg_harm_pred_std:.6f}"
        f"  total_fatal={agg_fatal}",
        flush=True,
    )
    print(f"\n  Verdict: {status}  ({criteria_met}/5, need 4/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    # ---- Build per-seed table rows ----
    seed_rows = ""
    for i, r in enumerate(all_seed_results):
        seed_rows += (
            f"| {seeds[i]} | {r['committed_step_count']} | {r['uncommitted_step_count']}"
            f" | {r['committed_z_world_var']:.6f} | {r['uncommitted_z_world_var']:.6f}"
            f" | {r['harm_eval_std_committed']:.6f} | {r['harm_eval_std_uncommitted']:.6f}"
            f" | {r['harm_pred_std']:.4f} | {r['fatal_errors']} |\n"
        )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    metrics = {
        "agg_committed_step_count":        float(agg_committed_count),
        "agg_uncommitted_step_count":      float(agg_uncommitted_count),
        "agg_committed_z_world_var":       float(agg_committed_z_var),
        "agg_uncommitted_z_world_var":     float(agg_uncommitted_z_var),
        "z_world_var_ratio":               float(agg_uncommitted_z_var / max(agg_committed_z_var, 1e-8)),
        "agg_harm_eval_std_committed":     float(agg_harm_std_committed),
        "agg_harm_eval_std_uncommitted":   float(agg_harm_std_uncommitted),
        "agg_harm_pred_std":               float(agg_harm_pred_std),
        "agg_fatal_errors":                float(agg_fatal),
        "breath_period":                   float(BREATH_PERIOD),
        "sweep_amplitude":                 float(SWEEP_AMPLITUDE),
        "sweep_duration":                  float(SWEEP_DURATION),
        "e3_steps_per_tick":               float(E3_RATE),
        "nav_bias":                        float(nav_bias),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "crit5_pass": 1.0 if c5_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    summary_markdown = f"""# V3-EXQ-201 -- MECH-029: DMN/Reflective Mode Probe (BreathOscillator)

**Status:** {status}
**Claims:** MECH-029
**Supersedes:** V3-EXQ-080
**Seeds:** {seeds}  |  **Train:** {train_episodes} eps  |  **Eval:** {eval_episodes} eps  |  **Steps/ep:** {steps_per_episode}
**Config:** alpha_world={alpha_world}, nav_bias={nav_bias}, use_event_classifier=True
**BreathOscillator:** period={BREATH_PERIOD}, sweep_amp={SWEEP_AMPLITUDE}, sweep_dur={SWEEP_DURATION}, e3_rate={E3_RATE}
**Env:** CausalGridWorldV2 size={ENV_SIZE}, n_hazards={N_HAZARDS}, use_proxy_fields=True

## Design Rationale

The BreathOscillator (MECH-108) creates periodic uncommitted windows by reducing the
effective commit_threshold during sweep phases. During these windows, E3 is more likely
to be uncommitted (running_variance > effective_threshold). MECH-029 predicts that
uncommitted/reflective mode shows higher z_world variability and more diverse E3 harm
evaluations than committed execution mode.

## Per-Seed Results

| Seed | Committed Steps | Uncommitted Steps | z_world_var_C | z_world_var_U | harm_std_C | harm_std_U | harm_pred_std | Fatal |
|------|----------------|-------------------|---------------|---------------|------------|------------|---------------|-------|
{seed_rows}

## Aggregate (mean across seeds)

| Metric | Committed | Uncommitted |
|--------|-----------|-------------|
| z_world_var | {agg_committed_z_var:.6f} | {agg_uncommitted_z_var:.6f} |
| harm_eval_std | {agg_harm_std_committed:.6f} | {agg_harm_std_uncommitted:.6f} |

**z_world_var_ratio** (uncommitted/committed) = {agg_uncommitted_z_var / max(agg_committed_z_var, 1e-8):.4f}

## PASS Criteria (need 4/5)

| Criterion | Threshold | Result | Value |
|-----------|-----------|--------|-------|
| C1: min(uncommitted_steps) >= 50 | 50 | {"PASS" if c1_pass else "FAIL"} | {agg_uncommitted_count} |
| C2: mean(harm_pred_std) > 0.01 | 0.01 | {"PASS" if c2_pass else "FAIL"} | {agg_harm_pred_std:.6f} |
| C3: uncommitted_z_var > committed*1.05 | 1.05x | {"PASS" if c3_pass else "FAIL"} | {agg_uncommitted_z_var:.6f} vs {agg_committed_z_var * 1.05:.6f} |
| C4: harm_std_uncommitted > harm_std_committed | > | {"PASS" if c4_pass else "FAIL"} | {agg_harm_std_uncommitted:.6f} vs {agg_harm_std_committed:.6f} |
| C5: no fatal errors | 0 | {"PASS" if c5_pass else "FAIL"} | {agg_fatal} |

Criteria met: {criteria_met}/5 -> **{status}**
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "supersedes": "V3-EXQ-080",
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if criteria_met >= 3 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": float(agg_fatal),
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="42,123",
                        help="Comma-separated seed list")
    parser.add_argument("--train-eps", type=int, default=TRAIN_EPISODES)
    parser.add_argument("--eval-eps", type=int, default=EVAL_EPISODES)
    parser.add_argument("--steps", type=int, default=STEPS_PER_EPISODE)
    parser.add_argument("--alpha-world", type=float, default=ALPHA_WORLD)
    parser.add_argument("--nav-bias", type=float, default=NAV_BIAS)
    parser.add_argument(
        "--smoke", action="store_true",
        help="Quick smoke test (5 train eps, 2 eval eps, 1 seed, no REE_assembly output)",
    )
    parser.add_argument("--smoke-out-dir", type=str, default="")
    args = parser.parse_args()

    seed_list = [int(s.strip()) for s in args.seeds.split(",")]

    result = run(
        seeds=seed_list,
        train_episodes=args.train_eps,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        nav_bias=args.nav_bias,
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
        smoke_dir_str = args.smoke_out_dir or "/tmp/exq201_smoke"
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
