"""
V3-EXQ-025 — E2_world Action-Conditional Divergence Diagnostic

Claims: SD-003, SD-008.

Motivation (2026-03-18):
  EXQ-023 showed E2_world MSE = 4–7× worse than identity baseline, meaning
  E2_world has learned nothing useful. But does it learn ANYTHING after adding
  the 1-step direct loss (EXQ-024 fix)? This experiment isolates E2_world quality
  WITHOUT going through the net_eval bottleneck.

  The SD-003 pipeline:
      causal_sig = net_eval(E2.world_forward(z, a_actual)) - net_eval(E2.world_forward(z, a_cf))

  If E2_world is not learning action-conditional world dynamics, the causal_sig is
  driven entirely by net_eval variability (which is ~0 when harm_signal ≈ 0).

  This experiment directly measures E2_world's action-conditional differentiation:
      div(z, a1, a2) = ||E2.world_forward(z, a1) - E2.world_forward(z, a2)||

  If E2_world has learned any useful structure:
    - At near-hazard positions: div should be LARGE (action choice matters — moving
      toward vs away from hazard produces very different world states)
    - At safe positions: div should be SMALLER (actions are less consequential)

  This test bypasses net_eval entirely and measures E2_world's raw discriminability.

  Three conditions are tested:
    A. No 1-step loss (baseline: rollout-only, replicates EXQ-023)
    B. With 1-step loss (EXQ-024 fix)
    C. With 1-step loss + reafference correction (lstsq applied before world_forward)

PASS criteria (ALL must hold):
  C1: e2w_improvement_ratio(1step) > 2.0 in condition B
      (E2_world with 1-step loss is 2× better than identity prediction)
  C2: action_div_near(B) > action_div_safe(B) with margin > 0.002
      (E2_world distinguishes near-hazard action consequences)
  C3: action_div_near(B) > action_div_near(A) * 1.2
      (1-step loss improves E2_world discriminability by >= 20%)
  C4: n_near_probes >= 50 and n_safe_probes >= 50
  C5: No fatal errors
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Dict, List, Tuple, Optional
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_025_e2w_action_divergence"
CLAIM_IDS = ["SD-003", "SD-008"]

E2_ROLLOUT_STEPS = 5
RECON_WEIGHT = 1.0


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _make_world_decoder(world_dim: int, world_obs_dim: int, hidden_dim: int = 64):
    return nn.Sequential(
        nn.Linear(world_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, world_obs_dim),
    )


def _compute_e2w_rollout_loss(agent, traj_buffer, rollout_steps, batch_size=8):
    if len(traj_buffer) < 2:
        return next(agent.e1.parameters()).sum() * 0.0
    n = min(batch_size, len(traj_buffer))
    idxs = torch.randperm(len(traj_buffer))[:n].tolist()
    total = next(agent.e1.parameters()).sum() * 0.0
    count = 0
    for idx in idxs:
        seg = traj_buffer[idx]
        if len(seg) < rollout_steps + 1:
            continue
        z = seg[0][0]
        z_target = seg[rollout_steps][0]
        for k in range(rollout_steps):
            z = agent.e2.world_forward(z, seg[k][1])
        total = total + F.mse_loss(z, z_target)
        count += 1
    return total / count if count > 0 else total


def _train_agent(
    agent: REEAgent,
    env: CausalGridWorld,
    world_decoder: nn.Module,
    optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
    use_1step_loss: bool,
    e2w_1step_weight: float = 1.0,
    e2w_rollout_weight: float = 0.5,
) -> Dict:
    """Train agent with or without 1-step E2_world loss."""
    agent.train()
    world_decoder.train()

    traj_buffer: List = []
    MAX_TRAJ_BUFFER = 200

    reaf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_REAF_DATA = 5000

    e2w_1step_mses: List[float] = []
    identity_mses:  List[float] = []

    total_harm = 0
    n_empty_steps = 0
    z_world_prev = None
    a_prev = None

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        episode_traj: List = []

        z_world_prev = None
        z_self_prev  = None
        a_prev       = None
        prev_ttype   = None

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_curr = latent.z_world.detach()
            z_self_curr  = latent.z_self.detach()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            episode_traj.append((latent.z_world.detach(), action.detach()))

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            if harm_signal < 0:
                total_harm += 1

            # Collect reafference data (empty steps)
            if (z_world_prev is not None and z_self_prev is not None and
                    a_prev is not None and ttype == "none"):
                dz_world = z_world_curr - z_world_prev
                reaf_data.append((z_self_prev.cpu(), a_prev.cpu(), dz_world.cpu()))
                n_empty_steps += 1
                if len(reaf_data) > MAX_REAF_DATA:
                    reaf_data = reaf_data[-MAX_REAF_DATA:]

            # Track E2_world 1-step MSE (whether or not we use it for training)
            if z_world_prev is not None and a_prev is not None:
                with torch.no_grad():
                    z_pred = agent.e2.world_forward(z_world_prev, a_prev)
                    e2w_1step_mses.append(float(F.mse_loss(z_pred, z_world_curr).item()))
                    identity_mses.append(float(F.mse_loss(z_world_prev, z_world_curr).item()))

            # Loss computation
            e1_loss      = agent.compute_prediction_loss()
            e2_self_loss = agent.compute_e2_loss()
            e2w_rollout  = _compute_e2w_rollout_loss(agent, traj_buffer, E2_ROLLOUT_STEPS)

            obs_w = obs_world.unsqueeze(0) if obs_world.dim() == 1 else obs_world
            z_w = agent.latent_stack.split_encoder.world_encoder(obs_w)
            recon_loss = F.mse_loss(world_decoder(z_w), obs_w)

            total_loss = (
                e1_loss
                + e2_self_loss
                + e2w_rollout_weight * e2w_rollout
                + RECON_WEIGHT * recon_loss
            )

            # 1-step loss (optional, condition B/C vs A)
            if use_1step_loss and z_world_prev is not None and a_prev is not None:
                z_pred_1s = agent.e2.world_forward(z_world_prev, a_prev)
                total_loss = total_loss + e2w_1step_weight * F.mse_loss(
                    z_pred_1s, z_world_curr.detach()
                )

            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(world_decoder.parameters(), 1.0)
                optimizer.step()

            z_world_prev = z_world_curr
            z_self_prev  = z_self_curr
            a_prev       = action.detach()
            prev_ttype   = ttype

            if done:
                break

        for start in range(0, len(episode_traj) - E2_ROLLOUT_STEPS):
            traj_buffer.append(episode_traj[start:start + E2_ROLLOUT_STEPS + 1])
        if len(traj_buffer) > MAX_TRAJ_BUFFER:
            traj_buffer = traj_buffer[-MAX_TRAJ_BUFFER:]

    mean_e2w = float(sum(e2w_1step_mses) / max(1, len(e2w_1step_mses)))
    mean_id  = float(sum(identity_mses)  / max(1, len(identity_mses)))
    improvement_ratio = mean_id / max(mean_e2w, 1e-10)

    label = "1step" if use_1step_loss else "rollout_only"
    print(f"  [{label}] E2_world 1-step MSE={mean_e2w:.5f}  "
          f"identity={mean_id:.5f}  ratio={improvement_ratio:.2f}x  harm={total_harm}",
          flush=True)

    return {
        "total_harm":           total_harm,
        "n_empty_steps":        n_empty_steps,
        "reaf_data":            reaf_data,
        "mean_e2w_1step_mse":   mean_e2w,
        "mean_identity_mse":    mean_id,
        "improvement_ratio":    improvement_ratio,
    }


def _fit_lstsq_predictor(reaf_data, self_dim, action_dim, world_dim):
    if len(reaf_data) < 20:
        return None
    n = len(reaf_data)
    n_train = int(n * 0.8)
    z_self_all = torch.cat([d[0] for d in reaf_data], dim=0)
    a_all      = torch.cat([d[1] for d in reaf_data], dim=0)
    dz_all     = torch.cat([d[2] for d in reaf_data], dim=0)
    ones_all   = torch.ones(n, 1)
    X_all = torch.cat([z_self_all, a_all, ones_all], dim=-1)

    with torch.no_grad():
        result = torch.linalg.lstsq(X_all[:n_train], dz_all[:n_train], driver="gelsd")
        return result.solution


def _measure_action_divergence(
    agent: REEAgent,
    probe_env: CausalGridWorld,
    num_resets: int,
    W: Optional[torch.Tensor],
    min_dist_safe: int = 2,
    n_action_pairs_per_cell: int = 3,
) -> Dict:
    """
    Measure E2_world action-conditional divergence at near-hazard vs safe positions.

    For each position, sample n_action_pairs_per_cell pairs of actions and compute:
        div = ||E2.world_forward(z_world, a1) - E2.world_forward(z_world, a2)||

    This directly measures whether E2_world produces different predictions for
    different actions — independent of net_eval.
    """
    agent.eval()
    device = agent.device
    wall_type   = probe_env.ENTITY_TYPES["wall"]
    hazard_type = probe_env.ENTITY_TYPES["hazard"]
    use_lstsq   = (W is not None)

    near_divs: List[float] = []
    safe_divs: List[float] = []
    fatal_errors = 0

    def _divergence_at(px, py, z_self_for_reaf=None):
        probe_env.agent_x = px
        probe_env.agent_y = py
        obs_dict = probe_env._get_observation_dict()
        with torch.no_grad():
            latent  = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
            z_world = latent.z_world  # [1, world_dim]
            z_self  = latent.z_self

        divs = []
        for _ in range(n_action_pairs_per_cell):
            a1_idx = random.randint(0, probe_env.action_dim - 1)
            a2_idx = random.choice(
                [a for a in range(probe_env.action_dim) if a != a1_idx]
            )
            a1 = _action_to_onehot(a1_idx, probe_env.action_dim, device)
            a2 = _action_to_onehot(a2_idx, probe_env.action_dim, device)

            with torch.no_grad():
                if use_lstsq:
                    ones = torch.ones(1, 1, device=device)
                    feat1 = torch.cat([z_self, a1, ones], dim=-1)
                    feat2 = torch.cat([z_self, a2, ones], dim=-1)
                    zw1 = z_world - feat1 @ W.to(device)
                    zw2 = z_world - feat2 @ W.to(device)
                else:
                    zw1 = z_world
                    zw2 = z_world

                pred1 = agent.e2.world_forward(zw1, a1)
                pred2 = agent.e2.world_forward(zw2, a2)
                div = float(torch.norm(pred1 - pred2).item())
                divs.append(div)

        return float(sum(divs) / max(1, len(divs)))

    try:
        for _ in range(num_resets):
            probe_env.reset()

            # Near-hazard positions: cell adjacent to hazard
            for hx, hy in probe_env.hazards:
                for action_idx, (dx, dy) in probe_env.ACTIONS.items():
                    if action_idx == 4:
                        continue
                    ax, ay = hx - dx, hy - dy
                    if 0 <= ax < probe_env.size and 0 <= ay < probe_env.size:
                        cell = int(probe_env.grid[ax, ay])
                        if cell not in (wall_type, hazard_type):
                            near_divs.append(_divergence_at(ax, ay))

            # Safe positions: far from all hazards
            for px in range(probe_env.size):
                for py in range(probe_env.size):
                    if int(probe_env.grid[px, py]) in (wall_type, hazard_type):
                        continue
                    min_dist = min(
                        abs(px - hx) + abs(py - hy) for hx, hy in probe_env.hazards
                    )
                    if min_dist > min_dist_safe:
                        safe_divs.append(_divergence_at(px, py))

    except Exception:
        import traceback
        fatal_errors += 1
        print(f"  FATAL divergence probe: {traceback.format_exc()}", flush=True)

    mean_near = float(sum(near_divs) / max(1, len(near_divs)))
    mean_safe = float(sum(safe_divs) / max(1, len(safe_divs)))
    div_gap   = mean_near - mean_safe

    print(f"  action_div  n_near={len(near_divs)} n_safe={len(safe_divs)}  "
          f"near={mean_near:.4f}  safe={mean_safe:.4f}  gap={div_gap:.4f}  "
          f"(min_dist_safe={min_dist_safe}  lstsq={use_lstsq})", flush=True)

    return {
        "action_div_near":   mean_near,
        "action_div_safe":   mean_safe,
        "action_div_gap":    div_gap,
        "n_near_probes":     len(near_divs),
        "n_safe_probes":     len(safe_divs),
        "fatal_errors":      fatal_errors,
    }


def _make_fresh_agent(env, config_kwargs, seed):
    """Create a fresh agent + decoder + optimizer."""
    torch.manual_seed(seed)
    config = REEConfig.from_dims(**config_kwargs)
    agent = REEAgent(config)
    world_decoder = _make_world_decoder(
        config_kwargs["world_dim"], env.world_obs_dim
    )
    e12_params = [p for n_, p in agent.named_parameters() if "harm_eval" not in n_]
    e12_params += list(world_decoder.parameters())
    optimizer = optim.Adam(e12_params, lr=1e-3)
    return agent, world_decoder, optimizer


def run(
    seed: int = 0,
    warmup_episodes: int = 1000,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    probe_num_hazards: int = 6,
    probe_min_dist_safe: int = 2,
    probe_resets: int = 10,
    train_hazard_harm: float = 0.02,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    # Both hazard_harm and contaminated_harm set low so episodes are long.
    # EXQ-023 defaults: hazard_harm=0.5, contaminated_harm=0.4 → avg episode=18 steps.
    # Root cause: agent's own contamination trail (left on every cell it visits) depleted
    # health at 0.4/step — not just explicit hazard cells.
    # With both=0.02: avg episode ≈ 182 steps, 46% hit the 200-step cap.
    # 1000 eps × ~182 steps = ~182,000 training steps (vs 5,500 in EXQ-023).
    train_env = CausalGridWorld(
        seed=seed, size=12, num_hazards=15, num_resources=5,
        env_drift_interval=3, env_drift_prob=0.5,
        hazard_harm=train_hazard_harm,
        contaminated_harm=train_hazard_harm,
    )
    probe_env = CausalGridWorld(
        seed=seed + 100, size=12, num_hazards=probe_num_hazards, num_resources=5,
        hazard_harm=train_hazard_harm,
        contaminated_harm=train_hazard_harm,
    )

    config_kwargs = dict(
        body_obs_dim=train_env.body_obs_dim,
        world_obs_dim=train_env.world_obs_dim,
        action_dim=train_env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
    )

    print(f"[V3-EXQ-025] E2_world action-conditional divergence diagnostic", flush=True)
    print(f"  alpha_world={alpha_world}  probe_env={probe_num_hazards}hz  "
          f"min_dist_safe>{probe_min_dist_safe}", flush=True)

    # --- Condition A: rollout-only (no 1-step loss) ---
    print(f"\n--- Condition A: rollout-only (no 1-step loss) ---", flush=True)
    agent_A, decoder_A, opt_A = _make_fresh_agent(train_env, config_kwargs, seed)
    stats_A = _train_agent(
        agent_A, train_env, decoder_A, opt_A,
        warmup_episodes, steps_per_episode,
        use_1step_loss=False,
    )
    print(f"[V3-EXQ-025] Condition A divergence probe (no lstsq)...", flush=True)
    div_A_raw = _measure_action_divergence(
        agent_A, probe_env, probe_resets, W=None,
        min_dist_safe=probe_min_dist_safe,
    )

    # --- Condition B: with 1-step loss ---
    print(f"\n--- Condition B: with 1-step loss ---", flush=True)
    agent_B, decoder_B, opt_B = _make_fresh_agent(train_env, config_kwargs, seed)
    stats_B = _train_agent(
        agent_B, train_env, decoder_B, opt_B,
        warmup_episodes, steps_per_episode,
        use_1step_loss=True, e2w_1step_weight=1.0, e2w_rollout_weight=0.5,
    )
    print(f"[V3-EXQ-025] Condition B divergence probe (no lstsq)...", flush=True)
    div_B_raw = _measure_action_divergence(
        agent_B, probe_env, probe_resets, W=None,
        min_dist_safe=probe_min_dist_safe,
    )

    # Fit lstsq on condition B reafference data
    print(f"[V3-EXQ-025] Fitting lstsq on condition B data ({len(stats_B['reaf_data'])} steps)...",
          flush=True)
    W_B = _fit_lstsq_predictor(
        stats_B["reaf_data"], self_dim, train_env.action_dim, world_dim
    )

    # --- Condition C: 1-step loss + lstsq reafference correction ---
    print(f"[V3-EXQ-025] Condition C divergence probe (with lstsq correction)...", flush=True)
    div_C_corrected = _measure_action_divergence(
        agent_B, probe_env, probe_resets, W=W_B,
        min_dist_safe=probe_min_dist_safe,
    )

    # --- PASS / FAIL ---
    fatal_errors = (
        div_A_raw["fatal_errors"]
        + div_B_raw["fatal_errors"]
        + div_C_corrected["fatal_errors"]
    )

    n_near = div_B_raw["n_near_probes"]
    n_safe = div_B_raw["n_safe_probes"]

    c1_pass = stats_B["improvement_ratio"] > 2.0
    c2_pass = div_B_raw["action_div_gap"] > 0.002
    c3_pass = (
        div_A_raw["action_div_near"] > 1e-6 and
        div_B_raw["action_div_near"] > div_A_raw["action_div_near"] * 1.2
    )
    c4_pass = n_near >= 50 and n_safe >= 50
    c5_pass = fatal_errors == 0

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status   = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: e2w_improvement_ratio(B)={stats_B['improvement_ratio']:.2f}x <= 2.0"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: action_div_gap(B)={div_B_raw['action_div_gap']:.4f} <= 0.002"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: div_near(B)={div_B_raw['action_div_near']:.4f} not ≥1.2× "
            f"div_near(A)={div_A_raw['action_div_near']:.4f}"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: n_near={n_near}, n_safe={n_safe} (both must be >= 50)"
        )
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: fatal_errors={fatal_errors}")

    print(f"\nV3-EXQ-025 verdict: {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "fatal_error_count":          float(fatal_errors),
        "alpha_world":                float(alpha_world),
        # Condition A (rollout-only baseline)
        "condA_e2w_1step_mse":        float(stats_A["mean_e2w_1step_mse"]),
        "condA_identity_mse":         float(stats_A["mean_identity_mse"]),
        "condA_improvement_ratio":    float(stats_A["improvement_ratio"]),
        "condA_action_div_near":      float(div_A_raw["action_div_near"]),
        "condA_action_div_safe":      float(div_A_raw["action_div_safe"]),
        "condA_action_div_gap":       float(div_A_raw["action_div_gap"]),
        # Condition B (with 1-step loss)
        "condB_e2w_1step_mse":        float(stats_B["mean_e2w_1step_mse"]),
        "condB_identity_mse":         float(stats_B["mean_identity_mse"]),
        "condB_improvement_ratio":    float(stats_B["improvement_ratio"]),
        "condB_action_div_near":      float(div_B_raw["action_div_near"]),
        "condB_action_div_safe":      float(div_B_raw["action_div_safe"]),
        "condB_action_div_gap":       float(div_B_raw["action_div_gap"]),
        # Condition C (1-step + lstsq)
        "condC_action_div_near":      float(div_C_corrected["action_div_near"]),
        "condC_action_div_safe":      float(div_C_corrected["action_div_safe"]),
        "condC_action_div_gap":       float(div_C_corrected["action_div_gap"]),
        # Shared
        "n_near_probes":              float(n_near),
        "n_safe_probes":              float(n_safe),
        "probe_num_hazards":          float(probe_num_hazards),
        "probe_min_dist_safe":        float(probe_min_dist_safe),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-025 — E2_world Action-Conditional Divergence Diagnostic

**Status:** {status}
**alpha_world:** {alpha_world}
**Probe env:** {probe_num_hazards} hazards, min_dist > {probe_min_dist_safe}
**Seed:** {seed}

## What This Tests

E2_world action-conditional divergence: `||E2.world_forward(z, a1) - E2.world_forward(z, a2)||`
Measured at near-hazard vs safe positions WITHOUT going through net_eval.

Conditions:
- **A**: Rollout-only (replicates EXQ-023 training)
- **B**: With 1-step direct loss (EXQ-024 fix)
- **C**: Condition B + lstsq reafference correction

## E2_world Quality

| Condition | 1-step MSE | Identity MSE | Improvement ratio |
|---|---|---|---|
| A (rollout-only) | {stats_A['mean_e2w_1step_mse']:.5f} | {stats_A['mean_identity_mse']:.5f} | {stats_A['improvement_ratio']:.2f}× |
| B (1-step loss)  | {stats_B['mean_e2w_1step_mse']:.5f} | {stats_B['mean_identity_mse']:.5f} | {stats_B['improvement_ratio']:.2f}× |

## Action-Conditional Divergence

| Condition | near-hazard div | safe div | gap |
|---|---|---|---|
| A (rollout-only, no lstsq) | {div_A_raw['action_div_near']:.4f} | {div_A_raw['action_div_safe']:.4f} | {div_A_raw['action_div_gap']:.4f} |
| B (1-step loss, no lstsq)  | {div_B_raw['action_div_near']:.4f} | {div_B_raw['action_div_safe']:.4f} | {div_B_raw['action_div_gap']:.4f} |
| C (1-step + lstsq)         | {div_C_corrected['action_div_near']:.4f} | {div_C_corrected['action_div_safe']:.4f} | {div_C_corrected['action_div_gap']:.4f} |

n_near={n_near}  n_safe={n_safe}  (probe_env: {probe_num_hazards} hazards)

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: improvement_ratio(B) > 2.0× | {"PASS" if c1_pass else "FAIL"} | {stats_B['improvement_ratio']:.2f}× |
| C2: action_div_gap(B) > 0.002 | {"PASS" if c2_pass else "FAIL"} | {div_B_raw['action_div_gap']:.4f} |
| C3: div_near(B) ≥ 1.2× div_near(A) | {"PASS" if c3_pass else "FAIL"} | {div_B_raw['action_div_near']:.4f} vs {div_A_raw['action_div_near']:.4f} |
| C4: n_near >= 50 and n_safe >= 50 | {"PASS" if c4_pass else "FAIL"} | {n_near}, {n_safe} |
| C5: No fatal errors | {"PASS" if c5_pass else "FAIL"} | {fatal_errors} |

Criteria met: {criteria_met}/5 → **{status}**
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "supports" if all_pass else ("mixed" if criteria_met >= 3 else "weakens"),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": fatal_errors,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--warmup",         type=int,   default=300)
    parser.add_argument("--steps",          type=int,   default=200)
    parser.add_argument("--alpha-world",    type=float, default=0.9)
    parser.add_argument("--alpha-self",     type=float, default=0.3)
    parser.add_argument("--probe-hazards",      type=int,   default=6)
    parser.add_argument("--probe-min-dist",     type=int,   default=2)
    parser.add_argument("--probe-resets",       type=int,   default=10)
    parser.add_argument("--train-hazard-harm",  type=float, default=0.02)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        probe_num_hazards=args.probe_hazards,
        probe_min_dist_safe=args.probe_min_dist,
        probe_resets=args.probe_resets,
        train_hazard_harm=args.train_hazard_harm,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    out_dir = Path(__file__).resolve().parents[1] / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}", flush=True)
