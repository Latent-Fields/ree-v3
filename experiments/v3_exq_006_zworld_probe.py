"""
V3-EXQ-006 — Diagnostic: What Does z_world Encode?

Motivation:
  EXQ-002r4 and EXQ-004r2 both produce calibration_gap ≈ 0 despite correct probe
  design and non-degenerate harm_eval. EXQ-005 shows E2.world_forward loss = 0.00002
  (delta ≈ 0 is trivially good) and E3 harm loss ≈ log(2) (random chance).

  Hypothesis: the world encoder (trained by E1 prediction loss) produces z_world
  representations where hazard proximity is NOT distinguishable. If z_world doesn't
  encode hazards, the SD-003 pipeline has no signal to work with.

Experiment:
  1. Train agent for 300 episodes (RANDOM policy, standard warmup)
  2. Collect 2000 labeled (z_world, label) pairs:
     - hazard_adjacent: 1 if a hazard is within Manhattan distance 1, else 0
     - agent_x, agent_y: grid position (regression targets)
  3. Fit a linear probe (logistic regression / linear regression) on z_world
  4. Report accuracy / R2 on held-out 20% test set

PASS criteria (ALL must hold):
  C1: hazard_adjacent probe accuracy > 0.70 (z_world encodes hazard proximity)
  C2: position_x probe R2 > 0.50 (z_world encodes spatial info)
  C3: position_y probe R2 > 0.50
  C4: fatal_error_count == 0
  C5: n_samples >= 1000

If C1 FAIL: world encoder doesn't learn hazard features → needs reconstruction loss
If C1 PASS but SD-003 still fails: E2/E3 training issue → needs multi-step or contrastive loss
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Dict, List, Tuple
import random
import math

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_006_zworld_probe"
CLAIM_IDS = ["SD-003", "SD-005"]


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _train_agent(
    agent: REEAgent,
    env: CausalGridWorld,
    num_episodes: int,
    steps_per_episode: int,
    lr: float,
) -> int:
    """Train agent with RANDOM policy, return total harm events."""
    agent.train()
    e12_params = [p for n, p in agent.named_parameters() if "harm_eval" not in n]
    opt_e12 = optim.Adam(e12_params, lr=lr)
    opt_e3  = optim.Adam(agent.e3.harm_eval_head.parameters(), lr=1e-4)

    world_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_buf:    List[torch.Tensor] = []
    no_harm_buf: List[torch.Tensor] = []
    total_harm = 0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev = None
        action_prev  = None

        for step in range(steps_per_episode):
            latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
            agent.clock.advance()

            if z_world_prev is not None and action_prev is not None:
                world_buf.append((z_world_prev.detach(), action_prev.detach(), latent.z_world.detach()))
                if len(world_buf) > 500: world_buf = world_buf[-500:]

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action
            z_world_prev = latent.z_world.detach()
            action_prev  = action.detach()

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            if harm_signal < 0:
                total_harm += 1
                harm_buf.append(latent.z_world.detach())
                if len(harm_buf) > 500: harm_buf = harm_buf[-500:]
            else:
                if step % 3 == 0:
                    no_harm_buf.append(latent.z_world.detach())
                    if len(no_harm_buf) > 500: no_harm_buf = no_harm_buf[-500:]

            # E1 + E2 training
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            e2w_loss = e1_loss.new_zeros(())
            if len(world_buf) >= 4:
                n = min(16, len(world_buf))
                idxs = torch.randperm(len(world_buf))[:n].tolist()
                zw_t, acts, zw_t1 = zip(*[world_buf[i] for i in idxs])
                e2w_loss = F.mse_loss(
                    agent.e2.world_forward(torch.cat(zw_t), torch.cat(acts)),
                    torch.cat(zw_t1)
                )
            total = e1_loss + e2_loss + e2w_loss
            if total.requires_grad:
                opt_e12.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt_e12.step()

            # E3 training
            n_h, n_nh = len(harm_buf), len(no_harm_buf)
            if n_h >= 4 and n_nh >= 4:
                k = min(16, n_h, n_nh)
                zw_h  = torch.cat([harm_buf[i]    for i in torch.randperm(n_h)[:k].tolist()])
                zw_nh = torch.cat([no_harm_buf[i] for i in torch.randperm(n_nh)[:k].tolist()])
                labels = torch.cat([torch.ones(k, 1, device=agent.device),
                                    torch.zeros(k, 1, device=agent.device)])
                pred = agent.e3.harm_eval(torch.cat([zw_h, zw_nh]))
                if not torch.isnan(pred).any():
                    e3_loss = F.binary_cross_entropy(pred.clamp(1e-6, 1-1e-6), labels)
                    opt_e3.zero_grad()
                    e3_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e3.harm_eval_head.parameters(), 0.5)
                    opt_e3.step()

            if done:
                break

        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            print(f"  [train] ep {ep+1}/{num_episodes}  harm={total_harm}", flush=True)

    return total_harm


def _collect_probe_data(
    agent: REEAgent,
    env: CausalGridWorld,
    num_resets: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collect (z_world, hazard_adjacent, agent_x, agent_y) at many grid states.
    """
    agent.eval()
    zw_list = []
    ha_list = []   # hazard_adjacent: 1 if hazard within Manhattan dist 1
    ax_list = []   # agent_x (normalized 0-1)
    ay_list = []   # agent_y (normalized 0-1)

    hazard_type = env.ENTITY_TYPES["hazard"]
    wall_type   = env.ENTITY_TYPES["wall"]

    for _ in range(num_resets):
        env.reset()
        # Sample every non-wall cell on this grid
        for px in range(env.size):
            for py in range(env.size):
                if int(env.grid[px, py]) in (wall_type, hazard_type):
                    continue
                env.agent_x = px
                env.agent_y = py
                obs_dict = env._get_observation_dict()

                with torch.no_grad():
                    latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
                    zw_list.append(latent.z_world.detach().squeeze(0))

                # Labels
                min_dist = min(abs(px - hx) + abs(py - hy) for hx, hy in env.hazards)
                ha_list.append(1.0 if min_dist <= 1 else 0.0)
                ax_list.append(px / env.size)
                ay_list.append(py / env.size)

    Z = torch.stack(zw_list)                      # [N, world_dim]
    H = torch.tensor(ha_list).unsqueeze(1)         # [N, 1]
    AX = torch.tensor(ax_list).unsqueeze(1)        # [N, 1]
    AY = torch.tensor(ay_list).unsqueeze(1)        # [N, 1]

    return Z, H, AX, AY


def _linear_probe_classification(
    Z: torch.Tensor, Y: torch.Tensor, train_frac: float = 0.8, lr: float = 0.01, epochs: int = 200,
) -> Tuple[float, float]:
    """Logistic regression probe. Returns (train_acc, test_acc)."""
    n = Z.size(0)
    perm = torch.randperm(n)
    split = int(n * train_frac)
    Z_tr, Y_tr = Z[perm[:split]], Y[perm[:split]]
    Z_te, Y_te = Z[perm[split:]], Y[perm[split:]]

    # Standardize
    mu, sigma = Z_tr.mean(0), Z_tr.std(0) + 1e-8
    Z_tr = (Z_tr - mu) / sigma
    Z_te = (Z_te - mu) / sigma

    d = Z.size(1)
    W = torch.zeros(d, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = optim.Adam([W, b], lr=lr)

    for _ in range(epochs):
        logits = Z_tr @ W + b
        loss = F.binary_cross_entropy_with_logits(logits, Y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        tr_pred = (torch.sigmoid(Z_tr @ W + b) > 0.5).float()
        te_pred = (torch.sigmoid(Z_te @ W + b) > 0.5).float()
        train_acc = (tr_pred == Y_tr).float().mean().item()
        test_acc  = (te_pred == Y_te).float().mean().item()

    return train_acc, test_acc


def _linear_probe_regression(
    Z: torch.Tensor, Y: torch.Tensor, train_frac: float = 0.8, lr: float = 0.01, epochs: int = 200,
) -> Tuple[float, float]:
    """Linear regression probe. Returns (train_R2, test_R2)."""
    n = Z.size(0)
    perm = torch.randperm(n)
    split = int(n * train_frac)
    Z_tr, Y_tr = Z[perm[:split]], Y[perm[:split]]
    Z_te, Y_te = Z[perm[split:]], Y[perm[split:]]

    mu, sigma = Z_tr.mean(0), Z_tr.std(0) + 1e-8
    Z_tr = (Z_tr - mu) / sigma
    Z_te = (Z_te - mu) / sigma

    d = Z.size(1)
    W = torch.zeros(d, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = optim.Adam([W, b], lr=lr)

    for _ in range(epochs):
        pred = Z_tr @ W + b
        loss = F.mse_loss(pred, Y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        te_pred = Z_te @ W + b
        ss_res = ((Y_te - te_pred) ** 2).sum().item()
        ss_tot = ((Y_te - Y_te.mean()) ** 2).sum().item()
        test_r2 = 1.0 - (ss_res / max(ss_tot, 1e-12))

        tr_pred = Z_tr @ W + b
        ss_res_tr = ((Y_tr - tr_pred) ** 2).sum().item()
        ss_tot_tr = ((Y_tr - Y_tr.mean()) ** 2).sum().item()
        train_r2 = 1.0 - (ss_res_tr / max(ss_tot_tr, 1e-12))

    return train_r2, test_r2


def run(
    seed: int = 0,
    warmup_episodes: int = 300,
    steps_per_episode: int = 200,
    probe_resets: int = 30,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorld(seed=seed, num_hazards=8)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
    )

    fatal_errors = 0

    # Train agent
    print(f"\n[V3-EXQ-006] Training agent ({warmup_episodes} episodes) ...", flush=True)
    agent = REEAgent(config)
    total_harm = _train_agent(agent, env, warmup_episodes, steps_per_episode, lr)
    print(f"  Training complete. Harm events: {total_harm}", flush=True)

    # Collect probe data
    print(f"  Collecting probe data ({probe_resets} grid resets) ...", flush=True)
    try:
        Z, H, AX, AY = _collect_probe_data(agent, env, probe_resets)
        n_samples = Z.size(0)
        print(f"  Collected {n_samples} samples. Hazard-adjacent rate: {H.mean().item():.3f}", flush=True)
    except Exception:
        import traceback
        fatal_errors += 1
        print(f"  FATAL: {traceback.format_exc()}", flush=True)
        n_samples = 0
        Z = H = AX = AY = torch.zeros(1, 1)

    # Run probes
    ha_train_acc, ha_test_acc = 0.0, 0.0
    px_train_r2, px_test_r2 = 0.0, 0.0
    py_train_r2, py_test_r2 = 0.0, 0.0

    if n_samples >= 100:
        print("  Running hazard_adjacent probe (logistic) ...", flush=True)
        ha_train_acc, ha_test_acc = _linear_probe_classification(Z, H)
        print(f"    train_acc={ha_train_acc:.3f}  test_acc={ha_test_acc:.3f}", flush=True)

        print("  Running position_x probe (linear) ...", flush=True)
        px_train_r2, px_test_r2 = _linear_probe_regression(Z, AX)
        print(f"    train_R2={px_train_r2:.3f}  test_R2={px_test_r2:.3f}", flush=True)

        print("  Running position_y probe (linear) ...", flush=True)
        py_train_r2, py_test_r2 = _linear_probe_regression(Z, AY)
        print(f"    train_R2={py_train_r2:.3f}  test_R2={py_test_r2:.3f}", flush=True)

    # Also probe on UNTRAINED agent for baseline
    print("\n  Running probes on UNTRAINED agent (baseline) ...", flush=True)
    torch.manual_seed(seed + 9999)
    agent_untrained = REEAgent(config)
    try:
        Z_u, H_u, AX_u, AY_u = _collect_probe_data(agent_untrained, env, probe_resets)
        ha_untrained_train, ha_untrained_test = _linear_probe_classification(Z_u, H_u)
        px_untrained_train, px_untrained_test = _linear_probe_regression(Z_u, AX_u)
        py_untrained_train, py_untrained_test = _linear_probe_regression(Z_u, AY_u)
        print(f"  UNTRAINED hazard_adj: test_acc={ha_untrained_test:.3f}", flush=True)
        print(f"  UNTRAINED pos_x: test_R2={px_untrained_test:.3f}", flush=True)
        print(f"  UNTRAINED pos_y: test_R2={py_untrained_test:.3f}", flush=True)
    except Exception:
        import traceback
        fatal_errors += 1
        print(f"  FATAL: {traceback.format_exc()}", flush=True)
        ha_untrained_test = px_untrained_test = py_untrained_test = 0.0

    # PASS criteria
    crit1_pass = ha_test_acc > 0.70
    crit2_pass = px_test_r2 > 0.50
    crit3_pass = py_test_r2 > 0.50
    crit4_pass = fatal_errors == 0
    crit5_pass = n_samples >= 1000

    all_pass = all([crit1_pass, crit2_pass, crit3_pass, crit4_pass, crit5_pass])
    status = "PASS" if all_pass else "FAIL"
    criteria_met = sum([crit1_pass, crit2_pass, crit3_pass, crit4_pass, crit5_pass])

    failure_notes = []
    if not crit1_pass: failure_notes.append(f"C1 FAIL: hazard_adj test_acc {ha_test_acc:.3f} <= 0.70 — z_world does NOT encode hazard proximity")
    if not crit2_pass: failure_notes.append(f"C2 FAIL: position_x test_R2 {px_test_r2:.3f} <= 0.50")
    if not crit3_pass: failure_notes.append(f"C3 FAIL: position_y test_R2 {py_test_r2:.3f} <= 0.50")
    if not crit4_pass: failure_notes.append(f"C4 FAIL: fatal_errors={fatal_errors}")
    if not crit5_pass: failure_notes.append(f"C5 FAIL: n_samples {n_samples} < 1000")

    print(f"\nV3-EXQ-006 verdict: {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-006 — Diagnostic: What Does z_world Encode?

**Status:** {status}
**Agent training:** {warmup_episodes} episodes, RANDOM policy
**Probe data:** {probe_resets} grid resets, {n_samples} total samples
**Seed:** {seed}

## Motivation

EXQ-002r4 and EXQ-004r2 produce calibration_gap ≈ 0 despite correct probe design.
EXQ-005 shows E2.world_forward loss = 0.00002 (delta ≈ 0 is trivially good) and
E3 harm loss ≈ log(2) (random chance). This diagnostic checks whether z_world
encodes the information needed for SD-003 attribution.

## Linear Probe Results

| Probe | Target | TRAINED test | UNTRAINED test | Criterion |
|---|---|---|---|---|
| Hazard adjacency | binary (dist <= 1) | {ha_test_acc:.3f} | {ha_untrained_test:.3f} | > 0.70 |
| Position X | regression (0-1) | R2={px_test_r2:.3f} | R2={px_untrained_test:.3f} | R2 > 0.50 |
| Position Y | regression (0-1) | R2={py_test_r2:.3f} | R2={py_untrained_test:.3f} | R2 > 0.50 |

Hazard-adjacent positive rate: {H.mean().item():.3f} (baseline accuracy for majority-class classifier)

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: hazard_adj test_acc > 0.70 | {"PASS" if crit1_pass else "FAIL"} | {ha_test_acc:.3f} |
| C2: position_x test_R2 > 0.50 | {"PASS" if crit2_pass else "FAIL"} | {px_test_r2:.3f} |
| C3: position_y test_R2 > 0.50 | {"PASS" if crit3_pass else "FAIL"} | {py_test_r2:.3f} |
| C4: No fatal errors | {"PASS" if crit4_pass else "FAIL"} | {fatal_errors} |
| C5: n_samples >= 1000 | {"PASS" if crit5_pass else "FAIL"} | {n_samples} |

## Interpretation

{"C1 PASS: z_world encodes hazard proximity. The bottleneck is in E2/E3 training, not the encoder. Fix: multi-step E2 training and/or contrastive loss." if crit1_pass else "C1 FAIL: z_world does NOT encode hazard proximity. The world encoder (trained by E1 prediction loss alone) discards hazard-relevant features. Fix: add reconstruction loss to world encoder so z_world preserves full observation content."}

{"C2+C3: z_world encodes spatial position — the encoder is learning meaningful representations." if crit2_pass and crit3_pass else "C2/C3 suggest z_world has limited spatial encoding — encoder may need reconstruction loss."}

Criteria met: {criteria_met}/5 -> **{status}**
{failure_section}
"""

    metrics = {
        "fatal_error_count": float(fatal_errors),
        "n_samples": float(n_samples),
        "hazard_adj_base_rate": float(H.mean().item()) if n_samples > 0 else 0.0,
        "hazard_adj_train_acc": ha_train_acc,
        "hazard_adj_test_acc": ha_test_acc,
        "hazard_adj_untrained_test_acc": ha_untrained_test,
        "position_x_train_r2": px_train_r2,
        "position_x_test_r2": px_test_r2,
        "position_x_untrained_test_r2": px_untrained_test,
        "position_y_train_r2": py_train_r2,
        "position_y_test_r2": py_test_r2,
        "position_y_untrained_test_r2": py_untrained_test,
        "warmup_harm_events": float(total_harm),
        "crit1_pass": 1.0 if crit1_pass else 0.0,
        "crit2_pass": 1.0 if crit2_pass else 0.0,
        "crit3_pass": 1.0 if crit3_pass else 0.0,
        "crit4_pass": 1.0 if crit4_pass else 0.0,
        "crit5_pass": 1.0 if crit5_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

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
    parser.add_argument("--seed",         type=int, default=0)
    parser.add_argument("--warmup",       type=int, default=300)
    parser.add_argument("--probe-resets", type=int, default=30)
    parser.add_argument("--steps",        type=int, default=200)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        probe_resets=args.probe_resets,
        steps_per_episode=args.steps,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]

    out_dir = Path(__file__).resolve().parents[1] / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}", flush=True)
