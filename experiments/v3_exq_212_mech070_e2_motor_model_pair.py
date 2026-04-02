"""
EXQ-212: MECH-070 E2 vs E1 rollout horizon comparison

Claim: E2 is a conceptual-sensorium motor model with a planning horizon exceeding E1.

Design:
  - E2 condition: chain E2.predict_next_self(z_self, action) h times using actual actions
  - E1 condition: E1.predict_long_horizon(total_state, horizon=h) z_self slice (no actions)
  - Both compared against actual z_self at each horizon step h=1,...,H_MAX
  - E2 uses action conditioning; E1 predicts without knowing actions taken
  - Key distinction: E2 trained on motor-sensory prediction error (z_self);
    E1 trained on sensory prediction error (context-based, no action input).
  - If MECH-070 holds: E2 (action-conditioned) should degrade slower with horizon
    because each chained step is grounded to the actual trajectory.

Pre-registered acceptance thresholds:
  C1: E2 R2 at h=1 > E1 R2 at h=1   (E2 better at immediate step due to action conditioning)
  C2: E2 R2 at h=10 > E1 R2 at h=10 (E2 maintains advantage at medium horizon)
  C3: |slope_e2| < |slope_e1| (E2 R2 degrades more slowly with horizon, 2/3 seeds)
  C4: n_total_windows >= 50 (coverage sanity)

3 seeds x 100 warmup + 30 eval x 200 steps = 78k steps total.
Estimated: ~90 min on any machine.

Run pack written to REE_assembly/evidence/experiments/
"""

import os
import sys
import json
import random
import time
import argparse
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn.functional as F

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig

EXPERIMENT_ID      = "V3-EXQ-212"
EXPERIMENT_TYPE    = "v3_exq_212_mech070_e2_motor_model_pair"
CLAIM_IDS          = ["MECH-070"]
EXPERIMENT_PURPOSE = "evidence"

# Training config
NUM_SEEDS         = 3
WARMUP_EPISODES   = 100
EVAL_EPISODES     = 30
STEPS_PER_EPISODE = 200
ENV_SIZE          = 10
NUM_HAZARDS       = 1
HAZARD_HARM       = 0.5

# Horizon evaluation
HORIZONS = [1, 2, 3, 5, 7, 10, 15]
H_MAX    = max(HORIZONS)

# Pre-registered thresholds
THRESH_C4_MIN_WINDOWS = 50

OUTPUT_DIR = os.path.abspath(
    os.path.join(REPO_ROOT, "..", "REE_assembly", "evidence", "experiments",
                 EXPERIMENT_TYPE)
)

DRY_RUN_WARMUP = 3
DRY_RUN_EVAL   = 3


def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _r2(pred: np.ndarray, actual: np.ndarray) -> float:
    if pred.shape[0] < 2:
        return float("nan")
    ss_res = float(np.sum((actual - pred) ** 2))
    ss_tot = float(np.sum((actual - actual.mean(axis=0)) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _linear_slope(xs: list, ys: list) -> float:
    if len(xs) < 2:
        return float("nan")
    xa = np.array(xs, dtype=float)
    ya = np.array(ys, dtype=float)
    xm, ym = xa.mean(), ya.mean()
    denom = float(np.sum((xa - xm) ** 2))
    if denom < 1e-12:
        return float("nan")
    return float(np.sum((xa - xm) * (ya - ym)) / denom)


def run_seed(seed: int, dry_run: bool) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_warmup = DRY_RUN_WARMUP if dry_run else WARMUP_EPISODES
    n_eval   = DRY_RUN_EVAL   if dry_run else EVAL_EPISODES

    device = torch.device("cpu")

    env = CausalGridWorldV2(
        size=ENV_SIZE,
        num_hazards=NUM_HAZARDS,
        hazard_harm=HAZARD_HARM,
        use_proxy_fields=True,
    )

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
    )
    # Extract dims from config for slicing
    self_dim  = config.e1.self_dim
    world_dim = config.e1.world_dim

    agent = REEAgent(config)
    agent.to(device)

    n_actions = env.action_dim

    # ------------------------------------------------------------------ #
    # Phase 0: warmup                                                     #
    # ------------------------------------------------------------------ #
    print(f"  [EXQ-212] seed={seed} warmup={n_warmup} self_dim={self_dim}")
    agent.train()
    for _ep in range(n_warmup):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        for _step in range(STEPS_PER_EPISODE):
            action_idx = random.randint(0, n_actions - 1)
            action = _action_to_onehot(action_idx, n_actions, device)
            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)
            latent_state = agent.sense(obs_body, obs_world)
            agent._e1_tick(latent_state)
            try:
                agent._train_step(harm_signal=float(harm_signal), action=action)
            except Exception:
                pass
            if done:
                break

    # ------------------------------------------------------------------ #
    # Phase 1: eval -- collect trajectories for horizon comparison        #
    # ------------------------------------------------------------------ #
    agent.eval()
    print(f"  [EXQ-212] seed={seed} eval={n_eval}")

    e1_preds_by_h = {h: [] for h in HORIZONS}
    e2_preds_by_h = {h: [] for h in HORIZONS}
    actual_by_h   = {h: [] for h in HORIZONS}

    for _ep in range(n_eval):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        ep_z_selfs      = []   # list of [1, self_dim] tensors (actual)
        ep_total_states = []   # list of [1, self_dim+world_dim] tensors
        ep_actions      = []   # list of [1, action_dim] tensors
        ep_h_states     = []   # E1 hidden state snapshots (before _e1_tick each step)

        for _step in range(STEPS_PER_EPISODE):
            action_idx = random.randint(0, n_actions - 1)
            action = _action_to_onehot(action_idx, n_actions, device)

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)
            latent_state = agent.sense(obs_body, obs_world)

            # Snapshot E1 hidden state BEFORE _e1_tick updates it
            if agent.e1._hidden_state is not None:
                h_snap = (agent.e1._hidden_state[0].clone(), agent.e1._hidden_state[1].clone())
            else:
                h_snap = None

            agent._e1_tick(latent_state)

            z_s = latent_state.z_self.detach().clone()   # [1, self_dim]
            z_w = latent_state.z_world.detach().clone()  # [1, world_dim]
            total = torch.cat([z_s, z_w], dim=-1)        # [1, self_dim+world_dim]

            ep_z_selfs.append(z_s)
            ep_total_states.append(total)
            ep_actions.append(action.detach().clone())
            ep_h_states.append(h_snap)

            if done:
                break

        T = len(ep_z_selfs)
        if T < H_MAX + 2:
            continue

        stride = max(1, T // 20)
        for t in range(0, T - H_MAX - 1, stride):
            total_t = ep_total_states[t]  # [1, total_dim]
            z_s_t   = ep_z_selfs[t]       # [1, self_dim]

            # ---- E1: predict_long_horizon(total_state, horizon=H_MAX), slice z_self ----
            # Save/restore hidden state to avoid corrupting inference
            old_h = agent.e1._hidden_state
            if ep_h_states[t] is not None:
                agent.e1._hidden_state = ep_h_states[t]
            else:
                agent.e1.reset_hidden_state()

            with torch.no_grad():
                # predictions: [1, H_MAX, total_dim]
                e1_preds_all = agent.e1.predict_long_horizon(
                    total_t.float(), horizon=H_MAX
                )

            agent.e1._hidden_state = old_h  # restore

            # ---- E2: chain predict_next_self with actual actions ----
            e2_preds_per_h = {}
            with torch.no_grad():
                for h in HORIZONS:
                    z_s_chain = z_s_t.clone()
                    for step_offset in range(h):
                        ti = t + step_offset
                        if ti >= T:
                            break
                        z_s_chain = agent.e2.predict_next_self(
                            z_s_chain.float(), ep_actions[ti].float()
                        )
                    e2_preds_per_h[h] = z_s_chain  # [1, self_dim]

            # ---- Record at each horizon ----
            for h in HORIZONS:
                t_future = t + h
                if t_future >= T:
                    continue

                actual_zs  = ep_z_selfs[t_future].detach().cpu().numpy().reshape(-1)  # [self_dim]
                e1_zs_pred = e1_preds_all[0, h - 1, :self_dim].cpu().numpy()          # [self_dim]
                e2_zs_pred = e2_preds_per_h[h][0].cpu().numpy()                       # [self_dim]

                e1_preds_by_h[h].append(e1_zs_pred)
                e2_preds_by_h[h].append(e2_zs_pred)
                actual_by_h[h].append(actual_zs)

    # ---- Compute R2 at each horizon ----
    e1_r2_by_h = {}
    e2_r2_by_h = {}
    n_windows_by_h = {}
    for h in HORIZONS:
        n = len(actual_by_h[h])
        n_windows_by_h[h] = n
        if n < 5:
            e1_r2_by_h[h] = float("nan")
            e2_r2_by_h[h] = float("nan")
            continue
        act = np.stack(actual_by_h[h])    # [N, self_dim]
        e1p = np.stack(e1_preds_by_h[h])  # [N, self_dim]
        e2p = np.stack(e2_preds_by_h[h])  # [N, self_dim]
        e1_r2_by_h[h] = _r2(e1p, act)
        e2_r2_by_h[h] = _r2(e2p, act)

    # ---- Degradation slopes (R2 vs horizon) ----
    valid_hs = [h for h in HORIZONS if not np.isnan(e1_r2_by_h.get(h, float("nan")))
                and not np.isnan(e2_r2_by_h.get(h, float("nan")))]
    slope_e1 = _linear_slope(valid_hs, [e1_r2_by_h[h] for h in valid_hs])
    slope_e2 = _linear_slope(valid_hs, [e2_r2_by_h[h] for h in valid_hs])

    r2_e1_h1  = e1_r2_by_h.get(1, float("nan"))
    r2_e2_h1  = e2_r2_by_h.get(1, float("nan"))
    r2_e1_h10 = e1_r2_by_h.get(10, float("nan"))
    r2_e2_h10 = e2_r2_by_h.get(10, float("nan"))

    c1 = (not np.isnan(r2_e2_h1)) and (not np.isnan(r2_e1_h1)) and r2_e2_h1 > r2_e1_h1
    c2 = (not np.isnan(r2_e2_h10)) and (not np.isnan(r2_e1_h10)) and r2_e2_h10 > r2_e1_h10
    c3 = (not np.isnan(slope_e1)) and (not np.isnan(slope_e2)) and abs(slope_e2) < abs(slope_e1)
    n_total = sum(n_windows_by_h.values())
    c4 = n_total >= THRESH_C4_MIN_WINDOWS

    print(f"  [EXQ-212] seed={seed} r2_e1_h1={r2_e1_h1:.4f} r2_e2_h1={r2_e2_h1:.4f} "
          f"r2_e1_h10={r2_e1_h10:.4f} r2_e2_h10={r2_e2_h10:.4f} "
          f"slope_e1={slope_e1:.4f} slope_e2={slope_e2:.4f} "
          f"C1={c1} C2={c2} C3={c3} C4={c4}")

    return {
        "seed": seed,
        "e1_r2_by_horizon":  {str(h): e1_r2_by_h.get(h) for h in HORIZONS},
        "e2_r2_by_horizon":  {str(h): e2_r2_by_h.get(h) for h in HORIZONS},
        "n_windows_by_horizon": {str(h): n_windows_by_h.get(h, 0) for h in HORIZONS},
        "slope_e1":  slope_e1,
        "slope_e2":  slope_e2,
        "r2_e1_h1":  r2_e1_h1,
        "r2_e2_h1":  r2_e2_h1,
        "r2_e1_h10": r2_e1_h10,
        "r2_e2_h10": r2_e2_h10,
        "c1": c1, "c2": c2, "c3": c3, "c4": c4,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("[EXQ-212] MECH-070 E2 vs E1 rollout horizon pair")
    print(f"  dry_run={args.dry_run}")

    seeds = [42, 7, 123]
    seed_results = []
    t0 = time.time()

    for seed in seeds:
        r = run_seed(seed, dry_run=args.dry_run)
        seed_results.append(r)

    def _mean_finite(vals):
        v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
        return float(np.mean(v)) if v else float("nan")

    agg_e1_r2  = {str(h): _mean_finite([r["e1_r2_by_horizon"].get(str(h)) for r in seed_results]) for h in HORIZONS}
    agg_e2_r2  = {str(h): _mean_finite([r["e2_r2_by_horizon"].get(str(h)) for r in seed_results]) for h in HORIZONS}
    agg_slope_e1  = _mean_finite([r["slope_e1"]  for r in seed_results])
    agg_slope_e2  = _mean_finite([r["slope_e2"]  for r in seed_results])
    agg_r2_e1_h1  = _mean_finite([r["r2_e1_h1"]  for r in seed_results])
    agg_r2_e2_h1  = _mean_finite([r["r2_e2_h1"]  for r in seed_results])
    agg_r2_e1_h10 = _mean_finite([r["r2_e1_h10"] for r in seed_results])
    agg_r2_e2_h10 = _mean_finite([r["r2_e2_h10"] for r in seed_results])
    n_total = sum(sum(r["n_windows_by_horizon"].values()) for r in seed_results)

    # Pass requires 2/3 seeds on each criterion
    c1_pass = sum(1 for r in seed_results if r["c1"]) >= 2
    c2_pass = sum(1 for r in seed_results if r["c2"]) >= 2
    c3_pass = sum(1 for r in seed_results if r["c3"]) >= 2
    c4_pass = n_total >= THRESH_C4_MIN_WINDOWS

    overall_pass = c1_pass and c2_pass and c3_pass and c4_pass
    evidence_direction = "supports" if overall_pass else "weakens"

    run_id  = f"{EXPERIMENT_TYPE}_{int(t0)}_v3"
    elapsed = time.time() - t0

    result = {
        "run_id":             run_id,
        "experiment_type":    EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids":          CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": evidence_direction,
        "pass":               overall_pass,
        "elapsed_seconds":    elapsed,
        "dry_run":            args.dry_run,
        "config": {
            "num_seeds":         NUM_SEEDS,
            "warmup_episodes":   WARMUP_EPISODES if not args.dry_run else DRY_RUN_WARMUP,
            "eval_episodes":     EVAL_EPISODES   if not args.dry_run else DRY_RUN_EVAL,
            "steps_per_episode": STEPS_PER_EPISODE,
            "horizons":          HORIZONS,
            "h_max":             H_MAX,
        },
        "aggregate": {
            "e1_r2_by_horizon":  agg_e1_r2,
            "e2_r2_by_horizon":  agg_e2_r2,
            "slope_e1":          agg_slope_e1,
            "slope_e2":          agg_slope_e2,
            "r2_e1_h1":          agg_r2_e1_h1,
            "r2_e2_h1":          agg_r2_e2_h1,
            "r2_e1_h10":         agg_r2_e1_h10,
            "r2_e2_h10":         agg_r2_e2_h10,
            "n_total_windows":   n_total,
            "c1_pass":           c1_pass,
            "c2_pass":           c2_pass,
            "c3_pass":           c3_pass,
            "c4_pass":           c4_pass,
        },
        "per_seed": seed_results,
        "acceptance_thresholds": {
            "c1_e2_r2_h1_gt_e1":      "e2 R2 at h=1 > e1 R2 at h=1 (action conditioning advantage)",
            "c2_e2_r2_h10_gt_e1":     "e2 R2 at h=10 > e1 R2 at h=10 (maintained advantage)",
            "c3_e2_slope_lt_e1_abs":  "abs(slope_e2) < abs(slope_e1) (slower degradation)",
            "c4_min_windows":         THRESH_C4_MIN_WINDOWS,
        },
    }

    result_str = "PASS" if overall_pass else "FAIL"
    print(f"[EXQ-212] RESULT: {result_str} "
          f"r2_e1_h1={agg_r2_e1_h1:.4f} r2_e2_h1={agg_r2_e2_h1:.4f} "
          f"r2_e1_h10={agg_r2_e1_h10:.4f} r2_e2_h10={agg_r2_e2_h10:.4f} "
          f"slope_e1={agg_slope_e1:.4f} slope_e2={agg_slope_e2:.4f} "
          f"C1={c1_pass} C2={c2_pass} C3={c3_pass} C4={c4_pass}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[EXQ-212] Written: {out_path}")
