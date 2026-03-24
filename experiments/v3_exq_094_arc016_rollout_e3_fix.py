"""
V3-EXQ-094 -- ARC-016 Harm Variance Commit: Rollout E3 Training Fix

Claims: ARC-016

Fix for EXQ-088 (FAIL: harm_var_mean identical across all hazard_harm levels,
Pearson r close to 0).

Root cause: agent.e3.harm_eval_z_harm_head was never trained during warmup.
std_params included the head but compute_prediction_loss() + compute_e2_loss()
produce no gradient through the harm_eval pathway, so the head stayed at random
initialisation. At eval time, harm_bridge(E2.world_forward(z_world, a_cand))
output was fed to an untrained head, producing noise rather than harm scores.

Additionally: E3 must see HarmBridge outputs (not just raw HarmEncoder outputs)
during training, because at eval time it receives z_harm from HarmBridge, not from
HarmEncoder directly. Training E3 only on HarmEncoder outputs creates a distribution
mismatch at eval.

Fix: during warmup, after the HarmBridge training step, add an explicit E3 training
step using z_harm_bridge = harm_bridge(z_world).detach() and the actual harm label.
This trains E3 on the same distribution it encounters at eval.

Protocol identical to EXQ-088 (warmup + hazard_harm sweep). Discriminative pair
mirrors EXQ-088 for direct comparison.

PASS criteria (ALL):
  C1: var_diff > 0 at >= 3 hazard_harm levels (variance higher in more dangerous envs)
  C2: Pearson r(hazard_harm, harm_var_mean) > 0.6
  C3: harm_var_cv > 0.10 (coefficient of variation -- scores differ across candidates)
  C4: commit_rate in low-danger env > commit_rate in high-danger env
  C5: no fatal errors
"""

import sys
import random
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.latent.stack import HarmEncoder, HarmBridge
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_094_arc016_rollout_e3_fix"
CLAIM_IDS = ["ARC-016"]

HARM_OBS_DIM = 51
Z_HARM_DIM   = 32

HAZARD_HARM_SWEEP = [0.005, 0.01, 0.02, 0.05, 0.10]
WARMUP_EPS   = 200
EVAL_EPS     = 80
STEPS_PER_EP = 150


def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _run_sweep_level(
    agent: REEAgent,
    harm_enc: HarmEncoder,
    harm_bridge: HarmBridge,
    hazard_harm: float,
    seed: int,
    eval_episodes: int,
    steps_per_episode: int,
    num_candidates: int = 8,
) -> Dict[str, float]:
    """Run evaluation at one hazard_harm level, return harm_variance stats."""
    env = CausalGridWorldV2(
        seed=seed + int(hazard_harm * 1000),
        size=10, num_hazards=4, num_resources=3,
        hazard_harm=hazard_harm,
        env_drift_prob=0.1,
        proximity_harm_scale=hazard_harm * 2.0,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        use_proxy_fields=True,
    )
    num_actions = env.action_dim

    agent.eval(); harm_enc.eval(); harm_bridge.eval()

    harm_var_list: List[float] = []
    commit_list:   List[float] = []
    commit_threshold = agent.e3.config.commitment_threshold

    for ep in range(eval_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            with torch.no_grad():
                latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
                agent.clock.advance()
                z_world = latent.z_world

                harm_scores_candidates: List[float] = []
                for ci in range(num_candidates):
                    a_cand = _action_to_onehot(
                        random.randint(0, num_actions-1), num_actions, agent.device)
                    z_w_next    = agent.e2.world_forward(z_world, a_cand)
                    z_harm_cand = harm_bridge(z_w_next)
                    h_score     = float(agent.e3.harm_eval_z_harm(z_harm_cand).item())
                    harm_scores_candidates.append(h_score)

                harm_var = float(np.var(harm_scores_candidates))
                harm_var_list.append(harm_var)

                committed = harm_var < commit_threshold
                commit_list.append(1.0 if committed else 0.0)

            action = _action_to_onehot(random.randint(0, num_actions-1), num_actions, agent.device)
            agent._last_action = action
            flat_obs, _, done, _, obs_dict = env.step(action)
            if done: break

    return {
        "harm_var_mean":  float(np.mean(harm_var_list)) if harm_var_list else 0.0,
        "harm_var_std":   float(np.std(harm_var_list))  if harm_var_list else 0.0,
        "commit_rate":    float(np.mean(commit_list))   if commit_list   else 0.0,
        "n_steps":        float(len(harm_var_list)),
    }


def run(
    seed: int = 0,
    warmup_episodes: int = WARMUP_EPS,
    eval_episodes: int = EVAL_EPS,
    steps_per_episode: int = STEPS_PER_EP,
    world_dim: int = 32,
    self_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env_warmup = CausalGridWorldV2(
        seed=seed, size=10, num_hazards=4, num_resources=3,
        hazard_harm=0.02,
        env_drift_prob=0.1,
        proximity_harm_scale=0.04,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        use_proxy_fields=True,
    )

    config = REEConfig.from_dims(
        body_obs_dim=env_warmup.body_obs_dim,
        world_obs_dim=env_warmup.world_obs_dim,
        action_dim=env_warmup.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=0.3,
        reafference_action_dim=env_warmup.action_dim,
        use_harm_stream=True,
        harm_obs_dim=HARM_OBS_DIM,
        z_harm_dim=Z_HARM_DIM,
    )
    agent = REEAgent(config)

    harm_enc    = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_DIM)
    harm_bridge = HarmBridge(world_dim=world_dim, z_harm_dim=Z_HARM_DIM)

    num_actions = env_warmup.action_dim

    print(
        f"[V3-EXQ-094] ARC-016 harm variance commit fix\n"
        f"  Fix: E3 harm_eval head trained on HarmBridge outputs during warmup\n"
        f"  Sweep: hazard_harm in {HAZARD_HARM_SWEEP}\n"
        f"  Warmup: {warmup_episodes} eps @ hazard_harm=0.02",
        flush=True,
    )

    # Exclude harm_eval from std_params -- it gets its own targeted optimizer
    std_params = [p for n, p in agent.named_parameters()
                  if "harm_eval" not in n
                  and "world_transition" not in n
                  and "world_action_encoder" not in n]
    wf_params  = (list(agent.e2.world_transition.parameters())
                  + list(agent.e2.world_action_encoder.parameters()))
    opt_std     = optim.Adam(std_params, lr=lr)
    opt_wf      = optim.Adam(wf_params, lr=1e-3)
    opt_he      = optim.Adam(harm_enc.parameters(), lr=1e-3)
    opt_hb      = optim.Adam(harm_bridge.parameters(), lr=1e-3)
    # FIX: dedicated optimizer for E3 harm head, trained on HarmBridge outputs
    opt_e3_harm = optim.Adam(agent.e3.harm_eval_z_harm_head.parameters(), lr=1e-4)

    wf_data: List = []
    agent.train(); harm_enc.train(); harm_bridge.train()

    for ep in range(warmup_episodes):
        flat_obs, obs_dict = env_warmup.reset()
        agent.reset()
        z_world_prev = None; a_prev = None

        for _ in range(steps_per_episode):
            latent  = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
            agent.clock.advance()
            z_world = latent.z_world.detach()

            harm_obs   = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM)).float()
            action_idx = random.randint(0, num_actions - 1)
            action     = _action_to_onehot(action_idx, num_actions, agent.device)
            agent._last_action = action
            flat_obs, _, done, _, obs_dict = env_warmup.step(action)

            label = harm_obs[12].unsqueeze(0).unsqueeze(0)

            # Train HarmEncoder on proximity label
            z_harm = harm_enc(harm_obs.unsqueeze(0))
            pred_he = agent.e3.harm_eval_z_harm(z_harm.detach())
            loss_he = F.mse_loss(pred_he, label)
            opt_he.zero_grad(); loss_he.backward(); opt_he.step()

            # Train HarmBridge: HarmBridge(z_world) ~= HarmEncoder(harm_obs)
            z_harm_target = harm_enc(harm_obs.unsqueeze(0)).detach()
            bridge_pred   = harm_bridge(z_world)
            loss_hb = F.mse_loss(bridge_pred, z_harm_target)
            opt_hb.zero_grad(); loss_hb.backward(); opt_hb.step()

            # FIX: train E3 harm head on HarmBridge outputs (the eval-time distribution)
            z_harm_bridge = harm_bridge(z_world).detach()
            pred_e3 = agent.e3.harm_eval_z_harm(z_harm_bridge)
            loss_e3 = F.mse_loss(pred_e3, label)
            opt_e3_harm.zero_grad(); loss_e3.backward(); opt_e3_harm.step()

            if z_world_prev is not None:
                wf_data.append((z_world_prev.cpu(), a_prev.cpu(), z_world.cpu()))
                if len(wf_data) > 3000: wf_data = wf_data[-3000:]

            if len(wf_data) >= 16:
                k = min(32, len(wf_data))
                idxs  = torch.randperm(len(wf_data))[:k].tolist()
                zw_b  = torch.cat([wf_data[i][0] for i in idxs]).to(agent.device)
                a_b   = torch.cat([wf_data[i][1] for i in idxs]).to(agent.device)
                zw1_b = torch.cat([wf_data[i][2] for i in idxs]).to(agent.device)
                wf_loss = F.mse_loss(agent.e2.world_forward(zw_b, a_b), zw1_b)
                if wf_loss.requires_grad:
                    opt_wf.zero_grad(); wf_loss.backward(); opt_wf.step()

            total = agent.compute_prediction_loss() + agent.compute_e2_loss()
            if total.requires_grad:
                opt_std.zero_grad(); total.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt_std.step()

            z_world_prev = z_world; a_prev = action.detach()
            if done: break

        if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
            print(f"  [warmup] ep {ep+1}/{warmup_episodes}", flush=True)

    # -- Sweep ----------------------------------------------------------------
    print(f"\n  Sweeping hazard_harm levels...", flush=True)
    sweep_results: Dict[float, Dict] = {}

    for hh in HAZARD_HARM_SWEEP:
        res = _run_sweep_level(
            agent, harm_enc, harm_bridge, hh, seed, eval_episodes, steps_per_episode)
        sweep_results[hh] = res
        print(
            f"  hazard_harm={hh:.3f}: harm_var_mean={res['harm_var_mean']:.6f}  "
            f"commit_rate={res['commit_rate']:.3f}",
            flush=True,
        )

    # -- Metrics --------------------------------------------------------------
    hh_vals  = HAZARD_HARM_SWEEP
    hv_means = [sweep_results[hh]["harm_var_mean"] for hh in hh_vals]
    cr_vals  = [sweep_results[hh]["commit_rate"]   for hh in hh_vals]

    diffs    = [hv_means[i+1] - hv_means[i] for i in range(len(hv_means)-1)]
    c1_count = sum(1 for d in diffs if d > 0)
    c1 = c1_count >= 3

    r_val = float(np.corrcoef(hh_vals, hv_means)[0, 1]) if len(hh_vals) >= 3 else 0.0
    c2 = r_val > 0.6

    mid_idx      = len(hh_vals) // 2
    hv_std_mid   = sweep_results[hh_vals[mid_idx]]["harm_var_std"]
    hv_mean_mid  = sweep_results[hh_vals[mid_idx]]["harm_var_mean"]
    hv_cv        = hv_std_mid / (hv_mean_mid + 1e-8)
    c3 = hv_cv > 0.10

    c4 = cr_vals[0] > cr_vals[-1]
    c5 = True

    all_pass = c1 and c2 and c3 and c4 and c5
    status   = "PASS" if all_pass else "FAIL"
    n_met    = sum([c1, c2, c3, c4, c5])

    failure_notes = []
    if not c1: failure_notes.append(
        f"C1 FAIL: harm_var increases at only {c1_count}/4 transitions. "
        f"harm_var_means: {[f'{v:.6f}' for v in hv_means]}")
    if not c2: failure_notes.append(
        f"C2 FAIL: Pearson r={r_val:.3f} <= 0.6. Harm variance not scaling with danger.")
    if not c3: failure_notes.append(
        f"C3 FAIL: harm_var CV={hv_cv:.3f} <= 0.10. Low between-candidate variance.")
    if not c4: failure_notes.append(
        f"C4 FAIL: commit_rate[low={hh_vals[0]}]={cr_vals[0]:.3f} <= "
        f"commit_rate[high={hh_vals[-1]}]={cr_vals[-1]:.3f}")

    print(f"\nV3-EXQ-094 verdict: {status}  ({n_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics: Dict[str, float] = {
        "pearson_r":               r_val,
        "c1_increase_count":       float(c1_count),
        "hv_cv_mid":               float(hv_cv),
        "commit_rate_low_danger":  float(cr_vals[0]),
        "commit_rate_high_danger": float(cr_vals[-1]),
        "crit1_pass":              1.0 if c1 else 0.0,
        "crit2_pass":              1.0 if c2 else 0.0,
        "crit3_pass":              1.0 if c3 else 0.0,
        "crit4_pass":              1.0 if c4 else 0.0,
        "crit5_pass":              1.0 if c5 else 0.0,
        "criteria_met":            float(n_met),
        "fatal_error_count":       0.0,
    }
    for hh in hh_vals:
        k = f"harm_var_mean_{str(hh).replace('.', 'p')}"
        metrics[k] = float(sweep_results[hh]["harm_var_mean"])

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    rows = "\n".join(
        f"| {hh} | {sweep_results[hh]['harm_var_mean']:.6f} | {sweep_results[hh]['commit_rate']:.3f} |"
        for hh in hh_vals
    )

    summary_markdown = f"""# V3-EXQ-094 -- ARC-016: Harm Variance Commit (Rollout E3 Fix)

**Status:** {status}
**Claims:** ARC-016
**Fixes:** EXQ-088 (E3 harm_eval head not trained on rollout distribution)

## Root Cause Fixed (EXQ-088)

In EXQ-088, agent.e3.harm_eval_z_harm_head was never trained during warmup.
std_params included it, but compute_prediction_loss() + compute_e2_loss() produce
no gradients through the harm_eval pathway, so the head stayed at random init.

Additionally: E3 at eval time receives z_harm from harm_bridge(E2.world_forward(z_world, a)),
but warmup only had the head see z_harm from HarmEncoder(harm_obs) -- a different
distribution. Even if the head had been trained, distribution mismatch would suppress
the variance signal.

Fix: explicit E3 training step in warmup loop using z_harm_bridge = harm_bridge(z_world).detach()
as input and the actual harm label as target. E3 now trains on exactly the
distribution it encounters at eval time.

## Sweep Results

| hazard_harm | harm_var_mean | commit_rate |
|-------------|---------------|-------------|
{rows}

- Pearson r(hazard_harm, harm_var_mean): {r_val:.4f}
- harm_var CV at mid level: {hv_cv:.4f}

## PASS Criteria

| Criterion | Result | Value |
|-----------|--------|-------|
| C1: harm_var increases at >= 3 transitions | {"PASS" if c1 else "FAIL"} | {c1_count}/4 |
| C2: Pearson r > 0.6 | {"PASS" if c2 else "FAIL"} | {r_val:.4f} |
| C3: harm_var CV > 0.10 | {"PASS" if c3 else "FAIL"} | {hv_cv:.4f} |
| C4: commit_rate[low] > commit_rate[high] | {"PASS" if c4 else "FAIL"} | {cr_vals[0]:.3f} vs {cr_vals[-1]:.3f} |
| C5: no fatal errors | {"PASS" if c5 else "FAIL"} | - |

Criteria met: {n_met}/5 -> **{status}**
{failure_section}
"""

    return {
        "status":             status,
        "metrics":            metrics,
        "summary_markdown":   summary_markdown,
        "claim_ids":          CLAIM_IDS,
        "evidence_direction": "supports" if all_pass else ("mixed" if n_met >= 3 else "weakens"),
        "experiment_type":    EXPERIMENT_TYPE,
        "fatal_error_count":  0,
    }


if __name__ == "__main__":
    import argparse, json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",   type=int, default=0)
    parser.add_argument("--warmup", type=int, default=WARMUP_EPS)
    parser.add_argument("--eval",   type=int, default=EVAL_EPS)
    parser.add_argument("--steps",  type=int, default=STEPS_PER_EP)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval,
        steps_per_episode=args.steps,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"]      = ts
    result["claim"]              = CLAIM_IDS[0]
    result["verdict"]            = result["status"]
    result["run_id"]             = f"{EXPERIMENT_TYPE}_{ts}_v3"
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
        print(f"  {k}: {v}", flush=True)
