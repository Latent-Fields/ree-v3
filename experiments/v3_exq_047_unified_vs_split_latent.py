"""
V3-EXQ-047 -- Unified vs Split Latent Ablation (SD-005 efficiency test)

Claims: SD-005, MECH-069

Tests whether the z_self/z_world dimensional split itself (not just optimizer
separation) provides efficiency advantages for calibration and attribution.

Two conditions (same env seed, same architecture -- only latent mode differs):

  split   -- Current architecture. z_self=32, z_world=32 maintained as separate
             channels with separate gradient flows. Each channel is specialized
             by its respective error signal (E2→z_self, E3→z_world).

  unified -- unified_latent_mode=True. After EMA blending, z_self and z_world
             are averaged into a single shared representation. Both channels
             carry the same vector. E2 and E3 gradients flow through the same
             latent dimensions -- no specialization.

Motivation: EXQ-034 showed E2-ablation doesn't hurt calibration_gap (E3 alone
suffices for harm detection). EXQ-035 tested optimizer separation but was
confounded. This experiment isolates the ENCODING split question: does
maintaining separate latent channels improve calibration or attribution even
when both modules are present and trained?

PASS criteria (ALL must hold):
  C1: calibration_gap_approach_split > calibration_gap_approach_unified + 0.05
      (split latent better for harm detection)
  C2: attribution_gap_split > attribution_gap_unified + 0.01
      (split latent better for attribution)
  C3: calibration_gap_approach_split > 0.10
      (full split system works)
  C4: attribution_gap_split > 0.0
      (split attribution direction correct)
  C5: n_approach_eval >= 50
      (sufficient approach steps)

NOTE: C1 or C2 FAIL -> unified performs comparably -> latent split provides
limited benefit beyond module specialization already covered by separate
optimizers. This is a valid scientific result (not a REE failure).

Follows EXQ-034 pattern: obs-only E3 training, Fix-2 on E2-predicted states,
same CausalGridWorldV2 config. See EXQ-034 for methodology details.
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_047_unified_vs_split_latent"
CLAIM_IDS = ["SD-005", "MECH-069"]


# ------------------------------------------------------------------
# Helpers (identical to EXQ-034)
# ------------------------------------------------------------------

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _compute_world_forward_r2(
    agent: REEAgent,
    wf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> float:
    if len(wf_data) < 20:
        return 0.0
    n = len(wf_data)
    n_train = int(n * 0.8)
    with torch.no_grad():
        zw_all  = torch.cat([d[0] for d in wf_data], dim=0)
        a_all   = torch.cat([d[1] for d in wf_data], dim=0)
        zw1_all = torch.cat([d[2] for d in wf_data], dim=0)
        pred_all  = agent.e2.world_forward(zw_all.to(agent.device), a_all.to(agent.device))
        pred_test = pred_all[n_train:]
        tgt_test  = zw1_all[n_train:].to(agent.device)
        if pred_test.shape[0] == 0:
            return 0.0
        ss_res = ((tgt_test - pred_test) ** 2).sum()
        ss_tot = ((tgt_test - tgt_test.mean(0, keepdim=True)) ** 2).sum()
        r2 = float((1 - ss_res / (ss_tot + 1e-8)).item())
    print(f"  world_forward R2 (test n={pred_test.shape[0]}): {r2:.4f}", flush=True)
    return r2


def _train(
    agent: REEAgent,
    env,
    optimizer: optim.Optimizer,
    harm_eval_optimizer: optim.Optimizer,
    world_forward_optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
    condition_label: str = "split",
) -> Dict:
    agent.train()

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF = 2000

    wf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_WF = 5000

    counts: Dict[str, int] = {}
    num_actions = env.action_dim

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev = None
        a_prev = None

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_curr = latent.z_world.detach()

            action_idx = random.randint(0, num_actions - 1)
            action = _action_to_onehot(action_idx, num_actions, agent.device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            counts[ttype] = counts.get(ttype, 0) + 1

            if harm_signal < 0:
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            if z_world_prev is not None and a_prev is not None:
                wf_data.append((z_world_prev.cpu(), a_prev.cpu(), z_world_curr.cpu()))
                if len(wf_data) > MAX_WF:
                    wf_data = wf_data[-MAX_WF:]

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if len(wf_data) >= 16:
                k = min(32, len(wf_data))
                idxs = torch.randperm(len(wf_data))[:k].tolist()
                zw_b  = torch.cat([wf_data[i][0] for i in idxs]).to(agent.device)
                a_b   = torch.cat([wf_data[i][1] for i in idxs]).to(agent.device)
                zw1_b = torch.cat([wf_data[i][2] for i in idxs]).to(agent.device)
                pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(pred, zw1_b)
                if wf_loss.requires_grad:
                    world_forward_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e2.world_transition.parameters(), 0.5
                    )
                    world_forward_optimizer.step()

            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()

                zw_pos_obs = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
                zw_neg_obs = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)

                with torch.no_grad():
                    a_rand_pos = torch.zeros(k_pos, num_actions, device=agent.device)
                    a_rand_pos[torch.arange(k_pos), torch.randint(0, num_actions, (k_pos,))] = 1.0
                    a_rand_neg = torch.zeros(k_neg, num_actions, device=agent.device)
                    a_rand_neg[torch.arange(k_neg), torch.randint(0, num_actions, (k_neg,))] = 1.0
                    zw_pos_pred = agent.e2.world_forward(zw_pos_obs.to(agent.device), a_rand_pos)
                    zw_neg_pred = agent.e2.world_forward(zw_neg_obs.to(agent.device), a_rand_neg)

                zw_b = torch.cat([
                    zw_pos_obs.to(agent.device), zw_neg_obs.to(agent.device),
                    zw_pos_pred, zw_neg_pred,
                ], dim=0)
                target = torch.cat([
                    torch.ones(k_pos,  1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                    torch.ones(k_pos,  1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                ], dim=0)

                pred_harm = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred_harm, target)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5
                    )
                    harm_eval_optimizer.step()

            z_world_prev = z_world_curr
            a_prev = action.detach()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == num_episodes - 1:
            approach = counts.get("hazard_approach", 0)
            contact  = counts.get("env_caused_hazard", 0) + counts.get("agent_caused_hazard", 0)
            print(
                f"  [{condition_label}|train] ep {ep+1}/{num_episodes}  "
                f"approach={approach}  contact={contact}",
                flush=True,
            )

    return {"counts": counts, "wf_data": wf_data}


def _eval_condition(
    agent: REEAgent,
    env,
    num_episodes: int,
    steps_per_episode: int,
    condition_label: str = "split",
) -> Dict:
    agent.eval()

    ttypes_tracked = ["none", "hazard_approach", "env_caused_hazard", "agent_caused_hazard"]
    num_actions = env.action_dim

    calibration_scores: Dict[str, List[float]] = {t: [] for t in ttypes_tracked}
    all_harm_scores: List[float] = []
    causal_sigs: Dict[str, List[float]] = {t: [] for t in ttypes_tracked}

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                z_world = latent.z_world

            action_idx = random.randint(0, num_actions - 1)
            action = _action_to_onehot(action_idx, num_actions, agent.device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            with torch.no_grad():
                harm_score = float(agent.e3.harm_eval(z_world).item())
                all_harm_scores.append(harm_score)
                if ttype in calibration_scores:
                    calibration_scores[ttype].append(harm_score)

                z_world_actual = agent.e2.world_forward(z_world, action)
                harm_actual    = agent.e3.harm_eval(z_world_actual)

                sigs: List[float] = []
                for cf_idx in range(num_actions):
                    if cf_idx == action_idx:
                        continue
                    a_cf    = _action_to_onehot(cf_idx, num_actions, agent.device)
                    z_cf    = agent.e2.world_forward(z_world, a_cf)
                    harm_cf = agent.e3.harm_eval(z_cf)
                    sigs.append(float((harm_actual - harm_cf).item()))

                mean_sig = float(np.mean(sigs)) if sigs else 0.0
                if ttype in causal_sigs:
                    causal_sigs[ttype].append(mean_sig)

            if done:
                break

    def _mean(lst: List[float]) -> float:
        return float(np.mean(lst)) if lst else 0.0

    cal_means = {t: _mean(calibration_scores[t]) for t in ttypes_tracked}
    sig_means  = {t: _mean(causal_sigs[t])        for t in ttypes_tracked}
    n_counts   = {t: len(calibration_scores[t])   for t in ttypes_tracked}

    harm_pred_std = float(torch.tensor(all_harm_scores).std().item()) if len(all_harm_scores) > 1 else 0.0
    calibration_gap_approach = cal_means["hazard_approach"] - cal_means["none"]
    calibration_gap_contact  = (
        (cal_means["env_caused_hazard"] + cal_means["agent_caused_hazard"]) / 2.0
        - cal_means["none"]
    )
    attribution_gap = sig_means["hazard_approach"] - sig_means["env_caused_hazard"]

    print(f"\n  --- {condition_label} Eval ---", flush=True)
    for t in ttypes_tracked:
        print(f"    {t:28s}: harm_eval={cal_means[t]:.4f}  n={n_counts[t]}", flush=True)
    print(f"    calibration_gap_approach: {calibration_gap_approach:.4f}", flush=True)
    print(f"    calibration_gap_contact:  {calibration_gap_contact:.4f}", flush=True)
    print(f"    harm_pred_std: {harm_pred_std:.4f}", flush=True)
    for t in ttypes_tracked:
        print(f"    {t:28s}: causal_sig={sig_means[t]:.6f}", flush=True)
    print(f"    attribution_gap (approach - env): {attribution_gap:.6f}", flush=True)

    return {
        "cal_means":                cal_means,
        "sig_means":                sig_means,
        "n_counts":                 n_counts,
        "calibration_gap_approach": calibration_gap_approach,
        "calibration_gap_contact":  calibration_gap_contact,
        "attribution_gap":          attribution_gap,
        "harm_pred_std":            harm_pred_std,
    }


def _build_agent_and_optimizers(
    env,
    self_dim: int,
    world_dim: int,
    alpha_world: float,
    alpha_self: float,
    lr: float,
    seed: int,
    unified_latent_mode: bool = False,
) -> Tuple[REEAgent, optim.Optimizer, optim.Optimizer, optim.Optimizer]:
    torch.manual_seed(seed)
    random.seed(seed)

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=env.action_dim,
    )
    config.latent.unified_latent_mode = unified_latent_mode

    agent = REEAgent(config)

    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n
        and "world_transition" not in n
        and "world_action_encoder" not in n
    ]
    world_forward_params = (
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters())
    )
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())

    optimizer               = optim.Adam(standard_params,      lr=lr)
    world_forward_optimizer = optim.Adam(world_forward_params, lr=1e-3)
    harm_eval_optimizer     = optim.Adam(harm_eval_params,     lr=1e-4)

    return agent, optimizer, world_forward_optimizer, harm_eval_optimizer


def run(
    seed: int = 0,
    warmup_episodes: int = 500,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_scale: float = 0.05,
    **kwargs,
) -> dict:
    def _make_env():
        return CausalGridWorldV2(
            seed=seed, size=12, num_hazards=4, num_resources=5,
            hazard_harm=harm_scale,
            env_drift_interval=5, env_drift_prob=0.1,
            proximity_harm_scale=proximity_scale,
            proximity_benefit_scale=proximity_scale * 0.6,
            proximity_approach_threshold=0.15,
            hazard_field_decay=0.5,
        )

    print(
        f"[V3-EXQ-044] Unified vs Split Latent Ablation\n"
        f"  seed={seed}  warmup={warmup_episodes}  eval_eps={eval_episodes}\n"
        f"  alpha_world={alpha_world}  harm_scale={harm_scale}  proximity_scale={proximity_scale}\n"
        f"  Conditions: split | unified",
        flush=True,
    )

    condition_results: Dict[str, Dict] = {}
    condition_wf_r2:   Dict[str, float] = {}

    for condition, unified_mode in [("split", False), ("unified", True)]:
        print(f"\n{'=' * 60}", flush=True)
        print(f"[V3-EXQ-044] CONDITION: {condition} (unified_latent_mode={unified_mode})", flush=True)
        print("=" * 60, flush=True)

        env_cond = _make_env()
        agent_cond, opt, wf_opt, he_opt = _build_agent_and_optimizers(
            env_cond, self_dim, world_dim, alpha_world, alpha_self, lr, seed,
            unified_latent_mode=unified_mode,
        )

        train_data = _train(
            agent_cond, env_cond, opt, he_opt, wf_opt,
            warmup_episodes, steps_per_episode,
            condition_label=condition,
        )
        condition_wf_r2[condition] = _compute_world_forward_r2(agent_cond, train_data["wf_data"])

        print(f"[V3-EXQ-044] Eval: {condition} ({eval_episodes} eps)...", flush=True)
        result = _eval_condition(
            agent_cond, env_cond, eval_episodes, steps_per_episode,
            condition_label=condition,
        )
        condition_results[condition] = result

    # ------------------------------------------------------------------
    # PASS / FAIL
    # ------------------------------------------------------------------
    r_split   = condition_results["split"]
    r_unified = condition_results["unified"]

    crit1_pass = r_split["calibration_gap_approach"] > r_unified["calibration_gap_approach"] + 0.05
    crit2_pass = r_split["attribution_gap"]          > r_unified["attribution_gap"]          + 0.01
    crit3_pass = r_split["calibration_gap_approach"] > 0.10
    crit4_pass = r_split["attribution_gap"]          > 0.0
    crit5_pass = r_split["n_counts"].get("hazard_approach", 0) >= 50

    criteria_met = sum([crit1_pass, crit2_pass, crit3_pass, crit4_pass, crit5_pass])
    all_pass = all([crit1_pass, crit2_pass, crit3_pass, crit4_pass, crit5_pass])
    status = "PASS" if all_pass else "FAIL"

    print(f"\n[V3-EXQ-044] Verdict: {status} ({criteria_met}/5 criteria met)", flush=True)
    print(f"  C1 cal_gap split>unified+0.05: {r_split['calibration_gap_approach']:.4f} vs {r_unified['calibration_gap_approach']:.4f} -> {'PASS' if crit1_pass else 'FAIL'}", flush=True)
    print(f"  C2 attr_gap split>unified+0.01: {r_split['attribution_gap']:.4f} vs {r_unified['attribution_gap']:.4f} -> {'PASS' if crit2_pass else 'FAIL'}", flush=True)
    print(f"  C3 cal_gap_split>0.10: {r_split['calibration_gap_approach']:.4f} -> {'PASS' if crit3_pass else 'FAIL'}", flush=True)
    print(f"  C4 attr_gap_split>0.0: {r_split['attribution_gap']:.4f} -> {'PASS' if crit4_pass else 'FAIL'}", flush=True)
    print(f"  C5 n_approach>={50}: {r_split['n_counts'].get('hazard_approach', 0)} -> {'PASS' if crit5_pass else 'FAIL'}", flush=True)

    metrics = {
        "calibration_gap_approach_split":   r_split["calibration_gap_approach"],
        "calibration_gap_approach_unified": r_unified["calibration_gap_approach"],
        "calibration_gap_contact_split":    r_split["calibration_gap_contact"],
        "calibration_gap_contact_unified":  r_unified["calibration_gap_contact"],
        "attribution_gap_split":            r_split["attribution_gap"],
        "attribution_gap_unified":          r_unified["attribution_gap"],
        "harm_pred_std_split":              r_split["harm_pred_std"],
        "harm_pred_std_unified":            r_unified["harm_pred_std"],
        "wf_r2_split":                      condition_wf_r2.get("split", 0.0),
        "wf_r2_unified":                    condition_wf_r2.get("unified", 0.0),
        "n_approach_eval_split":            float(r_split["n_counts"].get("hazard_approach", 0)),
        "n_approach_eval_unified":          float(r_unified["n_counts"].get("hazard_approach", 0)),
        "causal_sig_approach_split":        r_split["sig_means"].get("hazard_approach", 0.0),
        "causal_sig_approach_unified":      r_unified["sig_means"].get("hazard_approach", 0.0),
        "causal_sig_none_split":            r_split["sig_means"].get("none", 0.0),
        "causal_sig_none_unified":          r_unified["sig_means"].get("none", 0.0),
        "alpha_world":                      float(alpha_world),
        "proximity_scale":                  float(proximity_scale),
        "seed":                             float(seed),
        "warmup_episodes":                  float(warmup_episodes),
        "crit1_pass": 1.0 if crit1_pass else 0.0,
        "crit2_pass": 1.0 if crit2_pass else 0.0,
        "crit3_pass": 1.0 if crit3_pass else 0.0,
        "crit4_pass": 1.0 if crit4_pass else 0.0,
        "crit5_pass": 1.0 if crit5_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    summary_markdown = f"""# V3-EXQ-044 -- Unified vs Split Latent Ablation

**Status:** {status} ({criteria_met}/5 criteria)
**Conditions:** split (z_self≠z_world) vs unified (z_self=z_world=avg)
**alpha_world:** {alpha_world}  **Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps

## PASS Criteria

| Criterion | Split | Unified | Pass? |
|---|---|---|---|
| C1: cal_gap_approach split > unified+0.05 | {r_split['calibration_gap_approach']:.4f} | {r_unified['calibration_gap_approach']:.4f} | {'PASS' if crit1_pass else 'FAIL'} |
| C2: attribution_gap split > unified+0.01 | {r_split['attribution_gap']:.4f} | {r_unified['attribution_gap']:.4f} | {'PASS' if crit2_pass else 'FAIL'} |
| C3: cal_gap_split > 0.10 | {r_split['calibration_gap_approach']:.4f} | -- | {'PASS' if crit3_pass else 'FAIL'} |
| C4: attr_gap_split > 0.0 | {r_split['attribution_gap']:.4f} | -- | {'PASS' if crit4_pass else 'FAIL'} |
| C5: n_approach >= 50 | {r_split['n_counts'].get('hazard_approach', 0)} | -- | {'PASS' if crit5_pass else 'FAIL'} |

## World-Forward R2

- Split: {condition_wf_r2.get('split', 0.0):.4f}
- Unified: {condition_wf_r2.get('unified', 0.0):.4f}

## Interpretation

- **C1 PASS -> C2 PASS**: Latent split provides advantage on both calibration and attribution.
  SD-005 confirmed as efficiency-relevant (not just attribution-relevant).
- **C1 FAIL, C2 PASS**: Split helps attribution but not harm detection calibration.
  Consistent with EXQ-034 finding that E2 ablation doesn't hurt calibration.
- **Both FAIL**: Latent split provides no measurable advantage over unified representation.
  Efficiency gains are from module specialization (EXQ-034) and SD-004 action objects, not encoding split.
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "supports" if all_pass else ("mixed" if criteria_met >= 3 else "weakens"),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--warmup",         type=int,   default=500)
    parser.add_argument("--eval-eps",       type=int,   default=50)
    parser.add_argument("--steps",          type=int,   default=200)
    parser.add_argument("--alpha-world",    type=float, default=0.9)
    parser.add_argument("--harm-scale",     type=float, default=0.02)
    parser.add_argument("--proximity-scale",type=float, default=0.05)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        harm_scale=args.harm_scale,
        proximity_scale=args.proximity_scale,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]

    out_dir = Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}", flush=True)
