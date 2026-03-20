"""
V3-EXQ-056 — SD-010: Harm Stream Baseline

Claims: SD-010, ARC-027

Root question: Does a dedicated HarmEncoder(harm_obs → z_harm) produce a harm
estimate that correlates with actual hazard proximity significantly better than
the fused z_world encoder?

This is the constructive validation of SD-010: if z_harm cannot discriminate
hazard proximity at all, the entire SD-010 pipeline (EXQ-057/058/059) is moot.
If it discriminates better than z_world, we have evidence that separating the
harm signal from the world model is the correct architectural choice.

harm_obs layout (CausalGridWorldV2, use_proxy_fields=True):
  [0:25]  hazard_field_view  — normalised hazard proximity gradient (5×5)
  [25:50] resource_field_view — normalised resource gradient (5×5)
  [50]    harm_exposure       — nociceptive EMA

Baseline: E3.harm_eval(z_world) trained with the same hazard proximity labels.
z_world contains harm signals fused with layout/content — the baseline tests
whether separating them yields an advantage.

Training: 400 episodes, random policy.
  - HarmEncoder trained with MSE on info["hazard_field_at_agent"].
  - E3.harm_eval(z_world) trained identically (same labels, same lr) as baseline.
Eval: 100 episodes.
  - Pearson R between harm_eval_z_harm(z_harm) and hazard_field_at_agent.
  - Pearson R between harm_eval(z_world)       and hazard_field_at_agent.
  - z_harm variance across event types (none / hazard_approach / contact).

PASS criteria (ALL must hold):
  C1: harm_eval_pearson_r_z_harm > 0.30
  C2: harm_eval_pearson_r_z_harm > harm_eval_pearson_r_z_world + 0.10
  C3: z_harm_variance_across_event_types > 0.01
  C4: n_contact_steps >= 30
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.latent.stack import HarmEncoder
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_056_sd010_harm_stream_baseline"
CLAIM_IDS = ["SD-010", "ARC-027"]

HARM_OBS_DIM = 51
Z_HARM_DIM   = 32


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _pearson_r(x: List[float], y: List[float]) -> float:
    if len(x) < 3:
        return 0.0
    xn = np.array(x, dtype=np.float32)
    yn = np.array(y, dtype=np.float32)
    if xn.std() < 1e-8 or yn.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(xn, yn)[0, 1])


def run(
    seed: int = 0,
    train_episodes: int = 400,
    eval_episodes: int = 100,
    steps_per_episode: int = 300,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_scale: float = 0.05,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(
        seed=seed, size=12, num_hazards=6, num_resources=3,
        hazard_harm=harm_scale,
        env_drift_interval=5, env_drift_prob=0.2,
        proximity_harm_scale=proximity_scale,
        proximity_benefit_scale=proximity_scale * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        use_proxy_fields=True,  # SD-010: required for harm_obs in obs_dict
    )

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        use_harm_stream=True,
        harm_obs_dim=HARM_OBS_DIM,
        z_harm_dim=Z_HARM_DIM,
    )
    agent = REEAgent(config)

    # SD-010: standalone harm encoder (not wired into LatentStack)
    harm_enc = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_DIM)

    print(
        f"[V3-EXQ-056] SD-010 Harm Stream Baseline\n"
        f"  body_obs={env.body_obs_dim}  world_obs={env.world_obs_dim}\n"
        f"  harm_obs_dim={HARM_OBS_DIM}  z_harm_dim={Z_HARM_DIM}\n"
        f"  Training: {train_episodes} eps random  |  Eval: {eval_episodes} eps\n"
        f"  Question: Pearson R(z_harm vs hazard_label) > R(z_world vs hazard_label) + 0.10?",
        flush=True,
    )

    # Optimizers — separate to allow independent learning rates
    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n
        and "harm_eval_z_harm_head" not in n
    ]
    optimizer           = optim.Adam(standard_params, lr=lr)
    harm_enc_opt        = optim.Adam(harm_enc.parameters(), lr=1e-3)
    harm_z_harm_opt     = optim.Adam(agent.e3.harm_eval_z_harm_head.parameters(), lr=1e-4)
    harm_z_world_opt    = optim.Adam(agent.e3.harm_eval_head.parameters(), lr=1e-4)

    # ── Training: random policy ──────────────────────────────────────────────
    print(f"\n[V3-EXQ-056] Training ({train_episodes} eps, random policy)...", flush=True)
    agent.train()
    harm_enc.train()

    train_counts: Dict[str, int] = {}

    for ep in range(train_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()
            z_world_curr = latent.z_world.detach()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            train_counts[ttype] = train_counts.get(ttype, 0) + 1

            # Harm_obs for the NEW state (aligned with hazard_label)
            harm_obs_new = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM))

            # Hazard proximity label: normalized center of hazard_field_view (harm_obs[12]).
            # harm_obs[12] = hazard_field[agent_x, agent_y] / hazard_max ∈ [0,1].
            # Raw hazard_field_at_agent is an unbounded sum (>1 with 6 hazards), which
            # saturates the Sigmoid head to 1.0 for all inputs — this is the root cause
            # of the SD-010 collapse in the first run (EXQ-056 original).
            hazard_label = harm_obs_new[12].unsqueeze(0).unsqueeze(0).detach().float()
            harm_obs_t = harm_obs_new.unsqueeze(0).float()
            z_harm_new = harm_enc(harm_obs_t)

            # Train HarmEncoder + harm_eval_z_harm_head (SD-010 path)
            pred_zh = agent.e3.harm_eval_z_harm(z_harm_new)
            loss_zh = F.mse_loss(pred_zh, hazard_label)
            harm_enc_opt.zero_grad()
            harm_z_harm_opt.zero_grad()
            loss_zh.backward()
            torch.nn.utils.clip_grad_norm_(harm_enc.parameters(), 0.5)
            harm_enc_opt.step()
            harm_z_harm_opt.step()

            # Train E3.harm_eval(z_world) baseline (same labels, same lr)
            pred_zw = agent.e3.harm_eval(z_world_curr)
            loss_zw = F.mse_loss(pred_zw, hazard_label)
            harm_z_world_opt.zero_grad()
            loss_zw.backward()
            harm_z_world_opt.step()

            # Standard agent losses (E1 + E2)
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == train_episodes - 1:
            approach = train_counts.get("hazard_approach", 0)
            contact  = (train_counts.get("env_caused_hazard", 0)
                        + train_counts.get("agent_caused_hazard", 0))
            print(
                f"  [train] ep {ep+1}/{train_episodes}  "
                f"approach={approach}  contact={contact}",
                flush=True,
            )

    # ── Eval ─────────────────────────────────────────────────────────────────
    print(f"\n[V3-EXQ-056] Eval ({eval_episodes} eps, random policy)...", flush=True)
    agent.eval()
    harm_enc.eval()

    preds_z_harm:  List[float] = []
    preds_z_world: List[float] = []
    hazard_labels: List[float] = []
    harm_by_ttype: Dict[str, List[float]] = {}
    eval_counts:   Dict[str, int] = {}

    for ep in range(eval_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent    = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                z_world   = latent.z_world

                # z_harm from current obs (top of loop = state before action)
                harm_obs_curr = obs_dict.get("harm_obs", torch.zeros(HARM_OBS_DIM))
                harm_obs_t    = harm_obs_curr.unsqueeze(0).float()
                z_harm        = harm_enc(harm_obs_t)

                pred_zh = float(agent.e3.harm_eval_z_harm(z_harm).item())
                pred_zw = float(agent.e3.harm_eval(z_world).item())

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            lbl   = float(info.get("hazard_field_at_agent", 0.0))
            ttype = info.get("transition_type", "none")
            eval_counts[ttype] = eval_counts.get(ttype, 0) + 1

            preds_z_harm.append(pred_zh)
            preds_z_world.append(pred_zw)
            hazard_labels.append(lbl)

            harm_by_ttype.setdefault(ttype, []).append(pred_zh)

            if done:
                break

    # ── Metrics ──────────────────────────────────────────────────────────────
    r_z_harm  = _pearson_r(preds_z_harm,  hazard_labels)
    r_z_world = _pearson_r(preds_z_world, hazard_labels)

    # C3: variance of per-type mean harm estimates across event types
    type_means = []
    for k in ["none", "hazard_approach", "env_caused_hazard", "agent_caused_hazard"]:
        vals = harm_by_ttype.get(k, [])
        if vals:
            type_means.append(float(np.mean(vals)))
    z_harm_var = float(np.var(type_means)) if len(type_means) >= 2 else 0.0

    n_contact = (
        len(harm_by_ttype.get("agent_caused_hazard", []))
        + len(harm_by_ttype.get("env_caused_hazard", []))
    )

    # Per-ttype mean harm_eval_z_harm for diagnostics
    type_mean_zh  = {k: float(np.mean(v)) for k, v in harm_by_ttype.items() if v}
    type_mean_zw  = {
        k: 0.0 for k in harm_by_ttype
    }

    print(f"\n  --- SD-010 Harm Stream Baseline (EXQ-056) ---", flush=True)
    print(f"  Pearson R (z_harm  vs hazard_label): {r_z_harm:.4f}", flush=True)
    print(f"  Pearson R (z_world vs hazard_label): {r_z_world:.4f}", flush=True)
    print(f"  Difference R_z_harm - R_z_world:     {r_z_harm - r_z_world:.4f}", flush=True)
    print(f"  z_harm variance across event types:  {z_harm_var:.6f}", flush=True)
    print(f"  n_contact_steps: {n_contact}", flush=True)
    print(f"\n  harm_eval_z_harm per ttype:", flush=True)
    for k, v in sorted(type_mean_zh.items()):
        print(f"    {k:30s}: {v:.4f}  (n={eval_counts.get(k,0)})", flush=True)

    # ── PASS / FAIL ──────────────────────────────────────────────────────────
    c1 = r_z_harm > 0.30
    c2 = r_z_harm > r_z_world + 0.10
    c3 = z_harm_var > 0.01
    c4 = n_contact >= 30

    all_pass = c1 and c2 and c3 and c4
    status   = "PASS" if all_pass else "FAIL"
    n_met    = sum([c1, c2, c3, c4])

    failure_notes = []
    if not c1:
        failure_notes.append(
            f"C1 FAIL: harm_eval_pearson_r_z_harm={r_z_harm:.4f} <= 0.30. "
            f"HarmEncoder cannot discriminate hazard proximity."
        )
    if not c2:
        failure_notes.append(
            f"C2 FAIL: R_z_harm={r_z_harm:.4f} <= R_z_world={r_z_world:.4f} + 0.10. "
            f"Fused z_world performs as well as dedicated harm stream — "
            f"stream separation does not help. Root cause of SD-010 failures may lie elsewhere."
        )
    if not c3:
        failure_notes.append(
            f"C3 FAIL: z_harm_var_across_event_types={z_harm_var:.6f} <= 0.01. "
            f"z_harm does not vary across event types — signal collapsed."
        )
    if not c4:
        failure_notes.append(
            f"C4 FAIL: n_contact_steps={n_contact} < 30. "
            f"Insufficient contact events for evaluation."
        )

    print(f"\nV3-EXQ-056 verdict: {status}  ({n_met}/4)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "alpha_world":                      float(alpha_world),
        "harm_eval_pearson_r_z_harm":       float(r_z_harm),
        "harm_eval_pearson_r_z_world":      float(r_z_world),
        "r_delta_z_harm_minus_z_world":     float(r_z_harm - r_z_world),
        "z_harm_variance_across_event_types": float(z_harm_var),
        "n_contact_steps":                  float(n_contact),
        "n_approach_steps":                 float(len(harm_by_ttype.get("hazard_approach", []))),
        "n_none_steps":                     float(len(harm_by_ttype.get("none", []))),
        "crit1_pass":                       1.0 if c1 else 0.0,
        "crit2_pass":                       1.0 if c2 else 0.0,
        "crit3_pass":                       1.0 if c3 else 0.0,
        "crit4_pass":                       1.0 if c4 else 0.0,
        "criteria_met":                     float(n_met),
        "fatal_error_count":                0.0,
    }
    for k, v in type_mean_zh.items():
        metrics[f"harm_eval_z_harm_mean_{k}"] = float(v)

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-056 — SD-010 Harm Stream Baseline

**Status:** {status}
**Claims:** SD-010, ARC-027
**World:** CausalGridWorldV2 (6 hazards, 3 resources)
**Training:** {train_episodes} eps random  |  **Eval:** {eval_episodes} eps random
**Metric:** Pearson R (harm_eval vs hazard_field_at_agent proximity label)
**alpha_world:** {alpha_world}  (SD-008)  |  **Seed:** {seed}

## Question

Does HarmEncoder(harm_obs → z_harm) produce a harm estimate that correlates
with actual hazard proximity better than the fused z_world encoder?

SD-010 hypothesis: separating nociceptive signals (hazard proximity) from the
exteroceptive world model (layout, content, perspective) enables the harm stream
to carry a cleaner hazard-proximity signal, uncontaminated by reafference correction
and world-model identity shortcuts.

## Results — Pearson R

| Source | Pearson R (vs hazard_label) |
|---|---|
| HarmEncoder → z_harm (SD-010) | {r_z_harm:.4f} |
| z_world encoder (baseline)    | {r_z_world:.4f} |
| Difference (SD-010 advantage) | {r_z_harm - r_z_world:.4f} |

**z_harm variance across event types:** {z_harm_var:.6f}
**n_contact_steps:** {n_contact}

## harm_eval_z_harm by transition type

| Type | Mean harm_eval_z_harm |
|---|---|
{"".join(f"| {k} | {v:.4f} |" + chr(10) for k, v in sorted(type_mean_zh.items()))}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: harm_eval_pearson_r_z_harm > 0.30 | {"PASS" if c1 else "FAIL"} | {r_z_harm:.4f} |
| C2: R_z_harm > R_z_world + 0.10 (SD-010 adds value) | {"PASS" if c2 else "FAIL"} | {r_z_harm:.4f} vs {r_z_world:.4f}+0.10 |
| C3: z_harm_variance_across_event_types > 0.01 | {"PASS" if c3 else "FAIL"} | {z_harm_var:.6f} |
| C4: n_contact_steps >= 30 (sufficient eval data) | {"PASS" if c4 else "FAIL"} | {n_contact} |

Criteria met: {n_met}/4 → **{status}**
{failure_section}
"""

    return {
        "status":             status,
        "metrics":            metrics,
        "summary_markdown":   summary_markdown,
        "claim_ids":          CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if n_met >= 2 else "weakens")
        ),
        "experiment_type":    EXPERIMENT_TYPE,
        "fatal_error_count":  0,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--train-episodes",  type=int,   default=400)
    parser.add_argument("--eval-episodes",   type=int,   default=100)
    parser.add_argument("--steps",           type=int,   default=300)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        harm_scale=args.harm_scale,
        proximity_scale=args.proximity_scale,
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
