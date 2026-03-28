#!/opt/local/bin/python3
"""
V3-EXQ-101 -- SD-011 / ARC-016 harm_obs_a normalization fix validation

Claims: SD-011, ARC-016

EXQ-100b FAIL/RAW_EMA_FAIL (2026-03-27):
  C1 FAIL: raw harm_obs_a HIGH=0.131 vs LOW=0.197 (inverted -- more hazards = lower accumulation).
  C2 FAIL: raw autocorr_lag10=0.069 (expected ~0.60 for tau=20 EMA).
  Root cause identified: prox_now was normalized by hazard_field.max() per step.
    In high-density environments, overlapping hazard fields push the max higher,
    shrinking each proximity contribution -> EMA tracks density-relative not absolute exposure.
    This inverts the expected relationship (more hazards -> smaller values after normalization).

Fix applied to causal_grid_world.py (2026-03-27):
  Removed division by hazard_max and resource_max in harm_obs_a EMA update.
  Now uses raw field values clipped to [0, 1] -- absolute exposure, density-additive.

PASS criteria (same as EXQ-100b):
  C1: raw_harm_obs_a_mean_high > raw_harm_obs_a_mean_low * 1.1  (EMA responds to density)
  C2: raw_autocorr_lag10 > 0.3                                   (EMA has temporal persistence)
  C3: z_harm_a_mean_high > z_harm_a_mean_low * 1.1              (encoder output responds)
  C4: z_harm_a_autocorr_lag10 > z_harm_s_autocorr_lag10         (encoder preserves stability)
  C5: no fatal errors
  PASS if C1 AND C2 AND C3 AND C4 AND C5.

Expected results after fix:
  C1: high-density raw mean > low-density (previously inverted)
  C2: autocorr_lag10 ~0.5-0.6 (tau=20 -> (0.95)^10 ~ 0.60 theoretical)
  C3/C4: encoder should now train on a meaningful signal

Supersedes: V3-EXQ-100b
"""

import sys
import random
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.latent.stack import HarmEncoder, AffectiveHarmEncoder
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_101_harm_obs_a_normfix"
CLAIM_IDS = ["SD-011", "ARC-016"]

HARM_OBS_DIM   = 51
HARM_OBS_A_DIM = 50
Z_HARM_DIM     = 32
Z_HARM_A_DIM   = 16


def _autocorr_lag(series: List[float], lag: int) -> float:
    """Pearson autocorrelation at a given lag."""
    if len(series) < lag + 2:
        return 0.0
    x = np.array(series[:-lag], dtype=np.float64)
    y = np.array(series[lag:],  dtype=np.float64)
    x -= x.mean(); y -= y.mean()
    denom = np.sqrt((x**2).sum() * (y**2).sum())
    if denom < 1e-12:
        return 0.0
    return float((x * y).sum() / denom)


def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _collect_raw_stats(
    seed: int,
    num_hazards: int,
    collect_steps: int,
    steps_per_episode: int,
    world_dim: int,
    self_dim: int,
    alpha_world: float,
    harm_scale: float,
    proximity_scale: float,
) -> Dict:
    """Collect raw harm_obs_a stats from env WITHOUT encoder -- no training."""
    torch.manual_seed(seed)
    random.seed(seed)
    label = f"HAZARDS_{num_hazards}"

    env = CausalGridWorldV2(
        seed=seed, size=10, num_hazards=num_hazards, num_resources=3,
        hazard_harm=harm_scale,
        env_drift_interval=5, env_drift_prob=0.1,
        proximity_harm_scale=proximity_scale,
        proximity_benefit_scale=proximity_scale * 0.5,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        use_proxy_fields=True,
    )

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=0.3,
        reafference_action_dim=0,
        use_harm_stream=True,
        harm_obs_dim=HARM_OBS_DIM,
        z_harm_dim=Z_HARM_DIM,
    )
    agent = REEAgent(config)
    num_actions = env.action_dim

    harm_obs_a_means: List[float] = []
    harm_obs_s_means: List[float] = []
    steps_collected = 0

    _, obs_dict = env.reset()
    agent.reset()

    while steps_collected < collect_steps:
        obs_body  = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            agent.sense(obs_body, obs_world)
            agent.clock.advance()

        harm_obs   = obs_dict.get("harm_obs",   torch.zeros(HARM_OBS_DIM)).float()
        harm_obs_a = obs_dict.get("harm_obs_a", torch.zeros(HARM_OBS_A_DIM)).float()

        harm_obs_a_means.append(float(harm_obs_a.mean().item()))
        harm_obs_s_means.append(float(harm_obs[:HARM_OBS_DIM].mean().item()))

        action = _action_to_onehot(
            random.randint(0, num_actions - 1), num_actions, agent.device)
        agent._last_action = action
        _, _, done, _, obs_dict = env.step(action)
        steps_collected += 1

        if done:
            _, obs_dict = env.reset()
            agent.reset()

    mean_a = float(np.mean(harm_obs_a_means))
    std_a  = float(np.std(harm_obs_a_means))
    ac_a   = _autocorr_lag(harm_obs_a_means, lag=10)
    mean_s = float(np.mean(harm_obs_s_means))
    ac_s   = _autocorr_lag(harm_obs_s_means, lag=10)

    print(
        f"  RAW [{label}]: harm_obs_a mean={mean_a:.4f} std={std_a:.4f}"
        f" autocorr_lag10={ac_a:.3f}",
        flush=True,
    )
    print(
        f"  RAW [{label}]: harm_obs_s mean={mean_s:.4f}"
        f" autocorr_lag10={ac_s:.3f}",
        flush=True,
    )

    return {
        "label":        label,
        "num_hazards":  num_hazards,
        "mean_harm_a":  mean_a,
        "std_harm_a":   std_a,
        "autocorr_a":   ac_a,
        "mean_harm_s":  mean_s,
        "autocorr_s":   ac_s,
        "n_steps":      steps_collected,
    }


def _collect_encoder_stats(
    seed: int,
    num_hazards: int,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    self_dim: int,
    lr: float,
    alpha_world: float,
    harm_scale: float,
    proximity_scale: float,
) -> Dict:
    """Train encoders then collect z_harm_a and z_harm_s encoder output stats."""
    torch.manual_seed(seed)
    random.seed(seed)
    label = f"HAZARDS_{num_hazards}"

    env = CausalGridWorldV2(
        seed=seed, size=10, num_hazards=num_hazards, num_resources=3,
        hazard_harm=harm_scale,
        env_drift_interval=5, env_drift_prob=0.1,
        proximity_harm_scale=proximity_scale,
        proximity_benefit_scale=proximity_scale * 0.5,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        use_proxy_fields=True,
    )

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=0.3,
        reafference_action_dim=0,
        use_harm_stream=True,
        harm_obs_dim=HARM_OBS_DIM,
        z_harm_dim=Z_HARM_DIM,
    )
    agent = REEAgent(config)
    harm_enc   = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_DIM)
    harm_enc_a = AffectiveHarmEncoder(
        harm_obs_a_dim=HARM_OBS_A_DIM, z_harm_a_dim=Z_HARM_A_DIM)
    num_actions = env.action_dim

    opt_harm = optim.Adam(
        list(harm_enc.parameters())
        + list(harm_enc_a.parameters())
        + list(agent.e3.harm_eval_z_harm_head.parameters()),
        lr=lr,
    )

    print(
        f"  [enc-train] {label} seed={seed} warmup={warmup_episodes} eps",
        flush=True,
    )
    agent.train(); harm_enc.train(); harm_enc_a.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            agent.sense(obs_body, obs_world)
            agent.clock.advance()

            harm_obs   = obs_dict.get("harm_obs",   torch.zeros(HARM_OBS_DIM)).float()
            harm_obs_a = obs_dict.get("harm_obs_a", torch.zeros(HARM_OBS_A_DIM)).float()

            z_harm_s = harm_enc(harm_obs.unsqueeze(0))
            label_s  = harm_obs[12].unsqueeze(0).unsqueeze(0)
            pred_s   = agent.e3.harm_eval_z_harm(z_harm_s)
            loss_s   = F.mse_loss(pred_s, label_s)

            z_harm_a  = harm_enc_a(harm_obs_a.unsqueeze(0))
            label_a   = harm_obs_a.mean().unsqueeze(0).unsqueeze(0)
            pred_a    = z_harm_a.mean(dim=-1, keepdim=True)
            loss_a    = F.mse_loss(pred_a, label_a)

            loss = loss_s + loss_a
            opt_harm.zero_grad(); loss.backward(); opt_harm.step()

            action = _action_to_onehot(
                random.randint(0, num_actions - 1), num_actions, agent.device)
            agent._last_action = action
            _, _, done, _, obs_dict = env.step(action)

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
            print(f"    ep {ep+1}/{warmup_episodes}", flush=True)

    print(
        f"  [enc-eval] {label} seed={seed} eval={eval_episodes} eps",
        flush=True,
    )
    agent.eval(); harm_enc.eval(); harm_enc_a.eval()

    z_harm_s_norms: List[float] = []
    z_harm_a_norms: List[float] = []

    for ep in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            with torch.no_grad():
                obs_body  = obs_dict["body_state"]
                obs_world = obs_dict["world_state"]
                agent.sense(obs_body, obs_world)
                agent.clock.advance()

                harm_obs   = obs_dict.get("harm_obs",   torch.zeros(HARM_OBS_DIM)).float()
                harm_obs_a = obs_dict.get("harm_obs_a", torch.zeros(HARM_OBS_A_DIM)).float()

                z_harm_s = harm_enc(harm_obs.unsqueeze(0))
                z_harm_a = harm_enc_a(harm_obs_a.unsqueeze(0))

            z_harm_s_norms.append(float(z_harm_s.norm().item()))
            z_harm_a_norms.append(float(z_harm_a.norm().item()))

            action = _action_to_onehot(
                random.randint(0, num_actions - 1), num_actions, agent.device)
            agent._last_action = action
            _, _, done, _, obs_dict = env.step(action)
            if done:
                break

    mean_s = float(np.mean(z_harm_s_norms)) if z_harm_s_norms else 0.0
    mean_a = float(np.mean(z_harm_a_norms)) if z_harm_a_norms else 0.0
    std_s  = float(np.std(z_harm_s_norms))  if z_harm_s_norms else 0.0
    std_a  = float(np.std(z_harm_a_norms))  if z_harm_a_norms else 0.0
    ac_s   = _autocorr_lag(z_harm_s_norms, lag=10)
    ac_a   = _autocorr_lag(z_harm_a_norms, lag=10)

    print(
        f"    {label}: z_harm_a mean={mean_a:.4f} std={std_a:.4f}"
        f" ac10={ac_a:.3f}",
        flush=True,
    )
    print(
        f"    {label}: z_harm_s mean={mean_s:.4f} std={std_s:.4f}"
        f" ac10={ac_s:.3f}",
        flush=True,
    )

    return {
        "label":         label,
        "num_hazards":   num_hazards,
        "mean_z_harm_a": mean_a,
        "std_z_harm_a":  std_a,
        "ac_z_harm_a":   ac_a,
        "mean_z_harm_s": mean_s,
        "std_z_harm_s":  std_s,
        "ac_z_harm_s":   ac_s,
    }


def run(
    seed: int = 42,
    raw_collect_steps: int = 5000,
    warmup_episodes: int = 300,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    world_dim: int = 32,
    self_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    harm_scale: float = 0.02,
    proximity_scale: float = 0.05,
    **kwargs,
) -> dict:
    fatal_error_count = 0

    # ---- Phase 0: Raw harm_obs_a stats (no encoder) ----
    print("\n[V3-EXQ-101] Phase 0: Raw harm_obs_a after normalization fix...",
          flush=True)
    print("  Collecting HIGH_HAZARD (6 hazards)...", flush=True)
    raw_high = _collect_raw_stats(
        seed=seed, num_hazards=6, collect_steps=raw_collect_steps,
        steps_per_episode=steps_per_episode, world_dim=world_dim, self_dim=self_dim,
        alpha_world=alpha_world, harm_scale=harm_scale, proximity_scale=proximity_scale,
    )

    print("  Collecting LOW_HAZARD (2 hazards)...", flush=True)
    raw_low = _collect_raw_stats(
        seed=seed, num_hazards=2, collect_steps=raw_collect_steps,
        steps_per_episode=steps_per_episode, world_dim=world_dim, self_dim=self_dim,
        alpha_world=alpha_world, harm_scale=harm_scale, proximity_scale=proximity_scale,
    )

    c1_raw = raw_high["mean_harm_a"] > raw_low["mean_harm_a"] * 1.1
    c2_raw = raw_high["autocorr_a"] > 0.3

    print(
        f"\n  C1 (raw density response): {'PASS' if c1_raw else 'FAIL'}"
        f" high={raw_high['mean_harm_a']:.4f} vs low={raw_low['mean_harm_a']:.4f}"
        f" (threshold: 1.1x)",
        flush=True,
    )
    print(
        f"  C2 (raw autocorr): {'PASS' if c2_raw else 'FAIL'}"
        f" autocorr_lag10={raw_high['autocorr_a']:.3f}"
        f" (threshold: > 0.3)",
        flush=True,
    )

    # ---- Phase 1: Encoder training and eval ----
    print("\n[V3-EXQ-101] Phase 1: Encoder training and eval...", flush=True)
    print("  Training HIGH_HAZARD condition...", flush=True)
    enc_high = _collect_encoder_stats(
        seed=seed, num_hazards=6,
        warmup_episodes=warmup_episodes, eval_episodes=eval_episodes,
        steps_per_episode=steps_per_episode, world_dim=world_dim, self_dim=self_dim,
        lr=lr, alpha_world=alpha_world, harm_scale=harm_scale,
        proximity_scale=proximity_scale,
    )

    print("  Training LOW_HAZARD condition...", flush=True)
    enc_low = _collect_encoder_stats(
        seed=seed, num_hazards=2,
        warmup_episodes=warmup_episodes, eval_episodes=eval_episodes,
        steps_per_episode=steps_per_episode, world_dim=world_dim, self_dim=self_dim,
        lr=lr, alpha_world=alpha_world, harm_scale=harm_scale,
        proximity_scale=proximity_scale,
    )

    c3_enc = enc_high["mean_z_harm_a"] > enc_low["mean_z_harm_a"] * 1.1
    c4_enc = enc_high["ac_z_harm_a"] > enc_high["ac_z_harm_s"]
    c5 = fatal_error_count == 0

    print(
        f"\n  C3 (encoder density): {'PASS' if c3_enc else 'FAIL'}"
        f" z_harm_a_high={enc_high['mean_z_harm_a']:.4f}"
        f" vs low={enc_low['mean_z_harm_a']:.4f}",
        flush=True,
    )
    print(
        f"  C4 (encoder autocorr): {'PASS' if c4_enc else 'FAIL'}"
        f" z_harm_a_ac={enc_high['ac_z_harm_a']:.3f}"
        f" vs z_harm_s_ac={enc_high['ac_z_harm_s']:.3f}",
        flush=True,
    )

    all_pass = c1_raw and c2_raw and c3_enc and c4_enc and c5
    n_met    = sum([c1_raw, c2_raw, c3_enc, c4_enc, c5])
    status   = "PASS" if all_pass else "FAIL"

    if not c1_raw or not c2_raw:
        diagnosis = "EMA_STILL_BROKEN"
    elif not c3_enc or not c4_enc:
        diagnosis = "ENCODER_FAIL"
    else:
        diagnosis = "PASS"

    print(f"\n[V3-EXQ-101] {status} ({n_met}/5)  Diagnosis: {diagnosis}", flush=True)

    failure_notes = []
    if not c1_raw:
        failure_notes.append(
            f"C1 FAIL: harm_obs_a still not responding to density after normfix"
            f" (high={raw_high['mean_harm_a']:.4f} vs low={raw_low['mean_harm_a']:.4f})")
    if not c2_raw:
        failure_notes.append(
            f"C2 FAIL: raw autocorr={raw_high['autocorr_a']:.3f}"
            " still < 0.3 after normfix")
    if not c3_enc:
        failure_notes.append(
            f"C3 FAIL: z_harm_a encoder not density-sensitive"
            f" (high={enc_high['mean_z_harm_a']:.4f} vs low={enc_low['mean_z_harm_a']:.4f})")
    if not c4_enc:
        failure_notes.append(
            f"C4 FAIL: z_harm_a autocorr={enc_high['ac_z_harm_a']:.3f}"
            f" not > z_harm_s autocorr={enc_high['ac_z_harm_s']:.3f}")

    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "raw_harm_a_mean_high":     float(raw_high["mean_harm_a"]),
        "raw_harm_a_mean_low":      float(raw_low["mean_harm_a"]),
        "raw_harm_a_autocorr_high": float(raw_high["autocorr_a"]),
        "raw_harm_s_autocorr_high": float(raw_high["autocorr_s"]),
        "enc_z_harm_a_mean_high":   float(enc_high["mean_z_harm_a"]),
        "enc_z_harm_a_mean_low":    float(enc_low["mean_z_harm_a"]),
        "enc_z_harm_a_ac_high":     float(enc_high["ac_z_harm_a"]),
        "enc_z_harm_s_ac_high":     float(enc_high["ac_z_harm_s"]),
        "crit1_raw_density":        1.0 if c1_raw else 0.0,
        "crit2_raw_autocorr":       1.0 if c2_raw else 0.0,
        "crit3_enc_density":        1.0 if c3_enc else 0.0,
        "crit4_enc_autocorr":       1.0 if c4_enc else 0.0,
        "crit5_no_errors":          1.0 if c5 else 0.0,
        "criteria_met":             float(n_met),
        "fatal_error_count":        float(fatal_error_count),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-101 -- SD-011 / ARC-016 harm_obs_a Normalization Fix Validation

**Status:** {status}
**Claims:** SD-011, ARC-016
**Supersedes:** V3-EXQ-100b (RAW_EMA_FAIL: hazard_max normalization bug)
**Fix:** Removed per-step division by hazard_field.max() in harm_obs_a EMA update.
         Now uses raw field values clipped to [0, 1].

## Diagnosis: {diagnosis}

## Phase 0: Raw harm_obs_a (no encoder) -- after normfix

| Metric | HIGH (6 hazards) | LOW (2 hazards) | Criterion |
|--------|-----------------|-----------------|-----------|
| harm_obs_a mean | {raw_high['mean_harm_a']:.4f} | {raw_low['mean_harm_a']:.4f} | high > low * 1.1 (C1) |
| harm_obs_a autocorr_lag10 | {raw_high['autocorr_a']:.3f} | {raw_low['autocorr_a']:.3f} | > 0.3 (C2) |
| harm_obs_s autocorr_lag10 | {raw_high['autocorr_s']:.3f} | -- | reference |

| Criterion | Result |
|-----------|--------|
| C1: raw density response | {"PASS" if c1_raw else "FAIL"} |
| C2: raw autocorr > 0.3 | {"PASS" if c2_raw else "FAIL"} |

## Phase 1: Encoder output (after training)

| Metric | HIGH (6 hazards) | LOW (2 hazards) | Criterion |
|--------|-----------------|-----------------|-----------|
| z_harm_a mean norm | {enc_high['mean_z_harm_a']:.4f} | {enc_low['mean_z_harm_a']:.4f} | high > low * 1.1 (C3) |
| z_harm_a autocorr_lag10 | {enc_high['ac_z_harm_a']:.3f} | -- | > z_harm_s (C4) |
| z_harm_s autocorr_lag10 | {enc_high['ac_z_harm_s']:.3f} | -- | reference |

| Criterion | Result |
|-----------|--------|
| C3: encoder density | {"PASS" if c3_enc else "FAIL"} |
| C4: encoder autocorr | {"PASS" if c4_enc else "FAIL"} |
| C5: no fatal errors | {"PASS" if c5 else "FAIL"} |

Criteria met: {n_met}/5 -> **{status}**
{failure_section}
"""

    return {
        "status":             status,
        "metrics":            metrics,
        "summary_markdown":   summary_markdown,
        "claim_ids":          CLAIM_IDS,
        "evidence_direction": "supports" if all_pass else (
            "mixed" if n_met >= 3 else "weakens"),
        "experiment_type":    EXPERIMENT_TYPE,
        "fatal_error_count":  fatal_error_count,
        "diagnosis":          diagnosis,
        "supersedes":         "v3_exq_100b_affective_harm_diagnostic",
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--raw-steps", type=int, default=5000)
    parser.add_argument("--warmup",    type=int, default=300)
    parser.add_argument("--eval",      type=int, default=50)
    parser.add_argument("--steps",     type=int, default=200)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        raw_collect_steps=args.raw_steps,
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
