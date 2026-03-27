#!/opt/local/bin/python3
"""
V3-EXQ-100 -- SD-011 / ARC-016 Affective Harm Stream First Integration Test

Claims: SD-011, ARC-016

This is the first experiment to wire z_harm_a (AffectiveHarmEncoder on harm_obs_a)
into an active experiment loop. AffectiveHarmEncoder, z_harm_a in LatentState, and
harm_obs_a EMA from CausalGridWorldV2 all exist but have never been tested as a system.

Background (SD-011 dual nociceptive streams):
  z_harm_s (sensory-discriminative, Adelta-pathway analog):
    - Encoded from harm_obs_s (current proximity fields, 51 dims)
    - High-frequency, step-to-step variation
    - Forward-predictable from action (the E2_harm_s target in ARC-033)
  z_harm_a (affective-motivational, C-fiber/paleospinothalamic analog):
    - Encoded from harm_obs_a (EMA of harm_obs_s, tau~20 steps, 50 dims)
    - Low-frequency, temporally integrated threat accumulation
    - NOT forward-predictable -- integrative by design
    - Should carry higher temporal autocorrelation than z_harm_s
    - ARC-016 reframe: accumulated affective threat (z_harm_a.norm) as urgency
      signal for commit threshold modulation

What this experiment tests:
  (1) Signal quality: does z_harm_a norm correlate with accumulated hazard exposure?
      Expected: near_hazard runs accumulate z_harm_a; far_from_hazard does not.
  (2) Temporal stability: is z_harm_a norm more temporally autocorrelated than z_harm_s?
      Expected: autocorr_lag10_z_harm_a > autocorr_lag10_z_harm_s (EMA adds smoothing).
  (3) Discriminability: does z_harm_a distinguish prolonged vs brief hazard exposure?
      HIGH_HAZARD (6 hazards) vs LOW_HAZARD (2 hazards) conditions.

Protocol:
  Two conditions: HIGH_HAZARD (6 hazards, high accumulation) vs LOW_HAZARD (2 hazards).
  Both: 300 warmup episodes (random + harm stream training).
  Eval: 50 episodes collecting z_harm_s and z_harm_a norms + autocorrelation.

PASS criteria (ALL required):
  C1: mean_z_harm_a_norm_high > mean_z_harm_a_norm_low * 1.2
      (z_harm_a responds to accumulated hazard exposure)
  C2: autocorr_lag10_z_harm_a > autocorr_lag10_z_harm_s
      (z_harm_a is more temporally stable than z_harm_s -- EMA adds smoothing)
  C3: z_harm_a_std_step < z_harm_s_std_step
      (z_harm_a norm varies less step-to-step)
  C4: no fatal errors
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.latent.stack import HarmEncoder, AffectiveHarmEncoder
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_100_z_harm_a_integration"
CLAIM_IDS = ["SD-011", "ARC-016"]

HARM_OBS_DIM   = 51
HARM_OBS_A_DIM = 50
Z_HARM_DIM     = 32
Z_HARM_A_DIM   = 16


def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _autocorr_lag(series: List[float], lag: int) -> float:
    """Pearson autocorrelation of a 1-D series at a given lag."""
    if len(series) < lag + 2:
        return 0.0
    x = np.array(series[:-lag], dtype=np.float64)
    y = np.array(series[lag:],  dtype=np.float64)
    x -= x.mean(); y -= y.mean()
    denom = np.sqrt((x**2).sum() * (y**2).sum())
    if denom < 1e-12:
        return 0.0
    return float((x * y).sum() / denom)


def _run_condition(
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
    """Run one condition and return diagnostic metrics."""
    torch.manual_seed(seed)
    random.seed(seed)

    cond_label = f"HAZARDS_{num_hazards}"

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

    # SD-010: sensory-discriminative harm stream
    harm_enc = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_DIM)
    # SD-011: affective-motivational harm stream (EMA-smoothed)
    harm_enc_a = AffectiveHarmEncoder(
        harm_obs_a_dim=HARM_OBS_A_DIM, z_harm_a_dim=Z_HARM_A_DIM)

    num_actions = env.action_dim

    opt_std  = optim.Adam(
        [p for n, p in agent.named_parameters() if "harm_eval" not in n],
        lr=lr,
    )
    opt_harm = optim.Adam(
        list(harm_enc.parameters())
        + list(harm_enc_a.parameters())
        + list(agent.e3.harm_eval_z_harm_head.parameters()),
        lr=1e-3,
    )

    # -- Warmup training --
    print(
        f"  [train] {cond_label} seed={seed}"
        f" warmup={warmup_episodes} eps ...",
        flush=True,
    )
    agent.train()
    harm_enc.train(); harm_enc_a.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent    = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            harm_obs   = obs_dict.get(
                "harm_obs",   torch.zeros(HARM_OBS_DIM)).float()
            harm_obs_a = obs_dict.get(
                "harm_obs_a", torch.zeros(HARM_OBS_A_DIM)).float()

            # Encode both streams (with grad for joint training)
            z_harm_s = harm_enc(harm_obs.unsqueeze(0))
            z_harm_a = harm_enc_a(harm_obs_a.unsqueeze(0))   # SD-011: first use!

            # Train harm head on proximity label (supervision for z_harm_s)
            label = harm_obs[12].unsqueeze(0).unsqueeze(0)
            pred  = agent.e3.harm_eval_z_harm(z_harm_s)
            loss_harm = F.mse_loss(pred, label)
            opt_harm.zero_grad(); loss_harm.backward(); opt_harm.step()

            action_idx = random.randint(0, num_actions - 1)
            action = _action_to_onehot(action_idx, num_actions, agent.device)
            agent._last_action = action

            _, _, done, _, obs_dict = env.step(action)

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total   = e1_loss + e2_loss
            if total.requires_grad:
                opt_std.zero_grad(); total.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt_std.step()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
            print(
                f"    ep {ep+1}/{warmup_episodes}",
                flush=True,
            )

    # -- Eval: collect harm stream statistics --
    print(
        f"  [eval] {cond_label} seed={seed} eval={eval_episodes} eps ...",
        flush=True,
    )
    agent.eval()
    harm_enc.eval(); harm_enc_a.eval()

    z_harm_s_norms: List[float] = []   # per-step z_harm_s norm
    z_harm_a_norms: List[float] = []   # per-step z_harm_a norm
    harm_event_count = 0

    for ep in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            with torch.no_grad():
                obs_body  = obs_dict["body_state"]
                obs_world = obs_dict["world_state"]
                _ = agent.sense(obs_body, obs_world)
                agent.clock.advance()

                harm_obs   = obs_dict.get(
                    "harm_obs",   torch.zeros(HARM_OBS_DIM)).float()
                harm_obs_a = obs_dict.get(
                    "harm_obs_a", torch.zeros(HARM_OBS_A_DIM)).float()

                z_harm_s = harm_enc(harm_obs.unsqueeze(0))
                z_harm_a = harm_enc_a(harm_obs_a.unsqueeze(0))

            z_harm_s_norms.append(float(z_harm_s.norm().item()))
            z_harm_a_norms.append(float(z_harm_a.norm().item()))

            action = _action_to_onehot(
                random.randint(0, num_actions - 1), num_actions, agent.device)
            agent._last_action = action
            _, _, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            if ttype in ("env_caused_hazard", "agent_caused_hazard",
                         "hazard_approach"):
                harm_event_count += 1
            if done:
                break

    mean_s = float(np.mean(z_harm_s_norms)) if z_harm_s_norms else 0.0
    mean_a = float(np.mean(z_harm_a_norms)) if z_harm_a_norms else 0.0
    std_s  = float(np.std(z_harm_s_norms))  if z_harm_s_norms else 0.0
    std_a  = float(np.std(z_harm_a_norms))  if z_harm_a_norms else 0.0
    ac_s   = _autocorr_lag(z_harm_s_norms, lag=10)
    ac_a   = _autocorr_lag(z_harm_a_norms, lag=10)

    print(
        f"    {cond_label}: z_harm_s mean={mean_s:.4f} std={std_s:.4f}"
        f" ac10={ac_s:.3f}",
        flush=True,
    )
    print(
        f"    {cond_label}: z_harm_a mean={mean_a:.4f} std={std_a:.4f}"
        f" ac10={ac_a:.3f}",
        flush=True,
    )
    print(
        f"    {cond_label}: harm_events={harm_event_count}",
        flush=True,
    )

    return {
        "condition":          cond_label,
        "num_hazards":        num_hazards,
        "mean_z_harm_s_norm": mean_s,
        "mean_z_harm_a_norm": mean_a,
        "std_z_harm_s_norm":  std_s,
        "std_z_harm_a_norm":  std_a,
        "autocorr_lag10_s":   ac_s,
        "autocorr_lag10_a":   ac_a,
        "harm_event_count":   harm_event_count,
        "n_steps_eval":       len(z_harm_s_norms),
    }


def run(
    seeds: tuple = (42,),
    warmup_episodes: int = 300,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    harm_scale: float = 0.02,
    proximity_scale: float = 0.05,
    **kwargs,
) -> dict:
    """HIGH_HAZARD (6) vs LOW_HAZARD (2) conditions: does z_harm_a respond to
    accumulated exposure? Is it more temporally stable than z_harm_s?"""

    results_high: List[Dict] = []
    results_low:  List[Dict] = []

    for seed in seeds:
        print(f"\n[V3-EXQ-100] HIGH_HAZARD seed={seed}", flush=True)
        r_high = _run_condition(
            seed=seed, num_hazards=6,
            warmup_episodes=warmup_episodes, eval_episodes=eval_episodes,
            steps_per_episode=steps_per_episode,
            world_dim=world_dim, self_dim=self_dim, lr=lr,
            alpha_world=alpha_world, harm_scale=harm_scale,
            proximity_scale=proximity_scale,
        )
        results_high.append(r_high)

        print(f"\n[V3-EXQ-100] LOW_HAZARD seed={seed}", flush=True)
        r_low = _run_condition(
            seed=seed, num_hazards=2,
            warmup_episodes=warmup_episodes, eval_episodes=eval_episodes,
            steps_per_episode=steps_per_episode,
            world_dim=world_dim, self_dim=self_dim, lr=lr,
            alpha_world=alpha_world, harm_scale=harm_scale,
            proximity_scale=proximity_scale,
        )
        results_low.append(r_low)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    mean_harm_a_high  = _avg(results_high, "mean_z_harm_a_norm")
    mean_harm_a_low   = _avg(results_low,  "mean_z_harm_a_norm")
    autocorr_a_high   = _avg(results_high, "autocorr_lag10_a")
    autocorr_s_high   = _avg(results_high, "autocorr_lag10_s")
    std_s_high        = _avg(results_high, "std_z_harm_s_norm")
    std_a_high        = _avg(results_high, "std_z_harm_a_norm")

    c1 = mean_harm_a_high > mean_harm_a_low * 1.2
    c2 = autocorr_a_high  > autocorr_s_high
    c3 = std_a_high       < std_s_high
    c4 = True   # no fatal errors

    all_pass = c1 and c2 and c3 and c4
    n_met    = sum([c1, c2, c3, c4])
    status   = "PASS" if all_pass else "FAIL"

    print(f"\n[V3-EXQ-100] Final results:", flush=True)
    print(
        f"  z_harm_a_norm high={mean_harm_a_high:.4f}"
        f"  low={mean_harm_a_low:.4f}"
        f"  ratio={mean_harm_a_high/max(1e-8,mean_harm_a_low):.2f}x",
        flush=True,
    )
    print(
        f"  autocorr_lag10  z_harm_a={autocorr_a_high:.3f}"
        f"  z_harm_s={autocorr_s_high:.3f}",
        flush=True,
    )
    print(
        f"  step_std  z_harm_a={std_a_high:.4f}"
        f"  z_harm_s={std_s_high:.4f}",
        flush=True,
    )
    print(f"  {status} ({n_met}/4)", flush=True)

    failure_notes = []
    if not c1:
        failure_notes.append(
            f"C1 FAIL: z_harm_a_high={mean_harm_a_high:.4f}"
            f" vs low={mean_harm_a_low:.4f}"
            " -- affective stream does not respond to hazard density")
    if not c2:
        failure_notes.append(
            f"C2 FAIL: autocorr_a={autocorr_a_high:.3f}"
            f" not > autocorr_s={autocorr_s_high:.3f}"
            " -- EMA not adding temporal stability")
    if not c3:
        failure_notes.append(
            f"C3 FAIL: std_a={std_a_high:.4f} not < std_s={std_s_high:.4f}"
            " -- z_harm_a not smoother than z_harm_s")

    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "mean_z_harm_a_norm_high":  float(mean_harm_a_high),
        "mean_z_harm_a_norm_low":   float(mean_harm_a_low),
        "harm_a_ratio":             float(
            mean_harm_a_high / max(1e-8, mean_harm_a_low)),
        "autocorr_lag10_z_harm_a":  float(autocorr_a_high),
        "autocorr_lag10_z_harm_s":  float(autocorr_s_high),
        "std_z_harm_a_high":        float(std_a_high),
        "std_z_harm_s_high":        float(std_s_high),
        "crit1_harm_a_responds":    1.0 if c1 else 0.0,
        "crit2_temporal_stability": 1.0 if c2 else 0.0,
        "crit3_smoother":           1.0 if c3 else 0.0,
        "crit4_no_errors":          1.0 if c4 else 0.0,
        "criteria_met":             float(n_met),
        "fatal_error_count":        0.0,
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-100 -- SD-011 / ARC-016 Affective Harm Stream First Integration

**Status:** {status}
**Claims:** SD-011, ARC-016
**Design:** HIGH_HAZARD (6) vs LOW_HAZARD (2) -- does z_harm_a respond and stabilise?
**First test:** AffectiveHarmEncoder wired into active experiment loop (never done before)

## Biological Grounding

z_harm_s (A-delta analog): immediate, high-frequency proximity. Step-to-step variation.
z_harm_a (C-fiber/paleospinothalamic analog): integrated, homeostatic accumulation.
  harm_obs_a is an EMA (tau~20 steps) of harm_obs_s proximity fields.
  AffectiveHarmEncoder maps this to z_harm_a [16 dims].
  ARC-016 reframe: z_harm_a.norm as urgency for dynamic commit threshold.

## Results

| Metric | HIGH (6 hazards) | LOW (2 hazards) |
|--------|-----------------|-----------------|
| z_harm_a mean norm | {mean_harm_a_high:.4f} | {mean_harm_a_low:.4f} |
| z_harm_a / z_harm_a ratio | {mean_harm_a_high/max(1e-8,mean_harm_a_low):.2f}x | -- |
| autocorr_lag10 z_harm_a | {autocorr_a_high:.3f} | -- |
| autocorr_lag10 z_harm_s | {autocorr_s_high:.3f} | -- |
| step-std z_harm_a | {std_a_high:.4f} | -- |
| step-std z_harm_s | {std_s_high:.4f} | -- |

## PASS Criteria

| Criterion | Result | Value |
|-----------|--------|-------|
| C1: z_harm_a responds to hazard density | {"PASS" if c1 else "FAIL"} | {mean_harm_a_high:.4f} vs {mean_harm_a_low:.4f} * 1.2 |
| C2: z_harm_a more autocorrelated than z_harm_s | {"PASS" if c2 else "FAIL"} | {autocorr_a_high:.3f} > {autocorr_s_high:.3f} |
| C3: z_harm_a step-variance < z_harm_s | {"PASS" if c3 else "FAIL"} | {std_a_high:.4f} < {std_s_high:.4f} |
| C4: no fatal errors | {"PASS" if c4 else "FAIL"} | -- |

Criteria met: {n_met}/4 -> **{status}**
{failure_section}
"""

    return {
        "status":             status,
        "metrics":            metrics,
        "summary_markdown":   summary_markdown,
        "claim_ids":          CLAIM_IDS,
        "evidence_direction": "supports" if all_pass else (
            "mixed" if n_met >= 2 else "weakens"),
        "experiment_type":    EXPERIMENT_TYPE,
        "fatal_error_count":  0,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",   type=int, nargs="+", default=[42])
    parser.add_argument("--warmup",  type=int, default=300)
    parser.add_argument("--eval",    type=int, default=50)
    parser.add_argument("--steps",   type=int, default=200)
    parser.add_argument("--world-dim", type=int, default=32)
    args = parser.parse_args()

    result = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        eval_episodes=args.eval,
        steps_per_episode=args.steps,
        world_dim=args.world_dim,
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
