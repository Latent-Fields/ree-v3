"""
V3-EXQ-019 — MECH-058 V3 Timescale Separation

Claims: MECH-058 (latent_stack.e1_e2_timescale_separation).

Motivation (2026-03-17):
  MECH-058 was a V2 FAIL (EXQ-023) because z_gamma conflated proprioceptive
  (self) and exteroceptive (world) channels, making E1/E2 timescale separation
  unmeasurable. V3's SD-005 split gives us z_self (E2 domain: fast motor-sensory)
  and z_world (E1/E3 domain: slow world model), enabling a direct test.

  The prediction:
    E2 (z_self domain): step-local motor-sensory prediction. Body changes every
    step (position, momentum). z_self should have SHORT autocorrelation — fast
    recovery, low persistence.

    E1 (z_world domain): slow world model. World state is stable across many
    steps (hazards don't move often, world content persists). z_world should
    have LONG autocorrelation — high persistence.

  BUT: before reafference correction (SD-007), z_world changes EVERY step due
  to perspective shift from locomotion. This makes z_world look "fast" when
  it should look "slow". This experiment measures:
    Phase A (no correction): z_self and z_world autocorrelation → z_world
      will likely look fast (EXQ-013 expected result: z_world_delta large)
    Phase B (with correction): after applying ReafferencePredictor, z_world_corrected
      autocorrelation → should be longer (SD-007 restore slow timescale)

  The MECH-058 V3 claim is: z_world has longer autocorrelation than z_self,
  AFTER reafference correction. Before correction, the claim should FAIL due
  to perspective shift contamination.

Metrics:
  Per step: Δz_self = ||z_self_t - z_self_{t-1}||
           Δz_world_raw = ||z_world_t - z_world_{t-1}||
           Δz_world_corr = ||z_world_corr_t - z_world_corr_{t-1}||
  Autocorrelation of delta series at lag 1, 2, 5, 10

PASS criteria (ALL must hold):
  C1 (pre-correction, confirming the problem):
     mean(Δz_world_raw) > mean(Δz_self) -- world looks fast (perspective shift)
  C2 (post-correction, validating SD-007 restores correct timescale):
     autocorr(Δz_world_corrected, lag=5) > autocorr(Δz_self, lag=5) -- world now slower
  C3: mean(Δz_world_raw) > mean(Δz_world_corrected) -- correction reduced world churn
  C4: n_steps_collected >= 3000
  C5: No fatal errors

Note: C1 expected to PASS (confirming SD-005 still contaminated by perspective shift).
      C2 expected to be the key discriminator for SD-007 effect.
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
from ree_core.predictors.e2_fast import ReafferencePredictor


EXPERIMENT_TYPE = "v3_exq_019_timescale_v3"
CLAIM_IDS = ["MECH-058"]


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _make_world_decoder(world_dim: int, world_obs_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(world_dim, 64), nn.ReLU(), nn.Linear(64, world_obs_dim)
    )


def _compute_autocorr(series: List[float], lag: int) -> float:
    """Lag-k autocorrelation of a list of scalars."""
    if len(series) < lag + 2:
        return 0.0
    x = torch.tensor(series, dtype=torch.float32)
    x_mean = x.mean()
    x_c = x - x_mean
    var = (x_c ** 2).mean()
    if var < 1e-8:
        return 0.0
    cov = (x_c[:-lag] * x_c[lag:]).mean()
    return float((cov / var).item())


def _train_and_collect(
    agent: REEAgent,
    env: CausalGridWorld,
    world_decoder: nn.Module,
    optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict:
    """Train agent and collect per-step delta sequences."""
    agent.train()
    world_decoder.train()

    # Delta sequences
    dz_self_series: List[float] = []
    dz_world_raw_series: List[float] = []
    reaf_data: List[Tuple] = []  # (z_self_prev, a_prev, dz_world_vec)
    MAX_REAF_DATA = 3000

    total_harm = 0
    traj_buffer: List = []
    MAX_TRAJ_BUFFER = 200

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        z_self_prev = None
        z_world_prev = None
        a_prev = None

        for step in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_self_curr = latent.z_self.detach()
            z_world_curr = latent.z_world.detach()

            # Compute deltas
            if z_self_prev is not None and z_world_prev is not None:
                dz_self = float(torch.norm(z_self_curr - z_self_prev).item())
                dz_world = float(torch.norm(z_world_curr - z_world_prev).item())
                dz_self_series.append(dz_self)
                dz_world_raw_series.append(dz_world)

                # Collect reafference data
                if a_prev is not None:
                    dz_world_vec = (z_world_curr - z_world_prev)
                    reaf_data.append((z_self_prev.cpu(), a_prev.cpu(), dz_world_vec.cpu()))
                    if len(reaf_data) > MAX_REAF_DATA:
                        reaf_data = reaf_data[-MAX_REAF_DATA:]

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            traj_buffer.append((latent.z_world.detach(), action.detach()))
            if len(traj_buffer) > MAX_TRAJ_BUFFER:
                traj_buffer = traj_buffer[-MAX_TRAJ_BUFFER:]

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            if harm_signal < 0:
                total_harm += 1

            e1_loss = agent.compute_prediction_loss()
            e2_self_loss = agent.compute_e2_loss()

            obs_w = obs_world.unsqueeze(0) if obs_world.dim() == 1 else obs_world
            z_w = agent.latent_stack.split_encoder.world_encoder(obs_w)
            recon = world_decoder(z_w)
            recon_loss = F.mse_loss(recon, obs_w)

            total_loss = e1_loss + e2_self_loss + recon_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            z_self_prev = z_self_curr
            z_world_prev = z_world_curr
            a_prev = action.detach()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == num_episodes - 1:
            print(f"  [train] ep {ep+1}/{num_episodes}  "
                  f"n_deltas={len(dz_self_series)}  harm={total_harm}", flush=True)

    return {
        "dz_self_series": dz_self_series,
        "dz_world_raw_series": dz_world_raw_series,
        "reaf_data": reaf_data,
        "total_harm": total_harm,
    }


def _collect_eval_deltas(
    agent: REEAgent,
    reaf_predictor: ReafferencePredictor,
    env: CausalGridWorld,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict:
    """Collect delta sequences with and without reafference correction."""
    agent.eval()
    reaf_predictor.eval()

    dz_self: List[float] = []
    dz_world_raw: List[float] = []
    dz_world_corr: List[float] = []
    dz_world_by_type: Dict[str, List[float]] = {
        "none": [], "env_caused_hazard": [], "agent_caused_hazard": []
    }

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        z_self_prev = None
        z_world_prev = None
        z_world_corr_prev = None
        a_prev = None
        prev_ttype = None

        for step in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()

                z_self_curr = latent.z_self.detach()
                z_world_curr = latent.z_world.detach()

                # Compute corrected z_world
                if a_prev is not None:
                    z_world_corr_curr = reaf_predictor.correct_z_world(
                        z_world_curr, z_self_curr, a_prev
                    )
                else:
                    z_world_corr_curr = z_world_curr.clone()

                # Record deltas (prev_ttype is transition type that produced current state)
                if z_self_prev is not None:
                    dz_s = float(torch.norm(z_self_curr - z_self_prev).item())
                    dz_wr = float(torch.norm(z_world_curr - z_world_prev).item())
                    dz_wc = float(torch.norm(z_world_corr_curr - z_world_corr_prev).item())

                    dz_self.append(dz_s)
                    dz_world_raw.append(dz_wr)
                    dz_world_corr.append(dz_wc)

                    if prev_ttype is not None and prev_ttype in dz_world_by_type:
                        dz_world_by_type[prev_ttype].append(dz_wr)

                action_idx = random.randint(0, env.action_dim - 1)
                action = _action_to_onehot(action_idx, env.action_dim, agent.device)
                agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            prev_ttype = info.get("transition_type", "none")

            z_self_prev = z_self_curr
            z_world_prev = z_world_curr
            z_world_corr_prev = z_world_corr_curr
            a_prev = action.detach()

            if done:
                break

    return {
        "dz_self": dz_self,
        "dz_world_raw": dz_world_raw,
        "dz_world_corrected": dz_world_corr,
        "dz_world_by_type": dz_world_by_type,
    }


def _train_reafference_predictor(
    reaf_data: List,
    self_dim: int,
    action_dim: int,
    world_dim: int,
) -> Tuple[ReafferencePredictor, float]:
    if len(reaf_data) < 20:
        return ReafferencePredictor(self_dim, action_dim, world_dim), 0.0

    n = len(reaf_data)
    n_train = int(n * 0.8)

    z_self_all = torch.cat([d[0] for d in reaf_data], dim=0)
    a_all = torch.cat([d[1] for d in reaf_data], dim=0)
    dz_all = torch.cat([d[2] for d in reaf_data], dim=0)

    rp = ReafferencePredictor(self_dim, action_dim, world_dim)
    rp_opt = optim.Adam(rp.parameters(), lr=1e-3)

    BATCH = 32
    for _ in range(200):
        if n_train < BATCH:
            break
        idxs = torch.randperm(n_train)[:BATCH]
        loss = F.mse_loss(rp(z_self_all[idxs], a_all[idxs]), dz_all[idxs])
        rp_opt.zero_grad()
        loss.backward()
        rp_opt.step()

    rp.eval()
    with torch.no_grad():
        z_st = z_self_all[n_train:]
        a_t = a_all[n_train:]
        dz_t = dz_all[n_train:]
        if len(z_st) > 0:
            pred = rp(z_st, a_t)
            ss_res = ((dz_t - pred) ** 2).sum()
            ss_tot = ((dz_t - dz_t.mean(dim=0, keepdim=True)) ** 2).sum()
            r2 = float((1 - ss_res / (ss_tot + 1e-8)).item())
        else:
            r2 = 0.0

    print(f"  ReafferencePredictor: n_train={n_train}  R²_test={r2:.3f}", flush=True)
    return rp, r2


def run(
    seed: int = 0,
    warmup_episodes: int = 1000,
    eval_episodes: int = 100,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    harm_scale: float = 0.02,
    alpha_world: float = 0.9,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorld(
        seed=seed, size=12, num_hazards=15, num_resources=5,
        env_drift_interval=3, env_drift_prob=0.5,
        hazard_harm=harm_scale, contaminated_harm=harm_scale,
    )
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
    )
    agent = REEAgent(config)
    world_decoder = _make_world_decoder(world_dim, env.world_obs_dim)
    optimizer = optim.Adam(list(agent.parameters()) + list(world_decoder.parameters()), lr=lr)

    print(f"[V3-EXQ-019] Warmup: {warmup_episodes} eps", flush=True)
    train_out = _train_and_collect(
        agent, env, world_decoder, optimizer, warmup_episodes, steps_per_episode
    )

    dz_self_train = train_out["dz_self_series"]
    dz_world_raw_train = train_out["dz_world_raw_series"]
    reaf_data = train_out["reaf_data"]

    print(f"  Warmup done. n_deltas={len(dz_self_train)}  reaf_data={len(reaf_data)}",
          flush=True)

    # Train reafference predictor
    print(f"[V3-EXQ-019] Training ReafferencePredictor...", flush=True)
    reaf_predictor, reaf_r2 = _train_reafference_predictor(
        reaf_data, self_dim, env.action_dim, world_dim
    )

    # Eval phase: collect delta sequences before and after correction
    print(f"[V3-EXQ-019] Eval phase: {eval_episodes} eps...", flush=True)
    eval_out = _collect_eval_deltas(agent, reaf_predictor, env, eval_episodes, steps_per_episode)

    dz_self = eval_out["dz_self"]
    dz_raw = eval_out["dz_world_raw"]
    dz_corr = eval_out["dz_world_corrected"]
    n_steps = len(dz_self)

    # Compute means
    mean_dz_self = float(sum(dz_self) / max(1, len(dz_self)))
    mean_dz_raw = float(sum(dz_raw) / max(1, len(dz_raw)))
    mean_dz_corr = float(sum(dz_corr) / max(1, len(dz_corr)))

    # Compute autocorrelations at multiple lags
    lags = [1, 2, 5, 10]
    autocorr_self = {lag: _compute_autocorr(dz_self, lag) for lag in lags}
    autocorr_raw = {lag: _compute_autocorr(dz_raw, lag) for lag in lags}
    autocorr_corr = {lag: _compute_autocorr(dz_corr, lag) for lag in lags}

    print(f"  n_steps={n_steps}", flush=True)
    print(f"  mean_dz: self={mean_dz_self:.4f}  raw={mean_dz_raw:.4f}  "
          f"corrected={mean_dz_corr:.4f}", flush=True)
    print(f"  autocorr(lag=5): z_self={autocorr_self[5]:.4f}  "
          f"z_world_raw={autocorr_raw[5]:.4f}  z_world_corr={autocorr_corr[5]:.4f}",
          flush=True)

    # Event-type delta statistics
    dz_empty = eval_out["dz_world_by_type"].get("none", [])
    dz_env_hazard = eval_out["dz_world_by_type"].get("env_caused_hazard", [])
    mean_dz_empty = float(sum(dz_empty) / max(1, len(dz_empty)))
    mean_dz_env = float(sum(dz_env_hazard) / max(1, len(dz_env_hazard)))

    # ── PASS / FAIL ──────────────────────────────────────────────────────────
    # C1: z_world_raw > z_self (perspective shift makes world look fast) [confirms problem]
    c1_pass = mean_dz_raw > mean_dz_self

    # C2: z_world_corrected has longer autocorrelation than z_self [validates SD-007]
    c2_pass = autocorr_corr[5] > autocorr_self[5]

    # C3: correction reduced world churn
    c3_pass = mean_dz_corr < mean_dz_raw

    # C4: sufficient data
    c4_pass = n_steps >= 3000

    # C5: no fatal errors
    c5_pass = True

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: mean_dz_world_raw={mean_dz_raw:.4f} <= mean_dz_self={mean_dz_self:.4f} "
            f"(perspective shift not dominating — unexpected)"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: autocorr_corr[5]={autocorr_corr[5]:.4f} <= "
            f"autocorr_self[5]={autocorr_self[5]:.4f} "
            f"(correction did not restore slow-world timescale)"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: mean_dz_corr={mean_dz_corr:.4f} >= mean_dz_raw={mean_dz_raw:.4f}"
        )
    if not c4_pass:
        failure_notes.append(f"C4 FAIL: n_steps={n_steps} < 3000")

    print(f"\nV3-EXQ-019 verdict: {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "fatal_error_count": 0.0,
        "n_steps_collected": float(n_steps),
        "reafference_r2_test": float(reaf_r2),
        # Means
        "mean_dz_self": float(mean_dz_self),
        "mean_dz_world_raw": float(mean_dz_raw),
        "mean_dz_world_corrected": float(mean_dz_corr),
        "dz_correction_reduction": float(mean_dz_raw - mean_dz_corr),
        "dz_world_raw_self_ratio": float(mean_dz_raw / max(1e-8, mean_dz_self)),
        # Event-type deltas
        "mean_dz_world_raw_empty": float(mean_dz_empty),
        "mean_dz_world_raw_env_hazard": float(mean_dz_env),
        "n_empty_steps": float(len(dz_empty)),
        "n_env_hazard_steps": float(len(dz_env_hazard)),
        # Autocorrelation at key lags
        "autocorr_dz_self_lag1": float(autocorr_self[1]),
        "autocorr_dz_self_lag5": float(autocorr_self[5]),
        "autocorr_dz_self_lag10": float(autocorr_self[10]),
        "autocorr_dz_world_raw_lag1": float(autocorr_raw[1]),
        "autocorr_dz_world_raw_lag5": float(autocorr_raw[5]),
        "autocorr_dz_world_raw_lag10": float(autocorr_raw[10]),
        "autocorr_dz_world_corr_lag1": float(autocorr_corr[1]),
        "autocorr_dz_world_corr_lag5": float(autocorr_corr[5]),
        "autocorr_dz_world_corr_lag10": float(autocorr_corr[10]),
        "autocorr_corr_minus_self_lag5": float(autocorr_corr[5] - autocorr_self[5]),
        # Criteria
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-019 — MECH-058 V3 Timescale Separation

**Status:** {status}
**Warmup:** {warmup_episodes} eps (RANDOM policy, 12×12, 15 hazards, drift_interval=3, drift_prob=0.5)
**Eval:** {eval_episodes} eps
**Seed:** {seed}

## Motivation (MECH-058 — V2 FAIL EXQ-023)

V2 FAIL was because z_gamma conflated self and world channels. V3 has clean
z_self/z_world split (SD-005). MECH-058 predicts z_world has longer autocorrelation
than z_self (persistent world model vs fast motor-sensory). BUT: before reafference
correction (SD-007), perspective shift makes z_world look fast. This experiment
measures both before and after correction.

## Delta Statistics

| Channel | Mean Δ | Autocorr lag=1 | Autocorr lag=5 | Autocorr lag=10 |
|---|---|---|---|---|
| z_self | {mean_dz_self:.4f} | {autocorr_self[1]:.4f} | {autocorr_self[5]:.4f} | {autocorr_self[10]:.4f} |
| z_world raw | {mean_dz_raw:.4f} | {autocorr_raw[1]:.4f} | {autocorr_raw[5]:.4f} | {autocorr_raw[10]:.4f} |
| z_world corrected | {mean_dz_corr:.4f} | {autocorr_corr[1]:.4f} | {autocorr_corr[5]:.4f} | {autocorr_corr[10]:.4f} |

Autocorr(corrected, lag=5) - Autocorr(self, lag=5) = {autocorr_corr[5] - autocorr_self[5]:.4f}

Event-type z_world_raw means: empty={mean_dz_empty:.4f}  env_hazard={mean_dz_env:.4f}
n_steps = {n_steps}  |  ReafferencePredictor R²_test = {reaf_r2:.3f}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: Δz_world_raw > Δz_self (perspective shift dominates) | {"PASS" if c1_pass else "FAIL"} | {mean_dz_raw:.4f} vs {mean_dz_self:.4f} |
| C2: autocorr_corr[5] > autocorr_self[5] (corrected world slower) | {"PASS" if c2_pass else "FAIL"} | {autocorr_corr[5]:.4f} vs {autocorr_self[5]:.4f} |
| C3: Δz_world_corr < Δz_world_raw (correction reduced churn) | {"PASS" if c3_pass else "FAIL"} | {mean_dz_corr:.4f} vs {mean_dz_raw:.4f} |
| C4: n_steps >= 3000 | {"PASS" if c4_pass else "FAIL"} | {n_steps} |
| C5: No fatal errors | {"PASS" if c5_pass else "FAIL"} | 0 |

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
        "fatal_error_count": 0,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",          type=int,   default=0)
    parser.add_argument("--warmup",        type=int,   default=1000)
    parser.add_argument("--eval-episodes", type=int,   default=100)
    parser.add_argument("--steps",         type=int,   default=200)
    parser.add_argument("--harm-scale",    type=float, default=0.02)
    parser.add_argument("--alpha-world",   type=float, default=0.9)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_episodes,
        steps_per_episode=args.steps,
        harm_scale=args.harm_scale,
        alpha_world=args.alpha_world,
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
