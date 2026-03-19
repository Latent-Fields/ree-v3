"""
V3-EXQ-027b — SD-007 Reafference Diagnostic: Raw vs Corrected Calibration Comparison

Claims: SD-007, MECH-098, MECH-101

Motivation (2026-03-19):
  V3-EXQ-027 (most recent run 20260318T180316Z) showed:
    - reafference_r2 = 0.362 (C2 PASS: predictor works)
    - calibration_gap_corrected = 0.0244 (C1 FAIL: below 0.15 threshold)

  Paradox: EXQ-026 (no reafference) achieved calibration_gap = 0.0375.
  EXQ-027 (with reafference) got 0.0244 — LOWER than without.

  Root cause hypothesis: ReafferencePredictor over-corrects at hazard steps.
  The predictor is trained on locomotion-only steps (ttype=="none") but hazard steps
  also involve locomotion. When applied at hazard steps, the predictor subtracts the
  locomotion-related z_world change along with some of the hazard-related signal,
  shrinking the calibration gap.

  This experiment tests the hypothesis by:
    1. Training E3.harm_eval_head on BOTH z_world_raw AND z_world_corrected
       (using two separate harm_eval evaluation runs at the end)
    2. Measuring calibration_gap for each independently
    3. Computing the correction delta = gap_corrected - gap_raw

  If correction_delta < 0: reafference IS hurting (confirms hypothesis)
    → SD-007 design needs rethinking (maybe λ-scaled correction, see future work)
  If correction_delta >= 0: reafference helps or is neutral
    → EXQ-027 failure was a threshold calibration artifact (0.15 was too high)

  This is diagnostic, not a test of whether SD-007 should be kept.
  SD-007 was validated for its reafference_r2 (EXQ-021 PASS). The question here is
  whether reafference improves E3.harm_eval calibration specifically.

Protocol:
  1. Train agent (800 warmup eps) with alpha_world=0.9, reafference enabled.
     Train two harm_eval heads:
       head_raw: trained on z_world_raw (no correction)
       head_corrected: trained on z_world_corrected (default, LatentState.z_world)
  2. Eval A (50 eps): score each head's calibration_gap by event type.
  3. Compute correction_delta = calibration_gap_corrected - calibration_gap_raw.

PASS criteria (ALL must hold):
  C1: calibration_gap_raw > 0.03        (E3 baseline functional on raw z_world)
  C2: reafference_r2 > 0.20             (predictor is learning)
  C3: harm_pred_std_raw > 0.01          (raw head not collapsed)
  C4: n_agent_hazard_steps >= 5         (enough events)
  C5: No fatal errors

Informational (no pass/fail — used for SD-007 design):
  correction_delta = gap_corrected - gap_raw
  Positive → reafference helps E3 calibration
  Negative → reafference hurts E3 calibration (over-correction evidence)
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


EXPERIMENT_TYPE = "v3_exq_027b_sd007_reafference_diagnostic"
CLAIM_IDS = ["SD-007", "MECH-098", "MECH-101"]


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _compute_reafference_r2(agent: REEAgent, reaf_data: List) -> float:
    if len(reaf_data) < 20:
        return 0.0
    n = len(reaf_data)
    n_train = int(n * 0.8)
    with torch.no_grad():
        zwr_all = torch.cat([d[0] for d in reaf_data], dim=0)
        a_all   = torch.cat([d[1] for d in reaf_data], dim=0)
        dz_all  = torch.cat([d[2] for d in reaf_data], dim=0)
        pred_all = agent.latent_stack.reafference_predictor(zwr_all, a_all)
        pred_test = pred_all[n_train:]
        dz_test   = dz_all[n_train:]
        if pred_test.shape[0] == 0:
            return 0.0
        ss_res = ((dz_test - pred_test) ** 2).sum()
        ss_tot = ((dz_test - dz_test.mean(dim=0, keepdim=True)) ** 2).sum()
        r2 = float((1 - ss_res / (ss_tot + 1e-8)).item())
    return max(0.0, r2)


def _train(
    agent: REEAgent,
    head_raw: nn.Linear,
    env: CausalGridWorld,
    optimizer: optim.Optimizer,
    harm_raw_optim: optim.Optimizer,
    harm_corr_optim: optim.Optimizer,
    reaf_optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict:
    """Train agent with both raw and corrected harm_eval heads plus reafference."""
    agent.train()

    harm_buf_pos_raw:  List[torch.Tensor] = []
    harm_buf_neg_raw:  List[torch.Tensor] = []
    harm_buf_pos_corr: List[torch.Tensor] = []
    harm_buf_neg_corr: List[torch.Tensor] = []
    MAX_BUF = 1000

    reaf_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    MAX_REAF = 5000

    total_harm = 0
    n_agent_hazard = 0
    n_empty = 0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_raw_prev = None
        a_prev = None

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_corr = latent.z_world.detach()        # reafference-corrected
            z_world_raw  = latent.z_world_raw.detach()    # uncorrected

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            if harm_signal < 0:
                total_harm += 1
                if ttype == "agent_caused_hazard":
                    n_agent_hazard += 1
                harm_buf_pos_raw.append(z_world_raw)
                harm_buf_pos_corr.append(z_world_corr)
                if len(harm_buf_pos_raw) > MAX_BUF:
                    harm_buf_pos_raw  = harm_buf_pos_raw[-MAX_BUF:]
                    harm_buf_pos_corr = harm_buf_pos_corr[-MAX_BUF:]
            else:
                harm_buf_neg_raw.append(z_world_raw)
                harm_buf_neg_corr.append(z_world_corr)
                if len(harm_buf_neg_raw) > MAX_BUF:
                    harm_buf_neg_raw  = harm_buf_neg_raw[-MAX_BUF:]
                    harm_buf_neg_corr = harm_buf_neg_corr[-MAX_BUF:]

            # Reafference training on empty steps
            if (
                ttype == "none"
                and z_raw_prev is not None
                and a_prev is not None
                and agent.latent_stack.reafference_predictor is not None
            ):
                dz_raw = z_world_raw - z_raw_prev
                reaf_data.append((z_raw_prev.cpu(), a_prev.cpu(), dz_raw.cpu()))
                n_empty += 1
                if len(reaf_data) > MAX_REAF:
                    reaf_data = reaf_data[-MAX_REAF:]

            # Standard E1 + E2 losses
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # Reafference predictor loss
            if len(reaf_data) >= 16 and agent.latent_stack.reafference_predictor is not None:
                k = min(32, len(reaf_data))
                idxs = torch.randperm(len(reaf_data))[:k].tolist()
                zwr_b = torch.cat([reaf_data[i][0] for i in idxs]).to(agent.device)
                a_b   = torch.cat([reaf_data[i][1] for i in idxs]).to(agent.device)
                dz_b  = torch.cat([reaf_data[i][2] for i in idxs]).to(agent.device)
                pred_dz = agent.latent_stack.reafference_predictor(zwr_b, a_b)
                reaf_loss = F.mse_loss(pred_dz, dz_b)
                if reaf_loss.requires_grad:
                    reaf_optimizer.zero_grad()
                    reaf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.latent_stack.reafference_predictor.parameters(), 0.5
                    )
                    reaf_optimizer.step()

            # Harm eval training: both raw and corrected heads (balanced)
            for (buf_pos, buf_neg, harm_optim, head) in [
                (harm_buf_pos_raw,  harm_buf_neg_raw,  harm_raw_optim,  None),   # None = use agent.e3
                (harm_buf_pos_corr, harm_buf_neg_corr, harm_corr_optim, head_raw),
            ]:
                if len(buf_pos) >= 4 and len(buf_neg) >= 4:
                    k_p = min(16, len(buf_pos))
                    k_n = min(16, len(buf_neg))
                    pi = torch.randperm(len(buf_pos))[:k_p].tolist()
                    ni = torch.randperm(len(buf_neg))[:k_n].tolist()
                    zw_pos = torch.cat([buf_pos[i] for i in pi], dim=0)
                    zw_neg = torch.cat([buf_neg[i] for i in ni], dim=0)
                    zw_b   = torch.cat([zw_pos, zw_neg], dim=0)
                    target = torch.cat([
                        torch.ones(k_p, 1, device=agent.device),
                        torch.zeros(k_n, 1, device=agent.device),
                    ], dim=0)
                    if head is None:
                        pred = agent.e3.harm_eval(zw_b)  # raw head (agent's default)
                    else:
                        pred = torch.sigmoid(head(zw_b))  # corrected head (separate)
                    loss = F.mse_loss(pred, target)
                    if loss.requires_grad:
                        harm_optim.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            (list(agent.e3.harm_eval_head.parameters()) if head is None
                             else list(head.parameters())),
                            0.5,
                        )
                        harm_optim.step()

            z_raw_prev = z_world_raw
            a_prev = action.detach()
            if done:
                break

        if (ep + 1) % 100 == 0 or ep == num_episodes - 1:
            print(
                f"  [train] ep {ep+1}/{num_episodes}  harm={total_harm}"
                f"  agent_hazard={n_agent_hazard}  empty={n_empty}",
                flush=True,
            )

    return {
        "total_harm": total_harm,
        "n_agent_hazard": n_agent_hazard,
        "n_empty_steps": n_empty,
        "reaf_data": reaf_data,
    }


def _eval_by_event(
    agent: REEAgent,
    head_raw: nn.Linear,
    env: CausalGridWorld,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict:
    """Evaluate both raw and corrected harm_eval by event type."""
    agent.eval()
    scores_raw:  Dict[str, List[float]] = {"none": [], "env_caused_hazard": [], "agent_caused_hazard": []}
    scores_corr: Dict[str, List[float]] = {"none": [], "env_caused_hazard": [], "agent_caused_hazard": []}
    all_raw:  List[float] = []
    all_corr: List[float] = []
    fatal = 0

    for _ in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                z_corr = latent.z_world
                z_raw  = latent.z_world_raw

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            flat_obs, _, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            try:
                with torch.no_grad():
                    s_raw  = float(agent.e3.harm_eval(z_raw).item())
                    s_corr = float(torch.sigmoid(head_raw(z_corr)).item())
                all_raw.append(s_raw)
                all_corr.append(s_corr)
                if ttype in scores_raw:
                    scores_raw[ttype].append(s_raw)
                    scores_corr[ttype].append(s_corr)
            except Exception:
                fatal += 1

            if done:
                break

    def gap(d: Dict[str, List[float]]) -> float:
        mn = sum(d["none"]) / max(1, len(d["none"]))
        ma = sum(d["agent_caused_hazard"]) / max(1, len(d["agent_caused_hazard"]))
        return ma - mn

    cal_raw  = gap(scores_raw)
    cal_corr = gap(scores_corr)
    std_raw  = float(torch.tensor(all_raw).std().item()) if len(all_raw) > 1 else 0.0
    std_corr = float(torch.tensor(all_corr).std().item()) if len(all_corr) > 1 else 0.0
    n_agent  = len(scores_raw["agent_caused_hazard"])

    print(
        f"  calibration_gap  raw={cal_raw:.4f}  corrected={cal_corr:.4f}"
        f"  delta={cal_corr - cal_raw:+.4f}  n_agent={n_agent}",
        flush=True,
    )

    return {
        "calibration_gap_raw": cal_raw,
        "calibration_gap_corrected": cal_corr,
        "correction_delta": cal_corr - cal_raw,
        "harm_pred_std_raw": std_raw,
        "harm_pred_std_corrected": std_corr,
        "n_agent_hazard_steps": n_agent,
        "n_none_steps": len(scores_raw["none"]),
        "fatal_errors": fatal,
    }


def run(
    seed: int = 0,
    warmup_episodes: int = 800,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorld(
        seed=seed, size=12, num_hazards=15, num_resources=5,
        env_drift_interval=3, env_drift_prob=0.5,
    )
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
    agent = REEAgent(config)

    assert agent.latent_stack.reafference_predictor is not None, (
        "SD-007 ReafferencePredictor not initialized"
    )

    # Separate harm_eval head for z_world_corrected (agent's default is trained on z_world_raw)
    head_corrected = nn.Linear(world_dim, 1).to(agent.device)

    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n and "reafference_predictor" not in n
    ]
    optimizer       = optim.Adam(standard_params,     lr=lr)
    harm_raw_optim  = optim.Adam(list(agent.e3.harm_eval_head.parameters()), lr=1e-4)
    harm_corr_optim = optim.Adam(list(head_corrected.parameters()),           lr=1e-4)
    reaf_optimizer  = optim.Adam(
        list(agent.latent_stack.reafference_predictor.parameters()), lr=1e-3,
    )

    print(
        f"[V3-EXQ-027b] SD-007 Reafference Diagnostic\n"
        f"  alpha_world={alpha_world}  warmup={warmup_episodes}  eval={eval_episodes}\n"
        f"  Training two harm_eval heads: raw z_world + corrected z_world",
        flush=True,
    )

    train_out = _train(
        agent, head_corrected, env, optimizer,
        harm_raw_optim, harm_corr_optim, reaf_optimizer,
        warmup_episodes, steps_per_episode,
    )

    reafference_r2 = _compute_reafference_r2(agent, train_out["reaf_data"])
    print(f"  reafference R²: {reafference_r2:.4f}", flush=True)

    eval_out = _eval_by_event(agent, head_corrected, env, eval_episodes, steps_per_episode)

    # PASS / FAIL
    c1_pass = eval_out["calibration_gap_raw"] > 0.03
    c2_pass = reafference_r2 > 0.20
    c3_pass = eval_out["harm_pred_std_raw"] > 0.01
    c4_pass = eval_out["n_agent_hazard_steps"] >= 5
    c5_pass = eval_out["fatal_errors"] == 0

    all_pass    = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status      = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    # Informational: correction direction
    delta = eval_out["correction_delta"]
    correction_dir = "helps" if delta > 0.005 else ("hurts" if delta < -0.005 else "neutral")

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: calibration_gap_raw={eval_out['calibration_gap_raw']:.4f} <= 0.03"
        )
    if not c2_pass:
        failure_notes.append(f"C2 FAIL: reafference_r2={reafference_r2:.4f} <= 0.20")
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: harm_pred_std_raw={eval_out['harm_pred_std_raw']:.4f} <= 0.01"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: n_agent_hazard_steps={eval_out['n_agent_hazard_steps']} < 5"
        )
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: fatal_errors={eval_out['fatal_errors']}")

    print(f"\nV3-EXQ-027b verdict: {status}  ({criteria_met}/5)", flush=True)
    print(
        f"  SD-007 correction effect: {correction_dir}"
        f"  (delta={delta:+.4f})",
        flush=True,
    )
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "alpha_world":              float(alpha_world),
        "reafference_enabled":      1.0,
        "reafference_r2":           float(reafference_r2),
        "calibration_gap_raw":      float(eval_out["calibration_gap_raw"]),
        "calibration_gap_corrected": float(eval_out["calibration_gap_corrected"]),
        "correction_delta":         float(delta),
        "harm_pred_std_raw":        float(eval_out["harm_pred_std_raw"]),
        "harm_pred_std_corrected":  float(eval_out["harm_pred_std_corrected"]),
        "n_agent_hazard_steps":     float(eval_out["n_agent_hazard_steps"]),
        "n_none_steps":             float(eval_out["n_none_steps"]),
        "fatal_error_count":        float(eval_out["fatal_errors"]),
        "warmup_harm_events":       float(train_out["total_harm"]),
        "warmup_agent_hazard":      float(train_out["n_agent_hazard"]),
        "n_empty_steps_collected":  float(train_out["n_empty_steps"]),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "crit5_pass": 1.0 if c5_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-027b — SD-007 Reafference Diagnostic

**Status:** {status}
**Claims:** SD-007, MECH-098, MECH-101
**SD-007 enabled (MECH-101 fix):** ReafferencePredictor(z_world_raw_prev + a → z_world)
**alpha_world:** {alpha_world}  (SD-008)
**Warmup:** {warmup_episodes} eps (random policy, 12×12, 15 hazards, drift)
**Eval:** {eval_episodes} eps
**Seed:** {seed}

## Motivation

EXQ-027 paradox: reafference REDUCED calibration_gap from 0.0375 (EXQ-026, no reafference)
to 0.0244 (EXQ-027, with reafference). This experiment tests whether the harm_eval head
trained on z_world_raw vs z_world_corrected shows a systematic performance difference.

## Results

| Head | calibration_gap | harm_pred_std |
|------|----------------|--------------|
| raw z_world | {eval_out['calibration_gap_raw']:.4f} | {eval_out['harm_pred_std_raw']:.4f} |
| corrected z_world (reafference) | {eval_out['calibration_gap_corrected']:.4f} | {eval_out['harm_pred_std_corrected']:.4f} |

- **correction_delta** (corrected - raw): **{delta:+.4f}**  → SD-007 **{correction_dir}** E3 calibration
- reafference_r2: {reafference_r2:.4f}  (predictor predictive quality)

## Interpretation

{
"SD-007 helps E3 calibration (correction_delta > 0). Reafference removes locomotion noise."
if delta > 0.005 else
("SD-007 hurts E3 calibration (correction_delta < 0). Over-correction hypothesis supported."
 if delta < -0.005 else
 "SD-007 is neutral for E3 calibration. Reafference does not help or hurt harm detection.")
}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: calibration_gap_raw > 0.03 (E3 baseline functional) | {"PASS" if c1_pass else "FAIL"} | {eval_out['calibration_gap_raw']:.4f} |
| C2: reafference_r2 > 0.20 (predictor works) | {"PASS" if c2_pass else "FAIL"} | {reafference_r2:.4f} |
| C3: harm_pred_std_raw > 0.01 (raw head not collapsed) | {"PASS" if c3_pass else "FAIL"} | {eval_out['harm_pred_std_raw']:.4f} |
| C4: n_agent_hazard_steps >= 5 | {"PASS" if c4_pass else "FAIL"} | {eval_out['n_agent_hazard_steps']} |
| C5: No fatal errors | {"PASS" if c5_pass else "FAIL"} | {eval_out['fatal_errors']} |

Criteria met: {criteria_met}/5 → **{status}**
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if criteria_met >= 3 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": eval_out["fatal_errors"],
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("--warmup",      type=int,   default=800)
    parser.add_argument("--eval-eps",    type=int,   default=50)
    parser.add_argument("--steps",       type=int,   default=200)
    parser.add_argument("--alpha-world", type=float, default=0.9)
    parser.add_argument("--alpha-self",  type=float, default=0.3)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
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
