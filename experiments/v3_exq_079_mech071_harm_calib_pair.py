"""
V3-EXQ-079 -- MECH-071 Harm Calibration Discriminative Pair (SD-008)

Claims: MECH-071
Proposal: EXP-0006 / EVB-0006

MECH-071 asserts that E3.harm_eval is better calibrated for agent-caused vs
environment-caused transitions. The mechanism depends on SD-008: alpha_world
must be >= 0.9 so z_world responds sharply to hazard events rather than
smoothing them into an indistinct EMA average.

Directional conflict motivating this pair (EXP-0006 why_now):
  EXQ-027 FAIL: E2.predict_harm calibration_gap ~ 0.0007 (alpha=0.3, unified z_gamma)
  EXQ-026 PASS: E3.harm_eval calibration_gap = 0.037  (alpha_world=0.9, split z_world)
  Both experiments confirm that SD-008 is the enabling mechanism, but no matched
  discriminative pair has directly tested alpha=0.9 vs alpha=0.3 under identical
  conditions while measuring MECH-071's agent/env calibration asymmetry.

Discriminative pair (dispatch_mode=discriminative_pair, EXP-0006):

  SHARP_WORLD  -- alpha_world=0.9: z_world tracks hazard events sharply (SD-008)
  SMOOTH_WORLD -- alpha_world=0.3: z_world is a 3-step EMA, events suppressed ~30%

Both conditions: same CausalGridWorldV2 (proximity_harm_scale=0.05, env_drift), same
seeds, same training budget, same architecture, NO reafference (reafference_action_dim=0
isolates the alpha_world mechanism; SD-007 is a separate contribution).

Mechanism under test:
  In SHARP_WORLD, z_world transitions sharply at hazard events; E3.harm_eval learns to
  distinguish world states near hazards (agent_caused, env_caused) from safe locomotion
  (none). calibration_gap = mean(E3.harm_eval at agent_caused_hazard) - mean(E3.harm_eval
  at none) is large. In SMOOTH_WORLD, z_world is a 3-step weighted average -- hazard
  event response is attenuated to ~30%, all transition types produce similar z_world
  signals, and E3 cannot learn the hazard calibration. calibration_gap collapses to ~0.

Pre-registered primary discriminator (threshold >= 0.015):

  delta_calibration_gap = calibration_gap_SHARP - calibration_gap_SMOOTH
  calibration_gap = mean(E3.harm_eval at agent_caused_hazard) - mean(E3.harm_eval at none)

PASS criteria (ALL required):
  C1: delta_calibration_gap    > 0.015  (SD-008 enables MECH-071 calibration)
  C2: calibration_gap_SHARP    > 0.025  (positive asymmetry confirmed, SD-008 active)
  C3: calibration_gap_SMOOTH   < 0.020  (calibration collapses without SD-008)
  C4: both seeds individually: calibration_gap_SHARP > calibration_gap_SMOOTH (per-seed)

Decision scoring:
  retain_ree:       C1+C2+C3+C4 all pass
  hybridize:        C1 passes and delta >= 0 but delta < 0.015 (gap present, marginal)
  retire_ree_claim: C1 fails (SD-008 does not enable MECH-071 calibration)
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_079_mech071_harm_calib_pair"
CLAIM_IDS = ["MECH-071"]


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _run_single(
    seed: int,
    alpha_world: float,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    lr: float,
    alpha_self: float,
    harm_scale: float,
    proximity_harm_scale: float,
) -> Dict:
    """Run one (seed, alpha_world) cell and return calibration gap metrics."""
    torch.manual_seed(seed)
    random.seed(seed)

    cond_label = "SHARP_WORLD" if alpha_world >= 0.9 else "SMOOTH_WORLD"

    env = CausalGridWorldV2(
        seed=seed,
        size=12,
        num_hazards=4,
        num_resources=5,
        hazard_harm=harm_scale,
        env_drift_interval=3,
        env_drift_prob=0.3,
        proximity_harm_scale=proximity_harm_scale,
        proximity_benefit_scale=proximity_harm_scale * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )

    # reafference_action_dim=0 disables SD-007 in both conditions.
    # Only alpha_world varies -- isolates SD-008 contribution.
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=0,
    )
    agent = REEAgent(config)

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF_EACH = 2000

    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n
    ]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())

    optimizer = optim.Adam(standard_params, lr=lr)
    harm_eval_optimizer = optim.Adam(harm_eval_params, lr=1e-4)

    train_counts: Dict[str, int] = {
        "hazard_approach": 0,
        "env_caused_hazard": 0,
        "agent_caused_hazard": 0,
        "none": 0,
    }

    # --- TRAIN ---
    agent.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_curr = latent.z_world.detach()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            if ttype in train_counts:
                train_counts[ttype] += 1

            # Balanced harm_eval training buffers.
            # harm_signal < 0 at hazard_approach AND agent/env contact (proximity_harm_scale=0.05).
            is_harm = float(harm_signal) < 0
            if is_harm:
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > MAX_BUF_EACH:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF_EACH:]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > MAX_BUF_EACH:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF_EACH:]

            # Standard E1 + E2 losses
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # E3 harm_eval balanced training (16 pos + 16 neg per step)
            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()
                zw_pos = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
                zw_neg = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)
                zw_b = torch.cat([zw_pos, zw_neg], dim=0)
                target = torch.cat([
                    torch.ones(k_pos, 1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                ], dim=0)
                pred_harm = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred_harm, target)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5,
                    )
                    harm_eval_optimizer.step()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
            print(
                f"  [train] seed={seed} cond={cond_label}"
                f" ep {ep+1}/{warmup_episodes}"
                f" approach={train_counts['hazard_approach']}"
                f" agent_contact={train_counts['agent_caused_hazard']}"
                f" env_contact={train_counts['env_caused_hazard']}"
                f" buf_pos={len(harm_buf_pos)} buf_neg={len(harm_buf_neg)}",
                flush=True,
            )

    # --- EVAL: score z_world by transition type ---
    agent.eval()

    scores: Dict[str, List[float]] = {
        "none": [],
        "hazard_approach": [],
        "env_caused_hazard": [],
        "agent_caused_hazard": [],
    }
    all_scores: List[float] = []
    n_fatal = 0

    for _ in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()
                z_world_curr = latent.z_world

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            _, _, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            try:
                with torch.no_grad():
                    score = float(agent.e3.harm_eval(z_world_curr).item())
                all_scores.append(score)
                if ttype in scores:
                    scores[ttype].append(score)
            except Exception:
                n_fatal += 1

            if done:
                break

    means: Dict[str, float] = {
        k: float(sum(v) / max(1, len(v))) for k, v in scores.items()
    }
    n_counts = {k: len(v) for k, v in scores.items()}
    pred_std = float(torch.tensor(all_scores).std().item()) if len(all_scores) > 1 else 0.0

    calibration_gap = means["agent_caused_hazard"] - means["none"]
    env_gap         = means["env_caused_hazard"]   - means["none"]
    approach_gap    = means["hazard_approach"]      - means["none"]

    print(
        f"  [eval] seed={seed} cond={cond_label}"
        f" none={means['none']:.4f}"
        f" agent_caused={means['agent_caused_hazard']:.4f}"
        f" env_caused={means['env_caused_hazard']:.4f}"
        f" approach={means['hazard_approach']:.4f}",
        flush=True,
    )
    print(
        f"         calibration_gap={calibration_gap:.4f}"
        f" env_gap={env_gap:.4f}"
        f" approach_gap={approach_gap:.4f}"
        f" pred_std={pred_std:.4f}"
        f" n={n_counts}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "alpha_world": float(alpha_world),
        "calibration_gap": float(calibration_gap),
        "env_gap": float(env_gap),
        "approach_gap": float(approach_gap),
        "mean_score_none": float(means["none"]),
        "mean_score_agent_caused": float(means["agent_caused_hazard"]),
        "mean_score_env_caused": float(means["env_caused_hazard"]),
        "mean_score_approach": float(means["hazard_approach"]),
        "harm_pred_std": float(pred_std),
        "n_none_eval": int(n_counts.get("none", 0)),
        "n_agent_hazard_eval": int(n_counts.get("agent_caused_hazard", 0)),
        "n_env_hazard_eval": int(n_counts.get("env_caused_hazard", 0)),
        "n_approach_eval": int(n_counts.get("hazard_approach", 0)),
        "train_agent_hazard": int(train_counts["agent_caused_hazard"]),
        "train_env_hazard": int(train_counts["env_caused_hazard"]),
        "train_approach": int(train_counts["hazard_approach"]),
        "n_fatal": int(n_fatal),
    }


def run(
    seeds: Tuple = (42, 7),
    warmup_episodes: int = 350,
    eval_episodes: int = 60,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world_sharp: float = 0.9,
    alpha_world_smooth: float = 0.3,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_harm_scale: float = 0.05,
    **kwargs,
) -> dict:
    """Discriminative pair: SHARP_WORLD (alpha=0.9) vs SMOOTH_WORLD (alpha=0.3)."""
    results_sharp: List[Dict] = []
    results_smooth: List[Dict] = []

    for seed in seeds:
        for alpha in [alpha_world_sharp, alpha_world_smooth]:
            label = "SHARP_WORLD" if alpha >= 0.9 else "SMOOTH_WORLD"
            print(
                f"\n[V3-EXQ-079] {label} seed={seed}"
                f" alpha_world={alpha}"
                f" warmup={warmup_episodes} eval={eval_episodes}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                alpha_world=alpha,
                warmup_episodes=warmup_episodes,
                eval_episodes=eval_episodes,
                steps_per_episode=steps_per_episode,
                self_dim=self_dim,
                world_dim=world_dim,
                lr=lr,
                alpha_self=alpha_self,
                harm_scale=harm_scale,
                proximity_harm_scale=proximity_harm_scale,
            )
            if alpha >= 0.9:
                results_sharp.append(r)
            else:
                results_smooth.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    calibration_gap_sharp  = _avg(results_sharp,  "calibration_gap")
    calibration_gap_smooth = _avg(results_smooth, "calibration_gap")
    delta_calibration_gap  = calibration_gap_sharp - calibration_gap_smooth
    env_gap_sharp          = _avg(results_sharp,  "env_gap")
    approach_gap_sharp     = _avg(results_sharp,  "approach_gap")

    # Per-seed directionality check (C4)
    c4_per_seed = [
        r_sharp["calibration_gap"] > r_smooth["calibration_gap"]
        for r_sharp, r_smooth in zip(results_sharp, results_smooth)
    ]
    c4_pass = all(c4_per_seed)

    # Validity checks
    n_agent_min = min(r["n_agent_hazard_eval"] for r in results_sharp + results_smooth)
    n_env_min   = min(r["n_env_hazard_eval"]   for r in results_sharp + results_smooth)
    valid_counts = n_agent_min >= 3

    # Pre-registered PASS criteria
    c1_pass = delta_calibration_gap > 0.015
    c2_pass = calibration_gap_sharp > 0.025
    c3_pass = calibration_gap_smooth < 0.020
    all_pass = c1_pass and c2_pass and c3_pass and c4_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    status = "PASS" if all_pass else "FAIL"

    # Decision scoring
    if all_pass:
        decision = "retain_ree"
    elif c1_pass and delta_calibration_gap >= 0:
        decision = "hybridize"
    else:
        decision = "retire_ree_claim"

    print(f"\n[V3-EXQ-079] Final results:", flush=True)
    print(
        f"  calibration_gap_SHARP={calibration_gap_sharp:.4f}"
        f"  calibration_gap_SMOOTH={calibration_gap_smooth:.4f}",
        flush=True,
    )
    print(
        f"  delta_calibration_gap={delta_calibration_gap:+.4f}"
        f"  env_gap_SHARP={env_gap_sharp:.4f}"
        f"  approach_gap_SHARP={approach_gap_sharp:.4f}",
        flush=True,
    )
    print(
        f"  n_agent_min={n_agent_min}  n_env_min={n_env_min}"
        f"  valid_counts={valid_counts}",
        flush=True,
    )
    print(f"  decision={decision}  status={status} ({criteria_met}/4)", flush=True)

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: delta_calibration_gap={delta_calibration_gap:.4f} <= 0.015"
            " (SD-008 does not enable MECH-071 calibration asymmetry)"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: calibration_gap_SHARP={calibration_gap_sharp:.4f} <= 0.025"
            " (no positive asymmetry even with alpha_world=0.9)"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: calibration_gap_SMOOTH={calibration_gap_smooth:.4f} >= 0.020"
            " (calibration gap unexpectedly large without SD-008 -- confound present)"
        )
    if not c4_pass:
        failure_notes.append(
            "C4 FAIL: not all seeds show SHARP > SMOOTH directionality -- inconsistent"
        )
    if not valid_counts:
        failure_notes.append(
            f"WARNING: n_agent_hazard_eval_min={n_agent_min} < 3"
            " (insufficient agent-caused contacts for reliable mean)"
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "calibration_gap_sharp":   float(calibration_gap_sharp),
        "calibration_gap_smooth":  float(calibration_gap_smooth),
        "delta_calibration_gap":   float(delta_calibration_gap),
        "env_gap_sharp":           float(env_gap_sharp),
        "env_gap_smooth":          float(_avg(results_smooth, "env_gap")),
        "approach_gap_sharp":      float(approach_gap_sharp),
        "approach_gap_smooth":     float(_avg(results_smooth, "approach_gap")),
        "mean_score_none_sharp":   float(_avg(results_sharp,  "mean_score_none")),
        "mean_score_none_smooth":  float(_avg(results_smooth, "mean_score_none")),
        "mean_score_agent_sharp":  float(_avg(results_sharp,  "mean_score_agent_caused")),
        "mean_score_agent_smooth": float(_avg(results_smooth, "mean_score_agent_caused")),
        "n_agent_min":             float(n_agent_min),
        "n_env_min":               float(n_env_min),
        "n_seeds":                 float(len(seeds)),
        "alpha_world_sharp":       float(alpha_world_sharp),
        "alpha_world_smooth":      float(alpha_world_smooth),
        "crit1_pass":              1.0 if c1_pass else 0.0,
        "crit2_pass":              1.0 if c2_pass else 0.0,
        "crit3_pass":              1.0 if c3_pass else 0.0,
        "crit4_pass":              1.0 if c4_pass else 0.0,
        "criteria_met":            float(criteria_met),
    }

    if all_pass:
        interpretation = (
            "MECH-071 SUPPORTED: SD-008 (alpha_world=0.9) is the enabling mechanism"
            " for E3.harm_eval calibration asymmetry between agent-caused and env-caused"
            f" transitions. SHARP_WORLD calibration_gap={calibration_gap_sharp:.4f} vs"
            f" SMOOTH_WORLD {calibration_gap_smooth:.4f}."
            " Without sharp event responses in z_world, calibration collapses."
            " E3 attribution structure (MECH-071) depends on SD-008 as prerequisite."
        )
    elif c1_pass and delta_calibration_gap >= 0:
        interpretation = (
            "Weak positive: SHARP_WORLD shows higher calibration gap but discriminative"
            " margin vs SMOOTH_WORLD is below threshold. SD-008 helps but the mechanism"
            " is marginal -- z_world may encode some hazard structure even at alpha=0.3."
        )
    else:
        interpretation = (
            "MECH-071 NOT supported by this pair: SD-008 does not discriminate"
            f" calibration gap (delta={delta_calibration_gap:.4f} <= 0.015)."
            " Either z_world encodes hazard structure independently of EMA alpha,"
            " or E3 harm_eval cannot learn calibration asymmetry at all."
            " Consider whether proxy gradient fields (ARC-024) are sufficient alone"
            " to explain EXQ-026 calibration_gap=0.037 without alpha sensitivity."
        )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    per_sharp_rows = "\n".join(
        f"  seed={r['seed']}: calibration_gap={r['calibration_gap']:.4f}"
        f" env_gap={r['env_gap']:.4f}"
        f" approach_gap={r['approach_gap']:.4f}"
        f" n_agent={r['n_agent_hazard_eval']}"
        f" n_env={r['n_env_hazard_eval']}"
        for r in results_sharp
    )
    per_smooth_rows = "\n".join(
        f"  seed={r['seed']}: calibration_gap={r['calibration_gap']:.4f}"
        f" env_gap={r['env_gap']:.4f}"
        f" approach_gap={r['approach_gap']:.4f}"
        for r in results_smooth
    )

    summary_markdown = (
        f"# V3-EXQ-079 -- MECH-071 Harm Calibration Discriminative Pair\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-071\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**SHARP_WORLD alpha_world:** {alpha_world_sharp}  (SD-008)\n"
        f"**SMOOTH_WORLD alpha_world:** {alpha_world_smooth}  (baseline)\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: delta_calibration_gap        > 0.015\n"
        f"C2: calibration_gap_SHARP        > 0.025\n"
        f"C3: calibration_gap_SMOOTH       < 0.020\n"
        f"C4: per-seed SHARP > SMOOTH directionality\n\n"
        f"## Results\n\n"
        f"| Condition | calibration_gap | env_gap | approach_gap | mean_none | mean_agent |\n"
        f"|-----------|----------------|---------|--------------|-----------|------------|\n"
        f"| SHARP_WORLD  | {calibration_gap_sharp:.4f}  | {env_gap_sharp:.4f}"
        f" | {approach_gap_sharp:.4f}"
        f" | {_avg(results_sharp, 'mean_score_none'):.4f}"
        f" | {_avg(results_sharp, 'mean_score_agent_caused'):.4f} |\n"
        f"| SMOOTH_WORLD | {calibration_gap_smooth:.4f}"
        f" | {_avg(results_smooth, 'env_gap'):.4f}"
        f" | {_avg(results_smooth, 'approach_gap'):.4f}"
        f" | {_avg(results_smooth, 'mean_score_none'):.4f}"
        f" | {_avg(results_smooth, 'mean_score_agent_caused'):.4f} |\n\n"
        f"**delta_calibration_gap (SHARP - SMOOTH): {delta_calibration_gap:+.4f}**\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: delta_calibration_gap > 0.015    | {'PASS' if c1_pass else 'FAIL'}"
        f" | {delta_calibration_gap:.4f} |\n"
        f"| C2: calibration_gap_SHARP > 0.025    | {'PASS' if c2_pass else 'FAIL'}"
        f" | {calibration_gap_sharp:.4f} |\n"
        f"| C3: calibration_gap_SMOOTH < 0.020   | {'PASS' if c3_pass else 'FAIL'}"
        f" | {calibration_gap_smooth:.4f} |\n"
        f"| C4: per-seed SHARP > SMOOTH           | {'PASS' if c4_pass else 'FAIL'}"
        f" | {c4_per_seed} |\n\n"
        f"Criteria met: {criteria_met}/4 -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n\n"
        f"## Per-Seed\n\n"
        f"SHARP_WORLD:\n{per_sharp_rows}\n\n"
        f"SMOOTH_WORLD:\n{per_smooth_rows}\n"
        f"{failure_section}\n"
    )

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if criteria_met >= 2 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": sum(r["n_fatal"] for r in results_sharp + results_smooth),
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",          type=int,   nargs="+", default=[42, 7])
    parser.add_argument("--warmup",         type=int,   default=350)
    parser.add_argument("--eval-eps",       type=int,   default=60)
    parser.add_argument("--steps",          type=int,   default=200)
    parser.add_argument("--alpha-sharp",    type=float, default=0.9)
    parser.add_argument("--alpha-smooth",   type=float, default=0.3)
    parser.add_argument("--alpha-self",     type=float, default=0.3)
    parser.add_argument("--harm-scale",     type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
    args = parser.parse_args()

    result = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world_sharp=args.alpha_sharp,
        alpha_world_smooth=args.alpha_smooth,
        alpha_self=args.alpha_self,
        harm_scale=args.harm_scale,
        proximity_harm_scale=args.proximity_scale,
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
