"""
V3-EXQ-082 -- MECH-098 Reafference vs No-Reafference: Harm Calibration Discriminative Pair

Claims: MECH-098
Proposal: EXP-0001 / EVB-0001

MECH-098 asserts that z_world corrected by reafference cancellation (SD-007) isolates
genuine external world-state changes from self-motion perspective shifts, improving
E3.harm_eval's ability to detect hazard approach.

When the agent moves, the egocentric view shifts -- this is a reafferent change fully
predictable from the motor command. Without cancellation, z_world encodes perspective-
shift noise alongside genuine hazard proximity signals. The ReafferencePredictor
(SD-007) subtracts the predicted perspective-shift component:

  z_world_corrected = z_world_raw - ReafferencePredictor(z_world_raw_prev, a_prev)

This leaves only the genuine world-state change (exafference), giving E3.harm_eval
a cleaner spatial hazard signal.

Discriminative pair (dispatch_mode=discriminative_pair, EXP-0001):

  REAFFERENCE    -- reafference_action_dim=env.action_dim (SD-007 ON).
                   z_world_corrected is perspective-shifted corrected.
  NO_REAFFERENCE -- reafference_action_dim=0 (SD-007 OFF, ablation).
                   Raw z_world without correction.

Both conditions: alpha_world=0.9 (SD-008), z_self/z_world split (SD-005, unified=False),
SD-010 NOT enabled (isolates MECH-098 from SD-010 effects), same CausalGridWorldV2
with gradient harm fields, same seeds.

Mechanism under test:
  In REAFFERENCE, z_world_corrected represents genuine world-content changes (hazard
  field proximity, objects entering the scene) independent of self-motion direction.
  E3.harm_eval trained on this cleaner signal should show higher gap between
  hazard_approach and none states.
  In NO_REAFFERENCE, z_world blends genuine world change with perspective-shift noise.
  The SNR for hazard proximity is lower, impairing approach detection.

Prior data: EXQ-016 (R2=0.118, EMA alpha=0.3 caused dilution -- SD-008 prerequisite),
EXQ-027b PASS (ReafferencePredictor R2>0.20 with alpha_world=0.9 and z_world_raw_prev input).
EVB-0001 has conflict_ratio=1.0 (exp confidence 0.675 vs lit confidence 0.865) --
this discriminative pair resolves the conflict with a clean isolated test.

Pre-registered primary discriminator (threshold >= 0.02):

  delta_approach_gap = gap_approach_REAFFERENCE - gap_approach_NO_REAFFERENCE
  gap_approach = mean(harm_eval at hazard_approach) - mean(harm_eval at none)

PASS criteria (ALL required):
  C1: delta_approach_gap > 0.02  (reafference correction improves approach detection by >= 2pp)
  C2: gap_approach_REAFFERENCE > 0.06  (corrected z_world: approach detectable, not just less bad)
  C3: gap_approach_NO_REAFFERENCE > 0   (raw z_world also learns something -- not both flat)

Decision scoring:
  retain_ree:       C1+C2+C3 all pass
  hybridize:        C2+C3 pass, delta between 0 and 0.02 (directional but marginal benefit)
  retire_ree_claim: C2 fails (even reafference-corrected z_world cannot detect approach)

Also informs EXP-0002 (SD-007): if REAFFERENCE wins, provides direct behavioral evidence
that SD-007 is beneficial, complementing EXQ-027b's predictor R2 evidence.
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


EXPERIMENT_TYPE = "v3_exq_082_mech098_reafference_harm_pair"
CLAIM_IDS = ["MECH-098"]


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _run_single(
    seed: int,
    reafference: bool,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    lr: float,
    alpha_world: float,
    alpha_self: float,
    harm_scale: float,
    proximity_harm_scale: float,
) -> Dict:
    """Run one (seed, condition) cell and return calibration gap metrics."""
    torch.manual_seed(seed)
    random.seed(seed)

    cond_label = "REAFFERENCE" if reafference else "NO_REAFFERENCE"

    env = CausalGridWorldV2(
        seed=seed,
        size=12,
        num_hazards=4,
        num_resources=5,
        hazard_harm=harm_scale,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=proximity_harm_scale,
        proximity_benefit_scale=proximity_harm_scale * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        # SD-007: enable reafference correction (MECH-098 under test)
        reafference_action_dim=env.action_dim if reafference else 0,
    )
    # SD-005 split always enabled in both conditions (not under test)
    config.latent.unified_latent_mode = False

    agent = REEAgent(config)

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF_EACH = 2000

    # ReafferencePredictor (when enabled) is part of agent.parameters() via the
    # LatentStack and will be trained by backprop through compute_e2_loss().
    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n
    ]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())

    optimizer = optim.Adam(standard_params, lr=lr)
    harm_eval_optimizer = optim.Adam(harm_eval_params, lr=1e-4)

    counts: Dict[str, int] = {
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

            if ttype in counts:
                counts[ttype] += 1

            # Harm buffer
            is_harm = float(harm_signal) < 0
            if is_harm:
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > MAX_BUF_EACH:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF_EACH:]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > MAX_BUF_EACH:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF_EACH:]

            # Standard E1 + E2 losses (trains ReafferencePredictor implicitly when enabled)
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # E3 harm_eval balanced training
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
                f" approach={counts['hazard_approach']}"
                f" contact={counts['env_caused_hazard']+counts['agent_caused_hazard']}"
                f" buf_pos={len(harm_buf_pos)} buf_neg={len(harm_buf_neg)}",
                flush=True,
            )

    # --- EVAL ---
    agent.eval()

    scores: Dict[str, List[float]] = {
        "none": [],
        "hazard_approach": [],
        "env_caused_hazard": [],
        "agent_caused_hazard": [],
    }
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

    gap_approach = means["hazard_approach"] - means["none"]
    mean_contact = (means["env_caused_hazard"] + means["agent_caused_hazard"]) / 2.0
    gap_contact = mean_contact - means["none"]

    print(
        f"  [eval] seed={seed} cond={cond_label}"
        f" none={means['none']:.4f}"
        f" approach={means['hazard_approach']:.4f}"
        f" contact={means['env_caused_hazard']:.4f}/{means['agent_caused_hazard']:.4f}"
        f" gap_approach={gap_approach:.4f}"
        f" gap_contact={gap_contact:.4f}"
        f" n={n_counts}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "reafference": reafference,
        "gap_approach": float(gap_approach),
        "gap_contact": float(gap_contact),
        "mean_score_none": float(means["none"]),
        "mean_score_approach": float(means["hazard_approach"]),
        "mean_score_contact": float(mean_contact),
        "n_approach_eval": int(n_counts.get("hazard_approach", 0)),
        "n_contact_eval": int(
            n_counts.get("env_caused_hazard", 0) + n_counts.get("agent_caused_hazard", 0)
        ),
        "n_none_eval": int(n_counts.get("none", 0)),
        "train_approach_events": int(counts["hazard_approach"]),
        "train_contact_events": int(
            counts["env_caused_hazard"] + counts["agent_caused_hazard"]
        ),
        "n_fatal": int(n_fatal),
    }


def run(
    seeds: Tuple = (42, 7),
    warmup_episodes: int = 350,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_harm_scale: float = 0.05,
    **kwargs,
) -> dict:
    """Discriminative pair: REAFFERENCE (SD-007 ON) vs NO_REAFFERENCE (ablation)."""
    results_reafference:    List[Dict] = []
    results_no_reafference: List[Dict] = []

    for seed in seeds:
        for reafference in [True, False]:
            label = "REAFFERENCE" if reafference else "NO_REAFFERENCE"
            print(
                f"\n[V3-EXQ-082] {label} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" alpha_world={alpha_world}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                reafference=reafference,
                warmup_episodes=warmup_episodes,
                eval_episodes=eval_episodes,
                steps_per_episode=steps_per_episode,
                self_dim=self_dim,
                world_dim=world_dim,
                lr=lr,
                alpha_world=alpha_world,
                alpha_self=alpha_self,
                harm_scale=harm_scale,
                proximity_harm_scale=proximity_harm_scale,
            )
            if reafference:
                results_reafference.append(r)
            else:
                results_no_reafference.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    gap_approach_reafference    = _avg(results_reafference,    "gap_approach")
    gap_approach_no_reafference = _avg(results_no_reafference, "gap_approach")
    gap_contact_reafference     = _avg(results_reafference,    "gap_contact")
    gap_contact_no_reafference  = _avg(results_no_reafference, "gap_contact")
    delta_approach_gap          = gap_approach_reafference - gap_approach_no_reafference

    n_approach_min = min(
        r["n_approach_eval"] for r in results_reafference + results_no_reafference
    )

    # Pre-registered PASS criteria
    c1_pass = delta_approach_gap          > 0.02
    c2_pass = gap_approach_reafference    > 0.06
    c3_pass = gap_approach_no_reafference > 0

    all_pass     = c1_pass and c2_pass and c3_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass])
    status = "PASS" if all_pass else "FAIL"

    # Decision scoring
    if all_pass:
        decision = "retain_ree"
    elif c2_pass and c3_pass and delta_approach_gap >= 0:
        decision = "hybridize"
    else:
        decision = "retire_ree_claim"

    print(f"\n[V3-EXQ-082] Final results:", flush=True)
    print(
        f"  gap_approach_REAFFERENCE={gap_approach_reafference:.4f}"
        f"  gap_approach_NO_REAFFERENCE={gap_approach_no_reafference:.4f}",
        flush=True,
    )
    print(
        f"  delta_approach_gap={delta_approach_gap:+.4f}"
        f"  gap_contact_REAFFERENCE={gap_contact_reafference:.4f}"
        f"  gap_contact_NO_REAFFERENCE={gap_contact_no_reafference:.4f}",
        flush=True,
    )
    print(f"  decision={decision}  status={status} ({criteria_met}/3)", flush=True)

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: delta_approach_gap={delta_approach_gap:.4f} <= 0.02"
            " (reafference correction does not improve approach detection by 2pp)"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: gap_approach_REAFFERENCE={gap_approach_reafference:.4f} <= 0.06"
            " (corrected z_world cannot detect hazard approach at all)"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: gap_approach_NO_REAFFERENCE={gap_approach_no_reafference:.4f} <= 0"
            " (raw z_world approach detection is flat -- both conditions fail)"
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "gap_approach_reafference":        float(gap_approach_reafference),
        "gap_approach_no_reafference":     float(gap_approach_no_reafference),
        "gap_contact_reafference":         float(gap_contact_reafference),
        "gap_contact_no_reafference":      float(gap_contact_no_reafference),
        "delta_approach_gap":              float(delta_approach_gap),
        "delta_contact_gap":               float(gap_contact_reafference - gap_contact_no_reafference),
        "mean_score_none_reafference":     float(_avg(results_reafference,    "mean_score_none")),
        "mean_score_none_no_reafference":  float(_avg(results_no_reafference, "mean_score_none")),
        "mean_score_approach_reafference":    float(_avg(results_reafference,    "mean_score_approach")),
        "mean_score_approach_no_reafference": float(_avg(results_no_reafference, "mean_score_approach")),
        "n_approach_min":                  float(n_approach_min),
        "n_seeds":                         float(len(seeds)),
        "alpha_world":                     float(alpha_world),
        "proximity_harm_scale":            float(proximity_harm_scale),
        "crit1_pass":                      1.0 if c1_pass else 0.0,
        "crit2_pass":                      1.0 if c2_pass else 0.0,
        "crit3_pass":                      1.0 if c3_pass else 0.0,
        "criteria_met":                    float(criteria_met),
    }

    if all_pass:
        interpretation = (
            "MECH-098 SUPPORTED: reafference cancellation improves hazard approach detection."
            f" gap_approach_REAFFERENCE={gap_approach_reafference:.4f} vs"
            f" NO_REAFFERENCE={gap_approach_no_reafference:.4f},"
            f" delta={delta_approach_gap:+.4f}."
            " Perspective-shift subtraction gives E3 cleaner world-state signal with"
            " less self-motion noise. Resolves EVB-0001 conflict (exp/lit mismatch)."
            " Also supports SD-007 (behavioral evidence, complements EXQ-027b R2)."
        )
    elif c2_pass and c3_pass and delta_approach_gap >= 0:
        interpretation = (
            "Weak positive: REAFFERENCE approach detection > 0.06 (works) and direction"
            f" is correct (delta={delta_approach_gap:+.4f}) but improvement below 2pp"
            " threshold. MECH-098 cancellation is directionally beneficial; current model"
            " scale may be insufficient for full 2pp margin. Consider larger world_dim"
            " or more warmup episodes for stronger separation signal."
        )
    else:
        interpretation = (
            "MECH-098 NOT SUPPORTED: reafference correction does not improve approach"
            f" detection (delta={delta_approach_gap:+.4f},"
            f" gap_REAFFERENCE={gap_approach_reafference:.4f})."
            " Perspective-shift noise may not dominate at this model/task scale, or the"
            " implicit E2 training signal is insufficient to train the ReafferencePredictor."
            " Consider explicit predictor training (see EXQ-027b pattern) or larger"
            " world_dim for more separation."
        )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    per_reafference_rows = "\n".join(
        f"  seed={r['seed']}: gap_approach={r['gap_approach']:.4f}"
        f" gap_contact={r['gap_contact']:.4f}"
        f" n_approach={r['n_approach_eval']}"
        f" train_approach={r['train_approach_events']}"
        for r in results_reafference
    )
    per_no_reafference_rows = "\n".join(
        f"  seed={r['seed']}: gap_approach={r['gap_approach']:.4f}"
        f" gap_contact={r['gap_contact']:.4f}"
        f" n_approach={r['n_approach_eval']}"
        for r in results_no_reafference
    )

    summary_markdown = (
        f"# V3-EXQ-082 -- MECH-098 Reafference vs No-Reafference: Harm Calibration Pair\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-098\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: delta_approach_gap (REAFFERENCE - NO_REAFFERENCE) > 0.02\n"
        f"C2: gap_approach_REAFFERENCE > 0.06\n"
        f"C3: gap_approach_NO_REAFFERENCE > 0\n\n"
        f"## Results\n\n"
        f"| Condition | gap_approach | gap_contact | mean_none | mean_approach |\n"
        f"|-----------|-------------|-------------|-----------|---------------|\n"
        f"| REAFFERENCE    | {gap_approach_reafference:.4f} | {gap_contact_reafference:.4f}"
        f" | {_avg(results_reafference,    'mean_score_none'):.4f}"
        f" | {_avg(results_reafference,    'mean_score_approach'):.4f} |\n"
        f"| NO_REAFFERENCE | {gap_approach_no_reafference:.4f} | {gap_contact_no_reafference:.4f}"
        f" | {_avg(results_no_reafference, 'mean_score_none'):.4f}"
        f" | {_avg(results_no_reafference, 'mean_score_approach'):.4f} |\n\n"
        f"**delta_approach_gap (REAFFERENCE - NO_REAFFERENCE): {delta_approach_gap:+.4f}**\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: delta_approach_gap > 0.02 | {'PASS' if c1_pass else 'FAIL'}"
        f" | {delta_approach_gap:.4f} |\n"
        f"| C2: gap_approach_REAFFERENCE > 0.06 | {'PASS' if c2_pass else 'FAIL'}"
        f" | {gap_approach_reafference:.4f} |\n"
        f"| C3: gap_approach_NO_REAFFERENCE > 0  | {'PASS' if c3_pass else 'FAIL'}"
        f" | {gap_approach_no_reafference:.4f} |\n\n"
        f"Criteria met: {criteria_met}/3 -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n\n"
        f"## Per-Seed\n\n"
        f"REAFFERENCE:\n{per_reafference_rows}\n\n"
        f"NO_REAFFERENCE:\n{per_no_reafference_rows}\n"
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
        "fatal_error_count": sum(
            r["n_fatal"] for r in results_reafference + results_no_reafference
        ),
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",           type=int, nargs="+", default=[42, 7])
    parser.add_argument("--warmup",          type=int,   default=350)
    parser.add_argument("--eval-eps",        type=int,   default=50)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
    parser.add_argument(
        "--smoke", action="store_true",
        help="Quick smoke test: 1 seed, 5 warmup eps, 3 eval eps, 50 steps"
    )
    args = parser.parse_args()

    if args.smoke:
        seeds    = (42,)
        warmup   = 5
        eval_eps = 3
        steps    = 50
        print("[V3-EXQ-082] SMOKE TEST MODE", flush=True)
    else:
        seeds    = tuple(args.seeds)
        warmup   = args.warmup
        eval_eps = args.eval_eps
        steps    = args.steps

    result = run(
        seeds=seeds,
        warmup_episodes=warmup,
        eval_episodes=eval_eps,
        steps_per_episode=steps,
        alpha_world=args.alpha_world,
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

    if args.smoke:
        print(f"\n[SMOKE] Status: {result['status']}", flush=True)
        print("[SMOKE] Key metrics:", flush=True)
        for k in [
            "gap_approach_reafference", "gap_approach_no_reafference",
            "delta_approach_gap",
            "crit1_pass", "crit2_pass", "crit3_pass",
        ]:
            print(f"  {k}: {result['metrics'].get(k, 'N/A')}", flush=True)
        print("[SMOKE] Script ran without error. Not writing output file.", flush=True)
    else:
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
