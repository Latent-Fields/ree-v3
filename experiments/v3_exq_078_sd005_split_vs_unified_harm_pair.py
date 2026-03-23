"""
V3-EXQ-078 -- SD-005 Split vs Unified Latent: Harm Calibration Discriminative Pair

Claims: SD-005
Proposal: EXP-0005 / EVB-0005

SD-005 asserts that splitting z_gamma into z_self (body/motor domain, E2) and z_world
(world-state domain, E3/Hippocampus) is the correct architectural design. The split
should allow z_world to specialize as a cleaner world-state representation, improving
E3.harm_eval's ability to detect hazard approach (spatial proximity signal).

Discriminative pair (dispatch_mode=discriminative_pair, EXP-0005):

  SPLIT   -- unified_latent_mode=False (default): z_self and z_world are separate.
             z_world is trained only on world_obs (hazard_field_view, layout, content).
  UNIFIED -- unified_latent_mode=True: z_self = z_world = average(z_self_raw, z_world_raw).
             z_world receives diluted representation (body state mixed with world state).

Both conditions: same CausalGridWorldV2 with gradient harm fields, same seeds, same
E3.harm_eval training (harm_signal < 0 events). Reafference and SD-010 NOT enabled
(isolates SD-005 from SD-007 and SD-010).

Mechanism under test:
  In SPLIT, z_world is derived purely from world_obs which contains hazard_field_view
  (spatial proximity map for all hazards) -- the key signal for pre-contact hazard approach.
  In UNIFIED, z_world = (body_obs_encoded + world_obs_encoded) / 2, diluting the spatial
  proximity signal with body state (velocity, proprioception). Since E3.harm_eval reads
  z_world, SPLIT should produce better approach detection (the spatial hazard field is
  higher SNR in SPLIT's z_world than UNIFIED's).

Prior data (EXQ-047): split produced 3.4pp calibration improvement (0.1915 vs 0.1577)
in a multi-claim combined test. This isolates the same comparison with a single claim tag.

Pre-registered primary discriminator (threshold >= 0.02):

  delta_approach_gap = gap_approach_SPLIT - gap_approach_UNIFIED
  gap_approach = mean(E3.harm_eval at hazard_approach) - mean(E3.harm_eval at none)

PASS criteria (ALL required):
  C1: delta_approach_gap > 0.02  (SPLIT improves approach detection by >= 2pp)
  C2: gap_approach_SPLIT > 0.06  (SPLIT absolute: approach detectable, not just "less bad")
  C3: gap_approach_UNIFIED > 0   (UNIFIED is not flat -- some learning occurred)

Decision scoring:
  retain_ree:       C1+C2+C3 all pass
  hybridize:        C2+C3 pass, delta between 0 and 0.02 (directional but marginal benefit)
  retire_ree_claim: C2 fails (even SPLIT cannot detect approach, or UNIFIED beats SPLIT)
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


EXPERIMENT_TYPE = "v3_exq_078_sd005_split_vs_unified_harm_pair"
CLAIM_IDS = ["SD-005"]


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _run_single(
    seed: int,
    unified: bool,
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

    cond_label = "UNIFIED" if unified else "SPLIT"

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
        reafference_action_dim=0,   # SD-007 disabled -- isolate SD-005
    )
    # SD-005 ablation: unified_latent_mode=True averages z_self and z_world
    config.latent.unified_latent_mode = unified

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

            # Harm buffer: approach AND contact events are positive in both conditions
            # (gradient fields active in both). The question is whether z_world's
            # representation quality differs between SPLIT and UNIFIED.
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
        "unified": unified,
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
    """Discriminative pair: SPLIT (SD-005 implemented) vs UNIFIED (ablation)."""
    results_split:   List[Dict] = []
    results_unified: List[Dict] = []

    for seed in seeds:
        for unified in [False, True]:
            label = "UNIFIED" if unified else "SPLIT"
            print(
                f"\n[V3-EXQ-078] {label} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" alpha_world={alpha_world}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                unified=unified,
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
            if unified:
                results_unified.append(r)
            else:
                results_split.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    gap_approach_split   = _avg(results_split,   "gap_approach")
    gap_approach_unified = _avg(results_unified, "gap_approach")
    gap_contact_split    = _avg(results_split,   "gap_contact")
    gap_contact_unified  = _avg(results_unified, "gap_contact")
    delta_approach_gap   = gap_approach_split - gap_approach_unified

    n_approach_min = min(
        r["n_approach_eval"] for r in results_split + results_unified
    )

    # Pre-registered PASS criteria
    c1_pass = delta_approach_gap    > 0.02
    c2_pass = gap_approach_split    > 0.06
    c3_pass = gap_approach_unified  > 0

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

    print(f"\n[V3-EXQ-078] Final results:", flush=True)
    print(
        f"  gap_approach_SPLIT={gap_approach_split:.4f}"
        f"  gap_approach_UNIFIED={gap_approach_unified:.4f}",
        flush=True,
    )
    print(
        f"  delta_approach_gap={delta_approach_gap:+.4f}"
        f"  gap_contact_SPLIT={gap_contact_split:.4f}"
        f"  gap_contact_UNIFIED={gap_contact_unified:.4f}",
        flush=True,
    )
    print(f"  decision={decision}  status={status} ({criteria_met}/3)", flush=True)

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: delta_approach_gap={delta_approach_gap:.4f} <= 0.02"
            " (split does not improve approach detection over unified by 2pp)"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: gap_approach_SPLIT={gap_approach_split:.4f} <= 0.06"
            " (split condition cannot detect hazard approach at all)"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: gap_approach_UNIFIED={gap_approach_unified:.4f} <= 0"
            " (unified condition cannot detect approach -- both conditions fail)"
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "gap_approach_split":        float(gap_approach_split),
        "gap_approach_unified":      float(gap_approach_unified),
        "gap_contact_split":         float(gap_contact_split),
        "gap_contact_unified":       float(gap_contact_unified),
        "delta_approach_gap":        float(delta_approach_gap),
        "delta_contact_gap":         float(gap_contact_split - gap_contact_unified),
        "mean_score_none_split":     float(_avg(results_split,   "mean_score_none")),
        "mean_score_none_unified":   float(_avg(results_unified, "mean_score_none")),
        "mean_score_approach_split":   float(_avg(results_split,   "mean_score_approach")),
        "mean_score_approach_unified": float(_avg(results_unified, "mean_score_approach")),
        "n_approach_min":            float(n_approach_min),
        "n_seeds":                   float(len(seeds)),
        "alpha_world":               float(alpha_world),
        "proximity_harm_scale":      float(proximity_harm_scale),
        "crit1_pass":                1.0 if c1_pass else 0.0,
        "crit2_pass":                1.0 if c2_pass else 0.0,
        "crit3_pass":                1.0 if c3_pass else 0.0,
        "criteria_met":              float(criteria_met),
    }

    if all_pass:
        interpretation = (
            "SD-005 SUPPORTED: z_self/z_world split improves hazard approach detection."
            f" gap_approach_SPLIT={gap_approach_split:.4f} vs"
            f" UNIFIED={gap_approach_unified:.4f},"
            f" delta={delta_approach_gap:+.4f}."
            " Separate world-state latent (free of motor/body-state dilution) gives E3"
            " cleaner spatial hazard signal. Consistent with EXQ-047 (3.4pp calibration"
            " improvement, 82% attribution improvement)."
        )
    elif c2_pass and c3_pass and delta_approach_gap >= 0:
        interpretation = (
            "Weak positive: SPLIT approach detection > 0.06 (works) and direction is"
            f" correct (delta={delta_approach_gap:+.4f}) but improvement below 2pp threshold."
            " SD-005 split is directionally beneficial; current model/task scale may be"
            " insufficient to reach 2pp margin. Consider larger world_dim or deeper"
            " encoder for stronger separation."
        )
    else:
        interpretation = (
            "SD-005 NOT SUPPORTED: split does not improve or even hurts approach detection"
            f" (delta_approach_gap={delta_approach_gap:+.4f},"
            f" gap_approach_SPLIT={gap_approach_split:.4f})."
            " The averaging in unified mode may not dilute the hazard_field_view signal"
            " enough to matter at current model scale. SD-005 benefit may require larger"
            " architectures, SD-010 (harm stream separation), or different evaluation."
        )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    per_split_rows = "\n".join(
        f"  seed={r['seed']}: gap_approach={r['gap_approach']:.4f}"
        f" gap_contact={r['gap_contact']:.4f}"
        f" n_approach={r['n_approach_eval']}"
        f" train_approach={r['train_approach_events']}"
        for r in results_split
    )
    per_unified_rows = "\n".join(
        f"  seed={r['seed']}: gap_approach={r['gap_approach']:.4f}"
        f" gap_contact={r['gap_contact']:.4f}"
        f" n_approach={r['n_approach_eval']}"
        for r in results_unified
    )

    summary_markdown = (
        f"# V3-EXQ-078 -- SD-005 Split vs Unified Latent: Harm Calibration Pair\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** SD-005\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: delta_approach_gap (SPLIT - UNIFIED) > 0.02\n"
        f"C2: gap_approach_SPLIT > 0.06\n"
        f"C3: gap_approach_UNIFIED > 0\n\n"
        f"## Results\n\n"
        f"| Condition | gap_approach | gap_contact | mean_none | mean_approach |\n"
        f"|-----------|-------------|-------------|-----------|---------------|\n"
        f"| SPLIT   | {gap_approach_split:.4f} | {gap_contact_split:.4f}"
        f" | {_avg(results_split,   'mean_score_none'):.4f}"
        f" | {_avg(results_split,   'mean_score_approach'):.4f} |\n"
        f"| UNIFIED | {gap_approach_unified:.4f} | {gap_contact_unified:.4f}"
        f" | {_avg(results_unified, 'mean_score_none'):.4f}"
        f" | {_avg(results_unified, 'mean_score_approach'):.4f} |\n\n"
        f"**delta_approach_gap (SPLIT - UNIFIED): {delta_approach_gap:+.4f}**\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: delta_approach_gap > 0.02 | {'PASS' if c1_pass else 'FAIL'}"
        f" | {delta_approach_gap:.4f} |\n"
        f"| C2: gap_approach_SPLIT > 0.06 | {'PASS' if c2_pass else 'FAIL'}"
        f" | {gap_approach_split:.4f} |\n"
        f"| C3: gap_approach_UNIFIED > 0  | {'PASS' if c3_pass else 'FAIL'}"
        f" | {gap_approach_unified:.4f} |\n\n"
        f"Criteria met: {criteria_met}/3 -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n\n"
        f"## Per-Seed\n\n"
        f"SPLIT:\n{per_split_rows}\n\n"
        f"UNIFIED:\n{per_unified_rows}\n"
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
            r["n_fatal"] for r in results_split + results_unified
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
    args = parser.parse_args()

    result = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
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
