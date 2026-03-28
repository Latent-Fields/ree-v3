"""
V3-EXQ-107 -- ARC-024 Gradient vs Flat Proxy Structure Discriminative Test

Claim: ARC-024 -- "Harm and benefit signals have asymptotic proxy structure
in world latent space."

Design: Two conditions matched on seeds.
  GRADIENT -- use_proxy_fields=True: agent observes continuous hazard_field
              and resource_field proximity gradients in world_obs (dim=250,
              body_obs=12). Produces hazard_approach transition events when
              agent moves within proximity_approach_threshold of a hazard.

  FLAT     -- use_proxy_fields=False: agent observes only binary contact
              signals in world_obs (dim=200, body_obs=10). No hazard_approach
              events; harm only at direct contact.

Key question: After equivalent training, does z_world in the GRADIENT condition
encode proximity information BEFORE contact? If so, E3.harm_eval should predict
higher harm scores at hazard_approach transitions vs neutral movement. This
directly tests ARC-024: z_world must have asymptotic proxy structure -- not just
binary contact-triggered responses.

Evaluation: At each eval step, record E3.harm_eval(z_world) paired with the
transition_type from the current obs (i.e. what happened at the previous step).
  - hazard_approach: agent just moved near a hazard (GRADIENT only)
  - agent_caused_hazard / env_caused_hazard: contact harm
  - none: neutral

C1 (REQUIRED): gradient_gap_approach_none_mean >= 0.08
    mean(harm_eval | hazard_approach) - mean(harm_eval | none) >= 0.08.
    Confirms z_world encodes pre-contact gradient structure.

C2 (REQUIRED): gradient_harm_eval_auc_mean >= 0.60
    AUC(harm_eval; approach+contact vs none) in GRADIENT condition.
    Confirms graded prediction quality.

C3 (SUPPORTING): gradient_harm_eval_auc_mean > flat_harm_eval_auc_mean + 0.05
    GRADIENT outperforms FLAT on z_world harm discriminability.

PASS = C1 AND C2.
If C1 fails: ARC-024 not supported at current training scale.
If C1 passes but C2 fails: gradient propagates but harm_eval insufficient.
"""

import sys
import math
import random
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_107_arc024_gradient_vs_flat_structure"
CLAIM_IDS = ["ARC-024"]

# Pre-registered thresholds
C1_THRESHOLD = 0.08   # gap_approach_none in GRADIENT condition
C2_THRESHOLD = 0.60   # harm_eval AUC in GRADIENT condition
C3_ADVANTAGE = 0.05   # GRADIENT_auc > FLAT_auc + this (supporting, not required)

SEEDS = [0, 1, 2, 3, 4]
CONDITIONS = ["GRADIENT", "FLAT"]

WARMUP_EPISODES = 200
EVAL_EPISODES   = 50
STEPS_PER_EP    = 60

SELF_DIM    = 32
WORLD_DIM   = 32
LR          = 1e-3
ALPHA_WORLD = 0.9   # SD-008: must not use 0.3 default
ALPHA_SELF  = 0.3

MAX_BUF = 2000


def _auc(scores_pos: List[float], scores_neg: List[float]) -> float:
    """Rank-based AUC (Wilcoxon-Mann-Whitney). O(n*m)."""
    if not scores_pos or not scores_neg:
        return 0.5
    total = len(scores_pos) * len(scores_neg)
    wins = sum(1 for p in scores_pos for n in scores_neg if p > n)
    ties = sum(1 for p in scores_pos for n in scores_neg if p == n)
    return (wins + 0.5 * ties) / total


def _action_to_onehot(action_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, action_idx] = 1.0
    return v


def _run_single(
    seed: int,
    condition: str,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_ep: int,
) -> Dict:
    """Train and eval one (seed, condition) cell. Returns metrics dict."""
    torch.manual_seed(seed)
    random.seed(seed)

    use_proxy = (condition == "GRADIENT")

    env = CausalGridWorldV2(
        seed=seed,
        size=6,
        num_hazards=4,
        num_resources=5,
        hazard_harm=0.5,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.03,
        hazard_field_decay=0.5,
        resource_field_decay=0.5,
        use_proxy_fields=use_proxy,
        resource_respawn_on_consume=False,
    )

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=ALPHA_WORLD,
        alpha_self=ALPHA_SELF,
    )
    config.latent.unified_latent_mode = False  # SD-005 split

    agent = REEAgent(config)

    std_params      = [p for n, p in agent.named_parameters() if "harm_eval_head" not in n]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    optimizer        = optim.Adam(std_params, lr=LR)
    harm_eval_opt    = optim.Adam(harm_eval_params, lr=1e-4)

    buf_pos: List[torch.Tensor] = []
    buf_neg: List[torch.Tensor] = []

    # ------------------------------------------------------------------ TRAIN
    agent.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        cached_harm_score = None
        last_action_idx   = None

        for _ in range(steps_per_ep):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            latent = agent.sense(obs_body, obs_world)
            ticks  = agent.clock.advance()

            if ticks["e1_tick"]:
                agent._e1_tick(latent)

            if ticks["e3_tick"] or cached_harm_score is None:
                with torch.no_grad():
                    cached_harm_score = float(
                        agent.e3.harm_eval(latent.z_world.detach()).item()
                    )

            # Policy: flee if harm score high, else random exploration
            if cached_harm_score is not None and cached_harm_score > 0.5:
                action_idx = random.randint(0, env.action_dim - 1)
            else:
                action_idx = random.randint(0, env.action_dim - 1)
            last_action_idx = action_idx

            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            _, harm_signal, done, _, obs_dict = env.step(action)

            # Harm replay buffer
            z_w = latent.z_world.detach()
            if float(harm_signal) < 0:
                buf_pos.append(z_w)
                if len(buf_pos) > MAX_BUF:
                    buf_pos = buf_pos[-MAX_BUF:]
            else:
                buf_neg.append(z_w)
                if len(buf_neg) > MAX_BUF:
                    buf_neg = buf_neg[-MAX_BUF:]

            # E1 + E2 loss
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            # E3 harm_eval balanced training (MSE vs 0/1 labels)
            if len(buf_pos) >= 4 and len(buf_neg) >= 4:
                k_pos = min(16, len(buf_pos))
                k_neg = min(16, len(buf_neg))
                pi  = torch.randperm(len(buf_pos))[:k_pos].tolist()
                ni  = torch.randperm(len(buf_neg))[:k_neg].tolist()
                zw_b = torch.cat([buf_pos[i] for i in pi] + [buf_neg[i] for i in ni], dim=0)
                tgt  = torch.cat([
                    torch.ones( k_pos, 1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                ], dim=0)
                loss_h = F.mse_loss(agent.e3.harm_eval(zw_b), tgt)
                if loss_h.requires_grad:
                    harm_eval_opt.zero_grad()
                    loss_h.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5
                    )
                    harm_eval_opt.step()

            if done:
                break

        if (ep + 1) % 50 == 0 or ep == warmup_episodes - 1:
            print(
                f"  [train] cond={condition} seed={seed}"
                f" ep {ep+1}/{warmup_episodes}"
                f" buf_pos={len(buf_pos)} buf_neg={len(buf_neg)}",
                flush=True,
            )

    # ------------------------------------------------------------------ EVAL
    # At each step, sense the new obs and record harm_eval paired with the
    # transition_type from info (= what happened during the preceding step).
    # This correctly pairs z_world state AFTER the transition with its type:
    #   hazard_approach: agent just moved near a hazard (GRADIENT only)
    #   contact:         agent_caused_hazard or env_caused_hazard
    #   none:            no event
    agent.eval()

    approach_scores: List[float] = []
    contact_scores:  List[float] = []
    none_scores:     List[float] = []

    for _ in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        prev_ttype = "none"  # no preceding transition at episode start

        for _ in range(steps_per_ep):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                ticks  = agent.clock.advance()
                if ticks["e1_tick"]:
                    agent._e1_tick(latent)
                # Fresh harm_eval at every step for accurate recording
                score = float(agent.e3.harm_eval(latent.z_world.detach()).item())

            # Record harm_eval of the state reached by prev_ttype transition
            if prev_ttype == "hazard_approach":
                approach_scores.append(score)
            elif prev_ttype in ("agent_caused_hazard", "env_caused_hazard"):
                contact_scores.append(score)
            elif prev_ttype == "none":
                none_scores.append(score)

            # Policy: random actions for maximum coverage of transition types
            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            _, _, done, step_info, obs_dict = env.step(action)
            prev_ttype = step_info.get("transition_type", "none")

            if done:
                break

    # Compute metrics
    n_approach = len(approach_scores)
    n_contact  = len(contact_scores)
    n_none     = len(none_scores)

    if n_approach >= 3 and n_none >= 3:
        gap_approach_none = (
            sum(approach_scores) / n_approach - sum(none_scores) / n_none
        )
    else:
        gap_approach_none = float("nan")

    # AUC: harm events (approach+contact GRADIENT; contact only FLAT) vs none
    harm_scores_all = approach_scores + contact_scores if use_proxy else contact_scores
    harm_eval_auc = _auc(harm_scores_all, none_scores) if harm_scores_all and none_scores else 0.5

    print(
        f"  [eval]  cond={condition} seed={seed}"
        f" n_approach={n_approach} n_contact={n_contact} n_none={n_none}"
        f" gap={gap_approach_none:.4f}"
        f" auc={harm_eval_auc:.4f}",
        flush=True,
    )

    return {
        "seed":              seed,
        "condition":         condition,
        "gap_approach_none": gap_approach_none,
        "harm_eval_auc":     harm_eval_auc,
        "n_approach":        n_approach,
        "n_contact":         n_contact,
        "n_none":            n_none,
    }


def run(
    warmup_episodes: int = WARMUP_EPISODES,
    eval_episodes:   int = EVAL_EPISODES,
    steps_per_ep:    int = STEPS_PER_EP,
    seeds: Optional[List[int]] = None,
    **kwargs,
) -> dict:
    """
    Discriminative pair: GRADIENT (use_proxy_fields=True) vs FLAT (use_proxy_fields=False).
    5 seeds for statistical power. PASS = C1 AND C2.
    """
    if seeds is None:
        seeds = list(SEEDS)

    all_results: List[Dict] = []
    for seed in seeds:
        for cond in CONDITIONS:
            print(
                f"\n[V3-EXQ-107] {cond} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" steps={steps_per_ep}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                condition=cond,
                warmup_episodes=warmup_episodes,
                eval_episodes=eval_episodes,
                steps_per_ep=steps_per_ep,
            )
            all_results.append(r)

    grad_results = [r for r in all_results if r["condition"] == "GRADIENT"]
    flat_results = [r for r in all_results if r["condition"] == "FLAT"]

    def _mean_finite(vals: List[float]) -> float:
        finite = [v for v in vals if not math.isnan(v)]
        return sum(finite) / len(finite) if finite else float("nan")

    def _sem_finite(vals: List[float], mean: float) -> float:
        finite = [v for v in vals if not math.isnan(v)]
        n = len(finite)
        if n < 2:
            return float("nan")
        var = sum((v - mean) ** 2 for v in finite) / (n - 1)
        return math.sqrt(var / n)

    grad_gaps = [r["gap_approach_none"] for r in grad_results]
    grad_aucs = [r["harm_eval_auc"]     for r in grad_results]
    flat_aucs = [r["harm_eval_auc"]     for r in flat_results]

    grad_gap_mean = _mean_finite(grad_gaps)
    grad_auc_mean = _mean_finite(grad_aucs)
    flat_auc_mean = _mean_finite(flat_aucs)
    grad_gap_sem  = _sem_finite(grad_gaps, grad_gap_mean)
    grad_auc_sem  = _sem_finite(grad_aucs, grad_auc_mean)
    flat_auc_sem  = _sem_finite(flat_aucs, flat_auc_mean)

    if not math.isnan(grad_auc_mean) and not math.isnan(flat_auc_mean):
        discriminative_advantage = grad_auc_mean - flat_auc_mean
    else:
        discriminative_advantage = float("nan")

    c1_pass = (not math.isnan(grad_gap_mean)) and grad_gap_mean >= C1_THRESHOLD
    c2_pass = (not math.isnan(grad_auc_mean)) and grad_auc_mean >= C2_THRESHOLD
    c3_pass = (not math.isnan(discriminative_advantage)) and discriminative_advantage >= C3_ADVANTAGE

    status = "PASS" if (c1_pass and c2_pass) else "FAIL"

    gap_str  = f"{grad_gap_mean:.4f}" if not math.isnan(grad_gap_mean) else "nan"
    gauc_str = f"{grad_auc_mean:.4f}" if not math.isnan(grad_auc_mean) else "nan"
    fauc_str = f"{flat_auc_mean:.4f}" if not math.isnan(flat_auc_mean) else "nan"
    disc_str = f"{discriminative_advantage:.4f}" if not math.isnan(discriminative_advantage) else "nan"
    gsem_str = f"{grad_gap_sem:.4f}"  if not math.isnan(grad_gap_sem)  else "nan"
    gasem_str = f"{grad_auc_sem:.4f}" if not math.isnan(grad_auc_sem)  else "nan"
    fasem_str = f"{flat_auc_sem:.4f}" if not math.isnan(flat_auc_sem)  else "nan"

    print(f"\n[V3-EXQ-107] Final results:", flush=True)
    print(
        f"  C1: gradient gap_approach_none={gap_str} +/- {gsem_str}"
        f" (need >={C1_THRESHOLD}) -> {'PASS' if c1_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"  C2: gradient harm_eval_auc={gauc_str} +/- {gasem_str}"
        f" (need >={C2_THRESHOLD}) -> {'PASS' if c2_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"  C3 (supporting): disc_advantage={disc_str}"
        f" (need >={C3_ADVANTAGE}) -> {'PASS' if c3_pass else 'FAIL'}",
        flush=True,
    )
    print(f"  FLAT harm_eval_auc={fauc_str} +/- {fasem_str}", flush=True)
    print(f"  Status: {status}", flush=True)

    # Failure notes
    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: gradient gap_approach_none={gap_str} < {C1_THRESHOLD}"
            " -- z_world does not encode pre-contact gradient structure"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: gradient harm_eval_auc={gauc_str} < {C2_THRESHOLD}"
            " -- E3.harm_eval not sufficiently discriminative with gradient obs"
        )

    # Interpretation
    if status == "PASS":
        interpretation = (
            f"ARC-024 SUPPORTED: z_world encodes asymptotic proxy structure."
            f" Pre-contact gradient signal confirmed"
            f" (gap_approach_none={gap_str} >= {C1_THRESHOLD})."
            f" Harm_eval AUC={gauc_str} >= {C2_THRESHOLD}."
            f" Discriminative advantage vs FLAT: {disc_str}."
        )
    elif not c1_pass:
        interpretation = (
            f"ARC-024 NOT SUPPORTED: z_world fails to encode pre-contact gradient."
            f" gap_approach_none={gap_str} < {C1_THRESHOLD}."
            f" Gradient fields in obs do not propagate into z_world latent structure"
            f" at current training scale ({warmup_episodes} episodes)."
        )
    else:
        interpretation = (
            f"ARC-024 PARTIAL: pre-contact gap confirmed (C1={gap_str})"
            f" but harm_eval AUC below threshold (C2={gauc_str} < {C2_THRESHOLD})."
            f" Gradient propagates into z_world but E3.harm_eval is not yet"
            f" sufficiently discriminative at current training scale."
        )

    n_grad_approach = sum(r["n_approach"] for r in grad_results)
    n_grad_contact  = sum(r["n_contact"]  for r in grad_results)
    n_flat_contact  = sum(r["n_contact"]  for r in flat_results)
    n_none          = sum(r["n_none"]     for r in grad_results)

    summary_markdown = (
        f"# V3-EXQ-107 -- ARC-024 Gradient vs Flat Proxy Structure\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** ARC-024\n"
        f"**Seeds:** {seeds}\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps"
        f"  **Steps/ep:** {steps_per_ep}\n\n"
        f"## Design\n\n"
        f"Two conditions matched on seeds:\n"
        f"  GRADIENT: use_proxy_fields=True -- body_obs=12, world_obs=250,\n"
        f"            includes hazard_field_view and resource_field_view.\n"
        f"  FLAT:     use_proxy_fields=False -- body_obs=10, world_obs=200,\n"
        f"            binary contact harm only, no proximity gradient.\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1 (pre-contact gradient): gap_approach_none >= {C1_THRESHOLD} (GRADIENT)\n"
        f"C2 (AUC quality):          harm_eval_auc >= {C2_THRESHOLD} (GRADIENT)\n"
        f"C3 (discriminative):       GRADIENT_auc > FLAT_auc + {C3_ADVANTAGE} (supporting)\n"
        f"PASS: C1 AND C2\n\n"
        f"## Results\n\n"
        f"| Metric | GRADIENT | FLAT |\n"
        f"|--------|----------|------|\n"
        f"| gap_approach_none (mean +/- sem) | {gap_str} +/- {gsem_str} | N/A |\n"
        f"| harm_eval_auc (mean +/- sem)     | {gauc_str} +/- {gasem_str}"
        f" | {fauc_str} +/- {fasem_str} |\n\n"
        f"| Criterion | Result | Value |\n"
        f"|-----------|--------|-------|\n"
        f"| C1: gap >= {C1_THRESHOLD}            | {'PASS' if c1_pass else 'FAIL'} | {gap_str} |\n"
        f"| C2: AUC >= {C2_THRESHOLD}            | {'PASS' if c2_pass else 'FAIL'} | {gauc_str} |\n"
        f"| C3: disc >= {C3_ADVANTAGE} (support) | {'PASS' if c3_pass else 'FAIL'} | {disc_str} |\n\n"
        f"**PASS = C1 AND C2 -> {status}**\n\n"
        f"## Event Counts (total across seeds)\n\n"
        f"GRADIENT: approach={n_grad_approach} contact={n_grad_contact} none={n_none}\n"
        f"FLAT:     contact={n_flat_contact}\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n"
    )
    if failure_notes:
        summary_markdown += "\n## Failure Notes\n\n" + "\n".join(
            f"- {fn}" for fn in failure_notes
        ) + "\n"

    def _safe_float(v: float) -> float:
        return float(v) if not math.isnan(v) else -99.0

    def _safe_float_sem(v: float) -> float:
        return float(v) if not math.isnan(v) else -1.0

    return {
        "status":             status,
        "claim_ids":          CLAIM_IDS,
        "experiment_type":    EXPERIMENT_TYPE,
        "evidence_direction": "supports" if status == "PASS" else "weakens",
        "fatal_error_count":  0,
        "summary_markdown":   summary_markdown,
        "metrics": {
            "gradient_gap_approach_none_mean": _safe_float(grad_gap_mean),
            "gradient_gap_approach_none_sem":  _safe_float_sem(grad_gap_sem),
            "gradient_harm_eval_auc_mean":     _safe_float(grad_auc_mean),
            "gradient_harm_eval_auc_sem":      _safe_float_sem(grad_auc_sem),
            "flat_harm_eval_auc_mean":         _safe_float(flat_auc_mean),
            "flat_harm_eval_auc_sem":          _safe_float_sem(flat_auc_sem),
            "discriminative_auc_advantage":    _safe_float(discriminative_advantage),
            "c1_threshold":                    float(C1_THRESHOLD),
            "c2_threshold":                    float(C2_THRESHOLD),
            "c3_advantage":                    float(C3_ADVANTAGE),
            "c1_pass":                         1.0 if c1_pass else 0.0,
            "c2_pass":                         1.0 if c2_pass else 0.0,
            "c3_pass":                         1.0 if c3_pass else 0.0,
            "n_seeds":                         float(len(seeds)),
            "n_grad_approach_total":           float(n_grad_approach),
            "n_grad_contact_total":            float(n_grad_contact),
            "n_none_total":                    float(n_none),
            "warmup_episodes":                 float(warmup_episodes),
            "eval_episodes":                   float(eval_episodes),
        },
        "per_seed": all_results,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int,          default=WARMUP_EPISODES)
    parser.add_argument("--eval",   type=int,          default=EVAL_EPISODES)
    parser.add_argument("--steps",  type=int,          default=STEPS_PER_EP)
    parser.add_argument("--seeds",  type=int, nargs="+", default=list(SEEDS))
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Quick check: 5 warmup, 5 eval, 20 steps, seed 0 only",
    )
    args = parser.parse_args()

    if args.smoke_test:
        warmup, eval_eps, steps, seeds = 5, 5, 20, [0]
        print("[V3-EXQ-107] SMOKE TEST MODE", flush=True)
    else:
        warmup, eval_eps, steps, seeds = args.warmup, args.eval, args.steps, args.seeds

    result = run(
        warmup_episodes=warmup,
        eval_episodes=eval_eps,
        steps_per_ep=steps,
        seeds=seeds,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"]      = ts
    result["run_id"]             = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    if args.smoke_test:
        print(f"\n[SMOKE] Status: {result['status']}", flush=True)
        for k, v in result["metrics"].items():
            print(f"  {k}: {v}", flush=True)
        print("[SMOKE] Script ran without error. No file written.", flush=True)
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
