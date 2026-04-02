#!/opt/local/bin/python3
"""
V3-EXQ-211 -- MECH-153 / ARC-042: Supervised Context Labeling

Claims: MECH-153, ARC-042
Proposal: EVB-0057 (EXP-0058+EXP-0061)

Prior result: EXQ-181b -- cosim(mean_prior_A, mean_prior_B) ~= 0.9999
  ContextMemory priors are identical across hazard-proximate vs hazard-distal
  contexts without supervision. ContextMemory writes/reads slot content
  agnostically, failing to differentiate contextual situations.

MECH-153 asserts:
  ContextMemory in E1 requires supervised labeling (hazard-proximate vs
  hazard-distal) to achieve context-discriminative representations. Without
  supervision, ContextMemory memory slots drift to uniform content, yielding
  near-identical priors regardless of context (cosim ~ 1.0).

ARC-042 asserts:
  E3 ethical machinery is functionally dark until development completes.
  In V3, the proxy: E3's harm evaluation head should show GREATER context-
  sensitivity (stronger hazard_proximate vs distal discrimination) when E1
  ContextMemory is properly labeled, compared to unsupervised training.

Experiment design:
  Two conditions per seed:

    UNSUPERVISED:
      Standard training (same as EXQ-181b).
      E1 ContextMemory trained via E1 prediction loss only.
      Measure context cosim and E3 harm_eval context sensitivity.

    SUPERVISED:
      Add an auxiliary context-label head: Linear(world_dim, 2).
      At each hazard-proximate or hazard-distal step, apply cross-entropy
      loss to force E1 prior (generate_prior output) to predict context type.
      Same E1+E2 losses applied jointly.
      Backprop through generate_prior -> prior_generator -> ContextMemory.memory.

  After training, eval phase:
    Collect E1 priors and E3 harm_eval scores for context A and B.
    Compute:
      context_cosine_similarity   = cosine_sim(mean_prior_A, mean_prior_B)
      context_discrimination_acc  = accuracy of linear probe: context type from prior
      e3_harm_eval_diff           = |mean_harm_eval_A - mean_harm_eval_B|

  Context labels (from EXQ-181b calibration with num_hazards=1, use_proxy_fields=True):
    Context A (proximate):  hazard_field_view.max() > 0.7
    Context B (distal):     hazard_field_view.max() < 0.33
    Field range: [0.22, 1.0] with single hazard in 10x10 grid.

Pre-registered thresholds
--------------------------
C1 (MECH-153): cosim_supervised < cosim_unsupervised in >= 2/3 seeds.
    Supervision reduces context vector similarity.

C2 (MECH-153): cosim_supervised < THRESH_COSIM_SUP in >= 2/3 seeds.
    Supervised cosim falls below absolute threshold (demonstrates
    discrimination emerges from supervision, not just relative reduction).

C3 (ARC-042): e3_harm_eval_diff_supervised > e3_harm_eval_diff_unsupervised
    in >= 2/3 seeds.
    E3 ethical head shows greater context sensitivity when ContextMemory
    is supervised (ARC-042: machinery activates with proper labeling).

C4: n_context_A >= MIN_CONTEXT and n_context_B >= MIN_CONTEXT in all seeds.
    Sanity: sufficient labeled samples for evaluation.

PASS: C1 + C2 + C3 + C4
evidence_direction_per_claim:
  MECH-153: based on C1+C2  (cosim reduction)
  ARC-042:  based on C3     (E3 context sensitivity)

Seeds: [42, 7, 123]
Env:   CausalGridWorldV2 size=10, num_hazards=1, use_proxy_fields=True
Train: 200 warmup episodes x 200 steps per condition
Eval:  50 eval episodes x 200 steps
Estimated runtime: ~120 min (any machine)
"""

import sys
import random
import json
import time
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE = "v3_exq_211_mech153_arc042_supervised_labeling"
CLAIM_IDS = ["MECH-153", "ARC-042"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
THRESH_COSIM_SUP = 0.90    # C2: supervised cosim must be below this
MIN_CONTEXT      = 10      # C4: minimum labeled samples per context

# Context thresholds (calibrated for num_hazards=1, use_proxy_fields=True)
CTX_A_THRESH = 0.70   # hazard_proximate: max > 0.70
CTX_B_THRESH = 0.33   # hazard_distal:   max < 0.33

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WORLD_DIM  = 32
SELF_DIM   = 32

WARMUP_EPISODES = 200
EVAL_EPISODES   = 50
STEPS_PER_EP    = 200

SEEDS = [42, 7, 123]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=1,
        num_resources=3,
        hazard_harm=0.5,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.03,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_config(env: CausalGridWorldV2) -> REEConfig:
    return REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=0,
        novelty_bonus_weight=0.0,
        benefit_eval_enabled=False,
    )


def _get_hazard_max(obs_dict: Dict) -> float:
    """Extract hazard proximity signal (max hazard_field_view)."""
    if "harm_obs" in obs_dict:
        harm_obs = obs_dict["harm_obs"]
        if hasattr(harm_obs, 'shape') and harm_obs.shape[-1] >= 26:
            return float(harm_obs[..., :25].max().item())
    if "hazard_field_view" in obs_dict:
        hfv = obs_dict["hazard_field_view"]
        if hasattr(hfv, 'shape'):
            return float(hfv.max().item())
    return 0.0


def _get_context_label(hazard_max: float) -> Optional[int]:
    """
    Returns 0 (distal/B), 1 (proximate/A), or None (ambiguous).
    """
    if hazard_max > CTX_A_THRESH:
        return 1   # Context A: hazard-proximate
    if hazard_max < CTX_B_THRESH:
        return 0   # Context B: hazard-distal
    return None    # ambiguous zone -- skip


def _cosine_sim_tensor(a: torch.Tensor, b: torch.Tensor) -> float:
    dot = float((a * b).sum().item())
    na  = float(a.norm().item())
    nb  = float(b.norm().item())
    if na < 1e-10 or nb < 1e-10:
        return 1.0   # degenerate: treat as maximally similar (pessimistic)
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Run one condition (SUPERVISED or UNSUPERVISED)
# ---------------------------------------------------------------------------

def _run_condition(
    seed: int,
    condition: str,
    warmup: int,
    n_eval: int,
    steps: int,
) -> Dict:
    supervised = (condition == "SUPERVISED")

    torch.manual_seed(seed)
    random.seed(seed)

    env    = _make_env(seed)
    config = _make_config(env)
    agent  = REEAgent(config)
    action_dim = env.action_dim

    # Supervised: add context label head on E1 prior output
    ctx_head: Optional[nn.Linear] = None
    ctx_optimizer: Optional[optim.Optimizer] = None
    if supervised:
        ctx_head = nn.Linear(WORLD_DIM, 2)
        ctx_optimizer = optim.Adam(
            list(agent.e1.parameters()) + list(ctx_head.parameters()),
            lr=1e-3,
        )

    e1_e2_opt = optim.Adam(
        list(agent.e1.parameters()) + list(agent.e2.parameters()),
        lr=1e-3,
    )

    print(
        f"  [EXQ-211] {condition} seed={seed} warmup={warmup} eval={n_eval}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    agent.train()
    if ctx_head is not None:
        ctx_head.train()

    for ep in range(warmup):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            agent.clock.advance()

            action_idx = random.randint(0, action_dim - 1)
            action_oh  = torch.zeros(1, action_dim)
            action_oh[0, action_idx] = 1.0
            agent._last_action = action_oh

            # E1 + E2 base losses
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss

            if supervised and ctx_head is not None:
                # Supervised context label auxiliary loss
                hazard_max = _get_hazard_max(obs_dict)
                ctx_label  = _get_context_label(hazard_max)
                if ctx_label is not None and latent is not None:
                    total_state = torch.cat(
                        [latent.z_self, latent.z_world], dim=-1
                    )
                    with torch.no_grad() if not total_state.requires_grad else torch.enable_grad():
                        prior = agent.e1.generate_prior(total_state.detach().float())
                    # Train ctx_head to predict context from prior
                    label_t = torch.tensor([ctx_label], dtype=torch.long)
                    ctx_loss = F.cross_entropy(ctx_head(prior.detach().float()), label_t)
                    # Also add cross-entropy signal back through e1 parameters
                    prior_grad = agent.e1.generate_prior(total_state.float())
                    ctx_loss_grad = F.cross_entropy(ctx_head(prior_grad), label_t)
                    total_loss = total_loss + 0.5 * ctx_loss_grad

            if total_loss.requires_grad:
                e1_e2_opt.zero_grad()
                if ctx_optimizer is not None:
                    ctx_optimizer.zero_grad()
                total_loss.backward()
                e1_e2_opt.step()
                if ctx_optimizer is not None:
                    ctx_optimizer.step()

            _, _, done, _, obs_dict = env.step(action_oh)
            if done:
                break

        if (ep + 1) % 100 == 0:
            print(
                f"    [train] {condition} seed={seed} ep {ep+1}/{warmup}",
                flush=True,
            )

    # -----------------------------------------------------------------------
    # Eval: collect priors and E3 harm_eval scores by context
    # -----------------------------------------------------------------------
    agent.eval()
    if ctx_head is not None:
        ctx_head.eval()

    priors_A: List[torch.Tensor] = []
    priors_B: List[torch.Tensor] = []
    harm_eval_A: List[float] = []
    harm_eval_B: List[float] = []

    for ep in range(n_eval):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
                agent.clock.advance()

                total_state = torch.cat([latent.z_self, latent.z_world], dim=-1)
                prior = agent.e1.generate_prior(total_state)  # [1, world_dim]

                # E3 harm eval from z_world
                harm_score = float(
                    agent.e3.harm_eval(latent.z_world).mean().item()
                )

            hazard_max = _get_hazard_max(obs_dict)
            ctx_label  = _get_context_label(hazard_max)

            if ctx_label == 1:
                priors_A.append(prior[0].detach())
                harm_eval_A.append(harm_score)
            elif ctx_label == 0:
                priors_B.append(prior[0].detach())
                harm_eval_B.append(harm_score)

            action_oh = torch.zeros(1, action_dim)
            action_oh[0, random.randint(0, action_dim - 1)] = 1.0
            _, _, done, _, obs_dict = env.step(action_oh)
            if done:
                break

    n_A = len(priors_A)
    n_B = len(priors_B)
    print(
        f"  [{condition}] seed={seed} n_A={n_A} n_B={n_B}",
        flush=True,
    )

    if n_A < MIN_CONTEXT or n_B < MIN_CONTEXT:
        return {
            "seed":            seed,
            "condition":       condition,
            "n_context_A":     n_A,
            "n_context_B":     n_B,
            "context_cosine":  1.0,    # degenerate (not enough samples)
            "e3_harm_eval_diff": 0.0,
            "c4_sanity":       False,
        }

    # -----------------------------------------------------------------------
    # Compute metrics
    # -----------------------------------------------------------------------
    mean_prior_A = torch.stack(priors_A).mean(dim=0)   # [world_dim]
    mean_prior_B = torch.stack(priors_B).mean(dim=0)

    context_cosine = _cosine_sim_tensor(mean_prior_A, mean_prior_B)

    mean_harm_A = sum(harm_eval_A) / max(1, n_A)
    mean_harm_B = sum(harm_eval_B) / max(1, n_B)
    e3_harm_eval_diff = abs(mean_harm_A - mean_harm_B)

    print(
        f"  [{condition}] seed={seed}"
        f" cosim={context_cosine:.4f}"
        f" harm_eval_diff={e3_harm_eval_diff:.4f}"
        f" (harm_A={mean_harm_A:.4f} harm_B={mean_harm_B:.4f})",
        flush=True,
    )

    return {
        "seed":               seed,
        "condition":          condition,
        "n_context_A":        n_A,
        "n_context_B":        n_B,
        "context_cosine":     context_cosine,
        "mean_harm_A":        mean_harm_A,
        "mean_harm_B":        mean_harm_B,
        "e3_harm_eval_diff":  e3_harm_eval_diff,
        "c4_sanity":          True,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ts      = int(time.time())
    root    = Path(__file__).resolve().parents[2]
    out_dir = root / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"

    warmup = 3 if args.dry_run else WARMUP_EPISODES
    n_eval = 3 if args.dry_run else EVAL_EPISODES
    steps  = 20 if args.dry_run else STEPS_PER_EP

    print(f"[EXQ-211] MECH-153/ARC-042 Supervised Context Labeling", flush=True)
    print(f"  dry_run={args.dry_run}", flush=True)

    all_results: List[Dict] = []
    for seed in SEEDS:
        for condition in ["UNSUPERVISED", "SUPERVISED"]:
            res = _run_condition(seed, condition, warmup, n_eval, steps)
            all_results.append(res)

    # -----------------------------------------------------------------------
    # Aggregate per seed
    # -----------------------------------------------------------------------
    per_seed: Dict[int, Dict] = {}
    for r in all_results:
        s = r["seed"]
        if s not in per_seed:
            per_seed[s] = {}
        per_seed[s][r["condition"]] = r

    seed_metrics: List[Dict] = []
    for seed in SEEDS:
        if seed not in per_seed:
            continue
        if "SUPERVISED" not in per_seed[seed] or "UNSUPERVISED" not in per_seed[seed]:
            continue

        sup   = per_seed[seed]["SUPERVISED"]
        unsup = per_seed[seed]["UNSUPERVISED"]

        c1 = sup["context_cosine"] < unsup["context_cosine"]
        c2 = sup["context_cosine"] < THRESH_COSIM_SUP
        c3 = sup["e3_harm_eval_diff"] > unsup["e3_harm_eval_diff"]
        c4 = sup["c4_sanity"] and unsup["c4_sanity"]

        print(
            f"  [EXQ-211] seed={seed}"
            f" cosim_sup={sup['context_cosine']:.4f}"
            f" cosim_unsup={unsup['context_cosine']:.4f}"
            f" harm_diff_sup={sup['e3_harm_eval_diff']:.4f}"
            f" harm_diff_unsup={unsup['e3_harm_eval_diff']:.4f}"
            f" C1={c1} C2={c2} C3={c3} C4={c4}",
            flush=True,
        )

        seed_metrics.append({
            "seed":                     seed,
            "cosim_supervised":         sup["context_cosine"],
            "cosim_unsupervised":       unsup["context_cosine"],
            "supervised_vs_unsupervised_delta": unsup["context_cosine"] - sup["context_cosine"],
            "e3_harm_diff_supervised":  sup["e3_harm_eval_diff"],
            "e3_harm_diff_unsupervised": unsup["e3_harm_eval_diff"],
            "c1": c1,
            "c2": c2,
            "c3": c3,
            "c4": c4,
        })

    # -----------------------------------------------------------------------
    # Global criteria
    # -----------------------------------------------------------------------
    n_seeds  = len(seed_metrics)
    c1_count = sum(1 for m in seed_metrics if m["c1"])
    c2_count = sum(1 for m in seed_metrics if m["c2"])
    c3_count = sum(1 for m in seed_metrics if m["c3"])
    c4_count = sum(1 for m in seed_metrics if m["c4"])

    c1_pass = c1_count >= 2
    c2_pass = c2_count >= 2
    c3_pass = c3_count >= 2
    c4_pass = c4_count >= n_seeds

    # Per-claim evidence direction
    mech153_dir = "supports" if (c1_pass and c2_pass) else ("mixed" if c1_pass else "weakens")
    arc042_dir  = "supports" if c3_pass else "weakens"

    if c1_pass and c2_pass and c3_pass and c4_pass:
        outcome   = "PASS"
        direction = "supports"
    elif c1_pass or c3_pass:
        outcome   = "PARTIAL"
        direction = "mixed"
    else:
        outcome   = "FAIL"
        direction = "weakens"

    def _mean(key: str) -> float:
        return sum(m[key] for m in seed_metrics) / max(1, len(seed_metrics))

    print(
        f"\n[EXQ-211] RESULT: {outcome}"
        f" cosim_sup={_mean('cosim_supervised'):.4f}"
        f" cosim_unsup={_mean('cosim_unsupervised'):.4f}"
        f" e3_diff_sup={_mean('e3_harm_diff_supervised'):.4f}"
        f" C1={c1_pass} C2={c2_pass} C3={c3_pass} C4={c4_pass}",
        flush=True,
    )

    manifest = {
        "run_id":                     f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type":            EXPERIMENT_TYPE,
        "architecture_epoch":         "ree_hybrid_guardrails_v1",
        "claim_ids":                  CLAIM_IDS,
        "experiment_purpose":         EXPERIMENT_PURPOSE,
        "outcome":                    outcome,
        "evidence_direction":         direction,
        "evidence_direction_per_claim": {
            "MECH-153": mech153_dir,
            "ARC-042":  arc042_dir,
        },
        "timestamp":                  ts,
        "dry_run":                    args.dry_run,
        "seeds":                      SEEDS,
        "warmup_episodes":            warmup,
        "eval_episodes":              n_eval,
        "steps_per_episode":          steps,
        "thresh_cosim_sup":           THRESH_COSIM_SUP,
        "ctx_a_thresh":               CTX_A_THRESH,
        "ctx_b_thresh":               CTX_B_THRESH,
        # Aggregate metrics
        "mean_cosim_supervised":      _mean("cosim_supervised"),
        "mean_cosim_unsupervised":    _mean("cosim_unsupervised"),
        "supervised_vs_unsupervised_delta": _mean("supervised_vs_unsupervised_delta"),
        "context_cosine_similarity":  _mean("cosim_supervised"),
        "context_discrimination_accuracy": 0.0,   # TODO: placeholder (not computed in this version)
        "e3_harm_eval_diff_supervised":   _mean("e3_harm_diff_supervised"),
        "e3_harm_eval_diff_unsupervised": _mean("e3_harm_diff_unsupervised"),
        # Criteria
        "c1_cosim_reduced_pass":      c1_pass,
        "c2_cosim_absolute_pass":     c2_pass,
        "c3_e3_sensitivity_pass":     c3_pass,
        "c4_sanity_pass":             c4_pass,
        "c1_count":                   c1_count,
        "c2_count":                   c2_count,
        "c3_count":                   c3_count,
        "c4_count":                   c4_count,
        "n_seeds":                    n_seeds,
        "seed_metrics":               seed_metrics,
        "all_condition_results":      all_results,
    }

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[EXQ-211] Written: {out_path}", flush=True)


if __name__ == "__main__":
    main()
