#!/opt/local/bin/python3
"""
EXQ-181b: E1 Prior Context Discrimination Diagnostic (Context-B Fix)

Supersedes EXQ-181 (v3_exq_181_e1_prior_context_discrimination.py).

Root cause of EXQ-181 Context-B sampling failure:
  EXQ-181 classified Context B as hazard_field_view.max() < 0.1. With num_hazards=4
  in a 10x10 grid, each hazard emits a proximity-weighted hazard field with
  hazard_field_decay=0.5. At this density, the hazard signal is ambient across
  the entire grid -- virtually every cell has hazard_field_view.max() > 0.1.
  Result: n_context_B ~ 0 across all seeds, C1/C2 unevaluable, FAIL trivially.

Fix (two-part):
  (1) num_hazards=1 -- single hazard creates spatial gradient rather than ambient field.
  (2) Revised thresholds -- CausalGridWorldV2 with use_proxy_fields=True normalises the
      field so even the most distant cell never drops below ~0.22 (empirically confirmed:
      min=0.22 across 793 steps with num_hazards=1). Correct calibration:
        Context A: max > 0.7  (strongly hazard-proximate, ~27% of steps)
        Context B: max < 0.33 (hazard-distal, bottom quartile, ~25% of steps)
      Original thresholds (0.3/0.1) assumed raw field values, not the normalised output.
      The revised thresholds produce equivalent context discrimination while respecting
      the actual field value range. Scientific question is unchanged.

Hypothesis (unchanged from EXQ-181): E1.generate_prior() output is contextually
  discriminative -- priors differ between hazard-proximate and hazard-distal z_world
  contexts, and predict harm gradient linearly.

Pass criteria:
  C1 PASS: cosine_sim(mean_prior_A, mean_prior_B) < 0.85 (unchanged)
  C2 PASS: ridge regression R^2 > 0.3 (unchanged)
  Context thresholds (revised for proxy-field scale):
    Context A: hazard_field_view.max() > 0.7  (was > 0.3 in EXQ-181)
    Context B: hazard_field_view.max() < 0.33 (was < 0.1 in EXQ-181)
  Context coverage: n_A >= 10 AND n_B >= 10 per seed (otherwise SKIP that seed)
  PASS if both pass in >= 2/3 seeds; MIXED if one passes; FAIL if neither.

Claim IDs: MECH-150, ARC-041
Supersedes: V3-EXQ-181
"""

import sys
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


EXPERIMENT_TYPE = "v3_exq_181b_sd016_context_separation_fix"
CLAIM_IDS = ["MECH-150", "ARC-041"]


# ------------------------------------------------------------------ #
# Helpers (identical to EXQ-181)                                       #
# ------------------------------------------------------------------ #

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def get_hazard_max(obs_dict: Dict, world_obs: Optional[torch.Tensor]) -> float:
    if "harm_obs" in obs_dict:
        harm_obs = obs_dict["harm_obs"]
        if hasattr(harm_obs, 'shape') and harm_obs.shape[-1] >= 26:
            return float(harm_obs[..., :25].max().item())
    if "hazard_field_view" in obs_dict:
        hfv = obs_dict["hazard_field_view"]
        if hasattr(hfv, 'shape'):
            return float(hfv.max().item())
    if world_obs is not None and world_obs.shape[-1] >= 225:
        return float(world_obs[..., 200:225].max().item())
    return 0.0


def get_harm_scalar(obs_dict: Dict) -> float:
    if "harm_obs" in obs_dict:
        harm_obs = obs_dict["harm_obs"]
        if hasattr(harm_obs, 'shape') and harm_obs.shape[-1] > 50:
            return float(harm_obs[..., 50].mean().item())
    return 0.0


def ridge_r2(
    X: torch.Tensor,
    y: torch.Tensor,
    lam: float = 1e-3,
    test_frac: float = 0.2,
) -> float:
    N = X.shape[0]
    if N < 8:
        return 0.0
    n_test = max(1, int(N * test_frac))
    X_tr, y_tr = X[n_test:], y[n_test:]
    X_te, y_te = X[:n_test], y[:n_test]
    if X_tr.shape[0] < 4:
        return 0.0
    Xb = torch.cat([X_tr, torch.ones(X_tr.shape[0], 1, dtype=X_tr.dtype)], dim=1)
    XtX = Xb.T @ Xb + lam * torch.eye(Xb.shape[1], dtype=Xb.dtype)
    Xty = Xb.T @ y_tr
    try:
        w = torch.linalg.solve(XtX, Xty)
    except Exception:
        return 0.0
    Xb_te = torch.cat([X_te, torch.ones(X_te.shape[0], 1, dtype=X_te.dtype)], dim=1)
    y_pred = Xb_te @ w
    ss_res = ((y_te - y_pred) ** 2).sum()
    ss_tot = ((y_te - y_te.mean()) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-9))


# ------------------------------------------------------------------ #
# Single-seed runner                                                   #
# ------------------------------------------------------------------ #

def _run_seed(
    seed: int,
    warmup_episodes: int,
    collect_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    lr: float,
    alpha_world: float,
    alpha_self: float,
    num_hazards: int,
) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    # KEY CHANGE vs EXQ-181: (1) num_hazards=1 (was 4) -- single hazard creates
    # spatial gradient instead of ambient field. (2) Context thresholds revised:
    # Context A > 0.7 (strongly proximate), Context B < 0.33 (distal, bottom quartile).
    # With use_proxy_fields=True, field values range [0.22, 1.0] -- original thresholds
    # (0.3/0.1) were incompatible with this normalised range, causing n_B ~ 0.
    env = CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=num_hazards,
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

    action_dim = env.action_dim

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=0,
        novelty_bonus_weight=0.0,
        benefit_eval_enabled=False,
    )

    agent = REEAgent(config)
    optimizer = optim.Adam(agent.parameters(), lr=lr)

    # ---- Phase 0: Training warmup ----
    agent.train()
    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            agent.clock.advance()

            action_idx = random.randint(0, action_dim - 1)
            action_oh  = _action_to_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh

            _, harm_signal, done, info, obs_dict = env.step(action_oh)

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

        if (ep + 1) % 20 == 0 or ep == warmup_episodes - 1:
            print(
                f"  [train] seed={seed} ep {ep+1}/{warmup_episodes}",
                flush=True,
            )

    # ---- Phase 1: Context collection ----
    agent.eval()

    priors_A: List[torch.Tensor] = []
    priors_B: List[torch.Tensor] = []
    harm_A:   List[float]        = []
    harm_B:   List[float]        = []
    n_skipped = 0

    for ep in range(collect_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            agent.clock.advance()

            z_self_det  = latent.z_self.detach()
            z_world_det = latent.z_world.detach()
            total_state = torch.cat([z_self_det, z_world_det], dim=-1)

            with torch.no_grad():
                prior = agent.e1.generate_prior(total_state)

            hazard_max = get_hazard_max(obs_dict, obs_world if obs_world.dim() >= 1 else None)
            harm_val   = get_harm_scalar(obs_dict)

            # Context thresholds revised for use_proxy_fields=True field scale.
            # Empirical distribution (num_hazards=1): min=0.22, p25=0.33, p75=1.0.
            # Context A: strongly hazard-proximate (top ~27% of steps).
            # Context B: hazard-distal (bottom quartile, ~25% of steps).
            if hazard_max > 0.7:
                priors_A.append(prior.squeeze(0).cpu())
                harm_A.append(harm_val)
            elif hazard_max < 0.33:
                priors_B.append(prior.squeeze(0).cpu())
                harm_B.append(harm_val)
            else:
                n_skipped += 1

            action_idx = random.randint(0, action_dim - 1)
            action_oh  = _action_to_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh
            _, _, done, _, obs_dict = env.step(action_oh)

            if done:
                break

    n_A = len(priors_A)
    n_B = len(priors_B)

    print(
        f"  [collect] seed={seed}"
        f" n_context_A={n_A}"
        f" n_context_B={n_B}"
        f" n_skipped={n_skipped}",
        flush=True,
    )

    # ---- Phase 2: Metrics ----
    crit1_pass = False
    crit2_pass = False
    cos_sim_val = float("nan")
    r2_val = 0.0

    if n_A >= 10 and n_B >= 10:
        mean_A = torch.stack(priors_A).mean(dim=0)
        mean_B = torch.stack(priors_B).mean(dim=0)
        cos_sim_val = float(
            F.cosine_similarity(mean_A.unsqueeze(0), mean_B.unsqueeze(0)).item()
        )
        crit1_pass = cos_sim_val < 0.85
        print(
            f"  [C1] seed={seed}"
            f" cosine_sim_AB={cos_sim_val:.4f}"
            f" threshold=0.85"
            f" pass={'YES' if crit1_pass else 'NO'}",
            flush=True,
        )
    else:
        print(
            f"  [C1] seed={seed} SKIP -- insufficient context steps"
            f" (n_A={n_A}, n_B={n_B}, need >=10 each)",
            flush=True,
        )

    all_priors = priors_A + priors_B
    all_harms  = harm_A + harm_B
    if len(all_priors) >= 8:
        X = torch.stack(all_priors).float()
        y = torch.tensor(all_harms, dtype=torch.float32)
        r2_val = ridge_r2(X, y)
        crit2_pass = r2_val > 0.3
        print(
            f"  [C2] seed={seed}"
            f" harm_r2={r2_val:.4f}"
            f" threshold=0.3"
            f" pass={'YES' if crit2_pass else 'NO'}",
            flush=True,
        )
    else:
        print(
            f"  [C2] seed={seed} SKIP -- insufficient samples (n={len(all_priors)})",
            flush=True,
        )

    return {
        "seed": seed,
        "n_context_A": n_A,
        "n_context_B": n_B,
        "n_skipped": n_skipped,
        "cos_sim_AB": cos_sim_val if cos_sim_val == cos_sim_val else None,
        "harm_r2": float(r2_val),
        "crit1_pass": int(crit1_pass),
        "crit2_pass": int(crit2_pass),
    }


# ------------------------------------------------------------------ #
# Main run                                                             #
# ------------------------------------------------------------------ #

def run(
    seeds: Optional[List[int]] = None,
    warmup_episodes: int = 100,
    collect_episodes: int = 100,
    dry_run: bool = False,
    num_hazards: int = 1,
) -> Dict:
    if seeds is None:
        seeds = [42, 7, 11]

    steps_per_episode = 200
    self_dim  = 32
    world_dim = 32
    lr        = 1e-3
    alpha_world = 0.9
    alpha_self  = 0.3

    per_seed_results: List[Dict] = []

    for seed in seeds:
        print(
            f"\n[V3-EXQ-181b] seed={seed}"
            f" warmup={warmup_episodes}"
            f" collect={collect_episodes}"
            f" num_hazards={num_hazards}"
            f" alpha_world={alpha_world}",
            flush=True,
        )
        r = _run_seed(
            seed=seed,
            warmup_episodes=warmup_episodes,
            collect_episodes=collect_episodes,
            steps_per_episode=steps_per_episode,
            self_dim=self_dim,
            world_dim=world_dim,
            lr=lr,
            alpha_world=alpha_world,
            alpha_self=alpha_self,
            num_hazards=num_hazards,
        )
        per_seed_results.append(r)

    seeds_c1_pass = sum(r["crit1_pass"] for r in per_seed_results)
    seeds_c2_pass = sum(r["crit2_pass"] for r in per_seed_results)
    n_seeds       = len(per_seed_results)

    cos_vals = [
        r["cos_sim_AB"] for r in per_seed_results
        if r["cos_sim_AB"] is not None
    ]
    avg_cos_sim = float(sum(cos_vals) / len(cos_vals)) if cos_vals else float("nan")
    avg_harm_r2 = float(
        sum(r["harm_r2"] for r in per_seed_results) / max(1, n_seeds)
    )
    total_n_A = sum(r["n_context_A"] for r in per_seed_results)
    total_n_B = sum(r["n_context_B"] for r in per_seed_results)

    crit1_pass_global = seeds_c1_pass >= (n_seeds + 1) // 2
    crit2_pass_global = seeds_c2_pass >= (n_seeds + 1) // 2

    if crit1_pass_global and crit2_pass_global:
        status = "PASS"
        evidence_direction = "supports"
    elif crit1_pass_global or crit2_pass_global:
        status = "MIXED"
        evidence_direction = "inconclusive"
    else:
        status = "FAIL"
        evidence_direction = "does_not_support"

    if crit1_pass_global:
        c1_interp = (
            f"C1 PASS: cosine_sim={avg_cos_sim:.4f} < 0.85 in {seeds_c1_pass}/{n_seeds} seeds."
            " E1 priors are contextually distinct between hazard-proximate and hazard-distal states."
        )
    else:
        c1_interp = (
            f"C1 FAIL: cosine_sim={avg_cos_sim:.4f} >= 0.85 (passed {seeds_c1_pass}/{n_seeds} seeds)."
            " E1 priors do NOT differ between hazard-proximate and hazard-distal contexts."
            " SD-016 cue-indexed retrieval will require supervised ContextMemory training."
        )

    if crit2_pass_global:
        c2_interp = (
            f"C2 PASS: harm_r2={avg_harm_r2:.4f} > 0.3 in {seeds_c2_pass}/{n_seeds} seeds."
            " E1 prior predicts harm gradient -- cue context carries harm information."
        )
    else:
        c2_interp = (
            f"C2 FAIL: harm_r2={avg_harm_r2:.4f} <= 0.3 (passed {seeds_c2_pass}/{n_seeds} seeds)."
            " E1 prior does NOT reliably predict harm gradient."
            " SD-016 extract_cue_context() will need terrain_loss supervision to calibrate."
        )

    per_seed_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" n_A={r['n_context_A']}"
        f" n_B={r['n_context_B']}"
        f" cos_sim={r['cos_sim_AB'] if r['cos_sim_AB'] is not None else 'N/A'}"
        f" harm_r2={r['harm_r2']:.4f}"
        f" C1={'PASS' if r['crit1_pass'] else 'FAIL'}"
        f" C2={'PASS' if r['crit2_pass'] else 'FAIL'}"
        for r in per_seed_results
    )

    cos_display = f"{avg_cos_sim:.4f}" if avg_cos_sim == avg_cos_sim else "N/A"

    summary_markdown = (
        f"# V3-EXQ-181b -- E1 Prior Context Discrimination (Context-B Fix)\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-150, ARC-041\n"
        f"**Supersedes:** V3-EXQ-181\n"
        f"**Seeds:** {seeds}\n\n"
        f"## Context\n\n"
        f"EXQ-181 failed because Context B sampling (hazard_field_view.max() < 0.1) never"
        f" fired with num_hazards=4: the hazard field is ambient everywhere in a 10x10 grid"
        f" with 4 sources. Fix: num_hazards={num_hazards} produces genuine spatial context"
        f" separation -- cells near the single hazard (Context A) vs far from it (Context B).\n\n"
        f"## Design\n\n"
        f"**Phase 0 (warmup):** {warmup_episodes} episodes x 200 steps. Standard REEAgent"
        f" training (E1 + E2 losses, Adam lr=1e-3, alpha_world=0.9).\n\n"
        f"**Phase 1 (collection):** {collect_episodes} episodes x 200 steps. Random actions."
        f" Context A: hazard_field_view.max() > 0.7. Context B: < 0.33."
        f" (Thresholds revised from EXQ-181 0.3/0.1 for proxy-field normalised range.)\n\n"
        f"**Phase 2 (metrics):** C1 = cosine_sim(mean_prior_A, mean_prior_B) < 0.85."
        f" C2 = ridge R^2 of prior -> harm_scalar > 0.3.\n\n"
        f"## Key Results\n\n"
        f"| Metric | Value | Threshold |\n"
        f"|---|---|---|\n"
        f"| cosine_sim(mean_A, mean_B) | {cos_display} | < 0.85 |\n"
        f"| harm_r2 (prior -> harm) | {avg_harm_r2:.4f} | > 0.3 |\n"
        f"| n_context_A steps (total) | {total_n_A} | -- |\n"
        f"| n_context_B steps (total) | {total_n_B} | -- |\n\n"
        f"## Pass Criteria\n\n"
        f"| Criterion | Result | Seeds passing |\n"
        f"|---|---|---|\n"
        f"| C1: cosine_sim < 0.85 | {'PASS' if crit1_pass_global else 'FAIL'}"
        f" | {seeds_c1_pass}/{n_seeds} |\n"
        f"| C2: harm_r2 > 0.3 | {'PASS' if crit2_pass_global else 'FAIL'}"
        f" | {seeds_c2_pass}/{n_seeds} |\n\n"
        f"PASS rule: both criteria pass in >= 2/3 seeds -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{c1_interp}\n\n"
        f"{c2_interp}\n\n"
        f"## Per-Seed Results\n\n"
        f"{per_seed_rows}\n"
    )

    metrics = {
        "cosine_sim_AB":      float(avg_cos_sim) if avg_cos_sim == avg_cos_sim else -1.0,
        "harm_r2":            float(avg_harm_r2),
        "n_context_A_steps":  int(total_n_A),
        "n_context_B_steps":  int(total_n_B),
        "crit1_pass":         int(crit1_pass_global),
        "crit2_pass":         int(crit2_pass_global),
        "seeds_c1_pass":      int(seeds_c1_pass),
        "seeds_c2_pass":      int(seeds_c2_pass),
        "n_seeds":            int(n_seeds),
        "num_hazards":        int(num_hazards),
    }

    print(f"\n[V3-EXQ-181b] Final results:", flush=True)
    print(f"  cosine_sim_AB={cos_display}  harm_r2={avg_harm_r2:.4f}", flush=True)
    print(f"  n_A={total_n_A}  n_B={total_n_B}", flush=True)
    print(f"  C1: {seeds_c1_pass}/{n_seeds} seeds pass  C2: {seeds_c2_pass}/{n_seeds} seeds pass", flush=True)
    print(f"  {c1_interp}", flush=True)
    print(f"  {c2_interp}", flush=True)
    print(f"  status={status}", flush=True)

    return {
        "status":             status,
        "metrics":            metrics,
        "summary_markdown":   summary_markdown,
        "claim_ids":          CLAIM_IDS,
        "evidence_direction": evidence_direction,
        "experiment_type":    EXPERIMENT_TYPE,
        "fatal_error_count":  0,
    }


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(
        description="EXQ-181b: E1 prior context discrimination (num_hazards=1 fix)"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 11])
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--collect", type=int, default=100)
    parser.add_argument("--num-hazards", type=int, default=1,
                        help="Number of hazards (default 1; was 4 in EXQ-181)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Smoke test: 1 seed, 3 warmup, 5 collect. No output file.")
    args = parser.parse_args()

    if args.dry_run:
        args.seeds   = [42]
        args.warmup  = 3
        args.collect = 5
        print("[DRY-RUN] 1 seed, 3 warmup, 5 collect, num_hazards=1", flush=True)

    result = run(
        seeds=args.seeds,
        warmup_episodes=args.warmup,
        collect_episodes=args.collect,
        dry_run=args.dry_run,
        num_hazards=args.num_hazards,
    )

    if args.dry_run:
        print("\n[DRY-RUN] DRY-RUN complete. Not writing output file.", flush=True)
        print(f"Status: {result['status']}", flush=True)
        for k, v in result["metrics"].items():
            print(f"  {k}: {v}", flush=True)
        sys.exit(0)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"]      = ts
    result["claim"]              = CLAIM_IDS[0]
    result["verdict"]            = result["status"]
    result["run_id"]             = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["supersedes"]         = "v3_exq_181_e1_prior_context_discrimination"

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
