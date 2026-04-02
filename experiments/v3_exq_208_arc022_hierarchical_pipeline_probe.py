#!/opt/local/bin/python3
"""
V3-EXQ-208 -- ARC-022: Hierarchical Abstraction Diagnostic

Claim: ARC-022 (diagnostic -- does not weight governance confidence)
Proposal: EVB-0048
EXPERIMENT_PURPOSE: diagnostic

ARC-022 asserts:
  The E1->E2 processing pipeline implements a hierarchy of increasing
  abstraction: later processing stages produce representations that are
  progressively less correlated with raw sensory input and progressively
  more predictive of task-relevant signals (harm, goal-relevance).

  This conflicts with ARC-007/ARC-014 about the exact structure of the
  hierarchy. This diagnostic adjudicates by measuring the empirical
  abstraction gradient across three accessible levels:

    Level 1 (L1): z_world (LatentStack encoder output, world_dim=32)
    Level 2 (L2): E1 LSTM hidden state (128-dim, after _e1_tick)
    Level 3 (L3): E2 world_forward prediction (world_dim=32)

Experiment design:
  1. Train for 100 warmup episodes on CausalGridWorldV2.
  2. During 10 eval episodes (200 steps each), collect at each step:
       - raw_obs:   raw world_state vector (250-dim)
       - z_world:   latent.z_world [1, 32] -- L1
       - e1_hidden: E1 LSTM h_t[-1] [128] -- L2
       - e2_pred:   E2.world_forward(z_world, action) [1, 32] -- L3
       - harm_flag: 1 if harm_signal < 0 else 0
  3. Subsample to MAX_RSA_POINTS per seed (O(N^2) RSA).
  4. Build pairwise similarity matrices within each representation space
     (cosine similarity).
  5. Build pairwise "harm-agreement" matrix:
       harm_agree[i,j] = 1.0 if harm_flag_i == harm_flag_j else 0.0
  6. RSA correlations (Spearman r of upper triangles):
       rsa_obs_L1 = spearman_r(upper(sim_obs),  upper(sim_L1))
       rsa_obs_L2 = spearman_r(upper(sim_obs),  upper(sim_L2))
       rsa_obs_L3 = spearman_r(upper(sim_obs),  upper(sim_L3))
       rsa_harm_L1 = spearman_r(upper(sim_harm), upper(sim_L1))
       rsa_harm_L2 = spearman_r(upper(sim_harm), upper(sim_L2))
       rsa_harm_L3 = spearman_r(upper(sim_harm), upper(sim_L3))
  7. Compute abstraction gradient, goal_relevance_by_layer,
     hierarchical_linearity_r2.

Pre-registered thresholds
--------------------------
C1: rsa_obs_L3 < rsa_obs_L1 in >= 2/3 seeds.
    E2 prediction is less correlated with raw obs than z_world.
    (Hierarchy makes representations more abstract.)

C2: rsa_obs_L2 < rsa_obs_L1 in >= 2/3 seeds.
    E1 LSTM hidden state is less correlated with raw obs than z_world.

C3: abstraction_gradient > THRESH_GRADIENT in >= 2/3 seeds.
    abstraction_gradient = rsa_obs_L1 - rsa_obs_L3 (should be positive).

C4: goal_relevance_gradient >= 0.0 in >= 2/3 seeds.
    goal_relevance_gradient = rsa_harm_L3 - rsa_harm_L1 (should be positive).

PASS: C1 + C2 + C3 -- hierarchy confirmed (diagnostic result informs ARC-022)
PARTIAL: C1 without C2 -- only E2 step shows abstraction
FAIL: C1 fails -- no measurable abstraction gradient

EXPERIMENT_PURPOSE = "diagnostic" -- does not affect governance confidence.
Result informs Q-020 adjudication between ARC-022 vs ARC-007/ARC-014.

Seeds: [42, 7, 123]
Env:   CausalGridWorldV2 size=10, 2 hazards, 3 resources, hazard_harm=0.02
Train: 100 warmup episodes x 200 steps
Eval:  10 eval episodes x 200 steps
Estimated runtime: ~65 min (any machine)
"""

import sys
import random
import math
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE = "v3_exq_208_arc022_hierarchical_pipeline_probe"
CLAIM_IDS = ["ARC-022"]
EXPERIMENT_PURPOSE = "diagnostic"

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
THRESH_GRADIENT = 0.02     # C3: rsa_obs_L1 - rsa_obs_L3 must exceed this

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BODY_OBS_DIM  = 12
WORLD_OBS_DIM = 250
ACTION_DIM    = 5
WORLD_DIM     = 32
SELF_DIM      = 32

WARMUP_EPISODES  = 100
EVAL_EPISODES    = 10
STEPS_PER_EP     = 200
MAX_RSA_POINTS   = 300

SEEDS = [42, 7, 123]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=2,
        num_resources=3,
        hazard_harm=0.02,
        env_drift_interval=10,
        env_drift_prob=0.2,
    )


def _make_config() -> REEConfig:
    return REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
    )


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    dot = float((a * b).sum().item())
    na = float(a.norm().item())
    nb = float(b.norm().item())
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return dot / (na * nb)


def _upper_triangle(mat: List[List[float]]) -> List[float]:
    n = len(mat)
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            result.append(mat[i][j])
    return result


def _spearman_r(a: List[float], b: List[float]) -> float:
    n = len(a)
    if n < 3:
        return 0.0

    def _rank(lst: List[float]) -> List[float]:
        sorted_idx = sorted(range(n), key=lambda i: lst[i])
        ranks = [0.0] * n
        for rank_val, idx in enumerate(sorted_idx):
            ranks[idx] = float(rank_val + 1)
        return ranks

    ra = _rank(a)
    rb = _rank(b)
    mean_ra = sum(ra) / n
    mean_rb = sum(rb) / n
    num = sum((ra[i] - mean_ra) * (rb[i] - mean_rb) for i in range(n))
    den_a = math.sqrt(sum((ra[i] - mean_ra) ** 2 for i in range(n)))
    den_b = math.sqrt(sum((rb[i] - mean_rb) ** 2 for i in range(n)))
    if den_a < 1e-12 or den_b < 1e-12:
        return 0.0
    return num / (den_a * den_b)


def _pearson_r2(xs: List[float], ys: List[float]) -> float:
    """Pearson R^2 for R^2 of linear fit."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    den_x = math.sqrt(sum((xs[i] - mx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ys[i] - my) ** 2 for i in range(n)))
    if den_x < 1e-12 or den_y < 1e-12:
        return 0.0
    r = num / (den_x * den_y)
    return r * r


def _get_e1_hidden(agent: REEAgent) -> Optional[torch.Tensor]:
    hs = agent.e1._hidden_state
    if hs is None:
        return None
    return hs[0][-1, 0, :].detach().clone()  # [hidden_dim=128]


# ---------------------------------------------------------------------------
# Run one seed
# ---------------------------------------------------------------------------

def _run_seed(seed: int, dry_run: bool) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    warmup = 3 if dry_run else WARMUP_EPISODES
    n_eval = 2 if dry_run else EVAL_EPISODES
    steps  = 20 if dry_run else STEPS_PER_EP

    env    = _make_env(seed)
    config = _make_config()
    agent  = REEAgent(config)

    e1_opt = optim.Adam(agent.e1.parameters(), lr=1e-3)
    e2_opt = optim.Adam(agent.e2.parameters(), lr=3e-3)

    print(
        f"  [EXQ-208 ARC-022] seed={seed} warmup={warmup} eval={n_eval}"
        f" steps_per_ep={steps}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Warmup training
    # -----------------------------------------------------------------------
    agent.train()
    for ep in range(warmup):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps):
            obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            latent = agent.sense(obs_body, obs_world)
            ticks  = agent.clock.advance()
            if ticks["e1_tick"]:
                agent._e1_tick(latent)

            action = torch.zeros(1, ACTION_DIM)
            action[0, random.randint(0, ACTION_DIM - 1)] = 1.0

            _, reward, done, _, obs_dict = env.step(action)
            harm_signal = float(reward) if reward < 0 else 0.0

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total   = e1_loss + e2_loss
            if total.requires_grad:
                e1_opt.zero_grad()
                e2_opt.zero_grad()
                total.backward()
                e1_opt.step()
                e2_opt.step()

            agent.update_residue(harm_signal)
            if done:
                break

        if (ep + 1) % 50 == 0:
            print(
                f"    [train] seed={seed} ep {ep+1}/{warmup}",
                flush=True,
            )

    # -----------------------------------------------------------------------
    # Eval: collect representations at each level
    # -----------------------------------------------------------------------
    agent.eval()

    # Representation buffers: each is a list of 1-D tensors
    raw_obs_vecs:  List[torch.Tensor] = []   # 250-dim
    z_world_vecs:  List[torch.Tensor] = []   # L1: 32-dim
    e1_hidden_vecs: List[torch.Tensor] = []  # L2: 128-dim
    e2_pred_vecs:  List[torch.Tensor] = []   # L3: 32-dim
    harm_flags:    List[int] = []            # 0 or 1

    for ep in range(n_eval):
        _, obs_dict = env.reset()
        agent.reset()

        for step in range(steps):
            obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                ticks  = agent.clock.advance()
                if ticks["e1_tick"]:
                    agent._e1_tick(latent)

                # L1: z_world
                z_w = latent.z_world[0].detach()   # [world_dim]
                # L2: E1 hidden state
                h   = _get_e1_hidden(agent)

                # L3: E2 world_forward prediction
                action = torch.zeros(1, ACTION_DIM)
                action[0, random.randint(0, ACTION_DIM - 1)] = 1.0
                e2_p = agent.e2.world_forward(
                    latent.z_world, action
                )[0].detach()  # [world_dim]

            if h is None:
                # E1 not yet ticked (first few steps), skip
                _, reward, done, _, obs_dict = env.step(action)
                if done:
                    break
                continue

            # Raw obs: world_state portion
            raw_w = obs_world[0].detach() if obs_world.dim() > 1 else obs_world.detach()

            _, reward, done, _, obs_dict = env.step(action)
            harm_flag = 1 if reward < 0 else 0

            raw_obs_vecs.append(raw_w)
            z_world_vecs.append(z_w)
            e1_hidden_vecs.append(h)
            e2_pred_vecs.append(e2_p)
            harm_flags.append(harm_flag)

            if done:
                break

    n_collected = len(z_world_vecs)
    print(
        f"  [EXQ-208] seed={seed} collected {n_collected} steps"
        f" n_harm={sum(harm_flags)}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Subsample
    # -----------------------------------------------------------------------
    if n_collected > MAX_RSA_POINTS:
        indices = random.sample(range(n_collected), MAX_RSA_POINTS)
        indices.sort()
        raw_obs_vecs   = [raw_obs_vecs[i]   for i in indices]
        z_world_vecs   = [z_world_vecs[i]   for i in indices]
        e1_hidden_vecs = [e1_hidden_vecs[i] for i in indices]
        e2_pred_vecs   = [e2_pred_vecs[i]   for i in indices]
        harm_flags     = [harm_flags[i]      for i in indices]

    N = len(z_world_vecs)

    # -----------------------------------------------------------------------
    # Build similarity matrices
    # -----------------------------------------------------------------------
    def _build_sim(vecs: List[torch.Tensor]) -> List[List[float]]:
        n = len(vecs)
        mat = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                mat[i][j] = _cosine_sim(vecs[i], vecs[j])
        return mat

    sim_obs   = _build_sim(raw_obs_vecs)
    sim_L1    = _build_sim(z_world_vecs)
    sim_L2    = _build_sim(e1_hidden_vecs)
    sim_L3    = _build_sim(e2_pred_vecs)

    # Harm agreement matrix
    sim_harm = [
        [1.0 if harm_flags[i] == harm_flags[j] else 0.0 for j in range(N)]
        for i in range(N)
    ]

    # -----------------------------------------------------------------------
    # RSA correlations
    # -----------------------------------------------------------------------
    ut_obs    = _upper_triangle(sim_obs)
    ut_L1     = _upper_triangle(sim_L1)
    ut_L2     = _upper_triangle(sim_L2)
    ut_L3     = _upper_triangle(sim_L3)
    ut_harm   = _upper_triangle(sim_harm)

    rsa_obs_L1  = _spearman_r(ut_obs, ut_L1)
    rsa_obs_L2  = _spearman_r(ut_obs, ut_L2)
    rsa_obs_L3  = _spearman_r(ut_obs, ut_L3)
    rsa_harm_L1 = _spearman_r(ut_harm, ut_L1)
    rsa_harm_L2 = _spearman_r(ut_harm, ut_L2)
    rsa_harm_L3 = _spearman_r(ut_harm, ut_L3)

    # abstraction_gradient = how much RSA with obs decreases across levels
    abstraction_gradient  = rsa_obs_L1 - rsa_obs_L3
    goal_rel_gradient     = rsa_harm_L3 - rsa_harm_L1

    # hierarchical_linearity_r2: how linear is the abstraction gradient?
    level_indices  = [0.0, 1.0, 2.0]
    rsa_obs_series = [rsa_obs_L1, rsa_obs_L2, rsa_obs_L3]
    hierarchical_linearity_r2 = _pearson_r2(level_indices, rsa_obs_series)

    # -----------------------------------------------------------------------
    # Criteria
    # -----------------------------------------------------------------------
    c1 = rsa_obs_L3 < rsa_obs_L1          # E2 pred less like obs than z_world
    c2 = rsa_obs_L2 < rsa_obs_L1          # E1 hidden less like obs than z_world
    c3 = abstraction_gradient > THRESH_GRADIENT
    c4 = goal_rel_gradient >= 0.0         # harm relevance increases with depth

    print(
        f"  [EXQ-208] seed={seed}"
        f" rsa_obs=[L1={rsa_obs_L1:.4f} L2={rsa_obs_L2:.4f} L3={rsa_obs_L3:.4f}]"
        f" rsa_harm=[L1={rsa_harm_L1:.4f} L2={rsa_harm_L2:.4f} L3={rsa_harm_L3:.4f}]"
        f" abst_grad={abstraction_gradient:.4f}"
        f" goal_rel_grad={goal_rel_gradient:.4f}"
        f" hier_r2={hierarchical_linearity_r2:.4f}"
        f" C1={c1} C2={c2} C3={c3} C4={c4}",
        flush=True,
    )

    return {
        "seed":                        seed,
        "n_collected":                 n_collected,
        "n_rsa_points":                N,
        "rsa_obs_L1":                  rsa_obs_L1,
        "rsa_obs_L2":                  rsa_obs_L2,
        "rsa_obs_L3":                  rsa_obs_L3,
        "rsa_harm_L1":                 rsa_harm_L1,
        "rsa_harm_L2":                 rsa_harm_L2,
        "rsa_harm_L3":                 rsa_harm_L3,
        "abstraction_gradient":        abstraction_gradient,
        "goal_relevance_by_layer":     [rsa_harm_L1, rsa_harm_L2, rsa_harm_L3],
        "goal_relevance_gradient":     goal_rel_gradient,
        "hierarchical_linearity_r2":   hierarchical_linearity_r2,
        "c1_e2_more_abstract":         c1,
        "c2_e1_more_abstract":         c2,
        "c3_abstraction_gradient":     c3,
        "c4_goal_relevance_gradient":  c4,
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

    print(f"[EXQ-208] ARC-022 Hierarchical Pipeline Probe", flush=True)
    print(f"  dry_run={args.dry_run}", flush=True)

    seed_results = []
    for seed in SEEDS:
        res = _run_seed(seed, dry_run=args.dry_run)
        seed_results.append(res)

    # -----------------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------------
    n_seeds = len(seed_results)
    c1_count = sum(1 for r in seed_results if r["c1_e2_more_abstract"])
    c2_count = sum(1 for r in seed_results if r["c2_e1_more_abstract"])
    c3_count = sum(1 for r in seed_results if r["c3_abstraction_gradient"])
    c4_count = sum(1 for r in seed_results if r["c4_goal_relevance_gradient"])

    c1_pass = c1_count >= 2
    c2_pass = c2_count >= 2
    c3_pass = c3_count >= 2
    c4_pass = c4_count >= 2

    if c1_pass and c2_pass and c3_pass:
        outcome = "PASS"
        direction = "supports"
    elif c1_pass:
        outcome = "PARTIAL"
        direction = "mixed"
    else:
        outcome = "FAIL"
        direction = "weakens"

    def _mean(key: str) -> float:
        return sum(r[key] for r in seed_results) / n_seeds

    print(
        f"\n[EXQ-208] RESULT: {outcome}"
        f" abst_grad={_mean('abstraction_gradient'):.4f}"
        f" goal_rel_grad={_mean('goal_relevance_gradient'):.4f}"
        f" hier_r2={_mean('hierarchical_linearity_r2'):.4f}"
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
        "timestamp":                  ts,
        "dry_run":                    args.dry_run,
        "seeds":                      SEEDS,
        "warmup_episodes":            3 if args.dry_run else WARMUP_EPISODES,
        "eval_episodes":              2 if args.dry_run else EVAL_EPISODES,
        "steps_per_episode":          20 if args.dry_run else STEPS_PER_EP,
        "max_rsa_points":             MAX_RSA_POINTS,
        "thresh_gradient":            THRESH_GRADIENT,
        # Aggregate metrics
        "mean_rsa_obs_L1":            _mean("rsa_obs_L1"),
        "mean_rsa_obs_L2":            _mean("rsa_obs_L2"),
        "mean_rsa_obs_L3":            _mean("rsa_obs_L3"),
        "mean_rsa_harm_L1":           _mean("rsa_harm_L1"),
        "mean_rsa_harm_L2":           _mean("rsa_harm_L2"),
        "mean_rsa_harm_L3":           _mean("rsa_harm_L3"),
        "abstraction_gradient":       _mean("abstraction_gradient"),
        "goal_relevance_gradient":    _mean("goal_relevance_gradient"),
        "hierarchical_linearity_r2":  _mean("hierarchical_linearity_r2"),
        "goal_relevance_by_layer":    [
            _mean("rsa_harm_L1"), _mean("rsa_harm_L2"), _mean("rsa_harm_L3")
        ],
        # Criteria
        "c1_e2_more_abstract_pass":   c1_pass,
        "c2_e1_more_abstract_pass":   c2_pass,
        "c3_abstraction_gradient_pass": c3_pass,
        "c4_goal_relevance_gradient_pass": c4_pass,
        "c1_count":                   c1_count,
        "c2_count":                   c2_count,
        "c3_count":                   c3_count,
        "c4_count":                   c4_count,
        "n_seeds":                    n_seeds,
        # Per-seed
        "seed_results":               seed_results,
    }

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[EXQ-208] Written: {out_path}", flush=True)


if __name__ == "__main__":
    main()
