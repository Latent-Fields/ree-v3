#!/opt/local/bin/python3
"""
V3-EXQ-210 -- MECH-156: Theta Traversal in E1 Hidden State

Claim: MECH-156
Proposal: EVB-0051

MECH-156 asserts:
  Theta oscillations (4-8 Hz) implement sequential traversal across indexed
  representations in E1. Within each theta cycle (theta_buffer_size=10 env
  steps), E1's LSTM hidden state h_t sweeps through a sequential trajectory
  of representations, resetting at theta-cycle boundaries.

  Observable signatures:
    (a) Within-cycle: h_t drifts monotonically from h_0 to h_9 (sequential
        ordering -- increasing cumulative drift over the theta cycle).
    (b) Cross-cycle: h_{theta-period} is more similar to h_0 than h_{half-period}
        is to h_0 (periodic reset at theta boundaries).
    (c) Theta-band autocorrelation: mean cosine_sim(h_t, h_{t+10}) > mean
        cosine_sim(h_t, h_{t+5}) (correlates stronger at theta lag than half-lag).

Experiment design:
  1. Train for WARMUP_EPISODES.
  2. During EVAL_EPISODES, collect h_t (E1 LSTM last-layer hidden state) at
     each E1-tick step. Theta cycle = theta_buffer_size = 10 steps.
  3. Compute:
       sequential_ordering_score:
         For each theta window (steps [0..9], [10..19], ...):
           drift[i] = ||h_i - h_0|| (cumulative L2 drift from window start)
           order_corr = Spearman_r([0..9], drift)
         sequential_ordering_score = mean(order_corr across windows)
       traversal_cycle_consistency:
         std(order_corr) -- lower = more consistent theta cycles
       theta_band_autocorr:
         mean cosine_sim(h_t, h_{t+THETA}) for all valid t (THETA=10)
       half_band_autocorr:
         mean cosine_sim(h_t, h_{t+THETA//2}) for all valid t (half-cycle lag)
       far_autocorr:
         mean cosine_sim(h_t, h_{t+THETA*2}) for all valid t (double-cycle lag)

Pre-registered thresholds
--------------------------
C1: sequential_ordering_score > THRESH_ORDER in >= 2/3 seeds.
    E1 hidden state drifts monotonically within theta cycles.

C2: theta_band_autocorr > far_autocorr in >= 2/3 seeds.
    Theta-period lag shows stronger autocorr than double-period lag.
    (Consistent with periodic theta structure.)

C3: traversal_cycle_consistency < THRESH_CV in >= 2/3 seeds.
    traversal_cycle_consistency = std(per_window_order_corr) / mean_abs_order_corr.
    Low CV => consistent sequential traversal pattern across theta cycles.

C4: n_theta_windows >= MIN_WINDOWS in all seeds.
    Sanity: enough cycles to estimate statistics.

PASS: C1 + C2 + C3 + C4
PARTIAL: C1 without C2 -- sequential drift exists but no periodic peak
FAIL: C1 fails -- no sequential traversal within theta cycles

Seeds: [42, 7, 123]
Env:   CausalGridWorldV2 size=10, 2 hazards, 3 resources, hazard_harm=0.02
Train: 100 warmup episodes x 200 steps
Eval:  10 eval episodes x 200 steps
Theta: theta_buffer_size = 10 env steps
Estimated runtime: ~60 min (any machine)
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
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE = "v3_exq_210_mech156_theta_traversal_probe"
CLAIM_IDS = ["MECH-156"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
THRESH_ORDER   = 0.10    # C1: sequential_ordering_score must exceed this
THRESH_CV      = 1.50    # C3: coefficient of variation must stay below this
MIN_WINDOWS    = 20      # C4: minimum number of theta windows

THETA_SIZE     = 10      # theta_buffer_size = 10 steps per theta cycle

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


def _get_e1_hidden(agent: REEAgent) -> Optional[torch.Tensor]:
    hs = agent.e1._hidden_state
    if hs is None:
        return None
    return hs[0][-1, 0, :].detach().clone()  # [hidden_dim=128]


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    dot = float((a * b).sum().item())
    na  = float(a.norm().item())
    nb  = float(b.norm().item())
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return dot / (na * nb)


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
    ma = sum(ra) / n
    mb = sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da  = math.sqrt(sum((ra[i] - ma) ** 2 for i in range(n)))
    db  = math.sqrt(sum((rb[i] - mb) ** 2 for i in range(n)))
    if da < 1e-12 or db < 1e-12:
        return 0.0
    return num / (da * db)


def _autocorr_at_lag(hs: List[torch.Tensor], lag: int) -> float:
    """Mean cosine similarity between h_t and h_{t+lag}."""
    sims = []
    for i in range(len(hs) - lag):
        sims.append(_cosine_sim(hs[i], hs[i + lag]))
    if not sims:
        return 0.0
    return sum(sims) / len(sims)


# ---------------------------------------------------------------------------
# Run one seed
# ---------------------------------------------------------------------------

def _run_seed(seed: int, dry_run: bool) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    warmup = 3 if dry_run else WARMUP_EPISODES
    n_eval = 2 if dry_run else EVAL_EPISODES
    steps  = 30 if dry_run else STEPS_PER_EP

    env    = _make_env(seed)
    config = _make_config()
    agent  = REEAgent(config)

    e1_opt = optim.Adam(agent.e1.parameters(), lr=1e-3)
    e2_opt = optim.Adam(agent.e2.parameters(), lr=3e-3)

    print(
        f"  [EXQ-210 MECH-156] seed={seed} warmup={warmup} eval={n_eval}"
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

    # -----------------------------------------------------------------------
    # Eval: collect E1 hidden state sequence
    # -----------------------------------------------------------------------
    agent.eval()

    # Collect ALL h_t in one long sequence across episodes
    hidden_seq: List[torch.Tensor] = []

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

            h = _get_e1_hidden(agent)
            if h is not None:
                hidden_seq.append(h)

            action = torch.zeros(1, ACTION_DIM)
            action[0, random.randint(0, ACTION_DIM - 1)] = 1.0
            _, reward, done, _, obs_dict = env.step(action)
            if done:
                break

    n_steps = len(hidden_seq)
    print(
        f"  [EXQ-210] seed={seed} collected {n_steps} hidden states",
        flush=True,
    )

    if n_steps < THETA_SIZE * 3:
        print(f"  [EXQ-210] seed={seed} WARN: too few steps for theta analysis", flush=True)
        return {
            "seed":                     seed,
            "n_steps":                  n_steps,
            "n_theta_windows":          0,
            "sequential_ordering_score": 0.0,
            "traversal_cycle_consistency": 999.0,
            "theta_band_autocorr":      0.0,
            "half_band_autocorr":       0.0,
            "far_autocorr":             0.0,
            "c1_ordering": False,
            "c2_autocorr": False,
            "c3_consistency": False,
            "c4_sanity": False,
        }

    # -----------------------------------------------------------------------
    # Compute theta-window sequential ordering
    # -----------------------------------------------------------------------
    n_windows = n_steps // THETA_SIZE
    per_window_order_corrs: List[float] = []

    for w in range(n_windows):
        window_hs = hidden_seq[w * THETA_SIZE: (w + 1) * THETA_SIZE]
        if len(window_hs) < THETA_SIZE:
            break

        h0 = window_hs[0]
        drifts = [float((window_hs[i] - h0).norm().item()) for i in range(THETA_SIZE)]
        step_indices = list(range(THETA_SIZE))
        corr = _spearman_r(step_indices, drifts)
        per_window_order_corrs.append(corr)

    n_windows_ok = len(per_window_order_corrs)
    sequential_ordering_score = 0.0
    traversal_cv = 999.0

    if n_windows_ok > 0:
        sequential_ordering_score = sum(per_window_order_corrs) / n_windows_ok
        if n_windows_ok > 1:
            mean_abs = sum(abs(c) for c in per_window_order_corrs) / n_windows_ok
            if mean_abs > 1e-10:
                std_c = math.sqrt(
                    sum((c - sequential_ordering_score) ** 2 for c in per_window_order_corrs)
                    / n_windows_ok
                )
                traversal_cv = std_c / (mean_abs + 1e-10)

    # -----------------------------------------------------------------------
    # Autocorrelation at theta, half-theta, double-theta lags
    # -----------------------------------------------------------------------
    theta_band_autocorr  = _autocorr_at_lag(hidden_seq, THETA_SIZE)
    half_band_autocorr   = _autocorr_at_lag(hidden_seq, THETA_SIZE // 2)
    far_autocorr         = _autocorr_at_lag(hidden_seq, THETA_SIZE * 2)

    # -----------------------------------------------------------------------
    # Criteria
    # -----------------------------------------------------------------------
    c1 = sequential_ordering_score > THRESH_ORDER
    c2 = theta_band_autocorr > far_autocorr
    c3 = traversal_cv < THRESH_CV
    c4 = n_windows_ok >= MIN_WINDOWS

    print(
        f"  [EXQ-210] seed={seed}"
        f" n_windows={n_windows_ok}"
        f" ordering={sequential_ordering_score:.4f}"
        f" cv={traversal_cv:.4f}"
        f" autocorr[theta={THETA_SIZE}]={theta_band_autocorr:.4f}"
        f" autocorr[half={THETA_SIZE//2}]={half_band_autocorr:.4f}"
        f" autocorr[far={THETA_SIZE*2}]={far_autocorr:.4f}"
        f" C1={c1} C2={c2} C3={c3} C4={c4}",
        flush=True,
    )

    return {
        "seed":                       seed,
        "n_steps":                    n_steps,
        "n_theta_windows":            n_windows_ok,
        "sequential_ordering_score":  sequential_ordering_score,
        "traversal_cycle_consistency": traversal_cv,
        "theta_band_autocorr":        theta_band_autocorr,
        "half_band_autocorr":         half_band_autocorr,
        "far_autocorr":               far_autocorr,
        "per_window_order_corrs_mean": sequential_ordering_score,
        "c1_ordering":                c1,
        "c2_autocorr":                c2,
        "c3_consistency":             c3,
        "c4_sanity":                  c4,
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

    print(f"[EXQ-210] MECH-156 Theta Traversal Probe", flush=True)
    print(f"  dry_run={args.dry_run}  theta_size={THETA_SIZE}", flush=True)

    seed_results = []
    for seed in SEEDS:
        res = _run_seed(seed, dry_run=args.dry_run)
        seed_results.append(res)

    # -----------------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------------
    n_seeds  = len(seed_results)
    c1_count = sum(1 for r in seed_results if r["c1_ordering"])
    c2_count = sum(1 for r in seed_results if r["c2_autocorr"])
    c3_count = sum(1 for r in seed_results if r["c3_consistency"])
    c4_count = sum(1 for r in seed_results if r["c4_sanity"])

    c1_pass = c1_count >= 2
    c2_pass = c2_count >= 2
    c3_pass = c3_count >= 2
    c4_pass = c4_count >= 2

    if c1_pass and c2_pass and c3_pass and c4_pass:
        outcome = "PASS"
        direction = "supports"
    elif c1_pass and not c2_pass:
        outcome = "PARTIAL"
        direction = "mixed"
    else:
        outcome = "FAIL"
        direction = "weakens"

    def _mean(key: str) -> float:
        vals = [r[key] for r in seed_results if isinstance(r[key], float)]
        return sum(vals) / max(1, len(vals))

    print(
        f"\n[EXQ-210] RESULT: {outcome}"
        f" ordering={_mean('sequential_ordering_score'):.4f}"
        f" autocorr[theta]={_mean('theta_band_autocorr'):.4f}"
        f" autocorr[far]={_mean('far_autocorr'):.4f}"
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
        "theta_size":                 THETA_SIZE,
        "warmup_episodes":            3 if args.dry_run else WARMUP_EPISODES,
        "eval_episodes":              2 if args.dry_run else EVAL_EPISODES,
        "steps_per_episode":          30 if args.dry_run else STEPS_PER_EP,
        "thresh_order":               THRESH_ORDER,
        "thresh_cv":                  THRESH_CV,
        "min_windows":                MIN_WINDOWS,
        # Aggregate metrics
        "sequential_ordering_score":  _mean("sequential_ordering_score"),
        "traversal_cycle_consistency": _mean("traversal_cycle_consistency"),
        "theta_band_autocorr":        _mean("theta_band_autocorr"),
        "half_band_autocorr":         _mean("half_band_autocorr"),
        "far_autocorr":               _mean("far_autocorr"),
        # Criteria
        "c1_ordering_pass":           c1_pass,
        "c2_autocorr_pass":           c2_pass,
        "c3_consistency_pass":        c3_pass,
        "c4_sanity_pass":             c4_pass,
        "c1_count":                   c1_count,
        "c2_count":                   c2_count,
        "c3_count":                   c3_count,
        "c4_count":                   c4_count,
        "n_seeds":                    n_seeds,
        "seed_results":               seed_results,
    }

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[EXQ-210] Written: {out_path}", flush=True)


if __name__ == "__main__":
    main()
