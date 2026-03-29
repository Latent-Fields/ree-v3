#!/opt/local/bin/python3
"""
V3-EXQ-170 -- Q-002: What is the appropriate spatial resolution for R(x,t)?

Claim:    Q-002
Subject:  astrocyte.R_field_resolution

Q-002 asks whether the residue field needs fine vs coarse spatial granularity
for effective harm navigation.  This experiment tests two inline RBF-based residue
field implementations that differ only in the number of spatial centers:

  FINE   (num_rbf_centers=32): dense coverage of z_world latent space.
  COARSE (num_rbf_centers=4):  sparse coverage of z_world latent space.

Both conditions use the same lightweight architecture (WorldEncoder + RBFLayer +
greedy avoidance policy).  The encoder is trained on a random-action warmup phase
(200 episodes) so that z_world carries spatial content; the residue field
accumulates harm exposure at the nearest RBF center during warmup.  The eval phase
(100 episodes) then uses greedy avoidance: for each candidate action, predict the
next z_world with a learned forward model and score it via the residue field; take
the action with the lowest predicted residue score (tie-break randomly).

The discriminative question:
  Does a FINE field (32 centers) match or outperform a COARSE field (4 centers)
  on harm rate during eval?  Does the FINE field show at least weak positive
  correlation between residue scores and actual hazard proximity (residue_accuracy)?

This directly addresses Q-002's open question about spatial granularity.

Pre-registered thresholds
--------------------------
C1: FINE harm_rate <= COARSE harm_rate + THRESH_C1_MARGIN (both seeds).
    FINE should match or beat COARSE -- fine granularity should not hurt.
    THRESH_C1_MARGIN = 0.05 (within 5 pp; fine must not be strictly worse).

C2: FINE mean_residue_accuracy > THRESH_C2_ACCURACY (both seeds).
    residue_accuracy = Pearson r(residue_score_at_position, hazard_proximity_at_position)
    measured over eval episode steps.
    THRESH_C2_ACCURACY = 0.1 (any positive correlation demonstrates spatial learning).

C3: COARSE n_harm_events >= THRESH_C3_MIN_EVENTS (both seeds, data quality gate).
    THRESH_C3_MIN_EVENTS = 5 (must observe enough harm events to measure harm rate).

PASS: C1 + C2 + C3 (both seeds each) -- fine resolution matches or beats coarse,
      and fine field shows spatial learning.
FAIL: any criterion fails.

evidence_direction: "supports" if PASS, "weakens" if C1+C3 but NOT C2 (fine no
spatial learning), "mixed" otherwise.

Architecture (inline, no REEAgent):
  WorldEncoder:    Linear(250, 32) + LayerNorm -> z_world (dim=32)
  RBFLayer:        N RBF centers in z_world space
    forward(z):    softmax(-||z - c_i||^2 / bandwidth) @ accumulation_values -> scalar
    accumulate(z, harm_val): accumulation_values[nearest] += harm_val * ACCUM_ALPHA
  E2WorldForward:  Linear(32+5, 32) -> z_world_next (for greedy action selection)
  Encoder+E2 trained jointly on world prediction during warmup.

Protocol:
  2 conditions x 2 seeds x 300 total episodes (200 warmup + 100 eval) x 100 steps/ep.
  Estimated runtime: 2 x 2 x 300 x 0.005 min/ep = ~6 min (lightweight CPU).
"""

import sys
import random
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ree_core.environment.causal_grid_world import CausalGridWorldV2


EXPERIMENT_TYPE = "v3_exq_170_q002_r_field_resolution_pair"
CLAIM_IDS = ["Q-002"]

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
THRESH_C1_MARGIN       = 0.05   # C1: FINE harm_rate <= COARSE + this margin
THRESH_C2_ACCURACY     = 0.10   # C2: FINE residue_accuracy (Pearson r) > this
THRESH_C3_MIN_EVENTS   = 5      # C3: COARSE n_harm_events >= this (data quality)

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------
WARMUP_EPISODES     = 200
EVAL_EPISODES       = 100
TOTAL_EPISODES      = WARMUP_EPISODES + EVAL_EPISODES
STEPS_PER_EPISODE   = 100
LR                  = 3e-4
ACCUM_ALPHA         = 0.2   # residue accumulation rate at nearest center
RBF_BANDWIDTH       = 1.0   # RBF kernel bandwidth in z_world space
N_CANDIDATES        = 5     # number of candidate actions for greedy eval

FINE_NUM_CENTERS    = 32
COARSE_NUM_CENTERS  = 4

SEEDS               = [42, 123]
CONDITIONS          = ["FINE", "COARSE"]

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
WORLD_OBS_DIM   = 250   # CausalGridWorldV2 size=8 world_state dim
ACTION_DIM      = 5
WORLD_DIM       = 32


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class WorldEncoder(nn.Module):
    def __init__(self, obs_dim: int, world_dim: int):
        super().__init__()
        self.linear = nn.Linear(obs_dim, world_dim)
        self.norm   = nn.LayerNorm(world_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(F.relu(self.linear(x)))


class E2WorldForward(nn.Module):
    """Predict next z_world given current z_world and one-hot action."""
    def __init__(self, world_dim: int, action_dim: int):
        super().__init__()
        self.fc = nn.Linear(world_dim + action_dim, world_dim)

    def forward(self, z_world: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([z_world, action], dim=-1))


class RBFLayer(nn.Module):
    """
    Lightweight residue field: N RBF centers in z_world space.

    accumulation_values[i] tracks the accumulated harm signal near center i.
    forward() returns a scalar score = softmax(-distances / bw) @ acc_values.
    accumulate() updates the nearest center's value by ACCUM_ALPHA * harm_val.
    """
    def __init__(self, num_centers: int, world_dim: int, bandwidth: float):
        super().__init__()
        self.num_centers = num_centers
        self.bandwidth   = bandwidth
        # Centers are a fixed parameter (not trained -- populated lazily)
        self.centers            = nn.Parameter(torch.randn(num_centers, world_dim) * 0.1,
                                               requires_grad=False)
        self.accumulation_values = nn.Parameter(torch.zeros(num_centers),
                                                requires_grad=False)
        self._initialized = False
        self._init_count  = 0

    @torch.no_grad()
    def lazy_init_center(self, z_world: torch.Tensor) -> None:
        """
        Populate centers incrementally from observed z_world embeddings during warmup.
        Once all centers have been assigned at least one sample, initialization is done.
        We evenly assign each observed z into the next unfilled slot.
        """
        if self._initialized:
            return
        if self._init_count < self.num_centers:
            self.centers[self._init_count].copy_(z_world.detach())
            self._init_count += 1
        if self._init_count >= self.num_centers:
            self._initialized = True

    def forward(self, z_world: torch.Tensor) -> torch.Tensor:
        """Returns scalar residue score at z_world."""
        # z_world: [world_dim]
        dists   = torch.sum((self.centers - z_world.unsqueeze(0)) ** 2, dim=-1)  # [N]
        weights = torch.softmax(-dists / self.bandwidth, dim=0)                   # [N]
        return (weights * self.accumulation_values).sum()                          # scalar

    @torch.no_grad()
    def accumulate(self, z_world: torch.Tensor, harm_val: float) -> None:
        """Update nearest center's accumulation value."""
        if harm_val <= 0.0:
            return
        dists   = torch.sum((self.centers - z_world.detach().unsqueeze(0)) ** 2, dim=-1)
        nearest = int(torch.argmin(dists).item())
        self.accumulation_values[nearest] = (
            self.accumulation_values[nearest] + ACCUM_ALPHA * harm_val
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _pearson_r(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation. Returns 0.0 if degenerate."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx  = (sum((x - mx) ** 2 for x in xs) / n) ** 0.5
    sy  = (sum((y - my) ** 2 for y in ys) / n) ** 0.5
    if sx < 1e-8 or sy < 1e-8:
        return 0.0
    return num / (n * sx * sy)


def _get_world_obs(obs_dict: dict) -> torch.Tensor:
    """Extract world_state tensor, padded/truncated to WORLD_OBS_DIM."""
    raw = obs_dict.get("world_state")
    if raw is None:
        return torch.zeros(WORLD_OBS_DIM)
    t = raw.float() if isinstance(raw, torch.Tensor) else torch.tensor(raw, dtype=torch.float32)
    t = t.flatten()
    if t.shape[0] < WORLD_OBS_DIM:
        t = F.pad(t, (0, WORLD_OBS_DIM - t.shape[0]))
    elif t.shape[0] > WORLD_OBS_DIM:
        t = t[:WORLD_OBS_DIM]
    return t


def _hazard_proximity(obs_dict: dict) -> float:
    """
    Return local hazard proximity at current agent position.
    harm_obs = [hazard_field_view(25) + resource_field_view(25) + harm_exposure(1)].
    harm_obs[12] is the center of the 5x5 hazard view = hazard proximity at agent.
    """
    harm_obs = obs_dict.get("harm_obs")
    if harm_obs is None:
        return 0.0
    t = harm_obs if isinstance(harm_obs, torch.Tensor) else torch.tensor(harm_obs)
    return float(t[12].item()) if t.shape[0] > 12 else 0.0


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def _run_condition(
    seed: int,
    num_centers: int,
    condition_label: str,
    dry_run: bool,
) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(
        seed=seed,
        size=8,
        num_hazards=3,
        num_resources=3,
        hazard_harm=0.02,
        env_drift_interval=5,
        env_drift_prob=0.2,
    )

    world_enc = WorldEncoder(WORLD_OBS_DIM, WORLD_DIM)
    e2_fwd    = E2WorldForward(WORLD_DIM, ACTION_DIM)
    rbf_field = RBFLayer(num_centers, WORLD_DIM, RBF_BANDWIDTH)

    train_params = list(world_enc.parameters()) + list(e2_fwd.parameters())
    optimizer    = optim.Adam(train_params, lr=LR)

    # Tracking
    eval_harm_rates:    List[float] = []
    residue_scores_ep:  List[float] = []
    hazard_proxim_ep:   List[float] = []
    n_harm_events:      int = 0

    total_eps = TOTAL_EPISODES if not dry_run else 4

    _, obs_dict = env.reset()

    for ep in range(total_eps):
        is_eval    = ep >= WARMUP_EPISODES
        ep_harm    = 0.0
        ep_steps   = 0

        for step in range(STEPS_PER_EPISODE):
            obs_w = _get_world_obs(obs_dict)
            z_world = world_enc(obs_w)

            # Lazy initialize RBF centers from observations
            rbf_field.lazy_init_center(z_world)

            # Action selection
            if is_eval:
                # Greedy: pick action with lowest predicted residue score
                best_action = random.randint(0, ACTION_DIM - 1)
                best_score  = float("inf")
                for a_idx in range(ACTION_DIM):
                    a_oh = torch.zeros(ACTION_DIM)
                    a_oh[a_idx] = 1.0
                    with torch.no_grad():
                        z_next_pred = e2_fwd(z_world.detach(), a_oh)
                        score       = float(rbf_field(z_next_pred).item())
                    if score < best_score:
                        best_score  = score
                        best_action = a_idx
                action_idx = best_action
            else:
                action_idx = random.randint(0, ACTION_DIM - 1)

            a_oh = torch.zeros(ACTION_DIM)
            a_oh[action_idx] = 1.0

            flat_obs, harm_signal, done, _, obs_dict_next = env.step(a_oh.unsqueeze(0))

            obs_w_next  = _get_world_obs(obs_dict_next)
            harm_val    = max(0.0, -harm_signal)   # harm_signal < 0 = harm

            with torch.no_grad():
                z_world_next_actual = world_enc(obs_w_next)

            # Train encoder + E2 during warmup
            if not is_eval:
                z_world_next_pred = e2_fwd(z_world, a_oh)
                loss = F.mse_loss(z_world_next_pred, z_world_next_actual.detach())
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(train_params, 1.0)
                optimizer.step()
                # Re-encode after gradient step
                with torch.no_grad():
                    z_world_enc = world_enc(obs_w)
                rbf_field.accumulate(z_world_enc, harm_val)
            else:
                # Eval: accumulate residue and record metrics
                rbf_field.accumulate(z_world.detach(), harm_val)
                with torch.no_grad():
                    r_score = float(rbf_field(z_world.detach()).item())
                h_prox = _hazard_proximity(obs_dict)
                residue_scores_ep.append(r_score)
                hazard_proxim_ep.append(h_prox)

            ep_harm  += harm_val
            ep_steps += 1

            if harm_val > 0.0 and is_eval:
                n_harm_events += 1

            if done:
                _, obs_dict = env.reset()
            else:
                obs_dict = obs_dict_next

        if is_eval:
            eval_harm_rates.append(ep_harm / max(ep_steps, 1))

        if ep % 100 == 0 or ep == total_eps - 1:
            phase = "EVAL" if is_eval else "WARM"
            ep_harm_rate = ep_harm / max(ep_steps, 1)
            print(
                f"  [{condition_label}] seed={seed} ep={ep}/{total_eps}"
                f" phase={phase} ep_harm_rate={ep_harm_rate:.5f}",
                flush=True,
            )

    harm_rate        = float(sum(eval_harm_rates) / max(len(eval_harm_rates), 1))
    residue_accuracy = _pearson_r(residue_scores_ep, hazard_proxim_ep)

    print(
        f"  [{condition_label}] seed={seed} harm_rate={harm_rate:.5f}"
        f" residue_accuracy={residue_accuracy:.4f}"
        f" n_harm_events={n_harm_events}",
        flush=True,
    )

    return {
        "condition":         condition_label,
        "seed":              seed,
        "harm_rate":         harm_rate,
        "residue_accuracy":  residue_accuracy,
        "n_harm_events":     n_harm_events,
        "num_rbf_centers":   num_centers,
    }


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

def _evaluate_criteria(
    results: Dict[str, List[Dict]],
) -> Tuple[Dict[str, bool], str, str]:
    fine   = results["FINE"]
    coarse = results["COARSE"]
    n_s    = len(SEEDS)

    # C1: FINE harm_rate <= COARSE harm_rate + margin (both seeds)
    c1 = all(
        fine[i]["harm_rate"] <= coarse[i]["harm_rate"] + THRESH_C1_MARGIN
        for i in range(n_s)
    )

    # C2: FINE residue_accuracy > threshold (both seeds)
    c2 = all(fine[i]["residue_accuracy"] > THRESH_C2_ACCURACY for i in range(n_s))

    # C3: COARSE n_harm_events >= threshold (both seeds, data quality)
    c3 = all(coarse[i]["n_harm_events"] >= THRESH_C3_MIN_EVENTS for i in range(n_s))

    criteria = {
        "C1_fine_harm_rate_not_worse_than_coarse": c1,
        "C2_fine_residue_accuracy_positive":        c2,
        "C3_coarse_sufficient_harm_events":         c3,
    }

    if c1 and c2 and c3:
        outcome           = "PASS"
        evidence_direction = "supports"
    elif c1 and c3 and not c2:
        outcome           = "FAIL"
        evidence_direction = "weakens"
    else:
        outcome           = "FAIL"
        evidence_direction = "mixed"

    return criteria, outcome, evidence_direction


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> Dict:
    print("=== V3-EXQ-170: Q-002 R-field Resolution Pair ===", flush=True)
    print(f"Conditions: {CONDITIONS}  Seeds: {SEEDS}", flush=True)
    print("Pre-registered thresholds:", flush=True)
    print(f"  C1 THRESH_C1_MARGIN     = {THRESH_C1_MARGIN}", flush=True)
    print(f"  C2 THRESH_C2_ACCURACY   = {THRESH_C2_ACCURACY}", flush=True)
    print(f"  C3 THRESH_C3_MIN_EVENTS = {THRESH_C3_MIN_EVENTS}", flush=True)
    print(
        f"  WARMUP_EPISODES={WARMUP_EPISODES}"
        f"  EVAL_EPISODES={EVAL_EPISODES}"
        f"  STEPS_PER_EPISODE={STEPS_PER_EPISODE}",
        flush=True,
    )

    results: Dict[str, List[Dict]] = {"FINE": [], "COARSE": []}

    center_map = {"FINE": FINE_NUM_CENTERS, "COARSE": COARSE_NUM_CENTERS}

    for condition in CONDITIONS:
        num_centers = center_map[condition]
        print(f"\n=== Condition: {condition} (num_rbf_centers={num_centers}) ===",
              flush=True)
        for seed in SEEDS:
            r = _run_condition(
                seed=seed,
                num_centers=num_centers,
                condition_label=condition,
                dry_run=dry_run,
            )
            results[condition].append(r)

    print("\n=== Evaluating criteria ===", flush=True)
    criteria, outcome, evidence_direction = _evaluate_criteria(results)
    for k, v in criteria.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}", flush=True)
    print(f"Overall outcome: {outcome}", flush=True)

    def _mean(cond: str, key: str) -> float:
        vals = [r[key] for r in results[cond]]
        return float(sum(vals) / max(len(vals), 1))

    summary_metrics = {
        "fine_harm_rate":          _mean("FINE",   "harm_rate"),
        "coarse_harm_rate":        _mean("COARSE", "harm_rate"),
        "fine_residue_accuracy":   _mean("FINE",   "residue_accuracy"),
        "coarse_residue_accuracy": _mean("COARSE", "residue_accuracy"),
        "fine_n_harm_events":      _mean("FINE",   "n_harm_events"),
        "coarse_n_harm_events":    _mean("COARSE", "n_harm_events"),
        "delta_harm_coarse_minus_fine": (
            _mean("COARSE", "harm_rate") - _mean("FINE", "harm_rate")
        ),
    }

    run_id = (
        "v3_exq_170_q002_r_field_resolution_pair_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_v3"
    )

    pack = {
        "run_id":               run_id,
        "experiment_type":      EXPERIMENT_TYPE,
        "claim_ids":            CLAIM_IDS,
        "architecture_epoch":   "ree_hybrid_guardrails_v1",
        "outcome":              outcome,
        "evidence_direction":   evidence_direction,
        "evidence_class":       "discriminative_pair",
        "criteria":             criteria,
        "pre_registered_thresholds": {
            "THRESH_C1_MARGIN":     THRESH_C1_MARGIN,
            "THRESH_C2_ACCURACY":   THRESH_C2_ACCURACY,
            "THRESH_C3_MIN_EVENTS": THRESH_C3_MIN_EVENTS,
        },
        "summary_metrics":      summary_metrics,
        "protocol": {
            "warmup_episodes":      WARMUP_EPISODES,
            "eval_episodes":        EVAL_EPISODES,
            "steps_per_episode":    STEPS_PER_EPISODE,
            "lr":                   LR,
            "accum_alpha":          ACCUM_ALPHA,
            "rbf_bandwidth":        RBF_BANDWIDTH,
            "fine_num_centers":     FINE_NUM_CENTERS,
            "coarse_num_centers":   COARSE_NUM_CENTERS,
        },
        "seeds": SEEDS,
        "scenario": (
            "Two-condition residue-field spatial resolution test."
            " FINE (32 RBF centers) vs COARSE (4 RBF centers) in z_world latent space."
            " Architecture: WorldEncoder (Linear(250,32)+LayerNorm),"
            " E2WorldForward (Linear(37,32)),"
            " RBFLayer with N centers accumulating harm proximity signals."
            " Warmup: 200 episodes, random actions, train encoder+E2 on world prediction MSE,"
            " accumulate harm at nearest RBF center."
            " Eval: 100 episodes, greedy avoidance -- pick action minimizing predicted"
            " residue score (E2 roll one step)."
            " Metrics: harm_rate (lower is better), residue_accuracy (Pearson r between"
            " field score and hazard proximity; higher = field correctly represents hazard space)."
            " CausalGridWorldV2 size=8 num_hazards=3 num_resources=3 hazard_harm=0.02"
            " env_drift_interval=5 env_drift_prob=0.2. 2 conditions x 2 seeds = 4 cells."
        ),
        "interpretation": (
            "PASS => FINE resolution matches or beats COARSE on harm rate, and FINE field"
            " shows positive correlation with hazard proximity."
            " Supports Q-002: finer spatial resolution in R(x,t) is at least as effective"
            " as coarse for harm navigation; spatial structure of z_world latent is"
            " informative at fine granularity."
            " FAIL (C1+C3 but not C2) => FINE field does not achieve spatial learning"
            " (RBF centers not meaningfully differentiated in z_world space)."
            " Weakens Q-002 fine-resolution hypothesis: z_world may not be spatially"
            " organized enough at this scale to support fine-grained residue fields."
            " FAIL (other) => insufficient harm events or mixed-direction results;"
            " experiment inconclusive for Q-002."
        ),
        "per_seed_results":   {cond: results[cond] for cond in CONDITIONS},
        "dry_run":            dry_run,
        "timestamp_utc":      datetime.now(timezone.utc).isoformat(),
    }

    if not dry_run:
        out_dir = (
            Path(__file__).resolve().parents[2]
            / "REE_assembly"
            / "evidence"
            / "experiments"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{run_id}.json"
        with open(out_path, "w") as f:
            json.dump(pack, f, indent=2)
        print(f"\nResult pack written to: {out_path}", flush=True)
    else:
        print("\n[dry_run] Result pack NOT written.", flush=True)
        print(json.dumps(pack, indent=2), flush=True)

    return pack


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    result = run_experiment(dry_run=dry_run)
    print(f"Done. Outcome: {result['outcome']}", flush=True)
