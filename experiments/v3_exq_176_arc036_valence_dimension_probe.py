#!/opt/local/bin/python3
"""
V3-EXQ-176 -- ARC-036: Valence Dimension Dissociation Probe (targeted_probe)

Claim: ARC-036 -- hippocampus.multidimensional_valence_map
  "Hippocampal terrain maps encode separable valence dimensions (wanting, liking,
  harm-discriminative, surprise); ARC-007 residue-field navigation is the harm-causal
  special case of this general architecture."

Why now: insufficient_experimental_replication, missing_experimental_evidence,
  synthetic_signals_only. No V3 real-environment evidence yet for ARC-036.

Design: targeted_probe
  Train a WorldEncoder on CausalGridWorldV2 (random-action data collection).
  Maintain two RBF terrain accumulators:
    HarmRBF:    accumulates harm signal at hazard contacts (via harm_exposure)
    BenefitRBF: accumulates benefit signal at resource contacts (via benefit_exposure)
  After training, collect per-step terrain scores and compute Pearson correlation.
  Low |r| = dimensions dissociate = terrain encoding is multi-dimensional = supports ARC-036.

Architecture (inline, no REEAgent):
  WorldEncoder: Linear(250, 32) + LayerNorm -> z_world
  RBFLayer: 16 RBF centers, EMA accumulation, softmax-weighted scoring
  No action policy -- random actions throughout.

Pre-registered thresholds:
  C1: harm_score_variance > THRESH_C1_VAR = 0.001  (both seeds; field is active)
  C2: benefit_score_variance > THRESH_C2_VAR = 0.001  (both seeds; field is active)
  C3: abs(harm_benefit_correlation) < THRESH_C3_CORR = 0.7  (both seeds; dissociation)

outcome: "PASS" if C1+C2+C3; "FAIL" otherwise
evidence_direction: "supports" if PASS; "weakens" if FAIL; "mixed" if partial
evidence_class: "targeted_probe"
"""

import sys
import json
import random
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ree_core.environment.causal_grid_world import CausalGridWorldV2

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_TYPE = "v3_exq_176_arc036_valence_dimension_probe"
CLAIM_IDS = ["ARC-036"]
SEEDS = [42, 123]

# Environment
ENV_SIZE = 8
NUM_HAZARDS = 3
NUM_RESOURCES = 3
HAZARD_HARM = 0.02

# Architecture
WORLD_DIM = 32
WORLD_OBS_DIM = 250
RBF_CENTERS = 16
RBF_BANDWIDTH = 1.0
RBF_ALPHA = 0.1

# Training
N_EPISODES = 500
STEPS_PER_EPISODE = 100
LR = 3e-4

# Pre-registered thresholds
THRESH_C1_VAR = 0.001
THRESH_C2_VAR = 0.001
THRESH_C3_CORR = 0.7


# ---------------------------------------------------------------------------
# Inline models
# ---------------------------------------------------------------------------

class WorldEncoder(nn.Module):
    """Linear(250 -> WORLD_DIM) + LayerNorm with reconstruction head."""

    def __init__(self, obs_dim: int = WORLD_OBS_DIM, world_dim: int = WORLD_DIM):
        super().__init__()
        self.encoder = nn.Linear(obs_dim, world_dim)
        self.ln = nn.LayerNorm(world_dim)
        self.decoder = nn.Linear(world_dim, obs_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(F.relu(self.encoder(x)))

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class RBFLayer(nn.Module):
    """
    RBF terrain accumulator.
    Centers in z_world space; softmax-weighted scoring.
    Values accumulated via EMA.
    """

    def __init__(
        self,
        world_dim: int = WORLD_DIM,
        num_centers: int = RBF_CENTERS,
        bandwidth: float = RBF_BANDWIDTH,
    ):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_centers, world_dim) * 0.1)
        self.bandwidth = bandwidth
        self.register_buffer("values", torch.zeros(num_centers))

    def accumulate(self, z_world: torch.Tensor, signal: float, alpha: float = RBF_ALPHA):
        """Update accumulated terrain values with signal at z_world location."""
        with torch.no_grad():
            # z_world: (1, world_dim) or (world_dim,); centers: (num_centers, world_dim)
            zw = z_world.squeeze(0)  # (world_dim,)
            dists = ((zw.unsqueeze(0) - self.centers) ** 2).sum(-1)  # (num_centers,)
            weights = torch.softmax(-dists / self.bandwidth, dim=-1)  # (num_centers,)
            self.values = (1.0 - alpha) * self.values + alpha * (weights * signal)

    def score(self, z_world: torch.Tensor) -> float:
        """Return weighted sum of terrain values at z_world location."""
        with torch.no_grad():
            zw = z_world.squeeze(0)
            dists = ((zw.unsqueeze(0) - self.centers) ** 2).sum(-1)
            weights = torch.softmax(-dists / self.bandwidth, dim=-1)
            return (weights * self.values).sum().item()


# ---------------------------------------------------------------------------
# Pearson correlation
# ---------------------------------------------------------------------------

def pearson_r(xs: list, ys: list) -> float:
    """Pearson r between two lists; returns 0.0 if degenerate."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom_x = (sum((x - mx) ** 2 for x in xs)) ** 0.5
    denom_y = (sum((y - my) ** 2 for y in ys)) ** 0.5
    denom = denom_x * denom_y
    if denom < 1e-12:
        return 0.0
    return num / denom


# ---------------------------------------------------------------------------
# Per-seed run
# ---------------------------------------------------------------------------

def run_seed(seed: int, dry_run: bool = False) -> dict:
    """
    Run one seed: train WorldEncoder, accumulate RBF terrain, compute dissociation.
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)

    env = CausalGridWorldV2(
        size=ENV_SIZE,
        num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES,
        hazard_harm=HAZARD_HARM,
    )

    encoder = WorldEncoder()
    harm_rbf = RBFLayer()
    benefit_rbf = RBFLayer()

    optimizer = optim.Adam(encoder.parameters(), lr=LR)

    n_episodes = N_EPISODES if not dry_run else 5
    n_harm_events = 0
    n_benefit_events = 0

    # Phase 1: train WorldEncoder with reconstruction loss (random policy)
    print(f"  [seed {seed}] Training WorldEncoder ({n_episodes} eps)...", flush=True)
    for ep in range(n_episodes):
        obs_t, info = env.reset()
        ws = info["world_state"].float().unsqueeze(0)

        for _ in range(STEPS_PER_EPISODE):
            a = rng.randint(0, 4)
            obs_next, reward, done, info_next, obs_dict_next = env.step(a)
            ws_next = obs_dict_next["world_state"].float().unsqueeze(0)

            z = encoder(ws)
            ws_recon = encoder.reconstruct(z)
            loss = F.mse_loss(ws_recon, ws.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ws = ws_next
            if done:
                break

    encoder.eval()

    # Phase 2: collect terrain scores with frozen encoder + random policy
    print(f"  [seed {seed}] Collecting terrain scores...", flush=True)
    harm_scores = []
    benefit_scores = []

    for ep in range(n_episodes):
        obs_t, info = env.reset()
        ws = info["world_state"].float().unsqueeze(0)

        for _ in range(STEPS_PER_EPISODE):
            a = rng.randint(0, 4)
            obs_next, reward, done, info_next, obs_dict_next = env.step(a)
            ws_next = obs_dict_next["world_state"].float().unsqueeze(0)

            harm_exp = info_next.get("harm_exposure", 0.0)
            benefit_exp = info_next.get("benefit_exposure", 0.0)

            with torch.no_grad():
                z = encoder(ws)

            # Accumulate terrain
            if harm_exp > 0.0:
                harm_rbf.accumulate(z, harm_exp)
                n_harm_events += 1
            if benefit_exp > 0.0:
                benefit_rbf.accumulate(z, benefit_exp)
                n_benefit_events += 1

            # Score current location
            h_score = harm_rbf.score(z)
            b_score = benefit_rbf.score(z)
            harm_scores.append(h_score)
            benefit_scores.append(b_score)

            ws = ws_next
            if done:
                break

    # Compute metrics
    correlation = pearson_r(harm_scores, benefit_scores)
    harm_var = sum((x - sum(harm_scores) / len(harm_scores)) ** 2 for x in harm_scores) / max(len(harm_scores), 1)
    benefit_var = sum((x - sum(benefit_scores) / len(benefit_scores)) ** 2 for x in benefit_scores) / max(len(benefit_scores), 1)

    print(
        f"  [seed {seed}] harm_var={harm_var:.6f}  benefit_var={benefit_var:.6f}  "
        f"corr={correlation:.4f}  n_harm={n_harm_events}  n_benefit={n_benefit_events}",
        flush=True,
    )

    return {
        "harm_score_variance": harm_var,
        "benefit_score_variance": benefit_var,
        "harm_benefit_correlation": correlation,
        "n_harm_events": n_harm_events,
        "n_benefit_events": n_benefit_events,
        "n_score_samples": len(harm_scores),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> dict:
    per_seed_results = {}

    for seed in SEEDS:
        print(f"[EXQ-176] Seed {seed}...", flush=True)
        per_seed_results[str(seed)] = run_seed(seed, dry_run=dry_run)

    # Criteria evaluation
    c1_pass = all(
        per_seed_results[str(s)]["harm_score_variance"] > THRESH_C1_VAR
        for s in SEEDS
    )
    c2_pass = all(
        per_seed_results[str(s)]["benefit_score_variance"] > THRESH_C2_VAR
        for s in SEEDS
    )
    c3_pass = all(
        abs(per_seed_results[str(s)]["harm_benefit_correlation"]) < THRESH_C3_CORR
        for s in SEEDS
    )

    criteria = {
        "C1_harm_field_active": c1_pass,
        "C2_benefit_field_active": c2_pass,
        "C3_dissociation": c3_pass,
    }

    print(f"Criteria: {criteria}", flush=True)

    # Outcome
    if c1_pass and c2_pass and c3_pass:
        outcome = "PASS"
        evidence_direction = "supports"
    elif not c1_pass or not c2_pass:
        # Fields not active -- cannot assess dissociation
        outcome = "FAIL"
        evidence_direction = "mixed"
    else:
        # Fields active but correlated
        outcome = "FAIL"
        evidence_direction = "weakens"

    # Partial check: exactly one of C1/C2/C3 fails -> mixed
    n_fail = sum([not c1_pass, not c2_pass, not c3_pass])
    if n_fail == 1:
        evidence_direction = "mixed"

    print(f"Outcome: {outcome}  evidence_direction: {evidence_direction}", flush=True)

    # Aggregate summary metrics
    corr_vals = [per_seed_results[str(s)]["harm_benefit_correlation"] for s in SEEDS]
    harm_var_vals = [per_seed_results[str(s)]["harm_score_variance"] for s in SEEDS]
    benefit_var_vals = [per_seed_results[str(s)]["benefit_score_variance"] for s in SEEDS]

    pack = {
        "run_id": (
            f"{EXPERIMENT_TYPE}_"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_v3"
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "evidence_class": "targeted_probe",
        "criteria": criteria,
        "pre_registered_thresholds": {
            "THRESH_C1_VAR": THRESH_C1_VAR,
            "THRESH_C2_VAR": THRESH_C2_VAR,
            "THRESH_C3_CORR": THRESH_C3_CORR,
        },
        "summary_metrics": {
            "harm_benefit_correlation_mean": sum(corr_vals) / len(corr_vals),
            "harm_score_variance_mean": sum(harm_var_vals) / len(harm_var_vals),
            "benefit_score_variance_mean": sum(benefit_var_vals) / len(benefit_var_vals),
            "n_harm_events_per_seed": [per_seed_results[str(s)]["n_harm_events"] for s in SEEDS],
            "n_benefit_events_per_seed": [per_seed_results[str(s)]["n_benefit_events"] for s in SEEDS],
        },
        "seeds": SEEDS,
        "scenario": (
            "targeted_probe: WorldEncoder (Linear 250->32 + LayerNorm) trained with "
            "reconstruction loss (random policy). HarmRBF + BenefitRBF (16 centers, "
            f"bandwidth={RBF_BANDWIDTH}, alpha={RBF_ALPHA}) accumulate harm_exposure "
            "and benefit_exposure signals. Pearson r between per-step harm_score and "
            f"benefit_score measures valence dissociation. "
            f"{N_EPISODES} episodes x {STEPS_PER_EPISODE} steps; "
            f"CausalGridWorldV2 size={ENV_SIZE} hazards={NUM_HAZARDS} resources={NUM_RESOURCES}."
        ),
        "interpretation": (
            "PASS: both terrain fields active AND dissociated (|r|<0.7). "
            "Supports ARC-036: harm and benefit accumulate in orthogonal terrain "
            "subspaces, consistent with multi-dimensional valence map hypothesis. "
            "FAIL with C3 fail: fields are correlated -- harm and benefit terrain "
            "overlap, weakening ARC-036 multi-dimensional claim. "
            "FAIL with C1/C2: insufficient signal for assessment (mixed)."
        ),
        "per_seed_results": per_seed_results,
        "dry_run": dry_run,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if not dry_run:
        out_dir = (
            Path(__file__).resolve().parents[2]
            / "REE_assembly"
            / "evidence"
            / "experiments"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{pack['run_id']}.json"
        with open(out_path, "w") as f:
            json.dump(pack, f, indent=2)
        print(f"Result pack written to: {out_path}", flush=True)
    else:
        print("[dry_run] Result pack NOT written.", flush=True)

    return pack


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    result = run_experiment(dry_run=dry_run)
    print(f"Done. Outcome: {result['outcome']}", flush=True)
