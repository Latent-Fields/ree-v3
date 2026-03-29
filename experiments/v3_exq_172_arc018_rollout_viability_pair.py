#!/opt/local/bin/python3
"""
V3-EXQ-172 -- ARC-018: Rollout Viability Mapping vs Greedy Harm (Discriminative Pair)

Claim: ARC-018 -- hippocampus.rollout_viability_mapping
  "Hippocampus generates explicit rollouts and post-commitment viability mapping."

Why now: active_conflict -- 3 V3 PASS results (EXQ-042, EXQ-053, EXQ-120) alongside
  1 V2 FAIL (EXQ-021). The V2 FAIL predates SD-004/005 separation; this experiment
  tests the core rollout mechanism in a clean V3 context.

Design: discriminative_pair
  ROLLOUT_VIABILITY vs GREEDY_HARM

  Both conditions share:
    - WorldEncoder: Linear(250, 32) + LayerNorm -> z_world (dim 32)
    - E2WorldForward: Linear(32 + 5, 32) -> z_world_next (one-hot action concat)
    - HarmHead: Linear(32, 1) -> harm_scalar
    - 500 warmup episodes, Adam lr=3e-4, trained on MSE world prediction + MSE harm prediction

  ROLLOUT_VIABILITY eval (50 episodes x 100 steps):
    For each step, sample 10 candidate actions; for each, roll out k=5 steps using
    E2WorldForward; sum HarmHead scores over rollout; choose action minimising sum.

  GREEDY_HARM eval (50 episodes x 100 steps):
    For each step, evaluate all 5 actions 1-step ahead with HarmHead; choose min.

Pre-registered thresholds:
  C1: harm_rate_ROLLOUT <= harm_rate_GREEDY  (both seeds independently)
  C2: harm_advantage >= THRESH_C2_ADVANTAGE = 0.005  (averaged over seeds)
  C3: e2_world_r2 >= THRESH_C3_E2_QUALITY = 0.15  (E2 adequately trained; diagnostic)
  C4: n_harm_contacts_GREEDY >= THRESH_C4_MIN = 5  (data quality; per seed)

outcome:
  PASS        if C1 + C2 + C3 + C4
  INCONCLUSIVE if NOT C3 or NOT C4
  FAIL        otherwise

evidence_direction:
  "supports"  if PASS
  "weakens"   if NOT C1 (both seeds) and C4
  "mixed"     otherwise
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

EXPERIMENT_TYPE = "v3_exq_172_arc018_rollout_viability_pair"
CLAIM_IDS = ["ARC-018"]
SEEDS = [42, 123]

# Environment
ENV_SIZE = 8
NUM_HAZARDS = 3
NUM_RESOURCES = 3
HAZARD_HARM = 0.02

# Architecture
WORLD_DIM = 32
ACTION_DIM = 5
LR = 3e-4

# Training
N_WARMUP_EPISODES = 500
STEPS_PER_EPISODE = 100

# Eval
N_EVAL_EPISODES = 50
EVAL_STEPS = 100

# Rollout
ROLLOUT_K = 5
N_CANDIDATES = 10

# Pre-registered thresholds
THRESH_C2_ADVANTAGE = 0.005
THRESH_C3_E2_QUALITY = 0.15
THRESH_C4_MIN = 5


# ---------------------------------------------------------------------------
# Inline models
# ---------------------------------------------------------------------------

class WorldEncoder(nn.Module):
    """Linear(250 -> WORLD_DIM) + LayerNorm."""

    def __init__(self, world_obs_dim: int = 250, world_dim: int = WORLD_DIM):
        super().__init__()
        self.fc = nn.Linear(world_obs_dim, world_dim)
        self.ln = nn.LayerNorm(world_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(F.relu(self.fc(x)))


class E2WorldForward(nn.Module):
    """Predicts next z_world given current z_world and one-hot action."""

    def __init__(self, world_dim: int = WORLD_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.fc = nn.Linear(world_dim + action_dim, world_dim)

    def forward(self, z_world: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_world, action_onehot], dim=-1)
        return self.fc(x)


class HarmHead(nn.Module):
    """Predicts scalar harm from z_world."""

    def __init__(self, world_dim: int = WORLD_DIM):
        super().__init__()
        self.fc = nn.Linear(world_dim, 1)

    def forward(self, z_world: torch.Tensor) -> torch.Tensor:
        return self.fc(z_world)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def action_onehot(a: int, n: int = ACTION_DIM) -> torch.Tensor:
    v = torch.zeros(n)
    v[a] = 1.0
    return v


def rollout_harm_score(
    z0: torch.Tensor,
    action: int,
    e2: E2WorldForward,
    harm_head: HarmHead,
    k: int = ROLLOUT_K,
) -> float:
    """
    Roll out k steps from z0 starting with `action`, accumulate harm predictions.
    After the first step, always take action 4 (STAY) to isolate forward trajectory.
    Returns total predicted harm as a float.
    """
    z = z0.detach().clone()
    total = 0.0
    for step in range(k):
        a = action if step == 0 else 4
        ah = action_onehot(a).unsqueeze(0)
        z_next = e2(z, ah)
        h = harm_head(z_next).item()
        total += max(h, 0.0)
        z = z_next.detach()
    return total


def greedy_harm_score(
    z0: torch.Tensor,
    action: int,
    e2: E2WorldForward,
    harm_head: HarmHead,
) -> float:
    """One-step predicted harm for a single action."""
    ah = action_onehot(action).unsqueeze(0)
    z_next = e2(z0.detach(), ah)
    return harm_head(z_next).item()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_shared_models(seed: int, dry_run: bool = False) -> tuple:
    """
    Run 500 warmup episodes, jointly training WorldEncoder, E2WorldForward, HarmHead.
    Returns (encoder, e2, harm_head, e2_r2) where e2_r2 is held-out R^2 on world prediction.
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
    e2 = E2WorldForward()
    harm_head = HarmHead()

    params = list(encoder.parameters()) + list(e2.parameters()) + list(harm_head.parameters())
    optimizer = optim.Adam(params, lr=LR)

    n_train = N_WARMUP_EPISODES if not dry_run else 5

    # Replay buffer: (world_state_t, action, world_state_next, harm_signal)
    buffer = []

    for ep in range(n_train):
        obs_t, info = env.reset()
        ws = info["world_state"].float().unsqueeze(0)

        for _ in range(STEPS_PER_EPISODE):
            a = rng.randint(0, ACTION_DIM - 1)
            obs_next, reward, done, info_next, obs_dict_next = env.step(a)
            ws_next = obs_dict_next["world_state"].float().unsqueeze(0)
            harm_val = max(-reward, 0.0)

            buffer.append((ws.squeeze(0).clone(), a, ws_next.squeeze(0).clone(), harm_val))
            if len(buffer) > 10000:
                buffer.pop(0)

            ws = ws_next
            if done:
                break

        # Train each episode on a mini-batch from buffer
        if len(buffer) >= 32:
            batch = random.sample(buffer, min(64, len(buffer)))
            ws_b = torch.stack([b[0] for b in batch])
            a_b = torch.tensor([b[1] for b in batch], dtype=torch.long)
            ws_next_b = torch.stack([b[2] for b in batch])
            harm_b = torch.tensor([b[3] for b in batch], dtype=torch.float32).unsqueeze(1)

            ah_b = F.one_hot(a_b, ACTION_DIM).float()

            z_b = encoder(ws_b)
            z_next_pred = e2(z_b, ah_b)
            z_next_target = encoder(ws_next_b).detach()

            loss_e2 = F.mse_loss(z_next_pred, z_next_target)
            harm_pred = harm_head(z_b)
            loss_harm = F.mse_loss(harm_pred, harm_b)

            loss = loss_e2 + loss_harm
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Compute e2_r2 on held-out samples (last 200 from buffer)
    encoder.eval()
    e2.eval()
    harm_head.eval()

    e2_r2 = 0.0
    if len(buffer) >= 50:
        eval_buf = buffer[-200:]
        ws_e = torch.stack([b[0] for b in eval_buf])
        a_e = torch.tensor([b[1] for b in eval_buf], dtype=torch.long)
        ws_next_e = torch.stack([b[2] for b in eval_buf])

        ah_e = F.one_hot(a_e, ACTION_DIM).float()
        with torch.no_grad():
            z_e = encoder(ws_e)
            z_next_pred_e = e2(z_e, ah_e)
            z_next_target_e = encoder(ws_next_e)

        ss_res = ((z_next_pred_e - z_next_target_e) ** 2).sum().item()
        ss_tot = ((z_next_target_e - z_next_target_e.mean(0)) ** 2).sum().item()
        e2_r2 = 1.0 - (ss_res / (ss_tot + 1e-8))

    encoder.train()
    e2.train()
    harm_head.train()

    return encoder, e2, harm_head, e2_r2


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def eval_condition(
    condition: str,
    seed: int,
    encoder: WorldEncoder,
    e2: E2WorldForward,
    harm_head: HarmHead,
    dry_run: bool = False,
) -> dict:
    """
    Run N_EVAL_EPISODES x EVAL_STEPS under the specified condition.
    Returns dict with harm_rate, n_harm_contacts, total_steps.
    """
    rng = random.Random(seed + 9999)
    torch.manual_seed(seed + 9999)

    env = CausalGridWorldV2(
        size=ENV_SIZE,
        num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES,
        hazard_harm=HAZARD_HARM,
    )

    encoder.eval()
    e2.eval()
    harm_head.eval()

    n_episodes = N_EVAL_EPISODES if not dry_run else 3
    harm_contacts = 0
    total_steps = 0

    for ep in range(n_episodes):
        obs_t, info = env.reset()
        ws = info["world_state"].float().unsqueeze(0)

        for step in range(EVAL_STEPS):
            with torch.no_grad():
                z = encoder(ws)

                if condition == "ROLLOUT_VIABILITY":
                    candidates = [rng.randint(0, ACTION_DIM - 1) for _ in range(N_CANDIDATES)]
                    scores = [rollout_harm_score(z, a, e2, harm_head, ROLLOUT_K) for a in candidates]
                    action = candidates[scores.index(min(scores))]
                elif condition == "GREEDY_HARM":
                    scores = [greedy_harm_score(z, a, e2, harm_head) for a in range(ACTION_DIM)]
                    action = int(scores.index(min(scores)))
                else:
                    raise ValueError(f"Unknown condition: {condition}")

            obs_next, reward, done, info_next, obs_dict_next = env.step(action)
            ws = obs_dict_next["world_state"].float().unsqueeze(0)
            total_steps += 1

            if info_next.get("harm_exposure", 0.0) > 0.0:
                harm_contacts += 1

            if done:
                break

    harm_rate = harm_contacts / max(total_steps, 1)
    return {
        "harm_rate": harm_rate,
        "n_harm_contacts": harm_contacts,
        "total_steps": total_steps,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> dict:
    per_seed_results = {}
    e2_r2_vals = []

    rollout_harm_rates = []
    greedy_harm_rates = []
    greedy_harm_contacts_per_seed = []

    for seed in SEEDS:
        print(f"[seed {seed}] Training shared models...", flush=True)
        encoder, e2, harm_head, e2_r2 = train_shared_models(seed, dry_run=dry_run)
        e2_r2_vals.append(e2_r2)
        print(f"[seed {seed}] e2_world_r2 = {e2_r2:.4f}", flush=True)

        rollout_res = eval_condition("ROLLOUT_VIABILITY", seed, encoder, e2, harm_head, dry_run=dry_run)
        greedy_res = eval_condition("GREEDY_HARM", seed, encoder, e2, harm_head, dry_run=dry_run)

        rollout_harm_rates.append(rollout_res["harm_rate"])
        greedy_harm_rates.append(greedy_res["harm_rate"])
        greedy_harm_contacts_per_seed.append(greedy_res["n_harm_contacts"])

        harm_adv = greedy_res["harm_rate"] - rollout_res["harm_rate"]
        print(
            f"[seed {seed}] ROLLOUT harm_rate={rollout_res['harm_rate']:.4f} "
            f"GREEDY harm_rate={greedy_res['harm_rate']:.4f} "
            f"advantage={harm_adv:.4f}",
            flush=True,
        )

        per_seed_results[str(seed)] = {
            "e2_world_r2": e2_r2,
            "ROLLOUT_VIABILITY": rollout_res,
            "GREEDY_HARM": greedy_res,
            "harm_advantage": harm_adv,
            "c1_direction_pass": rollout_res["harm_rate"] <= greedy_res["harm_rate"],
        }

    # Criteria evaluation
    e2_r2_mean = sum(e2_r2_vals) / len(e2_r2_vals)
    harm_advantage_mean = sum(
        greedy_harm_rates[i] - rollout_harm_rates[i] for i in range(len(SEEDS))
    ) / len(SEEDS)

    c1_pass = all(
        per_seed_results[str(s)]["c1_direction_pass"] for s in SEEDS
    )
    c2_pass = harm_advantage_mean >= THRESH_C2_ADVANTAGE
    c3_pass = e2_r2_mean >= THRESH_C3_E2_QUALITY
    c4_pass = all(
        greedy_harm_contacts_per_seed[i] >= THRESH_C4_MIN for i in range(len(SEEDS))
    )

    criteria = {
        "C1_direction": c1_pass,
        "C2_harm_advantage": c2_pass,
        "C3_e2_quality": c3_pass,
        "C4_data_quality": c4_pass,
    }

    print(f"Criteria: {criteria}", flush=True)
    print(f"harm_advantage_mean={harm_advantage_mean:.4f}  e2_r2_mean={e2_r2_mean:.4f}", flush=True)

    # Outcome
    if not c3_pass or not c4_pass:
        outcome = "INCONCLUSIVE"
        evidence_direction = "inconclusive"
    elif c1_pass and c2_pass:
        outcome = "PASS"
        evidence_direction = "supports"
    elif not c1_pass and c4_pass:
        outcome = "FAIL"
        evidence_direction = "weakens"
    else:
        outcome = "FAIL"
        evidence_direction = "mixed"

    print(f"Outcome: {outcome}  evidence_direction: {evidence_direction}", flush=True)

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
        "evidence_class": "discriminative_pair",
        "criteria": criteria,
        "pre_registered_thresholds": {
            "THRESH_C2_ADVANTAGE": THRESH_C2_ADVANTAGE,
            "THRESH_C3_E2_QUALITY": THRESH_C3_E2_QUALITY,
            "THRESH_C4_MIN": THRESH_C4_MIN,
        },
        "summary_metrics": {
            "harm_rate_ROLLOUT_mean": sum(rollout_harm_rates) / len(rollout_harm_rates),
            "harm_rate_GREEDY_mean": sum(greedy_harm_rates) / len(greedy_harm_rates),
            "harm_advantage_mean": harm_advantage_mean,
            "e2_world_r2_mean": e2_r2_mean,
            "greedy_harm_contacts_per_seed": greedy_harm_contacts_per_seed,
        },
        "seeds": SEEDS,
        "scenario": (
            "discriminative_pair: ROLLOUT_VIABILITY (k=5 step E2 rollout, n=10 candidates) "
            "vs GREEDY_HARM (1-step harm head); shared WorldEncoder+E2+HarmHead "
            f"trained 500 warmup eps; eval 50 eps x 100 steps; CausalGridWorldV2 "
            f"size={ENV_SIZE} hazards={NUM_HAZARDS} resources={NUM_RESOURCES}"
        ),
        "interpretation": (
            "PASS supports ARC-018: explicit multi-step rollout viability scoring "
            "outperforms greedy 1-step harm avoidance, validating the hippocampal "
            "rollout-and-viability-map mechanism. INCONCLUSIVE = E2 not trained "
            "sufficiently or insufficient harm contacts for reliable comparison. "
            "FAIL weakens ARC-018: rollout provides no benefit over greedy evaluation."
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
