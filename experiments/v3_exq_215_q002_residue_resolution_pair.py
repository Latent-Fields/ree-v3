#!/opt/local/bin/python3
"""
V3-EXQ-215 -- Q-002: Residue Field Spatial Resolution Discriminative Pair

Claim: Q-002 (active open_question)
  "What is the appropriate spatial resolution for R(x,t)?"

Prior experiments: EXQ-170 (FAIL, inline architecture, N=2 seeds).
  EXQ-170's failure was a data quality issue (n_harm_events=0 in coarse condition
  dry-run). EXQ-170 also used inline lightweight architecture, not real ResidueField.
  This experiment uses the real ree_core.residue.ResidueField with num_basis_functions
  as the resolution parameter, and REEAgent for training.

Experimental design -- discriminative_pair:
  Q-002 asks whether finer-grained RBF resolution in the residue field produces
  measurably better harm avoidance than coarser resolution.

  Condition A: HIGH_RES (num_basis_functions=64, fine-grained coverage)
  Condition B: LOW_RES  (num_basis_functions=8,  coarse coverage)

  Both conditions:
    - Use the same training protocol (REEAgent with CausalGridWorldV2)
    - Same encoder (shared WorldEncoder trained jointly)
    - Same harm accumulation protocol (both use ResidueField.accumulate())
    - Eval: greedy residue-based action selection (argmin predicted residue over 5 candidates)

  Resolution difference:
    HIGH_RES: 64 RBF centers -> fine terrain coverage -> better discrimination
    LOW_RES:  8 RBF centers  -> coarse terrain coverage -> may miss hazard regions

  Primary metric: harm_rate during eval (lower = better terrain navigation)
  Secondary metric: residue_accuracy = Pearson r(residue_score_at_position,
                    hazard_field_max_at_position) over eval episodes.

Pre-registered acceptance criteria:
  C1: harm_rate_HIGH <= harm_rate_LOW (each seed)
      High resolution must not be worse than low resolution.

  C2: harm_rate_LOW - harm_rate_HIGH >= THRESH_C2_HARM_DELTA (mean across seeds)
      High resolution must provide a meaningful advantage.
      THRESH_C2_HARM_DELTA = 0.005 (0.5 pp absolute reduction).

  C3: residue_accuracy_HIGH > residue_accuracy_LOW (each seed)
      High resolution should show better terrain-hazard correlation.

  C4: n_harm_contacts_LOW >= THRESH_C4_MIN_HARM (each seed, data quality)
      Need sufficient harm events to measure harm rate.
      THRESH_C4_MIN_HARM = 5

  C5: e2_world_r2 >= THRESH_C5_E2_QUALITY (shared E2 quality gate)
      THRESH_C5_E2_QUALITY = 0.10

Outcome:
  PASS: C1 + C2 + C3 + C4 + C5 (all seeds)
  INCONCLUSIVE: NOT C4 or NOT C5
  FAIL: otherwise

Evidence direction:
  "supports"  if PASS -> retain high resolution for R(x,t)
  "weakens"   if C4+C5 pass but C1 fails -> resolution does not matter
  "mixed"     otherwise

Note: This addresses Q-002 as an open question. A PASS suggests finer resolution
is beneficial; a FAIL suggests Q-002's answer is "resolution is not critical at
this scale." Neither definitively resolves Q-002 but reduces uncertainty.

Estimated runtime: ~100 min on DLAPTOP-4.local (3 seeds x 300 train + 80 eval eps)
"""

import copy
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import E2Config, E3Config, ResidueConfig
from ree_core.predictors.e2_fast import E2FastPredictor
from ree_core.predictors.e3_selector import E3TrajectorySelector
from ree_core.residue.field import ResidueField

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_TYPE    = "v3_exq_215_q002_residue_resolution_pair"
CLAIM_IDS          = ["Q-002"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 7, 13]

# Environment
ENV_SIZE      = 10
NUM_HAZARDS   = 3
NUM_RESOURCES = 3
HAZARD_HARM   = 0.5

# Architecture
WORLD_DIM     = 32
SELF_DIM      = 16
ACTION_DIM    = 5
WORLD_OBS_DIM = 250
BODY_OBS_DIM  = 12

# Resolution parameters
HIGH_RES_NUM_CENTERS = 64
LOW_RES_NUM_CENTERS  = 8

# Training
LR               = 3e-4
N_TRAIN_EPISODES = 300
STEPS_PER_TRAIN  = 150

# Eval
N_EVAL_EPISODES = 80
STEPS_PER_EVAL  = 150
N_CANDIDATES    = 5     # greedy action selection: try N random actions, pick best

# Pre-registered thresholds
THRESH_C2_HARM_DELTA  = 0.005
THRESH_C4_MIN_HARM    = 5
THRESH_C5_E2_QUALITY  = 0.10


# ---------------------------------------------------------------------------
# Inline encoder
# ---------------------------------------------------------------------------

class WorldEncoder(nn.Module):
    def __init__(self, in_dim: int = WORLD_OBS_DIM, out_dim: int = WORLD_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Build modules (shared encoder, per-condition residue field)
# ---------------------------------------------------------------------------

def build_shared_modules(seed: int) -> dict:
    """Build encoder and E2 (shared across conditions)."""
    torch.manual_seed(seed)

    world_enc = WorldEncoder()

    e2_cfg = E2Config(
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=64,
    )
    e2 = E2FastPredictor(e2_cfg)

    e3_cfg = E3Config(
        world_dim=WORLD_DIM,
        hidden_dim=64,
        lambda_ethical=1.0,
        rho_residue=0.5,
    )

    return {"world_enc": world_enc, "e2": e2, "e3_cfg": e3_cfg}


def build_residue_field(num_centers: int) -> ResidueField:
    res_cfg = ResidueConfig(
        world_dim=WORLD_DIM,
        hidden_dim=32,
        accumulation_rate=0.2,
        num_basis_functions=num_centers,
        kernel_bandwidth=1.0,
    )
    return ResidueField(res_cfg)


# ---------------------------------------------------------------------------
# Training phase (shared encoder + E2)
# ---------------------------------------------------------------------------

def train_shared(shared: dict, seed: int, dry_run: bool = False) -> dict:
    """
    Train WorldEncoder and E2 on world prediction.
    Returns e2_world_r2 diagnostic.
    Residue fields are trained separately in each condition.
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)

    n_eps   = 10 if dry_run else N_TRAIN_EPISODES
    n_steps = 30 if dry_run else STEPS_PER_TRAIN

    env = CausalGridWorld(
        size=ENV_SIZE,
        num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES,
        hazard_harm=HAZARD_HARM,
        use_proxy_fields=True,
        seed=seed,
    )

    world_enc = shared["world_enc"]
    e2        = shared["e2"]

    optimizer = optim.Adam(
        list(world_enc.parameters()) + list(e2.parameters()), lr=LR
    )

    buffer: List = []
    MAX_BUF = 15000

    for ep in range(n_eps):
        flat_obs, obs_dict = env.reset()
        ws = obs_dict["world_state"].float()

        for _ in range(n_steps):
            a = rng.randint(0, ACTION_DIM - 1)
            a_onehot = torch.zeros(ACTION_DIM)
            a_onehot[a] = 1.0

            flat_next, harm_signal, done, info, obs_dict_next = env.step(a)
            ws_next = obs_dict_next["world_state"].float()
            harm_val = abs(float(harm_signal)) if harm_signal < 0 else 0.0

            buffer.append((ws.clone(), a_onehot.clone(), ws_next.clone(), harm_val))
            if len(buffer) > MAX_BUF:
                buffer.pop(0)

            ws = ws_next
            if done:
                break

        if len(buffer) >= 64:
            batch     = random.sample(buffer, min(128, len(buffer)))
            ws_b      = torch.stack([b[0] for b in batch])
            a_b       = torch.stack([b[1] for b in batch])
            ws_next_b = torch.stack([b[2] for b in batch])

            z_w      = world_enc(ws_b)
            z_w_tgt  = world_enc(ws_next_b).detach()
            z_w_pred = e2.world_forward(z_w, a_b)
            loss     = F.mse_loss(z_w_pred, z_w_tgt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Compute e2_world_r2
    world_enc.eval()
    e2.eval()
    e2_world_r2 = 0.0
    if len(buffer) >= 100:
        eval_buf  = buffer[-200:]
        ws_e      = torch.stack([b[0] for b in eval_buf])
        a_e       = torch.stack([b[1] for b in eval_buf])
        ws_next_e = torch.stack([b[2] for b in eval_buf])
        with torch.no_grad():
            z_e    = world_enc(ws_e)
            z_pred = e2.world_forward(z_e, a_e)
            z_tgt  = world_enc(ws_next_e)
        ss_res     = ((z_pred - z_tgt) ** 2).sum().item()
        ss_tot     = ((z_tgt - z_tgt.mean(0)) ** 2).sum().item()
        e2_world_r2 = 1.0 - (ss_res / (ss_tot + 1e-8))

    return {"e2_world_r2": e2_world_r2, "buffer": buffer}


# ---------------------------------------------------------------------------
# Populate residue field (condition-specific accumulation)
# ---------------------------------------------------------------------------

def populate_residue(
    residue: ResidueField,
    shared: dict,
    buffer: list,
    seed: int,
) -> dict:
    """
    Accumulate residue at harm events using the shared encoder.
    Uses a replay of the training buffer's harm events.
    """
    world_enc = shared["world_enc"]
    world_enc.eval()

    harm_events = 0
    for ws_b, a_b, ws_next_b, harm_val in buffer:
        if harm_val > 0.01:
            with torch.no_grad():
                z_w = world_enc(ws_b.unsqueeze(0))
            residue.accumulate(z_w, harm_magnitude=harm_val, hypothesis_tag=False)
            harm_events += 1

    return {
        "residue_total":     float(residue.total_residue.item()),
        "num_harm_events":   int(residue.num_harm_events.item()),
        "harm_events_replay": harm_events,
    }


# ---------------------------------------------------------------------------
# Eval phase
# ---------------------------------------------------------------------------

def eval_condition(
    condition: str,
    shared: dict,
    residue: ResidueField,
    seed: int,
    dry_run: bool = False,
) -> dict:
    """
    Evaluate harm rate and residue accuracy with greedy residue-avoidance policy.

    Action selection: for each step, sample N_CANDIDATES random actions, predict
    the next z_world for each using E2, evaluate residue score, take action with
    lowest residue score.
    """
    rng = random.Random(seed + 7777)
    torch.manual_seed(seed + 7777)

    n_eps   = 5  if dry_run else N_EVAL_EPISODES
    n_steps = 30 if dry_run else STEPS_PER_EVAL

    env = CausalGridWorld(
        size=ENV_SIZE,
        num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES,
        hazard_harm=HAZARD_HARM,
        use_proxy_fields=True,
        seed=seed + 7777,
    )

    world_enc = shared["world_enc"]
    e2        = shared["e2"]
    world_enc.eval()
    e2.eval()

    harm_steps   = 0
    total_steps  = 0
    residue_scores: List[float] = []
    hazard_proxies: List[float] = []

    for ep in range(n_eps):
        flat_obs, obs_dict = env.reset()
        ws = obs_dict["world_state"].float()

        for _ in range(n_steps):
            # Greedy residue-avoidance: pick action with lowest predicted residue
            with torch.no_grad():
                z_w = world_enc(ws.unsqueeze(0))
                best_a    = rng.randint(0, ACTION_DIM - 1)
                best_score = float("inf")

                for cand_a in range(ACTION_DIM):
                    a_onehot_c = torch.zeros(1, ACTION_DIM)
                    a_onehot_c[0, cand_a] = 1.0
                    z_w_next   = e2.world_forward(z_w, a_onehot_c)
                    score      = float(residue.evaluate(z_w_next).item())
                    if score < best_score:
                        best_score = score
                        best_a     = cand_a

                # Record residue score at current position for accuracy metric
                curr_score = float(residue.evaluate(z_w).item())

            a_onehot = torch.zeros(ACTION_DIM)
            a_onehot[best_a] = 1.0

            flat_next, harm_signal, done, info, obs_dict_next = env.step(best_a)
            ws_next = obs_dict_next["world_state"].float()
            total_steps += 1

            if harm_signal < 0:
                harm_steps += 1

            # Hazard proximity proxy for accuracy metric
            hfv = obs_dict_next.get("hazard_field_view", torch.zeros(25)).float()
            hazard_proxy = float(hfv.max().item())

            residue_scores.append(curr_score)
            hazard_proxies.append(hazard_proxy)

            ws = ws_next
            if done:
                break

    harm_rate = harm_steps / max(total_steps, 1)

    residue_accuracy = 0.0
    if len(residue_scores) >= 10:
        try:
            rs = np.array(residue_scores, dtype=float)
            hp = np.array(hazard_proxies, dtype=float)
            rs_std = rs.std()
            hp_std = hp.std()
            if rs_std > 1e-8 and hp_std > 1e-8:
                r = float(np.corrcoef(rs, hp)[0, 1])
                residue_accuracy = r if not (r != r) else 0.0
        except Exception:
            residue_accuracy = 0.0

    return {
        "harm_rate":        harm_rate,
        "harm_steps":       harm_steps,
        "total_steps":      total_steps,
        "residue_accuracy": residue_accuracy,
        "n_data_points":    len(residue_scores),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> dict:
    per_seed      = {}
    high_harm     = []
    low_harm      = []
    high_acc      = []
    low_acc       = []
    e2_r2_vals    = []
    low_harm_contacts = []

    for seed in SEEDS:
        print(f"[EXQ-215] seed={seed} Building modules...", flush=True)
        shared = build_shared_modules(seed)

        print(f"[EXQ-215] seed={seed} Training shared encoder+E2...", flush=True)
        train_result = train_shared(shared, seed, dry_run=dry_run)
        e2_r2 = train_result["e2_world_r2"]
        e2_r2_vals.append(e2_r2)
        buffer = train_result["buffer"]
        print(f"[EXQ-215] seed={seed} e2_r2={e2_r2:.4f}", flush=True)

        # Build per-condition residue fields
        res_high = build_residue_field(HIGH_RES_NUM_CENTERS)
        res_low  = build_residue_field(LOW_RES_NUM_CENTERS)

        # Populate residue fields from training buffer
        pop_high = populate_residue(res_high, shared, buffer, seed)
        pop_low  = populate_residue(res_low,  shared, buffer, seed)
        print(
            f"[EXQ-215] seed={seed} "
            f"HIGH_RES residue={pop_high['residue_total']:.4f} events={pop_high['num_harm_events']} "
            f"LOW_RES residue={pop_low['residue_total']:.4f} events={pop_low['num_harm_events']}",
            flush=True,
        )

        # Eval
        print(f"[EXQ-215] seed={seed} Eval HIGH_RES...", flush=True)
        high_res = eval_condition("HIGH_RES", shared, res_high, seed, dry_run=dry_run)
        print(f"[EXQ-215] seed={seed} Eval LOW_RES...", flush=True)
        low_res  = eval_condition("LOW_RES",  shared, res_low,  seed, dry_run=dry_run)

        high_harm.append(high_res["harm_rate"])
        low_harm.append(low_res["harm_rate"])
        high_acc.append(high_res["residue_accuracy"])
        low_acc.append(low_res["residue_accuracy"])
        low_harm_contacts.append(low_res["harm_steps"])

        print(
            f"[EXQ-215] seed={seed} "
            f"HIGH_harm={high_res['harm_rate']:.4f} LOW_harm={low_res['harm_rate']:.4f} "
            f"delta={low_res['harm_rate']-high_res['harm_rate']:.4f} "
            f"HIGH_acc={high_res['residue_accuracy']:.4f} LOW_acc={low_res['residue_accuracy']:.4f}",
            flush=True,
        )

        per_seed[str(seed)] = {
            "train": {"e2_world_r2": e2_r2},
            "populate_high": pop_high,
            "populate_low":  pop_low,
            "HIGH_RES": high_res,
            "LOW_RES":  low_res,
            "c1_direction_pass": high_res["harm_rate"] <= low_res["harm_rate"],
            "c3_acc_direction_pass": high_res["residue_accuracy"] > low_res["residue_accuracy"],
            "harm_delta": low_res["harm_rate"] - high_res["harm_rate"],
        }

    # -----------------------------------------------------------------------
    # Criteria
    # -----------------------------------------------------------------------
    e2_r2_mean     = sum(e2_r2_vals) / len(e2_r2_vals)
    mean_harm_delta = sum(low_harm[i] - high_harm[i] for i in range(len(SEEDS))) / len(SEEDS)

    c1_pass = all(per_seed[str(s)]["c1_direction_pass"] for s in SEEDS)
    c2_pass = mean_harm_delta >= THRESH_C2_HARM_DELTA
    c3_pass = all(per_seed[str(s)]["c3_acc_direction_pass"] for s in SEEDS)
    c4_pass = all(low_harm_contacts[i] >= THRESH_C4_MIN_HARM for i in range(len(SEEDS)))
    c5_pass = e2_r2_mean >= THRESH_C5_E2_QUALITY

    criteria = {
        "C1_harm_direction_all_seeds": c1_pass,
        "C2_mean_harm_delta":          c2_pass,
        "C3_acc_direction_all_seeds":  c3_pass,
        "C4_data_quality":             c4_pass,
        "C5_e2_quality":               c5_pass,
    }

    print(f"[EXQ-215] Criteria: {criteria}", flush=True)
    print(
        f"[EXQ-215] mean_harm_delta={mean_harm_delta:.4f} e2_r2_mean={e2_r2_mean:.4f}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Outcome
    # -----------------------------------------------------------------------
    if not c4_pass or not c5_pass:
        outcome            = "INCONCLUSIVE"
        evidence_direction = "inconclusive"
        decision           = "inconclusive"
    elif c1_pass and c2_pass and c3_pass:
        outcome            = "PASS"
        evidence_direction = "supports"
        decision           = "retain_ree"
    elif not c1_pass and c4_pass:
        # Resolution does not help -- fine is no better than coarse
        outcome            = "FAIL"
        evidence_direction = "weakens"
        decision           = "hybridize"
    else:
        outcome            = "FAIL"
        evidence_direction = "mixed"
        decision           = "hybridize"

    print(f"[EXQ-215] outcome={outcome} evidence_direction={evidence_direction}", flush=True)

    # -----------------------------------------------------------------------
    # Result pack
    # -----------------------------------------------------------------------
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts_str}_v3"

    pack = {
        "run_id":             run_id,
        "experiment_type":    EXPERIMENT_TYPE,
        "claim_ids":          CLAIM_IDS,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome":            outcome,
        "evidence_direction": evidence_direction,
        "evidence_class":     "discriminative_pair",
        "decision":           decision,
        "criteria":           criteria,
        "pre_registered_thresholds": {
            "HIGH_RES_NUM_CENTERS": HIGH_RES_NUM_CENTERS,
            "LOW_RES_NUM_CENTERS":  LOW_RES_NUM_CENTERS,
            "THRESH_C2_HARM_DELTA": THRESH_C2_HARM_DELTA,
            "THRESH_C4_MIN_HARM":   THRESH_C4_MIN_HARM,
            "THRESH_C5_E2_QUALITY": THRESH_C5_E2_QUALITY,
        },
        "summary_metrics": {
            "high_res_harm_mean":  sum(high_harm) / len(high_harm),
            "low_res_harm_mean":   sum(low_harm)  / len(low_harm),
            "mean_harm_delta":     mean_harm_delta,
            "high_res_acc_mean":   sum(high_acc) / len(high_acc),
            "low_res_acc_mean":    sum(low_acc)  / len(low_acc),
            "e2_world_r2_mean":    e2_r2_mean,
        },
        "seeds": SEEDS,
        "config": {
            "env_size":              ENV_SIZE,
            "num_hazards":           NUM_HAZARDS,
            "hazard_harm":           HAZARD_HARM,
            "world_dim":             WORLD_DIM,
            "action_dim":            ACTION_DIM,
            "high_res_num_centers":  HIGH_RES_NUM_CENTERS,
            "low_res_num_centers":   LOW_RES_NUM_CENTERS,
            "n_train_eps":           N_TRAIN_EPISODES,
            "n_eval_eps":            N_EVAL_EPISODES,
            "n_candidates":          N_CANDIDATES,
        },
        "scenario": (
            "discriminative_pair: HIGH_RES (ResidueField num_basis_functions=64) vs "
            "LOW_RES (num_basis_functions=8). Shared WorldEncoder + E2 trained on world "
            "prediction. Residue fields populated from training buffer harm events. "
            "Eval: greedy residue-avoidance policy (argmin E2-predicted residue across "
            "all 5 candidate actions). residue_accuracy = Pearson r(residue_score, "
            f"hazard_field_max). CausalGridWorld size={ENV_SIZE} hazards={NUM_HAZARDS}."
        ),
        "interpretation": (
            "PASS supports Q-002: higher RBF resolution (64 vs 8 centers) provides "
            "measurably better harm avoidance and residue-hazard correlation. "
            "Suggests Q-002 answer: fine resolution is beneficial for R(x,t). "
            "FAIL weakens Q-002: resolution does not matter at current scale. "
            "Q-002 answer may be: coarse resolution is sufficient, or the bottleneck "
            "is not RBF density but representation quality. INCONCLUSIVE: E2 not "
            "trained enough or insufficient harm contacts."
        ),
        "per_seed_results": per_seed,
        "supersedes": "V3-EXQ-170",    # EXQ-170 used inline arch; this uses ree_core.ResidueField
        "dry_run":   dry_run,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if not dry_run:
        out_dir = (
            Path(__file__).resolve().parents[2]
            / "REE_assembly" / "evidence" / "experiments"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{run_id}.json"
        with open(out_path, "w") as f:
            json.dump(pack, f, indent=2)
        print(f"[EXQ-215] Result written to: {out_path}", flush=True)
    else:
        print("[EXQ-215] dry_run: result NOT written.", flush=True)

    return pack


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    result = run_experiment(dry_run=dry_run)
    print(f"[EXQ-215] Done. outcome={result['outcome']}", flush=True)
