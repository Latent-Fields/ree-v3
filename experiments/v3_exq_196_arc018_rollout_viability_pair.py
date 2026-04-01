#!/opt/local/bin/python3
import copy
"""
V3-EXQ-196 -- ARC-018: Hippocampal Rollout + Viability Mapping (Discriminative Pair)

Claim: ARC-018 -- hippocampus.rollout_viability_mapping
  "Hippocampus generates explicit rollouts and post-commitment viability mapping."

Prior failures: EXQ-053 (FAIL), EXQ-120 (FAIL), EXQ-172 (FAIL).
Root cause analysis:
  - EXQ-172 used inline toy models instead of real ree_core modules (HippocampalModule,
    ResidueField, E3TrajectorySelector). The "viability mapping" was never actually
    tested -- it was just multi-step rollout vs greedy, with no residue field terrain.
  - EXQ-053/120 had similar issues with reframing: E2 latent space encodes action
    consequences (world-effects), not sensory state transitions.

Redesign: This experiment uses the REAL ree_core modules:
  - ResidueField: RBF terrain over z_world, accumulates harm locations
  - HippocampalModule: CEM in action-object space O, terrain-guided by residue field
  - E2FastPredictor: world_forward() for z_world predictions, action_object() for SD-004
  - E3TrajectorySelector: harm_eval() on z_world, score_trajectory() with residue cost

Design: discriminative_pair
  Condition A: VIABILITY_MAPPED
    - ResidueField accumulates harm at z_world locations during training
    - HippocampalModule proposes trajectories navigating residue terrain
    - E3 selects trajectory using J(zeta) = F + lambda*M + rho*Phi_R
    - Viability map = residue field terrain shaped by harm history

  Condition B: VIABILITY_ABLATED
    - ResidueField weights are ZEROED after training (terrain flattened)
    - HippocampalModule still proposes trajectories but terrain is flat
    - E3 selects but Phi_R(zeta) = 0 for all trajectories
    - Same models, same training -- only the viability map is ablated

  Both conditions share:
    - Same SplitEncoder, E2, E3, HippocampalModule trained together
    - CausalGridWorldV2 (proxy fields: hazard/resource gradients visible)
    - 400 training episodes, 150 steps/ep, Adam lr=3e-4
    - alpha_world=0.9 (SD-008: no event suppression)
    - Eval: 40 episodes x 150 steps per condition per seed

Pre-registered acceptance criteria (numeric thresholds):
  C1: harm_rate_MAPPED < harm_rate_ABLATED (each seed independently)
  C2: mean harm_advantage (ablated - mapped) >= 0.003
  C3: residue_total > 0 at end of training (viability map populated)
  C4: e2_world_r2 >= 0.10 (E2 world_forward trained adequately)
  C5: n_harm_contacts_ABLATED >= 3 per seed (data quality)

Outcome:
  PASS        if C1 + C2 + C3 + C4 + C5
  INCONCLUSIVE if NOT C4 or NOT C5 or NOT C3
  FAIL        otherwise

Evidence direction:
  "supports"  if PASS
  "weakens"   if NOT C1 (all seeds) and C5
  "mixed"     otherwise

Scoring:
  PASS -> retain_ree (viability mapping validated)
  FAIL -> retire_ree_claim (viability mapping provides no benefit)
  INCONCLUSIVE -> inconclusive
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

from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import (
    E2Config, E3Config, HippocampalConfig, ResidueConfig, LatentStackConfig,
)
from ree_core.predictors.e2_fast import E2FastPredictor, Trajectory
from ree_core.predictors.e3_selector import E3TrajectorySelector
from ree_core.residue.field import ResidueField
from ree_core.hippocampal.module import HippocampalModule

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_TYPE = "v3_exq_196_arc018_rollout_viability_pair"
CLAIM_IDS = ["ARC-018"]
SEEDS = [42, 7, 13]

# Environment -- use proxy fields for gradient visibility
ENV_SIZE = 8
NUM_HAZARDS = 3
NUM_RESOURCES = 3
HAZARD_HARM = 0.5   # enough harm to populate residue field

# Architecture -- V3 canonical dims
WORLD_DIM = 32
SELF_DIM = 32
ACTION_DIM = 4   # up/down/left/right (no stay for cleaner navigation)
ACTION_OBJECT_DIM = 16
BODY_OBS_DIM = 12   # CausalGridWorldV2 mode (use_proxy_fields=True)
WORLD_OBS_DIM = 250  # CausalGridWorldV2 mode

# Training
LR = 3e-4
N_TRAIN_EPISODES = 400
STEPS_PER_EPISODE = 150

# Eval
N_EVAL_EPISODES = 40
EVAL_STEPS = 150

# HippocampalModule CEM
CEM_CANDIDATES = 16
CEM_ITERATIONS = 3
CEM_HORIZON = 8
CEM_ELITE_FRAC = 0.25

# Pre-registered thresholds
THRESH_C2_ADVANTAGE = 0.003
THRESH_C4_E2_QUALITY = 0.10
THRESH_C5_MIN_CONTACTS = 3


# ---------------------------------------------------------------------------
# Inline encoder (lightweight SplitEncoder substitute for experiment)
# ---------------------------------------------------------------------------

class SimpleWorldEncoder(nn.Module):
    """Encode world_obs -> z_world (standalone for experiment training)."""

    def __init__(self, world_obs_dim: int = WORLD_OBS_DIM, world_dim: int = WORLD_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(world_obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, world_dim),
            nn.LayerNorm(world_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleSelfEncoder(nn.Module):
    """Encode body_obs -> z_self (standalone for experiment)."""

    def __init__(self, body_obs_dim: int = BODY_OBS_DIM, self_dim: int = SELF_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(body_obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self_dim),
            nn.LayerNorm(self_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Build the agent module set
# ---------------------------------------------------------------------------

def build_modules(seed: int):
    """Construct E2, E3, ResidueField, HippocampalModule with proper configs."""
    torch.manual_seed(seed)

    e2_cfg = E2Config(
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=128,
        action_object_dim=ACTION_OBJECT_DIM,
        rollout_horizon=CEM_HORIZON,
    )
    e2 = E2FastPredictor(e2_cfg)

    res_cfg = ResidueConfig(
        world_dim=WORLD_DIM,
        hidden_dim=64,
        accumulation_rate=0.15,
        num_basis_functions=32,
        kernel_bandwidth=1.0,
    )
    residue_field = ResidueField(res_cfg)

    e3_cfg = E3Config(
        world_dim=WORLD_DIM,
        hidden_dim=64,
        lambda_ethical=1.0,
        rho_residue=0.5,
    )
    e3 = E3TrajectorySelector(e3_cfg, residue_field=residue_field)

    hipp_cfg = HippocampalConfig(
        world_dim=WORLD_DIM,
        action_dim=ACTION_DIM,
        action_object_dim=ACTION_OBJECT_DIM,
        hidden_dim=128,
        horizon=CEM_HORIZON,
        num_candidates=CEM_CANDIDATES,
        num_cem_iterations=CEM_ITERATIONS,
        elite_fraction=CEM_ELITE_FRAC,
    )
    hippocampal = HippocampalModule(hipp_cfg, e2=e2, residue_field=residue_field)

    world_enc = SimpleWorldEncoder()
    self_enc = SimpleSelfEncoder()

    return {
        "e2": e2,
        "e3": e3,
        "residue_field": residue_field,
        "hippocampal": hippocampal,
        "world_enc": world_enc,
        "self_enc": self_enc,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_agent(modules: dict, seed: int, dry_run: bool = False):
    """
    Train all modules jointly on CausalGridWorldV2 episodes.

    Training targets:
      - E2 world_forward: MSE on z_world_next vs encoder(world_obs_next)
      - E2 self forward: MSE on z_self_next vs encoder(body_obs_next)
      - E3 harm_eval: MSE on harm_eval(z_world) vs harm_exposure signal
      - ResidueField: accumulates at z_world locations where harm occurs

    Returns e2_world_r2 diagnostic and residue statistics.
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)

    env = CausalGridWorld(
        size=ENV_SIZE,
        num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES,
        hazard_harm=HAZARD_HARM,
        use_proxy_fields=True,
        seed=seed,
    )

    e2 = modules["e2"]
    e3 = modules["e3"]
    residue_field = modules["residue_field"]
    world_enc = modules["world_enc"]
    self_enc = modules["self_enc"]

    all_params = (
        list(e2.parameters())
        + list(e3.parameters())
        + list(world_enc.parameters())
        + list(self_enc.parameters())
    )
    optimizer = optim.Adam(all_params, lr=LR)

    n_eps = N_TRAIN_EPISODES if not dry_run else 5
    steps = STEPS_PER_EPISODE if not dry_run else 20

    # Replay buffer: (world_obs_t, body_obs_t, action, world_obs_next, body_obs_next, harm)
    buffer = []
    MAX_BUFFER = 15000

    for ep in range(n_eps):
        flat_obs, obs_dict = env.reset()
        ws = obs_dict["world_state"].float()
        bs = obs_dict["body_state"].float()

        for step in range(steps):
            a = rng.randint(0, ACTION_DIM - 1)
            action_onehot = torch.zeros(ACTION_DIM)
            action_onehot[a] = 1.0

            flat_next, reward, done, info, obs_dict_next = env.step(a)
            ws_next = obs_dict_next["world_state"].float()
            bs_next = obs_dict_next["body_state"].float()

            harm_val = info.get("harm_exposure", max(-reward, 0.0))

            buffer.append((
                ws.clone(), bs.clone(), action_onehot.clone(),
                ws_next.clone(), bs_next.clone(), float(harm_val),
            ))
            if len(buffer) > MAX_BUFFER:
                buffer.pop(0)

            # Accumulate residue at harm locations
            if harm_val > 0.01:
                with torch.no_grad():
                    z_w = world_enc(ws.unsqueeze(0))
                residue_field.accumulate(
                    z_w, harm_magnitude=harm_val, hypothesis_tag=False,
                )

            ws = ws_next
            bs = bs_next
            if done:
                break

        # Train on mini-batch from buffer
        if len(buffer) >= 64:
            batch = random.sample(buffer, min(128, len(buffer)))
            ws_b = torch.stack([b[0] for b in batch])
            bs_b = torch.stack([b[1] for b in batch])
            act_b = torch.stack([b[2] for b in batch])
            ws_next_b = torch.stack([b[3] for b in batch])
            bs_next_b = torch.stack([b[4] for b in batch])
            harm_b = torch.tensor(
                [b[5] for b in batch], dtype=torch.float32
            ).unsqueeze(1)

            z_world = world_enc(ws_b)
            z_self = self_enc(bs_b)
            z_world_next_target = world_enc(ws_next_b).detach()
            z_self_next_target = self_enc(bs_next_b).detach()

            # E2 world_forward loss
            z_world_next_pred = e2.world_forward(z_world, act_b)
            loss_e2_world = F.mse_loss(z_world_next_pred, z_world_next_target)

            # E2 self forward loss
            z_self_next_pred = e2.predict_next_self(z_self, act_b)
            loss_e2_self = F.mse_loss(z_self_next_pred, z_self_next_target)

            # E3 harm_eval loss
            harm_pred = e3.harm_eval(z_world)
            loss_harm = F.mse_loss(harm_pred, harm_b)

            loss = loss_e2_world + loss_e2_self + loss_harm
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (ep + 1) % 100 == 0:
            print(
                f"  [ep {ep+1}/{n_eps}] buffer={len(buffer)} "
                f"residue_events={residue_field.num_harm_events.item()}",
                flush=True,
            )

    # Compute e2_world_r2 on held-out buffer tail
    e2_world_r2 = 0.0
    world_enc.eval()
    self_enc.eval()
    e2.eval()
    e3.eval()

    if len(buffer) >= 100:
        eval_buf = buffer[-200:]
        ws_e = torch.stack([b[0] for b in eval_buf])
        act_e = torch.stack([b[2] for b in eval_buf])
        ws_next_e = torch.stack([b[3] for b in eval_buf])

        with torch.no_grad():
            z_e = world_enc(ws_e)
            z_pred = e2.world_forward(z_e, act_e)
            z_tgt = world_enc(ws_next_e)

        ss_res = ((z_pred - z_tgt) ** 2).sum().item()
        ss_tot = ((z_tgt - z_tgt.mean(0)) ** 2).sum().item()
        e2_world_r2 = 1.0 - (ss_res / (ss_tot + 1e-8))

    residue_stats = residue_field.get_statistics()
    train_metrics = {
        "e2_world_r2": e2_world_r2,
        "residue_total": residue_stats["total_residue"].item(),
        "residue_harm_events": residue_stats["num_harm_events"].item(),
        "residue_active_centers": residue_stats["active_centers"].item(),
    }

    # Restore training mode for modules that need it
    world_enc.train()
    self_enc.train()
    e2.train()
    e3.train()

    return train_metrics


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def eval_condition(
    condition: str,
    modules: dict,
    seed: int,
    dry_run: bool = False,
) -> dict:
    """
    Evaluate a condition: VIABILITY_MAPPED or VIABILITY_ABLATED.

    For VIABILITY_ABLATED: zero out residue field weights before eval.

    Action selection:
      1. HippocampalModule proposes trajectories via CEM in action-object space
      2. E3 selects best trajectory via score_trajectory() (includes Phi_R)
      3. First action of selected trajectory is executed

    Returns: harm_rate, n_harm_contacts, total_steps, mean_residue_cost
    """
    rng = random.Random(seed + 7777)
    torch.manual_seed(seed + 7777)

    env = CausalGridWorld(
        size=ENV_SIZE,
        num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES,
        hazard_harm=HAZARD_HARM,
        use_proxy_fields=True,
        seed=seed + 7777,
    )

    e2 = modules["e2"]
    e3 = modules["e3"]
    residue_field = modules["residue_field"]
    hippocampal = modules["hippocampal"]
    world_enc = modules["world_enc"]
    self_enc = modules["self_enc"]

    world_enc.eval()
    self_enc.eval()
    e2.eval()
    e3.eval()

    # VIABILITY_ABLATED: zero residue field weights to flatten terrain
    if condition == "VIABILITY_ABLATED":
        with torch.no_grad():
            residue_field.rbf_field.weights.zero_()
            residue_field.rbf_field.active_mask.zero_()
            # Also zero the neural field contribution by zeroing last layer
            for layer in residue_field.neural_field:
                if hasattr(layer, "weight"):
                    layer.weight.zero_()
                if hasattr(layer, "bias"):
                    layer.bias.zero_()

    n_episodes = N_EVAL_EPISODES if not dry_run else 3
    eval_steps = EVAL_STEPS if not dry_run else 20
    harm_contacts = 0
    total_steps = 0
    residue_costs_sum = 0.0

    for ep in range(n_episodes):
        flat_obs, obs_dict = env.reset()
        ws = obs_dict["world_state"].float()
        bs = obs_dict["body_state"].float()

        for step in range(eval_steps):
            with torch.no_grad():
                z_world = world_enc(ws.unsqueeze(0))
                z_self = self_enc(bs.unsqueeze(0))

                # HippocampalModule proposes trajectories via CEM
                candidates = hippocampal.propose_trajectories(
                    z_world, z_self=z_self,
                    num_candidates=CEM_CANDIDATES,
                )

                if not candidates:
                    # Fallback: random action
                    a = rng.randint(0, ACTION_DIM - 1)
                else:
                    # E3 selects best trajectory
                    result = e3.select(candidates, temperature=0.5)
                    # Decode selected action to discrete
                    action_vec = result.selected_action.squeeze(0)  # [action_dim]
                    a = int(action_vec.argmax().item())

                    # Track residue cost for diagnostics
                    residue_costs_sum += e3.compute_residue_cost(
                        result.selected_trajectory
                    ).mean().item()

            flat_next, reward, done, info, obs_dict_next = env.step(a)
            ws = obs_dict_next["world_state"].float()
            bs = obs_dict_next["body_state"].float()
            total_steps += 1

            harm_exp = info.get("harm_exposure", 0.0)
            if harm_exp > 0.0:
                harm_contacts += 1

            if done:
                break

    harm_rate = harm_contacts / max(total_steps, 1)
    mean_residue_cost = residue_costs_sum / max(total_steps, 1)

    return {
        "harm_rate": harm_rate,
        "n_harm_contacts": harm_contacts,
        "total_steps": total_steps,
        "mean_residue_cost": mean_residue_cost,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> dict:
    per_seed_results = {}
    e2_r2_vals = []

    mapped_harm_rates = []
    ablated_harm_rates = []
    ablated_contacts_per_seed = []

    for seed in SEEDS:
        print(f"[seed {seed}] Building modules...", flush=True)
        modules = build_modules(seed)

        print(f"[seed {seed}] Training agent...", flush=True)
        train_metrics = train_agent(modules, seed, dry_run=dry_run)
        e2_r2 = train_metrics["e2_world_r2"]
        e2_r2_vals.append(e2_r2)
        print(
            f"[seed {seed}] e2_world_r2={e2_r2:.4f} "
            f"residue_total={train_metrics['residue_total']:.4f} "
            f"harm_events={train_metrics['residue_harm_events']}",
            flush=True,
        )

        # Eval VIABILITY_MAPPED first (with live residue field)
        print(f"[seed {seed}] Evaluating VIABILITY_MAPPED...", flush=True)
        mapped_res = eval_condition("VIABILITY_MAPPED", modules, seed, dry_run=dry_run)

        # Deep-copy trained modules so ABLATED has identical weights
        # (rebuilding + retraining risks weight divergence from global RNG state)
        print(f"[seed {seed}] Deep-copying for VIABILITY_ABLATED...", flush=True)
        modules_ablated = copy.deepcopy(modules)

        print(f"[seed {seed}] Evaluating VIABILITY_ABLATED...", flush=True)
        ablated_res = eval_condition(
            "VIABILITY_ABLATED", modules_ablated, seed, dry_run=dry_run,
        )

        mapped_harm_rates.append(mapped_res["harm_rate"])
        ablated_harm_rates.append(ablated_res["harm_rate"])
        ablated_contacts_per_seed.append(ablated_res["n_harm_contacts"])

        harm_adv = ablated_res["harm_rate"] - mapped_res["harm_rate"]
        print(
            f"[seed {seed}] MAPPED harm_rate={mapped_res['harm_rate']:.4f} "
            f"ABLATED harm_rate={ablated_res['harm_rate']:.4f} "
            f"advantage={harm_adv:.4f} "
            f"mapped_residue_cost={mapped_res['mean_residue_cost']:.4f}",
            flush=True,
        )

        per_seed_results[str(seed)] = {
            "e2_world_r2": e2_r2,
            "train_metrics": train_metrics,
            "VIABILITY_MAPPED": mapped_res,
            "VIABILITY_ABLATED": ablated_res,
            "harm_advantage": harm_adv,
            "c1_direction_pass": mapped_res["harm_rate"] < ablated_res["harm_rate"],
        }

    # -----------------------------------------------------------------------
    # Criteria evaluation
    # -----------------------------------------------------------------------
    e2_r2_mean = sum(e2_r2_vals) / len(e2_r2_vals)
    harm_advantage_mean = sum(
        ablated_harm_rates[i] - mapped_harm_rates[i] for i in range(len(SEEDS))
    ) / len(SEEDS)

    # C1: mapped < ablated for each seed
    c1_pass = all(
        per_seed_results[str(s)]["c1_direction_pass"] for s in SEEDS
    )
    # C2: mean advantage >= threshold
    c2_pass = harm_advantage_mean >= THRESH_C2_ADVANTAGE
    # C3: residue field was populated during training
    c3_pass = all(
        per_seed_results[str(s)]["train_metrics"]["residue_total"] > 0
        for s in SEEDS
    )
    # C4: E2 world_forward quality
    c4_pass = e2_r2_mean >= THRESH_C4_E2_QUALITY
    # C5: data quality -- enough harm contacts in ablated condition
    c5_pass = all(
        ablated_contacts_per_seed[i] >= THRESH_C5_MIN_CONTACTS
        for i in range(len(SEEDS))
    )

    criteria = {
        "C1_direction_all_seeds": c1_pass,
        "C2_harm_advantage": c2_pass,
        "C3_residue_populated": c3_pass,
        "C4_e2_quality": c4_pass,
        "C5_data_quality": c5_pass,
    }

    print(f"Criteria: {criteria}", flush=True)
    print(
        f"harm_advantage_mean={harm_advantage_mean:.4f}  "
        f"e2_r2_mean={e2_r2_mean:.4f}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Outcome + evidence direction
    # -----------------------------------------------------------------------
    if not c4_pass or not c5_pass or not c3_pass:
        outcome = "INCONCLUSIVE"
        evidence_direction = "inconclusive"
    elif c1_pass and c2_pass:
        outcome = "PASS"
        evidence_direction = "supports"
    elif not c1_pass and c5_pass:
        outcome = "FAIL"
        evidence_direction = "weakens"
    else:
        outcome = "FAIL"
        evidence_direction = "mixed"

    # Decision scoring
    if outcome == "PASS":
        decision = "retain_ree"
    elif outcome == "FAIL" and evidence_direction == "weakens":
        decision = "retire_ree_claim"
    elif outcome == "FAIL":
        decision = "hybridize"
    else:
        decision = "inconclusive"

    print(
        f"Outcome: {outcome}  evidence_direction: {evidence_direction}  "
        f"decision: {decision}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Result pack
    # -----------------------------------------------------------------------
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
        "decision": decision,
        "criteria": criteria,
        "pre_registered_thresholds": {
            "THRESH_C2_ADVANTAGE": THRESH_C2_ADVANTAGE,
            "THRESH_C4_E2_QUALITY": THRESH_C4_E2_QUALITY,
            "THRESH_C5_MIN_CONTACTS": THRESH_C5_MIN_CONTACTS,
        },
        "summary_metrics": {
            "harm_rate_MAPPED_mean": sum(mapped_harm_rates) / len(mapped_harm_rates),
            "harm_rate_ABLATED_mean": sum(ablated_harm_rates) / len(ablated_harm_rates),
            "harm_advantage_mean": harm_advantage_mean,
            "e2_world_r2_mean": e2_r2_mean,
            "ablated_harm_contacts_per_seed": ablated_contacts_per_seed,
        },
        "seeds": SEEDS,
        "config": {
            "env_size": ENV_SIZE,
            "num_hazards": NUM_HAZARDS,
            "num_resources": NUM_RESOURCES,
            "hazard_harm": HAZARD_HARM,
            "world_dim": WORLD_DIM,
            "self_dim": SELF_DIM,
            "action_dim": ACTION_DIM,
            "action_object_dim": ACTION_OBJECT_DIM,
            "n_train_episodes": N_TRAIN_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "n_eval_episodes": N_EVAL_EPISODES,
            "eval_steps": EVAL_STEPS,
            "cem_candidates": CEM_CANDIDATES,
            "cem_iterations": CEM_ITERATIONS,
            "cem_horizon": CEM_HORIZON,
            "lr": LR,
        },
        "scenario": (
            "discriminative_pair: VIABILITY_MAPPED (ResidueField accumulates harm, "
            "HippocampalModule navigates residue terrain via CEM in action-object "
            "space, E3 selects with Phi_R cost) vs VIABILITY_ABLATED (same modules "
            "but residue field zeroed -- flat terrain, no viability map). "
            f"CausalGridWorldV2 size={ENV_SIZE} hazards={NUM_HAZARDS} "
            f"resources={NUM_RESOURCES}. Train {N_TRAIN_EPISODES} eps x "
            f"{STEPS_PER_EPISODE} steps. Eval {N_EVAL_EPISODES} eps x "
            f"{EVAL_STEPS} steps."
        ),
        "interpretation": (
            "PASS supports ARC-018: hippocampal viability mapping (residue field "
            "terrain guiding trajectory proposals) produces measurably better harm "
            "avoidance than flat-terrain proposals. The viability map mechanism "
            "validated: harm history shapes z_world terrain, HippocampalModule CEM "
            "navigates it, E3 Phi_R cost term creates preference for low-residue "
            "paths. FAIL weakens ARC-018: viability mapping provides no benefit "
            "over terrain-blind trajectory proposals -- residue field terrain does "
            "not usefully inform hippocampal CEM search. INCONCLUSIVE: E2 world "
            "model not trained sufficiently or insufficient harm contacts."
        ),
        "per_seed_results": per_seed_results,
        "supersedes": None,
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
