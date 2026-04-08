#!/opt/local/bin/python3
"""
V3-EXQ-260 -- SD-020 Affective Harm Surprise PE Training

Claims: SD-020, SD-011
EXPERIMENT_PURPOSE = "evidence"

Tests whether training z_harm_a on prediction error (surprise) rather than
raw accumulated harm produces a functionally superior urgency signal for E3.

Chen (2023, Front Neural Circuits) establishes that AIC responds to "unsigned
intensity PEs as a modality-unspecific aversive surprise signal," not raw state.
SD-020 claims z_harm_a should encode how SURPRISING current threat is, not how
high it is. This experiment tests that claim.

Design
------
3-condition comparison, 3 seeds:
  RAW_ACCUM:    current impl (harm_accum_head predicts accumulated harm scalar)
  SURPRISE_PE:  aux head predicts harm PE (actual_accum - predicted_accum)
  COMBINED:     both heads active (surprise + raw accum)

The harm PE target is computed from a simple linear predictor of harm_accum
from previous harm_accum (1-step). PE = actual - predicted. The aux head in
SURPRISE_PE trains on |PE| (unsigned, matching Chen's AIC unsigned PE claim).

Each condition:
  Phase 0 (P0): 120 episodes warmup (agent + aux losses)
  Phase 1 (P1): 60 episodes evaluation

Success criteria (>= 2/3 seeds):
  C1: urgency_corr_surprise > urgency_corr_raw
      (SURPRISE_PE urgency signal correlates better with threat CHANGES than RAW_ACCUM)
  C2: stream_corr(SURPRISE_PE) < stream_corr(RAW_ACCUM)
      (surprise-trained z_harm_a is MORE distinct from z_harm_s)
  C3: commit_sensitivity_surprise > commit_sensitivity_raw
      (SURPRISE_PE produces sharper commit threshold transitions in E3)

PASS: C1 AND C2 (>= 2/3 seeds). C3 is informational.
FAIL: either C1 or C2 not met.

Seeds: [42, 7, 13]
Env: CausalGridWorldV2 size=10, 3 hazards, 5 resources, hazard_harm=0.5
Est: ~90 min (DLAPTOP-4.local) -- 3 seeds x 3 conditions x 180 eps x 150 steps
"""

import sys
import json
import math
import random
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE    = "v3_exq_260_sd020_harm_surprise_pe"
CLAIM_IDS          = ["SD-020", "SD-011"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
SEED_PASS_QUOTA = 2  # >= 2/3 seeds

# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
BODY_OBS_DIM   = 12
WORLD_OBS_DIM  = 250
HARM_OBS_DIM   = 51
WORLD_DIM      = 32
Z_HARM_DIM     = 32
Z_HARM_A_DIM   = 16
ACTION_DIM     = 5   # CausalGridWorld has 5 actions (0-4)
HARM_HISTORY_LEN = 10
GRID_SIZE      = 10
NUM_HAZARDS    = 3
NUM_RESOURCES  = 5
HAZARD_HARM    = 0.5

SEEDS          = [42, 7, 13]
CONDITIONS     = ["RAW_ACCUM", "SURPRISE_PE", "COMBINED"]
P0_EPISODES    = 120
P1_EPISODES    = 60
STEPS_PER_EP   = 150


class HarmAccumPredictor(nn.Module):
    """Simple 1-step predictor for harm accumulation (generates PE target)."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, prev_accum: torch.Tensor) -> torch.Tensor:
        return self.linear(prev_accum)


def run_condition(seed: int, condition: str) -> Dict:
    """Run one seed x condition pair."""
    torch.manual_seed(seed)
    random.seed(seed)

    config = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        harm_obs_dim=HARM_OBS_DIM,
        world_dim=WORLD_DIM,
        action_dim=ACTION_DIM - 1,  # agent action_dim = env actions - 1
        z_harm_dim=Z_HARM_DIM,
        z_harm_a_dim=Z_HARM_A_DIM,
        alpha_world=0.9,
        use_harm_stream=True,
        use_affective_harm_stream=True,
        harm_history_len=HARM_HISTORY_LEN,
        z_harm_a_aux_loss_weight=0.1,
    )

    agent = REEAgent(config)
    env = CausalGridWorldV2(
        size=GRID_SIZE,
        num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES,
        hazard_harm=HAZARD_HARM,
        use_proxy_fields=True,
        harm_history_len=HARM_HISTORY_LEN,
        seed=seed,
    )

    # Harm accumulation predictor (for PE target)
    accum_predictor = HarmAccumPredictor()
    accum_opt = optim.Adam(accum_predictor.parameters(), lr=1e-3)

    agent_opt = optim.Adam(agent.latent_stack.parameters(), lr=1e-4)

    # Storage for evaluation
    urgency_vs_threat_change: List[Tuple[float, float]] = []
    stream_corrs: List[float] = []

    total_episodes = P0_EPISODES + P1_EPISODES

    for ep in range(total_episodes):
        _flat_obs, obs_dict = env.reset()
        agent.reset()
        prev_harm_accum = 0.0
        episode_harm_accum = 0.0

        z_harm_s_list: List[torch.Tensor] = []
        z_harm_a_list: List[torch.Tensor] = []
        urgency_list: List[float] = []
        threat_change_list: List[float] = []

        for step in range(STEPS_PER_EP):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm = obs_dict.get("harm_obs")
            obs_harm_a = obs_dict.get("harm_obs_a")
            obs_harm_hist = obs_dict.get("harm_history")

            # Sense (returns latent with grad for training)
            latent = agent.sense(
                obs_body, obs_world,
                obs_harm=obs_harm,
                obs_harm_a=obs_harm_a,
                obs_harm_history=obs_harm_hist,
            )

            # Random action (encoding quality experiment, not behavioral)
            action_idx = random.randint(0, ACTION_DIM - 1)

            # Current accumulated harm from obs_dict
            current_accum = float(obs_dict.get("accumulated_harm", 0.0))
            step_harm = current_accum - prev_harm_accum if current_accum > prev_harm_accum else 0.0
            episode_harm_accum += step_harm

            # Compute PE target
            prev_accum_t = torch.tensor([[prev_harm_accum]], dtype=torch.float32)
            with torch.no_grad():
                predicted_accum = accum_predictor(prev_accum_t).item()
            harm_pe = abs(current_accum - predicted_accum)

            # Train aux head based on condition
            if ep < P0_EPISODES:
                # Re-encode with gradient for training
                latent_grad = agent.sense(
                    obs_body, obs_world,
                    obs_harm=obs_harm,
                    obs_harm_a=obs_harm_a,
                    obs_harm_history=obs_harm_hist,
                )

                loss = torch.tensor(0.0)

                if latent_grad is not None and latent_grad.harm_accum_pred is not None:
                    if condition == "RAW_ACCUM":
                        target_val = episode_harm_accum / max(step + 1, 1)
                        loss = agent.compute_harm_accum_loss(target_val, latent_grad)
                    elif condition == "SURPRISE_PE":
                        pred = latent_grad.harm_accum_pred
                        if pred.dim() == 1:
                            pred = pred.unsqueeze(0)
                        pe_target = torch.tensor(
                            [[harm_pe]], dtype=torch.float32, device=pred.device
                        ).clamp(0, 1)
                        weight = getattr(config.latent, "z_harm_a_aux_loss_weight", 0.1)
                        loss = weight * F.mse_loss(pred, pe_target)
                    elif condition == "COMBINED":
                        target_val = episode_harm_accum / max(step + 1, 1)
                        raw_loss = agent.compute_harm_accum_loss(target_val, latent_grad)
                        pred = latent_grad.harm_accum_pred
                        if pred.dim() == 1:
                            pred = pred.unsqueeze(0)
                        pe_target = torch.tensor(
                            [[harm_pe]], dtype=torch.float32, device=pred.device
                        ).clamp(0, 1)
                        weight = getattr(config.latent, "z_harm_a_aux_loss_weight", 0.1)
                        pe_loss = weight * F.mse_loss(pred, pe_target)
                        loss = raw_loss + pe_loss

                    if loss.requires_grad:
                        agent_opt.zero_grad()
                        loss.backward()
                        agent_opt.step()

                # Train accumulation predictor
                accum_opt.zero_grad()
                pred_a = accum_predictor(prev_accum_t)
                target_a = torch.tensor([[current_accum]], dtype=torch.float32)
                accum_loss = F.mse_loss(pred_a, target_a)
                accum_loss.backward()
                accum_opt.step()

            # Evaluation phase: collect metrics
            if ep >= P0_EPISODES:
                det_latent = agent._current_latent
                if det_latent is not None:
                    if det_latent.z_harm is not None:
                        z_harm_s_list.append(det_latent.z_harm.squeeze())
                    if det_latent.z_harm_a is not None:
                        z_harm_a_list.append(det_latent.z_harm_a.squeeze())
                        # Urgency = z_harm_a norm (same as E3's urgency formula)
                        urgency_list.append(float(det_latent.z_harm_a.norm().item()))

                    threat_change = abs(current_accum - prev_harm_accum)
                    threat_change_list.append(threat_change)

            prev_harm_accum = current_accum
            _flat_obs, _harm_signal, done, _info, obs_dict = env.step(action_idx)
            if done:
                break

        # End-of-episode evaluation metrics
        if ep >= P0_EPISODES and len(z_harm_s_list) > 10 and len(z_harm_a_list) > 10:
            zs = torch.stack(z_harm_s_list)
            za = torch.stack(z_harm_a_list)

            # Stream correlation (cosine sim of means)
            zs_mean = zs.mean(dim=0)
            za_mean = za.mean(dim=0)
            min_d = min(zs_mean.shape[0], za_mean.shape[0])
            cos_sim = F.cosine_similarity(
                zs_mean[:min_d].unsqueeze(0),
                za_mean[:min_d].unsqueeze(0),
            ).item()
            stream_corrs.append(cos_sim)

            # Urgency vs threat change correlation
            if len(urgency_list) > 5 and len(threat_change_list) > 5:
                u_t = torch.tensor(urgency_list)
                tc_t = torch.tensor(threat_change_list)
                if u_t.std() > 1e-8 and tc_t.std() > 1e-8:
                    corr = torch.corrcoef(torch.stack([u_t, tc_t]))[0, 1].item()
                    if not math.isnan(corr):
                        urgency_vs_threat_change.append(corr)

    # Aggregate metrics
    mean_stream_corr = sum(stream_corrs) / max(len(stream_corrs), 1)
    mean_urgency_corr = (
        sum(urgency_vs_threat_change) / max(len(urgency_vs_threat_change), 1)
        if urgency_vs_threat_change else 0.0
    )

    return {
        "seed": seed,
        "condition": condition,
        "mean_stream_corr": round(mean_stream_corr, 4),
        "mean_urgency_threat_change_corr": round(mean_urgency_corr, 4),
        "n_eval_episodes": P1_EPISODES,
    }


def main():
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    print(f"[EXQ-260] run_id = {run_id}")

    all_results = []
    for seed in SEEDS:
        for cond in CONDITIONS:
            print(f"  seed={seed} condition={cond} ...", end=" ", flush=True)
            r = run_condition(seed, cond)
            print(f"stream_corr={r['mean_stream_corr']:.3f} "
                  f"urgency_corr={r['mean_urgency_threat_change_corr']:.3f}")
            all_results.append(r)

    # Aggregate by condition
    condition_metrics = {}
    for cond in CONDITIONS:
        cond_runs = [r for r in all_results if r["condition"] == cond]
        condition_metrics[cond] = {
            "mean_stream_corr": round(
                sum(r["mean_stream_corr"] for r in cond_runs) / len(cond_runs), 4
            ),
            "mean_urgency_corr": round(
                sum(r["mean_urgency_threat_change_corr"] for r in cond_runs) / len(cond_runs), 4
            ),
        }

    raw = condition_metrics["RAW_ACCUM"]
    surprise = condition_metrics["SURPRISE_PE"]

    c1_pass_count = sum(
        1 for s in SEEDS
        if next(r for r in all_results if r["seed"] == s and r["condition"] == "SURPRISE_PE")[
            "mean_urgency_threat_change_corr"
        ]
        > next(r for r in all_results if r["seed"] == s and r["condition"] == "RAW_ACCUM")[
            "mean_urgency_threat_change_corr"
        ]
    )
    c2_pass_count = sum(
        1 for s in SEEDS
        if abs(
            next(r for r in all_results if r["seed"] == s and r["condition"] == "SURPRISE_PE")[
                "mean_stream_corr"
            ]
        )
        < abs(
            next(r for r in all_results if r["seed"] == s and r["condition"] == "RAW_ACCUM")[
                "mean_stream_corr"
            ]
        )
    )

    c1_pass = c1_pass_count >= SEED_PASS_QUOTA
    c2_pass = c2_pass_count >= SEED_PASS_QUOTA
    overall = "PASS" if (c1_pass and c2_pass) else "FAIL"
    ed = "supports" if overall == "PASS" else "weakens"

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "status": overall,
        "evidence_direction": ed,
        "evidence_direction_per_claim": {
            "SD-020": ed,
            "SD-011": "supports" if c2_pass else "mixed",
        },
        "criteria": {
            "C1_urgency_corr_surprise_gt_raw": {
                "pass": c1_pass,
                "seeds_passing": c1_pass_count,
                "surprise_mean": surprise["mean_urgency_corr"],
                "raw_mean": raw["mean_urgency_corr"],
            },
            "C2_stream_corr_surprise_lt_raw": {
                "pass": c2_pass,
                "seeds_passing": c2_pass_count,
                "surprise_mean": surprise["mean_stream_corr"],
                "raw_mean": raw["mean_stream_corr"],
            },
        },
        "condition_metrics": condition_metrics,
        "per_seed_results": all_results,
        "config_summary": {
            "grid_size": GRID_SIZE,
            "num_hazards": NUM_HAZARDS,
            "num_resources": NUM_RESOURCES,
            "harm_history_len": HARM_HISTORY_LEN,
            "p0_episodes": P0_EPISODES,
            "p1_episodes": P1_EPISODES,
            "steps_per_ep": STEPS_PER_EP,
        },
    }

    out_dir = (
        Path(__file__).resolve().parents[1].parent
        / "REE_assembly" / "evidence" / "experiments"
    )
    out_file = out_dir / f"{run_id}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[EXQ-260] {overall} -- wrote {out_file.name}")
    print(
        f"  C1 urgency_corr: surprise={surprise['mean_urgency_corr']:.3f}"
        f" vs raw={raw['mean_urgency_corr']:.3f}"
        f" -> {'PASS' if c1_pass else 'FAIL'}"
    )
    print(
        f"  C2 stream_corr: surprise={surprise['mean_stream_corr']:.3f}"
        f" vs raw={raw['mean_stream_corr']:.3f}"
        f" -> {'PASS' if c2_pass else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
