#!/opt/local/bin/python3
"""
V3-EXQ-241b -- SD-011 Second Source: Information Gain Test

Claims: SD-011
EXPERIMENT_PURPOSE = "diagnostic"
Supersedes: V3-EXQ-241a

EXQ-241a FAIL diagnosis: D3 criterion (r2_a < r2_s from z_world forward probe)
was flawed. z_harm_a's EMA smoothing makes it inherently more predictable from
any external signal (lower variance = easier to predict), regardless of what
information it carries. The probe measured smoothness, not redundancy.

Corrected design: measure INFORMATION GAIN -- does z_harm_a carry information
about temporal harm history that z_harm_s does NOT carry?

Design
------
2-condition ablation, 3 seeds:
  WITH_HISTORY:    harm_history_len=10, z_harm_a_aux_loss_weight=0.1
  WITHOUT_HISTORY: harm_history_len=0 (baseline)

Each condition:
  Phase 0 (P0): 100 episodes encoder warmup (agent + aux loss)
  Phase 1 (P1): 50 episodes evaluation

Information probes (trained during P0, evaluated during P1):
  Probe_A: MLP(z_harm_a) -> accumulated_harm   (can affective stream predict history?)
  Probe_S: MLP(z_harm_s) -> accumulated_harm   (can sensory stream predict history?)
  Probe_H: MLP(z_harm_s) -> z_harm_a           (can sensory predict affective? redundancy)

Success criteria (WITH_HISTORY, >= 2/3 seeds):
  C1: stream_corr < 0.85               (streams decorrelated -- carried from 241a)
  C2: r2_a_accum > r2_s_accum + 0.05   (z_harm_a predicts accumulated_harm BETTER
      than z_harm_s -- information gain from history input)
  C3: r2_s_to_a < 0.80                 (z_harm_s cannot fully reconstruct z_harm_a --
      streams not redundant)
  C4: harm_accum_loss < 0.05           (aux head learns -- carried from 241a)

Contrast: WITHOUT_HISTORY should show r2_a_accum ~ r2_s_accum (no information gain)
because both streams receive the same spatial proximity signal.

PASS: C1 AND C2 AND C3 AND C4 (all >= 2/3 seeds)
FAIL: any criterion not met

Seeds: [42, 7, 13]
Env: CausalGridWorldV2 size=10, 3 hazards, 5 resources, hazard_harm=0.5
Est: ~60 min (DLAPTOP-4.local)
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional

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
EXPERIMENT_TYPE    = "v3_exq_241b_sd011_second_source_info_gain"
CLAIM_IDS          = ["SD-011"]
EXPERIMENT_PURPOSE = "diagnostic"

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
THRESH_STREAM_CORR   = 0.85   # C1
THRESH_INFO_GAIN     = 0.05   # C2: r2_a_accum must exceed r2_s_accum by this
THRESH_REDUNDANCY    = 0.80   # C3: r2_s_to_a must be below this
THRESH_ACCUM_LOSS    = 0.05   # C4
SEED_PASS_QUOTA      = 2      # >= 2/3 seeds

# ---------------------------------------------------------------------------
# Architecture constants
# ---------------------------------------------------------------------------
BODY_OBS_DIM   = 12
WORLD_OBS_DIM  = 250
ACTION_DIM     = 5
WORLD_DIM      = 32
Z_HARM_S_DIM   = 32
Z_HARM_A_DIM   = 16

# ---------------------------------------------------------------------------
# Training schedule
# ---------------------------------------------------------------------------
SEEDS         = [42, 7, 13]
P0_TRAIN_EPS  = 100
P1_EVAL_EPS   = 50
STEPS_PER_EP  = 150

# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------
CONDITIONS = {
    "WITH_HISTORY": {"harm_history_len": 10, "z_harm_a_aux_loss_weight": 0.1},
    "WITHOUT_HISTORY": {"harm_history_len": 0, "z_harm_a_aux_loss_weight": 0.0},
}


# ---------------------------------------------------------------------------
# Information probes
# ---------------------------------------------------------------------------
class AccumProbe(nn.Module):
    """MLP(z_harm) -> accumulated_harm scalar."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class RedundancyProbe(nn.Module):
    """MLP(z_harm_s) -> z_harm_a (can sensory reconstruct affective?)."""
    def __init__(self, s_dim: int, a_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, 64),
            nn.ReLU(),
            nn.Linear(64, a_dim),
        )

    def forward(self, z_s: torch.Tensor) -> torch.Tensor:
        return self.net(z_s)


def _r2_score(preds: torch.Tensor, targets: torch.Tensor) -> float:
    p = preds.detach().float().reshape(-1)
    t = targets.detach().float().reshape(-1)
    ss_res = ((p - t) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum()
    if ss_tot < 1e-8:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _cosine_sim_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return 0.0
    min_d = min(a.shape[-1], b.shape[-1])
    cos = F.cosine_similarity(a[..., :min_d], b[..., :min_d], dim=-1)
    return float(cos.mean().item())


# ---------------------------------------------------------------------------
# Run one seed x condition
# ---------------------------------------------------------------------------
def run_one(seed: int, condition_name: str, condition_cfg: dict) -> dict:
    hhl = condition_cfg["harm_history_len"]
    aux_w = condition_cfg["z_harm_a_aux_loss_weight"]

    torch.manual_seed(seed)

    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM - 1,
        alpha_world=0.9,
        use_harm_stream=True,
        use_affective_harm_stream=True,
        harm_history_len=hhl,
        z_harm_a_aux_loss_weight=aux_w,
        use_resource_proximity_head=True,
    )
    agent = REEAgent(cfg)
    env = CausalGridWorldV2(
        use_proxy_fields=True,
        seed=seed,
        hazard_harm=0.5,
        num_hazards=3,
        num_resources=5,
        harm_history_len=hhl,
    )

    # Information probes
    probe_a_accum = AccumProbe(Z_HARM_A_DIM)       # z_harm_a -> accumulated_harm
    probe_s_accum = AccumProbe(Z_HARM_S_DIM)       # z_harm_s -> accumulated_harm
    probe_s_to_a  = RedundancyProbe(Z_HARM_S_DIM, Z_HARM_A_DIM)  # z_harm_s -> z_harm_a

    probe_a_opt = optim.Adam(probe_a_accum.parameters(), lr=3e-4)
    probe_s_opt = optim.Adam(probe_s_accum.parameters(), lr=3e-4)
    probe_r_opt = optim.Adam(probe_s_to_a.parameters(), lr=3e-4)

    # Agent optimizer (for encoding + aux loss)
    agent_opt = optim.Adam(agent.latent_stack.parameters(), lr=1e-4)

    # Eval buffers
    eval_z_harm_s: List[torch.Tensor] = []
    eval_z_harm_a: List[torch.Tensor] = []
    eval_accum_targets: List[float] = []
    eval_probe_a_preds: List[torch.Tensor] = []
    eval_probe_s_preds: List[torch.Tensor] = []
    eval_probe_r_preds: List[torch.Tensor] = []
    harm_accum_losses: List[float] = []

    total_eps = P0_TRAIN_EPS + P1_EVAL_EPS
    for ep in range(total_eps):
        is_eval = ep >= P0_TRAIN_EPS
        obs, obs_dict = env.reset()
        agent.reset()

        for step in range(STEPS_PER_EP):
            # Sense (for training: keep grads for aux loss)
            latent = agent.sense(
                obs_dict["body_state"].unsqueeze(0),
                obs_dict["world_state"].unsqueeze(0),
                obs_harm=obs_dict.get("harm_obs"),
                obs_harm_a=obs_dict.get("harm_obs_a"),
                obs_harm_history=obs_dict.get("harm_history"),
            )

            z_harm_s = latent.z_harm.detach() if latent.z_harm is not None else None
            z_harm_a = latent.z_harm_a.detach() if latent.z_harm_a is not None else None

            # Get accumulated_harm target
            accum_target = obs_dict.get("accumulated_harm", 0.0)
            if isinstance(accum_target, torch.Tensor):
                accum_target = float(accum_target.item())
            accum_t = torch.tensor([[accum_target]], dtype=torch.float32)

            # Train probes during P0
            if not is_eval and z_harm_s is not None and z_harm_a is not None:
                # Probe A: z_harm_a -> accumulated_harm
                pred_a = probe_a_accum(z_harm_a)
                loss_a = F.mse_loss(pred_a, accum_t)
                probe_a_opt.zero_grad()
                loss_a.backward()
                probe_a_opt.step()

                # Probe S: z_harm_s -> accumulated_harm
                pred_s = probe_s_accum(z_harm_s)
                loss_s = F.mse_loss(pred_s, accum_t)
                probe_s_opt.zero_grad()
                loss_s.backward()
                probe_s_opt.step()

                # Probe R: z_harm_s -> z_harm_a (redundancy)
                pred_r = probe_s_to_a(z_harm_s)
                loss_r = F.mse_loss(pred_r, z_harm_a)
                probe_r_opt.zero_grad()
                loss_r.backward()
                probe_r_opt.step()

            # Train aux head during P0
            if not is_eval and hhl > 0:
                latent_grad = agent.sense(
                    obs_dict["body_state"].unsqueeze(0),
                    obs_dict["world_state"].unsqueeze(0),
                    obs_harm=obs_dict.get("harm_obs"),
                    obs_harm_a=obs_dict.get("harm_obs_a"),
                    obs_harm_history=obs_dict.get("harm_history"),
                )
                aux_loss = agent.compute_harm_accum_loss(accum_target, latent_grad)
                rpt = float(obs_dict.get("resource_field_view", torch.zeros(1)).max().item())
                rp_loss = agent.compute_resource_proximity_loss(rpt, latent_grad)
                total_loss = aux_loss + rp_loss
                if total_loss.requires_grad:
                    agent_opt.zero_grad()
                    total_loss.backward()
                    agent_opt.step()
                    harm_accum_losses.append(float(aux_loss.item()))

            # Eval collection
            if is_eval and z_harm_s is not None and z_harm_a is not None:
                eval_z_harm_s.append(z_harm_s.squeeze(0))
                eval_z_harm_a.append(z_harm_a.squeeze(0))
                eval_accum_targets.append(accum_target)
                with torch.no_grad():
                    eval_probe_a_preds.append(probe_a_accum(z_harm_a).squeeze(0))
                    eval_probe_s_preds.append(probe_s_accum(z_harm_s).squeeze(0))
                    eval_probe_r_preds.append(probe_s_to_a(z_harm_s).squeeze(0))

            # Action
            action = torch.randint(0, ACTION_DIM, (1,)).item()
            obs, reward, done, info, obs_dict = env.step(action)

            if done:
                break

    # Compute metrics
    stream_corr = 0.0
    r2_a_accum = 0.0
    r2_s_accum = 0.0
    r2_s_to_a = 0.0
    final_accum_loss = 0.0

    if len(eval_z_harm_s) > 0 and len(eval_z_harm_a) > 0:
        stream_corr = _cosine_sim_mean(
            torch.stack(eval_z_harm_s), torch.stack(eval_z_harm_a),
        )

    if len(eval_probe_a_preds) > 0:
        targets_t = torch.tensor(eval_accum_targets).unsqueeze(-1)
        r2_a_accum = _r2_score(torch.stack(eval_probe_a_preds), targets_t)
        r2_s_accum = _r2_score(torch.stack(eval_probe_s_preds), targets_t)

    if len(eval_probe_r_preds) > 0 and len(eval_z_harm_a) > 0:
        r2_s_to_a = _r2_score(
            torch.stack(eval_probe_r_preds), torch.stack(eval_z_harm_a),
        )

    if len(harm_accum_losses) > 10:
        final_accum_loss = sum(harm_accum_losses[-50:]) / len(harm_accum_losses[-50:])

    info_gain = r2_a_accum - r2_s_accum

    c1_pass = abs(stream_corr) < THRESH_STREAM_CORR
    c2_pass = info_gain > THRESH_INFO_GAIN
    c3_pass = r2_s_to_a < THRESH_REDUNDANCY
    c4_pass = final_accum_loss < THRESH_ACCUM_LOSS if hhl > 0 else True

    result = {
        "seed": seed,
        "condition": condition_name,
        "harm_history_len": hhl,
        "stream_corr": round(stream_corr, 4),
        "r2_a_accum": round(r2_a_accum, 4),
        "r2_s_accum": round(r2_s_accum, 4),
        "info_gain": round(info_gain, 4),
        "r2_s_to_a": round(r2_s_to_a, 4),
        "final_accum_loss": round(final_accum_loss, 6),
        "c1_pass": c1_pass,
        "c2_pass": c2_pass,
        "c3_pass": c3_pass,
        "c4_pass": c4_pass,
        "all_pass": c1_pass and c2_pass and c3_pass and c4_pass,
    }
    tag = "PASS" if result["all_pass"] else "FAIL"
    print(f"  seed={seed} {condition_name}: {tag}")
    print(f"    stream_corr={stream_corr:.4f}")
    print(f"    r2_a_accum={r2_a_accum:.4f}, r2_s_accum={r2_s_accum:.4f}, info_gain={info_gain:.4f}")
    print(f"    r2_s_to_a={r2_s_to_a:.4f}, accum_loss={final_accum_loss:.6f}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"EXQ-241b: SD-011 Second Source -- Information Gain Test")
    print(f"Conditions: {list(CONDITIONS.keys())}")
    print(f"Seeds: {SEEDS}")
    print()

    all_results: List[dict] = []
    for cond_name, cond_cfg in CONDITIONS.items():
        print(f"--- Condition: {cond_name} ---")
        for seed in SEEDS:
            r = run_one(seed, cond_name, cond_cfg)
            all_results.append(r)
        print()

    # Aggregate
    with_history = [r for r in all_results if r["condition"] == "WITH_HISTORY"]
    without_history = [r for r in all_results if r["condition"] == "WITHOUT_HISTORY"]

    c1_count = sum(1 for r in with_history if r["c1_pass"])
    c2_count = sum(1 for r in with_history if r["c2_pass"])
    c3_count = sum(1 for r in with_history if r["c3_pass"])
    c4_count = sum(1 for r in with_history if r["c4_pass"])
    all_count = sum(1 for r in with_history if r["all_pass"])

    overall_pass = (
        c1_count >= SEED_PASS_QUOTA
        and c2_count >= SEED_PASS_QUOTA
        and c3_count >= SEED_PASS_QUOTA
        and c4_count >= SEED_PASS_QUOTA
    )

    # WITHOUT_HISTORY: info_gain should be near zero (no distinct temporal info)
    no_hist_gains = [r["info_gain"] for r in without_history]

    print("=" * 60)
    print(f"WITH_HISTORY:    C1={c1_count}/3  C2={c2_count}/3  C3={c3_count}/3  C4={c4_count}/3  ALL={all_count}/3")
    print(f"WITHOUT_HISTORY: info_gains={no_hist_gains} (expected: near 0)")
    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 60)

    # --- Output ---
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"v3_exq_241b_sd011_second_source_info_gain_{ts}_v3"
    evidence_dir = (
        Path(__file__).resolve().parents[1].parent
        / "REE_assembly" / "evidence" / "experiments"
        / f"v3_exq_241b_sd011_second_source_info_gain"
        / "runs" / run_id
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "c1_stream_corr_passes": c1_count,
        "c2_info_gain_passes": c2_count,
        "c3_redundancy_passes": c3_count,
        "c4_accum_loss_passes": c4_count,
        "with_history_all_pass": all_count,
        "without_history_info_gains": no_hist_gains,
        "per_seed": all_results,
        "mean_info_gain_with": round(sum(r["info_gain"] for r in with_history) / len(with_history), 4),
        "mean_info_gain_without": round(sum(r["info_gain"] for r in without_history) / len(without_history), 4),
        "mean_r2_s_to_a_with": round(sum(r["r2_s_to_a"] for r in with_history) / len(with_history), 4),
        "mean_stream_corr_with": round(sum(r["stream_corr"] for r in with_history) / len(with_history), 4),
    }

    manifest = {
        "experiment_type": EXPERIMENT_TYPE,
        "run_id": run_id,
        "claim_ids": CLAIM_IDS,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "status": "PASS" if overall_pass else "FAIL",
        "evidence_direction": "supports" if overall_pass else "does_not_support",
        "timestamp_utc": ts,
        "seeds": SEEDS,
        "conditions": list(CONDITIONS.keys()),
        "metrics": metrics,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes": "v3_exq_241a_sd011_second_source_validation",
    }

    with open(evidence_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(evidence_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults written to: {evidence_dir}")
    print(f"run_id: {run_id}")

    output = {
        "output_files": [str(evidence_dir / "manifest.json")],
        "run_id": run_id,
        "claim_ids": CLAIM_IDS,
        "status": "PASS" if overall_pass else "FAIL",
    }
    with open(evidence_dir.parent.parent / f"{EXPERIMENT_TYPE}_output.json", "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
