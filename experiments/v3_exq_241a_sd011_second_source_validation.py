#!/opt/local/bin/python3
"""
V3-EXQ-241a -- SD-011 Second Source Validation

Claims: SD-011
EXPERIMENT_PURPOSE = "diagnostic"

Validates the SD-011 second source implementation: harm_history (rolling FIFO
of past harm_exposure scalars) as additional input to AffectiveHarmEncoder,
plus auxiliary harm_accum_head.

EXQ-241 showed D3 reversal: R2(z_harm_a) > R2(z_harm_s) in all seeds,
meaning z_harm_a was monotonically redundant with z_harm_s (both received
the same spatial proximity signal). This experiment tests whether the
harm_history input resolves the reversal.

Design
------
2-condition ablation, 3 seeds:
  WITH_HISTORY:    harm_history_len=10, z_harm_a_aux_loss_weight=0.1
  WITHOUT_HISTORY: harm_history_len=0 (baseline, replicates EXQ-241 D3 reversal)

Each condition:
  Phase 0 (P0): 100 episodes encoder warmup (agent + harm_accum_loss)
  Phase 1 (P1): 50 episodes evaluation (forward model probe R2)

Forward model probes (same as EXQ-241):
  MLP(z_world, action) -> z_harm_s_next  (sensory stream)
  MLP(z_world, action) -> z_harm_a_next  (affective stream)
  Trained during P0, evaluated during P1.

Success criteria (WITH_HISTORY condition, >= 2/3 seeds):
  P1: stream_corr < 0.85       (streams remain decorrelated after history input)
  P2: r2_s > r2_a              (D3 reversal resolved -- sensory more predictable)
  P3: harm_accum_loss < 0.05   (aux head learns to predict accumulated harm)

PASS: P1 AND P2 AND P3 (all >= 2/3 seeds)
FAIL: any criterion not met

Seeds: [42, 7, 13]
Env: CausalGridWorldV2 size=10, 3 hazards, 5 resources, hazard_harm=0.5
Est: ~60 min (DLAPTOP-4.local) -- 3 seeds x 2 conditions x 150 eps x 150 steps
"""

import sys
import json
import math
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
EXPERIMENT_TYPE    = "v3_exq_241a_sd011_second_source_validation"
CLAIM_IDS          = ["SD-011"]
EXPERIMENT_PURPOSE = "diagnostic"

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
THRESH_STREAM_CORR    = 0.85   # P1: cosine sim below this = distinct
THRESH_HARM_ACCUM_LOSS = 0.05  # P3: aux head loss below this = learned
SEED_PASS_QUOTA       = 2      # >= 2/3 seeds

# ---------------------------------------------------------------------------
# Architecture constants
# ---------------------------------------------------------------------------
BODY_OBS_DIM   = 12
WORLD_OBS_DIM  = 250
ACTION_DIM     = 5
WORLD_DIM      = 32
SELF_DIM       = 32
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
# Forward model probe
# ---------------------------------------------------------------------------
class HarmStreamForwardProbe(nn.Module):
    """MLP(z_world_t, action_t) -> z_harm_next."""

    def __init__(self, z_world_dim: int, action_dim: int, z_harm_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_world_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, z_harm_dim),
        )

    def forward(self, z_world: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_world, action], dim=-1))


def _r2_score(preds: torch.Tensor, targets: torch.Tensor) -> float:
    p = preds.detach().float().reshape(-1)
    t = targets.detach().float().reshape(-1)
    ss_res = ((p - t) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum()
    if ss_tot < 1e-8:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _cosine_sim_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean cosine similarity between two [N, D] tensors."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return 0.0
    # Project to same dim (z_harm_a may be smaller)
    min_d = min(a.shape[-1], b.shape[-1])
    a2 = a[..., :min_d]
    b2 = b[..., :min_d]
    cos = F.cosine_similarity(a2, b2, dim=-1)
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
        action_dim=ACTION_DIM - 1,  # env has 5 actions (0-4) but action_dim=4 for agent
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

    # Forward model probes
    probe_s = HarmStreamForwardProbe(WORLD_DIM, ACTION_DIM, 32)  # z_harm_s dim=32
    probe_a = HarmStreamForwardProbe(WORLD_DIM, ACTION_DIM, Z_HARM_A_DIM)
    probe_s_opt = optim.Adam(probe_s.parameters(), lr=3e-4)
    probe_a_opt = optim.Adam(probe_a.parameters(), lr=3e-4)

    # Agent optimizer (for encoding + aux loss)
    agent_opt = optim.Adam(agent.latent_stack.parameters(), lr=1e-4)

    # Collection buffers for eval
    z_harm_s_eval: List[torch.Tensor] = []
    z_harm_a_eval: List[torch.Tensor] = []
    probe_s_preds: List[torch.Tensor] = []
    probe_s_tgts: List[torch.Tensor] = []
    probe_a_preds: List[torch.Tensor] = []
    probe_a_tgts: List[torch.Tensor] = []
    harm_accum_losses: List[float] = []

    total_eps = P0_TRAIN_EPS + P1_EVAL_EPS
    for ep in range(total_eps):
        is_eval = ep >= P0_TRAIN_EPS
        obs, obs_dict = env.reset()
        agent.reset()

        prev_z_world = None
        prev_z_harm_s = None
        prev_z_harm_a = None
        prev_action_oh = None

        for step in range(STEPS_PER_EP):
            # Sense
            latent = agent.sense(
                obs_dict["body_state"].unsqueeze(0),
                obs_dict["world_state"].unsqueeze(0),
                obs_harm=obs_dict.get("harm_obs"),
                obs_harm_a=obs_dict.get("harm_obs_a"),
                obs_harm_history=obs_dict.get("harm_history"),
            )

            z_world = latent.z_world.detach()
            z_harm_s = latent.z_harm.detach() if latent.z_harm is not None else None
            z_harm_a_now = latent.z_harm_a.detach() if latent.z_harm_a is not None else None

            # Train probes on (prev_z_world, prev_action) -> current z_harm
            if prev_z_world is not None and prev_action_oh is not None and not is_eval:
                if z_harm_s is not None:
                    pred_s = probe_s(prev_z_world, prev_action_oh)
                    loss_s = F.mse_loss(pred_s, z_harm_s)
                    probe_s_opt.zero_grad()
                    loss_s.backward()
                    probe_s_opt.step()

                if z_harm_a_now is not None:
                    pred_a = probe_a(prev_z_world, prev_action_oh)
                    loss_a = F.mse_loss(pred_a, z_harm_a_now)
                    probe_a_opt.zero_grad()
                    loss_a.backward()
                    probe_a_opt.step()

            # Train aux head (harm_accum_loss) during P0
            if not is_eval and hhl > 0:
                latent_grad = agent.sense(
                    obs_dict["body_state"].unsqueeze(0),
                    obs_dict["world_state"].unsqueeze(0),
                    obs_harm=obs_dict.get("harm_obs"),
                    obs_harm_a=obs_dict.get("harm_obs_a"),
                    obs_harm_history=obs_dict.get("harm_history"),
                )
                accum_target = obs_dict.get("accumulated_harm", 0.0)
                aux_loss = agent.compute_harm_accum_loss(accum_target, latent_grad)
                # Also train resource proximity head
                rpt = float(obs_dict.get("resource_field_view", torch.zeros(1)).max().item())
                rp_loss = agent.compute_resource_proximity_loss(rpt, latent_grad)
                total_loss = aux_loss + rp_loss
                if total_loss.requires_grad:
                    agent_opt.zero_grad()
                    total_loss.backward()
                    agent_opt.step()
                    harm_accum_losses.append(float(aux_loss.item()))

            # Eval collection
            if is_eval and prev_z_world is not None and prev_action_oh is not None:
                if z_harm_s is not None:
                    z_harm_s_eval.append(z_harm_s.squeeze(0))
                    with torch.no_grad():
                        probe_s_preds.append(probe_s(prev_z_world, prev_action_oh).squeeze(0))
                        probe_s_tgts.append(z_harm_s.squeeze(0))
                if z_harm_a_now is not None:
                    z_harm_a_eval.append(z_harm_a_now.squeeze(0))
                    with torch.no_grad():
                        probe_a_preds.append(probe_a(prev_z_world, prev_action_oh).squeeze(0))
                        probe_a_tgts.append(z_harm_a_now.squeeze(0))

            # Action
            action = torch.randint(0, ACTION_DIM, (1,)).item()
            action_oh = F.one_hot(torch.tensor([action]), ACTION_DIM).float()

            obs, reward, done, info, obs_dict = env.step(action)

            prev_z_world = z_world
            prev_z_harm_s = z_harm_s
            prev_z_harm_a = z_harm_a_now
            prev_action_oh = action_oh

            if done:
                break

    # Compute metrics
    r2_s = 0.0
    r2_a = 0.0
    stream_corr = 0.0
    final_accum_loss = 0.0

    if len(probe_s_preds) > 0:
        r2_s = _r2_score(torch.stack(probe_s_preds), torch.stack(probe_s_tgts))
    if len(probe_a_preds) > 0:
        r2_a = _r2_score(torch.stack(probe_a_preds), torch.stack(probe_a_tgts))
    if len(z_harm_s_eval) > 0 and len(z_harm_a_eval) > 0:
        stream_corr = _cosine_sim_mean(
            torch.stack(z_harm_s_eval), torch.stack(z_harm_a_eval),
        )
    if len(harm_accum_losses) > 10:
        final_accum_loss = sum(harm_accum_losses[-50:]) / len(harm_accum_losses[-50:])

    d3_reversed = r2_a > r2_s
    p1_pass = abs(stream_corr) < THRESH_STREAM_CORR
    p2_pass = not d3_reversed  # r2_s > r2_a
    p3_pass = final_accum_loss < THRESH_HARM_ACCUM_LOSS if hhl > 0 else True

    result = {
        "seed": seed,
        "condition": condition_name,
        "harm_history_len": hhl,
        "r2_s": round(r2_s, 4),
        "r2_a": round(r2_a, 4),
        "stream_corr": round(stream_corr, 4),
        "d3_reversed": d3_reversed,
        "final_accum_loss": round(final_accum_loss, 6),
        "p1_pass": p1_pass,
        "p2_pass": p2_pass,
        "p3_pass": p3_pass,
        "all_pass": p1_pass and p2_pass and p3_pass,
    }
    tag = "PASS" if result["all_pass"] else "FAIL"
    print(f"  seed={seed} condition={condition_name}: {tag}")
    print(f"    r2_s={r2_s:.4f}, r2_a={r2_a:.4f}, stream_corr={stream_corr:.4f}")
    print(f"    d3_reversed={d3_reversed}, accum_loss={final_accum_loss:.6f}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"EXQ-241a: SD-011 Second Source Validation")
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

    # Aggregate: check WITH_HISTORY seeds
    with_history = [r for r in all_results if r["condition"] == "WITH_HISTORY"]
    without_history = [r for r in all_results if r["condition"] == "WITHOUT_HISTORY"]

    p1_count = sum(1 for r in with_history if r["p1_pass"])
    p2_count = sum(1 for r in with_history if r["p2_pass"])
    p3_count = sum(1 for r in with_history if r["p3_pass"])
    all_count = sum(1 for r in with_history if r["all_pass"])

    overall_pass = (
        p1_count >= SEED_PASS_QUOTA
        and p2_count >= SEED_PASS_QUOTA
        and p3_count >= SEED_PASS_QUOTA
    )

    # WITHOUT_HISTORY should show D3 reversal (replicating EXQ-241)
    d3_reversed_count = sum(1 for r in without_history if r["d3_reversed"])

    print("=" * 60)
    print(f"WITH_HISTORY:    P1={p1_count}/3  P2={p2_count}/3  P3={p3_count}/3  ALL={all_count}/3")
    print(f"WITHOUT_HISTORY: D3_reversed={d3_reversed_count}/3 (expected: >= 2/3)")
    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 60)

    # --- Output flat JSON ---
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"v3_exq_241a_sd011_second_source_validation_{ts}_v3"
    evidence_dir = (
        Path(__file__).resolve().parents[1].parent
        / "REE_assembly" / "evidence" / "experiments"
        / f"v3_exq_241a_sd011_second_source_validation"
        / "runs" / run_id
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        # WITH_HISTORY aggregate
        "p1_stream_corr_passes": p1_count,
        "p2_d3_resolved_passes": p2_count,
        "p3_accum_loss_passes": p3_count,
        "with_history_all_pass": all_count,
        # WITHOUT_HISTORY aggregate (replication check)
        "without_history_d3_reversed": d3_reversed_count,
        # Per-seed details
        "per_seed": all_results,
        # Mean metrics (WITH_HISTORY)
        "mean_r2_s_with": round(sum(r["r2_s"] for r in with_history) / len(with_history), 4),
        "mean_r2_a_with": round(sum(r["r2_a"] for r in with_history) / len(with_history), 4),
        "mean_stream_corr_with": round(sum(r["stream_corr"] for r in with_history) / len(with_history), 4),
        "mean_accum_loss_with": round(sum(r["final_accum_loss"] for r in with_history) / len(with_history), 6),
        # Mean metrics (WITHOUT_HISTORY)
        "mean_r2_s_without": round(sum(r["r2_s"] for r in without_history) / len(without_history), 4),
        "mean_r2_a_without": round(sum(r["r2_a"] for r in without_history) / len(without_history), 4),
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
        "supersedes": "v3_exq_241_sd011_dual_nociceptive_stream_poc",
    }

    with open(evidence_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(evidence_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults written to: {evidence_dir}")
    print(f"run_id: {run_id}")

    # Flat output for runner
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
