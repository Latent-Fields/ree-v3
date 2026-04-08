#!/opt/local/bin/python3
"""
V3-EXQ-262 -- MECH-220 Cross-Stream Harm Hub Coordination

Claims: MECH-220, SD-011
EXPERIMENT_PURPOSE = "evidence"

Tests whether lightweight cross-stream coordination between z_harm_s and z_harm_a
improves both forward model R^2 and urgency calibration. MECH-220 claims the two
harm streams should share context via a hub mechanism (not fuse).

Chen (2023) central claim: the cingulate-insula hub coordinates the medial
(affective) and lateral (sensory-discriminative) pain pathways. Our current
implementation has zero cross-talk between z_harm_s and z_harm_a encoders.

Design
------
3-condition comparison, 3 seeds:
  NO_HUB:     baseline (current impl, independent streams)
  S_TO_A_HUB: z_harm_a receives z_harm_s norm as additional context input
               (sensory PE magnitude informs affective load update)
  BIDI_HUB:   bidirectional -- z_harm_a gets z_harm_s norm AND
               z_harm_s gets z_harm_a norm as gating signal
               (affective urgency modulates sensory gain -- hypervigilance)

Implementation: HubGate module -- simple learned scalar gating.
  S_TO_A: AffectiveHarmEncoder input expanded by 1 (z_harm_s.norm())
  BIDI: additionally, HarmEncoder output multiplied by (1 + hub_gain * z_harm_a.norm())

Each condition:
  Phase 0 (P0): 120 episodes training
  Phase 1 (P1): 60 episodes evaluation with forward model R^2 probe

Success criteria (>= 2/3 seeds):
  C1: fwd_r2(S_TO_A_HUB) > fwd_r2(NO_HUB)
      (hub improves sensory forward model -- affective context aids prediction)
  C2: stream_corr(S_TO_A_HUB) < 0.85
      (hub does NOT collapse streams -- they remain distinct)
  C3: urgency_stability(S_TO_A_HUB) > urgency_stability(NO_HUB)
      (hub produces more stable urgency signal -- less noise)

PASS: C1 AND C2 (>= 2/3 seeds). C3 is informational.
FAIL: either C1 or C2 not met.

Seeds: [42, 7, 13]
Env: CausalGridWorldV2 size=10, 3 hazards, 5 resources
Est: ~90 min (DLAPTOP-4.local) -- 3 seeds x 3 conditions x 180 eps x 150 steps
"""

import sys
import json
import random
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List

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
EXPERIMENT_TYPE    = "v3_exq_262_mech220_harm_hub"
CLAIM_IDS          = ["MECH-220", "SD-011"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
STREAM_CORR_CEILING = 0.85
SEED_PASS_QUOTA     = 2

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
CONDITIONS     = ["NO_HUB", "S_TO_A_HUB", "BIDI_HUB"]
P0_EPISODES    = 120
P1_EPISODES    = 60
STEPS_PER_EP   = 150


class ForwardProbe(nn.Module):
    """MLP probe: (z_harm_s, action) -> z_harm_s_next."""
    def __init__(self, z_dim: int = 32, action_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, z_dim),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, a], dim=-1))


class HubGate(nn.Module):
    """Lightweight scalar gating for cross-stream communication."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.gate = nn.Linear(input_dim + 1, input_dim)

    def forward(self, z: torch.Tensor, context_norm: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, context_norm], dim=-1)
        return self.gate(x)


def run_condition(seed: int, condition: str) -> Dict:
    """Run one seed x condition pair."""
    torch.manual_seed(seed)
    random.seed(seed)

    config = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        harm_obs_dim=HARM_OBS_DIM,
        world_dim=WORLD_DIM,
        action_dim=ACTION_DIM - 1,
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

    # Hub modules
    s_to_a_hub = HubGate(Z_HARM_A_DIM) if condition in ["S_TO_A_HUB", "BIDI_HUB"] else None
    bidi_gain = nn.Parameter(torch.tensor(0.1)) if condition == "BIDI_HUB" else None

    # Forward model probe
    fwd_probe = ForwardProbe(Z_HARM_DIM, ACTION_DIM)
    fwd_opt = optim.Adam(fwd_probe.parameters(), lr=1e-3)

    all_params = list(agent.latent_stack.parameters())
    if s_to_a_hub is not None:
        all_params += list(s_to_a_hub.parameters())
    if bidi_gain is not None:
        all_params += [bidi_gain]
    agent_opt = optim.Adam(all_params, lr=1e-4)

    # Evaluation storage
    fwd_r2_scores: List[float] = []
    stream_corrs: List[float] = []
    urgency_stds: List[float] = []

    total_episodes = P0_EPISODES + P1_EPISODES
    prev_z_harm_s = None
    prev_action_oh = None

    for ep in range(total_episodes):
        _flat_obs, obs_dict = env.reset()
        agent.reset()
        prev_z_harm_s = None
        prev_action_oh = None

        ep_z_harm_s: List[torch.Tensor] = []
        ep_z_harm_a: List[torch.Tensor] = []
        ep_urgency: List[float] = []
        ep_fwd_preds: List[torch.Tensor] = []
        ep_fwd_targets: List[torch.Tensor] = []

        for step in range(STEPS_PER_EP):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm = obs_dict.get("harm_obs")
            obs_harm_a = obs_dict.get("harm_obs_a")
            obs_harm_hist = obs_dict.get("harm_history")

            latent = agent.sense(
                obs_body, obs_world,
                obs_harm=obs_harm,
                obs_harm_a=obs_harm_a,
                obs_harm_history=obs_harm_hist,
            )

            z_s = agent._current_latent.z_harm.detach() if (
                agent._current_latent is not None
                and agent._current_latent.z_harm is not None
            ) else None
            z_a = agent._current_latent.z_harm_a.detach() if (
                agent._current_latent is not None
                and agent._current_latent.z_harm_a is not None
            ) else None

            # Apply hub gating
            if z_s is not None and z_a is not None:
                if s_to_a_hub is not None:
                    # S -> A: affective stream gets sensory norm as context
                    s_norm = z_s.norm(dim=-1, keepdim=True)
                    z_a_hubbed = s_to_a_hub(z_a, s_norm)
                    if agent._current_latent is not None:
                        agent._current_latent.z_harm_a = z_a_hubbed

                if bidi_gain is not None:
                    # A -> S: sensory stream modulated by affective urgency
                    a_norm = z_a.norm(dim=-1, keepdim=True)
                    z_s_modulated = z_s * (1.0 + bidi_gain * a_norm)
                    if agent._current_latent is not None:
                        agent._current_latent.z_harm = z_s_modulated
                    z_s = z_s_modulated.detach()

            # Random action
            action_idx = random.randint(0, ACTION_DIM - 1)
            action_onehot = F.one_hot(
                torch.tensor([action_idx]), ACTION_DIM
            ).float()

            # Forward probe training (P0)
            if ep < P0_EPISODES and prev_z_harm_s is not None and prev_action_oh is not None and z_s is not None:
                fwd_opt.zero_grad()
                pred = fwd_probe(prev_z_harm_s, prev_action_oh)
                target = z_s.detach()
                fwd_loss = F.mse_loss(pred, target)
                fwd_loss.backward()
                fwd_opt.step()

            # Train encoding + hub
            if ep < P0_EPISODES:
                if latent is not None and latent.harm_accum_pred is not None:
                    accum_target = float(obs_dict.get("accumulated_harm", 0.0)) / max(step + 1, 1)
                    loss = agent.compute_harm_accum_loss(accum_target, latent)
                    if loss.requires_grad:
                        agent_opt.zero_grad()
                        loss.backward()
                        agent_opt.step()

            # Evaluation (P1)
            if ep >= P0_EPISODES and z_s is not None and z_a is not None:
                ep_z_harm_s.append(z_s.squeeze())
                ep_z_harm_a.append(z_a.squeeze() if z_a.dim() > 0 else z_a)
                # Urgency = z_harm_a norm
                ep_urgency.append(float(z_a.norm().item()))

                if prev_z_harm_s is not None and prev_action_oh is not None:
                    with torch.no_grad():
                        pred = fwd_probe(prev_z_harm_s, prev_action_oh)
                        ep_fwd_preds.append(pred.squeeze())
                        ep_fwd_targets.append(z_s.squeeze())

            prev_z_harm_s = z_s
            prev_action_oh = action_onehot

            _flat_obs, _harm_signal, done, _info, obs_dict = env.step(action_idx)
            if done:
                break

        # End-of-episode metrics (P1)
        if ep >= P0_EPISODES:
            if len(ep_z_harm_s) > 10 and len(ep_z_harm_a) > 10:
                zs = torch.stack(ep_z_harm_s)
                za = torch.stack(ep_z_harm_a)
                min_d = min(zs.shape[1], za.shape[1])
                cos = F.cosine_similarity(
                    zs[:, :min_d].mean(0).unsqueeze(0),
                    za[:, :min_d].mean(0).unsqueeze(0),
                ).item()
                stream_corrs.append(cos)

            if len(ep_fwd_preds) > 5:
                preds = torch.stack(ep_fwd_preds)
                targets = torch.stack(ep_fwd_targets)
                ss_res = ((preds - targets) ** 2).sum().item()
                ss_tot = ((targets - targets.mean(0)) ** 2).sum().item()
                r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
                fwd_r2_scores.append(r2)

            if len(ep_urgency) > 5:
                u = torch.tensor(ep_urgency)
                urgency_stds.append(u.std().item())

    mean_r2 = sum(fwd_r2_scores) / max(len(fwd_r2_scores), 1)
    mean_corr = sum(stream_corrs) / max(len(stream_corrs), 1)
    mean_urg_std = sum(urgency_stds) / max(len(urgency_stds), 1)

    return {
        "seed": seed,
        "condition": condition,
        "mean_fwd_r2": round(mean_r2, 4),
        "mean_stream_corr": round(mean_corr, 4),
        "mean_urgency_std": round(mean_urg_std, 4),
    }


def main():
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    print(f"[EXQ-262] run_id = {run_id}")

    all_results = []
    for seed in SEEDS:
        for cond in CONDITIONS:
            print(f"  seed={seed} condition={cond} ...", end=" ", flush=True)
            r = run_condition(seed, cond)
            print(
                f"fwd_r2={r['mean_fwd_r2']:.3f} stream_corr={r['mean_stream_corr']:.3f} "
                f"urg_std={r['mean_urgency_std']:.4f}"
            )
            all_results.append(r)

    # Per-seed criteria
    c1_count = 0
    c2_count = 0
    c3_count = 0
    for s in SEEDS:
        hub = next(r for r in all_results if r["seed"] == s and r["condition"] == "S_TO_A_HUB")
        no = next(r for r in all_results if r["seed"] == s and r["condition"] == "NO_HUB")
        if hub["mean_fwd_r2"] > no["mean_fwd_r2"]:
            c1_count += 1
        if abs(hub["mean_stream_corr"]) < STREAM_CORR_CEILING:
            c2_count += 1
        if hub["mean_urgency_std"] < no["mean_urgency_std"]:
            c3_count += 1

    c1_pass = c1_count >= SEED_PASS_QUOTA
    c2_pass = c2_count >= SEED_PASS_QUOTA
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
            "MECH-220": ed,
            "SD-011": "supports" if c2_pass else "mixed",
        },
        "criteria": {
            "C1_fwd_r2_improvement": {"pass": c1_pass, "seeds_passing": c1_count},
            "C2_stream_distinctness": {
                "pass": c2_pass, "seeds_passing": c2_count, "ceiling": STREAM_CORR_CEILING,
            },
            "C3_urgency_stability": {"informational": True, "seeds_passing": c3_count},
        },
        "per_seed_results": all_results,
        "config_summary": {
            "grid_size": GRID_SIZE,
            "p0_episodes": P0_EPISODES,
            "p1_episodes": P1_EPISODES,
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
    print(f"\n[EXQ-262] {overall} -- wrote {out_file.name}")
    print(f"  C1 fwd_r2: {'PASS' if c1_pass else 'FAIL'} ({c1_count}/{len(SEEDS)})")
    print(f"  C2 stream_corr: {'PASS' if c2_pass else 'FAIL'} ({c2_count}/{len(SEEDS)})")


if __name__ == "__main__":
    main()
