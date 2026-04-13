#!/opt/local/bin/python3
"""
V3-EXQ-375: MECH-073 Valence Geometry Probe -- Discriminative Support vs Ablation Pair

experiment_purpose: evidence

MECH-073 claim: Valence is intrinsic to hippocampal map geometry, not applied
downstream. Under Q-020 Resolution A, this means the 4-component valence vectors
stored in the RBF-center map (SD-014, ResidueField.valence_vecs) encode valence
geometrically: harm and benefit information is recoverable from z_world position
alone, not from a separate downstream lookup.

Discriminative pair design (EVB-0057, EXP-0057):

  VALENCE_GEOM (support condition):
    valence_enabled=True, benefit_terrain_enabled=True
    Harm and benefit events populate valence vectors at z_world positions.
    After training, probe: VALENCE_HARM_DISCRIMINATIVE component is higher at
    z_world positions visited during harm events vs safe steps (spatial structure).
    Key metric: harm_valence_auroc -- AUROC of HARM_DISCRIMINATIVE component
    predicting harm event membership from z_world positions.

  VALENCE_ABLATED (ablation condition):
    valence_enabled=False -- no valence vectors written anywhere.
    evaluate_valence() returns zeros for all queries.
    Key metric: harm_valence_auroc should be near 0.5 (chance).

Pre-registered thresholds:
  C1: VALENCE_GEOM harm_valence_auroc >= 0.65 (>= 2/3 seeds)
      Valence HARM_DISCRIMINATIVE spatially discriminates harm vs safe zones
      in the residue map geometry.
  C2: VALENCE_ABLATED harm_valence_auroc < 0.60 (>= 2/3 seeds)
      Ablated condition stays near chance -- no spurious geometry.
  C3: VALENCE_GEOM harm_active_centers >= 4 (>= 2/3 seeds)
      Enough RBF centers are populated to make the geometry claim non-trivial.

PASS: C1 AND C2 AND C3 across >= 2/3 seeds.

Architecture notes:
  - SD-014: 4-component valence vector per RBF center, component order
    [wanting, liking, harm_discriminative, surprise].
  - VALENCE_HARM_DISCRIMINATIVE (index 2) is updated via
    residue_field.update_valence(z_world, VALENCE_HARM_DISCRIMINATIVE, value)
    during harm events. This is the primary test component.
  - Q-020 Resolution A: "No value computation" constraint is compatible with
    valence being geometrically encoded in the map via BTSP-mediated writes
    (Bittner 2017). ARC-007 and MECH-073 are co-true.
  - Conflict with ARC-007: this experiment tests Resolution A directly.
    A PASS supports Resolution A. A FAIL would support the strict reading
    (MECH-073 incompatible with ARC-007).

Phase structure:
  P0 (80 ep): Encoder warmup -- z_world learns environmental structure.
              No valence evaluation, minimal harm stream.
  P1 (80 ep): Full waking loop -- harm events populate valence vectors.
              Residue accumulates. Benefit terrain enabled in VALENCE_GEOM.
  P2 (40 ep): Eval + probe -- sample z_world at harm/safe events, query
              evaluate_valence(), compute AUROC.

Claims: MECH-073
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.optim as optim
import numpy as np

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig, ResidueConfig
from ree_core.agent import REEAgent
from ree_core.residue.field import (
    VALENCE_HARM_DISCRIMINATIVE, VALENCE_LIKING, VALENCE_WANTING, VALENCE_SURPRISE
)

# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE    = "v3_exq_375_mech073_valence_geometry_probe"
CLAIM_IDS          = ["MECH-073"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEEDS      = [42, 7, 13]
CONDITIONS = ["VALENCE_GEOM", "VALENCE_ABLATED"]

P0_EPISODES  = 80
P1_EPISODES  = 80
P2_EPISODES  = 40
STEPS_PER_EP = 200

GRID_SIZE     = 8
NUM_RESOURCES = 3
NUM_HAZARDS   = 3   # enough to populate harm valence in map
HAZARD_HARM   = 0.15

LR = 3e-4

# Pre-registered thresholds
AUROC_GEOM_THRESHOLD    = 0.65   # C1: VALENCE_GEOM harm AUROC must exceed this
AUROC_ABLATED_THRESHOLD = 0.60   # C2: VALENCE_ABLATED harm AUROC must stay below this
MIN_ACTIVE_CENTERS      = 4      # C3: at minimum this many active centers in GEOM
MIN_SEEDS_PASS          = 2      # out of 3

DRY_RUN_EPISODES = 3
DRY_RUN_STEPS    = 20


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_resources=NUM_RESOURCES,
        num_hazards=NUM_HAZARDS,
        hazard_harm=HAZARD_HARM,
        resource_benefit=0.5,
        resource_respawn_on_consume=True,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.05,
        proximity_approach_threshold=0.15,
        use_proxy_fields=True,
    )


def _make_agent(env: CausalGridWorldV2, condition: str, seed: int) -> REEAgent:
    torch.manual_seed(seed)
    geom = (condition == "VALENCE_GEOM")
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        use_event_classifier=True,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        # Benefit terrain: enabled in GEOM so benefit valence also populates
        benefit_eval_enabled=geom,
        benefit_weight=0.5 if geom else 0.0,
        drive_weight=2.0 if geom else 0.0,
        z_goal_enabled=geom,
    )
    # Set valence_enabled per condition (SD-014)
    config.residue.valence_enabled = geom
    config.residue.benefit_terrain_enabled = geom
    return REEAgent(config)


# ---------------------------------------------------------------------------
# AUROC helper (no sklearn dependency)
# ---------------------------------------------------------------------------

def _auroc(scores: List[float], labels: List[int]) -> float:
    """Compute AUROC of scores predicting binary labels.
    Returns 0.5 on trivial inputs (all same label, or empty).
    """
    n = len(scores)
    if n == 0:
        return 0.5
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return 0.5
    # Mann-Whitney U statistic
    n_pos = len(pos)
    n_neg = len(neg)
    concordant = 0
    for p in pos:
        for ng in neg:
            if p > ng:
                concordant += 1
            elif p == ng:
                concordant += 0.5
    return concordant / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_condition(
    seed: int,
    condition: str,
    dry_run: bool = False,
) -> Dict:
    total_p0  = DRY_RUN_EPISODES if dry_run else P0_EPISODES
    total_p1  = DRY_RUN_EPISODES if dry_run else P1_EPISODES
    total_p2  = DRY_RUN_EPISODES if dry_run else P2_EPISODES
    steps_per = DRY_RUN_STEPS    if dry_run else STEPS_PER_EP
    total_eps = total_p0 + total_p1 + total_p2

    print(f"  Seed {seed} Condition {condition}", flush=True)

    env   = _make_env(seed)
    agent = _make_agent(env, condition, seed)
    device = agent.device
    geom = (condition == "VALENCE_GEOM")

    optimizer = optim.Adam(list(agent.parameters()), lr=LR)

    # Probe buffers (P2): collect z_world + label (1=harm, 0=safe) per step
    probe_zworld: List[torch.Tensor] = []
    probe_labels: List[int]          = []

    prev_ttype = "none"

    for ep in range(total_eps):
        _, obs_dict = env.reset()
        agent.reset()

        phase   = "P0" if ep < total_p0 else ("P1" if ep < total_p0 + total_p1 else "P2")
        in_eval = (phase == "P2")

        ep_benefit   = 0.0
        ep_harm      = 0.0

        for _step in range(steps_per):
            obs_body  = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)

            z_self_prev: Optional[torch.Tensor] = None
            if agent._current_latent is not None:
                z_self_prev = agent._current_latent.z_self.detach().clone()

            latent = agent.sense(obs_body, obs_world)
            ticks  = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent)
                if ticks.get("e1_tick", True)
                else torch.zeros(1, 32, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action     = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item())

            # Goal seeding (GEOM only, P1/P2)
            if geom and phase != "P0":
                benefit_raw = float(obs_body.flatten()[11].item()) if obs_body.shape[-1] > 11 else 0.0
                drive_level = REEAgent.compute_drive_level(obs_body)
                agent.update_z_goal(benefit_raw, drive_level)

            flat_next, harm_signal, done, info, obs_dict_next = env.step(action_idx)
            ttype = info.get("transition_type", "none")

            harm_val    = abs(float(harm_signal)) if float(harm_signal) < 0 else 0.0
            benefit_val = max(0.0, float(info.get("benefit_exposure", 0.0)))

            ep_benefit += benefit_val
            ep_harm    += harm_val

            agent.update_residue(float(harm_signal))

            # In GEOM condition: populate HARM_DISCRIMINATIVE valence at z_world
            # when harm occurs (P1/P2), and LIKING valence on benefit events.
            if geom and phase != "P0" and latent.z_world is not None:
                z_w = latent.z_world.detach()
                if harm_val > 0.01:
                    agent.residue_field.update_valence(
                        z_w, VALENCE_HARM_DISCRIMINATIVE, harm_val * 0.5
                    )
                if benefit_val > 0.01:
                    agent.residue_field.update_valence(
                        z_w, VALENCE_LIKING, benefit_val * 0.5
                    )

            # P2: record z_world and harm/safe label for probe
            if in_eval and latent.z_world is not None:
                probe_zworld.append(latent.z_world.detach().cpu().squeeze(0))
                probe_labels.append(1 if harm_val > 0.01 else 0)

            if z_self_prev is not None:
                agent.record_transition(z_self_prev, action, latent.z_self.detach())

            if not in_eval:
                optimizer.zero_grad()
                e1_loss = agent.compute_prediction_loss()
                e2_loss = agent.compute_e2_loss()
                loss    = e1_loss + e2_loss

                rfv = obs_dict.get("resource_field_view", None)
                if rfv is not None:
                    rp_t = float(rfv.max().item())
                    loss = loss + agent.compute_resource_proximity_loss(rp_t, latent)

                latent2  = agent.sense(obs_body, obs_world)
                ec_loss  = agent.compute_event_contrastive_loss(prev_ttype, latent2)
                loss     = loss + ec_loss

                if geom and phase == "P1" and benefit_val > 0:
                    benefit_t = torch.tensor([[benefit_val]], dtype=torch.float32, device=device)
                    loss = loss + agent.compute_benefit_eval_loss(benefit_t)

                if loss.requires_grad:
                    loss.backward()
                    import torch.nn as nn
                    nn.utils.clip_grad_norm_(list(agent.parameters()), 1.0)
                    optimizer.step()

            prev_ttype = ttype
            obs_dict   = obs_dict_next

            if done:
                break

        if (ep + 1) % 40 == 0:
            print(
                f"    [train] seed={seed} {condition} ep {ep+1}/{total_eps} "
                f"phase={phase} benefit={ep_benefit:.3f} harm={ep_harm:.3f}",
                flush=True,
            )

    # -----------------------------------------------------------------------
    # Probe: compute AUROC of HARM_DISCRIMINATIVE valence vs harm labels
    # -----------------------------------------------------------------------
    harm_valence_auroc = 0.5
    harm_active_centers = 0
    n_probe_harm  = sum(probe_labels)
    n_probe_safe  = len(probe_labels) - n_probe_harm
    valence_scores: List[float] = []

    if probe_zworld:
        z_stack = torch.stack(probe_zworld)  # [N, world_dim]
        # Query valence geometry
        val_vec = agent.residue_field.evaluate_valence(z_stack)  # [N, 4]
        harm_component = val_vec[:, VALENCE_HARM_DISCRIMINATIVE].tolist()
        valence_scores = harm_component
        harm_valence_auroc = _auroc(harm_component, probe_labels)

    # Count active RBF centers
    rbf = agent.residue_field.rbf_field
    harm_active_centers = int(rbf.active_mask.sum().item())

    verdict_str = (
        f"auroc={harm_valence_auroc:.4f} centers={harm_active_centers} "
        f"n_harm={n_probe_harm} n_safe={n_probe_safe}"
    )
    print(f"  verdict: {verdict_str}", flush=True)

    return {
        "seed": seed,
        "condition": condition,
        "harm_valence_auroc": harm_valence_auroc,
        "harm_active_centers": harm_active_centers,
        "n_probe_harm": n_probe_harm,
        "n_probe_safe": n_probe_safe,
        "n_probe_total": len(probe_labels),
    }


# ---------------------------------------------------------------------------
# Criteria evaluation
# ---------------------------------------------------------------------------

def evaluate_criteria(all_results: List[Dict]) -> Dict:
    by_cond: Dict[str, List[Dict]] = defaultdict(list)
    for r in all_results:
        by_cond[r["condition"]].append(r)

    geom_list    = sorted(by_cond.get("VALENCE_GEOM",    []), key=lambda x: x["seed"])
    ablated_list = sorted(by_cond.get("VALENCE_ABLATED", []), key=lambda x: x["seed"])

    # C1: VALENCE_GEOM harm_valence_auroc >= AUROC_GEOM_THRESHOLD
    c1_seeds = sum(
        r["harm_valence_auroc"] >= AUROC_GEOM_THRESHOLD
        for r in geom_list
    )
    c1_pass = c1_seeds >= MIN_SEEDS_PASS

    # C2: VALENCE_ABLATED harm_valence_auroc < AUROC_ABLATED_THRESHOLD
    c2_seeds = sum(
        r["harm_valence_auroc"] < AUROC_ABLATED_THRESHOLD
        for r in ablated_list
    )
    c2_pass = c2_seeds >= MIN_SEEDS_PASS

    # C3: VALENCE_GEOM harm_active_centers >= MIN_ACTIVE_CENTERS
    c3_seeds = sum(
        r["harm_active_centers"] >= MIN_ACTIVE_CENTERS
        for r in geom_list
    )
    c3_pass = c3_seeds >= MIN_SEEDS_PASS

    overall_pass = c1_pass and c2_pass and c3_pass

    return {
        "c1_valence_geometry_auroc_pass": c1_pass,
        "c1_seeds_pass": c1_seeds,
        "c1_geom_aurocs": [r["harm_valence_auroc"] for r in geom_list],
        "c1_threshold": AUROC_GEOM_THRESHOLD,
        "c2_ablated_null_pass": c2_pass,
        "c2_seeds_pass": c2_seeds,
        "c2_ablated_aurocs": [r["harm_valence_auroc"] for r in ablated_list],
        "c2_threshold": AUROC_ABLATED_THRESHOLD,
        "c3_active_centers_pass": c3_pass,
        "c3_seeds_pass": c3_seeds,
        "c3_geom_active_centers": [r["harm_active_centers"] for r in geom_list],
        "c3_threshold": MIN_ACTIVE_CENTERS,
        "overall_pass": overall_pass,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        f"v3_exq_375_mech073_valence_geometry_probe_dry_{ts}_v3"
        if args.dry_run
        else f"v3_exq_375_mech073_valence_geometry_probe_{ts}_v3"
    )
    print(f"EXQ-375 start: {run_id}", flush=True)

    all_results: List[Dict] = []

    # Run seed boundary: for each seed, run both conditions with matched seed
    for seed in SEEDS:
        print(f"\n--- seed {seed} ---", flush=True)
        for condition in CONDITIONS:
            result = run_condition(seed, condition, dry_run=args.dry_run)
            all_results.append(result)

    criteria = evaluate_criteria(all_results)
    outcome  = "PASS" if criteria["overall_pass"] else "FAIL"

    print(f"\n=== EXQ-375 {outcome} ===", flush=True)
    print(
        f"C1 valence_geom_auroc: {criteria['c1_valence_geometry_auroc_pass']} "
        f"({criteria['c1_seeds_pass']}/{len(SEEDS)} seeds) "
        f"aurocs={criteria['c1_geom_aurocs']} threshold>={criteria['c1_threshold']}",
        flush=True,
    )
    print(
        f"C2 ablated_null: {criteria['c2_ablated_null_pass']} "
        f"({criteria['c2_seeds_pass']}/{len(SEEDS)} seeds) "
        f"aurocs={criteria['c2_ablated_aurocs']} threshold<{criteria['c2_threshold']}",
        flush=True,
    )
    print(
        f"C3 active_centers: {criteria['c3_active_centers_pass']} "
        f"({criteria['c3_seeds_pass']}/{len(SEEDS)} seeds) "
        f"centers={criteria['c3_geom_active_centers']} threshold>={criteria['c3_threshold']}",
        flush=True,
    )

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_class": "discriminative_pair",
        "evidence_direction": "supports" if criteria["overall_pass"] else "does_not_support",
        "outcome": outcome,
        "registered_thresholds": {
            "c1_auroc_geom_min": AUROC_GEOM_THRESHOLD,
            "c2_auroc_ablated_max": AUROC_ABLATED_THRESHOLD,
            "c3_min_active_centers": MIN_ACTIVE_CENTERS,
            "min_seeds_pass": MIN_SEEDS_PASS,
        },
        "criteria": criteria,
        "results_per_condition": all_results,
        "summary": (
            f"MECH-073 valence geometry probe (discriminative pair). "
            f"VALENCE_GEOM AUROC={criteria['c1_geom_aurocs']} vs "
            f"VALENCE_ABLATED AUROC={criteria['c2_ablated_aurocs']}. "
            f"Outcome: {outcome}. "
            f"Interpretation: {'valence is geometrically structured in the hippocampal map (Resolution A supported)' if criteria['overall_pass'] else 'valence geometry not detected at current scale -- MECH-073 not supported'}"
        ),
        "config": {
            "seeds": SEEDS,
            "conditions": CONDITIONS,
            "p0_episodes": P0_EPISODES,
            "p1_episodes": P1_EPISODES,
            "p2_episodes": P2_EPISODES,
            "steps_per_ep": STEPS_PER_EP,
            "num_hazards": NUM_HAZARDS,
            "hazard_harm": HAZARD_HARM,
            "grid_size": GRID_SIZE,
        },
        "timestamp_utc": ts,
    }

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "REE_assembly", "evidence", "experiments",
        EXPERIMENT_TYPE,
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
