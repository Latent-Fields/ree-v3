#!/opt/local/bin/python3
"""
V3-EXQ-961 -- MECH-144: Ventral-Analog Spatially-Graded Valence Gradient Probe

experiment_purpose: evidence
SLEEP DRIVER: n/a (no sleep loop used)

Claim: MECH-144 (hippocampus.ventral_ca1_valence_encoding)
Backlog: EVB-0599 (proposal EXP-0459)
Design docs:
  REE_assembly/docs/claims/claims.yaml (MECH-144, MECH-073, ARC-040)

MECH-144 claim
--------------
Ventral CA1 contains spatially organised valence encoding: Jimenez et al. (Cell
2018) found "anxiety cells" that fire in PROPORTION TO DISTANCE from an aversive
open space -- a GRADED spatial signal, not a binary discrimination. Knudsen &
Wallis (Neuron 2021) show primate hippocampus constructs abstract value maps
with value geometry intrinsic to the representational structure. MECH-144
explicitly supports MECH-073 (valence intrinsic to hippocampal map geometry).

WHY THIS IS NOT A RE-RUN OF V3-EXQ-375 / V3-EXQ-165
----------------------------------------------------
Two prior experiments already tag MECH-144:
  - V3-EXQ-375 (MECH-073, 2 PASS runs, supports): tests whether the
    HARM_DISCRIMINATIVE valence component (SD-014, ResidueField) BINARILY
    discriminates harm-visited z_world points from safe ones (AUROC of harm-
    event membership). It does not tag MECH-144 in claim_ids and does not test
    whether the signal is spatially GRADED.
  - V3-EXQ-165 (MECH-143 / MECH-144, 2 FAIL/inconclusive runs): tests whether
    the (undifferentiated V3) HippocampalModule's TRAJECTORY NAVIGATION degrades
    under goal-value shuffling. This is a navigation-robustness DV, not a
    representational-geometry DV, and both runs read inconclusive/mixed for
    MECH-144 specifically (evidence_direction_per_claim: MECH-144: inconclusive).

The specific empirical signature MECH-144 cites (anxiety cells firing in
PROPORTION TO DISTANCE) is a claim about GRADATION, which V3-EXQ-375's binary
AUROC design cannot distinguish from a step-function/threshold code -- an AUROC
of 1.0 is equally consistent with "valence increases smoothly with proximity"
and with "valence is high near hazard, flat zero everywhere else". This
experiment isolates the graded-vs-binary question directly: does the
HARM_DISCRIMINATIVE valence readout, sampled continuously across the grid,
CORRELATE with physical distance-to-nearest-hazard -- not just discriminate
harm-visited points from safe ones.

GOV-REUSE-1 check (Step 2.4)
-----------------------------
Decisive readout: Pearson correlation between per-step HARM_DISCRIMINATIVE
valence (evaluate_valence()) and physical distance-to-nearest-hazard, sampled
continuously (not just at harm-event steps). Manually inspected both existing
MECH-144/MECH-073-tagged manifests (V3-EXQ-375's two PASS runs, V3-EXQ-165's
two FAIL runs): neither records per-step z_world/position/distance raw data
(only aggregate AUROC / criteria dicts survive), and both carry
substrate_hash: None (pre-standard, 2026-04/2026-05 era) -- per the recording
standard's pre-standard caveat, an unverifiable substrate is treated as NOT
RECOVERABLE regardless. `reanalysis_query.py` (run from REE_assembly/) also
returned 0 matches on the candidate readout names. Not recoverable -> run.

Substrate readiness (Step 2.5 / 2.5a)
--------------------------------------
No v3_pending / implementation_phase gate on MECH-144 in claims.yaml. The
mechanism this experiment exercises -- SD-014's per-center HARM_DISCRIMINATIVE
valence vector (ree_core/residue/field.py, ResidueField.update_valence /
evaluate_valence) -- is BUILT and already empirically exercised end-to-end by
V3-EXQ-375 (2 PASS runs, harm_valence_auroc up to 0.977). ARC-040 (dorsal/
ventral architectural SEGREGATION into two distinct modules) is explicitly
V4-scoped ("V3 HippocampalModule is undifferentiated... V4 architectural
requirement"), but MECH-144 itself does not assert that segregation -- it
asserts that valence IS geometrically/spatially encoded, which the single V3
ResidueField/RBF valence map already implements as an undifferentiated
approximation of the ventral-analog pathway (exactly the substrate V3-EXQ-375
used to support MECH-073). No corrupting open substrate_queue.json entry
overlaps ree_core/residue/field.py (checked 2026-08-29; step 2.5c gate).

Design
------
Discriminative pair (mirrors V3-EXQ-375's proven harness):
  ARM_GEOM     -- valence_enabled=True, benefit_terrain_enabled=True,
                  z_goal_enabled=True (same REEConfig knobs as V3-EXQ-375's
                  VALENCE_GEOM condition).
  ARM_ABLATED  -- valence_enabled=False (evaluate_valence returns zeros).

3 seeds (101, 202, 303 -- distinct from V3-EXQ-375's [42,7,13] and
V3-EXQ-165's seeds, satisfying the proposal's distinct_seeds policy) x 2 arms
= 6 cells.

Phase structure (identical schedule to V3-EXQ-375, a known-working budget):
  P0 (80 ep): encoder warmup, alpha_world=0.9 (z_world fidelity requirement
              for a representational-geometry probe -- default 0.3 is SD-008's
              known failure mode).
  P1 (80 ep): waking loop. On harm events (ARM_GEOM only) write
              HARM_DISCRIMINATIVE valence at z_world scaled by harm magnitude,
              exactly as V3-EXQ-375's P1.
  P2 (40 ep): eval + probe. At EVERY step (not just harm-event steps), record
              (z_world, agent grid position, harm label). After the run,
              compute physical Euclidean distance from the agent's position to
              the nearest hazard cell (env.hazards, ground truth, NOT a
              learned quantity) for every recorded step, query
              evaluate_valence(z_stack) for the HARM_DISCRIMINATIVE component,
              and compute proximity = -distance. The correlation is computed
              in PHYSICAL/ground-truth distance space paired with the LEARNED
              z_world's valence readout -- so the DV is whether the substrate's
              representation recovers a spatial gradient that exists in the
              world, not a tautology.

Pre-registered thresholds
--------------------------
  C1 (load-bearing): ARM_GEOM pearson_r(harm_valence, proximity) >= R_GEOM_MIN
      on >= 2/3 seeds. A graded, monotonically-proximity-scaling valence
      signal (the anxiety-cell signature), not merely a binary discriminator.
  C2: ARM_ABLATED |pearson_r(harm_valence, proximity)| < R_NULL_MAX
      on >= 2/3 seeds. No spurious geometry without the valence mechanism
      (evaluate_valence returns exact zeros when disabled, so this should be
      trivially satisfied -- it is a sanity/instrument check, not a novel
      prediction).
  C3 (readiness / non-degeneracy, load-bearing): ARM_GEOM
      harm_active_centers >= MIN_ACTIVE_CENTERS AND distance_std >=
      MIN_DISTANCE_STD on >= 2/3 seeds -- enough spatial sampling variety and
      populated valence geometry to make the correlation test non-trivial
      (mirrors V3-EXQ-375's C3, plus the distance-spread precondition this
      design specifically needs: a probe sample clustered at one distance
      cannot support a correlation claim regardless of the true relationship).

PASS: C1 AND C2 AND C3 (all on >= 2/3 seeds).

DV-symmetry note: the manipulation (valence_enabled True/False) is not a
uniform additive constant, monotone rescaling, or permutation of the DV
(a Pearson correlation between a learned scalar readout and a ground-truth
physical distance) -- ARM_ABLATED's evaluate_valence returns identically zero
by construction (not a rescaled/shifted version of ARM_GEOM's signal), so the
manipulation is not invariant under any symmetry the correlation DV is blind
to. C2 is expected to hold trivially (r undefined on a constant zero vector is
defined as 0.0 by the guarded pearson helper below) and is recorded as a
sanity check on the harness rather than a novel discriminative claim.

Secondary (non-load-bearing, generous recording): LIKING-component-vs-
distance-to-nearest-resource is also recorded for ARM_GEOM under
`custom_information.secondary_reward_geometry` -- MECH-144's "abstract value
map" half of the citation is not restricted to aversive geography, but a
resource-approach analogue is not pre-registered here and does not gate
PASS/FAIL.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import argparse
import statistics
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from ree_core.residue.field import VALENCE_HARM_DISCRIMINATIVE, VALENCE_LIKING

from experiments.pack_writer import write_flat_manifest
from experiments._metrics import check_degeneracy
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiment_protocol import emit_outcome

# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE    = "v3_exq_961_mech144_ventral_valence_spatial_gradient_probe"
CLAIM_IDS          = ["MECH-144"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Hyperparameters (schedule matched to V3-EXQ-375's proven-working budget)
# ---------------------------------------------------------------------------
SEEDS      = [101, 202, 303]
ARMS       = ["ARM_GEOM", "ARM_ABLATED"]

P0_EPISODES  = 80
P1_EPISODES  = 80
P2_EPISODES  = 40
STEPS_PER_EP = 200

GRID_SIZE     = 8
NUM_RESOURCES = 3
NUM_HAZARDS   = 3
HAZARD_HARM   = 0.15

LR = 3e-4

# Pre-registered thresholds
R_GEOM_MIN        = 0.35   # C1: ARM_GEOM correlation floor (graded signal)
R_NULL_MAX        = 0.15   # C2: ARM_ABLATED correlation ceiling (no geometry)
MIN_ACTIVE_CENTERS = 4     # C3a: populated valence geometry (matches V3-EXQ-375)
MIN_DISTANCE_STD   = 0.5   # C3b: enough spatial spread in the sampled distances
MIN_SEEDS_PASS     = 2     # out of 3

DRY_RUN_EPISODES = 3
DRY_RUN_STEPS    = 20

_ZG = ZGoalStreamAccumulator()


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


def _make_agent(condition: str, seed: int, env: CausalGridWorldV2) -> REEAgent:
    torch.manual_seed(seed)
    geom = (condition == "ARM_GEOM")
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
        benefit_eval_enabled=geom,
        benefit_weight=0.5 if geom else 0.0,
        drive_weight=2.0 if geom else 0.0,
        z_goal_enabled=geom,
    )
    config.residue.valence_enabled = geom
    config.residue.benefit_terrain_enabled = geom
    return REEAgent(config)


def _config_slice(condition: str, seed: int) -> Dict:
    """Everything a cell's build+collect path reads (used for arm_fingerprint)."""
    geom = (condition == "ARM_GEOM")
    return {
        "condition": condition,
        "seed": seed,
        "grid_size": GRID_SIZE,
        "num_resources": NUM_RESOURCES,
        "num_hazards": NUM_HAZARDS,
        "hazard_harm": HAZARD_HARM,
        "p0_episodes": P0_EPISODES,
        "p1_episodes": P1_EPISODES,
        "p2_episodes": P2_EPISODES,
        "steps_per_ep": STEPS_PER_EP,
        "lr": LR,
        "alpha_world": 0.9,
        "valence_enabled": geom,
        "benefit_terrain_enabled": geom,
        "z_goal_enabled": geom,
        "benefit_weight": 0.5 if geom else 0.0,
        "drive_weight": 2.0 if geom else 0.0,
    }


# ---------------------------------------------------------------------------
# Stats helpers (no scipy dependency)
# ---------------------------------------------------------------------------

def _pearson_r(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation. Returns 0.0 (not NaN) on <2 points or zero variance
    in either variable -- both are treated as "no detectable relationship",
    which is the correct reading for the ARM_ABLATED null (evaluate_valence is
    an exact zero vector there, so xs has zero variance by construction)."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 1e-12 or var_y <= 1e-12:
        return 0.0
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / ((var_x ** 0.5) * (var_y ** 0.5))


def _nearest_distance(pos: tuple, targets: List[List[int]]) -> Optional[float]:
    if not targets:
        return None
    ax, ay = pos
    return min(((ax - tx) ** 2 + (ay - ty) ** 2) ** 0.5 for tx, ty in targets)


# ---------------------------------------------------------------------------
# Single cell (seed x arm)
# ---------------------------------------------------------------------------

def run_cell(seed: int, condition: str, dry_run: bool = False) -> Dict:
    total_p0  = DRY_RUN_EPISODES if dry_run else P0_EPISODES
    total_p1  = DRY_RUN_EPISODES if dry_run else P1_EPISODES
    total_p2  = DRY_RUN_EPISODES if dry_run else P2_EPISODES
    steps_per = DRY_RUN_STEPS    if dry_run else STEPS_PER_EP
    total_eps = total_p0 + total_p1 + total_p2

    print(f"  Seed {seed} Condition {condition}", flush=True)

    geom = (condition == "ARM_GEOM")
    env   = _make_env(seed)
    agent = _make_agent(condition, seed, env)
    device = agent.device
    optimizer = optim.Adam(list(agent.parameters()), lr=LR)

    # P2 probe buffers
    probe_zworld: List[torch.Tensor] = []
    probe_harm_dist: List[float] = []
    probe_labels: List[int] = []
    probe_resource_zworld: List[torch.Tensor] = []
    probe_resource_dist: List[float] = []

    prev_ttype = "none"

    for ep in range(total_eps):
        _, obs_dict = env.reset()
        agent.reset()

        phase   = "P0" if ep < total_p0 else ("P1" if ep < total_p0 + total_p1 else "P2")
        in_eval = (phase == "P2")

        ep_benefit = 0.0
        ep_harm    = 0.0

        for _step in range(steps_per):
            agent_pos = env.get_agent_position()
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

            if geom and phase != "P0":
                benefit_raw = float(obs_body.flatten()[11].item()) if obs_body.shape[-1] > 11 else 0.0
                drive_level = REEAgent.compute_drive_level(obs_body)
                agent.update_z_goal(benefit_raw, drive_level)
            _ZG.observe(agent)

            flat_next, harm_signal, done, info, obs_dict_next = env.step(action_idx)
            ttype = info.get("transition_type", "none")

            harm_val    = abs(float(harm_signal)) if float(harm_signal) < 0 else 0.0
            benefit_val = max(0.0, float(info.get("benefit_exposure", 0.0)))

            ep_benefit += benefit_val
            ep_harm    += harm_val

            agent.update_residue(float(harm_signal))

            if geom and phase != "P0" and latent.z_world is not None:
                z_w = latent.z_world.detach()
                if harm_val > 0.01:
                    agent.residue_field.update_valence(z_w, VALENCE_HARM_DISCRIMINATIVE, harm_val * 0.5)
                if benefit_val > 0.01:
                    agent.residue_field.update_valence(z_w, VALENCE_LIKING, benefit_val * 0.5)

            if in_eval and latent.z_world is not None:
                hazard_dist = _nearest_distance(agent_pos, env.hazards)
                if hazard_dist is not None:
                    probe_zworld.append(latent.z_world.detach().cpu().squeeze(0))
                    probe_harm_dist.append(hazard_dist)
                    probe_labels.append(1 if harm_val > 0.01 else 0)
                resources = getattr(env, "resources", None)
                if resources:
                    res_positions = [r[:2] if isinstance(r, (list, tuple)) else r for r in resources]
                    res_dist = _nearest_distance(agent_pos, res_positions)
                    if res_dist is not None:
                        probe_resource_zworld.append(latent.z_world.detach().cpu().squeeze(0))
                        probe_resource_dist.append(res_dist)

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
    # Probe: correlation of HARM_DISCRIMINATIVE valence with hazard proximity
    # -----------------------------------------------------------------------
    pearson_r = 0.0
    harm_active_centers = 0
    distance_std = 0.0
    n_probe = len(probe_labels)
    harm_component_vals: List[float] = []

    if probe_zworld:
        z_stack = torch.stack(probe_zworld)
        val_vec = agent.residue_field.evaluate_valence(z_stack)
        harm_component_vals = val_vec[:, VALENCE_HARM_DISCRIMINATIVE].tolist()
        proximity = [-d for d in probe_harm_dist]
        pearson_r = _pearson_r(harm_component_vals, proximity)
        distance_std = statistics.pstdev(probe_harm_dist) if len(probe_harm_dist) > 1 else 0.0

    rbf = agent.residue_field.rbf_field
    harm_active_centers = int(rbf.active_mask.sum().item())

    # Secondary, non-load-bearing: LIKING vs resource proximity (ARM_GEOM only)
    secondary_liking_r = 0.0
    if geom and probe_resource_zworld:
        z_res_stack = torch.stack(probe_resource_zworld)
        val_vec_res = agent.residue_field.evaluate_valence(z_res_stack)
        liking_vals = val_vec_res[:, VALENCE_LIKING].tolist()
        res_proximity = [-d for d in probe_resource_dist]
        secondary_liking_r = _pearson_r(liking_vals, res_proximity)

    verdict_str = (
        f"r={pearson_r:.4f} centers={harm_active_centers} "
        f"dist_std={distance_std:.4f} n={n_probe}"
    )
    print(f"  verdict: {verdict_str}", flush=True)

    return {
        "seed": seed,
        "condition": condition,
        "pearson_r_harm_vs_proximity": pearson_r,
        "harm_active_centers": harm_active_centers,
        "distance_std": distance_std,
        "n_probe": n_probe,
        "secondary_liking_vs_resource_proximity_r": secondary_liking_r,
        "harm_valence_values_sample": harm_component_vals[:5],
    }


# ---------------------------------------------------------------------------
# Criteria evaluation
# ---------------------------------------------------------------------------

def evaluate_criteria(all_results: List[Dict]) -> Dict:
    by_arm: Dict[str, List[Dict]] = defaultdict(list)
    for r in all_results:
        by_arm[r["condition"]].append(r)

    geom_list    = sorted(by_arm.get("ARM_GEOM",    []), key=lambda x: x["seed"])
    ablated_list = sorted(by_arm.get("ARM_ABLATED", []), key=lambda x: x["seed"])

    c1_seeds = sum(r["pearson_r_harm_vs_proximity"] >= R_GEOM_MIN for r in geom_list)
    c1_pass  = c1_seeds >= MIN_SEEDS_PASS

    c2_seeds = sum(abs(r["pearson_r_harm_vs_proximity"]) < R_NULL_MAX for r in ablated_list)
    c2_pass  = c2_seeds >= MIN_SEEDS_PASS

    c3_seeds = sum(
        (r["harm_active_centers"] >= MIN_ACTIVE_CENTERS) and (r["distance_std"] >= MIN_DISTANCE_STD)
        for r in geom_list
    )
    c3_pass = c3_seeds >= MIN_SEEDS_PASS

    overall_pass = c1_pass and c2_pass and c3_pass

    return {
        "c1_geom_gradient_pass": c1_pass,
        "c1_seeds_pass": c1_seeds,
        "c1_geom_r_values": [r["pearson_r_harm_vs_proximity"] for r in geom_list],
        "c1_threshold": R_GEOM_MIN,
        "c2_ablated_null_pass": c2_pass,
        "c2_seeds_pass": c2_seeds,
        "c2_ablated_r_values": [r["pearson_r_harm_vs_proximity"] for r in ablated_list],
        "c2_threshold": R_NULL_MAX,
        "c3_readiness_pass": c3_pass,
        "c3_seeds_pass": c3_seeds,
        "c3_geom_active_centers": [r["harm_active_centers"] for r in geom_list],
        "c3_geom_distance_std": [r["distance_std"] for r in geom_list],
        "c3_min_active_centers": MIN_ACTIVE_CENTERS,
        "c3_min_distance_std": MIN_DISTANCE_STD,
        "overall_pass": overall_pass,
        "criteria": [
            {"name": "C1_geom_gradient_correlation", "load_bearing": True, "passed": c1_pass},
            {"name": "C2_ablated_null", "load_bearing": True, "passed": c2_pass},
            {"name": "C3_readiness_non_degenerate", "load_bearing": True, "passed": c3_pass},
        ],
        "combination_rule": "PASS iff C1 AND C2 AND C3, each on >= 2/3 seeds",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run(dry_run: bool):
    t0 = time.perf_counter()

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        f"{EXPERIMENT_TYPE}_dry_{ts}_v3"
        if dry_run
        else f"{EXPERIMENT_TYPE}_{ts}_v3"
    )
    print(f"EXQ-961 start: {run_id}", flush=True)

    all_results: List[Dict] = []
    arm_results: List[Dict] = []

    for seed in SEEDS:
        print(f"\n--- seed {seed} ---", flush=True)
        for condition in ARMS:
            with arm_cell(
                seed,
                config_slice=_config_slice(condition, seed),
                script_path=Path(__file__),
                config_slice_declared=True,
            ) as cell:
                result = run_cell(seed, condition, dry_run=dry_run)
                cell.stamp(result)
            all_results.append(result)
            arm_results.append(result)

    criteria = evaluate_criteria(all_results)
    outcome  = "PASS" if criteria["overall_pass"] else "FAIL"

    print(f"\n=== EXQ-961 {outcome} ===", flush=True)
    print(
        f"C1 geom_gradient: {criteria['c1_geom_gradient_pass']} "
        f"({criteria['c1_seeds_pass']}/{len(SEEDS)} seeds) "
        f"r={criteria['c1_geom_r_values']} threshold>={criteria['c1_threshold']}",
        flush=True,
    )
    print(
        f"C2 ablated_null: {criteria['c2_ablated_null_pass']} "
        f"({criteria['c2_seeds_pass']}/{len(SEEDS)} seeds) "
        f"r={criteria['c2_ablated_r_values']} threshold<{criteria['c2_threshold']}",
        flush=True,
    )
    print(
        f"C3 readiness: {criteria['c3_readiness_pass']} "
        f"({criteria['c3_seeds_pass']}/{len(SEEDS)} seeds) "
        f"centers={criteria['c3_geom_active_centers']} dist_std={criteria['c3_geom_distance_std']}",
        flush=True,
    )

    # Non-degeneracy scoring net (evidence run -- Step 3 "Non-degeneracy scoring net")
    geom_r_values    = [r["pearson_r_harm_vs_proximity"] for r in all_results if r["condition"] == "ARM_GEOM"]
    geom_dist_groups = [
        [r["distance_std"]] for r in all_results if r["condition"] == "ARM_GEOM"
    ]
    degeneracy = check_degeneracy({
        "c1_geom_gradient_correlation": {"values": geom_r_values},
        "c3_geom_distance_spread": {"groups": geom_dist_groups, "floor": 1e-6},
    })

    full_config = {
        "seeds": SEEDS,
        "arms": ARMS,
        "p0_episodes": P0_EPISODES,
        "p1_episodes": P1_EPISODES,
        "p2_episodes": P2_EPISODES,
        "steps_per_ep": STEPS_PER_EP,
        "num_hazards": NUM_HAZARDS,
        "hazard_harm": HAZARD_HARM,
        "grid_size": GRID_SIZE,
        "num_resources": NUM_RESOURCES,
        "lr": LR,
        "alpha_world": 0.9,
        "r_geom_min": R_GEOM_MIN,
        "r_null_max": R_NULL_MAX,
        "min_active_centers": MIN_ACTIVE_CENTERS,
        "min_distance_std": MIN_DISTANCE_STD,
        "min_seeds_pass": MIN_SEEDS_PASS,
    }

    manifest = {
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
            "c1_r_geom_min": R_GEOM_MIN,
            "c2_r_ablated_max": R_NULL_MAX,
            "c3_min_active_centers": MIN_ACTIVE_CENTERS,
            "c3_min_distance_std": MIN_DISTANCE_STD,
            "min_seeds_pass": MIN_SEEDS_PASS,
        },
        "criteria": criteria,
        "arm_results": arm_results,
        "summary": (
            f"MECH-144 ventral-analog spatial-gradient valence probe. "
            f"ARM_GEOM pearson_r={criteria['c1_geom_r_values']} vs "
            f"ARM_ABLATED pearson_r={criteria['c2_ablated_r_values']}. "
            f"Outcome: {outcome}. "
            f"Interpretation: {'HARM_DISCRIMINATIVE valence forms a spatially graded gradient scaling with hazard proximity (anxiety-cell-analog signature); MECH-144 supported' if criteria['overall_pass'] else 'graded spatial valence gradient not detected at current scale -- MECH-144 not supported by this design'}"
        ),
        "dv_symmetry_note": (
            "Manipulation (valence_enabled True/False) is not a uniform additive "
            "constant, monotone rescaling, or permutation of the correlation DV: "
            "ARM_ABLATED's evaluate_valence returns an exact zero vector by "
            "construction (not a transformed copy of ARM_GEOM's signal), so the "
            "DV is not blind to the manipulation. C2's near-zero correlation is "
            "an instrument sanity check, not a novel discriminative claim."
        ),
        "custom_information": {
            "gov_reuse_1_check": (
                "Manually inspected V3-EXQ-375 (2 PASS, MECH-073) and V3-EXQ-165 "
                "(2 FAIL/inconclusive, MECH-143/MECH-144): neither records "
                "per-step z_world/position/distance raw data, and both carry "
                "substrate_hash: None (pre-standard). reanalysis_query.py (from "
                "REE_assembly/) returned 0 matches for candidate readout names. "
                "Not recoverable -> ran fresh."
            ),
            "secondary_reward_geometry": {
                "note": "Non-load-bearing. LIKING-vs-resource-proximity correlation, ARM_GEOM only.",
                "per_seed_r": [
                    {"seed": r["seed"], "r": r["secondary_liking_vs_resource_proximity_r"]}
                    for r in all_results if r["condition"] == "ARM_GEOM"
                ],
            },
        },
    }
    manifest.update(degeneracy)

    out_path = write_flat_manifest(
        manifest,
        dry_run=bool(dry_run),
        config=full_config,
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    print(f"Results -> {out_path}", flush=True)

    return outcome, out_path, run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _outcome, _out_path, _run_id = _run(args.dry_run)

    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(_out_path),
        run_id=_run_id,
        dry_run=bool(args.dry_run),
    )
