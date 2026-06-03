#!/opt/local/bin/python3
"""
V3-EXQ-632: MECH-230 z_goal Structured Latent -- Discriminative Pair.

experiment_purpose: evidence
dispatch_mode: discriminative_pair

CLAIM UNDER TEST
  MECH-230 (status: provisional): "E3 maintains a structured latent goal
  representation (positive attractor in z_goal sub-space, measurable as
  z_goal_norm > 0) distinct from harm avoidance, enabling hippocampal terrain
  navigation toward goal states."

  This experiment tests the latent-structure arm of MECH-230 directly: does a
  structured z_goal attractor form (z_goal_norm > 0, and persist) when the
  structured-goal mechanism is ON, and stay collapsed at ~0 when the mechanism
  is explicitly ablated OFF?

DISCRIMINATIVE PAIR (exactly two matched conditions, shared seeds)
  GOAL_STRUCTURED (mechanism ON):
    z_goal_enabled=True, drive_weight=2.0 (SD-012 homeostatic amplification),
    benefit_threshold=0.1. The structured-goal seeding mechanism is active:
    on resource contact, update_z_goal() seeds the z_goal attractor scaled by
    drive amplification, so the latent develops non-zero structured norm.
  GOAL_ABLATED (mechanism OFF):
    z_goal_enabled=False -> agent.goal_state is None -> update_z_goal() early-
    returns and NO z_goal attractor is ever instantiated. This is the cleanest
    possible ablation of the MECH-230 mechanism: the z_goal sub-space does not
    exist, so z_goal_norm is identically 0 by construction. The discriminative
    contrast is "structured z_goal forms" vs "no z_goal mechanism".

  Both conditions share the SAME seeds (seed_policy = matched_shared_seeds) and
  the SAME environment / training schedule. Total runs = len(SEEDS) x 2.

MEASUREMENT
  During the P2 measurement phase, after each resource-contact event we sample
  z_goal_norm at t=0 (contact tick, post-seeding), t=10, t=25, t=50 post-contact.
  In GOAL_ABLATED there is no goal_state, so z_goal_norm reads 0.0 at every
  offset (the structured representation is absent by construction).

PHASED TRAINING (encoder warmup then frozen-encoder measurement)
  This script trains the z_world encoder + E1/E2 + the SD-018 resource-proximity
  head, then MEASURES z_goal seeding on the frozen encoder. z_goal itself is a
  pure-arithmetic attractor (GoalState.update has no learned parameters), so the
  collapse hazard is in the encoder/heads, not in z_goal. We therefore:
    P0 (encoder warmup): train E1 + E2 + resource-proximity head + event-
       contrastive head on the waking stream. NO z_goal seeding during P0
       (the mechanism is exercised only during measurement).
    P1 (freeze encoder): encoder params frozen via .requires_grad_(False); no
       optimizer steps. z_goal seeding fires on .detach()ed encoder output
       (GoalState.update detaches its seed latent internally; we additionally
       stop all gradient flow by running measurement under no_grad-equivalent
       frozen-encoder conditions).
    P2 (eval): the P1 measurement phase IS the eval -- z_goal_norm trajectories
       are aggregated for the registered thresholds. (P1 and P2 are merged into
       one frozen measurement phase; encoder is frozen and no head is trained on
       z_goal, so there is no moving-target collapse risk.)

PRE-REGISTERED THRESHOLDS (constants below, fixed before any run logic)
  C1_SEED_AT_CONTACT  = 0.10  -- GOAL_STRUCTURED z_goal_norm at t=0 must exceed.
  C2_PERSIST_AT_T50   = 0.05  -- GOAL_STRUCTURED z_goal_norm at t=50 must exceed.
  C3_ABLATED_MAX      = 0.05  -- GOAL_ABLATED z_goal_norm (all offsets) must stay below.
  DELTA_MIN_AT_CONTACT= 0.05  -- (structured t=0) - (ablated t=0) must exceed.
  MIN_SEEDS_PASS      = 2 of 3 seeds must satisfy C1+C2 (structured) and C3 (ablated).
  PASS = (C1 AND C2 AND C3 AND pairwise delta) across >= MIN_SEEDS_PASS seeds.

Claims: MECH-230 (single claim; evidence_direction set, not per-claim).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent

from experiment_protocol import emit_outcome

# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE    = "v3_exq_632_mech230_zgoal_structured_latent_discriminative"
CLAIM_IDS          = ["MECH-230"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Discriminative-pair definition
# ---------------------------------------------------------------------------
SEEDS      = [42, 7, 13]                       # matched_shared_seeds (>= 2)
CONDITIONS = ["GOAL_STRUCTURED", "GOAL_ABLATED"]  # mechanism ON vs OFF (exactly 2)

# ---------------------------------------------------------------------------
# Training / measurement schedule
# ---------------------------------------------------------------------------
P0_EPISODES  = 150    # encoder warmup (no z_goal seeding)
P1_EPISODES  = 100    # frozen-encoder measurement (z_goal seeding active when ON)
STEPS_PER_EP = 200
POST_CONTACT_STEPS = [0, 10, 25, 50]
# A contact trajectory is only started if at least this many steps remain so the
# t=50 window can complete within the episode.
MIN_WINDOW_STEPS = 55

GRID_SIZE     = 8
NUM_RESOURCES = 3
NUM_HAZARDS   = 2
HAZARD_HARM   = 0.1
DRIVE_WEIGHT  = 2.0

LR = 3e-4

# ---------------------------------------------------------------------------
# PRE-REGISTERED pass/fail thresholds (fixed BEFORE any run logic).
# Not derived from any run's own statistics.
# ---------------------------------------------------------------------------
C1_SEED_AT_CONTACT   = 0.10   # GOAL_STRUCTURED z_goal_norm at t=0 must exceed this
C2_PERSIST_AT_T50    = 0.05   # GOAL_STRUCTURED z_goal_norm at t=50 must exceed this
C3_ABLATED_MAX       = 0.05   # GOAL_ABLATED z_goal_norm must stay below this
DELTA_MIN_AT_CONTACT = 0.05   # (structured t=0) - (ablated t=0) must exceed this
MIN_SEEDS_PASS       = 2

REGISTERED_THRESHOLDS = {
    "C1_seed_at_contact_t0_gt": C1_SEED_AT_CONTACT,
    "C2_persist_at_t50_gt": C2_PERSIST_AT_T50,
    "C3_ablated_norm_max_lt": C3_ABLATED_MAX,
    "delta_at_contact_gt": DELTA_MIN_AT_CONTACT,
    "min_seeds_pass": MIN_SEEDS_PASS,
    "n_seeds": len(SEEDS),
}

# Dry-run smoke schedule (still exercises init / stepping / training / write).
DRY_RUN_P0_EPISODES = 2
DRY_RUN_P1_EPISODES = 2
DRY_RUN_STEPS       = 120   # > MIN_WINDOW_STEPS + buffer so contacts can complete


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
        proximity_benefit_scale=0.05,
        proximity_harm_scale=0.05,
        proximity_approach_threshold=0.15,
        use_proxy_fields=True,
    )


def _make_agent(env: CausalGridWorldV2, condition: str, seed: int) -> REEAgent:
    torch.manual_seed(seed)
    structured = (condition == "GOAL_STRUCTURED")
    # GOAL_STRUCTURED: z_goal mechanism ON (z_goal_enabled=True + SD-012 drive).
    # GOAL_ABLATED:    z_goal mechanism OFF (z_goal_enabled=False -> goal_state
    #                  is None -> update_z_goal early-returns -> no attractor).
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        use_event_classifier=True,
        z_goal_enabled=structured,                 # the ablated mechanism
        drive_weight=DRIVE_WEIGHT if structured else 0.0,
        goal_weight=0.0,                           # not testing navigation here
        benefit_threshold=0.1,
        benefit_eval_enabled=False,
        e1_goal_conditioned=structured,
    )
    return REEAgent(config)


# ---------------------------------------------------------------------------
# Single seed x condition run
# ---------------------------------------------------------------------------

def run_condition(seed: int, condition: str, dry_run: bool = False) -> Dict:
    total_p0  = DRY_RUN_P0_EPISODES if dry_run else P0_EPISODES
    total_p1  = DRY_RUN_P1_EPISODES if dry_run else P1_EPISODES
    steps_per = DRY_RUN_STEPS       if dry_run else STEPS_PER_EP
    total_eps = total_p0 + total_p1   # full training episodes for this run

    print(f"Seed {seed} Condition {condition}", flush=True)

    env   = _make_env(seed)
    agent = _make_agent(env, condition, seed)
    device = agent.device

    optimizer = optim.Adam(list(agent.parameters()), lr=LR)

    contact_trajectories: List[Dict] = []  # each: {offset: z_goal_norm}
    ablated_norms: List[float] = []        # GOAL_ABLATED running z_goal_norm

    prev_ttype = "none"
    encoder_frozen = False

    for ep in range(total_eps):
        _, obs_dict = env.reset()
        agent.reset()

        phase   = "P0" if ep < total_p0 else "P1"
        in_meas = (phase == "P1")

        # On entry to the measurement phase, FREEZE the encoder + heads (P1):
        # no further gradient updates; z_goal seeding fires on frozen latents.
        if in_meas and not encoder_frozen:
            for p in agent.parameters():
                p.requires_grad_(False)
            encoder_frozen = True

        post_contact_buffer: Optional[Dict] = None
        contact_step_in_ep: Optional[int] = None

        for step in range(steps_per):
            obs_body  = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)

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
            flat_next, harm_signal, done, info, obs_dict_next = env.step(action_idx)
            ttype = info.get("transition_type", "none")

            agent.update_residue(float(harm_signal))

            # z_goal seeding (measurement phase only). Uses the raw post-step
            # harm_signal as the benefit magnitude at contact (the pre-step
            # obs_body[11] EMA is ~0 at contact and would never cross threshold;
            # this was the documented EXQ-328 measurement bug).
            if in_meas and ttype == "resource":
                steps_remaining = steps_per - step - 1
                drive_level = REEAgent.compute_drive_level(obs_body)
                # In GOAL_ABLATED, goal_state is None and update_z_goal early-
                # returns (no attractor ever forms). In GOAL_STRUCTURED it seeds.
                agent.update_z_goal(float(harm_signal), drive_level)
                if steps_remaining >= MIN_WINDOW_STEPS:
                    norm0 = (
                        agent.goal_state.goal_norm()
                        if agent.goal_state is not None else 0.0
                    )
                    post_contact_buffer = {0: norm0}
                    contact_step_in_ep = step
                else:
                    post_contact_buffer = None
                    contact_step_in_ep = None

            # Record post-contact z_goal_norm at the registered offsets.
            if in_meas and post_contact_buffer is not None and contact_step_in_ep is not None:
                offset = step - contact_step_in_ep
                if offset in POST_CONTACT_STEPS and offset not in post_contact_buffer:
                    norm_val = (
                        agent.goal_state.goal_norm()
                        if agent.goal_state is not None else 0.0
                    )
                    post_contact_buffer[offset] = norm_val
                if max(POST_CONTACT_STEPS) in post_contact_buffer:
                    contact_trajectories.append(dict(post_contact_buffer))
                    post_contact_buffer = None
                    contact_step_in_ep = None

            # GOAL_ABLATED: track z_goal_norm throughout (should stay ~0).
            if in_meas and condition == "GOAL_ABLATED":
                norm_val = (
                    agent.goal_state.goal_norm()
                    if agent.goal_state is not None else 0.0
                )
                ablated_norms.append(norm_val)

            # P0 ONLY: train encoder + E1/E2 + aux heads (frozen during P1).
            if not in_meas:
                optimizer.zero_grad()
                e1_loss = agent.compute_prediction_loss()
                e2_loss = agent.compute_e2_loss()
                loss = e1_loss + e2_loss

                rfv = obs_dict.get("resource_field_view", None)
                if rfv is not None:
                    rp_t = float(rfv.max().item())
                    rp_loss = agent.compute_resource_proximity_loss(rp_t, latent)
                    loss = loss + rp_loss

                latent2 = agent.sense(obs_body, obs_world)
                ec_loss = agent.compute_event_contrastive_loss(prev_ttype, latent2)
                loss = loss + ec_loss

                if loss.requires_grad:
                    loss.backward()
                    nn.utils.clip_grad_norm_(list(agent.parameters()), 1.0)
                    optimizer.step()

            prev_ttype = ttype
            obs_dict = obs_dict_next

            if done:
                break

        if (ep + 1) % 50 == 0 or (ep + 1) == total_eps:
            print(
                f"  [train] seed={seed} {condition} ep {ep+1}/{total_eps} "
                f"phase={phase} contacts={len(contact_trajectories)}",
                flush=True,
            )

    # Aggregate post-contact z_goal_norm by offset.
    norms_by_offset: Dict[int, List[float]] = defaultdict(list)
    for traj in contact_trajectories:
        for offset, norm in traj.items():
            norms_by_offset[int(offset)].append(norm)

    mean_norm_by_offset = {
        str(t): float(np.mean(norms_by_offset.get(t, [0.0])))
        for t in POST_CONTACT_STEPS
    }
    mean_ablated_norm = float(np.mean(ablated_norms)) if ablated_norms else 0.0

    c1_val = mean_norm_by_offset.get("0", 0.0)
    c2_val = mean_norm_by_offset.get("50", 0.0)

    # Per-run verdict line (exactly one per seed x condition run).
    if condition == "GOAL_STRUCTURED":
        run_pass = (c1_val > C1_SEED_AT_CONTACT) and (c2_val > C2_PERSIST_AT_T50)
    else:
        run_pass = (mean_ablated_norm < C3_ABLATED_MAX)
    print(
        f"  verdict: {'PASS' if run_pass else 'FAIL'} "
        f"(t0={c1_val:.4f} t50={c2_val:.4f} ablated={mean_ablated_norm:.4f} "
        f"n_contacts={len(contact_trajectories)})",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": condition,
        "mean_norm_by_offset": mean_norm_by_offset,
        "mean_ablated_norm": mean_ablated_norm,
        "num_contact_events": len(contact_trajectories),
        "c1_norm_at_contact": c1_val,
        "c2_norm_at_t50": c2_val,
        "run_pass": bool(run_pass),
    }


# ---------------------------------------------------------------------------
# Criteria (pairwise; vs registered thresholds)
# ---------------------------------------------------------------------------

def evaluate_criteria(all_results: List[Dict]) -> Dict:
    by_cond: Dict[str, Dict[int, Dict]] = {c: {} for c in CONDITIONS}
    for r in all_results:
        by_cond[r["condition"]][r["seed"]] = r

    structured = by_cond["GOAL_STRUCTURED"]
    ablated    = by_cond["GOAL_ABLATED"]

    c1_vals = [structured[s]["c1_norm_at_contact"] for s in SEEDS if s in structured]
    c2_vals = [structured[s]["c2_norm_at_t50"]    for s in SEEDS if s in structured]
    c3_vals = [ablated[s]["mean_ablated_norm"]    for s in SEEDS if s in ablated]

    # Pairwise delta per shared seed: structured(t0) - ablated(t0). Ablated t0
    # is 0 by construction, but we read it explicitly for honest pairing.
    delta_vals = []
    for s in SEEDS:
        if s in structured and s in ablated:
            delta_vals.append(
                structured[s]["c1_norm_at_contact"] - ablated[s]["c1_norm_at_contact"]
            )

    c1_seeds = sum(v > C1_SEED_AT_CONTACT for v in c1_vals)
    c2_seeds = sum(v > C2_PERSIST_AT_T50 for v in c2_vals)
    c3_seeds = sum(v < C3_ABLATED_MAX for v in c3_vals)
    delta_seeds = sum(v > DELTA_MIN_AT_CONTACT for v in delta_vals)

    c1_pass = c1_seeds >= MIN_SEEDS_PASS
    c2_pass = c2_seeds >= MIN_SEEDS_PASS
    c3_pass = c3_seeds >= MIN_SEEDS_PASS
    delta_pass = delta_seeds >= MIN_SEEDS_PASS

    overall_pass = c1_pass and c2_pass and c3_pass and delta_pass

    return {
        "c1_structured_seeded_at_contact": c1_pass,
        "c1_vals": c1_vals,
        "c1_seeds_pass": c1_seeds,
        "c2_structured_persists": c2_pass,
        "c2_vals": c2_vals,
        "c2_seeds_pass": c2_seeds,
        "c3_ablated_absent": c3_pass,
        "c3_vals": c3_vals,
        "c3_seeds_pass": c3_seeds,
        "delta_at_contact_pass": delta_pass,
        "delta_vals": delta_vals,
        "delta_seeds_pass": delta_seeds,
        "overall_pass": overall_pass,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = False):
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    suffix = "_dry" if dry_run else ""
    run_id = f"{EXPERIMENT_TYPE}{suffix}_{ts}_v3"
    print(f"EXQ-632 start: {run_id}", flush=True)

    all_results: List[Dict] = []
    # Matched shared seeds: outer loop seed, inner loop condition, so both
    # conditions of a given seed share env + torch seeding.
    for seed in SEEDS:
        for condition in CONDITIONS:
            all_results.append(run_condition(seed, condition, dry_run=dry_run))

    criteria = evaluate_criteria(all_results)
    overall_pass = criteria["overall_pass"]
    outcome  = "PASS" if overall_pass else "FAIL"
    direction = "supports" if overall_pass else "does_not_support"

    print(f"\n=== EXQ-632 {outcome} ===", flush=True)
    print(
        f"C1 structured_seeded(t0>{C1_SEED_AT_CONTACT}): "
        f"{criteria['c1_structured_seeded_at_contact']} "
        f"vals={[f'{v:.4f}' for v in criteria['c1_vals']]}",
        flush=True,
    )
    print(
        f"C2 structured_persists(t50>{C2_PERSIST_AT_T50}): "
        f"{criteria['c2_structured_persists']} "
        f"vals={[f'{v:.4f}' for v in criteria['c2_vals']]}",
        flush=True,
    )
    print(
        f"C3 ablated_absent(<{C3_ABLATED_MAX}): "
        f"{criteria['c3_ablated_absent']} "
        f"vals={[f'{v:.4f}' for v in criteria['c3_vals']]}",
        flush=True,
    )
    print(
        f"DELTA at contact(>{DELTA_MIN_AT_CONTACT}): "
        f"{criteria['delta_at_contact_pass']} "
        f"vals={[f'{v:.4f}' for v in criteria['delta_vals']]}",
        flush=True,
    )

    summary = (
        "MECH-230 z_goal structured-latent discriminative pair on CausalGridWorldV2. "
        "Scenario: GOAL_STRUCTURED (z_goal_enabled=True, drive_weight=2.0) vs "
        "GOAL_ABLATED (z_goal_enabled=False -> no goal_state). Phased training: "
        "P0 encoder/E1/E2/resource-proximity warmup (no z_goal seeding), then "
        "frozen-encoder P1/P2 measurement of z_goal_norm post resource contact. "
        f"Pairwise deltas (structured t0 - ablated t0) per shared seed: "
        f"{[round(v, 4) for v in criteria['delta_vals']]}. "
        f"Interpretation: PASS means a structured z_goal attractor forms and "
        f"persists with the mechanism ON (C1 t0>{C1_SEED_AT_CONTACT}, "
        f"C2 t50>{C2_PERSIST_AT_T50}) and is absent with the mechanism OFF "
        f"(C3 ablated<{C3_ABLATED_MAX}), with a per-seed pairwise separation "
        f">{DELTA_MIN_AT_CONTACT} on >= {MIN_SEEDS_PASS}/{len(SEEDS)} seeds -- "
        "supporting MECH-230 (z_goal develops measurable structure distinct "
        "from harm avoidance). FAIL means the structured attractor did not form "
        "or did not separate from the ablation, which does_not_support MECH-230 "
        "(likely the E1-frontal->hippocampal goal-projection substrate gap noted "
        "in the claim's evidence_quality_note). "
        f"OUTCOME: {outcome}."
    )

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_class": "discriminative_pair",
        "evidence_direction": direction,
        "outcome": outcome,
        "summary": summary,
        "registered_thresholds": REGISTERED_THRESHOLDS,
        "metrics": {
            "c1_structured_seeded_at_contact": bool(criteria["c1_structured_seeded_at_contact"]),
            "c1_seeds_pass": int(criteria["c1_seeds_pass"]),
            "c1_vals": [float(v) for v in criteria["c1_vals"]],
            "c2_structured_persists": bool(criteria["c2_structured_persists"]),
            "c2_seeds_pass": int(criteria["c2_seeds_pass"]),
            "c2_vals": [float(v) for v in criteria["c2_vals"]],
            "c3_ablated_absent": bool(criteria["c3_ablated_absent"]),
            "c3_seeds_pass": int(criteria["c3_seeds_pass"]),
            "c3_vals": [float(v) for v in criteria["c3_vals"]],
            "delta_at_contact_pass": bool(criteria["delta_at_contact_pass"]),
            "delta_seeds_pass": int(criteria["delta_seeds_pass"]),
            "delta_vals": [float(v) for v in criteria["delta_vals"]],
            "overall_pass": bool(overall_pass),
        },
        "config": {
            "seeds": SEEDS,
            "conditions": CONDITIONS,
            "seed_policy": "matched_shared_seeds",
            "p0_episodes": DRY_RUN_P0_EPISODES if dry_run else P0_EPISODES,
            "p1_episodes": DRY_RUN_P1_EPISODES if dry_run else P1_EPISODES,
            "steps_per_ep": DRY_RUN_STEPS if dry_run else STEPS_PER_EP,
            "drive_weight": DRIVE_WEIGHT,
            "post_contact_steps": POST_CONTACT_STEPS,
            "min_window_steps": MIN_WINDOW_STEPS,
            "dry_run": bool(dry_run),
        },
        "results_per_condition": all_results,
        "timestamp_utc": ts,
        "interpretation_grid": {
            "C1+C2+C3+delta_all_pass": "supports MECH-230 (structured z_goal forms, persists, absent under ablation)",
            "C1/C2_fail_structured": "does_not_support -- structured attractor did not form/persist (substrate gap: E1-frontal->hippocampal goal projection per claim note)",
            "C3_fail_ablated": "ablation leaked a non-zero z_goal -- check ablation wiring; likely measurement artifact",
            "delta_fail_only": "structured and ablated did not separate per-seed -- pairwise discrimination insufficient",
        },
    }

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "REE_assembly", "evidence", "experiments",
        EXPERIMENT_TYPE,
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Results -> {out_path}", flush=True)

    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _outcome, _out_path = main(dry_run=args.dry_run)

    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
    )
    sys.exit(0)
