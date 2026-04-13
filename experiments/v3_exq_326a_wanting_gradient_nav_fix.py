#!/opt/local/bin/python3
"""
V3-EXQ-326a: Wanting-Gradient Navigation -- SD-015 wiring fix + harm_signal seeding
Supersedes V3-EXQ-326.

experiment_purpose: evidence

EXQ-326 FAILED (C3 benefit_ratio never reached 1.3x). Two root causes identified:

  Root cause 1 (SD-015 wiring never applied):
    EXQ-326 never set cfg.latent.use_resource_encoder = True. The config was built
    via REEConfig.from_dims() with no 'latent' keyword argument at all -- the
    ResourceEncoder was never instantiated. update_z_goal() fell back to z_world
    (the fallback path), not z_resource. This means SD-015 was never actually tested.
    Fix: after REEConfig.from_dims(), set directly:
      cfg.latent.use_resource_encoder = True
      cfg.latent.z_resource_dim = 32
    (same fix as EXQ-322a and EXQ-354)

  Root cause 2 (pre-step EMA seeding bug):
    EXQ-326 used benefit_raw = obs_body[11] (the benefit_exposure EMA from BEFORE
    env.step()). At the moment of resource contact, the EMA reflects state prior to
    contact -- it is near 0. So effective_benefit << benefit_threshold and
    update_z_goal() rarely fired. z_goal was never properly seeded.
    Fix: use harm_signal from env.step() on resource contact steps (ttype=="resource").
    harm_signal at resource contact is resource_benefit + proximity_benefit (~0.5+),
    which reliably exceeds benefit_threshold=0.15.

EXQ-354 proved the SD-015 substrate works (PASS 4/4 seeds on MECH-112 latent test).
Navigation integration with wanting_weight works when z_goal is correctly seeded.
EXQ-326a just applies both fixes.

CLAIM RE-EVALUATION:
  EXQ-326 listed SD-015, MECH-216, SD-012. Re-evaluated for EXQ-326a:
  - SD-015: PRIMARY. This experiment specifically tests whether the z_resource encoder
    (SD-015) enables goal-directed navigation (wanting_weight in HippocampalModule).
    C3 benefit_ratio is the primary SD-015 evidence criterion.
  - MECH-229: PRIMARY. EXQ-326a tests behavioral wanting/liking dissociation in the
    navigation context. MECH-229 = behavioral wanting/liking dissociation (distinct from
    MECH-230 = latent structure). MECH-112 was split into MECH-229 + MECH-230 on 2026-04-13.
  - SD-012: NOT a primary claim here. SD-012 is tested in EXQ-328b. In EXQ-326a, drive
    modulation is used as a substrate (drive_weight=2.0 in WITH_WANTING) but we are not
    ablating drive specifically -- we are ablating the wanting pathway (wanting_weight=0).

Two conditions per seed:
  WITH_WANTING: wanting_weight=0.5 (HippocampalModule goal nav), goal + schema active,
                use_resource_encoder=True (SD-015 fix applied)
  ABLATED:      wanting_weight=0.0, goal disabled, schema disabled (no wanting seeding)

Both conditions use phased training: P0 = encoder warmup, P1 = full pipeline,
P2 = evaluation.

Pass criteria:
  C1: VALENCE_WANTING mean in WITH_WANTING > 0.01 (wanting populated)
  C2: resource_rate_lift = WITH_WANTING/ABLATED >= 1.3x (>= 2/3 seeds)
  C3: benefit_ratio WITH_WANTING >= 1.3x (paper gate primary, >= 2/3 seeds)

PASS: C1 AND C3 across >= 2/3 seeds.

Verification checklist diagnostics:
  goal_norm_peak       -- if 0.0, z_goal was never seeded (bootstrap deadlock)
  n_seeding_events     -- steps where ttype=="resource" and harm_signal fired seeding
  schema_salience_mean -- if near 0, E1 schema readout not firing (MECH-216 path dead)
  mech216_write_count  -- VALENCE_WANTING writes from schema pathway (0 = wanting terrain empty)

Claims: SD-015, MECH-229
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
import random
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.optim as optim
import numpy as np

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from ree_core.residue.field import VALENCE_WANTING

# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE    = "v3_exq_326a_wanting_gradient_nav_fix"
CLAIM_IDS          = ["SD-015", "MECH-229"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEEDS       = [42, 7, 13]
CONDITIONS  = ["WITH_WANTING", "ABLATED"]

P0_EPISODES = 100    # warmup: encoder + proximity head trains, no wanting seeding
P1_EPISODES = 100    # full pipeline: wanting + goal active in WITH_WANTING
P2_EPISODES = 50     # evaluation only (no training updates)
STEPS_PER_EP = 200

GRID_SIZE     = 8
NUM_RESOURCES = 3
NUM_HAZARDS   = 2
HAZARD_HARM   = 0.1   # mild -- allows exploration without frequent termination

WANTING_WEIGHT = 0.5   # HippocampalModule goal_weight / wanting_weight
GOAL_WEIGHT    = 1.0   # E3Config.goal_weight (in score_trajectory)
DRIVE_WEIGHT   = 2.0   # SD-012 benefit amplification (substrate, not ablated here)

# SD-015 fix: benefit_threshold > proximity_benefit but below amplified contact benefit
# At resource contact: harm_signal ~0.5; effective_benefit = 0.5*(1+2.0*drive_level)
# For WITH_WANTING at drive_level=0.3: effective = 0.5*(1+0.6) = 0.8 > 0.15 -> fires
BENEFIT_THRESHOLD = 0.15

LR = 3e-4

# Pass thresholds
C1_WANTING_THRESHOLD = 0.01   # VALENCE_WANTING must be > this in WITH_WANTING
C3_BENEFIT_RATIO     = 1.3    # WITH_WANTING benefit >= 1.3x ABLATED benefit
MIN_SEEDS_PASS       = 2      # >= 2 of 3 seeds must satisfy each criterion

# Dry-run scale
DRY_RUN_EPISODES = 3
DRY_RUN_STEPS    = 30


# ---------------------------------------------------------------------------
# Factory helpers
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
    with_wanting = condition == "WITH_WANTING"
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        # SD-018: resource proximity head (required for goal seeding to work)
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        # SD-009: event contrastive (world encoder quality)
        use_event_classifier=True,
        # Goal / drive (substrate for wanting, SD-012)
        z_goal_enabled=with_wanting,
        drive_weight=DRIVE_WEIGHT if with_wanting else 0.0,
        goal_weight=GOAL_WEIGHT if with_wanting else 0.0,
        benefit_threshold=BENEFIT_THRESHOLD,
        benefit_eval_enabled=with_wanting,
        benefit_weight=1.0,
        e1_goal_conditioned=with_wanting,
        # MECH-216: E1 schema wanting
        schema_wanting_enabled=with_wanting,
        schema_wanting_threshold=0.3,
        schema_wanting_gain=0.5,
        # HippocampalModule wanting gradient
        wanting_weight=WANTING_WEIGHT if with_wanting else 0.0,
    )
    if with_wanting:
        # FIX (EXQ-326a): SD-015 wiring fix.
        # from_dims() has no 'latent' kwarg -- must set directly on the returned config object.
        # This is the same pattern used in EXQ-322a and EXQ-354.
        cfg.latent.use_resource_encoder = True
        cfg.latent.z_resource_dim = 32  # must match goal_dim (= world_dim = 32)
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# Single condition run
# ---------------------------------------------------------------------------

def run_condition(
    seed: int,
    condition: str,
    dry_run: bool = False,
) -> Dict:
    total_p0 = DRY_RUN_EPISODES if dry_run else P0_EPISODES
    total_p1 = DRY_RUN_EPISODES if dry_run else P1_EPISODES
    total_p2 = DRY_RUN_EPISODES if dry_run else P2_EPISODES
    steps_per = DRY_RUN_STEPS if dry_run else STEPS_PER_EP
    total_eps = total_p0 + total_p1 + total_p2

    print(f"  Seed {seed} Condition {condition}")

    env   = _make_env(seed)
    agent = _make_agent(env, condition, seed)
    device = agent.device

    optimizer = optim.Adam(list(agent.parameters()), lr=LR)

    p2_resources: List[float] = []
    p2_benefit:   List[float] = []
    wanting_visited: List[float] = []
    schema_salience_samples: List[float] = []
    n_seeding_events: int = 0
    mech216_write_count: int = 0
    prev_ttype: str = "none"

    for ep in range(total_eps):
        _, obs_dict = env.reset()
        agent.reset()

        phase = "P0" if ep < total_p0 else ("P1" if ep < total_p0 + total_p1 else "P2")
        in_eval = (phase == "P2")

        ep_resources = 0
        ep_benefit   = 0.0

        for step in range(steps_per):
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
            sal = float(getattr(agent, '_schema_salience', 0.0) or 0.0)

            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action     = agent.select_action(candidates, ticks)

            drive_level = REEAgent.compute_drive_level(obs_body)

            if condition == "WITH_WANTING":
                schema_salience_samples.append(sal)

            action_idx = int(action.argmax(dim=-1).item())
            flat_next, harm_signal, done, info, obs_dict_next = env.step(action_idx)
            ttype = info.get("transition_type", "none")

            if ttype == "resource":
                ep_resources += 1
            ep_benefit += max(0.0, float(info.get("benefit_exposure", 0.0)))

            # Residue + E3 update
            agent.update_residue(float(harm_signal))

            # FIX (EXQ-326a): use harm_signal from env.step() for goal seeding.
            # Only seed on resource contact steps; harm_signal ~0.5 at contact.
            # The old code used obs_body[11] (pre-step EMA) which was ~0 at contact time.
            if condition == "WITH_WANTING" and phase != "P0":
                if ttype == "resource":
                    agent.update_z_goal(float(harm_signal), drive_level)
                    n_seeding_events += 1

                agent.update_schema_wanting(drive_level)

                # Verification diagnostics
                if sal >= 0.3:  # schema_wanting_threshold
                    mech216_write_count += 1

                # C1: sample VALENCE_WANTING
                if latent.z_world is not None:
                    with torch.no_grad():
                        val = agent.residue_field.evaluate_valence(latent.z_world)
                        wanting_visited.append(float(val[0, VALENCE_WANTING].item()))

            if z_self_prev is not None:
                agent.record_transition(z_self_prev, action, latent.z_self.detach())

            if not in_eval:
                optimizer.zero_grad()
                e1_loss = agent.compute_prediction_loss()
                e2_loss = agent.compute_e2_loss()
                loss = e1_loss + e2_loss

                # SD-018: resource proximity supervision (both conditions in P0/P1)
                rfv = obs_dict.get("resource_field_view", None)
                if rfv is not None:
                    rp_target = float(rfv.max().item())
                    rp_loss = agent.compute_resource_proximity_loss(rp_target, latent)
                    loss = loss + rp_loss

                # SD-009: event contrastive
                latent_for_event = agent.sense(obs_body, obs_world)
                ec_loss = agent.compute_event_contrastive_loss(prev_ttype, latent_for_event)
                loss = loss + ec_loss

                # SD-015 ResourceEncoder loss (WITH_WANTING P0 only -- encoder warmup)
                if condition == "WITH_WANTING" and phase == "P0":
                    re_loss = agent.compute_resource_encoder_loss(
                        float(rfv.max().item()) if rfv is not None else 0.0,
                        latent,
                    )
                    if re_loss is not None:
                        loss = loss + re_loss

                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(agent.parameters()), 1.0)
                    optimizer.step()

            prev_ttype = ttype
            obs_dict = obs_dict_next

            if done:
                break

        if in_eval:
            p2_resources.append(float(ep_resources))
            p2_benefit.append(ep_benefit / max(1, steps_per))

        if (ep + 1) % 50 == 0:
            print(
                f"    [train] seed={seed} {condition} ep {ep+1}/{total_eps} "
                f"phase={phase} resources={ep_resources} benefit={ep_benefit:.3f}",
                flush=True,
            )

    resource_rate        = float(np.mean(p2_resources)) if p2_resources else 0.0
    mean_benefit         = float(np.mean(p2_benefit))   if p2_benefit   else 0.0
    wanting_mean         = float(np.mean(wanting_visited)) if wanting_visited else 0.0
    schema_salience_mean = float(np.mean(schema_salience_samples)) if schema_salience_samples else 0.0
    goal_norm_peak       = float(getattr(getattr(agent, 'goal_state', None), '_goal_norm_peak', 0.0))

    print(f"  verdict: resource_rate={resource_rate:.3f} benefit={mean_benefit:.4f} "
          f"wanting_mean={wanting_mean:.4f} goal_norm_peak={goal_norm_peak:.4f} "
          f"n_seeding={n_seeding_events} sal_mean={schema_salience_mean:.4f} "
          f"mech216_writes={mech216_write_count}")
    return {
        "seed": seed,
        "condition": condition,
        "resource_rate": resource_rate,
        "mean_benefit_exposure": mean_benefit,
        "valence_wanting_mean": wanting_mean,
        # Verification checklist diagnostics
        "goal_norm_peak": goal_norm_peak,
        "n_seeding_events": n_seeding_events,
        "schema_salience_mean": schema_salience_mean,
        "mech216_write_count": mech216_write_count,
    }


# ---------------------------------------------------------------------------
# Criteria evaluation
# ---------------------------------------------------------------------------

def evaluate_criteria(all_results: List[Dict]) -> Dict:
    by_cond: Dict[str, List[Dict]] = defaultdict(list)
    for r in all_results:
        by_cond[r["condition"]].append(r)

    with_list    = sorted(by_cond.get("WITH_WANTING", []), key=lambda x: x["seed"])
    ablated_list = sorted(by_cond.get("ABLATED",       []), key=lambda x: x["seed"])

    # C1: VALENCE_WANTING populated in WITH_WANTING
    c1_vals = [r["valence_wanting_mean"] for r in with_list]
    c1_pass = all(v > C1_WANTING_THRESHOLD for v in c1_vals)

    # C3: benefit_ratio >= 1.3x (paper gate)
    c3_seeds_pass = 0
    c3_ratios = []
    for w, a in zip(with_list, ablated_list):
        base = max(a["mean_benefit_exposure"], 1e-6)
        ratio = w["mean_benefit_exposure"] / base
        c3_ratios.append(ratio)
        if ratio >= C3_BENEFIT_RATIO:
            c3_seeds_pass += 1
    c3_pass = c3_seeds_pass >= MIN_SEEDS_PASS

    overall_pass = c1_pass and c3_pass
    return {
        "c1_wanting_populated": c1_pass,
        "c1_vals": c1_vals,
        "c3_benefit_ratio_pass": c3_pass,
        "c3_ratios": c3_ratios,
        "c3_seeds_pass": c3_seeds_pass,
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
        f"v3_exq_326a_wanting_gradient_nav_fix_dry_{ts}_v3"
        if args.dry_run
        else f"v3_exq_326a_wanting_gradient_nav_fix_{ts}_v3"
    )
    print(f"EXQ-326a start: {run_id}")

    all_results: List[Dict] = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            result = run_condition(seed, condition, dry_run=args.dry_run)
            all_results.append(result)

    criteria = evaluate_criteria(all_results)
    outcome  = "PASS" if criteria["overall_pass"] else "FAIL"

    print(f"\n=== EXQ-326a {outcome} ===")
    print(f"C1 wanting_populated: {criteria['c1_wanting_populated']} "
          f"(vals={[f'{v:.4f}' for v in criteria['c1_vals']]})")
    print(f"C3 benefit_ratio: {criteria['c3_benefit_ratio_pass']} "
          f"(ratios={[f'{v:.3f}' for v in criteria['c3_ratios']]})")

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes": "V3-EXQ-326",
        "evidence_direction_per_claim": {
            "SD-015": "supports" if criteria["overall_pass"] else "does_not_support",
            "MECH-229": "supports" if criteria["c1_wanting_populated"] else "does_not_support",
        },
        "evidence_direction": "supports" if criteria["overall_pass"] else "does_not_support",
        "outcome": outcome,
        "criteria": criteria,
        "results_per_condition": all_results,
        "config": {
            "seeds": SEEDS,
            "conditions": CONDITIONS,
            "wanting_weight": WANTING_WEIGHT,
            "p0_episodes": P0_EPISODES,
            "p1_episodes": P1_EPISODES,
            "p2_episodes": P2_EPISODES,
            "steps_per_ep": STEPS_PER_EP,
            "grid_size": GRID_SIZE,
            "num_hazards": NUM_HAZARDS,
            "hazard_harm": HAZARD_HARM,
        },
        "timestamp_utc": ts,
        "bug_fix_note": (
            "EXQ-326 fix 1 (SD-015 wiring): cfg.latent.use_resource_encoder=True was never set. "
            "from_dims() has no 'latent' kwarg -- the ResourceEncoder was never instantiated. "
            "update_z_goal() used z_world fallback instead of z_resource. Fix: set directly on "
            "config object after from_dims() (same pattern as EXQ-322a, EXQ-354). "
            "EXQ-326 fix 2 (seeding signal): used obs_body[11] (pre-step benefit EMA, ~0 at "
            "contact time). Fix: use harm_signal from env.step() on ttype==resource steps. "
            "claim_ids updated: MECH-112 split into MECH-229/MECH-230 on 2026-04-13. "
            "This experiment tests behavioral dissociation (MECH-229) + SD-015 nav integration. "
            "SD-012 removed as primary claim (tested in EXQ-328b)."
        ),
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
    print(f"Results -> {out_path}")


if __name__ == "__main__":
    main()
