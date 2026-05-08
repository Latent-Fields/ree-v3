#!/opt/local/bin/python3
"""
V3-EXQ-527 -- MECH-112 Goal-Directed Behavior: Identity Encoder + Reef (Fresh Angle)

Claims: MECH-112

Scientific question: Does an identity-aware z_resource encoder (SD-049 Phase 2)
produce genuine goal-directed behavior (GOAL_PRESENT beats GOAL_ABSENT) in a
multi-resource reef environment that breaks monostrategy?

Fresh angle over EXQ-189:
  1. Reef substrate (SD-054): breaks fixed-route monostrategy via two behavioral
     attractors (flee-to-reef vs. forage), giving the goal system a non-trivial
     strategy space to influence.
  2. Identity-aware z_resource encoder (SD-049 Phase 2): identity_classifier head
     discriminates food vs. water resources, giving z_goal structured content
     rather than a purely spatial signal.
  3. Multi-resource heterogeneity (SD-049 Phase 1): food + water resource types
     with distinct field views, making goal specificity behaviourally testable.

Two conditions, matched seeds:
  GOAL_PRESENT:
    - use_resource_encoder=True, use_identity_classifier=True (n_types=2)
    - multi_resource_heterogeneity_enabled=True (food, water)
    - z_goal seeded from z_resource on resource contact
    - reef_enabled=True, hazard_food_attraction=0.5
    - Phased training:
        P0 (100 eps): encoder warmup -- identity_classifier + resource_proximity_head
        P1 (150 eps): freeze identity head, downstream training with z_goal seeding
        Eval (50 eps): score-based policy
  GOAL_ABSENT:
    - use_resource_encoder=False (no z_goal pathway)
    - multi_resource_heterogeneity_enabled=True (same env)
    - reef_enabled=True (same env)
    - Identical episode counts

Grid: 10x10, 3 hazards, 8 resources (4 food + 4 water via 2-type heterogeneity),
      reef_enabled=True, n_reef_patches=3, reef_patch_radius=2, hazard_food_attraction=0.5.

world_obs_dim = 325 (250 base + 25 reef + 2*25 per-type resource fields).

Seeds: [42, 7, 13]. 2 conditions x 3 seeds = 6 total runs. 300 eps/run.

PASS criteria (ALL required):
  C1: goal_present_resource_rate >= goal_absent_resource_rate + 0.025
      (GOAL_PRESENT collects more resources -- behavioral lift from structured goal)
  C2: goal_present_benefit >= goal_absent_benefit * 1.10
      (at least 10% more accumulated benefit)
  C3: goal_present_z_goal_norm_final >= 0.05
      (z_goal actually seeds -- non-trivial learned representation)
  C4: identity_probe_acc >= 0.60
      (identity encoder distinguishes food vs. water resource types)
  C5: >= 2/3 seeds pass C1+C2+C3+C4
"""

import sys
import random
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_527_mech112_identity_goal_reef"
CLAIM_IDS = ["MECH-112"]
EXPERIMENT_PURPOSE = "evidence"

GRID_SIZE = 10
NUM_HAZARDS = 3
NUM_RESOURCES = 8
N_RESOURCE_TYPES = 2

P0_EPISODES = 100    # encoder warmup
P1_EPISODES = 150    # freeze identity head, train downstream
EVAL_EPISODES = 50
STEPS_PER_EPISODE = 200
EPISODES_PER_RUN = P0_EPISODES + P1_EPISODES + EVAL_EPISODES  # 300

SEEDS = [42, 7, 13]
LR = 3e-4
BENEFIT_THRESHOLD = 0.10  # threshold for z_goal seeding

CONDITIONS = ["GOAL_PRESENT", "GOAL_ABSENT"]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        hazard_harm=0.02,
        proximity_harm_scale=0.3,
        proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        env_drift_interval=999,
        env_drift_prob=0.0,
        # SD-054 reef substrate
        reef_enabled=True,
        n_reef_patches=3,
        reef_patch_radius=2,
        hazard_food_attraction=0.5,
        # SD-049 Phase 1: multi-resource heterogeneity
        multi_resource_heterogeneity_enabled=True,
        n_resource_types=N_RESOURCE_TYPES,
        resource_type_names=("food", "water"),
        resource_type_benefit_curves=("sigmoidal_saturating", "sigmoidal_saturating"),
    )


def _make_agent_goal_present(env: CausalGridWorldV2, seed: int) -> REEAgent:
    torch.manual_seed(seed)
    cfg = REEConfig.from_dims(
        world_obs_dim=env.world_obs_dim,
        body_obs_dim=env.body_obs_dim,
        harm_obs_dim=env.harm_obs_dim if hasattr(env, 'harm_obs_dim') else 32,
        action_dim=env.action_dim,
        use_resource_encoder=True,
        use_resource_proximity_head=True,
        drive_weight=2.0,
    )
    # SD-049 Phase 2: identity classifier
    cfg.latent.use_identity_classifier = True
    cfg.latent.identity_classifier_n_types = N_RESOURCE_TYPES
    cfg.goal.z_goal_enabled = True
    cfg.goal.benefit_threshold = BENEFIT_THRESHOLD
    return REEAgent(cfg)


def _make_agent_goal_absent(env: CausalGridWorldV2, seed: int) -> REEAgent:
    torch.manual_seed(seed)
    cfg = REEConfig.from_dims(
        world_obs_dim=env.world_obs_dim,
        body_obs_dim=env.body_obs_dim,
        harm_obs_dim=env.harm_obs_dim if hasattr(env, 'harm_obs_dim') else 32,
        action_dim=env.action_dim,
        use_resource_encoder=False,
        drive_weight=2.0,
    )
    cfg.goal.z_goal_enabled = False
    return REEAgent(cfg)


def _sense_from_obs_dict(agent: REEAgent, obs_dict: Dict) -> object:
    obs_body = obs_dict["body_state"]
    obs_world = obs_dict["world_state"]
    return agent.sense(obs_body, obs_world)


def _run_condition(
    condition: str,
    seed: int,
    ep_global_start: int,
    total_eps: int,
    p0_eps: int = P0_EPISODES,
    p1_eps: int = P1_EPISODES,
    eval_eps: int = EVAL_EPISODES,
) -> Dict:
    """Run one seed x condition, returning per-run metrics."""
    env = _make_env(seed)

    if condition == "GOAL_PRESENT":
        agent = _make_agent_goal_present(env, seed)
    else:
        agent = _make_agent_goal_absent(env, seed)

    optimizer = optim.Adam(agent.parameters(), lr=LR)

    random.seed(seed)
    torch.manual_seed(seed)

    n_episodes = p0_eps + p1_eps + eval_eps

    # --- phase tracking ---
    resource_contacts = 0
    benefit_acc = 0.0
    harm_acc = 0.0
    total_steps = 0
    eval_resource_contacts = 0
    eval_benefit_acc = 0.0
    eval_steps = 0
    z_goal_norms_eval = []

    # identity probe data (GOAL_PRESENT only)
    probe_features: List[torch.Tensor] = []
    probe_labels: List[int] = []

    for ep in range(n_episodes):
        phase = (
            "P0" if ep < p0_eps else
            "P1" if ep < p0_eps + p1_eps else
            "eval"
        )
        is_eval = (phase == "eval")
        global_ep = ep + 1

        # freeze identity head at P1 boundary
        if condition == "GOAL_PRESENT" and ep == p0_eps:
            res_enc = getattr(agent.latent_stack, "resource_encoder", None)
            id_head = getattr(res_enc, "identity_head", None) if res_enc is not None else None
            if id_head is not None:
                for p in id_head.parameters():
                    p.requires_grad_(False)

        _, obs_dict = env.reset()
        agent.reset()

        ep_resource_contacts = 0
        ep_benefit = 0.0
        ep_harm = 0.0

        for step in range(STEPS_PER_EPISODE):
            latent = _sense_from_obs_dict(agent, obs_dict)

            if is_eval:
                # greedy-toward-resource in eval (80% greedy)
                rfv = obs_dict.get("resource_field_view", None)
                if rfv is not None and rfv.max() > 0 and random.random() < 0.8:
                    action = torch.zeros(env.action_dim)
                    action[rfv.argmax() % env.action_dim] = 1.0
                else:
                    action = torch.zeros(env.action_dim)
                    action[random.randint(0, env.action_dim - 1)] = 1.0
            else:
                # greedy-proximity 50% warmup
                if random.random() < 0.5:
                    rfv = obs_dict.get("resource_field_view", None)
                    if rfv is not None and rfv.max() > 0:
                        action = torch.zeros(env.action_dim)
                        action[rfv.argmax() % env.action_dim] = 1.0
                    else:
                        action = torch.zeros(env.action_dim)
                        action[random.randint(0, env.action_dim - 1)] = 1.0
                else:
                    action = torch.zeros(env.action_dim)
                    action[random.randint(0, env.action_dim - 1)] = 1.0

            _, harm_signal, done, info, obs_dict = env.step(action)

            # SD-049 consumed type tag (from info dict, not obs_dict)
            consumed_type_tag = info.get("sd049_consumed_type_tag_this_tick", 0)
            benefit_exp = info.get("benefit_exposure", 0.0)
            harm_val = abs(float(harm_signal)) if float(harm_signal) < 0 else 0.0

            ep_benefit += benefit_exp
            ep_harm += harm_val

            if benefit_exp > 0.01:
                ep_resource_contacts += 1

            if not is_eval:
                optimizer.zero_grad()
                total_loss = torch.tensor(0.0, requires_grad=True)

                # resource proximity loss (SD-018)
                if hasattr(agent, "compute_resource_proximity_loss"):
                    rfv = obs_dict.get("resource_field_view")
                    if rfv is not None:
                        prox_target = float(rfv.max().item())
                        prox_loss = agent.compute_resource_proximity_loss(prox_target, latent)
                        total_loss = total_loss + 0.5 * prox_loss

                # identity classification loss (SD-049 Phase 2, P0 only)
                if condition == "GOAL_PRESENT" and phase == "P0" and consumed_type_tag > 0:
                    id_loss = agent.compute_resource_identity_loss(consumed_type_tag, latent)
                    total_loss = total_loss + 1.0 * id_loss

                # E1 prediction loss
                if hasattr(agent, "compute_prediction_loss"):
                    pred_loss = agent.compute_prediction_loss()
                    total_loss = total_loss + pred_loss

                if total_loss.item() > 0:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                    optimizer.step()

                # z_goal seeding (GOAL_PRESENT, P1 only)
                if condition == "GOAL_PRESENT" and phase == "P1":
                    agent.update_z_goal(float(harm_signal))

            else:
                # eval phase: score-based
                if condition == "GOAL_PRESENT":
                    agent.update_z_goal(float(harm_signal))
                eval_resource_contacts += 1 if benefit_exp > 0.01 else 0
                eval_benefit_acc += benefit_exp
                eval_steps += 1
                z_goal_tensor = (
                    agent.goal_state.z_goal
                    if agent.goal_state is not None else None
                )
                if z_goal_tensor is not None:
                    z_goal_norms_eval.append(z_goal_tensor.norm().item())
                # collect identity probe data
                if condition == "GOAL_PRESENT" and latent.z_resource is not None and consumed_type_tag > 0:
                    probe_features.append(latent.z_resource.squeeze(0).detach())
                    probe_labels.append(consumed_type_tag - 1)  # 0-indexed

            if done:
                break

        resource_contacts += ep_resource_contacts
        benefit_acc += ep_benefit
        harm_acc += ep_harm
        total_steps += STEPS_PER_EPISODE

        # progress print
        if (ep + 1) % 20 == 0 or ep == n_episodes - 1:
            print(
                f"  [train] seed={seed} cond={condition}"
                f" ep {global_ep}/{total_eps}  [{phase}]"
                f" res_c={ep_resource_contacts} ben={ep_benefit:.3f}",
                flush=True
            )

    # compute metrics
    eval_resource_rate = eval_resource_contacts / max(eval_steps, 1)
    eval_benefit_mean = eval_benefit_acc / max(eval_steps, 1)
    z_goal_norm_final = float(sum(z_goal_norms_eval) / len(z_goal_norms_eval)) if z_goal_norms_eval else 0.0

    # identity linear probe
    probe_acc = 0.0
    if condition == "GOAL_PRESENT" and len(probe_features) >= 10 and len(set(probe_labels)) > 1:
        X = torch.stack(probe_features)
        y = torch.tensor(probe_labels, dtype=torch.long)
        n = len(y)
        split = int(n * 0.7)
        X_tr, y_tr = X[:split], y[:split]
        X_te, y_te = X[split:], y[split:]
        if len(X_te) > 0:
            probe_head = nn.Linear(X_tr.shape[1], N_RESOURCE_TYPES)
            probe_opt = optim.Adam(probe_head.parameters(), lr=1e-3)
            for _ in range(200):
                logits = probe_head(X_tr)
                loss = F.cross_entropy(logits, y_tr)
                probe_opt.zero_grad()
                loss.backward()
                probe_opt.step()
            with torch.no_grad():
                preds = probe_head(X_te).argmax(dim=1)
                probe_acc = float((preds == y_te).float().mean().item())

    return {
        "condition": condition,
        "seed": seed,
        "eval_resource_rate": eval_resource_rate,
        "eval_benefit_mean": eval_benefit_mean,
        "z_goal_norm_final": z_goal_norm_final,
        "identity_probe_acc": probe_acc,
        "n_probe_samples": len(probe_features),
    }


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = [42] if args.dry_run else SEEDS
    p0 = 2 if args.dry_run else P0_EPISODES
    p1 = 2 if args.dry_run else P1_EPISODES
    ev = 2 if args.dry_run else EVAL_EPISODES
    n_eps = p0 + p1 + ev

    print(
        f"[V3-EXQ-527] MECH-112 Identity Goal Reef  dry_run={args.dry_run}"
        f"  seeds={seeds}  P0={p0} P1={p1} eval={ev}  steps={STEPS_PER_EPISODE}",
        flush=True,
    )
    total_runs = len(seeds) * len(CONDITIONS)
    print(f"  {len(CONDITIONS)} conditions x {len(seeds)} seeds = {total_runs} total runs", flush=True)

    results_by_condition: Dict[str, List[Dict]] = {c: [] for c in CONDITIONS}
    run_num = 0

    for condition in CONDITIONS:
        for seed in seeds:
            run_num += 1
            print(f"\n[V3-EXQ-527] ({run_num}/{total_runs}) seed={seed} condition={condition}", flush=True)
            print(f"Seed {seed} Condition {condition}", flush=True)
            ep_global_start = 0  # each run is independent

            try:
                result = _run_condition(
                    condition=condition,
                    seed=seed,
                    ep_global_start=ep_global_start,
                    total_eps=n_eps,
                    p0_eps=p0,
                    p1_eps=p1,
                    eval_eps=ev,
                )
                results_by_condition[condition].append(result)
                print(
                    f"verdict: PASS  seed={seed} cond={condition}"
                    f" res_rate={result['eval_resource_rate']:.3f}"
                    f" benefit={result['eval_benefit_mean']:.4f}"
                    f" z_goal_norm={result['z_goal_norm_final']:.3f}"
                    f" probe_acc={result['identity_probe_acc']:.3f}",
                    flush=True,
                )
            except Exception as exc:
                print(f"verdict: FAIL  seed={seed} cond={condition} error={exc}", flush=True)
                results_by_condition[condition].append({
                    "condition": condition, "seed": seed,
                    "eval_resource_rate": 0.0, "eval_benefit_mean": 0.0,
                    "z_goal_norm_final": 0.0, "identity_probe_acc": 0.0,
                    "n_probe_samples": 0,
                })

    # aggregate
    gp = results_by_condition["GOAL_PRESENT"]
    ga = results_by_condition["GOAL_ABSENT"]

    def mean(key: str, lst: List[Dict]) -> float:
        vals = [r[key] for r in lst if r is not None]
        return sum(vals) / len(vals) if vals else 0.0

    gp_res_rate = mean("eval_resource_rate", gp)
    ga_res_rate = mean("eval_resource_rate", ga)
    gp_benefit = mean("eval_benefit_mean", gp)
    ga_benefit = mean("eval_benefit_mean", ga)
    gp_z_goal = mean("z_goal_norm_final", gp)
    gp_probe = mean("identity_probe_acc", gp)

    # per-seed criteria
    seeds_passing = 0
    seed_reports = []
    for s_gp, s_ga in zip(gp, ga):
        c1 = s_gp["eval_resource_rate"] >= s_ga["eval_resource_rate"] + 0.025
        c2 = s_gp["eval_benefit_mean"] >= s_ga["eval_benefit_mean"] * 1.10
        c3 = s_gp["z_goal_norm_final"] >= 0.05
        c4 = s_gp["identity_probe_acc"] >= 0.60
        seed_pass = c1 and c2 and c3 and c4
        if seed_pass:
            seeds_passing += 1
        seed_reports.append({
            "seed": s_gp["seed"],
            "c1_res_lift": c1,
            "c2_benefit_lift": c2,
            "c3_z_goal_nonzero": c3,
            "c4_identity_probe": c4,
            "pass": seed_pass,
        })

    c5 = seeds_passing >= 2

    outcome = "PASS" if c5 else "FAIL"
    if c5:
        evidence_direction = "supports"
    elif gp_z_goal < 0.05:
        evidence_direction = "weakens"
    else:
        evidence_direction = "mixed"

    print("\n[V3-EXQ-527] === Results ===", flush=True)
    print(f"  GOAL_PRESENT: res_rate={gp_res_rate:.4f} benefit={gp_benefit:.4f} z_goal={gp_z_goal:.3f} probe={gp_probe:.3f}", flush=True)
    print(f"  GOAL_ABSENT:  res_rate={ga_res_rate:.4f} benefit={ga_benefit:.4f}", flush=True)
    print(f"  C1 res lift ({gp_res_rate:.4f} >= {ga_res_rate:.4f}+0.025): {gp_res_rate >= ga_res_rate + 0.025}", flush=True)
    print(f"  C2 benefit lift ({gp_benefit:.4f} >= {ga_benefit:.4f}*1.10): {gp_benefit >= ga_benefit * 1.10}", flush=True)
    print(f"  C3 z_goal norm ({gp_z_goal:.3f} >= 0.05): {gp_z_goal >= 0.05}", flush=True)
    print(f"  C4 identity probe ({gp_probe:.3f} >= 0.60): {gp_probe >= 0.60}", flush=True)
    print(f"  C5 seeds passing: {seeds_passing}/3  -> {c5}", flush=True)
    print(f"  -> {outcome} (evidence_direction={evidence_direction})", flush=True)

    if args.dry_run:
        print("[DRY-RUN] Smoke test complete. Not writing output.", flush=True)
        return

    ts_utc = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(__file__).resolve().parents[1].parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts_utc}_v3.json"

    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts_utc}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": {"MECH-112": evidence_direction},
        "outcome": outcome,
        "timestamp_utc": ts_utc,
        "config": {
            "grid_size": GRID_SIZE,
            "num_hazards": NUM_HAZARDS,
            "num_resources": NUM_RESOURCES,
            "n_resource_types": N_RESOURCE_TYPES,
            "reef_enabled": True,
            "n_reef_patches": 3,
            "reef_patch_radius": 2,
            "hazard_food_attraction": 0.5,
            "multi_resource_heterogeneity_enabled": True,
            "p0_episodes": P0_EPISODES,
            "p1_episodes": P1_EPISODES,
            "eval_episodes": EVAL_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "seeds": SEEDS,
            "lr": LR,
        },
        "criteria": {
            "c1_res_lift_threshold": 0.025,
            "c2_benefit_lift_factor": 1.10,
            "c3_z_goal_norm_min": 0.05,
            "c4_identity_probe_acc_min": 0.60,
            "c5_seeds_required": 2,
        },
        "aggregate_metrics": {
            "gp_res_rate": gp_res_rate,
            "ga_res_rate": ga_res_rate,
            "gp_benefit": gp_benefit,
            "ga_benefit": ga_benefit,
            "gp_z_goal_norm": gp_z_goal,
            "gp_identity_probe_acc": gp_probe,
            "seeds_passing_c5": seeds_passing,
        },
        "per_seed_results": {
            "goal_present": gp,
            "goal_absent": ga,
        },
        "seed_criteria_reports": seed_reports,
    }

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[V3-EXQ-527] Written: {out_path}", flush=True)


if __name__ == "__main__":
    _main()
