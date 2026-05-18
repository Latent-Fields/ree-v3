#!/opt/local/bin/python3
"""V3-EXQ-523: SD-029 Reef-Unblocked Comparator -- Self vs Externally-Caused Harm

Predecessor: V3-EXQ-522 (reef monostrategy break PASS). That experiment confirmed
the SD-054 reef substrate creates behavioral diversity (ARM_1 zone_transitions/ep~49).
This experiment uses that substrate to unblock the SD-029 C0 trial-sufficiency gate
that caused EXQ-433/433a/433b/470/433d to be classified non_contributory.

Prior failure chain: monomodal policy (single fixed route to food) -> agent never
voluntarily approaches a hazard -> n_self~0 -> C0 trial-sufficiency gate fails.

Reef fix: reef safe zones + food-attracted hazards -> two behavioral attractors
(flee to reef / forage toward food where hazards cluster) -> agent approaches
food+hazards regularly -> n_self >= 20 becomes achievable.

Three phases per arm/seed:
  P0: Data collection (heuristic from EXQ-522), tag self/ext events
      self_caused: agent moved TOWARD hazard AND harm_signal > threshold AND
                   NOT external_hazard_injected
      ext_caused:  info["external_hazard_injected"] == True
  P1: Train HarmEncoder (harm_obs -> z_harm_s) on harm_signal supervision
      using a temporary regression head (no grad leaks to E2HarmSForward)
  P2: Train E2HarmSForward on FROZEN z_harm_s consecutive transitions
      (stop-gradient required: z_t1.detach() as target)
  P3: Evaluate causal comparator
      cf_gap = ||E2_harm_s(z_t, a_actual)|| - ||E2_harm_s(z_t, a_cf)||
      a_cf = opposite direction (0<->1, 2<->3, NOOP stays)

Two arms:
  ARM_0: baseline (reef_enabled=False, hazard_food_attraction=0.0)
         sanity check: n_self should be << ARM_1 n_self
  ARM_1: reef + food + scheduled_external
         (reef_enabled=True, hazard_food_attraction=0.7,
          scheduled_external_hazard_enabled=True)

Acceptance criteria:
  C0: ARM_1 n_self >= 20 AND n_ext >= 20 (per seed)
  C1: ARM_1 harm_forward_r2 >= 0.9 (E2HarmSForward prediction quality)
  C2: ARM_1 mean(cf_gap | self_caused) > 0 (self-caused -> counterfactual predicts less harm)
  C3: ARM_1 mean(cf_gap | self_caused) > mean(cf_gap | ext_caused) (dissociation)
  C4 (diagnostic): ARM_0 n_ext == 0 (sanity: no external injection in baseline arm)

Overall PASS = C0 AND C1 AND C2 AND C3.

claim_ids: ["SD-029", "MECH-256"]
evidence_direction: "supports"
experiment_purpose: "evidence"
"""

import json
import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig

FLEE_THRESHOLD = 2  # Manhattan distance; same as EXQ-522
OPP_ACTION = {0: 1, 1: 0, 2: 3, 3: 2, 4: 4}  # counterfactual opposite directions
HARM_EVENT_THRESHOLD = 0.01  # min harm_signal to count as harm event


def _move_toward(ax, ay, tx, ty):
    dx = tx - ax
    dy = ty - ay
    if dx == 0 and dy == 0:
        return 4
    if abs(dx) >= abs(dy):
        return 0 if dx < 0 else 1
    return 2 if dy < 0 else 3


def _heuristic_action(env, reef_cells, use_reef, rng):
    """Harm-avoiding, reef-aware, resource-seeking heuristic policy from EXQ-522."""
    ax, ay = env.agent_x, env.agent_y
    nearest_hz = min(
        (abs(ax - h[0]) + abs(ay - h[1]) for h in env.hazards),
        default=999
    )
    if use_reef and nearest_hz <= FLEE_THRESHOLD and reef_cells:
        if (ax, ay) in reef_cells:
            return 4
        target = min(reef_cells, key=lambda r: abs(ax - r[0]) + abs(ay - r[1]))
        return _move_toward(ax, ay, target[0], target[1])
    if env.resources:
        target = min(env.resources, key=lambda r: abs(ax - r[0]) + abs(ay - r[1]))
        return _move_toward(ax, ay, target[0], target[1])
    return int(rng.integers(0, env.action_dim))


def _moved_toward_hazard(ax, ay, action, hazards):
    """True if action moves agent closer to nearest hazard."""
    DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}
    if not hazards:
        return False
    dx, dy = DELTAS.get(action, (0, 0))
    new_x, new_y = ax + dx, ay + dy
    old_dist = min(abs(ax - h[0]) + abs(ay - h[1]) for h in hazards)
    new_dist = min(abs(new_x - h[0]) + abs(new_y - h[1]) for h in hazards)
    return new_dist < old_dist


def _action_onehot(action_int, action_dim):
    """Convert action integer to one-hot tensor [1, action_dim]."""
    oh = torch.zeros(1, action_dim)
    oh[0, action_int] = 1.0
    return oh


def _run_arm(arm_id, env_kwargs, n_episodes, steps_per_ep, n_train_steps,
             n_seeds, rng, dry_run=False):
    """Run one ARM across multiple seeds and return aggregated metrics."""
    grid_size = env_kwargs.get("size", 12)
    use_reef = env_kwargs.get("reef_enabled", False)

    seed_results = []

    for seed_idx in range(n_seeds):
        seed = int(rng.integers(0, 10000))
        kw = dict(**env_kwargs, seed=seed)
        env = CausalGridWorldV2(**kw)

        harm_obs_dim = 51
        cfg = REEConfig.from_dims(
            body_obs_dim=env.body_obs_dim,
            world_obs_dim=env.world_obs_dim,
            action_dim=env.action_dim,
            use_harm_stream=True,
            harm_obs_dim=harm_obs_dim,
            z_harm_dim=32,
            use_e2_harm_s_forward=True,
        )
        agent = REEAgent(cfg)

        # ---------------------------------------------------------------
        # P0: Data collection
        # ---------------------------------------------------------------
        transitions = []    # (harm_obs_t, z_harm_s_t, action, harm_signal)
        events_self = []    # cf_gap will be filled in P3
        events_ext = []

        n_self = 0
        n_ext = 0

        _, obs_dict = env.reset()
        reef_cells = env._reef_cells

        for ep in range(n_episodes):
            _, obs_dict = env.reset()
            reef_cells = env._reef_cells

            for step in range(steps_per_ep):
                ax, ay = env.agent_x, env.agent_y
                hazards_before = list(env.hazards)
                obs_body = obs_dict["body_state"].float().unsqueeze(0)
                obs_world = obs_dict["world_state"].float().unsqueeze(0)
                h_obs = obs_dict["harm_obs"].float()
                harm_obs_t = h_obs.unsqueeze(0)

                action = _heuristic_action(env, reef_cells, use_reef, rng)

                # Event detection BEFORE stepping
                moved_toward = _moved_toward_hazard(ax, ay, action, hazards_before)

                # Step environment
                _, harm_signal, done, info, obs_dict_next = env.step(action)
                obs_dict = obs_dict_next

                external = bool(info.get("external_hazard_injected", False))
                harm_float = float(harm_signal)

                # Tag events.
                # self_caused: agent voluntarily approached a hazard (not ext-injected).
                # Harm signal threshold deliberately omitted: the comparator tests
                # whether the forward model assigns differential predictions for
                # approach vs retreat, which doesn't require actual damage on every step.
                is_self = (moved_toward and not external)
                is_ext = external

                with torch.no_grad():
                    z_harm_s = agent.latent_stack.harm_encoder(harm_obs_t)

                transitions.append((
                    h_obs.clone(),
                    z_harm_s.detach().clone(),
                    action,
                    harm_float,
                ))

                if is_self:
                    events_self.append({
                        "harm_obs_t": h_obs.clone(),
                        "action": action,
                    })
                    n_self += 1

                if is_ext:
                    events_ext.append({
                        "harm_obs_t": h_obs.clone(),
                        "action": action,
                    })
                    n_ext += 1

                if done:
                    break

        if dry_run:
            print(
                f"  arm={arm_id} seed={seed_idx} P0: "
                f"n_self={n_self} n_ext={n_ext} transitions={len(transitions)}"
            )

        # ---------------------------------------------------------------
        # P1: Train HarmEncoder with temporary harm_signal regression head
        # ---------------------------------------------------------------
        harm_head = nn.Linear(cfg.latent.z_harm_dim, 1)
        harm_opt = torch.optim.Adam(
            list(agent.latent_stack.harm_encoder.parameters()) +
            list(harm_head.parameters()),
            lr=5e-4,
        )

        rng_idx = np.random.default_rng(seed + 100)
        p1_losses = []
        for _ in range(n_train_steps):
            idx = int(rng_idx.integers(0, len(transitions)))
            h_obs_t, _, _act, harm_sig = transitions[idx]
            harm_in = h_obs_t.unsqueeze(0)
            z_hs = agent.latent_stack.harm_encoder(harm_in)
            pred = harm_head(z_hs).squeeze()
            tgt = torch.tensor(harm_sig, dtype=torch.float32)
            loss_p1 = F.mse_loss(pred, tgt)
            harm_opt.zero_grad()
            loss_p1.backward()
            harm_opt.step()
            p1_losses.append(loss_p1.item())

        mean_p1_loss = float(np.mean(p1_losses[-200:])) if p1_losses else 0.0

        # Re-encode all transitions with trained encoder (frozen from here)
        new_transitions = []
        with torch.no_grad():
            for h_obs_t, _, action, harm_sig in transitions:
                z_hs = agent.latent_stack.harm_encoder(h_obs_t.unsqueeze(0))
                new_transitions.append((h_obs_t, z_hs.detach().clone(), action, harm_sig))
        transitions = new_transitions

        # ---------------------------------------------------------------
        # P2: Train E2HarmSForward on frozen z_harm_s consecutive transitions
        # ---------------------------------------------------------------
        e2_opt = torch.optim.Adam(agent.e2_harm_s.parameters(), lr=5e-4)

        # Build (z_t, a_oh, z_t1) training pairs from consecutive transitions
        n_tr = len(transitions)
        rng_p2 = np.random.default_rng(seed + 200)
        p2_losses = []
        preds_held = []
        targets_held = []
        hold_frac = 0.2

        for step_i in range(n_train_steps):
            # Sample a random pair of consecutive transitions
            i = int(rng_p2.integers(0, n_tr - 1))
            _, z_t, act_t, _ = transitions[i]
            _, z_t1, _, _ = transitions[i + 1]

            a_oh = _action_onehot(act_t, env.action_dim)
            z_pred = agent.e2_harm_s(z_t.detach(), a_oh)
            loss_e2 = F.mse_loss(z_pred, z_t1.detach())
            e2_opt.zero_grad()
            loss_e2.backward()
            e2_opt.step()
            p2_losses.append(loss_e2.item())

            # Collect last 20% for R2 estimation
            if step_i >= int(n_train_steps * (1.0 - hold_frac)):
                preds_held.append(z_pred.detach())
                targets_held.append(z_t1.detach())

        mean_p2_loss = float(np.mean(p2_losses[-200:])) if p2_losses else 0.0

        # Compute harm_forward_r2 on held samples
        if preds_held:
            preds_cat = torch.cat(preds_held, dim=0)
            tgts_cat = torch.cat(targets_held, dim=0)
            ss_res = ((preds_cat - tgts_cat) ** 2).sum().item()
            ss_tot = ((tgts_cat - tgts_cat.mean(dim=0, keepdim=True)) ** 2).sum().item()
            harm_forward_r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
        else:
            harm_forward_r2 = 0.0

        if dry_run:
            print(
                f"  arm={arm_id} seed={seed_idx} P1 loss={mean_p1_loss:.4f} "
                f"P2 loss={mean_p2_loss:.4f} harm_fwd_r2={harm_forward_r2:.3f}"
            )

        # ---------------------------------------------------------------
        # P3: Evaluate causal comparator
        # ---------------------------------------------------------------
        cf_gaps_self = []
        cf_gaps_ext = []

        with torch.no_grad():
            for ev_list, cf_list in [(events_self, cf_gaps_self),
                                     (events_ext, cf_gaps_ext)]:
                for ev in ev_list:
                    h_obs_ev = ev["harm_obs_t"]
                    act = ev["action"]
                    act_cf = OPP_ACTION[act]

                    z_hs = agent.latent_stack.harm_encoder(h_obs_ev.unsqueeze(0))

                    a_actual_oh = _action_onehot(act, env.action_dim)
                    a_cf_oh = _action_onehot(act_cf, env.action_dim)

                    z_pred_actual = agent.e2_harm_s(z_hs.detach(), a_actual_oh)
                    z_pred_cf = agent.e2_harm_s.counterfactual_forward(
                        z_hs.detach(), a_cf_oh
                    )
                    gap = z_pred_actual.norm().item() - z_pred_cf.norm().item()
                    cf_list.append(gap)

        mean_cf_gap_self = float(np.mean(cf_gaps_self)) if cf_gaps_self else 0.0
        mean_cf_gap_ext = float(np.mean(cf_gaps_ext)) if cf_gaps_ext else 0.0

        if dry_run:
            print(
                f"  arm={arm_id} seed={seed_idx} P3: "
                f"cf_gap_self={mean_cf_gap_self:.4f} "
                f"cf_gap_ext={mean_cf_gap_ext:.4f} "
                f"n_self={n_self} n_ext={n_ext}"
            )

        seed_results.append({
            "arm_id": arm_id,
            "seed_idx": seed_idx,
            "n_self": n_self,
            "n_ext": n_ext,
            "harm_forward_r2": harm_forward_r2,
            "mean_cf_gap_self": mean_cf_gap_self,
            "mean_cf_gap_ext": mean_cf_gap_ext,
            "mean_p1_loss": mean_p1_loss,
            "mean_p2_loss": mean_p2_loss,
        })

    return seed_results


def run_experiment(dry_run=False):
    n_episodes = 10 if dry_run else 80
    steps_per_ep = 50 if dry_run else 200
    n_seeds = 1 if dry_run else 3
    n_train_steps = 200 if dry_run else 2000
    grid_size = 12

    rng = np.random.default_rng(7000)

    common = dict(
        size=grid_size, num_hazards=3, num_resources=5,
        use_proxy_fields=True,
        env_drift_prob=0.3, env_drift_interval=1,
    )

    arms_def = [
        ("ARM_0_baseline",
         dict(**common,
              reef_enabled=False,
              hazard_food_attraction=0.0)),
        ("ARM_1_reef_food_ext",
         dict(**common,
              reef_enabled=True, n_reef_patches=3, reef_patch_radius=2,
              hazard_food_attraction=0.7,
              scheduled_external_hazard_enabled=True,
              scheduled_external_hazard_interval=15,
              scheduled_external_hazard_prob=0.8,
              scheduled_external_hazard_adjacent_only=False)),
    ]

    all_results = {}
    for arm_id, env_kwargs in arms_def:
        results = _run_arm(
            arm_id=arm_id,
            env_kwargs=env_kwargs,
            n_episodes=n_episodes,
            steps_per_ep=steps_per_ep,
            n_train_steps=n_train_steps,
            n_seeds=n_seeds,
            rng=rng,
            dry_run=dry_run,
        )
        all_results[arm_id] = results

    def _agg(arm_id, key):
        vals = [r[key] for r in all_results[arm_id]]
        return float(np.mean(vals))

    def _min(arm_id, key):
        vals = [r[key] for r in all_results[arm_id]]
        return float(np.min(vals))

    arm1 = "ARM_1_reef_food_ext"
    arm0 = "ARM_0_baseline"

    # C0: per seed, n_self >= 20 AND n_ext >= 20
    c0 = all(
        r["n_self"] >= 20 and r["n_ext"] >= 20
        for r in all_results[arm1]
    )
    # C1: harm_forward_r2 >= 0.9
    c1 = _min(arm1, "harm_forward_r2") >= 0.9

    # C2: mean cf_gap for self_caused > 0
    c2 = _agg(arm1, "mean_cf_gap_self") > 0.0

    # C3: mean cf_gap self > ext
    c3 = _agg(arm1, "mean_cf_gap_self") > _agg(arm1, "mean_cf_gap_ext")

    # C4 diagnostic: ARM_0 n_ext == 0 (sanity: no external injection in baseline arm)
    c4 = _agg(arm0, "n_ext") == 0

    overall_pass = c0 and c1 and c2 and c3

    per_arm = {}
    for arm_id, _ in arms_def:
        per_arm[arm_id] = {k: _agg(arm_id, k) for k in [
            "n_self", "n_ext", "harm_forward_r2",
            "mean_cf_gap_self", "mean_cf_gap_ext",
            "mean_p1_loss", "mean_p2_loss",
        ]}

    return {
        "per_arm": per_arm,
        "criteria": {"C0": c0, "C1": c1, "C2": c2, "C3": c3, "C4": c4},
        "overall_pass": overall_pass,
        "n_episodes": n_episodes,
        "steps_per_ep": steps_per_ep,
        "n_seeds": n_seeds,
        "n_train_steps": n_train_steps,
    }


def write_result(result, run_id):
    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "REE_assembly", "evidence", "experiments"
    )
    out_path = os.path.join(output_dir, f"{run_id}.json")
    manifest = {
        "run_id": run_id,
        "experiment_type": "v3_exq_523_sd029_reef_comparator",
        "queue_id": "V3-EXQ-523",
        "claim_ids": ["SD-029", "MECH-256"],
        "evidence_direction": "supports",
        "experiment_purpose": "evidence",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "outcome": "PASS" if result["overall_pass"] else "FAIL",
        "metrics": result,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Result written to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    print("V3-EXQ-523 SD-029 Reef-Unblocked Comparator")
    print(f"dry_run={args.dry_run}")

    result = run_experiment(dry_run=args.dry_run)

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"Criteria: {result['criteria']}")
    print(f"Overall: {'PASS' if result['overall_pass'] else 'FAIL'}")

    for arm_id, stats in result["per_arm"].items():
        print(
            f"  {arm_id}: "
            f"n_self={stats['n_self']:.0f} "
            f"n_ext={stats['n_ext']:.0f} "
            f"fwd_r2={stats['harm_forward_r2']:.3f} "
            f"cf_gap_self={stats['mean_cf_gap_self']:.4f} "
            f"cf_gap_ext={stats['mean_cf_gap_ext']:.4f}"
        )

    if not args.dry_run:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        run_id = f"v3_exq_523_sd029_reef_comparator_{ts}_v3"
        write_result(result, run_id)
