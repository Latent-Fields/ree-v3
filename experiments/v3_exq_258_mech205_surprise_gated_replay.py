#!/opt/local/bin/python3
"""
V3-EXQ-258 -- MECH-205: Surprise-Gated Replay Validation (Tier 1)

Claims: MECH-205
EXPERIMENT_PURPOSE = "evidence"

=== SCIENTIFIC QUESTION ===

Does surprise-gated replay prioritisation (MECH-205 Tier 1) produce
measurably different consolidation mechanics vs serotonergic replay alone?

Mechanism under test: PE-populated VALENCE_SURPRISE tags in residue field
cause replay start selection to favour high-surprise z_world locations.

=== DESIGN ===

2-condition (WITH vs WITHOUT surprise gating), 3 seeds.
  Condition A (CONTROL):  surprise_gated_replay=False, tonic_5ht_enabled=True
  Condition B (SURPRISE): surprise_gated_replay=True,  tonic_5ht_enabled=True

Both conditions use the serotonergic substrate (MECH-203) so replay is
valence-weighted in both cases. The only difference is whether the surprise
channel (VALENCE_SURPRISE) is populated with PE data.

=== PRE-REGISTERED CRITERIA (evaluated per seed, majority >= 2/3) ===

  P1: surprise_tag_populated -- VALENCE_SURPRISE has non-zero entries in B
      AND is zero in A (mechanism is the exclusive source)
  P2: pe_samples_collected   -- n_pe_samples > 0 in B (PE tracking fires)
  P3: pe_ema_varies          -- std(PE EMA) > 1e-6 in B (EMA is not constant)

  PASS: P1 AND P2 AND P3
"""

import sys
import random
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE    = "v3_exq_258_mech205_surprise_gated_replay"
CLAIM_IDS          = ["MECH-205"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
MAJORITY_THRESH = 2  # >= 2 out of 3 seeds

# ---------------------------------------------------------------------------
# Grid and episode parameters
# ---------------------------------------------------------------------------
GRID_SIZE       = 8
N_RESOURCES     = 2
N_HAZARDS       = 1
STEPS_PER_EP    = 80

# ---------------------------------------------------------------------------
# Training parameters
# ---------------------------------------------------------------------------
WARMUP_EPISODES = 200
EVAL_EPISODES   = 50
SEEDS           = [42, 7, 13]
GREEDY_FRAC     = 0.4
MAX_BUF         = 4000
WF_BUF_MAX      = 2000
WORLD_DIM       = 32
BATCH_SIZE      = 16
REPLAY_INTERVAL = 10  # trigger replay every N steps during eval

# Learning rates
LR_E1      = 1e-3
LR_E2_WF   = 1e-3
LR_HARM    = 1e-4
LR_BENEFIT = 1e-3

# SD-018: resource proximity supervision
LAMBDA_RESOURCE = 0.5

# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------
CONDITIONS = [
    ("A_CONTROL",  False),  # (label, surprise_gated_replay)
    ("B_SURPRISE", True),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _greedy_toward_resource(env) -> int:
    """Greedy action: move toward nearest resource (Manhattan)."""
    ax, ay = env.agent_x, env.agent_y
    if not env.resources:
        return random.randint(0, env.action_dim - 1)
    best_d = float("inf")
    nearest = None
    for r in env.resources:
        rx, ry = int(r[0]), int(r[1])
        d = abs(ax - rx) + abs(ay - ry)
        if d < best_d:
            best_d = d
            nearest = (rx, ry)
    if nearest is None or best_d == 0:
        return random.randint(0, env.action_dim - 1)
    rx, ry = nearest
    dx, dy = rx - ax, ry - ay
    if abs(dx) >= abs(dy):
        return 1 if dx > 0 else 0
    return 3 if dy > 0 else 2


def _dist_to_nearest_resource(env) -> int:
    if not env.resources:
        return 999
    ax, ay = env.agent_x, env.agent_y
    return min(abs(ax - int(r[0])) + abs(ay - int(r[1])) for r in env.resources)


def _get_benefit_exposure(obs_body: torch.Tensor) -> float:
    flat = obs_body.flatten()
    if flat.shape[0] > 11:
        return float(flat[11].item())
    return 0.0


def _get_energy(obs_body: torch.Tensor) -> float:
    flat = obs_body.flatten()
    if flat.shape[0] > 3:
        return float(flat[3].item())
    return 1.0


def _update_z_goal(agent: REEAgent, obs_body: torch.Tensor) -> None:
    b_exp = _get_benefit_exposure(obs_body)
    energy = _get_energy(obs_body)
    drive_level = max(0.0, 1.0 - energy)
    agent.update_z_goal(b_exp, drive_level=drive_level)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_resources=N_RESOURCES,
        num_hazards=N_HAZARDS,
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
    )


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _make_agent(
    surprise_gated: bool, env: CausalGridWorldV2, seed: int,
) -> REEAgent:
    torch.manual_seed(seed)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=16,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=0,
        novelty_bonus_weight=0.0,
        benefit_eval_enabled=True,
        benefit_weight=0.5,
        z_goal_enabled=True,
        e1_goal_conditioned=True,
        goal_weight=1.0,
        drive_weight=2.0,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        tonic_5ht_enabled=True,
        surprise_gated_replay=surprise_gated,
    )
    return REEAgent(config)


# ---------------------------------------------------------------------------
# Warmup: train encoder + heads, mixed policy
# ---------------------------------------------------------------------------

def _warmup(
    agent: REEAgent,
    env: CausalGridWorldV2,
    label: str,
    warmup_episodes: int,
    seed: int,
) -> None:
    device = agent.device
    n_act  = env.action_dim

    e1_params    = list(agent.e1.parameters())
    e2_wf_params = (
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters())
    )
    e1_opt      = optim.Adam(e1_params, lr=LR_E1)
    e2_wf_opt   = optim.Adam(e2_wf_params, lr=LR_E2_WF)
    harm_opt    = optim.Adam(agent.e3.harm_eval_head.parameters(), lr=LR_HARM)
    benefit_opt = optim.Adam(agent.e3.benefit_eval_head.parameters(), lr=LR_BENEFIT)

    wf_buf:       List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_pos_buf: List[torch.Tensor] = []
    harm_neg_buf: List[torch.Tensor] = []
    ben_zw_buf:   List[torch.Tensor] = []
    ben_lbl_buf:  List[float]        = []

    random.seed(seed)
    agent.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        z_world_prev: Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None

        for step_i in range(STEPS_PER_EP):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            ticks  = agent.clock.advance()

            if ticks.get("e1_tick", False):
                _ = agent._e1_tick(latent)

            z_world_curr = latent.z_world.detach()

            # E2 world_forward buffer
            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev, action_prev, z_world_curr))
                if len(wf_buf) > WF_BUF_MAX:
                    wf_buf = wf_buf[-WF_BUF_MAX:]

            # Mixed policy: greedy toward resource or random
            if random.random() < GREEDY_FRAC:
                action_idx = _greedy_toward_resource(env)
            else:
                action_idx = random.randint(0, n_act - 1)
            action_oh = _onehot(action_idx, n_act, device)
            agent._last_action = action_oh

            dist   = _dist_to_nearest_resource(env)
            is_near = 1.0 if dist <= 2 else 0.0

            _, harm_signal, done, info, obs_dict = env.step(action_oh)

            _update_z_goal(agent, obs_dict["body_state"])

            # Train E1
            if len(agent._world_experience_buffer) >= 2:
                e1_loss = agent.compute_prediction_loss()
                if e1_loss.requires_grad:
                    e1_opt.zero_grad()
                    e1_loss.backward()
                    torch.nn.utils.clip_grad_norm_(e1_params, 1.0)
                    e1_opt.step()

            # SD-018: resource proximity supervision
            rfv = obs_dict.get("resource_field_view", None)
            if rfv is not None:
                rp_target = rfv[12].item()
                rp_loss = agent.compute_resource_proximity_loss(rp_target, latent)
                if rp_loss.requires_grad:
                    e1_opt.zero_grad()
                    (LAMBDA_RESOURCE * rp_loss).backward()
                    torch.nn.utils.clip_grad_norm_(e1_params, 1.0)
                    e1_opt.step()

            # Train E2 world_forward
            if len(wf_buf) >= BATCH_SIZE:
                idxs  = random.sample(range(len(wf_buf)), min(BATCH_SIZE, len(wf_buf)))
                zw_b  = torch.cat([wf_buf[i][0] for i in idxs]).to(device)
                a_b   = torch.cat([wf_buf[i][1] for i in idxs]).to(device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(device)
                wf_pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(wf_pred, zw1_b)
                if wf_loss.requires_grad:
                    e2_wf_opt.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(e2_wf_params, 1.0)
                    e2_wf_opt.step()
                with torch.no_grad():
                    agent.e3.update_running_variance(
                        (wf_pred.detach() - zw1_b).detach()
                    )

            # Train E3 harm_eval (stratified)
            if float(harm_signal) < 0:
                harm_pos_buf.append(z_world_curr)
                if len(harm_pos_buf) > MAX_BUF:
                    harm_pos_buf = harm_pos_buf[-MAX_BUF:]
            else:
                harm_neg_buf.append(z_world_curr)
                if len(harm_neg_buf) > MAX_BUF:
                    harm_neg_buf = harm_neg_buf[-MAX_BUF:]

            if len(harm_pos_buf) >= 4 and len(harm_neg_buf) >= 4:
                k_p = min(BATCH_SIZE // 2, len(harm_pos_buf))
                k_n = min(BATCH_SIZE // 2, len(harm_neg_buf))
                pi  = torch.randperm(len(harm_pos_buf))[:k_p].tolist()
                ni  = torch.randperm(len(harm_neg_buf))[:k_n].tolist()
                zw_b = torch.cat(
                    [harm_pos_buf[i] for i in pi] + [harm_neg_buf[i] for i in ni],
                    dim=0,
                )
                tgt = torch.cat([
                    torch.ones(k_p,  1, device=device),
                    torch.zeros(k_n, 1, device=device),
                ], dim=0)
                pred = agent.e3.harm_eval(zw_b)
                hloss = F.binary_cross_entropy(pred, tgt)
                if hloss.requires_grad:
                    harm_opt.zero_grad()
                    hloss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5
                    )
                    harm_opt.step()

            # Train E3 benefit_eval
            ben_zw_buf.append(z_world_curr)
            ben_lbl_buf.append(is_near)
            if len(ben_zw_buf) > MAX_BUF:
                ben_zw_buf  = ben_zw_buf[-MAX_BUF:]
                ben_lbl_buf = ben_lbl_buf[-MAX_BUF:]

            if len(ben_zw_buf) >= 32 and step_i % 4 == 0:
                k    = min(32, len(ben_zw_buf))
                idxs = random.sample(range(len(ben_zw_buf)), k)
                zw_b = torch.cat([ben_zw_buf[i] for i in idxs], dim=0)
                lbl  = torch.tensor(
                    [ben_lbl_buf[i] for i in idxs],
                    dtype=torch.float32,
                ).unsqueeze(1).to(device)
                pred_b = agent.e3.benefit_eval(zw_b)
                bloss  = F.binary_cross_entropy(pred_b, lbl)
                if bloss.requires_grad:
                    benefit_opt.zero_grad()
                    bloss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.benefit_eval_head.parameters(), 0.5
                    )
                    benefit_opt.step()
                    agent.e3.record_benefit_sample(k)

            z_world_prev = z_world_curr
            action_prev  = action_oh

            if done:
                break

        if (ep + 1) % 50 == 0 or ep == warmup_episodes - 1:
            print(
                f"    [train] cond={label}"
                f" seed={seed}"
                f" ep {ep+1}/{warmup_episodes}"
                f" harm_pos={len(harm_pos_buf)}"
                f" harm_neg={len(harm_neg_buf)}",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Eval: full agent pipeline, collect MECH-205 metrics
# ---------------------------------------------------------------------------

def _eval(
    agent: REEAgent,
    env: CausalGridWorldV2,
    label: str,
    eval_episodes: int,
) -> Dict:
    """Eval with full agent pipeline. Collects MECH-205 PE/surprise metrics."""
    agent.eval()

    device    = agent.device
    world_dim = WORLD_DIM
    n_act     = env.action_dim

    pe_mags:    List[float] = []
    pe_emas:    List[float] = []
    surprises:  List[float] = []

    for ep in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for step_i in range(STEPS_PER_EP):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            ticks  = agent.clock.advance()

            if ticks.get("e1_tick", False):
                e1_prior = agent._e1_tick(latent)
            else:
                e1_prior = torch.zeros(1, world_dim, device=device)

            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action     = agent.select_action(candidates, ticks, temperature=1.0)

            if action is None:
                action = _onehot(
                    random.randint(0, n_act - 1), n_act, device
                )
                agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)

            # Collect MECH-205 metrics from update_residue
            residue_metrics = agent.update_residue(float(harm_signal))
            pe_mag = residue_metrics.get("mech205_pe_mag")
            if pe_mag is not None:
                pe_mags.append(float(pe_mag))
                pe_emas.append(float(residue_metrics.get("mech205_pe_ema", 0)))
                surprises.append(float(residue_metrics.get("mech205_surprise", 0)))

            # Trigger replay periodically (simulate quiescent cycles)
            if step_i % REPLAY_INTERVAL == 0 and agent._current_latent is not None:
                agent._do_replay(agent._current_latent)

            with torch.no_grad():
                _update_z_goal(agent, obs_dict["body_state"])

            if done:
                break

    # Check VALENCE_SURPRISE population on residue field
    surprise_tag_populated = False
    if hasattr(agent.residue_field, '_rbf_layer') and agent.residue_field._rbf_layer is not None:
        vv = agent.residue_field._rbf_layer.valence_vecs
        if vv is not None:
            surprise_vals = vv[:, 3].abs().sum().item()  # VALENCE_SURPRISE = index 3
            surprise_tag_populated = surprise_vals > 0.0

    pe_ema_std = float(torch.tensor(pe_emas).std().item()) if len(pe_emas) > 1 else 0.0

    return {
        "n_pe_samples":          len(pe_mags),
        "mean_pe_mag":           float(sum(pe_mags) / max(1, len(pe_mags))),
        "mean_pe_ema":           float(sum(pe_emas) / max(1, len(pe_emas))),
        "pe_ema_std":            pe_ema_std,
        "pe_ema_varies":         pe_ema_std > 1e-6,
        "mean_surprise":         float(sum(surprises) / max(1, len(surprises))),
        "surprise_tag_populated": surprise_tag_populated,
    }


# ---------------------------------------------------------------------------
# Per-seed run
# ---------------------------------------------------------------------------

def _run_seed(
    seed: int,
    warmup_episodes: int,
    eval_episodes: int,
) -> Dict:
    """Run both conditions for one seed."""
    results: Dict = {}

    for label, surprise_gated in CONDITIONS:
        print(
            f"\n[V3-EXQ-258] Seed {seed} Condition {label}"
            f" surprise_gated={surprise_gated}",
            flush=True,
        )
        env   = _make_env(seed)
        agent = _make_agent(surprise_gated, env, seed)

        _warmup(agent, env, label, warmup_episodes, seed)

        eval_res = _eval(agent, env, label, eval_episodes)

        print(
            f"  [eval done] seed={seed} cond={label}"
            f" n_pe={eval_res['n_pe_samples']}"
            f" pe_mag={eval_res['mean_pe_mag']:.4f}"
            f" pe_ema={eval_res['mean_pe_ema']:.4f}"
            f" pe_ema_std={eval_res['pe_ema_std']:.6f}"
            f" surprise={eval_res['mean_surprise']:.4f}"
            f" tag_pop={eval_res['surprise_tag_populated']}",
            flush=True,
        )
        print(f"verdict: {'PASS' if eval_res['surprise_tag_populated'] else 'FAIL'}", flush=True)

        results[label] = eval_res

    return results


# ---------------------------------------------------------------------------
# Aggregate and criteria
# ---------------------------------------------------------------------------

def _aggregate(all_results: List[Dict]) -> Dict:
    n_seeds = len(all_results)

    per_seed_criteria = []
    for r in all_results:
        ctrl = r["A_CONTROL"]
        surp = r["B_SURPRISE"]

        p1 = surp["surprise_tag_populated"] and not ctrl["surprise_tag_populated"]
        p2 = surp["n_pe_samples"] > 0
        p3 = surp["pe_ema_varies"]

        per_seed_criteria.append({
            "p1_surprise_tag": p1,
            "p2_pe_collected": p2,
            "p3_pe_ema_varies": p3,
            "seed_pass": p1 and p2 and p3,
        })

    p1_count = sum(1 for s in per_seed_criteria if s["p1_surprise_tag"])
    p2_count = sum(1 for s in per_seed_criteria if s["p2_pe_collected"])
    p3_count = sum(1 for s in per_seed_criteria if s["p3_pe_ema_varies"])
    pass_count = sum(1 for s in per_seed_criteria if s["seed_pass"])

    return {
        "n_seeds": n_seeds,
        "per_seed_criteria": per_seed_criteria,
        "p1_count": p1_count,
        "p2_count": p2_count,
        "p3_count": p3_count,
        "pass_count": pass_count,
        "p1_pass": p1_count >= MAJORITY_THRESH,
        "p2_pass": p2_count >= MAJORITY_THRESH,
        "p3_pass": p3_count >= MAJORITY_THRESH,
        "overall_pass": pass_count >= MAJORITY_THRESH,
    }


def _decide(agg: Dict) -> Tuple[str, str, str]:
    """Return (outcome, evidence_direction, decision)."""
    if agg["overall_pass"]:
        return "PASS", "supports", "surprise_gated_replay_functional"
    if not agg["p2_pass"]:
        return "FAIL", "non_contributory", "pe_tracking_not_operational"
    return "FAIL", "weakens", "surprise_gating_nonfunctional"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    warmup = 5    if args.dry_run else WARMUP_EPISODES
    n_eval = 5    if args.dry_run else EVAL_EPISODES
    seeds  = [42] if args.dry_run else SEEDS

    print(
        f"[V3-EXQ-258] MECH-205: Surprise-Gated Replay Validation"
        f" dry_run={args.dry_run}"
        f" warmup={warmup} eval={n_eval} seeds={seeds}",
        flush=True,
    )
    print(
        f"  2 conditions: A_CONTROL (no surprise gate), B_SURPRISE (surprise gate)",
        flush=True,
    )

    all_results = []
    for seed in seeds:
        print(f"\n[V3-EXQ-258] === Seed {seed} ===", flush=True)
        seed_results = _run_seed(
            seed=seed,
            warmup_episodes=warmup,
            eval_episodes=n_eval,
        )
        all_results.append(seed_results)

    agg = _aggregate(all_results)
    outcome, direction, decision = _decide(agg)

    print(f"\n[V3-EXQ-258] === Results ===", flush=True)
    for i, seed in enumerate(seeds):
        sc = agg["per_seed_criteria"][i]
        print(
            f"  Seed {seed}:"
            f" P1={sc['p1_surprise_tag']}"
            f" P2={sc['p2_pe_collected']}"
            f" P3={sc['p3_pe_ema_varies']}"
            f" -> {'PASS' if sc['seed_pass'] else 'FAIL'}",
            flush=True,
        )
    print(
        f"  P1 (surprise_tag): {agg['p1_count']}/{agg['n_seeds']}"
        f" {'PASS' if agg['p1_pass'] else 'FAIL'}",
        flush=True,
    )
    print(
        f"  P2 (pe_collected): {agg['p2_count']}/{agg['n_seeds']}"
        f" {'PASS' if agg['p2_pass'] else 'FAIL'}",
        flush=True,
    )
    print(
        f"  P3 (pe_ema_varies): {agg['p3_count']}/{agg['n_seeds']}"
        f" {'PASS' if agg['p3_pass'] else 'FAIL'}",
        flush=True,
    )
    print(
        f"  -> {outcome} decision={decision} direction={direction}",
        flush=True,
    )

    if args.dry_run:
        print("\n[DRY-RUN] Smoke test complete. Not writing output.", flush=True)
        return

    # Write output
    ts_utc  = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    ts_unix = int(time.time())
    root    = Path(__file__).resolve().parents[2]
    out_dir = root / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts_unix}.json"

    # Collect per-condition metrics across seeds
    per_condition = []
    for i, seed in enumerate(seeds):
        for label, _ in CONDITIONS:
            r = all_results[i][label]
            per_condition.append({
                "seed": seed,
                "condition": label,
                "n_pe_samples": r["n_pe_samples"],
                "mean_pe_mag": r["mean_pe_mag"],
                "mean_pe_ema": r["mean_pe_ema"],
                "pe_ema_std": r["pe_ema_std"],
                "pe_ema_varies": r["pe_ema_varies"],
                "mean_surprise": r["mean_surprise"],
                "surprise_tag_populated": r["surprise_tag_populated"],
            })

    manifest = {
        "run_id":             f"{EXPERIMENT_TYPE}_{ts_unix}_v3",
        "experiment_type":    EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids":          CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome":            outcome,
        "evidence_direction": direction,
        "decision":           decision,
        "timestamp_utc":      ts_utc,
        "timestamp":          ts_unix,
        "seeds":              seeds,
        # Parameters
        "warmup_episodes":    WARMUP_EPISODES,
        "eval_episodes":      EVAL_EPISODES,
        "steps_per_episode":  STEPS_PER_EP,
        "grid_size":          GRID_SIZE,
        "n_resources":        N_RESOURCES,
        "n_hazards":          N_HAZARDS,
        "replay_interval":    REPLAY_INTERVAL,
        "conditions": [
            {"label": lbl, "surprise_gated": sg}
            for lbl, sg in CONDITIONS
        ],
        # Aggregate criteria
        "p1_surprise_tag_pass":  agg["p1_pass"],
        "p1_count":              agg["p1_count"],
        "p2_pe_collected_pass":  agg["p2_pass"],
        "p2_count":              agg["p2_count"],
        "p3_pe_ema_varies_pass": agg["p3_pass"],
        "p3_count":              agg["p3_count"],
        "pass_count":            agg["pass_count"],
        "majority_thresh":       MAJORITY_THRESH,
        # Per-seed criteria
        "per_seed_criteria":     agg["per_seed_criteria"],
        # Per-condition detail
        "per_condition_results": per_condition,
    }

    with open(out_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\n[V3-EXQ-258] Output written: {out_path}", flush=True)
    print(f"verdict: {'PASS' if outcome == 'PASS' else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
