#!/opt/local/bin/python3
"""
V3-EXQ-227 -- ARC-029: Discrepancy Diagnostic (EXQ-063 vs EXQ-125)

Claims: ARC-029
EXPERIMENT_PURPOSE = "diagnostic"

Scientific question: Why did EXQ-063 PASS (committed mode reduces harm in stable
env) while EXQ-125 FAIL on matched seeds? Reproduce both condition sets on matched
seeds to identify the confound.

Context:
  EXQ-063 PASS (conf=0.774): 5/5 criteria, 2 seeds [0,1].
  EXQ-125 FAIL (3rd FAIL): same design but seeds [42,123] and stricter thresholds
  (MIN_HARM_GAP_STABLE=0.0001, VOLATILITY_MODULATION_RATIO=0.90).
  Governance noted "systematic discrepancy unexplained."

  Hypotheses:
    H1: Seed dependence -- seeds [0,1] produce committed collapse; [42,123] do not.
    H2: Threshold strictness -- 063 used bare > 0; 125 used >= 0.0001.
    H3: Both: gap is positive but tiny (< 0.0001) in 063 conditions.

Design:
  Run BOTH condition sets:
  (A) EXQ-063 exact conditions: seeds=[0,1], 400 warmup, 50 eval, 200 steps,
      size=12, hazards=4, resources=5, hazard_harm=0.02, alpha_world=0.9,
      world_dim=32, self_dim=32, reafference_action_dim=action_dim.
      PASS criterion: harm_gap_stable > 0 (bare > 0, EXQ-063 standard).
  (B) EXQ-125 exact conditions: seeds=[42,123], 400 warmup, 50 eval, 200 steps,
      same env, same arch.
      PASS criterion: harm_gap_stable >= 0.0001 (EXQ-125 stricter threshold).

  For each set: train 1 agent per seed, eval 2x2 (stable/volatile x committed/ablated).
  Diagnostic metrics (all seeds):
    - harm_gap_stable: committed - ablated mean_harm in stable env
    - harm_gap_volatile: same in volatile env
    - gap_reduction_ratio: volatile_gap / stable_gap
    - final_running_variance vs commit_threshold (did agent commit?)

PASS: EXQ-063 conditions reproduce PASS result (harm_gap_stable > 0 in >= 1/2 seeds
      AND committed_steps > uncommitted_steps in committed condition).
FAIL: EXQ-063 conditions do not reproduce -> confound identified.

Note: This is a diagnostic, not a standalone ARC-029 evidence experiment.
      The primary purpose is to understand the discrepancy.
"""

import sys
import random
import json
import time
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
EXPERIMENT_TYPE = "v3_exq_227_arc029_discrepancy_diagnostic"
CLAIM_IDS = ["ARC-029"]
EXPERIMENT_PURPOSE = "diagnostic"

# ---------------------------------------------------------------------------
# EXQ-063 / EXQ-125 shared env kwargs
# ---------------------------------------------------------------------------
TRAIN_ENV_KWARGS = dict(
    size=12, num_hazards=4, num_resources=5,
    hazard_harm=0.02,
    env_drift_interval=5, env_drift_prob=0.1,
    proximity_harm_scale=0.05,
    proximity_benefit_scale=0.03,
    proximity_approach_threshold=0.15,
    hazard_field_decay=0.5,
)

STABLE_ENV_KWARGS = dict(
    size=12, num_hazards=4, num_resources=5,
    hazard_harm=0.02,
    env_drift_interval=50, env_drift_prob=0.0,
    proximity_harm_scale=0.05,
    proximity_benefit_scale=0.03,
    proximity_approach_threshold=0.15,
    hazard_field_decay=0.5,
)

VOLATILE_ENV_KWARGS = dict(
    size=12, num_hazards=4, num_resources=5,
    hazard_harm=0.02,
    env_drift_interval=3, env_drift_prob=0.4,
    proximity_harm_scale=0.05,
    proximity_benefit_scale=0.03,
    proximity_approach_threshold=0.15,
    hazard_field_decay=0.5,
)

# EXQ-063 exact conditions
EXQ063_SEEDS       = [0, 1]
EXQ063_WARMUP      = 400
EXQ063_EVAL        = 50
EXQ063_STEPS       = 200
EXQ063_WORLD_DIM   = 32
EXQ063_SELF_DIM    = 32
EXQ063_ALPHA_WORLD = 0.9

# EXQ-125 exact conditions (same arch, different seeds + stricter threshold)
EXQ125_SEEDS       = [42, 123]
EXQ125_MIN_GAP     = 0.0001   # C1 from EXQ-125
EXQ125_MODULATION  = 0.90     # C3 from EXQ-125


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _mean_safe(lst: List[float]) -> float:
    return float(sum(lst) / len(lst)) if lst else 0.0


def _make_agent(
    seed: int,
    self_dim: int,
    world_dim: int,
    alpha_world: float,
    env: CausalGridWorldV2,
) -> REEAgent:
    torch.manual_seed(seed)
    random.seed(seed)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
    )
    return REEAgent(config)


def _train_agent(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
) -> Dict:
    """Train agent; track running_variance to detect committed collapse."""
    agent.train()
    optimizer = optim.Adam(list(agent.e1.parameters()), lr=1e-3)
    wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()),
        lr=1e-3,
    )
    harm_eval_opt = optim.Adam(list(agent.e3.harm_eval_head.parameters()), lr=1e-4)

    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    variance_history: List[float] = []

    for ep in range(num_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor]  = None

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)

            ticks   = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            theta_z    = agent.theta_buffer.summary()
            z_world_curr = latent.z_world.detach()

            action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = _onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                )
                agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > 2000:
                    wf_buf = wf_buf[-2000:]

            (harm_buf_pos if harm_signal < 0 else harm_buf_neg).append(theta_z.detach())
            if len(harm_buf_pos) > 1000: harm_buf_pos = harm_buf_pos[-1000:]
            if len(harm_buf_neg) > 1000: harm_buf_neg = harm_buf_neg[-1000:]

            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                optimizer.zero_grad()
                e1_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                optimizer.step()

            if len(wf_buf) >= 16:
                k = min(32, len(wf_buf))
                idxs = torch.randperm(len(wf_buf))[:k].tolist()
                zw_b  = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
                a_b   = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
                wf_pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(wf_pred, zw1_b)
                if wf_loss.requires_grad:
                    wf_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters()) +
                        list(agent.e2.world_action_encoder.parameters()), 1.0,
                    )
                    wf_optimizer.step()
                with torch.no_grad():
                    agent.e3.update_running_variance((wf_pred.detach() - zw1_b).detach())

            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_p = min(16, len(harm_buf_pos))
                k_n = min(16, len(harm_buf_neg))
                pi = torch.randperm(len(harm_buf_pos))[:k_p].tolist()
                ni = torch.randperm(len(harm_buf_neg))[:k_n].tolist()
                zw_b = torch.cat(
                    [harm_buf_pos[i] for i in pi] + [harm_buf_neg[i] for i in ni], dim=0
                )
                tgt = torch.cat([
                    torch.ones(k_p, 1, device=agent.device),
                    torch.zeros(k_n, 1, device=agent.device),
                ], dim=0)
                pred_h = agent.e3.harm_eval(zw_b)
                hloss  = F.mse_loss(pred_h, tgt)
                if hloss.requires_grad:
                    harm_eval_opt.zero_grad()
                    hloss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e3.harm_eval_head.parameters(), 0.5)
                    harm_eval_opt.step()

            z_world_prev = z_world_curr
            action_prev  = action.detach()
            if done:
                break

        rv = float(agent.e3._running_variance)
        variance_history.append(rv)

        if (ep + 1) % 100 == 0 or ep == num_episodes - 1:
            committed = rv < agent.e3.commit_threshold
            print(
                f"  [train] ep {ep+1}/{num_episodes}"
                f" running_var={rv:.6f} committed={committed}",
                flush=True,
            )

    return {
        "final_running_variance": float(agent.e3._running_variance),
        "commit_threshold":       float(agent.e3.commit_threshold),
        "committed":              float(agent.e3._running_variance) < float(agent.e3.commit_threshold),
        "variance_history":       variance_history[-10:],
    }


def _eval_condition(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    ablated: bool,
    label: str,
    train_variance: float,
) -> Dict:
    agent.eval()

    harm_signals: List[float] = []
    n_committed   = 0
    n_uncommitted = 0
    fatal         = 0

    for _ in range(num_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        agent.e3._running_variance = train_variance

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)

            # Ablation: force uncommitted state before SELECT
            if ablated:
                agent.e3._running_variance = agent.e3.commit_threshold + 0.1
                agent.e3._committed_trajectory = None

            try:
                with torch.no_grad():
                    ticks    = agent.clock.advance()
                    e1_prior = (
                        agent._e1_tick(latent) if ticks.get("e1_tick", False)
                        else torch.zeros(1, world_dim, device=agent.device)
                    )
                    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                    action     = agent.select_action(candidates, ticks, temperature=1.0)

                if action is None:
                    action = _onehot(
                        random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                    )
                    agent._last_action = action

                _, harm_signal, done, info, obs_dict = env.step(action)
                is_committed = agent.e3._running_variance < agent.e3.commit_threshold
                if is_committed:
                    n_committed += 1
                else:
                    n_uncommitted += 1
                harm_signals.append(float(harm_signal))

            except Exception as exc:
                fatal += 1
                _, obs_dict = env.reset()
                done = True

            if done:
                break

    mean_harm = _mean_safe(harm_signals)
    print(
        f"  [{label}] mean_harm={mean_harm:.5f}"
        f" committed={n_committed} uncommitted={n_uncommitted} fatal={fatal}",
        flush=True,
    )
    return {
        "mean_harm_per_step": mean_harm,
        "n_committed":        n_committed,
        "n_uncommitted":      n_uncommitted,
        "fatal_errors":       fatal,
    }


def _run_condition_set(
    seeds: List[int],
    set_label: str,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    self_dim: int,
    alpha_world: float,
) -> Dict:
    """Run one complete condition set (all seeds, all 4 eval conditions)."""
    results_by_seed = []

    for seed in seeds:
        print(f"\n{'='*60}", flush=True)
        print(f"[V3-EXQ-227] {set_label} Seed {seed}", flush=True)
        print('='*60, flush=True)

        train_env = CausalGridWorldV2(seed=seed, **TRAIN_ENV_KWARGS)
        agent     = _make_agent(seed, self_dim, world_dim, alpha_world, train_env)

        print(f"  [training] {set_label} seed={seed} warmup={warmup_episodes}", flush=True)
        train_out = _train_agent(agent, train_env, warmup_episodes, steps_per_episode, world_dim)
        train_var = train_out["final_running_variance"]

        print(
            f"  post-train running_variance={train_var:.6f}"
            f" commit_threshold={train_out['commit_threshold']:.4f}"
            f" committed={train_out['committed']}",
            flush=True,
        )

        if not train_out["committed"]:
            print(
                f"  WARNING: seed={seed} not collapsed to committed state --"
                f" gate-active vs ablated comparison may be uninformative.",
                flush=True,
            )

        stable_env   = CausalGridWorldV2(seed=seed + 500, **STABLE_ENV_KWARGS)
        volatile_env = CausalGridWorldV2(seed=seed + 500, **VOLATILE_ENV_KWARGS)

        seed_res = {}
        for env_name, env_obj in [("stable", stable_env), ("volatile", volatile_env)]:
            for gate_label, ablated in [("gate_active", False), ("gate_ablated", True)]:
                lbl = f"{set_label}_seed{seed}_{env_name}_{gate_label}"
                seed_res[f"{env_name}_{gate_label}"] = _eval_condition(
                    agent=agent,
                    env=env_obj,
                    num_episodes=eval_episodes,
                    steps_per_episode=steps_per_episode,
                    world_dim=world_dim,
                    ablated=ablated,
                    label=lbl,
                    train_variance=train_var,
                )

        seed_res["train"] = train_out
        seed_res["seed"]  = seed
        results_by_seed.append(seed_res)

    # Aggregate
    def _avg(key: str, subkey: str) -> float:
        vals = [r[key][subkey] for r in results_by_seed]
        return float(sum(vals) / len(vals))

    harm_active_stable    = _avg("stable_gate_active",   "mean_harm_per_step")
    harm_ablated_stable   = _avg("stable_gate_ablated",  "mean_harm_per_step")
    harm_active_volatile  = _avg("volatile_gate_active", "mean_harm_per_step")
    harm_ablated_volatile = _avg("volatile_gate_ablated","mean_harm_per_step")

    harm_gap_stable   = harm_active_stable   - harm_ablated_stable
    harm_gap_volatile = harm_active_volatile - harm_ablated_volatile

    gap_reduction_ratio = (
        harm_gap_volatile / harm_gap_stable
        if abs(harm_gap_stable) > 1e-9 else float("nan")
    )

    n_committed_active_stable   = _avg("stable_gate_active",  "n_committed")
    n_uncommitted_active_stable = _avg("stable_gate_active",  "n_uncommitted")
    n_committed_ablated_stable  = _avg("stable_gate_ablated", "n_committed")

    print(
        f"\n  [{set_label}] harm_gap_stable={harm_gap_stable:.6f}"
        f" harm_gap_volatile={harm_gap_volatile:.6f}"
        f" gap_reduction_ratio={gap_reduction_ratio:.4f}",
        flush=True,
    )
    print(
        f"  [{set_label}] committed_active={n_committed_active_stable:.0f}"
        f" uncommitted_active={n_uncommitted_active_stable:.0f}"
        f" committed_ablated={n_committed_ablated_stable:.0f}",
        flush=True,
    )

    return {
        "set_label":                    set_label,
        "seeds":                        seeds,
        "harm_gap_stable":              harm_gap_stable,
        "harm_gap_volatile":            harm_gap_volatile,
        "gap_reduction_ratio":          gap_reduction_ratio,
        "harm_active_stable":           harm_active_stable,
        "harm_ablated_stable":          harm_ablated_stable,
        "n_committed_active_stable":    n_committed_active_stable,
        "n_uncommitted_active_stable":  n_uncommitted_active_stable,
        "n_committed_ablated_stable":   n_committed_ablated_stable,
        "results_by_seed":              results_by_seed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    warmup     = 2     if args.dry_run else EXQ063_WARMUP
    eval_eps   = 2     if args.dry_run else EXQ063_EVAL
    steps      = 20    if args.dry_run else EXQ063_STEPS
    seeds_063  = [0]   if args.dry_run else EXQ063_SEEDS
    seeds_125  = [42]  if args.dry_run else EXQ125_SEEDS

    print(f"[V3-EXQ-227] ARC-029 Discrepancy Diagnostic dry_run={args.dry_run}", flush=True)
    print(
        f"  Reproducing EXQ-063 (seeds={seeds_063}) and EXQ-125 (seeds={seeds_125})",
        flush=True,
    )

    # ---- Condition set A: EXQ-063 exact conditions ----
    print(f"\n{'='*70}", flush=True)
    print("[V3-EXQ-227] CONDITION SET A -- EXQ-063 EXACT CONDITIONS", flush=True)
    print('='*70, flush=True)
    set_a = _run_condition_set(
        seeds=seeds_063,
        set_label="SET_A_EXQ063",
        warmup_episodes=warmup,
        eval_episodes=eval_eps,
        steps_per_episode=steps,
        world_dim=EXQ063_WORLD_DIM,
        self_dim=EXQ063_SELF_DIM,
        alpha_world=EXQ063_ALPHA_WORLD,
    )

    # ---- Condition set B: EXQ-125 exact conditions ----
    print(f"\n{'='*70}", flush=True)
    print("[V3-EXQ-227] CONDITION SET B -- EXQ-125 EXACT CONDITIONS", flush=True)
    print('='*70, flush=True)
    set_b = _run_condition_set(
        seeds=seeds_125,
        set_label="SET_B_EXQ125",
        warmup_episodes=warmup,
        eval_episodes=eval_eps,
        steps_per_episode=steps,
        world_dim=EXQ063_WORLD_DIM,
        self_dim=EXQ063_SELF_DIM,
        alpha_world=EXQ063_ALPHA_WORLD,
    )

    # ---- Evaluate ----
    # SET A PASS: harm_gap_stable > 0 (bare > 0, EXQ-063 criterion)
    # AND gate is active in committed condition
    a_gap_positive = set_a["harm_gap_stable"] > 0.0
    a_gate_active  = (
        set_a["n_committed_active_stable"] > set_a["n_uncommitted_active_stable"]
    )
    set_a_reproduces = a_gap_positive and a_gate_active

    # SET B: check against EXQ-125 stricter criteria
    b_gap_ok  = set_b["harm_gap_stable"] >= EXQ125_MIN_GAP
    b_ratio_ok = (
        set_b["gap_reduction_ratio"] <= EXQ125_MODULATION
        if not (set_b["gap_reduction_ratio"] != set_b["gap_reduction_ratio"])  # NaN check
        else False
    )
    set_b_passes_125 = b_gap_ok and b_ratio_ok

    if set_a_reproduces:
        outcome   = "PASS"
        direction = "supports"
        note      = "EXQ-063 conditions reproduce PASS result -- seed dependence confirmed."
    else:
        outcome   = "FAIL"
        direction = "non_contributory"
        note      = "EXQ-063 conditions do NOT reproduce -- possible version/state drift."

    # Diagnose discrepancy
    if set_a_reproduces and not set_b_passes_125:
        discrepancy_diagnosis = "seed_dependence: SET_A (seeds 0,1) reproduced but SET_B (seeds 42,123) failed EXQ-125 criteria"
    elif not set_a_reproduces:
        discrepancy_diagnosis = "baseline_drift: EXQ-063 conditions no longer reproduce PASS -- architecture or env has changed"
    else:
        discrepancy_diagnosis = "no_discrepancy: both sets agree"

    print(f"\n[V3-EXQ-227] Diagnostic Summary:", flush=True)
    print(
        f"  SET_A (EXQ-063): harm_gap_stable={set_a['harm_gap_stable']:.6f}"
        f" reproduces={set_a_reproduces}",
        flush=True,
    )
    print(
        f"  SET_B (EXQ-125): harm_gap_stable={set_b['harm_gap_stable']:.6f}"
        f" passes_125={set_b_passes_125}",
        flush=True,
    )
    print(f"  discrepancy_diagnosis: {discrepancy_diagnosis}", flush=True)
    print(f"  -> {outcome}", flush=True)
    print(f"  {note}", flush=True)

    if args.dry_run:
        print("\n[DRY-RUN] Smoke test complete. Not writing output.", flush=True)
        return

    ts      = int(time.time())
    root    = Path(__file__).resolve().parents[2]
    out_dir = root / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"

    # Serialise results_by_seed (remove non-JSON-serialisable tensors)
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(x) for x in obj]
        if isinstance(obj, (torch.Tensor,)):
            return float(obj.item())
        try:
            import math
            if math.isnan(obj):
                return None
        except (TypeError, ValueError):
            pass
        return obj

    manifest = {
        "run_id":                f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type":       EXPERIMENT_TYPE,
        "architecture_epoch":    "ree_hybrid_guardrails_v1",
        "claim_ids":             CLAIM_IDS,
        "experiment_purpose":    EXPERIMENT_PURPOSE,
        "outcome":               outcome,
        "evidence_direction":    direction,
        "timestamp":             ts,
        "discrepancy_diagnosis": discrepancy_diagnosis,
        "note":                  note,
        # SET A
        "set_a_label":           set_a["set_label"],
        "set_a_seeds":           set_a["seeds"],
        "set_a_harm_gap_stable": float(set_a["harm_gap_stable"]),
        "set_a_harm_gap_volatile": float(set_a["harm_gap_volatile"]),
        "set_a_reproduces":      set_a_reproduces,
        "set_a_gap_positive":    a_gap_positive,
        "set_a_gate_active":     a_gate_active,
        # SET B
        "set_b_label":           set_b["set_label"],
        "set_b_seeds":           set_b["seeds"],
        "set_b_harm_gap_stable": float(set_b["harm_gap_stable"]),
        "set_b_harm_gap_volatile": float(set_b["harm_gap_volatile"]),
        "set_b_passes_125":      set_b_passes_125,
        "set_b_gap_ok":          b_gap_ok,
        "set_b_ratio_ok":        b_ratio_ok,
        # Thresholds
        "exq063_min_gap":        0.0,
        "exq125_min_gap":        EXQ125_MIN_GAP,
        "exq125_modulation":     EXQ125_MODULATION,
        # Per-seed detail (cleaned for JSON)
        "set_a_per_seed":        _clean(set_a["results_by_seed"]),
        "set_b_per_seed":        _clean(set_b["results_by_seed"]),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[V3-EXQ-227] Written: {out_path}", flush=True)
    print(f"Status: {outcome}", flush=True)


if __name__ == "__main__":
    main()
