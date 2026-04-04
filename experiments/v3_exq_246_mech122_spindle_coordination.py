#!/opt/local/bin/python3
"""
V3-EXQ-246 -- MECH-122 Phase 3 Spindle Coordination Probe

Claims: MECH-122

Scientific question:
  Does Phase 3 spindle coordination (ThetaBuffer consolidation mode: theta-
  packaged E1 updates transferred back to ContextMemory) produce measurably
  better next-waking-cycle performance when added after correct SWS+REM phases?

MECH-122 asserts: Thalamo-cortical spindle bursts package E1 updates during
consolidation for long-horizon integration (bidirectional ThetaBuffer mode).
In Phase 3, theta-packaged E1 estimates flow in the REVERSE direction
(cx->hip: cortex to hippocampus), updating hippocampal context memory with
recency-weighted world state summaries. This enables the next waking cycle to
start with context that reflects BOTH the forward (waking) and reverse
(consolidation) theta-packaged experience.

offline_phases.md Phase 3 specification:
  "After schema is installed, bidirectional packaging transfers E1 updates
  for long-horizon integration. ThetaBuffer gains reverse-direction mode."

V3 PROXY (V4 full bidirectional ThetaBuffer is V4 scope):
  Phase 3 proxy: after SWS+REM, enter ThetaBuffer consolidation mode,
  compute consolidation_summary() (recency-weighted z_world), and write
  this consolidated summary back to ContextMemory as a Phase 3 "spindle
  transfer" slot. This represents E1 updates being theta-packaged for
  hippocampal re-integration.

Design:
  Three conditions, 3 seeds, 150 training episodes.

  WAKING_ONLY:
    No offline phases. Baseline performance.

  SWS_REM_NO_SPINDLE:
    After every SLEEP_INTERVAL: SWS (schema slots) + REM (attribution replay).
    NO Phase 3 spindle coordination. Tests whether SWS+REM alone asymptotes.

  SWS_REM_WITH_SPINDLE:
    After every SLEEP_INTERVAL: SWS + REM + Phase 3 proxy.
    Phase 3: theta_buffer.set_consolidation_mode(True), compute
    consolidation_summary() (recency-weighted recent z_world), write
    back to ContextMemory as consolidated world state slot.
    Prediction: slightly better context discrimination because the
    consolidated world state reinforces the most recent (behaviorally
    relevant) world state in ContextMemory alongside the SWS-installed
    contrastive slots.

Metrics:
  - harm_rate_dangerous: harm rate in DANGEROUS context (lower = better)
  - harm_discrimination: E3 harm_eval gap between SAFE and DANGEROUS
  - slot_cosine_sim: ContextMemory differentiation (lower = better)

PASS criteria:
  T1: harm_rate_dangerous(SWS_REM_WITH_SPINDLE) <=
      harm_rate_dangerous(SWS_REM_NO_SPINDLE) * 1.05 in >=2/3 seeds
      (Phase 3 does not worsen harm avoidance -- safety check)
  T2: harm_discrimination(SWS_REM_WITH_SPINDLE) >=
      harm_discrimination(SWS_REM_NO_SPINDLE) in >=2/3 seeds
      (Phase 3 maintains or improves context discrimination)

  NOTE: A PASS here is deliberately weak because MECH-122 is a V3 PROXY
  of a V4 mechanism. We expect Phase 3 to matter MORE in V4 with full
  bidirectional ThetaBuffer. In V3, the proxy is expected to produce
  modest benefit or neutral effect. The experiment establishes a baseline
  and tests that the proxy does not degrade performance.

PASS: T1 AND T2

experiment_purpose: "diagnostic"
  Rationale: MECH-122 is V3 scope per ree-v3/CLAUDE.md, but the full
  bidirectional ThetaBuffer mode is V4. This experiment tests a PROXY
  (consolidation_summary -> ContextMemory write). Diagnostic: establishes
  whether the Phase 3 proxy adds value at all, or whether SWS+REM
  already saturates the achievable benefit.

claim_ids: ["MECH-122"]
experiment_purpose: "diagnostic"
"""

import sys
import random
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_246_mech122_spindle_coordination"
CLAIM_IDS = ["MECH-122"]
EXPERIMENT_PURPOSE = "diagnostic"

SLEEP_INTERVAL = 10
SWS_WRITE_STEPS = 20
CONTEXT_SWITCH_EVERY = 5

TRAINING_EPISODES = 150
STEPS_PER_EPISODE = 150
EVAL_EPISODES_EACH = 20
SEEDS = [42, 7, 13]

CONDITIONS = ["WAKING_ONLY", "SWS_REM_NO_SPINDLE", "SWS_REM_WITH_SPINDLE"]


# ------------------------------------------------------------------ #
# Environment / agent helpers                                          #
# ------------------------------------------------------------------ #

def _make_env_safe(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed, size=10,
        num_hazards=1, num_resources=4,
        hazard_harm=0.02,
        env_drift_interval=50, env_drift_prob=0.05,
        proximity_harm_scale=0.10, proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5, energy_decay=0.005,
        use_proxy_fields=True, resource_respawn_on_consume=True,
    )


def _make_env_dangerous(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed + 1000, size=10,
        num_hazards=8, num_resources=4,
        hazard_harm=0.05,
        env_drift_interval=50, env_drift_prob=0.05,
        proximity_harm_scale=0.15, proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5, energy_decay=0.005,
        use_proxy_fields=True, resource_respawn_on_consume=True,
    )


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32, world_dim=32,
        alpha_world=0.9, alpha_self=0.3,
        reafference_action_dim=0,
        novelty_bonus_weight=0.0,
    )
    return REEAgent(config)


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _compute_slot_cosine_sim(agent: REEAgent) -> float:
    with torch.no_grad():
        mem = agent.e1.context_memory.memory
        n = mem.shape[0]
        norms = F.normalize(mem, dim=-1)
        sim_matrix = torch.mm(norms, norms.t())
        mask = ~torch.eye(n, dtype=torch.bool, device=mem.device)
        return sim_matrix[mask].mean().item()


# ------------------------------------------------------------------ #
# Sleep phase implementations                                          #
# ------------------------------------------------------------------ #

def _sws_pass(
    agent: REEAgent,
    safe_z_buf: List[torch.Tensor],
    dang_z_buf: List[torch.Tensor],
    self_dim: int,
    n_writes: int,
) -> None:
    """SWS-analog: alternating SAFE/DANGEROUS context_memory writes."""
    n = min(len(safe_z_buf), len(dang_z_buf), n_writes)
    if n < 2:
        return
    device = safe_z_buf[0].device
    z_self_zeros = torch.zeros(1, self_dim, device=device)
    for i in range(n):
        state_safe = torch.cat([z_self_zeros, safe_z_buf[-(i + 1)]], dim=-1)
        state_dang = torch.cat([z_self_zeros, dang_z_buf[-(i + 1)]], dim=-1)
        agent.e1.context_memory.write(state_safe)
        agent.e1.context_memory.write(state_dang)


def _rem_pass(
    agent: REEAgent,
    dang_harm_traj: List[Tuple[torch.Tensor, float]],
    safe_z_recent: List[torch.Tensor],
    harm_eval_opt,
) -> None:
    """REM-analog: DANGEROUS harm attribution replay + SAFE false-positive suppression."""
    if len(dang_harm_traj) >= 4:
        n = min(len(dang_harm_traj), 64)
        recent_dang = dang_harm_traj[-n:]
        zw_d = torch.cat([zw for (zw, _) in recent_dang], dim=0)
        tgt_d = torch.tensor(
            [[float(lbl)] for (_, lbl) in recent_dang],
            dtype=torch.float, device=agent.device
        )
        pred_d = agent.e3.harm_eval_head(zw_d)
        loss_d = F.binary_cross_entropy_with_logits(pred_d, tgt_d)
        harm_eval_opt.zero_grad()
        loss_d.backward()
        harm_eval_opt.step()

    if len(safe_z_recent) >= 4:
        n_s = min(len(safe_z_recent), 32)
        zw_s = torch.cat(safe_z_recent[-n_s:], dim=0)
        tgt_s = torch.zeros(zw_s.shape[0], 1, device=agent.device)
        pred_s = agent.e3.harm_eval_head(zw_s)
        loss_s = F.binary_cross_entropy_with_logits(pred_s, tgt_s)
        harm_eval_opt.zero_grad()
        loss_s.backward()
        harm_eval_opt.step()

    agent.residue_field.integrate(num_steps=5)


def _phase3_spindle_pass(
    agent: REEAgent,
    self_dim: int,
) -> Dict:
    """
    Phase 3 (MECH-122) spindle coordination proxy.

    Enters ThetaBuffer consolidation mode, computes recency-weighted z_world
    summary (consolidation_summary()), and writes it back to ContextMemory
    as a consolidated world state slot.

    This represents: E1 waking updates -> theta-packaged -> hippocampal
    context memory (cx->hip direction in bidirectional mode).
    """
    agent.theta_buffer.set_consolidation_mode(True)
    cs = agent.theta_buffer.consolidation_summary()
    agent.theta_buffer.set_consolidation_mode(False)

    if cs is None:
        return {"phase3_skip": True, "cs_norm": 0.0}

    # Write consolidation summary back to ContextMemory as a spindle-transferred slot
    z_self_zeros = torch.zeros(1, self_dim, device=agent.device)
    state_consolidated = torch.cat([z_self_zeros, cs], dim=-1)
    agent.e1.context_memory.write(state_consolidated)

    return {
        "phase3_skip": False,
        "cs_norm": float(cs.norm().item()),
    }


# ------------------------------------------------------------------ #
# Episode runner                                                       #
# ------------------------------------------------------------------ #

def _run_episode(
    agent: REEAgent,
    env: CausalGridWorldV2,
    steps: int,
    train: bool,
    optimizer,
    harm_eval_opt,
    harm_buf_pos: List,
    harm_buf_neg: List,
) -> Tuple[float, List[torch.Tensor], List[Tuple[torch.Tensor, float]]]:
    """Run single episode. Returns (harm_sum, z_world_list, harm_events)."""
    _, obs_dict = env.reset()
    agent.reset()
    ep_harm = 0.0
    z_world_list: List[torch.Tensor] = []
    harm_events: List[Tuple[torch.Tensor, float]] = []

    for _step in range(steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        obs_harm = obs_dict.get("harm_obs", None)

        latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
        agent.clock.advance()
        z_world = latent.z_world.detach().clone()
        z_world_list.append(z_world)

        with torch.no_grad():
            best_action = 0
            best_score = float("inf")
            for idx in range(env.action_dim):
                a_oh = _action_to_onehot(idx, env.action_dim, agent.device)
                z_next = agent.e2.world_forward(z_world, a_oh)
                score = agent.e3.harm_eval(z_next).mean().item()
                if score < best_score:
                    best_score = score
                    best_action = idx

        action_oh = _action_to_onehot(best_action, env.action_dim, agent.device)
        agent._last_action = action_oh

        _, harm_signal, done, _info, obs_dict = env.step(action_oh)
        is_harm = float(harm_signal) < 0

        harm_events.append((z_world, 1.0 if is_harm else 0.0))

        if is_harm:
            ep_harm += abs(float(harm_signal))

        if train:
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total = e1_loss + e2_loss
            if total.requires_grad:
                optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if is_harm:
                harm_buf_pos.append(z_world)
            else:
                harm_buf_neg.append(z_world)

            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()
                zw_pos = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
                zw_neg = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)
                zw_b = torch.cat([zw_pos, zw_neg], dim=0)
                target = torch.cat([
                    torch.ones(k_pos, 1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                ], dim=0)
                pred = agent.e3.harm_eval_head(zw_b)
                harm_loss = F.binary_cross_entropy_with_logits(pred, target)
                harm_eval_opt.zero_grad()
                harm_loss.backward()
                harm_eval_opt.step()

        if done:
            break

    return ep_harm, z_world_list, harm_events


# ------------------------------------------------------------------ #
# Condition runner                                                     #
# ------------------------------------------------------------------ #

def _run_condition(
    seed: int,
    condition: str,
    training_episodes: int,
    steps_per_episode: int,
    eval_episodes_each: int,
    verbose: bool = True,
) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env_safe = _make_env_safe(seed)
    env_dang = _make_env_dangerous(seed)
    agent = _make_agent(env_safe)

    standard_params = [p for n, p in agent.named_parameters()
                       if "harm_eval_head" not in n]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    optimizer = optim.Adam(standard_params, lr=1e-3)
    harm_eval_opt = optim.Adam(harm_eval_params, lr=1e-4)

    self_dim = agent.config.latent.self_dim
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    safe_z_buf: List[torch.Tensor] = []
    dang_z_buf: List[torch.Tensor] = []
    dang_harm_traj: List[Tuple[torch.Tensor, float]] = []
    sleep_log: List[Dict] = []

    agent.train()

    for ep in range(training_episodes):
        block = ep // CONTEXT_SWITCH_EVERY
        is_safe_ep = (block % 2 == 0)
        env = env_safe if is_safe_ep else env_dang

        ep_harm, z_list, harm_events = _run_episode(
            agent, env, steps_per_episode,
            train=True,
            optimizer=optimizer,
            harm_eval_opt=harm_eval_opt,
            harm_buf_pos=harm_buf_pos,
            harm_buf_neg=harm_buf_neg,
        )

        if is_safe_ep:
            safe_z_buf.extend(z_list)
            if len(safe_z_buf) > 2000:
                safe_z_buf = safe_z_buf[-2000:]
        else:
            dang_z_buf.extend(z_list)
            if len(dang_z_buf) > 2000:
                dang_z_buf = dang_z_buf[-2000:]
            dang_harm_traj.extend(harm_events)
            if len(dang_harm_traj) > 5000:
                dang_harm_traj = dang_harm_traj[-5000:]

        if len(harm_buf_pos) > 4000:
            harm_buf_pos = harm_buf_pos[-4000:]
        if len(harm_buf_neg) > 4000:
            harm_buf_neg = harm_buf_neg[-4000:]

        if condition != "WAKING_ONLY" and (ep + 1) % SLEEP_INTERVAL == 0:
            if len(safe_z_buf) >= 2 and len(dang_z_buf) >= 2:
                # SWS + REM (common to both non-waking conditions)
                _sws_pass(agent, safe_z_buf, dang_z_buf, self_dim, SWS_WRITE_STEPS)
                _rem_pass(
                    agent, dang_harm_traj,
                    safe_z_buf[-100:], harm_eval_opt,
                )

                phase3_m = {}
                if condition == "SWS_REM_WITH_SPINDLE":
                    # Phase 3 proxy: consolidation_summary -> ContextMemory
                    phase3_m = _phase3_spindle_pass(agent, self_dim)

                sleep_log.append({"ep": ep, **phase3_m})

    final_slot_sim = _compute_slot_cosine_sim(agent)

    # Evaluation
    agent.eval()
    eval_harm_safe: List[float] = []
    eval_harm_dang: List[float] = []
    eval_z_safe: List[torch.Tensor] = []
    eval_z_dang: List[torch.Tensor] = []

    for _ in range(eval_episodes_each):
        harm_s, z_s, _ = _run_episode(
            agent, env_safe, steps_per_episode,
            train=False, optimizer=optimizer,
            harm_eval_opt=harm_eval_opt,
            harm_buf_pos=harm_buf_pos,
            harm_buf_neg=harm_buf_neg,
        )
        eval_harm_safe.append(harm_s / steps_per_episode)
        eval_z_safe.extend(z_s)

    for _ in range(eval_episodes_each):
        harm_d, z_d, _ = _run_episode(
            agent, env_dang, steps_per_episode,
            train=False, optimizer=optimizer,
            harm_eval_opt=harm_eval_opt,
            harm_buf_pos=harm_buf_pos,
            harm_buf_neg=harm_buf_neg,
        )
        eval_harm_dang.append(harm_d / steps_per_episode)
        eval_z_dang.extend(z_d)

    with torch.no_grad():
        n_samp = min(len(eval_z_safe), len(eval_z_dang), 200)
        z_s_samp = torch.cat(eval_z_safe[:n_samp], dim=0)
        z_d_samp = torch.cat(eval_z_dang[:n_samp], dim=0)
        he_safe = agent.e3.harm_eval(z_s_samp).mean().item()
        he_dang = agent.e3.harm_eval(z_d_samp).mean().item()
        harm_discrimination = he_dang - he_safe

    harm_rate_safe = sum(eval_harm_safe) / len(eval_harm_safe)
    harm_rate_dang = sum(eval_harm_dang) / len(eval_harm_dang)

    if verbose:
        print(f"  [seed={seed} {condition}] "
              f"slot_sim={final_slot_sim:.4f} "
              f"harm_safe={harm_rate_safe:.4f} "
              f"harm_dang={harm_rate_dang:.4f} "
              f"discrim={harm_discrimination:.4f} "
              f"sleep_passes={len(sleep_log)}")

    return {
        "seed": seed,
        "condition": condition,
        "slot_cosine_sim": float(final_slot_sim),
        "harm_rate_safe": float(harm_rate_safe),
        "harm_rate_dangerous": float(harm_rate_dang),
        "harm_discrimination": float(harm_discrimination),
        "harm_eval_safe": float(he_safe),
        "harm_eval_dangerous": float(he_dang),
        "sleep_passes": len(sleep_log),
    }


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.dry_run:
        print("Smoke test: seed=42, 3 train episodes + 2 eval each condition")
        for cond in CONDITIONS:
            print(f"  Testing condition: {cond}")
            result = _run_condition(
                seed=42, condition=cond,
                training_episodes=3,
                steps_per_episode=30,
                eval_episodes_each=2,
                verbose=False,
            )
            print(f"    slot_sim={result['slot_cosine_sim']:.4f} "
                  f"harm_dang={result['harm_rate_dangerous']:.4f} "
                  f"discrim={result['harm_discrimination']:.4f} "
                  f"sleep_passes={result['sleep_passes']}")
        print("Smoke test PASSED")
        return

    ts_str = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts_str}_v3"

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        script_dir = Path(__file__).resolve().parents[1]
        out_dir = (script_dir.parent / "REE_assembly" / "evidence"
                   / "experiments" / EXPERIMENT_TYPE)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for seed in SEEDS:
        print(f"\nSeed {seed}")
        for cond in CONDITIONS:
            r = _run_condition(
                seed=seed, condition=cond,
                training_episodes=TRAINING_EPISODES,
                steps_per_episode=STEPS_PER_EPISODE,
                eval_episodes_each=EVAL_EPISODES_EACH,
            )
            all_results.append(r)

    def by_cond(c):
        return [r for r in all_results if r["condition"] == c]

    no_spindle  = by_cond("SWS_REM_NO_SPINDLE")
    with_spindle = by_cond("SWS_REM_WITH_SPINDLE")

    # T1: WITH_SPINDLE harm_rate_dangerous <= NO_SPINDLE * 1.05 in >=2/3 seeds
    t1_wins = sum(1 for a, b in zip(no_spindle, with_spindle)
                  if b["harm_rate_dangerous"] <= a["harm_rate_dangerous"] * 1.05)
    t1 = t1_wins >= 2

    # T2: WITH_SPINDLE harm_discrimination >= NO_SPINDLE in >=2/3 seeds
    t2_wins = sum(1 for a, b in zip(no_spindle, with_spindle)
                  if b["harm_discrimination"] >= a["harm_discrimination"])
    t2 = t2_wins >= 2

    outcome = "PASS" if (t1 and t2) else "FAIL"
    direction = "supports" if outcome == "PASS" else "weakens"

    summary = {
        "t1_spindle_no_harm_regression": {
            "wins": t1_wins, "pass": t1,
            "desc": "WITH_SPINDLE harm_dang <= NO_SPINDLE*1.05 in >=2/3 seeds",
        },
        "t2_spindle_maintains_discrimination": {
            "wins": t2_wins, "pass": t2,
            "desc": "WITH_SPINDLE harm_discrim >= NO_SPINDLE in >=2/3 seeds",
        },
    }

    print(f"\nOutcome: {outcome}")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts_str,
        "outcome": outcome,
        "evidence_direction": direction,
        "pass_criteria_summary": summary,
        "per_seed_results": all_results,
        "config": {
            "conditions": CONDITIONS,
            "sleep_interval": SLEEP_INTERVAL,
            "sws_write_steps": SWS_WRITE_STEPS,
            "context_switch_every": CONTEXT_SWITCH_EVERY,
            "training_episodes": TRAINING_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "eval_episodes_each": EVAL_EPISODES_EACH,
            "seeds": SEEDS,
            "env_safe_num_hazards": 1,
            "env_dangerous_num_hazards": 8,
        },
    }

    out_file = out_dir / f"{run_id}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Output written to: {out_file}")


if __name__ == "__main__":
    main()
