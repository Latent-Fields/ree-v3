#!/opt/local/bin/python3
"""
V3-EXQ-244 -- MECH-165 Replay Diversity: Forward + Reverse Balanced Replay

Claims: MECH-165

Scientific question:
  Does balanced forward+reverse replay (MECH-165) produce broader behavioral
  strategy diversity than forward-only or waking-only replay? Does reverse
  replay specifically prevent the monostrategy failure mode (convergence to
  a single behavioral pattern)?

MECH-165 asserts: Forward-only replay causes the world model to represent
only what the agent actually did, never what was possible. This induces
monostrategy convergence: the agent repeatedly chooses the same action
sequence that worked before, even when alternative routes are available.
Reverse replay traces backward from harm events to give causal antecedents --
enabling the agent to learn "had I taken a different path before arriving
at the harm, I would not have encountered it." This diversity is a
prerequisite for behavioral strategy breadth (Shin 2019).

Monostrategy is measured as action distribution entropy: low entropy = agent
always does the same thing; high entropy = explores multiple behavioral strategies.

Design:
  Three conditions, each run with 5 seeds on TWO-CONTEXT task.

  WAKING_ONLY:
    Standard online training. No offline replay phases.
    Baseline behavioral entropy.

  FORWARD_ONLY_REPLAY:
    After every SLEEP_INTERVAL=10 episodes:
      Forward replay: replay z_world states in the ORDER they occurred
      (oldest first). Train harm_eval_head on the forward trajectory.
      This represents what the agent experienced, not what was possible.
      Prediction: monostrategy convergence -- action entropy decreases.

  BALANCED_REPLAY:
    After every SLEEP_INTERVAL=10 episodes:
      Forward replay: same as FORWARD_ONLY first half.
      Reverse replay: then replay z_world states in REVERSE order,
      starting from the most recent harm event and tracing backward.
      This gives causal antecedents: agent learns which earlier states
      led to harm, enabling alternative route evaluation.
      Prediction: higher action entropy, fewer monostrategy episodes.

Metrics:
  - action_entropy: Shannon entropy of action distribution during eval
    (mean over eval episodes, higher = more diverse strategies)
  - monostrategy_rate: fraction of eval episodes where a single action
    comprises >= 60% of all steps (lower = better diversity)
  - harm_rate_dangerous: harm rate in DANGEROUS context
  - harm_rate_safe: harm rate in SAFE context

PASS criteria:
  M1: action_entropy(BALANCED_REPLAY) > action_entropy(FORWARD_ONLY_REPLAY)
      in >=3/5 seeds (balanced replay produces more diverse strategies)
  M2: monostrategy_rate(BALANCED_REPLAY) < monostrategy_rate(FORWARD_ONLY_REPLAY)
      in >=3/5 seeds (balanced replay reduces monostrategy convergence)
  M3: action_entropy(BALANCED_REPLAY) > action_entropy(WAKING_ONLY)
      in >=3/5 seeds (replay diversity exceeds no-replay baseline)

PASS: M1 AND (M2 OR M3)

experiment_purpose: "evidence"
  MECH-165 is a behavioral diversity claim directly tested by entropy metrics.

claim_ids: ["MECH-165"]
experiment_purpose: "evidence"
"""

import sys
import random
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_244_mech165_reverse_replay_diversity"
CLAIM_IDS = ["MECH-165"]
EXPERIMENT_PURPOSE = "evidence"

SLEEP_INTERVAL = 10        # episodes between replay phases
CONTEXT_SWITCH_EVERY = 5
REPLAY_STEPS = 64          # states replayed per pass (forward or reverse)

TRAINING_EPISODES = 200
STEPS_PER_EPISODE = 150
EVAL_EPISODES_EACH = 30
MAX_TRAJ_BUF = 5000
SEEDS = [42, 7, 13, 100, 200]

CONDITIONS = ["WAKING_ONLY", "FORWARD_ONLY_REPLAY", "BALANCED_REPLAY"]

# Monostrategy threshold: episode where single action >= MONO_THRESHOLD of steps
MONO_THRESHOLD = 0.60


# ------------------------------------------------------------------ #
# Environment / agent helpers                                          #
# ------------------------------------------------------------------ #

def _make_env_safe(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=1,
        num_resources=4,
        hazard_harm=0.02,
        env_drift_interval=50,
        env_drift_prob=0.05,
        proximity_harm_scale=0.10,
        proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_env_dangerous(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed + 1000,
        size=10,
        num_hazards=8,
        num_resources=4,
        hazard_harm=0.05,
        env_drift_interval=50,
        env_drift_prob=0.05,
        proximity_harm_scale=0.15,
        proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=0,
        novelty_bonus_weight=0.0,
    )
    return REEAgent(config)


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _compute_action_entropy(action_counts: Dict[int, int]) -> float:
    """Shannon entropy of action distribution."""
    total = sum(action_counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in action_counts.values()]
    return -sum(p * math.log(p + 1e-9) for p in probs if p > 0)


def _is_monostrategy_episode(action_counts: Dict[int, int]) -> bool:
    """True if a single action comprises >= MONO_THRESHOLD of all steps."""
    total = sum(action_counts.values())
    if total == 0:
        return False
    max_count = max(action_counts.values())
    return (max_count / total) >= MONO_THRESHOLD


# ------------------------------------------------------------------ #
# Replay implementations                                               #
# ------------------------------------------------------------------ #

def _forward_replay_pass(
    agent: REEAgent,
    traj_buf: List[Tuple[torch.Tensor, float]],
    harm_eval_opt,
    n: int,
) -> Dict:
    """
    Forward replay: replays z_world states in temporal order (oldest first).
    Trains harm_eval on the forward sequence.
    Represents what the agent experienced (history).
    """
    if len(traj_buf) < 4:
        return {"forward_replay_steps": 0}

    k = min(len(traj_buf), n)
    # Temporal order: start from (len - k) to end
    replay_slice = traj_buf[-k:]   # oldest to newest (temporal order)

    zw_list = [zw for (zw, _) in replay_slice]
    lbl_list = [float(lbl) for (_, lbl) in replay_slice]
    zw_b = torch.cat(zw_list, dim=0)
    tgt_b = torch.tensor([[l] for l in lbl_list], dtype=torch.float,
                          device=agent.device)
    pred_b = agent.e3.harm_eval_head(zw_b)
    loss_b = F.binary_cross_entropy_with_logits(pred_b, tgt_b)
    harm_eval_opt.zero_grad()
    loss_b.backward()
    harm_eval_opt.step()

    return {"forward_replay_steps": k, "forward_loss": loss_b.item()}


def _reverse_replay_pass(
    agent: REEAgent,
    dang_traj_buf: List[Tuple[torch.Tensor, float]],
    harm_eval_opt,
    n: int,
) -> Dict:
    """
    Reverse replay: replays DANGEROUS z_world states in REVERSE temporal order.
    Starts from the most recent harm event and traces backward.
    This reveals causal antecedents: which states preceded harm.

    MECH-165: reverse replay specifically enables learning of what paths
    led to harm, not just what harm felt like. The agent can then identify
    the choice-points where alternative routes were available.
    """
    if len(dang_traj_buf) < 4:
        return {"reverse_replay_steps": 0}

    k = min(len(dang_traj_buf), n)
    # Find last harm event in buffer
    last_harm_idx = -1
    for i in range(len(dang_traj_buf) - 1, -1, -1):
        if dang_traj_buf[i][1] > 0.5:  # harm event
            last_harm_idx = i
            break

    if last_harm_idx < 0:
        # No harm events: replay in reverse order from end of buffer
        replay_slice = list(reversed(dang_traj_buf[-k:]))
    else:
        # Trace backward from last harm event
        start = max(0, last_harm_idx - k + 1)
        replay_slice = list(reversed(dang_traj_buf[start:last_harm_idx + 1]))

    if len(replay_slice) < 2:
        return {"reverse_replay_steps": 0}

    zw_list = [zw for (zw, _) in replay_slice]
    lbl_list = [float(lbl) for (_, lbl) in replay_slice]
    zw_b = torch.cat(zw_list, dim=0)
    tgt_b = torch.tensor([[l] for l in lbl_list], dtype=torch.float,
                          device=agent.device)
    pred_b = agent.e3.harm_eval_head(zw_b)
    loss_b = F.binary_cross_entropy_with_logits(pred_b, tgt_b)
    harm_eval_opt.zero_grad()
    loss_b.backward()
    harm_eval_opt.step()

    return {"reverse_replay_steps": len(replay_slice), "reverse_loss": loss_b.item()}


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
    record_actions: bool = False,
) -> Tuple[float, List[Tuple[torch.Tensor, float]], Dict[int, int]]:
    """
    Run single episode.
    Returns (harm_sum, traj_events [(z_world, harm_label)], action_counts).
    """
    _, obs_dict = env.reset()
    agent.reset()
    ep_harm = 0.0
    traj_events: List[Tuple[torch.Tensor, float]] = []
    action_counts: Dict[int, int] = {}

    for _step in range(steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        obs_harm = obs_dict.get("harm_obs", None)

        latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
        agent.clock.advance()
        z_world = latent.z_world.detach().clone()

        # Action selection: harm-avoidance
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

        if record_actions:
            action_counts[best_action] = action_counts.get(best_action, 0) + 1

        _, harm_signal, done, _info, obs_dict = env.step(action_oh)
        is_harm = float(harm_signal) < 0

        traj_events.append((z_world, 1.0 if is_harm else 0.0))

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

    return ep_harm, traj_events, action_counts


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
    """Run one condition."""
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

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    dang_traj_buf: List[Tuple[torch.Tensor, float]] = []  # DANGEROUS context events
    all_traj_buf: List[Tuple[torch.Tensor, float]] = []   # all events (for forward)
    sleep_log: List[Dict] = []

    agent.train()
    per_ep_harm_safe: List[float] = []
    per_ep_harm_dang: List[float] = []

    for ep in range(training_episodes):
        block = ep // CONTEXT_SWITCH_EVERY
        is_safe_ep = (block % 2 == 0)
        env = env_safe if is_safe_ep else env_dang

        ep_harm, traj_events, _ = _run_episode(
            agent, env, steps_per_episode,
            train=True,
            optimizer=optimizer,
            harm_eval_opt=harm_eval_opt,
            harm_buf_pos=harm_buf_pos,
            harm_buf_neg=harm_buf_neg,
            record_actions=False,
        )
        harm_rate = ep_harm / steps_per_episode

        all_traj_buf.extend(traj_events)
        if len(all_traj_buf) > MAX_TRAJ_BUF:
            all_traj_buf = all_traj_buf[-MAX_TRAJ_BUF:]

        if is_safe_ep:
            per_ep_harm_safe.append(harm_rate)
        else:
            per_ep_harm_dang.append(harm_rate)
            dang_traj_buf.extend(traj_events)
            if len(dang_traj_buf) > MAX_TRAJ_BUF:
                dang_traj_buf = dang_traj_buf[-MAX_TRAJ_BUF:]

        if len(harm_buf_pos) > 4000:
            harm_buf_pos = harm_buf_pos[-4000:]
        if len(harm_buf_neg) > 4000:
            harm_buf_neg = harm_buf_neg[-4000:]

        # Offline replay at interval boundary
        if condition != "WAKING_ONLY" and (ep + 1) % SLEEP_INTERVAL == 0:
            fwd_m = {}
            rev_m = {}

            if condition == "FORWARD_ONLY_REPLAY":
                fwd_m = _forward_replay_pass(
                    agent, all_traj_buf, harm_eval_opt, REPLAY_STEPS
                )

            elif condition == "BALANCED_REPLAY":
                # Forward replay on all context history
                fwd_m = _forward_replay_pass(
                    agent, all_traj_buf, harm_eval_opt, REPLAY_STEPS // 2
                )
                # Reverse replay: trace back from DANGEROUS harm events (MECH-165)
                rev_m = _reverse_replay_pass(
                    agent, dang_traj_buf, harm_eval_opt, REPLAY_STEPS // 2
                )

            sleep_log.append({"ep": ep, **fwd_m, **rev_m})

    # Evaluation -- record action distributions
    agent.eval()
    eval_harm_safe: List[float] = []
    eval_harm_dang: List[float] = []
    eval_entropy_safe: List[float] = []
    eval_entropy_dang: List[float] = []
    eval_mono_safe: List[bool] = []
    eval_mono_dang: List[bool] = []

    for _ in range(eval_episodes_each):
        harm_s, _, action_counts = _run_episode(
            agent, env_safe, steps_per_episode,
            train=False, optimizer=optimizer,
            harm_eval_opt=harm_eval_opt,
            harm_buf_pos=harm_buf_pos,
            harm_buf_neg=harm_buf_neg,
            record_actions=True,
        )
        eval_harm_safe.append(harm_s / steps_per_episode)
        eval_entropy_safe.append(_compute_action_entropy(action_counts))
        eval_mono_safe.append(_is_monostrategy_episode(action_counts))

    for _ in range(eval_episodes_each):
        harm_d, _, action_counts = _run_episode(
            agent, env_dang, steps_per_episode,
            train=False, optimizer=optimizer,
            harm_eval_opt=harm_eval_opt,
            harm_buf_pos=harm_buf_pos,
            harm_buf_neg=harm_buf_neg,
            record_actions=True,
        )
        eval_harm_dang.append(harm_d / steps_per_episode)
        eval_entropy_dang.append(_compute_action_entropy(action_counts))
        eval_mono_dang.append(_is_monostrategy_episode(action_counts))

    harm_rate_safe = sum(eval_harm_safe) / len(eval_harm_safe)
    harm_rate_dang = sum(eval_harm_dang) / len(eval_harm_dang)
    action_entropy_safe = sum(eval_entropy_safe) / len(eval_entropy_safe)
    action_entropy_dang = sum(eval_entropy_dang) / len(eval_entropy_dang)
    action_entropy_mean = (action_entropy_safe + action_entropy_dang) / 2
    monostrategy_rate = sum(eval_mono_safe + eval_mono_dang) / (
        len(eval_mono_safe) + len(eval_mono_dang)
    )

    if verbose:
        print(f"  [seed={seed} {condition}] "
              f"entropy_safe={action_entropy_safe:.4f} "
              f"entropy_dang={action_entropy_dang:.4f} "
              f"mono_rate={monostrategy_rate:.4f} "
              f"harm_safe={harm_rate_safe:.4f} "
              f"harm_dang={harm_rate_dang:.4f} "
              f"sleep_passes={len(sleep_log)}")

    return {
        "seed": seed,
        "condition": condition,
        "action_entropy_safe": float(action_entropy_safe),
        "action_entropy_dangerous": float(action_entropy_dang),
        "action_entropy_mean": float(action_entropy_mean),
        "monostrategy_rate": float(monostrategy_rate),
        "harm_rate_safe": float(harm_rate_safe),
        "harm_rate_dangerous": float(harm_rate_dang),
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
            print(f"    entropy_mean={result['action_entropy_mean']:.4f} "
                  f"mono_rate={result['monostrategy_rate']:.4f} "
                  f"harm_dang={result['harm_rate_dangerous']:.4f} "
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

    waking  = by_cond("WAKING_ONLY")
    fwd     = by_cond("FORWARD_ONLY_REPLAY")
    bal     = by_cond("BALANCED_REPLAY")

    # M1: BALANCED entropy > FORWARD_ONLY entropy in >=3/5 seeds
    m1_wins = sum(1 for f, b in zip(fwd, bal)
                  if b["action_entropy_mean"] > f["action_entropy_mean"])
    m1 = m1_wins >= 3

    # M2: BALANCED monostrategy_rate < FORWARD_ONLY in >=3/5 seeds
    m2_wins = sum(1 for f, b in zip(fwd, bal)
                  if b["monostrategy_rate"] < f["monostrategy_rate"])
    m2 = m2_wins >= 3

    # M3: BALANCED entropy > WAKING_ONLY entropy in >=3/5 seeds
    m3_wins = sum(1 for w, b in zip(waking, bal)
                  if b["action_entropy_mean"] > w["action_entropy_mean"])
    m3 = m3_wins >= 3

    outcome = "PASS" if (m1 and (m2 or m3)) else "FAIL"

    summary = {
        "m1_entropy_balanced_gt_forward": {
            "wins": m1_wins, "pass": m1,
            "desc": "BALANCED entropy_mean > FORWARD_ONLY in >=3/5 seeds",
        },
        "m2_monostrategy_balanced_lt_forward": {
            "wins": m2_wins, "pass": m2,
            "desc": "BALANCED monostrategy_rate < FORWARD_ONLY in >=3/5 seeds",
        },
        "m3_entropy_balanced_gt_waking": {
            "wins": m3_wins, "pass": m3,
            "desc": "BALANCED entropy_mean > WAKING_ONLY in >=3/5 seeds",
        },
    }

    direction = "supports" if outcome == "PASS" else "weakens"

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
            "context_switch_every": CONTEXT_SWITCH_EVERY,
            "replay_steps": REPLAY_STEPS,
            "mono_threshold": MONO_THRESHOLD,
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
