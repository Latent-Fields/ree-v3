#!/opt/local/bin/python3
"""
V3-EXQ-098 -- MECH-099 Three-Stream vs Two-Stream Harm Channel Architecture

Claims: MECH-099

MECH-099 asserts that the encoder must implement THREE heads, not two:
  (1) Dorsal head  -> z_self:  spatial/egocentric/self-motion
  (2) Ventral head -> z_world: object identity, allocentric content
  (3) Lateral head -> z_harm:  dynamic motion, biological motion, agency/harm signal
      Terminates near TPJ (MECH-095). Direct hazard + contamination view routing.

Without the lateral head, harm detection must route through z_world, contaminating
the world-content channel with harm salience (violates MECH-069 incommensurability).

Prior conflict (EXQ-015 mixed, EXQ-017 weakens): those experiments ran BEFORE the
SD-008 fix (alpha_world=0.3). At alpha=0.3, z_world was a ~3-step EMA, suppressing
event responses to 30%. The lateral head received a diluted gradient and could not
learn a meaningful harm representation. Engineering confound, not conceptual refutation.
Current standard: alpha_world=0.9.

Discriminative pair:

  THREE_STREAM: harm_dim=16, lateral head enabled (MECH-099 ON).
    z_harm = lateral_head(world_obs[HAZARD_INDICES] + world_obs[175:200])
    harm_probe(z_harm) supervised with BCE on harm event labels.
    Gradient flows: harm_probe -> z_harm -> lateral_head.

  TWO_STREAM: harm_dim=0, lateral head disabled (ablation).
    No z_harm; harm must route through z_world.
    harm_probe(z_world) supervised with BCE on harm event labels.
    Gradient flows: harm_probe -> z_world -> world_encoder.

Both: alpha_world=0.9 (SD-008), use_harm_stream=False (SD-010 disabled so lateral
head output is NOT overridden), same env, same seeds.

C2 measurement: a SEPARATE world_harm_probe trained only on detached z_world (no
gradient to encoder) measures how much harm information remains in z_world.
THREE_STREAM should show lower z_world harm R2 (harm offloaded to lateral channel).

PASS criteria (ALL required):
  C1 (harm detection): THREE_STREAM harm_event_AUC >= TWO_STREAM harm_event_AUC + 0.05
     In THREE_STREAM: harm_probe(z_harm) vs harm events.
     In TWO_STREAM:   harm_probe(z_world) vs harm events.
  C2 (z_world purity): THREE_STREAM z_world_harm_R2 <= 0.8 * TWO_STREAM z_world_harm_R2
     Harm offloaded from z_world to lateral channel (tests MECH-069).
  C3 (behavioral): THREE_STREAM harm_avoidance_rate >= TWO_STREAM harm_avoidance_rate
     Flee policy uses harm_probe score; better harm detection -> fewer contacts.

PASS if C1 AND C2 AND C3.
Partial finding: C1 pass, C2 or C3 fail -> FAIL, note lateral helps detection but
MECH-069 incommensurability not confirmed.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_098_mech099_three_stream"
CLAIM_IDS = ["MECH-099"]

# Sigmoid threshold above which the agent "flees" (reverses last action)
FLEE_THRESHOLD = 0.4

# Reverse-action map for flee policy: up<->down, left<->right, stay=stay
# CausalGridWorldV2.ACTIONS: {0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1), 4:(0,0)}
REVERSE_ACTION = {0: 1, 1: 0, 2: 3, 3: 2, 4: 4}


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _compute_auc(scores: List[float], labels: List[float]) -> float:
    """Rank-based (Wilcoxon-Mann-Whitney) AUC.
    Returns 0.5 if no positives or no negatives.
    """
    if not scores:
        return 0.5
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    auc = 0.0
    running_neg = 0.0
    for _, label in pairs:
        if label == 0.0:
            running_neg += 1.0
        else:
            auc += running_neg
    return auc / (n_pos * n_neg)


def _compute_r2(pred_probs: List[float], labels: List[float]) -> float:
    """Coefficient of determination (R2) of sigmoid probe probabilities vs binary labels.
    Clipped to [0, 1] since negative R2 indicates probe is worse than mean predictor.
    """
    if len(pred_probs) < 2:
        return 0.0
    mean_y = sum(labels) / len(labels)
    ss_tot = sum((y - mean_y) ** 2 for y in labels)
    if ss_tot < 1e-9:
        return 0.0
    ss_res = sum((p - y) ** 2 for p, y in zip(pred_probs, labels))
    r2 = 1.0 - ss_res / ss_tot
    return max(0.0, min(1.0, r2))


def _run_single(
    seed: int,
    three_stream: bool,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    harm_dim: int,
    lr: float,
    alpha_world: float,
    alpha_self: float,
    harm_scale: float,
    proximity_harm_scale: float,
) -> Dict:
    """Run one (seed, condition) cell. Returns per-cell metrics dict."""
    torch.manual_seed(seed)
    random.seed(seed)

    cond = "THREE_STREAM" if three_stream else "TWO_STREAM"

    env = CausalGridWorldV2(
        seed=seed,
        size=12,
        num_hazards=4,
        num_resources=5,
        hazard_harm=harm_scale,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=proximity_harm_scale,
        proximity_benefit_scale=proximity_harm_scale * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )

    # Config: lateral head enabled (harm_dim=16) in THREE_STREAM, disabled in TWO_STREAM.
    # CRITICAL: use_harm_stream=False so SD-010 HarmEncoder does NOT override z_harm
    # from the lateral head (stack.py:926-932).
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        harm_dim=harm_dim if three_stream else 0,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
    )
    config.latent.unified_latent_mode = False
    config.latent.use_harm_stream = False  # SD-010 must be OFF

    agent = REEAgent(config)
    device = agent.device

    # Harm detection probe (C1):
    #   THREE_STREAM: probe_in_dim = harm_dim, takes z_harm from lateral head
    #   TWO_STREAM:   probe_in_dim = world_dim, takes z_world
    probe_in_dim = harm_dim if three_stream else world_dim
    harm_probe = nn.Linear(probe_in_dim, 1).to(device)

    # World harm probe (C2): trained ONLY on detached z_world (no gradient to encoder).
    # Measures residual harm info in z_world as a pure measurement probe.
    world_harm_probe = nn.Linear(world_dim, 1).to(device)

    # Combined optimizer: agent parameters + harm_probe jointly.
    # THREE_STREAM: harm_probe loss backprops through z_harm -> lateral_head.
    # TWO_STREAM:   harm_probe loss backprops through z_world -> world_encoder.
    all_train_params = list(agent.parameters()) + list(harm_probe.parameters())
    optimizer = optim.Adam(all_train_params, lr=lr)

    # C2 probe optimizer: separate, trains ONLY world_harm_probe on detached z_world.
    world_probe_opt = optim.Adam(world_harm_probe.parameters(), lr=lr)

    # Training counters
    train_counts: Dict[str, int] = {
        "hazard_approach": 0, "env_caused_hazard": 0,
        "agent_caused_hazard": 0, "none": 0, "total": 0, "harm_events": 0,
    }

    # --- TRAINING PHASE ---
    agent.train()
    harm_probe.train()
    world_harm_probe.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)

            # Do NOT pass obs_harm: lateral head z_harm must not be overridden by SD-010
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, device)
            agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)

            ttype = info.get("transition_type", "none")
            if ttype in train_counts:
                train_counts[ttype] += 1
            train_counts["total"] += 1

            is_harm_float = 1.0 if float(harm_signal) < 0 else 0.0
            if is_harm_float > 0.5:
                train_counts["harm_events"] += 1

            harm_label = torch.tensor([[is_harm_float]], device=device)

            # Standard E1 + E2 losses
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()

            # Harm probe loss (joint backward):
            # THREE_STREAM: trains lateral_head via z_harm gradient
            # TWO_STREAM:   trains world_encoder via z_world gradient
            if three_stream and latent.z_harm is not None:
                harm_pred = harm_probe(latent.z_harm)
            else:
                harm_pred = harm_probe(latent.z_world)
            harm_probe_loss = F.binary_cross_entropy_with_logits(harm_pred, harm_label)

            total_loss = e1_loss + e2_loss + harm_probe_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(all_train_params, 1.0)
                optimizer.step()

            # C2 probe: separate backward on detached z_world (measurement only)
            w_pred = world_harm_probe(latent.z_world.detach())
            w_loss = F.binary_cross_entropy_with_logits(w_pred, harm_label.detach())
            world_probe_opt.zero_grad()
            w_loss.backward()
            world_probe_opt.step()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
            print(
                f"  [train] seed={seed} cond={cond}"
                f" ep {ep + 1}/{warmup_episodes}"
                f" approach={train_counts['hazard_approach']}"
                f" harm_events={train_counts['harm_events']}"
                f" total={train_counts['total']}",
                flush=True,
            )

    # --- EVAL PHASE ---
    agent.eval()
    harm_probe.eval()
    world_harm_probe.eval()

    harm_scores: List[float] = []
    harm_labels_eval: List[float] = []
    world_harm_scores: List[float] = []
    harm_contacts = 0
    total_eval_steps = 0
    last_action_idx = None

    for _ in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        last_action_idx = None

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()

                if three_stream and latent.z_harm is not None:
                    raw_score = harm_probe(latent.z_harm).item()
                else:
                    raw_score = harm_probe(latent.z_world).item()
                world_raw = world_harm_probe(latent.z_world).item()

            # Harm-guided flee policy (C3):
            # If harm_probe sigmoid > FLEE_THRESHOLD, reverse last action.
            harm_sigmoid = 1.0 / (1.0 + math.exp(-raw_score))
            if harm_sigmoid > FLEE_THRESHOLD and last_action_idx is not None:
                action_idx = REVERSE_ACTION.get(last_action_idx, random.randint(0, env.action_dim - 1))
            else:
                action_idx = random.randint(0, env.action_dim - 1)

            action = _action_to_onehot(action_idx, env.action_dim, device)
            agent._last_action = action
            _, harm_signal, done, info, obs_dict = env.step(action)
            last_action_idx = action_idx

            is_harm_float = 1.0 if float(harm_signal) < 0 else 0.0
            harm_scores.append(raw_score)
            harm_labels_eval.append(is_harm_float)
            world_harm_scores.append(world_raw)

            if is_harm_float > 0.5:
                harm_contacts += 1
            total_eval_steps += 1

            if done:
                break

    harm_event_auc = _compute_auc(harm_scores, harm_labels_eval)
    world_harm_r2 = _compute_r2(
        [1.0 / (1.0 + math.exp(-s)) for s in world_harm_scores],
        harm_labels_eval,
    )
    harm_avoidance_rate = 1.0 - harm_contacts / max(1, total_eval_steps)

    n_harm_eval = sum(1 for x in harm_labels_eval if x > 0.5)

    print(
        f"  [eval] seed={seed} cond={cond}"
        f" auc={harm_event_auc:.4f}"
        f" world_r2={world_harm_r2:.4f}"
        f" avoidance={harm_avoidance_rate:.4f}"
        f" harm_contacts={harm_contacts}/{total_eval_steps}"
        f" n_harm_events={n_harm_eval}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond,
        "three_stream": three_stream,
        "harm_event_auc": float(harm_event_auc),
        "z_world_harm_r2": float(world_harm_r2),
        "harm_avoidance_rate": float(harm_avoidance_rate),
        "harm_contacts": int(harm_contacts),
        "total_eval_steps": int(total_eval_steps),
        "n_harm_events_eval": int(n_harm_eval),
        "train_harm_events": int(train_counts["harm_events"]),
        "train_approach_events": int(train_counts["hazard_approach"]),
        "train_total_steps": int(train_counts["total"]),
    }


def run(
    seeds: Tuple = (42, 7),
    warmup_episodes: int = 400,
    eval_episodes: int = 100,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    harm_dim: int = 16,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_harm_scale: float = 0.05,
    **kwargs,
) -> dict:
    """Discriminative pair: THREE_STREAM (harm_dim=16) vs TWO_STREAM (harm_dim=0)."""
    results_three: List[Dict] = []
    results_two: List[Dict] = []

    for seed in seeds:
        for three_stream in [True, False]:
            cond = "THREE_STREAM" if three_stream else "TWO_STREAM"
            print(
                f"\n[V3-EXQ-098] {cond} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" alpha_world={alpha_world} harm_dim={harm_dim if three_stream else 0}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                three_stream=three_stream,
                warmup_episodes=warmup_episodes,
                eval_episodes=eval_episodes,
                steps_per_episode=steps_per_episode,
                self_dim=self_dim,
                world_dim=world_dim,
                harm_dim=harm_dim,
                lr=lr,
                alpha_world=alpha_world,
                alpha_self=alpha_self,
                harm_scale=harm_scale,
                proximity_harm_scale=proximity_harm_scale,
            )
            if three_stream:
                results_three.append(r)
            else:
                results_two.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    auc_three = _avg(results_three, "harm_event_auc")
    auc_two   = _avg(results_two,   "harm_event_auc")
    r2_three  = _avg(results_three, "z_world_harm_r2")
    r2_two    = _avg(results_two,   "z_world_harm_r2")
    av_three  = _avg(results_three, "harm_avoidance_rate")
    av_two    = _avg(results_two,   "harm_avoidance_rate")

    auc_delta   = auc_three - auc_two
    r2_ratio    = r2_three / max(r2_two, 1e-6)
    av_delta    = av_three - av_two

    # Pre-registered PASS criteria
    c1_pass = auc_delta    >= 0.05
    c2_pass = r2_ratio     <= 0.80
    c3_pass = av_delta     >= 0.0

    all_pass     = c1_pass and c2_pass and c3_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass])
    status       = "PASS" if all_pass else "FAIL"

    print(f"\n[V3-EXQ-098] Final results:", flush=True)
    print(
        f"  auc_THREE={auc_three:.4f}  auc_TWO={auc_two:.4f}"
        f"  delta={auc_delta:+.4f}  (C1 thresh >=0.05)",
        flush=True,
    )
    print(
        f"  r2_world_THREE={r2_three:.4f}  r2_world_TWO={r2_two:.4f}"
        f"  ratio={r2_ratio:.4f}  (C2 thresh <=0.80)",
        flush=True,
    )
    print(
        f"  avoidance_THREE={av_three:.4f}  avoidance_TWO={av_two:.4f}"
        f"  delta={av_delta:+.4f}  (C3 thresh >=0)",
        flush=True,
    )
    print(f"  status={status} ({criteria_met}/3)", flush=True)

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: auc_THREE={auc_three:.4f} vs auc_TWO={auc_two:.4f}"
            f" (delta={auc_delta:+.4f}, needs >=0.05)."
            " Lateral head did not improve harm event AUC by 5pp."
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: r2_world_THREE={r2_three:.4f} vs r2_world_TWO={r2_two:.4f}"
            f" (ratio={r2_ratio:.4f}, needs <=0.80)."
            " z_world harm contamination not reduced by 20%% in THREE_STREAM."
            " MECH-069 incommensurability not confirmed via offloading."
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: avoidance_THREE={av_three:.4f} vs avoidance_TWO={av_two:.4f}"
            f" (delta={av_delta:+.4f}, needs >=0)."
            " Flee policy with lateral z_harm did not reduce harm contacts."
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    if all_pass:
        interpretation = (
            "MECH-099 SUPPORTED: dedicated lateral harm channel produces better harm"
            f" detection (AUC delta={auc_delta:+.4f}), cleaner z_world"
            f" (r2 ratio={r2_ratio:.4f}), and better avoidance"
            f" (av delta={av_delta:+.4f})."
            " Three-stream architecture is superior to two-stream for harm routing."
            " Supports MECH-069: harm offloaded from z_world to lateral z_harm."
            " Resolves EXQ-015/017 conflict (engineering confound, SD-008 alpha_world=0.3)."
        )
    elif c1_pass and (not c2_pass or not c3_pass):
        interpretation = (
            "PARTIAL: C1 passes (lateral channel improves harm AUC"
            f" delta={auc_delta:+.4f}) but MECH-069 incommensurability not"
            " confirmed (C2 fails). Lateral head helps harm detection but z_world"
            " remains contaminated with harm signal. Architecture may need explicit"
            " adversarial z_world purity loss (cf. AdversarialSplitHead)."
        )
    else:
        interpretation = (
            "MECH-099 NOT SUPPORTED at this training budget/scale."
            f" AUC delta={auc_delta:+.4f} (C1 needs 0.05)."
            " Lateral head did not produce a cleaner harm signal than z_world."
            " Possible causes: lateral_head input (HAZARD_INDICES + contamination_view)"
            " insufficient for 400-episode training; world_obs_encoder shared weights"
            " create interference; or both conditions reach similar representation quality."
        )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    per_three_rows = "\n".join(
        f"  seed={r['seed']}: auc={r['harm_event_auc']:.4f}"
        f" r2_world={r['z_world_harm_r2']:.4f}"
        f" avoidance={r['harm_avoidance_rate']:.4f}"
        f" harm_events_train={r['train_harm_events']}"
        for r in results_three
    )
    per_two_rows = "\n".join(
        f"  seed={r['seed']}: auc={r['harm_event_auc']:.4f}"
        f" r2_world={r['z_world_harm_r2']:.4f}"
        f" avoidance={r['harm_avoidance_rate']:.4f}"
        f" harm_events_train={r['train_harm_events']}"
        for r in results_two
    )

    summary_markdown = (
        f"# V3-EXQ-098 -- MECH-099 Three-Stream vs Two-Stream Harm Channel\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-099\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: auc_THREE - auc_TWO >= 0.05  (harm_event_AUC improvement)\n"
        f"C2: r2_world_THREE / r2_world_TWO <= 0.80  (z_world purity, MECH-069)\n"
        f"C3: avoidance_THREE >= avoidance_TWO  (behavioral, flee policy)\n\n"
        f"## Aggregate Results\n\n"
        f"| Metric | THREE_STREAM | TWO_STREAM | Delta/Ratio | Pass |\n"
        f"|--------|-------------|-----------|-------------|------|\n"
        f"| harm_event_AUC | {auc_three:.4f} | {auc_two:.4f} | {auc_delta:+.4f} | {'YES' if c1_pass else 'NO'} |\n"
        f"| z_world_harm_R2 | {r2_three:.4f} | {r2_two:.4f} | {r2_ratio:.4f} | {'YES' if c2_pass else 'NO'} |\n"
        f"| harm_avoidance | {av_three:.4f} | {av_two:.4f} | {av_delta:+.4f} | {'YES' if c3_pass else 'NO'} |\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n\n"
        f"## Per-Seed (THREE_STREAM)\n\n"
        f"{per_three_rows}\n\n"
        f"## Per-Seed (TWO_STREAM)\n\n"
        f"{per_two_rows}\n"
        f"{failure_section}\n"
    )

    return {
        "status": status,
        "metrics": {
            "harm_event_auc_three":    float(auc_three),
            "harm_event_auc_two":      float(auc_two),
            "auc_delta":               float(auc_delta),
            "z_world_harm_r2_three":   float(r2_three),
            "z_world_harm_r2_two":     float(r2_two),
            "r2_ratio":                float(r2_ratio),
            "harm_avoidance_three":    float(av_three),
            "harm_avoidance_two":      float(av_two),
            "avoidance_delta":         float(av_delta),
            "crit1_pass":              1.0 if c1_pass else 0.0,
            "crit2_pass":              1.0 if c2_pass else 0.0,
            "crit3_pass":              1.0 if c3_pass else 0.0,
            "criteria_met":            float(criteria_met),
            "n_seeds":                 float(len(seeds)),
            "alpha_world":             float(alpha_world),
            "harm_dim":                float(harm_dim),
            "proximity_harm_scale":    float(proximity_harm_scale),
        },
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if c1_pass else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "per_seed_three": results_three,
        "per_seed_two": results_two,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",           type=int, nargs="+", default=[42, 7])
    parser.add_argument("--warmup",          type=int,   default=400)
    parser.add_argument("--eval-eps",        type=int,   default=100)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--alpha-self",      type=float, default=0.3)
    parser.add_argument("--harm-dim",        type=int,   default=16)
    parser.add_argument("--harm-scale",      type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Quick smoke test: 1 seed, 5 warmup, 3 eval, 50 steps. Writes JSON.",
    )
    args = parser.parse_args()

    if args.smoke_test:
        seeds    = (42,)
        warmup   = 5
        eval_eps = 3
        steps    = 50
        print("[V3-EXQ-098] SMOKE TEST MODE", flush=True)
    else:
        seeds    = tuple(args.seeds)
        warmup   = args.warmup
        eval_eps = args.eval_eps
        steps    = args.steps

    result = run(
        seeds=seeds,
        warmup_episodes=warmup,
        eval_episodes=eval_eps,
        steps_per_episode=steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        harm_dim=args.harm_dim,
        harm_scale=args.harm_scale,
        proximity_harm_scale=args.proximity_scale,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    if args.smoke_test:
        print("[SMOKE TEST] Key metrics:", flush=True)
        for k in [
            "harm_event_auc_three", "harm_event_auc_two", "auc_delta",
            "z_world_harm_r2_three", "z_world_harm_r2_two",
            "harm_avoidance_three", "harm_avoidance_two",
            "crit1_pass", "crit2_pass", "crit3_pass",
        ]:
            print(f"  {k}: {result['metrics'].get(k, 'N/A')}", flush=True)
    else:
        for k, v in result["metrics"].items():
            print(f"  {k}: {v}", flush=True)
