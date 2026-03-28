#!/opt/local/bin/python3
"""
V3-EXQ-123 -- MECH-102: Harm Escalation Gradient Discriminative Pair (EXP-0021 / EVB-0017)

Claim: MECH-102
Proposal: EXP-0021 / EVB-0017
Dispatch mode: discriminative_pair
Min shared seeds: 2

MECH-102 asserts: "Violence is a terminal error-correction mechanism triggered only
when all other channels fail." More precisely, the claim predicts that E3's harm
evaluation produces a GRADED ethical valuation signal that ESCALATES with proximity
to harm -- approach -> contact -> post-contact -- rather than a binary flag.

The gradient ordering prediction:
  harm_score(approach) < harm_score(contact) < harm_score(post_contact)

This is a specific, testable prediction: the harm_eval signal must be monotonically
increasing across the three proximity phases. A binary flag (0/1) would show approach
and pre-contact states at the same (low) value with a step only at contact.

This experiment implements a clean discriminative pair:

  HARM_EVAL_ON:
    E3 harm_eval_head trained on harm_signal labels from CausalGridWorldV2.
    Proximity phases are: approach (proximity_signal > APPROACH_THRESH but no contact),
    contact (harm_signal < 0, i.e. hazard contact), post_contact (first N steps after
    leaving contact zone).
    The harm_eval output is recorded at each phase transition.

  HARM_EVAL_ABLATED:
    E3 harm_eval_head is NOT trained -- random init held frozen.
    All other components identical (same env, same random policy).
    The harm_eval output at each phase should be noise (no gradient).

PRIMARY METRIC:
  gradient_score_ON = fraction of seed/episode samples where
    harm_score_approach < harm_score_contact (ordering holds).
  A gradient score of 1.0 = perfect ordering; 0.5 = random (chance).
  The ablated condition should produce gradient_score ~0.5.

SECONDARY METRIC:
  escalation_gap = mean(harm_score_contact) - mean(harm_score_approach)
  Must be positive and meaningfully large in ON condition.

BOTH are required for PASS: the gradient must be ordered AND the gap must be real.

Conditions and seeds:
  - HARM_EVAL_ON vs HARM_EVAL_ABLATED
  - 2 matched seeds: [42, 123]
  - 4 cells total (2 seeds x 2 conditions)
  - 400 warmup + 50 eval episodes x 200 steps
  - CausalGridWorldV2: size=6, 4 hazards, 3 resources

PRE-REGISTERED THRESHOLDS (hardcoded, not inferred post-hoc):

  C1 (gradient ordering delta):
    gradient_score_ON - gradient_score_ABLATED >= THRESH_C1_GRAD_DELTA
    ON condition must show substantially more ordered harm gradient than ablated.
    Threshold: THRESH_C1_GRAD_DELTA = 0.15  (15pp lift over ablated baseline)

  C2 (absolute gradient ordering):
    gradient_score_ON >= THRESH_C2_GRAD_ABS
    ON condition must achieve reliable ordering (approach < contact) above chance.
    Threshold: THRESH_C2_GRAD_ABS = 0.65

  C3 (consistency across seeds):
    gradient_score_ON > gradient_score_ABLATED for BOTH seeds independently.
    Direction must replicate.

  C4 (data quality -- sufficient approach-contact pairs):
    n_approach_contact_pairs >= THRESH_C4_MIN_PAIRS per seed.
    Without sufficient transition events the gradient cannot be measured.
    Threshold: THRESH_C4_MIN_PAIRS = 20

  C5 (escalation gap diagnostic):
    escalation_gap_ON >= THRESH_C5_GAP  (diagnostic only -- not required for PASS)
    The mean contact harm score must exceed mean approach harm score.
    Threshold: THRESH_C5_GAP = 0.05

PASS criteria:
  C1 AND C2 AND C3 AND C4 -> PASS -> supports MECH-102
  C1 AND C3 AND C4, NOT C2 -> mixed (relative advantage, absolute threshold not met)
  NOT C1 AND C4            -> FAIL -> gradient escalation hypothesis refuted
  NOT C4                   -> inconclusive (data quality failure)

Decision mapping:
  PASS            -> retain_ree (graded ethical valuation supported)
  C1+C3+C4        -> hybridize (relative advantage replicated, absolute weak)
  NOT C1 AND C4   -> retire_ree_claim (binary flag, no gradient)
  NOT C4          -> inconclusive

CLAIM_IDS RATIONALE:
  The only claim directly tested is MECH-102: does E3 harm_eval produce a graded
  signal that escalates approach -> contact?
  MECH-102 directly predicts YES (harm eval escalates monotonically with proximity,
  not a binary flag). ARC-024 (opposing benefit gradient) is not tagged -- that
  tests the harm-benefit opposition, not escalation ordering per se.
"""

import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_123_mech102_harm_gradient_pair"
CLAIM_IDS = ["MECH-102"]

# Pre-registered thresholds
THRESH_C1_GRAD_DELTA  = 0.15   # gradient_score_ON - gradient_score_ABLATED >= 0.15
THRESH_C2_GRAD_ABS    = 0.65   # gradient_score_ON >= 0.65
THRESH_C4_MIN_PAIRS   = 20     # n_approach_contact_pairs >= 20 per seed
THRESH_C5_GAP         = 0.05   # escalation_gap_ON >= 0.05 (diagnostic only)

# Phase detection thresholds
# CausalGridWorldV2 proximity_harm_scale=0.01 -> approach signals ~[-0.012, -0.001]
# CausalGridWorldV2 hazard_harm=0.02          -> contact signals  ~[-0.025, -0.015]
# Threshold set midpoint: approach = harm_signal in (-0.015, 0); contact = harm_signal <= -0.015
APPROACH_HARM_THRESH  = -0.015  # harm_signal < 0 and > this = proximity (approach)
CONTACT_HARM_THRESH   = -0.015  # harm_signal <= this = contact


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_to_onehot(action_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, action_idx] = 1.0
    return v


def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
    return t.unsqueeze(0) if t.dim() == 1 else t


def _mean_safe(lst: List[float]) -> float:
    return float(sum(lst) / len(lst)) if lst else 0.0


def _build_config(
    env: CausalGridWorldV2,
    self_dim: int,
    world_dim: int,
    alpha_world: float,
    alpha_self: float,
) -> REEConfig:
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=0,  # SD-007 disabled -- isolate MECH-102 mechanism
    )
    # SD-005: split latent mode (z_self != z_world)
    config.latent.unified_latent_mode = False
    # Standard single-rate: all modules tick every step
    config.heartbeat.e1_steps_per_tick = 1
    config.heartbeat.e2_steps_per_tick = 1
    config.heartbeat.e3_steps_per_tick = 1
    return config


# ---------------------------------------------------------------------------
# Single cell runner
# ---------------------------------------------------------------------------

def _run_single(
    seed: int,
    harm_eval_on: bool,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    lr: float,
    alpha_world: float,
    alpha_self: float,
    harm_scale: float,
    proximity_harm_scale: float,
    env_drift_prob: float,
    env_drift_interval: int,
    dry_run: bool = False,
) -> Dict:
    """Run one (seed, condition) cell. Returns per-cell metrics."""
    torch.manual_seed(seed)
    random.seed(seed)

    cond = "HARM_EVAL_ON" if harm_eval_on else "HARM_EVAL_ABLATED"

    env = CausalGridWorldV2(
        seed=seed,
        size=6,
        num_hazards=4,
        num_resources=3,
        hazard_harm=harm_scale,
        env_drift_interval=env_drift_interval,
        env_drift_prob=env_drift_prob,
        proximity_harm_scale=proximity_harm_scale,
        proximity_benefit_scale=proximity_harm_scale * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        use_proxy_fields=True,
    )

    config = _build_config(env, self_dim, world_dim, alpha_world, alpha_self)
    agent = REEAgent(config)
    device = agent.device

    # Harm eval optimizer -- only used in ON condition
    e3_harm_opt = optim.Adam(agent.e3.harm_eval_head.parameters(), lr=lr)
    e1_opt = optim.Adam(agent.e1.parameters(), lr=lr)

    # In ABLATED condition: freeze harm_eval_head (random init, no training)
    if not harm_eval_on:
        for param in agent.e3.harm_eval_head.parameters():
            param.requires_grad_(False)

    # Replay buffers for harm eval training
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []

    actual_warmup = min(3, warmup_episodes) if dry_run else warmup_episodes
    actual_eval   = min(2, eval_episodes)   if dry_run else eval_episodes

    train_harm_steps = 0

    # ---- TRAINING PHASE ----
    agent.train()

    for ep in range(actual_warmup):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)

            latent = agent.sense(obs_body, obs_world)
            z_world_curr = _ensure_2d(latent.z_world.detach())

            agent.clock.advance()

            action_idx = random.randint(0, env.action_dim - 1)
            action_oh  = _action_to_onehot(action_idx, env.action_dim, device)
            agent._last_action = action_oh

            _, harm_signal, done, info, obs_dict = env.step(action_oh)
            hs = float(harm_signal)

            # E1 training (both conditions)
            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                e1_opt.zero_grad()
                e1_loss.backward()
                nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                e1_opt.step()

            # Harm eval training (ON condition only)
            if harm_eval_on:
                if hs < 0:
                    train_harm_steps += 1
                    harm_buf_pos.append(z_world_curr.detach())
                    if len(harm_buf_pos) > 1000:
                        harm_buf_pos = harm_buf_pos[-1000:]
                else:
                    harm_buf_neg.append(z_world_curr.detach())
                    if len(harm_buf_neg) > 1000:
                        harm_buf_neg = harm_buf_neg[-1000:]

                if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                    k_p = min(16, len(harm_buf_pos))
                    k_n = min(16, len(harm_buf_neg))
                    pi = torch.randperm(len(harm_buf_pos))[:k_p].tolist()
                    ni = torch.randperm(len(harm_buf_neg))[:k_n].tolist()
                    zw_b = torch.cat(
                        [harm_buf_pos[i] for i in pi] + [harm_buf_neg[i] for i in ni],
                        dim=0,
                    )
                    target = torch.cat([
                        torch.ones(k_p, 1, device=device),
                        torch.zeros(k_n, 1, device=device),
                    ], dim=0)
                    pred = agent.e3.harm_eval(zw_b)
                    harm_loss = F.mse_loss(pred, target)
                    if harm_loss.requires_grad:
                        e3_harm_opt.zero_grad()
                        harm_loss.backward()
                        nn.utils.clip_grad_norm_(
                            agent.e3.harm_eval_head.parameters(), 0.5
                        )
                        e3_harm_opt.step()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == actual_warmup - 1:
            print(
                f"  [train] seed={seed} cond={cond}"
                f" ep {ep + 1}/{actual_warmup}"
                f" harm_buf_pos={len(harm_buf_pos)}"
                f" harm_buf_neg={len(harm_buf_neg)}"
                f" train_harm_steps={train_harm_steps}",
                flush=True,
            )

    # ---- EVAL PHASE ----
    # We measure gradient ordering: approach -> contact
    # Phase detection:
    #   - "approach": harm_signal < 0 and > CONTACT_HARM_THRESH (proximity gradient, pre-contact)
    #   - "contact":  harm_signal <= CONTACT_HARM_THRESH (actual hazard contact)
    # Note: CausalGridWorldV2 uses proximity_harm_scale for approach and hazard_harm for contact.
    # We use the magnitude of harm_signal to distinguish phases.
    # approach_thresh is set so proximity signals (small negative) are approach,
    # contact signals (larger negative) are contact.

    agent.eval()

    # Collect (approach_score, contact_score) paired per episode
    # Pair = within the same episode, record the approach harm score and
    # the subsequent contact harm score
    approach_scores: List[float] = []   # harm_eval output at approach steps
    contact_scores:  List[float] = []   # harm_eval output at contact steps
    n_approach_steps = 0
    n_contact_steps  = 0
    total_eval_steps = 0

    for _ in range(actual_eval):
        _, obs_dict = env.reset()
        agent.reset()

        # Per-episode rolling: collect all approach and contact scores
        ep_approach_scores: List[float] = []
        ep_contact_scores:  List[float] = []

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                z_world_curr = _ensure_2d(latent.z_world.detach())
                agent.clock.advance()

                action_idx = random.randint(0, env.action_dim - 1)
                action_oh  = _action_to_onehot(action_idx, env.action_dim, device)
                agent._last_action = action_oh

                harm_score = float(agent.e3.harm_eval(z_world_curr).item())

            _, harm_signal, done, info, obs_dict = env.step(action_oh)
            hs = float(harm_signal)
            total_eval_steps += 1

            # Phase classification by harm_signal magnitude
            if hs < 0:
                # Any negative harm signal: distinguish approach vs contact by magnitude
                # Approach = small negative (proximity gradient only)
                # Contact  = large negative (actual hazard contact)
                if hs > CONTACT_HARM_THRESH:
                    # Approach (proximity signal, small negative)
                    ep_approach_scores.append(harm_score)
                    n_approach_steps += 1
                else:
                    # Contact (larger negative harm)
                    ep_contact_scores.append(harm_score)
                    n_contact_steps += 1

            if done:
                break

        # After each episode: collect any approach/contact pairs
        if ep_approach_scores:
            approach_scores.extend(ep_approach_scores)
        if ep_contact_scores:
            contact_scores.extend(ep_contact_scores)

    # Gradient ordering metric:
    # For each (approach_score, contact_score) pair drawn from the distributions,
    # what fraction has approach_score < contact_score?
    # We compute this as: P(approach < contact) using all cross-pairs.
    n_pairs = 0
    n_ordered = 0
    # Use up to 500 random cross-pairs to keep computation bounded
    max_pairs = 500
    if approach_scores and contact_scores:
        rng = random.Random(seed)
        for _ in range(max_pairs):
            a = rng.choice(approach_scores)
            c = rng.choice(contact_scores)
            n_pairs += 1
            if a < c:
                n_ordered += 1

    gradient_score = n_ordered / max(1, n_pairs)
    escalation_gap = _mean_safe(contact_scores) - _mean_safe(approach_scores)

    harm_rate = n_contact_steps / max(1, total_eval_steps)
    n_approach_contact_pairs = min(len(approach_scores), len(contact_scores))

    print(
        f"  [eval] seed={seed} cond={cond}"
        f" gradient_score={gradient_score:.4f}"
        f" escalation_gap={escalation_gap:+.4f}"
        f" n_approach={len(approach_scores)}"
        f" n_contact={len(contact_scores)}"
        f" harm_rate={harm_rate:.4f}"
        f" n_pairs={n_pairs}"
        f" total_eval_steps={total_eval_steps}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond,
        "harm_eval_on": harm_eval_on,
        "gradient_score": float(gradient_score),
        "escalation_gap": float(escalation_gap),
        "n_approach_steps": int(len(approach_scores)),
        "n_contact_steps": int(len(contact_scores)),
        "n_approach_contact_pairs": int(n_approach_contact_pairs),
        "n_ordered_pairs": int(n_ordered),
        "n_cross_pairs": int(n_pairs),
        "harm_rate": float(harm_rate),
        "total_eval_steps": int(total_eval_steps),
        "mean_approach_score": float(_mean_safe(approach_scores)),
        "mean_contact_score": float(_mean_safe(contact_scores)),
        "train_harm_steps": int(train_harm_steps),
        "harm_buf_pos_final": int(len(harm_buf_pos)),
        "harm_buf_neg_final": int(len(harm_buf_neg)),
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run(
    seeds: Tuple = (42, 123),
    warmup_episodes: int = 400,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_harm_scale: float = 0.01,
    env_drift_prob: float = 0.3,
    env_drift_interval: int = 3,
    dry_run: bool = False,
    **kwargs,
) -> dict:
    """Discriminative pair: HARM_EVAL_ON vs HARM_EVAL_ABLATED."""
    results_on:  List[Dict] = []
    results_off: List[Dict] = []

    for seed in seeds:
        for harm_eval_on in [True, False]:
            cond = "HARM_EVAL_ON" if harm_eval_on else "HARM_EVAL_ABLATED"
            print(
                f"\n[V3-EXQ-123] {cond} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" steps={steps_per_episode}"
                f" alpha_world={alpha_world}"
                f" harm_scale={harm_scale}"
                f" proximity_harm_scale={proximity_harm_scale}"
                f" drift_prob={env_drift_prob}"
                f" {'DRY_RUN' if dry_run else ''}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                harm_eval_on=harm_eval_on,
                warmup_episodes=warmup_episodes,
                eval_episodes=eval_episodes,
                steps_per_episode=steps_per_episode,
                self_dim=self_dim,
                world_dim=world_dim,
                lr=lr,
                alpha_world=alpha_world,
                alpha_self=alpha_self,
                harm_scale=harm_scale,
                proximity_harm_scale=proximity_harm_scale,
                env_drift_prob=env_drift_prob,
                env_drift_interval=env_drift_interval,
                dry_run=dry_run,
            )
            if harm_eval_on:
                results_on.append(r)
            else:
                results_off.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    grad_on   = _avg(results_on,  "gradient_score")
    grad_off  = _avg(results_off, "gradient_score")
    grad_delta = grad_on - grad_off

    # Per-seed C3 check (consistency)
    per_seed_c3 = [
        ron["gradient_score"] > roff["gradient_score"]
        for ron, roff in zip(results_on, results_off)
    ]
    c3_pass = all(per_seed_c3)

    # C4: data quality -- min approach-contact pairs across ON cells
    min_pairs = min(r["n_approach_contact_pairs"] for r in results_on)

    # C5: escalation gap diagnostic (ON condition only)
    mean_gap_on = _avg(results_on, "escalation_gap")
    c5_pass = mean_gap_on >= THRESH_C5_GAP

    c1_pass = grad_delta >= THRESH_C1_GRAD_DELTA
    c2_pass = grad_on    >= THRESH_C2_GRAD_ABS
    c4_pass = min_pairs  >= THRESH_C4_MIN_PAIRS
    all_pass = c1_pass and c2_pass and c3_pass and c4_pass

    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])
    status = "PASS" if all_pass else "FAIL"

    print(f"\n[V3-EXQ-123] Final results:", flush=True)
    print(
        f"  gradient_score_ON={grad_on:.4f}  gradient_score_ABLATED={grad_off:.4f}"
        f"  delta={grad_delta:+.4f}  (C1 thresh >={THRESH_C1_GRAD_DELTA})"
        f"  C1={'PASS' if c1_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"  gradient_score_ON={grad_on:.4f}  (C2 absolute thresh >={THRESH_C2_GRAD_ABS})"
        f"  C2={'PASS' if c2_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"  per_seed_ON_vs_ABLATED: {per_seed_c3}"
        f"  C3={'PASS' if c3_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"  min_approach_contact_pairs={min_pairs}  (C4 thresh >={THRESH_C4_MIN_PAIRS})"
        f"  C4={'PASS' if c4_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"  mean_escalation_gap_ON={mean_gap_on:+.4f}"
        f"  (C5 thresh >={THRESH_C5_GAP})"
        f"  C5={'PASS' if c5_pass else 'FAIL'} (diagnostic only)",
        flush=True,
    )
    print(f"  status={status}  ({criteria_met}/5 criteria met, 4 required)", flush=True)

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: gradient_score_ON={grad_on:.4f} vs"
            f" gradient_score_ABLATED={grad_off:.4f}"
            f" (delta={grad_delta:+.4f}, needs >={THRESH_C1_GRAD_DELTA})."
            " ON condition does not show substantially more ordered harm gradient"
            " than the frozen random ablation."
            " Possible causes: (1) harm_eval head does not discriminate approach vs"
            " contact, (2) proximity_harm_scale too small to create detectable gradient,"
            " (3) world_dim=32 insufficient to encode proximity phase information."
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: gradient_score_ON={grad_on:.4f} (needs >={THRESH_C2_GRAD_ABS})."
            " ON condition did not achieve reliable gradient ordering."
            " harm_eval head may not have learned proximity-phase distinctions."
            " Check harm_buf_pos_final -- insufficient positive samples."
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: per_seed direction inconsistent ({per_seed_c3})."
            " ON did not consistently beat ABLATED across seeds."
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: min_approach_contact_pairs={min_pairs} < {THRESH_C4_MIN_PAIRS}."
            " Insufficient approach-contact pairs in eval."
            " Increase proximity_harm_scale or hazard density."
        )
    if not c5_pass:
        failure_notes.append(
            f"C5 (diagnostic): mean_escalation_gap_ON={mean_gap_on:+.4f} < {THRESH_C5_GAP}."
            " Mean contact harm score does not substantially exceed mean approach score."
            " The harm_eval signal may be non-discriminative across phases."
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    if all_pass:
        interpretation = (
            "MECH-102 SUPPORTED: E3 harm_eval produces graded escalating signal"
            f" -- gradient_score_ON={grad_on:.4f} (>={THRESH_C2_GRAD_ABS}) and"
            f" outperforms frozen-random ablation by delta={grad_delta:+.4f}"
            f" (>={THRESH_C1_GRAD_DELTA})."
            " Direction consistent across all seeds (C3)."
            " Sufficient approach-contact pairs observed (C4)."
            " Harm evaluation signal escalates with proximity phase (approach -> contact)"
            " rather than producing a binary flag."
        )
    elif c1_pass and not c2_pass:
        interpretation = (
            "PARTIAL: C1 passes (relative delta={:.4f}) but C2 fails"
            " (gradient_score_ON={:.4f} < {:.2f})."
            " ON condition shows relative advantage over ablated but absolute"
            " ordering is below threshold."
            " Training budget or harm signal density insufficient."
        ).format(grad_delta, grad_on, THRESH_C2_GRAD_ABS)
    elif c2_pass and not c1_pass:
        interpretation = (
            "PARTIAL: C2 passes (gradient_score_ON={:.4f} >= {:.2f}) but C1 fails"
            " (delta={:+.4f} < {:.2f})."
            " ON condition shows above-chance gradient ordering but the ablated"
            " condition shows similar ordering (possibly due to random spatial correlations)."
            " MECH-102 gradient is not cleanly isolable from noise at this scale."
        ).format(grad_on, THRESH_C2_GRAD_ABS, grad_delta, THRESH_C1_GRAD_DELTA)
    else:
        interpretation = (
            "MECH-102 NOT SUPPORTED: E3 harm_eval does not produce graded escalation."
            f" gradient_score_ON={grad_on:.4f}, gradient_score_ABLATED={grad_off:.4f},"
            f" delta={grad_delta:+.4f}."
            " Harm evaluation signal did not escalate systematically from approach to"
            " contact phase. Consistent with a binary flag (contact/no-contact)"
            " rather than a graded ethical valuation signal."
        )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    per_on_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" gradient_score={r['gradient_score']:.4f}"
        f" escalation_gap={r['escalation_gap']:+.4f}"
        f" n_approach={r['n_approach_steps']}"
        f" n_contact={r['n_contact_steps']}"
        f" mean_approach={r['mean_approach_score']:.4f}"
        f" mean_contact={r['mean_contact_score']:.4f}"
        f" train_harm_steps={r['train_harm_steps']}"
        for r in results_on
    )
    per_off_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" gradient_score={r['gradient_score']:.4f}"
        f" escalation_gap={r['escalation_gap']:+.4f}"
        f" n_approach={r['n_approach_steps']}"
        f" n_contact={r['n_contact_steps']}"
        for r in results_off
    )

    summary_markdown = (
        f"# V3-EXQ-123 -- MECH-102 Harm Escalation Gradient Discriminative Pair\n\n"
        f"**Status:** {status}\n"
        f"**Claim:** MECH-102\n"
        f"**Proposal:** EXP-0021 / EVB-0017\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps\n"
        f"**env_drift_prob:** {env_drift_prob}  **env_drift_interval:** {env_drift_interval}\n"
        f"**harm_scale:** {harm_scale}  **proximity_harm_scale:** {proximity_harm_scale}\n\n"
        f"## Design\n\n"
        f"HARM_EVAL_ON: E3 harm_eval_head trained on harm_signal labels."
        f" Gradient score = P(approach_harm_score < contact_harm_score) via cross-pairs.\n"
        f"HARM_EVAL_ABLATED: E3 harm_eval_head frozen at random init (no training)."
        f" Gradient score should be ~0.5 (chance).\n\n"
        f"Phase detection: harm_signal in ({CONTACT_HARM_THRESH}, 0) = approach"
        f" (proximity gradient); harm_signal <= {CONTACT_HARM_THRESH} = contact"
        f" (hazard impact). Calibrated for hazard_harm={harm_scale},"
        f" proximity_harm_scale={proximity_harm_scale}.\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: gradient_score_ON - gradient_score_ABLATED >= {THRESH_C1_GRAD_DELTA}"
        f"  (relative advantage)\n"
        f"C2: gradient_score_ON >= {THRESH_C2_GRAD_ABS}  (absolute ordering above chance)\n"
        f"C3: gradient_score_ON > gradient_score_ABLATED for ALL seeds  (consistency)\n"
        f"C4: min_approach_contact_pairs >= {THRESH_C4_MIN_PAIRS}  (data quality)\n"
        f"C5 (diagnostic): escalation_gap_ON >= {THRESH_C5_GAP}  (mean contact > mean approach)\n\n"
        f"## Aggregate Results\n\n"
        f"| Metric | HARM_EVAL_ON | HARM_EVAL_ABLATED | Delta | Pass |\n"
        f"|--------|-------------|------------------|-------|------|\n"
        f"| gradient_score (C1 delta) | {grad_on:.4f} | {grad_off:.4f}"
        f" | {grad_delta:+.4f} | {'YES' if c1_pass else 'NO'} |\n"
        f"| gradient_score >= {THRESH_C2_GRAD_ABS} (C2) | {grad_on:.4f} | -- | --"
        f" | {'YES' if c2_pass else 'NO'} |\n"
        f"| seed consistency (C3) | {per_seed_c3} | -- | --"
        f" | {'YES' if c3_pass else 'NO'} |\n"
        f"| min_pairs (C4) | {min_pairs} | -- | --"
        f" | {'YES' if c4_pass else 'NO'} |\n"
        f"| escalation_gap (C5 diag) | {mean_gap_on:+.4f} | -- | --"
        f" | {'YES' if c5_pass else 'NO'} |\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n\n"
        f"## Per-Seed (HARM_EVAL_ON)\n\n"
        f"{per_on_rows}\n\n"
        f"## Per-Seed (HARM_EVAL_ABLATED)\n\n"
        f"{per_off_rows}\n"
        f"{failure_section}\n"
    )

    return {
        "status": status,
        "metrics": {
            "gradient_score_on":              float(grad_on),
            "gradient_score_ablated":         float(grad_off),
            "grad_delta":                     float(grad_delta),
            "min_approach_contact_pairs":     float(min_pairs),
            "mean_escalation_gap_on":         float(mean_gap_on),
            "crit1_pass":                     1.0 if c1_pass else 0.0,
            "crit2_pass":                     1.0 if c2_pass else 0.0,
            "crit3_pass":                     1.0 if c3_pass else 0.0,
            "crit4_pass":                     1.0 if c4_pass else 0.0,
            "crit5_pass":                     1.0 if c5_pass else 0.0,
            "criteria_met":                   float(criteria_met),
            "n_seeds":                        float(len(seeds)),
            "alpha_world":                    float(alpha_world),
            "harm_scale":                     float(harm_scale),
            "proximity_harm_scale":           float(proximity_harm_scale),
            "env_drift_prob":                 float(env_drift_prob),
        },
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if (c1_pass or c2_pass) else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "per_seed_on":  results_on,
        "per_seed_off": results_off,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",            type=int,   nargs="+", default=[42, 123])
    parser.add_argument("--warmup",           type=int,   default=400)
    parser.add_argument("--eval-eps",         type=int,   default=50)
    parser.add_argument("--steps",            type=int,   default=200)
    parser.add_argument("--self-dim",         type=int,   default=32)
    parser.add_argument("--world-dim",        type=int,   default=32)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--alpha-world",      type=float, default=0.9)
    parser.add_argument("--alpha-self",       type=float, default=0.3)
    parser.add_argument("--harm-scale",       type=float, default=0.02)
    parser.add_argument("--proximity-scale",  type=float, default=0.01)
    parser.add_argument("--drift-prob",       type=float, default=0.3)
    parser.add_argument("--drift-interval",   type=int,   default=3)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Quick dry-run: 3 warmup, 2 eval, 50 steps per cell. Writes JSON.",
    )
    args = parser.parse_args()

    result = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        self_dim=args.self_dim,
        world_dim=args.world_dim,
        lr=args.lr,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        harm_scale=args.harm_scale,
        proximity_harm_scale=args.proximity_scale,
        env_drift_prob=args.drift_prob,
        env_drift_interval=args.drift_interval,
        dry_run=args.dry_run,
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

    print(f"\n[V3-EXQ-123] Result written to {out_path}", flush=True)
    print(f"[V3-EXQ-123] status={result['status']}", flush=True)
