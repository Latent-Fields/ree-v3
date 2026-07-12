#!/opt/local/bin/python3
"""V3-EXQ-590b -- MECH-314a per-candidate RBF novelty Goldilocks calibration.

The MECH-314a-Phase-2-impl re-queue path, unblocked by V3-EXQ-648a (the
load-bearing C2 PASS that confirmed the per-candidate curiosity bias is
non-zero-RANGE on the e2.world_forward-fed substrate; gapA-cluster autopsy
2026-06-07). This is the FIRST behavioural-evidence run for MECH-314a -- it
calibrates how strongly the per-candidate curiosity bias should influence
committed action selection, and whether that influence buys an exploratory
benefit. claim_ids=[MECH-314a], experiment_purpose=evidence.

=== WHY THE SWEPT KNOB IS modulatory_authority_gain, NOT curiosity_novelty_weight ===

The 2026-05-25 root-cause finding (V3-EXQ-571) + the V3-EXQ-590a autopsy
established that the legacy "sweep novelty_bonus_weight" Goldilocks is degenerate
on this substrate: 590a's five weight arms produced BYTE-IDENTICAL coverage /
H_pos / novelty_ema (substrate_queue do_not_adopt_from_590a). Two corrections
have since landed that re-pose the question correctly:

  (1) V3-EXQ-648a (2026-06-07): curiosity_candidate_source="e2_world_forward"
      feeds the per-candidate novelty from the SD-056-trained
      e2.world_forward(z0,a_i) predictions, so the curiosity bias carries genuine
      cross-candidate RANGE (C2 PASS). The channel is live.
  (2) modulatory-bias-selection-authority (2026-06-03, float32 fix 2026-06-06
      V3-EXQ-643a PASS): the gap-relative authority RESCALES the combined
      modulatory contribution so its range == modulatory_authority_gain *
      raw_score_range, then adds it to the primary scores. This is what lets a
      small modulatory bias actually move the committed argmin (the 604a/624a/614d
      "drowning" fix).

Crucially, with MECH-314a curiosity as the SOLE modulatory channel, the authority
renormalizes the curiosity bias's RANGE to gain*raw_score_range REGARDLESS of
curiosity_novelty_weight -- the weight only scales an already-sole signal that is
then renormalized away, and the per-candidate PATTERN (which candidates are more
novel) is scale-invariant. So sweeping curiosity_novelty_weight under authority-ON
reproduces the 590a byte-identical null by construction. The faithful magnitude
knob on the post-643a substrate is the AUTHORITY GAIN: it sets how much the
(live, ranged) curiosity pattern is allowed to move selection. The Goldilocks
question becomes: at what authority gain does the per-candidate curiosity bias
maximise exploration WITHOUT drowning value-based (harm/goal) selection?

  gain = 0.0  -- curiosity present + ranged but NO selection authority (the
                 behavioural control: the bias cannot move the argmin; exploration
                 is whatever value-driven SP-CEM produces).
  gain rising -- curiosity gets progressively more authority over the committed
                 action; exploration should lift, then (too high) over-explore /
                 ignore value -> inverted-U Goldilocks.

curiosity_novelty_weight is held FIXED at the 648a value (0.25) so the curiosity
channel has a stable, non-degenerate per-candidate range for the authority to
amplify; the swept knob is modulatory_authority_gain.

=== SUBSTRATE (identical across arms except authority gain) ===

The 648a-validated config: SP-CEM main path (action-divergent pool) + V_s stack +
SD-056 online contrastive (e2.world_forward action-conditional divergence; rollout
output-norm clamp ON per the 643a numerical-stability lesson) + MECH-314a Phase-2
visitation-buffer novelty (curiosity_novelty_source="visitation",
first-action-onehot auto-augmentation) + curiosity_candidate_source=
"e2_world_forward". MECH-314a novelty is the SOLE modulatory bias channel (dacc /
lateral_pfc / ofc / mech295 / tonic_vigor / noise_floor / e3_score_diversity all
OFF). use_modulatory_selection_authority=True; modulatory_authority_gain swept.

Harm-free env (num_hazards=0): the residue field stays empty (so the Phase-1
residue source would be silent), which is exactly why the visitation source is
used; SP-CEM + resources give action-divergent candidates for SD-056 to train
z_world divergence on. Exploration is measured as env pos_entropy (Shannon entropy
of the agent's position-visitation window) -- the canonical "did curiosity make it
explore more of the grid" readout (the V3-EXQ-590 lineage DV), with selected-action
class entropy as a self-contained secondary.

=== ARMS (5 gains x 3 seeds) ===

  ARM_G00  modulatory_authority_gain=0.00  (curiosity has NO selection authority; control)
  ARM_G25  modulatory_authority_gain=0.25
  ARM_G50  modulatory_authority_gain=0.50  (the landed default)
  ARM_G100 modulatory_authority_gain=1.00
  ARM_G200 modulatory_authority_gain=2.00

=== READINESS / NON-VACUITY GATE (same-statistic; the V3-EXQ-643 lesson) ===

Two load-bearing legs + a finite guard. If any is unmet the run self-routes
substrate_not_ready_requeue (non_contributory), NEVER a MECH-314a weakens -- a
blind/flat readout must not masquerade as a falsification.

  Leg 1 (channel live; the 643a guard): at the highest-gain arm the per-candidate
    curiosity_bias_range_mean (pre-rescale cross-candidate RANGE) > BIAS_RANGE_FLOOR
    on >= MIN_SEEDS. Confirms there is a real per-candidate pattern for the
    authority to amplify (scaling zero is still zero).
  Leg 2 (the knob moves the routed readout; SAME statistic the Goldilocks routes
    on): RANGE across the gain arms of the healthy-seed-mean H_pos >= H_POS_RANGE_FLOOR.
    If exploration is flat across every authority gain, the lever produces no
    behavioural gradient to calibrate -> substrate_not_ready, not a verdict.
  Leg 3 (finite guard, 643a): max cand_world_pairwise_dist finite and < ceil
    (SD-056 online training numerical-stability).

=== ACCEPTANCE (pre-registered) ===

PASS (supports MECH-314a) = readiness met AND a Goldilocks gain is identified:
  the best non-zero-gain arm's mean H_pos exceeds the gain=0 control by
  >= H_POS_LIFT_MARGIN on >= MIN_SEEDS of 3 seeds (paired per seed). The per-candidate
  curiosity bias, given selection authority, improves exploration. nonmonotone=True
  (interior peak > both neighbours) reports the inverted-U over-authority ceiling.
FAIL (does_not_support MECH-314a behavioural reading) = readiness met (the knob
  moves H_pos and the channel is live) BUT no non-zero gain beats the control by
  the margin on a majority of seeds -- curiosity-with-authority does not buy
  exploration here.
substrate_not_ready_requeue = any readiness leg below floor / non-finite.

=== INTERPRETATION GRID ===

| Outcome                                   | label                              | next action |
|-------------------------------------------|------------------------------------|-------------|
| any readiness leg below floor / non-finite | substrate_not_ready_requeue        | re-queue as 590c at higher P0 budget / check e2+authority wiring; do NOT weaken MECH-314a |
| readiness OK + Goldilocks lift found       | mech314a_goldilocks_identified     | PASS (supports MECH-314a); adopt best gain for the curiosity channel |
| readiness OK + no lift over control        | mech314a_no_exploratory_benefit    | FAIL (does_not_support); /failure-autopsy the behavioural reading |

architecture_epoch: "ree_hybrid_guardrails_v1"

Smoke:
  /opt/local/bin/python3 experiments/v3_exq_590b_mech314a_novelty_goldilocks.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import (  # noqa: E402
    compute_arm_fingerprint,
    reset_all_rng,
)
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_590b_mech314a_novelty_goldilocks"
QUEUE_ID = "V3-EXQ-590b"
CLAIM_IDS: List[str] = ["MECH-314a"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 60            # SD-056 contrastive warmup (matches V3-EXQ-648a)
P1_MEASUREMENT_EPISODES = 30       # exploration measurement window
STEPS_PER_EPISODE = 200
MEASURE_AFTER_TICK = 20            # within-episode warmup before reading bias range

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_STEPS = 30
DRY_RUN_MEASURE_AFTER_TICK = 2

# Swept knob: modulatory-selection-authority gain (the post-643a magnitude knob).
AUTHORITY_GAINS: List[float] = [0.0, 0.25, 0.5, 1.0, 2.0]

# Pre-registered thresholds.
BIAS_RANGE_FLOOR = 1.0e-4     # readiness leg 1: curiosity bias non-zero per-candidate RANGE
H_POS_RANGE_FLOOR = 0.05      # readiness leg 2: H_pos RANGE across gain arms (same statistic)
MAGNITUDE_CEIL = 1.0e6        # readiness leg 3: rolled-out z_world finite guard (643a)
H_POS_LIFT_MARGIN = 0.05      # PASS: best non-zero gain H_pos - control H_pos, per seed
MIN_SEEDS_FOR_PASS = 2        # of 3

# SD-056 online contrastive training (mirror V3-EXQ-648a / 604a / 569d harness).
SD056_WEIGHT = 0.05
E2_CONTRASTIVE_LR = 1e-3
E2_TRAIN_EVERY_K_TICKS = 1
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 256
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0

# Curiosity magnitudes held FIXED (648a values); the authority gain is the swept knob.
CURIOSITY_BIAS_SCALE = 0.5
CURIOSITY_WEIGHT = 0.25
VISITATION_BUFFER_LEN = 256

# HARM-FREE env (num_hazards=0): residue field stays empty (visitation source is the
# point); SP-CEM + resources still give action-divergent candidates for SD-056.
ENV_KWARGS: Dict[str, Any] = dict(
    size=12,
    num_hazards=0,
    num_resources=5,
    hazard_harm=0.0,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.0,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=10,
)


def _arm_id(gain: float) -> str:
    return f"ARM_G{int(round(gain * 100)):02d}"


ARMS: List[Dict[str, Any]] = [
    {"arm_id": _arm_id(g), "authority_gain": g, "is_control": (g == 0.0)}
    for g in AUTHORITY_GAINS
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    """648a-validated substrate; MECH-314a curiosity is the SOLE modulatory channel,
    modulatory-selection-authority ON with the per-arm gain.

    SD-056 is trained online on every arm (the e2.world_forward divergence the
    curiosity novelty consumes); the rollout output-norm clamp is ON per the 643a
    numerical-stability lesson (online SD-056 can explode rolled-out z_world).
    """
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        use_harm_stream=True,
        z_harm_dim=32,
        use_affective_harm_stream=True,
        z_harm_a_dim=16,
        harm_history_len=10,
        z_goal_enabled=True,
        goal_weight=0.5,
        drive_weight=2.0,
        e1_goal_conditioned=True,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        benefit_eval_enabled=True,
        benefit_weight=1.0,
        # ARC-065 SP-CEM (Layer A) -- main-path default (action-divergent pool)
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # MECH-314a curiosity is the SOLE modulatory channel -- all other bias channels OFF
        use_e3_score_diversity=False,
        use_noise_floor=False,
        use_tonic_vigor=False,
        use_dacc=False,
        use_lateral_pfc_analog=False,
        use_ofc_analog=False,
        use_mech295_liking_bridge=False,
        # V_s substrate (main-path default; identical across arms)
        use_per_stream_vs=True,
        use_per_region_vs=True,
        use_event_segmenter=True,
        use_invalidation_trigger=True,
        use_anchor_sets=True,
        # SD-056 substrate present + trained online on every arm
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=2.0,
        # MECH-314 structured curiosity -- novelty sub-flavour ON (Phase-2 Candidate 5A)
        use_structured_curiosity=True,
        use_curiosity_novelty=True,
        use_curiosity_uncertainty=False,
        use_curiosity_learning_progress=False,
        curiosity_bias_scale=CURIOSITY_BIAS_SCALE,
        curiosity_novelty_weight=CURIOSITY_WEIGHT,
        curiosity_novelty_source="visitation",
        curiosity_visitation_buffer_len=VISITATION_BUFFER_LEN,
        curiosity_use_first_action_onehot=True,
        curiosity_first_action_augmentation_policy="auto",
        # V3-EXQ-648a fix: novelty + auto-augment _candidate_spread from
        # e2.world_forward(z0,a_i) (the SD-056-divergent consumed representation).
        curiosity_candidate_source="e2_world_forward",
        # modulatory-bias-selection-authority (643a): the swept knob. gain=0.0 ->
        # curiosity has no selection authority (the behavioural control).
        use_modulatory_selection_authority=True,
        modulatory_authority_gain=float(arm["authority_gain"]),
    )
    agent = REEAgent(cfg)
    # Per-channel score-bias decomposition so select_action records the
    # per-candidate curiosity bias range (the readiness leg-1 statistic).
    agent.e3.e3_score_decomp_enabled = True
    return agent


# ---------------------------------------------------------------------------
# SD-056 online contrastive helpers (from V3-EXQ-648a / 604a / 569d)
# ---------------------------------------------------------------------------

def _first_actions_K(candidates) -> torch.Tensor:
    rows = []
    for traj in candidates:
        rows.append(traj.actions[:, 0, :].detach().reshape(-1))
    return torch.stack(rows, dim=0)


def _obs(d: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    h = d.get(key)
    if h is None:
        return None
    return h.float().unsqueeze(0) if h.dim() == 1 else h.float()


def _sample_class_diverse_batch(
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    k: int,
    rng: random.Random,
) -> Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    if len(buffer) < MIN_BUFFER_BEFORE_TRAIN:
        return None
    pool = list(buffer)
    rng.shuffle(pool)
    seen_classes: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for tup in pool:
        cls = int(tup[1].argmax().item())
        if cls not in seen_classes:
            seen_classes[cls] = tup
        if len(seen_classes) >= k:
            break
    if len(seen_classes) < MIN_CLASSES_FOR_TRAIN:
        return None
    samples = list(seen_classes.values())
    picked_ids = {id(s) for s in samples}
    for tup in pool:
        if len(samples) >= k:
            break
        if id(tup) in picked_ids:
            continue
        samples.append(tup)
        picked_ids.add(id(tup))
    return samples


def _e2_contrastive_step(
    agent: REEAgent,
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    optimiser: torch.optim.Optimizer,
    rng: random.Random,
) -> Optional[float]:
    batch = _sample_class_diverse_batch(buffer, CONTRASTIVE_BATCH_K, rng)
    if batch is None:
        return None
    z0_K = torch.stack([t[0] for t in batch]).to(agent.device)
    actions_K = torch.stack([t[1] for t in batch]).to(agent.device)
    z1_K = torch.stack([t[2] for t in batch]).to(agent.device)
    optimiser.zero_grad(set_to_none=True)
    loss = agent.e2.world_forward_contrastive_loss(
        z_world_0=z0_K,
        actions=actions_K,
        z_world_1_targets=z1_K,
        simulation_mode=False,
    )
    if not torch.is_tensor(loss):
        return None
    loss_val = float(loss.detach().item())
    if not math.isfinite(loss_val):
        return loss_val
    if not loss.requires_grad or loss_val == 0.0:
        return loss_val
    weighted = SD056_WEIGHT * loss
    weighted.backward()
    torch.nn.utils.clip_grad_norm_(agent.e2.parameters(), max_norm=MAX_GRAD_NORM)
    optimiser.step()
    return loss_val


def _action_class_entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for n in counts.values():
        if n <= 0:
            continue
        p = n / total
        ent -= p * math.log(p)
    return float(ent)


# ---------------------------------------------------------------------------
# Per-(seed, arm) runner
# ---------------------------------------------------------------------------

def _run_seed_arm(
    arm: Dict[str, Any],
    seed: int,
    p0_episodes: int,
    p1_episodes: int,
    steps_per_episode: int,
    measure_after_tick: int,
) -> Dict[str, Any]:
    # Full RNG reset at cell entry -> arm_fingerprint is order-independent.
    reset_all_rng(seed)

    env = _make_env(seed)
    agent = _make_agent(env, arm)
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)

    transition_buffer: Deque[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)

    total_train_eps = p0_episodes + p1_episodes

    # P1 accumulators.
    ep_h_pos: List[float] = []                # per-P1-episode position entropy
    action_class_counts: Counter = Counter()  # over all P1 ticks
    curiosity_range_vals: List[float] = []
    pairwise_dists: List[float] = []
    pairwise_dist_max_seen = 0.0
    n_p0_ticks = 0
    n_p1_ticks = 0
    n_p1_ticks_past_window = 0
    n_contrastive_steps = 0
    n_h_pos_fallback = 0                       # episodes where pos_entropy was absent
    error_note: Optional[str] = None

    for ep in range(total_train_eps):
        is_p1 = ep >= p0_episodes
        phase_label = "P1" if is_p1 else "P0"

        _, obs_dict = env.reset()
        agent.reset()

        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        pending_capture: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        tick_in_ep = 0
        last_info: Dict[str, Any] = {}
        ep_action_counts: Counter = Counter()

        for _step in range(steps_per_episode):
            body = obs_dict["body_state"].float()
            world = obs_dict["world_state"].float()
            if body.dim() == 1:
                body = body.unsqueeze(0)
            if world.dim() == 1:
                world = world.unsqueeze(0)

            latent = agent.sense(
                obs_body=body, obs_world=world,
                obs_harm=_obs(obs_dict, "harm_obs"),
                obs_harm_a=_obs(obs_dict, "harm_obs_a"),
                obs_harm_history=_obs(obs_dict, "harm_history"),
            )

            if pending_capture is not None:
                z0_prev, a_prev = pending_capture
                z1_obs = latent.z_world.detach().reshape(-1).clone()
                if (
                    torch.isfinite(z0_prev).all()
                    and torch.isfinite(a_prev).all()
                    and torch.isfinite(z1_obs).all()
                ):
                    transition_buffer.append((z0_prev, a_prev, z1_obs))
                pending_capture = None

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(
                    z_self_prev, action_prev, latent.z_self.detach()
                )

            ticks = agent.clock.advance()
            wdim = latent.z_world.shape[-1]
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            past_window = is_p1 and tick_in_ep >= measure_after_tick
            if past_window and candidates and len(candidates) >= 2:
                actions_K = _first_actions_K(candidates).to(agent.device)
                z0 = latent.z_world.detach()
                with torch.no_grad():
                    dist = float(
                        agent.e2.cand_world_pairwise_dist(z0, actions_K).item()
                    )
                if math.isfinite(dist):
                    pairwise_dists.append(dist)
                    pairwise_dist_max_seen = max(pairwise_dist_max_seen, dist)

            if agent.goal_state is not None:
                try:
                    energy = float(body[0, 3].item())
                except Exception:
                    energy = 1.0
                drive_level = max(0.0, 1.0 - energy)
                agent.update_z_goal(benefit_exposure=0.0, drive_level=drive_level)

            action = agent.select_action(candidates, ticks)
            if action is None:
                idx = int(np.random.randint(0, env.action_dim))
                action = torch.zeros(1, env.action_dim, device=agent.device)
                action[0, idx] = 1.0
                agent._last_action = action
            if not torch.isfinite(action).all():
                if error_note is None:
                    error_note = (
                        f"non-finite action at arm={arm['arm_id']} seed={seed} "
                        f"phase={phase_label} ep={ep} step={_step}"
                    )
                break

            # Per-candidate curiosity bias RANGE (readiness leg 1) captured AFTER
            # select_action (its single curiosity call set the decomposition).
            if past_window:
                decomp = getattr(agent, "_last_score_bias_decomp", {}) or {}
                crange = float(decomp.get("curiosity_bias_range_mean", 0.0))
                if math.isfinite(crange):
                    curiosity_range_vals.append(crange)
                ep_action_counts.update([int(action.argmax().item())])
                n_p1_ticks_past_window += 1

            if is_p1:
                n_p1_ticks += 1
            else:
                n_p0_ticks += 1

            if tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0:
                loss_val = _e2_contrastive_step(
                    agent=agent, buffer=transition_buffer,
                    optimiser=e2_opt, rng=sample_rng,
                )
                if loss_val is not None and math.isfinite(loss_val) and is_p1:
                    n_contrastive_steps += 1

            if (
                torch.isfinite(latent.z_world).all()
                and torch.isfinite(action).all()
            ):
                pending_capture = (
                    latent.z_world.detach().reshape(-1).clone(),
                    action.detach().reshape(-1).clone(),
                )

            _, harm_signal, done, info, next_obs_dict = env.step(action)
            last_info = info if isinstance(info, dict) else {}

            with torch.no_grad():
                agent.update_residue(
                    harm_signal=float(harm_signal),
                    world_delta=None,
                    hypothesis_tag=False,
                    owned=True,
                )

            z_self_prev = latent.z_self.detach()
            action_prev = action
            obs_dict = next_obs_dict
            tick_in_ep += 1
            if done:
                break

        # End-of-episode exploration readout (P1 only).
        if is_p1:
            h_pos_raw = last_info.get("pos_entropy", None)
            if h_pos_raw is None:
                h_pos_raw = obs_dict.get("pos_entropy", None) if isinstance(obs_dict, dict) else None
            if h_pos_raw is None or not math.isfinite(float(h_pos_raw)) or float(h_pos_raw) < 0.0:
                # Fallback: per-episode selected-action class entropy (self-contained).
                h_pos_val = _action_class_entropy(ep_action_counts)
                n_h_pos_fallback += 1
            else:
                h_pos_val = float(h_pos_raw)
            ep_h_pos.append(h_pos_val)
            action_class_counts.update(ep_action_counts)

        if (ep + 1) % 10 == 0 or (ep + 1) == total_train_eps:
            print(
                f"  [train] arm={arm['arm_id']} seed={seed} phase={phase_label} "
                f"ep {ep + 1}/{total_train_eps}",
                flush=True,
            )

    def _mean(xs: List[float], default: float = 0.0) -> float:
        return float(sum(xs) / len(xs)) if xs else default

    return {
        "arm_id": arm["arm_id"],
        "authority_gain": float(arm["authority_gain"]),
        "is_control": bool(arm["is_control"]),
        "seed": int(seed),
        "n_p0_ticks": int(n_p0_ticks),
        "n_p1_ticks": int(n_p1_ticks),
        "n_p1_ticks_past_window": int(n_p1_ticks_past_window),
        "n_p1_episodes": int(len(ep_h_pos)),
        "n_contrastive_steps": int(n_contrastive_steps),
        "n_h_pos_fallback_episodes": int(n_h_pos_fallback),
        "error_note": error_note,
        # PRIMARY exploration DV (per-episode mean over P1).
        "h_pos_mean": round(_mean(ep_h_pos), 6),
        "ep_h_pos_last5": [round(x, 4) for x in ep_h_pos[-5:]],
        # Secondary exploration DV (pooled selected-action class entropy over P1).
        "action_class_entropy": round(_action_class_entropy(action_class_counts), 6),
        # Readiness leg 1: per-candidate curiosity bias range (pre-rescale).
        "curiosity_bias_range_mean": round(_mean(curiosity_range_vals), 8),
        # Readiness leg 3 input.
        "cand_world_pairwise_dist_mean": round(_mean(pairwise_dists), 6),
        "cand_world_pairwise_dist_max": round(pairwise_dist_max_seen, 6),
    }


# ---------------------------------------------------------------------------
# Cross-arm evaluation
# ---------------------------------------------------------------------------

def _arm_rows(rows: List[Dict[str, Any]], arm_id: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("arm_id") == arm_id]


def _n_seeds_satisfying(rows: List[Dict[str, Any]], predicate) -> int:
    return sum(1 for r in rows if predicate(r))


def _mean_key(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [float(r.get(key, 0.0)) for r in rows]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _evaluate(arm_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_id = {a["arm_id"]: _arm_rows(arm_results, a["arm_id"]) for a in ARMS}
    control_arm = next(a for a in ARMS if a["is_control"])
    nonzero_arms = [a for a in ARMS if not a["is_control"]]
    control_rows = by_id[control_arm["arm_id"]]
    highest_gain_arm = max(ARMS, key=lambda a: a["authority_gain"])
    highest_gain_rows = by_id[highest_gain_arm["arm_id"]]

    # --- READINESS leg 1: channel live (curiosity bias non-zero range) ---
    bias_range_seeds_ok = _n_seeds_satisfying(
        highest_gain_rows,
        lambda r: float(r.get("curiosity_bias_range_mean", 0.0)) > BIAS_RANGE_FLOOR,
    )
    highest_gain_bias_range_mean = _mean_key(highest_gain_rows, "curiosity_bias_range_mean")

    # --- READINESS leg 2 (SAME statistic the Goldilocks routes on): H_pos RANGE
    # across the gain arms (healthy-seed-mean per arm) >= floor. ---
    per_arm_h_pos_mean = {
        a["arm_id"]: round(_mean_key(by_id[a["arm_id"]], "h_pos_mean"), 6) for a in ARMS
    }
    h_pos_arm_means = list(per_arm_h_pos_mean.values())
    h_pos_range_across_arms = (
        float(max(h_pos_arm_means) - min(h_pos_arm_means)) if h_pos_arm_means else 0.0
    )
    h_pos_range_ok = bool(h_pos_range_across_arms >= H_POS_RANGE_FLOOR)

    # --- READINESS leg 3: finite / explosion guard ---
    max_pairwise = max(
        [float(r.get("cand_world_pairwise_dist_max", 0.0)) for r in arm_results] or [0.0]
    )
    magnitude_ok = bool(math.isfinite(max_pairwise) and max_pairwise < MAGNITUDE_CEIL)

    readiness_ok = bool(
        bias_range_seeds_ok >= MIN_SEEDS_FOR_PASS
        and h_pos_range_ok
        and magnitude_ok
    )

    # --- Goldilocks: best non-zero gain arm by mean H_pos; per-seed lift over control ---
    control_by_seed = {int(r["seed"]): float(r.get("h_pos_mean", 0.0)) for r in control_rows}

    def _seeds_lift_over_control(rows: List[Dict[str, Any]]) -> int:
        n = 0
        for r in rows:
            s = int(r["seed"])
            ctrl = control_by_seed.get(s, None)
            if ctrl is None:
                continue
            if float(r.get("h_pos_mean", 0.0)) - ctrl >= H_POS_LIFT_MARGIN:
                n += 1
        return n

    nonzero_arm_lift = {
        a["arm_id"]: {
            "authority_gain": a["authority_gain"],
            "h_pos_mean": round(_mean_key(by_id[a["arm_id"]], "h_pos_mean"), 6),
            "seeds_lift_over_control": int(_seeds_lift_over_control(by_id[a["arm_id"]])),
        }
        for a in nonzero_arms
    }
    # Goldilocks = highest mean-H_pos non-zero arm.
    best_arm_id = max(
        nonzero_arm_lift.keys(),
        key=lambda aid: nonzero_arm_lift[aid]["h_pos_mean"],
    )
    best_gain = float(nonzero_arm_lift[best_arm_id]["authority_gain"])
    best_seeds_lift = int(nonzero_arm_lift[best_arm_id]["seeds_lift_over_control"])
    goldilocks_lift_ok = bool(best_seeds_lift >= MIN_SEEDS_FOR_PASS)

    # Inverted-U detection over the FULL gain axis (control + non-zero).
    ordered_ids = [a["arm_id"] for a in ARMS]
    ordered_h = [per_arm_h_pos_mean[aid] for aid in ordered_ids]
    peak_idx = ordered_h.index(max(ordered_h)) if ordered_h else 0
    nonmonotone = bool(0 < peak_idx < len(ordered_h) - 1)

    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        overall_pass = False
    elif goldilocks_lift_ok:
        label = "mech314a_goldilocks_identified"
        overall_pass = True
    else:
        label = "mech314a_no_exploratory_benefit"
        overall_pass = False

    return {
        "readiness": {
            "leg1_bias_range_floor": BIAS_RANGE_FLOOR,
            "highest_gain_arm": highest_gain_arm["arm_id"],
            "highest_gain_bias_range_mean": round(highest_gain_bias_range_mean, 8),
            "leg1_seeds_above_floor": int(bias_range_seeds_ok),
            "leg2_h_pos_range_floor": H_POS_RANGE_FLOOR,
            "h_pos_range_across_arms": round(h_pos_range_across_arms, 6),
            "leg2_h_pos_range_ok": h_pos_range_ok,
            "leg3_magnitude_ceil": MAGNITUDE_CEIL,
            "max_pairwise_dist_observed": round(max_pairwise, 6),
            "leg3_magnitude_ok": magnitude_ok,
            "min_seeds_required": MIN_SEEDS_FOR_PASS,
            "readiness_ok": readiness_ok,
        },
        "goldilocks": {
            "control_arm": control_arm["arm_id"],
            "control_h_pos_mean": round(_mean_key(control_rows, "h_pos_mean"), 6),
            "best_arm": best_arm_id,
            "best_gain": best_gain,
            "best_arm_h_pos_mean": round(nonzero_arm_lift[best_arm_id]["h_pos_mean"], 6),
            "best_arm_seeds_lift_over_control": best_seeds_lift,
            "h_pos_lift_margin": H_POS_LIFT_MARGIN,
            "goldilocks_lift_ok": goldilocks_lift_ok,
            "nonmonotone_inverted_u": nonmonotone,
            "per_arm_h_pos_mean": per_arm_h_pos_mean,
            "per_nonzero_arm_lift": nonzero_arm_lift,
        },
        "action_class_entropy_per_arm_mean": {
            aid: round(_mean_key(rows, "action_class_entropy"), 6)
            for aid, rows in by_id.items()
        },
        "curiosity_bias_range_per_arm_mean": {
            aid: round(_mean_key(rows, "curiosity_bias_range_mean"), 8)
            for aid, rows in by_id.items()
        },
        "label": label,
        "overall_pass": overall_pass,
        # Readiness preconditions (same-statistic discipline; can self-route not-ready).
        "preconditions": [
            {
                "name": "curiosity_bias_range_supra_floor",
                "kind": "readiness",
                "description": (
                    "At the highest-gain arm the per-candidate curiosity_bias_range "
                    "(pre-rescale cross-candidate RANGE) clears the floor -- the MECH-314a "
                    "channel carries a real per-candidate pattern for the authority to "
                    "amplify (the 643a 'scaling zero is still zero' guard)."
                ),
                "control": "highest modulatory_authority_gain arm, curiosity_candidate_source=e2_world_forward",
                "measured": round(highest_gain_bias_range_mean, 8),
                "threshold": BIAS_RANGE_FLOOR,
                "met": bool(bias_range_seeds_ok >= MIN_SEEDS_FOR_PASS),
            },
            {
                "name": "h_pos_range_across_gain_arms_supra_floor",
                "kind": "readiness",
                "description": (
                    "SAME-statistic gate (V3-EXQ-643 lesson): the RANGE across the swept "
                    "authority-gain arms of the healthy-seed-mean H_pos -- the EXACT "
                    "statistic the Goldilocks decision routes on -- clears a floor. If H_pos "
                    "is flat across every gain the authority knob produces no behavioural "
                    "gradient to calibrate -> substrate_not_ready_requeue, not a verdict."
                ),
                "control": "RANGE of per-arm mean H_pos across gain {0,0.25,0.5,1.0,2.0}",
                "measured": round(h_pos_range_across_arms, 6),
                "threshold": H_POS_RANGE_FLOOR,
                "met": h_pos_range_ok,
            },
            {
                "name": "rolled_out_zworld_magnitude_bounded",
                "kind": "readiness",
                "description": (
                    "Rolled-out z_world spread stayed finite and below the 643a explosion "
                    "ceiling (SD-056 online training numerical stability)."
                ),
                "control": "max cand_world_pairwise_dist across all arms",
                "measured": round(max_pairwise, 6),
                "threshold": MAGNITUDE_CEIL,
                "direction": "upper",
                "met": magnitude_ok,
            },
        ],
        "criteria": [
            {
                "name": "mech314a_goldilocks_lift_over_control",
                "load_bearing": True,
                "passed": goldilocks_lift_ok,
            },
        ],
        "criteria_non_degenerate": {
            "mech314a_goldilocks_lift_over_control": h_pos_range_ok,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    p0 = DRY_RUN_P0 if dry_run else P0_WARMUP_EPISODES
    p1 = DRY_RUN_P1 if dry_run else P1_MEASUREMENT_EPISODES
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE
    measure_after = DRY_RUN_MEASURE_AFTER_TICK if dry_run else MEASURE_AFTER_TICK

    arm_results: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm['arm_id']}", flush=True)
            cell = _run_seed_arm(arm, seed, p0, p1, steps, measure_after)
            cell["arm_fingerprint"] = compute_arm_fingerprint(
                config_slice={
                    "arm": {k: arm[k] for k in ("arm_id", "authority_gain", "is_control")},
                    "env_kwargs": ENV_KWARGS,
                    "sd056_weight": SD056_WEIGHT,
                    "curiosity_bias_scale": CURIOSITY_BIAS_SCALE,
                    "curiosity_weight": CURIOSITY_WEIGHT,
                    "visitation_buffer_len": VISITATION_BUFFER_LEN,
                    "curiosity_candidate_source": "e2_world_forward",
                    "use_modulatory_selection_authority": True,
                    "p0_episodes": p0, "p1_episodes": p1, "steps_per_episode": steps,
                },
                seed=seed,
                script_path=Path(__file__),
                rng_fully_reset=True,
                extra_ineligible_reasons=["online_e2_training_stateful_per_cell"],
            )
            arm_results.append(cell)
            passed = cell.get("error_note") is None
            print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)

    summary = _evaluate(arm_results)
    outcome = "PASS" if summary["overall_pass"] else "FAIL"

    if summary["label"] == "substrate_not_ready_requeue":
        evidence_direction = "non_contributory"
    elif summary["overall_pass"]:
        evidence_direction = "supports"
    else:
        evidence_direction = "does_not_support"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    manifest: Dict[str, Any] = {
        "schema_version": "v1",
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": outcome,
        "result": outcome,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": evidence_direction,
        "evidence_direction_note": (
            "MECH-314a per-candidate RBF novelty Goldilocks calibration on the "
            "V3-EXQ-648a-validated substrate (curiosity_candidate_source="
            "'e2_world_forward' + SD-056 online + rollout clamp + SP-CEM + V_s) with "
            "MECH-314a curiosity as the SOLE modulatory channel and the 643a-validated "
            "modulatory-selection-authority ON. The swept knob is modulatory_authority_gain "
            "(NOT curiosity_novelty_weight: the authority renormalizes the combined "
            "modulatory range to gain*raw_score_range, so the weight is washed out -- "
            "sweeping it reproduces the 590a degenerate byte-identical-arms null). DV = env "
            "pos_entropy exploration. PASS (supports MECH-314a) = a Goldilocks gain lifts "
            "H_pos over the gain=0 control by >= margin on >=2/3 seeds. Readiness-below-floor "
            "(curiosity bias range OR H_pos-range-across-gains below floor, the same statistic "
            "the Goldilocks routes on) self-routes substrate_not_ready_requeue, NOT a MECH-314a "
            "weakens. FIRST MECH-314a behavioural-evidence run (the 590/590a lineage tested "
            "MECH-111 broadcast novelty, a different mechanism -- no inherited claim_ids)."
        ),
        "interpretation": {
            "label": summary["label"],
            "preconditions": summary["preconditions"],
            "criteria": summary["criteria"],
            "criteria_non_degenerate": summary["criteria_non_degenerate"],
            "routing": {
                "substrate_not_ready_requeue": "re-queue as V3-EXQ-590c at higher P0 budget / check e2+authority wiring; do NOT weaken MECH-314a",
                "mech314a_goldilocks_identified": "PASS (supports MECH-314a); adopt best gain for the curiosity channel",
                "mech314a_no_exploratory_benefit": "FAIL (does_not_support MECH-314a behavioural reading); /failure-autopsy",
            },
        },
        "dry_run": bool(dry_run),
        "config": {
            "seeds": seeds,
            "p0_warmup_episodes": p0,
            "p1_measurement_episodes": p1,
            "steps_per_episode": steps,
            "measure_after_tick": measure_after,
            "swept_knob": "modulatory_authority_gain",
            "authority_gains": AUTHORITY_GAINS,
            "env_kwargs": ENV_KWARGS,
            "sd056_weight": SD056_WEIGHT,
            "curiosity_bias_scale": CURIOSITY_BIAS_SCALE,
            "curiosity_novelty_weight_fixed": CURIOSITY_WEIGHT,
            "visitation_buffer_len": VISITATION_BUFFER_LEN,
            "thresholds": {
                "bias_range_floor": BIAS_RANGE_FLOOR,
                "h_pos_range_floor": H_POS_RANGE_FLOOR,
                "magnitude_ceil": MAGNITUDE_CEIL,
                "h_pos_lift_margin": H_POS_LIFT_MARGIN,
                "min_seeds_for_pass": MIN_SEEDS_FOR_PASS,
            },
        },
        "acceptance_criteria": {
            "readiness_substrate_ready": summary["readiness"]["readiness_ok"],
            "mech314a_goldilocks_lift_over_control": summary["goldilocks"]["goldilocks_lift_ok"],
            "overall_pass": summary["overall_pass"],
        },
        "summary": summary,
        "arm_results": arm_results,
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"

    if not dry_run:
        out_path = write_flat_manifest(
            manifest,
            out_dir,
            dry_run=False,
            config=manifest.get("config"),
            seeds=SEEDS,
            script_path=Path(__file__),
        )
        print(f"Manifest written: {out_path}", flush=True)
    else:
        out_path = Path("/dev/null")
        print("Dry run -- manifest not written.", flush=True)

    print(f"Outcome: {outcome} (label={summary['label']})", flush=True)
    for k, v in manifest["acceptance_criteria"].items():
        print(f"  {k}: {v}", flush=True)
    print(
        f"  goldilocks: best_arm={summary['goldilocks']['best_arm']} "
        f"gain={summary['goldilocks']['best_gain']} "
        f"H_pos {summary['goldilocks']['best_arm_h_pos_mean']} "
        f"vs control {summary['goldilocks']['control_h_pos_mean']} "
        f"(nonmonotone={summary['goldilocks']['nonmonotone_inverted_u']})",
        flush=True,
    )

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-590b MECH-314a per-candidate RBF novelty Goldilocks calibration (authority-gain sweep on the 648a-validated substrate)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Short smoke run.")
    args = parser.parse_args()

    result = run_experiment(dry_run=args.dry_run)

    if args.dry_run:
        sys.exit(0)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    _outcome = _outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL"
    emit_outcome(
        outcome=_outcome,
        manifest_path=str(result.get("manifest_path", Path("/dev/null"))),
    )
    sys.exit(0)
