"""
V3-EXQ-953 -- MECH-135/INV-088 E1 forward-model horizon sweep + one-step
per-action divergence probe

Claims: MECH-135, INV-088
Supersedes: none (a diagnostic probe, not a re-test of V3-EXQ-108b's own criteria)
DIAGNOSTIC. See EXPERIMENT_PURPOSE below.

WHY THIS RUN EXISTS
---------------------------------------------------------------------------
V3-EXQ-108b found e1coe_score_var collapsed (~1e-13/1e-14 vs threshold 0.002)
across 40 distinct 30-step candidate action sequences from the same starting
state -- E1's rollout-based scoring assigns functionally identical scores no
matter which actions are taken. The commissioned lit-pull
(REE_assembly/evidence/literature/targeted_review_e1_forward_model_rollout_consistency/
SYNTHESIS.md, Section 0) found this collapse is explained by EITHER of two
distinct mechanisms, both present in the code:

  (a) horizon mismatch -- E1 is trained at horizon=1 (ree_core/predictors/
      e1_deep.py, experiments/v3_exq_108b_..._zworld_disambiguation.py:303)
      but used autoregressively at horizon=30, with no training signal
      covering steps 2-30 (compounding error).
  (b) action-blindness -- e1_deep.py's forward()/predict_long_horizon() take
      NO action parameter at all; the only path from an action to z_world is
      action -> agent.e2.predict_next_self -> z_self -> E1's prior_generator,
      and inside predict_long_horizon the z_self half of the LSTM input is
      ZEROED, so the entire action signal is squeezed through one
      world_dim-wide projection.

Every training-objective fix the lit-pull ranked (TD-MPC-style multi-step
latent consistency, DaD scheduled unrolling, Asadi direct sequence-
conditioning, PlaNet latent overshooting) targets (a) and presupposes (b) is
already solved. If (b) dominates, all five candidates fail and
substrate_queue.json's SD-e1-rollout-consistency-training implementation_hint
("Add a multi-step/rollout-consistency term to E1's training objective") is
misdirected -- the real first work item would be action-conditioning E1's
transition, not a multi-step objective on top of an unchanged interface.

THIS SCRIPT (SYNTHESIS.md Section 4's probe, verbatim)
---------------------------------------------------------------------------
Phases 0a/0b/1/2/3 are IDENTICAL to V3-EXQ-108b (same SD-070 sanctioned
z_world encoder warmup, same bespoke E1/E2 single-step training, same goal
template, same warmup state, same 40 random 30-step candidate action
sequences from that warmup state) -- this is "the existing trained E1", not
a new architecture and not a loaded checkpoint (this codebase does not
checkpoint agents across runs; re-running the identical training procedure
inside one script IS the reuse the review asked for -- no objective or
interface change).

1. Horizon sweep. Phase 4 rolls out each of the 40 candidate action
   sequences ONCE (not once per checkpoint) and captures the predicted
   z_world endpoint at every horizon in HORIZON_CHECKPOINTS = [1, 2, 3, 5,
   10, 20, 30] (clipped to <= --rollout-horizon, which always stays a
   checkpoint itself). CR_rollout(h) is the same offset-invariant
   contrast-ratio statistic V3-EXQ-108b / zworld_near_static_
   characterisation_2026-07-18 use: CR = spread / ||centroid||, over the 40
   candidates' z_world endpoints at that horizon.
2. Horizon-matched real-state reference. CR_real(h) is ALSO collected at
   every checkpoint (independent random-policy rollouts from a fresh reset,
   sensed after exactly h real env steps) rather than reusing a single
   30-step CR_real as the denominator for every horizon. This matters: real
   state diversity itself grows with h (a 1-step random walk necessarily
   looks less diverse than a 30-step one), so a horizon-INVARIANT
   denominator would conflate "E1 differentiates less at low h" with "there
   is inherently less to differentiate at low h" -- an artifact unrelated to
   action-blindness. cr_ratio(h) = CR_rollout(h) / CR_real(h) is the
   horizon-matched, offset-invariant readout the decision rule below routes
   on. At h = --rollout-horizon this ratio is exactly V3-EXQ-108b's own
   cr_ratio, so this run is a strict superset of that check.
3. One-step per-action divergence (NEW instrument -- the E1 analogue of the
   sibling E2 review's cand_world_pairwise_dist). From the SAME Phase-2
   warmup state (z_self_0, z_world_0), for EACH of the K grid-world actions
   (deterministic, not sampled -- every action tested exactly once), a
   SINGLE E1 forward call (horizon=1) is made and its predicted z_world is
   recorded. The contrast-ratio statistic over these K predictions
   (cr_action_h1) directly tests (b): if E1's transition is action-blind,
   cr_action_h1 collapses toward 0 regardless of training quality, BEFORE
   any multi-step compounding can occur -- a cleaner isolation of (b) than
   cr_ratio(1) above, which still depends on which first action each of the
   40 RANDOM candidate sequences happened to draw. Raw pairwise L2 distances
   between the K one-step predictions are also recorded (the review's
   literal ask), alongside the contrast-ratio summary.

DECISION RULE (SYNTHESIS.md Section 4.3)
  - Compounding error (a) predicts SMOOTH degradation with depth:
    cr_ratio(1) healthy, cr_ratio(rollout_horizon) collapsed.
  - Action-blindness (b) predicts the ratio is ALREADY near-floor at h=1,
    before any compounding could have occurred, AND the direct per-action
    probe (cr_action_h1) is near-zero too.
These are cleanly distinguishable. See LABEL GRID in run() for the exact
routing, which also carries a "mechanism unclear" / "not replicated" /
"cross-seed inconsistent" residual bucket for any reading the two clean
hypotheses do not cover -- those route to FAIL and a follow-up autopsy,
never to a forced ceiling/does-not-support verdict.

READINESS (P0, both existential, gate below-floor to substrate_not_ready_requeue)
  P_ENCODER_TRAINED: at least one split_encoder.world_encoder tensor moved
    during Phase 0a, every seed (inherited unchanged from V3-EXQ-108b).
  P_REAL_ZWORLD_NONDEGENERATE_ALL_HORIZONS: CR_real(h) is a finite, positive
    number backed by at least MIN_REAL_SAMPLES_PER_HORIZON samples, at EVERY
    checkpoint horizon, every seed. If this fails anywhere, sensing itself
    (or the real-sample collection) is broken this run at that horizon --
    self-route substrate_not_ready_requeue, never a mechanism verdict.

DECLARED NULL. A FAIL here (any label) does not reopen V3-EXQ-108/108a's
original C1/C2/C3 result -- that weakens finding stands regardless of which
mechanism this run implicates. This run's job is explaining WHY, for
governance and the INV-088 thread, and for correctly scoping
SD-e1-rollout-consistency-training's implementation_hint -- not
re-litigating whether the collapse is real.

Re-derive brake (Step 2.5b, this session): 0 autopsies count a
substrate_ceiling hit against MECH-135 or INV-088 at this granularity
(checked via the standard grep-count method in the queue-experiment skill)
-- brake does not fire. This is also explicitly NOT a re-derive-braked
lettered retest of 108b's own criteria: it is a diagnostic on the horizon /
action-conditioning axis the lit-pull surfaced, which 108b itself never
measured (108b scored only at h=30, with a single first action per
candidate sequence, never isolating h=1 or a per-action-only comparison).

GOV-REUSE-1 (Step 2.4): the decisive readouts (CR_rollout(h) for h < 30 on
the SAME 40 candidate sequences, and the one-step per-action divergence at
the SAME warmup state) are not recorded in any existing manifest --
V3-EXQ-108b's own manifest (v3_exq_108b_mech135_inv088_zworld_
disambiguation_20260802T121643Z_v3.json) records CR_rollout only at h=30 (the
full 30-step endpoint) and never isolates a single action's one-step
prediction from a full 30-step rollout's first action. Not recoverable from
existing data -> proceed to author (this script).

SLEEP DRIVER: not applicable -- no sleep phase entered in this run.

Z_GOAL: real (GoalState with a genuine collected template, unchanged from
V3-EXQ-108a/108b) -- recorded via z_goal_stream_stats at manifest-write
time. The goal-proximity score is bonus/cross-reference data only in this
script (see e1coe_score_var_by_h below); the decision rule above routes
entirely on CR_rollout(h)/CR_real(h) and the one-step action-divergence
probe, per SYNTHESIS.md Section 4.
"""

import itertools
import sys
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.goal import GoalConfig, GoalState
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest
from experiments._lib.zworld_p0_warmup import run_zworld_p0
from experiments._lib.capability_eval import RandomPolicy
from experiments._lib.zworld_encoder_guard import (
    latent_stack_snapshot,
    assert_world_encoder_trained,
)


EXPERIMENT_TYPE = "v3_exq_953_mech135_inv088_e1_horizon_sweep_action_divergence_probe"
CLAIM_IDS = ["MECH-135", "INV-088"]
EXPERIMENT_PURPOSE = "diagnostic"
SUPERSEDES = None

# agent_construction_before_seed lint: run_seed() builds the agent
# (_build_agent, inside Phase 0a's _run_zworld_p0_warmup call chain) before
# any torch.manual_seed call in this function's own flow. Inherited from
# V3-EXQ-108b's identical exemption (itself inherited from V3-EXQ-108a,
# itself from V3-EXQ-108 row 9, agent_seed_order_lint_backlog_triage.md):
# every scored quantity in this script is read off the LITERAL SAME agent
# object within a seed (never two independently-constructed agents), so
# unseeded weight init cannot confound any within-seed comparison this
# script makes (horizon-vs-horizon, action-vs-action). Same single-
# agent-per-seed shape as 108a/108b, unchanged.
AGENT_SEED_ORDER_EXEMPT = (
    "Every within-seed comparison (horizon-vs-horizon, action-vs-action) "
    "scored off the literal same agent object (inherited from V3-EXQ-108b's "
    "identical exemption, itself inherited from V3-EXQ-108a / V3-EXQ-108 "
    "row 9 triage, agent_seed_order_lint_backlog_triage.md)"
)

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
CR_REAL_FLOOR = 1e-4               # below this, real-state differentiation itself is ~0
                                    # (unchanged from V3-EXQ-108b)
CR_ROLLOUT_COLLAPSE_RATIO = 0.1    # CR_rollout(h)/CR_real(h) below this = collapsed at h
                                    # (unchanged from V3-EXQ-108b -- same statistic, now
                                    # evaluated at every checkpoint instead of only h=30)
C3_VAR_THRESHOLD = 0.002           # bonus/cross-reference only -- unchanged from V3-EXQ-108/108a/108b
ZWORLD_P0_EPISODES = 60            # SD-070 encoder warmup -- matches V3-EXQ-108b
N_REAL_SAMPLES = 40                # per-checkpoint target sample count for CR_real(h)
MIN_REAL_SAMPLES_PER_HORIZON = 10  # readiness floor: a checkpoint needs at least this many
                                    # surviving real samples (early env termination can drop some)
HORIZON_CHECKPOINTS_FULL = [1, 2, 3, 5, 10, 20, 30]


# ---------------------------------------------------------------------------
# Helpers (unchanged from V3-EXQ-108b unless noted)
# ---------------------------------------------------------------------------

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _env_kwargs() -> Dict[str, Any]:
    """Env config, unchanged from V3-EXQ-108b."""
    return dict(
        size=10, num_hazards=2, num_resources=4,
        hazard_harm=0.02, env_drift_interval=8, env_drift_prob=0.05,
        proximity_harm_scale=0.03, proximity_benefit_scale=0.04,
        proximity_approach_threshold=0.15, hazard_field_decay=0.5,
        resource_respawn_on_consume=True,
    )


def _build_agent(seed: int, world_dim: int, self_dim: int) -> Tuple[REEAgent, CausalGridWorldV2]:
    env = CausalGridWorldV2(seed=seed, **_env_kwargs())
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=0.9,
        alpha_self=0.3,
    )
    config.latent.unified_latent_mode = False
    agent = REEAgent(config)
    return agent, env


# ---------------------------------------------------------------------------
# Phase 0a: SD-070 sanctioned z_world encoder warmup (unchanged from 108b)
# ---------------------------------------------------------------------------

def _run_zworld_p0_warmup(
    agent: REEAgent, seed: int, zworld_p0_episodes: int, steps_per_episode: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    before = latent_stack_snapshot(agent)
    warmup_env = CausalGridWorldV2(seed=seed, **_env_kwargs())
    p0a_report = run_zworld_p0(
        agent, warmup_env, seed, zworld_p0_episodes, steps_per_episode,
        policy=RandomPolicy(seed), label="v3_exq_953 P0a (SD-070 z_world encoder)",
        dry_run=dry_run,
    )
    encoder_report = assert_world_encoder_trained(
        agent, before, p0=zworld_p0_episodes, strict=False,
        context="v3_exq_953_mech135_inv088_e1_horizon_sweep_action_divergence_probe",
        escape_hint="pass zworld_p0_episodes=0 for a deliberate frozen-encoder run",
    )
    return {**p0a_report, **encoder_report}


# ---------------------------------------------------------------------------
# Phase 0b: bespoke E1/E2 single-step training (unchanged from 108b)
# ---------------------------------------------------------------------------

def _train_agent(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    n_episodes: int,
    steps_per_episode: int,
) -> None:
    """Train agent with random policy (E1 + E2 only). Byte-identical to
    V3-EXQ-108b's _train_agent -- calls agent.sense() exactly ONCE per env
    step (StepHarness invariant #1). sense() runs under torch.no_grad()
    throughout, so Phase 0a's now-trained encoder is never further
    disturbed by this phase."""
    torch.manual_seed(seed + 2000)
    random.seed(seed + 2000)
    agent.train()

    opt_e1 = optim.Adam(agent.e1.parameters(), lr=1e-3)
    opt_e2 = optim.Adam(agent.e2.parameters(), lr=1e-3)

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_loss_e1 = 0.0
        ep_loss_e2 = 0.0
        n_steps = 0

        latent_prev: Optional[object] = None
        action_prev: Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent_curr = agent.sense(obs_body, obs_world)

            action_idx = random.randint(0, env.action_dim - 1)
            action_curr = _action_to_onehot(action_idx, env.action_dim, agent.device)

            if latent_prev is not None:
                opt_e1.zero_grad()
                total_prev = torch.cat([latent_prev.z_self, latent_prev.z_world], dim=-1)
                total_curr = torch.cat([latent_curr.z_self, latent_curr.z_world], dim=-1)
                e1_pred, _ = agent.e1(total_prev, horizon=1)
                e1_loss = F.mse_loss(e1_pred[:, 0, :], total_curr.detach())
                e1_loss.backward()
                opt_e1.step()
                ep_loss_e1 += e1_loss.item()

                opt_e2.zero_grad()
                z_self_pred = agent.e2.predict_next_self(latent_prev.z_self.detach(), action_prev)
                e2_loss = F.mse_loss(z_self_pred, latent_curr.z_self.detach())
                e2_loss.backward()
                opt_e2.step()
                ep_loss_e2 += e2_loss.item()
                n_steps += 1

            _, _, done, _, obs_dict = env.step(action_curr)

            latent_prev = latent_curr
            action_prev = action_curr

            if done:
                break

        if (ep + 1) % 20 == 0:
            print(
                f"  [Train] ep {ep+1}/{n_episodes} "
                f"e1_loss={ep_loss_e1/max(n_steps,1):.5f} "
                f"e2_loss={ep_loss_e2/max(n_steps,1):.5f}",
                flush=True,
            )

    agent.eval()
    print(f"  [Train] Done. {n_episodes} episodes.", flush=True)


# ---------------------------------------------------------------------------
# Phase 1: goal template (unchanged from 108b)
# ---------------------------------------------------------------------------

def _collect_goal_template(
    agent: REEAgent, env: CausalGridWorldV2, seed: int, max_steps: int,
) -> Tuple[torch.Tensor, str]:
    torch.manual_seed(seed)
    random.seed(seed)
    _, obs_dict = env.reset()
    agent.reset()

    for _ in range(max_steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)
        agent.clock.advance()
        action_idx = random.randint(0, env.action_dim - 1)
        action = _action_to_onehot(action_idx, env.action_dim, agent.device)
        _, _, done, info, obs_dict = env.step(action)
        if info.get("transition_type", "none") == "resource":
            print(
                f"  [Phase1] Resource contact, z_world_norm={latent.z_world.norm().item():.3f}",
                flush=True,
            )
            return latent.z_world.detach(), "resource_contact"
        if done:
            _, obs_dict = env.reset()
            agent.reset()

    print("  [Phase1] WARNING: no resource contact -- using fallback unit vector", flush=True)
    z_goal = torch.randn(1, agent.config.latent.world_dim)
    z_goal = F.normalize(z_goal, dim=-1)
    return z_goal, "fallback_unit_vector"


# ---------------------------------------------------------------------------
# Phase 2: warmup state (unchanged from 108b)
# ---------------------------------------------------------------------------

def _get_warmup_state(
    agent: REEAgent, env: CausalGridWorldV2, seed: int, n_warmup_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    torch.manual_seed(seed + 1000)
    random.seed(seed + 1000)
    _, obs_dict = env.reset()
    agent.reset()
    latent = None
    warmup_actions: List[int] = []

    for _ in range(n_warmup_steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)
        agent.clock.advance()
        action_idx = random.randint(0, env.action_dim - 1)
        warmup_actions.append(action_idx)
        action = _action_to_onehot(action_idx, env.action_dim, agent.device)
        _, _, done, _, obs_dict = env.step(action)
        if done:
            _, obs_dict = env.reset()
            agent.reset()
            latent = None
            warmup_actions = []

    if latent is None:
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)

    return latent.z_self.detach(), latent.z_world.detach(), warmup_actions


# ---------------------------------------------------------------------------
# Phase 3: generate candidate sequences (unchanged from 108b)
# ---------------------------------------------------------------------------

def _generate_candidate_sequences(
    n_sequences: int, horizon: int, n_actions: int, seed: int,
) -> List[List[int]]:
    torch.manual_seed(seed + 500)
    random.seed(seed + 500)
    seqs = []
    for _ in range(n_sequences):
        seq = [random.randint(0, n_actions - 1) for _ in range(horizon)]
        seqs.append(seq)
    return seqs


# ---------------------------------------------------------------------------
# Phase 4 (NEW): score sequences, capturing z_world endpoints at EVERY
# horizon checkpoint within a single rollout per sequence.
# ---------------------------------------------------------------------------

def _score_sequence_e1coe_multi_horizon(
    agent: REEAgent,
    z_self_start: torch.Tensor,
    z_world_start: torch.Tensor,
    action_sequence: List[int],
    goal_state: GoalState,
    self_dim: int,
    checkpoints: List[int],
) -> Dict[int, Tuple[float, torch.Tensor]]:
    """One rollout of len(action_sequence) steps; returns {h: (score, endpoint)}
    for every h in checkpoints, h <= len(action_sequence)."""
    device = agent.device
    n_actions = agent.config.e2.action_dim
    checkpoint_set = set(checkpoints)

    agent.e1.reset_hidden_state()

    z_self_curr = z_self_start.clone()
    z_world_curr = z_world_start.clone()
    out: Dict[int, Tuple[float, torch.Tensor]] = {}

    for step_idx, a_idx in enumerate(action_sequence, start=1):
        action = _action_to_onehot(a_idx, n_actions, device)
        total_curr = torch.cat([z_self_curr, z_world_curr], dim=-1)
        with torch.no_grad():
            e1_preds, _ = agent.e1(total_curr, horizon=1)
        z_world_next = e1_preds[0, 0, self_dim:].unsqueeze(0)
        with torch.no_grad():
            z_self_next = agent.e2.predict_next_self(z_self_curr, action)
        z_self_curr = z_self_next
        z_world_curr = z_world_next

        if step_idx in checkpoint_set:
            score = float(goal_state.goal_proximity(z_world_curr).item())
            out[step_idx] = (score, z_world_curr.detach().clone())

    return out


# ---------------------------------------------------------------------------
# Phase 4b (NEW): real z_world sample at EVERY horizon checkpoint
# ---------------------------------------------------------------------------

def _collect_real_zworld_sample_multi_horizon(
    agent: REEAgent, env: CausalGridWorldV2, seed: int, n_samples: int,
    checkpoints: List[int],
) -> Dict[int, List[torch.Tensor]]:
    """n_samples independent random-policy rollouts from reset. Each
    trajectory is sensed AFTER every step (so "z_world at horizon h" means
    the sensed state after exactly h real env steps from reset -- the real-
    data analogue of the rollout's imagined h-step endpoint). A trajectory
    that terminates (done) before reaching max(checkpoints) simply does not
    contribute to the later checkpoints; readiness below requires
    MIN_REAL_SAMPLES_PER_HORIZON survivors at every checkpoint.
    Uses its own seed offset (+3000), mirroring V3-EXQ-108b's Phase 4b, so
    it does not disturb the deterministic warmup-state RNG stream."""
    torch.manual_seed(seed + 3000)
    random.seed(seed + 3000)
    max_h = max(checkpoints)
    checkpoint_set = set(checkpoints)
    samples_by_h: Dict[int, List[torch.Tensor]] = {h: [] for h in checkpoints}

    for _ in range(n_samples):
        _, obs_dict = env.reset()
        agent.reset()
        for step_idx in range(1, max_h + 1):
            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            _, _, done, _, obs_dict = env.step(action)
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
            if step_idx in checkpoint_set:
                samples_by_h[step_idx].append(latent.z_world.detach())
            if done:
                break

    return samples_by_h


def _contrast_ratio(vectors: List[torch.Tensor]) -> Dict[str, float]:
    """CR = spread / ||centroid||, per zworld_near_static_characterisation_2026-07-18
    sec 2's offset-invariant statistic. spread = RMS deviation from centroid.
    Unchanged from V3-EXQ-108b."""
    stacked = torch.cat(vectors, dim=0)  # [N, dim]
    centroid = stacked.mean(dim=0, keepdim=True)  # [1, dim]
    centroid_norm = float(centroid.norm().item())
    deviations = stacked - centroid
    spread = float(torch.sqrt((deviations.pow(2).sum(dim=-1)).mean()).item())
    cr = (spread / centroid_norm) if centroid_norm > 1e-12 else float("nan")
    return {"spread": spread, "centroid_norm": centroid_norm, "contrast_ratio": cr, "n": len(vectors)}


# ---------------------------------------------------------------------------
# Phase 4c (NEW): one-step per-action divergence probe
# ---------------------------------------------------------------------------

def _one_step_action_divergence(
    agent: REEAgent,
    z_self_0: torch.Tensor,
    z_world_0: torch.Tensor,
    self_dim: int,
) -> Dict[str, Any]:
    """From the SAME Phase-2 warmup state, one deterministic single-step E1
    forward call per action (every action tested exactly once, agent.e1's
    hidden state reset before each so the K calls are independent). Returns
    the predicted z_world per action, the pairwise L2 distances (the
    review's literal ask), and the same contrast-ratio statistic used
    elsewhere in this script for a directly comparable readout.

    CRITICAL: E1's forward() takes NO action parameter (SYNTHESIS.md Section
    0) -- the only path by which an action can influence E1's prediction is
    action -> agent.e2.predict_next_self -> z_self -> E1's prior_generator.
    So this function MUST route each action through predict_next_self to get
    a per-action z_self BEFORE calling E1, exactly as the rollout scoring
    path (_score_sequence_e1coe_multi_horizon) does every step. Calling E1
    directly on the same (z_self_0, z_world_0) for every action -- skipping
    the E2 step -- would feed E1 an IDENTICAL input regardless of which
    action was chosen, which is vacuously guaranteed to collapse to zero
    divergence by construction and could never discriminate anything about
    E1's own action-sensitivity."""
    device = agent.device
    n_actions = agent.config.e2.action_dim

    predictions: List[torch.Tensor] = []
    for a_idx in range(n_actions):
        agent.e1.reset_hidden_state()
        action = _action_to_onehot(a_idx, n_actions, device)
        with torch.no_grad():
            z_self_after_action = agent.e2.predict_next_self(z_self_0, action)
            total_curr = torch.cat([z_self_after_action, z_world_0], dim=-1)
            e1_preds, _ = agent.e1(total_curr, horizon=1)
        z_world_next = e1_preds[0, 0, self_dim:].unsqueeze(0)
        predictions.append(z_world_next.detach())

    pairwise_dists = [
        float((predictions[i] - predictions[j]).norm().item())
        for i, j in itertools.combinations(range(len(predictions)), 2)
    ]
    cr = _contrast_ratio(predictions)

    return {
        "n_actions": n_actions,
        "pairwise_dists": pairwise_dists,
        "pairwise_dist_mean": float(sum(pairwise_dists) / len(pairwise_dists)) if pairwise_dists else 0.0,
        "pairwise_dist_min": float(min(pairwise_dists)) if pairwise_dists else 0.0,
        "pairwise_dist_max": float(max(pairwise_dists)) if pairwise_dists else 0.0,
        "contrast_ratio": cr,
    }


# ---------------------------------------------------------------------------
# Single-seed runner
# ---------------------------------------------------------------------------

def run_seed(
    seed: int,
    world_dim: int,
    self_dim: int,
    n_train_episodes: int,
    steps_per_episode: int,
    n_sequences: int,
    rollout_horizon: int,
    n_warmup_steps: int,
    goal_max_steps: int,
    zworld_p0_episodes: int,
    n_real_samples: int,
    checkpoints: List[int],
    dry_run: bool = False,
) -> Dict[str, Any]:
    print(f"\n[EXQ-953] seed={seed}", flush=True)
    print(f"Seed {seed} Condition single", flush=True)

    agent, env = _build_agent(seed, world_dim, self_dim)

    # Phase 0a: SD-070 sanctioned encoder warmup (unchanged from 108b)
    print(f"[EXQ-953] Phase 0a: SD-070 z_world encoder warmup ({zworld_p0_episodes} eps)...", flush=True)
    readiness_report = _run_zworld_p0_warmup(
        agent, seed, zworld_p0_episodes, steps_per_episode, dry_run=dry_run,
    )
    print(
        f"  encoder_trained={readiness_report.get('zworld_encoder_trained')} "
        f"max_abs_delta={readiness_report.get('world_encoder_max_abs_delta'):.6f}",
        flush=True,
    )

    # Phase 0b: bespoke E1/E2 single-step training (unchanged from 108b)
    print(f"[EXQ-953] Phase 0b: training E1/E2 ({n_train_episodes} eps)...", flush=True)
    _train_agent(agent, env, seed, n_train_episodes, steps_per_episode)

    # Phase 1: goal template (unchanged from 108b)
    print("[EXQ-953] Phase 1: goal template...", flush=True)
    z_goal_tensor, goal_template_source = _collect_goal_template(agent, env, seed, goal_max_steps)
    goal_config = GoalConfig(goal_dim=world_dim, z_goal_enabled=True, goal_weight=1.0)
    goal_state = GoalState(goal_config, agent.device)
    goal_state._z_goal = z_goal_tensor.to(agent.device)
    print(f"  z_goal_norm={goal_state.goal_norm():.4f} source={goal_template_source}", flush=True)

    # Phase 2: warmup state (unchanged from 108b)
    print("[EXQ-953] Phase 2: warmup state...", flush=True)
    z_self_0, z_world_0, warmup_actions = _get_warmup_state(agent, env, seed, n_warmup_steps)
    base_prox = float(goal_state.goal_proximity(z_world_0).item())
    print(f"  base_prox={base_prox:.4f}", flush=True)

    # Phase 3: candidate sequences (unchanged from 108b)
    print(f"[EXQ-953] Phase 3: generating {n_sequences} candidate sequences...", flush=True)
    seqs = _generate_candidate_sequences(n_sequences, rollout_horizon, env.action_dim, seed)

    # Phase 4: score sequences, capturing every horizon checkpoint
    print(f"[EXQ-953] Phase 4: scoring sequences at horizons {checkpoints}...", flush=True)
    scores_by_h: Dict[int, List[float]] = {h: [] for h in checkpoints}
    endpoints_by_h: Dict[int, List[torch.Tensor]] = {h: [] for h in checkpoints}

    for i, seq in enumerate(seqs):
        per_h = _score_sequence_e1coe_multi_horizon(
            agent, z_self_0, z_world_0, seq, goal_state, self_dim, checkpoints,
        )
        for h, (score, endpoint) in per_h.items():
            scores_by_h[h].append(score)
            endpoints_by_h[h].append(endpoint)

        if (i + 1) % 10 == 0:
            print(f"  scored {i+1}/{n_sequences}", flush=True)

    e1coe_score_var_by_h: Dict[int, float] = {}
    cr_rollout_by_h: Dict[int, Dict[str, float]] = {}
    for h in checkpoints:
        scores_t = torch.tensor(scores_by_h[h])
        e1coe_score_var_by_h[h] = float(scores_t.var().item()) if len(scores_by_h[h]) > 1 else 0.0
        cr_rollout_by_h[h] = _contrast_ratio(endpoints_by_h[h])
        print(
            f"  h={h:>2d}: e1coe_score_var={e1coe_score_var_by_h[h]:.6e} "
            f"CR_rollout={cr_rollout_by_h[h]['contrast_ratio']:.6e}",
            flush=True,
        )

    # Phase 4b: real z_world sample at every horizon checkpoint
    print(f"[EXQ-953] Phase 4b: sampling {n_real_samples} real trajectories at horizons {checkpoints}...", flush=True)
    real_samples_by_h = _collect_real_zworld_sample_multi_horizon(
        agent, env, seed, n_real_samples, checkpoints,
    )
    cr_real_by_h: Dict[int, Dict[str, float]] = {}
    cr_ratio_by_h: Dict[int, float] = {}
    for h in checkpoints:
        samples = real_samples_by_h[h]
        if len(samples) >= 2:
            cr_real_by_h[h] = _contrast_ratio(samples)
        else:
            cr_real_by_h[h] = {"spread": 0.0, "centroid_norm": 0.0, "contrast_ratio": float("nan"), "n": len(samples)}
        cr_real = cr_real_by_h[h]["contrast_ratio"]
        cr_roll = cr_rollout_by_h[h]["contrast_ratio"]
        cr_ratio_by_h[h] = (cr_roll / cr_real) if (cr_real == cr_real and cr_real > 0) else float("nan")
        print(
            f"  h={h:>2d}: CR_real={cr_real:.6e} (n={cr_real_by_h[h]['n']}) "
            f"ratio={cr_ratio_by_h[h]:.6e}",
            flush=True,
        )

    # Phase 4c: one-step per-action divergence probe
    print("[EXQ-953] Phase 4c: one-step per-action divergence probe...", flush=True)
    action_probe = _one_step_action_divergence(agent, z_self_0, z_world_0, self_dim)
    cr_real_h1 = cr_real_by_h.get(1, {}).get("contrast_ratio", float("nan"))
    action_cr = action_probe["contrast_ratio"]["contrast_ratio"]
    ratio_action_vs_real_h1 = (
        (action_cr / cr_real_h1) if (cr_real_h1 == cr_real_h1 and cr_real_h1 > 0) else float("nan")
    )
    print(
        f"  K={action_probe['n_actions']} pairwise_dist mean={action_probe['pairwise_dist_mean']:.6e} "
        f"min={action_probe['pairwise_dist_min']:.6e} max={action_probe['pairwise_dist_max']:.6e} "
        f"cr_action_h1={action_cr:.6e} ratio_vs_CR_real(h=1)={ratio_action_vs_real_h1:.6e}",
        flush=True,
    )

    verdict = "PASS" if readiness_report.get("zworld_encoder_trained") else "FAIL"
    print(f"verdict: {verdict}", flush=True)

    return {
        "seed": seed,
        "readiness": readiness_report,
        "goal_template_source": goal_template_source,
        "z_goal_norm": goal_state.goal_norm(),
        "base_prox": base_prox,
        "checkpoints": checkpoints,
        "e1coe_score_var_by_h": e1coe_score_var_by_h,
        "cr_rollout_by_h": cr_rollout_by_h,
        "cr_real_by_h": cr_real_by_h,
        "cr_ratio_by_h": cr_ratio_by_h,
        "action_probe": action_probe,
        "ratio_action_vs_real_h1": ratio_action_vs_real_h1,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(
    seeds: List[int],
    world_dim: int,
    self_dim: int,
    n_train_episodes: int,
    steps_per_episode: int,
    n_sequences: int,
    rollout_horizon: int,
    n_warmup_steps: int,
    goal_max_steps: int,
    zworld_p0_episodes: int,
    n_real_samples: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    checkpoints = sorted(set(
        [h for h in HORIZON_CHECKPOINTS_FULL if h <= rollout_horizon] + [rollout_horizon]
    ))

    seed_results = []
    for seed in seeds:
        r = run_seed(
            seed=seed,
            world_dim=world_dim,
            self_dim=self_dim,
            n_train_episodes=n_train_episodes,
            steps_per_episode=steps_per_episode,
            n_sequences=n_sequences,
            rollout_horizon=rollout_horizon,
            n_warmup_steps=n_warmup_steps,
            goal_max_steps=goal_max_steps,
            zworld_p0_episodes=zworld_p0_episodes,
            n_real_samples=n_real_samples,
            checkpoints=checkpoints,
            dry_run=dry_run,
        )
        seed_results.append(r)

    h1 = checkpoints[0]
    hmax = checkpoints[-1]

    # ---- Readiness (P0) ----
    encoder_trained_per_seed = [bool(r["readiness"].get("zworld_encoder_trained")) for r in seed_results]
    p_encoder_trained_met = all(encoder_trained_per_seed)
    min_encoder_delta = min(
        r["readiness"].get("world_encoder_max_abs_delta", 0.0) for r in seed_results
    )

    real_nondegenerate_per_h = {}
    for h in checkpoints:
        ok_this_h = all(
            (r["cr_real_by_h"][h]["contrast_ratio"] == r["cr_real_by_h"][h]["contrast_ratio"])
            and r["cr_real_by_h"][h]["contrast_ratio"] > 0
            and r["cr_real_by_h"][h]["n"] >= MIN_REAL_SAMPLES_PER_HORIZON
            for r in seed_results
        )
        real_nondegenerate_per_h[h] = ok_this_h
    p_real_nondegenerate_met = all(real_nondegenerate_per_h.values())
    min_real_samples_seen = min(
        r["cr_real_by_h"][h]["n"] for r in seed_results for h in checkpoints
    )

    preconditions = [
        {
            "name": "encoder_trained",
            "kind": "readiness",
            "description": (
                "At least one split_encoder.world_encoder tensor moved during the "
                "Phase 0a SD-070 warmup, per every seed -- unchanged precondition "
                "from V3-EXQ-108b."
            ),
            "measured": min_encoder_delta,
            "threshold": 0.0,
            "direction": "lower",
            "met": p_encoder_trained_met,
        },
        {
            "name": "real_zworld_nondegenerate_all_horizons",
            "kind": "readiness",
            "description": (
                "CR_real(h) is a finite, positive number backed by at least "
                f"{MIN_REAL_SAMPLES_PER_HORIZON} surviving real samples, at EVERY "
                "checkpoint horizon, every seed -- confirms sensing and the "
                "real-sample collection are not degenerate at any horizon before "
                "drawing a mechanism conclusion from the ratio."
            ),
            "measured": float(min_real_samples_seen),
            "threshold": float(MIN_REAL_SAMPLES_PER_HORIZON),
            "direction": "lower",
            "met": p_real_nondegenerate_met,
        },
    ]

    non_degenerate = bool(p_encoder_trained_met and p_real_nondegenerate_met)

    if not non_degenerate:
        label = "substrate_not_ready_requeue"
        degeneracy_reason = (
            "P0 readiness unmet: "
            + ("encoder_trained failed. " if not p_encoder_trained_met else "")
            + ("real_zworld_nondegenerate_all_horizons failed. " if not p_real_nondegenerate_met else "")
        )
        status = "FAIL"
        evidence_direction = "non_contributory"
        evidence_direction_per_claim = {"MECH-135": "non_contributory", "INV-088": "non_contributory"}
        per_seed_labels = [None for _ in seed_results]
    else:
        degeneracy_reason = None
        per_seed_labels = []
        for r in seed_results:
            ratio_h1 = r["cr_ratio_by_h"][h1]
            ratio_hmax = r["cr_ratio_by_h"][hmax]
            ratio_action = r["ratio_action_vs_real_h1"]
            action_cr = r["action_probe"]["contrast_ratio"]["contrast_ratio"]

            floored_at_h1 = (ratio_h1 == ratio_h1) and (ratio_h1 < CR_ROLLOUT_COLLAPSE_RATIO)
            collapsed_at_hmax = (ratio_hmax == ratio_hmax) and (ratio_hmax < CR_ROLLOUT_COLLAPSE_RATIO)
            action_divergence_near_zero = (
                ((ratio_action == ratio_action) and (ratio_action < CR_ROLLOUT_COLLAPSE_RATIO))
                or ((action_cr == action_cr) and (action_cr < CR_REAL_FLOOR))
            )

            if floored_at_h1 and action_divergence_near_zero:
                seed_label = "action_blindness_confirmed"
            elif floored_at_h1 and not action_divergence_near_zero:
                seed_label = "floored_at_h1_mechanism_unclear"
            elif (not floored_at_h1) and collapsed_at_hmax:
                seed_label = "smooth_compounding_collapse"
            else:
                seed_label = "no_collapse_replicated"
            per_seed_labels.append(seed_label)

        distinct_labels = set(per_seed_labels)
        if len(distinct_labels) == 1:
            label = next(iter(distinct_labels))
        else:
            label = "cross_seed_inconsistent"

        if label == "action_blindness_confirmed":
            status = "PASS"
            evidence_direction = "non_contributory"
            evidence_direction_per_claim = {"MECH-135": "non_contributory", "INV-088": "non_contributory"}
        elif label == "smooth_compounding_collapse":
            status = "PASS"
            evidence_direction = "non_contributory"
            evidence_direction_per_claim = {"MECH-135": "non_contributory", "INV-088": "non_contributory"}
        else:
            # floored_at_h1_mechanism_unclear / no_collapse_replicated / cross_seed_inconsistent
            status = "FAIL"
            evidence_direction = "non_contributory"
            evidence_direction_per_claim = {"MECH-135": "non_contributory", "INV-088": "non_contributory"}

    print(f"\n[EXQ-953] Label: {label}", flush=True)
    print(f"[EXQ-953] Status: {status}", flush=True)

    criteria = [
        {
            "name": "C_MECHANISM_DISCRIMINATED",
            "load_bearing": True,
            "passed": bool(non_degenerate and label in ("action_blindness_confirmed", "smooth_compounding_collapse")),
            "measured": min(r["cr_ratio_by_h"][h1] for r in seed_results) if seed_results and non_degenerate else float("nan"),
            "threshold": CR_ROLLOUT_COLLAPSE_RATIO,
            "statement": (
                "cr_ratio(h=1) and the one-step per-action divergence probe "
                "jointly discriminate action-blindness (already floored at h=1) "
                "from smooth horizon-compounding (floored only by h=rollout_horizon), "
                "per SYNTHESIS.md Section 4.3's decision rule."
            ),
        },
    ]
    criteria_non_degenerate = {"C_MECHANISM_DISCRIMINATED": non_degenerate}

    result: Dict[str, Any] = {
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes": SUPERSEDES,
        "evidence_class": "diagnostic_disambiguation",
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": evidence_direction_per_claim,
        "seeds": seeds,
        "world_dim": world_dim,
        "self_dim": self_dim,
        "n_train_episodes": n_train_episodes,
        "steps_per_episode": steps_per_episode,
        "n_sequences": n_sequences,
        "rollout_horizon": rollout_horizon,
        "horizon_checkpoints": checkpoints,
        "n_warmup_steps": n_warmup_steps,
        "zworld_p0_episodes": zworld_p0_episodes,
        "n_real_samples": n_real_samples,
        "registered_cr_real_floor": CR_REAL_FLOOR,
        "registered_cr_rollout_collapse_ratio": CR_ROLLOUT_COLLAPSE_RATIO,
        "registered_c3_var_threshold": C3_VAR_THRESHOLD,
        "min_real_samples_per_horizon_floor": MIN_REAL_SAMPLES_PER_HORIZON,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "per_seed_labels": per_seed_labels,
        "status": status,
        "outcome": status,
        "verdict": status,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria": criteria,
            "criteria_non_degenerate": criteria_non_degenerate,
        },
        "source_autopsy": "failure_autopsy_V3-EXQ-108a_2026-08-02",
        "source_lit_pull": "targeted_review_e1_forward_model_rollout_consistency",
        "hypothesis_space_qid": "inv088_evaluator_degeneracy_cause",
    }

    # Flatten per-seed metrics (dict-of-dicts kept as-is -- JSON-serialisable)
    for r in seed_results:
        s = r["seed"]
        for k, v in r.items():
            if k != "seed":
                result[f"seed_{s}_{k}"] = v

    return result


if __name__ == "__main__":
    import argparse
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(
        description=(
            "V3-EXQ-953: E1 forward-model horizon sweep + one-step per-action "
            "divergence probe (MECH-135/INV-088 diagnostic)"
        )
    )
    parser.add_argument("--seeds", type=str, default="42,123")
    parser.add_argument("--world-dim", type=int, default=32)
    parser.add_argument("--self-dim", type=int, default=32)
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--steps-per-episode", type=int, default=200)
    parser.add_argument("--rollout-horizon", type=int, default=30)
    parser.add_argument("--n-sequences", type=int, default=40)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--goal-max-steps", type=int, default=2000)
    parser.add_argument("--zworld-p0-episodes", type=int, default=ZWORLD_P0_EPISODES)
    parser.add_argument("--n-real-samples", type=int, default=N_REAL_SAMPLES)
    parser.add_argument("--dry-run", "--smoke-test", dest="dry_run", action="store_true")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    if args.dry_run:
        n_train = 2
        steps_ep = 50
        n_sequences = 5
        horizon = 10
        warmup = 5
        goal_max = 300
        zworld_p0 = 3
        n_real = 15  # > MIN_REAL_SAMPLES_PER_HORIZON so the smoke exercises the label branch too
        seeds = seeds[:1]
        print("[V3-EXQ-953] SMOKE TEST MODE", flush=True)
    else:
        n_train = args.train_episodes
        steps_ep = args.steps_per_episode
        n_sequences = args.n_sequences
        horizon = args.rollout_horizon
        warmup = args.warmup_steps
        goal_max = args.goal_max_steps
        zworld_p0 = args.zworld_p0_episodes
        n_real = args.n_real_samples

    t0 = time.perf_counter()
    result = run(
        seeds=seeds,
        world_dim=args.world_dim,
        self_dim=args.self_dim,
        n_train_episodes=n_train,
        steps_per_episode=steps_ep,
        n_sequences=n_sequences,
        rollout_horizon=horizon,
        n_warmup_steps=warmup,
        goal_max_steps=goal_max,
        zworld_p0_episodes=zworld_p0,
        n_real_samples=n_real,
        dry_run=args.dry_run,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["timestamp_utc"] = ts
    result["run_timestamp"] = ts
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    full_config = {
        "seeds": seeds,
        "world_dim": args.world_dim,
        "self_dim": args.self_dim,
        "n_train_episodes": n_train,
        "steps_per_episode": steps_ep,
        "n_sequences": n_sequences,
        "rollout_horizon": horizon,
        "n_warmup_steps": warmup,
        "goal_max_steps": goal_max,
        "zworld_p0_episodes": zworld_p0,
        "n_real_samples": n_real,
        "cr_real_floor": CR_REAL_FLOOR,
        "cr_rollout_collapse_ratio": CR_ROLLOUT_COLLAPSE_RATIO,
        "c3_var_threshold": C3_VAR_THRESHOLD,
        "min_real_samples_per_horizon_floor": MIN_REAL_SAMPLES_PER_HORIZON,
        "alpha_world": 0.9,
        "alpha_self": 0.3,
        "unified_latent_mode": False,
    }

    out_path = write_flat_manifest(
        result,
        dry_run=args.dry_run,
        config=full_config,
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t0,
    )

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    print(f"Label: {result['interpretation']['label']}", flush=True)

    if args.dry_run:
        print("[V3-EXQ-953] SMOKE TEST COMPLETE", flush=True)
        for k in ["status", "non_degenerate", "degeneracy_reason"]:
            print(f"  {k}: {result.get(k, 'N/A')}", flush=True)
        print(f"  label: {result['interpretation']['label']}", flush=True)

    # --- runner-conformance sentinel ---
    _outcome_raw = str(result.get("status", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
