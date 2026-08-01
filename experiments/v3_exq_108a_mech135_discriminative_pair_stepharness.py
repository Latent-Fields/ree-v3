"""
V3-EXQ-108a -- MECH-135 Governance-Grade Discriminative Pair (StepHarness re-run)

Claims: MECH-135
Supersedes: V3-EXQ-108

MECH-135 predicts: "During trajectory evaluation, E2 (cerebellar, z_self) must run in
parallel with E1 (cortical, z_world) so that z_world co-evolves during the planning
rollout; a frozen z_world causes E3 to evaluate goal achievement against a stale world
state."

WHY THIS RE-RUN EXISTS (multi_sense_audit_2026_05_08)
-------------------------------------------------------
claims.yaml's MECH-135 evidence_quality_note flags a training-time substrate confound in
V3-EXQ-108: its ``_train_agent()`` called ``agent.sense()`` TWICE per env step -- once for
``latent_prev`` before the action, once for ``latent_next`` after -- doubling the
substrate-tick rate during training relative to eval (which senses once per step). This is
exactly the failure mode ``experiments/_harness.py``'s ``StepHarness`` class exists to make
"structurally impossible" (its own module docstring names it verbatim: "double-sense pattern
that advanced GABA decay / AIC EMA / V_s readouts / anchor-set hysteresis at 2x the env rate
during warmup"). Confirmed present in 108's source at the time of this audit (sense() calls at
what are now lines 131 and 141 of v3_exq_108_mech135_discriminative_pair.py; the note's
original 129/139 references have drifted by a couple of lines from unrelated edits, same
construct). Both arms (FROZEN, E1_COE) were equally affected by the training-time confound
(training is arm-agnostic -- both conditions share the same trained E1/E2), so the C1/C2
comparison structure was preserved, but magnitude estimates -- and therefore the pass/fail
verdict on the real pre-registered thresholds -- may have been biased.

SEPARATE FINDING, surfaced while re-reading 108's evidence for this re-run (recorded here for
audit trail; not itself fixed by this script -- claims.yaml is governance-only-edit territory):
the "PASS ... second support entry" claims.yaml counts for MECH-135 (run
v3_exq_108_mech135_discriminative_pair_20260328T125004Z) is verifiably a --smoke-test run
(n_train_episodes=2, thresholds relaxed to c1=-99.0/c3=0.0 -- matches the script's hardcoded
smoke-test override branch exactly), i.e. a vacuous test-mode pass, not real evidence. Its own
run_id even carries the string ``evidence_direction_note`` from the 2026-05-08 audit. Meanwhile
the SAME DAY's full-production-scale run (100 episodes, 200 steps, 40 sequences, horizon 30,
real thresholds, seeds [42,123]; run
v3_exq_108_mech135_discriminative_pair_20260328T195341Z) actually FAILED under the real
protocol (c1_val=0.00229 << 0.05 threshold; c3 variance ~1.4e-14 << 0.002 threshold) -- but
claims.yaml describes this run as having "empty metrics -- runner artifact" and dismisses it,
which does not match the manifest's actual (complete, non-empty) content. This re-run uses the
PRODUCTION scale (matching the mischaracterized-but-real FAIL run), not the smoke-test scale.

PORTING APPROACH -- what changed and what did not
--------------------------------------------------
Only Phase 0 (``_train_agent``) carried the confound. Phases 1-6 (goal template collection,
warmup-state collection, candidate-sequence generation, FROZEN/E1_COE rollout scoring, real-env
execution of the best sequence) already call ``agent.sense()`` exactly once per env step and
are UNCHANGED from V3-EXQ-108 -- byte-identical logic, because they replay
externally-specified/deterministic action sequences that StepHarness's own on-policy
``select_action`` cannot reproduce.

``_train_agent`` is restructured to call ``agent.sense()`` exactly ONCE per env step, matching
eval's cadence, by retaining ``(latent, action)`` from step N and pairing it with the sense()
result at step N+1 to form the ``(s_t, a_t, s_{t+1})`` triple the E1/E2 losses need. This is the
identical accumulation pattern StepHarness's own module docstring recommends for
warmup-with-aux-losses callers: "compute and apply optimiser steps AFTER harness.step()
returns, using result.latent as the encoder output this tick. The next harness.step() call will
encode the next observation through the post-update encoder... The previously-broken pattern of
doing a second sense() mid-tick is unnecessary and harmful." The bespoke E1 single-step MSE loss
(``concat(z_self, z_world) -> agent.e1(..., horizon=1)`` target) and E2 single-step MSE loss
(``agent.e2.predict_next_self``) are UNCHANGED from V3-EXQ-108 -- not switched to
``agent.compute_prediction_loss()``/``record_transition`` (the loss formulation other
StepHarness scripts such as V3-EXQ-514g use), since that is a materially different loss
(multi-step sampled-window MSE vs single-step MSE) that would change more than the audited
confound. The original uniform-random training-exploration policy is also UNCHANGED.

NOT PORTED ONTO THE LITERAL ``StepHarness`` CLASS -- and why. ``StepHarness.step()`` always
selects its action via ``agent.select_action(candidates, ticks, ...)`` (E3-driven, on-policy),
which would replace 108's uniform-random training exploration -- a bigger design change than
the audited confound calls for. More importantly, ``StepHarness.step()``'s internal
``agent._e1_tick(latent)`` (fired on the multi-rate clock schedule inside
``generate_trajectories``) calls into E1's forward pass and mutates
``agent.e1._hidden_state`` (``ree_core/predictors/e1_deep.py`` -- E1 is a stateful LSTM;
``reset_hidden_state()`` / persistent ``_hidden_state`` confirmed by direct read). This script's
E1 loss ALSO calls ``agent.e1(...)`` directly every step and would contend for that same hidden
state -- reintroducing an uncontrolled multi-rate substrate-advancement hazard, just relocated
from sense() to E1's recurrent state, i.e. a NEW instance of the exact failure class this re-run
exists to eliminate. ``agent.compute_prediction_loss()`` (the loss V3-EXQ-514g's
StepHarness-driven training uses) avoids this by explicitly saving/restoring
``e1._hidden_state`` around its own sampled-window computation
(``ree_core/agent.py:9081-9082,9118``); V3-EXQ-108's bespoke single-step losses carry no such
guard and were never designed to run alongside a live ``_e1_tick``. So this script applies
StepHarness's documented INVARIANT (exactly one ``sense()`` per env step) via a minimal,
targeted restructuring of 108's own loop, without instantiating the ``StepHarness`` class for
training. ``agent.clock.advance()`` is intentionally still not called during training (108's
original training loop never called it either) -- that is a separate, non-audited discrepancy
from eval's cadence, left untouched to keep this a narrow, auditable fix rather than a broader
redesign.

Design (Phases 1-6, unchanged from V3-EXQ-108; improves on EXQ-104b diagnostic):
  Given 40 candidate action sequences from the same starting state:
  - FROZEN condition: z_world frozen at t=0. All 40 sequences score identically (the
    initial proximity to goal). Plan selection is therefore random (no information).
  - E1_COE condition: z_world updated step-by-step via dynamic E1 prediction, reset per
    sequence. Scores vary across sequences. The best-scoring sequence is selected.

  Both best-selected sequences are then executed in the REAL ENVIRONMENT from the same
  starting state. Final goal proximity and resource contact are recorded.

  MECH-135 predicts: E1_COE best-selection will achieve higher real goal proximity than
  FROZEN best-selection (which is just random selection), because E1 co-evolution lets the
  planner identify action sequences that approach the goal.

Pass criteria (pre-registered, unchanged from V3-EXQ-108):
  C1: mean(real_prox_e1coe - real_prox_frozen) >= 0.05  [across 2 seeds]
  C2: e1coe contact >= frozen contact in >= 1 of 2 seeds  [directional]
  C3: e1coe score variance >= 0.002 in each seed  [selection is non-trivial]

Outcome scoring:
  PASS -> evidence_direction: "supports"  (retain_ree)
  FAIL -> evidence_direction: "weakens"   (hybridize / retire_ree_claim consideration)
"""

import sys
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from experiments._metrics import check_degeneracy


EXPERIMENT_TYPE = "v3_exq_108a_mech135_discriminative_pair_stepharness"
CLAIM_IDS = ["MECH-135"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES = "V3-EXQ-108"

# agent_construction_before_seed lint: run_seed() builds the agent before the
# torch.manual_seed(seed+2000) call inside _train_agent(). Inherited unchanged
# from V3-EXQ-108, whose identical shape was triaged IMMATERIAL for the same
# reason: FROZEN and E1_COE are both scored off the LITERAL SAME agent object
# within a seed (never two independently-constructed agents), so unseeded
# weight init cannot confound the within-seed arm comparison this experiment
# tests. See REE_assembly/evidence/planning/agent_seed_order_lint_backlog_triage.md
# row 9 (v3_exq_108_mech135_discriminative_pair.py).
AGENT_SEED_ORDER_EXEMPT = (
    "FROZEN/E1_COE scored off the literal same agent object within a seed "
    "(inherited from V3-EXQ-108 row 9 triage, agent_seed_order_lint_backlog_triage.md)"
)

# ---------------------------------------------------------------------------
# Pre-registered thresholds (unchanged from V3-EXQ-108)
# ---------------------------------------------------------------------------
C1_PROX_DELTA_THRESHOLD = 0.05   # mean(real_prox_e1coe - real_prox_frozen) >= this
C2_SEEDS_NEEDED = 1              # e1coe contact >= frozen contact in >= N seeds
C3_VAR_THRESHOLD = 0.002         # e1coe score variance >= this per seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _build_agent(seed: int, world_dim: int, self_dim: int) -> Tuple[REEAgent, CausalGridWorldV2]:
    env = CausalGridWorldV2(
        seed=seed, size=10, num_hazards=2, num_resources=4,
        hazard_harm=0.02, env_drift_interval=8, env_drift_prob=0.05,
        proximity_harm_scale=0.03, proximity_benefit_scale=0.04,
        proximity_approach_threshold=0.15, hazard_field_decay=0.5,
        resource_respawn_on_consume=True,
    )
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
# Phase 0: train agent -- StepHarness-invariant fix (single sense()/step)
# ---------------------------------------------------------------------------

def _train_agent(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    n_episodes: int,
    steps_per_episode: int,
) -> None:
    """Train agent with random policy (E1 + E2 only).

    Calls agent.sense() exactly ONCE per env step (StepHarness's invariant #1),
    matching the eval-phase cadence used elsewhere in this script. See the
    module docstring "PORTING APPROACH" section for the full rationale,
    including why the literal StepHarness class is not instantiated here.

    Retains (latent, action) from step N and pairs it with the sense() result
    at step N+1 to form the (s_t, a_t, s_{t+1}) triple the E1/E2 losses need
    -- the same one-step-behind accumulation StepHarness's own module
    docstring recommends for warmup-with-aux-losses callers.
    """
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
                latent_curr = agent.sense(obs_body, obs_world)  # exactly one sense() this step

            action_idx = random.randint(0, env.action_dim - 1)
            action_curr = _action_to_onehot(action_idx, env.action_dim, agent.device)

            if latent_prev is not None:
                # E1 update (world prediction error): predict (z_self, z_world) at
                # this tick from (z_self, z_world) at the previous tick.
                opt_e1.zero_grad()
                total_prev = torch.cat([latent_prev.z_self, latent_prev.z_world], dim=-1)
                total_curr = torch.cat([latent_curr.z_self, latent_curr.z_world], dim=-1)
                e1_pred, _ = agent.e1(total_prev, horizon=1)
                e1_loss = F.mse_loss(e1_pred[:, 0, :], total_curr.detach())
                e1_loss.backward()
                opt_e1.step()
                ep_loss_e1 += e1_loss.item()

                # E2 update (motor-sensory prediction error on z_self): predict
                # z_self at this tick from (z_self at t-1, action taken at t-1).
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
# Phase 1: collect goal template (unchanged from V3-EXQ-108 -- already
# single-sense-per-step)
# ---------------------------------------------------------------------------

def _collect_goal_template(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    max_steps: int,
) -> Tuple[torch.Tensor, str]:
    """Return (z_goal_tensor, source) where source is 'resource_contact' or 'fallback_unit_vector'."""
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
# Phase 2: warmup state (unchanged from V3-EXQ-108)
# ---------------------------------------------------------------------------

def _get_warmup_state(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    n_warmup_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Walk agent for n_warmup_steps and return (z_self, z_world, warmup_actions).
    warmup_actions is stored so that _execute_in_real_env can reproduce the same
    starting state deterministically.
    """
    torch.manual_seed(seed + 1000)
    random.seed(seed + 1000)
    _, obs_dict = env.reset()
    agent.reset()
    latent = None
    warmup_actions = []

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
# Phase 3: generate candidate sequences (unchanged from V3-EXQ-108)
# ---------------------------------------------------------------------------

def _generate_candidate_sequences(
    n_sequences: int,
    horizon: int,
    n_actions: int,
    seed: int,
) -> List[List[int]]:
    """Generate n_sequences random action index lists of length horizon."""
    torch.manual_seed(seed + 500)
    random.seed(seed + 500)
    seqs = []
    for _ in range(n_sequences):
        seq = [random.randint(0, n_actions - 1) for _ in range(horizon)]
        seqs.append(seq)
    return seqs


# ---------------------------------------------------------------------------
# Phase 4: score sequences (unchanged from V3-EXQ-108)
# ---------------------------------------------------------------------------

def _score_sequence_frozen(
    z_world_start: torch.Tensor,
    goal_state: GoalState,
) -> float:
    """
    Score under FROZEN condition: z_world never changes, so all sequences score
    identically = initial proximity. Plan selection is therefore random (no info).
    """
    return float(goal_state.goal_proximity(z_world_start).item())


def _score_sequence_e1coe(
    agent: REEAgent,
    z_self_start: torch.Tensor,
    z_world_start: torch.Tensor,
    action_sequence: List[int],
    goal_state: GoalState,
    self_dim: int,
) -> float:
    """
    Score under E1_COE condition: z_world updated step-by-step via dynamic E1 prediction.

    E1 hidden state is reset before each sequence so scores are independent (each
    sequence represents a fresh imagined rollout from the same starting state).
    """
    device = agent.device
    n_actions = agent.config.e2.action_dim

    # Reset E1 hidden state: fresh rollout per sequence
    agent.e1.reset_hidden_state()

    z_self_curr = z_self_start.clone()
    z_world_curr = z_world_start.clone()

    for a_idx in action_sequence:
        action = _action_to_onehot(a_idx, n_actions, device)
        total_curr = torch.cat([z_self_curr, z_world_curr], dim=-1)
        with torch.no_grad():
            e1_preds, _ = agent.e1(total_curr, horizon=1)
        # Extract world component: preds shape [1, 1, total_dim]
        z_world_next = e1_preds[0, 0, self_dim:].unsqueeze(0)
        with torch.no_grad():
            z_self_next = agent.e2.predict_next_self(z_self_curr, action)
        z_self_curr = z_self_next
        z_world_curr = z_world_next

    return float(goal_state.goal_proximity(z_world_curr).item())


# ---------------------------------------------------------------------------
# Phase 5: execute best sequence in real env (unchanged from V3-EXQ-108)
# ---------------------------------------------------------------------------

def _execute_in_real_env(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    warmup_actions: List[int],
    action_sequence: List[int],
    goal_state: GoalState,
) -> Tuple[float, bool]:
    """
    Reproduce the warmup state using the stored warmup_actions, then execute
    action_sequence in the real environment. Returns (final_prox, resource_contacted).

    Using stored warmup_actions (rather than re-seeding and re-running random)
    guarantees bit-identical reproduction of the starting state regardless of any
    global random state changes during scoring.
    """
    device = agent.device

    # Reproduce warmup
    torch.manual_seed(seed + 1000)
    random.seed(seed + 1000)
    _, obs_dict = env.reset()
    agent.reset()
    latent = None

    for a_idx in warmup_actions:
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)
        agent.clock.advance()
        action = _action_to_onehot(a_idx, env.action_dim, device)
        _, _, done, _, obs_dict = env.step(action)
        if done:
            break

    # Execute candidate sequence in real env
    resource_contacted = False
    for a_idx in action_sequence:
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        with torch.no_grad():
            latent = agent.sense(obs_body, obs_world)
        agent.clock.advance()
        action = _action_to_onehot(a_idx, env.action_dim, device)
        _, _, done, info, obs_dict = env.step(action)
        if info.get("transition_type", "none") == "resource":
            resource_contacted = True
        if done:
            break

    # Final obs
    obs_body = obs_dict["body_state"]
    obs_world = obs_dict["world_state"]
    with torch.no_grad():
        latent_final = agent.sense(obs_body, obs_world)

    final_prox = float(goal_state.goal_proximity(latent_final.z_world).item())
    return final_prox, resource_contacted


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
    c1_threshold: float,
    c3_var_threshold: float,
) -> Dict:
    print(f"\n[EXQ-108a] seed={seed}", flush=True)
    print(f"Seed {seed} Condition single", flush=True)

    agent, env = _build_agent(seed, world_dim, self_dim)

    # Phase 0: train (StepHarness-invariant: single sense()/step)
    print(f"[EXQ-108a] Phase 0: training ({n_train_episodes} eps)...", flush=True)
    _train_agent(agent, env, seed, n_train_episodes, steps_per_episode)

    # Phase 1: goal template
    print("[EXQ-108a] Phase 1: goal template...", flush=True)
    z_goal_tensor, goal_template_source = _collect_goal_template(agent, env, seed, goal_max_steps)
    goal_config = GoalConfig(goal_dim=world_dim, z_goal_enabled=True, goal_weight=1.0)
    goal_state = GoalState(goal_config, agent.device)
    goal_state._z_goal = z_goal_tensor.to(agent.device)
    print(f"  z_goal_norm={goal_state.goal_norm():.4f} source={goal_template_source}", flush=True)

    # Phase 2: warmup state
    print("[EXQ-108a] Phase 2: warmup state...", flush=True)
    z_self_0, z_world_0, warmup_actions = _get_warmup_state(agent, env, seed, n_warmup_steps)
    base_prox = float(goal_state.goal_proximity(z_world_0).item())
    print(f"  base_prox={base_prox:.4f}", flush=True)

    # Phase 3: candidate sequences
    print(f"[EXQ-108a] Phase 3: generating {n_sequences} candidate sequences...", flush=True)
    seqs = _generate_candidate_sequences(n_sequences, rollout_horizon, env.action_dim, seed)

    # Phase 4: score under FROZEN and E1_COE
    print("[EXQ-108a] Phase 4: scoring sequences...", flush=True)
    frozen_scores = []
    e1coe_scores = []

    for i, seq in enumerate(seqs):
        # FROZEN: all sequences score identically = base_prox
        frozen_scores.append(_score_sequence_frozen(z_world_0, goal_state))

        # E1_COE: dynamic E1 co-evolution, hidden state reset per sequence
        e1coe_score = _score_sequence_e1coe(agent, z_self_0, z_world_0, seq, goal_state, self_dim)
        e1coe_scores.append(e1coe_score)

        if (i + 1) % 10 == 0:
            print(f"  scored {i+1}/{n_sequences}", flush=True)

    e1coe_scores_t = torch.tensor(e1coe_scores)
    e1coe_score_var = float(e1coe_scores_t.var().item()) if len(e1coe_scores) > 1 else 0.0
    e1coe_score_min = float(e1coe_scores_t.min().item())
    e1coe_score_max = float(e1coe_scores_t.max().item())
    e1coe_score_mean = float(e1coe_scores_t.mean().item())

    print(
        f"  E1_COE scores: min={e1coe_score_min:.4f} max={e1coe_score_max:.4f} "
        f"mean={e1coe_score_mean:.4f} var={e1coe_score_var:.6f}",
        flush=True,
    )
    print(f"  FROZEN scores: all={frozen_scores[0]:.4f} (constant)", flush=True)

    # Phase 5: select best sequences
    frozen_best_idx = int(torch.tensor(frozen_scores).argmax().item())
    e1coe_best_idx = int(e1coe_scores_t.argmax().item())
    print(
        f"  FROZEN best_idx={frozen_best_idx} (random) "
        f"E1_COE best_idx={e1coe_best_idx} score={e1coe_scores[e1coe_best_idx]:.4f}",
        flush=True,
    )

    # Phase 6: execute best sequences in real env
    print("[EXQ-108a] Phase 6: real-env execution...", flush=True)
    agent.eval()
    real_prox_frozen, resource_frozen = _execute_in_real_env(
        agent, env, seed, warmup_actions, seqs[frozen_best_idx], goal_state
    )
    real_prox_e1coe, resource_e1coe = _execute_in_real_env(
        agent, env, seed, warmup_actions, seqs[e1coe_best_idx], goal_state
    )

    prox_delta = real_prox_e1coe - real_prox_frozen
    print(
        f"  FROZEN: real_prox={real_prox_frozen:.4f} contact={resource_frozen}",
        flush=True,
    )
    print(
        f"  E1_COE: real_prox={real_prox_e1coe:.4f} contact={resource_e1coe} "
        f"delta={prox_delta:+.4f}",
        flush=True,
    )
    verdict = "PASS" if prox_delta >= c1_threshold and e1coe_score_var >= c3_var_threshold else "FAIL"
    print(f"verdict: {verdict}", flush=True)

    return {
        "seed": seed,
        "goal_template_source": goal_template_source,
        "z_goal_norm": goal_state.goal_norm(),
        "base_prox": base_prox,
        "e1coe_score_min": e1coe_score_min,
        "e1coe_score_max": e1coe_score_max,
        "e1coe_score_mean": e1coe_score_mean,
        "e1coe_score_var": e1coe_score_var,
        "e1coe_best_score": float(e1coe_scores[e1coe_best_idx]),
        "frozen_base_score": frozen_scores[0],
        "real_prox_frozen": real_prox_frozen,
        "real_prox_e1coe": real_prox_e1coe,
        "prox_delta": prox_delta,
        "resource_frozen": resource_frozen,
        "resource_e1coe": resource_e1coe,
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
    c1_threshold: float,
    c3_var_threshold: float,
) -> Dict:
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
            c1_threshold=c1_threshold,
            c3_var_threshold=c3_var_threshold,
        )
        seed_results.append(r)

    # Aggregate
    prox_deltas = [r["prox_delta"] for r in seed_results]
    c1_val = sum(prox_deltas) / len(prox_deltas)
    c1_pass = c1_val >= c1_threshold

    c2_per_seed = [int(r["resource_e1coe"]) >= int(r["resource_frozen"]) for r in seed_results]
    c2_seeds_passing = sum(c2_per_seed)
    c2_pass = c2_seeds_passing >= C2_SEEDS_NEEDED

    c3_per_seed = [r["e1coe_score_var"] >= c3_var_threshold for r in seed_results]
    c3_pass = all(c3_per_seed)

    print(f"\n[EXQ-108a] Criteria:", flush=True)
    print(f"  C1 (mean_delta>={c1_threshold:.3f}): {c1_pass} val={c1_val:+.4f}", flush=True)
    print(f"  C2 (contact directional, >= {C2_SEEDS_NEEDED} seeds): {c2_pass} ({c2_seeds_passing}/{len(seeds)} seeds)", flush=True)
    print(f"  C3 (e1coe_var>={c3_var_threshold:.4f}): {c3_pass} {[r['e1coe_score_var'] for r in seed_results]}", flush=True)

    criteria_met = sum([c1_pass, c2_pass, c3_pass])
    status = "PASS" if (c1_pass and c2_pass and c3_pass) else "FAIL"
    evidence_direction = "supports" if status == "PASS" else "weakens"

    print(f"[EXQ-108a] Status: {status}", flush=True)

    result = {
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes": SUPERSEDES,
        "evidence_class": "discriminative_pair",
        "evidence_direction": evidence_direction,
        "seeds": seeds,
        "world_dim": world_dim,
        "self_dim": self_dim,
        "n_train_episodes": n_train_episodes,
        "steps_per_episode": steps_per_episode,
        "n_sequences": n_sequences,
        "rollout_horizon": rollout_horizon,
        "n_warmup_steps": n_warmup_steps,
        # Pre-registered thresholds (recorded for governance)
        "registered_c1_threshold": c1_threshold,
        "registered_c3_var_threshold": c3_var_threshold,
        "registered_c2_seeds_needed": C2_SEEDS_NEEDED,
        # Per-seed metrics
        "c1_mean_prox_delta": c1_val,
        "c1_pass": bool(c1_pass),
        "c2_pass": bool(c2_pass),
        "c2_seeds_passing": c2_seeds_passing,
        "c3_pass": bool(c3_pass),
        "c3_per_seed_vars": [r["e1coe_score_var"] for r in seed_results],
        "criteria_met": criteria_met,
        "criteria_total": 3,
        "status": status,
        "outcome": status,
        "verdict": status,
        "stepharness_invariant_note": (
            "training loop restructured to call agent.sense() exactly once per env "
            "step (StepHarness invariant #1), matching eval's cadence; see module "
            "docstring PORTING APPROACH section. Bespoke E1/E2 single-step losses "
            "and uniform-random training policy unchanged from V3-EXQ-108."
        ),
    }

    # Flatten per-seed metrics
    for r in seed_results:
        s = r["seed"]
        for k, v in r.items():
            if k != "seed":
                result[f"seed_{s}_{k}"] = v

    # Non-degeneracy self-report: the two load-bearing discriminative
    # readouts are e1coe_score_var (C3 -- is rollout scoring non-trivial at
    # all?) and prox_delta (C1 -- does the real-env outcome actually differ
    # between arms?). A run where either is pinned near-zero across every
    # seed is the exact failure mode the 2026-03-28 19:53:41Z full-scale
    # V3-EXQ-108 run exhibited (c3 variance ~1.4e-14) -- flag it automatically
    # rather than relying on a manual failure-autopsy to notice.
    result.update(check_degeneracy({
        "e1coe_score_var": [r["e1coe_score_var"] for r in seed_results],
        "prox_delta": prox_deltas,
    }))

    return result


if __name__ == "__main__":
    import argparse
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(
        description="V3-EXQ-108a: MECH-135 discriminative pair, StepHarness-invariant re-run"
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
    parser.add_argument("--c1-threshold", type=float, default=C1_PROX_DELTA_THRESHOLD)
    parser.add_argument("--c3-var-threshold", type=float, default=C3_VAR_THRESHOLD)
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
        c1_thresh = -99.0
        c3_thresh = 0.0
        seeds = seeds[:1]  # single seed for speed
        print("[V3-EXQ-108a] SMOKE TEST MODE", flush=True)
    else:
        n_train = args.train_episodes
        steps_ep = args.steps_per_episode
        n_sequences = args.n_sequences
        horizon = args.rollout_horizon
        warmup = args.warmup_steps
        goal_max = args.goal_max_steps
        c1_thresh = args.c1_threshold
        c3_thresh = args.c3_var_threshold

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
        c1_threshold=c1_thresh,
        c3_var_threshold=c3_thresh,
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
        "c1_threshold": c1_thresh,
        "c3_var_threshold": c3_thresh,
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

    if args.dry_run:
        print("[V3-EXQ-108a] SMOKE TEST COMPLETE", flush=True)
        for k in ["c1_mean_prox_delta", "c1_pass", "c2_pass", "c3_pass",
                  "c3_per_seed_vars", "criteria_met"]:
            print(f"  {k}: {result.get(k, 'N/A')}", flush=True)

    # --- runner-conformance sentinel ---
    _outcome_raw = str(result.get("status", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
