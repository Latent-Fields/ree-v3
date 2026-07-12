#!/opt/local/bin/python3
"""
V3-EXQ-732a -- POWER-FIXED POLICY-LEARNING DISCRIMINATOR (H1 REE action-stack vs H2
observation interface). Supersedes V3-EXQ-732.

SLEEP DRIVER: none (no sleep loop; use_sleep_loop / sws_enabled / rem_enabled all OFF).

WHY THIS EXISTS (a DIAGNOSTIC, not an evidence falsifier; the self-routed label below is a
HYPOTHESIS the adjudication pipeline can falsify, NOT a governance verdict).

V3-EXQ-732 asked the same H1/H2 question but was UNDER-POWERED. Its failure autopsy
(REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-732_2026-07-10.{md,json}) REJECTED
732's self-routed `H2_observation_interface_unlearnable` as UNCONFIRMED: the shared
200-episode Monte-Carlo A2C budget (~200 gradient updates, entropy 0.01, net-negative
contamination-dominated reward) was sized for the B1-vs-B2 representation contrast, NOT for
learnability. 732's H2 arm B2 (vanilla raw-obs A2C) was in fact the BEST arm (0.35 res/ep)
and foraged in 12/20 episodes on seed 42 -- a partial-LEARNING signature, not an unlearnable
interface. 732's readiness battery validated env-achievability (oracle 6.05) but had NO
learner-adequacy gate, so a sub-floor B2 conflated "unlearnable observation" with
"under-powered learner" (the exact V3-EXQ-642 precondition-unmet failure mode).

732a is the implementation POWER-FIX (same scientific question, alphabetic suffix). Four
changes prescribed by the autopsy requeue_spec:

  1. PRIMARY LEVER -- replace the end-of-episode Monte-Carlo A2C (~200 updates) with an
     off-policy/minibatched PPO learner doing MANY minibatched gradient updates per rollout,
     at a 10-50x larger EPISODE budget known to solve comparable gridworld foraging. The SAME
     upgraded learner + budget is applied to BOTH B1 and B2 so the representation-front-end
     contrast stays powered (~200 MC updates -> ~2000 PPO episodes x 4 epochs x minibatches =
     order 6000+ gradient updates, ~30x update budget at ~10x env-step budget).
  2. STRONGER EXPLORATION -- entropy_beta raised 0.01 -> 0.03 AND a count-based novelty bonus
     (NOVELTY_COEF / sqrt(visit_count[x,y])) to cover the sparse pre-first-resource phase.
  3. RETURN NORMALIZATION -- an explicit dense FORAGE_BONUS per resource collected + a
     running-std reward scaler + per-batch advantage normalization, so foraging is NOT
     drowned by the (small, dense) contamination penalties that made 732's net reward
     negative. The load-bearing DV (P2 mean_resources_per_episode) is measured on a PURE
     GREEDY eval that counts REAL env resource transitions -- reward shaping touches ONLY the
     training signal, never the DV.
  4. LEARNER-ADEQUACY READINESS GATE (the direct fix for the 732 mis-read) -- an L0 sanity
     cell runs the SAME vanilla PPO learner on a PLAIN sanity env (hazards OFF, reef OFF,
     contamination OFF: pure foraging). The learner must clear >= LEARNER_ADEQUACY_FRAC of
     the sanity oracle within budget BEFORE any sub-floor B2 is allowed to read as an H2
     verdict. If the vanilla learner cannot forage even the trivially-solvable sanity env,
     the learner is too weak -> `substrate_not_ready_requeue`, NEVER an H2 verdict. This gate
     asserts the SAME statistic (resources/episode) that the H2 branch routes on.

This experiment discriminates H1 vs H2 BEFORE any /implement-substrate build commits to a
target. It does NOT test or weight any claim (claim_ids=[], experiment_purpose=diagnostic,
non_contributory) -- it decides which substrate the follow-on f_dominance_conversion_ceiling
build should target (a policy/action-learning substrate vs the observation encoding).

BRAKE-EXEMPT. This asks a DIFFERENT question ("is REE's action stack or the observation
interface the competence bottleneck") than the conversion-ceiling claims ARC-062 / MECH-309.
It tags NO claim, so the /failure-autopsy re-derive brake does not apply.

------------------------------------------------------------------------------------------
DESIGN -- three discriminator arms + L0 learner-adequacy sanity arm + two greedy-oracle
positive controls. Env, seeds (42/43/44), steps/episode (200), P2 eval protocol (20 eps),
competence floor (1.0 res/ep), and the greedy nearest-resource ORACLE are ALL reused VERBATIM
from V3-EXQ-724 (imported, not copied), so the DV statistic (P2 mean resources/episode via
env.step info transition_type=='resource') is bit-identical across the lineage.

  B0  ree_biashead_reinforce_allon  -- H0 INCOMPETENCE ANCHOR. Exactly V3-EXQ-724 arm A0:
        all-ON REE stack, P0=200 SD-056 e2 prediction warmup, P1=90 TWO-head REINFORCE
        (lateral-PFC bias head + OFC devaluation head), encoder FROZEN through P1, P2 frozen
        eval. Computed by calling the 724 harness `_run_cell` on the A0 arm dict, so B0
        reproduces the ~0.25-res/ep incompetence by construction. (Budget faithful to 724,
        NOT power-matched -- B0 is the known-incompetent anchor, not a learner under test.)

  B1  ree_repr_full_ppo_head  -- H1 TEST (REE action stack). REE's prediction-trained
        representation given a REAL POWERED action-learning head:
          - P0 = 200 episodes SD-056 e2 world-forward contrastive warmup over the all-ON REE
            encoder (random-action transitions); encoder then FROZEN (phased training).
          - P1 = P1_PPO_EPISODES online PPO. Each tick the frozen all-ON REE encoder produces
            the latent (z_self (+) z_world, no_grad, detached); a FULL-CAPACITY trainable
            trunk + PPO actor-critic head (own Adam) maps that latent to an action and is
            trained by clipped-surrogate PPO with GAE. Phased: encoder frozen, head trained
            on detached latents.
          - P2 = 20 episodes frozen greedy eval.

  B2  vanilla_ppo_rawobs  -- H2 TEST (observation interface). A NON-REE vanilla PPO policy on
        the IDENTICAL observation vector, no REE machinery:
          - P1 = P1_PPO_EPISODES online PPO over a matched-capacity trainable trunk + PPO
            head reading the raw observation vector (body_state (+) world_state (+) harm_obs
            (+) harm_obs_a (+) harm_history) -- the SAME raw channels B1's REE encoder senses.
          - P2 = 20 episodes frozen greedy eval.

  L0  learner_adequacy_ppo_sanity  -- LEARNER-ADEQUACY GATE (NOT a discriminator arm). The
        SAME vanilla PPO learner as B2, on a PLAIN sanity env (hazards OFF, reef OFF,
        contamination OFF). Proves the learner can extract foraging from raw observation when
        the env is trivially solvable. Compared against the sanity-env oracle. If L0 clears
        >= LEARNER_ADEQUACY_FRAC of the sanity oracle on a majority of seeds, the learner is
        ADEQUATE and a sub-floor B2 is licensed to read as H2; otherwise the learner is the
        bottleneck -> substrate_not_ready_requeue.

MATCHED LEARNER. B1 and B2 (and L0) use the SAME PPO learner with the SAME hyperparameters
and the SAME P1_PPO_EPISODES budget; the ONLY difference between B1 and B2 is the
representation front-end (REE prediction-trained latent vs raw observation). A single shared
learner isolates the representation front-end with NO learning-algorithm confound.

POSITIVE CONTROLS (readiness). (i) The greedy nearest-resource ORACLE (V3-EXQ-724
`_oracle_action`) on the REAL env proves the 1.0 floor is ACHIEVABLE. (ii) The SAME oracle on
the sanity env sets the L0 adequacy reference. If even the real-env oracle cannot clear the
floor, NO architecture conclusion is licensed -> substrate_not_ready_requeue.

DV (load-bearing): P2 mean_resources_per_episode (majority of seeds vs the 1.0 floor),
measured on PURE GREEDY eval counting REAL env resource transitions (untouched by training
reward shaping).

PRE-REGISTERED SELF-ROUTE (HYPOTHESIS, not a verdict):
  * READINESS fails (real-env oracle below floor OR B0 does NOT reproduce incompetence OR any
    cell logs < MIN_P2_EPISODES OR the LEARNER-ADEQUACY gate fails) -> `substrate_not_ready_requeue`.
  * B2 clears AND B1 clears -> `H1_policy_head_recovers_competence`: bias-head-only was the
    bottleneck; a real POWERED action head over REE's representation forages. Route to
    /implement-substrate to build an action/policy-learning substrate under
    f_dominance_conversion_ceiling (informed by MECH-455 competence-based IM).
  * B2 clears AND B1 does NOT clear -> `H1_deeper_ree_representation_obstructs`: a vanilla
    policy learns the raw observation but the SAME policy over REE's representation cannot ->
    the deficit is in REE's representation processing, not just the head.
  * B2 does NOT clear (WITH the learner-adequacy gate PASSED) -> `H2_observation_interface_unlearnable`:
    even a POWERED vanilla policy on the identical observation cannot forage, yet the SAME
    learner clears the sanity env -> the observation ENCODING is the bottleneck. Route to
    /implement-substrate on the observation encoding. (Only meaningful BECAUSE the adequacy
    gate rules out "learner too weak" -- the 732 mis-read this branch could not exclude.)
  * B1 clears AND B2 does NOT clear -> `flag_implausible_leakage_check`: a REE-encoded head
    beating a vanilla policy on observation derived from the SAME env information is
    implausible -> check for leakage / privileged input before trusting either arm.

EVIDENCE-FOR / EVIDENCE-AGAINST (diagnostic-description requirement), per arm:
  * B0 (anchor): EVIDENCE-FOR reproducing 724's incompetence (below floor) is the premise
    that licenses the localization; B0 clearing the floor would be EVIDENCE-AGAINST the
    premise -> substrate_not_ready_requeue.
  * B1 (H1): B1 clearing the floor is EVIDENCE that REE's representation is usable for action
    with a real powered head (deficit is the action head / policy-learning mechanism). B1
    below floor is EVIDENCE-AGAINST "the head alone recovers competence".
  * B2 (H2): B2 clearing the floor is EVIDENCE-AGAINST H2 (the observation IS learnable) and
    FOR H1. B2 below floor (adequacy gate passed) is EVIDENCE-FOR H2.
  * L0 (adequacy): L0 clearing the sanity floor is EVIDENCE the learner is adequate (so a
    B2 sub-floor is a real observation-encoding signal); L0 below the sanity floor is
    EVIDENCE the learner is the bottleneck -> substrate_not_ready_requeue, NOT an H2 verdict.
  The self-routed label is a HYPOTHESIS for /failure-autopsy adjudication; this experiment
  PROMOTES / DEMOTES NOTHING and tags NO claim.

Sourced substrate: experiments/v3_exq_724_competence_localization_diagnostic.py (env, oracle,
all-ON stack, B0 harness, SD-056 e2 warmup, obs helpers), ree_core/agent.py
(REEAgent.sense -> latent.z_self/z_world), ree_core/environment/causal_grid_world.py (step()
-> harm_signal reward, info.transition_type, obs_dict channels). See
REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-732_2026-07-10.md and
REE_assembly/evidence/planning/ree_ai_design_critique_plan.md (WS-1).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from collections import deque

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell

# Reuse the V3-EXQ-724 harness verbatim (env, oracle, B0 all-ON cell, SD-056 e2 warmup,
# obs helpers, and all shared constants) so B0 and the oracle are bit-identical to 724/732.
import experiments.v3_exq_724_competence_localization_diagnostic as x724
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_732a_policy_learning_discriminator"
QUEUE_ID = "V3-EXQ-732a"
CLAIM_IDS: List[str] = []                 # tags NO claim -- pure diagnostic
EXPERIMENT_PURPOSE = "diagnostic"

# ---------------------------------------------------------------------------
# Shared knobs reused from V3-EXQ-724 (identical env / floor / protocol)
# ---------------------------------------------------------------------------
SEEDS = list(x724.SEEDS)                          # [42, 43, 44]
P0_WARMUP_EPISODES = x724.P0_WARMUP_EPISODES      # 200 (REE encoder / e2 prediction warmup)
P1_B0_REINFORCE = x724.P1_SHORT                   # 90  (B0 = 724 A0 bias-head REINFORCE)
P2_EVAL_EPISODES = x724.P2_EVAL_EPISODES          # 20
STEPS_PER_EPISODE = x724.STEPS_PER_EPISODE        # 200
N_ORACLE_EPISODES = x724.N_ORACLE_EPISODES        # 20
COMPETENCE_RESOURCE_FLOOR = x724.COMPETENCE_RESOURCE_FLOOR  # 1.0
COMPETENCE_MIN_SEEDS = x724.COMPETENCE_MIN_SEEDS  # 2 of 3 (majority)
MIN_P2_EPISODES = x724.MIN_P2_EPISODES            # 5

# ---------------------------------------------------------------------------
# PPO learner (matched across B1 / B2 / L0) -- the 732->732a power fix.
# Budget: P1_PPO_EPISODES = 2000 = 10x the 732 A2C episode budget (200). PPO does
# PPO_EPOCHS x (rollout_steps / minibatch) gradient updates per rollout, so at ROLLOUT of 8
# episodes (~1600 steps) this is ~250 rollouts x 4 epochs x ~6 minibatches ~= 6000 gradient
# updates (~30x the 732 ~200 MC updates) -- squarely inside the autopsy's 10-50x band.
# ---------------------------------------------------------------------------
P1_PPO_EPISODES = 2000           # online PPO training episodes (B1 / B2 / L0, matched budget)
PPO_TRUNK_HIDDEN = 128           # 2-layer trainable trunk width (all arms)
PPO_LR = 3e-4
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP = 0.2
PPO_ENTROPY_BETA = 0.03          # raised from 732's 0.01 (stronger exploration)
PPO_VALUE_COEF = 0.5
PPO_GRAD_CLIP = 0.5
PPO_EPOCHS = 4                   # minibatch update epochs per rollout
PPO_MINIBATCH_SIZE = 256
PPO_ROLLOUT_EPISODES = 8         # episodes collected per PPO update batch

# Return normalization / exploration shaping (training reward only; DV is greedy + untouched).
FORAGE_BONUS = 1.0               # explicit dense +reward per resource collected in training
NOVELTY_COEF = 0.1               # count-based novelty bonus = NOVELTY_COEF / sqrt(visits[x,y])
REWARD_STD_EPS = 1e-6            # running-std reward scaler epsilon

# Learner-adequacy gate.
LEARNER_ADEQUACY_FRAC = 0.5      # L0 must clear >= this fraction of the sanity oracle

# SD-056 e2 prediction-warmup levers (reuse 724 constants)
E2_CONTRASTIVE_LR = x724.E2_CONTRASTIVE_LR
E2_TRAIN_EVERY_K_TICKS = x724.E2_TRAIN_EVERY_K_TICKS
TRANSITION_BUFFER_MAX = x724.TRANSITION_BUFFER_MAX

# ---------------------------------------------------------------------------
# Sanity env for the learner-adequacy gate: pure foraging -- hazards / reef / contamination
# OFF, resources + respawn ON. Everything else identical to ENV_KWARGS.
# ---------------------------------------------------------------------------
SANITY_ENV_KWARGS = dict(x724.ENV_KWARGS)
SANITY_ENV_KWARGS.update(
    num_hazards=0,
    hazard_harm=0.0,
    reef_enabled=False,
    n_reef_patches=0,
    hazard_food_attraction=0.0,
    reef_bipartite_layout=False,
)


def _make_sanity_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **SANITY_ENV_KWARGS)


# ---------------------------------------------------------------------------
# Dry-run budget (tiny; smoke stays fast)
# ---------------------------------------------------------------------------
DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1_B0 = 2
DRY_RUN_P1_PPO = 6               # >= 2 rollouts at DRY rollout size so PPO update path runs
DRY_RUN_P2 = 2
DRY_RUN_STEPS = 15
DRY_RUN_ORACLE_EPS = 2
DRY_RUN_ROLLOUT_EPISODES = 3

# Arm ids / roles
B0_ARM_ID = "B0_ree_biashead_reinforce_allon"
B1_ARM_ID = "B1_ree_repr_full_ppo_head"
B2_ARM_ID = "B2_vanilla_ppo_rawobs"
L0_ARM_ID = "L0_learner_adequacy_ppo_sanity"


# ---------------------------------------------------------------------------
# PPO policy/value network (shared by B1 / B2 / L0).
#   B1: in_dim = self_dim + world_dim (REE latent), trunk trainable, encoder frozen.
#   B2/L0: in_dim = raw observation width, trunk trainable, no encoder.
# ---------------------------------------------------------------------------
class PPOPolicy(nn.Module):
    def __init__(self, in_dim: int, action_dim: int, hidden: int = PPO_TRUNK_HIDDEN):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(state)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value


class _RunningStd:
    """Welford running variance (no mean subtraction on apply -- preserves reward sign)."""

    def __init__(self) -> None:
        self.count = 0.0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float) -> None:
        self.count += 1.0
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)

    @property
    def std(self) -> float:
        if self.count < 2.0:
            return 1.0
        return float(math.sqrt(self.m2 / (self.count - 1.0)))


def _novelty_bonus(counter: Dict[Tuple[int, int], int], pos: Tuple[int, int]) -> float:
    """Count-based exploration bonus for the sparse pre-first-resource phase."""
    counter[pos] = counter.get(pos, 0) + 1
    return float(NOVELTY_COEF / math.sqrt(counter[pos]))


def _ppo_update(
    policy: PPOPolicy,
    optimiser: torch.optim.Optimizer,
    states: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    device: torch.device,
) -> None:
    """PPO clipped-surrogate update: PPO_EPOCHS passes of minibatched SGD over one rollout."""
    n = states.shape[0]
    if n == 0:
        return
    # Per-batch advantage normalization (return/advantage scale control).
    adv = advantages
    if n > 1:
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    for _epoch in range(PPO_EPOCHS):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, PPO_MINIBATCH_SIZE):
            idx = perm[start:start + PPO_MINIBATCH_SIZE]
            mb_states = states[idx]
            mb_actions = actions[idx]
            mb_old_logp = old_log_probs[idx]
            mb_returns = returns[idx]
            mb_adv = adv[idx]
            logits, values = policy(mb_states)
            dist = torch.distributions.Categorical(logits=logits)
            new_logp = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_logp - mb_old_logp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (mb_returns - values).pow(2).mean()
            loss = policy_loss + PPO_VALUE_COEF * value_loss - PPO_ENTROPY_BETA * entropy
            if not torch.isfinite(loss):
                continue
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), PPO_GRAD_CLIP)
            optimiser.step()


def _compute_gae(
    rewards: List[float],
    values: List[float],
    bootstrap_value: float,
    terminal: bool,
) -> Tuple[List[float], List[float]]:
    """GAE-lambda advantages + returns for ONE episode. Truncation bootstraps the last
    value; a true env `done` bootstraps 0."""
    n = len(rewards)
    advantages = [0.0] * n
    last_gae = 0.0
    for t in reversed(range(n)):
        if t == n - 1:
            next_value = 0.0 if terminal else bootstrap_value
            next_nonterminal = 0.0 if terminal else 1.0
        else:
            next_value = values[t + 1]
            next_nonterminal = 1.0
        delta = rewards[t] + PPO_GAMMA * next_value * next_nonterminal - values[t]
        last_gae = delta + PPO_GAMMA * PPO_GAE_LAMBDA * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = [advantages[t] + values[t] for t in range(n)]
    return advantages, returns


# ---------------------------------------------------------------------------
# Raw observation vector for B2 / L0 (identical channels the REE encoder senses).
# ---------------------------------------------------------------------------
_RAW_OBS_KEYS = ("body_state", "world_state", "harm_obs", "harm_obs_a", "harm_history")


def _raw_obs_dim(obs_dict: Dict[str, Any]) -> int:
    total = 0
    for k in _RAW_OBS_KEYS:
        v = obs_dict.get(k)
        if v is None:
            raise KeyError(f"raw obs key {k!r} absent from obs_dict")
        total += int(v.reshape(-1).shape[0])
    return total


def _raw_obs_vector(obs_dict: Dict[str, Any], device: torch.device) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    for k in _RAW_OBS_KEYS:
        v = obs_dict[k]
        parts.append(v.float().reshape(-1))
    return torch.cat(parts, dim=0).to(device).unsqueeze(0)


# ---------------------------------------------------------------------------
# B1 REE-latent state from a frozen all-ON encoder pass (no grad; z_self (+) z_world).
# ---------------------------------------------------------------------------
def _b1_latent_state(agent, obs_dict: Dict[str, Any], device: torch.device) -> torch.Tensor:
    body = obs_dict["body_state"].float()
    world = obs_dict["world_state"].float()
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    with torch.no_grad():
        latent = agent.sense(
            obs_body=body,
            obs_world=world,
            obs_harm=x724._obs_harm(obs_dict),
            obs_harm_a=x724._obs_harm_a(obs_dict),
            obs_harm_history=x724._obs_harm_history(obs_dict),
        )
        state = torch.cat(
            [latent.z_self.reshape(1, -1), latent.z_world.reshape(1, -1)], dim=-1
        ).detach()
    return state.to(device)


def _make_all_on_agent(env):
    kwargs = x724._base_config_kwargs(env)
    kwargs.update(x724._all_on_extra_kwargs())
    cfg = x724.REEConfig.from_dims(**kwargs)
    return x724.REEAgent(cfg)


# ---------------------------------------------------------------------------
# P2 frozen greedy eval shared by B1 / B2 / L0 (SAME resource statistic as 724).
# Reward shaping is NOT applied here -- the DV counts REAL env resource transitions.
# ---------------------------------------------------------------------------
def _eval_policy(
    env,
    state_fn: Callable[[Dict[str, Any]], torch.Tensor],
    policy: PPOPolicy,
    n_eval: int,
    steps_per_episode: int,
    on_episode_reset: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    ep_resources: List[int] = []
    ep_hazard_hits: List[int] = []
    ep_contaminations: List[int] = []
    ep_rewards: List[float] = []
    policy.eval()
    for _ep in range(n_eval):
        _, obs_dict = env.reset()
        if on_episode_reset is not None:
            on_episode_reset()
        resources = 0
        hazard_hits = 0
        contaminations = 0
        reward_signal = 0.0
        for _step in range(steps_per_episode):
            state = state_fn(obs_dict)
            with torch.no_grad():
                logits, _v = policy(state)
                a_idx = int(torch.argmax(logits, dim=-1).item())
            _, harm_signal, done, info, obs_dict = env.step(a_idx)
            reward_signal += float(harm_signal)
            ttype = str(info.get("transition_type", "none"))
            if ttype == "resource":
                resources += 1
            elif ttype == "env_caused_hazard":
                hazard_hits += 1
            if ttype == "agent_caused_hazard" or float(
                info.get("contamination_delta", 0.0)
            ) > 0.0:
                contaminations += 1
            if done:
                break
        ep_resources.append(resources)
        ep_hazard_hits.append(hazard_hits)
        ep_contaminations.append(contaminations)
        ep_rewards.append(reward_signal)
    policy.train()
    n = len(ep_resources)
    return {
        "n_p2_eps_completed": int(n),
        "mean_resources_per_episode": round(
            float(sum(ep_resources) / n) if n else 0.0, 6
        ),
        "mean_hazard_hits_per_episode": round(
            float(sum(ep_hazard_hits) / n) if n else 0.0, 6
        ),
        "mean_contaminations_per_episode": round(
            float(sum(ep_contaminations) / n) if n else 0.0, 6
        ),
        "mean_episode_reward": round(float(sum(ep_rewards) / n) if n else 0.0, 6),
        "per_p2_episode_resources": [int(x) for x in ep_resources],
    }


# ---------------------------------------------------------------------------
# PPO training loop over one env (state_fn abstracts REE-latent vs raw-obs).
# Collects PPO_ROLLOUT_EPISODES-episode rollouts, applies FORAGE_BONUS + novelty + running-std
# reward scaling, computes per-episode GAE, then minibatched clipped-surrogate updates.
# Returns mean training resources/ep over the last rollout (progress signal only).
# ---------------------------------------------------------------------------
def _ppo_train(
    env,
    state_fn: Callable[[Dict[str, Any]], torch.Tensor],
    policy: PPOPolicy,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    n_episodes: int,
    rollout_episodes: int,
    steps_per_episode: int,
    arm_id: str,
    seed: int,
    total_denominator: int,
    ep_offset: int,
    on_episode_reset: Optional[Callable[[], None]] = None,
) -> None:
    reward_std = _RunningStd()
    novelty_counter: Dict[Tuple[int, int], int] = {}
    episodes_done = 0

    while episodes_done < n_episodes:
        batch_states: List[torch.Tensor] = []
        batch_actions: List[int] = []
        batch_old_logp: List[float] = []
        batch_returns: List[float] = []
        batch_advantages: List[float] = []

        eps_this_batch = min(rollout_episodes, n_episodes - episodes_done)
        for _b in range(eps_this_batch):
            _, obs_dict = env.reset()
            if on_episode_reset is not None:
                on_episode_reset()
            ep_states: List[torch.Tensor] = []
            ep_actions: List[int] = []
            ep_logp: List[float] = []
            ep_values: List[float] = []
            ep_rewards: List[float] = []
            terminal = False
            bootstrap_value = 0.0
            for _step in range(steps_per_episode):
                state = state_fn(obs_dict)
                with torch.no_grad():
                    logits, value = policy(state)
                    dist = torch.distributions.Categorical(logits=logits.reshape(1, -1))
                    a = dist.sample()
                    logp = dist.log_prob(a)
                a_idx = int(a.item())
                _, harm_signal, done, info, obs_dict = env.step(a_idx)
                ttype = str(info.get("transition_type", "none"))
                pos = (int(env.agent_x), int(env.agent_y))
                shaped = (
                    float(harm_signal)
                    + (FORAGE_BONUS if ttype == "resource" else 0.0)
                    + _novelty_bonus(novelty_counter, pos)
                )
                reward_std.update(shaped)
                ep_states.append(state.reshape(-1).detach())
                ep_actions.append(a_idx)
                ep_logp.append(float(logp.item()))
                ep_values.append(float(value.item()))
                ep_rewards.append(shaped)
                if done:
                    terminal = True
                    break
            # Bootstrap value for truncated (non-terminal) episodes -- one extra state_fn
            # call on the final obs (NOT per-step, which would double B1's encoder cost).
            # The value head is trained on SCALED returns, so its output is already in the
            # scaled-return space that GAE consumes -- do NOT re-divide by `scale`.
            if not terminal:
                with torch.no_grad():
                    _, bv = policy(state_fn(obs_dict))
                bootstrap_value = float(bv.item())
            # Running-std reward scaling (no mean shift -> foraging sign preserved).
            scale = reward_std.std + REWARD_STD_EPS
            scaled_rewards = [r / scale for r in ep_rewards]
            advs, rets = _compute_gae(
                scaled_rewards, ep_values, bootstrap_value, terminal
            )
            batch_states.extend(ep_states)
            batch_actions.extend(ep_actions)
            batch_old_logp.extend(ep_logp)
            batch_returns.extend(rets)
            batch_advantages.extend(advs)
            episodes_done += 1
            cur = ep_offset + episodes_done
            if episodes_done % 200 == 0 or episodes_done == n_episodes:
                print(
                    f"  [train] disc arm={arm_id} seed={seed} phase=P1 "
                    f"ep {cur}/{total_denominator}", flush=True,
                )

        if not batch_states:
            continue
        states_t = torch.stack(batch_states).to(device)
        actions_t = torch.tensor(batch_actions, dtype=torch.long, device=device)
        old_logp_t = torch.tensor(batch_old_logp, dtype=torch.float32, device=device)
        returns_t = torch.tensor(batch_returns, dtype=torch.float32, device=device)
        adv_t = torch.tensor(batch_advantages, dtype=torch.float32, device=device)
        _ppo_update(
            policy, optimiser, states_t, actions_t, old_logp_t, returns_t, adv_t, device
        )


# ---------------------------------------------------------------------------
# B1 cell: REE prediction warmup (encoder + e2) -> frozen -> PPO head over latent.
# ---------------------------------------------------------------------------
def _run_b1_cell(
    seed: int,
    p0_episodes: int,
    p1_ppo_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
    rollout_episodes: int,
) -> Dict[str, Any]:
    device = torch.device("cpu")
    env = x724._make_env(seed)
    agent = _make_all_on_agent(env)

    total = p0_episodes + p1_ppo_episodes + p2_episodes

    # ---- P0: SD-056 e2 world-forward prediction warmup (random-action transitions) ----
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)
    transition_buffer: Deque[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)
    for ep in range(p0_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        pending_capture: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        tick_in_ep = 0
        for _step in range(steps_per_episode):
            body = obs_dict["body_state"].float()
            world = obs_dict["world_state"].float()
            if body.dim() == 1:
                body = body.unsqueeze(0)
            if world.dim() == 1:
                world = world.unsqueeze(0)
            with torch.no_grad():
                latent = agent.sense(
                    obs_body=body, obs_world=world,
                    obs_harm=x724._obs_harm(obs_dict),
                    obs_harm_a=x724._obs_harm_a(obs_dict),
                    obs_harm_history=x724._obs_harm_history(obs_dict),
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

            a_idx = int(np.random.randint(0, env.action_dim))
            action = torch.zeros(1, env.action_dim, device=agent.device)
            action[0, a_idx] = 1.0
            if torch.isfinite(latent.z_world).all():
                pending_capture = (
                    latent.z_world.detach().reshape(-1).clone(),
                    action.detach().reshape(-1).clone(),
                )
            if tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0:
                x724._e2_contrastive_step(
                    agent=agent, buffer=transition_buffer,
                    optimiser=e2_opt, rng=sample_rng,
                )
            _, _harm, done, _info, obs_dict = env.step(a_idx)
            tick_in_ep += 1
            if done:
                break
        cur = ep + 1
        if cur % 25 == 0 or cur == p0_episodes:
            print(
                f"  [train] disc arm={B1_ARM_ID} seed={seed} phase=P0 "
                f"ep {cur}/{total}", flush=True,
            )

    # ---- Freeze encoder; PPO head over the frozen REE latent ----
    def _state_fn(obs_dict: Dict[str, Any]) -> torch.Tensor:
        return _b1_latent_state(agent, obs_dict, device)

    _, probe_obs = env.reset()
    agent.reset()
    in_dim = int(_state_fn(probe_obs).shape[-1])
    policy = PPOPolicy(in_dim=in_dim, action_dim=env.action_dim).to(device)
    optimiser = torch.optim.Adam(policy.parameters(), lr=PPO_LR)

    _ppo_train(
        env, _state_fn, policy, optimiser, device,
        n_episodes=p1_ppo_episodes, rollout_episodes=rollout_episodes,
        steps_per_episode=steps_per_episode, arm_id=B1_ARM_ID, seed=seed,
        total_denominator=total, ep_offset=p0_episodes,
        on_episode_reset=agent.reset,
    )

    # ---- P2 frozen greedy eval ----
    ev = _eval_policy(
        env, _state_fn, policy, p2_episodes, steps_per_episode,
        on_episode_reset=agent.reset,
    )
    print(
        f"  [train] disc arm={B1_ARM_ID} seed={seed} phase=P2 ep {total}/{total}",
        flush=True,
    )
    return _finalize_cell_row(
        arm_id=B1_ARM_ID, role="h1_ree_representation_full_head", seed=seed,
        p0=p0_episodes, p1=p1_ppo_episodes, p2=p2_episodes, ev=ev, error_note=None,
    )


# ---------------------------------------------------------------------------
# Vanilla PPO over the raw observation vector (no REE machinery). Used by BOTH
# B2 (real env) and L0 (sanity env).
# ---------------------------------------------------------------------------
def _run_rawobs_ppo_cell(
    arm_id: str,
    role: str,
    env,
    seed: int,
    p1_ppo_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
    rollout_episodes: int,
) -> Dict[str, Any]:
    device = torch.device("cpu")
    _, obs0 = env.reset()
    in_dim = _raw_obs_dim(obs0)
    policy = PPOPolicy(in_dim=in_dim, action_dim=env.action_dim).to(device)
    optimiser = torch.optim.Adam(policy.parameters(), lr=PPO_LR)

    def _state_fn(obs_dict: Dict[str, Any]) -> torch.Tensor:
        return _raw_obs_vector(obs_dict, device)

    total = p1_ppo_episodes + p2_episodes
    _ppo_train(
        env, _state_fn, policy, optimiser, device,
        n_episodes=p1_ppo_episodes, rollout_episodes=rollout_episodes,
        steps_per_episode=steps_per_episode, arm_id=arm_id, seed=seed,
        total_denominator=total, ep_offset=0,
    )

    ev = _eval_policy(env, _state_fn, policy, p2_episodes, steps_per_episode)
    print(
        f"  [train] disc arm={arm_id} seed={seed} phase=P2 ep {total}/{total}",
        flush=True,
    )
    return _finalize_cell_row(
        arm_id=arm_id, role=role, seed=seed,
        p0=0, p1=p1_ppo_episodes, p2=p2_episodes, ev=ev, error_note=None,
        raw_obs_dim=in_dim,
    )


def _finalize_cell_row(
    arm_id: str,
    role: str,
    seed: int,
    p0: int,
    p1: int,
    p2: int,
    ev: Dict[str, Any],
    error_note: Optional[str],
    raw_obs_dim: Optional[int] = None,
) -> Dict[str, Any]:
    mean_res = float(ev.get("mean_resources_per_episode", 0.0))
    row = {
        "arm_id": arm_id,
        "arm_role": role,
        "seed": int(seed),
        "p0_episodes": int(p0),
        "p1_episodes": int(p1),
        "p2_episodes_requested": int(p2),
        "n_p2_eps_completed": int(ev.get("n_p2_eps_completed", 0)),
        "error_note": error_note,
        # ----- LOAD-BEARING DV -----
        "mean_resources_per_episode": round(mean_res, 6),
        "competence_supra_floor": bool(mean_res >= COMPETENCE_RESOURCE_FLOOR),
        # ----- context -----
        "mean_hazard_hits_per_episode": ev.get("mean_hazard_hits_per_episode", 0.0),
        "mean_contaminations_per_episode": ev.get("mean_contaminations_per_episode", 0.0),
        "mean_episode_reward": ev.get("mean_episode_reward", 0.0),
        "per_p2_episode_resources": ev.get("per_p2_episode_resources", []),
    }
    if raw_obs_dim is not None:
        row["raw_obs_dim"] = int(raw_obs_dim)
    return row


# ---------------------------------------------------------------------------
# B0 cell: exactly V3-EXQ-724 arm A0 (imported harness).
# ---------------------------------------------------------------------------
def _run_b0_cell(
    seed: int,
    p0_episodes: int,
    p1_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    arm = {
        "arm_id": B0_ARM_ID, "kind": "all_on",
        "p1": int(p1_episodes), "e2_train_in_p1": False, "role": "baseline_anchor",
    }
    row = x724._run_cell(arm, seed, p0_episodes, p2_episodes, steps_per_episode)
    row["arm_role"] = "h0_incompetence_anchor"
    return row


# ---------------------------------------------------------------------------
# Oracle on an arbitrary env (real or sanity). Same greedy nearest-resource policy + same
# resources/episode statistic as 724.
# ---------------------------------------------------------------------------
def _run_oracle_on(
    env_factory: Callable[[int], CausalGridWorldV2],
    seed: int,
    n_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    env = env_factory(seed)
    ep_resources: List[int] = []
    for _ep in range(n_episodes):
        env.reset()
        collected = 0
        for _step in range(steps_per_episode):
            a = x724._oracle_action(env)
            _, _harm, done, info, _obs = env.step(a)
            if str(info.get("transition_type", "none")) == "resource":
                collected += 1
            if done:
                break
        ep_resources.append(collected)
    mean_res = float(sum(ep_resources) / len(ep_resources)) if ep_resources else 0.0
    return {
        "seed": int(seed),
        "n_episodes": int(len(ep_resources)),
        "mean_resources_per_episode": round(mean_res, 6),
        "max_resources_in_episode": int(max(ep_resources)) if ep_resources else 0,
    }


def _mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def _arm_majority_supra(rows: List[Dict[str, Any]], min_seeds: int) -> bool:
    n = sum(
        1 for r in rows
        if r.get("error_note") is None and r.get("competence_supra_floor")
    )
    return bool(n >= min_seeds)


def _aggregate_arm(arm_id: str, role: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if r.get("error_note") is None]
    return {
        "arm_id": arm_id,
        "role": role,
        "n_seeds_ok": int(len(ok)),
        "n_seeds_min_p2": int(
            sum(1 for r in ok if r.get("n_p2_eps_completed", 0) >= MIN_P2_EPISODES)
        ),
        "mean_resources_per_episode_mean": round(
            _mean([float(r["mean_resources_per_episode"]) for r in ok]), 6
        ),
        "n_seeds_supra_floor": int(sum(1 for r in ok if r.get("competence_supra_floor"))),
        "majority_supra_floor": _arm_majority_supra(ok, COMPETENCE_MIN_SEEDS),
        "mean_episode_reward_mean": round(
            _mean([float(r.get("mean_episode_reward", 0.0)) for r in ok]), 6
        ),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_experiment(
    seeds: List[int],
    p0_episodes: int,
    p1_b0_episodes: int,
    p1_ppo_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
    oracle_episodes: int,
    rollout_episodes: int,
    dry_run: bool,
) -> Dict[str, Any]:
    print(
        f"Power-fixed policy-learning discriminator (3 discriminator arms + L0 "
        f"learner-adequacy arm) x {len(seeds)} seeds + 2 greedy-oracle controls; "
        f"PPO P1={p1_ppo_episodes} rollout={rollout_episodes} epochs={PPO_EPOCHS} "
        f"entropy={PPO_ENTROPY_BETA}; P0={p0_episodes}, P1_B0={p1_b0_episodes}, "
        f"P2={p2_episodes}, steps={steps_per_episode}, oracle_eps={oracle_episodes}, "
        f"dry_run={dry_run})",
        flush=True,
    )

    # ----- Positive-control oracles (real env + sanity env; per seed) -----
    oracle_rows: List[Dict[str, Any]] = []
    sanity_oracle_rows: List[Dict[str, Any]] = []
    for s in seeds:
        orow = _run_oracle_on(x724._make_env, s, oracle_episodes, steps_per_episode)
        oracle_rows.append(orow)
        srow = _run_oracle_on(_make_sanity_env, s, oracle_episodes, steps_per_episode)
        sanity_oracle_rows.append(srow)
        print(
            f"  [oracle] seed={s} real/ep={orow['mean_resources_per_episode']} "
            f"sanity/ep={srow['mean_resources_per_episode']}",
            flush=True,
        )
    oracle_mean_resources = _mean([o["mean_resources_per_episode"] for o in oracle_rows])
    oracle_min_resources = min(
        [o["mean_resources_per_episode"] for o in oracle_rows], default=0.0
    )
    oracle_clears_floor = bool(oracle_min_resources >= COMPETENCE_RESOURCE_FLOOR)
    sanity_oracle_by_seed = {
        int(r["seed"]): float(r["mean_resources_per_episode"]) for r in sanity_oracle_rows
    }

    # ----- Arm x seed cells (B0, B1, B2 discriminators + L0 adequacy) -----
    arm_specs = [
        (B0_ARM_ID, "h0_incompetence_anchor"),
        (B1_ARM_ID, "h1_ree_representation_full_head"),
        (B2_ARM_ID, "h2_observation_interface"),
        (L0_ARM_ID, "learner_adequacy_sanity"),
    ]
    cells: List[Dict[str, Any]] = []
    for arm_id, role in arm_specs:
        for s in seeds:
            print(f"Seed {s} Condition {arm_id}", flush=True)
            slice_cfg = {
                "arm_id": arm_id,
                "role": role,
                "p0_episodes": int(p0_episodes if arm_id == B1_ARM_ID
                                   else (p0_episodes if arm_id == B0_ARM_ID else 0)),
                "p1_episodes": int(
                    p1_b0_episodes if arm_id == B0_ARM_ID else p1_ppo_episodes
                ),
                "p2_episodes": int(p2_episodes),
                "steps_per_episode": int(steps_per_episode),
                "ppo": {
                    "hidden": PPO_TRUNK_HIDDEN, "lr": PPO_LR, "gamma": PPO_GAMMA,
                    "gae_lambda": PPO_GAE_LAMBDA, "clip": PPO_CLIP,
                    "entropy_beta": PPO_ENTROPY_BETA, "value_coef": PPO_VALUE_COEF,
                    "epochs": PPO_EPOCHS, "minibatch": PPO_MINIBATCH_SIZE,
                    "rollout_episodes": rollout_episodes,
                    "forage_bonus": FORAGE_BONUS, "novelty_coef": NOVELTY_COEF,
                } if arm_id != B0_ARM_ID else None,
                "env_kwargs": dict(
                    SANITY_ENV_KWARGS if arm_id == L0_ARM_ID else x724.ENV_KWARGS
                ),
            }
            with arm_cell(
                s,
                config_slice=slice_cfg,
                script_path=Path(__file__),
                config_slice_declared=True,
            ) as cell:
                if arm_id == B0_ARM_ID:
                    row = _run_b0_cell(
                        s, p0_episodes, p1_b0_episodes, p2_episodes, steps_per_episode
                    )
                elif arm_id == B1_ARM_ID:
                    row = _run_b1_cell(
                        s, p0_episodes, p1_ppo_episodes, p2_episodes, steps_per_episode,
                        rollout_episodes,
                    )
                elif arm_id == B2_ARM_ID:
                    row = _run_rawobs_ppo_cell(
                        B2_ARM_ID, "h2_observation_interface", x724._make_env(s), s,
                        p1_ppo_episodes, p2_episodes, steps_per_episode, rollout_episodes,
                    )
                else:  # L0
                    row = _run_rawobs_ppo_cell(
                        L0_ARM_ID, "learner_adequacy_sanity", _make_sanity_env(s), s,
                        p1_ppo_episodes, p2_episodes, steps_per_episode, rollout_episodes,
                    )
                    # L0 adequacy is judged vs the sanity oracle, not the 1.0 floor.
                    s_oracle = sanity_oracle_by_seed.get(int(s), 0.0)
                    row["sanity_oracle_resources_per_episode"] = round(s_oracle, 6)
                    row["learner_adequacy_threshold"] = round(
                        LEARNER_ADEQUACY_FRAC * s_oracle, 6
                    )
                    row["learner_adequate_this_seed"] = bool(
                        row["error_note"] is None
                        and row["mean_resources_per_episode"]
                        >= LEARNER_ADEQUACY_FRAC * s_oracle
                    )
                cell.stamp(row)
            cells.append(row)
            verdict = "PASS" if row.get("error_note") is None else "FAIL"
            print(
                f"verdict: {verdict} (arm={arm_id} seed={s} "
                f"resources/ep={row['mean_resources_per_episode']} "
                f"supra_floor={row['competence_supra_floor']})",
                flush=True,
            )

    # ----- Per-arm aggregation -----
    per_arm: Dict[str, Dict[str, Any]] = {}
    for arm_id, role in arm_specs:
        rows = [c for c in cells if c["arm_id"] == arm_id]
        per_arm[arm_id] = _aggregate_arm(arm_id, role, rows)

    # ----- Learner-adequacy gate (L0 vs sanity oracle, majority of seeds) -----
    l0_rows = [c for c in cells if c["arm_id"] == L0_ARM_ID and c.get("error_note") is None]
    n_l0_adequate = sum(1 for r in l0_rows if r.get("learner_adequate_this_seed"))
    learner_adequate = bool(n_l0_adequate >= COMPETENCE_MIN_SEEDS)
    l0_mean_resources = _mean(
        [float(r["mean_resources_per_episode"]) for r in l0_rows]
    )
    sanity_oracle_mean = _mean(
        [float(r["mean_resources_per_episode"]) for r in sanity_oracle_rows]
    )

    # ----- Readiness (now INCLUDES learner adequacy) -----
    b0_stats = per_arm[B0_ARM_ID]
    baseline_reproduces_incompetence = bool(not b0_stats["majority_supra_floor"])
    # Sufficient-P2 is judged over the discriminator arms + L0 (all completed cells).
    all_cells_ok = [c for c in cells if c.get("error_note") is None]
    min_p2 = min([c.get("n_p2_eps_completed", 0) for c in all_cells_ok], default=0)
    sufficient_p2 = bool(
        all_cells_ok
        and all(c.get("n_p2_eps_completed", 0) >= MIN_P2_EPISODES for c in all_cells_ok)
    )
    readiness_met = bool(
        oracle_clears_floor
        and baseline_reproduces_incompetence
        and sufficient_p2
        and learner_adequate
    )

    # ----- Discrimination (B1/B2 only) -----
    b1_clears = bool(per_arm[B1_ARM_ID]["majority_supra_floor"])
    b2_clears = bool(per_arm[B2_ARM_ID]["majority_supra_floor"])

    # Non-degeneracy: real-env oracle must separate from the floor, the sanity learner must
    # be adequate, and the three DISCRIMINATOR arm means must not be bit-identical.
    disc_means = [
        per_arm[a]["mean_resources_per_episode_mean"]
        for a in (B0_ARM_ID, B1_ARM_ID, B2_ARM_ID)
    ]
    arms_not_identical = bool(len(set(round(m, 6) for m in disc_means)) > 1)
    non_degenerate = bool(oracle_clears_floor and learner_adequate and arms_not_identical)
    degeneracy_reason = None
    if not non_degenerate:
        if not oracle_clears_floor:
            degeneracy_reason = "oracle_below_floor"
        elif not learner_adequate:
            degeneracy_reason = "learner_inadequate_on_sanity_env"
        else:
            degeneracy_reason = "all_three_discriminator_arms_identical_resources_per_episode"

    # ----- Self-route (HYPOTHESIS, not a verdict) -----
    if not readiness_met:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
    elif b2_clears and b1_clears:
        outcome = "PASS"
        label = "H1_policy_head_recovers_competence"
    elif b2_clears and not b1_clears:
        outcome = "PASS"
        label = "H1_deeper_ree_representation_obstructs"
    elif (not b2_clears) and b1_clears:
        outcome = "FAIL"
        label = "flag_implausible_leakage_check"
    else:  # not b2_clears and not b1_clears -- learner adequacy already confirmed by readiness
        outcome = "FAIL"
        label = "H2_observation_interface_unlearnable"
    direction = "non_contributory"

    interpretation = {
        "label": label,
        "b1_clears_floor": b1_clears,
        "b2_clears_floor": b2_clears,
        "learner_adequate": learner_adequate,
        "preconditions": [
            {
                "name": "oracle_resource_channel_clears_floor",
                "kind": "readiness",
                "description": (
                    "Greedy nearest-resource ORACLE (no agent) clears the competence floor "
                    "in the REAL env, proving the floor is ACHIEVABLE. SAME statistic as the "
                    "agent DV (env.step info transition_type=='resource'). Below-floor => "
                    "substrate_not_ready_requeue, NEVER an H1/H2 verdict."
                ),
                "control": "greedy nearest-resource oracle forager, same ENV_KWARGS/seed",
                "measured": float(round(oracle_min_resources, 6)),
                "threshold": float(COMPETENCE_RESOURCE_FLOOR),
                "met": bool(oracle_clears_floor),
            },
            {
                "name": "learner_adequacy_on_sanity_env",
                "kind": "readiness",
                "description": (
                    "The SAME vanilla PPO learner (as B2) must clear >= "
                    f"{LEARNER_ADEQUACY_FRAC} of the sanity-env oracle on a majority of seeds "
                    "on a PLAIN foraging env (hazards/reef/contamination OFF) within budget, "
                    "BEFORE a sub-floor B2 is allowed to read as H2. Asserts the SAME statistic "
                    "(mean_resources_per_episode) the H2 branch routes on -- this is the direct "
                    "fix for the V3-EXQ-732 mis-read (under-powered learner mistaken for an "
                    "unlearnable observation). Below => substrate_not_ready_requeue (learner too "
                    "weak), NEVER an H2 verdict. measured = n_seeds L0 adequate; threshold = "
                    "majority seed count."
                ),
                "control": (
                    "L0 vanilla PPO on sanity env vs sanity oracle; adequate if "
                    f"L0 res/ep >= {LEARNER_ADEQUACY_FRAC} x sanity_oracle res/ep (per seed)"
                ),
                "measured": float(n_l0_adequate),
                "threshold": float(COMPETENCE_MIN_SEEDS),
                "met": bool(learner_adequate),
            },
            {
                "name": "baseline_reproduces_incompetence",
                "kind": "readiness",
                "description": (
                    "B0 (724 A0 bias-head REINFORCE all-ON) must forage BELOW the floor on a "
                    "majority of seeds -- i.e. the 724 incompetence must reproduce -- for the "
                    "H1/H2 discrimination to be meaningful. If B0 already clears the floor the "
                    "premise is not reproduced => substrate_not_ready_requeue."
                ),
                "control": "B0 mean_resources/ep vs floor (majority of seeds)",
                "measured": float(b0_stats["mean_resources_per_episode_mean"]),
                "threshold": float(COMPETENCE_RESOURCE_FLOOR),
                "direction": "upper",
                "met": bool(baseline_reproduces_incompetence),
            },
            {
                "name": "sufficient_p2_episodes_all_cells",
                "kind": "readiness",
                "description": (
                    "Every completed cell must log >= MIN_P2_EPISODES P2 eval episodes so "
                    "mean_resources_per_episode is estimable. Below => "
                    "substrate_not_ready_requeue."
                ),
                "control": "min completed P2 episodes across all cells",
                "measured": float(min_p2),
                "threshold": float(MIN_P2_EPISODES),
                "met": bool(sufficient_p2),
            },
        ],
        "criteria": [
            {
                "name": "discriminator_resolved_nondegenerate",
                "load_bearing": True,
                "passed": bool(readiness_met and non_degenerate),
            },
        ],
        "criteria_non_degenerate": {
            "oracle_clears_floor": bool(oracle_clears_floor),
            "learner_adequate_on_sanity": bool(learner_adequate),
            "baseline_reproduces_incompetence": bool(baseline_reproduces_incompetence),
            "sufficient_p2_episodes": bool(sufficient_p2),
            "discriminator_arms_not_identical": bool(arms_not_identical),
        },
    }

    hypotheses = {
        "H1": (
            "The competence deficit is in REE's ACTION-GENERATION stack (bias-head-only "
            "REINFORCE over prediction-only representations). A real POWERED trainable action "
            "head over REE's representation recovers foraging."
        ),
        "H2": (
            "The observation / env interface is unlearnable by ANY policy at this scale; the "
            "oracle only wins on privileged nearest-resource access. The target is the "
            "observation encoding, not the policy. (Only credible once the learner-adequacy "
            "gate rules out 'learner too weak'.)"
        ),
    }

    interpretation_grid = {
        "H1_policy_head_recovers_competence": (
            "readiness holds (incl. learner adequate) AND B2 (vanilla PPO on raw obs) clears "
            "the floor AND B1 (PPO head over REE representation) clears the floor. Bias-head-"
            "only was the bottleneck; a real powered action head over REE's representation "
            "forages. HYPOTHESIS: route to /implement-substrate to build a proper action/"
            "policy-learning substrate under f_dominance_conversion_ceiling (MECH-455 IM)."
        ),
        "H1_deeper_ree_representation_obstructs": (
            "readiness holds AND B2 clears the floor AND B1 does NOT. A vanilla policy learns "
            "the raw observation but the SAME policy over REE's representation cannot -> the "
            "deficit is in REE's representation processing (drives / world-model / encoder), "
            "not just the head. HYPOTHESIS: narrow the build to the representation."
        ),
        "H2_observation_interface_unlearnable": (
            "readiness holds -- CRUCIALLY the learner-adequacy gate PASSED (the SAME vanilla "
            "PPO clears the sanity env) -- AND B2 does NOT clear the floor. A POWERED vanilla "
            "policy on the identical observation cannot forage the real env though it forages "
            "the sanity env -> target the OBSERVATION ENCODING, not the policy. HYPOTHESIS: "
            "route to /implement-substrate on the observation encoding. (This is the branch "
            "V3-EXQ-732 could not license -- 732 had no learner-adequacy gate.)"
        ),
        "flag_implausible_leakage_check": (
            "readiness holds AND B1 clears the floor AND B2 does NOT. A REE-encoded head "
            "beating a vanilla policy on observation derived from the SAME env information is "
            "implausible -> check for leakage / privileged input before trusting either arm."
        ),
        "substrate_not_ready_requeue": (
            "the real-env greedy oracle cannot clear the floor (env does not permit it), OR "
            "the LEARNER-ADEQUACY gate failed (the vanilla PPO cannot even forage the sanity "
            "env -> learner too weak, NOT an unlearnable observation), OR B0 already forages "
            ">= floor (724 premise not reproduced), OR a cell logged fewer than MIN_P2_EPISODES "
            "eval episodes. NOT a verdict -- re-examine learner/env/floor/budget and re-queue. "
            "Draw NO conclusion about H1 vs H2."
        ),
    }

    return {
        "outcome": outcome,
        "overall_direction": direction,
        "interpretation_label": label,
        "interpretation": interpretation,
        "hypotheses": hypotheses,
        "interpretation_grid": interpretation_grid,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "seeds": seeds,
        "p0_episodes": int(p0_episodes),
        "p1_b0_episodes": int(p1_b0_episodes),
        "p1_ppo_episodes": int(p1_ppo_episodes),
        "p2_episodes": int(p2_episodes),
        "steps_per_episode": int(steps_per_episode),
        "oracle_episodes": int(oracle_episodes),
        "rollout_episodes": int(rollout_episodes),
        "decision_rule_thresholds": {
            "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
            "competence_min_seeds": int(COMPETENCE_MIN_SEEDS),
            "min_p2_episodes": int(MIN_P2_EPISODES),
            "learner_adequacy_frac": float(LEARNER_ADEQUACY_FRAC),
            "b0_arm_id": B0_ARM_ID,
            "b1_arm_id": B1_ARM_ID,
            "b2_arm_id": B2_ARM_ID,
            "l0_arm_id": L0_ARM_ID,
        },
        "readiness_gates": {
            "oracle_clears_floor": oracle_clears_floor,
            "oracle_mean_resources_per_episode": round(oracle_mean_resources, 6),
            "oracle_min_resources_per_episode": round(oracle_min_resources, 6),
            "learner_adequate": learner_adequate,
            "n_l0_adequate_seeds": int(n_l0_adequate),
            "l0_mean_resources_per_episode": round(l0_mean_resources, 6),
            "sanity_oracle_mean_resources_per_episode": round(sanity_oracle_mean, 6),
            "baseline_reproduces_incompetence": baseline_reproduces_incompetence,
            "baseline_mean_resources_per_episode": b0_stats[
                "mean_resources_per_episode_mean"
            ],
            "sufficient_p2_episodes": sufficient_p2,
            "readiness_met": readiness_met,
        },
        "discrimination_gates": {
            "b1_clears_floor": b1_clears,
            "b2_clears_floor": b2_clears,
        },
        "oracle_results": oracle_rows,
        "sanity_oracle_results": sanity_oracle_rows,
        "per_arm": per_arm,
        "arm_results": cells,
    }


def _build_manifest(
    result: Dict[str, Any],
    timestamp_utc: str,
    dry_run: bool,
) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    rg = result["readiness_gates"]
    dg = result["discrimination_gates"]
    return {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "supersedes": "V3-EXQ-732",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp_utc,
        "outcome": result["outcome"],
        "evidence_direction": result["overall_direction"],
        "interpretation_label": result["interpretation_label"],
        "interpretation": result["interpretation"],
        "hypotheses": result["hypotheses"],
        "interpretation_grid": result["interpretation_grid"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "sleep_driver_pattern": "none",
        "evidence_direction_note": (
            f"V3-EXQ-732a POWER-FIXED POLICY-LEARNING DISCRIMINATOR (experiment_purpose="
            f"diagnostic, claim_ids=[], non_contributory -- EXCLUDED from governance scoring; "
            f"PROMOTES / DEMOTES NOTHING; supersedes V3-EXQ-732). Re-runs the H1/H2 fork left "
            f"open by failure_autopsy_V3-EXQ-724 with the four power-fixes prescribed by "
            f"failure_autopsy_V3-EXQ-732_2026-07-10 (which REJECTED 732's H2 self-route as "
            f"under-powered): (1) PPO minibatched off-policy learner at {result['p1_ppo_episodes']} "
            f"episodes (~10x env-step, ~30x update budget) on BOTH B1 and B2; (2) entropy "
            f"{PPO_ENTROPY_BETA} + count-based novelty; (3) explicit forage-bonus + running-std "
            f"reward scaling + advantage normalization; (4) a LEARNER-ADEQUACY gate (L0 vanilla "
            f"PPO on a plain sanity env vs its oracle) that must PASS before any sub-floor B2 "
            f"reads as H2. Real-env oracle_min/ep={rg['oracle_min_resources_per_episode']} "
            f"clears_floor={rg['oracle_clears_floor']}; learner_adequate={rg['learner_adequate']} "
            f"(n_l0_adequate={rg['n_l0_adequate_seeds']}, l0/ep={rg['l0_mean_resources_per_episode']} "
            f"vs sanity_oracle/ep={rg['sanity_oracle_mean_resources_per_episode']}). Load-bearing "
            f"DV: P2 mean_resources_per_episode (greedy eval, real resource transitions; training "
            f"reward shaping does NOT touch it). Self-route (HYPOTHESIS, not a verdict): "
            f"readiness_met={rg['readiness_met']}; b1_clears={dg['b1_clears_floor']}, "
            f"b2_clears={dg['b2_clears_floor']} -> label={result['interpretation_label']}. Feeds "
            f"the WS-1 build-direction decision in ree_ai_design_critique_plan.md (B2 clears -> "
            f"H1 action/policy substrate; B2 sub-floor WITH adequacy gate passed -> H2 "
            f"observation-encoding build). NO observation-encoding substrate build until this run "
            f"confirms H2. Route to /failure-autopsy for adjudication before any governance action "
            f"or build."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(x724.ENV_KWARGS),
        "sanity_env_kwargs": dict(SANITY_ENV_KWARGS),
        "config_summary": {
            "design": (
                "power-fixed H1/H2 policy-learning discriminator; 3 discriminator arms + L0 "
                "learner-adequacy arm x seeds + real-env & sanity-env greedy oracles; env / "
                "oracle / B0 reused verbatim from V3-EXQ-724"
            ),
            "arms": {
                B0_ARM_ID: (
                    "H0 anchor = 724 A0: all-ON REE, P0 e2 warmup, P1 two-head REINFORCE "
                    "(lateral-PFC bias + OFC devaluation), encoder frozen in P1"
                ),
                B1_ARM_ID: (
                    "H1: REE all-ON encoder prediction-warmed in P0 then FROZEN; a full "
                    "trainable trunk+PPO head (own optimizer) over z_self(+)z_world learns "
                    "the action"
                ),
                B2_ARM_ID: (
                    "H2: vanilla PPO, matched trunk+head, over the raw observation vector "
                    "(body_state(+)world_state(+)harm_obs(+)harm_obs_a(+)harm_history); no "
                    "REE machinery"
                ),
                L0_ARM_ID: (
                    "LEARNER-ADEQUACY GATE: the SAME vanilla PPO as B2 on a plain sanity env "
                    "(hazards/reef/contamination OFF); must clear >= "
                    f"{LEARNER_ADEQUACY_FRAC} of the sanity oracle before a B2 sub-floor reads "
                    "as H2"
                ),
            },
            "matched_learner": (
                f"B1, B2 and L0 share online PPO (clip {PPO_CLIP}, GAE lambda {PPO_GAE_LAMBDA}, "
                f"{PPO_EPOCHS} epochs, minibatch {PPO_MINIBATCH_SIZE}, entropy {PPO_ENTROPY_BETA}) "
                f"with a {result['p1_ppo_episodes']}-episode budget; only the representation "
                f"front-end differs between B1 (REE latent) and B2 (raw obs). B0 budget fixed by "
                f"faithfulness to 724 A0."
            ),
            "return_normalization": (
                "training reward = env harm_signal + FORAGE_BONUS per resource + count-based "
                "novelty bonus, then running-std reward scaling (sign-preserving) + per-batch "
                "advantage normalization. The greedy P2 DV counts real resource transitions and "
                "is NOT shaped."
            ),
            "load_bearing_dv": (
                "P2 mean_resources_per_episode (env.step info transition_type=='resource'), "
                "majority of seeds vs 1.0 floor"
            ),
            "positive_controls": (
                "greedy nearest-resource oracle (724 _oracle_action) on BOTH the real env "
                "(floor achievability) and the sanity env (L0 adequacy reference)"
            ),
            "ppo_hyperparameters": {
                "trunk_hidden": PPO_TRUNK_HIDDEN, "lr": PPO_LR, "gamma": PPO_GAMMA,
                "gae_lambda": PPO_GAE_LAMBDA, "clip": PPO_CLIP,
                "entropy_beta": PPO_ENTROPY_BETA, "value_coef": PPO_VALUE_COEF,
                "grad_clip": PPO_GRAD_CLIP, "epochs": PPO_EPOCHS,
                "minibatch": PPO_MINIBATCH_SIZE,
                "rollout_episodes": result["rollout_episodes"],
                "p1_ppo_episodes": result["p1_ppo_episodes"],
                "forage_bonus": FORAGE_BONUS, "novelty_coef": NOVELTY_COEF,
                "learner_adequacy_frac": LEARNER_ADEQUACY_FRAC,
            },
            "use_sleep_loop": False,
        },
        "result": result,
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description=(
            "V3-EXQ-732a power-fixed policy-learning DIAGNOSTIC (H1 REE action stack vs H2 "
            "observation interface; PPO + learner-adequacy gate; claim_ids=[])"
        )
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    _run_started = datetime.now(timezone.utc)

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        p0 = DRY_RUN_P0
        p1_b0 = DRY_RUN_P1_B0
        p1_ppo = DRY_RUN_P1_PPO
        p2 = DRY_RUN_P2
        steps = DRY_RUN_STEPS
        oracle_eps = DRY_RUN_ORACLE_EPS
        rollout = DRY_RUN_ROLLOUT_EPISODES
    else:
        seeds = list(SEEDS)
        p0 = P0_WARMUP_EPISODES
        p1_b0 = P1_B0_REINFORCE
        p1_ppo = P1_PPO_EPISODES
        p2 = P2_EVAL_EPISODES
        steps = STEPS_PER_EPISODE
        oracle_eps = N_ORACLE_EPISODES
        rollout = PPO_ROLLOUT_EPISODES

    result = run_experiment(
        seeds=seeds,
        p0_episodes=p0,
        p1_b0_episodes=p1_b0,
        p1_ppo_episodes=p1_ppo,
        p2_episodes=p2,
        steps_per_episode=steps,
        oracle_episodes=oracle_eps,
        rollout_episodes=rollout,
        dry_run=bool(args.dry_run),
    )

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run))

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=args.dry_run,
        config=manifest.get("config") or manifest.get("config_summary"),
        seeds=SEEDS,
        script_path=Path(__file__),
        elapsed_seconds=(datetime.now(timezone.utc) - _run_started).total_seconds(),
    )

    print(f"manifest: {out_path}", flush=True)
    if not args.dry_run:
        print(f"Result written to: {out_path}", flush=True)
    rg = result["readiness_gates"]
    dg = result["discrimination_gates"]
    print(
        f"outcome: {result['outcome']} label={result['interpretation_label']} "
        f"readiness={rg['readiness_met']} "
        f"oracle_min/ep={rg['oracle_min_resources_per_episode']} "
        f"learner_adequate={rg['learner_adequate']} "
        f"b0/ep={rg['baseline_mean_resources_per_episode']} "
        f"b1_clears={dg['b1_clears_floor']} b2_clears={dg['b2_clears_floor']}",
        flush=True,
    )
    for aid, st in result["per_arm"].items():
        print(
            f"  ARM {aid}: resources/ep_mean={st['mean_resources_per_episode_mean']} "
            f"supra_seeds={st['n_seeds_supra_floor']}/{st['n_seeds_ok']} "
            f"majority_supra={st['majority_supra_floor']}",
            flush=True,
        )

    if args.dry_run:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass

    outcome_norm = result["outcome"].upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    manifest_for_sentinel = str(out_path) if not args.dry_run else None
    return outcome_emit, manifest_for_sentinel, bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry_run = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path, dry_run=_dry_run)
    sys.exit(0)
