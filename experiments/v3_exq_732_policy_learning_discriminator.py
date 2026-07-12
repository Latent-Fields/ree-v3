#!/opt/local/bin/python3
"""
V3-EXQ-732 -- POLICY-LEARNING DISCRIMINATOR (H1 REE action-stack vs H2 observation interface).

SLEEP DRIVER: none (no sleep loop; use_sleep_loop / sws_enabled / rem_enabled all OFF).

WHY THIS EXISTS (a DIAGNOSTIC, not an evidence falsifier; the self-routed label below is a
HYPOTHESIS the adjudication pipeline can falsify, NOT a governance verdict). The failure
autopsy failure_autopsy_V3-EXQ-724_2026-07-09 adjudicated the V3-EXQ-724 competence-
localization diagnostic to `competence_deficit_diffuse`: no single OFAT factor (P1 budget /
encoder-freeze / mechanism-count / their combination) recovered foraging in the integrated
all-ON REE-v3 agent -- every arm stayed below the 1.0 resources/episode floor while a hand-
coded greedy oracle cleared it at 6.05. The autopsy localized the deficit to the ONE
invariant 724 never varied: the policy is learned ONLY through a thin bias_head REINFORCE
over representations trained for PREDICTION (SD-056 e2 world-forward contrastive), not for
ACTION. REE is prediction-rich, action-poor.

That leaves an open fork the autopsy (section 6/7) could not resolve, because 724 had no
LEARNED non-REE control and never varied the policy-learning mechanism:

  * H1 -- the deficit is in REE's ACTION-GENERATION stack (bias-head policy over prediction-
    only representations; possibly also the surrounding representation machinery). A real
    trainable action head would recover competence.
  * H2 -- the ENV / OBSERVATION INTERFACE is unlearnable by ANY policy at this scale (the
    oracle only wins on PRIVILEGED nearest-resource access the agent's observation may not
    afford). No policy substrate would help; the target is the observation encoding.

This experiment discriminates H1 vs H2 BEFORE any /implement-substrate build commits to a
target. It does NOT test or weight any claim (claim_ids=[], experiment_purpose=diagnostic,
non_contributory) -- it decides which substrate the follow-on f_dominance_conversion_ceiling
build should target (a policy/action-learning substrate vs the observation encoding).

BRAKE-EXEMPT. This asks a DIFFERENT question ("is REE's action stack or the observation
interface the competence bottleneck") than the conversion-ceiling claims ARC-062 / MECH-309
("does the agent's committed action collapse to a single class"). It tags NO claim, so the
/failure-autopsy re-derive brake does not apply.

------------------------------------------------------------------------------------------
DESIGN -- three arms + greedy-oracle positive control, IDENTICAL env to V3-EXQ-724.

Env, seeds (42/43/44), steps/episode, P2 eval protocol, competence floor (1.0 res/ep),
readiness gates, and the greedy nearest-resource ORACLE positive control are ALL reused
verbatim from V3-EXQ-724 (imported, not copied), so the DV statistic (P2 mean resources/
episode via env.step info transition_type=='resource') is bit-identical across the two runs.

  B0  ree_biashead_reinforce_allon  -- H0 INCOMPETENCE ANCHOR. Exactly V3-EXQ-724 arm A0:
        all-ON REE stack, P0=200 SD-056 e2 prediction warmup, P1=90 TWO-head REINFORCE
        (lateral-PFC bias head + OFC devaluation head) with the encoder FROZEN through P1,
        P2 frozen eval. Computed by calling the 724 harness `_run_cell` on the A0 arm dict,
        so B0 reproduces the 0.25-res/ep incompetence by construction.

  B1  ree_repr_full_a2c_head  -- H1 TEST (REE action stack). REE's prediction-trained
        representation given a REAL action-learning head instead of B0's thin bias head:
          - P0 = 200 episodes SD-056 e2 world-forward contrastive warmup over the all-ON
            REE encoder (random-action transitions) -- the SAME prediction-training that
            makes REE "prediction-rich". Encoder is then FROZEN (phased training).
          - P1 = 200 episodes online advantage-actor-critic (A2C). Each tick the frozen
            all-ON REE encoder produces the latent (z_self (+) z_world, no_grad); a
            FULL-CAPACITY trainable trunk+A2C head (own Adam optimizer) maps that latent to
            an action and is trained by A2C. This is the "full trainable policy/action head
            (not bias-head-only)" the autopsy prescribes -- action-specific representation
            adaptation lives in the trainable trunk over REE's representation.
          - P2 = 20 episodes frozen greedy eval.

  B2  vanilla_a2c_rawobs  -- H2 TEST (observation interface). A NON-REE vanilla learned
        policy on the IDENTICAL observation vector, no REE machinery:
          - P1 = 200 episodes online A2C over a matched-capacity trainable trunk+A2C head
            reading the raw observation vector (body_state (+) world_state (+) harm_obs (+)
            harm_obs_a (+) harm_history) -- the SAME raw channels B1's REE encoder senses.
          - P2 = 20 episodes frozen greedy eval.

MATCHED LEARNER (design choice, stated for the record). B1 and B2 use the SAME online A2C
learner with the SAME hyperparameters and the SAME 200-episode training budget; the ONLY
difference between them is the representation front-end (REE prediction-trained latent vs
raw observation). The autopsy sketch suggested "actor-critic for B1, PPO/DQN for B2"; a
single shared learner is used INSTEAD so the B1-vs-B2 contrast isolates the representation
front-end with NO learning-algorithm confound. A2C is a standard vanilla policy-gradient
learner, so B2 remains a legitimate non-REE control. Its exploration is entropy-regularised
(no replay buffer), which is what lets B1 hold a frozen prediction-trained encoder cleanly
(phased training) rather than fighting replay-staleness against an adapting encoder. B0's
budget is fixed by faithfulness to 724 A0 (the known-incompetent anchor), not matched.

POSITIVE CONTROL (readiness). The greedy nearest-resource ORACLE (V3-EXQ-724 `_run_oracle`,
no agent) is run per seed on the SAME ENV_KWARGS/seed and its mean resources/episode is
measured with the SAME statistic as the agent DV. It proves the 1.0 floor is ACHIEVABLE by
a resource-seeking policy in this exact env. If even the oracle cannot clear the floor, NO
architecture conclusion is licensed -> substrate_not_ready_requeue.

DV (load-bearing): P2 mean_resources_per_episode (majority of seeds vs the 1.0 floor).

PRE-REGISTERED SELF-ROUTE (HYPOTHESIS, not a verdict):
  * READINESS fails (oracle below floor OR B0 does NOT reproduce incompetence OR any cell
    logs < MIN_P2_EPISODES) -> `substrate_not_ready_requeue`.
  * B2 clears AND B1 clears -> `H1_policy_head_recovers_competence`: bias-head-only was the
    bottleneck; a real action head over REE's representation forages. Route to
    /implement-substrate to build a proper action/policy-learning substrate under
    f_dominance_conversion_ceiling.
  * B2 clears AND B1 does NOT clear -> `H1_deeper_ree_representation_obstructs`: a vanilla
    policy learns the raw observation but the SAME policy over REE's representation cannot ->
    the deficit is in REE's representation processing, not just the head. Route to a build
    that narrows to the representation / drives / world-model, NOT the head alone.
  * B2 does NOT clear -> `H2_observation_interface_unlearnable`: even a vanilla policy on the
    identical observation cannot forage at this budget -> target the OBSERVATION ENCODING,
    not the policy. (A policy-learning build would be wasted. Prudent follow-up before a
    large obs rebuild: confirm with a stronger learner / longer budget, since "sub-floor
    under vanilla A2C at 200 episodes" is a strong hint, not a proof of unlearnability.)
  * B1 clears AND B2 does NOT clear -> `flag_implausible_leakage_check`: a REE-encoded head
    beating a vanilla policy on observation derived from the SAME env information is
    implausible -> check for leakage / privileged input before trusting either arm.

EVIDENCE-FOR / EVIDENCE-AGAINST (diagnostic-description requirement), per arm:
  * B0 (anchor): EVIDENCE-FOR reproducing 724's incompetence (below floor) is the premise
    that licenses the localization; B0 clearing the floor would be EVIDENCE-AGAINST the
    premise (719a/724 not reproduced) -> substrate_not_ready_requeue.
  * B1 (H1): B1 clearing the floor is EVIDENCE that REE's representation is usable for
    action with a real head (deficit is the action head / policy-learning mechanism). B1
    below floor is EVIDENCE-AGAINST "the head alone recovers competence" (points deeper into
    the REE representation, conditional on B2).
  * B2 (H2): B2 clearing the floor is EVIDENCE-AGAINST H2 (the observation IS learnable) and
    FOR H1. B2 below floor is EVIDENCE-FOR H2 (observation interface is the bottleneck).
  The self-routed label is a HYPOTHESIS for /failure-autopsy adjudication; this experiment
  PROMOTES / DEMOTES NOTHING and tags NO claim.

Sourced substrate: experiments/v3_exq_724_competence_localization_diagnostic.py (env, oracle,
all-ON stack, B0 harness), ree_core/agent.py (REEAgent.sense -> latent.z_self/z_world),
ree_core/environment/causal_grid_world.py (step() -> harm_signal reward, info.transition_type,
obs_dict channels). See REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-724_2026-07-09.md
and REE_assembly/evidence/planning/ree_ai_design_critique_plan.md (WS-1).
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
# obs helpers, and all shared constants) so B0 and the oracle are bit-identical to 724.
import experiments.v3_exq_724_competence_localization_diagnostic as x724
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_732_policy_learning_discriminator"
QUEUE_ID = "V3-EXQ-732"
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
# A2C learner (matched across B1 and B2)
# ---------------------------------------------------------------------------
P1_A2C_EPISODES = 200            # online A2C training episodes (B1 and B2, matched budget)
A2C_TRUNK_HIDDEN = 128           # 2-layer trainable trunk width (both arms)
A2C_LR = 1e-3
A2C_GAMMA = 0.99
A2C_ENTROPY_BETA = 0.01
A2C_VALUE_COEF = 0.5
A2C_GRAD_CLIP = 1.0

# SD-056 e2 prediction-warmup levers (reuse 724 constants)
E2_CONTRASTIVE_LR = x724.E2_CONTRASTIVE_LR
E2_TRAIN_EVERY_K_TICKS = x724.E2_TRAIN_EVERY_K_TICKS
TRANSITION_BUFFER_MAX = x724.TRANSITION_BUFFER_MAX

# ---------------------------------------------------------------------------
# Dry-run budget (tiny; smoke stays fast)
# ---------------------------------------------------------------------------
DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1_B0 = 2
DRY_RUN_P1_A2C = 2
DRY_RUN_P2 = 2
DRY_RUN_STEPS = 15
DRY_RUN_ORACLE_EPS = 2

# Arm ids / roles
B0_ARM_ID = "B0_ree_biashead_reinforce_allon"
B1_ARM_ID = "B1_ree_repr_full_a2c_head"
B2_ARM_ID = "B2_vanilla_a2c_rawobs"


# ---------------------------------------------------------------------------
# A2C policy/value network (shared by B1 and B2).
#   B1: in_dim = self_dim + world_dim (REE latent), trunk trainable, encoder frozen.
#   B2: in_dim = raw observation width, trunk trainable, no encoder.
# ---------------------------------------------------------------------------
class A2CPolicy(nn.Module):
    def __init__(self, in_dim: int, action_dim: int, hidden: int = A2C_TRUNK_HIDDEN):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(state)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value


def _a2c_update(
    policy: A2CPolicy,
    optimiser: torch.optim.Optimizer,
    log_probs: List[torch.Tensor],
    values: List[torch.Tensor],
    entropies: List[torch.Tensor],
    rewards: List[float],
    device: torch.device,
) -> None:
    """One end-of-episode A2C update (Monte-Carlo discounted returns baseline)."""
    if not rewards:
        return
    returns: List[float] = []
    g = 0.0
    for r in reversed(rewards):
        g = float(r) + A2C_GAMMA * g
        returns.insert(0, g)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
    values_t = torch.stack(values)
    log_probs_t = torch.stack(log_probs)
    entropies_t = torch.stack(entropies)
    advantages = (returns_t - values_t).detach()
    policy_loss = -(log_probs_t * advantages).mean()
    value_loss = 0.5 * (returns_t - values_t).pow(2).mean()
    entropy_bonus = entropies_t.mean()
    loss = policy_loss + A2C_VALUE_COEF * value_loss - A2C_ENTROPY_BETA * entropy_bonus
    if not torch.isfinite(loss):
        return
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), A2C_GRAD_CLIP)
    optimiser.step()


# ---------------------------------------------------------------------------
# Raw observation vector for B2 (identical channels the REE encoder senses).
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
# P2 frozen greedy eval shared by B1 / B2 (SAME resource statistic as 724).
# ---------------------------------------------------------------------------
def _eval_policy(
    env,
    state_fn: Callable[[Dict[str, Any]], torch.Tensor],
    policy: A2CPolicy,
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


def _a2c_train_episode(
    env,
    state_fn: Callable[[Dict[str, Any]], torch.Tensor],
    policy: A2CPolicy,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    steps_per_episode: int,
    on_episode_reset: Optional[Callable[[], None]] = None,
) -> None:
    _, obs_dict = env.reset()
    if on_episode_reset is not None:
        on_episode_reset()
    log_probs: List[torch.Tensor] = []
    values: List[torch.Tensor] = []
    entropies: List[torch.Tensor] = []
    rewards: List[float] = []
    for _step in range(steps_per_episode):
        state = state_fn(obs_dict)
        logits, value = policy(state)
        dist = torch.distributions.Categorical(logits=logits.reshape(1, -1))
        a = dist.sample()
        a_idx = int(a.item())
        log_probs.append(dist.log_prob(a).reshape(()))
        values.append(value.reshape(()))
        entropies.append(dist.entropy().reshape(()))
        _, harm_signal, done, info, obs_dict = env.step(a_idx)
        rewards.append(float(harm_signal))
        if done:
            break
    _a2c_update(policy, optimiser, log_probs, values, entropies, rewards, device)


# ---------------------------------------------------------------------------
# B1 cell: REE prediction warmup (encoder + e2) -> frozen -> A2C head over latent.
# ---------------------------------------------------------------------------
def _run_b1_cell(
    seed: int,
    p0_episodes: int,
    p1_a2c_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    device = torch.device("cpu")
    env = x724._make_env(seed)
    agent = _make_all_on_agent(env)

    # ---- P0: SD-056 e2 world-forward prediction warmup (random-action transitions) ----
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)
    transition_buffer: Deque[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)
    n_p0_ticks = 0
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
            n_p0_ticks += 1
            tick_in_ep += 1
            if done:
                break
        cur = ep + 1
        total = p0_episodes + p1_a2c_episodes + p2_episodes
        if cur % 25 == 0 or cur == p0_episodes:
            print(
                f"  [train] disc arm={B1_ARM_ID} seed={seed} phase=P0 "
                f"ep {cur}/{total}", flush=True,
            )

    # ---- Freeze encoder; A2C head over the frozen REE latent ----
    def _state_fn(obs_dict: Dict[str, Any]) -> torch.Tensor:
        return _b1_latent_state(agent, obs_dict, device)

    # Derive the latent feature width from an actual encoder pass (REEConfig does not
    # expose self_dim/world_dim as attributes; z_self (+) z_world = 64 here).
    _, probe_obs = env.reset()
    agent.reset()
    in_dim = int(_state_fn(probe_obs).shape[-1])
    policy = A2CPolicy(in_dim=in_dim, action_dim=env.action_dim).to(device)
    optimiser = torch.optim.Adam(policy.parameters(), lr=A2C_LR)

    total = p0_episodes + p1_a2c_episodes + p2_episodes
    for ep in range(p1_a2c_episodes):
        _a2c_train_episode(
            env, _state_fn, policy, optimiser, device, steps_per_episode,
            on_episode_reset=agent.reset,
        )
        cur = p0_episodes + ep + 1
        if (ep + 1) % 25 == 0 or (ep + 1) == p1_a2c_episodes:
            print(
                f"  [train] disc arm={B1_ARM_ID} seed={seed} phase=P1 "
                f"ep {cur}/{total}", flush=True,
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
        p0=p0_episodes, p1=p1_a2c_episodes, p2=p2_episodes, ev=ev, error_note=None,
    )


# ---------------------------------------------------------------------------
# B2 cell: vanilla A2C over the raw observation vector (no REE machinery).
# ---------------------------------------------------------------------------
def _run_b2_cell(
    seed: int,
    p1_a2c_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    device = torch.device("cpu")
    env = x724._make_env(seed)
    _, obs0 = env.reset()
    in_dim = _raw_obs_dim(obs0)
    policy = A2CPolicy(in_dim=in_dim, action_dim=env.action_dim).to(device)
    optimiser = torch.optim.Adam(policy.parameters(), lr=A2C_LR)

    def _state_fn(obs_dict: Dict[str, Any]) -> torch.Tensor:
        return _raw_obs_vector(obs_dict, device)

    total = p1_a2c_episodes + p2_episodes
    for ep in range(p1_a2c_episodes):
        _a2c_train_episode(
            env, _state_fn, policy, optimiser, device, steps_per_episode
        )
        cur = ep + 1
        if (ep + 1) % 25 == 0 or (ep + 1) == p1_a2c_episodes:
            print(
                f"  [train] disc arm={B2_ARM_ID} seed={seed} phase=P1 "
                f"ep {cur}/{total}", flush=True,
            )

    ev = _eval_policy(env, _state_fn, policy, p2_episodes, steps_per_episode)
    print(
        f"  [train] disc arm={B2_ARM_ID} seed={seed} phase=P2 ep {total}/{total}",
        flush=True,
    )
    return _finalize_cell_row(
        arm_id=B2_ARM_ID, role="h2_observation_interface", seed=seed,
        p0=0, p1=p1_a2c_episodes, p2=p2_episodes, ev=ev, error_note=None,
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
    # 724's row already carries mean_resources_per_episode / competence_supra_floor /
    # n_p2_eps_completed / error_note in the SAME schema _finalize_cell_row produces.
    row["arm_role"] = "h0_incompetence_anchor"
    return row


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
    p1_a2c_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
    oracle_episodes: int,
    dry_run: bool,
) -> Dict[str, Any]:
    print(
        f"Policy-learning discriminator (3 arms x {len(seeds)} seeds + greedy-oracle "
        f"positive control; P0={p0_episodes}, P1_B0={p1_b0_episodes}, "
        f"P1_A2C={p1_a2c_episodes}, P2={p2_episodes}, steps={steps_per_episode}, "
        f"oracle_eps={oracle_episodes}, dry_run={dry_run})",
        flush=True,
    )

    # ----- Positive-control oracle (per seed; reuse 724 _run_oracle) -----
    oracle_rows: List[Dict[str, Any]] = []
    for s in seeds:
        orow = x724._run_oracle(s, oracle_episodes, steps_per_episode)
        oracle_rows.append(orow)
        print(
            f"  [oracle] seed={s} mean_resources/ep="
            f"{orow['mean_resources_per_episode']} "
            f"max={orow['max_resources_in_episode']}",
            flush=True,
        )
    oracle_mean_resources = _mean([o["mean_resources_per_episode"] for o in oracle_rows])
    oracle_min_resources = min(
        [o["mean_resources_per_episode"] for o in oracle_rows], default=0.0
    )
    oracle_clears_floor = bool(oracle_min_resources >= COMPETENCE_RESOURCE_FLOOR)

    # ----- Arm x seed cells -----
    arm_specs = [
        (B0_ARM_ID, "h0_incompetence_anchor"),
        (B1_ARM_ID, "h1_ree_representation_full_head"),
        (B2_ARM_ID, "h2_observation_interface"),
    ]
    cells: List[Dict[str, Any]] = []
    for arm_id, role in arm_specs:
        for s in seeds:
            print(f"Seed {s} Condition {arm_id}", flush=True)
            slice_cfg = {
                "arm_id": arm_id,
                "role": role,
                "p0_episodes": int(p0_episodes if arm_id != B2_ARM_ID else 0),
                "p1_episodes": int(
                    p1_b0_episodes if arm_id == B0_ARM_ID else p1_a2c_episodes
                ),
                "p2_episodes": int(p2_episodes),
                "steps_per_episode": int(steps_per_episode),
                "a2c": {
                    "hidden": A2C_TRUNK_HIDDEN, "lr": A2C_LR, "gamma": A2C_GAMMA,
                    "entropy_beta": A2C_ENTROPY_BETA, "value_coef": A2C_VALUE_COEF,
                } if arm_id != B0_ARM_ID else None,
                "env_kwargs": dict(x724.ENV_KWARGS),
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
                        s, p0_episodes, p1_a2c_episodes, p2_episodes, steps_per_episode
                    )
                else:
                    row = _run_b2_cell(
                        s, p1_a2c_episodes, p2_episodes, steps_per_episode
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

    # ----- Readiness -----
    b0_stats = per_arm[B0_ARM_ID]
    baseline_reproduces_incompetence = bool(not b0_stats["majority_supra_floor"])
    all_cells_ok = [c for c in cells if c.get("error_note") is None]
    min_p2 = min([c.get("n_p2_eps_completed", 0) for c in all_cells_ok], default=0)
    sufficient_p2 = bool(
        all_cells_ok
        and all(c.get("n_p2_eps_completed", 0) >= MIN_P2_EPISODES for c in all_cells_ok)
    )
    readiness_met = bool(
        oracle_clears_floor and baseline_reproduces_incompetence and sufficient_p2
    )

    # ----- Discrimination -----
    b1_clears = bool(per_arm[B1_ARM_ID]["majority_supra_floor"])
    b2_clears = bool(per_arm[B2_ARM_ID]["majority_supra_floor"])

    # Non-degeneracy: oracle must separate from the floor, and the three arm means
    # must not be bit-identical (a degenerate identical-arm result cannot discriminate).
    arm_means = [
        per_arm[a]["mean_resources_per_episode_mean"]
        for a in (B0_ARM_ID, B1_ARM_ID, B2_ARM_ID)
    ]
    arms_not_identical = bool(len(set(round(m, 6) for m in arm_means)) > 1)
    non_degenerate = bool(oracle_clears_floor and arms_not_identical)
    degeneracy_reason = None
    if not non_degenerate:
        degeneracy_reason = (
            "oracle_below_floor" if not oracle_clears_floor
            else "all_three_arms_identical_resources_per_episode"
        )

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
    else:  # not b2_clears and not b1_clears
        outcome = "FAIL"
        label = "H2_observation_interface_unlearnable"
    direction = "non_contributory"

    interpretation = {
        "label": label,
        "b1_clears_floor": b1_clears,
        "b2_clears_floor": b2_clears,
        "preconditions": [
            {
                "name": "oracle_resource_channel_clears_floor",
                "kind": "readiness",
                "description": (
                    "Greedy nearest-resource ORACLE (no agent) clears the competence floor "
                    "in this exact env, proving the floor is ACHIEVABLE by a resource-seeking "
                    "policy. SAME statistic as the agent DV (env.step info "
                    "transition_type=='resource'). Below-floor => substrate_not_ready_requeue, "
                    "NEVER an H1/H2 verdict."
                ),
                "control": "greedy nearest-resource oracle forager, same ENV_KWARGS/seed",
                "measured": float(round(oracle_min_resources, 6)),
                "threshold": float(COMPETENCE_RESOURCE_FLOOR),
                "met": bool(oracle_clears_floor),
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
            "baseline_reproduces_incompetence": bool(baseline_reproduces_incompetence),
            "sufficient_p2_episodes": bool(sufficient_p2),
            "arms_not_identical": bool(arms_not_identical),
        },
    }

    hypotheses = {
        "H1": (
            "The competence deficit is in REE's ACTION-GENERATION stack (bias-head-only "
            "REINFORCE over prediction-only representations). A real trainable action head "
            "over REE's representation recovers foraging."
        ),
        "H2": (
            "The observation / env interface is unlearnable by ANY policy at this scale; the "
            "oracle only wins on privileged nearest-resource access. The target is the "
            "observation encoding, not the policy."
        ),
    }

    interpretation_grid = {
        "H1_policy_head_recovers_competence": (
            "readiness holds AND B2 (vanilla A2C on raw obs) clears the floor AND B1 (A2C head "
            "over REE representation) clears the floor. Bias-head-only was the bottleneck; a "
            "real action head over REE's representation forages. HYPOTHESIS: route to "
            "/implement-substrate to build a proper action/policy-learning substrate under "
            "f_dominance_conversion_ceiling."
        ),
        "H1_deeper_ree_representation_obstructs": (
            "readiness holds AND B2 clears the floor AND B1 does NOT. A vanilla policy learns "
            "the raw observation but the SAME policy over REE's representation cannot -> the "
            "deficit is in REE's representation processing (drives / world-model / encoder), "
            "not just the head. HYPOTHESIS: narrow the build to the representation, do NOT "
            "build the action head alone."
        ),
        "H2_observation_interface_unlearnable": (
            "readiness holds AND B2 does NOT clear the floor. Even a vanilla policy on the "
            "identical observation cannot forage at this budget -> target the OBSERVATION "
            "ENCODING, not the policy; a policy-learning build would be wasted. HYPOTHESIS "
            "(prudent follow-up before a large obs rebuild: confirm with a stronger learner / "
            "longer budget -- sub-floor under vanilla A2C at 200 episodes is a strong hint, "
            "not a proof of unlearnability)."
        ),
        "flag_implausible_leakage_check": (
            "readiness holds AND B1 clears the floor AND B2 does NOT. A REE-encoded head "
            "beating a vanilla policy on observation derived from the SAME env information is "
            "implausible -> check for leakage / privileged input before trusting either arm."
        ),
        "substrate_not_ready_requeue": (
            "the greedy oracle cannot clear the floor (env does not permit it), OR B0 already "
            "forages >= floor (724 premise not reproduced), OR a cell logged fewer than "
            "MIN_P2_EPISODES eval episodes. NOT a verdict -- re-examine env/floor/budget and "
            "re-queue. Draw NO conclusion about H1 vs H2."
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
        "p1_a2c_episodes": int(p1_a2c_episodes),
        "p2_episodes": int(p2_episodes),
        "steps_per_episode": int(steps_per_episode),
        "oracle_episodes": int(oracle_episodes),
        "decision_rule_thresholds": {
            "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
            "competence_min_seeds": int(COMPETENCE_MIN_SEEDS),
            "min_p2_episodes": int(MIN_P2_EPISODES),
            "b0_arm_id": B0_ARM_ID,
            "b1_arm_id": B1_ARM_ID,
            "b2_arm_id": B2_ARM_ID,
        },
        "readiness_gates": {
            "oracle_clears_floor": oracle_clears_floor,
            "oracle_mean_resources_per_episode": round(oracle_mean_resources, 6),
            "oracle_min_resources_per_episode": round(oracle_min_resources, 6),
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
            f"V3-EXQ-732 POLICY-LEARNING DISCRIMINATOR (experiment_purpose=diagnostic, "
            f"claim_ids=[], non_contributory -- EXCLUDED from governance scoring; PROMOTES / "
            f"DEMOTES NOTHING). Resolves the H1/H2 fork left open by "
            f"failure_autopsy_V3-EXQ-724_2026-07-09 (competence_deficit_diffuse; deficit "
            f"localized to bias-head-only REINFORCE over prediction-only representations). "
            f"Three arms, IDENTICAL env/seeds/protocol/floor/oracle to 724: B0 = 724 A0 "
            f"incompetence anchor (bias-head REINFORCE all-ON); B1 = REE prediction-trained "
            f"representation + a FULL trainable A2C action head (H1); B2 = vanilla A2C on the "
            f"identical raw observation vector, no REE machinery (H2). B1 and B2 share the "
            f"same A2C learner + budget so the contrast isolates the representation front-end. "
            f"Greedy nearest-resource oracle (positive control) validates the floor is "
            f"achievable (oracle_min/ep={rg['oracle_min_resources_per_episode']}, "
            f"clears_floor={rg['oracle_clears_floor']}). Load-bearing DV: P2 "
            f"mean_resources_per_episode. Self-route (HYPOTHESIS, not a verdict): "
            f"readiness_met={rg['readiness_met']}; b1_clears={dg['b1_clears_floor']}, "
            f"b2_clears={dg['b2_clears_floor']} -> label={result['interpretation_label']}. "
            f"Feeds the WS-1 build-direction decision in "
            f"ree_ai_design_critique_plan.md (H1 -> action/policy substrate; H2 -> observation "
            f"encoding). Route to /failure-autopsy for adjudication before any governance "
            f"action or build."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(x724.ENV_KWARGS),
        "config_summary": {
            "design": (
                "H1/H2 policy-learning discriminator; 3 arms x seeds + greedy-oracle "
                "positive control; env/oracle/B0 reused verbatim from V3-EXQ-724"
            ),
            "arms": {
                B0_ARM_ID: (
                    "H0 anchor = 724 A0: all-ON REE, P0 e2 warmup, P1 two-head REINFORCE "
                    "(lateral-PFC bias + OFC devaluation), encoder frozen in P1"
                ),
                B1_ARM_ID: (
                    "H1: REE all-ON encoder prediction-warmed in P0 then FROZEN; a full "
                    "trainable trunk+A2C head (own optimizer) over z_self(+)z_world learns "
                    "the action"
                ),
                B2_ARM_ID: (
                    "H2: vanilla A2C, matched trunk+head, over the raw observation vector "
                    "(body_state(+)world_state(+)harm_obs(+)harm_obs_a(+)harm_history); no "
                    "REE machinery"
                ),
            },
            "matched_learner": (
                "B1 and B2 share online advantage-actor-critic (A2C) with identical "
                "hyperparameters and a 200-episode budget; only the representation front-end "
                "differs (REE latent vs raw obs). B0 budget fixed by faithfulness to 724 A0."
            ),
            "load_bearing_dv": (
                "P2 mean_resources_per_episode (env.step info transition_type=='resource'), "
                "majority of seeds vs 1.0 floor"
            ),
            "positive_control": "greedy nearest-resource oracle forager (724 _run_oracle)",
            "a2c_hyperparameters": {
                "trunk_hidden": A2C_TRUNK_HIDDEN, "lr": A2C_LR, "gamma": A2C_GAMMA,
                "entropy_beta": A2C_ENTROPY_BETA, "value_coef": A2C_VALUE_COEF,
                "grad_clip": A2C_GRAD_CLIP, "p1_a2c_episodes": result["p1_a2c_episodes"],
            },
            "use_sleep_loop": False,
        },
        "result": result,
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description=(
            "V3-EXQ-732 policy-learning DIAGNOSTIC (H1 REE action stack vs H2 observation "
            "interface; claim_ids=[])"
        )
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        p0 = DRY_RUN_P0
        p1_b0 = DRY_RUN_P1_B0
        p1_a2c = DRY_RUN_P1_A2C
        p2 = DRY_RUN_P2
        steps = DRY_RUN_STEPS
        oracle_eps = DRY_RUN_ORACLE_EPS
    else:
        seeds = list(SEEDS)
        p0 = P0_WARMUP_EPISODES
        p1_b0 = P1_B0_REINFORCE
        p1_a2c = P1_A2C_EPISODES
        p2 = P2_EVAL_EPISODES
        steps = STEPS_PER_EPISODE
        oracle_eps = N_ORACLE_EPISODES

    result = run_experiment(
        seeds=seeds,
        p0_episodes=p0,
        p1_b0_episodes=p1_b0,
        p1_a2c_episodes=p1_a2c,
        p2_episodes=p2,
        steps_per_episode=steps,
        oracle_episodes=oracle_eps,
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
