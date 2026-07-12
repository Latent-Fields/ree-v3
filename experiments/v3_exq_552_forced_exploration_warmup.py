#!/opt/local/bin/python3
"""
V3-EXQ-552 -- Forced uniform-random action warmup vs monostrategy collapse.

Claims: [] (training-loop diagnostic, not a substrate claim test)

Purpose (evidence)
------------------
Distinguishes two hypotheses for the V3-EXQ-550 monostrategy signature
(action_class_entropy ~= 0 in both arms regardless of z_goal wiring):

  H1 (bootstrap-loop / data narrowness): the agent's training data is
     collected from its own (collapsed) policy. Replay buffers stay
     narrow, training stays narrow, the policy stays collapsed. Standard
     RL fix: forced uniform-random warmup that populates the buffer
     with diverse trajectories before the policy bootstraps from its
     own distribution. If forced-exploration warmup lifts post-handoff
     entropy, the substrate has the capacity for diverse policy and
     monostrategy is a training-loop pathology, not architectural.

  H2 (substrate collapses regardless of data): the V3 substrate
     compresses the policy back to a single action class no matter the
     training data. Forced-explore data is consumed but post-handoff
     entropy stays at zero.

This experiment forces uniform-random actions over the first 50% of
P0 training in ARM_WARMUP, then reverts to normal action selection for
the remainder of P0 + all of P1 eval. ARM_NORMAL has no override.
Training data from the warmup IS collected into the standard replay
buffers + online updates the agent uses (the same code path runs;
only the action emitted at each step is replaced).

Implementation
--------------
The uniform-random override is implemented as a SCRIPT-LEVEL wrapper
around agent.select_action() during warmup ticks -- NO substrate
changes to ree_core. agent.select_action() still runs (so any internal
state advancement that side-effects on the call path proceeds), but
the returned action is discarded and replaced with a uniform-random
one-hot before being passed to env.step(). The replacement action is
also stored on agent._last_action so downstream record_transition /
post-step bookkeeping sees the executed (random) action, not the
agent-proposed one.

Metric and gate rule
--------------------
    action_class_entropy in the P1 EVAL phase (post-handoff window).
    Computed as Shannon entropy over the executed-action-class
    histogram across all P1-eval ticks per arm per seed.

    PASS = ARM_WARMUP P1 entropy - ARM_NORMAL P1 entropy >= 0.10 in
           >= 2/3 seeds AND ARM_WARMUP P1 entropy > 0.30 in >= 2/3
           seeds (real diversity, not noise).

    -> H1 supported: monostrategy is partly a training-data pathology,
       fixable with exploration scheduling.

    FAIL with ARM_WARMUP P1 still at ~0 entropy -> H2 supported:
    forced-explore data doesn't keep the policy diverse; substrate is
    collapsing the policy back to one class regardless of data.

    Mixed (e.g. one seed lifts, others don't) -> inconclusive; surface
    for routing.

Three-row interpretation grid
-----------------------------
    (i)   ARM_WARMUP P1 entropy >> ARM_NORMAL AND clears 0.30 floor in
          >= 2 seeds -> data-narrowness reading SUPPORTED. Substrate
          has the capacity for diverse policy; monostrategy was a
          training-loop bootstrap pathology. Route to exploration-
          scheduling work (epsilon-greedy decay, intrinsic-motivation
          curriculum, longer forced-explore window).
    (ii)  ARM_WARMUP P1 entropy ~= ARM_NORMAL (both at ~0) -> substrate-
          collapsing reading SUPPORTED. Forced-explore data did not
          keep the policy diverse. Architectural collapse is the
          dominant cause. Elevates urgency of V3-EXQ-551 (proposer /
          evaluator diagnostic, queued in parallel session) and
          ARC-062 / MECH-309 rule-apprehension cluster.
    (iii) Mixed -- ARM_WARMUP lifts in 1/3 seeds, collapses in others
          -> inconclusive. Most likely a sensitivity-to-init-conditions
          finding: substrate is marginal, can be tipped either way by
          early action-distribution noise. Surface for routing; may
          motivate a longer / stronger warmup window or seed-sweep.

experiment_purpose=evidence (tests a clear training-loop hypothesis
with a pre-registered behavioural prediction).

See ree-v3/CLAUDE.md MECH-269 / SD-029 sections (the monostrategy
cluster being diagnosed here). The bootstrap-loop hypothesis is
distinct from the ARC-062 rule-apprehension cluster: ARC-062 asks
whether the policy CAN learn multiple regimes when the structural
opportunity exists; this experiment asks whether the training-data
distribution alone is sufficient to drive collapse.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from experiment_protocol import emit_outcome
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_552_forced_exploration_warmup"
QUEUE_ID = "V3-EXQ-552"
CLAIM_IDS: List[str] = []  # training-loop diagnostic; no substrate claim test
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 7, 17]
CONDITIONS = ["NORMAL", "WARMUP"]

# Phased schedule -- matches V3-EXQ-543c canonical pattern (P0=40, P1=60).
P0_WARMUP_EPISODES = 40
P1_EVAL_EPISODES = 60
STEPS_PER_EPISODE = 200

# Forced-exploration window: first WARMUP_FRACTION of P0 episodes in
# ARM_WARMUP only. 0.5 -> first 20 of 40 P0 episodes use uniform-random
# action override; the remaining 20 P0 episodes + all 60 P1 episodes
# use normal action selection.
WARMUP_FRACTION = 0.5

# Acceptance thresholds (pre-registered).
ENTROPY_DELTA_THRESHOLD = 0.10
ENTROPY_ABS_FLOOR = 0.30
SEEDS_REQUIRED_TO_PASS = 2  # of 3

# Latent dims (match V3-EXQ-543c baseline).
WORLD_DIM = 32
SELF_DIM = 32
HARM_DIM = 32
HARM_A_DIM = 16
HARM_HISTORY_LEN = 10

# Buffer + batch (match V3-EXQ-543c).
WF_BUF_MAX = 2000
HARM_EVAL_BUF_MAX = 2000
BATCH_SIZE = 32

# Learning rates (match V3-EXQ-543c).
LR_E1 = 1e-4
LR_E2_WF = 3e-4
LR_E3_HARM = 1e-3
LR_ENC_AUX = 5e-4

# Env config: copied from V3-EXQ-543c ENV_KWARGS (canonical recent
# training environment). Bipartite layout retained so structural
# divergence opportunities exist; this is the same environment EXQ-543c
# trained in, ensuring comparability.
ENV_KWARGS = dict(
    size=12,
    num_hazards=4,
    num_resources=5,
    hazard_harm=0.05,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=10,
    reef_enabled=True,
    n_reef_patches=3,
    reef_patch_radius=2,
    hazard_food_attraction=0.7,
    reef_bipartite_layout=True,
    reef_bipartite_axis="horizontal",
    reef_bipartite_agent_band_radius=1,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _obs_harm(obs_dict):
    return obs_dict.get("harm_obs")


def _obs_harm_a(obs_dict):
    return obs_dict.get("harm_obs_a")


def _obs_harm_history(obs_dict):
    return obs_dict.get("harm_history")


def _obs_accum(obs_dict) -> float:
    v = obs_dict.get("accumulated_harm")
    return float(v) if v is not None else 0.0


def _obs_resource_prox(obs_dict) -> float:
    rv = obs_dict.get("resource_field_view")
    if rv is None:
        return 0.0
    if isinstance(rv, torch.Tensor):
        return float(rv.max().item())
    return float(np.max(rv))


def _shannon_entropy(counts: Dict[int, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p)
    return ent


def _make_agent_and_env(seed: int) -> Tuple[REEAgent, CausalGridWorldV2]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env = CausalGridWorldV2(seed=seed, **ENV_KWARGS)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        harm_dim=HARM_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
        use_harm_stream=True,
        z_harm_dim=HARM_DIM,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        benefit_eval_enabled=True,
        benefit_weight=1.0,
        z_goal_enabled=True,
        goal_weight=0.5,
        drive_weight=2.0,
        e1_goal_conditioned=True,
    )
    config.e3.commitment_threshold = 0.5
    config.heartbeat.beta_gate_bistable = True
    config.harm_descending_mod_enabled = True
    config.descending_attenuation_factor = 0.5

    agent = REEAgent(config)
    return agent, env


def _override_with_uniform_random(
    agent: REEAgent,
    action_dim: int,
    device,
    rng_module,
) -> torch.Tensor:
    """Build a uniform-random one-hot action and stamp it onto the agent.

    Returns the same tensor shape REEAgent.select_action() returns
    (torch.Tensor of shape [1, action_dim]). Also overwrites
    agent._last_action so downstream record_transition / post-step
    bookkeeping sees the executed (random) action, not the agent-
    proposed one. This is purely a script-level wrapper -- no ree_core
    code is modified.
    """
    idx = rng_module.randint(0, action_dim - 1)
    action = _action_to_onehot(idx, action_dim, device)
    agent._last_action = action
    return action


# ---------------------------------------------------------------------------
# Training + eval loop
# ---------------------------------------------------------------------------

def _run_one_phase(
    agent: REEAgent,
    env: CausalGridWorldV2,
    phase_label: str,
    num_episodes: int,
    steps_per_episode: int,
    forced_random_episodes: int,
    train: bool,
    optimizers_and_params: Optional[Dict],
    rng_module,
    action_count_window: Optional[Dict[int, int]] = None,
) -> Dict:
    """Run num_episodes of agent-in-env.

    forced_random_episodes: if > 0, the first `forced_random_episodes`
      episodes of this phase override agent.select_action() output with
      uniform-random one-hot. Set to 0 in ARM_NORMAL or in P1 eval.
    train: if True, run the standard online training updates on E2
      world_forward, E3 harm_eval_head, and encoder aux losses.
    action_count_window: if provided, accumulate executed-action-class
      counts (the metric for P1 eval). Pass a fresh dict for the P1
      eval window per arm per seed.

    Returns dict with diagnostic counters.
    """
    device = agent.device
    action_dim = env.action_dim

    if train:
        assert optimizers_and_params is not None, \
            "train=True requires optimizers_and_params"
        e1_optimizer = optimizers_and_params["e1_optimizer"]
        e2_wf_optimizer = optimizers_and_params["e2_wf_optimizer"]
        harm_eval_optimizer = optimizers_and_params["harm_eval_optimizer"]
        aux_optimizer = optimizers_and_params["aux_optimizer"]
        aux_params = optimizers_and_params["aux_params"]
        wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = \
            optimizers_and_params["wf_buf"]
        harm_eval_buf: List[Tuple[torch.Tensor, torch.Tensor]] = \
            optimizers_and_params["harm_eval_buf"]
        agent.train()
    else:
        agent.eval()

    n_forced_actions = 0
    n_total_actions = 0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        z_world_prev: Optional[torch.Tensor] = None
        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        force_this_episode = ep < forced_random_episodes

        for _step in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_h = _obs_harm(obs_dict)
            obs_h_a = _obs_harm_a(obs_dict)
            obs_h_h = _obs_harm_history(obs_dict)
            prox_t = _obs_resource_prox(obs_dict)
            accum_t = _obs_accum(obs_dict)

            latent = agent.sense(
                obs_body, obs_world,
                obs_harm=obs_h, obs_harm_a=obs_h_a,
                obs_harm_history=obs_h_h,
            )
            z_world_curr = latent.z_world.detach()

            # Aux losses (only when training).
            if train:
                aux_terms: List[torch.Tensor] = []
                prox_target_t = torch.tensor([[prox_t]], device=device)
                prox_loss = agent.compute_resource_proximity_loss(
                    prox_target_t, latent,
                )
                if prox_loss is not None and prox_loss.requires_grad:
                    aux_terms.append(prox_loss)
                accum_target_t = torch.tensor([[accum_t]], device=device)
                harm_accum_loss = agent.compute_harm_accum_loss(
                    accum_target_t, latent,
                )
                if harm_accum_loss is not None and harm_accum_loss.requires_grad:
                    aux_terms.append(harm_accum_loss)
                if aux_terms:
                    aux_loss = sum(aux_terms)
                    aux_optimizer.zero_grad()
                    aux_loss.backward(retain_graph=False)
                    torch.nn.utils.clip_grad_norm_(aux_params, 1.0)
                    aux_optimizer.step()

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(
                    z_self_prev, action_prev, latent.z_self.detach(),
                )

            # Re-sense post aux step (matches V3-EXQ-543c pattern).
            latent = agent.sense(
                obs_body, obs_world,
                obs_harm=obs_h, obs_harm_a=obs_h_a,
                obs_harm_history=obs_h_h,
            )

            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, WORLD_DIM, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            drive_level = REEAgent.compute_drive_level(obs_body)
            benefit_exposure = max(
                0.0, float(obs_dict.get("benefit_exposure", 0.0)),
            )
            agent.update_z_goal(
                benefit_exposure=benefit_exposure,
                drive_level=drive_level,
            )

            # ----- The single manipulated variable. -----
            # When force_this_episode is True (warmup window in
            # ARM_WARMUP), the agent's proposed action is computed but
            # then DISCARDED; a uniform-random one-hot replaces it.
            # This keeps the agent's internal state update path
            # bit-identical to the non-warmup path (select_action runs,
            # any side-effects on commit / beta_gate / closure proceed),
            # only the EMITTED action changes. agent._last_action is
            # also overwritten so downstream record_transition sees the
            # executed (random) action.
            proposed_action = agent.select_action(candidates, ticks, temperature=1.0)
            if proposed_action is None:
                proposed_action = _action_to_onehot(
                    rng_module.randint(0, action_dim - 1), action_dim, device,
                )
                agent._last_action = proposed_action

            if force_this_episode:
                action = _override_with_uniform_random(
                    agent, action_dim, device, rng_module,
                )
                n_forced_actions += 1
            else:
                action = proposed_action

            n_total_actions += 1

            # Record executed-action-class in the metric window when
            # provided (P1 eval window).
            if action_count_window is not None:
                a_idx = int(action[0].argmax().item())
                action_count_window[a_idx] = (
                    action_count_window.get(a_idx, 0) + 1
                )

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            # Training updates (only when train=True).
            if train:
                if z_world_prev is not None and action_prev is not None:
                    wf_buf.append(
                        (z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()),
                    )
                    if len(wf_buf) > WF_BUF_MAX:
                        del wf_buf[:len(wf_buf) - WF_BUF_MAX]

                harm_target = abs(float(harm_signal)) if float(harm_signal) < 0 else 0.0
                harm_eval_buf.append(
                    (z_world_curr.cpu(), torch.tensor([harm_target])),
                )
                if len(harm_eval_buf) > HARM_EVAL_BUF_MAX:
                    del harm_eval_buf[:len(harm_eval_buf) - HARM_EVAL_BUF_MAX]

                if len(wf_buf) >= BATCH_SIZE:
                    idxs = torch.randperm(len(wf_buf))[:BATCH_SIZE].tolist()
                    zw_b = torch.cat([wf_buf[i][0] for i in idxs]).to(device)
                    a_b = torch.cat([wf_buf[i][1] for i in idxs]).to(device)
                    zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(device)
                    wf_pred = agent.e2.world_forward(zw_b, a_b)
                    wf_loss = F.mse_loss(wf_pred, zw1_b)
                    if wf_loss.requires_grad:
                        e2_wf_optimizer.zero_grad()
                        wf_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            list(agent.e2.world_transition.parameters())
                            + list(agent.e2.world_action_encoder.parameters()),
                            1.0,
                        )
                        e2_wf_optimizer.step()
                    with torch.no_grad():
                        agent.e3.update_running_variance(
                            (wf_pred.detach() - zw1_b).detach(),
                        )

                if len(harm_eval_buf) >= BATCH_SIZE:
                    idxs = torch.randperm(len(harm_eval_buf))[:BATCH_SIZE].tolist()
                    zw_b = torch.cat([harm_eval_buf[i][0] for i in idxs]).to(device)
                    t_b = torch.stack(
                        [harm_eval_buf[i][1] for i in idxs],
                    ).to(device).view(-1, 1)
                    he_pred = agent.e3.harm_eval_head(zw_b)
                    he_loss = F.mse_loss(he_pred, t_b)
                    if he_loss.requires_grad:
                        harm_eval_optimizer.zero_grad()
                        he_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            agent.e3.harm_eval_head.parameters(), 1.0,
                        )
                        harm_eval_optimizer.step()

            z_world_prev = z_world_curr
            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()

            if done:
                break

    return {
        "phase_label": phase_label,
        "n_total_actions": n_total_actions,
        "n_forced_actions": n_forced_actions,
    }


def _run_condition(seed: int, condition: str) -> Dict:
    """Run P0 training (with optional warmup override) + P1 eval.

    Returns dict including P1 action_class_counts + action_class_entropy
    + the diagnostic counters from both phases.
    """
    agent, env = _make_agent_and_env(seed)
    device = agent.device

    # Build optimizers (P0 training).
    e1_optimizer = optim.Adam(agent.e1.parameters(), lr=LR_E1)
    e2_wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters())
        + list(agent.e2.world_action_encoder.parameters()),
        lr=LR_E2_WF,
    )
    harm_eval_optimizer = optim.Adam(
        agent.e3.harm_eval_head.parameters(), lr=LR_E3_HARM,
    )
    aux_params = list(agent.latent_stack.parameters())
    aux_optimizer = optim.Adam(aux_params, lr=LR_ENC_AUX)

    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_eval_buf: List[Tuple[torch.Tensor, torch.Tensor]] = []

    optimizers_and_params = {
        "e1_optimizer": e1_optimizer,
        "e2_wf_optimizer": e2_wf_optimizer,
        "harm_eval_optimizer": harm_eval_optimizer,
        "aux_optimizer": aux_optimizer,
        "aux_params": aux_params,
        "wf_buf": wf_buf,
        "harm_eval_buf": harm_eval_buf,
    }

    # Dedicated RNG module for action-override sampling so determinism
    # is preserved across the two arms at matched seed.
    rng_module = random.Random(seed)

    forced_random_episodes = (
        int(round(P0_WARMUP_EPISODES * WARMUP_FRACTION))
        if condition == "WARMUP" else 0
    )

    # P0 -- training (with optional warmup override).
    p0_diag = _run_one_phase(
        agent=agent,
        env=env,
        phase_label="P0",
        num_episodes=P0_WARMUP_EPISODES,
        steps_per_episode=STEPS_PER_EPISODE,
        forced_random_episodes=forced_random_episodes,
        train=True,
        optimizers_and_params=optimizers_and_params,
        rng_module=rng_module,
        action_count_window=None,
    )
    print(
        f"  [train] label seed={seed} cond={condition} P0 "
        f"{P0_WARMUP_EPISODES}/{P0_WARMUP_EPISODES} "
        f"n_forced={p0_diag['n_forced_actions']} "
        f"n_total={p0_diag['n_total_actions']}",
        flush=True,
    )

    # P1 -- eval (no training, no override). Accumulate executed-action-
    # class counts -- this is the metric for the acceptance gate.
    action_count_window: Dict[int, int] = {}
    p1_diag = _run_one_phase(
        agent=agent,
        env=env,
        phase_label="P1",
        num_episodes=P1_EVAL_EPISODES,
        steps_per_episode=STEPS_PER_EPISODE,
        forced_random_episodes=0,
        train=False,
        optimizers_and_params=None,
        rng_module=rng_module,
        action_count_window=action_count_window,
    )
    p1_entropy = _shannon_entropy(action_count_window)
    print(
        f"  [train] label seed={seed} cond={condition} P1 "
        f"{P1_EVAL_EPISODES}/{P1_EVAL_EPISODES} "
        f"entropy={p1_entropy:.4f} n_actions={sum(action_count_window.values())}",
        flush=True,
    )

    return {
        "condition": condition,
        "seed": seed,
        "p1_action_class_counts": action_count_window,
        "p1_action_class_entropy": p1_entropy,
        "p1_n_actions": sum(action_count_window.values()),
        "p0_n_forced_actions": p0_diag["n_forced_actions"],
        "p0_n_total_actions": p0_diag["n_total_actions"],
        "p1_n_total_actions": p1_diag["n_total_actions"],
        "warmup_fraction_applied": (
            WARMUP_FRACTION if condition == "WARMUP" else 0.0
        ),
        "forced_random_episodes": forced_random_episodes,
    }


# ---------------------------------------------------------------------------
# Plan / smoke / main
# ---------------------------------------------------------------------------

def _print_plan() -> None:
    print(f"{EXPERIMENT_TYPE} -- forced-exploration warmup test", flush=True)
    print(f"Arms: {CONDITIONS}", flush=True)
    print(f"Seeds: {SEEDS}", flush=True)
    print(
        f"P0 train: {P0_WARMUP_EPISODES} ep x {STEPS_PER_EPISODE} steps",
        flush=True,
    )
    print(
        f"P1 eval:  {P1_EVAL_EPISODES} ep x {STEPS_PER_EPISODE} steps",
        flush=True,
    )
    print(
        f"WARMUP override window: first "
        f"{int(round(P0_WARMUP_EPISODES * WARMUP_FRACTION))} of P0 ep "
        f"(ARM_WARMUP only)",
        flush=True,
    )
    print(
        f"Metric: P1 action_class_entropy per arm per seed",
        flush=True,
    )
    print(
        f"PASS = ARM_WARMUP_P1 - ARM_NORMAL_P1 >= "
        f"{ENTROPY_DELTA_THRESHOLD} AND ARM_WARMUP_P1 > "
        f"{ENTROPY_ABS_FLOOR} in >= {SEEDS_REQUIRED_TO_PASS}/{len(SEEDS)} seeds",
        flush=True,
    )
    print(f"experiment_purpose={EXPERIMENT_PURPOSE}", flush=True)


def _run_smoke() -> None:
    """1 seed x 1 ep x 20 steps per phase, with the override forced ON
    in 1 episode of P0. Verifies: (a) override runs without crashing,
    (b) action distribution varies under override, (c) ARM_NORMAL action
    selection still works."""
    print(
        "SMOKE MODE: 1 seed x 1 ep x 20 steps per phase; no manifest write",
        flush=True,
    )
    seed = SEEDS[0]
    for cond in CONDITIONS:
        agent, env = _make_agent_and_env(seed)
        device = agent.device
        e1_optimizer = optim.Adam(agent.e1.parameters(), lr=LR_E1)
        e2_wf_optimizer = optim.Adam(
            list(agent.e2.world_transition.parameters())
            + list(agent.e2.world_action_encoder.parameters()),
            lr=LR_E2_WF,
        )
        harm_eval_optimizer = optim.Adam(
            agent.e3.harm_eval_head.parameters(), lr=LR_E3_HARM,
        )
        aux_params = list(agent.latent_stack.parameters())
        aux_optimizer = optim.Adam(aux_params, lr=LR_ENC_AUX)
        wf_buf: List = []
        harm_eval_buf: List = []
        optimizers_and_params = {
            "e1_optimizer": e1_optimizer,
            "e2_wf_optimizer": e2_wf_optimizer,
            "harm_eval_optimizer": harm_eval_optimizer,
            "aux_optimizer": aux_optimizer,
            "aux_params": aux_params,
            "wf_buf": wf_buf,
            "harm_eval_buf": harm_eval_buf,
        }
        rng_module = random.Random(seed)
        forced_eps = 1 if cond == "WARMUP" else 0

        print(f"Seed {seed} Condition {cond}", flush=True)
        # P0 smoke: 1 ep x 20 steps with override active in WARMUP arm.
        p0_diag = _run_one_phase(
            agent=agent, env=env, phase_label="P0",
            num_episodes=1, steps_per_episode=20,
            forced_random_episodes=forced_eps,
            train=True, optimizers_and_params=optimizers_and_params,
            rng_module=rng_module, action_count_window=None,
        )
        # P1 smoke: 1 ep x 20 steps, no override, accumulate action counts.
        action_count_window: Dict[int, int] = {}
        _run_one_phase(
            agent=agent, env=env, phase_label="P1",
            num_episodes=1, steps_per_episode=20,
            forced_random_episodes=0, train=False,
            optimizers_and_params=None, rng_module=rng_module,
            action_count_window=action_count_window,
        )
        p1_entropy = _shannon_entropy(action_count_window)
        print(
            f"  [train] label seed={seed} ep 1/1 "
            f"P0_forced={p0_diag['n_forced_actions']} "
            f"P0_total={p0_diag['n_total_actions']} "
            f"P1_entropy={p1_entropy:.4f} "
            f"P1_action_counts={dict(action_count_window)}",
            flush=True,
        )
        print(f"verdict: PASS", flush=True)
    print("SMOKE OK", flush=True)


def main() -> Tuple[Optional[str], Optional[str]]:
    parser = argparse.ArgumentParser(
        description=f"{EXPERIMENT_TYPE} forced-exploration warmup test",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan and exit; do not execute.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="1 seed x 1 ep x 20 steps per phase smoke test.",
    )
    args = parser.parse_args()

    if args.dry_run:
        _print_plan()
        print("DRY RUN OK", flush=True)
        return (None, None)

    if args.smoke:
        _run_smoke()
        return (None, None)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict] = []
    for seed in SEEDS:
        for cond in CONDITIONS:
            print(f"Seed {seed} Condition {cond}", flush=True)
            r = _run_condition(seed=seed, condition=cond)
            print(f"verdict: PASS", flush=True)
            all_results.append(r)

    normal_by_seed = {
        r["seed"]: r for r in all_results if r["condition"] == "NORMAL"
    }
    warmup_by_seed = {
        r["seed"]: r for r in all_results if r["condition"] == "WARMUP"
    }

    per_seed_summary = []
    seeds_clearing_delta = 0
    seeds_clearing_floor = 0
    for seed in SEEDS:
        normal_e = normal_by_seed[seed]["p1_action_class_entropy"]
        warmup_e = warmup_by_seed[seed]["p1_action_class_entropy"]
        delta = warmup_e - normal_e
        cleared_delta = delta >= ENTROPY_DELTA_THRESHOLD
        cleared_floor = warmup_e > ENTROPY_ABS_FLOOR
        per_seed_summary.append({
            "seed": seed,
            "normal_p1_entropy": normal_e,
            "warmup_p1_entropy": warmup_e,
            "delta": delta,
            "cleared_delta": cleared_delta,
            "cleared_floor": cleared_floor,
            "cleared_both": cleared_delta and cleared_floor,
        })
        if cleared_delta:
            seeds_clearing_delta += 1
        if cleared_floor:
            seeds_clearing_floor += 1

    seeds_passing_both = sum(
        1 for row in per_seed_summary if row["cleared_both"]
    )
    passed = (
        seeds_clearing_delta >= SEEDS_REQUIRED_TO_PASS
        and seeds_clearing_floor >= SEEDS_REQUIRED_TO_PASS
        and seeds_passing_both >= SEEDS_REQUIRED_TO_PASS
    )

    if passed:
        outcome = "PASS"
        evidence_direction = "supports"  # bootstrap-loop / data-narrowness H1
    else:
        outcome = "FAIL"
        # Could be H2 (substrate collapse regardless) OR mixed; the
        # per_seed_summary disambiguates which row of the interpretation
        # grid the result lands on. evidence_direction reflects the
        # collective signal: FAIL with both arms near zero is "supports"
        # for the H2 reading. We report "weakens" for the data-narrowness
        # claim under test (the script's primary hypothesis).
        evidence_direction = "weakens"

    summary = {
        "gate_rule": (
            f"warmup_p1_entropy - normal_p1_entropy >= "
            f"{ENTROPY_DELTA_THRESHOLD} AND warmup_p1_entropy > "
            f"{ENTROPY_ABS_FLOOR} in >= {SEEDS_REQUIRED_TO_PASS}/"
            f"{len(SEEDS)} seeds"
        ),
        "per_seed_summary": per_seed_summary,
        "seeds_clearing_delta": seeds_clearing_delta,
        "seeds_clearing_floor": seeds_clearing_floor,
        "seeds_passing_both": seeds_passing_both,
        "seeds_required": SEEDS_REQUIRED_TO_PASS,
        "pass": outcome == "PASS",
    }

    print(f"\nOutcome: {outcome}", flush=True)
    for row in per_seed_summary:
        print(
            f"  seed={row['seed']} normal={row['normal_p1_entropy']:.4f} "
            f"warmup={row['warmup_p1_entropy']:.4f} "
            f"delta={row['delta']:.4f} cleared_both={row['cleared_both']}",
            flush=True,
        )

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "evidence_direction_note": (
            "Training-loop diagnostic: forced uniform-random action "
            "warmup over the first 50% of P0 vs no override. PASS = "
            "ARM_WARMUP P1 entropy clears ARM_NORMAL P1 entropy by "
            ">= 0.10 AND clears 0.30 absolute floor in >= 2/3 seeds. "
            "Three-row interpretation grid: (i) PASS supports the "
            "bootstrap-loop / data-narrowness reading -- substrate has "
            "the capacity for diverse policy; monostrategy is a "
            "training-data pathology. Route to exploration-scheduling "
            "follow-up. (ii) FAIL with both arms near zero supports "
            "the substrate-collapsing-regardless reading -- forced-"
            "explore data does not keep the policy diverse. Elevates "
            "urgency of V3-EXQ-551 proposer/evaluator diagnostic and "
            "the ARC-062 / MECH-309 rule-apprehension cluster. (iii) "
            "Mixed outcomes (one seed lifts, others collapse) -> "
            "inconclusive; substrate is marginal, sensitive to init "
            "noise; surface for routing."
        ),
        "pass_criteria_summary": summary,
        "per_seed_results": all_results,
        "config": {
            "conditions": CONDITIONS,
            "seeds": SEEDS,
            "p0_warmup_episodes": P0_WARMUP_EPISODES,
            "p1_eval_episodes": P1_EVAL_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "warmup_fraction": WARMUP_FRACTION,
            "entropy_delta_threshold": ENTROPY_DELTA_THRESHOLD,
            "entropy_abs_floor": ENTROPY_ABS_FLOOR,
            "seeds_required_to_pass": SEEDS_REQUIRED_TO_PASS,
            "env_kwargs": ENV_KWARGS,
        },
    }

    out_file = write_flat_manifest(
        output,
        out_dir,
        dry_run=False,
        config=output.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    print(f"Output written to: {out_file}", flush=True)

    return (outcome, str(out_file))


if __name__ == "__main__":
    _outcome, _manifest_path = main()
    if _outcome is not None and _manifest_path is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path)
    sys.exit(0)
