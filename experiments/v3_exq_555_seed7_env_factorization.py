#!/opt/local/bin/python3
"""
V3-EXQ-555 -- Seed-7 env vs agent factorization diagnostic.

Claims: [] (monostrategy-investigation diagnostic; no substrate claim under test)

Purpose (evidence_direction_note: diagnostic)
---------------------------------------------
V3-EXQ-552 (forced-exploration warmup, completed 2026-05-11) found that
seed=7 sustains action_class_entropy ~ 0.68 in BOTH the ARM_NORMAL and
ARM_WARMUP arms, while seeds 42 and 17 collapse to entropy=0 in both arms
under identical script + env config + agent code. This is the only positive
lead in the V3 monostrategy investigation: it shows the substrate CAN
produce diverse policies under SOME seed-of-env conditions.

The standard ree-v3 experiment template (mirrored from V3-EXQ-552's
_make_agent_and_env) seeds BOTH env and agent from a single seed value:

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env = CausalGridWorldV2(seed=seed, **ENV_KWARGS)
    agent = REEAgent(REEConfig.from_dims(...))   # weight init reads
                                                  # torch.manual_seed state

So seed=7 is BOTH the env-initialization seed (hazard placement, resource
placement, reef placement, env RNG stream) AND the agent-initialization
seed (E1/E2/E3 weight init, LatentStack init, all torch.nn.Linear
weight draws).

The diagnostic question: what is seed=7 different in -- env init, agent
init, or both? This experiment factors seed = (env_seed, agent_seed) and
runs a 2x2 cell sweep.

Four-cell factorial
-------------------
    C0_baseline:        env_seed=7,  agent_seed=7   (replicates EXQ-552
                                                      seed=7, expected
                                                      entropy ~ 0.68)
    C1_env_only:        env_seed=7,  agent_seed=42  (does seed-7 env
                                                      survive a
                                                      "collapsing" agent
                                                      init?)
    C2_agent_only:      env_seed=42, agent_seed=7   (does seed-7 agent
                                                      init survive a
                                                      "collapsing" env?)
    C3_baseline_control: env_seed=42, agent_seed=42 (replicates EXQ-552
                                                      seed=42, expected
                                                      entropy=0)

Run depth: REPLICATE V3-EXQ-552 ARM_NORMAL verbatim. P0=40 ep training,
P1=60 ep eval, 200 steps/ep, CausalGridWorldV2 with the same ENV_KWARGS
as EXQ-552. One arm per cell (single ARM_NORMAL run; no WARMUP override).

Each cell runs once -- the seed IS the variable.

Independent seeding
-------------------
The factorization is achieved by:
  (a) setting torch.manual_seed / random.seed / np.random.seed to
      agent_seed BEFORE constructing REEAgent (so weight init reads
      agent_seed),
  (b) passing env_seed (independently) to CausalGridWorldV2(seed=env_seed).
  (c) re-setting the RNGs to agent_seed once more AFTER env construction
      but BEFORE REEAgent construction, in case env construction consumed
      torch RNG state (defensive; np.random.default_rng is independent
      of torch but env constructor allocates a few small tensors).
  (d) the action-override RNG (random.Random) is seeded with env_seed
      so that env-side stochasticity in step() / drift / hazard injection
      is fully attributed to the env_seed factor. This matches the
      EXQ-552 architecture where the rng_module is per-seed.

Verification (UC of the smoke test): the agent's E3
harm_eval_head.weight.flatten()[:8] is read at C0 and C3; the env's
starting hazard positions (env.hazards) are read at C0 and C3. These
must differ as expected:
  C0 (env=7, agent=7) agent_weights == C2 (env=42, agent=7) agent_weights
  C0 (env=7, agent=7) env_hazards   == C1 (env=7, agent=42) env_hazards
  C0 (env=7, agent=7) agent_weights != C3 (env=42, agent=42) agent_weights
  C0 (env=7, agent=7) env_hazards   != C3 (env=42, agent=42) env_hazards

Pre-registered interpretation grid (5 rows; embedded in
evidence_direction_note)
--------------------------------------------------------------------------
  R1 env_side_only:  C1 entropy >> 0.30 AND C2 entropy ~ 0
    -> the env-side seed-7 condition (terrain / hazard / resource
       placement) is what enables diversity. Routes to env-config
       diagnostic: which specific env placement differs?
  R2 agent_side_only: C1 entropy ~ 0 AND C2 entropy >> 0.30
    -> the agent-side seed-7 condition (weight init) is what enables
       diversity. Routes to agent-init diagnostic: which module's init
       differs critically?
  R3 conjunctive:    C1 ~ 0 AND C2 ~ 0 AND C0 ~ 0.68
    -> diversity requires BOTH env-7 AND agent-7 simultaneously
       (interaction, not main effect). Routes to a more careful 2x2
       with replication.
  R4 either_sufficient: BOTH C1 and C2 >> 0.30
    -> either side alone breaks monostrategy. Strongest result:
       implies the substrate has the capacity but typical seed
       combinations land in a collapsing basin.
  R5 replication_failure: C0 NOT in [0.58, 0.78] (~0.68 +/- 0.10)
    -> the V3-EXQ-552 seed-7 finding doesn't replicate. Routes to
       stochasticity audit / non-deterministic substrate components.

experiment_purpose=diagnostic (decomposes a known anomaly; no falsifiable
substrate claim under test).
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


EXPERIMENT_TYPE = "v3_exq_555_seed7_env_factorization"
QUEUE_ID = "V3-EXQ-555"
CLAIM_IDS: List[str] = []  # monostrategy-investigation diagnostic; no substrate claim
EXPERIMENT_PURPOSE = "diagnostic"

# Four factorial cells: (cell_label, env_seed, agent_seed).
CELLS: List[Tuple[str, int, int]] = [
    ("C0_baseline",         7,  7),
    ("C1_env_only",         7,  42),
    ("C2_agent_only",       42, 7),
    ("C3_baseline_control", 42, 42),
]

# Phased schedule -- matches V3-EXQ-552 ARM_NORMAL canonical pattern.
P0_TRAIN_EPISODES = 40
P1_EVAL_EPISODES = 60
STEPS_PER_EPISODE = 200

# Latent dims (match V3-EXQ-552 / V3-EXQ-543c baseline).
WORLD_DIM = 32
SELF_DIM = 32
HARM_DIM = 32
HARM_A_DIM = 16
HARM_HISTORY_LEN = 10

# Buffer + batch (match V3-EXQ-552).
WF_BUF_MAX = 2000
HARM_EVAL_BUF_MAX = 2000
BATCH_SIZE = 32

# Learning rates (match V3-EXQ-552).
LR_E1 = 1e-4
LR_E2_WF = 3e-4
LR_E3_HARM = 1e-3
LR_ENC_AUX = 5e-4

# Env config: VERBATIM copy of V3-EXQ-552 ENV_KWARGS for direct replication.
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


def _make_agent_and_env(
    env_seed: int, agent_seed: int,
) -> Tuple[REEAgent, CausalGridWorldV2]:
    """Construct env and agent with INDEPENDENT seeds.

    Order matters:
      1. Seed torch / random / numpy from agent_seed first (defensive --
         in case env construction reads from these, though
         CausalGridWorldV2 only uses np.random.default_rng(seed) which
         is independent of the global np.random.seed RNG state).
      2. Construct env with env_seed -- env's _rng is np.random.default_rng(env_seed),
         a SEPARATE RNG stream from numpy's global one.
      3. Re-seed torch / random / numpy from agent_seed (defensive
         against any torch RNG consumption during env construction --
         the env constructor allocates harm_obs_a_ema and harm_history
         numpy buffers but no torch tensors).
      4. Construct REEAgent -- all torch.nn.Linear weight inits draw
         from torch's current RNG state, which is agent_seed.
    """
    # Step 1: defensive global seeding from agent_seed.
    torch.manual_seed(agent_seed)
    random.seed(agent_seed)
    np.random.seed(agent_seed)

    # Step 2: env constructed with env_seed (uses its own _rng stream).
    env = CausalGridWorldV2(seed=env_seed, **ENV_KWARGS)

    # Step 3: re-seed from agent_seed (defensive; env construction
    # should not consume torch RNG but we re-seed to be safe).
    torch.manual_seed(agent_seed)
    random.seed(agent_seed)
    np.random.seed(agent_seed)

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

    # Step 4: agent construction reads torch RNG (currently set to agent_seed).
    agent = REEAgent(config)
    return agent, env


def _agent_init_signature(agent: REEAgent) -> List[float]:
    """Return first 8 elements of E3 harm_eval_head's first learnable
    parameter as a sanity fingerprint of the agent's weight-init RNG path.

    harm_eval_head is a Sequential, so we walk parameters() to find the
    first leaf weight tensor.
    """
    params = list(agent.e3.harm_eval_head.parameters())
    if not params:
        # Defensive fallback to e2.world_transition first param.
        params = list(agent.e2.world_transition.parameters())
    w = params[0].detach().flatten()
    n = min(8, w.numel())
    return [float(w[i].item()) for i in range(n)]


def _env_init_signature(env: CausalGridWorldV2) -> Dict:
    """Return env's starting hazard + resource positions as a fingerprint
    of the env-side seed path."""
    return {
        "hazards": [list(p) for p in env.hazards],
        "resources": [list(p) for p in env.resources],
        "agent_pos": [int(env.agent_x), int(env.agent_y)],
    }


# ---------------------------------------------------------------------------
# Training + eval loop (matches V3-EXQ-552 ARM_NORMAL verbatim)
# ---------------------------------------------------------------------------

def _run_one_phase(
    agent: REEAgent,
    env: CausalGridWorldV2,
    phase_label: str,
    num_episodes: int,
    steps_per_episode: int,
    train: bool,
    optimizers_and_params: Optional[Dict],
    rng_module,
    action_count_window: Optional[Dict[int, int]] = None,
) -> Dict:
    """Run num_episodes of agent-in-env (ARM_NORMAL only -- no warmup)."""
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

    n_total_actions = 0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        z_world_prev: Optional[torch.Tensor] = None
        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None

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

            # Re-sense post aux step (matches V3-EXQ-552 pattern).
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

            proposed_action = agent.select_action(candidates, ticks, temperature=1.0)
            if proposed_action is None:
                proposed_action = _action_to_onehot(
                    rng_module.randint(0, action_dim - 1), action_dim, device,
                )
                agent._last_action = proposed_action

            action = proposed_action
            n_total_actions += 1

            if action_count_window is not None:
                a_idx = int(action[0].argmax().item())
                action_count_window[a_idx] = (
                    action_count_window.get(a_idx, 0) + 1
                )

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

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
    }


def _run_cell(cell_label: str, env_seed: int, agent_seed: int) -> Dict:
    """Run one (env_seed, agent_seed) cell.

    Returns dict including init signatures (for the smoke-test verification
    check) + P1 action_class_counts + action_class_entropy.
    """
    agent, env = _make_agent_and_env(env_seed=env_seed, agent_seed=agent_seed)
    device = agent.device

    init_agent_sig = _agent_init_signature(agent)
    init_env_sig = _env_init_signature(env)

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

    # Action-fallback RNG seeded from env_seed so any env-side stochasticity
    # is attributable to env_seed alone.
    rng_module = random.Random(env_seed)

    # P0 -- training (no warmup override; ARM_NORMAL only).
    p0_diag = _run_one_phase(
        agent=agent,
        env=env,
        phase_label="P0",
        num_episodes=P0_TRAIN_EPISODES,
        steps_per_episode=STEPS_PER_EPISODE,
        train=True,
        optimizers_and_params=optimizers_and_params,
        rng_module=rng_module,
        action_count_window=None,
    )
    print(
        f"  [train] cell={cell_label} env_seed={env_seed} "
        f"agent_seed={agent_seed} P0 {P0_TRAIN_EPISODES}/{P0_TRAIN_EPISODES} "
        f"n_total={p0_diag['n_total_actions']}",
        flush=True,
    )

    # P1 -- eval. Accumulate executed-action-class counts.
    action_count_window: Dict[int, int] = {}
    p1_diag = _run_one_phase(
        agent=agent,
        env=env,
        phase_label="P1",
        num_episodes=P1_EVAL_EPISODES,
        steps_per_episode=STEPS_PER_EPISODE,
        train=False,
        optimizers_and_params=None,
        rng_module=rng_module,
        action_count_window=action_count_window,
    )
    p1_entropy = _shannon_entropy(action_count_window)
    print(
        f"  [eval] cell={cell_label} env_seed={env_seed} "
        f"agent_seed={agent_seed} P1 {P1_EVAL_EPISODES}/{P1_EVAL_EPISODES} "
        f"entropy={p1_entropy:.4f} n_actions={sum(action_count_window.values())}",
        flush=True,
    )

    return {
        "cell": cell_label,
        "env_seed": env_seed,
        "agent_seed": agent_seed,
        "p1_action_class_counts": action_count_window,
        "p1_action_class_entropy": p1_entropy,
        "p1_n_actions": sum(action_count_window.values()),
        "p0_n_total_actions": p0_diag["n_total_actions"],
        "p1_n_total_actions": p1_diag["n_total_actions"],
        "init_agent_signature": init_agent_sig,
        "init_env_signature": init_env_sig,
    }


# ---------------------------------------------------------------------------
# Interpretation grid
# ---------------------------------------------------------------------------

def _classify_interpretation(
    c0: float, c1: float, c2: float, c3: float,
) -> Tuple[str, str]:
    """Map (C0, C1, C2, C3) entropy values to one of 5 interpretation rows.

    Returns (row_label, row_description). Thresholds:
      0.30 = "clears 0.30 floor" (real diversity, not noise)
      0.10 = "near zero"
      C0 must be in [0.58, 0.78] for non-R5 rows (within +/- 0.10 of 0.68).
    """
    c0_replicates = (0.58 <= c0 <= 0.78)
    if not c0_replicates:
        return (
            "R5_replication_failure",
            f"C0_baseline entropy {c0:.4f} is NOT within [0.58, 0.78] of "
            f"EXQ-552's seed-7 ARM_NORMAL value (0.6787). The seed-7 "
            f"finding does not replicate. Route to stochasticity audit / "
            f"non-deterministic substrate components."
        )

    c1_high = c1 > 0.30
    c2_high = c2 > 0.30
    c1_low = c1 < 0.10
    c2_low = c2 < 0.10

    if c1_high and c2_low:
        return (
            "R1_env_side_only",
            f"C1_env_only entropy {c1:.4f} >> 0.30 AND C2_agent_only "
            f"entropy {c2:.4f} ~ 0. The env-side seed-7 condition "
            f"(terrain / hazard / resource placement) is what enables "
            f"diversity. Routes to env-config diagnostic."
        )
    if c1_low and c2_high:
        return (
            "R2_agent_side_only",
            f"C1_env_only entropy {c1:.4f} ~ 0 AND C2_agent_only "
            f"entropy {c2:.4f} >> 0.30. The agent-side seed-7 condition "
            f"(weight init) is what enables diversity. Routes to "
            f"agent-init diagnostic (which module's init differs "
            f"critically?)."
        )
    if c1_high and c2_high:
        return (
            "R4_either_sufficient",
            f"BOTH C1_env_only {c1:.4f} AND C2_agent_only {c2:.4f} >> "
            f"0.30. Either side alone breaks monostrategy. Strongest "
            f"result: the substrate has capacity but typical seed "
            f"combinations land in a collapsing basin."
        )
    if c1_low and c2_low:
        return (
            "R3_conjunctive",
            f"C1_env_only entropy {c1:.4f} ~ 0 AND C2_agent_only "
            f"entropy {c2:.4f} ~ 0 AND C0_baseline {c0:.4f} ~ 0.68. "
            f"Diversity requires BOTH env-7 AND agent-7 simultaneously "
            f"(interaction, not main effect). Routes to a more careful "
            f"2x2 with replication."
        )

    # Intermediate case (neither clearly low nor clearly high on one or both)
    return (
        "R_inconclusive",
        f"C1={c1:.4f}, C2={c2:.4f} fall in the 0.10-0.30 transition band; "
        f"neither cleanly low nor cleanly high. Surface for routing; "
        f"may need replication or a tighter env/agent-side decomposition."
    )


# ---------------------------------------------------------------------------
# Plan / smoke / main
# ---------------------------------------------------------------------------

def _print_plan() -> None:
    print(f"{EXPERIMENT_TYPE} -- seed-7 env vs agent factorization", flush=True)
    print(f"Cells: {[c[0] for c in CELLS]}", flush=True)
    for label, env_s, agent_s in CELLS:
        print(f"  {label}: env_seed={env_s}, agent_seed={agent_s}", flush=True)
    print(
        f"P0 train: {P0_TRAIN_EPISODES} ep x {STEPS_PER_EPISODE} steps",
        flush=True,
    )
    print(
        f"P1 eval:  {P1_EVAL_EPISODES} ep x {STEPS_PER_EPISODE} steps",
        flush=True,
    )
    print(
        f"Metric: P1 action_class_entropy per cell (single run, "
        f"no per-cell replication -- the seed IS the variable).",
        flush=True,
    )
    print(
        f"5-row interpretation grid: R1 env_side_only / R2 agent_side_only "
        f"/ R3 conjunctive / R4 either_sufficient / R5 replication_failure.",
        flush=True,
    )
    print(f"experiment_purpose={EXPERIMENT_PURPOSE}", flush=True)


def _run_smoke() -> None:
    """1 cell (C0_baseline: env=7, agent=7) x 1 ep x 20 steps.

    Verifies:
      (a) factored seed split applies independently -- prints agent
          init signature (first 8 elems of harm_eval_head.weight) and
          env init signature (hazard / resource / agent positions);
      (b) at smoke depth, also constructs C3 (env=42, agent=42) and
          verifies that init signatures DIFFER from C0 (env hazards
          differ, agent weights differ);
      (c) the run-loop boots end-to-end without crashing.
    """
    print(
        "SMOKE MODE: C0 (env=7, agent=7) + C3 (env=42, agent=42) "
        "1 ep x 20 steps per phase; no manifest write",
        flush=True,
    )

    # Build C0 and C3 to compare init signatures.
    agent_c0, env_c0 = _make_agent_and_env(env_seed=7, agent_seed=7)
    sig_c0_agent = _agent_init_signature(agent_c0)
    sig_c0_env = _env_init_signature(env_c0)

    agent_c3, env_c3 = _make_agent_and_env(env_seed=42, agent_seed=42)
    sig_c3_agent = _agent_init_signature(agent_c3)
    sig_c3_env = _env_init_signature(env_c3)

    # Cross-cell consistency check: build C1 (env=7, agent=42).
    # Its env signature MUST equal C0's, and its agent signature MUST equal C3's.
    agent_c1, env_c1 = _make_agent_and_env(env_seed=7, agent_seed=42)
    sig_c1_agent = _agent_init_signature(agent_c1)
    sig_c1_env = _env_init_signature(env_c1)

    print("--- Init signature inspection ---", flush=True)
    print(f"C0 (env=7,  agent=7)  agent_weights[:8]: {sig_c0_agent}", flush=True)
    print(f"C3 (env=42, agent=42) agent_weights[:8]: {sig_c3_agent}", flush=True)
    print(f"C1 (env=7,  agent=42) agent_weights[:8]: {sig_c1_agent}", flush=True)
    print(f"C0 (env=7,  agent=7)  env hazards: {sig_c0_env['hazards']}", flush=True)
    print(f"C3 (env=42, agent=42) env hazards: {sig_c3_env['hazards']}", flush=True)
    print(f"C1 (env=7,  agent=42) env hazards: {sig_c1_env['hazards']}", flush=True)

    # Verification: env signatures must vary with env_seed only;
    # agent signatures must vary with agent_seed only.
    agents_differ_c0_c3 = sig_c0_agent != sig_c3_agent
    envs_differ_c0_c3 = sig_c0_env["hazards"] != sig_c3_env["hazards"]
    c1_env_matches_c0 = sig_c1_env["hazards"] == sig_c0_env["hazards"]
    c1_agent_matches_c3 = sig_c1_agent == sig_c3_agent

    print("--- Factorization verification ---", flush=True)
    print(f"  agents_differ C0 vs C3 (agent_seed 7 vs 42):     "
          f"{agents_differ_c0_c3}", flush=True)
    print(f"  envs_differ   C0 vs C3 (env_seed 7 vs 42):       "
          f"{envs_differ_c0_c3}", flush=True)
    print(f"  C1 env matches C0 (same env_seed=7):             "
          f"{c1_env_matches_c0}", flush=True)
    print(f"  C1 agent matches C3 (same agent_seed=42):        "
          f"{c1_agent_matches_c3}", flush=True)

    all_pass = (
        agents_differ_c0_c3
        and envs_differ_c0_c3
        and c1_env_matches_c0
        and c1_agent_matches_c3
    )
    if not all_pass:
        print("verdict: FAIL -- factorization invariants not satisfied", flush=True)
        raise RuntimeError(
            "Smoke factorization-invariant check failed. "
            "See printed signatures above."
        )

    # Run a quick C0 boot loop (1 ep x 20 steps in P0 + P1) to confirm
    # end-to-end stability.
    e1_optimizer = optim.Adam(agent_c0.e1.parameters(), lr=LR_E1)
    e2_wf_optimizer = optim.Adam(
        list(agent_c0.e2.world_transition.parameters())
        + list(agent_c0.e2.world_action_encoder.parameters()),
        lr=LR_E2_WF,
    )
    harm_eval_optimizer = optim.Adam(
        agent_c0.e3.harm_eval_head.parameters(), lr=LR_E3_HARM,
    )
    aux_params = list(agent_c0.latent_stack.parameters())
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
    rng_module = random.Random(7)
    _run_one_phase(
        agent=agent_c0, env=env_c0, phase_label="P0",
        num_episodes=1, steps_per_episode=20,
        train=True, optimizers_and_params=optimizers_and_params,
        rng_module=rng_module, action_count_window=None,
    )
    action_count_window: Dict[int, int] = {}
    _run_one_phase(
        agent=agent_c0, env=env_c0, phase_label="P1",
        num_episodes=1, steps_per_episode=20,
        train=False, optimizers_and_params=None,
        rng_module=rng_module, action_count_window=action_count_window,
    )
    print(f"  smoke C0 P1 action_counts={dict(action_count_window)}", flush=True)
    print("verdict: PASS", flush=True)
    print("SMOKE OK", flush=True)


def main() -> Tuple[Optional[str], Optional[str]]:
    parser = argparse.ArgumentParser(
        description=f"{EXPERIMENT_TYPE} seed-7 env vs agent factorization",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan and exit; do not execute.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="C0 + C3 + C1 init-signature smoke + 1 ep x 20 step boot test.",
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
    for cell_label, env_seed, agent_seed in CELLS:
        print(f"Cell {cell_label}: env_seed={env_seed} agent_seed={agent_seed}",
              flush=True)
        r = _run_cell(cell_label, env_seed, agent_seed)
        print(f"verdict: PASS", flush=True)
        all_results.append(r)

    by_label = {r["cell"]: r for r in all_results}
    c0 = by_label["C0_baseline"]["p1_action_class_entropy"]
    c1 = by_label["C1_env_only"]["p1_action_class_entropy"]
    c2 = by_label["C2_agent_only"]["p1_action_class_entropy"]
    c3 = by_label["C3_baseline_control"]["p1_action_class_entropy"]

    row_label, row_description = _classify_interpretation(c0, c1, c2, c3)

    summary = {
        "c0_baseline_entropy": c0,
        "c1_env_only_entropy": c1,
        "c2_agent_only_entropy": c2,
        "c3_baseline_control_entropy": c3,
        "interpretation_row": row_label,
        "interpretation_description": row_description,
    }

    # Outcome convention: this is a diagnostic, not a substrate-claim test.
    # The runner expects an outcome of PASS/FAIL/ERROR; we map "ran to
    # completion + produced an interpretable row" to PASS and report the
    # row in evidence_direction.
    if row_label == "R5_replication_failure":
        outcome = "FAIL"
        evidence_direction = "inconclusive"
    elif row_label == "R_inconclusive":
        outcome = "FAIL"
        evidence_direction = "inconclusive"
    else:
        outcome = "PASS"
        # The result type ITSELF is what we want -- there is no "positive"
        # direction. Mark non_contributory so governance does not weight it
        # against any claim's confidence.
        evidence_direction = "non_contributory"

    print(f"\nOutcome: {outcome}", flush=True)
    print(f"Interpretation row: {row_label}", flush=True)
    print(f"  C0 (env=7,  agent=7)  entropy = {c0:.4f}", flush=True)
    print(f"  C1 (env=7,  agent=42) entropy = {c1:.4f}", flush=True)
    print(f"  C2 (env=42, agent=7)  entropy = {c2:.4f}", flush=True)
    print(f"  C3 (env=42, agent=42) entropy = {c3:.4f}", flush=True)
    print(f"  {row_description}", flush=True)

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
            "Monostrategy-investigation diagnostic decomposing the "
            "V3-EXQ-552 seed-7 anomaly: action_class_entropy ~ 0.68 "
            "in seed=7 vs entropy=0 in seeds 42/17 under identical "
            "script + env + agent code. Factors seed = (env_seed, "
            "agent_seed) and runs the 2x2 cell sweep. Pre-registered "
            "5-row interpretation grid: "
            "(R1) env_side_only -- C1 entropy >> 0.30 AND C2 ~ 0; the "
            "env-side seed-7 condition enables diversity; route to "
            "env-config diagnostic. "
            "(R2) agent_side_only -- C1 ~ 0 AND C2 >> 0.30; the "
            "agent-side seed-7 condition enables diversity; route to "
            "agent-init diagnostic. "
            "(R3) conjunctive -- C1 ~ 0 AND C2 ~ 0 AND C0 ~ 0.68; "
            "diversity requires both simultaneously (interaction, not "
            "main effect); route to careful 2x2 with replication. "
            "(R4) either_sufficient -- BOTH C1 and C2 >> 0.30; either "
            "side alone breaks monostrategy; substrate has capacity "
            "but typical seed combinations land in a collapsing basin. "
            "(R5) replication_failure -- C0 NOT within [0.58, 0.78] "
            "of EXQ-552's 0.6787; the seed-7 finding does not "
            "replicate; route to stochasticity audit. "
            "experiment_purpose=diagnostic; evidence_direction set to "
            "non_contributory so governance does not weight any claim "
            "from this run."
        ),
        "pass_criteria_summary": summary,
        "per_cell_results": all_results,
        "config": {
            "cells": [
                {"label": label, "env_seed": env_s, "agent_seed": agent_s}
                for label, env_s, agent_s in CELLS
            ],
            "p0_train_episodes": P0_TRAIN_EPISODES,
            "p1_eval_episodes": P1_EVAL_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "env_kwargs": ENV_KWARGS,
        },
    }

    out_file = out_dir / f"{run_id}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Result written to: {out_file}", flush=True)

    return (outcome, str(out_file))


if __name__ == "__main__":
    _outcome, _manifest_path = main()
    if _outcome is not None and _manifest_path is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path)
    sys.exit(0)
