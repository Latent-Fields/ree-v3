"""Canonical OFF/baseline arm of the V3-EXQ-610 (INV-074 crystallization
necessity) lineage -- the content-hashed contract a future 610g MUST build its
OFF arm from.

Design plan: REE_assembly/evidence/planning/arm_reuse_fingerprint_plan.md (7b).

WHY THIS MODULE EXISTS
----------------------
The 610 lineage's OFF/baseline arm (ARM_0_stripped_control in
v3_exq_610f_inv074_crystallization_necessity.py) is the matched seed-shared
reference for the crystallization treatment arm (ARM_1). It is a deliberate
replication: a future 610g would re-train the SAME OFF baseline from scratch.
Arm-reuse (Regime A, same machine-class) lets a later run reuse a recorded OFF
cell instead of re-running it -- but ONLY if the later run's OFF cell is
provably the SAME random variable. The safest narrowing (plan 7b) is a single
*canonical baseline module*: 610g constructs its OFF arm from THIS module, the
mint constructs the OFF arm from THIS module, and the arm fingerprint's
substrate_hash already content-hashes everything under experiments/_lib/**/*.py
(this file included), so ANY change here correctly refuses a stale reuse.

CORRECTNESS CONTRACT (the one failure mode that invalidates reuse is
under-declaration). This module reproduces 610f's ARM_0 OFF path EXACTLY:
  * env config            -- ENV_BASE_KWARGS (verbatim from 610f)
  * training schedule     -- MAX_EPISODES / STEPS_PER_EPISODE + the 4-phase
                             InfantCurriculumScheduler (episode-count gates only;
                             610f calls scheduler.update(episode=ep) with NO
                             telemetry, so phase advancement is deterministic),
                             including the Phase-3 destabilising-pressure env
                             (IGW-20260601-023: SD-047 multi_source + SD-048
                             interoceptive noise + accelerated drift) that the
                             scheduler's env_kwargs(phase=3) injects.
  * substrate-operating / crystallization config -- the OFF arm runs
                             crystallize=False: NO gated_policy.crystallize(),
                             NO residue EWC anchor, residue_ewc_lambda unset
                             (the ewc_penalty() add in the REINFORCE update
                             returns exactly 0 because the anchor is never
                             snapshotted). The phases-0-2 entropy GENERATOR
                             (ENTROPY_BONUS_TRAIN=0.02) runs in the OFF arm too;
                             the Phase-3 entropy FLOOR is 0.0 (no floor).
  * OFF arm flags         -- crystallize=False, entropy_bonus_phase3=0.0,
                             use_noise_floor=False (MECH-313 off),
                             use_e3_diversity=False (MECH-341 off),
                             dacc_suppression_weight=0.0 (MECH-260 off),
                             structured curiosity decoupled (absent).

If 610f's ARM_0 ever changes, this module must change in lockstep (and the
fingerprint will then refuse the stale mint -- safe: a refused reuse just
re-runs the arm; it can never corrupt a result).

This module is import-only (no __main__, no side effects). The mint script
v3_exq_610_inv074_crystallization_baseline_mint.py imports build_off_arm /
train_off_arm / off_path_config_slice from here.

ASCII-only output (repo rule).
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from infant_curriculum import InfantCurriculumScheduler


# Stable identity tag for the canonical OFF baseline (carried into the manifest
# so a reviewer can see which baseline-module version produced a cell).
BASELINE_ID = "exq610_inv074_crystallization_off_v1"
SOURCE_LINEAGE = "V3-EXQ-610"            # OFF arm == 610f ARM_0_stripped_control
SOURCE_SCRIPT = "v3_exq_610f_inv074_crystallization_necessity.py"

# ---------------------------------------------------------------------------
# OFF-arm config (verbatim from v3_exq_610f, ARM_0_stripped_control path)
# ---------------------------------------------------------------------------

# Env config: simple grid with hazards + resources for diversity opportunity.
ENV_BASE_KWARGS = dict(
    size=12,
    num_hazards=3,
    num_resources=4,
    hazard_harm=0.05,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=10,
    use_proxy_fields=True,
)

# Training schedule: 4 phases (0-3) of the infant curriculum.
MAX_EPISODES = 2500
STEPS_PER_EPISODE = 200

# Latent dims.
WORLD_DIM = 32
SELF_DIM = 32
HARM_DIM = 32
HARM_A_DIM = 16
HARM_HISTORY_LEN = 10

# Diversity mechanism weights. For the OFF baseline the noise floor / e3
# diversity are OFF, so NOISE_FLOOR_WEIGHT is never applied (kept for config
# parity with 610f). DACC_SUPPRESSION_WEIGHT=0.0 (MECH-260 anti-recency
# stripped for the true-negative). DACC_WEIGHT=1.0 keeps the DACCtoE3Adapter
# score_bias plumbing (harm/effort biasing; the suppression component is zeroed).
NOISE_FLOOR_WEIGHT = 0.3
DACC_SUPPRESSION_WEIGHT = 0.0
DACC_WEIGHT = 1.0

# Crystallization config: OFF for the baseline (kept for parity; never applied
# because crystallize=False -> xtal_kwargs={} and the anchor is never taken).
RESIDUE_EWC_LAMBDA = 0.1

# ----- REINFORCE policy config -----
GATED_BIAS_SCALE = 2.0
POLICY_TEMP = 1.0
POLICY_GAMMA = 0.95

# PHASE-DEPENDENT entropy bonus. ENTROPY_BONUS_TRAIN is the diversity GENERATOR
# in phases 0-2 (identical in every 610f arm). The OFF baseline's Phase-3 FLOOR
# is 0.0 (ARM_0 entropy_bonus_phase3).
ENTROPY_BONUS_TRAIN = 0.02
OFF_ENTROPY_BONUS_PHASE3 = 0.0

# Learning rates.
LR_E1 = 1e-4
LR_E2_WF = 3e-4
LR_E3_HARM = 1e-3
LR_ENC_AUX = 5e-4
LR_POLICY = 5e-4

# Buffer + batch.
WF_BUF_MAX = 2000
HARM_EVAL_BUF_MAX = 2000
BATCH_SIZE = 32

# Seeds (the lineage's matched seeds).
SEEDS = [42, 43, 44]

# Phase boundary episodes for metric capture (match 610f).
PHASE_2_MIN_EP = 500
PHASE_3_MIN_EP = 2000
ENTROPY_SAMPLE_WINDOW = 50

# The OFF arm flag set (ARM_0_stripped_control), for the fingerprint slice.
OFF_ARM = {
    "label": "ARM_0_stripped_control",
    "crystallize": False,
    "entropy_bonus_phase3": OFF_ENTROPY_BONUS_PHASE3,
    "use_noise_floor": False,
    "use_e3_diversity": False,
}


# ---------------------------------------------------------------------------
# Helpers (verbatim from v3_exq_610f)
# ---------------------------------------------------------------------------

def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def compute_action_entropy(action_counts: Counter) -> float:
    """Shannon entropy of action distribution (nats)."""
    total = sum(action_counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in action_counts.values()]
    return float(-sum(p * np.log(p + 1e-12) for p in probs if p > 0))


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
    return float(rv.max().item()) if isinstance(rv, torch.Tensor) else float(np.max(rv))


def _build_candidate_inputs(candidates, latent, device):
    """Per-candidate first-step z_world summary [K, world_dim] + first-action
    one-hots [K, action_dim] (mirrors the agent.select_action gated_policy block)."""
    gp_list: List[torch.Tensor] = []
    fa_list: List[torch.Tensor] = []
    for c in candidates:
        if c.world_states is not None:
            ws = c.get_world_state_sequence()  # [1, horizon, world_dim]
            gp_list.append(ws[0, 0, :].detach())
        else:
            gp_list.append(latent.z_world[0].detach())
        fa_list.append(c.actions[:, 0, :][0].detach().float())
    gp_summaries = torch.stack(gp_list, dim=0).detach().to(device)  # [K, world_dim]
    first_action_onehots = torch.stack(fa_list, dim=0).to(device)  # [K, action_dim]
    return gp_summaries, first_action_onehots


def _gated_policy_select(agent, latent, candidates, device):
    """Build a differentiable Categorical over candidates from the gated_policy
    bias, sample, and return (action, log_prob, entropy, sel_idx)."""
    K = len(candidates)
    if K == 0:
        return None, None, None, -1
    gp_summaries, first_action_onehots = _build_candidate_inputs(
        candidates, latent, device,
    )
    use_onehot = bool(getattr(agent.gated_policy.config, "use_first_action_onehot", False))
    # Detach the latent streams: REINFORCE trains ONLY the gated_policy params.
    _zw = latent.z_world.detach()
    _zs = latent.z_self.detach()
    _za = latent.z_harm_a.detach() if latent.z_harm_a is not None else None
    gp_out = agent.gated_policy(
        z_world=_zw,
        z_self=_zs,
        z_harm_a=_za,
        candidate_features=gp_summaries,
        first_action_onehots=first_action_onehots if use_onehot else None,
        simulation_mode=False,
    )
    gp_bias = gp_out.gated_score_bias  # [K], differentiable
    # E3 lower-is-better: lower bias = more preferred -> logits = -bias / T.
    logits = -gp_bias / POLICY_TEMP
    log_probs_all = F.log_softmax(logits, dim=0)
    if K >= 2:
        dist = Categorical(logits=logits)
        sel_idx = int(dist.sample().item())
        entropy_t = -(log_probs_all.exp() * log_probs_all).sum()
    else:
        sel_idx = 0
        entropy_t = torch.zeros((), device=device)
    log_prob_sel = log_probs_all[sel_idx]
    action = candidates[sel_idx].actions[:, 0, :].to(device)  # [1, action_dim]
    return action, log_prob_sel, entropy_t, sel_idx


def _reinforce_update(
    policy_optimizer,
    policy_param_list,
    ep_log_probs,
    ep_entropies,
    ep_rewards,
    ewc_penalty_fn,
    in_phase3: bool,
    entropy_bonus_weight: float,
    device,
) -> Tuple[int, float]:
    """REINFORCE policy update at episode end.

    entropy_bonus_weight is PHASE-DEPENDENT (caller passes ENTROPY_BONUS_TRAIN in
    phases 0-2; OFF_ENTROPY_BONUS_PHASE3 == 0.0 in phase 3 for the OFF arm).
    ewc_penalty_fn() returns exactly 0 for the OFF arm (no anchor snapshotted),
    so the in_phase3 add is a safe no-op.

    Returns (stepped:int 0/1, ewc_penalty_value:float).
    """
    if not ep_log_probs:
        return 0, 0.0
    # Discounted returns.
    returns: List[float] = []
    G = 0.0
    for r in reversed(ep_rewards):
        G = float(r) + POLICY_GAMMA * G
        returns.insert(0, G)
    returns_t = torch.tensor(returns, device=device, dtype=torch.float32)
    advantages = returns_t - returns_t.mean()  # mean-baseline advantage
    if advantages.numel() > 1 and float(advantages.std()) > 1e-6:
        advantages = advantages / (advantages.std() + 1e-8)
    log_probs_t = torch.stack(ep_log_probs)  # [T] (grad)
    entropies_t = torch.stack(ep_entropies)  # [T] (grad)
    policy_loss = -(log_probs_t * advantages.detach()).sum()
    entropy_bonus = -entropy_bonus_weight * entropies_t.sum()  # subtract entropy = maximize it
    total_loss = policy_loss + entropy_bonus

    ewc_val = 0.0
    if in_phase3:
        ewc_term = ewc_penalty_fn()
        total_loss = total_loss + ewc_term
        ewc_val = float(ewc_term.detach())

    policy_optimizer.zero_grad()
    total_loss.backward()
    if policy_param_list:
        torch.nn.utils.clip_grad_norm_(policy_param_list, 1.0)
    policy_optimizer.step()
    return 1, ewc_val


# ---------------------------------------------------------------------------
# OFF-arm agent + env factory (crystallize=False; floors OFF)
# ---------------------------------------------------------------------------

def build_off_arm(
    seed: int,
    grid_size: Optional[int] = None,
) -> Tuple[REEAgent, CausalGridWorldV2, InfantCurriculumScheduler]:
    """Build agent + env + scheduler for the V3-EXQ-610 OFF/baseline arm.

    This is 610f._make_agent_and_env specialised to ARM_0_stripped_control:
    crystallize=False (xtal_kwargs={}), use_noise_floor=False (MECH-313 off),
    use_e3_diversity=False (MECH-341 off), dacc_suppression_weight=0.0.
    The on_phase3_entry crystallization hook is None (crystallize=False), so the
    scheduler never crystallizes the gated policy nor snapshots the EWC anchor.
    """
    if grid_size is None:
        grid_size = ENV_BASE_KWARGS["size"]

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    env = CausalGridWorldV2(seed=seed, **ENV_BASE_KWARGS)

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
        # Policy substrate (both arms).
        use_gated_policy=True,
        gated_policy_use_differential_heads=True,  # ARC-062 fix.
        gated_policy_use_first_action_onehot=True,  # ARC-062 head input.
        use_dacc=True,
        dacc_weight=DACC_WEIGHT,
        dacc_suppression_weight=DACC_SUPPRESSION_WEIGHT,  # 0.0 -- stripped.
        # MECH-313 noise floor OFF for the OFF baseline.
        use_noise_floor=False,
        noise_floor_weight=0.0,
        # MECH-341 E3 score diversity OFF for the OFF baseline.
        use_e3_score_diversity=False,
        use_e3_diversity_entropy_bonus=False,
        use_e3_diversity_stratified_select=False,
        # Crystallization kwargs: NONE (OFF arm; crystallize=False).
    )
    config.e3.commitment_threshold = 0.5
    config.heartbeat.beta_gate_bistable = True
    config.harm_descending_mod_enabled = True
    config.descending_attenuation_factor = 0.5
    config.gated_policy_bias_scale = GATED_BIAS_SCALE

    agent = REEAgent(config)

    # OFF arm: no crystallization hook (crystallize=False).
    scheduler = InfantCurriculumScheduler(
        grid_size=grid_size,
        on_phase3_entry=None,
    )

    return agent, env, scheduler


# ---------------------------------------------------------------------------
# OFF-arm training loop (verbatim from 610f, specialised to the OFF arm)
# ---------------------------------------------------------------------------

def train_off_arm(
    agent: REEAgent,
    scheduler: InfantCurriculumScheduler,
    seed: int,
    dry_run: bool = False,
) -> Dict:
    """Run the OFF/baseline arm training (610f._train_infant_curriculum with
    crystallize=False, entropy_bonus_phase3=0.0). Returns the per-cell metrics
    dict (same shape as 610f arm_results rows, with arm/seed/flags annotated)."""
    crystallize = False
    entropy_bonus_phase3 = OFF_ENTROPY_BONUS_PHASE3
    device = agent.device

    # Encoder / aux / forward-model optimizers (substrate training; unchanged).
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

    # REINFORCE policy optimizer over the gated_policy parameters.
    policy_param_list = (
        [p for p in agent.gated_policy.parameters() if p.requires_grad]
        if agent.gated_policy is not None else []
    )
    policy_optimizer = optim.Adam(policy_param_list, lr=LR_POLICY)
    expansion_optimizer_active = False  # never True for the OFF arm (crystallize=False).

    def _ewc_penalty_fn():
        return agent.residue_field.ewc_penalty()

    # Buffers.
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_eval_buf: List[Tuple[torch.Tensor, torch.Tensor]] = []

    # Metrics.
    reward_log: List[float] = []
    action_history: List[int] = []

    end_phase_2_entropy: Optional[float] = None
    end_phase_3_entropy: Optional[float] = None
    phase_2_capture_started = False
    phase_3_capture_started = False

    # Wiring counters (manifest-visible).
    n_policy_steps = 0
    n_policy_steps_phase3 = 0
    n_expansion_steps_phase3 = 0
    n_ewc_terms_phase3 = 0
    ewc_penalty_last = 0.0
    n_random_fallback_steps = 0

    agent.train()
    max_eps = 5 if dry_run else MAX_EPISODES

    for ep in range(max_eps):
        env_kwargs = {**ENV_BASE_KWARGS, **scheduler.env_kwargs()}
        env = CausalGridWorldV2(seed=seed + ep, **env_kwargs)
        action_dim = env.action_dim

        flat_obs, obs_dict = env.reset()
        agent.reset()

        z_world_prev: Optional[torch.Tensor] = None
        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        ep_reward = 0.0

        ep_log_probs: List[torch.Tensor] = []
        ep_entropies: List[torch.Tensor] = []
        ep_rewards: List[float] = []

        in_phase3 = scheduler.current_phase == 3
        ep_entropy_bonus = entropy_bonus_phase3 if in_phase3 else ENTROPY_BONUS_TRAIN

        steps_this_ep = 20 if dry_run else STEPS_PER_EPISODE
        for step_idx in range(steps_this_ep):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_h = _obs_harm(obs_dict)
            obs_h_a = _obs_harm_a(obs_dict)
            obs_h_h = _obs_harm_history(obs_dict)
            prox_t = _obs_resource_prox(obs_dict)
            accum_t = _obs_accum(obs_dict)

            latent = agent.sense(
                obs_body, obs_world,
                obs_harm=obs_h, obs_harm_a=obs_h_a, obs_harm_history=obs_h_h,
            )
            z_world_curr = latent.z_world.detach()

            aux_terms: List[torch.Tensor] = []
            prox_target_t = torch.tensor([[prox_t]], device=device)
            prox_loss = agent.compute_resource_proximity_loss(prox_target_t, latent)
            if prox_loss is not None and prox_loss.requires_grad:
                aux_terms.append(prox_loss)
            accum_target_t = torch.tensor([[accum_t]], device=device)
            harm_accum_loss = agent.compute_harm_accum_loss(accum_target_t, latent)
            if harm_accum_loss is not None and harm_accum_loss.requires_grad:
                aux_terms.append(harm_accum_loss)
            if aux_terms:
                aux_loss = sum(aux_terms)
                aux_optimizer.zero_grad()
                aux_loss.backward(retain_graph=False)
                torch.nn.utils.clip_grad_norm_(aux_params, 1.0)
                aux_optimizer.step()

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            latent = agent.sense(
                obs_body, obs_world,
                obs_harm=obs_h, obs_harm_a=obs_h_a, obs_harm_history=obs_h_h,
            )

            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, WORLD_DIM, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            drive_level = REEAgent.compute_drive_level(obs_body)
            benefit_exposure = max(0.0, float(obs_dict.get("benefit_exposure", 0.0)))
            agent.update_z_goal(
                benefit_exposure=benefit_exposure,
                drive_level=drive_level,
            )

            action, log_prob_sel, entropy_t, sel_idx = _gated_policy_select(
                agent, latent, candidates, device,
            )
            if action is None:
                action = _action_to_onehot(
                    random.randint(0, action_dim - 1), action_dim, device,
                )
                agent._last_action = action
                n_random_fallback_steps += 1
            else:
                agent._last_action = action
                ep_log_probs.append(log_prob_sel)
                ep_entropies.append(entropy_t)

            action_idx = int(torch.argmax(action).item())
            action_history.append(action_idx)

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ep_reward += float(harm_signal)
            ep_rewards.append(float(harm_signal))

            agent.update_residue(harm_signal=float(harm_signal))

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > WF_BUF_MAX:
                    wf_buf = wf_buf[-WF_BUF_MAX:]

            harm_target = abs(float(harm_signal)) if float(harm_signal) < 0 else 0.0
            harm_eval_buf.append((z_world_curr.cpu(), torch.tensor([harm_target])))
            if len(harm_eval_buf) > HARM_EVAL_BUF_MAX:
                harm_eval_buf = harm_eval_buf[-HARM_EVAL_BUF_MAX:]

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
                        + list(agent.e2.world_action_encoder.parameters()), 1.0,
                    )
                    e2_wf_optimizer.step()
                with torch.no_grad():
                    agent.e3.update_running_variance(
                        (wf_pred.detach() - zw1_b).detach()
                    )

            if len(harm_eval_buf) >= BATCH_SIZE:
                idxs = torch.randperm(len(harm_eval_buf))[:BATCH_SIZE].tolist()
                zw_b = torch.cat([harm_eval_buf[i][0] for i in idxs]).to(device)
                ht_b = torch.cat([harm_eval_buf[i][1] for i in idxs]).to(device)
                hp = agent.e3.harm_eval(zw_b)
                he_loss = F.mse_loss(hp.squeeze(), ht_b.squeeze())
                if he_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    he_loss.backward()
                    harm_eval_optimizer.step()

            if len(agent._world_experience_buffer) >= 2:
                e1_loss = agent.compute_prediction_loss()
                if e1_loss.requires_grad:
                    e1_optimizer.zero_grad()
                    e1_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                    e1_optimizer.step()

            z_world_prev = z_world_curr
            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()

            if done:
                break

        stepped, ewc_val = _reinforce_update(
            policy_optimizer=policy_optimizer,
            policy_param_list=policy_param_list,
            ep_log_probs=ep_log_probs,
            ep_entropies=ep_entropies,
            ep_rewards=ep_rewards,
            ewc_penalty_fn=_ewc_penalty_fn,
            in_phase3=in_phase3,
            entropy_bonus_weight=ep_entropy_bonus,
            device=device,
        )
        if stepped:
            n_policy_steps += 1
            if in_phase3:
                n_policy_steps_phase3 += 1
                if expansion_optimizer_active:
                    n_expansion_steps_phase3 += 1
                if ewc_val > 0.0:
                    n_ewc_terms_phase3 += 1
                    ewc_penalty_last = ewc_val

        reward_log.append(ep_reward)

        # Advance the curriculum (no telemetry -> episode-count gates only).
        scheduler.update(episode=ep)

        # Capture entropy at phase boundaries (selected-action distribution).
        if ep >= (PHASE_3_MIN_EP - ENTROPY_SAMPLE_WINDOW) and ep < PHASE_3_MIN_EP:
            phase_2_capture_started = True
        if phase_2_capture_started and ep == PHASE_3_MIN_EP - 1:
            recent_actions = action_history[-(ENTROPY_SAMPLE_WINDOW * STEPS_PER_EPISODE):]
            end_phase_2_entropy = compute_action_entropy(Counter(recent_actions))
            phase_2_capture_started = False

        if ep >= (max_eps - ENTROPY_SAMPLE_WINDOW):
            phase_3_capture_started = True
        if phase_3_capture_started and ep == max_eps - 1:
            recent_actions = action_history[-(ENTROPY_SAMPLE_WINDOW * STEPS_PER_EPISODE):]
            end_phase_3_entropy = compute_action_entropy(Counter(recent_actions))

        if (ep + 1) % 100 == 0 or ep == max_eps - 1 or scheduler.phase_changed:
            print(
                f"  [train] {OFF_ARM['label']} seed={seed} ep {ep+1}/{max_eps} "
                f"phase={scheduler.current_phase} "
                f"ep_eb={ep_entropy_bonus:.4f} "
                f"ep_reward={ep_reward:.4f} "
                f"rv={agent.e3._running_variance:.4f} "
                f"pol_steps={n_policy_steps}",
                flush=True,
            )

        # OFF arm: crystallize=False -> the expansion-rebuild block never fires.

    return {
        "arm": OFF_ARM["label"],
        "mean_reward": float(np.mean(reward_log)) if reward_log else 0.0,
        "final_phase": scheduler.current_phase,
        "end_phase_2_entropy": end_phase_2_entropy,
        "end_phase_3_entropy": end_phase_3_entropy,
        "total_episodes": len(reward_log),
        "entropy_bonus_phase3": entropy_bonus_phase3,
        "n_policy_steps": n_policy_steps,
        "n_policy_steps_phase3": n_policy_steps_phase3,
        "n_expansion_steps_phase3": n_expansion_steps_phase3,
        "n_ewc_terms_phase3": n_ewc_terms_phase3,
        "ewc_penalty_last": ewc_penalty_last,
        "n_random_fallback_steps": n_random_fallback_steps,
        "crystallized": bool(getattr(agent.gated_policy, "crystallized", False)),
        "seed": seed,
        "crystallize": crystallize,
        "use_noise_floor": OFF_ARM["use_noise_floor"],
        "use_e3_diversity": OFF_ARM["use_e3_diversity"],
    }


# ---------------------------------------------------------------------------
# Fingerprint config slice (the OFF-path config the cell actually reads)
# ---------------------------------------------------------------------------

def off_path_config_slice() -> Dict:
    """The deliberately-narrowed OFF-path config slice for compute_arm_fingerprint
    (plan decision 3 opt-in narrowing; plan 7b canonical-baseline-module form).

    Contains ONLY what the OFF computation reads: env config, training schedule,
    the substrate-operating config that fires for the OFF arm (entropy generator,
    dacc plumbing, learning rates, buffers), and the OFF arm's own flags. It does
    NOT contain ON-arm gains (crystallization kwargs, noise-floor / e3-diversity
    weights), acceptance thresholds, or labels -- none of which the OFF cell reads.

    The baseline-module CONTENT itself is bound separately via the fingerprint's
    substrate_hash (experiments/_lib/**/*.py is content-hashed), so any logic
    change here is caught even if a key is omitted from this slice.
    """
    return {
        "baseline_id": BASELINE_ID,
        "source_lineage": SOURCE_LINEAGE,
        "env_base_kwargs": dict(ENV_BASE_KWARGS),
        "schedule": {
            "max_episodes": MAX_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "phase_2_min_ep": PHASE_2_MIN_EP,
            "phase_3_min_ep": PHASE_3_MIN_EP,
            "entropy_sample_window": ENTROPY_SAMPLE_WINDOW,
            "curriculum": "InfantCurriculumScheduler(episode-count gates only)",
        },
        "latent_dims": {
            "world_dim": WORLD_DIM,
            "self_dim": SELF_DIM,
            "harm_dim": HARM_DIM,
            "harm_a_dim": HARM_A_DIM,
            "harm_history_len": HARM_HISTORY_LEN,
        },
        "substrate_operating": {
            "dacc_weight": DACC_WEIGHT,
            "dacc_suppression_weight": DACC_SUPPRESSION_WEIGHT,
            "gated_bias_scale": GATED_BIAS_SCALE,
            "policy_temp": POLICY_TEMP,
            "policy_gamma": POLICY_GAMMA,
            "entropy_bonus_train_phases_0_2": ENTROPY_BONUS_TRAIN,
            "lr_e1": LR_E1,
            "lr_e2_wf": LR_E2_WF,
            "lr_e3_harm": LR_E3_HARM,
            "lr_enc_aux": LR_ENC_AUX,
            "lr_policy": LR_POLICY,
            "wf_buf_max": WF_BUF_MAX,
            "harm_eval_buf_max": HARM_EVAL_BUF_MAX,
            "batch_size": BATCH_SIZE,
            "alpha_world": 0.9,
            "alpha_self": 0.3,
            "drive_weight": 2.0,
        },
        "off_arm_flags": dict(OFF_ARM),
    }


__all__ = [
    "BASELINE_ID",
    "SOURCE_LINEAGE",
    "SOURCE_SCRIPT",
    "ENV_BASE_KWARGS",
    "MAX_EPISODES",
    "STEPS_PER_EPISODE",
    "SEEDS",
    "OFF_ARM",
    "compute_action_entropy",
    "build_off_arm",
    "train_off_arm",
    "off_path_config_slice",
]
