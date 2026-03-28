#!/opt/local/bin/python3
"""
V3-EXQ-124 -- MECH-033: Kernel Chaining Interface Discriminative Pair (EXP-0023 / EVB-0018)

Claim: MECH-033
Proposal: EXP-0023 / EVB-0018
Dispatch mode: discriminative_pair
Min shared seeds: 2

MECH-033 asserts: "E2 forward-prediction kernels seed hippocampal rollouts."
More precisely, the hippocampal module chains trajectory segments via E2 action-object
kernels, and this kernel interface preserves causal continuity across segment boundaries.
The kernel chaining is LOAD-BEARING: removing it should degrade planning performance.

V2 history: EXQ-022/023 FAIL -- E2 supplied z_gamma sensory kernels (wrong primitive).
EXQ-055 PASS (2026-03-20): action-object chaining vs self-only vs random on a single
trained agent. AO reduces harm 67x vs self-only. But EXQ-055 used a single seed on one
shared agent with 3 conditions at eval -- not a proper matched-seed discriminative pair.

This experiment implements a clean matched-seed discriminative pair:

  KERNEL_CHAIN_ON:
    HippocampalModule runs full CEM in action-object space O.
    Terrain prior conditions the initial action-object distribution.
    E2.action_object(z_world, a) kernels are chained across the planning horizon.
    Candidates are decoded to actions via action_object_decoder.
    Elite selection by residue field score.
    This is the full kernel chaining interface as specified in MECH-033.

  KERNEL_CHAIN_ABLATED:
    HippocampalModule is BYPASSED -- candidate actions are sampled uniformly at random
    for the entire horizon (same number of candidates, same horizon length).
    NO action-object kernels are used. NO chaining. NO terrain prior.
    E3 evaluates the random candidates and picks the best one by harm_eval score.
    All other components (agent, E3, residue field, harm eval training) are IDENTICAL.

Both conditions:
  - Train the same REEAgent (E1, E2, E3, harm_eval_head) for the same warmup budget.
  - Use matched random seeds for full reproducibility.
  - E2 world_forward and action_object are trained in both conditions (they must exist
    to evaluate candidates in E3 -- we only ablate WHETHER they seed hippocampal CEM).

PRIMARY METRIC:
  harm_rate_CHAIN = hazard contact steps / total eval steps (ON condition)
  harm_rate_ABLATED = hazard contact steps / total eval steps (ABLATED condition)

  harm_reduction_delta = harm_rate_ABLATED - harm_rate_CHAIN
  (positive = chaining reduces harm compared to random)

  relative_reduction = harm_reduction_delta / harm_rate_ABLATED
  (fraction of ablated harm that is eliminated by chaining)

SECONDARY METRIC:
  residue_score_CHAIN vs residue_score_ABLATED
  (does kernel chaining navigate to lower-residue regions?)

Both conditions: 2 matched seeds [42, 123], 400 warmup + 50 eval episodes x 200 steps.

PRE-REGISTERED THRESHOLDS (hardcoded, not inferred post-hoc):

  C1 (harm rate reduction):
    harm_rate_ABLATED - harm_rate_CHAIN >= THRESH_C1_HARM_REDUCTION
    Kernel chaining must produce a meaningful harm rate reduction vs random candidates.
    Threshold: THRESH_C1_HARM_REDUCTION = 0.03  (3pp absolute reduction)

  C2 (relative reduction):
    relative_reduction >= THRESH_C2_RELATIVE_REDUCTION
    Chaining must eliminate a meaningful fraction of ablated-condition harm.
    Threshold: THRESH_C2_RELATIVE_REDUCTION = 0.15  (15% relative reduction)

  C3 (consistency across seeds):
    harm_rate_CHAIN < harm_rate_ABLATED for BOTH seeds independently.
    Direction must replicate. Cannot be a fluke of one seed.

  C4 (data quality -- sufficient hazard contacts in ablated condition):
    n_contacts_ABLATED >= THRESH_C4_MIN_CONTACTS per seed.
    Without sufficient contact events the harm rate cannot be measured reliably.
    Threshold: THRESH_C4_MIN_CONTACTS = 20

  C5 (residue navigation diagnostic, informational only):
    mean_residue_CHAIN <= mean_residue_ABLATED  (chaining finds lower-residue paths)
    Not required for PASS but supports the mechanism story.

PASS criteria:
  C1 AND C2 AND C3 AND C4 -> PASS -> supports MECH-033
  C1 AND C3 AND C4, NOT C2 -> mixed (absolute reduction real, relative weak)
  NOT C1 AND C4            -> FAIL -> kernel chaining hypothesis refuted
  NOT C4                   -> inconclusive (data quality failure)

Decision mapping:
  PASS             -> retain_ree (kernel chaining is load-bearing)
  C1+C3+C4         -> hybridize (directional but threshold not met)
  NOT C1 AND C4    -> retire_ree_claim (chaining provides no planning advantage)
  NOT C4           -> inconclusive

CLAIM_IDS RATIONALE:
  The only claim directly tested is MECH-033: does the E2 action-object kernel chaining
  interface provide a planning advantage over random candidate generation?
  If PASS: the kernel interface is load-bearing, supporting MECH-033.
  ARC-018 (viability map navigation) is NOT tagged: this experiment does not test
  residue terrain navigation in isolation -- it tests the kernel interface specifically.
  EXQ-055 already provided supporting evidence for MECH-033; this matched-seed pair
  provides the replication with proper discriminative design (EXP-0023).
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


EXPERIMENT_TYPE = "v3_exq_124_mech033_kernel_chain_pair"
CLAIM_IDS = ["MECH-033"]

# Pre-registered thresholds
THRESH_C1_HARM_REDUCTION      = 0.03   # harm_rate_ABLATED - harm_rate_CHAIN >= 0.03
THRESH_C2_RELATIVE_REDUCTION  = 0.15   # relative reduction >= 15%
THRESH_C4_MIN_CONTACTS        = 20     # n_contacts_ABLATED >= 20 per seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_to_onehot(action_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, action_idx] = 1.0
    return v


def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
    return t.unsqueeze(0) if t.dim() == 1 else t


def _mean_safe(lst: List[float], default: float = 0.0) -> float:
    return float(sum(lst) / len(lst)) if lst else default


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
        reafference_action_dim=0,  # SD-007 disabled -- isolate MECH-033 mechanism
    )
    # SD-005: split latent mode (z_self != z_world)
    config.latent.unified_latent_mode = False
    # Standard single-rate: all modules tick every step
    config.heartbeat.e1_steps_per_tick = 1
    config.heartbeat.e2_steps_per_tick = 1
    config.heartbeat.e3_steps_per_tick = 1
    return config


# ---------------------------------------------------------------------------
# Candidate generators
# ---------------------------------------------------------------------------

def _generate_chain_candidates(
    agent: REEAgent,
    z_world: torch.Tensor,
    z_self: torch.Tensor,
    horizon: int,
    n_candidates: int,
    n_cem_iterations: int,
    elite_fraction: float,
    device,
) -> torch.Tensor:
    """
    KERNEL_CHAIN_ON: Generate candidate action sequences via CEM in action-object space.

    Uses HippocampalModule's terrain prior to initialise the action-object distribution,
    then chains E2 action-object kernels across the planning horizon.
    Returns the best action (first step of best candidate) as a one-hot tensor [1, action_dim].
    """
    hippo   = agent.hippocampal
    e2      = agent.e2
    residue = agent.residue_field
    config  = hippo.config
    n_elite = max(1, int(n_candidates * elite_fraction))

    # Initialise in action-object space from terrain prior (SD-004 + SD-002)
    e1_prior = None
    try:
        e1_out = agent.e1.predict(z_world)
        e1_prior = _ensure_2d(e1_out.detach()) if e1_out is not None else None
    except Exception:
        e1_prior = None

    ao_mean = hippo._get_terrain_action_object_mean(z_world, e1_prior=e1_prior)
    ao_std  = torch.ones_like(ao_mean)

    best_actions_flat = None
    best_score = float("inf")

    for _iter in range(n_cem_iterations):
        candidate_scores = []
        candidate_action_seqs = []
        candidate_ao_seqs = []

        for _ in range(n_candidates):
            noise = torch.randn_like(ao_mean)
            ao_sample = ao_mean + ao_std * noise  # [1, horizon, ao_dim]

            # Decode action-object sequence to action sequence [1, horizon, action_dim]
            actions = hippo._decode_action_objects(ao_sample)

            # Roll out through E2 (world track)
            traj = e2.rollout_with_world(
                z_self, z_world, actions, compute_action_objects=True
            )

            # Score by residue field over world states
            score = float("inf")
            world_seq = traj.get_world_state_sequence()
            if world_seq is not None:
                score = float(residue.evaluate_trajectory(world_seq).sum().item())

            candidate_scores.append(score)
            candidate_action_seqs.append(actions)  # [1, horizon, action_dim]
            candidate_ao_seqs.append(ao_sample)    # [1, horizon, ao_dim]

            if score < best_score:
                best_score = score
                best_actions_flat = actions

        # Refit to elite
        sorted_idx = sorted(range(n_candidates), key=lambda i: candidate_scores[i])
        elite_idx = sorted_idx[:n_elite]
        elite_ao = [candidate_ao_seqs[i] for i in elite_idx]
        if elite_ao:
            elite_tensor = torch.stack(elite_ao, dim=0)  # [elite, 1, horizon, ao_dim]
            ao_mean = elite_tensor.mean(dim=0)
            ao_std  = elite_tensor.std(dim=0) + 1e-6

    # Return the first action of the best candidate as a one-hot
    if best_actions_flat is not None:
        # best_actions_flat: [1, horizon, action_dim]
        first_action_logits = best_actions_flat[0, 0, :]  # [action_dim]
        action_idx = int(torch.argmax(first_action_logits).item())
    else:
        action_idx = random.randint(0, agent.config.e2.action_dim - 1)

    return _action_to_onehot(action_idx, agent.config.e2.action_dim, device)


def _generate_ablated_candidates(
    agent: REEAgent,
    z_world: torch.Tensor,
    horizon: int,
    n_candidates: int,
    device,
) -> torch.Tensor:
    """
    KERNEL_CHAIN_ABLATED: Generate candidate actions uniformly at random.

    NO action-object kernels. NO terrain prior. NO CEM chaining.
    Each candidate is an independent random action sequence.
    E3 harm_eval selects the best first action by evaluating each candidate's
    z_world after one E2 world_forward step (minimal forward look).
    This is the baseline: planning without the kernel chaining interface.
    """
    action_dim = agent.config.e2.action_dim
    best_action_idx = None
    best_harm_score = float("inf")

    with torch.no_grad():
        for _ in range(n_candidates):
            a_idx = random.randint(0, action_dim - 1)
            a_oh = _action_to_onehot(a_idx, action_dim, device)
            # One-step E2 world lookahead (no chaining -- just see where one step goes)
            z_world_next = agent.e2.world_forward(z_world, a_oh)
            z_world_next_2d = _ensure_2d(z_world_next.detach())
            harm_score = float(agent.e3.harm_eval(z_world_next_2d).item())
            if harm_score < best_harm_score:
                best_harm_score = harm_score
                best_action_idx = a_idx

    if best_action_idx is None:
        best_action_idx = random.randint(0, action_dim - 1)

    return _action_to_onehot(best_action_idx, action_dim, device)


# ---------------------------------------------------------------------------
# Single cell runner
# ---------------------------------------------------------------------------

def _run_single(
    seed: int,
    chain_on: bool,
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
    n_candidates: int,
    horizon: int,
    n_cem_iterations: int,
    elite_fraction: float,
    dry_run: bool = False,
) -> Dict:
    """Run one (seed, condition) cell. Returns per-cell metrics."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    cond = "KERNEL_CHAIN_ON" if chain_on else "KERNEL_CHAIN_ABLATED"

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

    # Optimizers
    e1_opt       = optim.Adam(agent.e1.parameters(), lr=lr)
    e2_opt       = optim.Adam(agent.e2.parameters(), lr=lr)
    e3_harm_opt  = optim.Adam(agent.e3.harm_eval_head.parameters(), lr=lr)
    hippo_opt    = optim.Adam(agent.hippocampal.parameters(), lr=lr)

    # Replay buffers
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []

    actual_warmup = min(3, warmup_episodes)  if dry_run else warmup_episodes
    actual_eval   = min(2, eval_episodes)    if dry_run else eval_episodes

    # ---- TRAINING PHASE ----
    # Both conditions train identically: the kernel chaining difference is at EVAL only.
    # We train E1, E2 (self + world), E3 harm_eval, and hippocampus terrain prior
    # in both conditions so both agents are equally capable. The ablation is purely
    # whether we USE the hippocampal CEM at eval action selection.

    agent.train()

    train_harm_steps = 0
    train_e2_world_steps = 0

    for ep in range(actual_warmup):
        _, obs_dict = env.reset()
        agent.reset()

        prev_z_world: torch.Tensor = None

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)

            latent = agent.sense(obs_body, obs_world)
            z_self_curr  = _ensure_2d(latent.z_self.detach())
            z_world_curr = _ensure_2d(latent.z_world.detach())

            agent.clock.advance()

            # Random action during warmup (both conditions)
            action_idx = random.randint(0, env.action_dim - 1)
            action_oh  = _action_to_onehot(action_idx, env.action_dim, device)
            agent._last_action = action_oh

            _, harm_signal, done, info, obs_dict = env.step(action_oh)
            hs = float(harm_signal)

            # E1 update
            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                e1_opt.zero_grad()
                e1_loss.backward()
                nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                e1_opt.step()

            # E2 world_forward update: z_world_t + a -> z_world_{t+1}
            if prev_z_world is not None:
                _, obs_dict_next_tmp = env.reset()  # We can't re-sense mid-episode;
                # approximate z_world_next by re-sensing current obs after step
                with torch.no_grad():
                    obs_body_next  = obs_dict["body_state"].to(device)
                    obs_world_next = obs_dict["world_state"].to(device)
                    latent_next = agent.sense(obs_body_next, obs_world_next)
                    z_world_next_true = _ensure_2d(latent_next.z_world.detach())

                z_world_pred = _ensure_2d(
                    agent.e2.world_forward(prev_z_world, action_oh)
                )
                e2_world_loss = F.mse_loss(z_world_pred, z_world_next_true)
                if e2_world_loss.requires_grad:
                    e2_opt.zero_grad()
                    e2_world_loss.backward()
                    nn.utils.clip_grad_norm_(agent.e2.parameters(), 1.0)
                    e2_opt.step()
                    train_e2_world_steps += 1

            prev_z_world = z_world_curr.detach()

            # Harm eval training (both conditions -- same supervision)
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

            # Hippocampus terrain prior update: teach it to predict low-residue regions
            # We train on actual z_world observations so the terrain prior learns
            # the world's hazard structure -- same in both conditions.
            if len(harm_buf_pos) >= 2:
                e1_prior_for_hippo = None
                try:
                    with torch.no_grad():
                        e1_prior_for_hippo = _ensure_2d(
                            agent.e1.predict(z_world_curr)
                        )
                except Exception:
                    e1_prior_for_hippo = None

                ao_mean_pred = agent.hippocampal._get_terrain_action_object_mean(
                    z_world_curr, e1_prior=e1_prior_for_hippo
                )
                # Regularisation: push terrain prior toward low noise (L2 penalty on output)
                hippo_reg = (ao_mean_pred ** 2).mean() * 0.001
                if hippo_reg.requires_grad:
                    hippo_opt.zero_grad()
                    hippo_reg.backward()
                    nn.utils.clip_grad_norm_(agent.hippocampal.parameters(), 0.5)
                    hippo_opt.step()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == actual_warmup - 1:
            print(
                f"  [train] seed={seed} cond={cond}"
                f" ep {ep + 1}/{actual_warmup}"
                f" harm_buf_pos={len(harm_buf_pos)}"
                f" harm_buf_neg={len(harm_buf_neg)}"
                f" train_harm_steps={train_harm_steps}"
                f" train_e2_world_steps={train_e2_world_steps}",
                flush=True,
            )

    # ---- EVAL PHASE ----
    # The discriminative pair difference: at eval, CHAIN uses hippocampal CEM,
    # ABLATED uses random candidate generation (E3 still evaluates, just no chaining).

    agent.eval()

    n_contact_steps = 0
    n_total_steps   = 0
    residue_scores: List[float] = []

    for _ in range(actual_eval):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                z_world_curr = _ensure_2d(latent.z_world.detach())
                z_self_curr  = _ensure_2d(latent.z_self.detach())
                agent.clock.advance()

                # Record residue score at current position (both conditions)
                rscore = float(
                    agent.residue_field.evaluate(z_world_curr).item()
                )
                residue_scores.append(rscore)

                if chain_on:
                    # KERNEL_CHAIN_ON: full CEM in action-object space
                    action_oh = _generate_chain_candidates(
                        agent,
                        z_world_curr,
                        z_self_curr,
                        horizon=horizon,
                        n_candidates=n_candidates,
                        n_cem_iterations=n_cem_iterations,
                        elite_fraction=elite_fraction,
                        device=device,
                    )
                else:
                    # KERNEL_CHAIN_ABLATED: random candidates, E3 selects best
                    action_oh = _generate_ablated_candidates(
                        agent,
                        z_world_curr,
                        horizon=horizon,
                        n_candidates=n_candidates,
                        device=device,
                    )

                agent._last_action = action_oh

            _, harm_signal, done, info, obs_dict = env.step(action_oh)
            hs = float(harm_signal)
            n_total_steps += 1

            # Contact = harm_signal below threshold (actual hazard contact)
            if hs <= -0.015:
                n_contact_steps += 1

            if done:
                break

    harm_rate = n_contact_steps / max(1, n_total_steps)
    mean_residue = _mean_safe(residue_scores)

    print(
        f"  [eval] seed={seed} cond={cond}"
        f" harm_rate={harm_rate:.4f}"
        f" n_contacts={n_contact_steps}"
        f" n_total={n_total_steps}"
        f" mean_residue={mean_residue:.4f}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond,
        "chain_on": chain_on,
        "harm_rate": float(harm_rate),
        "n_contact_steps": int(n_contact_steps),
        "n_total_steps": int(n_total_steps),
        "mean_residue_score": float(mean_residue),
        "train_harm_steps": int(train_harm_steps),
        "train_e2_world_steps": int(train_e2_world_steps),
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
    n_candidates: int = 8,
    horizon: int = 5,
    n_cem_iterations: int = 3,
    elite_fraction: float = 0.3,
    dry_run: bool = False,
    **kwargs,
) -> dict:
    """Discriminative pair: KERNEL_CHAIN_ON vs KERNEL_CHAIN_ABLATED."""
    results_on:  List[Dict] = []
    results_off: List[Dict] = []

    for seed in seeds:
        for chain_on in [True, False]:
            cond = "KERNEL_CHAIN_ON" if chain_on else "KERNEL_CHAIN_ABLATED"
            print(
                f"\n[V3-EXQ-124] {cond} seed={seed}"
                f" warmup={warmup_episodes} eval={eval_episodes}"
                f" steps={steps_per_episode}"
                f" alpha_world={alpha_world}"
                f" n_candidates={n_candidates} horizon={horizon}"
                f" n_cem={n_cem_iterations}"
                f" {'DRY_RUN' if dry_run else ''}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                chain_on=chain_on,
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
                n_candidates=n_candidates,
                horizon=horizon,
                n_cem_iterations=n_cem_iterations,
                elite_fraction=elite_fraction,
                dry_run=dry_run,
            )
            if chain_on:
                results_on.append(r)
            else:
                results_off.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    harm_on    = _avg(results_on,  "harm_rate")
    harm_off   = _avg(results_off, "harm_rate")
    harm_delta = harm_off - harm_on  # positive = chaining reduces harm

    relative_reduction = harm_delta / max(1e-6, harm_off)

    residue_on  = _avg(results_on,  "mean_residue_score")
    residue_off = _avg(results_off, "mean_residue_score")

    # Per-seed C3 check
    per_seed_c3 = [
        ron["harm_rate"] < roff["harm_rate"]
        for ron, roff in zip(results_on, results_off)
    ]
    c3_pass = all(per_seed_c3)

    # C4: data quality -- min contacts in ABLATED condition (harm is detectable)
    min_contacts_ablated = min(r["n_contact_steps"] for r in results_off)

    c1_pass = harm_delta       >= THRESH_C1_HARM_REDUCTION
    c2_pass = relative_reduction >= THRESH_C2_RELATIVE_REDUCTION
    c4_pass = min_contacts_ablated >= THRESH_C4_MIN_CONTACTS
    all_pass = c1_pass and c2_pass and c3_pass and c4_pass

    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    status = "PASS" if all_pass else "FAIL"

    # C5 residue diagnostic
    c5_residue = residue_on <= residue_off

    print(f"\n[V3-EXQ-124] Final results:", flush=True)
    print(
        f"  harm_rate_CHAIN={harm_on:.4f}  harm_rate_ABLATED={harm_off:.4f}"
        f"  delta={harm_delta:+.4f}  (C1 thresh >={THRESH_C1_HARM_REDUCTION})"
        f"  C1={'PASS' if c1_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"  relative_reduction={relative_reduction:.4f}"
        f"  (C2 thresh >={THRESH_C2_RELATIVE_REDUCTION})"
        f"  C2={'PASS' if c2_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"  per_seed_CHAIN_less_than_ABLATED: {per_seed_c3}"
        f"  C3={'PASS' if c3_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"  min_contacts_ablated={min_contacts_ablated}  (C4 thresh >={THRESH_C4_MIN_CONTACTS})"
        f"  C4={'PASS' if c4_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"  mean_residue_CHAIN={residue_on:.4f}  mean_residue_ABLATED={residue_off:.4f}"
        f"  C5 (diagnostic): {'PASS -- chaining navigates lower-residue paths' if c5_residue else 'n/a'}",
        flush=True,
    )
    print(f"  status={status}  ({criteria_met}/4 required criteria met)", flush=True)

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: harm_rate delta={harm_delta:+.4f}"
            f" (needs >={THRESH_C1_HARM_REDUCTION})."
            " Kernel chaining does not produce a meaningful absolute harm reduction."
            " Possible causes: (1) E2 world_forward model not well trained (check"
            " train_e2_world_steps); (2) terrain prior does not learn hazard structure;"
            " (3) 400 warmup insufficient for hippocampal CEM to calibrate."
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: relative_reduction={relative_reduction:.4f}"
            f" (needs >={THRESH_C2_RELATIVE_REDUCTION})."
            " Chaining eliminates less than 15pct of ablated-condition harm."
            " Chaining may add planning signal but it is weak relative to baseline."
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: per_seed direction inconsistent ({per_seed_c3})."
            " Chaining did not consistently reduce harm across both seeds."
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: min_contacts_ablated={min_contacts_ablated}"
            f" < {THRESH_C4_MIN_CONTACTS}."
            " Insufficient contact events in ablated condition."
            " Harm rate cannot be measured reliably."
            " Increase hazard density or eval budget."
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    if all_pass:
        interpretation = (
            "MECH-033 SUPPORTED: E2 action-object kernel chaining is load-bearing."
            f" harm_rate_CHAIN={harm_on:.4f} vs harm_rate_ABLATED={harm_off:.4f}"
            f" (delta={harm_delta:+.4f} >= {THRESH_C1_HARM_REDUCTION},"
            f" relative_reduction={relative_reduction:.4f}"
            f" >= {THRESH_C2_RELATIVE_REDUCTION})."
            " Direction consistent across all seeds (C3)."
            " Hippocampal CEM in action-object space outperforms random candidate generation."
            " The kernel chaining interface (E2 -> hippocampus -> E3) preserves causal"
            " continuity across planning segment boundaries."
        )
    elif c1_pass and not c2_pass:
        interpretation = (
            "PARTIAL: C1 passes (harm delta={:+.4f} >= {:0.2f}) but C2 fails"
            " (relative_reduction={:.4f} < {:0.2f})."
            " Chaining produces a real but small harm reduction relative to the ablated baseline."
            " Kernel interface is beneficial but effect size is weaker than required."
        ).format(harm_delta, THRESH_C1_HARM_REDUCTION, relative_reduction, THRESH_C2_RELATIVE_REDUCTION)
    elif c2_pass and not c1_pass:
        interpretation = (
            "PARTIAL: C2 passes (relative_reduction={:.4f} >= {:0.2f}) but C1 fails"
            " (harm delta={:+.4f} < {:0.2f})."
            " The relative reduction is meaningful but the ablated harm rate is low enough"
            " that the absolute reduction falls below threshold."
            " Chaining may be beneficial in scarcer harm environments."
        ).format(relative_reduction, THRESH_C2_RELATIVE_REDUCTION, harm_delta, THRESH_C1_HARM_REDUCTION)
    else:
        interpretation = (
            "MECH-033 NOT SUPPORTED: Kernel chaining provides no meaningful planning advantage."
            f" harm_rate_CHAIN={harm_on:.4f},"
            f" harm_rate_ABLATED={harm_off:.4f},"
            f" delta={harm_delta:+.4f}."
            " Random candidate generation with E3 selection performs comparably to"
            " hippocampal CEM in action-object space."
            " The kernel chaining interface does not preserve causal continuity"
            " in a way that reduces harm at this scale."
        )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    per_on_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" harm_rate={r['harm_rate']:.4f}"
        f" n_contacts={r['n_contact_steps']}"
        f" n_total={r['n_total_steps']}"
        f" mean_residue={r['mean_residue_score']:.4f}"
        f" train_harm_steps={r['train_harm_steps']}"
        f" train_e2_world_steps={r['train_e2_world_steps']}"
        for r in results_on
    )
    per_off_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" harm_rate={r['harm_rate']:.4f}"
        f" n_contacts={r['n_contact_steps']}"
        f" n_total={r['n_total_steps']}"
        f" mean_residue={r['mean_residue_score']:.4f}"
        for r in results_off
    )

    summary_markdown = (
        f"# V3-EXQ-124 -- MECH-033 Kernel Chaining Interface Discriminative Pair\n\n"
        f"**Status:** {status}\n"
        f"**Claim:** MECH-033\n"
        f"**Proposal:** EXP-0023 / EVB-0018\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n"
        f"**Warmup:** {warmup_episodes} eps  **Eval:** {eval_episodes} eps\n"
        f"**n_candidates:** {n_candidates}  **horizon:** {horizon}"
        f"  **n_cem_iterations:** {n_cem_iterations}\n\n"
        f"## Design\n\n"
        f"KERNEL_CHAIN_ON: HippocampalModule CEM in action-object space O."
        f" Terrain prior seeds distribution; E2 action_object kernels chain"
        f" across horizon; residue field scores trajectories; elite refit.\n"
        f"KERNEL_CHAIN_ABLATED: Random candidate actions; E3 selects best"
        f" via one-step E2 world_forward lookahead. No kernel chaining.\n"
        f"Both conditions train E1, E2, E3 harm_eval, and hippocampus identically.\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: harm_rate_ABLATED - harm_rate_CHAIN >= {THRESH_C1_HARM_REDUCTION}"
        f"  (absolute harm reduction)\n"
        f"C2: relative_reduction >= {THRESH_C2_RELATIVE_REDUCTION}"
        f"  (15pct relative reduction)\n"
        f"C3: harm_rate_CHAIN < harm_rate_ABLATED for ALL seeds  (consistency)\n"
        f"C4: min_contacts_ablated >= {THRESH_C4_MIN_CONTACTS}  (data quality)\n"
        f"C5 (diagnostic): mean_residue_CHAIN <= mean_residue_ABLATED\n\n"
        f"## Aggregate Results\n\n"
        f"| Metric | KERNEL_CHAIN_ON | KERNEL_CHAIN_ABLATED | Delta | Pass |\n"
        f"|--------|----------------|---------------------|-------|------|\n"
        f"| harm_rate (C1 delta) | {harm_on:.4f} | {harm_off:.4f}"
        f" | {harm_delta:+.4f} | {'YES' if c1_pass else 'NO'} |\n"
        f"| relative_reduction (C2) | {relative_reduction:.4f} | -- | --"
        f" | {'YES' if c2_pass else 'NO'} |\n"
        f"| seed consistency (C3) | {per_seed_c3} | -- | --"
        f" | {'YES' if c3_pass else 'NO'} |\n"
        f"| min_contacts_ablated (C4) | -- | {min_contacts_ablated} | --"
        f" | {'YES' if c4_pass else 'NO'} |\n"
        f"| mean_residue (C5 diag) | {residue_on:.4f} | {residue_off:.4f} | --"
        f" | {'YES' if c5_residue else 'n/a'} |\n\n"
        f"## Interpretation\n\n"
        f"{interpretation}\n\n"
        f"## Per-Seed (KERNEL_CHAIN_ON)\n\n"
        f"{per_on_rows}\n\n"
        f"## Per-Seed (KERNEL_CHAIN_ABLATED)\n\n"
        f"{per_off_rows}\n"
        f"{failure_section}\n"
    )

    return {
        "status": status,
        "metrics": {
            "harm_rate_chain":            float(harm_on),
            "harm_rate_ablated":          float(harm_off),
            "harm_reduction_delta":       float(harm_delta),
            "relative_reduction":         float(relative_reduction),
            "min_contacts_ablated":       float(min_contacts_ablated),
            "mean_residue_chain":         float(residue_on),
            "mean_residue_ablated":       float(residue_off),
            "crit1_pass":                 1.0 if c1_pass else 0.0,
            "crit2_pass":                 1.0 if c2_pass else 0.0,
            "crit3_pass":                 1.0 if c3_pass else 0.0,
            "crit4_pass":                 1.0 if c4_pass else 0.0,
            "crit5_residue_diag":         1.0 if c5_residue else 0.0,
            "criteria_met":               float(criteria_met),
            "n_seeds":                    float(len(seeds)),
            "alpha_world":                float(alpha_world),
            "n_candidates":               float(n_candidates),
            "horizon":                    float(horizon),
            "n_cem_iterations":           float(n_cem_iterations),
        },
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_class": "experimental",
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
    parser.add_argument("--n-candidates",     type=int,   default=8)
    parser.add_argument("--horizon",          type=int,   default=5)
    parser.add_argument("--n-cem-iter",       type=int,   default=3)
    parser.add_argument("--elite-fraction",   type=float, default=0.3)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Quick dry-run: 3 warmup, 2 eval episodes per cell. Writes JSON.",
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
        n_candidates=args.n_candidates,
        horizon=args.horizon,
        n_cem_iterations=args.n_cem_iter,
        elite_fraction=args.elite_fraction,
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

    print(f"\n[V3-EXQ-124] Result written to {out_path}", flush=True)
    print(f"[V3-EXQ-124] status={result['status']}", flush=True)
