"""Shared trainers + scaffolding for the MECH-457 GOV-FANOUT-1 competence-discrimination
portfolio (V3-EXQ-747 / 748 / 749).

WHY THIS EXISTS. V3-EXQ-742 (MECH-457 ON/OFF validation) refuted the 734-737 autopsy's
"single missing mechanism = MECH-457" conclusion: all four first-class RPE actor-critic arms
forage 0.20-0.27/ep at D3 -- below random_walk (0.93) and ~0.5% of the local_view_greedy
ceiling (48.05) -- while EVERY readiness precondition held. Necessity is NOT falsified; the
deficit sits UPSTREAM of the actor-critic head. The 742 autopsy (failure_autopsy_morning-
digest-742-744a-745-746-746a_2026-07-13) routed this to a GOV-FANOUT-1 DISCRIMINATION
portfolio over the surviving hypothesis space, and REFUSED a same-question floor re-pose
(737a/732b-style). This module carries the shared machinery so the three legs stay
byte-comparable to 742 and to each other.

THE 2x2 FACTORIAL (representation axis x teacher axis). 742 = the (z_world, sparse) cell:

                     sparse foraging RL          dense teacher (shaping + BC-clone)
    z_world  (R0)    742: FAIL (done, cited)     Leg B  (V3-EXQ-748, H-explore)
    raw 5x5  (R1)    Leg A  (V3-EXQ-747, H-rep)  Leg C  (V3-EXQ-749, conjunction)

Design-axis definitions (held clean so the factorial attributes cleanly):
  * Representation axis R -- what the actor READS.
      R0 = z_world: the prediction-trained latent, IN the loop via agent.actor_critic_step,
           cotrain_encoder=True (742's most-favorable arm; 742 showed frozen-vs-cotrain does
           not matter for the sparse-RL cell, so R0 is pinned to its best-shot cotrain form,
           NOT re-split as a third axis).
      R1 = raw 5x5 resource_field_view: a standalone ActorCriticPolicy(world_dim=25) reading
           the EXACT input LocalViewGreedyPolicy uses (capability_eval), bypassing z_world
           entirely. No REE encoder, no P0 world-model warmup.
  * Teacher axis E -- the LEARNING SIGNAL density.
      E0 = sparse foraging RL: r = harm + FORAGE_BONUS*[resource] + novelty  (742's teacher).
      E1 = dense, two instantiations run as sibling arms:
          - shaped RL: E0 + potential-based distance-to-nearest-resource shaping
                       (Ng et al. 1999 policy-invariant potential Phi(s) = -manhattan_to_
                       nearest_resource; speeds learning without changing the optimum).
          - BC-clone: SUPERVISED behavior-cloning of LocalViewGreedyPolicy actions (no RL),
                       eval the cloned actor. Directly probes whether the input carries enough
                       to REPRODUCE the competent expert's action -- for z_world (Leg B) a
                       failed CE fit IS the "prediction-trained latent action-inadequate"
                       signal; for raw-view (Leg C) it confirms the view is action-adequate.

DIAGNOSTIC, not evidence. Each leg discriminates WHY competence fails upstream of MECH-457.
experiment_purpose="diagnostic" -> excluded from governance confidence/conflict scoring;
routes to /failure-autopsy for adjudication before any governance action. MECH-457 stays
candidate / v3_pending. claim_ids=["MECH-457"] tags relevance only (diagnostic => context).

READINESS + P0 readiness-assert (the /queue-experiment adjudication gate). The load-bearing
criterion reads a LEARNED quantity (the trained actor's foraging_competence @D3 vs the 1.0
floor), so each leg measures the SAME statistic (foraging_competence) on a KNOWN-POSITIVE
control in setup and emits it as a readiness-kind precondition: LocalViewGreedyPolicy reading
the SAME 5x5 view forages >= floor @D3 (it does: 48.05 in 742). Below readiness -> self-route
substrate_not_ready_requeue (NEVER a substrate-verdict label). The verdict labels below are
HYPOTHESES; /failure-autopsy adjudicates before governance use.

Sourced APIs (verified 2026-07-13):
  ree_core/action_learning/actor_critic.py -- ActorCriticPolicy(world_dim, action_dim, ...)
    is input-agnostic (trunk = Linear(world_dim, hidden)); .select / .forward.
  ree_core/agent.py -- REEAgent.actor_critic_step / actor_critic_parameters /
    actor_critic_encoder_parameters (the co-shape seam; reused for R0).
  experiments/_lib/capability_eval.py -- evaluate_seed / OraclePolicy / RandomPolicy /
    LocalViewGreedyPolicy / COMPETENCE_RESOURCE_FLOOR / nearest_resource_manhattan / Policy.
  experiments/v3_exq_742_mech457_actor_critic_onoff.py (x742) -- _make_actor_critic_agent /
    _sense / ActorCriticEvalPolicy (reused verbatim for the R0 z_world path).
  experiments/v3_exq_734_env_difficulty_competence_recovery_sweep.py (x734) -- DIFFICULTY_RUNGS
    / _env_kwargs_for_rung / _make_env / _train_all_on_agent / _compute_gae / _RunningStd /
    _novelty_bonus / FORAGE_BONUS / REWARD_STD_EPS / PPO_* / EVAL_EPISODES / STEPS_PER_EPISODE /
    P0_WARMUP_EPISODES / P1_PPO_EPISODES / _strict_majority.
ASCII-only in all runtime strings (repo rule).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ree_core.action_learning.actor_critic import ActorCriticPolicy
from experiments._lib.capability_eval import (
    COMPETENCE_RESOURCE_FLOOR,
    LocalViewGreedyPolicy,
    OraclePolicy,
    Policy,
    RandomPolicy,
    evaluate_seed,
    nearest_resource_manhattan,
)
import experiments.v3_exq_724_competence_localization_diagnostic as x724  # noqa: F401 (re-export path)
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734
import experiments.v3_exq_742_mech457_actor_critic_onoff as x742

DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# Shared budget (full) -- reuses 742 / 734 constants for byte-comparability.
# ---------------------------------------------------------------------------
SEEDS: List[int] = [42, 43, 44]
RUNG: Dict[str, Any] = x734.DIFFICULTY_RUNGS[-1]     # D3_hazard_free (the clean foraging rung)
RUNG_ID: str = RUNG["rung_id"]

P0_WARMUP_EPISODES = x734.P0_WARMUP_EPISODES         # 200  -- z_world all-ON warmup (R0 only)
RL_EPISODES = x734.P1_PPO_EPISODES                   # 1000 -- actor-critic RL budget (== 742)
BC_EPISODES = 300                                    # supervised behavior-cloning rollouts
EVAL_EPISODES = x734.EVAL_EPISODES                   # 20   -- unshaped foraging eval per cell
STEPS_PER_EPISODE = x734.STEPS_PER_EPISODE           # 200

AC_LR = x742.AC_LR                                   # 3e-4  (RL)
BC_LR = 1e-3                                          # supervised CE learning rate
AC_GAMMA = x742.AC_GAMMA                             # 0.99
AC_ENTROPY_BETA = x734.PPO_ENTROPY_BETA              # 0.03
AC_VALUE_COEF = x734.PPO_VALUE_COEF                  # 0.5
AC_GRAD_CLIP = x734.PPO_GRAD_CLIP                    # 0.5
SHAPING_COEF = 1.0                                    # potential-shaping weight (~ FORAGE_BONUS)
ACTOR_CRITIC_HIDDEN = 128                             # matches the 742/734 trunk width
TRAIN_FORAGE_WINDOW = 50                              # recent-episodes reward-hacking guard
RAW_VIEW_DIM = 25                                     # flattened 5x5 resource_field_view

# 738 reference denominator (local-view-achievable ceiling @D3); computed LIVE per run.
DENOM_738_D3_REFERENCE = 48.05

# ---------------------------------------------------------------------------
# Dry-run budget (tiny: exercises setup + one training backward + eval + self-route).
# ---------------------------------------------------------------------------
DRY_SEEDS = [42]
DRY_P0 = 2
DRY_RL = 6
DRY_BC = 4
DRY_EVAL = 2
DRY_STEPS = 15

ANCHOR_ARMS: Tuple[str, ...] = ("local_view_greedy", "greedy_oracle", "random_walk")

# Distributional-critic bin support (MECH-457 H-retention-critic). Only consulted when a
# caller enables the distributional critic; the scalar path never reads these.
DIST_CRITIC_N_BINS = 41       # categorical head width
DIST_CRITIC_LIMIT = 10.0      # support half-width in SYMLOG space
DIST_CRITIC_SIGMA = 0.75      # HL-Gauss sigma in bin widths (Farebrother 2024); 0.0 -> two-hot


# ---------------------------------------------------------------------------
# Potential-based reward shaping (Ng et al. 1999): Phi(s) = -manhattan_to_nearest_resource.
# ---------------------------------------------------------------------------
def _potential(env: Any) -> float:
    d = nearest_resource_manhattan(env)
    return 0.0 if d is None else -float(d)


# ---------------------------------------------------------------------------
# Critic loss dispatch (MECH-457 H-retention-critic, 2026-07-18).
# ---------------------------------------------------------------------------
def critic_value_loss(policy, value_logits_t, value_t, ret_t):
    """The AC_VALUE_COEF-weighted critic term for one batch of returns.

    Distributional critic ON  -> cross-entropy of the predicted value distribution
                                 against the two-hot / HL-Gauss projection of the return.
    Distributional critic OFF -> the pre-existing 0.5 * (V - G)^2 scalar MSE, byte-identical.

    ANTI-ALIAS: only the VALUE term is dispatched here. policy_loss, the entropy bonus, the
    advantage weighting, the BC auxiliary and the credit-replay policy term are untouched on
    both branches -- the update rule is identical, only what the baseline knows differs.
    """
    if (
        policy is not None
        and getattr(policy, "use_distributional_critic", False)
        and value_logits_t is not None
    ):
        return AC_VALUE_COEF * policy.critic_loss(value_logits_t, ret_t.detach())
    return AC_VALUE_COEF * 0.5 * (value_t - ret_t.detach()).pow(2).mean()


# ---------------------------------------------------------------------------
# R1 (raw 5x5 view) actor-critic: a STANDALONE ActorCriticPolicy(world_dim=25). No REE
# encoder, no P0 warmup. Reads obs_dict["resource_field_view"] directly.
# ---------------------------------------------------------------------------
def make_rawview_ac(
    hidden_dim: int = ACTOR_CRITIC_HIDDEN,
    use_distributional_critic: bool = False,
    n_value_bins: int = DIST_CRITIC_N_BINS,
    value_bin_limit: float = DIST_CRITIC_LIMIT,
    value_bin_sigma: float = DIST_CRITIC_SIGMA,
) -> ActorCriticPolicy:
    """Standalone raw-view actor-critic. `hidden_dim` defaults to the 742/734 trunk width
    (128, byte-identical for the fanout legs); the MECH-457 capacity-amend build passes a
    larger width to raise policy capacity on the raw 5x5 path."""
    return ActorCriticPolicy(
        world_dim=RAW_VIEW_DIM, action_dim=5,
        hidden_dim=int(hidden_dim), use_sf_critic=False,
        use_distributional_critic=bool(use_distributional_critic),
        n_value_bins=int(n_value_bins),
        value_bin_limit=float(value_bin_limit),
        value_bin_sigma=float(value_bin_sigma),
    ).to(DEVICE)


def _rawview_tensor(obs_dict: Dict[str, Any]) -> torch.Tensor:
    rfv = obs_dict.get("resource_field_view")
    if rfv is None:
        raise KeyError(
            "resource_field_view absent -- the raw-view legs require the 5x5 proxy field "
            "channel (env config lacks it)."
        )
    v = torch.as_tensor(np.asarray(rfv, dtype=np.float32).reshape(-1), device=DEVICE)
    return v.unsqueeze(0)


class RawViewACEvalPolicy(Policy):
    """Greedy (argmax) eval of a raw-view actor-critic over UNSHAPED foraging."""

    def __init__(self, ac: ActorCriticPolicy, label: str) -> None:
        self.ac = ac
        self.name = label

    def act(self, env: Any, obs_dict: Dict[str, Any]) -> int:
        with torch.no_grad():
            logits, _v, _phi, _psi = self.ac.forward(_rawview_tensor(obs_dict))
        if not torch.isfinite(logits).all():
            return int(np.random.randint(0, int(env.action_dim)))
        return int(torch.argmax(logits, dim=-1).reshape(-1)[0].item())


def train_rawview_ac_rl(
    ac: ActorCriticPolicy, env: Any, seed: int, n_episodes: int, steps: int,
    arm_label: str, denom: int, shaping_coef: float,
) -> Dict[str, Any]:
    """A2C single-backward-per-episode (GAE), reading the raw 5x5 view. shaping_coef>0 adds
    potential-based distance-to-nearest-resource shaping (the E1 shaped variant); ==0 is the
    sparse foraging teacher (E0)."""
    optimiser = torch.optim.Adam(ac.parameters(), lr=AC_LR)
    _policy = ac
    reward_std = x734._RunningStd()
    novelty_counter: Dict[Tuple[int, int], int] = {}
    train_forage_recent: deque = deque(maxlen=TRAIN_FORAGE_WINDOW)

    for ep in range(n_episodes):
        _flat, obs_dict = env.reset()
        ep_logp: List[torch.Tensor] = []
        ep_value_t: List[torch.Tensor] = []
        ep_entropy: List[torch.Tensor] = []
        ep_value_f: List[float] = []
        ep_value_logits: List[torch.Tensor] = []   # distributional critic only (else empty)
        ep_rewards: List[float] = []
        terminal = False
        bootstrap_value = 0.0
        ep_resources = 0

        view = _rawview_tensor(obs_dict)
        for _step in range(steps):
            step = ac.select(view, deterministic=False)
            a_idx = int(step.action.reshape(-1)[0].item())
            phi_s = _potential(env) if shaping_coef > 0.0 else 0.0
            _flat, harm_signal, done, info, obs_dict = env.step(a_idx)
            ttype = str(info.get("transition_type", "none"))
            if ttype == "resource":
                ep_resources += 1
            pos = (int(env.agent_x), int(env.agent_y))
            shaped = (
                float(harm_signal)
                + (x734.FORAGE_BONUS if ttype == "resource" else 0.0)
                + x734._novelty_bonus(novelty_counter, pos)
            )
            if shaping_coef > 0.0:
                phi_next = _potential(env)
                shaped += shaping_coef * (AC_GAMMA * phi_next - phi_s)
            reward_std.update(shaped)

            ep_logp.append(step.log_prob.reshape(-1)[0])
            ep_value_t.append(step.value.reshape(-1)[0])
            ep_entropy.append(step.entropy.reshape(-1)[0])
            ep_value_f.append(float(step.value.reshape(-1)[0].item()))
            ep_rewards.append(shaped)
            if step.value_logits is not None:
                ep_value_logits.append(step.value_logits.reshape(-1))

            if done:
                terminal = True
                break
            view = _rawview_tensor(obs_dict)

        if not terminal:
            with torch.no_grad():
                boot = ac.select(view, deterministic=False)
            bootstrap_value = float(boot.value.reshape(-1)[0].item())

        T = len(ep_logp)
        if T > 0:
            scale = reward_std.std + x734.REWARD_STD_EPS
            scaled = [r / scale for r in ep_rewards]
            advs, rets = x734._compute_gae(scaled, ep_value_f, bootstrap_value, terminal)
            logp_t = torch.stack(ep_logp)
            value_t = torch.stack(ep_value_t)
            entropy_t = torch.stack(ep_entropy)
            adv_t = torch.tensor(advs, dtype=torch.float32, device=DEVICE)
            ret_t = torch.tensor(rets, dtype=torch.float32, device=DEVICE)
            if T > 1:
                adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            policy_loss = -(logp_t * adv_t.detach()).mean()
            vlogits_t = torch.stack(ep_value_logits) if ep_value_logits else None
            value_loss = critic_value_loss(_policy, vlogits_t, value_t, ret_t)
            entropy_bonus = entropy_t.mean()
            loss = policy_loss + value_loss - AC_ENTROPY_BETA * entropy_bonus
            if torch.isfinite(loss):
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ac.parameters(), AC_GRAD_CLIP)
                optimiser.step()

        train_forage_recent.append(ep_resources)
        cur = ep + 1
        if cur % 200 == 0 or cur == n_episodes:
            print(f"  [train] {arm_label} seed={seed} phase=RL ep {cur}/{denom}", flush=True)

    mtf = float(sum(train_forage_recent) / len(train_forage_recent)) if train_forage_recent else 0.0
    return {"mean_train_forage_recent": round(mtf, 6)}


def bc_warmup_rawview(
    ac: ActorCriticPolicy, env: Any, seed: int, n_bc: int, steps: int,
    arm_label: str, denom: int,
) -> Dict[str, Any]:
    """Behavior-cloning of LocalViewGreedyPolicy from the raw 5x5 view (supervised CE, no RL).
    Rolls the expert to generate on-expert states; CE-trains the actor logits toward the
    expert's action at each visited state. Returns the mean per-episode action-match accuracy
    over the final window (a teacher-signal readout)."""
    expert = LocalViewGreedyPolicy(seed=seed)
    optimiser = torch.optim.Adam(ac.parameters(), lr=BC_LR)
    acc_recent: deque = deque(maxlen=TRAIN_FORAGE_WINDOW)

    for ep in range(n_bc):
        _flat, obs_dict = env.reset()
        logits_list: List[torch.Tensor] = []
        target_list: List[int] = []
        for _step in range(steps):
            a_expert = int(expert.act(env, obs_dict))
            logits, _v, _phi, _psi = ac.forward(_rawview_tensor(obs_dict))
            logits_list.append(logits.reshape(-1))
            target_list.append(a_expert)
            _flat, _harm, done, _info, obs_dict = env.step(a_expert)
            if done:
                break
        T = len(logits_list)
        if T > 0:
            logit_stack = torch.stack(logits_list)                       # [T, action_dim]
            target_t = torch.tensor(target_list, dtype=torch.long, device=DEVICE)
            loss = torch.nn.functional.cross_entropy(logit_stack, target_t)
            with torch.no_grad():
                pred = torch.argmax(logit_stack, dim=-1)
                acc_recent.append(float((pred == target_t).float().mean().item()))
            if torch.isfinite(loss):
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ac.parameters(), AC_GRAD_CLIP)
                optimiser.step()
        cur = ep + 1
        if cur % 100 == 0 or cur == n_bc:
            print(f"  [train] {arm_label} seed={seed} phase=BC ep {cur}/{denom}", flush=True)

    macc = float(sum(acc_recent) / len(acc_recent)) if acc_recent else 0.0
    return {"bc_action_match_accuracy_recent": round(macc, 6)}


# ---------------------------------------------------------------------------
# R0 (z_world) actor-critic. Reuses x742's factory + sense + eval policy so the encoder /
# co-shaping path is byte-identical to 742's cotrain arms. cotrain=True (best-shot; pinned).
# ---------------------------------------------------------------------------
def make_zworld_agent(
    env: Any, cotrain: bool = True, actor_critic_hidden: int = ACTOR_CRITIC_HIDDEN,
    use_distributional_critic: bool = False,
    n_value_bins: int = DIST_CRITIC_N_BINS,
    value_bin_limit: float = DIST_CRITIC_LIMIT,
    value_bin_sigma: float = DIST_CRITIC_SIGMA,
):
    """742-style agent: all-ON REE stack + MECH-457 actor-critic, plain critic (742 showed SF
    adds nothing).

    Defaults (cotrain=True, hidden=128) reproduce the 742/751/765 cotrain z_world arm
    byte-identical. The MECH-457 capacity-amend build (2026-07-16) passes cotrain=False for the
    ON arm -- train the policy on the FROZEN prediction-trained encoder (z_world.detach() inside
    agent.actor_critic_step), per the Stooke 2021 decoupled-representation caution, since the
    765 retest showed cotrain is DESTRUCTIVE on z_world (ON 0.35 < OFF 5.22) -- and a larger
    actor_critic_hidden to raise policy capacity."""
    return x742._make_actor_critic_agent(
        env, cotrain=bool(cotrain), sf=False, hidden=int(actor_critic_hidden),
        distributional=bool(use_distributional_critic),
        n_value_bins=int(n_value_bins),
        value_bin_limit=float(value_bin_limit),
        value_bin_sigma=float(value_bin_sigma),
    )


def warmup_zworld(agent: Any, env: Any, seed: int, p0: int, steps: int) -> None:
    """P0 all-ON world-model warmup (724-A0 recipe; p1_episodes=0 -> pure encoder/forward
    warmup, latent_stack un-co-shaped at entry to the AC phase -- identical to 742)."""
    x734._train_all_on_agent(
        agent, env, seed=seed, p0_episodes=p0, p1_episodes=0,
        steps_per_episode=steps, rung_id=RUNG_ID, total_denominator=max(1, p0),
    )


def train_zworld_ac_shaped(
    agent: Any, env: Any, seed: int, n_episodes: int, steps: int,
    arm_label: str, denom: int, shaping_coef: float,
) -> Dict[str, Any]:
    """z_world A2C (cotrain) with potential-based shaping added to the 742 foraging teacher.
    Mirrors x742._train_actor_critic (plain critic) with the shaping hook; the actor reads
    LIVE z_world so the gradient reaches latent_stack (cotrain)."""
    params = list(agent.actor_critic_parameters()) + list(agent.actor_critic_encoder_parameters())
    optimiser = torch.optim.Adam(params, lr=AC_LR)
    _policy = getattr(agent, "action_critic", None)
    reward_std = x734._RunningStd()
    novelty_counter: Dict[Tuple[int, int], int] = {}
    train_forage_recent: deque = deque(maxlen=TRAIN_FORAGE_WINDOW)

    for ep in range(n_episodes):
        _flat, obs_dict = env.reset()
        agent.reset()
        ep_logp: List[torch.Tensor] = []
        ep_value_t: List[torch.Tensor] = []
        ep_entropy: List[torch.Tensor] = []
        ep_value_f: List[float] = []
        ep_value_logits: List[torch.Tensor] = []   # distributional critic only (else empty)
        ep_rewards: List[float] = []
        terminal = False
        bootstrap_value = 0.0
        ep_resources = 0

        latent = x742._sense(agent, obs_dict)
        for _step in range(steps):
            step = agent.actor_critic_step(latent, deterministic=False)
            a_idx = int(step.action.reshape(-1)[0].item())
            phi_s = _potential(env) if shaping_coef > 0.0 else 0.0
            _flat, harm_signal, done, info, obs_dict = env.step(a_idx)
            ttype = str(info.get("transition_type", "none"))
            if ttype == "resource":
                ep_resources += 1
            pos = (int(env.agent_x), int(env.agent_y))
            shaped = (
                float(harm_signal)
                + (x734.FORAGE_BONUS if ttype == "resource" else 0.0)
                + x734._novelty_bonus(novelty_counter, pos)
            )
            if shaping_coef > 0.0:
                phi_next = _potential(env)
                shaped += shaping_coef * (AC_GAMMA * phi_next - phi_s)
            reward_std.update(shaped)

            ep_logp.append(step.log_prob.reshape(-1)[0])
            ep_value_t.append(step.value.reshape(-1)[0])
            ep_entropy.append(step.entropy.reshape(-1)[0])
            ep_value_f.append(float(step.value.reshape(-1)[0].item()))
            ep_rewards.append(shaped)
            if step.value_logits is not None:
                ep_value_logits.append(step.value_logits.reshape(-1))

            if done:
                terminal = True
                break
            latent = x742._sense(agent, obs_dict)

        if not terminal:
            with torch.no_grad():
                boot = agent.actor_critic_step(latent, deterministic=False)
            bootstrap_value = float(boot.value.reshape(-1)[0].item())

        T = len(ep_logp)
        if T > 0:
            scale = reward_std.std + x734.REWARD_STD_EPS
            scaled = [r / scale for r in ep_rewards]
            advs, rets = x734._compute_gae(scaled, ep_value_f, bootstrap_value, terminal)
            logp_t = torch.stack(ep_logp)
            value_t = torch.stack(ep_value_t)
            entropy_t = torch.stack(ep_entropy)
            adv_t = torch.tensor(advs, dtype=torch.float32, device=DEVICE)
            ret_t = torch.tensor(rets, dtype=torch.float32, device=DEVICE)
            if T > 1:
                adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            policy_loss = -(logp_t * adv_t.detach()).mean()
            vlogits_t = torch.stack(ep_value_logits) if ep_value_logits else None
            value_loss = critic_value_loss(_policy, vlogits_t, value_t, ret_t)
            entropy_bonus = entropy_t.mean()
            loss = policy_loss + value_loss - AC_ENTROPY_BETA * entropy_bonus
            if torch.isfinite(loss):
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, AC_GRAD_CLIP)
                optimiser.step()

        train_forage_recent.append(ep_resources)
        cur = ep + 1
        if cur % 200 == 0 or cur == n_episodes:
            print(f"  [train] {arm_label} seed={seed} phase=RL ep {cur}/{denom}", flush=True)

    mtf = float(sum(train_forage_recent) / len(train_forage_recent)) if train_forage_recent else 0.0
    return {"mean_train_forage_recent": round(mtf, 6)}


def bc_warmup_zworld(
    agent: Any, env: Any, seed: int, n_bc: int, steps: int, arm_label: str, denom: int,
) -> Dict[str, Any]:
    """Behavior-cloning of LocalViewGreedyPolicy through the z_world path (cotrain: the CE
    gradient reaches actor_critic AND the encoder). A failed CE fit here IS the direct
    "prediction-trained z_world is action-inadequate" signal (the expert reads the raw view;
    the actor must reproduce its action from z_world)."""
    expert = LocalViewGreedyPolicy(seed=seed)
    params = list(agent.actor_critic_parameters()) + list(agent.actor_critic_encoder_parameters())
    optimiser = torch.optim.Adam(params, lr=BC_LR)
    acc_recent: deque = deque(maxlen=TRAIN_FORAGE_WINDOW)

    for ep in range(n_bc):
        _flat, obs_dict = env.reset()
        agent.reset()
        logits_list: List[torch.Tensor] = []
        target_list: List[int] = []
        for _step in range(steps):
            a_expert = int(expert.act(env, obs_dict))
            latent = x742._sense(agent, obs_dict)
            step = agent.actor_critic_step(latent, deterministic=True)
            logits_list.append(step.logits.reshape(-1))
            target_list.append(a_expert)
            _flat, _harm, done, _info, obs_dict = env.step(a_expert)
            if done:
                break
        T = len(logits_list)
        if T > 0:
            logit_stack = torch.stack(logits_list)
            target_t = torch.tensor(target_list, dtype=torch.long, device=DEVICE)
            loss = torch.nn.functional.cross_entropy(logit_stack, target_t)
            with torch.no_grad():
                pred = torch.argmax(logit_stack, dim=-1)
                acc_recent.append(float((pred == target_t).float().mean().item()))
            if torch.isfinite(loss):
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, AC_GRAD_CLIP)
                optimiser.step()
        cur = ep + 1
        if cur % 100 == 0 or cur == n_bc:
            print(f"  [train] {arm_label} seed={seed} phase=BC ep {cur}/{denom}", flush=True)

    macc = float(sum(acc_recent) / len(acc_recent)) if acc_recent else 0.0
    return {"bc_action_match_accuracy_recent": round(macc, 6)}


# ---------------------------------------------------------------------------
# Anchor eval (readiness / denominators). Cheap, deterministic reference policies.
# ---------------------------------------------------------------------------
def run_anchor_cell(arm_id: str, env: Any, seed: int, eval_eps: int, steps: int) -> Dict[str, Any]:
    if arm_id == "local_view_greedy":
        policy: Policy = LocalViewGreedyPolicy(seed=seed)
    elif arm_id == "greedy_oracle":
        policy = OraclePolicy()
    elif arm_id == "random_walk":
        policy = RandomPolicy(seed)
    else:
        raise ValueError(f"unknown anchor {arm_id!r}")
    return evaluate_seed(policy, env, eval_eps, steps)


# ---------------------------------------------------------------------------
# Shared summary + self-route scaffolding.
# ---------------------------------------------------------------------------
def summarize(forages: List[float]) -> Dict[str, Any]:
    n = len(forages)
    n_supra = int(sum(1 for f in forages if f >= COMPETENCE_RESOURCE_FLOOR))
    return {
        "foraging_competence_mean": round(float(sum(forages) / n), 6) if n else 0.0,
        "foraging_competence_per_seed": [round(f, 6) for f in forages],
        "n_seeds": n,
        "n_seeds_supra_floor": n_supra,
        "majority_supra_floor": bool(n_supra >= (n + 1) // 2) if n else False,
    }


def readiness_precondition(local_view_mean: float) -> Dict[str, Any]:
    """The load-bearing readiness precondition, same-statistic as the verdict criterion:
    LocalViewGreedyPolicy (reading the same 5x5 view the AC / expert reads) forages >= the
    1.0 floor @D3. Below -> substrate_not_ready_requeue (env not solvable from the view;
    a sub-floor AC reading would be uninterpretable)."""
    return {
        "name": "local_view_greedy_clears_floor_at_d3",
        "kind": "readiness",
        "description": (
            "LocalViewGreedyPolicy reading the SAME 5x5 resource_field_view forages >= the "
            "1.0 competence floor at D3 -- the positive control that the env is solvable from "
            "the local view (same statistic the verdict criterion routes on). Below-floor "
            "means the substrate/env is not ready, NOT that the actor-critic failed."
        ),
        "control": "local_view_greedy foraging_competence @D3 (738 denominator; 48.05 in 742)",
        "measured": round(float(local_view_mean), 6),
        "threshold": float(COMPETENCE_RESOURCE_FLOOR),
        "met": bool(local_view_mean >= COMPETENCE_RESOURCE_FLOOR),
    }
