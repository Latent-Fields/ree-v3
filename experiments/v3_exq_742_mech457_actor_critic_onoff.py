#!/opt/local/bin/python3
"""V3-EXQ-742 -- MECH-457 first-class actor-critic action-learning ON/OFF VALIDATION.

EVIDENCE experiment (experiment_purpose=evidence, claim_ids=[MECH-457]); PROMOTES / DEMOTES
NOTHING until run + reviewed + routed to /failure-autopsy. This is the ON/OFF validation the
734/737 conversion-ceiling autopsy sanctioned as the next test after the MECH-457 substrate
landed (ree_core/action_learning/actor_critic.py, 2026-07-12). It settles, empirically,
whether a DEDICATED RPE-driven actor-critic (a dorsal-striatal actor + value/successor-feature
critic) converts foraging drive into competent action where the thin bias_head REINFORCE readout
(the 724/737 incompetence control) cannot -- and whether that requires CO-SHAPING the z_world
encoder (the 737 refinement).

RPE TEACHER (user-confirmed 2026-07-12): FORAGING REWARD -- per step
    r_t = harm_signal_t + FORAGE_BONUS * [transition_type == "resource"] + novelty_bonus_t
(grounded by construction; matches the reward shaping of V3-EXQ-734/737 exactly so 742 is
directly comparable to the failure record). The eval DV is UNSHAPED real foraging. The
substrate benefit-harm signal (agent.actor_critic_reward, ARC-108) is NOT used -- those E3
heads are random-init behind the ARC-030 warmup gate, so using them would need an extra P0
grounding phase + readiness gate; the foraging teacher needs neither.

ARMS (8) -- the full MECH-457 arm matrix + the incompetence control + three yardstick anchors:
  A0 actor_critic_frozen_plain   cotrain_encoder=F, sf_critic=F  reproduces the 737 frozen-latent
                                 inadequacy level (AC gradient does NOT reach latent_stack).
  A1 actor_critic_cotrain_plain  cotrain_encoder=T, sf_critic=F  cand-A: un-freeze the encoder +
                                 minimal co-training -- does it clear the floor?
  A2 actor_critic_frozen_sf      cotrain_encoder=F, sf_critic=T  isolates the SF-critic absent
                                 encoder co-adaptation.
  A3 actor_critic_cotrain_sf     cotrain_encoder=T, sf_critic=T  the deep form (cand-A + cand-B).
  bias_head_baseline             the 724-A0 all-ON control (x734._make_all_on_agent /
                                 _train_all_on_agent -> capability_eval.REEForwardPolicy). The
                                 incompetence CONTROL -- must reproduce the sub-floor deficit.
  local_view_greedy              the V3-EXQ-738 LOCAL-VIEW-achievable ceiling (5x5
                                 resource_field_view greedy climber). THE FAIR DENOMINATOR
                                 (48.05 @D3), a subset of world_state the encoder senses -- NOT
                                 the privileged global oracle (which inflates the bar ~28x; WS-1
                                 observability confound, failure_autopsy_V3-EXQ-732a).
  greedy_oracle                  global nearest-resource forager. READINESS / achievability anchor.
  random_walk                    uniform-random FLOOR anchor.

RUNGS: D0 (724 baseline) + D3 (hazard-free pure foraging), reusing x734.DIFFICULTY_RUNGS byte
for byte, so 742 is comparable to the 734/737/738 sibling sweep.

DV (load-bearing): capability_eval.evaluate_seed foraging_competence (mean resources/episode),
absolute 1.0 competence floor, higher-is-better. Normalized against the LIVE local_view_greedy
anchor measured in THIS run (the 738 denominator; 48.05 @D3 reference). SEEDS [42, 43, 44].

TRAINING (phased, per (arm, seed) cell): P0 = 724-A0 all-ON world-model warmup (reuses
x734._train_all_on_agent with p1_episodes=0 -- warms the SD-056 e2 forward model; latent_stack
starts un-co-shaped, = the 737 frozen condition) -> P1 = actor-critic training driving actions
via agent.actor_critic_step, A2C single-backward-per-episode with GAE advantages (reuses
x734._compute_gae / _RunningStd / _novelty_bonus). Cotrain arms (A1/A3) ADD
agent.actor_critic_encoder_parameters() (= latent_stack) to the optimizer and read LIVE z_world;
frozen arms (A0/A2) read z_world.detach() (substrate applies the lever via
config.actor_critic_cotrain_encoder). SF arms (A2/A3) add an SF-TD loss
(psi -> phi.detach() + gamma*psi_next.detach()) + a reward-regression loss
(action_critic.sf_reward_prediction(phi) -> the per-step reward) and use V_SF = psi.w as the
critic value. -> P2 = UNSHAPED foraging eval via capability_eval.evaluate_seed.

READINESS GATE (below-readiness self-routes substrate_not_ready_requeue -- NEVER a verdict):
D0 AND D3 greedy_oracle clear the 1.0 floor, D0 AND D3 local_view_greedy clear the floor, and
bias_head_baseline stays BELOW the floor @D0 (the 724 incompetence premise reproduced).

SUCCESS (MECH-457 supported): a cotrain arm (A1 and/or A3) forages >= the 1.0 floor on a strict
majority of seeds @D3 while the bias_head_baseline does not.

PRE-REGISTERED SELF-ROUTE (HYPOTHESIS, adjudicate via /failure-autopsy before any governance use):
  * readiness fails -> substrate_not_ready_requeue (FAIL; draw NO conclusion).
  * cotrain supra-floor @D3 AND frozen sub-floor -> co_shaping_is_the_lever (PASS; MECH-457
    supported; the A0/A1 contrast confirms cand-A: un-freezing the encoder is the lever).
  * any actor-critic arm supra-floor including a FROZEN arm -> actor_critic_encoder_agnostic
    (PASS; MECH-457 supported; a dedicated actor-critic recovers competence even without encoder
    co-shaping -- co-shaping is not the discriminator).
  * NO actor-critic arm supra-floor @D3 -> deeper_than_action_learning (FAIL; weakens MECH-457).
    Even a dedicated RPE actor-critic cannot forage an oracle-and-local-view-achievable env ->
    the deficit is deeper than action-learning credit assignment. Route to RE-AUTOPSY, NOT a
    lettered floor re-test (the 734/737 autopsy's RESOLVED_BY_FANOUT refusal stands; REFUSE any
    737a/732b-style same-question floor re-pose).
  cand-B (SF-vs-plain): the A1/A3 contrast settles the successor-feature critic -- it earns its
  keep only if A3 lifts foraging strict-above A1 @D3.

REWARD-HACKING GUARD (instrument-only here; load-bearing ONLY under the benefit-harm teacher,
which is NOT used): reports train_return_vs_eval_foraging_divergence per AC arm. With the
foraging teacher the training return IS foraging, so a large eval-vs-train divergence would flag
a substrate/optimisation pathology rather than reward-hacking; recorded for audit, does not gate
the outcome.

EVIDENCE-FOR / EVIDENCE-AGAINST MECH-457:
  * SUPPORTS: a cotrain (or any) actor-critic arm clears the 1.0 floor @D3 on a majority of seeds
    while the bias_head_baseline stays sub-floor -> the dedicated actor-critic is the lever.
  * WEAKENS: no actor-critic arm clears the floor even @D3 (oracle + local-view achievable) ->
    the dedicated actor-critic did not recover competence; the deficit is deeper.
  * NON-CONTRIBUTORY (substrate_not_ready_requeue): a readiness anchor is below floor, or the
    bias_head premise is not reproduced -> re-examine env/floor/budget, draw NO conclusion.

ethics_preflight:
  involves_negative_valence: false
  involves_suffering_like_state: false
  involves_self_model: false
  involves_inescapability_or_helplessness: false
  involves_offline_replay_over_harm: false
  involves_social_mind_or_language: false
  involves_human_data_or_clinical_context: false
  decision: allow

SLEEP DRIVER: none (no sleep loop; use_sleep_loop / sws_enabled / rem_enabled all OFF).

Sourced APIs (verified 2026-07-12):
  ree_core/action_learning/actor_critic.py -- ActorCriticPolicy(.select -> ActorCriticStep with
    action/log_prob/value/entropy/logits[/phi/psi]; .sf_reward_prediction).
  ree_core/agent.py -- REEAgent.actor_critic_step (co-shape/detach lever) /
    actor_critic_parameters / actor_critic_encoder_parameters (= latent_stack, the co-shape seam).
  experiments/_lib/capability_eval.py -- evaluate_seed / OraclePolicy / RandomPolicy /
    LocalViewGreedyPolicy / REEForwardPolicy / COMPETENCE_RESOURCE_FLOOR.
  experiments/v3_exq_724_competence_localization_diagnostic.py (x724) -- ENV_KWARGS,
    _base_config_kwargs, _all_on_extra_kwargs, obs helpers, REEConfig/REEAgent.
  experiments/v3_exq_734_env_difficulty_competence_recovery_sweep.py (x734) -- DIFFICULTY_RUNGS,
    _env_kwargs_for_rung, _make_env, _make_all_on_agent, _train_all_on_agent, _compute_gae,
    _RunningStd, _novelty_bonus, FORAGE_BONUS, REWARD_STD_EPS, PPO_* constants, _strict_majority.
This module is ASCII-only in all runtime strings.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.capability_eval import (  # noqa: E402
    COMPETENCE_RESOURCE_FLOOR,
    LocalViewGreedyPolicy,
    OraclePolicy,
    Policy,
    RandomPolicy,
    evaluate_seed,
)
from experiments._metrics import check_degeneracy  # noqa: E402
from experiments._lib.baselines import exq742_mech457_bias_head_baseline as ac_baseline  # noqa: E402
import experiments.v3_exq_724_competence_localization_diagnostic as x724  # noqa: E402
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_742_mech457_actor_critic_onoff"
QUEUE_ID = "V3-EXQ-742"
CLAIM_IDS: List[str] = ["MECH-457"]
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# Budget (full). Reuses the 724-A0 warmup recipe + 737 comparison shape.
# ---------------------------------------------------------------------------
SEEDS: List[int] = [42, 43, 44]
P0_WARMUP_EPISODES = x734.P0_WARMUP_EPISODES     # 200  -- all-ON world-model warmup (724 A0)
AC_TRAIN_EPISODES = x734.P1_PPO_EPISODES         # 1000 -- actor-critic P1 (matches 737 PPO budget)
EVAL_EPISODES = x734.EVAL_EPISODES               # 20   -- unshaped foraging eval per cell
STEPS_PER_EPISODE = x734.STEPS_PER_EPISODE       # 200

# Actor-critic P1 hyper-parameters (shared with the 734/737 PPO net where applicable).
AC_LR = x734.PPO_LR                              # 3e-4
AC_GAMMA = x734.PPO_GAMMA                        # 0.99
AC_VALUE_COEF = x734.PPO_VALUE_COEF              # 0.5  (plain critic value loss weight)
AC_ENTROPY_BETA = x734.PPO_ENTROPY_BETA          # 0.03 (exploration bonus)
AC_GRAD_CLIP = x734.PPO_GRAD_CLIP                # 0.5
SF_TD_COEF = 0.5                                 # successor-feature TD loss weight (cand-B)
SF_REWARD_REG_COEF = 0.5                         # SF reward-regression loss weight (cand-B)
TRAIN_FORAGE_WINDOW = 50                         # recent episodes for the reward-hacking guard

# 738 reference denominator (local-view-achievable ceiling @D3). We compute local_view_greedy
# LIVE in this run and normalize against the live anchor; this constant is a cross-check only.
DENOM_738_D3_REFERENCE = 48.05

# ---------------------------------------------------------------------------
# Dry-run budget (tiny; exercises P0 warmup + AC training backward + eval + self-route).
# ---------------------------------------------------------------------------
DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_AC = 6
DRY_RUN_EVAL = 2
DRY_RUN_STEPS = 15

# ---------------------------------------------------------------------------
# Two rungs: D0 (724 baseline) + D3 (hazard-free), byte-identical to 734/737/738.
# ---------------------------------------------------------------------------
RUNGS = [x734.DIFFICULTY_RUNGS[0], x734.DIFFICULTY_RUNGS[-1]]
D0_RUNG_ID = RUNGS[0]["rung_id"]
D3_RUNG_ID = RUNGS[-1]["rung_id"]

# Arm registry. AC arms carry (cotrain_encoder, use_sf_critic) + the A0..A3 label.
AC_ARMS: Dict[str, Dict[str, Any]] = {
    "actor_critic_frozen_plain": {"cotrain": False, "sf": False, "code": "A0"},
    "actor_critic_cotrain_plain": {"cotrain": True, "sf": False, "code": "A1"},
    "actor_critic_frozen_sf": {"cotrain": False, "sf": True, "code": "A2"},
    "actor_critic_cotrain_sf": {"cotrain": True, "sf": True, "code": "A3"},
}
ANCHOR_ARMS = ("local_view_greedy", "greedy_oracle", "random_walk")
ARM_ORDER = (
    list(AC_ARMS.keys())
    + ["bias_head_baseline"]
    + list(ANCHOR_ARMS)
)


# ---------------------------------------------------------------------------
# Agent factories + obs sensing.
# ---------------------------------------------------------------------------
def _make_actor_critic_agent(env, cotrain: bool, sf: bool):
    """All-ON REE stack (724 config) + the MECH-457 actor-critic substrate enabled."""
    kwargs = x724._base_config_kwargs(env)
    kwargs.update(x724._all_on_extra_kwargs())
    kwargs.update(
        dict(
            use_actor_critic=True,
            actor_critic_cotrain_encoder=bool(cotrain),
            actor_critic_use_sf_critic=bool(sf),
            actor_critic_hidden=128,
            actor_critic_sf_feature_dim=32,
        )
    )
    cfg = x724.REEConfig.from_dims(**kwargs)
    return x724.REEAgent(cfg)


def _sense(agent, obs_dict: Dict[str, Any]):
    """Encode one observation through the agent (NO no_grad -- for cotrain arms the z_world
    gradient must be able to reach latent_stack). Callers that only need argmax wrap in
    torch.no_grad() themselves (eval)."""
    body = obs_dict["body_state"].float()
    world = obs_dict["world_state"].float()
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    return agent.sense(
        obs_body=body,
        obs_world=world,
        obs_harm=x724._obs_harm(obs_dict),
        obs_harm_a=x724._obs_harm_a(obs_dict),
        obs_harm_history=x724._obs_harm_history(obs_dict),
    )


# ---------------------------------------------------------------------------
# P1 actor-critic trainer (A2C single-backward-per-episode, GAE advantages).
# Actions are driven by agent.actor_critic_step (the dorsal-striatal pathway; NOT E3 argmin).
# ---------------------------------------------------------------------------
def _train_actor_critic(
    agent,
    env,
    cotrain: bool,
    sf: bool,
    seed: int,
    n_episodes: int,
    steps_per_episode: int,
    arm_label: str,
    rung_id: str,
    total_denominator: int,
) -> Dict[str, Any]:
    params = list(agent.actor_critic_parameters())
    if cotrain:
        params = params + list(agent.actor_critic_encoder_parameters())
    optimiser = torch.optim.Adam(params, lr=AC_LR)
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
        ep_scaled_rewards_pending: List[float] = []
        ep_phi: List[torch.Tensor] = []
        ep_psi: List[torch.Tensor] = []
        terminal = False
        bootstrap_value = 0.0
        bootstrap_psi: Optional[torch.Tensor] = None
        ep_resources = 0

        latent = _sense(agent, obs_dict)
        for _step in range(steps_per_episode):
            step = agent.actor_critic_step(latent, deterministic=False)
            a_idx = int(step.action.reshape(-1)[0].item())
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
            reward_std.update(shaped)

            ep_logp.append(step.log_prob.reshape(-1)[0])
            ep_value_t.append(step.value.reshape(-1)[0])
            ep_entropy.append(step.entropy.reshape(-1)[0])
            ep_value_f.append(float(step.value.reshape(-1)[0].item()))
            ep_scaled_rewards_pending.append(shaped)   # scaled below once reward_std is final
            if sf:
                ep_phi.append(step.phi.reshape(-1))
                ep_psi.append(step.psi.reshape(-1))

            if done:
                terminal = True
                break
            latent = _sense(agent, obs_dict)

        # Bootstrap value / successor features for a non-terminal episode.
        if not terminal:
            with torch.no_grad():
                boot = agent.actor_critic_step(latent, deterministic=False)
            bootstrap_value = float(boot.value.reshape(-1)[0].item())
            if sf:
                bootstrap_psi = boot.psi.reshape(-1).detach()

        T = len(ep_logp)
        if T > 0:
            scale = reward_std.std + x734.REWARD_STD_EPS
            scaled_rewards = [r / scale for r in ep_scaled_rewards_pending]
            advs, rets = x734._compute_gae(scaled_rewards, ep_value_f, bootstrap_value, terminal)

            logp_t = torch.stack(ep_logp)
            value_t = torch.stack(ep_value_t)
            entropy_t = torch.stack(ep_entropy)
            adv_t = torch.tensor(advs, dtype=torch.float32, device=DEVICE)
            ret_t = torch.tensor(rets, dtype=torch.float32, device=DEVICE)
            if T > 1:
                adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            policy_loss = -(logp_t * adv_t.detach()).mean()
            entropy_bonus = entropy_t.mean()

            if sf:
                phi_stack = torch.stack(ep_phi)           # [T, D] (grad-connected)
                psi_stack = torch.stack(ep_psi)           # [T, D] (grad-connected)
                with torch.no_grad():
                    psi_next = torch.zeros_like(psi_stack)
                    if T > 1:
                        psi_next[:-1] = psi_stack[1:].detach()
                    if not terminal and bootstrap_psi is not None:
                        psi_next[-1] = bootstrap_psi
                    # terminal -> psi_next[-1] stays 0.
                    sf_target = phi_stack.detach() + AC_GAMMA * psi_next
                sf_td_loss = 0.5 * (psi_stack - sf_target).pow(2).mean()
                scaled_t = torch.tensor(scaled_rewards, dtype=torch.float32, device=DEVICE)
                r_hat = agent.action_critic.sf_reward_prediction(phi_stack)  # [T]  phi . w
                reward_reg_loss = 0.5 * (r_hat - scaled_t.detach()).pow(2).mean()
                critic_loss = SF_TD_COEF * sf_td_loss + SF_REWARD_REG_COEF * reward_reg_loss
            else:
                critic_loss = AC_VALUE_COEF * 0.5 * (value_t - ret_t.detach()).pow(2).mean()

            loss = policy_loss + critic_loss - AC_ENTROPY_BETA * entropy_bonus
            if torch.isfinite(loss):
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, AC_GRAD_CLIP)
                optimiser.step()

        train_forage_recent.append(ep_resources)
        cur = ep + 1
        if cur % 200 == 0 or cur == n_episodes:
            print(
                f"  [train] {arm_label} rung={rung_id} seed={seed} phase=P1 "
                f"ep {cur}/{total_denominator}",
                flush=True,
            )

    mean_train_forage = (
        float(sum(train_forage_recent) / len(train_forage_recent)) if train_forage_recent else 0.0
    )
    return {"mean_train_forage_recent": round(mean_train_forage, 6)}


class ActorCriticEvalPolicy(Policy):
    """Greedy (deterministic argmax) eval of the trained MECH-457 actor over UNSHAPED foraging."""

    def __init__(self, agent, label: str) -> None:
        self.agent = agent
        self.name = label

    def reset(self, env: Any) -> None:
        self.agent.reset()

    def act(self, env: Any, obs_dict: Dict[str, Any]) -> int:
        with torch.no_grad():
            latent = _sense(self.agent, obs_dict)
            step = self.agent.actor_critic_step(latent, deterministic=True)
        if step.logits is not None and not torch.isfinite(step.logits).all():
            return int(np.random.randint(0, int(env.action_dim)))
        return int(step.action.reshape(-1)[0].item())


# ---------------------------------------------------------------------------
# Cell builders. Each returns an eval row (capability_eval.evaluate_seed output) plus, for the
# AC arms, the reward-hacking guard's train-forage readout.
# ---------------------------------------------------------------------------
def _run_ac_cell(
    arm_id: str,
    rung: Dict[str, Any],
    seed: int,
    env_kwargs: Dict[str, Any],
    p0: int,
    ac_eps: int,
    eval_eps: int,
    steps: int,
) -> Dict[str, Any]:
    cfg = AC_ARMS[arm_id]
    cotrain, sf = bool(cfg["cotrain"]), bool(cfg["sf"])
    rid = rung["rung_id"]

    # P0: all-ON world-model warmup (724 A0 recipe; p1_episodes=0 -> pure encoder/forward warmup).
    warm_env = x734._make_env(seed, env_kwargs)
    agent = _make_actor_critic_agent(warm_env, cotrain, sf)
    x734._train_all_on_agent(
        agent, warm_env, seed=seed, p0_episodes=p0, p1_episodes=0,
        steps_per_episode=steps, rung_id=rid, total_denominator=p0,
    )

    # P1: actor-critic training on THIS rung's env.
    train_env = x734._make_env(seed, env_kwargs)
    guard = _train_actor_critic(
        agent, train_env, cotrain, sf, seed=seed, n_episodes=ac_eps,
        steps_per_episode=steps, arm_label=arm_id, rung_id=rid, total_denominator=ac_eps,
    )

    # P2: UNSHAPED foraging eval.
    eval_env = x734._make_env(seed, env_kwargs)
    row = evaluate_seed(ActorCriticEvalPolicy(agent, arm_id), eval_env, eval_eps, steps)
    row["mean_train_forage_recent"] = guard["mean_train_forage_recent"]
    return row


def _run_baseline_cell(
    rung: Dict[str, Any],
    seed: int,
    env_kwargs: Dict[str, Any],
    p0: int,
    p1: int,
    eval_eps: int,
    steps: int,
) -> Dict[str, Any]:
    """bias_head_baseline: the 724-A0 all-ON control (P0 warmup + P1 two-head REINFORCE).

    Delegates to the canonical lineage baseline module so this OFF cell's computation AND
    its arm fingerprint match the separate V3-EXQ-742-m mint BY CONSTRUCTION (both call
    ac_baseline.run_off_cell + ac_baseline.off_path_config_slice)."""
    return ac_baseline.run_off_cell(
        env_kwargs, seed,
        p0_warmup_episodes=p0, p1_reinforce_episodes=p1,
        eval_episodes=eval_eps, steps_per_episode=steps, rung_id=rung["rung_id"],
    )


def _run_anchor_cell(
    arm_id: str,
    rung: Dict[str, Any],
    seed: int,
    env_kwargs: Dict[str, Any],
    eval_eps: int,
    steps: int,
) -> Dict[str, Any]:
    if arm_id == "local_view_greedy":
        policy: Policy = LocalViewGreedyPolicy(seed=seed)
    elif arm_id == "greedy_oracle":
        policy = OraclePolicy()
    elif arm_id == "random_walk":
        policy = RandomPolicy(seed)
    else:
        raise ValueError(f"unknown anchor {arm_id!r}")
    eval_env = x734._make_env(seed, env_kwargs)
    return evaluate_seed(policy, eval_env, eval_eps, steps)


def _arm_config_slice(arm_id: str, rid: str, env_kwargs: Dict[str, Any],
                      p0: int, p1: int, ac_eps: int, eval_eps: int, steps: int) -> Dict[str, Any]:
    """Reuse-fingerprint config slice: only what THIS arm's computation reads."""
    base = {
        "arm_id": arm_id,
        "rung_id": rid,
        "env_kwargs": dict(env_kwargs),
        "eval_episodes": int(eval_eps),
        "steps_per_episode": int(steps),
    }
    if arm_id in AC_ARMS:
        base.update({
            "kind": "actor_critic",
            "cotrain_encoder": bool(AC_ARMS[arm_id]["cotrain"]),
            "use_sf_critic": bool(AC_ARMS[arm_id]["sf"]),
            "actor_critic_hidden": 128,
            "actor_critic_sf_feature_dim": 32,
            "p0_warmup_episodes": int(p0),
            "ac_train_episodes": int(ac_eps),
            "reward_teacher": "foraging",
        })
    elif arm_id == "bias_head_baseline":
        base.update({
            "kind": "bias_head_all_on",
            "p0_warmup_episodes": int(p0),
            "p1_reinforce_episodes": int(p1),
        })
    else:
        base.update({"kind": "anchor"})
    return base


# ---------------------------------------------------------------------------
# Summaries + self-route.
# ---------------------------------------------------------------------------
def _summarize(forages: List[float]) -> Dict[str, Any]:
    n = len(forages)
    n_supra = int(sum(1 for f in forages if f >= COMPETENCE_RESOURCE_FLOOR))
    return {
        "foraging_competence_mean": round(float(sum(forages) / n), 6) if n else 0.0,
        "foraging_competence_per_seed": [round(f, 6) for f in forages],
        "n_seeds": n,
        "n_seeds_supra_floor": n_supra,
        "majority_supra_floor": bool(n_supra >= (n + 1) // 2) if n else False,
    }


def run_experiment(
    seeds: List[int],
    p0: int,
    p1: int,
    ac_eps: int,
    eval_eps: int,
    steps: int,
) -> Dict[str, Any]:
    print(
        f"MECH-457 actor-critic ON/OFF validation ({len(ARM_ORDER)} arms x {len(RUNGS)} rungs x "
        f"{len(seeds)} seeds; P0={p0}, P1_reinforce={p1}, AC={ac_eps}, eval={eval_eps}, "
        f"steps={steps})",
        flush=True,
    )

    # per_rung_forage[rid][arm] = list of per-seed foraging_competence.
    per_rung_forage: Dict[str, Dict[str, List[float]]] = {
        r["rung_id"]: {a: [] for a in ARM_ORDER} for r in RUNGS
    }
    # per_rung_trainforage[rid][ac_arm] = list of per-seed mean_train_forage_recent (guard).
    per_rung_trainforage: Dict[str, Dict[str, List[float]]] = {
        r["rung_id"]: {a: [] for a in AC_ARMS} for r in RUNGS
    }
    all_cells: List[Dict[str, Any]] = []

    for rung in RUNGS:
        rid = rung["rung_id"]
        env_kwargs = x734._env_kwargs_for_rung(rung)
        for arm_id in ARM_ORDER:
            for seed in seeds:
                print(f"Seed {seed} Condition {rid}:{arm_id}", flush=True)
                if arm_id == "bias_head_baseline":
                    # Same slice the V3-EXQ-742-m mint emits -> self-mint fingerprint aligns
                    # with the separate mint (both from ac_baseline.off_path_config_slice).
                    slice_cfg = ac_baseline.off_path_config_slice(env_kwargs, p0, p1, eval_eps, steps)
                else:
                    slice_cfg = _arm_config_slice(arm_id, rid, env_kwargs, p0, p1, ac_eps, eval_eps, steps)
                with arm_cell(
                    seed,
                    config_slice=slice_cfg,
                    script_path=Path(__file__),
                    config_slice_declared=True,
                    include_driver_script_in_hash=False,
                ) as cell:
                    if arm_id in AC_ARMS:
                        row = _run_ac_cell(arm_id, rung, seed, env_kwargs, p0, ac_eps, eval_eps, steps)
                    elif arm_id == "bias_head_baseline":
                        row = _run_baseline_cell(rung, seed, env_kwargs, p0, p1, eval_eps, steps)
                    else:
                        row = _run_anchor_cell(arm_id, rung, seed, env_kwargs, eval_eps, steps)
                    row["rung_id"] = rid
                    row["arm_id"] = arm_id
                    row["seed"] = int(seed)
                    cell.stamp(row)

                forage = float(row["foraging_competence"])
                per_rung_forage[rid][arm_id].append(forage)
                if arm_id in AC_ARMS:
                    per_rung_trainforage[rid][arm_id].append(float(row.get("mean_train_forage_recent", 0.0)))
                all_cells.append(row)
                print(
                    f"verdict: {'PASS' if row['competence_supra_floor'] else 'FAIL'} "
                    f"(rung={rid} arm={arm_id} seed={seed} forage/ep={forage})",
                    flush=True,
                )

    per_rung: Dict[str, Dict[str, Any]] = {}
    for rid in per_rung_forage:
        per_rung[rid] = {a: _summarize(per_rung_forage[rid][a]) for a in ARM_ORDER}

    def _mean(rid: str, arm: str) -> float:
        return float(per_rung[rid][arm]["foraging_competence_mean"])

    def _maj(rid: str, arm: str) -> bool:
        return bool(per_rung[rid][arm]["majority_supra_floor"])

    # ---- readiness ----------------------------------------------------------
    d0_oracle_ok = _mean(D0_RUNG_ID, "greedy_oracle") >= COMPETENCE_RESOURCE_FLOOR
    d3_oracle_ok = _mean(D3_RUNG_ID, "greedy_oracle") >= COMPETENCE_RESOURCE_FLOOR
    d0_localview_ok = _mean(D0_RUNG_ID, "local_view_greedy") >= COMPETENCE_RESOURCE_FLOOR
    d3_localview_ok = _mean(D3_RUNG_ID, "local_view_greedy") >= COMPETENCE_RESOURCE_FLOOR
    bias_reproduces_d0 = not _maj(D0_RUNG_ID, "bias_head_baseline")
    readiness_met = bool(
        d0_oracle_ok and d3_oracle_ok and d0_localview_ok and d3_localview_ok and bias_reproduces_d0
    )

    # ---- load-bearing @D3 ---------------------------------------------------
    a0 = _maj(D3_RUNG_ID, "actor_critic_frozen_plain")
    a1 = _maj(D3_RUNG_ID, "actor_critic_cotrain_plain")
    a2 = _maj(D3_RUNG_ID, "actor_critic_frozen_sf")
    a3 = _maj(D3_RUNG_ID, "actor_critic_cotrain_sf")
    cotrain_supra = bool(a1 or a3)
    frozen_supra = bool(a0 or a2)
    any_ac_supra = bool(a0 or a1 or a2 or a3)
    bias_subfloor_d3 = not _maj(D3_RUNG_ID, "bias_head_baseline")
    # cand-B: SF earns its keep only if A3 lifts strict-above A1 @D3.
    sf_earns_keep = bool(a3 and (_mean(D3_RUNG_ID, "actor_critic_cotrain_sf") > _mean(D3_RUNG_ID, "actor_critic_cotrain_plain")))

    if not readiness_met:
        outcome, label, direction = "FAIL", "substrate_not_ready_requeue", "non_contributory"
    elif cotrain_supra and not frozen_supra:
        outcome, label, direction = "PASS", "co_shaping_is_the_lever", "supports"
    elif any_ac_supra:
        outcome, label, direction = "PASS", "actor_critic_encoder_agnostic", "supports"
    else:
        outcome, label, direction = "FAIL", "deeper_than_action_learning", "weakens"

    # ---- non-degeneracy (evidence-run scoring net): cross-arm foraging spread @D3 -----------
    d3_action_learner_means = [
        _mean(D3_RUNG_ID, a) for a in list(AC_ARMS.keys()) + ["bias_head_baseline"]
    ]
    degeneracy = check_degeneracy(
        {"d3_action_learner_foraging": {"values": d3_action_learner_means}}
    )

    # ---- reward-hacking guard (instrument-only under the foraging teacher) ------------------
    guard_block: Dict[str, Any] = {}
    for a in AC_ARMS:
        tf = per_rung_trainforage[D3_RUNG_ID][a]
        train_mean = round(float(sum(tf) / len(tf)), 6) if tf else 0.0
        eval_mean = _mean(D3_RUNG_ID, a)
        guard_block[a] = {
            "d3_mean_train_forage_recent": train_mean,
            "d3_eval_foraging": round(eval_mean, 6),
            "train_return_vs_eval_foraging_divergence": round(eval_mean - train_mean, 6),
        }

    local_view_d3 = _mean(D3_RUNG_ID, "local_view_greedy")

    interpretation = {
        "label": label,
        "preconditions": [
            {"name": "d0_greedy_oracle_clears_floor", "kind": "readiness",
             "description": "D0 (724 baseline) env must be floor-achievable with global info.",
             "control": "greedy_oracle on D0 vs the 1.0 floor",
             "measured": round(_mean(D0_RUNG_ID, "greedy_oracle"), 6),
             "threshold": float(COMPETENCE_RESOURCE_FLOOR), "met": bool(d0_oracle_ok)},
            {"name": "d3_greedy_oracle_clears_floor", "kind": "readiness",
             "description": "Hazard-free D3 env must be floor-achievable with global info.",
             "control": "greedy_oracle on D3 vs the 1.0 floor",
             "measured": round(_mean(D3_RUNG_ID, "greedy_oracle"), 6),
             "threshold": float(COMPETENCE_RESOURCE_FLOOR), "met": bool(d3_oracle_ok)},
            {"name": "d0_local_view_greedy_clears_floor", "kind": "readiness",
             "description": "D0 env must be floor-achievable from the 5x5 local view the encoder senses.",
             "control": "local_view_greedy (738 denominator) on D0 vs the 1.0 floor",
             "measured": round(_mean(D0_RUNG_ID, "local_view_greedy"), 6),
             "threshold": float(COMPETENCE_RESOURCE_FLOOR), "met": bool(d0_localview_ok)},
            {"name": "d3_local_view_greedy_clears_floor", "kind": "readiness",
             "description": "Hazard-free D3 env must be floor-achievable from the local view (the fair denominator).",
             "control": "local_view_greedy (738 denominator) on D3 vs the 1.0 floor",
             "measured": round(local_view_d3, 6),
             "threshold": float(COMPETENCE_RESOURCE_FLOOR), "met": bool(d3_localview_ok)},
            {"name": "bias_head_reproduces_incompetence_at_d0", "kind": "readiness",
             "description": (
                 "The 724-A0 bias_head_baseline must forage BELOW the floor at D0 (reproduce the "
                 "724/737 deficit) for the MECH-457 contrast premise to hold; if it already clears "
                 "the floor the deficit is not present and no ON/OFF read is licensed."),
             "control": "bias_head_baseline D0 mean resources/ep vs floor (strict majority of seeds)",
             "measured": round(_mean(D0_RUNG_ID, "bias_head_baseline"), 6),
             "threshold": float(COMPETENCE_RESOURCE_FLOOR),
             "direction": "upper",  # premise MET when measured < threshold (stays below floor)
             "met": bool(bias_reproduces_d0)},
        ],
        "criteria": [
            {"name": "C_actor_critic_clears_floor_at_D3", "load_bearing": True, "passed": bool(any_ac_supra)},
        ],
        "criteria_non_degenerate": {
            "oracle_clears_floor_both_rungs": bool(d0_oracle_ok and d3_oracle_ok),
            "local_view_clears_floor_both_rungs": bool(d0_localview_ok and d3_localview_ok),
            "bias_head_reproduces_incompetence_at_d0": bool(bias_reproduces_d0),
            "bias_head_subfloor_at_d3_contrast_holds": bool(bias_subfloor_d3),
            "cross_arm_foraging_spread_d3": bool(degeneracy["non_degenerate"]),
        },
        "candB_sf_settlement": {
            "a1_cotrain_plain_maj_d3": bool(a1),
            "a3_cotrain_sf_maj_d3": bool(a3),
            "a3_forage_gt_a1_d3": bool(_mean(D3_RUNG_ID, "actor_critic_cotrain_sf") > _mean(D3_RUNG_ID, "actor_critic_cotrain_plain")),
            "sf_earns_keep": sf_earns_keep,
        },
    }

    result: Dict[str, Any] = {
        "outcome": outcome,
        "interpretation": interpretation,
        "interpretation_label": label,
        "evidence_direction": direction,
        "evidence_direction_per_claim": {"MECH-457": direction},
        "readiness": {
            "readiness_met": readiness_met,
            "d0_greedy_oracle_clears_floor": d0_oracle_ok,
            "d3_greedy_oracle_clears_floor": d3_oracle_ok,
            "d0_local_view_greedy_clears_floor": d0_localview_ok,
            "d3_local_view_greedy_clears_floor": d3_localview_ok,
            "bias_head_reproduces_incompetence_at_d0": bias_reproduces_d0,
        },
        "headline": {
            "d3_a0_frozen_plain_forage": round(_mean(D3_RUNG_ID, "actor_critic_frozen_plain"), 6),
            "d3_a1_cotrain_plain_forage": round(_mean(D3_RUNG_ID, "actor_critic_cotrain_plain"), 6),
            "d3_a2_frozen_sf_forage": round(_mean(D3_RUNG_ID, "actor_critic_frozen_sf"), 6),
            "d3_a3_cotrain_sf_forage": round(_mean(D3_RUNG_ID, "actor_critic_cotrain_sf"), 6),
            "d3_bias_head_baseline_forage": round(_mean(D3_RUNG_ID, "bias_head_baseline"), 6),
            "d3_local_view_greedy_denominator": round(local_view_d3, 6),
            "d3_greedy_oracle_forage": round(_mean(D3_RUNG_ID, "greedy_oracle"), 6),
            "d3_random_walk_forage": round(_mean(D3_RUNG_ID, "random_walk"), 6),
            "a1_or_a3_cotrain_supra_floor_d3": cotrain_supra,
            "any_frozen_supra_floor_d3": frozen_supra,
            "any_actor_critic_supra_floor_d3": any_ac_supra,
            "bias_head_subfloor_d3": bias_subfloor_d3,
            "cotrain_normalized_frac_of_local_view_d3": (
                round(max(_mean(D3_RUNG_ID, "actor_critic_cotrain_plain"),
                          _mean(D3_RUNG_ID, "actor_critic_cotrain_sf")) / local_view_d3, 6)
                if local_view_d3 > 1e-9 else None
            ),
        },
        "reward_hacking_guard": {
            "teacher": "foraging_reward",
            "load_bearing": False,
            "note": (
                "Foraging teacher: training return IS foraging, so this divergence is a "
                "substrate/optimisation sanity readout, not a reward-hacking gate. Load-bearing "
                "ONLY if the substrate benefit-harm teacher (ARC-108) were used (it is not)."
            ),
            "per_arm_d3": guard_block,
        },
        "per_rung": per_rung,
        "denominators": {
            "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
            "local_view_greedy_d3_live": round(local_view_d3, 6),
            "local_view_greedy_d3_738_reference": float(DENOM_738_D3_REFERENCE),
            "global_oracle_NOT_the_denominator": round(_mean(D3_RUNG_ID, "greedy_oracle"), 6),
        },
        "non_degenerate": bool(degeneracy["non_degenerate"]),
        "degeneracy_reason": degeneracy["degeneracy_reason"],
        "degenerate_metrics": degeneracy["degenerate_metrics"],
    }
    return result


def _build_manifest(result: Dict[str, Any], timestamp_utc: str, dry_run: bool) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "timestamp_utc": timestamp_utc,
        "dry_run": bool(dry_run),
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "interpretation": result["interpretation"],
        "interpretation_label": result["interpretation_label"],
        "readiness": result["readiness"],
        "headline": result["headline"],
        "reward_hacking_guard": result["reward_hacking_guard"],
        "denominators": result["denominators"],
        "per_rung": result["per_rung"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "degenerate_metrics": result["degenerate_metrics"],
        "arm_matrix": {a: AC_ARMS[a]["code"] for a in AC_ARMS},
        "config": {
            "seeds": SEEDS if not dry_run else DRY_RUN_SEEDS,
            "rungs": [r["rung_id"] for r in RUNGS],
            "arms": list(ARM_ORDER),
            "p0_warmup_episodes": P0_WARMUP_EPISODES if not dry_run else DRY_RUN_P0,
            "ac_train_episodes": AC_TRAIN_EPISODES if not dry_run else DRY_RUN_AC,
            "bias_head_p1_reinforce_episodes": x734.P1_REINFORCE_EPISODES if not dry_run else DRY_RUN_P0,
            "eval_episodes": EVAL_EPISODES if not dry_run else DRY_RUN_EVAL,
            "steps_per_episode": STEPS_PER_EPISODE if not dry_run else DRY_RUN_STEPS,
            "reward_teacher": "foraging (FORAGE_BONUS per resource + harm term + novelty)",
            "ac_lr": AC_LR, "ac_gamma": AC_GAMMA,
            "sf_td_coef": SF_TD_COEF, "sf_reward_reg_coef": SF_REWARD_REG_COEF,
            "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
        },
        "load_bearing_dv": (
            "D3 actor-critic foraging_competence (mean resources/ep, unshaped) vs the 1.0 "
            "competence floor, strict majority of seeds, denominator = live local_view_greedy "
            "(738 ceiling); MECH-457 supported iff an actor-critic arm clears while "
            "bias_head_baseline does not."
        ),
        "notes": (
            "MECH-457 first-class actor-critic ON/OFF validation. PROMOTES/DEMOTES NOTHING; route "
            "to /failure-autopsy (accepts a PASS target) before any governance action / MECH-457 "
            "promotion. deeper_than_action_learning routes to RE-AUTOPSY, NOT a lettered floor "
            "re-test (the 734/737 RESOLVED_BY_FANOUT refusal stands). Foraging teacher (user-"
            "confirmed); substrate benefit-harm teacher NOT used (would need ARC-030 grounding)."
        ),
    }
    return manifest


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-742 MECH-457 actor-critic ON/OFF validation (evidence; claim_ids=[MECH-457])"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        p0, p1, ac_eps = DRY_RUN_P0, DRY_RUN_P0, DRY_RUN_AC
        eval_eps, steps = DRY_RUN_EVAL, DRY_RUN_STEPS
    else:
        seeds = list(SEEDS)
        p0, p1, ac_eps = P0_WARMUP_EPISODES, x734.P1_REINFORCE_EPISODES, AC_TRAIN_EPISODES
        eval_eps, steps = EVAL_EPISODES, STEPS_PER_EPISODE

    result = run_experiment(seeds=seeds, p0=p0, p1=p1, ac_eps=ac_eps, eval_eps=eval_eps, steps=steps)

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run))

    out_dir = Path(args.out_dir) if args.out_dir is not None else (
        REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{manifest['run_id']}.json"
    if args.dry_run:
        out_path = out_dir / f"_dry_{manifest['run_id']}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"manifest: {out_path}", flush=True)
    if not args.dry_run:
        print(f"Result written to: {out_path}", flush=True)
    hl = result["headline"]
    print(
        f"outcome: {result['outcome']} label={result['interpretation_label']} "
        f"dir={result['evidence_direction']} readiness_met={result['readiness']['readiness_met']} "
        f"non_degenerate={result['non_degenerate']}",
        flush=True,
    )
    print(
        f"  D3: a0={hl['d3_a0_frozen_plain_forage']} a1={hl['d3_a1_cotrain_plain_forage']} "
        f"a2={hl['d3_a2_frozen_sf_forage']} a3={hl['d3_a3_cotrain_sf_forage']} "
        f"bias={hl['d3_bias_head_baseline_forage']} local_view={hl['d3_local_view_greedy_denominator']} "
        f"(cotrain_supra={hl['a1_or_a3_cotrain_supra_floor_d3']} any_ac_supra={hl['any_actor_critic_supra_floor_d3']})",
        flush=True,
    )

    if args.dry_run:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass

    outcome_norm = str(result["outcome"]).upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    manifest_for_sentinel = str(out_path) if not args.dry_run else None
    return outcome_emit, manifest_for_sentinel, bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry_run = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path, dry_run=_dry_run)
