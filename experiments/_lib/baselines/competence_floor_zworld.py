"""Shared z_world cell machinery for the competence_floor R4 instrument-validity leg.

WHY THIS EXISTS. V3-EXQ-819 asks whether a PREDICTION-TRAINED z_world confers any
competence advantage over a FROZEN RANDOM-PROJECTION z_world of the same dimensionality
(re-pose R4 / D12; competence_floor_reposing_2026-07-20.md section 4 R4). The two arms are
built here through ONE module so the RANDOM-PROJECTION control arm and the TRAINED treatment
arm agree by CONSTRUCTION rather than by two code paths happening to spell the same config,
and so a later sibling leg can reuse the minted control cell across a different driver
(first-of-lineage factoring, per the /queue-experiment skill).

THE ONE DECLARED MANIPULATION is the P0a SD-070 z_world encoder warmup (`zworld_p0`):
  * RANDOM control  -- warmup_arm(trained=False): zworld_p0=0. The all-ON P0 warmup does NOT
       touch split_encoder.world_encoder (the documented V3-EXQ-780 defect), so the encoder
       stays BIT-IDENTICAL to its random initialisation -- a frozen random projection. This is
       the measured baseline that reached the 748 z_world+BC band (32.72) while running AS the
       treatment for sixteen legs. The zworld_encoder_guard is EXPECTED to fire here; that is
       the deliberate condition under test, not a substrate failure, so warmup_arm uses
       strict=False and the driver scopes the encoder-trained precondition OUT of this arm.
  * TRAINED treat   -- warmup_arm(trained=True): zworld_p0=ZWORLD_P0_EPISODES (60, the
       SD-070-validated operating point; ree-v3 b523b9c70a, validated REE_assembly 25a69fcd4c
       2026-07-22 -- world_encoder tensors move on every seed). run_zworld_p0 (P0a) runs AHEAD
       of the e2 warmup inside _train_all_on_agent, so the encoder is prediction-trained. The
       encoder-trained precondition is GATING on this arm: a guard fire here means the install
       did not take -> substrate_not_ready_requeue for this arm, NEVER a scientific verdict.

EVERYTHING ELSE IS IDENTICAL ACROSS THE TWO ARMS BY CONSTRUCTION. Both build the same
make_zworld_agent(cotrain=False, hidden=128) -- z_world DETACHED, so the encoder is FROZEN
throughout BC + RL in BOTH arms and only its initial content (trained vs random) differs. That
is what makes this a clean instrument-validity contrast: the representation is fixed during
learning, the actor-critic head is trained identically, and the only thing that can move the
competence trajectory is what the encoder encodes. Reference build (NON-REGRESSED, NOT the
769-falsified 256/5x): 128-wide actor-critic, 3x budget, z_world detached. Credit-replay 3 /
topk 32 are raw_view bootstrap-explorer knobs with no analogue on the z_world A2C path the
16-leg campaign's z_world arms (and the 32.72 baseline) actually ran on, so they do not apply
here; the contrast is z_world-vs-z_world, so the refinement details cancel.

THE DV IS A COMPETENCE TRAJECTORY, NOT A TERMINAL SCALAR (measurement_requirement 1). Every
arm x seed cell records post-BC install competence plus a mid-training competence probe on a
fixed cadence, so a "trained z_world installs an advantage that then decays" outcome is
visible (measurement_requirement 2). The probe is wrapped in _rng_neutral so it never
perturbs the training RNG stream (measurement neutrality), exactly as the retention harness's
substrate probe guarantees.

train_trajectory is adapted verbatim from mech457_fanout.train_zworld_ac_shaped (the z_world
A2C loop the fanout legs use) with two additions: a PERSISTENT optimizer across the whole
budget (so the trajectory is a single continuous refinement, not a sequence of Adam resets)
and a periodic competence probe. All the RL math primitives (GAE, reward std, novelty bonus,
critic value loss, _sense) are the shared, tested functions -- nothing about the update rule
is re-derived here.

ASCII-only in all runtime strings (repo rule).
"""

from __future__ import annotations

import contextlib
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from experiments._lib.capability_eval import (
    COMPETENCE_RESOURCE_FLOOR,
    evaluate_seed,
)
import experiments._lib.mech457_fanout as fan
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734
import experiments.v3_exq_742_mech457_actor_critic_onoff as x742
# Trajectory statistics are representation-agnostic (they read foraging_competence / episode
# off the trajectory dicts), so reuse the retention lineage's shared DV machinery rather than
# spelling a second copy that could drift from it.
from experiments._lib.baselines.mech457_retention import (
    competence_half_life,
    retained_fraction,
)

__all__ = [
    "LINEAGE",
    "OFF_ARM_ID",
    "ON_ARM_ID",
    "P0_WARMUP_EPISODES",
    "ZWORLD_P0_EPISODES",
    "BC_WARMSTART_EPISODES",
    "POST_BC_INSTALL_FLOOR",
    "REF_ACTOR_CRITIC_HIDDEN",
    "REF_BUDGET_MULTIPLIER",
    "SHAPING_COEF",
    "arm_config_slice",
    "build_zworld_agent",
    "warmup_arm",
    "install_bc",
    "train_trajectory",
    "competence_half_life",
    "retained_fraction",
]

DEVICE = fan.DEVICE

LINEAGE = "competence_floor_zworld"
OFF_ARM_ID = "zworld_random_proj"   # frozen random-projection z_world (control / measured baseline)
ON_ARM_ID = "zworld_trained"        # SD-070 prediction-trained z_world (treatment)

# --- Warmup schedule (the ONE manipulation lives in zworld_p0) -----------------------------
P0_WARMUP_EPISODES = fan.P0_WARMUP_EPISODES         # 200 -- all-ON e2 warmup, IDENTICAL both arms
ZWORLD_P0_EPISODES = fan.ZWORLD_P0_EPISODES         # 60  -- SD-070 encoder warmup, TRAINED arm ONLY

# --- Install + reference build (128-wide / 3x budget / z_world detached) --------------------
BC_WARMSTART_EPISODES = fan.BC_EPISODES             # 300 supervised BC-clone rollouts
POST_BC_INSTALL_FLOOR = COMPETENCE_RESOURCE_FLOOR   # an install below this did NOT take
REF_ACTOR_CRITIC_HIDDEN = fan.ACTOR_CRITIC_HIDDEN   # 128 (NOT the 769-falsified 256)
REF_BUDGET_MULTIPLIER = 3                            # 3x (NOT the 769-falsified 5x)
REF_COTRAIN_ENCODER = False                         # z_world DETACHED -- encoder frozen in BC+RL
# Sparse foraging teacher (742's E0): isolates RETENTION of the installed prior from re-teaching
# by dense shaping. Identical across both arms, so it cannot bear on the trained-vs-random contrast.
SHAPING_COEF = 0.0


@contextlib.contextmanager
def _rng_neutral():
    """Restore the global torch + numpy RNG streams on exit.

    A mid-training competence probe builds a fresh env and rolls the eval policy, which can draw
    from the global torch / numpy streams; without this the probe would shift every downstream
    training draw and confound the measurement with a pure-noise offset. Snapshotting makes the
    probe a no-op on the training stream -- the measurement neutrality the retention harness gets
    from the substrate, provided here around each reading.
    """
    torch_state = torch.get_rng_state()
    numpy_state = np.random.get_state()
    try:
        yield
    finally:
        torch.set_rng_state(torch_state)
        np.random.set_state(numpy_state)


def arm_config_slice(
    env_kwargs: Dict[str, Any], *, trained: bool, eval_eps: int, steps: int,
    on_budget: int, probe_every: int, p0: int, zworld_p0: int, n_bc: int,
) -> Dict[str, Any]:
    """The fingerprint config_slice for one arm. Declares ONLY what the arm's build+collect
    path reads: env layout, the warmup / install / refinement schedule, the operating config
    (hidden, detached encoder, sparse teacher, lr, gamma) and the ONE manipulation (zworld_p0).
    NOT the acceptance thresholds / margins (those are the driver's, not the computation's)."""
    return {
        "lineage": LINEAGE,
        "arm_id": ON_ARM_ID if trained else OFF_ARM_ID,
        "trained_encoder": bool(trained),
        "rung_id": fan.RUNG_ID,
        "env_kwargs": dict(env_kwargs),
        "representation": "z_world",
        "cotrain_encoder": REF_COTRAIN_ENCODER,
        "actor_critic_hidden": REF_ACTOR_CRITIC_HIDDEN,
        "p0_warmup_episodes": int(p0),
        "zworld_p0_episodes": int(zworld_p0),
        "bc_warmstart_episodes": int(n_bc),
        "on_budget_episodes": int(on_budget),
        "retention_probe_every": int(probe_every),
        "shaping_coef": float(SHAPING_COEF),
        "ac_lr": float(fan.AC_LR),
        "ac_gamma": float(fan.AC_GAMMA),
        "bc_lr": float(fan.BC_LR),
        "eval_episodes": int(eval_eps),
        "steps_per_episode": int(steps),
    }


def build_zworld_agent(seed: int, env_kwargs: Dict[str, Any]):
    """The all-ON REE stack + MECH-457 actor-critic on a DETACHED z_world (cotrain=False), so
    the encoder is frozen throughout BC + RL and only its initial content differs between arms.
    Identical construction for both arms."""
    env = x734._make_env(seed, env_kwargs)
    return fan.make_zworld_agent(
        env, cotrain=REF_COTRAIN_ENCODER, actor_critic_hidden=REF_ACTOR_CRITIC_HIDDEN,
    )


def warmup_arm(
    agent: Any, seed: int, env_kwargs: Dict[str, Any], *, steps: int, trained: bool,
    p0: int, zworld_p0: int, dry_run: bool = False,
) -> Dict[str, Any]:
    """Run the P0 warmup for one arm and return the zworld_encoder_guard audit record.

    strict=False on BOTH arms: the run must not crash on a frozen encoder. The RANDOM control
    is DEFINED by a frozen encoder (the guard firing is expected and non-gating); the TRAINED
    arm's guard result is fed to a GATING precondition in the driver instead, so a genuine
    training failure self-routes substrate_not_ready_requeue rather than raising. The warmup env
    is a dedicated instance (its rollout consumes env RNG; reusing a training env would shift
    the layout sequence the subsequent phases see)."""
    warm_env = x734._make_env(seed, env_kwargs)
    zp0 = int(zworld_p0) if trained else 0
    report = fan.warmup_zworld(
        agent, warm_env, seed, p0=int(p0), steps=int(steps), strict=False,
        zworld_p0=zp0, zworld_p0_dry_run=bool(dry_run),
    )
    report["arm_trained_encoder"] = bool(trained)
    report["zworld_p0_episodes"] = int(zp0)
    return report


def install_bc(
    agent: Any, seed: int, env_kwargs: Dict[str, Any], *, steps: int, eval_eps: int,
    n_bc: int, arm_label: str,
) -> Dict[str, Any]:
    """BC-clone LocalViewGreedyPolicy through the (detached) z_world path, then measure the
    post-BC / pre-RL competence -- the install point and the load-bearing readiness covariate."""
    bc_env = x734._make_env(seed, env_kwargs)
    bc_stats = fan.bc_warmup_zworld(
        agent, bc_env, seed, int(n_bc), int(steps), arm_label, denom=int(n_bc),
    )
    with _rng_neutral():
        eval_env = x734._make_env(seed, env_kwargs)
        row = evaluate_seed(
            x742.ActorCriticEvalPolicy(agent, f"{arm_label}_post_bc"),
            eval_env, int(eval_eps), int(steps),
        )
    post_bc = float(row["foraging_competence"])
    return {
        "post_bc_foraging_competence": round(post_bc, 6),
        "install_took": bool(post_bc >= POST_BC_INSTALL_FLOOR),
        "bc_action_match_recent": float(bc_stats.get("bc_action_match_accuracy_recent", 0.0)),
    }


def _probe(agent: Any, seed: int, env_kwargs: Dict[str, Any], *, steps: int, eval_eps: int,
           arm_label: str, episode: int) -> Dict[str, Any]:
    """One non-perturbing mid-training competence reading. Fresh env, deterministic-argmax eval
    policy, wrapped in _rng_neutral so the training stream is bit-identical to an unprobed run.
    Emits both `episode` and `foraging_competence` so competence_half_life (which reads
    `episode`) and retained_fraction (which reads the terminal `foraging_competence`) both work."""
    with _rng_neutral():
        probe_env = x734._make_env(seed, env_kwargs)
        row = evaluate_seed(
            x742.ActorCriticEvalPolicy(agent, f"{arm_label}_probe_ep{int(episode)}"),
            probe_env, int(eval_eps), int(steps),
        )
    return {"episode": int(episode), "foraging_competence": round(float(row["foraging_competence"]), 6)}


def train_trajectory(
    agent: Any, env: Any, seed: int, *, n_episodes: int, steps: int, arm_label: str,
    denom: int, probe_every: int, env_kwargs: Dict[str, Any], eval_eps: int,
    shaping_coef: float = SHAPING_COEF,
) -> Dict[str, Any]:
    """z_world A2C refinement of the BC-installed policy, probed on a fixed cadence.

    Adapted verbatim from mech457_fanout.train_zworld_ac_shaped (the z_world A2C loop) with a
    PERSISTENT optimizer over the whole budget and a periodic competence probe added -- the two
    changes the trajectory DV requires. With cotrain=False the encoder is detached inside
    agent.actor_critic_step, so although actor_critic_encoder_parameters() are in the optimizer
    they receive no gradient and the encoder stays FROZEN (trained or random, as warmed). Every
    RL primitive (GAE, reward std, novelty bonus, critic value term, _sense) is the shared,
    tested function."""
    params = list(agent.actor_critic_parameters()) + list(agent.actor_critic_encoder_parameters())
    optimiser = torch.optim.Adam(params, lr=fan.AC_LR)
    policy_ref = getattr(agent, "action_critic", None)
    reward_std = x734._RunningStd()
    novelty_counter: Dict[Tuple[int, int], int] = {}
    train_forage_recent: deque = deque(maxlen=fan.TRAIN_FORAGE_WINDOW)
    trajectory: List[Dict[str, Any]] = []

    for ep in range(int(n_episodes)):
        _flat, obs_dict = env.reset()
        agent.reset()
        ep_logp: List[torch.Tensor] = []
        ep_value_t: List[torch.Tensor] = []
        ep_entropy: List[torch.Tensor] = []
        ep_value_f: List[float] = []
        ep_value_logits: List[torch.Tensor] = []
        ep_rewards: List[float] = []
        terminal = False
        bootstrap_value = 0.0
        ep_resources = 0

        latent = x742._sense(agent, obs_dict)
        for _step in range(int(steps)):
            step = agent.actor_critic_step(latent, deterministic=False)
            a_idx = int(step.action.reshape(-1)[0].item())
            phi_s = fan._potential(env) if shaping_coef > 0.0 else 0.0
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
                phi_next = fan._potential(env)
                shaped += shaping_coef * (fan.AC_GAMMA * phi_next - phi_s)
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
            value_loss = fan.critic_value_loss(policy_ref, vlogits_t, value_t, ret_t)
            entropy_bonus = entropy_t.mean()
            loss = policy_loss + value_loss - fan.AC_ENTROPY_BETA * entropy_bonus
            if torch.isfinite(loss):
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, fan.AC_GRAD_CLIP)
                optimiser.step()

        train_forage_recent.append(ep_resources)
        cur = ep + 1
        # Non-perturbing competence probe on the fixed cadence + a guaranteed final reading.
        if (cur % int(probe_every) == 0) or (cur == int(n_episodes)):
            reading = _probe(
                agent, seed, env_kwargs, steps=steps, eval_eps=eval_eps,
                arm_label=arm_label, episode=cur,
            )
            trajectory.append(reading)
        if cur % 200 == 0 or cur == int(n_episodes):
            print(f"  [train] {arm_label} seed={seed} phase=RL ep {cur}/{denom}", flush=True)

    mtf = float(sum(train_forage_recent) / len(train_forage_recent)) if train_forage_recent else 0.0
    return {
        "mean_train_forage_recent": round(mtf, 6),
        "competence_trajectory": trajectory,
    }
