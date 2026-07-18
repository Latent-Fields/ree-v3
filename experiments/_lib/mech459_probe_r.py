"""MECH-459 probe R instrumentation: advantage composition + critic calibration.

WHY THIS IS A SEPARATE MODULE (and not a hook added to mech457_explorer_classes.train_a2c).
V3-EXQ-780 (ree-cloud-2) and V3-EXQ-781 (ree-cloud-3) are CLAIMED and RUNNING against
mech457_explorer_classes.py / mech457_bootstrap_explorer.py / mech457_fanout.py. Editing any of
those files -- even to add a default-None probe hook that is numerically byte-identical when OFF
-- changes their bytes and therefore the arm_fingerprint substrate_hash of the live portfolio,
and would be picked up by any worker that restarts mid-run. The 2026-07-18 MECH-459 adjudication
(decision_MECH-459_registry_and_brake_2026-07-18.md) explicitly requires probe R to leave
780/781 untouched. So this module MIRRORS the composed-bootstrap episode loop rather than
hooking it.

FAITHFULNESS CONTRACT (what makes the mirror a valid read of the real path). Every quantity that
defines the update operator is IMPORTED from the live modules, not re-declared here:
  * reward composition  -- x734.FORAGE_BONUS, x734._novelty_bonus, x734._RunningStd,
                           x734.REWARD_STD_EPS  (mirrors explorer_classes.py:646-665)
  * running-std divide  -- scale = reward_std.std + x734.REWARD_STD_EPS   (:688)
  * GAE                 -- x734._compute_gae (PPO_GAMMA / PPO_GAE_LAMBDA)  (:690)
  * standardisation     -- (adv - adv.mean()) / (adv.std() + 1e-8)          (:697)
  * loss + optimiser    -- mech.AC_LR / AC_VALUE_COEF / AC_GRAD_CLIP        (:698-712)
  * credit replay       -- mech._prioritized_credit_replay                  (:715-720)
  * schedules           -- boot.warm_then_anneal                            (bootstrap_explorer)
The ONLY additions are read-only accumulators. The probe never changes an update.

READOUT R-(a) -- ADVANTAGE COMPOSITION, logged BOTH pre- and post-standardisation.
Per training step, classify the step by what happened at it:
    forage      : info["transition_type"] == "resource"   (a forage CONTACT)
    harm        : harm_signal != 0                        (a harm event)
    novelty_only: neither                                 (the novelty bonus fires EVERY step)
and accumulate |adv_t| mass per class, once BEFORE the per-episode standardisation
(adv = GAE over the running-std-scaled rewards) and once AFTER it.

ANALYTIC NOTE THE GRID DEPENDS ON (state it so the reading is not over-claimed). The
standardisation is (a - mean) / (std + eps). Dividing by a positive scalar is a GLOBAL rescale
and therefore leaves every |adv| mass FRACTION exactly invariant. So the ONLY operation that can
move the composition is the MEAN SUBTRACTION. A "rescaled to parity" signature is thus a
specific, demanding claim: the episode-mean advantage must be large and negative relative to the
forage-step advantages. This is what makes R-(a) a real discriminator rather than a formality.

PREVALENCE NORMALISATION. Forage contacts are RARE, so a small forage mass fraction is partly
just rarity. The load-bearing statistic is therefore the CONCENTRATION
    C = (forage |adv| mass fraction) / (forage step fraction)
C == 1 means forage steps carry exactly their per-step share of the gradient; C < 1 means they
carry LESS than their share; C > 1 means the gradient is concentrated on them.

READOUT R-(b) -- CRITIC CALIBRATION ON BC-VISITED STATES.
LocalViewGreedyPolicy (the BC demonstrator; 48.05 @D3, and the policy V3-EXQ-748 distilled to
32.72) is rolled out against the TRAINED critic. At every visited state we record V(s) and the
realized discounted return-to-go G_t computed on the SAME reward scale the critic was trained on
(shaped reward / the trainer's final running-std scale). Because the demonstrator actually
forages, the G distribution is genuinely bimodal (pre-reward states are worth a lot; states far
from any resource are worth little), so a critic that has learned anything must separate them.

  MSE / bimodal-collapse : std(V) << std(G) AND mean(V) ~ mean(G) -- V has collapsed onto the
                           never-observed MEAN of a bimodal return distribution.
  signal-absence         : std(V) << std(G) AND mean(V) NOT near mean(G) -- a flat critic that
                           never learned the mean either.
  calibrated             : std(V) comparable to std(G) and pre/post-reward states separated.

ASCII-only in every runtime string (Windows cp1252 runners).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

import experiments._lib.mech457_bootstrap_explorer as boot
import experiments._lib.mech457_explorer_classes as mech
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734

DEVICE = mech.DEVICE

# --- step classes for the R-(a) composition readout -------------------------------------
STEP_CLASSES: Tuple[str, ...] = ("forage", "harm", "novelty_only")


class _MassAccumulator:
    """Accumulates |adv| mass per step class, pre- and post-standardisation, plus step counts."""

    def __init__(self) -> None:
        self.pre: Dict[str, float] = {c: 0.0 for c in STEP_CLASSES}
        self.post: Dict[str, float] = {c: 0.0 for c in STEP_CLASSES}
        self.steps: Dict[str, int] = {c: 0 for c in STEP_CLASSES}
        self.n_episodes = 0
        self.ep_mean_adv: List[float] = []   # the subtracted mean -- the only fraction-mover
        self.ep_std_adv: List[float] = []

    def add_step(self, cls: str, adv_pre: float, adv_post: float) -> None:
        self.steps[cls] += 1
        self.pre[cls] += abs(float(adv_pre))
        self.post[cls] += abs(float(adv_post))

    def add_episode(self, mean_adv: float, std_adv: float) -> None:
        self.n_episodes += 1
        self.ep_mean_adv.append(float(mean_adv))
        self.ep_std_adv.append(float(std_adv))

    def report(self) -> Dict[str, Any]:
        tot_pre = sum(self.pre.values())
        tot_post = sum(self.post.values())
        tot_steps = sum(self.steps.values())

        def _frac(d: Dict[str, float], tot: float) -> Dict[str, float]:
            return {c: (round(d[c] / tot, 8) if tot > 0.0 else 0.0) for c in STEP_CLASSES}

        step_frac = {
            c: (round(self.steps[c] / tot_steps, 8) if tot_steps > 0 else 0.0)
            for c in STEP_CLASSES
        }
        pre_frac = _frac(self.pre, tot_pre)
        post_frac = _frac(self.post, tot_post)

        def _conc(frac: Dict[str, float]) -> Dict[str, float]:
            return {
                c: (round(frac[c] / step_frac[c], 6) if step_frac[c] > 0.0 else 0.0)
                for c in STEP_CLASSES
            }

        n_mean = len(self.ep_mean_adv)
        return {
            "n_episodes_measured": int(self.n_episodes),
            "n_steps_measured": int(tot_steps),
            "n_steps_by_class": dict(self.steps),
            "step_fraction_by_class": step_frac,
            "abs_adv_mass_pre_standardisation": {c: round(self.pre[c], 6) for c in STEP_CLASSES},
            "abs_adv_mass_post_standardisation": {c: round(self.post[c], 6) for c in STEP_CLASSES},
            "abs_adv_mass_fraction_pre": pre_frac,
            "abs_adv_mass_fraction_post": post_frac,
            "concentration_pre": _conc(pre_frac),
            "concentration_post": _conc(post_frac),
            "mean_episode_mean_adv": (
                round(float(sum(self.ep_mean_adv) / n_mean), 8) if n_mean else 0.0
            ),
            "mean_episode_std_adv": (
                round(float(sum(self.ep_std_adv) / n_mean), 8) if n_mean else 0.0
            ),
        }


def _classify_step(transition_type: str, harm_signal: float) -> str:
    """Priority forage > harm > novelty_only. A step can be both a forage contact and a harm
    event; the forage label is the one the readout is about, so it wins."""
    if transition_type == "resource":
        return "forage"
    if abs(float(harm_signal)) > 0.0:
        return "harm"
    return "novelty_only"


# =========================================================================================
# R-(a): the composed-bootstrap trainer, mirrored with read-only advantage instrumentation.
# Mirrors mech457_explorer_classes.train_a2c for the composed-bootstrap call path only
# (RND intrinsic + coef/entropy schedules + prioritized credit replay). The hooks the probe
# does NOT exercise -- mode_gate, GoExploreArchive/return_prob, bc_demo/bc_aux, approach_drive
# -- are deliberately absent rather than stubbed, so there is no dead branch to drift.
# =========================================================================================
def train_a2c_probed(
    rep: mech.RepAgent, env: Any, seed: int, n_episodes: int, steps: int, arm_label: str,
    denom: int,
    *,
    intrinsic: Optional[mech.RNDModule] = None,
    entropy_beta: float = mech.AC_ENTROPY_BETA,
    intrinsic_coef: float = 0.0,
    credit_replay: bool = False,
    credit_replay_passes: int = mech.CREDIT_REPLAY_PASSES,
    credit_topk: int = mech.CREDIT_TOPK,
    coef_schedule: Optional[Callable[[int, int], float]] = None,
    entropy_schedule: Optional[Callable[[int, int], float]] = None,
    measure_window: int = 200,
) -> Dict[str, Any]:
    """Train and return the usual guard dict PLUS `adv_composition` (early + late windows).

    `measure_window` episodes are accumulated at the START of training (the "early" window,
    a covariate showing whether the composition is a training-stage artifact) and at the END
    (the "late" window, which is the load-bearing read -- the converged gradient composition)."""
    params = rep.params()
    optimiser = torch.optim.Adam(params, lr=mech.AC_LR)
    reward_std = x734._RunningStd()
    novelty_counter: Dict[Tuple[int, int], int] = {}
    train_forage_recent: deque = deque(maxlen=mech.TRAIN_FORAGE_WINDOW)
    intrinsic_recent: deque = deque(maxlen=mech.TRAIN_FORAGE_WINDOW)
    n_credit_passes = 0

    win = max(1, int(measure_window))
    early = _MassAccumulator()
    late = _MassAccumulator()
    late_start = max(win, int(n_episodes) - win)

    for ep in range(n_episodes):
        _flat, obs_dict = env.reset()
        rep.reset_episode()

        ep_logp: List[torch.Tensor] = []
        ep_value_t: List[torch.Tensor] = []
        ep_entropy: List[torch.Tensor] = []
        ep_value_f: List[float] = []
        ep_rewards: List[float] = []
        ep_classes: List[str] = []
        replay_obs: List[Dict[str, Any]] = []
        replay_actions: List[int] = []
        terminal = False
        bootstrap_value = 0.0
        ep_resources = 0
        ep_intrinsic = 0.0
        cum_resources = 0

        beta_eff = (
            float(entropy_schedule(ep, n_episodes)) if entropy_schedule is not None
            else entropy_beta
        )
        coef_eff = (
            float(coef_schedule(ep, n_episodes)) if coef_schedule is not None
            else intrinsic_coef
        )

        state = rep.encode(obs_dict)
        for _step in range(steps):
            z_prev = rep.z_detached(state)
            step = rep.step(state, deterministic=False)
            a_idx = int(step.action.reshape(-1)[0].item())
            if credit_replay:
                replay_obs.append(obs_dict)
                replay_actions.append(a_idx)
            _flat, harm_signal, done, info, obs_dict = env.step(a_idx)
            ttype = str(info.get("transition_type", "none"))
            if ttype == "resource":
                ep_resources += 1
                cum_resources += 1
            pos = (int(env.agent_x), int(env.agent_y))
            next_state = rep.encode(obs_dict)
            z_next = rep.z_detached(next_state)
            shaped = (
                float(harm_signal)
                + (x734.FORAGE_BONUS if ttype == "resource" else 0.0)
                + x734._novelty_bonus(novelty_counter, pos)
            )
            if intrinsic is not None:
                intr = float(intrinsic.intrinsic_reward(z_prev, a_idx, z_next))
                ep_intrinsic += intr
                shaped += coef_eff * intr
            reward_std.update(shaped)

            ep_logp.append(step.log_prob.reshape(-1)[0])
            ep_value_t.append(step.value.reshape(-1)[0])
            ep_entropy.append(step.entropy.reshape(-1)[0])
            ep_value_f.append(float(step.value.reshape(-1)[0].item()))
            ep_rewards.append(shaped)
            ep_classes.append(_classify_step(ttype, harm_signal))

            if done:
                terminal = True
                break
            state = next_state

        if not terminal:
            with torch.no_grad():
                bootv = rep.step(state, deterministic=False)
            bootstrap_value = float(bootv.value.reshape(-1)[0].item())

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

            # ---- R-(a) PROBE: read the advantage vector on BOTH sides of the standardiser ----
            acc: Optional[_MassAccumulator] = None
            if ep < win:
                acc = early
            elif ep >= late_start:
                acc = late
            adv_pre_list = [float(v) for v in advs]
            ep_mean = float(adv_t.mean().item()) if T > 0 else 0.0
            ep_std = float(adv_t.std().item()) if T > 1 else 0.0
            # -------------------------------------------------------------------------------

            if T > 1:
                adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            if acc is not None:
                adv_post_list = [float(v) for v in adv_t.detach().cpu().reshape(-1).tolist()]
                for t in range(T):
                    acc.add_step(ep_classes[t], adv_pre_list[t], adv_post_list[t])
                acc.add_episode(ep_mean, ep_std)

            policy_loss = -(logp_t * adv_t.detach()).mean()
            value_loss = mech.AC_VALUE_COEF * 0.5 * (value_t - ret_t.detach()).pow(2).mean()
            entropy_bonus = entropy_t.mean()
            loss = policy_loss + value_loss - beta_eff * entropy_bonus
            if torch.isfinite(loss):
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, mech.AC_GRAD_CLIP)
                optimiser.step()

            if credit_replay and ep_resources > 0:
                n_credit_passes += mech._prioritized_credit_replay(
                    rep, optimiser, params, replay_obs, replay_actions, rets, ep_value_f,
                    passes=int(credit_replay_passes), topk=int(credit_topk),
                )

        if intrinsic is not None:
            intrinsic.update()

        train_forage_recent.append(ep_resources)
        intrinsic_recent.append(ep_intrinsic / max(1, T))
        cur = ep + 1
        if cur % 200 == 0 or cur == n_episodes:
            print(
                f"  [train] {arm_label} seed={seed} phase=RL ep {cur}/{denom} "
                f"(credit_passes={n_credit_passes})",
                flush=True,
            )

    mtf = (
        float(sum(train_forage_recent) / len(train_forage_recent))
        if train_forage_recent else 0.0
    )
    mir = float(sum(intrinsic_recent) / len(intrinsic_recent)) if intrinsic_recent else 0.0
    return {
        "mean_train_forage_recent": round(mtf, 6),
        "mean_intrinsic_reward_recent": round(mir, 6),
        "n_credit_replay_passes": int(n_credit_passes),
        "final_reward_std_scale": round(float(reward_std.std + x734.REWARD_STD_EPS), 8),
        "adv_composition": {
            "measure_window_episodes": win,
            "early": early.report(),
            "late": late.report(),
        },
    }


def train_probed_bootstrap(
    rep: mech.RepAgent, env: Any, seed: int, steps: int, arm_label: str,
    cfg: boot.BootstrapExplorerConfig, denom: Optional[int] = None,
    measure_window: int = 200,
) -> Dict[str, Any]:
    """The composed bootstrap (RND + developmental anneal + credit replay), probed.

    Mirrors boot.train_bootstrap_explorer's wiring exactly, calling train_a2c_probed instead of
    mech.train_a2c. cfg must be a NO-BC / NO-approach reference config (this probe measures the
    reference ctrl path, which is the shared control arm of both 780 and 781)."""
    if float(cfg.bc_aux_coef) > 0.0 or bool(cfg.use_approach_primitive):
        raise ValueError(
            "train_probed_bootstrap: probe R measures the reference ctrl path only; "
            "bc_aux_coef must be 0 and use_approach_primitive False."
        )
    n_episodes = int(cfg.n_episodes)
    denom = int(denom) if denom is not None else n_episodes
    intrinsic = mech.RNDModule(rep.feature_dim) if cfg.use_rnd else None
    coef_schedule = (
        lambda ep, n: boot.warm_then_anneal(
            cfg.intrinsic_coef_start, cfg.intrinsic_coef_end,
            cfg.warm_start_fraction, cfg.anneal_fraction, ep, n
        )
    )
    entropy_schedule = (
        lambda ep, n: boot.warm_then_anneal(
            cfg.entropy_beta_start, cfg.entropy_beta_end,
            cfg.warm_start_fraction, cfg.anneal_fraction, ep, n
        )
    )
    return train_a2c_probed(
        rep, env, seed=seed, n_episodes=n_episodes, steps=steps,
        arm_label=arm_label, denom=denom,
        intrinsic=intrinsic,
        entropy_beta=cfg.entropy_beta_start,
        intrinsic_coef=cfg.intrinsic_coef_start,
        credit_replay=bool(cfg.credit_replay),
        credit_replay_passes=int(cfg.credit_replay_passes),
        credit_topk=int(cfg.credit_topk),
        coef_schedule=coef_schedule,
        entropy_schedule=entropy_schedule,
        measure_window=measure_window,
    )


# =========================================================================================
# R-(b): critic calibration on BC-visited states.
# =========================================================================================
def _mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def _std(vals: List[float]) -> float:
    n = len(vals)
    if n < 2:
        return 0.0
    m = _mean(vals)
    return float((sum((v - m) ** 2 for v in vals) / (n - 1)) ** 0.5)


def critic_calibration_on_demo_states(
    rep: mech.RepAgent, env: Any, demo: Any, seed: int, episodes: int, steps: int,
    reward_scale: float, pre_reward_horizon: int = 10,
) -> Dict[str, Any]:
    """Roll the BC demonstrator out and compare the TRAINED critic V(s) to the realized
    discounted return-to-go G_t at each visited state.

    `reward_scale` is the trainer's FINAL running-std scale -- the critic's regression targets
    were GAE returns over shaped/scale, so G must be computed on that same scale or the
    comparison is a unit error rather than a calibration reading.

    States are split into `pre_reward` (a forage contact occurs within `pre_reward_horizon`
    steps ahead) and `far` (it does not). The demonstrator forages, so both classes are
    populated and G is genuinely bimodal -- which is exactly what makes a collapsed critic
    visible."""
    gamma = float(x734.PPO_GAMMA)
    scale = float(reward_scale) if float(reward_scale) > 0.0 else 1.0
    novelty_counter: Dict[Tuple[int, int], int] = {}

    values: List[float] = []
    returns: List[float] = []
    is_pre_reward: List[bool] = []
    n_contacts = 0

    for _ep in range(int(episodes)):
        _flat, obs_dict = env.reset()
        rep.reset_episode()
        ep_values: List[float] = []
        ep_rewards: List[float] = []
        ep_contact: List[bool] = []
        for _step in range(int(steps)):
            state = rep.encode(obs_dict)
            with torch.no_grad():
                sv = rep.step(state, deterministic=True)
            ep_values.append(float(sv.value.reshape(-1)[0].item()))
            a_idx = int(demo.act(env, obs_dict))
            _flat, harm_signal, done, info, obs_dict = env.step(a_idx)
            ttype = str(info.get("transition_type", "none"))
            contact = (ttype == "resource")
            if contact:
                n_contacts += 1
            pos = (int(env.agent_x), int(env.agent_y))
            shaped = (
                float(harm_signal)
                + (x734.FORAGE_BONUS if contact else 0.0)
                + x734._novelty_bonus(novelty_counter, pos)
            )
            ep_rewards.append(shaped / scale)
            ep_contact.append(contact)
            if done:
                break

        T = len(ep_values)
        g = 0.0
        ep_returns = [0.0] * T
        for t in reversed(range(T)):
            g = ep_rewards[t] + gamma * g
            ep_returns[t] = g
        for t in range(T):
            hi = min(T, t + int(pre_reward_horizon) + 1)
            is_pre_reward.append(any(ep_contact[t:hi]))
        values.extend(ep_values)
        returns.extend(ep_returns)

    n = len(values)
    v_mean, v_std = _mean(values), _std(values)
    g_mean, g_std = _mean(returns), _std(returns)

    pre_v = [values[i] for i in range(n) if is_pre_reward[i]]
    far_v = [values[i] for i in range(n) if not is_pre_reward[i]]
    pre_g = [returns[i] for i in range(n) if is_pre_reward[i]]
    far_g = [returns[i] for i in range(n) if not is_pre_reward[i]]

    value_sep = _mean(pre_v) - _mean(far_v)
    return_sep = _mean(pre_g) - _mean(far_g)
    sep_ratio = float(value_sep / return_sep) if abs(return_sep) > 1e-9 else 0.0

    # Pearson correlation between V and G (0.0 when either side is constant).
    corr = 0.0
    if n >= 2 and v_std > 1e-12 and g_std > 1e-12:
        cov = sum((values[i] - v_mean) * (returns[i] - g_mean) for i in range(n)) / (n - 1)
        corr = float(cov / (v_std * g_std))

    return {
        "n_demo_states": int(n),
        "n_demo_forage_contacts": int(n_contacts),
        "reward_scale_used": round(scale, 8),
        "pre_reward_horizon": int(pre_reward_horizon),
        "value_mean": round(v_mean, 6), "value_std": round(v_std, 6),
        "value_min": round(min(values), 6) if values else 0.0,
        "value_max": round(max(values), 6) if values else 0.0,
        "return_mean": round(g_mean, 6), "return_std": round(g_std, 6),
        "return_min": round(min(returns), 6) if returns else 0.0,
        "return_max": round(max(returns), 6) if returns else 0.0,
        "value_std_over_return_std": round(float(v_std / g_std), 6) if g_std > 1e-12 else 0.0,
        "value_minus_return_mean_in_return_sd": (
            round(float(abs(v_mean - g_mean) / g_std), 6) if g_std > 1e-12 else 0.0
        ),
        "n_pre_reward_states": len(pre_v), "n_far_states": len(far_v),
        "pre_reward_value_mean": round(_mean(pre_v), 6),
        "far_value_mean": round(_mean(far_v), 6),
        "pre_reward_return_mean": round(_mean(pre_g), 6),
        "far_return_mean": round(_mean(far_g), 6),
        "value_separation": round(value_sep, 6),
        "return_separation": round(return_sep, 6),
        "separation_ratio": round(sep_ratio, 6),
        "value_return_correlation": round(corr, 6),
    }
