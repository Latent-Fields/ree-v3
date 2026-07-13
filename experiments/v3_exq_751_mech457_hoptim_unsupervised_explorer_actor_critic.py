#!/opt/local/bin/python3
"""V3-EXQ-751 -- MECH-457 GOV-FANOUT-1 leg D (H-optim) -- stronger UNSUPERVISED explorer DIAGNOSTIC.

DIAGNOSTIC discrimination probe (experiment_purpose=diagnostic; claim_ids=["MECH-457"] tags
relevance only -> excluded from governance confidence/conflict scoring). PROMOTES / DEMOTES
NOTHING. Routes to /failure-autopsy for adjudication before any governance action. MECH-457
stays candidate / v3_pending.

WHY. The 747/748/749 portfolio (combined autopsy failure_autopsy_MECH-457-fanout-747-748-749_
2026-07-13) closed the REPRESENTATION and REWARD-DENSITY axes: H-rep REFUTED (z_world+BC 32.72 >
raw+BC 20.93; raw+sparse ~ z_world+sparse), and reward SHAPING rescued neither representation
(0.217 / 0.767, both sub-floor). Only behavior-cloning (an EXPERT action-level teacher) cleared
the 1.0 floor. The discriminated cause was the RL exploration / credit-assignment bootstrap --
but ONE live discrimination remains on the ALGORITHM axis neither leg isolated (autopsy sec7.4 /
sec8): is an expert teacher NECESSARY, or was vanilla policy-gradient exploration merely too
WEAK -- would a stronger UNSUPERVISED explorer clear the floor WITHOUT an oracle to clone? This
leg is that empirical discrimination. The lit-pull (REE_assembly master 5883691b2b, 2026-07-13:
Burda2018 RND, Pathak2017 ICM, Hester2018 DQfD, Ross2011 DAgger, Kang2021 observational) has
landed; the autopsy deferred queuing this leg until it did.

THE ALGORITHM AXIS (representation pinned = z_world cotrain; teacher = UNSUPERVISED, no expert):
  742/748 sparse z_world actor-critic ALREADY carries entropy regularization (beta=0.03) + a
  count-based novelty bonus, yet foraged 0.20-0.27/ep. So "PPO+entropy" alone is the BASELINE,
  not the treatment. A genuinely STRONGER unsupervised explorer must go beyond that -- a LEARNED
  intrinsic-motivation signal + raised entropy. Two sibling explorer arms, both on the SAME
  z_world (cotrain) path via agent.actor_critic_step, NO expert / NO behavior-cloning targets:
    * ac_zworld_rnd -- Random Network Distillation (Burda et al. 2018): intrinsic reward =
      ||predictor(z_world) - frozen_random_target(z_world)||^2 (normalized), + raised entropy
      (beta=0.10). Novelty as prediction error against a fixed random projection.
    * ac_zworld_icm -- Intrinsic Curiosity Module (Pathak et al. 2017): intrinsic reward =
      forward-model prediction error 0.5*||f(z_t, a_t) - z_{t+1}||^2 (normalized), with a joint
      inverse-model auxiliary, + raised entropy (beta=0.10). Curiosity as own-dynamics surprise.
  Both intrinsic modules train on z_world.detach() (their objective does NOT perturb the cotrain
  policy gradient; the intrinsic reward is a scalar bonus, exactly as the count-novelty already
  is). Compared head-to-head, in the SAME run and seeds, against:
    * ac_zworld_sparse -- the 742 sparse-teacher z_world actor-critic (shaping_coef=0.0 through
      fan.train_zworld_ac_shaped = the byte-canonical 742 cotrain sparse arm). This is the OFF /
      baseline reference AND the reuse-eligible MINT (mint-as-you-go; see MINT below).

DECLARED NULL (so a sub-floor leg is informative, not wasted):
  * either explorer arm clears the 1.0 floor (strict majority of seeds) -> UNSUPERVISED
    exploration was the wall; a stronger explorer bootstraps the z_world policy WITHOUT an
    expert. SELF-ROUTE: stronger_unsupervised_explorer_clears_floor_exploration_was_the_wall.
    DECIDES the build target = a better unsupervised explorer (RND/ICM are viable but
    scale-discounted per Burda2018/Pathak2017 -- the owed build is engineering, not a new
    dependency).
  * BOTH explorer arms sub-floor while the portfolio's BC arm PASSED (748 z_world+BC 32.72) ->
    unsupervised exploration is NOT sufficient; an EXPERT action-level teaching signal /
    developmental scaffold is NECESSARY. SELF-ROUTE:
    unsupervised_exploration_insufficient_expert_teaching_signal_necessary. DECIDES the build
    target = imitation-PLUS-closed-loop-correction (NOT static BC: DAgger/Ross2011 shows pure BC
    has an O(eps*T^2) covariate-shift ceiling; DQfD/Hester2018 shows scaffold-then-continued-RL
    as the escape).

READINESS (P0 readiness-assert; SAME statistic as the verdict = foraging_competence @D3 vs the
1.0 floor). LocalViewGreedyPolicy (5x5 view) and greedy_oracle clear the floor @D3 (env solvable;
742 live denominators local_view_greedy=48.05, greedy_oracle=57.2, random_walk=0.933). Below
readiness -> substrate_not_ready_requeue (FAIL; NEVER a substrate-verdict label); explorer
training skipped.

INPUT CHOICE (z_world only; NOT re-run on raw view). The 747/748/749 portfolio already
discriminated the representation axis: H-rep is REFUTED and z_world+BC (32.72) BEATS raw+BC
(20.93), so z_world is action-ADEQUATE -- indeed the better control substrate. Re-running the
H-optim discrimination on the raw 5x5 view would only re-litigate a closed axis. This leg holds
representation fixed at z_world (the substrate's actual MECH-457 action-learning path, cotrain)
and varies ONLY the explorer strength -- the clean single-axis test the autopsy asked for.

MINT (mint-as-you-go, standing default). The ac_zworld_sparse OFF arm is emitted REUSE-ELIGIBLE
(rng_fully_reset via arm_cell + config_slice_declared + include_driver_script_in_hash=False), so
a later MECH-457 z_world-sparse consumer can cite reuse_baseline_from this run_id and skip
re-training it. Its OFF computation lives entirely in the _lib module fan.train_zworld_ac_shaped
(shaping_coef=0.0) + fan.make_zworld_agent + fan.warmup_zworld, all under experiments/_lib/** =
in the substrate hash, so the fingerprint refuses correctly on substrate drift. The two explorer
arms are marked reuse-INELIGIBLE (their intrinsic-module logic is DRIVER-defined, excluded from
the substrate hash by include_driver_script_in_hash=False -- they must never be reuse targets).
No compelling disqualifier applies (not Mac-only: machine_affinity any/cloud; substrate stable
for this lineage; not one-shot: live MECH-457 successor family), so the sparse baseline is
minted. A SEPARATE order-independent baseline-only mint is skipped WITH EYES OPEN: this leg
self-mints the sparse cell and is the high-priority decision-relevant probe, and no sibling
currently needs the z_world-sparse baseline BEFORE this leg lands (the ordering bet the standing
default names -- floor = self-mint here, order-agnostic ceiling deferred).

evidence_direction = "unknown" (DIAGNOSTIC; the discrimination verdict lives in
interpretation.label / discrimination_verdict, adjudicated by /failure-autopsy).

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

Shared machinery: experiments/_lib/mech457_fanout.py (imported read-only). The RND / ICM
intrinsic modules + the explorer A2C trainer live in THIS driver (excluded from the substrate
hash, so the explorer arms are reuse-ineligible by construction). ASCII-only in all runtime
strings.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.capability_eval import COMPETENCE_RESOURCE_FLOOR, evaluate_seed  # noqa: E402
from experiments._metrics import check_degeneracy  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
import experiments._lib.mech457_fanout as fan  # noqa: E402
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402
import experiments.v3_exq_742_mech457_actor_critic_onoff as x742  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_751_mech457_hoptim_unsupervised_explorer_actor_critic"
QUEUE_ID = "V3-EXQ-751"
CLAIM_IDS: List[str] = ["MECH-457"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

DEVICE = fan.DEVICE

# ---------------------------------------------------------------------------
# Explorer hyperparameters (leg-specific; the intrinsic-motivation additions the
# 742/748 sparse baseline does NOT carry). The baseline already has entropy beta 0.03 +
# count-based novelty, so a genuinely STRONGER explorer = a LEARNED intrinsic signal +
# raised entropy.
# ---------------------------------------------------------------------------
EXPLORE_ENTROPY_BETA = 0.10          # raised from the 0.03 sparse baseline (stronger entropy)
INTRINSIC_COEF = 1.0                 # weight on the normalized intrinsic reward (~ FORAGE_BONUS)
RND_FEAT_DIM = 64                    # RND random-projection output width
RND_HIDDEN = 128
RND_LR = 1e-3
ICM_HIDDEN = 128
ICM_LR = 1e-3
ICM_FORWARD_BETA = 0.2               # forward-loss weight in ICM's OWN objective (inverse = 0.8)

BASELINE_ARM = "ac_zworld_sparse"
EXPLORER_ARMS: Tuple[str, ...] = ("ac_zworld_rnd", "ac_zworld_icm")
ZWORLD_ARMS: Tuple[str, ...] = (BASELINE_ARM,) + EXPLORER_ARMS
ARM_ORDER: Tuple[str, ...] = ZWORLD_ARMS + fan.ANCHOR_ARMS

# 742 reference band for the sparse cotrain z_world arm (sanity that the OFF re-run reproduces it).
REFERENCE_742_SPARSE_LOW = 0.20
REFERENCE_742_SPARSE_HIGH = 0.27


# ---------------------------------------------------------------------------
# Intrinsic-motivation modules. Both train on z_world.detach() -- their objective does
# NOT reach the cotrain policy gradient; the intrinsic reward is a scalar bonus (exactly
# like the count-novelty already in the baseline).
# ---------------------------------------------------------------------------
class RNDModule:
    """Random Network Distillation (Burda et al. 2018). Novelty = predictor error against a
    FROZEN random target projection of z_world; normalized by a running std of the error."""

    def __init__(self, world_dim: int) -> None:
        self.target = nn.Sequential(
            nn.Linear(world_dim, RND_HIDDEN), nn.ReLU(),
            nn.Linear(RND_HIDDEN, RND_FEAT_DIM),
        ).to(DEVICE)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictor = nn.Sequential(
            nn.Linear(world_dim, RND_HIDDEN), nn.ReLU(),
            nn.Linear(RND_HIDDEN, RND_HIDDEN), nn.ReLU(),
            nn.Linear(RND_HIDDEN, RND_FEAT_DIM),
        ).to(DEVICE)
        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=RND_LR)
        self.rstd = x734._RunningStd()
        self._buf: List[torch.Tensor] = []

    def intrinsic_reward(self, z_prev: torch.Tensor, action: int, z_next: torch.Tensor) -> float:
        with torch.no_grad():
            t = self.target(z_next)
            p = self.predictor(z_next)
            err = float(((p - t) ** 2).mean().item())
        self.rstd.update(err)
        self._buf.append(z_next.detach())
        return err / (self.rstd.std + x734.REWARD_STD_EPS)

    def update(self) -> None:
        if not self._buf:
            return
        z = torch.cat(self._buf, dim=0)
        with torch.no_grad():
            tgt = self.target(z)
        pred = self.predictor(z)
        loss = ((pred - tgt) ** 2).mean()
        if torch.isfinite(loss):
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.predictor.parameters(), fan.AC_GRAD_CLIP)
            self.opt.step()
        self._buf = []


class ICMModule:
    """Intrinsic Curiosity Module (Pathak et al. 2017). Curiosity = forward-model prediction
    error on own dynamics 0.5*||f(z_t, a_t) - z_{t+1}||^2 (normalized); a joint inverse model
    (predict a_t from z_t, z_{t+1}) is trained as the standard auxiliary. Features = z_world
    directly (already a learned representation), used detached."""

    def __init__(self, world_dim: int, action_dim: int) -> None:
        self.world_dim = int(world_dim)
        self.action_dim = int(action_dim)
        self.forward_model = nn.Sequential(
            nn.Linear(world_dim + action_dim, ICM_HIDDEN), nn.ReLU(),
            nn.Linear(ICM_HIDDEN, world_dim),
        ).to(DEVICE)
        self.inverse_model = nn.Sequential(
            nn.Linear(2 * world_dim, ICM_HIDDEN), nn.ReLU(),
            nn.Linear(ICM_HIDDEN, action_dim),
        ).to(DEVICE)
        params = list(self.forward_model.parameters()) + list(self.inverse_model.parameters())
        self.opt = torch.optim.Adam(params, lr=ICM_LR)
        self.rstd = x734._RunningStd()
        self._buf: List[Tuple[torch.Tensor, int, torch.Tensor]] = []

    def _onehot(self, action: int) -> torch.Tensor:
        v = torch.zeros(1, self.action_dim, device=DEVICE)
        v[0, int(action)] = 1.0
        return v

    def intrinsic_reward(self, z_prev: torch.Tensor, action: int, z_next: torch.Tensor) -> float:
        with torch.no_grad():
            pred_next = self.forward_model(torch.cat([z_prev, self._onehot(action)], dim=-1))
            err = float((0.5 * (pred_next - z_next) ** 2).mean().item())
        self.rstd.update(err)
        self._buf.append((z_prev.detach(), int(action), z_next.detach()))
        return err / (self.rstd.std + x734.REWARD_STD_EPS)

    def update(self) -> None:
        if not self._buf:
            return
        zp = torch.cat([b[0] for b in self._buf], dim=0)
        zn = torch.cat([b[2] for b in self._buf], dim=0)
        acts = torch.tensor([b[1] for b in self._buf], dtype=torch.long, device=DEVICE)
        onehots = torch.zeros(len(self._buf), self.action_dim, device=DEVICE)
        onehots[torch.arange(len(self._buf)), acts] = 1.0
        pred_next = self.forward_model(torch.cat([zp, onehots], dim=-1))
        fwd_loss = 0.5 * ((pred_next - zn) ** 2).mean()
        inv_logits = self.inverse_model(torch.cat([zp, zn], dim=-1))
        inv_loss = F.cross_entropy(inv_logits, acts)
        loss = ICM_FORWARD_BETA * fwd_loss + (1.0 - ICM_FORWARD_BETA) * inv_loss
        if torch.isfinite(loss):
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            params = list(self.forward_model.parameters()) + list(self.inverse_model.parameters())
            nn.utils.clip_grad_norm_(params, fan.AC_GRAD_CLIP)
            self.opt.step()
        self._buf = []


# ---------------------------------------------------------------------------
# z_world (cotrain) A2C explorer trainer. Mirrors fan.train_zworld_ac_shaped (single-backward-
# per-episode, GAE) but: (a) raised entropy beta, (b) a LEARNED intrinsic bonus added to the
# per-step reward, (c) NO potential shaping and NO expert. The actor reads LIVE z_world so the
# policy gradient reaches latent_stack (cotrain); the intrinsic module trains on z_world.detach().
# ---------------------------------------------------------------------------
def _train_zworld_ac_explore(
    agent: Any, env: Any, seed: int, n_episodes: int, steps: int, arm_label: str, denom: int,
    intrinsic: Any, entropy_beta: float,
) -> Dict[str, Any]:
    params = list(agent.actor_critic_parameters()) + list(agent.actor_critic_encoder_parameters())
    optimiser = torch.optim.Adam(params, lr=fan.AC_LR)
    reward_std = x734._RunningStd()
    novelty_counter: Dict[Tuple[int, int], int] = {}
    train_forage_recent: deque = deque(maxlen=fan.TRAIN_FORAGE_WINDOW)
    intrinsic_recent: deque = deque(maxlen=fan.TRAIN_FORAGE_WINDOW)

    for ep in range(n_episodes):
        _flat, obs_dict = env.reset()
        agent.reset()
        ep_logp: List[torch.Tensor] = []
        ep_value_t: List[torch.Tensor] = []
        ep_entropy: List[torch.Tensor] = []
        ep_value_f: List[float] = []
        ep_rewards: List[float] = []
        terminal = False
        bootstrap_value = 0.0
        ep_resources = 0
        ep_intrinsic = 0.0

        latent = x742._sense(agent, obs_dict)
        for _step in range(steps):
            z_prev = latent.z_world.detach()
            step = agent.actor_critic_step(latent, deterministic=False)
            a_idx = int(step.action.reshape(-1)[0].item())
            _flat, harm_signal, done, info, obs_dict = env.step(a_idx)
            ttype = str(info.get("transition_type", "none"))
            if ttype == "resource":
                ep_resources += 1
            pos = (int(env.agent_x), int(env.agent_y))
            # Next latent (with grad for cotrain continuity; detached copy for the intrinsic module).
            next_latent = x742._sense(agent, obs_dict)
            z_next = next_latent.z_world.detach()
            intr = float(intrinsic.intrinsic_reward(z_prev, a_idx, z_next))
            ep_intrinsic += intr
            shaped = (
                float(harm_signal)
                + (x734.FORAGE_BONUS if ttype == "resource" else 0.0)
                + x734._novelty_bonus(novelty_counter, pos)
                + INTRINSIC_COEF * intr
            )
            reward_std.update(shaped)

            ep_logp.append(step.log_prob.reshape(-1)[0])
            ep_value_t.append(step.value.reshape(-1)[0])
            ep_entropy.append(step.entropy.reshape(-1)[0])
            ep_value_f.append(float(step.value.reshape(-1)[0].item()))
            ep_rewards.append(shaped)

            if done:
                terminal = True
                break
            latent = next_latent

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
            value_loss = fan.AC_VALUE_COEF * 0.5 * (value_t - ret_t.detach()).pow(2).mean()
            entropy_bonus = entropy_t.mean()
            loss = policy_loss + value_loss - entropy_beta * entropy_bonus
            if torch.isfinite(loss):
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, fan.AC_GRAD_CLIP)
                optimiser.step()

        # Intrinsic module update (separate optimizer, on detached z; does NOT touch the policy).
        intrinsic.update()

        train_forage_recent.append(ep_resources)
        intrinsic_recent.append(ep_intrinsic / max(1, T))
        cur = ep + 1
        if cur % 200 == 0 or cur == n_episodes:
            print(f"  [train] {arm_label} seed={seed} phase=RL ep {cur}/{denom}", flush=True)

    mtf = float(sum(train_forage_recent) / len(train_forage_recent)) if train_forage_recent else 0.0
    mir = float(sum(intrinsic_recent) / len(intrinsic_recent)) if intrinsic_recent else 0.0
    return {"mean_train_forage_recent": round(mtf, 6),
            "mean_intrinsic_reward_recent": round(mir, 6)}


def _arm_config_slice(arm_id: str, env_kwargs: Dict[str, Any], p0: int, rl_eps: int,
                      eval_eps: int, steps: int) -> Dict[str, Any]:
    base = {
        "arm_id": arm_id, "rung_id": fan.RUNG_ID, "env_kwargs": dict(env_kwargs),
        "eval_episodes": int(eval_eps), "steps_per_episode": int(steps),
    }
    if arm_id in ZWORLD_ARMS:
        base.update({
            "kind": "zworld_actor_critic", "representation": "z_world_cotrain",
            "cotrain_encoder": True, "use_sf_critic": False,
            "actor_critic_hidden": fan.ACTOR_CRITIC_HIDDEN,
            "p0_warmup_episodes": int(p0), "rl_episodes": int(rl_eps),
        })
        if arm_id == BASELINE_ARM:
            # The 742 sparse cotrain arm: foraging RL + count-novelty + entropy 0.03, NO shaping.
            base.update({"teacher": "sparse_foraging_rl", "shaping_coef": 0.0,
                         "entropy_beta": fan.AC_ENTROPY_BETA})
        elif arm_id == "ac_zworld_rnd":
            base.update({"teacher": "unsupervised_rnd_intrinsic",
                         "entropy_beta": EXPLORE_ENTROPY_BETA, "intrinsic_coef": INTRINSIC_COEF,
                         "rnd_feat_dim": RND_FEAT_DIM, "rnd_hidden": RND_HIDDEN})
        else:  # ac_zworld_icm
            base.update({"teacher": "unsupervised_icm_curiosity",
                         "entropy_beta": EXPLORE_ENTROPY_BETA, "intrinsic_coef": INTRINSIC_COEF,
                         "icm_hidden": ICM_HIDDEN, "icm_forward_beta": ICM_FORWARD_BETA})
    else:
        base.update({"kind": "anchor"})
    return base


def _run_zworld_cell(arm_id: str, env_kwargs: Dict[str, Any], seed: int, p0: int, rl_eps: int,
                     eval_eps: int, steps: int) -> Dict[str, Any]:
    warm_env = x734._make_env(seed, env_kwargs)
    agent = fan.make_zworld_agent(warm_env)
    fan.warmup_zworld(agent, warm_env, seed=seed, p0=p0, steps=steps)

    train_env = x734._make_env(seed, env_kwargs)
    extra: Dict[str, Any] = {}
    if arm_id == BASELINE_ARM:
        guard = fan.train_zworld_ac_shaped(
            agent, train_env, seed=seed, n_episodes=rl_eps, steps=steps,
            arm_label=arm_id, denom=rl_eps, shaping_coef=0.0,
        )
        extra["mean_train_forage_recent"] = guard["mean_train_forage_recent"]
    else:
        world_dim = int(agent.config.latent.world_dim)
        action_dim = int(train_env.action_dim)
        if arm_id == "ac_zworld_rnd":
            intrinsic: Any = RNDModule(world_dim)
        else:
            intrinsic = ICMModule(world_dim, action_dim)
        guard = _train_zworld_ac_explore(
            agent, train_env, seed=seed, n_episodes=rl_eps, steps=steps,
            arm_label=arm_id, denom=rl_eps, intrinsic=intrinsic,
            entropy_beta=EXPLORE_ENTROPY_BETA,
        )
        extra["mean_train_forage_recent"] = guard["mean_train_forage_recent"]
        extra["mean_intrinsic_reward_recent"] = guard["mean_intrinsic_reward_recent"]

    eval_env = x734._make_env(seed, env_kwargs)
    row = evaluate_seed(x742.ActorCriticEvalPolicy(agent, arm_id), eval_env, eval_eps, steps)
    row.update(extra)
    return row


def run_experiment(seeds: List[int], p0: int, rl_eps: int, eval_eps: int,
                   steps: int) -> Dict[str, Any]:
    print(
        f"MECH-457 GOV-FANOUT-1 leg D (H-optim: stronger unsupervised explorer) diagnostic "
        f"({len(ARM_ORDER)} arms x 1 rung [{fan.RUNG_ID}] x {len(seeds)} seeds; "
        f"P0={p0}, RL={rl_eps}, eval={eval_eps}, steps={steps})",
        flush=True,
    )
    env_kwargs = x734._env_kwargs_for_rung(fan.RUNG)
    per_arm_forage: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
    per_arm_trainforage: Dict[str, List[float]] = {a: [] for a in ZWORLD_ARMS}
    per_arm_intrinsic: Dict[str, List[float]] = {a: [] for a in EXPLORER_ARMS}
    all_cells: List[Dict[str, Any]] = []

    def _run_cell(arm_id: str, seed: int) -> Dict[str, Any]:
        print(f"Seed {seed} Condition {fan.RUNG_ID}:{arm_id}", flush=True)
        slice_cfg = _arm_config_slice(arm_id, env_kwargs, p0, rl_eps, eval_eps, steps)
        # The sparse baseline is the reuse-eligible MINT; the explorer arms are reuse-INELIGIBLE
        # (their intrinsic-module logic is driver-defined, excluded from the substrate hash).
        ineligible = None if arm_id in (BASELINE_ARM,) + fan.ANCHOR_ARMS else [
            "driver_defined_intrinsic_module_not_in_substrate_hash"
        ]
        with arm_cell(seed, config_slice=slice_cfg, script_path=Path(__file__),
                      config_slice_declared=True, include_driver_script_in_hash=False,
                      extra_ineligible_reasons=ineligible) as cell:
            if arm_id in ZWORLD_ARMS:
                row = _run_zworld_cell(arm_id, env_kwargs, seed, p0, rl_eps, eval_eps, steps)
            else:
                anchor_env = x734._make_env(seed, env_kwargs)
                row = fan.run_anchor_cell(arm_id, anchor_env, seed, eval_eps, steps)
            row["rung_id"] = fan.RUNG_ID
            row["arm_id"] = arm_id
            row["seed"] = int(seed)
            cell.stamp(row)
        forage = float(row["foraging_competence"])
        per_arm_forage[arm_id].append(forage)
        if arm_id in ZWORLD_ARMS:
            per_arm_trainforage[arm_id].append(float(row.get("mean_train_forage_recent", 0.0)))
        if arm_id in EXPLORER_ARMS:
            per_arm_intrinsic[arm_id].append(float(row.get("mean_intrinsic_reward_recent", 0.0)))
        all_cells.append(row)
        print(
            f"verdict: {'PASS' if row['competence_supra_floor'] else 'FAIL'} "
            f"(arm={arm_id} seed={seed} forage/ep={forage})", flush=True,
        )
        return row

    # Anchors first (readiness gate + denominators), then the z_world arms if ready.
    for arm_id in fan.ANCHOR_ARMS:
        for seed in seeds:
            _run_cell(arm_id, seed)

    def _mean(arm: str) -> float:
        vals = per_arm_forage[arm]
        return float(sum(vals) / len(vals)) if vals else 0.0

    local_view_mean = _mean("local_view_greedy")
    oracle_mean = _mean("greedy_oracle")
    readiness_met = bool(
        local_view_mean >= COMPETENCE_RESOURCE_FLOOR and oracle_mean >= COMPETENCE_RESOURCE_FLOOR
    )

    if readiness_met:
        for arm_id in ZWORLD_ARMS:
            for seed in seeds:
                _run_cell(arm_id, seed)
    else:
        print(
            f"readiness UNMET (local_view={local_view_mean} oracle={oracle_mean}); "
            f"skipping z_world training -> substrate_not_ready_requeue", flush=True,
        )

    sparse = fan.summarize(per_arm_forage[BASELINE_ARM])
    rnd = fan.summarize(per_arm_forage["ac_zworld_rnd"])
    icm = fan.summarize(per_arm_forage["ac_zworld_icm"])
    rnd_maj = bool(rnd["majority_supra_floor"])
    icm_maj = bool(icm["majority_supra_floor"])
    any_explorer_maj = bool(rnd_maj or icm_maj)
    sparse_mean = float(sparse["foraging_competence_mean"])
    sparse_subfloor = bool(sparse_mean < COMPETENCE_RESOURCE_FLOOR)

    if not readiness_met:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
    elif any_explorer_maj:
        outcome, label = "PASS", "stronger_unsupervised_explorer_clears_floor_exploration_was_the_wall"
    else:
        outcome, label = "FAIL", "unsupervised_exploration_insufficient_expert_teaching_signal_necessary"

    degeneracy = check_degeneracy({
        "d3_zworld_explorer_vs_baseline_vs_anchor_foraging": {
            "values": [rnd["foraging_competence_mean"], icm["foraging_competence_mean"],
                       sparse_mean, local_view_mean, _mean("random_walk")]
        }
    })

    def _tf(arm: str) -> float:
        vals = per_arm_trainforage.get(arm, [])
        return round(float(sum(vals) / len(vals)), 6) if vals else 0.0

    def _ir(arm: str) -> float:
        vals = per_arm_intrinsic.get(arm, [])
        return round(float(sum(vals) / len(vals)), 6) if vals else 0.0

    interpretation = {
        "label": label,
        "preconditions": [
            fan.readiness_precondition(local_view_mean),
            {"name": "greedy_oracle_clears_floor_at_d3", "kind": "readiness",
             "description": "Env is floor-achievable with global info (achievability anchor).",
             "control": "greedy_oracle foraging_competence @D3 vs the 1.0 floor",
             "measured": round(oracle_mean, 6), "threshold": float(COMPETENCE_RESOURCE_FLOOR),
             "met": bool(oracle_mean >= COMPETENCE_RESOURCE_FLOOR)},
        ],
        "criteria": [
            {"name": "C_any_unsupervised_explorer_clears_floor_at_D3", "load_bearing": True,
             "passed": bool(any_explorer_maj)},
        ],
        "criteria_non_degenerate": {
            "local_view_clears_floor_at_d3": bool(local_view_mean >= COMPETENCE_RESOURCE_FLOOR),
            "oracle_clears_floor_at_d3": bool(oracle_mean >= COMPETENCE_RESOURCE_FLOOR),
            "explorer_vs_baseline_vs_anchor_foraging_spread": bool(degeneracy["non_degenerate"]),
        },
    }

    result: Dict[str, Any] = {
        "outcome": outcome,
        "interpretation": interpretation,
        "interpretation_label": label,
        "discrimination_verdict": label,
        "evidence_direction": "unknown",
        "evidence_direction_per_claim": {"MECH-457": "unknown"},
        "readiness": {
            "readiness_met": readiness_met,
            "local_view_greedy_d3": round(local_view_mean, 6),
            "greedy_oracle_d3": round(oracle_mean, 6),
        },
        "headline": {
            "d3_zworld_rnd_forage": rnd["foraging_competence_mean"],
            "d3_zworld_rnd_per_seed": rnd["foraging_competence_per_seed"],
            "d3_zworld_icm_forage": icm["foraging_competence_mean"],
            "d3_zworld_icm_per_seed": icm["foraging_competence_per_seed"],
            "d3_zworld_sparse_baseline_forage": sparse["foraging_competence_mean"],
            "d3_zworld_sparse_baseline_per_seed": sparse["foraging_competence_per_seed"],
            "d3_any_explorer_majority_supra_floor": any_explorer_maj,
            "d3_rnd_majority_supra_floor": rnd_maj,
            "d3_icm_majority_supra_floor": icm_maj,
            "d3_rnd_mean_intrinsic_reward_recent": _ir("ac_zworld_rnd"),
            "d3_icm_mean_intrinsic_reward_recent": _ir("ac_zworld_icm"),
            "d3_local_view_greedy_denominator": round(local_view_mean, 6),
            "d3_greedy_oracle": round(oracle_mean, 6),
            "d3_random_walk": round(_mean("random_walk"), 6),
            "reference_748_zworld_bc_forage_d3": 32.72,
        },
        "baseline_guard": {
            "load_bearing": False,
            "d3_sparse_baseline_forage": sparse_mean,
            "sparse_baseline_subfloor": sparse_subfloor,
            "sparse_baseline_reproduces_742_band": bool(
                REFERENCE_742_SPARSE_LOW <= sparse_mean <= REFERENCE_742_SPARSE_HIGH
            ),
            "reference_742_sparse_band": [REFERENCE_742_SPARSE_LOW, REFERENCE_742_SPARSE_HIGH],
            "note": (
                "The sparse z_world baseline is the SAME-RUN reference (742 cotrain sparse cell). "
                "Expected sub-floor ~0.20-0.27. If it unexpectedly clears the floor, the "
                "discrimination is confounded by substrate drift (flag; do NOT self-route)."
            ),
        },
        "per_arm": {a: fan.summarize(per_arm_forage[a]) for a in ARM_ORDER},
        "per_arm_train_forage_recent": {a: _tf(a) for a in ZWORLD_ARMS},
        "denominators": {
            "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
            "local_view_greedy_d3_live": round(local_view_mean, 6),
            "local_view_greedy_d3_738_reference": float(fan.DENOM_738_D3_REFERENCE),
        },
        "arm_results": all_cells,
        "non_degenerate": bool(degeneracy["non_degenerate"]),
        "degeneracy_reason": degeneracy["degeneracy_reason"],
        "degenerate_metrics": degeneracy["degenerate_metrics"],
    }
    return result


def _build_manifest(result: Dict[str, Any], timestamp_utc: str, dry_run: bool,
                    cfg: Dict[str, Any]) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    return {
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
        "discrimination_verdict": result["discrimination_verdict"],
        "readiness": result["readiness"],
        "headline": result["headline"],
        "baseline_guard": result["baseline_guard"],
        "denominators": result["denominators"],
        "per_arm": result["per_arm"],
        "per_arm_train_forage_recent": result["per_arm_train_forage_recent"],
        "arm_results": result["arm_results"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "degenerate_metrics": result["degenerate_metrics"],
        "portfolio": {
            "gov_fanout_1": "MECH-457 competence-discrimination portfolio (742 + 747/748/749 autopsies)",
            "leg": "D (H-optim, algorithm / exploration-strength axis)",
            "factorial_cell": "(z_world_cotrain, unsupervised explorer: RND + ICM, no expert)",
            "siblings": ["V3-EXQ-747 (H-rep)", "V3-EXQ-748 (H-explore/reward-density)",
                         "V3-EXQ-749 (conjunction)"],
            "reference_cells": {
                "742_zworld_sparse": "FAIL (0.20-0.27; re-run here as the ac_zworld_sparse baseline)",
                "748_zworld_bc": "PASS (32.72; the expert-teacher cell this leg tests against)",
            },
        },
        "reuse_mint": {
            "reusable_arm": BASELINE_ARM,
            "reuse_eligible": True,
            "note": (
                "ac_zworld_sparse is emitted reuse-eligible (mint-as-you-go): rng_fully_reset "
                "via arm_cell + config_slice_declared + include_driver_script_in_hash=False; its "
                "OFF computation lives in fan.train_zworld_ac_shaped (shaping_coef=0.0) under "
                "experiments/_lib/** (in the substrate hash). A later MECH-457 z_world-sparse "
                "consumer can cite reuse_baseline_from this run_id. The explorer arms are "
                "reuse-INELIGIBLE (driver-defined intrinsic modules, excluded from the hash)."
            ),
        },
        "config": cfg,
        "load_bearing_dv": (
            "D3 z_world (cotrain) actor-critic foraging_competence under a stronger UNSUPERVISED "
            "explorer (RND novelty OR ICM curiosity + raised entropy, NO expert), unshaped eval, "
            "vs the 1.0 floor, strict majority of seeds; readiness = local_view_greedy + oracle "
            "clear the floor @D3."
        ),
        "notes": (
            "GOV-FANOUT-1 leg D (algorithm axis). DIAGNOSTIC (excluded from scoring); "
            "PROMOTES/DEMOTES NOTHING; route to /failure-autopsy before any governance action. "
            "Decides the open algorithm-axis discrimination the 747/748/749 portfolio left "
            "(autopsy sec7.4/sec8): is an expert teacher NECESSARY or was unsupervised "
            "exploration merely too weak. NULL (explorer clears floor) -> build a better "
            "unsupervised explorer; FAIL (both explorers sub-floor, BC passed) -> build "
            "imitation+closed-loop-correction (NOT static BC; DAgger O(eps*T^2) ceiling, DQfD "
            "escape). NOT a power-bump of the sparse-RL design -- adds a LEARNED intrinsic signal "
            "(RND/ICM) the 742/748 baseline lacks (it already had entropy 0.03 + count-novelty). "
            "z_world only (representation axis closed by 747/748/749; z_world+BC > raw+BC). "
            "MECH-457 stays candidate/v3_pending; competence_implementation_gap. Do NOT amend the "
            "f_dominance_conversion_ceiling substrate_queue entry (autopsy sec8 action=none). "
            "GOV-REUSE-1: the decisive readout (z_world actor-critic foraging under an "
            "RND/ICM unsupervised explorer) is recorded in NO prior manifest (742 sparse-only, "
            "748 dense-teacher, 747/749 raw-view) -> run. Re-derive brake: does NOT fire (0 "
            "substrate_ceiling/non_contributory autopsies on MECH-457). ac_zworld_sparse minted "
            "reuse-eligible (mint-as-you-go)."
        ),
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-751 MECH-457 H-optim stronger-unsupervised-explorer diagnostic"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)

    if args.dry_run:
        seeds = list(fan.DRY_SEEDS)
        p0, rl_eps = fan.DRY_P0, fan.DRY_RL
        eval_eps, steps = fan.DRY_EVAL, fan.DRY_STEPS
    else:
        seeds = list(fan.SEEDS)
        p0, rl_eps = fan.P0_WARMUP_EPISODES, fan.RL_EPISODES
        eval_eps, steps = fan.EVAL_EPISODES, fan.STEPS_PER_EPISODE

    result = run_experiment(seeds=seeds, p0=p0, rl_eps=rl_eps, eval_eps=eval_eps, steps=steps)

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cfg = {
        "seeds": seeds, "rung": fan.RUNG_ID, "arms": list(ARM_ORDER),
        "p0_warmup_episodes": p0, "rl_episodes": rl_eps,
        "eval_episodes": eval_eps, "steps_per_episode": steps,
        "actor_critic_hidden": fan.ACTOR_CRITIC_HIDDEN, "cotrain_encoder": True,
        "ac_lr": fan.AC_LR, "ac_gamma": fan.AC_GAMMA,
        "baseline_entropy_beta": fan.AC_ENTROPY_BETA, "explore_entropy_beta": EXPLORE_ENTROPY_BETA,
        "intrinsic_coef": INTRINSIC_COEF, "rnd_feat_dim": RND_FEAT_DIM, "rnd_hidden": RND_HIDDEN,
        "rnd_lr": RND_LR, "icm_hidden": ICM_HIDDEN, "icm_lr": ICM_LR,
        "icm_forward_beta": ICM_FORWARD_BETA,
        "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
    }
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run), cfg=cfg)

    out_dir = Path(args.out_dir) if args.out_dir is not None else (
        REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    )
    out_path = write_flat_manifest(
        manifest, out_dir, dry_run=args.dry_run, config=cfg, seeds=seeds,
        script_path=Path(__file__),
        elapsed_seconds=(datetime.now(timezone.utc) - started).total_seconds(),
    )

    print(f"manifest: {out_path}", flush=True)
    hl = result["headline"]
    print(
        f"outcome: {result['outcome']} label={result['interpretation_label']} "
        f"readiness_met={result['readiness']['readiness_met']} "
        f"non_degenerate={result['non_degenerate']}", flush=True,
    )
    print(
        f"  D3: rnd={hl['d3_zworld_rnd_forage']} icm={hl['d3_zworld_icm_forage']} "
        f"sparse={hl['d3_zworld_sparse_baseline_forage']} "
        f"local_view={hl['d3_local_view_greedy_denominator']} "
        f"(any_explorer_supra={hl['d3_any_explorer_majority_supra_floor']})", flush=True,
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
