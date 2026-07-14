"""Shared mechanism library for the MECH-457 GOV-FANOUT-1 combination-aware discrimination
portfolio (V3-EXQ-752 H-credit / 753 H-return / 754 H-curriculum / 755 H-mode / 756 the
H-credit x H-return PAIR).

WHY THIS EXISTS. V3-EXQ-751 (combined autopsy failure_autopsy_MECH-457-fanout-751-750_
2026-07-14) settled the ALGORITHM axis: a learned unsupervised novelty signal (RND) clears the
1.0 forage floor on the z_world(cotrain) actor-critic with NO expert (5.22, 3/3 seeds); ICM
does not; an expert teacher is NOT necessary. But RND plateaus far below the BC expert (32.72)
and the local_view_greedy observability ceiling (48.05). The class-choice /lit-pull
(targeted_review_action_learning_bootstrap_class_choice/SYNTHESIS.md) REJECTED building another
novelty mechanism (coverage not competence; noisy-TV / detachment; duplicates the landed
ARC-065 / MECH-314 novelty substrate) and named FOUR composable candidate classes on distinct
design axes that plausibly close the floor->competent gap. This portfolio discriminates which
class (or additive combination) reaches matched-competent UNSUPERVISED foraging.

WHY THE MECHANISMS LIVE IN _lib (not the driver). 751's RND/ICM arms were driver-defined and
therefore reuse-INELIGIBLE (their logic sits outside the substrate hash). Housing every
mechanism HERE (experiments/_lib/**) folds it into the arm_fingerprint substrate_hash via the
_lib glob, so each (mechanism x representation x seed) cell can be emitted reuse-ELIGIBLE
(mint-as-you-go). A future MECH-457 consumer -- including the INV-088 V3-EXQ-750 retest, which
needs matched-competent unsupervised policies on BOTH z_world and the raw 5x5 view -- can then
cite reuse_baseline_from and skip re-training a byte-identical cell (user directive 2026-07-14:
separate arms per representation to maximise reuse).

THE FIVE MECHANISM CLASSES (each ALONE on a DIFFERENT design axis; NEVER a power-bump of the
braked sparse-RL / novelty design):
  * H-credit  (credit-assignment axis)  -- prioritized backward credit replay: the SAME
        exploration as the 742 sparse baseline (entropy 0.03 + count-novelty, NO intrinsic
        bonus, NO shaping), but after each reward-bearing episode, K extra backward passes over
        the trajectory transitions PRIORITIZED by |TD-error| (Mattar & Daw 2018), with credit
        backward-discounted from the reward endpoint (Foster & Wilson 2006 reverse replay). The
        RL-native homolog of the substrate's hippocampal backward_credit_sweep (MECH-290). Only
        the UPDATE changes; the rollout/exploration is byte-identical to sparse.
  * H-return  (memory / frontier axis)  -- Go-Explore (Ecoffet 2021): an archive keyed by a
        discretized position cell stores a RESTORABLE env snapshot (agent_pos + resources; D3 is
        hazard-free) with a score; at episode start, with return_prob, env.reset_to() a selected
        frontier cell and explore on-policy from there. Directly targets the detachment that
        makes a from-spawn learner (and RND) abandon the frontier every episode.
  * H-curriculum (teaching-signal axis) -- IMGEP/AMIGo-lite (Forestier 2022 / Campero 2021): a
        learning-progress curriculum over goal DISTANCE trains a goal-conditioned actor-critic to
        reach proposed cells, NO expert. Forage eval selects the goal UNSUPERVISED from the
        agent's own 5x5 resource_field_view (nearest visible resource) and navigates via the
        goal-conditioned policy. The unsupervised analogue of the BC arm.
  * H-mode (policy-control axis)        -- a critic-utility-gated explore/exploit mode scalar
        (Aston-Jones & Cohen 2005 / Daw 2006) that anneals the entropy temperature and the RND
        intrinsic coefficient toward greedy exploitation as utility rises. A meta-controller OVER
        exploration (consolidation-into-exploitation), tested against a fixed-coefficient RND arm
        SAME-RUN -- the treatment is the GATE, not more novelty.
  * PAIR H-credit x H-return            -- the two highest-composition classes together, run
        alongside each alone SAME-RUN so additivity/complementarity is directly readable.

REFERENCE BAND (cited constants, same substrate, days apart -- the 751 precedent cited BC 32.72
without re-running it): floor 1.0 / RND novelty-plateau ~5.22 (751) / BC expert 32.72 (748) /
local_view_greedy observability ceiling 48.05 (738/742). Each leg re-runs the SPARSE baseline
SAME-RUN on both reps (the OFF/floor reference + a reuse mint + the substrate-drift guard);
anchors (local_view_greedy / greedy_oracle / random_walk) are evaluated live for readiness +
denominators.

DIAGNOSTIC, not evidence. experiment_purpose="diagnostic" for every leg -> excluded from
governance confidence/conflict scoring; each routes to /failure-autopsy for adjudication before
any governance action. MECH-457 stays candidate / v3_pending; INV-088 stays candidate /
pending_substrate_reconfirmation (754/756 also tag INV-088 relevance only). Re-derive brake does
NOT fire (0 prior substrate_ceiling/non_contributory autopsies on MECH-457; each leg tests a
DIFFERENT mechanism on a DIFFERENT axis = new EXQ, not a lettered power-bump).

READINESS + P0 readiness-assert (the /queue-experiment adjudication gate). The load-bearing
criterion reads a LEARNED quantity (the trained policy's foraging_competence @D3 vs the 1.0
floor), so each leg measures the SAME statistic on a KNOWN-POSITIVE control: LocalViewGreedyPolicy
(reading the same 5x5 view) + greedy_oracle clear the floor @D3. Below readiness -> self-route
substrate_not_ready_requeue (NEVER a substrate-verdict label).

Sourced APIs (verified 2026-07-14):
  experiments/_lib/mech457_fanout.py (fan) -- make_zworld_agent / warmup_zworld /
    train_zworld_ac_shaped / make_rawview_ac / RawViewACEvalPolicy / run_anchor_cell /
    summarize / readiness_precondition / _rawview_tensor / ANCHOR_ARMS / budgets.
  experiments/v3_exq_742_mech457_actor_critic_onoff.py (x742) -- _sense / ActorCriticEvalPolicy.
  experiments/v3_exq_734_env_difficulty_competence_recovery_sweep.py (x734) -- _make_env /
    _env_kwargs_for_rung / _compute_gae / _RunningStd / _novelty_bonus / FORAGE_BONUS /
    REWARD_STD_EPS / _strict_majority / RUNG (D3_hazard_free, num_hazards=0).
  ree_core/action_learning/actor_critic.py -- ActorCriticPolicy(world_dim, action_dim, ...).select.
  ree_core/environment/causal_grid_world.py -- reset_to(agent_pos, hazards, resources).
  experiments/_lib/capability_eval.py -- evaluate_seed / Policy / LocalViewGreedyPolicy /
    OraclePolicy / RandomPolicy / COMPETENCE_RESOURCE_FLOOR / nearest_resource_manhattan.
ASCII-only in all runtime strings (repo rule).

SLEEP DRIVER: none (no sleep loop; use_sleep_loop / sws_enabled / rem_enabled all OFF).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ree_core.action_learning.actor_critic import ActorCriticPolicy
from experiments._lib.capability_eval import (
    COMPETENCE_RESOURCE_FLOOR,
    Policy,
)
import experiments._lib.mech457_fanout as fan
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734
import experiments.v3_exq_742_mech457_actor_critic_onoff as x742

DEVICE = fan.DEVICE

# ---------------------------------------------------------------------------
# Reference band (cited constants -- comparability boundary; NOT re-run per leg).
# ---------------------------------------------------------------------------
RND_PLATEAU_5_22 = 5.22          # 751 z_world RND novelty plateau (the class the lit-pull rejected)
BC_REFERENCE_32_72 = 32.72       # 748 z_world+BC expert teacher
CEILING_48_05 = 48.05            # 738/742 local_view_greedy observability ceiling
ORACLE_57_2 = 57.2               # 742 greedy_oracle (global-info achievability)

# Pre-registered load-bearing thresholds (declared here, not derived from a run's own stats).
LIFT_MARGIN_FRAC = 0.5           # a mechanism "lifts above the novelty plateau" only if its
#                                  mean forage exceeds RND_PLATEAU by >=50% (~7.83) -- a real
#                                  step toward BC 32.72, not "6 > 5" noise.
LIFT_ABOVE_PLATEAU = RND_PLATEAU_5_22 * (1.0 + LIFT_MARGIN_FRAC)   # ~7.83
CONSOLIDATION_ABS_MARGIN = 1.0   # H-mode: mode arm must beat fixed-RND by >=1.0 resource/ep
ADDITIVITY_ABS_MARGIN = 1.0      # PAIR: pair must beat the better single by >=1.0 resource/ep

# 742 sparse cotrain reference band (drift guard for the same-run OFF baseline).
REFERENCE_742_SPARSE_LOW = 0.20
REFERENCE_742_SPARSE_HIGH = 0.27

# Explorer hyperparameters (shared with 751's H-optim leg for the RND-based arms).
EXPLORE_ENTROPY_BETA = 0.10
INTRINSIC_COEF = 1.0
RND_FEAT_DIM = 64
RND_HIDDEN = 128
RND_LR = 1e-3

# H-credit prioritized-replay hyperparameters.
CREDIT_REPLAY_PASSES = 3         # extra backward sweeps per reward-bearing episode
CREDIT_TOPK = 32                 # transitions per pass, taken in |TD-error| priority order
CREDIT_GAMMA = 0.9               # reverse-replay backward discount (Foster & Wilson endpoint)
CREDIT_LR_SCALE = 1.0            # the replay update uses the same AC_LR

# H-return Go-Explore hyperparameters.
RETURN_PROB = 0.5                # probability an episode starts by returning to a frontier cell
ARCHIVE_CELL_SIZE = 2            # discretization of the position archive key (coarse cells)
ARCHIVE_MAX = 512                # archive capacity (frontier of promising restorable states)

# H-curriculum (IMGEP/AMIGo-lite) hyperparameters.
GOAL_DIM = 2                     # relative (dx, dy) to the target cell, normalized by grid size
CURRICULUM_D0 = 1                # initial goal Manhattan distance
CURRICULUM_D_MAX = 12            # cap on curriculum goal distance
CURRICULUM_SUCCESS_HI = 0.7      # raise difficulty when reach-rate exceeds this
CURRICULUM_SUCCESS_LO = 0.3      # lower difficulty when reach-rate falls below this
CURRICULUM_WINDOW = 20           # reach-rate window driving the learning-progress curriculum
GOAL_REACH_REWARD = 1.0
GOAL_STEP_COST = 0.01

# H-mode utility-gate hyperparameters.
MODE_UTILITY_K = 2.0             # sigmoid slope on (utility - baseline)
MODE_ENTROPY_HI = EXPLORE_ENTROPY_BETA   # explore end of the mode anneal
MODE_ENTROPY_LO = 0.01                    # exploit end of the mode anneal

AC_LR = fan.AC_LR
AC_GAMMA = fan.AC_GAMMA
AC_ENTROPY_BETA = fan.AC_ENTROPY_BETA     # 0.03 -- the sparse baseline exploration level
AC_VALUE_COEF = fan.AC_VALUE_COEF
AC_GRAD_CLIP = fan.AC_GRAD_CLIP
ACTOR_CRITIC_HIDDEN = fan.ACTOR_CRITIC_HIDDEN
RAW_VIEW_DIM = fan.RAW_VIEW_DIM
TRAIN_FORAGE_WINDOW = fan.TRAIN_FORAGE_WINDOW

REPRESENTATIONS: Tuple[str, ...] = ("z_world", "raw_view")


# ===========================================================================
# Representation adapters -- unify the z_world (cotrain, via agent.actor_critic_step)
# and raw 5x5 view (standalone ActorCriticPolicy) paths behind one interface so the
# mechanism trainers below are representation-agnostic.
# ===========================================================================
class RepAgent:
    """Common interface. encode(obs)->state (grad-carrying); step(state)->ActorCriticStep;
    z_detached(state)->[1,dim] detached feature for the intrinsic modules; params()->optimizer
    param list; reset_episode(); eval_policy(label)->Policy; feature_dim; action_dim."""

    representation: str = "?"
    feature_dim: int = 0
    action_dim: int = 5

    def reset_episode(self) -> None:
        return None

    def encode(self, obs_dict: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def step(self, state: Any, deterministic: bool = False):
        raise NotImplementedError

    def z_detached(self, state: Any) -> torch.Tensor:
        raise NotImplementedError

    def params(self) -> List[torch.nn.Parameter]:
        raise NotImplementedError

    def eval_policy(self, label: str) -> Policy:
        raise NotImplementedError


class ZWorldRep(RepAgent):
    """z_world cotrain: an all-ON 742 agent; the actor reads LIVE z_world so the policy gradient
    reaches latent_stack (cotrain). state = the encoded LatentState."""

    representation = "z_world"

    def __init__(self, env: Any, seed: int, p0: int, steps: int) -> None:
        self.agent = fan.make_zworld_agent(env)
        fan.warmup_zworld(self.agent, env, seed=seed, p0=p0, steps=steps)
        self.feature_dim = int(self.agent.config.latent.world_dim)
        self.action_dim = int(env.action_dim)

    def reset_episode(self) -> None:
        self.agent.reset()

    def encode(self, obs_dict: Dict[str, Any]) -> Any:
        return x742._sense(self.agent, obs_dict)

    def step(self, state: Any, deterministic: bool = False):
        return self.agent.actor_critic_step(state, deterministic=deterministic)

    def z_detached(self, state: Any) -> torch.Tensor:
        return state.z_world.detach()

    def params(self) -> List[torch.nn.Parameter]:
        return list(self.agent.actor_critic_parameters()) + list(
            self.agent.actor_critic_encoder_parameters()
        )

    def eval_policy(self, label: str) -> Policy:
        return x742.ActorCriticEvalPolicy(self.agent, label)


class RawViewRep(RepAgent):
    """Raw 5x5 resource_field_view: a standalone ActorCriticPolicy(world_dim=25). No REE encoder,
    no P0 warmup. state = the flattened view tensor."""

    representation = "raw_view"

    def __init__(self, env: Any) -> None:
        self.ac = fan.make_rawview_ac()
        self.feature_dim = RAW_VIEW_DIM
        self.action_dim = int(env.action_dim)

    def encode(self, obs_dict: Dict[str, Any]) -> Any:
        return fan._rawview_tensor(obs_dict)

    def step(self, state: Any, deterministic: bool = False):
        return self.ac.select(state, deterministic=deterministic)

    def z_detached(self, state: Any) -> torch.Tensor:
        return state.detach()

    def params(self) -> List[torch.nn.Parameter]:
        return list(self.ac.parameters())

    def eval_policy(self, label: str) -> Policy:
        return fan.RawViewACEvalPolicy(self.ac, label)


def make_rep(representation: str, env: Any, seed: int, p0: int, steps: int) -> RepAgent:
    if representation == "z_world":
        return ZWorldRep(env, seed, p0, steps)
    if representation == "raw_view":
        return RawViewRep(env)
    raise ValueError(f"unknown representation {representation!r}")


# ===========================================================================
# Intrinsic module (RND) -- reused for H-mode and (as the cited plateau) the comparison.
# Trains on detached features; the intrinsic reward is a scalar bonus (does not perturb the
# policy gradient), exactly like the count-novelty already in the sparse baseline.
# ===========================================================================
class RNDModule:
    """Random Network Distillation (Burda et al. 2018). Novelty = predictor error against a
    FROZEN random projection of the (detached) feature; normalized by a running std."""

    def __init__(self, feature_dim: int) -> None:
        self.target = nn.Sequential(
            nn.Linear(feature_dim, RND_HIDDEN), nn.ReLU(),
            nn.Linear(RND_HIDDEN, RND_FEAT_DIM),
        ).to(DEVICE)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, RND_HIDDEN), nn.ReLU(),
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
            nn.utils.clip_grad_norm_(self.predictor.parameters(), AC_GRAD_CLIP)
            self.opt.step()
        self._buf = []


# ===========================================================================
# H-mode: critic-utility-gated explore/exploit mode scalar.
# ===========================================================================
class ModeGate:
    """m_t = sigmoid(k * (utility - baseline)) in [0,1]. utility = an EMA of the episode return
    (critic-grounded exploit signal). As utility rises the gate anneals the entropy temperature
    and the RND intrinsic coefficient TOWARD greedy exploitation (consolidation)."""

    def __init__(self) -> None:
        self.utility_ema: Optional[float] = None
        self.baseline_ema: Optional[float] = None
        self._m = 0.0

    def mode(self) -> float:
        if self.utility_ema is None or self.baseline_ema is None:
            return 0.0
        x = float(np.clip(MODE_UTILITY_K * (self.utility_ema - self.baseline_ema), -30.0, 30.0))
        return float(1.0 / (1.0 + np.exp(-x)))

    def entropy_beta(self) -> float:
        m = self.mode()
        return MODE_ENTROPY_HI * (1.0 - m) + MODE_ENTROPY_LO * m

    def intrinsic_coef(self) -> float:
        return INTRINSIC_COEF * (1.0 - self.mode())

    def update(self, episode_return: float) -> None:
        r = float(episode_return)
        self.utility_ema = r if self.utility_ema is None else 0.2 * r + 0.8 * self.utility_ema
        # A slower baseline: the gate fires when recent utility outruns the long-run mean.
        self.baseline_ema = r if self.baseline_ema is None else 0.02 * r + 0.98 * self.baseline_ema


# ===========================================================================
# H-return: Go-Explore archive of restorable frontier states.
# ===========================================================================
class GoExploreArchive:
    """Archive keyed by a coarse discretized position cell -> the best RESTORABLE env snapshot
    (agent_pos + resource_positions; D3 is hazard-free so hazards are empty) reaching that cell,
    with a score (resources collected en route). At episode start the trainer may reset_to a
    selected frontier cell (prefer under-visited, higher-score) and explore on-policy from there.
    Restores use env.reset_to (a resettable simulator) -- NOT policy steps, so the return phase
    is a teleport that re-seats on-policy exploration at the frontier."""

    def __init__(self, seed: int) -> None:
        self._rng = np.random.RandomState(int(seed) + 90001)
        self.cells: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.visits: Dict[Tuple[int, int], int] = {}

    @staticmethod
    def _key(x: int, y: int) -> Tuple[int, int]:
        return (int(x) // ARCHIVE_CELL_SIZE, int(y) // ARCHIVE_CELL_SIZE)

    def observe(self, env: Any, score: float) -> None:
        """Snapshot the current restorable state if this cell is new or the score improved."""
        key = self._key(env.agent_x, env.agent_y)
        cur = self.cells.get(key)
        if cur is not None and score <= cur["score"]:
            return
        if cur is None and len(self.cells) >= ARCHIVE_MAX:
            return
        self.cells[key] = {
            "score": float(score),
            "agent_pos": (int(env.agent_x), int(env.agent_y)),
            "resources": [(int(r[0]), int(r[1])) for r in getattr(env, "resources", [])],
        }

    def ready(self) -> bool:
        return len(self.cells) > 0

    def select(self) -> Dict[str, Any]:
        """Frontier selection: weight by 1/(1+visits) so under-explored cells are preferred
        (the derailment fix -- keep returning to the frontier rather than re-consuming a
        well-trodden cell)."""
        keys = list(self.cells.keys())
        weights = np.array([1.0 / (1.0 + self.visits.get(k, 0)) for k in keys], dtype=np.float64)
        weights = weights / weights.sum()
        idx = int(self._rng.choice(len(keys), p=weights))
        key = keys[idx]
        self.visits[key] = self.visits.get(key, 0) + 1
        return self.cells[key]


# ===========================================================================
# Generic A2C trainer with mechanism hooks (sparse / RND / H-credit / H-return / H-mode /
# the H-credit x H-return PAIR). Single-backward-per-episode, GAE advantages -- byte-comparable
# to fan.train_zworld_ac_shaped. Hooks:
#   intrinsic     : RNDModule or None    -- adds a learned novelty bonus (RND / H-mode)
#   entropy_beta  : rollout entropy level (0.03 = sparse exploration; 0.10 = raised explorer)
#   mode_gate     : ModeGate or None     -- utility-gated anneal of entropy + intrinsic coef
#   credit_replay : bool                 -- H-credit prioritized backward replay after the update
#   archive       : GoExploreArchive or None + return_prob -- H-return frontier return
# ===========================================================================
def train_a2c(
    rep: RepAgent, env: Any, seed: int, n_episodes: int, steps: int, arm_label: str, denom: int,
    *,
    intrinsic: Optional[RNDModule] = None,
    entropy_beta: float = AC_ENTROPY_BETA,
    intrinsic_coef: float = 0.0,
    mode_gate: Optional[ModeGate] = None,
    credit_replay: bool = False,
    archive: Optional[GoExploreArchive] = None,
    return_prob: float = 0.0,
) -> Dict[str, Any]:
    params = rep.params()
    optimiser = torch.optim.Adam(params, lr=AC_LR)
    reward_std = x734._RunningStd()
    novelty_counter: Dict[Tuple[int, int], int] = {}
    train_forage_recent: deque = deque(maxlen=TRAIN_FORAGE_WINDOW)
    intrinsic_recent: deque = deque(maxlen=TRAIN_FORAGE_WINDOW)
    n_returns = 0
    n_credit_passes = 0

    for ep in range(n_episodes):
        used_return = False
        if (
            archive is not None and return_prob > 0.0 and archive.ready()
            and float(np.random.rand()) < return_prob
        ):
            snap = archive.select()
            _flat, obs_dict = env.reset_to(snap["agent_pos"], [], snap["resources"])
            used_return = True
            n_returns += 1
        else:
            _flat, obs_dict = env.reset()
        rep.reset_episode()

        ep_logp: List[torch.Tensor] = []
        ep_value_t: List[torch.Tensor] = []
        ep_entropy: List[torch.Tensor] = []
        ep_value_f: List[float] = []
        ep_rewards: List[float] = []
        # For H-credit prioritized replay: keep the detached obs + action + reward per step.
        replay_obs: List[Dict[str, Any]] = []
        replay_actions: List[int] = []
        terminal = False
        bootstrap_value = 0.0
        ep_resources = 0
        ep_intrinsic = 0.0
        cum_resources = 0

        beta_eff = mode_gate.entropy_beta() if mode_gate is not None else entropy_beta
        coef_eff = mode_gate.intrinsic_coef() if mode_gate is not None else intrinsic_coef

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

            if archive is not None:
                archive.observe(env, score=float(cum_resources))

            if done:
                terminal = True
                break
            state = next_state

        if not terminal:
            with torch.no_grad():
                boot = rep.step(state, deterministic=False)
            bootstrap_value = float(boot.value.reshape(-1)[0].item())

        T = len(ep_logp)
        advs: List[float] = []
        rets: List[float] = []
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
            value_loss = AC_VALUE_COEF * 0.5 * (value_t - ret_t.detach()).pow(2).mean()
            entropy_bonus = entropy_t.mean()
            loss = policy_loss + value_loss - beta_eff * entropy_bonus
            if torch.isfinite(loss):
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, AC_GRAD_CLIP)
                optimiser.step()

        # H-credit: prioritized backward credit replay on reward-bearing episodes only.
        if credit_replay and ep_resources > 0 and T > 0:
            n_credit_passes += _prioritized_credit_replay(
                rep, optimiser, params, replay_obs, replay_actions, rets, ep_value_f,
            )

        if intrinsic is not None:
            intrinsic.update()
        if mode_gate is not None:
            mode_gate.update(float(sum(ep_rewards)))

        train_forage_recent.append(ep_resources)
        intrinsic_recent.append(ep_intrinsic / max(1, T))
        cur = ep + 1
        if cur % 200 == 0 or cur == n_episodes:
            print(
                f"  [train] {arm_label} seed={seed} phase=RL ep {cur}/{denom} "
                f"(returns={n_returns} credit_passes={n_credit_passes})",
                flush=True,
            )

    mtf = float(sum(train_forage_recent) / len(train_forage_recent)) if train_forage_recent else 0.0
    mir = float(sum(intrinsic_recent) / len(intrinsic_recent)) if intrinsic_recent else 0.0
    return {
        "mean_train_forage_recent": round(mtf, 6),
        "mean_intrinsic_reward_recent": round(mir, 6),
        "n_return_episodes": int(n_returns),
        "n_credit_replay_passes": int(n_credit_passes),
    }


def _prioritized_credit_replay(
    rep: RepAgent, optimiser: torch.optim.Optimizer, params: List[torch.nn.Parameter],
    replay_obs: List[Dict[str, Any]], replay_actions: List[int], rets: List[float],
    values_f: List[float],
) -> int:
    """Mattar & Daw prioritized sweeping + Foster & Wilson reverse replay, RL-native.

    Priority = |TD-error| = |return_t - value_t|. Take the top-CREDIT_TOPK transitions in
    priority order; sweep them in REVERSE temporal order applying an extra actor-critic update
    whose per-transition credit is backward-discounted from the reward endpoint. Re-encodes the
    stored observations (grad-carrying) so the policy gradient reaches the actor (and, for
    z_world cotrain, the encoder). Changes ONLY the credit assigned to already-collected reward;
    the rollout/exploration is untouched. Returns the number of replay passes applied."""
    T = min(len(replay_obs), len(rets), len(values_f))
    if T == 0:
        return 0
    td = [abs(float(rets[t]) - float(values_f[t])) for t in range(T)]
    order = sorted(range(T), key=lambda t: td[t], reverse=True)[:CREDIT_TOPK]
    # Reverse temporal order among the selected (endpoint first -> reverse replay).
    order = sorted(order, reverse=True)
    passes = 0
    for _p in range(CREDIT_REPLAY_PASSES):
        logps: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        credits: List[float] = []
        ret_targets: List[float] = []
        for rank, t in enumerate(order):
            state = rep.encode(replay_obs[t])
            step = rep.step(state, deterministic=False)
            logp = _action_logprob(step, int(replay_actions[t]))
            logps.append(logp)
            values.append(step.value.reshape(-1)[0])
            # credit = advantage backward-discounted by rank distance from the endpoint.
            adv = float(rets[t]) - float(values_f[t])
            credits.append(adv * (CREDIT_GAMMA ** rank))
            ret_targets.append(float(rets[t]))
        if not logps:
            break
        logp_t = torch.stack(logps)
        value_t = torch.stack(values)
        credit_t = torch.tensor(credits, dtype=torch.float32, device=DEVICE)
        ret_t = torch.tensor(ret_targets, dtype=torch.float32, device=DEVICE)
        policy_loss = -(logp_t * credit_t.detach()).mean()
        value_loss = AC_VALUE_COEF * 0.5 * (value_t - ret_t.detach()).pow(2).mean()
        loss = CREDIT_LR_SCALE * (policy_loss + value_loss)
        if torch.isfinite(loss):
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, AC_GRAD_CLIP)
            optimiser.step()
            passes += 1
    return passes


def _action_logprob(step: Any, action: int) -> torch.Tensor:
    """log pi(action | state) from an ActorCriticStep's logits (graph-connected)."""
    logits = step.logits.reshape(-1)
    logp_all = torch.log_softmax(logits, dim=-1)
    return logp_all[int(action)]


# ===========================================================================
# H-curriculum: goal-conditioned actor-critic + IMGEP/AMIGo-lite learning-progress curriculum.
# A DRIVER-side ActorCriticPolicy(feature_dim + GOAL_DIM); for z_world the live z_world is
# concatenated with the goal so the policy gradient still reaches the encoder (cotrain).
# ===========================================================================
class GoalCondModel:
    """Goal-conditioned actor-critic over [feature, rel_goal]. `feature_fn(obs)` returns the
    grad-carrying feature (live z_world for cotrain; the raw view otherwise); `enc_params` are
    the upstream encoder params to co-train (empty for raw)."""

    def __init__(self, representation: str, rep: RepAgent) -> None:
        self.representation = representation
        self.rep = rep
        self.ac = ActorCriticPolicy(
            world_dim=rep.feature_dim + GOAL_DIM, action_dim=rep.action_dim,
            hidden_dim=ACTOR_CRITIC_HIDDEN, use_sf_critic=False,
        ).to(DEVICE)

    def _feature(self, obs_dict: Dict[str, Any]) -> torch.Tensor:
        if self.representation == "z_world":
            return self.rep.encode(obs_dict).z_world      # live (grad)
        return fan._rawview_tensor(obs_dict)

    def encode(self, obs_dict: Dict[str, Any], goal: Tuple[float, float]) -> torch.Tensor:
        feat = self._feature(obs_dict)
        g = torch.tensor([[float(goal[0]), float(goal[1])]], dtype=torch.float32, device=DEVICE)
        return torch.cat([feat, g], dim=-1)

    def step(self, inp: torch.Tensor, deterministic: bool = False):
        return self.ac.select(inp, deterministic=deterministic)

    def params(self) -> List[torch.nn.Parameter]:
        p = list(self.ac.parameters())
        if self.representation == "z_world":
            p = p + list(self.rep.agent.actor_critic_encoder_parameters())
        return p


def _rel_goal(env: Any, target: Tuple[int, int]) -> Tuple[float, float]:
    """Relative (dx, dy) from the agent to a target cell, normalized by grid size."""
    n = float(getattr(env, "size", 10))
    return ((float(target[0]) - float(env.agent_x)) / n, (float(target[1]) - float(env.agent_y)) / n)


def _reachable_goal_cell(env: Any, dist: int, rng: np.random.RandomState) -> Tuple[int, int]:
    """A random in-bounds cell at ~Manhattan distance `dist` from the agent (the curriculum
    goal). Toroidal or bounded, clipped into the interior."""
    size = int(getattr(env, "size", 10))
    ax, ay = int(env.agent_x), int(env.agent_y)
    for _ in range(20):
        dx = int(rng.randint(-dist, dist + 1))
        dy = dist - abs(dx)
        dy = dy if rng.rand() < 0.5 else -dy
        tx, ty = ax + dx, ay + dy
        if getattr(env, "toroidal", False):
            tx, ty = tx % size, ty % size
        else:
            tx, ty = int(np.clip(tx, 1, size - 2)), int(np.clip(ty, 1, size - 2))
        if (tx, ty) != (ax, ay):
            return (tx, ty)
    return (min(ax + 1, size - 2), ay)


def train_goal_curriculum(
    model: GoalCondModel, env: Any, seed: int, n_episodes: int, steps: int,
    arm_label: str, denom: int,
) -> Dict[str, Any]:
    """IMGEP/AMIGo-lite. Each episode: propose a goal cell at the current curriculum distance
    d_t; train the goal-conditioned policy to reach it (reward +GOAL_REACH_REWARD on reaching,
    -GOAL_STEP_COST per step). Adapt d_t by learning progress (reach-rate over a window). NO
    expert. Trains navigation competence the forage eval then exploits via observation-selected
    goals."""
    params = model.params()
    optimiser = torch.optim.Adam(params, lr=AC_LR)
    reward_std = x734._RunningStd()
    rng = np.random.RandomState(int(seed) + 70007)
    reach_recent: deque = deque(maxlen=CURRICULUM_WINDOW)
    d_t = CURRICULUM_D0

    for ep in range(n_episodes):
        _flat, obs_dict = env.reset()
        model.rep.reset_episode()
        target = _reachable_goal_cell(env, d_t, rng)

        ep_logp: List[torch.Tensor] = []
        ep_value_t: List[torch.Tensor] = []
        ep_entropy: List[torch.Tensor] = []
        ep_value_f: List[float] = []
        ep_rewards: List[float] = []
        terminal = False
        bootstrap_value = 0.0
        reached = False

        goal = _rel_goal(env, target)
        inp = model.encode(obs_dict, goal)
        for _step in range(steps):
            step = model.step(inp, deterministic=False)
            a_idx = int(step.action.reshape(-1)[0].item())
            _flat, _harm, done, _info, obs_dict = env.step(a_idx)
            at_goal = (int(env.agent_x), int(env.agent_y)) == (int(target[0]), int(target[1]))
            r = (GOAL_REACH_REWARD if at_goal else 0.0) - GOAL_STEP_COST
            reward_std.update(r)
            ep_logp.append(step.log_prob.reshape(-1)[0])
            ep_value_t.append(step.value.reshape(-1)[0])
            ep_entropy.append(step.entropy.reshape(-1)[0])
            ep_value_f.append(float(step.value.reshape(-1)[0].item()))
            ep_rewards.append(r)
            if at_goal:
                reached = True
                terminal = True
                break
            if done:
                terminal = True
                break
            goal = _rel_goal(env, target)
            inp = model.encode(obs_dict, goal)

        if not terminal:
            with torch.no_grad():
                boot = model.step(inp, deterministic=False)
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
            value_loss = AC_VALUE_COEF * 0.5 * (value_t - ret_t.detach()).pow(2).mean()
            entropy_bonus = entropy_t.mean()
            loss = policy_loss + value_loss - EXPLORE_ENTROPY_BETA * entropy_bonus
            if torch.isfinite(loss):
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, AC_GRAD_CLIP)
                optimiser.step()

        reach_recent.append(1.0 if reached else 0.0)
        rate = float(sum(reach_recent) / len(reach_recent)) if reach_recent else 0.0
        if len(reach_recent) >= CURRICULUM_WINDOW:
            if rate > CURRICULUM_SUCCESS_HI and d_t < CURRICULUM_D_MAX:
                d_t += 1
                reach_recent.clear()
            elif rate < CURRICULUM_SUCCESS_LO and d_t > CURRICULUM_D0:
                d_t -= 1
                reach_recent.clear()

        cur = ep + 1
        if cur % 200 == 0 or cur == n_episodes:
            print(
                f"  [train] {arm_label} seed={seed} phase=RL ep {cur}/{denom} "
                f"(curriculum_d={d_t} reach_rate={round(rate, 3)})",
                flush=True,
            )

    return {"final_curriculum_distance": int(d_t), "final_goal_reach_rate": round(rate, 6)}


class GoalCondForagePolicy(Policy):
    """Unsupervised forage eval for the curriculum arm: at each step read the agent's own 5x5
    resource_field_view, pick the nearest VISIBLE resource cell as the goal (from OBSERVATION,
    no privileged env.resources), and act greedily via the goal-conditioned policy. Falls back to
    a zero goal (explore) when no resource is visible."""

    def __init__(self, model: GoalCondModel, label: str) -> None:
        self.model = model
        self.name = label

    def act(self, env: Any, obs_dict: Dict[str, Any]) -> int:
        goal = self._observed_goal(env, obs_dict)
        with torch.no_grad():
            inp = self.model.encode(obs_dict, goal)
            step = self.model.step(inp, deterministic=True)
        logits = step.logits.reshape(-1)
        if not torch.isfinite(logits).all():
            return int(np.random.randint(0, int(env.action_dim)))
        return int(torch.argmax(logits, dim=-1).item())

    @staticmethod
    def _observed_goal(env: Any, obs_dict: Dict[str, Any]) -> Tuple[float, float]:
        rfv = obs_dict.get("resource_field_view")
        if rfv is None:
            return (0.0, 0.0)
        view = np.asarray(rfv, dtype=np.float32).reshape(5, 5)
        if float(view.max()) <= 1e-6:
            return (0.0, 0.0)
        # Agent-centered at [2,2]; view cell [2+di, 2+dj] = field at (ax+di, ay+dj).
        bi, bj = np.unravel_index(int(np.argmax(view)), view.shape)
        di, dj = int(bi) - 2, int(bj) - 2
        n = float(getattr(env, "size", 10))
        return (float(di) / n, float(dj) / n)


# ===========================================================================
# Shared self-route + manifest scaffolding (generic across the five legs).
# ===========================================================================
def readiness_from_anchors(local_view_mean: float, oracle_mean: float) -> bool:
    return bool(
        local_view_mean >= COMPETENCE_RESOURCE_FLOOR and oracle_mean >= COMPETENCE_RESOURCE_FLOOR
    )


def reference_band() -> Dict[str, Any]:
    return {
        "floor": float(COMPETENCE_RESOURCE_FLOOR),
        "rnd_novelty_plateau_751": RND_PLATEAU_5_22,
        "bc_expert_748": BC_REFERENCE_32_72,
        "local_view_ceiling_738": CEILING_48_05,
        "greedy_oracle_742": ORACLE_57_2,
        "lift_above_plateau_threshold": round(LIFT_ABOVE_PLATEAU, 6),
    }
