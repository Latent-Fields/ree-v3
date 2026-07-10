#!/opt/local/bin/python3
"""
V3-EXQ-735 -- DRIVE/REWARD-BALANCE APPROACH-WEIGHTING SWEEP (Track-1b of the post-724
competence-recovery campaign).

WHY THIS EXISTS (a DIAGNOSTIC, not an evidence falsifier; the self-routed label below is a
HYPOTHESIS the adjudication pipeline can falsify, NOT a governance verdict; claim_ids=[],
experiment_purpose=diagnostic, non_contributory -- PROMOTES / DEMOTES NOTHING).

V3-EXQ-728 landed the TRAINED all-ON stack on the WS-3 capability yardstick and found a
FORAGE-vs-SURVIVE INVERSION: the agent SURVIVES but does NOT forage --
foraging_competence normalized -0.016 (0.167 res/ep, BELOW the random-walk floor 0.267)
while survival_horizon normalized +6.07 (~70 ticks vs oracle ~21). With
hazard_food_attraction=0.7 the reef-bipartite layout co-locates food with hazard, so an
agent that over-weights AVOIDANCE relative to APPROACH refuses to go near food: it lives a
long time by doing nothing appetitive. V3-EXQ-724 localized the competence deficit as
DIFFUSE across the architecture faces it tested (P1 budget / encoder-thaw / mechanism-count)
-- but 724 held the DRIVE/REWARD BALANCE FIXED. That balance is THIS experiment's axis.

HYPOTHESIS (T1b). The foraging deficit is a DRIVE/REWARD-BALANCE failure (approach
under-weighted vs avoidance), not an architecture-face failure. In the E3 selection argmin
(ree_core/predictors/e3_selector.py score_trajectory):

    score = F(zeta) + lambda_ethical * M(zeta) + rho_residue * Phi(zeta)   # AVOID (costs)
                    - benefit_weight * B(zeta)                              # APPROACH (Go/D1)
                    - goal_weight    * G(zeta)                              # APPROACH (wanting)

(lower score wins). The approach-vs-avoid balance is therefore a single monotone axis:
APPROACH gains {benefit_weight, goal_weight} UP and AVOID gains {lambda_ethical, rho_residue}
DOWN. A0 uses the 724/728 default balance (benefit_weight=1.0, goal_weight=0.5,
lambda_ethical=1.0, rho_residue=0.5) and REPRODUCES the incompetence; A1..A3 staircase
progressively toward approach.

DESIGN -- staircase along ONE approach-vs-avoid axis; 4 drive arms x 3 seeds, plus the two
capability_eval anchors (random-walk FLOOR, greedy-oracle CEILING) per seed.

  A0  default_balance   -- benefit=1.0 goal=0.5 lambda_ethical=1.0 rho_residue=0.5.
        The 724/728 all-ON default balance; the incompetence to explain (control).
  A1  mild_approach     -- benefit=2.0 goal=1.0 lambda_ethical=0.5  rho_residue=0.35.
  A2  moderate_approach -- benefit=4.0 goal=2.0 lambda_ethical=0.25 rho_residue=0.2.
  A3  strong_approach   -- benefit=8.0 goal=4.0 lambda_ethical=0.1  rho_residue=0.1.

Every drive arm is trained with the EXACT V3-EXQ-724 A0 recipe (P0=200 world-model warmup
THEN P1=90 two-head REINFORCE with the lateral-PFC bias + OFC devaluation heads, SD-056 e2
encoder FROZEN through P1) on the IDENTICAL 728 env, and evaluated with the reusable
capability_eval REEForwardPolicy (mechanism-agnostic forward eval; NO P2 OFC-viability
injection) -- so A0 is directly comparable to the 728 trained-all-ON capability point, and
the ONLY thing that changes across arms is the E3 drive/reward balance. The all-ON matched
stack + training helpers are IMPORTED from v3_exq_724 (not copied) so the substrate + recipe
are byte-identical to that verified diagnostic.

POSITIVE CONTROL / ANCHORS (readiness). capability_eval's greedy nearest-resource
OraclePolicy (CEILING / achievability anchor) and uniform RandomPolicy (FLOOR anchor) are
measured per seed under the identical env/protocol. If even the oracle cannot clear
COMPETENCE_RESOURCE_FLOOR the floor is not achievable here => substrate_not_ready_requeue,
NEVER a reward-balance verdict.

DV (load-bearing): foraging_competence = mean resources collected per eval episode
(env.step info transition_type == 'resource'; the SAME statistic as the 724 competence DV,
reused via capability_eval). SECONDARY (reported, not load-bearing): survival_horizon (mean
ticks survived). The EXPECTED reward-balance signature is foraging UP as survival comes DOWN
off its +6.07 outlier -- i.e. the agent trades pure avoidance for appetitive approach.

PRE-REGISTERED SELF-ROUTE (readiness / lever / not-lever) -- HYPOTHESIS, not a verdict:
  * READINESS fails (oracle below floor OR A0 does NOT reproduce incompetence -- A0 already
    forages >= floor -- OR too few eval episodes) -> label `substrate_not_ready_requeue`.
    The premise is not measurable / not reproduced; draw NO reward-balance conclusion.
  * LEVER: readiness holds AND at least one approach-weighted arm (A1/A2/A3) clears the
    floor on a majority of seeds while A0 does not -> label `reward_balance_is_the_lever`.
    The competence deficit is reward-balance-buildable. HYPOTHESIS: route to
    /implement-substrate on principled drive-gain retuning.
  * NOT-LEVER: readiness holds, A0 reproduces incompetence, and NO approach arm clears the
    floor even at strong approach-weighting -> label `reward_balance_not_the_lever`. Reward
    balance is not the lever; route to a different substrate investigation.

EVIDENCE-FOR / EVIDENCE-AGAINST (diagnostic-description requirement):
  * EVIDENCE that the deficit is REWARD-BALANCE: an approach-weighted arm clears the 1.0
    foraging floor on a majority of seeds while A0 does not (and survival trades down off the
    +6.07 outlier, confirming we moved the agent off pure avoidance).
  * EVIDENCE AGAINST reward-balance as the lever: A0 reproduces the incompetence, readiness
    holds, but NO arm -- including strong-approach A3 -- clears the floor.
  * EVIDENCE AGAINST any conclusion (substrate_not_ready_requeue): the oracle cannot clear
    the floor, OR A0 already forages >= floor (728 premise not reproduced), OR insufficient
    eval episodes. No conclusion licensed.
  The self-routed label is a HYPOTHESIS for /failure-autopsy adjudication; this experiment
  PROMOTES / DEMOTES NOTHING and tags NO claim.

BRAKE STATUS: COMPETENCE/reward-balance diagnostic, NOT a conversion or de-commit falsifier
-- the re-derive brakes in conversion_ceiling_campaign_plan.md do NOT apply. New EXQ NUMBER;
tags no claim (re-derive brake counter is zero by construction).

SLEEP DRIVER: none (no sleep loop; use_sleep_loop / sws_enabled / rem_enabled all OFF).

Sourced APIs (verified 2026-07-10):
  experiments/v3_exq_724_competence_localization_diagnostic.py -- ENV_KWARGS, _make_env,
      _base_config_kwargs, _all_on_extra_kwargs, and the P0/P1 training helpers
      (_e2_contrastive_step, _lpfc_reinforce_loss, _ofc_deval_reinforce_loss,
      _consumed_summaries, _obs_harm*/_obs_harm_a/_obs_harm_history) + budget constants.
  experiments/_lib/capability_eval.py -- RandomPolicy / OraclePolicy / REEForwardPolicy,
      evaluate_seed, summarize_arm, build_report, COMPETENCE_RESOURCE_FLOOR.
  ree_core/predictors/e3_selector.py -- score_trajectory approach/avoid weights on config.e3.
  ree_core/utils/config.py -- E3Config {benefit_weight, goal_weight, lambda_ethical,
      rho_residue}; from_dims -> cfg.e3 / cfg.goal.
See REE_assembly/evidence/planning/conversion_ceiling_campaign_plan.md (node CAMPAIGN),
REE_assembly/evidence/planning/ree_ai_design_critique_plan.md (WS-1/WS-3),
REE_assembly/evidence/experiments/v3_exq_728_trained_allon_capability_point_*.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import random

import numpy as np
import torch

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.capability_eval import (
    COMPETENCE_RESOURCE_FLOOR,
    OraclePolicy,
    RandomPolicy,
    REEForwardPolicy,
    build_report,
    evaluate_seed,
    summarize_arm,
)
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

# Reuse the VERIFIED all-ON matched stack + training recipe from V3-EXQ-724 (import, not
# copy) so A0 is byte-identical to the 724/728 trained-all-ON point and the training loop is
# the same code path that ran cleanly on 2026-07-09/10.
from experiments.v3_exq_724_competence_localization_diagnostic import (  # noqa: E402
    ENV_KWARGS,
    E2_CONTRASTIVE_LR,
    E2_TRAIN_EVERY_K_TICKS,
    EMA_DECAY,
    LR_LPFC_BIAS,
    LR_OFC_DEVAL,
    OUTCOME_BUF_MAX,
    TRANSITION_BUFFER_MAX,
    _all_on_extra_kwargs,
    _base_config_kwargs,
    _consumed_summaries,
    _e2_contrastive_step,
    _lpfc_reinforce_loss,
    _make_env,
    _obs_harm,
    _obs_harm_a,
    _obs_harm_history,
    _ofc_deval_reinforce_loss,
)

EXPERIMENT_TYPE = "v3_exq_735_drive_reward_balance_sweep"
QUEUE_ID = "V3-EXQ-735"
CLAIM_IDS: List[str] = []                 # tags NO claim -- pure diagnostic
EXPERIMENT_PURPOSE = "diagnostic"

# ---------------------------------------------------------------------------
# Budget (mirrors V3-EXQ-724 A0 recipe / V3-EXQ-728)
# ---------------------------------------------------------------------------
SEEDS = [42, 43, 44]              # same seeds as 724/728 (A0 directly comparable to 728)
P0_WARMUP_EPISODES = 200          # world-model / e2 warmup (724 A0)
P1_REINFORCE_EPISODES = 90        # two-head REINFORCE (724 A0 short); e2 frozen in P1
P2_EVAL_EPISODES = 20             # capability_eval episodes per (arm, seed)
STEPS_PER_EPISODE = 200

COMPETENCE_MIN_SEEDS = 2          # majority of 3

# ---------------------------------------------------------------------------
# Dry-run budget (tiny; smoke stays fast)
# ---------------------------------------------------------------------------
DRY_RUN_SEEDS = [42, 43]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_P2 = 2
DRY_RUN_STEPS = 30

# ---------------------------------------------------------------------------
# Drive/reward-balance arm table -- ONE monotone approach-vs-avoid staircase.
# A0 = 724/728 default balance (control); A1..A3 progressively approach-weighted.
# ---------------------------------------------------------------------------
DRIVE_ARMS: List[Dict[str, Any]] = [
    {"arm_id": "A0_default_balance", "role": "control",
     "benefit_weight": 1.0, "goal_weight": 0.5, "lambda_ethical": 1.0, "rho_residue": 0.5},
    {"arm_id": "A1_mild_approach", "role": "approach_weighted",
     "benefit_weight": 2.0, "goal_weight": 1.0, "lambda_ethical": 0.5, "rho_residue": 0.35},
    {"arm_id": "A2_moderate_approach", "role": "approach_weighted",
     "benefit_weight": 4.0, "goal_weight": 2.0, "lambda_ethical": 0.25, "rho_residue": 0.2},
    {"arm_id": "A3_strong_approach", "role": "approach_weighted",
     "benefit_weight": 8.0, "goal_weight": 4.0, "lambda_ethical": 0.1, "rho_residue": 0.1},
]
CONTROL_ARM_ID = "A0_default_balance"
APPROACH_ARM_IDS = ("A1_mild_approach", "A2_moderate_approach", "A3_strong_approach")

RANDOM_ARM = "random_walk"
ORACLE_ARM = "greedy_oracle"


# ---------------------------------------------------------------------------
# Agent construction: all-ON stack (724 ARM_ON) with the arm's drive/reward balance
# applied to the E3 selection argmin weights on cfg.e3 (and cfg.goal for goal wanting).
# ---------------------------------------------------------------------------
def _make_agent(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEAgent:
    kwargs = dict(_base_config_kwargs(env))
    kwargs.update(_all_on_extra_kwargs())
    cfg = REEConfig.from_dims(**kwargs)
    # Approach-vs-avoid balance -- set directly on the E3 selector config (the object the
    # score_trajectory argmin reads). A0 values equal the from_dims defaults => byte-identical
    # to the 724/728 all-ON config; A1..A3 shift the balance toward approach.
    cfg.e3.benefit_weight = float(arm["benefit_weight"])
    cfg.e3.goal_weight = float(arm["goal_weight"])
    cfg.e3.lambda_ethical = float(arm["lambda_ethical"])
    cfg.e3.rho_residue = float(arm["rho_residue"])
    # Keep the appetitive goal-seeding gain (GoalState) consistent with the E3 wanting gain.
    cfg.goal.goal_weight = float(arm["goal_weight"])
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# P0 world-model warmup + P1 two-head REINFORCE (724 A0 recipe; e2 FROZEN in P1).
# No P2 here -- capability_eval owns the evaluation. Returns training tick counts.
# ---------------------------------------------------------------------------
def _train_agent(
    agent: REEAgent,
    env: CausalGridWorldV2,
    seed: int,
    p0_episodes: int,
    p1_episodes: int,
) -> Dict[str, Any]:
    has_ofc = getattr(agent, "ofc", None) is not None
    has_lpfc = getattr(agent, "lateral_pfc", None) is not None

    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_CONTRASTIVE_LR)
    bias_opt = (
        torch.optim.Adam(list(agent.lateral_pfc.bias_head_parameters()), lr=LR_LPFC_BIAS)
        if has_lpfc else None
    )
    ofc_deval_opt = (
        torch.optim.Adam(
            list(agent.ofc.devaluation_bias_head_parameters()), lr=LR_OFC_DEVAL
        )
        if has_ofc else None
    )
    transition_buffer: Deque[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = deque(maxlen=TRANSITION_BUFFER_MAX)
    sample_rng = random.Random(seed)

    total_train_eps = p0_episodes + p1_episodes
    p1_start = p0_episodes
    n_p0_ticks = 0
    n_p1_ticks = 0
    error_note: Optional[str] = None

    reinforce_baseline = 0.0
    outcome_buf: List[Tuple[torch.Tensor, int, float]] = []

    for ep in range(total_train_eps):
        is_p1 = ep >= p1_start
        is_p0 = not is_p1
        phase_label = "P1" if is_p1 else "P0"

        _, obs_dict = env.reset()
        agent.reset()

        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        pending_capture: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        tick_in_ep = 0

        ep_reward = 0.0
        ep_buf: List[Tuple[torch.Tensor, int]] = []

        for _step in range(STEPS_PER_EPISODE):
            body = obs_dict["body_state"].float()
            world = obs_dict["world_state"].float()
            if body.dim() == 1:
                body = body.unsqueeze(0)
            if world.dim() == 1:
                world = world.unsqueeze(0)

            latent = agent.sense(
                obs_body=body, obs_world=world,
                obs_harm=_obs_harm(obs_dict),
                obs_harm_a=_obs_harm_a(obs_dict),
                obs_harm_history=_obs_harm_history(obs_dict),
            )

            if pending_capture is not None:
                z0_prev, a_prev = pending_capture
                z1_obs = latent.z_world.detach().reshape(-1).clone()
                if (
                    torch.isfinite(z0_prev).all()
                    and torch.isfinite(a_prev).all()
                    and torch.isfinite(z1_obs).all()
                ):
                    transition_buffer.append((z0_prev, a_prev, z1_obs))
                pending_capture = None

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(
                    z_self_prev, action_prev, latent.z_self.detach()
                )

            ticks = agent.clock.advance()
            wdim = latent.z_world.shape[-1]
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            # P1 REINFORCE snapshot of candidate summaries (heads present in all-ON).
            p1_snap_summaries: Optional[torch.Tensor] = None
            if is_p1 and has_lpfc and candidates and len(candidates) >= 2:
                cs = _consumed_summaries(agent, candidates)
                if cs is not None and torch.isfinite(cs).all():
                    p1_snap_summaries = cs.clone()

            action = agent.select_action(candidates, ticks)
            if action is None:
                idx = int(np.random.randint(0, env.action_dim))
                action = torch.zeros(1, env.action_dim, device=agent.device)
                action[0, idx] = 1.0
                agent._last_action = action
            if not torch.isfinite(action).all():
                if error_note is None:
                    error_note = (
                        f"non-finite action at arm-train seed={seed} "
                        f"phase={phase_label} ep={ep} step={_step}"
                    )
                break

            committed_class = int(action[0].argmax().item())

            if is_p1 and p1_snap_summaries is not None:
                sel = 0
                for ci, c in enumerate(candidates):
                    if (
                        getattr(c, "actions", None) is not None
                        and c.actions.shape[1] >= 1
                        and int(c.actions[:, 0, :].argmax(-1).reshape(-1)[0].item())
                        == committed_class
                    ):
                        sel = min(ci, p1_snap_summaries.shape[0] - 1)
                        break
                ep_buf.append((p1_snap_summaries, sel))

            if is_p1:
                n_p1_ticks += 1
            else:
                n_p0_ticks += 1

            if torch.isfinite(latent.z_world).all() and torch.isfinite(action).all():
                pending_capture = (
                    latent.z_world.detach().reshape(-1).clone(),
                    action.detach().reshape(-1).clone(),
                )

            # SD-056 e2 training -- P0 only (encoder FROZEN through P1, 724 A0 recipe).
            if is_p0 and (tick_in_ep % E2_TRAIN_EVERY_K_TICKS == 0):
                _e2_contrastive_step(
                    agent=agent, buffer=transition_buffer,
                    optimiser=e2_opt, rng=sample_rng,
                )

            _, _harm_signal, done, info, obs_dict = env.step(action)
            harm_signal = float(_harm_signal)
            if is_p1:
                ep_reward += harm_signal

            with torch.no_grad():
                agent.update_residue(
                    harm_signal=harm_signal,
                    world_delta=None,
                    hypothesis_tag=False,
                    owned=True,
                )

            if agent.goal_state is not None:
                benefit_exposure = float(info.get("benefit_exposure", 0.0))
                energy = float(body[0, 3].item())
                drive_level = max(0.0, 1.0 - energy)
                agent.update_z_goal(
                    benefit_exposure=benefit_exposure,
                    drive_level=drive_level,
                )

            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            tick_in_ep += 1
            if done:
                break

        # P1 end-of-episode two-head REINFORCE (lateral-PFC bias + OFC devaluation heads).
        if is_p1 and (has_lpfc or has_ofc):
            reinforce_baseline = (
                EMA_DECAY * reinforce_baseline + (1.0 - EMA_DECAY) * ep_reward
            )
            for cand_features, sel in ep_buf:
                outcome_buf.append((cand_features, sel, ep_reward))
            if len(outcome_buf) > OUTCOME_BUF_MAX:
                outcome_buf = outcome_buf[-OUTCOME_BUF_MAX:]
            if has_lpfc and bias_opt is not None:
                l_loss = _lpfc_reinforce_loss(
                    agent, outcome_buf, reinforce_baseline, agent.device
                )
                if l_loss.requires_grad:
                    bias_opt.zero_grad()
                    l_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.lateral_pfc.bias_head_parameters(), 1.0
                    )
                    bias_opt.step()
            if has_ofc and ofc_deval_opt is not None:
                ofc_loss = _ofc_deval_reinforce_loss(
                    agent, outcome_buf, reinforce_baseline, agent.device
                )
                if ofc_loss.requires_grad:
                    ofc_deval_opt.zero_grad()
                    ofc_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.ofc.devaluation_bias_head_parameters(), 1.0
                    )
                    ofc_deval_opt.step()

        cur = ep + 1
        if cur % 25 == 0 or cur == total_train_eps or is_p1:
            print(
                f"  [train] balance seed={seed} phase={phase_label} "
                f"ep {cur}/{total_train_eps}",
                flush=True,
            )
        if error_note is not None:
            break

    return {
        "n_p0_ticks": int(n_p0_ticks),
        "n_p1_ticks": int(n_p1_ticks),
        "error_note": error_note,
    }


def _mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def _arm_majority_supra(rows: List[Dict[str, Any]], min_seeds: int) -> bool:
    n = sum(1 for r in rows if r is not None and r.get("competence_supra_floor"))
    return bool(n >= min_seeds)


# ---------------------------------------------------------------------------
# One (arm, seed) cell: [drive arms] train all-ON, then capability_eval forward-eval;
# [anchors] capability_eval only. Each cell is wrapped in arm_cell (RNG reset + fingerprint).
# ---------------------------------------------------------------------------
def _anchor_slice(policy_name: str, p2_episodes: int, steps: int) -> Dict[str, Any]:
    return {
        "arm_id": policy_name,
        "kind": "anchor",
        "p2_episodes": int(p2_episodes),
        "steps_per_episode": int(steps),
        "env_kwargs": dict(ENV_KWARGS),
    }


def _drive_slice(arm: Dict[str, Any], p0: int, p1: int, p2: int, steps: int) -> Dict[str, Any]:
    return {
        "arm_id": arm["arm_id"],
        "kind": "all_on_trained",
        "benefit_weight": float(arm["benefit_weight"]),
        "goal_weight": float(arm["goal_weight"]),
        "lambda_ethical": float(arm["lambda_ethical"]),
        "rho_residue": float(arm["rho_residue"]),
        "p0_episodes": int(p0),
        "p1_episodes": int(p1),
        "e2_train_in_p1": False,
        "p2_episodes": int(p2),
        "steps_per_episode": int(steps),
        "all_on_config_sourced_from": "V3-EXQ-714 ARM_ON via V3-EXQ-724",
        "env_kwargs": dict(ENV_KWARGS),
    }


def _run_anchor_cell(
    policy_name: str, seed: int, p2_episodes: int, steps: int
) -> Dict[str, Any]:
    with arm_cell(
        seed,
        config_slice=_anchor_slice(policy_name, p2_episodes, steps),
        script_path=Path(__file__),
        config_slice_declared=True,
        include_driver_script_in_hash=False,   # mint-as-you-go: cross-driver reusable
    ) as cell:
        env = _make_env(seed)
        policy = RandomPolicy(seed) if policy_name == RANDOM_ARM else OraclePolicy()
        row = evaluate_seed(policy, env, n_episodes=p2_episodes, steps_per_episode=steps)
        row["arm_id"] = policy_name
        row["seed"] = int(seed)
        row["n_p0_ticks"] = 0
        row["n_p1_ticks"] = 0
        cell.stamp(row)
    return row


def _run_drive_cell(
    arm: Dict[str, Any], seed: int, p0: int, p1: int, p2: int, steps: int
) -> Dict[str, Any]:
    with arm_cell(
        seed,
        config_slice=_drive_slice(arm, p0, p1, p2, steps),
        script_path=Path(__file__),
        config_slice_declared=True,
        include_driver_script_in_hash=False,   # mint-as-you-go: cross-driver reusable
    ) as cell:
        train_env = _make_env(seed)
        agent = _make_agent(train_env, arm)
        train_info = _train_agent(agent, train_env, seed, p0, p1)
        eval_env = _make_env(seed)
        policy = REEForwardPolicy(agent, name=arm["arm_id"])
        row = evaluate_seed(policy, eval_env, n_episodes=p2, steps_per_episode=steps)
        row["arm_id"] = arm["arm_id"]
        row["arm_role"] = arm["role"]
        row["seed"] = int(seed)
        row["benefit_weight"] = float(arm["benefit_weight"])
        row["goal_weight"] = float(arm["goal_weight"])
        row["lambda_ethical"] = float(arm["lambda_ethical"])
        row["rho_residue"] = float(arm["rho_residue"])
        row["n_p0_ticks"] = int(train_info["n_p0_ticks"])
        row["n_p1_ticks"] = int(train_info["n_p1_ticks"])
        row["train_error_note"] = train_info["error_note"]
        cell.stamp(row)
    return row


def run_experiment(
    seeds: List[int],
    p0_episodes: int,
    p1_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
    dry_run: bool,
) -> Dict[str, Any]:
    print(
        f"Drive/reward-balance approach-weighting sweep "
        f"({len(DRIVE_ARMS)} drive arms + 2 anchors x {len(seeds)} seeds; "
        f"P0={p0_episodes}, P1={p1_episodes}, P2_eval={p2_episodes}, "
        f"steps={steps_per_episode}, dry_run={dry_run})",
        flush=True,
    )

    seed_rows_by_arm: Dict[str, List[Dict[str, Any]]] = {}
    cells: List[Dict[str, Any]] = []

    # ----- Anchors (per seed; drive-balance-independent) -----
    for anchor_name in (RANDOM_ARM, ORACLE_ARM):
        rows: List[Dict[str, Any]] = []
        for s in seeds:
            print(f"Seed {s} Condition {anchor_name}", flush=True)
            row = _run_anchor_cell(anchor_name, s, p2_episodes, steps_per_episode)
            rows.append(row)
            cells.append(row)
            print(
                f"verdict: PASS (anchor={anchor_name} seed={s} "
                f"forage/ep={row['foraging_competence']} "
                f"survival={row['survival_horizon']})",
                flush=True,
            )
        seed_rows_by_arm[anchor_name] = rows

    # ----- Drive-balance arms (train all-ON, then forward-eval) -----
    for arm in DRIVE_ARMS:
        rows = []
        for s in seeds:
            print(f"Seed {s} Condition {arm['arm_id']}", flush=True)
            row = _run_drive_cell(
                arm, s, p0_episodes, p1_episodes, p2_episodes, steps_per_episode
            )
            rows.append(row)
            cells.append(row)
            verdict = "PASS" if row.get("train_error_note") is None else "FAIL"
            print(
                f"verdict: {verdict} (arm={arm['arm_id']} seed={s} "
                f"forage/ep={row['foraging_competence']} "
                f"survival={row['survival_horizon']} "
                f"supra_floor={row['competence_supra_floor']})",
                flush=True,
            )
        seed_rows_by_arm[arm["arm_id"]] = rows

    # ----- Arm summaries + capability report (normalized to [random, oracle]) -----
    arm_summaries = {
        name: summarize_arm(rows) for name, rows in seed_rows_by_arm.items()
    }
    report = build_report(arm_summaries, floor=RANDOM_ARM, ceiling=ORACLE_ARM)

    oracle_foraging = float(
        arm_summaries[ORACLE_ARM].get("foraging_competence_mean", 0.0)
    )
    oracle_min_foraging = min(
        arm_summaries[ORACLE_ARM].get("foraging_competence_per_seed", [0.0]),
        default=0.0,
    )
    oracle_clears_floor = bool(oracle_min_foraging >= COMPETENCE_RESOURCE_FLOOR)

    control_rows = seed_rows_by_arm[CONTROL_ARM_ID]
    control_majority_supra = _arm_majority_supra(control_rows, COMPETENCE_MIN_SEEDS)
    control_reproduces_incompetence = bool(not control_majority_supra)
    control_foraging_mean = float(
        arm_summaries[CONTROL_ARM_ID].get("foraging_competence_mean", 0.0)
    )

    min_eval_eps = min(
        (int(r.get("n_episodes", 0)) for r in cells if r is not None), default=0
    )
    sufficient_eval = bool(min_eval_eps >= 5)

    readiness_met = bool(
        oracle_clears_floor and control_reproduces_incompetence and sufficient_eval
    )

    # ----- Lever gate: any approach arm recovers foraging on a majority of seeds -----
    recovering_arms = [
        aid for aid in APPROACH_ARM_IDS
        if _arm_majority_supra(seed_rows_by_arm[aid], COMPETENCE_MIN_SEEDS)
    ]
    reward_balance_is_lever = bool(recovering_arms)

    # ----- Self-route (HYPOTHESIS, not a verdict) -----
    if not readiness_met:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
    elif reward_balance_is_lever:
        outcome = "PASS"
        label = "reward_balance_is_the_lever"
    else:
        outcome = "FAIL"
        label = "reward_balance_not_the_lever"
    direction = "non_contributory"

    # Survival trade context: control vs best-foraging approach arm.
    control_survival = float(
        arm_summaries[CONTROL_ARM_ID].get("survival_horizon_mean", 0.0)
    )
    approach_survival = {
        aid: float(arm_summaries[aid].get("survival_horizon_mean", 0.0))
        for aid in APPROACH_ARM_IDS
    }
    approach_foraging = {
        aid: float(arm_summaries[aid].get("foraging_competence_mean", 0.0))
        for aid in APPROACH_ARM_IDS
    }

    interpretation = {
        "label": label,
        "recovering_approach_arms": recovering_arms,
        "preconditions": [
            {
                "name": "oracle_foraging_clears_floor",
                "kind": "readiness",
                "description": (
                    "The greedy nearest-resource ORACLE (capability_eval ceiling anchor, no "
                    "agent) clears COMPETENCE_RESOURCE_FLOOR resources/episode in this exact "
                    "env, proving the foraging floor is ACHIEVABLE. Same statistic as the "
                    "load-bearing DV (env.step info transition_type=='resource'). Below-floor "
                    "=> the floor is not achievable here => substrate_not_ready_requeue, NEVER "
                    "a reward-balance verdict."
                ),
                "control": "capability_eval greedy OraclePolicy, same ENV_KWARGS/seed, no agent",
                "measured": float(round(oracle_min_foraging, 6)),
                "threshold": float(COMPETENCE_RESOURCE_FLOOR),
                "met": bool(oracle_clears_floor),
            },
            {
                "name": "control_reproduces_incompetence",
                "kind": "readiness",
                "description": (
                    "The A0 default-balance arm (724/728 all-ON balance) must forage BELOW the "
                    "floor on a majority of seeds -- i.e. the 728 incompetence must reproduce -- "
                    "for a reward-balance conclusion to be meaningful. If A0 already clears the "
                    "floor the premise is not reproduced => substrate_not_ready_requeue."
                ),
                "control": "A0 mean foraging_competence vs floor (majority of seeds)",
                "measured": float(round(control_foraging_mean, 6)),
                "threshold": float(COMPETENCE_RESOURCE_FLOOR),
                "direction": "upper",
                "met": bool(control_reproduces_incompetence),
            },
            {
                "name": "sufficient_eval_episodes_all_cells",
                "kind": "readiness",
                "description": (
                    "Every completed cell must log >= 5 eval episodes so foraging_competence "
                    "is estimable. Below => substrate_not_ready_requeue."
                ),
                "control": "min completed eval episodes across all cells",
                "measured": float(min_eval_eps),
                "threshold": 5.0,
                "met": bool(sufficient_eval),
            },
        ],
        "criteria": [
            {
                "name": "approach_weighting_recovers_foraging",
                "load_bearing": True,
                "passed": bool(reward_balance_is_lever),
            },
        ],
        "criteria_non_degenerate": {
            "oracle_clears_floor": bool(oracle_clears_floor),
            "control_reproduces_incompetence": bool(control_reproduces_incompetence),
            "yardstick_discriminates": bool(
                report["readiness"]["yardstick_discriminates"]
            ),
            "sufficient_eval_episodes": bool(sufficient_eval),
            "any_approach_arm_supra": bool(reward_balance_is_lever),
        },
    }

    return {
        "outcome": outcome,
        "overall_direction": direction,
        "interpretation_label": label,
        "interpretation": interpretation,
        "seeds": seeds,
        "p0_episodes": int(p0_episodes),
        "p1_episodes": int(p1_episodes),
        "p2_episodes": int(p2_episodes),
        "steps_per_episode": int(steps_per_episode),
        "decision_rule_thresholds": {
            "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
            "competence_min_seeds": int(COMPETENCE_MIN_SEEDS),
            "control_arm_id": CONTROL_ARM_ID,
            "approach_arm_ids": list(APPROACH_ARM_IDS),
        },
        "readiness_gates": {
            "oracle_clears_floor": oracle_clears_floor,
            "oracle_mean_foraging": round(oracle_foraging, 6),
            "oracle_min_foraging": round(oracle_min_foraging, 6),
            "control_reproduces_incompetence": control_reproduces_incompetence,
            "control_mean_foraging": round(control_foraging_mean, 6),
            "sufficient_eval_episodes": sufficient_eval,
            "readiness_met": readiness_met,
        },
        "lever_gates": {
            "recovering_approach_arms": recovering_arms,
            "reward_balance_is_the_lever": reward_balance_is_lever,
        },
        "survival_trade_context": {
            "control_foraging_mean": round(control_foraging_mean, 6),
            "control_survival_mean": round(control_survival, 6),
            "approach_foraging_mean": {k: round(v, 6) for k, v in approach_foraging.items()},
            "approach_survival_mean": {k: round(v, 6) for k, v in approach_survival.items()},
        },
        "capability_report": report,
        "arm_summaries": arm_summaries,
        "arm_results": cells,
        "interpretation_grid": {
            "reward_balance_is_the_lever": (
                "readiness holds AND at least one approach-weighted arm (A1/A2/A3) clears the "
                "1.0 foraging floor on a majority of seeds while A0 does not. The competence "
                "deficit is reward-balance-buildable. HYPOTHESIS (not a verdict): route to "
                "/implement-substrate on principled drive-gain retuning."
            ),
            "reward_balance_not_the_lever": (
                "readiness holds, A0 reproduces the incompetence, and NO approach arm -- "
                "including strong-approach A3 -- clears the floor. Reward balance is not the "
                "lever; route to a different substrate investigation, NOT a drive-gain build."
            ),
            "substrate_not_ready_requeue": (
                "the greedy oracle cannot clear the floor (env does not permit it), OR A0 "
                "already forages >= floor (728 premise not reproduced), OR a cell logged fewer "
                "than 5 eval episodes. NOT a verdict -- re-examine env/floor/budget and "
                "re-queue. Draw NO conclusion about the reward-balance root."
            ),
        },
    }


def _build_manifest(
    result: Dict[str, Any],
    timestamp_utc: str,
    dry_run: bool,
) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    rg = result["readiness_gates"]
    lg = result["lever_gates"]
    return {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp_utc,
        "outcome": result["outcome"],
        "evidence_direction": result["overall_direction"],
        "interpretation_label": result["interpretation_label"],
        "interpretation": result["interpretation"],
        "sleep_driver_pattern": "none",
        "evidence_direction_note": (
            f"V3-EXQ-735 DRIVE/REWARD-BALANCE APPROACH-WEIGHTING SWEEP (Track-1b; "
            f"experiment_purpose=diagnostic, claim_ids=[], non_contributory -- EXCLUDED from "
            f"governance scoring; PROMOTES / DEMOTES NOTHING). Tests whether the V3-EXQ-728 "
            f"forage-vs-survive INVERSION (foraging normalized -0.016 / survival +6.07 on the "
            f"trained all-ON stack) is a DRIVE/REWARD-BALANCE failure (approach under-weighted "
            f"vs avoidance) rather than an architecture-face failure (724 held the balance "
            f"FIXED and found the deficit diffuse). Staircase along ONE approach-vs-avoid axis "
            f"in the E3 selection argmin: APPROACH gains benefit_weight/goal_weight UP + AVOID "
            f"gains lambda_ethical/rho_residue DOWN. A0=724/728 default balance (control, "
            f"reproduces incompetence); A1/A2/A3 progressively approach-weighted. Each drive "
            f"arm trains the all-ON stack with the 724 A0 recipe (P0={result['p0_episodes']} "
            f"warmup + P1={result['p1_episodes']} two-head REINFORCE, e2 frozen in P1) and is "
            f"evaluated via the reusable capability_eval REEForwardPolicy, so A0 is directly "
            f"comparable to the 728 trained-all-ON point. Load-bearing DV: foraging_competence "
            f"(mean resources/episode via env.step transition_type=='resource'); secondary: "
            f"survival_horizon (expect forage UP / survival DOWN off the +6.07 outlier). "
            f"Readiness: greedy oracle clears the floor (oracle_min_foraging="
            f"{rg['oracle_min_foraging']}, clears_floor={rg['oracle_clears_floor']}). "
            f"Self-route (HYPOTHESIS, not a verdict): readiness_met={rg['readiness_met']}; if "
            f"an approach arm recovers competence on a majority of seeds -> "
            f"reward_balance_is_the_lever (recovering_arms={lg['recovering_approach_arms']}, "
            f"route /implement-substrate on drive-gain retuning); else "
            f"reward_balance_not_the_lever; if readiness fails -> "
            f"substrate_not_ready_requeue. interpretation_label="
            f"{result['interpretation_label']}. Maps conversion_ceiling_campaign:CAMPAIGN. "
            f"Route to /failure-autopsy for adjudication before any governance action."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "design": (
                "drive/reward-balance approach-weighting staircase; 4 drive arms x 3 seeds "
                "+ capability_eval random-walk floor / greedy-oracle ceiling anchors"
            ),
            "approach_vs_avoid_axis": (
                "E3 score_trajectory argmin: score = F + lambda_ethical*M + rho_residue*Phi "
                "- benefit_weight*B - goal_weight*G; APPROACH={benefit_weight, goal_weight} UP, "
                "AVOID={lambda_ethical, rho_residue} DOWN"
            ),
            "drive_arms": {
                a["arm_id"]: {
                    "role": a["role"],
                    "benefit_weight": a["benefit_weight"],
                    "goal_weight": a["goal_weight"],
                    "lambda_ethical": a["lambda_ethical"],
                    "rho_residue": a["rho_residue"],
                }
                for a in DRIVE_ARMS
            },
            "load_bearing_dv": "foraging_competence (capability_eval; env.step transition_type=='resource')",
            "secondary_dv": "survival_horizon (mean ticks survived; expect trade DOWN off +6.07 outlier)",
            "reusable_block": "experiments/_lib/capability_eval.py",
            "all_on_config_sourced_from": "V3-EXQ-714 ARM_ON via V3-EXQ-724 (imported, not copied)",
            "training_recipe_sourced_from": "V3-EXQ-724 A0_baseline_allon_p1short_frozen",
            "p0_warmup_episodes": int(result["p0_episodes"]),
            "p1_reinforce_episodes": int(result["p1_episodes"]),
            "e2_train_in_p1": False,
            "alpha_world": 0.9,
            "reef_bipartite_layout": True,
            "use_sleep_loop": False,
            "control_arm_reuse_note": (
                "A0 default-balance all-ON cell emitted reuse-eligible "
                "(include_driver_script_in_hash=False) -- mint-as-you-go self-mint; the "
                "approach-weighted arms re-train so their fingerprints correctly refuse."
            ),
        },
        "result": result,
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description=(
            "V3-EXQ-735 drive/reward-balance approach-weighting sweep DIAGNOSTIC "
            "(is the 728 forage-vs-survive inversion a reward-balance failure; claim_ids=[])"
        )
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        p0 = DRY_RUN_P0
        p1 = DRY_RUN_P1
        p2 = DRY_RUN_P2
        steps = DRY_RUN_STEPS
    else:
        seeds = list(SEEDS)
        p0 = P0_WARMUP_EPISODES
        p1 = P1_REINFORCE_EPISODES
        p2 = P2_EVAL_EPISODES
        steps = STEPS_PER_EPISODE

    result = run_experiment(
        seeds=seeds,
        p0_episodes=p0,
        p1_episodes=p1,
        p2_episodes=p2,
        steps_per_episode=steps,
        dry_run=bool(args.dry_run),
    )

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run))

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{manifest['run_id']}.json"
    if args.dry_run:
        out_path = out_dir / f"_dry_{manifest['run_id']}.json"

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"manifest: {out_path}", flush=True)
    if not args.dry_run:
        print(f"Result written to: {out_path}", flush=True)
    rg = result["readiness_gates"]
    lg = result["lever_gates"]
    print(
        f"outcome: {result['outcome']} label={result['interpretation_label']} "
        f"readiness={rg['readiness_met']} "
        f"oracle_min_forage={rg['oracle_min_foraging']} "
        f"control_forage={rg['control_mean_foraging']} "
        f"recovering_arms={lg['recovering_approach_arms']}",
        flush=True,
    )
    for aid, st in result["arm_summaries"].items():
        print(
            f"  ARM {aid}: forage/ep_mean={st.get('foraging_competence_mean')} "
            f"survival_mean={st.get('survival_horizon_mean')} "
            f"supra_seeds={st.get('n_seeds_supra_floor')}/{st.get('n_seeds')} "
            f"majority_supra={st.get('majority_supra_floor')}",
            flush=True,
        )

    if args.dry_run:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass

    outcome_norm = result["outcome"].upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    manifest_for_sentinel = str(out_path) if not args.dry_run else None
    return outcome_emit, manifest_for_sentinel, bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry_run = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path, dry_run=_dry_run)
    sys.exit(0)
