"""
capability_eval.py -- CLAIM-AGNOSTIC capability yardstick for the ree-v3 harness (WS-3).

WHY THIS EXISTS. The conversion-ceiling campaign cannot currently separate "the
structural claim is wrong" from "the substrate is too coarse to carry the signal": the
fully-integrated all-ON agent forages 0.065/0.0/0.455 resources/episode (below the 1.0
competence floor, 0/3 seeds; failure_autopsy_V3-EXQ-719a_2026-07-08), so every
committed-action structural DV is measured on an agent that cannot act. This module is the
DENOMINATOR: a minimal set of behavioural capability metrics that depend on NO REE
mechanism claim, so any experiment can report "structure X moved capability metric Y by Z
on a substrate already above the competence floor."

It is deliberately claim-agnostic: it measures only what the environment exposes (resource
contacts, agent health/position, episode length). It tags no claim, promotes/demotes
nothing, and is excluded from governance scoring by construction (the experiments that call
it declare experiment_purpose in {"baseline", "diagnostic"}).

THE FOUR METRICS (all read straight off env transitions / env state; no latent, no gate):
  1. foraging_competence  -- mean resources collected per episode
                             (env.step info transition_type == "resource"). This is the
                             SAME statistic as the V3-EXQ-724 competence DV, reused here.
  2. survival_horizon     -- mean ticks survived per episode, plus death_rate (fraction of
                             episodes ending with agent_health <= 0 before the step cap).
  3. goal_reach_rate      -- fraction of episodes in which the agent reached its appetitive
                             goal at all (collected >= 1 resource).
  4. planning_depth       -- mean over episodes of the longest run of strictly-decreasing
                             Manhattan distance to the nearest resource (an env-observable
                             proxy for sustained multi-step directed approach; a greedy
                             forager yields long runs, a random walker ~1).

CALIBRATION ANCHORS. The scale of each metric is fixed by two reference policies measured
under the identical env/seed/protocol:
  * greedy_oracle -- a nearest-resource greedy forager (no agent). CEILING / achievability
                     control, reused verbatim from V3-EXQ-724's positive control. If even
                     the oracle cannot clear COMPETENCE_RESOURCE_FLOOR, the floor is not
                     achievable in this env and NO agent conclusion is licensed
                     (readiness -> substrate_not_ready_requeue).
  * random_walk   -- a uniform-random action policy. FLOOR anchor.
The report normalizes any measured policy to [random_floor, oracle_ceiling] per metric.

USAGE (from an experiment script; the script owns arm_cell fingerprinting + training):

    from experiments._lib.capability_eval import (
        OraclePolicy, RandomPolicy, REEForwardPolicy,
        evaluate_seed, summarize_arm, build_report, COMPETENCE_RESOURCE_FLOOR,
    )
    # per (arm, seed) cell (inside arm_cell(...)):
    policy = REEForwardPolicy(trained_agent)          # or OraclePolicy() / RandomPolicy(seed)
    seed_row = evaluate_seed(policy, env, n_episodes=20, steps_per_episode=200)
    # after all cells:
    arm_summaries = {name: summarize_arm(seed_rows) for name, seed_rows in ...}
    report = build_report(arm_summaries, floor="random_walk", ceiling="greedy_oracle")

Sourced APIs (verified 2026-07-09):
  ree_core/environment/causal_grid_world.py -- step() returns
      (flat_obs, harm_signal, done, info, obs_dict); action int or one-hot (argmax);
      env.resources / env.agent_x / env.agent_y / env.agent_health; done on health<=0.
  ree_core/agent.py -- REEAgent.sense / clock.advance / _e1_tick / generate_trajectories
      / select_action / reset / update_z_goal / update_residue / goal_state.
This module is ASCII-only in all runtime strings.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

# Pre-registered behavioural competence floor (shared with V3-EXQ-724): a decisive forager
# clears >= 1.0 resource/episode comfortably; its ACHIEVABILITY in-env is validated per run
# by the greedy-oracle ceiling anchor (readiness gate).
COMPETENCE_RESOURCE_FLOOR = 1.0

# The four claim-agnostic capability metric keys, in report order.
METRIC_KEYS = (
    "foraging_competence",
    "survival_horizon",
    "goal_reach_rate",
    "planning_depth",
)


# ---------------------------------------------------------------------------
# Env-observable helpers (no agent, no latent)
# ---------------------------------------------------------------------------
def nearest_resource_manhattan(env: Any) -> Optional[int]:
    """Manhattan distance from the agent to the nearest resource, or None if none exist."""
    resources = getattr(env, "resources", None)
    if not resources:
        return None
    ax, ay = int(env.agent_x), int(env.agent_y)
    return int(min(abs(int(r[0]) - ax) + abs(int(r[1]) - ay) for r in resources))


def _longest_strictly_decreasing_run(seq: List[Optional[int]]) -> int:
    """Longest run of strictly-decreasing consecutive values (None breaks a run)."""
    best = 0
    cur = 0
    prev: Optional[int] = None
    for v in seq:
        if v is None or prev is None:
            cur = 0
        elif v < prev:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
        prev = v
    # `best` = the largest number of consecutive strictly-decreasing transitions, i.e. the
    # count of successive approach steps in the longest directed run (a single decrease -> 1).
    return int(best)


def _obs_harm(obs_dict: Dict[str, Any]) -> Optional[torch.Tensor]:
    h = obs_dict.get("harm_obs")
    return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_a(obs_dict: Dict[str, Any]) -> Optional[torch.Tensor]:
    h = obs_dict.get("harm_obs_a")
    return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_history(obs_dict: Dict[str, Any]) -> Optional[torch.Tensor]:
    h = obs_dict.get("harm_history")
    return h.float().unsqueeze(0) if h is not None else None


# ---------------------------------------------------------------------------
# Policy interface. act(env, obs_dict) -> int action index; reset/post_step optional.
# ---------------------------------------------------------------------------
class Policy:
    name: str = "policy"

    def reset(self, env: Any) -> None:  # noqa: D401
        """Called at the start of every eval episode (env already reset)."""
        return None

    def act(self, env: Any, obs_dict: Dict[str, Any]) -> int:
        raise NotImplementedError

    def post_step(self, env: Any, info: Dict[str, Any], obs_dict: Dict[str, Any]) -> None:
        """Called after each env.step (deployed-policy internal updates); default no-op."""
        return None


class RandomPolicy(Policy):
    """Uniform-random action policy -- the FLOOR anchor."""

    name = "random_walk"

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.RandomState(int(seed))

    def act(self, env: Any, obs_dict: Dict[str, Any]) -> int:
        return int(self._rng.randint(0, int(env.action_dim)))


class OraclePolicy(Policy):
    """Greedy nearest-resource forager (no agent) -- the CEILING / achievability anchor.

    Verbatim policy from V3-EXQ-724's positive control. ACTIONS index convention:
    {0:N(-1,0), 1:S(1,0), 2:W(0,-1), 3:E(0,1), 4:stay}; grid[x,y] with x=row.
    """

    name = "greedy_oracle"

    def act(self, env: Any, obs_dict: Dict[str, Any]) -> int:
        resources = getattr(env, "resources", None)
        if not resources:
            return 4
        ax, ay = int(env.agent_x), int(env.agent_y)
        best = min(resources, key=lambda r: abs(int(r[0]) - ax) + abs(int(r[1]) - ay))
        rx, ry = int(best[0]), int(best[1])
        dx, dy = rx - ax, ry - ay
        if dx == 0 and dy == 0:
            return 4
        if abs(dx) >= abs(dy):
            return 1 if dx > 0 else 0
        return 3 if dy > 0 else 2


class LocalViewGreedyPolicy(Policy):
    """Greedy resource-gradient climber that reads ONLY the agent's 5x5 local view
    (obs_dict["resource_field_view"]) -- the LOCAL-VIEW-ACHIEVABLE ceiling anchor
    (WS-1 re-operationalization; failure_autopsy_V3-EXQ-732a_2026-07-10).

    WHY THIS EXISTS. OraclePolicy reads PRIVILEGED GLOBAL state (env.resources, all coords)
    and can beeline from anywhere; it proves the floor is achievable *with global info*, but
    NOT that it is achievable from the same partial observation the learner (REE or a vanilla
    RL control) actually sees. That gap is the 732/732a confound: a sub-floor learner reading
    was uninterpretable -- "observation interface unlearnable" vs "the yardstick is unfair"
    (global oracle vs 5x5 local view). This policy closes it: it forages using ONLY the 5x5
    resource_field_view (a subset of world_state, exactly what the REE encoder senses), so if
    it clears COMPETENCE_RESOURCE_FLOOR then the floor is reachable FROM THE LOCAL VIEW, and a
    same-obs learner that stays sub-floor is genuinely under-powered, not obs-starved.

    Mechanism: the field view is agent-centered at cell [2,2]; env action a has delta
    ACTIONS[a] = (dx, dy), landing on view cell [2+dx, 2+dy] (verified 2026-07-10 against
    causal_grid_world.py ACTIONS {0:N(-1,0),1:S(1,0),2:W(0,-1),3:E(0,1),4:stay} and the
    field-view construction t_view[di+2, dj+2] = field[ax+di, ay+dj]). One-step gradient
    ascent: pick the action whose destination cell has the highest resource-field value. When
    the local window carries no gradient (all destination cells within `flat_eps`, e.g. no
    resource within the 5x5 radius), take a random MOVE step (never stay) so the forager
    explores out of a flat region instead of stalling -- a fair local chemotaxis policy, not a
    hobbled one. Falls back to uniform-random if resource_field_view is absent
    (use_proxy_fields=False), which makes it degrade to the RandomPolicy floor rather than
    silently mis-report achievability.
    """

    name = "local_view_greedy"

    # Action index -> (dx, dy), mirroring env.ACTIONS (row=x, col=y). Destination view cell
    # for action a is [2 + dx, 2 + dy] within the agent-centered 5x5 resource_field_view.
    _DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}

    def __init__(self, seed: int = 0, flat_eps: float = 1e-3) -> None:
        self._rng = np.random.RandomState(int(seed))
        self._flat_eps = float(flat_eps)

    def act(self, env: Any, obs_dict: Dict[str, Any]) -> int:
        rfv = obs_dict.get("resource_field_view")
        action_dim = int(env.action_dim)
        if rfv is None:
            # No local resource-field channel (use_proxy_fields=False) -> degrade to floor.
            return int(self._rng.randint(0, action_dim))
        view = np.asarray(rfv, dtype=np.float32).reshape(5, 5)
        # Only actions the env actually exposes; destination-cell value per action.
        move_actions = [a for a in range(action_dim) if a in self._DELTAS]
        dest_vals = {}
        for a in move_actions:
            dx, dy = self._DELTAS[a]
            dest_vals[a] = float(view[2 + dx, 2 + dy])
        vals = list(dest_vals.values())
        if (max(vals) - min(vals)) < self._flat_eps:
            # Flat window (no gradient signal): explore with a random NON-stay move.
            moves = [a for a in move_actions if a != 4] or move_actions
            return int(moves[self._rng.randint(0, len(moves))])
        best = max(dest_vals, key=lambda a: dest_vals[a])
        return int(best)


class REEForwardPolicy(Policy):
    """Plain frozen forward-eval of an REEAgent -- proves the yardstick attaches to the
    real substrate. Mirrors the V3-EXQ-724 / 719a eval inner loop (sense -> e1 tick ->
    generate_trajectories -> select_action), plus the deployed-policy z_goal/residue
    updates that drive appetitive foraging. Does NO training and injects NO gate signals.
    """

    name = "ree_forward"

    def __init__(self, agent: Any, name: Optional[str] = None) -> None:
        self.agent = agent
        if name is not None:
            self.name = name

    def reset(self, env: Any) -> None:
        self.agent.reset()

    def act(self, env: Any, obs_dict: Dict[str, Any]) -> int:
        agent = self.agent
        body = obs_dict["body_state"].float()
        world = obs_dict["world_state"].float()
        if body.dim() == 1:
            body = body.unsqueeze(0)
        if world.dim() == 1:
            world = world.unsqueeze(0)
        latent = agent.sense(
            obs_body=body,
            obs_world=world,
            obs_harm=_obs_harm(obs_dict),
            obs_harm_a=_obs_harm_a(obs_dict),
            obs_harm_history=_obs_harm_history(obs_dict),
        )
        ticks = agent.clock.advance()
        wdim = latent.z_world.shape[-1]
        e1_prior = (
            agent._e1_tick(latent)
            if ticks.get("e1_tick", False)
            else torch.zeros(1, wdim, device=agent.device)
        )
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        action = agent.select_action(candidates, ticks)
        if action is None or not torch.isfinite(action).all():
            return int(np.random.randint(0, int(env.action_dim)))
        return int(action[0].argmax().item())

    def post_step(self, env: Any, info: Dict[str, Any], obs_dict: Dict[str, Any]) -> None:
        agent = self.agent
        harm_signal = float(info.get("harm_signal", 0.0)) if isinstance(info, dict) else 0.0
        with torch.no_grad():
            agent.update_residue(
                harm_signal=harm_signal,
                world_delta=None,
                hypothesis_tag=False,
                owned=True,
            )
        if getattr(agent, "goal_state", None) is not None:
            benefit_exposure = float(info.get("benefit_exposure", 0.0)) if isinstance(info, dict) else 0.0
            body = obs_dict["body_state"].float()
            if body.dim() == 1:
                body = body.unsqueeze(0)
            energy = float(body[0, 3].item())
            drive_level = max(0.0, 1.0 - energy)
            agent.update_z_goal(benefit_exposure=benefit_exposure, drive_level=drive_level)


# ---------------------------------------------------------------------------
# Per-episode measurement loop (policy-agnostic)
# ---------------------------------------------------------------------------
def rollout_episode(
    env: Any,
    obs_dict: Dict[str, Any],
    policy: Policy,
    steps_per_episode: int,
) -> Dict[str, Any]:
    """Run ONE episode under `policy`, capturing the four claim-agnostic capability metrics.

    Caller must have already called env.reset() (yielding obs_dict) and policy.reset(env).
    """
    resources = 0
    hazard_hits = 0
    contaminations = 0
    reward = 0.0
    ticks = 0
    died = False
    dist_seq: List[Optional[int]] = []
    cur_obs = obs_dict

    for _step in range(steps_per_episode):
        dist_seq.append(nearest_resource_manhattan(env))  # pre-step directedness sample
        action = policy.act(env, cur_obs)
        _flat, harm_signal, done, info, cur_obs = env.step(action)
        if not isinstance(info, dict):
            info = {}
        info.setdefault("harm_signal", float(harm_signal))
        ticks += 1
        reward += float(harm_signal)
        ttype = str(info.get("transition_type", "none"))
        if ttype == "resource":
            resources += 1
        elif ttype == "env_caused_hazard":
            hazard_hits += 1
        if ttype == "agent_caused_hazard" or float(info.get("contamination_delta", 0.0)) > 0.0:
            contaminations += 1
        policy.post_step(env, info, cur_obs)
        if done:
            # done fires on agent_health<=0 (death) or the env's internal step cap.
            if float(getattr(env, "agent_health", 1.0)) <= 0.0:
                died = True
            break

    planning_depth = _longest_strictly_decreasing_run(dist_seq)
    return {
        "resources": int(resources),
        "ticks_survived": int(ticks),
        "died": bool(died),
        "reached_goal": bool(resources >= 1),
        "planning_depth": int(planning_depth),
        "hazard_hits": int(hazard_hits),
        "contaminations": int(contaminations),
        "episode_reward": round(float(reward), 6),
    }


def evaluate_seed(
    policy: Policy,
    env: Any,
    n_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    """Run a policy on one env/seed for n_episodes; return the four metrics + context.

    `env` is a freshly-constructed env for this seed. `policy` is bound to this seed (for
    REEForwardPolicy the agent must already be built/trained for this seed).
    """
    ep_rows: List[Dict[str, Any]] = []
    for _ep in range(int(n_episodes)):
        _flat, obs_dict = env.reset()
        policy.reset(env)
        ep_rows.append(rollout_episode(env, obs_dict, policy, steps_per_episode))

    n = len(ep_rows)

    def _mean(key: str) -> float:
        return float(sum(r[key] for r in ep_rows) / n) if n else 0.0

    foraging = _mean("resources")
    survival = _mean("ticks_survived")
    death_rate = float(sum(1 for r in ep_rows if r["died"]) / n) if n else 0.0
    goal_reach = float(sum(1 for r in ep_rows if r["reached_goal"]) / n) if n else 0.0
    planning = _mean("planning_depth")
    return {
        "policy": policy.name,
        "n_episodes": int(n),
        # ---- the four claim-agnostic capability metrics ----
        "foraging_competence": round(foraging, 6),
        "survival_horizon": round(survival, 6),
        "death_rate": round(death_rate, 6),
        "goal_reach_rate": round(goal_reach, 6),
        "planning_depth": round(planning, 6),
        # ---- context ----
        "mean_hazard_hits": round(_mean("hazard_hits"), 6),
        "mean_contaminations": round(_mean("contaminations"), 6),
        "mean_episode_reward": round(_mean("episode_reward"), 6),
        "competence_supra_floor": bool(foraging >= COMPETENCE_RESOURCE_FLOOR),
        "per_episode_resources": [int(r["resources"]) for r in ep_rows],
    }


def _mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def summarize_arm(seed_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate a policy arm's per-seed rows into arm-level metric means + spread."""
    ok = [r for r in seed_rows if r is not None]
    out: Dict[str, Any] = {
        "policy": ok[0]["policy"] if ok else None,
        "n_seeds": int(len(ok)),
    }
    for key in ("foraging_competence", "survival_horizon", "death_rate",
                "goal_reach_rate", "planning_depth", "mean_hazard_hits",
                "mean_contaminations", "mean_episode_reward"):
        out[key + "_mean"] = round(_mean([r[key] for r in ok]), 6)
        out[key + "_per_seed"] = [round(float(r[key]), 6) for r in ok]
    out["n_seeds_supra_floor"] = int(sum(1 for r in ok if r["competence_supra_floor"]))
    out["majority_supra_floor"] = bool(out["n_seeds_supra_floor"] >= (len(ok) + 1) // 2) if ok else False
    return out


def _normalize(value: float, floor: float, ceiling: float) -> Optional[float]:
    """Position of `value` within [floor, ceiling]; None when the anchor span is ~0."""
    span = ceiling - floor
    if abs(span) < 1e-9:
        return None
    return round((value - floor) / span, 6)


def build_report(
    arm_summaries: Dict[str, Dict[str, Any]],
    floor: str = "random_walk",
    ceiling: str = "greedy_oracle",
) -> Dict[str, Any]:
    """Build the standard capability reporting block.

    For each of the four metrics: the arm-level mean for every policy, plus every non-anchor
    policy's normalized position in [floor_anchor, ceiling_anchor]. Also emits the readiness
    gate (oracle clears the competence floor) and a non-degeneracy gate (ceiling > floor
    spread on foraging), so a calling baseline/diagnostic experiment can self-route.
    """
    floor_arm = arm_summaries.get(floor)
    ceil_arm = arm_summaries.get(ceiling)
    metrics_block: Dict[str, Any] = {}
    for key in METRIC_KEYS:
        mkey = key + "_mean"
        per_policy = {name: arm.get(mkey) for name, arm in arm_summaries.items()}
        entry: Dict[str, Any] = {"per_policy_mean": per_policy}
        if floor_arm is not None and ceil_arm is not None:
            f = float(floor_arm.get(mkey, 0.0))
            c = float(ceil_arm.get(mkey, 0.0))
            entry["floor"] = round(f, 6)
            entry["ceiling"] = round(c, 6)
            entry["normalized_position"] = {
                name: _normalize(float(arm.get(mkey, 0.0)), f, c)
                for name, arm in arm_summaries.items()
                if name not in (floor, ceiling)
            }
        metrics_block[key] = entry

    oracle_foraging = float(ceil_arm.get("foraging_competence_mean", 0.0)) if ceil_arm else 0.0
    floor_foraging = float(floor_arm.get("foraging_competence_mean", 0.0)) if floor_arm else 0.0
    oracle_clears_floor = bool(ceil_arm is not None and oracle_foraging >= COMPETENCE_RESOURCE_FLOOR)
    yardstick_discriminates = bool(
        ceil_arm is not None and floor_arm is not None
        and (oracle_foraging - floor_foraging) > 0.0
    )
    return {
        "metrics": metrics_block,
        "floor_anchor": floor,
        "ceiling_anchor": ceiling,
        "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
        "readiness": {
            "oracle_clears_floor": oracle_clears_floor,
            "oracle_foraging_competence": round(oracle_foraging, 6),
            "floor_foraging_competence": round(floor_foraging, 6),
            "yardstick_discriminates": yardstick_discriminates,
        },
    }
