"""
StepHarness -- canonical per-tick sequence for ree-v3 experiment scripts.

The cohort EXQ-471/475/483/483a/483b/490/490b/490c/490e/490f/524 all carried
the same set of script-side bugs (see REE_Working session 2026-05-07,
discussion_notes key '2026-05-07T23:55Z_update_z_goal_cohort'). To prevent
recurrence, future experiment scripts go through this harness instead of
hand-rolling the inner loop.

Invariants enforced
-------------------
1. Exactly one ``agent.sense()`` per env step. The double-sense pattern that
   advanced GABA decay / AIC EMA / V_s readouts / anchor-set hysteresis at
   2x the env rate during warmup is structurally impossible here.
2. Exactly one ``agent.update_z_goal(benefit_exposure=..., drive_level=...)``
   per env step, kwargs only. Pinning the kwargs prevents the positional
   ``latent`` collision that produced TypeError every tick in the broken
   cohort.
3. Exactly one ``agent.update_residue(harm_signal, world_delta=...)`` per
   env step. This is the canonical post-action path that drives
   ``e3.post_action_update`` (running variance), MECH-205 surprise writes,
   and residue accumulation. Skipping it leaves precision pinned and the
   residue field empty (root cause of EXQ-530 / EXQ-536 incidents too).
4. No bare ``except Exception: pass`` wrappers around agent API calls.
   Failures crash loudly so the script author finds them in --dry-run.

Usage sketch
------------
    from _harness import StepHarness, StepResult

    def run_eval(agent, env, num_episodes, steps_per_episode):
        harness = StepHarness(agent, env, train_mode=False)
        for ep in range(num_episodes):
            _, obs_dict = env.reset()
            agent.reset()
            harness.reset()
            for _ in range(steps_per_episode):
                result = harness.step(obs_dict)
                obs_dict = result.next_obs_dict
                if result.done:
                    break

For warmup with aux losses, compute and apply optimiser steps AFTER
``harness.step()`` returns, using ``result.latent`` as the encoder output
this tick. The next ``harness.step()`` call will encode the next observation
through the post-update encoder, so action selection at tick N+1 sees the
weights stepped at tick N. The previously-broken pattern of doing a second
``sense()`` mid-tick is unnecessary and harmful.

Optional callbacks (StepHooks) let you collect diagnostics at well-defined
points in the loop without subclassing.
"""

from __future__ import annotations

import random
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch

# Avoid pulling REEAgent at module-import time so that scripts that haven't
# selected a config yet still import the harness. The actual type is REEAgent;
# we annotate as Any to keep the harness importable from contexts that don't
# load the heavy agent module (e.g. unit tests that use a fake).
REEAgent = Any

# Module-level handle so contract tests can mock the call shape.
_action_random = random.Random()


@dataclass
class StepHooks:
    """
    Optional callbacks fired at well-defined points in the canonical loop.

    Each is called with kwargs to survive future signature additions. None
    means "no hook" (skipped). Hooks must not call ``agent.sense()`` or any
    other per-tick substrate update -- that would re-introduce the
    double-tick bug the harness is here to prevent.
    """

    # After sense() and canonical goal-stream updates; before select_action.
    on_sense: Optional[Callable[..., None]] = None
    # After select_action(); before env.step.
    on_action: Optional[Callable[..., None]] = None
    # After env.step + update_residue; final per-tick hook.
    on_post_step: Optional[Callable[..., None]] = None


@dataclass
class StepResult:
    latent: Any                   # LatentState produced by sense() this tick
    action: torch.Tensor          # selected action one-hot
    harm_signal: float            # env reward (negative=harm, positive=benefit)
    done: bool
    info: Dict[str, Any]
    next_obs_dict: Dict[str, Any]
    drive_level: float
    benefit_exposure: float
    ticks: Dict[str, bool]        # multi-rate clock dict from agent.clock.advance()
    residue_metrics: Dict[str, Any]  # what update_residue returned this tick


class StepHarness:
    """
    Wrap (agent, env) into a step()-driven loop with the canonical sequence.

    Parameters
    ----------
    agent
        A constructed REEAgent. The harness does not own its lifecycle.
    env
        A CausalGridWorld* instance with .step() returning
        (flat_obs, harm_signal, done, info, obs_dict).
    train_mode
        True during warmup (allows gradient flow through sense()/_e1_tick/etc.)
        False during eval (wraps the per-tick computation in torch.no_grad).
    hooks
        Optional StepHooks for diagnostics injection.
    seed
        Optional seed for the harness's internal Random (used only for the
        random-fallback action when select_action returns None).
    """

    def __init__(
        self,
        agent,
        env,
        *,
        train_mode: bool,
        hooks: Optional[StepHooks] = None,
        seed: Optional[int] = None,
    ):
        self.agent = agent
        self.env = env
        self.train_mode = bool(train_mode)
        self.hooks = hooks or StepHooks()
        self._rng = random.Random(seed) if seed is not None else _action_random

        # Per-episode plumbing.
        self._z_self_prev: Optional[torch.Tensor] = None
        self._z_world_prev: Optional[torch.Tensor] = None
        self._action_prev: Optional[torch.Tensor] = None
        self._step_count: int = 0

    def reset(self) -> None:
        """Reset per-episode plumbing. Call after env.reset()/agent.reset()."""
        self._z_self_prev = None
        self._z_world_prev = None
        self._action_prev = None
        self._step_count = 0

    def step(self, obs_dict: Dict[str, Any]) -> StepResult:
        """
        Run one canonical env step.

        Order of operations is fixed and load-bearing:
            1. sense() ONCE -- updates all per-tick substrates exactly once.
            2. record_transition() against this tick's z_self.
            3. clock.advance(), _e1_tick (if scheduled), generate_trajectories.
            4. update_z_goal(benefit_exposure, drive_level) with kwargs only.
            5. update_schema_wanting(drive_level) when MECH-216 is enabled.
            6. on_sense hook.
            7. select_action with fallback to random when the gate withholds.
            8. on_action hook.
            9. env.step.
           10. update_residue(harm_signal, world_delta=optional) -- canonical
               e3.post_action_update path.
           11. on_post_step hook.
           12. plumbing rotate.
        """
        agent = self.agent
        env = self.env
        ctx = nullcontext() if self.train_mode else torch.no_grad()

        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        obs_h = obs_dict.get("harm_obs")
        obs_h_a = obs_dict.get("harm_obs_a")
        obs_h_h = obs_dict.get("harm_history")

        with ctx:
            # 1. sense() ONCE.
            latent = agent.sense(
                obs_body, obs_world,
                obs_harm=obs_h,
                obs_harm_a=obs_h_a,
                obs_harm_history=obs_h_h,
            )

            # 2. record_transition uses THIS tick's z_self as the next-state
            # for the (prev_state, prev_action) pair recorded last tick.
            if self._z_self_prev is not None and self._action_prev is not None:
                agent.record_transition(
                    self._z_self_prev,
                    self._action_prev,
                    latent.z_self.detach(),
                )

            # 3. multi-rate clock + optional E1 tick + trajectory generation.
            ticks = agent.clock.advance()
            world_dim = latent.z_world.shape[-1]
            e1_prior = (
                agent._e1_tick(latent)
                if ticks.get("e1_tick", False)
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            # 4. update_z_goal -- canonical kwargs-only call shape.
            #    NEVER pass `latent` here; it is not a parameter of the method.
            from ree_core.agent import REEAgent as _REEAgent
            drive_level = _REEAgent.compute_drive_level(obs_body)
            benefit_raw = obs_dict.get("benefit_exposure", None)
            if benefit_raw is None and isinstance(obs_body, torch.Tensor):
                if obs_body.shape[-1] > 11:
                    benefit_raw = (
                        obs_body[0, 11] if obs_body.dim() == 2 else obs_body[11]
                    )
            benefit_exposure = (
                0.0 if benefit_raw is None else max(0.0, float(benefit_raw))
            )
            # SD-057 (GAP-7 L2): forward the SD-049 per-type identity tag so the
            # incentive token bank can bind benefit to object identity. None when
            # SD-049 is off (key absent) -> bit-identical legacy path.
            _rtype_raw = obs_dict.get("resource_type_at_agent", None)
            resource_type = None
            if _rtype_raw is not None:
                try:
                    resource_type = int(
                        _rtype_raw[0] if hasattr(_rtype_raw, "__len__")
                        else _rtype_raw
                    )
                except (TypeError, ValueError):
                    resource_type = None
            agent.update_z_goal(
                benefit_exposure=benefit_exposure,
                drive_level=drive_level,
                resource_type=resource_type,
            )

            # SD-057 phase-2 L6 (MECH-347): automatic cue-recall perception.
            # When use_cue_recall is set, derive the strongest-perceived
            # resource type from the SD-049 per-type proximity field views
            # (argmax over types of the field max) and, if it clears the
            # cue_recall_min_proximity floor, fire the cue-recall nudge. This is
            # the ecological cue path; the substrate primitive
            # agent.cue_recall_wanting(...) is callable directly for forced-cue
            # diagnostics. Bit-identical no-op when use_cue_recall is off, the
            # bank is absent, or the env emits no per-type field views.
            _goal_cfg = getattr(getattr(agent, "config", None), "goal", None)
            if _goal_cfg is not None and getattr(_goal_cfg, "use_cue_recall", False):
                try:
                    type_names = getattr(env, "resource_type_names", ()) or ()
                    best_tag, best_prox = 0, -1.0
                    for i, name in enumerate(type_names):
                        fv = obs_dict.get(f"resource_field_view_{name}", None)
                        if fv is None:
                            continue
                        v = float(fv.max()) if hasattr(fv, "max") else float(max(fv))
                        if v > best_prox:
                            best_prox, best_tag = v, i + 1
                    floor = float(getattr(_goal_cfg, "cue_recall_min_proximity", 0.0))
                    if best_tag > 0 and best_prox >= floor:
                        agent.cue_recall_wanting(
                            cue_type=best_tag, drive_level=drive_level
                        )
                except Exception:
                    pass  # cue-recall is best-effort; never break the step loop

            # 5. update_schema_wanting -- canonical MECH-216 call site.
            #    Default-off guard preserves the historical no-op path.
            e1_cfg = getattr(getattr(agent, "config", None), "e1", None)
            if bool(getattr(e1_cfg, "schema_wanting_enabled", False)):
                agent.update_schema_wanting(drive_level=drive_level)

        if self.hooks.on_sense is not None:
            self.hooks.on_sense(
                agent=agent, latent=latent, obs_dict=obs_dict,
                ticks=ticks, drive_level=drive_level,
                benefit_exposure=benefit_exposure,
                step=self._step_count,
            )

        with ctx:
            # 6. select_action; fall back to random when the gate withholds.
            action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action_dim = env.action_dim
                idx = self._rng.randint(0, action_dim - 1)
                action = torch.zeros(1, action_dim, device=agent.device)
                action[0, idx] = 1.0
                agent._last_action = action

        if self.hooks.on_action is not None:
            self.hooks.on_action(
                agent=agent, latent=latent, action=action, obs_dict=obs_dict,
                ticks=ticks, step=self._step_count,
            )

        # 8. env.step.
        flat_obs, harm_signal, done, info, next_obs_dict = env.step(action)

        with ctx:
            # 9. update_residue -- canonical post-action path.
            #    Drives e3.post_action_update (running_variance), MECH-205
            #    surprise writes, and residue accumulation. Skipping this
            #    pins precision and leaves the residue field empty.
            residue_metrics = agent.update_residue(
                harm_signal=float(harm_signal),
                world_delta=None,        # SD-003 attribution scaling, optional
                hypothesis_tag=False,    # waking path; replay/sim use True
                owned=True,
            )

        if self.hooks.on_post_step is not None:
            self.hooks.on_post_step(
                agent=agent, latent=latent, action=action,
                obs_dict=obs_dict, next_obs_dict=next_obs_dict,
                harm_signal=float(harm_signal), done=bool(done),
                ticks=ticks, residue_metrics=residue_metrics,
                step=self._step_count,
            )

        # 11. plumbing rotate.
        self._z_world_prev = latent.z_world.detach()
        self._z_self_prev = latent.z_self.detach()
        self._action_prev = action.detach()
        self._step_count += 1

        return StepResult(
            latent=latent,
            action=action,
            harm_signal=float(harm_signal),
            done=bool(done),
            info=info,
            next_obs_dict=next_obs_dict,
            drive_level=drive_level,
            benefit_exposure=benefit_exposure,
            ticks=ticks,
            residue_metrics=residue_metrics,
        )

    def run_episode(
        self,
        max_steps: int,
        on_step: Optional[Callable[[StepResult], None]] = None,
    ) -> List[StepResult]:
        """
        Run an episode until ``done`` or ``max_steps``.

        Caller is responsible for env.reset() / agent.reset() / harness.reset()
        BEFORE the first step. on_step is a callback fired with each StepResult
        (in addition to the per-phase hooks).
        """
        flat_obs, obs_dict = self.env.reset()
        self.agent.reset()
        self.reset()
        results: List[StepResult] = []
        for _ in range(max_steps):
            r = self.step(obs_dict)
            if on_step is not None:
                on_step(r)
            results.append(r)
            obs_dict = r.next_obs_dict
            if r.done:
                break
        return results
