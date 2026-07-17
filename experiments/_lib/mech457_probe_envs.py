"""MECH-457 GOV-FANOUT-1 discrimination probe env-wrappers (experiment-layer, NOT ree_core).

WHY THIS EXISTS. failure_autopsy_V3-EXQ-769_2026-07-17 EMPIRICALLY FALSIFIED the capacity/
reliability/integration axis of the MECH-457 competence bootstrap-explorer (raising the ON
actor-critic 128->256 + budget 3x->5x + warm-start + z_world detach made raw ON foraging
REGRESS 6.48->0.12). The behavioral signature -- ON survives the full 200 steps / death 0 /
contamination avoided / forages ~0 -- is AVOIDANCE learned WITHOUT APPROACH: on the
D3_hazard_free rung survival is DECOUPLED from foraging, so once the intrinsic drive anneals
there is no gradient toward forage. The autopsy localises the wall to THREE competing axes (a
discrimination, not a build) and routes GOV-FANOUT-1:
  H1 (drive-schedule)   -- the intrinsic drive anneals before approach is established.
  H2 (env reward-coupling) -- D3 is survivable WITHOUT foraging (passive-survival optimum).
  H3 (credit-horizon)   -- the actor-critic cannot assign credit from sparse foraging events.

This module carries the two ENV manipulations the portfolio needs (H1 is a config-only
manipulation of the anneal schedule and needs no wrapper):

  * MetabolicForageWrapper (H2, axis=environment) -- couples agent_energy to survival so
    that NOT foraging is lethal. IMPORTANT (discovered 2026-07-17): on the D3_hazard_free rung
    the passive-survival optimum is NOT "do nothing" -- a stay/random agent DIES within ~40-70
    steps from SELF-CONTAMINATION (footprint_density over contamination_threshold -> the cell
    turns "contaminated" -> contaminated_harm=0.4 drains health, ~3 contaminated contacts kill).
    So "avoidance without approach" = the trained ON arm learns to keep MOVING to fresh cells
    (avoid self-contamination) while ignoring resources. To make survival REQUIRE foraging, the
    H2 metabolic variant does BOTH, together = "recouple survival from contamination-avoidance
    to foraging" (ONE reward-coupling axis, two env edits):
      (a) CLOSE the non-foraging survival route -- set contaminated_harm=0.0 (via
          metabolic_env_kwargs) so self-contamination is no longer lethal; and
      (b) OPEN a foraging survival requirement -- this wrapper's STARVATION coupling: raise the
          per-step energy_decay so energy is depletable in-episode, and once agent_energy is
          depleted drain agent_health each step. Foraging replenishes energy (contact_benefit)
          and resources respawn on consume (D3 default), so a competent forager keeps energy up
          and survives 200 steps; a passive / non-foraging agent starves and dies. Verified
          2026-07-17: oracle/local_view forage ~60 and survive 200 (death 0); stay/random die
          (death 1.0). The passive-survival optimum is thereby dissolved.

  * ForageShapingWrapper (H3, axis=measurement) -- adds a dense potential-based shaping term
    to the extrinsic reward channel (Ng, Harada & Russell 1999): F(s,a,s') = gamma*Phi(s') -
    Phi(s) with Phi(s) = -manhattan_distance_to_nearest_resource. This gives the actor-critic
    a DENSE per-step credit gradient toward resources, removing the sparse-forage credit-
    assignment difficulty, WITHOUT changing the optimal policy (potential-based shaping is
    policy-invariant). Wraps ONLY the TRAINING env; the eval env is left unwrapped so
    foraging_competence (the DV) is measured UNSHAPED.

DESIGN STATUS. Both wrappers are experiment-layer PROBE scaffolds (a discrimination decides
WHETHER an axis matters at all), NOT ree_core substrate features. Per
failure_autopsy_V3-EXQ-769 recommended_substrate_queue_entry.action == "none": do not amend a
substrate entry until the portfolio resolves which axis to build. A WINNING axis routes to
/implement-substrate for the proper ree_core build (a native energy->survival coupling for
H2, or a shaped-reward env option for H3) with contract tests; a NULL axis is discarded.

Both wrappers are transparent delegating proxies: every attribute the training / eval / rep
code reads (agent_x, agent_y, resources, agent_health, agent_energy, action_dim, steps, ...)
delegates to the wrapped env via __getattr__; only step() (and, for symmetry, reset()) are
overridden. Because these modules live under experiments/_lib/**, they are folded into the
arm_fingerprint substrate_hash, so every wrapped cell mints its own reuse-eligible baseline.

Sourced APIs (verified 2026-07-17 against ree_core/environment/causal_grid_world.py):
  env.step(action:int) -> (flat_obs, harm_signal, done, info, obs_dict)
  env.reset()          -> (flat_obs, obs_dict)
  env.agent_x / agent_y / agent_health / agent_energy / resources / action_dim / steps
  done fires on agent_health <= 0.0 (death) or the env's internal step cap (500).
  foraging: info["transition_type"] == "resource"; contact replenishes energy (+benefit*0.5).

ASCII-only in all runtime strings (repo rule).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from experiments._lib.capability_eval import nearest_resource_manhattan

# ---------------------------------------------------------------------------
# H2 metabolic coupling constants (pre-registered; a competent forager clears them easily,
# a non-forager starves before the 200-step episode ends).
# ---------------------------------------------------------------------------
METABOLIC_ENERGY_DECAY = 0.02            # per-step energy drawdown (2x the 0.01 D3 default)
STARVATION_ENERGY_THRESHOLD = 0.0        # health drains only once energy is fully depleted
STARVATION_HEALTH_DRAIN = 0.05           # health lost per step while starving (~20 steps to death)
METABOLIC_CONTAMINATED_HARM = 0.0        # close the contamination-avoidance survival route (D3 default 0.4)


def metabolic_env_kwargs(base_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """H2 metabolic-variant env_kwargs: base D3 kwargs with the contamination-avoidance survival
    route CLOSED (contaminated_harm=0.0). The energy->health starvation coupling is applied
    separately by MetabolicForageWrapper (it also raises energy_decay at construction). Together
    they recouple survival from contamination-avoidance to foraging."""
    kw = dict(base_kwargs)
    kw["contaminated_harm"] = METABOLIC_CONTAMINATED_HARM
    return kw

# ---------------------------------------------------------------------------
# H3 potential-based shaping constants.
# ---------------------------------------------------------------------------
FORAGE_SHAPING_COEF = 1.0                # potential-shaping weight (~ FORAGE_BONUS scale)
FORAGE_SHAPING_GAMMA = 0.99             # discount in F = gamma*Phi(s') - Phi(s) (== AC_GAMMA)


class _EnvProxy:
    """Transparent delegating proxy: all attribute reads fall through to the wrapped env.

    Subclasses override step() (and may override reset()); everything else -- agent_x,
    agent_y, resources, agent_health, agent_energy, action_dim, steps, render, ... -- is
    served from the wrapped CausalGridWorldV2 by __getattr__. `_env` is a real instance
    attribute set in __init__, so __getattr__ (which fires only for MISSING attributes) never
    recurses on it.
    """

    def __init__(self, env: Any) -> None:
        self._env = env

    def __getattr__(self, name: str) -> Any:
        # Only reached when `name` is not a real attribute of the proxy instance.
        return getattr(self._env, name)

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        return self._env.reset()

    def step(self, action: int) -> Tuple[Any, float, bool, Dict[str, Any], Dict[str, Any]]:
        raise NotImplementedError


class MetabolicForageWrapper(_EnvProxy):
    """H2 (axis=environment): couple agent_energy to survival so NOT foraging is lethal.

    After the inner env.step (which already decays energy and replenishes it on a forage
    contact), if agent_energy is at / below `energy_threshold`, drain agent_health by
    `health_drain` and OR that death into `done`. Resources respawn on consume (D3 default),
    so a competent forager keeps energy up and survives; a passive / non-foraging agent
    starves before the episode ends -- dissolving the passive-survival optimum. The wrapper
    also sets a higher per-step `energy_decay` on the wrapped env at construction so the
    pressure bites within the 200-step episode. Set BOTH train and eval env to this wrapper
    (foraging_competence for H2 is measured under the same metabolic dynamics; the readiness
    anchors must clear the floor on THIS env or the run self-routes substrate_not_ready).
    """

    def __init__(
        self,
        env: Any,
        energy_decay: float = METABOLIC_ENERGY_DECAY,
        energy_threshold: float = STARVATION_ENERGY_THRESHOLD,
        health_drain: float = STARVATION_HEALTH_DRAIN,
    ) -> None:
        super().__init__(env)
        # Raise the wrapped env's native energy drawdown so starvation is reachable in-episode.
        self._env.energy_decay = float(energy_decay)
        self._energy_threshold = float(energy_threshold)
        self._health_drain = float(health_drain)
        self._starved_steps = 0

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        self._starved_steps = 0
        return self._env.reset()

    def step(self, action: int) -> Tuple[Any, float, bool, Dict[str, Any], Dict[str, Any]]:
        flat, harm_signal, done, info, obs_dict = self._env.step(action)
        energy = float(getattr(self._env, "agent_energy", 1.0))
        if energy <= self._energy_threshold:
            self._starved_steps += 1
            new_health = max(0.0, float(getattr(self._env, "agent_health", 1.0)) - self._health_drain)
            self._env.agent_health = new_health
            if new_health <= 0.0:
                done = True
        if isinstance(info, dict):
            info = dict(info)
            info["metabolic_starving"] = bool(energy <= self._energy_threshold)
            info["metabolic_starved_steps"] = int(self._starved_steps)
        return flat, harm_signal, done, info, obs_dict


class ForageShapingWrapper(_EnvProxy):
    """H3 (axis=measurement): add dense potential-based forage shaping to the reward channel.

    Adds F(s,a,s') = gamma*Phi(s') - Phi(s), Phi(s) = -manhattan_to_nearest_resource, to the
    returned harm_signal (the extrinsic reward the trainer reads). Potential-based shaping is
    policy-invariant (Ng et al. 1999), so it removes credit-assignment difficulty without
    changing the optimal policy. Wrap ONLY the TRAINING env; leave the eval env unwrapped so
    foraging_competence is measured UNSHAPED. Phi is computed from the wrapped env's state
    BEFORE and AFTER the inner step; when no resource exists Phi = 0.
    """

    def __init__(
        self,
        env: Any,
        shaping_coef: float = FORAGE_SHAPING_COEF,
        gamma: float = FORAGE_SHAPING_GAMMA,
    ) -> None:
        super().__init__(env)
        self._coef = float(shaping_coef)
        self._gamma = float(gamma)

    def _potential(self) -> float:
        d = nearest_resource_manhattan(self._env)
        return 0.0 if d is None else -float(d)

    def step(self, action: int) -> Tuple[Any, float, bool, Dict[str, Any], Dict[str, Any]]:
        phi_before = self._potential()
        flat, harm_signal, done, info, obs_dict = self._env.step(action)
        phi_after = self._potential()
        shaping = self._coef * (self._gamma * phi_after - phi_before)
        harm_signal = float(harm_signal) + float(shaping)
        if isinstance(info, dict):
            info = dict(info)
            info["forage_shaping_delta"] = round(float(shaping), 6)
        return flat, harm_signal, done, info, obs_dict


def config_slice_extra() -> Dict[str, Any]:
    """Probe-env constants for the arm_fingerprint config_slice / manifest (declared)."""
    return {
        "metabolic_energy_decay": METABOLIC_ENERGY_DECAY,
        "metabolic_energy_threshold": STARVATION_ENERGY_THRESHOLD,
        "metabolic_health_drain": STARVATION_HEALTH_DRAIN,
        "metabolic_contaminated_harm": METABOLIC_CONTAMINATED_HARM,
        "forage_shaping_coef": FORAGE_SHAPING_COEF,
        "forage_shaping_gamma": FORAGE_SHAPING_GAMMA,
    }
