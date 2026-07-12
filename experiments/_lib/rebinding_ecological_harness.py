"""
rebinding_ecological_harness.py -- shared MECH-456 ECOLOGICAL-rebinding harness
(DESIGN + DV diversity for the V3-EXQ-745 patch-set-flip leg).

WHAT THIS IS
------------
The two prior MECH-456 PASSes (V3-EXQ-733b / 733c) rest on ONE teleport-based
directed-traversal harness (rebinding_functional_harness.py) measuring a
POLICY-INERT affinity readout (binder.binding_score, used only for DV1). This
module adds the missing DESIGN + DV diversity along two independent axes so the
claim can move provisional -> active on more than a single test-bed monoculture:

  (a) TEST-BED axis -- PatchFlipController: a WORLD-DRIVEN productive-patch-set
      flip. No teleport. The world's foragable resource configuration is one of
      N distinct spatial clusters; every flip_period steps a COMPETING config
      OVERTAKES the current one (the old cluster is cleared, a new cluster is
      placed). The ground-truth world-config g(t) is the flip schedule index --
      binder-INDEPENDENT, read off the controller, not off anything the binder
      produces. Between flips the current cluster is kept foragable (restock).

  (b) DV axis -- an ECOLOGICAL behavioural DV riding the POLICY-COUPLED binder
      path. binder.couple() (the ONLY policy-coupled binder path; fires inside
      e2.rollout_with_world during CEM planning) injects the binding vector into
      the rollout world-states the planner scores, steering the committed action.
      DV2 compares foraging competence of a live-rebinding arm (ree_on) against a
      FROZEN-factor arm (ree_frozen) that binds ONCE at episode entry and never
      re-evaluates. DV1 is the (policy-inert) affinity readout kept for continuity
      with the 733 harness: does argmax binding_score track the true productive
      config above a config-label-shuffle control.

FROZEN arm mechanism (FrozenFactor). We wrap binder.factor: it computes the real
factor from the episode-entry latent ONCE and returns that same g for every CEM
candidate for the rest of the episode. binder.binding_score (DV1 readout) is left
untouched -- freezing factor() changes BEHAVIOUR (couple path) but not the affinity
readout, which is the point: DV2 is the behavioural consequence, DV1 the readout.

NO ree_core change -- this is harness-level measurement + a harness-level
factor-freeze monkeypatch over a trained-then-frozen learned cross_stream_binder.

Pre-registered constants (FIXED; NEVER derived from a run):
  ALIGN_MARGIN     -- reused from rebinding_functional_harness (=0.10). DV1: real
                      alignment must beat the config-label-shuffle baseline by this.
  MIN_OVERTAKES    -- 8. Per-seed floor of within-episode ground-truth config
                      changes; world-GUARANTEED by the flip schedule.
  DV2_ABS_MARGIN   -- 0.15 resources/episode. A stale-binding forager must collect
                      at least this many FEWER resources/episode than live rebinding
                      for DV2 to count on a seed. Pre-registered effect-size floor.
  MIN_SEEDS_FOR_PASS / MIN_SEEDS_COMPLETED -- reused (=4 / =4).
  COMPETENCE_RESOURCE_FLOOR -- reused (=1.0). Oracle must clear it (achievability).

ASCII-only in all runtime strings (repo rule).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from experiments._lib import rebinding_functional_harness as H
from experiments._lib.capability_eval import (
    COMPETENCE_RESOURCE_FLOOR,
    OraclePolicy,
    RandomPolicy,
    Policy,
)

# ---------------------------------------------------------------------------
# Pre-registered acceptance constants (FIXED -- never derived from a run).
# ---------------------------------------------------------------------------
ALIGN_MARGIN = H.ALIGN_MARGIN            # DV1 alignment margin (reuse 733).
MIN_OVERTAKES = 8                        # per-seed within-episode config-change floor.
DV2_ABS_MARGIN = 0.15                    # DV2 resources/episode effect-size floor.
COUPLE_AUTHORITY_FLOOR = 0.15            # readiness positive-control: disabling the
                                         # binding (strength=0) must change foraging by
                                         # >= this many resources/episode, else the
                                         # couple() path is behaviourally INERT and a
                                         # null ON-vs-FROZEN DV2 is UNinterpretable
                                         # (substrate_ceiling / non_contributory), NOT
                                         # an inert-rebinding WEAKENS. Guards the
                                         # V3-EXQ-478 false-weakens hazard the smoke
                                         # surfaced (couple too weak to move foraging).
MIN_SEEDS_FOR_PASS = H.MIN_SEEDS_FOR_PASS
MIN_SEEDS_COMPLETED = H.MIN_SEEDS_COMPLETED


# ===========================================================================
# (A) PatchFlipController -- world-driven productive-config flip test-bed.
# ===========================================================================
class PatchFlipController:
    """World-side competing-configuration overtake generator (no teleport).

    The world exposes ONE of `n_configs` distinct foragable resource clusters at a
    time. `install` places config 0. Every `flip_period` steps `tick` clears the
    current cluster and places the next -- a genuine competing-config OVERTAKE that
    is a pure function of the flip schedule (binder-independent ground truth
    g(t) = config_of()). Between flips `tick` restocks the current cluster so it
    stays continuously foragable. All state writes are pure grid/resource writes
    (mirroring the 733 teleport's no-env.step discipline); nothing calls env.step.

    Determinism: cluster jitter is drawn from np.random.RandomState(seed) at
    __init__ ONLY -- no global RNG, no wallclock.
    """

    def __init__(self, size: int, n_configs: int, resources_per_config: int,
                 flip_period: int, seed: int) -> None:
        self.size = int(size)
        self.n_configs = int(n_configs)
        self.resources_per_config = int(resources_per_config)
        self.flip_period = int(flip_period)
        self.seed = int(seed)
        self.current = 0
        rng = np.random.RandomState(self.seed)

        # Config cluster CENTERS: spread on a near-square lattice of centers across
        # the interior, well-separated so the resource layouts are distinguishable.
        g = int(np.ceil(np.sqrt(max(1, self.n_configs))))
        lo, hi = 1, self.size - 2  # interior band (non-wall on a non-toroidal grid)
        centers: List[Tuple[int, int]] = []
        for idx in range(self.n_configs):
            rx, ry = divmod(idx, g)
            cx = int(round(lo + (rx + 0.5) * (hi - lo) / g))
            cy = int(round(lo + (ry + 0.5) * (hi - lo) / g))
            # small deterministic jitter so different seeds get distinct layouts
            cx += int(rng.randint(-1, 2))
            cy += int(rng.randint(-1, 2))
            cx = min(hi, max(lo, cx))
            cy = min(hi, max(lo, cy))
            centers.append((cx, cy))
        self.centers = centers

        # Each config's cluster = the first resources_per_config interior cells on
        # an outward ring order around the (jittered) center, de-duplicated.
        ring = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1),
                (2, 0), (-2, 0), (0, 2), (0, -2)]
        self.clusters: List[List[Tuple[int, int]]] = []
        for (cx, cy) in centers:
            cells: List[Tuple[int, int]] = []
            for (dx, dy) in ring:
                x = min(hi, max(lo, cx + dx))
                y = min(hi, max(lo, cy + dy))
                if (x, y) not in cells:
                    cells.append((x, y))
                if len(cells) >= self.resources_per_config:
                    break
            self.clusters.append(cells[: self.resources_per_config])

    # -- internal grid helpers ------------------------------------------------
    def _agent_cell(self, env) -> Tuple[int, int]:
        return int(env.agent_x), int(env.agent_y)

    def _clear_all(self, env) -> None:
        """Clear EVERY currently-placed resource (grid + env.resources)."""
        empty = env.ENTITY_TYPES["empty"]
        ax, ay = self._agent_cell(env)
        for cell in list(getattr(env, "resources", []) or []):
            x, y = int(cell[0]), int(cell[1])
            if (x, y) == (ax, ay):
                continue  # never clobber the agent marker
            if 0 <= x < env.size and 0 <= y < env.size:
                env.grid[x, y] = empty
        env.resources = []

    def _place(self, env, cfg: int) -> None:
        """Clear all resources, then place config `cfg`'s cluster (the overtake)."""
        resource_t = env.ENTITY_TYPES["resource"]
        wall_t = env.ENTITY_TYPES["wall"]
        ax, ay = self._agent_cell(env)
        self._clear_all(env)
        for (x, y) in self.clusters[cfg]:
            if (x, y) == (ax, ay):
                continue
            if env.grid[x, y] == wall_t:
                continue
            env.grid[x, y] = resource_t
            env.resources.append([int(x), int(y)])

    def restock(self, env) -> None:
        """Top the CURRENT cluster back up to full (re-add consumed/cleared cells)."""
        resource_t = env.ENTITY_TYPES["resource"]
        wall_t = env.ENTITY_TYPES["wall"]
        empty_t = env.ENTITY_TYPES["empty"]
        ax, ay = self._agent_cell(env)
        present = {(int(c[0]), int(c[1])) for c in (getattr(env, "resources", []) or [])}
        for (x, y) in self.clusters[self.current]:
            if (x, y) in present:
                continue
            if (x, y) == (ax, ay):
                continue
            if env.grid[x, y] == wall_t:
                continue
            if env.grid[x, y] == empty_t:
                env.grid[x, y] = resource_t
                env.resources.append([int(x), int(y)])

    # -- public API -----------------------------------------------------------
    def install(self, env) -> None:
        """Called after env.reset(): reset to config 0 and place it."""
        self.current = 0
        self._place(env, 0)

    def config_of(self) -> int:
        """Binder-INDEPENDENT ground-truth world-config index g(t)."""
        return int(self.current)

    def tick(self, env, step_idx: int) -> bool:
        """Called after each env.step. Returns True iff a config OVERTAKE happened."""
        if (int(step_idx) + 1) % self.flip_period == 0:
            self.current = (self.current + 1) % self.n_configs
            self._place(env, self.current)
            return True
        self.restock(env)
        return False


# ===========================================================================
# (B) FrozenFactor -- FROZEN-arm manipulation (harness-level wrap of factor).
# ===========================================================================
class FrozenFactor:
    """Freeze binder.factor to the EPISODE-ENTRY binding.

    On the first factor() call of an episode, compute the real factor from the
    batch-1 episode-entry latent and cache it; every subsequent call returns that
    same g expanded to the CEM candidate batch. So couple() injects a fixed
    (never-re-evaluated) binding into the rollout -- "bind once, never rebind".
    binder.binding_score (the DV1 readout) is left untouched.
    """

    def __init__(self, binder) -> None:
        self.binder = binder
        self._orig = binder.factor        # bound method captured before install
        self._g: Optional[torch.Tensor] = None

    def install(self) -> None:
        self.binder.factor = self.__call__  # instance attr shadows the class method

    def restore(self) -> None:
        try:
            del self.binder.factor          # remove instance attr -> class method returns
        except AttributeError:
            self.binder.factor = self._orig

    def reset_episode(self) -> None:
        self._g = None

    def __call__(self, z_self: torch.Tensor, z_world: torch.Tensor) -> torch.Tensor:
        if self._g is None:
            self._g = self._orig(z_self[:1], z_world[:1]).detach()
        return self._g.expand(z_self.shape[0], -1)


# ===========================================================================
# (C) Decision primitive + per-seed pipeline functions.
# ===========================================================================
def _sense(agent, obs_dict):
    """Replicate rebinding_functional_harness.step_decision's sense block. Returns
    the LatentState (caller reads latent.z_self / latent.z_world)."""
    body = obs_dict["body_state"].float()
    world = obs_dict["world_state"].float()
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    return agent.sense(
        obs_body=body, obs_world=world,
        obs_harm=H.obs_harm(obs_dict),
        obs_harm_a=H.obs_harm_a(obs_dict),
        obs_harm_history=H.obs_harm_history(obs_dict),
    )


def _decide(agent, obs_dict, zself_prev, act_prev, train_binder: bool):
    """One natural main-path decision (sense -> [record] -> [binder curriculum] ->
    generate -> select_action). Mirrors step_decision but exposes z_self/z_world and
    switches the binder curriculum on `train_binder` (P0 True; measurement False)."""
    latent = _sense(agent, obs_dict)
    z_self_now = latent.z_self.detach()
    z_world_now = latent.z_world.detach()
    if zself_prev is not None and act_prev is not None:
        agent.record_transition(zself_prev, act_prev, z_self_now)
    if train_binder:
        agent.update_cross_stream_binder(latent.z_self, latent.z_world)
    ticks = agent.clock.advance()
    wdim = latent.z_world.shape[-1]
    e1_prior = (
        agent._e1_tick(latent) if ticks.get("e1_tick", False)
        else torch.zeros(1, wdim, device=agent.device)
    )
    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
    action = agent.select_action(candidates, ticks)
    return action, z_self_now, z_world_now


def _update_goal(agent, info, obs_dict) -> None:
    if agent.goal_state is not None:
        be = float(info.get("benefit_exposure", 0.0)) if isinstance(info, dict) else 0.0
        en = float(obs_dict["body_state"].float().reshape(-1)[3].item())
        agent.update_z_goal(benefit_exposure=be, drive_level=max(0.0, 1.0 - en))


def train_binder_p0(agent, env_factory: Callable[[int], Any], flip_kwargs: Dict[str, Any],
                    seed: int, p0_episodes: int, steps_per_episode: int):
    """P0 binder curriculum on the patch-flip test-bed. Accumulates per-config
    z_world prototypes (mean-pooled; empty config -> global mean) and config-visit
    counts. Returns (protos, proto_cnt, config_visits, agent)."""
    agent.eval()
    env = env_factory(seed)
    flip = PatchFlipController(seed=seed, **flip_kwargs)
    n_configs = int(flip.n_configs)

    world_dim = int(agent.config.latent.world_dim)
    proto_sum = [torch.zeros(1, world_dim) for _ in range(n_configs)]
    proto_cnt = [0 for _ in range(n_configs)]
    global_sum = torch.zeros(1, world_dim)
    global_cnt = 0
    config_visits = [0 for _ in range(n_configs)]

    for ep in range(int(p0_episodes)):
        _, obs = env.reset()
        agent.reset()
        flip.install(env)
        zp = ap = None
        for step_idx in range(int(steps_per_episode)):
            cfg = flip.config_of()
            action, z_self_now, z_world_now = _decide(agent, obs, zp, ap, train_binder=True)
            if not torch.isfinite(action).all():
                break
            zw = z_world_now.reshape(1, -1)
            proto_sum[cfg] = proto_sum[cfg] + zw
            proto_cnt[cfg] += 1
            config_visits[cfg] += 1
            global_sum = global_sum + zw
            global_cnt += 1

            _, _hs, done, info, obs = env.step(action)
            _update_goal(agent, info, obs)
            zp, ap = z_self_now, action.detach()
            if done:
                break
            flip.tick(env, step_idx)
            # Refresh obs so the NEXT decision senses the post-flip/restock grid
            # (a pure read; mirrors the 733 teleport's _get_observation_dict()
            # refresh). Without this the obs from env.step lags the flip by one
            # step -> z_world mislabeled against the new config for ~1/flip_period
            # of steps.
            obs = env._get_observation_dict()
        if (ep + 1) % 10 == 0 or ep == int(p0_episodes) - 1:
            print(
                f"  [train] seed={seed} ep {ep + 1}/{int(p0_episodes)} "
                f"configs_covered={sum(1 for c in proto_cnt if c > 0)}/{n_configs}",
                flush=True,
            )

    gmean = (global_sum / global_cnt) if global_cnt > 0 else torch.zeros(1, world_dim)
    protos = [
        (proto_sum[k] / proto_cnt[k]) if proto_cnt[k] > 0 else gmean.clone()
        for k in range(n_configs)
    ]
    return protos, proto_cnt, config_visits, agent


def measure_arm(agent, env_factory: Callable[[int], Any], flip_kwargs: Dict[str, Any],
                seed: int, m_episodes: int, steps_per_episode: int,
                protos: List[torch.Tensor], arm_name: str, frozen: bool,
                record_dv1: bool = False, couple_off: bool = False) -> Dict[str, Any]:
    """Measure foraging_competence + survival_horizon of the FROZEN-binder agent on
    the patch-flip test-bed under one of three couple() modes:
      - ON     (frozen=False, couple_off=False): couple() fires LIVE (re-evaluated each step).
      - FROZEN (frozen=True): FrozenFactor replaces binder.factor -> couple() injects the
        episode-entry binding for the whole episode ("bind once, never rebind").
      - NOCOUPLE (couple_off=True): the couple() perturbation is DISABLED (binder.strength
        set to 0.0 -> couple() returns the streams unchanged). The positive-control arm:
        it measures the agent with NO binding influence, so |foraging_on - foraging_nocouple|
        is the couple() path's behavioural authority. If ~0 the couple() path is inert and a
        null ON-vs-FROZEN DV2 is UNinterpretable (not an inert-rebinding weakens).
    The binder is NOT trained here (weights frozen). For the ON arm (record_dv1=True) also
    record the DV1 affinity streams. `frozen` and `couple_off` are mutually exclusive."""
    if frozen and couple_off:
        raise ValueError("measure_arm: frozen and couple_off are mutually exclusive")
    agent.eval()
    binder = agent.cross_stream_binder
    error_note: Optional[str] = None
    if binder is None:
        return {
            "arm_name": arm_name, "error_note": f"binder absent seed={seed} (substrate OFF)",
            "foraging_competence": 0.0, "survival_horizon": 0.0,
            "b_on": [], "gts": [], "ep_ids": [], "n_flips": 0,
        }

    frozen_ctx: Optional[FrozenFactor] = None
    if frozen:
        frozen_ctx = FrozenFactor(binder)
        frozen_ctx.install()
    # NOCOUPLE positive control: disable the coupling perturbation entirely. couple()
    # short-circuits to identity when k_t == 0.0 (strength * theta_gate == 0), so setting
    # strength=0.0 removes ALL binding influence on the policy. Restored in finally.
    _saved_strength: Optional[float] = None
    if couple_off:
        _saved_strength = float(getattr(binder, "strength", 0.0))
        binder.strength = 0.0

    resources_per_ep: List[int] = []
    ticks_per_ep: List[int] = []
    b_on: List[int] = []
    gts: List[int] = []
    ep_ids: List[int] = []
    n_flips = 0

    try:
        env = env_factory(seed)
        flip = PatchFlipController(seed=seed, **flip_kwargs)
        for ep in range(int(m_episodes)):
            _, obs = env.reset()
            agent.reset()
            flip.install(env)
            if frozen_ctx is not None:
                frozen_ctx.reset_episode()
            zp = ap = None
            ep_resources = 0
            ep_ticks = 0
            for step_idx in range(int(steps_per_episode)):
                cfg = flip.config_of()
                action, z_self_now, z_world_now = _decide(
                    agent, obs, zp, ap, train_binder=False)
                # DV1 readout (policy-inert affinity argmax; ON arm only).
                if record_dv1:
                    anchor = z_self_now.reshape(1, -1)
                    idx, _scores = H.argmax_binding(binder, anchor, protos)
                    b_on.append(int(idx))
                    gts.append(int(cfg))
                    ep_ids.append(int(ep))
                if not torch.isfinite(action).all():
                    error_note = f"non-finite action seed={seed} arm={arm_name} ep={ep}"
                    break

                _, _hs, done, info, obs = env.step(action)
                ep_ticks += 1
                if isinstance(info, dict) and str(info.get("transition_type", "")) == "resource":
                    ep_resources += 1
                _update_goal(agent, info, obs)
                zp, ap = z_self_now, action.detach()
                if done:
                    break
                if flip.tick(env, step_idx):
                    n_flips += 1
                # Refresh obs to the post-flip/restock grid (see train_binder_p0).
                obs = env._get_observation_dict()
            resources_per_ep.append(ep_resources)
            ticks_per_ep.append(ep_ticks)
            if error_note is not None:
                break
            if (ep + 1) % 5 == 0 or ep == int(m_episodes) - 1:
                print(
                    f"  [measure] seed={seed} arm={arm_name} ep {ep + 1}/{int(m_episodes)} "
                    f"resources={ep_resources} ticks={ep_ticks}",
                    flush=True,
                )
    finally:
        if frozen_ctx is not None:
            frozen_ctx.restore()
        if _saved_strength is not None:
            binder.strength = _saved_strength

    n = len(resources_per_ep)
    foraging = float(sum(resources_per_ep) / n) if n else 0.0
    survival = float(sum(ticks_per_ep) / n) if n else 0.0
    return {
        "arm_name": arm_name,
        "error_note": error_note,
        "foraging_competence": foraging,
        "survival_horizon": survival,
        "b_on": b_on,
        "gts": gts,
        "ep_ids": ep_ids,
        "n_flips": int(n_flips),
    }


def calib_arm(policy_factory: Callable[[], Policy], env_factory: Callable[[int], Any],
              flip_kwargs: Dict[str, Any], seed: int, m_episodes: int,
              steps_per_episode: int) -> Dict[str, Any]:
    """Run a calibration policy (Oracle / Random) over the SAME flip schedule (flip
    controller active, no binder). Oracle reads env.resources, which the controller
    keeps populated -> a valid achievability ceiling on THIS test-bed."""
    policy = policy_factory()
    resources_per_ep: List[int] = []
    ticks_per_ep: List[int] = []
    env = env_factory(seed)
    flip = PatchFlipController(seed=seed, **flip_kwargs)
    for ep in range(int(m_episodes)):
        _, obs = env.reset()
        policy.reset(env)
        flip.install(env)
        ep_resources = 0
        ep_ticks = 0
        for step_idx in range(int(steps_per_episode)):
            act = policy.act(env, obs)
            _, _hs, done, info, obs = env.step(int(act))
            ep_ticks += 1
            if isinstance(info, dict) and str(info.get("transition_type", "")) == "resource":
                ep_resources += 1
            policy.post_step(env, info, obs)
            if done:
                break
            flip.tick(env, step_idx)
        resources_per_ep.append(ep_resources)
        ticks_per_ep.append(ep_ticks)
    n = len(resources_per_ep)
    return {
        "policy": policy.name,
        "foraging_competence": float(sum(resources_per_ep) / n) if n else 0.0,
        "survival_horizon": float(sum(ticks_per_ep) / n) if n else 0.0,
    }


def compute_dv1(b_on: List[int], gts: List[int], ep_ids: List[int],
                k_regions: int, gen: torch.Generator) -> Dict[str, Any]:
    """DV1 over the ON-arm affinity stream: real config-tracking alignment vs a
    config-label-shuffle baseline. Also counts within-episode ground-truth
    config-change overtakes (the seed's n_overtakes)."""
    n = len(gts)
    a_real = (sum(1 for t in range(n) if b_on[t] == gts[t]) / n) if n else 0.0
    a_shuf = H.label_shuffle_alignment(b_on, gts, k_regions, gen) if n else 0.0
    dv1 = bool(a_real >= a_shuf + ALIGN_MARGIN)
    n_overtakes = sum(
        1 for t in range(1, n)
        if ep_ids[t] == ep_ids[t - 1] and gts[t] != gts[t - 1]
    )
    return {
        "alignment_real": float(a_real),
        "alignment_shuffle": float(a_shuf),
        "alignment_margin_obs": float(a_real - a_shuf),
        "dv1": dv1,
        "n_overtakes": int(n_overtakes),
        "n_dv1_steps": int(n),
    }


# ===========================================================================
# (D) Interpretation + evidence-direction mapping.
# ===========================================================================
def _seed_ready(row: Dict[str, Any]) -> bool:
    return bool(
        row.get("binder_converged", False)
        and row.get("all_configs_visited", False)
        and int(row.get("n_overtakes", 0)) >= MIN_OVERTAKES
        and float(row.get("foraging_oracle", 0.0)) >= COMPETENCE_RESOURCE_FLOOR
        # positive control: the couple() path must have behavioural authority
        # (disabling the binding changes foraging), else DV2 is uninterpretable.
        and row.get("couple_authority", False)
    )


def _dv2_seed(row: Dict[str, Any], dv2_abs_margin: float) -> bool:
    gap = float(row.get("foraging_on", 0.0)) - float(row.get("foraging_frozen", 0.0))
    return bool(gap >= dv2_abs_margin and row.get("foraging_on", 0.0) > row.get("foraging_frozen", 0.0))


def interpret_ecological(per_seed_rows: List[Dict[str, Any]], dv2_abs_margin: float,
                         conv_frac: float) -> Dict[str, Any]:
    ok = [r for r in per_seed_rows if r.get("error_note") is None]

    all_ready = bool(ok) and all(_seed_ready(r) for r in ok)
    n_dv1 = sum(1 for r in ok if r.get("dv1", False))
    dv2_flags = [_dv2_seed(r, dv2_abs_margin) for r in ok]
    n_dv2 = sum(1 for f in dv2_flags if f)
    dv1_pass = bool(n_dv1 >= MIN_SEEDS_FOR_PASS)
    dv2_pass = bool(n_dv2 >= MIN_SEEDS_FOR_PASS)

    # non-saturation: on the DV2-passing seeds, live rebinding must not be pinned at
    # the achievable ceiling (foraging_on < foraging_oracle) -- else the gap is an
    # artefact of a frozen arm floored rather than a live arm doing real work.
    dv2_pass_rows = [r for r, f in zip(ok, dv2_flags) if f]
    dv2_non_saturating = bool(
        dv2_pass_rows and all(
            float(r.get("foraging_on", 0.0)) < float(r.get("foraging_oracle", 0.0))
            for r in dv2_pass_rows
        )
    )

    if not all_ready:
        label = "substrate_not_ready_requeue"
    elif dv1_pass and dv2_pass:
        label = "ecological_rebinding_supported"
    elif dv1_pass and not dv2_pass:
        label = "rebinding_inert_off_equals_on"
    elif not dv1_pass:
        label = "rebinding_not_tracking_truth"
    else:
        label = "rebinding_inert_off_equals_on"

    # readiness convergence stat (mirror functional harness: worst loss_ema vs
    # conv_frac * worst chance_floor; upper-bound direction).
    emas = [r.get("binder_loss_ema") for r in ok if r.get("binder_loss_ema") is not None]
    chances = [r.get("binder_chance_floor") for r in ok if r.get("binder_chance_floor") is not None]
    worst_ema = float(max(emas)) if emas else None
    worst_chance = float(min(chances)) if chances else None
    conv_threshold = (float(conv_frac) * worst_chance) if worst_chance is not None else None

    min_overtakes = min((int(r.get("n_overtakes", 0)) for r in ok), default=0)
    n_configs = int(ok[0].get("n_configs", 0)) if ok else 0
    min_configs_covered = min((int(r.get("n_configs_covered", 0)) for r in ok), default=0)
    min_oracle = min((float(r.get("foraging_oracle", 0.0)) for r in ok), default=0.0)
    min_couple_authority = min(
        (float(r.get("couple_authority_obs", 0.0)) for r in ok), default=0.0)

    preconditions = [
        {
            "name": "learned_binder_converged",
            "kind": "readiness",
            "description": (
                "Every completed seed's learned binder must converge (worst loss_ema "
                "<= conv_frac * chance_floor); an unconverged binder is the 725 "
                "untrained-substrate artifact, not a rebinding verdict."
            ),
            "measured": (worst_ema if worst_ema is not None else 1e9),
            "threshold": (conv_threshold if conv_threshold is not None else 0.0),
            "direction": "upper",
            "control": "binder_converged on the trained-then-frozen P0 binder",
            "met": bool(ok and all(r.get("binder_converged", False) for r in ok)),
        },
        {
            "name": "config_coverage_adequate",
            "kind": "readiness",
            "description": (
                "Every productive config must be visited in P0 across all completed "
                "seeds so each config's prototype is mean-poolable; a missing config "
                "-> ill-defined prototype -> uninterpretable DV1."
            ),
            "measured": int(min_configs_covered),
            "threshold": int(n_configs),
            "direction": "lower",
            "control": "per-config P0 visit counts",
            "met": bool(ok and all(r.get("all_configs_visited", False) for r in ok)),
        },
        {
            "name": "overtake_events_adequate",
            "kind": "readiness",
            "description": (
                "min within-episode ground-truth config-change (overtake) count in the "
                "ON stream clears the floor -- world-GUARANTEED by the flip schedule."
            ),
            "measured": int(min_overtakes),
            "threshold": int(MIN_OVERTAKES),
            "direction": "lower",
            "control": "per-seed within-episode config-change count",
            "met": bool(ok and min_overtakes >= MIN_OVERTAKES),
        },
        {
            "name": "oracle_clears_floor",
            "kind": "readiness",
            "description": (
                "The greedy oracle must clear the competence floor on THIS test-bed "
                "(achievability); if even the oracle cannot forage the flip clusters, "
                "no agent foraging conclusion (DV2) is licensed."
            ),
            "measured": float(min_oracle),
            "threshold": float(COMPETENCE_RESOURCE_FLOOR),
            "direction": "lower",
            "control": "greedy_oracle foraging_competence over the flip schedule",
            "met": bool(ok and min_oracle >= COMPETENCE_RESOURCE_FLOOR),
        },
        {
            "name": "couple_path_has_behavioural_authority",
            "kind": "readiness",
            "description": (
                "POSITIVE CONTROL for the DV2 measurement: disabling the binding "
                "coupling (binder.strength=0 -> couple() identity) must change foraging "
                "by >= COUPLE_AUTHORITY_FLOOR resources/episode "
                "(min |foraging_on - foraging_nocouple| across seeds). If the couple() "
                "path cannot move foraging AT ALL, a null ON-vs-FROZEN DV2 is "
                "UNinterpretable (the binding is behaviourally inert on this substrate, "
                "the V3-EXQ-478 false-weakens hazard) -> substrate_not_ready_requeue / "
                "non_contributory, NOT an inert-rebinding weakens. This is the same "
                "load-bearing-statistic readiness gate the skill mandates: the control "
                "measures the SAME quantity (foraging delta via couple) that DV2 routes on."
            ),
            "measured": float(min_couple_authority),
            "threshold": float(COUPLE_AUTHORITY_FLOOR),
            "direction": "lower",
            "control": "|foraging_on - foraging_nocouple| (couple disabled via strength=0)",
            "met": bool(ok and all(r.get("couple_authority", False) for r in ok)),
        },
    ]

    criteria = [
        {"name": "learned_binder_converged", "load_bearing": True,
         "passed": bool(ok and all(r.get("binder_converged", False) for r in ok))},
        {"name": "DV1_tracks_true_config_above_shuffle", "load_bearing": True,
         "passed": dv1_pass},
        {"name": "DV2_ecological_foraging_consequence", "load_bearing": True,
         "passed": dv2_pass},
        {"name": "DV2_non_saturating", "load_bearing": False,
         "passed": dv2_non_saturating},
    ]

    return {
        "label": label,
        "preconditions": preconditions,
        "criteria": criteria,
        "all_ready": all_ready,
        "n_completed": len(ok),
        "n_DV1": int(n_dv1),
        "n_DV2": int(n_dv2),
        "dv1_pass": dv1_pass,
        "dv2_pass": dv2_pass,
        "dv2_non_saturating": dv2_non_saturating,
        "min_seeds_for_pass": int(MIN_SEEDS_FOR_PASS),
        "min_overtakes_obs": int(min_overtakes),
        "min_overtakes_floor": int(MIN_OVERTAKES),
        "dv2_abs_margin": float(dv2_abs_margin),
        "min_couple_authority_obs": float(min_couple_authority),
        "couple_authority_floor": float(COUPLE_AUTHORITY_FLOOR),
        "worst_binder_loss_ema": worst_ema,
        "binder_conv_threshold": conv_threshold,
    }


def evidence_direction_ecological(label: str) -> Tuple[str, str]:
    """Map the ecological interpretation label to (evidence_direction, note)."""
    if label == "ecological_rebinding_supported":
        return "supports", (
            "DV1 (argmax binding_score tracks the TRUE productive config above the "
            "config-label-shuffle control) AND DV2 (live rebinding forages >= "
            "DV2_ABS_MARGIN more resources/episode than a factor-FROZEN arm, "
            "non-saturating below the oracle ceiling) both clear on >= "
            "MIN_SEEDS_FOR_PASS seeds on an INDEPENDENT world-driven patch-flip "
            "test-bed via the POLICY-COUPLED couple() path -> ecological rebinding "
            "demonstrated; SUPPORTS MECH-456 (design + DV diversity beyond the 733 "
            "teleport/inert-readout monoculture). Does NOT by itself promote "
            "(v3_pending gate)."
        )
    if label == "rebinding_inert_off_equals_on":
        return "weakens", (
            "The binder tracks the true config (DV1) AND the couple() path has "
            "behavioural authority (the couple_path_has_behavioural_authority positive "
            "control passed: disabling the binding DOES change foraging), but re-binding "
            "confers no graded FORAGING advantage over a frozen episode-entry binding "
            "(DV2 fails: ON ~= FROZEN) -- the MECH-269(b)/V3-EXQ-478 inert-reset "
            "signature at the entity locus, now on the policy-coupled ecological path. "
            "Because the authority control passed, this null is a GENUINE inert-rebinding "
            "signal (the binding matters but re-evaluating it does not), not a "
            "measurement-resolution artefact; rebinding is readable but not "
            "behaviourally functional. Does NOT support promotion."
        )
    if label == "rebinding_not_tracking_truth":
        return "weakens", (
            "Rebinding affinity is not above the config-label-shuffle control -- "
            "arbitrary anchor-sensitivity, not functional tracking of the competing "
            "productive config. Does NOT support promotion."
        )
    return "non_contributory", (
        "Readiness gate unmet (unconverged binder / a config never visited in P0 / too "
        "few overtakes / oracle below the achievability floor / the couple() path lacks "
        "behavioural authority -- disabling the binding does not move foraging, so a null "
        "ON-vs-FROZEN DV2 is UNinterpretable, the V3-EXQ-478 false-weakens hazard). NOT a "
        "rebinding verdict and NOT a refutation (declared NULL: a small/absent DV2 on an "
        "unready or couple-inert substrate is substrate_ceiling, not clean evidence "
        "against MECH-456). Re-queue at adequate P0 / a substrate whose couple() path "
        "carries foraging authority. Not weighted in confidence/conflict scoring."
    )
