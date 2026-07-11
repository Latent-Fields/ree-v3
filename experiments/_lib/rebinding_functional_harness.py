"""
rebinding_functional_harness.py -- shared MECH-456 functional-rebinding measurement
harness (V3-EXQ-733 measurement code, factored for the V3-EXQ-733a 2-leg portfolio).

WHAT THIS IS
------------
The DV1/DV2 measurement machinery from V3-EXQ-733
(v3_exq_733_rebinding_functional_ground_truth.py), lifted VERBATIM into a shared
module so the two V3-EXQ-733a legs (P-A survival-onboarded agent; P-B directed-
traversal test-bed) reuse ONE measurement harness and differ only in how the
agent is built/trained and how the test-bed generates ground-truth overtakes.
NO ree_core change -- this is harness-level measurement over a trained-then-frozen
learned cross_stream_binder.

GROUND TRUTH (binder-independent). The agent moves through a size x size grid.
Partition it into a G x G coarse lattice -> K = G*G spatial REGIONS. The
ground-truth current world-configuration g(t) is the region the agent's position
(env.agent_x, env.agent_y) falls in -- read directly off the env, NOTHING the
binder produces. As the agent crosses a region boundary mid-episode the ground-
truth config CHANGES -- a genuine competing-configuration OVERTAKE.

BINDER READOUT. b_hat(anchor) = argmax_k binding_score(anchor, proto_k). Two
rebinding rules are read over ONE shared trajectory per seed (so g(t) is IDENTICAL
across arms and any DV difference is PURELY the rebinding rule):
  - REBIND-ON     : b_on(t) = b_hat(z_self_t)            (re-binds every step)
  - REBIND-FROZEN : b_frozen(t) = b_hat(z_self_episode_entry)  (never re-evaluated)

DV1 (tracks the TRUE competitor): A_real = frac steps b_on(t)==g(t) must beat a
region-label-shuffle chance baseline A_shuf by >= ALIGN_MARGIN.
DV2 (graded behavioural consequence): mean re-acquisition latency after an overtake
(censored, NON-saturating) must be >= LATENCY_MARGIN steps LOWER for ON than FROZEN.

READINESS. Each seed is READY only if the binder converged, every region was
visited >= min_region_visits_floor(K) in P0 (a granularity-scaled floor: K=4 -> 10,
K=16 -> 3; the fixed absolute 10 was granularity-blind and blocked the K=16
V3-EXQ-733a-b run at min 9 < 10), and >= MIN_OVERTAKES_P1 overtakes accrued in P1. If
any completed seed is not ready -> the run self-routes `substrate_not_ready_requeue`
(evidence_direction non_contributory), NEVER a rebinding verdict. (This is exactly the
gate V3-EXQ-733 failed: only seed 42 reached the 20-overtake floor because a cold
agent died in ~10-40 steps.)

The leg drivers own: SEEDS, G_PARTITION, the env factory, the agent
factory/onboarding, and the optional per-episode spawn_director. A leg running a
scripted directed tour that must NOT be truncated by P0 starvation passes
directed_tour_p0_coverage=True to run_functional_test_seed (V3-EXQ-733a fix; see its
docstring). Everything below is shared and leg-agnostic.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2


# ---------------------------------------------------------------------------
# Pre-registered acceptance thresholds (SHARED across both legs; identical to
# V3-EXQ-733 -- never derived from a run's own stats).
# ---------------------------------------------------------------------------
ALIGN_MARGIN = 0.10          # DV1: real alignment must beat the shuffle chance
                             # baseline by >= 10 percentage points (per seed).
LATENCY_MARGIN = 2.0         # DV2: frozen re-acq latency must exceed ON by >= 2
                             # steps (per seed) -- ON re-binds meaningfully faster.
REACQ_WINDOW = 20            # censor cap (steps) for re-acq latency + the
                             # post-overtake alignment window.
N_SHUFFLE_PERMS = 200        # random region relabelings for the chance baseline.

MIN_SEEDS_FOR_PASS = 4       # of 6 seeds must satisfy DV1 (and DV2) -- 2/3 ratio.
MIN_SEEDS_COMPLETED = 4      # of 6 runs must reach P1 without error.

MIN_REGION_VISITS_P0_BASE = 10  # calibrated at K=4 (G_PARTITION=2): the K=4 floor.
MIN_REGION_VISITS_P0_HARD_MIN = 3  # never demand fewer than this many P0 samples/region.
# Back-compat alias (drivers / manifests referencing the old name still resolve to the
# K=4-calibrated value). The LIVE readiness floor is min_region_visits_floor(k_regions).
MIN_REGION_VISITS_P0 = MIN_REGION_VISITS_P0_BASE
MIN_OVERTAKES_P1 = 20        # per seed: enough overtake events for latency stats.


def min_region_visits_floor(k_regions: int) -> int:
    """Granularity-scaled P0 region-coverage floor (V3-EXQ-733a fix).

    The base floor (10) was calibrated for a K=4 (G_PARTITION=2) lattice, where each
    region is a large cell the agent dwells in for many steps. A fixed absolute floor
    is granularity-BLIND: on a finer K=16 (G=4) lattice each region is a 3x3 cell the
    agent drifts out of in 1-2 moves, so per-region dwell is ~K/4 times smaller for the
    same step budget. Applying the K=4 floor of 10 to K=16 over-demands and tripped the
    V3-EXQ-733a-b readiness gate (min P0 region visits 9 < 10 on one starved seed) even
    though the mechanism passed DV1/DV2 6/6. Scale the floor inversely with K vs the K=4
    baseline, clamped to a hard minimum so a genuinely thin prototype is still caught.
    K=4 -> 10 (bit-identical to the calibrated value); K=9 -> 4; K=16 -> 3 (clamp).
    """
    base_k = 4
    scaled = int(round(MIN_REGION_VISITS_P0_BASE * base_k / max(1, int(k_regions))))
    return max(MIN_REGION_VISITS_P0_HARD_MIN, scaled)


# ---------------------------------------------------------------------------
# obs helpers (mirror V3-EXQ-733 / 725a / 641)
# ---------------------------------------------------------------------------


def obs_harm(obs_dict):
    h = obs_dict.get("harm_obs")
    return h.float().unsqueeze(0) if h is not None else None


def obs_harm_a(obs_dict):
    h = obs_dict.get("harm_obs_a")
    return h.float().unsqueeze(0) if h is not None else None


def obs_harm_history(obs_dict):
    h = obs_dict.get("harm_history")
    return h.float().unsqueeze(0) if h is not None else None


# ---------------------------------------------------------------------------
# Ground-truth region (binder-independent) + argmax binder readout
# ---------------------------------------------------------------------------


def region_of(env: CausalGridWorldV2, g_partition: int) -> int:
    """Ground-truth world-configuration label = the G x G lattice region the
    agent's position falls in. Read directly off the env (agent_x/agent_y);
    NOTHING the binder produces enters this."""
    s = int(env.size)
    g = int(g_partition)
    rx = min(g - 1, (int(env.agent_x) * g) // s)
    ry = min(g - 1, (int(env.agent_y) * g) // s)
    return rx * g + ry


def argmax_binding(binder, anchor: torch.Tensor, protos: List[torch.Tensor]) -> Tuple[int, List[float]]:
    """b_hat = argmax_k binding_score(anchor, proto_k). anchor [1,self_dim];
    protos: list of [1,world_dim]. Returns (index, per-proto scores)."""
    with torch.no_grad():
        scores = [float(binder.binding_score(anchor, p).item()) for p in protos]
    return max(range(len(scores)), key=lambda i: scores[i]), scores


# ---------------------------------------------------------------------------
# One natural-locomotion decision, exposing the observed latent
# ---------------------------------------------------------------------------


def step_decision(
    agent: REEAgent,
    obs_dict,
    zself_prev: Optional[torch.Tensor],
    act_prev: Optional[torch.Tensor],
    is_p1: bool,
):
    """Natural main-path decision (sense -> generate -> select_action). Returns
    (action[1,A], z_self_now[1,self_dim], z_world_now[1,world_dim]). During P0
    (is_p1=False) runs one binder curriculum step on the observed joint state;
    in P1 the binder is FROZEN (no update). Online E2 self-model updates fire in
    both phases (identical to the V3-EXQ-733 discipline)."""
    body = obs_dict["body_state"].float()
    world = obs_dict["world_state"].float()
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)

    latent = agent.sense(
        obs_body=body, obs_world=world,
        obs_harm=obs_harm(obs_dict),
        obs_harm_a=obs_harm_a(obs_dict),
        obs_harm_history=obs_harm_history(obs_dict),
    )
    z_self_now = latent.z_self.detach()
    z_world_now = latent.z_world.detach()
    if zself_prev is not None and act_prev is not None:
        agent.record_transition(zself_prev, act_prev, z_self_now)

    # P0 binder curriculum only; FROZEN in P1.
    if not is_p1:
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


# ---------------------------------------------------------------------------
# Alignment + latency helpers (over ONE seed's recorded P1 stream)
# ---------------------------------------------------------------------------


def label_shuffle_alignment(
    b_on: List[int], gts: List[int], k_regions: int, gen: torch.Generator
) -> float:
    """Chance baseline: mean over N_SHUFFLE_PERMS random NON-identity relabelings
    sigma of frac_t 1[b_on[t] == sigma(gts[t])]. Same argmax outputs, ground-truth
    correspondence destroyed (analogue of the 725a shuffle-of-real control)."""
    n = len(gts)
    if n == 0:
        return 0.0
    total = 0.0
    perms = 0
    for _ in range(N_SHUFFLE_PERMS):
        perm = torch.randperm(k_regions, generator=gen).tolist()
        if perm == list(range(k_regions)):
            continue  # skip identity so the baseline is a genuine shuffle
        agree = sum(1 for t in range(n) if b_on[t] == perm[gts[t]])
        total += agree / n
        perms += 1
    return (total / perms) if perms else 0.0


def anchor_shuffle_alignment(
    binder, anchors: List[torch.Tensor], protos: List[torch.Tensor],
    gts: List[int], gen: torch.Generator,
) -> float:
    """Report-only: b_hat(z_self_pi(t)) vs g(t) over one random timestep
    permutation pi -- isolates whether the CORRECT anchor matters (functional)
    vs any anchor (arbitrary anchor-sensitivity)."""
    n = len(gts)
    if n < 2:
        return 0.0
    perm = torch.randperm(n, generator=gen).tolist()
    agree = 0
    for t in range(n):
        b, _ = argmax_binding(binder, anchors[perm[t]], protos)
        if b == gts[t]:
            agree += 1
    return agree / n


def reacq_latencies(
    committed: List[int], gts: List[int], ep_ids: List[int]
) -> List[int]:
    """Re-acquisition latency after each overtake. committed[t] = the arm's
    committed binding at step t; gts[t] = ground truth; ep_ids[t] = episode id
    (latency search does not cross episode boundaries). An overtake at t is a
    within-episode ground-truth change (gts[t] != gts[t-1]); latency = steps until
    committed==gts again, censored at REACQ_WINDOW or episode end."""
    lats: List[int] = []
    n = len(gts)
    for t in range(1, n):
        if ep_ids[t] != ep_ids[t - 1]:
            continue  # new episode -- not an in-episode overtake
        if gts[t] == gts[t - 1]:
            continue  # no overtake
        # Overtake at t: find first d>=0 with committed[t+d]==gts[t+d], same episode.
        cap = REACQ_WINDOW
        d = 0
        lat = cap
        while d < cap and (t + d) < n and ep_ids[t + d] == ep_ids[t]:
            if committed[t + d] == gts[t + d]:
                lat = d
                break
            d += 1
        lats.append(lat)
    return lats


# ---------------------------------------------------------------------------
# One seed: P0 warmup (train binder + build prototypes) then P1 measurement.
# Leg-agnostic. The leg driver supplies the (already-built) agent, an env
# factory, G_PARTITION, and an optional per-episode spawn_director.
# ---------------------------------------------------------------------------


def run_functional_test_seed(
    seed: int,
    agent: REEAgent,
    env_factory: Callable[[int], CausalGridWorldV2],
    g_partition: int,
    p0_episodes: int,
    p1_episodes: int,
    steps_per_episode: int,
    spawn_director: Optional[
        Callable[[CausalGridWorldV2, Dict[str, Any], int, bool, torch.Generator], Dict[str, Any]]
    ] = None,
    step_director: Optional[
        Callable[[CausalGridWorldV2, Dict[str, Any], int, int, bool, torch.Generator], Dict[str, Any]]
    ] = None,
    directed_tour_p0_coverage: bool = False,
) -> Dict[str, Any]:
    """Run the binder P0 curriculum + P1 measurement for one seed on a
    (possibly pre-onboarded) agent. Returns the per-seed row dict (schema
    identical to V3-EXQ-733).

    spawn_director (optional, P-B): called immediately after env.reset() as
    spawn_director(env, obs, ep, is_p1, rng) and MUST return the (possibly
    refreshed) obs dict. It may reposition the agent (region-spread start spawn)
    to generate ground-truth overtakes independent of foraging survival. None ->
    use the reset obs unchanged (the V3-EXQ-733 path).

    step_director (optional, P-B): called after each env.step() (and after the
    z_goal update) as step_director(env, obs, ep, step_idx, is_p1, rng) and MUST
    return the (possibly refreshed) obs dict for the NEXT decision. It may
    directed-respawn/teleport the agent (scripted region-traversal) so a within-
    episode ground-truth overtake is GENERATED regardless of the policy's
    foraging competence. None -> the loop is bit-identical to V3-EXQ-733.

    directed_tour_p0_coverage (optional, P-B / V3-EXQ-733b): DECOUPLE P0
    prototype-coverage from foraging survival, mirroring the way step_director
    decouples P1 overtakes (V3-EXQ-733a fix). When True (requires a spawn_director
    and/or step_director), during P0 ONLY: do NOT truncate the P0 episode on a
    health-death `done` (`env.agent_health <= 0`). Rationale: the teleport is a pure
    position read that does NOT refill health, so on unlucky seeds starvation still
    ended P0 episodes early (V3-EXQ-733a-b seed 43: ~545 P0 step-visits vs ~4600 on
    seed 42), which was the proximate cause of the min-region-visits 9<10 coverage
    block. Suppressing that health-death break lets the scripted tour run its full
    step budget, which in turn makes the harness's EXISTING teleport-entry accrual
    guaranteed by construction: after each directed relocation the very next loop
    iteration reads `region_of(env)` at the directed-respawn cell (BEFORE the agent
    acts/drifts) and accrues that region's clean z_world prototype sample + visit, so
    P0 coverage becomes a deterministic function of the scripted tour rather than of
    where a starving forager happens to dwell -- WITHOUT a second `agent.sense()`
    (the binder-training / prototype mechanism that passed DV1/DV2 6/6 is unchanged;
    only the truncation that starved it is removed). Any other `done` cause (e.g. the
    env step cap `steps >= 500`) still breaks; P1 measurement episodes and no-director
    legs are unaffected. Pairs with the granularity-scaled min_region_visits_floor(K)
    readiness floor (a fixed K=4 floor of 10 over-demands on the finer K=16 lattice).
    Default False -> bit-identical to V3-EXQ-733 (and to the P-A survival-onboarded
    leg, K=4). NO ree_core change -- this is test-bed measurement decoupling only.
    """
    k_regions = g_partition * g_partition
    error_note: Optional[str] = None

    agent.eval()
    env = env_factory(seed)

    world_dim = int(agent.config.latent.world_dim)
    proto_sum = [torch.zeros(1, world_dim) for _ in range(k_regions)]
    proto_cnt = [0 for _ in range(k_regions)]
    global_zworld_sum = torch.zeros(1, world_dim)
    global_zworld_cnt = 0

    p1_anchors: List[torch.Tensor] = []
    p1_gts: List[int] = []
    p1_ep_ids: List[int] = []
    p1_b_on: List[int] = []
    p1_b_frozen: List[int] = []
    region_visits_p0 = [0 for _ in range(k_regions)]

    protos: Optional[List[torch.Tensor]] = None
    binder = None

    total_train_eps = p0_episodes + p1_episodes

    gen_align = torch.Generator()
    gen_align.manual_seed(seed * 104729 + 7)
    gen_spawn = torch.Generator()
    gen_spawn.manual_seed(seed * 100003 + 11)

    for ep in range(total_train_eps):
        is_p1 = ep >= p0_episodes
        _, obs = env.reset()
        agent.reset()
        if spawn_director is not None:
            obs = spawn_director(env, obs, ep, is_p1, gen_spawn)
        zp = ap = None
        frozen_binding: Optional[int] = None  # set at episode entry in P1

        for _step in range(steps_per_episode):
            region_now = region_of(env, g_partition)  # region the CURRENT obs is in

            action, zself_now, zworld_now = step_decision(agent, obs, zp, ap, is_p1)
            if not torch.isfinite(action).all():
                error_note = f"non-finite action seed={seed} ep={ep} step={_step}"
                break

            if not is_p1:
                proto_sum[region_now] = proto_sum[region_now] + zworld_now.reshape(1, -1)
                proto_cnt[region_now] += 1
                region_visits_p0[region_now] += 1
                global_zworld_sum = global_zworld_sum + zworld_now.reshape(1, -1)
                global_zworld_cnt += 1
            else:
                if protos is None:
                    gmean = (
                        (global_zworld_sum / global_zworld_cnt)
                        if global_zworld_cnt > 0 else torch.zeros(1, world_dim)
                    )
                    protos = [
                        (proto_sum[k] / proto_cnt[k]) if proto_cnt[k] > 0 else gmean.clone()
                        for k in range(k_regions)
                    ]
                    binder = agent.cross_stream_binder
                if binder is None:
                    error_note = f"binder absent seed={seed} (substrate OFF/fixed)"
                    break
                anchor = zself_now.reshape(1, -1)
                b_on, _ = argmax_binding(binder, anchor, protos)
                if frozen_binding is None:
                    frozen_binding, _ = argmax_binding(binder, anchor, protos)
                p1_anchors.append(anchor.clone())
                p1_gts.append(int(region_now))
                p1_ep_ids.append(int(ep))
                p1_b_on.append(int(b_on))
                p1_b_frozen.append(int(frozen_binding))

            _, _hs, done, info, obs = env.step(action)
            if agent.goal_state is not None:
                be = float(info.get("benefit_exposure", 0.0))
                en = float(obs["body_state"].float().reshape(-1)[3].item())
                agent.update_z_goal(benefit_exposure=be, drive_level=max(0.0, 1.0 - en))

            zp, ap = zself_now, action.detach()
            if done:
                # V3-EXQ-733a fix (2): during the P0 scripted directed tour, do NOT
                # let a health-death `done` truncate the episode -- the teleport does
                # not refill health, so starvation would otherwise starve P0 coverage
                # (seed 43). Suppress ONLY the health-death cause and ONLY in P0 tour
                # mode; any other `done` (e.g. env step cap) still ends the episode.
                suppress_p0_health_death = (
                    directed_tour_p0_coverage
                    and not is_p1
                    and float(env.agent_health) <= 0.0
                )
                if not suppress_p0_health_death:
                    break
            if step_director is not None:
                obs = step_director(env, obs, ep, _step, is_p1, gen_spawn)

        if error_note is not None:
            break
        if (ep + 1) % 10 == 0 or ep == total_train_eps - 1:
            print(
                f"  [measure] seed={seed} ep {ep + 1}/{total_train_eps} "
                f"phase={'P1' if is_p1 else 'P0'} "
                f"p1_steps={len(p1_gts)} "
                f"regions_covered={sum(1 for c in proto_cnt if c > 0)}/{k_regions}",
                flush=True,
            )

    return _compute_seed_row(
        seed=seed,
        g_partition=g_partition,
        k_regions=k_regions,
        error_note=error_note,
        agent=agent,
        protos=protos,
        proto_cnt=proto_cnt,
        region_visits_p0=region_visits_p0,
        p1_anchors=p1_anchors,
        p1_gts=p1_gts,
        p1_ep_ids=p1_ep_ids,
        p1_b_on=p1_b_on,
        p1_b_frozen=p1_b_frozen,
        gen_align=gen_align,
    )


def _compute_seed_row(
    *,
    seed: int,
    g_partition: int,
    k_regions: int,
    error_note: Optional[str],
    agent: REEAgent,
    protos: Optional[List[torch.Tensor]],
    proto_cnt: List[int],
    region_visits_p0: List[int],
    p1_anchors: List[torch.Tensor],
    p1_gts: List[int],
    p1_ep_ids: List[int],
    p1_b_on: List[int],
    p1_b_frozen: List[int],
    gen_align: torch.Generator,
) -> Dict[str, Any]:
    """DV1 / DV2 + readiness over one seed's recorded P1 stream (V3-EXQ-733
    _run_seed tail, verbatim except K_REGIONS -> k_regions parameter)."""
    binder_b = agent.cross_stream_binder
    binder_learned = bool(binder_b is not None and getattr(binder_b, "learned", False))
    binder_loss_ema = getattr(binder_b, "loss_ema", None) if binder_b else None
    binder_chance_floor = getattr(binder_b, "chance_floor", None) if binder_b else None
    binder_converged = bool(getattr(binder_b, "binder_converged", False)) if binder_b else False

    n_p1 = len(p1_gts)
    n_overtakes = sum(
        1 for t in range(1, n_p1)
        if p1_ep_ids[t] == p1_ep_ids[t - 1] and p1_gts[t] != p1_gts[t - 1]
    )

    a_real = (sum(1 for t in range(n_p1) if p1_b_on[t] == p1_gts[t]) / n_p1) if n_p1 else 0.0
    a_shuf = label_shuffle_alignment(p1_b_on, p1_gts, k_regions, gen_align) if n_p1 else 0.0
    a_anchor_shuf = (
        anchor_shuffle_alignment(binder_b, p1_anchors, protos, p1_gts, gen_align)
        if (n_p1 and protos is not None and binder_b is not None) else 0.0
    )

    post_hits = 0
    post_tot = 0
    for t in range(1, n_p1):
        if p1_ep_ids[t] == p1_ep_ids[t - 1] and p1_gts[t] != p1_gts[t - 1]:
            d = 0
            while d < REACQ_WINDOW and (t + d) < n_p1 and p1_ep_ids[t + d] == p1_ep_ids[t]:
                post_tot += 1
                if p1_b_on[t + d] == p1_gts[t + d]:
                    post_hits += 1
                d += 1
    post_overtake_align_on = (post_hits / post_tot) if post_tot else 0.0

    lats_on = reacq_latencies(p1_b_on, p1_gts, p1_ep_ids)
    lats_frozen = reacq_latencies(p1_b_frozen, p1_gts, p1_ep_ids)
    mean_lat_on = (sum(lats_on) / len(lats_on)) if lats_on else float(REACQ_WINDOW)
    mean_lat_frozen = (sum(lats_frozen) / len(lats_frozen)) if lats_frozen else float(REACQ_WINDOW)

    stale_rate_on = (
        (sum(1 for t in range(n_p1) if p1_b_on[t] != p1_gts[t]) / n_p1) if n_p1 else 1.0
    )
    stale_rate_frozen = (
        (sum(1 for t in range(n_p1) if p1_b_frozen[t] != p1_gts[t]) / n_p1) if n_p1 else 1.0
    )

    dv1 = bool(a_real >= a_shuf + ALIGN_MARGIN)
    dv2 = bool((mean_lat_frozen - mean_lat_on) >= LATENCY_MARGIN)

    min_region_visits = min(region_visits_p0) if region_visits_p0 else 0
    region_visits_floor = min_region_visits_floor(k_regions)  # V3-EXQ-733a: K-scaled
    regions_covered = sum(1 for c in proto_cnt if c > 0)
    seed_ready = bool(
        binder_learned and binder_converged
        and regions_covered == k_regions
        and min_region_visits >= region_visits_floor
        and n_overtakes >= MIN_OVERTAKES_P1
    )

    return {
        "seed": int(seed),
        "error_note": error_note,
        "n_p1_steps": int(n_p1),
        "n_overtakes": int(n_overtakes),
        "regions_covered": int(regions_covered),
        "k_regions": int(k_regions),
        "g_partition": int(g_partition),
        "min_region_visits_p0": int(min_region_visits),
        "region_visits_p0_floor": int(region_visits_floor),  # V3-EXQ-733a K-scaled floor
        "region_visits_p0": [int(v) for v in region_visits_p0],
        # DV1
        "alignment_real": float(a_real),
        "alignment_shuffle": float(a_shuf),
        "alignment_margin_obs": float(a_real - a_shuf),
        "alignment_anchor_shuffle": float(a_anchor_shuf),
        "post_overtake_alignment_on": float(post_overtake_align_on),
        "DV1_tracks_truth_above_shuffle": dv1,
        # DV2
        "mean_reacq_latency_on": float(mean_lat_on),
        "mean_reacq_latency_frozen": float(mean_lat_frozen),
        "latency_gap_obs": float(mean_lat_frozen - mean_lat_on),
        "n_overtake_events_scored": int(len(lats_on)),
        "stale_rate_on": float(stale_rate_on),
        "stale_rate_frozen": float(stale_rate_frozen),
        "stale_rate_gap_obs": float(stale_rate_frozen - stale_rate_on),
        "DV2_behavioural_consequence": dv2,
        # readiness
        "binder_learned": binder_learned,
        "binder_loss_ema": binder_loss_ema,
        "binder_chance_floor": binder_chance_floor,
        "binder_converged": binder_converged,
        "seed_ready": seed_ready,
    }


# ---------------------------------------------------------------------------
# Interpretation grid (shared; identical to V3-EXQ-733 _interpret)
# ---------------------------------------------------------------------------


def interpret(rows: List[Dict[str, Any]], conv_frac: float) -> Dict[str, Any]:
    ok = [r for r in rows if r["error_note"] is None]

    all_ready = bool(ok) and all(r["seed_ready"] for r in ok)
    n_dv1 = sum(1 for r in ok if r["DV1_tracks_truth_above_shuffle"])
    n_dv2 = sum(1 for r in ok if r["DV2_behavioural_consequence"])
    n_both = sum(
        1 for r in ok
        if r["DV1_tracks_truth_above_shuffle"] and r["DV2_behavioural_consequence"]
    )

    dv1_pass = bool(n_dv1 >= MIN_SEEDS_FOR_PASS)
    dv2_pass = bool(n_dv2 >= MIN_SEEDS_FOR_PASS)
    both_pass = bool(n_both >= MIN_SEEDS_FOR_PASS)

    if not all_ready:
        label = "substrate_not_ready_requeue"
    elif dv1_pass and dv2_pass and both_pass:
        label = "functional_rebinding_supported"
    elif dv1_pass and not dv2_pass:
        label = "rebinding_inert_off_equals_on"
    elif not dv1_pass:
        label = "rebinding_not_tracking_truth"
    else:
        label = "rebinding_inert_off_equals_on"

    worst_ema = None
    worst_chance = None
    _emas = [r.get("binder_loss_ema") for r in ok if r.get("binder_loss_ema") is not None]
    _ch = [r.get("binder_chance_floor") for r in ok if r.get("binder_chance_floor") is not None]
    if _emas:
        worst_ema = float(max(_emas))
    if _ch:
        worst_chance = float(min(_ch))
    conv_threshold = (
        float(conv_frac) * worst_chance if worst_chance is not None else None
    )
    min_overtakes = min((r["n_overtakes"] for r in ok), default=0)
    min_region_visits = min((r["min_region_visits_p0"] for r in ok), default=0)
    # V3-EXQ-733a: the readiness floor is granularity-scaled to the lattice K (all
    # completed seeds share K); fall back to the K=4 base if no seed completed.
    k_regions_interp = int(ok[0]["k_regions"]) if ok else 4
    region_visits_floor = min_region_visits_floor(k_regions_interp)

    preconditions = [
        {
            "name": "learned_binder_converged",
            "kind": "readiness",
            "description": (
                "Every completed seed's binder must converge (loss_ema < "
                "conv_frac*log(batch)); an unconverged binder is the 725 "
                "untrained-substrate artifact, not a rebinding verdict. Routed "
                "statistic: WORST (max) loss_ema vs the convergence threshold "
                "(upper-bound: met when measured <= threshold)."
            ),
            "measured": (worst_ema if worst_ema is not None else 1e9),
            "threshold": (conv_threshold if conv_threshold is not None else 0.0),
            "direction": "upper",
            "control": "substrate binder_converged on the trained-then-frozen P0 binder",
            "met": bool(ok and all(r["binder_converged"] for r in ok)),
        },
        {
            "name": "region_coverage_adequate",
            "kind": "readiness",
            "description": (
                "min region visit count in P0 across completed seeds clears the "
                "prototype-stability floor (every region seen enough to mean-pool a "
                "prototype); below floor -> thin/ill-defined prototypes. Floor is "
                "granularity-scaled to K (V3-EXQ-733a): the fixed K=4 floor of 10 was "
                "granularity-blind on the finer K=16 lattice."
            ),
            "measured": int(min_region_visits),
            "threshold": int(region_visits_floor),
            "control": "per-region P0 visit counts",
            "met": bool(ok and min_region_visits >= region_visits_floor),
        },
        {
            "name": "overtake_events_adequate",
            "kind": "readiness",
            "description": (
                "min overtake (ground-truth region change) count in P1 across "
                "completed seeds clears the floor so the latency DV is not a mean "
                "over a near-empty set of events. THIS is the gate V3-EXQ-733 "
                "failed (min 2 vs 20; only seed 42 reached it)."
            ),
            "measured": int(min_overtakes),
            "threshold": int(MIN_OVERTAKES_P1),
            "control": "per-seed in-episode ground-truth region-change count",
            "met": bool(ok and min_overtakes >= MIN_OVERTAKES_P1),
        },
    ]

    criteria = [
        {"name": "learned_binder_converged", "load_bearing": True,
         "passed": bool(ok and all(r["binder_converged"] for r in ok))},
        {"name": "DV1_tracks_truth_above_shuffle", "load_bearing": True,
         "passed": dv1_pass},
        {"name": "DV2_behavioural_consequence", "load_bearing": True,
         "passed": dv2_pass},
    ]

    return {
        "label": label,
        "preconditions": preconditions,
        "criteria": criteria,
        "all_ready": all_ready,
        "n_completed": len(ok),
        "n_DV1": int(n_dv1),
        "n_DV2": int(n_dv2),
        "n_both": int(n_both),
        "min_seeds_for_pass": int(MIN_SEEDS_FOR_PASS),
        "dv1_pass": dv1_pass,
        "dv2_pass": dv2_pass,
        "both_pass": both_pass,
        "worst_binder_loss_ema": worst_ema,
        "binder_conv_threshold": conv_threshold,
        "min_overtakes": int(min_overtakes),
        "min_region_visits_p0": int(min_region_visits),
        "min_region_visits_p0_floor": int(region_visits_floor),
    }


def evidence_direction(label: str) -> Tuple[str, str]:
    """Map the interpretation label to a (direction, note) for MECH-456
    (identical mapping to V3-EXQ-733)."""
    if label == "functional_rebinding_supported":
        return "supports", (
            "DV1 (rebinding-alignment-with-ground-truth above the shuffle control) "
            "AND DV2 (rebinding-ON re-acquires the correct binding faster than a "
            "rebinding-frozen arm on a graded, non-saturating latency metric) both "
            "clear on >= MIN_SEEDS_FOR_PASS seeds -> functional rebinding demonstrated; "
            "SUPPORTS MECH-456 promotion (candidate -> shown; clear v3_pending)."
        )
    if label == "rebinding_inert_off_equals_on":
        return "weakens", (
            "The binder tracks ground truth (DV1) but re-binding confers no graded "
            "behavioural advantage over a frozen binding (DV2 fails: ON==FROZEN on "
            "re-acquisition latency) -- the MECH-269(b)/V3-EXQ-478 inert-reset "
            "signature at the entity locus. Rebinding is exercisable but not "
            "functional; does NOT support promotion."
        )
    if label == "rebinding_not_tracking_truth":
        return "weakens", (
            "Rebinding alignment is not above the shuffle control -- arbitrary "
            "anchor-sensitivity, not functional tracking of the true competitor "
            "(the n_rebind>0 of 725a was exercisability only). Does NOT support "
            "promotion."
        )
    return "non_contributory", (
        "Readiness gate unmet (unconverged binder / thin region coverage / too few "
        "overtakes). NOT a rebinding verdict; re-queue at adequate P0 / more P1 "
        "steps. Not weighted in confidence/conflict scoring."
    )
