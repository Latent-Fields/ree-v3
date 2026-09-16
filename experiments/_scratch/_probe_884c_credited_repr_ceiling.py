"""
SCRATCH PROBE (not an experiment, no queue entry, no manifest).

Purpose: the achievable-ceiling measurement that
REE_assembly/evidence/planning/exq884b_mech428_c1_content_dv_redteam_blocking_20260914.md
section 4 makes a PRECONDITION for ratifying any V3-EXQ-884b redesign --
"measure the achievable ceiling BEFORE fixing CONTENT_DELTA_ABS_FLOOR, not
after" -- and which the parked driver's own dv_headroom block failed to do
(it recorded a bound, [-1,1], not an achievable estimate).

It answers ONE question for each of the two candidate fix directions named in
section 4, in RAW OBSERVATION SPACE (no encoder, no P0 training, no RNG beyond
the env seed and the permutation draws):

  How far apart are the credited-tick group mean and the non-credited-tick
  group mean, and is that separation distinguishable from a label-permutation
  null?

Raw-observation separation is the right quantity to gate on first because the
red-team's F4 finding is that the information is ABSENT FROM THE OBSERVATION;
an encoder cannot amplify a signature the observation never carried. A
direction whose RAW separation sits at the median of its own permutation null
is dead before any P0 recipe is chosen. (The converse is not implied: raw
separation is necessary, not sufficient -- a direction that clears this still
needs the full P0/G3 pass.)

Directions measured:
  D0  BASELINE (the refused 884b design): credit the POST-arrival observation.
  D1  Section-4 direction 1: credit the PRE-arrival observation (tick t-1).
  D2  Section-4 direction 2: credit the SD-WAYPOINT-FIELD observable
      (waypoint_proximity_field_view) at the arrival tick.

Env construction is verbatim from the parked driver's _build_env (plus
waypoint_proximity_field_enabled for D2), and the walk is its _scripted_action.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from ree_core.environment.causal_grid_world import CausalGridWorld

GRID_SIZE = 12
NUM_WAYPOINTS = 3
N_STEPS = 400
STAY_ACTION = 4
SEEDS = [42, 43, 44]
N_PERMUTATIONS = 200
WAYPOINT_TYPE = CausalGridWorld.ENTITY_TYPES["waypoint"]
NUM_ENTITY_TYPES = CausalGridWorld.NUM_ENTITY_TYPES
LOCAL_VIEW_DIM = 5 * 5 * NUM_ENTITY_TYPES  # 175


def build_env(seed, waypoint_field=False):
    env = CausalGridWorld(
        size=GRID_SIZE,
        num_hazards=0,
        num_resources=0,
        subgoal_mode=True,
        num_waypoints=NUM_WAYPOINTS,
        seed=seed,
        subgoal_arrival_position_check=True,
        hazard_free_contamination_gate=True,
        **({"waypoint_proximity_field_enabled": True,
            "use_proxy_fields": True} if waypoint_field else {}),
    )
    assert env.subgoal_arrival_position_check is True
    assert float(env.contamination_spread) == 0.0
    return env


def scripted_action(env):
    sub = env.get_subgoal_state()
    idx = sub["next_waypoint_idx"]
    if not env.waypoints or idx >= len(env.waypoints):
        return STAY_ACTION
    wx, wy = env.waypoints[idx]
    ax, ay = env.get_agent_position()
    if ax != wx:
        return 1 if wx > ax else 0
    if ay != wy:
        return 3 if wy > ay else 2
    return STAY_ACTION


def local_view_of(obs_flat):
    """world_state[0:175] reshaped to (5,5,7); see causal_grid_world.py:3762."""
    return obs_flat.reshape(-1)[:LOCAL_VIEW_DIM].reshape(5, 5, NUM_ENTITY_TYPES)


def waypoint_marker_visible(obs_flat):
    return bool(local_view_of(obs_flat)[:, :, WAYPOINT_TYPE].sum() > 0)


def mean_direction_dissim(group_a, group_b, eps=1e-8):
    """1 - cos(mean(a), mean(b)) -- the statistic that upper-bounds the
    parent-to-parent delta_group the C1' criterion computes, because both
    parents are EMAs (i.e. weighted means) of their group's members."""
    ma = torch.stack(group_a).mean(dim=0).reshape(-1).float()
    mb = torch.stack(group_b).mean(dim=0).reshape(-1).float()
    denom = (ma.norm() * mb.norm()).clamp_min(eps)
    return float(1.0 - (ma @ mb) / denom)


def permutation_null(all_reprs, n_a, n_perm, gen):
    """F3's test: n_perm random partitions of the SAME representations into
    groups of the SAME sizes. Returns the sorted null distribution."""
    null = []
    n_total = len(all_reprs)
    for _ in range(n_perm):
        perm = torch.randperm(n_total, generator=gen).tolist()
        ga = [all_reprs[i] for i in perm[:n_a]]
        gb = [all_reprs[i] for i in perm[n_a:]]
        null.append(mean_direction_dissim(ga, gb))
    return sorted(null)


def percentile_of(value, sorted_null):
    below = sum(1 for v in sorted_null if v < value)
    return 100.0 * below / len(sorted_null)


def walk(seed, waypoint_field):
    """One scripted walk.

    Two passes are needed because the SD-WAYPOINT-FIELD channel requires
    use_proxy_fields=True, and in proxy mode world_state carries the proxy
    field views INSTEAD of local_view -- so D0/D1 (which slice local_view out
    of world_state) and D2 cannot be read from the same env. The walk itself
    is driven by env state (agent/waypoint positions) via scripted_action, not
    by the observation, so the two passes visit an identical trajectory.
    """
    env = build_env(seed, waypoint_field=waypoint_field)
    obs_flat, obs_dict = env.reset()

    prev_obs = obs_flat.reshape(-1).clone()
    d0_att, d0_non = [], []
    d1_att, d1_non = [], []
    d2_att, d2_non = [], []
    vis_post, vis_pre, vis_transit = [], [], []
    positions = []

    for _ in range(N_STEPS):
        action_idx = scripted_action(env)
        obs_flat, _harm, done, info, obs_dict = env.step(action_idx)
        cur = obs_flat.reshape(-1).clone()
        ttype = info.get("transition_type", "none")
        attained = ttype in ("waypoint", "sequence_complete")
        positions.append((env.get_agent_position(), ttype))

        if waypoint_field:
            wf = obs_dict.get("waypoint_proximity_field_view")
            assert wf is not None, "waypoint_proximity_field_view absent in proxy mode"
            wf = wf.reshape(-1).clone().float()
            (d2_att if attained else d2_non).append(wf)
        else:
            cur_lv = cur[:LOCAL_VIEW_DIM]
            prev_lv = prev_obs[:LOCAL_VIEW_DIM]
            if attained:
                d0_att.append(cur_lv)
                d1_att.append(prev_lv)
                vis_post.append(waypoint_marker_visible(cur))
                vis_pre.append(waypoint_marker_visible(prev_obs))
            else:
                d0_non.append(cur_lv)
                d1_non.append(prev_lv)
                vis_transit.append(waypoint_marker_visible(cur))

        prev_obs = cur
        if done:
            break

    if waypoint_field:
        return {"D2_waypoint_field_at_arrival": (d2_att, d2_non)}, None, positions
    return (
        {"D0_post_arrival_local_view": (d0_att, d0_non),
         "D1_pre_arrival_local_view": (d1_att, d1_non)},
        (vis_post, vis_pre, vis_transit),
        positions,
    )


def frac(bools):
    return (100.0 * sum(bools) / len(bools)) if bools else float("nan")


def main():
    print("=" * 74)
    print("V3-EXQ-884b redesign -- achievable-ceiling probe (raw observation space)")
    print("seeds=%s  n_steps=%d  n_permutations=%d" % (SEEDS, N_STEPS, N_PERMUTATIONS))
    print("=" * 74)

    for seed in SEEDS:
        gen = torch.Generator().manual_seed(seed)
        res, vis, pos_a = walk(seed, waypoint_field=False)
        res2, _v2, pos_b = walk(seed, waypoint_field=True)
        assert pos_a == pos_b, (
            "proxy and non-proxy walks diverged -- the two passes are not "
            "comparable; D2 cannot be compared against D0/D1")
        res.update(res2)
        vis_post, vis_pre, vis_transit = vis
        n_att = len(vis_post)
        print()
        print("--- seed %d: %d attained ticks, %d transit ticks ---"
              % (seed, n_att, len(vis_transit)))
        print("  waypoint marker visible in local_view:")
        print("    at the POST-arrival tick (what 884b credits): %5.1f%%" % frac(vis_post))
        print("    at the PRE-arrival tick  (what D1 credits)  : %5.1f%%" % frac(vis_pre))
        print("    at transit ticks         (the contrast set) : %5.1f%%" % frac(vis_transit))

        if n_att < 2:
            print("  SKIP: fewer than 2 attained ticks, no group statistic possible")
            continue

        for name, (att, non) in res.items():
            if not att or not non:
                print("  %-34s UNAVAILABLE (empty group)" % name)
                continue
            observed = mean_direction_dissim(att, non)
            null = permutation_null(att + non, len(att), N_PERMUTATIONS, gen)
            pct = percentile_of(observed, null)
            median = null[len(null) // 2]
            p95 = null[int(0.95 * len(null))]
            verdict = "SEPARATES" if observed > p95 else "AT-CHANCE"
            print("  %-34s observed=%.6f  null_median=%.6f  null_p95=%.6f"
                  % (name, observed, median, p95))
            print("  %-34s percentile_of_own_null=%5.1f  -> %s"
                  % ("", pct, verdict))


if __name__ == "__main__":
    main()
