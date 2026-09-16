"""
SCRATCH PROBE (not an experiment, no queue entry, no manifest).

Step 2.5a / achievable-ceiling measurement for V3-EXQ-884c, in ENCODED
z_world space -- the space the C1' criterion actually routes on.

The batch-5b probe (_probe_884c_credited_repr_ceiling.py) measured the ceiling
in RAW OBSERVATION space and found:
  D0 post-arrival local_view : 0.0132 / 0.0182 / 0.0140  (SEPARATES, 100th pct)
  D1 pre-arrival  local_view : 0.0261 / 0.0242 / 0.0222  (SEPARATES, 100th pct)
  D2 waypoint field          : 0.0018 / 0.0036 / 0.0023  (AT-CHANCE on seed 42)
i.e. F4's STRONG form ("the observation never carried the signature") is too
strong -- the raw observation does separate. What kills it is the ENCODER
path: 884b's calibration trail measured the same statistic on ENCODED z_world
at ~0.0007 (preservation_weight=1000), against a permutation null with
median 0.000859 and p95 0.0027 -- i.e. BELOW its own null.

So the question this probe answers is not "does the observation carry it" but
"where does the raw ~0.024 go, and is any config on the P0/encoder path that
preserves enough of it to clear a permutation null".

TWO LEVERS MEASURED, crossed:
  credited tick : D0 (post-arrival, what 884b credited) vs D1 (pre-arrival)
  alpha_world   : 0.3 (884b's value -- the REEConfig default) vs 0.9

alpha_world is the untested lever. ree_core/latent/stack.py:1584 does
    z_world = alpha_world * z_world + (1 - alpha_world) * prev_state.z_world
so at the default 0.3 EVERY sensed z_world is 70 percent inherited from the
previous tick -- a temporal EMA that smears an attainment tick into its
transit neighbours before any crediting happens. config.py:80-84 says
"alpha_world should be >= 0.9 (or 1.0 = no blending)" and stack.py:1537 names
0.3 as backward-compat only ("set alpha_world >= 0.9 to fix event
suppression"). 884b never set it (grep: no alpha_world anywhere in the parked
driver), so it ran at 0.3. An event-suppression EMA is exactly the mechanism
that would map a raw 0.024 separation onto an encoded 0.0007.

Statistic and null are IDENTICAL to the batch-5b raw probe so the two are
directly comparable.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.latent.zworld_p0 import ZWorldP0Config
from experiments._lib.zworld_p0_warmup import run_zworld_p0
from experiments._lib.capability_eval import RandomPolicy

GRID_SIZE = 12
NUM_WAYPOINTS = 3
N_STEPS = 400
STAY_ACTION = 4
WORLD_DIM = 32
P0_EPISODES = 20
P0_STEPS_PER_EPISODE = 50
P0_PRESERVATION_WEIGHT = 1000.0
N_PERMUTATIONS = 200

SEEDS = [int(s) for s in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["42"])]
ALPHAS = [float(a) for a in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["0.3", "0.9"])]


def build_env(seed):
    env = CausalGridWorld(
        size=GRID_SIZE, num_hazards=0, num_resources=0, subgoal_mode=True,
        num_waypoints=NUM_WAYPOINTS, seed=seed,
        subgoal_arrival_position_check=True, hazard_free_contamination_gate=True,
    )
    assert env.subgoal_arrival_position_check is True
    assert float(env.contamination_spread) == 0.0
    return env


def build_agent(env, alpha_world):
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim, world_dim=WORLD_DIM,
        z_goal_enabled=True, use_world_encoder_skip=True,
        alpha_world=alpha_world,
    )
    got = float(getattr(cfg.latent, "alpha_world", -1.0))
    assert abs(got - alpha_world) < 1e-9, (
        "alpha_world did not reach cfg.latent (from_dims swallowed it): got %r" % got)
    agent = REEAgent(cfg)
    agent.goal_state.config.use_hierarchical_goal_credit = True
    return agent


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


def mean_direction_dissim(group_a, group_b, eps=1e-8):
    ma = torch.stack(group_a).mean(dim=0).reshape(-1).float()
    mb = torch.stack(group_b).mean(dim=0).reshape(-1).float()
    denom = (ma.norm() * mb.norm()).clamp_min(eps)
    return float(1.0 - (ma @ mb) / denom)


def permutation_null(all_reprs, n_a, n_perm, gen):
    null = []
    n_total = len(all_reprs)
    for _ in range(n_perm):
        perm = torch.randperm(n_total, generator=gen).tolist()
        null.append(mean_direction_dissim([all_reprs[i] for i in perm[:n_a]],
                                          [all_reprs[i] for i in perm[n_a:]]))
    return sorted(null)


def pct_of(value, sorted_null):
    return 100.0 * sum(1 for v in sorted_null if v < value) / len(sorted_null)


def run_one(seed, alpha_world):
    warmup_env = build_env(seed)
    master = build_agent(warmup_env, alpha_world)
    run_zworld_p0(master, warmup_env, seed=seed, episodes=P0_EPISODES,
                  steps_per_episode=P0_STEPS_PER_EPISODE, policy=RandomPolicy(seed),
                  label="probe884c", dry_run=False,
                  config=ZWorldP0Config(preservation_weight=P0_PRESERVATION_WEIGHT))
    trained = {k: v.detach().clone() for k, v in master.latent_stack.state_dict().items()}

    env = build_env(seed)
    agent = build_agent(env, alpha_world)
    agent.latent_stack.load_state_dict(trained)

    obs_flat, _ = env.reset()
    agent.act(obs_flat)
    prev_z = agent._current_latent.z_world.detach().clone().reshape(-1).float()

    d0_att, d0_non, d1_att, d1_non = [], [], [], []
    for _ in range(N_STEPS):
        a = scripted_action(env)
        obs_flat, _harm, done, info, _od = env.step(a)
        agent.act(obs_flat)
        z_now = agent._current_latent.z_world.detach().clone().reshape(-1).float()
        attained = info.get("transition_type", "none") in ("waypoint", "sequence_complete")
        (d0_att if attained else d0_non).append(z_now)
        (d1_att if attained else d1_non).append(prev_z)
        prev_z = z_now
        if done:
            break
    return {"D0_post_arrival_zworld": (d0_att, d0_non),
            "D1_pre_arrival_zworld": (d1_att, d1_non)}


def main():
    print("=" * 78)
    print("V3-EXQ-884c Step 2.5a -- ACHIEVABLE CEILING in ENCODED z_world space")
    print("seeds=%s alphas=%s P0=%dx%d pres_w=%g n_perm=%d"
          % (SEEDS, ALPHAS, P0_EPISODES, P0_STEPS_PER_EPISODE,
             P0_PRESERVATION_WEIGHT, N_PERMUTATIONS))
    print("raw-space reference (batch 5b): D0 ~0.015, D1 ~0.024, both 100th pct")
    print("=" * 78)
    for alpha in ALPHAS:
        for seed in SEEDS:
            gen = torch.Generator().manual_seed(seed)
            res = run_one(seed, alpha)
            print()
            print("--- alpha_world=%.2f seed=%d ---" % (alpha, seed))
            for name, (att, non) in res.items():
                if len(att) < 2 or not non:
                    print("  %-26s UNAVAILABLE (n_att=%d n_non=%d)" % (name, len(att), len(non)))
                    continue
                obs = mean_direction_dissim(att, non)
                null = permutation_null(att + non, len(att), N_PERMUTATIONS, gen)
                p95 = null[int(0.95 * len(null))]
                print("  %-26s n_att=%3d observed=%.6f null_med=%.6f null_p95=%.6f pct=%5.1f -> %s"
                      % (name, len(att), obs, null[len(null) // 2], p95,
                         pct_of(obs, null), "SEPARATES" if obs > p95 else "AT-CHANCE"))


if __name__ == "__main__":
    main()
