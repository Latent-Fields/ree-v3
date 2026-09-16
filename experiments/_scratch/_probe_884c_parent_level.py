"""
SCRATCH PROBE (not an experiment, no queue entry, no manifest).

V3-EXQ-884c Step 2.5a, part 2: the PARENT-LEVEL achievable ceiling, at the
config the encoded-space probe (_probe_884c_encoded_ceiling.py) selected --
pre-arrival credited representation (D1) at alpha_world=0.9.

The encoded-space probe measured the GROUP-MEAN-direction statistic. C1'
routes on a PARENT-level statistic (an EMA over the credited group), which is
a recency-weighted mean and therefore on a DIFFERENT scale. The chip's own
precondition ("measure the achievable ceiling BEFORE fixing the floor, not
after") applies to the statistic the criterion actually routes on, so it has
to be measured here too, not extrapolated from the group-mean number.

Two quantities per seed:

  C0 (fidelity)  cos(parent_replayed_true, agent.goal_state.z_goal_parent)
                 The credit goes through the REAL substrate call --
                 agent.notify_subgoal_attainment(ttype, child_representation=
                 prev_z_world) -- which GoalState.credit_subgoal_attainment's
                 own docstring sanctions ("child_representation is
                 caller-supplied ... which representation counts as 'the
                 attained subgoal' ... is an experiment-design decision ...
                 left to the call site, not baked into the substrate").
                 So the real _z_goal_parent IS the pre-arrival-credited
                 parent, and C0 checks that the replay used to build the
                 CONTROL parents reproduces it -- i.e. that the driver's
                 attained-tick set equals the substrate's credited-tick set
                 and the replay arithmetic matches. This is the control
                 red-team finding F2 asked for.

  S (the DV)     1 - cos(parent_true, parent_control), both built by the SAME
                 replay at the SAME credit-tick positions, differing ONLY in
                 which representations are credited; referenced against a
                 label-permutation null of the identical construction (F3's
                 own test, promoted from a diagnostic to the criterion).
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

GRID_SIZE, NUM_WAYPOINTS, N_STEPS, STAY_ACTION, WORLD_DIM = 12, 3, 400, 4, 32
P0_EPISODES, P0_STEPS_PER_EPISODE, P0_PRESERVATION_WEIGHT = 20, 50, 1000.0
PARENT_GOAL_ALPHA, PARENT_GOAL_DECAY = 0.05, 0.005
ALPHA_WORLD = 0.9
N_PERMUTATIONS = 200
SEEDS = [int(s) for s in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["42", "43", "44"])]


def build_env(seed):
    return CausalGridWorld(size=GRID_SIZE, num_hazards=0, num_resources=0,
                           subgoal_mode=True, num_waypoints=NUM_WAYPOINTS, seed=seed,
                           subgoal_arrival_position_check=True,
                           hazard_free_contamination_gate=True)


def build_agent(env):
    cfg = REEConfig.from_dims(body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
                              action_dim=env.action_dim, world_dim=WORLD_DIM,
                              z_goal_enabled=True, use_world_encoder_skip=True,
                              alpha_world=ALPHA_WORLD)
    assert abs(float(cfg.latent.alpha_world) - ALPHA_WORLD) < 1e-9
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


def cos(a, b, eps=1e-8):
    a = a.reshape(-1).float(); b = b.reshape(-1).float()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(eps))


def replay(reprs, positions, n_steps, dim):
    parent = torch.zeros(dim, dtype=torch.float32)
    cset = set(positions); k = 0
    a = min(1.0, PARENT_GOAL_ALPHA * 1.0)
    for tick in range(1, n_steps + 1):
        parent = parent * (1.0 - PARENT_GOAL_DECAY)
        if tick in cset and k < len(reprs):
            parent = (1.0 - a) * parent + a * reprs[k].reshape(-1).float()
            k += 1
    return parent


def run_seed(seed):
    wenv = build_env(seed)
    master = build_agent(wenv)
    run_zworld_p0(master, wenv, seed=seed, episodes=P0_EPISODES,
                  steps_per_episode=P0_STEPS_PER_EPISODE, policy=RandomPolicy(seed),
                  label="probe884cP", dry_run=False,
                  config=ZWorldP0Config(preservation_weight=P0_PRESERVATION_WEIGHT))
    trained = {k: v.detach().clone() for k, v in master.latent_stack.state_dict().items()}

    env = build_env(seed)
    agent = build_agent(env)
    agent.latent_stack.load_state_dict(trained)
    obs_flat, _ = env.reset()
    agent.act(obs_flat)
    prev_z = agent._current_latent.z_world.detach().clone().reshape(-1).float()

    att, non, cpos, steps = [], [], [], 0
    n_credits = 0
    for _ in range(N_STEPS):
        a = scripted_action(env)
        obs_flat, _h, done, info, _od = env.step(a)
        steps += 1
        agent.act(obs_flat)
        agent.update_z_goal(benefit_exposure=0.0, drive_level=0.0)
        ttype = info.get("transition_type", "none")
        # PRE-ARRIVAL credit through the REAL substrate path.
        res = agent.notify_subgoal_attainment(ttype, child_representation=prev_z)
        if res:
            n_credits = res["n_subgoal_credits"]
        if ttype in ("waypoint", "sequence_complete"):
            att.append(prev_z); cpos.append(steps)
        else:
            non.append(prev_z)
        prev_z = agent._current_latent.z_world.detach().clone().reshape(-1).float()
        if done:
            break

    real_parent = agent.goal_state.z_goal_parent.detach().clone().reshape(-1).float()
    dim = real_parent.numel()
    p_true = replay(att, cpos, steps, dim)
    c0 = cos(p_true, real_parent)

    gen = torch.Generator().manual_seed(seed)
    n_a = len(att)
    pool = att + non
    # observed control: same-size sample of NON-attained reprs at the same slots
    idx = torch.randperm(len(non), generator=gen)[:n_a].tolist()
    p_ctrl = replay([non[i] for i in idx], cpos, steps, dim)
    s_obs = 1.0 - cos(p_true, p_ctrl)

    null = []
    for _ in range(N_PERMUTATIONS):
        perm = torch.randperm(len(pool), generator=gen).tolist()
        a_set = [pool[i] for i in perm[:n_a]]
        b_set = [pool[i] for i in perm[n_a:n_a * 2]]
        null.append(1.0 - cos(replay(a_set, cpos, steps, dim),
                              replay(b_set, cpos, steps, dim)))
    null.sort()
    return dict(seed=seed, n_att=n_a, n_non=len(non), n_credits=n_credits, steps=steps,
                c0=c0, s_obs=s_obs, null_med=null[len(null) // 2],
                null_p95=null[int(0.95 * len(null))],
                pct=100.0 * sum(1 for v in null if v < s_obs) / len(null))


def main():
    print("=" * 78)
    print("V3-EXQ-884c Step 2.5a part 2 -- PARENT-LEVEL ceiling")
    print("alpha_world=%.2f  pre-arrival credit via the REAL substrate call  n_perm=%d"
          % (ALPHA_WORLD, N_PERMUTATIONS))
    print("=" * 78)
    for seed in SEEDS:
        r = run_seed(seed)
        print()
        print("--- seed %d: n_att=%d n_non=%d n_credits=%d steps=%d ---"
              % (r["seed"], r["n_att"], r["n_non"], r["n_credits"], r["steps"]))
        print("  C0 fidelity  cos(replay_true, real substrate parent) = %.9f" % r["c0"])
        print("  S  observed=%.6f  null_med=%.6f  null_p95=%.6f  pct=%5.1f -> %s"
              % (r["s_obs"], r["null_med"], r["null_p95"], r["pct"],
                 "SEPARATES" if r["s_obs"] > r["null_p95"] else "AT-CHANCE"))


if __name__ == "__main__":
    main()
