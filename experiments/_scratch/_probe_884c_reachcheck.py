"""SCRATCH: Step 2.5c empirical reach check for V3-EXQ-884c.

Does the 884b/884c driver's agent loop (scripted walk + agent.act +
update_z_goal + notify_subgoal_attainment, NO agent.select_action) actually
reach the code paths named by the OPEN, severity=corrupting substrate_queue
entries? A module-level import is not reach; only a call is.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
import ree_core.residue.field as residue_field
import ree_core.predictors.e1_deep as e1_deep
import ree_core.predictors.e3_selector as e3_selector

CALLS = {}


def _count(name):
    CALLS[name] = CALLS.get(name, 0) + 1


def wrap(obj, attr, label):
    orig = getattr(obj, attr, None)
    if orig is None:
        print("  (absent: %s)" % label)
        return
    def wrapped(*a, **k):
        _count(label)
        return orig(*a, **k)
    setattr(obj, attr, wrapped)


wrap(residue_field.RBFLayer, "add_residue", "residue.RBFLayer.add_residue")
wrap(residue_field.ResidueField, "accumulate", "residue.accumulate")
wrap(residue_field.ResidueField, "accumulate_benefit", "residue.accumulate_benefit")
wrap(residue_field.RBFLayer, "forward", "residue.RBFLayer.forward")
wrap(e1_deep.ContextMemory, "write", "e1.ContextMemory.write")
wrap(e3_selector.E3TrajectorySelector, "score_trajectory", "e3.score_trajectory")
wrap(e3_selector.E3TrajectorySelector, "select", "e3.select")

env = CausalGridWorld(size=12, num_hazards=0, num_resources=0, subgoal_mode=True,
                      num_waypoints=3, seed=42, subgoal_arrival_position_check=True,
                      hazard_free_contamination_gate=True)
cfg = REEConfig.from_dims(body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
                          action_dim=env.action_dim, world_dim=32, z_goal_enabled=True,
                          use_world_encoder_skip=True)
agent = REEAgent(cfg)
agent.goal_state.config.use_hierarchical_goal_credit = True

obs_flat, _ = env.reset()
agent.act(obs_flat)
STAY = 4
for _ in range(80):
    sub = env.get_subgoal_state()
    idx = sub["next_waypoint_idx"]
    if not env.waypoints or idx >= len(env.waypoints):
        a = STAY
    else:
        wx, wy = env.waypoints[idx]
        ax, ay = env.get_agent_position()
        a = (1 if wx > ax else 0) if ax != wx else ((3 if wy > ay else 2) if ay != wy else STAY)
    obs_flat, _h, done, info, _od = env.step(a)
    agent.act(obs_flat)
    agent.update_z_goal(benefit_exposure=0.0, drive_level=0.0)
    agent.notify_subgoal_attainment(info.get("transition_type", "none"))
    if done:
        break

print("REACH CHECK over 80 steps of the 884c loop:")
for label in ("residue.RBFLayer.add_residue", "residue.accumulate", "residue.accumulate_benefit",
              "residue.RBFLayer.forward", "e1.ContextMemory.write",
              "e3.score_trajectory", "e3.select"):
    print("  %-28s %s" % (label, CALLS.get(label, 0)))
