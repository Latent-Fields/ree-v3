"""SCRATCH: Step 2.5c empirical reach check for V3-EXQ-884c, extended.

Batch 5b's _probe_884c_reachcheck.py covered the entries open at that time.
The Step 2.5c listing was re-derived today after REE_assembly c0676eb1611
closed f_dominance_conversion_ceiling / MECH122-...; the OPEN corrupting set
is now MECH-320 (tonic_vigor), contextmemory-write-path-addressing-degeneracy
(e1_deep ContextMemory.write), sd_blocked_agency_mismatch_floor_calibration
(affect/blocked_agency), sd105_frozen_shared_entropy_floor_multiplier
(regulators/selection_entropy_floor). A module-level import is not reach;
only a call is. The loop below is 884c's exact loop, INCLUDING the
alpha_world=0.9 config and the explicit child_representation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
import ree_core.policy.tonic_vigor as tonic_vigor
import ree_core.affect.blocked_agency as blocked_agency
import ree_core.regulators.selection_entropy_floor as sef
import ree_core.predictors.e1_deep as e1_deep

CALLS = {}


def wrap_all_methods(cls, prefix):
    for name in list(vars(cls)):
        if name.startswith("__"):
            continue
        orig = vars(cls)[name]
        if not callable(orig):
            continue
        def make(o, label):
            def wrapped(*a, **k):
                CALLS[label] = CALLS.get(label, 0) + 1
                return o(*a, **k)
            return wrapped
        setattr(cls, name, make(orig, prefix + "." + name))


wrap_all_methods(tonic_vigor.TonicVigor, "MECH-320:TonicVigor")
wrap_all_methods(blocked_agency.BlockedAgency, "blocked_agency:BlockedAgency")
wrap_all_methods(sef.SelectionEntropyFloor, "sd105:SelectionEntropyFloor")
_orig_ne = sef.normalized_entropy
def _ne(*a, **k):
    CALLS["sd105:normalized_entropy"] = CALLS.get("sd105:normalized_entropy", 0) + 1
    return _orig_ne(*a, **k)
sef.normalized_entropy = _ne
_orig_w = e1_deep.ContextMemory.write
def _w(*a, **k):
    CALLS["contextmemory:ContextMemory.write"] = CALLS.get("contextmemory:ContextMemory.write", 0) + 1
    return _orig_w(*a, **k)
e1_deep.ContextMemory.write = _w

env = CausalGridWorld(size=12, num_hazards=0, num_resources=0, subgoal_mode=True,
                      num_waypoints=3, seed=42, subgoal_arrival_position_check=True,
                      hazard_free_contamination_gate=True)
cfg = REEConfig.from_dims(body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
                          action_dim=env.action_dim, world_dim=32, z_goal_enabled=True,
                          use_world_encoder_skip=True, alpha_world=0.9)
agent = REEAgent(cfg)
agent.goal_state.config.use_hierarchical_goal_credit = True

obs_flat, _ = env.reset()
agent.act(obs_flat)
prev_z = agent._current_latent.z_world.detach().clone().reshape(-1).float()
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
    agent.notify_subgoal_attainment(info.get("transition_type", "none"),
                                    child_representation=prev_z)
    prev_z = agent._current_latent.z_world.detach().clone().reshape(-1).float()
    if done:
        break

print("REACH CHECK over 80 steps of the 884c loop (no agent.select_action):")
if not CALLS:
    print("  (no wrapped callable was reached)")
for label in sorted(CALLS):
    print("  %-46s %d" % (label, CALLS[label]))
print("UNREACHED of the 4 open corrupting entries:")
for pre in ("MECH-320", "blocked_agency", "sd105", "contextmemory"):
    if not any(k.startswith(pre) for k in CALLS):
        print("  %s -- NOT REACHED" % pre)
