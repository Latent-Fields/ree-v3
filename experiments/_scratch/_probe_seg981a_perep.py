import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch, torch.optim as optim
import experiments.v3_exq_981a_mech027_control_plane_pathological_modes as D
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2

env = CausalGridWorldV2(seed=11, size=D.ENV_SIZE, num_hazards=D.NUM_HAZARDS,
                        num_resources=D.NUM_RESOURCES, hazard_harm=D.HAZARD_HARM,
                        hazard_field_decay=D.HAZARD_FIELD_DECAY)
agent = REEAgent(D.build_config(env))
opt = optim.Adam(list(agent.e1.parameters()), lr=1e-3)
wf = optim.Adam(list(agent.e2.world_transition.parameters()) + list(agent.e2.world_action_encoder.parameters()), lr=1e-3)
he = optim.Adam(list(agent.e3.harm_eval_head.parameters()), lr=1e-4)
D._train_warmup(agent, env, opt, wf, he, int(sys.argv[1]) if len(sys.argv)>1 else 20, 120, D.WORLD_DIM, torch.device("cpu"))

# instrument: pool size at each forced firing
sizes = []
orig = agent.force_sleep_cycle_at_eval_boundary
def spy():
    aset = agent.hippocampal.anchor_set
    sizes.append(len(aset.all_anchors()) if aset is not None else -1)
    return orig()
agent.force_sleep_cycle_at_eval_boundary = spy

zg = D.ZGoalStreamAccumulator() if hasattr(D, "ZGoalStreamAccumulator") else None
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
zg = ZGoalStreamAccumulator()
out = D._run_eval_block(agent, env, "EVAL_BASELINE", 8, 120, D.WORLD_DIM, torch.device("cpu"), zg)
print("pool size at each firing:", sizes)
print("mech285_n_draws per episode:", out.get("sleep_mech285_draws"))
dr = out.get("sleep_mech285_draws") or []
print("MIN across firings =", min(dr) if dr else None, " (gate floor = 1)")
