import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.v3_exq_981a_mech027_control_plane_pathological_modes import (
    build_config, ENV_SIZE, NUM_HAZARDS, NUM_RESOURCES, HAZARD_HARM, HAZARD_FIELD_DECAY,
    _train_warmup, WORLD_DIM,
)
import torch.optim as optim
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2

env = CausalGridWorldV2(seed=11, size=ENV_SIZE, num_hazards=NUM_HAZARDS,
                        num_resources=NUM_RESOURCES, hazard_harm=HAZARD_HARM,
                        hazard_field_decay=HAZARD_FIELD_DECAY)
cfg = build_config(env)
print("use_event_segmenter =", cfg.hippocampal.use_event_segmenter)
agent = REEAgent(cfg)
hip = agent.hippocampal
print("segment attrs:", [a for a in dir(hip) if 'segment' in a.lower()])
print("anchor attrs:", [a for a in dir(hip) if 'anchor' in a.lower()])
opt = optim.Adam(list(agent.e1.parameters()), lr=1e-3)
wf = optim.Adam(list(agent.e2.world_transition.parameters()) + list(agent.e2.world_action_encoder.parameters()), lr=1e-3)
he = optim.Adam(list(agent.e3.harm_eval_head.parameters()), lr=1e-4)
out = _train_warmup(agent, env, opt, wf, he, 20, 120, WORLD_DIM, torch.device("cpu"))
print("warmup sleep_fires:", out["sleep_fires"])
for name in ("anchor_set", "_anchor_set", "anchors"):
    aset = getattr(hip, name, None)
    if aset is None: continue
    print(name, type(aset))
    if hasattr(aset, "all_with_dual_trace"):
        print("  all_with_dual_trace ->", len(aset.all_with_dual_trace()))
    if hasattr(aset, "__len__"):
        print("  len ->", len(aset))
m = agent.force_sleep_cycle_at_eval_boundary()
print("mech285_n_draws =", (m or {}).get("mech285_n_draws"))
print("metrics keys:", sorted((m or {}).keys()))
