import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch, torch.optim as optim
from experiments.v3_exq_981a_mech027_control_plane_pathological_modes import (
    build_config, ENV_SIZE, NUM_HAZARDS, NUM_RESOURCES, HAZARD_HARM,
    HAZARD_FIELD_DECAY, _train_warmup, WORLD_DIM,
)
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2

env = CausalGridWorldV2(seed=11, size=ENV_SIZE, num_hazards=NUM_HAZARDS,
                        num_resources=NUM_RESOURCES, hazard_harm=HAZARD_HARM,
                        hazard_field_decay=HAZARD_FIELD_DECAY)
cfg = build_config(env)
print("use_event_segmenter =", cfg.hippocampal.use_event_segmenter)
print("use_anchor_sets     =", getattr(cfg.hippocampal, "use_anchor_sets", None))
agent = REEAgent(cfg)
hip = agent.hippocampal
print("event_segmenter obj =", type(hip.event_segmenter).__name__ if hip.event_segmenter else None)
print("anchor_set obj      =", type(hip.anchor_set).__name__ if hip.anchor_set else None)

CNT = {"tick_anchor_set": 0, "tick_with_events": 0, "n_events": 0, "write_anchor": 0}
orig_tick = hip.tick_anchor_set
def spy_tick(latent, events, **kw):
    CNT["tick_anchor_set"] += 1
    n = len(events) if events else 0
    if n:
        CNT["tick_with_events"] += 1
        CNT["n_events"] += n
    return orig_tick(latent, events, **kw)
hip.tick_anchor_set = spy_tick

aset = hip.anchor_set
if aset is not None:
    orig_wa = aset.write_anchor
    def spy_wa(*a, **k):
        CNT["write_anchor"] += 1
        return orig_wa(*a, **k)
    aset.write_anchor = spy_wa

opt = optim.Adam(list(agent.e1.parameters()), lr=1e-3)
wf = optim.Adam(list(agent.e2.world_transition.parameters()) + list(agent.e2.world_action_encoder.parameters()), lr=1e-3)
he = optim.Adam(list(agent.e3.harm_eval_head.parameters()), lr=1e-4)
_train_warmup(agent, env, opt, wf, he, 20, 120, WORLD_DIM, torch.device("cpu"))
print("COUNTS:", CNT)
seg = hip.event_segmenter
print("segmenter public attrs:", [a for a in dir(seg) if not a.startswith('_')][:30])
