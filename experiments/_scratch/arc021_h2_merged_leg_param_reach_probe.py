#!/opt/local/bin/python3
"""ARC-021 H2 leg -- per-loss PARAMETER REACH probe (writes nothing, ~1 min on a worker).

Why this exists (2026-09-24, science-20260924-arc021-h2-merged-leg): the H2 leg was to be
authored from experiments/v3_spark_arc021_three_loop_scale.py, whose MERGED arm puts one Adam
over agent.parameters() with combined loss E1 + E2 + harm, against SEPARATE (three optimizers).
This probe asks which parameters each of those three losses can actually move.

Measured on ree-v3 351f0364 (REEConfig.large, detach_carried_prev_action=True, seed 42, after
12 untrained steps with record_transition so both replay buffers are populated):

    E1 compute_prediction_loss -> {'e1': 28}
    E2 compute_e2_loss         -> {'e2': 8}
    harm on z_world.detach()   -> {'e3': 4}
    harm on z_world (undet.)   -> {'latent_stack': 41, 'e3': 4,
                                   'body_obs_encoder': 2, 'world_obs_encoder': 2}

The reach sets are PAIRWISE DISJOINT. Structural reason: E1 and E2 train by replay on
DETACHED buffered latents (agent.py _world_experience_buffer / _self_experience_buffer append
z.detach().clone(); _e2_transition_buffer stores z_self_t.detach()), so neither loss reaches
the encoder, and neither reaches e3. Adam is elementwise per parameter and the driver applies no
gradient clipping, so one optimizer over a SUM of losses with disjoint parameter reach produces
exactly the same per-parameter update as three separate optimizers. The spark's MERGED arm
therefore differs from SEPARATE only in (i) learning rates (1e-3 everywhere vs E1 1e-4 / E2 3e-4)
and (ii) whether the harm gradient reaches the encoder (the 993a autopsy's repair 4). With
repair 4 applied symmetrically the optimizer-topology manipulation cannot reach any DV:
/queue-experiment Step 2.5d INERT. Record:
REE_assembly/evidence/planning/arc021_h2_leg_refused_merge_inert_disjoint_param_reach_20260924.md
"""
import sys, random, importlib.util, collections
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("spark", str(REPO / "experiments" / "v3_spark_arc021_three_loop_scale.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import torch, torch.nn.functional as F
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig
torch.manual_seed(42); random.seed(42)
env = m._make_env(42)
cfg = REEConfig.large(body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim, action_dim=env.action_dim, reafference_action_dim=env.action_dim)
cfg.detach_carried_prev_action = True
agent = REEAgent(cfg); wd = cfg.latent.world_dim; agent.train()
_, obs = env.reset(); agent.reset()
zs = None
for s in range(12):  # a few UNTRAINED steps so both replay buffers fill
    if agent._current_latent is not None: zs = agent._current_latent.z_self.detach().clone()
    latent = agent.sense(torch.as_tensor(obs["body_state"], dtype=torch.float32), torch.as_tensor(obs["world_state"], dtype=torch.float32))
    ticks = agent.clock.advance()
    e1p = agent._e1_tick(latent) if ticks["e1_tick"] else torch.zeros(1, wd)
    a = agent.select_action(agent.generate_trajectories(latent, e1p, ticks), ticks)
    if zs is not None: agent.record_transition(zs, a, latent.z_self.detach())
    _, r, done, info, obs = env.step(a)
print("e2 buffer len", len(agent._e2_transition_buffer), "world buffer len", len(agent._world_experience_buffer))
def reach(loss):
    agent.zero_grad(set_to_none=True)
    if not loss.requires_grad: return "NO GRAD"
    loss.backward(retain_graph=True)
    c = collections.Counter()
    for n, p in agent.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0: c[n.split(".")[0]] += 1
    return dict(c)
tgt = torch.tensor([[0.0]])
print("E1 compute_prediction_loss ->", reach(agent.compute_prediction_loss()))
print("E2 compute_e2_loss        ->", reach(agent.compute_e2_loss()))
print("harm on z_world.detach()  ->", reach(F.mse_loss(agent.e3.harm_eval(latent.z_world.detach()), tgt)))
print("harm on z_world (undet.)  ->", reach(F.mse_loss(agent.e3.harm_eval(latent.z_world), tgt)))
