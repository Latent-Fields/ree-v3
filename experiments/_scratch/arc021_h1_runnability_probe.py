"""ARC-021 H1 pre-build probe 1: RUNNABILITY of all three arms with the encoder UNFROZEN.

The H2 leg was blocked because its MERGED arm crashed DETERMINISTICALLY ON STEP 2
(ContextMemory writes an nn.Parameter in place via .data, invalidating the retained
cross-step graph). The 993a/1022 driver family imports no ree_core.agent and no predictor
module, so that mechanism cannot reproduce -- but "cannot reproduce" is a prediction and
this is the measurement. Multi-step, because a single step cannot see a retained-graph bug.
Scratch; not an experiment; writes no manifest.
"""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import experiments._scratch.arc021_h1_encoder_unfrozen_drive_axis_instrument as D

def probe(arm, n_steps=6):
    torch.manual_seed(0)
    env = D.SmallViewEnv(size=D.GRID_SIZE, num_hazards=15, seed=7)
    enc, dec = D._make_world_codec(env.world_obs_dim)
    ch = D._make_channels(arm, env.action_dim, enc)
    _, obs = env.reset()
    grads_seen = {"encoder": False}
    for i in range(n_steps):
        w0 = obs["world_state"]
        a = i % 4
        _, h, done, _info, obs = env.step(a)
        w1 = obs["world_state"]
        lbl = torch.ones(1,1) if h < 0 else torch.zeros(1,1)
        st = ch.train_step(w0.unsqueeze(0), D._action_onehot(a, env.action_dim), w1.unsqueeze(0), lbl)
        # confirm the ENCODER actually received gradient -- the whole point of the unfreeze
        gp = [p.grad for p in ch.harm_encoder.parameters() if p.grad is not None]
        if gp and any(float(g.abs().sum()) > 0 for g in gp):
            grads_seen["encoder"] = True
        if done:
            _, obs = env.reset()
    return st, grads_seen

print("arm                    runnable  encoder_grad  pre_clip_norm  harm_loss")
ok_all = True
for arm in D.ARMS:
    try:
        st, g = probe(arm)
        print(f"{arm:22s} {'True':8s}  {str(g['encoder']):12s}  {st['pre_clip_norm']:13.5f}  {st['harm_loss']:.5f}")
        if not g["encoder"]:
            ok_all = False
            print(f"   *** {arm}: NO ENCODER GRADIENT -- the unfreeze is INERT ***")
    except Exception as e:
        ok_all = False
        import traceback; traceback.print_exc()
        print(f"{arm:22s} CRASH: {type(e).__name__}: {e}")
print()
print("SUBSTRATE VERDICT:", "RUNNABLE (all arms, encoder gradient live)" if ok_all else "BLOCKED")
