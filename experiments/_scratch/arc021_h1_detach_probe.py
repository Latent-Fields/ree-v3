"""ARC-021 H1 pre-build probe 2: MEASURE the design decision, do not argue it.

With the encoder unfrozen, both MSE terms admit the trivial global optimum "encoder emits a
constant" (z == c, head(c) == c -> loss 0). If that attractor is real, the latent collapses
in EVERY arm, the control arm produces no signal, and the run is non-contributory BY
CONSTRUCTION -- a design that cannot answer its own question. The driver therefore takes
DETACHED MSE targets. This probe measures whether that attractor is real rather than
asserting it, so the choice is evidenced.

Compares, at identical seed and identical data, four variants x both criterion arms:
  detach_both  (the shipped design)     sensory target sg(z), forward target sg(z')
  detach_none                            both targets differentiable
DV: per-dimension std of z over a fixed probe batch (-> 0 under collapse), plus the harm
head's action sensitivity, which is the statistic the load-bearing criteria route on.
Scratch; not an experiment; writes no manifest.
"""
import sys, statistics, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import experiments._scratch.arc021_h1_encoder_unfrozen_drive_axis_instrument as D
import torch.nn.functional as F

EPISODES = 25          # short but long enough for a collapse attractor to bite
STEPS = 120


def _run(arm, detach, seed=1301, cond="DENSE"):
    torch.manual_seed(seed)
    import random as _r; _r.seed(seed)
    env = D.SmallViewEnv(size=D.GRID_SIZE, num_hazards=D.NUM_HAZARDS[cond], seed=seed)
    enc, dec = D._make_world_codec(env.world_obs_dim)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=D.LR_ENCODER)
    # short P0 so both variants start from the same warmed encoder
    for _ in range(10):
        _, o = env.reset()
        for _s in range(STEPS):
            w = o["world_state"].unsqueeze(0)
            loss = F.mse_loss(dec(enc(w)), w)
            opt.zero_grad(); loss.backward(); opt.step()
            _, _h, dn, _i, o = env.step(_r.randint(0, env.action_dim - 1))
            if dn: break
    ch = D._make_channels(arm, env.action_dim, enc)

    if not detach:
        # UNDETACHED VARIANT: make both MSE targets differentiable. Patch at the two places
        # the driver detaches -- the module flag (sensory target) and `_encode`'s no_grad
        # (forward target) -- so the comparison isolates exactly the detach decision.
        orig_encode = ch._encode
        def undetached_encode(w0, w1, _c=ch):
            if _c.arm == "SEPARATED":
                z_s = _c.enc_sensory(w0); z_f = _c.enc_forward(w0)
                z_f1 = _c.enc_forward(w1)                 # NO no_grad
                return z_s, z_f, z_f1, _c.enc_harm(w0)
            z = _c.encoder(w0); z1 = _c.encoder(w1)       # NO no_grad
            return z, z, z1, z
        ch._encode = undetached_encode

    prev = D.DETACH_MSE_TARGETS
    D.DETACH_MSE_TARGETS = bool(detach)
    try:
        for _ in range(EPISODES):
            _, o = env.reset()
            for _s in range(STEPS):
                w0 = o["world_state"]
                a = _r.randint(0, env.action_dim - 1)
                _, h, dn, _i, o = env.step(a)
                w1 = o["world_state"]
                lbl = torch.ones(1, 1) if h < 0 else torch.zeros(1, 1)
                ch.train_step(w0.unsqueeze(0), D._action_onehot(a, env.action_dim),
                              w1.unsqueeze(0), lbl)
                if dn: break
    finally:
        D.DETACH_MSE_TARGETS = prev

    ch.eval_mode()
    with torch.no_grad():
        probe = torch.cat([env.reset()[1]["world_state"].unsqueeze(0) for _ in range(64)], dim=0)
        z = ch.harm_encoder(probe)
        disp = float(z.std(dim=0).mean().item())
        absm = float(z.abs().mean().item())
    sens = D._harm_action_sensitivity(ch.harm_encoder, ch, env, env.action_dim)
    return disp, absm, sens


print(f"{EPISODES} episodes x {STEPS} steps per cell\n")
print(f"{'arm':22s} {'variant':12s} {'z_dispersion':>13s} {'z_abs_mean':>11s} {'harm_action_sens':>17s}")
res = {}
for arm in ("SEPARATED", "MERGED"):
    for detach, name in ((True, "detach_both"), (False, "detach_none")):
        d, a, s = _run(arm, detach)
        res[(arm, name)] = (d, a, s)
        print(f"{arm:22s} {name:12s} {d:13.6f} {a:11.6f} {s:17.6f}")
print()
print("FLOOR for reference: HARM_ACTION_SENSITIVITY_FLOOR =", D.HARM_ACTION_SENSITIVITY_FLOOR)
for arm in ("SEPARATED", "MERGED"):
    dd = res[(arm, "detach_both")][0]; dn = res[(arm, "detach_none")][0]
    print(f"{arm:22s} dispersion ratio detach_none/detach_both = {dn/dd if dd else float('nan'):.4f}")
