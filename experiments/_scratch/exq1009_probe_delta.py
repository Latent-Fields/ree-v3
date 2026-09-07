"""Q4b: does grounding on the world-effect DELTA (not the absolute next state)
raise E2's across-candidate action-object std? Throwaway; writes nothing."""
import sys
sys.path.insert(0, "/Users/dgolden/REE_Working/ree-v3")
import torch
import torch.nn.functional as F
from torch import optim
from ree_core.utils.config import E2Config
from ree_core.predictors.e2_fast import E2FastPredictor

WORLD_DIM, SELF_DIM, ACTION_DIM, AO_DIM, NUM_CANDIDATES = 32, 16, 4, 16, 16


def mk(seed):
    torch.manual_seed(seed)
    return E2FastPredictor(E2Config(self_dim=SELF_DIM, world_dim=WORLD_DIM,
                                    action_dim=ACTION_DIM,
                                    action_object_dim=AO_DIM, hidden_dim=64))


def projection(src, dst):
    g = torch.Generator().manual_seed(4242)
    R = (torch.randn(src, max(dst, src), generator=g)[:, :dst]
         if src < dst else torch.randn(src, dst, generator=g))
    Q, _ = torch.linalg.qr(R)
    return Q[:, :dst]


def ground(e2, mode, n_steps=800, lr=1e-3, seed=11, n=2048):
    g = torch.Generator().manual_seed(seed)
    zt = torch.randn(n, WORLD_DIM, generator=g)
    a = torch.randn(n, ACTION_DIM, generator=g)
    with torch.no_grad():
        znext = e2.world_forward(zt, a)
    raw = znext if mode == "absolute" else (znext - zt)   # <-- the world EFFECT
    s = raw.std(dim=0, keepdim=True)
    s = torch.where(s > 1e-8, s, torch.ones_like(s))
    Z = (raw - raw.mean(dim=0, keepdim=True)) / s
    T = Z @ projection(Z.shape[1], AO_DIM)
    ts = T.std(dim=0, keepdim=True)
    ts = torch.where(ts > 1e-8, ts, torch.ones_like(ts))
    T = (T - T.mean(dim=0, keepdim=True)) / ts
    opt = optim.Adam(list(e2.action_object_head.parameters()), lr=lr)
    g2 = torch.Generator().manual_seed(seed + 11000)
    first = last = float("nan")
    for it in range(n_steps):
        idx = torch.randint(0, n, (256,), generator=g2)
        loss = F.mse_loss(e2.action_object(zt[idx].detach(), a[idx].detach()),
                          T[idx].detach())
        if it == 0:
            first = float(loss.item())
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss.item())
    return first, last


def acs(e2, seed):
    """Across-candidate action-object std at a FIXED z_world (pure action-dependence)."""
    g = torch.Generator().manual_seed(seed + 555)
    zw = torch.randn(1, WORLD_DIM, generator=g).expand(NUM_CANDIDATES, -1)
    acts = torch.randn(NUM_CANDIDATES, ACTION_DIM, generator=g)
    with torch.no_grad():
        o = e2.action_object(zw, acts)
    return float(o.std(dim=0).mean())


def action_dependence_ratio(e2, seed):
    """Across-ACTION std / across-STATE std -- how much of o's variance the action carries."""
    g = torch.Generator().manual_seed(seed + 777)
    zw_fixed = torch.randn(1, WORLD_DIM, generator=g).expand(64, -1)
    acts_var = torch.randn(64, ACTION_DIM, generator=g)
    zw_var = torch.randn(64, WORLD_DIM, generator=g)
    a_fixed = torch.randn(1, ACTION_DIM, generator=g).expand(64, -1)
    with torch.no_grad():
        o_a = e2.action_object(zw_fixed, acts_var)      # action varies
        o_s = e2.action_object(zw_var, a_fixed)         # state varies
    return float(o_a.std(dim=0).mean()) / max(float(o_s.std(dim=0).mean()), 1e-12)


print("mode      seed  acs_pre   acs_post   x      adr_pre  adr_post  mse_first->last")
for mode in ("absolute", "delta"):
    for seed in (0, 1, 2):
        e2 = mk(seed)
        pre, adr_pre = acs(e2, seed), action_dependence_ratio(e2, seed)
        f, l = ground(e2, mode, seed=11 + seed)
        post, adr_post = acs(e2, seed), action_dependence_ratio(e2, seed)
        print(f"{mode:9s} {seed}     {pre:.5f}   {post:.5f}   "
              f"x{post/max(pre,1e-12):.2f}  {adr_pre:.4f}   {adr_post:.4f}    "
              f"{f:.3f}->{l:.4f}")
