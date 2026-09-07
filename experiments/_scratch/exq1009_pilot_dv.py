"""Pilot: the actual 2x2 DV (oracle-elite centroid ceiling) at 1-2 seeds.
Reuses the archived probe-C structure verbatim in substance. Writes nothing."""
import sys, math, statistics, itertools
sys.path.insert(0, "/Users/dgolden/REE_Working/ree-v3")
import torch
import torch.nn.functional as F
from torch import optim
from ree_core.hippocampal.module import HippocampalModule
from ree_core.utils.config import HippocampalConfig, E2Config, ResidueConfig
from ree_core.predictors.e2_fast import E2FastPredictor
from ree_core.residue.field import ResidueField

WORLD_DIM, SELF_DIM, ACTION_DIM, AO_DIM = 32, 16, 4, 16
NUM_CANDIDATES, HORIZON, NUM_CEM_ITERATIONS = 16, 4, 3
MODES = ["internal_planning", "external_task", "internal_replay", "offline_consolidation"]
MODE_PAIRS = list(itertools.combinations(MODES, 2))
_MODE_OFFSET = {m: i * 7_919 for i, m in enumerate(MODES)}


def _cell_sampling_seed(seed, mode):
    return seed * 104_729 + _MODE_OFFSET[mode]


def _location_separation(mu_a, sd_a, mu_b, sd_b):
    if not (len(mu_a) == len(sd_a) == len(mu_b) == len(sd_b)) or not mu_a:
        return None
    acc = []
    for i in range(len(mu_a)):
        pv = 0.5 * (float(sd_a[i]) ** 2 + float(sd_b[i]) ** 2)
        if not math.isfinite(pv) or pv <= 0.0:
            return None
        acc.append((float(mu_a[i]) - float(mu_b[i])) ** 2 / pv)
    return math.sqrt(sum(acc) / len(acc))


def _projection(src, dst):
    g = torch.Generator().manual_seed(4242)
    R = (torch.randn(src, max(dst, src), generator=g)[:, :dst]
         if src < dst else torch.randn(src, dst, generator=g))
    Q, _ = torch.linalg.qr(R)
    return Q[:, :dst]


def _ground_delta(e2, seed, n_steps=800, lr=1e-3, n=2048):
    g = torch.Generator().manual_seed(seed + 31_000)
    zt = torch.randn(n, WORLD_DIM, generator=g)
    a = torch.randn(n, ACTION_DIM, generator=g)
    with torch.no_grad():
        raw = e2.world_forward(zt, a) - zt
    s = raw.std(dim=0, keepdim=True)
    s = torch.where(s > 1e-8, s, torch.ones_like(s))
    T = ((raw - raw.mean(dim=0, keepdim=True)) / s) @ _projection(WORLD_DIM, AO_DIM)
    ts = T.std(dim=0, keepdim=True)
    ts = torch.where(ts > 1e-8, ts, torch.ones_like(ts))
    T = (T - T.mean(dim=0, keepdim=True)) / ts
    opt = optim.Adam(list(e2.action_object_head.parameters()), lr=lr)
    g2 = torch.Generator().manual_seed(seed + 41_000)
    last = float("nan")
    for _ in range(n_steps):
        idx = torch.randint(0, n, (256,), generator=g2)
        loss = F.mse_loss(e2.action_object(zt[idx].detach(), a[idx].detach()),
                          T[idx].detach())
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss.item())
    return last


def _acs(e2, seed):
    g = torch.Generator().manual_seed(seed + 555)
    zw = torch.randn(1, WORLD_DIM, generator=g).expand(NUM_CANDIDATES, -1)
    acts = torch.randn(NUM_CANDIDATES, ACTION_DIM, generator=g)
    with torch.no_grad():
        o = e2.action_object(zw, acts)
    return float(o.std(dim=0).mean())


def make(seed, grounded, floor_on):
    torch.manual_seed(seed)
    e2 = E2FastPredictor(E2Config(self_dim=SELF_DIM, world_dim=WORLD_DIM,
                                  action_dim=ACTION_DIM, action_object_dim=AO_DIM,
                                  hidden_dim=64))
    res = ResidueField(ResidueConfig(world_dim=WORLD_DIM, hidden_dim=32,
                                     num_basis_functions=8))
    acs_pre = _acs(e2, seed)
    mse = None
    if grounded:
        mse = _ground_delta(e2, seed)
    acs_post = _acs(e2, seed)
    kw = dict(world_dim=WORLD_DIM, action_dim=ACTION_DIM, action_object_dim=AO_DIM,
              hidden_dim=32, horizon=HORIZON, num_candidates=NUM_CANDIDATES,
              num_cem_iterations=NUM_CEM_ITERATIONS,
              mode_conditioning_enabled=False, mode_value_weight={},
              mode_partitioned_cem=False)
    if not floor_on:
        kw.update(use_support_preserving_cem=False,
                  support_preserving_stratified_elites=False,
                  support_preserving_ao_std_floor=0.0)
    return HippocampalModule(HippocampalConfig(**kw), e2, res), acs_pre, acs_post, mse


def oracle_dir(u):
    def f(trajectories, scores_tensor, elite_indices):
        k = int(elite_indices.numel())
        proj = torch.stack([(t.actions.mean(dim=(0, 1)) * u).sum() for t in trajectories])
        return torch.argsort(proj, descending=True)[:k]
    return f


def run_cell(seed, grounded, floor_on, mode, elite_oracle=None):
    hip, acs_pre, acs_post, mse = make(seed, grounded, floor_on)
    orig = hip._support_preserving_elite_indices

    def patched(trajectories, scores_tensor, elite_indices):
        idx, diag = orig(trajectories=trajectories, scores_tensor=scores_tensor,
                         elite_indices=elite_indices)
        if elite_oracle is not None:
            return elite_oracle(trajectories, scores_tensor, elite_indices), diag
        return idx, diag

    hip._support_preserving_elite_indices = patched
    torch.manual_seed(seed + 900_000)
    zw = torch.randn(1, WORLD_DIM); zs_ = torch.randn(1, SELF_DIM)
    torch.manual_seed(_cell_sampling_seed(seed, mode))
    hip.propose_trajectories(zw, zs_, num_candidates=NUM_CANDIDATES,
                             operating_mode={mode: 1.0})
    st = hip.get_last_propose_diagnostics()["action_object_decoder_raw_output_stats"]
    return st["mean_by_action_dim"], st["std_by_action_dim"], acs_pre, acs_post, mse


torch.manual_seed(4242)
U = [torch.randn(ACTION_DIM) for _ in range(len(MODES))]
U = [u / u.norm() for u in U]

print("cell                        seed  dbar_CTRL  dbar_ORACLE   DELTA     acs_pre->post   mse")
for grounded in (False, True):
    for floor_on in (True, False):
        for seed in (0, 1):
            ctrl, orc = {}, {}
            meta = None
            for mi, mode in enumerate(MODES):
                mu, sd, ap, aq, ms = run_cell(seed, grounded, floor_on, mode)
                ctrl[mode] = (mu, sd); meta = (ap, aq, ms)
                mu2, sd2, _, _, _ = run_cell(seed, grounded, floor_on, mode,
                                             elite_oracle=oracle_dir(U[mi]))
                orc[mode] = (mu2, sd2)

            def dbar(c):
                vals = [_location_separation(c[a][0], c[a][1], c[b][0], c[b][1])
                        for a, b in MODE_PAIRS]
                return statistics.fmean(v for v in vals if v is not None) if all(
                    v is not None for v in vals) else None
            dc, do = dbar(ctrl), dbar(orc)
            ap, aq, ms = meta
            name = f"{'GROUND' if grounded else 'FROZEN'}/{'floor0.2' if floor_on else 'floor0.0'}"
            dstr = f"{do - dc:+.5f}" if (dc is not None and do is not None) else "  None "
            print(f"{name:26s}  {seed}   "
                  f"{('%.5f' % dc) if dc is not None else ' None ':>9s}  "
                  f"{('%.5f' % do) if do is not None else ' None ':>9s}   {dstr}   "
                  f"{ap:.4f}->{aq:.4f}   {('%.4f' % ms) if ms is not None else '  -  '}")
