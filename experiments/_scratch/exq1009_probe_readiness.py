"""Step 2.5a readiness probe for V3-EXQ-1009 (throwaway; writes nothing).

Four questions, all of which must be answered before the driver is written:
  Q1 HippocampalConfig accepts + lands the two CEM floor knobs.
  Q2 floor OFF actually removes the iteration>=1 clamp (ao_std_min != 0.2).
  Q3 _support_preserving_elite_indices is CALLED in all four cells (the oracle patch point).
  Q4 the 817a-style world-effect grounding materially raises E2's ACROSS-CANDIDATE
     action-object std -- the load-bearing readiness fact. If it does not, the
     "grounded" cells are vacuous and cannot support a "no cell clears" verdict.
"""
import sys
from pathlib import Path

REPO = "/Users/dgolden/REE_Working/ree-v3"
sys.path.insert(0, REPO)

import torch
import torch.nn.functional as F
from torch import optim

from ree_core.hippocampal.module import HippocampalModule
from ree_core.utils.config import HippocampalConfig, E2Config, ResidueConfig
from ree_core.predictors.e2_fast import E2FastPredictor
from ree_core.residue.field import ResidueField

WORLD_DIM, SELF_DIM, ACTION_DIM, AO_DIM = 32, 16, 4, 16
NUM_CANDIDATES, HORIZON, NUM_CEM_ITERATIONS = 16, 4, 3


def make(floor_on: bool, seed: int = 0):
    torch.manual_seed(seed)
    e2 = E2FastPredictor(E2Config(self_dim=SELF_DIM, world_dim=WORLD_DIM,
                                  action_dim=ACTION_DIM, action_object_dim=AO_DIM,
                                  hidden_dim=64))
    res = ResidueField(ResidueConfig(world_dim=WORLD_DIM, hidden_dim=32,
                                     num_basis_functions=8))
    kw = dict(world_dim=WORLD_DIM, action_dim=ACTION_DIM, action_object_dim=AO_DIM,
              hidden_dim=32, horizon=HORIZON, num_candidates=NUM_CANDIDATES,
              num_cem_iterations=NUM_CEM_ITERATIONS,
              mode_conditioning_enabled=False, mode_value_weight={},
              mode_partitioned_cem=False)
    if not floor_on:
        kw.update(use_support_preserving_cem=False,
                  support_preserving_stratified_elites=False,
                  support_preserving_ao_std_floor=0.0)
    cfg = HippocampalConfig(**kw)
    return HippocampalModule(cfg, e2, res), cfg, e2


# --- Q1 -------------------------------------------------------------------
print("=== Q1: knobs accepted and landed ===")
for on in (True, False):
    _, cfg, _ = make(on)
    print(f"  floor_on={on}: use_sp_cem={cfg.use_support_preserving_cem} "
          f"strat={cfg.support_preserving_stratified_elites} "
          f"floor={cfg.support_preserving_ao_std_floor}")
    assert cfg.use_support_preserving_cem is on
    assert float(cfg.support_preserving_ao_std_floor) == (0.2 if on else 0.0)
print("  OK")


# --- grounding (817a world-effect objective, applied to this bench) --------
def collect_transitions(e2, n=2048, seed=7):
    """(z_t, a_t, z_{t+1}) with z_{t+1} = e2.world_forward(z_t, a_t)."""
    g = torch.Generator().manual_seed(seed)
    zt = torch.randn(n, WORLD_DIM, generator=g)
    # one-hot-ish actions spanning the action space, as the decoder produces
    a = torch.randn(n, ACTION_DIM, generator=g)
    with torch.no_grad():
        znext = e2.world_forward(zt, a)
    return zt, a, znext


def projection(src_dim, ao_dim):
    g = torch.Generator().manual_seed(4242)
    R = (torch.randn(src_dim, max(ao_dim, src_dim), generator=g)[:, :ao_dim]
         if src_dim < ao_dim else torch.randn(src_dim, ao_dim, generator=g))
    Q, _ = torch.linalg.qr(R)
    return Q[:, :ao_dim]


def ground_head(e2, n_steps=600, lr=1e-3, seed=11):
    zt, a, znext = collect_transitions(e2, seed=seed)
    zs = znext.std(dim=0, keepdim=True)
    zs = torch.where(zs > 1e-8, zs, torch.ones_like(zs))
    Z = (znext - znext.mean(dim=0, keepdim=True)) / zs
    Q = projection(Z.shape[1], AO_DIM)
    T = Z @ Q
    ts = T.std(dim=0, keepdim=True)
    ts = torch.where(ts > 1e-8, ts, torch.ones_like(ts))
    T = (T - T.mean(dim=0, keepdim=True)) / ts
    opt = optim.Adam(list(e2.action_object_head.parameters()), lr=lr)
    g2 = torch.Generator().manual_seed(seed + 11000)
    first = last = float("nan")
    for it in range(n_steps):
        idx = torch.randint(0, zt.shape[0], (256,), generator=g2)
        o = e2.action_object(zt[idx].detach(), a[idx].detach())
        loss = F.mse_loss(o, T[idx].detach())
        if it == 0:
            first = float(loss.item())
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss.item())
    return first, last


def across_candidate_ao_std(e2, seed=0):
    """The quantity the elite channel's leverage is bounded by: how much E2's
    recomputed action objects vary ACROSS candidates at fixed z_world."""
    g = torch.Generator().manual_seed(seed + 555)
    zw = torch.randn(1, WORLD_DIM, generator=g).expand(NUM_CANDIDATES, -1)
    acts = torch.randn(NUM_CANDIDATES, ACTION_DIM, generator=g)
    with torch.no_grad():
        o = e2.action_object(zw, acts)          # [n_cand, ao_dim]
    return float(o.std(dim=0).mean()), float(o.std(dim=0).max())


print()
print("=== Q4: does grounding raise ACROSS-CANDIDATE action-object std? ===")
for seed in (0, 1, 2):
    _, _, e2 = make(True, seed=seed)
    pre_mean, pre_max = across_candidate_ao_std(e2, seed)
    f, l = ground_head(e2, seed=11 + seed)
    post_mean, post_max = across_candidate_ao_std(e2, seed)
    print(f"  seed{seed}: ao across-cand std mean {pre_mean:.5f} -> {post_mean:.5f} "
          f"(x{post_mean / max(pre_mean, 1e-12):.1f})  max {pre_max:.5f} -> {post_max:.5f}  "
          f"| grounding MSE {f:.4f} -> {l:.4f}")


# --- Q2 + Q3 --------------------------------------------------------------
print()
print("=== Q2/Q3: clamp behaviour and oracle patch-point reachability ===")
for floor_on in (True, False):
    hip, cfg, e2 = make(floor_on, seed=0)
    calls = []
    orig = hip._support_preserving_elite_indices

    def patched(trajectories, scores_tensor, elite_indices, _o=orig, _c=calls):
        idx, diag = _o(trajectories=trajectories, scores_tensor=scores_tensor,
                       elite_indices=elite_indices)
        _c.append(int(idx.numel()))
        return idx, diag

    hip._support_preserving_elite_indices = patched
    torch.manual_seed(900_000)
    zw = torch.randn(1, WORLD_DIM); zs_ = torch.randn(1, SELF_DIM)
    torch.manual_seed(12345)
    hip.propose_trajectories(zw, zs_, num_candidates=NUM_CANDIDATES,
                             operating_mode={"internal_planning": 1.0})
    itd = hip.get_last_propose_diagnostics()["cem_iteration_diagnostics"]
    stds = [(round(d["ao_std_min"], 4), round(d["ao_std_max"], 4)) for d in itd]
    print(f"  floor_on={floor_on}: elite-fn calls={len(calls)} (expect {NUM_CEM_ITERATIONS}) "
          f"| per-iter (ao_std_min, ao_std_max) = {stds}")

print()
print("PROBE DONE")
