# Probe archived with the V3-EXQ-1005 refusal record (written by red-team reviewer (fable), 2026-09-07).
# See REE_assembly/evidence/planning/exq1005_mech267_location_dv_redteam_blocking_20260907.md
# Throwaway substrate probe: run from ree-v3 root; writes nothing under evidence/.
"""Red-team probe for V3-EXQ-1005: how much can ELITE CHOICE alone move the centroid DV?
Monkeypatches _support_preserving_elite_indices with an oracle chooser.
"""
import sys, math, statistics, itertools
sys.path.insert(0, "/Users/dgolden/REE_Working/ree-v3")
sys.path.insert(0, "/Users/dgolden/REE_Working/ree-v3/experiments")
import torch
import importlib.util
spec = importlib.util.spec_from_file_location(
    "exq1005", "/Users/dgolden/REE_Working/ree-v3/experiments/v3_exq_1005_mech267_mode_content_location_separation.py")
X = importlib.util.module_from_spec(spec); spec.loader.exec_module(X)

def run_cell(seed, arm, mode, elite_oracle=None, record=None, sample_seed=None):
    torch.manual_seed(seed)
    hip = X._make_hippocampal(arm)
    torch.manual_seed(seed + 900_000)
    z_world = torch.randn(1, X.WORLD_DIM); z_self = torch.randn(1, X.SELF_DIM)
    orig = hip._support_preserving_elite_indices
    def patched(trajectories, scores_tensor, elite_indices):
        idx, diag = orig(trajectories=trajectories, scores_tensor=scores_tensor, elite_indices=elite_indices)
        if record is not None:
            ao = torch.stack([t.get_action_object_sequence() for t in trajectories])  # [n,1,H,ao]
            acts = torch.stack([t.actions for t in trajectories])  # [n,1,H,a]
            record.append({
                "ao_across_cand_std_mean": float(ao.std(dim=0).mean()),
                "ao_across_cand_std_max": float(ao.std(dim=0).max()),
                "act_across_cand_std_mean": float(acts.std(dim=0).mean()),
                "argsort_top": torch.argsort(scores_tensor)[:len(idx)].tolist(),
                "substrate_elites": idx.tolist(),
                "sp_active": diag.get("support_preserving_elite_active"),
                "classes_before": diag.get("support_preserving_elite_classes_before"),
            })
        if elite_oracle is not None:
            return elite_oracle(trajectories, scores_tensor, elite_indices), diag
        return idx, diag
    hip._support_preserving_elite_indices = patched
    torch.manual_seed(X._cell_sampling_seed(seed, arm, mode) if sample_seed is None else sample_seed)
    trajs = hip.propose_trajectories(z_world, z_self, num_candidates=X.NUM_CANDIDATES, operating_mode={mode: 1.0})
    st = hip.get_last_propose_diagnostics()["action_object_decoder_raw_output_stats"]
    itd = hip.get_last_propose_diagnostics()["cem_iteration_diagnostics"]
    return st["mean_by_action_dim"], st["std_by_action_dim"], itd, hip

def oracle_dir(u, sign):
    def f(trajectories, scores_tensor, elite_indices):
        k = int(elite_indices.numel())
        proj = torch.stack([ (t.actions.mean(dim=(0,1)) * u).sum() for t in trajectories ])
        order = torch.argsort(proj, descending=(sign > 0))
        return order[:k]
    return f

torch.manual_seed(4242)
U = [torch.randn(X.ACTION_DIM) for _ in range(4)]
U = [u / u.norm() for u in U]

print("=== A. per-iteration internals, CTRL_OFF vs H2_ONLY, seed 0, mode internal_planning ===")
for arm in ["CTRL_OFF", "H2_ONLY", "NOISE_H3_REF"]:
    rec = []
    mu, sd, itd, hip = run_cell(0, arm, "internal_planning", record=rec)
    for i,(r,d) in enumerate(zip(rec, itd)):
        print(f"{arm:13s} iter{i} ao_std_in(mean/min/max)=({d['ao_std_mean']:.3f},{d['ao_std_min']:.3f},{d['ao_std_max']:.3f}) "
              f"E2-ao across-cand std mean={r['ao_across_cand_std_mean']:.4f} max={r['ao_across_cand_std_max']:.4f} "
              f"act across-cand std={r['act_across_cand_std_mean']:.4f} argsort_top={r['argsort_top']} substrate_elites={r['substrate_elites']} sp_active={r['sp_active']} classes={r['classes_before']}")
    print(f"   final mean_by_action_dim={[round(v,4) for v in mu]} std={[round(v,4) for v in sd]}")

print()
print("=== B. ORACLE ELITE BOUND: same sampling seed, elites pushed +u vs -u at every iteration -> d(A,B) from elite choice ALONE ===")
for seed in range(3):
    ds = []
    for u in U:
        muA, sdA, _, _ = run_cell(seed, "CTRL_OFF", "internal_planning", elite_oracle=oracle_dir(u, +1), sample_seed=12345+seed)
        muB, sdB, _, _ = run_cell(seed, "CTRL_OFF", "internal_planning", elite_oracle=oracle_dir(u, -1), sample_seed=12345+seed)
        ds.append(X._location_separation(muA, sdA, muB, sdB))
        print(f"  seed{seed} u={[round(float(v),2) for v in u]} d_oracle(same eps)={ds[-1]:.5f} |dmu|={[round(a-b,5) for a,b in zip(muA,muB)]}")
    print(f"  seed{seed} mean d_oracle(same eps) = {statistics.fmean(ds):.5f}")

print()
print("=== C. ORACLE dbar vs CTRL_OFF dbar under the SCRIPT's per-mode seeds (best case for a content arm) ===")
for seed in range(3):
    cells_ctrl = {}; cells_orc = {}
    for mi, mode in enumerate(X.MODES):
        mu, sd, _, _ = run_cell(seed, "CTRL_OFF", mode)
        cells_ctrl[mode] = (mu, sd)
        # each mode gets its OWN oracle direction (max content separation a re-ranker could produce)
        mu2, sd2, _, _ = run_cell(seed, "CTRL_OFF", mode, elite_oracle=oracle_dir(U[mi], +1))
        cells_orc[mode] = (mu2, sd2)
    def dbar(cells):
        return statistics.fmean(X._location_separation(cells[a][0], cells[a][1], cells[b][0], cells[b][1]) for a,b in X.MODE_PAIRS)
    dc, do = dbar(cells_ctrl), dbar(cells_orc)
    bs = lambda cells: max(statistics.fmean(cells[m][1]) for m in X.MODES) - min(statistics.fmean(cells[m][1]) for m in X.MODES)
    print(f"  seed{seed} dbar CTRL_OFF={dc:.5f} dbar ORACLE={do:.5f} delta={do-dc:+.5f}  breadth_spread ctrl={bs(cells_ctrl):.5f} oracle={bs(cells_orc):.5f}")

print()
print("=== D. READOUT-ONLY scale leak: CTRL_OFF final-iteration samples, rescale deviations about the sample mean, decode, compare ===")
# capture final-iteration samples via _decode_action_objects wrapper
for seed in range(2):
    torch.manual_seed(seed); hip = X._make_hippocampal("CTRL_OFF")
    torch.manual_seed(seed + 900_000); z_world = torch.randn(1, X.WORLD_DIM); z_self = torch.randn(1, X.SELF_DIM)
    samples = []
    orig_dec = hip._decode_action_objects
    def dec(ao, _o=orig_dec):
        samples.append(ao.detach().clone()); return _o(ao)
    hip._decode_action_objects = dec
    torch.manual_seed(X._cell_sampling_seed(seed, "CTRL_OFF", "internal_planning"))
    hip.propose_trajectories(z_world, z_self, num_candidates=X.NUM_CANDIDATES, operating_mode={"internal_planning": 1.0})
    print(f"  seed{seed} n_decode_calls={len(samples)} (expect 48 = 3 iters x 16 if nothing else decodes)")
    last = torch.stack(samples[-16:])  # [16,1,H,ao]
    mu_ao = last.mean(dim=0, keepdim=True)
    dev = last - mu_ao
    def centroid(s):
        acts = torch.stack([orig_dec((mu_ao + s*dev)[k]) for k in range(16)])  # [16,1,H,a]
        flat = acts.reshape(-1, X.ACTION_DIM)
        return flat.mean(0).tolist(), flat.std(0, unbiased=False).tolist()
    m1, s1 = centroid(1.0); m03, s03 = centroid(0.3); m13, s13 = centroid(1.3)
    print(f"  seed{seed} centroid s=1.0 {[round(v,4) for v in m1]} | s=0.3 {[round(v,4) for v in m03]} | s=1.3 {[round(v,4) for v in m13]}")
    print(f"  seed{seed} d(s=1.3 vs s=0.3, IDENTICAL eps, IDENTICAL ao_mean) = {X._location_separation(m13,s13,m03,s03):.5f}   d(1.0 vs 0.3)={X._location_separation(m1,s1,m03,s03):.5f}")
