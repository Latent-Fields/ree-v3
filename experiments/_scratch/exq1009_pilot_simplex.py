"""Does the simplex oracle make delta_dbar a first-order, non-negative bound?
Compares simplex vs the OLD independent-random directions, same cells, same seeds."""
import sys, statistics, importlib.util
sys.path.insert(0, "/Users/dgolden/REE_Working/ree-v3")
import torch
spec = importlib.util.spec_from_file_location(
    "x", "/Users/dgolden/REE_Working/ree-v3/experiments/v3_exq_1009_mech267_elite_channel_ceiling_spike.py")
X = importlib.util.module_from_spec(spec); spec.loader.exec_module(X)

simplex = X._maximally_separating_directions()
g = torch.Generator().manual_seed(4242)
randdirs = [torch.randn(X.ACTION_DIM, generator=g) for _ in X.MODES]
randdirs = [d / d.norm() for d in randdirs]

print(f"{'cell':22s} {'seed':>4s}  {'delta_dbar SIMPLEX':>19s} {'delta_dbar RANDOM':>18s}  {'reloc':>8s}  {'raw_disp':>10s}")
agg = {}
for ao in X.AO_HEADS:
    for fl in X.CEM_FLOORS:
        lab = f"{ao}/{fl}"
        for seed in (0, 1):
            rs = X._run_cell(seed, ao, fl, simplex, X.GROUND_STEPS, 0, 8)
            rr = X._run_cell(seed, ao, fl, randdirs, X.GROUND_STEPS, 0, 8)
            agg.setdefault(lab, []).append((rs["delta_dbar"], rr["delta_dbar"],
                                            rs["relocation_ratio"], rs["raw_centroid_displacement"]))
            print(f"{lab:22s} {seed:>4d}  {rs['delta_dbar']:>+19.6f} {rr['delta_dbar']:>+18.6f}  "
                  f"{rs['relocation_ratio']:>8.5f}  {rs['raw_centroid_displacement']:>10.6f}")
print()
print("MEANS over seeds:")
for lab, v in agg.items():
    ms = statistics.fmean(x[0] for x in v); mr = statistics.fmean(x[1] for x in v)
    print(f"  {lab:22s} simplex={ms:+.6f}  random={mr:+.6f}  "
          f"clears0.02(simplex)={'YES' if ms>=0.02 else 'no'}  "
          f"raw_disp={statistics.fmean(x[3] for x in v):.6f}")
