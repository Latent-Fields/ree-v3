"""If all modes share ONE sampling seed, the CTRL centroids coincide (mode conditioning
is off in every cell), so dbar(CTRL)=0 and delta_dbar = dbar(ORACLE) is a pure,
non-negative, first-order oracle-induced separation. Does it clear the 0.02 floor?"""
import sys, statistics, importlib.util
sys.path.insert(0, "/Users/dgolden/REE_Working/ree-v3")
import torch
spec = importlib.util.spec_from_file_location(
    "x", "/Users/dgolden/REE_Working/ree-v3/experiments/v3_exq_1009_mech267_elite_channel_ceiling_spike.py")
X = importlib.util.module_from_spec(spec); spec.loader.exec_module(X)

# SHARED seed across modes: drop the mode offset.
X._MODE_OFFSET = {m: 0 for m in X.MODES}

simplex = X._maximally_separating_directions()
print(f"{'cell':22s} {'seed':>4s} {'dbar_CTRL':>11s} {'dbar_ORACLE':>12s} {'delta_dbar':>12s} {'reloc':>8s}")
agg = {}
for ao in X.AO_HEADS:
    for fl in X.CEM_FLOORS:
        lab = f"{ao}/{fl}"
        for seed in (0, 1, 2):
            r = X._run_cell(seed, ao, fl, simplex, X.GROUND_STEPS, 0, 8)
            agg.setdefault(lab, []).append(r)
            print(f"{lab:22s} {seed:>4d} {r['dbar_ctrl']:>11.6f} {r['dbar_oracle']:>12.6f} "
                  f"{r['delta_dbar']:>+12.6f} {r['relocation_ratio']:>8.5f}")
print()
print("MEANS:")
for lab, v in agg.items():
    md = statistics.fmean(x["delta_dbar"] for x in v)
    mc = statistics.fmean(x["dbar_ctrl"] for x in v)
    print(f"  {lab:22s} dbar_CTRL={mc:.6f}  delta_dbar={md:+.6f}  "
          f"clears_0.02={'YES' if md >= 0.02 else 'no'}  "
          f"raw_disp={statistics.fmean(x['raw_centroid_displacement'] for x in v):.6f}")
