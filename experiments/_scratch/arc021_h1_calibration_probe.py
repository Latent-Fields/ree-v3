"""ARC-021 H1 pre-build probe 3: measure the UNFROZEN regime's control-arm signal and
paired-diff SD, so MARGIN_AUC and SEEDS are sized from THIS regime rather than inherited
from the frozen-encoder one (993a autopsy section 8 repair 5).

Run THROUGH THE DRIVER'S OWN `_run_cell` -- never a probe subclass. V3-EXQ-993a red-team F3:
its design probe consumed the torch RNG differently before head construction, so its numbers
were a different head-init draw and the whole control-arm measurement had to be redone.

Seeds are DISJOINT from the run's own SEEDS so nothing calibrated here is later scored.
Scratch; not an experiment; writes no manifest.
"""
import sys, json, statistics, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import experiments._scratch.arc021_h1_encoder_unfrozen_drive_axis_instrument as D

PROBE_SEEDS = [1001 + 7 * i for i in range(8)]
assert not (set(PROBE_SEEDS) & set(D.SEEDS)), "probe seeds must be disjoint from run seeds"

t0 = time.time()
rows = []
for cond in D.CONDITIONS:
    for seed in PROBE_SEEDS:
        for arm in ("SEPARATED", "MERGED"):
            rows.append(D._run_cell(cond, arm, seed))
elapsed = time.time() - t0

out = {"elapsed_s": elapsed, "n_cells": len(rows), "seeds": PROBE_SEEDS, "rows": rows}
p = Path(__file__).with_suffix(".result.json")
p.write_text(json.dumps(out, indent=1, default=str))

print("\n================ PROBE SUMMARY ================", flush=True)
print(f"{len(rows)} cells in {elapsed:.1f}s ({elapsed/len(rows):.2f}s/cell)")
for dv in D.DVS:
    for cond in D.CONDITIONS:
        sep = {r["seed"]: r[dv] for r in rows if r["condition"] == cond and r["arm"] == "SEPARATED" and r.get(dv) is not None}
        mer = {r["seed"]: r[dv] for r in rows if r["condition"] == cond and r["arm"] == "MERGED" and r.get(dv) is not None}
        pair = sorted(set(sep) & set(mer))
        d = [mer[k] - sep[k] for k in pair]
        if len(d) < 2:
            print(f"{dv:18s} {cond:7s} INSUFFICIENT n={len(d)}"); continue
        print(f"{dv:18s} {cond:7s} n={len(d):2d} sep_mean={statistics.fmean(sep.values()):+.4f} "
              f"mer_mean={statistics.fmean(mer.values()):+.4f} "
              f"paired_mean={statistics.fmean(d):+.4f} paired_SD={statistics.stdev(d):.4f} "
              f"sep_min={min(sep.values()):+.4f} sep_max={max(sep.values()):+.4f}")
for key in ("harm_action_sensitivity", "zharm_dispersion", "clip_active_frac", "mean_pre_clip_norm", "p1_harm_events"):
    for arm in ("SEPARATED", "MERGED"):
        v = [r[key] for r in rows if r["arm"] == arm and r.get(key) is not None]
        if v:
            print(f"{key:24s} {arm:10s} min={min(v):.5f} mean={statistics.fmean(v):.5f} max={max(v):.5f}")
print("wrote", p, flush=True)
