"""Render the SD-ZWORLD-SENSE-PATH-PARITY probe results as a readable table.

Read-only: takes the JSON `zworld_sense_path_parity_probe.py` wrote and prints the
per-seed arm accuracies, the paired per-(seed, split) contrasts, and the 1030 comparison.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ARMS_ORDER = [
    "R_direct_raw", "R_direct_wobs", "R_direct_fresh_random", "R_direct_raw_ema",
    "R_direct_wobs_ema", "R_sense", "T_direct_wobs", "T_direct_raw", "T_sense",
    "R_wobs_output", "T_wobs_output", "ctrl_world_state",
    "ctrl_world_state_fresh_random", "ctrl_raw_slice",
]
CONDS = ("unnormalised", "standardised", "unnormalised_long")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    for p in args.paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        print("=" * 92)
        print("ARM %s   seeds=%s   split_seeds=%s   p0_episodes=%s   collect_episodes=%s"
              % (d["arm"], d["seeds"], d["per_seed"][0]["split_seeds"],
                 d.get("p0_episodes"), d.get("collect_episodes")))
        s = d["summary"]
        print("\nPER-ARM MEAN HELD-OUT ACCURACY (5 seeds x 10 splits = 50 fits per cell)")
        print("  %-32s %-9s %-9s %-9s | %s" % ("arm", "unnorm", "std", "unnorm10x", "train acc (unnorm/std)"))
        for a in ARMS_ORDER:
            if a not in s["arm_means"]["unnormalised"]:
                continue
            vals = [s["arm_means"][c].get(a) for c in CONDS]
            tr = []
            for c in ("unnormalised", "standardised"):
                t = [cell["probes"][a][c].get("mean_train_accuracy") for cell in d["per_seed"]
                     if cell["probes"][a][c].get("mean_train_accuracy") is not None]
                tr.append(statistics.fmean(t) if t else float("nan"))
            print("  %-32s %s | %.3f / %.3f"
                  % (a, " ".join("%.4f   " % v if v is not None else "  n/a    " for v in vals),
                     tr[0], tr[1]))
        print("\n  max between-SPLIT spread within one arm+seed: %s"
              % {c: round(s["max_split_spread"][c], 4) for c in CONDS})

        print("\nPAIRED PER-(SEED, SPLIT) CONTRASTS -- mean +/- sd, frac of 50 cells with a>b")
        for c in CONDS:
            print("  -- %s" % c)
            for k, v in s["paired"][c].items():
                print("     %-32s %+.4f +/- %.4f   frac=%.2f   range [%+.3f, %+.3f]"
                      % (k, v["mean"], v["sd"], v["frac_a_greater"], v["min"], v["max"]))

        print("\nPER-SEED, the headline arms (unnormalised / unnormalised_long):")
        for cell in d["per_seed"]:
            row = []
            for a in ("R_direct_raw", "R_sense", "R_direct_wobs", "T_direct_wobs", "T_sense"):
                row.append("%s=%.3f/%.3f" % (a, cell["probes"][a]["unnormalised"]["mean_accuracy"],
                                             cell["probes"][a]["unnormalised_long"]["mean_accuracy"]))
            print("  seed %d n=%d  %s" % (cell["seed"], cell["n_samples"], "  ".join(row)))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
