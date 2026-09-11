"""Analysis for the ARC-021 H1 unfreeze-DOSE sweep (`arc021_h1_unfreeze_dose_sweep.py`).

Merges the shard files and evaluates, AT EVERY DOSE, the two halves of the spike's question
against the criteria pre-registered in the sweep's own docstring BEFORE it ran:

  (S) does the instrument survive -- the DRIVER's own gates, recomputed with the DRIVER's own
      constants over the control arm, exactly as `_build_preconditions` computes them;
  (M) is the manipulation still real -- function-space drift, cross-arm readout divergence,
      and the internally-calibrated merge-vs-channel-shaping ratio.

Every dose is printed, INCLUDING the failing ones, so the table cannot be read as cherry-
picked. The merge-vs-separate contrast is printed too, and is NOT an input to any verdict.

Scratch; writes no manifest.
"""
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import experiments._scratch.arc021_h1_encoder_unfrozen_drive_axis_instrument as D
import experiments._scratch.arc021_h1_unfreeze_dose_sweep as S

HERE = Path(__file__).parent


def load() -> List[Dict[str, Any]]:
    """Prefer the condensed `.rows.json` (what is committed); fall back to raw shards.

    The raw shards carry a 64x32 latent matrix per cell -- 10 MB of `_z_end` across 320
    cells, retained only long enough to compute the pairwise cross-arm divergence (M-B).
    The condenser attaches that scalar to BOTH rows of each arm pair as
    `cross_arm_fn_div_rel` and drops the matrices, so every statistic in this file stays
    re-derivable from the committed artifact without keeping the raw latents in git.
    """
    condensed = HERE / "arc021_h1_unfreeze_dose_sweep.rows.json"
    if condensed.exists():
        return json.loads(condensed.read_text())["rows"]
    rows: List[Dict[str, Any]] = []
    for p in sorted(HERE.glob("arc021_h1_unfreeze_dose_sweep.shard*of*.json")):
        if ".pilot." in p.name:
            continue
        rows.extend(json.loads(p.read_text())["rows"])
    return rows


def _m(vals) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def _fmt(v, spec="%+.5f"):
    return "None" if v is None else (spec % v)


def main() -> int:
    rows = load()
    doses = sorted({r["dose_k"] for r in rows})
    print(f"{len(rows)} cells loaded; doses={doses}; "
          f"seeds={sorted({r['seed'] for r in rows})}; "
          f"conditions={sorted({r['condition'] for r in rows})}\n")

    out: Dict[str, Any] = {"doses": [], "n_cells": len(rows)}

    # ---- (S) SURVIVAL: the driver's own gates, control arm only ---------------------------
    print("=" * 108)
    print("(S) DOES THE INSTRUMENT SURVIVE -- CONTROL ARM (SEPARATED) ONLY, driver's own gates")
    print("=" * 108)
    print(f"{'dose':>4} {'gap_DENSE':>10} {'gap_SPARSE':>11} {'S1':>4} "
          f"{'range':>8} {'S2':>4} {'max_abs':>8} {'S3':>4} "
          f"{'minAUC-0.5':>11} {'S4':>4} {'meanAUC':>8}  SURVIVES")
    print(f"     {'(>=0.20)':>10} {'(>=0.20)':>11}      {'(>=0.30)':>8}      {'(>=0.40)':>8}"
          f"      {'(>=0.105)':>11}")
    surv: Dict[int, bool] = {}
    for dose in doses:
        ctl = [r for r in rows if r["dose_k"] == dose and r["arm"] == "SEPARATED"]
        gaps = [r["calibration_gap"] for r in ctl if r["calibration_gap"] is not None]
        aucs = [r["attribution_auc"] for r in ctl if r["attribution_auc"] is not None]
        gd = _m([r["calibration_gap"] for r in ctl if r["condition"] == "DENSE"])
        gs = _m([r["calibration_gap"] for r in ctl if r["condition"] == "SPARSE"])
        s1 = (gd is not None and gd >= D.SEPARATED_SIGNAL_FLOOR
              and gs is not None and gs >= D.SEPARATED_SIGNAL_FLOOR)
        rng = (max(gaps) - min(gaps)) if gaps else None
        s2 = rng is not None and rng >= 2.0 * D.MARGIN
        mab = max(abs(g) for g in gaps) if gaps else None
        s3 = mab is not None and mab >= 2.0 * D.SEPARATED_SIGNAL_FLOOR
        fh = (min(aucs) - 0.5) if aucs else None
        s4 = fh is not None and fh >= D.MARGIN_AUC
        ok = bool(s1 and s2 and s3 and s4)
        surv[dose] = ok
        print(f"{dose:>4} {_fmt(gd):>10} {_fmt(gs):>11} {'Y' if s1 else 'n':>4} "
              f"{_fmt(rng, '%.5f'):>8} {'Y' if s2 else 'n':>4} {_fmt(mab, '%.5f'):>8} "
              f"{'Y' if s3 else 'n':>4} {_fmt(fh, '%+.5f'):>11} {'Y' if s4 else 'n':>4} "
              f"{_fmt(_m(aucs), '%.4f'):>8}  {'SURVIVES' if ok else '--'}")
        out["doses"].append({"dose_k": dose, "control_gap_dense": gd, "control_gap_sparse": gs,
                             "S1_nondegeneracy": s1, "S2_range": rng, "S2_met": s2,
                             "S3_max_abs": mab, "S3_met": s3, "S4_auc_floor_headroom": fh,
                             "S4_met": s4, "control_mean_auc": _m(aucs), "survives": ok})

    # ---- (M) MANIPULATION REALITY ---------------------------------------------------------
    print("\n" + "=" * 108)
    print("(M) IS THE MANIPULATION STILL REAL -- function space, normalised by ||z_P0||")
    print("=" * 108)
    print(f"{'dose':>4} {'M-A merged':>11} {'sep_drift':>10} {'M-B x-arm':>10} "
          f"{'interchan':>10} {'M-C ratio':>10} {'M-A ok':>7} {'M-C ok':>7}  REAL")
    real: Dict[int, bool] = {}
    for dose in doses:
        sub = [r for r in rows if r["dose_k"] == dose]
        ma = _m([r["enc_fn_drift_rel"] for r in sub if r["arm"] == "MERGED"])
        sd = _m([r["enc_fn_drift_rel"] for r in sub if r["arm"] == "SEPARATED"])
        inter = _m([r["interchannel_fn_div_rel"] for r in sub if r["arm"] == "SEPARATED"])
        xs = []
        for cond in sorted({r["condition"] for r in sub}):
            for seed in sorted({r["seed"] for r in sub}):
                m = next((r for r in sub if r["condition"] == cond and r["seed"] == seed
                          and r["arm"] == "MERGED"), None)
                s = next((r for r in sub if r["condition"] == cond and r["seed"] == seed
                          and r["arm"] == "SEPARATED"), None)
                if not m or not s:
                    continue
                if "cross_arm_fn_div_rel" in m:          # condensed artifact
                    xs.append(m["cross_arm_fn_div_rel"])
                else:                                    # raw shards, latents still present
                    d = math.sqrt(sum((a - b) ** 2 for a, b in zip(m["_z_end"], s["_z_end"])))
                    xs.append(d / m["z_denom"])
        mb = _m(xs)
        mc = (mb / inter) if (mb is not None and inter not in (None, 0.0)) else None
        ok_a = ma is not None and ma >= S.MANIP_DRIFT_FLOOR
        ok_c = mc is not None and mc >= S.MANIP_RATIO_FLOOR
        ok = bool(ok_a and ok_c)
        real[dose] = ok
        print(f"{dose:>4} {_fmt(ma, '%.5f'):>11} {_fmt(sd, '%.5f'):>10} {_fmt(mb, '%.5f'):>10} "
              f"{_fmt(inter, '%.5f'):>10} {_fmt(mc, '%.4f'):>10} {'Y' if ok_a else 'n':>7} "
              f"{'Y' if ok_c else 'n':>7}  {'REAL' if ok else '--'}")
        e = next(d for d in out["doses"] if d["dose_k"] == dose)
        e.update({"MA_merged_fn_drift": ma, "separated_fn_drift": sd,
                  "MB_cross_arm_fn_div": mb, "interchannel_fn_div": inter,
                  "MC_merge_vs_channel_ratio": mc, "MA_met": ok_a, "MC_met": ok_c,
                  "manipulation_real": ok})

    # ---- the intersection -- THE SPIKE'S QUESTION -----------------------------------------
    both = [d for d in doses if surv[d] and real[d]]
    print("\n" + "=" * 108)
    print(f"SURVIVING doses:        {[d for d in doses if surv[d]]}")
    print(f"MANIPULATION-REAL doses:{[d for d in doses if real[d]]}")
    print(f"INTERSECTION:           {both}   -> ANSWER: "
          f"{'a workable dose EXISTS' if both else 'NO WORKABLE DOSE'}")
    print("=" * 108)
    out["surviving_doses"] = [d for d in doses if surv[d]]
    out["manipulation_real_doses"] = [d for d in doses if real[d]]
    out["intersection"] = both

    # ---- (X) cross-channel gradient + the clip asymmetry -----------------------------------
    print("\n(X) SUPPORTING -- cross-channel gradient into the HARM-READOUT encoder, at P1 end")
    print("    (g_other is STRUCTURALLY ZERO for SEPARATED: that is this decomposition's own"
          " positive control)")
    print(f"{'dose':>4} | {'SEP g_other/g_harm':>19} {'SEP clip_frac':>13} | "
          f"{'MRG g_other/g_harm':>19} {'MRG cos(o,h)':>12} {'MRG clip_frac':>13}")
    for dose in doses:
        sub = [r for r in rows if r["dose_k"] == dose]
        row_out = {}
        vals = []
        for arm in ("SEPARATED", "MERGED"):
            a = [r for r in sub if r["arm"] == arm]
            ratio = _m([(r["xgrad_at_end"] or {}).get("g_other_over_harm") for r in a])
            cos = _m([(r["xgrad_at_end"] or {}).get("cos_other_harm") for r in a])
            clip = _m([r["clip_active_frac_unfrozen"] for r in a])
            vals.append((ratio, cos, clip))
            row_out[arm] = {"g_other_over_harm": ratio, "cos_other_harm": cos,
                            "clip_active_frac_unfrozen": clip}
        (sr, _sc, sclip), (mr, mc_, mclip) = vals
        print(f"{dose:>4} | {_fmt(sr, '%.6f'):>19} {_fmt(sclip, '%.4f'):>13} | "
              f"{_fmt(mr, '%.6f'):>19} {_fmt(mc_, '%+.4f'):>12} {_fmt(mclip, '%.4f'):>13}")
        next(d for d in out["doses"] if d["dose_k"] == dose)["xgrad_clip"] = row_out

    # ---- the contrast: REPORTED, NEVER an input to dose selection --------------------------
    print("\nMERGE-vs-SEPARATE CONTRAST (paired MERGED - SEPARATED). REPORTED ONLY.")
    print("  It is NOT an input to any verdict above: a contrast that varies with dose is"
          " EVIDENCE THE")
    print("  CONTRAST IS A TEST OF THE DOSE, which is the thing this spike is asking about.")
    print(f"{'dose':>4} {'d_gap DENSE':>12} {'d_gap SPARSE':>13} {'d_auc DENSE':>12} "
          f"{'d_auc SPARSE':>13}")
    for dose in doses:
        sub = [r for r in rows if r["dose_k"] == dose]
        cells = {}
        for cond in sorted({r["condition"] for r in sub}):
            for dv in ("calibration_gap", "attribution_auc"):
                ds = []
                for seed in sorted({r["seed"] for r in sub}):
                    m = next((r for r in sub if r["condition"] == cond and r["seed"] == seed
                              and r["arm"] == "MERGED"), None)
                    s = next((r for r in sub if r["condition"] == cond and r["seed"] == seed
                              and r["arm"] == "SEPARATED"), None)
                    if m and s and m[dv] is not None and s[dv] is not None:
                        ds.append(m[dv] - s[dv])
                cells[(cond, dv)] = _m(ds)
        print(f"{dose:>4} {_fmt(cells.get(('DENSE','calibration_gap'))):>12} "
              f"{_fmt(cells.get(('SPARSE','calibration_gap'))):>13} "
              f"{_fmt(cells.get(('DENSE','attribution_auc'))):>12} "
              f"{_fmt(cells.get(('SPARSE','attribution_auc'))):>13}")
        next(d for d in out["doses"] if d["dose_k"] == dose)["contrast"] = {
            f"{k[0]}_{k[1]}": v for k, v in cells.items()}

    p = HERE / "arc021_h1_unfreeze_dose_sweep.analysis.json"
    p.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__" and "--addendum" not in sys.argv:
    raise SystemExit(main())


# =============================================================================================
# ADDENDUM (written after the sweep, and labelled as such): two decompositions the first pass
# did not carry, both forced by what the sweep actually showed.
#
#   (1) THE ENCODER-LEVEL INCREMENT. At DOSE 0 the encoder is frozen in BOTH arms, so the
#       arms differ ONLY in trunk topology -- which is exactly V3-EXQ-993a/1011's comparison,
#       already adjudicated null. Any contrast at dose 0 is therefore the TRUNK-level effect,
#       not the encoder-level one this leg is posed against. The encoder-level increment is
#       contrast(dose) - contrast(0), and it is the only part of the contrast the leg's own
#       name ("H-encoder-level-merge-degrades") refers to.
#   (2) SIGN CONSISTENCY of the paired contrast, per dose and condition -- the leg's own null
#       clause names it ("no sign consistency"), and a direction held by 15 of 16 cells is a
#       different object from the same mean held by 9 of 16.
#
# Run:  python3 arc021_h1_unfreeze_dose_analyse.py --addendum
# =============================================================================================
def addendum() -> int:
    rows = load()
    doses = sorted({r["dose_k"] for r in rows})
    conds = sorted({r["condition"] for r in rows})

    def paired(dose, cond, dv):
        ds = []
        sub = [r for r in rows if r["dose_k"] == dose and r["condition"] == cond]
        for seed in sorted({r["seed"] for r in sub}):
            m = next((r for r in sub if r["seed"] == seed and r["arm"] == "MERGED"), None)
            s = next((r for r in sub if r["seed"] == seed and r["arm"] == "SEPARATED"), None)
            if m and s and m[dv] is not None and s[dv] is not None:
                ds.append(m[dv] - s[dv])
        return ds

    base = {(c, dv): _m(paired(0, c, dv))
            for c in conds for dv in ("calibration_gap", "attribution_auc")}

    print("ENCODER-LEVEL INCREMENT = contrast(dose) - contrast(dose 0).")
    print("  dose 0 = both arms frozen = the TRUNK-level contrast V3-EXQ-993a/1011 already")
    print("  measured and adjudicated null. MARGIN (the leg's own pre-registered effect) = "
          f"{D.MARGIN}.\n")
    hdr = f"{'dose':>4}"
    for c in conds:
        hdr += f" | {c+' d_gap':>13} {'incr':>9} {'incr/MARGIN':>11} {'sign':>6}"
    print(hdr)
    for dose in doses:
        line = f"{dose:>4}"
        for c in conds:
            ds = paired(dose, c, "calibration_gap")
            mean = _m(ds)
            incr = (mean - base[(c, "calibration_gap")]) if mean is not None else None
            npos = sum(1 for d in ds if d > 0)
            line += (f" | {_fmt(mean):>13} {_fmt(incr):>9} "
                     f"{_fmt((incr / D.MARGIN) if incr is not None else None, '%.2f'):>11} "
                     f"{str(npos)+'/'+str(len(ds)):>6}")
        print(line)
    print("\n  'sign' = cells with MERGED > SEPARATED. ARC-021's H1 predicts MERGED < SEPARATED;")
    print("  every dose in both conditions runs the OTHER way.")
    return 0


if __name__ == "__main__" and "--addendum" in sys.argv:
    raise SystemExit(addendum())
