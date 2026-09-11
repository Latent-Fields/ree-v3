"""ARC-021 H1 pre-build probe 4: THE DECISIVE CONTROL.

The unfrozen calibration probe reads the control arm's calibration_gap at ~0.00 and its
attribution_auc at ~0.51, against a frozen-encoder reference of ~0.41 / ~0.69 (V3-EXQ-1011,
n=96). Two hypotheses explain that:
  (A) unfreezing the encoder genuinely destroys the readout's signal, in BOTH arms; or
  (B) this driver has a plumbing bug introduced while unfreezing.
They are distinguished by ONE measurement: run THIS driver's own `_run_cell` with the
encoder RE-FROZEN. If it reproduces V3-EXQ-1011's ~0.41 control gap, the plumbing is sound
and (A) holds. If it also reads ~0.00, the bug is mine.

Freezing is applied at `_encode` -- the one place the gradient path into the encoder exists
-- so everything else (heads, optimizers, clip, probes, verdict) runs on the SHIPPED code.
Scratch; not an experiment; writes no manifest.
"""
import sys, statistics, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import experiments._scratch.arc021_h1_encoder_unfrozen_drive_axis_instrument as D

SEEDS = [1001, 1008, 1015, 1022]

_orig_make = D._make_channels

def _frozen_make(arm, action_dim, base_encoder):
    ch = _orig_make(arm, action_dim, base_encoder)
    def frozen_encode(w0, w1, _c=ch):
        with torch.no_grad():
            if _c.arm == "SEPARATED":
                return (_c.enc_sensory(w0), _c.enc_forward(w0),
                        _c.enc_forward(w1), _c.enc_harm(w0))
            z = _c.encoder(w0); z1 = _c.encoder(w1)
            return z, z, z1, z
    ch._encode = frozen_encode
    return ch

print("REGIME      arm         seed   calibration_gap  attribution_auc  harm_sens  zharm_disp")
res = {}
for regime, maker in (("FROZEN", _frozen_make), ("UNFROZEN", _orig_make)):
    D._make_channels = maker
    for arm in ("SEPARATED", "MERGED"):
        gaps, aucs = [], []
        for seed in SEEDS:
            r = D._run_cell("DENSE", arm, seed)
            gaps.append(r["calibration_gap"]); aucs.append(r["attribution_auc"])
            print(f"{regime:11s} {arm:11s} {seed:5d}  {r['calibration_gap']:15.5f}  "
                  f"{r['attribution_auc']:15.5f}  {r['harm_action_sensitivity']:9.5f}  "
                  f"{r['zharm_dispersion']:10.5f}", flush=True)
        res[(regime, arm)] = (gaps, aucs)
D._make_channels = _orig_make

print("\n================ VERDICT ================")
for regime in ("FROZEN", "UNFROZEN"):
    for arm in ("SEPARATED", "MERGED"):
        g, a = res[(regime, arm)]
        print(f"{regime:9s} {arm:11s} mean_gap={statistics.fmean(g):+.5f} "
              f"mean_auc={statistics.fmean(a):.5f}  (floor {D.SEPARATED_SIGNAL_FLOOR} on the control gap)")
fg = statistics.fmean(res[("FROZEN", "SEPARATED")][0])
ug = statistics.fmean(res[("UNFROZEN", "SEPARATED")][0])
print(f"\nFrozen control gap reproduces V3-EXQ-1011's DENSE 0.4147? measured {fg:+.5f}")
print("-> PLUMBING SOUND, unfreezing is the cause" if fg >= D.SEPARATED_SIGNAL_FLOOR
      else "-> FROZEN ALSO DEAD: the defect is in this driver, not in the unfreeze")
print(f"unfrozen/frozen control-gap ratio = {ug/fg if fg else float('nan'):.4f}")
