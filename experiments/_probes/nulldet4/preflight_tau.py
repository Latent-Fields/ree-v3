"""Pre-flight (V3-EXQ-1105a): per-arm D_W z-scale from recorded harm-weight trajectories.
theta_final = sum of per-tick increments d_t (telescoping), so z_W = sum(d)/(sd(d) sqrt(n)).
Var(z_W) ~ tau_int = 1 + 2 sum_k rho_k (integrated autocorrelation of the increments).
For the M4 walk (iid increments) tau_int = 1 by construction. ASCII only."""
import json, math, sys
import numpy as np
P = "/Users/dgolden/REE_Working/REE_assembly/evidence/planning/probes/valuation/results/"
S = "/Users/dgolden/REE_Working/.scratch/breakthrough-20260924/nulldet2/results/"
def tau(d, kmax=60):
    d = np.asarray(d, float); d = d - d.mean(); n = len(d); v = (d * d).mean()
    if v <= 0: return float('nan'), []
    rho = [float((d[:-k] * d[k:]).mean() / v) for k in range(1, min(kmax, n // 4))]
    # initial-positive-sequence style truncation on pairs is overkill; plain window sum + first 5 lags
    return 1 + 2 * sum(rho), rho[:5]
rows = []
def arm(label, a):
    th = np.array([t["theta"][1] for t in a["ticks"]], float)
    d = np.array([t["dtheta"][1] for t in a["ticks"]], float)
    n = len(d); sd = d.std()
    z = th[-1] / (sd * math.sqrt(n)) if sd > 0 else float('nan')
    ti, r5 = tau(d)
    rows.append((label, n, round(float(sd), 4), round(float(th[-1]), 3), round(z, 2), round(ti, 2), [round(x, 2) for x in r5]))
d42 = json.load(open(P + "SMOKE_s42.json"))["arms"]; d45 = json.load(open(P + "SMOKE2_s45.json"))["arms"]
for k in ("M1RAW", "M2", "M4"): arm("s42 " + k, d42[k])
for k in d45:
    if k != "M0": arm("s45 " + k, d45[k])
for s in (66, 69):
    a = json.load(open(S + "NULLDET2_s%d.json" % s))
    arms = a.get("arms", a)
    for k in arms:
        if k in ("M0", "MAXHACK"): continue
        arm("s%d %s" % (s, k), arms[k])
print("label | n_ticks | sd_h | theta_harm_final | z_W(own sd) | tau_int | rho1..5")
for r in rows: print(" | ".join(str(x) for x in r))
