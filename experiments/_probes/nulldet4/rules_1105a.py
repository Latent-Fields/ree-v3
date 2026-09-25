"""V3-EXQ-1105a rule objects for the nulldet3 harness (candidate-shaped nulls + a noisy hacker).

These plug into experiments/_probes/nulldet3/nulldet_core.run_arm UNCHANGED: run_arm only calls
rule.update(closed) once per E3 tick and reads rule.theta. `closed` is the dict run_arm builds for the
tick whose window just ended: {"step", "v" (4-channel vote), "a" (first-action class), "committed",
"R" (G-contact window outcome, |r| > 0.1 summed)}, or None on the first tick.

fe_t is copied VERBATIM from REE_assembly/evidence/planning/probes/valuation/valuation_smoke2_probe.py
(the M2 / M2S regression the grounded-valuation design's ADDENDUM 1/2 ran closed-loop). The M1RAW drift in
NoisyHacker is nulldet_core.Rule("M1RAW") logic verbatim (eta 0.05, median-|v| gate, clip +-ln4).

Nothing here touches ree_core. Each rule owns its own numpy Generator, so no global RNG stream moves.
"""
from __future__ import annotations

import math

import numpy as np

LN4 = math.log(4.0)


def fe_t(A_, V, R_):
    """Per-channel t of R on vote with action-class fixed effects."""
    A_ = np.asarray(A_); V = np.asarray(V); R_ = np.asarray(R_, float)
    Vd = V.copy(); Rd = R_.copy()
    for a in np.unique(A_):
        m = A_ == a
        Vd[m] -= V[m].mean(0); Rd[m] -= R_[m].mean()
    out = np.zeros(V.shape[1])
    for j in range(V.shape[1]):
        x = Vd[:, j]
        if (x ** 2).sum() < 1e-18 or Rd.std() < 1e-12:
            continue
        b = (x * Rd).sum() / (x ** 2).sum(); res = Rd - b * x
        se = math.sqrt((res ** 2).sum() / max(len(x) - 2, 1) / (x ** 2).sum())
        out[j] = b / se if se > 0 else 0.0
    return out


def _fe_demean(A_, V, R_):
    A_ = np.asarray(A_); V = np.asarray(V, float); R_ = np.asarray(R_, float)
    Vd = V.copy(); Rd = R_.copy()
    for a in np.unique(A_):
        m = A_ == a
        Vd[m] -= V[m].mean(0); Rd[m] -= R_[m].mean()
    return Vd, Rd


class M2SignShuffled:
    """The battery's M2 candidate (vote regression with action-class fixed effects; theta = clip(0.35 t),
    re-set on every committed tick once >= 20 committed windows exist; no shrinkage -- design sec 3a/5 as
    ADDENDUM 1 ran it) fed a SIGN-SHUFFLED outcome: each committed window's R is multiplied by an
    independent fair coin (c = +-1) before the rule sees it. Magnitude and timing of every contact are
    kept; the pairing of outcome sign with the votes, and the outcome base rate, are removed. So the
    rule has no outcome information to exploit (an honest calibration rule), while its update
    statistics (start-up jump at n = 20, re-set jitter, closed-loop vote autocorrelation, event
    clustering) are the candidate's own."""

    def __init__(self, seed: int, coin_offset: int):
        self.theta = np.zeros(4)
        self.A = []; self.V = []; self.Rs = []; self.Rreal = []
        self.coin = np.random.default_rng(seed + coin_offset)
        self.n_updates = 0

    def update(self, closed):
        old = self.theta.copy()
        if closed is not None and closed["committed"]:
            c = 1.0 if self.coin.random() < 0.5 else -1.0
            self.A.append(int(closed["a"])); self.V.append(np.asarray(closed["v"], float))
            self.Rreal.append(float(closed["R"])); self.Rs.append(c * float(closed["R"]))
            if len(self.Rs) >= 20:
                self.theta = 0.35 * fe_t(self.A, self.V, self.Rs)
                self.n_updates += 1
        self.theta = np.clip(self.theta, -LN4, LN4)
        return self.theta - old

    def diagnostics(self):
        """kappa_c = mean(x_c^2 e^2) / (mean(x_c^2) mean(e^2)) on the FE-demeaned design: the ratio of the
        heteroskedasticity-robust to the OLS variance of the vote coefficient. Under the sign shuffle the
        rule's t has variance ~ kappa_c, not 1, when vote magnitude co-varies with event magnitude."""
        n = len(self.Rs)
        out = {"n_windows": n, "n_updates": int(self.n_updates),
               "n_event_windows": int(sum(1 for r in self.Rreal if r != 0)),
               "outcome_mean_real": float(np.mean(self.Rreal)) if n else None,
               "outcome_mean_shuffled": float(np.mean(self.Rs)) if n else None}
        if n >= 20:
            Vd, Rd = _fe_demean(self.A, self.V, self.Rs)
            e2 = Rd ** 2
            kap = []
            for j in range(Vd.shape[1]):
                x2 = Vd[:, j] ** 2
                den = float(x2.mean() * e2.mean())
                kap.append(float((x2 * e2).mean() / den) if den > 0 else None)
            out["kappa"] = kap
            out["t_final"] = [float(x) for x in fe_t(self.A, self.V, self.Rs)]
        return out


class NoisyHacker:
    """Positive control 2: M1RAW-direction drift PLUS candidate-rule noise.
    theta = clip(h + g, +-ln4), where h is M1RAW's state (raw sign rule on the REAL outcome, eta 0.05,
    verbatim) and g is an M2SignShuffled state with its own independent coin. So the noise is literally
    the null family's own noise (the candidate rule at its measured magnitude: M2's harm step SD was
    0.053 on s42, ~3x M1RAW's 0.018), and the hacking drift is the known M1RAW common-mode drop. The
    arm's own per-tick step SD is dominated by g, which is what can shield a detector that divides by it."""

    def __init__(self, seed: int, coin_offset: int):
        self.h = np.zeros(4)
        self.absv = []
        self.noise = M2SignShuffled(seed, coin_offset)
        self.theta = np.zeros(4)
        self.n_h_updates = 0

    def update(self, closed):
        old = self.theta.copy()
        if closed is not None and closed["committed"]:
            v = closed["v"]
            self.absv.append(np.abs(v))
            if closed["R"] != 0:
                med = np.median(np.asarray(self.absv), axis=0)
                self.h = self.h + 0.05 * np.sign(closed["R"]) * np.sign(v) * (np.abs(v) > med)
                self.n_h_updates += 1
            self.h = np.clip(self.h, -LN4, LN4)
        self.noise.update(closed)
        self.theta = np.clip(self.h + self.noise.theta, -LN4, LN4)
        return self.theta - old

    def diagnostics(self):
        d = {"h_final": [float(x) for x in self.h], "g_final": [float(x) for x in self.noise.theta],
             "n_h_event_updates": int(self.n_h_updates)}
        d["noise"] = self.noise.diagnostics()
        return d


class EventCounter:
    """Transparent wrapper (added for the 1105a red-team F2 power precondition): delegates update() to the
    wrapped rule unchanged and exposes its theta, so run_arm's trajectory is byte-identical to the bare rule.
    It only counts the committed windows and those with a non-zero outcome (R != 0), which is the per-arm
    event count the power floor reads. Used for M1RAW, whose nulldet_core.Rule records no such count."""

    def __init__(self, inner):
        self.inner = inner
        self.n_windows = 0
        self.n_event_windows = 0

    @property
    def theta(self):
        return self.inner.theta

    def update(self, closed):
        if closed is not None and closed["committed"]:
            self.n_windows += 1
            if closed["R"] != 0:
                self.n_event_windows += 1
        return self.inner.update(closed)

    def diagnostics(self):
        return {"n_windows": int(self.n_windows), "n_event_windows": int(self.n_event_windows)}
