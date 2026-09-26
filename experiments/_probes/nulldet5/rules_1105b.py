"""V3-EXQ-1105b rule objects for the nulldet3 harness (1105a's rules, with the hack eta as a parameter).

Plug into experiments/_probes/nulldet3/nulldet_core.run_arm UNCHANGED (run_arm calls rule.update(closed) once
per E3 tick and reads rule.theta; `closed` = {"step", "v", "a", "committed", "R"} or None on the first tick).

WHAT CHANGED FROM 1105a (autopsy failure_autopsy_V3-EXQ-1105a_2026-09-26.json, route A):
  - SignHack(eta) is nulldet_core.Rule("M1RAW") logic VERBATIM except that the per-event step 0.05 is the
    parameter `eta`. 1105b runs it at eta 0.10 so the drift saturates at -ln4 in half the events (1105a's
    drift stopped at -0.75 / -0.85 on the two event-sparse admitted seeds, 66 / 77 event windows).
    It also counts committed windows and R != 0 windows itself (1105a wrapped M1RAW in EventCounter).
  - NoisyHackerEta: theta = clip(h + g, +-ln4) with h = SignHack(eta) on the REAL outcome and g =
    rules_1105a.M2SignShuffled (imported, byte-identical) on an independent coin. So h is the P1 control's
    rule verbatim, as in 1105a.
The null rule (M2SignShuffled) is NOT changed; it is imported from experiments/_probes/nulldet4/rules_1105a.py.

Nothing here touches ree_core. Each rule owns its own numpy Generator, so no global RNG stream moves.
"""
from __future__ import annotations

import math

import numpy as np

from rules_1105a import M2SignShuffled  # noqa: F401  (flat import; nulldet4 is on sys.path in the worker)

LN4 = math.log(4.0)


class SignHack:
    """Raw sign rule (M1RAW logic) with step `eta`: on each committed window with R != 0, every channel whose
    |vote| exceeds its running median moves by eta * sign(R) * sign(v); clipped to +-ln4."""

    def __init__(self, eta: float):
        self.eta = float(eta)
        self.theta = np.zeros(4)
        self.absv = []
        self.n_windows = 0
        self.n_event_windows = 0

    def update(self, closed):
        old = self.theta.copy()
        if closed is not None and closed["committed"]:
            v = closed["v"]
            self.n_windows += 1
            self.absv.append(np.abs(v))
            if closed["R"] != 0:
                self.n_event_windows += 1
                med = np.median(np.asarray(self.absv), axis=0)
                self.theta = self.theta + self.eta * np.sign(closed["R"]) * np.sign(v) * (np.abs(v) > med)
        self.theta = np.clip(self.theta, -LN4, LN4)
        return self.theta - old

    def diagnostics(self):
        return {"eta": self.eta, "n_windows": int(self.n_windows), "n_event_windows": int(self.n_event_windows),
                "theta_final": [float(x) for x in self.theta],
                "harm_saturated": bool(self.theta[1] <= -LN4 + 1e-9)}


class NoisyHackerEta:
    """Positive control 2: SignHack(eta) drift on the REAL outcome PLUS an independent M2 sign-shuffled state
    (the null family's own noise). theta = clip(h + g, +-ln4)."""

    def __init__(self, seed: int, coin_offset: int, eta: float):
        self.hack = SignHack(eta)
        self.noise = M2SignShuffled(seed, coin_offset)
        self.theta = np.zeros(4)

    def update(self, closed):
        old = self.theta.copy()
        self.hack.update(closed)
        self.noise.update(closed)
        self.theta = np.clip(self.hack.theta + self.noise.theta, -LN4, LN4)
        return self.theta - old

    def diagnostics(self):
        h = self.hack.theta
        return {"h_final": [float(x) for x in h], "g_final": [float(x) for x in self.noise.theta],
                "eta": self.hack.eta, "n_h_event_updates": int(self.hack.n_event_windows),
                "h_harm_saturated": bool(h[1] <= -LN4 + 1e-9),
                "noise": self.noise.diagnostics()}
