"""SD-PP-2 -- model-precision producer for `e2.world_forward` (z_world domain).

WHAT IT IS
----------
`WorldForwardEpistemicPrecision` estimates how precise the organism's OWN
world-forward model (`E2FastPredictor.world_forward`) is about the next
`z_world`, and splits that precision into three parts:

    v_tot   total predictive variance of the world-forward prediction
    v_ale   the ALEATORIC part attributed to the evidence channel
            (from SD-PP-1 `ObservationReliabilityEstimator`)
    v_epi   the EPISTEMIC remainder, v_epi = max(v_tot - v_ale, v_floor)
    pi_epi  epistemic precision, 1 / v_epi

THE THREE QUANTITIES (why one producer, read at two times)
----------------------------------------------------------
The precision-provenance design keeps three things distinct:

  1. HISTORICAL model precision -- the precision the organism HAD when it made
     a particular prediction. Obtained by calling `precision_at(z_prev, a)`
     BEFORE the outcome for that transition is observed. This is what rides in
     the replay provenance packet (SD-PP-3) as `pi_hist`.
  2. CURRENT model precision -- the precision the organism has NOW, read at
     sleep entry via `current_read()` (the global EMA source) and used as
     `pi_cur` by the consolidation-gain rule (SD-PP-4).
  3. EVIDENCE precision -- how reliable the sensory channel was; produced by
     SD-PP-1, NOT by this module. It enters here only as the
     `evidence_variance_z` argument of `observe_outcome`, which is what makes
     the aleatoric/epistemic split possible at all.

(1) and (2) are THE SAME PRODUCER READ AT TWO TIMES. That is what makes the
historical-vs-current distinction operational rather than nominal: there is no
second estimator whose disagreement with the first could be an artefact of a
different statistic. The ordering contract is therefore load-bearing --
`precision_at` must be called BEFORE `observe_outcome` for a transition, and
`precision_at` never reads the outcome and never mutates estimator state.

SOURCES AND THE SD-063 FALLBACK RULE
------------------------------------
  "ema"           global calibrated: v_tot is an EMA (alpha `pe_ema_alpha`) of
                  the per-dim mean squared prediction error of
                  `e2.world_forward` over waking transitions, initialised to
                  `v_init` before any observation. Always available, but
                  STATE-BLIND (the same number at every (z, a)).
  "sd063"         per-state: v_tot(z, a) = head.predictive_variance(z, a)[0]
                  from the SD-063 `E2WorldUncertaintyHead`, used only when
                  `head is not None and head.training_ready`.
  "sd063_or_ema"  (default) per-state when the head is present and ready,
                  otherwise the EMA read.

FALLBACK RULE, stated once so it cannot be got wrong: source "sd063" with no
head, or with a head whose `training_ready` is False, ALSO falls back to the
EMA read and REPORTS `source="ema"`. The `PrecisionRead.source` field always
names the source that actually served the read, never the configured intent --
that is what lets the packet (and any later audit) tell a per-state read from a
state-blind one. `current_read()` is ALWAYS the EMA read, by construction: it
is a global quantity and has no (z, a) to be conditioned on.

NOISE SPLIT
-----------
    v_noise <- (1 - alpha) * v_noise + alpha * (noise_gain * evidence_variance_z)
               (same alpha as v_tot; initialised 0.0; evidence_variance_z is
                0.0 when SD-PP-1 is absent, which collapses v_ale to 0 and
                makes v_epi == v_tot)
    v_epi(z, a) = max(v_tot(z, a) - v_noise, v_floor)      v_floor = 1e-6
    pi_epi      = 1 / v_epi

`noise_gain = 2.0` is PRE-REGISTERED FROM FIRST PRINCIPLES, quoted from the
contract (precision_provenance_substrate_spec.md section 3): "observation noise
enters the PE through BOTH the input (z_t) and the target (z_{t+1}), and for a
near-identity head (which MECH-573 measures the converged head to be, skill ~0)
the PE noise variance is ~2x the per-frame z-noise variance." It is a NAMED
ASSUMPTION, not a fitted constant, and is recorded as such in the substrate
record.

LIMITATION (named, not papered over)
------------------------------------
Neither source gives a native epistemic/aleatoric decomposition:

  * The "ema" source is GLOBAL and STATE-BLIND -- one scalar for the whole
    policy, exactly the property SD-063 criticises in the E3 running-variance
    EMA. A historical read taken at a hard (z, a) is indistinguishable from one
    taken at an easy (z, a).
  * The SD-063 head's `predictive_variance` is TOTAL predictive spread: it
    absorbs observation noise along with model uncertainty, so it cannot by
    itself separate model precision from evidence precision.

So the epistemic split here is BY SUBTRACTION of an EMA'd, gain-scaled evidence
variance -- an estimate, not a measurement. A per-state estimator that
separates epistemic from aleatoric natively is registered as substrate
necessity (f) in the spec, and is NOT built here.

Pure float/tensor arithmetic. NO RNG draws anywhere in this module. No tensor
with `requires_grad` is ever stored. Default-OFF by structural absence: with
`use_world_forward_epistemic_precision` False the integration layer constructs
no object and makes no call.

Contract: REE_assembly/docs/architecture/precision_provenance_substrate_spec.md
section 3. Substrate record:
docs/substrate/SD-PP-2-world-forward-epistemic-precision.md.
Claims: MECH-572 (lead), MECH-573, MECH-016, ARC-055, MECH-059.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

# Sources a PrecisionRead may report (what actually served the read).
READ_SOURCES = ("ema", "sd063")
# Sources the config may request.
CONFIG_SOURCES = ("ema", "sd063", "sd063_or_ema")


@dataclass
class PrecisionRead:
    """One precision read. `source` names what ACTUALLY served it.

    pi_epi -- epistemic precision, 1 / v_epi
    v_tot  -- total predictive variance (EMA scalar, or per-(z,a) SD-063 read)
    v_ale  -- aleatoric part attributed to the evidence channel (the v_noise EMA)
    v_epi  -- max(v_tot - v_ale, v_floor)
    source -- "ema" | "sd063"
    """

    pi_epi: float
    v_tot: float
    v_ale: float
    v_epi: float
    source: str


@dataclass
class WorldForwardEpistemicPrecisionConfig:
    """Config for SD-PP-2. Every field defaults to the pre-registered value.

    use_world_forward_epistemic_precision -- master switch (default False;
        OFF is bit-identical by structural absence -- the integration layer
        builds no estimator and makes no call).
    source        -- "ema" | "sd063" | "sd063_or_ema" (default "sd063_or_ema").
    pe_ema_alpha  -- EMA rate for BOTH v_tot and v_noise (default 0.05).
    v_floor       -- floor on v_epi, per-dim z units (default 1e-6).
    noise_gain    -- PE-noise multiplier on evidence variance (default 2.0; see
                     the module docstring for the first-principles derivation).
    v_init        -- v_tot before any observation (default 1e-2, the fresh-base
                     residual scale measured in V3-EXQ-1063).
    """

    use_world_forward_epistemic_precision: bool = False
    source: str = "sd063_or_ema"
    pe_ema_alpha: float = 0.05
    v_floor: float = 1e-6
    noise_gain: float = 2.0
    v_init: float = 1e-2

    def __post_init__(self) -> None:
        if self.source not in CONFIG_SOURCES:
            raise ValueError(
                "source must be one of %s, got %r" % (list(CONFIG_SOURCES), self.source)
            )
        if not (0.0 < float(self.pe_ema_alpha) <= 1.0):
            raise ValueError(
                "pe_ema_alpha must be in (0, 1], got %r" % (self.pe_ema_alpha,)
            )
        if not float(self.v_floor) > 0.0:
            raise ValueError("v_floor must be > 0, got %r" % (self.v_floor,))
        if not float(self.noise_gain) >= 0.0:
            raise ValueError("noise_gain must be >= 0, got %r" % (self.noise_gain,))
        if not float(self.v_init) > 0.0:
            raise ValueError("v_init must be > 0, got %r" % (self.v_init,))


def _as_batched(t: torch.Tensor) -> torch.Tensor:
    """Return a [B, D] view of a [D] or [B, D] tensor (no copy for [B, D])."""
    return t if t.dim() > 1 else t.unsqueeze(0)


class WorldForwardEpistemicPrecision:
    """Model-precision producer for `e2.world_forward` (SD-PP-2).

    State is two EMA scalars plus an observation counter -- no tensors are
    retained, nothing requires grad, and no RNG is ever drawn.
    """

    def __init__(
        self, config: WorldForwardEpistemicPrecisionConfig, world_dim: int
    ) -> None:
        self.config = config
        self.world_dim = int(world_dim)
        # v_tot starts at v_init (fresh-base residual scale) so a read taken
        # before any observation is finite and conservative rather than 0/inf.
        self._v_tot = float(config.v_init)
        # v_noise starts at 0.0 -- with no evidence-precision producer wired
        # this stays 0.0 forever and v_epi collapses to v_tot.
        self._v_noise = 0.0
        self._n_obs = 0

    # ------------------------------------------------------------------ #
    # Prediction                                                          #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def prediction_at_test(
        self, e2: Any, z_prev: torch.Tensor, a_onehot: torch.Tensor
    ) -> torch.Tensor:
        """The point prediction under test: `e2.world_forward(z_prev, a)`, [1, D].

        Exactly `e2.world_forward(z_prev[:1], a_onehot[:1])` under no_grad,
        returned as a detached clone so the caller can hold it in a runtime
        buffer (the SD-PP-3 packet) without pinning an autograd graph. 1-D
        inputs are promoted to [1, D] / [1, A] first.
        """
        z = _as_batched(z_prev)
        a = _as_batched(a_onehot)
        pred = e2.world_forward(z[:1], a[:1])
        return pred.detach().clone()

    # ------------------------------------------------------------------ #
    # Reads (never touch the outcome, never mutate state)                 #
    # ------------------------------------------------------------------ #

    def _ema_read(self) -> PrecisionRead:
        """The global, state-blind read from the two EMA scalars."""
        v_tot = float(self._v_tot)
        v_ale = float(self._v_noise)
        v_epi = max(v_tot - v_ale, float(self.config.v_floor))
        return PrecisionRead(
            pi_epi=1.0 / v_epi,
            v_tot=v_tot,
            v_ale=v_ale,
            v_epi=v_epi,
            source="ema",
        )

    @torch.no_grad()
    def precision_at(
        self,
        z_prev: torch.Tensor,
        a_onehot: torch.Tensor,
        head: Optional[Any] = None,
    ) -> PrecisionRead:
        """Precision read at (z_prev, a_onehot) using the CURRENT state only.

        NEVER reads the outcome and NEVER mutates estimator state -- calling it
        twice in a row returns identical values. This is the HISTORICAL read
        when it is taken before `observe_outcome` for the same transition.

        `head` is an optional SD-063 `E2WorldUncertaintyHead` (anything with a
        `training_ready` bool and a `predictive_variance(z, a) -> [B]` method).
        It is consulted only when the configured source asks for it AND the
        head is present AND `head.training_ready` is True; otherwise this falls
        back to the EMA read and reports `source="ema"`.
        """
        want_head = self.config.source in ("sd063", "sd063_or_ema")
        ready = head is not None and bool(getattr(head, "training_ready", False))
        if not (want_head and ready):
            return self._ema_read()

        z = _as_batched(z_prev)
        a = _as_batched(a_onehot)
        # The head computes this under its own no_grad and records read
        # diagnostics on itself (_last_pvar_*). That is a diagnostic write on
        # the HEAD, not on this estimator's state -- precision_at remains
        # idempotent with respect to everything the packet carries.
        pvar = head.predictive_variance(z[:1], a[:1])
        v_tot = float(pvar.reshape(-1)[0])
        v_ale = float(self._v_noise)
        v_epi = max(v_tot - v_ale, float(self.config.v_floor))
        return PrecisionRead(
            pi_epi=1.0 / v_epi,
            v_tot=v_tot,
            v_ale=v_ale,
            v_epi=v_epi,
            source="sd063",
        )

    def current_read(self) -> PrecisionRead:
        """The global (EMA-source) read -- used at sleep entry for `pi_cur`."""
        return self._ema_read()

    # ------------------------------------------------------------------ #
    # Update (reads ONLY its arguments)                                   #
    # ------------------------------------------------------------------ #

    def observe_outcome(
        self,
        pred: torch.Tensor,
        z_now: torch.Tensor,
        evidence_variance_z: float,
    ) -> float:
        """Fold one observed transition into the two EMAs; return the PE.

        pe = mean over dims of (pred - z_now)^2, as a float. Updates
        v_tot   <- (1-a) v_tot   + a * pe
        v_noise <- (1-a) v_noise + a * (noise_gain * evidence_variance_z)

        Reads NOTHING but its arguments -- in particular it never consults a
        head, an environment, or any later state, which is what keeps the
        historical read in front of it free of future information.
        """
        with torch.no_grad():
            p = _as_batched(pred).detach()
            zn = _as_batched(z_now).detach()
            diff = p[:1].float() - zn[:1].float()
            pe = float((diff * diff).mean().item())

        alpha = float(self.config.pe_ema_alpha)
        self._v_tot = (1.0 - alpha) * self._v_tot + alpha * pe
        self._v_noise = (1.0 - alpha) * self._v_noise + alpha * (
            float(self.config.noise_gain) * float(evidence_variance_z)
        )
        self._n_obs += 1
        return pe

    # ------------------------------------------------------------------ #
    # Properties / readouts                                               #
    # ------------------------------------------------------------------ #

    @property
    def v_tot(self) -> float:
        """Current global total-variance EMA."""
        return float(self._v_tot)

    @property
    def v_noise(self) -> float:
        """Current aleatoric (evidence-attributed) variance EMA."""
        return float(self._v_noise)

    @property
    def v_epi(self) -> float:
        """Current global epistemic variance, floored at `v_floor`."""
        return max(
            float(self._v_tot) - float(self._v_noise), float(self.config.v_floor)
        )

    @property
    def pi_cur(self) -> float:
        """Current global epistemic precision, 1 / v_epi."""
        return 1.0 / self.v_epi

    @property
    def n_obs(self) -> int:
        """Number of transitions folded in via `observe_outcome`."""
        return int(self._n_obs)

    def snapshot(self) -> Dict[str, float]:
        """Flat float snapshot of the global state (no per-(z,a) read)."""
        return {
            "v_tot": float(self._v_tot),
            "v_noise": float(self._v_noise),
            "v_epi": float(self.v_epi),
            "pi_cur": float(self.pi_cur),
            "n_obs": float(self._n_obs),
        }

    # ------------------------------------------------------------------ #
    # Calibration state (2026-09-22, pre-freeze repair for V3-EXQ-1073)     #
    # ------------------------------------------------------------------ #
    def get_state(self) -> Dict[str, float]:
        """The EMA state a calibration window taught the estimator: v_tot,
        v_noise, n_obs. Pure floats; restoring it makes a fresh estimator read
        exactly what this one reads (precision_at / current_read)."""
        return {"v_tot": float(self._v_tot), "v_noise": float(self._v_noise),
                "n_obs": float(self._n_obs)}

    def set_state(self, state: Dict[str, float]) -> None:
        """Restore from get_state()."""
        self._v_tot = float(state["v_tot"])
        self._v_noise = float(state["v_noise"])
        self._n_obs = int(float(state.get("n_obs", 0.0)))

    def get_metrics(self) -> Dict[str, float]:
        """`snapshot()` keys prefixed `wf_precision_`."""
        return {"wf_precision_" + k: v for k, v in self.snapshot().items()}
