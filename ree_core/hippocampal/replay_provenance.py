"""SD-PP-3 -- replay provenance packet: carrier.

One packet per replay-buffer entry, recorded at the moment `_e1_tick` appends
the outcome state to `agent._world_experience_buffer`, bound by INDEX to that
buffer so the consolidator can look it up for the triple it drew. Packet
`p[j]` describes the transition `(world[j-1], action[j], world[j])`; the
training triple at replay index `i` is `(world[i], action[i+1], world[i+1])`,
so its packet is `p[i+1]`. See
`REE_assembly/docs/architecture/precision_provenance_substrate_spec.md`
section 4 -- this module implements that contract exactly.

This module is a pure CARRIER: it holds no state about the world model or the
evidence channel itself, only a bounded ring of packets produced by reading
the two producers (SD-PP-1 `observation_reliability`, SD-PP-2
`world_forward_epistemic_precision`). Those two producers are built
CONCURRENTLY by other work and may not exist on disk while this module is
written, so `epistemic` and `reliability` are duck-typed rather than imported
at module top level:

    epistemic.prediction_at_test(e2, z_prev, a_onehot) -> Tensor[1, D]
    epistemic.precision_at(z_prev, a_onehot, head=head) -> object with
        .pi_epi .v_tot .v_ale .v_epi .source
    epistemic.observe_outcome(pred, z_now, evidence_variance_z) -> float
    reliability.snapshot() -> Dict[str, float] with keys
        sigma_obs, sigma_obs_sq, precision_obs, kappa, evidence_variance_z,
        evidence_precision_z, ready

Ordering contract enforced inside `record()` (the whole point of this
module): when a previous state exists, the epistemic PRECISION READ always
happens strictly BEFORE the OUTCOME is observed --

    pred = epistemic.prediction_at_test(...)
    read = epistemic.precision_at(...)          # historical / no-future-info
    ev   = reliability.snapshot() if reliability else zeros
    pe   = epistemic.observe_outcome(pred, z_now, ev["evidence_variance_z"])  # AFTER read

-- so `pi_hist` on every packet is genuinely the precision BEFORE this
outcome was seen, never contaminated by it. When there is no previous state
(episode start), no epistemic method is called at all and a placeholder
packet is recorded instead (`has_prev=False`, `pe`/`pi_hist`/`surprise` NaN).

No RNG draws anywhere in this module. MECH-094: callers run this recorder on
WAKING ticks only; a hypothesis-tagged tick is the caller's business to skip,
not this module's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

PROVENANCE_SCHEMA_VERSION = 1


@dataclass
class ReplayProvenancePacket:
    """One record bound by index to `agent._world_experience_buffer`.

    Field order matches spec section 4 exactly.
    """

    schema_version: int
    buffer_index: int
    tick: int
    provenance: str
    pred_at_test: torch.Tensor
    pe: float
    pi_hist: float
    v_tot_hist: float
    v_ale_hist: float
    precision_source: str
    evidence_variance_z: float
    evidence_precision_z: float
    sigma_obs: float
    kappa: float
    evidence_ready: bool
    surprise: float
    has_prev: bool


def _zero_evidence() -> Dict[str, float]:
    """Evidence snapshot used when no `reliability` producer is wired."""
    return {
        "sigma_obs": 0.0,
        "sigma_obs_sq": 0.0,
        "precision_obs": 0.0,
        "kappa": 0.0,
        "evidence_variance_z": 0.0,
        "evidence_precision_z": 0.0,
        "ready": False,
    }


def _mean_min_max(values: List[float]) -> Dict[str, float]:
    if not values:
        nan = float("nan")
        return {"mean": nan, "min": nan, "max": nan}
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


class ReplayProvenanceRecorder:
    """SD-PP-3 recorder: one packet per replay-buffer append.

    `epistemic` and `reliability` are duck-typed (see module docstring); they
    are never imported here, only called. `reliability` may be None (SD-PP-1
    absent / off) -- evidence fields on every packet are then zero / not
    ready, and `observe_outcome` is called with `evidence_variance_z=0.0`.
    """

    def __init__(
        self,
        epistemic: Any,
        reliability: Optional[Any],
        noise_gain: float,
        max_len: int = 1000,
    ) -> None:
        self.epistemic = epistemic
        self.reliability = reliability
        self.noise_gain = float(noise_gain)
        self.max_len = int(max_len)
        self.packets: List[ReplayProvenancePacket] = []

    def _snapshot_evidence(self) -> Dict[str, float]:
        if self.reliability is None:
            return _zero_evidence()
        return self.reliability.snapshot()

    def _trim(self) -> None:
        if len(self.packets) > self.max_len:
            del self.packets[: -self.max_len]

    @torch.no_grad()
    def record(
        self,
        e2: Any,
        z_prev: Optional[torch.Tensor],
        a_onehot: torch.Tensor,
        z_now: torch.Tensor,
        buffer_index: int,
        tick: int,
        head: Any = None,
    ) -> ReplayProvenancePacket:
        # (1) episode-start placeholder -- no epistemic method is called.
        if z_prev is None:
            ev = self._snapshot_evidence()
            d = int(z_now.shape[-1])
            pred_zeros = torch.zeros(1, d, dtype=z_now.dtype)
            packet = ReplayProvenancePacket(
                schema_version=PROVENANCE_SCHEMA_VERSION,
                buffer_index=buffer_index,
                tick=tick,
                provenance="real",
                pred_at_test=pred_zeros,
                pe=float("nan"),
                pi_hist=float("nan"),
                v_tot_hist=float("nan"),
                v_ale_hist=float("nan"),
                precision_source="none",
                evidence_variance_z=float(ev["evidence_variance_z"]),
                evidence_precision_z=float(ev["evidence_precision_z"]),
                sigma_obs=float(ev["sigma_obs"]),
                kappa=float(ev["kappa"]),
                evidence_ready=bool(ev["ready"]),
                surprise=float("nan"),
                has_prev=False,
            )
            self.packets.append(packet)
            self._trim()
            return packet

        # (2) point prediction at test time (no grad, detached inside epistemic).
        pred = self.epistemic.prediction_at_test(e2, z_prev, a_onehot)
        # (3) historical precision read -- BEFORE the outcome is observed.
        read = self.epistemic.precision_at(z_prev, a_onehot, head=head)
        # (4) evidence snapshot -- independent of the epistemic producer.
        ev = self._snapshot_evidence()
        # (5) ONLY NOW: observe the outcome (updates epistemic EMA state).
        pe = self.epistemic.observe_outcome(pred, z_now, ev["evidence_variance_z"])
        # (6) standardised epistemic surprise at test.
        surprise = float(read.pi_epi) * max(
            float(pe) - self.noise_gain * float(ev["evidence_variance_z"]), 0.0
        )

        # (7) build + append + trim.
        packet = ReplayProvenancePacket(
            schema_version=PROVENANCE_SCHEMA_VERSION,
            buffer_index=buffer_index,
            tick=tick,
            provenance="real",
            pred_at_test=pred.detach().clone(),
            pe=float(pe),
            pi_hist=float(read.pi_epi),
            v_tot_hist=float(read.v_tot),
            v_ale_hist=float(read.v_ale),
            precision_source=str(read.source),
            evidence_variance_z=float(ev["evidence_variance_z"]),
            evidence_precision_z=float(ev["evidence_precision_z"]),
            sigma_obs=float(ev["sigma_obs"]),
            kappa=float(ev["kappa"]),
            evidence_ready=bool(ev["ready"]),
            surprise=surprise,
            has_prev=True,
        )
        self.packets.append(packet)
        self._trim()
        return packet

    def get(self, index: int) -> Optional[ReplayProvenancePacket]:
        """Return the packet at CURRENT list position `index`, or None if out of range."""
        try:
            return self.packets[index]
        except IndexError:
            return None

    def trim_to(self, n_keep: int) -> None:
        """Keep only the most recent `n_keep` packets (n_keep <= 0 clears all)."""
        if n_keep <= 0:
            del self.packets[:]
            return
        excess = len(self.packets) - n_keep
        if excess > 0:
            del self.packets[:excess]

    def stats(self) -> Dict[str, float]:
        n = len(self.packets)
        has_prev_packets = [p for p in self.packets if p.has_prev]
        n_has_prev = len(has_prev_packets)
        frac_has_prev = (n_has_prev / n) if n > 0 else float("nan")

        pe_agg = _mean_min_max([p.pe for p in has_prev_packets])
        pi_agg = _mean_min_max([p.pi_hist for p in has_prev_packets])
        evp_agg = _mean_min_max([p.evidence_precision_z for p in has_prev_packets])
        surprise_agg = _mean_min_max([p.surprise for p in has_prev_packets])

        return {
            "n": float(n),
            "n_has_prev": float(n_has_prev),
            "frac_has_prev": float(frac_has_prev),
            "pe_mean": pe_agg["mean"],
            "pe_min": pe_agg["min"],
            "pe_max": pe_agg["max"],
            "pi_hist_mean": pi_agg["mean"],
            "pi_hist_min": pi_agg["min"],
            "pi_hist_max": pi_agg["max"],
            "evidence_precision_z_mean": evp_agg["mean"],
            "evidence_precision_z_min": evp_agg["min"],
            "evidence_precision_z_max": evp_agg["max"],
            "surprise_mean": surprise_agg["mean"],
            "surprise_min": surprise_agg["min"],
            "surprise_max": surprise_agg["max"],
        }

    def get_metrics(self) -> Dict[str, float]:
        return {f"replay_provenance_{k}": v for k, v in self.stats().items()}
