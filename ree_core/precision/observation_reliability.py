"""SD-PP-1 -- evidence (sensory) precision producer.

An organism-side estimate of how reliable the exteroceptive channel is right
now, expressed both in observation units (sigma_obs / precision_obs) and, via
an estimated encoder gain kappa, in z_world units (evidence_variance_z /
evidence_precision_z). It is the "evidence precision" leg of the three-way
precision-provenance distinction (evidence vs model vs historical); see
`REE_assembly/docs/architecture/precision_provenance_substrate_spec.md`
section 2. It uses ONLY the observation stream (`obs_world` handed to
`agent.sense()`) and the encoder's own `z_world` output -- no environment
internals, no labels, no future frames.

Statistic (consecutive exteroceptive frames o_{t-1}, o_t):

    d_t          = |o_t - o_{t-1}|                       elementwise
    sigma_hat_t  = median_e(d_t) / mad_scale              mad_scale = 0.954
                                                            (median|N(0,2s^2)| = 0.954 s)
    sigma_sq_ema <- (1-a_o) sigma_sq_ema + a_o sigma_hat_t^2   a_o = obs_ema_alpha
                                                            (first differenced frame initialises)
    sigma_obs_sq = max(sigma_sq_ema, sigma_floor^2)        sigma_floor = instrument floor
    precision_obs = 1 / sigma_obs_sq

Measured 2026-09-22 (probe, seed 42, 300 random-action steps, CausalGridWorldV2
world_state, 250 elements): clean median|d| = 0.0000 (p90 0.0000); sigma 0.03
-> recovered 0.0333; sigma 0.12 -> recovered 0.1271. The statistic separates
cleanly because in a mostly-static exteroceptive field fewer than half the
elements change per frame (clean frac_changed = 0.127), so the median sits at
0 without noise and at ~sigma with it.

Encoder gain (expresses evidence precision in z units):

    kappa_t = mean_d((z_t - z_{t-1})^2) / mean_e((o_t - o_{t-1})^2)   skipped when denom < 1e-12
    kappa  <- (1-a_k) kappa + a_k kappa_t     a_k = kappa_ema_alpha
                                               (initialised to the first kappa_t; ready=False until then)
    evidence_variance_z  = kappa * sigma_obs_sq
    evidence_precision_z = 1 / (evidence_variance_z + eps)

LIMITATION: kappa is measured on real motion (the actual encoder response to
actual state change) and then applied uniformly to noise. An encoder that
suppresses high-frequency jitter has a smaller true gain on noise than on
motion, so this proxy OVER-estimates z-space noise and the downstream
evidence_precision_z / consolidation gain is therefore conservative under
noisy observation conditions. This is a named proxy, not a learned per-state
sensory-precision head; that upgrade is registered separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class ObservationReliabilityConfig:
    use_observation_reliability: bool = False
    obs_ema_alpha: float = 0.2
    kappa_ema_alpha: float = 0.05
    sigma_floor: float = 0.005
    mad_scale: float = 0.954
    eps: float = 1e-9


def _flatten_1d(x: torch.Tensor) -> torch.Tensor:
    """Accept [N] or [1,N] (or [D]/[1,D]) and return a detached 1-D clone."""
    if x.dim() == 2:
        if x.shape[0] != 1:
            raise ValueError(f"expected [N] or [1,N], got shape {tuple(x.shape)}")
        x = x[0]
    elif x.dim() != 1:
        raise ValueError(f"expected [N] or [1,N], got shape {tuple(x.shape)}")
    return x.detach().clone()


class ObservationReliabilityEstimator:
    """SD-PP-1 evidence precision producer. Pure arithmetic; no RNG draws."""

    def __init__(self, config: ObservationReliabilityConfig) -> None:
        self.config = config

        self._prev_obs: Optional[torch.Tensor] = None
        self._prev_latent: Optional[torch.Tensor] = None

        self._sigma_sq_ema: Optional[float] = None
        self._kappa: float = 1.0
        self._kappa_ready: bool = False

        self._n_frames: int = 0

        # per-tick cache, cleared after use by observe_latent / on_episode_reset
        self._cached_mean_obs_diff_sq: Optional[float] = None

    # -- observation side -----------------------------------------------

    def observe_obs(self, obs_world: torch.Tensor) -> None:
        obs = _flatten_1d(obs_world)

        if self._prev_obs is None:
            self._prev_obs = obs
            return

        d = (obs - self._prev_obs).abs()
        sigma_hat = torch.median(d).item() / self.config.mad_scale
        sigma_hat_sq = sigma_hat * sigma_hat

        if self._sigma_sq_ema is None:
            self._sigma_sq_ema = sigma_hat_sq
        else:
            a_o = self.config.obs_ema_alpha
            self._sigma_sq_ema = (1.0 - a_o) * self._sigma_sq_ema + a_o * sigma_hat_sq

        self._cached_mean_obs_diff_sq = torch.mean((obs - self._prev_obs) ** 2).item()

        self._prev_obs = obs
        self._n_frames += 1

    def observe_latent(self, z_world: torch.Tensor) -> None:
        if self._cached_mean_obs_diff_sq is None:
            # no cached obs difference for this tick -- no-op (also covers the
            # "observe_latent before any observe_obs" and "second call this tick" cases)
            return

        z = _flatten_1d(z_world)
        mean_obs_diff_sq = self._cached_mean_obs_diff_sq
        # clear the per-tick cache now so a second observe_latent this tick is a no-op
        self._cached_mean_obs_diff_sq = None

        if self._prev_latent is not None and mean_obs_diff_sq >= 1e-12:
            mean_z_diff_sq = torch.mean((z - self._prev_latent) ** 2).item()
            kappa_t = mean_z_diff_sq / mean_obs_diff_sq

            if not self._kappa_ready:
                self._kappa = kappa_t
                self._kappa_ready = True
            else:
                a_k = self.config.kappa_ema_alpha
                self._kappa = (1.0 - a_k) * self._kappa + a_k * kappa_t

        self._prev_latent = z

    def on_episode_reset(self) -> None:
        self._prev_obs = None
        self._prev_latent = None
        self._cached_mean_obs_diff_sq = None

    # -- read-only properties --------------------------------------------

    @property
    def sigma_obs_sq(self) -> float:
        floor_sq = self.config.sigma_floor * self.config.sigma_floor
        if self._sigma_sq_ema is None:
            return floor_sq
        return max(self._sigma_sq_ema, floor_sq)

    @property
    def sigma_obs(self) -> float:
        return self.sigma_obs_sq ** 0.5

    @property
    def precision_obs(self) -> float:
        return 1.0 / self.sigma_obs_sq

    @property
    def kappa(self) -> float:
        return self._kappa if self._kappa_ready else 1.0

    @property
    def ready(self) -> bool:
        return self._kappa_ready

    @property
    def evidence_variance_z(self) -> float:
        return self.kappa * self.sigma_obs_sq

    @property
    def evidence_precision_z(self) -> float:
        return 1.0 / (self.evidence_variance_z + self.config.eps)

    @property
    def n_frames(self) -> int:
        return self._n_frames

    # -- reporting ---------------------------------------------------------

    def snapshot(self) -> Dict[str, float]:
        return {
            "sigma_obs": self.sigma_obs,
            "sigma_obs_sq": self.sigma_obs_sq,
            "precision_obs": self.precision_obs,
            "kappa": self.kappa,
            "evidence_variance_z": self.evidence_variance_z,
            "evidence_precision_z": self.evidence_precision_z,
            "ready": self.ready,
        }

    # ------------------------------------------------------------------ #
    # Calibration state (2026-09-22, pre-freeze repair for V3-EXQ-1073)     #
    # ------------------------------------------------------------------ #
    def get_state(self) -> Dict[str, float]:
        """EMA state only -- what a calibration window taught the estimator.

        The per-tick frame/latent cache is deliberately NOT included: it is
        the previous OBSERVATION, and carrying it across a reset would make
        the next frame a cross-episode difference (see on_episode_reset).
        """
        return {
            "sigma_sq_ema": (float("nan") if self._sigma_sq_ema is None
                             else float(self._sigma_sq_ema)),
            "kappa": float(self._kappa),
            "kappa_ready": float(bool(self._kappa_ready)),
            "n_frames": float(self._n_frames),
        }

    def set_state(self, state: Dict[str, float]) -> None:
        """Restore EMA state from get_state(); drops the per-tick caches."""
        v = float(state.get("sigma_sq_ema", float("nan")))
        self._sigma_sq_ema = None if v != v else v
        self._kappa = float(state.get("kappa", 1.0))
        self._kappa_ready = bool(float(state.get("kappa_ready", 0.0)) >= 0.5)
        self._n_frames = int(float(state.get("n_frames", 0.0)))
        self.on_episode_reset()

    def get_metrics(self) -> Dict[str, float]:
        snap = self.snapshot()
        metrics = {f"obs_reliability_{k}": v for k, v in snap.items()}
        metrics["obs_reliability_n_frames"] = self.n_frames
        return metrics
