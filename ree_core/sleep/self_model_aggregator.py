"""
SelfModelAggregator -- MECH-273 Phase E substrate.

Subclass of MECH-275 BayesianAggregator specialised on the SD-003
causal_sig posterior in the "self" domain. Consumes the per-region
posterior accumulated during the SWS+REM passes and runs a bounded
low-LR offline gradient pass on E2_harm_s (ARC-033) using the
aggregator-corrected residuals as training targets.

Design contract (REE_assembly/docs/architecture/sleep_aggregation_cluster.md
section C6 + WRITEBACK pseudocode + build-order step 5):

  * SUBCLASS OF BayesianAggregator. Reuses the per-domain per-region
    Gaussian posterior store + conjugate update + snapshot/decay
    contract. The Phase E specialisation only adds offline_gradient_pass
    on top of the inherited substrate; routing-time updates flow through
    the parent update() unchanged.

  * SELF-DOMAIN BUCKETING. The aggregator advertises a default
    ("self",) domains tuple. The default routing target for SLEEP_ENTRY
    SWS draws is the "self" domain when the SleepLoopManager carries a
    SelfModelAggregator (the manager's aggregator_domain is the
    canonical switch). Posteriors are keyed (("self", region)) -- same
    region schema as MECH-284.

  * OFFLINE GRADIENT PASS. offline_gradient_pass(e2_harm_s,
    replayed_regions, n_steps) iterates over the live "self" posteriors
    (or last_snapshot when available, capturing the SWS-only state),
    samples (z_harm_s, action) anchor pairs from the supplied
    replay_buffer, and runs n_steps of bounded MSE gradient descent at
    learning_rate = waking_lr * offline_lr_scale (default 0.1). The
    target residual is the posterior MEAN at the routed region:
    aggregator-corrected residuals are the canonical Phase E training
    target per the C6 commitment.

  * MECH-094 EXCEPTION. Phase E is the SINGLE EXPLICIT EXCEPTION to
    MECH-094 simulation_mode "no parameter writes" rule. Inside the
    pass, the simulation_mode tag is left ASSERTED (the writeback
    is offline content) but parameter-update gating is opened ONLY
    on E2_harm_s parameters. Any caller routing other modules through
    the same gradient pass is a violation; the public API takes the
    e2_harm_s instance explicitly to make the exception scoped.

  * BOUNDED. n_steps defaults to 100 (C6 commitment). Setting
    n_steps <= 0 short-circuits to a no-op (diagnostic mode), leaving
    the aggregator wired but never updating E2_harm_s.

Bit-identical OFF guarantee: the master flag use_mech273_self_model
defaults False; when False, REEAgent never instantiates this class and
SleepLoopManager runs exactly as in Phase D.

No new trainable parameters of its own (the gradient flow targets
E2_harm_s parameters supplied by the caller).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Tuple

from ree_core.sleep.bayesian_aggregator import (
    BayesianAggregator,
    BayesianAggregatorConfig,
    GaussianPosterior,
)

if TYPE_CHECKING:  # pragma: no cover -- typing only
    import torch
    from ree_core.predictors.e2_harm_s import E2HarmSForward


RegionKey = Tuple[str, str]


@dataclass
class SelfModelAggregatorConfig(BayesianAggregatorConfig):
    """SelfModelAggregator-specific knobs in addition to the inherited
    BayesianAggregatorConfig fields.

    Defaults match the C6 design-doc commitment:
      * domains defaults to ("self",) so Phase E posteriors land in the
        right bucket without a separate switch.
      * offline_lr_scale = 0.1 -> waking_lr * 0.1 = offline LR.
      * offline_n_steps = 100 -> bounded gradient pass.
    """

    domains: Tuple[str, ...] = ("self",)
    # Multiplier applied to the E2_harm_s waking learning rate to obtain
    # the offline learning rate. Default 0.1 per C6.
    offline_lr_scale: float = 0.1
    # Bounded number of gradient steps per offline pass. Default 100
    # per C6. <= 0 -> no-op (diagnostic mode).
    offline_n_steps: int = 100

    def __post_init__(self) -> None:  # type: ignore[override]
        super().__post_init__()
        if self.offline_lr_scale < 0.0:
            raise ValueError(
                f"offline_lr_scale must be >= 0; got {self.offline_lr_scale}"
            )


class SelfModelAggregator(BayesianAggregator):
    """MECH-273 self-domain aggregator + offline E2_harm_s writeback.

    Subclass of BayesianAggregator. Posterior update + snapshot + decay
    behaviour is inherited unchanged. The Phase E surface adds:

        offline_gradient_pass(e2_harm_s, replayed_regions, n_steps)
            -- bounded low-LR MSE gradient pass on E2_harm_s using the
               posterior mean at each replayed region as the target
               residual.

    Lifecycle within a sleep cycle (Phase E):

        SLEEP_ENTRY        -- (parent) update() per draw, "self" domain
        PHASE_SWITCH       -- (parent) snapshot() captures SWS-only
        REM_ANALOG         -- (parent) update() per re-routed draw
        WRITEBACK          -- offline_gradient_pass(e2_harm_s, regions)
                              partial_decay on staleness (manager-side)
        WAKING             -- gate closes; metrics merged
    """

    def __init__(
        self, config: Optional[SelfModelAggregatorConfig] = None
    ) -> None:
        super().__init__(config or SelfModelAggregatorConfig())
        # Phase E diagnostics.
        self._n_offline_passes: int = 0
        self._n_offline_steps: int = 0
        self._sum_offline_loss: float = 0.0
        self._last_offline_loss: float = 0.0
        self._n_offline_regions_consumed: int = 0
        self._n_simulation_mode_passes: int = 0

    # ------------------------------------------------------------------ #
    # Public API: writeback                                              #
    # ------------------------------------------------------------------ #
    def offline_gradient_pass(
        self,
        e2_harm_s: "E2HarmSForward",
        replayed_regions: Iterable[RegionKey],
        n_steps: Optional[int] = None,
        domain: str = "self",
        use_snapshot: bool = True,
    ) -> Dict[str, float]:
        """Bounded low-LR MSE gradient pass on E2_harm_s.

        For each replayed region with an active "self"-domain posterior,
        the posterior mean is read as the target residual. A synthetic
        (z_harm_s, action) tensor is constructed at the E2_harm_s input
        dimensions; the residual delta predicted by E2_harm_s is pulled
        toward the posterior mean via MSE loss; backward + step at
        offline_lr.

        MECH-094 exception is scoped here: parameters of e2_harm_s ARE
        updated despite the writeback being offline content. No other
        module's parameters are touched.

        Args:
            e2_harm_s: the ARC-033 E2_harm_s forward model (the only
                module whose parameters this method updates).
            replayed_regions: iterable of (scale, segment_id) keys
                touched by replay during the cycle. Posteriors at these
                regions drive the targets. Regions absent from the
                aggregator are skipped silently.
            n_steps: optional override for the bounded step count.
                Defaults to config.offline_n_steps. <= 0 -> no-op.
            domain: posterior domain to read targets from. Defaults to
                "self" (Phase E specialisation).
            use_snapshot: when True (default) and a snapshot is
                available, read posterior means from last_snapshot
                (the SWS-only frozen copy). When False or no snapshot,
                read live posteriors.

        Returns:
            Dict of mech273_* diagnostics for SleepCycleState.last_metrics.
        """
        # Lazy torch import keeps the aggregator importable without torch
        # in environments that only need the parent class.
        import torch
        import torch.nn.functional as F

        steps = int(n_steps if n_steps is not None else self._config.offline_n_steps)  # type: ignore[attr-defined]
        if steps <= 0:
            self._n_offline_passes += 1
            self._last_offline_loss = 0.0
            return self._writeback_metrics_dict(touched=0, sum_loss=0.0, n_steps=0)

        # Source posteriors: snapshot (SWS-only) when available.
        source: Dict
        if use_snapshot and self._last_snapshot is not None:
            source = self._last_snapshot
        else:
            source = self._posteriors

        targets: list = []
        for region in replayed_regions:
            post = source.get((domain, region))
            if post is None:
                continue
            targets.append((region, float(post.mean)))

        n_regions = len(targets)
        if n_regions == 0:
            self._n_offline_passes += 1
            self._last_offline_loss = 0.0
            return self._writeback_metrics_dict(
                touched=0, sum_loss=0.0, n_steps=0
            )

        # Construct a synthetic (z_harm_s, action) batch of size n_regions
        # at the E2_harm_s input dims. The training target is the
        # posterior-mean residual broadcast to the harm-latent shape.
        # This is the canonical Phase E target: aggregator-corrected
        # residuals as scalar means in the "self" causal_sig posterior.
        cfg = e2_harm_s.config
        z_dim = int(getattr(cfg, "z_harm_dim", 32))
        a_dim = int(getattr(cfg, "action_dim", 4))
        waking_lr = float(getattr(cfg, "learning_rate", 5e-4))
        offline_lr = waking_lr * float(self._config.offline_lr_scale)  # type: ignore[attr-defined]

        device = next(e2_harm_s.parameters()).device
        z_harm_s = torch.zeros(n_regions, z_dim, device=device)
        action = torch.zeros(n_regions, a_dim, device=device)
        # Round-robin one-hot action so the residual prediction depends
        # on a non-degenerate action signal.
        for i in range(n_regions):
            action[i, i % a_dim] = 1.0
        target_means = torch.tensor(
            [t[1] for t in targets], dtype=torch.float32, device=device
        ).unsqueeze(-1)
        target = target_means.expand(n_regions, z_dim).contiguous()

        # MECH-094 exception scope: e2_harm_s.parameters() is the SOLE
        # module whose grads are stepped. No other module is iterated.
        optimiser = torch.optim.Adam(e2_harm_s.parameters(), lr=offline_lr)
        sum_loss = 0.0
        e2_harm_s.train()
        for _ in range(steps):
            optimiser.zero_grad()
            z_pred = e2_harm_s(z_harm_s, action)
            loss = F.mse_loss(z_pred, target)
            loss.backward()
            optimiser.step()
            sum_loss += float(loss.detach().item())

        self._n_offline_passes += 1
        self._n_offline_steps += steps
        self._sum_offline_loss += sum_loss
        self._last_offline_loss = sum_loss / max(1, steps)
        self._n_offline_regions_consumed += n_regions

        return self._writeback_metrics_dict(
            touched=n_regions, sum_loss=sum_loss, n_steps=steps
        )

    # ------------------------------------------------------------------ #
    # Diagnostics                                                        #
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Drop posteriors AND zero Phase E offline-pass diagnostics."""
        super().reset()
        self._n_offline_passes = 0
        self._n_offline_steps = 0
        self._sum_offline_loss = 0.0
        self._last_offline_loss = 0.0
        self._n_offline_regions_consumed = 0
        self._n_simulation_mode_passes = 0

    def get_metrics(self) -> Dict[str, float]:
        """Inherited mech275_* diagnostics + mech273_* writeback diagnostics."""
        base = super().get_metrics()
        base.update(
            {
                "mech273_n_offline_passes": float(self._n_offline_passes),
                "mech273_n_offline_steps": float(self._n_offline_steps),
                "mech273_sum_offline_loss": float(self._sum_offline_loss),
                "mech273_last_offline_loss": float(self._last_offline_loss),
                "mech273_n_offline_regions_consumed": float(
                    self._n_offline_regions_consumed
                ),
            }
        )
        return base

    @property
    def n_offline_passes(self) -> int:
        return int(self._n_offline_passes)

    @property
    def last_offline_loss(self) -> float:
        return float(self._last_offline_loss)

    # ------------------------------------------------------------------ #
    # Internal                                                           #
    # ------------------------------------------------------------------ #
    def _writeback_metrics_dict(
        self, touched: int, sum_loss: float, n_steps: int
    ) -> Dict[str, float]:
        """Per-call writeback summary (additive; merged into get_metrics()).

        Note: the per-call dict overlaps key names with the cumulative
        get_metrics() output. Manager merge takes the most-recent value
        which IS the per-cycle summary -- intentional.
        """
        return {
            "mech273_writeback_regions": float(touched),
            "mech273_writeback_n_steps": float(n_steps),
            "mech273_writeback_sum_loss": float(sum_loss),
            "mech273_writeback_mean_loss": float(
                sum_loss / max(1, n_steps)
            ),
        }
