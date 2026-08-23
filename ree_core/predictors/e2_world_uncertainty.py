"""
E2WorldUncertaintyHead -- SD-063: conditional predictive-uncertainty head on the
E2 world-forward (distribution-free quantile / pinball form).

The current E2 world-forward is a POINT predictor trained on MSE (e2_fast.py
world_forward / e2_world.py); the only uncertainty signal reaching E3's commit
gate is the running-variance EMA (e3_selector.py current_precision), a
temporally-smoothed GLOBAL, state-BLIND estimate whose predicted uncertainty has
near-zero per-point correlation with realized error (precision_error_corr ~ 0.0
by construction). SD-063 gives the forward model a CONDITIONAL predictive
distribution so E3 can gate action commitment on where THIS prediction is
uncertain rather than on a running average -- the concrete realization of the
MECH-059 confidence channel (uncertainty-derived precision distinct from
residual error).

Preferred form (V3-EXQ-712 winner): distribution-free quantile regression with
the pinball loss. The 712 diagnostic held the encoder + transition set fixed and
swapped only the forward head + loss across four formulations on the same
held-out (z_world_t, a_t) -> z_world_{t+1} transitions (5 seeds, 4760 test
transitions):

    head              CRPS (lower=better)   precision_error_corr
    quantile_pinball  0.00486  (winner)     0.379  (winner)
    mse_point         0.00514               0.0    (the EMA null)
    mixture_gaussian  0.00682               0.038
    hetero_gaussian   0.00708               0.040

The load-bearing finding is narrow: ONLY the distribution-free quantile head
helped; imposing a Gaussian shape (hetero, mixture) on the next-state predictive
distribution did WORSE than the point baseline. The quantile head also delivered
a genuine per-point error signal (0.379) the EMA structurally cannot carry.

CRITICAL -- SD-031 agency-residual guard (design caveat):
    E2 is an agency detector via the residual
        residual_world = z_world_observed - E2WorldForward(z_world_prev, a_actual)
    (SD-031 / MECH-256). A predictive-variance head can absorb the agent-caused
    component of next-state variance into "expected spread," quietly killing the
    agency signal. THIS MODULE DEFENDS THAT STRUCTURALLY:
      - It is a SEPARATE nn.Module. It shares NO parameters with E2WorldForward
        or the z_world encoder. Training it cannot move the forward model that
        produces the agency residual.
      - Its P1 loss reads DETACHED z_world inputs and DETACHED z_world_next
        targets (the caller applies .detach()); its gradients never reach the
        encoder. This is the same stop-gradient discipline e2_world.py uses.
    The structural separation makes explaining-away impossible by construction;
    the SD-063 validation experiment must STILL confirm empirically that the
    E2WorldForward comparator residual is preserved when this head trains jointly.

Phased training (mirrors SD-031 / ARC-033):
    P0: z_world encoder warmup (SD-009 event-contrastive + SD-018 resource
        proximity) so z_world carries trained discriminative structure. An
        untrained encoder yields a near-invariant z_world -> the head fits a
        trivial spread and the comparison is vacuous (the MECH-353 / V3-EXQ-642
        lesson, and the 712 readiness gate).
    P1: the head trains on FROZEN z_world (stop-gradient: detach BOTH the
        (z_world_t, a_t) inputs AND the z_world_next target). Do NOT propagate
        head loss into the encoder or the point forward model.
    P2: evaluate held-out CRPS, precision_error_corr, and -- the SD-063
        falsifier -- that the E2WorldForward agency residual is unchanged.

MECH-094: this head is a WAKING online forward-model read used for commitment
gating. It does NOT write to memory and does NOT run under simulation / replay,
so the MECH-094 hypothesis_tag constraint DOES NOT APPLY. (Contrast the E2World
comparator's retrospective comparator-mode read, which IS MECH-094-gated.)

Biological grounding (shared with MECH-059 / SD-031): parietal / ACC-insula
outcome-prediction circuits carry a confidence estimate about predicted outcomes
distinct from the outcome prediction error itself; precision-weighting of
prediction (active inference, Friston) modulates action commitment on a
per-context basis, not via a single global gain.

See docs/architecture/sd_063_e2_conditional_uncertainty_head.md,
    ree_core/predictors/e2_world.py (the SD-031 agency comparator this must not
      disturb),
    ree_core/predictors/e3_selector.py (current_precision EMA -- the null this
      replaces at the commit gate under E3Config.use_conditional_precision_gate).
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# Fixed quantile levels -- the V3-EXQ-712 winner config (9 levels, 0.1..0.9).
# Central interval [q0.1, q0.9] is the nominal-80% predictive interval.
QUANTILE_LEVELS: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Interquantile-range -> std conversion for the [q0.1, q0.9] span under a
# reference Gaussian: z_0.9 - z_0.1 = 2 * 1.2816 = 2.5631. Used only to turn the
# distribution-free IQR into a variance-scaled spread the E3 commit gate (which
# lives in variance space) can compare against its threshold.
IQR_TO_STD_10_90: float = 2.5631


@dataclass
class E2WorldUncertaintyConfig:
    """Configuration for the SD-063 E2 conditional predictive-uncertainty head.

    Disabled by default for full backward compatibility. Set
    use_e2_world_uncertainty=True to enable.

    z_world_dim has NO safe literal default (Optional[int]=None, REQUIRED when
    enabled) -- the same discipline as E2WorldConfig: a literal default would
    silently mis-size the head if a caller forgot to thread world_dim through.
    Unlike E2WorldConfig, there is NO world_dim>=128 hard-assert here: the SD-063
    head is a predictive-spread readout, not the SD-031 discriminative/attribution
    comparator, and the V3-EXQ-712 diagnostic that motivated it ran at world_dim=32.
    """

    # Master switch -- disabled by default (backward compat).
    use_e2_world_uncertainty: bool = False

    # Dimensions -- z_world_dim is REQUIRED (no safe literal default; see above).
    # Must match LatentStackConfig.world_dim of the agent.
    z_world_dim: Optional[int] = None
    action_dim: int = 4

    # Architecture (V3-EXQ-712 winner values).
    hidden_dim: int = 128

    # Quantile levels (fixed to the 712 winner set; overridable for ablation).
    quantile_levels: List[float] = field(default_factory=lambda: list(QUANTILE_LEVELS))

    # Training.
    learning_rate: float = 1e-3

    # ---------------------------------------------------------------- #
    # Phased ONLINE training (P0 warmup -> P1 head update -> P2 eval)   #
    # ---------------------------------------------------------------- #
    # All four are inert unless train_online=True. With train_online=False the
    # module is bit-identical to the pre-2026-08-22 head: no optimizer is ever
    # built, no replay buffer is allocated, and observe_transition() is a no-op.
    # This is the keystone the ARC-065 GAP-A follow-on named -- without it the
    # agent-level head stays at its random init and predictive_variance is
    # near-uniform (the vacuous-channel class the readiness gate refuses).
    train_online: bool = False

    # P0: number of observed transitions buffered BEFORE the first head update.
    # The z_world encoder is still warming during this window (SD-009
    # event-contrastive + SD-018 resource proximity), so its z_world is not yet
    # discriminative; training the head on it fits a trivial spread (the
    # MECH-353 / V3-EXQ-642 vacuous-comparison lesson). Transitions ARE buffered
    # during warmup and age out of the bounded replay as the encoder moves --
    # exactly the V3-EXQ-716 driver's schedule (`replay.append` while not in_p2,
    # head updates in P1 only).
    warmup_steps: int = 200

    # P1: bounded replay over DETACHED (z_world_t, a_t, z_world_{t+1}) triples.
    # Bounded on purpose: the encoder keeps moving through P1 (the 716 "joint
    # phase"), so stale-representation transitions must age out rather than
    # accumulate.
    replay_capacity: int = 2048
    batch_size: int = 32

    # Diagnostic-only readiness floor for `training_ready`. NOT a gate on
    # predictive_variance -- see the training_ready docstring for why the read
    # is deliberately never filtered on it.
    ready_min_train_steps: int = 50


class E2WorldUncertaintyHead(nn.Module):
    """SD-063: conditional predictive-uncertainty head over (z_world, action).

    f(z_world_t, a_t) -> per-dim quantiles of z_world_{t+1}, plus a per-input
    predictive-variance readout for the E3 commit gate.

    Input:  x = concat([z_world_t, a_onehot])  [batch, z_world_dim + action_dim]
    Output: quantiles [batch, z_world_dim, n_quantiles]

    This is a STANDALONE module (shares no parameters with E2WorldForward or the
    encoder). Callers MUST detach both the inputs and the target during training
    (P1 stop-gradient discipline) so the head cannot explain away the SD-031
    agency residual. See module docstring.

    Usage in experiment scripts:
        from ree_core.predictors.e2_world_uncertainty import (
            E2WorldUncertaintyHead, E2WorldUncertaintyConfig)
        cfg = E2WorldUncertaintyConfig(
            use_e2_world_uncertainty=True, z_world_dim=32, action_dim=4)
        head = E2WorldUncertaintyHead(cfg)
        opt = torch.optim.Adam(head.parameters(), lr=cfg.learning_rate)

        # P1 training step (detach BOTH inputs and target):
        q = head(z_world_t.detach(), action_t)
        loss = head.compute_loss(q, z_world_next.detach())
        opt.zero_grad(); loss.backward(); opt.step()

        # Commit-gate read (per-input predictive variance -> E3.select):
        pvar = head.predictive_variance(z_world_t, action_t)   # [batch]
        result = e3.select(candidates, conditional_predictive_variance=float(pvar.mean()))

    Usage in the AGENT (online P1, added 2026-08-22 -- the ARC-065 GAP-A
    follow-on keystone that makes MECH-314b's per-candidate source genuinely
    live rather than a random-init near-uniform vector):
        cfg = E2WorldUncertaintyConfig(..., train_online=True)
        # then each waking tick, from REEAgent._train_e2_world_uncertainty:
        head.observe_transition(z_world_prev, action_onehot, z_world_now)
    The agent path owns the P0/P1 schedule (warmup_steps) and the bounded
    replay; P2 is `agent.eval()` (the hook skips when self.training is False).
    Do NOT combine it with an externally-built optimizer -- pick one.
    """

    def __init__(self, config: Optional[E2WorldUncertaintyConfig] = None):
        super().__init__()
        self.config = config or E2WorldUncertaintyConfig()

        z_world_dim = self.config.z_world_dim
        if z_world_dim is None:
            raise ValueError(
                "E2WorldUncertaintyConfig.z_world_dim is required (no safe "
                "default) -- pass config.latent.world_dim so the head is sized "
                "to the agent's z_world."
            )

        self.z_world_dim = int(z_world_dim)
        self.action_dim = int(self.config.action_dim)
        self.hidden_dim = int(self.config.hidden_dim)

        levels = list(self.config.quantile_levels)
        if len(levels) < 2:
            raise ValueError(
                "E2WorldUncertaintyConfig.quantile_levels needs >= 2 levels; "
                "got {}.".format(levels)
            )
        if any(not (0.0 < lv < 1.0) for lv in levels):
            raise ValueError(
                "quantile_levels must lie strictly in (0, 1); got {}.".format(levels)
            )
        if any(levels[i] >= levels[i + 1] for i in range(len(levels) - 1)):
            raise ValueError(
                "quantile_levels must be strictly increasing; got {}.".format(levels)
            )
        self.n_quantiles = len(levels)
        # Registered buffer so it moves with .to(device) and serializes with the
        # module, without being a trainable parameter.
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.float32))

        # --- Phased online-training state (inert unless train_online) ---
        # The optimizer is built LAZILY on the first train_step so constructing
        # the head is unchanged for the existing callers that build their own
        # (tests/contracts/test_sd063_conditional_uncertainty_head.py and the
        # V3-EXQ-716/716a drivers all do `torch.optim.Adam(head.parameters())`).
        # DO NOT mix the two: an external optimizer plus train_step() gives two
        # Adam moment estimates over one parameter set.
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._replay: Optional[Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None
        self._n_observed: int = 0
        self._n_train_steps: int = 0
        self._last_train_loss: float = 0.0
        # Diagnostics of the LAST predictive_variance() read. The relative
        # spread is the quantity that actually discriminates a trained head from
        # a random-init one -- see the predictive_variance docstring.
        self._last_pvar_mean: float = 0.0
        self._last_pvar_range: float = 0.0
        self._last_pvar_relative_spread: float = 0.0

        in_dim = self.z_world_dim + self.action_dim
        # Trunk matches V3-EXQ-712 _trunk (2-layer MLP, ReLU).
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.out = nn.Linear(self.hidden_dim, self.z_world_dim * self.n_quantiles)

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #

    def forward(self, z_world: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict per-dim quantiles of z_world_{t+1}.

        IMPORTANT: during P1 training, pass z_world.detach() (and detach the
        target in compute_loss) so gradients never reach the z_world encoder or
        the E2WorldForward params -- this is what preserves the SD-031 agency
        residual.

        Args:
            z_world: [batch, z_world_dim] -- world-state latent (current step)
            action:  [batch, action_dim]  -- action taken (one-hot or continuous)

        Returns:
            quantiles: [batch, z_world_dim, n_quantiles]
        """
        if z_world.dim() == 1:
            z_world = z_world.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        x = torch.cat([z_world, action], dim=-1)
        q = self.out(self.trunk(x))
        return q.view(-1, self.z_world_dim, self.n_quantiles)

    # ------------------------------------------------------------------ #
    # Loss (pinball / quantile regression)                               #
    # ------------------------------------------------------------------ #

    def compute_loss(
        self,
        quantiles: torch.Tensor,
        z_world_next: torch.Tensor,
    ) -> torch.Tensor:
        """Pinball (quantile-regression) loss for P1 head training.

        The target z_world_next MUST be detached from the encoder graph before
        calling this (the caller applies .detach() -- this method does not, to
        support any calling convention). Detaching the target is load-bearing
        for the SD-031 agency-residual guard.

        pinball_tau(err) = max(tau * err, (tau - 1) * err),  err = y - q_tau

        Args:
            quantiles:    [batch, z_world_dim, n_quantiles] -- from forward()
            z_world_next: [batch, z_world_dim] -- actual next world latent
                          (should be .detach()ed by the caller)

        Returns:
            loss: scalar pinball loss (mean over batch, dims, quantiles)
        """
        if z_world_next.dim() == 1:
            z_world_next = z_world_next.unsqueeze(0)
        err = z_world_next.unsqueeze(-1) - quantiles          # [B, D, Q]
        lv = self._levels.view(1, 1, -1).to(err.device)       # [1, 1, Q]
        return torch.maximum(lv * err, (lv - 1.0) * err).mean()

    # ------------------------------------------------------------------ #
    # Phased online training (P0 warmup -> P1 update -> P2 eval)         #
    # ------------------------------------------------------------------ #

    @property
    def n_train_steps(self) -> int:
        """Number of P1 optimizer steps applied to this head."""
        return int(self._n_train_steps)

    @property
    def n_observed_transitions(self) -> int:
        """Number of transitions handed to observe_transition() (incl. warmup)."""
        return int(self._n_observed)

    @property
    def training_ready(self) -> bool:
        """True once the head has taken >= config.ready_min_train_steps updates.

        DIAGNOSTIC ONLY -- deliberately NOT consulted by predictive_variance /
        predictive_std, and deliberately NOT consulted by the agent's MECH-314b
        per-candidate read. Filtering the read on this flag would convert a
        vacuous channel into a SILENT fallback to the Phase-1 broadcast, which
        is precisely the failure the ARC-065 readiness gate exists to catch in
        the open: the gate is measured on StructuredCuriosity plus this head's
        `_last_pvar_relative_spread`, never a self-report of "I have taken N
        steps". A head that is trained but still emits a flat spread must be
        visible as such -- and note that step count and differentiation really do
        come apart, since a random-init head already passes
        `last_uncertainty_dev_range > 0` (see predictive_variance).
        """
        return self._n_train_steps >= int(self.config.ready_min_train_steps)

    def _ensure_optimizer(self) -> torch.optim.Optimizer:
        """Build the P1 optimizer on first use (see __init__ for why lazily)."""
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(
                self.parameters(), lr=float(self.config.learning_rate)
            )
        return self._optimizer

    def train_step(
        self,
        z_world_prev: torch.Tensor,
        action: torch.Tensor,
        z_world_next: torch.Tensor,
        simulation_mode: bool = False,
    ) -> Optional[float]:
        """One P1 pinball update on the FROZEN z_world target.

        Detaches inputs AND target internally -- the SD-031 agency-residual
        guard (gradients must never reach the z_world encoder or E2WorldForward;
        see the module docstring). This is the same stop-gradient discipline the
        documented external-optimizer recipe applies by hand, made a method so
        the agent-side loop cannot get it wrong.

        Args:
            z_world_prev: [B, z_world_dim] (or [z_world_dim]) previous z_world.
            action:       [B, action_dim]  (or [action_dim]) action taken.
            z_world_next: [B, z_world_dim] (or [z_world_dim]) observed next z_world.
            simulation_mode: True -> no-op, returns None. Imagined transitions
                carry no OBSERVED next state, so training on them would fit the
                head to the proposer's own rollout rather than to the world.

        Returns:
            The scalar pinball loss, or None if the step was skipped.
        """
        if simulation_mode:
            return None
        z_prev = z_world_prev.detach()
        a = action.detach()
        target = z_world_next.detach()
        if z_prev.dim() == 1:
            z_prev = z_prev.unsqueeze(0)
        if a.dim() == 1:
            a = a.unsqueeze(0)
        if target.dim() == 1:
            target = target.unsqueeze(0)
        if z_prev.shape[0] != target.shape[0] or z_prev.shape[0] != a.shape[0]:
            # Caller error -- skipped rather than silently broadcast-averaged
            # (mirrors AttributionHead.train_step's refusal).
            return None
        a = a.to(device=z_prev.device, dtype=z_prev.dtype)
        target = target.to(device=z_prev.device, dtype=z_prev.dtype)

        opt = self._ensure_optimizer()
        opt.zero_grad()
        loss = self.compute_loss(self.forward(z_prev, a), target)
        loss.backward()
        opt.step()
        self._n_train_steps += 1
        self._last_train_loss = float(loss.detach().item())
        return self._last_train_loss

    def observe_transition(
        self,
        z_world_prev: torch.Tensor,
        action: torch.Tensor,
        z_world_next: torch.Tensor,
        simulation_mode: bool = False,
    ) -> Optional[float]:
        """Buffer one observed transition and, once warm, take a P1 update.

        The online (per-tick) entry point, mirroring the V3-EXQ-716 driver's
        schedule: buffer every observed transition; take NO head update until
        `config.warmup_steps` transitions have been seen (P0 -- the encoder is
        still becoming discriminative); thereafter sample a minibatch of
        `config.batch_size` from the bounded replay and call train_step on it.
        A minibatch rather than the single current transition because the
        pinball loss estimates 9 quantile levels per z_world dim -- a batch of
        one is a near-useless gradient, and the 712 winner config was measured
        with batched updates.

        No-op (returns None) when `config.train_online` is False, under
        simulation_mode, or while the replay has fewer than one batch.

        Returns:
            The scalar pinball loss of this tick's update, or None if no update
            was taken.
        """
        if not bool(self.config.train_online) or simulation_mode:
            return None

        z_prev = z_world_prev.detach()
        a = action.detach()
        nxt = z_world_next.detach()
        if z_prev.dim() == 1:
            z_prev = z_prev.unsqueeze(0)
        if a.dim() == 1:
            a = a.unsqueeze(0)
        if nxt.dim() == 1:
            nxt = nxt.unsqueeze(0)
        # Only the first row is buffered -- the online caller supplies one tick.
        z_prev = z_prev[:1].clone()
        a = a[:1].to(device=z_prev.device, dtype=z_prev.dtype).clone()
        nxt = nxt[:1].clone()

        if self._replay is None:
            self._replay = deque(maxlen=int(self.config.replay_capacity))
        self._replay.append((z_prev, a, nxt))
        self._n_observed += 1

        # P0: buffer only. Stale-encoder transitions age out of the bounded
        # replay rather than being filtered, so P1 starts from a mixed buffer
        # that converges as the encoder settles (the 716 joint-phase schedule).
        if self._n_observed <= int(self.config.warmup_steps):
            return None
        batch = int(self.config.batch_size)
        if len(self._replay) < batch:
            return None

        # torch RNG (not `random`) so the sampling participates in the agent's
        # existing torch seeding rather than a second, separately-seeded stream.
        idx = torch.randint(0, len(self._replay), (batch,))
        rows = [self._replay[int(i)] for i in idx]
        z0b = torch.cat([r[0] for r in rows], dim=0)
        ab = torch.cat([r[1] for r in rows], dim=0)
        z1b = torch.cat([r[2] for r in rows], dim=0)
        return self.train_step(z0b, ab, z1b)

    def get_state(self) -> Dict[str, float]:
        """Diagnostic snapshot for experiment manifests / readiness reporting."""
        return {
            "e2_world_uncertainty_n_train_steps": int(self._n_train_steps),
            "e2_world_uncertainty_n_observed": int(self._n_observed),
            "e2_world_uncertainty_last_train_loss": float(self._last_train_loss),
            "e2_world_uncertainty_replay_size": (
                0 if self._replay is None else int(len(self._replay))
            ),
            "e2_world_uncertainty_training_ready": bool(self.training_ready),
            # Last predictive_variance() read -- the readiness discriminator.
            # relative_spread is the one to gate on; range alone is passed by a
            # random-init head (see the predictive_variance docstring).
            "e2_world_uncertainty_last_pvar_mean": float(self._last_pvar_mean),
            "e2_world_uncertainty_last_pvar_range": float(self._last_pvar_range),
            "e2_world_uncertainty_last_pvar_relative_spread": float(
                self._last_pvar_relative_spread
            ),
        }

    # ------------------------------------------------------------------ #
    # Predictive-spread readouts (for the E3 commit gate)                #
    # ------------------------------------------------------------------ #

    def predictive_std(self, z_world: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Per-input predictive standard deviation (mean over z_world dims).

        Distribution-free: derived from the [q0.1, q0.9] interquantile range,
        monotone-rearranged (torch.sort) to defend against quantile crossing,
        then scaled IQR / 2.5631 to a Gaussian-reference std so the E3 commit
        gate (variance space) has a comparable scale. Returned per batch item.

        Args:
            z_world: [batch, z_world_dim]
            action:  [batch, action_dim]

        Returns:
            std: [batch] -- per-input predictive std, meaned over z_world dims
        """
        q = self.forward(z_world, action)                     # [B, D, Q]
        qs, _ = torch.sort(q, dim=-1)                         # rearrange (no crossing)
        iqr = qs[..., -1] - qs[..., 0]                        # [B, D] (q_hi - q_lo)
        std = iqr / IQR_TO_STD_10_90
        return std.mean(dim=-1)                              # [B]

    def predictive_variance(self, z_world: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Per-input predictive variance (mean over z_world dims) -- [batch].

        This is the signal the SD-063 E3 commit gate consumes: pass its mean (a
        scalar) as E3.select(conditional_predictive_variance=...). It is also
        the MECH-314b per-candidate source (agent._curiosity_per_candidate_
        uncertainty). HIGH exactly where THIS prediction is about to be wrong
        (the V3-EXQ-712 property the state-blind EMA structurally cannot carry).
        Computed under no_grad -- it is a read for gating, never a training path.

        READINESS DIAGNOSTIC -- read this before trusting an absolute range.
        Each call records `_last_pvar_mean`, `_last_pvar_range` and
        `_last_pvar_relative_spread` (= range / mean) over the batch.
        RELATIVE spread, NOT absolute range, is what discriminates a trained
        head from a random-init one. Measured on a real CausalGridWorldV2
        rollout (seed 71, 4 ep x 80 steps, 2026-08-22), evaluating the head on
        all 5 action classes at one z_world:

            untrained : [0.00486, 0.00478, 0.00536, 0.00467, 0.00450]
                        absolute range 8.63e-04, max/min 1.19x   <- near-uniform
            trained   : [7.39e-04, 5.54e-05, 2.52e-05, 2.98e-04, 3.56e-04]
                        absolute range 7.14e-04, max/min 29.3x   <- differentiated

        Note the direction: the UNTRAINED head has the LARGER absolute range.
        Training lowers the overall predicted spread (the world is more
        predictable than a random init assumes) while raising the relative
        differentiation, so any gate phrased on the absolute range alone --
        including `last_uncertainty_dev_range > 0` as written in the ARC-065
        readiness gate -- is passed by a random-init head and is therefore
        necessary but NOT sufficient. See
        REE_assembly/evidence/planning/mech314bc_percandidate_extension_staged_2026-08-08.md
        section 5.
        """
        with torch.no_grad():
            q = self.forward(z_world, action)                # [B, D, Q]
            qs, _ = torch.sort(q, dim=-1)
            iqr = qs[..., -1] - qs[..., 0]                    # [B, D]
            std = iqr / IQR_TO_STD_10_90
            pvar = (std ** 2).mean(dim=-1)                    # [B]
            if pvar.numel() >= 2:
                mean = float(pvar.mean().item())
                rng = float((pvar.max() - pvar.min()).item())
                self._last_pvar_mean = mean
                self._last_pvar_range = rng
                # Guarded: a genuinely-zero mean spread has no defined relative
                # spread, and reporting inf/nan would corrupt a manifest.
                self._last_pvar_relative_spread = (
                    rng / mean if abs(mean) > 1e-12 else 0.0
                )
            return pvar
