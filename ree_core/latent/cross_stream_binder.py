"""Cross-stream binding factor (cross_stream_binding_substrate).

Installs a genuine shared latent cause between the z_self (`states`) and z_world
(`world_states`) rollout streams produced by E2FastPredictor.rollout_with_world.

WHY THIS EXISTS
---------------
failure_autopsy_V3-EXQ-641a_2026-06-06 adjudicated V3-EXQ-641a as a substrate
ceiling: with a fair contrast-matched control, cross-stream phase-alignment
C(tau) over the two rollout streams carried NO selection information beyond the
prediction-error cost E. The root cause is structural -- in rollout_with_world
the two streams advance through two INDEPENDENT forward models
(predict_next_self, world_forward) whose only shared input is the action. So any
co-variation the harness reads as "coherence" is exactly what E already scores.

This module makes the two streams genuinely bound: at each rollout step it
derives a shared factor from the JOINT (z_self, z_world) state and injects the
SAME additive perturbation into both post-transition states. The two streams'
step-deltas then share an explicit common component, so real cross-stream
coherence carries a per-candidate signature that a shuffle of the coherence
values destroys.

TWO BINDER MODES
----------------
FIXED (learned=False, the original 2026-07-08 build): W_enc/W_out are FIXED
  random projections. V3-EXQ-720 (strength 0.5) showed a fixed field is
  SYMBOL-COMPLETE but FUNCTION-PARTIAL: coherence-specificity lifted 1/6 (641a
  unbound) -> 3/6 (720 bound) but did NOT clear the 4/6 SPEC gate, and n_rebind
  stayed 0 across 641/641a/720. Nothing SHAPES a random projection so that real
  cross-stream conjunctions are robustly more selection-informative than a
  contrast-matched shuffle. This path is preserved byte-identical.

LEARNED (learned=True, 2026-07-09; cross_stream_binding_substrate V4 next-step
  per failure_autopsy_V3-EXQ-720_2026-07-09): phi_self / phi_world are PLASTIC
  projections trained by contrastive co-encoding (InfoNCE): within-tick observed
  (z_self, z_world) pairs are POSITIVES; in-batch shuffled pairs are NEGATIVES.
  The shared factor is the MULTIPLICATIVE conjunction g_t = tanh(phi_self(z_self)
  * phi_world(z_world)) -- a coincidence/AND detector that is large only when
  both learned projections co-activate (a genuine binding), and whose value a
  shuffle destroys. This is the load-bearing biology divergence the 720 autopsy
  named: binding-by-synchrony / communication-through-coherence is LEARNED and
  plastic (Hebbian: fire-together -> wire-together), not a fixed field. The
  learned binder REQUIRES phased training (a P0 binder curriculum trains it;
  P1 freezes it and runs the 641a measurement).

  CONVERGENCE REPAIR (failure_autopsy_V3-EXQ-725_2026-07-09): the first learned
  build scored UN-normalized projections. In the slow-drift bipartite gridworld
  the observed latents are near-collinear buffer-wide (cos ~0.99), so the
  dot-product InfoNCE logit was dominated by the near-constant projection
  MAGNITUDE and carried no per-pair contrast -- the loss pinned at chance
  log(64)=4.16 (V3-EXQ-725 observed 3.75-3.96) and the binder never trained (an
  untrained-substrate artifact, not a coherence verdict). The repair L2-NORMALIZES
  phi_self/phi_world before the dot (COSINE InfoNCE, SimCLR-standard): direction,
  not magnitude, is scored, exposing the residual conjunction signal. The loss
  then drops to 0.65-0.80 of chance across seeds. The substrate reports
  binder_converged = loss_ema < conv_frac*log(batch) so a retest gates on
  convergence (not the vacuous n_learn_steps>1 check). See
  evidence/planning/binder_convergence_probe_2026-07-09.md.

DESIGN NOTES
------------
- SAME perturbation into both streams. Two independent random projections of the
  shared factor would average to ~0 cross-stream alignment; adding the identical
  perturbation b_t into the first min(self_dim, world_dim) components of both
  streams guarantees a genuine shared delta-component. Unchanged across modes so
  the e2_fast wiring (factor -> couple) is mode-agnostic.
- Theta-gated (MECH-089 theta-gamma nesting). The per-step shared code b_t is the
  gamma-rate content; the cosine theta window it is scaled by is the theta cycle
  it nests within. Unchanged across modes.
- ENCODER ISOLATION. The binder trains on DETACHED (z_self, z_world) so gradient
  never leaks into E1/E2's online self/world models. The coupling alters only the
  imagined rollout states used for candidate scoring; the online model updates
  (record_transition) use OBSERVED z_self, not rollout states.
- SUBSTRATE-LEVEL REBINDING PROBE (learned only). binding_score() exposes the
  learned binding affinity of a (z_self, z_world) conjunction; rebinding_probe()
  reads whether a competing config OVERTAKES under perturbation. A fixed field
  cannot be perturbed into a rebind (all conjunctions get the same undiscriminating
  field -> n_rebind=0). The learned binder makes the binding intake's own
  falsifier exercisable AT THE SUBSTRATE, folded in rather than a separate
  harness instrument.
- MECH-094 does NOT newly apply: no memory-write surface is added and
  hypothesis_tag semantics are unchanged. Contrastive training uses observed
  (waking) pairs, not replayed/hypothesis content.

See docs/architecture/sd_cross_stream_binding_substrate.md.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossStreamBinder(nn.Module):
    """Shared-latent-factor binding coupling for E2 rollouts.

    factor(z_self, z_world) -> g_t          shared factor from the joint state
    couple(z_self, z_world, g_t, t) -> (z_self', z_world')
        adds the SAME theta-gated perturbation b_t = W_out . g_t (fixed) or
        to_common(g_t) (learned) into the first min(self_dim, world_dim)
        components of both streams.

    Fixed mode (learned=False): W_enc/W_out are fixed nn.Linear layers; the
    module is used under eval and no training method fires. Byte-identical to the
    original 2026-07-08 build.

    Learned mode (learned=True): phi_self/phi_world are plastic; observe() buffers
    detached observed pairs and learn_step() runs one contrastive (InfoNCE)
    optimizer step. binding_score()/rebinding_probe() expose the learned affinity.
    """

    def __init__(
        self,
        self_dim: int,
        world_dim: int,
        bind_dim: int = 16,
        strength: float = 0.15,
        theta_period: int = 4,
        learned: bool = False,
        lr: float = 1e-3,
        temperature: float = 0.5,
        buffer_size: int = 512,
        batch: int = 64,
        conv_frac: float = 0.85,
    ) -> None:
        super().__init__()
        self.self_dim = int(self_dim)
        self.world_dim = int(world_dim)
        self.bind_dim = int(bind_dim)
        self.strength = float(strength)
        # theta_period must be >= 1; a period of 1 degenerates the gate to a
        # constant 1.0 window (cos(2*pi*t) == 1), i.e. ungated coupling.
        self.theta_period = max(1, int(theta_period))
        # Shared output space = the overlap both streams can receive the SAME
        # perturbation in.
        self.bind_out_dim = min(self.self_dim, self.world_dim)
        self.learned = bool(learned)
        self.temperature = max(1e-3, float(temperature))
        self.buffer_size = max(2, int(buffer_size))
        self.batch = max(2, int(batch))
        # Convergence gate fraction (failure_autopsy_V3-EXQ-725 repair): the
        # binder is "converged" when the smoothed InfoNCE loss is below
        # conv_frac * log(effective_batch) (chance).
        self.conv_frac = float(conv_frac)

        if not self.learned:
            # FIXED mode (byte-identical to the 2026-07-08 build):
            # joint state -> shared factor; shared factor -> common perturbation.
            self.encode = nn.Linear(self.self_dim + self.world_dim, self.bind_dim)
            self.to_common = nn.Linear(self.bind_dim, self.bind_out_dim, bias=False)
        else:
            # LEARNED mode: separate plastic projections per stream so the shared
            # factor is a MULTIPLICATIVE conjunction (coincidence detector). The
            # SAME to_common maps the conjunction back into the overlap space so
            # couple() is mode-agnostic.
            self.phi_self = nn.Linear(self.self_dim, self.bind_dim)
            self.phi_world = nn.Linear(self.world_dim, self.bind_dim)
            self.to_common = nn.Linear(self.bind_dim, self.bind_out_dim, bias=False)
            self._optimizer = torch.optim.Adam(self.parameters(), lr=float(lr))
            # Detached observed-pair buffer for the P0 contrastive curriculum.
            self._buf_self: List[torch.Tensor] = []
            self._buf_world: List[torch.Tensor] = []
            self._n_learn_steps = 0
            self._last_loss: Optional[float] = None
            # Smoothed loss + last effective chance floor for the convergence gate.
            self._loss_ema: Optional[float] = None
            self._last_chance: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Shared factor + coupling (mode-agnostic external interface)         #
    # ------------------------------------------------------------------ #

    def factor(self, z_self: torch.Tensor, z_world: torch.Tensor) -> torch.Tensor:
        """Shared binding factor g_t from the joint pre-transition state.

        Args:
            z_self:  [batch, self_dim]
            z_world: [batch, world_dim]
        Returns:
            g_t: [batch, bind_dim]
        """
        if not self.learned:
            joint = torch.cat([z_self, z_world], dim=-1)
            return torch.tanh(self.encode(joint))
        # Learned: multiplicative conjunction of the two plastic projections --
        # large only when both stream projections co-activate (genuine binding).
        h_self = self.phi_self(z_self)
        h_world = self.phi_world(z_world)
        return torch.tanh(h_self * h_world)

    def _theta_gate(self, t: int) -> float:
        """MECH-089 theta window in [0, 1]: 0.5*(1 + cos(2*pi*t/theta_period))."""
        return 0.5 * (1.0 + math.cos(2.0 * math.pi * float(t) / float(self.theta_period)))

    def couple(
        self,
        z_self: torch.Tensor,
        z_world: torch.Tensor,
        g_t: torch.Tensor,
        t: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inject the SAME theta-gated perturbation b_t into both streams.

        Args:
            z_self:  [batch, self_dim]  (post base-transition)
            z_world: [batch, world_dim] (post base-transition)
            g_t:     [batch, bind_dim]  from factor() on the pre-transition state
            t:       rollout step index (theta phase)
        Returns:
            (z_self', z_world') with b_t added into the first bind_out_dim dims.
        """
        k_t = self.strength * self._theta_gate(t)
        if k_t == 0.0:
            return z_self, z_world
        b_t = self.to_common(g_t)  # [batch, bind_out_dim]
        d = self.bind_out_dim
        # Out-of-place add on the overlap slice so autograd / downstream reads are
        # clean and the untouched tail dimensions pass through unchanged.
        z_self = torch.cat([z_self[..., :d] + k_t * b_t, z_self[..., d:]], dim=-1)
        z_world = torch.cat([z_world[..., :d] + k_t * b_t, z_world[..., d:]], dim=-1)
        return z_self, z_world

    # ------------------------------------------------------------------ #
    # Learned-binder training (P0 curriculum) + binding affinity          #
    # ------------------------------------------------------------------ #

    def binding_score(
        self, z_self: torch.Tensor, z_world: torch.Tensor
    ) -> torch.Tensor:
        """Learned binding affinity of a (z_self, z_world) conjunction.

        score = cos(phi_self(z_self), phi_world(z_world)) -- the L2-normalized
        bilinear form (an InfoNCE cosine logit, temperature omitted). High when
        the two learned projections point the same way. The substrate-level read
        the rebinding probe uses. Fixed mode: not defined (returns zeros -- a
        fixed field has no learned affinity).

        NORMALIZATION (failure_autopsy_V3-EXQ-725 repair): the raw bilinear form
        sum_k phi_self_k*phi_world_k is dominated by the near-constant projection
        MAGNITUDE across the highly-collinear latents (buffer-wide cos ~0.99), so
        it carries almost no per-conjunction contrast. L2-normalizing exposes the
        residual DIRECTIONAL conjunction signal -- the same geometry learn_step
        now trains -- so the rebinding probe ranks candidates consistently with
        the trained objective. See binder_convergence_probe_2026-07-09.md.

        Args:
            z_self:  [batch, self_dim]
            z_world: [batch, world_dim]
        Returns:
            score: [batch]
        """
        if not self.learned:
            return torch.zeros(z_self.shape[0], device=z_self.device)
        h_self = F.normalize(self.phi_self(z_self), dim=-1)
        h_world = F.normalize(self.phi_world(z_world), dim=-1)
        return (h_self * h_world).sum(dim=-1)

    def observe(self, z_self: torch.Tensor, z_world: torch.Tensor) -> None:
        """Buffer a DETACHED observed (z_self, z_world) pair for the P0 curriculum.

        No-op in fixed mode. Inputs are detached + cloned so training never leaks
        gradient into E1/E2's encoders and the buffer holds no graph.
        """
        if not self.learned:
            return
        s = z_self.detach().reshape(1, -1).clone()
        w = z_world.detach().reshape(1, -1).clone()
        self._buf_self.append(s)
        self._buf_world.append(w)
        if len(self._buf_self) > self.buffer_size:
            self._buf_self = self._buf_self[-self.buffer_size:]
            self._buf_world = self._buf_world[-self.buffer_size:]

    def learn_step(self) -> Optional[float]:
        """One contrastive (InfoNCE) co-encoding optimizer step. Returns the loss
        (float) or None if not enough buffered pairs / fixed mode.

        POSITIVES = within-tick aligned pairs (the diagonal of the score matrix);
        NEGATIVES = in-batch shuffled pairs (off-diagonal). Symmetric cross
        entropy (self->world and world->self). Trains phi_self/phi_world so
        genuine conjunctions bind and a shuffle collapses.

        NORMALIZATION (failure_autopsy_V3-EXQ-725 repair): phi_self/phi_world are
        L2-normalized before the dot -- a COSINE InfoNCE (SimCLR-standard). The
        latents are near-collinear buffer-wide (cos ~0.99), so the un-normalized
        dot-product logit is dominated by the near-constant projection MAGNITUDE
        and carries no per-pair contrast -> the loss pinned at chance log(batch)
        across 487-1760 steps in V3-EXQ-725 (the untrained-substrate artifact).
        Normalizing scores DIRECTION only, exposing the residual conjunction
        signal; the loss then drops to 0.65-0.80 of chance across seeds (temp 0.2
        deepens the margin). See binder_convergence_probe_2026-07-09.md.
        """
        if not self.learned:
            return None
        n = len(self._buf_self)
        if n < 2:
            return None
        b = min(self.batch, n)
        idx = torch.randperm(n)[:b]
        z_self = torch.cat([self._buf_self[i] for i in idx.tolist()], dim=0)
        z_world = torch.cat([self._buf_world[i] for i in idx.tolist()], dim=0)

        # L2-normalize the projections (cosine InfoNCE) -- the load-bearing repair.
        h_self = F.normalize(self.phi_self(z_self), dim=-1)    # [b, bind_dim]
        h_world = F.normalize(self.phi_world(z_world), dim=-1)  # [b, bind_dim]
        # Full pairwise score matrix: logits[i, j] = cos(phi_self_i, phi_world_j).
        logits = (h_self @ h_world.t()) / self.temperature  # [b, b]
        targets = torch.arange(b, device=logits.device)
        loss = 0.5 * (
            F.cross_entropy(logits, targets)
            + F.cross_entropy(logits.t(), targets)
        )
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._n_learn_steps += 1
        self._last_loss = float(loss.detach().item())
        # Smoothed loss (EMA decay 0.9) + effective chance floor log(b) for a
        # robust binder_converged read (a single-step loss is noisy).
        if self._loss_ema is None:
            self._loss_ema = self._last_loss
        else:
            self._loss_ema = 0.9 * self._loss_ema + 0.1 * self._last_loss
        self._last_chance = math.log(b)
        return self._last_loss

    def rebinding_probe(
        self,
        z_self: torch.Tensor,
        z_world_candidates: List[torch.Tensor],
        perturbation: torch.Tensor,
    ) -> dict:
        """Substrate-level rebinding falsifier (learned mode only).

        Given a pool of candidate world-configs bound to one anchor z_self, does
        the argmax-binding-affinity candidate CHANGE when the SHARED ANCHOR z_self
        is perturbed -- i.e. does a competing config OVERTAKE the currently-bound
        one? This is the binding intake's own falsifier: a competing configuration
        overtakes the current one under perturbation.

        The perturbation is applied to the ANCHOR (z_self), NOT uniformly to the
        candidate world-configs. That is load-bearing: binding_score is bilinear,
        so a uniform additive perturbation on the candidates shifts EVERY score by
        the same candidate-independent constant <phi_self(z_self), W_world . p> and
        can never flip the argmax. Perturbing the anchor gives a per-candidate
        shift <W_self . p, phi_world(c)> that DOES vary with c -- so a competitor
        can overtake. A FIXED field cannot express any of this (binding_score is
        identically 0, undiscriminating), which is exactly why n_rebind stayed 0
        across 641/641a/720. The learned binder makes the falsifier exercisable at
        the substrate; the retest reads n_rebind through this probe.

        Args:
            z_self:              [1, self_dim] (or [self_dim]) shared anchor
            z_world_candidates:  list of [., world_dim] candidate world configs
            perturbation:        [self_dim] additive perturbation on the anchor
        Returns:
            {"clean_pick", "perturbed_pick", "rebound": bool,
             "clean_scores", "perturbed_scores"} or {"rebound": False, ...} if
             fixed mode / degenerate pool.
        """
        if not self.learned or len(z_world_candidates) < 2:
            return {"rebound": False, "clean_pick": None, "perturbed_pick": None,
                    "clean_scores": [], "perturbed_scores": []}
        zs = z_self.detach().reshape(1, -1)
        zs_pert = zs + perturbation.detach().reshape(1, -1)
        with torch.no_grad():
            cands = [c.detach().reshape(1, -1) for c in z_world_candidates]
            clean_scores = [float(self.binding_score(zs, c).item()) for c in cands]
            perturbed_scores = [
                float(self.binding_score(zs_pert, c).item()) for c in cands
            ]
        clean_pick = max(range(len(clean_scores)), key=lambda i: clean_scores[i])
        perturbed_pick = max(
            range(len(perturbed_scores)), key=lambda i: perturbed_scores[i]
        )
        return {
            "clean_pick": clean_pick,
            "perturbed_pick": perturbed_pick,
            "rebound": bool(clean_pick != perturbed_pick),
            "clean_scores": clean_scores,
            "perturbed_scores": perturbed_scores,
        }

    @property
    def n_learn_steps(self) -> int:
        return getattr(self, "_n_learn_steps", 0)

    @property
    def last_loss(self) -> Optional[float]:
        return getattr(self, "_last_loss", None)

    @property
    def loss_ema(self) -> Optional[float]:
        """EMA-smoothed InfoNCE loss (decay 0.9), or None if untrained/fixed."""
        return getattr(self, "_loss_ema", None)

    @property
    def chance_floor(self) -> Optional[float]:
        """log(effective_batch) of the last learn_step -- the InfoNCE chance floor."""
        return getattr(self, "_last_chance", None)

    @property
    def binder_converged(self) -> bool:
        """True when the smoothed loss is below conv_frac * chance.

        The hard convergence gate a learned-binder retest MUST check (replacing
        the vacuous n_learn_steps>1 readiness check that green-lit the untrained
        binder in V3-EXQ-725). False in fixed mode or before any training.
        """
        if not getattr(self, "learned", False):
            return False
        ema = getattr(self, "_loss_ema", None)
        chance = getattr(self, "_last_chance", None)
        if ema is None or chance is None:
            return False
        return ema < self.conv_frac * chance
