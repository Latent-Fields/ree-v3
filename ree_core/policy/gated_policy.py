"""ARC-062 (weak reading): gated-policy heads + learned context discriminator.

Phase 1 of the rule-apprehension cluster (architecture doc:
REE_assembly/evidence/planning/arc_062_rule_apprehension_plan.md). This module
is the V3-tractable instantiation of the architectural slot MECH-309 says is
required for behavioural diversity in non-stationary environments where strict
parametric-policy learners collapse to a single regime.

ARCHITECTURE (Phase 1, weak reading)

  Two scoring heads (head_0, head_1) sharing E3 candidate features. Each head
  takes a per-candidate feature summary [K, world_dim] and emits a per-candidate
  scalar score-bias contribution [K]. Heads are MLPs (Linear -> ReLU -> Linear)
  with symmetry-broken init (head_0 last-Linear bias initialised positive,
  head_1 negative) so they can differentiate from step 0 under any training
  pressure.

  Small context discriminator: takes a 3-stream input
  (z_world, z_self, z_harm_a), passes it through Linear(world_dim+self_dim+
  harm_a_dim, disc_hidden) -> ReLU -> Linear(disc_hidden, 1) -> sigmoid,
  producing a scalar gating weight w in [0, 1].

  Gated bias: gated_score_bias = w * head_0(features) + (1 - w) * head_1(features)

  Output is composed additively into E3.select(score_bias=...) at the call
  site, parallel to the existing dACC / lateral_pfc / ofc / mech295
  composition (E3 lower-is-better convention).

PULL A SYNTHESIS VERDICTS (resolved defaults, do NOT re-litigate at Phase 1)

  R1 (discriminator input streams) -- DEFAULT: multi-stream (z_world, z_self,
      z_harm_a). Pull A SYNTHESIS verdict 1: Miller & Cohen 2001 + Rigotti 2013
      + Mitchell 2016 (macaque MD network with insular cluster). Single-stream
      z_world-only is the impoverished case and is reserved for the Phase 2
      input-ablation sub-arm (ARM_1a / b / c).

  R2 (discrete heads vs continuous mixed selectivity) -- DEFAULT: N=2 heads
      at Phase 1. Substrate-constrained by SD-054 reef-vs-forage two-mode
      partition. Rigotti 2013 mixed-selectivity caveat is a Phase 4 / GAP-E
      flag, not a Phase 1 blocker.

  R3 (gating site) -- DEFAULT: score_bias level (option iii). Engineering
      reasons dominate: SD-033a substrate is wired, the gradient path through
      E3 score-aggregation is clean. FAIL routes the discriminator output to
      (i) BG-side score-aggregation first, then (ii) trajectory-proposal
      hippocampal preplay seeding, then ARC-063 V4 strong reading.

PHASE 1 SCOPE

  Phase 1 lands the substrate ONLY. There is no connection to SD-033a
  LateralPFCAnalog in Phase 1 -- that wiring is Phase 3 (closes
  commitment_closure GAP-1) per the plan-doc. The gated-policy bias is added
  to E3.select()'s score_bias parameter directly, parallel to the
  dacc_score_bias / lateral_pfc_score_bias composition pattern in
  REEAgent.select_action.

  With use_lateral_pfc_analog=False, the gated-policy bias is the only
  PFC-side contribution.

INPUTS (forward())

  latent_state : LatentState
      Provides z_world [batch, world_dim], z_self [batch, self_dim],
      z_harm_a [batch, harm_a_dim]. z_harm_a is required when
      use_gated_policy=True (Phase 1 commits to multi-stream per Pull A R1
      verdict); when None, the discriminator fallback path emits w=0.5 and
      the implementation logs a diagnostic counter (single-stream sub-arm
      ablation can be wired up cleanly in Phase 2 by passing zero tensors
      for the dropped streams).
  candidate_features : torch.Tensor [K, world_dim]
      Per-candidate first-step z_world summary (caller builds this; same
      shape the lateral_pfc / ofc heads consume).
  simulation_mode : bool, default False
      MECH-094 gate. When True, the module returns (0.5, zeros[K]) without
      advancing diagnostic counters. Match the SD-035 amygdala / MECH-279
      PAG simulation_mode pattern.

OUTPUTS

  GatedPolicyOutput dataclass:
    gating_weight : float in [0, 1] -- scalar discriminator output for the
        current latent state (single-batch).
    gated_score_bias : torch.Tensor [K] -- per-candidate additive bias,
        clamped to [-bias_scale, +bias_scale]. E3 lower-is-better convention.
    head_0_bias : torch.Tensor [K] -- diagnostic raw output of head_0 (no
        gating, no clamp). Useful for the Phase 1 substrate-readiness
        diagnostic + the Phase 2 input-ablation arm.
    head_1_bias : torch.Tensor [K] -- diagnostic raw output of head_1.

MECH-094

  This module has NO internal state buffer (no EMA, no rule_state). The
  simulation_mode argument is for forward-compatibility with MECH-094 gating
  patterns: simulation/replay paths must not get a non-trivial bias from a
  policy module whose discriminator was trained on waking observations.

ARC-063 (V4-deferred) extension path

  ARC-063 strong reading (V4) replaces the discrete-N-heads + soft-
  discriminator architecture with a distributed CandidateRule field with
  continuous tolerance gates. The Phase 4 / GAP-E multi-strategy scaling
  probe is the test of when Phase 1's two-head approximation breaks.
  See evidence/planning/arc_062_rule_apprehension_plan.md Phase 5 (GAP-F /
  GAP-G) for the full ARC-063 V4 deferral.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class GatedPolicyConfig:
    """ARC-062 Phase 1 gated-policy configuration.

    Attributes:
        use_gated_policy : master switch. False = disabled (default,
            backward-compatible). REEAgent does not instantiate GatedPolicy
            when False.
        n_heads : Phase 1 fixes this at 2 (substrate-constrained by SD-054
            two-mode partition per Pull A R2 verdict). The implementation
            asserts on construction; multi-head extension is Phase 4 / GAP-E.
        disc_hidden : context-discriminator hidden-layer width.
        disc_init_scale : init scale for discriminator weights. Small value
            (default 0.1) prevents early discriminator over-commitment to one
            head before either head has differentiated.
        head_hidden : per-head MLP hidden-layer width.
        bias_scale : clamp on |gated_score_bias|. Mirrors the lateral_pfc
            bias_scale default (0.1) so Phase 1 score-bias magnitudes are
            comparable to existing PFC-side contributions.
        head_init_bias_offset : symmetry-broken init magnitude. Head_0's
            last-Linear bias is set to +offset, head_1's to -offset, so the
            two heads can differentiate from step 0 even before the
            discriminator has any training signal.
    """

    use_gated_policy: bool = False
    n_heads: int = 2
    disc_hidden: int = 24
    disc_init_scale: float = 0.1
    head_hidden: int = 32
    bias_scale: float = 0.1
    head_init_bias_offset: float = 0.05
    # ARC-062 GAP-B option-2: head-input first-action one-hot augmentation.
    # When True, each scoring head receives [K, world_dim + first_action_dim]
    # instead of [K, world_dim], bypassing E2 world-forward compression
    # (EXQ-543e autopsy: first-action diversity compressed to 0.22% of
    # z_world magnitude before reaching the z_world-only head).
    # first_action_dim must equal the environment action-space size (set by
    # REEAgent.__init__ from config.e2.action_dim). Default False = no-op,
    # bit-identical backward compat.
    use_first_action_onehot: bool = False
    first_action_dim: int = 0
    # INV-074 / MECH-333 / MECH-334: Phase-3 plasticity-injection
    # crystallization (Nikishin et al. 2023 NeurIPS). When True the module
    # supports a one-shot crystallize() call: head_0/head_1/discriminator
    # params are frozen (requires_grad=False) and a fresh plastic
    # expansion MLP is added whose forward is gated(x) + expansion(
    # x.detach()). The .detach() blocks diversity gradient (dACC /
    # MECH-313 / MECH-314a / MECH-320) from overwriting the crystallized
    # discrimination. crystallize_enabled only ARMS the capability; the
    # actual crystallize() call happens at the infant-curriculum Phase
    # 2->3 transition. Default False = no-op, bit-identical backward
    # compat (forward never touches the expansion branch).
    crystallize_enabled: bool = False
    # Plastic expansion-MLP hidden width. Mirrors head_hidden.
    crystallize_expansion_hidden: int = 32


@dataclass
class GatedPolicyOutput:
    """Per-tick output bundle.

    Attributes:
        gating_weight : float, discriminator output in [0, 1] for the
            current latent state.
        gated_score_bias : Tensor [K], composed bias per candidate (clamped).
        head_0_bias : Tensor [K], raw head_0 output (un-gated, un-clamped).
        head_1_bias : Tensor [K], raw head_1 output (un-gated, un-clamped).
    """

    gating_weight: float
    gated_score_bias: torch.Tensor
    head_0_bias: torch.Tensor
    head_1_bias: torch.Tensor


class GatedPolicy(nn.Module):
    """ARC-062 Phase 1: two scoring heads + 3-stream context discriminator.

    Stateless across ticks (no buffers). Trainable parameters:
    - head_0 / head_1 MLPs (head_hidden -> 1)
    - discriminator MLP (disc_hidden -> 1, sigmoid)
    """

    def __init__(
        self,
        world_dim: int,
        self_dim: int,
        harm_a_dim: int,
        config: Optional[GatedPolicyConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config if config is not None else GatedPolicyConfig()
        if self.config.n_heads != 2:
            raise ValueError(
                "ARC-062 Phase 1 fixes n_heads at 2 (substrate-constrained by "
                "SD-054 reef-vs-forage two-mode partition; multi-head "
                "extension is Phase 4 / GAP-E). "
                f"Got n_heads={self.config.n_heads}."
            )
        self.world_dim = world_dim
        self.self_dim = self_dim
        self.harm_a_dim = harm_a_dim

        head_hidden = self.config.head_hidden
        disc_hidden = self.config.disc_hidden
        bias_offset = self.config.head_init_bias_offset

        # ARC-062 GAP-B option-2: expand head input to include first-action
        # one-hot when use_first_action_onehot=True. Default (False) keeps
        # head_in_dim = world_dim for exact backward compat.
        if self.config.use_first_action_onehot and self.config.first_action_dim > 0:
            head_in_dim = world_dim + self.config.first_action_dim
        else:
            head_in_dim = world_dim
        self._head_in_dim = head_in_dim

        # Two scoring heads. Each takes per-candidate features and emits a
        # per-candidate scalar [K, 1] -> squeeze to [K].
        # Input: [K, world_dim] (default) or [K, world_dim+action_dim] (option-2).
        self.head_0 = nn.Sequential(
            nn.Linear(head_in_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )
        self.head_1 = nn.Sequential(
            nn.Linear(head_in_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )
        # Symmetry-broken init: head_0 +offset, head_1 -offset on the LAST
        # Linear's bias term. Heads can differentiate under any training
        # pressure from step 0 -- avoids the all-zero degenerate equilibrium
        # where w*head_0 + (1-w)*head_1 = head_0 = head_1 = 0 for any w.
        with torch.no_grad():
            self.head_0[-1].bias.fill_(bias_offset)
            self.head_1[-1].bias.fill_(-bias_offset)

        # Context discriminator: 3-stream input -> hidden -> sigmoid scalar.
        # Per Pull A R1 verdict (Miller-Cohen 2001 explicit "inputs, internal
        # states, and outputs" + Rigotti 2013 single-cell mixed selectivity
        # + Mitchell 2016 macaque MD insular cluster).
        disc_in_dim = world_dim + self_dim + harm_a_dim
        self.discriminator = nn.Sequential(
            nn.Linear(disc_in_dim, disc_hidden),
            nn.ReLU(),
            nn.Linear(disc_hidden, 1),
            nn.Sigmoid(),
        )
        # Scale discriminator weights down so the early sigmoid output sits
        # near 0.5 (no premature head commitment). disc_init_scale=0.1
        # multiplies the standard PyTorch init.
        with torch.no_grad():
            for layer in self.discriminator:
                if isinstance(layer, nn.Linear):
                    layer.weight.mul_(self.config.disc_init_scale)
                    layer.bias.mul_(self.config.disc_init_scale)

        # Diagnostics (not persisted, not part of state_dict contract).
        self._last_gating_weight: float = 0.5
        self._last_bias_abs_mean: float = 0.0
        self._last_n_simulation_skips: int = 0
        self._last_z_harm_a_was_none: bool = False
        self._last_onehot_was_none: bool = False

        # INV-074 / MECH-334 crystallization state. The expansion module
        # is created lazily by crystallize(); until then forward() never
        # references it (bit-identical pre-crystallization).
        self._crystallized: bool = False
        self.expansion: Optional[nn.Module] = None
        self._n_frozen_params: int = 0
        self._n_crystallize_calls: int = 0

    # ------------------------------------------------------------------
    # INV-074 / MECH-334: Phase-3 plasticity-injection crystallization
    # ------------------------------------------------------------------
    def crystallize(self) -> dict:
        """Freeze the established discrimination and add a plastic channel.

        One-shot, idempotent. Called by the infant-curriculum Phase 2->3
        transition hook (the experiment harness wires a callback). After
        this returns:

        - head_0 / head_1 / discriminator params have requires_grad=False
          (the discrimination ARM_2 established during the open window is
          crystallized -- diversity gradient can no longer overwrite it,
          the MECH-334 closure / EWC-write-protect analog for the policy
          heads).
        - self.expansion is a fresh plastic MLP with the SAME input
          contract as the scoring heads, last-Linear ZERO-initialised so
          the module's output is bit-identical at the instant of
          crystallization (the expansion contributes exactly 0 until
          diversity gradient grows it).
        - forward() now returns gated(x) + expansion(x.detach()); the
          .detach() is essential -- it prevents the expansion's gradient
          from flowing back into the frozen heads' upstream feature
          producers (Nikishin et al. 2023 NeurIPS plasticity injection).

        Requires crystallize_enabled=True (config arms the capability).
        Idempotent: a second call is a no-op (counter advanced only).

        Returns:
            dict with crystallized (bool), n_frozen_params (int),
            n_expansion_params (int), was_already_crystallized (bool).
        """
        self._n_crystallize_calls += 1
        if not self.config.crystallize_enabled:
            # Capability not armed -- refuse silently (caller installs the
            # Phase-3 hook only when crystallize_at_phase3=True, so this
            # path is defensive). Bit-identical: nothing frozen, no
            # expansion, forward() unchanged.
            return {
                "crystallized": False,
                "n_frozen_params": 0,
                "n_expansion_params": 0,
                "was_already_crystallized": False,
                "reason": "crystallize_enabled=False",
            }
        if self._crystallized:
            return {
                "crystallized": True,
                "n_frozen_params": self._n_frozen_params,
                "n_expansion_params": sum(
                    p.numel() for p in self.expansion.parameters()
                ) if self.expansion is not None else 0,
                "was_already_crystallized": True,
            }

        # Freeze the crystallized discrimination.
        n_frozen = 0
        for module in (self.head_0, self.head_1, self.discriminator):
            for p in module.parameters():
                p.requires_grad = False
                n_frozen += int(p.numel())
        self._n_frozen_params = n_frozen

        # Fresh plastic expansion channel. Same input contract as the
        # heads (head_in_dim). Last-Linear zero-init -> output is exactly
        # zero at the crystallization instant (bit-identical transition);
        # diversity gradient (dACC / MECH-313 / MECH-314a / MECH-320)
        # grows it thereafter.
        h = int(self.config.crystallize_expansion_hidden)
        # device/dtype: follow head_0's first Linear weight.
        ref_w = self.head_0[0].weight
        self.expansion = nn.Sequential(
            nn.Linear(self._head_in_dim, h),
            nn.ReLU(),
            nn.Linear(h, 1),
        ).to(device=ref_w.device, dtype=ref_w.dtype)
        with torch.no_grad():
            self.expansion[-1].weight.zero_()
            self.expansion[-1].bias.zero_()

        self._crystallized = True
        return {
            "crystallized": True,
            "n_frozen_params": n_frozen,
            "n_expansion_params": sum(
                p.numel() for p in self.expansion.parameters()
            ),
            "was_already_crystallized": False,
        }

    def expansion_parameters(self):
        """Iterator over the plastic expansion params (post-crystallize).

        The experiment's post-Phase-3 optimizer targets these (mirrors
        the SD-033a GAP-D lateral_pfc.bias_head_parameters() pattern).
        Empty iterator before crystallize() / when expansion is None.
        """
        if self.expansion is None:
            return iter(())
        return self.expansion.parameters()

    @property
    def crystallized(self) -> bool:
        """True once crystallize() has frozen the heads + added expansion."""
        return self._crystallized

    # ------------------------------------------------------------------
    # Forward path
    # ------------------------------------------------------------------
    def forward(
        self,
        z_world: torch.Tensor,
        z_self: torch.Tensor,
        z_harm_a: Optional[torch.Tensor],
        candidate_features: torch.Tensor,
        first_action_onehots: Optional[torch.Tensor] = None,
        simulation_mode: bool = False,
    ) -> GatedPolicyOutput:
        """Compute the gated per-candidate score bias.

        Args:
            z_world : [batch, world_dim] -- exteroceptive latent.
            z_self  : [batch, self_dim]  -- proprioceptive/interoceptive latent.
            z_harm_a : Optional [batch, harm_a_dim] -- affective harm latent.
                When None, the discriminator falls back to w=0.5 (uniform
                head mixture) and a diagnostic flag is raised. The Phase 2
                input-ablation arm wires single-stream variants by passing
                zero tensors of the right shape, NOT by passing None.
            candidate_features : [K, world_dim] -- per-candidate first-step
                z_world summary (caller builds this from trajectory rollouts).
            first_action_onehots : Optional [K, action_dim] -- per-candidate
                first-action one-hot vectors. Required when
                use_first_action_onehot=True; concatenated onto
                candidate_features before feeding the scoring heads (ARC-062
                GAP-B option-2). When None with use_first_action_onehot=True,
                falls back to zeros and raises the _last_onehot_was_none
                diagnostic. Ignored when use_first_action_onehot=False.
            simulation_mode : bool, default False. MECH-094 gate. When True,
                returns (0.5, zeros[K], zeros[K], zeros[K]) without updating
                diagnostic counters.

        Returns:
            GatedPolicyOutput.
        """
        K = int(candidate_features.shape[0])
        device = candidate_features.device
        dtype = candidate_features.dtype

        if simulation_mode:
            self._last_n_simulation_skips += 1
            zeros = torch.zeros(K, device=device, dtype=dtype)
            return GatedPolicyOutput(
                gating_weight=0.5,
                gated_score_bias=zeros,
                head_0_bias=zeros.clone(),
                head_1_bias=zeros.clone(),
            )

        # Discriminator input. Multi-stream (Pull A R1 default).
        # Single-batch path: ensure 2D shapes.
        zw = z_world if z_world.dim() == 2 else z_world.unsqueeze(0)
        zs = z_self if z_self.dim() == 2 else z_self.unsqueeze(0)
        if z_harm_a is None:
            # Fallback: synthesise a zero z_harm_a so the discriminator forward
            # is well-defined; flag diagnostically. Phase 2 input-ablation
            # arm passes explicit zero tensors and exercises this code path
            # under cleaner control.
            self._last_z_harm_a_was_none = True
            za = torch.zeros(zw.shape[0], self.harm_a_dim,
                             device=device, dtype=dtype)
        else:
            self._last_z_harm_a_was_none = False
            za = z_harm_a if z_harm_a.dim() == 2 else z_harm_a.unsqueeze(0)

        # Pool across batch dim with mean (single-batch agent loop typical;
        # batch>1 falls back to a coherent single scalar w per tick).
        disc_input = torch.cat(
            [zw.mean(dim=0, keepdim=True),
             zs.mean(dim=0, keepdim=True),
             za.mean(dim=0, keepdim=True)],
            dim=-1,
        )  # [1, world_dim+self_dim+harm_a_dim]
        w_tensor = self.discriminator(disc_input).squeeze()  # scalar
        w = float(w_tensor.detach().item())

        # ARC-062 GAP-B option-2: augment head input with first-action one-hot.
        # Mirrors the z_harm_a=None fallback pattern for the None-guard.
        if self.config.use_first_action_onehot and self.config.first_action_dim > 0:
            if first_action_onehots is None:
                self._last_onehot_was_none = True
                first_action_onehots = torch.zeros(
                    K, self.config.first_action_dim,
                    device=device, dtype=dtype,
                )
            else:
                self._last_onehot_was_none = False
            head_input = torch.cat(
                [candidate_features,
                 first_action_onehots.to(device=device, dtype=dtype)],
                dim=-1,
            )  # [K, world_dim + action_dim]
        else:
            self._last_onehot_was_none = False
            head_input = candidate_features  # [K, world_dim]

        # Per-head bias.
        head_0_raw = self.head_0(head_input).squeeze(-1)  # [K]
        head_1_raw = self.head_1(head_input).squeeze(-1)  # [K]

        # Composed gated bias. Use w_tensor (not float w) so the gradient
        # path through the discriminator is preserved when grad is enabled.
        gated_bias_raw = w_tensor * head_0_raw + (1.0 - w_tensor) * head_1_raw

        # INV-074 / MECH-334: post-crystallization plastic channel
        # (Nikishin et al. 2023). forward = frozen_gated(x) +
        # expansion(x.detach()). The .detach() blocks the expansion's
        # gradient from reaching the (frozen) heads' upstream feature
        # producers; the heads themselves are also requires_grad=False so
        # the crystallized discrimination cannot be overwritten by the
        # routed diversity gradient. Bit-identical until crystallize()
        # runs (zero-initialised last Linear -> +0 at the transition).
        if self._crystallized and self.expansion is not None:
            expansion_bias = self.expansion(head_input.detach()).squeeze(-1)
            gated_bias_raw = gated_bias_raw + expansion_bias

        gated_bias = gated_bias_raw.clamp(
            min=-self.config.bias_scale,
            max=+self.config.bias_scale,
        )

        # Diagnostics (no_grad on the cached scalars).
        with torch.no_grad():
            self._last_gating_weight = w
            self._last_bias_abs_mean = float(gated_bias.abs().mean().item())

        return GatedPolicyOutput(
            gating_weight=w,
            gated_score_bias=gated_bias,
            head_0_bias=head_0_raw,
            head_1_bias=head_1_raw,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset per-episode diagnostic counters.

        Does NOT un-crystallize: crystallization is a developmental
        closure (persists across episodes once Phase 3 is reached), not
        per-episode state. _crystallized / expansion are intentionally
        left untouched here.
        """
        self._last_gating_weight = 0.5
        self._last_bias_abs_mean = 0.0
        self._last_n_simulation_skips = 0
        self._last_z_harm_a_was_none = False
        self._last_onehot_was_none = False

    def get_state(self) -> dict:
        """Diagnostic snapshot for experiment manifests."""
        return {
            "last_gating_weight": self._last_gating_weight,
            "last_bias_abs_mean": self._last_bias_abs_mean,
            "last_n_simulation_skips": self._last_n_simulation_skips,
            "last_z_harm_a_was_none": self._last_z_harm_a_was_none,
            "last_onehot_was_none": self._last_onehot_was_none,
            "crystallized": self._crystallized,
            "n_crystallize_calls": self._n_crystallize_calls,
            "n_frozen_params": self._n_frozen_params,
            "expansion_active": self.expansion is not None,
        }
