"""
E3 Trajectory Selector — V3 Implementation

V3 extensions:
1. harm_eval(z_world) — new method; E3 is the correct locus for harm prediction,
   NOT E2. Used in SD-003 V3 attribution pipeline:
     harm_actual = e3.harm_eval(z_world_actual)
     harm_cf     = e3.harm_eval(z_world_cf)
     causal_sig  = harm_actual - harm_cf

2. Dynamic precision (ARC-016):
   Precision is derived from E3's own prediction error variance (EMA of MSE),
   NOT hardcoded. This is required to test ARC-016 (dynamic precision
   behavioural distinction).
   committed = running_variance < commit_threshold  (variance-space, fixed 2026-03-18)

3. Operates over z_world (SD-005):
   E3's input domain is z_world (exteroceptive world model), not z_gamma.

E3 trains on harm + goal error over z_world.

Scoring equation J(ζ) = F(ζ) + λ·M(ζ) + ρ·Φ_R(ζ) remains a working
hypothesis — see ARCHITECTURE NOTE below. All weights are placeholder.
"""

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ree_core.utils.config import E3Config
from ree_core.policy.noisy_selection_head import (
    NoisySelectionHead,
    NoisySelectionHeadConfig,
)

# ARC-108 JOB-1 step-1 (learned dopamine-gated E3 gating): the modulatory channels
# that feed _modulatory_accum at the select() composition site. Each is a genuinely
# separable per-candidate bias term added to the accumulator; the learned per-channel
# weight vector w_chan is indexed by this fixed registry (a channel absent on a given
# tick simply does not contribute, so w_chan stays a stable learned object). NOTE: at
# this composition site "score_bias" is already the composed dACC+lPFC+OFC+MECH-295+
# MECH-314+MECH-320 chain (summed UPSTREAM in agent.py before reaching select()); a
# finer per-head channel split is a documented follow-on, out of step-1 scope.
_LCG_CHANNEL_NAMES: Tuple[str, ...] = ("score_bias", "mech341", "route")
_LCG_CHANNEL_INDEX: Dict[str, int] = {n: i for i, n in enumerate(_LCG_CHANNEL_NAMES)}
# w_chan init value x s.t. softplus(x) == 1.0 exactly -> bit-identical unweighted
# accumulator at init: ln(1 + e^x) = 1  =>  x = ln(e - 1).
_LCG_W_INIT: float = math.log(math.e - 1.0)

# MECH-451 finer-channel registry: the single ARC-108 "score_bias" slot exploded
# into its genuinely-separable constituents (the dACC+lPFC+OFC+MECH-295+MECH-320+
# gated_policy chain summed UPSTREAM in agent.py before reaching select()), each a
# separately-learnable channel. "residual" captures everything ELSE summed into
# score_bias that is not broken out (MECH-314 curiosity / MECH-353 blocked-agency /
# SD-058 avoidance / SD-059 escape-affordance / any future term), computed by
# subtraction so the finer decomposition is EXHAUSTIVE of score_bias -- the
# bit-identical-at-init guarantee in the authority/shortlist path (the recomposed
# _modulatory_accum at softplus==1 reproduces the summed score_bias EXACTLY). The
# existing mech341 + route channels are preserved unchanged. The named constituent
# channels feeding score_bias_channels (everything before "residual") are the
# FINER_NAMED_CHANNELS; a constituent absent on a given tick (e.g. mech295 with no
# active goal) simply does not contribute and earns no eligibility, so its
# w_chan_finer entry stays stable (the ARC-108 "channel absent -> does not
# contribute" semantic). See MECH-451 / EXP-0391.
FINER_NAMED_CHANNELS: Tuple[str, ...] = (
    "ofc", "dacc", "lpfc", "vigour", "liking", "gated_policy",
)
_FCG_CHANNEL_NAMES: Tuple[str, ...] = (
    FINER_NAMED_CHANNELS + ("residual", "mech341", "route")
)
_FCG_CHANNEL_INDEX: Dict[str, int] = {
    n: i for i, n in enumerate(_FCG_CHANNEL_NAMES)
}

# ARC-110: parallel segregated cortico-BG-thalamic loops. The MOTOR loop is F
# (raw_scores) itself; the modulatory channels split between the ASSOCIATIVE
# (dACC conflict + lateral-PFC rule-evidence) and LIMBIC (OFC value + liking +
# vigour) loops. Channels not named here fall to loop_segregation_default_loop.
# Keys match BOTH registries: the finer (MECH-451) channel names break score_bias
# into constituents that can be assigned to loops; with the coarse ARC-108 base
# registry only "score_bias" exists (one lump -> the default loop), so loop
# segregation is meaningful only WITH finer-channel gating on (the validation runs
# the full stack). Functional translation of Alexander/DeLong/Strick (ARC-106).
_LOOP_NAMES: Tuple[str, ...] = ("motor", "associative", "limbic")
_LOOP_DEFAULT_CHANNEL_MAP: Dict[str, str] = {
    "dacc": "associative",
    "lpfc": "associative",
    "ofc": "limbic",
    "liking": "limbic",
    "vigour": "limbic",
    # gated_policy / residual / mech341 / route / score_bias -> default loop
}
from ree_core.predictors.e2_fast import Trajectory
from ree_core.residue.field import ResidueField
from ree_core.goal import GoalState


@dataclass
class SelectionResult:
    """Result of trajectory selection.

    selected_trajectory: chosen trajectory
    selected_index:      index within candidates
    selected_action:     first action from selected trajectory
    scores:              all trajectory scores [num_candidates]
    precision:           current dynamic precision value
    committed:           whether commitment threshold was met
    log_prob:            log-probability for REINFORCE (connected to computation graph)
    """
    selected_trajectory: Trajectory
    selected_index: int
    selected_action: torch.Tensor
    scores: torch.Tensor
    precision: float
    committed: bool
    log_prob: Optional[torch.Tensor] = None
    urgency: float = 0.0  # SD-011: z_harm_a urgency applied to commit threshold


def project_channel_range(features: torch.Tensor) -> torch.Tensor:
    """
    modulatory-bias-selection-authority AMEND (route-range, 569f/661/654a).

    Parameter-free, range-preserving projection of a channel-under-test's
    per-candidate representation into a per-candidate scalar bias [K], so that a
    channel whose REPRESENTATION carries genuine cross-candidate range (e.g.
    cand_world_summaries spread 0.196; minted rule_state) yields a per-candidate
    bias that actually enters the modulatory accumulator the E3 selection
    authority rescales. (569f/661/654a: the channel range existed in the
    representation but was flattened by the consuming bias head before it reached
    the bias term -- so the authority had nothing to amplify.)

    - features [K, D] (a per-candidate feature matrix): center across the K
      candidates and project onto the leading right-singular vector of the
      centered matrix -> [K] signed scalar capturing the dominant cross-candidate
      variation. Deterministic, no learned parameters; the projection preserves
      the channel's cross-candidate range by construction. (The singular-vector
      sign is arbitrary -- routing makes the channel range REACH and MOVE the
      committed argmax, which is the readiness property; making the movement
      BENEFICIAL is the channel's own trained head, the separate per-claim
      evidence retest.)
    - features [K] (an already-per-candidate bias): returned as-is (identity).

    Returns a 1-D [K] tensor on the same device/dtype as ``features``. A
    degenerate input (K < 2, or zero cross-candidate variation) yields a zeroed
    [K] vector, which the P0 readiness gate reads as below-floor.
    """
    if features.dim() == 1:
        return features
    if features.dim() != 2:
        features = features.reshape(features.shape[0], -1)
    k = features.shape[0]
    if k < 2:
        return features.new_zeros(k)
    centered = features - features.mean(dim=0, keepdim=True)  # [K, D]
    if float(centered.abs().max().item()) <= 0.0:
        return features.new_zeros(k)
    # Leading right-singular vector of the centered matrix; project onto it.
    # SVD on a detached copy (the routing direction is a parameter-free read of
    # the channel structure, not a grad path); the projection itself uses the
    # live centered tensor so any caller-supplied grad still flows if present.
    try:
        _, _, vh = torch.linalg.svd(centered.detach(), full_matrices=False)
        u = vh[0]  # [D] leading right-singular vector
    except Exception:
        # Numerical fallback: per-candidate signed deviation along the pool-mean
        # difference axis (still range-preserving for the common case).
        u = centered.detach().abs().mean(dim=0)
        nrm = float(u.norm().item())
        if nrm <= 0.0:
            return features.new_zeros(k)
        u = u / nrm
    return centered @ u.to(dtype=centered.dtype, device=centered.device)  # [K]


def variance_commit_threshold(config_threshold: float) -> float:
    """
    Return the variance-space commit threshold (ARC-016).

    Commitment fires when running_variance < threshold (low variance = confident).
    Threshold lives in variance space [~0.01-0.05], NOT precision space (~100).
    (Prior precision_to_threshold() was on the wrong scale: returning ~0.703
    vs current_precision ~95 → always committed. Fixed 2026-03-18.)
    """
    return config_threshold


class E3TrajectorySelector(nn.Module):
    """
    E3 Trajectory Selector — V3.

    Evaluates candidate trajectories and selects one by minimising:
        J(ζ) = F(ζ) + λ·M(ζ) + ρ·Φ_R(ζ)   [working hypothesis]

    V3 additions:
    - harm_eval(z_world) for SD-003 attribution
    - Dynamic precision via prediction error variance (ARC-016)
    - All scoring operates over z_world (SD-005)

    # ARCHITECTURE NOTE (V3)
    # -----------------------------------------------------------------
    # The scoring equation is a WORKING HYPOTHESIS. The weights
    # lambda_ethical and rho_residue are placeholder parameters.
    # The scoring function as a whole requires calibration experiments.
    #
    # In V3:
    # - F(ζ): proxy (smoothness + final-state viability). Not settled.
    # - M(ζ): uses harm_eval() on z_world trajectory states. This is
    #   categorically distinct from V2's E2.predict_harm (which belonged
    #   to E2, not E3). harm_eval() is the correct locus per spec §5.4.
    # - Φ_R(ζ): residue field now operates over z_world (SD-005).
    #   Self-change (z_self_delta) does not contribute to residue.
    # -----------------------------------------------------------------
    """

    def __init__(
        self,
        config: Optional[E3Config] = None,
        residue_field: Optional[ResidueField] = None,
    ):
        super().__init__()
        self.config = config or E3Config()
        self.residue_field = residue_field

        world_dim = self.config.world_dim

        # Reality constraint scorer F(ζ): operates on z_world
        self.reality_scorer = nn.Sequential(
            nn.Linear(world_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1),
        )

        # Ethical cost scorer M(ζ): operates on z_world
        self.ethical_scorer = nn.Sequential(
            nn.Linear(world_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1),
        )

        # harm_eval head (SD-003 V3 attribution):
        # Predicts harm of a z_world state. This belongs to E3, NOT E2.
        self.harm_eval_head = nn.Sequential(
            nn.Linear(world_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Sigmoid(),  # harm in [0, 1]
        )

        # ARC-030 / MECH-112: benefit_eval head — Go channel (symmetric to harm_eval NoGo).
        # Evaluates resource/goal proximity from z_world states.
        # Instantiated unconditionally (adds ~4K params); only receives gradients
        # in experiments that call benefit_eval() or enable benefit_eval_enabled.
        # Biological basis: D1 (Go) pathway evaluates same proposals as D2 (NoGo).
        self.benefit_eval_head = nn.Sequential(
            nn.Linear(world_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Sigmoid(),  # benefit in [0, 1]
        )

        # MECH-111: novelty EMA — tracks recent E1 prediction error per trajectory.
        # Stored as a scalar EMA; decays toward zero between updates.
        self._novelty_ema: float = 0.0
        self._novelty_ema_alpha: float = 0.1

        # SD-010: harm_eval head operating on z_harm (dedicated nociceptive stream).
        # z_harm comes from HarmEncoder(harm_obs), NOT from z_world.
        # This head is always instantiated (adds ~4K params); it only receives
        # gradients in SD-010 experiments that explicitly call harm_eval_z_harm().
        # Fallback z_harm_dim = world_dim if config does not carry z_harm_dim
        # (all existing E3Config instances, which lack the field).
        _z_harm_dim = getattr(self.config, "z_harm_dim", world_dim)
        self.harm_eval_z_harm_head = nn.Sequential(
            nn.Linear(_z_harm_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1),
            # Sigmoid removed (2026-03-20): this is a linear regression head for hazard
            # proximity, trained with normalized labels in [0,1] (harm_obs[12]).
            # MSE loss with [0,1] labels constrains outputs naturally without Sigmoid.
            # Sigmoid was causing saturation when raw hazard_field_at_agent labels (>1)
            # were used — root cause of SD-010 collapse in EXQ-056 original.
        )

        # Dynamic precision state (ARC-016)
        # Maintained as EMA of prediction error MSE across committed trajectories.
        self._running_variance: float = self.config.precision_init
        self._ema_alpha: float = self.config.precision_ema_alpha

        # Q-007: volatility estimate = var(rv) over a sliding window.
        # Raw rv tracks moment-to-moment prediction error (fast EMA, half-life
        # ~14 steps).  In both stable and volatile environments rv converges to
        # near-zero once E2 adapts, losing the between-condition signal.
        # var(rv) captures how much rv *fluctuates*: stable env -> rv flat ->
        # var(rv) ~ 0; volatile env -> rv spikes on hazard moves -> var(rv) high.
        # This is the LC-NE tonic firing analog (Yu & Dayan 2005).
        from collections import deque
        self._rv_history: deque = deque(maxlen=100)
        self._volatility_estimate: float = 0.0

        # Commitment state
        self._committed_trajectory: Optional[Trajectory] = None
        # Closure-plane commit-ENTRY primitive (rung-6 amend; commitment_closure:GAP-4;
        # failure_autopsy_V3-EXQ-460k/460l). An F-INDEPENDENT sticky latch SET by
        # REEAgent.select_action on a goal-active rule-directed commitment (Option A:
        # goal_state.is_active() AND a trajectory selected toward it AND lateral_pfc
        # rule_state norm above a floor) when use_closure_commit_entry is on, and CLEARED
        # on the SD-034 closure fire / de-commit refractory install / episode reset. Unlike
        # _committed_trajectory (set ONLY under the F-driven `if committed:` path and torn
        # down every tick by post_action_update), this latch persists across ticks until a
        # principled closure teardown, so agent.py:6365 _closure_commit_active (now the
        # UNION of the two) can arm the closure-exclusive eval without a sustained F-commit.
        # init False -> with use_closure_commit_entry off, _closure_commit_active reduces to
        # the legacy `_committed_trajectory is not None` -> bit-identical.
        self._closure_committed_active: bool = False
        # Closure-plane commit-ENTRY TRAJECTORY primitive (rung-6 amend extension;
        # use_closure_commit_entry_trajectory). The PARALLEL sticky trajectory the bool
        # latch above lacks: SET by REEAgent.select_action (to the goal/rule-directed
        # result.selected_trajectory) on a closure-coupled commit when the trajectory
        # flag is on, and CLEARED at the same de-commit / closure-fire / reset sites as
        # the bool latch. Unlike _committed_trajectory it is NOT torn down by
        # post_action_update -- it persists across ticks so the between-E3-tick stepping
        # path can advance a closure-formed committed PROGRAM (C-STEP) rather than
        # repeating _last_action. init None -> with the trajectory flag off every
        # consuming union reduces to the bool-latch behaviour -> bit-identical.
        self._closure_committed_trajectory: Optional[Trajectory] = None
        # ARC-016: store last selected trajectory for rv updates regardless of
        # commitment.  Without this, rv only updates when committed, creating a
        # deadlock: rv starts above commit_threshold -> agent never commits ->
        # rv never updates -> agent can never commit.
        self._last_selected_trajectory: Optional[Trajectory] = None
        self.last_scores: Optional[torch.Tensor] = None
        # ControlVector logging (rec-B): pre-bias per-candidate scores (value
        # axis / V_outcome), written each select() call. None until first call.
        self.last_raw_scores: Optional[torch.Tensor] = None
        # V3-EXQ-563c: score / bias scale diagnostics written each select() call.
        self.last_score_diagnostics: dict = {}
        # V3-EXQ-571: per-component score decomposition (default OFF, bit-identical)
        self.e3_score_decomp_enabled: bool = False
        self.last_score_decomp: dict = {}
        self._last_traj_components: dict = {}

        # ARC-030: benefit_eval warmup gate.
        # benefit_eval_head starts at random init — scoring with it before training
        # converges adds harmful noise to trajectory selection. Gate is lifted once
        # enough benefit samples have been seen (call record_benefit_sample() from
        # the experiment training loop when adding to the benefit buffer).
        self._benefit_samples_seen: int = 0
        self._BENEFIT_WARMUP_SAMPLES: int = 50

        # ARC-108 JOB-1 step-1: learned per-channel selection-weight vector w_chan.
        # register_buffer (NOT nn.Parameter) -- the three-factor plasticity rule is a
        # LOCAL update, not gradient descent, so w_chan is never touched by an
        # optimizer / autograd (it still rides device + state_dict). Init so
        # softplus(w_chan[c]) == 1.0 for every channel -> bit-identical OFF. The
        # eligibility trace (decayed |channel_bias_c[selected]| Hebbian co-activation)
        # and the slow value baseline V-hat_t live alongside it. All inert unless
        # config.use_learned_channel_gating is True. MECH-450 settling weights W_lat
        # are NOT instantiated here -- factor 2, OFF in this build.
        _lcg_n = len(_LCG_CHANNEL_NAMES)
        self.register_buffer(
            "w_chan", torch.full((_lcg_n,), _LCG_W_INIT, dtype=torch.float32)
        )
        self.register_buffer(
            "_lcg_elig_trace", torch.zeros(_lcg_n, dtype=torch.float32)
        )
        self._lcg_value_baseline: float = 0.0   # V-hat_t (slow EMA of realised R_t)
        self._lcg_pending: bool = False         # a waking eligibility trace awaits an outcome
        self._lcg_last_delta: float = 0.0       # last signed RPE delta_t (diagnostic)
        self._lcg_n_updates: int = 0            # count of w_chan updates (diagnostic)

        # MECH-451: the finer-channel learned-gating buffers. PARALLEL to the
        # ARC-108 w_chan / _lcg_elig_trace above so the ARC-108 path is byte-identical
        # (this is the ARC-106 reuse-the-mechanism pattern: same softplus-unity init,
        # same three-factor rule, same waking-only gate, new buffer over the finer
        # registry). w_chan_finer init so softplus(w_chan_finer[c]) == 1.0 for every
        # channel -> the finer decomposition reproduces the compressed blend EXACTLY
        # at init. V-hat_t is SHARED with the ARC-108 baseline (_lcg_value_baseline)
        # since the two gating modes are mutually exclusive in practice (A1 vs A2
        # arms). All inert unless config.use_finer_channel_gating is True.
        _fcg_n = len(_FCG_CHANNEL_NAMES)
        self.register_buffer(
            "w_chan_finer", torch.full((_fcg_n,), _LCG_W_INIT, dtype=torch.float32)
        )
        self.register_buffer(
            "_fcg_elig_trace", torch.zeros(_fcg_n, dtype=torch.float32)
        )
        self._fcg_pending: bool = False         # a waking finer-eligibility trace awaits an outcome
        self._fcg_last_delta: float = 0.0       # last signed RPE delta_t applied to w_chan_finer (diagnostic)
        self._fcg_n_updates: int = 0            # count of w_chan_finer updates (diagnostic)

        # ARC-108 JOB-1 step-2 / MECH-450: the learned lateral-inhibition matrix W_lat
        # over candidate first-action CLASSES (the SECOND factor of the learned-gating
        # 2x2, sharing the JOB-1 signed-RPE delta_t / V-hat_t / D1-D2 asym). register_buffer
        # (NOT nn.Parameter -- the three-factor plasticity is a LOCAL update, never touched
        # by an optimizer / autograd; rides device + state_dict). Init 0 -> the settling
        # step is a no-op -> bit-identical OFF and bit-identical at init. Per-action-class
        # (not per-candidate [K,K] -- the candidate set is variable-size with no stable
        # identity) keeps W_lat a stable learned object: the BG surround-inhibition between
        # competing motor programs (Mink 1996; MECH-449 grounds the same opponency). The
        # cross-tick decayed co-activation trace + the pending flag mirror the w_chan
        # eligibility plumbing. All inert unless config.use_learned_settling_step is True.
        _lcg_nc = int(getattr(self.config, "learned_settling_n_action_classes", 8))
        _lcg_nc = max(1, _lcg_nc)
        self.register_buffer(
            "W_lat", torch.zeros((_lcg_nc, _lcg_nc), dtype=torch.float32)
        )
        self.register_buffer(
            "_wlat_coact_trace", torch.zeros((_lcg_nc, _lcg_nc), dtype=torch.float32)
        )
        self._wlat_pending: bool = False        # a waking settling trace awaits an outcome
        self._wlat_last_delta: float = 0.0       # last signed RPE delta_t applied to W_lat (diag)
        self._wlat_n_updates: int = 0            # count of W_lat updates (diagnostic)
        self._wlat_last_settle_delta: float = 0.0  # L2 cross-round movement of accum (non-vacuity)

        # ARC-110 / MECH-452: per-tick segregated-loop bookkeeping written by
        # _segregated_loop_arbitrate and read by the eligibility-trace recording.
        # _loop_of_channel: channel_index -> loop name (for loop-local credit).
        # _loop_voted: loop name -> did this loop's within-loop winner match the
        #              committed action (so MECH-452 credits only voting loops).
        # All inert unless use_loop_segregation is on. Reset each arbitration.
        self._loop_of_channel: Dict[int, str] = {}
        self._loop_voted: Dict[str, bool] = {}

        # MECH-440: NoisyNet propagating selection-head weight noise. LAZY-built
        # on the first select() that needs it (action_dim is read from the
        # candidate first-action width, not plumbed through E3Config). None when
        # use_noisy_selection_head is False -> bit-identical OFF. Set here so the
        # attribute always exists. _last_explore_term carries the per-candidate
        # MECH-440/441 bias to the segregated-loop arbitration (ARC-110 path).
        self.noisy_selection_head: Optional[NoisySelectionHead] = None
        self._last_explore_term: Optional[torch.Tensor] = None

    def _candidate_action_features(
        self, candidates: List["Trajectory"]
    ) -> Optional[torch.Tensor]:
        """MECH-440: per-candidate first-action feature matrix [K, action_dim].

        Each candidate's first action (actions[:, 0, :]) flattened to a row. The
        input that makes the noisy head's per-candidate bias differentiated by
        construction (distinct candidates carry distinct actions) -- robust to
        the z_world monostrategy collapse that sank the 648/614e channels.
        Returns None if the candidate action shape is unreadable.
        """
        if not candidates:
            return None
        try:
            rows = [c.actions[:, 0, :].reshape(-1) for c in candidates]
            feats = torch.stack(rows, dim=0)
        except Exception:
            return None
        return feats

    def _ensure_noisy_head(self, action_dim: int, device, dtype) -> None:
        """MECH-440: lazy-build the noisy selection head once action_dim is known."""
        if not getattr(self.config, "use_noisy_selection_head", False):
            return
        if self.noisy_selection_head is not None:
            return
        cfg = NoisySelectionHeadConfig(
            action_dim=int(action_dim),
            sigma_init=float(getattr(self.config, "noisy_selection_sigma_init", 0.0)),
            weight=float(getattr(self.config, "noisy_selection_weight", 1.0)),
            anneal=bool(getattr(self.config, "noisy_selection_anneal", True)),
            anneal_floor=float(getattr(self.config, "noisy_selection_anneal_floor", 0.1)),
            anneal_ema_alpha=float(
                getattr(self.config, "noisy_selection_anneal_ema_alpha", 0.01)
            ),
        )
        head = NoisySelectionHead(cfg)
        self.noisy_selection_head = head.to(device=device)

    # ------------------------------------------------------------------ #
    # Dynamic precision (ARC-016)                                         #
    # ------------------------------------------------------------------ #

    @property
    def current_precision(self) -> float:
        """Current precision estimate (inverse of running variance)."""
        return 1.0 / (self._running_variance + 1e-6)

    @property
    def commit_threshold(self) -> float:
        """Variance-space commit threshold (ARC-016). Committed when variance < threshold."""
        return variance_commit_threshold(self.config.commitment_threshold)

    @property
    def volatility_estimate(self) -> float:
        """Q-007: var(rv) over sliding window -- LC-NE tonic volatility signal."""
        return self._volatility_estimate

    def update_running_variance(self, prediction_error: torch.Tensor) -> None:
        """Update EMA of prediction error variance (ARC-016 dynamic precision)."""
        error_var = prediction_error.pow(2).mean().item()
        self._running_variance = (
            (1 - self._ema_alpha) * self._running_variance
            + self._ema_alpha * error_var
        )
        # Q-007: track rv history and compute volatility estimate
        self._rv_history.append(self._running_variance)
        if len(self._rv_history) >= 10:
            vals = list(self._rv_history)
            mean = sum(vals) / len(vals)
            self._volatility_estimate = sum((v - mean) ** 2 for v in vals) / len(vals)
        else:
            self._volatility_estimate = 0.0

    def recalibrate_precision_to(self, target_precision: float, step: float = 0.1) -> Tuple[float, float]:
        """
        MECH-204 Option A: nudge ``_running_variance`` toward the variance
        implied by ``target_precision``, by a tunable step in [0, 1].

        target_precision is a precision scalar (inverse variance). Implementation
        converts it to a variance and applies a one-shot linear interpolation:

            new_variance = (1 - step) * current_variance + step * target_variance

        Returns (running_variance_before, running_variance_after) for diagnostics.

        Caller (SleepLoopManager WRITEBACK phase) is responsible for gating
        on the master flag and on whether REM was actually entered. This
        method does not check enablement, but it is a no-op when
        target_precision <= 0.0 (the "no target captured" sentinel from
        SerotoninModule.compute_recalibration_target()).
        """
        rv_before = float(self._running_variance)
        if target_precision <= 0.0:
            return rv_before, rv_before
        if step <= 0.0:
            return rv_before, rv_before
        step_clamped = min(1.0, float(step))
        target_variance = 1.0 / (float(target_precision) + 1e-6)
        self._running_variance = (
            (1.0 - step_clamped) * rv_before + step_clamped * target_variance
        )
        return rv_before, float(self._running_variance)

    def record_benefit_sample(self, n: int = 1) -> None:
        """Record that n benefit training samples have been added to the buffer.

        Call from the experiment training loop each time a positive benefit
        sample is added. Once _benefit_samples_seen >= _BENEFIT_WARMUP_SAMPLES,
        benefit scoring is activated in score_trajectory().
        """
        self._benefit_samples_seen += n

    # ------------------------------------------------------------------ #
    # harm_eval (SD-003 V3 pipeline)                                      #
    # ------------------------------------------------------------------ #

    def harm_eval(self, z_world: torch.Tensor) -> torch.Tensor:
        """
        Evaluate harm of a world-state (SD-003 V3 attribution pipeline).

        This is the correct locus for harm prediction — NOT E2.
        Used externally as:
            harm_actual = e3.harm_eval(z_world_actual)
            harm_cf     = e3.harm_eval(z_world_cf)
            causal_sig  = harm_actual - harm_cf

        Args:
            z_world: [batch, world_dim]

        Returns:
            harm estimate [batch, 1] in [0, 1]
        """
        return self.harm_eval_head(z_world)

    def benefit_eval(self, z_world: torch.Tensor) -> torch.Tensor:
        """
        Evaluate benefit/goal proximity from a world-state (ARC-030 Go channel).

        Symmetric to harm_eval() — both evaluate the same z_world trajectory
        proposals. This is the D1 (Go) pathway counterpart to harm_eval()'s
        D2 (NoGo) role (Bariselli 2018). Commit threshold is the balance point.

        Args:
            z_world: [batch, world_dim]

        Returns:
            benefit estimate [batch, 1] in [0, 1]
        """
        return self.benefit_eval_head(z_world)

    def update_novelty_ema(self, e1_prediction_error_mse: float) -> None:
        """
        MECH-111: Update novelty EMA from E1 prediction error.

        Higher E1 error = more novel state. Used in score_trajectory()
        when novelty_bonus_weight > 0 to reward unexplored regions.
        """
        self._novelty_ema = (
            (1 - self._novelty_ema_alpha) * self._novelty_ema
            + self._novelty_ema_alpha * e1_prediction_error_mse
        )

    def harm_eval_lateral(
        self,
        z_harm: torch.Tensor,
        lateral_head: nn.Module,
    ) -> torch.Tensor:
        """
        Evaluate harm using the lateral encoder head's z_harm embedding (MECH-099).

        The lateral head produces a harm-salient embedding directly from
        hazard + contamination channels, bypassing E2_world's identity shortcut.
        This method applies a small projection from z_harm to a scalar harm estimate.

        Args:
            z_harm: [batch, harm_dim] — output of SplitEncoder.lateral_head
            lateral_head: nn.Module that maps z_harm [harm_dim] → harm scalar [1]

        Returns:
            harm estimate [batch, 1]
        """
        return lateral_head(z_harm)

    def harm_eval_z_harm(self, z_harm: torch.Tensor) -> torch.Tensor:
        """
        SD-010: Evaluate harm using the dedicated nociceptive stream latent.

        z_harm is the output of HarmEncoder(harm_obs), NOT of the z_world encoder.
        Because HarmEncoder is instantiated outside LatentStack.encode(), z_harm
        is never subject to reafference correction — which resolves the EXQ-027b
        over-correction paradox (ReafferencePredictor was subtracting hazard signal
        when it was fused into z_world).

        Used in SD-003 attribution with SD-010:
            z_harm_actual = harm_enc(harm_obs_actual)
            z_harm_cf     = harm_enc(harm_bridge(E2.world_forward(z_world, a_cf)))
            causal_sig    = harm_eval_z_harm(z_harm_actual) - harm_eval_z_harm(z_harm_cf)

        Args:
            z_harm: [batch, z_harm_dim] — output of HarmEncoder

        Returns:
            harm estimate [batch, 1] in [0, 1]
        """
        return self.harm_eval_z_harm_head(z_harm)

    # ------------------------------------------------------------------ #
    # Trajectory scoring                                                   #
    # ------------------------------------------------------------------ #

    def _get_world_states(self, trajectory: Trajectory) -> torch.Tensor:
        """Extract z_world trajectory. Falls back to z_self if no world_states."""
        if trajectory.world_states is not None:
            ws = trajectory.get_world_state_sequence()
            if ws is not None:
                return ws
        # Fallback: use z_self states (for V3-EXQ-001 before full SD-005 wiring)
        return trajectory.get_state_sequence()

    def compute_reality_cost(self, trajectory: Trajectory) -> torch.Tensor:
        """Reality constraint F(ζ) — proxy: smoothness + viability of z_world final state."""
        world_seq = self._get_world_states(trajectory)  # [batch, horizon, world_dim]

        if world_seq.shape[1] > 1:
            transitions = world_seq[:, 1:, :] - world_seq[:, :-1, :]
            coherence_cost = transitions.pow(2).sum(dim=-1).mean(dim=-1)
        else:
            coherence_cost = torch.zeros(world_seq.shape[0], device=world_seq.device)

        final_world = world_seq[:, -1, :]  # [batch, world_dim]
        viability = self.reality_scorer(final_world).squeeze(-1)
        return coherence_cost - viability

    def compute_ethical_cost(self, trajectory: Trajectory) -> torch.Tensor:
        """
        Ethical cost M(ζ) via harm_eval over z_world trajectory.

        V3 change: uses harm_eval_head on z_world states rather than
        E2.harm_predictions (which belonged to E2 in V2, incorrectly).
        """
        world_seq = self._get_world_states(trajectory)  # [batch, horizon+1, world_dim]
        batch, horizon_p1, _ = world_seq.shape

        # Evaluate harm at each step
        flat = world_seq.reshape(batch * horizon_p1, -1)
        harm_flat = self.harm_eval_head(flat)              # [batch*horizon, 1]
        harm = harm_flat.reshape(batch, horizon_p1)        # [batch, horizon+1]
        harm_cost = harm.sum(dim=-1)                       # [batch]

        # Additional scoring from final z_world state
        final_world = world_seq[:, -1, :]
        ethical_score = self.ethical_scorer(final_world).squeeze(-1)

        return harm_cost - ethical_score

    def compute_residue_cost(self, trajectory: Trajectory) -> torch.Tensor:
        """Residue field cost Φ_R(ζ) — evaluated over z_world (SD-005)."""
        if self.residue_field is None:
            return torch.zeros(
                trajectory.states[0].shape[0],
                device=trajectory.states[0].device,
            )
        world_seq = self._get_world_states(trajectory)   # [batch, horizon+1, world_dim]
        return self.residue_field.evaluate_trajectory(world_seq)

    def compute_benefit_score(self, trajectory: Trajectory) -> torch.Tensor:
        """
        Go channel benefit score B(ζ) — ARC-030 / MECH-112.

        Evaluates resource/goal proximity across z_world trajectory.
        Returns summed benefit signal (higher = more beneficial trajectory).
        Subtracted from total score (lower J = better, so benefit reduces J).
        """
        world_seq = self._get_world_states(trajectory)  # [batch, horizon+1, world_dim]
        batch, horizon_p1, _ = world_seq.shape
        flat = world_seq.reshape(batch * horizon_p1, -1)
        benefit_flat = self.benefit_eval_head(flat)         # [batch*horizon, 1]
        benefit = benefit_flat.reshape(batch, horizon_p1)   # [batch, horizon+1]
        return benefit.sum(dim=-1)                          # [batch]

    def compute_goal_score(
        self, trajectory: Trajectory, goal_state: GoalState
    ) -> torch.Tensor:
        """
        Goal proximity score across trajectory (wanting signal, MECH-112/117).
        Shape: [batch]. Higher = trajectory ends closer to z_goal.
        """
        world_seq = self._get_world_states(trajectory)
        batch, horizon_p1, _ = world_seq.shape
        flat = world_seq.reshape(batch * horizon_p1, -1)
        prox_flat = goal_state.goal_proximity(flat)
        prox = prox_flat.reshape(batch, horizon_p1)
        return prox.sum(dim=-1)

    def compute_harm_stream_cost(
        self,
        trajectory: Trajectory,
        harm_bridge: "nn.Module",
    ) -> torch.Tensor:
        """
        SD-010: Ethical cost M(ζ) via harm stream (z_harm) rather than z_world.

        Applies HarmBridge to each z_world state in the trajectory to get z_harm
        approximations, then evaluates via harm_eval_z_harm_head. This is the
        SD-010 replacement for compute_ethical_cost() when the nociceptive stream
        is available.

        Used in select() when harm_bridge is provided.

        Args:
            trajectory:  candidate trajectory with world_states
            harm_bridge: HarmBridge instance (z_world -> z_harm)

        Returns:
            harm_cost: [batch] — summed harm cost over trajectory horizon
        """
        world_seq = self._get_world_states(trajectory)  # [batch, horizon+1, world_dim]
        batch, horizon_p1, world_dim = world_seq.shape

        flat = world_seq.reshape(batch * horizon_p1, world_dim)
        z_harm_flat = harm_bridge(flat)                              # [batch*horizon, z_harm_dim]
        harm_flat = self.harm_eval_z_harm_head(z_harm_flat)         # [batch*horizon, 1]
        harm = harm_flat.reshape(batch, horizon_p1)                  # [batch, horizon+1]
        return harm.sum(dim=-1)                                      # [batch]

    def compute_harm_forward_cost(
        self,
        trajectory: Trajectory,
        harm_forward_model: "nn.Module",
        z_harm_s_current: torch.Tensor,
    ) -> torch.Tensor:
        """
        SD-011/ARC-033: Ethical cost M(zeta) via ResidualHarmForward rollout.

        Replaces compute_harm_stream_cost (HarmBridge-based, deprecated).
        Rolls out z_harm_s step-by-step through the trajectory actions using
        the residual harm forward model, then evaluates each predicted state
        via harm_eval_z_harm_head.

        Args:
            trajectory:          candidate trajectory with actions
            harm_forward_model:  ResidualHarmForward instance
            z_harm_s_current:    [batch, z_harm_dim] current sensory-discriminative
                                 harm latent

        Returns:
            harm_cost: [batch] -- summed harm cost over trajectory horizon
        """
        actions = trajectory.actions  # [batch, horizon, action_dim]
        batch, horizon, _ = actions.shape

        z_harm_step = z_harm_s_current  # [batch, z_harm_dim]
        harm_total = torch.zeros(batch, device=z_harm_step.device)

        for t in range(horizon):
            a_t = actions[:, t, :]  # [batch, action_dim]
            z_harm_step = harm_forward_model(z_harm_step, a_t)  # [batch, z_harm_dim]
            harm_t = self.harm_eval_z_harm_head(z_harm_step)    # [batch, 1]
            harm_total = harm_total + harm_t.squeeze(-1)

        return harm_total

    def _pe_confidence_penalty(
        self, e2_forward_pe: torch.Tensor, device, dtype
    ) -> torch.Tensor:
        """DR-12 (self_model_v4:SELF-4): monotone confidence-deficit penalty from the
        E2 forward-PE magnitude attributed to a trajectory's region.

        Returns a non-negative scalar penalty (in score/cost units) that is monotone
        non-decreasing in the E2 forward-PE magnitude. Added to score_trajectory's
        cost so a poorly-modelled (high-PE) trajectory is discounted. Modes:
          "linear"     : penalty = pe_magnitude
          "saturating" : penalty = 1 - exp(-pe_magnitude / pe_confidence_scale) in [0,1)
        """
        pe = e2_forward_pe.to(device=device, dtype=dtype).reshape(())
        pe_mag = pe.clamp(min=0.0)  # PE magnitude is non-negative; guard
        mode = getattr(self.config, "pe_confidence_mode", "linear")
        if mode == "saturating":
            scale = float(getattr(self.config, "pe_confidence_scale", 1.0))
            scale = scale if scale > 1e-12 else 1.0
            return 1.0 - torch.exp(-pe_mag / scale)
        # default: linear (penalty == PE magnitude)
        return pe_mag

    def score_trajectory(
        self,
        trajectory: Trajectory,
        goal_state: Optional[GoalState] = None,
        harm_bridge: Optional["nn.Module"] = None,
        terrain_weight: Optional[torch.Tensor] = None,
        harm_forward_model: Optional["nn.Module"] = None,
        z_harm_s_current: Optional[torch.Tensor] = None,
        z_harm_a: Optional[torch.Tensor] = None,
        e2_forward_pe: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Total score J(ζ) = F(ζ) + λ·M(ζ) + ρ·Φ_R(ζ) - β·B(ζ).
        Lower is better.

        - F(ζ): reality cost (smoothness + viability)
        - M(ζ): ethical cost via harm_eval (NoGo channel, D2 pathway)
        - Φ_R(ζ): residue field cost
        - B(ζ): benefit score (Go channel, D1 pathway) — subtracted when enabled

        Per-candidate novelty / curiosity / liking / dACC biases enter via the
        score_bias kwarg of select() (composed in REEAgent.select_action), NOT
        inside score_trajectory. The legacy MECH-111 broadcast novelty branch
        was deleted 2026-05-25; see
        evidence/planning/v3_exq_571_root_cause_2026-05-25.md.

        M(ζ) evaluation priority:
        1. harm_forward_model + z_harm_s_current (SD-011/ARC-033, preferred)
        2. harm_bridge (SD-010, deprecated but backward compat)
        3. harm_eval_head on z_world (default)

        SD-011: z_harm_a amplifies lambda_ethical when affective_harm_scale > 0.
        SD-016 (MECH-152): optional terrain_weight scales M/B after evaluation.
        """
        f = self.compute_reality_cost(trajectory)
        if harm_forward_model is not None and z_harm_s_current is not None:
            m = self.compute_harm_forward_cost(
                trajectory, harm_forward_model, z_harm_s_current
            )
        elif harm_bridge is not None:
            m = self.compute_harm_stream_cost(trajectory, harm_bridge)
        else:
            m = self.compute_ethical_cost(trajectory)
        phi = self.compute_residue_cost(trajectory)

        # V3-EXQ-571: component trackers (used only when e3_score_decomp_enabled).
        # _dc_novelty_w is retained at 0.0 for backward-compat with the
        # last_score_decomp schema; the MECH-111 broadcast branch that
        # populated it was deleted 2026-05-25 (dead-by-construction --
        # uniform scalar shift is argmin-invariant). MECH-314a is the
        # per-candidate replacement, composed into score_bias at the
        # select() call site rather than inside score_trajectory.
        # See evidence/planning/v3_exq_571_root_cause_2026-05-25.md.
        _dc_benefit_w = 0.0
        _dc_novelty_w = 0.0
        _dc_goal_w = 0.0

        # SD-011: z_harm_a amplification of ethical cost.
        # When accumulated threat is high, harm costs weigh more.
        lambda_eff = self.config.lambda_ethical
        if z_harm_a is not None and self.config.affective_harm_scale > 0.0:
            z_harm_a_norm = z_harm_a.norm(dim=-1).mean().item()
            lambda_eff = lambda_eff * (1.0 + self.config.affective_harm_scale * z_harm_a_norm)

        # SD-016 (MECH-152): scale harm cost by w_harm from terrain_weight
        if terrain_weight is not None:
            w_harm = terrain_weight[:, 0]  # [batch]
            m = m * w_harm

        score = f + lambda_eff * m + self.config.rho_residue * phi

        # ARC-030 / MECH-112: Go channel — subtract benefit from cost.
        # Gated until _benefit_samples_seen >= _BENEFIT_WARMUP_SAMPLES to prevent
        # random-init noise from corrupting trajectory selection early in training.
        if (self.config.benefit_eval_enabled
                and self.config.benefit_weight > 0.0
                and self._benefit_samples_seen >= self._BENEFIT_WARMUP_SAMPLES):
            b = self.compute_benefit_score(trajectory)
            # SD-016 (MECH-152): scale benefit by w_goal from terrain_weight
            if terrain_weight is not None:
                w_goal = terrain_weight[:, 1]  # [batch]
                b = b * w_goal
            score = score - self.config.benefit_weight * b
            if self.e3_score_decomp_enabled:
                _dc_benefit_w = float((self.config.benefit_weight * b).detach().mean().item())

        # MECH-112 / MECH-117: wanting signal via z_goal distance
        if (goal_state is not None
                and goal_state.is_active()
                and self.config.goal_weight > 0.0):
            g = self.compute_goal_score(trajectory, goal_state)
            # SD-016 (MECH-152): scale goal proximity by w_goal from terrain_weight
            if terrain_weight is not None:
                w_goal = terrain_weight[:, 1]  # [batch]
                g = g * w_goal
            score = score - self.config.goal_weight * g
            if self.e3_score_decomp_enabled:
                _dc_goal_w = float((self.config.goal_weight * g).detach().mean().item())

        # DR-12 (self_model_v4:SELF-4, FIRST V4 substrate build): E2 forward-PE
        # confidence down-weight. score is a COST (lower is better), so a positive
        # penalty proportional to a monotone function of the E2 forward-PE magnitude
        # in this trajectory's region DISCOUNTS the trajectory's viability/confidence.
        # NEW lever on EXISTING machinery (sibling to _running_variance / _novelty_ema
        # PE consumption). Bit-identical OFF: skipped unless use_pe_confidence_weighting
        # AND a per-trajectory e2_forward_pe is supplied (per-candidate via select()).
        # generation:v4, off the V3 critical path; promotes nothing in V3.
        if (getattr(self.config, "use_pe_confidence_weighting", False)
                and e2_forward_pe is not None
                and self.config.pe_confidence_weight != 0.0):
            pen = self._pe_confidence_penalty(
                e2_forward_pe, device=score.device, dtype=score.dtype
            )
            score = score + self.config.pe_confidence_weight * pen

        if self.e3_score_decomp_enabled:
            self._last_traj_components = {
                "f": float(f.detach().mean().item()),
                "harm_weighted": float((lambda_eff * m).detach().mean().item()),
                "residue_weighted": float((self.config.rho_residue * phi).detach().mean().item()),
                "benefit_weighted": _dc_benefit_w,
                "novelty_weighted": _dc_novelty_w,
                "goal_weighted": _dc_goal_w,
                "lambda_eff": float(lambda_eff),
            }
        return score

    # ------------------------------------------------------------------ #
    # Selection                                                            #
    # ------------------------------------------------------------------ #

    def _f_eligibility_envelope(
        self,
        raw_scores: torch.Tensor,
    ) -> torch.Tensor:
        """MECH-448 (ARC-107): graded, rank-preserving F->eligibility envelope.

        The pallidal-permission reading of the conversion ceiling: F decides who
        is ELIGIBLE to compete, not who wins. F (``raw_scores``, a per-candidate
        cost, lower-is-better) is renormalised against the COMPETING FIELD by a
        divisive-normalisation analog and thresholded by an ABSOLUTE share floor:

            merit[i] = clamp(raw_scores.max() - raw_scores[i], min=0)  # best=highest
            pooled   = f_eligibility_dn_sigma + merit.sum()
            elig[i]  = merit[i] / pooled                               # share of field
            eligible = { i : elig[i] >= f_eligibility_envelope_floor }

        The absolute-share floor (NOT a fraction-of-max, which cancels the pooled
        term and degenerates to the margin shortlist) makes the envelope graded,
        conflict-scaled, and env-general: a decisive F-winner commands most of the
        merit share so others fall below the floor (narrow envelope); a near-tie
        spreads the share (wide envelope). ``elig`` is monotone in ``merit`` ->
        monotone in -F, so the eligible set is an F-RANK PREFIX (rank-preserving).

        CHANNEL-ADAPTIVE floor (use_f_eligibility_adaptive_floor, 2026-06-21): the
        FIXED absolute floor (default 0.30) was tuned to the GAP-A foraging bank
        (V3-EXQ-689d); each downstream channel has a different F-merit distribution
        so the same fixed floor mis-fires (654h: every share < 0.30 -> all-admit
        no-op; 485i: needed a bespoke per-seed floor to engage). With the adaptive
        flag set, the floor is computed RELATIVE to the field's own mean share --
        ``floor = f_eligibility_adaptive_mean_factor * elig.mean()`` -- so a
        candidate is eligible iff it commands at least ``mean_factor`` of the field
        AVERAGE share rather than an absolute constant. Mean-relative is
        scale-invariant (auto-calibrates per channel, no hand-tuning) AND keeps the
        conflict-grade: a decisive winner pulls the mean up so others fall below
        (narrow), a near-tie sits near the mean (wide). It is still a threshold on
        ``elig`` (monotone in merit), so the eligible set stays an F-RANK PREFIX
        (rank-preserving). For mean_factor >= 1.0 on any NON-uniform field at least
        one candidate is below the mean share, so the envelope EXCLUDES
        (excluded_count > 0) by construction -- the 654h all-admit no-op cannot
        recur. Default False -> reads the fixed floor -> bit-identical.

        Returns a 1-D LongTensor of eligible candidate indices. Guaranteed
        non-empty: when F cannot discriminate (range ~ 0) or the floor admits no
        candidate (a genuine N-way tie where no candidate clears the share floor),
        the envelope falls back to ALL candidates -- the correct "low conflict ->
        wide envelope" behaviour (and reported as excluded_count == 0, the
        non-degeneracy signal the falsifier checks against a divergent pool).
        """
        n = int(raw_scores.shape[0])
        all_idx = torch.arange(n, device=raw_scores.device)
        if n < 2:
            return all_idx
        merit = (raw_scores.max() - raw_scores).clamp(min=0.0)
        merit_sum = float(merit.sum().item())
        if merit_sum <= 1e-8:
            # Flat F -- no discrimination -> wide envelope (all eligible).
            return all_idx
        sigma = float(getattr(self.config, "f_eligibility_dn_sigma", 0.0))
        pooled = sigma + merit_sum
        elig = merit / (pooled + 1e-8)
        if bool(getattr(self.config, "use_f_eligibility_adaptive_floor", False)):
            # CHANNEL-ADAPTIVE: floor relative to the field's own MEAN share, so
            # the threshold auto-calibrates to each channel's F-merit distribution
            # (no per-channel hand-tuned absolute floor) while staying a threshold
            # on ``elig`` -> rank-preserving. mean_factor >= 1.0 keeps "above the
            # average share" candidates and excludes the below-average ones, so a
            # non-uniform field always excludes (excluded_count > 0).
            mean_factor = float(
                getattr(self.config, "f_eligibility_adaptive_mean_factor", 1.0)
            )
            floor = mean_factor * float(elig.mean().item())
        else:
            floor = float(getattr(self.config, "f_eligibility_envelope_floor", 0.30))
        eligible_idx = torch.nonzero(elig >= floor, as_tuple=False).flatten()
        if int(eligible_idx.numel()) == 0:
            # Floor admits nobody (e.g. an exact N-way tie whose per-candidate
            # share is below the floor) -> genuine low-conflict case -> wide.
            return all_idx
        return eligible_idx

    def _go_nogo_eligibility_gate(
        self,
        eligible_idx: torch.Tensor,
        raw_scores: torch.Tensor,
        signals: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        """MECH-449 (ARC-107): bounded Go/No-Go eligibility constitution.

        The core opponency leg of the basal-ganglia selector constitution. Acts
        on the MECH-448 eligible set AFTER the envelope/shortlist builds it and
        BEFORE the within-eligible ``_modulatory_accum`` arbitration -- governing
        WHICH candidates may compete for the pallidal-like permission-to-commit
        gate, so lawful channel-specific ACCESS (not scalar F-dominance) decides.

        Generalises MECH-260 (dACC anti-recency No-Go) from a drowned score-bias
        into an eligibility-access gate (reuse-before-duplicate, ARC-106 G2: the
        ``perseveration`` axis CONSUMES MECH-260's per-candidate suppression
        vector). Two opponent pressures over the candidate set
        (``raw_scores`` = per-candidate F cost, lower = better; ``eligible_idx``
        = the F-graded eligible set):

          No-Go (suppress): drop a candidate from the eligible set when ANY
            bounded axis crosses its floor -- ``safety`` (undesirability >=
            gng_safety_floor), ``staleness`` (>= gng_staleness_floor),
            ``perseveration`` (recency-share >= gng_perseveration_floor;
            MECH-260 reuse), ``viability`` (< gng_viability_floor = low). These
            act on an axis ORTHOGONAL to F-rank: rank-preserving demotion is
            order-preserving over F and CANNOT exclude an F-eligible-but-
            undesirable candidate -- only an active No-Go can (V3-EXQ-689f).

          Go (promote): add back into the eligible set, bounded by
            gng_go_max_promote, any candidate F demoted OUT of the envelope whose
            ``go`` evidence >= gng_go_threshold (and that is not itself No-Go'd).

        SAFETY: a No-Go'd candidate is removed from the eligible set, so the
        within-eligible argmin can never select it regardless of its modulatory
        pull (the orthogonal-to-F guarantee). FAIL-OPEN: No-Go never drops the
        eligible set below ``gng_protect_min_eligible`` survivors UNLESS those
        survivors are SAFETY-No-Go'd (safety is never overridden by the fail-open
        -- a clearly-harmful candidate stays suppressed even if it is the last
        one). This guards the No-Go-over-pressure -> catatonia/avolition failure
        pole from deadlocking the gate.

        ``signals`` is an optional dict of per-candidate [K] tensors keyed
        ``safety`` / ``staleness`` / ``perseveration`` / ``viability`` / ``go``;
        a missing axis is inert. Returns the (possibly modified) eligible-index
        LongTensor; never returns empty.
        """
        n = int(raw_scores.shape[0])
        all_idx = torch.arange(n, device=raw_scores.device)
        # Eligible-set membership mask over all K candidates.
        elig_mask = torch.zeros(n, dtype=torch.bool, device=raw_scores.device)
        elig_mask[eligible_idx] = True

        def _axis(name: str) -> Optional[torch.Tensor]:
            if not signals:
                return None
            v = signals.get(name, None)
            if v is None:
                return None
            t = v if isinstance(v, torch.Tensor) else torch.as_tensor(
                v, dtype=raw_scores.dtype, device=raw_scores.device
            )
            return t.detach().to(device=raw_scores.device).reshape(-1)

        safety = _axis("safety")
        staleness = _axis("staleness")
        perseveration = _axis("perseveration")
        viability = _axis("viability")
        go_evidence = _axis("go")

        # SAFETY No-Go is the absolute, fail-open-immune suppression.
        safety_nogo = torch.zeros(n, dtype=torch.bool, device=raw_scores.device)
        if safety is not None and safety.numel() == n:
            safety_nogo = safety >= float(self.config.gng_safety_floor)
        # Soft No-Go axes (staleness / perseveration / low-viability) -- subject
        # to the fail-open protect-min guard.
        soft_nogo = torch.zeros(n, dtype=torch.bool, device=raw_scores.device)
        if staleness is not None and staleness.numel() == n:
            soft_nogo = soft_nogo | (staleness >= float(self.config.gng_staleness_floor))
        if perseveration is not None and perseveration.numel() == n:
            soft_nogo = soft_nogo | (
                perseveration >= float(self.config.gng_perseveration_floor)
            )
        if viability is not None and viability.numel() == n:
            soft_nogo = soft_nogo | (viability < float(self.config.gng_viability_floor))

        # Apply No-Go to the eligible set: safety always; soft subject to fail-open.
        n_safety_nogo = int((safety_nogo & elig_mask).sum().item())
        elig_mask = elig_mask & (~safety_nogo)
        # Candidates the soft axes WOULD suppress, restricted to those still eligible.
        soft_drop_candidates = soft_nogo & elig_mask
        n_soft_requested = int(soft_drop_candidates.sum().item())
        protect_min = int(self.config.gng_protect_min_eligible)
        n_soft_applied = 0
        if n_soft_requested > 0:
            # Fail-open: keep at least protect_min eligible survivors. Drop the
            # soft-No-Go'd candidates in WORST-F-first order (highest cost first)
            # so the protected survivors are the strongest-F of the soft set.
            applied_mask = soft_drop_candidates.clone()
            n_remaining_after = int((elig_mask & (~applied_mask)).sum().item())
            if n_remaining_after < protect_min:
                # Re-admit the strongest-F (lowest raw cost) soft-No-Go'd
                # candidates until protect_min survivors remain.
                drop_idx = torch.nonzero(applied_mask, as_tuple=False).flatten()
                # order soft-drops by descending cost (worst first to actually drop)
                order = torch.argsort(raw_scores[drop_idx], descending=True)
                ordered_drop = drop_idx[order]
                n_can_drop = max(
                    0, int(elig_mask.sum().item()) - protect_min
                )
                keep_dropping = ordered_drop[:n_can_drop]
                applied_mask = torch.zeros_like(applied_mask)
                if keep_dropping.numel() > 0:
                    applied_mask[keep_dropping] = True
            elig_mask = elig_mask & (~applied_mask)
            n_soft_applied = int(applied_mask.sum().item())

        # Bounded Go promotion: re-admit demoted candidates with go-evidence.
        n_go = 0
        if go_evidence is not None and go_evidence.numel() == n:
            go_max = int(self.config.gng_go_max_promote)
            go_thr = float(self.config.gng_go_threshold)
            # Candidates not currently eligible, not No-Go'd, clearing the Go bar.
            blocked = safety_nogo | soft_nogo
            go_candidates = (~elig_mask) & (~blocked) & (go_evidence >= go_thr)
            go_idx = torch.nonzero(go_candidates, as_tuple=False).flatten()
            if go_idx.numel() > 0 and go_max > 0:
                # Promote the highest-go-evidence candidates first, bounded.
                order = torch.argsort(go_evidence[go_idx], descending=True)
                promote = go_idx[order][:go_max]
                elig_mask[promote] = True
                n_go = int(promote.numel())

        new_eligible = torch.nonzero(elig_mask, as_tuple=False).flatten()
        if int(new_eligible.numel()) == 0:
            # Last-resort: every eligible candidate was No-Go'd. Per SAFETY, do
            # NOT re-admit a safety-No-Go'd candidate; fall back to the
            # strongest-F candidate that is not safety-No-Go'd, else (all unsafe)
            # the strongest-F overall (the gate cannot manufacture a safe option
            # that does not exist; this is the avolition pole, flagged below).
            safe_pool = torch.nonzero(~safety_nogo, as_tuple=False).flatten()
            pool = safe_pool if safe_pool.numel() > 0 else all_idx
            best = int(pool[torch.argmin(raw_scores[pool]).item()].item())
            new_eligible = torch.tensor(
                [best], dtype=torch.long, device=raw_scores.device
            )
        self.last_score_diagnostics["go_nogo_constitution_active"] = True
        self.last_score_diagnostics["go_nogo_n_safety_nogo"] = n_safety_nogo
        self.last_score_diagnostics["go_nogo_n_soft_requested"] = n_soft_requested
        self.last_score_diagnostics["go_nogo_n_soft_applied"] = n_soft_applied
        self.last_score_diagnostics["go_nogo_n_go_promoted"] = n_go
        self.last_score_diagnostics["go_nogo_envelope_size"] = int(
            new_eligible.numel()
        )
        return new_eligible

    def _gap_scaled_commit_pick(
        self,
        cost: torch.Tensor,
        gap_norm: float,
        base_temperature: float,
    ) -> int:
        """Factor B (MECH-439): gap-scaled entropy-regularized committed pick.

        Softens a HARD argmin over ``cost`` (a per-eligible cost vector,
        lower-is-better) into a multinomial sample whose temperature is graded
        by the normalized top-F gap. T_eff = base_temperature +
        gap_scaled_commit_entropy_alpha * (1 - gap_norm): near-ties
        (gap_norm ~ 0) commit HOTTER (softer argmax); a decisive gap (~1)
        commits COLD (T_eff -> base, recovering the argmin in the limit). The
        (1 - gap_norm) scaling is load-bearing -- a flat (gap-blind) commit-T
        reduces to the 569g temperature control that under-lifted. Returns the
        index into ``cost`` of the sampled element.
        """
        alpha = float(
            getattr(self.config, "gap_scaled_commit_entropy_alpha", 1.0)
        )
        t_eff = base_temperature + alpha * (1.0 - float(gap_norm))
        t_eff = max(t_eff, 1e-6)
        self.last_score_diagnostics["gap_scaled_commit_active"] = True
        self.last_score_diagnostics["gap_scaled_commit_gap_norm"] = float(gap_norm)
        self.last_score_diagnostics["gap_scaled_commit_temperature_eff"] = float(
            t_eff
        )
        probs = F.softmax(-cost.detach() / t_eff, dim=0)
        return int(torch.multinomial(probs, 1).item())

    def _lateral_settle(
        self,
        mod_eligible: torch.Tensor,
        candidates: List[Trajectory],
        eligible_idx: torch.Tensor,
        record_trace: bool = True,
    ) -> torch.Tensor:
        """MECH-450 (ARC-108 JOB-1 factor 2): bounded recurrent lateral-inhibition
        settling over the F-bounded eligible set, parametrised by the LEARNED
        per-action-class inhibition matrix ``W_lat``.

        ``mod_eligible`` [n_elig] is the within-eligible modulatory accumulator
        (``_modulatory_accum[eligible_idx]``, COST units, lower = better). The
        settling runs a few rounds of mutual inhibition so the committed winner
        emerges from a recurrent SETTLING competition rather than a one-shot global
        argmin (divergence B1), and so the additive blend becomes a competitive
        winner-take-most (divergence B3-blend):

            accum = mod_eligible
            for r in range(R):
                a       = softmax(-accum / T)            # support over eligible (low cost -> high support)
                a_class = onehot.T @ a                   # [C] per-action-class aggregated support
                inhib   = onehot @ (W_lat @ a_class)     # [n_elig] inhibition received per candidate
                accum   = accum + inhib                  # raise the cost of inhibited candidates

        ``W_lat[i, j]`` is how much class i is suppressed by class j (a candidate
        whose rival classes carry strong support has its cost raised). At init
        ``W_lat == 0`` -> ``inhib == 0`` -> ``accum`` unchanged across rounds ->
        bit-identical to the legacy one-shot path. ``W_lat`` may go negative (the
        signed three-factor rule renders dis-inhibition) but SAFETY is inherited from
        the envelope regardless: the settling transforms ONLY the eligible subset, so
        a No-Go-excluded candidate is never touched and never selectable however the
        weights move.

        Records the decayed Hebbian class co-activation trace (the outer product of
        the per-round class activations) and arms ``_wlat_pending`` so the next
        ``post_action_update`` applies the same signed-RPE three-factor update used
        for ``w_chan``. The caller has already gated this on the waking path
        (``not simulation_mode``), so a replay/DMN tick neither settles nor records a
        trace (MECH-094). Returns the SETTLED accumulator (same shape as the input).
        """
        n = int(mod_eligible.numel())
        n_classes = int(self.W_lat.shape[0])
        dtype = mod_eligible.dtype
        device = mod_eligible.device

        # First-action class of each eligible candidate (clamped into W_lat's range).
        cls = []
        for gi in eligible_idx.tolist():
            a0 = candidates[int(gi)].actions[:, 0, :].reshape(-1)
            cls.append(min(int(a0.argmax().item()), n_classes - 1))
        cls_t = torch.tensor(cls, dtype=torch.long, device=device)
        onehot = torch.zeros((n, n_classes), dtype=dtype, device=device)
        onehot[torch.arange(n, device=device), cls_t] = 1.0

        rounds = int(getattr(self.config, "learned_settling_rounds", 3))
        temp = max(float(getattr(self.config, "learned_settling_temperature", 1.0)), 1e-6)
        W = self.W_lat.to(dtype=dtype, device=device)

        accum = mod_eligible.clone()
        accum0 = accum.clone()
        coact = torch.zeros((n_classes, n_classes), dtype=dtype, device=device)
        for _r in range(max(0, rounds)):
            a = F.softmax(-accum / temp, dim=0)        # [n_elig] support (low cost -> high)
            a_class = onehot.t() @ a                   # [C] per-class aggregated support
            inhib = onehot @ (W @ a_class)             # [n_elig] received inhibition
            accum = accum + inhib                      # raise inhibited candidates' cost
            coact = coact + torch.outer(a_class, a_class)  # Hebbian class co-activation

        # Record the decayed co-activation trace + arm the W_lat update (waking only).
        # ARC-110: per-loop settling calls this with record_trace=False (a read-only
        # W_lat transform) so multiple loops do not double-arm / overwrite the coact
        # trace; the segregated path arms W_lat learning ONCE from the limbic loop
        # (the ascending-spiral source). The legacy single-arena path always records.
        if record_trace:
            decay = float(getattr(self.config, "learned_settling_elig_decay", 0.9))
            self._wlat_coact_trace = (
                decay * self._wlat_coact_trace
                + coact.detach().to(
                    dtype=self._wlat_coact_trace.dtype, device=self._wlat_coact_trace.device
                )
            )
            self._wlat_pending = True
            self._wlat_last_settle_delta = float((accum - accum0).norm().item())
        return accum

    # ------------------------------------------------------------------ #
    # ARC-110 parallel segregated loops + S2 null + ARC-109 D1/D2 split   #
    # ------------------------------------------------------------------ #

    def _loop_normalize(self, pref: torch.Tensor, mode: str) -> torch.Tensor:
        """Normalise a loop's within-eligible preference so F's raw magnitude
        carries no cross-loop advantage (the ARC-110 conversion mechanism). A loop
        with no spread (empty / flat) contributes nothing (zeros)."""
        if pref.numel() <= 1:
            return pref.new_zeros(pref.shape)
        if mode == "none":
            return pref
        if mode == "range":
            rng = float((pref.max() - pref.min()).item())
            if rng < 1e-9:
                return pref.new_zeros(pref.shape)
            return (pref - pref.min()) / rng
        # zscore (default-when-ON)
        std = float(pref.std().item())
        if std < 1e-9:
            return pref.new_zeros(pref.shape)
        return (pref - pref.mean()) / std

    def _loop_inlayer_null(self, accum: torch.Tensor, alpha: float) -> torch.Tensor:
        """ARC-110 S2: replace a non-motor loop's accumulator (the eligibility/
        settling field the loop settles on) with a magnitude-matched random-structure
        (gaussian) perturbation -- range == alpha * the real accumulator range. Because
        it perturbs the SAME layer the loops settle on (NOT policy softmax temperature,
        the decoupled 700-lineage null), it can actually move the committed-class DV,
        so noise_verified_lifting becomes a meaningful non-vacuity precondition. An
        empty/flat loop (range ~0) yields a zero null (nothing to perturb)."""
        n = int(accum.numel())
        if n <= 1:
            return accum
        real_range = float((accum.max() - accum.min()).item())
        if real_range < 1e-9:
            return accum.new_zeros(accum.shape)
        noise = torch.randn(n, dtype=accum.dtype, device=accum.device)
        noise_range = float((noise.max() - noise.min()).item())
        if noise_range < 1e-9:
            return accum.new_zeros(accum.shape)
        return noise * (alpha * real_range / noise_range)

    def _d1_d2_split(
        self, accum: torch.Tensor, da: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ARC-109: decompose a loop's accumulator (COST units, lower=better) into two
        opponent populations with asymmetric dopamine gain. Go/D1 = relu(-accum) (the
        promote side) potentiated by DA (LTP, gain 1 + d1_da_gain*da); No-Go/D2 =
        relu(+accum) (the suppress side) depressed by DA (LTD, gain 1 - d2_da_gain*da).
        Net preference (COST) = D2_activity - D1_activity. At da==0 and gains==1 the net
        == relu(accum) - relu(-accum) == accum EXACTLY (bit-identical to the additive
        scalar; the dissociation is earned only once da != 0). Returns (net, D1, D2);
        D1 and D2 carry the conflict-vs-indifference distinction the scalar destroys."""
        go = F.relu(-accum)
        nogo = F.relu(accum)
        d1_gain = 1.0 + float(getattr(self.config, "d1_da_gain", 1.0)) * da
        d2_gain = max(0.0, 1.0 - float(getattr(self.config, "d2_da_gain", 1.0)) * da)
        d1_act = d1_gain * go
        d2_act = d2_gain * nogo
        return (d2_act - d1_act), d1_act, d2_act

    def _segregated_loop_arbitrate(
        self,
        eligible_idx: torch.Tensor,
        raw_scores: torch.Tensor,
        lcg_terms: List[Tuple[int, torch.Tensor]],
        use_finer: bool,
        candidates: List[Trajectory],
        committed: bool,
        temperature: float,
        simulation_mode: bool,
    ) -> int:
        """ARC-110: parallel segregated cortico-BG-thalamic loop arbitration.

        Replaces the single-arena within-eligible argmin (when use_loop_segregation is
        on). Each loop runs its OWN within-loop competition FIRST -- the MOTOR loop is F
        (raw_scores); the ASSOCIATIVE and LIMBIC loops accumulate their own modulatory
        channel subsets, optionally settle (MECH-450 W_lat) and split into D1/D2
        populations (ARC-109) -- then cross-loop arbitration AFTER via Haber's ascending
        dopamine spiral (limbic -> assoc -> motor). Each loop's preference is NORMALISED
        before arbitration so F's raw magnitude no longer dominates (the conversion
        mechanism). Operates STRICTLY within the F+Go/No-Go eligible set, so a non-motor
        loop can FLIP the within-eligible winner but can NEVER re-admit a suppressed
        candidate (safety inherited from the envelope). Returns the LOCAL index into
        eligible_idx of the committed candidate.
        """
        device = raw_scores.device
        dtype = raw_scores.dtype
        n_elig = int(eligible_idx.numel())

        # Motor loop = F over the eligible set (COST, lower=better).
        motor_pref = raw_scores.detach()[eligible_idx].to(dtype=dtype, device=device)

        # Partition modulatory channels into the non-motor loops by name.
        index_names = _FCG_CHANNEL_NAMES if use_finer else _LCG_CHANNEL_NAMES
        cmap = dict(_LOOP_DEFAULT_CHANNEL_MAP)
        cmap.update(getattr(self.config, "loop_segregation_channel_map", {}) or {})
        default_loop = getattr(self.config, "loop_segregation_default_loop", "associative")
        if default_loop not in ("associative", "limbic"):
            default_loop = "associative"
        loop_accum = {"associative": None, "limbic": None}
        loop_of_channel: Dict[int, str] = {}
        n_loop_channels = {"associative": 0, "limbic": 0}
        for ch_idx, term in (lcg_terms or []):
            name = index_names[ch_idx] if 0 <= ch_idx < len(index_names) else None
            loop = cmap.get(name, default_loop)
            if loop not in loop_accum:
                loop = default_loop
            t = term.detach()[eligible_idx].to(dtype=dtype, device=device)
            loop_accum[loop] = t if loop_accum[loop] is None else loop_accum[loop] + t
            loop_of_channel[ch_idx] = loop
            n_loop_channels[loop] += 1

        zeros = motor_pref.new_zeros(n_elig)
        assoc_accum = loop_accum["associative"] if loop_accum["associative"] is not None else zeros
        limbic_accum = loop_accum["limbic"] if loop_accum["limbic"] is not None else zeros

        # S2: in-layer same-layer null (replaces non-motor loop accums with
        # magnitude-matched random structure). Motor (F) is never nulled.
        noise_on = bool(getattr(self.config, "loop_segregation_noise_on", False))
        if noise_on:
            alpha = float(getattr(self.config, "loop_segregation_noise_alpha", 1.0))
            assoc_accum = self._loop_inlayer_null(assoc_accum, alpha)
            limbic_accum = self._loop_inlayer_null(limbic_accum, alpha)

        # ARC-109: D1/D2 opponent populations per loop (dissociates conflict vs
        # indifference). da = bounded tonic-DA proxy (the shared ARC-108 value
        # baseline V-hat_t, tanh-squashed). At da==0 the split is bit-identical.
        d1d2 = bool(getattr(self.config, "use_d1_d2_population_split", False))
        d1d2_conflict = 0.0
        if d1d2:
            da = math.tanh(float(self._lcg_value_baseline))
            assoc_accum, _a_d1, _a_d2 = self._d1_d2_split(assoc_accum, da)
            limbic_accum, _l_d1, _l_d2 = self._d1_d2_split(limbic_accum, da)
            # Conflict (both populations co-active) is representable here but NOT in
            # the additive scalar; near-zero == indifference. Mean over the limbic
            # loop (the value loop the disorder axis localises to).
            if _l_d1.numel() >= 1:
                d1d2_conflict = float(torch.minimum(_l_d1, _l_d2).mean().item())

        # Per-loop within-loop settling (MECH-450 W_lat as a shared surround-inhibition
        # over action classes). record_trace only on the limbic loop (spiral source) so
        # W_lat learning is single-sourced and not double-armed. Waking-only.
        settle_on = (
            bool(getattr(self.config, "use_learned_settling_step", False))
            and not simulation_mode
            and n_elig >= 2
        )
        if settle_on:
            assoc_accum = self._lateral_settle(
                assoc_accum, candidates, eligible_idx, record_trace=False
            )
            limbic_accum = self._lateral_settle(
                limbic_accum, candidates, eligible_idx, record_trace=True
            )
            self.last_score_diagnostics["learned_settling_active"] = True
            self.last_score_diagnostics["learned_settling_round_delta"] = (
                self._wlat_last_settle_delta
            )

        # Normalise each loop's preference, then arbitrate (Haber ascending spiral).
        norm = getattr(self.config, "loop_segregation_normalize", "zscore")
        motor_z = self._loop_normalize(motor_pref, norm)
        assoc_z = self._loop_normalize(assoc_accum, norm)
        limbic_z = self._loop_normalize(limbic_accum, norm)
        g_a = float(getattr(self.config, "loop_segregation_spiral_gain_assoc", 1.0))
        g_l = float(getattr(self.config, "loop_segregation_spiral_gain_limbic", 1.0))
        m_a = float(getattr(self.config, "loop_segregation_motor_authority", 1.0))
        final = m_a * motor_z + g_a * assoc_z + g_l * limbic_z

        # MECH-440 / MECH-441: add the propagating exploration term (weight noise +
        # disagreement bonus) to the cross-loop arbitration so it reaches the
        # committed action under loop segregation too (the ARC-110 ON arm of the
        # MECH-440 falsifier). Indexed onto the eligible set; None / OFF -> no-op.
        if self._last_explore_term is not None:
            _et = self._last_explore_term.detach().to(
                dtype=final.dtype, device=final.device
            )
            if _et.numel() > int(eligible_idx.max().item()):
                final = final + _et[eligible_idx]

        # Commit (argmin when committed; gap-agnostic softmax sample otherwise).
        if n_elig == 1:
            local = 0
        elif committed:
            local = int(final.argmin().item())
        else:
            probs = F.softmax(-final / max(temperature, 1e-6), dim=0)
            local = int(torch.multinomial(probs, 1).item())

        # Non-degeneracy diagnostics + per-loop winners.
        motor_win = int(motor_pref.argmin().item())
        assoc_win = int(assoc_z.argmin().item()) if assoc_z.numel() >= 1 else motor_win
        limbic_win = int(limbic_z.argmin().item()) if limbic_z.numel() >= 1 else motor_win
        self.last_score_diagnostics["loop_segregation_active"] = True
        self.last_score_diagnostics["loop_segregation_noise_active"] = noise_on
        self.last_score_diagnostics["loop_d1_d2_active"] = d1d2
        self.last_score_diagnostics["loop_d1_d2_conflict_signal"] = d1d2_conflict
        self.last_score_diagnostics["loop_committed_neq_motor_winner"] = bool(
            local != motor_win
        )
        self.last_score_diagnostics["loop_cross_loop_winner_disagreement"] = bool(
            assoc_win != motor_win or limbic_win != motor_win
        )
        self.last_score_diagnostics["loop_assoc_pref_range"] = float(
            (assoc_z.max() - assoc_z.min()).item()
        ) if assoc_z.numel() >= 1 else 0.0
        self.last_score_diagnostics["loop_limbic_pref_range"] = float(
            (limbic_z.max() - limbic_z.min()).item()
        ) if limbic_z.numel() >= 1 else 0.0
        self.last_score_diagnostics["loop_n_assoc_channels"] = n_loop_channels["associative"]
        self.last_score_diagnostics["loop_n_limbic_channels"] = n_loop_channels["limbic"]

        # MECH-452: which loop "voted" for the committed action (its within-loop winner
        # matched the commit), so the eligibility-trace recording credits only voting
        # loops. Stored for the recording site below.
        self._loop_of_channel = loop_of_channel
        self._loop_voted = {
            "motor": (motor_win == local),
            "associative": (assoc_win == local),
            "limbic": (limbic_win == local),
        }
        return local

    def select(
        self,
        candidates: List[Trajectory],
        temperature: float = 1.0,
        goal_state: Optional[GoalState] = None,
        harm_bridge: Optional["nn.Module"] = None,
        use_harm_variance_commit: bool = False,
        terrain_weight: Optional[torch.Tensor] = None,
        sweep_threshold_reduction: float = 0.0,
        z_harm_a: Optional[torch.Tensor] = None,
        harm_forward_model: Optional["nn.Module"] = None,
        z_harm_s_current: Optional[torch.Tensor] = None,
        score_bias: Optional[torch.Tensor] = None,
        score_bias_channels: Optional[Dict[str, torch.Tensor]] = None,
        score_diversity: Optional[Any] = None,
        channel_route_bias: Optional[torch.Tensor] = None,
        e2_forward_pe_per_candidate: Optional[torch.Tensor] = None,
        model_disagreement_per_candidate: Optional[torch.Tensor] = None,
        go_nogo_signals: Optional[Dict[str, Any]] = None,
        simulation_mode: bool = False,
    ) -> SelectionResult:
        """
        Select the best trajectory from candidates.

        Uses dynamic precision (ARC-016) to determine commit threshold.

        Args:
            candidates:             list of Trajectory objects
            temperature:            softmax temperature (exploration vs exploitation)
            harm_bridge:            optional HarmBridge (SD-010); if provided, M(zeta) uses
                                    harm stream scores
            use_harm_variance_commit: if True, commit decision uses variance of harm
                                    scores across candidates rather than z_world running
                                    variance (ARC-016 reframe).
            terrain_weight:         [batch, 2] or None (SD-016 MECH-152).
            sweep_threshold_reduction: MECH-108 BreathOscillator threshold reduction.
            z_harm_a:               SD-011 affective-motivational harm latent [batch, z_harm_a_dim].
                                    When provided and urgency_weight > 0, lowers effective
                                    commit threshold under accumulated threat (D2 avoidance).
                                    When provided and affective_harm_scale > 0, amplifies
                                    lambda_ethical in score_trajectory().
            harm_forward_model:     SD-011/ARC-033 ResidualHarmForward instance. When
                                    provided with z_harm_s_current, replaces harm_bridge
                                    for M(zeta) computation via step-by-step rollout.
            z_harm_s_current:       [batch, z_harm_dim] current sensory-discriminative harm
                                    latent. Required when harm_forward_model is provided.
            score_bias:             SD-032b dACC bias, [K] tensor (one entry per candidate).
                                    Added directly to per-candidate scores before softmax.
                                    Sign convention: lower is better (favourable bias is
                                    negative). None means no bias (backward compat default).
            score_diversity:        MECH-341 substrate handle (Optional). When supplied,
                                    apply_entropy_bonus is composed into scores after
                                    score_bias (Option 1, post-bias-chain), and
                                    stratified_select replaces argmin in the committed
                                    selection path (Option 2). None means no MECH-341
                                    intervention (backward compat default).
            channel_route_bias:     modulatory-bias-selection-authority AMEND
                                    (route-range, 569f/661/654a). Optional [K]
                                    per-candidate bias projected (parameter-free,
                                    range-preserving) from the channel-under-test's
                                    per-candidate representation by the caller (see
                                    project_channel_range). When
                                    use_modulatory_channel_routing is True, it is
                                    folded into the modulatory accumulator the
                                    authority rescales, so the channel's
                                    cross-candidate range reaches the committed argmax.
                                    Its pre-rescale range is exposed as the P0
                                    readiness diagnostic modulatory_channel_route_range.
                                    None means no routing (backward compat default).

        Returns:
            SelectionResult
        """
        if not candidates:
            raise ValueError("No candidate trajectories provided")

        _score_list = []
        _cand_components = [] if self.e3_score_decomp_enabled else None
        for _i, _cand_t in enumerate(candidates):
            # DR-12 (self_model_v4:SELF-4): per-candidate E2 forward-PE so the
            # confidence down-weight can change the committed argmin. None ->
            # bit-identical (no penalty). A uniform scalar would be argmin-invariant
            # (V3-EXQ-571 lesson), so the signal is supplied per-candidate.
            _pe_i = (
                e2_forward_pe_per_candidate[_i]
                if e2_forward_pe_per_candidate is not None
                else None
            )
            _s = self.score_trajectory(
                _cand_t, goal_state=goal_state, harm_bridge=harm_bridge,
                terrain_weight=terrain_weight,
                harm_forward_model=harm_forward_model,
                z_harm_s_current=z_harm_s_current,
                z_harm_a=z_harm_a,
                e2_forward_pe=_pe_i,
            )
            _score_list.append(_s)
            if self.e3_score_decomp_enabled:
                _cand_components.append(dict(self._last_traj_components))
        scores = torch.stack(_score_list)
        scores = scores.mean(dim=-1)

        # V3-EXQ-563c: capture raw score range for diagnostics + normalisation.
        raw_scores = scores.detach()
        raw_score_range = float((raw_scores.max() - raw_scores.min()).item())
        raw_score_std = float(raw_scores.std().item())
        # ControlVector logging (rec-B): the PRE-bias per-candidate scores are
        # the primary value axis (V_outcome). Stored unconditionally alongside
        # last_scores (the post-bias scores); a detached tensor ref, negligible
        # cost. REE convention is lower-is-better, so value = -score.
        self.last_raw_scores = raw_scores

        # modulatory-bias-selection-authority (V3-EXQ-643a fix, 2026-06-06):
        # Track the COMBINED modulatory contribution (score_bias + MECH-341
        # entropy bonus) EXPLICITLY as a small accumulated tensor, instead of
        # reconstructing it below as (scores - raw_scores). Reconstruction by
        # subtraction suffers float32 catastrophic cancellation when the primary
        # scores are large: in V3-EXQ-643 the SD-056-online-trained scores grew
        # to ~1e32, and the real ~0.17-magnitude modulatory range is far below
        # the float32 ULP at 1e32 (~1e25), so (scores - raw_scores) collapsed to
        # EXACTLY 0.0 -> modulatory_range < floor -> the authority gate never
        # fired (active_frac=0.0, scale_factor=0.0 on every arm). The explicit
        # accumulator sums the small bias tensors actually added, so its range is
        # independent of the primary-score magnitude. Mathematically identical to
        # (scores - raw_scores) in exact arithmetic; numerically robust.
        _modulatory_accum: Optional[torch.Tensor] = None

        # ARC-108 JOB-1 step-1: collect the genuinely-separable per-channel bias
        # terms (channel_index, [K] tensor) as they are added to _modulatory_accum,
        # so the learned per-channel weight w_chan can recompose the accumulator as
        # sum_c softplus(w_chan[c]) * channel_bias_c below, and the post-selection
        # eligibility trace can read |channel_bias_c[selected]|. Populated only at
        # the three _modulatory_accum add-sites; inert (unused) unless
        # use_learned_channel_gating is True.
        _lcg_terms: List[Tuple[int, torch.Tensor]] = []

        # MECH-451: select the ACTIVE channel registry. When use_finer_channel_gating
        # is on, _lcg_terms is keyed by the FINER registry (the single score_bias slot
        # exploded into ofc/dacc/lpfc/vigour/liking/gated_policy/residual + the
        # preserved mech341/route) and the recompose / eligibility / three-factor
        # update below ride the parallel w_chan_finer buffer; otherwise the ARC-108
        # base registry + w_chan (byte-identical). The two are mutually exclusive at
        # the score_bias slot -- finer takes precedence.
        _fcg = bool(getattr(self.config, "use_finer_channel_gating", False))
        _chan_index = _FCG_CHANNEL_INDEX if _fcg else _LCG_CHANNEL_INDEX

        # SD-032b dACC bias: additive, same sign convention as raw scores.
        # Applied before last_scores / softmax so downstream consumers see
        # the biased values consistently.
        if score_bias is not None:
            if score_bias.shape != scores.shape:
                raise ValueError(
                    f"score_bias shape {tuple(score_bias.shape)} does not match "
                    f"scores shape {tuple(scores.shape)}"
                )
            bias_tensor = score_bias.to(dtype=scores.dtype, device=scores.device)
            # Optional: rescale bias proportional to raw score range so the
            # relative push stays consistent across environments. SKIPPED when
            # use_modulatory_selection_authority is on -- that flag takes
            # precedence and performs a single gap-relative rescale of the
            # COMBINED modulatory contribution below (pre-scaling the bias here
            # would shift the bias-vs-bonus proportions before that rescale).
            if (
                getattr(self.config, "normalize_score_bias_to_e3_range", False)
                and not getattr(self.config, "use_modulatory_selection_authority", False)
                and raw_score_range > 1e-6
            ):
                bias_range = float(
                    (bias_tensor.max() - bias_tensor.min()).abs().item()
                )
                if bias_range > 1e-9:
                    scale = raw_score_range / bias_range
                    bias_tensor = bias_tensor * scale
            # Base scores add the FULL (summed) score_bias in BOTH modes -- so the
            # scores tensor (and the authority-OFF selection path) is unchanged. Only
            # the _lcg_terms registration (and hence the recomposed _modulatory_accum
            # the authority/shortlist consumes) differs in finer mode.
            scores = scores + bias_tensor
            # Explicit modulatory-contribution tracking (see note above).
            _modulatory_accum = bias_tensor
            if _fcg:
                # MECH-451: register one _lcg_term per FINER channel instead of the
                # single compressed "score_bias" term. score_bias_channels carries the
                # un-summed per-head biases captured in agent.select_action; each is
                # scaled by the SAME normalize factor applied to bias_tensor above so
                # the parts still sum to the (post-normalize) whole. "residual" =
                # bias_tensor - sum(named present) captures everything ELSE summed into
                # score_bias (curiosity / blocked-agency / avoidance / escape / future),
                # so the finer decomposition is EXHAUSTIVE -> at softplus==1 the
                # recompose reproduces bias_tensor EXACTLY (bit-identical at init).
                _named_sum = scores.new_zeros(scores.shape)
                if score_bias_channels:
                    # Re-derive the normalize scale actually applied to bias_tensor so
                    # the captured (pre-normalize) channel tensors are put on the same
                    # footing. When normalization did not fire, _norm_scale == 1.0.
                    _norm_scale = 1.0
                    _raw_sb = score_bias.to(dtype=scores.dtype, device=scores.device)
                    _raw_sb_range = float((_raw_sb.max() - _raw_sb.min()).abs().item())
                    if (
                        getattr(self.config, "normalize_score_bias_to_e3_range", False)
                        and not getattr(self.config, "use_modulatory_selection_authority", False)
                        and raw_score_range > 1e-6
                        and _raw_sb_range > 1e-9
                    ):
                        _norm_scale = raw_score_range / _raw_sb_range
                    for _ch_name in FINER_NAMED_CHANNELS:
                        _ch_bias = score_bias_channels.get(_ch_name)
                        if _ch_bias is None:
                            continue
                        _ch_t = _ch_bias.to(dtype=scores.dtype, device=scores.device)
                        if _ch_t.shape != scores.shape:
                            raise ValueError(
                                f"score_bias_channels['{_ch_name}'] shape "
                                f"{tuple(_ch_t.shape)} does not match scores shape "
                                f"{tuple(scores.shape)}"
                            )
                        _ch_t = _ch_t * _norm_scale
                        _named_sum = _named_sum + _ch_t
                        _lcg_terms.append((_chan_index[_ch_name], _ch_t))
                # residual = (post-normalize) full bias minus the named parts.
                _residual = bias_tensor - _named_sum
                _lcg_terms.append((_chan_index["residual"], _residual))
            else:
                _lcg_terms.append((_chan_index["score_bias"], bias_tensor))

        # MECH-341 Option 1 (entropy bonus): per-candidate POSITIVE bias on
        # first-action classes over-represented in the pool. Composed AFTER
        # the dACC / lateral_pfc / ofc / mech295 / curiosity / tonic_vigor
        # score_bias chain and BEFORE last_scores / softmax so diagnostics
        # capture the post-MECH-341 scores. Bit-identical when score_diversity
        # is None or its use_entropy_bonus sub-flag is False. See MECH-341
        # / behavioral_diversity_isolation_plan.md / V3-EXQ-608 routing.
        mech341_bonus_tensor = None
        if score_diversity is not None:
            mech341_bonus = score_diversity.apply_entropy_bonus(
                scores=scores, candidates=candidates, simulation_mode=False
            )
            mech341_bonus_tensor = mech341_bonus
            scores = scores + mech341_bonus
            # Explicit modulatory-contribution tracking (see note above).
            _modulatory_accum = (
                mech341_bonus if _modulatory_accum is None
                else _modulatory_accum + mech341_bonus
            )
            _lcg_terms.append((_chan_index["mech341"], mech341_bonus))

        # modulatory-bias-selection-authority AMEND (route-range, 569f/661/654a,
        # 2026-06-10): fold the channel-under-test's range-preserving per-candidate
        # routed bias (caller-built via project_channel_range) into BOTH scores and
        # the modulatory accumulator the authority rescales, so a channel whose
        # REPRESENTATION carries cross-candidate range (world-summary / rule_state /
        # curiosity / coherence) reaches the committed argmax instead of being
        # flattened by its consuming bias head. modulatory_channel_route_range is the
        # P0 readiness diagnostic = the RAW routed range (pre-normalise, pre-rescale):
        # a retest asserts it > floor for the channel under test BEFORE scoring any
        # behavioural falsifier (so an unrouted channel cannot self-route a false
        # negative). The routed bias is normalised to unit zero-mean range so its
        # contribution stays bounded even when the authority is OFF; when the
        # authority is ON it re-normalises the combined accumulator to
        # gain * raw_score_range regardless.
        modulatory_channel_route_active = False
        modulatory_channel_route_range = 0.0
        if (getattr(self.config, "use_modulatory_channel_routing", False)
                and channel_route_bias is not None):
            route = channel_route_bias.to(dtype=scores.dtype, device=scores.device)
            if route.shape != scores.shape:
                raise ValueError(
                    f"channel_route_bias shape {tuple(route.shape)} does not match "
                    f"scores shape {tuple(scores.shape)}"
                )
            route_range = float((route.max() - route.min()).item())
            modulatory_channel_route_range = route_range
            route_floor = getattr(
                self.config, "modulatory_channel_route_min_range_floor", 1e-6
            )
            if route_range > route_floor:
                route_unit = (route - route.min()) / route_range  # [0, 1]
                route_unit = route_unit - route_unit.mean()        # zero-mean, range 1
                route_weight = float(
                    getattr(self.config, "modulatory_channel_route_weight", 1.0)
                )
                routed = route_weight * route_unit
                scores = scores + routed
                _modulatory_accum = (
                    routed if _modulatory_accum is None
                    else _modulatory_accum + routed
                )
                _lcg_terms.append((_chan_index["route"], routed))
                modulatory_channel_route_active = True

        # ARC-108 JOB-1 step-1: recompose _modulatory_accum as the LEARNED weighted
        # sum  sum_c softplus(w_chan[c]) * channel_bias_c  over the channels actually
        # present this tick. At init softplus(w_chan[c]) == 1.0 -> this reproduces the
        # unweighted accumulator EXACTLY (bit-identical OFF, and bit-identical at init
        # even when ON). Only _modulatory_accum is re-weighted -- raw scores / F (the
        # MECH-448 envelope + the commit decision) are untouched, so learning
        # composes strictly INSIDE the F-bounded MECH-448/449 eligible set the
        # within-eligible shortlist-argmin + the authority rescale consume, and a
        # learned weight can never re-admit a No-Go-suppressed candidate. MECH-450
        # settling (learned W_lat) would compose HERE as factor 2 -- OFF in this
        # build (W_lat == 0): integration point reserved, not enabled.
        # MECH-451: the recompose is registry-agnostic -- it fires for EITHER the
        # ARC-108 global w_chan OR the finer w_chan_finer, picking the active buffer.
        # At init softplus == 1.0 for every channel, so in BOTH modes the recompose
        # reproduces the (summed) modulatory contribution EXACTLY (bit-identical OFF,
        # and bit-identical at init even when ON). The finer mode's per-channel terms
        # sum to bias_tensor by construction (named + residual), so the only thing
        # that diverges from the global path is the per-channel weighting once the
        # finer w_chan_finer entries are trained apart.
        _recompose_on = (
            (getattr(self.config, "use_learned_channel_gating", False) and not _fcg)
            or (_fcg and getattr(self.config, "use_finer_channel_gating", False))
        )
        if _recompose_on and _lcg_terms:
            _w_buf = self.w_chan_finer if _fcg else self.w_chan
            w_soft = F.softplus(_w_buf)  # [C], all == 1.0 at init
            _lcg_acc = None
            for ch_idx, term in _lcg_terms:
                weighted = w_soft[ch_idx].to(
                    dtype=term.dtype, device=term.device
                ) * term
                _lcg_acc = weighted if _lcg_acc is None else _lcg_acc + weighted
            _modulatory_accum = _lcg_acc

        # modulatory-bias-selection-authority (2026-06-03): rescale the COMBINED
        # modulatory contribution (score_bias + mech341_bonus) so its range equals
        # modulatory_authority_gain * raw_score_range. Gives modulatory signals
        # (MECH-314 curiosity, MECH-320 vigor, MECH-341 within-class temperature,
        # dACC, lateral_pfc, ofc, mech295) genuine but bounded authority at
        # committed selection. When OFF (default), modulatory biases are applied
        # as-is (bit-identical to pre-substrate baseline). When ON, the rescaling
        # lets modulatory signals change the argmin in near-tie regimes while staying
        # subdominant when primary harm/goal gaps exceed gain * raw_range.
        # See REE_assembly/evidence/planning/modulatory_bias_selection_authority_*.md
        modulatory_authority_active = False
        modulatory_authority_scale_factor = 0.0
        modulatory_authority_range = 0.0
        if self.config.use_modulatory_selection_authority:
            scores_raw = raw_scores  # captured before score_bias application above
            # Total modulatory contribution measured DIRECTLY from the accumulated
            # small bias tensors (V3-EXQ-643a fix) -- NOT reconstructed as
            # (scores - scores_raw), which catastrophically cancels at large
            # primary-score magnitude (see note at the accumulator declaration).
            if _modulatory_accum is not None:
                modulatory_total = _modulatory_accum
            else:
                modulatory_total = scores.new_zeros(scores.shape)
            modulatory_range = float((modulatory_total.max() - modulatory_total.min()).item())
            modulatory_authority_range = modulatory_range
            # CONVERSION amend (a) (569g/682, 2026-06-15): normalize_basis selects
            # how the additive authority anchors its target. "range" (default,
            # bit-identical legacy) matches gain*raw_score_RANGE -> outlier-sensitive,
            # flips only near-tie outliers. "std" matches gain*raw_score_STD and
            # rescales by the modulatory STD (robust to outliers) so the structured
            # channel competes against the TYPICAL primary spread (near-decisive
            # candidates), the 569g residual-conversion fix. See E3Config.
            basis = getattr(
                self.config, "modulatory_authority_normalize_basis", "range"
            )
            if basis == "std":
                modulatory_spread = float(modulatory_total.std().item())
                target_range = self.config.modulatory_authority_gain * raw_score_std
            else:
                modulatory_spread = modulatory_range
                target_range = self.config.modulatory_authority_gain * raw_score_range
            if modulatory_spread > self.config.modulatory_authority_min_range_floor:
                # Rescale modulatory contribution to target_range (basis-dependent).
                scale_factor = target_range / modulatory_spread
                modulatory_authority_scale_factor = scale_factor
                modulatory_authority_active = True
                # Apply: scores = raw_scores + scale_factor * (modulatory_total)
                scores = scores_raw + scale_factor * modulatory_total

        # V3-EXQ-563c: score / bias diagnostics (pre-softmax, post-bias).
        bias_detached = (
            score_bias.detach().to(dtype=raw_scores.dtype, device=raw_scores.device)
            if score_bias is not None else raw_scores.new_zeros(raw_scores.shape)
        )

        # DR-12 (self_model_v4:SELF-4) diagnostics: the cross-candidate range of the
        # supplied E2 forward-PE (the pilot's NON-VACUITY gate -- a flat PE cannot
        # change selection) and of the applied penalty. pe_confidence_active is True
        # only when the lever is enabled AND a per-candidate PE was supplied.
        _pe_active = bool(
            getattr(self.config, "use_pe_confidence_weighting", False)
            and e2_forward_pe_per_candidate is not None
            and self.config.pe_confidence_weight != 0.0
        )
        _pe_range = 0.0
        _pe_penalty_range = 0.0
        if e2_forward_pe_per_candidate is not None:
            _pe_vec = e2_forward_pe_per_candidate.detach().reshape(-1).to(
                dtype=raw_scores.dtype, device=raw_scores.device
            )
            if _pe_vec.numel() >= 1:
                _pe_range = float((_pe_vec.max() - _pe_vec.min()).item())
                if _pe_active:
                    _pen_vec = torch.stack([
                        self._pe_confidence_penalty(
                            _pe_vec[_j], device=raw_scores.device, dtype=raw_scores.dtype
                        )
                        for _j in range(_pe_vec.numel())
                    ])
                    _pen_vec = self.config.pe_confidence_weight * _pen_vec
                    _pe_penalty_range = float((_pen_vec.max() - _pen_vec.min()).item())

        self.last_score_diagnostics = {
            "e3_raw_score_range_mean": raw_score_range,
            "e3_raw_score_std_mean": raw_score_std,
            "score_bias_abs_mean": float(bias_detached.abs().mean().item()),
            "score_bias_range_mean": float(
                (bias_detached.max() - bias_detached.min()).item()
            ),
            "score_bias_to_raw_range_ratio": (
                float(
                    (bias_detached.max() - bias_detached.min()).abs().item()
                ) / max(raw_score_range, 1e-9)
            ),
            "normalize_score_bias_active": bool(
                getattr(self.config, "normalize_score_bias_to_e3_range", False)
                and score_bias is not None
                and raw_score_range > 1e-6
            ),
            "modulatory_authority_active": modulatory_authority_active,
            "modulatory_authority_scale_factor": modulatory_authority_scale_factor,
            # V3-EXQ-643a: the true cross-candidate modulatory range the gate
            # keyed on (explicitly tracked; immune to large-score cancellation).
            "modulatory_authority_range": modulatory_authority_range,
            # route-range AMEND (569f/661/654a): P0 readiness gate signals.
            # modulatory_channel_route_range = RAW cross-candidate range of the
            # routed channel bias (pre-normalise, pre-rescale); a retest asserts
            # it > floor for the channel under test before scoring a falsifier.
            "modulatory_channel_route_active": modulatory_channel_route_active,
            "modulatory_channel_route_range": modulatory_channel_route_range,
            # CONVERSION amend (569g/682, 2026-06-15): which normalize basis the
            # additive authority used, and the shortlist-then-modulate diagnostics
            # (active + near-tie-set size). Pre-seeded here; shortlist keys are
            # overwritten at the selection site when lever (b) fires.
            "modulatory_authority_normalize_basis": getattr(
                self.config, "modulatory_authority_normalize_basis", "range"
            ),
            "modulatory_shortlist_active": False,
            "modulatory_shortlist_size": 0,
            "modulatory_shortlist_mode": getattr(
                self.config, "modulatory_shortlist_mode", "margin"
            ),
            # CONVERSION-CEILING / F-dominance conflict-grade (MECH-439, 2026-06-18).
            # conflict_gap_norm is ALWAYS set (when >= 2 candidates) regardless of
            # which factor is active, so the per-tick F-gap regression falsifier can
            # bin every tick (incl. fixed-k / hard-argmin control cells) uniformly.
            "conflict_gap_norm": -1.0,
            "modulatory_shortlist_conflict_graded": False,
            "modulatory_shortlist_k_effective": -1,
            "modulatory_shortlist_gap_norm": -1.0,
            "gap_scaled_commit_active": False,
            "gap_scaled_commit_gap_norm": -1.0,
            "gap_scaled_commit_temperature_eff": -1.0,
            # MECH-448 / ARC-107 rank-preserving F->eligibility demotion (2026-06-20).
            # Pre-seeded; overwritten at the selection site when the lever fires.
            # f_eligibility_excluded_count > 0 is the NON-DEGENERACY signal (the
            # envelope actually excluded a candidate -- not an all-admit vacuous
            # self-route); f_eligibility_winner_neq_f_argmin records when the
            # committed winner differs from the F-argmin (F demoted at commit);
            # f_eligibility_rank_preserving confirms the eligible set is an F-rank
            # prefix (the order-preserving-numerators falsifier check).
            "f_eligibility_demotion_active": False,
            "f_eligibility_envelope_size": -1,
            "f_eligibility_excluded_count": -1,
            "f_eligibility_winner_neq_f_argmin": False,
            "f_eligibility_rank_preserving": True,
            # MECH-450 (ARC-108 factor 2) recurrent-settling step. Pre-seeded;
            # overwritten at the within-eligible site when the settling runs.
            # learned_settling_round_delta is the L2 cross-round movement of the
            # eligible accumulator -- the NON-DEGENERACY signal (the settling actually
            # MOVED the field, not a no-op pass) the falsifier checks.
            "learned_settling_active": False,
            "learned_settling_round_delta": -1.0,
            # ARC-110 / ARC-109 / MECH-452 segregated-loop diagnostics. Pre-seeded;
            # overwritten by _segregated_loop_arbitrate when loop segregation runs.
            # loop_committed_neq_motor_winner + loop_cross_loop_winner_disagreement
            # are the NON-DEGENERACY signals (a non-motor loop actually flipped the
            # within-eligible winner / loops disagreed -- not a vacuous split pinned
            # to the F winner). loop_*_pref_range > 0 confirms a loop carries live
            # cross-candidate variance.
            "loop_segregation_active": False,
            "loop_segregation_noise_active": False,
            "loop_d1_d2_active": False,
            "loop_d1_d2_conflict_signal": 0.0,
            "loop_committed_neq_motor_winner": False,
            "loop_cross_loop_winner_disagreement": False,
            "loop_assoc_pref_range": 0.0,
            "loop_limbic_pref_range": 0.0,
            "loop_n_assoc_channels": 0,
            "loop_n_limbic_channels": 0,
            "loop_local_credited_channels": -1,
            # DR-12 (self_model_v4:SELF-4): E2 forward-PE -> E3 confidence down-weight.
            "pe_confidence_active": _pe_active,
            "pe_confidence_weight": float(
                getattr(self.config, "pe_confidence_weight", 0.0)
            ),
            "e2_forward_pe_range": _pe_range,
            "pe_confidence_penalty_range": _pe_penalty_range,
            # Filled in after selection (requires selected_idx).
            "selected_candidate_rank_before_bias": -1,
            "selected_candidate_rank_after_bias": -1,
        }

        # MECH-440 / MECH-441: propagating exploration signals into the committed
        # selection. Injected AFTER the modulatory-authority rescale + diagnostics
        # build and BEFORE the softmax / within-eligible argmin, so the per-candidate
        # terms PROPAGATE into the committed action (the V3-EXQ-687 non-propagation
        # fix). Added to BOTH `scores` (softmax / log_prob / non-shortlist argmin)
        # AND `_modulatory_accum` (the shortlist/demotion within-eligible argmin at
        # the selection site). For the segregated-loop path (ARC-110) the term is
        # carried via self._last_explore_term and added to the cross-loop `final`.
        # Bit-identical OFF: head not built (MECH-440) / weight 0 (MECH-441) /
        # sigma_init 0 -> _explore_term stays None and nothing is added.
        self._last_explore_term = None
        _explore_term = None
        _ns_bias_range = 0.0
        _md_range = 0.0
        _md_mean = 0.0
        if getattr(self.config, "use_noisy_selection_head", False) and len(candidates) >= 1:
            _act_feats = self._candidate_action_features(candidates)
            if _act_feats is not None:
                self._ensure_noisy_head(
                    int(_act_feats.shape[-1]), scores.device, scores.dtype
                )
            if _act_feats is not None and self.noisy_selection_head is not None:
                # Local F-gap confidence signal for the self-anneal EMA (MECH-094:
                # observe is a no-op on simulation ticks).
                if raw_scores.shape[0] >= 2:
                    _sr, _ = torch.sort(raw_scores)
                    _gn = max(
                        0.0,
                        min(1.0, float((_sr[1] - _sr[0]).item()) / (raw_score_range + 1e-8)),
                    )
                    self.noisy_selection_head.observe_gap(
                        _gn, simulation_mode=simulation_mode
                    )
                _nb = self.noisy_selection_head(
                    _act_feats, simulation_mode=simulation_mode
                ).to(dtype=scores.dtype, device=scores.device)
                if _nb.numel() == scores.shape[0]:
                    _explore_term = _nb
                    _ns_bias_range = self.noisy_selection_head._last_bias_range
        if (
            getattr(self.config, "use_model_disagreement_curiosity", False)
            and model_disagreement_per_candidate is not None
            and not simulation_mode
            and float(getattr(self.config, "model_disagreement_weight", 0.0)) != 0.0
        ):
            _dv = model_disagreement_per_candidate.detach().reshape(-1).to(
                dtype=scores.dtype, device=scores.device
            )
            if _dv.numel() == scores.shape[0]:
                _mode = str(getattr(self.config, "model_disagreement_mode", "linear"))
                if _mode == "saturating":
                    _scl = float(getattr(self.config, "model_disagreement_scale", 1.0))
                    _dval = 1.0 - torch.exp(-_dv / max(_scl, 1e-9))
                else:
                    _dval = _dv
                # Curiosity BONUS = a COST reduction (scores are costs, lower=preferred).
                _md_term = -float(self.config.model_disagreement_weight) * _dval
                _md_range = (
                    float((_md_term.max() - _md_term.min()).item())
                    if _dv.numel() >= 2 else 0.0
                )
                _md_mean = float(_dval.mean().item())
                _explore_term = (
                    _md_term if _explore_term is None else _explore_term + _md_term
                )
        if _explore_term is not None and not bool(torch.isfinite(_explore_term).all()):
            # Defensive: a non-finite exploration term (e.g. unstable untrained
            # activations) must never poison the committed selection. Drop it.
            _explore_term = None
            _ns_bias_range = 0.0
            _md_range = 0.0
        if _explore_term is not None:
            scores = scores + _explore_term
            _modulatory_accum = (
                _explore_term if _modulatory_accum is None
                else _modulatory_accum + _explore_term
            )
            self._last_explore_term = _explore_term.detach()
        self.last_score_diagnostics["noisy_selection_active"] = bool(
            getattr(self.config, "use_noisy_selection_head", False)
            and self.noisy_selection_head is not None
        )
        self.last_score_diagnostics["noisy_selection_bias_range"] = _ns_bias_range
        self.last_score_diagnostics["model_disagreement_active"] = bool(
            getattr(self.config, "use_model_disagreement_curiosity", False)
            and model_disagreement_per_candidate is not None
        )
        self.last_score_diagnostics["model_disagreement_bonus_range"] = _md_range
        self.last_score_diagnostics["model_disagreement_mean"] = _md_mean

        self.last_scores = scores.detach()
        if self.e3_score_decomp_enabled and _cand_components:
            self.last_score_decomp = {
                "per_candidate": _cand_components,
                "n_candidates": len(_cand_components),
            }

        probs = F.softmax(-scores / temperature, dim=0)
        # Pure diagnostic: the pre-commit softmax sampling distribution over candidates
        # (post-noise, post-bias). Lets a falsifier measure pre-commit entropy independent
        # of the committed argmin (the MECH-440 thrash-vs-carve discriminator: a temperature
        # lift raises THIS but not the committed-class entropy). No behaviour change.
        self.last_precommit_probs = probs.detach()

        # ARC-016 commit decision: two modes
        # Default: commit when z_world running_variance is LOW (world-stability signal)
        # Reframe: commit when variance of harm scores across candidates is LOW
        #
        # MECH-108: BreathOscillator sweep reduces effective threshold, creating
        # periodic uncommitted windows even after training converges variance below
        # base threshold. Without this, the agent becomes permanently committed.
        effective_threshold = self.commit_threshold
        if sweep_threshold_reduction > 0.0:
            effective_threshold = effective_threshold * (1.0 - sweep_threshold_reduction)

        # SD-011: z_harm_a urgency modulation.
        # High accumulated threat -> LOWER effective threshold -> commit faster.
        # D2 avoidance escape response. Capped by urgency_max to prevent
        # threshold collapse to zero (which would produce permanent commitment).
        urgency_applied = 0.0
        if z_harm_a is not None and self.config.urgency_weight > 0.0:
            z_harm_a_norm = z_harm_a.norm(dim=-1).mean().item()
            urgency_applied = min(
                z_harm_a_norm * self.config.urgency_weight,
                self.config.urgency_max,
            )
            effective_threshold = effective_threshold * (1.0 - urgency_applied)

        if use_harm_variance_commit and harm_bridge is not None:
            harm_scores = torch.stack([
                self.compute_harm_stream_cost(t, harm_bridge).mean()
                for t in candidates
            ])
            harm_score_variance = harm_scores.var().item()
            committed = harm_score_variance < effective_threshold
        else:
            committed = self._running_variance < effective_threshold
        # CONVERSION amend (b) -- shortlist-then-modulate (569g/682, 2026-06-15).
        # The pre-registered architectural fallback: F (raw primary scores) filters
        # to a near-tie set (candidates within modulatory_shortlist_margin *
        # raw_score_range of the best raw score), then the modulatory channel
        # (_modulatory_accum) arbitrates the winner WITHIN that set -- argmin when
        # committed, softmax-sampled when uncommitted. The structured channel is
        # load-bearing without out-magnitude-ing F, and SAFETY is preserved at any
        # internal strength (clearly-harmful candidates outside the shortlist are
        # never selectable). Takes precedence over the additive-authority rescale +
        # argmin/stratified selection when enabled. Bit-identical OFF. See
        # failure_autopsy_V3-EXQ-569g_2026-06-14 + behavioral_diversity_isolation:GAP-A.
        #
        # CONVERSION-CEILING / F-dominance conflict-grade (MECH-439, 2026-06-18).
        # Normalized top-F gap in [0, 1]: the gap between the F-best and
        # F-second-best raw (primary) scores, scaled by the raw-score range. This
        # ONE quantity drives BOTH Factor A (conflict-graded shortlist width) and
        # Factor B (gap-scaled commit temperature) -- two renderings of the BG
        # hyperdirect conflict-grade (grade the decision by the top-F gap). None
        # when there are < 2 candidates (both factors then no-op).
        _conflict_gap_norm: Optional[float] = None
        if raw_scores.shape[0] >= 2:
            _sorted_raw, _ = torch.sort(raw_scores)
            _top_f_gap = float((_sorted_raw[1] - _sorted_raw[0]).item())
            _conflict_gap_norm = max(
                0.0, min(1.0, _top_f_gap / (raw_score_range + 1e-8))
            )
            self.last_score_diagnostics["conflict_gap_norm"] = float(
                _conflict_gap_norm
            )
        _f_demotion_active = bool(
            getattr(self.config, "use_f_eligibility_demotion", False)
        )
        shortlist_idx: Optional[int] = None
        if (
            (
                getattr(self.config, "use_modulatory_shortlist_then_modulate", False)
                or _f_demotion_active
            )
            and _modulatory_accum is not None
            and raw_scores.shape[0] >= 2
        ):
            # MECH-448 (ARC-107) f_demotion takes precedence over the margin/top_k
            # shortlist modes when both flags are set.
            shortlist_mode = (
                "f_demotion" if _f_demotion_active
                else str(getattr(self.config, "modulatory_shortlist_mode", "margin"))
            )
            if shortlist_mode == "f_demotion":
                # MECH-448 / ARC-107 (2026-06-20): F decides ELIGIBILITY only. Build
                # the graded, rank-preserving divisive-normalisation envelope
                # (share-of-competing-field, absolute floor); the existing
                # within-eligible _modulatory_accum arbitration below then picks the
                # committed action with F REMOVED from the final argmin (the
                # constitutional escalation past the conflict-grade near-tie family).
                eligible_idx = self._f_eligibility_envelope(raw_scores)
            elif shortlist_mode == "top_k":
                # TOP-K shortlist (569h-autopsy CONVERSION amend, 2026-06-16):
                # the F-best k candidates by primary score (lower-is-better ->
                # k smallest raw_scores). A SMALL fixed k gives an eligible set
                # whose MEMBERSHIP rotates with state, so the within-set argmin
                # of the routed modulatory channel converts the per-candidate
                # range into committed-action diversity -- where the margin mode
                # (a near-whole, state-stable set: V3-EXQ-684 size 6.25-8.54)
                # collapsed to the channel's global favorite (entropy 0.337 <
                # proposer 0.549). Safety preserved: only the k F-best are
                # selectable, so clearly-harmful candidates are never eligible.
                #
                # FACTOR A (MECH-439): conflict-graded width. When enabled, the
                # fixed k is replaced by a state-graded k that widens at near-ties
                # and narrows to 1 at a decisive F-gap:
                #   k = clamp(round(k_max - (k_max-1)*gap_norm), 1, K)
                # gap_norm ~ 0 (near-tie) -> k = k_max (wider eligible set, slower
                # commit, the STN threshold-raise); gap_norm ~ 1 (decisive) ->
                # k = 1 (fast commit on the F-winner). F still gates ELIGIBILITY
                # only -- absent from the within-set arbitration below. Safety is
                # the same top-k guarantee (a candidate with a large F-gap above
                # the best is never admitted). Bit-identical OFF (fixed-k path).
                if (
                    getattr(
                        self.config,
                        "modulatory_shortlist_conflict_graded",
                        False,
                    )
                    and _conflict_gap_norm is not None
                ):
                    k_max = int(
                        getattr(self.config, "modulatory_shortlist_k_max", 6)
                    )
                    k_max = max(1, min(k_max, int(raw_scores.shape[0])))
                    k = int(round(k_max - (k_max - 1) * _conflict_gap_norm))
                    k = max(1, min(k, int(raw_scores.shape[0])))
                    self.last_score_diagnostics[
                        "modulatory_shortlist_conflict_graded"
                    ] = True
                    self.last_score_diagnostics[
                        "modulatory_shortlist_k_effective"
                    ] = k
                    self.last_score_diagnostics[
                        "modulatory_shortlist_gap_norm"
                    ] = float(_conflict_gap_norm)
                else:
                    k = int(getattr(self.config, "modulatory_shortlist_k", 3))
                    k = max(1, min(k, int(raw_scores.shape[0])))
                eligible_idx = torch.topk(
                    raw_scores, k, largest=False
                ).indices.flatten()
            else:
                # MARGIN shortlist (legacy, bit-identical): near-tie set within
                # modulatory_shortlist_margin * raw_score_range of the best raw.
                margin = float(
                    getattr(self.config, "modulatory_shortlist_margin", 0.25)
                )
                best_raw = float(raw_scores.min().item())
                cutoff = best_raw + margin * raw_score_range
                eligible_idx = torch.nonzero(
                    raw_scores <= cutoff, as_tuple=False
                ).flatten()
            # MECH-449 (ARC-107): the Go/No-Go eligibility constitution governs
            # WHICH candidates may compete for the pallidal gate. It acts on the
            # F-built eligible set BEFORE the within-eligible arbitration, so
            # No-Go suppression on an axis ORTHOGONAL to F-rank (safety/staleness/
            # perseveration/low-viability) reaches the committed argmin where
            # rank-preserving demotion structurally cannot, and bounded Go can
            # re-admit a lawfully-eligible demoted channel. No-op default ->
            # gate skipped -> bit-identical OFF.
            if getattr(self.config, "use_go_nogo_constitution", False):
                eligible_idx = self._go_nogo_eligibility_gate(
                    eligible_idx, raw_scores, go_nogo_signals
                )
            n_eligible = int(eligible_idx.numel())
            if n_eligible >= 1:
                # ARC-110: parallel segregated cortico-BG-thalamic loop arbitration
                # REPLACES the single-arena within-eligible argmin when
                # use_loop_segregation is on. The motor (F) / associative / limbic
                # loops each run within-loop competition first, then cross-loop
                # arbitration -- so F dominates only the motor loop. Operates strictly
                # within the F+Go/No-Go eligible set (safety inherited). Default OFF ->
                # the legacy single-arena path below runs UNCHANGED (bit-identical).
                if getattr(self.config, "use_loop_segregation", False):
                    local = self._segregated_loop_arbitrate(
                        eligible_idx=eligible_idx,
                        raw_scores=raw_scores,
                        lcg_terms=_lcg_terms,
                        use_finer=_fcg,
                        candidates=candidates,
                        committed=committed,
                        temperature=temperature,
                        simulation_mode=simulation_mode,
                    )
                    shortlist_idx = int(eligible_idx[local].item())
                else:
                    mod_eligible = _modulatory_accum.detach()[eligible_idx]
                    # MECH-450 (ARC-108 factor 2): bounded recurrent lateral-inhibition
                    # settling over the eligible set BEFORE the within-eligible commit, so
                    # the committed action emerges from a settling competition (B1) and the
                    # additive blend becomes competitive winner-take-most (B3-blend). Operates
                    # ONLY on the eligible subset -> safety inherited from the envelope.
                    # Waking-only (no settling on a simulation tick); no-op at init (W_lat==0)
                    # -> bit-identical OFF. Needs >= 2 eligible candidates to compete.
                    if (getattr(self.config, "use_learned_settling_step", False)
                            and not simulation_mode
                            and n_eligible >= 2):
                        mod_eligible = self._lateral_settle(
                            mod_eligible, candidates, eligible_idx
                        )
                        self.last_score_diagnostics["learned_settling_active"] = True
                        self.last_score_diagnostics["learned_settling_round_delta"] = (
                            self._wlat_last_settle_delta
                        )
                    if n_eligible == 1:
                        local = 0
                    elif committed:
                        # FACTOR B (MECH-439): gap-scaled entropy-regularized commit.
                        # Soften the within-shortlist committed argmin over the routed
                        # modulatory channel into a gap-scaled multinomial. SAFETY is
                        # preserved by Factor A: the eligible set is the F-bounded
                        # near-tie set, so a hot commit-T cannot promote a clearly-
                        # harmful candidate (none are eligible). Bit-identical OFF
                        # (hard argmin path).
                        if (
                            getattr(
                                self.config,
                                "use_gap_scaled_commit_temperature",
                                False,
                            )
                            and _conflict_gap_norm is not None
                        ):
                            local = self._gap_scaled_commit_pick(
                                mod_eligible, _conflict_gap_norm, temperature
                            )
                        else:
                            local = int(mod_eligible.argmin().item())
                    else:
                        shortlist_probs = F.softmax(
                            -mod_eligible / temperature, dim=0
                        )
                        local = int(torch.multinomial(shortlist_probs, 1).item())
                    shortlist_idx = int(eligible_idx[local].item())
            if _f_demotion_active:
                # MECH-448 diagnostics. excluded_count > 0 is the NON-DEGENERACY
                # signal (the envelope actually excluded a candidate, not all-admit);
                # winner_neq_f_argmin records F demoted at commit; rank_preserving
                # confirms the eligible set is an F-rank prefix (every eligible cost
                # <= every excluded cost -- tie-robust).
                _n_total = int(raw_scores.shape[0])
                _f_argmin = int(raw_scores.argmin().item())
                if 0 < n_eligible < _n_total:
                    _mask = torch.ones(
                        _n_total, dtype=torch.bool, device=raw_scores.device
                    )
                    _mask[eligible_idx] = False
                    _excl_idx = torch.nonzero(_mask, as_tuple=False).flatten()
                    _rank_preserving = bool(
                        (
                            raw_scores[eligible_idx].max()
                            <= raw_scores[_excl_idx].min() + 1e-6
                        ).item()
                    )
                else:
                    _rank_preserving = True
                self.last_score_diagnostics["f_eligibility_demotion_active"] = True
                self.last_score_diagnostics["f_eligibility_envelope_size"] = n_eligible
                self.last_score_diagnostics["f_eligibility_excluded_count"] = (
                    _n_total - n_eligible
                )
                self.last_score_diagnostics["f_eligibility_winner_neq_f_argmin"] = bool(
                    shortlist_idx is not None and shortlist_idx != _f_argmin
                )
                self.last_score_diagnostics["f_eligibility_rank_preserving"] = (
                    _rank_preserving
                )
            else:
                self.last_score_diagnostics["modulatory_shortlist_active"] = True
                self.last_score_diagnostics["modulatory_shortlist_size"] = n_eligible
                self.last_score_diagnostics["modulatory_shortlist_mode"] = shortlist_mode

        if shortlist_idx is not None:
            # Lever (b) selected within the F-filtered near-tie set.
            selected_idx = int(shortlist_idx)
        elif committed:
            # MECH-341 Option 2 (class-stratified selection): replace argmin
            # with softmax-sampling across first-action-class representatives
            # whenever the pool admits stratification. Forces >= 2 first-
            # action classes to survive the selection step when the proposer
            # supplied >= 2 classes. Falls through to legacy argmin when the
            # substrate is disabled, when sub-flag is False, or when the
            # pool has too few unique classes. See MECH-341 /
            # behavioral_diversity_isolation_plan.md / V3-EXQ-608 routing.
            stratified_idx: Optional[int] = None
            if score_diversity is not None:
                stratified_idx = score_diversity.stratified_select(
                    scores=scores, candidates=candidates, simulation_mode=False
                )
            if stratified_idx is not None:
                selected_idx = int(stratified_idx)
            elif (
                getattr(self.config, "use_gap_scaled_commit_temperature", False)
                and _conflict_gap_norm is not None
            ):
                # FACTOR B standalone (no Factor-A shortlist active): soften the
                # committed argmin over the F-dominated scores, but RESTRICT the
                # softmax to an F-eligibility envelope (candidates within
                # gap_scaled_commit_harm_floor * raw_score_range of the best raw
                # score) so a hot commit-T in a near-tie can NEVER softmax-promote
                # a clearly-harmful candidate -- the SAFETY GATE the backfill
                # flagged. Bit-identical OFF (hard argmin path).
                harm_floor = float(
                    getattr(self.config, "gap_scaled_commit_harm_floor", 0.25)
                )
                best_raw = float(raw_scores.min().item())
                envelope = best_raw + harm_floor * raw_score_range
                env_idx = torch.nonzero(
                    raw_scores <= envelope, as_tuple=False
                ).flatten()
                if int(env_idx.numel()) <= 1:
                    selected_idx = int(scores.argmin().item())
                else:
                    local = self._gap_scaled_commit_pick(
                        scores[env_idx], _conflict_gap_norm, temperature
                    )
                    selected_idx = int(env_idx[local].item())
            else:
                selected_idx = int(scores.argmin().item())
        else:
            # MECH-341 retune (2026-05-28): apply stratified_select on the
            # uncommitted branch too. V3-EXQ-611 (2026-05-27) showed
            # n_stratified_fired=0 across all 3 seeds because the committed
            # branch was never entered in the validation episodes; the prior
            # implementation gated Option-2 to committed selection only. The
            # categorical-preservation semantic applies equally to the
            # uncommitted (multinomial) path: when the pool admits >= 2
            # first-action classes, sample across class-representatives
            # rather than across raw softmax probabilities. Bit-identical
            # when score_diversity is None or when the sub-flavour flag is
            # False (stratified_select returns None; legacy multinomial path
            # taken). MECH-094 preserved by simulation_mode=False kwarg.
            # Falsifier: V3-EXQ-611b 6-arm retune sweep.
            stratified_idx = None
            if score_diversity is not None:
                stratified_idx = score_diversity.stratified_select(
                    scores=scores, candidates=candidates, simulation_mode=False
                )
            if stratified_idx is not None:
                selected_idx = int(stratified_idx)
            else:
                selected_idx = int(torch.multinomial(probs, 1).item())

        # V3-EXQ-571: record which candidate was selected into decomp dict.
        if self.e3_score_decomp_enabled and self.last_score_decomp:
            self.last_score_decomp["selected_idx"] = selected_idx

        # Update post-selection rank diagnostics now that selected_idx is known.
        if score_bias is not None:
            sorted_raw = torch.argsort(raw_scores).tolist()
            sorted_biased = torch.argsort(scores.detach()).tolist()
            self.last_score_diagnostics["selected_candidate_rank_before_bias"] = (
                sorted_raw.index(selected_idx)
            )
            self.last_score_diagnostics["selected_candidate_rank_after_bias"] = (
                sorted_biased.index(selected_idx)
            )
        # When no bias: rank is the same before and after.
        else:
            sorted_raw = torch.argsort(raw_scores).tolist()
            rank = sorted_raw.index(selected_idx)
            self.last_score_diagnostics["selected_candidate_rank_before_bias"] = rank
            self.last_score_diagnostics["selected_candidate_rank_after_bias"] = rank

        selected_trajectory = candidates[selected_idx]
        selected_action = selected_trajectory.actions[:, 0, :]

        log_probs = F.log_softmax(-scores / temperature, dim=0)
        log_prob = log_probs[selected_idx]

        if committed:
            self._committed_trajectory = selected_trajectory
        # Always store for rv updates (ARC-016 deadlock fix)
        self._last_selected_trajectory = selected_trajectory

        # ARC-108 JOB-1 step-1: record the Hebbian co-activation eligibility trace
        # (how much each channel "spoke for" the committed action this tick) so the
        # next post_action_update can credit the channels that voted for the realised
        # outcome. eligibility_c = |channel_bias_c[selected]|, accumulated into a
        # decayed last-K-ticks trace. Recorded ONLY on the waking selection path
        # (simulation_mode False): a replay/DMN tick records no eligibility, so it
        # forms no delta_t and writes no w_chan (MECH-094). Setting _lcg_pending arms
        # the three-factor update for the next post_action_update.
        # MECH-451: eligibility recording rides the active registry/buffer too. The
        # finer path writes _fcg_elig_trace + arms _fcg_pending; the ARC-108 path
        # writes _lcg_elig_trace + arms _lcg_pending (unchanged). Both decayed
        # |channel_bias_c[selected]| Hebbian co-activation; both waking-only (MECH-094).
        _elig_on = (
            (getattr(self.config, "use_learned_channel_gating", False) and not _fcg)
            or (_fcg and getattr(self.config, "use_finer_channel_gating", False))
        )
        if _elig_on and not simulation_mode and _lcg_terms:
            decay = float(getattr(self.config, "learned_channel_gating_elig_decay", 0.9))
            _elig_buf = self._fcg_elig_trace if _fcg else self._lcg_elig_trace
            contribution = _elig_buf.new_zeros(_elig_buf.shape)
            # MECH-452: loop-local eligibility traces. When on (with loop segregation),
            # credit a channel ONLY if its loop's within-loop winner matched the
            # committed action (the loop "voted for" the realised outcome), so the
            # shared signed-RPE delta_t stays loop-local. Default off -> every voting
            # channel is credited (bit-identical to ARC-108/MECH-451).
            _loop_local = bool(
                getattr(self.config, "use_loop_local_eligibility_traces", False)
                and getattr(self.config, "use_loop_segregation", False)
                and self._loop_of_channel
            )
            _n_credited = 0
            for ch_idx, term in _lcg_terms:
                if _loop_local:
                    _loop = self._loop_of_channel.get(ch_idx, "motor")
                    if not self._loop_voted.get(_loop, True):
                        continue
                contribution[ch_idx] = contribution[ch_idx] + term[selected_idx].detach().abs().to(
                    dtype=contribution.dtype, device=contribution.device
                )
                _n_credited += 1
            if _loop_local:
                self.last_score_diagnostics["loop_local_credited_channels"] = _n_credited
            if _fcg:
                self._fcg_elig_trace = decay * self._fcg_elig_trace + contribution
                self._fcg_pending = True
            else:
                self._lcg_elig_trace = decay * self._lcg_elig_trace + contribution
                self._lcg_pending = True

        return SelectionResult(
            selected_trajectory=selected_trajectory,
            selected_index=selected_idx,
            selected_action=selected_action,
            scores=scores,
            precision=self.current_precision,
            committed=committed,
            log_prob=log_prob,
            urgency=urgency_applied,
        )

    # ------------------------------------------------------------------ #
    # Post-action update                                                   #
    # ------------------------------------------------------------------ #

    def post_action_update(
        self,
        actual_z_world: torch.Tensor,
        harm_occurred: bool,
    ) -> Dict[str, torch.Tensor]:
        """
        Update E3 after action execution (ARC-016 dynamic precision update).

        Args:
            actual_z_world: Observed z_world after action [batch, world_dim]
            harm_occurred:  Whether harm occurred (for residue gating)

        Returns:
            Dictionary of update metrics
        """
        metrics: Dict[str, torch.Tensor] = {}

        # ARC-016: update running variance from any selected trajectory, not
        # just committed ones.  Previous code gated on _committed_trajectory,
        # creating a deadlock: rv starts at precision_init (0.5), above
        # commit_threshold (0.40), so the agent never commits, rv never
        # updates, and the agent can never commit.
        ref_trajectory = self._committed_trajectory or self._last_selected_trajectory
        if ref_trajectory is not None and ref_trajectory.world_states is not None:
            predicted_world = ref_trajectory.world_states[1]
            prediction_error = actual_z_world - predicted_world

            self.update_running_variance(prediction_error)

            metrics["prediction_error"] = prediction_error.pow(2).mean()
            metrics["running_variance"] = torch.tensor(self._running_variance)
            metrics["dynamic_precision"] = torch.tensor(self.current_precision)

        # Residue accumulation stays commitment-gated (only accumulate for
        # actions the agent was committed to)
        if self._committed_trajectory is not None and harm_occurred and self.residue_field is not None:
            self.residue_field.accumulate(
                actual_z_world, harm_magnitude=1.0, hypothesis_tag=False
            )
            metrics["residue_updated"] = torch.tensor(1.0)

        # ARC-108 JOB-1: ONE signed-RPE delta_t drives BOTH the learned per-channel
        # selection weights w_chan (step 1) AND the learned lateral-inhibition matrix
        # W_lat (MECH-450 step 2). On the waking committed-selection path only (each
        # gated on a pending trace recorded by a non-simulation select() this tick); a
        # replay/DMN tick leaves both pending flags False -> no delta_t, no write
        # (MECH-094). delta_t / V-hat_t / the D1-D2 asym are SHARED so the two learned
        # objects ride a single dopaminergic RPE.
        _lcg_on = bool(
            getattr(self.config, "use_learned_channel_gating", False)
            and self._lcg_pending
        )
        # MECH-451: the finer w_chan_finer three-factor update -- the SAME signed-RPE
        # delta_t / V-hat_t / D1-D2 asym as the ARC-108 path, applied over the finer
        # eligibility trace. Gated on a pending WAKING finer trace (a replay/DMN tick
        # leaves _fcg_pending False -> no write; MECH-094).
        _fcg_on = bool(
            getattr(self.config, "use_finer_channel_gating", False)
            and self._fcg_pending
        )
        _wlat_on = bool(
            getattr(self.config, "use_learned_settling_step", False)
            and self._wlat_pending
        )
        # ARC-108 JOB-2 (control plane): the habenula negative-delta_t de-commit
        # REUSES the SAME signed-RPE delta_t / V-hat_t this block forms. When
        # use_habenula_decommit is on, compute delta_t + advance the shared V-hat_t
        # on EVERY waking post-action (NOT gated on a JOB-1 eligibility trace -- the
        # de-commit must read the realised outcome whenever the agent acts) and emit
        # it so REEAgent.update_residue can route a negative delta_t into the SD-034
        # ClosureOperator's habenula abort. No w_chan / W_lat write on this path.
        # Bit-identical when use_habenula_decommit is False (condition unchanged).
        _hab_on = bool(getattr(self.config, "use_habenula_decommit", False))
        if _lcg_on or _fcg_on or _wlat_on or _hab_on:
            with torch.no_grad():
                # R_t = realised outcome valence at the resulting state, from the
                # ALREADY-TRAINED valuation heads (reuse; no new encoder, no phased
                # training): benefit (Go) minus harm (No-Go). Detached -- the
                # three-factor rule is a LOCAL update, not backprop into the heads.
                zw = actual_z_world.detach()
                benefit = self.benefit_eval_head(zw).mean()
                harm = self.harm_eval_head(zw).mean()
                R_t = float((benefit - harm).item())
                # delta_t = SIGNED dopaminergic-RPE analog = R_t - V-hat_t. This is
                # explicitly NOT the unsigned ARC-016 prediction-error variance
                # (_running_variance), which is kept separate (divergence B5): an
                # unsigned magnitude cannot supply the directional credit a learned
                # gate needs. V-hat_t (the slow EMA "expected" term) is read BEFORE it
                # is updated toward R_t.
                v_hat = self._lcg_value_baseline
                delta_t = R_t - v_hat
                # ARC-108 sec-7 C3 (divergence B5): the LEARNING teaching signal that
                # drives BOTH w_chan and W_lat. learned_channel_rpe_mode "signed"
                # (default) uses the SIGNED delta_t -- bit-identical to the original
                # substrate. "unsigned" substitutes the UNSIGNED ARC-016 prediction-error
                # magnitude (self._running_variance, always >= 0), removing the
                # directional potentiate-vs-depress credit a learned gate needs (the
                # B5 ablation that MUST fail to convert; if it converts the mechanism
                # collapses to a precision re-weighting -> route back to ARC-016). The
                # signed delta_t itself is kept intact for the JOB-2 habenula de-commit
                # and the V-hat_t baseline EMA below.
                if getattr(self.config, "learned_channel_rpe_mode", "signed") == "unsigned":
                    learn_signal = abs(float(self._running_variance))
                else:
                    learn_signal = delta_t
                # asym renders the D1-LTP / D2-LTD asymmetry as a single asymmetric
                # gain on the LEARNING update: a positive teaching signal potentiates
                # the voting channels faster than a negative one depresses them. ARC-110
                # adds the SELECTION-layer D1/D2 opponent-population split
                # (use_d1_d2_population_split; _segregated_loop_arbitrate /
                # _d1_d2_split), which is where the representational two-population
                # distinction (conflict-vs-indifference) the additive scalar destroys is
                # earned; this learning-update asymmetry is its credit-side partner.
                # Under the unsigned ablation learn_signal >= 0 always, so asym is fixed
                # at potentiation -- the structural reason an unsigned signal CANNOT
                # produce opposite-sign w_chan moves (the C3/C7 contract).
                learn_asym = (
                    float(getattr(self.config, "learned_channel_asym_potentiation", 1.0))
                    if learn_signal >= 0.0
                    else float(getattr(self.config, "learned_channel_asym_depression", 0.5))
                )
                if _lcg_on:
                    eta = float(getattr(self.config, "learned_channel_gating_eta", 0.01))
                    # Delta w[c] = eta * learn_signal * eligibility_c * asym
                    self.w_chan.add_(
                        eta * learn_signal * learn_asym * self._lcg_elig_trace.to(
                            dtype=self.w_chan.dtype, device=self.w_chan.device
                        )
                    )
                    self._lcg_last_delta = learn_signal
                    self._lcg_n_updates += 1
                if _fcg_on:
                    # MECH-451: SAME three-factor rule, applied to the finer
                    # w_chan_finer over the finer eligibility trace.
                    # Delta w_finer[c] = eta * learn_signal * eligibility_c * asym.
                    eta_f = float(getattr(self.config, "learned_channel_gating_eta", 0.01))
                    self.w_chan_finer.add_(
                        eta_f * learn_signal * learn_asym * self._fcg_elig_trace.to(
                            dtype=self.w_chan_finer.dtype, device=self.w_chan_finer.device
                        )
                    )
                    self._fcg_last_delta = learn_signal
                    self._fcg_n_updates += 1
                if _wlat_on:
                    # MECH-450: SAME three-factor rule, applied to W_lat over the
                    # decayed Hebbian class co-activation trace recorded during the
                    # settling step.  Delta W_lat[i,j] = eta_w * learn_signal * coact[i,j] * asym.
                    eta_w = float(getattr(self.config, "learned_settling_eta", 0.01))
                    self.W_lat.add_(
                        eta_w * learn_signal * learn_asym * self._wlat_coact_trace.to(
                            dtype=self.W_lat.dtype, device=self.W_lat.device
                        )
                    )
                    self._wlat_last_delta = learn_signal
                    self._wlat_n_updates += 1
                # Update the slow value baseline toward the realised R_t (shared; once).
                beta = float(
                    getattr(self.config, "learned_channel_value_baseline_beta", 0.05)
                )
                self._lcg_value_baseline = (1.0 - beta) * v_hat + beta * R_t
            if _lcg_on:
                self._lcg_pending = False
                metrics["lcg_delta_t"] = torch.tensor(self._lcg_last_delta)
                metrics["lcg_value_baseline"] = torch.tensor(self._lcg_value_baseline)
                metrics["lcg_w_chan_range"] = torch.tensor(
                    float((self.w_chan.max() - self.w_chan.min()).item())
                )
            if _fcg_on:
                self._fcg_pending = False
                # MECH-451 readiness/diagnostics: the cross-channel learned-weight
                # RANGE is the EXP-0391 non-degeneracy signal -- finer channels that
                # move IDENTICALLY (range ~0) are the compressed blend re-labelled.
                metrics["fcg_delta_t"] = torch.tensor(self._fcg_last_delta)
                metrics["fcg_value_baseline"] = torch.tensor(self._lcg_value_baseline)
                metrics["fcg_w_chan_finer_range"] = torch.tensor(
                    float((self.w_chan_finer.max() - self.w_chan_finer.min()).item())
                )
                metrics["fcg_w_chan_finer_std"] = torch.tensor(
                    float(self.w_chan_finer.std().item())
                )
            if _wlat_on:
                self._wlat_pending = False
                metrics["wlat_delta_t"] = torch.tensor(self._wlat_last_delta)
                metrics["wlat_range"] = torch.tensor(
                    float((self.W_lat.max() - self.W_lat.min()).item())
                )
            if _hab_on:
                # ARC-108 JOB-2: surface the signed RPE so the habenula de-commit
                # (REEAgent.update_residue) can fire the SD-034 abort on a negative
                # ("worse than expected") delta_t. delta_t is in scope here whenever
                # the block ran. No-op metric when use_habenula_decommit is False.
                metrics["habenula_delta_t"] = torch.tensor(delta_t)
                metrics["habenula_value_baseline"] = torch.tensor(
                    self._lcg_value_baseline
                )

        self._committed_trajectory = None
        return metrics

    def clear_learned_channel_eligibility(self) -> None:
        """ARC-108 JOB-1: per-episode clear of the within-episode credit window for
        BOTH learned objects (w_chan step 1 + MECH-450 W_lat step 2). The learned
        state (w_chan, W_lat, V-hat_t) PERSISTS across episodes -- only the Hebbian
        traces + the pending-update flags are cleared so a dangling commit at an
        episode boundary does not credit the first outcome of the next episode. No-op
        (cheap zero) when learned gating / the settling step is off."""
        self._lcg_elig_trace = torch.zeros_like(self._lcg_elig_trace)
        self._lcg_pending = False
        # MECH-451: clear the finer eligibility trace + pending flag too (w_chan_finer
        # persists across episodes, parallel to w_chan).
        self._fcg_elig_trace = torch.zeros_like(self._fcg_elig_trace)
        self._fcg_pending = False
        self._wlat_coact_trace = torch.zeros_like(self._wlat_coact_trace)
        self._wlat_pending = False

    def get_commitment_state(self) -> Dict[str, float]:
        return {
            "precision": self.current_precision,
            "running_variance": self._running_variance,
            "commit_threshold": self.commit_threshold,
            "committed_now": self._running_variance < self.commit_threshold,
            # rung-6 amend: is_committed reflects a closure-FORMED commit too (honest
            # telemetry on the closure-exclusive eval where _committed_trajectory stays
            # None); bit-identical when the trajectory latch is unused (None).
            "is_committed": (
                self._committed_trajectory is not None
                or self._closure_committed_trajectory is not None
            ),
        }

    def forward(
        self,
        candidates: List[Trajectory],
        temperature: float = 1.0,
        goal_state: Optional[GoalState] = None,
        harm_bridge: Optional["nn.Module"] = None,
        use_harm_variance_commit: bool = False,
        terrain_weight: Optional[torch.Tensor] = None,
    ) -> SelectionResult:
        return self.select(
            candidates,
            temperature,
            goal_state=goal_state,
            harm_bridge=harm_bridge,
            use_harm_variance_commit=use_harm_variance_commit,
            terrain_weight=terrain_weight,
        )
