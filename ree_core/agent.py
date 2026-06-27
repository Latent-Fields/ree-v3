"""
REE-v3 Agent Implementation

Integrates all V3 architectural components:
  - LatentStack (SD-005 split: z_self + z_world)
  - E1 DeepPredictor (associative prior over z_self + z_world)
  - E2 FastPredictor (SD-004: action objects; SD-005: world_forward)
  - E3 TrajectorySelector (harm_eval + dynamic precision)
  - HippocampalModule (action-object space navigation, MECH-092 replay)
  - ResidueField (z_world substrate, MECH-094 hypothesis_tag gate)
  - MultiRateClock (SD-006 phase 1: time-multiplexed multi-rate)
  - ThetaBuffer (MECH-089 cross-rate integration)
  - BetaGate (MECH-090 policy propagation gate)

V3 REE loop:
  1. SENSE     — split observation into body/world channels
  2. E1 TICK   — update E1 (every step); push z_self, z_world to ThetaBuffer
  3. E2 TICK   — update z_self via E2 motor-sensory model (every N_e2 steps)
  4. E3 TICK   — E3 consumes theta-cycle summary; proposes via HippocampalModule
                 (every N_e3 steps, or on phase reset)
  5. SELECT    — E3 selects trajectory; BetaGate controls policy propagation
  6. ACT       — execute first action from selected trajectory
  7. RESIDUE   — accumulate world_delta into ResidueField (MECH-094 gated)
  8. REPLAY    — quiescent E3 cycles trigger HippocampalModule.replay() (MECH-092)
  9. OFFLINE   — periodic sleep-like residue integration

Multi-rate execution:
  The clock fires e1_tick, e2_tick, e3_tick based on configured rates.
  E3 only fires at its tick; between ticks, the held action from the last
  E3 selection is repeated (MECH-057a: action-loop gate).
"""

from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
from collections import Counter, deque
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ree_core.utils.config import REEConfig
from ree_core.latent.stack import LatentStack, LatentState
from ree_core.goal import GoalState, SuperOrdinalGoalMemory
from ree_core.latent.theta_buffer import ThetaBuffer
from ree_core.latent.multi_content_theta_packet import (
    MultiContentThetaPacket,
    MultiContentThetaPacketConfig,
)
from ree_core.predictors.e1_deep import E1DeepPredictor
from ree_core.predictors.e2_fast import E2FastPredictor, Trajectory
from ree_core.predictors.e3_score_diversity import (
    E3ScoreDiversity,
    build_from_ree_config as build_e3_score_diversity_from_ree_config,
)
from ree_core.predictors.e3_selector import E3TrajectorySelector, project_channel_range
from ree_core.residue.field import ResidueField
from ree_core.hippocampal.ghost_goal_bank import PersistenceAppraisal
from ree_core.hippocampal.module import HippocampalModule
from ree_core.hippocampal.persistence_appraisal_compute import (
    compute_agent_persistence_appraisal,
)
from ree_core.heartbeat.clock import MultiRateClock
from ree_core.heartbeat.beta_gate import BetaGate
from ree_core.neuromodulation.serotonin import SerotoninModule
from ree_core.predictors.e2_harm_a import E2HarmAConfig, E2HarmAForward
from ree_core.predictors.e2_harm_s import E2HarmSConfig, E2HarmSForward
from ree_core.predictors.e2_world import E2WorldConfig, E2WorldForward
from ree_core.policy.model_disagreement import (
    ModelDisagreementConfig,
    ModelDisagreementEnsemble,
)
from ree_core.cingulate import (
    AICAnalog,
    AICConfig,
    DACCAdaptiveControl,
    DACCConfig,
    DACCtoE3Adapter,
    PACCAnalog,
    PACCConfig,
    PCCAnalog,
    PCCConfig,
    SalienceCoordinator,
    SalienceCoordinatorConfig,
    StuckStateDetector,
    StuckStateDetectorConfig,
)
from ree_core.latent.stack import HarmForwardTrunk
from ree_core.pfc import LateralPFCAnalog, OFCAnalog
from ree_core.pfc.lateral_pfc_analog import LateralPFCConfig
from ree_core.pfc.ofc_analog import OFCConfig
from ree_core.pfc.infralimbic_avoidance_gate import (
    InstrumentalAvoidanceGate,
    InstrumentalAvoidanceGateConfig,
)
from ree_core.pfc.escape_affordance_bridge import (
    EscapeAffordanceBridge,
    EscapeAffordanceBridgeConfig,
)
from ree_core.entities.object_file_buffer import (
    EntityObservation,
    ObjectFileBuffer,
    ObjectFileBufferConfig,
)
from ree_core.pfc.trainable_escape_affordance_learner import (
    TrainableEscapeAffordanceLearner,
    TrainableEscapeAffordanceLearnerConfig,
)
from ree_core.pfc.e2_escape_affordance_linker import (
    E2EscapeAffordanceLinker,
    E2EscapeAffordanceLinkerConfig,
)
from ree_core.policy import (
    CandidateRuleField,
    CandidateRuleFieldConfig,
    CommitMaintenanceRelease,
    CommitMaintenanceReleaseConfig,
    CommitReadiness,
    CommitReadinessConfig,
    NaturalCommitUrgencyRelease,
    NaturalCommitUrgencyReleaseConfig,
    RhoMaintenanceRamp,
    RhoMaintenanceRampConfig,
    DifficultyGatedProposalEntropy,
    DifficultyGatedProposalEntropyConfig,
    GatedPolicy,
    GatedPolicyConfig,
    NoiseFloor,
    NoiseFloorConfig,
    StructuredCuriosity,
    StructuredCuriosityConfig,
    TonicVigor,
    TonicVigorConfig,
)
from ree_core.governance import (
    ClosureEvent,
    ClosureOperator,
    ClosureOperatorConfig,
)
from ree_core.amygdala import (
    BLAAnalog,
    BLAConfig,
    BLAOutput,
    CeAAnalog,
    CeAConfig,
    CeAOutput,
)
from ree_core.comparator.tpj_comparator import TPJComparator
from ree_core.affect.blocked_agency import BlockedAgency, BlockedAgencyConfig
from ree_core.affect.harm_suffering_accumulator import (
    HarmSufferingAccumulator,
    HarmSufferingAccumulatorConfig,
)
from ree_core.attribution.scientist_attribution_buffer import (
    ScientistAttributionBuffer,
    ScientistAttributionConfig,
)
from ree_core.regulators import (
    GABAergicDecayConfig,
    GABAergicDecayRegulator,
    BroadcastOverrideConfig,
    BroadcastOverrideRegulator,
    LPBInteroceptiveRouter,
    LPBInteroceptiveRoutingConfig,
    LPBInteroceptiveRoutingOutput,
    MECH295LikingBridge,
    MECH295LikingBridgeConfig,
    SimulationModeRuleGate,
    SimulationModeRuleGateConfig,
    SITE_GATED_POLICY,
    SITE_LATERAL_PFC,
)
from ree_core.comparator.suffering_derivative_comparator import (
    SufferingDerivativeComparator,
)
from ree_core.safety import ConditionedSafetyStore
from ree_core.pag import (
    PAGFreezeGate,
    PAGFreezeGateConfig,
    PAGFreezeGateOutput,
)
from ree_core.sleep import SleepLoopManager
from ree_core.residue.field import (
    VALENCE_WANTING,
    VALENCE_LIKING,
    VALENCE_HARM_DISCRIMINATIVE,
    VALENCE_SURPRISE,
    VALENCE_POSITIVE_SURPRISE,
    VALENCE_NEGATIVE_SURPRISE,
)


def candidate_support_preflight(
    candidates: List[Trajectory],
    forced_score_bias_per_class: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Compute first-action support metrics for behavioural interpretation.

    This is diagnostic-only: callers decide whether to abort interpretation.
    """
    first_action_classes: List[int] = []
    for candidate in candidates:
        if candidate.actions.shape[1] <= 0:
            continue
        first = candidate.actions[:, 0, :]
        first_action_classes.append(
            int(first.argmax(dim=-1).flatten()[0].item())
        )

    counts = Counter(first_action_classes)
    total = sum(counts.values())
    entropy = 0.0
    if total > 0:
        for count in counts.values():
            p = float(count) / float(total)
            if p > 0.0:
                entropy -= p * math.log(p)

    forced_bias_abs_mean: Optional[float] = None
    forced_bias_nonzero_candidate_count: Optional[int] = None
    if forced_score_bias_per_class is not None:
        abs_vals: List[float] = []
        nonzero = 0
        for cls in first_action_classes:
            bias = (
                float(forced_score_bias_per_class[cls])
                if cls < len(forced_score_bias_per_class)
                else 0.0
            )
            if abs(bias) > 0.0:
                nonzero += 1
            abs_vals.append(abs(bias))
        forced_bias_abs_mean = (
            sum(abs_vals) / len(abs_vals) if abs_vals else 0.0
        )
        forced_bias_nonzero_candidate_count = nonzero

    status = "RUN" if len(counts) >= 2 else "NOT_RUN"
    reason = None if status == "RUN" else "candidate_support_collapse"
    return {
        "preflight_status": status,
        "not_run_reason": reason,
        "interpretation": (
            "RUN" if status == "RUN" else "NOT_RUN: candidate_support_collapse"
        ),
        "candidate_first_action_classes": first_action_classes,
        "candidate_first_action_counts": dict(sorted(counts.items())),
        "candidate_unique_first_action_classes": int(len(counts)),
        "candidate_first_action_entropy": float(entropy),
        "forced_bias_abs_mean": forced_bias_abs_mean,
        "forced_bias_nonzero_candidate_count": forced_bias_nonzero_candidate_count,
    }


@dataclass
class AgentState:
    """Complete state of the REE-v3 agent at a timestep."""
    latent_state: LatentState
    precision: float
    running_variance: float
    step: int
    harm_accumulated: float
    is_committed: bool
    beta_elevated: bool
    e3_steps_per_tick: int
    # MECH-203/204: serotonergic state (None when disabled)
    serotonin_state: Optional[Dict[str, Any]] = None


class REEAgent(nn.Module):
    """
    Reflective-Ethical Engine Agent — V3.

    Implements all V3 architectural invariants:
    1. Ethical cost is PERSISTENT (residue cannot be erased)
    2. Residue tracks WORLD-DELTA (z_world changes), not body-state changes
    3. E3 is the harm evaluator (harm_eval on z_world)
    4. E2 trains on motor-sensory error (z_self). NOT harm/goal error.
    5. HippocampalModule navigates action-object space O (SD-004)
    6. All replay/simulation carries hypothesis_tag=True (MECH-094)
    7. Precision is E3-derived (ARC-016). NOT hardcoded.
    8. E3 consumes theta-cycle summaries, never raw E1 output (MECH-089)
    9. Beta state gates policy propagation, not E3 internal updating (MECH-090)
    """

    def __init__(self, config: REEConfig):
        super().__init__()
        self.config = config

        # Core components
        self.latent_stack = LatentStack(config.latent)
        self.e1 = E1DeepPredictor(config.e1)
        self.e2 = E2FastPredictor(config.e2)
        self.residue_field = ResidueField(config.residue)
        self.e3 = E3TrajectorySelector(config.e3, self.residue_field)
        self.hippocampal = HippocampalModule(config.hippocampal, self.e2, self.residue_field)

        # SD-006: multi-rate clock
        self.clock = MultiRateClock(
            e1_steps_per_tick=config.heartbeat.e1_steps_per_tick,
            e2_steps_per_tick=config.heartbeat.e2_steps_per_tick,
            e3_steps_per_tick=config.heartbeat.e3_steps_per_tick,
            theta_buffer_size=config.heartbeat.theta_buffer_size,
            beta_rate_min_steps=config.heartbeat.beta_rate_min_steps,
            beta_rate_max_steps=config.heartbeat.beta_rate_max_steps,
            beta_magnitude_scale=config.heartbeat.beta_magnitude_scale,
            breath_period=config.heartbeat.breath_period,
            sweep_amplitude=config.heartbeat.breath_sweep_amplitude,
            sweep_duration=config.heartbeat.breath_sweep_duration,
        )

        # MECH-089: ThetaBuffer for cross-rate integration
        self.theta_buffer = ThetaBuffer(
            self_dim=config.latent.self_dim,
            world_dim=config.latent.world_dim,
            buffer_size=config.heartbeat.theta_buffer_size,
        )

        # MECH-294: multi-content theta-burst packet (sibling to MECH-089).
        # Binds a joint {goal_latent, action_proposal, risk_estimate,
        # state_summary} tuple within one theta cycle. Composes the ThetaBuffer
        # above for its state_summary slot (MECH-089 untouched). Default OFF ->
        # self.multi_content_theta_packet is None and the agent is bit-identical
        # to pre-MECH-294. Requires use_per_stream_vs (the packet consumes the
        # MECH-269 per-stream V_s for vintaging) -- loud precondition, same
        # pattern as MECH-269b / MECH-293.
        self.multi_content_theta_packet = None
        self.last_theta_packet = None
        if getattr(config, "use_multi_content_theta_packet", False):
            if not getattr(config.hippocampal, "use_per_stream_vs", False):
                raise ValueError(
                    "use_multi_content_theta_packet=True requires "
                    "use_per_stream_vs=True (the packet consumes the MECH-269 "
                    "per-stream V_s for snapshot-or-hold vintaging)."
                )
            self.multi_content_theta_packet = MultiContentThetaPacket(
                MultiContentThetaPacketConfig(
                    binding_mode=getattr(config, "theta_packet_binding_mode", "joint"),
                    snapshot_refresh_threshold=getattr(
                        config, "theta_packet_snapshot_refresh_threshold", 0.5),
                    hold_threshold=getattr(config, "theta_packet_hold_threshold", 0.4),
                    coherence_hold_weight=getattr(
                        config, "theta_packet_coherence_hold_weight", 0.5),
                )
            )

        # MECH-090: BetaGate for policy propagation. MECH-090 R-c commit-entry
        # readiness conjunction knobs are read from HeartbeatConfig and forwarded
        # to the gate so should_admit_elevation() can implement the predicate
        # (default no-op when use_commit_readiness_gate=False).
        # See HeartbeatConfig in ree_core/utils/config.py and
        # REE_assembly/docs/architecture/mech_090_commit_entry_predicate.md.
        self.beta_gate = BetaGate(
            use_commit_readiness_gate=getattr(
                config.heartbeat, "use_commit_readiness_gate", False
            ),
            commit_readiness_floor=getattr(
                config.heartbeat, "commit_readiness_floor", 0.05
            ),
            commit_readiness_strict_single_candidate=getattr(
                config.heartbeat,
                "commit_readiness_strict_single_candidate",
                False,
            ),
        )

        # MECH-203/204: serotonergic neuromodulation (sleep + benefit-salience)
        self.serotonin = SerotoninModule(config.serotonin)

        # MECH-258: E2_harm_a affective-pain forward model (SD-032b prerequisite).
        # ARC-058 shared-trunk path: if use_shared_harm_trunk, construct a single
        # HarmForwardTrunk and pass into E2HarmAForward. (E2HarmSForward in its
        # current form owns its trunk internally; the competing-claim experiment
        # builds E2HarmSForward via the same trunk at the experiment script level.)
        self.harm_forward_trunk: Optional[HarmForwardTrunk] = None
        self.e2_harm_a: Optional[E2HarmAForward] = None
        if getattr(config, "use_e2_harm_a", False):
            z_harm_a_dim = config.latent.z_harm_a_dim
            action_dim = config.e2.action_dim
            harm_a_cfg = E2HarmAConfig(
                z_harm_a_dim=z_harm_a_dim,
                action_dim=action_dim,
                learning_rate=config.e2_harm_a_lr,
            )
            shared_trunk = None
            if getattr(config, "use_shared_harm_trunk", False):
                # ARC-058: construct shared trunk. Uses z_harm_a_dim; experiments
                # that run the sensory stream through the same trunk must match
                # this dim via projection heads (see experiment scripts).
                self.harm_forward_trunk = HarmForwardTrunk(
                    z_harm_dim=z_harm_a_dim,
                    action_dim=action_dim,
                    hidden_dim=harm_a_cfg.hidden_dim,
                    action_enc_dim=harm_a_cfg.action_enc_dim,
                )
                shared_trunk = self.harm_forward_trunk
            self.e2_harm_a = E2HarmAForward(harm_a_cfg, shared_trunk=shared_trunk)

        # ARC-033: E2_harm_s sensory-discriminative harm forward model.
        # Constructed at the agent level when use_e2_harm_s_forward is on
        # (LatentStackConfig). Required by Phase E (MECH-273) self-model
        # writeback. Backward compatible: None when the flag is off.
        self.e2_harm_s: Optional[E2HarmSForward] = None
        if getattr(config.latent, "use_e2_harm_s_forward", False):
            harm_s_cfg = E2HarmSConfig(
                use_e2_harm_s_forward=True,
                z_harm_dim=config.latent.z_harm_dim,
                action_dim=config.e2.action_dim,
            )
            self.e2_harm_s = E2HarmSForward(harm_s_cfg)

        # SD-031: E2_world causal-footprint forward model + single-pass
        # comparator (the z_world instantiation of MECH-256; sibling to
        # ARC-033 on z_harm_s). Constructed at the agent level when
        # use_e2_world_forward is on (LatentStackConfig). z_world_dim is read
        # from config.latent.world_dim -- NEVER a literal -- and E2WorldForward
        # hard-asserts world_dim >= 128 (the discriminative-granularity guard
        # from failure_autopsy_zworld-integration-cluster_2026-06-06). Backward
        # compatible: None when the flag is off.
        self.e2_world: Optional[E2WorldForward] = None
        if getattr(config.latent, "use_e2_world_forward", False):
            world_cfg = E2WorldConfig(
                use_e2_world_forward=True,
                z_world_dim=config.latent.world_dim,
                action_dim=config.e2.action_dim,
            )
            self.e2_world = E2WorldForward(world_cfg)

        # MECH-441: model-disagreement directed curiosity ensemble (RND / Plan2Explore
        # analog). A standalone K-head forward-model ensemble over (z_world, action);
        # per-candidate cross-head variance feeds E3 selection as a propagating
        # curiosity bonus. Built only when the E3-side lever is on AND
        # n_disagreement_heads >= 2; None otherwise -> bit-identical OFF. z_world_dim /
        # action_dim from config (NEVER literals). The phased-training driver trains
        # the heads via disagreement_ensemble.train_step(); the waking read is no_grad.
        self.disagreement_ensemble: Optional[ModelDisagreementEnsemble] = None
        _n_dis_heads = int(getattr(config.latent, "n_disagreement_heads", 0))
        if (
            getattr(config.e3, "use_model_disagreement_curiosity", False)
            and _n_dis_heads >= 2
        ):
            dis_cfg = ModelDisagreementConfig(
                world_dim=config.latent.world_dim,
                action_dim=config.e2.action_dim,
                n_heads=_n_dis_heads,
                bootstrap_mask_prob=float(
                    getattr(config.latent, "disagreement_bootstrap_mask_prob", 0.0)
                ),
                learning_rate=float(
                    getattr(config.latent, "disagreement_learning_rate", 3e-4)
                ),
            )
            self.disagreement_ensemble = ModelDisagreementEnsemble(dis_cfg)

        # SD-032b: dACC/aMCC-analog adaptive control.
        self.dacc: Optional[DACCAdaptiveControl] = None
        self.dacc_adapter: Optional[DACCtoE3Adapter] = None
        if getattr(config, "use_dacc", False):
            dacc_cfg = DACCConfig(
                dacc_weight=config.dacc_weight,
                dacc_interaction_weight=config.dacc_interaction_weight,
                dacc_foraging_weight=config.dacc_foraging_weight,
                dacc_suppression_weight=config.dacc_suppression_weight,
                dacc_suppression_memory=config.dacc_suppression_memory,
                dacc_precision_scale=config.dacc_precision_scale,
                dacc_effort_cost=config.dacc_effort_cost,
                dacc_drive_coupling=config.dacc_drive_coupling,
                dacc_bias_max_abs=getattr(config, "dacc_bias_max_abs", 0.0),
                dacc_goal_readout_weight=getattr(
                    config, "dacc_goal_readout_weight", 0.0
                ),  # SD-057 L7 (MECH-348)
            )
            self.dacc = DACCAdaptiveControl(dacc_cfg)
            # STOPGAP adapter -- still the score_bias source until SD-033
            # substrates consume operating_mode natively. With SD-032a active,
            # the adapter's bias may be scaled by the coordinator's e3_policy
            # write-gate (see select_action; gated by salience_apply_to_dacc_bias).
            self.dacc_adapter = DACCtoE3Adapter(dacc_cfg)

        # SD-057 phase-2 preconditions (loud-not-silent).
        if getattr(config, "use_mech_consume", False) and not getattr(
            config, "use_dacc", False
        ):
            raise ValueError(
                "use_mech_consume=True (SD-057 L7) requires use_dacc=True -- the "
                "object-discriminative goal readout rides the dACC bundle."
            )
        _goal_cfg = getattr(config, "goal", None)
        if (
            _goal_cfg is not None
            and getattr(_goal_cfg, "use_cue_recall", False)
            and not getattr(_goal_cfg, "use_incentive_token_bank", False)
        ):
            raise ValueError(
                "use_cue_recall=True (SD-057 L6) requires use_incentive_token_bank=True "
                "-- the cue-recall path reads per-object tokens from the bank."
            )

        # SD-032a: salience-network coordinator. Reads SD-032b dACC bundle +
        # drive_level + offline-mode flag; emits operating_mode soft vector +
        # mode_switch_trigger. Hosts MECH-261 write-gate registry.
        self.salience: Optional[SalienceCoordinator] = None
        if getattr(config, "use_salience_coordinator", False):
            sal_cfg = SalienceCoordinatorConfig(
                external_task_bias=config.salience_external_task_bias,
                softmax_temperature=config.salience_softmax_temperature,
                switch_threshold=config.salience_switch_threshold,
                stability_scaling=config.salience_stability_scaling,
            )
            sal_cfg.salience_weights = {
                "dacc_pe": config.salience_dacc_pe_weight,
                "dacc_foraging": config.salience_dacc_foraging_weight,
                "aic_salience": 1.0,
            }
            self.salience = SalienceCoordinator(sal_cfg)
        # Cache of last coordinator tick output (for diagnostics / experiments).
        self._salience_last_tick: Optional[Dict[str, object]] = None

        # SD-032c: AIC-analog interoceptive-salience / urgency module.
        # Emits aic_salience -> salience coordinator and harm_s_gain ->
        # descending z_harm_s attenuation path (subsumes SD-021 raw beta_gate
        # check). Reads operating_mode from the coordinator's previous tick
        # (one-step lag is biologically plausible; AIC->dACC->SAL is a circuit,
        # not instantaneous).
        self.aic: Optional[AICAnalog] = None
        if getattr(config, "use_aic_analog", False):
            aic_cfg = AICConfig(
                baseline_alpha=config.aic_baseline_alpha,
                drive_coupling=config.aic_drive_coupling,
                urgency_threshold=config.aic_urgency_threshold,
                base_attenuation=config.aic_base_attenuation,
                drive_protect_weight=config.aic_drive_protect_weight,
                extra_weight=config.aic_extra_weight,
            )
            self.aic = AICAnalog(aic_cfg)
        # Cache of last AIC tick output (for diagnostics / experiments).
        self._aic_last_tick: Optional[Dict[str, float]] = None

        # SD-032d: PCC-analog metastability scalar. Emits pcc_stability ->
        # SalienceCoordinator (multiplied into MECH-259 effective threshold).
        # Non-trainable arithmetic over success EMA + drive_level + steps-
        # since-last-offline-phase. Coordinates within-session (MECH-092) and
        # cross-session (INV-049) offline phases via enter_offline_mode().
        self.pcc: Optional[PCCAnalog] = None
        if getattr(config, "use_pcc_analog", False):
            pcc_cfg = PCCConfig(
                success_alpha=config.pcc_success_alpha,
                success_weight=config.pcc_success_weight,
                fatigue_weight=config.pcc_fatigue_weight,
                offline_recency_window=config.pcc_offline_recency_window,
                offline_weight=config.pcc_offline_weight,
                stability_baseline=config.pcc_stability_baseline,
            )
            self.pcc = PCCAnalog(pcc_cfg)
        # Cache of last PCC tick output (for diagnostics / experiments).
        self._pcc_last_tick: Optional[Dict[str, float]] = None

        # SD-032e: pACC-analog slow autonomic write-back. Accumulates
        # tanh-normalised z_harm_a magnitude into a bounded drive_bias that
        # shifts the effective drive_level passed into GoalState.update(),
        # SalienceCoordinator.tick(), SD-032c AIC, and SD-032d PCC. Gated by
        # coordinator.write_gate("autonomic") so the write is mode-conditioned
        # per MECH-261 (active in external_task, attenuated in planning /
        # replay / offline). Architectural path for chronic-pain-like
        # sensitisation (Baliki 2012). See scoping lit-pull synthesis.
        self.pacc: Optional[PACCAnalog] = None
        if getattr(config, "use_pacc_analog", False):
            pacc_cfg = PACCConfig(
                drive_alpha=config.pacc_drive_alpha,
                drive_scale=config.pacc_drive_scale,
                drive_bias_cap=config.pacc_drive_bias_cap,
                z_harm_a_min=config.pacc_z_harm_a_min,
                offline_decay=config.pacc_offline_decay,
            )
            self.pacc = PACCAnalog(pacc_cfg)
        # Cache of last PACC tick output (for diagnostics / experiments).
        self._pacc_last_tick: Optional[Dict[str, float]] = None

        # MECH-302: suffering-derivative comparator substrate.
        # Non-trainable rolling-window descent detector on z_harm_a norm.
        # Fires relief_completion_event, which reuses the MECH-057a commitment
        # release + MECH-094 categorical tag write (VALENCE_LIKING) pipeline.
        # Architecturally adjacent to the MECH-091 urgency block; opposite polarity.
        self.suffering_comparator: Optional[SufferingDerivativeComparator] = None
        if getattr(config, "use_suffering_derivative_comparator", False):
            self.suffering_comparator = SufferingDerivativeComparator(
                window_length=getattr(config, "suffering_window_length", 5),
                drop_threshold=getattr(config, "suffering_drop_threshold", 0.10),
                min_initial_norm=getattr(config, "suffering_min_initial_norm", 0.05),
            )
        # Event flag: set by sense(), consumed and cleared by select_action().
        self._relief_completion_event: bool = False

        # SD-051 / MECH-304: conditioned safety store.
        # Non-trainable EMA prototype of z_world at MECH-302 event ticks.
        # Cosine similarity to current z_world yields safety_prediction.
        # When safety_prediction > threshold and beta_gate elevated: release commitment.
        # Encoding pathway (dorsal striatum / dlPFC analog): EMA prototype.
        # Expression pathway (IL->CeA analog): similarity -> beta_gate.release().
        self.conditioned_safety_store: Optional[ConditionedSafetyStore] = None
        if getattr(config, "use_conditioned_safety_store", False):
            self.conditioned_safety_store = ConditionedSafetyStore(
                world_dim=config.latent.world_dim,
                ema_alpha=getattr(config, "safety_store_ema_alpha", 0.1),
                decay_rate=getattr(config, "safety_store_decay_rate", 0.001),
                min_norm=getattr(config, "safety_store_min_norm", 0.1),
                threshold=getattr(config, "safety_store_threshold", 0.5),
            )
        # Safety prediction: computed by sense(), consumed and cleared by select_action().
        self._conditioned_safety_signal: float = 0.0

        # MECH-095: TPJ agency comparator. Stores an E2 efference-copy
        # prediction for the selected action, then compares it against the
        # next sensed z_self. Runtime output is cached for diagnostics and any
        # experiment loop that wants an ownership signal on resolved
        # transitions.
        self.tpj: Optional[TPJComparator] = None
        if getattr(config, "use_tpj_comparator", False):
            self.tpj = TPJComparator(
                self_dim=config.latent.self_dim,
                agency_threshold=config.tpj_agency_threshold,
            )

        # SD-033a: Lateral-PFC-analog (rule/goal substrate, MECH-261 primary
        # consumer). When use_lateral_pfc_analog=True, maintains a rule_state
        # vector updated via gate-modulated EMA using write_gate("sd_033a")
        # and emits a per-candidate score_bias composed with dACC bias before
        # E3.select(). False = disabled (default, backward compat).
        self.lateral_pfc: Optional[LateralPFCAnalog] = None
        if getattr(config, "use_lateral_pfc_analog", False):
            lpfc_cfg = LateralPFCConfig(
                use_lateral_pfc_analog=True,
                rule_dim=config.lateral_pfc_rule_dim,
                update_eta=config.lateral_pfc_update_eta,
                world_pool_weight=config.lateral_pfc_world_pool_weight,
                bias_scale=config.lateral_pfc_bias_scale,
                hidden_dim=config.lateral_pfc_hidden_dim,
                use_discriminator_source=getattr(
                    config, "lateral_pfc_use_discriminator_source", False
                ),
                discriminator_pool_weight=getattr(
                    config, "lateral_pfc_discriminator_pool_weight", 0.3
                ),
                train_rule_bias_head=getattr(
                    config, "lateral_pfc_train_rule_bias_head", False
                ),
            )
            self.lateral_pfc = LateralPFCAnalog(
                delta_dim=config.latent.delta_dim,
                world_dim=config.latent.world_dim,
                config=lpfc_cfg,
            )

        # ARC-063 v1: distributed CandidateRule field (the non-Bayesian
        # rule-creator resolving arc_062_rule_apprehension:GAP-B). Mints distinct
        # subspace-partitioned rule slots on recurring (context -> action-object
        # -> outcome) regularities; the active-AND-context-matched rules combine
        # into a differentiated rule_state vector that REPLACES the legacy EMA
        # source in LateralPFCAnalog.update (the literal GAP-B fix for the
        # 543/598b rule_state collapse). Precondition: requires the SD-033a
        # consumer (use_lateral_pfc_analog=True). Loud-not-silent (matches the
        # use_closure_operator / MECH-269b / MECH-293 precondition pattern).
        self.candidate_rule_field: Optional[CandidateRuleField] = None
        # Per-tick stash for next-tick credit (the outcome of the rules active on
        # tick t arrives on tick t+1).
        self._crf_prev_action_class: int = -1
        self._crf_prev_outcome: float = 0.0
        if getattr(config, "use_candidate_rule_field", False):
            if self.lateral_pfc is None:
                raise ValueError(
                    "use_candidate_rule_field=True requires "
                    "use_lateral_pfc_analog=True (SD-033a is the rule_state "
                    "consumer the field populates -- ARC-063 GAP-B wiring)."
                )
            crf_cfg = CandidateRuleFieldConfig(
                use_candidate_rule_field=True,
                n_slots=getattr(config, "crf_n_slots", 16),
                rule_dim=getattr(config, "crf_rule_dim", 16),
                mint_recurrence_threshold=getattr(
                    config, "crf_mint_recurrence_threshold", 3
                ),
                tolerance_floor=getattr(config, "crf_tolerance_floor", 0.3),
                tolerance_conflict_gain=getattr(
                    config, "crf_tolerance_conflict_gain", 1.0
                ),
                availability_alpha=getattr(config, "crf_availability_alpha", 0.1),
                availability_decay=getattr(config, "crf_availability_decay", 0.005),
                eligibility_window=getattr(config, "crf_eligibility_window", 20),
                context_match_threshold=getattr(
                    config, "crf_context_match_threshold", 0.5
                ),
                seed_from_arc062=getattr(config, "crf_seed_from_arc062", True),
                persist_rules_across_episode_reset=getattr(
                    config, "crf_persist_rules_across_episode_reset", False
                ),
                # ARC-063 amend (V3-EXQ-654b): mature-pool gate/credit/retire
                # dynamics. All getattr-fallback to the recalibrated default so an
                # absent flat REEConfig attr is bit-identical; consulted only when
                # mature_pool_dynamics is True.
                mature_pool_dynamics=getattr(
                    config, "crf_mature_pool_dynamics", False
                ),
                mature_availability_decay=getattr(
                    config, "crf_mature_availability_decay", 0.001
                ),
                mature_retire_floor=getattr(
                    config, "crf_mature_retire_floor", 0.05
                ),
                mature_availability_alpha_negative=getattr(
                    config, "crf_mature_availability_alpha_negative", 0.02
                ),
                mature_tolerance_floor=getattr(
                    config, "crf_mature_tolerance_floor", 0.15
                ),
                mature_tolerance_conflict_gain=getattr(
                    config, "crf_mature_tolerance_conflict_gain", 0.25
                ),
                mature_mint_block_threshold=getattr(
                    config, "crf_mature_mint_block_threshold", 0.8
                ),
                mature_mint_protection_ticks=getattr(
                    config, "crf_mature_mint_protection_ticks", 30
                ),
                # crf-availability-maintenance (V3-EXQ-666 successor): activity-
                # silent maintenance trace + maintained-pool readout. getattr-
                # fallback to the default so an absent flat REEConfig attr is
                # bit-identical; consulted only when availability_maintenance=True.
                availability_maintenance=getattr(
                    config, "crf_availability_maintenance", False
                ),
                maintenance_floor=getattr(config, "crf_maintenance_floor", 0.45),
                maintenance_decay=getattr(config, "crf_maintenance_decay", 0.0),
                engaged_sustain=getattr(config, "crf_engaged_sustain", False),
                engaged_sustain_rate=getattr(
                    config, "crf_engaged_sustain_rate", 0.1
                ),
                maintained_reactivation_threshold=getattr(
                    config, "crf_maintained_reactivation_threshold", 0.0
                ),
                # CRF conflict-gate calibration amend (V3-EXQ-654d successor;
                # crf-availability-maintenance at the CRF locus). getattr-fallback to
                # the no-op sentinel so an absent flat REEConfig attr is bit-identical;
                # consulted only when mature_pool_dynamics is True.
                mature_context_match_threshold=getattr(
                    config, "crf_mature_context_match_threshold", -1.0
                ),
                tolerance_conflict_cap=getattr(
                    config, "crf_tolerance_conflict_cap", -1
                ),
                maintenance_couple_to_theta=getattr(
                    config, "crf_maintenance_couple_to_theta", False
                ),
            )
            self.candidate_rule_field = CandidateRuleField(
                context_dim=config.latent.world_dim,
                config=crf_cfg,
            )
            # Tell SD-033a to source its rule_state from the field (GAP-B wiring).
            self.lateral_pfc.config.use_candidate_rule_source = True

        # SD-033b: OFC-analog (specific-outcome / task-structure substrate,
        # MECH-261 second consumer). When use_ofc_analog=True, maintains a
        # state_code vector updated via gate-modulated EMA using
        # write_gate("sd_033b") and emits a per-candidate score_bias composed
        # additively with dACC + lateral_pfc bias before E3.select(). Initial
        # bias output is exactly zero (last Linear zeroed) so use_ofc_analog
        # =True with an untrained head is bit-identical to OFF.
        # MECH-094: state-structure persistence is gated by the registry --
        # MECH-261 generalises the hypothesis_tag (sd_033b weight=0.05 under
        # internal_replay).
        self.ofc: Optional[OFCAnalog] = None
        if getattr(config, "use_ofc_analog", False):
            ofc_cfg = OFCConfig(
                use_ofc_analog=True,
                state_dim=config.ofc_state_dim,
                update_eta=config.ofc_update_eta,
                outcome_pool_weight=config.ofc_outcome_pool_weight,
                bias_scale=config.ofc_bias_scale,
                hidden_dim=config.ofc_hidden_dim,
                harm_dim=config.ofc_harm_dim,
                use_outcome_oracle=getattr(config, "use_ofc_outcome_oracle", False),
                train_state_bias_head=getattr(
                    config, "ofc_train_state_bias_head", False
                ),
                use_devaluation_head=getattr(
                    config, "use_ofc_devaluation_head", False
                ),
                devaluation_bias_scale=getattr(
                    config, "ofc_devaluation_bias_scale", 2.0
                ),
                train_devaluation_head=getattr(
                    config, "ofc_train_devaluation_head", False
                ),
            )
            self.ofc = OFCAnalog(
                world_dim=config.latent.world_dim,
                config=ofc_cfg,
            )
        # Per-candidate oracle predictions cache (List[Tensor] or None).
        # Populated each tick when ofc.oracle_is_ready and e2_harm_s are on.
        self._ofc_oracle_predictions: Optional[List[torch.Tensor]] = None

        # ARC-062 Phase 1 (rule-apprehension layer, weak reading): gated-
        # policy heads + learned context discriminator. Substrate for the
        # rule-apprehension architectural slot identified by MECH-309.
        # Default-off (backward compatible). When True, the GatedPolicy
        # module is constructed and its per-candidate score_bias is
        # composed additively into the dACC / lateral_pfc / ofc / mech295
        # chain before E3.select(). NO connection to SD-033a in Phase 1
        # -- that wiring is Phase 3 of arc_062_rule_apprehension_plan.md.
        # See ree_core/policy/gated_policy.py for the module + Pull A
        # SYNTHESIS verdicts behind R1 / R2 / R3 defaults.
        self.gated_policy: Optional[GatedPolicy] = None
        if getattr(config, "use_gated_policy", False):
            gp_cfg = GatedPolicyConfig(
                use_gated_policy=True,
                n_heads=config.gated_policy_n_heads,
                disc_hidden=config.gated_policy_disc_hidden,
                disc_init_scale=config.gated_policy_disc_init_scale,
                head_hidden=config.gated_policy_head_hidden,
                bias_scale=config.gated_policy_bias_scale,
                head_init_bias_offset=config.gated_policy_head_init_bias_offset,
                # ARC-062 GAP-B option-2: head-input first-action one-hot.
                # first_action_dim derived from config.e2.action_dim (single
                # source of truth; not exposed as a separate REEConfig knob).
                use_first_action_onehot=getattr(
                    config, "gated_policy_use_first_action_onehot", False
                ),
                first_action_dim=config.e2.action_dim,
                # INV-074 / MECH-333 / MECH-334: arm Phase-3 plasticity-
                # injection crystallization. crystallize_at_phase3 only
                # ARMS the capability; the actual crystallize() call fires
                # at the infant-curriculum Phase 2->3 transition via the
                # experiment's on_phase3_entry callback. Default False =
                # bit-identical (forward never touches the expansion).
                crystallize_enabled=getattr(
                    config, "crystallize_at_phase3", False
                ),
                crystallize_expansion_hidden=getattr(
                    config, "gated_policy_crystallize_expansion_hidden", 32
                ),
                # ARC-062 robustness fix (2026-05-18, V3-EXQ-543h autopsy):
                # base + candidate-axis-norm-pinned differential
                # reparameterization so head_0==head_1 collapse is a
                # structural non-equilibrium. getattr fallback => bit-
                # identical when the REEConfig attr is absent.
                use_differential_heads=getattr(
                    config, "gated_policy_use_differential_heads", False
                ),
                differential_bias_scale=getattr(
                    config, "gated_policy_differential_bias_scale", 0.1
                ),
                mode_separation_floor=getattr(
                    config, "gated_policy_mode_separation_floor", 0.0
                ),
                p1_w_deviation_aux_weight=getattr(
                    config, "gated_policy_p1_w_deviation_aux_weight", 0.0
                ),
            )
            self.gated_policy = GatedPolicy(
                world_dim=config.latent.world_dim,
                self_dim=config.latent.self_dim,
                harm_a_dim=config.latent.z_harm_a_dim,
                config=gp_cfg,
            )

        # MECH-313 (ARC-065): stochastic_noise_floor (LC-NE tonic / SAC analog).
        # State-independent softmax-temperature lift applied at the
        # e3.select() call site in select_action(). Bit-identical baseline
        # when use_noise_floor=False (regulator is None and the call site
        # passes the unmodified baseline temperature). See
        # ree_core/policy/noise_floor.py for the module + Pull 1 SYNTHESIS
        # verdicts. Distinct from MECH-260 dACC anti-recency; Q-045
        # falsifies whether they collapse into a single substrate.
        self.noise_floor: Optional[NoiseFloor] = None
        if getattr(config, "use_noise_floor", False):
            nf_cfg = NoiseFloorConfig(
                use_noise_floor=True,
                noise_floor_alpha=config.noise_floor_alpha,
                noise_floor_min_temperature=config.noise_floor_min_temperature,
            )
            self.noise_floor = NoiseFloor(config=nf_cfg)

        # MECH-090 R-c conjunction: commit-entry readiness signal. Adds a
        # readiness_above_floor conjunction to the rv-only BetaGate commit-
        # entry predicate. CommitReadiness maintains an EMA over per-tick
        # outcome signals (env-emitted resource contacts by default; harness
        # may override via notify_outcome). The conjunction wraps the two
        # beta_gate.elevate() call sites in select_action(); the
        # BetaGate.elevate() signature is unchanged. See
        # ree_core/policy/commit_readiness.py and
        # REE_assembly/docs/architecture/mech_090_commit_entry_predicate.md.
        # Bit-identical baseline when use_commit_readiness=False AND
        # use_mech090_readiness_conjunction=False; the conjunction master
        # flag auto-arms use_commit_readiness via REEConfig.__post_init__ /
        # from_dims, so callers only set the conjunction flag.
        self.commit_readiness: Optional[CommitReadiness] = None
        if getattr(config, "use_commit_readiness", False):
            cr_cfg = CommitReadinessConfig(
                use_commit_readiness=True,
                commit_readiness_window=getattr(
                    config, "commit_readiness_window", 20
                ),
                commit_readiness_ema_alpha=getattr(
                    config, "commit_readiness_ema_alpha", 0.1
                ),
                commit_readiness_initial=getattr(
                    config, "commit_readiness_initial", 1.0
                ),
            )
            self.commit_readiness = CommitReadiness(config=cr_cfg)

        # MECH-342: maintenance-time readiness-driven commitment release
        # (B3b). Release-side complement to the MECH-090 admission
        # conjunction above: the SAME two R-c readiness signals
        # (score_margin decisiveness + nav_competence) that admit a
        # commitment here drive a graded, bounded-accumulation RELEASE of an
        # already-elevated beta latch when they degrade mid-commitment.
        # Closes the V3-EXQ-592f reach gap (predicates fire under forced
        # beta-elevated state but produce zero state-occupancy suppression).
        # Bit-identical baseline when use_maintenance_release=False (the
        # release branch in select_action is skipped when this is None).
        # See ree_core/policy/commit_maintenance_release.py and
        # REE_assembly/docs/architecture/mech_342_commit_maintenance_release.md.
        self.maintenance_release: Optional[CommitMaintenanceRelease] = None
        if getattr(config, "use_maintenance_release", False):
            mr_cfg = CommitMaintenanceReleaseConfig(
                use_maintenance_release=True,
                score_margin_floor=getattr(
                    config, "maintenance_release_score_margin_floor", 0.05
                ),
                score_margin_reengage=getattr(
                    config, "maintenance_release_score_margin_reengage", 0.10
                ),
                nav_floor=getattr(
                    config, "maintenance_release_nav_floor", 0.3
                ),
                nav_reengage=getattr(
                    config, "maintenance_release_nav_reengage", 0.5
                ),
                accumulation_rate=getattr(
                    config, "maintenance_release_accumulation_rate", 0.2
                ),
                leak_rate=getattr(
                    config, "maintenance_release_leak_rate", 0.1
                ),
                release_bound=getattr(
                    config, "maintenance_release_bound", 1.0
                ),
                pressure_cap=getattr(
                    config, "maintenance_release_pressure_cap", 1.5
                ),
            )
            self.maintenance_release = CommitMaintenanceRelease(config=mr_cfg)

        # Commit/release-DURATION lever: graded natural-commit-occupancy release
        # (rung-6 of f_dominance_conversion_ceiling; the duration face PARALLEL to
        # the selection-face MECH-448). Reduces the F-driven natural-commit latch
        # occupancy (~2400-2600 steps on strong seeds, V3-EXQ-460h) so weak-
        # natural-commit is the norm across seeds. BG-3 SYNTHESIS divergence D1:
        # GRADED release (Thura/Cisek 2022 urgency + Jin 2014 behaviour-co-
        # extensive), NOT another fixed refractory. Reuses BetaGate.committed_run_length
        # (no parallel latch; ARC-106 G2). No-op when the master flag is False
        # (self.natural_commit_urgency is None) -> bit-identical. See
        # ree_core/policy/natural_commit_urgency.py and
        # REE_assembly/docs/architecture/natural_commit_occupancy_release.md.
        self.natural_commit_urgency: Optional[NaturalCommitUrgencyRelease] = None
        if getattr(config, "use_natural_commit_urgency_release", False):
            ncur_cfg = NaturalCommitUrgencyReleaseConfig(
                use_natural_commit_urgency_release=True,
                urgency_mode=getattr(
                    config, "natural_commit_release_urgency_mode", True
                ),
                action_extent_mode=getattr(
                    config, "natural_commit_release_action_extent_mode", True
                ),
                urgency_rate=getattr(
                    config, "natural_commit_urgency_rate", 0.01
                ),
                release_bound=getattr(
                    config, "natural_commit_urgency_release_bound", 1.0
                ),
                urgency_cap=getattr(
                    config, "natural_commit_urgency_cap", 1.5
                ),
                gap_entry_sensitivity=getattr(
                    config, "natural_commit_gap_entry_sensitivity", 1.0
                ),
                onset_ticks=getattr(
                    config, "natural_commit_urgency_onset_ticks", 0
                ),
            )
            self.natural_commit_urgency = NaturalCommitUrgencyRelease(
                config=ncur_cfg
            )

        # Natural-commit LATCH-HOLD (rung-6 amend, 2026-06-21,
        # failure_autopsy_V3-EXQ-460i). SEPARATE from the release lever above (and
        # independent of it -- arms even when natural_commit_urgency is None, e.g.
        # the ARM_LEVER_OFF baseline). Establishes a sustained natural-commit
        # beta-latch occupancy for the rung-6 release to act on: a natural commit
        # (result.committed) arms the hold; while armed AND the committed trajectory
        # persists, the beta latch is RE-ASSERTED each tick against the SD-034
        # de-commit churn that fragmented the 460i latch to ~1-tick blips. The hold
        # YIELDS to the three principled releases (MECH-091 threat, the rung-6
        # duration release, an SD-034 closure de-commit via the active refractory).
        # Default False -> _ncl_hold_active stays False, no re-assert -> bit-identical.
        self._ncl_hold_active: bool = False
        self._ncl_hold_ticks: int = 0
        self._ncl_hold_reassert_count: int = 0
        # Per-tick principled-release flags (cleared at the top of each select_action
        # release region; set at the MECH-091 / rung-6 release sites; read by the
        # end-of-tick hold re-assertion so the hold yields rather than fights them).
        self._ncl_mech091_fired: bool = False
        self._ncl_lever_fired: bool = False
        # Closure-exclusive de-commit eval mode (rung-6 BUILD, 2026-06-22, the named
        # dissociable substrate from failure_autopsy_V3-EXQ-460j). When on, the eval
        # makes beta elevation closure-exclusive (the F-driven result.committed path
        # is suppressed from _commit_for_beta) AND arms the latch-hold on the
        # closure-coupled commit (_closure_commit_active), so a sustained occupancy
        # forms via the closure plane independently of the fragile natural commit and
        # the SD-034 closure de-commit can act on it. PRECONDITION: requires both the
        # closure->beta coupling and the latch-hold (the eval has nothing to form/sustain
        # the occupancy otherwise). Counter certifies the eval-mode arm path fired.
        self._ncl_hold_closure_armed_count: int = 0
        if getattr(config, "closure_exclusive_decommit_eval", False):
            if not getattr(config, "use_closure_commit_beta_coupling", False):
                raise ValueError(
                    "closure_exclusive_decommit_eval=True requires "
                    "use_closure_commit_beta_coupling=True (the closure->beta coupling "
                    "is the only beta-elevation path in the eval)."
                )
            if not getattr(config, "use_natural_commit_latch_hold", False):
                raise ValueError(
                    "closure_exclusive_decommit_eval=True requires "
                    "use_natural_commit_latch_hold=True (the latch-hold sustains the "
                    "closure-coupled occupancy the de-commit acts on)."
                )

        # Closure-plane commit-ENTRY primitive (rung-6 amend; commitment_closure:GAP-4;
        # failure_autopsy_V3-EXQ-460k/460l). The F-INDEPENDENT latch that lets the
        # closure-exclusive eval arm WITHOUT a sustained F-commit. PRECONDITIONS: it feeds
        # _closure_commit_active (gated on the coupling) and arms the latch-hold, so both
        # must be on -- a misconfig is surfaced loud (the MECH-269b / SD-058 precondition
        # pattern), never a silent no-op.
        if getattr(config, "use_closure_commit_entry", False):
            if not getattr(config, "use_closure_commit_beta_coupling", False):
                raise ValueError(
                    "use_closure_commit_entry=True requires "
                    "use_closure_commit_beta_coupling=True (the closure->beta coupling is "
                    "the consumer of the F-independent closure-plane commit-entry latch)."
                )
            if not getattr(config, "use_natural_commit_latch_hold", False):
                raise ValueError(
                    "use_closure_commit_entry=True requires "
                    "use_natural_commit_latch_hold=True (the latch-hold sustains the "
                    "closure-formed occupancy the commit-entry latch arms)."
                )
        # The C-STEP trajectory extension rides on the bool latch; arming the trajectory
        # without the parent flag would install a closure trajectory that the
        # _closure_commit_active arm gate never reads -- surface the misconfig loud.
        if getattr(config, "use_closure_commit_entry_trajectory", False):
            if not getattr(config, "use_closure_commit_entry", False):
                raise ValueError(
                    "use_closure_commit_entry_trajectory=True requires "
                    "use_closure_commit_entry=True (the trajectory latch extends the "
                    "F-independent closure-plane commit-entry bool latch)."
                )

        # ARC-108 JOB-2 (c): rho_t MAINTENANCE RAMP -- the proximity-scaled driver
        # that REPLACES the flat-hold maintenance DRIVER of the natural-commit
        # latch-hold above (the per-tick unconditional re-assert / deviation B6).
        # rho_t = goal-proximity x value (reuse the goal/benefit valuation feeding
        # F) ramps up while approaching the goal and DECLINES past the proximity
        # peak, so the hold self-limits instead of running ~2400 steps. COMPOSES
        # with the latch machinery (gate/operator/refractory kept); the ramp only
        # decides WHEN the hold ends. PRECONDITION: requires the latch-hold (the
        # hold this ramp drives). Default False -> the latch-hold's flat re-assertion
        # is unchanged (bit-identical). See ree_core/policy/rho_maintenance_ramp.py.
        self.rho_maintenance_ramp: Optional[RhoMaintenanceRamp] = None
        if getattr(config, "use_rho_maintenance_ramp", False):
            if not getattr(config, "use_natural_commit_latch_hold", False):
                raise ValueError(
                    "use_rho_maintenance_ramp=True requires "
                    "use_natural_commit_latch_hold=True (the rho_t ramp REPLACES the "
                    "flat-hold maintenance driver of the natural-commit latch-hold; "
                    "there is no hold to drive otherwise)."
                )
            self.rho_maintenance_ramp = RhoMaintenanceRamp(
                config=RhoMaintenanceRampConfig(
                    use_rho_maintenance_ramp=True,
                    hold_floor=getattr(config, "rho_hold_floor", 0.05),
                    release_margin=getattr(config, "rho_release_margin", 0.5),
                    onset_grace_ticks=getattr(config, "rho_onset_grace_ticks", 3),
                )
            )

        # SD-061: difficulty-gated proposal-entropy regulator (MECH-343 blocker
        # part 2). A stuck-state detector integrates goal-progress stall + dACC
        # choice difficulty + E3 score margin + committed-action diversity
        # (guarded by goal salience) into a graded stuck_score; the regulator
        # maps it to a transient gain on the ARC-018/CEM proposal layer (wider
        # candidate set + within-class temperature), decaying as the impasse
        # clears. Bit-identical baseline when
        # use_difficulty_gated_proposal_entropy=False (both are None and the
        # _e3_tick proposal-gain + select_action detector-update blocks are
        # skipped). See ree_core/cingulate/stuck_state_detector.py,
        # ree_core/policy/difficulty_gated_proposal_entropy.py and
        # REE_assembly/docs/architecture/sd_061_difficulty_gated_proposal_entropy.md.
        self.stuck_state_detector: Optional[StuckStateDetector] = None
        self.difficulty_gated_proposal_entropy: Optional[
            DifficultyGatedProposalEntropy
        ] = None
        self._last_stuck_score: float = 0.0
        if getattr(config, "use_difficulty_gated_proposal_entropy", False):
            self.stuck_state_detector = StuckStateDetector(
                config=StuckStateDetectorConfig(
                    use_stuck_state_detector=True,
                    progress_window=getattr(config, "stuck_progress_window", 8),
                    progress_stall_eps=getattr(
                        config, "stuck_progress_stall_eps", 0.01
                    ),
                    score_margin_floor=getattr(
                        config, "stuck_score_margin_floor", 0.05
                    ),
                    committed_diversity_window=getattr(
                        config, "stuck_committed_diversity_window", 8
                    ),
                    committed_diversity_floor=getattr(
                        config, "stuck_committed_diversity_floor", 0.34
                    ),
                    choice_difficulty_ref=getattr(
                        config, "stuck_choice_difficulty_ref", 0.05
                    ),
                    goal_salience_floor=getattr(
                        config, "stuck_goal_salience_floor", 0.05
                    ),
                    ema_alpha_rise=getattr(config, "stuck_ema_alpha_rise", 0.3),
                    ema_alpha_fall=getattr(config, "stuck_ema_alpha_fall", 0.05),
                    stuck_threshold=getattr(config, "stuck_threshold", 0.5),
                    combine_mode=getattr(config, "stuck_combine_mode", "mean"),
                )
            )
            self.difficulty_gated_proposal_entropy = DifficultyGatedProposalEntropy(
                config=DifficultyGatedProposalEntropyConfig(
                    use_difficulty_gated_proposal_entropy=True,
                    candidate_widen_max=getattr(
                        config, "dgpe_candidate_widen_max", 8
                    ),
                    temperature_gain_max=getattr(
                        config, "dgpe_temperature_gain_max", 1.0
                    ),
                )
            )

        # MECH-314 (ARC-065): structured_curiosity_bonus (frontopolar / EFE
        # analog). State-DEPENDENT score-bias on E3 candidate scoring.
        # Sibling to MECH-313 stochastic_noise_floor at the policy layer:
        # MECH-313 lifts the softmax temperature uniformly (state-
        # independent); MECH-314 adds a per-candidate (314a) or broadcast-
        # scalar (314b/c Phase 1) NEGATIVE score_bias that makes novel /
        # uncertain / learning-progress-rich candidates more attractive
        # under the lower-is-better convention. Three sub-flavours
        # registered separately under ARC-065 (Pull 1 SYNTHESIS R3 +
        # Q-044): MECH-314a striatal novelty (Wittmann 2008), MECH-314b
        # frontopolar uncertainty (Daw 2006 / Friston EFE), MECH-314c
        # learning-progress (Schmidhuber / Pathak; least biologically
        # anchored). Master + 3 sub-switches independently togglable so
        # Q-044's three-arm ablation is a flag-set decision. See
        # ree_core/policy/structured_curiosity.py and
        # REE_assembly/docs/architecture/mech_314_structured_curiosity_bonus.md.
        # Bit-identical baseline when use_structured_curiosity=False.
        self.curiosity: Optional[StructuredCuriosity] = None
        if getattr(config, "use_structured_curiosity", False):
            cur_cfg = StructuredCuriosityConfig(
                use_structured_curiosity=True,
                use_curiosity_novelty=config.use_curiosity_novelty,
                use_curiosity_uncertainty=config.use_curiosity_uncertainty,
                use_curiosity_learning_progress=config.use_curiosity_learning_progress,
                curiosity_novelty_weight=config.curiosity_novelty_weight,
                curiosity_uncertainty_weight=config.curiosity_uncertainty_weight,
                curiosity_learning_progress_weight=config.curiosity_learning_progress_weight,
                curiosity_bias_scale=config.curiosity_bias_scale,
                curiosity_lp_ema_alpha=config.curiosity_lp_ema_alpha,
                curiosity_lp_window_k=config.curiosity_lp_window_k,
                # MECH-314a Phase 2 (Candidate 5A): novelty-source +
                # first-action one-hot augmentation. getattr fallbacks keep
                # configs built before Phase 2 bit-identical.
                novelty_source=getattr(
                    config, "curiosity_novelty_source", "residue"
                ),
                use_first_action_onehot=getattr(
                    config, "curiosity_use_first_action_onehot", False
                ),
                first_action_augmentation_policy=getattr(
                    config, "curiosity_first_action_augmentation_policy", "never"
                ),
                min_spread_threshold=getattr(
                    config, "curiosity_min_spread_threshold", 0.01
                ),
                min_spread_consecutive_ticks=getattr(
                    config, "curiosity_min_spread_consecutive_ticks", 5
                ),
            )
            self.curiosity = StructuredCuriosity(config=cur_cfg)

        # MECH-314a Phase 2 (Candidate 5A): rolling z_world visitation buffer.
        # Populated on waking ticks (MECH-094-gated) in sense() and consumed by
        # StructuredCuriosity as the "visitation" novelty comparison set. Built
        # only when curiosity is active AND the novelty source needs it, so the
        # default "residue" path allocates nothing (bit-identical OFF).
        self._zworld_visitation_buffer: Optional[deque] = None
        if self.curiosity is not None and getattr(
            config, "curiosity_novelty_source", "residue"
        ) in ("visitation", "auto"):
            self._zworld_visitation_buffer = deque(
                maxlen=int(getattr(config, "curiosity_visitation_buffer_len", 256))
            )

        # MECH-320 (ARC-066 child): tonic_vigor_coupling_score_bias.
        # Capacity-keyed additive (or multiplicative-gain) bias on E3
        # action-vs-no-op scoring. Vigor scalar = slow EWMA over realised
        # E3-score-receipt, gated by secondary internal-state modulators
        # (energy / drive / recent PE). Sister to MECH-313 (noise floor,
        # orthogonal axis) and MECH-314 (curiosity bonus, candidate-level
        # axis); MECH-320 is the action-vs-no-op axis. R3 lit-pull verdict
        # ADDITIVE primary; MULTIPLICATIVE GAIN falsifiable secondary --
        # both implementable via tonic_vigor_form. R4 verdict slow EWMA
        # over reward history is the primary scalar (Niv 2007 / Beierholm
        # 2013); internal-state proxies are secondary modulators. See
        # ree_core/policy/tonic_vigor.py and ARC-066 lit-pull SYNTHESIS
        # at REE_assembly/evidence/literature/targeted_review_arc_066_tonic_vigor/
        # synthesis.md (lit_conf 0.789, supports). Bit-identical baseline
        # when use_tonic_vigor=False.
        self.tonic_vigor: Optional[TonicVigor] = None
        if getattr(config, "use_tonic_vigor", False):
            tv_cfg = TonicVigorConfig(
                use_tonic_vigor=True,
                half_life=config.tonic_vigor_half_life,
                w_action=config.tonic_vigor_w_action,
                w_passive=config.tonic_vigor_w_passive,
                bias_scale=config.tonic_vigor_bias_scale,
                gate_energy_min=config.tonic_vigor_gate_energy_min,
                gate_drive_max=config.tonic_vigor_gate_drive_max,
                gate_pe_max=config.tonic_vigor_gate_pe_max,
                form=config.tonic_vigor_form,
                noop_class=config.tonic_vigor_noop_class,
                v_t_floor=getattr(config, "tonic_vigor_v_t_floor", 0.0),
            )
            self.tonic_vigor = TonicVigor(config=tv_cfg)

        # MECH-353: blocked-agency / control-failure affect stream (z_block).
        # Pure-arithmetic regulator integrating the SD-029 agency comparator on
        # the action-outcome / z_world channel (intended effect predicted by
        # E2.world_forward vs realised), behind an external-attribution gate
        # (motor intact on z_self) and a capacity gate (assert while
        # capacity-belief retained; hand off to z_harm_a as it collapses).
        # Bit-identical baseline when use_blocked_agency=False (the new
        # LatentState.z_block field stays None and no consumer fires).
        # See ree_core/affect/blocked_agency.py +
        # REE_assembly/docs/architecture/mech_353_blocked_agency_zblock.md.
        self.blocked_agency: Optional[BlockedAgency] = None
        if getattr(config, "use_blocked_agency", False):
            ba_cfg = BlockedAgencyConfig(
                use_blocked_agency=True,
                accumulation_rate=config.blocked_agency_accumulation_rate,
                leak_rate=config.blocked_agency_leak_rate,
                outcome_mismatch_floor=config.blocked_agency_outcome_mismatch_floor,
                attribution_motor_floor=config.blocked_agency_attribution_motor_floor,
                capacity_collapse_weight=config.blocked_agency_capacity_collapse_weight,
                require_goal_active=config.blocked_agency_require_goal_active,
                z_block_cap=config.blocked_agency_z_block_cap,
                assert_action_weight=config.blocked_agency_assert_action_weight,
                assert_passive_weight=config.blocked_agency_assert_passive_weight,
                assert_alt_action_weight=config.blocked_agency_assert_alt_action_weight,
                assert_bias_scale=config.blocked_agency_assert_bias_scale,
                decommit_bound=config.blocked_agency_decommit_bound,
                decommit_consecutive_ticks=config.blocked_agency_decommit_consecutive_ticks,
                noop_class=config.blocked_agency_noop_class,
            )
            self.blocked_agency = BlockedAgency(config=ba_cfg)
        # MECH-353 per-tick caches: previous z_world / z_self latents for the
        # action-outcome and motor-agency comparators (None until first sense()).
        self._ba_prev_z_world: Optional[torch.Tensor] = None
        self._ba_prev_z_self: Optional[torch.Tensor] = None

        # MECH-219 (SD-019b): affective-harm hysteretic integrator. Turns the
        # SD-019a medium-timescale unpleasantness channel (z_harm_un) into a
        # slow, persistent, controllability-gated SUFFERING load state
        # (z_harm_suffering) via an asymmetric (hysteretic) integrator. Pure-
        # arithmetic regulator; bit-identical when
        # use_harm_suffering_accumulator=False (the accumulator is None,
        # LatentState.z_harm_suffering stays None, no consumer redirect fires).
        # Requires the SD-019a unpleasantness channel (use_harm_un=True) as its
        # drive input. See
        # REE_assembly/evidence/planning/mech_219_hysteretic_integrator_design.md.
        self.harm_suffering_accumulator: Optional[HarmSufferingAccumulator] = None
        if getattr(config, "use_harm_suffering_accumulator", False):
            if not getattr(config.latent, "use_harm_un", False):
                raise ValueError(
                    "use_harm_suffering_accumulator=True requires use_harm_un=True "
                    "(MECH-219 integrates the SD-019a z_harm_un unpleasantness "
                    "channel; without it there is no drive signal)."
                )
            hsa_cfg = HarmSufferingAccumulatorConfig(
                use_harm_suffering_accumulator=True,
                alpha_rise=config.harm_suffering_alpha_rise,
                alpha_fall=config.harm_suffering_alpha_fall,
                escapability_mode=config.harm_suffering_escapability_mode,
                escapability_constant=config.harm_suffering_escapability_constant,
                s_cap=config.harm_suffering_s_cap,
                body_damage_weight=config.harm_suffering_body_damage_weight,
                pe_gain=config.harm_suffering_pe_gain,
                use_bistable_latch=config.harm_suffering_use_bistable_latch,
                theta_on=config.harm_suffering_theta_on,
                theta_off=config.harm_suffering_theta_off,
            )
            self.harm_suffering_accumulator = HarmSufferingAccumulator(config=hsa_cfg)
        # MECH-219 external-escapability seam: the `external` escapability mode
        # reads this scalar (a validation experiment drives it via
        # set_harm_suffering_escapability); ignored in constant/avoidance_efficacy
        # modes.
        self._harm_suffering_external_escapability: float = float(
            getattr(config, "harm_suffering_external_escapability", 1.0)
        )

        # MECH-276: scientist-agent counterfactual-backed attribution buffer.
        # The waking-phase feedstock for the MECH-275 sleep aggregator. On each
        # waking tick the agent computes a counterfactual-backed attribution
        # from the SD-031 E2WorldForward (domain "place") and/or ARC-033
        # E2HarmSForward (domain "self") comparators and buffers it keyed by the
        # current MECH-269 anchor region. Backward compatible: None when off.
        # PRECONDITION: requires a comparator to source attributions from --
        # use_e2_world_forward OR use_e2_harm_s_forward (loud, not silent).
        self.scientist_attribution_buffer: Optional[ScientistAttributionBuffer] = None
        if getattr(config, "use_scientist_attribution", False):
            if self.e2_world is None and self.e2_harm_s is None:
                raise ValueError(
                    "use_scientist_attribution=True requires a single-pass "
                    "comparator to source attributions from: set "
                    "use_e2_world_forward=True (SD-031, domain 'place') and/or "
                    "use_e2_harm_s_forward=True (ARC-033, domain 'self'). "
                    "MECH-276 has no feedstock source without one."
                )
            sci_cfg = ScientistAttributionConfig(
                cf_margin=float(getattr(config, "scientist_attribution_cf_margin", 0.05)),
                only_counterfactual_backed=bool(
                    getattr(config, "scientist_attribution_only_counterfactual_backed", True)
                ),
                ema_alpha=float(getattr(config, "scientist_attribution_ema_alpha", 0.3)),
                decay=float(getattr(config, "scientist_attribution_decay", 1.0)),
            )
            self.scientist_attribution_buffer = ScientistAttributionBuffer(config=sci_cfg)
        # MECH-276 prev-latent caches (one-tick lag, mirroring the MECH-353
        # blocked-agency comparator cache).
        self._sci_prev_z_world: Optional[torch.Tensor] = None
        self._sci_prev_z_harm_s: Optional[torch.Tensor] = None

        # SD-058 / MECH-357: ilPFC-analog instrumental-avoidance gate. Resolves
        # the Pavlovian-instrumental conflict (Moscarello & LeDoux 2013) -- a
        # per-candidate anti-passivity score-bias (release the instrumental
        # action), an ilPFC suppression of the MECH-279 freeze no-op, and an
        # eligibility-trace avoidance-efficacy learner driven by z_harm_a drops.
        # Bit-identical baseline when use_instrumental_avoidance=False
        # (agent.instrumental_avoidance is None and no consumer fires). See
        # ree_core/pfc/infralimbic_avoidance_gate.py +
        # REE_assembly/docs/architecture/sd_058_instrumental_avoidance_acquisition.md.
        self.instrumental_avoidance: Optional[InstrumentalAvoidanceGate] = None
        if getattr(config, "use_instrumental_avoidance", False):
            ia_cfg = InstrumentalAvoidanceGateConfig(
                learn_rate=config.avoidance_learn_rate,
                leak_rate=config.avoidance_leak_rate,
                initial_efficacy=config.avoidance_initial_efficacy,
                scaffold_floor=config.avoidance_scaffold_floor,
                efficacy_reward_floor=config.avoidance_efficacy_reward_floor,
                threat_floor=config.avoidance_threat_floor,
                threat_ref=config.avoidance_threat_ref,
                action_bias_gain=config.avoidance_action_bias_gain,
                bias_scale=config.avoidance_bias_scale,
                noop_class=config.avoidance_noop_class,
                suppression_threshold=config.avoidance_suppression_threshold,
            )
            self.instrumental_avoidance = InstrumentalAvoidanceGate(config=ia_cfg)
        # MECH-357 per-tick cache: whether the last emitted action was directed
        # (non-noop). Fed to the eligibility-trace update on the NEXT sense().
        self._ia_last_action_directed: bool = False

        # SD-059 / MECH-358: relief/safety escape-affordance bridge. Extends the
        # MECH-357 scalar avoidance_efficacy into a per-first-action-class credit
        # table so the agent acquires a DIRECTED escape (the V3-EXQ-603h gap:
        # MECH-357 un-freezes but binds no specific action to relief/safety).
        # Bit-identical baseline when use_escape_affordance_bridge=False
        # (agent.escape_affordance_bridge is None and no consumer fires). See
        # ree_core/pfc/escape_affordance_bridge.py +
        # REE_assembly/docs/architecture/sd_059_escape_affordance_bridge.md.
        self.escape_affordance_bridge: Optional[EscapeAffordanceBridge] = None
        if getattr(config, "use_escape_affordance_bridge", False):
            eab_cfg = EscapeAffordanceBridgeConfig(
                n_action_classes=int(config.e2.action_dim),
                use_relief_credit=config.use_escape_relief_credit,
                use_safety_credit=config.use_escape_safety_credit,
                relief_learn_rate=config.escape_relief_learn_rate,
                safety_learn_rate=config.escape_safety_learn_rate,
                leak_rate=config.escape_bridge_leak_rate,
                relief_reward_floor=config.escape_relief_reward_floor,
                threat_floor=config.escape_threat_floor,
                threat_ref=config.escape_threat_ref,
                approach_gain=config.escape_approach_gain,
                bias_scale=config.escape_bias_scale,
                noop_class=config.escape_noop_class,
                use_trained_safety_signal=getattr(
                    config, "escape_use_trained_safety_signal", False
                ),
                safety_signal_threshold=getattr(
                    config, "escape_safety_signal_threshold", 0.5
                ),
            )
            self.escape_affordance_bridge = EscapeAffordanceBridge(config=eab_cfg)
        # SD-059 per-tick cache: the first-action class of the last emitted
        # action. Fed to the bridge eligibility update on the NEXT sense().
        self._eab_last_action_class: Optional[int] = None

        # ARC-006 / MECH-045: token-instance object-file / entity-persistence
        # buffer. The TOKEN store of the ARC-080 type/token/anchor triad. v1
        # lands STANDALONE -- no action-stream consumer -- so the action stream
        # is bit-identical whether the buffer is on or off; only buffer state
        # changes. Driven on the waking stream via update_object_file_buffer()
        # by an experiment / harness that supplies the perceived entities. See
        # ree_core/entities/object_file_buffer.py +
        # REE_assembly/docs/architecture/mech_045_object_file_buffer.md.
        self.object_file_buffer: Optional[ObjectFileBuffer] = None
        if getattr(config, "use_object_file_buffer", False):
            obf_cfg = ObjectFileBufferConfig(
                use_object_file_buffer=True,
                max_tokens=int(getattr(config, "obf_max_tokens", 5)),
                continuity_radius=float(getattr(config, "obf_continuity_radius", 2.0)),
                w_motion=float(getattr(config, "obf_w_motion", 1.0)),
                w_feat=float(getattr(config, "obf_w_feat", 1.0)),
                feature_alpha=float(getattr(config, "obf_feature_alpha", 0.3)),
                persist_ttl=int(getattr(config, "obf_persist_ttl", 8)),
                min_birth_salience=float(getattr(config, "obf_min_birth_salience", 0.0)),
                use_precision_weighting=bool(
                    getattr(config, "obf_use_precision_weighting", True)
                ),
            )
            self.object_file_buffer = ObjectFileBuffer(config=obf_cfg)

        # Post-603i successor scaffold: trainable-head relief/safety escape-
        # affordance learner. This is intentionally separate from the arithmetic
        # SD-059 bridge so V3-EXQ-603i remains unchanged. When enabled together,
        # the two score-bias sources compose additively, each under its own clamp.
        self.trainable_escape_affordance_learner: Optional[
            TrainableEscapeAffordanceLearner
        ] = None
        if getattr(config, "use_trainable_escape_affordance_learner", False):
            teal_cfg = TrainableEscapeAffordanceLearnerConfig(
                enabled=True,
                n_action_classes=int(config.e2.action_dim),
                use_relief_critic=bool(config.use_trainable_relief_critic),
                use_safety_predictor=bool(config.use_trainable_safety_predictor),
                bias_scale=float(config.trainable_escape_bias_scale),
                relief_learn_rate=float(config.trainable_escape_relief_learn_rate),
                safety_learn_rate=float(config.trainable_escape_safety_learn_rate),
                optimizer_lr=float(config.trainable_escape_optimizer_lr),
                leak_rate=float(config.trainable_escape_leak_rate),
                relief_reward_floor=float(config.trainable_escape_relief_reward_floor),
                relief_target_scale=float(config.trainable_escape_relief_target_scale),
                threat_floor=float(config.trainable_escape_threat_floor),
                noop_class=int(config.trainable_escape_noop_class),
                hidden_dim=int(config.trainable_escape_hidden_dim),
                action_embedding_dim=int(config.trainable_escape_action_embedding_dim),
                prediction_floor=float(config.trainable_escape_prediction_floor),
            )
            self.trainable_escape_affordance_learner = (
                TrainableEscapeAffordanceLearner(config=teal_cfg)
            )
        self._teal_last_action_class: Optional[int] = None

        # Post-603i E2 escape-affordance linker. A READOUT over the EXISTING E2
        # (cerebellar-analog) action-consequence forward model (self.e2.world_forward)
        # -- NOT a new forward predictor. It indexes E2 geometry into escape-
        # affordance viability readouts (V3-EXQ-603i: the relief/safety bridge
        # needed a trained encoder/world-forward it did not have), exposes
        # escape_affordance_features for the relief/safety heads, and (behind
        # use_e2_escape_linker_e3_bias) emits a bounded threat-gated E3 bias.
        # Bit-identical baseline when use_e2_escape_affordance_linker=False
        # (agent.e2_escape_affordance_linker is None and no consumer fires). See
        # ree_core/pfc/e2_escape_affordance_linker.py +
        # docs/substrate_plans/post_603i_e2_escape_affordance_linkage.md.
        self.e2_escape_affordance_linker: Optional[E2EscapeAffordanceLinker] = None
        if getattr(config, "use_e2_escape_affordance_linker", False):
            eal_cfg = E2EscapeAffordanceLinkerConfig(
                enabled=True,
                n_action_classes=int(config.e2.action_dim),
                hidden_dim=int(config.escape_linker_hidden_dim),
                action_embedding_dim=int(config.escape_linker_action_embedding_dim),
                learn_rate=float(config.escape_linker_learn_rate),
                optimizer_lr=float(config.escape_linker_optimizer_lr),
                leak_rate=float(config.escape_linker_leak_rate),
                bias_scale=float(config.escape_linker_bias_scale),
                threat_floor=float(config.escape_linker_threat_floor),
                threat_ref=float(config.escape_linker_threat_ref),
                noop_class=int(config.escape_linker_noop_class),
                relief_reward_floor=float(config.escape_linker_relief_reward_floor),
                harm_delta_scale=float(config.escape_linker_harm_delta_scale),
                prediction_floor=float(config.escape_linker_prediction_floor),
                block_hypothesis_learning=bool(
                    config.escape_linker_block_hypothesis_learning
                ),
            )
            self.e2_escape_affordance_linker = E2EscapeAffordanceLinker(config=eal_cfg)
        # Per-tick caches: the just-emitted action class and the z_world at the
        # time of that action, used on the NEXT sense() to read the detached E2
        # action-consequence feature for the executed (state, action) pair.
        self._eal_last_action_class: Optional[int] = None
        self._eal_prev_z_world: Optional[torch.Tensor] = None

        # MECH-341 (ARC-065 Layer-B child): e3_scoring_preserves_trajectory_
        # class_diversity. Layer-B diversity-preservation substrate triggered
        # by V3-EXQ-608 P2 R2a_e3_collapse_confirmed_large_gap finding
        # (2026-05-26): CEM delivers >=2 first-action classes but E3 scoring
        # collapses to a single class with large score gap (rules out option
        # 3 jittered tie-breaking; routes to options 1+2). Two togglable sub-
        # flavours under one master per behavioral_diversity_isolation_plan.md
        # "Substrate design options" section. Bit-identical baseline when
        # use_e3_score_diversity=False. See ree_core/predictors/e3_score_diversity.py
        # and REE_assembly/docs/architecture/mech_341_e3_score_diversity_preservation.md.
        self.score_diversity: Optional[E3ScoreDiversity] = (
            build_e3_score_diversity_from_ree_config(config)
        )

        # MECH-319 (arc_062 GAP-K): simulation_mode_rule_write_gate.
        # Substrate-level instantiation of MECH-094 at the rule-arbitration
        # layer. Unified categorical write gate consulted by the arbitration-
        # write call sites (gated_policy.forward + lateral_pfc_analog.update)
        # to translate caller-supplied simulation_mode tag + admit_writes
        # falsifier flag into the final admit/block decision. Bit-identical
        # OFF when use_simulation_mode_rule_gate=False (None handle; call
        # sites pass simulation_mode=False directly). Construction raises
        # ValueError when admit_writes=True without master ON (loud-not-
        # silent guard). See ree_core/regulators/simulation_mode_rule_gate.py
        # and REE_assembly/docs/architecture/mech_319_simulation_mode_rule_gate.md.
        self.simulation_mode_rule_gate: Optional[SimulationModeRuleGate] = None
        if getattr(config, "use_simulation_mode_rule_gate", False):
            smrg_cfg = SimulationModeRuleGateConfig(
                use_simulation_mode_rule_gate=True,
                admit_writes=getattr(
                    config, "simulation_mode_rule_gate_admit_writes", False
                ),
            )
            self.simulation_mode_rule_gate = SimulationModeRuleGate(config=smrg_cfg)

        # SD-034: governance.closure_operator (five-part "done" token)
        # Coordinates release of BetaGate latch, targeted No-Go via dACC,
        # rule-domain residue discharge, closure signal to SalienceCoordinator,
        # and dACC pe buffer reset/cap. Master switch defaults False (backward
        # compat). Requires lateral_pfc (MECH-262 rule_state is the completion
        # source), dacc (MECH-260/268), residue_field, beta_gate. salience
        # coordinator optional (falls back to direct fire when None).
        self.closure_operator: Optional[ClosureOperator] = None
        if getattr(config, "use_closure_operator", False):
            if self.lateral_pfc is None:
                raise ValueError(
                    "use_closure_operator=True requires use_lateral_pfc_analog=True "
                    "(SD-033a rule_state is the completion source)."
                )
            clo_cfg = ClosureOperatorConfig(
                use_closure_operator=True,
                completion_rule_delta_threshold=config.closure_rule_delta_threshold,
                completion_stable_ticks=config.closure_stable_ticks,
                require_beta_elevated=config.closure_require_beta_elevated,
                min_sd033a_write_gate=config.closure_min_sd033a_gate,
                nogo_injection_count=config.closure_nogo_injection_count,
                residue_discharge_factor=config.closure_residue_discharge_factor,
                residue_discharge_radius=config.closure_residue_discharge_radius,
                closure_signal_value=config.closure_signal_value,
                reset_pe_ema=config.closure_reset_pe_ema,
                pe_cap_after_closure=config.closure_pe_cap_after,
                # SD-034 commitment-closure-control-plane (2026-06-12):
                # de-commitment hold/refractory on the closure-driven release.
                # getattr fallback keeps from_dims-absent callers bit-identical.
                decommit_hold_ticks=getattr(
                    config, "closure_decommit_hold_ticks", 0
                ),
                # SD-034 de-commit-authority MAGNITUDE amend (2026-06-19):
                # scale the Leg-B refractory by committed-run length. getattr
                # fallback keeps from_dims-absent callers bit-identical.
                decommit_hold_scale_with_run=getattr(
                    config, "closure_decommit_hold_scale_with_run", 0.0
                ),
                decommit_hold_max_ticks=getattr(
                    config, "closure_decommit_hold_max_ticks", 0
                ),
                # ARC-108 JOB-2 (d): habenula negative-RPE de-commit abort input.
                # getattr fallback keeps from_dims-absent callers bit-identical.
                habenula_abort_enabled=getattr(
                    config, "use_habenula_decommit", False
                ),
                habenula_delta_threshold=getattr(
                    config, "habenula_decommit_delta_threshold", 0.0
                ),
            )
            self.closure_operator = ClosureOperator(
                config=clo_cfg,
                beta_gate=self.beta_gate,
                dacc=self.dacc,
                residue=self.residue_field,
                salience=self.salience,
                lateral_pfc=self.lateral_pfc,
            )
            if (
                self.salience is not None
                and config.closure_signal_affinity_internal_planning > 0.0
            ):
                self.closure_operator.register_on_coordinator(
                    affinity_modes={
                        "internal_planning": config.closure_signal_affinity_internal_planning,
                    },
                )

        # SD-035: Amygdala analogue (BLAAnalog + CeAAnalog peer modules).
        # Master switch use_amygdala_analog gates both modules; per-module
        # switches use_bla_analog / use_cea_analog give granular control.
        # Both modules are non-trainable arithmetic; they read z_harm_a
        # (SD-011) and write to different downstream consumers:
        #   BLAAnalog -> encoding_gain / retrieval_bias / remap_signal
        #                (MECH-074a/b/d; hippocampal consumer wiring is a
        #                follow-up pass -- see TODO markers below and in
        #                select_action).
        #   CeAAnalog -> cea_mode_prior / cea_fast_prime
        #                (MECH-046 / MECH-074c; injected into
        #                SalienceCoordinator via update_signal below).
        self.bla: Optional[BLAAnalog] = None
        self.cea: Optional[CeAAnalog] = None
        if getattr(config, "use_amygdala_analog", False):
            if getattr(config, "use_bla_analog", True):
                bla_cfg = BLAConfig(
                    encoding_gain_max=config.bla_encoding_gain_max,
                    encoding_gain_floor=config.bla_encoding_gain_floor,
                    arousal_threshold_on=config.bla_arousal_threshold_on,
                    arousal_peak=config.bla_arousal_peak,
                    window_steps=config.bla_window_steps,
                    window_half_life_steps=config.bla_window_half_life_steps,
                    retrieval_bias_alpha=config.bla_retrieval_bias_alpha,
                    retrieval_bias_compensation=config.bla_retrieval_bias_compensation,
                    retrieval_tag_at_encoding=config.bla_retrieval_tag_at_encoding,
                    remap_pe_sigma_threshold=config.bla_remap_pe_sigma_threshold,
                    remap_pe_ema_alpha=config.bla_remap_pe_ema_alpha,
                    remap_pe_std_init=config.bla_remap_pe_std_init,
                    remap_code_fraction=config.bla_remap_code_fraction,
                    remap_requires_attribution=config.bla_remap_requires_attribution,
                )
                self.bla = BLAAnalog(bla_cfg)
            if getattr(config, "use_cea_analog", True):
                cea_cfg = CeAConfig(
                    fast_route_threshold=config.cea_fast_route_threshold,
                    fast_route_input_is_lowfreq=config.cea_fast_route_input_is_lowfreq,
                    mode_prior_log_odds_max=config.cea_mode_prior_log_odds_max,
                    mode_prior_gain=config.cea_mode_prior_gain,
                    pre_softmax_additive=config.cea_pre_softmax_additive,
                    fast_prime_amplitude=config.cea_fast_prime_amplitude,
                    fast_prime_decay_tau_steps=config.cea_fast_prime_decay_tau_steps,
                    fast_prime_override_window_steps=config.cea_fast_prime_override_window_steps,
                    cortical_confirmation_weight=config.cea_cortical_confirmation_weight,
                )
                self.cea = CeAAnalog(cea_cfg)
            # Register CeA signals as salience/affinity sources on the
            # coordinator if one exists. cea_mode_prior is an affinity
            # signal (biases the operating-mode distribution toward harm-
            # relevant modes). cea_fast_prime is a salience signal (a
            # salience-aggregate contribution for the MECH-259 trigger).
            # Weights are conservative defaults; experiments can tune
            # via direct SalienceCoordinatorConfig edits.
            if self.salience is not None and self.cea is not None:
                self.salience.config.affinity_weights["cea_mode_prior"] = {
                    "external_task": 1.0,
                }
                self.salience.config.salience_weights["cea_fast_prime"] = 0.5
        # Cache of last amygdala tick outputs (for diagnostics/experiments).
        self._bla_last_output: Optional[BLAOutput] = None
        self._cea_last_output: Optional[CeAOutput] = None

        # SD-036: GABAergic cross-stream decay regulator.
        # When use_gabaergic_decay=True, instantiate a regulator with the three
        # default streams (z_harm, z_harm_a, z_beta) registered. Ticks once per
        # sense() after LatentStack.encode() and before mode-arbitration
        # consumers read the latent state. Backward-compatible no-op when the
        # master switch is off.
        self.gabaergic_decay: Optional[GABAergicDecayRegulator] = None
        if getattr(config, "use_gabaergic_decay", False):
            gaba_cfg = GABAergicDecayConfig(
                enabled=True,
                gaba_tone=float(getattr(config, "gaba_tone", 1.0)),
                gaba_tone_min=float(getattr(config, "gaba_tone_min", 0.0)),
                gaba_tone_max=float(getattr(config, "gaba_tone_max", 2.0)),
                tau_z_harm_s=float(getattr(config, "gaba_tau_z_harm_s", 0.05)),
                tau_z_harm_a=float(getattr(config, "gaba_tau_z_harm_a", 0.02)),
                tau_z_beta=float(getattr(config, "gaba_tau_z_beta", 0.03)),
                decay_z_harm_s=bool(getattr(config, "gaba_decay_z_harm_s", True)),
                decay_z_harm_a=bool(getattr(config, "gaba_decay_z_harm_a", True)),
                decay_z_beta=bool(getattr(config, "gaba_decay_z_beta", True)),
                input_threshold_z_harm_s=float(
                    getattr(config, "gaba_input_threshold_z_harm_s", 0.0)
                ),
                input_threshold_z_harm_a=float(
                    getattr(config, "gaba_input_threshold_z_harm_a", 0.0)
                ),
                input_threshold_z_beta=float(
                    getattr(config, "gaba_input_threshold_z_beta", 0.0)
                ),
            )
            self.gabaergic_decay = GABAergicDecayRegulator(gaba_cfg)
            self.gabaergic_decay.register_default_streams(gaba_cfg)

        # MECH-279: PAG freeze-gate. When use_pag_freeze_gate=True, the gate
        # ticks each select_action() and constrains the action selector to a
        # no-op action class while freeze_active. Exit threshold uses the
        # SD-036 gaba_tone (theta_freeze * gaba_tone). Backward-compatible
        # no-op when the master switch is off.
        self.pag_freeze_gate: Optional[PAGFreezeGate] = None
        if getattr(config, "use_pag_freeze_gate", False):
            pag_cfg = PAGFreezeGateConfig(
                enabled=True,
                theta_freeze=float(getattr(config, "pag_theta_freeze", 2.0)),
                duration_input_threshold=float(
                    getattr(config, "pag_duration_input_threshold", 0.4)
                ),
                min_freeze_duration=int(
                    getattr(config, "pag_min_freeze_duration", 0)
                ),
                max_freeze_duration=int(
                    getattr(config, "pag_max_freeze_duration", 0)
                ),
                # SD-037: broadcast-override scaling on theta_freeze. Only takes
                # effect when use_broadcast_override is also True (override_signal
                # stays at 0.0 otherwise).
                alpha_override=float(getattr(config, "override_alpha_pag", 0.5))
                if getattr(config, "use_broadcast_override", False)
                else 0.0,
            )
            self.pag_freeze_gate = PAGFreezeGate(pag_cfg)
        # Cache of last PAG freeze-gate output (diagnostics).
        self._pag_last_output: Optional[PAGFreezeGateOutput] = None

        # SD-037: Broadcast Override Regulator (orexin-analog). Drives a scalar
        # override_signal in [0, 1] from drive_level + sustained-threat magnitude
        # over z_harm. Consumed by PAG freeze-gate (theta_freeze scaling),
        # SalienceCoordinator (operating-mode reweight), and GoalState (drive ->
        # z_goal seeding gate). Backward-compatible no-op when master switch off.
        self.broadcast_override: Optional[BroadcastOverrideRegulator] = None
        if getattr(config, "use_broadcast_override", False):
            override_cfg = BroadcastOverrideConfig(
                enabled=True,
                recruitment_threshold=float(
                    getattr(config, "override_recruitment_threshold", 0.5)
                ),
                alpha_override=float(getattr(config, "override_alpha_pag", 0.5)),
                salience_reweight_alpha=float(
                    getattr(config, "override_salience_reweight_alpha", 0.3)
                ),
                drive_weight=float(getattr(config, "override_drive_weight", 1.0)),
                harm_weight=float(getattr(config, "override_harm_weight", 1.0)),
                sustained_threat_window=int(
                    getattr(config, "override_sustained_threat_window", 12)
                ),
                sustained_threat_threshold=float(
                    getattr(config, "override_sustained_threat_threshold", 0.4)
                ),
                decay_rate=float(getattr(config, "override_decay_rate", 0.05)),
            )
            self.broadcast_override = BroadcastOverrideRegulator(override_cfg)
            # Register SD-037 override_signal as a SalienceCoordinator signal
            # source. Recruited override biases the operating-mode aggregate
            # toward external_task (engaged action). Backward-compatible no-op
            # when salience is None or the alpha is 0.
            if self.salience is not None:
                self.salience.config.affinity_weights["override_signal"] = {
                    "external_task": float(
                        getattr(config, "override_salience_reweight_alpha", 0.3)
                    ),
                }

        # mode-governance-engagement substrate (2026-06-13): register the
        # external_task_drive signal slot on the SalienceCoordinator. Registered in
        # BOTH affinity_weights (-> external_task, mode SELECTION pull) AND
        # salience_weights (-> aggregate, so a switch INTO external_task can fire the
        # MECH-259 trigger). The engagement scalar is injected per-tick in
        # select_action (see the tick block below). No-op default: slot registered
        # only when use_external_task_drive=True, so OFF leaves the coordinator
        # bit-identical (the signal name never enters _input_signals).
        if getattr(config, "use_external_task_drive", False) and self.salience is not None:
            self.salience.config.affinity_weights["external_task_drive"] = {
                "external_task": float(
                    getattr(config, "external_task_drive_affinity_weight", 1.0)
                ),
            }
            self.salience.config.salience_weights["external_task_drive"] = float(
                getattr(config, "external_task_drive_salience_weight", 1.0)
            )

        # MECH-282: LPB interoceptive harm routing (parallel to external z_harm).
        self.lpb_router: Optional[LPBInteroceptiveRouter] = None
        self._lpb_last_output: Optional[LPBInteroceptiveRoutingOutput] = None
        if getattr(config, "use_lpb_interoceptive_routing", False):
            lpb_cfg = LPBInteroceptiveRoutingConfig(
                enabled=True,
                intero_z_dim=int(getattr(config, "lpb_intero_z_dim", 16)),
                drive_weight=float(getattr(config, "lpb_drive_weight", 1.0)),
                resource_weight=float(getattr(config, "lpb_resource_weight", 1.0)),
            )
            self.lpb_router = LPBInteroceptiveRouter(lpb_cfg)

        # MECH-295: drive -> liking-stream -> approach_cue bridge (weak reading).
        # The bridge wires SD-012 drive amplification through the liking-stream
        # (anticipatory write at the goal location, gated by drive * z_goal_norm)
        # and into action selection (per-candidate approach_cue_signal supplied
        # to E3 as a negative score_bias). Without this bridge, drive amplification
        # produces a passive z_goal latent without behavioural consequence -- the
        # EXQ-483 catatonic-lock signature (override fires, PAG releases up,
        # approach_commit = 0.0 across all arms).
        #
        # Read-site contract (important): both bridge call sites
        # (compute_anticipatory_liking_write at _e3_tick and
        # compute_approach_cue_score_bias at select_action) gate on
        # self.goal_state.is_active() AND self.goal_state.goal_norm() >=
        # mech295_min_z_goal_norm_to_fire. Both predicates read the PERSISTENT
        # attractor (self.goal_state._z_goal). They do NOT see the action-time
        # MECH-188 inject (cfg.goal.z_goal_inject), which only constructs
        # _goal_state_for_select via with_injection() for E3.select() and does
        # not mutate the persistent attractor. EXQ-536b confirmed this dual-
        # path: with z_goal_inject=0.3, inject_observed_fraction=1.0 at the
        # E3 per-candidate read site but approach_commit_rate=0.0 because the
        # bridge gate sees goal_norm=0.0 on the persistent attractor and
        # short-circuits. Force-arm tests of MECH-295 must seed
        # self.goal_state._z_goal directly (or set
        # mech295_min_z_goal_norm_to_fire below the inject floor); see
        # REE_assembly/docs/architecture/mech188_vs_mech295_dual_path.md
        # for the disambiguation and the EXQ-536c/d/e proposal.
        # See REE_assembly/docs/architecture/mech_295_drive_liking_approach_bridge.md.
        # Backward-compatible no-op when master switch off.
        self.mech295_bridge: Optional[MECH295LikingBridge] = None
        if getattr(config, "use_mech295_liking_bridge", False):
            bridge_cfg = MECH295LikingBridgeConfig(
                drive_to_liking_gain=float(
                    getattr(config, "mech295_drive_to_liking_gain", 1.0)
                ),
                liking_to_approach_cue_gain=float(
                    getattr(config, "mech295_liking_to_approach_cue_gain", 0.5)
                ),
                min_drive_to_fire=float(
                    getattr(config, "mech295_min_drive_to_fire", 0.01)
                ),
                min_z_goal_norm_to_fire=float(
                    getattr(config, "mech295_min_z_goal_norm_to_fire", 0.05)
                ),
                # MECH-307 Path B: consumer-side conjunction read.
                use_mech307_conjunction_read=bool(
                    getattr(config, "use_mech307_consumer_conjunction_read", False)
                ),
                mech307_conjunction_wanting_threshold=float(
                    getattr(config, "mech307_conjunction_wanting_threshold", 0.6)
                ),
                mech307_conjunction_liking_threshold=float(
                    getattr(config, "mech307_conjunction_liking_threshold", 0.3)
                ),
                mech307_conjunction_z_beta_threshold=float(
                    getattr(config, "mech307_conjunction_z_beta_threshold", 0.3)
                ),
                mech307_conjunction_gain=float(
                    getattr(config, "mech307_conjunction_gain", 1.0)
                ),
            )
            self.mech295_bridge = MECH295LikingBridge(bridge_cfg)

        # MECH-269b: Symmetric V_s gating on E1/E2 cortical rollouts.
        # Read-side consumer of MECH-269 Phase 1 per_stream_vs. Snapshots per-
        # stream latent values when V_s is at or above
        # vs_gate_snapshot_refresh_threshold and substitutes the held snapshot
        # at E1 / E2_harm_a forward call sites when V_s falls below the per-
        # side threshold. Prevents cortical predictors from rolling forward
        # off stale-but-confident-looking inputs (the EXQ-483 wired-but-inert
        # failure mode).
        self.vs_rollout_gate = None
        if getattr(self.hippocampal.config, "use_vs_rollout_gating", False):
            if not getattr(self.hippocampal.config, "use_per_stream_vs", False):
                raise ValueError(
                    "use_vs_rollout_gating=True requires use_per_stream_vs=True "
                    "(the gate consumes hippocampal.per_stream_vs)."
                )
            from ree_core.regulators.vs_rollout_gate import (
                VsRolloutGate, VsRolloutGateConfig,
            )
            gate_cfg = VsRolloutGateConfig(
                streams=tuple(
                    getattr(
                        self.hippocampal.config,
                        "vs_gate_streams",
                        ("z_world", "z_self", "z_harm_s",
                         "z_harm_a", "z_goal", "z_beta"),
                    )
                ),
                snapshot_refresh_threshold=float(
                    getattr(
                        self.hippocampal.config,
                        "vs_gate_snapshot_refresh_threshold",
                        0.5,
                    )
                ),
                e1_threshold=float(
                    getattr(self.hippocampal.config, "vs_gate_e1_threshold", 0.4)
                ),
                e2_threshold=float(
                    getattr(self.hippocampal.config, "vs_gate_e2_threshold", 0.4)
                ),
                unknown_stream_passes=bool(
                    getattr(
                        self.hippocampal.config,
                        "vs_gate_unknown_stream_passes",
                        True,
                    )
                ),
                use_staleness_lookup=bool(
                    getattr(
                        self.hippocampal.config,
                        "use_vs_gate_staleness_lookup",
                        False,
                    )
                ),
            )
            # Q-040b strong reading: when use_vs_gate_staleness_lookup is on,
            # the gate consumes per-stream staleness aggregated by
            # HippocampalModule.compute_per_stream_staleness, which itself
            # requires use_staleness_accumulator AND use_anchor_sets. Loud
            # failure on the missing precondition keeps experiment configs
            # honest -- no silent passthrough into the legacy raw-V_s path.
            if gate_cfg.use_staleness_lookup:
                if not getattr(
                    self.hippocampal.config, "use_staleness_accumulator", False
                ):
                    raise ValueError(
                        "use_vs_gate_staleness_lookup=True requires "
                        "use_staleness_accumulator=True (the gate consumes "
                        "MECH-284 region staleness)."
                    )
                if not getattr(
                    self.hippocampal.config, "use_anchor_sets", False
                ):
                    raise ValueError(
                        "use_vs_gate_staleness_lookup=True requires "
                        "use_anchor_sets=True (the per-stream aggregator "
                        "walks active anchors)."
                    )
            self.vs_rollout_gate = VsRolloutGate(gate_cfg)
            # Cache for the per-tick aggregator output. Populated by
            # _refresh_vs_gate_staleness() before every gate / gate_stream
            # call. Empty dict means raw-V_s path (no staleness available
            # this tick).
            self._vs_gate_staleness_cache: Dict[str, float] = {}

        # Sleep-aggregation cluster Phase A: deterministic K-episode driver
        # for the existing SD-017 surface (run_sleep_cycle). Wraps, does not
        # replace. Phases B-E (replay sampler, routing gate, Bayesian
        # aggregator, self-model writeback) extend this manager via additional
        # master flags. Backward-compatible no-op when use_sleep_loop=False.
        self.sleep_loop: Optional[SleepLoopManager] = None
        self.sleep_replay_sampler = None  # Phase B (MECH-285)
        self.sleep_routing_gate = None  # Phase C (MECH-272)
        self.sleep_bayesian_aggregator = None  # Phase D (MECH-275)
        self.sleep_self_model_aggregator = None  # Phase E (MECH-273)

        # MECH-423 R3: module-tagged interleaved cross-module consolidation.
        # Built independent of use_sleep_loop so an experiment can drive it
        # standalone via agent.cross_module_consolidator; also passed to the
        # SleepLoopManager below when the sleep loop is active. None (default)
        # -> bit-identical: nothing in the waking or sleep pipeline references it.
        self.cross_module_consolidator = None
        if getattr(config, "use_cross_module_consolidation", False):
            from ree_core.sleep.cross_module_consolidation import (
                CrossModuleConsolidator,
                CrossModuleConsolidatorConfig,
            )
            self.cross_module_consolidator = CrossModuleConsolidator(
                CrossModuleConsolidatorConfig(
                    schedule=str(
                        getattr(
                            config,
                            "cross_module_consolidation_schedule",
                            "interleaved",
                        )
                    ),
                    n_steps=int(
                        getattr(config, "cross_module_consolidation_steps", 0)
                    ),
                    lr=float(
                        getattr(config, "cross_module_consolidation_lr", 1e-3)
                    ),
                )
            )

        if getattr(config, "use_sleep_loop", False):
            # Phase B: when use_mech285_sampler is on AND anchor_set exists,
            # construct SleepReplaySampler over the broad pool. Falls back
            # to None silently when anchor_set is absent (Phase B requires
            # MECH-269 Phase 2 ii to be wired). StalenessAccumulator is
            # optional -- the sampler runs in uniform-fallback mode when
            # absent (controlled by mech285_allow_uniform_fallback).
            if (
                getattr(config, "use_mech285_sampler", False)
                and getattr(self.hippocampal, "anchor_set", None) is not None
            ):
                from ree_core.sleep.replay_sampler import SleepReplaySampler
                self.sleep_replay_sampler = SleepReplaySampler(
                    anchor_set=self.hippocampal.anchor_set,
                    staleness_accumulator=getattr(
                        self.hippocampal, "staleness_accumulator", None
                    ),
                    temperature=float(
                        getattr(config, "mech285_temperature", 1.0)
                    ),
                    allow_uniform_fallback=bool(
                        getattr(config, "mech285_allow_uniform_fallback", True)
                    ),
                )
            # Phase C: when use_mech272_routing is on, construct RoutingGate
            # with per-phase weights from config. The gate is wired into
            # SleepLoopManager so SLEEP_ENTRY -> SWS row and PHASE_SWITCH ->
            # REM row are flipped automatically; route() is called per draw.
            # Bit-identical OFF when use_mech272_routing=False (gate is None).
            if getattr(config, "use_mech272_routing", False):
                from ree_core.sleep.routing_gate import (
                    RoutingGate,
                    RoutingGateConfig,
                )
                self.sleep_routing_gate = RoutingGate(
                    RoutingGateConfig(
                        waking_anchor_weight=float(
                            getattr(config, "mech272_waking_anchor_weight", 1.0)
                        ),
                        waking_probe_weight=float(
                            getattr(config, "mech272_waking_probe_weight", 0.0)
                        ),
                        sws_anchor_weight=float(
                            getattr(config, "mech272_sws_anchor_weight", 0.6)
                        ),
                        sws_probe_weight=float(
                            getattr(config, "mech272_sws_probe_weight", 0.4)
                        ),
                        rem_anchor_weight=float(
                            getattr(config, "mech272_rem_anchor_weight", 0.2)
                        ),
                        rem_probe_weight=float(
                            getattr(config, "mech272_rem_probe_weight", 0.8)
                        ),
                    )
                )
            # Phase D: when use_mech275_aggregator is on, construct the
            # general Bayesian aggregator. Posterior updates are gated on
            # the routing-gate probe channel (so the gate must also be on
            # for the aggregator to fire). Bit-identical OFF when
            # use_mech275_aggregator=False (aggregator is None).
            if getattr(config, "use_mech275_aggregator", False):
                from ree_core.sleep.bayesian_aggregator import (
                    BayesianAggregator,
                    BayesianAggregatorConfig,
                )
                self.sleep_bayesian_aggregator = BayesianAggregator(
                    BayesianAggregatorConfig(
                        domains=tuple(
                            getattr(config, "mech275_domains", ("place",))
                        ),
                        prior_mean=float(
                            getattr(config, "mech275_prior_mean", 0.0)
                        ),
                        prior_variance=float(
                            getattr(config, "mech275_prior_variance", 1.0)
                        ),
                        likelihood_variance=float(
                            getattr(config, "mech275_likelihood_variance", 1.0)
                        ),
                        decay_factor=float(
                            getattr(config, "mech275_decay_factor", 1.0)
                        ),
                        probe_gain=float(
                            getattr(config, "mech275_probe_gain", 1.0)
                        ),
                    )
                )
            # Phase E: when use_mech273_self_model is on, construct the
            # SelfModelAggregator (subclass of BayesianAggregator specialised
            # on the SD-003 causal_sig posterior). Requires e2_harm_s on the
            # agent (ARC-033). Bit-identical OFF when use_mech273_self_model
            # is False (aggregator is None; SleepLoopManager skips WRITEBACK).
            if (
                getattr(config, "use_mech273_self_model", False)
                and getattr(self, "e2_harm_s", None) is not None
            ):
                from ree_core.sleep.self_model_aggregator import (
                    SelfModelAggregator,
                    SelfModelAggregatorConfig,
                )
                self.sleep_self_model_aggregator = SelfModelAggregator(
                    SelfModelAggregatorConfig(
                        domains=("self",),
                        prior_mean=float(
                            getattr(config, "mech275_prior_mean", 0.0)
                        ),
                        prior_variance=float(
                            getattr(config, "mech275_prior_variance", 1.0)
                        ),
                        likelihood_variance=float(
                            getattr(config, "mech275_likelihood_variance", 1.0)
                        ),
                        decay_factor=float(
                            getattr(config, "mech275_decay_factor", 1.0)
                        ),
                        probe_gain=float(
                            getattr(config, "mech275_probe_gain", 1.0)
                        ),
                        offline_lr_scale=float(
                            getattr(config, "mech273_offline_lr_scale", 0.1)
                        ),
                        offline_n_steps=int(
                            getattr(config, "mech273_offline_n_steps", 100)
                        ),
                    )
                )
            self.sleep_loop = SleepLoopManager(
                cycle_every_k_episodes=int(
                    getattr(config, "sleep_loop_episodes_K", 1)
                ),
                require_sleep_passes_enabled=bool(
                    getattr(config, "sleep_loop_require_passes", True)
                ),
                replay_sampler=self.sleep_replay_sampler,
                draws_per_cycle=int(
                    getattr(config, "mech285_draws_per_cycle", 0)
                ) if self.sleep_replay_sampler is not None else 0,
                routing_gate=self.sleep_routing_gate,
                bayesian_aggregator=self.sleep_bayesian_aggregator,
                aggregator_domain=str(
                    getattr(config, "mech275_aggregator_domain", "place")
                ),
                self_model_aggregator=self.sleep_self_model_aggregator,
                self_model_offline_n_steps=int(
                    getattr(config, "mech273_offline_n_steps", 100)
                ),
                self_model_partial_decay_factor=float(
                    getattr(config, "mech273_partial_decay_factor", 0.5)
                ),
                self_model_domain="self",
                use_rem_precision_recalibration=bool(
                    getattr(config, "use_rem_precision_recalibration", False)
                ),
                rem_precision_recalibration_step=float(
                    getattr(config, "rem_precision_recalibration_step", 0.1)
                ),
                use_mech272_routing_consumer=bool(
                    getattr(config, "use_mech272_routing_consumer", False)
                ),
                # MECH-423 R3: pass the consolidator + schedule knobs so the
                # cross-module consolidation runs inside the live MECH-121
                # sleep pipeline. None -> the SleepLoopManager hook is skipped.
                cross_module_consolidator=self.cross_module_consolidator,
                cross_module_consolidation_steps=int(
                    getattr(config, "cross_module_consolidation_steps", 0)
                ),
                cross_module_consolidation_schedule=str(
                    getattr(
                        config, "cross_module_consolidation_schedule", "interleaved"
                    )
                ),
                cross_module_consolidation_lr=float(
                    getattr(config, "cross_module_consolidation_lr", 1e-3)
                ),
                cross_module_consolidation_batch=int(
                    getattr(config, "cross_module_consolidation_batch", 16)
                ),
            )

        # Observation encoders (maps raw body/world obs to latent input)
        # Body encoder: body_obs_dim → latent input for LatentStack
        self.body_obs_encoder = nn.Sequential(
            nn.Linear(config.latent.body_obs_dim, config.latent.body_obs_dim),
            nn.ReLU(),
        )
        self.world_obs_encoder = nn.Sequential(
            nn.Linear(config.latent.world_obs_dim, config.latent.world_obs_dim),
            nn.ReLU(),
        )

        # State tracking
        self._current_latent: Optional[LatentState] = None
        self._step_count: int = 0
        self._harm_this_episode: float = 0.0

        # Experience buffers for training
        self._self_experience_buffer: List[torch.Tensor] = []   # z_self history
        self._world_experience_buffer: List[torch.Tensor] = []  # z_world history
        self._e2_transition_buffer: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        # GAP-4 / MECH-273: waking-stream (z_harm_s, action) pairs for sleep WRITEBACK.
        self._harm_replay_buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []

        # MECH-423 R2: cached iterative-inference convergence readout (None unless
        # use_iterative_inference; populated each sense() tick from the LatentState).
        self.last_inference_convergence: Optional[Dict[str, object]] = None

        # MECH-057a: cached E3 candidates for action-loop gate
        self._committed_candidates: Optional[List[Trajectory]] = None
        # MECH-340 / Q-053: last appraisal passed into ghost bank (diagnostics).
        self._last_persistence_appraisal: Optional[PersistenceAppraisal] = None
        self._last_candidate_support_preflight: Dict[str, Any] = {}
        # Last selected action (held between E3 ticks)
        self._last_action: Optional[torch.Tensor] = None
        self._last_e3_selection_result: Optional[Any] = None
        self._last_e3_score_bias: Optional[torch.Tensor] = None
        # DR-12 (self_model_v4:SELF-4): optional injected per-candidate E2 forward-PE
        # [K] for the E3 confidence down-weight. None -> bit-identical (no penalty).
        # Set per tick via set_injected_e2_forward_pe(); the lever applies it only
        # when E3Config.use_pe_confidence_weighting is True.
        self._injected_e2_forward_pe: Optional[torch.Tensor] = None
        # MECH-449 (ARC-107): optional injected per-candidate Go/No-Go signals
        # (dict of [K] tensors keyed safety/staleness/perseveration/viability/go)
        # for the eligibility constitution. The MECH-449 falsifier sets the
        # constructed-bank axes per tick via set_injected_go_nogo_signals(); the
        # default waking loop wires only the MECH-260 perseveration reuse. None ->
        # bit-identical (the gate is also master-gated by use_go_nogo_constitution).
        self._injected_go_nogo_signals: Optional[Dict[str, Any]] = None
        # V3-EXQ-571: per-component bias decomposition (written when e3.e3_score_decomp_enabled)
        self._last_score_bias_decomp: dict = {}
        # ControlVector logging (rec-B four-signal adjudication 2026-06-07):
        # read-only per-tick telemetry of the four control signals
        # (V_outcome / C_effort / C_time / G_vigor), written when
        # config.use_control_vector_logging. Read directly by experiment
        # scripts (the _last_score_bias_decomp pattern). _cv_vigor holds the
        # MECH-320 action/no-op split cached in the tonic_vigor block.
        self._last_control_vector: dict = {}
        self._cv_vigor: Optional[dict] = None
        # MECH-090: step index within committed trajectory (Layer 1 trajectory stepping).
        # Incremented each committed step so a0->a1->a2->... is executed in sequence.
        self._committed_step_idx: int = 0

        # MECH-269 / MECH-090 read-side hook: V_s -> commit release.
        # Snapshot of AnchorSet active_anchor keys at commit entry. None when
        # uncommitted or when use_vs_commit_release is False. Cleared on
        # release. Used by select_action() to detect anchor invalidation:
        # if any snapshot key has dropped out of the current active set, the
        # commitment is released.
        self._committed_anchor_keys: Optional[set] = None

        # SD-016: cached frontal cue signals (updated each E1 tick).
        # None when sd016_enabled=False (all existing experiments unaffected).
        self._cue_action_bias:    Optional[torch.Tensor] = None
        self._cue_terrain_weight: Optional[torch.Tensor] = None

        self.device = torch.device(config.device)

        # MECH-216: cached schema salience from E1 readout head
        self._schema_salience: Optional[torch.Tensor] = None

        # MECH-307 Gap 4: cached E1 forward prediction (z_world predicted) used
        # as write target for schema-readout VALENCE_WANTING / LIKING writes
        # when use_mech307_predicted_location_write=True. Set in _e1_tick after
        # e1_prior is computed; falls back to current z_world when None.
        self._cached_e1_prior: Optional[torch.Tensor] = None

        # MECH-205: surprise-gated replay PE tracking
        self._pe_ema: float = 0.0  # EMA of prediction error magnitude
        self._pe_ema_alpha: float = config.pe_ema_alpha  # from config (default 0.02)
        self._surprise_write_count: int = 0  # diagnostic counter

        # SD-020: harm PE tracker for affective surprise target.
        # Running EMA of observed harm (expected harm estimate).
        # PE = |actual_harm_obs - _harm_obs_ema| used as z_harm_a training target.
        self._harm_obs_ema: float = 0.0

        # SD-019a: harm_unpleasantness_channel EMA buffer.
        # Non-trainable stateful EMA of z_harm_s; same dim as z_harm_s.
        # Reset per-episode (None = no history yet this episode).
        self._harm_un_ema: Optional[torch.Tensor] = None

        # SD-049 Phase 3: per-axis drive vector from the env's
        # obs_dict["per_axis_drive"]. None when the env does not surface
        # per-axis drive (SD-049 OFF) OR when sense() is called without
        # obs_per_axis_drive. Cleared in reset() so cross-episode leakage
        # is impossible. Threaded through to all 7 SD-032 consumer tick
        # call sites by the Phase 3 _per_axis_drive_for_consumers helper.
        self._per_axis_drive: Optional[torch.Tensor] = None

        # MECH-258 / SD-032b: previous-step z_harm_a for E2_harm_a rollout
        # and previous-step predicted z_harm_a for dACC PE computation.
        self._harm_a_prev: Optional[torch.Tensor] = None
        self._harm_a_pred_prev: Optional[torch.Tensor] = None
        # Diagnostic: last dACC bundle (for experiments).
        self._dacc_last_bundle: Optional[Dict[str, Any]] = None
        self._dacc_last_bias: Optional[torch.Tensor] = None

        # MECH-095: pending TPJ efference-copy prediction and resolved agency
        # readout for the most recently observed transition.
        self._tpj_predicted_z_self: Optional[torch.Tensor] = None
        self._tpj_last_agency_signal: Optional[torch.Tensor] = None
        self._tpj_last_is_self_caused: Optional[torch.Tensor] = None

        # MECH-165: episode trajectory recording for exploration buffer
        self._episode_world_states: List[torch.Tensor] = []
        self._episode_self_states: List[torch.Tensor] = []
        self._episode_actions: List[torch.Tensor] = []
        self._episode_bla_peak_tag: float = 0.0
        self._episode_bla_peak_encoding_gain: float = 1.0

        # MECH-165: pass config to hippocampal for buffer sizing
        if self.config.replay_diversity_enabled:
            self.hippocampal._exploration_buffer_maxlen = self.config.exploration_buffer_len
            self.hippocampal._reverse_fraction = self.config.reverse_replay_fraction
            self.hippocampal._random_fraction = self.config.random_replay_fraction

        # MECH-112/116: persistent goal representation
        self.goal_state: Optional[GoalState] = None
        _goal_cfg = getattr(self.config, "goal", None)
        if _goal_cfg is not None and _goal_cfg.z_goal_enabled:
            self.goal_state = GoalState(_goal_cfg, self.device)

        # MECH-189: super-ordinal goal-anchor ContextMemory writes substrate
        # (infant_substrate:GAP-11 / DEV-NEED-006). AGENT-owned and NOT reset
        # per episode (cross-episode persistence is the point of a super-ordinal
        # goal hierarchy). Requires z_goal_enabled (the store keys on z_world and
        # stores z_goal anchors). Bit-identical OFF when the flag is unset.
        self.super_ordinal_goal_memory: Optional[SuperOrdinalGoalMemory] = None
        if (
            self.goal_state is not None
            and getattr(_goal_cfg, "use_super_ordinal_goal_anchors", False)
        ):
            self.super_ordinal_goal_memory = SuperOrdinalGoalMemory(
                _goal_cfg,
                context_dim=self.config.latent.world_dim,
                device=self.device,
            )

    @classmethod
    def from_config(
        cls,
        body_obs_dim: int,
        world_obs_dim: int,
        action_dim: int,
        self_dim: int = 32,
        world_dim: int = 32,
        **kwargs,
    ) -> "REEAgent":
        config = REEConfig.from_dims(
            body_obs_dim=body_obs_dim,
            world_obs_dim=world_obs_dim,
            action_dim=action_dim,
            self_dim=self_dim,
            world_dim=world_dim,
        )
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return cls(config)

    def _eal_linker_context_feature(self) -> Optional[torch.Tensor]:
        """Optional E2-linker representation handed to the relief/safety heads.

        Returns the E2EscapeAffordanceLinker's escape_affordance_features for a
        representative non-noop class at the current state, but ONLY when
        ``use_e2_escape_linker_for_relief_safety`` is enabled and the linker
        model has been instantiated. Returns None otherwise, so the
        TrainableEscapeAffordanceLearner stays bit-identical by default and the
        linker only ever provides representational substrate (the heads keep
        their relief/safety-specific labels). The linker update runs before the
        learner update in sense(), so the linker model already exists by the time
        the learner sees these features on the first trainable transition.
        """
        if not getattr(self.config, "use_e2_escape_linker_for_relief_safety", False):
            return None
        linker = getattr(self, "e2_escape_affordance_linker", None)
        if linker is None or linker.model is None:
            return None
        n = max(1, int(self.config.e2.action_dim))
        noop = int(getattr(self.config, "escape_linker_noop_class", 0))
        rep = next((c for c in range(n) if c != noop), 0)
        return linker.escape_affordance_features(rep)

    def reset(self) -> None:
        """Reset agent for a new episode. Does NOT reset residue (invariant)."""
        # MECH-165: flush waking trajectory to exploration buffer before reset
        self._flush_exploration_episode()

        # Sleep-aggregation cluster Phase A: notify the SleepLoopManager that
        # an episode has just ended. The manager fires a sleep cycle every
        # K episodes through the existing SD-017 surface. Runs BEFORE the
        # per-episode resets below so sleep operates on the final waking
        # state of the episode that just completed (theta_buffer / experience
        # buffers still populated). No-op when use_sleep_loop=False.
        if self.sleep_loop is not None:
            self.sleep_loop.notify_episode_end(self)

        self._current_latent = self.latent_stack.init_state(
            batch_size=1, device=self.device
        )
        self.e1.reset_hidden_state()
        self._step_count = 0
        self._harm_this_episode = 0.0
        self._committed_candidates = None
        self._last_persistence_appraisal = None
        self._harm_replay_buffer = []
        self._last_action = None
        self._last_e3_selection_result = None
        self._last_e3_score_bias = None
        self._committed_step_idx = 0
        # ARC-108 JOB-1 step-1: clear the within-episode learned-channel-gating
        # eligibility trace + pending flag (w_chan / V-hat_t persist across episodes).
        self.e3.clear_learned_channel_eligibility()
        # MECH-269 / MECH-090 read-side hook: clear V_s -> commit release snapshot.
        self._committed_anchor_keys = None
        self._cue_action_bias    = None
        self._cue_terrain_weight = None
        self.clock.reset()
        self.theta_buffer.reset()
        # MECH-294: clear the multi-content packet window / snapshots / history.
        if self.multi_content_theta_packet is not None:
            self.multi_content_theta_packet.reset()
            self.last_theta_packet = None
        self.beta_gate.reset()
        self.serotonin.reset()
        self._pe_ema = 0.0
        # MECH-307 Gap 4: clear cached E1 prior on episode boundary.
        self._cached_e1_prior = None
        # SD-019a: reset harm_unpleasantness EMA on episode boundary.
        self._harm_un_ema = None
        # SD-049 Phase 3: clear per-axis drive cache on episode boundary
        # (sense() repopulates from obs_per_axis_drive on the first tick).
        self._per_axis_drive = None
        # SD-032b: clear dACC state + previous-step harm_a cache.
        self._harm_a_prev = None
        self._harm_a_pred_prev = None
        self._dacc_last_bundle = None
        self._dacc_last_bias = None
        self._tpj_predicted_z_self = None
        self._tpj_last_agency_signal = None
        self._tpj_last_is_self_caused = None
        if self.dacc is not None:
            self.dacc.reset()
        # SD-032a: reset salience coordinator on episode boundary.
        if self.salience is not None:
            self.salience.reset()
        self._salience_last_tick = None

        # SD-032c: reset AIC-analog interoceptive baseline.
        if self.aic is not None:
            self.aic.reset()
        self._aic_last_tick = None

        # SD-032d: reset PCC-analog per-episode state. Note: does NOT reset
        # _steps_since_offline (cross-episode counter; only note_offline_entry
        # resets it -- a new episode starting does not constitute rest).
        if self.pcc is not None:
            self.pcc.reset()
        self._pcc_last_tick = None

        # SD-032e: reset PACC-analog diagnostics cache. Does NOT clear
        # _drive_bias -- SD-032e's architectural purpose is cross-episode
        # accumulation. Use non-zero pacc_offline_decay via offline entry
        # (or reinstantiate the module) for a hard reset.
        if self.pacc is not None:
            self.pacc.reset()
        self._pacc_last_tick = None

        # MECH-302: reset rolling norm buffer and clear pending event flag
        # on episode boundary.
        if self.suffering_comparator is not None:
            self.suffering_comparator.reset()
        self._relief_completion_event = False

        # SD-051 / MECH-304: reset safety store prototype on episode boundary.
        if self.conditioned_safety_store is not None:
            self.conditioned_safety_store.reset()
        self._conditioned_safety_signal = 0.0

        # SD-033a: reset rule_state on episode boundary (SD-033a spec: rule
        # persists across ticks within episode; fresh episode starts without
        # a carried rule).
        if self.lateral_pfc is not None:
            self.lateral_pfc.reset()

        # ARC-063: reset the CandidateRuleField slot pool + recurrence counters
        # + per-tick credit stash on episode boundary (cross-episode carry-over
        # is a V3 follow-on, not v1 -- matches the SD-033a rule_state reset).
        if self.candidate_rule_field is not None:
            self.candidate_rule_field.reset()
            self._crf_prev_action_class = -1
            self._crf_prev_outcome = 0.0

        # SD-033b: reset state_code + oracle cache on episode boundary.
        if self.ofc is not None:
            self.ofc.reset()
        self._ofc_oracle_predictions = None

        # ARC-062 Phase 1: reset GatedPolicy diagnostic counters on episode
        # boundary. Module is stateless across ticks (no buffers); reset()
        # only clears the cached gating_weight / bias_abs_mean / sim-skip /
        # z_harm_a-was-none diagnostics.
        if self.gated_policy is not None:
            self.gated_policy.reset()
        if self.noise_floor is not None:
            self.noise_floor.reset()
        # MECH-314 (ARC-065): reset structured-curiosity diagnostics + 314c
        # learning-progress EMA buffer on episode boundary. Per-episode
        # because a fresh task/environment carries a fresh learning curve;
        # 314a / 314b are stateless across ticks.
        if self.curiosity is not None:
            self.curiosity.reset()
        # MECH-314a Phase 2 (Candidate 5A): clear the rolling z_world
        # visitation buffer per-episode (fresh novelty landscape each episode,
        # matching the StructuredCuriosity LP-buffer per-episode reset).
        if self._zworld_visitation_buffer is not None:
            self._zworld_visitation_buffer.clear()
        if self.tonic_vigor is not None:
            self.tonic_vigor.reset()
        if self.score_diversity is not None:
            self.score_diversity.reset()

        # MECH-090 R-c conjunction: reset commit-entry readiness EMA + per-
        # episode diagnostic counters. readiness returns to
        # commit_readiness_initial (default 1.0, fail-open) so a fresh
        # episode starts with the conjunction reducing to rv-only behaviour
        # until real outcome data has been collected.
        if self.commit_readiness is not None:
            self.commit_readiness.reset()

        # MECH-342: reset maintenance-release pressure accumulator + per-
        # episode diagnostic counters so each episode starts at zero release
        # pressure.
        if self.maintenance_release is not None:
            self.maintenance_release.reset()
        # Commit/release-DURATION lever: per-episode reset of the natural-commit
        # urgency accumulator + diagnostics (no cross-episode hold carried over).
        if self.natural_commit_urgency is not None:
            self.natural_commit_urgency.reset()
        # ARC-108 JOB-2: per-episode reset of the rho_t maintenance ramp.
        if self.rho_maintenance_ramp is not None:
            self.rho_maintenance_ramp.reset()
        # Natural-commit latch-hold: per-episode reset of the hold state +
        # diagnostics (a fresh trial starts with no carried-over hold).
        self._ncl_hold_active = False
        self._ncl_hold_ticks = 0
        self._ncl_hold_reassert_count = 0
        self._ncl_hold_closure_armed_count = 0
        self._ncl_mech091_fired = False
        self._ncl_lever_fired = False
        # rung-6 amend: clear the F-independent closure-plane commit-entry latch on the
        # episode boundary (a fresh trial starts with no carried-over closure commit).
        self.e3._closure_committed_active = False
        self.e3._closure_committed_trajectory = None  # C-STEP extension: clear on reset
        # SD-061: reset the stuck-state detector + proposal-entropy regulator
        # per episode (preserve no cross-episode impasse state) + the lagged
        # stuck_score the next _e3_tick reads.
        if self.stuck_state_detector is not None:
            self.stuck_state_detector.reset()
        if self.difficulty_gated_proposal_entropy is not None:
            self.difficulty_gated_proposal_entropy.reset()
        self._last_stuck_score = 0.0
        # MECH-353: reset blocked-agency integrator + the per-tick latent caches.
        if self.blocked_agency is not None:
            self.blocked_agency.reset()
        self._ba_prev_z_world = None
        self._ba_prev_z_self = None

        # MECH-276: clear the scientist-attribution prev caches. The buffer
        # itself PERSISTS across episodes (the MECH-275 aggregator runs
        # cross-episode); it is reset only by the sleep cycle decay / an explicit
        # buffer.reset(), not the per-episode agent.reset().
        self._sci_prev_z_world = None
        self._sci_prev_z_harm_s = None

        # MECH-219: reset the suffering accumulator per episode (s_t -> 0).
        if self.harm_suffering_accumulator is not None:
            self.harm_suffering_accumulator.reset()

        # SD-058 / MECH-357: clear the within-episode threat trace ONLY -- the
        # learned avoidance_efficacy PERSISTS across episodes (developmental
        # acquisition does not un-learn at episode boundaries). Also clear the
        # last-action-directed cache.
        if self.instrumental_avoidance is not None:
            self.instrumental_avoidance.reset()
        self._ia_last_action_directed = False
        # SD-059 / MECH-358: clear the within-episode threat trace + action cache.
        # Learned affordance tables persist across episodes (developmental
        # acquisition), same as the MECH-357 efficacy.
        if self.escape_affordance_bridge is not None:
            self.escape_affordance_bridge.reset()
        self._eab_last_action_class = None

        # MECH-045: per-episode clear of the object-file buffer.
        if self.object_file_buffer is not None:
            self.object_file_buffer.reset()

        # Post-603i trainable learner: clear the within-episode trace + action
        # cache. Learned relief/safety predictions persist across episodes.
        if self.trainable_escape_affordance_learner is not None:
            self.trainable_escape_affordance_learner.reset()
        self._teal_last_action_class = None

        # Post-603i E2 escape-affordance linker: clear within-episode traces +
        # action/z_world caches. Learned readout heads + viability index persist.
        if self.e2_escape_affordance_linker is not None:
            self.e2_escape_affordance_linker.reset()
        self._eal_last_action_class = None
        self._eal_prev_z_world = None

        # MECH-319: reset simulation-mode rule-gate diagnostic counters on
        # episode boundary. The gate has no persistent state across ticks
        # beyond counters, so reset is purely a diagnostic boundary.
        if self.simulation_mode_rule_gate is not None:
            self.simulation_mode_rule_gate.reset()

        # SD-034: reset closure-operator completion detector on episode boundary.
        if self.closure_operator is not None:
            self.closure_operator.reset()

        # SD-035: reset amygdala analogues (BLA PE EMA + window onset; CeA
        # fast-route counters). Both modules are stateful within an
        # episode; cross-episode state not retained.
        if self.bla is not None:
            self.bla.reset()
        if self.cea is not None:
            self.cea.reset()
        self._bla_last_output = None
        self._cea_last_output = None

        # SD-036: reset GABAergic decay regulator per-episode state (last-norms
        # cache for the suspend-on-input gate; diagnostics counters). Stream
        # registrations and gaba_tone are preserved across reset.
        if self.gabaergic_decay is not None:
            self.gabaergic_decay.reset()

        # MECH-279: reset PAG freeze gate per-episode state.
        if self.pag_freeze_gate is not None:
            self.pag_freeze_gate.reset()
        self._pag_last_output = None

        # SD-037: reset broadcast override regulator per-episode state (threat
        # window, EMA, diagnostics). Master flag is preserved across reset.
        if self.broadcast_override is not None:
            self.broadcast_override.reset()

        if self.lpb_router is not None:
            self.lpb_router.reset()
        self._lpb_last_output = None

        # MECH-295: reset bridge per-tick diagnostic cache. Fire counters
        # persist across reset so end-of-run reporting reflects the full
        # session.
        if self.mech295_bridge is not None:
            self.mech295_bridge.reset()

        # MECH-269 base: reset per-stream V_s cache on episode boundary.
        # No-op when use_per_stream_vs is False (per_stream_vs stays empty).
        if self.hippocampal is not None and getattr(
            self.hippocampal.config, "use_per_stream_vs", False
        ):
            self.hippocampal.reset_per_stream_vs()

        # MECH-269b: reset V_s rollout gate snapshots + diagnostic counters
        # on episode boundary. Snapshots and held counts are per-episode
        # (the fresh-episode latents have not yet had time to accumulate
        # any V_s history; held substitution from a prior episode would be
        # the wrong reference).
        if self.vs_rollout_gate is not None:
            self.vs_rollout_gate.reset()
            self._vs_gate_staleness_cache = {}

        # MECH-288: reset event segmenter state on episode boundary
        # (detector buffers, run-length posterior, outer.inner counters,
        # pending BoundaryEvent queue). No-op when use_event_segmenter is False.
        if self.hippocampal is not None and getattr(
            self.hippocampal.config, "use_event_segmenter", False
        ):
            self.hippocampal.reset_event_segmenter()

        # MECH-287: reset invalidation trigger on episode boundary (tonic
        # history, diagnostic counters, pending broadcast queue). No-op when
        # use_invalidation_trigger is False.
        if self.hippocampal is not None and getattr(
            self.hippocampal.config, "use_invalidation_trigger", False
        ):
            self.hippocampal.reset_invalidation_trigger()

        # MECH-269 Phase 2 (ii): reset anchor set on episode boundary
        # (active + inactive anchors, tick counter). No-op when
        # use_anchor_sets is False.
        if self.hippocampal is not None and getattr(
            self.hippocampal.config, "use_anchor_sets", False
        ):
            self.hippocampal.reset_anchor_set()

        # MECH-284 Phase 3: reset staleness accumulator on episode boundary
        # (region map, diagnostic counters). No-op when
        # use_staleness_accumulator is False.
        if self.hippocampal is not None and getattr(
            self.hippocampal.config, "use_staleness_accumulator", False
        ):
            self.hippocampal.reset_staleness_accumulator()

        # MECH-292: reset ghost-goal bank diagnostics on episode boundary.
        # No-op when use_mech292_ghost_bank is False. The anchor pool
        # itself is reset by reset_anchor_set() above.
        if self.hippocampal is not None and getattr(
            self.hippocampal.config, "use_mech292_ghost_bank", False
        ):
            self.hippocampal.reset_ghost_goal_bank()

        # MECH-290: clear committed trajectory buffer on episode boundary so a
        # stale trajectory from the previous episode cannot be swept on the
        # first completion of the new one. No-op when use_backward_credit_sweep
        # is False.
        if self.hippocampal is not None and getattr(
            self.hippocampal.config, "use_backward_credit_sweep", False
        ):
            self.hippocampal.reset_committed_trajectory()

    def _record_exploration_state(self) -> None:
        """MECH-165: record current latent state for exploration trajectory."""
        if not self.config.replay_diversity_enabled:
            return
        if self._current_latent is None:
            return
        self._episode_world_states.append(
            self._current_latent.z_world.detach().clone()
        )
        self._episode_self_states.append(
            self._current_latent.z_self.detach().clone()
        )

    def _record_exploration_action(self, action: torch.Tensor) -> None:
        """MECH-165: record selected action for exploration trajectory."""
        if not self.config.replay_diversity_enabled:
            return
        self._episode_actions.append(action.detach().clone())

    def _flush_exploration_episode(self) -> None:
        """
        MECH-165: build Trajectory from accumulated episode data and
        store in HippocampalModule exploration buffer. Clears episode buffers.
        Minimum 5 steps required to form a useful trajectory.
        """
        if not self.config.replay_diversity_enabled:
            return
        min_steps = 5
        n_states = len(self._episode_world_states)
        n_actions = len(self._episode_actions)
        if n_states < min_steps or n_actions < min_steps:
            self._episode_world_states.clear()
            self._episode_self_states.clear()
            self._episode_actions.clear()
            self._episode_bla_peak_tag = 0.0
            self._episode_bla_peak_encoding_gain = 1.0
            return

        # Align: states[0..N], actions[0..N-1]
        # Trim to min(n_states, n_actions+1) for clean pairing
        n = min(n_states, n_actions + 1)
        states = self._episode_self_states[:n]
        world_states = self._episode_world_states[:n]
        actions_list = self._episode_actions[:n - 1]

        # Build actions tensor [batch, horizon, action_dim]
        actions_tensor = torch.stack(actions_list, dim=1)  # [batch, horizon, action_dim]

        traj = Trajectory(
            states=states,
            actions=actions_tensor,
            world_states=world_states,
            is_reverse=False,
            memory_strength=float(self._episode_bla_peak_encoding_gain),
            arousal_tag=float(self._episode_bla_peak_tag),
        )
        self.hippocampal.record_exploration_trajectory(traj)

        self._episode_world_states.clear()
        self._episode_self_states.clear()
        self._episode_actions.clear()
        self._episode_bla_peak_tag = 0.0
        self._episode_bla_peak_encoding_gain = 1.0

    def _cache_tpj_prediction_for_action(self, action: Optional[torch.Tensor]) -> None:
        """Store the efference-copy z_self prediction for the chosen action."""
        if (
            self.tpj is None
            or action is None
            or self._current_latent is None
            or self._current_latent.z_self is None
        ):
            self._tpj_predicted_z_self = None
            return

        with torch.no_grad():
            z_self = self._current_latent.z_self.detach()
            a_in = action.detach()
            if z_self.dim() == 1:
                z_self = z_self.unsqueeze(0)
            if a_in.dim() == 1:
                a_in = a_in.unsqueeze(0)
            self._tpj_predicted_z_self = self.e2.predict_next_self(z_self, a_in).detach().clone()

    def _update_tpj_comparator(self, new_latent: LatentState) -> None:
        """Resolve the most recent TPJ agency comparison on the waking path."""
        if self.tpj is None or self._tpj_predicted_z_self is None or new_latent.z_self is None:
            self._tpj_last_agency_signal = None
            self._tpj_last_is_self_caused = None
            return

        with torch.no_grad():
            z_obs = new_latent.z_self.detach()
            if z_obs.dim() == 1:
                z_obs = z_obs.unsqueeze(0)
            agency_signal, is_self_caused = self.tpj.compare(
                self._tpj_predicted_z_self,
                z_obs,
            )
            self._tpj_last_agency_signal = agency_signal.detach().clone()
            self._tpj_last_is_self_caused = is_self_caused.detach().clone()
        self._tpj_predicted_z_self = None

    # ------------------------------------------------------------------
    # MECH-219 escapability source resolution
    # ------------------------------------------------------------------
    def set_harm_suffering_escapability(self, value: float) -> None:
        """MECH-219 `external` escapability mode seam.

        Lets a validation experiment drive the controllability gate directly
        (e.g. a scripted escapable-vs-inescapable schedule). Ignored unless
        harm_suffering_escapability_mode == 'external'. Clamped to [0, 1].
        """
        self._harm_suffering_external_escapability = float(
            max(0.0, min(1.0, value))
        )

    def set_injected_e2_forward_pe(self, pe: "Optional[torch.Tensor]") -> None:
        """DR-12 (self_model_v4:SELF-4) per-candidate E2-forward-PE injection seam.

        Lets a validation experiment supply the per-candidate E2 forward-PE [K]
        consumed by the E3 confidence down-weight on the next select_action tick.
        None clears it (bit-identical). The lever applies it only when
        E3Config.use_pe_confidence_weighting is True; supplying a PE with the lever
        OFF is a no-op. The caller is responsible for matching the current
        candidate count K.
        """
        self._injected_e2_forward_pe = pe

    def set_injected_go_nogo_signals(
        self, signals: "Optional[Dict[str, Any]]"
    ) -> None:
        """MECH-449 (ARC-107) per-candidate Go/No-Go signal injection seam.

        Lets the MECH-449 falsifier supply the constructed-bank per-candidate
        Go/No-Go axes (dict of [K] tensors keyed safety / staleness /
        perseveration / viability / go) consumed by the eligibility constitution
        on the next select_action tick. These merge over (and override) the
        default MECH-260 perseveration reuse. None clears it (bit-identical). The
        gate applies them only when E3Config.use_go_nogo_constitution is True;
        supplying signals with the constitution OFF is a no-op. The caller is
        responsible for matching the current candidate count K.
        """
        self._injected_go_nogo_signals = signals

    def _resolve_harm_suffering_escapability(self) -> float:
        """Resolve the MECH-219 escapability scalar in [0, 1] for the current tick.

        Source per config.harm_suffering_escapability_mode:
          constant            -> harm_suffering_escapability_constant (default 1.0;
                                 dependency-free; g=0 -> inert).
          avoidance_efficacy  -> SD-058 InstrumentalAvoidanceGate.effective_efficacy()
                                 (the literal escapability construct; soft dependency
                                 on the v3_pending SD-058 substrate). Falls back to
                                 the constant when the gate is absent.
          external            -> the scalar set via set_harm_suffering_escapability()
                                 (a validation experiment drives it).
        Never sourced from MECH-353 capacity_belief (= 1 - w*||z_harm_a||) -- that
        would close a z_harm_a -> capacity_belief -> z_harm_a loop; capacity_belief
        is a validation cross-check only (memo Section 3 / R1).
        """
        mode = getattr(
            self.config, "harm_suffering_escapability_mode", "constant"
        )
        if mode == "avoidance_efficacy":
            if self.instrumental_avoidance is not None:
                return float(
                    max(0.0, min(1.0, self.instrumental_avoidance.effective_efficacy()))
                )
            return float(
                getattr(self.config, "harm_suffering_escapability_constant", 1.0)
            )
        if mode == "external":
            return float(
                max(0.0, min(1.0, self._harm_suffering_external_escapability))
            )
        # constant (default)
        return float(
            max(0.0, min(1.0, getattr(
                self.config, "harm_suffering_escapability_constant", 1.0
            )))
        )

    def _update_blocked_agency(self, new_latent: LatentState) -> None:
        """MECH-353: resolve the blocked-agency / control-failure readout.

        Applies the SD-029 agency comparator to the ACTION-OUTCOME / z_world
        channel (outcome_mismatch = ||E2.world_forward(z_world_prev, a) -
        z_world_now|| normalised) and the z_self channel (motor_agency =
        1/(1+||E2.predict_next_self(z_self_prev, a) - z_self_now||)), then
        advances the BlockedAgency integrator behind the external-attribution
        and capacity gates. Sets new_latent.z_block. Caches the current latents
        for the next tick's comparison. No-op (z_block left None) when
        use_blocked_agency is off.

        MECH-094: a hypothesis-tagged (replay / simulation) latent advances
        nothing (update() is gated on simulation_mode) and z_block is left None.
        """
        if self.blocked_agency is None:
            return

        sim = bool(getattr(new_latent, "hypothesis_tag", False))
        z_world_now = new_latent.z_world
        z_self_now = new_latent.z_self

        # Need a previous latent + the action that bridged it to now.
        have_history = (
            self._ba_prev_z_world is not None
            and self._ba_prev_z_self is not None
            and self._last_action is not None
        )

        if not have_history or sim:
            # First waking tick of the episode (or a simulation tick): advance
            # the integrator with a no-block / sim signal so z_block stays
            # well-defined, then cache current latents.
            out = self.blocked_agency.update(
                outcome_mismatch=0.0,
                motor_agency=1.0,
                goal_active=(
                    self.goal_state is not None and self.goal_state.is_active()
                    if self.goal_state is not None else False
                ),
                capacity_belief=1.0,
                blocked_action_class=-1,
                simulation_mode=sim,
            )
            if not sim:
                self._ba_prev_z_world = z_world_now.detach().clone()
                self._ba_prev_z_self = z_self_now.detach().clone()
            new_latent.z_block = torch.tensor(
                [[out.z_block]], dtype=z_world_now.dtype, device=z_world_now.device
            )
            return

        with torch.no_grad():
            # Feed world_forward / predict_next_self the ONE-HOT of the discrete
            # action the env actually executed (env.step does argmax(action)),
            # NOT the raw continuous pre-argmax policy output -- otherwise the
            # forward model sees an out-of-distribution action encoding and its
            # prediction (trained on the executed discrete action) is unusable.
            _raw_a = self._last_action.detach()
            if _raw_a.dim() == 1:
                _raw_a = _raw_a.unsqueeze(0)
            _a_idx = int(_raw_a.argmax(dim=-1).flatten()[0].item())
            a_in = torch.zeros_like(_raw_a)
            a_in[0, _a_idx] = 1.0
            zw_prev = self._ba_prev_z_world
            zs_prev = self._ba_prev_z_self
            zw_now = z_world_now.detach()
            zs_now = z_self_now.detach()
            if zw_now.dim() == 1:
                zw_now = zw_now.unsqueeze(0)
            if zs_now.dim() == 1:
                zs_now = zs_now.unsqueeze(0)

            # Action-outcome comparator on z_world (SD-029 applied to the goal
            # channel; Carruthers 2012). E2.world_forward gives the predicted
            # next z_world for the action taken; the block signature is the
            # divergence of the realised z_world from that prediction, scaled by
            # the predicted-effect magnitude:
            #   zw_pred  = world_forward(zw_prev, a)        (intended outcome)
            #   pred_mag = ||zw_pred - zw_prev||            (intended effect size)
            #   outcome_mismatch = ||zw_pred - zw_now|| / (pred_mag + eps)
            # Calibration (with a trained, action-conditional world_forward):
            #   blocked  -> zw_now ~ zw_prev -> ||zw_pred - zw_now|| ~ pred_mag
            #               -> outcome_mismatch ~ 1.0 (the predicted effect did
            #               not happen);
            #   success  -> zw_now ~ zw_pred -> outcome_mismatch ~ 0.0.
            # Gated by a predicted-effect floor: when pred_mag < floor the action
            # was not predicted to produce an effect -> 0 (nothing to be blocked
            # from; also fails safe on an untrained world_forward).
            # IMPORTANT: this comparator is only DISCRIMINATIVE once the encoder
            # represents the scene/position in z_world AND world_forward is
            # action-conditional (SD-056). The validation experiment trains both
            # in P0; at smoke / untrained scale z_world deltas do not track moves
            # so the comparator is uninformative (expected).
            # Noop-baseline formulation: is the realised outcome closer to the
            # predicted MOVE (action succeeded) or to STAYING PUT (action
            # blocked)? zw_prev is the "no-effect" baseline.
            #   to_move = ||zw_now - zw_pred||   (distance from the predicted move)
            #   to_stay = ||zw_now - zw_prev||   (distance from staying put)
            #   outcome_mismatch = max(0, to_move - to_stay) / (pred_mag + eps)
            # blocked  -> zw_now ~ zw_prev -> to_stay~0, to_move~pred_mag -> ~1.0;
            # success  -> zw_now ~ zw_pred -> to_move~0, to_stay~pred_mag -> clipped 0.
            # This discriminates whenever world_forward predicts a non-trivial
            # action effect (pred_mag >= floor), without requiring the prediction
            # to be globally accurate -- only that the action's predicted effect
            # is distinguishable from no-effect.
            zw_pred = self.e2.world_forward(zw_prev, a_in)
            pred_mag = float((zw_pred - zw_prev).norm(dim=-1).mean().item())
            _eff_floor = float(
                getattr(self.config, "blocked_agency_predicted_effect_floor", 0.05)
            )
            if pred_mag < _eff_floor:
                outcome_mismatch = 0.0
            else:
                to_move = float((zw_pred - zw_now).norm(dim=-1).mean().item())
                to_stay = float((zw_now - zw_prev).norm(dim=-1).mean().item())
                outcome_mismatch = max(0.0, to_move - to_stay) / (pred_mag + 1e-6)

            # Motor-agency comparator on z_self (attribution gate): high =
            # motor command executed as predicted -> external block, not own
            # motor error.
            zs_pred = self.e2.predict_next_self(zs_prev, a_in)
            motor_mismatch = float((zs_pred - zs_now).norm(dim=-1).mean().item())
            motor_agency = 1.0 / (1.0 + motor_mismatch)

            # Capacity-belief proxy: collapses with affective suffering load
            # (z_harm_a). When the harm stream is off, capacity is fully retained
            # (assert pole) -- correct for the harm-held-constant regime.
            if new_latent.z_harm_a is not None:
                z_harm_a_norm = float(new_latent.z_harm_a.detach().norm(dim=-1).mean().item())
            else:
                z_harm_a_norm = 0.0
            capacity_belief = max(
                0.0,
                min(
                    1.0,
                    1.0
                    - self.blocked_agency.config.capacity_collapse_weight
                    * z_harm_a_norm,
                ),
            )

            goal_active = (
                self.goal_state is not None and self.goal_state.is_active()
            )
            blocked_cls = int(self._last_action.detach().argmax(dim=-1).flatten()[0].item())

        out = self.blocked_agency.update(
            outcome_mismatch=outcome_mismatch,
            motor_agency=motor_agency,
            goal_active=goal_active,
            capacity_belief=capacity_belief,
            blocked_action_class=blocked_cls,
            simulation_mode=False,
        )
        new_latent.z_block = torch.tensor(
            [[out.z_block]], dtype=z_world_now.dtype, device=z_world_now.device
        )
        self._ba_prev_z_world = zw_now.detach().clone()
        self._ba_prev_z_self = zs_now.detach().clone()

    def _scientist_attribution_region(self) -> Tuple[str, str]:
        """Current MECH-269 anchor region (scale, segment_id) for MECH-276.

        Reads the most-recently-active anchor from the hippocampal AnchorSet so
        attribution records are keyed on the same RegionKey the sleep loop
        routes on (matches phase_manager's (anchor.key[0], anchor.key[1])). When
        no anchor substrate / active anchor is available, returns the
        ScientistAttributionBuffer.GLOBAL_REGION sentinel -- those records still
        contribute to the global-mean fallback the sleep loop reads.
        """
        hippocampal = getattr(self, "hippocampal", None)
        anchor_set = getattr(hippocampal, "anchor_set", None)
        if anchor_set is not None:
            try:
                actives = anchor_set.active_anchors()
            except Exception:  # pragma: no cover -- defensive
                actives = []
            if actives:
                # Most recently created active anchor is the current region.
                chosen = max(
                    actives, key=lambda a: getattr(a, "created_at", 0)
                )
                key = getattr(chosen, "key", None)
                if (
                    isinstance(key, tuple)
                    and len(key) >= 2
                    and isinstance(key[0], str)
                    and isinstance(key[1], str)
                ):
                    return (key[0], key[1])
        return ScientistAttributionBuffer.GLOBAL_REGION

    def _update_scientist_attribution(self, new_latent: LatentState) -> None:
        """MECH-276: buffer this tick's counterfactual-backed attribution.

        The waking-phase feedstock for the MECH-275 sleep aggregator. Using the
        prev-latent caches (one-tick lag, mirroring _update_blocked_agency),
        runs the single-pass comparators on (z_prev, observed, a_actual):

          place domain (SD-031 E2WorldForward, requires attribution_ready /
            world_dim>=128): attribution = ||z_world_obs - E2(z_prev, a)||;
            counterfactual_contrast = ||E2(z_prev, a) - E2(z_prev, a_cf)||.
          self domain (ARC-033 E2HarmSForward): attribution = ||z_harm_s_obs -
            E2_harm_s(z_prev, a)||; counterfactual_contrast =
            ||E2_harm_s(z_prev, a) - E2_harm_s.counterfactual_forward(z_prev, a_cf)||.

        a_cf is a deterministic argmax-shifted action (a discriminating
        alternative). The buffer flags the attribution counterfactual-backed
        iff the contrast clears cf_margin; correlational records are skipped
        when only_counterfactual_backed.

        No-op (nothing buffered) when use_scientist_attribution is off. MECH-094:
        a hypothesis-tagged (replay / simulation) latent buffers nothing (the
        comparators return zeros under simulation_mode AND record() is a no-op).
        """
        buf = self.scientist_attribution_buffer
        if buf is None:
            return

        sim = bool(getattr(new_latent, "hypothesis_tag", False))

        # First waking tick / simulation tick: just refresh caches (no record).
        have_history = self._last_action is not None and (
            self._sci_prev_z_world is not None
            or self._sci_prev_z_harm_s is not None
        )
        if sim or not have_history:
            if not sim:
                if new_latent.z_world is not None:
                    self._sci_prev_z_world = new_latent.z_world.detach().clone()
                if new_latent.z_harm is not None:
                    self._sci_prev_z_harm_s = new_latent.z_harm.detach().clone()
            return

        region = self._scientist_attribution_region()

        with torch.no_grad():
            # One-hot of the discrete action the env actually executed.
            _raw_a = self._last_action.detach()
            if _raw_a.dim() == 1:
                _raw_a = _raw_a.unsqueeze(0)
            n_act = _raw_a.shape[-1]
            _a_idx = int(_raw_a.argmax(dim=-1).flatten()[0].item())
            a_in = torch.zeros_like(_raw_a)
            a_in[0, _a_idx] = 1.0
            # Deterministic discriminating counterfactual: the next action class.
            _cf_idx = (_a_idx + 1) % max(1, n_act)
            a_cf = torch.zeros_like(_raw_a)
            a_cf[0, _cf_idx] = 1.0

            # Place domain -- SD-031 E2WorldForward causal-footprint comparator.
            if (
                self.e2_world is not None
                and self.e2_world.attribution_ready
                and self._sci_prev_z_world is not None
                and new_latent.z_world is not None
            ):
                zw_prev = self._sci_prev_z_world
                zw_obs = new_latent.z_world.detach()
                if zw_obs.dim() == 1:
                    zw_obs = zw_obs.unsqueeze(0)
                pred_actual = self.e2_world.forward(zw_prev, a_in)
                pred_cf = self.e2_world.forward(zw_prev, a_cf)
                attribution = float((zw_obs - pred_actual).norm(dim=-1).mean().item())
                cf_contrast = float((pred_actual - pred_cf).norm(dim=-1).mean().item())
                buf.record(
                    domain="place",
                    region=region,
                    attribution=attribution,
                    counterfactual_contrast=cf_contrast,
                    simulation_mode=False,
                )

            # Self domain -- ARC-033 E2HarmSForward (SD-003 causal_sig).
            if (
                self.e2_harm_s is not None
                and self._sci_prev_z_harm_s is not None
                and new_latent.z_harm is not None
            ):
                zh_prev = self._sci_prev_z_harm_s
                zh_obs = new_latent.z_harm.detach()
                if zh_obs.dim() == 1:
                    zh_obs = zh_obs.unsqueeze(0)
                pred_actual = self.e2_harm_s.forward(zh_prev, a_in)
                pred_cf = self.e2_harm_s.counterfactual_forward(zh_prev, a_cf)
                attribution = float((zh_obs - pred_actual).norm(dim=-1).mean().item())
                cf_contrast = float((pred_actual - pred_cf).norm(dim=-1).mean().item())
                buf.record(
                    domain="self",
                    region=region,
                    attribution=attribution,
                    counterfactual_contrast=cf_contrast,
                    simulation_mode=False,
                )

        # Refresh caches for the next tick.
        if new_latent.z_world is not None:
            self._sci_prev_z_world = new_latent.z_world.detach().clone()
        if new_latent.z_harm is not None:
            self._sci_prev_z_harm_s = new_latent.z_harm.detach().clone()

    def _get_context_memory_code_contributions(
        self,
        z_self: torch.Tensor,
        z_world: torch.Tensor,
    ) -> Optional[Dict[int, float]]:
        """Approximate active code contributions from ContextMemory slot attention."""
        if not hasattr(self.e1, "context_memory"):
            return None

        cm = self.e1.context_memory
        with torch.no_grad():
            state = torch.cat([z_self.detach(), z_world.detach()], dim=-1)
            query = cm.query_proj(state)
            scores = torch.mm(query, cm.memory.detach().t())
            weights = torch.softmax(scores, dim=-1).mean(dim=0)

        return {
            int(idx): float(weights[idx].item())
            for idx in range(weights.numel())
            if float(weights[idx].item()) > 0.0
        }

    def _apply_bla_context_remap(
        self,
        remap_signal: Dict[int, float],
        z_self: torch.Tensor,
        z_world: torch.Tensor,
    ) -> None:
        """Apply BLA remap to the targeted ContextMemory slots in place."""
        if not remap_signal or not hasattr(self.e1, "context_memory"):
            return

        cm = self.e1.context_memory
        blend_base = float(getattr(self.config, "bla_context_remap_blend", 0.5))
        with torch.no_grad():
            state = torch.cat([z_self.detach(), z_world.detach()], dim=-1)
            write_signal = cm.write_gate(state).mean(dim=0)
            n_slots = cm.memory.shape[0]
            for code_idx, amplitude in remap_signal.items():
                idx = int(code_idx)
                if idx < 0 or idx >= n_slots:
                    continue
                blend = max(0.0, min(1.0, blend_base * float(amplitude)))
                cm.memory.data[idx] = (
                    (1.0 - blend) * cm.memory.data[idx]
                    + blend * write_signal
                )

    def _per_axis_drive_for_consumers(self) -> Optional[torch.Tensor]:
        """SD-049 Phase 3 plumbing gate.

        Returns the cached per-axis drive vector iff (a) the SD-049 Phase 3
        consumer cascade is enabled in config AND (b) sense() received a
        non-None obs_per_axis_drive on the last tick. Otherwise returns
        None so each consumer falls back to the legacy scalar drive_level
        path -- bit-identical OFF guarantee.
        """
        if not getattr(
            self.config, "use_sd049_per_axis_consumer_cascade", False
        ):
            return None
        return self._per_axis_drive

    def update_object_file_buffer(
        self,
        observations: "List[EntityObservation]",
        simulation_mode: bool = False,
    ) -> "Dict[int, int]":
        """MECH-045: advance the token-instance object-file buffer one waking
        tick over the supplied perceived entities (memo Section 4.2).

        The caller (experiment / harness) builds `observations` from the env's
        perceived entity cells (SD-049 per-type resource-field views + grid
        object/hazard cells) -- the v1 detector dependency (memo Section 4.4).
        Returns {observation_index -> token_id}; an empty dict when the buffer
        is disabled or under simulation_mode (MECH-094: the buffer updates only
        on the waking stream). This does NOT touch the action stream -- v1 has
        no consumer, so the buffer is observational only.
        """
        if self.object_file_buffer is None:
            return {}
        return self.object_file_buffer.update(
            observations, simulation_mode=simulation_mode
        )

    def sense(
        self,
        obs_body: torch.Tensor,
        obs_world: torch.Tensor,
        obs_harm: Optional[torch.Tensor] = None,
        obs_harm_a: Optional[torch.Tensor] = None,
        obs_harm_history: Optional[torch.Tensor] = None,
        obs_per_axis_drive: Optional[torch.Tensor] = None,
        mech090_readiness_outcome: Optional[float] = None,
    ) -> LatentState:
        """
        SENSE + UPDATE step: encode split observation -> update latent state.

        Args:
            obs_body:   [batch, body_obs_dim] proprioceptive channels
            obs_world:  [batch, world_obs_dim] exteroceptive channels
            obs_harm:   SD-010 nociceptive channels [batch, harm_obs_dim] or None.
                        When provided and use_harm_stream=True in config, routes
                        through HarmEncoder to z_harm (bypasses reafference correction).
            obs_harm_a: SD-011 affective-motivational harm channels
                        [batch, harm_obs_a_dim] or None. EMA-accumulated proximity
                        signal from environment. When provided and
                        use_affective_harm_stream=True, routes through
                        AffectiveHarmEncoder to z_harm_a.
            obs_harm_history: SD-011 second source [batch, harm_history_len] or None.
                        Rolling window of past harm_exposure scalars. Concatenated
                        with harm_obs_a as AffectiveHarmEncoder input when
                        harm_history_len > 0.
            mech090_readiness_outcome: MECH-090 R-c continuation (nav_competence
                        axis) per-tick outcome signal in [0, 1] or None. The
                        Phase-2 follow-on source for the across-tick commit-
                        readiness EMA: the caller forwards the env-emitted
                        info["mech090_readiness_outcome"] (e.g. CausalGridWorldV2
                        with mech090_readiness_outcome_enabled=True surfaces
                        1.0 - mean(limb_damage)). When provided AND
                        self.commit_readiness is not None, advances the readiness
                        EMA so the across-tick nav_competence gate at the
                        beta_gate elevate sites is fed automatically (no out-of-
                        band notify_outcome harness push). None (the default --
                        key absent) leaves the EMA un-advanced (bit-identical
                        fail-open). MECH-094: simulation/replay ticks
                        (hypothesis_tag) do not advance the EMA.

        Returns:
            Updated LatentState
        """
        if obs_body.dim() == 1:
            obs_body  = obs_body.unsqueeze(0)
            obs_world = obs_world.unsqueeze(0)
        obs_body  = obs_body.to(self.device).float()
        obs_world = obs_world.to(self.device).float()

        # SD-049 Phase 3: cache the per-axis drive vector for downstream
        # SD-032 consumer ticks. Stored verbatim (detached, on-device);
        # the per_axis_drive helper in ree_core/utils accepts torch/numpy/
        # python-sequence interchangeably. None when the env did not
        # surface per-axis drive (multi_resource_heterogeneity disabled or
        # per_axis_drive_enabled=False) OR when the caller did not pass
        # obs_per_axis_drive (legacy entry points).
        if obs_per_axis_drive is not None:
            self._per_axis_drive = (
                obs_per_axis_drive.detach().to(self.device).float()
            )
        else:
            self._per_axis_drive = None

        enc_body  = self.body_obs_encoder(obs_body)
        enc_world = self.world_obs_encoder(obs_world)

        # Concatenate for LatentStack (which splits internally)
        enc_combined = torch.cat([enc_body, enc_world], dim=-1)
        # SD-007: pass last action as prev_action for reafference correction.
        # _last_action is None on the first step (no correction applied).
        # Q-007: pass volatility estimate (var(rv) over sliding window) to
        # z_beta encoder.  Raw rv converges to near-zero in both stable and
        # volatile environments once E2 adapts.  var(rv) captures how much rv
        # *fluctuates*: high in volatile envs (rv spikes on hazard moves),
        # low in stable envs (rv is flat).  LC-NE tonic firing analog.
        vol_signal = None
        if self.config.latent.volatility_signal_dim > 0:
            vol_signal = self.e3.volatility_estimate
        harm_for_encoder = obs_harm
        if self.lpb_router is not None and obs_harm is not None:
            harm_for_encoder = self.lpb_router.mask_external_harm_obs(obs_harm)
        new_latent = self.latent_stack.encode(
            enc_combined, self._current_latent,
            prev_action=self._last_action,
            harm_obs=harm_for_encoder,  # SD-010 / MECH-282 external-only when LPB on
            harm_obs_a=obs_harm_a,   # SD-011: affective harm stream (None = disabled)
            harm_history=obs_harm_history,  # SD-011 second source (None = disabled)
            volatility_signal=vol_signal,
        )

        # MECH-423 R2: cache the iterative-inference convergence readout for the
        # EXP-0380 R2 readiness check (None unless use_iterative_inference is on).
        self.last_inference_convergence = getattr(
            new_latent, "inference_convergence", None
        )

        # MECH-282: LPB interoceptive channel (non-trainable routing).
        self._lpb_last_output = None
        if self.lpb_router is not None:
            lpb_drive = 0.0
            if self.goal_state is not None:
                lpb_drive = float(
                    getattr(self.goal_state, "_last_drive_level", 0.0)
                )
            self._lpb_last_output = self.lpb_router.tick(
                drive_level=lpb_drive,
                harm_obs=obs_harm,
                harm_obs_a=obs_harm_a,
                device=self.device,
                batch_size=new_latent.z_self.shape[0],
                simulation_mode=bool(getattr(new_latent, "hypothesis_tag", False)),
            )
            new_latent.z_harm_intero = self._lpb_last_output.z_harm_intero

        # MECH-095: resolve the TPJ efference-copy comparison for the most
        # recently executed action. Runs immediately after encoding so the
        # comparator sees the freshly observed z_self.
        self._update_tpj_comparator(new_latent)

        # MECH-353: resolve the blocked-agency / control-failure readout for the
        # most recently executed action. Runs after encoding so the comparator
        # sees the freshly observed z_world / z_self. No-op when
        # use_blocked_agency is False.
        self._update_blocked_agency(new_latent)

        # MECH-276: buffer this tick's counterfactual-backed attribution (the
        # waking-phase feedstock for the MECH-275 sleep aggregator). No-op when
        # use_scientist_attribution is False (buffer is None).
        self._update_scientist_attribution(new_latent)

        # SD-058 / MECH-357: advance the avoidance-efficacy eligibility trace.
        # Compares the current z_harm_a (SD-011 affective stream) to the threat
        # the previous action responded to: a directed action under threat that
        # dropped z_harm_a credits efficacy; freezing / failed avoidance decays
        # it. One-tick lag (the outcome is the just-experienced threat change).
        # MECH-094: no-op when the latent carries hypothesis_tag (replay/DMN
        # must not credit avoidance on imagined outcomes). Bit-identical when
        # self.instrumental_avoidance is None.
        if self.instrumental_avoidance is not None:
            _ia_sim = bool(getattr(new_latent, "hypothesis_tag", False))
            _ia_zha = getattr(new_latent, "z_harm_a", None)
            _ia_zn = (
                float(_ia_zha.detach().norm().item())
                if _ia_zha is not None
                else 0.0
            )
            self.instrumental_avoidance.update(
                z_harm_a_norm=_ia_zn,
                action_was_directed=self._ia_last_action_directed,
                simulation_mode=_ia_sim,
            )

        # SD-059 / MECH-358: advance the relief/safety escape-affordance
        # eligibility traces. A directed action under threat that DROPS z_harm_a
        # credits relief_affordance[action_class]; a directed action after which
        # threat is absent credits safety_affordance[action_class]. One-tick lag
        # (the outcome is the just-experienced threat change), same as MECH-357.
        # MECH-094: no-op under hypothesis_tag. Bit-identical when the bridge is
        # None.
        if self.escape_affordance_bridge is not None:
            _eab_sim = bool(getattr(new_latent, "hypothesis_tag", False))
            _eab_zha = getattr(new_latent, "z_harm_a", None)
            _eab_zn = (
                float(_eab_zha.detach().norm().item())
                if _eab_zha is not None
                else 0.0
            )
            # 603i SAFETY-half fix: feed the trained MECH-303/304 threat-absence
            # prediction for the CURRENT post-action state so the safety half can
            # credit non-vacuously (on 603i the raw threat_scale<=0 check fired
            # 0/3 because the threat never goes fully absent under Stage-H). max()
            # over whichever trained predictors are enabled. Pure reads (no store
            # mutation): MECH-303 evaluate_safety is read-only; MECH-304 predict()
            # is the non-mutating accessor (the store's own update() runs LATER in
            # this same sense(), so _conditioned_safety_signal would be one tick
            # stale). None when the flag is off -> bit-identical.
            _eab_safety_signal: Optional[float] = None
            if (
                not _eab_sim
                and getattr(self.config, "escape_use_trained_safety_signal", False)
                and getattr(new_latent, "z_world", None) is not None
            ):
                _sig = 0.0
                # MECH-304 cue-specific conditioned safety store.
                if self.conditioned_safety_store is not None:
                    try:
                        _sig = max(
                            _sig,
                            float(self.conditioned_safety_store.predict(new_latent.z_world)),
                        )
                    except Exception:
                        pass
                # MECH-303 contextual passive safety terrain (RBF read).
                if (
                    getattr(self.config, "use_contextual_safety_terrain", False)
                    and hasattr(self.residue_field, "evaluate_safety")
                ):
                    try:
                        _sv = self.residue_field.evaluate_safety(new_latent.z_world)
                        _sig = max(_sig, float(_sv.mean().item()))
                    except Exception:
                        pass
                _eab_safety_signal = _sig
            # Directedness is the non-noop check, which the bridge applies
            # internally on last_action_class -- pass True so the bridge works
            # standalone (independent of whether MECH-357 is enabled).
            self.escape_affordance_bridge.update(
                z_harm_a_norm=_eab_zn,
                last_action_class=self._eab_last_action_class,
                last_action_directed=(self._eab_last_action_class is not None),
                simulation_mode=_eab_sim,
                safety_signal=_eab_safety_signal,
            )

        # Post-603i trainable relief/safety learner. Same one-tick lag as the
        # arithmetic bridge, but with compact detached z_world/z_self/z_harm_a
        # hooks feeding local PyTorch heads.
        # MECH-094: no-op under hypothesis_tag. Bit-identical when disabled.
        if self.trainable_escape_affordance_learner is not None:
            _teal_sim = bool(getattr(new_latent, "hypothesis_tag", False))
            _teal_zha = getattr(new_latent, "z_harm_a", None)
            _teal_zn = (
                float(_teal_zha.detach().norm().item())
                if _teal_zha is not None
                else 0.0
            )
            self.trainable_escape_affordance_learner.update(
                z_harm_a_norm=_teal_zn,
                last_action_class=self._teal_last_action_class,
                z_world=getattr(new_latent, "z_world", None),
                z_self=getattr(new_latent, "z_self", None),
                z_harm_a=_teal_zha,
                last_action_directed=(self._teal_last_action_class is not None),
                extra_features=self._eal_linker_context_feature(),
                simulation_mode=_teal_sim,
                hypothesis_tag=_teal_sim,
            )

        # Post-603i E2 escape-affordance linker. Reads the DETACHED E2 action-
        # consequence feature for the just-executed (prev_z_world, last_action)
        # pair from the EXISTING cerebellar-analog forward model
        # (self.e2.world_forward) -- it does NOT re-predict. The feature indexes
        # E2 geometry; the linker learns only the action-contingent escape-
        # viability labels on top of it. One-tick lag (the outcome is the just-
        # experienced threat change), MECH-094-gated, bit-identical when None.
        if self.e2_escape_affordance_linker is not None:
            _eal_sim = bool(getattr(new_latent, "hypothesis_tag", False))
            _eal_zha = getattr(new_latent, "z_harm_a", None)
            _eal_zn = (
                float(_eal_zha.detach().norm().item()) if _eal_zha is not None else 0.0
            )
            _eal_zw_now = getattr(new_latent, "z_world", None)
            _eal_e2_feat = None
            if (
                not _eal_sim
                and self._eal_prev_z_world is not None
                and self._eal_last_action_class is not None
            ):
                try:
                    _a_dim = int(self.config.e2.action_dim)
                    _zw_prev = self._eal_prev_z_world
                    if _zw_prev.dim() == 1:
                        _zw_prev = _zw_prev.unsqueeze(0)
                    _a_oh = torch.zeros(
                        1, _a_dim, dtype=_zw_prev.dtype, device=_zw_prev.device
                    )
                    _a_oh[0, min(max(0, int(self._eal_last_action_class)), _a_dim - 1)] = 1.0
                    with torch.no_grad():
                        _zw_pred = self.e2.world_forward(_zw_prev, _a_oh)
                    _eal_e2_feat = torch.cat(
                        [
                            _zw_pred.detach().flatten(),
                            (_zw_pred - _zw_prev).detach().flatten(),
                        ]
                    )
                except Exception:
                    _eal_e2_feat = None
            self.e2_escape_affordance_linker.update(
                z_harm_a_norm=_eal_zn,
                last_action_class=self._eal_last_action_class,
                e2_features=_eal_e2_feat,
                z_world=_eal_zw_now,
                z_self=getattr(new_latent, "z_self", None),
                z_harm_a=_eal_zha,
                last_action_directed=(self._eal_last_action_class is not None),
                simulation_mode=_eal_sim,
                hypothesis_tag=_eal_sim,
            )
            if _eal_zw_now is not None:
                self._eal_prev_z_world = _eal_zw_now.detach().clone()

        # SD-036: tick the GABAergic cross-stream decay regulator. Applies
        # z_s(t+1) = z_s(t) * exp(-tau_s * gaba_tone) to registered streams
        # (z_harm, z_harm_a, z_beta) BEFORE downstream mode-arbitration
        # consumers (AIC, BLA, CeA, salience coordinator) read the latent
        # state. This is the architectural fix for V3-EXQ-471 catatonic lock:
        # without decay, a single hazard contact pinned z_harm_norm at ~0.7
        # for 199 steps; with the regulator, z_harm decays with ~20-step
        # half-life and mode arbitration can flip back to goal-seeking. The
        # regulator is a no-op when use_gabaergic_decay=False.
        # MECH-094: hypothesis_tag from the LatentState gates simulation/
        # replay content -- the regulator does not touch decay state for those
        # ticks.
        if self.gabaergic_decay is not None:
            self.gabaergic_decay.tick(
                new_latent,
                simulation_mode=bool(getattr(new_latent, "hypothesis_tag", False)),
            )

        # SD-019a: harm_unpleasantness_channel EMA.
        # Maintains a medium-timescale EMA of z_harm_s (alpha=0.2, ~5-step rise).
        # NOT modulated by controllability (Loffler 2018 three-way dissociation).
        # MECH-094: only updated in waking sense() calls (hypothesis_tag=False).
        # Populates new_latent.z_harm_un for downstream AIC + E3 consumers.
        if (
            self.config.latent.use_harm_un
            and new_latent.z_harm is not None
            and not bool(getattr(new_latent, "hypothesis_tag", False))
        ):
            alpha_un = self.config.latent.harm_un_ema_alpha
            with torch.no_grad():
                if self._harm_un_ema is None:
                    self._harm_un_ema = new_latent.z_harm.detach().clone()
                else:
                    self._harm_un_ema = (
                        (1.0 - alpha_un) * self._harm_un_ema
                        + alpha_un * new_latent.z_harm.detach()
                    )
            new_latent.z_harm_un = self._harm_un_ema.clone()

        # MECH-219 (SD-019b): controllability-gated hysteretic suffering
        # integrator. Reads the SD-019a unpleasantness magnitude ||z_harm_un||
        # as the drive and the resolved escapability scalar as the
        # controllability gate (g_t = 1 - escapability); accumulates with
        # asymmetric alpha_rise >> alpha_fall into a slow suffering scalar s_t.
        # The output z_harm_suffering LatentState vector points in the
        # z_harm_un direction with magnitude s_t (same dim as z_harm_un). Runs
        # BEFORE the SD-032 consumers (AIC below, pACC/PAG in select_action) so
        # any per-consumer redirect reads the suffering output on the same tick.
        # MECH-094: no-op under hypothesis_tag (replay must not accumulate
        # suffering). Bit-identical when the accumulator is None, and inert under
        # the default escapability_mode=constant=1.0 (g=0 -> s->0).
        if (
            self.harm_suffering_accumulator is not None
            and new_latent.z_harm_un is not None
        ):
            sim_mode = bool(getattr(new_latent, "hypothesis_tag", False))
            with torch.no_grad():
                u_norm = float(new_latent.z_harm_un.detach().norm().item())
                body_norm = 0.0
                if new_latent.z_harm_a is not None:
                    body_norm = float(new_latent.z_harm_a.detach().norm().item())
                escap = self._resolve_harm_suffering_escapability()
                out = self.harm_suffering_accumulator.update(
                    unpleasantness_norm=u_norm,
                    escapability=escap,
                    body_damage_norm=body_norm,
                    unsigned_pe=0.0,
                    simulation_mode=sim_mode,
                )
                # Build the z_harm_suffering vector: the z_harm_un direction
                # scaled to magnitude s_t (so ||z_harm_suffering|| == s_t).
                zun = new_latent.z_harm_un.detach()
                zun_norm = float(zun.norm().item())
                if zun_norm > 1e-8:
                    direction = zun / zun_norm
                else:
                    direction = torch.zeros_like(zun)
                new_latent.z_harm_suffering = (direction * out.s).clone()

        # SD-037: tick the broadcast override regulator (orexin-analog).
        # Combines drive_level (SD-012) and a sustained-threat magnitude window
        # over z_harm into a scalar override_signal in [0, 1]. The signal is
        # consumed downstream at PAG (theta_freeze scaling), SalienceCoordinator
        # (operating-mode reweight), and GoalState (drive -> z_goal seeding gate).
        # No-op when use_broadcast_override=False.
        # MECH-094: simulation/replay content does not recruit the override
        # system -- the regulator's internal threat window and EMA are frozen
        # for hypothesis-tagged ticks.
        if self.broadcast_override is not None:
            override_drive = 0.0
            if self.goal_state is not None:
                override_drive = float(
                    getattr(self.goal_state, "_last_drive_level", 0.0)
                )
            override_z_harm = 0.0
            if self._lpb_last_output is not None:
                override_z_harm = float(self._lpb_last_output.external_magnitude)
            elif new_latent.z_harm is not None:
                override_z_harm = float(new_latent.z_harm.norm().item())
            intero_norm = None
            lpb_split = False
            if self._lpb_last_output is not None:
                intero_norm = float(self._lpb_last_output.intero_magnitude)
                lpb_split = True
            self.broadcast_override.tick(
                drive_level=override_drive,
                z_harm_norm=override_z_harm,
                simulation_mode=bool(getattr(new_latent, "hypothesis_tag", False)),
                z_harm_intero_norm=intero_norm,
                lpb_split_recruitment=lpb_split,
                # SD-049 Phase 3: orexin recruitment on worst-deficit axis
                # when per-axis cascade is on.
                per_axis_drive=self._per_axis_drive_for_consumers(),
                per_axis_combiner=getattr(
                    self.config, "sd049_override_per_axis_combiner", "max"
                ),
            )

        # SD-032c: tick the AIC-analog interoceptive-salience module.
        # Reads z_harm_a_norm + drive_level + beta_gate_elevated + the
        # coordinator's previous operating_mode readout. Produces aic_salience
        # (fed to the coordinator below in select_action) and harm_s_gain
        # (descending attenuation multiplier that subsumes the raw SD-021
        # beta_gate check).
        if self.aic is not None:
            # SD-019a: when harm_unpleasantness_channel is active, AIC urgency
            # reads z_harm_un (medium-timescale EMA of z_harm_s) instead of
            # z_harm_a (slow accumulator). z_harm_un reflects the per-Loffler
            # 2018 unpleasantness dimension, which is the correct AIC input.
            if self.config.latent.use_harm_un and new_latent.z_harm_un is not None:
                aic_z_norm = float(new_latent.z_harm_un.norm().item())
            elif new_latent.z_harm_a is not None:
                aic_z_norm = float(new_latent.z_harm_a.norm().item())
            else:
                aic_z_norm = 0.0
            # MECH-219 (SD-019b) AIC redirect: when the suffering accumulator is
            # on AND the AIC redirect flag is set, source AIC urgency from the
            # slow suffering magnitude ||z_harm_suffering|| instead. Default off
            # -> bit-identical. (Magnitude-only redirect; the AIC reads a scalar.)
            if (
                getattr(self.config, "use_harm_suffering_accumulator", False)
                and getattr(self.config, "harm_suffering_redirect_aic", False)
                and new_latent.z_harm_suffering is not None
            ):
                aic_z_norm = float(new_latent.z_harm_suffering.norm().item())
            aic_drive = 0.0
            if self.goal_state is not None:
                aic_drive = float(getattr(self.goal_state, "_last_drive_level", 0.0))
            # SD-032e: if pACC is active, AIC sees the effective (sensitised)
            # drive_level with one-step lag -- the pACC tick in select_action()
            # produced drive_bias from the previous step's z_harm_a, which is
            # the correct causal ordering (pACC accumulates first, then AIC
            # reads the resulting baseline).
            if self.pacc is not None:
                aic_drive = self.pacc.effective_drive(aic_drive)
            aic_op_mode = None
            if self.salience is not None:
                aic_op_mode = self.salience.operating_mode
            self._aic_last_tick = self.aic.tick(
                z_harm_a_norm=aic_z_norm,
                drive_level=aic_drive,
                beta_gate_elevated=self.beta_gate.is_elevated,
                operating_mode=aic_op_mode,
                # SD-049 Phase 3: AIC urgency tracks worst-deficit axis.
                per_axis_drive=self._per_axis_drive_for_consumers(),
                per_axis_combiner=getattr(
                    self.config, "sd049_aic_per_axis_combiner", "max"
                ),
            )

        # SD-035: Amygdala analogue peer ticks (BLAAnalog + CeAAnalog).
        # Both modules read z_harm_a (produced by SD-011 AffectiveHarmEncoder
        # above) and run in sense() so downstream consumers in
        # select_action() (SalienceCoordinator for CeA, cached BLA outputs
        # for future hippocampal consumer wiring) see the current-tick
        # outputs without a one-step lag beyond the natural one from
        # z_harm_a_pred (which uses the previous step's E2_harm_a output).
        if (self.bla is not None or self.cea is not None):
            z_harm_a_cur = new_latent.z_harm_a
            if z_harm_a_cur is not None:
                if self.bla is not None:
                    # z_harm_a_pred_prev is produced in select_action() by the
                    # E2_harm_a forward pass using the previous step's
                    # z_harm_a and action. None on the first step of an
                    # episode -- BLA handles None by skipping remap_signal
                    # (no PE available).
                    z_harm_a_pred = self._harm_a_pred_prev
                    # Read path: use stored exploration-trace arousal tags as
                    # the hippocampal context for retrieval_bias.
                    arousal_tags_in_context = None
                    if self.hippocampal is not None:
                        arousal_tags_in_context = self.hippocampal.get_exploration_arousal_tags()
                    # Conservative attribution proxy: slot-attention over
                    # ContextMemory supplies per-code contributions until a
                    # dedicated harm-forward attribution head lands.
                    candidate_code_contributions = self._get_context_memory_code_contributions(
                        new_latent.z_self,
                        new_latent.z_world,
                    )
                    # SD-037 MECH-281 motor-coupling axis: BLA consolidation
                    # gain scaled by override_signal when both substrates on.
                    # broadcast_override.tick() ran earlier in sense() so the
                    # cached signal is current.
                    _ov_sig_bla = (
                        float(self.broadcast_override.override_signal)
                        if self.broadcast_override is not None
                        else 0.0
                    )
                    _ov_bla_gain = float(
                        getattr(self.config, "override_bla_encoding_gain", 0.0)
                    )
                    self._bla_last_output = self.bla.tick(
                        z_harm_a=z_harm_a_cur.detach(),
                        z_harm_a_pred=(
                            z_harm_a_pred.detach()
                            if z_harm_a_pred is not None
                            else None
                        ),
                        candidate_code_contributions=candidate_code_contributions,
                        arousal_tags_in_context=arousal_tags_in_context,
                        step_index=self._step_count,
                        simulation_mode=False,
                        override_signal=_ov_sig_bla,
                        override_encoding_gain=_ov_bla_gain,
                    )
                    self._episode_bla_peak_tag = max(
                        self._episode_bla_peak_tag,
                        float(self._bla_last_output.arousal_tag),
                    )
                    self._episode_bla_peak_encoding_gain = max(
                        self._episode_bla_peak_encoding_gain,
                        float(self._bla_last_output.encoding_gain),
                    )
                    if self._bla_last_output.remap_signal:
                        self._apply_bla_context_remap(
                            self._bla_last_output.remap_signal,
                            new_latent.z_self,
                            new_latent.z_world,
                        )
                if self.cea is not None:
                    # cortical_confirmation is reserved for a future cortical
                    # gate wired off the AIC / dACC fast-signal path.
                    # Passing None here lets the fast_prime pulse decay
                    # naturally per its time constant.
                    # SD-037 MECH-281 motor-coupling axis: CeA fast-route
                    # scalars (mode_prior + fast_prime) amplified by
                    # override_signal. Bounded by mode_prior_log_odds_max
                    # inside the module so cortex is still not over-ruled.
                    _ov_sig_cea = (
                        float(self.broadcast_override.override_signal)
                        if self.broadcast_override is not None
                        else 0.0
                    )
                    _ov_cea_gain = float(
                        getattr(self.config, "override_cea_amplitude_gain", 0.0)
                    )
                    self._cea_last_output = self.cea.tick(
                        z_harm_a=z_harm_a_cur.detach(),
                        cue_features=None,
                        cortical_confirmation=None,
                        escapability_hint=None,
                        simulation_mode=False,
                        override_signal=_ov_sig_cea,
                        override_amplitude_gain=_ov_cea_gain,
                    )

        # SD-021: descending pain modulation (commitment-gated sensory harm attenuation).
        # When harm_descending_mod_enabled=True, attenuate z_harm (sensory-discriminative
        # stream) before it reaches E3 / E2_harm_s. z_harm_a is NOT attenuated: affective
        # load (C-fiber) persists regardless of commitment -- you can tolerate expected
        # pain but it still matters motivationally.
        #
        # Two gating paths (selected by use_aic_analog):
        #   use_aic_analog=True  -- multiplier is self.aic.harm_s_gain (drive- and
        #     operating-mode-gated). This is the SD-032c-subsumed path -- resolves the
        #     EXQ-325a FAIL by making the descending branch a genuinely different
        #     function of state (not a redundant beta_gate flag).
        #   use_aic_analog=False -- legacy path: if beta_gate is elevated, multiplier is
        #     config.descending_attenuation_factor; otherwise 1.0.
        #
        # Biological basis: pgACC / AIC -> PAG -> RVM descending inhibitory pathway
        # provides endogenous analgesia during committed escape/approach (Basbaum 1984,
        # Keltner 2006). The AIC-gated routing is the biologically correct one
        # (Craig 2009 AIC as interoceptive-salience hub with descending efferents);
        # the legacy raw-beta_gate routing is retained for backward compatibility.
        # MECH-094: applies to waking observation stream (not replay content).
        if (
            getattr(self.config, "harm_descending_mod_enabled", False)
            and new_latent.z_harm is not None
        ):
            if self.aic is not None:
                attn = float(self.aic.harm_s_gain)
                if attn < 1.0:
                    new_latent.z_harm = new_latent.z_harm * attn
            elif self.beta_gate.is_elevated:
                attn = getattr(self.config, "descending_attenuation_factor", 0.5)
                new_latent.z_harm = new_latent.z_harm * attn

        # SD-014 h-component: write post-attenuation z_harm_s norm to VALENCE_HARM_DISCRIMINATIVE.
        # Using the post-attenuation value means SD-021 commitment gating automatically produces
        # a stale (attenuated) h at committed-state nodes -- exactly the analgesia-as-underestimated-h
        # signature needed for the SD-021/SD-014 cross-connection (surprise spike post-execution).
        if (
            getattr(self.config, "valence_harm_enabled", False)
            and new_latent.z_harm is not None
            and new_latent.z_world is not None
            and hasattr(self.residue_field, "update_valence")
        ):
            h_val = float(new_latent.z_harm.norm().item())
            self.residue_field.update_valence(
                new_latent.z_world,
                component=VALENCE_HARM_DISCRIMINATIVE,
                value=h_val,
                hypothesis_tag=False,
            )

        # SD-016 Part B2 (2026-04-25, EXQ-477 follow-up):
        # Per-tick ContextMemory write hook. When sd016_writepath_mode is
        # "sense_only" or "both", concatenate (z_self, z_world) and write to
        # ContextMemory via context_memory.write(). The MECH-120 SHY offline
        # gate is checked HERE at the call site -- ContextMemory.write itself
        # is NOT internally _offline_mode-gated (only update_from_observation
        # is), so the explicit guard is required to keep waking writes from
        # overwriting fresh schema slots installed during SWS.
        # Detach to keep the write outside the autograd graph; the write
        # itself is wrapped in torch.no_grad() inside ContextMemory.write.
        # See REE_assembly/docs/architecture/context_memory_writepath_fix.md.
        _wp_mode = getattr(self.e1.config, "sd016_writepath_mode", "off")
        if (
            _wp_mode in ("sense_only", "both")
            and not self.e1._offline_mode
            and new_latent.z_self is not None
            and new_latent.z_world is not None
        ):
            obs_state = torch.cat(
                [new_latent.z_self.detach(), new_latent.z_world.detach()],
                dim=-1,
            )
            self.e1.context_memory.write(obs_state)

        # Detach before storing: prevents EMA from linking computational graphs
        # across time steps. Without detach, optimizer.step() modifies weights
        # in-place, invalidating the old graph's version -- causing RuntimeError
        # "modified by an inplace operation" on the next backward() call.
        self._current_latent = new_latent.detach()
        # MECH-165: record state for exploration trajectory
        self._record_exploration_state()

        # MECH-258 / SD-032b: cache current z_harm_a so the next step's dACC
        # computation has access to both z_harm_a_prev (input to E2_harm_a) and
        # z_harm_a_current (realised target). Detached to avoid graph retention.
        if new_latent.z_harm_a is not None:
            self._harm_a_prev = new_latent.z_harm_a.detach().clone()

        # Tick-boundary queue flush (MECH-288 / MECH-287): drain any V_s events
        # left from the previous tick before generating this tick's events.
        # Phase 3 consumers (MECH-284 staleness accumulator, MECH-269 anchor-reset
        # T3) will call drain_boundary_events() / drain_broadcast_events() in
        # select_action() once they are implemented. Until Phase 3 lands nothing
        # in select_action() drains the queues, so they grow without bound within
        # an episode (confirmed telemetry artefact in EXQ-476b: queue length read
        # each tick inflated counts ~N/2x). Running the flush at the START of
        # each sense() call -- before new events are generated -- keeps the queues
        # bounded to at-most-one-tick's events and gives Phase 3 the correct
        # per-tick event set when it lands. No-op if the flags are off.
        if self.hippocampal is not None:
            if getattr(self.hippocampal.config, "use_event_segmenter", False):
                self.hippocampal.drain_boundary_events()
            if getattr(self.hippocampal.config, "use_invalidation_trigger", False):
                self.hippocampal.drain_broadcast_events()

        # MECH-288 (Phase 2 of V_s invalidation runtime): tick the hierarchical
        # event segmenter on the waking observation stream. Boundary events are
        # queued on HippocampalModule for downstream MECH-287 broadcast /
        # MECH-269 anchor-reset consumers. No-op when use_event_segmenter is
        # False. Runs AFTER latent encoding and BEFORE per-stream V_s update
        # so the Phase 3 V_s-anchor-reset path can consume both signals on the
        # same tick without a one-step lag.
        # MECH-094: this call is on the waking sense path. Replay / simulation
        # routes that need segment IDs use force_boundary() with an explicit
        # reason; the segmenter does not silently advance IDs from hypothesised
        # content.
        if self.hippocampal is not None and getattr(
            self.hippocampal.config, "use_event_segmenter", False
        ) and self.hippocampal.event_segmenter is not None:
            latent_dict = {
                "z_world": new_latent.z_world,
                "z_self": new_latent.z_self,
                "z_harm": new_latent.z_harm,
                "z_harm_s": new_latent.z_harm,
                "z_harm_a": new_latent.z_harm_a,
                "z_beta": new_latent.z_beta,
                "z_goal": (
                    self.goal_state.z_goal if self.goal_state is not None else None
                ),
            }
            events = self.hippocampal.event_segmenter.step(
                latent_dict=latent_dict,
                pe_dict=None,
                t=int(self._step_count),
            )
            if events:
                self.hippocampal._boundary_event_queue.extend(events)

            # MECH-287 (Phase 2 iv of V_s invalidation runtime): broadcast
            # invalidation trigger. Subscribes to the BoundaryEvents just
            # emitted by MECH-288 and re-emits them as graded BroadcastEvent
            # objects. Graded output: broadcast_strength = posterior * gain
            # (NO binary thresholding of strength). Phasic/tonic guardrail:
            # a rolling tonic-noise estimate suppresses the whole tick's
            # broadcast when it exceeds threshold (Aston-Jones & Cohen 2005;
            # Clewett 2025 failure signature 2).
            #
            # Verdict-3 dissociation: the trigger has no independent
            # comparator -- it is a BoundaryEvent subscriber. When
            # use_event_segmenter is False (no BoundaryEvents produced),
            # the trigger is silent regardless of any internal state.
            # Tick the trigger AFTER the segmenter so this tick's boundaries
            # are visible.
            if (
                getattr(self.hippocampal.config, "use_invalidation_trigger", False)
                and self.hippocampal.invalidation_trigger is not None
            ):
                broadcasts = self.hippocampal.invalidation_trigger.step(
                    boundary_events=events,
                    t=int(self._step_count),
                )
                if broadcasts:
                    self.hippocampal._broadcast_event_queue.extend(broadcasts)
        elif self.hippocampal is not None and getattr(
            self.hippocampal.config, "use_invalidation_trigger", False
        ) and self.hippocampal.invalidation_trigger is not None:
            # Segmenter disabled but trigger enabled: tick the trigger with
            # an empty boundary list so its tonic history advances in step
            # with the simulation clock. No broadcasts can fire (verdict-3
            # dissociation).
            self.hippocampal.invalidation_trigger.step(
                boundary_events=[], t=int(self._step_count)
            )

        # MECH-269 base substrate (Phase 1, 2026-04-22): per-stream
        # verisimilitude V_s update. Identity-prediction proxy across
        # registered streams, EMA-smoothed. No-op when use_per_stream_vs is
        # False. Populates self.hippocampal.per_stream_vs as the OBSERVABLE
        # foundation for Phase 2 MECH-287 broadcast trigger and MECH-284
        # staleness accumulator wiring.
        if self.hippocampal is not None and getattr(
            self.hippocampal.config, "use_per_stream_vs", False
        ):
            self.hippocampal.update_per_stream_vs(
                new_latent, goal_state=self.goal_state
            )

        # MECH-314a Phase 2 (Candidate 5A): append the current waking-tick
        # z_world to the rolling visitation buffer (the "visitation" novelty
        # comparison set for StructuredCuriosity). MECH-094-gated: replay /
        # DMN ticks (hypothesis_tag=True) do NOT write, so the novelty
        # landscape reflects only genuinely-visited waking states. No-op when
        # the buffer is None (default "residue" novelty source).
        if self._zworld_visitation_buffer is not None:
            if not bool(getattr(new_latent, "hypothesis_tag", False)):
                if new_latent.z_world is not None:
                    self._zworld_visitation_buffer.append(
                        new_latent.z_world[0].detach().clone()
                    )

        # SD-039 population layer (2026-04-27): build the AnchorGoalPayload
        # once per tick from the current waking-stream signals (z_goal from
        # GoalState, VALENCE_WANTING readout from ResidueField at z_world,
        # arousal_tag from BLA, last_vs as cross-stream mean of per_stream_vs,
        # staleness_at_write from MECH-284 accumulator snapshot). Returns
        # None when the AnchorSet substrate is disabled or the master flag
        # use_sd039_anchor_payload is False -- in which case all downstream
        # call sites are bit-identical to pre-SD-039. simulation_mode=False
        # because sense() is the waking observation stream (MECH-094 gate).
        sd039_payload = None
        if (
            self.hippocampal is not None
            and self.hippocampal.anchor_set is not None
        ):
            sd039_payload = self.hippocampal.build_goal_payload(
                latent_state=new_latent,
                goal_state=self.goal_state,
                residue_field=self.residue_field,
                bla_output=self._bla_last_output,
                current_step=int(self._step_count),
                simulation_mode=False,
            )

        # MECH-269 Phase 2 (ii): scale-tagged anchor set. Installs / remaps
        # anchors for each BoundaryEvent emitted this tick (from the local
        # events list, same pattern as MECH-287) and advances hysteresis
        # counters against the per_stream_vs scores just updated. Runs AFTER
        # update_per_stream_vs so hysteresis reads the current-tick V_s.
        # No-op when use_anchor_sets is False.
        if (
            self.hippocampal is not None
            and getattr(self.hippocampal.config, "use_anchor_sets", False)
            and self.hippocampal.anchor_set is not None
        ):
            anchor_events = events if "events" in locals() else []
            self.hippocampal.tick_anchor_set(
                new_latent, anchor_events, goal_payload=sd039_payload
            )

        # MECH-269 Phase 2 (iii, T4): per-region per-stream V_s update.
        # Step (a): apply any MECH-287 broadcast-driven resets. Broadcasts
        # on (scale, segment_id_old) drop the matching per_region_vs entry
        # and mark_inactive the matching anchor (T3 hysteresis-shortcut
        # reset path). We peek the broadcast queue without draining so
        # Phase 3 consumers (MECH-284 staleness accumulator) still see
        # the events. Tick_anchor_set's consume_boundary_events has
        # already applied the dual-trace remap for boundary events; this
        # broadcast path is the explicit safety net keyed on source_scale
        # and source_segment_id_old.
        # Step (b): iterate active anchors and compute per-region V_s.
        # No-op when use_per_region_vs is False.
        if self.hippocampal is not None and getattr(
            self.hippocampal.config, "use_per_region_vs", False
        ):
            broadcasts_for_regions = list(
                getattr(self.hippocampal, "_broadcast_event_queue", [])
            )
            if broadcasts_for_regions:
                self.hippocampal.apply_invalidation_broadcasts_to_regions(
                    broadcasts_for_regions,
                    goal_payload=sd039_payload,
                )
            self.hippocampal.update_per_region_vs(
                new_latent, goal_state=self.goal_state
            )

        # MECH-269b: refresh per-stream snapshots from current latent when
        # V_s[s] >= vs_gate_snapshot_refresh_threshold. Runs AFTER
        # update_per_stream_vs / update_per_region_vs so the snapshot reflects
        # the current-tick V_s reading. No-op when the gate is None.
        if self.vs_rollout_gate is not None:
            self.vs_rollout_gate.update_snapshots(
                new_latent,
                self.hippocampal.per_stream_vs,
                goal_state=self.goal_state,
            )

        # MECH-302: tick the suffering-derivative comparator on the waking
        # observation stream only. Simulation mode (hypothesis_tag) prevents
        # buffer advance (MECH-094). The resulting flag is consumed and cleared
        # in select_action() adjacent to the MECH-091 urgency block.
        if self.suffering_comparator is not None and new_latent.z_harm_a is not None:
            sdc_norm = float(new_latent.z_harm_a.norm().item())
            sim_mode = bool(getattr(new_latent, "hypothesis_tag", False))
            self._relief_completion_event = self.suffering_comparator.tick(
                sdc_norm, sim_mode
            )

        # SD-051 / MECH-304: tick the conditioned safety store.
        # Updates EMA prototype when MECH-302 event fired this tick, then
        # returns cosine similarity to current z_world. Simulation mode
        # (hypothesis_tag) returns 0.0 without advancing the prototype (MECH-094).
        if self.conditioned_safety_store is not None:
            sim_mode = bool(getattr(new_latent, "hypothesis_tag", False))
            self._conditioned_safety_signal = self.conditioned_safety_store.update(
                new_latent.z_world,
                event_fired=self._relief_completion_event,
                sim_mode=sim_mode,
            )

        # MECH-303: contextual passive safety terrain accumulation.
        # Each waking step where z_harm_a norm is below the quiescent threshold,
        # write a small increment to the safety terrain at current z_world.
        # Hypothesis_tag=True blocks accumulation (MECH-094: waking path only).
        if (
            getattr(self.config, "use_contextual_safety_terrain", False)
            and new_latent.z_harm_a is not None
            and new_latent.z_world is not None
            and hasattr(self.residue_field, "accumulate_safety")
        ):
            harm_norm = float(new_latent.z_harm_a.norm().item())
            harm_thresh = float(getattr(self.config, "contextual_safety_harm_threshold", 0.05))
            if harm_norm < harm_thresh:
                accum_w = float(getattr(self.config, "contextual_safety_accum_weight", 0.01))
                hyp_tag = bool(getattr(new_latent, "hypothesis_tag", False))
                self.residue_field.accumulate_safety(
                    new_latent.z_world,
                    safety_magnitude=accum_w,
                    hypothesis_tag=hyp_tag,
                )

        # GAP-4 / MECH-273: buffer (z_harm_s, action) pairs for sleep WRITEBACK.
        # Collected only on the waking observation stream (not simulation/replay).
        if (
            new_latent.z_harm is not None
            and self._last_action is not None
            and not bool(getattr(new_latent, "hypothesis_tag", False))
        ):
            self._harm_replay_buffer.append((
                new_latent.z_harm.detach().clone(),
                self._last_action.detach().clone(),
            ))
            if len(self._harm_replay_buffer) > 1000:
                self._harm_replay_buffer = self._harm_replay_buffer[-1000:]

        # MECH-090 R-c continuation (nav_competence axis) -- Phase-2 follow-on
        # (2026-06-02). Advance the commit-readiness EMA from the per-tick
        # env-emitted nav-competence/motor-program-readiness outcome the caller
        # forwards (info["mech090_readiness_outcome"]). The 2026-05-29 landing
        # wired the CONSUMER (is_above_floor AND-composed at both beta_gate
        # elevate sites) + the notify_outcome seam, but left no automatic SOURCE
        # -- the across-tick axis sat fail-open (readiness pinned at the initial
        # 1.0) in any ecological run because nothing advanced the EMA. This
        # closes that gap: the substrate now advances readiness automatically
        # from an env signal without an out-of-band harness push.
        # Bit-identical OFF: no-op when self.commit_readiness is None (master
        # flags off) OR when mech090_readiness_outcome is None (key absent; the
        # CommitReadiness.update None-sentinel returns readiness unchanged).
        # MECH-094: simulation/replay ticks (hypothesis_tag) do not advance the
        # EMA -- update() honours simulation_mode.
        if self.commit_readiness is not None:
            self.commit_readiness.update(
                outcome_signal=mech090_readiness_outcome,
                simulation_mode=bool(getattr(new_latent, "hypothesis_tag", False)),
            )

        return new_latent

    def sense_flat(self, observation: torch.Tensor) -> LatentState:
        """
        SENSE step with flat observation (backward compat / pre-SD-005 wiring).

        Splits observation into [body, world] based on configured dims.
        """
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        observation = observation.to(self.device).float()
        body_dim = self.config.latent.body_obs_dim
        obs_body  = observation[:, :body_dim]
        obs_world = observation[:, body_dim:]
        return self.sense(obs_body, obs_world)

    def _refresh_vs_gate_staleness(self) -> None:
        """Refresh self._vs_gate_staleness_cache from the per-stream
        aggregator on HippocampalModule (Q-040b strong reading).

        No-op when the gate is absent or staleness lookup is disabled --
        the cache stays empty and gate() / gate_stream() fall back to the
        legacy raw-V_s threshold path. Called once per waking tick at the
        top of _e1_tick; the cached dict is reused for both the E1 gate
        call and the later E2_harm_a gate call in select_action.
        """
        if self.vs_rollout_gate is None:
            return
        if not self.vs_rollout_gate.config.use_staleness_lookup:
            self._vs_gate_staleness_cache = {}
            return
        self._vs_gate_staleness_cache = (
            self.hippocampal.compute_per_stream_staleness()
        )

    def _e1_tick(self, latent_state: LatentState) -> torch.Tensor:
        """
        E1 tick: run E1 prediction and push to ThetaBuffer (MECH-089).

        Returns E1 world-domain prior for HippocampalModule (SD-002).
        """
        # Q-040b strong reading: refresh per-stream staleness from
        # MECH-284 once per tick. Cache is consumed by all gate / gate_stream
        # calls in this tick. No-op when staleness lookup is disabled.
        self._refresh_vs_gate_staleness()
        # MECH-269b: gate the latent for E1 forward consumption. Streams whose
        # V_s falls below vs_gate_e1_threshold are substituted with held
        # snapshots; aligned streams pass through. No-op (gated_for_e1 is
        # latent_state) when vs_rollout_gate is None.
        if self.vs_rollout_gate is not None:
            gated_for_e1 = self.vs_rollout_gate.gate(
                latent_state,
                self.hippocampal.per_stream_vs,
                side="e1",
                goal_state=self.goal_state,
                per_stream_staleness=(
                    self._vs_gate_staleness_cache or None
                ),
            )
        else:
            gated_for_e1 = latent_state

        total_state = torch.cat(
            [gated_for_e1.z_self, gated_for_e1.z_world], dim=-1
        )

        # Store in experience buffers (canonical un-gated values: experience
        # buffers feed training / replay, not forward-prediction inputs).
        self._self_experience_buffer.append(latent_state.z_self.detach().clone())
        self._world_experience_buffer.append(latent_state.z_world.detach().clone())
        for buf in [self._self_experience_buffer, self._world_experience_buffer]:
            if len(buf) > 1000:
                del buf[:-1000]

        # Run E1 for prior generation. z_goal is also under MECH-269b gating
        # because GoalState.z_goal is one of the streams MECH-269 Phase 1
        # tracks; gate_stream returns the held snapshot when V_s_z_goal drops
        # below the E1 threshold.
        _z_goal_input = None
        _goal_cfg = getattr(self.config, "goal", None)
        if (self.goal_state is not None
                and _goal_cfg is not None
                and _goal_cfg.e1_goal_conditioned
                and self.goal_state.is_active()):
            _z_goal_input = self.goal_state.z_goal
            if self.vs_rollout_gate is not None:
                _z_goal_input = self.vs_rollout_gate.gate_stream(
                    "z_goal",
                    _z_goal_input,
                    self.hippocampal.per_stream_vs,
                    side="e1",
                    per_stream_staleness=(
                        self._vs_gate_staleness_cache or None
                    ),
                )
        _, e1_prior = self.e1(total_state, z_goal=_z_goal_input)

        # MECH-089: push z_self, z_world estimates to ThetaBuffer
        self.theta_buffer.update(latent_state.z_world, latent_state.z_self)

        # MECH-294: push the current per-stream content latents into the open
        # theta-packet binding window (goal_latent, risk_sensory=z_harm_s,
        # risk_affective=z_harm_a, + per-stream V_s for vintaging). The
        # state_summary slot is filled at seal time from theta_buffer.summary().
        # Waking call site (MECH-094: simulation_mode=False).
        if self.multi_content_theta_packet is not None:
            _pkt_z_goal = (
                self.goal_state.z_goal
                if (self.goal_state is not None and self.goal_state.is_active())
                else None
            )
            self.multi_content_theta_packet.observe(
                z_goal=_pkt_z_goal,
                z_harm_s=latent_state.z_harm,
                z_harm_a=latent_state.z_harm_a,
                per_stream_vs=getattr(self.hippocampal, "per_stream_vs", None),
                simulation_mode=False,
            )

        # MECH-093: update E3 rate from z_beta magnitude
        self.clock.update_e3_rate_from_beta(latent_state.z_beta)

        # MECH-216: cache schema salience from E1 readout head.
        schema_sal = self.e1.get_schema_salience()
        self._schema_salience = schema_sal.detach() if schema_sal is not None else None

        # MECH-307 Gap 4: cache the E1 forward prediction so that
        # update_schema_wanting can write at the predicted z_world location
        # rather than the agent's current z_world. Detached so the schema-
        # readout write site does not propagate gradients into E1.
        self._cached_e1_prior = e1_prior.detach() if e1_prior is not None else None

        # SD-016 (MECH-150/151/152): frontal cue-indexed integration.
        # Extract action_bias and terrain_weight from z_world-only ContextMemory query.
        # Detached: cue signals are modulation inputs, not part of current-step gradient graph.
        # Cached until next E1 tick (same theta-cycle rate as generate_prior).
        # MECH-269b: feed gated_for_e1.z_world (already substituted by snapshot
        # if V_s_z_world is below E1 threshold).
        if hasattr(self.e1, 'world_query_proj'):
            action_bias, terrain_weight = self.e1.extract_cue_context(
                gated_for_e1.z_world.detach()
            )
            self._cue_action_bias    = action_bias.detach()
            self._cue_terrain_weight = terrain_weight.detach()
        else:
            self._cue_action_bias    = None
            self._cue_terrain_weight = None

        return e1_prior

    def _compute_persistence_appraisal(
        self, z_world: torch.Tensor
    ) -> Optional[PersistenceAppraisal]:
        """MECH-340 / Q-053: one-shot persistence gate inputs for ghost bank.

        Returns None when the MECH-340 gate is off (rank() uses the
        configured missing-appraisal default). When on, maps goal proximity,
        prior hippocampal completion, and E3 commitment into
        control_efficacy / goal_unattainability (not staleness / failure).
        """
        bank_cfg = getattr(
            getattr(self.hippocampal, "config", None),
            "ghost_goal_bank_config",
            None,
        )
        if bank_cfg is None or not bool(
            getattr(bank_cfg, "use_persistence_efficacy_gate", False)
        ):
            self._last_persistence_appraisal = None
            return None

        goal_active = (
            self.goal_state is not None and self.goal_state.is_active()
        )
        proximity: Optional[float] = None
        if goal_active:
            prox_t = self.goal_state.goal_proximity(z_world)
            proximity = float(prox_t.mean().item())

        commit_state = self.e3.get_commitment_state()
        appraisal = compute_agent_persistence_appraisal(
            goal_active=goal_active,
            goal_proximity=proximity,
            prior_completion_signal=float(
                getattr(self.hippocampal, "_last_completion_signal", 0.0)
            ),
            e3_is_committed=bool(commit_state.get("is_committed", False)),
            e3_committed_now=bool(commit_state.get("committed_now", False)),
            cfg=self.hippocampal.config.persistence_appraisal_compute,
        )
        self._last_persistence_appraisal = appraisal
        return appraisal

    def _e3_tick(
        self,
        latent_state: LatentState,
        e1_prior: torch.Tensor,
        num_candidates: Optional[int] = None,
    ) -> List[Trajectory]:
        """
        E3 tick: propose trajectories via HippocampalModule, select via E3.

        E3 receives theta-cycle summary (MECH-089), not raw z_world.

        Returns candidate trajectories.
        """
        # MECH-089: E3 consumes theta-cycle summary
        z_world_for_e3 = self.theta_buffer.summary()  # theta-averaged z_world

        # HippocampalModule proposes in action-object space (SD-004).
        # SD-016 (MECH-151): pass cached action_bias so each action_object()
        # call in the CEM loop is contextually biased by z_world cue retrieval.
        # MECH-293: thread the current goal through to propose_trajectories
        # so the MECH-292 ghost-goal bank can rank anchors by goal_match.
        # When goal_state is absent / inactive, current_z_goal=None -> the
        # ghost branch is silent (bank.rank() returns []).
        _current_z_goal = (
            self.goal_state.z_goal
            if (self.goal_state is not None and self.goal_state.is_active())
            else None
        )
        _persistence_appraisal = self._compute_persistence_appraisal(z_world_for_e3)
        # SD-061: difficulty-gated proposal-entropy. When stuck (the previous
        # tick's stuck_score), transiently WIDEN the CEM candidate set and lift
        # the within-class CEM sampling temperature on the PROPOSAL layer
        # (ARC-018), restoring both afterward. No-op when the regulator is None
        # or stuck_score=0 -> bit-identical proposal. Scoring / commitment /
        # selection authority are untouched (a hard problem widens proposals,
        # not behaviour).
        _dgpe_num_candidates = num_candidates
        _dgpe_temp_restore: Optional[float] = None
        if self.difficulty_gated_proposal_entropy is not None:
            _dgpe_extra, _dgpe_temp_gain = (
                self.difficulty_gated_proposal_entropy.compute_proposal_gain(
                    self._last_stuck_score, simulation_mode=False
                )
            )
            if _dgpe_extra > 0:
                _dgpe_base = (
                    num_candidates
                    if num_candidates is not None
                    else getattr(self.hippocampal.config, "num_candidates", 0)
                )
                _dgpe_num_candidates = int(_dgpe_base) + int(_dgpe_extra)
            if _dgpe_temp_gain > 1.0 and hasattr(
                self.hippocampal.config, "differentiable_cem_temperature"
            ):
                _dgpe_temp_restore = float(
                    self.hippocampal.config.differentiable_cem_temperature
                )
                self.hippocampal.config.differentiable_cem_temperature = (
                    _dgpe_temp_restore * _dgpe_temp_gain
                )
        try:
            candidates = self.hippocampal.propose_trajectories(
                z_world=z_world_for_e3,
                z_self=latent_state.z_self,
                num_candidates=_dgpe_num_candidates,
                e1_prior=e1_prior,
                action_bias=self._cue_action_bias,
                current_z_goal=_current_z_goal,
                persistence_appraisal=_persistence_appraisal,
            )
        finally:
            if _dgpe_temp_restore is not None:
                self.hippocampal.config.differentiable_cem_temperature = (
                    _dgpe_temp_restore
                )
        self._committed_candidates = candidates

        # MECH-294: this is the theta-cycle (E3-heartbeat) boundary -- push the
        # proposer's lead first-action object into the open packet window, then
        # SEAL the packet for this cycle and expose it as agent.last_theta_packet
        # for the proposer / E3 to read as a joint object on the next cycle. The
        # state_summary slot reuses the MECH-089 averaged z_world (z_world_for_e3,
        # in scope above). Waking call site (MECH-094: simulation_mode=False).
        if self.multi_content_theta_packet is not None:
            _first_action = None
            if candidates:
                _lead = candidates[0]
                _acts = getattr(_lead, "actions", None)
                if _acts is not None and _acts.shape[1] > 0:
                    _first_action = _acts[:, 0, :]
            self.multi_content_theta_packet.observe_action_proposal(
                _first_action, simulation_mode=False
            )
            self.last_theta_packet = self.multi_content_theta_packet.seal(
                state_summary=z_world_for_e3, simulation_mode=False
            )

        # ARC-028 / MECH-105: hippocampal completion signal -> BetaGate release.
        # compute_completion_signal() scores all proposals; high-quality trajectory
        # found -> dopamine analog -> beta drops -> gate opens (Lisman & Grace 2005).
        # MECH-090 bistable: release triggered by completion, not by variance re-eval.
        if self.config.heartbeat.beta_gate_bistable and self.beta_gate.is_elevated:
            completion = self.hippocampal.compute_completion_signal(candidates)
            released = self.beta_gate.receive_hippocampal_completion(completion)
            # MECH-269 / MECH-090: clear V_s -> commit release snapshot when
            # beta releases via hippocampal completion signal so the snapshot
            # does not leak across commitment boundaries.
            if released:
                self._committed_anchor_keys = None
            # MECH-290: backward credit sweep on goal arrival (Foster & Wilson 2006).
            # Fires synchronously on waking path when BetaGate releases via
            # hippocampal completion signal. Sweeps committed trajectory backward,
            # updating VALENCE_WANTING proportional to outcome_quality * gamma^t.
            # No-op when use_backward_credit_sweep is False (backward compat).
            if released and getattr(
                self.hippocampal.config, "use_backward_credit_sweep", False
            ):
                self.hippocampal.backward_credit_sweep(
                    self.hippocampal._last_completion_signal
                )

        return candidates

    def generate_trajectories(
        self,
        latent_state: LatentState,
        e1_prior: torch.Tensor,
        ticks: dict,
        num_candidates: Optional[int] = None,
        sequence_in_progress: bool = False,
    ) -> List[Trajectory]:
        """
        GENERATE step: propose candidate trajectories.

        Multi-rate: only re-generate when e3_tick fires.
        Between ticks: return cached candidates (MECH-057a gate).
        """
        # If E3 hasn't ticked and we have cached candidates, use them
        if (
            not ticks["e3_tick"]
            and self._committed_candidates is not None
        ):
            return self._committed_candidates

        # If gate enabled and sequence in progress and no new tick, hold
        if (
            self.config.action_loop_gate_enabled
            and sequence_in_progress
            and not ticks["e3_tick"]
            and self._committed_candidates is not None
        ):
            return self._committed_candidates

        return self._e3_tick(latent_state, e1_prior, num_candidates)

    def _curiosity_candidate_summaries(
        self, candidates: List[Trajectory]
    ) -> Optional[torch.Tensor]:
        """MECH-314a Phase-2 amend (V3-EXQ-648a): per-candidate novelty
        signature sourced from the SD-056-trained action-conditional
        e2.world_forward(z0, a_i) predictions.

        Returns None when curiosity_candidate_source is the default
        "proposer" (the caller then uses the legacy proposer-summary
        reuse-chain, bit-identical to pre-amend) or when the e2
        world-forward / current latent are unavailable. Otherwise returns
        a [K, world_dim] tensor of per-candidate predicted next-z_world
        (z0 = current observed z_world, a_i = each candidate's first
        action), which feeds BOTH the 314a RBF novelty and the
        auto-augmentation _candidate_spread key in StructuredCuriosity.

        Root cause this fixes (V3-EXQ-648 autopsy 2026-06-07): the proposer
        first-step z_world (trajectory.world_states[:,0,:]) collapses to
        cross-candidate spread <0.01 under monostrategy, while the
        SD-056-trained e2.world_forward predictions carry spread ~0.11 --
        the representation the SD-056 readiness gate already validates.

        The e2.world_forward read is no_grad on the waking select_action
        path (no replay / memory write surface; MECH-094 not implicated).
        """
        if (
            getattr(self.config, "curiosity_candidate_source", "proposer")
            != "e2_world_forward"
        ):
            return None
        e2 = getattr(self, "e2", None)
        if e2 is None or self._current_latent is None or len(candidates) == 0:
            return None
        z0 = self._current_latent.z_world.detach()
        if z0.dim() == 1:
            z0 = z0.unsqueeze(0)
        z0 = z0[:1]  # [1, world_dim]
        adim = int(candidates[0].actions.shape[-1])
        act_rows: List[torch.Tensor] = []
        for c in candidates:
            act_rows.append(c.actions[:, 0, :].detach().reshape(adim))
        actions_K = torch.stack(act_rows, dim=0).to(
            device=z0.device, dtype=z0.dtype
        )  # [K, action_dim]
        z0_K = z0.expand(actions_K.shape[0], -1)  # [K, world_dim]
        with torch.no_grad():
            preds = e2.world_forward(z0_K, actions_K)  # [K, world_dim]
        return preds.detach()

    def _candidate_world_summaries(
        self, candidates: List[Trajectory]
    ) -> Optional[torch.Tensor]:
        """ARC-065 GAP-A: SHARED per-candidate cand_world_summaries sourced from
        the SD-056-trained action-conditional e2.world_forward(z0, a_i)
        predictions instead of the collapsed proposer first-step z_world.

        Returns None when candidate_summary_source is the default "proposer"
        (callers then build cand_world_summaries from
        trajectory.world_states[:,0,:], bit-identical to pre-GAP-A) or when the
        e2 world-forward / current latent are unavailable. Otherwise returns a
        [K, world_dim] tensor consumed by the E3-side bias channels
        (lateral_pfc / ofc / mech295 / gated_policy / tonic_vigor).

        Root cause this fixes (V3-EXQ-614e autopsy 2026-06-07): the proposer
        first-step z_world collapses to cross-candidate spread ~0 under
        monostrategy (cand_world_pairwise_dist=0.0000), so every E3-side bias
        channel sees a class-uniform candidate pool; the SD-056-trained
        e2.world_forward predictions carry per-action spread. The
        modulatory-bias-selection-authority gate (V3-EXQ-643a) then lets the
        now-divergent bias reach the committed argmin. Shared-channel sibling of
        _curiosity_candidate_summaries (which fixes the curiosity channel only).

        The e2.world_forward read is no_grad on the waking select_action path
        (no replay / memory write surface; MECH-094 not implicated).
        """
        if (
            getattr(self.config, "candidate_summary_source", "proposer")
            != "e2_world_forward"
        ):
            return None
        e2 = getattr(self, "e2", None)
        if e2 is None or self._current_latent is None or len(candidates) == 0:
            return None
        z0 = self._current_latent.z_world.detach()
        if z0.dim() == 1:
            z0 = z0.unsqueeze(0)
        z0 = z0[:1]  # [1, world_dim]
        adim = int(candidates[0].actions.shape[-1])
        act_rows: List[torch.Tensor] = []
        for c in candidates:
            act_rows.append(c.actions[:, 0, :].detach().reshape(adim))
        actions_K = torch.stack(act_rows, dim=0).to(
            device=z0.device, dtype=z0.dtype
        )  # [K, action_dim]
        z0_K = z0.expand(actions_K.shape[0], -1)  # [K, world_dim]
        with torch.no_grad():
            preds = e2.world_forward(z0_K, actions_K)  # [K, world_dim]
        return preds.detach()

    def select_action(
        self,
        candidates: List[Trajectory],
        ticks: dict,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        SELECT step: E3 selects trajectory; BetaGate controls propagation.

        If E3 hasn't ticked, return the held action (MECH-090).

        Layer 1 (MECH-090 trajectory stepping): when committed, step through
        a0->a1->a2->... using _committed_step_idx instead of repeating a0.

        Layer 2 (MECH-091 urgency interrupt): when beta is elevated and z_harm_a
        norm exceeds urgency_interrupt_threshold, abort commitment and re-select.
        """
        # SD-011: extract z_harm_a for E3 urgency gating and ethical cost amplification.
        z_harm_a = None
        if self._current_latent is not None and self._current_latent.z_harm_a is not None:
            z_harm_a = self._current_latent.z_harm_a

        # SD-019a: redirect E3 short-horizon urgency signal to z_harm_un
        # (medium EMA of z_harm_s) when harm_unpleasantness_channel is active.
        # z_harm_un encodes the unpleasantness dimension (NOT modulated by
        # controllability per Loffler 2018), making it the correct urgency_weight
        # input to E3.select() for short-horizon avoidance scoring.
        # z_harm_a (slow accumulator) is still consumed by dACC (SD-032b) and
        # pACC (SD-032e) via their own read paths; only E3 select + MECH-091
        # interrupt are redirected here.
        if (
            self.config.latent.use_harm_un
            and self._current_latent is not None
            and self._current_latent.z_harm_un is not None
        ):
            z_harm_a = self._current_latent.z_harm_un

        # Natural-commit latch-hold: clear the per-tick principled-release flags
        # at the top of the release region. The MECH-091 and rung-6 release sites
        # set them; the end-of-tick hold re-assertion reads them so the hold yields
        # to (does not re-fight) a principled release. No-op when the hold is off.
        self._ncl_mech091_fired = False
        self._ncl_lever_fired = False

        # MECH-091: urgency interrupt -- phase-reset commitment on high harm signal.
        # When beta is elevated (committed) and affective harm load is extreme,
        # abort the committed trajectory and fall through to fresh E3 selection.
        # SD-019a: when harm_unpleasantness_channel is active, urgency reads
        # z_harm_un (medium EMA of z_harm_s) as the interrupt signal instead of
        # z_harm_a (slow accumulator). z_harm_un provides faster-rising urgency
        # proportional to current unpleasantness, not accumulated suffering.
        if self.beta_gate.is_elevated and z_harm_a is not None:
            urgency_threshold = getattr(
                self.config.e3, "urgency_interrupt_threshold", 0.8
            )
            # SD-037 MECH-281 motor-coupling axis (2026-05-30): orexin-
            # recruited state lowers the urgency-interrupt threshold so the
            # committed motor program is more readily aborted under recruited
            # arousal (orexin -> escape-from-freeze on the motor side,
            # parallel to PAG alpha_override on the freeze-gate). At
            # override_beta_interrupt_gain=0.0 (default) this is exactly
            # urgency_threshold -> bit-identical to pre-MECH-281.
            _ov_beta_gain = float(
                getattr(self.config, "override_beta_interrupt_gain", 0.0)
            )
            if _ov_beta_gain != 0.0 and self.broadcast_override is not None:
                _ov_sig_beta = float(
                    max(0.0, min(1.0, self.broadcast_override.override_signal))
                )
                # Multiplier in [1 - gain, 1]; floor at 0.0 so the threshold
                # cannot go negative (which would short-circuit the interrupt
                # unconditionally even at zero urgency).
                _mult = max(0.0, 1.0 - _ov_beta_gain * _ov_sig_beta)
                urgency_threshold = urgency_threshold * _mult
            _urgency_signal = z_harm_a
            if (
                self.config.latent.use_harm_un
                and self._current_latent is not None
                and self._current_latent.z_harm_un is not None
            ):
                _urgency_signal = self._current_latent.z_harm_un
            # MECH-219 (SD-019b) MECH-091 redirect: source the urgency-interrupt
            # signal from the slow suffering channel z_harm_suffering when the
            # accumulator + redirect flag are on. Default off -> bit-identical.
            if (
                getattr(self.config, "use_harm_suffering_accumulator", False)
                and getattr(self.config, "harm_suffering_redirect_mech091", False)
                and self._current_latent is not None
                and self._current_latent.z_harm_suffering is not None
            ):
                _urgency_signal = self._current_latent.z_harm_suffering
            if float(_urgency_signal.norm().item()) > urgency_threshold:
                self.beta_gate.release()
                self._committed_step_idx = 0
                self._committed_anchor_keys = None
                # Natural-commit latch-hold yields to a genuine-threat interrupt
                # (safety -- the hold must NEVER override MECH-091).
                self._ncl_mech091_fired = True

        # MECH-342: maintenance-time readiness-driven commitment release (B3b).
        # Architecturally adjacent to the MECH-091 urgency block above, but a
        # DIFFERENT axis: MECH-091 releases on an acute z_harm threat spike;
        # MECH-342 releases on degraded EXECUTION READINESS (the same R-c
        # score_margin decisiveness + nav_competence signals that MECH-090
        # AND-composes to ADMIT a commitment) accumulated while already
        # beta-elevated. Graded bounded-accumulation (Resulaj 2009), conflict-
        # scaled (Cavanagh/Frank 2011), targeted + hysteretic with a
        # reengagement leak (Falasconi/Arber 2025 movement-specific vs Wessel
        # 2022 non-selective). Closes the V3-EXQ-592f reach gap. Fires with
        # z_harm_a BELOW threshold (vs MECH-091), with poor/low-decisiveness
        # options (opposite regime to ARC-028 completion), with a stable
        # schema (vs MECH-269b V_s), on the active beta latch at motor-program
        # timescale (vs MECH-340 ghost-goal). No-op when use_maintenance_release
        # is False (self.maintenance_release is None). DO NOT modify the
        # MECH-091 block above.
        if self.maintenance_release is not None and self.beta_gate.is_elevated:
            _mr_margin: Optional[float] = None
            _mr_n: int = 0
            # Decisiveness axis: per-candidate first-action margin off the
            # last completed E3 selection (REE lower-is-better -> margin =
            # sorted[1] - sorted[0]). last_scores is available every tick
            # (including between-E3-tick steps); a controlled state-machine
            # probe sets it directly. None when no prior selection -> the
            # decisiveness axis is inert this tick (nav axis still drives).
            if self.e3.last_scores is not None:
                try:
                    _mr_scores = self.e3.last_scores.detach()
                    _mr_n = int(_mr_scores.numel())
                    if _mr_n >= 2:
                        _mr_sorted, _ = torch.sort(_mr_scores)
                        _mr_margin = float(
                            _mr_sorted[1].item() - _mr_sorted[0].item()
                        )
                except (AttributeError, RuntimeError, TypeError):
                    _mr_margin = None
                    _mr_n = 0
            # nav_competence axis: current CommitReadiness EMA. None when the
            # readiness module is not instantiated -> the nav axis is inert
            # and only the decisiveness axis drives release.
            _mr_nav: Optional[float] = (
                self.commit_readiness.get_readiness()
                if self.commit_readiness is not None
                else None
            )
            if self.maintenance_release.tick(
                score_margin=_mr_margin,
                n_candidates=_mr_n,
                nav_competence=_mr_nav,
                simulation_mode=False,
            ):
                self.beta_gate.release()
                self._committed_step_idx = 0
                self._committed_anchor_keys = None
                # Clear the committed trajectory pointer so the decommit is
                # observable (the V3-EXQ-592f probe measures
                # e3._committed_trajectory presence as the decommit signal).
                self.e3._committed_trajectory = None
                # rung-6 amend: tear down the F-independent closure-plane commit-entry
                # latch on this de-commit so it cannot keep re-arming the eval.
                self.e3._closure_committed_active = False
                self.e3._closure_committed_trajectory = None  # C-STEP extension

        # Commit/release-DURATION lever: graded natural-commit-occupancy release
        # (rung-6 of f_dominance_conversion_ceiling; the duration face PARALLEL to
        # MECH-448). Fires when a natural F-driven commit has held too long
        # (Thura/Cisek graded urgency, rate scaled by entry decisiveness) OR when
        # its executed action sequence completes (Jin maintenance-co-extensive).
        # Distinct from MECH-342 above (which fires on DEGRADED readiness and is
        # silent on the healthy-but-prolonged decisive commit that monopolises the
        # latch -- the 460h ~2400-2600-step occupancy). No-op when the lever is
        # disabled (self.natural_commit_urgency is None). Only acts on a run armed
        # by note_commit_entry at a natural commit; tick() returns False otherwise.
        if (
            self.natural_commit_urgency is not None
            and self.beta_gate.is_elevated
        ):
            # action_sequence_complete: the agent has stepped through all of the
            # committed trajectory's actions (_committed_step_idx reaches the
            # horizon) rather than repeating the last action indefinitely.
            _ncur_seq_complete = False
            _ncur_traj = self.e3._committed_trajectory
            if _ncur_traj is not None:
                try:
                    _ncur_horizon = int(_ncur_traj.actions.shape[1])
                    _ncur_seq_complete = (
                        self._committed_step_idx >= _ncur_horizon
                    )
                except (AttributeError, IndexError, TypeError):
                    _ncur_seq_complete = False
            if self.natural_commit_urgency.tick(
                committed_run_length=self.beta_gate.committed_run_length,
                action_sequence_complete=_ncur_seq_complete,
                simulation_mode=False,
            ):
                self.beta_gate.release()
                self._committed_step_idx = 0
                self._committed_anchor_keys = None
                self.e3._committed_trajectory = None
                self.e3._closure_committed_active = False  # rung-6 amend: clear latch
                self.e3._closure_committed_trajectory = None  # C-STEP extension
                # Natural-commit latch-hold yields to the rung-6 duration release
                # -- this IS the lever shortening the held natural commit (the
                # whole point); the hold disarms so the occupancy stays shortened.
                self._ncl_lever_fired = True

        # SD-061: update the stuck-state detector every tick (not gated on beta
        # elevation -- impasse can build before commitment). Feeds the next
        # tick's _e3_tick proposal-gain via self._last_stuck_score (one-tick
        # lag, the standard regulator seam). All inputs are read defensively;
        # an absent signal is inert (None). No-op when the detector is None.
        if self.stuck_state_detector is not None:
            _ss_margin: Optional[float] = None
            _ss_n: int = 0
            try:
                if self.e3.last_scores is not None:
                    _ss_scores = self.e3.last_scores.detach()
                    _ss_n = int(_ss_scores.numel())
                    if _ss_n >= 2:
                        _ss_sorted, _ = torch.sort(_ss_scores)
                        _ss_margin = float(
                            _ss_sorted[1].item() - _ss_sorted[0].item()
                        )
            except (AttributeError, RuntimeError, TypeError):
                _ss_margin = None
                _ss_n = 0
            _ss_choice_difficulty: Optional[float] = None
            _ss_bundle = getattr(self, "_dacc_last_bundle", None)
            if isinstance(_ss_bundle, dict) and "choice_difficulty" in _ss_bundle:
                try:
                    _ss_choice_difficulty = float(_ss_bundle["choice_difficulty"])
                except (TypeError, ValueError):
                    _ss_choice_difficulty = None
            _ss_goal_prox: Optional[float] = None
            _ss_goal_salience: Optional[float] = None
            _ss_committed_class: Optional[int] = None
            if self.goal_state is not None and self.goal_state.is_active():
                try:
                    _ss_zw = self._current_latent.z_world
                    _ss_goal_prox = float(self.goal_state.goal_proximity(_ss_zw))
                    _ss_goal_salience = float(self.goal_state.goal_norm())
                except (AttributeError, RuntimeError, TypeError):
                    _ss_goal_prox = None
                    _ss_goal_salience = None
            try:
                _ss_traj = self.e3._committed_trajectory
                if _ss_traj is not None and _ss_traj.actions is not None:
                    _ss_committed_class = int(
                        torch.argmax(_ss_traj.actions[0, 0, :]).item()
                    )
            except (AttributeError, RuntimeError, TypeError, IndexError):
                _ss_committed_class = None
            self._last_stuck_score = self.stuck_state_detector.update(
                goal_proximity=_ss_goal_prox,
                score_margin=_ss_margin,
                n_candidates=_ss_n,
                committed_action_class=_ss_committed_class,
                choice_difficulty=_ss_choice_difficulty,
                goal_salience=_ss_goal_salience,
                simulation_mode=False,
            )

        # MECH-353 DECOMMIT consumer: if z_block asserted hard across the window
        # but the block persisted (assertion failed), release the blocked
        # commitment rather than escalate effort unboundedly -- gated by the
        # ARC-016 commitment-threshold (the prefrontal-analogue inhibition of
        # reactive aggression; Bertsch 2020). The decommit signal is set during
        # sense() by BlockedAgency.update(); here it is consumed only while beta
        # is elevated and ARC-016 permits abort. No-op when use_blocked_agency
        # is False. Mirrors the MECH-091 / MECH-342 release template.
        if self.blocked_agency is not None and self.beta_gate.is_elevated:
            _ba_out = self.blocked_agency.last_output()
            if _ba_out.decommit_signal:
                # ARC-016 gate: release only when E3 precision (confidence) is at
                # or below the threshold (low confidence -> abort permitted).
                # threshold <= 0.0 disables the gate (always permit).
                _arc016_max = float(
                    getattr(
                        self.config,
                        "blocked_agency_decommit_arc016_precision_max",
                        0.0,
                    )
                )
                _precision = float(getattr(self.e3, "current_precision", 0.0))
                _arc016_permits = (_arc016_max <= 0.0) or (_precision <= _arc016_max)
                if _arc016_permits:
                    self.beta_gate.release()
                    self._committed_step_idx = 0
                    self._committed_anchor_keys = None
                    self.e3._committed_trajectory = None
                    self.e3._closure_committed_active = False  # rung-6 amend: clear latch
                    self.e3._closure_committed_trajectory = None  # C-STEP extension

        # MECH-269 / MECH-090 read-side hook: V_s -> commit release.
        # If any anchor key snapshotted at commit entry has dropped out of the
        # active anchor set since then, the schema region the commitment was
        # anchored to has been invalidated -- release beta and fall through to
        # fresh E3 selection. Mirrors the MECH-091 urgency-interrupt template.
        # No-op when the flag is off, when beta is not elevated, or when no
        # snapshot exists. Diagnostic counter incremented each time the
        # release fires (read by V3-EXQ-481b as vs_commit_release_count).
        # Empty-snapshot fix (GAP-5): if the snapshot was empty at commit entry
        # (no active anchors yet), re-populate from the first non-empty set
        # observed while still committed; the trivial issubset of an empty
        # set made the release predicate vacuously false every tick.
        if (
            getattr(self.hippocampal.config, "use_vs_commit_release", False)
            and self.beta_gate.is_elevated
            and self._committed_anchor_keys is not None
            and self.hippocampal.anchor_set is not None
        ):
            current_keys = {
                a.key for a in self.hippocampal.anchor_set.active_anchors()
            }
            if not self._committed_anchor_keys and current_keys:
                self._committed_anchor_keys = current_keys
            elif self._committed_anchor_keys and not self._committed_anchor_keys.issubset(current_keys):
                self.beta_gate.release()
                self._committed_step_idx = 0
                self._committed_anchor_keys = None
                self._vs_commit_release_count = (
                    getattr(self, "_vs_commit_release_count", 0) + 1
                )

        # MECH-302: relief-completion event -- release commitment on sustained
        # suffering drop. Polarity set at input (suffering-derivative-crossing
        # vs goal-attainment). Reuses MECH-057a commitment release + MECH-094
        # categorical tag write (VALENCE_LIKING at current z_world).
        # Architecturally adjacent to the MECH-091 urgency block above;
        # opposite polarity. DO NOT modify the MECH-091 block.
        if self.suffering_comparator is not None and self._relief_completion_event:
            if self.beta_gate.is_elevated:
                self.beta_gate.release()
                self._committed_step_idx = 0
                self._committed_anchor_keys = None
            if (
                self._current_latent is not None
                and self._current_latent.z_world is not None
                and getattr(self.config, "valence_liking_enabled", False)
                and hasattr(self.residue_field, "update_valence")
            ):
                relief_val = float(getattr(self.config, "relief_completion_weight", 1.0))
                self.residue_field.update_valence(
                    self._current_latent.z_world,
                    component=VALENCE_LIKING,
                    value=relief_val,
                    hypothesis_tag=False,
                )
            self._relief_completion_event = False

        # SD-051 / MECH-304: conditioned safety gate (IL->CeA expression pathway).
        # When the EMA prototype recognises a safety cue in current z_world and
        # beta_gate is elevated, release the avoidance commitment.
        # MECH-094: _conditioned_safety_signal is 0.0 during simulation ticks
        # (set by conditioned_safety_store.update() with sim_mode=True in sense()).
        if (
            self.conditioned_safety_store is not None
            and self._conditioned_safety_signal
            > getattr(self.config, "safety_store_threshold", 0.5)
            and self.beta_gate.is_elevated
        ):
            self.beta_gate.release()
            self._committed_step_idx = 0
            self._committed_anchor_keys = None
            if (
                self._current_latent is not None
                and self._current_latent.z_world is not None
                and getattr(self.config, "valence_liking_enabled", False)
                and hasattr(self.residue_field, "update_valence")
            ):
                safety_val = float(
                    getattr(self.config, "safety_store_commitment_weight", 1.0)
                )
                self.residue_field.update_valence(
                    self._current_latent.z_world,
                    component=VALENCE_LIKING,
                    value=safety_val,
                    hypothesis_tag=False,
                )
        self._conditioned_safety_signal = 0.0

        # MECH-303: contextual passive safety gate (background vigilance reduction).
        # When accumulated safety terrain at current z_world exceeds the release
        # threshold and beta_gate is elevated, release the avoidance commitment.
        # Distinct from MECH-304 (event-driven cue) -- this fires from accumulated
        # passive exposure to harmless context, not from a specific safety cue.
        if (
            getattr(self.config, "use_contextual_safety_terrain", False)
            and self.beta_gate.is_elevated
            and self._current_latent is not None
            and self._current_latent.z_world is not None
            and hasattr(self.residue_field, "evaluate_safety")
        ):
            safety_pred = self.residue_field.evaluate_safety(
                self._current_latent.z_world.detach()
            )
            release_thresh = float(
                getattr(self.config, "contextual_safety_release_threshold", 1.0)
            )
            # Read the scalar graph-free: evaluate_safety returns a grad-carrying
            # tensor (the safety-terrain RBF centers/weights require grad while the
            # terrain is being trained), so a bare float(tensor) on this per-tick
            # release path both warns (UserWarning: requires_grad=True -> scalar)
            # and needlessly retains the autograd graph. .detach() is
            # correctness-neutral -- the comparison value is bit-identical.
            if float(safety_pred.mean().detach()) >= release_thresh:
                self.beta_gate.release()
                self._committed_step_idx = 0
                self._committed_anchor_keys = None

        # Natural-commit LATCH-HOLD re-assertion (rung-6 amend, 2026-06-21,
        # failure_autopsy_V3-EXQ-460i). Runs every tick (E3 + non-E3), AFTER all
        # release sites, BEFORE the between-tick branch. While a natural-commit-armed
        # hold is active and the committed trajectory persists, RE-ASSERT the beta
        # latch (keep it elevated against the SD-034 de-commit churn that fragmented
        # the 460i latch to ~1-tick blips) so the natural-commit occupancy sustains
        # by construction -- the sustained reference the rung-6 release shortens and
        # the gate-3 sustained-hold proxy certifies. YIELDS to (disarms on) the three
        # PRINCIPLED releases so it never papers over them: (a) an SD-034 closure
        # de-commit (refractory_remaining > 0 -> the latch is being held DOWN by the
        # closure; don't fight it -- preserves the MECH-446 occupancy-drop DV), (b)
        # the MECH-091 genuine-threat interrupt (safety -- NEVER overridden), (c) the
        # rung-6 NaturalCommitUrgencyRelease's own duration release (the lever
        # shortening the hold -- the whole point). Also disarms when the committed
        # trajectory ends or the optional max-ticks safety cap is reached. No-op when
        # use_natural_commit_latch_hold is False (self._ncl_hold_active stays False).
        if (
            getattr(self.config, "use_natural_commit_latch_hold", False)
            and self._ncl_hold_active
        ):
            _ncl_max = int(
                getattr(self.config, "natural_commit_latch_hold_max_ticks", 0)
            )
            # rung-6 amend (commit-ENTRY primitive): the "committed trajectory persists"
            # persistence check is UNION-aware -- the hold also persists while the
            # F-INDEPENDENT closure-plane commit-entry latch is active. Without this the
            # F-independent occupancy would arm then immediately yield (the F-commit
            # _committed_trajectory is None on the 460j substrate, and the user's design
            # forbids installing a trajectory on the closure plane), so the sustained
            # occupancy the de-commit acts on could never form. Bit-identical OFF:
            # use_closure_commit_entry off -> _closure_committed_active always False ->
            # _ncl_commit_present reduces to (_committed_trajectory is not None) and the
            # yield reduces to the legacy (_committed_trajectory is None). The other
            # principled yields (closure refractory / MECH-091 / rung-6 lever / max-ticks)
            # are unchanged, so the SD-034 de-commit still tears the hold down.
            _ncl_commit_present = (
                self.e3._committed_trajectory is not None
                or self.e3._closure_committed_active
                # C-STEP extension: a closure-FORMED committed trajectory also keeps the
                # hold present so it re-asserts beta instead of yielding (None unless the
                # trajectory flag is on -> bit-identical to the bool-latch union).
                or self.e3._closure_committed_trajectory is not None
            )
            _ncl_yield = (
                not _ncl_commit_present
                or self.beta_gate.refractory_remaining > 0
                or self._ncl_mech091_fired
                or self._ncl_lever_fired
                or (_ncl_max > 0 and self._ncl_hold_ticks >= _ncl_max)
            )
            # ARC-108 JOB-2 (c): rho_t MAINTENANCE RAMP -- REPLACE the flat
            # (unconditional) re-assert DRIVER with a proximity-scaled one. The ramp
            # gives the hold an intrinsic decay term it lacked: rho_t = goal-proximity
            # x value peaks-then-declines, so when it has fallen past its proximity
            # peak the hold SELF-LIMITS (the structural B6 / 460h-monolith fix). This
            # ADDS a yield condition; all the existing safety-bearing yields above are
            # kept. No-op when the ramp is disabled -> the flat re-assert is unchanged.
            if (
                not _ncl_yield
                and self.rho_maintenance_ramp is not None
                and self.rho_maintenance_ramp.is_active
            ):
                if self.rho_maintenance_ramp.tick(self._compute_rho_t()):
                    _ncl_yield = True
            if _ncl_yield:
                self._ncl_hold_active = False
            else:
                self._ncl_hold_ticks += 1
                if not self.beta_gate.is_elevated:
                    self.beta_gate.elevate()
                    self._ncl_hold_reassert_count += 1

        if not ticks["e3_tick"] and self._last_action is not None:
            # Between E3 ticks: step through committed trajectory (Layer 1) or hold.
            # C-STEP extension: prefer the F-driven _committed_trajectory; fall back to
            # the closure-FORMED _closure_committed_trajectory (None unless the trajectory
            # flag is on) so a closure-armed hold advances an actual committed PROGRAM
            # instead of repeating _last_action. Bit-identical when the closure latch is
            # unused (the `or` reduces to _committed_trajectory).
            _step_traj = (
                self.e3._committed_trajectory
                or self.e3._closure_committed_trajectory
            )
            if self.beta_gate.is_elevated and _step_traj is not None:
                traj = _step_traj
                horizon = traj.actions.shape[1]
                step_idx = min(self._committed_step_idx, horizon - 1)
                action = traj.actions[:, step_idx, :]
                self._committed_step_idx += 1
            else:
                action = self._last_action
            # MECH-165: record held/stepped action for exploration trajectory (every step)
            self._record_exploration_action(action)
            self._last_action = action
            return action

        # SD-016 (MECH-152): pass cached terrain_weight so harm/goal scoring
        # precision reflects current z_world cue context.
        # MECH-108: pass BreathOscillator sweep reduction during sweep phase.
        # Creates periodic uncommitted windows even when running_variance has
        # converged below base commit_threshold after training.
        sweep_reduction = self.clock.sweep_amplitude if self.clock.sweep_active else 0.0
        # MECH-188: PFC top-down z_goal injection (EXQ-253).
        # When z_goal_inject > 0, apply a constant norm floor to z_goal for
        # action selection ONLY. Does not modify the persistent attractor.
        # Simulates DRN-mPFC top-down goal persistence bypassing terrain seeding.
        _goal_inject = getattr(
            getattr(self.config, "goal", None), "z_goal_inject", 0.0
        )
        _goal_state_for_select = self.goal_state
        if _goal_inject > 0.0 and self.goal_state is not None:
            _goal_state_for_select = self.goal_state.with_injection(_goal_inject)

        # SD-032e: tick pACC-analog before dACC / salience consumers so that
        # effective_drive_level (base + drive_bias) is available for all SD-032
        # downstream modules. Gate on coordinator's previous-tick "autonomic"
        # write_gate (one-step lag; the pACC->autonomic->sensitisation circuit
        # is slow enough that instantaneous coupling is not required). When
        # salience is disabled the gate defaults to 1.0 so the drift remains
        # observable under ablation. MECH-094: hypothesis_tag=False here
        # (waking action selection); simulation/replay paths do not route
        # through select_action().
        if self.pacc is not None:
            _zha_norm = 0.0
            if z_harm_a is not None:
                _zha_norm = float(z_harm_a.norm().item())
            _auto_gate = 1.0
            if self.salience is not None:
                _auto_gate = float(self.salience.write_gate("autonomic"))
            self._pacc_last_tick = self.pacc.tick(
                z_harm_a_norm=_zha_norm,
                write_gate=_auto_gate,
                hypothesis_tag=False,
                # SD-049 Phase 3: per-axis cache for diagnostics + the
                # effective_drive_from_per_axis() helper. The EMA write-back
                # is whole-organism; per-axis pACC is its own SD claim.
                per_axis_drive=self._per_axis_drive_for_consumers(),
                per_axis_combiner=getattr(
                    self.config, "sd049_pacc_per_axis_combiner", "sum"
                ),
            )

        # SD-032e: resolve the effective drive_level once, used by dACC,
        # salience coordinator, AIC, PCC. Falls back to base when pACC is off.
        _base_drive_level = float(
            getattr(self.goal_state, "_last_drive_level", 0.0)
        ) if self.goal_state is not None else 0.0
        if self.pacc is not None:
            _effective_drive_level = self.pacc.effective_drive(_base_drive_level)
        else:
            _effective_drive_level = _base_drive_level

        # SD-032b: compute dACC bundle + stopgap-adapter score bias before E3.select.
        # The bundle reads the (precision-weighted) affective-pain PE from the last
        # forward-model prediction, and mixes in per-candidate payoff/effort terms.
        dacc_score_bias: Optional[torch.Tensor] = None
        # MECH-451: when use_finer_channel_gating is on, capture each finer
        # constituent's UN-summed per-candidate [K] bias into this dict (keyed by the
        # FINER_NAMED_CHANNELS names) so e3.select() can register one learned channel
        # per head instead of the single compressed "score_bias" blend. None ->
        # bit-identical (legacy single-channel path). The captured tensors are the
        # SAME ones already summed into dacc_score_bias; anything NOT captured here
        # (curiosity / blocked-agency / avoidance / escape) lands in the select()
        # "residual" channel by subtraction, so the decomposition stays exhaustive.
        _fcg_channels: Optional[Dict[str, torch.Tensor]] = (
            {} if getattr(self.config, "use_finer_channel_gating", False) else None
        )
        # V3-EXQ-571: per-component bias trackers (zero-cost when flag OFF)
        _bdc_dacc: Optional[torch.Tensor] = None
        _bdc_lpfc: Optional[torch.Tensor] = None
        _bdc_ofc: Optional[torch.Tensor] = None
        _bdc_gp: Optional[torch.Tensor] = None
        _bdc_m295: Optional[torch.Tensor] = None
        _bdc_curiosity: Optional[torch.Tensor] = None
        _bdc_vigor: Optional[torch.Tensor] = None
        _bdc_forced: Optional[torch.Tensor] = None
        # route-range AMEND (569f/661/654a): MECH-294 compose per-candidate bias
        # tracker, stashed so the "coherence" route source is available at the
        # e3.select() site (parallel to the other _bdc_* per-candidate biases).
        _bdc_coherence: Optional[torch.Tensor] = None
        # ControlVector logging (rec-B): reset the cached MECH-320 split each
        # tick so a tonic-vigor-off / no-fire tick does not carry stale state.
        self._cv_vigor = None
        if self.dacc is not None and z_harm_a is not None:
            # Per-candidate payoff proxy: negative of E3 running candidate score if
            # available (lower score = better, so payoff = -score). Falls back to
            # a zero payoff vector until E3 has run at least once.
            K = len(candidates)
            if self.e3.last_scores is not None and self.e3.last_scores.numel() == K:
                payoffs = -self.e3.last_scores.detach().float()
            else:
                payoffs = torch.zeros(K, device=self.device)
            # Per-candidate effort proxy: trajectory length / horizon. Future
            # refinement (Croxson): harm-forward rollout cost.
            effort = torch.tensor(
                [float(c.actions.shape[1]) for c in candidates],
                dtype=payoffs.dtype,
                device=self.device,
            )
            # Action-class tags for MECH-260 suppression: argmax of first action.
            action_classes = [
                int(c.actions[0, 0].argmax().item()) for c in candidates
            ]
            # GoalState does not persist drive_level (it is passed per-update);
            # SD-032e: use effective drive (base + pACC bias) so dACC sees the
            # chronic-pain-sensitised drive regime.
            drive_level = _effective_drive_level
            # SD-057 L7 (MECH-348): per-candidate goal_proximity to the (now
            # object-bound, via SD-057 L4) z_goal, so the dACC readout becomes
            # object-discriminative. Gated by use_mech_consume; None otherwise
            # (adapter skips the goal term -> bit-identical). Mirrors the
            # MECH-295 first-step z_world summary build.
            _dacc_goal_prox: Optional[torch.Tensor] = None
            if (
                getattr(self.config, "use_mech_consume", False)
                and self.goal_state is not None
                and self.goal_state.is_active()
                and self._current_latent is not None
            ):
                _gp_list: List[torch.Tensor] = []
                for c in candidates:
                    if c.world_states is not None:
                        _gp_list.append(c.get_world_state_sequence()[0, 0, :])
                    else:
                        _gp_list.append(self._current_latent.z_world[0].detach())
                with torch.no_grad():
                    _dacc_goal_prox = self.goal_state.goal_proximity(
                        torch.stack(_gp_list, dim=0)
                    ).detach()
            bundle = self.dacc(
                z_harm_a=z_harm_a.squeeze(0) if z_harm_a.dim() > 1 else z_harm_a,
                z_harm_a_pred=self._harm_a_pred_prev,
                candidate_payoffs=payoffs,
                candidate_effort=effort,
                candidate_action_classes=action_classes,
                precision=float(self.e3.current_precision),
                drive_level=drive_level,
                # SD-049 Phase 3: control demand tracks worst-axis deficit.
                per_axis_drive=self._per_axis_drive_for_consumers(),
                per_axis_combiner=getattr(
                    self.config, "sd049_dacc_per_axis_combiner", "max"
                ),
                candidate_goal_proximity=_dacc_goal_prox,  # SD-057 L7 (MECH-348)
            )
            self._dacc_last_bundle = bundle
            if self.dacc_adapter is not None:
                dacc_score_bias = self.dacc_adapter(bundle)
                self._dacc_last_bias = dacc_score_bias.detach().clone()
                if self.e3.e3_score_decomp_enabled:
                    _bdc_dacc = dacc_score_bias.detach().clone()

        # SD-032a: tick the salience-network coordinator. Aggregates the dACC
        # bundle (live), drive_level (live), and offline-mode flag (proxy for
        # SD-032d stability) into the operating_mode soft vector and the
        # MECH-259 mode_switch_trigger. SD-032c/d/e signal slots remain at zero
        # until those subdivisions land.
        if self.salience is not None:
            # SD-032e: use effective drive (base + pACC bias) so salience,
            # AIC, and PCC all see the sensitised drive regime coherently.
            sal_drive = _effective_drive_level
            sal_offline = bool(getattr(self.e1, "_offline_mode", False))
            sal_bundle = self._dacc_last_bundle  # may be None if dACC is off
            # SD-032c: inject AIC salience into the coordinator BEFORE tick so
            # the MECH-259 urgency-trigger source is live on this cycle.
            if self.aic is not None:
                self.salience.update_signal(
                    "aic_salience", float(self.aic.aic_salience)
                )
            # SD-032d: tick PCC-analog and inject pcc_stability into the
            # coordinator BEFORE tick so MECH-259 effective_threshold is
            # modulated on this cycle. High stability -> harder to switch;
            # low stability (depleted / no recent rest / failing) -> easier.
            if self.pcc is not None:
                self._pcc_last_tick = self.pcc.tick(
                    drive_level=sal_drive,
                    # SD-049 Phase 3: PCC fatigue integrates across axes
                    # (mean combiner default; whole-organism fatigue scalar).
                    per_axis_drive=self._per_axis_drive_for_consumers(),
                    per_axis_combiner=getattr(
                        self.config, "sd049_pcc_per_axis_combiner", "mean"
                    ),
                )
                self.salience.update_signal(
                    "pcc_stability", float(self.pcc.pcc_stability)
                )
            # SD-035 / MECH-046 / MECH-074c: inject CeA mode_prior and
            # fast_prime into the coordinator BEFORE tick so the fast
            # subcortical route biases this cycle's mode affinity and
            # salience aggregate. mode_prior is a pre-softmax additive
            # log-odds bias (registered on affinity_weights["cea_mode_prior"]
            # = {"external_task": 1.0} at __init__); fast_prime is a scalar
            # salience pulse (registered on salience_weights["cea_fast_prime"]
            # = 0.5 at __init__). Both zero at rest -> no-op when CeA is off
            # or below threshold.
            if self.cea is not None and self._cea_last_output is not None:
                self.salience.update_signal(
                    "cea_mode_prior",
                    float(self._cea_last_output.mode_prior),
                )
                self.salience.update_signal(
                    "cea_fast_prime",
                    float(self._cea_last_output.fast_prime),
                )
            # SD-037: inject broadcast override_signal as an affinity contribution
            # toward external_task. Biases mode SELECTION (not switch threshold);
            # zero at rest -> no-op when override is below recruitment threshold.
            if self.broadcast_override is not None:
                self.salience.update_signal(
                    "override_signal",
                    float(self.broadcast_override.override_signal),
                )
            # mode-governance-engagement substrate (2026-06-13): inject the
            # external_task_drive engagement scalar BEFORE tick so external_task can
            # win the mode argmax (affinity) AND clear the MECH-259 switch aggregate
            # (salience) during committed pursuit of an active goal. engagement =
            # goal_active ? clip(commit_w*float(beta_elevated)
            # + prox_w*goal_proximity(z_world), 0, 1) : 0. DYNAMIC -> releases toward
            # internal_planning when the goal is inactive / uncommitted, preserving
            # genuine mode competition (not the 464b 100%-external_task saturation).
            # MECH-094: waking-only by call-site scoping (select_action), as with the
            # AIC / CeA / override injections above. No-op default (flag off).
            if getattr(self.config, "use_external_task_drive", False):
                _et_require_goal = getattr(
                    self.config, "external_task_drive_require_goal_active", True
                )
                _et_goal_active = (not _et_require_goal) or (
                    self.goal_state is not None and self.goal_state.is_active()
                )
                _et_engagement = 0.0
                if _et_goal_active:
                    _et_commit_w = float(
                        getattr(self.config, "external_task_drive_commit_weight", 1.0)
                    )
                    _et_prox_w = float(
                        getattr(self.config, "external_task_drive_proximity_weight", 1.0)
                    )
                    _et_commit = _et_commit_w * (1.0 if self.beta_gate.is_elevated else 0.0)
                    _et_prox = 0.0
                    if (
                        _et_prox_w != 0.0
                        and self.goal_state is not None
                        and self._current_latent is not None
                    ):
                        _et_zp = self.goal_state.goal_proximity(
                            self._current_latent.z_world
                        )
                        _et_prox = _et_prox_w * float(_et_zp.reshape(-1)[0].item())
                    _et_engagement = max(0.0, min(1.0, _et_commit + _et_prox))
                self.salience.update_signal("external_task_drive", _et_engagement)
            self._salience_last_tick = self.salience.tick(
                dacc_bundle=sal_bundle,
                drive_level=sal_drive,
                is_offline=sal_offline,
                # SD-049 Phase 3: SalienceCoordinator's drive_level signal
                # reads worst-axis when per-axis cascade is on.
                per_axis_drive=self._per_axis_drive_for_consumers(),
                per_axis_combiner=getattr(
                    self.config, "sd049_salience_per_axis_combiner", "max"
                ),
            )
            # Optional: scale dACC score_bias by the e3_policy write-gate, so
            # that during internal_replay the dACC bias is suppressed near zero
            # (per MECH-261 table). Default-off (backward compatible).
            if (
                self.config.salience_apply_to_dacc_bias
                and dacc_score_bias is not None
            ):
                e3_gate = self.salience.write_gate("e3_policy")
                dacc_score_bias = dacc_score_bias * float(e3_gate)
                self._dacc_last_bias = dacc_score_bias.detach().clone()

        # MECH-451: capture the dACC-conflict finer channel = the dACC contribution
        # to score_bias (post-e3_gate, before any other head adds). Captured here so
        # the gp / lpfc / ofc / mech295 / vigour contributions added below are NOT
        # folded into it.
        if _fcg_channels is not None and dacc_score_bias is not None:
            _fcg_channels["dacc"] = dacc_score_bias.detach().clone()

        # ARC-062 Phase 1/3: gated-policy heads + context discriminator.
        # Block is ordered BEFORE lateral_pfc (ARC-062 GAP-C reorder 2026-05-17)
        # so gp_output.gating_weight is available to pass as disc_output into
        # lateral_pfc.update(). Score-bias composition is additive and
        # order-independent; net score_bias is bit-identical to pre-reorder.
        # Phase 1 weak-reading: Per-candidate score_bias = w * head_0(features)
        # + (1 - w) * head_1(features), where w is a sigmoid over a 3-stream
        # (z_world, z_self, z_harm_a) discriminator (Pull A R1 verdict). N=2
        # heads (Pull A R2 substrate-constrained). Composed additively into the
        # dACC / lateral_pfc / ofc score_bias chain (Pull A R3 verdict).
        # MECH-094: simulation_mode=False (waking action selection).
        _gp_output = None  # sentinel used by lateral_pfc GAP-C block below
        if (
            self.gated_policy is not None
            and self._current_latent is not None
        ):
            # Build per-candidate z_world summaries (first-step z_world of
            # each trajectory). gated_policy runs first so cand_world_summaries
            # is not yet set; build fresh and expose as cand_world_summaries
            # so lateral_pfc / ofc / mech295 blocks below can reuse it.
            # ARC-065 GAP-A: prefer the SD-056-divergent e2.world_forward source
            # when candidate_summary_source="e2_world_forward"; None -> legacy
            # collapsed proposer first-step z_world (bit-identical).
            K = len(candidates)
            gp_summaries = self._candidate_world_summaries(candidates)
            if gp_summaries is None:
                _gp_list: List[torch.Tensor] = []
                for c in candidates:
                    if c.world_states is not None:
                        ws = c.get_world_state_sequence()
                        _gp_list.append(ws[0, 0, :])
                    else:
                        _gp_list.append(
                            self._current_latent.z_world[0].detach()
                        )
                gp_summaries = torch.stack(_gp_list, dim=0)
            cand_world_summaries = gp_summaries  # expose for downstream blocks
            # MECH-319: route the simulation_mode argument through the
            # rule-write gate. select_action runs on the waking path so
            # caller_sim=False; gate returns False (admit) when MECH-319
            # is OFF or when master-ON-with-waking-caller (bit-identical).
            # The seam is exposed for V3-EXQ-543c artificial-write-channel-
            # routing where admit_writes=True flips the falsifier control
            # for replay-driven invocations.
            if self.simulation_mode_rule_gate is not None:
                _gp_sim = self.simulation_mode_rule_gate.effective_simulation_mode(
                    simulation_mode=False, site=SITE_GATED_POLICY
                )
            else:
                _gp_sim = False
            # ARC-062 GAP-B option-2: build first-action one-hots [K, action_dim]
            # when use_first_action_onehot is enabled. Each candidate always has
            # actions shape [batch, horizon, action_dim]; we take batch-dim 0,
            # step 0. simulation_mode early-return in forward() already bypasses
            # this data before it reaches the heads (MECH-094 safe).
            if self.gated_policy.config.use_first_action_onehot:
                _fa_list: List[torch.Tensor] = []
                for _gpc in candidates:
                    _fa_list.append(
                        _gpc.actions[:, 0, :][0].detach().float()
                    )
                _gp_first_action_onehots = torch.stack(_fa_list, dim=0)
            else:
                _gp_first_action_onehots = None
            with torch.no_grad():
                _gp_output = self.gated_policy(
                    z_world=self._current_latent.z_world,
                    z_self=self._current_latent.z_self,
                    z_harm_a=self._current_latent.z_harm_a,
                    candidate_features=gp_summaries,
                    first_action_onehots=_gp_first_action_onehots,
                    simulation_mode=_gp_sim,
                )
            gp_bias = _gp_output.gated_score_bias
            if self.e3.e3_score_decomp_enabled:
                _bdc_gp = gp_bias.detach().clone()
            if _fcg_channels is not None:  # MECH-451: gated_policy finer channel
                _fcg_channels["gated_policy"] = gp_bias.detach().clone()
            if dacc_score_bias is None:
                dacc_score_bias = gp_bias
            else:
                dacc_score_bias = dacc_score_bias + gp_bias.to(
                    dtype=dacc_score_bias.dtype, device=dacc_score_bias.device
                )

        # SD-033a: Lateral-PFC-analog tick. Primary consumer of MECH-261
        # write_gate("sd_033a"). Gate-modulated EMA update of rule_state,
        # then per-candidate score_bias composed additively with dACC bias.
        # ARC-062 GAP-C (2026-05-17): when use_discriminator_source=True,
        # passes gp_output.gating_weight as disc_output into update() so the
        # discriminator's context-classification signal reaches rule_state.
        # ARC-062 GAP-D (2026-05-17): rule_bias_head trainable when
        # lateral_pfc_train_rule_bias_head=True (last Linear not zeroed).
        # MECH-094: rule persistence is gated by the registry -- no separate
        # hypothesis_tag check (the gate IS the tag via MECH-261).
        if (
            self.lateral_pfc is not None
            and self._current_latent is not None
        ):
            # MECH-319: consult the simulation-mode rule-write gate before
            # the lateral_pfc.update call. select_action runs on the waking
            # path so caller_sim=False; for MECH-319-OFF or master-ON-with-
            # waking-caller the gate returns False (admit) and update
            # proceeds normally (bit-identical). The seam is exposed for
            # V3-EXQ-543c artificial-write-channel-routing tests where a
            # replay-driven invocation passes simulation_mode=True; with
            # admit_writes=False the gate returns True (skip update);
            # with admit_writes=True (falsifier) the gate returns False
            # (admit despite tag).
            if self.simulation_mode_rule_gate is not None:
                _lpfc_skip = self.simulation_mode_rule_gate.effective_simulation_mode(
                    simulation_mode=False, site=SITE_LATERAL_PFC
                )
            else:
                _lpfc_skip = False
            if not _lpfc_skip:
                if self.salience is not None:
                    lpfc_gate = float(self.salience.write_gate("sd_033a"))
                else:
                    # Coordinator disabled -> full gate (lateral_pfc active under
                    # use_lateral_pfc_analog alone, so ablation is possible without
                    # requiring SD-032a to be on).
                    lpfc_gate = 1.0
                # ARC-062 GAP-C: build disc_output from gp_output.gating_weight
                # when use_discriminator_source is True and gated_policy ran this
                # tick. gp_output is under no_grad; disc_output is detached inside
                # lateral_pfc.update(). Default None = no-op (backward compat).
                _lpfc_disc: Optional[torch.Tensor] = None
                if (
                    self.lateral_pfc.config.use_discriminator_source
                    and _gp_output is not None
                ):
                    _lpfc_disc = torch.tensor(
                        [[_gp_output.gating_weight]],
                        dtype=self._current_latent.z_world.dtype,
                        device=self._current_latent.z_world.device,
                    )
                # SD-037 MECH-281 motor-coupling axis: orexin-recruited
                # state accelerates rule_state EMA by scaling eff_eta by
                # (1 + override_pfc_eta_gain * override_signal). Gain=0.0
                # default -> bit-identical to pre-MECH-281.
                _ov_sig_lpfc = (
                    float(self.broadcast_override.override_signal)
                    if self.broadcast_override is not None
                    else 0.0
                )
                _ov_lpfc_gain = float(
                    getattr(self.config, "override_pfc_eta_gain", 0.0)
                )
                # ARC-063 GAP-B: tick the CandidateRuleField and source the
                # rule_state from its differentiated active-rule stack. The
                # field credits the PREVIOUS tick's active rules with this
                # tick's outcome proxy (lower harm = success), mints on a
                # recurring (context-bucket, prev-action) regularity, gates +
                # selects against the current z_world context, and returns the
                # [1, rule_dim] differentiated vector. crf_source replaces the
                # legacy EMA source inside update() (use_candidate_rule_source
                # set True at __init__ when the field is on). _lpfc_skip already
                # gates the MECH-319 simulation path, so this runs only on the
                # waking write path (simulation_mode=False).
                _crf_source: Optional[torch.Tensor] = None
                if self.candidate_rule_field is not None:
                    _crf_ctx = self._current_latent.z_world.detach().reshape(-1)
                    # ARC-063 amend (V3-EXQ-654b): optionally source the CRF
                    # mint/match context from e2.world_forward(z_world, prev_action)
                    # instead of raw z_world, so the mint-block does not collapse
                    # under low raw-z_world spread (mirrors ARC-065 GAP-A). no_grad
                    # waking read; falls back to raw z_world when prev-action is
                    # unset or e2 is unavailable. Default OFF -> raw z_world.
                    if getattr(
                        self.config, "crf_context_from_e2_world_forward", False
                    ):
                        _crf_e2 = getattr(self, "e2", None)
                        _crf_pa = self._crf_prev_action_class
                        if (
                            _crf_e2 is not None
                            and _crf_pa >= 0
                            and len(candidates) > 0
                        ):
                            try:
                                _crf_adim = int(candidates[0].actions.shape[-1])
                            except (AttributeError, IndexError, TypeError):
                                _crf_adim = 0
                            if 0 <= _crf_pa < _crf_adim:
                                _crf_z0 = (
                                    self._current_latent.z_world.detach()
                                )
                                if _crf_z0.dim() == 1:
                                    _crf_z0 = _crf_z0.unsqueeze(0)
                                _crf_z0 = _crf_z0[:1]  # [1, world_dim]
                                _crf_act = torch.zeros(
                                    1,
                                    _crf_adim,
                                    device=_crf_z0.device,
                                    dtype=_crf_z0.dtype,
                                )
                                _crf_act[0, _crf_pa] = 1.0
                                with torch.no_grad():
                                    _crf_pred = _crf_e2.world_forward(
                                        _crf_z0, _crf_act
                                    )
                                _crf_ctx = _crf_pred.detach().reshape(-1)
                    _crf_outcome = 0.0
                    _z_ha = getattr(self._current_latent, "z_harm_a", None)
                    if _z_ha is not None:
                        _crf_outcome = -float(_z_ha.norm().item())
                    elif self._current_latent.z_harm is not None:
                        _crf_outcome = -float(self._current_latent.z_harm.norm().item())
                    _crf_arc062 = (
                        float(_gp_output.gating_weight)
                        if (
                            self.candidate_rule_field.config.seed_from_arc062
                            and _gp_output is not None
                        )
                        else None
                    )
                    _crf_source = self.candidate_rule_field.step(
                        context=_crf_ctx,
                        action_object_idx=self._crf_prev_action_class,
                        outcome_signal=_crf_outcome,
                        arc062_seed=_crf_arc062,
                        simulation_mode=False,
                    )
                # Update rule_state (in-place on buffer, no gradient flow).
                self.lateral_pfc.update(
                    z_delta=self._current_latent.z_delta,
                    z_world=self._current_latent.z_world,
                    gate=lpfc_gate,
                    disc_output=_lpfc_disc,
                    override_signal=_ov_sig_lpfc,
                    override_eta_gain=_ov_lpfc_gain,
                    crf_source=_crf_source,
                )
            # Per-candidate z_world summary: reuse from gated_policy block if
            # it ran this tick (cand_world_summaries set above), otherwise build
            # fresh from candidates (backward-compatible path when gated_policy
            # is disabled).
            try:
                cand_world_summaries  # type: ignore[name-defined]
            except NameError:
                # ARC-065 GAP-A: prefer the e2.world_forward source; None ->
                # legacy collapsed proposer path (bit-identical).
                cand_world_summaries = self._candidate_world_summaries(candidates)
                if cand_world_summaries is None:
                    K = len(candidates)
                    cand_world_list: List[torch.Tensor] = []
                    for c in candidates:
                        if c.world_states is not None:
                            ws = c.get_world_state_sequence()  # [batch, horizon+1, world_dim]
                            cand_world_list.append(ws[0, 0, :])  # first-step, first batch
                        else:
                            cand_world_list.append(
                                self._current_latent.z_world[0].detach()
                            )
                    cand_world_summaries = torch.stack(cand_world_list, dim=0)  # [K, world_dim]
            lpfc_bias = self.lateral_pfc.compute_bias(cand_world_summaries)
            if _fcg_channels is not None:  # MECH-451: lateral-PFC rule-evidence channel
                _fcg_channels["lpfc"] = lpfc_bias.detach().clone()
            if self.e3.e3_score_decomp_enabled:
                _bdc_lpfc = lpfc_bias.detach().clone()
            # Compose additively with dACC score_bias (lower-is-better in E3).
            if dacc_score_bias is None:
                dacc_score_bias = lpfc_bias
            else:
                dacc_score_bias = dacc_score_bias + lpfc_bias.to(
                    dtype=dacc_score_bias.dtype, device=dacc_score_bias.device
                )

        # SD-033b: OFC-analog tick. Second consumer of MECH-261
        # write_gate("sd_033b"). Gate-modulated EMA update of state_code
        # using z_world plus optional pooled z_harm (outcome-context),
        # then per-candidate score_bias composed additively with dACC +
        # lateral_pfc bias. Initial bias output is exactly zero (last
        # Linear zeroed at init) so use_ofc_analog=True with an untrained
        # head is bit-identical to OFF until the head is deliberately
        # trained.
        if (
            self.ofc is not None
            and self._current_latent is not None
        ):
            if self.salience is not None:
                ofc_gate = float(self.salience.write_gate("sd_033b"))
            else:
                ofc_gate = 1.0
            ofc_z_harm = (
                self._current_latent.z_harm
                if self.ofc.config.harm_dim > 0
                else None
            )
            self.ofc.update(
                z_world=self._current_latent.z_world,
                z_harm=ofc_z_harm,
                gate=ofc_gate,
            )
            # Reuse the per-candidate z_world summaries built above when
            # lateral_pfc is on; otherwise build them here.
            if self.lateral_pfc is not None:
                ofc_summaries = cand_world_summaries
            else:
                # ARC-065 GAP-A: prefer the e2.world_forward source; None ->
                # legacy collapsed proposer path (bit-identical).
                ofc_summaries = self._candidate_world_summaries(candidates)
                if ofc_summaries is None:
                    K = len(candidates)
                    _ofc_list: List[torch.Tensor] = []
                    for c in candidates:
                        if c.world_states is not None:
                            ws = c.get_world_state_sequence()
                            _ofc_list.append(ws[0, 0, :])
                        else:
                            _ofc_list.append(
                                self._current_latent.z_world[0].detach()
                            )
                    ofc_summaries = torch.stack(_ofc_list, dim=0)
            ofc_bias = self.ofc.compute_bias(ofc_summaries)
            if _fcg_channels is not None:  # MECH-451: OFC-devaluation channel
                _fcg_channels["ofc"] = ofc_bias.detach().clone()
            if self.e3.e3_score_decomp_enabled:
                _bdc_ofc = ofc_bias.detach().clone()
            if dacc_score_bias is None:
                dacc_score_bias = ofc_bias
            else:
                dacc_score_bias = dacc_score_bias + ofc_bias.to(
                    dtype=dacc_score_bias.dtype, device=dacc_score_bias.device
                )

            # MECH-263 specific-outcome oracle: per-candidate z_harm_s
            # prediction via E2HarmSForward. Queries OFCAnalog.query_outcome()
            # for each candidate's first action and caches predictions for
            # diagnostic access and future score modulation. No effect on
            # E3 scores until a non-zero oracle_score_weight is added in a
            # later training pass.
            if (
                self.ofc.oracle_is_ready
                and self.e2_harm_s is not None
                and self._current_latent is not None
                and self._current_latent.z_harm is not None
            ):
                z_harm_s_curr = self._current_latent.z_harm.detach()
                _oracle_list: List[torch.Tensor] = []
                for _c in candidates:
                    _a = _c.actions[:, 0, :]
                    _oracle_list.append(
                        self.ofc.query_outcome(z_harm_s_curr, _a, self.e2_harm_s)
                    )
                self._ofc_oracle_predictions = _oracle_list
            else:
                self._ofc_oracle_predictions = None

        # MECH-295: drive -> liking-stream -> approach_cue at action selection.
        # Per-candidate liking signal: drive * goal_proximity (each
        # candidate's first-step z_world). Negated so it favours approach
        # under E3's lower-is-better convention. Composed additively with
        # the existing dACC / lateral_pfc / ofc score_bias.
        # Weak-reading: with mech295_liking_to_approach_cue_gain=0.0 the
        # bias is exactly zero (severed bridge arm).
        if (
            self.mech295_bridge is not None
            and self.goal_state is not None
            and self.goal_state.is_active()
            and self._current_latent is not None
        ):
            # Build per-candidate first-step z_world summaries. Reuse the
            # cand_world_summaries from the lateral_pfc / ofc blocks if
            # available; otherwise build them here from the candidates list.
            try:
                m295_summaries = cand_world_summaries  # type: ignore[name-defined]
            except NameError:
                # ARC-065 GAP-A: prefer the e2.world_forward source; None ->
                # legacy collapsed proposer path (bit-identical).
                m295_summaries = self._candidate_world_summaries(candidates)
                if m295_summaries is None:
                    K = len(candidates)
                    _m295_list: List[torch.Tensor] = []
                    for c in candidates:
                        if c.world_states is not None:
                            ws = c.get_world_state_sequence()
                            _m295_list.append(ws[0, 0, :])
                        else:
                            _m295_list.append(
                                self._current_latent.z_world[0].detach()
                            )
                    m295_summaries = torch.stack(_m295_list, dim=0)
            # Per-candidate goal proximity in [0, 1]: GoalState.goal_proximity
            # returns 1 / (1 + dist) -- shape [K] when input is [K, world_dim].
            with torch.no_grad():
                cand_proximities = self.goal_state.goal_proximity(
                    m295_summaries
                ).detach()
                base_drive = float(
                    getattr(self.goal_state, "_last_drive_level", 0.0)
                )
                # Apply pACC sensitisation if present so the cue side sees
                # the same effective drive that the write side used.
                if self.pacc is not None:
                    eff_drive_m295 = self.pacc.effective_drive(base_drive)
                else:
                    eff_drive_m295 = base_drive
                # SD-049 Phase 3: pass the per-axis drive vector through.
                # When goal_axis_idx is supplied at the agent level (not
                # wired in this Phase -- experiments set it via a separate
                # API in follow-on work), MECH-295 routes via the matched
                # axis. Default fallback is the combiner (max).
                _m295_pad = self._per_axis_drive_for_consumers()
                _m295_combiner = getattr(
                    self.config, "sd049_mech295_per_axis_combiner", "max"
                )
                m295_bias = self.mech295_bridge.compute_approach_cue_score_bias(
                    drive_level=eff_drive_m295,
                    candidate_proximities=cand_proximities,
                    simulation_mode=False,
                    per_axis_drive=_m295_pad,
                    goal_axis_idx=getattr(self, "_current_goal_axis_idx", None),
                    per_axis_combiner=_m295_combiner,
                )
                # MECH-307 Path B: consumer-side conjunction read. Adds an
                # additional approach bias when the four-way conjunction
                # (wanting + liking + signed-positive surprise + z_beta
                # arousal) holds at the candidate's predicted-imminent
                # location. Bit-identical zero when the bridge config flag
                # use_mech307_conjunction_read is False (the bridge's own
                # internal short-circuit handles that). Reads the
                # PRE-elevated z_beta from self._current_latent (Gap 3
                # writes update z_beta in update_schema_wanting before this
                # block).
                if (
                    self.mech295_bridge.config.use_mech307_conjunction_read
                    and self._current_latent is not None
                    and self._current_latent.z_beta is not None
                    and self._current_latent.z_beta.numel() > 0
                ):
                    z_beta_arousal = float(
                        self._current_latent.z_beta[..., 0].abs().mean().item()
                    )
                    m307_bias = (
                        self.mech295_bridge.compute_conjunction_score_bias(
                            candidate_z_locs=m295_summaries,
                            residue_field=self.residue_field,
                            z_beta_arousal=z_beta_arousal,
                            drive_level=eff_drive_m295,
                            simulation_mode=False,
                        )
                    )
                    m295_bias = m295_bias + m307_bias.to(
                        dtype=m295_bias.dtype, device=m295_bias.device
                    )

            if self.e3.e3_score_decomp_enabled:
                _bdc_m295 = m295_bias.detach().clone()
            if _fcg_channels is not None:  # MECH-451: liking (MECH-295) channel
                _fcg_channels["liking"] = m295_bias.detach().clone()
            if dacc_score_bias is None:
                dacc_score_bias = m295_bias
            else:
                dacc_score_bias = dacc_score_bias + m295_bias.to(
                    dtype=dacc_score_bias.dtype, device=dacc_score_bias.device
                )

        # MECH-314 (ARC-065): structured_curiosity_bonus. Per-candidate
        # (314a) and broadcast-scalar (314b/c Phase 1) negative score-bias
        # composed additively into dacc_score_bias. Fires only when
        # self.curiosity is not None (master flag ON). Bit-identical
        # baseline when curiosity is None. Sub-flavour switches inside
        # the module gate per-flavour computation; flag-set Q-044
        # ablation is "master ON + per-sub flag flips."
        if self.curiosity is not None:
            # MECH-314a Phase-2 amend (V3-EXQ-648a): optionally source the
            # per-candidate novelty signature from the SD-056-trained
            # action-conditional e2.world_forward(z0, a_i) predictions.
            # Returns None for the default "proposer" source, in which case
            # the legacy reuse-chain below runs unchanged (bit-identical).
            cur_summaries = self._curiosity_candidate_summaries(candidates)
            if cur_summaries is None:
                # Build per-candidate first-step z_world summaries. Reuse
                # m295_summaries when the MECH-295 block ran; else
                # cand_world_summaries from lateral_pfc / ofc; else build
                # fresh from candidates.
                try:
                    cur_summaries = m295_summaries  # type: ignore[name-defined]
                except NameError:
                    try:
                        cur_summaries = cand_world_summaries  # type: ignore[name-defined]
                    except NameError:
                        K = len(candidates)
                        _cur_list: List[torch.Tensor] = []
                        for c in candidates:
                            if c.world_states is not None:
                                ws = c.get_world_state_sequence()
                                _cur_list.append(ws[0, 0, :])
                            elif self._current_latent is not None:
                                _cur_list.append(
                                    self._current_latent.z_world[0].detach()
                                )
                            else:
                                _cur_list.append(
                                    torch.zeros(self.config.latent.world_dim)
                                )
                        cur_summaries = torch.stack(_cur_list, dim=0)
            # MECH-314a Phase 2 (Candidate 5A): build per-candidate first-action
            # one-hots for the substrate-robustness augmentation leg. Only built
            # when the augmentation is configured on (use_first_action_onehot +
            # policy != "never"); otherwise None -> compute_score_bias ignores
            # it and the path is bit-identical.
            cur_onehots = None
            if (
                getattr(self.config, "curiosity_use_first_action_onehot", False)
                and getattr(
                    self.config,
                    "curiosity_first_action_augmentation_policy",
                    "never",
                )
                != "never"
                and len(candidates) > 0
            ):
                _adim = int(candidates[0].actions.shape[-1])
                _oh = torch.zeros(len(candidates), _adim)
                for _i, _c in enumerate(candidates):
                    _cls = int(
                        _c.actions[:, 0, :].argmax(dim=-1).flatten()[0].item()
                    )
                    _oh[_i, _cls] = 1.0
                cur_onehots = _oh
            with torch.no_grad():
                cur_bias = self.curiosity.compute_score_bias(
                    candidate_world_summaries=cur_summaries.detach(),
                    residue_field=self.residue_field,
                    e3=self.e3,
                    simulation_mode=False,
                    visitation_source=self._zworld_visitation_buffer,
                    first_action_onehots=cur_onehots,
                )
            if self.e3.e3_score_decomp_enabled:
                _bdc_curiosity = cur_bias.detach().clone()
            if dacc_score_bias is None:
                dacc_score_bias = cur_bias
            else:
                dacc_score_bias = dacc_score_bias + cur_bias.to(
                    dtype=dacc_score_bias.dtype, device=dacc_score_bias.device
                )

        # MECH-320 (ARC-066 child): tonic_vigor_coupling_score_bias.
        # Target-free, capacity-keyed additive (or multiplicative-gain
        # falsifiable secondary) bias on E3 trajectory scoring. Negative
        # bias on action-trajectories (REE lower-is-better favours action);
        # positive bias on no-op-trajectories (penalises passivity). Vigor
        # scalar = slow EWMA over realised E3-score-receipt, gated by
        # secondary internal-state modulators (energy / drive / recent PE).
        # Composed AFTER MECH-314 curiosity (orthogonal axis: curiosity
        # rewards novelty; vigor rewards target-free action) and BEFORE
        # MECH-313 noise_floor (which lifts softmax temperature, not
        # scores). Bit-identical baseline when self.tonic_vigor is None.
        if self.tonic_vigor is not None:
            K_cands = len(candidates)
            # Per-candidate first-step action class.
            tv_action_classes = torch.zeros(K_cands, dtype=torch.long)
            for _i, _c in enumerate(candidates):
                # _c.actions has shape [batch, horizon, action_dim]; first-
                # step action class = argmax over action_dim.
                _first_action = _c.actions[:, 0, :]
                tv_action_classes[_i] = int(_first_action.argmax(dim=-1).flatten()[0].item())
            # Multiplicative-form magnitude anchor: pre-MECH-320 score_bias
            # (additive contributions from upstream sources). When
            # dacc_score_bias is None, use zeros -- multiplicative form
            # then yields zero bias (no pre-existing preference to
            # amplify). Additive form ignores the magnitude argument.
            if dacc_score_bias is not None:
                tv_score_anchor = dacc_score_bias.detach()
            else:
                tv_score_anchor = torch.zeros(K_cands)
            # Drive: post-pACC effective drive computed earlier in
            # select_action (matches AIC / PCC / SalienceCoordinator reads).
            tv_drive = float(_effective_drive_level)
            # Energy proxy: 1 - drive. Bounded [0, 1]; SD-012 owns the
            # depleted regime via gate_drive, but gate_energy provides an
            # additional rail when the deficit is severe.
            tv_energy = max(0.0, min(1.0, 1.0 - tv_drive))
            # Recent PE: e3._running_variance (per-tick PE-MSE accumulator;
            # same signal MECH-314c learning-progress reads).
            tv_pe = float(getattr(self.e3, "_running_variance", 0.0))
            with torch.no_grad():
                tv_bias = self.tonic_vigor.compute_score_bias(
                    per_candidate_scores=tv_score_anchor,
                    action_classes=tv_action_classes,
                    energy=tv_energy,
                    drive=tv_drive,
                    recent_pe=tv_pe,
                    simulation_mode=False,
                )
            if self.e3.e3_score_decomp_enabled:
                _bdc_vigor = tv_bias.detach().clone()
            if _fcg_channels is not None:  # MECH-451: vigour (MECH-320) channel
                _fcg_channels["vigour"] = tv_bias.detach().clone()
            # ControlVector logging (rec-B): split the single MECH-320 bias into
            # its action half (G_vigor = -w_action*v_t) and no-op half
            # (C_time = +w_passive*v_t), and record the shared v_t scalar + the
            # two config weights. Logging both halves AND their common v_t is
            # what makes the C_time<->G_vigor collapse (one scalar, two weights)
            # directly computable from a manifest. Read-only; no scoring change.
            if getattr(self.config, "use_control_vector_logging", False):
                _tv_state = self.tonic_vigor.get_state()
                _tv_noop_cls = int(self.tonic_vigor.config.noop_class)
                _tv_b = tv_bias.detach().reshape(-1)
                _tv_is_noop = tv_action_classes.reshape(-1) == _tv_noop_cls
                _n_noop = int(_tv_is_noop.sum().item())
                _n_action = int(_tv_b.numel() - _n_noop)
                _action_mean = (
                    float(_tv_b[~_tv_is_noop].mean().item()) if _n_action > 0 else 0.0
                )
                _noop_mean = (
                    float(_tv_b[_tv_is_noop].mean().item()) if _n_noop > 0 else 0.0
                )
                self._cv_vigor = {
                    "v_t": float(_tv_state.get("last_v_t", 0.0)),
                    "v_raw": float(_tv_state.get("v_raw", 0.0)),
                    "w_action": float(self.tonic_vigor.config.w_action),
                    "w_passive": float(self.tonic_vigor.config.w_passive),
                    # Potential per-half magnitudes (independent of whether a
                    # candidate of each class exists this tick): the collapse is
                    # that both are w * v_t for the SAME v_t.
                    "C_time_potential": float(self.tonic_vigor.config.w_passive)
                    * float(_tv_state.get("last_v_t", 0.0)),
                    "G_vigor_potential": float(self.tonic_vigor.config.w_action)
                    * float(_tv_state.get("last_v_t", 0.0)),
                    # Realised per-candidate means actually applied this tick.
                    "C_time_realised_mean": _noop_mean,
                    "G_vigor_realised_mean": _action_mean,
                    "n_action_candidates": _n_action,
                    "n_noop_candidates": _n_noop,
                }
            if dacc_score_bias is None:
                dacc_score_bias = tv_bias
            else:
                dacc_score_bias = dacc_score_bias + tv_bias.to(
                    dtype=dacc_score_bias.dtype, device=dacc_score_bias.device
                )

        # MECH-353 ASSERT consumer: when z_block is asserting (an external block
        # is sustained while capacity-belief is retained), bias selection toward
        # ACTION (escalate effort -- the "raise MECH-320 vigor" pole) and away
        # from the just-blocked action class (alternative-action search). Pure
        # additive score-bias composed like the other regulators; zero when no
        # asserting block this tick, so OFF / no-block ticks add nothing.
        if self.blocked_agency is not None and len(candidates) > 0:
            _ba_classes = [
                int(c.actions[:, 0, :].argmax(dim=-1).flatten()[0].item())
                for c in candidates
            ]
            _ba_dtype = (
                dacc_score_bias.dtype if dacc_score_bias is not None else torch.float32
            )
            _ba_device = (
                dacc_score_bias.device if dacc_score_bias is not None else self.device
            )
            ba_bias = self.blocked_agency.compute_assert_score_bias(
                action_classes=_ba_classes,
                device=_ba_device,
                dtype=_ba_dtype,
            )
            if dacc_score_bias is None:
                dacc_score_bias = ba_bias
            else:
                dacc_score_bias = dacc_score_bias + ba_bias.to(
                    dtype=dacc_score_bias.dtype, device=dacc_score_bias.device
                )

        # SD-058 / MECH-357: instrumental-avoidance ACTION pathway. Under
        # retained threat (z_harm_a), penalise the no-op / freeze class
        # proportional to the learned/scaffolded avoidance-efficacy so the
        # instrumental action is RELEASED (the ilPFC resolution of the
        # Pavlovian-instrumental conflict). The gate does NOT pick the escape
        # direction -- E3's existing harm gradient ranks the directed candidates.
        # Composed last in the additive score-bias chain. Zero (bit-identical)
        # below threat / at zero efficacy / when the gate is disabled.
        if self.instrumental_avoidance is not None and len(candidates) > 0:
            _ia_zha = (
                self._current_latent.z_harm_a
                if self._current_latent is not None
                else None
            )
            _ia_zn = (
                float(_ia_zha.detach().norm().item()) if _ia_zha is not None else 0.0
            )
            _ia_classes = [
                int(c.actions[:, 0, :].argmax(dim=-1).flatten()[0].item())
                for c in candidates
            ]
            _ia_dtype = (
                dacc_score_bias.dtype if dacc_score_bias is not None else torch.float32
            )
            _ia_device = (
                dacc_score_bias.device if dacc_score_bias is not None else self.device
            )
            ia_bias = self.instrumental_avoidance.compute_action_bias(
                z_harm_a_norm=_ia_zn,
                action_classes=_ia_classes,
                noop_class=int(self.config.avoidance_noop_class),
                device=_ia_device,
                dtype=_ia_dtype,
                simulation_mode=False,
            )
            if dacc_score_bias is None:
                dacc_score_bias = ia_bias
            else:
                dacc_score_bias = dacc_score_bias + ia_bias.to(
                    dtype=dacc_score_bias.dtype, device=dacc_score_bias.device
                )

        # SD-059 / MECH-358: relief/safety escape-affordance APPROACH bonus.
        # Under FUTURE threat (z_harm_a), favour (negative bias -- REE
        # lower-is-better) each candidate whose first-action class carries
        # learned escape-affordance credit -- the DIRECTED escape MECH-357
        # cannot supply (its scalar bias only penalises the no-op class).
        # Threat-context-gated (zero when safe, so it never swamps food/goal
        # approach) and clamped to escape_bias_scale. Composed after the
        # MECH-357 action-bias so the two compose additively. Bit-identical when
        # the bridge is disabled / no class has credit / below threat.
        if self.escape_affordance_bridge is not None and len(candidates) > 0:
            _eab_zha = (
                self._current_latent.z_harm_a
                if self._current_latent is not None
                else None
            )
            _eab_zn = (
                float(_eab_zha.detach().norm().item()) if _eab_zha is not None else 0.0
            )
            _eab_classes = [
                int(c.actions[:, 0, :].argmax(dim=-1).flatten()[0].item())
                for c in candidates
            ]
            _eab_dtype = (
                dacc_score_bias.dtype if dacc_score_bias is not None else torch.float32
            )
            _eab_device = (
                dacc_score_bias.device if dacc_score_bias is not None else self.device
            )
            eab_bias = self.escape_affordance_bridge.compute_approach_bias(
                z_harm_a_norm=_eab_zn,
                action_classes=_eab_classes,
                device=_eab_device,
                dtype=_eab_dtype,
                simulation_mode=False,
            )
            if dacc_score_bias is None:
                dacc_score_bias = eab_bias
            else:
                dacc_score_bias = dacc_score_bias + eab_bias.to(
                    dtype=dacc_score_bias.dtype, device=dacc_score_bias.device
                )

        # Post-603i trainable relief/safety escape-affordance learner. Bounded,
        # threat-gated negative score bias toward actions predicted to produce
        # relief and/or response-produced safety. Composes safely with the
        # arithmetic bridge; each module clamps its own contribution.
        if (
            self.trainable_escape_affordance_learner is not None
            and len(candidates) > 0
        ):
            _teal_zha = (
                self._current_latent.z_harm_a
                if self._current_latent is not None
                else None
            )
            _teal_zn = (
                float(_teal_zha.detach().norm().item())
                if _teal_zha is not None
                else 0.0
            )
            _teal_classes = [
                int(c.actions[:, 0, :].argmax(dim=-1).flatten()[0].item())
                for c in candidates
            ]
            _teal_dtype = (
                dacc_score_bias.dtype if dacc_score_bias is not None else torch.float32
            )
            _teal_device = (
                dacc_score_bias.device if dacc_score_bias is not None else self.device
            )
            teal_bias = (
                self.trainable_escape_affordance_learner.compute_approach_bias(
                    z_harm_a_norm=_teal_zn,
                    action_classes=_teal_classes,
                    z_world=(
                        getattr(self._current_latent, "z_world", None)
                        if self._current_latent is not None
                        else None
                    ),
                    z_self=(
                        getattr(self._current_latent, "z_self", None)
                        if self._current_latent is not None
                        else None
                    ),
                    z_harm_a=_teal_zha,
                    extra_features=self._eal_linker_context_feature(),
                    device=_teal_device,
                    dtype=_teal_dtype,
                    simulation_mode=False,
                    hypothesis_tag=False,
                )
            )
            if dacc_score_bias is None:
                dacc_score_bias = teal_bias
            else:
                dacc_score_bias = dacc_score_bias + teal_bias.to(
                    dtype=dacc_score_bias.dtype, device=dacc_score_bias.device
                )

        # Post-603i E2 escape-affordance linker: bounded, threat-gated negative
        # score bias toward the action class its readouts predict will produce
        # relief/escape under CURRENT threat -- the DIRECTED escape over E2
        # geometry. Behind use_e2_escape_linker_e3_bias (separate from the
        # learner/training flags). Threat-gated (zero when safe), clamped to
        # escape_linker_bias_scale, never applied to the no-op class. Composes
        # additively with the bridge / learner; each module clamps its own term.
        if (
            self.e2_escape_affordance_linker is not None
            and getattr(self.config, "use_e2_escape_linker_e3_bias", False)
            and len(candidates) > 0
        ):
            _eal_zha2 = (
                self._current_latent.z_harm_a
                if self._current_latent is not None
                else None
            )
            _eal_zn2 = (
                float(_eal_zha2.detach().norm().item())
                if _eal_zha2 is not None
                else 0.0
            )
            _eal_classes = [
                int(c.actions[:, 0, :].argmax(dim=-1).flatten()[0].item())
                for c in candidates
            ]
            _eal_dtype = (
                dacc_score_bias.dtype if dacc_score_bias is not None else torch.float32
            )
            _eal_device = (
                dacc_score_bias.device if dacc_score_bias is not None else self.device
            )
            eal_bias = self.e2_escape_affordance_linker.compute_approach_bias(
                z_harm_a_norm=_eal_zn2,
                action_classes=_eal_classes,
                z_world=(
                    getattr(self._current_latent, "z_world", None)
                    if self._current_latent is not None
                    else None
                ),
                z_self=(
                    getattr(self._current_latent, "z_self", None)
                    if self._current_latent is not None
                    else None
                ),
                z_harm_a=_eal_zha2,
                device=_eal_device,
                dtype=_eal_dtype,
                simulation_mode=False,
                hypothesis_tag=False,
            )
            if dacc_score_bias is None:
                dacc_score_bias = eal_bias
            else:
                dacc_score_bias = dacc_score_bias + eal_bias.to(
                    dtype=dacc_score_bias.dtype, device=dacc_score_bias.device
                )

        _fsb = getattr(self.config, "forced_score_bias_per_class", None)
        self._last_candidate_support_preflight = candidate_support_preflight(
            candidates,
            forced_score_bias_per_class=_fsb,
        )

        # V3-EXQ-563 diagnostic: forced_score_bias_per_class injection.
        # Bypasses all naturalistic signal generation to verify the
        # score-bias -> action-change seam independently of MECH-313/314/320
        # signal failures. Bit-identical when config field is None (default).
        if _fsb is not None and len(candidates) > 0:
            _fb_classes = self._last_candidate_support_preflight.get(
                "candidate_first_action_classes", []
            )
            if len(_fb_classes) != len(candidates):
                _fb_classes = [
                    int(c.actions[:, 0, :].argmax(dim=-1).flatten()[0].item())
                    for c in candidates
                ]
            _forced_bias = torch.tensor(
                [_fsb[cls] if cls < len(_fsb) else 0.0 for cls in _fb_classes],
                dtype=torch.float32,
            )

            if self.e3.e3_score_decomp_enabled:
                _bdc_forced = _forced_bias.detach().clone()
            if dacc_score_bias is None:
                dacc_score_bias = _forced_bias
            else:
                dacc_score_bias = dacc_score_bias + _forced_bias.to(
                    dtype=dacc_score_bias.dtype, device=dacc_score_bias.device
                )

        # MECH-294: OPTIONAL read-only-first compose of the sealed theta packet's
        # joint_context into the E3 score-bias chain. Gated behind
        # theta_packet_compose_into_e3_bias (default False -> bit-identical OFF,
        # the packet stays read-only and the substrate-readiness validation
        # measures it without behavioural authority). Parameter-free (cosine of
        # candidate first-action vs the packet's co-bound action_proposal,
        # clamped) -> no trained head, no phased training (memo S8). Composed last
        # so it never dominates the primary chain.
        if (
            self.multi_content_theta_packet is not None
            and getattr(self.config, "theta_packet_compose_into_e3_bias", False)
            and candidates
        ):
            _tp_fa_list = []
            for _c in candidates:
                _a = getattr(_c, "actions", None)
                if _a is not None and _a.shape[1] > 0:
                    _tp_fa_list.append(_a[:, 0, :].reshape(-1))
            if _tp_fa_list:
                _tp_cand_fa = torch.stack(_tp_fa_list, dim=0)  # [K, action_dim]
                # Per-candidate co-binding coherence (route-range amend, triage
                # 2026-06-19): when on, the compose bias carries a mode-distinct
                # cross-candidate RANGE (joint full / alternation different pattern
                # / shuffled ~0) so the route-range authority + 569i top-k can
                # carve it -- instead of the legacy scalar-gated action-only cosine
                # whose per-candidate pattern is mode-invariant. _bdc_coherence
                # (the route source "coherence") then routes the carve-able channel.
                if getattr(
                    self.config, "theta_packet_compose_per_candidate_coherence", False
                ):
                    _tp_bias = self.multi_content_theta_packet.compose_per_candidate_coherence(
                        _tp_cand_fa,
                        bias_scale=getattr(self.config, "theta_packet_bias_scale", 0.1),
                    )
                else:
                    _tp_bias = self.multi_content_theta_packet.compose_e3_bias(
                        _tp_cand_fa,
                        bias_scale=getattr(self.config, "theta_packet_bias_scale", 0.1),
                        use_joint_coherence=getattr(
                            self.config, "theta_packet_compose_use_joint_coherence", True
                        ),
                    )
                if _tp_bias is not None:
                    _bdc_coherence = _tp_bias.detach().clone()
                    if dacc_score_bias is None:
                        dacc_score_bias = _tp_bias
                    else:
                        dacc_score_bias = dacc_score_bias + _tp_bias.to(
                            dtype=dacc_score_bias.dtype, device=dacc_score_bias.device
                        )

        # MECH-313 (ARC-065): stochastic_noise_floor. Lifts the softmax
        # temperature uniformly to prevent argmax collapse (LC-NE tonic
        # analog; SAC max-entropy regularisation analog). State-
        # independent -- Q-045 falsifies whether this and MECH-260
        # (state-dependent dACC anti-recency) collapse into a single
        # substrate. simulation_mode=False here (waking action selection).
        # Bit-identical when self.noise_floor is None.
        if self.noise_floor is not None:
            effective_temperature = self.noise_floor.compute_effective_temperature(
                baseline_temperature=temperature,
                simulation_mode=False,
            )
        else:
            effective_temperature = temperature

        self._last_e3_score_bias = (
            dacc_score_bias.detach().clone()
            if dacc_score_bias is not None
            else None
        )
        if self.e3.e3_score_decomp_enabled:
            def _bdc_mean(t: "Optional[torch.Tensor]") -> float:
                return float(t.mean().item()) if t is not None else 0.0

            # Governance disposition #3 from
            # evidence/planning/v3_exq_571_root_cause_2026-05-25.md: the
            # channel-keyed mean collapses [K] per-candidate bias vectors and
            # reads ~0 whenever cross-K spread varies on a stationary mean.
            # Surface the spread directly via per-channel std_across_K +
            # bias_range_mean. Existing channel-keyed mean float preserved so
            # current consumers (e.g. EXQ-571 BIAS_COMPONENTS reader) keep
            # working bit-identically.
            def _bdc_std(t: "Optional[torch.Tensor]") -> float:
                if t is None:
                    return 0.0
                flat = t.reshape(-1)
                if flat.numel() < 2:
                    return 0.0
                return float(flat.std(unbiased=False).item())

            def _bdc_range_mean(t: "Optional[torch.Tensor]") -> float:
                if t is None:
                    return 0.0
                # Average max-minus-min over batch dim when present, else
                # global range. _bdc_* tensors are typically [K] (one
                # candidate vector per channel); for [batch, K] inputs we
                # average per-row ranges.
                if t.dim() >= 2:
                    rng = t.amax(dim=-1) - t.amin(dim=-1)
                    return float(rng.mean().item())
                if t.numel() == 0:
                    return 0.0
                return float((t.max() - t.min()).item())

            self._last_score_bias_decomp = {
                "dacc": _bdc_mean(_bdc_dacc),
                "lateral_pfc": _bdc_mean(_bdc_lpfc),
                "ofc": _bdc_mean(_bdc_ofc),
                "gated_policy": _bdc_mean(_bdc_gp),
                "mech295_liking": _bdc_mean(_bdc_m295),
                "curiosity": _bdc_mean(_bdc_curiosity),
                "tonic_vigor": _bdc_mean(_bdc_vigor),
                "forced": _bdc_mean(_bdc_forced),
                "noise_floor_temp": float(effective_temperature),
                "total_bias": _bdc_mean(self._last_e3_score_bias),
                "dacc_std_across_K": _bdc_std(_bdc_dacc),
                "lateral_pfc_std_across_K": _bdc_std(_bdc_lpfc),
                "ofc_std_across_K": _bdc_std(_bdc_ofc),
                "gated_policy_std_across_K": _bdc_std(_bdc_gp),
                "mech295_liking_std_across_K": _bdc_std(_bdc_m295),
                "curiosity_std_across_K": _bdc_std(_bdc_curiosity),
                "tonic_vigor_std_across_K": _bdc_std(_bdc_vigor),
                "forced_std_across_K": _bdc_std(_bdc_forced),
                "total_bias_std_across_K": _bdc_std(self._last_e3_score_bias),
                "dacc_bias_range_mean": _bdc_range_mean(_bdc_dacc),
                "lateral_pfc_bias_range_mean": _bdc_range_mean(_bdc_lpfc),
                "ofc_bias_range_mean": _bdc_range_mean(_bdc_ofc),
                "gated_policy_bias_range_mean": _bdc_range_mean(_bdc_gp),
                "mech295_liking_bias_range_mean": _bdc_range_mean(_bdc_m295),
                "curiosity_bias_range_mean": _bdc_range_mean(_bdc_curiosity),
                "tonic_vigor_bias_range_mean": _bdc_range_mean(_bdc_vigor),
                "forced_bias_range_mean": _bdc_range_mean(_bdc_forced),
                "total_bias_bias_range_mean": _bdc_range_mean(self._last_e3_score_bias),
            }
        # modulatory-bias-selection-authority AMEND (route-range, 569f/661/654a,
        # 2026-06-10): build the channel-under-test's per-candidate routed bias and
        # pass it to E3 so its cross-candidate range is folded into the modulatory
        # accumulator the authority rescales (instead of being flattened by the
        # consuming bias head). The route source selects which channel's
        # representation is projected (parameter-free, range-preserving):
        #   cand_world_summary -> cand_world_summaries [K, world_dim] (the 569f
        #     world-summary channel; the genuine [K, world_dim] projection case)
        #   curiosity / gated_policy / mech295 / coherence -> the already-computed
        #     per-candidate [K] bias for that channel (identity-routed).
        # Default "none" -> channel_route_bias=None -> bit-identical.
        channel_route_bias = None
        if getattr(self.config, "use_modulatory_channel_routing", False):
            _route_source = getattr(self.config, "modulatory_channel_route_source", "none")
            _route_repr = None
            if _route_source == "cand_world_summary":
                # cand_world_summaries is set only inside the bias blocks; use the
                # in-scope value when a block ran this tick, else build it (the
                # e2_world_forward / proposer summaries via the ARC-065 GAP-A helper).
                _route_repr = locals().get("cand_world_summaries", None)
                if _route_repr is None:
                    _route_repr = self._candidate_world_summaries(candidates)
            elif _route_source == "curiosity":
                _route_repr = _bdc_curiosity
            elif _route_source == "gated_policy":
                _route_repr = _bdc_gp
            elif _route_source == "mech295":
                _route_repr = _bdc_m295
            elif _route_source == "coherence":
                _route_repr = _bdc_coherence
            if _route_repr is not None:
                channel_route_bias = project_channel_range(_route_repr)
        # DR-12 (self_model_v4:SELF-4) VERSION-LAYERING GUARD (2026-06-17): the
        # e2_forward_pe_per_candidate kwarg is a V4 call-site into the shared V3
        # E3.select path. It is passed ONLY when the V4 DR-12 feature is engaged
        # (config.e3.use_pe_confidence_weighting) OR a per-candidate PE has been
        # injected. The DEFAULT V3 path therefore never passes the kwarg, so a
        # skewed / older e3_selector.select() that lacks the param cannot raise a
        # TypeError on the V3 critical path. This is the fix for the 2026-06-17
        # V3-EXQ-654e crash-burn (an unconditional V4 call-site entered the V3
        # hot path). See ree_core/version_layering.py + the Version-Layering
        # Doctrine (REE_assembly/docs/architecture/version_layering_doctrine.md).
        _e3_select_kwargs = dict(
            goal_state=_goal_state_for_select,
            terrain_weight=self._cue_terrain_weight,
            sweep_threshold_reduction=sweep_reduction,
            z_harm_a=z_harm_a,
            score_bias=dacc_score_bias,
            score_diversity=self.score_diversity,
            channel_route_bias=channel_route_bias,
        )
        # MECH-451: pass the un-summed finer per-head biases ONLY when finer gating
        # is engaged (version-layering guard: the default V3 path never sends the
        # kwarg, so an older e3.select() that lacks the param cannot raise on the
        # critical path -- same doctrine as the DR-12 / Go-No-Go guards below).
        if getattr(self.config.e3, "use_finer_channel_gating", False):
            _e3_select_kwargs["score_bias_channels"] = _fcg_channels
        # MECH-449 (ARC-107): Go/No-Go eligibility constitution. The perseveration
        # No-Go axis REUSES MECH-260 (dACC anti-recency suppression) -- the
        # existing per-candidate recency-share vector from the dACC bundle is
        # routed in as the perseveration signal (generalising MECH-260 from a
        # drowned score-bias into an eligibility-access gate, ARC-106 G2). The
        # other axes (safety / staleness / viability / go) are experiment-supplied
        # on a constructed candidate bank (the MECH-449 falsifier); the default
        # waking loop wires only the always-available MECH-260 reuse. Passed ONLY
        # when the constitution is engaged (version-layering guard: the default
        # path never sends the kwarg, so an older e3.select cannot raise).
        if getattr(self.config.e3, "use_go_nogo_constitution", False):
            _gng_signals: Dict[str, Any] = {}
            _gng_bundle = getattr(self, "_dacc_last_bundle", None)
            if _gng_bundle is not None:
                _gng_supp = _gng_bundle.get("suppression", None)
                if _gng_supp is not None:
                    _gng_signals["perseveration"] = _gng_supp
            _injected_gng = getattr(self, "_injected_go_nogo_signals", None)
            if _injected_gng:
                _gng_signals.update(_injected_gng)
            if _gng_signals:
                _e3_select_kwargs["go_nogo_signals"] = _gng_signals
        _injected_e2_pe = getattr(self, "_injected_e2_forward_pe", None)
        if (
            getattr(self.config.e3, "use_pe_confidence_weighting", False)
            or _injected_e2_pe is not None
        ):
            _e3_select_kwargs["e2_forward_pe_per_candidate"] = _injected_e2_pe
        # MECH-441 (model_disagreement_directed_curiosity): per-candidate forward-
        # model disagreement (cross-head variance of the K-head ensemble) -> E3
        # selection as a propagating curiosity bonus. Version-layering guard: the
        # kwarg is passed ONLY when the lever is on AND the ensemble is built, so
        # the default V3 path never sends it (an older e3.select cannot raise --
        # same doctrine as the DR-12 / MECH-449 guards above). Waking read (no_grad).
        if (
            getattr(self.config.e3, "use_model_disagreement_curiosity", False)
            and getattr(self, "disagreement_ensemble", None) is not None
            and self._current_latent is not None
            and len(candidates) >= 1
        ):
            try:
                _md_adim = int(candidates[0].actions.shape[-1])
                _md_acts = torch.stack(
                    [c.actions[:, 0, :].detach().reshape(_md_adim) for c in candidates],
                    dim=0,
                )
                _md_vec = self.disagreement_ensemble.disagreement_per_candidate(
                    self._current_latent.z_world.detach(),
                    _md_acts,
                    simulation_mode=False,
                )
                _e3_select_kwargs["model_disagreement_per_candidate"] = _md_vec
            except Exception:
                pass
        result = self.e3.select(
            candidates, effective_temperature,
            **_e3_select_kwargs,
        )
        self._last_e3_selection_result = result

        # Closure-plane commit-ENTRY primitive (rung-6 amend; commitment_closure:GAP-4;
        # failure_autopsy_V3-EXQ-460k/460l 2026-06-22). Option A (user-confirmed): SET the
        # F-INDEPENDENT latch e3._closure_committed_active when a goal-active, rule-directed
        # commitment forms -- goal_state.is_active() AND a trajectory was selected toward it
        # (the hippocampal proposer is goal-seeded under an active goal, so any selection is
        # goal-directed) AND a rule is being followed (lateral_pfc.rule_state norm above the
        # closure_commit_entry_rule_norm_floor, mirroring the SD-034 closure operator's
        # rule-stability precursor). This is the F-independent path the closure-exclusive
        # de-commit eval needs: on the 460j substrate the F-driven natural commit never
        # sustains, so _committed_trajectory is rarely non-None and the eval never arms
        # (ncl_hold_closure_armed_total=0, the 460k/460l signature). The latch is STICKY
        # across ticks (it is NOT torn down by post_action_update like _committed_trajectory)
        # and is cleared only on a principled closure teardown (SD-034 closure fire /
        # de-commit refractory install / episode reset). MECH-094: waking control-state
        # transition only (select_action is the waking path, simulation_mode=False here, as
        # the neighbouring tonic_vigor / closure-coupling sites assume); no replay/memory
        # write surface. No-op default -> the latch is never set -> agent.py:6365
        # _closure_commit_active reduces to the legacy `_committed_trajectory is not None` ->
        # bit-identical for every existing run.
        if (
            getattr(self.config, "use_closure_commit_entry", False)
            and self.goal_state is not None
            and self.goal_state.is_active()
            and result is not None
            and result.selected_action is not None
            and self.lateral_pfc is not None
        ):
            _rule_norm = float(self.lateral_pfc.rule_state.norm().item())
            _rule_floor = float(
                getattr(self.config, "closure_commit_entry_rule_norm_floor", 0.01)
            )
            if _rule_norm >= _rule_floor:
                self.e3._closure_committed_active = True
                # C-STEP extension: also install the goal/rule-directed selected
                # trajectory into the PARALLEL sticky latch so the between-E3-tick
                # stepping path can advance a closure-formed committed PROGRAM rather
                # than repeat _last_action. On a FRESH closure arm (latch was None)
                # reset the step counter so stepping starts at the program head; on
                # subsequent E3 ticks the trajectory is refreshed but the counter keeps
                # advancing across the held occupancy (mirrors the F-commit stepping,
                # which resets _committed_step_idx only on beta release / reset). No-op
                # default: skipped unless use_closure_commit_entry_trajectory is on, so
                # _closure_committed_trajectory stays None and every union below reduces
                # to the bool-latch behaviour.
                if getattr(
                    self.config, "use_closure_commit_entry_trajectory", False
                ):
                    if self.e3._closure_committed_trajectory is None:
                        self._committed_step_idx = 0
                    self.e3._closure_committed_trajectory = (
                        result.selected_trajectory
                    )

        # ControlVector logging (rec-B four-signal adjudication 2026-06-07).
        # Read-only assembly of the four control signals AFTER selection, so
        # V_outcome reflects the realised primary scores. No scoring/selection
        # effect. Bit-identical when use_control_vector_logging is False (block
        # skipped). Read directly by experiment scripts via
        # agent._last_control_vector (the V3-EXQ-571 _last_score_bias_decomp
        # pattern).
        if getattr(self.config, "use_control_vector_logging", False):
            self._assemble_control_vector(
                effective_temperature=float(effective_temperature),
                baseline_temperature=float(temperature),
            )

        action = result.selected_action

        # MECH-314c learning-progress feed: push the current waking
        # tick's PE-magnitude scalar into the structured-curiosity LP
        # buffer so the next tick's 314c bonus reflects |PE_t - PE_{t-K}|
        # rate-of-improvement (Schmidhuber 1991 first-difference). Phase 1
        # signal source is e3._running_variance (a per-tick PE-MSE
        # accumulator). simulation_mode=False here (waking path).
        if self.curiosity is not None:
            pe_scalar = float(getattr(self.e3, "_running_variance", 0.0))
            self.curiosity.update_prediction_error(
                pe_scalar=pe_scalar, simulation_mode=False,
            )

        # MECH-320 EWMA feed: push the realised E3 score of the SELECTED
        # candidate into the tonic-vigor EWMA so the NEXT tick's vigor
        # scalar reflects the current waking reward stream (Niv 2007 long-
        # run avg-reward-rate formalism). REE convention is lower-is-
        # better; module internally negates so v_raw climbs in reward-
        # rich regimes. simulation_mode=False (waking action selection).
        if self.tonic_vigor is not None:
            try:
                _selected_score_val = float(
                    result.scores[result.selected_index].detach().item()
                )
            except (IndexError, AttributeError, RuntimeError):
                _selected_score_val = 0.0
            self.tonic_vigor.update_score_receipt(
                score=_selected_score_val, simulation_mode=False,
            )

        # MECH-090: gate policy propagation based on beta state.
        bistable = self.config.heartbeat.beta_gate_bistable
        # MECH-090 R-c commit-entry readiness conjunction (2026-05-28).
        # Per-candidate first-action margin off the post-bias E3 scores
        # (REE lower-is-better -> winner = argmin, margin = sorted[1] - sorted[0]).
        # When use_commit_readiness_gate=False the gate returns True
        # unconditionally and the rv-only legacy path is bit-identical.
        # See HeartbeatConfig + ree_core/heartbeat/beta_gate.py.
        _readiness_margin: Optional[float] = None
        _n_candidates: int = 0
        if (
            result.committed
            and getattr(
                self.config.heartbeat, "use_commit_readiness_gate", False
            )
        ):
            try:
                _scores = result.scores.detach()
                _n_candidates = int(_scores.numel())
                if _n_candidates >= 2:
                    _sorted, _ = torch.sort(_scores)
                    _readiness_margin = float(_sorted[1].item() - _sorted[0].item())
            except (AttributeError, RuntimeError, TypeError):
                # Fall-through: should_admit_elevation receives None + 0
                # and the permissive single-candidate path admits.
                _readiness_margin = None
                _n_candidates = 0
        # MECH-090 R-c readiness-conjunction (nav_competence axis, 2026-05-28).
        # Composes WITH the score_margin gate via AND: both gates must admit
        # for elevation. score_margin is the within-tick decisiveness reading
        # (Hanes & Schall 1996); CommitReadiness is the across-tick motor-
        # program readiness reading (Cisek-Kalaska affordance + Roesch
        # premature-commit pathology). With use_mech090_readiness_conjunction
        # =False, _readiness_admits returns True unconditionally and the
        # combined predicate reduces to the score_margin gate (or rv-only when
        # both flags are off). See ree_core/policy/commit_readiness.py.
        _readiness_admits: bool = (
            self.commit_readiness.is_above_floor(
                float(getattr(self.config, "mech090_readiness_floor", 0.3))
            )
            if (
                self.commit_readiness is not None
                and getattr(
                    self.config, "use_mech090_readiness_conjunction", False
                )
                and result.committed
            )
            else True
        )
        # MECH-090 R-c diagnostic: count blocks at the source. Only fires
        # when the readiness conjunction was actually consulted AND returned
        # False (the rv-low signal would have elevated under rv-only). The
        # counter lives on the CommitReadiness module via notify_block.
        if (
            not _readiness_admits
            and self.commit_readiness is not None
            and result.committed
        ):
            self.commit_readiness.notify_block()
        if bistable:
            # Bistable latch: only elevate on commit ENTRY (not every tick).
            # Release is triggered by hippocampal completion signal in _e3_tick(),
            # not by variance re-evaluation. This prevents flickering when variance
            # hovers near the commit threshold.
            # SD-034 commitment-closure-control-plane BETA-ENGAGEMENT amend
            # (2026-06-17, failure_autopsy_V3-EXQ-460e). With both MECH-090 R-c
            # gates OFF the natural trigger is result.committed (running_variance <
            # commit_threshold), a decisive crossing that fires on only 1/3 seeds on
            # the 603n foraging substrate -- so beta never elevates even though the
            # closure control-plane installs a committed_trajectory + fires closures
            # (commit-without-beta dissociation; total_beta_elevated=0 on 2/3 seeds).
            # When use_closure_commit_beta_coupling is set, an active closure-plane
            # commitment (e3._committed_trajectory is not None) ALSO triggers
            # elevation, so latch occupancy is readable on every seed where a
            # commitment forms and the Leg-B de-commit refractory yields a
            # measurable ON<OFF occupancy drop (the 460f DV). The
            # should_admit_elevation / _readiness_admits conjunction is preserved so
            # the coupling composes with MECH-090 when those gates are on (both
            # return True permissively on the coupled path when result.committed is
            # False). No-op default -> _commit_for_beta == result.committed ->
            # bit-identical.
            # Closure-plane commit-entry UNION (rung-6 amend, 460k/460l). The closure
            # plane is "actively committed" when EITHER the legacy F-driven commit
            # trajectory is present OR the F-INDEPENDENT closure-plane commit-entry latch
            # is set (use_closure_commit_entry, SET above after e3.select). The union
            # preserves the legacy path exactly (latch never set when the flag is off ->
            # bit-identical), and gives the closure-exclusive eval a way to arm WITHOUT a
            # sustained F-commit -- closing the 460k/460l ncl_hold_closure_armed_total=0
            # signature. See REE_assembly/docs/architecture/natural_commit_occupancy_release.md.
            _closure_commit_active = (
                getattr(self.config, "use_closure_commit_beta_coupling", False)
                and (
                    self.e3._committed_trajectory is not None
                    or self.e3._closure_committed_active
                    # C-STEP extension: a closure-FORMED committed trajectory also arms
                    # the coupling (None unless use_closure_commit_entry_trajectory is on
                    # -> bit-identical to the bool-latch union when unused).
                    or self.e3._closure_committed_trajectory is not None
                )
            )
            # Closure-exclusive de-commit eval mode (rung-6 BUILD, 460j): SUPPRESS the
            # fragile F-driven natural commit (result.committed) from beta elevation so
            # the occupancy is formed EXCLUSIVELY by the closure->beta coupling -- the
            # named dissociable substrate. Default (flag off) -> the legacy union
            # (result.committed OR _closure_commit_active), bit-identical.
            if getattr(self.config, "closure_exclusive_decommit_eval", False):
                _commit_for_beta = _closure_commit_active
            else:
                _commit_for_beta = bool(result.committed) or _closure_commit_active
            # SD-034 460h refractory-INDEPENDENT coupling certifier
            # (failure_autopsy_V3-EXQ-460g). Count the closure-plane commit INTENT
            # -- a closure-coupled commitment forming while the natural
            # running_variance path did NOT fire -- BEFORE the elevate/refractory
            # gate, so MECH-445 coupling engagement is certifiable even when the
            # 460g de-commit-magnitude lever pins the refractory at its cap and a
            # no-op elevate() suppresses note_closure_coupled_elevation (the S5
            # entanglement). No-op when use_closure_commit_beta_coupling is off
            # (_closure_commit_active stays False) -> bit-identical.
            if _closure_commit_active and not result.committed:
                self.beta_gate.note_closure_commit_intent()
            if (
                _commit_for_beta
                and not self.beta_gate.is_elevated
                and self.beta_gate.should_admit_elevation(
                    score_margin=_readiness_margin, n_candidates=_n_candidates
                )
                and _readiness_admits
            ):
                self.beta_gate.elevate()
                # SD-034 diagnostic: count elevations driven by the closure-plane
                # coupling rather than a natural running_variance crossing.
                if _closure_commit_active and not result.committed:
                    self.beta_gate.note_closure_coupled_elevation()
                # Natural-commit latch-hold: ARM on a fresh NATURAL commit
                # (result.committed) so the end-of-tick re-assertion sustains its
                # beta-latch occupancy. A purely closure-coupled elevation
                # (result.committed False) does NOT arm the hold by default -- its
                # occupancy is governed by the SD-034 closure machinery, not the
                # rung-6 hold. EXCEPTION: under closure_exclusive_decommit_eval the
                # hold ALSO arms on the closure-coupled commit (_closure_commit_active),
                # because the fragile result.committed does not form on this substrate
                # (460j ncl_hold_reassert_total=0) -- arming on the closure plane is
                # what makes a sustained occupancy form for the de-commit to act on.
                # No-op when use_natural_commit_latch_hold is False. Re-arming an
                # already-active hold (a re-commit while held) just restarts its
                # tick budget on the fresh commit.
                _ncl_eval = getattr(
                    self.config, "closure_exclusive_decommit_eval", False
                )
                # The eval-mode closure-arm respects the SD-034 closure de-commit: do
                # NOT re-arm while a refractory is actively holding beta down (the hold
                # must yield to the de-commit, preserving the MECH-446 occupancy-drop;
                # it re-arms only once the refractory expires and a fresh closure commit
                # re-elevates). The legacy result.committed arm path is unchanged.
                _ncl_closure_arm = (
                    _ncl_eval
                    and _closure_commit_active
                    and self.beta_gate.refractory_remaining == 0
                )
                if (
                    getattr(
                        self.config, "use_natural_commit_latch_hold", False
                    )
                    and (bool(result.committed) or _ncl_closure_arm)
                ):
                    self._ncl_hold_active = True
                    self._ncl_hold_ticks = 0
                    if _ncl_closure_arm and not result.committed:
                        self._ncl_hold_closure_armed_count += 1
                    # ARC-108 JOB-2 (c): arm the rho_t maintenance ramp on the same
                    # hold entry (reset its running proximity peak). The ramp then
                    # drives the hold's self-limit at the end-of-tick re-assertion.
                    # No-op when the ramp is disabled.
                    if self.rho_maintenance_ramp is not None:
                        self.rho_maintenance_ramp.note_commit_entry()
                self._committed_step_idx = 0  # reset step counter on new commitment
                # MECH-342: zero the maintenance-release pressure accumulator
                # at commit entry so each committed program accumulates
                # release pressure independently. No-op when disabled.
                if self.maintenance_release is not None:
                    self.maintenance_release.reset_pressure()
                # Commit/release-DURATION lever: arm the natural-commit-occupancy
                # release at a fresh NATURAL commit entry (result.committed). A
                # purely closure-coupled elevation (result.committed False) is NOT
                # armed -- its occupancy is governed by the SD-034 machinery.
                # gap_norm = normalised top-F decisiveness in [0,1] (1 = a
                # decisive F-gap = the kind of commit that monopolises the latch).
                # No-op when the lever is disabled (natural_commit_urgency None).
                if (
                    self.natural_commit_urgency is not None
                    and result.committed
                ):
                    _ncur_gap_norm = 1.0
                    try:
                        _ncur_scores = result.scores.detach()
                        if int(_ncur_scores.numel()) >= 2:
                            _ncur_sorted, _ = torch.sort(_ncur_scores)
                            _ncur_gap = float(
                                _ncur_sorted[1].item() - _ncur_sorted[0].item()
                            )
                            _ncur_range = float(
                                _ncur_sorted[-1].item() - _ncur_sorted[0].item()
                            )
                            _ncur_gap_norm = (
                                _ncur_gap / (_ncur_range + 1e-8)
                                if _ncur_range > 0.0
                                else 1.0
                            )
                    except (AttributeError, RuntimeError, TypeError):
                        _ncur_gap_norm = 1.0
                    self.natural_commit_urgency.note_commit_entry(
                        _ncur_gap_norm
                    )
                # MECH-269 / MECH-090: snapshot active anchor keys at commit entry.
                # Read by select_action()'s V_s -> commit release block on
                # subsequent ticks. No-op when use_vs_commit_release is False.
                if (
                    getattr(
                        self.hippocampal.config, "use_vs_commit_release", False
                    )
                    and self.hippocampal.anchor_set is not None
                ):
                    self._committed_anchor_keys = {
                        a.key for a in self.hippocampal.anchor_set.active_anchors()
                    }
                # MECH-290: record committed trajectory at commit entry so that
                # backward_credit_sweep() has it when BetaGate releases.
                # No-op when use_backward_credit_sweep is False (backward compat).
                if (
                    self.e3._committed_trajectory is not None
                    and getattr(
                        self.hippocampal.config, "use_backward_credit_sweep", False
                    )
                ):
                    self.e3._committed_trajectory.memory_strength = float(
                        self._bla_last_output.encoding_gain
                    ) if self._bla_last_output is not None else 1.0
                    self.e3._committed_trajectory.arousal_tag = float(
                        self._bla_last_output.arousal_tag
                    ) if self._bla_last_output is not None else 0.0
                    self.hippocampal.record_committed_trajectory(
                        self.e3._committed_trajectory
                    )
            # If not committed and beta already released: no-op (already open).
            # If committed and beta already elevated: no-op (stay latched).
        else:
            # Legacy behavior (backward compat): re-evaluate every E3 tick.
            # MECH-090 R-c: an effective_committed conjunction of the rv-low
            # signal (result.committed) AND the readiness gate. With
            # use_commit_readiness_gate=False, should_admit_elevation
            # returns True unconditionally so effective_committed == result.committed
            # and the legacy path is bit-identical.
            _legacy_admit = (
                (
                    self.beta_gate.should_admit_elevation(
                        score_margin=_readiness_margin, n_candidates=_n_candidates
                    )
                    and _readiness_admits
                )
                if result.committed
                else True
            )
            if result.committed and _legacy_admit:
                # Fix: reset on EVERY E3 commit, not only on uncommitted->committed
                # transition. In non-bistable mode E3 re-selects a trajectory on every
                # E3 tick, so the counter must restart from 0 each time -- otherwise it
                # saturates at rollout_horizon-1 and replays the last action forever.
                # MECH-269 anchor snapshot and MECH-290 record_committed_trajectory
                # remain inside the inner guard: they are new-commit-only events.
                self._committed_step_idx = 0
                if not self.beta_gate.is_elevated:
                    # MECH-269 / MECH-090: snapshot active anchor keys at
                    # commit entry. No-op when use_vs_commit_release is False.
                    if (
                        getattr(
                            self.hippocampal.config,
                            "use_vs_commit_release",
                            False,
                        )
                        and self.hippocampal.anchor_set is not None
                    ):
                        self._committed_anchor_keys = {
                            a.key
                            for a in self.hippocampal.anchor_set.active_anchors()
                        }
                    # MECH-290: record committed trajectory at new commit entry.
                    # No-op when use_backward_credit_sweep is False (backward compat).
                    if (
                        self.e3._committed_trajectory is not None
                        and getattr(
                            self.hippocampal.config,
                            "use_backward_credit_sweep",
                            False,
                        )
                    ):
                        self.e3._committed_trajectory.memory_strength = float(
                            self._bla_last_output.encoding_gain
                        ) if self._bla_last_output is not None else 1.0
                        self.e3._committed_trajectory.arousal_tag = float(
                            self._bla_last_output.arousal_tag
                        ) if self._bla_last_output is not None else 0.0
                        self.hippocampal.record_committed_trajectory(
                            self.e3._committed_trajectory
                        )
                # MECH-342: zero the maintenance-release pressure accumulator
                # ONLY on a genuine not-elevated -> elevated transition (legacy
                # mode re-calls elevate() every committed tick, so an
                # unconditional reset here would defeat accumulation). No-op
                # when disabled.
                if (
                    not self.beta_gate.is_elevated
                    and self.maintenance_release is not None
                ):
                    self.maintenance_release.reset_pressure()
                self.beta_gate.elevate()
            else:
                if self.beta_gate.is_elevated:
                    self._committed_step_idx = 0  # reset when gate opens
                    self._committed_anchor_keys = None
                self.beta_gate.release()

        propagated = self.beta_gate.propagate(action)
        if propagated is None:
            # Beta elevated: step through committed trajectory (Layer 1).
            if self.e3._committed_trajectory is not None:
                traj = self.e3._committed_trajectory
                horizon = traj.actions.shape[1]
                step_idx = min(self._committed_step_idx, horizon - 1)
                action = traj.actions[:, step_idx, :]
                self._committed_step_idx += 1
            elif self._last_action is not None:
                action = self._last_action

        # MECH-279: PAG freeze-gate. When freeze_active, the action selector
        # is constrained to a no-op (minimal-movement) action. Entry condition:
        # z_harm_a_norm * duration_above_threshold > theta_freeze. Exit
        # condition: z_harm_a_norm < theta_freeze * gaba_tone (so a higher
        # gaba_tone -- benzo agonist -- raises exit_threshold and accelerates
        # termination, matching the clinical observation that GABA agonists
        # treat freeze catatonia). Backward-compatible no-op when
        # use_pag_freeze_gate=False.
        # MECH-094: simulation_mode=True from hypothesis_tag returns a zeroed
        # output without updating internal counters. select_action() runs on
        # the waking path so hypothesis_tag is False here.
        if self.pag_freeze_gate is not None:
            if (
                self._lpb_last_output is not None
                and getattr(self.config, "use_lpb_interoceptive_routing", False)
            ):
                pag_z_norm = float(self._lpb_last_output.external_magnitude)
            elif z_harm_a is not None:
                pag_z_norm = float(z_harm_a.detach().norm().item())
            else:
                pag_z_norm = 0.0
            # MECH-219 (SD-019b) PAG freeze-drive redirect: source the freeze
            # drive from the slow suffering channel when the accumulator +
            # redirect flag are on. Default off -> bit-identical.
            if (
                getattr(self.config, "use_harm_suffering_accumulator", False)
                and getattr(self.config, "harm_suffering_redirect_pag", False)
                and self._current_latent is not None
                and self._current_latent.z_harm_suffering is not None
            ):
                pag_z_norm = float(
                    self._current_latent.z_harm_suffering.detach().norm().item()
                )
            pag_tone = (
                float(self.gabaergic_decay.gaba_tone)
                if self.gabaergic_decay is not None
                else 1.0
            )
            pag_override = (
                float(self.broadcast_override.override_signal)
                if self.broadcast_override is not None
                else 0.0
            )
            self._pag_last_output = self.pag_freeze_gate.tick(
                z_harm_a_norm=pag_z_norm,
                gaba_tone=pag_tone,
                simulation_mode=False,
                override_signal=pag_override,
            )
            # SD-058 / MECH-357: ilPFC freeze-SUPPRESSION. When the learned/
            # scaffolded avoidance-efficacy x threat is high enough, the
            # infralimbic gate suppresses the MECH-279 freeze no-op so the agent
            # takes its selected instrumental-avoidance action instead of
            # freezing (Moscarello & LeDoux: ilPFC suppresses CeA-driven
            # freezing). Inert when the gate is disabled. Below the threshold
            # the freeze override applies exactly as pre-MECH-357.
            _ia_suppress_freeze = False
            if self.instrumental_avoidance is not None:
                _ia_zha = (
                    self._current_latent.z_harm_a
                    if self._current_latent is not None
                    else None
                )
                _ia_zn = (
                    float(_ia_zha.detach().norm().item())
                    if _ia_zha is not None
                    else 0.0
                )
                _ia_suppress_freeze = self.instrumental_avoidance.should_suppress_freeze(
                    _ia_zn, simulation_mode=False
                )
            if (
                self._pag_last_output.freeze_active
                and action is not None
                and not _ia_suppress_freeze
            ):
                # Constrain action to a no-op one-hot vector. Match the
                # action's shape, dtype, and device. The no-op class index
                # defaults to 0 (configurable via pag_freeze_noop_action_class).
                noop_class = int(
                    getattr(self.config, "pag_freeze_noop_action_class", 0)
                )
                noop = torch.zeros_like(action)
                if noop.dim() == 2:
                    noop_class = max(0, min(noop_class, noop.shape[1] - 1))
                    noop[0, noop_class] = 1.0
                else:
                    noop_class = max(0, min(noop_class, noop.shape[0] - 1))
                    noop[noop_class] = 1.0
                action = noop

        self._cache_tpj_prediction_for_action(action)
        self._last_action = action
        # SD-058 / MECH-357: cache whether the emitted action is directed
        # (non-noop), fed to the eligibility-trace update on the next sense().
        # A freeze (no-op) under threat is NOT credited as avoidance.
        if self.instrumental_avoidance is not None and action is not None:
            _ia_nc = int(self.config.avoidance_noop_class)
            self._ia_last_action_directed = bool(
                int(action.argmax(dim=-1).flatten()[0].item()) != _ia_nc
            )
        # SD-059 / MECH-358: cache the emitted action's first-action class, fed
        # to the bridge eligibility update on the next sense() (relief/safety
        # credit is attributed to this specific directed action class).
        if self.escape_affordance_bridge is not None and action is not None:
            self._eab_last_action_class = int(
                action.argmax(dim=-1).flatten()[0].item()
            )
        # Post-603i trainable learner: cache the emitted action class for
        # action-contingent relief/safety targets on the next sense() tick.
        if self.trainable_escape_affordance_learner is not None and action is not None:
            self._teal_last_action_class = int(
                action.argmax(dim=-1).flatten()[0].item()
            )
        # Post-603i E2 escape-affordance linker: cache the emitted action class,
        # read on the next sense() to pull the detached E2 action-consequence
        # feature for the executed (prev_z_world, action) pair.
        if self.e2_escape_affordance_linker is not None and action is not None:
            self._eal_last_action_class = int(
                action.argmax(dim=-1).flatten()[0].item()
            )
        # MECH-165: record action for exploration trajectory
        self._record_exploration_action(action)

        # MECH-258 / SD-032b: roll E2_harm_a forward for the chosen action, so
        # the next step's dACC tick has a prediction to compute PE against.
        # MECH-269b: gate the z_harm_a input to the forward model. If
        # V_s_z_harm_a falls below the E2 threshold, the forward model
        # consumes the held snapshot of z_harm_a instead of the current
        # _harm_a_prev cache. Held substitution prevents E2_harm_a from
        # rolling forward off a stale-but-confident-looking affective stream.
        if (
            self.e2_harm_a is not None
            and self._harm_a_prev is not None
            and action is not None
        ):
            with torch.no_grad():
                a_in = action if action.dim() > 1 else action.unsqueeze(0)
                z_in = self._harm_a_prev
                if self.vs_rollout_gate is not None:
                    z_in = self.vs_rollout_gate.gate_stream(
                        "z_harm_a",
                        z_in,
                        self.hippocampal.per_stream_vs,
                        side="e2",
                        per_stream_staleness=(
                            self._vs_gate_staleness_cache or None
                        ),
                    )
                if z_in.dim() == 1:
                    z_in = z_in.unsqueeze(0)
                pred = self.e2_harm_a(z_in, a_in)
                self._harm_a_pred_prev = pred.squeeze(0).detach().clone()

        # MECH-260: record chosen action class in dACC recency history.
        if self.dacc is not None and action is not None:
            try:
                a_row = action[0] if action.dim() > 1 else action
                self.dacc.record_action(int(a_row.argmax().item()))
            except Exception:
                # One-hot discretisation fallback: hash raw action. Silent to
                # preserve backward-compatible select_action control flow.
                pass

        # ARC-063: stash the chosen action class so the next tick's
        # CandidateRuleField mint keys the (context-bucket, action-object)
        # regularity on the action the agent actually took. Silent fallback
        # keeps select_action control flow backward-compatible.
        if self.candidate_rule_field is not None and action is not None:
            try:
                _crf_a = action[0] if action.dim() > 1 else action
                self._crf_prev_action_class = int(_crf_a.argmax().item())
            except Exception:
                pass

        # SD-034: run closure completion detector. Stability-based: fires when
        # rule_state (MECH-262) has been flat for N consecutive ticks AND
        # beta is elevated AND current mode is in allowed_closure_modes AND
        # sd_033a write gate is above threshold. Falls through to the five-part
        # fire sequence if all predicates hold.
        if (
            self.closure_operator is not None
            and action is not None
            and self._current_latent is not None
        ):
            try:
                a_row = action[0] if action.dim() > 1 else action
                action_class = int(a_row.argmax().item())
                current_mode = (
                    self.salience.current_mode if self.salience is not None else None
                )
                sd033a_gate = (
                    float(self.salience.write_gate("sd_033a"))
                    if self.salience is not None
                    else None
                )
                _auto_closure_evt = self.closure_operator.tick(
                    current_z_world=self._current_latent.z_world,
                    current_action_class=action_class,
                    current_mode=current_mode,
                    sd033a_gate=sd033a_gate,
                    hypothesis_tag=False,
                )
                # rung-6 amend: an SD-034 closure fire (auto rule-stability detector;
                # _fire installs the de-commit refractory + releases beta) tears down
                # the F-independent closure-plane commit-entry latch.
                if _auto_closure_evt is not None and _auto_closure_evt.fired:
                    self.e3._closure_committed_active = False
                    self.e3._closure_committed_trajectory = None  # C-STEP extension
            except Exception:
                # Closure detector failure must not break action selection.
                pass

        return action

    def _compute_rho_t(self) -> float:
        """ARC-108 JOB-2 (c): the proximity-scaled maintenance drive rho_t.

        rho_t = goal_proximity(z_world) x value, reusing the goal/benefit valuation
        already feeding F (no new substrate): goal_proximity in [0, 1] from
        GoalState (rises approaching the goal, peaks at it, declines past it) x the
        benefit valuation E3.benefit_eval_head(z_world) clamped >= 0. Returns 0.0
        when there is no active goal / latent (the ramp then releases at the floor;
        the falsifier's readiness gate -- not the substrate -- guards a no-proximity-
        variance regime). Pure read; no state mutation.
        """
        if self.goal_state is None or self._current_latent is None:
            return 0.0
        z_world = self._current_latent.z_world
        try:
            prox = float(self.goal_state.goal_proximity(z_world))
        except Exception:
            return 0.0
        try:
            benefit = float(
                self.e3.benefit_eval_head(z_world.detach()).mean().item()
            )
        except Exception:
            benefit = 0.0
        value = max(0.0, benefit)
        return max(0.0, prox) * value

    def notify_env_completion(
        self,
        action_class: Optional[int] = None,
        z_world: Optional[torch.Tensor] = None,
        bypass_mode_conditioning: bool = False,
        simulation_mode: bool = False,
    ):
        """
        SD-034 commitment-closure-control-plane explicit env-completion hook.

        Routes the environment's task-completion event (e.g.
        info["transition_type"] == "sequence_complete") into
        ClosureOperator.emit_closure -- the explicit hook the closure docstring
        describes but the *c cohort left unwired (V3-EXQ-460c got n_closures=0
        despite real env completions because the experiment relied solely on the
        automatic rule_state-stability detector). The caller (experiment harness)
        invokes this right after env.step() on a completion tick, passing the
        just-executed action class.

        Returns the ClosureEvent (so the caller can count fires / No-Go installs)
        or None when the hook is disabled / no operator / simulation. No-op when:
          - use_closure_env_completion_hook is False (default -> bit-identical),
          - self.closure_operator is None,
          - simulation_mode / hypothesis_tag is True (MECH-094: a replay/DMN
            completion must not emit a waking closure done-token).

        z_world defaults to the current latent's z_world (the post-step location
        at which residue discharge is centred) when not supplied.
        """
        if not getattr(self.config, "use_closure_env_completion_hook", False):
            return None
        if self.closure_operator is None:
            return None
        if simulation_mode:
            return None
        zw = z_world
        if zw is None:
            if self._current_latent is None:
                return None
            zw = self._current_latent.z_world
        _env_closure_evt = self.closure_operator.emit_closure(
            action_class=action_class,
            z_world=zw,
            bypass_mode_conditioning=bypass_mode_conditioning,
        )
        # rung-6 amend: a fired env-completion closure (SD-034 _fire installs the
        # de-commit refractory + releases beta) tears down the F-independent
        # closure-plane commit-entry latch.
        if _env_closure_evt is not None and _env_closure_evt.fired:
            self.e3._closure_committed_active = False
        return _env_closure_evt

    def act(
        self,
        observation: torch.Tensor,
        temperature: float = 1.0,
        sequence_in_progress: bool = False,
    ) -> torch.Tensor:
        """
        Complete V3 REE action loop.

        Accepts flat observation — auto-splits into body/world channels.

        Returns action tensor.
        """
        # SENSE + UPDATE
        latent_state = self.sense_flat(observation)

        # Advance clock
        ticks = self.clock.advance()

        # E1 tick
        e1_prior = self._e1_tick(latent_state) if ticks["e1_tick"] else torch.zeros(
            1, self.config.latent.world_dim, device=self.device
        )

        # GENERATE
        candidates = self.generate_trajectories(
            latent_state, e1_prior, ticks,
            sequence_in_progress=sequence_in_progress,
        )

        # SELECT (multi-rate + beta gate)
        action = self.select_action(candidates, ticks, temperature)

        # MECH-092: replay on quiescent E3 cycles
        if ticks.get("e3_quiescent", False):
            self._do_replay(latent_state)

        self._step_count += 1
        return action

    def act_with_split_obs(
        self,
        obs_body: torch.Tensor,
        obs_world: torch.Tensor,
        temperature: float = 1.0,
        sequence_in_progress: bool = False,
    ) -> torch.Tensor:
        """Act with pre-split observations (preferred V3 interface)."""
        latent_state = self.sense(obs_body, obs_world)
        ticks = self.clock.advance()
        e1_prior = self._e1_tick(latent_state) if ticks["e1_tick"] else torch.zeros(
            1, self.config.latent.world_dim, device=self.device
        )
        candidates = self.generate_trajectories(
            latent_state, e1_prior, ticks,
            sequence_in_progress=sequence_in_progress,
        )
        action = self.select_action(candidates, ticks, temperature)
        if ticks.get("e3_quiescent", False):
            self._do_replay(latent_state)
        self._step_count += 1
        return action

    def act_with_log_prob(
        self,
        observation: torch.Tensor,
        temperature: float = 1.0,
        sequence_in_progress: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Action selection + log-prob for REINFORCE."""
        latent_state = self.sense_flat(observation)
        ticks = self.clock.advance()
        e1_prior = self._e1_tick(latent_state) if ticks["e1_tick"] else torch.zeros(
            1, self.config.latent.world_dim, device=self.device
        )
        candidates = self.generate_trajectories(
            latent_state, e1_prior, ticks,
            sequence_in_progress=sequence_in_progress,
        )
        z_harm_a = None
        if self._current_latent is not None and self._current_latent.z_harm_a is not None:
            z_harm_a = self._current_latent.z_harm_a
        result = self.e3.select(candidates, temperature, z_harm_a=z_harm_a)
        self._cache_tpj_prediction_for_action(result.selected_action)
        self._last_action = result.selected_action
        # MECH-165: record action for exploration trajectory
        self._record_exploration_action(result.selected_action)
        self._step_count += 1
        return result.selected_action, result.log_prob

    def _do_replay(self, latent_state: LatentState) -> None:
        """
        MECH-092: SWR-equivalent replay on quiescent E3 cycles.

        All replay content carries hypothesis_tag=True — cannot produce residue.
        MECH-203: when serotonin is enabled, passes drive_state to replay()
        for valence-weighted start point selection (balanced consolidation).
        """
        recent = self.theta_buffer.recent
        if recent is None:
            return
        # MECH-203 + MECH-205: build drive_state for valence-weighted replay
        drive_state = None
        if self.serotonin.enabled or self.config.surprise_gated_replay:
            # Drive state = [wanting_weight, liking_weight, harm_weight, surprise_weight]
            t5ht = self.serotonin.tonic_5ht if self.serotonin.enabled else 0.0
            # MECH-205: surprise weight scales with recent PE magnitude
            surprise_weight = 0.3
            if self.config.surprise_gated_replay and self._pe_ema > 0:
                surprise_weight = min(1.0, self._pe_ema * 5.0)
            drive_state = torch.tensor(
                [t5ht, 0.5, 1.0 - t5ht, surprise_weight],
                device=self.device,
            )
        # MECH-165: use diverse replay scheduler when enabled
        if self.config.replay_diversity_enabled:
            retrieval_bias = (
                self._bla_last_output.retrieval_bias
                if self._bla_last_output is not None
                else None
            )
            replay_trajs = self.hippocampal.diverse_replay(
                recent,
                drive_state=drive_state,
                mode="auto",
                retrieval_bias=retrieval_bias,
            )
        else:
            replay_trajs = self.hippocampal.replay(recent, drive_state=drive_state)
        # hypothesis_tag=True: these trajectories cannot update residue
        # (MECH-094 -- enforced in ResidueField.accumulate)

    def update_residue(
        self,
        harm_signal: float,
        world_delta: Optional[float] = None,
        hypothesis_tag: bool = False,
        owned: bool = True,
    ) -> Dict[str, Any]:
        """
        RESIDUE step: accumulate residue at z_world location after action.

        V3 changes:
        - Residue accumulates on z_world (SD-005), not z_gamma
        - world_delta from SD-003 attribution pipeline scales accumulation
        - hypothesis_tag=True blocks accumulation (MECH-094)

        Args:
            harm_signal:   Harm from environment (negative = harm)
            world_delta:   Optional |z_world_actual - z_world_cf| from SD-003
            hypothesis_tag: If True, no residue (MECH-094)
            owned:         If False, no residue (not agent-attributed harm)
        """
        metrics: Dict[str, Any] = {}

        # ARC-016: update running variance on EVERY step so rv tracks world
        # prediction error continuously.  Previously gated on harm_signal < 0,
        # which meant rv only updated on harm steps -- too sparse for
        # volatility tracking (Q-007) and caused rv deadlock.
        if self._current_latent is not None:
            z_world = self._current_latent.z_world
            e3_metrics = self.e3.post_action_update(
                actual_z_world=z_world,
                harm_occurred=(harm_signal < 0),
            )
            metrics.update({f"e3_{k}": v for k, v in e3_metrics.items()})

            # ARC-108 JOB-2 (d): HABENULA negative-RPE de-commit. post_action_update
            # surfaced the signed RPE delta_t (= R_t - V-hat_t, the SAME signal JOB-1
            # uses) as e3_metrics["habenula_delta_t"] when use_habenula_decommit is on.
            # Route it into the SD-034 ClosureOperator's habenula abort: a negative
            # ("worse than expected") delta_t fires a content-driven de-commit,
            # dissociable from the latch's refractory state. The operator no-ops when
            # the abort is disabled / beta not elevated / delta_t above threshold, so
            # this is bit-identical OFF. Waking-only: update_residue is the waking
            # post-action path; hypothesis_tag passed through (MECH-094).
            if (
                getattr(self.config, "use_habenula_decommit", False)
                and self.closure_operator is not None
            ):
                _hab_delta = e3_metrics.get("habenula_delta_t")
                if _hab_delta is not None:
                    _hab_action_class = None
                    if self._last_action is not None:
                        try:
                            _ha = (
                                self._last_action[0]
                                if self._last_action.dim() > 1
                                else self._last_action
                            )
                            _hab_action_class = int(_ha.argmax().item())
                        except Exception:
                            _hab_action_class = None
                    _hab_event = self.closure_operator.habenula_tick(
                        delta_t=float(_hab_delta.item()),
                        z_world=z_world,
                        action_class=_hab_action_class,
                        hypothesis_tag=hypothesis_tag,
                    )
                    if _hab_event.fired:
                        # Mirror the SD-034 de-commit cleanup the other release sites
                        # perform so the de-committed program is fully torn down.
                        self._committed_step_idx = 0
                        self._committed_anchor_keys = None
                        self.e3._committed_trajectory = None
                        self.e3._closure_committed_active = False  # rung-6 amend: clear latch
                        self._ncl_hold_active = False
                        metrics["habenula_decommit_fired"] = torch.tensor(1.0)

            # MECH-205: populate VALENCE_SURPRISE on residue field
            if self.config.surprise_gated_replay:
                pe_val = e3_metrics.get("prediction_error")
                if pe_val is not None:
                    pe_mag = float(pe_val.detach())
                    self._pe_ema = (1 - self._pe_ema_alpha) * self._pe_ema + self._pe_ema_alpha * pe_mag
                    surprise = max(0.0, pe_mag - self._pe_ema)
                    # Gate: only write genuine surprises above threshold
                    if surprise > self.config.pe_surprise_threshold:
                        # MECH-307 Gap 1 write-site dispatch (2026-05-11 user-
                        # override landing). Three paths, in priority order:
                        # (1) use_mech307_split_surprise (Option b, 2026-05-11):
                        #     route surprise to VALENCE_POSITIVE_SURPRISE or
                        #     VALENCE_NEGATIVE_SURPRISE based on harm_signal
                        #     sign. ALSO write magnitude to legacy
                        #     VALENCE_SURPRISE so MECH-205 / SD-014 consumers
                        #     reading the magnitude slot stay bit-identical.
                        # (2) use_mech307_signed_pe (Option a, 2026-05-08 legacy):
                        #     signed single-channel write -- harm-paired
                        #     surprise stored as negative, non-harm as positive,
                        #     into VALENCE_SURPRISE. Magnitude-readers get the
                        #     correct magnitude via |signed_surprise| = surprise.
                        # (3) Both flags False (true legacy): unsigned magnitude
                        #     into VALENCE_SURPRISE.
                        if getattr(self.config, "use_mech307_split_surprise", False):
                            target_channel = (
                                VALENCE_NEGATIVE_SURPRISE
                                if harm_signal < 0
                                else VALENCE_POSITIVE_SURPRISE
                            )
                            self.residue_field.update_valence(
                                z_world, target_channel, surprise,
                                hypothesis_tag=False,
                            )
                            # Backward-compat magnitude write to legacy slot.
                            self.residue_field.update_valence(
                                z_world, VALENCE_SURPRISE, surprise,
                                hypothesis_tag=False,
                            )
                        elif getattr(self.config, "use_mech307_signed_pe", False):
                            signed_surprise = (
                                -surprise if harm_signal < 0 else surprise
                            )
                            self.residue_field.update_valence(
                                z_world, VALENCE_SURPRISE, signed_surprise,
                                hypothesis_tag=False,
                            )
                        else:
                            self.residue_field.update_valence(
                                z_world, VALENCE_SURPRISE, surprise,
                                hypothesis_tag=False,
                            )
                        self._surprise_write_count += 1
                    metrics["mech205_pe_mag"] = pe_mag
                    metrics["mech205_pe_ema"] = self._pe_ema
                    metrics["mech205_surprise"] = surprise
                    metrics["mech205_write_count"] = self._surprise_write_count

        if harm_signal < 0:
            harm_magnitude = abs(harm_signal)
            self._harm_this_episode += harm_magnitude

            if owned and not hypothesis_tag and self._current_latent is not None:
                z_world = self._current_latent.z_world
                residue_metrics = self.residue_field.accumulate(
                    z_world,
                    harm_magnitude=harm_magnitude,
                    world_delta=world_delta,
                    hypothesis_tag=hypothesis_tag,
                )
                metrics.update({f"residue_{k}": v for k, v in residue_metrics.items()})

                # MECH-091: harm is salient -> phase reset
                self.clock.phase_reset()

        metrics["harm_signal"] = harm_signal
        metrics["harm_this_episode"] = self._harm_this_episode
        return metrics

    def record_transition(
        self,
        z_self_t: torch.Tensor,
        action: torch.Tensor,
        z_self_t1: torch.Tensor,
    ) -> None:
        """Record (z_self_t, a_t, z_self_{t+1}) for E2 motor-sensory training."""
        # Detach all three: replay buffer entries must not retain computation graphs.
        # action from E3 select() is a slice (AsStrided) of trajectory tensors that
        # may have E3 parameter grad_fns. Without detach, compute_e2_loss() backward()
        # propagates through E3's graph and version-increments E3 tensors mid-backward,
        # causing "modified by an inplace operation" RuntimeError on the next call.
        self._e2_transition_buffer.append((
            z_self_t.detach().clone(),
            action.detach().clone(),
            z_self_t1.detach().clone(),
        ))
        if len(self._e2_transition_buffer) > 1000:
            self._e2_transition_buffer = self._e2_transition_buffer[-1000:]

    def compute_prediction_loss(self) -> torch.Tensor:
        """
        E1 world-model prediction loss for training.

        Samples from experience buffer; restores hidden state after.
        """
        zero_loss = next(self.e1.parameters()).sum() * 0.0
        if len(self._world_experience_buffer) < 2:
            return zero_loss

        buf_len = len(self._world_experience_buffer)
        horizon = self.e1.config.prediction_horizon
        max_start = max(1, buf_len - 1)
        start_idx = int(torch.randint(0, max_start, (1,)).item())
        end_idx = min(start_idx + horizon + 1, buf_len)

        if end_idx - start_idx < 2:
            return zero_loss

        # Build [z_self, z_world] sequence for E1
        self_seq  = self._self_experience_buffer[start_idx:end_idx]
        world_seq = self._world_experience_buffer[start_idx:end_idx]
        combined = [torch.cat([s.squeeze(0), w.squeeze(0)]) for s, w in zip(self_seq, world_seq)]
        sequence = torch.stack(combined).unsqueeze(0)  # [1, seq_len, total_dim]

        saved_hidden = self.e1._hidden_state
        self.e1.reset_hidden_state()

        initial = sequence[:, 0, :]
        horizon_len = sequence.shape[1] - 1
        predictions = self.e1.predict_long_horizon(initial, horizon=horizon_len)
        targets = sequence[:, 1:, :]
        loss = F.mse_loss(predictions[:, :targets.shape[1], :], targets)

        # SD-016 Part B1 (2026-04-25, EXQ-477 follow-up):
        # Training-time ContextMemory write hook. When sd016_writepath_mode is
        # "train_only" or "both", the (initial-obs, prediction-error) pair is
        # routed to E1.update_from_observation(), which writes through to
        # ContextMemory.write under the _offline_mode guard. Detached so this
        # cannot push gradient back into the encoders or the LSTM.
        # Gated on E1Config.sd016_writepath_mode -- default "off" preserves
        # bit-identical legacy behaviour.
        # See REE_assembly/docs/architecture/context_memory_writepath_fix.md.
        _wp_mode = getattr(self.e1.config, "sd016_writepath_mode", "off")
        if _wp_mode in ("train_only", "both"):
            obs_state = initial.detach()
            pred_err = (predictions[:, :targets.shape[1], :] - targets).detach()
            self.e1.update_from_observation(obs_state, pred_err)

        # SD-016 Path 1 (2026-04-25, V3-EXQ-418e):
        # Auxiliary diversification loss on ContextMemory slots. Adds gradient
        # pressure that pushes slot vectors toward mutual orthogonality on the
        # unit sphere. EXQ-418d showed v2 read+write gradients alone cannot
        # break slot symmetry (entropy=ln(16) regardless of write mode); this
        # loss term is the smallest substrate change that introduces explicit
        # symmetry-breaking pressure. Gated on weight > 0 AND sd016_enabled;
        # default 0.0 preserves bit-identical legacy behaviour.
        # See REE_assembly/docs/architecture/sd_016_writepath_v3_diversification_loss.md.
        _div_w = getattr(self.config, "sd016_diversification_weight", 0.0)
        if _div_w > 0.0 and getattr(self.e1.config, "sd016_enabled", False):
            loss = loss + _div_w * self.e1.context_memory.compute_diversification_loss()

        self.e1._hidden_state = saved_hidden
        return loss

    def compute_benefit_eval_loss(
        self,
        benefit_exposure: torch.Tensor,
    ) -> torch.Tensor:
        """
        ARC-030 / MECH-112: Train benefit_eval_head from benefit_exposure signal.

        benefit_exposure = body_state[11] (CausalGridWorldV2 use_proxy_fields=True).
        Supervises E3.benefit_eval_head to predict resource proximity.
        Gradient flows through E3's benefit_eval_head only.

        Args:
            benefit_exposure: [batch, 1] or scalar tensor in [0, 1]

        Returns:
            MSE loss scalar (zero if current_latent is not set)
        """
        zero_loss = next(self.e3.parameters()).sum() * 0.0
        if self._current_latent is None:
            return zero_loss
        z_world = self._current_latent.z_world.detach()
        benefit_pred = self.e3.benefit_eval(z_world)  # [batch, 1]
        target = benefit_exposure.to(z_world.device).float()
        if target.dim() == 0:
            target = target.unsqueeze(0).unsqueeze(0)
        elif target.dim() == 1:
            target = target.unsqueeze(-1)
        return F.mse_loss(benefit_pred, target.expand_as(benefit_pred))

    # SD-009 / MECH-100: event label mapping.
    # obs_t is labeled with the transition_type that PRODUCED obs_t --
    # pass info_{t-1}["transition_type"] from the previous env.step(), not the current one.
    _EVENT_LABEL_MAP = {"none": 0, "env_caused_hazard": 1, "agent_caused_hazard": 2}

    def compute_event_contrastive_loss(
        self,
        transition_type: str,
        latent_state: "LatentState",
    ) -> torch.Tensor:
        """
        SD-009 / MECH-100: Event contrastive CE loss on z_world encoder.

        Trains the SplitEncoder event_classifier head to distinguish transition
        types (none / env_caused_hazard / agent_caused_hazard) from z_world,
        forcing the world encoder to represent harm-relevance distinctions.
        Without this, E1/E2/reconstruction losses are invariant to event type
        and z_world converges to a harm-agnostic representation.

        Requires use_event_classifier=True in LatentStackConfig. Returns zero
        otherwise (backward-compatible with all prior experiments).

        IMPORTANT: pass the LatentState returned directly by sense(), NOT
        agent._current_latent (which is detached). Gradient must flow from the
        CE loss back through the world encoder. Usage:

            latent = agent.sense(obs_body, obs_world)
            # ... end of step t ... record transition_type from info ...
            # at step t+1, label the t+1 encoding with t's transition_type:
            latent_next = agent.sense(obs_body_next, obs_world_next)
            loss += lambda_event * agent.compute_event_contrastive_loss(
                prev_ttype, latent_next)

        Labeling convention (from EXQ-020): obs_t is labeled with the transition
        that PRODUCED obs_t. Pass info_{t-1}["transition_type"], not the current
        step's info. On the first step (no prev_ttype), skip or pass "none".

        Args:
            transition_type: string from env info dict, one of
                "none", "env_caused_hazard", "agent_caused_hazard".
                Unknown strings map to label 0 (none).
            latent_state: LatentState from sense() with retained gradients.

        Returns:
            CE loss scalar. Gradient flows through latent_stack encoder.
        """
        zero_loss = next(self.latent_stack.parameters()).sum() * 0.0
        if not getattr(self.config.latent, "use_event_classifier", False):
            return zero_loss
        if latent_state.event_logits is None:
            return zero_loss
        label = self._EVENT_LABEL_MAP.get(transition_type, 0)
        label_t = torch.tensor([label], dtype=torch.long,
                               device=latent_state.event_logits.device)
        # event_logits: [batch, 3]; F.cross_entropy expects [N, C]
        logits = latent_state.event_logits
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        return F.cross_entropy(logits, label_t)

    def compute_resource_proximity_loss(
        self,
        resource_proximity_target: float,
        latent_state: "LatentState",
    ) -> torch.Tensor:
        """
        SD-018: Resource proximity regression loss on z_world encoder.

        Trains the SplitEncoder resource_proximity_head to predict
        max(resource_field_view) from z_world, forcing the world encoder to
        represent resource proximity. Without this, benefit_eval_head(z_world)
        produces R2=-0.004 (EXQ-085m) because E1/reconstruction losses are
        invariant to resource saliency.

        Requires use_resource_proximity_head=True in LatentStackConfig. Returns
        zero otherwise (backward-compatible with all prior experiments).

        IMPORTANT: pass the LatentState returned directly by sense(), NOT
        agent._current_latent (which is detached). Gradient must flow from the
        MSE loss back through the world encoder.

        Labeling: target is max(resource_field_view) from the observation that
        produced this LatentState. The environment provides this in
        obs_dict["resource_field_view"] when use_proxy_fields=True.

        Args:
            resource_proximity_target: float in [0, 1], the peak resource
                proximity from the agent's current 5x5 view.
            latent_state: LatentState from sense() with retained gradients.

        Returns:
            MSE loss scalar. Gradient flows through latent_stack encoder.
        """
        zero_loss = next(self.latent_stack.parameters()).sum() * 0.0
        if not getattr(self.config.latent, "use_resource_proximity_head", False):
            return zero_loss
        if latent_state.resource_prox_pred is None:
            return zero_loss
        pred = latent_state.resource_prox_pred  # [batch, 1]
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        target = torch.tensor(
            [[resource_proximity_target]],
            dtype=torch.float32,
            device=pred.device,
        )
        return F.mse_loss(pred, target)

    def compute_resource_encoder_loss(
        self,
        resource_proximity_target: float,
        latent_state: "LatentState",
    ) -> torch.Tensor:
        """
        SD-015 / MECH-112: auxiliary loss for ResourceEncoder object-type supervision.

        Trains the ResourceEncoder's resource_prox_head to predict resource proximity
        from z_resource. Same supervision signal as SD-018 (max(resource_field_view)),
        but backpropagates through the separate ResourceEncoder rather than through
        z_world. This forces z_resource to represent object-type features (what is
        present) rather than spatial position.

        Requires use_resource_encoder=True in LatentStackConfig. Returns zero
        otherwise (backward-compatible with all prior experiments).

        Args:
            resource_proximity_target: scalar in [0, 1]; typically max(resource_field_view)
                from the environment observation.
            latent_state: current LatentState (from sense()); must contain resource_prox_pred_r.

        Returns:
            MSE loss scalar (zero when ResourceEncoder is disabled).
        """
        zero_loss = next(self.latent_stack.parameters()).sum() * 0.0
        if not getattr(self.config.latent, "use_resource_encoder", False):
            return zero_loss
        if latent_state.resource_prox_pred_r is None:
            return zero_loss
        pred = latent_state.resource_prox_pred_r  # [batch, 1]
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        target = torch.tensor(
            [[resource_proximity_target]],
            dtype=torch.float32,
            device=pred.device,
        )
        return F.mse_loss(pred, target)

    def compute_resource_identity_loss(
        self,
        resource_type_target: int,
        latent_state: "LatentState",
    ) -> torch.Tensor:
        """
        SD-049 Phase 2 hybrid identity-classifier auxiliary loss (Option C).

        Trains the ResourceEncoder identity_head to predict the resource type
        at the agent's current cell from z_resource via cross-entropy. Target
        is obs_dict["resource_type_at_agent"] from the SD-049 multi-resource
        substrate (0 = no-resource-at-agent / no-supervision; type_idx + 1 for
        type slot 0..n_resource_types-1).

        Backprop through identity_head + trunk encoder during P0 phased
        training; freeze identity_head.requires_grad_(False) at P1 to allow
        the trunk to develop similarity structure beyond what classifier
        supervision alone provides (Schapiro 2016/2017 distributed substrate
        per the lit-pull verdict).

        Lit-pull provenance: REE_assembly/evidence/literature/
        targeted_review_sd_049_encoder_identity_expansion/verdict.md
        (Option C hybrid, confidence 0.78).

        Requires use_resource_encoder=True AND use_identity_classifier=True
        in LatentStackConfig. Returns zero otherwise (backward-compatible).
        Returns zero when target is 0 (no-resource-at-agent; no supervision).

        Args:
            resource_type_target: int in [0, n_resource_types]; 0 means
                no-resource-at-agent (skip supervision); type_idx+1 for valid
                contact. This matches the obs_dict["resource_type_at_agent"]
                emission convention from causal_grid_world.py.
            latent_state: current LatentState (from sense()); must contain
                identity_logits.

        Returns:
            Cross-entropy loss scalar (zero when classifier disabled, when
            identity_logits is None, or when target is 0 / no-resource).
        """
        zero_loss = next(self.latent_stack.parameters()).sum() * 0.0
        if not getattr(self.config.latent, "use_identity_classifier", False):
            return zero_loss
        if latent_state.identity_logits is None:
            return zero_loss
        # Target convention: obs_dict["resource_type_at_agent"] is type_idx+1
        # (1..n_types) when agent is on a resource cell, 0 otherwise. Skip
        # supervision on the 0 case -- there is no ground-truth identity to
        # supervise on when the agent is on an empty cell.
        target_int = int(resource_type_target)
        if target_int <= 0:
            return zero_loss
        # Convert to 0-indexed type id for cross-entropy.
        type_idx = target_int - 1
        logits = latent_state.identity_logits  # [batch, n_resource_types]
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        n_types = logits.shape[-1]
        if type_idx < 0 or type_idx >= n_types:
            # Out-of-range target -- skip silently (defensive against
            # n_resource_types config mismatch with env).
            return zero_loss
        target = torch.tensor([type_idx], dtype=torch.long, device=logits.device)
        # Expand target if logits has batch > 1.
        if logits.shape[0] > 1:
            target = target.expand(logits.shape[0])
        return F.cross_entropy(logits, target)

    def compute_harm_nonredundancy_loss(
        self,
        latent_state: "LatentState",
    ) -> torch.Tensor:
        """
        SD-019: Affective harm non-redundancy constraint loss.

        Penalises cosine alignment between z_harm_s and z_harm_a using a squared
        cosine similarity penalty. This enforces that the two harm streams encode
        non-redundant information, consistent with the A-delta/C-fiber distinction:
        sensory-discriminative (z_harm_s) encodes immediate proximity/intensity,
        affective-motivational (z_harm_a) encodes accumulated threat burden with
        different temporal scope and persistence.

        Penalty = cos_sim(z_harm_s, z_harm_a)^2
        - Penalises both positive AND negative alignment (symmetric).
        - Zero gradient when streams are orthogonal.
        - Large gradient when streams are parallel (redundant).

        ARC-016 coupling: when harm_nonredundancy_precision_scale > 0, the penalty
        is scaled by normalised E3 precision, enforcing non-redundancy more strongly
        in high-confidence (low-variance) states. Biological grounding: Barlow 1961
        redundancy reduction; C-fiber/A-delta distinct temporal integration profiles.

        Returns zero when harm_nonredundancy_weight=0.0 (default, backward compat)
        or when either harm stream is absent from the current latent state.

        Args:
            latent_state: current LatentState (from sense()); must contain z_harm and z_harm_a.

        Returns:
            Penalty loss scalar (>= 0).
        """
        zero_loss = next(self.latent_stack.parameters()).sum() * 0.0
        weight = getattr(self.config, "harm_nonredundancy_weight", 0.0)
        if weight <= 0.0:
            return zero_loss
        if latent_state.z_harm is None or latent_state.z_harm_a is None:
            return zero_loss
        z_s = latent_state.z_harm    # [batch, harm_dim]
        z_a = latent_state.z_harm_a  # [batch, z_harm_a_dim]
        # Project to common dim if needed (use the smaller)
        min_dim = min(z_s.shape[-1], z_a.shape[-1])
        z_s_proj = z_s[..., :min_dim]
        z_a_proj = z_a[..., :min_dim]
        cos_sim = F.cosine_similarity(z_s_proj, z_a_proj, dim=-1).mean()
        penalty = cos_sim.pow(2)
        # ARC-016 coupling: scale by normalised precision when enabled
        prec_scale = getattr(self.config, "harm_nonredundancy_precision_scale", 0.0)
        if prec_scale > 0.0:
            precision_norm = min(self.e3.current_precision / 500.0, 2.0)
            penalty = penalty * (1.0 + prec_scale * precision_norm)
        return weight * penalty

    def compute_harm_accum_loss(
        self,
        accumulated_harm_target: float,
        latent_state: "LatentState",
    ) -> torch.Tensor:
        """
        SD-011 second source: auxiliary loss for harm accumulation prediction.

        Trains the AffectiveHarmEncoder harm_accum_head to predict accumulated
        harm exposure from z_harm_a, forcing the affective encoder to integrate
        temporal harm information that z_harm_s does not receive. This creates
        gradient pressure for genuine stream divergence (resolving EXQ-241 D3
        reversal where z_harm_a was monotonically redundant with z_harm_s).

        Requires harm_history_len > 0 in LatentStackConfig. Returns zero
        otherwise (backward-compatible with all prior experiments).

        IMPORTANT: pass the LatentState returned directly by sense(), NOT
        agent._current_latent (which is detached). Gradient must flow from the
        MSE loss back through the affective harm encoder.

        Args:
            accumulated_harm_target: float in [0, 1], running average of
                harm_exposure over the episode. From obs_dict["accumulated_harm"].
            latent_state: LatentState from sense() with retained gradients.

        Returns:
            Weighted MSE loss scalar. Weight = z_harm_a_aux_loss_weight.
        """
        zero_loss = next(self.latent_stack.parameters()).sum() * 0.0
        if getattr(self.config.latent, "harm_history_len", 0) <= 0:
            return zero_loss
        if latent_state.harm_accum_pred is None:
            return zero_loss
        pred = latent_state.harm_accum_pred  # [batch, 1]
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)

        # SD-020: optional precision-weighted PE target for z_harm_a training.
        # When harm_surprise_pe_enabled=True, replace EMA accumulated_harm_target
        # with |actual_harm - expected_harm| * precision_norm (affective surprise).
        # Chen 2023: AIC encodes unsigned intensity prediction errors, not raw magnitude.
        # precision_norm = min(E3.current_precision / 500, 3.0) provides ARC-016 coupling.
        harm_surprise_pe_enabled = getattr(self.config, "harm_surprise_pe_enabled", False)
        if harm_surprise_pe_enabled:
            alpha = getattr(self.config, "harm_obs_ema_alpha", 0.1)
            # Update running expected harm
            self._harm_obs_ema = (1.0 - alpha) * self._harm_obs_ema + alpha * accumulated_harm_target
            # PE = |actual - expected|
            harm_pe = abs(accumulated_harm_target - self._harm_obs_ema)
            # ARC-016: scale by normalised precision
            precision_norm = min(self.e3.current_precision / 500.0, 3.0)
            surprise_target = harm_pe * precision_norm
            target_val = surprise_target
        else:
            target_val = accumulated_harm_target

        target = torch.tensor(
            [[target_val]],
            dtype=torch.float32,
            device=pred.device,
        )
        weight = getattr(self.config.latent, "z_harm_a_aux_loss_weight", 0.1)
        return weight * F.mse_loss(pred, target)

    def update_z_goal(
        self,
        benefit_exposure: float,
        drive_level: float = 1.0,
        resource_type: Optional[int] = None,
    ) -> None:
        """Update z_goal from benefit signal (MECH-112 wanting update).

        SD-015: when ResourceEncoder is enabled and z_resource is populated in the
        current latent state, seeds z_goal from z_resource (object-type latent) rather
        than z_world (full scene latent). z_resource encodes "what kind of resource is
        present" independent of spatial position -- resources respawn at random locations,
        so z_world at contact has no predictive value for future resource locations.

        SD-057 (GAP-7 L2-L3-L4): when GoalState.use_incentive_token_bank is set,
        the benefit pulse binds to the SD-049 resource-type tag (L2 MECH-344), each
        type accrues a slow-decay revaluable incentive token (L3 MECH-345), and the
        z_goal seed is sourced from the MOST-WANTED object's stored embedding
        (L4 MECH-346; argmax over base_value x per-axis-drive) rather than the raw
        last-contacted z_resource. The GoalState.update firing gate is unchanged --
        only the seed SOURCE changes.

        Args:
            benefit_exposure: scalar benefit this step (obs_body[11] in proxy mode)
            drive_level: homeostatic drive 0=sated, 1=depleted (SD-012).
                         Compute as: 1.0 - float(obs_body[0, 3]) where obs_body[3]=energy.
            resource_type: SD-049 per-type identity tag of the contacted resource
                         (1..n_resource_types; 0 = none). Optional -- supplied by
                         callers from obs_dict["resource_type_at_agent"] /
                         info["sd049_consumed_type_tag_this_tick"]. Only consumed by
                         the SD-057 incentive bank; ignored (legacy path) otherwise.
        """
        if self.goal_state is None or self._current_latent is None:
            return

        # MECH-189 READ (adult z_goal seeding readout): before the within-episode
        # benefit update, when the live z_goal is below the seed floor (the agent
        # has no strong episodic goal of its own yet), seed it toward the
        # childhood-formed super-ordinal anchor whose stored context best matches
        # the current z_world. This is the "stored z_goal anchors bias z_goal
        # seeding in adult episodes even in novel contexts" readout. cue_pull
        # raises z_goal without a benefit pulse (GoalState.cue_pull). No-op when
        # the store is empty / no anchor matches above threshold.
        som = self.super_ordinal_goal_memory
        if (
            som is not None
            and som.n_occupied() > 0
            and self.goal_state.goal_norm()
            < self.config.goal.super_ordinal_seed_below_norm
        ):
            retrieved = som.retrieve(self._current_latent.z_world)
            if (
                retrieved is not None
                and retrieved[1]
                >= self.config.goal.super_ordinal_seed_match_threshold
            ):
                self.goal_state.cue_pull(
                    retrieved[0],
                    self.config.goal.super_ordinal_seed_strength,
                )
                som.note_seed()

        # SD-015: use z_resource if available (object-type seeding), else z_world
        use_resource = (
            getattr(self.config.latent, "use_resource_encoder", False)
            and self._current_latent.z_resource is not None
        )
        seed_latent = (
            self._current_latent.z_resource if use_resource
            else self._current_latent.z_world
        )
        # SD-032: cache base drive_level on goal_state so AIC / PCC / pACC /
        # salience consumers can read a live value via getattr. Convention:
        # stored value is the BASE drive_level (no pACC bias); SD-032e
        # consumers apply pacc.effective_drive() themselves to avoid double-
        # counting.
        self.goal_state._last_drive_level = float(drive_level)
        # SD-032e: the wanting gain scaling inside GoalState.update() should
        # see the SENSITISED drive regime. Sustained affective pain ->
        # elevated drive_bias -> stronger wanting pull toward resources.
        effective_drive = drive_level
        if self.pacc is not None:
            effective_drive = self.pacc.effective_drive(drive_level)
        # SD-037: broadcast-override gate on drive -> z_goal seeding. When the
        # override regulator is recruited (override_signal in [0, 1]), amplify
        # the effective drive by up to override_goal_seeding_gain at full
        # recruitment. override=0 -> no amplification (legacy SD-012 path).
        # override=1 -> drive is multiplied by override_goal_seeding_gain.
        # Result is clipped to [0, 1] (drive_level domain).
        if self.broadcast_override is not None:
            override_signal = float(self.broadcast_override.override_signal)
            seeding_gain = float(
                getattr(self.config, "override_goal_seeding_gain", 2.0)
            )
            multiplier = 1.0 + (seeding_gain - 1.0) * override_signal
            effective_drive = max(0.0, min(1.0, effective_drive * multiplier))
        # SD-057 (GAP-7 L2-L3-L4): object-bound incentive-salience layer.
        # When the bank is active, bind the benefit pulse to object identity,
        # revalue the per-object token, and source the z_goal seed from the
        # MOST-WANTED object's stored embedding (L4 MECH-346) instead of the
        # raw last-contacted z_resource. The GoalState firing gate below is
        # unchanged; only seed_latent is redirected.
        bank = getattr(self.goal_state, "incentive_bank", None)
        if bank is not None:
            bank.decay()  # L3 slow decay of all per-object tokens this tick
            # L2 bind: only when a resource type was actually contacted and the
            # object-type embedding (z_resource) is available to store.
            if (
                resource_type is not None
                and int(resource_type) > 0
                and use_resource
                and self._current_latent.z_resource is not None
            ):
                bank.update(
                    int(resource_type),
                    benefit_exposure,
                    self._current_latent.z_resource,
                )
            # L4 pointer: seed z_goal FROM the most-wanted object's embedding.
            # Read the cached per-axis drive directly (NOT the SD-049-cascade-
            # gated helper) so SD-057 drive-specific wanting works whenever the
            # caller passes obs_per_axis_drive, independent of the separate
            # SD-049 consumer-cascade flag.
            mw = bank.most_wanted(
                per_axis_drive=getattr(self, "_per_axis_drive", None),
                scalar_drive=effective_drive,
            )
            if mw is not None:
                seed_latent = mw[1]
        self.goal_state.update(
            seed_latent,
            benefit_exposure,
            drive_level=effective_drive,
        )

        # MECH-295: anticipatory liking-stream write at the goal location.
        # Bridge fires after GoalState.update() so z_goal reflects this
        # tick's seeding. Write magnitude = drive_to_liking_gain *
        # effective_drive * z_goal_norm; gated by min_drive_to_fire and
        # min_z_goal_norm_to_fire. Write target is the current goal latent
        # (z_goal), not the agent's current location -- this is the
        # anticipatory cue-side pulse, distinct from the consummatory
        # write in update_liking().
        # MECH-094: hypothesis_tag=False (waking write); the bridge's
        # internal simulation_mode argument keeps consistency with the
        # MECH-094 substrate-write convention used by the residue field.
        if (
            self.mech295_bridge is not None
            and self.goal_state is not None
            and self.goal_state.is_active()
            and hasattr(self.residue_field, "update_valence")
        ):
            z_goal_norm = self.goal_state.goal_norm()
            write_value = self.mech295_bridge.compute_anticipatory_liking_write(
                drive_level=effective_drive,
                z_goal_norm=z_goal_norm,
                simulation_mode=False,
                # SD-049 Phase 3: axis-matched liking write at the goal
                # location. When _current_goal_axis_idx is set by an
                # experiment caller (e.g. via update_z_goal_with_axis), the
                # write uses per_axis_drive[axis_idx]. Default fallback is
                # the per-consumer combiner (max).
                per_axis_drive=self._per_axis_drive_for_consumers(),
                goal_axis_idx=getattr(self, "_current_goal_axis_idx", None),
                per_axis_combiner=getattr(
                    self.config, "sd049_mech295_per_axis_combiner", "max"
                ),
            )
            if write_value > 0.0:
                # Write at the goal location (z_goal latent), not the agent's
                # current z_world. update_valence accepts a [batch, world_dim]
                # tensor; goal_state.z_goal already has shape [1, world_dim].
                self.residue_field.update_valence(
                    self.goal_state.z_goal,
                    component=VALENCE_LIKING,
                    value=write_value,
                    hypothesis_tag=False,
                )

        # MECH-189 WRITE (child-phase super-ordinal anchor formation): after the
        # within-episode update, write the current z_goal as a persistent
        # super-ordinal anchor keyed on the current z_world context, gated on the
        # MECH-189 conjunction -- (a) a high-salience benefit contact AND (b) a
        # high-contextual-complexity context -- and only while writes are enabled
        # (the child phase; the curriculum freezes writes for adult measurement
        # via set_super_ordinal_write_enabled). salience = drive-modulated
        # benefit (the "large benefit_exposure spike"); the threshold + the
        # complexity mode/threshold are the DEV-NEED-024 adjudication targets.
        # MECH-094: waking-only (this method is the waking seeding path); the
        # store's write() also no-ops under simulation_mode.
        if som is not None and self.goal_state.is_active():
            salience = float(benefit_exposure) * (
                1.0 + float(self.config.goal.drive_weight) * float(effective_drive)
            )
            som.write(
                self._current_latent.z_world,
                self.goal_state.z_goal,
                salience=salience,
                simulation_mode=False,
            )

    def set_super_ordinal_write_enabled(self, enabled: bool) -> None:
        """MECH-189 developmental-window control: open (child phase) or freeze
        (adult phase) super-ordinal anchor writes. The curriculum calls this at
        the child->adult transition so adult measurement reads from the
        childhood-formed hierarchy without forming new anchors (selective
        neoteny). No-op when the substrate is disabled."""
        if self.super_ordinal_goal_memory is not None:
            self.super_ordinal_goal_memory.write_enabled = bool(enabled)

    def reset_super_ordinal_anchors(self) -> None:
        """MECH-189: clear the persistent super-ordinal store -- for a NEW
        developmental stage / fresh agent only. NOT called on per-episode
        agent.reset() (cross-episode persistence is the point). No-op when the
        substrate is disabled."""
        if self.super_ordinal_goal_memory is not None:
            self.super_ordinal_goal_memory.reset_anchors()

    def cue_recall_wanting(
        self,
        cue_type: int,
        drive_level: float = 1.0,
        simulation_mode: bool = False,
    ) -> float:
        """SD-057 phase-2 L6 (MECH-347): cue-triggered wanting.

        A PERCEIVED cue/object type (NO benefit pulse) retrieves its incentive
        token from the bank and nudges z_goal toward that object's stored
        embedding BEFORE consumption -- identity-matched (pulls toward THIS
        cue's object), drive-specific (amplitude scales with the cue-type's
        per-axis drive). The downstream E3 goal_proximity + MECH-295 approach
        bridge then raise pre-consummatory approach toward the cued object.

        Distinct from update_z_goal: no benefit pulse, NO token revaluation
        (bank.update is not called), and a weaker (cue_recall_gain) pull than
        the benefit-driven seed. Returns the pull strength actually applied
        (0.0 when no-op).

        MECH-094: simulation_mode=True is a no-op (replay must not move z_goal
        via a cue). Requires GoalConfig.use_cue_recall + an active bank.
        """
        if simulation_mode:
            return 0.0
        gs = self.goal_state
        if gs is None or not getattr(gs.config, "use_cue_recall", False):
            return 0.0
        bank = getattr(gs, "incentive_bank", None)
        if bank is None:
            return 0.0
        k = int(cue_type)
        if k <= 0 or k not in bank._base_value:
            return 0.0
        # Recall-time wanting amplitude for this cue (drive-specific via the
        # bank's own per-axis logic), reusing the cached per-axis drive.
        per_axis = getattr(self, "_per_axis_drive", None)
        amp = bank.wanting(per_axis_drive=per_axis, scalar_drive=float(drive_level)).get(k, 0.0)
        if amp <= 0.0:
            return 0.0
        # Pull strength = cue gain * clamped wanting amplitude (clamp keeps the
        # pre-consummatory nudge bounded regardless of accumulated token value).
        strength = float(gs.config.cue_recall_gain) * float(min(1.0, amp))
        z_object = bank._z_object.get(k, None)
        if z_object is None or strength <= 0.0:
            return 0.0
        gs.cue_pull(z_object, strength)
        return strength

    def serotonin_step(self, benefit_exposure: float) -> None:
        """
        MECH-203: Update tonic 5-HT and dynamically modulate GoalConfig.

        Extracts z_harm_a norm from current latent state. Updates tonic_5ht,
        then writes current_seeding_gain and current_wanting_floor into
        the GoalConfig so that goal.update() uses dynamic serotonergic values.

        No-op when serotonin.enabled is False (default).
        """
        if not self.serotonin.enabled:
            return

        z_harm_a_norm = 0.0
        if self._current_latent is not None and self._current_latent.z_harm_a is not None:
            z_harm_a_norm = float(self._current_latent.z_harm_a.norm().item())

        self.serotonin.serotonin_step(benefit_exposure, z_harm_a_norm)

        # Dynamically modulate GoalConfig parameters
        if self.goal_state is not None:
            self.goal_state.config.z_goal_seeding_gain = self.serotonin.current_seeding_gain()
            self.goal_state.config.valence_wanting_floor = self.serotonin.current_wanting_floor()

    def update_benefit_salience(self, benefit_exposure: float) -> None:
        """
        MECH-203 (SR-2): Tag current residue field location with benefit salience.

        Uses SD-014 VALENCE_WANTING infrastructure: writes benefit_salience into
        the residue field's valence vector at the current z_world position.

        No-op when serotonin is disabled or no current latent state.
        """
        if not self.serotonin.enabled or self._current_latent is None:
            return

        salience = self.serotonin.benefit_salience(benefit_exposure)
        if salience <= 0.0:
            return

        z_world = self._current_latent.z_world
        if hasattr(self.residue_field, 'update_valence'):
            # Write benefit_salience into VALENCE_WANTING (index 0) at current z_world
            self.residue_field.update_valence(
                z_world,
                component=VALENCE_WANTING,
                value=salience,
                hypothesis_tag=False,
            )

    def update_liking(self, benefit_exposure: float) -> None:
        """SD-014 l-component: write consummatory benefit signal to VALENCE_LIKING.

        Berridge liking = hedonic impact at resource contact (opioid-mediated).
        Unlike wanting (anticipatory), liking fires on consummatory contact only.
        Threshold gating prevents proximity gradients from contaminating the liking map.
        No-op when valence_liking_enabled is False or no current latent state.
        """
        if not getattr(self.config, "valence_liking_enabled", False):
            return
        if self._current_latent is None:
            return
        threshold = getattr(self.config, "liking_threshold", 0.1)
        if benefit_exposure < threshold:
            return
        z_world = self._current_latent.z_world
        if hasattr(self.residue_field, "update_valence"):
            self.residue_field.update_valence(
                z_world,
                component=VALENCE_LIKING,
                value=benefit_exposure,
                hypothesis_tag=False,
            )

    @staticmethod
    def compute_drive_level(obs_body: torch.Tensor) -> float:
        """SD-012: Compute homeostatic drive from body observation.

        drive_level = 1.0 - energy (obs_body[3]). 0=sated, 1=depleted.
        Canonical formula -- single source of truth for SD-012.
        """
        if obs_body.dim() == 2:
            energy = float(obs_body[0, 3])
        else:
            energy = float(obs_body[3])
        return max(0.0, 1.0 - energy)

    def update_schema_wanting(self, drive_level: float = 1.0) -> None:
        """MECH-216: seed VALENCE_WANTING from E1 schema salience.

        Zhang/Berridge: W_m = kappa (drive_level) x V_hat (schema_salience).
        Only writes when schema_salience >= threshold and schema_wanting is enabled.

        MECH-307 amendments (registered 2026-05-08, all default OFF):
          Gap 2: when use_mech307_schema_multichannel=True, also writes
            anticipatory VALENCE_LIKING and pulses z_beta arousal. This
            implements the cue-stage NAcc-anticipation conjunction biology
            shows is dissociable from VALENCE_WANTING amplitude alone.
          Gap 4: when use_mech307_predicted_location_write=True, the write
            target is the cached E1 forward prediction (self._cached_e1_prior)
            rather than the agent's current z_world. This mirrors hippocampal
            place-cell preplay marking the predicted goal location.
        """
        if self._schema_salience is None or self._current_latent is None:
            return
        if not getattr(self.config.e1, 'schema_wanting_enabled', False):
            return
        sal_val = float(self._schema_salience.squeeze())
        threshold = getattr(self.config, 'schema_wanting_threshold', 0.3)
        if sal_val < threshold:
            return
        gain = getattr(self.config, 'schema_wanting_gain', 0.5)
        wanting_value = sal_val * gain * max(drive_level, 0.1)

        # MECH-307 Gap 4: write target is e1_prior (predicted z_world) when the
        # flag is set and a cached prediction exists; otherwise current z_world
        # (legacy behaviour, bit-identical when flag is False).
        if (
            getattr(self.config, "use_mech307_predicted_location_write", False)
            and self._cached_e1_prior is not None
        ):
            write_target = self._cached_e1_prior
        else:
            write_target = self._current_latent.z_world

        if hasattr(self.residue_field, 'update_valence'):
            self.residue_field.update_valence(
                write_target, component=VALENCE_WANTING,
                value=wanting_value, hypothesis_tag=False,
            )

        # MECH-307 Gap 2 + Gap 3: multi-channel schema readout. Adds an
        # anticipatory VALENCE_LIKING write at the same target plus a salience-
        # proportional pulse to z_beta arousal. Bit-identical when flag is
        # False (no extra writes, no z_beta modification).
        if getattr(self.config, "use_mech307_schema_multichannel", False):
            liking_gain = getattr(
                self.config, "mech307_anticipatory_liking_gain", 0.5
            )
            liking_value = sal_val * liking_gain * max(drive_level, 0.1)
            if hasattr(self.residue_field, 'update_valence'):
                self.residue_field.update_valence(
                    write_target, component=VALENCE_LIKING,
                    value=liking_value, hypothesis_tag=False,
                )

            # Gap 3: z_beta arousal pulse. Adds a salience-proportional value
            # to the first dimension of z_beta in-place. Subsequent encoder
            # passes will integrate / decay this through the existing
            # alpha_shared blend in LatentStack.encode (line ~1294). The
            # pulse is small (gain * salience) so it does not overwhelm the
            # encoder's natural z_beta dynamics; it acts as a cue-time
            # anticipatory bump that MECH-093 reads to elevate E3 rate.
            z_beta_gain = getattr(
                self.config, "mech307_z_beta_schema_gain", 0.3
            )
            pulse_magnitude = sal_val * z_beta_gain
            if (
                pulse_magnitude > 0.0
                and self._current_latent.z_beta is not None
                and self._current_latent.z_beta.numel() > 0
            ):
                with torch.no_grad():
                    self._current_latent.z_beta[..., 0] = (
                        self._current_latent.z_beta[..., 0] + pulse_magnitude
                    )

    def compute_schema_readout_loss(
        self, resource_proximity_target: float
    ) -> torch.Tensor:
        """MECH-216: passthrough to E1.compute_schema_readout_loss."""
        if not getattr(self.config.e1, 'schema_wanting_enabled', False):
            return torch.tensor(0.0)
        target = torch.tensor([[resource_proximity_target]], dtype=torch.float32)
        return self.e1.compute_schema_readout_loss(target)

    def compute_goal_maintenance_diagnostic(self) -> dict:
        """Goal state metrics for MECH-116 (EXQ-076 criterion C1)."""
        if self.goal_state is None or self._current_latent is None:
            return {"goal_norm": 0.0, "goal_proximity": 0.0, "is_active": False}
        prox = float(
            self.goal_state.goal_proximity(
                self._current_latent.z_world
            ).mean().item()
        )
        return {
            "goal_norm": self.goal_state.goal_norm(),
            "goal_proximity": prox,
            "is_active": self.goal_state.is_active(),
        }

    def compute_z_self_d_eff(self) -> Optional[float]:
        """
        MECH-113: Compute z_self participation ratio (D_eff).

        D_eff = (sum|z_self|)^2 / sum(z_self^2)
        From epistemic-mapping: measures effective dimensionality of the
        self-model. High D_eff = distributed/uncertain; low D_eff = focused.
        Self-maintenance loss penalises D_eff exceeding the target threshold.

        Returns:
            D_eff scalar or None if no latent state available.
        """
        if self._current_latent is None:
            return None
        z = self._current_latent.z_self.detach().squeeze(0)  # [self_dim]
        abs_z = z.abs()
        numerator = abs_z.sum().pow(2)
        denominator = z.pow(2).sum()
        if denominator.item() < 1e-8:
            return None
        return (numerator / denominator).item()

    def compute_self_maintenance_loss(self) -> torch.Tensor:
        """
        MECH-113: Self-maintenance auxiliary loss on z_self D_eff.

        Penalises D_eff > self_maintenance_d_eff_target.
        This creates a homeostatic pressure to keep z_self representations
        focused (low D_eff = coherent self-model). Disabled when weight=0.

        Returns:
            Scalar loss tensor (zero if disabled or no latent state).
        """
        weight = self.config.e3.self_maintenance_weight
        zero_loss = next(self.e3.parameters()).sum() * 0.0
        if weight == 0.0 or self._current_latent is None:
            return zero_loss
        z = self._current_latent.z_self  # [batch, self_dim] — keep in graph
        abs_z = z.abs()
        numerator = abs_z.sum(dim=-1).pow(2)     # [batch]
        denominator = z.pow(2).sum(dim=-1)        # [batch]
        d_eff = numerator / (denominator + 1e-8)  # [batch]
        target = self.config.e3.self_maintenance_d_eff_target
        excess = F.relu(d_eff - target)           # penalise only excess
        return weight * excess.mean()

    def compute_e2_loss(self, batch_size: int = 16) -> torch.Tensor:
        """E2 motor-sensory forward-model loss (z_self domain)."""
        zero_loss = next(self.e2.parameters()).sum() * 0.0
        if len(self._e2_transition_buffer) < 2:
            return zero_loss

        n = min(batch_size, len(self._e2_transition_buffer))
        indices = torch.randperm(len(self._e2_transition_buffer))[:n].tolist()
        batch = [self._e2_transition_buffer[i] for i in indices]
        z_t_list, a_list, z_t1_list = zip(*batch)

        z_t  = torch.cat(z_t_list, dim=0)
        acts = torch.cat(a_list, dim=0)
        z_t1 = torch.cat(z_t1_list, dim=0)

        z_t1_pred = self.e2.predict_next_self(z_t, acts)
        return F.mse_loss(z_t1_pred, z_t1)

    def offline_integration(self) -> Dict[str, float]:
        """Offline integration (residue contextualisation + E1 replay)."""
        metrics: Dict[str, float] = {}
        if len(self._world_experience_buffer) > 10:
            e1_metrics = self.e1.integrate_experience(
                [torch.cat([s, w]) for s, w in
                 zip(self._self_experience_buffer, self._world_experience_buffer)]
            )
            metrics.update({f"e1_{k}": v for k, v in e1_metrics.items()})
        residue_metrics = self.residue_field.integrate()
        metrics.update({f"residue_{k}": v for k, v in residue_metrics.items()})
        return metrics

    def should_integrate(self) -> bool:
        return self._step_count % self.config.offline_integration_frequency == 0

    def enter_offline_mode(self) -> None:
        """
        MECH-120 / SD-017: Gate E1 context_memory writes during offline phases.

        Sets e1._offline_mode = True, which suppresses context_memory.write()
        calls in E1.update_from_observation(). This prevents new waking
        observations from overwriting schema slots installed during the
        SWS-analog pass (Phase 0 sensory gate from offline_phases.md).

        Call before running SWS/REM-analog passes.
        Paired with exit_offline_mode() to resume normal waking writes.

        SD-032d: also resets the PCC steps_since_offline counter so the
        coordinator's effective_threshold relaxes after rest. Both MECH-092
        within-session quiescence and INV-049 cross-session sleep paths
        funnel through here, giving SD-032d a single integration point.
        """
        self.e1._offline_mode = True
        if self.pcc is not None:
            self.pcc.note_offline_entry()
        # SD-032e: optional offline decay of drive_bias (default 0.0 = no-op).
        # Non-zero decay would instantiate a distinct sleep-recalibration
        # claim per the SD-032e scoping synthesis; default preserves the
        # cross-session pACC accumulator untouched.
        if self.pacc is not None:
            self.pacc.note_offline_entry()

    def exit_offline_mode(self) -> None:
        """Resume normal waking context_memory writes (undo enter_offline_mode)."""
        self.e1._offline_mode = False

    def note_task_outcome(self, outcome: float) -> None:
        """SD-032d: feed a per-step task-outcome scalar into the PCC success EMA.

        Convenience pass-through to self.pcc.note_task_outcome(outcome). No-op
        when use_pcc_analog=False. Experiment loops choose what counts as a
        task outcome (e.g., 1.0 on benefit-collection, 0.0 on harm event,
        0.5 otherwise). Without any calls, the PCC success channel stays
        neutral and contributes zero to stability.
        """
        if self.pcc is not None:
            self.pcc.note_task_outcome(outcome)

    # -- Sleep mode convenience methods (SD-017 + MECH-203/204) --

    def enter_sws_mode(self) -> None:
        """
        Enter SWS-analog phase.

        Composes: enter_offline_mode() + SHY normalisation + serotonin.enter_sws().
        Phase 0: gate waking writes. Phase 1: SHY normalisation (MECH-120).
        Phase 2 (replay) runs after this method returns.
        """
        self.enter_offline_mode()
        # MECH-120: SHY normalization before replay (Phase 1)
        if self.config.shy_enabled:
            self.e1.shy_normalise(decay=self.config.shy_decay_rate)
        self.serotonin.enter_sws()

    def enter_rem_mode(self) -> None:
        """
        Enter REM-analog phase.

        Composes: enter_offline_mode() + serotonin.enter_rem().
        SR-3: captures current E3 precision as zero-point reference.
        """
        self.enter_offline_mode()
        self.serotonin.enter_rem(current_precision=self.e3.current_precision)

    def exit_sleep_mode(self) -> None:
        """
        Exit sleep, return to waking mode.

        Composes: exit_offline_mode() + serotonin.exit_sleep().
        Restores E1 context_memory writes and pre-sleep 5-HT level.
        """
        self.exit_offline_mode()
        self.serotonin.exit_sleep()

    # -- SD-017: SWS-analog and REM-analog passes --

    def run_sws_schema_pass(self, anchor_weight: float = 1.0) -> Dict[str, float]:
        """
        SD-017 SWS-analog pass: hippocampus-to-cortex schema installation.

        Installs differentiated context attractors in E1.ContextMemory by
        writing compressed z_world prototypes from recent waking experience.
        This is the slot-formation phase (MECH-166): before this pass, the
        ContextMemory slots are undifferentiated (cosine_sim -> 1.0); after,
        they form attractors that slot-filling (REM-analog) can populate.

        Must be called after enter_sws_mode() (which gates waking writes and
        runs MECH-120 SHY normalisation). The offline gate prevents new waking
        observations from overwriting slots installed here.

        Algorithm:
        1. Sample sws_consolidation_steps prototype z_worlds from the world
           experience buffer (diverse sampling: early, mid, and recent windows).
        2. For each prototype, construct the full E1 input [z_self_mean, z_world],
           scale by anchor_weight (MECH-272 routing gate anchor channel), and
           write directly to ContextMemory bypassing the offline gate.
           (The offline gate suppresses waking writes; this pass writes offline
           schema content, which is the intended action during SWS.)
        3. Compute slot diversity (mean pairwise cosine distance) as a metric
           for context differentiation quality.

        Args:
            anchor_weight: MECH-272 routing gate anchor_channel value for this
                           cycle. Scales the e1_input before writing to
                           ContextMemory (1.0 = full strength, 0.6 = SWS default
                           per routing table). Defaults to 1.0 (no scaling) for
                           callers that do not use the routing gate.

        Returns:
            dict with keys:
              sws_n_writes: number of prototype writes attempted
              sws_slot_diversity: mean pairwise cosine distance of ContextMemory
                                  slots after pass (higher = more differentiated)
              sws_buffer_size: size of experience buffer used
              sws_anchor_weight_applied: the anchor_weight value used this pass
        """
        metrics: Dict[str, float] = {
            "sws_n_writes": 0.0,
            "sws_slot_diversity": 0.0,
            "sws_buffer_size": 0.0,
            "sws_anchor_weight_applied": float(anchor_weight),
        }

        if not self.config.sws_enabled:
            return metrics

        wb = self._world_experience_buffer
        sb = self._self_experience_buffer
        n_buf = len(wb)
        metrics["sws_buffer_size"] = float(n_buf)

        if n_buf < 2:
            return metrics

        # Temporarily lift offline gate for schema writes
        # (gate suppresses waking obs; schema installation is intentional offline write)
        was_offline = self.e1._offline_mode
        self.e1._offline_mode = False

        n_steps = min(self.config.sws_consolidation_steps, n_buf)
        # Diverse sampling: spread across buffer windows
        if n_steps >= n_buf:
            indices = list(range(n_buf))
        else:
            # Sample from early, mid, recent thirds plus random fill
            step = max(1, n_buf // n_steps)
            indices = list(range(0, n_buf, step))[:n_steps]

        n_writes = 0
        self_dim = self.config.latent.self_dim

        for idx in indices:
            z_world = wb[idx].detach()     # [1, world_dim] or [world_dim]
            if z_world.dim() == 1:
                z_world = z_world.unsqueeze(0)

            # Pair with corresponding z_self if available, else zeros
            if idx < len(sb):
                z_self = sb[idx].detach()
                if z_self.dim() == 1:
                    z_self = z_self.unsqueeze(0)
            else:
                z_self = torch.zeros(1, self_dim, device=self.device)

            # Full E1 input: [z_self, z_world] concatenated
            e1_input = torch.cat([z_self, z_world], dim=-1)  # [1, self_dim+world_dim]

            # GAP-8: scale by MECH-272 anchor_channel weight before writing.
            # ContextMemory.write has no weight param; scaling the input is the
            # correct approach -- it scales the magnitude of content being written.
            if anchor_weight != 1.0:
                e1_input = e1_input * anchor_weight

            # Write to ContextMemory (offline gate lifted for this block only)
            self.e1.context_memory.write(e1_input)
            n_writes += 1

        # Restore gate
        self.e1._offline_mode = was_offline

        metrics["sws_n_writes"] = float(n_writes)

        # Compute slot diversity: mean pairwise cosine distance across memory slots
        with torch.no_grad():
            mem = self.e1.context_memory.memory  # [num_slots, memory_dim]
            num_slots = mem.shape[0]
            if num_slots > 1:
                # Normalise rows
                norms = mem.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                normed = mem / norms  # [num_slots, memory_dim]
                # Cosine similarity matrix [num_slots, num_slots]
                sim_mat = torch.mm(normed, normed.t())
                # Mask diagonal (self-similarity = 1.0)
                mask = torch.eye(num_slots, device=sim_mat.device, dtype=torch.bool)
                off_diag = sim_mat[~mask]
                # Diversity = mean pairwise distance (1 - cosine similarity)
                diversity = float((1.0 - off_diag).mean().item())
                metrics["sws_slot_diversity"] = diversity

        return metrics

    def run_rem_attribution_pass(self) -> Dict[str, float]:
        """
        SD-017 REM-analog pass: causal attribution replay (slot-filling, MECH-166).

        Replays recent trajectory experience through the hippocampal module in
        both forward and reverse temporal order (ARC-045 bidirectional flow proxy).
        Evaluates residue terrain per trajectory segment WITHOUT accumulating new
        residue (hypothesis_tag=True per MECH-094). This is the slot-filling phase:
        with schema attractors installed by the SWS pass, trajectory evidence can
        now be attributed to differentiated context slots.

        Must be called after run_sws_schema_pass() (slots must exist before
        filling). The offline gate should still be active (enter_rem_mode()
        ensures this).

        Algorithm:
        1. Take recent z_world from theta_buffer (waking experience).
        2. For each attribution step:
           a. Forward replay: hippocampal.replay() -- random rollout from recent z_world.
           b. Reverse replay: hippocampal.reverse_replay() on a stored trajectory.
           c. Evaluate residue terrain on each trajectory (read-only, no writes).
        3. Aggregate attribution metrics: mean harm terrain, benefit terrain (if enabled),
           context differentiation proxy (variance of terrain scores across rollouts).

        Returns:
            dict with keys:
              rem_n_rollouts: number of rollouts attempted
              rem_mean_harm_terrain: mean residue terrain score across rollouts
              rem_terrain_variance: variance of terrain scores (context differentiation proxy)
              rem_n_reverse: number of reverse-order rollouts included
        """
        metrics: Dict[str, float] = {
            "rem_n_rollouts": 0.0,
            "rem_mean_harm_terrain": 0.0,
            "rem_terrain_variance": 0.0,
            "rem_n_reverse": 0.0,
        }

        if not self.config.rem_enabled:
            return metrics

        recent = self.theta_buffer.recent
        if recent is None:
            return metrics

        n_steps = self.config.rem_attribution_steps
        terrain_scores: List[float] = []
        n_reverse = 0
        retrieval_bias = (
            self._bla_last_output.retrieval_bias
            if self._bla_last_output is not None
            else None
        )

        # Forward replay pass: hypothesis_tag=True (read-only, no residue writes)
        forward_trajs = self.hippocampal.replay(
            recent,
            num_replay_steps=max(1, n_steps // 2),
            drive_state=None,  # attribution mode: drive-neutral
        )

        for traj in forward_trajs:
            score = self.hippocampal._score_trajectory(traj)
            terrain_scores.append(float(score.item() if isinstance(score, torch.Tensor) else score))

        # Reverse replay pass (ARC-045 bidirectional flow proxy)
        # Only if exploration buffer has stored trajectories
        n_reverse_steps = max(1, n_steps - len(forward_trajs))
        if len(self.hippocampal._exploration_buffer) > 0:
            reverse_trajs = self.hippocampal.diverse_replay(
                recent,
                num_replay_steps=n_reverse_steps,
                drive_state=None,
                mode="reverse",
                retrieval_bias=retrieval_bias,
            )
            for traj in reverse_trajs:
                score = self.hippocampal._score_trajectory(traj)
                terrain_scores.append(float(score.item() if isinstance(score, torch.Tensor) else score))
                n_reverse += 1
        else:
            # No exploration buffer yet: extra forward rollouts
            extra = self.hippocampal.replay(recent, num_replay_steps=n_reverse_steps)
            for traj in extra:
                score = self.hippocampal._score_trajectory(traj)
                terrain_scores.append(float(score.item() if isinstance(score, torch.Tensor) else score))

        if terrain_scores:
            metrics["rem_n_rollouts"] = float(len(terrain_scores))
            metrics["rem_mean_harm_terrain"] = float(sum(terrain_scores) / len(terrain_scores))
            if len(terrain_scores) > 1:
                mean_s = metrics["rem_mean_harm_terrain"]
                var_s = sum((s - mean_s) ** 2 for s in terrain_scores) / len(terrain_scores)
                metrics["rem_terrain_variance"] = var_s
            metrics["rem_n_reverse"] = float(n_reverse)

        return metrics

    def run_sleep_cycle(self, sws_anchor_weight: float = 1.0) -> Dict[str, float]:
        """
        SD-017: Run a complete SWS->REM sleep cycle.

        Convenience method: calls enter_sws_mode(), run_sws_schema_pass(),
        enter_rem_mode(), run_rem_attribution_pass(), then exit_sleep_mode().

        Returns merged metrics from both passes.

        Args:
            sws_anchor_weight: MECH-272 anchor_channel weight forwarded to
                run_sws_schema_pass(). Default 1.0 preserves waking
                bit-identical behaviour (GAP-8 consumer wiring).

        IMPORTANT: Only call this when sws_enabled=True or rem_enabled=True.
        If both are False this is a no-op returning empty metrics.
        """
        all_metrics: Dict[str, float] = {}
        if not self.config.sws_enabled and not self.config.rem_enabled:
            return all_metrics

        # SWS phase
        if self.config.sws_enabled:
            self.enter_sws_mode()
            sws_metrics = self.run_sws_schema_pass(anchor_weight=sws_anchor_weight)
            all_metrics.update(sws_metrics)
            self.exit_sleep_mode()

        # REM phase (requires slots installed by SWS; safe to run standalone too)
        if self.config.rem_enabled:
            self.enter_rem_mode()
            rem_metrics = self.run_rem_attribution_pass()
            all_metrics.update(rem_metrics)
            self.exit_sleep_mode()

        return all_metrics

    def _assemble_control_vector(
        self,
        effective_temperature: float,
        baseline_temperature: float,
    ) -> None:
        """Assemble the read-only ControlVector telemetry (recommendation B).

        Writes self._last_control_vector with four separately-inspectable
        control signals for the current E3 tick:

          V_outcome -- primary value axis (E3 pre-bias scores; lower-is-better,
                       so value = -score).
          C_effort  -- dACC Shenhav EVC effort term (control_required *
                       candidate_effort), from the dACC bundle.
          C_time    -- MECH-320 tonic-vigor +w_passive*v_t no-op /
                       opportunity-cost-of-time half.
          G_vigor   -- MECH-320 tonic-vigor -w_action*v_t action half plus the
                       MECH-313 noise-floor temperature lift.

        The shared MECH-320 v_t is logged explicitly so the C_time<->G_vigor
        collapse (both halves are w*v_t for the SAME v_t) is computable from a
        manifest -- the Stage-B diagnostic this telemetry is built to expose.
        Pure read-only: no scoring or selection effect. Called only when
        config.use_control_vector_logging is True.
        """

        def _stats(t: "Optional[torch.Tensor]") -> dict:
            if t is None or t.numel() == 0:
                return {"mean": 0.0, "range": 0.0, "std": 0.0, "present": False}
            f = t.detach().reshape(-1).float()
            rng = float((f.max() - f.min()).item()) if f.numel() > 1 else 0.0
            std = float(f.std(unbiased=False).item()) if f.numel() > 1 else 0.0
            return {
                "mean": float(f.mean().item()),
                "range": rng,
                "std": std,
                "present": True,
            }

        # V_outcome: primary value axis = pre-bias E3 scores (lower = better).
        _vo = _stats(getattr(self.e3, "last_raw_scores", None))
        v_outcome = {
            **_vo,
            "value_mean": -_vo["mean"] if _vo["present"] else 0.0,
        }

        # C_effort: dACC EVC effort term from the last bundle (None when dACC
        # is disabled this tick).
        bundle = getattr(self, "_dacc_last_bundle", None)
        if bundle is not None and bundle.get("effort_term", None) is not None:
            c_effort = {
                **_stats(bundle["effort_term"]),
                "control_required": float(bundle.get("control_required", 0.0)),
            }
        else:
            c_effort = {
                "mean": 0.0,
                "range": 0.0,
                "std": 0.0,
                "present": False,
                "control_required": 0.0,
            }

        # C_time / G_vigor: from the MECH-320 split cached in the tonic_vigor
        # block (None when tonic_vigor is disabled / did not fire this tick).
        cv = self._cv_vigor
        noise_floor_temp_lift = float(effective_temperature) - float(
            baseline_temperature
        )
        if cv is not None:
            c_time = {
                "potential": cv["C_time_potential"],
                "realised_mean": cv["C_time_realised_mean"],
                "n_noop_candidates": cv["n_noop_candidates"],
                "present": True,
            }
            g_vigor = {
                "potential": cv["G_vigor_potential"],
                "realised_mean": cv["G_vigor_realised_mean"],
                "noise_floor_temp_lift": noise_floor_temp_lift,
                "n_action_candidates": cv["n_action_candidates"],
                "present": True,
            }
            shared = {
                "tonic_vigor_v_t": cv["v_t"],
                "tonic_vigor_v_raw": cv["v_raw"],
                "w_action": cv["w_action"],
                "w_passive": cv["w_passive"],
                # The collapse, stated for any downstream reader: opportunity
                # cost and vigor are not independent -- both are w * v_t for the
                # same v_t (ARC-068 absorbed into MECH-320; see the four-signal
                # control adjudication 2026-06-07).
                "collapse_note": (
                    "C_time and G_vigor are both w*v_t for the SAME MECH-320 "
                    "v_t scalar (ARC-068 collapsed into MECH-320)."
                ),
            }
        else:
            # Tonic vigor off: opportunity-cost / vigor have no MECH-320 source.
            # G_vigor still carries the noise-floor activation lift if any.
            c_time = {
                "potential": 0.0,
                "realised_mean": 0.0,
                "n_noop_candidates": 0,
                "present": False,
            }
            g_vigor = {
                "potential": 0.0,
                "realised_mean": 0.0,
                "noise_floor_temp_lift": noise_floor_temp_lift,
                "n_action_candidates": 0,
                "present": noise_floor_temp_lift != 0.0,
            }
            shared = {
                "tonic_vigor_v_t": 0.0,
                "tonic_vigor_v_raw": 0.0,
                "w_action": 0.0,
                "w_passive": 0.0,
                "collapse_note": "tonic_vigor disabled; no MECH-320 source.",
            }

        # Authority context: did the modulatory bias actually have selection
        # authority this tick? (modulatory-bias-selection-authority substrate.)
        _diag = getattr(self.e3, "last_score_diagnostics", {}) or {}
        authority = {
            "modulatory_authority_active": bool(
                _diag.get("modulatory_authority_active", False)
            ),
            "modulatory_authority_scale_factor": float(
                _diag.get("modulatory_authority_scale_factor", 0.0)
            ),
            "e3_raw_score_range_mean": float(
                _diag.get("e3_raw_score_range_mean", v_outcome["range"])
            ),
        }

        self._last_control_vector = {
            "step": int(self._step_count),
            "V_outcome": v_outcome,
            "C_effort": c_effort,
            "C_time": c_time,
            "G_vigor": g_vigor,
            "shared": shared,
            "authority": authority,
        }

    def get_state(self) -> AgentState:
        return AgentState(
            latent_state=self._current_latent,
            precision=self.e3.current_precision,
            running_variance=self.e3._running_variance,
            step=self._step_count,
            harm_accumulated=self._harm_this_episode,
            is_committed=(
                self.e3._committed_trajectory is not None
                or self.e3._closure_committed_trajectory is not None
            ),
            beta_elevated=self.beta_gate.is_elevated,
            e3_steps_per_tick=self.clock.e3_steps_per_tick,
            serotonin_state=self.serotonin.get_state() if self.serotonin.enabled else None,
        )

    def get_residue_statistics(self) -> Dict[str, torch.Tensor]:
        return self.residue_field.get_statistics()

    def forward(self, observation: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        return self.act(observation, temperature)
