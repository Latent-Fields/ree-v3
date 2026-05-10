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

import torch
import torch.nn as nn
import torch.nn.functional as F

from ree_core.utils.config import REEConfig
from ree_core.latent.stack import LatentStack, LatentState
from ree_core.goal import GoalState
from ree_core.latent.theta_buffer import ThetaBuffer
from ree_core.predictors.e1_deep import E1DeepPredictor
from ree_core.predictors.e2_fast import E2FastPredictor, Trajectory
from ree_core.predictors.e3_selector import E3TrajectorySelector
from ree_core.residue.field import ResidueField
from ree_core.hippocampal.module import HippocampalModule
from ree_core.heartbeat.clock import MultiRateClock
from ree_core.heartbeat.beta_gate import BetaGate
from ree_core.neuromodulation.serotonin import SerotoninModule
from ree_core.predictors.e2_harm_a import E2HarmAConfig, E2HarmAForward
from ree_core.predictors.e2_harm_s import E2HarmSConfig, E2HarmSForward
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
)
from ree_core.latent.stack import HarmForwardTrunk
from ree_core.pfc import LateralPFCAnalog, OFCAnalog
from ree_core.pfc.lateral_pfc_analog import LateralPFCConfig
from ree_core.pfc.ofc_analog import OFCConfig
from ree_core.policy import (
    GatedPolicy,
    GatedPolicyConfig,
    NoiseFloor,
    NoiseFloorConfig,
    StructuredCuriosity,
    StructuredCuriosityConfig,
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
from ree_core.regulators import (
    GABAergicDecayConfig,
    GABAergicDecayRegulator,
    BroadcastOverrideConfig,
    BroadcastOverrideRegulator,
    MECH295LikingBridge,
    MECH295LikingBridgeConfig,
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
)


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

        # MECH-090: BetaGate for policy propagation
        self.beta_gate = BetaGate()

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
            )
            self.dacc = DACCAdaptiveControl(dacc_cfg)
            # STOPGAP adapter -- still the score_bias source until SD-033
            # substrates consume operating_mode natively. With SD-032a active,
            # the adapter's bias may be scaled by the coordinator's e3_policy
            # write-gate (see select_action; gated by salience_apply_to_dacc_bias).
            self.dacc_adapter = DACCtoE3Adapter(dacc_cfg)

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
            )
            self.lateral_pfc = LateralPFCAnalog(
                delta_dim=config.latent.delta_dim,
                world_dim=config.latent.world_dim,
                config=lpfc_cfg,
            )

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
            )
            self.curiosity = StructuredCuriosity(config=cur_cfg)

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
                    getattr(config, "mech295_min_drive_to_fire", 0.1)
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
                    getattr(config, "mech307_conjunction_z_beta_threshold", 0.6)
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

        # MECH-057a: cached E3 candidates for action-loop gate
        self._committed_candidates: Optional[List[Trajectory]] = None
        # Last selected action (held between E3 ticks)
        self._last_action: Optional[torch.Tensor] = None
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
        self._last_action = None
        self._committed_step_idx = 0
        # MECH-269 / MECH-090 read-side hook: clear V_s -> commit release snapshot.
        self._committed_anchor_keys = None
        self._cue_action_bias    = None
        self._cue_terrain_weight = None
        self.clock.reset()
        self.theta_buffer.reset()
        self.beta_gate.reset()
        self.serotonin.reset()
        self._pe_ema = 0.0
        # MECH-307 Gap 4: clear cached E1 prior on episode boundary.
        self._cached_e1_prior = None
        # SD-019a: reset harm_unpleasantness EMA on episode boundary.
        self._harm_un_ema = None
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

    def sense(
        self,
        obs_body: torch.Tensor,
        obs_world: torch.Tensor,
        obs_harm: Optional[torch.Tensor] = None,
        obs_harm_a: Optional[torch.Tensor] = None,
        obs_harm_history: Optional[torch.Tensor] = None,
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

        Returns:
            Updated LatentState
        """
        if obs_body.dim() == 1:
            obs_body  = obs_body.unsqueeze(0)
            obs_world = obs_world.unsqueeze(0)
        obs_body  = obs_body.to(self.device).float()
        obs_world = obs_world.to(self.device).float()

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
        new_latent = self.latent_stack.encode(
            enc_combined, self._current_latent,
            prev_action=self._last_action,
            harm_obs=obs_harm,       # SD-010: nociceptive stream (None = disabled)
            harm_obs_a=obs_harm_a,   # SD-011: affective harm stream (None = disabled)
            harm_history=obs_harm_history,  # SD-011 second source (None = disabled)
            volatility_signal=vol_signal,
        )

        # MECH-095: resolve the TPJ efference-copy comparison for the most
        # recently executed action. Runs immediately after encoding so the
        # comparator sees the freshly observed z_self.
        self._update_tpj_comparator(new_latent)

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
            if new_latent.z_harm is not None:
                override_z_harm = float(new_latent.z_harm.norm().item())
            self.broadcast_override.tick(
                drive_level=override_drive,
                z_harm_norm=override_z_harm,
                simulation_mode=bool(getattr(new_latent, "hypothesis_tag", False)),
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
                    self._cea_last_output = self.cea.tick(
                        z_harm_a=z_harm_a_cur.detach(),
                        cue_features=None,
                        cortical_confirmation=None,
                        escapability_hint=None,
                        simulation_mode=False,
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
        candidates = self.hippocampal.propose_trajectories(
            z_world=z_world_for_e3,
            z_self=latent_state.z_self,
            num_candidates=num_candidates,
            e1_prior=e1_prior,
            action_bias=self._cue_action_bias,
            current_z_goal=_current_z_goal,
        )
        self._committed_candidates = candidates

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
            _urgency_signal = z_harm_a
            if (
                self.config.latent.use_harm_un
                and self._current_latent is not None
                and self._current_latent.z_harm_un is not None
            ):
                _urgency_signal = self._current_latent.z_harm_un
            if float(_urgency_signal.norm().item()) > urgency_threshold:
                self.beta_gate.release()
                self._committed_step_idx = 0
                self._committed_anchor_keys = None

        # MECH-269 / MECH-090 read-side hook: V_s -> commit release.
        # If any anchor key snapshotted at commit entry has dropped out of the
        # active anchor set since then, the schema region the commitment was
        # anchored to has been invalidated -- release beta and fall through to
        # fresh E3 selection. Mirrors the MECH-091 urgency-interrupt template.
        # No-op when the flag is off, when beta is not elevated, or when no
        # snapshot exists. Diagnostic counter incremented each time the
        # release fires (read by V3-EXQ-481 as commit_release_via_vs_count).
        if (
            getattr(self.hippocampal.config, "use_vs_commit_release", False)
            and self.beta_gate.is_elevated
            and self._committed_anchor_keys is not None
            and self.hippocampal.anchor_set is not None
        ):
            current_keys = {
                a.key for a in self.hippocampal.anchor_set.active_anchors()
            }
            if not self._committed_anchor_keys.issubset(current_keys):
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
            if float(safety_pred.mean()) >= release_thresh:
                self.beta_gate.release()
                self._committed_step_idx = 0
                self._committed_anchor_keys = None

        if not ticks["e3_tick"] and self._last_action is not None:
            # Between E3 ticks: step through committed trajectory (Layer 1) or hold.
            if self.beta_gate.is_elevated and self.e3._committed_trajectory is not None:
                traj = self.e3._committed_trajectory
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
            bundle = self.dacc(
                z_harm_a=z_harm_a.squeeze(0) if z_harm_a.dim() > 1 else z_harm_a,
                z_harm_a_pred=self._harm_a_pred_prev,
                candidate_payoffs=payoffs,
                candidate_effort=effort,
                candidate_action_classes=action_classes,
                precision=float(self.e3.current_precision),
                drive_level=drive_level,
            )
            self._dacc_last_bundle = bundle
            if self.dacc_adapter is not None:
                dacc_score_bias = self.dacc_adapter(bundle)
                self._dacc_last_bias = dacc_score_bias.detach().clone()

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
                self._pcc_last_tick = self.pcc.tick(drive_level=sal_drive)
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
            self._salience_last_tick = self.salience.tick(
                dacc_bundle=sal_bundle,
                drive_level=sal_drive,
                is_offline=sal_offline,
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

        # SD-033a: Lateral-PFC-analog tick. Primary consumer of MECH-261
        # write_gate("sd_033a"). Gate-modulated EMA update of rule_state,
        # then per-candidate score_bias composed additively with dACC bias.
        # Initial bias output is exactly zero (last Linear zeroed at init)
        # so use_lateral_pfc_analog=True with an untrained head is
        # backward-compatible until the head is deliberately trained.
        # MECH-094: rule persistence is gated by the registry -- no separate
        # hypothesis_tag check (the gate IS the tag via MECH-261).
        if (
            self.lateral_pfc is not None
            and self._current_latent is not None
        ):
            if self.salience is not None:
                lpfc_gate = float(self.salience.write_gate("sd_033a"))
            else:
                # Coordinator disabled -> full gate (lateral_pfc active under
                # use_lateral_pfc_analog alone, so ablation is possible without
                # requiring SD-032a to be on).
                lpfc_gate = 1.0
            # Update rule_state (in-place on buffer, no gradient flow).
            self.lateral_pfc.update(
                z_delta=self._current_latent.z_delta,
                z_world=self._current_latent.z_world,
                gate=lpfc_gate,
            )
            # Per-candidate z_world summary: first-step z_world of each
            # trajectory (trajectory.world_states[:, 0, :]). Falls back to
            # current z_world replicated across candidates if trajectories
            # lack world_states.
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

        # ARC-062 Phase 1: gated-policy heads + context discriminator.
        # Phase 1 weak-reading instantiation. Per-candidate score_bias
        # = w * head_0(features) + (1 - w) * head_1(features), where w
        # is a sigmoid over a 3-stream (z_world, z_self, z_harm_a)
        # discriminator (Pull A R1 multi-stream verdict). N=2 heads at
        # Phase 1 (Pull A R2 substrate-constrained). Composed additively
        # into the dACC / lateral_pfc / ofc score_bias chain (Pull A R3
        # score_bias-level verdict). NO connection to SD-033a in Phase 1
        # (that wiring is Phase 3 of arc_062_rule_apprehension_plan.md).
        # MECH-094: simulation_mode=False (waking action selection).
        if (
            self.gated_policy is not None
            and self._current_latent is not None
        ):
            # Reuse cand_world_summaries if lateral_pfc or ofc built them
            # earlier this tick; otherwise build fresh.
            try:
                gp_summaries = cand_world_summaries  # type: ignore[name-defined]
            except NameError:
                K = len(candidates)
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
            with torch.no_grad():
                gp_output = self.gated_policy(
                    z_world=self._current_latent.z_world,
                    z_self=self._current_latent.z_self,
                    z_harm_a=self._current_latent.z_harm_a,
                    candidate_features=gp_summaries,
                    simulation_mode=False,
                )
            gp_bias = gp_output.gated_score_bias
            if dacc_score_bias is None:
                dacc_score_bias = gp_bias
            else:
                dacc_score_bias = dacc_score_bias + gp_bias.to(
                    dtype=dacc_score_bias.dtype, device=dacc_score_bias.device
                )

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
                m295_bias = self.mech295_bridge.compute_approach_cue_score_bias(
                    drive_level=eff_drive_m295,
                    candidate_proximities=cand_proximities,
                    simulation_mode=False,
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
            with torch.no_grad():
                cur_bias = self.curiosity.compute_score_bias(
                    candidate_world_summaries=cur_summaries.detach(),
                    residue_field=self.residue_field,
                    e3=self.e3,
                    simulation_mode=False,
                )
            if dacc_score_bias is None:
                dacc_score_bias = cur_bias
            else:
                dacc_score_bias = dacc_score_bias + cur_bias.to(
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

        result = self.e3.select(
            candidates, effective_temperature,
            goal_state=_goal_state_for_select,
            terrain_weight=self._cue_terrain_weight,
            sweep_threshold_reduction=sweep_reduction,
            z_harm_a=z_harm_a,
            score_bias=dacc_score_bias,
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

        # MECH-090: gate policy propagation based on beta state.
        bistable = self.config.heartbeat.beta_gate_bistable
        if bistable:
            # Bistable latch: only elevate on commit ENTRY (not every tick).
            # Release is triggered by hippocampal completion signal in _e3_tick(),
            # not by variance re-evaluation. This prevents flickering when variance
            # hovers near the commit threshold.
            if result.committed and not self.beta_gate.is_elevated:
                self.beta_gate.elevate()
                self._committed_step_idx = 0  # reset step counter on new commitment
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
            if result.committed:
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
            if z_harm_a is not None:
                pag_z_norm = float(z_harm_a.detach().norm().item())
            else:
                pag_z_norm = 0.0
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
            if self._pag_last_output.freeze_active and action is not None:
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
                self.closure_operator.tick(
                    current_z_world=self._current_latent.z_world,
                    current_action_class=action_class,
                    current_mode=current_mode,
                    sd033a_gate=sd033a_gate,
                    hypothesis_tag=False,
                )
            except Exception:
                # Closure detector failure must not break action selection.
                pass

        return action

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

            # MECH-205: populate VALENCE_SURPRISE on residue field
            if self.config.surprise_gated_replay:
                pe_val = e3_metrics.get("prediction_error")
                if pe_val is not None:
                    pe_mag = float(pe_val.detach())
                    self._pe_ema = (1 - self._pe_ema_alpha) * self._pe_ema + self._pe_ema_alpha * pe_mag
                    surprise = max(0.0, pe_mag - self._pe_ema)
                    # Gate: only write genuine surprises above threshold
                    if surprise > self.config.pe_surprise_threshold:
                        # MECH-307 Gap 1: signed VALENCE_SURPRISE write. Sign is
                        # derived from concurrent harm signal -- harm-paired
                        # surprise gets stored as negative (dread-correlate),
                        # non-harm surprise as positive (excitement-correlate).
                        # Backward compat: when the flag is False, store the
                        # unsigned magnitude as before so consumers reading
                        # |VALENCE_SURPRISE| are bit-identical.
                        if getattr(self.config, "use_mech307_signed_pe", False):
                            signed_surprise = -surprise if harm_signal < 0 else surprise
                        else:
                            signed_surprise = surprise
                        self.residue_field.update_valence(
                            z_world, VALENCE_SURPRISE, signed_surprise,
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

    def update_z_goal(self, benefit_exposure: float, drive_level: float = 1.0) -> None:
        """Update z_goal from benefit signal (MECH-112 wanting update).

        SD-015: when ResourceEncoder is enabled and z_resource is populated in the
        current latent state, seeds z_goal from z_resource (object-type latent) rather
        than z_world (full scene latent). z_resource encodes "what kind of resource is
        present" independent of spatial position -- resources respawn at random locations,
        so z_world at contact has no predictive value for future resource locations.

        Args:
            benefit_exposure: scalar benefit this step (obs_body[11] in proxy mode)
            drive_level: homeostatic drive 0=sated, 1=depleted (SD-012).
                         Compute as: 1.0 - float(obs_body[0, 3]) where obs_body[3]=energy.
        """
        if self.goal_state is None or self._current_latent is None:
            return
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

    def run_sws_schema_pass(self) -> Dict[str, float]:
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
        2. For each prototype, construct the full E1 input [z_self_mean, z_world]
           and write it directly to ContextMemory bypassing the offline gate.
           (The offline gate suppresses waking writes; this pass writes offline
           schema content, which is the intended action during SWS.)
        3. Compute slot diversity (mean pairwise cosine distance) as a metric
           for context differentiation quality.

        Args: none (uses self._world_experience_buffer internally)

        Returns:
            dict with keys:
              sws_n_writes: number of prototype writes attempted
              sws_slot_diversity: mean pairwise cosine distance of ContextMemory
                                  slots after pass (higher = more differentiated)
              sws_buffer_size: size of experience buffer used
        """
        metrics: Dict[str, float] = {
            "sws_n_writes": 0.0,
            "sws_slot_diversity": 0.0,
            "sws_buffer_size": 0.0,
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

    def run_sleep_cycle(self) -> Dict[str, float]:
        """
        SD-017: Run a complete SWS->REM sleep cycle.

        Convenience method: calls enter_sws_mode(), run_sws_schema_pass(),
        enter_rem_mode(), run_rem_attribution_pass(), then exit_sleep_mode().

        Returns merged metrics from both passes.

        IMPORTANT: Only call this when sws_enabled=True or rem_enabled=True.
        If both are False this is a no-op returning empty metrics.
        """
        all_metrics: Dict[str, float] = {}
        if not self.config.sws_enabled and not self.config.rem_enabled:
            return all_metrics

        # SWS phase
        if self.config.sws_enabled:
            self.enter_sws_mode()
            sws_metrics = self.run_sws_schema_pass()
            all_metrics.update(sws_metrics)
            self.exit_sleep_mode()

        # REM phase (requires slots installed by SWS; safe to run standalone too)
        if self.config.rem_enabled:
            self.enter_rem_mode()
            rem_metrics = self.run_rem_attribution_pass()
            all_metrics.update(rem_metrics)
            self.exit_sleep_mode()

        return all_metrics

    def get_state(self) -> AgentState:
        return AgentState(
            latent_state=self._current_latent,
            precision=self.e3.current_precision,
            running_variance=self.e3._running_variance,
            step=self._step_count,
            harm_accumulated=self._harm_this_episode,
            is_committed=self.e3._committed_trajectory is not None,
            beta_elevated=self.beta_gate.is_elevated,
            e3_steps_per_tick=self.clock.e3_steps_per_tick,
            serotonin_state=self.serotonin.get_state() if self.serotonin.enabled else None,
        )

    def get_residue_statistics(self) -> Dict[str, torch.Tensor]:
        return self.residue_field.get_statistics()

    def forward(self, observation: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        return self.act(observation, temperature)
