"""
CausalGridWorld V3 — Split Observation Channels (SD-005)

V3 change: observation is split into explicit body_state and world_state channels,
matching the SD-005 z_self/z_world split in the latent stack.

Observation structure (V3, use_proxy_fields=False — default):
  body_state:        position_local (2), health, energy, footprint_density (1),
                     heading (4 one-hot), episode_progress (1) — proprioceptive/interoceptive
                     → fed to z_self encoder
  world_state:       local_view (5×5×7 = 175), contamination_view (5×5 = 25) flattened
                     → fed to z_world encoder

  body_obs_dim  = 10
  world_obs_dim = 175 + 25 = 200

Observation structure (V3, use_proxy_fields=True — CausalGridWorldV2 mode):
  body_state adds: harm_exposure (1), benefit_exposure (1)
                   → nociceptive/hedonic EMA interoceptive channels
  world_state adds: hazard_field_view (5×5 = 25), resource_field_view (5×5 = 25)
                    → proximity gradient fields visible exteroceptively

  body_obs_dim  = 12
  world_obs_dim = 250 (300 when n_landmarks_a > 0 or n_landmarks_b > 0)

  Extra obs_dict keys (not in flat observation):
    harm_obs   [51]: SD-010 sensory-discriminative harm stream (Adelta-pathway analog).
                     hazard_field_view[25] + resource_field_view[25] + harm_exposure[1].
                     Immediate proximity; forward-predictable from action (HarmForwardModel).
    harm_obs_a [50]: SD-011 affective-motivational harm stream (C-fiber analog).
                     EMA of proximity fields at tau~20 steps. Accumulated homeostatic
                     deviation. NOT forward-predicted -- feeds E3 directly (ARC-033).
    landmark_a_field_view [25]: SD-023 Landmark A gradient field view (when n_landmarks_a>0).
    landmark_b_field_view [25]: SD-023 Landmark B gradient field view (when n_landmarks_b>0).

Proxy-gradient rationale (ARC-024 / INV-025-029):
  Harm and benefit signals are proxies along gradients pointing toward asymptotic
  limits (death / complete union) that are unreachable from within experience.
  A world that generates harm only at contact models the endpoint, not the gradient.
  CausalGridWorldV2 (use_proxy_fields=True) generates observable gradient fields
  that precede contact events, allowing E2.world_forward to learn action-conditional
  dynamics and E3.harm_eval to learn gradient-approach patterns.

Ground truth transition_type (for V3-EXQ-002, SD-003) is preserved.
New types (proxy mode only): "hazard_approach", "benefit_approach"

Sub-goal mode preserved unchanged from V2.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class CausalGridWorld:
    """
    2D grid world with persistent agent causal footprint — V3.

    V3 key change: _get_observation() returns a dict AND a flat tensor.
    The dict has:
      "body_state":   [body_obs_dim]  — proprioceptive channels → z_self
      "world_state":  [world_obs_dim] — exteroceptive channels  → z_world
      "contamination_view": [25]      — subset of world_state (for convenience)

    With use_proxy_fields=True (CausalGridWorldV2 mode):
      "hazard_field_view":   [25] — proximity gradient (subset of world_state)
      "resource_field_view": [25] — resource gradient (subset of world_state)

    Actions (unchanged from V2):
        0: up, 1: down, 2: left, 3: right, 4: stay

    Transition types (ground truth for SD-003 attribution):
        "agent_caused_hazard", "env_caused_hazard", "resource", "none"
        "waypoint", "sequence_complete" (subgoal_mode only)
        "hazard_approach", "benefit_approach" (use_proxy_fields=True only)
    """

    ACTIONS: Dict[int, Tuple[int, int]] = {
        0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0),
    }

    ENTITY_TYPES: Dict[str, int] = {
        "empty": 0, "wall": 1, "resource": 2, "hazard": 3,
        "contaminated": 4, "agent": 5, "waypoint": 6,
    }
    NUM_ENTITY_TYPES = 7

    def __init__(
        self,
        size: int = 10,
        num_hazards: int = 3,
        num_resources: int = 5,
        contamination_spread: float = 0.5,
        contamination_threshold: float = 2.0,
        env_drift_interval: int = 5,
        env_drift_prob: float = 0.3,
        hazard_harm: float = 0.5,
        contaminated_harm: float = 0.4,
        resource_benefit: float = 0.3,
        energy_decay: float = 0.01,
        seed: Optional[int] = None,
        # Sub-goal mode (preserved from V2 for MECH-057a)
        subgoal_mode: bool = False,
        num_waypoints: int = 3,
        waypoint_visit_reward: float = 0.2,
        waypoint_completion_reward: float = 0.8,
        sequence_commitment_timeout: int = 20,
        # Proxy-gradient field mode (ARC-024, CausalGridWorldV2)
        use_proxy_fields: bool = False,
        # Toroidal wrapping: no walls, movement wraps at grid edges
        toroidal: bool = False,
        # SD-012: resource respawn for repeated drive-reduction cycles
        resource_respawn_on_consume: bool = False,
        hazard_field_decay: float = 0.5,
        resource_field_decay: float = 0.5,
        proximity_harm_scale: float = 0.05,
        proximity_benefit_scale: float = 0.03,
        nociception_ema_alpha: float = 0.1,
        harm_obs_a_ema_alpha: float = 0.05,
        proximity_approach_threshold: float = 0.15,
        # SD-011 second source: rolling window of past harm_exposure values.
        # 0 = disabled (backward compat). When > 0, obs_dict includes
        # "harm_history" [harm_history_len] and "accumulated_harm" scalar.
        harm_history_len: int = 0,
        # SD-022: directional limb damage.
        # Each of 4 directional limbs accumulates tissue damage from hazard transits.
        # Provides causal independence between z_harm_a (body state) and z_harm_s (world).
        # Disabled by default for backward compatibility.
        limb_damage_enabled: bool = False,
        damage_increment: float = 0.15,
        residual_pain_scale: float = 0.5,
        failure_prob_scale: float = 0.3,
        heal_rate: float = 0.002,
        residual_pain_threshold: float = 0.05,
        # SD-023: environmental gradient texture -- landmark objects.
        # Landmark A ("pillar"): navigation anchor, placed randomly. Strong short-range gradient.
        # Landmark B ("trace"): predictive resource cue, biased near resources. Weaker medium-range.
        # Both default to 0 (disabled) for backward compatibility.
        # world_obs_dim grows by 50 (25+25) when n_landmarks_a > 0 or n_landmarks_b > 0.
        n_landmarks_a: int = 0,
        n_landmarks_b: int = 0,
        landmark_a_sigma: float = 1.5,
        landmark_a_scale: float = 1.0,
        landmark_b_sigma: float = 2.5,
        landmark_b_scale: float = 0.6,
        landmark_b_resource_bias: float = 0.7,
        # EXQ-332: resource observation visibility scale.
        # Multiplies resource_field_view in world_state only (not benefit calculation).
        # 1.0 = full visibility (default, backward compat); 0.3 = strongly attenuated.
        # Forces agent to use landmark B rather than reactive resource gradient detection.
        resource_obs_scale: float = 1.0,
        # SD-029: balanced-hazard-event curriculum.
        # When enabled, scheduled "external hazard injection" events spawn a
        # hazard at a cell adjacent to the agent every scheduled_external_hazard_interval
        # steps (with probability scheduled_external_hazard_prob). Purpose: guarantee
        # both self-caused and externally-caused hazard events per seed for event-
        # conditioned comparator SNR (SD-029 C3/C4). Curriculum-level fix; not a
        # substrate change to agent or latents. Disabled by default (backward compat).
        scheduled_external_hazard_enabled: bool = False,
        scheduled_external_hazard_interval: int = 50,
        scheduled_external_hazard_prob: float = 0.5,
        scheduled_external_hazard_adjacent_only: bool = True,
        # SD-047: multi-source environmental dynamics.
        # Three concurrent stochastic event sources at distinct spatial / temporal scales,
        # each agent-independent. Provides the textured causal background that
        # agency-detection comparators (MECH-095) and reafference cancellation
        # (MECH-098) require for honest substrate-level testing. Bit-identical OFF
        # when multi_source_dynamics_enabled=False (default); per-source switches
        # allow independent ablation.
        #
        #   Source 1 (weather field): AR(1) coarse-grid additive perturbation on
        #     hazard_field. Continuous, smooth, autocorrelated, agent-independent.
        #   Source 2 (transient events): Poisson appear / disappear of transient
        #     hazard cells (tracked separately from self.hazards so they auto-decay).
        #   Source 3 (background drift): n_drift_sources mobile single-cell hazard-
        #     analog objects with independent random-walk / drift / Levy dynamics.
        #
        # multi_source_intensity_scale is the 4-arm noise-sweep lever per the SD-047
        # validation protocol (Asai 2016 non-monotonic): scales sigma, p_appear,
        # and drift step rate uniformly so a single knob produces ARM_0/1/2/3.
        multi_source_dynamics_enabled: bool = False,
        multi_source_intensity_scale: float = 1.0,
        weather_field_enabled: bool = False,
        weather_super_cells: int = 4,
        weather_alpha_ar1: float = 0.95,
        weather_sigma: float = 0.05,
        transient_events_enabled: bool = False,
        transient_p_appear: float = 1e-3,
        transient_p_disappear: float = 0.1,
        transient_intensity: float = 1.0,
        background_drift_enabled: bool = False,
        n_drift_sources: int = 1,
        drift_policy: str = "random_walk",
        # SD-048: interoceptive noise dynamics. Three concurrent agent-independent
        # stochastic body-state noise sources applied to harm_obs_a so the Level 2
        # interoceptive forward-model comparator (ARC-058 / ARC-033 path) has a
        # body-noise calibration background to discriminate from agent-caused
        # body-state change. Mirrors SD-047 architecturally at the body-state
        # readout layer. Bit-identical OFF when interoceptive_noise_enabled=False
        # (default); per-source switches allow independent ablation.
        #
        #   Source 1 (autonomic noise): per-step i.i.d. Gaussian additive noise on
        #     harm_obs_a. Fast, continuous, low-amplitude (HRV / sympathetic-
        #     fluctuation analog).
        #   Source 2 (sensitisation spikes): Poisson onset of transient
        #     multiplicative amplification on harm_obs_a, exponentially decaying
        #     (inflammatory sensitisation analog).
        #   Source 3 (fatigue drift): slow AR(1) latent fatigue state additively
        #     contributing to harm_obs_a across the episode (allostatic-load
        #     analog). Resets per episode.
        #
        # interoceptive_noise_scale is the 4-arm sweep lever per the SD-048
        # validation protocol (Asai 2016 non-monotonic comparator competence):
        # scales autonomic_noise_scale, sensitisation_rate, and
        # fatigue_noise_scale uniformly so a single knob produces ARM_0/1/2/3.
        # interoceptive_change_threshold defines the |delta_harm_obs_a| floor
        # used to count body-noise-caused vs agent-caused harm-state-change
        # events (the SD doc 1:1-2:1 calibration target).
        interoceptive_noise_enabled: bool = False,
        interoceptive_noise_scale: float = 1.0,
        autonomic_noise_enabled: bool = True,
        autonomic_noise_scale: float = 0.02,
        sensitisation_enabled: bool = True,
        sensitisation_rate: float = 0.008,
        sensitisation_magnitude: float = 1.8,
        sensitisation_halflife: int = 15,
        fatigue_enabled: bool = True,
        fatigue_ar_coeff: float = 0.995,
        fatigue_noise_scale: float = 0.005,
        fatigue_contribution_weight: float = 0.15,
        interoceptive_change_threshold: float = 0.01,
        # SD-049: multi-resource heterogeneity. Three additions to the env, layered
        # on top of SD-012's homeostatic drive (the substrate that GoalState consumes).
        # Master switch: multi_resource_heterogeneity_enabled=False -> bit-identical
        # to legacy single-anonymous-resource behaviour. When enabled:
        #   (1) num_resources cells split across n_resource_types qualitatively
        #       distinct types (default 3: food / water / novelty). Each cell
        #       carries an identity tag; per-type proximity field views appended
        #       to world_obs (world_obs_dim grows by n_resource_types * 25).
        #   (2) Per-axis homeostatic drive vector (per_axis_drive[n_axes]) tracked
        #       alongside legacy agent_energy. agent_energy collapses to
        #       1.0 - max(per_axis_drive) when per_axis_drive_enabled so all
        #       legacy SD-032 consumers (AIC / PCC / pACC / dACC / salience /
        #       override) continue to read obs_body[3] without modification.
        #       New observable: obs_dict["per_axis_drive"] for new experiments.
        #   (3) resource_introduction_schedule controls per-type spawn availability
        #       by global step count. Defaults to None -> all types available
        #       from step 0 (existing-experiment-equivalent behaviour even when
        #       master switch is on).
        # Per-resource-type bit-identical OFF: setting an entry to 0.0 in
        # resource_type_distribution drops that type without code change
        # (recovers ARM_1 from ARM_2 in the validation 4-arm sweep).
        # Lit-pull provenance: REE_assembly/evidence/literature/targeted_review_sd_049/
        multi_resource_heterogeneity_enabled: bool = False,
        n_resource_types: int = 3,
        resource_type_names: tuple = ("food", "water", "novelty"),
        resource_type_drive_axes: tuple = ("hunger", "thirst", "curiosity"),
        resource_type_benefit_curves: tuple = (
            "sigmoidal_saturating",
            "sharp_saturation",
            "novelty_decay",
        ),
        resource_type_distribution: Optional[tuple] = None,
        resource_type_benefit_amplitudes: Optional[tuple] = None,
        per_axis_drive_enabled: bool = False,
        per_axis_drive_decay: tuple = (0.001, 0.0015, 0.0005),
        per_axis_drive_combiner: str = "max",
        novelty_familiarity_increment: float = 0.2,
        novelty_familiarity_recovery: float = 0.0,
        resource_introduction_schedule: Optional[Dict[str, int]] = None,
        # Behavioral diversity substrate: reef safe zones + food-attracted hazards.
        # reef_enabled: hazards excluded from reef cells; reef cells excluded from
        # hazard/resource spawn; agent takes zero harm while in a reef cell.
        # hazard_food_attraction: per-drift-tick probability that a hazard biases its
        # random walk toward the nearest food cell instead of pure random shuffle.
        # reef_scent_sigma: Manhattan-distance decay scale for reef gradient field.
        reef_enabled: bool = False,
        n_reef_patches: int = 3,
        reef_patch_radius: int = 2,
        reef_scent_sigma: float = 2.5,
        hazard_food_attraction: float = 0.0,
        # SD-054 bipartite layout extension (2026-05-11).
        # When reef_bipartite_layout=True, reef cells are clustered in one half of
        # the grid (along reef_bipartite_axis) and food / hazards spawn ONLY in the
        # opposite half. Agent spawns in a band of width 2*agent_band_radius + 1
        # centered on the midline. The geometric structure forces reef-approach
        # vs forage-approach trajectories to have categorically different
        # first-action argmaxes by construction, addressing the V3-EXQ-543b
        # CEM-candidate-distinguishability bottleneck (TASK_CLAIMS session
        # diagnose-v3-exq-543c-2026-05-11T0635Z; arc_062_rule_apprehension_plan.md
        # decision-log entry 2026-05-11 option 3a).
        # reef_bipartite_axis: "horizontal" -> reef bottom rows, food top rows
        #                      "vertical"   -> reef right columns, food left columns
        # reef_bipartite_agent_band_radius: half-width of the agent spawn band
        #                                   measured from the midline (inclusive).
        #                                   radius=0 -> midline only (1 row/col).
        #                                   radius=1 -> midline +/- 1 (3 rows/cols).
        #                                   radius=2 -> midline +/- 2 (5 rows/cols).
        # All defaults preserve bit-identical legacy SD-054 behavior.
        reef_bipartite_layout: bool = False,
        reef_bipartite_axis: str = "horizontal",
        reef_bipartite_agent_band_radius: int = 1,
        # infant_substrate:GAP-1 -- harm gradient env feature.
        # Graduated harm proximity reward proportional to distance from nearest
        # hazard. Fires when no direct-contact / approach transition has fired
        # (transition_type == "none"). Pure reward signal; no health deduction.
        # Reward: -hazard_harm * (1 - d/r_outer)**2 * scale
        # for inner_radius < d <= outer_radius.
        # Env-only; not surfaced through REEConfig.from_dims. All defaults
        # are no-op when disabled (bit-identical legacy behaviour).
        harm_gradient_enabled: bool = False,
        harm_gradient_outer_radius: float = 3.0,
        harm_gradient_inner_radius: float = 0.0,
        harm_gradient_scale: float = 1.0,
        # infant_substrate:GAP-2 -- microhabitat zones env feature.
        # Per-episode Voronoi-seeded zone map (A/B/C + automatic D border)
        # modulating per-cell resource/hazard spawn weighting, plus a
        # zone-C ambient presence bonus that decays with repeated zone-C
        # visits. Env-only; not surfaced through REEConfig.from_dims.
        # All defaults are no-op when disabled (bit-identical legacy
        # behaviour: no zone map, unweighted pop(), zero new RNG draws).
        # Spec: REE_assembly/docs/architecture/infant_substrate_expansion.md
        # Section 5.2.
        microhabitat_enabled: bool = False,
        n_microhabitats: int = 3,
        zone_A_resource_factor: float = 1.5,
        zone_A_hazard_factor: float = 0.3,
        zone_B_resource_factor: float = 0.8,
        zone_B_hazard_factor: float = 1.8,
        zone_C_resource_factor: float = 0.3,
        zone_C_hazard_factor: float = 0.0,
        zone_C_ambient_bonus: float = 0.05,
        zone_novelty_decay: float = 0.95,
        # infant_substrate:GAP-3 -- transient benefit patches env feature.
        # Stochastic high-salience benefit patch spawn for z_goal seeding.
        # Each step, with probability transient_benefit_prob, a benefit
        # patch is placed at a (zone-weighted, when microhabitat zones are
        # active) random empty interior cell. Patches expire after
        # transient_benefit_duration steps. Contact reward =
        # resource_benefit * transient_benefit_multiplier. Env-only; not
        # surfaced through REEConfig.from_dims. All defaults are no-op when
        # disabled (bit-identical legacy behaviour: no patches, no extra
        # RNG draws). Spec: REE_assembly/docs/architecture/
        # infant_substrate_expansion.md Section 5.3.
        transient_benefit_enabled: bool = False,
        transient_benefit_prob: float = 0.02,
        transient_benefit_duration: int = 15,
        transient_benefit_multiplier: float = 2.0,
        # infant_substrate:GAP-5 -- H_pos / zone_coverage telemetry.
        # Always-present info-dict keys: pos_entropy (Shannon entropy in
        # nats of the agent position histogram over a rolling
        # pos_entropy_window) and zone_coverage ({zone: fraction of that
        # zone's cells visited this episode}, consuming the GAP-2
        # _zone_map; single stub zone 0 = whole interior when microhabitat
        # is disabled). Unlike GAP-1/2/3 and SD-047/48/49 this telemetry
        # has NO RNG and never feeds back into env/agent/obs dynamics, so
        # results are bit-identical whether ON or OFF -- there is nothing
        # to be "bit-identical OFF" about. It is a DEV-NEED-001 blocking
        # gate, so it defaults ON: experiments get H_pos / zone_coverage
        # without having to flip a flag. The master switch is retained so
        # the contract OFF path and zero-overhead runs can disable it.
        # Env-only; not surfaced through REEConfig.from_dims. Spec:
        # REE_assembly/evidence/planning/infant_substrate_plan.md GAP-5.
        pos_telemetry_enabled: bool = True,
        pos_entropy_window: int = 100,
        zone_coverage_stub_single_zone: bool = True,
    ):
        self.size = size
        self.num_hazards = num_hazards
        self.num_resources = num_resources
        self.contamination_spread = contamination_spread
        self.contamination_threshold = contamination_threshold
        self.env_drift_interval = env_drift_interval
        self.env_drift_prob = env_drift_prob
        self.hazard_harm = hazard_harm
        self.contaminated_harm = contaminated_harm
        self.resource_benefit = resource_benefit
        self.energy_decay = energy_decay

        self.subgoal_mode = subgoal_mode
        self.num_waypoints = num_waypoints
        self.waypoint_visit_reward = waypoint_visit_reward
        self.waypoint_completion_reward = waypoint_completion_reward
        self.sequence_commitment_timeout = sequence_commitment_timeout

        # Proxy-gradient parameters
        self.use_proxy_fields = use_proxy_fields
        self.toroidal = toroidal
        self.resource_respawn_on_consume = resource_respawn_on_consume
        self.hazard_field_decay = hazard_field_decay
        self.resource_field_decay = resource_field_decay
        self.proximity_harm_scale = proximity_harm_scale
        self.proximity_benefit_scale = proximity_benefit_scale
        self.nociception_ema_alpha = nociception_ema_alpha
        self.harm_obs_a_ema_alpha = harm_obs_a_ema_alpha
        self.proximity_approach_threshold = proximity_approach_threshold
        self.harm_history_len = harm_history_len

        # SD-022: directional limb damage state
        self.limb_damage_enabled = limb_damage_enabled
        self.damage_increment = damage_increment
        self.residual_pain_scale = residual_pain_scale
        self.failure_prob_scale = failure_prob_scale
        self.heal_rate = heal_rate
        self.residual_pain_threshold = residual_pain_threshold
        # [N=0, E=1, S=2, W=3] -- initialized here, reset in reset() when enabled.
        self.limb_damage: np.ndarray = np.zeros(4, dtype=np.float32)

        # SD-023: environmental gradient texture -- landmark objects.
        self.n_landmarks_a = n_landmarks_a
        self.n_landmarks_b = n_landmarks_b
        self.landmark_a_sigma = landmark_a_sigma
        self.landmark_a_scale = landmark_a_scale
        self.landmark_b_sigma = landmark_b_sigma
        self.landmark_b_scale = landmark_b_scale
        self.landmark_b_resource_bias = landmark_b_resource_bias
        self.resource_obs_scale = resource_obs_scale
        # SD-029: balanced-hazard-event curriculum state
        self.scheduled_external_hazard_enabled = scheduled_external_hazard_enabled
        self.scheduled_external_hazard_interval = scheduled_external_hazard_interval
        self.scheduled_external_hazard_prob = scheduled_external_hazard_prob
        self.scheduled_external_hazard_adjacent_only = scheduled_external_hazard_adjacent_only
        self._external_hazard_event_count: int = 0
        self._last_external_hazard_step: int = -1

        # SD-047: multi-source environmental dynamics state.
        self.multi_source_dynamics_enabled = multi_source_dynamics_enabled
        self.multi_source_intensity_scale = float(multi_source_intensity_scale)
        self.weather_field_enabled = weather_field_enabled
        self.weather_super_cells = int(max(1, weather_super_cells))
        self.weather_alpha_ar1 = float(weather_alpha_ar1)
        self.weather_sigma = float(weather_sigma)
        self.transient_events_enabled = transient_events_enabled
        self.transient_p_appear = float(transient_p_appear)
        self.transient_p_disappear = float(transient_p_disappear)
        self.transient_intensity = float(transient_intensity)
        self.background_drift_enabled = background_drift_enabled
        self.n_drift_sources = int(max(0, n_drift_sources))
        self.drift_policy = str(drift_policy)
        # Coarse AR(1) state and full-grid additive perturbation; populated in reset().
        self._weather_super_field: np.ndarray = np.zeros(
            (self.weather_super_cells, self.weather_super_cells), dtype=np.float32
        )
        self._weather_perturbation: np.ndarray = np.zeros((size, size), dtype=np.float32)
        # Transient hazards: List[List[int]] of [x, y, age]. Distinct from self.hazards.
        self._transient_hazards: List[List[int]] = []
        # Drift sources: List[List[int]] of [x, y, vx, vy, age]. vx/vy used by linear/Levy.
        self._drift_sources: List[List[int]] = []
        # Per-step diagnostic counters (env-caused vs agent-caused change events).
        # Reset every step in step(); incremented inside step paths.
        self._multi_source_n_env_events: int = 0
        self._multi_source_n_agent_events: int = 0

        # SD-048: interoceptive noise dynamics state.
        self.interoceptive_noise_enabled = interoceptive_noise_enabled
        self.interoceptive_noise_scale = float(interoceptive_noise_scale)
        self.autonomic_noise_enabled = autonomic_noise_enabled
        self.autonomic_noise_scale = float(autonomic_noise_scale)
        self.sensitisation_enabled = sensitisation_enabled
        self.sensitisation_rate = float(sensitisation_rate)
        self.sensitisation_magnitude = float(sensitisation_magnitude)
        self.sensitisation_halflife = int(max(1, sensitisation_halflife))
        self.fatigue_enabled = fatigue_enabled
        self.fatigue_ar_coeff = float(np.clip(fatigue_ar_coeff, 0.0, 0.99999))
        self.fatigue_noise_scale = float(fatigue_noise_scale)
        self.fatigue_contribution_weight = float(fatigue_contribution_weight)
        self.interoceptive_change_threshold = float(interoceptive_change_threshold)
        # Cached transition_type from step() so _apply_interoceptive_noise can
        # classify agent-caused vs body-noise-caused delta events. "none" before
        # the first step() of an episode (no agent action observed yet).
        self._last_transition_type: str = "none"
        # AR(1) latent fatigue (resets per episode, see reset()).
        self._fatigue_state: float = 0.0
        # Active multiplicative amplification from sensitisation events
        # (sum of exponentially-decaying contributions; resets per episode).
        self._sensitisation_amplification: float = 0.0
        # Previous-tick harm_obs_a snapshot for delta-event detection
        # (resets per episode; first tick of an episode counts no events).
        self._prev_harm_obs_a: Optional[np.ndarray] = None
        # Per-tick diagnostic counters (incremented inside _apply_interoceptive_noise).
        self._interoceptive_n_autonomic_events: int = 0
        self._interoceptive_n_sensitisation_events: int = 0
        self._interoceptive_n_fatigue_events: int = 0
        self._interoceptive_n_body_noise_events: int = 0
        self._interoceptive_n_agent_caused_harm_events: int = 0

        # SD-049: multi-resource heterogeneity state. Validated and normalised
        # at construction so reset() / step() can rely on consistent shapes.
        self.multi_resource_heterogeneity_enabled = bool(multi_resource_heterogeneity_enabled)
        self.n_resource_types = int(max(1, n_resource_types))
        # Truncate / pad name + axis + curve tuples to n_resource_types.
        # Defaults cover types 0..2; types 3+ get generic "type_<i>" / "axis_<i>"
        # / "sigmoidal_saturating" entries (used by ARM_3 5-type overshoot).
        def _padded(seq, default_factory):
            seq = list(seq)
            while len(seq) < self.n_resource_types:
                seq.append(default_factory(len(seq)))
            return tuple(seq[: self.n_resource_types])
        self.resource_type_names = _padded(
            resource_type_names, lambda i: f"type_{i}"
        )
        self.resource_type_drive_axes = _padded(
            resource_type_drive_axes, lambda i: f"axis_{i}"
        )
        self.resource_type_benefit_curves = _padded(
            resource_type_benefit_curves, lambda i: "sigmoidal_saturating"
        )
        # Distribution: defaults to uniform when None. Negative entries clipped to 0.
        # All-zero distribution falls back to uniform to avoid spawn-zero pathology
        # (an all-zero distribution disables SD-049 in spirit; use master switch instead).
        if resource_type_distribution is None:
            dist = [1.0] * self.n_resource_types
        else:
            dist = list(resource_type_distribution)
            while len(dist) < self.n_resource_types:
                dist.append(0.0)
            dist = [max(0.0, float(d)) for d in dist[: self.n_resource_types]]
        if sum(dist) <= 0.0:
            dist = [1.0] * self.n_resource_types
        self.resource_type_distribution = tuple(dist)
        # Per-type benefit amplitudes scale the contact-side restoration on the
        # matching drive axis (uniform 1.0 default; allows tuning without changing
        # the curve choice).
        if resource_type_benefit_amplitudes is None:
            amps = [1.0] * self.n_resource_types
        else:
            amps = list(resource_type_benefit_amplitudes)
            while len(amps) < self.n_resource_types:
                amps.append(1.0)
            amps = [float(a) for a in amps[: self.n_resource_types]]
        self.resource_type_benefit_amplitudes = tuple(amps)
        # Per-axis drive vector + decay rates (per axis index 0..n_resource_types-1).
        self.per_axis_drive_enabled = bool(per_axis_drive_enabled)
        decay = list(per_axis_drive_decay)
        while len(decay) < self.n_resource_types:
            decay.append(decay[-1] if decay else 0.001)
        self.per_axis_drive_decay = tuple(
            float(max(0.0, d)) for d in decay[: self.n_resource_types]
        )
        if per_axis_drive_combiner not in ("max", "mean", "sum"):
            per_axis_drive_combiner = "max"
        self.per_axis_drive_combiner = per_axis_drive_combiner
        # Novelty per-cell familiarity dynamics (used only when
        # any benefit curve is "novelty_decay"; harmless arithmetic otherwise).
        self.novelty_familiarity_increment = float(novelty_familiarity_increment)
        self.novelty_familiarity_recovery = float(max(0.0, novelty_familiarity_recovery))
        # Curriculum hook: dict mapping resource_type_name -> step at which type
        # becomes available. Types not listed are available from step 0.
        # Step counter is _global_step (cross-episode), see init below.
        self.resource_introduction_schedule = dict(resource_introduction_schedule or {})
        # Per-type state (allocated even when master switch off so type checks
        # never crash on missing attrs; stays empty / zero when disabled).
        # Per-type resource positions: list of [x, y] for each type.
        self._resources_by_type: List[List[List[int]]] = [[] for _ in range(self.n_resource_types)]
        # Per-cell type index grid (0 = no resource; type_idx + 1 elsewhere) for fast lookup.
        # Allocated in reset() with correct shape.
        self._resource_type_grid: np.ndarray = np.zeros((size, size), dtype=np.int8)
        # Per-type proximity fields (parallel to legacy resource_field).
        self._resource_field_by_type: np.ndarray = np.zeros(
            (self.n_resource_types, size, size), dtype=np.float32
        )
        # Per-axis homeostatic drive vector. Drive axis i corresponds to resource
        # type i (1:1 mapping at this stage; future MECH may decouple via a
        # learned mapping). per_axis_drive[i] = depletion in [0, 1]; 0 = sated,
        # 1 = fully depleted. Reset per-episode (homeostatic state is episode-local
        # in V3 -- cross-episode allostatic load is V4 work).
        self._per_axis_drive: np.ndarray = np.zeros(self.n_resource_types, dtype=np.float32)
        # Per-cell novelty familiarity counter (used only when novelty_decay curve
        # is active). Bounded [0, 1] via clip on increment.
        self._novelty_familiarity: np.ndarray = np.zeros((size, size), dtype=np.float32)
        # Cross-episode global step counter for curriculum hook. Persists across
        # reset() so resource_introduction_schedule references are interpretable.
        self._global_step: int = 0
        # Per-tick diagnostic counters reset every step in step(); incremented inside
        # the SD-049 paths.
        self._sd049_n_resource_contacts_by_type: np.ndarray = np.zeros(
            self.n_resource_types, dtype=np.int32
        )
        self._sd049_n_axis_depletion_steps: int = 0

        # Behavioral diversity substrate: reef safe zones + food-attracted hazards.
        self.reef_enabled = bool(reef_enabled)
        self.n_reef_patches = int(max(0, n_reef_patches))
        self.reef_patch_radius = int(max(1, reef_patch_radius))
        self.reef_scent_sigma = float(max(0.1, reef_scent_sigma))
        self.hazard_food_attraction = float(np.clip(hazard_food_attraction, 0.0, 1.0))
        # SD-054 bipartite layout extension (2026-05-11). See __init__ docstring
        # above the kwargs and the SD-054 doc for the rationale.
        self.reef_bipartite_layout = bool(reef_bipartite_layout)
        if reef_bipartite_axis not in ("horizontal", "vertical"):
            raise ValueError(
                f"reef_bipartite_axis must be 'horizontal' or 'vertical', "
                f"got {reef_bipartite_axis!r}"
            )
        self.reef_bipartite_axis = reef_bipartite_axis
        self.reef_bipartite_agent_band_radius = int(
            max(0, reef_bipartite_agent_band_radius)
        )
        # Per-reset diagnostic; populated by _build_bipartite_pools() when bipartite
        # is active. 0 in legacy mode and in successful bipartite resets.
        self._sd054_bipartite_band_widen_count: int = 0
        # Populated by _place_reef_patches() each reset(); empty set / zero field when OFF.
        self._reef_cells: set = set()
        self._reef_field: np.ndarray = np.zeros((size, size), dtype=np.float32)

        # infant_substrate:GAP-1 -- harm gradient env feature.
        self.harm_gradient_enabled = bool(harm_gradient_enabled)
        self.harm_gradient_outer_radius = float(max(0.0, harm_gradient_outer_radius))
        self.harm_gradient_inner_radius = float(max(0.0, harm_gradient_inner_radius))
        self.harm_gradient_scale = float(harm_gradient_scale)

        # infant_substrate:GAP-2 -- microhabitat zones env feature.
        self.microhabitat_enabled = bool(microhabitat_enabled)
        self.n_microhabitats = int(max(1, n_microhabitats))
        # Zone factor lookup: index 0=A, 1=B, 2=C. Voronoi zones beyond the
        # first 3 (when n_microhabitats > 3) and the automatic D border zone
        # get neutral 1.0/1.0 factors and no ambient bonus, mirroring the
        # SD-049 "extra types get generic entries" precedent.
        self.zone_A_resource_factor = float(max(0.0, zone_A_resource_factor))
        self.zone_A_hazard_factor = float(max(0.0, zone_A_hazard_factor))
        self.zone_B_resource_factor = float(max(0.0, zone_B_resource_factor))
        self.zone_B_hazard_factor = float(max(0.0, zone_B_hazard_factor))
        self.zone_C_resource_factor = float(max(0.0, zone_C_resource_factor))
        self.zone_C_hazard_factor = float(max(0.0, zone_C_hazard_factor))
        self.zone_C_ambient_bonus = float(zone_C_ambient_bonus)
        self.zone_novelty_decay = float(np.clip(zone_novelty_decay, 0.0, 1.0))
        # Zone codes: -1 = wall / non-interior, 0 = A, 1 = B, 2 = C,
        # 3 = D (transition/border, neutral), 4+ = extra Voronoi zone (neutral).
        self._zone_map: Optional[np.ndarray] = None
        self._zone_c_visit_count: int = 0
        self._zone_c_ambient_this_tick: float = 0.0

        # infant_substrate:GAP-3 -- transient benefit patches env feature.
        self.transient_benefit_enabled = bool(transient_benefit_enabled)
        self.transient_benefit_prob = float(
            np.clip(transient_benefit_prob, 0.0, 1.0)
        )
        self.transient_benefit_duration = int(max(1, transient_benefit_duration))
        self.transient_benefit_multiplier = float(
            max(0.0, transient_benefit_multiplier)
        )
        # Active patches: list of [x, y, expiry_step]. Parallel set of
        # (x, y) for O(1) contact lookup. Both empty when disabled.
        self._transient_benefits: List[List[int]] = []
        self._transient_benefit_cells: set = set()
        # Per-episode diagnostics (reset in reset()).
        self._transient_benefit_n_spawned: int = 0
        self._transient_benefit_n_contacted: int = 0
        self._transient_benefit_n_expired: int = 0
        self._transient_benefit_contact_this_tick: float = 0.0

        # infant_substrate:GAP-5 -- H_pos / zone_coverage telemetry.
        self.pos_telemetry_enabled = bool(pos_telemetry_enabled)
        self.pos_entropy_window = int(max(1, pos_entropy_window))
        self.zone_coverage_stub_single_zone = bool(zone_coverage_stub_single_zone)
        # Rolling window of recent (x, y) positions (most recent last,
        # capped at pos_entropy_window) and the set of all cells visited
        # this episode. Both per-episode; reset by _reset_pos_telemetry().
        self._pos_window: List[Tuple[int, int]] = []
        self._visited_cells: set = set()

        # Fields and positions initialized in reset().
        self.landmark_a_positions: List[Tuple[int, int]] = []
        self.landmark_b_positions: List[Tuple[int, int]] = []
        self._landmark_a_field: np.ndarray = np.zeros((size, size), dtype=np.float32)
        self._landmark_b_field: np.ndarray = np.zeros((size, size), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        # SD-011: harm_obs_a_ema persists across episodes (homeostatic accumulator).
        # Initialized here, NOT in reset(), so accumulated threat state carries over.
        # Resetting per-episode destroys autocorrelation (EXQ-106 C4 FAIL root cause).
        self.harm_obs_a_ema: np.ndarray = np.zeros(50, dtype=np.float32)
        # SD-011 second source: rolling harm history buffer (FIFO of harm_exposure).
        # Persists across episodes like harm_obs_a_ema (same rationale).
        if self.harm_history_len > 0:
            self._harm_history = np.zeros(self.harm_history_len, dtype=np.float32)
            self._harm_history_ptr = 0
            self._accumulated_harm_exposure = 0.0
            self._accumulated_harm_steps = 0
        self.reset()

    # ------------------------------------------------------------------ #
    # Dimension properties                                                 #
    # ------------------------------------------------------------------ #

    @property
    def body_obs_dim(self) -> int:
        """SD-005 body (proprioceptive/interoceptive) observation dimension."""
        # position_local (2) + health (1) + energy (1) + footprint_density (1)
        # + heading one-hot (4) + episode_progress (1) = 10
        # + harm_exposure (1) + benefit_exposure (1) = 12 (proxy mode)
        # SD-022: + damage[4] + residual_pain (1) = 17 (proxy mode + limb_damage_enabled)
        if self.use_proxy_fields and self.limb_damage_enabled:
            return 17
        return 12 if self.use_proxy_fields else 10

    @property
    def world_obs_dim(self) -> int:
        """SD-005 world (exteroceptive) observation dimension."""
        local_view_dim = 5 * 5 * self.NUM_ENTITY_TYPES  # 175
        contamination_view_dim = 5 * 5                   # 25
        base = local_view_dim + contamination_view_dim   # 200
        if self.use_proxy_fields:
            hazard_field_dim = 5 * 5                     # 25
            resource_field_dim = 5 * 5                   # 25
            proxy_base = base + hazard_field_dim + resource_field_dim  # 250
            # SD-023: landmark gradient texture -- 25 dims per landmark type when enabled.
            if self.n_landmarks_a > 0 or self.n_landmarks_b > 0:
                proxy_base = proxy_base + 25 + 25        # 300
            # SD-049: per-resource-type proximity field views appended when
            # multi_resource_heterogeneity is on. Adds n_resource_types * 25 dims.
            if self.multi_resource_heterogeneity_enabled:
                proxy_base = proxy_base + self.n_resource_types * 25
            # Behavioral diversity: reef scent gradient field view (+25 dims when enabled).
            if self.reef_enabled:
                proxy_base = proxy_base + 25
            return proxy_base
        return base                                      # 200

    @property
    def observation_dim(self) -> int:
        """Total flat observation dimension (body + world)."""
        return self.body_obs_dim + self.world_obs_dim

    @property
    def action_dim(self) -> int:
        return len(self.ACTIONS)

    # ------------------------------------------------------------------ #
    # Reset                                                                #
    # ------------------------------------------------------------------ #

    def reset(self) -> Tuple[torch.Tensor, Dict]:
        """Reset environment. Returns (flat_obs, obs_dict)."""
        self.grid = np.zeros((self.size, self.size), dtype=np.int32)
        if not self.toroidal:
            self.grid[0, :] = self.ENTITY_TYPES["wall"]
            self.grid[-1, :] = self.ENTITY_TYPES["wall"]
            self.grid[:, 0] = self.ENTITY_TYPES["wall"]
            self.grid[:, -1] = self.ENTITY_TYPES["wall"]

        self.contamination_grid = np.zeros((self.size, self.size), dtype=np.float32)
        self.footprint_grid = np.zeros((self.size, self.size), dtype=np.int32)

        if self.toroidal:
            available = [
                (i, j)
                for i in range(self.size)
                for j in range(self.size)
            ]
        else:
            available = [
                (i, j)
                for i in range(1, self.size - 1)
                for j in range(1, self.size - 1)
            ]
        self._rng.shuffle(available)

        # Behavioral diversity: place reef patches before any entity placement so
        # reef cells are excluded from agent/hazard/resource spawn pools.
        # SD-054 bipartite extension (2026-05-11): when reef_bipartite_layout=True,
        # place reef in one half (bottom rows for axis=horizontal, right cols for
        # axis=vertical) and partition `available` so agent spawns from the
        # midline band and hazards/resources spawn from the opposite half. Pools
        # are aliased to a single `available` list in the legacy path so the
        # legacy pop()-from-shared-pool behavior is bit-identical.
        if self.reef_enabled:
            if self.reef_bipartite_layout:
                self._place_reef_patches_bipartite()
                agent_pool, forage_pool = self._build_bipartite_pools(available)
            else:
                self._place_reef_patches(available)
                agent_pool = forage_pool = available
        else:
            self._reef_cells = set()
            self._reef_field = np.zeros((self.size, self.size), dtype=np.float32)
            agent_pool = forage_pool = available

        # infant_substrate:GAP-2 -- per-episode microhabitat zone map.
        # Built over the full interior cell list (independent of reef
        # removal) BEFORE entity placement so spawn weighting can read it.
        # Reset per-episode zone state regardless of switch so diagnostics
        # are deterministic. When disabled: _zone_map stays None, the
        # spawn paths use a bare pop() and no extra RNG draws occur
        # (bit-identical legacy behaviour, SD-047/048/049 OFF precedent).
        self._zone_c_visit_count = 0
        self._zone_c_ambient_this_tick = 0.0
        if self.microhabitat_enabled:
            if self.toroidal:
                _interior = [
                    (i, j) for i in range(self.size) for j in range(self.size)
                ]
            else:
                _interior = [
                    (i, j)
                    for i in range(1, self.size - 1)
                    for j in range(1, self.size - 1)
                ]
            self._build_microhabitat_zones(_interior)
        else:
            self._zone_map = None

        # infant_substrate:GAP-5 -- clear rolling position window + visited
        # set. Reset regardless of switch (GAP-2 precedent) so the
        # pos_entropy / zone_coverage diagnostics are deterministic across
        # episodes.
        self._reset_pos_telemetry()

        ax, ay = agent_pool.pop()
        self.agent_x = ax
        self.agent_y = ay
        self.agent_health = 1.0
        self.agent_energy = 1.0
        self.grid[ax, ay] = self.ENTITY_TYPES["agent"]
        self._last_action = 4  # stay

        self.hazards: List[List[int]] = []
        for _ in range(min(self.num_hazards, len(forage_pool))):
            if self.microhabitat_enabled:
                hx, hy = self._pop_zone_weighted(forage_pool, "hazard")
            else:
                hx, hy = forage_pool.pop()
            self.grid[hx, hy] = self.ENTITY_TYPES["hazard"]
            self.hazards.append([hx, hy])

        self.resources: List[List[int]] = []
        # SD-049: per-type spawn (default OFF -> single anonymous pool, identical
        # to legacy). When enabled, distribute num_resources across types per
        # resource_type_distribution, respecting the curriculum hook.
        # Reset SD-049 per-episode state (preserves cross-episode _global_step
        # and _novelty_familiarity for curriculum + cross-episode novelty decay).
        for tlist in self._resources_by_type:
            tlist.clear()
        self._resource_type_grid = np.zeros((self.size, self.size), dtype=np.int8)
        self._resource_field_by_type = np.zeros(
            (self.n_resource_types, self.size, self.size), dtype=np.float32
        )
        self._per_axis_drive = np.zeros(self.n_resource_types, dtype=np.float32)
        self._sd049_n_resource_contacts_by_type = np.zeros(
            self.n_resource_types, dtype=np.int32
        )
        self._sd049_n_axis_depletion_steps = 0
        # SD-049 Phase 2: per-tick cached consumed-type tag (type_idx + 1, or 0
        # if no consumption this tick). Surfaced as
        # info["sd049_consumed_type_tag_this_tick"] for the V3-EXQ-514
        # identity-classifier supervision target after the cell tag has been
        # cleared in the resource-consumption branch.
        self._consumed_type_tag_this_tick = 0
        if self.multi_resource_heterogeneity_enabled:
            # Determine which types are currently introduced via the curriculum.
            active_types = [
                i for i in range(self.n_resource_types)
                if self._global_step >= int(
                    self.resource_introduction_schedule.get(self.resource_type_names[i], 0)
                )
            ]
            # Build per-cell type assignment via weighted draw over active types
            # with non-zero distribution mass. Falls back to active-uniform if
            # all distribution masses on active types are zero.
            active_weights = [self.resource_type_distribution[i] for i in active_types]
            if active_types and sum(active_weights) > 0.0:
                weights = np.array(active_weights, dtype=np.float64)
                weights = weights / weights.sum()
                n_to_spawn = min(self.num_resources, len(forage_pool))
                for _ in range(n_to_spawn):
                    if self.microhabitat_enabled:
                        rx, ry = self._pop_zone_weighted(forage_pool, "resource")
                    else:
                        rx, ry = forage_pool.pop()
                    type_idx = int(active_types[
                        int(self._rng.choice(len(active_types), p=weights))
                    ])
                    self.grid[rx, ry] = self.ENTITY_TYPES["resource"]
                    self.resources.append([rx, ry])
                    self._resources_by_type[type_idx].append([rx, ry])
                    # Type tag stored with +1 offset so 0 = no-resource.
                    self._resource_type_grid[rx, ry] = type_idx + 1
            # If no active types (edge case: all behind a curriculum gate), no
            # resources spawn this episode -- the agent has no appetitive
            # signal until the curriculum advances. Intentional.
        else:
            for _ in range(min(self.num_resources, len(forage_pool))):
                if self.microhabitat_enabled:
                    rx, ry = self._pop_zone_weighted(forage_pool, "resource")
                else:
                    rx, ry = forage_pool.pop()
                self.grid[rx, ry] = self.ENTITY_TYPES["resource"]
                self.resources.append([rx, ry])

        self.waypoints: List[List[int]] = []
        self._next_waypoint_idx: int = 0
        self._sequence_in_progress: bool = False
        self._sequence_step: int = 0
        self._steps_since_waypoint: int = 0
        self._sequences_completed: int = 0

        if self.subgoal_mode:
            for _ in range(min(self.num_waypoints, len(forage_pool))):
                wx, wy = forage_pool.pop()
                self.grid[wx, wy] = self.ENTITY_TYPES["waypoint"]
                self.waypoints.append([wx, wy])

        self.steps = 0
        self.total_harm = 0.0
        self.total_benefit = 0.0
        # SD-029: per-episode external hazard counter.
        self._external_hazard_event_count = 0
        self._last_external_hazard_step = -1
        # SD-047: per-episode multi-source counters reset.
        self._multi_source_n_env_events = 0
        self._multi_source_n_agent_events = 0
        if self.multi_source_dynamics_enabled:
            self._init_multi_source_state()
        # SD-048: per-episode interoceptive-noise state reset.
        # _fatigue_state and _sensitisation_amplification are episode-local per the
        # SD doc (allostatic load / inflammatory flares dissipate by next episode).
        # _prev_harm_obs_a is cleared so the first tick of each episode counts no
        # body-noise / agent-caused delta events (no anchor to compare against).
        # Per-tick counters are zeroed by _apply_interoceptive_noise on each call;
        # zeroing them here as well keeps reset() output deterministic for diag readers.
        self._fatigue_state = 0.0
        self._sensitisation_amplification = 0.0
        self._prev_harm_obs_a = None
        self._last_transition_type = "none"
        self._interoceptive_n_autonomic_events = 0
        self._interoceptive_n_sensitisation_events = 0
        self._interoceptive_n_fatigue_events = 0
        self._interoceptive_n_body_noise_events = 0
        self._interoceptive_n_agent_caused_harm_events = 0

        # infant_substrate:GAP-3 -- per-episode transient benefit patch reset.
        # Patches are episode-local (a fresh stochastic schedule per
        # episode). self.grid was rebuilt at the top of reset(), so any
        # prior-episode patch cells are already cleared; only the Python-
        # side tracking + diagnostic counters need resetting here.
        self._transient_benefits = []
        self._transient_benefit_cells = set()
        self._transient_benefit_n_spawned = 0
        self._transient_benefit_n_contacted = 0
        self._transient_benefit_n_expired = 0
        self._transient_benefit_contact_this_tick = 0.0

        # Proxy-gradient state
        self.harm_exposure: float = 0.0
        self.benefit_exposure: float = 0.0
        self.hazard_field = np.zeros((self.size, self.size), dtype=np.float32)
        self.resource_field = np.zeros((self.size, self.size), dtype=np.float32)
        # SD-022: reset limb damage state on episode boundary.
        # Damage is episode-local: within-episode dissociation is sufficient for
        # stream separation tests (agent can accumulate damage and retreat to safety
        # within same episode, producing the A-delta/C-fiber dissociation).
        if self.limb_damage_enabled:
            self.limb_damage[:] = 0.0
        # harm_obs_a_ema is NOT reset here -- it persists across episodes (see __init__).
        if self.use_proxy_fields:
            self._compute_proximity_fields()

        # SD-023: place landmark objects and precompute their static gradient fields.
        # Landmarks are gradient-only (no grid entity type), so they can share cells
        # with any object. Use the full interior cell list (not just remaining available).
        if self.n_landmarks_a > 0 or self.n_landmarks_b > 0:
            if self.toroidal:
                _interior = [(i, j) for i in range(self.size) for j in range(self.size)]
            else:
                _interior = [
                    (i, j) for i in range(1, self.size - 1) for j in range(1, self.size - 1)
                ]
            self.landmark_a_positions = self._place_random_landmarks(self.n_landmarks_a, _interior)
            self.landmark_b_positions = self._place_biased_near_resources(
                self.n_landmarks_b, self.landmark_b_resource_bias, radius=2, available=_interior
            )
            self._landmark_a_field = self._compute_landmark_field(
                self.landmark_a_positions, self.landmark_a_sigma, self.landmark_a_scale
            )
            self._landmark_b_field = self._compute_landmark_field(
                self.landmark_b_positions, self.landmark_b_sigma, self.landmark_b_scale
            )
        else:
            self.landmark_a_positions = []
            self.landmark_b_positions = []
            self._landmark_a_field = np.zeros((self.size, self.size), dtype=np.float32)
            self._landmark_b_field = np.zeros((self.size, self.size), dtype=np.float32)

        obs_dict = self._get_observation_dict()
        flat_obs = self._dict_to_flat(obs_dict)
        return flat_obs, obs_dict

    def reset_to(
        self,
        agent_pos: Tuple[int, int],
        hazard_positions: List[Tuple[int, int]],
        resource_positions: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Deterministic reset for scripted-eval comparator harness (SD-029 / EXQ-433a).

        Bypasses random placement; places agent/hazards/resources at provided coords.
        Mirrors reset() state init except for placement. Landmarks (SD-023) are not
        placed by this method (n_landmarks_a/b assumed 0 for scripted eval).
        """
        self.grid = np.zeros((self.size, self.size), dtype=np.int32)
        if not self.toroidal:
            self.grid[0, :] = self.ENTITY_TYPES["wall"]
            self.grid[-1, :] = self.ENTITY_TYPES["wall"]
            self.grid[:, 0] = self.ENTITY_TYPES["wall"]
            self.grid[:, -1] = self.ENTITY_TYPES["wall"]

        self.contamination_grid = np.zeros((self.size, self.size), dtype=np.float32)
        self.footprint_grid = np.zeros((self.size, self.size), dtype=np.int32)

        ax, ay = int(agent_pos[0]), int(agent_pos[1])
        self.agent_x = ax
        self.agent_y = ay
        self.agent_health = 1.0
        self.agent_energy = 1.0
        self.grid[ax, ay] = self.ENTITY_TYPES["agent"]
        self._last_action = 4

        self.hazards = []
        for hx, hy in hazard_positions:
            hx, hy = int(hx), int(hy)
            if (hx, hy) == (ax, ay):
                continue
            self.grid[hx, hy] = self.ENTITY_TYPES["hazard"]
            self.hazards.append([hx, hy])

        self.resources = []
        if resource_positions:
            for rx, ry in resource_positions:
                rx, ry = int(rx), int(ry)
                if self.grid[rx, ry] != self.ENTITY_TYPES["empty"]:
                    continue
                self.grid[rx, ry] = self.ENTITY_TYPES["resource"]
                self.resources.append([rx, ry])

        self.waypoints = []
        self._next_waypoint_idx = 0
        self._sequence_in_progress = False
        self._sequence_step = 0
        self._steps_since_waypoint = 0
        self._sequences_completed = 0

        self.steps = 0
        self.total_harm = 0.0
        self.total_benefit = 0.0
        # SD-029: per-episode external hazard counter.
        self._external_hazard_event_count = 0
        self._last_external_hazard_step = -1
        # SD-047: scripted-eval reset path leaves multi-source state OFF; experiments
        # using reset_to() are SD-029 / EXQ-433a comparator harnesses that intentionally
        # bypass multi-source dynamics for clean self-vs-externally-caused tagging.
        self._multi_source_n_env_events = 0
        self._multi_source_n_agent_events = 0
        # SD-048: scripted-eval reset path also clears interoceptive-noise state.
        # Same reasoning as SD-047 above: comparator-harness experiments need clean
        # tagging, so per-episode body-noise state starts at zero.
        self._fatigue_state = 0.0
        # infant_substrate:GAP-5 -- scripted-eval path also clears the
        # position-telemetry window/visited set so pos_entropy /
        # zone_coverage are deterministic across reset_to() episodes.
        self._reset_pos_telemetry()
        self._sensitisation_amplification = 0.0
        self._prev_harm_obs_a = None
        self._last_transition_type = "none"
        self._interoceptive_n_autonomic_events = 0
        self._interoceptive_n_sensitisation_events = 0
        self._interoceptive_n_fatigue_events = 0
        self._interoceptive_n_body_noise_events = 0
        self._interoceptive_n_agent_caused_harm_events = 0

        self.harm_exposure = 0.0
        self.benefit_exposure = 0.0
        self.hazard_field = np.zeros((self.size, self.size), dtype=np.float32)
        self.resource_field = np.zeros((self.size, self.size), dtype=np.float32)
        if self.limb_damage_enabled:
            self.limb_damage[:] = 0.0
        if self.use_proxy_fields:
            self._compute_proximity_fields()

        self.landmark_a_positions = []
        self.landmark_b_positions = []
        self._landmark_a_field = np.zeros((self.size, self.size), dtype=np.float32)
        self._landmark_b_field = np.zeros((self.size, self.size), dtype=np.float32)

        obs_dict = self._get_observation_dict()
        flat_obs = self._dict_to_flat(obs_dict)
        return flat_obs, obs_dict

    # ------------------------------------------------------------------ #
    # Step                                                                 #
    # ------------------------------------------------------------------ #

    def step(
        self,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, bool, Dict, Dict]:
        """
        Execute one step.

        Returns:
            flat_obs:        flat observation tensor [observation_dim]
            harm_signal:     float — negative = harm, positive = benefit
            done:            bool
            info:            dict with transition_type, contamination_delta, etc.
            obs_dict:        SD-005 split observation dict
        """
        if isinstance(action, torch.Tensor):
            action = action.argmax().item() if action.dim() > 0 else action.item()
        action = int(action) % len(self.ACTIONS)
        self._last_action = action

        # SD-049 Phase 2: reset per-tick consumed-type cache before any
        # consumption logic runs. The flag is set inside the resource branch
        # below and read by the info dict surfacing at the bottom of step().
        self._consumed_type_tag_this_tick = 0

        dx, dy = self.ACTIONS[action]
        if self.toroidal:
            new_x = (self.agent_x + dx) % self.size
            new_y = (self.agent_y + dy) % self.size
        else:
            new_x = self.agent_x + dx
            new_y = self.agent_y + dy

        harm_signal = 0.0
        transition_type = "none"
        contamination_delta = 0.0
        env_drift_occurred = False

        # SD-022: save position before movement for potential limb-failure rollback.
        prev_x, prev_y = self.agent_x, self.agent_y

        # infant_substrate:GAP-1: per-tick gradient diagnostics (always reset; set inside movement block).
        _grad_reward = 0.0
        _grad_dist = float("inf")
        # infant_substrate:GAP-2: per-tick zone-C ambient bonus (always reset;
        # set inside movement block when the agent enters a zone-C cell).
        self._zone_c_ambient_this_tick = 0.0
        # infant_substrate:GAP-3: per-tick transient benefit contact reward
        # (always reset; set inside the resource-contact branch when the
        # agent steps onto a transient patch cell).
        self._transient_benefit_contact_this_tick = 0.0

        # Move agent if not wall (toroidal has no walls, so always move)
        if self.toroidal or self.grid[new_x, new_y] != self.ENTITY_TYPES["wall"]:
            old_x, old_y = self.agent_x, self.agent_y

            if self.contamination_grid[old_x, old_y] >= self.contamination_threshold:
                self.grid[old_x, old_y] = self.ENTITY_TYPES["contaminated"]
            else:
                self.grid[old_x, old_y] = self.ENTITY_TYPES["empty"]

            target_type = self.grid[new_x, new_y]

            if target_type == self.ENTITY_TYPES["hazard"]:
                contact_harm = self.hazard_harm
                if self.use_proxy_fields:
                    # Proximity harm is additive to contact harm at the hazard cell
                    proximity_harm = self.proximity_harm_scale * float(
                        self.hazard_field[new_x, new_y]
                    )
                    harm_signal = -(contact_harm + proximity_harm)
                else:
                    harm_signal = -contact_harm
                # Behavioral diversity: reef cells suppress all contact harm
                # (hazards should not enter reef cells, but guard here for safety).
                if self.reef_enabled and (new_x, new_y) in self._reef_cells:
                    harm_signal = 0.0
                else:
                    self.agent_health = max(0.0, self.agent_health - abs(harm_signal))
                    self.total_harm += abs(harm_signal)
                transition_type = "env_caused_hazard"

            elif target_type == self.ENTITY_TYPES["contaminated"]:
                harm_signal = -self.contaminated_harm
                self.agent_health = max(0.0, self.agent_health - self.contaminated_harm)
                transition_type = "agent_caused_hazard"
                self.total_harm += self.contaminated_harm

            elif target_type == self.ENTITY_TYPES["resource"]:
                # SD-049: identify resource type at the consumed cell. type_tag
                # is 0 (no SD-049) or type_idx+1 (per the spawn convention).
                type_tag = (
                    int(self._resource_type_grid[new_x, new_y])
                    if self.multi_resource_heterogeneity_enabled
                    else 0
                )
                contact_type_idx = type_tag - 1 if type_tag > 0 else -1
                # SD-049 Phase 2: cache the consumed-this-tick type so the
                # info dict can report it after the cell tag is cleared.
                # This is the identity-classifier supervision target for
                # V3-EXQ-514 (z_resource -> identity_logits cross-entropy).
                # Reset to 0 at the top of every step() (see _consumed_type_tag_this_tick init).
                self._consumed_type_tag_this_tick = type_tag
                # Per-type benefit amplitude scales the contact restoration. Default
                # 1.0 recovers legacy contact_benefit semantics.
                amp = 1.0
                if contact_type_idx >= 0:
                    amp = float(self.resource_type_benefit_amplitudes[contact_type_idx])
                contact_benefit = self.resource_benefit * amp
                # infant_substrate:GAP-3 -- transient benefit patch contact.
                # A transient patch is a high-salience benefit: its contact
                # reward is resource_benefit * transient_benefit_multiplier
                # (overrides the SD-049 per-type amplitude; transient
                # patches are intentionally NOT SD-049 typed -- their
                # _resource_type_grid tag is 0). The patch is consumed here
                # (dropped from transient tracking); the normal
                # self.resources / grid removal below handles the rest.
                if (new_x, new_y) in self._transient_benefit_cells:
                    contact_benefit = (
                        self.resource_benefit * self.transient_benefit_multiplier
                    )
                    self._transient_benefit_cells.discard((new_x, new_y))
                    self._transient_benefits = [
                        tb for tb in self._transient_benefits
                        if not (tb[0] == new_x and tb[1] == new_y)
                    ]
                    self._transient_benefit_n_contacted += 1
                    self._transient_benefit_contact_this_tick = float(
                        contact_benefit
                    )
                # SD-049 novelty curve: scale benefit by (1 - cell_familiarity).
                # First-visit cells give full benefit; re-visits give attenuated
                # benefit reflecting non-homeostatic familiarity decay.
                if (
                    contact_type_idx >= 0
                    and self.resource_type_benefit_curves[contact_type_idx]
                    == "novelty_decay"
                ):
                    cell_fam = float(
                        np.clip(self._novelty_familiarity[new_x, new_y], 0.0, 1.0)
                    )
                    contact_benefit = contact_benefit * (1.0 - cell_fam)
                if self.use_proxy_fields:
                    proximity_benefit = self.proximity_benefit_scale * float(
                        self.resource_field[new_x, new_y]
                    )
                    harm_signal = contact_benefit + proximity_benefit
                else:
                    harm_signal = contact_benefit
                self.agent_health = min(1.0, self.agent_health + contact_benefit * 0.5)
                self.agent_energy = min(1.0, self.agent_energy + contact_benefit * 0.5)
                transition_type = "resource"
                self.total_benefit += harm_signal
                self.resources = [
                    r for r in self.resources if not (r[0] == new_x and r[1] == new_y)
                ]
                # SD-049: also remove from per-type list and clear cell tag.
                if self.multi_resource_heterogeneity_enabled and contact_type_idx >= 0:
                    self._resources_by_type[contact_type_idx] = [
                        r for r in self._resources_by_type[contact_type_idx]
                        if not (r[0] == new_x and r[1] == new_y)
                    ]
                    self._resource_type_grid[new_x, new_y] = 0
                    self._sd049_n_resource_contacts_by_type[contact_type_idx] += 1
                    # Restoration on the matching drive axis. Curve choice maps
                    # to magnitude / saturation profile; all curves saturate at
                    # full restoration (drive=0). novelty_decay treats axis as
                    # exploratory tone -- contact returns full restoration
                    # gated by per-cell familiarity above.
                    curve = self.resource_type_benefit_curves[contact_type_idx]
                    cur_drive = float(self._per_axis_drive[contact_type_idx])
                    if curve == "sigmoidal_saturating":
                        # Restoration proportional to how depleted the axis was.
                        # Full deficit -> full restoration; near-sated -> small.
                        restore = cur_drive * 1.0
                    elif curve == "sharp_saturation":
                        # Sharper: any contact restores most of the deficit.
                        restore = cur_drive * 0.8
                    elif curve == "novelty_decay":
                        # Familiarity-gated; cell_fam already applied to
                        # contact_benefit above. Axis restoration mirrors the
                        # sigmoidal shape but on the curiosity axis.
                        restore = cur_drive * 1.0
                    else:
                        restore = cur_drive * 1.0
                    self._per_axis_drive[contact_type_idx] = float(
                        np.clip(cur_drive - restore, 0.0, 1.0)
                    )
                    # Per-cell novelty familiarity increment (whether or not the
                    # contact was on a novelty-curve resource: every visit teaches
                    # the agent something about that cell -- the per-type curve
                    # determines whether that familiarity is used in scoring).
                    self._novelty_familiarity[new_x, new_y] = float(
                        np.clip(
                            self._novelty_familiarity[new_x, new_y]
                            + self.novelty_familiarity_increment,
                            0.0,
                            1.0,
                        )
                    )
                # SD-012: optional resource respawn for repeated drive-reduction cycles
                if self.resource_respawn_on_consume:
                    self._respawn_resource()
                # Resource consumed (or respawned) — recompute resource field
                if self.use_proxy_fields:
                    self._compute_proximity_fields()

            elif target_type == self.ENTITY_TYPES["waypoint"] and self.subgoal_mode:
                wp_idx = next(
                    (i for i, w in enumerate(self.waypoints)
                     if w[0] == new_x and w[1] == new_y),
                    None,
                )
                if wp_idx == self._next_waypoint_idx:
                    self._next_waypoint_idx += 1
                    self._sequence_step = wp_idx
                    self._steps_since_waypoint = 0
                    if not self._sequence_in_progress:
                        self._sequence_in_progress = True

                    if self._next_waypoint_idx >= len(self.waypoints):
                        # Sequence complete
                        harm_signal += self.waypoint_completion_reward
                        self.total_benefit += self.waypoint_completion_reward
                        transition_type = "sequence_complete"
                        self._sequences_completed += 1
                        self._sequence_in_progress = False
                        self._next_waypoint_idx = 0
                        self._respawn_waypoints()
                    else:
                        harm_signal += self.waypoint_visit_reward
                        self.total_benefit += self.waypoint_visit_reward
                        transition_type = "waypoint"

            elif self.use_proxy_fields and transition_type == "none":
                # Proxy-gradient approach transitions (only when no contact event)
                h_field_val = float(self.hazard_field[new_x, new_y])
                r_field_val = float(self.resource_field[new_x, new_y])
                if h_field_val >= self.proximity_approach_threshold:
                    harm_signal = -self.proximity_harm_scale * h_field_val
                    transition_type = "hazard_approach"
                    self.total_harm += abs(harm_signal)
                    self.agent_health = max(0.0, self.agent_health - abs(harm_signal))
                elif r_field_val >= self.proximity_approach_threshold:
                    harm_signal = self.proximity_benefit_scale * r_field_val
                    transition_type = "benefit_approach"
                    self.total_benefit += harm_signal

            # infant_substrate:GAP-1 -- harm gradient env feature.
            # Graduated approach signal based on Euclidean distance to nearest
            # hazard. Fires only when no direct-contact or existing-approach
            # transition fired (transition_type == "none"). Pure reward signal:
            # no agent health deduction. No RNG draws; bit-identical OFF when disabled.
            if self.harm_gradient_enabled and transition_type == "none" and self.hazards:
                _min_dist_sq = float("inf")
                for _h in self.hazards:
                    _dx = float(_h[0] - new_x)
                    _dy = float(_h[1] - new_y)
                    _dsq = _dx * _dx + _dy * _dy
                    if _dsq < _min_dist_sq:
                        _min_dist_sq = _dsq
                _grad_dist = _min_dist_sq ** 0.5
                _r_out = self.harm_gradient_outer_radius
                _r_in = self.harm_gradient_inner_radius
                if _r_in < _grad_dist <= _r_out and _r_out > 0.0:
                    _grad_reward = (
                        -self.hazard_harm
                        * (1.0 - _grad_dist / _r_out) ** 2
                        * self.harm_gradient_scale
                    )
                    harm_signal += _grad_reward
                    transition_type = "harm_gradient"

            # infant_substrate:GAP-2 -- zone-C ambient presence bonus.
            # Small positive reward for occupying a zone-C (novelty) cell
            # when no contact / approach / gradient event fired. Decays
            # multiplicatively with the number of zone-C visits this episode
            # (zone_novelty_decay): a familiar novelty zone yields less.
            # No RNG draws; bit-identical OFF when disabled.
            if (
                self.microhabitat_enabled
                and self._zone_map is not None
                and transition_type == "none"
                and int(self._zone_map[new_x, new_y]) == 2
            ):
                _ambient = self.zone_C_ambient_bonus * (
                    self.zone_novelty_decay ** self._zone_c_visit_count
                )
                harm_signal += _ambient
                self._zone_c_ambient_this_tick = float(_ambient)
                self._zone_c_visit_count += 1
                transition_type = "zone_c_ambient"

            # Move agent
            self.agent_x = new_x
            self.agent_y = new_y
            self.grid[new_x, new_y] = self.ENTITY_TYPES["agent"]

            # Update causal footprint
            self.footprint_grid[new_x, new_y] += 1
            old_cont = self.contamination_grid[new_x, new_y]
            self.contamination_grid[new_x, new_y] += self.contamination_spread
            contamination_delta = self.contamination_grid[new_x, new_y] - old_cont

            # SD-022: directional limb damage accumulation and movement failure.
            # Map action index to limb index: N=0(action 0), E=1(action 3), S=2(action 1), W=3(action 2).
            # Action 4 (stay) uses no limb (d=-1, no damage or failure).
            _ACTION_TO_LIMB = {0: 0, 1: 2, 2: 3, 3: 1, 4: -1}
            if self.limb_damage_enabled and action < 4:
                d = _ACTION_TO_LIMB[action]
                # Accumulate damage when harm signal is negative (hazard encounter).
                if harm_signal < 0:
                    harm_mag = abs(harm_signal)
                    self.limb_damage[d] = min(1.0,
                        self.limb_damage[d] + self.damage_increment * harm_mag)
                # Heal all limbs each step.
                self.limb_damage *= (1.0 - self.heal_rate)
                # Movement failure: damaged limb may fail, reverting agent to prev position.
                if self._rng.random() < float(self.limb_damage[d]) * self.failure_prob_scale:
                    self.grid[self.agent_x, self.agent_y] = self.ENTITY_TYPES["empty"]
                    self.agent_x, self.agent_y = prev_x, prev_y
                    self.grid[prev_x, prev_y] = self.ENTITY_TYPES["agent"]
            elif self.limb_damage_enabled:
                # Stay action: still apply healing.
                self.limb_damage *= (1.0 - self.heal_rate)

        # Energy decay
        self.agent_energy = max(0.0, self.agent_energy - self.energy_decay)

        # SD-049: per-axis drive depletion + legacy agent_energy collapse.
        # Per-axis drive[i] increases each step by per_axis_drive_decay[i]
        # (depletion analog -- fatigue accumulates in the axis between contacts).
        # Novelty per-cell familiarity recovery (slow recovery means novelty
        # benefit comes back over time when configured).
        # When per_axis_drive_enabled, agent_energy is overridden by
        # 1.0 - combined_drive so legacy SD-032 consumers (AIC / PCC / pACC /
        # dACC / salience / override / MECH-295) continue to read obs_body[3]
        # without modification. The per-axis vector is also surfaced in
        # obs_dict["per_axis_drive"] for new experiments / future encoder upgrade.
        if self.multi_resource_heterogeneity_enabled and self.per_axis_drive_enabled:
            for i in range(self.n_resource_types):
                cur = float(self._per_axis_drive[i])
                self._per_axis_drive[i] = float(
                    np.clip(cur + self.per_axis_drive_decay[i], 0.0, 1.0)
                )
            self._sd049_n_axis_depletion_steps += 1
            if self.per_axis_drive_combiner == "max":
                combined_drive = float(np.max(self._per_axis_drive))
            elif self.per_axis_drive_combiner == "mean":
                combined_drive = float(np.mean(self._per_axis_drive))
            else:  # "sum"
                combined_drive = float(np.clip(np.sum(self._per_axis_drive), 0.0, 1.0))
            self.agent_energy = float(np.clip(1.0 - combined_drive, 0.0, 1.0))
        # Per-cell novelty familiarity slow recovery (when configured).
        if (
            self.multi_resource_heterogeneity_enabled
            and self.novelty_familiarity_recovery > 0.0
        ):
            self._novelty_familiarity = np.maximum(
                0.0,
                self._novelty_familiarity - self.novelty_familiarity_recovery,
            ).astype(np.float32)
        # SD-049: advance the cross-episode global step counter (for curriculum
        # introduction). Done here (per env tick) so the counter increments
        # whether or not the agent contacted anything, and survives reset().
        if self.multi_resource_heterogeneity_enabled:
            self._global_step += 1

        # Update interoceptive EMA channels
        if self.use_proxy_fields:
            alpha = self.nociception_ema_alpha
            if harm_signal < 0:
                self.harm_exposure = (1 - alpha) * self.harm_exposure + alpha * abs(harm_signal)
            else:
                self.harm_exposure = (1 - alpha) * self.harm_exposure
            if harm_signal > 0:
                self.benefit_exposure = (1 - alpha) * self.benefit_exposure + alpha * harm_signal
            else:
                self.benefit_exposure = (1 - alpha) * self.benefit_exposure
            # SD-011: update affective harm accumulator (C-fiber / paleospinothalamic analog).
            # EXQ-102 confirmed the prior spatial-window EMA had autocorr~0: as the agent
            # moves, the 5x5 local window content changes each step, destroying temporal
            # persistence. C-fiber / affective harm signal represents accumulated homeostatic
            # unpleasantness -- it follows the agent's trajectory, not the current grid view.
            # Fix (2026-03-28): EMA of the agent's current-cell hazard/resource scalar,
            # replicated uniformly across all 25 dims per channel. This gives autocorr ~
            # (1-alpha)^lag (e.g. lag=10 -> ~0.60 >> threshold 0.30). Interface stays 50-dim.
            alpha_a = self.harm_obs_a_ema_alpha
            ax2, ay2 = int(self.agent_x), int(self.agent_y)
            hazard_at_agent = float(np.clip(self.hazard_field[ax2, ay2], 0.0, 1.0))
            resource_at_agent = float(np.clip(self.resource_field[ax2, ay2], 0.0, 1.0))
            self.harm_obs_a_ema[:25] = (1.0 - alpha_a) * self.harm_obs_a_ema[:25] + alpha_a * hazard_at_agent
            self.harm_obs_a_ema[25:] = (1.0 - alpha_a) * self.harm_obs_a_ema[25:] + alpha_a * resource_at_agent

            # SD-011 second source: record harm_exposure into rolling history buffer.
            if self.harm_history_len > 0:
                idx = self._harm_history_ptr % self.harm_history_len
                self._harm_history[idx] = float(np.clip(self.harm_exposure, 0.0, 1.0))
                self._harm_history_ptr += 1
                self._accumulated_harm_exposure += float(np.clip(self.harm_exposure, 0.0, 1.0))
                self._accumulated_harm_steps += 1

        # SD-047: per-step diagnostic counters reset; agent-caused events are tagged
        # by the existing transition_type machinery above (env_caused_hazard /
        # agent_caused_hazard / hazard_approach / resource / etc).
        self._multi_source_n_agent_events = 0
        self._multi_source_n_env_events = 0
        if transition_type in (
            "agent_caused_hazard",
            "hazard_approach",
            "resource",
            "benefit_approach",
            "waypoint",
            "sequence_complete",
        ):
            self._multi_source_n_agent_events += 1
        if transition_type == "env_caused_hazard":
            self._multi_source_n_env_events += 1

        # SD-047: multi-source environmental dynamics. Three concurrent stochastic
        # event sources, each agent-independent, gated by master switch + per-source
        # switches. RNG draws guarded inside the master if so seed sequences for
        # existing experiments are bit-identical when disabled.
        weather_step_delta = 0.0
        n_transient_appear = 0
        n_transient_disappear = 0
        n_drift_moved = 0
        if self.multi_source_dynamics_enabled:
            if self.weather_field_enabled:
                weather_step_delta = self._step_weather_field()
            if self.transient_events_enabled:
                n_transient_appear, n_transient_disappear = self._step_transient_events()
            if self.background_drift_enabled:
                n_drift_moved = self._step_background_drift()
            # Multi-source contributions to env-event count: weather is continuous (not
            # counted as discrete events); transients add appearances and disappearances;
            # drift moves count as env events because each is an unscheduled hazard
            # relocation visible to the agent's perception.
            self._multi_source_n_env_events += int(n_transient_appear)
            self._multi_source_n_env_events += int(n_transient_disappear)
            self._multi_source_n_env_events += int(n_drift_moved)
            # Recompute proximity fields if any source perturbed hazard layout / weather.
            if self.use_proxy_fields and (
                self.weather_field_enabled
                or n_transient_appear > 0
                or n_transient_disappear > 0
                or n_drift_moved > 0
            ):
                self._compute_proximity_fields()

        # SD-029: scheduled external-hazard injection (balanced curriculum).
        # Deterministically scheduled "externally-caused" hazard events that are
        # independent of the agent's action. When enabled, every
        # scheduled_external_hazard_interval steps, with probability
        # scheduled_external_hazard_prob, move a hazard adjacent to the agent
        # (or to any empty cell if adjacent_only=False and no adjacency is empty).
        external_hazard_injected = False
        if (
            self.scheduled_external_hazard_enabled
            and self.steps > 0
            and (self.steps % self.scheduled_external_hazard_interval == 0)
            and self._rng.random() < self.scheduled_external_hazard_prob
        ):
            external_hazard_injected = self._inject_external_hazard()
            if external_hazard_injected:
                self._external_hazard_event_count += 1
                self._last_external_hazard_step = self.steps
                if self.use_proxy_fields:
                    self._compute_proximity_fields()

        # Env-caused drift
        if self.steps % self.env_drift_interval == 0 and self.steps > 0:
            self._drift_hazards()
            env_drift_occurred = True

        # infant_substrate:GAP-3 -- transient benefit patches.
        # Agent-independent stochastic high-salience benefit spawn for
        # z_goal seeding. Expiry runs first (drop patches whose lifetime
        # elapsed), then one spawn attempt with probability
        # transient_benefit_prob. All RNG draws are guarded by
        # transient_benefit_enabled so seed sequences for existing
        # experiments are bit-identical when disabled. Uses self.steps
        # pre-increment as the clock: a patch spawned at tick T expires
        # at tick T + transient_benefit_duration.
        _tb_changed = False
        if self.transient_benefit_enabled:
            if self._transient_benefits:
                _surv: List[List[int]] = []
                for _tb in self._transient_benefits:
                    if self.steps >= _tb[2]:
                        # Expire. Only clear the grid cell if it still
                        # carries the resource tag (the agent standing on
                        # it would have tagged it "agent"; a contacted
                        # patch was already removed via the contact hook).
                        if (
                            self.grid[_tb[0], _tb[1]]
                            == self.ENTITY_TYPES["resource"]
                        ):
                            self.grid[_tb[0], _tb[1]] = self.ENTITY_TYPES[
                                "empty"
                            ]
                        self.resources = [
                            r for r in self.resources
                            if not (r[0] == _tb[0] and r[1] == _tb[1])
                        ]
                        self._transient_benefit_cells.discard(
                            (_tb[0], _tb[1])
                        )
                        self._transient_benefit_n_expired += 1
                        _tb_changed = True
                    else:
                        _surv.append(_tb)
                self._transient_benefits = _surv
            if self._rng.random() < self.transient_benefit_prob:
                if self._spawn_transient_benefit() is not None:
                    _tb_changed = True
        if _tb_changed and self.use_proxy_fields:
            self._compute_proximity_fields()

        # Subgoal timeout
        if self.subgoal_mode and self._sequence_in_progress:
            self._steps_since_waypoint += 1
            if self._steps_since_waypoint > self.sequence_commitment_timeout:
                self._sequence_in_progress = False
                self._next_waypoint_idx = 0
                self._steps_since_waypoint = 0
                self._respawn_waypoints()

        self.steps += 1
        done = self.agent_health <= 0.0 or self.steps >= 500

        # infant_substrate:GAP-5 -- record the final agent cell this tick
        # into the rolling pos-entropy window + per-episode visited set.
        # Guarded by the master switch so disabled runs do no work.
        if self.pos_telemetry_enabled:
            cell = (int(self.agent_x), int(self.agent_y))
            self._pos_window.append(cell)
            if len(self._pos_window) > self.pos_entropy_window:
                self._pos_window.pop(0)
            self._visited_cells.add(cell)

        # SD-048: cache transition_type so _apply_interoceptive_noise can classify
        # agent-caused vs body-noise-caused harm-state-change events when computed
        # inside _get_observation_dict.
        self._last_transition_type = transition_type
        obs_dict = self._get_observation_dict()
        flat_obs = self._dict_to_flat(obs_dict)

        info = {
            "transition_type": transition_type,
            "contamination_delta": contamination_delta,
            "env_drift_occurred": env_drift_occurred,
            "footprint_at_cell": int(self.footprint_grid[self.agent_x, self.agent_y]),
            "health": self.agent_health,
            "energy": self.agent_energy,
            "steps": self.steps,
            "total_harm": self.total_harm,
            "total_benefit": self.total_benefit,
            "sequence_in_progress": self._sequence_in_progress,
            "sequence_step": self._sequence_step,
            # SD-029 balanced curriculum tags (always present; 0 when disabled).
            "external_hazard_injected": bool(external_hazard_injected),
            "external_hazard_event_count": int(self._external_hazard_event_count),
            # SD-047 multi-source tags (always present; 0 / False when disabled).
            "multi_source_dynamics_enabled": bool(self.multi_source_dynamics_enabled),
            "multi_source_intensity_scale": float(self.multi_source_intensity_scale),
            "multi_source_weather_step_delta": float(weather_step_delta),
            "multi_source_n_transient_appear": int(n_transient_appear),
            "multi_source_n_transient_disappear": int(n_transient_disappear),
            "multi_source_n_transient_active": int(len(self._transient_hazards)),
            "multi_source_n_drift_moved": int(n_drift_moved),
            "multi_source_n_drift_active": int(len(self._drift_sources)),
            "multi_source_n_env_events": int(self._multi_source_n_env_events),
            "multi_source_n_agent_events": int(self._multi_source_n_agent_events),
            # SD-048 interoceptive-noise tags (always present; 0 / False when disabled).
            "interoceptive_noise_enabled": bool(self.interoceptive_noise_enabled),
            "interoceptive_noise_scale": float(self.interoceptive_noise_scale),
            "interoceptive_n_autonomic_events": int(self._interoceptive_n_autonomic_events),
            "interoceptive_n_sensitisation_events": int(self._interoceptive_n_sensitisation_events),
            "interoceptive_n_fatigue_events": int(self._interoceptive_n_fatigue_events),
            "interoceptive_n_body_noise_events": int(self._interoceptive_n_body_noise_events),
            "interoceptive_n_agent_caused_harm_events": int(self._interoceptive_n_agent_caused_harm_events),
            "interoceptive_fatigue_state": float(self._fatigue_state),
            "interoceptive_sensitisation_amplification": float(self._sensitisation_amplification),
            # SD-049 multi-resource heterogeneity tags (always present; 0 / False when disabled).
            "multi_resource_heterogeneity_enabled": bool(self.multi_resource_heterogeneity_enabled),
            "sd049_n_resource_types": int(self.n_resource_types),
            "sd049_per_axis_drive_enabled": bool(self.per_axis_drive_enabled),
            "sd049_per_axis_drive_max": (
                float(np.max(self._per_axis_drive))
                if self.multi_resource_heterogeneity_enabled
                else 0.0
            ),
            "sd049_per_axis_drive_mean": (
                float(np.mean(self._per_axis_drive))
                if self.multi_resource_heterogeneity_enabled
                else 0.0
            ),
            "sd049_n_resource_contacts_total": int(
                np.sum(self._sd049_n_resource_contacts_by_type)
            ),
            "sd049_n_active_resources_by_type": (
                [len(t) for t in self._resources_by_type]
                if self.multi_resource_heterogeneity_enabled
                else [0] * self.n_resource_types
            ),
            "sd049_global_step": int(self._global_step),
            "sd049_resource_type_at_agent": (
                int(self._resource_type_grid[self.agent_x, self.agent_y])
                if self.multi_resource_heterogeneity_enabled
                else 0
            ),
            # SD-049 Phase 2: type_idx + 1 (1..n_types) of the resource consumed
            # THIS TICK (cleared from grid by consumption logic; cached before
            # clearing). 0 if no consumption this tick. This is the supervision
            # target for the V3-EXQ-514 identity classifier (z_resource ->
            # identity_logits cross-entropy). Always present (0 when SD-049 OFF).
            "sd049_consumed_type_tag_this_tick": int(self._consumed_type_tag_this_tick),
            # infant_substrate:GAP-1 harm gradient diagnostics (always present; 0/False when disabled).
            "harm_gradient_enabled": bool(self.harm_gradient_enabled),
            "harm_gradient_reward_this_tick": float(_grad_reward),
            "harm_gradient_dist_to_nearest": (
                float(_grad_dist) if _grad_dist != float("inf") else -1.0
            ),
            # infant_substrate:GAP-2 microhabitat diagnostics (always present; 0/-1 when disabled).
            "microhabitat_enabled": bool(self.microhabitat_enabled),
            "microhabitat_zone_at_agent": (
                int(self._zone_map[self.agent_x, self.agent_y])
                if self._zone_map is not None
                else -1
            ),
            "microhabitat_zone_c_ambient_this_tick": float(
                self._zone_c_ambient_this_tick
            ),
            "microhabitat_zone_counts": (
                [int((self._zone_map == z).sum()) for z in range(4)]
                if self._zone_map is not None
                else [0, 0, 0, 0]
            ),
            # infant_substrate:GAP-3 transient benefit diagnostics (always
            # present; 0 / False when disabled).
            "transient_benefit_enabled": bool(self.transient_benefit_enabled),
            "transient_benefit_n_active": int(len(self._transient_benefits)),
            "transient_benefit_n_spawned": int(
                self._transient_benefit_n_spawned
            ),
            "transient_benefit_n_contacted": int(
                self._transient_benefit_n_contacted
            ),
            "transient_benefit_n_expired": int(
                self._transient_benefit_n_expired
            ),
            "transient_benefit_contact_this_tick": float(
                self._transient_benefit_contact_this_tick
            ),
            # infant_substrate:GAP-5 H_pos / zone_coverage telemetry
            # (always present; -1.0 / {} sentinels when disabled).
            "pos_telemetry_enabled": bool(self.pos_telemetry_enabled),
            "pos_entropy": self._pos_entropy(),
            "pos_entropy_window": int(self.pos_entropy_window),
            "zone_coverage": self._zone_coverage(),
        }
        if self.use_proxy_fields:
            info["hazard_field_at_agent"] = float(
                self.hazard_field[self.agent_x, self.agent_y]
            )
            info["resource_field_at_agent"] = float(
                self.resource_field[self.agent_x, self.agent_y]
            )
            info["harm_exposure"] = self.harm_exposure
            info["benefit_exposure"] = self.benefit_exposure
        return flat_obs, harm_signal, done, info, obs_dict

    # ------------------------------------------------------------------ #
    # infant_substrate:GAP-5  H_pos / zone_coverage telemetry              #
    # ------------------------------------------------------------------ #

    def _reset_pos_telemetry(self) -> None:
        """Clear the rolling position window and per-episode visited set."""
        self._pos_window = []
        self._visited_cells = set()

    def _interior_cells_spec(self):
        """(predicate, count) for the interior used by the single-zone stub.

        Mirrors the GAP-2 _build_microhabitat_zones interior definition:
        the full grid when toroidal, else the non-border interior.
        """
        if self.toroidal:
            return (lambda x, y: True), self.size * self.size
        n = max(0, self.size - 2)
        return (
            lambda x, y: 1 <= x <= self.size - 2 and 1 <= y <= self.size - 2
        ), n * n

    def _pos_entropy(self) -> float:
        """Shannon entropy (nats) of the position histogram over the
        rolling window. -1.0 when telemetry is disabled or the window is
        empty. A stationary agent yields 0.0; a uniform spread over K
        distinct cells yields ln(K)."""
        if not self.pos_telemetry_enabled or not self._pos_window:
            return -1.0
        counts: Dict[Tuple[int, int], int] = {}
        for cell in self._pos_window:
            counts[cell] = counts.get(cell, 0) + 1
        total = float(len(self._pos_window))
        h = 0.0
        for c in counts.values():
            p = c / total
            h -= p * float(np.log(p))
        # Clamp tiny negative dust from float rounding (e.g. -1e-16).
        return float(max(0.0, h))

    def _zone_coverage(self) -> Dict[int, float]:
        """{zone_id: fraction of that zone's cells visited this episode}.

        GAP-2 active (_zone_map present): zones 0..3 keyed off the zone
        map, same per-zone denominator as the microhabitat_zone_counts
        info key. GAP-2 inactive + stub: single zone 0 = whole interior.
        Empty dict when telemetry disabled, or when no zone map and the
        single-zone stub is switched off."""
        if not self.pos_telemetry_enabled:
            return {}
        if self._zone_map is not None:
            cov: Dict[int, float] = {}
            for z in range(4):
                denom = int((self._zone_map == z).sum())
                if denom <= 0:
                    continue
                visited = sum(
                    1
                    for (x, y) in self._visited_cells
                    if int(self._zone_map[x, y]) == z
                )
                cov[z] = float(visited) / float(denom)
            return cov
        if not self.zone_coverage_stub_single_zone:
            return {}
        in_interior, n_interior = self._interior_cells_spec()
        if n_interior <= 0:
            return {0: 0.0}
        visited = sum(
            1 for (x, y) in self._visited_cells if in_interior(x, y)
        )
        return {0: float(visited) / float(n_interior)}

    # ------------------------------------------------------------------ #
    # SD-005 Observation construction                                      #
    # ------------------------------------------------------------------ #

    def _get_observation_dict(self) -> Dict[str, torch.Tensor]:
        """
        Build the SD-005 split observation dict.

        Returns dict with keys:
          "body_state":   [body_obs_dim]  — proprioceptive/interoceptive
          "world_state":  [world_obs_dim] — exteroceptive
          "contamination_view": [25]      — convenience subset

        body_state channels (use_proxy_fields=False):
          [0]: agent_x / size  (normalised)
          [1]: agent_y / size  (normalised)
          [2]: agent_health
          [3]: agent_energy
          [4]: footprint_density at current cell
          [5-8]: last action one-hot (4 actions: up/down/left/right)
          [9]: steps / 500 (normalised episode progress)

        Additional body_state channels (use_proxy_fields=True):
          [10]: harm_exposure (nociceptive EMA)
          [11]: benefit_exposure (hedonic EMA)

        world_state channels (use_proxy_fields=False):
          [0:175]:   local_view (5×5×7 entity types, one-hot flattened)
          [175:200]: contamination_view (5×5 float, normalised)

        Additional world_state channels (use_proxy_fields=True):
          [200:225]: hazard_field_view (5×5, normalised by max field value)
          [225:250]: resource_field_view (5×5, normalised by max field value)
        """
        ax, ay = self.agent_x, self.agent_y

        # --- body_state ---
        body = torch.zeros(self.body_obs_dim)
        body[0] = ax / self.size
        body[1] = ay / self.size
        body[2] = self.agent_health
        body[3] = self.agent_energy
        max_vis = max(1, self.footprint_grid.max())
        body[4] = float(self.footprint_grid[ax, ay]) / max_vis
        action_enc = self._last_action if self._last_action < 4 else 0
        body[5 + action_enc] = 1.0  # one-hot last action (indices 5,6,7,8)
        body[9] = min(1.0, self.steps / 500.0)
        if self.use_proxy_fields:
            body[10] = float(np.clip(self.harm_exposure, 0.0, 1.0))
            body[11] = float(np.clip(self.benefit_exposure, 0.0, 1.0))
            # SD-022: append directional limb damage state to body_state (12 -> 17 dims).
            # [12]: damage[N], [13]: damage[E], [14]: damage[S], [15]: damage[W]
            # [16]: residual_pain = sum(damage) * residual_pain_scale
            if self.limb_damage_enabled:
                residual_pain = float(np.sum(self.limb_damage) * self.residual_pain_scale)
                body[12] = float(self.limb_damage[0])
                body[13] = float(self.limb_damage[1])
                body[14] = float(self.limb_damage[2])
                body[15] = float(self.limb_damage[3])
                body[16] = float(np.clip(residual_pain, 0.0, 1.0))

        # --- local_view (5×5×7) → world_state part 1 ---
        local_view = torch.zeros(5, 5, self.NUM_ENTITY_TYPES)
        for di in range(-2, 3):
            for dj in range(-2, 3):
                if self.toroidal:
                    ni, nj = (ax + di) % self.size, (ay + dj) % self.size
                    etype = self.grid[ni, nj]
                else:
                    ni, nj = ax + di, ay + dj
                    if 0 <= ni < self.size and 0 <= nj < self.size:
                        etype = self.grid[ni, nj]
                    else:
                        etype = self.ENTITY_TYPES["wall"]
                local_view[di + 2, dj + 2, etype] = 1.0
        local_view_flat = local_view.reshape(-1)  # [175]

        # --- contamination_view (5×5) → world_state part 2 ---
        cont_view = torch.zeros(5, 5)
        for di in range(-2, 3):
            for dj in range(-2, 3):
                if self.toroidal:
                    ni, nj = (ax + di) % self.size, (ay + dj) % self.size
                    cont_view[di + 2, dj + 2] = float(self.contamination_grid[ni, nj])
                else:
                    ni, nj = ax + di, ay + dj
                    if 0 <= ni < self.size and 0 <= nj < self.size:
                        cont_view[di + 2, dj + 2] = float(self.contamination_grid[ni, nj])
        cont_view_flat = (cont_view / (self.contamination_threshold + 1e-6)).reshape(-1)  # [25]

        world_parts = [local_view_flat, cont_view_flat]

        # --- proxy-gradient field views (use_proxy_fields=True only) ---
        hazard_field_flat = torch.zeros(25)
        resource_field_flat = torch.zeros(25)
        if self.use_proxy_fields:
            hazard_max = float(self.hazard_field.max()) + 1e-6
            resource_max = float(self.resource_field.max()) + 1e-6
            h_view = torch.zeros(5, 5)
            r_view = torch.zeros(5, 5)
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    if self.toroidal:
                        ni, nj = (ax + di) % self.size, (ay + dj) % self.size
                        h_view[di + 2, dj + 2] = float(self.hazard_field[ni, nj]) / hazard_max
                        r_view[di + 2, dj + 2] = float(self.resource_field[ni, nj]) / resource_max * self.resource_obs_scale
                    else:
                        ni, nj = ax + di, ay + dj
                        if 0 <= ni < self.size and 0 <= nj < self.size:
                            h_view[di + 2, dj + 2] = float(self.hazard_field[ni, nj]) / hazard_max
                            r_view[di + 2, dj + 2] = float(self.resource_field[ni, nj]) / resource_max * self.resource_obs_scale
            hazard_field_flat = h_view.reshape(-1)    # [25]
            resource_field_flat = r_view.reshape(-1)  # [25]
            world_parts.extend([hazard_field_flat, resource_field_flat])

        # SD-023: landmark gradient field views (only when use_proxy_fields=True and landmarks enabled).
        landmark_a_flat = torch.zeros(25)
        landmark_b_flat = torch.zeros(25)
        if self.use_proxy_fields and (self.n_landmarks_a > 0 or self.n_landmarks_b > 0):
            la_max = float(self._landmark_a_field.max()) + 1e-6
            lb_max = float(self._landmark_b_field.max()) + 1e-6
            la_view = torch.zeros(5, 5)
            lb_view = torch.zeros(5, 5)
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    if self.toroidal:
                        ni, nj = (ax + di) % self.size, (ay + dj) % self.size
                        la_view[di + 2, dj + 2] = float(self._landmark_a_field[ni, nj]) / la_max
                        lb_view[di + 2, dj + 2] = float(self._landmark_b_field[ni, nj]) / lb_max
                    else:
                        ni, nj = ax + di, ay + dj
                        if 0 <= ni < self.size and 0 <= nj < self.size:
                            la_view[di + 2, dj + 2] = float(self._landmark_a_field[ni, nj]) / la_max
                            lb_view[di + 2, dj + 2] = float(self._landmark_b_field[ni, nj]) / lb_max
            landmark_a_flat = la_view.reshape(-1)   # [25]
            landmark_b_flat = lb_view.reshape(-1)   # [25]
            world_parts.extend([landmark_a_flat, landmark_b_flat])

        # SD-049: per-resource-type proximity field views (only when
        # multi_resource_heterogeneity_enabled AND use_proxy_fields). One 5x5
        # patch per type, normalised by per-type field max so the encoder sees
        # type-distinct gradients with comparable magnitudes regardless of
        # absolute spawn density. Stored for obs_dict surfacing below.
        per_type_field_flats: List[torch.Tensor] = []
        if self.use_proxy_fields and self.multi_resource_heterogeneity_enabled:
            for type_idx in range(self.n_resource_types):
                t_field = self._resource_field_by_type[type_idx]
                t_max = float(t_field.max()) + 1e-6
                t_view = torch.zeros(5, 5)
                for di in range(-2, 3):
                    for dj in range(-2, 3):
                        if self.toroidal:
                            ni, nj = (ax + di) % self.size, (ay + dj) % self.size
                            t_view[di + 2, dj + 2] = float(t_field[ni, nj]) / t_max
                        else:
                            ni, nj = ax + di, ay + dj
                            if 0 <= ni < self.size and 0 <= nj < self.size:
                                t_view[di + 2, dj + 2] = float(t_field[ni, nj]) / t_max
                t_flat = t_view.reshape(-1).float()  # [25]
                per_type_field_flats.append(t_flat)
                world_parts.append(t_flat)

        # Behavioral diversity: reef scent gradient field view (25 dims when reef_enabled).
        reef_field_flat = torch.zeros(25)
        if self.use_proxy_fields and self.reef_enabled and self._reef_cells:
            reef_max = float(self._reef_field.max()) + 1e-6
            rf_view = torch.zeros(5, 5)
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    if self.toroidal:
                        ni, nj = (ax + di) % self.size, (ay + dj) % self.size
                        rf_view[di + 2, dj + 2] = float(self._reef_field[ni, nj]) / reef_max
                    else:
                        ni, nj = ax + di, ay + dj
                        if 0 <= ni < self.size and 0 <= nj < self.size:
                            rf_view[di + 2, dj + 2] = float(self._reef_field[ni, nj]) / reef_max
            reef_field_flat = rf_view.reshape(-1)   # [25]
            world_parts.append(reef_field_flat)

        world_state = torch.cat(world_parts)

        result = {
            "body_state": body.float(),
            "world_state": world_state.float(),
            "contamination_view": cont_view_flat.float(),
        }
        if self.use_proxy_fields:
            result["hazard_field_view"] = hazard_field_flat.float()
            result["resource_field_view"] = resource_field_flat.float()
            # SD-023: landmark gradient field views (only when landmarks enabled).
            if self.n_landmarks_a > 0 or self.n_landmarks_b > 0:
                result["landmark_a_field_view"] = landmark_a_flat.float()
                result["landmark_b_field_view"] = landmark_b_flat.float()
            # SD-049: per-resource-type field views + per-axis drive vector + the
            # at-agent type tag for the consumed cell (0 = no resource at agent
            # cell, type_idx + 1 otherwise -- a one-step proxy for the identity
            # signal the SD-015 + downstream Phase 2 encoder upgrade will consume).
            if self.multi_resource_heterogeneity_enabled:
                for i, t_flat in enumerate(per_type_field_flats):
                    name = self.resource_type_names[i]
                    result[f"resource_field_view_{name}"] = t_flat
                # Always emit the per-axis drive vector and resource_type_at_agent
                # tag when SD-049 is on (zero values when types are gated out).
                result["per_axis_drive"] = torch.from_numpy(
                    self._per_axis_drive.copy()
                ).float()
                result["resource_type_at_agent"] = torch.tensor(
                    [int(self._resource_type_grid[ax, ay])], dtype=torch.int64
                )
            # Behavioral diversity: reef scent gradient field view.
            if self.reef_enabled:
                result["reef_field_view"] = reef_field_flat.float()
            # SD-010: dedicated harm_obs for HarmEncoder (nociceptive separation).
            # Sensory-discriminative stream (z_harm_s, Adelta-pathway analog):
            # Layout: hazard_field_view[25] + resource_field_view[25] + harm_exposure[1]
            result["harm_obs"] = torch.cat([
                hazard_field_flat.float(),
                resource_field_flat.float(),
                torch.tensor([float(np.clip(self.harm_exposure, 0.0, 1.0))]),
            ], dim=0)  # [51]
            # SD-011: harm_obs_a for AffectiveHarmEncoder (affective-motivational stream,
            # C-fiber/paleospinothalamic analog). EMA of proximity fields at slower tau
            # (~20 steps vs ~10 for harm_exposure). Represents accumulated homeostatic
            # threat state, not immediate proximity. Does NOT need a forward model.
            # SD-022: when limb_damage_enabled, re-source harm_obs_a from body damage state
            # (7 dims: damage[4] + max_damage + mean_damage + residual_pain) instead of the
            # 50-dim EMA. This provides causal independence: agent in safe area with
            # accumulated damage has high harm_obs_a but near-zero harm_obs (world signal).
            if self.limb_damage_enabled:
                _residual_pain = float(np.sum(self.limb_damage) * self.residual_pain_scale)
                harm_obs_a_body = np.array([
                    float(self.limb_damage[0]),
                    float(self.limb_damage[1]),
                    float(self.limb_damage[2]),
                    float(self.limb_damage[3]),
                    float(np.max(self.limb_damage)),
                    float(np.mean(self.limb_damage)),
                    float(np.clip(_residual_pain, 0.0, 1.0)),
                ], dtype=np.float32)
                # SD-048: apply interoceptive-noise perturbations to harm_obs_a
                # readout (no-op when interoceptive_noise_enabled=False).
                harm_obs_a_body = self._apply_interoceptive_noise(harm_obs_a_body)
                result["harm_obs_a"] = torch.from_numpy(harm_obs_a_body)  # [7]
            else:
                harm_obs_a_legacy = self.harm_obs_a_ema.copy()
                # SD-048: same readout-side perturbation on the legacy 50-dim path.
                harm_obs_a_legacy = self._apply_interoceptive_noise(harm_obs_a_legacy)
                result["harm_obs_a"] = torch.from_numpy(harm_obs_a_legacy).float()  # [50]
            # SD-011 second source: rolling harm history and accumulated harm target.
            if self.harm_history_len > 0:
                # FIFO oldest-first: roll so oldest entry comes first.
                rolled = np.roll(self._harm_history, -self._harm_history_ptr)
                result["harm_history"] = torch.from_numpy(rolled.copy()).float()  # [harm_history_len]
                # Accumulated harm target for auxiliary loss (running average, clipped [0,1]).
                accum = self._accumulated_harm_exposure / max(self._accumulated_harm_steps, 1)
                result["accumulated_harm"] = float(np.clip(accum, 0.0, 1.0))
        return result

    def _dict_to_flat(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate body_state + world_state into flat observation vector."""
        return torch.cat([obs_dict["body_state"], obs_dict["world_state"]]).float()

    # V2 backward-compat method
    def _get_observation(self) -> torch.Tensor:
        return self._dict_to_flat(self._get_observation_dict())

    # ------------------------------------------------------------------ #
    # SD-023: Landmark placement and gradient field computation           #
    # ------------------------------------------------------------------ #

    def _place_random_landmarks(
        self, n: int, available: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Place n landmarks at random positions from remaining available cells.

        Does not remove from `available` (landmarks don't block other placement).
        Returns list of (x, y) tuples.
        """
        if n <= 0 or not available:
            return []
        idxs = self._rng.choice(len(available), size=min(n, len(available)), replace=False)
        return [available[i] for i in idxs]

    def _place_biased_near_resources(
        self,
        n: int,
        bias_prob: float,
        radius: int,
        available: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """Place n landmark-B objects with a bias toward resource proximity.

        Each landmark is placed near a resource (within `radius` cells) with
        probability `bias_prob`, otherwise placed randomly. Landmarks may
        share cells with other objects (gradient only; no grid entity type).

        Returns list of (x, y) tuples.
        """
        if n <= 0:
            return []
        positions: List[Tuple[int, int]] = []
        for _ in range(n):
            if self.resources and self._rng.random() < bias_prob:
                # Pick a random resource and place near it.
                res = self.resources[self._rng.integers(0, len(self.resources))]
                rx, ry = int(res[0]), int(res[1])
                candidates = []
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if dx == 0 and dy == 0:
                            continue
                        cx, cy = rx + dx, ry + dy
                        if 0 <= cx < self.size and 0 <= cy < self.size:
                            candidates.append((cx, cy))
                if candidates:
                    idx = self._rng.integers(0, len(candidates))
                    positions.append(candidates[idx])
                    continue
            # Fallback: place randomly in available cells.
            if available:
                idx = self._rng.integers(0, len(available))
                positions.append(available[idx])
            elif self.size > 2:
                # If available is exhausted, pick any interior cell.
                cx = int(self._rng.integers(1, self.size - 1))
                cy = int(self._rng.integers(1, self.size - 1))
                positions.append((cx, cy))
        return positions

    def _compute_landmark_field(
        self,
        positions: List[Tuple[int, int]],
        sigma: float,
        scale: float,
    ) -> np.ndarray:
        """Compute a Gaussian gradient field for a set of landmark positions.

        field[x, y] = sum_i scale * exp(-d2_i / (2 * sigma^2))
        where d2_i is squared Euclidean distance from (x,y) to landmark i.
        Field is static per episode (landmarks do not move).
        """
        field = np.zeros((self.size, self.size), dtype=np.float32)
        if not positions:
            return field
        two_sigma2 = 2.0 * sigma * sigma
        for lx, ly in positions:
            for x in range(self.size):
                for y in range(self.size):
                    d2 = float((x - lx) ** 2 + (y - ly) ** 2)
                    field[x, y] += scale * float(np.exp(-d2 / two_sigma2))
        return field

    # ------------------------------------------------------------------ #
    # Behavioral diversity: reef safe zones                               #
    # ------------------------------------------------------------------ #

    def _place_reef_patches(self, available: list) -> None:
        """Place reef safe zones at fixed corner areas and remove from spawn pool.

        Reef cells are positioned at grid corners using Manhattan-radius patches so
        their location is predictable and stable across episodes. Hazards cannot
        enter reef cells (_drift_hazards exclusion) and agents take no harm there
        (step() guard). The reef scent gradient field is precomputed here and
        remains static for the episode (same as SD-023 landmark fields).

        Modifies `available` in-place to remove reef cell positions.
        """
        sz = self.size
        self._reef_cells = set()
        if self.n_reef_patches <= 0 or sz < 5:
            self._reef_field = np.zeros((sz, sz), dtype=np.float32)
            return

        # Fixed corner centres (interior cells at distance 2 from each corner wall).
        # Order: top-left, top-right, bottom-left, bottom-right, then mid-edges.
        corner_centres = [
            (2, 2),
            (2, sz - 3),
            (sz - 3, 2),
            (sz - 3, sz - 3),
            (2, sz // 2),
            (sz - 3, sz // 2),
        ]
        centres = corner_centres[: self.n_reef_patches]

        for cx, cy in centres:
            for i in range(1, sz - 1):
                for j in range(1, sz - 1):
                    if abs(i - cx) + abs(j - cy) <= self.reef_patch_radius:
                        self._reef_cells.add((i, j))

        # Compute static reef scent field: sum of Manhattan-decay kernels.
        field = np.zeros((sz, sz), dtype=np.float32)
        for rx, ry in self._reef_cells:
            for i in range(sz):
                for j in range(sz):
                    dist = abs(i - rx) + abs(j - ry)
                    field[i, j] += float(np.exp(-dist / self.reef_scent_sigma))
        max_val = float(field.max())
        if max_val > 0.0:
            field /= max_val
        self._reef_field = field

        # Remove reef cells from the spawn pool so hazards and resources never start there.
        if self._reef_cells:
            available[:] = [(x, y) for (x, y) in available if (x, y) not in self._reef_cells]

    # ------------------------------------------------------------------ #
    # SD-054 bipartite layout extension (2026-05-11)                      #
    # ------------------------------------------------------------------ #

    def _place_reef_patches_bipartite(self) -> None:
        """SD-054 bipartite layout: place reef patches in the reef-half of the grid.

        Replaces the legacy `_place_reef_patches` corner-pattern when
        `reef_bipartite_layout=True`. Reef patch centres are positioned along
        the reef-half edge (bottom edge for axis="horizontal" with reef-bottom
        convention; right edge for axis="vertical" with reef-right convention).
        n_reef_patches patches are placed evenly spaced along that edge.

        The reef scent field is computed identically to the legacy helper. The
        spawn pool is NOT modified here -- partitioning is handled separately
        by `_build_bipartite_pools()`, which respects the reef_half / agent_band
        / forage_half partition independently.

        Modifies `self._reef_cells` and `self._reef_field` in place.
        """
        sz = self.size
        self._reef_cells = set()
        if self.n_reef_patches <= 0 or sz < 5:
            self._reef_field = np.zeros((sz, sz), dtype=np.float32)
            return

        radius = self.reef_bipartite_agent_band_radius
        midline = sz // 2

        if self.reef_bipartite_axis == "horizontal":
            # Reef in bottom rows (row > midline + radius). Edge row for centres:
            # sz - 3 (the conventional inner edge offset, matching legacy corner
            # centres). Column positions distributed across the available range.
            edge_row = sz - 3
            # Choose column centres: for n=1, mid; for n=2, left + right; for
            # n>=3, equally spaced across the interior. Match legacy aesthetics
            # so the n=3 default tile [(2, c0), (mid, c1), (sz-3, c2)]-like.
            interior_cols = list(range(2, sz - 2))
            if self.n_reef_patches == 1:
                col_centres = [sz // 2]
            else:
                step = max(1, (sz - 5) // max(1, self.n_reef_patches - 1))
                col_centres = [2 + step * i for i in range(self.n_reef_patches)]
                col_centres[-1] = min(col_centres[-1], sz - 3)
            centres = [(edge_row, c) for c in col_centres]
        else:  # "vertical"
            # Reef in right columns (col > midline + radius).
            edge_col = sz - 3
            if self.n_reef_patches == 1:
                row_centres = [sz // 2]
            else:
                step = max(1, (sz - 5) // max(1, self.n_reef_patches - 1))
                row_centres = [2 + step * i for i in range(self.n_reef_patches)]
                row_centres[-1] = min(row_centres[-1], sz - 3)
            centres = [(r, edge_col) for r in row_centres]

        for cx, cy in centres:
            for i in range(1, sz - 1):
                for j in range(1, sz - 1):
                    if abs(i - cx) + abs(j - cy) <= self.reef_patch_radius:
                        # Guard: only add cells that are on the reef-side of
                        # the partition. Cells in the agent band or forage half
                        # are never reef cells under bipartite layout.
                        if self._is_in_reef_half(i, j):
                            self._reef_cells.add((i, j))

        # Compute static reef scent field: sum of Manhattan-decay kernels.
        field = np.zeros((sz, sz), dtype=np.float32)
        for rx, ry in self._reef_cells:
            for i in range(sz):
                for j in range(sz):
                    dist = abs(i - rx) + abs(j - ry)
                    field[i, j] += float(np.exp(-dist / self.reef_scent_sigma))
        max_val = float(field.max())
        if max_val > 0.0:
            field /= max_val
        self._reef_field = field

    def _is_in_reef_half(self, i: int, j: int) -> bool:
        """SD-054 bipartite: is (i, j) on the reef side of the partition?"""
        midline = self.size // 2
        radius = self.reef_bipartite_agent_band_radius
        if self.reef_bipartite_axis == "horizontal":
            return i > midline + radius
        else:  # "vertical"
            return j > midline + radius

    # ------------------------------------------------------------------ #
    # infant_substrate:GAP-2 -- microhabitat zones                        #
    # ------------------------------------------------------------------ #

    def _build_microhabitat_zones(self, interior_cells: list) -> None:
        """Build a per-episode Voronoi zone map over interior cells.

        n_microhabitats seed points are sampled (without replacement) from
        the interior cells via self._rng. Each interior cell is assigned to
        the nearest seed (Euclidean) -> a base zone index 0..n-1. A cell
        whose 4-neighbourhood contains a different base zone is promoted to
        the automatic transition/border zone D (code 3). Zone codes:
        -1 = wall / non-interior, 0 = A, 1 = B, 2 = C, 3 = D border,
        >=4 = extra Voronoi zone when n_microhabitats > 3 (neutral factors).

        Per the design doc (Section 5.2): zone identity determines per-cell
        baseline resource/hazard spawn weighting; D promotes boundary
        exploration with neutral factors. Only called when enabled.

        Sets self._zone_map (np.int8 [size, size]).
        """
        sz = self.size
        zone_map = np.full((sz, sz), -1, dtype=np.int8)
        if not interior_cells:
            self._zone_map = zone_map
            return

        n_seeds = min(self.n_microhabitats, len(interior_cells))
        seed_idx = self._rng.choice(
            len(interior_cells), size=n_seeds, replace=False
        )
        seeds = [interior_cells[int(k)] for k in np.atleast_1d(seed_idx)]

        # Base Voronoi assignment (nearest seed by squared Euclidean).
        for (i, j) in interior_cells:
            best_k = 0
            best_dsq = float("inf")
            for k, (sx, sy) in enumerate(seeds):
                dsq = (sx - i) ** 2 + (sy - j) ** 2
                if dsq < best_dsq:
                    best_dsq = dsq
                    best_k = k
            zone_map[i, j] = best_k

        # Promote boundary cells to zone D (code 3): a cell adjacent (4-conn)
        # to a different base zone. Computed against the base assignment so
        # promotion does not cascade.
        base = zone_map.copy()
        for (i, j) in interior_cells:
            z = int(base[i, j])
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < sz and 0 <= nj < sz:
                    nz = int(base[ni, nj])
                    if nz != -1 and nz != z:
                        zone_map[i, j] = 3
                        break
        self._zone_map = zone_map

    def _zone_resource_factor(self, cell: Tuple[int, int]) -> float:
        """Resource spawn weight for a cell's zone. 1.0 when no zone map."""
        if self._zone_map is None:
            return 1.0
        z = int(self._zone_map[cell[0], cell[1]])
        if z == 0:
            return self.zone_A_resource_factor
        if z == 1:
            return self.zone_B_resource_factor
        if z == 2:
            return self.zone_C_resource_factor
        return 1.0  # D border, extra Voronoi zone, or wall: neutral

    def _zone_hazard_factor(self, cell: Tuple[int, int]) -> float:
        """Hazard spawn weight for a cell's zone. 1.0 when no zone map."""
        if self._zone_map is None:
            return 1.0
        z = int(self._zone_map[cell[0], cell[1]])
        if z == 0:
            return self.zone_A_hazard_factor
        if z == 1:
            return self.zone_B_hazard_factor
        if z == 2:
            return self.zone_C_hazard_factor
        return 1.0  # D border, extra Voronoi zone, or wall: neutral

    def _pop_zone_weighted(self, pool: list, kind: str) -> Tuple[int, int]:
        """Pop a cell from `pool` with probability proportional to its zone
        spawn factor for `kind` ("resource" or "hazard").

        Falls back to a uniform pop() when microhabitat is disabled, the
        zone map is absent, or every candidate weight is zero (degenerate
        config). Only called from the enabled spawn paths -- the disabled
        path uses a bare pool.pop() so legacy RNG sequences are
        bit-identical.
        """
        if self._zone_map is None or not pool:
            return pool.pop()
        if kind == "hazard":
            weights = np.array(
                [self._zone_hazard_factor(c) for c in pool], dtype=np.float64
            )
        else:
            weights = np.array(
                [self._zone_resource_factor(c) for c in pool], dtype=np.float64
            )
        total = weights.sum()
        if total <= 0.0:
            return pool.pop()
        idx = int(self._rng.choice(len(pool), p=weights / total))
        return pool.pop(idx)

    def _is_in_forage_half(self, i: int, j: int) -> bool:
        """SD-054 bipartite: is (i, j) on the forage side of the partition?"""
        midline = self.size // 2
        radius = self.reef_bipartite_agent_band_radius
        if self.reef_bipartite_axis == "horizontal":
            return i < midline - radius
        else:  # "vertical"
            return j < midline - radius

    def _is_in_agent_band(self, i: int, j: int) -> bool:
        """SD-054 bipartite: is (i, j) within the agent spawn band?"""
        midline = self.size // 2
        radius = self.reef_bipartite_agent_band_radius
        if self.reef_bipartite_axis == "horizontal":
            return abs(i - midline) <= radius
        else:  # "vertical"
            return abs(j - midline) <= radius

    def _build_bipartite_pools(self, available: list) -> Tuple[list, list]:
        """SD-054 bipartite: partition `available` into (agent_pool, forage_pool).

        agent_pool: cells in the agent spawn band (midline +/- agent_band_radius)
                    that are not reef cells.
        forage_pool: cells in the forage half (opposite side from reef) that are
                     not reef cells.

        Cells in the reef half but not used as reef cells are dropped from both
        pools -- they are intentionally inaccessible to non-reef entities under
        bipartite layout. This is a feature: the reef half is reef-exclusive,
        the forage half is forage-exclusive, and the agent band is the only
        contestable territory.

        Fallback: if agent_pool would be empty (e.g., agent_band_radius=0 on a
        grid where the midline is mostly walls), widen the band by +1 radius
        until at least one valid cell exists. Logs a diagnostic via a per-reset
        counter. If forage_pool is empty, fall back to using agent_pool for
        hazards/resources (degenerate but won't crash).
        """
        sz = self.size
        reef = self._reef_cells

        def _filter(predicate):
            return [
                (x, y) for (x, y) in available
                if predicate(x, y) and (x, y) not in reef
            ]

        agent_pool = _filter(self._is_in_agent_band)
        forage_pool = _filter(self._is_in_forage_half)

        # Fallback: widen agent band if empty (rare; only on degenerate sizes).
        _band_widen_count = 0
        while not agent_pool and _band_widen_count < sz:
            _band_widen_count += 1
            midline = sz // 2
            r = self.reef_bipartite_agent_band_radius + _band_widen_count
            if self.reef_bipartite_axis == "horizontal":
                widened_predicate = lambda x, y: abs(x - midline) <= r
            else:
                widened_predicate = lambda x, y: abs(y - midline) <= r
            agent_pool = [
                (x, y) for (x, y) in available
                if widened_predicate(x, y) and (x, y) not in reef
            ]
        self._sd054_bipartite_band_widen_count = _band_widen_count
        # Defensive: if forage pool empty (degenerate config), fall back to
        # agent_pool so the reset doesn't crash on hazards.pop(). The
        # substrate-readiness diagnostic will catch this via low forage-pool
        # cell counts in info diagnostics.
        if not forage_pool:
            forage_pool = list(agent_pool)

        # Shuffle each pool independently for determinism under self._rng.
        self._rng.shuffle(agent_pool)
        self._rng.shuffle(forage_pool)
        return agent_pool, forage_pool

    # ------------------------------------------------------------------ #
    # Proxy-gradient field computation (ARC-024)                          #
    # ------------------------------------------------------------------ #

    def _compute_proximity_fields(self) -> None:
        """
        Compute hazard and resource proximity fields across the full grid.

        Field value at (i,j): sum over all sources of 1 / (1 + dist * decay).
        Uses Manhattan distance. Peaks at 1.0 at source cell (dist=0).

        Called after placement, after drift (hazards), after consumption (resources).

        SD-047: when multi_source_dynamics is on, transient hazards and drift sources
        contribute to hazard_field alongside self.hazards (they are real hazards on
        the grid; the per-source lists exist for bookkeeping). The weather field is
        added as an additive coarse-grid perturbation at the end -- agent-independent
        continuous noise on the agent's hazard percept.
        """
        self.hazard_field = np.zeros((self.size, self.size), dtype=np.float32)
        # SD-047: aggregate hazard sources. self.hazards already holds the entries
        # for transient and drift hazards (added when each spawns); _transient_hazards
        # and _drift_sources are the per-source bookkeeping lists, NOT separate
        # contributors to the field.
        for hx, hy in self.hazards:
            for i in range(self.size):
                for j in range(self.size):
                    dist = abs(i - hx) + abs(j - hy)
                    self.hazard_field[i, j] += 1.0 / (1.0 + dist * self.hazard_field_decay)

        # SD-047: add weather perturbation (additive, can be negative). Clipped at 0
        # to preserve the field's non-negative semantics expected by downstream
        # consumers (proximity_harm = scale * field; resource and harm encoders
        # expect non-negative inputs).
        if (
            self.multi_source_dynamics_enabled
            and self.weather_field_enabled
            and self._weather_perturbation.shape == self.hazard_field.shape
        ):
            self.hazard_field = np.maximum(
                0.0, self.hazard_field + self._weather_perturbation
            ).astype(np.float32)

        self.resource_field = np.zeros((self.size, self.size), dtype=np.float32)
        for rx, ry in self.resources:
            for i in range(self.size):
                for j in range(self.size):
                    dist = abs(i - rx) + abs(j - ry)
                    self.resource_field[i, j] += 1.0 / (1.0 + dist * self.resource_field_decay)

        # SD-049: per-resource-type proximity fields (parallel to legacy
        # resource_field). Computed only when master switch is on; otherwise
        # the per-type array stays zero and is not surfaced through obs_dict.
        if self.multi_resource_heterogeneity_enabled:
            self._resource_field_by_type = np.zeros(
                (self.n_resource_types, self.size, self.size), dtype=np.float32
            )
            for type_idx, type_resources in enumerate(self._resources_by_type):
                for rx, ry in type_resources:
                    for i in range(self.size):
                        for j in range(self.size):
                            dist = abs(i - rx) + abs(j - ry)
                            self._resource_field_by_type[type_idx, i, j] += (
                                1.0 / (1.0 + dist * self.resource_field_decay)
                            )

    # ------------------------------------------------------------------ #
    # SD-047: Multi-source environmental dynamics                         #
    # ------------------------------------------------------------------ #

    def _init_multi_source_state(self) -> None:
        """
        Reset SD-047 multi-source state on episode boundary.

        Weather super_field reseeds at zero (stationary AR(1) starts at the
        unconditional mean). Transient hazards and drift sources are emptied;
        drift sources are then placed at random interior cells when the
        per-source switch is on. Drift cells are added to self.hazards so
        proximity-field computation includes them.
        """
        # Weather coarse field: zero-init each episode (stationary mean of AR(1)).
        self._weather_super_field = np.zeros(
            (self.weather_super_cells, self.weather_super_cells), dtype=np.float32
        )
        self._weather_perturbation = np.zeros((self.size, self.size), dtype=np.float32)
        # Transient hazards: empty at episode start; populated stochastically by
        # _step_transient_events.
        self._transient_hazards = []
        # Drift sources: place at random interior cells when enabled. Each drift
        # source is also entered into self.hazards so it shows up in the grid
        # (entity type "hazard") and contributes to hazard_field.
        self._drift_sources = []
        if self.background_drift_enabled and self.n_drift_sources > 0:
            if self.toroidal:
                pool = [
                    (i, j)
                    for i in range(self.size)
                    for j in range(self.size)
                    if self.grid[i, j] == self.ENTITY_TYPES["empty"]
                ]
            else:
                pool = [
                    (i, j)
                    for i in range(1, self.size - 1)
                    for j in range(1, self.size - 1)
                    if self.grid[i, j] == self.ENTITY_TYPES["empty"]
                ]
            self._rng.shuffle(pool)
            n = min(self.n_drift_sources, len(pool))
            for _ in range(n):
                if not pool:
                    break
                cx, cy = pool.pop()
                # Random initial velocity for linear/Levy policies (random_walk ignores it).
                vx, vy = 0, 0
                if self.drift_policy in ("linear_drift", "levy_walk"):
                    vchoices = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    vx, vy = vchoices[int(self._rng.integers(0, len(vchoices)))]
                self._drift_sources.append([int(cx), int(cy), int(vx), int(vy), 0])
                self.grid[cx, cy] = self.ENTITY_TYPES["hazard"]
                self.hazards.append([int(cx), int(cy)])

    def _step_weather_field(self) -> float:
        """
        AR(1) update on the coarse weather super-field; bilinear-interpolate to
        per-cell additive perturbation.

        Stationary AR(1) form: x_{t+1} = alpha*x_t + sqrt(1-alpha^2) * sigma * N(0, 1).
        Variance stays bounded at sigma^2 (no blowup over long episodes).

        Returns the mean absolute change in the super-field this tick (diagnostic).
        """
        alpha = float(np.clip(self.weather_alpha_ar1, 0.0, 0.999))
        sigma = float(self.weather_sigma) * float(self.multi_source_intensity_scale)
        # noise scale preserves stationary variance sigma^2 across alpha values.
        noise_scale = float(np.sqrt(max(0.0, 1.0 - alpha * alpha))) * sigma
        prev = self._weather_super_field.copy()
        noise = self._rng.standard_normal(self._weather_super_field.shape).astype(np.float32)
        self._weather_super_field = (alpha * self._weather_super_field + noise_scale * noise).astype(np.float32)
        # Map super-field to full grid via nearest-neighbour block expansion.
        block_h = max(1, self.size // self.weather_super_cells)
        block_w = max(1, self.size // self.weather_super_cells)
        pert = np.zeros((self.size, self.size), dtype=np.float32)
        for si in range(self.weather_super_cells):
            for sj in range(self.weather_super_cells):
                lo_i = si * block_h
                hi_i = min(self.size, (si + 1) * block_h) if si < self.weather_super_cells - 1 else self.size
                lo_j = sj * block_w
                hi_j = min(self.size, (sj + 1) * block_w) if sj < self.weather_super_cells - 1 else self.size
                pert[lo_i:hi_i, lo_j:hi_j] = self._weather_super_field[si, sj]
        self._weather_perturbation = pert
        return float(np.mean(np.abs(self._weather_super_field - prev)))

    def _step_transient_events(self) -> Tuple[int, int]:
        """
        Poisson appear / disappear of transient hazard cells.

        Each cell independently has p_appear * intensity_scale chance per tick of
        spawning a transient hazard (only on currently empty cells; never on the
        agent). Each existing transient has p_disappear chance per tick of removing.

        Returns (n_appeared, n_disappeared) counts for this tick.
        """
        scale = float(self.multi_source_intensity_scale)
        p_appear = float(np.clip(self.transient_p_appear * scale, 0.0, 1.0))
        p_disappear = float(np.clip(self.transient_p_disappear, 0.0, 1.0))
        n_appeared = 0
        n_disappeared = 0
        # Disappearances first (compress survivors, free their grid cells if not
        # otherwise occupied).
        survivors: List[List[int]] = []
        for entry in self._transient_hazards:
            tx, ty, age = int(entry[0]), int(entry[1]), int(entry[2])
            if self._rng.random() < p_disappear:
                if self.grid[tx, ty] == self.ENTITY_TYPES["hazard"]:
                    self.grid[tx, ty] = self.ENTITY_TYPES["empty"]
                self.hazards = [h for h in self.hazards if not (h[0] == tx and h[1] == ty)]
                n_disappeared += 1
            else:
                survivors.append([tx, ty, age + 1])
        self._transient_hazards = survivors
        # Appearances: independent per-cell Bernoulli on empty cells (excluding agent).
        if p_appear > 0.0:
            if self.toroidal:
                cells = [
                    (i, j)
                    for i in range(self.size)
                    for j in range(self.size)
                    if self.grid[i, j] == self.ENTITY_TYPES["empty"]
                ]
            else:
                cells = [
                    (i, j)
                    for i in range(1, self.size - 1)
                    for j in range(1, self.size - 1)
                    if self.grid[i, j] == self.ENTITY_TYPES["empty"]
                ]
            for (i, j) in cells:
                if self._rng.random() < p_appear:
                    self.grid[i, j] = self.ENTITY_TYPES["hazard"]
                    self.hazards.append([int(i), int(j)])
                    self._transient_hazards.append([int(i), int(j), 0])
                    n_appeared += 1
        return n_appeared, n_disappeared

    def _step_background_drift(self) -> int:
        """
        Update positions of drift sources per drift_policy.

        random_walk:   sample new direction each tick, move if target empty.
        linear_drift:  keep current velocity, bounce off walls / occupied cells.
        levy_walk:     long-step occasional with low probability (~5% per tick),
                       short steps otherwise; resample velocity each move.

        Returns count of drift sources that actually moved this tick.
        """
        scale = float(self.multi_source_intensity_scale)
        # Move-probability scaling: at scale=1.0 every drift source attempts a move
        # per tick. At scale<1.0 some ticks skip; at scale>1.0 the upper bound is
        # one move per source per tick (capped — this is a movement rate, not a
        # multi-step jump).
        p_move = float(np.clip(scale, 0.0, 1.0))
        n_moved = 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for entry in self._drift_sources:
            cx, cy = int(entry[0]), int(entry[1])
            vx, vy = int(entry[2]), int(entry[3])
            if self._rng.random() > p_move:
                continue
            # Pick movement direction per policy.
            if self.drift_policy == "linear_drift":
                dx, dy = vx, vy
                if (dx, dy) == (0, 0):
                    dx, dy = directions[int(self._rng.integers(0, len(directions)))]
            elif self.drift_policy == "levy_walk":
                # 5% chance of a 2-step Levy hop; otherwise a 1-step move.
                hop = 2 if self._rng.random() < 0.05 else 1
                ddx, ddy = directions[int(self._rng.integers(0, len(directions)))]
                dx, dy = ddx * hop, ddy * hop
            else:  # random_walk
                dx, dy = directions[int(self._rng.integers(0, len(directions)))]
            # Compute target.
            if self.toroidal:
                nx = (cx + dx) % self.size
                ny = (cy + dy) % self.size
            else:
                nx, ny = cx + dx, cy + dy
                if not (0 < nx < self.size - 1 and 0 < ny < self.size - 1):
                    # Wall bounce for linear_drift; skip for others.
                    if self.drift_policy == "linear_drift":
                        entry[2] = -vx
                        entry[3] = -vy
                    continue
            if self.grid[nx, ny] != self.ENTITY_TYPES["empty"]:
                # Target occupied: bounce velocity for linear, otherwise skip.
                if self.drift_policy == "linear_drift":
                    entry[2] = -vx
                    entry[3] = -vy
                continue
            # Apply move: clear old grid cell, mark new, update self.hazards entry.
            self.grid[cx, cy] = self.ENTITY_TYPES["empty"]
            self.grid[nx, ny] = self.ENTITY_TYPES["hazard"]
            for h in self.hazards:
                if h[0] == cx and h[1] == cy:
                    h[0], h[1] = int(nx), int(ny)
                    break
            entry[0], entry[1] = int(nx), int(ny)
            entry[4] = int(entry[4]) + 1  # age increment
            # For linear_drift, store the velocity used (so future ticks continue).
            if self.drift_policy == "linear_drift":
                entry[2], entry[3] = int(dx), int(dy)
            n_moved += 1
        return n_moved

    # ------------------------------------------------------------------ #
    # SD-048: Interoceptive noise dynamics                                 #
    # ------------------------------------------------------------------ #

    def _apply_interoceptive_noise(self, harm_obs_a: np.ndarray) -> np.ndarray:
        """
        Apply SD-048 interoceptive-noise perturbations to a harm_obs_a array.

        Three concurrent agent-independent body-state noise sources are layered
        on the readout:
          (1) Fatigue drift   -- AR(1) latent fatigue, additive contribution.
          (2) Sensitisation   -- Poisson onset, multiplicative amplification,
                                 exponential decay.
          (3) Autonomic noise -- per-element i.i.d. Gaussian additive noise.

        The order is biologically motivated: fatigue and sensitisation modulate
        the underlying gain / baseline of the interoceptive readout, then
        autonomic noise is added on top. This keeps autonomic noise a true
        readout-noise floor rather than amplifying it through the multiplier.

        Bit-identical OFF: when interoceptive_noise_enabled=False, returns the
        input unchanged with no RNG draws, no state advance, and zeroed
        per-tick diagnostic counters.

        Per-source bit-identical OFF: when master switch is on but a per-source
        switch is False, that source's RNG draws / state updates are skipped.

        Calibration counter: |delta_harm_obs_a| events vs the previous tick's
        harm_obs_a are classified as agent-caused (transition_type indicates
        the agent caused the change) or body-noise-caused (otherwise). Both
        counts use the same threshold (interoceptive_change_threshold).
        """
        # Reset per-tick counters before any updates.
        self._interoceptive_n_autonomic_events = 0
        self._interoceptive_n_sensitisation_events = 0
        self._interoceptive_n_fatigue_events = 0
        self._interoceptive_n_body_noise_events = 0
        self._interoceptive_n_agent_caused_harm_events = 0

        if not self.interoceptive_noise_enabled:
            # No-op path: do not advance state, do not consume RNG, do not
            # populate _prev_harm_obs_a. Behaviour is bit-identical to legacy.
            return harm_obs_a

        scale = float(max(0.0, self.interoceptive_noise_scale))
        prev = self._prev_harm_obs_a
        # Snapshot input for delta-event detection BEFORE perturbation.
        baseline = harm_obs_a.astype(np.float32, copy=True)
        out = baseline.copy()

        # Source 3 (fatigue drift): AR(1) latent state, additive contribution.
        fatigue_delta_norm = 0.0
        if self.fatigue_enabled:
            sigma_f = self.fatigue_noise_scale * scale
            innovation = float(self._rng.standard_normal()) * sigma_f
            new_fatigue = self.fatigue_ar_coeff * self._fatigue_state + innovation
            fatigue_delta = new_fatigue - self._fatigue_state
            self._fatigue_state = float(new_fatigue)
            additive = self.fatigue_contribution_weight * self._fatigue_state
            out = out + np.float32(additive)
            fatigue_delta_norm = abs(self.fatigue_contribution_weight * fatigue_delta)
            if fatigue_delta_norm > self.interoceptive_change_threshold:
                self._interoceptive_n_fatigue_events += 1

        # Source 2 (sensitisation spikes): Poisson onset, multiplicative
        # amplification, exponential decay.
        sensitisation_delta_norm = 0.0
        if self.sensitisation_enabled:
            decay = float(np.exp(-np.log(2.0) / float(self.sensitisation_halflife)))
            self._sensitisation_amplification = float(
                max(0.0, self._sensitisation_amplification * decay)
            )
            p_event = float(np.clip(self.sensitisation_rate * scale, 0.0, 1.0))
            if p_event > 0.0 and self._rng.random() < p_event:
                # New transient sensitisation event: increment the active
                # multiplicative bias by (magnitude - 1) so e.g. magnitude=1.8
                # contributes 0.8x amplification on top of baseline. Cap the
                # cumulative amplification at 5.0x so long Poisson tails do
                # not blow up the readout.
                self._sensitisation_amplification = float(
                    min(5.0, self._sensitisation_amplification + (self.sensitisation_magnitude - 1.0))
                )
                self._interoceptive_n_sensitisation_events += 1
            if self._sensitisation_amplification > 0.0:
                amplified = out * np.float32(1.0 + self._sensitisation_amplification)
                sensitisation_delta_norm = float(
                    np.mean(np.abs(amplified - out))
                )
                out = amplified

        # Source 1 (autonomic noise): per-element i.i.d. Gaussian additive.
        autonomic_delta_norm = 0.0
        if self.autonomic_noise_enabled:
            sigma_a = self.autonomic_noise_scale * scale
            if sigma_a > 0.0:
                noise = self._rng.standard_normal(size=out.shape).astype(np.float32) * np.float32(sigma_a)
                autonomic_delta_norm = float(np.mean(np.abs(noise)))
                out = out + noise
                if autonomic_delta_norm > self.interoceptive_change_threshold:
                    self._interoceptive_n_autonomic_events += 1

        # Calibration-target classification: count steps where the readout
        # changed by more than the threshold compared to the previous tick.
        # Classify as agent-caused only if the agent's THIS-tick action drove
        # new body damage -- per SD-048 doc: "I moved N through a hazard ->
        # limb N is more damaged". Narrow to transition_type ==
        # "agent_caused_hazard" (the only transition that actually increments
        # limb_damage in step()). Passive proximity readouts (hazard_approach,
        # resource, benefit_approach) and waypoint events are not agent-caused
        # body damage and are excluded; they fall through to body-noise-caused
        # along with all "none" transitions. First tick of each episode
        # (prev is None) counts no event.
        if prev is not None and prev.shape == out.shape:
            delta_l1 = float(np.mean(np.abs(out - prev)))
            if delta_l1 > self.interoceptive_change_threshold:
                if self._last_transition_type == "agent_caused_hazard":
                    self._interoceptive_n_agent_caused_harm_events += 1
                else:
                    self._interoceptive_n_body_noise_events += 1

        # Cache perturbed readout (NOT baseline) for the next tick's delta
        # computation -- the agent / encoder sees the perturbed value, so
        # the "previous observation" the next tick should compare against is
        # also the perturbed value.
        self._prev_harm_obs_a = out.copy()

        return out

    # ------------------------------------------------------------------ #
    # Internal helpers (unchanged from V2)                                #
    # ------------------------------------------------------------------ #

    def _inject_external_hazard(self) -> bool:
        """
        SD-029 balanced-hazard-event curriculum: inject an externally-caused
        hazard event adjacent to the agent (or at any empty cell if
        adjacent_only=False and no adjacency is empty).

        Mechanism: pick an existing hazard and move it to a target cell;
        if no hazards exist but a target is available, spawn one there.
        The event is externally-caused because the agent did not initiate
        the transition into proximity.

        Returns True if a hazard was placed, False if no suitable target
        was found (e.g., agent in a corner with all neighbours occupied).
        """
        ax, ay = self.agent_x, self.agent_y
        # Candidate target cells.
        candidates: List[Tuple[int, int]] = []
        neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in neigh:
            if self.toroidal:
                nx = (ax + dx) % self.size
                ny = (ay + dy) % self.size
            else:
                nx, ny = ax + dx, ay + dy
                if not (0 < nx < self.size - 1 and 0 < ny < self.size - 1):
                    continue
            if self.grid[nx, ny] == self.ENTITY_TYPES["empty"]:
                candidates.append((nx, ny))

        if not candidates and not self.scheduled_external_hazard_adjacent_only:
            # Fall back to any empty cell.
            if self.toroidal:
                all_cells = [
                    (i, j)
                    for i in range(self.size)
                    for j in range(self.size)
                    if self.grid[i, j] == self.ENTITY_TYPES["empty"]
                ]
            else:
                all_cells = [
                    (i, j)
                    for i in range(1, self.size - 1)
                    for j in range(1, self.size - 1)
                    if self.grid[i, j] == self.ENTITY_TYPES["empty"]
                ]
            candidates = all_cells

        if not candidates:
            return False

        self._rng.shuffle(candidates)
        tx, ty = candidates[0]

        # Prefer to MOVE an existing hazard (keeps num_hazards invariant).
        if self.hazards:
            # Pick a hazard that is not already adjacent to the agent.
            movable = [
                h for h in self.hazards
                if not (abs(h[0] - ax) + abs(h[1] - ay) == 1)
            ]
            src = movable[0] if movable else self.hazards[0]
            sx, sy = src[0], src[1]
            self.grid[sx, sy] = self.ENTITY_TYPES["empty"]
            src[0], src[1] = tx, ty
            self.grid[tx, ty] = self.ENTITY_TYPES["hazard"]
        else:
            # No hazards exist; spawn a new one.
            self.grid[tx, ty] = self.ENTITY_TYPES["hazard"]
            self.hazards.append([tx, ty])
        return True

    def _drift_hazards(self) -> None:
        """Drift environment-caused hazards randomly.

        Behavioral diversity extensions (reef_enabled):
          - Reef exclusion: hazards cannot move into reef cells.
          - Food attraction: with probability hazard_food_attraction, a drifting
            hazard sorts candidate directions toward the nearest food cell instead
            of a pure random shuffle, making foraging inherently more dangerous.
        """
        available_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        drifted = False
        for hazard in self.hazards:
            if self._rng.random() < self.env_drift_prob:
                # Food-attraction bias (reef substrate): sort dirs toward nearest food.
                if (self.reef_enabled and self.hazard_food_attraction > 0.0
                        and self.resources
                        and self._rng.random() < self.hazard_food_attraction):
                    hx, hy = hazard[0], hazard[1]
                    nearest = min(
                        self.resources,
                        key=lambda r: abs(r[0] - hx) + abs(r[1] - hy)
                    )
                    fx, fy = nearest[0], nearest[1]
                    dirs_ordered = sorted(
                        available_dirs,
                        key=lambda d: abs(hx + d[0] - fx) + abs(hy + d[1] - fy)
                    )
                else:
                    dirs_ordered = list(available_dirs)
                    self._rng.shuffle(dirs_ordered)

                for dx, dy in dirs_ordered:
                    if self.toroidal:
                        nx = (hazard[0] + dx) % self.size
                        ny = (hazard[1] + dy) % self.size
                        if (self.grid[nx, ny] == self.ENTITY_TYPES["empty"]
                                and (nx, ny) not in self._reef_cells):
                            self.grid[hazard[0], hazard[1]] = self.ENTITY_TYPES["empty"]
                            hazard[0], hazard[1] = nx, ny
                            self.grid[nx, ny] = self.ENTITY_TYPES["hazard"]
                            drifted = True
                            break
                    else:
                        nx, ny = hazard[0] + dx, hazard[1] + dy
                        if (0 < nx < self.size - 1 and 0 < ny < self.size - 1
                                and self.grid[nx, ny] == self.ENTITY_TYPES["empty"]
                                and (nx, ny) not in self._reef_cells):
                            self.grid[hazard[0], hazard[1]] = self.ENTITY_TYPES["empty"]
                            hazard[0], hazard[1] = nx, ny
                            self.grid[nx, ny] = self.ENTITY_TYPES["hazard"]
                            drifted = True
                            break
        # Recompute hazard field after any drift
        if self.use_proxy_fields and drifted:
            self._compute_proximity_fields()

    def _respawn_resource(self) -> None:
        """SD-012: Spawn one new resource at a random empty cell after consumption."""
        available = [
            (i, j)
            for i in range(1, self.size - 1)
            for j in range(1, self.size - 1)
            if self.grid[i, j] == self.ENTITY_TYPES["empty"]
        ]
        if not available:
            return
        self._rng.shuffle(available)
        rx, ry = available[0]
        self.grid[rx, ry] = self.ENTITY_TYPES["resource"]
        self.resources.append([rx, ry])

    def _spawn_transient_benefit(self) -> Optional[Tuple[int, int]]:
        """infant_substrate:GAP-3 -- spawn one transient benefit patch.

        Picks an empty interior cell (zone-weighted toward higher
        resource-factor zones when microhabitat zones are active, uniform
        otherwise; reef cells excluded), tags it as a resource entity so
        the proximity field and perception treat it as a high-salience
        benefit, and registers it in self.resources + self._transient_benefits
        (with an expiry step) + self._transient_benefit_cells. Returns the
        (x, y) cell, or None when no empty interior cell is available.

        Only called from the enabled spawn path; the disabled path makes no
        RNG draws so legacy seed sequences are bit-identical.
        """
        if self.toroidal:
            empties = [
                (i, j)
                for i in range(self.size)
                for j in range(self.size)
                if self.grid[i, j] == self.ENTITY_TYPES["empty"]
            ]
        else:
            empties = [
                (i, j)
                for i in range(1, self.size - 1)
                for j in range(1, self.size - 1)
                if self.grid[i, j] == self.ENTITY_TYPES["empty"]
            ]
        # Reef safe zones never spawn resources (mirrors reset() placement).
        if self.reef_enabled and self._reef_cells:
            empties = [c for c in empties if c not in self._reef_cells]
        if not empties:
            return None
        if self.microhabitat_enabled and self._zone_map is not None:
            # _pop_zone_weighted pops from the supplied list (local here).
            cx, cy = self._pop_zone_weighted(empties, "resource")
        else:
            self._rng.shuffle(empties)
            cx, cy = empties[0]
        self.grid[cx, cy] = self.ENTITY_TYPES["resource"]
        self.resources.append([int(cx), int(cy)])
        self._transient_benefits.append(
            [int(cx), int(cy), int(self.steps + self.transient_benefit_duration)]
        )
        self._transient_benefit_cells.add((int(cx), int(cy)))
        self._transient_benefit_n_spawned += 1
        return (int(cx), int(cy))

    def _respawn_waypoints(self) -> None:
        """Respawn waypoints after sequence completion or timeout."""
        for wp in self.waypoints:
            if self.grid[wp[0], wp[1]] == self.ENTITY_TYPES["waypoint"]:
                self.grid[wp[0], wp[1]] = self.ENTITY_TYPES["empty"]

        available = [
            (i, j)
            for i in range(1, self.size - 1)
            for j in range(1, self.size - 1)
            if self.grid[i, j] == self.ENTITY_TYPES["empty"]
        ]
        self._rng.shuffle(available)
        self.waypoints = []
        for _ in range(min(self.num_waypoints, len(available))):
            wx, wy = available.pop()
            self.grid[wx, wy] = self.ENTITY_TYPES["waypoint"]
            self.waypoints.append([wx, wy])

    # ------------------------------------------------------------------ #
    # Utilities                                                            #
    # ------------------------------------------------------------------ #

    def get_subgoal_state(self) -> dict:
        return {
            "sequence_in_progress": self._sequence_in_progress,
            "sequence_step": self._sequence_step,
            "next_waypoint_idx": self._next_waypoint_idx,
            "sequences_completed": self._sequences_completed,
        }

    def get_contamination_map(self) -> np.ndarray:
        return self.contamination_grid.copy()

    def get_footprint_map(self) -> np.ndarray:
        return self.footprint_grid.copy()

    def get_agent_position(self) -> Tuple[int, int]:
        return (self.agent_x, self.agent_y)

    def get_hazard_field(self) -> np.ndarray:
        """Return hazard proximity field (proxy mode only)."""
        return self.hazard_field.copy()

    def get_resource_field(self) -> np.ndarray:
        """Return resource proximity field (proxy mode only)."""
        return self.resource_field.copy()

    def render(self, mode: str = "text") -> Optional[str]:
        if mode != "text":
            return None
        symbols = {
            self.ENTITY_TYPES["empty"]: ".",
            self.ENTITY_TYPES["wall"]: "#",
            self.ENTITY_TYPES["resource"]: "R",
            self.ENTITY_TYPES["hazard"]: "X",
            self.ENTITY_TYPES["contaminated"]: "c",
            self.ENTITY_TYPES["agent"]: "A",
            self.ENTITY_TYPES["waypoint"]: "W",
        }
        lines = []
        for i in range(self.size):
            row = "".join(symbols.get(self.grid[i, j], "?") for j in range(self.size))
            lines.append(row)
        lines.append(
            f"\nHealth: {self.agent_health:.2f} | Energy: {self.agent_energy:.2f} | "
            f"Steps: {self.steps}"
        )
        lines.append(
            f"Harm: {self.total_harm:.2f} | Benefit: {self.total_benefit:.2f} | "
            f"Max contamination: {self.contamination_grid.max():.2f}"
        )
        if self.use_proxy_fields:
            lines.append(
                f"Harm exposure: {self.harm_exposure:.3f} | "
                f"Benefit exposure: {self.benefit_exposure:.3f} | "
                f"Hazard field @ agent: {self.hazard_field[self.agent_x, self.agent_y]:.3f}"
            )
        return "\n".join(lines)


# Convenience alias for CausalGridWorldV2 mode
def CausalGridWorldV2(**kwargs) -> CausalGridWorld:
    """
    CausalGridWorldV2: CausalGridWorld with proxy-gradient fields enabled.

    Implements ARC-024 proxy-gradient structure:
    - Hazard proximity field generates continuous harm signal before contact
    - Resource proximity field generates continuous benefit signal before collection
    - Interoceptive EMA channels (harm_exposure, benefit_exposure) in body_state
    - Field views (hazard_field_view, resource_field_view) in world_state
    - New transition types: "hazard_approach", "benefit_approach"

    body_obs_dim = 12, world_obs_dim = 250
    """
    kwargs.setdefault("use_proxy_fields", True)
    return CausalGridWorld(**kwargs)
