"""Canonical OFF/baseline arm for the ARC-005 control-plane-routing lineage.

Lineage: v3_exq_802_arc005_control_plane_routing_double_dissociation (first of lineage).

WHY THIS MODULE EXISTS
----------------------
Per `/queue-experiment` "Saving a baseline for reuse", the FIRST experiment of a
lineage mints its own reusable baseline in-line rather than via a separate mint job.
Two things make that work, and both live here:

  1. the OFF arm is CONSTRUCTED from this module (so a later sibling with a different
     driver builds a bit-identical cell by construction, not by coincidence), and
  2. the caller emits the OFF cell's fingerprint with
     `include_driver_script_in_hash=False` (done in the driver), so a future
     consumer's distinct driver can still match.

`_lib/**` is inside the substrate glob, so any edit here correctly BUSTS a stale
reuse rather than silently serving one.

WHAT THE BASELINE IS
--------------------
The OFF arm of the ARC-005 double dissociation: control-plane channel level L0
(the substrate's own operating settings for all four implemented channels) on
content-set A. It is a DRIVEN MEASUREMENT arm -- nothing is trained, no gradient
flows; the driver senses, ticks the clock, generates candidates, selects, and steps
the env, exactly as the treatment arms do.

THE FOUR IMPLEMENTED CONTROL-PLANE CHANNELS (the scope of ARC-005's test)
------------------------------------------------------------------------
  1. 5-HT rigidity        ree_core/neuromodulation/serotonin.py
                          tonic_5ht_baseline -> z_goal_seeding_gain + wanting floor
  2. phasic-burst gain    ree_core/regulators/phasic_surprise_burst.py
                          phasic_burst_temp_delta -> E3 softmax temperature transient
  3. mode prior           ree_core/cingulate/salience_coordinator.py
                          external_task_bias -> the external_task LOGIT only
  4. pcc_stability mu     ree_core/cingulate/pcc_analog.py
                          pcc_stability_baseline -> mode-softmax temperature (mu leg)
                          AND the MECH-259 switch-threshold multiplier

All four are held at L0 here and laddered monotonically by the driver at L1/L2.
"""

from typing import Any, Dict

# ------------------------------------------------------------------ #
# Schedule (part of the OFF-path config slice -- it changes the cell)  #
# ------------------------------------------------------------------ #
WARMUP_TICKS = 200
MEASURE_TICKS = 1800
TOTAL_TICKS = WARMUP_TICKS + MEASURE_TICKS

# Fixed arena. IDENTICAL across content sets -- only the CONTENT varies.
ARENA: Dict[str, Any] = {
    "size": 10,
    "toroidal": False,
    "hazard_harm": 0.5,
    "contaminated_harm": 0.4,
    "resource_benefit": 0.3,
    "energy_decay": 0.01,
    "env_drift_interval": 5,
    "env_drift_prob": 0.3,
    "use_proxy_fields": True,
}

# Content sets. Same arena, different WHAT-IS-IN-IT: hazard/resource density,
# contamination spread, and layout (via a seed offset so placements differ).
CONTENT_SETS: Dict[str, Dict[str, Any]] = {
    "A": {"num_hazards": 3, "num_resources": 5, "contamination_spread": 0.5,
          "seed_offset": 0},
    "B": {"num_hazards": 6, "num_resources": 2, "contamination_spread": 0.9,
          "seed_offset": 10007},
}

# Channel ladder. level in [0, 1]; L0 = substrate operating settings.
CHANNEL_LEVELS = [0.0, 0.5, 1.0]


def content_env_kwargs(content: str, seed: int) -> Dict[str, Any]:
    """Env constructor kwargs for a content set at a seed. Arena is invariant."""
    spec = CONTENT_SETS[content]
    kw = dict(ARENA)
    kw.update(
        num_hazards=spec["num_hazards"],
        num_resources=spec["num_resources"],
        contamination_spread=spec["contamination_spread"],
        seed=int(seed) + int(spec["seed_offset"]),
    )
    return kw


def channel_settings(level: float) -> Dict[str, Any]:
    """The four control-plane channel settings at a ladder level.

    Every channel is LIVE at every level (no on/off structural difference between
    arms) -- only its setting moves. Monotone in `level` by construction.
    """
    lam = float(level)
    return {
        # (1) 5-HT rigidity, set at the CHANNEL'S OUTPUT rather than its input.
        #     The tonic level is NOT a settable lever: serotonin_step subtracts
        #     harm_suppress_rate * ||z_harm_a|| every waking step, which on this
        #     env crushes tonic_5ht from any baseline to ~0 within tens of steps
        #     (measured 2026-07-21: baseline 0.50 -> realised mean 0.0019 over 40
        #     steps). Pinning gain_min == gain_max (and floor_min == floor_max)
        #     makes current_seeding_gain() / current_wanting_floor() return the
        #     ladder value EXACTLY, which is what "rigidity" denotes -- the
        #     z_goal_seeding_gain / valence_wanting_floor the channel imposes.
        #     L0 = 0.90 / 0.040 reproduce the substrate's own values at the
        #     default tonic baseline 0.5 (gain = 0.3 + 0.5 * (1.5 - 0.3)).
        "serotonin_seeding_gain": 0.90 + 0.60 * lam,     # -> gain_max 1.5 at L2
        "serotonin_wanting_floor": 0.040 + 0.040 * lam,  # -> floor_max 0.08 at L2
        # (2) phasic-burst gain: -0.10 -> -1.00 temperature delta at full burst
        #     (NEGATIVE = sharpening; magnitude rises with level).
        "phasic_burst_temp_delta": -0.10 - 0.90 * lam,
        # (3) mode prior: external_task LOGIT bias 1.0 -> 3.0. Mode-SPECIFIC,
        #     never a broadcast constant across modes (see the DV-symmetry
        #     declaration in the driver docstring).
        "salience_external_task_bias": 1.0 + 2.0 * lam,
        # (4) pcc_stability mu: 0.50 -> 0.95.
        "pcc_stability_baseline": 0.50 + 0.45 * lam,
    }


def agent_kwargs(level: float) -> Dict[str, Any]:
    """REEConfig.from_dims kwargs for a channel level (obs dims added by caller)."""
    ch = channel_settings(level)
    return dict(
        alpha_world=0.9,
        # --- channel 1: 5-HT (live at every level) ---
        tonic_5ht_enabled=True,
        # --- channel 2: phasic surprise burst (live at every level) ---
        use_phasic_burst=True,
        phasic_burst_temp_delta=ch["phasic_burst_temp_delta"],
        # The default "running_variance" source is a monotonically-decaying EMA
        # that washes out per-tick PE spikes, so the lever fires ZERO natural
        # events (config.py's own comment says so; confirmed here 2026-07-21,
        # burst_level_mean 0.0 in every arm). "instantaneous_pe" + "carry" are
        # what make channel 2 actually fire, i.e. what makes it a CHANNEL rather
        # than an inert knob.
        phasic_burst_signal_source="instantaneous_pe",
        phasic_burst_baseline_continuity="carry",
        # --- channels 3 + 4: salience coordinator + PCC-analog ---
        use_salience_coordinator=True,
        # SD-032b dACC and SD-032c AIC are the coordinator's ONLY live sources of
        # affinity logits beyond external_task_bias, and the ONLY sources of the
        # salience_aggregate compared against the switch threshold. With both off
        # the aggregate is identically 0, argmax is always external_task, and the
        # discrete mode can NEVER switch -- occupancy is then single-mode by
        # CONFIGURATION, not by any substrate property (measured 2026-07-21:
        # external_task occupancy 1.0 in all six arms). They are held at IDENTICAL
        # settings in every arm, so they are part of the shared substrate config,
        # not part of either factor.
        use_dacc=True,
        dacc_weight=0.5,
        dacc_foraging_weight=0.5,
        use_aic_analog=True,
        salience_external_task_bias=ch["salience_external_task_bias"],
        use_pcc_analog=True,
        pcc_stability_baseline=ch["pcc_stability_baseline"],
        # mu leg of MECH-048 ON so pcc_stability reaches the mode prior, not just
        # the switch threshold. kappa left at its 0.0 default (single-lever).
        salience_use_stability_temperature=True,
        salience_temperature_mu_alpha=1.0,
        # --- substrate operating config, identical in every arm ---
        use_harm_stream=True,
        use_affective_harm_stream=True,
        use_support_preserving_cem=True,
        support_preserving_min_first_action_classes=2,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=4.0,
    )


def off_path_config_slice() -> Dict[str, Any]:
    """The declared config slice for the OFF/baseline cell (L0, content A).

    Declares ONLY what the OFF computation reads: env kwargs, schedule, and the
    agent config at level 0. NEVER the treatment levels or the acceptance
    thresholds -- an under- or over-declared slice is what produces false
    fingerprint hits/misses (arm_reuse_fingerprint_plan.md sec 7b).

    The env `seed` is excluded here because the seed is a separate fingerprint
    field; the content spec that DERIVES it is declared instead.
    """
    env_kw = dict(ARENA)
    env_kw.update(
        num_hazards=CONTENT_SETS["A"]["num_hazards"],
        num_resources=CONTENT_SETS["A"]["num_resources"],
        contamination_spread=CONTENT_SETS["A"]["contamination_spread"],
        content_seed_offset=CONTENT_SETS["A"]["seed_offset"],
    )
    return {
        "lineage": "exq802_arc005_control_plane",
        "arm_id": arm_id(CHANNEL_LEVELS[0], "A"),
        "channel_level": CHANNEL_LEVELS[0],
        "content": "A",
        "env_kwargs": env_kw,
        "agent_kwargs": agent_kwargs(CHANNEL_LEVELS[0]),
        "schedule": {
            "warmup_ticks": WARMUP_TICKS,
            "measure_ticks": MEASURE_TICKS,
            "total_ticks": TOTAL_TICKS,
        },
    }


def cell_config_slice(level: float, content: str) -> Dict[str, Any]:
    """Config slice for ANY cell of the grid. The OFF cell's value is identical
    to `off_path_config_slice()` by construction (asserted in the driver)."""
    env_kw = dict(ARENA)
    spec = CONTENT_SETS[content]
    env_kw.update(
        num_hazards=spec["num_hazards"],
        num_resources=spec["num_resources"],
        contamination_spread=spec["contamination_spread"],
        content_seed_offset=spec["seed_offset"],
    )
    return {
        "lineage": "exq802_arc005_control_plane",
        "arm_id": arm_id(level, content),
        "channel_level": float(level),
        "content": content,
        "env_kwargs": env_kw,
        "agent_kwargs": agent_kwargs(level),
        "schedule": {
            "warmup_ticks": WARMUP_TICKS,
            "measure_ticks": MEASURE_TICKS,
            "total_ticks": TOTAL_TICKS,
        },
    }


def arm_id(level: float, content: str) -> str:
    idx = CHANNEL_LEVELS.index(float(level))
    return f"L{idx}_{content}"
