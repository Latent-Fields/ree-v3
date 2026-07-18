"""Flag-inertness harness -- guard against silently dead / inert / mis-wired flags.

WHY THIS EXISTS
---------------
The 2026-07-09 design+implementation audit found a recurring, high-cost failure
mode: a config-gated mechanism that is *silently inert or silently wrong*. It
does not crash. When an experiment enables the flag to TEST that mechanism, it
measures the wrong thing and returns a plausible-looking null -- which then
weights claim confidence as if it were a real negative result. That is worse
than a crash (a crash gets re-queued; a false null looks like clean evidence).

Confirmed instances from that audit (see
`REE_assembly/design_implementation_audit_2026-07-09.md`):

  F-P1  MECH-074a BLA encoding-gain is pinned to `gmax` on every above-threshold
        tick -> the documented inverted-U collapses to a step function; the
        falling arm (panic-level -> poorer consolidation) is dead code.
  F-C1  Trainable escape-affordance learner truncates its own state vector
        (update-order vs frozen `_state_dim`). Zero live exposure today (no
        experiment enables it) -- guard before first use.
  F-C2  `dacc_foraging_weight` adds a uniform scalar to every candidate -> an
        argmin/softmax(-cost) selector is invariant to it (dead-by-construction
        on the E3 leg; still acts via SalienceCoordinator).
  F-C3  `dacc_saturation_enabled` reads `_outcome_history`, populated only by
        `DACC.record_outcome(...)` -- which has ZERO callers in the live agent
        path. Saturation is always 1.0; habituation/rumination never fires.
  F-C4  `use_iterative_inference=True` with the default `inference_settle_iters=1`
        runs `range(settle_iters-1) == range(0)` -> inert, and emits a NaN
        `final_rel_delta` readout.
  F-P6  `vs_rollout_gate.unknown_stream_passes` -- both branches are byte-identical.

WHAT THIS FILE DOES
-------------------
1. Behavioural probes that assert enabling a flag actually changes an observable.
   Known-broken flags are marked `xfail(strict=True)` tied to a finding id: the
   suite stays GREEN now, and the moment someone fixes the bug the test XPASSes,
   the strict marker fails, and the fixer is forced to delete the marker. That is
   the regression latch -- a fixed bug cannot silently un-fix.
2. A registry-drift guard (`test_flag_registry_is_current`) that enumerates every
   top-level `use_*` / `*_enabled` flag on REEConfig and fails if a NEW flag
   appears that nobody has categorized. Adding a flag then forces a decision:
   write a probe, or record it in KNOWN_UNPROBED with a reason.

HOW TO ADD A PROBE (when you add or touch a `use_*` flag)
---------------------------------------------------------
- Write a test that builds ON vs OFF configs, drives the activating condition on
  a fixed seed, and asserts some observable differs. Add the flag to PROBED.
- If you genuinely cannot probe it yet, add it to KNOWN_UNPROBED with a one-line
  reason. Do not just extend the snapshot silently.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from ree_core.utils import config as config_mod


# --------------------------------------------------------------------------- #
# Behavioural probes                                                          #
# --------------------------------------------------------------------------- #


def test_fp1_bla_encoding_gain_is_an_inverted_u_not_a_step():
    """MECH-074a: encoding_gain must RISE from floor to a peak then FALL.

    Under the bug it is `floor` below threshold and `gmax` at/above threshold --
    a step function. Two observable consequences, both asserted here:

      * rising arm: just above threshold (< peak) the gain must be strictly
        between floor and gmax, not already saturated at gmax.
      * falling arm: well above the peak the gain must be strictly less than at
        the peak (poorer consolidation at panic arousal -- the whole point of
        the inverted-U and of MECH-074a's own falsification signature).

    Each arousal level uses a FRESH BLAAnalog so the post-event window cannot
    carry elevation across probes.
    """
    from ree_core.amygdala.bla import BLAAnalog, BLAConfig

    cfg = BLAConfig()
    floor = float(cfg.encoding_gain_floor)
    gmax = float(cfg.encoding_gain_max)
    peak = float(cfg.arousal_peak)
    thr = float(cfg.arousal_threshold_on)

    def gain_at(arousal: float) -> float:
        bla = BLAAnalog(BLAConfig())
        z = torch.zeros(1, 4)
        z[0, 0] = float(arousal)  # 2-norm == arousal
        return float(bla.tick(z, step_index=0).encoding_gain)

    rising = gain_at((thr + peak) / 2.0)  # between threshold and peak
    at_peak = gain_at(peak)
    falling = gain_at(peak + 4.0)  # far above the peak -> panic arousal

    # rising arm: not yet saturated at the ceiling
    assert floor < rising < gmax, (
        f"rising arm absent: gain just above threshold = {rising}, "
        f"expected strictly between floor {floor} and gmax {gmax}"
    )
    # falling arm: panic arousal is down-weighted relative to the peak
    assert falling < at_peak - 1e-6, (
        f"falling arm absent: gain(panic) = {falling} is not < gain(peak) = "
        f"{at_peak}; the inverted-U has collapsed to a step function"
    )


def test_fc3_dacc_saturation_is_fed_from_the_live_path():
    """With saturation enabled, the live agent must populate the outcome history.

    `_saturation_factor` reads `DACC._outcome_history`, which only
    `record_outcome(...)` fills. We spy on that method across a few default
    steps; under the bug it is never invoked, so the history stays empty and
    saturation is a no-op.
    """
    from ree_core.agent import REEAgent
    from tests.fixtures.seed_utils import set_all_seeds
    from tests.fixtures.tiny_configs import make_tiny_config
    from tests.fixtures.tiny_env import make_tiny_env
    from tests.fixtures.tiny_loop import run_episode

    set_all_seeds(0)
    env = make_tiny_env(seed=0)
    cfg = make_tiny_config(env, use_dacc=True, dacc_saturation_enabled=True)
    agent = REEAgent(cfg)

    assert agent.dacc is not None, "use_dacc did not construct a DACC instance"

    calls = {"n": 0}
    original = agent.dacc.record_outcome

    def _spy(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    agent.dacc.record_outcome = _spy  # type: ignore[assignment]

    run_episode(agent, env, steps=5)

    assert calls["n"] > 0, (
        "DACC.record_outcome was never called during a live episode; "
        "dacc_saturation_enabled is inert (F-C3)"
    )


def test_sd069_phasic_burst_fires_and_changes_the_action_stream():
    """SD-069: `use_phasic_burst` must reach the live E3 select() path.

    The regulator adds an event-locked temperature delta to the softmax that
    E3 selects with, so enabling it on a stream that actually produces surprise
    spikes must (a) fire events and (b) change the committed action stream.
    Asserting BOTH matters: (b) alone could pass on incidental RNG drift, and
    (a) alone would only prove the regulator ticks internally without proving
    it propagates.

    The probe drives `phasic_burst_signal_source="instantaneous_pe"` -- the RAW
    per-tick PE-MSE. The second half of the test pins WHY: the default
    "running_variance" source reads the smoothed EMA, which washes out the
    spikes, so it fires nothing on this same stream. That contrast is the
    documented SD-069 finding (V3-EXQ-779 ran its PHASIC-ON arms on
    "instantaneous_pe" for exactly this reason) and is what makes the source
    selection load-bearing rather than cosmetic.
    """
    from ree_core.agent import REEAgent
    from tests.fixtures.seed_utils import set_all_seeds
    from tests.fixtures.tiny_configs import make_tiny_config
    from tests.fixtures.tiny_env import make_tiny_env
    from tests.fixtures.tiny_loop import run_episode

    def arm(**overrides):
        set_all_seeds(0)
        env = make_tiny_env(seed=0)
        agent = REEAgent(make_tiny_config(env, **overrides))
        actions = run_episode(agent, env, steps=20)
        return agent, actions

    agent_off, actions_off = arm()
    assert agent_off.phasic_burst is None, "flag off must not build the regulator"

    agent_on, actions_on = arm(
        use_phasic_burst=True, phasic_burst_signal_source="instantaneous_pe"
    )
    assert agent_on.phasic_burst is not None, "use_phasic_burst did not wire a regulator"

    n_events = agent_on.phasic_burst.get_state()["n_events"]
    assert n_events > 0, (
        "use_phasic_burst=True fired zero surprise events over 20 live steps; "
        "the regulator ticks but never bursts, so the flag is inert"
    )
    assert actions_on != actions_off, (
        f"use_phasic_burst=True fired {n_events} events but the action stream is "
        f"identical to OFF -- the burst does not reach E3 select() (inert flag)"
    )

    # Contrast: the smoothed default source produces no events on this stream.
    # If this ever starts firing, SD-069's signal-source rationale changed and
    # the probe above should be re-pointed rather than silently left stale.
    agent_smoothed, _ = arm(
        use_phasic_burst=True, phasic_burst_signal_source="running_variance"
    )
    assert agent_smoothed.phasic_burst.get_state()["n_events"] == 0, (
        "the smoothed 'running_variance' source now fires events; SD-069's "
        "sharp-source rationale has changed -- revisit this probe"
    )


# --------------------------------------------------------------------------- #
# Batch probes                                                                #
# --------------------------------------------------------------------------- #

# Flags measured (2026-07-18 sweep, tiny fixture, 15 steps) to change the
# committed action stream with NO sub-knob tuning -- just flipping the flag.
# Two of them (use_actor_critic, use_frontopolar_decommit) are CONDITIONAL:
# they diverge on seeds 0 and 1 but not 2, because their activating condition
# does not occur on every seed. Hence the assertion below is "changes
# behaviour on at least one seed", which is the honest claim for a
# state-gated mechanism and is what keeps this probe non-flaky.
FLAGS_WITH_DEFAULT_BEHAVIOURAL_DELTA = [
    "goal_stream_enabled",
    "use_actor_critic",
    "use_contextual_safety_terrain",
    "use_e2_harm_a",
    "use_e3_score_diversity",
    "use_frontopolar_decommit",
    "use_gated_policy",
    "use_lateral_pfc_analog",
    "use_ofc_analog",
]

# Flags whose REEConfig/agent construction REFUSES a config missing their
# stated dependency. A dropped precondition is its own inertness bug: the
# flag would look enabled while its consumer never runs (the composite-config
# version of F-C3). flag -> substring the error must name.
FLAGS_WITH_LOUD_PRECONDITION = {
    "use_candidate_rule_field": "use_lateral_pfc_analog",
    "use_closure_commit_entry": "use_closure_commit_beta_coupling",
    "use_closure_commit_entry_trajectory": "use_closure_commit_entry",
    "use_closure_operator": "use_lateral_pfc_analog",
    "use_harm_suffering_accumulator": "use_harm_un",
    "use_mech_consume": "use_dacc",
    "use_multi_content_theta_packet": "use_per_stream_vs",
    "use_rho_maintenance_ramp": "use_natural_commit_latch_hold",
    "use_scientist_attribution": "comparator",
}


def _actions_for(flag_overrides: dict, seed: int, steps: int = 15) -> list:
    """Run one fixed-seed tiny episode under the given config overrides."""
    from ree_core.agent import REEAgent
    from tests.fixtures.seed_utils import set_all_seeds
    from tests.fixtures.tiny_configs import make_tiny_config
    from tests.fixtures.tiny_env import make_tiny_env
    from tests.fixtures.tiny_loop import run_episode

    set_all_seeds(seed)
    env = make_tiny_env(seed=seed)
    agent = REEAgent(make_tiny_config(env, **flag_overrides))
    return run_episode(agent, env, steps=steps)


@pytest.mark.parametrize("flag", FLAGS_WITH_DEFAULT_BEHAVIOURAL_DELTA)
def test_flag_changes_the_action_stream(flag):
    """Enabling the flag must change committed behaviour on some seed.

    This is the minimum bar for "not inert": the mechanism reaches action
    selection. It deliberately does NOT assert a direction or magnitude --
    that is the owning experiment's job, not the harness's.
    """
    seeds = (0, 1, 2)
    changed = [
        s for s in seeds if _actions_for({flag: True}, s) != _actions_for({}, s)
    ]
    assert changed, (
        f"{flag}=True produced a byte-identical action stream on every seed "
        f"{seeds}; the flag does not reach action selection at default "
        f"sub-knobs (inert, or its activating condition is never driven here)"
    )


@pytest.mark.parametrize(
    "flag,required", sorted(FLAGS_WITH_LOUD_PRECONDITION.items())
)
def test_flag_precondition_is_loud_not_silent(flag, required):
    """A flag with an unmet dependency must RAISE, not run silently inert.

    Silently tolerating the missing dependency is the composite-config form
    of the F-C3 bug: the flag reads as enabled in the manifest while its
    consumer never runs, so an experiment measures a false null.
    """
    with pytest.raises(ValueError) as excinfo:
        _actions_for({flag: True}, seed=0, steps=1)
    assert required in str(excinfo.value), (
        f"{flag} raised, but the message does not name its missing dependency "
        f"{required!r}: {excinfo.value}"
    )


# --------------------------------------------------------------------------- #
# Flag registry-drift guard                                                   #
# --------------------------------------------------------------------------- #

# Flags with a behavioural probe in this file (asserting ON changes an observable).
PROBED = {
    "use_amygdala_analog",  # F-P1 probe drives BLAAnalog encoding_gain
    "use_bla_analog",       #   (gated by use_amygdala_analog; default True)
    "dacc_saturation_enabled",  # F-C3 wiring spy
    "use_phasic_burst",  # SD-069 fires-and-propagates probe (instantaneous_pe)
} | set(FLAGS_WITH_DEFAULT_BEHAVIOURAL_DELTA) | set(FLAGS_WITH_LOUD_PRECONDITION)

# Audit-confirmed inert / mis-wired flags (finding id -> reason). Documented here
# even when the concrete lever is not a top-level flag (dacc_foraging_weight is a
# float; use_iterative_inference lives under config.latent), so the record is in
# one place. See design_implementation_audit_2026-07-09.md.
KNOWN_INERT = {
    "use_trainable_escape_affordance_learner": "F-C1: truncates its own state "
    "vector; zero live exposure -- guard before first use",
    # F-C3 FIXED 2026-07-09: dacc_saturation_enabled now fed from the live path
    # (agent.py select_action tail calls DACC.record_outcome each waking tick +
    # the DACCConfig saturation knobs are propagated from REEConfig). Moved to
    # PROBED (the test_fc3 wiring spy). See design_implementation_audit_2026-07-09
    # F-C3 / section 6.
    # non-top-level, documented for completeness:
    # dacc_foraging_weight            F-C2 uniform scalar -> argmin-invariant
    # latent.use_iterative_inference  F-C4 range(settle_iters-1) no-op + NaN readout
    # vs_rollout_gate.unknown_stream_passes  F-P6 identical branches
}

# Flags with NO behavioural probe yet, acknowledged so the drift guard passes.
#
# STATUS (2026-07-18 sweep): every flag below was measured ON-vs-OFF at DEFAULT
# sub-knobs on the tiny fixture (15 steps, seed 0) and produced a byte-identical
# action stream. That is NOT evidence of inertness -- most of these are gated
# mechanisms that are correctly no-op until their activating condition is driven
# (a weight left at 0.0, a sleep cycle that never fires in 15 steps, a harm
# event that never occurs, a consumer flag whose producer is off). Promoting one
# of these to PROBED means supplying its activating condition, which is per-flag
# work, not a batch operation.
#
# Priority order for that work is by LANDED CONTRIBUTORY EVIDENCE -- but only the
# evidence a probe could actually invalidate. The bar is ARM-LEVEL: the flag must
# have been the MANIPULATED VARIABLE (an OFF arm vs an ON arm) in a run whose
# manifest carries evidence_direction supports / weakens / does_not_support and
# non-empty claim_ids. Only then would an inert flag mean a landed manifest is a
# false null. A flag held CONSTANT across every arm is substrate, not the thing
# under test: probing it cannot overturn a landed result.
#
# Corrected ranking (2026-07-18 arm-level audit, N contributory runs where the
# flag was manipulated):
#   sws_enabled (7), rem_enabled (7)            -- 265a/385/418/429x2/503a/691
#   use_noise_floor (4)                          -- 544/544a UC-OFF vs ON; 614a/615
#   use_suffering_derivative_comparator (4)      -- 516/517c/517d/519b
#   valence_liking_enabled (3)                   -- 516/517c/517d
#   use_sleep_loop (2), shy_enabled (2),
#     use_conditioned_safety_store (2),
#     replay_diversity_enabled (2)
#   1 run each: use_mech295_liking_bridge (493), use_structured_curiosity +
#     use_curiosity_{novelty,uncertainty,learning_progress} (604c),
#     use_ofc_outcome_oracle (485a), use_object_file_buffer (658),
#     use_rem_precision_recalibration (541a), use_sleep_aggregation_cluster (702),
#     use_closure_env_completion_hook (466e)
#
# DEMOTED -- never the manipulated variable in ANY contributory run; every
# occurrence is a constant baseline setting in the one shared config builder all
# arms go through (so the earlier script-level fan-out counts were misleading):
#   use_dacc                (14 contributory runs: 9 constant-ON, 5 constant-OFF)
#   use_lateral_pfc_analog  (14, all constant-ON)
#   use_pag_freeze_gate     (7, all constant-ON)
#   use_modulatory_selection_authority (7; UPPERCASE module constants in 652/660/707b)
#   use_mech307_conjunction (6, all constant-ON)
#   use_salience_coordinator (6), use_instrumental_avoidance (5)
# Two traps that ranking has to avoid, both hit during the audit:
#   * 490j "severed_bridge_baseline" severs via cfg.goal.z_goal_enabled=False and
#     leaves use_mech295_liking_bridge / use_pag_freeze_gate ON in BOTH arms -- so
#     the MECH-295 `weakens` did not come from toggling either flag.
#   * 776 (MECH-279 supports) is load-bearing for use_pag_freeze_gate without
#     contrasting it: it drives agent.pag_freeze_gate directly and RAISES if the
#     flag fails to build the gate, so inertness there fails loudly, not silently.
#     Manipulated variable is gaba_tone.
# Method note: arm_fingerprint_index.json does NOT record these flags (its
# cell_keys are per-cell metrics/knobs), and manifest arm-config slices name them
# only incidentally -- attribution came from reading each contributory run's arm
# construction in ree-v3/experiments/.
KNOWN_UNPROBED = {
    "action_loop_gate_enabled", "harm_descending_mod_enabled",
    "harm_surprise_pe_enabled", "rem_enabled", "replay_diversity_enabled",
    "shy_enabled", "sws_enabled",
    "use_aic_analog", "use_blocked_agency",
    "use_broadcast_override", "use_cea_analog",
    "use_closure_commit_beta_coupling", "use_closure_env_completion_hook",
    "use_commit_readiness", "use_conditioned_safety_store",
    "use_control_vector_logging",
    "use_cross_module_consolidation", "use_curiosity_learning_progress",
    "use_curiosity_novelty", "use_curiosity_uncertainty", "use_dacc",
    "use_difficulty_gated_proposal_entropy", "use_e2_escape_affordance_linker",
    "use_e2_escape_linker_e3_bias", "use_e2_escape_linker_for_relief_safety",
    "use_e3_diversity_entropy_bonus",
    "use_e3_diversity_stratified_select",
    "use_escape_affordance_bridge", "use_escape_relief_credit",
    "use_escape_safety_credit", "use_external_task_drive",
    "use_gabaergic_decay",
    "use_habenula_decommit",
    "use_instrumental_avoidance",
    "use_lpb_interoceptive_routing", "use_maintenance_release",
    "use_mech090_readiness_conjunction", "use_mech272_routing",
    "use_mech272_routing_consumer", "use_mech273_self_model",
    "use_mech275_aggregator", "use_mech285_sampler", "use_mech286_sleep_onset_gate",
    "use_mech295_liking_bridge", "use_mech307_conjunction",
    "use_mech307_consumer_conjunction_read", "use_mech307_predicted_location_write",
    "use_mech307_schema_multichannel", "use_mech307_signed_pe",
    "use_mech307_split_surprise", "use_mel_consumer",
    "use_mel_entry", "use_modulatory_channel_routing",
    "use_modulatory_selection_authority",
    "use_natural_commit_latch_hold", "use_natural_commit_urgency_release",
    "use_noise_floor", "use_object_file_buffer",
    "use_ofc_devaluation_head", "use_ofc_outcome_oracle", "use_pacc_analog",
    "use_pag_freeze_gate", "use_pcc_analog", "use_rem_precision_recalibration",
    "use_salience_coordinator",
    "use_sd049_per_axis_consumer_cascade",
    "use_shared_harm_trunk", "use_simulation_mode_rule_gate",
    "use_sleep_aggregation_cluster", "use_sleep_loop", "use_structured_curiosity",
    "use_suffering_derivative_comparator", "use_tonic_vigor", "use_tpj_comparator",
    "use_trainable_relief_critic", "use_trainable_safety_predictor",
    "valence_harm_enabled", "valence_liking_enabled",
}


def _current_toplevel_flags() -> set:
    fields = dataclasses.fields(config_mod.REEConfig)
    return {
        f.name
        for f in fields
        if f.name.startswith("use_") or f.name.endswith("_enabled")
    }


def test_flag_registry_is_current():
    """Fail when a top-level `use_*`/`*_enabled` flag is not categorized.

    Adding a flag forces a decision: write a probe (PROBED) or record it in
    KNOWN_UNPROBED with a reason. This is the recurrence guard -- a new dead
    flag cannot slip in un-noticed the way F-C2..F-P6 did.
    """
    covered = PROBED | set(KNOWN_INERT) | KNOWN_UNPROBED
    current = _current_toplevel_flags()

    uncategorized = sorted(current - covered)
    assert not uncategorized, (
        "New/uncategorized top-level config flag(s): "
        f"{uncategorized}. Add a behavioural probe to test_flag_inertness.py "
        "(PROBED) or record the flag in KNOWN_UNPROBED with a reason."
    )

    # Keep the snapshot honest: a flag that was renamed/removed should be pruned
    # from the registry rather than lingering as a phantom entry.
    stale = sorted((PROBED | KNOWN_UNPROBED) - current - set(KNOWN_INERT))
    assert not stale, (
        f"Registry lists flag(s) no longer on REEConfig: {stale}. "
        "Remove them from PROBED / KNOWN_UNPROBED."
    )
