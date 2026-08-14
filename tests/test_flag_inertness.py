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
        FIXED 2026-07-27: LatentStack.__init__ REFUSES settle_iters < 2 when the
        flag is on (ValueError naming both knobs), and final_rel_delta is now the
        last measured delta rather than a NaN placeholder. Probed by
        test_fc4_iterative_inference_settles_and_refuses_the_inert_config.
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


def test_fc4_iterative_inference_settles_and_refuses_the_inert_config():
    """F-C4: the settling loop must actually run, and the degenerate config raise.

    Under the bug, `use_iterative_inference=True` with the default
    `inference_settle_iters=1` ran `range(settle_iters - 1) == range(0)`: zero
    settling rounds, latents bit-identical to OFF, and a NaN `final_rel_delta`
    readout -- while the manifest recorded the flag as ENABLED. An experiment
    enabling it to test ARC-004 measured nothing and returned a plausible null.

    Two halves, matching the two halves of the fix:
      * settle_iters < 2 with the flag ON refuses to build (loud, not inert),
        following the FLAGS_WITH_LOUD_PRECONDITION precedent.
      * settle_iters >= 2 produces a non-empty, non-NaN convergence readout.
    """
    from ree_core.agent import REEAgent
    from tests.fixtures.seed_utils import set_all_seeds
    from tests.fixtures.tiny_configs import make_tiny_config
    from tests.fixtures.tiny_env import make_tiny_env

    # -- half 1: the inert combination is refused, and names both knobs --------
    set_all_seeds(0)
    env = make_tiny_env(seed=0)
    for iters in (0, 1):
        with pytest.raises(ValueError) as excinfo:
            REEAgent(
                make_tiny_config(
                    env,
                    use_iterative_inference=True,
                    inference_settle_iters=iters,
                )
            )
        msg = str(excinfo.value)
        assert "inference_settle_iters" in msg and "use_iterative_inference" in msg, (
            f"settle_iters={iters} raised, but the message does not name the "
            f"knobs an experimenter must change: {msg}"
        )

    # -- half 2: an admissible config actually settles -------------------------
    set_all_seeds(0)
    env = make_tiny_env(seed=0)
    agent = REEAgent(
        make_tiny_config(
            env,
            use_iterative_inference=True,
            inference_settle_iters=6,
            inference_convergence_rel_tol=1e-9,  # force the full budget
        )
    )
    agent.reset()
    _flat, od = env.reset()
    b = od["body_state"]
    w = od["world_state"]
    if b.dim() == 1:
        b = b.unsqueeze(0)
    if w.dim() == 1:
        w = w.unsqueeze(0)
    with torch.no_grad():
        latent = agent.sense(obs_body=b, obs_world=w)

    ic = latent.inference_convergence
    assert ic is not None, "flag ON produced no convergence readout"
    assert ic["per_step_rel_delta"], (
        "the settling loop ran zero rounds; use_iterative_inference is inert "
        "(F-C4)"
    )
    assert ic["n_iters"] > 1, f"only {ic['n_iters']} inference round(s) ran"
    frd = ic["final_rel_delta"]
    assert frd == frd, "final_rel_delta is NaN (F-C4 readout half)"
    assert frd == ic["per_step_rel_delta"][-1]


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

    ON PROPERTY (b) -- WHY THIS IS NOT AN ACTION-STREAM COMPARISON.
    This probe originally asserted `actions_on != actions_off`. That pinned
    bit-level determinism of the committed action, which the substrate does
    not hold across machine classes: the action is drawn with
    torch.multinomial, and torch.multinomial returns DIFFERENT categories on
    linux-x86_64 vs darwin-arm64 from a bit-identical probability tensor at
    the same seed (verified 2026-07-19; randperm/randint/bernoulli/rand all
    match, multinomial does not). So the discrete stream differed on the Mac
    and happened to coincide on the fleet, failing there for a reason that has
    nothing to do with SD-069.
    We therefore assert propagation UPSTREAM of the discrete quantizer, which
    is both machine-independent and a strictly stronger statement of the same
    property: the regulator must produce a live non-zero temperature delta,
    and that delta must actually move the selection temperature. `select_action`
    feeds the tonic temperature through `apply_to_temperature`, so a non-zero
    delta there IS the burst reaching E3 select(). Whether the moved
    temperature then happens to flip an argmax is a property of the score
    margins, not of the flag being wired.
    """
    from ree_core.agent import REEAgent
    from tests.fixtures.seed_utils import set_all_seeds
    from tests.fixtures.tiny_configs import make_tiny_config
    from tests.fixtures.tiny_env import make_tiny_env
    from tests.fixtures.tiny_loop import step_once

    def arm(steps=20, **overrides):
        set_all_seeds(0)
        env = make_tiny_env(seed=0)
        agent = REEAgent(make_tiny_config(env, **overrides))
        agent.reset()
        _flat, obs = env.reset()
        deltas = []
        for _ in range(steps):
            _a, _idx, _ticks, obs = step_once(agent, env, obs)
            if agent.phasic_burst is not None:
                deltas.append(float(agent.phasic_burst.get_state()["temperature_delta"]))
        return agent, deltas

    agent_off, _ = arm()
    assert agent_off.phasic_burst is None, "flag off must not build the regulator"

    agent_on, deltas_on = arm(
        use_phasic_burst=True, phasic_burst_signal_source="instantaneous_pe"
    )
    assert agent_on.phasic_burst is not None, "use_phasic_burst did not wire a regulator"

    n_events = agent_on.phasic_burst.get_state()["n_events"]
    assert n_events > 0, (
        "use_phasic_burst=True fired zero surprise events over 20 live steps; "
        "the regulator ticks but never bursts, so the flag is inert"
    )
    # (b) the burst produces a live temperature delta ...
    max_delta = max((abs(d) for d in deltas_on), default=0.0)
    assert max_delta > 0.0, (
        f"use_phasic_burst=True fired {n_events} events but the regulator's "
        f"temperature_delta never left zero over {len(deltas_on)} steps -- the "
        f"burst does not reach the selection temperature (inert flag)"
    )
    # ... and that delta actually moves the temperature select() is handed.
    _probe_T = 1.0
    _moved_T = float(agent_on.phasic_burst.apply_to_temperature(_probe_T))
    assert _moved_T != _probe_T, (
        f"regulator reached temperature_delta={max_delta} but "
        f"apply_to_temperature({_probe_T}) returned {_moved_T} unchanged -- the "
        f"delta is computed but never applied in E3 select() (inert flag)"
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


def _sleep_cycle_probe(seed: int = 0, steps: int = 12, **overrides) -> dict:
    """Run waking steps, then one SD-017 sleep cycle; report both sides of it.

    The waking steps are the ACTIVATING CONDITION and are the whole reason the
    2026-07-18 batch sweep could not probe these flags: `run_sws_schema_pass`
    early-returns unless `_world_experience_buffer` holds >= 2 entries, and
    `run_rem_attribution_pass` early-returns unless `theta_buffer.recent` is
    populated. Both buffers are filled only by `_e1_tick` on the waking path, so
    flipping the flag without stepping first measures nothing. (Every sleep test
    in tests/contracts/ runs zero waking steps, so they all exercise the
    zeroed early-return path and assert key presence only -- these two probes
    are the first to drive a pass that actually fires.)

    Returns the cycle metrics plus the two DOWNSTREAM observables the passes
    write into, captured across the cycle only:
      * context_memory_changed -- E1 ContextMemory slots (the SWS write target)
      * n_hippocampal_replay   -- calls into HippocampalModule.replay (REM's)
    """
    from ree_core.agent import REEAgent
    from tests.fixtures.seed_utils import set_all_seeds
    from tests.fixtures.tiny_configs import make_tiny_config
    from tests.fixtures.tiny_env import make_tiny_env
    from tests.fixtures.tiny_loop import run_episode

    set_all_seeds(seed)
    env = make_tiny_env(seed=seed)
    agent = REEAgent(make_tiny_config(env, **overrides))
    run_episode(agent, env, steps=steps)  # supply the activating condition

    mem_before = agent.e1.context_memory.memory.detach().clone()
    calls = {"n": 0}
    original_replay = agent.hippocampal.replay

    def _spy(*args, **kwargs):
        calls["n"] += 1
        return original_replay(*args, **kwargs)

    agent.hippocampal.replay = _spy  # type: ignore[assignment]
    metrics = agent.run_sleep_cycle()
    mem_after = agent.e1.context_memory.memory.detach().clone()

    return {
        "metrics": metrics,
        "context_memory_changed": not torch.equal(mem_before, mem_after),
        "context_memory_after": mem_after,
        "n_hippocampal_replay": calls["n"],
        "world_buffer": len(agent._world_experience_buffer),
        "theta_recent_present": agent.theta_buffer.recent is not None,
    }


def test_sd017_sws_enabled_fires_and_writes_into_context_memory():
    """SD-017: `sws_enabled` must run the schema pass AND mutate ContextMemory.

    Seven landed contributory manifests toggled this flag as their manipulated
    variable (265a / 385 / 418 / 429 x2 / 503a / 691), so a silently inert
    `sws_enabled` would make all seven false nulls.

    Asserting BOTH levels matters, and they are different failure modes:
      * fires    -- `sws_n_writes > 0` proves the pass got past its guards
                    (flag check, then the >= 2 buffer-size check).
      * lands    -- the E1 ContextMemory tensor actually changed. Without this a
                    pass could count writes that go nowhere; ContextMemory IS
                    the SWS write target (hippocampus-to-cortex schema
                    installation), so this is the propagation step.

    Deliberately no assertion on the DIRECTION or MAGNITUDE of slot diversity --
    whether consolidation helps is the owning experiment's question. The bar
    here is only "not inert".
    """
    off = _sleep_cycle_probe()
    on = _sleep_cycle_probe(sws_enabled=True)

    # The activating condition really was supplied (otherwise ON would return
    # zeros for a reason that has nothing to do with the flag).
    assert on["world_buffer"] >= 2, (
        f"waking steps did not fill the world-experience buffer "
        f"(size {on['world_buffer']}); the probe cannot distinguish an inert "
        f"flag from an unmet precondition"
    )

    assert off["metrics"] == {}, (
        f"sws_enabled=False still produced sleep metrics {off['metrics']}"
    )
    assert not off["context_memory_changed"], (
        "a sleep cycle with sws_enabled=False mutated ContextMemory; the SWS "
        "write path is not actually gated by the flag"
    )

    n_writes = on["metrics"].get("sws_n_writes", 0.0)
    assert n_writes > 0, (
        f"sws_enabled=True performed zero schema writes over {on['world_buffer']} "
        f"buffered waking observations; the pass ticks but never writes (inert). "
        f"metrics={on['metrics']}"
    )
    assert on["context_memory_changed"], (
        f"sws_enabled=True reported {n_writes} schema writes but E1 ContextMemory "
        f"is byte-identical -- the writes do not reach their target (inert flag)"
    )

    # Dissociation from rem_enabled: the SWS pass must not be driving the REM
    # replay path. If this ever starts firing, the two flags have been coupled
    # and BOTH probes need re-pointing rather than being left stale.
    assert on["n_hippocampal_replay"] == 0, (
        "the SWS pass now drives HippocampalModule.replay; sws_enabled and "
        "rem_enabled are no longer dissociable -- revisit both probes"
    )


def test_sd017_rem_enabled_fires_and_drives_hippocampal_replay():
    """SD-017: `rem_enabled` must run the attribution pass AND reach the hippocampus.

    Same seven contributory runs as `sws_enabled` manipulate this flag, and 691's
    ARM_REPLAY_ABLATED contrasts it against SWS specifically -- so it is probed
    separately, not bundled into one "sleep on" test.

    Two levels again:
      * fires -- `rem_n_rollouts > 0` proves the pass cleared its guards (flag
                 check, then the `theta_buffer.recent is not None` check).
      * lands -- HippocampalModule.replay was actually invoked. The REM pass is
                 read-only by design (MECH-094: it scores residue terrain with
                 hypothesis_tag semantics and writes no residue), so a
                 state-delta assertion is the wrong instrument; the honest
                 propagation evidence is that the rollouts genuinely execute in
                 the hippocampal module rather than being counted locally.

    NOTE on `rem_n_reverse`: it stays 0 on this fixture. That is correct, not a
    failure -- `_exploration_buffer` is empty after a plain waking episode, so
    the pass takes its documented else-branch (extra forward rollouts) instead
    of `diverse_replay(mode="reverse")`. Probing the reverse arm needs the
    exploration buffer seeded (MECH-165 / replay_diversity_enabled), which is
    that flag's probe to write, not this one's.
    """
    off = _sleep_cycle_probe()
    on = _sleep_cycle_probe(rem_enabled=True)

    assert on["theta_recent_present"], (
        "waking steps did not populate theta_buffer.recent; the probe cannot "
        "distinguish an inert flag from an unmet precondition"
    )

    assert off["metrics"] == {}, (
        f"rem_enabled=False still produced sleep metrics {off['metrics']}"
    )
    assert off["n_hippocampal_replay"] == 0, (
        "a sleep cycle with rem_enabled=False drove hippocampal replay; the REM "
        "pass is not actually gated by the flag"
    )

    n_rollouts = on["metrics"].get("rem_n_rollouts", 0.0)
    assert n_rollouts > 0, (
        f"rem_enabled=True produced zero attribution rollouts despite a "
        f"populated theta buffer; the pass ticks but never replays (inert). "
        f"metrics={on['metrics']}"
    )
    assert on["n_hippocampal_replay"] > 0, (
        f"rem_enabled=True reported {n_rollouts} rollouts but never called "
        f"HippocampalModule.replay -- the rollouts are counted without being "
        f"executed against the hippocampus (inert flag)"
    )

    # Dissociation from sws_enabled: REM is slot-FILLING, not slot-formation,
    # so it must not be installing schema content. Same re-point rule as above.
    assert not on["context_memory_changed"], (
        "the REM pass now writes ContextMemory; sws_enabled and rem_enabled are "
        "no longer dissociable -- revisit both probes"
    )


def test_mech122_spindle_content_selection_fires_and_differentiates_writes():
    """MECH-122 content-packaging (V3 proxy, IGW-20260801-197):
    `use_mech122_spindle_content_selection` must (a) be OFF-inert -- report
    zero selection weight and zero `sws_spindle_selection_applied` when the
    master flag is off -- and (b) when ON, actually blend each schema-write
    prototype toward the ThetaBuffer consolidation reference and thereby
    change WHAT gets written to E1.ContextMemory, not just tick a counter.

    Bar (b) is deliberately stronger than "fires". V3-EXQ-246 already probed
    a naive MECH-122 Phase-3 proxy -- a single extra post-hoc write of
    consolidation_summary() appended after SWS+REM -- and measured ZERO
    effect on its downstream metric in both runs, all 3 seeds (see
    claims.yaml MECH-122 evidence_quality_note). A flag that merely "ticks"
    without changing the ContextMemory write target would repeat that null
    silently. So this probe compares the ACTUAL post-pass ContextMemory
    tensor between the ON and OFF arms (same seed, same waking episode, same
    buffered z_world content going into the pass) and requires it to differ.
    """
    off = _sleep_cycle_probe(sws_enabled=True)
    on = _sleep_cycle_probe(sws_enabled=True, use_mech122_spindle_content_selection=True)

    # activating conditions really supplied (world buffer + theta buffer)
    assert on["world_buffer"] >= 2, (
        f"waking steps did not fill the world-experience buffer "
        f"(size {on['world_buffer']}); the probe cannot distinguish an inert "
        f"flag from an unmet precondition"
    )
    assert on["theta_recent_present"], (
        "waking steps did not populate theta_buffer.recent; consolidation_summary() "
        "has nothing to reference"
    )

    # OFF-inert: no selection reported, and behaves exactly like plain SWS.
    assert off["metrics"].get("sws_spindle_selection_applied", 0.0) == 0.0, (
        f"sws_spindle_selection_applied nonzero with the master flag OFF: "
        f"metrics={off['metrics']}"
    )
    assert off["metrics"].get("sws_spindle_selection_mean_weight", 0.0) == 0.0, (
        f"sws_spindle_selection_mean_weight nonzero with the master flag OFF: "
        f"metrics={off['metrics']}"
    )
    assert off["context_memory_changed"], (
        "plain sws_enabled=True (flag OFF) did not write ContextMemory -- "
        "unrelated regression, not this flag"
    )

    # ON: selection actually ran, with an in-range, non-trivial weight.
    applied = on["metrics"].get("sws_spindle_selection_applied", 0.0)
    weight = on["metrics"].get("sws_spindle_selection_mean_weight", 0.0)
    assert applied == 1.0, (
        f"use_mech122_spindle_content_selection=True did not apply selection "
        f"this pass despite a populated theta buffer; metrics={on['metrics']}"
    )
    assert 0.0 <= weight <= 1.0, (
        f"sws_spindle_selection_mean_weight={weight} out of [0,1] range"
    )

    # ON actually changes the write target relative to OFF -- the bar V3-EXQ-246's
    # naive proxy failed to clear.
    assert not torch.equal(off["context_memory_after"], on["context_memory_after"]), (
        "ON arm produced a byte-identical ContextMemory to OFF despite "
        f"reporting mean selection weight {weight} -- the content-selection "
        "blend does not change what gets written (inert wiring), the same "
        "null V3-EXQ-246's naive post-hoc-write proxy measured"
    )


def test_mech122_spindle_content_selection_gain_zero_collapses_to_off():
    """Sanity check on the gain lever: gain=0.0 forces selection_weight=0.0 for
    every prototype (novelty*0 clamped to 0), which blends every write FULLY
    toward the consolidation reference. This is a distinct code path from the
    master flag being off (it still reports `sws_spindle_selection_applied=1.0`
    and still calls set_consolidation_mode), so it is worth pinning separately
    from the OFF-inert half of the probe above.
    """
    on_zero_gain = _sleep_cycle_probe(
        sws_enabled=True,
        use_mech122_spindle_content_selection=True,
        mech122_spindle_selection_gain=0.0,
    )
    assert on_zero_gain["metrics"].get("sws_spindle_selection_applied", 0.0) == 1.0
    assert on_zero_gain["metrics"].get("sws_spindle_selection_mean_weight", -1.0) == 0.0


def test_mech122_mel_relative_novelty_gate_semantics():
    """MELConsumer.relative_novelty() must be a clamped, calibrated novelty
    scalar -- the re-sourced signal the spindle selection gate reads.

        relative_novelty = clamp(mel/ref - 1, 0, cap)

    Pins the four boundary cases that make it a *relative* novelty (not the raw
    MEL magnitude): 0 with no accumulated PE, 0 at the calibrated baseline
    (mel == ref, the NONE-arm case), a graded value between, and the top clamp.
    This is the unit the V3-EXQ-861a autopsy asked for -- a signal that tracks
    the world_rule_shift/MEL axis rather than the self-referential recency
    buffer whose novelty collapsed to ~0 regardless of arm.
    """
    from ree_core.sleep.mel_consumer import MELConsumer, MELConsumerConfig

    mc = MELConsumer(MELConsumerConfig(mel_reference=0.01, mel_reference_mode="fixed"))
    assert mc.relative_novelty() == 0.0, "no accumulated PE must read zero novelty"

    mc.note_step_pe(0.01)  # mel == calibrated base -> NONE-arm case
    assert mc.relative_novelty() == 0.0, "baseline MEL (mel==ref) must read zero novelty"

    mc.reset()
    mc.config.mel_reference = 0.01
    mc.note_step_pe(0.013)  # mel = 1.3 * ref
    assert mc.relative_novelty() == pytest.approx(0.3), "graded novelty must pass through"

    mc.reset()
    mc.config.mel_reference = 0.01
    mc.note_step_pe(0.05)  # mel = 5 * ref -> saturates the [0,1] blend range
    assert mc.relative_novelty() == 1.0, "novelty must clamp to cap for the convex blend"


def _spindle_pass_under_mel(rel_multiple: float, mode: str = "mel_pe",
                            seed: int = 0, steps: int = 12) -> dict:
    """One SWS schema pass with the MEL state pinned to `rel_multiple` x the
    calibrated stable-base reference.

    Both arms build a FRESH agent at the SAME seed, so the waking-filled
    world/theta buffers (and hence the per-prototype recency novelty) are
    byte-identical between calls -- the ONLY thing that differs is the injected
    MEL, exactly as the driver's arms differ only in world_rule_shift-driven
    MEL. So any change in the returned selection weight is attributable to the
    re-sourced gate, not to divergent buffered content.
    """
    from ree_core.agent import REEAgent
    from tests.fixtures.seed_utils import set_all_seeds
    from tests.fixtures.tiny_configs import make_tiny_config
    from tests.fixtures.tiny_env import make_tiny_env
    from tests.fixtures.tiny_loop import run_episode

    set_all_seeds(seed)
    env = make_tiny_env(seed=seed)
    agent = REEAgent(make_tiny_config(
        env,
        sws_enabled=True,
        use_mech122_spindle_content_selection=True,
        mech122_novelty_reference_mode=mode,
        use_mel_consumer=True,
        mel_reference=0.0,
        mel_reference_mode="fixed",
    ))
    run_episode(agent, env, steps=steps)  # fill world+theta buffers (+ some PE)
    assert agent.mel_consumer is not None, "use_mel_consumer did not build a consumer"

    # Pin the MEL state deterministically: fix the stable-base reference, then
    # inject one waking PE step at rel_multiple x that reference.
    ref = 0.01
    agent.config.mel_reference = ref
    agent.mel_consumer.config.mel_reference = ref
    agent.mel_consumer.reset()
    agent.mel_consumer.note_step_pe(rel_multiple * ref)

    return agent.run_sws_schema_pass()


def test_mech122_novelty_reference_mel_pe_tracks_mel_not_flat():
    """V3-EXQ-861a repair: with mech122_novelty_reference_mode='mel_pe' the mean
    selection_weight must MEASURABLY DIFFER across two contrasting novelty
    conditions -- the exact property 861a lacked (its weight was ~0.004-0.01 and
    flat across every arm/seed, because the novelty reference was a 10-tick
    self-referential recency buffer decoupled from the MEL axis).

    Asserting a DIFFERENCE across novelty conditions is the point -- 'fires but
    stays flat' reproduces 861a exactly and is the failure this fix exists to
    remove. The second half pins that the difference comes from the RE-SOURCING:
    the legacy 'recency' mode, on the identical buffers, is invariant to MEL.
    """
    low = _spindle_pass_under_mel(1.0)   # mel == calibrated base -> gate 0
    high = _spindle_pass_under_mel(3.0)  # mel = 3x base -> gate saturates toward 1

    # mechanism fired in both conditions (not silently inert)
    assert low["sws_spindle_selection_applied"] == 1.0
    assert high["sws_spindle_selection_applied"] == 1.0

    w_low = low["sws_spindle_selection_mean_weight"]
    w_high = high["sws_spindle_selection_mean_weight"]
    # low condition is non-degenerate (keeps the per-prototype recency floor,
    # so low-MEL writes are not collapsed onto a single reference vector)
    assert 0.0 <= w_low <= 1.0
    # the whole point: weight tracks the MEL/novelty axis, does not stay flat
    assert w_high > w_low + 0.1, (
        f"mel_pe selection weight did not track MEL: low(mel==ref)={w_low}, "
        f"high(mel=3ref)={w_high}. A weight that fires but stays flat reproduces "
        f"the V3-EXQ-861a failure this re-sourcing exists to fix."
    )

    # Re-sourcing is real: the legacy recency mode reads the same buffers and is
    # invariant to the injected MEL (bit-identical to the pre-repair 861a build).
    r_low = _spindle_pass_under_mel(1.0, mode="recency")["sws_spindle_selection_mean_weight"]
    r_high = _spindle_pass_under_mel(3.0, mode="recency")["sws_spindle_selection_mean_weight"]
    assert r_low == pytest.approx(r_high), (
        f"legacy 'recency' mode changed with MEL (low={r_low}, high={r_high}); "
        f"it must be invariant -- otherwise the mel_pe delta above is not "
        f"attributable to the re-sourced gate"
    )


def test_mech122_novelty_reference_mode_is_wired_through_from_dims():
    """Guard the reference-reeconfig-from-dims-silent-kwargs hazard for the new
    knob: the value passed to from_dims must be the value the agent actually
    holds at runtime (a knob dropped at any of the 3 wiring sites would be
    silently unreachable, making the fix inert while the manifest records it as
    set).
    """
    from ree_core.agent import REEAgent
    from tests.fixtures.tiny_configs import make_tiny_config
    from tests.fixtures.tiny_env import make_tiny_env

    env = make_tiny_env(seed=0)
    for mode in ("mel_pe", "recency"):
        cfg = make_tiny_config(env, mech122_novelty_reference_mode=mode)
        assert cfg.mech122_novelty_reference_mode == mode, (
            f"from_dims swallowed mech122_novelty_reference_mode={mode!r} "
            f"(config sees {cfg.mech122_novelty_reference_mode!r})"
        )
        agent = REEAgent(cfg)
        assert agent.config.mech122_novelty_reference_mode == mode
    # neutral default when unspecified
    assert make_tiny_config(env).mech122_novelty_reference_mode == "mel_pe"


# --------------------------------------------------------------------------- #
# SD-091/MECH-481 coalition controller                                        #
# --------------------------------------------------------------------------- #


def _coalition_agent(use_coalition: bool = True, seed: int = 0, **overrides):
    """Tiny-fixture agent + a fixed (body, world) observation pair.

    Returns (agent, body_obs, world_obs). The env is reset once and the SAME
    observation is re-fed every step, so any action difference between two arms
    is attributable to the config, not to divergent environment state.
    """
    from tests.fixtures.tiny_configs import make_tiny_config
    from tests.fixtures.tiny_env import make_tiny_env

    from ree_core.agent import REEAgent

    torch.manual_seed(seed)
    env = make_tiny_env(seed=seed)
    cfg = make_tiny_config(
        env, use_coalition_controller=use_coalition, **overrides
    )
    torch.manual_seed(123)
    agent = REEAgent(cfg)
    agent.reset()
    _flat, obs_dict = env.reset()
    body = obs_dict["body_state"]
    world = obs_dict["world_state"]
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    return agent, body, world


def _coalition_actions(agent, body, world, steps: int = 8) -> list:
    acts = []
    for i in range(steps):
        torch.manual_seed(1000 + i)
        with torch.no_grad():
            acts.append(agent.act_with_split_obs(body, world).clone())
    return acts


def test_use_coalition_controller_is_inert_until_requested_then_moves_actions():
    """SD-091/MECH-481: the flag must be inert alone and live once driven.

    This flag is a deliberate two-stage gate -- `use_coalition_controller=True`
    only CONSTRUCTS the controller; nothing calls `request_coalition()` unless a
    caller does so explicitly (agent.py's own comment says so). So the ordinary
    "flip the flag, watch the action stream move" probe would report NO delta and
    be indistinguishable from a dead flag. Both halves are asserted here:

      * inert half: ON with no coalition requested is bit-identical to OFF.
        (Also pinned as W2 in tests/contracts/test_sd091_coalition_controller_wiring.py;
        restated here because it is what makes the second half meaningful rather
        than an accident of a noisy substrate.)
      * live half: with a SENSORY_RESAMPLE coalition open, the same seeds and the
        same observation produce a DIFFERENT action stream -- so the gates
        (e1_sensory_encoder write_gate 0.9, e3_candidate_count channel_gain 1.5)
        actually reach the live decision path rather than being read nowhere.
    """
    from ree_core.claustrum.control_demand import ControlDemandType

    agent_off, b_off, w_off = _coalition_agent(use_coalition=False)
    assert agent_off.coalition is None

    agent_idle, b_idle, w_idle = _coalition_agent(use_coalition=True)
    assert agent_idle.coalition is not None

    acts_off = _coalition_actions(agent_off, b_off, w_off)
    acts_idle = _coalition_actions(agent_idle, b_idle, w_idle)
    for i, (a, b) in enumerate(zip(acts_off, acts_idle)):
        assert torch.equal(a, b), (
            f"enabled-but-unrequested coalition changed the action stream at "
            f"step {i}; the flag alone is supposed to be a no-op"
        )

    agent_on, b_on, w_on = _coalition_agent(use_coalition=True)
    state = agent_on.coalition.request_coalition(
        ControlDemandType.SENSORY_RESAMPLE, tick=0, max_duration_ticks=10_000
    )
    assert state is not None, "SENSORY_RESAMPLE template refused"
    assert float(agent_on.coalition.write_gate("e1_sensory_encoder")) < 1.0
    assert float(agent_on.coalition.channel_gain("e3_candidate_count")) > 1.0

    acts_on = _coalition_actions(agent_on, b_on, w_on)
    assert any(
        not torch.equal(a, b) for a, b in zip(acts_off, acts_on)
    ), (
        "an OPEN SENSORY_RESAMPLE coalition left the action stream byte-identical "
        "to the uncoalitioned baseline -- the coalition gates are inert at the "
        "consumer sites"
    )


def test_coalition_types_enabled_gates_which_templates_can_open():
    """`coalition_types_enabled` must actually reach CoalitionControllerConfig.

    Not a boolean master switch -- it is the tuple of ControlDemandType names the
    controller will honour, and it is the one knob that could silently widen or
    narrow the coalition surface without any other symptom. Two agents identical
    except for this tuple: the one that lists provenance_check opens the template
    and pulls its write gates off 1.0; the one that does not refuses the request
    (returns None, increments unregistered_request_count) and leaves every gate at
    the no-op baseline.
    """
    from ree_core.claustrum.control_demand import ControlDemandType

    agent_full, _b, _w = _coalition_agent(
        use_coalition=True,
        coalition_types_enabled=("sensory_resample", "provenance_check"),
    )
    agent_narrow, _b2, _w2 = _coalition_agent(
        use_coalition=True, coalition_types_enabled=("sensory_resample",)
    )

    opened = agent_full.coalition.request_coalition(
        ControlDemandType.PROVENANCE_CHECK, tick=0
    )
    refused = agent_narrow.coalition.request_coalition(
        ControlDemandType.PROVENANCE_CHECK, tick=0
    )

    assert opened is not None, (
        "provenance_check listed in coalition_types_enabled but the request was "
        "refused -- the REEConfig tuple is not reaching types_enabled"
    )
    assert refused is None, (
        "provenance_check NOT listed in coalition_types_enabled but the request "
        "was honoured -- the tuple is being ignored (default frozenset of every "
        "template leaking through)"
    )
    assert agent_narrow.coalition.unregistered_request_count == 1
    assert agent_full.coalition.unregistered_request_count == 0

    assert float(agent_full.coalition.write_gate("hippocampal_anchor_set")) < 1.0
    assert float(agent_narrow.coalition.write_gate("hippocampal_anchor_set")) == 1.0

    # The still-listed type stays openable on the narrowed agent -- the tuple
    # narrows the surface, it does not disable the controller wholesale.
    assert (
        agent_narrow.coalition.request_coalition(
            ControlDemandType.SENSORY_RESAMPLE, tick=0
        )
        is not None
    )


# --------------------------------------------------------------------------- #
# SD-014 incentive sensitization                                              #
# --------------------------------------------------------------------------- #


def test_incentive_sensitization_amplifies_the_wanting_write():
    """SD-014 decouple fix (V3-EXQ-887): WANTING must diverge from raw salience.

    OFF, `update_benefit_salience()` writes `salience` straight into
    VALENCE_WANTING via `update_valence()` and the per-node sensitization gain is
    never touched. ON, the write is routed through
    `ResidueField.update_wanting_sensitized()`, which first accumulates a
    drive-coupled per-node gain and then writes `salience * (1 + coupling * g)`.

    The activating condition has three parts, all of which the probe drives: the
    tonic-5HT master switch (salience is 0.0 otherwise, and the method returns
    early), at least one ACTIVE RBF center (there is no nearest center to write to
    before any residue has accumulated), and a non-zero drive_level (the gain
    increment is `rate * drive_level`, so a zero-drive arm is legitimately
    bit-identical to OFF and would make this look like a dead flag).

    Two observables, not one: the accumulated gain (exactly 0.0 OFF, since that
    code path is not entered at all) and the resulting WANTING readout.
    """
    from ree_core.residue.field import VALENCE_WANTING

    def wanting_and_gain(sensitized: bool, steps: int = 20, seed: int = 0):
        from tests.fixtures.tiny_configs import make_tiny_config
        from tests.fixtures.tiny_env import make_tiny_env

        from ree_core.agent import REEAgent

        torch.manual_seed(seed)
        env = make_tiny_env(seed=seed)
        cfg = make_tiny_config(
            env,
            tonic_5ht_enabled=True,
            incentive_sensitization_enabled=sensitized,
            sensitization_rate=0.5,
            sensitization_max=4.0,
            sensitization_coupling=1.0,
        )
        torch.manual_seed(123)
        agent = REEAgent(cfg)
        agent.reset()
        _flat, obs_dict = env.reset()
        body = obs_dict["body_state"]
        world = obs_dict["world_state"]
        if body.dim() == 1:
            body = body.unsqueeze(0)
        if world.dim() == 1:
            world = world.unsqueeze(0)

        for i in range(steps):
            torch.manual_seed(1000 + i)
            with torch.no_grad():
                latent = agent.sense(body, world)
                agent.act_with_split_obs(body, world)
                agent.update_residue(-1.0)  # activate RBF centers
                agent.serotonin_step(1.0)
                agent.update_benefit_salience(1.0, drive_level=1.0)

        with torch.no_grad():
            z = agent._current_latent.z_world
            wanting = float(
                agent.residue_field.evaluate_valence(z)[0, VALENCE_WANTING]
            )
            gain = float(agent.residue_field.rbf_field.sensitization_gain.max())
        assert agent.residue_field.rbf_field.active_mask.any(), (
            "no active RBF center -- the probe never drove the activating "
            "condition, so a null here would be meaningless"
        )
        return wanting, gain

    off_wanting, off_gain = wanting_and_gain(False)
    on_wanting, on_gain = wanting_and_gain(True)

    assert off_gain == 0.0, (
        f"sensitization gain moved to {off_gain} with the flag OFF -- the "
        "sensitized write path is running unconditionally"
    )
    assert on_gain > 0.0, (
        "sensitization gain stayed at 0.0 with the flag ON despite a non-zero "
        "drive_level -- update_wanting_sensitized() is not being reached"
    )
    assert on_wanting > off_wanting + 1e-6, (
        f"WANTING readout unchanged by incentive sensitization "
        f"(ON={on_wanting}, OFF={off_wanting}); the amplification is inert and "
        "wanting has not decoupled from raw benefit_exposure"
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


def test_sd_residue_valence_bound_bounds_the_accumulator():
    """SD-RESIDUE-VALENCE-BOUND: valence_bounding_enabled must actually bound
    RBFLayer.update_valence()'s accumulator, and OFF (default) must be
    bit-identical to the pre-fix unclamped `+=`.

    Direct ResidueField probe (not the REEAgent tiny harness) -- the fix is a
    single write-path change on ResidueField/RBFLayer, so this is the right
    level, matching V3-EXQ-918's own validation design.
    """
    from ree_core.residue.field import ResidueField, VALENCE_POSITIVE_SURPRISE
    from ree_core.utils.config import ResidueConfig

    z = torch.zeros(4)
    n_writes = 100

    off = ResidueField(ResidueConfig(world_dim=4, num_basis_functions=4))
    off.accumulate(z, harm_magnitude=0.1)
    for _ in range(n_writes):
        off.update_valence(z, VALENCE_POSITIVE_SURPRISE, 1.0)
    off_value = off.evaluate_valence(z.unsqueeze(0))[0, VALENCE_POSITIVE_SURPRISE].item()
    assert off_value == pytest.approx(float(n_writes)), (
        f"valence_bounding_enabled=False (default) must be bit-identical to the "
        f"pre-fix `+=` -- expected exactly {n_writes}.0, got {off_value}"
    )

    on = ResidueField(ResidueConfig(
        world_dim=4, num_basis_functions=4,
        valence_bounding_enabled=True, valence_decay_rate=0.02, valence_clamp_abs=5.0,
    ))
    on.accumulate(z, harm_magnitude=0.1)
    for _ in range(n_writes):
        on.update_valence(z, VALENCE_POSITIVE_SURPRISE, 1.0)
    on_value = on.evaluate_valence(z.unsqueeze(0))[0, VALENCE_POSITIVE_SURPRISE].item()
    assert on_value < 10.0, (
        f"valence_bounding_enabled=True must bound the accumulator well below "
        f"the unbounded {n_writes}.0 -- got {on_value}, expected near clamp_abs=5.0"
    )
    assert on_value != off_value


# --------------------------------------------------------------------------- #
# E3Config candidate-scoring levers (2026-08-11 nested-scan individual audit) #
# --------------------------------------------------------------------------- #

def _e3_selector(**cfg_kw):
    """Direct E3TrajectorySelector construction (no REEAgent harness needed).

    Mirrors tests/contracts/test_dr12_pe_confidence.py's `_selector` helper.
    """
    from ree_core.predictors.e3_selector import E3Config, E3TrajectorySelector

    sel = E3TrajectorySelector(E3Config(world_dim=6, hidden_dim=8, **cfg_kw))
    sel._running_variance = 0.0  # deterministic committed argmin path
    return sel


def _e3_candidate(action_class: int, action_dim: int = 5):
    """One candidate Trajectory with a one-hot first action (distinguishable
    per-candidate action features -- what MECH-440/441 read)."""
    import torch as _torch

    from ree_core.predictors.e2_fast import Trajectory

    world_dim = 6
    horizon = 3
    states = [_torch.zeros(1, world_dim) for _ in range(horizon + 1)]
    world_states = [_torch.zeros(1, world_dim) for _ in range(horizon + 1)]
    actions = _torch.zeros(1, horizon, action_dim)
    actions[:, 0, action_class] = 1.0
    return Trajectory(states=states, actions=actions, world_states=world_states)


def test_use_noisy_selection_head_bias_is_nonzero_only_when_enabled():
    """MECH-440: the head's per-candidate additive bias must actually reach
    last_score_diagnostics, and must be exactly zero unless BOTH the flag is
    on and sigma_init > 0 (the documented no-op-default contract).
    """
    cands = [_e3_candidate(0), _e3_candidate(1), _e3_candidate(2)]

    sel_off = _e3_selector()  # use_noisy_selection_head defaults False
    sel_off.select(cands, temperature=1.0)
    assert sel_off.last_score_diagnostics["noisy_selection_active"] is False
    assert sel_off.last_score_diagnostics["noisy_selection_bias_range"] == 0.0

    sel_on_zero_sigma = _e3_selector(
        use_noisy_selection_head=True, noisy_selection_sigma_init=0.0
    )
    sel_on_zero_sigma.select(cands, temperature=1.0)
    assert sel_on_zero_sigma.last_score_diagnostics["noisy_selection_active"] is True
    assert sel_on_zero_sigma.last_score_diagnostics["noisy_selection_bias_range"] == 0.0, (
        "sigma_init=0.0 must still produce exactly zero bias (bit-identical "
        "no-op contract), even with the master flag ON"
    )

    torch.manual_seed(0)
    sel_on = _e3_selector(use_noisy_selection_head=True, noisy_selection_sigma_init=1.0)
    sel_on.select(cands, temperature=1.0)
    assert sel_on.last_score_diagnostics["noisy_selection_active"] is True
    assert sel_on.last_score_diagnostics["noisy_selection_bias_range"] > 0.0, (
        "use_noisy_selection_head=True with a nonzero sigma_init produced zero "
        "bias range; the NoisyNet head is not reaching the modulatory "
        "accumulator (inert flag)"
    )


def test_use_model_disagreement_curiosity_bonus_flips_selection():
    """MECH-441: a per-candidate model-disagreement bonus must propagate all
    the way to the committed argmin, not just tick a diagnostic counter.
    """
    cands = [_e3_candidate(0), _e3_candidate(1), _e3_candidate(2)]
    bias = torch.tensor([-1.0, 0.0, 0.0])  # candidate 0 favoured by the primary

    sel_off = _e3_selector()
    off = sel_off.select(cands, temperature=1.0, score_bias=bias)
    assert off.selected_index == 0  # unconditional-trust baseline picks the primary-best
    assert sel_off.last_score_diagnostics["model_disagreement_active"] is False

    sel_on = _e3_selector(
        use_model_disagreement_curiosity=True, model_disagreement_weight=5.0
    )
    on = sel_on.select(
        cands, temperature=1.0, score_bias=bias,
        model_disagreement_per_candidate=torch.tensor([0.0, 10.0, 0.0]),  # bonus on cand 1
    )
    assert on.selected_index != 0, (
        "use_model_disagreement_curiosity=True with a decisive bonus on a "
        "non-primary candidate did not change the committed argmin (inert flag)"
    )
    assert sel_on.last_score_diagnostics["model_disagreement_active"] is True
    assert sel_on.last_score_diagnostics["model_disagreement_bonus_range"] > 0.0


# --------------------------------------------------------------------------- #
# LatentStackConfig construction-gate flags (2026-08-11 nested-scan individual #
# audit, batch: LatentStackConfig)                                            #
# --------------------------------------------------------------------------- #

def _latent_stack(seed: int = 0, **kw):
    """Direct LatentStack construction -- mirrors test_dr13_self_recurrence.py's
    `_stack` helper. No REEAgent harness needed for a construction-gate probe.
    """
    from ree_core.latent.stack import LatentStack
    from ree_core.utils.config import LatentStackConfig

    torch.manual_seed(seed)
    return LatentStack(LatentStackConfig(**kw))


def test_use_harm_stream_populates_z_harm_only_when_enabled():
    """SD-010: z_harm is produced by the dedicated HarmEncoder only when
    use_harm_stream=True AND harm_obs is supplied. The MECH-099 lateral head
    (the only other z_harm source) is off by default (harm_dim=0), so an
    OFF-arm z_harm can only come from a leak past this gate.
    """
    stack_off = _latent_stack()
    stack_on = _latent_stack(use_harm_stream=True)
    assert stack_off.harm_encoder is None
    assert stack_on.harm_encoder is not None

    obs = torch.randn(1, stack_off.config.body_obs_dim + stack_off.config.world_obs_dim)
    harm_obs = torch.randn(1, stack_off.config.harm_obs_dim)

    off_state = stack_off.encode(obs, harm_obs=harm_obs)
    on_state = stack_on.encode(obs, harm_obs=harm_obs)

    assert off_state.z_harm is None, (
        "z_harm populated with use_harm_stream=False; harm_obs is reaching "
        "z_harm through a path other than the dedicated HarmEncoder"
    )
    assert on_state.z_harm is not None, (
        "use_harm_stream=True with harm_obs supplied did not populate "
        "z_harm (inert flag)"
    )
    assert on_state.z_harm.shape == (1, stack_on.config.z_harm_dim)


def test_use_affective_harm_stream_populates_z_harm_a_only_when_enabled():
    """SD-011: z_harm_a is produced by AffectiveHarmEncoder only when
    use_affective_harm_stream=True AND harm_obs_a is supplied.
    """
    stack_off = _latent_stack()
    stack_on = _latent_stack(use_affective_harm_stream=True)
    assert stack_off.affective_harm_encoder is None
    assert stack_on.affective_harm_encoder is not None

    obs = torch.randn(1, stack_off.config.body_obs_dim + stack_off.config.world_obs_dim)
    harm_obs_a = torch.randn(1, stack_off.config.harm_obs_a_dim)

    off_state = stack_off.encode(obs, harm_obs_a=harm_obs_a)
    on_state = stack_on.encode(obs, harm_obs_a=harm_obs_a)

    assert off_state.z_harm_a is None
    assert on_state.z_harm_a is not None, (
        "use_affective_harm_stream=True with harm_obs_a supplied did not "
        "populate z_harm_a (inert flag)"
    )
    assert on_state.z_harm_a.shape == (1, stack_on.config.z_harm_a_dim)


def test_use_event_classifier_populates_event_logits_only_when_enabled():
    """SD-009: SplitEncoder.event_classifier gates event_logits -- None when
    use_event_classifier=False, [batch, 3] logits when True.
    """
    stack_off = _latent_stack()
    stack_on = _latent_stack(use_event_classifier=True)
    assert stack_off.split_encoder.event_classifier is None
    assert stack_on.split_encoder.event_classifier is not None

    obs = torch.randn(1, stack_off.config.body_obs_dim + stack_off.config.world_obs_dim)
    off_state = stack_off.encode(obs)
    on_state = stack_on.encode(obs)

    assert off_state.event_logits is None
    assert on_state.event_logits is not None, (
        "use_event_classifier=True did not populate event_logits (inert flag)"
    )
    assert on_state.event_logits.shape == (1, 3)


def test_use_resource_proximity_head_populates_resource_prox_pred_only_when_enabled():
    """SD-018: SplitEncoder.resource_proximity_head gates resource_prox_pred --
    None when use_resource_proximity_head=False, [batch, 1] when True.
    """
    stack_off = _latent_stack()
    stack_on = _latent_stack(use_resource_proximity_head=True)
    assert stack_off.split_encoder.resource_proximity_head is None
    assert stack_on.split_encoder.resource_proximity_head is not None

    obs = torch.randn(1, stack_off.config.body_obs_dim + stack_off.config.world_obs_dim)
    off_state = stack_off.encode(obs)
    on_state = stack_on.encode(obs)

    assert off_state.resource_prox_pred is None
    assert on_state.resource_prox_pred is not None, (
        "use_resource_proximity_head=True did not populate resource_prox_pred "
        "(inert flag)"
    )
    assert on_state.resource_prox_pred.shape == (1, 1)


def test_use_resource_encoder_populates_z_resource_only_when_enabled():
    """SD-015/MECH-112: ResourceEncoder produces z_resource independently of
    z_world only when use_resource_encoder=True. Also pins that the sibling
    identity-classifier head stays off by its own default even with the
    resource encoder itself enabled (see the paired test below).
    """
    stack_off = _latent_stack()
    stack_on = _latent_stack(use_resource_encoder=True)
    assert stack_off.resource_encoder is None
    assert stack_on.resource_encoder is not None

    obs = torch.randn(1, stack_off.config.body_obs_dim + stack_off.config.world_obs_dim)
    off_state = stack_off.encode(obs)
    on_state = stack_on.encode(obs)

    assert off_state.z_resource is None
    assert on_state.z_resource is not None, (
        "use_resource_encoder=True did not populate z_resource (inert flag)"
    )
    assert on_state.z_resource.shape == (1, stack_on.config.z_resource_dim)
    assert on_state.identity_logits is None, (
        "use_identity_classifier defaults False; identity_logits must stay "
        "None even with the resource encoder itself on"
    )


def test_use_identity_classifier_populates_identity_logits_only_when_enabled():
    """SD-049 Phase 2: the identity-classifier head rides on ResourceEncoder's
    trunk and requires BOTH use_resource_encoder AND use_identity_classifier
    -- the resource encoder alone (previous test) leaves identity_logits None.
    """
    stack_resource_only = _latent_stack(use_resource_encoder=True)
    stack_both = _latent_stack(use_resource_encoder=True, use_identity_classifier=True)
    assert stack_resource_only.resource_encoder.identity_head is None
    assert stack_both.resource_encoder.identity_head is not None

    obs = torch.randn(
        1, stack_resource_only.config.body_obs_dim + stack_resource_only.config.world_obs_dim
    )
    resource_only_state = stack_resource_only.encode(obs)
    both_state = stack_both.encode(obs)

    assert resource_only_state.identity_logits is None
    assert both_state.identity_logits is not None, (
        "use_resource_encoder=True + use_identity_classifier=True did not "
        "populate identity_logits (inert flag)"
    )
    assert both_state.identity_logits.shape == (
        1, stack_both.config.identity_classifier_n_types
    )


def test_use_harm_un_populates_z_harm_un_only_when_enabled():
    """SD-019a: the harm_unpleasantness_channel EMA (z_harm_un) is written on
    a waking sense() only when use_harm_un=True and a live z_harm exists
    (use_harm_stream=True + harm_obs supplied). Mirrors
    test_mech_219_harm_suffering_accumulator.py's `_harm_cfg` harness.
    """
    from ree_core.agent import REEAgent
    from ree_core.utils.config import REEConfig

    def _agent(use_harm_un: bool) -> REEAgent:
        cfg = REEConfig.from_dims(
            body_obs_dim=10, world_obs_dim=54, action_dim=4,
            use_harm_stream=True,
        )
        cfg.latent.use_harm_un = use_harm_un
        return REEAgent(cfg)

    obs_body = torch.randn(1, 10)
    obs_world = torch.randn(1, 54)
    harm_obs = torch.randn(1, 51)

    off_latent = _agent(False).sense(obs_body, obs_world, obs_harm=harm_obs)
    on_latent = _agent(True).sense(obs_body, obs_world, obs_harm=harm_obs)

    assert off_latent.z_harm_un is None, (
        "z_harm_un populated with use_harm_un=False (inert-flag guard)"
    )
    assert on_latent.z_harm_un is not None, (
        "use_harm_un=True with a live z_harm did not populate z_harm_un "
        "(inert flag)"
    )


def test_use_e2_harm_s_forward_constructs_agent_e2_harm_s_only_when_enabled():
    """ARC-033: agent.e2_harm_s (E2HarmSForward) is constructed at agent build
    time iff use_e2_harm_s_forward=True -- a pure construction gate.
    """
    from ree_core.agent import REEAgent
    from ree_core.utils.config import REEConfig

    def _agent(flag: bool) -> REEAgent:
        cfg = REEConfig.from_dims(body_obs_dim=10, world_obs_dim=54, action_dim=4)
        cfg.latent.use_e2_harm_s_forward = flag
        return REEAgent(cfg)

    assert _agent(False).e2_harm_s is None
    assert _agent(True).e2_harm_s is not None, (
        "use_e2_harm_s_forward=True did not construct agent.e2_harm_s "
        "(inert flag)"
    )


def test_use_e2_world_uncertainty_constructs_agent_head_only_when_enabled():
    """SD-063: agent.e2_world_uncertainty (E2WorldUncertaintyHead) is
    constructed at agent build time iff use_e2_world_uncertainty=True -- a
    pure construction gate, standalone module sharing no params with E2World.
    """
    from ree_core.agent import REEAgent
    from ree_core.utils.config import REEConfig

    def _agent(flag: bool) -> REEAgent:
        cfg = REEConfig.from_dims(body_obs_dim=10, world_obs_dim=54, action_dim=4)
        cfg.latent.use_e2_world_uncertainty = flag
        return REEAgent(cfg)

    assert _agent(False).e2_world_uncertainty is None
    assert _agent(True).e2_world_uncertainty is not None, (
        "use_e2_world_uncertainty=True did not construct "
        "agent.e2_world_uncertainty (inert flag)"
    )


# --------------------------------------------------------------------------- #
# E2Config / HeartbeatConfig / ResidueConfig flags (2026-08-11 nested-scan     #
# individual audit, batch: E2Config/HeartbeatConfig/ResidueConfig)            #
# --------------------------------------------------------------------------- #

def test_benefit_terrain_enabled_gates_accumulate_and_evaluate_benefit():
    """ARC-030/MECH-117: benefit_rbf_field is constructed, and
    accumulate_benefit/evaluate_benefit stop being no-ops, only when
    benefit_terrain_enabled=True.
    """
    from ree_core.residue.field import ResidueField
    from ree_core.utils.config import ResidueConfig

    field_off = ResidueField(ResidueConfig(world_dim=8, num_basis_functions=8))
    field_on = ResidueField(
        ResidueConfig(world_dim=8, num_basis_functions=8, benefit_terrain_enabled=True)
    )
    assert not hasattr(field_off, "benefit_rbf_field")
    assert hasattr(field_on, "benefit_rbf_field")

    z = torch.randn(1, 8)
    field_off.accumulate_benefit(z, benefit_magnitude=5.0)
    field_on.accumulate_benefit(z, benefit_magnitude=5.0)

    off_val = field_off.evaluate_benefit(z)
    on_val = field_on.evaluate_benefit(z)
    assert torch.all(off_val == 0.0), (
        "benefit_terrain_enabled=False must keep evaluate_benefit at zeros"
    )
    assert torch.any(on_val != 0.0), (
        "benefit_terrain_enabled=True with a real accumulate_benefit call did "
        "not produce a nonzero evaluate_benefit read (inert flag)"
    )


def test_safety_terrain_enabled_gates_accumulate_and_evaluate_safety():
    """MECH-303: safety_terrain_rbf_field is constructed, and
    accumulate_safety/evaluate_safety stop being no-ops, only when
    safety_terrain_enabled=True.
    """
    from ree_core.residue.field import ResidueField
    from ree_core.utils.config import ResidueConfig

    field_off = ResidueField(ResidueConfig(world_dim=8, num_basis_functions=8))
    field_on = ResidueField(
        ResidueConfig(world_dim=8, num_basis_functions=8, safety_terrain_enabled=True)
    )
    assert not hasattr(field_off, "safety_terrain_rbf_field")
    assert hasattr(field_on, "safety_terrain_rbf_field")

    z = torch.randn(1, 8)
    for _ in range(10):
        field_off.accumulate_safety(z, safety_magnitude=0.5)
        field_on.accumulate_safety(z, safety_magnitude=0.5)

    off_val = field_off.evaluate_safety(z)
    on_val = field_on.evaluate_safety(z)
    assert torch.all(off_val == 0.0), (
        "safety_terrain_enabled=False must keep evaluate_safety at zeros"
    )
    assert torch.any(on_val != 0.0), (
        "safety_terrain_enabled=True with real accumulate_safety calls did "
        "not produce a nonzero evaluate_safety read (inert flag)"
    )


def test_valence_enabled_gates_update_and_evaluate_valence():
    """ResidueField.update_valence/evaluate_valence early-return (skip write /
    zeros) when valence_enabled=False. Default is True (unlike its terrain
    siblings), so the ON arm here is the bare default config.
    """
    from ree_core.residue.field import ResidueField, VALENCE_WANTING
    from ree_core.utils.config import ResidueConfig

    z = torch.zeros(4)

    off = ResidueField(ResidueConfig(world_dim=4, num_basis_functions=4, valence_enabled=False))
    off.accumulate(z, harm_magnitude=0.1)  # seeds an active center
    off.update_valence(z, VALENCE_WANTING, 5.0)
    off_val = off.evaluate_valence(z.unsqueeze(0))
    assert torch.all(off_val == 0.0), (
        "valence_enabled=False must keep evaluate_valence at zeros even after "
        "a real update_valence call"
    )

    on = ResidueField(ResidueConfig(world_dim=4, num_basis_functions=4))  # valence_enabled=True default
    on.accumulate(z, harm_magnitude=0.1)
    on.update_valence(z, VALENCE_WANTING, 5.0)
    on_val = on.evaluate_valence(z.unsqueeze(0))
    assert on_val[0, VALENCE_WANTING].item() != 0.0, (
        "valence_enabled=True (default) with a real update_valence call did "
        "not populate evaluate_valence (inert flag)"
    )


def test_ewc_enabled_gates_ewc_penalty():
    """MECH-334: ewc_penalty() returns a hard 0.0 scalar unless ewc_enabled=True
    (and lambda>0 and an anchor was snapshotted); ON it must produce a nonzero
    Fisher-weighted pull once weights drift away from the snapshotted anchor.
    """
    from ree_core.residue.field import ResidueField
    from ree_core.utils.config import ResidueConfig

    def _field(ewc_enabled: bool) -> ResidueField:
        return ResidueField(ResidueConfig(
            world_dim=4, num_basis_functions=4,
            ewc_enabled=ewc_enabled, ewc_lambda=1.0,
        ))

    z = torch.zeros(4)

    off = _field(False)
    off.accumulate(z, harm_magnitude=1.0)
    off.snapshot_ewc_anchor()
    with torch.no_grad():
        off.rbf_field.weights += 10.0
    assert off.ewc_penalty().item() == 0.0, (
        "ewc_enabled=False must return exactly 0.0 regardless of weight drift"
    )

    on = _field(True)
    on.accumulate(z, harm_magnitude=1.0)
    on.snapshot_ewc_anchor()
    with torch.no_grad():
        on.rbf_field.weights += 10.0
    penalty = on.ewc_penalty()
    assert penalty.item() > 0.0, (
        "ewc_enabled=True with a captured anchor and drifted weights produced "
        "a zero penalty (inert flag)"
    )


def test_cross_stream_binding_enabled_couples_the_rollout_streams():
    """cross_stream_binding_substrate: the binder is constructed only when
    cross_stream_binding_enabled=True, and when constructed it must actually
    perturb rollout_with_world's per-step z_self/z_world beyond the two
    independent forward models.

    Compared WITHIN one ON predictor instance (coupled rollout vs a manual
    replay of predict_next_self/world_forward alone on the SAME weights) --
    the KNOWN_UNPROBED_NESTED reason this flag carried flagged the obvious
    two-instance comparison as confounded by construction-time RNG (the
    binder's own Linear layers consume extra random draws), so this probe
    deliberately never constructs a second ON-vs-OFF pair to compare rollouts.
    """
    from ree_core.predictors.e2_fast import E2FastPredictor
    from ree_core.utils.config import E2Config

    torch.manual_seed(0)
    pred_off = E2FastPredictor(E2Config(self_dim=8, world_dim=8, action_dim=4))
    assert pred_off.cross_stream_binder is None

    torch.manual_seed(0)
    pred_on = E2FastPredictor(
        E2Config(self_dim=8, world_dim=8, action_dim=4, cross_stream_binding_enabled=True)
    )
    assert pred_on.cross_stream_binder is not None

    torch.manual_seed(1)
    z_self = torch.randn(1, 8)
    z_world = torch.randn(1, 8)
    actions = torch.randn(1, 3, 4)

    coupled = pred_on.rollout_with_world(z_self, z_world, actions, compute_action_objects=False)

    # Manual uncoupled replay using pred_on's OWN weights -- isolates the
    # binder's couple() step as the only variable.
    zs, zw = z_self, z_world
    for t in range(3):
        a = actions[:, t, :]
        zs = pred_on.predict_next_self(zs, a)
        zw = pred_on.world_forward(zw, a)

    assert not torch.allclose(coupled.states[-1], zs), (
        "cross_stream_binding_enabled=True did not perturb the rollout's "
        "z_self stream relative to the uncoupled prediction (inert flag)"
    )
    assert not torch.allclose(coupled.world_states[-1], zw), (
        "cross_stream_binding_enabled=True did not perturb the rollout's "
        "z_world stream relative to the uncoupled prediction (inert flag)"
    )


def test_use_commit_readiness_gate_blocks_elevation_below_the_floor():
    """MECH-090 R-c: should_admit_elevation is unconditionally True with the
    gate OFF (legacy rv-only elevation), and blocks a below-floor score_margin
    only when use_commit_readiness_gate=True.
    """
    from ree_core.heartbeat.beta_gate import BetaGate

    gate_off = BetaGate(use_commit_readiness_gate=False, commit_readiness_floor=0.05)
    assert gate_off.should_admit_elevation(score_margin=0.0, n_candidates=5) is True

    gate_on = BetaGate(use_commit_readiness_gate=True, commit_readiness_floor=0.05)
    assert gate_on.should_admit_elevation(score_margin=0.0, n_candidates=5) is False, (
        "use_commit_readiness_gate=True with score_margin below the floor did "
        "not block elevation (inert flag)"
    )
    assert gate_on.should_admit_elevation(score_margin=0.5, n_candidates=5) is True


# --------------------------------------------------------------------------- #
# HippocampalConfig flags (2026-08-11 nested-scan individual audit, batch:    #
# HippocampalConfig)                                                          #
# --------------------------------------------------------------------------- #

def _hippo_module(use_anchor_sets: bool = False, **kw):
    """Direct HippocampalModule construction -- mirrors
    test_mech_269_anchor_set.py's `_make_module` helper.
    """
    from ree_core.utils.config import HippocampalConfig, E2Config, ResidueConfig
    from ree_core.predictors.e2_fast import E2FastPredictor
    from ree_core.residue.field import ResidueField
    from ree_core.hippocampal.module import HippocampalModule

    hcfg = HippocampalConfig(
        world_dim=8, action_dim=4, action_object_dim=4,
        hidden_dim=16, horizon=3, num_candidates=4,
        num_cem_iterations=1, elite_fraction=0.5,
        use_anchor_sets=use_anchor_sets,
        **kw,
    )
    e2cfg = E2Config(self_dim=8, world_dim=8, action_dim=4, action_object_dim=4)
    rcfg = ResidueConfig(world_dim=8, num_basis_functions=4)
    return HippocampalModule(hcfg, E2FastPredictor(e2cfg), ResidueField(rcfg))


def test_mode_conditioning_enabled_gates_mode_weighted_cem_scales():
    """MECH-267 / SD-MECH267-HORIZON-DEPTH: _compute_mode_noise_scale and
    _compute_mode_horizon_scale are pure functions of an operating_mode dict.
    Both return None when mode_conditioning_enabled=False (caller leaves the
    CEM proposal std/scoring window untouched); ON, they return the real
    probability-weighted average from config.mode_noise_scale /
    mode_horizon_scale.
    """
    operating_mode = {"explore": 1.0}

    mod_off = _hippo_module(
        mode_conditioning_enabled=False, mode_noise_scale={"explore": 2.0}
    )
    assert mod_off._compute_mode_noise_scale(operating_mode) is None
    assert mod_off._compute_mode_horizon_scale(operating_mode) is None

    mod_on = _hippo_module(
        mode_conditioning_enabled=True, mode_noise_scale={"explore": 2.0}
    )
    on_noise = mod_on._compute_mode_noise_scale(operating_mode)
    assert on_noise == 2.0, (
        "mode_conditioning_enabled=True did not apply the configured "
        "mode_noise_scale weighting (inert flag)"
    )
    on_horizon = mod_on._compute_mode_horizon_scale(operating_mode)
    assert on_horizon is not None


def test_use_staleness_accumulator_gates_construction_and_integration():
    """MECH-284 Phase 3: HippocampalModule.staleness_accumulator is
    constructed, and integrate_staleness() stops being a no-op, only when
    use_staleness_accumulator=True.
    """
    from ree_core.regulators.invalidation_trigger import BroadcastEvent

    mod_off = _hippo_module(use_anchor_sets=True, use_staleness_accumulator=False)
    mod_on = _hippo_module(use_anchor_sets=True, use_staleness_accumulator=True)
    assert mod_off.staleness_accumulator is None
    assert mod_on.staleness_accumulator is not None

    z = torch.randn(8)
    mod_off.anchor_set.write_anchor("fast", "0.1", ("z_world",), z)
    mod_on.anchor_set.write_anchor("fast", "0.1", ("z_world",), z)

    bcast = BroadcastEvent(
        t=0, strength=1.0, posterior=1.0, targets=["fast"],
        source_scale="fast", source_segment_id_old="0.0",
        source_segment_id_new="0.1", source_sources=["z_world"],
    )

    mod_off.integrate_staleness([bcast])  # no accumulator -> no-op
    mod_on.integrate_staleness([bcast])

    assert mod_off.staleness_accumulator is None
    assert mod_on.staleness_accumulator.get(("fast", "0.1")) > 0.0, (
        "use_staleness_accumulator=True with a real broadcast + active "
        "anchor did not credit any staleness (inert flag)"
    )


def test_use_mech284_hysteresis_swaps_the_staleness_lookup_source():
    """MECH-284/MECH-269: with use_mech284_hysteresis=False, tick_anchor_set
    drives AnchorSet.tick_hysteresis off the internal tick-delta proxy
    (near-zero immediately after a fresh write). With it True, the SAME
    per-tick inputs instead read a pre-populated StalenessAccumulator entry,
    so an anchor whose region has accumulated high staleness is released
    where the proxy-driven OFF arm would not release it -- the synthetic
    staleness-vs-tick-delta divergence scenario the prior audit's
    KNOWN_UNPROBED_NESTED reason said was needed but not yet built.
    """
    from types import SimpleNamespace
    from ree_core.utils.config import AnchorSetConfig

    def _mod(hysteresis_flag: bool):
        return _hippo_module(
            use_anchor_sets=True,
            use_staleness_accumulator=True,
            use_mech284_hysteresis=hysteresis_flag,
            anchor_set=AnchorSetConfig(
                hysteresis_k=1, reset_threshold=0.3, staleness_rate=0.005
            ),
        )

    for flag, expect_fire in [(False, False), (True, True)]:
        mod = _mod(flag)
        z = torch.randn(8)
        mod.anchor_set.write_anchor("fast", "0.1", ("z_world",), z)
        if flag:
            # Pre-populate the accumulator with near-max staleness for this
            # region -- the internal tick-delta proxy would read ~0.005 at
            # this point (one tick after write), which would NOT cross
            # reset_threshold=0.3 given per_stream_vs=0.5 below.
            mod.staleness_accumulator._staleness[("fast", "0.1")] = 1.0
        mod.per_stream_vs = {"z_world": 0.5}
        mod.tick_anchor_set(SimpleNamespace(z_world=z.unsqueeze(0)), events=[])
        active_keys = [a.key for a in mod.anchor_set.active_anchors()]
        fired = ("fast", "0.1", ("z_world",)) not in active_keys
        assert fired == expect_fire, (
            f"use_mech284_hysteresis={flag}: expected fired={expect_fire}, "
            f"got {fired} (flag did not swap the staleness_lookup source)"
        )


def test_use_backward_credit_sweep_writes_retroactive_valence_credit():
    """MECH-290: record_committed_trajectory/backward_credit_sweep are no-ops
    (empty dict, no valence write) unless use_backward_credit_sweep=True; ON,
    a real committed trajectory's z_world states get decayed VALENCE_WANTING
    credit written backward from the endpoint.
    """
    from ree_core.predictors.e2_fast import Trajectory
    from ree_core.residue.field import VALENCE_WANTING

    world_states = [torch.zeros(1, 8) for _ in range(4)]
    actions = torch.zeros(1, 3, 4)

    def _traj():
        return Trajectory(
            states=[torch.zeros(1, 8) for _ in range(4)],
            actions=actions,
            world_states=[s.clone() for s in world_states],
        )

    mod_off = _hippo_module(use_backward_credit_sweep=False)
    mod_off.residue_field.accumulate(torch.zeros(8), harm_magnitude=0.1)
    mod_off.record_committed_trajectory(_traj())
    result_off = mod_off.backward_credit_sweep(outcome_quality=0.9)
    assert result_off == {}
    off_val = mod_off.residue_field.evaluate_valence(torch.zeros(1, 8))[0, VALENCE_WANTING].item()
    assert off_val == 0.0

    mod_on = _hippo_module(use_backward_credit_sweep=True)
    mod_on.residue_field.accumulate(torch.zeros(8), harm_magnitude=0.1)
    mod_on.record_committed_trajectory(_traj())
    result_on = mod_on.backward_credit_sweep(outcome_quality=0.9)
    assert result_on["n_steps_swept"] == 4, (
        "use_backward_credit_sweep=True did not sweep the recorded committed "
        "trajectory (inert flag)"
    )
    assert result_on["mean_credit"] > 0.0
    on_val = mod_on.residue_field.evaluate_valence(torch.zeros(1, 8))[0, VALENCE_WANTING].item()
    assert on_val > 0.0, (
        "use_backward_credit_sweep=True did not actually write VALENCE_WANTING "
        "credit into the residue field (inert flag)"
    )


def test_use_offline_wanting_spread_writes_decayed_credit_to_earlier_waypoints():
    """MECH-217: spread_reverse_replay_wanting is a no-op (empty dict) unless
    use_offline_wanting_spread=True; ON, a decayed fraction of the terminus's
    VALENCE_WANTING is written to earlier waypoints of a reverse-replayed
    trajectory.
    """
    from ree_core.predictors.e2_fast import Trajectory
    from ree_core.residue.field import VALENCE_WANTING

    z = torch.zeros(1, 8)

    def _traj():
        return Trajectory(
            states=[z.clone() for _ in range(4)],
            actions=torch.zeros(1, 3, 4),
            world_states=[z.clone() for _ in range(4)],
        )

    mod_off = _hippo_module(use_offline_wanting_spread=False)
    mod_off.residue_field.accumulate(z.squeeze(0), harm_magnitude=0.1)
    mod_off.residue_field.update_valence(z, VALENCE_WANTING, 1.0)
    result_off = mod_off.spread_reverse_replay_wanting(_traj())
    assert result_off == {}

    mod_on = _hippo_module(use_offline_wanting_spread=True)
    mod_on.residue_field.accumulate(z.squeeze(0), harm_magnitude=0.1)
    mod_on.residue_field.update_valence(z, VALENCE_WANTING, 1.0)
    before = mod_on.residue_field.evaluate_valence(z)[0, VALENCE_WANTING].item()
    result_on = mod_on.spread_reverse_replay_wanting(_traj())
    assert result_on["n_steps_spread"] == 3
    assert result_on["wanting_at_terminus"] > 0.0
    after = mod_on.residue_field.evaluate_valence(z)[0, VALENCE_WANTING].item()
    assert after > before, (
        "use_offline_wanting_spread=True did not increase VALENCE_WANTING at "
        "an earlier waypoint (inert flag)"
    )


# --------------------------------------------------------------------------- #
# E3Config benefit-eval warmup gate + AnchorSet/GhostGoalBank/GoalConfig       #
# flags (2026-08-11 nested-scan individual audit, batch: AnchorSetConfig /    #
# GhostGoalBankConfig / GoalConfig / benefit_eval_enabled)                    #
# --------------------------------------------------------------------------- #

def test_benefit_eval_enabled_gates_score_trajectory_after_warmup():
    """ARC-030/MECH-112: score_trajectory subtracts benefit_weight *
    compute_benefit_score from cost only once BOTH benefit_eval_enabled=True
    AND the warmup gate (_BENEFIT_WARMUP_SAMPLES) has been cleared via
    record_benefit_sample(). Before warmup (even with the flag on) it must
    be bit-identical to OFF.

    Compared WITHIN one selector instance (toggle config.benefit_eval_enabled
    after construction) rather than two separately-constructed selectors, so
    unrelated weight-init RNG divergence between instances cannot be mistaken
    for the flag's effect.
    """
    sel = _e3_selector()  # benefit_eval_enabled=False by default
    traj = _e3_candidate(0)

    off_score = sel.score_trajectory(traj)

    sel.config.benefit_eval_enabled = True
    sel.config.benefit_weight = 5.0
    on_before_warmup = sel.score_trajectory(traj)
    assert torch.allclose(off_score, on_before_warmup), (
        "benefit_eval_enabled=True before record_benefit_sample() warmup "
        "must be bit-identical to OFF"
    )

    sel.record_benefit_sample(sel._BENEFIT_WARMUP_SAMPLES)
    on_after_warmup = sel.score_trajectory(traj)
    assert not torch.allclose(on_after_warmup, off_score), (
        "benefit_eval_enabled=True with the warmup gate cleared did not "
        "change score_trajectory's output (inert flag)"
    )


def test_use_sd039_anchor_payload_gates_build_goal_payload():
    """SD-039: HippocampalModule.build_goal_payload() returns None unless
    AnchorSetConfig.use_sd039_anchor_payload=True (in addition to
    use_anchor_sets=True on the module itself).
    """
    from types import SimpleNamespace
    from ree_core.utils.config import AnchorSetConfig

    mod_off = _hippo_module(
        use_anchor_sets=True, anchor_set=AnchorSetConfig(use_sd039_anchor_payload=False)
    )
    mod_on = _hippo_module(
        use_anchor_sets=True, anchor_set=AnchorSetConfig(use_sd039_anchor_payload=True)
    )
    latent_state = SimpleNamespace(z_world=torch.randn(1, 8))

    assert mod_off.build_goal_payload(latent_state) is None
    payload_on = mod_on.build_goal_payload(latent_state)
    assert payload_on is not None, (
        "use_sd039_anchor_payload=True with use_anchor_sets=True did not "
        "build a payload (inert flag)"
    )


def test_use_composite_cue_outshining_gates_the_context_term():
    """MECH-339 Constraint 1: GhostGoalBank.rank() adds a fifth "context"
    component (gated open on a weak direct goal_match, scaled by
    arousal_tag) only when use_composite_cue_outshining=True. Master OFF
    carries no "context" key at all; master ON with the context_weight=0.0
    default carries the key but pinned at exactly 0.0 (the documented
    no-op-default contract); master ON with a real context_weight and a
    weak-match anchor produces a nonzero term that changes ghost_priority.
    Direct AnchorSet/GhostGoalBank construction mirrors
    test_mech340_persistence_efficacy_gate.py's `_anchor`/`_bank` helpers.
    """
    from ree_core.hippocampal.anchor_set import Anchor, AnchorGoalPayload, AnchorSet
    from ree_core.hippocampal.ghost_goal_bank import GhostGoalBank, GhostGoalBankConfig
    from ree_core.utils.config import AnchorSetConfig

    def _anchor(zsnap, arousal_tag):
        a = Anchor(key=("fast", "A", ("s",)), z_world=torch.zeros(4), active=False)
        a.goal_payload = AnchorGoalPayload(
            z_goal_snapshot=torch.tensor(zsnap, dtype=torch.float32).unsqueeze(0),
            wanting_strength=0.3,
            arousal_tag=arousal_tag,
            last_vs=0.6,
        )
        return a

    def _bank(anchor, cfg):
        s = AnchorSet(AnchorSetConfig())
        s._all = {anchor.key: anchor}
        return GhostGoalBank(cfg, s)

    z_goal = torch.tensor([1.0, 0.0, 0.0, 0.0])
    anchor_weak_match = _anchor([0.2, 1.0, 0.0, 0.0], arousal_tag=0.9)  # clears goal_match_floor, well below outshine_pivot

    off = _bank(anchor_weak_match, GhostGoalBankConfig()).rank(z_goal)
    assert len(off) == 1
    assert "context" not in off[0].components

    on_zero_weight = _bank(
        anchor_weak_match, GhostGoalBankConfig(use_composite_cue_outshining=True)
    ).rank(z_goal)
    assert on_zero_weight[0].components.get("context") == 0.0, (
        "context_weight defaults to 0.0; a bare master-flag-on must still "
        "carry an exactly-zero context term (no-op-default contract)"
    )

    on = _bank(
        anchor_weak_match,
        GhostGoalBankConfig(use_composite_cue_outshining=True, context_weight=1.0),
    ).rank(z_goal)
    assert on[0].components.get("context", 0.0) > 0.0, (
        "use_composite_cue_outshining=True with context_weight>0 and a weak "
        "direct goal_match did not add a nonzero context term (inert flag)"
    )
    assert on[0].ghost_priority != off[0].ghost_priority


def test_use_cue_recall_gates_cue_recall_wanting():
    """SD-057 L6 (MECH-347): REEAgent.cue_recall_wanting() returns exactly
    0.0 unless GoalConfig.use_cue_recall=True -- even with an active
    IncentiveTokenBank carrying a real, matching, positive-value token
    (use_incentive_token_bank=True on BOTH arms, isolating use_cue_recall as
    the only variable).
    """
    from ree_core.utils.config import REEConfig
    from ree_core.agent import REEAgent

    def _agent(use_cue_recall: bool) -> REEAgent:
        cfg = REEConfig.from_dims(body_obs_dim=10, world_obs_dim=54, action_dim=4)
        cfg.goal.z_goal_enabled = True
        cfg.goal.use_incentive_token_bank = True
        cfg.goal.use_cue_recall = use_cue_recall
        return REEAgent(cfg)

    ag_off = _agent(use_cue_recall=False)
    ag_on = _agent(use_cue_recall=True)

    for ag in (ag_off, ag_on):
        gs = ag.goal_state
        assert gs.incentive_bank is not None
        z_obj = torch.randn(1, gs.config.goal_dim)
        gs.incentive_bank.update(resource_type=1, benefit=1.0, z_object=z_obj)

    off_strength = ag_off.cue_recall_wanting(cue_type=1, drive_level=1.0)
    on_strength = ag_on.cue_recall_wanting(cue_type=1, drive_level=1.0)
    assert off_strength == 0.0, (
        "use_cue_recall=False must return exactly 0.0 even with a matching "
        "token in an active incentive bank"
    )
    assert on_strength > 0.0, (
        "use_cue_recall=True with a matching token did not return a "
        "positive pull strength (inert flag)"
    )


# --------------------------------------------------------------------------- #
# ARC-110 loop-segregation cluster (2026-08-11 nested-scan individual audit,  #
# batch: E3Config loop-segregation). Each probe below was found by a          #
# computational search (not hand-derived), then pinned as literal tensors --  #
# a hand-derived "one outlier vs identical others" vector pair almost always  #
# lands on an exact zscore-symmetric tie (verified while building these),     #
# which is a worse test than a searched, robustly-margined divergence.       #
# --------------------------------------------------------------------------- #

def test_use_loop_segregation_changes_the_committed_index():
    """ARC-110 master switch: _segregated_loop_arbitrate (per-loop zscore-
    normalize then Haber-spiral-combine) REPLACES the flat single-arena
    within-eligible argmin when ON. A loop whose RAW magnitude dominates the
    flat OFF sum need not dominate once each loop is normalized to unit
    variance before combining (the "F's raw magnitude carries no cross-loop
    advantage" conversion mechanism _loop_normalize's own docstring names) --
    so the two arbitration algorithms can commit to DIFFERENT candidates.

    The prior KNOWN_UNPROBED_NESTED reason for this flag noted a real risk:
    V3-EXQ-707b's live-run entropy comparison was nearly flat, so a naively
    constructed toy fixture could pass by accident without exercising real
    divergence. These exact vectors were found by search (trial 0 of a
    dacc~N(0,50)/ofc~N(0,3) sweep) specifically because they produce a clean,
    non-tied flip via the real select() call (not a hand-picked toy).
    """
    from ree_core.utils.config import REEConfig
    from ree_core.predictors.e2_fast import Trajectory
    from ree_core.predictors.e3_selector import E3TrajectorySelector

    n = 6
    world_dim = 6

    def _candidate(action_class, action_dim=8):
        horizon = 3
        states = [torch.zeros(1, world_dim) for _ in range(horizon + 1)]
        world_states = [torch.zeros(1, world_dim) for _ in range(horizon + 1)]
        actions = torch.zeros(1, horizon, action_dim)
        actions[:, 0, action_class % action_dim] = 1.0
        return Trajectory(states=states, actions=actions, world_states=world_states)

    def _patch_raw(selector, candidates, raw_costs):
        raw_map = {id(c): torch.tensor([float(v)]) for c, v in zip(candidates, raw_costs)}
        selector.score_trajectory = lambda cand, **kw: raw_map[id(cand)]

    def _selector(use_loop_segregation):
        cfg = REEConfig.from_dims(
            body_obs_dim=8, world_obs_dim=8, action_dim=8, self_dim=16, world_dim=16,
            use_loop_segregation=use_loop_segregation,
            use_finer_channel_gating=True,
            use_modulatory_shortlist_then_modulate=True,
        )
        sel = E3TrajectorySelector(cfg.e3, None)
        sel._running_variance = 0.0  # force the committed (deterministic) path
        return sel

    cands = [_candidate(i) for i in range(n)]
    raw = [0.0] * n  # flat motor F -> the modulatory channels decide

    dacc = torch.tensor([
        33.06760787963867, 13.346205711364746, 3.0838630199432373,
        31.065866470336914, -22.595298767089844, -8.306511878967285,
    ])  # associative-default channel
    ofc = torch.tensor([
        -4.568305492401123, 1.1450517177581787, -3.0828258991241455,
        -1.689158320426941, -2.6768715381622314, -0.17475053668022156,
    ])  # limbic-default channel

    results = {}
    for flag in (False, True):
        sel = _selector(flag)
        _patch_raw(sel, cands, raw)
        r = sel.select(
            cands,
            score_bias=torch.zeros(n),
            score_bias_channels={"dacc": dacc, "ofc": ofc},
        )
        results[flag] = r.selected_index

    assert results[False] != results[True], (
        "use_loop_segregation=True's per-loop-normalized arbitration must be "
        "able to pick a DIFFERENT committed candidate than the flat "
        "single-arena OFF sum (inert flag / not a genuine mechanism)"
    )


def test_use_named_channel_routing_substitutes_the_routed_representation():
    """ARC-110 C2 release: the real select()-level gate for the loop-
    arbitration override requires use_finer_channel_gating AND
    use_named_channel_routing AND a real score_bias_channel_routed entry --
    all three, exactly as select()'s own composition site requires. The
    prior KNOWN_UNPROBED_NESTED reason noted the one existing test
    (TestNamedChannelRoutingC2Release) calls _segregated_loop_arbitrate
    DIRECTLY with an explicit override, bypassing this three-way gate
    entirely -- so this probe instead drives the real select() call, with an
    intentionally FLAT unrouted "ofc" representation (score_bias_channels)
    standing in for the untrained project_channel_range zeros the prior note
    described, and a real-range routed representation
    (score_bias_channel_routed) supplied unconditionally by the caller in
    BOTH arms -- only the flag differs.
    """
    from ree_core.utils.config import REEConfig
    from ree_core.predictors.e2_fast import Trajectory
    from ree_core.predictors.e3_selector import E3TrajectorySelector

    n = 6
    world_dim = 6

    def _candidate(action_class, action_dim=8):
        horizon = 3
        states = [torch.zeros(1, world_dim) for _ in range(horizon + 1)]
        world_states = [torch.zeros(1, world_dim) for _ in range(horizon + 1)]
        actions = torch.zeros(1, horizon, action_dim)
        actions[:, 0, action_class % action_dim] = 1.0
        return Trajectory(states=states, actions=actions, world_states=world_states)

    def _patch_raw(selector, candidates, raw_costs):
        raw_map = {id(c): torch.tensor([float(v)]) for c, v in zip(candidates, raw_costs)}
        selector.score_trajectory = lambda cand, **kw: raw_map[id(cand)]

    def _selector(use_named_channel_routing):
        cfg = REEConfig.from_dims(
            body_obs_dim=8, world_obs_dim=8, action_dim=8, self_dim=16, world_dim=16,
            use_loop_segregation=True,
            use_finer_channel_gating=True,
            use_named_channel_routing=use_named_channel_routing,
            use_modulatory_shortlist_then_modulate=True,
        )
        sel = E3TrajectorySelector(cfg.e3, None)
        sel._running_variance = 0.0
        return sel

    cands = [_candidate(i) for i in range(n)]
    raw = [0.0] * n
    routed_ofc = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0, -20.0])  # strong preference for cand 5

    results = {}
    for flag in (False, True):
        sel = _selector(flag)
        _patch_raw(sel, cands, raw)
        r = sel.select(
            cands,
            score_bias=torch.zeros(n),
            score_bias_channels={"ofc": torch.zeros(n)},  # flat unrouted representation
            score_bias_channel_routed={"ofc": routed_ofc},  # real range, supplied regardless
        )
        results[flag] = r.selected_index

    assert results[False] != results[True], (
        "use_named_channel_routing=True did not substitute the routed "
        "representation for the flat unrouted one in loop arbitration "
        "(inert flag)"
    )
    assert results[True] == 5, (
        "use_named_channel_routing=True should commit to the routed "
        "channel's strongly-preferred candidate"
    )


def test_use_d1_d2_population_split_is_bit_identical_at_da_zero_then_diverges():
    """ARC-109: at da==0 (tanh of the freshly-constructed value baseline),
    _d1_d2_split's relu(accum)-relu(-accum) == accum exactly, so
    use_d1_d2_population_split=True must be bit-identical (same committed
    local index) to OFF. Once da moves away from 0 -- set directly on
    _lcg_value_baseline, bypassing the ARC-108 multi-tick learning loop this
    probe does not need -- the asymmetric D1/D2 gain must be able to change
    the committed index relative to that same da==0 baseline.
    """
    from ree_core.utils.config import REEConfig
    from ree_core.predictors.e3_selector import E3TrajectorySelector, _FCG_CHANNEL_INDEX

    def _selector(d1d2_flag):
        cfg = REEConfig.from_dims(
            body_obs_dim=8, world_obs_dim=8, action_dim=5, self_dim=32, world_dim=32,
            use_loop_segregation=True,
            use_d1_d2_population_split=d1d2_flag,
        )
        return E3TrajectorySelector(cfg.e3, None)

    n = 4
    elig = torch.arange(n)
    raw = torch.zeros(n)
    ofc = torch.tensor([-6.264134407043457, -1.78110933303833, 3.355997085571289, 0.849435031414032])
    dacc = torch.tensor([7.372593879699707, 3.526642322540283, -3.452648162841797, -2.195103645324707])
    terms = [(_FCG_CHANNEL_INDEX["ofc"], ofc), (_FCG_CHANNEL_INDEX["dacc"], dacc)]

    sel_off = _selector(False)
    loc_off = sel_off._segregated_loop_arbitrate(
        elig, raw, terms, True, [None] * n, True, 1.0, True
    )

    sel_on_da0 = _selector(True)
    assert sel_on_da0._lcg_value_baseline == 0.0
    loc_on_da0 = sel_on_da0._segregated_loop_arbitrate(
        elig, raw, terms, True, [None] * n, True, 1.0, True
    )
    assert loc_off == loc_on_da0, (
        "use_d1_d2_population_split=True at da==0 must be bit-identical to "
        "OFF (relu(accum)-relu(-accum) == accum exactly)"
    )

    sel_on_da = _selector(True)
    sel_on_da._lcg_value_baseline = 3.0  # da = tanh(3.0) ~= 0.995
    loc_on_da = sel_on_da._segregated_loop_arbitrate(
        elig, raw, terms, True, [None] * n, True, 1.0, True
    )
    assert loc_on_da != loc_on_da0, (
        "use_d1_d2_population_split=True with da moved away from 0 did not "
        "change the committed index relative to the da==0 baseline (inert "
        "flag / never earns its dissociation)"
    )


def test_use_loop_local_eligibility_traces_excludes_the_losing_loops_channel():
    """MECH-452: the eligibility trace recorded inside select() (feeding the
    next post_action_update's three-factor credit) normally credits EVERY
    channel that spoke, regardless of whether its loop's within-loop winner
    matched the committed action. With use_loop_local_eligibility_traces=True,
    a channel whose loop did NOT win gets EXCLUDED (zero credit) -- keeping
    DA credit assignment loop-local. Probed at the eligibility-trace level
    (immediately after select()) rather than the full multi-tick learned-
    weight trajectory the mechanism ultimately feeds, per the flag's own
    KNOWN_UNPROBED_NESTED reason that its effect "shows only in the learned
    weight trajectory" -- the eligibility trace IS the credit-assignment
    step that trajectory is built from, so a divergence there is direct
    evidence of the mechanism, not an indirect proxy for it.
    """
    from ree_core.utils.config import REEConfig
    from ree_core.predictors.e2_fast import Trajectory
    from ree_core.predictors.e3_selector import E3TrajectorySelector, _FCG_CHANNEL_INDEX

    n = 6
    world_dim = 6

    def _candidate(action_class, action_dim=8):
        horizon = 3
        states = [torch.zeros(1, world_dim) for _ in range(horizon + 1)]
        world_states = [torch.zeros(1, world_dim) for _ in range(horizon + 1)]
        actions = torch.zeros(1, horizon, action_dim)
        actions[:, 0, action_class % action_dim] = 1.0
        return Trajectory(states=states, actions=actions, world_states=world_states)

    def _patch_raw(selector, candidates, raw_costs):
        raw_map = {id(c): torch.tensor([float(v)]) for c, v in zip(candidates, raw_costs)}
        selector.score_trajectory = lambda cand, **kw: raw_map[id(cand)]

    def _selector(loop_local_flag):
        cfg = REEConfig.from_dims(
            body_obs_dim=8, world_obs_dim=8, action_dim=8, self_dim=16, world_dim=16,
            use_loop_segregation=True,
            use_finer_channel_gating=True,
            use_modulatory_shortlist_then_modulate=True,
            use_loop_local_eligibility_traces=loop_local_flag,
        )
        sel = E3TrajectorySelector(cfg.e3, None)
        sel._running_variance = 0.0
        return sel

    cands = [_candidate(i) for i in range(n)]
    raw = [0.0] * n
    # Associative ("dacc") wins the committed candidate; limbic ("ofc") speaks
    # (real pref range) but its own within-loop winner differs -- the
    # search-found scenario where _loop_voted == {associative: True, limbic: False}.
    dacc = torch.tensor([
        100.53577423095703, 26.137012481689453, 14.004121780395508,
        -48.0182991027832, -73.77421569824219, -30.43423080444336,
    ])
    ofc = torch.tensor([
        -4.7723002433776855, 0.09279485046863556, -10.92197322845459,
        1.0217534303665161, -2.833395481109619, 0.7296481132507324,
    ])
    ofc_idx = _FCG_CHANNEL_INDEX["ofc"]

    sel_off = _selector(False)
    _patch_raw(sel_off, cands, raw)
    sel_off.select(cands, score_bias=torch.zeros(n), score_bias_channels={"dacc": dacc, "ofc": ofc})
    assert not sel_off._loop_voted["limbic"], "fixture assumption: limbic must lose this round"
    off_ofc_credit = float(sel_off._fcg_elig_trace[ofc_idx].item())

    sel_on = _selector(True)
    _patch_raw(sel_on, cands, raw)
    sel_on.select(cands, score_bias=torch.zeros(n), score_bias_channels={"dacc": dacc, "ofc": ofc})
    on_ofc_credit = float(sel_on._fcg_elig_trace[ofc_idx].item())

    assert off_ofc_credit > 0.0, (
        "fixture assumption: the OFF arm must credit the losing loop's "
        "channel (that is the ARC-108/MECH-451 baseline this flag changes)"
    )
    assert on_ofc_credit == 0.0, (
        "use_loop_local_eligibility_traces=True did not exclude the losing "
        "loop's channel from the eligibility trace (inert flag)"
    )


# --------------------------------------------------------------------------- #
# Flag registry-drift guard                                                   #
# --------------------------------------------------------------------------- #

# Flags with a behavioural probe in this file (asserting ON changes an observable).
PROBED = {
    # SD-RESIDUE-VALENCE-BOUND (ResidueConfig, not REEConfig top-level -- covered by
    # the nested-config scan below). Probed by
    # test_sd_residue_valence_bound_bounds_the_accumulator: OFF is bit-identical to
    # the pre-fix unclamped `+=` (exactly n_writes after n_writes unit increments),
    # ON bounds the same accumulator near valence_clamp_abs.
    "valence_bounding_enabled",
    "use_amygdala_analog",  # F-P1 probe drives BLAAnalog encoding_gain
    "use_bla_analog",       #   (gated by use_amygdala_analog; default True)
    "dacc_saturation_enabled",  # F-C3 wiring spy
    "use_phasic_burst",  # SD-069 fires-and-propagates probe (instantaneous_pe)
    "sws_enabled",  # SD-017 schema pass: writes -> E1 ContextMemory
    "rem_enabled",  # SD-017 attribution pass: rollouts -> HippocampalModule.replay
    # MECH-122 content-packaging half (V3 proxy, IGW-20260801-197). Probed by
    # test_mech122_spindle_content_selection_fires_and_differentiates_writes:
    # OFF reports zero selection weight/applied and is unchanged from plain
    # sws_enabled=True; ON reports an in-range weight AND produces a
    # different post-pass ContextMemory tensor than OFF on the same buffered
    # content -- stronger than "fires", since V3-EXQ-246's naive single
    # post-hoc consolidation_summary() write measured zero effect.
    "use_mech122_spindle_content_selection",
    # ARC-071 chunking. Probed by tests/contracts/test_arc071_policy_chunking.py
    # (C1 OFF-is-inert / C6 accumulator fires / C7 formation-only dissociation /
    # C9 proposer injection), not by a probe in this file.
    "use_policy_chunking",
    "use_chunk_maintenance",
    # MECH-324 dissolution-with-retention (Barnes 2005 / Bouton 2012). Probed by
    # the same file's C10 block: OFF pins DISSOLVED as an absorbing tombstone,
    # ON pins re-formation below R_min. Its precondition on
    # use_chunk_maintenance is enforced in PolicyChunkingConfig.validate() and
    # asserted there rather than in FLAGS_WITH_LOUD_PRECONDITION below, because
    # that harness routes through the tiny-fixture REEAgent build, which never
    # constructs the chunking operator unless use_policy_chunking is also on --
    # so this flag alone would not reach the raise.
    "use_chunk_dissolution_retention",
    # MECH-324 reacquisition-window isolation (data-flow fix for V3-EXQ-829's
    # confirmed flat r_reacq signature). Probed by the same file's C10 block:
    # OFF reproduces the bug (reformed-after == window_trials, contaminated
    # whole-lifetime tally), ON pins reformed-after == the reduced bar via an
    # isolated post-dissolution window, plus a numerical-floor case
    # (len(window) < 2 refuses even when the repetition bar is cleared). Its
    # precondition on use_chunk_dissolution_retention is likewise asserted in
    # PolicyChunkingConfig.validate() rather than here, for the same
    # tiny-fixture-never-reaches-the-raise reason as that flag.
    "use_reacquisition_window_isolation",
    # MECH-323 growable chunk-size ceiling (Ramkumar 2016 / Bo 2009). Probed by
    # the same file's C11 block: OFF pins the effective ceiling at
    # max_chunk_size with zero growths, ON grows it end-to-end when a merge
    # delivers a realised marginal return. Not probed here because the
    # behaviour needs a driver in which the full-length context predicts and
    # both one-element-shorter contexts are aliased -- a regime the tiny
    # fixture in this file cannot produce.
    "use_growable_chunk_ceiling",
    # ARC-071 growable chunk-DEPTH ceiling (Solway 2014). Probed by the same
    # file's C12 block: OFF pins the effective depth at max_depth with zero
    # growths, ON grows it end-to-end and mints a depth-4 chunk the OFF arm
    # refuses. Not probed here for the same reason as the size ceiling -- the
    # driver needs a nested suffix chain in which each further level pays.
    "use_growable_chunk_depth",
    # MECH-323 credit rule (V3-EXQ-810 readiness FAIL). Probed by the same
    # file's C13 block: OFF is asserted bit-identical against a hand-recomputed
    # trailing-only tally, ON is pinned to credit non-trailing sub-sequences,
    # to dedup a key to one credit per outcome, and to move _was_executed with
    # it. Not probed here because the OFF-vs-ON difference is a TALLY-SHAPE
    # difference over a multi-step episode buffer, which the single-step tiny
    # fixture in this file cannot produce.
    "use_chunk_all_position_credit",
    "use_chunk_replay_origin_path",
    "use_chunk_proposal_injection",
    # ARC-070 decomposition (MECH-321). Probed by
    # tests/contracts/test_arc070_policy_decomposition.py (C1 OFF-is-inert /
    # C2 loud precondition / C6 R1 OR-trigger / C7 depth-cap / C8 pre-commit
    # withhold-and-replace / C9 mid-execution latch release).
    "use_policy_decomposition",
    # MECH-321 scale-resolved rollout boundary probe (scoping spike 2026-07-27
    # section 5b). Probed by the same file's C17 block: OFF pins the rollout
    # latent_signature to exactly {z_world, z_self} (so MECH-288's slow BOCPD
    # scale stays structurally dead on the rollout stream, as shipped), ON pins
    # z_goal present. Deliberately NOT a pure diagnostic -- with z_goal there
    # the slow scale can contribute to boundary.fired and so change decisions,
    # which is the whole reason it is flagged rather than always-on.
    "use_decomposition_scale_resolved_probe",
    # The MID-EXECUTION half of the same probe (R4 second phase, the hook in
    # agent.py:select_action rather than hippocampal/module.py). Probed by
    # test_mech321_scale_resolved_boundary.py C18h-k: OFF pins the
    # mid-execution signature to exactly {z_world, z_self}, ON pins z_goal
    # present, and C18k pins that the two probe flags are INDEPENDENT in both
    # directions. Same not-a-pure-diagnostic caveat as its pre-commitment
    # sibling, one step sharper: a mid-execution fire releases the commit
    # latch, aborting the remaining macro.
    "use_decomposition_scale_resolved_probe_midexec",
    # SD-084 persistent committed-program handle. Probed by
    # test_mech321_midexec_natural_reachability.py: ON, the MECH-321 R4
    # mid-execution hook fires in a REAL rollout (decomp_n_evaluated_midexec >
    # 0) with nothing injected; OFF, the same loop yields exactly 0 WHILE STILL
    # committing multi-action programs -- so the zero is post_action_update's
    # unconditional teardown, not an empty arm. That pairing is the probe: it
    # is the flag, not the absence of committed programs, that moves the
    # counter. Note this flag is what makes the sibling
    # use_decomposition_scale_resolved_probe_midexec above REACHABLE at all --
    # until SD-084 landed, that flag modified a dict inside a block that never
    # executed in any experiment (V3-EXQ-830: 0 mid-execution evaluations in
    # all 10 cells). NOT a pure diagnostic: a reachable mid-execution hook can
    # newly reach boundary.fired and its fire releases the commit latch,
    # aborting the remaining macro.
    "use_persistent_committed_program_handle",
    # ARC-071/MECH-090 E3-tick reselection short-circuit (diagnostic_
    # arc071_e3_reselection_probe_2026-08-01.md; substrate_queue.json
    # arc071_e3_reselection_on_committed_program). Probed by
    # test_arc071_e3_reselection_shortcircuit.py: ON, a real rollout committed
    # to an unexpired arc071_chunk program keeps the SAME persistent
    # trajectory object across forced E3 ticks and _committed_step_idx
    # advances to chunk_length - 1 with zero premature reselections at
    # chunk_max_size 5 and 15; OFF, the identical forced-tick schedule
    # reproduces the diagnosed defect exactly (a distinct trajectory
    # installed on effectively every forced tick, step_idx never advancing
    # past its immediate post-commit value). A fifth test in that file pins
    # that MECH-091's urgency-interrupt release is never swallowed by the
    # short-circuit (via the _ncl_mech091_fired flag, since a fresh
    # re-commit can legitimately re-elevate beta_gate later in the same
    # tick). Requires use_persistent_committed_program_handle=True to have
    # anything to check -- inert without it, by construction.
    "use_e3_reselection_shortcircuit",
    # SD-091/MECH-481 claustrum coalition controller. Probed by
    # test_use_coalition_controller_is_inert_until_requested_then_moves_actions
    # above: the flag alone is bit-identical to OFF *by design* (it only
    # constructs the controller; nothing calls request_coalition()), so the probe
    # asserts BOTH halves -- inert-when-unrequested, and a different action stream
    # once a SENSORY_RESAMPLE coalition is actually open. Wiring-level coverage
    # (W1-W8: default-OFF, reset(), per-template target tables, the BetaGate
    # monotonicity guardrail) lives in
    # tests/contracts/test_sd091_coalition_controller_wiring.py.
    "use_coalition_controller",
    # The types tuple on the same controller -- NOT a boolean switch, it is which
    # ControlDemandType templates may open. Probed by
    # test_coalition_types_enabled_gates_which_templates_can_open: a listed type
    # opens and pulls its write gates off 1.0, an unlisted one is refused
    # (unregistered_request_count increments) with every gate left at baseline.
    # Registered separately from the master flag because it is the knob that
    # could silently WIDEN the coalition surface -- if the tuple were dropped on
    # the floor, CoalitionControllerConfig's default is every template, so the
    # failure mode is permissive, not inert, and the master flag's probe would
    # still pass.
    "coalition_types_enabled",
    # SD-014 incentive sensitization (V3-EXQ-887 decouple fix, 2026-08-07).
    # Probed by test_incentive_sensitization_amplifies_the_wanting_write: OFF
    # pins the per-node gain at exactly 0.0 (path not entered), ON accumulates a
    # drive-coupled gain and lifts the VALENCE_WANTING readout above the raw
    # salience an OFF arm writes. Its activating condition is three-part
    # (tonic_5ht_enabled, >=1 active RBF center, drive_level > 0) -- a zero-drive
    # arm is legitimately bit-identical, which is exactly the shape that would
    # otherwise get mis-read as a dead flag.
    "incentive_sensitization_enabled",
    # F-C4 (LatentStackConfig, not REEConfig top-level -- covered by the
    # nested-config scan). Probed by
    # test_fc4_iterative_inference_settles_and_refuses_the_inert_config above:
    # settle_iters<2 with the flag ON refuses to build (naming both knobs),
    # settle_iters>=2 produces a non-empty, non-NaN convergence readout. Was
    # bulk-seeded into KNOWN_UNPROBED_NESTED by the 2026-08-11 scanner
    # widening despite already being probed in this same file -- moved here
    # during the follow-on audit of that bulk seed (chip
    # chip-20260811-flaginertness-nested-audit).
    "use_iterative_inference",
    # --- Nested-config individual audit (chip chip-20260811-flaginertness-      #
    # nested-audit), batch: E3Config -----------------------------------------#
    # SD-076 waking-confidence inflation. Probed by
    # test_sd076_rv_floor_headroom.py: OFF is bit-identical to the pre-repair
    # symmetric EMA; ON drives `_running_variance` measurably below both true
    # error and the OFF-equivalent `_wci_symmetric_rv_ref` counterfactual,
    # with a genuine LO/HI asymmetry dose-response.
    "use_waking_confidence_inflation",
    # SD-063 conditional predictive-precision commit gate. Probed by
    # test_sd063_conditional_uncertainty_head.py: OFF ignores
    # `conditional_predictive_variance` entirely (EMA-only commit decision);
    # ON lets it override the EMA in BOTH directions (veto a low-EMA commit,
    # rescue a high-EMA non-commit) on the same `_running_variance` state.
    "use_conditional_precision_gate",
    # ARC-108 learned channel gating. Probed by
    # test_arc108_learned_channel_gating.py C2/C3: ON-at-init is bit-identical
    # to OFF (exact score/selected_index equality); under a non-flat
    # realised-outcome delta_t sequence `w_chan` moves off its softplus-unity
    # init only when ON, never when OFF.
    "use_learned_channel_gating",
    # MECH-451 finer-channel gating. Probed by
    # test_mech451_finer_channel_gating.py C2/C3: ON-at-init reproduces the
    # legacy compressed blend exactly; under a non-flat delta_t sequence
    # `w_chan_finer` moves off init only when ON (ARC-108's global `w_chan`
    # buffer never moves under this path), stays frozen when OFF.
    "use_finer_channel_gating",
    # ARC-108/MECH-450 learned settling step. Probed by
    # test_mech450_learned_settling_step.py C3: over an alternating good/bad
    # realised-outcome sequence, `W_lat` accrues nonzero entries only when ON;
    # stays exactly zero when OFF.
    "use_learned_settling_step",
    # MECH-140/450 soft-competitive settling. Probed by
    # test_soft_competitive_settling.py test_settling_flips_the_committed_winner:
    # with identical raw F costs, OFF picks the argmin-best-bias candidate in
    # a crowded action class; ON (gain>0) suppresses the crowded cluster and
    # FLIPS the committed selected_index to an isolated, worse-biased
    # candidate -- the committed action itself differs, not just an internal
    # metric.
    "use_soft_competitive_settling",
    # ARC-108xARC-110 learned cross-loop arbitration. Probed by
    # test_learned_cross_loop_arbitration.py: OFF is bit-identical to
    # ON-at-init across seeds (W_cross==I); with the same flag ON but a
    # limbic-boosted M_cross, the committed index flips from the motor(F)
    # winner to the limbic winner on an input where OFF/ON-at-init both
    # commit to the motor candidate -- a genuine, chained ON!=OFF divergence.
    "use_learned_cross_loop_arbitration",
    # ARC-110 ascending-spiral gain. Probed by test_ascending_spiral_gain.py
    # TestLimbicCanNowWin: a fixed 709-signature M_cross does not let limbic
    # override motor at gain 1.0 but does at gain 30.0 (commit flips), while
    # TestByteIdenticalOff pins the flag OFF as bit-identical to the plain
    # learned combine.
    "use_ascending_spiral_gain",
    # V3-EXQ-711 bounded parity controller. Probed by
    # test_ascending_parity_controller.py: TestByteIdenticalOff/TestInertOn
    # pin OFF and no-op sub-knobs bit-identical; TestParityCeilingBounds shows
    # the controller holds w_eff[limbic]/w_eff[motor] at/under the ceiling
    # where the raw scalar (use_ascending_spiral_gain) runs away;
    # TestParityWinNotMonopoly shows limbic actually reaches parity.
    "use_ascending_parity_controller",
    # 569f/661/654a shortlist-then-modulate conversion lever. Probed by
    # test_e3_score_bias_candidate_support.py: OFF is bit-identical to the
    # pre-amend authority path; ON restricts the eligible set to an
    # F-near-tie/top-k shortlist so a clearly-worse-by-F candidate is refused
    # even under an overwhelming modulatory pull, and among F-tied candidates
    # the modulatory argmin alone decides the winner.
    "use_modulatory_shortlist_then_modulate",
    # MECH-439 Factor B gap-scaled commit temperature. Probed by
    # test_e3_conflict_graded_conversion.py: OFF is the legacy hard argmin;
    # ON softens the committed pick near ties (spreads across candidates over
    # seeds) and recovers a decisive argmin at a large F-gap, with T_eff shown
    # load-bearing on the (1-gap_norm) scaling.
    "use_gap_scaled_commit_temperature",
    # MECH-448/ARC-107 F-eligibility demotion. Probed by
    # test_mech_448_f_eligibility_demotion.py: OFF is the legacy
    # argmin(F+bias); ON excludes a clearly-harmful-by-F candidate from the
    # eligible set even under an overwhelming modulatory pull, and the
    # within-eligible winner is the modulatory argmin with F removed.
    "use_f_eligibility_demotion",
    # MECH-448 channel-adaptive envelope amend. Probed by
    # test_mech_448_f_eligibility_demotion.py
    # test_adaptive_excludes_across_two_differing_scale_distributions: on the
    # SAME near-uniform F-share distribution, the fixed 0.30 floor (flag OFF)
    # all-admits (excluded_count==0), while the adaptive floor (flag ON)
    # productively excludes on that distribution and a second,
    # differently-scaled one.
    "use_f_eligibility_adaptive_floor",
    # MECH-449/ARC-107 Go/No-Go eligibility constitution. Probed by
    # test_mech_449_go_nogo_constitution.py: OFF ignores go_nogo_signals
    # entirely; ON a No-Go signal drops the modulatory-favoured candidate from
    # eligibility (holds under an overwhelming modulatory pull).
    "use_go_nogo_constitution",
    # DR-12 (self_model_v4:SELF-4) E2-forward-PE confidence down-weight.
    # Probed by test_dr12_pe_confidence.py
    # test_c2_high_pe_on_primary_best_flips_selection: OFF picks the
    # primary-favoured candidate; ON with a high per-candidate PE on that same
    # candidate flips the committed argmin away from it.
    "use_pe_confidence_weighting",
    # DR-10 (self_model_v4:SELF-3) z_self-derived viability cost. Probed by
    # test_dr10_z_self_viability.py
    # test_c2_differential_self_viability_flips_selection: a decisive
    # per-candidate viability penalty placed on the OFF arm's own winner flips
    # the committed argmin away from it.
    "use_self_viability_weighting",
    # SD-081/MECH-477 dual-system uncertainty arbitration. Probed by
    # test_sd081_dualsystem_arbitration.py: OFF reproduces a deterministic
    # action stream and never writes last_arbitration; ON changes the
    # committed action stream with a live, non-degenerate arbitration weight
    # that is monotone in relative uncertainty, fixing the V3-EXQ-786a flat-
    # recruitment signature.
    "use_dualsystem_arbitration",
    # --- batch: GoalConfig / SerotoninConfig ---------------------------------#
    # SD-092 cross-level subgoal credit. OFF pins _z_goal_parent at None and
    # every credit call at a true {} no-op; ON bootstraps a material,
    # direction-aligned parent attractor from repeated credit events against a
    # decay-only control that stays at exactly 0.0. See
    # tests/contracts/test_sd092_cross_level_subgoal_credit.py (R1/R3/R4).
    "use_hierarchical_goal_credit",
    # SD-057 object-bound incentive-salience bank. OFF: gs.incentive_bank is
    # None, legacy single-attractor seeding runs unchanged
    # (test_sd_057_incentive_token_bank.py test_c1_default_off_...). The
    # bank's most_wanted()/wanting() arithmetic that the live agent-level
    # z_goal-seed redirect (agent.py update_z_goal, unconditional once the
    # bank is non-None) reads is itself probed by test_c3_l3_per_axis_wanting_
    # is_drive_specific and test_sd049_phase2_drive_coupling.py; a real live
    # agent forming Stage-0 bank entries is covered by
    # test_scaffolded_sd054_onboarding.py test_c9_stage0_binding_populates_bank.
    # No test isolates a full-agent ON-vs-OFF z_goal-seed-source A/B directly.
    "use_incentive_token_bank",
    # SD-093/MECH-426 progress-velocity effort modulation. Probed by
    # test_mech426_progress_velocity.py: OFF (test_c2_flag_off_is_true_noop)
    # is a true no-op -- record_progress always returns 0.0, history stays
    # empty; ON reaches the real E3TrajectorySelector.select() committed
    # boolean and flips it relative to OFF at an identical _running_variance
    # (test_c8_stalling_raises_threshold_commits_more_readily /
    # test_c8_coasting_lowers_threshold_commits_less_readily /
    # test_c8_flag_off_leaves_threshold_at_baseline).
    "use_progress_velocity_effort_modulation",
    # MECH-189 super-ordinal goal anchors. Probed by
    # test_mech_189_super_ordinal_goal_anchors.py: OFF pins
    # agent.super_ordinal_goal_memory at None (test_c1_default_off); ON forms
    # a real anchor from a live update_z_goal() call in a high-salience novel
    # context (test_c7_agent_write_forms_anchor) and, strongest of all, a
    # later zero-benefit tick in the same context pulls z_goal toward the
    # stored anchor after writes are frozen and z_goal reset sub-floor
    # (test_c8_agent_read_seeds_zgoal, cosine > 0.5) -- a full write+read
    # round trip reaching z_goal itself.
    "use_super_ordinal_goal_anchors",
    # GoalConfig's own master construction gate -- REEAgent.goal_state is None
    # unless this is set (agent.py). Probed in isolation by
    # test_z_goal_stream_counter.py
    # test_z3_goal_disabled_counts_nothing_and_frac_is_none (OFF:
    # goal_state is None, z_goal_active_frac is None, not 0.0); ON is
    # evidenced by the entire SD-092/093/189/057 contract-test family, all of
    # which construct z_goal_enabled=True and show real divergence downstream
    # of a non-None goal_state. (The already-PROBED top-level
    # goal_stream_enabled sets this as the first line of its own
    # enable_goal_stream() helper, so that flag's own probe exercises this
    # switch too, as a bundle rather than in isolation.)
    "z_goal_enabled",
    # SR-1/SR-2 tonic-5HT master switch (SerotoninModule). Probed by
    # test_mech203_harm_salience_writepath.py: OFF pins a live agent's
    # update_harm_salience() call at exactly 0.0 on VALENCE_HARM_DISCRIMINATIVE
    # before and after (test_c4_default_off_is_bit_identical); ON writes a
    # non-trivial value on the identical call
    # (test_c2_live_path_populates_harm_discriminative), and
    # test_c1_module_surface_and_arithmetic pins the module-level
    # harm_salience() arithmetic itself (0.0 OFF, computed from
    # tonic_5ht_baseline ON). Cleaner isolation than this file's own
    # test_incentive_sensitization_amplifies_the_wanting_write, which only
    # uses this flag as one of three co-required activating conditions.
    "tonic_5ht_enabled",
    # MECH-440 NoisyNet propagating selection-head weight noise. Probed by
    # test_use_noisy_selection_head_bias_is_nonzero_only_when_enabled above:
    # OFF and ON-with-sigma_init=0.0 both produce exactly zero bias range
    # (bit-identical no-op contract); ON with sigma_init>0.0 produces a
    # nonzero bias range reaching last_score_diagnostics.
    "use_noisy_selection_head",
    # MECH-441 E2-forward-model-disagreement curiosity bonus. Probed by
    # test_use_model_disagreement_curiosity_bonus_flips_selection above: a
    # decisive per-candidate disagreement bonus on a non-primary candidate
    # flips the committed argmin away from the primary-favoured winner.
    "use_model_disagreement_curiosity",
    # --- batch: HippocampalConfig ---------------------------------------------#
    # V3-EXQ-553 orthogonal CEM seeding. Probed by test_orthogonal_cem_seeding.py
    # C4: same-seed ARM_ORTHO min pairwise-L2 among CEM candidates exceeds
    # ARM_IID baseline (variance reduction in worst-case distinguishability).
    "use_orthogonal_cem_seeding",
    # Diagnostic candidate-support scaffold. Probed by
    # test_hippocampal_candidate_support.py: ON guarantees every action class
    # appears among first-step candidates; OFF has no such guarantee.
    "use_action_class_scaffold_candidates",
    # SP-CEM support-preserving repair. Probed by
    # test_hippocampal_candidate_support.py: under an artificially collapsed
    # decoder, ON repairs first-action class diversity (>=2 classes); OFF
    # leaves it collapsed to exactly 1 class.
    "use_support_preserving_cem",
    # SD-025 familiarity discount on curiosity novelty. Probed by
    # test_sd025_curiosity_drive.py C5: ON, repeat waking visits decay the
    # curiosity bonus (anti-perseveration); OFF, the bonus is pure density and
    # is bit-identical across visits.
    "use_curiosity_familiarity",
    # MECH-269 Phase 1 per-stream V_s. Probed by test_mech_269_per_stream_vs.py
    # C2/C3: OFF leaves per_stream_vs empty (no-op); ON seeds V_s=1.0 on first
    # observation and a perturbed z_world measurably drops it.
    "use_per_stream_vs",
    # MECH-288 hierarchical event segmenter. Probed by
    # test_mech_288_event_segmenter.py: C1 pins OFF -> module.event_segmenter
    # is None; C2/C3 drive the same EventSegmenter class ON, proving it fires
    # BoundaryEvents on a synthetic PE-spike/BOCPD changepoint and stays
    # silent on a constant baseline.
    "use_event_segmenter",
    # MECH-287 broadcast invalidation trigger. Probed by
    # test_mech_287_invalidation_trigger.py: C1 pins OFF ->
    # module.invalidation_trigger is None; C2 drives the same
    # InvalidationTrigger class ON, proving a BoundaryEvent produces a
    # BroadcastEvent with strength = posterior * gain.
    "use_invalidation_trigger",
    # MECH-269 Phase 2 anchor sets. Probed by test_mech_269_anchor_set.py: OFF,
    # tick_anchor_set is a pure no-op (anchor_set is None); ON, a queued
    # BoundaryEvent installs exactly one active anchor with the correct
    # (scale, segment_id, stream_mixture) key.
    "use_anchor_sets",
    # MECH-269 Phase 2(iii) per-region V_s. Probed by
    # test_mech_269_per_region_vs.py C1/C2: OFF, update_per_region_vs is a
    # no-op even with an active anchor and populated per_stream_vs
    # (per_region_vs stays {}); ON, the same setup populates a region-keyed
    # V_s entry.
    "use_per_region_vs",
    # MECH-269b symmetric V_s rollout gating. The VsRolloutGate class the
    # agent constructs when ON is probed by
    # test_mech_269b_vs_rollout_gate_staleness.py C2/C4 (gate() substitutes a
    # held snapshot only when effective V_s crosses threshold) and C8
    # (agent-level precondition raises, proving construction is really
    # reached, not silently skipped).
    "use_vs_rollout_gating",
    # MECH-269b + MECH-284 staleness-aware V_s gating. Probed by
    # test_mech_269b_vs_rollout_gate_staleness.py C2/C4: identical scenario
    # (raw V_s 0.9, staleness 0.7), OFF leaves the gate output unmodified (0
    # holds); ON substitutes the held snapshot (1 hold) because
    # effective_vs = raw_vs - staleness crosses threshold.
    "use_vs_gate_staleness_lookup",
    # MECH-292 ranked ghost-goal bank. Probed by test_mech_293_ghost_probes.py:
    # C3 proves construction is loud-gated (ValueError naming the flag when a
    # MECH-293 consumer needs it but it's off); C4 proves the bank ON produces
    # a real observable effect (>=1 ghost-tagged CEM candidate reaching
    # propose_trajectories' return value) via a seeded anchor payload.
    "use_mech292_ghost_bank",
    # MECH-293 waking ghost-goal probe search. Probed by
    # test_mech_293_ghost_probes.py C2/C4: OFF, zero ghost-tagged trajectories
    # and no mech293_* diagnostics; ON, >=1 CEM candidate is seeded from the
    # ranked bank, carrying hypothesis_tag=True and provenance metadata.
    "use_mech293_ghost_probes",
    # --- batch: LatentStackConfig / GhostGoalBankConfig -----------------------#
    # DR-13 self-recurrence. Probed by test_dr13_self_recurrence.py: OFF is
    # bit-identical (no module, no readout); ON, a varying observation
    # sequence produces nonzero state_departure in self_recurrence_diag,
    # proving the recurrence carries state the instantaneous encode does not.
    "use_self_recurrence",
    # SD-031 E2WorldForward. Probed by test_e2_world_forward.py: C1 pins
    # bit-identical OFF (agent.e2_world is None, action stream unchanged from
    # explicit-False); C4 pins the ON module is not an identity map and is
    # action-conditional; C6 pins the agent-level construction gate at
    # world_dim=128.
    "use_e2_world_forward",
    # MECH-340 persistence/efficacy gate. Probed by
    # test_mech340_persistence_efficacy_gate.py: OFF ignores any supplied
    # persistence_appraisal (ranks bit-identical to none-supplied); ON, the
    # identical low-control/high-unattainability appraisal that was ignored
    # OFF now excludes the anchor from the ranked bank entirely (rank()==[]).
    "use_persistence_efficacy_gate",
    # --- batch: E1Config / E2Config / HeartbeatConfig / ResidueConfig --------#
    # SD-016 frontal cue-indexed integration. Probed by tests/test_sd016.py
    # TestAgentSD016Wiring: a live e1_tick populates
    # agent._cue_action_bias/_cue_terrain_weight with the expected shapes when
    # ON, and both stay None when OFF.
    "sd016_enabled",
    # MECH-216 schema readout. Probed by
    # tests/contracts/test_step_harness_contract.py H6/H7:
    # update_schema_wanting is never called through StepHarness with the flag
    # OFF (default), and is called (before select_action) with it ON.
    "schema_wanting_enabled",
    # SD-056 rollout stability (V3-EXQ-569e amend). Probed by
    # test_sd_056_multistep_amend.py A9/A10: with a deliberately unstable
    # world_transition scale that would otherwise blow up the rollout, ON
    # enforces the B2 per-step norm bound end-to-end (A9) and prevents
    # NaN/Inf under stress (A10).
    "e2_rollout_output_norm_clamp_enabled",
    # SD-024/MECH-232 DA-modulated RBF density. Probed by
    # test_sd024_da_modulated_rbf_density.py C1 (OFF: no per-center bandwidth
    # buffer) / C3 (same reward events, ON produces strictly higher
    # compute_benefit_density than OFF).
    "use_da_modulated_rbf_density",
    # SD-100 (ARC-032/MECH-089) theta-phase-weighted ThetaBuffer.summary().
    # Probed by test_theta_buffer_phase_aware_summary.py: pushing the same
    # multiset of z_world vectors in original vs reversed order gives an
    # identical summary OFF (order-invariant flat mean) but a different
    # summary ON (order-sensitive phase kernel).
    "use_theta_phase_weighted_summary",
    # LatentStackConfig construction-gate cluster (2026-08-11 nested-scan
    # individual audit, batch: LatentStackConfig). Each probed by a direct
    # LatentStack (or REEAgent) construction-gate test above: OFF leaves the
    # gated module/output None, ON constructs it and populates the field.
    "use_harm_stream",              # test_use_harm_stream_populates_z_harm_only_when_enabled
    "use_affective_harm_stream",    # test_use_affective_harm_stream_populates_z_harm_a_only_when_enabled
    "use_harm_un",                  # test_use_harm_un_populates_z_harm_un_only_when_enabled
    "use_event_classifier",         # test_use_event_classifier_populates_event_logits_only_when_enabled
    "use_resource_proximity_head",  # test_use_resource_proximity_head_populates_resource_prox_pred_only_when_enabled
    "use_e2_harm_s_forward",        # test_use_e2_harm_s_forward_constructs_agent_e2_harm_s_only_when_enabled
    "use_e2_world_uncertainty",     # test_use_e2_world_uncertainty_constructs_agent_head_only_when_enabled
    "use_resource_encoder",         # test_use_resource_encoder_populates_z_resource_only_when_enabled
    "use_identity_classifier",      # test_use_identity_classifier_populates_identity_logits_only_when_enabled
    # E2Config/HeartbeatConfig/ResidueConfig cluster (same 2026-08-11 audit,
    # batch: E2Config/HeartbeatConfig/ResidueConfig). Each probed above.
    "benefit_terrain_enabled",       # test_benefit_terrain_enabled_gates_accumulate_and_evaluate_benefit
    "safety_terrain_enabled",        # test_safety_terrain_enabled_gates_accumulate_and_evaluate_safety
    "valence_enabled",               # test_valence_enabled_gates_update_and_evaluate_valence
    "ewc_enabled",                   # test_ewc_enabled_gates_ewc_penalty
    "cross_stream_binding_enabled",  # test_cross_stream_binding_enabled_couples_the_rollout_streams
    "use_commit_readiness_gate",     # test_use_commit_readiness_gate_blocks_elevation_below_the_floor
    # HippocampalConfig cluster (same 2026-08-11 audit, batch: HippocampalConfig).
    "mode_conditioning_enabled",     # test_mode_conditioning_enabled_gates_mode_weighted_cem_scales
    "use_staleness_accumulator",     # test_use_staleness_accumulator_gates_construction_and_integration
    "use_mech284_hysteresis",        # test_use_mech284_hysteresis_swaps_the_staleness_lookup_source
    "use_backward_credit_sweep",     # test_use_backward_credit_sweep_writes_retroactive_valence_credit
    "use_offline_wanting_spread",    # test_use_offline_wanting_spread_writes_decayed_credit_to_earlier_waypoints
    # AnchorSetConfig/GhostGoalBankConfig/GoalConfig + benefit_eval_enabled
    # cluster (same 2026-08-11 audit).
    "benefit_eval_enabled",          # test_benefit_eval_enabled_gates_score_trajectory_after_warmup
    "use_sd039_anchor_payload",      # test_use_sd039_anchor_payload_gates_build_goal_payload
    "use_composite_cue_outshining",  # test_use_composite_cue_outshining_gates_the_context_term
    "use_cue_recall",                # test_use_cue_recall_gates_cue_recall_wanting
    # ARC-110 loop-segregation cluster (same 2026-08-11 audit, batch:
    # E3Config loop-segregation). Each probed above by a computationally-
    # searched, non-tied divergence (not a hand-derived toy fixture).
    "use_loop_segregation",             # test_use_loop_segregation_changes_the_committed_index
    "use_named_channel_routing",        # test_use_named_channel_routing_substitutes_the_routed_representation
    "use_d1_d2_population_split",       # test_use_d1_d2_population_split_is_bit_identical_at_da_zero_then_diverges
    "use_loop_local_eligibility_traces",  # test_use_loop_local_eligibility_traces_excludes_the_losing_loops_channel
} | set(FLAGS_WITH_DEFAULT_BEHAVIOURAL_DELTA) | set(FLAGS_WITH_LOUD_PRECONDITION)

# Audit-confirmed inert / mis-wired flags (finding id -> reason). Documented here
# even when the concrete lever is not a top-level flag (dacc_foraging_weight is a
# float; use_iterative_inference lives under config.latent), so the record is in
# one place. See design_implementation_audit_2026-07-09.md.
KNOWN_INERT = {
    "use_trainable_escape_affordance_learner": "F-C1: truncates its own state "
    "vector; zero live exposure -- guard before first use",
    # F-N1 (2026-08-11 nested-scan individual audit, E2Config -- not top-level).
    # `world_forward_contrastive_loss()` (e2_fast.py:279-402) computes the
    # SD-056 InfoNCE loss unconditionally on every call -- it reads
    # simulation_mode/K/min_batch_classes but NEVER
    # `self.config.e2_action_contrastive_enabled`. Confirmed by direct read of
    # the method body and a repo-wide grep of ree_core/: the flag's only
    # occurrences are its dataclass declaration, from_dims() plumbing, and the
    # config.e2.<flag> = <flag> assignment -- zero conditional reads anywhere.
    # The loss method itself has zero callers inside ree_core/ (only
    # individual experiment driver scripts call it directly by hand). So the
    # flag does not gate anything: whether it is True or False, the method
    # behaves identically, and nothing on the live agent path calls the
    # method at all regardless. tests/contracts/test_sd_056_e2_action_contrastive.py
    # only asserts the flag round-trips through config, never that it changes
    # world_forward_contrastive_loss's behaviour.
    "e2_action_contrastive_enabled": "F-N1: never read anywhere in ree_core/ "
    "(world_forward_contrastive_loss computes unconditionally); the method "
    "has zero callers on the live agent path -- config-only, not wired to a "
    "gate. See the F-N1 block comment above for the full evidence trail.",
    # F-N2, same class of finding as F-N1 and same E2Config sibling pair.
    # `world_forward_contrastive_loss_multistep()` (e2_fast.py:404+) never
    # reads `self.config.e2_action_contrastive_multistep_enabled` either, and
    # has zero callers anywhere in ree_core/ (only
    # experiments/v3_exq_617_sd056_multistep_substrate_readiness.py calls it
    # directly). tests/contracts/test_sd_056_multistep_amend.py's A1/A2 tests
    # only check config round-trip, per their own docstrings, never a
    # behavioural gate.
    "e2_action_contrastive_multistep_enabled": "F-N2: never read anywhere in "
    "ree_core/ (world_forward_contrastive_loss_multistep computes "
    "unconditionally); zero callers on the live agent path -- config-only, "
    "not wired to a gate. See the F-N2 block comment above.",
    # F-C3 FIXED 2026-07-09: dacc_saturation_enabled now fed from the live path
    # (agent.py select_action tail calls DACC.record_outcome each waking tick +
    # the DACCConfig saturation knobs are propagated from REEConfig). Moved to
    # PROBED (the test_fc3 wiring spy). See design_implementation_audit_2026-07-09
    # F-C3 / section 6.
    # non-top-level, documented for completeness:
    # dacc_foraging_weight            F-C2 uniform scalar -> argmin-invariant
    # F-C4 FIXED 2026-07-27: latent.use_iterative_inference no longer tolerates
    # the inert settle_iters<2 combination (LatentStack.__init__ raises) and the
    # NaN final_rel_delta placeholder is gone. Probed by test_fc4_* above; see
    # design_implementation_audit_2026-07-09 F-C4.
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
    "harm_surprise_pe_enabled", "replay_diversity_enabled",
    "shy_enabled",
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
    # use_rem_precision_broadcast: MECH-204 Phase 7 Option B (accuracy-anchored
    # broadcast REM), landed in 8ac193d. Registered here rather than probed for
    # the same reason as its already-registered sibling
    # use_rem_precision_recalibration directly above -- the pair is the MECH-204
    # REM-precision family, and behavioural probes for it are pending that
    # cluster's own validation. Added by a THIRD-PARTY session (SD-024
    # benefit-terrain work) to unbreak trunk: 8ac193d landed the flag without
    # its registry entry, so test_flag_registry_is_current was red on main for
    # every session. The owning session should replace this with a real probe,
    # or a more specific reason, when Phase 7 validates.
    "use_rem_precision_broadcast",
    "use_salience_coordinator",
    "use_sd049_per_axis_consumer_cascade",
    "use_shared_harm_trunk", "use_simulation_mode_rule_gate",
    "use_sleep_aggregation_cluster", "use_sleep_loop", "use_structured_curiosity",
    "use_suffering_derivative_comparator", "use_tonic_vigor", "use_tpj_comparator",
    "use_trainable_relief_critic", "use_trainable_safety_predictor",
    # sleep_substrate:GAP-9 within-life sleep trigger master switch (REEConfig
    # top-level). Registered here rather than probed, consistent with its sibling
    # sleep-cadence flags already in this set (use_sleep_loop, use_mel_entry,
    # use_mel_consumer, use_sleep_aggregation_cluster): the cluster's behavioural
    # validation runs through the V3-EXQ experiments, not flag-inertness probes.
    # Behavioural inertness OFF and effect ON are already pinned by dedicated
    # contracts in tests/contracts/test_sleep_within_life_trigger_gap9.py (G2 OFF
    # byte-identical; G3/G8/G11/G12 ON fires), and validated by V3-EXQ-929 (v1
    # ceiling arm) plus its need-arm successor.
    # Added to unbreak trunk: v1 (5f14036, ceiling arm) landed the flag without
    # its registry entry, so test_flag_registry_is_current was red on main --
    # same situation as use_mel/use_rem_precision_broadcast above.
    "use_within_life_sleep_trigger",
    "valence_harm_enabled", "valence_liking_enabled",
    # SD-ORIENTING-DECISION-SCALE / SD-099 defensive-orienting master switch
    # (REEConfig top-level, landed 2026-08-08/2026-08-10 -- genuinely predates
    # this session and was ALREADY missing from this registry before the
    # 2026-08-11 nested-scan widening surfaced it, i.e. a pre-existing
    # top-level gap, not a nested-scan artifact). Off by default; only
    # V3-EXQ-910 has ever exercised it (per
    # REE_assembly/docs/architecture/sd_orienting_decision_scale.md), which
    # found and fixed the SD-ORIENTING-DECISION-SCALE scale-mismatch bug
    # inside its Component 4/5 consumer. A dedicated behavioural probe belongs
    # to that cluster's own validation, not this session's SD-RESIDUE-
    # VALENCE-BOUND work; recorded here so the registry-drift guard is green
    # again rather than left red for an unrelated task to trip over.
    "use_defensive_orienting",
}

# --------------------------------------------------------------------------- #
# Nested-config flag registry (widened 2026-08-11, SD-RESIDUE-VALENCE-BOUND    #
# session). `_current_toplevel_flags` below only ever scanned REEConfig's OWN  #
# fields -- it silently missed every `use_*`/`*_enabled` flag living on one of #
# REEConfig's 13 nested config classes (GoalConfig, LatentStackConfig, E1/E2/  #
# E3Config, HippocampalConfig, ResidueConfig, HeartbeatConfig, plus classes    #
# nested a level deeper still, e.g. AnchorSetConfig/GhostGoalBankConfig under  #
# HippocampalConfig). `valence_bounding_enabled` (ResidueConfig, added this    #
# session) would itself have landed uncategorized under the old scan -- which #
# is what surfaced the gap. `_current_nested_flags()` below scans EVERY        #
# dataclass defined in this module (not a REEConfig-field-tree walk -- every   #
# nested config in this codebase happens to also be its own top-level module  #
# dataclass, so a flat per-class scan finds classes at any nesting depth       #
# without needing to unwrap Optional/List/dataclass-typed fields).             #
#                                                                              #
# SCOPE OF THIS WIDENING (user-directed, 2026-08-11): landed the scanner so no #
# new nested flag can slip in uncategorized ever again. The 85 flags it newly #
# discovered that day were initially bulk-seeded here with one GENERIC        #
# placeholder reason -- a deliberate, honest "not yet individually assessed"  #
# marker, not a claim of real audit coverage. The follow-on chip
# (chip-20260811-flaginertness-nested-audit) then worked through them one at  #
# a time: 46 moved to PROBED (real existing behavioural probes the bulk-seed  #
# didn't cross-reference, or new direct probes written for this audit), 2     #
# moved to KNOWN_INERT (F-N1/F-N2, confirmed genuinely dead -- never read     #
# anywhere in ree_core/), and the flags remaining below each now carry an     #
# INDIVIDUAL, SPECIFIC reason (not the generic placeholder) naming the real   #
# consumer and exactly why no existing test isolates its own ON/OFF effect.   #
#                                                                              #
# KNOWN_NAME-COLLISION CAVEAT: this registry is keyed by FLAG NAME, not        #
# (class, name) -- pre-existing design, not something this widening changes.  #
# 5 names exist on BOTH REEConfig top-level AND a nested class with (as far as #
# this audit went) unrelated semantics: use_chunk_proposal_injection,          #
# use_decomposition_scale_resolved_probe, use_habenula_decommit,               #
# use_modulatory_channel_routing, use_modulatory_selection_authority. Each is  #
# categorized once above (top-level) and is not re-listed below; a reader     #
# auditing one of these five should confirm which CLASS's field is meant.     #
KNOWN_UNPROBED_NESTED = {
    # --- HippocampalConfig ----------------------------------------------------#
    # SD-055: replaces the legacy argsort-elite CEM refit with a
    # softmax-weighted mean over ALL candidates so gradient can flow to
    # cue_action_proj. Investigated 2026-08-11 (flaginertness-probe-writing
    # follow-on): the refit `ao_mean`/`ao_std` this branch produces are LOCAL
    # to propose_trajectories's CEM loop and only feed the NEXT iteration's
    # sampling -- with the common num_cem_iterations=1 they are dead output
    # entirely (computed, never consumed). Even at num_cem_iterations>=2, the
    # SD-055 claim is specifically that ao_mean gains a gradient edge back
    # through the per-candidate SCORES tensor (softmax(-scores/T) weighting)
    # that the legacy argsort+index selection structurally cannot have
    # (selection-by-index has no autograd path from the selected values back
    # to the scores that chose them) -- an ON!=OFF numeric diff on the
    # sampled candidates is NOT evidence of this (the KNOWN_UNPROBED_NESTED
    # reason this flag already carried, confirmed correct by this
    # investigation). Neither `ao_mean` nor the per-iteration `scores` tensor
    # is exposed outside the method (not on self, not in
    # get_last_propose_diagnostics()), so isolating the scores-mediated
    # gradient edge from the (present in BOTH branches) direct
    # trajectory-content-to-z_world gradient edge would need either new
    # production instrumentation or monkeypatching private loop-local
    # variables (_score_trajectory / _stack_std) to intercept them -- a
    # larger surgical change than a test file should make unilaterally.
    "use_differentiable_cem",
    # MECH-269/MECH-090: when an E3 commitment is active (beta_gate.
    # is_elevated), releases it if any snapshotted active-anchor key drops out
    # of the current active anchor set. The activating condition is genuinely
    # multi-tick and stateful -- a real commitment in flight, a non-empty
    # snapshotted _committed_anchor_keys, and the live anchor set actually
    # changing membership mid-commitment -- none of which is reachable via a
    # direct class-level call the way the anchor-set/staleness-accumulator
    # flags above are. A confident probe needs the full REEAgent tiny-fixture
    # harness driving a commit-then-invalidate sequence over several ticks,
    # comparable in scope to this file's own MECH-321 mid-execution probes.
    "use_vs_commit_release",
}


def _current_toplevel_flags() -> set:
    fields = dataclasses.fields(config_mod.REEConfig)
    return {
        f.name
        for f in fields
        if f.name.startswith("use_") or f.name.endswith("_enabled")
    }


def _current_nested_flags() -> set:
    """Every `use_*`/`*_enabled` field on any OTHER dataclass in this module.

    Deliberately a flat per-class scan, not a REEConfig field-tree walk: every
    nested config in this codebase is also its own top-level module dataclass
    (GoalConfig, LatentStackConfig, ..., AnchorSetConfig two levels deep under
    HippocampalConfig), so `vars(config_mod)` finds all of them regardless of
    nesting depth without needing to unwrap Optional/List/dataclass field
    types. REEConfig itself is excluded (covered by `_current_toplevel_flags`).
    """
    found = set()
    for _name, obj in vars(config_mod).items():
        if not (dataclasses.is_dataclass(obj) and isinstance(obj, type)):
            continue
        if obj is config_mod.REEConfig:
            continue
        for f in dataclasses.fields(obj):
            if f.name.startswith("use_") or f.name.endswith("_enabled"):
                found.add(f.name)
    return found


def test_flag_registry_is_current():
    """Fail when a top-level OR nested `use_*`/`*_enabled` flag is not categorized.

    Adding a flag forces a decision: write a probe (PROBED) or record it in
    KNOWN_UNPROBED / KNOWN_UNPROBED_NESTED with a reason. This is the
    recurrence guard -- a new dead flag cannot slip in un-noticed the way
    F-C2..F-P6 did, and (as of 2026-08-11) neither can one living on a nested
    config class, which the original REEConfig-only scan could not see.
    """
    covered = PROBED | set(KNOWN_INERT) | KNOWN_UNPROBED | KNOWN_UNPROBED_NESTED
    current = _current_toplevel_flags() | _current_nested_flags()

    uncategorized = sorted(current - covered)
    assert not uncategorized, (
        "New/uncategorized top-level or nested config flag(s): "
        f"{uncategorized}. Add a behavioural probe to test_flag_inertness.py "
        "(PROBED) or record the flag in KNOWN_UNPROBED / KNOWN_UNPROBED_NESTED "
        "with a reason."
    )

    # Keep the snapshot honest: a flag that was renamed/removed should be pruned
    # from the registry rather than lingering as a phantom entry.
    stale = sorted(
        (PROBED | KNOWN_UNPROBED | KNOWN_UNPROBED_NESTED) - current - set(KNOWN_INERT)
    )
    assert not stale, (
        f"Registry lists flag(s) no longer on REEConfig (or a nested config): "
        f"{stale}. Remove them from PROBED / KNOWN_UNPROBED / KNOWN_UNPROBED_NESTED."
    )
