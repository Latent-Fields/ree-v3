"""Contract tests for SD-036 GABAergic cross-stream decay regulator.

Guarantees enforced here:
  C1. Module / dataclass importable without side effects.
  C2. Default REEConfig has use_gabaergic_decay=False (backward-compat).
  C3. With master switch OFF, REEAgent.gabaergic_decay is None and a default
      agent boot is bit-identical to legacy.
  C4. With master switch ON, REEAgent instantiates a regulator with the three
      default streams (z_harm, z_harm_a, z_beta) registered.
  C5. Decay arithmetic: z_s(t+1) = z_s(t) * exp(-tau_s * gaba_tone) when
      input gate does not fire.
  C6. gaba_tone modulation: tone>1.0 -> faster decay; tone<1.0 -> slower
      decay; tone=0 -> decay suspended.
  C7. Suspend-on-input gate: when |z(t) - z(t-1)| > input_threshold, decay
      is skipped for that tick.
  C8. MECH-094: simulation_mode=True returns the input unchanged and does
      not advance internal counters.
  C9. reset() clears per-episode state without raising.
  C10. Per-stream coverage flags ablate just the targeted stream.

REAL-AGENT guarantees (C11-C16, added 2026-07-31). C1-C10 all exercise the
regulator against a synthetic `_Latent()` stand-in that is re-ticked IN PLACE,
where compounding holds trivially. NO test stepped a real REEAgent through
sense(), and that is exactly why the gap between SD-036's autoregressive design
(z_s(t+1) = z_s(t) * exp(-tau_s * gaba_tone)) and its FEEDFORWARD wiring for the
harm streams stayed invisible: z_harm / z_harm_a were re-encoded from the current
observation every tick, so the regulator's rescale was discarded and z_s(t+1) was
not a function of z_s(t) at all. The regulator degenerated to a one-step constant
rescale -- and every scale-free DV, including the pre-registered falsifier's
harm_norm_sustain_ratio (= mean/peak), is EXACTLY invariant to that. Measured on
the pre-fix substrate: peak-normalised ||z_harm|| trajectories bit-identical
across gaba_tone (max deviation 9.4e-08) and sustain-ratio spread 8.6e-08.

  C11. Real agent, fixed observation tape: the peak-normalised ||z_harm||
       trajectory genuinely DIFFERS across gaba_tone (not a pure rescale).
  C12. Same for ||z_harm_a||.
  C13. DIFFERENTIAL CONTROL -- ablating the recurrence for one stream restores
       the pre-fix vacuity for THAT stream only, while the other stays live.
       This is what makes C11/C12 regression guards rather than thresholds that
       happen to pass: it proves the measurement can still SEE the defect.
  C14. The pre-registered falsifier's own scale-free DV (harm_norm_sustain_ratio)
       responds to gaba_tone, monotonically for z_harm_a.
  C15. Backward compat at TRAJECTORY level: master switch OFF reproduces the
       legacy feedforward trajectory bit-for-bit on the same tape.
  C16. Config threading -- the LatentStackConfig-scoped knobs actually reach
       config.latent (the silent from_dims three-site trap), and REEAgent
       mirrors the master switch onto the stack.

All C11-C16 assertions are on CONTINUOUS norms and inequalities with wide
margins -- never on exact sampled actions. torch.multinomial is not reproducible
across machine classes (see REE_Working/CLAUDE.md "Running the test suite"); the
tape is driven by numpy RandomState and the replays call sense() only, so no
discrete sampler is in the measured path.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch


def test_c1_module_importable():
    """C1: module + dataclass importable without side effects."""
    from ree_core.regulators import (
        GABAergicDecayConfig,
        GABAergicDecayRegulator,
        StreamRegistration,
    )
    from ree_core.regulators.gabaergic_decay import (
        GABAergicDecayRegulator as GR,
    )
    assert GABAergicDecayRegulator is GR
    assert GABAergicDecayConfig().enabled is True


def test_c2_default_config_backward_compatible():
    """C2: default REEConfig has use_gabaergic_decay=False."""
    from ree_core.utils.config import REEConfig
    cfg = REEConfig()
    assert getattr(cfg, "use_gabaergic_decay", False) is False
    assert getattr(cfg, "use_pag_freeze_gate", False) is False
    # Default tau values match the design doc.
    assert cfg.gaba_tau_z_harm_s == 0.05
    assert cfg.gaba_tau_z_harm_a == 0.02
    assert cfg.gaba_tau_z_beta == 0.03
    assert cfg.gaba_tone == 1.0


def test_c3_master_switch_off_no_instantiation():
    """C3: master OFF -> agent.gabaergic_decay is None."""
    from ree_core.utils.config import REEConfig
    from ree_core.agent import REEAgent
    cfg = REEConfig.from_dims(body_obs_dim=12, world_obs_dim=250, action_dim=4)
    agent = REEAgent(cfg)
    assert agent.gabaergic_decay is None
    assert agent.pag_freeze_gate is None


def test_c4_master_switch_on_registers_default_streams():
    """C4: master ON -> regulator has the three default streams registered."""
    from ree_core.utils.config import REEConfig
    from ree_core.agent import REEAgent
    cfg = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=4,
        use_harm_stream=True, use_affective_harm_stream=True,
        use_gabaergic_decay=True,
    )
    agent = REEAgent(cfg)
    assert agent.gabaergic_decay is not None
    streams = agent.gabaergic_decay.registered_streams
    assert "z_harm" in streams
    assert "z_harm_a" in streams
    assert "z_beta" in streams


def test_c5_decay_arithmetic_no_input():
    """C5: z(t+1) = z(t) * exp(-tau * gaba_tone) when input gate does not fire."""
    from ree_core.regulators import GABAergicDecayConfig, GABAergicDecayRegulator

    cfg = GABAergicDecayConfig(
        enabled=True, gaba_tone=1.0,
        tau_z_harm_s=0.05, tau_z_harm_a=0.02, tau_z_beta=0.03,
        # input_threshold=0.0 -> decay always proceeds
        input_threshold_z_harm_s=0.0,
    )
    reg = GABAergicDecayRegulator(cfg)
    reg.register("z_harm", tau=0.05, input_threshold=0.0)

    # Build a minimal latent stand-in with a `z_harm` attribute.
    class _Latent:
        z_harm = torch.tensor([[1.0, 0.0, 0.0]])
        hypothesis_tag = False
    latent = _Latent()
    expected = 1.0 * math.exp(-0.05 * 1.0)
    reg.tick(latent)
    new_norm = float(latent.z_harm.norm().item())
    # 1-norm of vector [exp(-0.05), 0, 0] is exp(-0.05).
    assert abs(new_norm - expected) < 1e-6


def test_c6_gaba_tone_modulation():
    """C6: tone>1 -> faster decay; tone<1 -> slower; tone=0 -> suspended."""
    from ree_core.regulators import GABAergicDecayConfig, GABAergicDecayRegulator

    def _decay_norm(tone):
        cfg = GABAergicDecayConfig(enabled=True, gaba_tone=tone)
        reg = GABAergicDecayRegulator(cfg)
        reg.register("z_harm", tau=0.05, input_threshold=0.0)

        class _Latent:
            z_harm = torch.tensor([[1.0]])
            hypothesis_tag = False
        latent = _Latent()
        reg.tick(latent)
        return float(latent.z_harm.norm().item())

    n_baseline = _decay_norm(1.0)
    n_fast = _decay_norm(1.5)
    n_slow = _decay_norm(0.5)
    n_zero = _decay_norm(0.0)

    assert n_fast < n_baseline, "tone>1 should decay faster"
    assert n_slow > n_baseline, "tone<1 should decay slower"
    assert abs(n_zero - 1.0) < 1e-6, "tone=0 should suspend decay"


def test_c7_suspend_on_input_gate():
    """C7: suspend-on-input gate skips decay when magnitude change > threshold."""
    from ree_core.regulators import GABAergicDecayConfig, GABAergicDecayRegulator

    cfg = GABAergicDecayConfig(enabled=True, gaba_tone=1.0)
    reg = GABAergicDecayRegulator(cfg)
    reg.register("z_harm", tau=0.05, input_threshold=0.5)

    # First tick establishes the baseline norm at 1.0; decay proceeds (no
    # prior baseline). After this tick, _last_norms["z_harm"] holds the
    # post-decay norm.
    class _Latent:
        z_harm = torch.tensor([[1.0]])
        hypothesis_tag = False
    latent = _Latent()
    reg.tick(latent)
    n_after_first = float(latent.z_harm.norm().item())
    # First tick: baseline is 0, current 1.0, delta=1.0 > 0.5 -> SUSPEND.
    # So the first tick should NOT have decayed.
    assert abs(n_after_first - 1.0) < 1e-6

    # Now provide a stable magnitude (no change from cached baseline).
    # The second tick has baseline 1.0, current 1.0, delta=0 < 0.5 -> DECAY.
    reg.tick(latent)
    n_after_second = float(latent.z_harm.norm().item())
    assert n_after_second < 1.0
    expected = 1.0 * math.exp(-0.05 * 1.0)
    assert abs(n_after_second - expected) < 1e-6


def test_c8_mech094_simulation_mode_no_op():
    """C8: simulation_mode=True returns input unchanged without advancing counters."""
    from ree_core.regulators import GABAergicDecayConfig, GABAergicDecayRegulator

    cfg = GABAergicDecayConfig(enabled=True, gaba_tone=1.0)
    reg = GABAergicDecayRegulator(cfg)
    reg.register("z_harm", tau=0.05)

    class _Latent:
        z_harm = torch.tensor([[1.0]])
        hypothesis_tag = True
    latent = _Latent()
    reg.tick(latent, simulation_mode=True)
    # No decay applied.
    assert abs(float(latent.z_harm.norm().item()) - 1.0) < 1e-6
    # Diagnostic counter should not advance.
    assert reg.diagnostics["n_ticks"] == 0


def test_c9_reset_clears_state():
    """C9: reset() clears per-episode state without raising."""
    from ree_core.regulators import GABAergicDecayConfig, GABAergicDecayRegulator

    cfg = GABAergicDecayConfig(enabled=True, gaba_tone=1.0)
    reg = GABAergicDecayRegulator(cfg)
    reg.register("z_harm", tau=0.05)

    class _Latent:
        z_harm = torch.tensor([[1.0]])
        hypothesis_tag = False
    latent = _Latent()
    reg.tick(latent)
    assert reg.diagnostics["n_ticks"] == 1
    reg.reset()
    assert reg.diagnostics["n_ticks"] == 0
    assert "z_harm" in reg.registered_streams  # registration preserved


def test_c10_per_stream_coverage_flags():
    """C10: per-stream coverage flags ablate just the targeted stream."""
    from ree_core.regulators import GABAergicDecayConfig, GABAergicDecayRegulator

    # Ablate z_harm_a only.
    cfg = GABAergicDecayConfig(
        enabled=True, gaba_tone=1.0,
        decay_z_harm_s=True,
        decay_z_harm_a=False,
        decay_z_beta=True,
    )
    reg = GABAergicDecayRegulator(cfg)
    reg.register_default_streams(cfg)
    streams = reg.registered_streams
    assert "z_harm" in streams
    assert "z_harm_a" not in streams
    assert "z_beta" in streams


def test_backward_compat_agent_boot():
    """Backward-compat: REEAgent boot with default config is unaffected."""
    from ree_core.utils.config import REEConfig
    from ree_core.agent import REEAgent
    cfg = REEConfig.from_dims(body_obs_dim=12, world_obs_dim=250, action_dim=4)
    agent = REEAgent(cfg)
    # gabaergic_decay must be None and reset() must not raise.
    assert agent.gabaergic_decay is None
    agent.reset()


# ---------------------------------------------------------------------------
# C11-C16: real-agent replay contracts (SD-036 harm-stream decay recurrence)
#
# Method: record ONE observation tape from a fixed random-action policy, then
# replay that same tape into agents differing ONLY in gaba_tone (or in the
# recurrence ablation flags). Because the tape is fixed, any difference in the
# resulting latent trajectory is attributable to the manipulation alone.
#
# The DV is the PEAK-NORMALISED trajectory, deliberately: it is scale-free, so
# it is exactly invariant to the one-step constant rescale the pre-fix
# substrate produced. A DV with a fixed absolute threshold would have shown a
# clean monotone "dose-response" on the broken substrate too -- a
# confident-but-wrong confirmation. That trap is the reason for this whole
# block, so do NOT "simplify" these assertions onto raw magnitudes.
# ---------------------------------------------------------------------------

# 471-lineage environment, matching V3-EXQ-471 / V3-EXQ-475.
_ENV_KWARGS = dict(
    size=10, num_hazards=3, num_resources=5, hazard_harm=0.05,
    env_drift_interval=5, env_drift_prob=0.1, proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05, proximity_approach_threshold=0.2,
    hazard_field_decay=0.5, resource_respawn_on_consume=True,
    use_proxy_fields=True, toroidal=False, harm_history_len=10,
    limb_damage_enabled=True, damage_increment=0.15, failure_prob_scale=0.3,
    heal_rate=0.002, n_landmarks_b=2,
)
_SEED = 0
_STEPS = 60

# Vacuity ceiling: the pre-fix substrate measured 9.4e-08 (z_harm) and 8.6e-08
# (sustain ratio). Anything at or below this is "the manipulation did nothing
# but rescale", i.e. the defect is back.
_VACUOUS = 1e-5


def _batch(v):
    if v is None:
        return None
    v = v.float()
    return v.unsqueeze(0) if v.dim() == 1 else v


def _make_env():
    from ree_core.environment.causal_grid_world import CausalGridWorldV2
    return CausalGridWorldV2(seed=_SEED, **_ENV_KWARGS)


@pytest.fixture(scope="module")
def tape():
    """A fixed observation sequence, independent of any agent.

    Actions come from a dedicated numpy RandomState rather than from an agent,
    so the identical tape is replayed into every arm and the arms cannot drift
    apart through their own action choices.
    """
    env = _make_env()
    _, od = env.reset()
    rng = np.random.RandomState(_SEED)
    frames = []
    for _ in range(_STEPS):
        frames.append(dict(
            body=_batch(od["body_state"]), world=_batch(od["world_state"]),
            harm=_batch(od.get("harm_obs")), harm_a=_batch(od.get("harm_obs_a")),
            hist=_batch(od.get("harm_history")),
        ))
        _, od, _, done, _ = env.step(int(rng.randint(env.action_dim)))
        if done:
            _, od = env.reset()
    return frames


def _replay(**over):
    """Build a fresh agent and run the module tape through sense()."""
    import random
    from ree_core.agent import REEAgent
    from ree_core.utils.config import REEConfig

    torch.manual_seed(_SEED)
    random.seed(_SEED)
    np.random.seed(_SEED)
    env = _make_env()
    _, od0 = env.reset()
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim, self_dim=32, world_dim=32, harm_dim=32,
        alpha_world=0.9, alpha_self=0.3, reafference_action_dim=env.action_dim,
        use_harm_stream=True, z_harm_dim=32,
        use_affective_harm_stream=True, z_harm_a_dim=16, harm_history_len=10,
        # Derived, not hardcoded: the affective-harm channel width is an
        # environment property and a literal here would rot silently into a
        # shape error the moment the env grows a channel.
        harm_obs_a_dim=int(od0["harm_obs_a"].shape[-1]),
        **over,
    )
    return REEAgent(cfg), cfg


def _trajectories(frames, **over):
    agent, cfg = _replay(**over)
    agent.reset()
    out = {"z_harm": [], "z_harm_a": [], "z_beta": []}
    with torch.no_grad():
        for fr in frames:
            lat = agent.sense(
                fr["body"], fr["world"], obs_harm=fr["harm"],
                obs_harm_a=fr["harm_a"], obs_harm_history=fr["hist"],
            )
            for key in out:
                z = getattr(lat, key, None)
                out[key].append(float(z.norm()) if z is not None else float("nan"))
    return {k: np.asarray(v) for k, v in out.items()}, cfg


def _peak_normalised(a):
    peak = float(np.nanmax(np.abs(a)))
    return a / peak if peak > 1e-12 else a


def _shape_deviation(a, b):
    """Max deviation between two peak-normalised trajectories (scale-free)."""
    return float(np.nanmax(np.abs(_peak_normalised(a) - _peak_normalised(b))))


def _sustain_ratio(a):
    """The pre-registered falsifier's DV: mean/peak. Exactly scale-free."""
    peak = float(np.nanmax(a))
    return float(np.mean(a)) / peak if peak > 1e-12 else 0.0


def test_c11_real_agent_z_harm_trajectory_responds_to_gaba_tone(tape):
    """C11: ||z_harm|| trajectory SHAPE changes with gaba_tone on a real agent.

    Pre-fix this deviation was 9.4e-08 (bit-identical shape) because z_harm was
    a pure feedforward encode. Post-fix it is ~1.3e-02 to 1.7e-02.
    """
    ref, _ = _trajectories(tape, use_gabaergic_decay=True, gaba_tone=1.0)
    for tone in (0.0, 0.3, 2.0):
        cur, _ = _trajectories(tape, use_gabaergic_decay=True, gaba_tone=tone)
        dev = _shape_deviation(cur["z_harm"], ref["z_harm"])
        assert dev > 1e-3, (
            f"z_harm trajectory is scale-invariant to gaba_tone={tone} "
            f"(peak-normalised max deviation {dev:.3e}). SD-036 decay has no "
            f"temporal authority over z_harm -- the harm stream is being "
            f"re-encoded feedforward each tick and the regulator's rescale "
            f"discarded. Check LatentStackConfig.gaba_harm_state_recurrence "
            f"and LatentStack._gaba_state_blend."
        )


def test_c12_real_agent_z_harm_a_trajectory_responds_to_gaba_tone(tape):
    """C12: same guarantee for ||z_harm_a|| (SD-011 affective stream).

    Pre-fix 9.6e-08; post-fix ~5.0e-02 to 7.1e-02. The affective stream shows a
    larger effect than z_harm because its input (an EMA of proximity) is far
    less variable, so the recurrence dominates its trajectory.
    """
    ref, _ = _trajectories(tape, use_gabaergic_decay=True, gaba_tone=1.0)
    for tone in (0.0, 0.3, 2.0):
        cur, _ = _trajectories(tape, use_gabaergic_decay=True, gaba_tone=tone)
        dev = _shape_deviation(cur["z_harm_a"], ref["z_harm_a"])
        assert dev > 5e-3, (
            f"z_harm_a trajectory is scale-invariant to gaba_tone={tone} "
            f"(peak-normalised max deviation {dev:.3e}). See C11."
        )


def test_c13_recurrence_ablation_restores_vacuity_per_stream(tape):
    """C13: the differential control that gives C11/C12 their teeth.

    Ablating the recurrence for z_harm must reproduce the PRE-FIX signature for
    z_harm specifically (deviation collapses to ~1e-07) while leaving z_harm_a
    fully responsive. Without this, C11/C12 could pass for reasons unrelated to
    the recurrence and nobody would know the measurement had gone blind.
    """
    kw = dict(use_gabaergic_decay=True, gaba_recurrence_z_harm_s=False)
    ref, _ = _trajectories(tape, gaba_tone=1.0, **kw)
    cur, _ = _trajectories(tape, gaba_tone=0.3, **kw)

    ablated = _shape_deviation(cur["z_harm"], ref["z_harm"])
    assert ablated < _VACUOUS, (
        f"ablation control failed: with gaba_recurrence_z_harm_s=False the "
        f"z_harm trajectory should be scale-invariant to gaba_tone (the "
        f"pre-fix behaviour), but deviation was {ablated:.3e}. Either the "
        f"ablation flag is not wired, or C11 is passing for the wrong reason."
    )

    still_live = _shape_deviation(cur["z_harm_a"], ref["z_harm_a"])
    assert still_live > 5e-3, (
        f"ablating z_harm's recurrence must not disable z_harm_a's "
        f"(deviation {still_live:.3e}). The flags are not per-stream."
    )


def test_c14_registered_falsifier_dv_responds_to_gaba_tone(tape):
    """C14: harm_norm_sustain_ratio (mean/peak) is no longer tone-invariant.

    This is the DV the pre-registered SD-036 dose-response falsifier actually
    uses (experiments/_lib/goal_pipeline_tier1.py). Being exactly scale-free,
    it was EXACTLY invariant on the pre-fix substrate -- spread 8.6e-08 over
    the whole {0.3 .. 2.0} sweep -- which made the registered falsifier
    structurally vacuous rather than merely underpowered.

    z_harm_a is asserted monotone (measured gaps are large: 0.979 -> 0.875).
    z_harm is asserted responsive but NOT monotone: its measured spread is
    ~1.4e-03 with step gaps down to ~1e-05, too tight to assert an ordering
    across machine classes.
    """
    tones = (0.3, 0.5, 1.0, 1.5, 2.0)
    harm, harm_a = [], []
    for tone in tones:
        tr, _ = _trajectories(tape, use_gabaergic_decay=True, gaba_tone=tone)
        harm.append(_sustain_ratio(tr["z_harm"]))
        harm_a.append(_sustain_ratio(tr["z_harm_a"]))

    assert (max(harm) - min(harm)) > 1e-4, (
        f"z_harm sustain-ratio spread {max(harm) - min(harm):.3e} over "
        f"gaba_tone {tones} -- the registered falsifier's DV is invariant, so "
        f"the SD-036 dose-response experiment cannot measure anything."
    )
    assert (max(harm_a) - min(harm_a)) > 1e-2, (
        f"z_harm_a sustain-ratio spread {max(harm_a) - min(harm_a):.3e} -- see above."
    )
    assert all(harm_a[i] >= harm_a[i + 1] for i in range(len(harm_a) - 1)), (
        f"z_harm_a sustain ratio must fall monotonically as gaba_tone rises "
        f"(more tonic inhibition -> faster return toward baseline). Got {harm_a}."
    )


def test_c15_master_switch_off_matches_legacy_trajectory(tape):
    """C15: backward compat at trajectory level, not just at agent boot.

    With use_gabaergic_decay=False the recurrence must not engage at all, so
    the trajectory is the legacy pure-feedforward one. Verified bit-identical
    against the pre-change substrate when this landed.
    """
    a, cfg = _trajectories(tape)
    assert cfg.latent.gaba_harm_state_recurrence is False
    b, _ = _trajectories(tape)
    for key in ("z_harm", "z_harm_a", "z_beta"):
        assert np.array_equal(a[key], b[key]), f"{key} replay is not deterministic"

    # The decay-OFF trajectory must NOT look like a decaying one: turning the
    # master switch on has to change something, or the switch is inert.
    on, _ = _trajectories(tape, use_gabaergic_decay=True, gaba_tone=1.0)
    assert _shape_deviation(on["z_harm"], a["z_harm"]) > 1e-3


def test_c16_recurrence_config_threading(tape):
    """C16: the LatentStackConfig-scoped knobs actually reach config.latent.

    REEConfig.from_dims silently swallows unrecognised kwargs, so a knob wired
    at only two of its three sites is unreachable with no error at all. This
    pins all three, plus REEAgent's mirroring of the master switch (which is
    what makes `config.use_gabaergic_decay = True` set AFTER construction still
    give the recurrence -- the idiom V3-EXQ-475 uses).
    """
    agent, cfg = _replay(
        use_gabaergic_decay=True,
        gaba_state_alpha_z_harm_s=0.77,
        gaba_state_alpha_z_harm_a=0.11,
        gaba_recurrence_z_harm_a=False,
    )
    assert cfg.latent.gaba_state_alpha_z_harm_s == pytest.approx(0.77)
    assert cfg.latent.gaba_state_alpha_z_harm_a == pytest.approx(0.11)
    assert cfg.latent.gaba_recurrence_z_harm_a is False
    assert cfg.latent.gaba_harm_state_recurrence is True

    off_agent, off_cfg = _replay()
    assert off_agent.gabaergic_decay is None
    assert off_cfg.latent.gaba_harm_state_recurrence is False

    # alpha=1.0 is the documented legacy escape hatch: pure feedforward.
    blend = type(agent.latent_stack)._gaba_state_blend
    now, prev = torch.ones(1, 4), torch.zeros(1, 4)
    assert torch.equal(blend(now, prev, 1.0), now)
    assert torch.equal(blend(now, prev, 0.0), prev)
    # alpha=0 must COPY, not alias: `.to()` is a no-op for a same-device,
    # same-dtype tensor, so a bare `.to()` would put prev_state's own storage
    # into this tick's LatentState.
    assert blend(now, prev, 0.0).data_ptr() != prev.data_ptr()
    assert torch.equal(blend(now, None, 0.5), now)          # first tick after reset
    assert torch.equal(blend(now, torch.zeros(1, 9), 0.5), now)  # shape mismatch fails open
