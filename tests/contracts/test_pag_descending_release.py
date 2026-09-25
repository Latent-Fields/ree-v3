"""MECH-287 option B: hippocampal-invalidation -> PAG freeze-EXIT descending release.

Design: REE_assembly/evidence/planning/mech287_anchor_freeze_exit_design_20260925.md
(user option B, 2026-09-25). Before this build the PAG freeze exit read only
||z_harm_a||, gaba_tone (constant 1.0) and the SD-037 override (not built in
the MECH-287 lineage), so no MECH-287 manipulation could move the lock DV
(mech287_lock_dv_inert_pag_path_20260925.md).

Contracts pinned here:

  C1  Config reaches the gate through all three REEConfig sites, and the
      unknown-source guard fires.
  C1b Bit-identity: path absent (defaults) == master ON at alpha 0, on a
      rollout where invalidations DO drive the trace (non-vacuous: the
      trace is asserted > 0). Actions, latents, RNG state and every PAG
      exit_threshold are identical.
  C2  Gate-level reach (D2): on a matched z_harm_a trajectory, a descending
      release pulse shortens freeze duration versus the matched control; at
      alpha 0 the pulse changes nothing (positive control against vacuity).
  C3  Agent-level reach (D2): an injected broadcast that invalidates an
      ACTIVE anchor inside sense() drives the trace, and the next PAG tick's
      exit_threshold exceeds a matched control agent's, with every earlier
      tick identical.
  C4  Source exclusion: ordinary boundary remaps (successor anchor installed)
      never drive the trace, and source="broadcast" ignores hysteresis resets.
  C5  Liveness: with the path ON at alpha > 0 the recorded exit_threshold
      differs from OFF on the same scripted-boundary rollout.

Boundary events are injected through the segmenter's own force_boundary()
API and hysteresis resets are induced by raising AnchorSet.reset_threshold:
both sit UPSTREAM of the subject under test (the invalidation -> trace ->
exit-threshold path), which runs unstubbed. An untrained tiny rollout emits
no boundaries on its own (measured 2026-09-25: 0 in 200 steps), so without
the injection every assertion below would be vacuous.
"""

from __future__ import annotations

import hashlib
import warnings

import pytest
import torch

from ree_core.agent import REEAgent
from ree_core.pag.freeze_gate import PAGFreezeGate, PAGFreezeGateConfig
from ree_core.regulators.invalidation_trigger import BroadcastEvent
from ree_core.utils.config import REEConfig

from tests.fixtures.seed_utils import set_all_seeds
from tests.fixtures.tiny_configs import make_tiny_config
from tests.fixtures.tiny_env import make_tiny_env
from tests.fixtures.tiny_loop import step_once

warnings.filterwarnings("ignore", category=UserWarning)

LINEAGE = dict(
    use_pag_freeze_gate=True,
    use_per_stream_vs=True,
    use_event_segmenter=True,
    use_invalidation_trigger=True,
    use_anchor_sets=True,
    use_per_region_vs=True,
    use_staleness_accumulator=True,
    use_mech284_hysteresis=True,
)
STEPS = 40
BOUNDARY_EVERY = 10  # > hysteresis_k (5), so a hysteresis reset can fire first


# ---------------------------------------------------------------- helpers ---

def _build(seed, *, hysteresis=True, **overrides):
    set_all_seeds(seed)
    env = make_tiny_env(seed=seed)
    flags = dict(LINEAGE)
    if not hysteresis:
        # No H invalidations at all: without the MECH-284 staleness lookup the
        # internal proxy staleness is (ticks * 0.005), far below what would
        # drop V_s_anchor under reset_threshold inside a short rollout.
        flags.update(use_staleness_accumulator=False, use_mech284_hysteresis=False)
    flags.update(overrides)
    cfg = make_tiny_config(env, action_dim=env.action_dim, **flags)
    if hysteresis:
        # V_s_anchor <= 1 < 2.0, so every active anchor's streak grows and it
        # fires at hysteresis_k: a real H invalidation, a few ticks after
        # each install.
        cfg.hippocampal.anchor_set.reset_threshold = 2.0
    agent = REEAgent(cfg)
    seg = agent.hippocampal.event_segmenter
    real_step = seg.step
    # A local sense() call counter: the tiny loop does not advance
    # agent._step_count, so the segmenter's own `t` kwarg stays 0.
    calls = {"n": -1}
    agent._contract_sense_calls = calls

    def scripted_step(*a, **k):
        calls["n"] += 1
        events = list(real_step(*a, **k))
        if calls["n"] % BOUNDARY_EVERY == 2:
            events.append(seg.force_boundary("fast", "contract"))
        return events

    seg.step = scripted_step
    return agent, env


def _rollout(seed, steps=STEPS, inject_at=None, **overrides):
    """Returns (actions, pag_exit_thresholds_per_step, digest, agent)."""
    agent, env = _build(seed, **overrides)
    if inject_at is not None:
        trig = agent.hippocampal.invalidation_trigger
        real_trig = trig.step

        def trig_step(*a, **k):
            out = list(real_trig(*a, **k))
            # The trigger ticks right after the segmenter in the same sense().
            if agent._contract_sense_calls["n"] == inject_at:
                active = agent.hippocampal.anchor_set.active_anchors()
                assert active, "injection needs an active anchor"
                anc = active[0]
                out.append(BroadcastEvent(
                    t=inject_at, strength=0.8, posterior=0.8,
                    targets=["anchor_reset"], source_scale=anc.key[0],
                    source_segment_id_old=anc.key[1],
                    source_segment_id_new=anc.key[1] + ".x",
                    source_sources=["contract"],
                ))
            return out

        trig.step = trig_step
    agent.reset()
    _flat, obs = env.reset()
    actions, thr = [], []
    for _ in range(steps):
        _a, idx, _t, obs = step_once(agent, env, obs)
        actions.append(idx)
        o = agent._pag_last_output
        thr.append(None if o is None else float(o.exit_threshold))
    h = hashlib.sha256()
    lat = agent._current_latent
    for t in (lat.z_world, lat.z_self):
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    h.update(torch.get_rng_state().numpy().tobytes())
    return actions, thr, h.hexdigest(), agent


# -------------------------------------------------------------------- C1 ---

def test_c1_config_three_sites_reach_the_gate():
    base = REEConfig()
    assert base.use_pag_descending_release is False
    fd = REEConfig.from_dims(body_obs_dim=12, world_obs_dim=250, action_dim=5)
    assert fd.use_pag_descending_release is False
    env = make_tiny_env(seed=1)
    on = REEAgent(make_tiny_config(
        env, action_dim=env.action_dim, use_pag_freeze_gate=True,
        use_pag_descending_release=True, pag_descending_release_alpha=2.5,
        pag_descending_release_decay=0.8, pag_descending_release_source="broadcast",
    ))
    assert on.pag_freeze_gate.config.alpha_descending == 2.5
    d = on.pag_descending_release_diagnostics()
    assert d["enabled"] and d["decay"] == 0.8 and d["source"] == "broadcast"
    # Master off -> gate gain is an exact 0.0 no-op regardless of alpha.
    off = REEAgent(make_tiny_config(
        env, action_dim=env.action_dim, use_pag_freeze_gate=True,
        pag_descending_release_alpha=2.5,
    ))
    assert off.pag_freeze_gate.config.alpha_descending == 0.0
    assert off.pag_descending_release_diagnostics()["enabled"] is False
    # No freeze gate -> path not built even with the master on.
    nogate = REEAgent(make_tiny_config(
        env, action_dim=env.action_dim, use_pag_descending_release=True,
    ))
    assert nogate.pag_descending_release_diagnostics()["enabled"] is False


def test_c1_unknown_source_raises():
    env = make_tiny_env(seed=1)
    with pytest.raises(ValueError):
        REEAgent(make_tiny_config(
            env, action_dim=env.action_dim, use_pag_freeze_gate=True,
            use_pag_descending_release=True,
            pag_descending_release_source="anything",
        ))


def test_c1b_bit_identity_absent_vs_on_alpha0_with_live_drive():
    absent = _rollout(7)
    on0 = _rollout(7, use_pag_descending_release=True, pag_descending_release_alpha=0.0)
    diag = on0[3].pag_descending_release_diagnostics()
    # Non-vacuous: invalidations actually drove the trace in this rollout.
    assert diag["n_h_events"] > 0 and diag["n_drive_steps"] > 0, diag
    assert any(t is not None for t in absent[1])
    assert absent[0] == on0[0]
    assert absent[1] == on0[1]
    assert absent[2] == on0[2]


# -------------------------------------------------------------------- C2 ---

def _gate_duration(alpha, pulse_from=None, ticks=30):
    g = PAGFreezeGate(PAGFreezeGateConfig(theta_freeze=2.0, alpha_descending=alpha))
    # z = 3.0 commits on tick 1 (3 * 1 > 2) and holds the lock (3 > 2);
    # at tick 12 z falls to 1.5 < 2 and the control releases.
    first_release = None
    for i in range(ticks):
        z = 3.0 if i < 12 else 1.5
        r = 1.0 if (pulse_from is not None and i >= pulse_from) else 0.0
        out = g.tick(z_harm_a_norm=z, descending_release=r)
        if i == 0:
            assert out.freeze_commit
        if out.freeze_release and first_release is None:
            first_release = i
    return first_release


def test_c2_gate_pulse_shortens_freeze_vs_matched_control():
    control = _gate_duration(alpha=1.0, pulse_from=None)
    pulsed = _gate_duration(alpha=1.0, pulse_from=4)
    assert control == 12
    # exit_threshold = 2 * (1 + 1 * 1) = 4 > 3 from tick 4.
    assert pulsed == 4
    assert pulsed < control


def test_c2_positive_control_alpha0_pulse_is_inert():
    assert _gate_duration(alpha=0.0, pulse_from=4) == _gate_duration(alpha=0.0)


def test_c2_output_records_descending_release_and_clamps():
    g = PAGFreezeGate(PAGFreezeGateConfig(theta_freeze=2.0, alpha_descending=1.0))
    out = g.tick(z_harm_a_norm=0.1, descending_release=5.0)
    assert out.descending_release == 1.0
    assert out.exit_threshold == pytest.approx(4.0)
    out = g.tick(z_harm_a_norm=0.1, descending_release=-1.0)
    assert out.descending_release == 0.0
    assert out.exit_threshold == 2.0


# -------------------------------------------------------------------- C3 ---

def test_c3_injected_invalidation_raises_next_pag_exit_threshold():
    # Hysteresis off, so the ONLY invalidation is the injected T3 broadcast.
    kw = dict(use_pag_descending_release=True, pag_descending_release_alpha=1.0,
              hysteresis=False)
    inject_at = 5  # after the t=2 forced boundary installed an anchor
    ctrl = _rollout(3, steps=20, **kw)
    inj = _rollout(3, steps=20, inject_at=inject_at, **kw)
    d_inj = inj[3].pag_descending_release_diagnostics()
    d_ctrl = ctrl[3].pag_descending_release_diagnostics()
    assert d_inj["n_t3_events"] == 1, d_inj
    assert d_ctrl["n_t3_events"] == 0 and d_ctrl["n_drive_steps"] == 0, d_ctrl
    diffs = [i for i, (a, b) in enumerate(zip(inj[1], ctrl[1])) if a != b]
    assert diffs, "injected invalidation never reached the PAG exit threshold"
    first = diffs[0]
    # Loop index i is the i-th sense() call (0-based), so the divergence
    # cannot precede the injection step.
    assert first >= inject_at
    assert inj[1][first] > ctrl[1][first]
    assert inj[0][:first] == ctrl[0][:first]


# -------------------------------------------------------------------- C4 ---

def test_c4_ordinary_boundary_remap_does_not_drive():
    _a, _t, _d, agent = _rollout(
        9, use_pag_descending_release=True, hysteresis=False
    )
    aset = agent.hippocampal.anchor_set
    # Non-vacuous: remaps DID happen (inactive dual-trace anchors exist).
    assert len(aset.all_anchors()) > len(aset.active_anchors())
    d = agent.pag_descending_release_diagnostics()
    assert d["n_drive_steps"] == 0 and d["trace"] == 0.0, d


def test_c4_broadcast_source_ignores_hysteresis_resets():
    _a, _t, _d, agent = _rollout(
        7, use_pag_descending_release=True, pag_descending_release_source="broadcast"
    )
    d = agent.pag_descending_release_diagnostics()
    assert d["n_h_events"] > 0, d
    assert d["n_drive_steps"] == 0 and d["trace"] == 0.0, d


# -------------------------------------------------------------------- C5 ---

def test_c5_liveness_on_changes_exit_threshold():
    off = _rollout(7)
    on = _rollout(7, use_pag_descending_release=True, pag_descending_release_alpha=1.0)
    pairs = [(a, b) for a, b in zip(off[1], on[1]) if a is not None and b is not None]
    assert pairs
    assert any(b > a for a, b in pairs), "INERT: path changed no exit threshold"
    assert all(b >= a for a, b in pairs)
