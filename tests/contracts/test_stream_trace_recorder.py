"""
Contract tests for the Q-081 per-step multi-stream trace recorder.

Pins the properties the Q-081 telemetry audit
(REE_assembly/evidence/planning/q081_cross_stream_telemetry_audit.md) identified as
load-bearing. Each test names the audit finding it defends:

  C1 VECTORS, NOT NORMS (audit section 1). z_world is recorded at full width, not
     collapsed to a scalar. The legacy episode-log `z_world_norm` is derivable from the
     recording; the reverse is not, which is the whole reason the corpus cannot answer
     Q-081 retrospectively.
  C2 PER-SIGNAL FRESHNESS (audit section 3). E3-derived streams are flagged fresh on the
     E3 tick and held in between. Without this, a naive recorder writes 9 duplicates +
     1 fresh value per E3 stream and any cross-stream analysis finds strong regular
     shared structure that is a SAMPLER ARTEFACT -- Outcome B, the exact null the
     experiment exists to exclude.
  C3 BOUNDARY EVENTS ARE READ NON-DESTRUCTIVELY (audit section 3). The recorder copies
     hippocampal._boundary_event_queue and never drains it, so the agent's own
     start-of-sense() flush (agent.py:4217-4220) is unchanged.
  C4 STORE BY REFERENCE (audit section 4 item 2). finalize() returns a lean pointer with
     no array payload, and the blob verifies against its content digest.
  C5 THE CONFIG PROFILE (audit section 4 item 3). Every flag it flips is still False by
     default -- the silent-null trap -- and each is a real from_dims kwarg that actually
     lands on the config (a typo'd kwarg is swallowed silently).
  C6 THE RECORDER IS INERT. Attaching it does not change what the agent does.

ASCII-only. Run: pytest tests/contracts/test_stream_trace_recorder.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig

from experiments._lib import q081_profile as prof
from experiments._lib import trace_store as ts
from experiments._lib.stream_recorder import (
    FRESH_IDENTITY,
    StreamTraceRecorder,
    derive_norms,
    rate_matched_shuffle_index,
)

WORLD_DIM = 16


def _build(seed: int = 11, **flags):
    from ree_core.environment.causal_grid_world import CausalGridWorldV2

    torch.manual_seed(seed)
    env = CausalGridWorldV2(
        seed=seed, size=5, num_hazards=1, num_resources=1, use_proxy_fields=True
    )
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        self_dim=16,
        world_dim=WORLD_DIM,
        **flags,
    )
    agent = REEAgent(cfg)
    agent.reset()
    return env, agent, cfg


def _split(od):
    b, w = od["body_state"], od["world_state"]
    if b.dim() == 1:
        b = b.unsqueeze(0)
    if w.dim() == 1:
        w = w.unsqueeze(0)
    return b, w


def _run(agent, env, recorder, n_steps: int, drive_harm: bool = False):
    """Drive the agent, sampling AFTER act() and BEFORE the next act() (audit sec 3).

    `drive_harm=True` uses the explicit sense/select loop that feeds the harm channels.
    act_with_split_obs() calls sense() WITHOUT them, so the harm streams are structurally
    null on the convenience path however the flags are set -- see
    q081_profile.LOOP_DRIVEN_REQUIREMENTS.
    """
    _flat, od = env.reset()
    actions = []
    for _ in range(n_steps):
        b, w = _split(od)
        with torch.no_grad():
            if drive_harm:
                ticks = agent.clock.advance()
                latent = agent.sense(b, w, **prof.sense_kwargs_from_obs(od))
                e1_prior = (
                    agent._e1_tick(latent) if ticks["e1_tick"]
                    else torch.zeros(1, agent.config.latent.world_dim)
                )
                cands = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(cands, ticks)
                agent._step_count += 1
            else:
                action = agent.act_with_split_obs(b, w)
        actions.append(action.detach().clone())
        # CausalGridWorldV2.step -> (flat_obs, harm_signal, done, info, obs_dict)
        _flat, _harm, done, _info, od = env.step(action)
        if done:
            _flat, od = env.reset()
        if recorder is not None:
            recorder.on_step()
    return actions


# --------------------------------------------------------------------------- #
# C1 vectors, not norms
# --------------------------------------------------------------------------- #

def test_c1_latents_recorded_as_full_vectors_not_norms(tmp_path):
    env, agent, _ = _build()
    rec = StreamTraceRecorder(agent, run_id="c1", store=ts.TraceStore(root=tmp_path))
    _run(agent, env, rec, 12)
    ptr = rec.finalize()
    arrays = ts.TraceStore(root=tmp_path).get(ptr)["arrays"]

    assert arrays["z_world"].shape == (12, WORLD_DIM), (
        "z_world must be recorded at full width; a norm destroys the "
        "system-level configuration structure Q-081 is about"
    )
    assert arrays["z_self"].shape[1] == 16
    assert arrays["z_beta"].shape[1] > 1
    # More than one distinct column value: this is genuinely multi-dimensional data.
    assert np.nanstd(arrays["z_world"], axis=0).max() > 0.0

    # The legacy episode-log scalar is recoverable from the vector recording.
    norms = derive_norms(arrays, "z_world")
    assert norms.shape == (12,)


# --------------------------------------------------------------------------- #
# C2 per-signal freshness -- the Outcome-B defence
# --------------------------------------------------------------------------- #

def test_c2_e3_streams_are_fresh_on_the_tick_and_held_between(tmp_path):
    env, agent, cfg = _build()
    assert cfg.heartbeat.e3_steps_per_tick == 10, "test assumes the stock E3 rate"
    rec = StreamTraceRecorder(agent, run_id="c2", store=ts.TraceStore(root=tmp_path))
    n = 40
    _run(agent, env, rec, n)
    ptr = rec.finalize()
    loaded = ts.TraceStore(root=tmp_path).get(ptr)
    arrays, meta = loaded["arrays"], loaded["meta"]

    cols = meta["clock_columns"]
    e3_tick = arrays["clock"][:, cols.index("e3_tick")].astype(bool)

    # The clock block itself must show the multi-rate schedule, not every-step E3.
    assert 0 < e3_tick.sum() < n, "E3 must tick sometimes and not every step"

    # E3 commitment state is clock-derived: fresh exactly on the tick.
    assert np.array_equal(arrays["e3_commitment__fresh"], e3_tick)

    # z_world is re-encoded every step (E1 rate) -- the contrast that makes the flag
    # informative rather than a constant.
    assert arrays["z_world__fresh"].all()
    assert arrays["z_world__fresh"].sum() > arrays["e3_commitment__fresh"].sum()

    # No E3-derived quantity may be carried by an every-step-fresh stream: that would
    # hand an analyst a stream that LOOKS E1-rate while its content only moves at the E3
    # rate -- the artefact the flags exist to prevent, smuggled in via bundling.
    e3_derived = ("precision", "running_variance", "is_committed")
    for name, desc in meta["streams"].items():
        if arrays[f"{name}__fresh"].all():
            note = desc["note"].lower()
            for term in e3_derived:
                assert term not in note.split("e3-derived")[0], (
                    f"stream '{name}' is flagged fresh every step but its note mentions "
                    f"the E3-derived quantity '{term}'"
                )

    # Every stream declares how its flag was derived.
    for name, desc in meta["streams"].items():
        assert desc["freshness_source"] in meta["freshness_semantics"], name


def test_c2_identity_freshness_survives_an_equal_valued_recompute():
    """Identity-based freshness must not be fooled by a recompute that yields the same
    numbers -- the case a value-difference heuristic silently misreads as 'held'."""
    rec = StreamTraceRecorder.__new__(StreamTraceRecorder)
    rec._prev_obj = {}
    same_value_a = [1.0, 2.0]
    same_value_b = [1.0, 2.0]  # equal, but a DIFFERENT object == a recompute
    assert rec._fresh_by_identity("k", same_value_a) is True    # first sight
    assert rec._fresh_by_identity("k", same_value_a) is False   # held
    assert rec._fresh_by_identity("k", same_value_b) is True    # recomputed


def test_c2_rate_matched_shuffle_preserves_the_hold_pattern():
    """The Q-081 shuffle control must permute FRESH samples only. A whole-series
    shuffle would also destroy the hold structure and let the control pass trivially."""
    fresh = np.zeros(30, dtype=bool)
    fresh[::10] = True
    idx = rate_matched_shuffle_index(None, [], fresh, np.random.default_rng(0))
    assert np.array_equal(np.sort(idx), np.arange(30))
    held = ~fresh
    assert np.array_equal(idx[held], np.flatnonzero(held)), "held samples must not move"
    assert set(idx[fresh]) == set(np.flatnonzero(fresh)), "fresh samples permute among themselves"


# --------------------------------------------------------------------------- #
# C3 boundary events -- non-destructive read at the right point
# --------------------------------------------------------------------------- #

def test_c3_recorder_does_not_drain_the_boundary_queue(tmp_path):
    env, agent, _ = _build(**prof.q081_profile_kwargs())
    if agent.hippocampal is None:
        pytest.skip("hippocampal module not constructed under this config")

    rec = StreamTraceRecorder(agent, run_id="c3", store=ts.TraceStore(root=tmp_path))
    _flat, od = env.reset()
    b, w = _split(od)
    with torch.no_grad():
        agent.act_with_split_obs(b, w)

    # Plant a sentinel so the assertion does not depend on the segmenter firing.
    class _Ev:
        t, posterior, scale = 3, 0.75, "fast"
        segment_id_old, segment_id_new = "0.0", "0.1"

    agent.hippocampal._boundary_event_queue.append(_Ev())
    before = len(agent.hippocampal._boundary_event_queue)
    rec.on_step()
    after = len(agent.hippocampal._boundary_event_queue)

    assert after == before, (
        "the recorder must COPY the queue, never drain it -- draining would steal the "
        "events from the agent's own start-of-sense() flush"
    )
    ptr = rec.finalize()
    loaded = ts.TraceStore(root=tmp_path).get(ptr)
    assert loaded["arrays"]["boundary_events"][0, 0] == 1.0
    assert loaded["arrays"]["boundary_events__fresh"][0]
    assert any(e["stream"] == "boundary_events" for e in loaded["meta"]["events"])


# --------------------------------------------------------------------------- #
# C4 store by reference
# --------------------------------------------------------------------------- #

def test_c4_pointer_is_lean_and_carries_no_arrays(tmp_path):
    env, agent, _ = _build()
    rec = StreamTraceRecorder(agent, run_id="c4", store=ts.TraceStore(root=tmp_path))
    _run(agent, env, rec, 20)
    ptr = rec.finalize()

    assert ts.pointer_is_lean(ptr), "the manifest gets a pointer, never a payload"
    assert "events" not in ptr and "streams_meta" not in ptr
    for value in ptr.values():
        assert not isinstance(value, (np.ndarray, list)) or all(
            isinstance(x, str) for x in (value if isinstance(value, list) else [])
        )
    assert ptr["bytes"] > 0
    assert (tmp_path / ptr["filename"]).exists()


def test_c4_content_digest_is_packaging_independent_and_verifies(tmp_path):
    arrays = {"a": np.arange(6, dtype=np.float32).reshape(3, 2)}
    s1 = ts.TraceStore(root=tmp_path / "one")
    s2 = ts.TraceStore(root=tmp_path / "two")
    p1 = s1.put(arrays, meta={"k": 1})
    p2 = s2.put({"a": arrays["a"].copy()}, meta={"k": 1})
    assert p1["sha256"] == p2["sha256"], (
        "identical content must land on the same address; digesting the .npz FILE bytes "
        "would fail here because numpy embeds wallclock time in the zip headers"
    )
    assert np.array_equal(s1.get(p1)["arrays"]["a"], arrays["a"])

    # Tampering is caught.
    (tmp_path / "one" / p1["filename"]).write_bytes(
        (tmp_path / "two" / p2["filename"]).read_bytes()[:-3] + b"xxx"
    )
    with pytest.raises(Exception):
        s1.get(p1)


def test_c4_missing_blob_reports_the_machine_it_was_written_on(tmp_path):
    store = ts.TraceStore(root=tmp_path, machine="ree-cloud-3")
    ptr = store.put({"a": np.zeros(2, dtype=np.float32)}, meta={})
    (tmp_path / ptr["filename"]).unlink()
    with pytest.raises(FileNotFoundError, match="ree-cloud-3"):
        store.get(ptr)


# --------------------------------------------------------------------------- #
# C5 the config profile
# --------------------------------------------------------------------------- #

def test_c5_every_profile_flag_is_still_off_by_default():
    observed = prof.verify_stock_defaults(REEConfig)
    on = [k for k, v in observed.items() if v]
    assert not on, (
        "these flags are no longer default-OFF, so the non-default-substrate "
        "declaration is now wrong: %s" % on
    )


def test_c5_profile_kwargs_actually_land_on_the_config():
    """from_dims silently swallows unknown kwargs, so a typo'd flag name would leave the
    signal dark while the profile reported success."""
    _env, _agent, cfg = _build(**prof.q081_profile_kwargs())
    for name in prof.Q081_FLAGS:
        assert prof._read_flag(cfg, name) is True, f"{name} did not take effect"


def test_c5_declaration_names_every_flag_and_the_e2_decision():
    _env, _agent, cfg = _build(**prof.q081_profile_kwargs())
    decl = prof.q081_substrate_declaration(cfg)
    assert decl["is_default_substrate"] is False
    assert set(decl["flags"]) == set(prof.Q081_FLAGS)
    for name, entry in decl["flags"].items():
        assert entry["stock_default"] is False
        assert entry["effective_value"] is True, name
    # The E2 substitution hazard must stay documented in the artifact itself.
    assert "tpj" in decl["e2_pe_decision"].lower()
    assert set(prof.STREAMS_GATED_BY.values()) <= set(prof.Q081_FLAGS)


def test_c5_profile_run_populates_the_streams_that_are_dark_by_default(tmp_path):
    env, agent, _ = _build(**prof.q081_profile_kwargs())
    rec = StreamTraceRecorder(agent, run_id="c5", store=ts.TraceStore(root=tmp_path),
                              substrate_declaration=prof.q081_substrate_declaration())
    _run(agent, env, rec, 25, drive_harm=True)
    loaded = ts.TraceStore(root=tmp_path).get(rec.finalize())
    arrays, meta = loaded["arrays"], loaded["meta"]

    assert meta["non_default_substrate"]["is_default_substrate"] is False
    # z_goal needs z_goal_enabled to EXIST at all (agent.goal_state is None otherwise);
    # driving it with update_z_goal is then the loop's job.
    assert arrays["z_goal"].shape[1] > 1, "z_goal stream absent -- z_goal_enabled off?"
    for stream in ("z_harm", "z_harm_a", "operating_mode"):
        assert arrays[f"{stream}__valid"].any(), (
            f"{stream} is null even with the profile on -- the silent-null trap"
        )
    assert arrays["z_harm"].shape[1] > 1, "z_harm must be a vector, not a norm"


def test_c5_e2_stream_is_e3_cadence_not_per_step(tmp_path):
    """Pins the measured caveat on the E2 decision (q081_profile "E2 PE").

    use_tpj_comparator buys a genuinely DISTINCT quantity from E3's own error, but not a
    distinct RATE: _cache_tpj_prediction_for_action runs at the end of select_action
    (agent.py:7582), after the between-tick short-circuit (agent.py:5463), so nothing is
    staged on a held step. Q-081 therefore has no true middle-rate stream, and an
    analysis that assumed one would be wrong. If this ever becomes per-step the test
    fails loudly and the declaration must be re-worded rather than quietly inherited.
    """
    env, agent, _ = _build(**prof.q081_profile_kwargs())
    rec = StreamTraceRecorder(agent, run_id="c5d", store=ts.TraceStore(root=tmp_path))
    n = 60
    _run(agent, env, rec, n, drive_harm=True)
    loaded = ts.TraceStore(root=tmp_path).get(rec.finalize())
    arrays, meta = loaded["arrays"], loaded["meta"]

    valid = arrays["e2_self_pe__valid"]
    assert valid.any(), "use_tpj_comparator did not produce any E2 samples at all"
    assert not valid.all(), "E2 is unexpectedly per-step -- re-word the E2 declaration"

    e3_tick = arrays["clock"][:, meta["clock_columns"].index("e3_tick")].astype(bool)
    # Resolves on E3 ticks only (allowing the one-step settle at episode start).
    assert valid.sum() <= e3_tick.sum() + 1
    assert "E3 CADENCE" in meta["streams"]["e2_self_pe"]["note"]


def test_c5_flags_alone_do_not_populate_the_harm_streams(tmp_path):
    """The flags are necessary but NOT sufficient (q081_profile.LOOP_DRIVEN_REQUIREMENTS).

    agent.act()/act_with_split_obs() call sense() with no harm channels, and
    LatentStack.encode gates the harm encoders on `harm_obs is not None`. So a Q-081 loop
    that uses the convenience act() path records z_harm as a well-formed NULL while the
    profile reports every flag enabled -- exactly the silent-null trap, one layer deeper
    than the audit documented. This test pins the asymmetry so it stays visible.
    """
    env, agent, _ = _build(**prof.q081_profile_kwargs())
    rec = StreamTraceRecorder(agent, run_id="c5c", store=ts.TraceStore(root=tmp_path))
    _run(agent, env, rec, 12, drive_harm=False)
    arrays = ts.TraceStore(root=tmp_path).get(rec.finalize())["arrays"]
    assert not arrays["z_harm__valid"].any()
    assert "z_harm" in prof.LOOP_DRIVEN_REQUIREMENTS
    assert "z_harm_a" in prof.LOOP_DRIVEN_REQUIREMENTS


def test_c5_default_config_leaves_the_gated_streams_null(tmp_path):
    """The contrast that makes the profile necessary: without it a quarter of the
    checklist records as null and the run still looks fine."""
    env, agent, _ = _build()
    rec = StreamTraceRecorder(agent, run_id="c5b", store=ts.TraceStore(root=tmp_path))
    _run(agent, env, rec, 12)
    arrays = ts.TraceStore(root=tmp_path).get(rec.finalize())["arrays"]
    assert not arrays["z_harm__valid"].any()
    assert not arrays["operating_mode__valid"].any()


# --------------------------------------------------------------------------- #
# C6 the recorder is inert
# --------------------------------------------------------------------------- #

def test_c6_attaching_the_recorder_does_not_change_the_agent(tmp_path):
    env_a, agent_a, _ = _build(seed=5)
    without = _run(agent_a, env_a, None, 25)

    env_b, agent_b, _ = _build(seed=5)
    rec = StreamTraceRecorder(agent_b, run_id="c6", store=ts.TraceStore(root=tmp_path))
    with_rec = _run(agent_b, env_b, rec, 25)

    # Asserted on the continuous pre-quantizer action tensor, NOT on a sampled discrete
    # action: torch.multinomial is not reproducible across machine classes
    # (MEMORY reference_cross_machine_class_contract_divergence).
    assert len(without) == len(with_rec)
    for a, b in zip(without, with_rec):
        assert torch.allclose(a, b, atol=1e-6), "recorder perturbed the agent"

    # And the agent's own state counters agree.
    assert agent_a.get_state().step == agent_b.get_state().step
    assert agent_a.clock.global_step == agent_b.clock.global_step


def test_c6_finalize_is_single_shot(tmp_path):
    env, agent, _ = _build()
    rec = StreamTraceRecorder(agent, run_id="c6b", store=ts.TraceStore(root=tmp_path))
    _run(agent, env, rec, 3)
    rec.finalize()
    with pytest.raises(RuntimeError):
        rec.finalize()
    with pytest.raises(RuntimeError):
        rec.on_step()


def test_c6_every_stream_has_exactly_one_row_per_step(tmp_path):
    env, agent, _ = _build()
    rec = StreamTraceRecorder(agent, run_id="c6c", store=ts.TraceStore(root=tmp_path))
    n = 17
    _run(agent, env, rec, n)
    loaded = ts.TraceStore(root=tmp_path).get(rec.finalize())
    arrays, meta = loaded["arrays"], loaded["meta"]
    assert meta["n_steps"] == n
    for name in meta["streams"]:
        assert arrays[name].shape[0] == n, name
        assert arrays[f"{name}__fresh"].shape == (n,), name
        assert arrays[f"{name}__valid"].shape == (n,), name
    assert arrays["clock"].shape == (n, len(meta["clock_columns"]))
    # Ragged block stays aligned to the step axis.
    assert arrays["e3_scores__offsets"].shape == (n + 1,)
    assert arrays["e3_scores__fresh"].shape == (n,)
    assert int(arrays["e3_scores__offsets"][-1]) == arrays["e3_scores__values"].shape[0]
