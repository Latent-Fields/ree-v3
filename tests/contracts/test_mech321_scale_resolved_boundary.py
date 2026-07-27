"""MECH-321 scale-resolved rollout boundary diagnostic -- interface contracts.

Scoping spike: REE_assembly/evidence/planning/
mech321_decomposition_scale_scoping_spike_2026-07-27.md section 5b.
Claim: MECH-321 (candidate / v3_pending) -- UNCHANGED by this file. These are
interface guarantees only; whether the two scales actually dissociate is the
queued experiment's question, not a contract.

THE FINDING THESE PIN. MECH-288's EventSegmenter ships two qualitatively
heterogeneous scales -- fast (PE-threshold sliding-window z-score over
z_world + z_self, per-tick) and slow (BOCPD-Gaussian over z_goal, hazard
1/40). MECH-321 discards that structure TWICE:

  1. boundary_on() collapses List[BoundaryEvent] to fired:bool (any scale) +
     posterior:max. `.events` carries `.scale` per event, but
     PolicyDecomposition.evaluate() reads only .fired / .posterior.
  2. HippocampalModule._evaluate_decomposition_ticks builds latent_signature
     as {z_world, z_self} with NO z_goal, and the BOCPD detector returns
     (False, 0.0, []) when none of its streams is present -- so the slow
     scale is STRUCTURALLY DEAD on the rollout stream.

MECH-321 is therefore a single-scale consumer of a two-scale substrate, and
the one scale it does consume arrives as a boolean with the label stripped.

C18a  default path is bit-identical: OFF -> the rollout signature is exactly
      {z_world, z_self}, no z_goal key at all
C18b  the flag is NOT inert: ON -> z_goal is present in that signature
C18c  from_dims three-site wiring pin (field + signature + assignment; see
      memory reference-reeconfig-from-dims-silent-kwargs -- from_dims ends in
      **kwargs and silently drops an unknown one)
C18d  per-scale counters sum against decomp_n_boundary_fires by the exact
      identity fires_fast + fires_slow - cofire == n_boundary_fires
C18e  default path counters are inert-but-present: with only the fast scale
      live, slow and cofire stay 0 and fast == n_boundary_fires
C18f  co-fire is OBSERVABLE on a real EventSegmenter. The cross-scale rule
      makes a slow fire SUPPRESS the same-tick fast event, so an events-only
      cofire counter would be a constant 0 and could not tell spike outcome 1
      ("dissociable") from outcome 3 ("two detectors, one signal") -- the very
      question the probe exists to answer
C18g  BoundaryQueryResult.suppressed_scales is diagnostic ONLY: it never
      perturbs fired / posterior / events for existing MECH-288 consumers

THE MID-EXECUTION HALF (C18h-k). C18a-g close the gap on MECH-321's R4 FIRST
phase only -- the pre-commitment sweep in HippocampalModule. The SECOND phase,
the mid-execution hook in agent.py:select_action() that re-evaluates the
REMAINING content of an already-committed chunk trajectory, builds its OWN
{z_world, z_self} signature and was untouched. That is a MEASUREMENT bug, not
merely a missing feature: PolicyDecomposition accumulates the per-scale
counters inside evaluate(), which is phase-blind, so every mid-execution tick
adds to fires_fast and to the denominator while being structurally incapable
of adding to fires_slow -- diluting the observed slow fraction by exactly the
mid-execution share of evaluations (measurable via decomp_n_evaluated_
{precommit,midexec}, but not removable).

C18h  default path is bit-identical: OFF -> the mid-execution signature is
      exactly {z_world, z_self}, no z_goal key at all
C18i  the flag is NOT inert on the LIVE WIRED PATH: ON -> z_goal is present in
      the signature the real select_action() hook hands to evaluate(). Driven
      through select_action rather than a helper on purpose -- the four-site
      bug C18c pins was an UNREACHABLE knob, which only a wired-path test can
      catch, and the mid-execution flag has no sub-config mirror to inspect
C18j  from_dims THREE-site wiring pin (field + signature + assignment) -- and
      the deliberate ABSENCE of a hippocampal mirror, because the consumer is
      REEAgent reading top-level REEConfig, not HippocampalModule reading its
      own sub-config. Do not add a fourth site by analogy with C18c
C18k  the two probe flags are INDEPENDENT in both directions. They are
      separate knobs because V3-EXQ-830 defines its ARM_PROBE_OFF /
      ARM_PROBE_ON manipulation as the pre-commitment flag ALONE and was
      claimed and running when the mid-execution half landed; merging them
      would have silently redefined an in-flight experiment's arms

NOTE ON FILE PLACEMENT. These would naturally live in
tests/contracts/test_arc070_policy_decomposition.py with C1-C17. They are
split out because that file was being concurrently extended with the spike's
OTHER follow-on (section 5a, the depth_cap/chunk_max_depth coupling guard) by
a separate session, and co-editing it would have forced the two independent
changes to land as one coupled commit.
"""

import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.hippocampal.event_segmenter import (
    BoundaryEvent,
    BoundaryQueryResult,
    EventSegmenter,
    Scale,
)
from ree_core.policy import (
    ChunkedPrimitive,
    ChunkState,
    DecompositionDecision,
    PolicyDecomposition,
    PolicyDecompositionConfig,
)
from ree_core.predictors.e2_fast import Trajectory
from ree_core.utils.config import REEConfig


def _env_agent(**cfg_kwargs):
    env = CausalGridWorldV2()
    env.reset()
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        reafference_action_dim=env.action_dim,
        **cfg_kwargs,
    )
    return REEAgent(cfg)


class _RecordingDecompositionSource:
    """Stub PolicyDecomposition that captures the latent_signature it is
    handed, so the module-side wiring can be pinned without depending on
    stochastic rollout content."""

    def __init__(self):
        self.signatures = []
        self.config = PolicyDecompositionConfig()

    def evaluate(self, **kwargs):
        self.signatures.append(dict(kwargs["latent_signature"]))
        return DecompositionDecision(
            should_decompose=False,
            decomposed=False,
            marked_unreliable=False,
            v_s=1.0,
            boundary_fired=False,
            boundary_posterior=0.0,
            depth=int(kwargs.get("depth", 1)),
            hypothesis_tag=bool(kwargs.get("hypothesis_tag", True)),
        )


class _ScriptedScaleSegmenter:
    """MECH-288 stub emitting a scripted (emitted_scales, suppressed_scales)
    per boundary_on() call, using the REAL BoundaryEvent/BoundaryQueryResult
    dataclasses so the shape under test is the shipped one."""

    slow_scale_name = "slow"

    def __init__(self, script):
        self._script = list(script)
        self._i = 0

    def boundary_on(self, stream, latent, pe=None, t=None):
        emitted, suppressed = self._script[self._i % len(self._script)]
        self._i += 1
        events = [
            BoundaryEvent(
                segment_id_old="0.0",
                segment_id_new="0.1",
                scale=name,
                posterior=1.0,
                sources=[],
                t=self._i,
                input_stream=stream,
            )
            for name in emitted
        ]
        return BoundaryQueryResult(
            fired=bool(events),
            posterior=1.0 if events else 0.0,
            events=events,
            suppressed_scales=list(suppressed),
        )


def _rollout_signatures(flag_value):
    """Drive HippocampalModule's pre-commit sweep once and return every
    latent_signature it built."""
    agent = _env_agent(
        use_event_segmenter=True,
        use_policy_decomposition=True,
        use_decomposition_scale_resolved_probe=flag_value,
    )
    source = _RecordingDecompositionSource()
    traj = Trajectory(
        states=[torch.zeros(1, 32) for _ in range(3)],
        actions=torch.zeros(1, 3, 4),
        world_states=[torch.zeros(1, 32) for _ in range(3)],
    )
    agent.hippocampal._evaluate_decomposition_ticks(
        source,
        traj,
        1,
        (0, 1, 2),
        hypothesis_tag=True,
        library=None,
        current_z_goal=torch.ones(1, 8),
    )
    assert source.signatures, "sweep must call evaluate() at least once"
    return source.signatures


def test_c18a_default_path_signature_has_no_z_goal():
    """OFF is bit-identical: the dict handed to boundary_on carries exactly
    the fast scale's two streams, so MECH-288's slow BOCPD detector still
    short-circuits on `not sources` and cannot fire on rollout -- the
    as-shipped behaviour the probe is measured against."""
    for sig in _rollout_signatures(False):
        assert set(sig.keys()) == {"z_world", "z_self"}
        assert "z_goal" not in sig


def test_c18b_flag_is_not_inert_signature_gains_z_goal():
    """ON adds z_goal and nothing else. This is deliberately NOT a pure
    diagnostic -- the slow scale can now reach boundary.fired and so change
    decisions -- which is exactly why it sits behind a flag."""
    for sig in _rollout_signatures(True):
        assert set(sig.keys()) == {"z_world", "z_self", "z_goal"}
        assert sig["z_goal"] is not None


def test_c18c_from_dims_forwards_the_probe_flag():
    """FOUR-site wiring pin: REEConfig field / from_dims signature / from_dims
    assignment / mirror onto config.hippocampal.

    from_dims ends in **kwargs and SILENTLY DROPS an unknown one, so a
    short-of-complete change leaves the flag unreachable with NO error -- see
    memory reference-reeconfig-from-dims-silent-kwargs. The fourth site is the
    one this file caught in development: HippocampalModule reads its OWN
    HippocampalConfig, not REEConfig, so the top-level knob alone left the
    probe dead and the defensive getattr default made it fail SILENTLY -- the
    probe would have reported a clean 'slow never fires' null (spike outcome 2)
    that was really just a wiring bug. Both halves are asserted separately so a
    regression names which site was dropped."""
    default = REEConfig.from_dims(body_obs_dim=8, world_obs_dim=16, action_dim=4)
    assert default.use_decomposition_scale_resolved_probe is False
    assert default.hippocampal.use_decomposition_scale_resolved_probe is False

    on = REEConfig.from_dims(
        body_obs_dim=8,
        world_obs_dim=16,
        action_dim=4,
        use_decomposition_scale_resolved_probe=True,
    )
    assert on.use_decomposition_scale_resolved_probe is True, "top-level site"
    assert on.hippocampal.use_decomposition_scale_resolved_probe is True, (
        "hippocampal sub-config mirror -- HippocampalModule reads this one"
    )


def _drive(script, n_calls):
    seg = _ScriptedScaleSegmenter(script)
    op = PolicyDecomposition(PolicyDecompositionConfig(use_policy_decomposition=True))
    for _ in range(n_calls):
        op.evaluate(
            region_vs=1.0,  # well above threshold: the V_s half never fires
            latent_signature={"z_world": torch.zeros(1, 4)},
            event_segmenter=seg,
            depth=1,
            sequence=(),
        )
    return op.get_state()


def test_c18d_per_scale_counters_satisfy_the_sum_identity():
    """fires_fast + fires_slow - cofire == n_boundary_fires, over all four
    reachable tick shapes: nothing, fast alone, slow alone, and slow with a
    cross-scale-suppressed fast."""
    script = [
        ([], []),                 # neither detector fired
        (["fast"], []),           # fast alone
        (["slow"], []),           # slow alone, fast silent
        (["slow"], ["fast"]),     # co-fire: slow emitted, fast suppressed
    ]
    st = _drive(script, len(script))

    assert st["decomp_n_boundary_fires"] == 3
    assert st["decomp_n_boundary_fires_fast"] == 2
    assert st["decomp_n_boundary_fires_slow"] == 2
    assert st["decomp_n_boundary_cofire"] == 1
    assert (
        st["decomp_n_boundary_fires_fast"]
        + st["decomp_n_boundary_fires_slow"]
        - st["decomp_n_boundary_cofire"]
        == st["decomp_n_boundary_fires"]
    )


def test_c18e_default_single_scale_path_leaves_slow_and_cofire_at_zero():
    """The counters are safe to add unconditionally: on the shipped
    single-scale rollout stream (fast only) slow and cofire stay 0 and fast
    degenerates to n_boundary_fires, so nothing about the default manifest
    reading changes."""
    st = _drive([(["fast"], []), ([], [])], 6)
    assert st["decomp_n_boundary_fires_slow"] == 0
    assert st["decomp_n_boundary_cofire"] == 0
    assert st["decomp_n_boundary_fires_fast"] == st["decomp_n_boundary_fires"] == 3


def _cofire_segmenter():
    return EventSegmenter(
        scales=[
            Scale(
                name="fast",
                streams=("z_world", "z_self"),
                algorithm="pe_threshold",
                tau=2,
                min_segment_length=1,
                pe_threshold=0.5,
                window_length=8,
            ),
            Scale(
                name="slow",
                streams=("z_goal",),
                algorithm="bocpd_gaussian",
                tau=40,
                min_segment_length=1,
                hazard=0.1,
                posterior_threshold=0.5,
                top_k=20,
                prior_var=1.0,
            ),
        ],
        slow_scale_name="slow",
    )


def _lat(goal, world):
    return {
        "z_world": torch.ones(1, 4) * world,
        "z_self": torch.ones(1, 4) * world,
        "z_goal": torch.ones(1, 4) * goal,
    }


def test_c18f_cofire_is_observable_on_a_real_segmenter():
    """The cross-scale rule makes a slow fire REPLACE the same-tick fast
    event, so `events` alone can never show both scales and an events-only
    cofire counter would be a constant 0 -- unable to distinguish spike
    outcome 1 (dissociable) from outcome 3 (two detectors, one signal).
    suppressed_scales recovers it. Pinned end-to-end on a REAL
    EventSegmenter, not a stub, because this is a property of step()'s
    suppression branch rather than of the counter arithmetic."""
    seg = _cofire_segmenter()
    for i in range(12):
        seg.boundary_on("rollout", _lat(1.0, 1.0 + 0.01 * i))

    # Joint jump: the goal stream moves decisively (BOCPD change-point) at the
    # same tick as a large PE on z_world/z_self.
    res = seg.boundary_on("rollout", _lat(60.0, 160.0))

    assert [e.scale for e in res.events] == ["slow"], (
        "slow must suppress the same-tick fast event -- if this changes, the "
        "cofire counter's definition must change with it"
    )
    assert "fast" in res.suppressed_scales, (
        "the withheld fast fire must still be reported, else co-fire is "
        "structurally unmeasurable"
    )
    assert res.fired is True


def test_c18g_suppressed_scales_is_diagnostic_only():
    """Existing MECH-288 consumers are bit-identical: suppressed_scales never
    changes fired / posterior / events, is empty whenever nothing was
    cross-scale-suppressed, and does not leak across streams or survive
    reset()."""
    seg = _cofire_segmenter()
    for i in range(12):
        res = seg.boundary_on("rollout", _lat(1.0, 1.0 + 0.01 * i))
        # No slow fire on these ticks -> nothing can be suppressed.
        assert res.suppressed_scales == []
        assert res.fired is bool(res.events)

    res = seg.boundary_on("rollout", _lat(60.0, 160.0))
    assert res.suppressed_scales  # the co-fire tick from C18f
    # MECH-094: the observation stream is untouched by rollout ticks.
    assert seg._last_suppressed["observation"] == []
    seg.reset()
    assert seg._last_suppressed["rollout"] == []


# ======================================================================
# C18h-k -- the MID-EXECUTION half (R4 second phase, agent.py)
# ======================================================================
def _live_env_agent(**cfg_kwargs):
    env = CausalGridWorldV2()
    _, obs_dict = env.reset()
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        reafference_action_dim=env.action_dim,
        **cfg_kwargs,
    )
    return env, obs_dict, REEAgent(cfg)


def _drive_both_phases(seed_goal=False, **cfg_kwargs):
    """Drive ONE live agent through both MECH-321 phases and return every
    latent_signature handed to PolicyDecomposition.evaluate(), split by phase.

    The driving idiom is C9's in test_arc070_policy_decomposition.py (the
    shipped mid-execution latch-release contract), so this pins the SAME wired
    path that contract already exercises rather than a parallel construction.

    evaluate() is WRAPPED, not replaced, so real decision logic and the real
    per-scale counters still run underneath -- and because both phases funnel
    through the one object, a single drive yields the mid-execution signature
    AND the pre-commitment signature, which is what makes the C18k cross-talk
    check exact rather than inferred.
    """
    env, obs_dict, agent = _live_env_agent(
        use_event_segmenter=True,
        use_policy_chunking=True,
        use_chunk_proposal_injection=True,
        use_policy_decomposition=True,
        decomposition_vs_threshold=0.99,  # near-certain trigger, as in C8/C9
        **cfg_kwargs,
    )
    seq = (0, 1, 2)
    agent.policy_chunking.library.register(
        ChunkedPrimitive(
            sequence=seq, depth=1, state=ChunkState.CRYSTALLISED, selection_weight=1.0
        )
    )

    precommit, midexec = [], []
    _real_evaluate = agent.policy_decomposition.evaluate

    def _recording_evaluate(**kwargs):
        sink = precommit if kwargs.get("hypothesis_tag") else midexec
        sink.append(dict(kwargs["latent_signature"]))
        return _real_evaluate(**kwargs)

    agent.policy_decomposition.evaluate = _recording_evaluate

    latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
    if seed_goal:
        # update_z_goal is the SOLE z_goal write hook. Without it goal_state
        # never activates and z_goal stays the zero vector / None -- which is
        # layer (c) of the three-layer silent inertness V3-EXQ-830's authoring
        # gate caught, and the reason "the key is present" and "the slow scale
        # has a stream" are DIFFERENT assertions.
        agent.update_z_goal(benefit_exposure=1.0, drive_level=1.0)
    ticks = agent.clock.advance()
    e1_prior = (
        agent._e1_tick(latent)
        if ticks["e1_tick"]
        else torch.zeros(1, 32, device=agent.device)
    )
    candidates = agent.generate_trajectories(latent, e1_prior, ticks)

    fake_traj = candidates[0]
    fake_traj.metadata = {
        "source": "arc071_chunk",
        "chunk_sequence": list(seq),
        "chunk_depth": 1,
    }
    agent.e3._committed_trajectory = fake_traj
    agent._committed_step_idx = 1  # 2 steps remaining -- above the >1 floor
    agent.beta_gate.elevate()
    assert agent.beta_gate.is_elevated

    agent.select_action(candidates, ticks)

    # Guard against a VACUOUS pass: if the hook were never reached, every
    # assertion below would hold over an empty list and the contract would
    # silently stop testing anything.
    assert midexec, "select_action must reach the mid-execution hook"
    assert agent.get_policy_decomposition_state()["decomp_n_evaluated_midexec"] >= 1
    return precommit, midexec


def test_c18h_default_path_midexec_signature_has_no_z_goal():
    """OFF is bit-identical: the dict the mid-execution hook hands to
    evaluate() carries exactly the fast scale's two streams, so MECH-288's
    slow BOCPD detector still short-circuits on `not sources` and cannot fire
    on mid-execution ticks -- the as-shipped behaviour, and the source of the
    slow-fraction dilution the flag exists to remove."""
    _, midexec = _drive_both_phases()
    for sig in midexec:
        assert set(sig.keys()) == {"z_world", "z_self"}
        assert "z_goal" not in sig


def test_c18i_midexec_flag_is_not_inert_on_the_live_wired_path():
    """ON adds z_goal to the mid-execution signature and nothing else.

    Driven through the real select_action() rather than a testable helper on
    purpose. The failure this guards against is the one C18c caught on the
    pre-commitment side: a knob that is set on the config but never reaches
    its consumer fails SILENTLY behind a defensive getattr default, and reads
    out as a clean 'slow never fires' null rather than as a wiring bug. The
    mid-execution flag has no sub-config mirror to inspect, so the wired path
    is the only place that unreachability is observable."""
    _, midexec = _drive_both_phases(
        use_decomposition_scale_resolved_probe_midexec=True
    )
    for sig in midexec:
        assert set(sig.keys()) == {"z_world", "z_self", "z_goal"}


def test_c18i2_midexec_z_goal_is_a_real_stream_not_just_a_present_key():
    """Key presence is NOT sufficient, and asserting only that would repeat the
    mistake this whole probe exists to correct.

    MECH-288's BOCPD detector skips a None entry (`if z is None: continue`)
    and then short-circuits on `not sources`, so a signature carrying
    z_goal=None leaves the slow scale exactly as dead as omitting the key --
    a flag that looks wired and measures nothing. The default agent has
    z_goal_enabled=False, so goal_state is None and the ON path DOES hand over
    None; that is honest, not a bug (no goal stream, nothing to give). This
    contract pins the other half: with the goal stream enabled and its sole
    write hook called, the mid-execution signature carries a real tensor.

    Layers (a) and (c) of the three-layer inertness V3-EXQ-830's readiness
    gate refused twice before queuing -- see that queue entry's note."""
    _, midexec = _drive_both_phases(
        seed_goal=True,
        z_goal_enabled=True,
        benefit_threshold=0.05,  # the substrate's own enable_goal_stream default
        use_decomposition_scale_resolved_probe_midexec=True,
    )
    assert any(sig["z_goal"] is not None for sig in midexec), (
        "with the goal stream live, z_goal must reach the mid-execution "
        "signature as a tensor -- a present-but-None key leaves MECH-288's "
        "slow BOCPD scale just as structurally dead as an absent one"
    )


def test_c18j_from_dims_forwards_the_midexec_flag_and_adds_no_mirror():
    """THREE-site wiring pin (field / from_dims signature / from_dims
    assignment), and the deliberate absence of a fourth.

    from_dims ends in **kwargs and SILENTLY DROPS an unknown one -- see memory
    reference-reeconfig-from-dims-silent-kwargs -- so the forwarding half must
    be pinned exactly as C18c pins it for the pre-commitment flag.

    The mirror half is asserted NEGATIVELY on purpose. C18c's fourth site
    exists because HippocampalModule reads its OWN HippocampalConfig; the
    mid-execution consumer is REEAgent.select_action(), which reads top-level
    REEConfig off self.config. A mirror added here by analogy would be a field
    nothing reads -- the inverse failure, and just as invisible. If a future
    change makes a sub-config the reader, this assertion is the right place to
    fail and be updated deliberately."""
    default = REEConfig.from_dims(body_obs_dim=8, world_obs_dim=16, action_dim=4)
    assert default.use_decomposition_scale_resolved_probe_midexec is False

    on = REEConfig.from_dims(
        body_obs_dim=8,
        world_obs_dim=16,
        action_dim=4,
        use_decomposition_scale_resolved_probe_midexec=True,
    )
    assert on.use_decomposition_scale_resolved_probe_midexec is True, "top-level site"
    assert not hasattr(
        on.hippocampal, "use_decomposition_scale_resolved_probe_midexec"
    ), (
        "no hippocampal mirror: the consumer is REEAgent reading top-level "
        "REEConfig, not HippocampalModule reading its own sub-config"
    )


def test_c18k_the_two_probe_flags_are_independent_in_both_directions():
    """Pre-commitment and mid-execution are SEPARATE knobs, and each moves
    only its own phase.

    Why they are separate rather than one widened flag: V3-EXQ-830 defines its
    ARM_PROBE_OFF / ARM_PROBE_ON manipulation as
    use_decomposition_scale_resolved_probe ALONE, and was claimed and running
    (2026-07-27T18:27:03Z, ~330 min) when the mid-execution half landed.
    Widening the existing flag would have silently redefined the arms of an
    in-flight experiment. This contract is what makes a later 'simplification'
    that merges them fail loudly."""
    precommit_only, midexec_under_precommit = _drive_both_phases(
        use_decomposition_scale_resolved_probe=True
    )
    assert precommit_only, "the pre-commitment sweep must have run"
    assert any("z_goal" in sig for sig in precommit_only), (
        "pre-commitment flag must still reach its own phase"
    )
    for sig in midexec_under_precommit:
        assert "z_goal" not in sig, (
            "the pre-commitment flag must NOT reach the mid-execution hook"
        )

    precommit_under_midexec, midexec_only = _drive_both_phases(
        use_decomposition_scale_resolved_probe_midexec=True
    )
    for sig in midexec_only:
        assert "z_goal" in sig, "mid-execution flag must reach its own phase"
    for sig in precommit_under_midexec:
        assert "z_goal" not in sig, (
            "the mid-execution flag must NOT reach the pre-commitment sweep"
        )
