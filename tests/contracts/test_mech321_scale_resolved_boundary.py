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
