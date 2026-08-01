"""ARC-070 policy_decomposition_via_event_segmenter (MECH-321) -- interface contracts.

Interface-level guarantees, NOT magnitude thresholds. The magnitudes (how
often decomposition fires, whether execution-time prediction failures drop)
belong to the queued validation experiment, not here.

C1  config defaults + from_dims surfaces them + master OFF is bit-identical
    (no hippocampal sub-config mirror needed -- see config.py field docstring)
C2  loud precondition: use_policy_decomposition=True requires
    hippocampal.use_event_segmenter=True
C3  decompose_sequence(): a depth-1 sequence unpacks to raw-action leaves
    (regression pin -- this session's own activation smoke test caught a
    `depth <= 1` early-return bug that made depth-1 chunks, the common case,
    permanently un-decomposable; see policy_decomposition.py history)
C4  decompose_sequence(): depth>1 tiles against a registered shallower
    sub-chunk in the library (longest-match) before falling back to raw
    actions for uncovered positions
C5  decompose_sequence(): an atomic (len<=1) sequence returns no tiles
C6  evaluate(): R1 is an OR trigger -- V_s alone, boundary alone, or neither
C7  evaluate(): R3 depth_cap -- a primitive at/above depth_cap that should
    decompose is marked_unreliable, not decomposed further
C8  live agent (pre-commit): a withheld ARC-071 chunk candidate is replaced
    in the CEM pool by its leaf-tile candidates, not silently dropped
C9  live agent (mid-execution, R4 second phase): a triggering remainder
    releases the beta-gate commit latch
C10 chunk injection stays additive: an untriggered chunk candidate passes
    through the pool unchanged

R5 bottleneck trigger mode (trigger_mode="bottleneck"; ARM_2 of the MECH-321
discriminative validation -- added 2026-07-25):
C11 config: trigger_mode defaults "vs_boundary"; validate() rejects an
    unknown mode and non-positive bottleneck gates; from_dims forwards the
    bottleneck params (three-site wiring pin)
C12 bottleneck mode IGNORES V_s: a region V_s far below threshold (which
    WOULD fire the R1 OR trigger) does NOT trigger in bottleneck mode absent
    bottleneck support -- and the V_s audit counter still increments, proving
    V_s was low yet not decision-driving (the ARM_2 discriminative evidence)
C13 bottleneck one-shot-rare vs repeated-fires: a once-seen region never
    triggers; a region revisited past bottleneck_min_visits with >=
    bottleneck_min_distinct_neighbors distinct neighbours DOES
C14 bottleneck diagnostics surfaced in get_state (trigger_mode, fires,
    regions_tracked)
C15 "vs_boundary" mode is bit-identical: the bottleneck accumulator is never
    touched (regions_tracked == 0) even over many low-V_s evaluate() calls
C16 bottleneck mode still respects R3 depth_cap: a bottleneck-triggering
    primitive at depth_cap is marked_unreliable, not decomposed

R3 depth_cap / chunk_max_depth coupling guard (MECH-321 scoping spike
2026-07-27 section 5a -- added 2026-07-27):
C17 depth_cap is a DERIVED MIRROR of ARC-071's chunk_max_depth, not a free
    parameter. depth_cap_config_issues() flags exactly two silent
    mis-configurations -- INERT (> chunk_max_depth: the mark-unreliable-by-cap
    branch is unreachable, so 4 and 100 are the same run) and DEGENERATE
    (== 1: every chunk has depth >= 1, so MECH-321 collapses to pure
    withholding) -- and REEAgent emits them as UserWarnings at the wiring
    site, which is the only place both knobs are visible. The useful range
    [2, chunk_max_depth] must stay SILENT and BEHAVIOURALLY UNCHANGED: this
    is a warning, never a raise, because shipped MECH-321 experiments already
    run the inert value 4.
C17b the mirrored ceiling MOVES: under ARC-071's use_growable_chunk_depth
    (Solway 2014) chunk_max_depth is only the STARTING value of a ceiling
    derived from the deliberation budget, so the INERT test follows
    derived_chunk_max_depth instead. Omitted / not-higher leaves the guard
    byte-identical, which is the do-no-harm pin for every shipped run.

SD-hazard-aware-policy-decomposition (V3-EXQ-844 autopsy successor; two-stage
threat-modulated selection among a withheld chunk's OWN candidate re-tilings
-- added 2026-08-01):
C18 config: use_harm_aware_selection defaults False; from_dims forwards the
    harm_* params (three-site wiring pin, mirrors C1/C11)
C19 PolicyDecomposition.harm_threat_scale: linear ramp 0 at floor, 1 at ref,
    1.0 when ref<=floor (degenerate-ramp safety, mirrors
    InstrumentalAvoidanceGate.threat_scale)
C20 PolicyDecomposition.harm_bias: 0 when the flag is off or w<=0; clamped to
    harm_bias_scale; increments the nonzero diagnostic counter
C21 PolicyDecomposition.select_harm_aware_leaves: below
    harm_override_w_threshold keeps every item unchanged (harm-blind
    default); at/above it keeps only the single lowest-penalty item (stable
    argmin); flag off is a pure passthrough regardless of w
C22 live agent, pre-commit, BELOW threshold: every leaf tile of a withheld
    chunk is kept (additive recombination preserved) and tagged with
    decomposition_harm_penalty / decomposition_harm_bias metadata
C23 live agent, pre-commit, AT/ABOVE threshold: only the lowest-harm-penalty
    leaf survives; mech321_harm_override_fires == 1
C24 live agent: a candidate's decomposition_harm_bias metadata is gathered
    into REEAgent.select_action's composed score_bias chain (differential
    check against an untagged candidate, so any other uniform bias source
    already active in the minimal test config cancels out)
"""

import warnings

import pytest
import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.policy import (
    ChunkedPrimitive,
    ChunkLibrary,
    ChunkState,
    DecompositionDecision,
    PolicyChunking,
    PolicyDecomposition,
    PolicyDecompositionConfig,
    depth_cap_config_issues,
)
from ree_core.residue.field import VALENCE_HARM_DISCRIMINATIVE
from ree_core.utils.config import REEConfig


class _FakeBoundary:
    def __init__(self, fired: bool, posterior: float = 0.0):
        self.fired = fired
        self.posterior = posterior
        self.events = []


class _FakeSegmenter:
    """Stub MECH-288 -- returns a scripted boundary_on() result."""

    def __init__(self, fired: bool = False, posterior: float = 0.0):
        self._fired = fired
        self._posterior = posterior
        self.calls = []

    def boundary_on(self, stream, latent, pe=None, t=None):
        self.calls.append({"stream": stream, "latent": latent, "pe": pe, "t": t})
        return _FakeBoundary(self._fired, self._posterior)


class _FakeDecompositionSource:
    """Stub PolicyDecomposition -- returns a scripted evaluate() result
    every call, for deterministic (non-stochastic-rollout-dependent) tests
    of HippocampalModule's pool-filtering wiring."""

    def __init__(self, should_decompose: bool):
        self._should_decompose = should_decompose
        self.n_calls = 0

    def evaluate(self, **kwargs):
        self.n_calls += 1
        return DecompositionDecision(
            should_decompose=self._should_decompose,
            decomposed=False,
            marked_unreliable=False,
            v_s=1.0,
            boundary_fired=False,
            boundary_posterior=0.0,
            depth=int(kwargs.get("depth", 1)),
            hypothesis_tag=bool(kwargs.get("hypothesis_tag", True)),
        )


def _env_agent(**cfg_kwargs):
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
    agent = REEAgent(cfg)
    return env, obs_dict, agent


# ----------------------------------------------------------------------
# C1 -- config surface + OFF is bit-identical
# ----------------------------------------------------------------------
def test_c1_defaults_are_off_and_from_dims_forwards():
    cfg = REEConfig.from_dims(body_obs_dim=8, world_obs_dim=16, action_dim=4)
    assert cfg.use_policy_decomposition is False
    assert cfg.decomposition_vs_threshold == 0.4
    assert cfg.decomposition_depth_cap == 3

    agent = REEAgent(cfg)
    assert agent.policy_decomposition is None
    assert agent.get_policy_decomposition_state() == {}


def test_c1_from_dims_forwards_non_default_values():
    # NB depth_cap=4 exceeds the default chunk_max_depth=3, so this config
    # also trips the C17 inert-cap UserWarning. That is deliberate and
    # harmless: the guard warns, never raises, and the value still wires
    # through unchanged (asserted below). See C17 for the warning contract.
    cfg = REEConfig.from_dims(
        body_obs_dim=8,
        world_obs_dim=16,
        action_dim=4,
        use_event_segmenter=True,
        use_policy_decomposition=True,
        decomposition_vs_threshold=0.7,
        decomposition_depth_cap=4,
    )
    assert cfg.use_policy_decomposition is True
    assert cfg.decomposition_vs_threshold == 0.7
    assert cfg.decomposition_depth_cap == 4
    agent = REEAgent(cfg)
    assert agent.policy_decomposition is not None
    assert agent.policy_decomposition.config.vs_decompose_threshold == 0.7
    assert agent.policy_decomposition.config.depth_cap == 4


# ----------------------------------------------------------------------
# C2 -- loud precondition
# ----------------------------------------------------------------------
def test_c2_requires_event_segmenter():
    cfg = REEConfig.from_dims(
        body_obs_dim=8,
        world_obs_dim=16,
        action_dim=4,
        use_policy_decomposition=True,  # use_event_segmenter left OFF
    )
    try:
        REEAgent(cfg)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "use_event_segmenter" in str(exc)


# ----------------------------------------------------------------------
# C3 -- depth-1 decomposes to raw actions (regression pin)
# ----------------------------------------------------------------------
def test_c3_depth1_sequence_unpacks_to_raw_actions():
    tiles = PolicyDecomposition.decompose_sequence((0, 1, 2), depth=1, library=None)
    assert tiles == (((0,), 0), ((1,), 0), ((2,), 0))


# ----------------------------------------------------------------------
# C4 -- library-aware tiling at depth > 1
# ----------------------------------------------------------------------
def test_c4_tiles_against_registered_shallower_chunk():
    library = ChunkLibrary()
    sub = ChunkedPrimitive(sequence=(0, 1), depth=1, state=ChunkState.CRYSTALLISED)
    library.register(sub)
    # depth-2 sequence = [sub-chunk (0,1)] + [raw action 2]
    tiles = PolicyDecomposition.decompose_sequence((0, 1, 2), depth=2, library=library)
    assert tiles == (((0, 1), 1), ((2,), 0))


def test_c4_dissolved_chunks_are_not_tiled_against():
    library = ChunkLibrary()
    sub = ChunkedPrimitive(sequence=(0, 1), depth=1, state=ChunkState.DISSOLVED)
    library.register(sub)
    tiles = PolicyDecomposition.decompose_sequence((0, 1, 2), depth=2, library=library)
    # No live sub-chunk covers (0, 1) -> falls back to raw actions throughout.
    assert tiles == (((0,), 0), ((1,), 0), ((2,), 0))


# ----------------------------------------------------------------------
# C5 -- atomic sequence has no tiles
# ----------------------------------------------------------------------
def test_c5_atomic_sequence_returns_no_tiles():
    assert PolicyDecomposition.decompose_sequence((0,), depth=1, library=None) == ()
    assert PolicyDecomposition.decompose_sequence((), depth=1, library=None) == ()


# ----------------------------------------------------------------------
# C6 -- R1 OR trigger
# ----------------------------------------------------------------------
def test_c6_vs_alone_triggers():
    pd = PolicyDecomposition(PolicyDecompositionConfig(vs_decompose_threshold=0.5))
    seg = _FakeSegmenter(fired=False)
    decision = pd.evaluate(
        region_vs=0.1,  # below threshold
        latent_signature={"z_world": None},
        event_segmenter=seg,
        depth=1,
        sequence=(0, 1),
    )
    assert decision.should_decompose is True
    assert decision.v_s == 0.1
    assert decision.boundary_fired is False


def test_c6_boundary_alone_triggers():
    pd = PolicyDecomposition(PolicyDecompositionConfig(vs_decompose_threshold=0.1))
    seg = _FakeSegmenter(fired=True, posterior=0.9)
    decision = pd.evaluate(
        region_vs=0.9,  # comfortably above threshold
        latent_signature={"z_world": None},
        event_segmenter=seg,
        depth=1,
        sequence=(0, 1),
    )
    assert decision.should_decompose is True
    assert decision.boundary_fired is True
    assert decision.boundary_posterior == 0.9


def test_c6_neither_triggers_nothing():
    pd = PolicyDecomposition(PolicyDecompositionConfig(vs_decompose_threshold=0.1))
    seg = _FakeSegmenter(fired=False)
    decision = pd.evaluate(
        region_vs=0.9,
        latent_signature={"z_world": None},
        event_segmenter=seg,
        depth=1,
        sequence=(0, 1),
    )
    assert decision.should_decompose is False
    assert decision.decomposed is False
    assert decision.marked_unreliable is False


# ----------------------------------------------------------------------
# C7 -- depth_cap marks unreliable instead of decomposing
# ----------------------------------------------------------------------
def test_c7_depth_at_cap_marks_unreliable():
    pd = PolicyDecomposition(PolicyDecompositionConfig(vs_decompose_threshold=0.9, depth_cap=2))
    seg = _FakeSegmenter(fired=False)
    decision = pd.evaluate(
        region_vs=0.0,  # triggers
        latent_signature={"z_world": None},
        event_segmenter=seg,
        depth=2,  # == depth_cap
        sequence=(0, 1, 2),
    )
    assert decision.should_decompose is True
    assert decision.marked_unreliable is True
    assert decision.decomposed is False
    assert decision.sub_elements == ()
    assert pd.get_state()["decomp_n_marked_unreliable"] == 1
    assert pd.get_state()["decomp_n_decomposed_precommit"] == 0


# ----------------------------------------------------------------------
# C8 -- live agent, pre-commit: withheld chunk is replaced by leaf tiles
# ----------------------------------------------------------------------
def test_c8_withheld_chunk_replaced_by_leaf_tiles_in_pool():
    env, obs_dict, agent = _env_agent(
        use_event_segmenter=True,
        use_policy_chunking=True,
        use_chunk_proposal_injection=True,
        use_policy_decomposition=True,
        decomposition_vs_threshold=0.99,  # near-certain trigger
    )
    seq = (0, 1, 2)
    chunk = ChunkedPrimitive(
        sequence=seq, depth=1, state=ChunkState.CRYSTALLISED, selection_weight=1.0
    )
    agent.policy_chunking.library.register(chunk)

    obs_body, obs_world = obs_dict["body_state"], obs_dict["world_state"]
    latent = agent.sense(obs_body, obs_world)
    ticks = agent.clock.advance()
    e1_prior = (
        agent._e1_tick(latent)
        if ticks["e1_tick"]
        else torch.zeros(1, 32, device=agent.device)
    )
    agent.generate_trajectories(latent, e1_prior, ticks)

    diag = agent.hippocampal._last_propose_diagnostics
    assert diag.get("mech321_chunks_withheld", 0) >= 1
    assert diag.get("mech321_chunks_decomposed", 0) >= 1
    assert diag.get("mech321_leaf_tiles_added", 0) >= 1
    # The full chunk sequence must not survive as a single atomic candidate.
    surviving = diag.get("arc071_chunk_sequences", [])
    assert list(seq) not in surviving
    state = agent.get_policy_decomposition_state()
    assert state["decomp_n_evaluated_precommit"] > 0
    assert state["decomp_n_decomposed_precommit"] >= 1


# ----------------------------------------------------------------------
# C9 -- live agent, mid-execution: triggering remainder releases the latch
# ----------------------------------------------------------------------
def test_c9_midexecution_trigger_releases_commit_latch():
    env, obs_dict, agent = _env_agent(
        use_event_segmenter=True,
        use_policy_chunking=True,
        use_chunk_proposal_injection=True,
        use_policy_decomposition=True,
        decomposition_vs_threshold=0.99,
    )
    seq = (0, 1, 2)
    chunk = ChunkedPrimitive(
        sequence=seq, depth=1, state=ChunkState.CRYSTALLISED, selection_weight=1.0
    )
    agent.policy_chunking.library.register(chunk)

    obs_body, obs_world = obs_dict["body_state"], obs_dict["world_state"]
    latent = agent.sense(obs_body, obs_world)
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

    assert agent.beta_gate.is_elevated is False
    assert agent.get_policy_decomposition_state()["decomp_n_evaluated_midexec"] >= 1


# ----------------------------------------------------------------------
# C10 -- injection stays additive when nothing triggers
# ----------------------------------------------------------------------
def test_c10_untriggered_chunk_passes_through_unchanged():
    """Deterministic version of the passthrough guarantee: stub the
    decomposition source to always report should_decompose=False (rather
    than relying on a real, stochastically-rolled-out E2 predictor to stay
    quiet over an 8-tick sweep -- see C8/C9 for the live-segmenter path)."""
    env, obs_dict, agent = _env_agent(
        use_event_segmenter=True,
        use_policy_chunking=True,
        use_chunk_proposal_injection=True,
        use_policy_decomposition=True,
    )
    seq = (0, 1, 2)
    chunk = ChunkedPrimitive(
        sequence=seq, depth=1, state=ChunkState.CRYSTALLISED, selection_weight=1.0
    )
    agent.policy_chunking.library.register(chunk)

    fake_source = _FakeDecompositionSource(should_decompose=False)
    agent.hippocampal.set_decomposition_source(fake_source)

    obs_body, obs_world = obs_dict["body_state"], obs_dict["world_state"]
    latent = agent.sense(obs_body, obs_world)
    ticks = agent.clock.advance()
    e1_prior = (
        agent._e1_tick(latent)
        if ticks["e1_tick"]
        else torch.zeros(1, 32, device=agent.device)
    )
    agent.generate_trajectories(latent, e1_prior, ticks)

    assert fake_source.n_calls > 0
    diag = agent.hippocampal._last_propose_diagnostics
    assert diag.get("arc071_chunk_candidates_added", 0) >= 1
    surviving = diag.get("arc071_chunk_sequences", [])
    assert list(seq) in surviving
    assert diag.get("mech321_chunks_withheld", 0) == 0


def test_c10_no_decomposition_source_is_bit_identical():
    """set_decomposition_source() never called -> _apply_policy_decomposition
    is a no-op passthrough, matching pre-MECH-321 ARC-071-only behaviour."""
    env, obs_dict, agent = _env_agent(
        use_policy_chunking=True,
        use_chunk_proposal_injection=True,
        # use_policy_decomposition left OFF: agent.policy_decomposition is
        # None, set_decomposition_source() is never called.
    )
    assert agent.policy_decomposition is None
    seq = (0, 1, 2)
    chunk = ChunkedPrimitive(
        sequence=seq, depth=1, state=ChunkState.CRYSTALLISED, selection_weight=1.0
    )
    agent.policy_chunking.library.register(chunk)

    obs_body, obs_world = obs_dict["body_state"], obs_dict["world_state"]
    latent = agent.sense(obs_body, obs_world)
    ticks = agent.clock.advance()
    e1_prior = (
        agent._e1_tick(latent)
        if ticks["e1_tick"]
        else torch.zeros(1, 32, device=agent.device)
    )
    agent.generate_trajectories(latent, e1_prior, ticks)

    diag = agent.hippocampal._last_propose_diagnostics
    surviving = diag.get("arc071_chunk_sequences", [])
    assert list(seq) in surviving
    assert "mech321_chunks_withheld" not in diag


# ----------------------------------------------------------------------
# R5 bottleneck trigger mode (ARM_2) -- C11..C16
# ----------------------------------------------------------------------
def _region(*vals):
    """A latent_signature carrying a z_world tensor for the region key."""
    return {"z_world": torch.tensor([list(vals)], dtype=torch.float32)}


# Three well-separated regions (bins are 1.0 wide over the leading dims).
_A = _region(0.0, 0.0, 0.0, 0.0)
_B = _region(5.0, 5.0, 5.0, 5.0)
_C = _region(9.0, 9.0, 9.0, 9.0)


def _bottleneck_pd(**overrides):
    kw = dict(
        trigger_mode="bottleneck",
        vs_decompose_threshold=0.9,  # would fire the R1 OR trigger on any low V_s
        bottleneck_min_visits=3,
        bottleneck_min_distinct_neighbors=2,
        bottleneck_region_quant=1.0,
        bottleneck_region_dims=4,
    )
    kw.update(overrides)
    return PolicyDecomposition(PolicyDecompositionConfig(**kw))


def test_c11_trigger_mode_default_and_validation():
    assert PolicyDecompositionConfig().trigger_mode == "vs_boundary"
    # unknown mode rejected
    for bad in [
        dict(trigger_mode="nonsense"),
        dict(bottleneck_min_visits=0),
        dict(bottleneck_min_distinct_neighbors=0),
        dict(bottleneck_region_quant=0.0),
        dict(bottleneck_region_dims=0),
    ]:
        try:
            PolicyDecompositionConfig(**bad).validate()
            assert False, f"expected ValueError for {bad}"
        except ValueError:
            pass
    # from_dims forwards the bottleneck params (three-site wiring pin)
    cfg = REEConfig.from_dims(
        body_obs_dim=8,
        world_obs_dim=16,
        action_dim=4,
        use_event_segmenter=True,
        use_policy_decomposition=True,
        decomposition_trigger_mode="bottleneck",
        decomposition_bottleneck_min_visits=2,
        decomposition_bottleneck_min_distinct_neighbors=1,
        decomposition_bottleneck_region_quant=0.5,
        decomposition_bottleneck_region_dims=6,
    )
    agent = REEAgent(cfg)
    pdc = agent.policy_decomposition.config
    assert pdc.trigger_mode == "bottleneck"
    assert pdc.bottleneck_min_visits == 2
    assert pdc.bottleneck_min_distinct_neighbors == 1
    assert pdc.bottleneck_region_quant == 0.5
    assert pdc.bottleneck_region_dims == 6


def test_c12_bottleneck_mode_ignores_vs():
    """A V_s far below threshold does NOT trigger in bottleneck mode (no
    bottleneck support yet), and the V_s audit counter still increments --
    the discriminative evidence that V_s was low yet not decision-driving."""
    pd = _bottleneck_pd()
    seg = _FakeSegmenter(fired=False)
    d = pd.evaluate(
        region_vs=0.0,  # would trip R1 OR trigger outright
        latent_signature=_A,
        event_segmenter=seg,
        depth=1,
        sequence=(0, 1, 2),
    )
    assert d.trigger_mode == "bottleneck"
    assert d.should_decompose is False  # first visit -> not a bottleneck
    assert d.bottleneck_fired is False
    st = pd.get_state()
    assert st["decomp_n_vs_trigger"] == 1  # V_s WAS low (audited)
    assert st["decomp_n_bottleneck_fires"] == 0  # but did not drive the decision


def test_c13_bottleneck_one_shot_rare_vs_repeated_fires():
    pd = _bottleneck_pd()
    seg = _FakeSegmenter(fired=False)
    seq = (0, 1, 2)

    def ev(lat):
        return pd.evaluate(
            region_vs=0.0, latent_signature=lat, event_segmenter=seg,
            depth=1, sequence=seq,
        )

    # Walk B -> A -> C -> A -> B -> A: region A reaches visits=3 with two
    # distinct neighbours {B, C}. Nothing fires until that last A.
    outcomes = [ev(lat).should_decompose for lat in (_B, _A, _C, _A, _B, _A)]
    assert outcomes[-1] is True, outcomes
    assert not any(outcomes[:-1]), outcomes  # one-shot / under-gated visits stay quiet

    # A region seen exactly once never fires despite low V_s.
    pd2 = _bottleneck_pd()
    d_once = pd2.evaluate(
        region_vs=0.0, latent_signature=_A, event_segmenter=seg, depth=1, sequence=seq
    )
    assert d_once.should_decompose is False


def test_c14_bottleneck_diagnostics_surfaced():
    pd = _bottleneck_pd()
    seg = _FakeSegmenter(fired=False)
    for lat in (_B, _A, _C, _A, _B, _A):
        pd.evaluate(region_vs=0.0, latent_signature=lat, event_segmenter=seg,
                    depth=1, sequence=(0, 1, 2))
    st = pd.get_state()
    assert st["decomp_trigger_mode"] == "bottleneck"
    assert st["decomp_n_bottleneck_fires"] >= 1
    assert st["decomp_n_bottleneck_regions_tracked"] == 3  # A, B, C


def test_c15_vs_boundary_mode_never_touches_accumulator():
    """Bit-identity guard: in the default mode the bottleneck accumulator is
    never populated, even over many low-V_s evaluate() calls."""
    pd = PolicyDecomposition(PolicyDecompositionConfig(vs_decompose_threshold=0.9))
    seg = _FakeSegmenter(fired=False)
    for lat in (_A, _B, _C, _A, _B, _A):
        pd.evaluate(region_vs=0.0, latent_signature=lat, event_segmenter=seg,
                    depth=1, sequence=(0, 1, 2))
    st = pd.get_state()
    assert st["decomp_trigger_mode"] == "vs_boundary"
    assert st["decomp_n_bottleneck_fires"] == 0
    assert st["decomp_n_bottleneck_regions_tracked"] == 0


def test_c16_bottleneck_at_depth_cap_marks_unreliable():
    """The mode changes only the TRIGGER; the R3 depth_cap path downstream is
    unchanged -- a bottleneck-triggering primitive at depth_cap is marked
    unreliable, not decomposed."""
    pd = _bottleneck_pd(depth_cap=2)
    seg = _FakeSegmenter(fired=False)
    seq = (0, 1, 2)
    # Prime A into a bottleneck at depth 1 (below cap) so those calls tile...
    for lat in (_B, _A, _C, _A, _B):
        pd.evaluate(region_vs=0.0, latent_signature=lat, event_segmenter=seg,
                    depth=1, sequence=seq)
    # ...then the triggering A visit at depth == cap must mark unreliable.
    d = pd.evaluate(region_vs=0.0, latent_signature=_A, event_segmenter=seg,
                    depth=2, sequence=seq)
    assert d.should_decompose is True
    assert d.bottleneck_fired is True
    assert d.marked_unreliable is True
    assert d.decomposed is False
    assert d.sub_elements == ()


# ----------------------------------------------------------------------
# C17 -- R3 depth_cap / ARC-071 chunk_max_depth coupling guard
#
# MECH-321 scoping spike 2026-07-27 section 5a
# (REE_assembly/evidence/planning/mech321_decomposition_scale_scoping_spike_2026-07-27.md):
# decomposition_depth_cap is NOT independent. The depth it tests is read off
# traj.metadata["chunk_depth"], and ARC-071 cannot mint a chunk above
# ChunkLibrary.max_depth -- so above chunk_max_depth the cap is unreachable
# (INERT) and at 1 it drops every triggering chunk instead of re-tiling it
# (DEGENERATE). Both were silent before this guard.
# ----------------------------------------------------------------------
_INERT = "is INERT"
_DEGENERATE = "is DEGENERATE"


def _guard_warnings(**cfg_kwargs):
    """Build an agent and return only the guard's own warning messages."""
    cfg = REEConfig.from_dims(
        body_obs_dim=8,
        world_obs_dim=16,
        action_dim=4,
        use_event_segmenter=True,
        use_policy_decomposition=True,
        **cfg_kwargs,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent = REEAgent(cfg)
    msgs = [
        str(w.message)
        for w in caught
        if _INERT in str(w.message) or _DEGENERATE in str(w.message)
    ]
    return agent, msgs


def test_c17_useful_range_is_silent_and_unchanged():
    """[2, chunk_max_depth] is the useful range and must stay silent. This is
    the do-no-harm pin: the guard must not fire on any currently-valid
    configuration, including the shipped default of 3 against chunk_max_depth
    3, and must not disturb what gets wired."""
    for cap in (2, 3):
        agent, msgs = _guard_warnings(
            decomposition_depth_cap=cap, chunk_max_depth=3
        )
        assert msgs == [], "depth_cap=%d/chunk_max_depth=3 must be silent" % cap
        assert agent.policy_decomposition is not None
        assert agent.policy_decomposition.config.depth_cap == cap


def test_c17_inert_above_chunk_max_depth_warns():
    """> chunk_max_depth is unreachable: no chunk can be minted that deep, so
    the mark-unreliable-by-cap branch never runs and 4 behaves exactly like
    3. Warned, not raised -- shipped MECH-321 experiments run 4."""
    agent, msgs = _guard_warnings(decomposition_depth_cap=4, chunk_max_depth=3)
    assert len(msgs) == 1
    assert _INERT in msgs[0]
    assert "chunk_max_depth=3" in msgs[0]
    # Behaviour is unchanged by the warning -- the value still wires through.
    assert agent.policy_decomposition.config.depth_cap == 4


def test_c17_degenerate_cap_of_one_warns():
    """depth_cap == 1 disables decomposition entirely (every chunk has
    depth >= 1 -> every triggering chunk is marked unreliable rather than
    re-tiled), degenerating MECH-321 into a pure withholding mechanism.
    validate() permits it (>= 1), so only this guard surfaces it."""
    agent, msgs = _guard_warnings(decomposition_depth_cap=1, chunk_max_depth=3)
    assert len(msgs) == 1
    assert _DEGENERATE in msgs[0]
    assert agent.policy_decomposition.config.depth_cap == 1


def test_c17_guard_tracks_chunk_max_depth_not_a_hardcoded_3():
    """The ceiling is ARC-071's knob, not a constant: raising chunk_max_depth
    makes a previously-inert cap valid, and lowering it makes a previously-
    valid cap inert. This is what 'derived mirror' means operationally."""
    _, msgs_ok = _guard_warnings(decomposition_depth_cap=4, chunk_max_depth=5)
    assert msgs_ok == []
    _, msgs_inert = _guard_warnings(decomposition_depth_cap=3, chunk_max_depth=2)
    assert len(msgs_inert) == 1
    assert _INERT in msgs_inert[0]


def test_c17_predicate_is_pure_and_ascii():
    """depth_cap_config_issues() is the contract-pinnable predicate: pure,
    no agent needed, and ASCII-clean because it reaches stderr (repo rule)."""
    assert depth_cap_config_issues(3, 3) == ()
    assert depth_cap_config_issues(2, 3) == ()
    assert depth_cap_config_issues(2, 2) == ()
    assert len(depth_cap_config_issues(100, 3)) == 1
    assert len(depth_cap_config_issues(1, 3)) == 1
    # depth_cap=1 is degenerate whatever the ceiling -- reported once, and as
    # DEGENERATE (the more actionable diagnosis) rather than as INERT.
    only = depth_cap_config_issues(1, 1)
    assert len(only) == 1 and _DEGENERATE in only[0]
    for cap, ceiling in ((100, 3), (1, 3), (4, 3), (3, 2)):
        for msg in depth_cap_config_issues(cap, ceiling):
            assert msg.isascii(), "guard message must be ASCII-only: %r" % msg


def test_c17_guard_never_raises_and_is_off_when_decomposition_is_off():
    """No warning when MECH-321 itself is off: an inert cap on a config that
    never instantiates the operator is not a mis-configuration to shout
    about, and the master-OFF path must stay bit-identical (C1)."""
    cfg = REEConfig.from_dims(
        body_obs_dim=8,
        world_obs_dim=16,
        action_dim=4,
        decomposition_depth_cap=100,
        chunk_max_depth=3,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent = REEAgent(cfg)
    assert agent.policy_decomposition is None
    assert [w for w in caught if _INERT in str(w.message)] == []


def test_c17_warning_category_is_userwarning():
    """Pinned category, so a caller can filter or escalate the guard
    specifically (e.g. -W error::UserWarning in an experiment harness)."""
    cfg = REEConfig.from_dims(
        body_obs_dim=8,
        world_obs_dim=16,
        action_dim=4,
        use_event_segmenter=True,
        use_policy_decomposition=True,
        decomposition_depth_cap=1,
    )
    with pytest.warns(UserWarning, match=_DEGENERATE):
        REEAgent(cfg)


# ----------------------------------------------------------------------
# C17b -- the mirrored ceiling MOVES under ARC-071's growable depth
#
# chunk_max_depth stopped being a constant (ree-v3, ARC-071 growable depth,
# Solway 2014): under use_growable_chunk_depth it is only the STARTING value of
# a ceiling derived from the deliberation budget. The INERT test must follow it,
# or the guard would warn about a depth_cap the growing ceiling does reach.
# The mirror RELATION is unchanged -- what the cap mirrors is still ARC-071's
# depth budget; that budget simply stopped being fixed.
# ----------------------------------------------------------------------
def test_c17b_predicate_ignores_a_derived_bound_that_does_not_raise_the_ceiling():
    """None, or any value at/below chunk_max_depth, must leave the guard EXACTLY
    as it was. This is the do-no-harm pin for every shipped MECH-321 run."""
    for derived in (None, 2, 3):
        assert depth_cap_config_issues(3, 3, derived) == ()
        inert = depth_cap_config_issues(4, 3, derived)
        assert len(inert) == 1 and _INERT in inert[0]
        assert "chunk_max_depth=3" in inert[0]


def test_c17b_predicate_uses_the_derived_bound_when_it_is_higher():
    """A cap the growing ceiling will reach is NOT inert and must not warn."""
    assert depth_cap_config_issues(4, 3, 6) == ()
    assert depth_cap_config_issues(6, 3, 6) == ()
    # ...but above the derived bound it is inert again, and the message names
    # the bound that actually binds rather than blaming chunk_max_depth.
    beyond = depth_cap_config_issues(7, 3, 6)
    assert len(beyond) == 1 and _INERT in beyond[0]
    assert "derived_chunk_max_depth=6" in beyond[0]
    assert "chunk_max_depth=3" not in beyond[0]
    for msg in beyond:
        assert msg.isascii()


def test_c17b_degenerate_still_wins_and_names_the_moved_bound():
    """depth_cap == 1 is degenerate whatever the ceiling, and the remedy range
    it prints must be the one now in force."""
    only = depth_cap_config_issues(1, 3, 6)
    assert len(only) == 1 and _DEGENERATE in only[0]
    assert "derived_chunk_max_depth=6" in only[0]


def test_c17b_agent_passes_the_derived_bound_only_when_depth_grows():
    """Wiring pin. The guard sees the growable bound only when ARC-071 is on
    AND growing; otherwise it must behave as before.

    depth_cap=4 against chunk_max_depth=3 is the discriminating case: inert
    without growth, reachable with it (horizon 60 derives 6).
    """
    def warns(**kw):
        cfg = REEConfig.from_dims(
            body_obs_dim=8, world_obs_dim=16, action_dim=4,
            use_event_segmenter=True, use_policy_decomposition=True,
            decomposition_depth_cap=4, chunk_max_depth=3, **kw)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            REEAgent(cfg)
        return [str(w.message) for w in caught if _INERT in str(w.message)]

    # No chunking at all -> unchanged (still inert).
    assert len(warns()) == 1
    # Chunking on but depth NOT growable -> unchanged (still inert).
    assert len(warns(use_policy_chunking=True)) == 1
    # Depth growable at a budget that derives 6 -> 4 is reachable, so silent.
    assert warns(use_policy_chunking=True, use_growable_chunk_depth=True,
                 chunk_deliberation_horizon=60) == []
    # Growable but at REE's real budget the derivation is still 3 -> inert.
    assert len(warns(use_policy_chunking=True, use_growable_chunk_depth=True)) == 1


# ----------------------------------------------------------------------
# SD-hazard-aware-policy-decomposition (V3-EXQ-844 autopsy successor)
# ----------------------------------------------------------------------
def test_c18_harm_aware_config_defaults_off_and_from_dims_forwards():
    cfg = REEConfig.from_dims(body_obs_dim=8, world_obs_dim=16, action_dim=4)
    assert cfg.decomposition_use_harm_aware_selection is False
    assert cfg.decomposition_harm_bias_gain == 0.1
    assert cfg.decomposition_harm_bias_scale == 0.1
    assert cfg.decomposition_harm_threat_floor == 0.1
    assert cfg.decomposition_harm_threat_ref == 0.5
    assert cfg.decomposition_harm_override_w_threshold == 0.9

    cfg2 = REEConfig.from_dims(
        body_obs_dim=8, world_obs_dim=16, action_dim=4,
        use_event_segmenter=True, use_policy_decomposition=True,
        decomposition_use_harm_aware_selection=True,
        decomposition_harm_bias_gain=0.2,
        decomposition_harm_bias_scale=0.3,
        decomposition_harm_threat_floor=0.05,
        decomposition_harm_threat_ref=0.6,
        decomposition_harm_override_w_threshold=0.8,
    )
    assert cfg2.decomposition_use_harm_aware_selection is True
    agent = REEAgent(cfg2)
    pdc = agent.policy_decomposition.config
    assert pdc.use_harm_aware_selection is True
    assert pdc.harm_bias_gain == 0.2
    assert pdc.harm_bias_scale == 0.3
    assert pdc.harm_threat_floor == 0.05
    assert pdc.harm_threat_ref == 0.6
    assert pdc.harm_override_w_threshold == 0.8


# ----------------------------------------------------------------------
# C19 -- harm_threat_scale pure ramp
# ----------------------------------------------------------------------
def test_c19_harm_threat_scale_ramp():
    pd = PolicyDecomposition(
        config=PolicyDecompositionConfig(harm_threat_floor=0.2, harm_threat_ref=0.8)
    )
    assert pd.harm_threat_scale(0.0) == 0.0
    assert pd.harm_threat_scale(0.2) == 0.0
    assert pd.harm_threat_scale(0.8) == pytest.approx(1.0)
    assert pd.harm_threat_scale(5.0) == 1.0  # clamped, not extrapolated
    assert pd.harm_threat_scale(0.5) == pytest.approx(0.5)  # midpoint


def test_c19_harm_threat_scale_degenerate_ramp_is_safe():
    """ref <= floor must not divide by zero or raise -- mirrors
    InstrumentalAvoidanceGate.threat_scale's own degenerate-ramp handling."""
    pd = PolicyDecomposition(
        config=PolicyDecompositionConfig(harm_threat_floor=0.5, harm_threat_ref=0.5)
    )
    assert pd.harm_threat_scale(0.4) == 0.0
    assert pd.harm_threat_scale(0.6) == 1.0


# ----------------------------------------------------------------------
# C20 -- harm_bias graded term
# ----------------------------------------------------------------------
def test_c20_harm_bias_off_flag_and_below_floor_are_zero():
    pd_off = PolicyDecomposition(
        config=PolicyDecompositionConfig(use_harm_aware_selection=False)
    )
    assert pd_off.harm_bias(harm_penalty=10.0, z_harm_a_norm=10.0) == 0.0

    pd_on = PolicyDecomposition(
        config=PolicyDecompositionConfig(
            use_harm_aware_selection=True, harm_threat_floor=0.5, harm_threat_ref=1.0
        )
    )
    assert pd_on.harm_bias(harm_penalty=10.0, z_harm_a_norm=0.1) == 0.0


def test_c20_harm_bias_clamped_and_diagnostic_counted():
    pd = PolicyDecomposition(
        config=PolicyDecompositionConfig(
            use_harm_aware_selection=True,
            harm_threat_floor=0.0,
            harm_threat_ref=1.0,
            harm_bias_gain=1.0,
            harm_bias_scale=0.1,
        )
    )
    # w(1.0) = 1.0, gain=1.0, penalty=5.0 -> raw 5.0, clamped to harm_bias_scale.
    biased = pd.harm_bias(harm_penalty=5.0, z_harm_a_norm=1.0)
    assert biased == pytest.approx(0.1)
    assert pd._n_harm_bias_nonzero == 1

    # A small penalty under the clamp should scale through unclamped.
    small = pd.harm_bias(harm_penalty=0.05, z_harm_a_norm=1.0)
    assert small == pytest.approx(0.05)
    assert pd._n_harm_bias_nonzero == 2

    # Negative penalty is never a favourable bias -- floors at 0.
    assert pd.harm_bias(harm_penalty=-3.0, z_harm_a_norm=1.0) == 0.0
    assert pd._n_harm_bias_nonzero == 2


# ----------------------------------------------------------------------
# C21 -- select_harm_aware_leaves categorical override
# ----------------------------------------------------------------------
def test_c21_select_harm_aware_leaves_below_threshold_keeps_all():
    pd = PolicyDecomposition(
        config=PolicyDecompositionConfig(
            use_harm_aware_selection=True,
            harm_threat_floor=0.0,
            harm_threat_ref=1.0,
            harm_override_w_threshold=0.9,
        )
    )
    leaves = [("a", 0.9), ("b", 0.1), ("c", 0.5)]
    kept = pd.select_harm_aware_leaves(leaves, z_harm_a_norm=0.5)  # w=0.5 < 0.9
    assert kept == ["a", "b", "c"]
    assert pd._n_harm_override_fires == 0


def test_c21_select_harm_aware_leaves_at_threshold_keeps_lowest_penalty():
    pd = PolicyDecomposition(
        config=PolicyDecompositionConfig(
            use_harm_aware_selection=True,
            harm_threat_floor=0.0,
            harm_threat_ref=1.0,
            harm_override_w_threshold=0.9,
        )
    )
    leaves = [("a", 0.9), ("b", 0.1), ("c", 0.5)]
    kept = pd.select_harm_aware_leaves(leaves, z_harm_a_norm=0.9)  # w=0.9 >= 0.9
    assert kept == ["b"]
    assert pd._n_harm_override_fires == 1

    # A single-leaf chunk never "overrides" anything -- counter must not move.
    single = pd.select_harm_aware_leaves([("only", 0.9)], z_harm_a_norm=1.0)
    assert single == ["only"]
    assert pd._n_harm_override_fires == 1


def test_c21_select_harm_aware_leaves_off_flag_is_passthrough():
    pd = PolicyDecomposition(config=PolicyDecompositionConfig())
    leaves = [("a", 0.9), ("b", 0.1)]
    assert pd.select_harm_aware_leaves(leaves, z_harm_a_norm=999.0) == ["a", "b"]
    assert pd._n_harm_override_fires == 0


# ----------------------------------------------------------------------
# C22/C23 -- live agent pool-admission (real _apply_policy_decomposition,
# real residue_field call site; the VALENCE read itself is monkeypatched to
# a deterministic, call-order-keyed function so the test is independent of
# live E2/RBF network values and depends only on this SD's wiring).
# ----------------------------------------------------------------------
def _harm_aware_env_agent(**extra_cfg):
    env, obs_dict, agent = _env_agent(
        use_event_segmenter=True,
        use_policy_chunking=True,
        use_chunk_proposal_injection=True,
        use_policy_decomposition=True,
        decomposition_vs_threshold=0.99,  # near-certain trigger, as in C8
        decomposition_use_harm_aware_selection=True,
        decomposition_harm_threat_floor=0.0,
        decomposition_harm_threat_ref=1.0,
        decomposition_harm_override_w_threshold=0.5,
        **extra_cfg,
    )
    seq = (0, 1, 2)
    chunk = ChunkedPrimitive(
        sequence=seq, depth=1, state=ChunkState.CRYSTALLISED, selection_weight=1.0
    )
    agent.policy_chunking.library.register(chunk)

    penalties = [0.9, 0.1, 0.5]  # deterministic per-leaf-call-order penalties
    calls = {"n": 0}

    def fake_evaluate_valence(z):
        i = calls["n"] % len(penalties)
        calls["n"] += 1
        out = torch.zeros(z.shape[0], 6, dtype=z.dtype, device=z.device)
        out[:, VALENCE_HARM_DISCRIMINATIVE] = penalties[i]
        return out

    agent.hippocampal.residue_field.evaluate_valence = fake_evaluate_valence
    return env, obs_dict, agent, seq


def test_c22_harm_aware_below_threshold_keeps_all_leaves_and_tags_metadata():
    env, obs_dict, agent, seq = _harm_aware_env_agent()
    obs_body, obs_world = obs_dict["body_state"], obs_dict["world_state"]
    latent = agent.sense(obs_body, obs_world)
    latent.z_harm_a = None  # w(h)=0 -- below harm_override_w_threshold
    ticks = agent.clock.advance()
    e1_prior = (
        agent._e1_tick(latent)
        if ticks["e1_tick"]
        else torch.zeros(1, 32, device=agent.device)
    )
    agent.generate_trajectories(latent, e1_prior, ticks)

    diag = agent.hippocampal._last_propose_diagnostics
    assert diag.get("mech321_harm_aware_active") is True
    assert diag.get("mech321_harm_override_fires", 0) == 0
    n_leaves = diag["mech321_leaf_tiles_added"]
    assert n_leaves >= 2  # the seq=(0,1,2) depth-1 chunk unpacks to >=2 leaves

    committed = agent._committed_candidates
    tagged = [
        c for c in committed
        if c.metadata and "decomposition_harm_penalty" in c.metadata
    ]
    assert len(tagged) == n_leaves
    for c in tagged:
        assert c.metadata["decomposition_harm_bias"] == 0.0  # w=0 -> graded bias 0


def test_c23_harm_aware_at_threshold_keeps_only_lowest_penalty_leaf():
    env, obs_dict, agent, seq = _harm_aware_env_agent()
    obs_body, obs_world = obs_dict["body_state"], obs_dict["world_state"]
    latent = agent.sense(obs_body, obs_world)
    # High-norm z_harm_a -> w(h) saturates to 1.0 >= override_w_threshold=0.5.
    latent.z_harm_a = torch.ones(1, 8)
    ticks = agent.clock.advance()
    e1_prior = (
        agent._e1_tick(latent)
        if ticks["e1_tick"]
        else torch.zeros(1, 32, device=agent.device)
    )
    agent.generate_trajectories(latent, e1_prior, ticks)

    diag = agent.hippocampal._last_propose_diagnostics
    assert diag.get("mech321_harm_aware_active") is True
    assert diag.get("mech321_harm_override_fires", 0) == 1
    assert diag["mech321_leaf_tiles_added"] == 1

    committed = agent._committed_candidates
    tagged = [
        c for c in committed
        if c.metadata and "decomposition_harm_penalty" in c.metadata
    ]
    assert len(tagged) == 1
    # penalties = [0.9, 0.1, 0.5] -- the second leaf (0.1) must be the
    # survivor, ruling out an accidental "always keeps the first" bug.
    assert tagged[0].metadata["decomposition_harm_penalty"] == pytest.approx(0.1)


# ----------------------------------------------------------------------
# C24 -- score_bias composition in REEAgent.select_action
# ----------------------------------------------------------------------
def test_c24_decomposition_harm_bias_metadata_gathered_into_score_bias():
    env, obs_dict, agent = _env_agent(use_event_segmenter=True)
    obs_body, obs_world = obs_dict["body_state"], obs_dict["world_state"]
    latent = agent.sense(obs_body, obs_world)
    ticks = agent.clock.advance()
    e1_prior = (
        agent._e1_tick(latent)
        if ticks["e1_tick"]
        else torch.zeros(1, 32, device=agent.device)
    )
    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
    assert len(candidates) >= 2

    candidates[0].metadata = dict(candidates[0].metadata or {})
    candidates[0].metadata["decomposition_harm_bias"] = 0.05
    candidates[1].metadata = dict(candidates[1].metadata or {})
    candidates[1].metadata.pop("decomposition_harm_bias", None)

    agent.select_action(candidates, ticks)

    assert agent._last_e3_score_bias is not None
    # Differential check: any OTHER uniform bias source already active in
    # this minimal config cancels out, isolating this SD's contribution.
    delta = float(agent._last_e3_score_bias[0].item()) - float(
        agent._last_e3_score_bias[1].item()
    )
    assert delta == pytest.approx(0.05, abs=1e-5)
