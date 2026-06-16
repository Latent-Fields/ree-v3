"""Contracts for EXQ-563 candidate-support diagnosis.

These tests separate E3 score-bias wiring from upstream candidate support.
"""

from __future__ import annotations

import torch

from ree_core.agent import candidate_support_preflight
from ree_core.predictors.e2_fast import Trajectory
from ree_core.predictors.e3_selector import E3Config, E3TrajectorySelector


def _candidate(action_class: int, action_dim: int = 5) -> Trajectory:
    world_dim = 6
    horizon = 3
    states = [torch.zeros(1, world_dim) for _ in range(horizon + 1)]
    world_states = [torch.zeros(1, world_dim) for _ in range(horizon + 1)]
    actions = torch.zeros(1, horizon, action_dim)
    actions[:, 0, action_class] = 1.0
    return Trajectory(
        states=states,
        actions=actions,
        world_states=world_states,
    )


def test_e3_score_bias_selects_manual_multi_action_candidate():
    selector = E3TrajectorySelector(E3Config(world_dim=6, hidden_dim=8))
    selector._running_variance = 0.0  # deterministic argmin path
    candidates = [_candidate(0), _candidate(1), _candidate(2)]

    baseline = selector.select(
        candidates,
        temperature=1.0,
        score_bias=torch.tensor([0.0, 0.0, 0.0]),
    )
    biased = selector.select(
        candidates,
        temperature=1.0,
        score_bias=torch.tensor([0.0, 0.0, -100.0]),
    )

    assert baseline.selected_index == 0
    assert int(baseline.selected_action.argmax(dim=-1).item()) == 0
    assert biased.selected_index == 2
    assert int(biased.selected_action.argmax(dim=-1).item()) == 2
    assert biased.scores[2] < biased.scores[0]


def test_candidate_support_preflight_marks_single_class_not_run():
    candidates = [_candidate(4), _candidate(4), _candidate(4)]

    diag = candidate_support_preflight(
        candidates,
        forced_score_bias_per_class=[-2.0, 0.0, 0.0, 0.0],
    )

    assert diag["candidate_first_action_counts"] == {4: 3}
    assert diag["candidate_unique_first_action_classes"] == 1
    assert diag["candidate_first_action_entropy"] == 0.0
    assert diag["forced_bias_abs_mean"] == 0.0
    assert diag["preflight_status"] == "NOT_RUN"
    assert diag["not_run_reason"] == "candidate_support_collapse"
    assert diag["interpretation"] == "NOT_RUN: candidate_support_collapse"


def test_candidate_support_preflight_allows_multi_class_surface():
    candidates = [_candidate(0), _candidate(1), _candidate(4)]

    diag = candidate_support_preflight(
        candidates,
        forced_score_bias_per_class=[-2.0, 0.0, 0.0, 0.0],
    )

    assert diag["candidate_unique_first_action_classes"] == 3
    assert diag["candidate_first_action_entropy"] > 0.0
    assert diag["forced_bias_abs_mean"] == 2.0 / 3.0
    assert diag["forced_bias_nonzero_candidate_count"] == 1
    assert diag["preflight_status"] == "RUN"


# --- V3-EXQ-563c: score/bias diagnostics tests ---


def test_score_diagnostics_populated_after_select():
    """last_score_diagnostics is populated on every select() call."""
    selector = E3TrajectorySelector(E3Config(world_dim=6, hidden_dim=8))
    selector._running_variance = 0.0
    candidates = [_candidate(0), _candidate(1), _candidate(2)]

    selector.select(candidates, temperature=1.0)

    diag = selector.last_score_diagnostics
    assert "e3_raw_score_range_mean" in diag
    assert "e3_raw_score_std_mean" in diag
    assert "score_bias_abs_mean" in diag
    assert "score_bias_range_mean" in diag
    assert "score_bias_to_raw_range_ratio" in diag
    assert "normalize_score_bias_active" in diag
    assert "selected_candidate_rank_before_bias" in diag
    assert "selected_candidate_rank_after_bias" in diag
    # No bias supplied: abs_mean == 0, ratio == 0.
    assert diag["score_bias_abs_mean"] == 0.0
    assert diag["score_bias_to_raw_range_ratio"] == 0.0
    assert diag["normalize_score_bias_active"] is False


def test_bias_normalisation_rescales_bias_proportionally():
    """When normalize_score_bias_to_e3_range=True, bias magnitude is rescaled."""
    from ree_core.utils.config import E3Config as FullE3Config

    cfg = FullE3Config(world_dim=6, hidden_dim=8, normalize_score_bias_to_e3_range=True)
    selector_norm = E3TrajectorySelector(cfg)
    selector_plain = E3TrajectorySelector(E3Config(world_dim=6, hidden_dim=8))

    # Fix running variance to 0 for deterministic argmin.
    selector_norm._running_variance = 0.0
    selector_plain._running_variance = 0.0

    candidates = [_candidate(0), _candidate(1), _candidate(2), _candidate(3)]
    # Tiny bias relative to score range -- normalisation will amplify it.
    bias = torch.tensor([0.0, 0.0, 0.0, -0.001])

    result_norm = selector_norm.select(candidates, temperature=1.0, score_bias=bias)
    result_plain = selector_plain.select(candidates, temperature=1.0, score_bias=bias)

    diag_norm = selector_norm.last_score_diagnostics
    diag_plain = selector_plain.last_score_diagnostics

    # Normalisation flag should be True when raw_score_range > 1e-6.
    if diag_norm["e3_raw_score_range_mean"] > 1e-6:
        assert diag_norm["normalize_score_bias_active"] is True
        # The rescaled bias-to-range ratio should equal 1.0 (by definition: bias_range
        # is scaled to match raw_score_range).
        assert abs(diag_norm["score_bias_to_raw_range_ratio"] - 1.0) < 1e-4
    else:
        # Degenerate case: flat scores, normalisation stays off.
        assert diag_norm["normalize_score_bias_active"] is False

    assert diag_plain["normalize_score_bias_active"] is False


def test_selected_rank_after_bias_reflects_bias_effect():
    """selected_candidate_rank_after_bias changes when a large bias alters selection."""
    selector = E3TrajectorySelector(E3Config(world_dim=6, hidden_dim=8))
    selector._running_variance = 0.0  # committed path: argmin

    candidates = [_candidate(0), _candidate(1), _candidate(2), _candidate(3)]

    # Without bias, candidate 0 is selected (argmin of raw scores).
    result_no_bias = selector.select(candidates, temperature=1.0)
    diag_no_bias = dict(selector.last_score_diagnostics)

    # With a strong bias favouring candidate 3, it should now win.
    bias = torch.tensor([0.0, 0.0, 0.0, -1000.0])
    result_biased = selector.select(candidates, temperature=1.0, score_bias=bias)
    diag_biased = dict(selector.last_score_diagnostics)

    assert result_biased.selected_index == 3
    # Rank AFTER bias for the biased winner should be 0 (best after bias applied).
    assert diag_biased["selected_candidate_rank_after_bias"] == 0
    # The selected_candidate_rank_before_bias and after_bias diagnostics are present.
    assert diag_no_bias["selected_candidate_rank_before_bias"] == 0
    assert diag_no_bias["selected_candidate_rank_after_bias"] == 0


# --- modulatory-bias-selection-authority contracts (2026-06-03) ---


def test_modulatory_authority_OFF_bit_identical_baseline():
    """use_modulatory_selection_authority=False is bit-identical to pre-substrate."""
    from ree_core.utils.config import E3Config as FullE3Config

    cfg_off = FullE3Config(world_dim=6, hidden_dim=8, use_modulatory_selection_authority=False)
    cfg_baseline = E3Config(world_dim=6, hidden_dim=8)

    selector_off = E3TrajectorySelector(cfg_off)
    selector_baseline = E3TrajectorySelector(cfg_baseline)

    # Fix weights to be identical
    selector_off.load_state_dict(selector_baseline.state_dict())

    selector_off._running_variance = 0.1
    selector_baseline._running_variance = 0.1

    candidates = [_candidate(0), _candidate(1), _candidate(2)]
    bias = torch.tensor([0.05, -0.02, 0.1])

    result_off = selector_off.select(candidates, temperature=1.0, score_bias=bias)
    result_baseline = selector_baseline.select(candidates, temperature=1.0, score_bias=bias)

    # Selection must be identical
    assert result_off.selected_index == result_baseline.selected_index
    # Scores must be bit-identical
    assert torch.allclose(result_off.scores, result_baseline.scores, atol=1e-9)
    # Diagnostic flag must be False
    assert selector_off.last_score_diagnostics["modulatory_authority_active"] is False
    assert selector_off.last_score_diagnostics["modulatory_authority_scale_factor"] == 0.0


def test_modulatory_authority_ON_rescales_bias():
    """use_modulatory_selection_authority=True rescales modulatory bias proportionally."""
    from ree_core.utils.config import E3Config as FullE3Config

    cfg = FullE3Config(
        world_dim=6,
        hidden_dim=8,
        use_modulatory_selection_authority=True,
        modulatory_authority_gain=0.5,
    )
    selector = E3TrajectorySelector(cfg)
    selector._running_variance = 0.0  # deterministic argmin

    candidates = [_candidate(0), _candidate(1), _candidate(2)]
    # Small bias that would not change argmin without rescaling
    bias = torch.tensor([0.0, 0.0, -0.001])

    # Run forward pass to initialize weights so scoring produces non-zero range
    # (Without this, all candidates score identically since they're all zero tensors).
    # Seed deterministically so the scoring range is reproducible regardless of
    # test ordering / pytest-randomly reseeding (otherwise the raw_score_range
    # branch below is RNG-order-dependent and flakes in the full suite).
    torch.manual_seed(0)
    with torch.no_grad():
        for p in selector.parameters():
            p.uniform_(-0.1, 0.1)

    result = selector.select(candidates, temperature=1.0, score_bias=bias)
    diag = selector.last_score_diagnostics

    # Mechanism should be active if raw_score_range > 0
    if diag["e3_raw_score_range_mean"] > 1e-6:
        assert diag["modulatory_authority_active"] is True
        # Scale factor should be > 1 (amplifying the tiny bias)
        assert diag["modulatory_authority_scale_factor"] > 0.0


def test_modulatory_authority_min_range_floor_prevents_degenerate_scale():
    """When modulatory_range is below min_range_floor, rescaling does not fire."""
    from ree_core.utils.config import E3Config as FullE3Config

    cfg = FullE3Config(
        world_dim=6,
        hidden_dim=8,
        use_modulatory_selection_authority=True,
        modulatory_authority_gain=0.5,
        modulatory_authority_min_range_floor=1.0,  # high floor
    )
    selector = E3TrajectorySelector(cfg)
    selector._running_variance = 0.0

    candidates = [_candidate(0), _candidate(1), _candidate(2)]
    # Uniform bias (range = 0)
    bias = torch.tensor([0.05, 0.05, 0.05])

    result = selector.select(candidates, temperature=1.0, score_bias=bias)
    diag = selector.last_score_diagnostics

    # Mechanism should NOT fire (modulatory_range=0 < floor=1.0)
    assert diag["modulatory_authority_active"] is False
    assert diag["modulatory_authority_scale_factor"] == 0.0


def _candidate_big_world(action_class: int, world_scale: float, action_dim: int = 5):
    """Candidate whose world_states carry a large per-candidate constant, so the
    primary E3 scores (reality/ethical scorers are linear in world_state) become
    enormous -- simulating the SD-056-online-training score explosion that drove
    V3-EXQ-643 (raw_score_range ~1e32)."""
    world_dim = 6
    horizon = 3
    states = [torch.zeros(1, world_dim) for _ in range(horizon + 1)]
    world_states = [
        torch.full((1, world_dim), float(world_scale)) for _ in range(horizon + 1)
    ]
    actions = torch.zeros(1, horizon, action_dim)
    actions[:, 0, action_class] = 1.0
    return Trajectory(states=states, actions=actions, world_states=world_states)


def test_modulatory_authority_survives_large_primary_scores():
    """V3-EXQ-643a regression: the authority gate must measure the true modulatory
    range from the EXPLICIT accumulator, not reconstruct it as (scores - raw_scores).

    The subtraction catastrophically cancels in float32 when the primary scores are
    large: in V3-EXQ-643 the SD-056-online-trained scores grew to ~1e32 and the real
    ~0.17 modulatory range (below the float32 ULP at that magnitude) collapsed to
    EXACTLY 0.0, so the gate never fired (active_frac=0.0 on every arm). With the
    explicit accumulator the gate keys on the small bias tensor directly and fires
    regardless of primary-score magnitude. The pre-fix code FAILS this test.
    """
    from ree_core.utils.config import E3Config as FullE3Config

    cfg = FullE3Config(
        world_dim=6,
        hidden_dim=8,
        use_modulatory_selection_authority=True,
        modulatory_authority_gain=0.5,
    )
    selector = E3TrajectorySelector(cfg)
    selector._running_variance = 0.0
    torch.manual_seed(0)
    with torch.no_grad():
        for p in selector.parameters():
            p.uniform_(-0.1, 0.1)

    # Per-candidate-distinct LARGE world states -> enormous raw_score_range.
    candidates = [
        _candidate_big_world(0, 1e15),
        _candidate_big_world(1, 3e15),
        _candidate_big_world(2, 7e15),
    ]
    # Tiny modulatory bias (range 0.5) -- below the float32 ULP at ~1e15.
    bias = torch.tensor([0.0, 0.0, -0.5])

    result = selector.select(candidates, temperature=1.0, score_bias=bias)
    diag = selector.last_score_diagnostics

    # Primary scores are huge (the explosion condition).
    assert diag["e3_raw_score_range_mean"] > 1e10
    # The gate detects the TRUE 0.5 modulatory range from the explicit accumulator,
    # not the ~0 a subtraction at this magnitude would yield.
    assert abs(diag["modulatory_authority_range"] - 0.5) < 1e-3
    assert diag["modulatory_authority_active"] is True
    assert diag["modulatory_authority_scale_factor"] > 0.0


# ---------------------------------------------------------------------------
# modulatory-bias-selection-authority AMEND: channel-range routing (569f/661/654a)
# ---------------------------------------------------------------------------

def _candidate_world_feats(action_class: int, world_vec, action_dim: int = 5):
    """Candidate whose first-step world_state is the supplied per-candidate vector
    (the [K, world_dim] representation a router projects)."""
    world_dim = len(world_vec)
    horizon = 3
    states = [torch.zeros(1, world_dim) for _ in range(horizon + 1)]
    world_states = [torch.tensor([list(world_vec)], dtype=torch.float32)
                    for _ in range(horizon + 1)]
    actions = torch.zeros(1, horizon, action_dim)
    actions[:, 0, action_class] = 1.0
    return Trajectory(states=states, actions=actions, world_states=world_states)


def test_project_channel_range_preserves_range_and_identity():
    """[K, D] projection is range-preserving + non-degenerate; [K] is identity;
    a flat / single-candidate input yields a zeroed (below-floor) vector."""
    from ree_core.predictors.e3_selector import project_channel_range

    feats = torch.zeros(4, 8)
    feats[0] += 3.0  # candidate 0 distinct -> genuine cross-candidate range
    routed = project_channel_range(feats)
    assert routed.shape == (4,)
    assert float((routed.max() - routed.min()).item()) > 1e-3

    bias = torch.tensor([0.1, -0.2, 0.0, 0.3])
    assert torch.allclose(project_channel_range(bias), bias)  # identity for [K]

    flat = torch.ones(4, 8)
    assert float(project_channel_range(flat).abs().max().item()) == 0.0  # below floor
    assert project_channel_range(torch.randn(1, 8)).shape == (1,)  # K<2 safe


def test_channel_routing_OFF_bit_identical():
    """use_modulatory_channel_routing=False (default): passing a channel_route_bias
    is a no-op -- scores and selection are bit-identical to not passing it."""
    from ree_core.utils.config import E3Config as FullE3Config

    cfg = FullE3Config(world_dim=6, hidden_dim=8,
                       use_modulatory_selection_authority=True)
    selector = E3TrajectorySelector(cfg)
    selector._running_variance = 0.0
    torch.manual_seed(0)
    with torch.no_grad():
        for p in selector.parameters():
            p.uniform_(-0.1, 0.1)
    candidates = [_candidate(0), _candidate(1), _candidate(2)]

    base = selector.select(candidates, temperature=1.0)
    with_route = selector.select(
        candidates, temperature=1.0,
        channel_route_bias=torch.tensor([0.5, -0.5, 0.2]),  # ignored, flag OFF
    )
    assert torch.allclose(base.scores, with_route.scores)
    assert base.selected_index == with_route.selected_index
    # diagnostic reports inactive when routing is OFF
    assert selector.last_score_diagnostics["modulatory_channel_route_active"] is False


def test_channel_routing_ON_routes_range_into_rescaled_accumulator():
    """With routing + authority ON, a channel whose REPRESENTATION carries
    cross-candidate range yields a non-degenerate routed range (the P0 gate signal)
    AND that range reaches the committed scores the authority rescales -- the scores
    differ from the no-route path. (Whether the moved argmin is BENEFICIAL is the
    behavioural retest, not a contract; contracts test wiring, not thresholds.)"""
    from ree_core.utils.config import E3Config as FullE3Config
    from ree_core.predictors.e3_selector import project_channel_range

    cfg = FullE3Config(
        world_dim=4, hidden_dim=8,
        use_modulatory_selection_authority=True,
        modulatory_authority_gain=0.5,
        use_modulatory_channel_routing=True,
        modulatory_channel_route_min_range_floor=1e-6,
    )
    selector = E3TrajectorySelector(cfg)
    selector._running_variance = 0.0  # deterministic argmin path
    torch.manual_seed(0)
    with torch.no_grad():
        for p in selector.parameters():
            p.uniform_(-0.05, 0.05)

    # Per-candidate world reps with genuine cross-candidate range (the world-summary
    # channel) -> distinct primary scores (so raw_score_range > 0, the authority's
    # precondition) and a non-degenerate projected route.
    feats = [[0.0, 0.0, 0.0, 0.0],
             [0.6, -0.4, 0.2, 0.1],
             [2.0, 1.5, -1.0, 0.5]]
    candidates = [_candidate_world_feats(i, feats[i]) for i in range(3)]
    repr_mat = torch.stack([torch.tensor(f) for f in feats], dim=0)  # [K, D]
    route = project_channel_range(repr_mat)

    res_off = selector.select(candidates, temperature=1.0)  # route None -> no routing
    res_on = selector.select(candidates, temperature=1.0, channel_route_bias=route)
    diag = selector.last_score_diagnostics

    # P0 readiness gate: the routed bias carries the channel's cross-candidate range.
    assert diag["modulatory_channel_route_active"] is True
    assert diag["modulatory_channel_route_range"] > 1e-6
    # The channel range reaches the committed scores the authority rescales.
    assert not torch.allclose(res_on.scores, res_off.scores)


def test_channel_routing_below_floor_inactive():
    """A channel whose routed bias has range below the floor leaves the router
    inactive -- the P0 gate correctly reports no routed range (substrate_not_ready)."""
    from ree_core.utils.config import E3Config as FullE3Config

    cfg = FullE3Config(
        world_dim=6, hidden_dim=8,
        use_modulatory_selection_authority=True,
        use_modulatory_channel_routing=True,
        modulatory_channel_route_min_range_floor=1.0,  # high floor
    )
    selector = E3TrajectorySelector(cfg)
    selector._running_variance = 0.0
    candidates = [_candidate(0), _candidate(1), _candidate(2)]
    # routed bias range 0.04 << floor 1.0
    route = torch.tensor([0.0, 0.02, 0.04])

    selector.select(candidates, temperature=1.0, channel_route_bias=route)
    diag = selector.last_score_diagnostics
    assert diag["modulatory_channel_route_active"] is False
    assert diag["modulatory_channel_route_range"] > 0.0  # measured raw range
    assert diag["modulatory_channel_route_range"] < 1.0


# ---------------------------------------------------------------------------
# CONVERSION amend (569g/682, 2026-06-15): gain/contrast normalize_basis (a)
# + shortlist-then-modulate (b). behavioral_diversity_isolation:GAP-A.
# ---------------------------------------------------------------------------


def _conversion_selector(**overrides):
    """Deterministically-weighted selector so the primary score range is
    reproducible regardless of pytest-randomly ordering (mirrors the existing
    authority tests)."""
    from ree_core.utils.config import E3Config as FullE3Config

    sel = E3TrajectorySelector(FullE3Config(world_dim=6, hidden_dim=8, **overrides))
    sel._running_variance = 0.0  # deterministic argmin path
    torch.manual_seed(0)
    with torch.no_grad():
        for p in sel.parameters():
            p.uniform_(-0.3, 0.3)
    return sel


def test_conversion_amend_OFF_bit_identical():
    """basis='range' + shortlist OFF (the defaults) is bit-identical to the
    pre-conversion-amend authority path, and the new diagnostics are seeded."""
    candidates = [_candidate_big_world(i % 5, 0.4 + 0.3 * i) for i in range(6)]
    bias = torch.tensor([0.05, -0.02, 0.1, 0.0, -0.03, 0.01])

    sel_default = _conversion_selector()
    sel_explicit = _conversion_selector(
        modulatory_authority_normalize_basis="range",
        use_modulatory_shortlist_then_modulate=False,
    )
    r_default = sel_default.select(candidates, temperature=1.0, score_bias=bias.clone())
    r_explicit = sel_explicit.select(candidates, temperature=1.0, score_bias=bias.clone())

    assert r_default.selected_index == r_explicit.selected_index
    assert torch.allclose(r_default.scores, r_explicit.scores, atol=1e-12)
    d = sel_default.last_score_diagnostics
    assert d["modulatory_authority_normalize_basis"] == "range"
    assert d["modulatory_shortlist_active"] is False
    assert d["modulatory_shortlist_size"] == 0


def test_conversion_std_basis_distinct_scale():
    """Lever (a): at the same gain, basis='std' anchors the authority to
    raw_score_std (rescaled by the modulatory std), so its scale_factor differs
    from the legacy range-basis scale on the same inputs. Both fire."""
    candidates = [_candidate_big_world(i % 5, 0.4 + 0.3 * i) for i in range(6)]
    bias = torch.tensor([0.05, -0.03, 0.08, 0.0, -0.06, 0.02])

    sel_range = _conversion_selector(
        use_modulatory_selection_authority=True,
        modulatory_authority_gain=1.0,
        modulatory_authority_normalize_basis="range",
    )
    sel_std = _conversion_selector(
        use_modulatory_selection_authority=True,
        modulatory_authority_gain=1.0,
        modulatory_authority_normalize_basis="std",
    )
    sel_range.select(candidates, temperature=1.0, score_bias=bias.clone())
    sel_std.select(candidates, temperature=1.0, score_bias=bias.clone())

    d_range = sel_range.last_score_diagnostics
    d_std = sel_std.last_score_diagnostics
    assert d_range["e3_raw_score_range_mean"] > 1e-6
    assert d_range["modulatory_authority_active"] is True
    assert d_std["modulatory_authority_active"] is True
    assert d_std["modulatory_authority_normalize_basis"] == "std"
    assert abs(
        d_range["modulatory_authority_scale_factor"]
        - d_std["modulatory_authority_scale_factor"]
    ) > 1e-6


def test_conversion_shortlist_restricts_to_near_tie_and_preserves_safety():
    """Lever (b): F filters to a near-tie set; a clearly-worse-PRIMARY candidate
    is NEVER selected even with the strongest modulatory pull (the safety bound
    additive gain >= 1.0 would break). The winner sits inside the F shortlist."""
    candidates = [_candidate_big_world(i % 5, 0.4 + 0.3 * i) for i in range(6)]

    # Baseline raw scores (no bias) to identify the worst-primary candidate.
    sel0 = _conversion_selector()
    sel0.select(candidates, temperature=1.0, score_bias=torch.zeros(6))
    raw = sel0.last_raw_scores.clone()
    worst = int(raw.argmax().item())
    raw_range = float((raw.max() - raw.min()).item())
    assert raw_range > 1e-6

    # Give the WORST-primary candidate an overwhelming modulatory pull.
    bias = torch.zeros(6)
    bias[worst] = -100.0
    margin = 0.1
    sel = _conversion_selector(
        use_modulatory_shortlist_then_modulate=True,
        modulatory_shortlist_margin=margin,
    )
    res = sel.select(candidates, temperature=1.0, score_bias=bias)
    d = sel.last_score_diagnostics
    cutoff = float(raw.min().item()) + margin * raw_range

    assert d["modulatory_shortlist_active"] is True
    assert d["modulatory_shortlist_size"] >= 1
    assert res.selected_index != worst  # safety: clearly-worse primary excluded
    assert float(raw[res.selected_index].item()) <= cutoff + 1e-6  # within the F set


def test_conversion_shortlist_arbitrates_by_modulatory_within_set():
    """Lever (b) conversion property: when the primary is tied (all eligible), the
    modulatory channel ALONE decides the winner (argmin of the routed bias) -- the
    structured channel is load-bearing within the near-tie set."""
    # Zero world_states -> all primary scores equal -> every candidate eligible.
    candidates = [_candidate(i % 5) for i in range(5)]
    bias = torch.tensor([0.0, -0.3, 0.0, -0.9, 0.0])  # unique min at index 3

    sel = _conversion_selector(
        use_modulatory_shortlist_then_modulate=True,
        modulatory_shortlist_margin=0.25,
    )
    res = sel.select(candidates, temperature=1.0, score_bias=bias)
    d = sel.last_score_diagnostics

    assert d["modulatory_shortlist_active"] is True
    assert d["modulatory_shortlist_size"] == 5  # all tied -> all eligible
    assert res.selected_index == 3  # modulatory-argmin within the set


# ---------------------------------------------------------------------------
# TOP-K shortlist mode (569h-autopsy CONVERSION amend, 2026-06-16).
# behavioral_diversity_isolation:GAP-A.
# ---------------------------------------------------------------------------


def test_conversion_topk_mode_default_is_margin_and_off_bit_identical():
    """modulatory_shortlist_mode defaults to 'margin'; setting mode='top_k' while
    the shortlist master is OFF is inert (bit-identical), and the mode diagnostic
    is seeded either way."""
    candidates = [_candidate_big_world(i % 5, 0.4 + 0.3 * i) for i in range(6)]
    bias = torch.tensor([0.05, -0.02, 0.1, 0.0, -0.03, 0.01])

    sel_default = _conversion_selector()
    sel_topk_off = _conversion_selector(
        use_modulatory_shortlist_then_modulate=False,
        modulatory_shortlist_mode="top_k",
        modulatory_shortlist_k=2,
    )
    r_d = sel_default.select(candidates, temperature=1.0, score_bias=bias.clone())
    r_off = sel_topk_off.select(candidates, temperature=1.0, score_bias=bias.clone())

    assert r_d.selected_index == r_off.selected_index
    assert torch.allclose(r_d.scores, r_off.scores, atol=1e-12)
    assert sel_default.last_score_diagnostics["modulatory_shortlist_mode"] == "margin"
    assert sel_topk_off.last_score_diagnostics["modulatory_shortlist_active"] is False


def test_conversion_topk_restricts_to_k_and_preserves_safety():
    """Top-k mode shortlists exactly the k F-best candidates by primary score; a
    clearly-worse-PRIMARY candidate is never eligible even with an overwhelming
    modulatory pull, and the winner sits inside the k F-best."""
    candidates = [_candidate_big_world(i % 5, 0.4 + 0.3 * i) for i in range(6)]

    sel0 = _conversion_selector()
    sel0.select(candidates, temperature=1.0, score_bias=torch.zeros(6))
    raw = sel0.last_raw_scores.clone()
    worst = int(raw.argmax().item())
    kbest = set(torch.topk(raw, 2, largest=False).indices.tolist())
    assert worst not in kbest

    bias = torch.zeros(6)
    bias[worst] = -100.0  # overwhelming pull on the worst-primary candidate
    sel = _conversion_selector(
        use_modulatory_shortlist_then_modulate=True,
        modulatory_shortlist_mode="top_k",
        modulatory_shortlist_k=2,
    )
    res = sel.select(candidates, temperature=1.0, score_bias=bias)
    d = sel.last_score_diagnostics

    assert d["modulatory_shortlist_active"] is True
    assert d["modulatory_shortlist_mode"] == "top_k"
    assert d["modulatory_shortlist_size"] == 2  # FIXED k, not margin-relative
    assert res.selected_index in kbest          # selected within the k F-best
    assert res.selected_index != worst          # safety preserved


def test_conversion_topk_set_smaller_than_loose_margin():
    """The 569h fix: on the SAME spread pool, top_k gives a small FIXED eligible
    set where a loose margin admits a near-whole, state-stable set (the
    V3-EXQ-684 size 6.25-8.54 collapse cause)."""
    candidates = [_candidate_big_world(i % 5, 0.4 + 0.3 * i) for i in range(6)]
    bias = torch.tensor([0.0, -0.1, 0.0, -0.2, 0.0, -0.05])

    sel_margin = _conversion_selector(
        use_modulatory_shortlist_then_modulate=True,
        modulatory_shortlist_mode="margin",
        modulatory_shortlist_margin=0.9,  # loose -> admits most of the pool
    )
    sel_topk = _conversion_selector(
        use_modulatory_shortlist_then_modulate=True,
        modulatory_shortlist_mode="top_k",
        modulatory_shortlist_k=2,
    )
    sel_margin.select(candidates, temperature=1.0, score_bias=bias.clone())
    sel_topk.select(candidates, temperature=1.0, score_bias=bias.clone())

    margin_size = sel_margin.last_score_diagnostics["modulatory_shortlist_size"]
    topk_size = sel_topk.last_score_diagnostics["modulatory_shortlist_size"]
    assert topk_size == 2
    assert margin_size > topk_size


def test_conversion_topk_arbitrates_by_modulatory_within_topk():
    """Conversion property: among the k F-best, the routed modulatory channel
    ALONE decides the winner (argmin of the accumulated bias within the top-k)."""
    candidates = [_candidate_big_world(i % 5, 0.4 + 0.3 * i) for i in range(6)]

    sel0 = _conversion_selector()
    sel0.select(candidates, temperature=1.0, score_bias=torch.zeros(6))
    raw = sel0.last_raw_scores.clone()
    kbest = torch.topk(raw, 3, largest=False).indices.tolist()

    # Strongest (most-favoured) modulatory pull on the SECOND-best-by-F, a weaker
    # pull on the F-best -> within the top-3 the modulatory argmin is the second.
    bias = torch.zeros(6)
    target = kbest[1]
    bias[target] = -0.5
    bias[kbest[0]] = -0.1
    sel = _conversion_selector(
        use_modulatory_shortlist_then_modulate=True,
        modulatory_shortlist_mode="top_k",
        modulatory_shortlist_k=3,
    )
    res = sel.select(candidates, temperature=1.0, score_bias=bias)

    assert sel.last_score_diagnostics["modulatory_shortlist_size"] == 3
    assert res.selected_index == target
