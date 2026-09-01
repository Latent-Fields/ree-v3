"""GFLAG-0051 / MECH-151 (ARC-007 option A): E3's action-object ranking channel.

[chip_ref: chip-20260901-gflag0051-e3-action-object-channel]

User decision 2026-09-01: BUILD THE CHANNEL, even to just test the idea.
substrate_queue mech151-action-bias-has-no-e3-ranking-channel is flipped off
DO_NOT_BUILD_YET by this build. See the "GAP CLOSED" note in
tests/contracts/test_exp0155_action_bias_no_scoring_authority.py's module
docstring for the finding this build closes, and the GFLAG-0051 paragraph in
ree-v3/CLAUDE.md's SD-016 section for the full writeup.

WHAT IS PINNED HERE
--------------------
1. ree_core.predictors.e3_selector.compute_action_object_alignment_bias:
   a pure per-candidate function, deliberately NOT baked into
   E3TrajectorySelector's learned-channel registries (_LCG_CHANNEL_INDEX /
   _FCG_CHANNEL_INDEX / w_chan) for this first build -- it composes into the
   existing generic score_bias accumulator the same way dACC/OFC/lateral-PFC/
   MECH-295/curiosity/orienting-decision already do, per score_trajectory()'s
   own docstring ("Per-candidate novelty / curiosity / liking / dACC biases
   enter via the score_bias kwarg of select() ... NOT inside score_trajectory").
     - ELEVATED (cue-aligned) candidates get a MORE NEGATIVE bias (favoured,
       REE's lower-is-better convention); SUPPRESSED (anti-aligned) get a
       MORE POSITIVE one -- MECH-151's own wording, literally.
     - Per-candidate by construction (each candidate's OWN action-objects
       differ), so a differential alignment changes the committed argmin --
       unlike the CEM proposal-mean shift, which cannot (V3-EXQ-571 lesson).
     - None in / None out on a missing action_bias or when no candidate
       carries action_objects -- inert, not a fabricated bias.
2. REEConfig.e3.use_action_object_bias_channel / action_object_bias_weight
   wiring (default False / 1.0) through REEConfig.from_dims.
3. agent.py composes the channel's output into dacc_score_bias ONLY behind
   the use_action_object_bias_channel guard (structural pin, not a full
   REEAgent construction -- see test_dr12_pe_confidence.py / test_arc108_*
   for why a direct E3TrajectorySelector.select() call is this codebase's
   established pattern for exercising a score_bias channel; agent.py's own
   composition-guard shape is pinned separately by grep, mirroring how
   test_exp0155_action_bias_no_scoring_authority.py pins e3_selector.py's
   consumer surface).

WHAT IS NOT PINNED HERE (out of scope for this build, reported not done)
--------------------------------------------------------------------
No experiment was queued against this channel and no claim is promoted by
it. Whether the channel moves selection in a REAL rollout -- as opposed to
this file's constructed candidates -- is untested, and the upstream
ContextMemory write-path degeneracy (action_bias_divergence ~= 0.0,
chip-20260816-implsub-contextmemory-writepath-degeneracy) still applies to
any driver that enables sd016_enabled + this flag together.
"""

from __future__ import annotations

import pathlib

import pytest
import torch

from ree_core.predictors.e2_fast import Trajectory
from ree_core.predictors.e3_selector import (
    E3Config,
    E3TrajectorySelector,
    compute_action_object_alignment_bias,
)
from ree_core.utils.config import REEConfig

AO_DIM = 4
WORLD_DIM = 6
HORIZON = 3


def _candidate(action_class: int, ao_direction, action_dim: int = 5) -> Trajectory:
    """A candidate whose action_objects sit near ``ao_direction`` at every
    horizon step (small per-step noise so get_action_object_sequence() is not
    perfectly degenerate), and whose first action is one-hot at action_class
    (mirroring the sibling DR-12 / EXQ-563 contract fixtures)."""
    states = [torch.zeros(1, WORLD_DIM) for _ in range(HORIZON + 1)]
    world_states = [torch.zeros(1, WORLD_DIM) for _ in range(HORIZON + 1)]
    actions = torch.zeros(1, HORIZON, action_dim)
    actions[:, 0, action_class] = 1.0
    ao_direction = torch.as_tensor(ao_direction, dtype=torch.float32).reshape(1, AO_DIM)
    gen = torch.Generator().manual_seed(action_class + 1)
    action_objects = [
        ao_direction + 0.01 * torch.randn(1, AO_DIM, generator=gen)
        for _ in range(HORIZON)
    ]
    return Trajectory(
        states=states,
        actions=actions,
        world_states=world_states,
        action_objects=action_objects,
    )


def _no_ao_candidate(action_class: int, action_dim: int = 5) -> Trajectory:
    states = [torch.zeros(1, WORLD_DIM) for _ in range(HORIZON + 1)]
    actions = torch.zeros(1, HORIZON, action_dim)
    actions[:, 0, action_class] = 1.0
    return Trajectory(states=states, actions=actions)


# ---------------------------------------------------------------------------
# 1. compute_action_object_alignment_bias -- pure-function correctness
# ---------------------------------------------------------------------------

def test_aligned_candidate_is_favoured_and_antialigned_is_suppressed():
    """MECH-151's own wording made literal: ELEVATED (aligned) -> lower score
    contribution (favoured); SUPPRESSED (anti-aligned) -> higher (penalised)."""
    bias = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    aligned = _candidate(0, [1.0, 0.0, 0.0, 0.0])
    antialigned = _candidate(1, [-1.0, 0.0, 0.0, 0.0])
    orthogonal = _candidate(2, [0.0, 1.0, 0.0, 0.0])

    out = compute_action_object_alignment_bias(
        [aligned, antialigned, orthogonal], bias, weight=1.0
    )

    assert out is not None
    assert out[0] < out[2] < out[1]
    assert float(out[0].item()) == pytest.approx(-1.0, abs=1e-3)
    assert float(out[1].item()) == pytest.approx(1.0, abs=1e-3)
    assert abs(float(out[2].item())) < 0.05


def test_weight_scales_the_bias_linearly():
    bias = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    cands = [_candidate(0, [1.0, 0.0, 0.0, 0.0]), _candidate(1, [-1.0, 0.0, 0.0, 0.0])]
    out_w1 = compute_action_object_alignment_bias(cands, bias, weight=1.0)
    out_w3 = compute_action_object_alignment_bias(cands, bias, weight=3.0)
    assert torch.allclose(out_w3, 3.0 * out_w1, atol=1e-5)


def test_none_action_bias_returns_none():
    cands = [_candidate(0, [1.0, 0.0, 0.0, 0.0])]
    assert compute_action_object_alignment_bias(cands, None) is None


def test_zero_norm_action_bias_returns_none():
    """No direction to align against -> inert (None), not a fabricated zero bias."""
    cands = [_candidate(0, [1.0, 0.0, 0.0, 0.0])]
    zero_bias = torch.zeros(1, AO_DIM)
    assert compute_action_object_alignment_bias(cands, zero_bias) is None


def test_no_candidate_carries_action_objects_returns_none():
    bias = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    cands = [_no_ao_candidate(0), _no_ao_candidate(1)]
    assert compute_action_object_alignment_bias(cands, bias) is None


def test_mixed_candidates_treat_missing_action_objects_as_neutral():
    """A candidate with action_objects and one without, mixed: the one
    lacking action_objects contributes exactly 0.0 (neutral), not a penalty
    or a bonus, and the overall call still returns a real tensor because at
    least one candidate carries action_objects."""
    bias = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    cands = [_candidate(0, [1.0, 0.0, 0.0, 0.0]), _no_ao_candidate(1)]
    out = compute_action_object_alignment_bias(cands, bias, weight=1.0)
    assert out is not None
    assert float(out[1].item()) == 0.0
    assert float(out[0].item()) < 0.0


def test_empty_candidate_list_returns_none():
    bias = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert compute_action_object_alignment_bias([], bias) is None


# ---------------------------------------------------------------------------
# 2. Composed as score_bias into E3TrajectorySelector.select() -- decisiveness
# ---------------------------------------------------------------------------

def test_channel_output_composed_as_score_bias_flips_the_committed_argmin():
    """Direct-selector pattern (test_dr12_pe_confidence.py / EXQ-563): compose
    the channel's output the same way agent.py does (additively into
    score_bias) and confirm it can change WHICH candidate is selected --
    the property EXP-0155 found missing end-to-end before this build."""
    selector = E3TrajectorySelector(E3Config(world_dim=WORLD_DIM, hidden_dim=8))
    selector._running_variance = 0.0  # deterministic committed argmin path

    bias_direction = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    cands = [
        _candidate(0, [-1.0, 0.0, 0.0, 0.0]),  # anti-aligned
        _candidate(1, [0.0, 1.0, 0.0, 0.0]),   # orthogonal
        _candidate(2, [1.0, 0.0, 0.0, 0.0]),   # aligned -- should win once channel is ON
    ]

    baseline = selector.select(
        cands, temperature=1.0, score_bias=torch.tensor([0.0, 0.0, 0.0]),
    )
    aob_bias = compute_action_object_alignment_bias(cands, bias_direction, weight=50.0)
    biased = selector.select(cands, temperature=1.0, score_bias=aob_bias)

    assert biased.selected_index == 2
    assert biased.scores[2] < biased.scores[0]
    assert biased.scores[2] < biased.scores[1]


def test_none_bias_direction_leaves_selection_unaffected():
    """No cue action_bias available (e.g. sd016 disabled) -> compute_action_object_alignment_bias
    returns None -> callers pass score_bias=None -> selection is exactly the
    no-channel baseline. Mirrors the agent.py `if _aob_bias is not None:` guard."""
    selector = E3TrajectorySelector(E3Config(world_dim=WORLD_DIM, hidden_dim=8))
    selector._running_variance = 0.0
    cands = [_candidate(0, [1.0, 0.0, 0.0, 0.0]), _candidate(1, [-1.0, 0.0, 0.0, 0.0])]

    baseline = selector.select(cands, temperature=1.0)
    aob_bias = compute_action_object_alignment_bias(cands, None)
    assert aob_bias is None
    with_none = selector.select(cands, temperature=1.0, score_bias=aob_bias)
    assert torch.allclose(baseline.scores, with_none.scores)
    assert baseline.selected_index == with_none.selected_index


# ---------------------------------------------------------------------------
# 3. Config wiring -- default OFF, threading through REEConfig.from_dims
# ---------------------------------------------------------------------------

def test_config_default_is_off_and_from_dims_threads_the_override():
    default_cfg = REEConfig.from_dims(body_obs_dim=8, world_obs_dim=8, action_dim=4)
    assert default_cfg.e3.use_action_object_bias_channel is False
    assert default_cfg.e3.action_object_bias_weight == 1.0

    on_cfg = REEConfig.from_dims(
        body_obs_dim=8, world_obs_dim=8, action_dim=4,
        use_action_object_bias_channel=True,
        action_object_bias_weight=2.5,
    )
    assert on_cfg.e3.use_action_object_bias_channel is True
    assert on_cfg.e3.action_object_bias_weight == 2.5


# ---------------------------------------------------------------------------
# 4. agent.py structurally gates the composition behind the config flag
# ---------------------------------------------------------------------------

def test_agent_gates_the_channel_behind_the_config_flag():
    """Structural pin (no REEAgent construction needed): the call to
    compute_action_object_alignment_bias in agent.py is textually guarded by
    an `if getattr(self.config.e3, "use_action_object_bias_channel", ...)`
    check immediately above it, so the channel is bit-identical OFF by
    construction rather than by convention."""
    root = pathlib.Path(__file__).resolve().parents[2]
    src = (root / "ree_core" / "agent.py").read_text()
    assert "compute_action_object_alignment_bias(" in src
    call_idx = src.index("compute_action_object_alignment_bias(")
    guard_idx = src.rindex(
        'getattr(self.config.e3, "use_action_object_bias_channel", False)',
        0, call_idx,
    )
    assert call_idx - guard_idx < 400, (
        "compute_action_object_alignment_bias call site has moved away from "
        "its use_action_object_bias_channel guard -- re-verify the gate still "
        "wraps it before editing this bound"
    )
