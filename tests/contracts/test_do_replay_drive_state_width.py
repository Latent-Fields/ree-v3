"""Contracts for the SD-014 / MECH-203 replay drive_state WIDTH.

Substrate: ree_core/agent.py::REEAgent._do_replay and
ree_core/residue/field.py::ResidueField.get_valence_priority.

CONFIRMED DEFECT (2026-08-03, found by the V3-EXQ-887 SD-014 GOV-CONFIRM-1
Step 2.5a substrate probe). _do_replay built a FOUR-element drive_state

    drive_state = torch.tensor([t5ht, 0.5, 1.0 - t5ht, surprise_weight])

and handed it to HippocampalModule.replay() -> _select_valence_weighted_start()
-> ResidueField.get_valence_priority(z, drive_state), which dots it against
evaluate_valence()'s [batch, VALENCE_DIM] output. MECH-307 (2026-05-11) widened
VALENCE_DIM from 4 to 6 (VALENCE_POSITIVE_SURPRISE=4, VALENCE_NEGATIVE_SURPRISE=5)
and this construction was not widened with it, so the broadcast raised

    RuntimeError: The size of tensor a (6) must match the size of tensor b (4)
                  at non-singleton dimension 1

with NO try/except anywhere on that path. Consequence: the AGENT-LEVEL
valence-weighted replay entry point -- the live consumer of SD-014 replay
prioritisation and MECH-203/MECH-205 -- could not run at all whenever serotonin
or surprise_gated_replay was enabled.

The contracts here are:
  (a) the drive_state _do_replay builds is EXACTLY VALENCE_DIM wide, on every
      branch that builds one -- this is the pin that makes a future widening of
      the valence vector fail loudly at test time instead of at runtime;
  (b) the four SD-014 components land at their named indices with the values the
      MECH-203/MECH-205 formula specifies (asserted on the weight vector itself,
      upstream of any sampling or discrete selection -- per REE_Working/CLAUDE.md
      "Running the test suite", cross-machine torch.multinomial divergence makes
      exact-selection assertions unportable);
  (c) the MECH-307 split-surprise channels are supplied DELIBERATELY at zero
      weight in BOTH flag states. The write site routes surprise magnitude to the
      split channels AND to the legacy VALENCE_SURPRISE slot, so the split
      channels are a decomposition of a magnitude this vector already weights --
      non-zero weight there would double-count surprise. Zero also makes the
      prioritisation bit-identical to the intended 4-component SD-014 formula;
  (d) the end-to-end agent replay path EXECUTES -- the direct regression test for
      the RuntimeError above -- and reaches get_valence_priority at full width
      (pad counter stays 0), i.e. the agent is fixed at the source, not rescued
      by the defensive padding in (e);
  (e) get_valence_priority zero-pads a genuinely legacy narrow drive_state and
      RECORDS doing so, but does NOT coerce a mis-sized one: longer-than-
      VALENCE_DIM and non-1-D still raise, so the pad cannot mask a real bug.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.residue.field import (
    VALENCE_DIM,
    VALENCE_HARM_DISCRIMINATIVE,
    VALENCE_LIKING,
    VALENCE_NEGATIVE_SURPRISE,
    VALENCE_POSITIVE_SURPRISE,
    VALENCE_SURPRISE,
    VALENCE_WANTING,
    ResidueField,
)
from ree_core.utils.config import REEConfig, ResidueConfig


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _build_agent(seed: int = 7, **cfg_overrides):
    torch.manual_seed(seed)
    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        action_dim=4,
        self_dim=16,
        world_dim=16,
        tonic_5ht_enabled=cfg_overrides.pop("tonic_5ht_enabled", True),
    )
    for k, v in cfg_overrides.items():
        assert hasattr(cfg, k), f"unknown config knob in test: {k}"
        setattr(cfg, k, v)
    agent = REEAgent(cfg)
    agent.reset()
    return agent


def _fill_theta_buffer(agent, n: int = 5, seed: int = 7):
    """Populate theta_buffer so _do_replay has a >1-entry window to score.

    _select_valence_weighted_start only engages when the buffer holds more than
    one entry, so a single sense() would silently take the most-recent fallback
    and never exercise get_valence_priority at all.
    """
    torch.manual_seed(seed)
    for _ in range(n):
        agent.sense(torch.randn(1, 12), torch.randn(1, 250))
        agent.theta_buffer.update(
            agent._current_latent.z_world, agent._current_latent.z_self
        )
    assert agent.theta_buffer.recent is not None
    assert agent.theta_buffer.recent.shape[0] > 1
    return agent


def _capture_drive_state(agent):
    """Run _do_replay once, returning the drive_state it handed to hippocampal.

    Spies on BOTH replay entry points (plain and MECH-165 diverse) so the same
    helper covers either branch of the replay_diversity_enabled switch.
    """
    captured = {}

    def _spy(name):
        orig = getattr(agent.hippocampal, name)

        def wrapper(*args, **kwargs):
            captured["called"] = name
            captured["drive_state"] = kwargs.get("drive_state")
            return orig(*args, **kwargs)

        return wrapper

    agent.hippocampal.replay = _spy("replay")
    agent.hippocampal.diverse_replay = _spy("diverse_replay")
    agent._do_replay(agent._current_latent)
    assert "called" in captured, "_do_replay did not reach a hippocampal replay entry point"
    return captured


def _mk_field():
    return ResidueField(ResidueConfig())


# ----------------------------------------------------------------------
# (a) width -- the pin that makes a future widening fail loudly
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "label,overrides",
    [
        ("serotonin_on", dict(tonic_5ht_enabled=True)),
        ("surprise_gated", dict(tonic_5ht_enabled=False, surprise_gated_replay=True)),
        ("both", dict(tonic_5ht_enabled=True, surprise_gated_replay=True)),
        (
            "diverse_replay",
            dict(tonic_5ht_enabled=True, replay_diversity_enabled=True),
        ),
        (
            "mech307_split_surprise",
            dict(tonic_5ht_enabled=True, use_mech307_split_surprise=True),
        ),
    ],
)
def test_do_replay_drive_state_is_exactly_valence_dim_wide(label, overrides):
    """Every branch that BUILDS a drive_state must build it at VALENCE_DIM.

    This is the contract the confirmed defect violated: 4 built against a 6-wide
    valence store. It is asserted against the imported VALENCE_DIM, not a
    literal, so widening the valence vector without widening _do_replay fails
    here at test time rather than as a runtime RuntimeError in a live run.
    """
    agent = _fill_theta_buffer(_build_agent(**overrides))
    ds = _capture_drive_state(agent)["drive_state"]
    assert ds is not None, f"{label}: expected a drive_state, got None"
    assert ds.dim() == 1, f"{label}: drive_state must be 1-D, got {tuple(ds.shape)}"
    assert ds.numel() == VALENCE_DIM, (
        f"{label}: drive_state width {ds.numel()} != VALENCE_DIM {VALENCE_DIM}. "
        "This is the MECH-307 width drift that broke agent-level valence-weighted "
        "replay -- widen the construction in _do_replay by index."
    )


def test_do_replay_builds_no_drive_state_when_both_switches_off():
    """Negative control: with serotonin and surprise_gated_replay both off, the
    valence-weighted path is not meant to engage at all (drive_state stays None
    -> most-recent-entry fallback). Without this, a construction that always
    built a VALENCE_DIM vector would pass (a) while silently turning MECH-092
    plain replay into MECH-203 prioritised replay everywhere."""
    agent = _fill_theta_buffer(
        _build_agent(tonic_5ht_enabled=False, surprise_gated_replay=False)
    )
    assert _capture_drive_state(agent)["drive_state"] is None


# ----------------------------------------------------------------------
# (b) the SD-014 components land at their named indices
# ----------------------------------------------------------------------
def test_sd014_components_are_at_their_named_indices():
    """[wanting, liking, harm_discriminative, surprise] = [t5ht, 0.5, 1-t5ht, w_s].

    Asserted on the weight vector (a shape/value assertion upstream of any
    discrete selection), not on which buffer entry gets replayed.
    """
    agent = _fill_theta_buffer(_build_agent(tonic_5ht_enabled=True))
    t5ht = float(agent.serotonin.tonic_5ht)
    ds = _capture_drive_state(agent)["drive_state"]
    assert float(ds[VALENCE_WANTING]) == pytest.approx(t5ht)
    assert float(ds[VALENCE_LIKING]) == pytest.approx(0.5)
    assert float(ds[VALENCE_HARM_DISCRIMINATIVE]) == pytest.approx(1.0 - t5ht)
    # surprise_gated_replay is off here, so the MECH-205 default holds
    assert float(ds[VALENCE_SURPRISE]) == pytest.approx(0.3)


def test_serotonin_off_zeroes_wanting_and_saturates_harm():
    """t5ht is 0.0 when serotonin is disabled, so the wanting/harm split is the
    degenerate [0, 0.5, 1.0, ...] -- pinned because it is the branch the
    surprise_gated_replay-only path takes."""
    agent = _fill_theta_buffer(
        _build_agent(tonic_5ht_enabled=False, surprise_gated_replay=True)
    )
    ds = _capture_drive_state(agent)["drive_state"]
    assert float(ds[VALENCE_WANTING]) == pytest.approx(0.0)
    assert float(ds[VALENCE_HARM_DISCRIMINATIVE]) == pytest.approx(1.0)


# ----------------------------------------------------------------------
# (c) MECH-307 split channels are deliberately zero-weighted
# ----------------------------------------------------------------------
@pytest.mark.parametrize("split_surprise", [False, True])
def test_mech307_split_channels_are_zero_weighted_in_both_flag_states(split_surprise):
    """The split-surprise channels carry ZERO replay-drive weight either way.

    OFF: those channels are never written, so any weight is a no-op and zero
    keeps the prioritisation bit-identical to the 4-component SD-014 formula.
    ON:  the write site routes the surprise magnitude to the split channel AND
    writes the same magnitude to the legacy VALENCE_SURPRISE slot (explicitly,
    so magnitude-readers stay bit-identical). Weighting the split channels here
    would therefore DOUBLE-COUNT surprise. Changing this is a claims-level
    decision about MECH-203 semantics, not a substrate tidy-up -- which is why
    it is pinned rather than left to whoever next reads the vector.
    """
    agent = _fill_theta_buffer(
        _build_agent(
            tonic_5ht_enabled=True, use_mech307_split_surprise=split_surprise
        )
    )
    ds = _capture_drive_state(agent)["drive_state"]
    assert float(ds[VALENCE_POSITIVE_SURPRISE]) == pytest.approx(0.0)
    assert float(ds[VALENCE_NEGATIVE_SURPRISE]) == pytest.approx(0.0)


def test_zero_weighted_split_channels_leave_priority_equal_to_four_component():
    """The zero weights are not merely cosmetic: a VALENCE_DIM-wide drive_state
    whose trailing channels are zero must give the SAME priority as the intended
    4-component dot product, even when the split channels hold non-zero valence.
    This is the "bit-identical to the intended prioritisation" claim, checked
    rather than asserted."""
    field = _mk_field()
    z = torch.randn(4, field.config.world_dim)
    torch.manual_seed(3)
    wide = torch.rand(VALENCE_DIM)
    wide[VALENCE_POSITIVE_SURPRISE] = 0.0
    wide[VALENCE_NEGATIVE_SURPRISE] = 0.0
    valence = field.evaluate_valence(z)
    four_component = (valence[:, :4] * wide[:4].unsqueeze(0)).sum(dim=-1) + 1e-6
    assert torch.allclose(field.get_valence_priority(z, wide), four_component)


# ----------------------------------------------------------------------
# (d) the end-to-end regression: the path actually executes
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "label,overrides",
    [
        ("serotonin_on", dict(tonic_5ht_enabled=True)),
        ("surprise_gated", dict(tonic_5ht_enabled=False, surprise_gated_replay=True)),
        (
            "diverse_replay",
            dict(tonic_5ht_enabled=True, replay_diversity_enabled=True),
        ),
    ],
)
def test_agent_valence_weighted_replay_path_executes(label, overrides):
    """Direct regression test for the confirmed RuntimeError.

    Before the fix this raised out of the broadcast in get_valence_priority,
    uncaught, on every quiescent replay cycle with either switch enabled.
    """
    agent = _fill_theta_buffer(_build_agent(**overrides))
    agent._do_replay(agent._current_latent)  # must not raise


def test_agent_reaches_get_valence_priority_at_full_width():
    """The agent must be fixed AT THE SOURCE, not rescued by the defensive pad.

    If _do_replay ever regresses to a narrow vector, the pad in (e) would keep
    the run alive and this contract is the only thing that would notice.
    """
    agent = _fill_theta_buffer(_build_agent(tonic_5ht_enabled=True))
    calls = {"n": 0}
    orig = agent.residue_field.get_valence_priority

    def _spy(z_world, drive_state, *a, **kw):
        calls["n"] += 1
        assert drive_state.numel() == VALENCE_DIM
        return orig(z_world, drive_state, *a, **kw)

    agent.residue_field.get_valence_priority = _spy
    agent._do_replay(agent._current_latent)
    assert calls["n"] > 0, (
        "valence-weighted start selection never reached get_valence_priority -- "
        "the test is passing vacuously"
    )
    assert agent.residue_field.valence_priority_pad_count == 0, (
        "_do_replay is relying on the legacy narrow-drive_state padding; "
        "the drive_state must be built at VALENCE_DIM in the first place"
    )


# ----------------------------------------------------------------------
# (e) get_valence_priority: pad the legacy, refuse the mis-sized
# ----------------------------------------------------------------------
def test_legacy_narrow_drive_state_is_zero_padded_not_rejected():
    """A pre-MECH-307 4-element caller gets the semantics it intended."""
    field = _mk_field()
    z = torch.randn(3, field.config.world_dim)
    narrow = torch.tensor([0.4, 0.5, 0.6, 0.3])
    explicit = torch.cat([narrow, torch.zeros(VALENCE_DIM - narrow.numel())])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        padded = field.get_valence_priority(z, narrow)
        reference = field.get_valence_priority(z, explicit)
    assert torch.equal(padded, reference)


def test_the_pad_is_recorded_so_it_cannot_pass_silently():
    """Padding is a compatibility shim, not a licence -- it must leave a trace."""
    field = _mk_field()
    z = torch.randn(2, field.config.world_dim)
    assert field.valence_priority_pad_count == 0
    assert field.last_valence_priority_pad_width is None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        field.get_valence_priority(z, torch.ones(4))
    assert field.valence_priority_pad_count == 1
    assert field.last_valence_priority_pad_width == 4
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)
    # Full-width calls must not touch the counter.
    field.get_valence_priority(z, torch.ones(VALENCE_DIM))
    assert field.valence_priority_pad_count == 1


def test_drive_state_longer_than_valence_dim_still_raises():
    """Refusing to truncate is the point: silently dropping trailing components
    would remove the MECH-307 channels from prioritisation without saying so."""
    field = _mk_field()
    z = torch.randn(2, field.config.world_dim)
    with pytest.raises(ValueError, match="longer than the valence"):
        field.get_valence_priority(z, torch.ones(VALENCE_DIM + 1))


def test_non_1d_drive_state_still_raises():
    field = _mk_field()
    z = torch.randn(2, field.config.world_dim)
    with pytest.raises(ValueError, match="must be 1-D"):
        field.get_valence_priority(z, torch.ones(2, VALENCE_DIM))


def test_pad_counters_are_not_in_state_dict():
    """The instrumentation must stay plain python attrs -- registering them as
    buffers would change checkpoint compatibility for every saved ResidueField."""
    keys = set(_mk_field().state_dict().keys())
    for k in (
        "valence_priority_pad_count",
        "last_valence_priority_pad_width",
        "_valence_priority_pad_warned",
    ):
        assert k not in keys
