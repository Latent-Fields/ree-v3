"""SD-008: REEConfig.from_dims() warns once, and records explicitness, when
`alpha_world` is not passed explicitly.

WHAT THIS GUARDS. `REEConfig.from_dims(..., alpha_world=0.3)` silently defaults
`alpha_world` to 0.3 -- a value SD-008 says must be >= 0.9 (or 1.0) to avoid
encoder EMA double-smoothing suppressing event responses. An AST sweep
(REE_Working/.scratch/orch-20260924-1707/QUESTIONS.md, "OPEN for user: from_dims
alpha_world default (SD-008)") found 462 call sites across 305 files that inherit
the 0.3 default with no signal it happened -- including two runs
(V3-EXQ-1073/1075) whose governance verdict (SD-PP-B5 validated-negative) had to
be qualified after the fact because nobody could tell, from the manifest, which
alpha_world the run actually used.

DECIDED FIX (option (b) WARN-ONLY + option (c) auditability, both zero blast
radius): the default itself is UNCHANGED (still 0.3, both on the `from_dims`
kwarg and on `LatentStackConfig.alpha_world`), and `from_dims` now (1) emits one
ASCII-only warning per PROCESS the first time it is called without an explicit
`alpha_world=`, and (2) always records whether `alpha_world` was explicit on
`config.latent.alpha_world_explicit`, so any manifest whose `config` field
carries the full snapshot is retroactively auditable for this defect even
without re-running the AST sweep. Raising the default, refusing when unset, and
the separate `REEAgent.from_config` `setattr` no-op for `alpha_world`
(ree_core/agent.py ~3636) are OUT OF SCOPE here -- left for the user.

THE TEST HALF (CLAUDE.md "Negative instruments" / "The test half"): every test
below FAILS against the pre-fix `from_dims` (plain `alpha_world: float = 0.3`
signature, no warning, no `alpha_world_explicit` field) -- there is no warning
attribute or module to stub, and the explicit path is compared against the
implicit path within the same test so a no-op "warns every time" or "always
records True" implementation cannot pass silently.
"""

import warnings

import pytest

from ree_core.utils import config as config_module
from ree_core.utils.config import REEConfig


def _reset_warned_once_flag():
    """The warn-once flag is process-global by design (CLAUDE.md: "one warning
    per process", not per-test) -- reset it between tests so each test's
    "fires once" assertion is self-contained rather than order-dependent on
    whichever test ran first in this process.
    """
    config_module._ALPHA_WORLD_IMPLICIT_WARNED = False


@pytest.fixture(autouse=True)
def _isolate_alpha_world_warn_state():
    _reset_warned_once_flag()
    yield
    _reset_warned_once_flag()


def _from_dims_minimal(**kwargs):
    """Minimal REEConfig.from_dims() call -- only the three positional dims
    from_dims requires, plus whatever the test wants to vary."""
    return REEConfig.from_dims(
        body_obs_dim=10,
        world_obs_dim=20,
        action_dim=4,
        **kwargs,
    )


class TestAlphaWorldImplicitWarning:
    def test_warns_when_alpha_world_not_passed(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _from_dims_minimal()

        alpha_world_warnings = [
            w for w in caught
            if "alpha_world" in str(w.message)
        ]
        assert len(alpha_world_warnings) == 1, (
            f"expected exactly one alpha_world warning on the implicit path, "
            f"got {len(alpha_world_warnings)}: {[str(w.message) for w in caught]}"
        )
        msg = str(alpha_world_warnings[0].message)
        assert "0.3" in msg, f"warning should name the default in effect (0.3): {msg!r}"
        assert "0.9" in msg, f"warning should name the SD-008 stable floor (0.9): {msg!r}"
        # ASCII-only (CLAUDE.md "ASCII-Only in Python Output").
        assert msg.isascii(), f"warning text must be ASCII-only: {msg!r}"

    def test_does_not_warn_when_alpha_world_passed_explicitly(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _from_dims_minimal(alpha_world=0.9)

        alpha_world_warnings = [
            w for w in caught
            if "alpha_world" in str(w.message)
        ]
        assert not alpha_world_warnings, (
            f"explicit alpha_world= must not warn, got: "
            f"{[str(w.message) for w in alpha_world_warnings]}"
        )

    def test_does_not_warn_when_alpha_world_passed_explicitly_at_the_default_value(self):
        # Passing alpha_world=0.3 explicitly is a legitimate choice (keep current
        # behaviour on purpose) and must not be indistinguishable from never
        # having passed it at all.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _from_dims_minimal(alpha_world=0.3)

        alpha_world_warnings = [
            w for w in caught
            if "alpha_world" in str(w.message)
        ]
        assert not alpha_world_warnings, (
            f"explicit alpha_world=0.3 must not warn (same value as the "
            f"default, but explicitly chosen), got: "
            f"{[str(w.message) for w in alpha_world_warnings]}"
        )

    def test_warning_fires_only_once_per_process_across_many_implicit_calls(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(5):
                _from_dims_minimal()

        alpha_world_warnings = [
            w for w in caught
            if "alpha_world" in str(w.message)
        ]
        assert len(alpha_world_warnings) == 1, (
            f"expected the once-per-process flag to suppress repeats, got "
            f"{len(alpha_world_warnings)} warnings across 5 implicit calls"
        )

    def test_warning_does_not_fire_a_second_time_after_an_explicit_call(self):
        # Order independence: an explicit call after the warning already fired
        # must not un-suppress it, and must not itself warn.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _from_dims_minimal()  # fires the one warning
            _from_dims_minimal(alpha_world=0.95)  # explicit -- must stay silent
            _from_dims_minimal()  # implicit again -- already warned this process

        alpha_world_warnings = [
            w for w in caught
            if "alpha_world" in str(w.message)
        ]
        assert len(alpha_world_warnings) == 1


class TestAlphaWorldBehaviourUnchanged:
    """Behaviour must be bit-identical to the pre-fix code: the resolved value
    on an implicit call is still exactly 0.3, and an explicit call resolves to
    exactly the value the caller passed."""

    def test_implicit_alpha_world_is_still_0_3(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            config = _from_dims_minimal()
        assert config.latent.alpha_world == 0.3

    def test_explicit_alpha_world_is_honoured_exactly(self):
        config = _from_dims_minimal(alpha_world=0.9)
        assert config.latent.alpha_world == 0.9

    def test_explicit_alpha_world_at_default_value_is_honoured(self):
        config = _from_dims_minimal(alpha_world=0.3)
        assert config.latent.alpha_world == 0.3

    def test_bare_reeconfig_is_unaffected(self):
        # The 42 bare-REEConfig() sites named in the SD-008 measurement bypass
        # from_dims entirely; this fix must not touch that path at all.
        config = REEConfig()
        assert config.latent.alpha_world == 0.3


class TestAlphaWorldExplicitFlag:
    """option (c): config.latent.alpha_world_explicit records whether the LAST
    from_dims() call that built this config passed alpha_world explicitly."""

    def test_flag_is_false_on_implicit_call(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            config = _from_dims_minimal()
        assert config.latent.alpha_world_explicit is False

    def test_flag_is_true_on_explicit_call(self):
        config = _from_dims_minimal(alpha_world=0.9)
        assert config.latent.alpha_world_explicit is True

    def test_flag_is_true_on_explicit_call_at_the_default_value(self):
        # The flag tracks EXPLICITNESS, not "differs from the default" -- an
        # explicit alpha_world=0.3 must still read back True.
        config = _from_dims_minimal(alpha_world=0.3)
        assert config.latent.alpha_world_explicit is True

    def test_flag_is_none_on_a_bare_reeconfig(self):
        # A config never built via from_dims() carries no verdict either way --
        # None is the honest "never set" signal, distinct from False.
        config = REEConfig()
        assert config.latent.alpha_world_explicit is None

    def test_flag_survives_into_a_dataclass_asdict_snapshot(self):
        # This is what makes the flag auditable from a manifest's `config` field
        # (experiments/_lib/manifest_core.py stamp_recording_core(): "config...
        # Recorded verbatim under `config`") -- at least one driver
        # (v3_exq_876a_mech025_doing_mode_convergence_redesign.py) snapshots via
        # dataclasses.asdict(config), which only ever includes real dataclass
        # fields, never ad-hoc post-hoc attributes.
        import dataclasses

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            config = _from_dims_minimal()
        snapshot = dataclasses.asdict(config)
        assert snapshot["latent"]["alpha_world_explicit"] is False
        assert snapshot["latent"]["alpha_world"] == 0.3
