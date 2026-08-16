"""Contract: preservation fires AUTOMATICALLY when a designated life ends.

Third in the preservation chain. `test_reconstruction_record.py` pins the record
primitive, `test_preservation_capture.py` pins the explicit `preserve_life(...)`
emitter; this one pins the 2026-08-16 auto-fire seam -- the default-off
`REEConfig` designation (`preserve_on_life_end` + `preserve_archive_dir` +
`preserve_on_life_end_strict`) and the `life_scope` / `preserve_life_if_designated`
firing path in `experiments/_lib/preservation.py`.

WHY THIS EXISTS (GOV-PRESERVE-1, plan doc "Increment 1f"). The explicit emitter
is a call at the end of a driver -- exactly the line a driver that RAISES never
reaches. So under the explicit form the lives most worth preserving, the ones
that ended unexpectedly, are the ones silently dropped. Auto-fire moves the
switch onto the config so a designated life preserves itself however it ends.

THE TWO POLES THIS FILE HOLDS APART, which is the whole point of the gate:

  * ENABLED -> a record is written at life-end (normal AND raising paths), and
    it reconstructs a bit-identical birth agent from disk alone.
  * DISABLED (the default) -> a hard no-op. Nothing written, nothing captured,
    no destination touched, None returned. This is the byte-identical guarantee
    that lets a fleet-touching flag land on `main` at all: `main` is what every
    cloud worker pulls, so an undesignated run must be indistinguishable from
    the pre-2026-08-16 substrate.

AND THE ASYMMETRY BETWEEN THE TWO ERROR CLASSES, pinned as a pair because a
later simplification collapsing them would be silent in both directions:

  * MISCONFIGURATION (designated, but no destination / no seed / two
    destinations) ALWAYS raises, non-strict included. A silent skip there is
    indistinguishable from never designating the life, which is the one outcome
    the designation exists to rule out.
  * A failed WRITE warns and continues by default. Auto-fire runs at the very
    end of a completed run; a full disk must not convert a PASS into an ERROR.
    `preserve_on_life_end_strict=True` inverts that -- and is itself downgraded
    when the life is ALREADY unwinding, so a bookkeeping failure never replaces
    the cause of death.
"""

import pytest
import torch

from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from ree_core.preservation import load_record, reconstruct_config
from experiments._lib.arm_fingerprint import seeded_construct
from experiments._lib.preservation import (
    LifeScope,
    life_scope,
    life_is_designated,
    preserve_life_if_designated,
    NATURAL_END_REASON,
)

_DIMS = dict(body_obs_dim=12, world_obs_dim=250, action_dim=4)
_SEED = 4242


def _env_spec():
    return {
        "class": "ree_core.environment.causal_grid_world.CausalGridWorld",
        "params": {"seed": _SEED, "size": 9},
    }


def _cfg(**overrides):
    cfg = REEConfig.from_dims(**_DIMS)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _designated(tmp_path, **overrides):
    return _cfg(
        seed=_SEED,
        preserve_on_life_end=True,
        preserve_archive_dir=str(tmp_path),
        **overrides,
    )


def _birth(cfg):
    return seeded_construct(_SEED, lambda: REEAgent(cfg)).state_dict()


# --------------------------------------------------------------------------- #
# The default is OFF -- the byte-identical guarantee                          #
# --------------------------------------------------------------------------- #

def test_designation_defaults_to_off():
    """A config nobody touched designates nothing and names no destination."""
    for cfg in (REEConfig.from_dims(**_DIMS), _cfg()):
        assert cfg.preserve_on_life_end is False
        assert cfg.preserve_archive_dir is None
        assert cfg.preserve_on_life_end_strict is False
        assert life_is_designated(cfg) is False


def test_disabled_is_a_hard_no_op(tmp_path):
    """Flag off -> nothing written, None returned, destination untouched.

    The destination IS configured here on purpose: it is `preserve_on_life_end`
    alone that must gate the write, not the incidental absence of a directory.
    """
    cfg = _cfg(seed=_SEED, preserve_archive_dir=str(tmp_path))
    out = preserve_life_if_designated(
        config=cfg, record_id="off_life_v3", environment=_env_spec()
    )
    assert out is None
    assert list(tmp_path.iterdir()) == []


def test_disabled_life_scope_is_a_hard_no_op(tmp_path):
    cfg = _cfg(seed=_SEED, preserve_archive_dir=str(tmp_path))
    with life_scope(
        config=cfg, record_id="off_scope_v3", environment=_env_spec()
    ) as life:
        life.note(claims=["GOV-PRESERVE-1"]).aged(age_steps=7)
    assert life.fired is False
    assert life.record_path is None
    assert life.designated is False
    assert list(tmp_path.iterdir()) == []


def test_designation_reader_tolerates_a_config_without_the_fields():
    """A config round-tripped from a pre-2026-08-16 record has no such fields.

    `life_is_designated` must read False rather than raising, or loading an old
    record to inspect it would blow up.
    """

    class _OldConfig:
        pass

    assert life_is_designated(_OldConfig()) is False


# --------------------------------------------------------------------------- #
# Enabled -> a record is written at life-end                                  #
# --------------------------------------------------------------------------- #

def test_enabled_writes_a_reconstructable_record_at_life_end(tmp_path):
    cfg = _designated(tmp_path)
    with life_scope(
        config=cfg, record_id="on_life_v3", environment=_env_spec()
    ) as life:
        life.note(claims=["GOV-PRESERVE-1"], metrics={"reward": 1.0})
        life.aged(age_steps=123, phase=2)

    assert life.fired is True
    assert life.record_path is not None
    assert len(list(tmp_path.iterdir())) == 1

    rec = load_record(life.record_path)          # re-verifies integrity
    assert rec.record_id == "on_life_v3"
    assert rec.seed == _SEED
    assert rec.reason_for_ending == NATURAL_END_REASON
    assert rec.lifetime["age_steps"] == 123 and rec.lifetime["phase"] == 2
    assert rec.understanding["claims"] == ["GOV-PRESERVE-1"]

    # The point of the record: birth is reconstructable from disk alone.
    orig = _birth(cfg)
    recon = _birth(reconstruct_config(rec))
    assert orig.keys() == recon.keys()
    assert not [k for k in orig if not torch.equal(orig[k], recon[k])]


def test_seed_falls_back_to_the_config(tmp_path):
    """`config.seed` is the birth seed, so auto-fire needs no seed= argument."""
    cfg = _designated(tmp_path)
    path = preserve_life_if_designated(
        config=cfg, record_id="seedless_call_v3", environment=_env_spec()
    )
    assert load_record(path).seed == _SEED


def test_explicit_destination_beats_the_config(tmp_path):
    """So a credentialed backend never has to be serialized into every record."""
    other = tmp_path / "explicit"
    other.mkdir()
    cfg = _designated(tmp_path / "from_config")
    (tmp_path / "from_config").mkdir()
    path = preserve_life_if_designated(
        config=cfg,
        record_id="override_life_v3",
        environment=_env_spec(),
        archive_dir=str(other),
    )
    assert str(other) in str(path)
    assert len(list(other.iterdir())) == 1
    assert list((tmp_path / "from_config").iterdir()) == []


# --------------------------------------------------------------------------- #
# The abnormal end -- the capability the explicit call cannot provide         #
# --------------------------------------------------------------------------- #

def test_a_life_that_ends_by_exception_is_still_preserved(tmp_path):
    cfg = _designated(tmp_path)
    with pytest.raises(RuntimeError, match="grid collapsed"):
        with life_scope(
            config=cfg, record_id="crash_life_v3", environment=_env_spec()
        ) as life:
            life.aged(age_steps=9)
            raise RuntimeError("grid collapsed")

    assert life.fired is True
    rec = load_record(life.record_path)
    assert rec.reason_for_ending == "RuntimeError: grid collapsed"
    assert rec.lifetime["age_steps"] == 9


def test_keyboard_interrupt_ends_a_life_too(tmp_path):
    """A runner SIGTERM / operator Ctrl-C is a real cause of death, not a bug.

    KeyboardInterrupt derives from BaseException, so a naive `except Exception`
    in the scope's exit path would let this life vanish unrecorded.
    """
    cfg = _designated(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        with life_scope(
            config=cfg, record_id="interrupted_life_v3", environment=_env_spec()
        ) as life:
            raise KeyboardInterrupt()
    assert life.fired is True
    assert load_record(life.record_path).reason_for_ending.startswith("KeyboardInterrupt")


def test_an_explicit_reason_is_not_overwritten(tmp_path):
    cfg = _designated(tmp_path)
    with life_scope(
        config=cfg,
        record_id="reasoned_life_v3",
        environment=_env_spec(),
        reason_for_ending="curriculum phase 3 entry",
    ) as life:
        pass
    assert load_record(life.record_path).reason_for_ending == "curriculum phase 3 entry"


# --------------------------------------------------------------------------- #
# Misconfiguration ALWAYS raises; a failed write does not (unless strict)     #
# --------------------------------------------------------------------------- #

def test_designated_without_a_destination_raises_even_when_not_strict(tmp_path):
    cfg = _cfg(seed=_SEED, preserve_on_life_end=True)   # no archive dir
    assert cfg.preserve_on_life_end_strict is False
    with pytest.raises(ValueError, match="no destination"):
        preserve_life_if_designated(
            config=cfg, record_id="nodest_life_v3", environment=_env_spec()
        )


def test_designated_without_a_seed_raises_even_when_not_strict(tmp_path):
    cfg = _cfg(preserve_on_life_end=True, preserve_archive_dir=str(tmp_path))
    assert cfg.seed is None and cfg.preserve_on_life_end_strict is False
    with pytest.raises(ValueError, match="no seed"):
        preserve_life_if_designated(
            config=cfg, record_id="noseed_life_v3", environment=_env_spec()
        )


def test_two_destinations_raise_even_when_not_strict(tmp_path):
    cfg = _designated(tmp_path)
    with pytest.raises(ValueError, match="not both"):
        preserve_life_if_designated(
            config=cfg,
            record_id="twodest_life_v3",
            environment=_env_spec(),
            archive=object(),
            archive_dir=str(tmp_path),
        )


def test_a_failed_write_warns_and_the_life_ends_normally(tmp_path, capsys):
    """Append-only refusal (duplicate record_id) is a WRITE failure, not a crash."""
    cfg = _designated(tmp_path)
    kw = dict(config=cfg, record_id="dup_auto_v3", environment=_env_spec())
    first = preserve_life_if_designated(**kw)
    assert first is not None

    second = preserve_life_if_designated(**kw)          # must not raise
    assert second is None
    assert "auto-fire failed" in capsys.readouterr().err


def test_strict_mode_raises_on_a_failed_write(tmp_path):
    cfg = _designated(tmp_path, preserve_on_life_end_strict=True)
    kw = dict(config=cfg, record_id="dup_strict_v3", environment=_env_spec())
    preserve_life_if_designated(**kw)
    with pytest.raises(Exception):
        preserve_life_if_designated(**kw)


def test_strict_failure_never_displaces_the_cause_of_death(tmp_path, capsys):
    """A life already unwinding keeps ITS exception, even under strict mode."""
    cfg = _designated(tmp_path, preserve_on_life_end_strict=True)
    preserve_life_if_designated(
        config=cfg, record_id="clash_life_v3", environment=_env_spec()
    )
    with pytest.raises(ValueError, match="the real cause"):
        with life_scope(
            config=cfg, record_id="clash_life_v3", environment=_env_spec()
        ):
            raise ValueError("the real cause")
    assert "original exception preserved" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Shape                                                                       #
# --------------------------------------------------------------------------- #

def test_life_scope_alias_is_the_class():
    assert life_scope is LifeScope


def test_the_scope_never_suppresses_the_life_exception(tmp_path):
    """Undesignated too -- the no-op path must not swallow anything either."""
    cfg = _cfg(seed=_SEED, preserve_archive_dir=str(tmp_path))
    with pytest.raises(ZeroDivisionError):
        with life_scope(
            config=cfg, record_id="nosuppress_v3", environment=_env_spec()
        ):
            1 / 0
