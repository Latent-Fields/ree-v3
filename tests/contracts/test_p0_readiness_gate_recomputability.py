"""Contracts for `experiments._metrics.p0_readiness_gate` -- the shared P0 abort
gate used by 24 drivers.

WHY. The REE_assembly indexer
(`build_experiment_indexes._precondition_unmet`) RECOMPUTES every precondition's
`met` from the reported (measured, threshold[, direction, comparator]) tuple and
treats the recompute as AUTHORITATIVE over the author's `met`. A precondition
whose declared bounds cannot reproduce its own `met` is therefore mis-adjudicated
in one of two directions:

  FALSE_UNMET  -- a sound diagnostic wrongly flagged `precondition_unmet`.
  MISSED_UNMET -- a genuine premise failure silently cleared, run wrongly trusted.

`precondition_unmet` is in the indexer's BLOCKING_ADJUDICATIONS, so today the
impact is a wrong human-read trust flag during /governance and /failure-autopsy.

Three defects fixed 2026-07-19 and pinned here:

  (1) NaN hole (MISSED_UNMET, latent in all 24 consumers). The gate read
      `nan >= t` as False -> met=False; the indexer's floor recompute
      `nan < t` is ALSO False -> met. Confirmed on V3-EXQ-680c
      (`r1_grad_cosine_not_net_negative`, nan vs 0.0, met False, read as met).
  (2) Strictness inexpressible. The comparison was hardcoded inclusive, so a
      driver whose shipped predicate is strict got an inclusive `met` at the
      boundary, disagreeing with its own science. The indexer honours
      `comparator`; the gate now emits it.
  (3) Extra keys dropped. The returned dict was rebuilt from a fixed key set,
      discarding `comparator` and any non-bound diagnostics (four drivers were
      silently losing their `control` annotations).

The load-bearing test here is `test_round_trip_*`: gate output -> indexer
recompute -> same verdict. The second load-bearing test is
`test_legacy_calls_are_bit_identical`, which pins backward compatibility for the
24 existing callers by differentially comparing against the pre-fix
implementation over the whole legacy input space.
"""
import math
import sys
from pathlib import Path

import pytest

_REE_V3_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REE_V3_ROOT))

from experiments._metrics import (  # noqa: E402
    NON_FINITE_CEILING_SENTINEL,
    NON_FINITE_FLOOR_SENTINEL,
    P0NotReady,
    p0_readiness_gate,
)

_INDEXER_PATH = (
    _REE_V3_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / "scripts"
    / "build_experiment_indexes.py"
)


def _load_indexer():
    """Import the REE_assembly indexer by path (sibling repo, not a package)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_bei_for_p0_test", _INDEXER_PATH)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: the indexer defines dataclasses, and dataclasses
    # resolves annotations via sys.modules[cls.__module__].
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


needs_indexer = pytest.mark.skipif(
    not _INDEXER_PATH.exists(),
    reason="REE_assembly sibling checkout not present")


def _gate(checks):
    """Run the gate and return preconditions whether or not it raised."""
    try:
        return p0_readiness_gate(checks)
    except P0NotReady as exc:
        return exc.preconditions


# --------------------------------------------------------------------------- #
# The pre-fix implementation, verbatim, for the backward-compatibility diff.
# --------------------------------------------------------------------------- #
def _legacy_gate(checks):
    preconditions = []
    for c in checks:
        m = float(c["measured"])
        t = float(c["threshold"])
        direction = str(c.get("direction", "lower"))
        met = (m <= t) if direction == "upper" else (m >= t)
        preconditions.append({
            "name": str(c["name"]),
            "measured": m,
            "threshold": t,
            "direction": direction,
            "met": bool(met),
            "kind": "readiness",
        })
    return preconditions


_LEGACY_KEYS = ("name", "measured", "threshold", "direction", "met", "kind")


def _legacy_inputs():
    """The whole legacy input space: both directions x boundary/either side,
    at several magnitudes. No comparator, finite measurements -- i.e. exactly
    what the 24 existing callers pass."""
    out = []
    for direction in ("lower", "upper"):
        for t in (0.0, 1.0, -2.5, 1e6, 0.02):
            for m in (t, t - 1e-9, t + 1e-9, t - 3.0, t + 3.0, 0.0):
                out.append({"name": f"c_{direction}_{t}_{m}", "measured": m,
                            "threshold": t, "direction": direction})
    # direction omitted entirely -> defaults to "lower"
    out.append({"name": "no_direction", "measured": 0.5, "threshold": 0.5})
    return out


# --------------------------------------------------------------------------- #
# (0) Backward compatibility -- the 24 callers must be bit-identical.
# --------------------------------------------------------------------------- #
def test_legacy_calls_are_bit_identical():
    """No comparator + finite measured (every existing caller) -> byte-for-byte
    the pre-fix output on the legacy key set. Any drift here changes a shipped
    run's routing."""
    checks = _legacy_inputs()
    new = _gate(checks)
    old = _legacy_gate(checks)
    assert len(new) == len(old)
    for n, o in zip(new, old):
        assert {k: n[k] for k in _LEGACY_KEYS} == o, f"drift on {o['name']}"
        # and nothing extra was invented for a plain legacy check
        assert set(n) == set(o), f"unexpected key on {o['name']}: {set(n) - set(o)}"


def test_legacy_raise_behaviour_unchanged():
    """The raise-on-unmet contract, and the payload carried by the exception."""
    ok = [{"name": "a", "measured": 1.0, "threshold": 0.5, "direction": "lower"}]
    assert p0_readiness_gate(ok)[0]["met"] is True

    with pytest.raises(P0NotReady) as exc:
        p0_readiness_gate(ok + [
            {"name": "b", "measured": 0.1, "threshold": 0.5, "direction": "lower"},
            {"name": "c", "measured": 9.0, "threshold": 0.5, "direction": "upper"},
        ])
    assert "b" in str(exc.value) and "c" in str(exc.value)
    # payload carries ALL entries, met and unmet alike, for the manifest
    assert [p["met"] for p in exc.value.preconditions] == [True, False, False]


# --------------------------------------------------------------------------- #
# (1) NaN hole.
# --------------------------------------------------------------------------- #
def test_nan_floor_is_unmet_and_recomputable():
    p = _gate([{"name": "n", "measured": float("nan"), "threshold": 0.0,
                "direction": "lower"}])[0]
    assert p["met"] is False
    assert p["measured"] == NON_FINITE_FLOOR_SENTINEL
    assert p["measured_non_finite"] == "nan" and p["non_finite"] is True


def test_nan_ceiling_uses_the_other_sentinel():
    """A large NEGATIVE sentinel would recompute as MET against a ceiling --
    the sentinel must follow the resolved direction."""
    p = _gate([{"name": "n", "measured": float("nan"), "threshold": 1e6,
                "direction": "upper"}])[0]
    assert p["met"] is False
    assert p["measured"] == NON_FINITE_CEILING_SENTINEL


def test_infinities_are_not_substituted():
    """+/-inf already compares identically in gate and recompute; substituting
    would corrupt an honest measurement."""
    lo = _gate([{"name": "i", "measured": float("-inf"), "threshold": 0.0,
                 "direction": "lower"}])[0]
    assert lo["met"] is False and lo["measured"] == float("-inf")
    assert "non_finite" not in lo
    hi = _gate([{"name": "i", "measured": float("inf"), "threshold": 0.0,
                 "direction": "lower"}])[0]
    assert hi["met"] is True and hi["measured"] == float("inf")


# --------------------------------------------------------------------------- #
# (2) Strictness.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("comparator,direction,met_at_boundary", [
    (">=", "lower", True),
    (">", "lower", False),
    ("<=", "upper", True),
    ("<", "upper", False),
])
def test_boundary_strictness(comparator, direction, met_at_boundary):
    """measured == threshold: inclusive is met, strict is unmet."""
    p = _gate([{"name": "b", "measured": 0.25, "threshold": 0.25,
                "direction": direction, "comparator": comparator}])[0]
    assert p["met"] is met_at_boundary
    assert p["comparator"] == comparator


def test_comparator_alone_resolves_direction():
    """Mirrors the indexer, which reads comparator BEFORE direction."""
    p = _gate([{"name": "c", "measured": 5.0, "threshold": 10.0,
                "comparator": "<="}])[0]
    assert p["met"] is True   # a ceiling, despite the "lower" default


def test_bad_comparator_raises_rather_than_loosening():
    """A typo must not silently fall back to an inclusive bound."""
    with pytest.raises(ValueError, match="comparator"):
        p0_readiness_gate([{"name": "c", "measured": 1.0, "threshold": 0.0,
                            "comparator": "=>"}])


def test_two_sided_band_is_refused():
    """The gate is single-bound; a band must not be read as a one-legged floor."""
    with pytest.raises(ValueError, match="two-sided"):
        p0_readiness_gate([{"name": "b", "measured": 1.0, "threshold_low": 0.0,
                            "threshold_high": 2.0}])


# --------------------------------------------------------------------------- #
# (3) Extra keys.
# --------------------------------------------------------------------------- #
def test_extra_keys_pass_through():
    """Four drivers (716/716a/775/776) annotate their checks with `control`;
    those annotations were being dropped before reaching the manifest."""
    p = _gate([{"name": "k", "measured": 1.0, "threshold": 0.0,
                "direction": "lower", "control": "what this measures",
                "n_seeds": 3}])[0]
    assert p["control"] == "what this measures"
    assert p["n_seeds"] == 3
    assert p["kind"] == "readiness"


def test_caller_may_override_kind_but_not_met():
    p = _gate([{"name": "k", "measured": 0.0, "threshold": 1.0,
                "direction": "lower", "kind": "anchor", "met": True}])[0]
    assert p["kind"] == "anchor"
    assert p["met"] is False   # computed, never taken from the caller


# --------------------------------------------------------------------------- #
# Round-trip: gate `met` == NOT indexer-recomputed-unmet, for every entry.
# --------------------------------------------------------------------------- #
def _round_trip_cases():
    cases = list(_legacy_inputs())
    for direction, comp in (("lower", ">="), ("lower", ">"),
                            ("upper", "<="), ("upper", "<")):
        for t in (0.0, 0.25, 1e6):
            for m in (t, t - 1e-6, t + 1e-6, t - 2.0, t + 2.0):
                cases.append({"name": f"s_{direction}_{comp}_{t}_{m}",
                              "measured": m, "threshold": t,
                              "direction": direction, "comparator": comp})
        cases.append({"name": f"nan_{direction}_{comp}",
                      "measured": float("nan"), "threshold": 0.5,
                      "direction": direction, "comparator": comp})
    for direction in ("lower", "upper"):
        for m in (float("inf"), float("-inf"), float("nan")):
            cases.append({"name": f"nf_{direction}_{m}", "measured": m,
                          "threshold": 0.0, "direction": direction})
    return cases


@needs_indexer
def test_round_trip_gate_met_matches_indexer_recompute():
    """THE contract. Every entry the gate emits must be recomputable by the
    indexer, and must recompute to the gate's own verdict."""
    bei = _load_indexer()
    for p in _gate(_round_trip_cases()):
        unmet = bei._precondition_unmet(p)
        assert unmet is not None, f"{p['name']}: not recomputable by the indexer"
        assert (not unmet) == p["met"], (
            f"{p['name']}: gate met={p['met']} but indexer recomputed "
            f"unmet={unmet} from measured={p['measured']} "
            f"threshold={p['threshold']} direction={p['direction']} "
            f"comparator={p.get('comparator')}")


@needs_indexer
def test_the_680c_regression_case():
    """The confirmed mis-scoring: a NaN cosine against a 0.0 floor, met False,
    which the indexer previously read as MET."""
    bei = _load_indexer()
    p = _gate([{"name": "r1_grad_cosine_not_net_negative",
                "measured": float("nan"), "threshold": 0.0,
                "direction": "lower"}])[0]
    assert p["met"] is False
    assert bei._precondition_unmet(p) is True

    # and the pre-fix entry is what it used to be -- the hole, pinned.
    legacy = _legacy_gate([{"name": "r1_grad_cosine_not_net_negative",
                            "measured": float("nan"), "threshold": 0.0,
                            "direction": "lower"}])[0]
    assert legacy["met"] is False
    assert math.isnan(legacy["measured"])
    assert bei._precondition_unmet(legacy) is False   # MISSED_UNMET
