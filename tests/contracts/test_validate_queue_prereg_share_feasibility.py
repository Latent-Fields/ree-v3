"""
Contract tests for the pre-registration feasibility check in validate_queue.py
(prereg_share_feasibility_lint), added 2026-07-19.

THE DEFECT IT CLOSES. An experiment could be queued with a pre-registered effect
size arithmetically inconsistent with a precondition the same run would be gated
on, and nothing checked it -- so the run burned full compute and then failed on a
gate it could never have passed.

Canonical instance: V3-EXQ-785 (MECH-463 arousal variance-amplifier decomposition)
committed, in its own config, REGIMES[1]["expected_incumbent_share"] = 1.043 while
gating that regime on "n_components_with_nontrivial_share" (>= 2 components holding
|share| > 0.01). The decomposition's shares sum to exactly 1.0 by construction, so
an incumbent share at or above unity leaves <= 0 for every other component combined
and the gate was unsatisfiable before the run started. Cost: 460s of compute, one
vacated arm, and a GREEN arm's strong finding buried under a whole-run
"substrate not ready" label until an autopsy recovered it.
Autopsy: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-785_2026-07-19.md s2a.

CONSERVATISM IS THE POINT. A check that fires on judgement calls gets routed around
and is worse than nothing. The trigger is therefore an arithmetic impossibility
only -- a pre-registered share AT OR ABOVE unity, combined with a non-triviality
count gate over a sum-to-one shares mapping. Several tests below exist specifically
to pin the cases that must NOT fire, including the near-unity margin call
(test_near_unity_share_does_not_fire) and the real 785 sibling regime at 0.823.

Branches pinned:
  (1) the 785 shape ERRORs, naming both the config key and the precondition.
  (2) the REAL 785 script on disk trips it (fixture of record, if still present).
  (3) a legitimate multi-component config (shares < 1.0) does NOT fire.
  (4) near-unity (0.995) does NOT fire -- deliberate non-implementation of the
      tighter (1-S) < floor*(K-1) inequality.
  (5) a pre-registered share >= 1.0 with NO non-triviality gate does NOT fire.
  (6) a non-triviality gate with NO pre-registered share does NOT fire.
  (7) the count gate must be over a SHARES mapping -- an identical comprehension
      over a non-share mapping does NOT fire (sum-to-one is what makes it fatal).
  (8) the comprehension must be a COUNT (elt == 1), not a magnitude sum.
  (9) unparseable source fails soft (no findings, no raise).
 (10) the finding reaches validate()'s returned ERRORS (it blocks, unlike the
      warn-only re-derive brake).
 (11) the whole experiments/ corpus has no false positives.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
sys.path.insert(0, str(REPO_ROOT))

import validate_queue  # noqa: E402


EXPERIMENTS_DIR = REPO_ROOT / "experiments"

# The 785 shape, reduced to its two load-bearing elements.
SRC_785_SHAPE = '''
REGIMES = [
    {"id": "harm_incumbent", "expected_incumbent": "harm_weighted",
     "expected_incumbent_share": 0.823},
    {"id": "entropy_incumbent", "expected_incumbent": "CH:mech341",
     "expected_incumbent_share": 1.043},
]

def gate(mean_shares):
    n_nontrivial = sum(1 for v in mean_shares.values() if abs(v) > 0.01)
    return {"name": "n_components_with_nontrivial_share",
            "measured": float(n_nontrivial), "threshold": 1.5}
'''

# Same design, but every pre-registered share leaves room for others.
SRC_LEGITIMATE = '''
REGIMES = [
    {"id": "harm_incumbent", "expected_incumbent_share": 0.823},
    {"id": "benefit_incumbent", "expected_incumbent_share": 0.61},
]

def gate(mean_shares):
    n_nontrivial = sum(1 for v in mean_shares.values() if abs(v) > 0.01)
    return {"name": "n_components_with_nontrivial_share",
            "measured": float(n_nontrivial), "threshold": 1.5}
'''


def _lint(src: str) -> list[str]:
    return validate_queue.prereg_share_feasibility_lint(src)


# ---- (1) the confirmed defect shape ---------------------------------------
def test_785_shape_is_rejected():
    findings = _lint(SRC_785_SHAPE)
    assert len(findings) == 1, f"expected exactly one finding, got {findings}"
    msg = findings[0]
    # The message must name BOTH sides of the contradiction so the author can act.
    assert "expected_incumbent_share" in msg, "config key not named"
    assert "1.043" in msg, "offending value not named"
    assert "n_components_with_nontrivial_share" in msg, "precondition not named"
    assert "0.01" in msg, "gate floor not named"


def test_785_shape_does_not_flag_the_sibling_regime():
    """0.823 is a legitimate pre-registration in the very same config."""
    assert "0.823" not in _lint(SRC_785_SHAPE)[0]


# ---- (2) the real script on disk, as fixture of record ---------------------
def test_real_785_script_trips_the_check():
    script = EXPERIMENTS_DIR / "v3_exq_785_mech463_arousal_variance_amplifier_decomp.py"
    if not script.is_file():
        pytest.skip("V3-EXQ-785 script no longer present")
    findings = _lint(script.read_text(encoding="utf-8", errors="ignore"))
    assert len(findings) == 1
    assert "1.043" in findings[0]


# ---- (3)-(8) cases that must NOT fire --------------------------------------
def test_legitimate_config_does_not_fire():
    assert _lint(SRC_LEGITIMATE) == []


def test_near_unity_share_does_not_fire():
    """0.995 with a 0.01 floor is infeasible under strict non-negativity, but real
    covariance attributions produce small negative components (785 measured
    f = -0.0013). That is a margin judgement, not an arithmetic impossibility, and
    firing on it would get the check routed around. Deliberately not implemented."""
    src = SRC_785_SHAPE.replace("1.043", "0.995")
    assert _lint(src) == []


def test_share_at_exactly_unity_does_fire():
    """S == 1.0 leaves exactly nothing; this is the boundary, and it is inclusive."""
    src = SRC_785_SHAPE.replace("1.043", "1.0")
    assert len(_lint(src)) == 1


def test_prereg_share_without_gate_does_not_fire():
    src = SRC_785_SHAPE.split("def gate")[0]
    assert _lint(src) == []


def test_gate_without_prereg_share_does_not_fire():
    src = "def gate(mean_shares):\n    return sum(1 for v in mean_shares.values() if abs(v) > 0.01)\n"
    assert _lint(src) == []


def test_count_gate_over_non_share_mapping_does_not_fire():
    """Sum-to-one is what makes >= 1.0 fatal. An identical comprehension over a
    mapping that is not a decomposition of shares carries no such constraint."""
    src = SRC_785_SHAPE.replace("mean_shares", "mean_latencies")
    assert _lint(src) == []


def test_magnitude_sum_is_not_a_count_gate():
    """sum(v for ...) totals magnitudes; only sum(1 for ...) counts components."""
    src = SRC_785_SHAPE.replace(
        "sum(1 for v in mean_shares.values() if abs(v) > 0.01)",
        "sum(v for v in mean_shares.values() if abs(v) > 0.01)",
    )
    assert _lint(src) == []


# ---- named-constant floor + label quality ----------------------------------
def test_floor_factored_into_named_constant_is_still_detected():
    """785 inlined its floor (abs(v) > 0.01); its successor 785a factored it into
    NONTRIVIAL_SHARE_FLOOR. A literal-only matcher would silently stop applying to
    the successor lineage -- the scripts most likely to inherit the defect."""
    src = "NONTRIVIAL_SHARE_FLOOR = 0.01\n" + SRC_785_SHAPE.replace(
        "abs(v) > 0.01", "abs(v) > NONTRIVIAL_SHARE_FLOOR"
    )
    findings = _lint(src)
    assert len(findings) == 1
    assert "0.01" in findings[0], "resolved floor must appear in the message"


def test_unresolvable_floor_fails_soft():
    """A computed floor is not resolved and the gate is skipped -- never guessed."""
    src = "FLOOR = compute_floor()\n" + SRC_785_SHAPE.replace(
        "abs(v) > 0.01", "abs(v) > FLOOR"
    )
    assert _lint(src) == []


def test_prose_is_not_quoted_as_the_precondition_name():
    """A docstring describing the gate matches the same regex; quoting it back as
    the precondition name sends the author to the wrong place."""
    src = ('"""P7 requires >= 2 components with non-trivial share values."""\n'
           + SRC_785_SHAPE)
    msg = _lint(src)[0]
    assert "n_components_with_nontrivial_share" in msg
    assert "P7 requires" not in msg


def test_real_785a_successor_is_clean():
    """The in-flight successor pre-registers 0.9368 and must NOT be blocked -- but
    its gate must still be DETECTED (it uses the named-constant floor), so the
    check genuinely applies to it rather than passing by omission."""
    script = EXPERIMENTS_DIR / "v3_exq_785a_mech463_arousal_exogenous_urgency_decomp.py"
    if not script.is_file():
        pytest.skip("V3-EXQ-785a script not present")
    import ast as _ast
    src = script.read_text(encoding="utf-8", errors="ignore")
    assert validate_queue._share_nontriviality_gate(_ast.parse(src)) is not None, \
        "785a's gate must be detected, else the check passes it by omission"
    assert _lint(src) == []


def test_observed_measurement_table_is_not_a_preregistration():
    """785a records an OBSERVED table containing {"share": 1.1478}. A bare 'share'
    key is a measurement, not a pre-registration, and must not be flagged."""
    src = SRC_785_SHAPE + '\nOBSERVED = {"incumbent": "CH:mech341", "share": 1.1478}\n'
    findings = _lint(src)
    assert len(findings) == 1, "only the pre-registered key should be flagged"
    assert "1.1478" not in findings[0]


# ---- (9) fail-soft ---------------------------------------------------------
def test_unparseable_source_fails_soft():
    assert _lint("def broken(:\n  pass\n") == []


# ---- (10) it blocks, via validate()'s error list ---------------------------
def test_finding_reaches_validate_errors(tmp_path, monkeypatch):
    """Unlike the warn-only re-derive brake, this is a blocking ERROR: an
    unsatisfiable gate is not a judgement call."""
    script_rel = "experiments/__prereg_share_feasibility_test__.py"
    script_path = REPO_ROOT / script_rel
    script_path.write_text(SRC_785_SHAPE, encoding="utf-8")
    # The tracked-in-git check is separate; stub it so this test pins only the lint.
    monkeypatch.setattr(validate_queue, "_is_tracked", lambda *a, **k: True)
    monkeypatch.setattr(validate_queue, "_scan_completed_queue_ids", lambda: {})
    monkeypatch.setattr(validate_queue, "QUEUE_FILE", REPO_ROOT / "experiment_queue.json")
    try:
        queue = {
            "schema_version": "v1",
            "calibration": {},
            "items": [{
                "queue_id": "V3-EXQ-999",
                "script": script_rel,
                "priority": 1,
                "machine_affinity": "any",
                "status": "pending",
                "estimated_minutes": 10,
                "claim_ids": [],
            }],
        }
        queue_path = REPO_ROOT / "__prereg_test_queue__.json"
        queue_path.write_text(json.dumps(queue), encoding="utf-8")
        try:
            errors = validate_queue.validate(queue_path)
        finally:
            queue_path.unlink()
    finally:
        script_path.unlink()

    matching = [e for e in errors if "expected_incumbent_share" in e]
    assert len(matching) == 1, f"expected a blocking error, got: {errors}"
    assert "V3-EXQ-999" in matching[0], "error must identify the queue item"


# ---- (11) corpus-wide false-positive guard ---------------------------------
def test_no_false_positives_across_experiments_corpus():
    """The check must be quiet on the entire existing corpus except the one known
    defect. If a new script trips this, that script has an unsatisfiable gate --
    fix the script, do not loosen the check."""
    hits = []
    for p in sorted(EXPERIMENTS_DIR.rglob("*.py")):
        try:
            src = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if _lint(src):
            hits.append(p.name)
    assert hits in (
        [], ["v3_exq_785_mech463_arousal_variance_amplifier_decomp.py"]
    ), f"unexpected pre-registration feasibility hits: {hits}"
