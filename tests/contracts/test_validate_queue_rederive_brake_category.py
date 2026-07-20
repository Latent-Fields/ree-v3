"""
Contract tests for the MOVE-3 re-derive brake's EPISTEMIC-CATEGORY predicate
(landed 2026-07-20). Companion to test_validate_queue_rederive_brake.py, which
pins the surrounding warn-only machinery; this file pins WHICH autopsies count.

THE DEFECT THIS LOCKS OUT. The brake exists to stop a claim being re-tested letter
after letter against a substrate ceiling already autopsied twice or more. Its original
predicate matched on the evidence DIRECTION alone:

    'substrate_ceiling' in cat or 'non_contributory' in direction

An INSTRUMENT / MEASUREMENT defect -- a run that failed to *measure* its DV -- almost
always carries direction "non_contributory" (the run measured nothing usable), so it
counted toward the brake even when the autopsy explicitly recorded that NO substrate
build is owed. That inverts the brake's purpose: instrument repair is exactly the right
route, and the brake's whole job is to divert to /implement-substrate.

CONFIRMED INSTANCE (2026-07-20, the fixtures below). MECH-448 reached 3 counted
autopsies -- V3-EXQ-689d, V3-EXQ-689d-D3 and V3-EXQ-699 -- every one of them
recommended_epistemic_category "measurement_test_design_defect" with
recommended_substrate_queue_entry.action "none" and a rationale reading "the substrate
is built and validated -- this is instrument repair, not a substrate gap". The brake had
to be adjudicated released by hand. The ARC-110 V3-EXQ-707c author independently hit the
same false positive. Four live queue items (689i, 699b, 707c, 708a) had to carry the
're-derive brake' note escape hatch to get past it.

THE PREDICATE (validate_queue._autopsy_counts_toward_brake), order-sensitive:
  (1) A GENUINE substrate_ceiling reading always counts. A category that merely NEGATES
      a ceiling in prose ("measurement_calibration_not_substrate_ceiling") does not.
  (2) Otherwise an INSTRUMENT/MEASUREMENT category owing NO build does not count.
  (3) Otherwise an EXPLICIT producer release does not count -- but ONLY in the
      unambiguous form `fired: false` AND `literal_count_meets_threshold: true`.
  (4) Otherwise fall back to the direction reading.

WHY (3) IS SO NARROW -- the over-correction this file also guards. A bare
`re_derive_brake.fired: false` is NOT authoritative: it is the value every FIRST autopsy
in a lineage carries, because the count was below threshold when it was written.
Measured over the corpus on 2026-07-20: 42 counted targets carry `fired: false` and
exactly ONE carries `literal_count_meets_threshold`. Honouring a bare `fired: false`
would drop those 42 and gut the brake -- see
test_bare_fired_false_is_not_authoritative_and_still_counts.

WHY (2) IS GUARDED BY owes_build. An instrument-category autopsy that still names a
substrate to create/amend is routing to /implement-substrate anyway, so it KEEPS
counting: a re-test before that build lands re-derives the ceiling regardless of how the
category was labelled. This is what keeps MECH-095 and MECH-140 braked.

MUST-KEEP-FIRING. MECH-457 (8 autopsies, WARNing on V3-EXQ-742a) is a CORRECT brake and
its categories are competence_implementation_gap / substrate_starved_precondition_unmet /
standard -- none an instrument marker. See
test_competence_implementation_gap_still_counts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import validate_queue  # noqa: E402


BRAKE_MARK = "re-derive brake"


# ------------------------------------------------------------------
# Real fixtures: the two MECH-448 autopsies that tripped the brake.
# Discriminating fields reproduced verbatim from
#   REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-689d_2026-07-20.json
#   REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-699_2026-07-20.json
# Kept inline so the contract is hermetic (REE_assembly is a separate repo and is
# not guaranteed present on a cloud worker); test_real_corpus_* below cross-checks
# against the real files when they ARE present.
# ------------------------------------------------------------------
FIXTURE_689D = {
    "recommended_epistemic_category": "measurement_test_design_defect",
    "recommended_evidence_direction": "non_contributory",
    "claim_ids": ["MECH-448"],
    "recommended_substrate_queue_entry": {
        "action": "none",
        "rationale": (
            "the substrate is built and validated -- readiness, rank-preservation and "
            "safety all survive. This is instrument repair, not a substrate gap."
        ),
    },
    "re_derive_brake": {
        "fired": False,
        "threshold": 2,
        "literal_count_meets_threshold": True,
        "refused_requeue": False,
        "route_to": "queue-experiment",
        "upstream_substrate": None,
        "user_ratified": True,
    },
}

FIXTURE_699 = {
    "recommended_epistemic_category": "measurement_test_design_defect",
    "recommended_evidence_direction": "non_contributory",
    "claim_ids": ["MECH-448", "MECH-449"],
    "recommended_substrate_queue_entry": {
        "action": "none",
        "rationale": (
            "The defect is in driver instrumentation, not substrate. Both levers are "
            "built and independently validated (689d, 689g). No substrate build is "
            "owed by this finding."
        ),
    },
    "re_derive_brake": {
        "fired": False,
        "threshold": 2,
        "prior_substrate_ceiling_autopsies": [],
        "refused_requeue": False,
        "route_to": "queue-experiment",
        "upstream_substrate": None,
    },
}

GENUINE_CEILING = {
    "recommended_epistemic_category": "substrate_ceiling",
    "recommended_evidence_direction": "non_contributory",
    "claim_ids": ["MECH-448"],
    "recommended_substrate_queue_entry": {"action": "create", "target_sd_id": "SD-AAA"},
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _write(planning_dir: Path, slug: str, date: str, target: dict) -> None:
    (planning_dir / f"failure_autopsy_{slug}_{date}.json").write_text(
        json.dumps({"targets": [target]}), encoding="utf-8"
    )


def _make_item(claim_ids, queue_id="V3-EXQ-901") -> dict:
    return {
        "queue_id": queue_id,
        "script": "experiments/__nonexistent_brake_test__.py",
        "priority": 1,
        "machine_affinity": "any",
        "status": "pending",
        "estimated_minutes": 10,
        "claim_ids": claim_ids,
    }


def _run(tmp_path, monkeypatch, targets, claim, claude_md_text=""):
    """Seed one autopsy per target, queue an item tagging `claim`, run validate()."""
    planning_dir = tmp_path / "planning"
    planning_dir.mkdir(exist_ok=True)
    for i, t in enumerate(targets):
        _write(planning_dir, f"case{i}", f"2026-07-{10 + i:02d}", t)

    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text(claude_md_text, encoding="utf-8")
    monkeypatch.setattr(
        validate_queue, "_REE_ASSEMBLY_PLANNING_DIR_CANDIDATES", [planning_dir]
    )
    monkeypatch.setattr(validate_queue, "_REE_V3_CLAUDE_MD_CANDIDATES", [claude_md])

    queue_path = tmp_path / "experiment_queue.json"
    queue_path.write_text(
        json.dumps(
            {"schema_version": "v1", "calibration": {}, "items": [_make_item([claim])]}
        ),
        encoding="utf-8",
    )
    errors = validate_queue.validate(queue_path)
    return errors, [w for w in validate_queue._LAST_WARNINGS if BRAKE_MARK in w]


# ==================================================================
# (A) The confirmed instance -- the two MECH-448 autopsies must NOT brake
# ==================================================================
def test_mech448_real_fixtures_do_not_trip_the_brake(tmp_path, monkeypatch):
    """The exact pair that tripped it on 2026-07-20. Both are instrument repair
    (measurement_test_design_defect + action 'none'), so the count must be 0 and no
    advisory may be emitted -- even though both read 'non_contributory'."""
    _, brake_warnings = _run(
        tmp_path, monkeypatch, [FIXTURE_689D, FIXTURE_699], claim="MECH-448"
    )
    assert brake_warnings == [], (
        "instrument-repair autopsies must not brake MECH-448 -- "
        f"got: {brake_warnings}"
    )


def test_mech448_real_fixtures_count_zero(tmp_path, monkeypatch):
    """Count directly, independent of the warn plumbing."""
    planning_dir = tmp_path / "planning"
    planning_dir.mkdir()
    _write(planning_dir, "689d", "2026-07-20", FIXTURE_689D)
    _write(planning_dir, "699", "2026-07-20", FIXTURE_699)
    monkeypatch.setattr(
        validate_queue, "_REE_ASSEMBLY_PLANNING_DIR_CANDIDATES", [planning_dir]
    )
    counted = validate_queue._scan_substrate_ceiling_autopsies()
    assert counted.get("MECH-448", []) == []
    assert counted.get("MECH-449", []) == []


def test_mech449_not_braked_by_the_shared_699_autopsy(tmp_path, monkeypatch):
    """699 tags MECH-449 too; the exclusion must apply per-target, not per-claim."""
    _, brake_warnings = _run(tmp_path, monkeypatch, [FIXTURE_699], claim="MECH-449")
    assert brake_warnings == []


# ==================================================================
# (B) A genuine substrate_ceiling autopsy still MUST brake
# ==================================================================
def test_genuine_substrate_ceiling_still_trips_the_brake(tmp_path, monkeypatch):
    """The load-bearing counter-case: without this the fix would be a brake deletion."""
    _, brake_warnings = _run(
        tmp_path,
        monkeypatch,
        [GENUINE_CEILING, dict(GENUINE_CEILING)],
        claim="MECH-448",
        claude_md_text="# ree-v3\nnothing built here\n",
    )
    assert len(brake_warnings) == 1
    assert "MECH-448" in brake_warnings[0]


def test_one_genuine_ceiling_plus_two_instrument_stays_below_threshold(
    tmp_path, monkeypatch
):
    """Mixed lineage: only the genuine ceiling counts, so 1 < threshold -> no warn."""
    _, brake_warnings = _run(
        tmp_path,
        monkeypatch,
        [GENUINE_CEILING, FIXTURE_689D, FIXTURE_699],
        claim="MECH-448",
    )
    assert brake_warnings == []


def test_competence_implementation_gap_still_counts(tmp_path, monkeypatch):
    """MECH-457's real shape -- competence_implementation_gap + non_contributory.
    It is NOT an instrument marker and MUST keep braking (8 autopsies, WARNing on
    V3-EXQ-742a). This is the explicit do-not-over-correct guard."""
    t = {
        "recommended_epistemic_category": "competence_implementation_gap",
        "recommended_evidence_direction": "non_contributory",
        "claim_ids": ["MECH-457"],
    }
    _, brake_warnings = _run(tmp_path, monkeypatch, [t, dict(t)], claim="MECH-457")
    assert len(brake_warnings) == 1
    assert "MECH-457" in brake_warnings[0]


def test_standard_category_with_non_contributory_still_counts(tmp_path, monkeypatch):
    """The plain direction-only path must survive for non-instrument categories."""
    t = {
        "recommended_epistemic_category": "standard",
        "recommended_evidence_direction": "non_contributory",
        "claim_ids": ["MECH-900"],
    }
    _, brake_warnings = _run(tmp_path, monkeypatch, [t, dict(t)], claim="MECH-900")
    assert len(brake_warnings) == 1


# ==================================================================
# (C) owes_build guard -- instrument category that still owes a build KEEPS counting
# ==================================================================
@pytest.mark.parametrize("action", ["create", "amend"])
def test_instrument_category_owing_a_build_still_counts(tmp_path, monkeypatch, action):
    """A measurement autopsy that still names a substrate to build routes to
    /implement-substrate anyway, so a re-test before that build does re-derive.
    This is what keeps MECH-095 (047m) and MECH-140 (710) braked."""
    t = {
        "recommended_epistemic_category": "measurement_degeneracy",
        "recommended_evidence_direction": "non_contributory",
        "claim_ids": ["MECH-095"],
        "recommended_substrate_queue_entry": {"action": action, "target_sd_id": "SD-AAA"},
    }
    _, brake_warnings = _run(
        tmp_path, monkeypatch, [t, dict(t)], claim="MECH-095",
        claude_md_text="# nothing built\n",
    )
    assert len(brake_warnings) == 1, f"action={action} owes a build and must count"


def test_instrument_category_with_action_none_does_not_count(tmp_path, monkeypatch):
    t = {
        "recommended_epistemic_category": "measurement_degeneracy",
        "recommended_evidence_direction": "non_contributory",
        "claim_ids": ["INV-064"],
        "recommended_substrate_queue_entry": {"action": "none"},
    }
    _, brake_warnings = _run(tmp_path, monkeypatch, [t, dict(t)], claim="INV-064")
    assert brake_warnings == []


# ==================================================================
# (D) Instrument-marker coverage + the negated-ceiling correction
# ==================================================================
@pytest.mark.parametrize(
    "category",
    [
        "measurement_test_design_defect",
        "measurement_gap",
        "measurement_degeneracy",
        "measurement_artifact",
        "measurement_invalid",
        "measurement_reframe",
        "measurement_regime_mismatch",
        "instrumentation_defect",
        "test_design_ceiling",
        "vacuous_pass",
        "measurement_test_design_gap (N/A for governance -- no claim)",
    ],
)
def test_instrument_categories_do_not_count(category):
    """Every instrument/measurement family member, owing no build, is excluded."""
    t = {
        "recommended_epistemic_category": category,
        "recommended_evidence_direction": "non_contributory",
        "claim_ids": ["MECH-900"],
    }
    assert validate_queue._autopsy_counts_toward_brake(t) is False, category


@pytest.mark.parametrize(
    "category",
    [
        "measurement_calibration_not_substrate_ceiling",
        "non_contributory_run_not_substrate_ceiling",
        "precondition_unmet (642-pattern, NOT substrate_ceiling)",
    ],
)
def test_categories_that_negate_a_ceiling_are_not_read_as_genuine_ceilings(category):
    """Substring-matching 'substrate_ceiling' on a NEGATED phrase read these as
    genuine ceilings. Rule (1) must strip the negation before deciding."""
    t = {
        "recommended_epistemic_category": category,
        "recommended_evidence_direction": "non_contributory",
        "claim_ids": ["MECH-900"],
    }
    low = category.lower()
    if any(
        m in low for m in validate_queue.RE_DERIVE_INSTRUMENT_CATEGORY_MARKERS
    ):
        assert validate_queue._autopsy_counts_toward_brake(t) is False, category
    else:
        # Not an instrument marker -> still counts, but via DIRECTION (rule 4),
        # never by mistaking the negated phrase for a genuine ceiling (rule 1).
        assert validate_queue._autopsy_counts_toward_brake(t) is True, category


def test_genuine_ceiling_wins_over_an_instrument_marker_in_the_same_string():
    """Rule (1) precedes rule (2): a category asserting BOTH still counts."""
    t = {
        "recommended_epistemic_category": "substrate_ceiling + measurement gap",
        "recommended_evidence_direction": "non_contributory",
        "claim_ids": ["MECH-900"],
    }
    assert validate_queue._autopsy_counts_toward_brake(t) is True


# ==================================================================
# (E) The over-correction guard: a bare fired:false is NOT authoritative
# ==================================================================
def test_bare_fired_false_is_not_authoritative_and_still_counts():
    """`fired: false` is what EVERY first autopsy in a lineage carries (count below
    threshold at write time). 42 counted targets carry it corpus-wide; honouring it
    blanket would gut the brake. Only fired:false + literal_count_meets_threshold:true
    is an explicit release."""
    t = {
        "recommended_epistemic_category": "substrate_ceiling",
        "recommended_evidence_direction": "non_contributory",
        "claim_ids": ["MECH-900"],
        "re_derive_brake": {"fired": False, "threshold": 2},
    }
    assert validate_queue._autopsy_counts_toward_brake(t) is True


def test_explicit_release_requires_both_fields():
    """fired:false + literal_count_meets_threshold:true, on a non-ceiling category
    owing no build, is the one unambiguous producer release."""
    base = {
        "recommended_epistemic_category": "standard",
        "recommended_evidence_direction": "non_contributory",
        "claim_ids": ["MECH-900"],
    }
    released = dict(base)
    released["re_derive_brake"] = {"fired": False, "literal_count_meets_threshold": True}
    assert validate_queue._autopsy_counts_toward_brake(released) is False

    # fired True + meets threshold -> the brake DID fire; counts.
    fired = dict(base)
    fired["re_derive_brake"] = {"fired": True, "literal_count_meets_threshold": True}
    assert validate_queue._autopsy_counts_toward_brake(fired) is True


def test_explicit_release_is_overridden_when_a_build_is_owed():
    t = {
        "recommended_epistemic_category": "standard",
        "recommended_evidence_direction": "non_contributory",
        "claim_ids": ["MECH-900"],
        "recommended_substrate_queue_entry": {"action": "create", "target_sd_id": "SD-A"},
        "re_derive_brake": {"fired": False, "literal_count_meets_threshold": True},
    }
    assert validate_queue._autopsy_counts_toward_brake(t) is True


# ==================================================================
# (F) Non-counting readings are still non-counting (no widening)
# ==================================================================
def test_supports_direction_with_standard_category_never_counts():
    t = {
        "recommended_epistemic_category": "standard",
        "recommended_evidence_direction": "supports",
        "claim_ids": ["MECH-900"],
    }
    assert validate_queue._autopsy_counts_toward_brake(t) is False


# ==================================================================
# (G) Cross-check against the REAL corpus when REE_assembly is present
# ==================================================================
def _real_planning_dir():
    for cand in validate_queue._REE_ASSEMBLY_PLANNING_DIR_CANDIDATES:
        if Path(cand).is_dir():
            return Path(cand)
    return None


@pytest.mark.skipif(
    _real_planning_dir() is None, reason="REE_assembly planning dir not present"
)
def test_real_corpus_mech448_released_and_mech457_still_braked():
    """End-to-end on the live corpus: the confirmed false positive is released and
    the correct brake keeps firing. Guards both directions in one assertion pair."""
    counted = validate_queue._scan_substrate_ceiling_autopsies()
    n448 = len(counted.get("MECH-448", []))
    n457 = len(counted.get("MECH-457", []))
    assert n448 < validate_queue.RE_DERIVE_BRAKE_THRESHOLD, (
        f"MECH-448 still braked at {n448} -- the instrument-repair exclusion regressed"
    )
    assert n457 >= validate_queue.RE_DERIVE_BRAKE_THRESHOLD, (
        f"MECH-457 dropped to {n457} -- OVER-CORRECTION; it must keep braking"
    )
