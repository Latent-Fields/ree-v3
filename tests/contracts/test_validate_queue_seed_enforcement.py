"""
Contract tests for the queue-declared-seed-count enforcement check in
validate_queue.py (seed_enforcement_lint / _script_seeds_default_count),
added for SD-QUEUE-SEED-ENFORCEMENT (2026-08-13).

THE DEFECT IT CLOSES. experiment_queue.json's "seeds": N field is consumed
ONLY by experiment_runner.py's _run_axis_count, for progress-bar/ETA
denominators -- it is NEVER translated into a --seeds CLI argument. A
driver's own argparse default is therefore the sole source of truth for how
many seeds actually run, and nothing cross-checked the declared count
against either an explicit --seeds override in the queue item's 'args' or
the target script's own argparse default.

Confirmed twice within two days, on different drivers (source_autopsy
failure_autopsy_V3-EXQ-912-913-fishtank-cluster_2026-08-11.json):
  - V3-EXQ-912: queue declared "seeds": 2, script's own --seeds default was
    [0] (1 seed) -- n_segments_total=60 not the designed 120;
    n_uncensored_deaths_total=4 < MIN_UNCENSORED_DEATHS_TOTAL=10, driving
    the FAIL on an under-powered run.
  - V3-EXQ-920: queue correctly declared "seeds": 8, only 1 of 8 ran. Its
    manifest also self-routed a flatly incorrect censoring label on top of
    the under-powered run (pct_right_censored_pooled=0.0, zero censoring).
  - Sibling V3-EXQ-913 avoided this only because its author happened to set
    the script's own SEEDS_DEFAULT correctly -- luck, not a property of the
    system.

CONSERVATISM IS THE POINT (same discipline as prereg_share_feasibility_lint
in this file, and per CLAUDE.md's over-eager-commit-hook warning): this is
a PreToolUse commit-blocking check AND a runner-startup-blocking check (it
is wired into validate(), which experiment_runner.load_queue() already
calls and treats any error as fatal -- so this same check also protects
every machine's runner at startup, including cloud workers that pull `main`
directly and never go through a locally-issued `git commit`). It fires
ONLY on the fully-conjunctive, statically-verified case: declared seeds > 1,
no explicit --seeds override in 'args', and the script's own --seeds
argparse default is confidently resolvable to fewer entries than declared.
Anything unresolvable (default=None, a computed expression, a type=str
comma-string contract, an unresolvable name) is skipped, never guessed.

Branches pinned:
  (1) the real V3-EXQ-912 script (fixture of record) trips it.
  (2) the real V3-EXQ-920 script (fixture of record) trips it.
  (3) an explicit --seeds override in 'args' (list form) suppresses it.
  (4) an explicit --seeds override in 'args' (shell-string form) suppresses it.
  (5) declared seeds == 1 does not fire (mirrors _run_axis_count's floor).
  (6) declared seeds == script default count does not fire (exact match, no gap).
  (7) declared seeds < script default count does not fire (defaults already cover it).
  (8) default=None (the largest corpus pattern) does not fire -- unresolvable.
  (9) type=str comma-string contract does not fire -- unresolvable.
 (10) a module-level list constant (SEEDS = [...]) is resolved.
 (11) a module-level tuple constant via list(SEEDS)/tuple(SEEDS) is resolved.
 (12) no --seeds argument in the script at all does not fire -- unresolvable,
      not "assume 0".
 (13) unparseable source fails soft.
 (14) the finding reaches validate()'s returned ERRORS (blocking, like the
      prereg-share-feasibility check, not the warn-only re-derive brake).
 (15) the resolver never crashes across the full experiments/ corpus.
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

SRC_ONE_SEED_DEFAULT = '''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, nargs="+", default=[0])
'''

SRC_SUFFICIENT_DEFAULT = '''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
'''

SRC_NONE_DEFAULT = '''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, nargs="*", default=None)
'''

SRC_STR_CONTRACT = '''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=str, default="42,123")
'''

SRC_MODULE_LIST_CONST = '''
SEEDS = [42, 123]
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
'''

SRC_MODULE_TUPLE_LIST_CALL = '''
SEEDS = (42, 43, 44)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
'''

SRC_NO_SEEDS_ARG = '''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=200)
'''


def _lint(src: str, item: dict, filename: str = "<test>") -> "str | None":
    return validate_queue.seed_enforcement_lint(src, item, filename)


# ---- (1)-(2) the real confirmed-defect scripts, as fixtures of record -----
def test_real_912_script_trips_the_check():
    script = EXPERIMENTS_DIR / "v3_exq_912_uncensored_survival_fishtank.py"
    if not script.is_file():
        pytest.skip("V3-EXQ-912 script no longer present")
    src = script.read_text(encoding="utf-8", errors="ignore")
    finding = _lint(src, {"seeds": 2, "args": []}, str(script))
    assert finding is not None, "must catch the confirmed V3-EXQ-912 defect"
    assert "seeds\": 2" in finding
    assert "1 seed value" in finding


def test_real_920_script_trips_the_check():
    script = EXPERIMENTS_DIR / "v3_exq_920_uncensored_survival_single_life_fishtank.py"
    if not script.is_file():
        pytest.skip("V3-EXQ-920 script no longer present")
    src = script.read_text(encoding="utf-8", errors="ignore")
    finding = _lint(src, {"seeds": 8, "args": []}, str(script))
    assert finding is not None, "must catch the confirmed V3-EXQ-920 defect"
    assert "seeds\": 8" in finding


# ---- (3)-(4) explicit override suppresses it -------------------------------
def test_explicit_seeds_override_list_form_suppresses():
    item = {"seeds": 2, "args": ["--episodes", "200", "--seeds", "0", "1"]}
    assert _lint(SRC_ONE_SEED_DEFAULT, item) is None


def test_explicit_seeds_override_shell_string_form_suppresses():
    item = {"seeds": 2, "args": "--episodes 200 --seeds 0 1"}
    assert _lint(SRC_ONE_SEED_DEFAULT, item) is None


# ---- (5)-(7) counts that must NOT fire -------------------------------------
def test_declared_seeds_of_one_does_not_fire():
    item = {"seeds": 1, "args": []}
    assert _lint(SRC_ONE_SEED_DEFAULT, item) is None


def test_declared_seeds_matching_default_exactly_does_not_fire():
    item = {"seeds": 1, "args": []}
    assert _lint(SRC_ONE_SEED_DEFAULT, item) is None
    item2 = {"seeds": 3, "args": []}
    assert _lint(SRC_SUFFICIENT_DEFAULT, item2) is None


def test_declared_seeds_below_default_does_not_fire():
    item = {"seeds": 2, "args": []}
    assert _lint(SRC_SUFFICIENT_DEFAULT, item) is None


# ---- (8)-(9), (12) unresolvable defaults fail soft (never guess) ----------
def test_none_default_does_not_fire():
    item = {"seeds": 8, "args": []}
    assert _lint(SRC_NONE_DEFAULT, item) is None


def test_str_type_comma_contract_does_not_fire():
    item = {"seeds": 5, "args": []}
    assert _lint(SRC_STR_CONTRACT, item) is None


def test_no_seeds_argument_in_script_does_not_fire():
    item = {"seeds": 4, "args": []}
    assert _lint(SRC_NO_SEEDS_ARG, item) is None


# ---- (10)-(11) module-level constant resolution ----------------------------
def test_module_level_list_constant_is_resolved():
    item_ok = {"seeds": 2, "args": []}
    assert _lint(SRC_MODULE_LIST_CONST, item_ok) is None
    item_bad = {"seeds": 3, "args": []}
    finding = _lint(SRC_MODULE_LIST_CONST, item_bad)
    assert finding is not None
    assert "2 seed value" in finding


def test_module_level_tuple_via_list_call_is_resolved():
    item_ok = {"seeds": 3, "args": []}
    assert _lint(SRC_MODULE_TUPLE_LIST_CALL, item_ok) is None
    item_bad = {"seeds": 5, "args": []}
    finding = _lint(SRC_MODULE_TUPLE_LIST_CALL, item_bad)
    assert finding is not None
    assert "3 seed value" in finding


# ---- (13) fail-soft ---------------------------------------------------------
def test_unparseable_source_fails_soft():
    item = {"seeds": 5, "args": []}
    assert _lint("def broken(:\n  pass\n", item) is None


def test_declared_seeds_none_or_absent_does_not_fire():
    assert _lint(SRC_ONE_SEED_DEFAULT, {"args": []}) is None
    assert _lint(SRC_ONE_SEED_DEFAULT, {"seeds": True, "args": []}) is None  # bool guard


# ---- (14) it blocks, via validate()'s error list ---------------------------
def test_finding_reaches_validate_errors(tmp_path, monkeypatch):
    """Unlike a warn-only advisory, this is a blocking ERROR: a confirmed
    seed-count shortfall is not a judgement call -- it silently converts an
    under-powered run into a scientific FAIL."""
    script_rel = "experiments/__seed_enforcement_test__.py"
    script_path = REPO_ROOT / script_rel
    script_path.write_text(SRC_ONE_SEED_DEFAULT, encoding="utf-8")
    monkeypatch.setattr(validate_queue, "_is_tracked", lambda *a, **k: True)
    monkeypatch.setattr(validate_queue, "_scan_completed_queue_ids", lambda: {})
    monkeypatch.setattr(validate_queue, "QUEUE_FILE", REPO_ROOT / "experiment_queue.json")
    try:
        queue = {
            "schema_version": "v1",
            "calibration": {},
            "items": [{
                "queue_id": "V3-EXQ-998",
                "script": script_rel,
                "priority": 1,
                "machine_affinity": "any",
                "status": "pending",
                "estimated_minutes": 10,
                "claim_ids": [],
                "seeds": 2,
                "args": [],
            }],
        }
        queue_path = REPO_ROOT / "__seed_enforcement_test_queue__.json"
        queue_path.write_text(json.dumps(queue), encoding="utf-8")
        try:
            errors = validate_queue.validate(queue_path)
        finally:
            queue_path.unlink()
    finally:
        script_path.unlink()

    matching = [e for e in errors if "seeds\": 2" in e]
    assert len(matching) == 1, f"expected a blocking error, got: {errors}"
    assert "V3-EXQ-998" in matching[0], "error must identify the queue item"


def test_explicit_override_does_not_block_validate(tmp_path, monkeypatch):
    """Negative control for (14): an explicit --seeds override must NOT block."""
    script_rel = "experiments/__seed_enforcement_ok_test__.py"
    script_path = REPO_ROOT / script_rel
    script_path.write_text(SRC_ONE_SEED_DEFAULT, encoding="utf-8")
    monkeypatch.setattr(validate_queue, "_is_tracked", lambda *a, **k: True)
    monkeypatch.setattr(validate_queue, "_scan_completed_queue_ids", lambda: {})
    monkeypatch.setattr(validate_queue, "QUEUE_FILE", REPO_ROOT / "experiment_queue.json")
    try:
        queue = {
            "schema_version": "v1",
            "calibration": {},
            "items": [{
                "queue_id": "V3-EXQ-997",
                "script": script_rel,
                "priority": 1,
                "machine_affinity": "any",
                "status": "pending",
                "estimated_minutes": 10,
                "claim_ids": [],
                "seeds": 2,
                "args": ["--seeds", "0", "1"],
            }],
        }
        queue_path = REPO_ROOT / "__seed_enforcement_ok_test_queue__.json"
        queue_path.write_text(json.dumps(queue), encoding="utf-8")
        try:
            errors = validate_queue.validate(queue_path)
        finally:
            queue_path.unlink()
    finally:
        script_path.unlink()

    matching = [e for e in errors if "seeds\":" in e and "V3-EXQ-997" in e]
    assert matching == [], f"explicit --seeds override must not block: {matching}"


# ---- (15) corpus-wide crash guard ------------------------------------------
def test_resolver_never_crashes_across_experiments_corpus():
    """_script_seeds_default_count must never raise on any real script in the
    corpus -- it is fail-soft (returns None) on anything it cannot resolve,
    never an exception. A crash here would take down validate() fleet-wide."""
    if not EXPERIMENTS_DIR.is_dir():
        pytest.skip("experiments/ directory not present")
    n_checked = 0
    for f in sorted(EXPERIMENTS_DIR.glob("*.py")):
        try:
            source = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        n_checked += 1
        # Must not raise, regardless of what it resolves to.
        validate_queue._script_seeds_default_count(source, str(f))
        # Also exercise the full lint with a representative declared count,
        # via a synthetic item -- must not raise either.
        validate_queue.seed_enforcement_lint(source, {"seeds": 999, "args": []}, str(f))
    assert n_checked > 0, "expected at least one script in experiments/"
