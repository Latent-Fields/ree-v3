"""Contracts for arm-reuse fingerprint enforcement + the arm_cell() helper.

Two surfaces:
  (1) validate_experiments.arm_fingerprint_lint -- detects multi-arm scripts
      (write "arm_results") that omit the per-cell reset_all_rng + fingerprint
      emission, with an ARM_FINGERPRINT_EXEMPT opt-out.
  (2) experiments._lib.arm_fingerprint.arm_cell -- the bundled context manager
      that resets RNG on enter and stamps the fingerprint, producing a payload
      identical to the low-level reset_all_rng + compute_arm_fingerprint pair.

Design plan: REE_assembly/evidence/planning/arm_reuse_fingerprint_plan.md
(determinism gate closed + ratified 2026-06-07).
"""
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import validate_experiments as V  # noqa: E402
from _lib.arm_fingerprint import (  # noqa: E402
    arm_cell,
    compute_arm_fingerprint,
    reset_all_rng,
)

EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def _lint(src: str):
    """Write src to a temp .py under experiments/ and return the lint result."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(src)
        name = f.name
    try:
        return V.arm_fingerprint_lint(Path(name))
    finally:
        os.unlink(name)


# ---- (1) lint detection branches -------------------------------------------

def test_c1_single_arm_not_flagged():
    """No "arm_results" -> not a multi-arm grid -> no issue."""
    assert _lint('y = {"per_seed_results": []}\n') is None


def test_c2_multi_arm_missing_both_flagged():
    """Writes arm_results but no reset / no fingerprint -> issue names both."""
    issue = _lint('x = {"arm_results": []}\n')
    assert issue is not None
    assert "reset_all_rng" in issue
    assert "compute_arm_fingerprint" in issue


def test_c3_multi_arm_low_level_pair_ok():
    """reset_all_rng + compute_arm_fingerprint present -> no issue."""
    src = (
        "from _lib.arm_fingerprint import reset_all_rng, compute_arm_fingerprint\n"
        'x = {"arm_results": []}\n'
        "reset_all_rng(1)\n"
        "compute_arm_fingerprint(config_slice={}, seed=1, rng_fully_reset=True)\n"
    )
    assert _lint(src) is None


def test_c4_multi_arm_arm_cell_ok():
    """arm_cell() alone discharges BOTH obligations -> no issue."""
    src = (
        "from _lib.arm_fingerprint import arm_cell\n"
        'x = {"arm_results": []}\n'
        "with arm_cell(1, config_slice={}, script_path=None) as c:\n"
        "    c.stamp({})\n"
    )
    assert _lint(src) is None


def test_c5_multi_arm_only_reset_still_flagged():
    """RNG reset without fingerprint emission is still incomplete."""
    src = (
        "from _lib.arm_fingerprint import reset_all_rng\n"
        'x = {"arm_results": []}\n'
        "reset_all_rng(1)\n"
    )
    issue = _lint(src)
    assert issue is not None
    assert "compute_arm_fingerprint" in issue
    assert "reset_all_rng" not in issue.split("missing", 1)[1]


def test_c6_exempt_marker_suppresses():
    """ARM_FINGERPRINT_EXEMPT opt-out suppresses the check."""
    assert _lint('ARM_FINGERPRINT_EXEMPT = "single cell"\nx = {"arm_results": []}\n') is None


# ---- (2) arm_cell helper ----------------------------------------------------

def test_c7_arm_cell_stamps_row_reuse_eligible():
    row = {"arm": "A", "seed": 42}
    with arm_cell(42, config_slice={"p0": 60}, script_path=None) as cell:
        fp = cell.stamp(row)
    assert row["arm_fingerprint"] is fp
    assert fp["schema"] == "arm_fp/v1"
    assert fp["seed"] == 42
    assert fp["reuse_eligible"] is True
    assert fp["reuse_ineligible_reasons"] == []


def test_c8_arm_cell_matches_manual_pair():
    """arm_cell payload is identical to the low-level reset+compute pair."""
    cfg = {"p0": 60, "p1": 20}
    sp = EXPERIMENTS_DIR / "v3_exq_646_mint_modulatory_authority_off_baseline.py"
    with arm_cell(43, config_slice=cfg, script_path=sp) as cell:
        viacell = cell.stamp({})
    reset_all_rng(43)
    manual = compute_arm_fingerprint(config_slice=cfg, seed=43, script_path=sp,
                                     rng_fully_reset=True)
    assert viacell["arm_fingerprint"] == manual["arm_fingerprint"]
    assert viacell["substrate_hash"] == manual["substrate_hash"]


def test_c9_arm_cell_no_reset_marks_ineligible():
    """do_reset=False -> rng not reset -> reuse_eligible False."""
    with arm_cell(7, config_slice={}, script_path=None, do_reset=False) as cell:
        fp = cell.stamp({})
    assert fp["reuse_eligible"] is False
    assert "incomplete_rng_reset" in fp["reuse_ineligible_reasons"]
