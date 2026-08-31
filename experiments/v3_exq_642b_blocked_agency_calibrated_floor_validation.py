#!/opt/local/bin/python3
"""V3-EXQ-642b -- MECH-353 calibrated blocked-agency floor validation.

Claims: [] (diagnostic substrate-readiness run; excluded from claim-confidence
scoring). A PASS validates the post-642a calibration build and permits governance
to clear MECH-353 v3_pending. Bears on, but does not tag: MECH-353, SD-029,
MECH-112, MECH-320, MECH-342, ARC-016, SD-011, SD-019b, SD-070, and SD-056.

Supersedes: V3-EXQ-642a. That run established a trained, action-discriminative
comparator (C0 passed on all three seeds) but its peak-separation criterion was
unsatisfiable: the fixed absolute outcome_mismatch_floor=0.1 sat below the
ordinary free-step mismatch baseline (~0.38-0.50), so ARM_BLOCK and ARM_CONTROL
both saturated at z_block_cap=1.5 and C1/C2 had no dynamic range.

This successor intentionally reuses 642a's environment, P0a encoder warmup, P0b
SD-056 world-forward training, readiness gate, seeds, arms, budgets, and
pre-registered C0-C3 thresholds. The sole causal change is enabling the
post-build baseline-relative mismatch floor from ree-v3 commit d49db86f3e64670:

  blocked_agency_outcome_mismatch_floor_mode = "baseline_relative"
  blocked_agency_outcome_mismatch_baseline_alpha = 0.02
  blocked_agency_outcome_mismatch_floor_ratio = 1.5
  blocked_agency_outcome_mismatch_baseline_min_floor = 0.02

The free-step exponential moving average seeds on the first real action-outcome
observation (not the initial no-history sense tick), which bootstraps rather than
classifies that observation. It then updates only on ticks classified free. A
sustained external block therefore cannot raise its own threshold. The legacy
absolute mode remains the bit-identical default and is not changed globally by
this experiment.

Pre-registered interpretation is inherited unchanged from 642a. In particular,
PASS requires C0, C1, C2, and C3 on at least two thirds of readiness-cleared
seeds; merely reducing CONTROL saturation is not sufficient. A failure after C0
passes remains a regulator/consumer result, while readiness failure self-routes
to substrate_not_ready_requeue rather than weakening MECH-353.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from experiments import v3_exq_642a_blocked_agency_zblock_discriminative as base


EXPERIMENT_TYPE = "v3_exq_642b_blocked_agency_calibrated_floor_validation"
SUPERSEDES = "V3-EXQ-642a"
CALIBRATION_CONFIG = {
    "blocked_agency_outcome_mismatch_floor_mode": "baseline_relative",
    "blocked_agency_outcome_mismatch_baseline_alpha": 0.02,
    "blocked_agency_outcome_mismatch_floor_ratio": 1.5,
    "blocked_agency_outcome_mismatch_baseline_min_floor": 0.02,
}

# Reuse the reviewed 642a protocol while changing exactly the built lever under
# validation. Mutating the imported module's globals is deliberate: base.main()
# reads these names at runtime when it builds the agent, manifest, and script
# provenance record.
base.__doc__ = __doc__
base.__file__ = __file__
base.EXPERIMENT_TYPE = EXPERIMENT_TYPE
base.SUPERSEDES = SUPERSEDES
base.CFG_KWARGS = {**base.CFG_KWARGS, **CALIBRATION_CONFIG}

_build_642a_manifest = base._build_manifest


def _build_manifest(result, timestamp_utc, dry_run):
    manifest = _build_642a_manifest(result, timestamp_utc, dry_run)
    # This diagnostic carries no claim IDs, so even a protocol FAIL must not
    # weaken a claim through metadata inherited from the claim-bearing base.
    manifest["evidence_direction"] = "non_contributory"
    manifest["evidence_direction_note"] = (
        "Claimless post-build substrate validation; outcome routes governance "
        "readiness only and does not update claim confidence."
    )
    manifest["calibration_under_test"] = dict(CALIBRATION_CONFIG)
    manifest["predecessor_result"] = {
        "queue_id": "V3-EXQ-642a",
        "run_id": (
            "v3_exq_642a_blocked_agency_zblock_discriminative_"
            "20260829T185417Z_v3"
        ),
        "outcome": "FAIL",
        "evidence_direction": "non_contributory",
        "failure_signature": (
            "z_block_peak=1.5 in both arms on all three seeds; "
            "CONTROL z_block_mean=1.26-1.35; C1 separation=0.0"
        ),
    }
    manifest["notes"] = (
        "Post-build validation of the baseline-relative blocked-agency "
        "outcome-mismatch floor implemented in ree-v3 d49db86f3e64670. "
        "The complete V3-EXQ-642a protocol and thresholds are reused; the sole "
        "causal change is CALIBRATION_CONFIG. Existing evidence cannot answer "
        "this question because 642a ran before the calibration build and used "
        "the legacy absolute floor. PASS requires C0-C3 on >=2/3 readiness-"
        "cleared seeds and permits governance to clear MECH-353 v3_pending."
    )
    return manifest


base._build_manifest = _build_manifest


def main():
    return base.main()


if __name__ == "__main__":
    _outcome, _manifest_path = main()
    _dry = "--dry-run" in sys.argv
    if _outcome is not None:
        base.emit_outcome(
            outcome=_outcome,
            manifest_path=_manifest_path,
            dry_run=_dry,
        )
    raise SystemExit(0)
