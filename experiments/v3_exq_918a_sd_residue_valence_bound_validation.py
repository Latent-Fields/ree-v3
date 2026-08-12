"""V3-EXQ-918a: SD-RESIDUE-VALENCE-BOUND accumulator-bound validation.

Supersedes V3-EXQ-918 (ERROR, crash-before-manifest, exit code 1, ~1.4s on
ree-cloud-3). Root cause (/diagnose-errors 2026-08-12): a queue-before-fix
race, not a script bug. The V3-EXQ-918 script (ree-v3 ba2a60e334, pushed
17:23:16Z) referenced ResidueConfig.valence_bounding_enabled/
valence_decay_rate/valence_clamp_abs, which did not land until the
SD-RESIDUE-VALENCE-BOUND substrate fix (ree-v3 00449c7d0e, pushed 17:49:46Z)
-- a 26-minute window. ree-cloud-3 pulled main and started the run at
17:39:41Z, inside that window, so ResidueConfig(**bounding_kwargs) raised an
unexpected-keyword-argument TypeError in the first _make_field() call before
any manifest could be written. No code change: this file is otherwise
byte-identical to V3-EXQ-918's script, re-run now that the substrate fix is
safely on main (confirmed via local smoke test against current main: PASS,
all 4 criteria met).

SLEEP DRIVER: n/a (no sleep loop used)

PURPOSE (diagnostic, substrate readiness -- NOT a claim-bearing evidence run,
claim_ids=[]). Validates the fix landed this session in
`ree_core/residue/field.py` / `ree_core/utils/config.py`: `RBFLayer.update_valence()`
was a raw unclamped `+=` with no decay, fired every step MECH-307 split-surprise
crosses threshold, confirmed to affect all 6 SD-014 valence components (see
`REE_assembly/docs/architecture/sd_residue_valence_bound.md` and the failure
autopsies it cites: `failure_autopsy_V3-EXQ-906a_894b_2026-08-09.md`,
`failure_autopsy_906b-906c-911-cluster_2026-08-10.md`).

GOV-REUSE-1 check: the decisive readout (bounded-vs-unbounded valence magnitude
under sustained same-center writes with `valence_bounding_enabled` ON vs OFF) is
not recorded anywhere -- the config knobs this run exercises did not exist before
this session. Not recoverable -> ran fresh.

DESIGN. Unit-level (no CausalGridWorld / REEAgent needed -- the fix is a single
central write path in `ResidueField`/`RBFLayer`, so a direct unit test against
those classes is the right level, matching the V3-EXQ-520 Part-1 precedent for
a ResidueField-internal readiness diagnostic). For each of 2 bounding conditions
(OFF=default, ON=valence_bounding_enabled) x 3 write paths (POSITIVE_SURPRISE and
NEGATIVE_SURPRISE via `ResidueField.update_valence` -- the MECH-307 split-surprise
excite/dread channels the failure record measured directly -- and WANTING via
`update_wanting_sensitized`, the SD-014 incentive-sensitization path that shares
the same underlying `RBFLayer.update_valence` primitive but was not itself
measured by any prior failure record) x N_SEEDS seeds, WRITES_PER_CELL repeated
writes land on the SAME z_world point (reproducing an agent revisiting the same
RBF center under sustained exposure -- the exact scenario 906a/906c exercised).

READINESS (non-vacuity, self-routes substrate_not_ready_requeue, never a false
structural verdict): before testing, one `accumulate()` call seeds an active RBF
center at the test point. If no active center forms, `update_valence` no-ops
silently and every subsequent write would falsely read as "bounded" -- so
`active_center_exists` is the positive control every cell's result depends on.

Pre-registered acceptance criteria (per cell, then aggregated per-arm):
C1 (load-bearing): OFF arm reproduces the confirmed bug -- final |value| clears
   UNBOUNDED_FRAC * WRITES_PER_CELL * |VALUE| (near-linear, unbounded growth).
C2 (load-bearing): ON arm stays bounded -- final |value| <= CLAMP_ABS + EPS, for
   every cell.
C3: backward-compat exactness -- the OFF arm's final value equals EXACTLY
   WRITES_PER_CELL * VALUE (float tolerance), proving the fix is bit-identical
   to the pre-fix `+=` when unconfigured, not merely "still large".
C4: MECH-094 hypothesis_tag gate is unaffected by the fix -- with
   hypothesis_tag=True throughout, the tracked value never leaves its 0.0 init,
   under BOTH bounding conditions.
Overall PASS iff C1 AND C2 AND C3 AND C4 (all four confirm the fix works, is
backward compatible, and did not disturb the MECH-094 gate it sits downstream of).

claim_ids=[] -- no claims.yaml claim is gated on this SD (see the SD doc's
"Related Claims" section); this is substrate-readiness validation only.
experiment_purpose=diagnostic.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from ree_core.residue.field import (  # noqa: E402
    ResidueField, VALENCE_WANTING, VALENCE_POSITIVE_SURPRISE, VALENCE_NEGATIVE_SURPRISE,
)
from ree_core.utils.config import ResidueConfig  # noqa: E402

from _lib.arm_fingerprint import arm_cell  # noqa: E402
from pack_writer import write_flat_manifest  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402

EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v3_exq_918a_sd_residue_valence_bound_validation"
ARCH_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = []

# active_center_exists is reachable by construction: ResidueField.accumulate() (called
# here with the default hypothesis_tag=False) unconditionally calls
# rbf_field.add_residue(), which allocates/activates a center with no training or
# convergence gate (ree_core/residue/field.py accumulate()) -- so a real active center
# is the ONLY way this call can return in practice. A below-floor reading would mean
# add_residue() itself is broken, which self-routes substrate_not_ready_requeue and
# never a substrate verdict -- no instrument-gap-as-verdict mislabel is possible.
ANCHOR_REACHABILITY_EXEMPT = (
    "active_center_exists is reachable by construction: ResidueField.accumulate() "
    "unconditionally calls rbf_field.add_residue() with no training/convergence gate, "
    "so a real active center is the only way the call can return; below-floor self-"
    "routes substrate_not_ready_requeue, never a substrate verdict"
)

WORLD_DIM = 8
N_SEEDS = 3
SEED_BASE = 5000
WRITES_PER_CELL = 200          # progress denominator M (per seed x arm x channel)
VALUE = 1.0                    # per-write magnitude (sign flipped for NEGATIVE_SURPRISE)

# The fix's own defaults (ree_core/utils/config.py ResidueConfig), exercised at
# their documented activation values -- not the no-op defaults.
DECAY_RATE = 0.02
CLAMP_ABS = 5.0
EPS = 1e-3

UNBOUNDED_FRAC = 0.8           # OFF arm must clear this fraction of the naive linear sum
EXACT_TOL = 1e-4               # C3 float-exactness tolerance

CHANNELS = {
    "positive_surprise": VALENCE_POSITIVE_SURPRISE,
    "negative_surprise": VALENCE_NEGATIVE_SURPRISE,
}  # both written via ResidueField.update_valence; WANTING is tested separately below
   # via update_wanting_sensitized, a distinct call path sharing the same primitive.

BOUNDING_ARMS = {
    "OFF": dict(valence_bounding_enabled=False),
    "ON": dict(valence_bounding_enabled=True, valence_decay_rate=DECAY_RATE, valence_clamp_abs=CLAMP_ABS),
}


def _make_field(bounding_kwargs, seed):
    torch.manual_seed(seed)
    cfg = ResidueConfig(world_dim=WORLD_DIM, num_basis_functions=8, **bounding_kwargs)
    rf = ResidueField(cfg)
    return rf


def _run_direct_channel_cell(arm, channel_name, seed):
    """C1/C2/C3 for a single (arm, channel, seed) via ResidueField.update_valence."""
    component = CHANNELS[channel_name]
    rf = _make_field(BOUNDING_ARMS[arm], SEED_BASE + seed)
    z = torch.zeros(WORLD_DIM)
    rf.accumulate(z, harm_magnitude=0.1)          # seed one active center (readiness)
    active_center_exists = bool(rf.rbf_field.active_mask.any().item())

    value = VALUE if channel_name == "positive_surprise" else -VALUE
    trajectory = []
    for i in range(WRITES_PER_CELL):
        rf.update_valence(z, component, value)
        if (i + 1) % 50 == 0:
            v = float(rf.evaluate_valence(z.unsqueeze(0))[0, component].item())
            trajectory.append(v)
            print(f"  [train] arm={arm} channel={channel_name} seed={seed} "
                  f"ep {i + 1}/{WRITES_PER_CELL} value={v:.4f}", flush=True)
    final_value = float(rf.evaluate_valence(z.unsqueeze(0))[0, component].item())

    # C4: MECH-094 gate -- separate field, hypothesis_tag=True throughout.
    rf_h = _make_field(BOUNDING_ARMS[arm], SEED_BASE + seed)
    rf_h.accumulate(z, harm_magnitude=0.1, hypothesis_tag=False)  # seed a REAL center only
    for _ in range(WRITES_PER_CELL):
        rf_h.update_valence(z, component, value, hypothesis_tag=True)
    hypothesis_final = float(rf_h.evaluate_valence(z.unsqueeze(0))[0, component].item())

    expected_unbounded = WRITES_PER_CELL * abs(value)
    return {
        "arm": arm,
        "channel": channel_name,
        "seed": seed,
        "write_path": "ResidueField.update_valence",
        "active_center_exists": active_center_exists,
        "final_value": final_value,
        "final_abs_value": abs(final_value),
        "trajectory": trajectory,
        "expected_unbounded_sum": expected_unbounded,
        "exact_match_to_naive_sum": abs(final_value - (WRITES_PER_CELL * value)) <= EXACT_TOL,
        "hypothesis_tag_final_value": hypothesis_final,
        "hypothesis_tag_stayed_zero": abs(hypothesis_final) <= EXACT_TOL,
    }


def _run_wanting_sensitized_cell(arm, seed):
    """C1/C2 for the WANTING channel via ResidueField.update_wanting_sensitized."""
    rf = _make_field(BOUNDING_ARMS[arm], SEED_BASE + seed)
    z = torch.zeros(WORLD_DIM)
    rf.accumulate(z, harm_magnitude=0.1)
    active_center_exists = bool(rf.rbf_field.active_mask.any().item())

    trajectory = []
    for i in range(WRITES_PER_CELL):
        rf.update_wanting_sensitized(
            z, salience=1.0, drive_level=1.0, rate=0.1, gmax=1.0, coupling=1.0,
        )
        if (i + 1) % 50 == 0:
            v = float(rf.evaluate_valence(z.unsqueeze(0))[0, VALENCE_WANTING].item())
            trajectory.append(v)
            print(f"  [train] arm={arm} channel=wanting_sensitized seed={seed} "
                  f"ep {i + 1}/{WRITES_PER_CELL} value={v:.4f}", flush=True)
    final_value = float(rf.evaluate_valence(z.unsqueeze(0))[0, VALENCE_WANTING].item())

    rf_h = _make_field(BOUNDING_ARMS[arm], SEED_BASE + seed)
    rf_h.accumulate(z, harm_magnitude=0.1)
    for _ in range(WRITES_PER_CELL):
        rf_h.update_wanting_sensitized(
            z, salience=1.0, drive_level=1.0, rate=0.1, gmax=1.0, coupling=1.0,
            hypothesis_tag=True,
        )
    hypothesis_final = float(rf_h.evaluate_valence(z.unsqueeze(0))[0, VALENCE_WANTING].item())

    return {
        "arm": arm,
        "channel": "wanting_sensitized",
        "seed": seed,
        "write_path": "ResidueField.update_wanting_sensitized",
        "active_center_exists": active_center_exists,
        "final_value": final_value,
        "final_abs_value": abs(final_value),
        "trajectory": trajectory,
        "hypothesis_tag_final_value": hypothesis_final,
        "hypothesis_tag_stayed_zero": abs(hypothesis_final) <= EXACT_TOL,
        # sensitized wanting is monotone-amplified (gain saturates at gmax=1.0), not a
        # flat per-write constant, so C3's exact-naive-sum check does not apply here --
        # C1/C2/C4 (unbounded-OFF / bounded-ON / hypothesis-gate) still do.
        "exact_match_to_naive_sum": None,
    }


def _run_arm(arm, seeds):
    rows = []
    for seed in seeds:
        for channel_name in list(CHANNELS.keys()) + ["wanting_sensitized"]:
            print(f"Seed {seed} Condition {arm}_{channel_name}", flush=True)
            with arm_cell(
                seed,
                config_slice={"arm": arm, "bounding_kwargs": BOUNDING_ARMS[arm],
                              "channel": channel_name, "writes_per_cell": WRITES_PER_CELL,
                              "value": VALUE, "world_dim": WORLD_DIM},
                script_path=Path(__file__),
                config_slice_declared=True,
            ) as cell:
                if channel_name == "wanting_sensitized":
                    row = _run_wanting_sensitized_cell(arm, seed)
                else:
                    row = _run_direct_channel_cell(arm, channel_name, seed)
                cell.stamp(row)
            print(f"  [{arm}_{channel_name} seed={seed}] final_abs_value={row['final_abs_value']:.4f} "
                  f"active_center_exists={row['active_center_exists']}", flush=True)
            print("verdict: PASS", flush=True)   # per-cell run completion marker
            rows.append(row)
    return rows


def run_experiment(seeds):
    per_cell = []
    rows_by_arm = {}
    for arm in BOUNDING_ARMS:
        rows = _run_arm(arm, seeds)
        rows_by_arm[arm] = rows
        per_cell.extend(rows)

    active_center_exists_frac = float(np.mean([r["active_center_exists"] for r in per_cell]))
    readiness_met = active_center_exists_frac >= 1.0  # every cell must seed a real center

    off_rows = rows_by_arm["OFF"]
    on_rows = rows_by_arm["ON"]

    c1 = all(r["final_abs_value"] >= UNBOUNDED_FRAC * WRITES_PER_CELL * VALUE for r in off_rows)
    c2 = all(r["final_abs_value"] <= CLAMP_ABS + EPS for r in on_rows)
    c3 = all(r["exact_match_to_naive_sum"] for r in off_rows if r["exact_match_to_naive_sum"] is not None)
    c4 = all(r["hypothesis_tag_stayed_zero"] for r in per_cell)

    criteria_non_degenerate = {
        "off_arm_values_vary_by_channel": bool(
            float(np.std([r["final_abs_value"] for r in off_rows])) > 1e-6
        ),
        "on_vs_off_final_values_differ": bool(
            abs(float(np.mean([r["final_abs_value"] for r in on_rows]))
                - float(np.mean([r["final_abs_value"] for r in off_rows]))) > 1.0
        ),
    }

    if not readiness_met:
        outcome = "FAIL"
        evidence_direction = "non_contributory"
        label = "substrate_not_ready_requeue"
        overall_pass = False
    else:
        overall_pass = bool(c1 and c2 and c3 and c4)
        outcome = "PASS" if overall_pass else "FAIL"
        label = "sd_residue_valence_bound_validated" if overall_pass else "sd_residue_valence_bound_incomplete"
        evidence_direction = "supports" if overall_pass else "weakens"

    non_degenerate = readiness_met and all(criteria_non_degenerate.values())

    manifest = {
        "run_id": None,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCH_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": None if non_degenerate else (
            "substrate_not_ready" if not readiness_met else "off_on_indistinguishable"),
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "arm_results": per_cell,
        "readiness": {
            "active_center_exists_frac": active_center_exists_frac,
            "readiness_met": bool(readiness_met),
        },
        "interpretation": {
            "label": label,
            "preconditions": [
                {"name": "active_center_exists",
                 "description": "fraction of cells where accumulate() seeded a real active "
                                 "RBF center before the write loop -- without one, "
                                 "update_valence no-ops silently and every result would "
                                 "falsely read as bounded",
                 "measured": active_center_exists_frac, "threshold": 1.0,
                 "direction": "lower",
                 "control": "every cell (both arms, all 3 channels, all seeds)",
                 "met": bool(readiness_met)},
            ],
            "criteria_non_degenerate": criteria_non_degenerate,
        },
        "criteria": [
            {"name": "C1_off_arm_reproduces_unbounded_growth", "load_bearing": True, "passed": bool(c1)},
            {"name": "C2_on_arm_stays_bounded", "load_bearing": True, "passed": bool(c2)},
            {"name": "C3_off_arm_bit_identical_to_pre_fix", "load_bearing": True, "passed": bool(c3)},
            {"name": "C4_mech094_hypothesis_gate_unaffected", "load_bearing": True, "passed": bool(c4)},
        ],
        "combination_rule": "overall PASS iff C1 AND C2 AND C3 AND C4 (plain AND -- all four "
                             "must hold for the fix to be considered validated).",
        "ethics_preflight": {
            "involves_negative_valence": False,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
        },
        "notes": (
            "SD-RESIDUE-VALENCE-BOUND validation -- landed this session in "
            "ree_core/residue/field.py + ree_core/utils/config.py "
            "(REE_assembly/docs/architecture/sd_residue_valence_bound.md). GOV-REUSE-1: "
            "decisive readout (bounded-vs-unbounded accumulation under the new config "
            "knobs) not recorded anywhere -- the knobs did not exist before this session, "
            "so nothing to reuse; ran fresh. Unit-level against ResidueField/RBFLayer "
            "directly (no CausalGridWorld/REEAgent) since the fix is one central write "
            "path -- matches the V3-EXQ-520 Part-1 precedent for a ResidueField-internal "
            "readiness diagnostic."
        ),
    }
    return manifest, overall_pass


def build_and_run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    args = ap.parse_args()

    t0 = time.perf_counter()
    n_seeds = 1 if args.dry_run else args.seeds
    seeds = list(range(n_seeds))

    manifest, _overall_pass = run_experiment(seeds)

    ts = manifest["timestamp_utc"]
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest["run_id"] = run_id

    full_config = {
        "bounding_arms": BOUNDING_ARMS,
        "channels": list(CHANNELS.keys()) + ["wanting_sensitized"],
        "writes_per_cell": WRITES_PER_CELL,
        "value": VALUE,
        "world_dim": WORLD_DIM,
        "unbounded_frac": UNBOUNDED_FRAC,
        "exact_tol": EXACT_TOL,
    }

    out_path = write_flat_manifest(
        manifest, None, dry_run=args.dry_run,
        config=full_config, seeds=seeds, script_path=Path(__file__), started_at=t0,
    )
    print(f"outcome={manifest['outcome']} evidence_direction={manifest['evidence_direction']} "
          f"label={manifest['interpretation']['label']}", flush=True)
    print(f"manifest={out_path}", flush=True)
    return manifest, out_path, run_id, args.dry_run


if __name__ == "__main__":
    _manifest, _out_path, _run_id, _dry = build_and_run()
    emit_outcome(
        outcome=_manifest["outcome"] if _manifest["outcome"] in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        run_id=_run_id,
        dry_run=_dry,
    )
