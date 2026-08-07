"""
V3-EXQ-891: MECH-286 override-gated sleep-onset THREE-TERM CONJUNCTION signature.

SLEEP DRIVER: K=never (SleepLoopManager built but no cycle fired; the MECH-286
gate is evaluated directly via evaluate_sleep_onset_permit at each cell -- the DV
is the gate's representational signature, not any downstream sleep behaviour).

GOV-CONFIRM-1 confirming run for MECH-286 (sleep.override_gated_state_transition).
First SCORING experimental evidence: the prior run
v3_exq_599a_mech286_sleep_onset_gate_validation was experiment_purpose="diagnostic"
(substrate smoke) and therefore scored nothing (exp_conf=0, genuine_exp_count=0).

WHAT THIS TESTS (wall-independent representational/functional signature, per the
GOV-CONFIRM-1 lane -- NOT a behavioural/reward proxy):

  The claim asserts sleep-onset is a POSITIVE RECRUITMENT gated by a three-way
  conjunction, one regulator (SD-037 override) at a second decision point:

      permitted = (override_signal    <  theta_sleep_permit)     # orexin wake-stability
              AND (max staleness      >  theta_sleep_recruit)     # MECH-284 recruitment demand
              AND (z_harm_a tonic norm <  threat_tonic_threshold) # permissive threat context

  DV = the gate signature read directly off its three inputs
  (evaluate_sleep_onset_permit -> permitted + the three per-term _ok flags + the
  raw term values). Each of the three terms is manipulated INDEPENDENTLY in a full
  2x2x2 factorial, so the design tests:

    - NECESSITY of each term  : flipping ONE term to its blocking level -> permitted 0
        * override-block cell (C1) IS the claim's pre-registered hyperarousal-lesion
          falsifier (pin override high under otherwise-permissive conditions).
        * threat-block cell (C3) is the NOVEL third conjunct -- 599a always left
          z_harm_a at 0 (threat_ok True by default) and never exercised it.
    - SUFFICIENCY of the conjunction (C4): permit+recruit+safe -> permitted 1.
    - INSUFFICIENCY of any single permissive term: every cell with >=1 blocking
      term -> permitted 0 (a single permissive term never forces sleep).
    - CONJUNCTION EXACTNESS (C5, LOAD-BEARING): the full 8-cell permitted vector
      equals the predicted AND vector across ALL seeds.

NON-DEGENERACY (anti-tautology guard -- this is what makes it a measurement, not a
unit test of Python `and`): each factor's two levels must be measured on OPPOSITE
sides of its threshold, i.e. the manipulation ACTUALLY moved the term. The override
term in particular is driven through the REAL SD-037 regulator dynamics
(broadcast_override.tick) rather than pinned -- whether ticking it with high
drive/threat pushes override above theta_sleep_permit is an empirical property of
the regulator, not a given.

DV-SYMMETRY (per queue-experiment skill): the DV `permitted` is a threshold
conjunction of the three terms; each manipulation moves a term ACROSS its own
threshold, flipping the corresponding _ok flag and hence permitted. The
manipulation is therefore NOT invariant under any symmetry of the DV (it is
designed to flip the gate) -- so no delta here is an arithmetic identity fixed
before the run.

SCOPE / honesty note: this confirms that the BUILT gate substrate implements the
claimed three-way conjunction signature under independent manipulation of all three
terms across seeds. It does NOT test that the upstream producers naturally generate
these term values inside a full task -- that would be a downstream behavioural test,
which the GOV-CONFIRM-1 lane explicitly excludes.

Substrate: ree_core/sleep/sleep_onset_gate.py (built 2026-05-21),
REEConfig.use_mech286_sleep_onset_gate. New question vs 599/599a -> new number.
"""

from __future__ import annotations

import argparse
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.sleep.sleep_onset_gate import evaluate_sleep_onset_permit  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_891_mech286_sleep_onset_conjunction_signature"
QUEUE_ID = "V3-EXQ-891"
CLAIM_IDS: List[str] = ["MECH-286"]
EXPERIMENT_PURPOSE = "evidence"
SLEEP_DRIVER_PATTERN = "K=never (SleepLoopManager built; MECH-286 gate evaluated directly at each cell)"

SEEDS = [42, 43, 45]  # 44 avoided (known reef-config per-seed instability)

# MECH-286 gate thresholds (REEConfig defaults; pre-registered here).
THETA_SLEEP_PERMIT = 0.5
THETA_SLEEP_RECRUIT = 0.3
THREAT_TONIC_THRESHOLD = 0.4

# Manipulation levels (pre-registered, in the script -- not derived post-hoc).
OVERRIDE_BLOCK_TICKS = 50           # ticks of high drive/threat -> override -> ~0.8 (> permit)
OVERRIDE_PERMIT_TICKS = 10          # ticks of zero drive/threat -> override stays low (< permit)
OVERRIDE_BLOCK_DRIVE = 0.95
OVERRIDE_BLOCK_HARM = 0.85
STALENESS_RECRUIT_VALUE = 0.8       # > theta_sleep_recruit
STALENESS_NORECRUIT_VALUE = 0.0     # <= theta_sleep_recruit (empty accumulator)
THREAT_SAFE_NORM = 0.10             # z_harm_a tonic norm < threat_tonic_threshold
THREAT_HIGH_NORM = 0.90             # z_harm_a tonic norm > threat_tonic_threshold


def _factorial_arms() -> List[Dict[str, Any]]:
    """The 8-cell 2x2x2 factorial over the three gate terms."""
    arms: List[Dict[str, Any]] = []
    for ov in ("permit", "block"):
        for st in ("recruit", "norecruit"):
            for th in ("safe", "threat"):
                predicted = 1.0 if (ov == "permit" and st == "recruit" and th == "safe") else 0.0
                arms.append({
                    "arm": f"OV_{ov}__ST_{st}__TH_{th}",
                    "override_level": ov,
                    "stale_level": st,
                    "threat_level": th,
                    "predicted_permitted": predicted,
                })
    return arms


ARMS = _factorial_arms()

# config_slice for the per-cell arm fingerprint (multi-arm requirement). Every
# cell runs the same substrate config; the arms differ only by runtime state pokes.
_CFG_KWARGS = dict(
    body_obs_dim=12,
    world_obs_dim=250,
    action_dim=4,
    use_sleep_loop=True,
    sleep_loop_episodes_K=10_000,  # K=never: no cycle fires during this run
    sws_enabled=True,
    rem_enabled=False,
    use_staleness_accumulator=True,
    use_e2_harm_a=True,
    use_broadcast_override=True,
    use_mech286_sleep_onset_gate=True,
)


def _build_agent() -> REEAgent:
    cfg = REEConfig.from_dims(**_CFG_KWARGS)
    return REEAgent(cfg)


def _drive_override(agent: REEAgent, level: str) -> None:
    reg = agent.broadcast_override
    if reg is None:
        raise RuntimeError("broadcast_override required (use_broadcast_override=True)")
    if level == "block":
        for _ in range(OVERRIDE_BLOCK_TICKS):
            reg.tick(drive_level=OVERRIDE_BLOCK_DRIVE, z_harm_norm=OVERRIDE_BLOCK_HARM)
    elif level == "permit":
        for _ in range(OVERRIDE_PERMIT_TICKS):
            reg.tick(drive_level=0.0, z_harm_norm=0.0)
    else:
        raise ValueError(f"unknown override_level {level}")


def _set_staleness(agent: REEAgent, level: str) -> None:
    acc = agent.hippocampal.staleness_accumulator
    if acc is None:
        raise RuntimeError("staleness accumulator required (use_staleness_accumulator=True)")
    value = STALENESS_RECRUIT_VALUE if level == "recruit" else STALENESS_NORECRUIT_VALUE
    # norecruit: leave the accumulator empty (max staleness 0.0). recruit: seed one
    # region above the recruit threshold (the exposed instrumentation, as 599a).
    if level == "recruit":
        acc._staleness[("fast", "0.0")] = float(value)


def _set_threat_latent(agent: REEAgent, level: str) -> None:
    target_norm = THREAT_HIGH_NORM if level == "threat" else THREAT_SAFE_NORM
    v = torch.ones(4, dtype=torch.float32)
    v = v / v.norm() * float(target_norm)
    # The gate reads getattr(agent._current_latent, "z_harm_a", None).norm(); a
    # minimal stand-in with just the z_harm_a channel exercises exactly that path.
    agent._current_latent = types.SimpleNamespace(z_harm_a=v)


def run_cell(arm: Dict[str, Any], seed: int) -> Dict[str, Any]:
    with arm_cell(seed, config_slice=_CFG_KWARGS, script_path=Path(__file__)) as cell:
        agent = _build_agent()
        _drive_override(agent, arm["override_level"])
        _set_staleness(agent, arm["stale_level"])
        _set_threat_latent(agent, arm["threat_level"])

        permitted, diag = evaluate_sleep_onset_permit(agent)

        row = {
            "arm": arm["arm"],
            "seed": seed,
            "override_level": arm["override_level"],
            "stale_level": arm["stale_level"],
            "threat_level": arm["threat_level"],
            "predicted_permitted": arm["predicted_permitted"],
            "permitted": 1.0 if permitted else 0.0,
            "override_signal": float(diag["mech286_override_signal"]),
            "staleness_max": float(diag["mech286_staleness_max"]),
            "z_harm_a_norm": float(diag["mech286_z_harm_a_norm"]),
            "override_ok": float(diag["mech286_override_ok"]),
            "staleness_ok": float(diag["mech286_staleness_ok"]),
            "threat_ok": float(diag["mech286_threat_ok"]),
        }
        cell.stamp(row)
    return row


def evaluate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    checks: Dict[str, bool] = {}

    # C5 (load-bearing): conjunction exactness across ALL cells and seeds.
    c5 = all(r["permitted"] == r["predicted_permitted"] for r in results)
    checks["C5_conjunction_exact"] = c5

    def cells(ov=None, st=None, th=None):
        return [r for r in results
                if (ov is None or r["override_level"] == ov)
                and (st is None or r["stale_level"] == st)
                and (th is None or r["threat_level"] == th)]

    # C1 override necessity / hyperarousal-lesion falsifier: override-block, else permissive -> 0.
    checks["C1_override_necessity_falsifier"] = all(
        r["permitted"] == 0.0 for r in cells(ov="block", st="recruit", th="safe"))
    # C2 staleness necessity: no recruitment demand -> 0.
    checks["C2_staleness_necessity"] = all(
        r["permitted"] == 0.0 for r in cells(ov="permit", st="norecruit", th="safe"))
    # C3 threat necessity (NOVEL third conjunct): high z_harm_a tonic -> 0.
    checks["C3_threat_necessity_novel"] = all(
        r["permitted"] == 0.0 for r in cells(ov="permit", st="recruit", th="threat"))
    # C4 sufficiency: all three permissive -> 1.
    checks["C4_conjunction_sufficiency"] = all(
        r["permitted"] == 1.0 for r in cells(ov="permit", st="recruit", th="safe"))

    # Non-degeneracy: each factor's two levels straddle its threshold (manipulation moved the term).
    ov_permit = [r["override_signal"] for r in cells(ov="permit")]
    ov_block = [r["override_signal"] for r in cells(ov="block")]
    st_recruit = [r["staleness_max"] for r in cells(st="recruit")]
    st_norecruit = [r["staleness_max"] for r in cells(st="norecruit")]
    th_safe = [r["z_harm_a_norm"] for r in cells(th="safe")]
    th_threat = [r["z_harm_a_norm"] for r in cells(th="threat")]

    nd_override = max(ov_permit) < THETA_SLEEP_PERMIT <= min(ov_block)
    nd_staleness = min(st_recruit) > THETA_SLEEP_RECRUIT >= max(st_norecruit)
    nd_threat = max(th_safe) < THREAT_TONIC_THRESHOLD <= min(th_threat)
    non_degenerate = bool(nd_override and nd_staleness and nd_threat)
    degeneracy_reason = None
    if not non_degenerate:
        bad = []
        if not nd_override:
            bad.append(f"override levels did not straddle {THETA_SLEEP_PERMIT} "
                       f"(permit_max={max(ov_permit):.4f}, block_min={min(ov_block):.4f})")
        if not nd_staleness:
            bad.append(f"staleness levels did not straddle {THETA_SLEEP_RECRUIT} "
                       f"(recruit_min={min(st_recruit):.4f}, norecruit_max={max(st_norecruit):.4f})")
        if not nd_threat:
            bad.append(f"threat levels did not straddle {THREAT_TONIC_THRESHOLD} "
                       f"(safe_max={max(th_safe):.4f}, threat_min={min(th_threat):.4f})")
        degeneracy_reason = "; ".join(bad)

    passed = non_degenerate and all(checks.values())
    return {
        "checks": checks,
        "criteria": [
            {"name": "C5_conjunction_exact", "load_bearing": True, "passed": checks["C5_conjunction_exact"]},
            {"name": "C1_override_necessity_falsifier", "load_bearing": False, "passed": checks["C1_override_necessity_falsifier"]},
            {"name": "C2_staleness_necessity", "load_bearing": False, "passed": checks["C2_staleness_necessity"]},
            {"name": "C3_threat_necessity_novel", "load_bearing": False, "passed": checks["C3_threat_necessity_novel"]},
            {"name": "C4_conjunction_sufficiency", "load_bearing": False, "passed": checks["C4_conjunction_sufficiency"]},
        ],
        "combination_rule": "PASS = non_degenerate AND C1 AND C2 AND C3 AND C4 AND C5 (all seeds)",
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "passed": passed,
    }


def main(dry_run: bool = False, seeds: List[int] | None = None) -> Dict[str, Any]:
    use_seeds = SEEDS if seeds is None else seeds
    t0 = time.perf_counter()

    results: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in use_seeds:
            print(f"Seed {seed} Condition {arm['arm']}", flush=True)
            results.append(run_cell(arm, seed))

    verdict = evaluate(results)
    outcome = "PASS" if verdict["passed"] else "FAIL"

    # Single-unit progress line for the runner (no training loop in this probe).
    print("  [eval] mech286 conjunction signature ep 1/1", flush=True)
    print(f"verdict: {outcome}", flush=True)
    for k, v in verdict["checks"].items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}", flush=True)
    print(f"  non_degenerate: {verdict['non_degenerate']}", flush=True)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    manifest: Dict[str, Any] = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "evidence_class": "simulation",
        "evidence_direction": "supports" if verdict["passed"] else "weakens",
        "evidence_direction_per_claim": {
            "MECH-286": "supports" if verdict["passed"] else "weakens",
        },
        "sleep_driver_pattern": SLEEP_DRIVER_PATTERN,
        "outcome": outcome,
        "non_degenerate": verdict["non_degenerate"],
        "degeneracy_reason": verdict["degeneracy_reason"],
        "thresholds": {
            "theta_sleep_permit": THETA_SLEEP_PERMIT,
            "theta_sleep_recruit": THETA_SLEEP_RECRUIT,
            "threat_tonic_threshold": THREAT_TONIC_THRESHOLD,
        },
        "interpretation": {
            "label": "three_term_conjunction_confirmed" if verdict["passed"] else "conjunction_signature_deviated",
            "checks": verdict["checks"],
            "criteria": verdict["criteria"],
            "combination_rule": verdict["combination_rule"],
        },
        "arm_results": results,
        "custom_information": {
            "supersedes_note": "New question (three-term conjunction signature); NOT a lettered fix to V3-EXQ-599/599a (those were substrate smoke).",
            "prior_diagnostic": "v3_exq_599a_mech286_sleep_onset_gate_validation (experiment_purpose=diagnostic, scored 0)",
            "n_cells": len(results),
        },
        "dry_run": bool(dry_run),
    }

    # Representative agent so write_flat_manifest records enabled_default_off_flags
    # (confirms use_mech286_sleep_onset_gate was ON). z_goal is unused here, so the
    # z_goal block correctly reads unmeasured -- not a WRITER DEFECT.
    rep_agent = _build_agent()
    out_path = write_flat_manifest(
        manifest,
        dry_run=dry_run,
        config=_CFG_KWARGS,
        seeds=use_seeds,
        script_path=Path(__file__),
        started_at=t0,
        agent=rep_agent,
    )
    print(f"Result written to: {out_path}", flush=True)

    return {
        "outcome": outcome,
        "all_pass": bool(verdict["passed"]),
        "manifest_path": str(out_path),
        "run_id": run_id,
        "dry_run": bool(dry_run),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    args = parser.parse_args()
    result = main(dry_run=args.dry_run, seeds=args.seeds)
    emit_outcome(
        outcome=result["outcome"] if result["outcome"] in ("PASS", "FAIL") else "FAIL",
        manifest_path=result["manifest_path"],
        run_id=result["run_id"],
        queue_id=QUEUE_ID,
        dry_run=args.dry_run,
    )
    sys.exit(0 if result["all_pass"] else 1)
