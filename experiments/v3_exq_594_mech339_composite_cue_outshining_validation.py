"""V3-EXQ-594: MECH-339 (C1) Composite Cue + Outshining Gate -- Substrate Validation.

experiment_purpose: diagnostic
status_when_drafted: lands with the MECH-339 C1 substrate (smallest step,
2026-05-19). Substrate readiness gate; NOT governance evidence.

CLAIM TESTED:
MECH-339 (hippocampal.composite_retrieval_cue_outshining_gate) -- Constraint 1
of the retrieval-cue reframe (REE_assembly/docs/architecture/ghost_goal_search.md
Section 0.2; claims.yaml MECH-339 / ARC-078). The landed MECH-292 bank matched
the retrieval cue by z_goal cosine only and ignored the SD-039 payload fields
the match does not use. MECH-339 adds a context channel built from the
already-stored-but-match-unused payload (smallest step: arousal_tag only),
combined by an OUTSHINING gate so a strong direct goal_match suppresses the
context channel rather than it being summed in with fixed weight.

This is a pure-arithmetic substrate validation: deterministic anchor pools are
constructed directly via AnchorSet + AnchorGoalPayload (no env, no training, no
agent loop), then ranked under three GhostGoalBankConfig conditions over the
SAME pool. No phased training (no trainable parameters anywhere in the bank).

DESIGN: 4 sub-tests, each exercising one MECH-339 falsifiable prediction.

  T1 backward_compat_bit_identical: with use_composite_cue_outshining=False
     (default) the bank is bit-identical to the pre-MECH-339 four-term form.
     Every entry's components dict has exactly {wanting, goal_match,
     staleness, recoverability} (no "context" key); diagnostics
     component_sums carries no "context"; each ghost_priority equals an
     independently hand-computed 4-term composite within 1e-9; and
     sum(components.values()) == ghost_priority within 1e-9.

  T2 strong_match_outshines (falsifiable (i)): pool = {A_strong (goal_match
     >= outshine_pivot, low arousal), B_weak_hi (goal_match in
     (floor, pivot), high arousal)}. Enabling the context channel must NOT
     change the top-ranked entry: the calibration-independent core is that
     A_strong's gate is fully closed -> its context term is EXACTLY 0.0 and
     its ghost_priority is bit-identical OFF vs ON; additionally, at the
     calibrated context_weight, A_strong remains rank-1.

  T3 weak_match_context_changes_top (falsifiable (ii)): pool = {D_lo
     (goal_match in (floor, pivot), low arousal), E_hi (goal_match weaker
     than D but still in (floor, pivot), high arousal)}. No anchor reaches
     outshine_pivot. OFF the top entry is D_lo (higher direct match). ON
     the context channel must promote E_hi to rank-1 -- the top entry
     changes (D_lo -> E_hi).

  T4 ungated_degrades_top1 (falsifiable (iii)): pool = {A_strong (clean
     direct match, low arousal), B_weak_hi (weak match, high arousal)}.
     z_goal-only (OFF) ranks A_strong top (correct). The properly GATED
     channel keeps A_strong top (the gate protects the clean match). An
     UNGATED fixed-weight additive context term -- simulated by setting
     outshine_pivot huge so gate ~ 1.0 for every anchor -- lets B_weak_hi
     overtake A_strong, degrading top-1 relative to z_goal-only. PASS
     requires: OFF top == A, gated-ON top == A, ungated top == B (!= A).

PASS CRITERIA: T1 AND T2 AND T3 AND T4.

DIAGNOSTIC INTERPRETATION GRID (one row per outcome -> next action):

  | Outcome                          | Interpretation                         | Next action |
  |----------------------------------|----------------------------------------|-------------|
  | T1 & T2 & T3 & T4 all PASS       | C1 substrate validated: outshining     | Substrate ready; MECH-339 stays |
  |                                  | gate behaves per all three falsifiable | candidate/v3_pending. Wire the  |
  |                                  | predictions; backward compatible.      | MECH-293 behavioural consumer.  |
  | T1 FAIL                          | Backward-compat regression: defaults   | Fix ghost_goal_bank.py so the   |
  |                                  | are no longer no-op (context leaking   | "context" channel is absent     |
  |                                  | with master switch off).               | when the master switch is off.  |
  | T2 FAIL (A ctx != 0 / A moved)   | Outshining not effective: a strong     | Recheck _outshine_gate pivot    |
  |                                  | direct match does not close the gate / | math; recalibrate outshine_     |
  |                                  | context_weight too large for pivot.    | pivot / context_weight.         |
  | T3 FAIL (top unchanged)          | Context channel inert: it never        | arousal_scale too large or      |
  |                                  | changes ordering even with the direct  | context_weight too small --     |
  |                                  | match weak.                            | recalibrate; re-run.            |
  | T4 FAIL (ungated keeps A top)    | The gate is doing no work -- removing  | Regression in _outshine_gate    |
  |                                  | it does not degrade top-1, so the      | (gate not a function of         |
  |                                  | composite reduces to z_goal-only.      | goal_match). Fix and re-run.    |

claim_ids: ['MECH-339'] (single-claim diagnostic; ARC-078 is the parent
architectural framing and MECH-292 is the host bank -- both out of scope as
direct test targets here).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_594_mech339_composite_cue_outshining_validation.py

Writes a flat JSON manifest to REE_assembly/evidence/experiments/.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome
from ree_core.hippocampal.anchor_set import Anchor, AnchorGoalPayload, AnchorSet
from ree_core.hippocampal.ghost_goal_bank import GhostGoalBank
from ree_core.utils.config import AnchorSetConfig, GhostGoalBankConfig

EVIDENCE_ROOT = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

EXPERIMENT_PURPOSE = "diagnostic"  # substrate-readiness; excluded from governance scoring

# Pre-registered constants (defined here, NOT inferred from run statistics).
CURRENT_Z_GOAL = torch.tensor([1.0, 0.0, 0.0, 0.0])
OUTSHINE_PIVOT = 0.5
AROUSAL_SCALE = 1.0
CONTEXT_WEIGHT = 1.0          # calibrated: keeps a clean match top when gated,
                              # flips when ungated (see T4 derivation).
UNGATED_PIVOT = 1.0e9         # gate ~ 1.0 for every anchor -> fixed-weight add.
LOW_AROUSAL = 0.05
HIGH_AROUSAL = 6.0
TOL = 1e-9


def _make_anchor(
    seg: str,
    zsnap: List[float],
    arousal: float,
    wanting: float = 0.2,
    last_vs: float = 0.6,
) -> Anchor:
    """Construct one inactive anchor with a populated SD-039 payload.

    Inactive so the bank's default pool (include_inactive=True,
    include_active=False) scores it.
    """
    a = Anchor(key=("fast", seg, ("s",)), z_world=torch.zeros(4), active=False)
    a.goal_payload = AnchorGoalPayload(
        z_goal_snapshot=torch.tensor(zsnap, dtype=torch.float32).unsqueeze(0),
        wanting_strength=wanting,
        arousal_tag=arousal,
        last_vs=last_vs,
        staleness_at_write=0.0,
        payload_written_step=0,
    )
    return a


def _bank(anchors: List[Anchor], cfg: GhostGoalBankConfig) -> GhostGoalBank:
    """A GhostGoalBank over a fixed pool. No StalenessAccumulator -> the
    proxy staleness is (tick - last_accessed) * rate; with a fresh AnchorSet
    (_tick=0) and last_accessed=0 this is deterministically 0.0."""
    s = AnchorSet(AnchorSetConfig())
    s._all = {a.key: a for a in anchors}
    return GhostGoalBank(cfg, s)


def _cfg_off() -> GhostGoalBankConfig:
    return GhostGoalBankConfig()


def _cfg_on(pivot: float = OUTSHINE_PIVOT) -> GhostGoalBankConfig:
    return GhostGoalBankConfig(
        use_composite_cue_outshining=True,
        context_weight=CONTEXT_WEIGHT,
        outshine_pivot=pivot,
        arousal_scale=AROUSAL_SCALE,
    )


def _top_key(bank: GhostGoalBank, z_goal: torch.Tensor) -> str:
    entries = bank.rank(z_goal)
    return entries[0].anchor.key[1] if entries else ""


# ---------------------------------------------------------------------------
# Sub-tests
# ---------------------------------------------------------------------------
def run_t1_backward_compat() -> Dict[str, Any]:
    """T1: master OFF -> bit-identical to the pre-MECH-339 four-term bank."""
    try:
        a = _make_anchor("A", [1.0, 0.0, 0.0, 0.0], LOW_AROUSAL)
        d = _make_anchor("D", [0.2041, 1.0, 0.0, 0.0], LOW_AROUSAL)
        e = _make_anchor("E", [0.1209, 1.0, 0.0, 0.0], HIGH_AROUSAL)
        cfg = _cfg_off()
        bank = _bank([a, d, e], cfg)
        entries = bank.rank(CURRENT_Z_GOAL)
        diag = bank.get_diagnostics()

        required = {"wanting", "goal_match", "staleness", "recoverability"}
        keys_ok = True
        no_context_key = True
        max_sum_diff = 0.0
        max_hand_diff = 0.0
        for ent in entries:
            ks = set(ent.components.keys())
            if ks != required:
                keys_ok = False
            if "context" in ks:
                no_context_key = False
            max_sum_diff = max(
                max_sum_diff,
                abs(sum(ent.components.values()) - ent.ghost_priority),
            )
            # Independent 4-term recomputation.
            gm = ent.anchor.goal_match(CURRENT_Z_GOAL)
            pay = ent.anchor.goal_payload
            hand = (
                cfg.wanting_weight * float(pay.wanting_strength)
                + cfg.goal_match_weight * gm
                + cfg.staleness_weight * 0.0
                + cfg.recoverability_weight * float(pay.last_vs)
            )
            max_hand_diff = max(
                max_hand_diff, abs(hand - ent.ghost_priority)
            )

        sums_no_context = "context" not in diag.get("component_sums", {})
        ok = (
            len(entries) == 3
            and keys_ok
            and no_context_key
            and sums_no_context
            and max_sum_diff < TOL
            and max_hand_diff < TOL
        )
        return {
            "pass": bool(ok),
            "n_entries": len(entries),
            "components_keys_exactly_four": bool(keys_ok),
            "no_context_in_components": bool(no_context_key),
            "no_context_in_diag_sums": bool(sums_no_context),
            "max_sum_vs_priority_diff": float(max_sum_diff),
            "max_handcomputed_diff": float(max_hand_diff),
        }
    except Exception as exc:  # noqa: BLE001
        return {"pass": False, "error": repr(exc)}


def run_t2_strong_match_outshines() -> Dict[str, Any]:
    """T2 (falsifiable i): a strong direct match closes the gate; enabling
    the context channel does not change the top entry."""
    try:
        a = _make_anchor("A", [1.0, 0.0, 0.0, 0.0], LOW_AROUSAL)        # gm 1.0
        b = _make_anchor("B", [0.1496, 1.0, 0.0, 0.0], HIGH_AROUSAL)    # gm ~0.148
        gm_a = a.goal_match(CURRENT_Z_GOAL)
        gm_b = b.goal_match(CURRENT_Z_GOAL)

        off = _bank([a, b], _cfg_off())
        on = _bank([a, b], _cfg_on())
        e_off = {x.anchor.key[1]: x for x in off.rank(CURRENT_Z_GOAL)}
        e_on = {x.anchor.key[1]: x for x in on.rank(CURRENT_Z_GOAL)}

        a_ctx = e_on["A"].components.get("context", None)
        a_prio_off = e_off["A"].ghost_priority
        a_prio_on = e_on["A"].ghost_priority
        top_off = _top_key(off, CURRENT_Z_GOAL)
        top_on = _top_key(on, CURRENT_Z_GOAL)

        # Calibration-independent core: gm_a >= pivot -> gate 0 -> ctx 0,
        # and A's priority is bit-identical OFF vs ON.
        gate_closed = (a_ctx == 0.0)
        a_priority_unchanged = abs(a_prio_on - a_prio_off) < TOL
        top_unchanged = (top_off == top_on == "A")
        ok = (
            gm_a >= OUTSHINE_PIVOT
            and OUTSHINE_PIVOT > gm_b > _cfg_off().goal_match_floor
            and gate_closed
            and a_priority_unchanged
            and top_unchanged
        )
        return {
            "pass": bool(ok),
            "gm_strong": float(gm_a),
            "gm_weak": float(gm_b),
            "strong_anchor_context_term": float(a_ctx) if a_ctx is not None else None,
            "strong_priority_off": float(a_prio_off),
            "strong_priority_on": float(a_prio_on),
            "gate_fully_closed_on_strong": bool(gate_closed),
            "strong_priority_bit_identical": bool(a_priority_unchanged),
            "top_off": top_off,
            "top_on": top_on,
            "top_unchanged": bool(top_unchanged),
        }
    except Exception as exc:  # noqa: BLE001
        return {"pass": False, "error": repr(exc)}


def run_t3_weak_match_context_changes_top() -> Dict[str, Any]:
    """T3 (falsifiable ii): with the direct match weak/absent, enabling the
    context channel changes the top-ranked entry."""
    try:
        d = _make_anchor("D", [0.2041, 1.0, 0.0, 0.0], LOW_AROUSAL)   # gm ~0.20
        e = _make_anchor("E", [0.1209, 1.0, 0.0, 0.0], HIGH_AROUSAL)  # gm ~0.12
        gm_d = d.goal_match(CURRENT_Z_GOAL)
        gm_e = e.goal_match(CURRENT_Z_GOAL)
        floor = _cfg_off().goal_match_floor

        off = _bank([d, e], _cfg_off())
        on = _bank([d, e], _cfg_on())
        top_off = _top_key(off, CURRENT_Z_GOAL)
        top_on = _top_key(on, CURRENT_Z_GOAL)

        # Neither anchor reaches the pivot; OFF favours the stronger direct
        # match (D); ON must promote the high-arousal weaker match (E).
        no_strong_match = (gm_d < OUTSHINE_PIVOT and gm_e < OUTSHINE_PIVOT)
        both_above_floor = (gm_d > floor and gm_e > floor)
        ok = (
            no_strong_match
            and both_above_floor
            and top_off == "D"
            and top_on == "E"
            and top_off != top_on
        )
        return {
            "pass": bool(ok),
            "gm_d": float(gm_d),
            "gm_e": float(gm_e),
            "no_anchor_at_pivot": bool(no_strong_match),
            "both_above_floor": bool(both_above_floor),
            "top_off": top_off,
            "top_on": top_on,
            "context_changed_top": bool(top_off != top_on),
        }
    except Exception as exc:  # noqa: BLE001
        return {"pass": False, "error": repr(exc)}


def run_t4_ungated_degrades_top1() -> Dict[str, Any]:
    """T4 (falsifiable iii): an ungated fixed-weight context term degrades
    top-1 on a clean direct match; the gate is doing real work."""
    try:
        a = _make_anchor("A", [1.0, 0.0, 0.0, 0.0], LOW_AROUSAL)      # clean match
        b = _make_anchor("B", [0.1496, 1.0, 0.0, 0.0], HIGH_AROUSAL)  # weak, hi arousal

        top_zgoal_only = _top_key(_bank([a, b], _cfg_off()), CURRENT_Z_GOAL)
        top_gated = _top_key(_bank([a, b], _cfg_on()), CURRENT_Z_GOAL)
        # Ungated: outshine_pivot huge -> gate ~ 1.0 for every anchor, so the
        # context term is added with fixed weight regardless of goal_match.
        top_ungated = _top_key(
            _bank([a, b], _cfg_on(pivot=UNGATED_PIVOT)), CURRENT_Z_GOAL
        )

        ok = (
            top_zgoal_only == "A"
            and top_gated == "A"
            and top_ungated == "B"
            and top_ungated != top_zgoal_only
        )
        return {
            "pass": bool(ok),
            "top_zgoal_only": top_zgoal_only,
            "top_gated": top_gated,
            "top_ungated": top_ungated,
            "gate_protects_clean_match": bool(top_gated == "A"),
            "ungated_degrades_top1": bool(top_ungated != top_zgoal_only),
        }
    except Exception as exc:  # noqa: BLE001
        return {"pass": False, "error": repr(exc)}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_experiment(dry_run: bool = False) -> Tuple[Dict[str, Any], bool]:
    if dry_run:
        t1 = run_t1_backward_compat()
        metrics = {"T1_backward_compat_bit_identical": t1}
        all_pass = bool(t1["pass"])
        return metrics, all_pass
    t1 = run_t1_backward_compat()
    t2 = run_t2_strong_match_outshines()
    t3 = run_t3_weak_match_context_changes_top()
    t4 = run_t4_ungated_degrades_top1()
    metrics = {
        "T1_backward_compat_bit_identical": t1,
        "T2_strong_match_outshines": t2,
        "T3_weak_match_context_changes_top": t3,
        "T4_ungated_degrades_top1": t4,
    }
    all_pass = all(m["pass"] for m in metrics.values())
    return metrics, all_pass


def main(dry_run: bool = False) -> Dict[str, Any]:
    print("[v3_exq_594] MECH-339 C1 composite-cue + outshining-gate validation...",
          flush=True)
    metrics, all_pass = run_experiment(dry_run=dry_run)
    for name, m in metrics.items():
        print(f"  {name}: {'PASS' if m['pass'] else 'FAIL'}  {m}", flush=True)
    print(f"[v3_exq_594] overall: {'PASS' if all_pass else 'FAIL'}", flush=True)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"v3_exq_594_mech339_composite_cue_outshining_validation_{ts}_v3"
    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": "v3_exq_594_mech339_composite_cue_outshining_validation",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": ["MECH-339"],
        "evidence_direction": "supports" if all_pass else "weakens",
        "evidence_direction_per_claim": {
            "MECH-339": "supports" if all_pass else "weakens",
        },
        "evidence_direction_note": (
            "MECH-339 Constraint 1 substrate validation (diagnostic; NOT "
            "governance evidence). T1 confirms master-OFF is bit-identical to "
            "the pre-MECH-339 four-term bank (no 'context' channel leaking); "
            "T2 confirms a strong direct match closes the outshining gate "
            "(strong-anchor context term exactly 0.0, priority bit-identical, "
            "top entry unchanged -- falsifiable (i)); T3 confirms a weak "
            "direct match lets the context channel change the top entry -- "
            "falsifiable (ii); T4 confirms an ungated fixed-weight context "
            "term degrades top-1 on a clean direct match, i.e. the gate is "
            "doing real work -- falsifiable (iii). Behavioural validation "
            "belongs to the MECH-293 consumer and is out of scope here."
        ),
        "outcome": "PASS" if all_pass else "FAIL",
        "metrics": metrics,
        "dry_run": bool(dry_run),
    }

    out_path = None
    if not dry_run:
        out_dir = EVIDENCE_ROOT / "v3_exq_594_mech339_composite_cue_outshining_validation"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{run_id}.json"
        with open(out_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"Result written to: {out_path}", flush=True)

    return {
        "all_pass": bool(all_pass),
        "outcome": "PASS" if all_pass else "FAIL",
        "manifest_path": str(out_path) if out_path is not None else None,
        "run_id": run_id,
        "dry_run": bool(dry_run),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="T1 only (backward-compat check); no manifest write.")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    # Runner-conformance sentinel. Skipped on --dry-run (no manifest written;
    # a dry-run is not a real queued run). Crashes propagate as a non-zero
    # exit + missing sentinel -> the runner classifies them ERROR.
    if not result["dry_run"]:
        emit_outcome(
            outcome=result["outcome"],
            manifest_path=result["manifest_path"],
            run_id=result["run_id"],
        )
    sys.exit(0 if result["all_pass"] else 1)
