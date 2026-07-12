"""V3-EXQ-603j -- SD-059/MECH-358 escape-affordance-bridge SAFETY-HALF readiness microdiagnostic.

LINEAGE / ROUTING
-----------------
- Originating result : V3-EXQ-603i (route substrate_not_ready_requeue,
  evidence_direction=non_contributory, NO claim weakening). SECONDARY gap of
  failure_autopsy_V3-EXQ-603i_2026-06-08: the bridge SAFETY half credited 0/3 in
  every arm (relief half 2/3, functional) because its raw credit condition
  (threat_scale(z_now) <= 0, i.e. z_harm_a norm below threat_floor) almost never
  fires under Stage-H -- the threat never goes fully absent after a directed
  action. The half was wired STRUCTURALLY but STARVED of a trained threat-absence
  input.
- This session's fix (already landed + smoke-passed): the EscapeAffordanceBridge
  safety credit now ALSO fires when a trained MECH-303 (contextual safety terrain)
  / MECH-304 (conditioned safety store) threat-absence signal clears a threshold,
  fed by the agent at the bridge.update site (max over the enabled trained
  predictors). OR-composed with the raw check; inside the existing under-threat +
  directed-action gate -> genuine response-produced safety.
- THIS microdiagnostic validates ONLY that the safety half now credits
  non-vacuously when a trained predictor signals safe, DECOUPLED from G_H (same
  decoupling pattern as V3-EXQ-636/637/653). It is NOT the full 4-arm G_H
  behavioural bridge retest -- that stays gated on the PRIMARY nav/survival-
  competence ceiling (scaffolded_sd054_onboarding Stage-H leg / separate chip);
  a retest can only score G_H once nav competence clears too.

CLAIM HANDLING
--------------
claim_ids = []  (diagnostic / substrate-readiness microdiagnostic).
evidence_direction = non_contributory. Does NOT validate or weaken SD-059 /
MECH-358; does NOT validate or weaken MECH-302 / MECH-303 / MECH-304. A PASS is
NOT V3 closure and NOT a bridge verdict; a FAIL is NOT bridge falsification.

DESIGN (2 arms x 3 seeds [42,43,44]; real agent, controlled retained-threat probe)
----------------------------------------------------------------------------------
Each cell builds a REAL REEAgent with the escape-affordance bridge ON (both halves)
+ the MECH-304 conditioned safety store (+ MECH-303 contextual safety terrain) +
the SD-011 affective harm stream so z_harm_a is populated. A short warm-up under
sustained threat settles z_world; the MECH-304 prototype is then SEEDED to the
agent's own z_world (a CONTROLLED positive control -- a trained predictor that has
learned a safe context for this state), so predict(z_world) reliably clears the
safety threshold. Then N measured rounds drive a RETAINED-THREAT pattern (z_harm_a
drops a little each round -> relief fires -- but stays ABOVE threat_floor -> the RAW
safety check never fires), with a directed (non-noop) last_action_class feeding the
bridge. The agent computes _eab_safety_signal = max(MECH-304 predict, MECH-303
evaluate_safety) and feeds it to bridge.update.

  ARM_OFF (escape_use_trained_safety_signal=False): reproduces the 603i
           starvation -- the safety half has no trained input, raw never fires
           under retained threat, so it credits ~0.
  ARM_ON  (escape_use_trained_safety_signal=True): the trained MECH-303/304 signal
           feeds the safety half, so it credits via the trained path.
The ONLY difference between arms is escape_use_trained_safety_signal.

READINESS GATES (substrate readiness only; thresholds pre-registered)
---------------------------------------------------------------------
  G0 OFF reproduces starvation: ARM_OFF safety credit == 0          (3/3 seeds)
  G1 ON safety half credits via the trained signal                 (>=2/3) [load-bearing]
     (mech358_n_safety_credit_trained > 0)
  G2 ON safety affordance table is non-zero                        (>=2/3)
  G3 relief half UNAFFECTED (only the safety path changed)         (>=2/3 both arms)
     (mech358_n_relief_credit > 0 in BOTH arms)

PASS = G0 AND G1 AND G2 AND G3. PASS label safety_half_trained_signal_ready.

DIAGNOSTIC ADJUDICATION (skill Step 3.5)
----------------------------------------
Verdict-class diagnostic reading a measured quantity (the trained safety credit
count). Readiness precondition keyed on the SAME gate the load-bearing G1 routes
on: the trained safety_signal the agent FEEDS must clear safety_signal_threshold on
the positive control (and the under-threat gate must open). Below floor (the
positive control never presented a clear trained-safe signal -- a probe-driver /
predictor-population regression, NOT a bridge verdict) self-routes
substrate_not_ready_requeue, NEVER a substrate verdict. Non-degeneracy guards: the
ON-arm credits must be attributable to the TRAINED path (raw did not carry them),
relief must actually fire (so the G3 comparison is meaningful), and the under-threat
gate must have opened in the OFF arm too (so G0's zero is a genuine starvation, not
a no-threat artifact).

Usage:
  /opt/local/bin/python3 experiments/v3_exq_603j_escape_bridge_safety_half_readiness.py --dry-run
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_603j_escape_bridge_safety_half_readiness"
QUEUE_ID = "V3-EXQ-603j"
CLAIM_IDS: List[str] = []  # claim-free substrate-readiness microdiagnostic
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44]
N_TRAIN_ROUNDS = 40          # measured rounds per (arm x seed); the [train] ep N/M denominator
WARMUP_TICKS = 8
DRY_RUN_SEEDS = [42]
DRY_RUN_TRAIN_ROUNDS = 6
DRY_RUN_WARMUP = 4

# Env / obs dims (standard small CausalGridWorld-style harm-stream config).
WORLD_OBS_DIM = 250
BODY_OBS_DIM = 12
HARM_OBS_DIM = 50
HARM_OBS_A_DIM = 50
ACTION_DIM = 5

DIRECTED_CLASS = 2           # a non-noop (directed) first-action class
NOOP_CLASS = 0

# Controlled retained-threat harm_obs_a pattern: high then lower-but-still-threat.
# tick1 high -> tick2 lower (z_harm_a drops a little: relief fires) but the lower
# level keeps z_harm_a above threat_floor (the RAW safety check stays unmet, so the
# ON-arm credit must come from the TRAINED path).
HARM_A_HIGH = 0.85
HARM_A_LOW = 0.45

# Pre-registered thresholds.
SAFETY_THRESHOLD = 0.5       # escape_safety_signal_threshold (trained signal gate)
ESCAPE_THREAT_FLOOR = 0.1    # escape_threat_floor (under-threat gate)
MIN_SEEDS_2OF3 = 2

ARMS: List[Dict[str, Any]] = [
    {"arm_id": "ARM_OFF", "trained_safety": False},
    {"arm_id": "ARM_ON", "trained_safety": True},
]

_CUR_ROUNDS = N_TRAIN_ROUNDS
_CUR_WARMUP = WARMUP_TICKS


def _build_agent(trained_safety: bool) -> REEAgent:
    # No manual_seed here: arm_cell.__enter__ already ran reset_all_rng(seed), so
    # per-seed weight init is distinct.
    cfg = REEConfig.from_dims(
        world_obs_dim=WORLD_OBS_DIM, body_obs_dim=BODY_OBS_DIM,
        harm_obs_dim=HARM_OBS_DIM, harm_obs_a_dim=HARM_OBS_A_DIM,
        action_dim=ACTION_DIM,
        use_escape_affordance_bridge=True,
        use_escape_relief_credit=True,
        use_escape_safety_credit=True,
        escape_threat_floor=ESCAPE_THREAT_FLOOR,
        escape_threat_ref=0.5,
        escape_use_trained_safety_signal=trained_safety,
        escape_safety_signal_threshold=SAFETY_THRESHOLD,
        # Trained threat-absence predictors (the wiring under test).
        use_conditioned_safety_store=True,        # MECH-304
        use_contextual_safety_terrain=True,       # MECH-303
    )
    # SD-011 affective harm stream so z_harm_a is populated (the bridge keys on it).
    cfg.latent.use_affective_harm_stream = True
    return REEAgent(cfg)


def _run_cell(arm: Dict[str, Any], seed: int, n_rounds: int, warmup: int) -> Dict[str, Any]:
    arm_id = arm["arm_id"]
    ag = _build_agent(arm["trained_safety"])
    bridge = ag.escape_affordance_bridge

    # Capture the safety_signal the AGENT feeds the bridge (the quantity G1 routes
    # on); a thin spy over the instance method (does not change behaviour).
    fed_signals: List[Optional[float]] = []
    _real_update = bridge.update

    def _spy(*a: Any, **k: Any) -> None:
        fed_signals.append(k.get("safety_signal"))
        return _real_update(*a, **k)

    bridge.update = _spy  # type: ignore[assignment]

    body = torch.zeros(1, BODY_OBS_DIM)
    world = torch.ones(1, WORLD_OBS_DIM)
    harm = torch.zeros(1, HARM_OBS_DIM)
    harm_a_high = torch.ones(1, HARM_OBS_A_DIM) * HARM_A_HIGH
    harm_a_low = torch.ones(1, HARM_OBS_A_DIM) * HARM_A_LOW

    # Warm-up under sustained threat to settle z_world; capture it.
    last_lat = None
    for _ in range(max(1, warmup)):
        last_lat = ag.sense(body, world, obs_harm=harm, obs_harm_a=harm_a_high)
        ag._eab_last_action_class = DIRECTED_CLASS

    # Seed the MECH-304 prototype to the agent's own (normalised) z_world -- a
    # CONTROLLED positive control: a trained predictor that has learned a safe
    # context for this state, so predict(z_world) reliably clears the threshold.
    zw = last_lat.z_world.detach().squeeze(0)
    zw_norm = float(zw.norm().item()) + 1e-8
    proto = (zw / zw_norm).tolist()
    store = ag.conditioned_safety_store
    if store is not None:
        store._prototype = list(proto)

    # Sanity: the positive-control trained signal the agent will read.
    pc_predict = float(store.predict(world.squeeze(0))) if store is not None else 0.0

    # Measured rounds: retained-threat pattern (high -> lower-but-still-threat).
    z_seen: List[float] = []
    log_every = max(1, n_rounds // 6)
    for r in range(n_rounds):
        ag.sense(body, world, obs_harm=harm, obs_harm_a=harm_a_high)
        ag._eab_last_action_class = DIRECTED_CLASS
        ag.sense(body, world, obs_harm=harm, obs_harm_a=harm_a_low)
        ag._eab_last_action_class = DIRECTED_CLASS
        zp = bridge._z_harm_a_prev
        z_seen.append(float(zp) if zp is not None else 0.0)
        if (r + 1) % log_every == 0 or (r + 1) == n_rounds:
            print(f"  [train] arm={arm_id} seed={seed} ep {r + 1}/{n_rounds}", flush=True)

    st = bridge.get_state()
    fed_on = [float(f) for f in fed_signals if f is not None]
    mean_fed = sum(fed_on) / len(fed_on) if fed_on else 0.0
    mean_z_seen = sum(z_seen) / len(z_seen) if z_seen else 0.0

    n_safety = int(st["mech358_n_safety_credit"])
    n_safety_trained = int(st["mech358_n_safety_credit_trained"])
    n_safety_raw = n_safety - n_safety_trained
    n_relief = int(st["mech358_n_relief_credit"])
    safety_aff_max = float(st["mech358_safety_affordance_max"])

    return {
        "arm_id": arm_id,
        "seed": int(seed),
        "trained_safety": bool(arm["trained_safety"]),
        "n_safety_credit": n_safety,
        "n_safety_credit_trained": n_safety_trained,
        "n_safety_credit_raw": n_safety_raw,
        "n_relief_credit": n_relief,
        "safety_affordance_max": round(safety_aff_max, 6),
        "mean_fed_safety_signal": round(mean_fed, 6),
        "positive_control_predict": round(pc_predict, 6),
        "mean_z_harm_a_seen": round(mean_z_seen, 6),
        "diag": st,
    }


# ---------------------------------------------------------------------------
# Evaluation / interpretation
# ---------------------------------------------------------------------------

def _seeds(rows: List[Dict[str, Any]], pred) -> int:
    return sum(1 for r in rows if pred(r))


def _evaluate(arm_results: List[Dict[str, Any]], n_seeds: int) -> Dict[str, Any]:
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for r in arm_results:
        by_arm.setdefault(r["arm_id"], []).append(r)
    off = by_arm.get("ARM_OFF", [])
    on = by_arm.get("ARM_ON", [])

    need2 = MIN_SEEDS_2OF3 if n_seeds >= 3 else 1
    need3 = n_seeds

    # G0: OFF reproduces starvation (safety half credits 0 -- raw never fires under
    # retained threat; the trained path is off).
    g0 = _seeds(off, lambda r: r["n_safety_credit"] == 0)
    # G1 (load-bearing): ON safety half credits via the trained signal.
    g1 = _seeds(on, lambda r: r["n_safety_credit_trained"] > 0)
    # G2: ON safety affordance table is non-zero.
    g2 = _seeds(on, lambda r: r["safety_affordance_max"] > 0.0)
    # G3: relief half unaffected -- fires in BOTH arms.
    g3_off = _seeds(off, lambda r: r["n_relief_credit"] > 0)
    g3_on = _seeds(on, lambda r: r["n_relief_credit"] > 0)

    gate_pass = {
        "G0_off_reproduces_starvation": g0 >= need3,
        "G1_on_safety_credits_via_trained_signal": g1 >= need2,
        "G2_on_safety_affordance_nonzero": g2 >= need2,
        "G3_relief_unaffected_both_arms": (g3_off >= need2 and g3_on >= need2),
    }

    # Readiness: the load-bearing G1 routes on n_safety_credit_trained, which
    # increments only when the agent-fed trained safety_signal clears the threshold
    # AND the under-threat gate opened. Match the readiness measure to that gate:
    # the mean fed safety signal in ARM_ON must clear SAFETY_THRESHOLD, and the
    # under-threat gate must have opened. Below floor => the positive control never
    # presented a clear trained-safe signal (probe-driver / predictor-population
    # regression) => substrate_not_ready_requeue, NOT a bridge verdict.
    mean_fed_on = (
        sum(r["mean_fed_safety_signal"] for r in on) / len(on) if on else 0.0
    )
    mean_z_on = (
        sum(r["mean_z_harm_a_seen"] for r in on) / len(on) if on else 0.0
    )
    mean_z_off = (
        sum(r["mean_z_harm_a_seen"] for r in off) / len(off) if off else 0.0
    )
    signal_clears = bool(mean_fed_on >= SAFETY_THRESHOLD)
    threat_gate_open_on = bool(mean_z_on > ESCAPE_THREAT_FLOOR)
    threat_gate_open_off = bool(mean_z_off > ESCAPE_THREAT_FLOOR)
    readiness_ok = bool(signal_clears and threat_gate_open_on)

    all_primary = all(gate_pass.values())
    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        overall_pass = False
    elif all_primary:
        label = "safety_half_trained_signal_ready"
        overall_pass = True
    else:
        label = "safety_half_trained_signal_inert"
        overall_pass = False

    # Non-degeneracy per gate.
    # G1: the ON credits must be attributable to the TRAINED path (raw didn't carry
    # them) -- else the "trained" gate is vacuous.
    on_trained_carries = (
        all((r["n_safety_credit_trained"] >= r["n_safety_credit_raw"]) for r in on)
        if on else False
    )
    criteria_non_degenerate = {
        # G0 is a genuine starvation only if the OFF under-threat gate actually
        # opened (the credit path had the opportunity and still credited 0).
        "G0": bool(off) and threat_gate_open_off,
        "G1": bool(on) and on_trained_carries,
        "G2": bool(on),
        # G3 is meaningful only if relief actually fired somewhere.
        "G3": (g3_off > 0 or g3_on > 0),
    }

    return {
        "label": label,
        "overall_pass": overall_pass,
        "readiness_ok": readiness_ok,
        "gate_seeds_passing": {
            "G0": g0, "G1": g1, "G2": g2,
            "G3_off": g3_off, "G3_on": g3_on,
        },
        "gate_pass": gate_pass,
        "n_seeds": n_seeds,
        "min_seeds_2of3": need2,
        "min_seeds_3of3": need3,
        "preconditions": [
            {
                "name": "trained_safety_signal_clears_threshold",
                "kind": "readiness",
                "description": (
                    "Mean trained safety_signal the AGENT feeds the bridge in "
                    "ARM_ON -- the SAME gate the load-bearing G1 "
                    "(n_safety_credit_trained) routes on (safety_signal >= "
                    "safety_signal_threshold). Below floor => the positive control "
                    "never presented a clear trained-safe signal (probe-driver / "
                    "predictor-population regression) => substrate_not_ready_requeue, "
                    "NOT a 'safety half cannot credit' verdict."
                ),
                "control": "ARM_ON: MECH-304 prototype seeded to the agent's own z_world",
                "measured": round(float(mean_fed_on), 6),
                "threshold": SAFETY_THRESHOLD,
                "met": signal_clears,
            },
            {
                "name": "under_threat_gate_open_on",
                "kind": "readiness",
                "description": (
                    "Mean z_harm_a norm seen by the bridge in ARM_ON must exceed "
                    "escape_threat_floor -- the under-threat gate that the safety "
                    "credit path requires. Below floor => no retained threat (the "
                    "whole credit path is starved for an unrelated reason) => "
                    "substrate_not_ready_requeue."
                ),
                "control": "ARM_ON: sustained HARM_A_HIGH/LOW retained-threat pattern",
                "measured": round(float(mean_z_on), 6),
                "threshold": ESCAPE_THREAT_FLOOR,
                "met": threat_gate_open_on,
            },
        ],
        "criteria": [
            {"name": "G0_off_reproduces_starvation", "load_bearing": False,
             "passed": gate_pass["G0_off_reproduces_starvation"]},
            {"name": "G1_on_safety_credits_via_trained_signal", "load_bearing": True,
             "passed": gate_pass["G1_on_safety_credits_via_trained_signal"]},
            {"name": "G2_on_safety_affordance_nonzero", "load_bearing": False,
             "passed": gate_pass["G2_on_safety_affordance_nonzero"]},
            {"name": "G3_relief_unaffected_both_arms", "load_bearing": False,
             "passed": gate_pass["G3_relief_unaffected_both_arms"]},
        ],
        "criteria_non_degenerate": criteria_non_degenerate,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    global _CUR_ROUNDS, _CUR_WARMUP
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    _CUR_ROUNDS = DRY_RUN_TRAIN_ROUNDS if dry_run else N_TRAIN_ROUNDS
    _CUR_WARMUP = DRY_RUN_WARMUP if dry_run else WARMUP_TICKS

    arm_results: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm['arm_id']}", flush=True)
            with arm_cell(
                seed,
                config_slice={
                    "arm": arm,
                    "world_obs_dim": WORLD_OBS_DIM,
                    "harm_obs_a_dim": HARM_OBS_A_DIM,
                    "action_dim": ACTION_DIM,
                    "directed_class": DIRECTED_CLASS,
                    "harm_a_high": HARM_A_HIGH,
                    "harm_a_low": HARM_A_LOW,
                    "safety_threshold": SAFETY_THRESHOLD,
                    "escape_threat_floor": ESCAPE_THREAT_FLOOR,
                    "n_train_rounds": _CUR_ROUNDS,
                    "warmup": _CUR_WARMUP,
                },
                script_path=Path(__file__),
                extra_ineligible_reasons=["microdiagnostic_per_cell_agent_build"],
            ) as cell:
                row = _run_cell(arm, seed, _CUR_ROUNDS, _CUR_WARMUP)
                cell.stamp(row)
            arm_results.append(row)
            _v = "PASS" if (
                (arm["arm_id"] == "ARM_ON" and row["n_safety_credit_trained"] > 0)
                or (arm["arm_id"] == "ARM_OFF" and row["n_safety_credit"] == 0)
            ) else "FAIL"
            print(f"verdict: {_v}", flush=True)

    summary = _evaluate(arm_results, n_seeds=len(seeds))
    outcome = "PASS" if summary["overall_pass"] else "FAIL"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    manifest: Dict[str, Any] = {
        "schema_version": "v1",
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": outcome,
        "result": outcome,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "non_contributory",
        "evidence_direction_note": (
            "SD-059/MECH-358 escape-affordance-bridge SAFETY-HALF readiness "
            "microdiagnostic for the V3-EXQ-603i SECONDARY gap (safety half "
            "credited 0/3 because the raw threat_scale<=0 check almost never fires "
            "under Stage-H). 2-arm OFF/ON probe over a REAL agent under controlled "
            "retained threat with a seeded MECH-304 positive control. claim_ids=[] "
            "(does NOT weight claim confidence). A PASS "
            "(label=safety_half_trained_signal_ready) means ONLY that the bridge "
            "safety half credits non-vacuously when fed a trained MECH-303/304 "
            "threat-absence signal -- it does NOT validate SD-059/MECH-358 or "
            "MECH-302/303/304, is NOT the full 4-arm G_H behavioural bridge retest "
            "(that stays gated on the PRIMARY nav/survival-competence ceiling), is "
            "NOT ecological survival, and is NOT V3 closure. A FAIL is NOT bridge "
            "falsification. Readiness-below-floor self-routes "
            "substrate_not_ready_requeue (positive control did not present a clear "
            "trained-safe signal / no retained threat), NOT a substrate verdict. "
            "SD-059/MECH-358 (+ MECH-302/303/304) stay unchanged "
            "(candidate / v3_pending / pending_retest_after_substrate)."
        ),
        "interpretation": {
            "label": summary["label"],
            "preconditions": summary["preconditions"],
            "criteria": summary["criteria"],
            "criteria_non_degenerate": summary["criteria_non_degenerate"],
            "routing": {
                "safety_half_trained_signal_ready": (
                    "PASS -> the SECONDARY 603i gap is closed at the substrate-"
                    "readiness level. Route to the full 603-lineage behavioural "
                    "bridge retest -- which is ALSO gated on the PRIMARY nav-"
                    "competence ceiling (scaffolded_sd054_onboarding Stage-H leg), "
                    "so it can only score G_H once nav competence clears too. Do "
                    "NOT queue the behavioural retest here; do NOT mark "
                    "SD-059/MECH-358 validated."
                ),
                "substrate_not_ready_requeue": (
                    "re-queue as V3-EXQ-603k (fix the probe's predictor-population / "
                    "retained-threat driver -- the positive control did not present "
                    "a clear trained-safe signal, or no retained threat); do NOT "
                    "weaken SD-059/MECH-358."
                ),
                "safety_half_trained_signal_inert": (
                    "FAIL with readiness met -> /failure-autopsy on THIS readiness "
                    "diagnostic: the agent fed a clear trained signal + the threat "
                    "gate opened but the safety half did not credit -> a wiring "
                    "regression in the bridge/agent safety-credit path. Do NOT "
                    "rerun 603i; do NOT treat as bridge falsification."
                ),
            },
        },
        "dry_run": bool(dry_run),
        "config": {
            "seeds": seeds,
            "n_train_rounds": _CUR_ROUNDS,
            "warmup_ticks": _CUR_WARMUP,
            "arms": [a["arm_id"] for a in ARMS],
            "directed_class": DIRECTED_CLASS,
            "harm_a_high": HARM_A_HIGH,
            "harm_a_low": HARM_A_LOW,
            "obs_dims": {
                "world_obs_dim": WORLD_OBS_DIM, "body_obs_dim": BODY_OBS_DIM,
                "harm_obs_dim": HARM_OBS_DIM, "harm_obs_a_dim": HARM_OBS_A_DIM,
                "action_dim": ACTION_DIM,
            },
            "thresholds": {
                "safety_signal_threshold": SAFETY_THRESHOLD,
                "escape_threat_floor": ESCAPE_THREAT_FLOOR,
                "min_seeds_2of3": MIN_SEEDS_2OF3,
            },
        },
        "acceptance_criteria": {
            "readiness_ok": summary["readiness_ok"],
            **summary["gate_pass"],
            "overall_pass": summary["overall_pass"],
        },
        "summary": summary,
        "arm_results": arm_results,
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = out_dir / f"{run_id}.json"
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = write_flat_manifest(
            manifest,
            out_dir,
            dry_run=False,
            config=manifest.get("config"),
            seeds=SEEDS,
            script_path=Path(__file__),
        )
        print(f"Manifest written: {out_path}", flush=True)
    else:
        out_path = Path("/dev/null")
        print("Dry run -- manifest not written.", flush=True)

    print(f"Outcome: {outcome} (label={summary['label']})", flush=True)
    for k, v in summary["gate_seeds_passing"].items():
        print(f"  {k} seeds passing: {v}", flush=True)

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-603j escape-affordance-bridge safety-half readiness microdiagnostic"
    )
    parser.add_argument("--dry-run", action="store_true", help="Short smoke run.")
    args = parser.parse_args()

    result = run_experiment(dry_run=args.dry_run)

    if args.dry_run:
        sys.exit(0)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    _outcome = _outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL"
    emit_outcome(
        outcome=_outcome,
        manifest_path=str(result.get("manifest_path", Path("/dev/null"))),
    )
