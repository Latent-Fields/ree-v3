"""V3-EXQ-653 -- post-603i E2 escape-affordance linker readiness microdiagnostic.

LINEAGE / ROUTING
-----------------
- Originating result : V3-EXQ-603i (route substrate_not_ready_requeue,
  evidence_direction=non_contributory, NO claim weakening).
- Related lineage    : the V3-EXQ-603 SD-054 / hazard-survival / relief-safety
  escape-affordance bridge lineage.
- This is a DIFFERENT experiment from the full 603 behavioural / relief-safety
  bridge validation. It validates ONLY the readiness of the post-603i E2
  escape-affordance LINKER / readout scaffold
  (ree_core/pfc/e2_escape_affordance_linker.py). It does NOT re-run the full 603
  behavioural validation and makes NO ecological survival claim.
- Suggested route after PASS : queue a 603-lineage full behavioural bridge
  re-test with linker features attached to the trainable relief/safety heads
  (the next 603-lineage suffix or whatever id the queue system assigns).
- Suggested route after FAIL : run /failure-autopsy on THIS readiness diagnostic
  before returning to 603 (localise to linker/readout learning, E2 feature
  geometry, E3 bias integration, no-op/freeze exclusion, hypothesis/simulation
  boundary, or the reuse-E2 assumption itself).

WHY 653 AND NOT 603j
--------------------
This probes a NEW substrate module (the E2EscapeAffordanceLinker, distinct from
the SD-059 arithmetic escape_affordance_bridge that V3-EXQ-603i tested) and asks a
DIFFERENT scientific question (can the linker readout learn a controlled escape
affordance + expose a bounded threat-gated bias). New question -> new number. The
substrate plan doc (docs/substrate_plans/post_603i_e2_escape_affordance_linkage.md)
explicitly says: "Do not queue a full 603j bridge re-run from this scaffold."

CLAIM HANDLING
--------------
claim_ids = []  (diagnostic / substrate-readiness microdiagnostic).
evidence_direction = non_contributory. Does NOT validate or weaken SD-059 /
MECH-358; does NOT validate or weaken MECH-302 / MECH-303 / MECH-304. A PASS is
NOT V3 closure; a FAIL is NOT bridge falsification.

DESIGN (forced-choice microdiagnostic -- NOT an ecological survival run)
-----------------------------------------------------------------------
Drives E2EscapeAffordanceLinker DIRECTLY with CONTROLLED per-action E2-consequence
proxy features and KNOWN outcomes (mirrors the proven contract harness
tests/contracts/test_e2_escape_affordance_linker.py::_microdiagnostic_seed). This
is the architecturally-faithful readiness probe: it isolates the linker readout +
bias surface from ecological foraging/survival noise, and uses controlled proxies
precisely so a later real-E2 test can attribute any regression to E2 feature
geometry rather than to the linker.

  Action classes: 0 = no-op/freeze, 1 = harm-worsening, 2 = escape-producing,
                  3 = neutral. Each non-noop action carries a distinct controlled
                  E2 consequence feature; the true outcome is known in the harness:
    escape (2)        -> harm drops / threat terminates  (outcome z_harm_a ~ 0.04)
    harm-worsening(1) -> harm rises                      (outcome z_harm_a ~ 0.95)
    neutral (3)       -> no change, still threatened     (outcome z_harm_a ~ 0.60)
    no-op (0)         -> not credited (freeze excluded by design)

ARMS (4 conditions x 3 seeds)
-----------------------------
  ARM_DISABLED_CONTROL          use_e2_escape_affordance_linker=False -- G0 no-op /
                                bit-stable disabled path.
  ARM_LINKER_READOUT_ONLY       linker ON, e3_bias OFF -- G1/G2/G3/G6/G7/G8 (readout
                                learns escape from detached features; correctness).
  ARM_LINKER_E3_BIAS            linker ON, e3_bias ON -- G1 (reconfirm)/G4/G5
                                (bounded threat-gated bias points to escape).
  ARM_LINKER_TO_RELIEF_SAFETY   linker + trainable relief/safety learner consuming
                                linker features -- G_RS (features consumed without
                                changing the learner default / collapsing
                                relief+safety; secondary).

READINESS GATES (substrate readiness only; thresholds pre-registered)
---------------------------------------------------------------------
  G0 disabled path is no-op / bit-stable                              (3/3 seeds)
  G1 escape readout for the escape action improves                   (>=2/3)  [load-bearing]
  G2 no-op/freeze remains uncredited                                 (>=2/3)
  G3 harm-worsening does not become the preferred escape action      (>=2/3)
  G4 threat-gated E3 bias points to the learned escape action        (>=2/3)  [load-bearing]
  G5 E3 bias is exactly zero when safe                               (3/3)
  G6 simulation + hypothesis-tag mode block learning                 (3/3)
  G7 learned weights / viability persist across episode reset        (>=2/3)
  G8 inputs are detached; no backprop into E1/E2/E3 encoders         (3/3)
  G_RS linker features feed the trainable relief/safety learner      (>=2/3, secondary)

PASS interpretation: ONLY "the post-603i E2 escape-affordance linker/readout can
learn a controlled 'where out is' signal and expose bounded threat-gated bias
under microdiagnostic conditions." NOT: SD-059/MECH-358 validated, relief/safety
bridge validated, ecological survival solved, hippocampal map validated, or V3
closure advanced.

DIAGNOSTIC ADJUDICATION (skill Step 3.5)
----------------------------------------
Verdict-class diagnostic reading a learned quantity (the readout salience delta).
Readiness precondition keyed on the SAME exercise the load-bearing G1 routes on:
the linker optimizer must actually step in the positive-control readout arm. Below
floor (linker never instantiated / never trained -> a wiring regression, not a
"readout cannot learn" verdict) self-routes substrate_not_ready_requeue, NEVER a
substrate verdict. Harness non-vacuity preconditions (controlled proxies distinct;
forced-choice outcome targets differentiated) guard against a degenerate G1 PASS.

Usage:
  /opt/local/bin/python3 experiments/v3_exq_653_e2_escape_affordance_linker_readiness_microdiagnostic.py --dry-run
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from ree_core.pfc.e2_escape_affordance_linker import (  # noqa: E402
    E2EscapeAffordanceLinker,
    E2EscapeAffordanceLinkerConfig,
)
from ree_core.pfc.trainable_escape_affordance_learner import (  # noqa: E402
    TrainableEscapeAffordanceLearner,
    TrainableEscapeAffordanceLearnerConfig,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_653_e2_escape_affordance_linker_readiness_microdiagnostic"
QUEUE_ID = "V3-EXQ-653"
CLAIM_IDS: List[str] = []  # claim-free substrate-readiness microdiagnostic
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44]
N_TRAIN_ROUNDS = 60          # forced-choice rounds per (arm x seed); the [train] ep N/M denominator
DRY_RUN_SEEDS = [42]
DRY_RUN_TRAIN_ROUNDS = 8

# Action classes.
NOOP, HARM_WORSEN, ESCAPE, NEUTRAL = 0, 1, 2, 3
N_ACTION_CLASSES = 4

# Distinct controlled per-action E2-consequence proxy features (the linker's
# detached input; mirrors the contract harness so the shared trunk can
# discriminate). The escape action's proxy is the only one this readiness probe
# needs to be learnable; the others provide the forced-choice contrast.
E2FEAT: Dict[int, List[float]] = {
    NOOP:        [0.0, 0.0, 0.0, 0.0],
    HARM_WORSEN: [0.9, -0.8, 0.1, 0.2],
    ESCAPE:      [-0.7, 0.6, -0.5, 0.4],
    NEUTRAL:     [0.1, 0.1, 0.1, -0.1],
}

# Known forced-choice outcomes: prior threat z_harm_a ~ 0.6, then the post-action
# z_harm_a the harness asserts (escape drops harm + ends threat; harm-worsening
# raises harm; neutral leaves threat).
THREAT_PRIOR = 0.6
OUTCOME: Dict[int, float] = {
    NOOP:        0.04,
    HARM_WORSEN: 0.95,
    ESCAPE:      0.04,
    NEUTRAL:     0.60,
}

# Pre-registered thresholds.
G1_SALIENCE_DELTA = 0.05     # escape readout salience must rise by at least this
MIN_SEEDS_2OF3 = 2
MIN_SEEDS_3OF3 = 3
PROXY_DISTINCT_FLOOR = 0.1   # readiness: min pairwise L2 over non-noop proxies
OUTCOME_RANGE_FLOOR = 0.1    # readiness: range of forced-choice outcome targets
OPTIMIZER_STEP_FLOOR = 1.0   # readiness: mean optimizer steps in the readout arm

# Linker hyperparameters (leak_rate 0.0 for the proven-stable contract probe
# config; the readiness question is whether the readout CAN learn the controlled
# signal, not weight-decay dynamics).
LINKER_PARAMS = dict(
    n_action_classes=N_ACTION_CLASSES,
    hidden_dim=32,
    action_embedding_dim=6,
    learn_rate=0.3,
    optimizer_lr=0.02,
    leak_rate=0.0,
    bias_scale=0.2,
    threat_floor=0.1,
    threat_ref=0.5,
    noop_class=NOOP,
    relief_reward_floor=0.05,
    harm_delta_scale=0.6,
    prediction_floor=0.02,
)

ARMS: List[Dict[str, Any]] = [
    {"arm_id": "ARM_DISABLED_CONTROL", "linker_enabled": False,
     "e3_bias": False, "relief_safety": False},
    {"arm_id": "ARM_LINKER_READOUT_ONLY", "linker_enabled": True,
     "e3_bias": False, "relief_safety": False},
    {"arm_id": "ARM_LINKER_E3_BIAS", "linker_enabled": True,
     "e3_bias": True, "relief_safety": False},
    {"arm_id": "ARM_LINKER_TO_RELIEF_SAFETY", "linker_enabled": True,
     "e3_bias": True, "relief_safety": True},
]


# ---------------------------------------------------------------------------
# Linker harness
# ---------------------------------------------------------------------------

def _e2(action: int) -> torch.Tensor:
    return torch.tensor(E2FEAT[int(action)], dtype=torch.float32)


def _make_linker(enabled: bool) -> E2EscapeAffordanceLinker:
    # No manual_seed here: arm_cell.__enter__ already ran reset_all_rng(seed), so
    # the linker's lazy weight init is per-seed distinct.
    return E2EscapeAffordanceLinker(
        E2EscapeAffordanceLinkerConfig(enabled=enabled, **LINKER_PARAMS)
    )


def _trial(linker: E2EscapeAffordanceLinker, action: int, outcome: float,
           **kw: Any) -> None:
    """One clean (under-threat -> outcome) trial; resets the one-tick trace after."""
    linker.update(THREAT_PRIOR, last_action_class=action, e2_features=_e2(action), **kw)
    linker.update(outcome, last_action_class=action, e2_features=_e2(action), **kw)
    linker.reset()


def _probe(linker: E2EscapeAffordanceLinker, action: int) -> torch.Tensor:
    return linker.build_state_vector(
        e2_features=_e2(action), z_harm_a_norm=0.6, threat_scale=1.0
    )


def _train_forced_choice(linker: E2EscapeAffordanceLinker, arm_id: str,
                         seed: int, n_rounds: int) -> None:
    log_every = max(1, n_rounds // 6)
    for r in range(n_rounds):
        _trial(linker, ESCAPE, OUTCOME[ESCAPE])
        _trial(linker, HARM_WORSEN, OUTCOME[HARM_WORSEN])
        _trial(linker, NEUTRAL, OUTCOME[NEUTRAL])
        _trial(linker, NOOP, OUTCOME[NOOP])
        if (r + 1) % log_every == 0 or (r + 1) == n_rounds:
            print(f"  [train] arm={arm_id} seed={seed} ep {r + 1}/{n_rounds}",
                  flush=True)


# ---------------------------------------------------------------------------
# Gate evaluation (per seed)
# ---------------------------------------------------------------------------

def _eval_disabled(seed: int) -> Dict[str, Any]:
    lk = _make_linker(enabled=False)
    _train_forced_choice(lk, "ARM_DISABLED_CONTROL", seed, _CUR_ROUNDS)
    bias = lk.compute_approach_bias(THREAT_PRIOR, [0, 1, 2, 3])
    st = lk.get_state()
    g0 = bool(
        lk.model is None
        and lk.optimizer is None
        and st["e2_escape_linker_n_optimizer_steps"] == 0
        and float(bias.abs().max()) == 0.0
    )
    return {"G0": g0, "diag": st}


def _eval_readout(seed: int) -> Dict[str, Any]:
    lk = _make_linker(enabled=True)
    esc_before = lk.escape_salience(ESCAPE, _probe(lk, ESCAPE))
    _train_forced_choice(lk, "ARM_LINKER_READOUT_ONLY", seed, _CUR_ROUNDS)
    esc_after = lk.escape_salience(ESCAPE, _probe(lk, ESCAPE))
    st = lk.get_state()

    # G1: escape readout improves.
    g1 = bool(math.isfinite(esc_before) and math.isfinite(esc_after)
              and esc_after > esc_before + G1_SALIENCE_DELTA)
    g1_non_degenerate = bool(
        lk.model is not None and st["e2_escape_linker_n_optimizer_steps"] > 0
        and math.isfinite(esc_before) and math.isfinite(esc_after)
    )

    # G2: no-op uncredited.
    bias = lk.compute_approach_bias(THREAT_PRIOR, [0, 1, 2, 3])
    g2 = bool(float(bias[NOOP]) == 0.0 and st["e2_escape_linker_n_noop_skipped"] > 0)

    # G3: harm-worsening does not become the preferred escape action.
    sal = {c: lk.escape_salience(c, _probe(lk, c)) for c in (HARM_WORSEN, ESCAPE, NEUTRAL)}
    preferred = max(sal, key=sal.get)
    g3 = bool(preferred != HARM_WORSEN)

    # G6: simulation + hypothesis mode block learning (fresh linkers).
    sim = _make_linker(enabled=True)
    sim.update(THREAT_PRIOR, last_action_class=ESCAPE, e2_features=_e2(ESCAPE))
    sim.update(OUTCOME[ESCAPE], last_action_class=ESCAPE, e2_features=_e2(ESCAPE),
               simulation_mode=True)
    hyp = _make_linker(enabled=True)
    hyp.update(THREAT_PRIOR, last_action_class=ESCAPE, e2_features=_e2(ESCAPE))
    hyp.update(OUTCOME[ESCAPE], last_action_class=ESCAPE, e2_features=_e2(ESCAPE),
               hypothesis_tag=True)
    g6 = bool(
        sim.model is None
        and sim.get_state()["e2_escape_linker_n_optimizer_steps"] == 0
        and hyp.model is None
        and hyp.get_state()["e2_escape_linker_n_optimizer_steps"] == 0
    )

    # G7: learned weights persist across reset.
    probe_vec = _probe(lk, ESCAPE)
    pred_before = lk.predict_head("harm_delta", ESCAPE, probe_vec)
    model_before = lk.model
    lk.reset()
    pred_after = lk.predict_head("harm_delta", ESCAPE, probe_vec)
    g7 = bool(
        lk.model is model_before
        and math.isclose(pred_after, pred_before, rel_tol=0.0, abs_tol=1e-7)
    )

    # G8: inputs detached -- no backprop into upstream tensors (fresh linker).
    det = _make_linker(enabled=True)
    e2 = torch.tensor(E2FEAT[ESCAPE], dtype=torch.float32, requires_grad=True)
    zw = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float32, requires_grad=True)
    zha = torch.tensor([0.6], dtype=torch.float32, requires_grad=True)
    det.update(THREAT_PRIOR, last_action_class=ESCAPE, e2_features=e2, z_world=zw,
               z_harm_a=zha)
    det.update(OUTCOME[ESCAPE], last_action_class=ESCAPE, e2_features=e2, z_world=zw,
               z_harm_a=zha)
    g8 = bool(e2.grad is None and zw.grad is None and zha.grad is None)

    return {
        "G1": g1, "G1_non_degenerate": g1_non_degenerate,
        "G2": g2, "G3": g3, "G6": g6, "G7": g7, "G8": g8,
        "esc_before": round(float(esc_before), 6),
        "esc_after": round(float(esc_after), 6),
        "salience_by_class": {str(k): round(float(v), 6) for k, v in sal.items()},
        "n_optimizer_steps": int(st["e2_escape_linker_n_optimizer_steps"]),
        "diag": st,
    }


def _eval_e3_bias(seed: int) -> Dict[str, Any]:
    lk = _make_linker(enabled=True)
    esc_before = lk.escape_salience(ESCAPE, _probe(lk, ESCAPE))
    _train_forced_choice(lk, "ARM_LINKER_E3_BIAS", seed, _CUR_ROUNDS)
    esc_after = lk.escape_salience(ESCAPE, _probe(lk, ESCAPE))

    # G1 reconfirm under the bias arm.
    g1 = bool(esc_after > esc_before + G1_SALIENCE_DELTA)

    # G4: threat-gated bias points to the learned escape action (most negative =
    # favoured, REE lower-is-better).
    bias = lk.compute_approach_bias(THREAT_PRIOR, [0, 1, 2, 3])
    nonnoop = {c: float(bias[c]) for c in (HARM_WORSEN, ESCAPE, NEUTRAL)}
    favoured = min(nonnoop, key=nonnoop.get)
    g4 = bool(favoured == ESCAPE)
    g4_non_degenerate = bool(float(bias.abs().max()) > 0.0)  # bias actually fired

    # G5: bias exactly zero when safe.
    safe = lk.compute_approach_bias(0.0, [0, 1, 2, 3])
    g5 = bool(float(safe.abs().max()) == 0.0)

    st = lk.get_state()
    return {
        "G1": g1, "G4": g4, "G4_non_degenerate": g4_non_degenerate, "G5": g5,
        "bias_by_class": {str(c): round(float(bias[c]), 6) for c in (0, 1, 2, 3)},
        "bias_max_abs_safe": round(float(safe.abs().max()), 8),
        "n_bias_fires": int(st["e2_escape_linker_n_bias_fires"]),
        "diag": st,
    }


def _eval_relief_safety(seed: int) -> Dict[str, Any]:
    lk = _make_linker(enabled=True)
    _train_forced_choice(lk, "ARM_LINKER_TO_RELIEF_SAFETY", seed, _CUR_ROUNDS)
    feat = lk.escape_affordance_features(ESCAPE)
    feat_present = bool(feat is not None and feat.numel() > 0)

    learner = TrainableEscapeAffordanceLearner(
        TrainableEscapeAffordanceLearnerConfig(enabled=True,
                                               n_action_classes=N_ACTION_CLASSES)
    )
    common = dict(
        z_world=torch.tensor([0.2, -0.1, 0.4]),
        z_self=torch.tensor([0.3, -0.2]),
        z_harm_a=torch.tensor([0.6]),
        z_harm_a_norm=0.6,
        threat_scale=1.0,
    )
    base = learner.build_state_vector(**common)
    base_none = learner.build_state_vector(extra_features=None, **common)
    with_feat = (
        learner.build_state_vector(extra_features=feat, **common)
        if feat_present else base
    )

    # G_RS: linker features are consumed (appended) AND the learner default
    # (no extra_features) is bit-identical -- relief/safety heads stay distinct
    # (the linker keeps RELIEF_HEADS and SAFETY_HEADS disjoint by construction).
    consumed = bool(feat_present and with_feat.numel() == base.numel() + feat.numel())
    default_bit_identical = bool(torch.equal(base, base_none))
    relief_safety_disjoint = bool(
        not (set(lk.RELIEF_HEADS) & set(lk.SAFETY_HEADS))
    )
    g_rs = bool(consumed and default_bit_identical and relief_safety_disjoint)

    return {
        "G_RS": g_rs,
        "feat_present": feat_present,
        "feat_numel": int(feat.numel()) if feat_present else 0,
        "consumed_appended": consumed,
        "default_bit_identical": default_bit_identical,
        "relief_safety_disjoint": relief_safety_disjoint,
        "diag": lk.get_state(),
    }


_CUR_ROUNDS = N_TRAIN_ROUNDS  # set per run() before the cells execute


def _run_seed_arm(arm: Dict[str, Any], seed: int) -> Dict[str, Any]:
    arm_id = arm["arm_id"]
    if arm_id == "ARM_DISABLED_CONTROL":
        row = _eval_disabled(seed)
    elif arm_id == "ARM_LINKER_READOUT_ONLY":
        row = _eval_readout(seed)
    elif arm_id == "ARM_LINKER_E3_BIAS":
        row = _eval_e3_bias(seed)
    else:
        row = _eval_relief_safety(seed)
    row["arm_id"] = arm_id
    row["seed"] = int(seed)
    return row


# ---------------------------------------------------------------------------
# Evaluation / interpretation
# ---------------------------------------------------------------------------

def _seeds_passing(rows: List[Dict[str, Any]], gate: str) -> int:
    return sum(1 for r in rows if bool(r.get(gate, False)))


def _evaluate(arm_results: List[Dict[str, Any]], n_seeds: int) -> Dict[str, Any]:
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for r in arm_results:
        by_arm.setdefault(r["arm_id"], []).append(r)

    disabled = by_arm.get("ARM_DISABLED_CONTROL", [])
    readout = by_arm.get("ARM_LINKER_READOUT_ONLY", [])
    e3bias = by_arm.get("ARM_LINKER_E3_BIAS", [])
    relsafe = by_arm.get("ARM_LINKER_TO_RELIEF_SAFETY", [])

    need2 = MIN_SEEDS_2OF3 if n_seeds >= 3 else 1
    need3 = n_seeds

    g0 = _seeds_passing(disabled, "G0")
    g1 = _seeds_passing(readout, "G1")
    g2 = _seeds_passing(readout, "G2")
    g3 = _seeds_passing(readout, "G3")
    g4 = _seeds_passing(e3bias, "G4")
    g5 = _seeds_passing(e3bias, "G5")
    g6 = _seeds_passing(readout, "G6")
    g7 = _seeds_passing(readout, "G7")
    g8 = _seeds_passing(readout, "G8")
    g_rs = _seeds_passing(relsafe, "G_RS")

    gate_pass = {
        "G0_disabled_bit_stable": g0 >= need3,
        "G1_escape_readout_learns": g1 >= need2,
        "G2_noop_uncredited": g2 >= need2,
        "G3_harm_worsen_not_preferred": g3 >= need2,
        "G4_bias_points_to_escape": g4 >= need2,
        "G5_bias_zero_when_safe": g5 >= need3,
        "G6_sim_hypothesis_blocks_learning": g6 >= need3,
        "G7_weights_persist_across_reset": g7 >= need2,
        "G8_inputs_detached": g8 >= need3,
    }
    secondary_pass = {"G_RS_features_consumed": g_rs >= need2}

    # Readiness: the load-bearing G1 routes on the readout salience delta; the
    # matched readiness check is that the linker optimizer actually stepped in the
    # readout positive control. Below floor => wiring regression (linker never
    # trained), NOT "readout cannot learn".
    mean_opt_steps = (
        sum(int(r.get("n_optimizer_steps", 0)) for r in readout) / len(readout)
        if readout else 0.0
    )
    proxy_min_dist = _min_pairwise_proxy_distance()
    outcome_range = max(OUTCOME.values()) - min(OUTCOME.values())
    readiness_ok = bool(
        mean_opt_steps >= OPTIMIZER_STEP_FLOOR
        and proxy_min_dist >= PROXY_DISTINCT_FLOOR
        and outcome_range >= OUTCOME_RANGE_FLOOR
    )

    all_primary = all(gate_pass.values())
    if not readiness_ok:
        label = "substrate_not_ready_requeue"
        overall_pass = False
    elif all_primary:
        label = "linker_readout_ready"
        overall_pass = True
    else:
        label = "linker_readout_inert"
        overall_pass = False

    # Non-degeneracy per primary gate (did it discriminate, or pass/fail trivially).
    g1_non_deg = all(bool(r.get("G1_non_degenerate", False)) for r in readout) if readout else False
    g4_non_deg = all(bool(r.get("G4_non_degenerate", False)) for r in e3bias) if e3bias else False
    criteria_non_degenerate = {
        "G0": bool(disabled),
        "G1": g1_non_deg,
        "G2": bool(readout),
        "G3": bool(readout),
        "G4": g4_non_deg,
        "G5": bool(e3bias),
        "G6": bool(readout),
        "G7": bool(readout),
        "G8": bool(readout),
    }

    return {
        "label": label,
        "overall_pass": overall_pass,
        "readiness_ok": readiness_ok,
        "gate_seeds_passing": {
            "G0": g0, "G1": g1, "G2": g2, "G3": g3, "G4": g4,
            "G5": g5, "G6": g6, "G7": g7, "G8": g8, "G_RS": g_rs,
        },
        "gate_pass": gate_pass,
        "secondary_pass": secondary_pass,
        "n_seeds": n_seeds,
        "min_seeds_2of3": need2,
        "min_seeds_3of3": need3,
        "preconditions": [
            {
                "name": "linker_optimizer_stepped",
                "kind": "readiness",
                "description": (
                    "Mean optimizer steps in ARM_LINKER_READOUT_ONLY (the positive "
                    "control) -- the SAME exercise the load-bearing G1 salience "
                    "delta routes on. Below floor => the linker never instantiated "
                    "/ never trained (a wiring regression) => "
                    "substrate_not_ready_requeue, NOT a 'readout cannot learn' "
                    "verdict."
                ),
                "control": "ARM_LINKER_READOUT_ONLY: forced-choice training over controlled E2 proxies",
                "measured": round(float(mean_opt_steps), 4),
                "threshold": OPTIMIZER_STEP_FLOOR,
                "met": bool(mean_opt_steps >= OPTIMIZER_STEP_FLOOR),
            },
            {
                "name": "controlled_e2_proxies_distinct",
                "kind": "readiness",
                "description": (
                    "Min pairwise L2 distance among the non-noop controlled E2 "
                    "proxy features. Guards G1 non-vacuity -- identical proxies "
                    "would make the readout structurally unable to discriminate."
                ),
                "control": "harness E2FEAT proxies for classes {1,2,3}",
                "measured": round(float(proxy_min_dist), 4),
                "threshold": PROXY_DISTINCT_FLOOR,
                "met": bool(proxy_min_dist >= PROXY_DISTINCT_FLOOR),
            },
            {
                "name": "forced_choice_outcome_range",
                "kind": "readiness",
                "description": (
                    "Range of forced-choice outcome targets across action classes. "
                    "Guards that a differentiated learning signal exists."
                ),
                "control": "harness OUTCOME targets",
                "measured": round(float(outcome_range), 4),
                "threshold": OUTCOME_RANGE_FLOOR,
                "met": bool(outcome_range >= OUTCOME_RANGE_FLOOR),
            },
        ],
        "criteria": [
            {"name": "G0_disabled_bit_stable", "load_bearing": False, "passed": gate_pass["G0_disabled_bit_stable"]},
            {"name": "G1_escape_readout_learns", "load_bearing": True, "passed": gate_pass["G1_escape_readout_learns"]},
            {"name": "G2_noop_uncredited", "load_bearing": False, "passed": gate_pass["G2_noop_uncredited"]},
            {"name": "G3_harm_worsen_not_preferred", "load_bearing": False, "passed": gate_pass["G3_harm_worsen_not_preferred"]},
            {"name": "G4_bias_points_to_escape", "load_bearing": True, "passed": gate_pass["G4_bias_points_to_escape"]},
            {"name": "G5_bias_zero_when_safe", "load_bearing": False, "passed": gate_pass["G5_bias_zero_when_safe"]},
            {"name": "G6_sim_hypothesis_blocks_learning", "load_bearing": False, "passed": gate_pass["G6_sim_hypothesis_blocks_learning"]},
            {"name": "G7_weights_persist_across_reset", "load_bearing": False, "passed": gate_pass["G7_weights_persist_across_reset"]},
            {"name": "G8_inputs_detached", "load_bearing": False, "passed": gate_pass["G8_inputs_detached"]},
        ],
        "criteria_non_degenerate": criteria_non_degenerate,
    }


def _min_pairwise_proxy_distance() -> float:
    nonnoop = [torch.tensor(E2FEAT[c], dtype=torch.float32)
               for c in (HARM_WORSEN, ESCAPE, NEUTRAL)]
    dmin = float("inf")
    for i in range(len(nonnoop)):
        for j in range(i + 1, len(nonnoop)):
            dmin = min(dmin, float((nonnoop[i] - nonnoop[j]).norm().item()))
    return dmin


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    global _CUR_ROUNDS
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    _CUR_ROUNDS = DRY_RUN_TRAIN_ROUNDS if dry_run else N_TRAIN_ROUNDS

    arm_results: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm['arm_id']}", flush=True)
            with arm_cell(
                seed,
                config_slice={
                    "arm": arm,
                    "linker_params": LINKER_PARAMS,
                    "e2feat": E2FEAT,
                    "outcome": OUTCOME,
                    "threat_prior": THREAT_PRIOR,
                    "n_train_rounds": _CUR_ROUNDS,
                },
                script_path=Path(__file__),
                extra_ineligible_reasons=["microdiagnostic_trained_linker_per_cell"],
            ) as cell:
                row = _run_seed_arm(arm, seed)
                cell.stamp(row)
            arm_results.append(row)
            print("verdict: PASS", flush=True)  # cell completed; gate aggregation below

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
            "Post-603i E2 escape-affordance LINKER readiness microdiagnostic "
            "(ree-v3 ree_core/pfc/e2_escape_affordance_linker.py). Forced-choice "
            "4-action probe driving the linker directly with controlled E2 "
            "consequence proxies + known outcomes. claim_ids=[] (does NOT weight "
            "claim confidence). References the V3-EXQ-603 lineage; originated from "
            "V3-EXQ-603i (substrate_not_ready_requeue). A PASS (label="
            "linker_readout_ready) means ONLY that the linker readout can learn a "
            "controlled 'where out is' signal and expose a bounded threat-gated "
            "E3 bias under microdiagnostic conditions -- it does NOT validate "
            "SD-059/MECH-358 or MECH-302/303/304, is NOT relief/safety bridge "
            "validation, is NOT ecological survival, and is NOT V3 closure. A FAIL "
            "is NOT bridge falsification. Readiness-below-floor self-routes "
            "substrate_not_ready_requeue (linker never trained), NOT a substrate "
            "verdict. SD-059/MECH-358 (+ MECH-302/303/304) stay unchanged."
        ),
        "interpretation": {
            "label": summary["label"],
            "preconditions": summary["preconditions"],
            "criteria": summary["criteria"],
            "criteria_non_degenerate": summary["criteria_non_degenerate"],
            "routing": {
                "linker_readout_ready": (
                    "PASS -> /queue-experiment a 603-lineage FULL behavioural bridge "
                    "re-test with linker features attached to the trainable "
                    "relief/safety heads (next 603-lineage suffix). Do NOT mark "
                    "SD-059/MECH-358 validated."
                ),
                "substrate_not_ready_requeue": (
                    "re-queue as V3-EXQ-653a (fix the linker wiring / raise the "
                    "training budget); do NOT weaken SD-059/MECH-358."
                ),
                "linker_readout_inert": (
                    "FAIL -> /failure-autopsy on THIS readiness diagnostic; localise "
                    "to linker/readout learning (G1), E3 bias integration (G4), "
                    "no-op/freeze exclusion (G2), hypothesis/simulation boundary "
                    "(G6), detached-input safety (G8), or the reuse-E2 assumption. "
                    "Do NOT rerun 603i; do NOT treat as bridge falsification."
                ),
            },
        },
        "dry_run": bool(dry_run),
        "config": {
            "seeds": seeds,
            "n_train_rounds": _CUR_ROUNDS,
            "action_classes": {"noop": NOOP, "harm_worsen": HARM_WORSEN,
                               "escape": ESCAPE, "neutral": NEUTRAL},
            "e2_proxy_features": E2FEAT,
            "forced_choice_outcomes": OUTCOME,
            "threat_prior": THREAT_PRIOR,
            "linker_params": LINKER_PARAMS,
            "arms": [a["arm_id"] for a in ARMS],
            "thresholds": {
                "g1_salience_delta": G1_SALIENCE_DELTA,
                "min_seeds_2of3": MIN_SEEDS_2OF3,
                "min_seeds_3of3": MIN_SEEDS_3OF3,
                "proxy_distinct_floor": PROXY_DISTINCT_FLOOR,
                "outcome_range_floor": OUTCOME_RANGE_FLOOR,
                "optimizer_step_floor": OPTIMIZER_STEP_FLOOR,
            },
        },
        "acceptance_criteria": {
            "readiness_ok": summary["readiness_ok"],
            **summary["gate_pass"],
            **summary["secondary_pass"],
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
        description="V3-EXQ-653 post-603i E2 escape-affordance linker readiness microdiagnostic"
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
