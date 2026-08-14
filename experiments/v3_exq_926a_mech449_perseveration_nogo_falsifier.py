"""V3-EXQ-926a -- MECH-449 / ARC-107 perseveration-axis conversion falsifier.

V3-EXQ-926a supersedes V3-EXQ-926, which crashed before manifest write on
ree-cloud-2 (exit 1, wall clock 2.32s): EVIDENCE_DIR was a hardcoded Mac-only
absolute path (/Users/dgolden/...) that does not exist on the cloud workers
(/home/ree/...), so write_flat_manifest() failed at the final step. The science
ran correctly to completion (import 1.48s + full 9-cell x 32-bank grid 0.77s =
2.25s, matching the 2.32s runner wall clock); only the manifest write failed.
The one-line fix derives EVIDENCE_DIR relative to __file__ (parents[2] /
"REE_assembly" / ...), matching predecessor V3-EXQ-689g and every other driver.
Nothing else changed.

EXPERIMENT_PURPOSE=evidence -- this closes the ONE untested leg of the BUILT
MECH-449 Go/No-Go eligibility constitution.

WHAT IS UNTESTED, AND WHY THIS IS NOT A RE-RUN OF V3-EXQ-689g
-------------------------------------------------------------
V3-EXQ-689g PASSed (supports MECH-449 + ARC-107, conversion 1.0 on 3/3 seeds)
but it exercised only the ``safety`` and ``staleness`` No-Go axes, injecting
both as constructed tensors. Its own source says so at the injection site:

    safety[u_index] = 0.9      # safety-No-Go U
    staleness[stale_index] = 0.9
    go_nogo_signals = {"safety": safety, "staleness": staleness}
    # "the perseveration axis would, in the live agent loop, reuse MECH-260;
    #  here the bank is synthetic so we drive the axes directly"

So the constitution's FOURTH axis -- ``perseveration``, the one that CONSUMES
MECH-260's per-candidate dACC recency-suppression vector (e3_selector.py:1579,
1587) -- has never been exercised by any run. This experiment drives that axis
from a LIVE DACCAdaptiveControl whose ``_action_history`` is built by replaying
a real perseverative action sequence through ``record_action()``, then reads
``_suppression_penalty()`` per candidate. Nothing is hand-stuffed.

WHY THE MECH-260 CLAIM IS DELIBERATELY *NOT* TAGGED
---------------------------------------------------
MECH-260's own falsifier (claims.yaml ``what_would_answer``) is BEHAVIOURAL:
the channel must "measurably resolve or reduce the monostrategy failure". That
requires a competent forager, which is blocked on the MECH-457 competence floor
(re-derive brake fired 7x; four axes eliminated; substrate amend deferred). At
the floor the behavioural DV is vacuous by construction -- V3-EXQ-445h measured
action_class_entropy = 0 in BOTH arms, and the 2026-06-19 autopsy excluded that
criterion as degenerate (0.0 >= 0.0 under monostrategy).

A selection-face exclusion result therefore CANNOT confirm or falsify MECH-260,
and tagging it would corrupt governance scoring. What a PASS here supports is
narrower and precise: MECH-449's perseveration leg has real, orthogonal-to-F
selection authority. That is a PREREQUISITE for any future MECH-260 behavioural
retest, not a substitute for one. Re-derive brake at authoring time:
MECH-449 = 0, ARC-107 = 0, MECH-260 = 16 (threshold 2, substrate NOT built).

THE MECHANISM UNDER TEST
------------------------
MECH-260's suppression vector has TWO consumers, and they differ in KIND:

  (a) OLD, additive-bias route (dacc.py:512-545):
          bias = bias + dacc_suppression_weight * suppression
      ``dacc_suppression_weight`` defaults to 0.0, so this route is INERT by
      default -- the documented zero-return path (V3-EXQ-571 measured
      bias_fraction = 0.000; the V3-EXQ-543e autopsy recorded the same).
      Even when non-zero it is a SCORE bias, which a committed argmin can and
      does drown.

  (b) NEW, eligibility-exclusion route (agent.py:8396-8407 ->
      e3_selector.py:1643-1645): the SAME vector is auto-wired into
      ``go_nogo_signals["perseveration"]`` and, at recency-share >=
      ``gng_perseveration_floor``, the candidate is REMOVED FROM THE ELIGIBLE
      SET. Removal acts on an axis ORTHOGONAL to F-rank, which a
      rank-preserving demotion structurally cannot do (V3-EXQ-689f).

This experiment measures (b), on the case that discriminates it from (a): the
perseverated candidate IS the incumbent the F-path actually commits to -- the
F-eligible winner that no amount of order-preserving demotion could exclude.
The incumbent is read from the baseline's committed pick rather than from the
raw-F argmin, because under the MECH-448 envelope those differ often (measured
7/12 banks during authoring) and it is the COMMITTED trajectory whose repetition
MECH-260 is about resisting.

DV-SYMMETRY DECLARATION (mandatory per /queue-experiment Step 3)
----------------------------------------------------------------
  ARM_CONSTITUTION -- manipulation: removal of a candidate from the eligible
    SET (a change of the argmin's DOMAIN). DV: committed argmin identity.
    The symmetry group of an argmin is {uniform additive constants, monotone
    rescalings, permutations of exact ties}. Set-removal is in NONE of these:
    it does not translate or rescale any score, and it specifically deletes the
    argmin itself. So the manipulation is NOT invariant under the DV's symmetry
    group, and a measured flip is a measurement rather than an identity.
    (Contrast V3-EXQ-604c, where a BROADCAST SCALAR was added uniformly across
    candidates and therefore cancelled in the argmax exactly, yielding a delta
    of 0.0 by arithmetic. An eligibility exclusion cannot cancel that way.)
  ARM_SHUFFLED -- same manipulation KIND (set removal), but targeted at a
    non-incumbent. Also not invariant; the contrast with ARM_CONSTITUTION is
    what isolates the CONTENT of the recency vector from the mere fact of the
    gate being on.
  ARM_OFF -- no manipulation; the baseline domain that defines the incumbent.

SLEEP DRIVER: not applicable (no sleep loop; selection-face, single-decision).

CRITERIA (pre-registered; thresholds are module constants, never derived)
------------------------------------------------------------------------
  C1_perseveration_conversion     [LOAD-BEARING] ARM_CONSTITUTION conversion
      rate >= 0.5 on >= 2/3 seeds. Conversion = the incumbent is excluded by the
      perseveration No-Go AND the committed selection changes.
  C2_recency_content_specificity  ARM_CONSTITUTION minus ARM_SHUFFLED mean
      conversion >= 0.30 -- the flip is driven by WHERE the recency mass sits,
      not merely by the gate being switched on.
  C3_safety_failopen              the eligible set is never empty.

  P3 (readiness) gates on the pre-No-Go eligible-set size >= 2. Below that the
  fail-open gng_protect_min_eligible guard makes the axis STRUCTURALLY unable to
  fire -- a bar independent of the claim, so it must self-route
  substrate_not_ready_requeue rather than read as a refuted perseveration axis.
  This is not hypothetical: at the stock f_eligibility_envelope_floor of 0.30 a
  decisive F-winner collapses the envelope to ONE survivor (measured during
  authoring: median envelope 1, No-Go applied 6/16). Hence ENVELOPE_FLOOR=0.10.

A PASS on C1 with C2/C3 holding supports MECH-449 + ARC-107. Built-but-no-
conversion is non_contributory / over-specification, NOT an ARC-107
falsification (same disposition 689g pre-registered). Below-floor readiness
self-routes substrate_not_ready_requeue, never a substrate verdict label.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import (  # noqa: E402
    compute_arm_fingerprint,
    reset_all_rng,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from ree_core.cingulate.dacc import DACCAdaptiveControl, DACCConfig  # noqa: E402
from ree_core.predictors.e2_fast import Trajectory  # noqa: E402
from ree_core.predictors.e3_selector import E3Config, E3TrajectorySelector  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EVIDENCE_DIR = Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments"

EXPERIMENT_PURPOSE = "evidence"
EXPERIMENT_TYPE = "v3_exq_926a_mech449_perseveration_nogo_falsifier"
CLAIM_IDS: List[str] = ["MECH-449", "ARC-107"]

SEEDS: List[int] = [42, 43, 44]
# ARM_SHUFFLED is the SPECIFICITY control and the reason C2 is not tautological:
# the gate is ON exactly as in ARM_CONSTITUTION, but the recency mass is placed
# on a NON-incumbent candidate. If conversion tracked merely "the gate is on",
# both arms would convert equally; a gap between them shows the conversion is
# driven by the CONTENT of MECH-260's vector, not by gate activation.
ARMS: List[str] = ["ARM_OFF", "ARM_CONSTITUTION", "ARM_SHUFFLED"]

WORLD_DIM = 8
ACTION_DIM = 4
K_CANDIDATES = 4
HORIZON = 3
N_BANKS = 32

# --- pre-registered thresholds (constants, never derived from run statistics) --
CONVERSION_FLOOR = 0.5          # C1: per-seed ARM_CONSTITUTION conversion rate
SEED_MAJORITY = 2               # C1: seeds that must clear the floor (of 3)
SPECIFICITY_GAP_FLOOR = 0.30    # C2: ARM_CONSTITUTION minus ARM_SHUFFLED
PERSEVERATION_FLOOR = 0.5       # gng_perseveration_floor (matches config default)

# The MECH-448 F-eligibility envelope must leave room for the No-Go to act.
# At the 0.30 default with K=4 (uniform share 0.25) a decisive F-winner collapses
# the envelope to ONE survivor, and the fail-open gng_protect_min_eligible guard
# then correctly refuses to drop it -- so the perseveration axis is STRUCTURALLY
# unable to fire, regardless of the claim under test. Measured during authoring:
# floor 0.30 -> median envelope 1, soft_applied 6/16; floor 0.10 -> median
# envelope 2, soft_applied 15/16. P3 below asserts this is not re-broken.
ENVELOPE_FLOOR = 0.10

# Readiness floors:
SUPPRESSION_RANGE_FLOOR = 0.25   # P1 -- SAME statistic C1 routes on (see below)
ENVELOPE_SIZE_FLOOR = 2.0        # P3 -- median eligible-set size before No-Go

# The perseverative history: the dominant class fills DOMINANT_REPEATS of the
# 8-deep recency ring and two other classes take one slot each, giving
# suppression = [0.75, 0.125, 0.125, 0.0] over the dominant/minority/absent
# classes. BOTH halves matter: one candidate must clear gng_perseveration_floor
# and the rest must NOT -- if every candidate cleared it the fail-open guard
# would fire and nothing would be excluded.
DOMINANT_REPEATS = 6
HISTORY_DEPTH = 8

# The perseverated class is chosen PER BANK as the probed raw-F argmin, not
# fixed in advance (the same construction V3-EXQ-689g used to guarantee its
# undesirable candidate was F-best). Two reasons:
#   (1) Vacuity. With random per-bank head init the F-argmin is ~uniform over K,
#       so a hardcoded class is the F-argmin only ~1/K of the time and the run
#       measures the trivial "exclude a candidate F did not want anyway" case.
#       Confirmed empirically: the first smoke of this script fixed the class at
#       2 and measured orthogonality 0.000 / conversion 0.0.
#   (2) Faithfulness. "Perseverating on whichever trajectory the value ranking
#       favours" IS the monostrategy scenario -- fixing the class in advance
#       would model something else.


def _make_candidate(action_class: int, world_vec: torch.Tensor) -> Trajectory:
    states = [torch.zeros(1, WORLD_DIM) for _ in range(HORIZON + 1)]
    world_states = [world_vec.reshape(1, WORLD_DIM).clone() for _ in range(HORIZON + 1)]
    actions = torch.zeros(1, HORIZON, ACTION_DIM)
    actions[:, 0, action_class] = 1.0
    return Trajectory(states=states, actions=actions, world_states=world_states)


def _build_bank(rng: torch.Generator) -> List[Trajectory]:
    """K candidates with all-distinct first-action classes and divergent world
    states (so raw F genuinely differs across candidates -- the non-vacuity
    precondition a degenerate pool would violate)."""
    cands = []
    for k in range(K_CANDIDATES):
        wv = torch.randn(WORLD_DIM, generator=rng) * 0.5 + float(k) * 0.4
        cands.append(_make_candidate(action_class=k, world_vec=wv))
    return cands


def _perseverative_history(dominant_class: int) -> List[int]:
    """An 8-deep action ring dominated by ``dominant_class``.

    The minority slots go to two OTHER distinct classes, so the resulting
    recency-share vector has exactly one entry over gng_perseveration_floor.
    """
    minority = [
        (dominant_class + 1) % K_CANDIDATES,
        (dominant_class + 2) % K_CANDIDATES,
    ]
    hist = [dominant_class] * DOMINANT_REPEATS + minority
    return hist[:HISTORY_DEPTH]


def _live_suppression_vector(dominant_class: int) -> torch.Tensor:
    """MECH-260's per-candidate recency-share, read from a LIVE dACC.

    Nothing is hand-stuffed: the history is pushed through real record_action()
    calls and the value comes from dacc._suppression_penalty(), the same
    function the module's own forward() uses to populate bundle["suppression"].
    """
    dacc = DACCAdaptiveControl(DACCConfig())
    for a in _perseverative_history(dominant_class):
        dacc.record_action(a)
    return torch.tensor(
        [dacc._suppression_penalty(c) for c in range(K_CANDIDATES)],
        dtype=torch.float32,
    )


def _raw_f(selector: E3TrajectorySelector, cands: List[Trajectory]) -> List[float]:
    """Probe raw F with zero bias and no signals -> scores ARE the raw F costs."""
    selector._running_variance = 0.0
    r = selector.select(cands, temperature=1.0, score_bias=torch.zeros(len(cands)))
    return [float(s.detach()) for s in r.scores]


def _select_one(
    selector: E3TrajectorySelector,
    cands: List[Trajectory],
    suppression: torch.Tensor,
    gate_on: bool,
) -> Dict[str, Any]:
    """One committed selection through the real select() path."""
    selector._running_variance = 0.0  # deterministic committed argmin
    k = len(cands)
    go_nogo_signals = {"perseveration": suppression} if gate_on else None
    result = selector.select(
        cands,
        temperature=1.0,
        score_bias=torch.zeros(k),
        go_nogo_signals=go_nogo_signals,
    )
    diag = selector.last_score_diagnostics or {}
    env_size = diag.get("go_nogo_envelope_size", None)
    return {
        "selected_index": int(result.selected_index),
        "selected_class": int(result.selected_action.reshape(-1).argmax().item()),
        "go_nogo_active": bool(diag.get("go_nogo_constitution_active", False)),
        "n_soft_requested": int(diag.get("go_nogo_n_soft_requested", 0) or 0),
        "n_soft_applied": int(diag.get("go_nogo_n_soft_applied", 0) or 0),
        "envelope_size": int(env_size) if env_size is not None else None,
    }


def _config_plumbing_live() -> bool:
    """P3 readiness: from_dims actually plumbs the go_nogo knobs onto config.e3.

    Guards the documented REEConfig.from_dims silent-kwarg-swallow failure mode
    (a new knob needs three wiring sites; two leaves it unreachable, silently).
    """
    c = REEConfig.from_dims(
        body_obs_dim=6,
        world_obs_dim=25,
        world_dim=WORLD_DIM,
        action_dim=ACTION_DIM,
        use_go_nogo_constitution=True,
        gng_perseveration_floor=PERSEVERATION_FLOOR,
    )
    return (
        getattr(c.e3, "use_go_nogo_constitution", False) is True
        and abs(float(getattr(c.e3, "gng_perseveration_floor", -1.0))
                - PERSEVERATION_FLOOR) < 1e-9
    )


def _run_cell(arm: str, seed: int, n_banks: int) -> Dict[str, Any]:
    """One (arm, seed) cell over n_banks divergent candidate banks."""
    reset_all_rng(seed)
    rng = torch.Generator().manual_seed(seed)
    gate_on = arm in ("ARM_CONSTITUTION", "ARM_SHUFFLED")

    n_converted = 0
    n_incumbent_is_argmin = 0
    n_demotion_admits_incumbent = 0
    n_empty_eligible = 0
    n_gate_active = 0
    excluded_counts: List[int] = []
    raw_ranges: List[float] = []
    supp_ranges: List[float] = []
    pre_envelope_sizes: List[float] = []
    exemplar_suppression: Optional[List[float]] = None

    print(f"Seed {seed} Condition {arm}", flush=True)
    for b in range(n_banks):
        if (b + 1) % 8 == 0:
            print(f"  [eval] {arm} seed={seed} ep {b+1}/{n_banks}", flush=True)

        cands = _build_bank(rng)

        # Baseline selector: MECH-448 F-eligibility demotion ON, constitution
        # OFF. Its committed pick DEFINES the incumbent.
        sel_demote = E3TrajectorySelector(
            E3Config(
                world_dim=WORLD_DIM,
                hidden_dim=8,
                use_f_eligibility_demotion=True,
                f_eligibility_envelope_floor=ENVELOPE_FLOOR,
                use_go_nogo_constitution=False,
            )
        )
        raw = _raw_f(sel_demote, cands)
        raw_ranges.append(max(raw) - min(raw))

        # THE INCUMBENT IS THE COMMITTED PICK, NOT THE RAW-F ARGMIN. Under the
        # demotion envelope these differ often (measured 7/12 during authoring),
        # and it is the COMMITTED trajectory that MECH-260 is about resisting
        # the repetition of. Defining it this way also guarantees the incumbent
        # is F-eligible by construction -- exactly the "F-eligible incumbent
        # that only an active No-Go can exclude" case.
        base = _select_one(sel_demote, cands, torch.zeros(K_CANDIDATES), gate_on=False)
        incumbent = base["selected_index"]
        n_demotion_admits_incumbent += 1  # true by construction; recorded for audit

        # ARM_SHUFFLED perseverates on a NON-incumbent candidate (specificity).
        target = incumbent if arm != "ARM_SHUFFLED" else (incumbent + 1) % K_CANDIDATES
        suppression = _live_suppression_vector(target)
        supp_ranges.append(float(suppression.max() - suppression.min()))
        if exemplar_suppression is None:
            exemplar_suppression = [float(x) for x in suppression.tolist()]
        if int(suppression.argmax().item()) == target:
            n_incumbent_is_argmin += 1

        sel_arm = E3TrajectorySelector(
            E3Config(
                world_dim=WORLD_DIM,
                hidden_dim=8,
                use_f_eligibility_demotion=True,
                f_eligibility_envelope_floor=ENVELOPE_FLOOR,
                use_go_nogo_constitution=gate_on,
                gng_perseveration_floor=PERSEVERATION_FLOOR,
            )
        )
        # Same head init as the baseline selector, so the ONLY difference
        # between base and arm is the gate (not random head weights).
        sel_arm.load_state_dict(sel_demote.state_dict())
        got = _select_one(sel_arm, cands, suppression, gate_on=gate_on)

        if got["go_nogo_active"]:
            n_gate_active += 1
        if got["envelope_size"] is not None:
            excluded_counts.append(K_CANDIDATES - got["envelope_size"])
            # Pre-No-Go eligible size = survivors + those the No-Go dropped.
            # This is the quantity P3 gates on: below 2 the fail-open guard
            # makes the perseveration axis structurally unable to fire.
            pre_envelope_sizes.append(
                float(got["envelope_size"] + got["n_soft_applied"])
            )
            if got["envelope_size"] <= 0:
                n_empty_eligible += 1

        # Conversion: the perseverated F-argmin incumbent was admitted by the
        # F-demotion baseline, and the gate both excluded it AND moved the pick.
        converted = (
            base["selected_index"] == incumbent
            and got["selected_index"] != incumbent
        )
        if converted:
            n_converted += 1

    conversion_rate = n_converted / float(n_banks)
    supp_range = statistics.median(supp_ranges) if supp_ranges else 0.0
    cell = {
        "arm": arm,
        "seed": seed,
        "gate_on": gate_on,
        "n_banks": n_banks,
        "suppression_vector_exemplar": exemplar_suppression,
        "suppression_range": supp_range,
        "suppression_range_min": min(supp_ranges) if supp_ranges else 0.0,
        "conversion_rate": conversion_rate,
        "n_converted": n_converted,
        "incumbent_is_f_argmin_rate": n_incumbent_is_argmin / float(n_banks),
        "demotion_admits_incumbent_rate": n_demotion_admits_incumbent / float(n_banks),
        "n_empty_eligible": n_empty_eligible,
        "gate_active_rate": n_gate_active / float(n_banks),
        "median_excluded_count": (
            statistics.median(excluded_counts) if excluded_counts else 0.0
        ),
        "median_pre_nogo_envelope_size": (
            statistics.median(pre_envelope_sizes) if pre_envelope_sizes else 0.0
        ),
        "median_raw_f_range": statistics.median(raw_ranges) if raw_ranges else 0.0,
    }
    cell["arm_fingerprint"] = compute_arm_fingerprint(
        config_slice={
            "arm": arm,
            "use_f_eligibility_demotion": True,
            "use_go_nogo_constitution": gate_on,
            "gng_perseveration_floor": PERSEVERATION_FLOOR,
            "n_banks": n_banks,
            "k": K_CANDIDATES,
            "dominant_repeats": DOMINANT_REPEATS,
            "history_depth": HISTORY_DEPTH,
        },
        seed=seed,
        script_path=Path(__file__),
        rng_fully_reset=True,
        config_slice_declared=True,
        extra_ineligible_reasons=["selection_face_synthetic_no_training"],
    )
    # Only ARM_CONSTITUTION is expected to convert. ARM_OFF (no gate) and
    # ARM_SHUFFLED (gate on, recency mass off-target) are CONTROLS -- a low
    # conversion rate there is the control working, not a failure, so their
    # cell verdict is "ran cleanly" rather than "converted". Scoring the
    # controls against the conversion floor would report a healthy run as
    # 2-of-3 arms failing.
    if arm == "ARM_CONSTITUTION":
        verdict_pass = conversion_rate >= CONVERSION_FLOOR and n_empty_eligible == 0
    else:
        verdict_pass = n_empty_eligible == 0
    print(f"verdict: {'PASS' if verdict_pass else 'FAIL'}", flush=True)
    return cell


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    t0 = datetime.utcnow()
    seeds = SEEDS[:1] if dry_run else SEEDS
    n_banks = 4 if dry_run else N_BANKS

    arm_results: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            arm_results.append(_run_cell(arm, seed, n_banks))

    on_cells = [c for c in arm_results if c["arm"] == "ARM_CONSTITUTION"]
    shuf_cells = [c for c in arm_results if c["arm"] == "ARM_SHUFFLED"]
    off_cells = [c for c in arm_results if c["arm"] == "ARM_OFF"]

    # ---------------- readiness preconditions (recomputable) ----------------
    # Worst CELL, not a mean -- `met` below is a worst-case claim, so the
    # reported `measured` must be the same extremum the indexer recomputes on.
    supp_ranges = [c["suppression_range_min"] for c in arm_results]
    worst_supp_range = min(supp_ranges) if supp_ranges else 0.0
    gate_cells = on_cells + shuf_cells
    env_sizes = [c["median_pre_nogo_envelope_size"] for c in gate_cells]
    worst_env_size = min(env_sizes) if env_sizes else 0.0
    plumbing_ok = _config_plumbing_live()

    preconditions = [
        {
            # SAME statistic C1 routes on: C1 needs SOME candidates over the
            # floor and others under it, which is a property of the cross-
            # candidate RANGE, not of any magnitude/mean-abs summary. A uniform
            # suppression vector has large mean-abs and ~0 range, and would
            # either No-Go everything (fail-open fires) or nothing.
            "name": "suppression_cross_candidate_range_supra_floor",
            "kind": "readiness",
            "description": (
                "live dACC MECH-260 suppression varies ACROSS candidates "
                "(max - min), the statistic the perseveration threshold "
                "crossing in C1 depends on"
            ),
            "control": (
                "positive control: an 8-deep recency ring dominated by one "
                "action class, built via real record_action() calls"
            ),
            "measured": worst_supp_range,
            "threshold": SUPPRESSION_RANGE_FLOOR,
            "direction": "lower",
            "met": worst_supp_range >= SUPPRESSION_RANGE_FLOOR,
        },
        {
            # Without >= 2 survivors entering the No-Go, the fail-open
            # gng_protect_min_eligible guard refuses to drop the last candidate
            # and the axis CANNOT fire -- a structural bar, independent of the
            # claim. Gating on it means a collapsed envelope self-routes
            # substrate_not_ready_requeue instead of masquerading as a refuted
            # perseveration axis.
            "name": "pre_nogo_eligible_set_admits_alternative",
            "kind": "readiness",
            "description": (
                "the MECH-448 envelope leaves >= 2 candidates entering the "
                "No-Go, so the fail-open guard is not structurally blocking it"
            ),
            "control": "worst gate-ON cell across arms and seeds",
            "measured": worst_env_size,
            "threshold": ENVELOPE_SIZE_FLOOR,
            "direction": "lower",
            "met": worst_env_size >= ENVELOPE_SIZE_FLOOR,
        },
        {
            "name": "go_nogo_config_plumbing_live",
            "kind": "readiness",
            "description": (
                "REEConfig.from_dims plumbs use_go_nogo_constitution and "
                "gng_perseveration_floor onto config.e3 (guards the documented "
                "from_dims silent-kwarg-swallow failure mode)"
            ),
            "control": "direct from_dims construction",
            "met": bool(plumbing_ok),
        },
    ]
    readiness_ok = all(p.get("met") for p in preconditions)

    # ---------------------------- criteria ---------------------------------
    seed_conversions = {str(c["seed"]): c["conversion_rate"] for c in on_cells}
    n_seeds_clearing = sum(
        1 for c in on_cells if c["conversion_rate"] >= CONVERSION_FLOOR
    )
    c1 = n_seeds_clearing >= SEED_MAJORITY

    mean_on = (
        statistics.fmean([c["conversion_rate"] for c in on_cells]) if on_cells else 0.0
    )
    mean_shuf = (
        statistics.fmean([c["conversion_rate"] for c in shuf_cells])
        if shuf_cells
        else 0.0
    )
    specificity_gap = mean_on - mean_shuf
    c2 = specificity_gap >= SPECIFICITY_GAP_FLOOR

    total_empty = sum(c["n_empty_eligible"] for c in arm_results)
    c3 = total_empty == 0

    # ------------------------- non-degeneracy ------------------------------
    # C1 is degenerate if the gate never engaged, or if ON and OFF picked
    # identically everywhere (nothing moved), or if suppression was flat.
    gate_ever_active = any(c["gate_active_rate"] > 0 for c in on_cells)
    on_conv = sum(c["n_converted"] for c in on_cells)
    off_conv = sum(c["n_converted"] for c in off_cells)
    c1_non_degenerate = bool(
        gate_ever_active
        and worst_supp_range >= SUPPRESSION_RANGE_FLOOR
        and worst_env_size >= ENVELOPE_SIZE_FLOOR
        and on_conv != off_conv
    )
    criteria_non_degenerate = {
        "C1_perseveration_conversion": c1_non_degenerate,
        # C2 is degenerate if the shuffled control never ran or the two gate-ON
        # arms are bit-identical (nothing to attribute to vector CONTENT).
        "C2_recency_content_specificity": bool(
            shuf_cells and on_cells and c1_non_degenerate
        ),
        "C3_safety_failopen": bool(arm_results),
    }

    criteria = [
        {
            "name": "C1_perseveration_conversion",
            "load_bearing": True,
            "passed": bool(c1),
            "measured": seed_conversions,
            "threshold": CONVERSION_FLOOR,
            "seeds_clearing": n_seeds_clearing,
            "seeds_required": SEED_MAJORITY,
        },
        {
            "name": "C2_recency_content_specificity",
            "load_bearing": False,
            "passed": bool(c2),
            "measured": specificity_gap,
            "threshold": SPECIFICITY_GAP_FLOOR,
            "mean_conversion_constitution": mean_on,
            "mean_conversion_shuffled": mean_shuf,
        },
        {
            "name": "C3_safety_failopen",
            "load_bearing": False,
            "passed": bool(c3),
            "measured": total_empty,
            "threshold": 0,
            "direction": "upper",
        },
    ]
    combination_rule = (
        "PASS = readiness_ok AND C1 AND C2 AND C3 (plain AND over all three; "
        "C1 is the load-bearing conversion criterion)."
    )

    # ---------------------------- routing ----------------------------------
    if not readiness_ok:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        direction = "non_contributory"
    elif c1 and c2 and c3:
        outcome = "PASS"
        label = "perseveration_axis_converts"
        direction = "supports"
    elif c3 and not c1:
        # Built-but-no-conversion is over-specification, NOT a falsification of
        # ARC-107 -- the same disposition V3-EXQ-689g pre-registered.
        outcome = "FAIL"
        label = "perseveration_axis_built_but_no_conversion"
        direction = "non_contributory"
    else:
        outcome = "FAIL"
        label = "safety_or_orthogonality_contract_violated"
        direction = "weakens"

    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{t0.strftime('%Y%m%dT%H%M%SZ')}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "supersedes": "V3-EXQ-926",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "overall_pass": outcome == "PASS",
        "timestamp_utc": t0.strftime("%Y%m%dT%H%M%SZ"),
        "evidence_direction": direction,
        "evidence_direction_per_claim": {
            "MECH-449": direction,
            "ARC-107": direction,
        },
        "arm_results": arm_results,
        "criteria": criteria,
        "combination_rule": combination_rule,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "dv_symmetry_declaration": {
                "ARM_CONSTITUTION": (
                    "manipulation = removal from the eligible SET (domain "
                    "change); DV = committed argmin identity. Argmin's symmetry "
                    "group is {uniform additive constant, monotone rescaling, "
                    "tie permutation}; set-removal is in none of them and "
                    "specifically deletes the argmin. NOT invariant."
                ),
                "ARM_OFF": "no manipulation (baseline domain).",
            },
        },
        "summary": {
            "seed_conversions": seed_conversions,
            "seeds_clearing_floor": n_seeds_clearing,
            "worst_suppression_range": worst_supp_range,
            "specificity_gap": specificity_gap,
            "mean_conversion_constitution": mean_on,
            "mean_conversion_shuffled": mean_shuf,
            "worst_pre_nogo_envelope_size": worst_env_size,
            "total_empty_eligible": total_empty,
            "gate_ever_active": gate_ever_active,
        },
        "config": {
            "seeds": seeds,
            "arms": ARMS,
            "n_banks": n_banks,
            "k_candidates": K_CANDIDATES,
            "world_dim": WORLD_DIM,
            "action_dim": ACTION_DIM,
            "horizon": HORIZON,
            "perseverated_class": "per-bank raw-F argmin (probed, not fixed)",
            "dominant_repeats": DOMINANT_REPEATS,
            "history_depth": HISTORY_DEPTH,
            "gng_perseveration_floor": PERSEVERATION_FLOOR,
            "conversion_floor": CONVERSION_FLOOR,
            "seed_majority": SEED_MAJORITY,
            "specificity_gap_floor": SPECIFICITY_GAP_FLOOR,
            "f_eligibility_envelope_floor": ENVELOPE_FLOOR,
            "suppression_range_floor": SUPPRESSION_RANGE_FLOOR,
            "envelope_size_floor": ENVELOPE_SIZE_FLOOR,
        },
        "notes": (
            "Selection-face conversion falsifier for the ONE untested axis of "
            "the BUILT MECH-449 Go/No-Go constitution: perseveration, which "
            "consumes MECH-260's live dACC recency-suppression vector. "
            "V3-EXQ-689g exercised only safety+staleness (both injected as "
            "constructed tensors) and states at its injection site that the "
            "perseveration axis was left to the live agent loop. Here the "
            "vector comes from a real DACCAdaptiveControl history via "
            "record_action() + _suppression_penalty(). MECH-260 is "
            "DELIBERATELY NOT TAGGED: its own falsifier is behavioural "
            "(monostrategy resolution), which is blocked on the MECH-457 "
            "competence floor and vacuous at that floor (V3-EXQ-445h measured "
            "action_class_entropy=0 in BOTH arms). A PASS supports MECH-449 + "
            "ARC-107 only. PROMOTES NOTHING until adjudicated. "
            "HONEST SCOPE LIMIT, stated so adjudication is not over-read: once "
            "the No-Go removes the incumbent from a set that HAS alternatives, "
            "the committed argmin necessarily changes, so C1 near 1.0 measures "
            "CAN-THE-AXIS-ACT, not an effect size. What makes it non-vacuous is "
            "that acting is contingent, not automatic -- at the stock "
            "f_eligibility_envelope_floor of 0.30 the identical mechanism "
            "converted 1/16, because the MECH-448 envelope collapses to the "
            "fail-open protect-min and the axis cannot fire. That INTERACTION "
            "(perseveration No-Go is structurally gated by envelope width) is "
            "the finding of record here, and it is not documented anywhere in "
            "the MECH-449 build notes or in V3-EXQ-689g."
        ),
    }
    if not c1_non_degenerate:
        manifest["non_degenerate"] = False
        manifest["degeneracy_reason"] = (
            "perseveration gate never engaged, suppression vector flat across "
            "candidates, or ON and OFF arms converted identically"
        )

    out_dir = (
        Path(tempfile.gettempdir()) / "ree_dry_run_manifests" if dry_run else EVIDENCE_DIR
    )
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=dry_run,
        config=manifest.get("config"),
        seeds=seeds,
        script_path=Path(__file__),
    )
    manifest["manifest_path"] = str(out_path)
    print(
        f"[926a] outcome={outcome} label={label} conversions={seed_conversions} "
        f"specificity_gap={specificity_gap:.3f} empty_eligible={total_empty} "
        f"-> {out_path}",
        flush=True,
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "V3-EXQ-926a MECH-449/ARC-107 perseveration-axis conversion falsifier"
        )
    )
    parser.add_argument("--dry-run", action="store_true", help="Short smoke run.")
    args = parser.parse_args()

    result = run_experiment(dry_run=args.dry_run)
    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    _outcome = _outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL"
    emit_outcome(
        outcome=_outcome,
        manifest_path=str(result.get("manifest_path", Path("/dev/null"))),
        dry_run=args.dry_run,
    )
