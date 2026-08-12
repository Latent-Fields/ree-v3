"""Shared behavioural-trajectory-metrics library: descriptive statistics over
a step sequence's positions and/or action labels.

WHY THIS EXISTS. The same trajectory-organisation statistics -- turning-angle
entropy, tortuosity, straight-run length, hazard-conditioned turning -- were
independently reinvented at least twice in one week: first as an
uncommitted, one-off script (`sleep_boundary_trajectory_reanalysis.py`,
described but never committed, per
`sleep_transition_investigation_906_lineage_2026-08-10.md` Section 3), then
reimplemented from that document's own prose method description -- not
copied -- as `_trajectory_organization_stats()` in
`experiments/v3_exq_913_developmental_ecology_fishtank.py:387-472` (that
function's own docstring says so explicitly). A THIRD feature family
(action-label reversal-rate / run-length / repeat-rate) was computed a third
time, ad hoc and never committed, in the same document's 2026-08-11 addendum
(Section 13b). See
`REE_assembly/evidence/planning/behavioural_trajectory_metrics_library_scoping_2026-08-11.md`
for the full scoping analysis (three independent reinventions, one already
definitionally drifted from its own prior art since it was rebuilt from
prose rather than shared code) and the recommendation this module
implements: extract the spatial-geometry family from the one committed
implementation VERBATIM (see `tests/contracts/test_trajectory_metrics.py`'s
bit-identical regression test against that source function), and formalise
the action-label family that was previously only prose-specified.

WHAT THIS DELIBERATELY DOES NOT COVER. The scoping proposal's Section 3b
item 4 explicitly excludes the ECOLOGICAL/EXCURSION feature family
(`in_reef` excursion counts, harm-rate-in-vs-out-of-shelter ratios, from
`reef_ecology_strategy_affective_occupancy_review_2026-08-10.md`) -- it
needs an environment-specific ground-truth flag most environments do not
have, and is an episode-level summary over many transitions, not a
window-level trajectory-shape statistic like the two families here. It also
does NOT provide a discriminability/classifier-test harness (the "umpire"
methodology of
`thought_intake_2026-08-11_behavioural_diversity_umpire.md` / `Q-092`) --
that is a statistical PROCEDURE consuming feature vectors, deliberately
kept at a separate abstraction layer from this pure feature-COMPUTATION
module (scoping proposal Section 3a). Per that same proposal (Section 4),
this module does NOT wire into `precondition_gate.py` or any
`/queue-experiment` acceptance-check pattern -- `degeneracy_flags()` below
follows that module's floor/ceiling naming convention so a future formal
integration is a rename-free wrap, but no such integration is built here.

TWO FEATURE FAMILIES, KEPT SEPARATE AND INDEPENDENTLY CALLABLE.

  `spatial_trajectory_stats()` -- geometry of the POSITION sequence: turning
  angle (mean, entropy), straight-run length, tortuosity, hazard-conditioned
  turning. Ported verbatim (same algorithm, same default constant) from
  `_trajectory_organization_stats()` above.

  `action_sequence_stats()` -- structure of the ACTION-LABEL sequence
  itself: reversal rate, run length, repeat rate. Kept distinct from the
  spatial family on purpose -- a 90-degree turn and a full reversal are the
  same "one turn" to the spatial family but very different events in
  policy-output space, per the finding that motivated adding this family
  (sleep_transition_investigation Section 13b: "spatial turning/tortuosity
  ... conflate a 90-degree turn with a full reversal").

THE WORLD-RULE-SHIFT CAVEAT (load-bearing -- read before using
`action_sequence_stats` output as a claim about spatial behaviour). Some
Fishtank configs periodically permute the live action-ID -> spatial-
direction map (`world_rule_shift_enabled`). A reversal/run-length statistic
computed on action IDs therefore measures POLICY-OUTPUT-SEQUENCE structure
(does REE keep re-selecting the same abstract action), not necessarily
spatial backtracking, once a window straddles a rule-shift boundary.
`action_sequence_stats` accepts an optional `rule_shift_boundaries` +
`window_start_index` and reports `spans_rule_shift` explicitly: `True` /
`False` when boundary information was supplied, `None` -- not `False` --
when it was not, so a caller can never read "no boundary given" as
"confirmed stable." This is a deliberate fail-informative design, not an
oversight: silently defaulting an unknown to `False` would let the two
interpretations be conflated exactly the way the un-formalised ad hoc
script risked doing.

CANONICAL ACTION INVERSE PAIRING. Verified against
`ree_core/environment/causal_grid_world.py` `CausalGridWorld.ACTIONS`:
`{0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}` -- 0 and 1 are
exact opposites, 2 and 3 are exact opposites, 4 (stay) has no inverse.
`CONSUME_ACTION = 5` (mech457 consummatory-act, when enabled) is likewise
given no inverse by the default pairing -- it is explicitly "NOT a member
of ACTIONS / _action_map ... immune to the world-rule-shift permutation"
per that module's own comment, so treating it as reversal-incapable is
correct, not an omission. `DEFAULT_ACTION_INVERSE_PAIRS` is overridable for
a different action space.

Input shape: a list of per-step dicts, matching the shape already converged
on by `event_window_browser.py`, the 906-lineage drivers, and this module's
own source function -- `{"pos": [x, y], "action": int, ...}` per step. Only
the keys each function actually reads are required.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Same default as `_trajectory_organization_stats`'s module-level
# HAZARD_NEAR_RADIUS in v3_exq_913_developmental_ecology_fishtank.py -- kept
# as an overridable parameter default here (scoping proposal Section 5 item
# 1) rather than a bare module constant, so a caller with a different notion
# of "near" does not need to monkeypatch a global.
HAZARD_NEAR_RADIUS = 3

# Canonical inverse-action pairing for CausalGridWorld.ACTIONS -- see module
# docstring "CANONICAL ACTION INVERSE PAIRING" for the source citation.
DEFAULT_ACTION_INVERSE_PAIRS: Dict[int, int] = {0: 1, 1: 0, 2: 3, 3: 2}


def spatial_trajectory_stats(
    steps: List[Dict[str, Any]],
    hazard_positions: Optional[Sequence[Tuple[int, int]]] = None,
    window: Optional[int] = None,
    hazard_near_radius: int = HAZARD_NEAR_RADIUS,
) -> Dict[str, Any]:
    """Turning-angle distribution, straight-run length, tortuosity,
    hazard-conditioned turning -- computed from logged POSITIONS.

    Verbatim port of `_trajectory_organization_stats()` in
    `v3_exq_913_developmental_ecology_fishtank.py:387-472` (same algorithm,
    same output keys, `HAZARD_NEAR_RADIUS` made an explicit, overridable
    parameter with the same default value).
    `tests/contracts/test_trajectory_metrics.py` asserts this reproduces
    that function's output bit-for-bit on shared fixtures, including its
    degenerate branches -- read that test before changing anything below
    this line.

    `steps` is the already-completed per-step log for one segment (or a
    slice of it); `hazard_positions` is a list of `[x, y]` ground-truth
    hazard cells (omit or pass `None`/`[]` to skip the hazard-conditioned
    split); `window` (if given) uses only the first `window` entries.
    """
    seq = steps[:window] if window is not None else steps
    n = len(seq)
    out: Dict[str, Any] = {"n_steps": n}
    if n < 2:
        return out
    positions = [tuple(s["pos"]) for s in seq]
    deltas = [(positions[i + 1][0] - positions[i][0], positions[i + 1][1] - positions[i][1])
              for i in range(n - 1)]
    headings = []
    for dx, dy in deltas:
        if dx == 0 and dy == 0:
            headings.append(None)
        else:
            headings.append(float(np.arctan2(dy, dx)))

    # Turning angle: absolute angular change between consecutive non-null headings.
    turning_angles: List[float] = []
    turning_near_hazard: List[float] = []
    turning_far_hazard: List[float] = []
    prev_heading = None
    for i, h in enumerate(headings):
        if h is not None and prev_heading is not None:
            diff = abs(h - prev_heading)
            if diff > np.pi:
                diff = 2 * np.pi - diff
            turning_angles.append(float(diff))
            if hazard_positions:
                px, py = positions[i]
                nearest = min(abs(px - hx) + abs(py - hy) for hx, hy in hazard_positions)
                if nearest <= hazard_near_radius:
                    turning_near_hazard.append(float(diff))
                else:
                    turning_far_hazard.append(float(diff))
        if h is not None:
            prev_heading = h

    # Straight-run length: consecutive steps sharing the same non-null heading.
    straight_runs: List[int] = []
    run_len = 0
    run_heading = None
    for h in headings:
        if h is None:
            continue
        if run_heading is not None and abs(h - run_heading) < 1e-6:
            run_len += 1
        else:
            if run_len > 0:
                straight_runs.append(run_len)
            run_len = 1
            run_heading = h
    if run_len > 0:
        straight_runs.append(run_len)

    path_length = sum(abs(dx) + abs(dy) for dx, dy in deltas)
    net_displacement = abs(positions[-1][0] - positions[0][0]) + abs(positions[-1][1] - positions[0][1])
    tortuosity = (float(path_length) / net_displacement) if net_displacement > 0 else None

    out.update({
        "turning_angle_mean": float(np.mean(turning_angles)) if turning_angles else None,
        "turning_angle_entropy_bits": (
            float(-np.sum((h := np.histogram(turning_angles, bins=8, range=(0, np.pi))[0]
                           / len(turning_angles)) * np.log2(h + 1e-12)))
            if turning_angles else None
        ),
        "mean_straight_run_length": float(np.mean(straight_runs)) if straight_runs else None,
        "max_straight_run_length": int(max(straight_runs)) if straight_runs else None,
        "tortuosity": tortuosity,
        "path_length": int(path_length),
        "net_displacement": int(net_displacement),
        "turning_near_hazard_mean": float(np.mean(turning_near_hazard)) if turning_near_hazard else None,
        "turning_far_hazard_mean": float(np.mean(turning_far_hazard)) if turning_far_hazard else None,
        "n_turning_near_hazard": len(turning_near_hazard),
        "n_turning_far_hazard": len(turning_far_hazard),
    })
    return out


def action_sequence_stats(
    actions: List[int],
    inverse_pairs: Optional[Dict[int, int]] = None,
    rule_shift_boundaries: Optional[Sequence[int]] = None,
    window_start_index: int = 0,
    window: Optional[int] = None,
) -> Dict[str, Any]:
    """Reversal rate, run length, repeat rate -- computed from the
    ACTION-ID sequence itself, not positions. Formalises the ad hoc metric
    set from `sleep_transition_investigation_906_lineage_2026-08-10.md`
    Section 13b, which found spatial turning/tortuosity "conflate a
    90-degree turn with a full reversal" and added this family specifically
    to separate the two.

    See module docstring "THE WORLD-RULE-SHIFT CAVEAT" before treating a
    non-`None` `spans_rule_shift` as license to read this as spatial
    structure.

    `actions` is the already-completed per-step action-ID sequence (or a
    slice of it); `inverse_pairs` defaults to `DEFAULT_ACTION_INVERSE_PAIRS`
    (see module docstring "CANONICAL ACTION INVERSE PAIRING"); `window` (if
    given) uses only the first `window` entries; `rule_shift_boundaries`
    are GLOBAL step indices (in the same numbering as `window_start_index`)
    where the action->direction map changed -- omit to leave
    `spans_rule_shift` explicitly `None` rather than a silently-assumed
    `False`.
    """
    pairs = inverse_pairs if inverse_pairs is not None else DEFAULT_ACTION_INVERSE_PAIRS
    seq = actions[:window] if window is not None else actions
    n = len(seq)
    out: Dict[str, Any] = {"n_actions": n}
    if n < 2:
        out.update({
            "reversal_rate": None,
            "reversal_count": 0,
            "repeat_rate": None,
            "repeat_count": 0,
            "mean_run_length": None,
            "max_run_length": None,
            "n_transitions": 0,
            "spans_rule_shift": None,
        })
        return out

    n_transitions = n - 1
    reversal_count = sum(1 for i in range(n_transitions) if pairs.get(seq[i]) == seq[i + 1])
    repeat_count = sum(1 for i in range(n_transitions) if seq[i] == seq[i + 1])

    run_lengths: List[int] = []
    run_len = 1
    for i in range(1, n):
        if seq[i] == seq[i - 1]:
            run_len += 1
        else:
            run_lengths.append(run_len)
            run_len = 1
    run_lengths.append(run_len)

    if rule_shift_boundaries:
        window_end_index = window_start_index + n  # exclusive
        spans_rule_shift = any(
            window_start_index <= b < window_end_index for b in rule_shift_boundaries
        )
    else:
        spans_rule_shift = None

    out.update({
        "reversal_rate": reversal_count / n_transitions,
        "reversal_count": reversal_count,
        "repeat_rate": repeat_count / n_transitions,
        "repeat_count": repeat_count,
        "mean_run_length": float(np.mean(run_lengths)),
        "max_run_length": int(max(run_lengths)),
        "n_transitions": n_transitions,
        "spans_rule_shift": spans_rule_shift,
    })
    return out


def degeneracy_flags(
    steps: List[Dict[str, Any]],
    window: Optional[int] = None,
    static_frac_ceiling: float = 0.9,
    min_turning_samples: int = 5,
) -> Dict[str, Any]:
    """Non-degeneracy signal for a trajectory window, using the two failure
    modes actually found in this codebase's own prior-art (scoping proposal
    Section 3c), not hypothesised ones: (a) `net_displacement == 0`
    (already guarded inside `spatial_trajectory_stats`'s `tortuosity`,
    surfaced here as an explicit flag rather than inferred from a `None`);
    (b) a near-static window (`seed1/no_sleep/seg9` in V3-EXQ-913's own
    data: the agent moved 3 ticks then sat motionless for the remaining
    96 -- not caught by any existing check at the time, per
    `sleep_transition_investigation_906_lineage_2026-08-10.md` Section 13b).

    Uses `precondition_gate.py`'s floor/ceiling naming convention
    (`*_exceeds_ceiling`, `*_below_floor`) so a future formal-experiment
    integration (scoping proposal Section 4b) does not need to rename
    anything -- this function does not itself depend on
    `precondition_gate`, it only follows its naming so the eventual wiring
    is a rename-free `PreconditionSpec` wrap.

    Recomputes turning-angle count directly from `steps` rather than
    depending on `spatial_trajectory_stats`'s return shape, so this
    function has exactly one dependency (`steps`), not two.
    """
    seq = steps[:window] if window is not None else steps
    n = len(seq)
    out: Dict[str, Any] = {"n_steps": n}
    if n < 2:
        out.update({
            "static_frac": None,
            "static_frac_exceeds_ceiling": None,
            "n_turning_samples": 0,
            "turning_samples_below_floor": True,
            "net_displacement_zero": None,
        })
        return out

    positions = [tuple(s["pos"]) for s in seq]
    static_count = sum(1 for i in range(1, n) if positions[i] == positions[i - 1])
    static_frac = static_count / (n - 1)
    net_displacement = abs(positions[-1][0] - positions[0][0]) + abs(positions[-1][1] - positions[0][1])

    deltas = [(positions[i + 1][0] - positions[i][0], positions[i + 1][1] - positions[i][1])
              for i in range(n - 1)]
    headings = [None if (dx == 0 and dy == 0) else float(np.arctan2(dy, dx)) for dx, dy in deltas]
    n_turning = 0
    prev_heading = None
    for h in headings:
        if h is not None and prev_heading is not None:
            n_turning += 1
        if h is not None:
            prev_heading = h

    out.update({
        "static_frac": static_frac,
        "static_frac_exceeds_ceiling": static_frac > static_frac_ceiling,
        "n_turning_samples": n_turning,
        "turning_samples_below_floor": n_turning < min_turning_samples,
        "net_displacement_zero": net_displacement == 0,
    })
    return out
