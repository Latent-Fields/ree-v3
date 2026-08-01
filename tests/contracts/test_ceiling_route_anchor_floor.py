"""Contracts for the ceiling-below-random-anchor standing lint + its runtime guard.

WHAT WENT WRONG, AND WHAT THESE CONTRACTS DEFEND. The competence-objective autopsy
(734/737b/742a, 2026-07-22, user-adjudicated confirmed) found ceiling-class verdicts
(substrate_ceiling / learner_or_observability_ceiling) being drawn on runs whose learners
scored BELOW their own declared random_walk anchor -- ppo_ree_latent 0.233 vs random 0.933 at
D3. Under a genuine ceiling a learner asymptotes toward its random anchor from below; a learner
WORSE than random on an oracle-achievable env is optimising a different objective, not hitting a
capacity limit (the survival-vs-forage inversion: PPO survived 175.0 steps vs the oracle's 20.4
while foraging 17x less). The below-random score was in every manifest and consumed by nothing.

Two halves, both pinned here:
  * RUNTIME (experiments/_lib/anchor_floor_guard.py) -- refuse_ceiling_below_random downgrades a
    ceiling label resting on a sub-random learner to substrate_not_ready_requeue and records the
    refuting numbers. Pinned by G1-G6 with the autopsy's own numbers, and by X1 as wired into
    the v3_exq_734 reference consumer.
  * STANDING (validate_experiments.ceiling_route_anchor_floor_lint) -- fires when a
    diagnostic/baseline script self-routes to a ceiling label with a random_walk floor but never
    wires the guard. Pinned by L1-L5, incl. that it stays SILENT on the current corpus (precise:
    keys on an EMITTED ceiling route, not a prose mention) and fires on the unguarded shape.

ASCII-only (repo rule). Time-independent: pure-function assertions + AST scans of string
fixtures; no run data, no clock, no network.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import pytest  # noqa: E402

import validate_experiments as V  # noqa: E402
from experiments._lib.anchor_floor_guard import (  # noqa: E402
    CEILING_ROUTE_LABELS,
    REQUEUE_LABEL,
    CeilingBelowRandomAnchorError,
    ceiling_route_refusal,
    is_ceiling_route,
    refuse_ceiling_below_random,
    sub_random_learner_arms,
)

EXPERIMENTS_DIR = REPO_ROOT / "experiments"

# The autopsy's own D3 numbers (sec 1c): both learners below the random_walk anchor 0.933.
_AUTOPSY_LEARNERS = {"ppo_ree_latent": 0.233, "ppo_raw_obs": 0.567}
_AUTOPSY_RANDOM = 0.933


# ----------------------------------------------------------------------------- runtime guard
def test_g1_ceiling_route_below_random_is_refused():
    """The confirmed case: a ceiling label with learners below random -> downgrade + record."""
    label, rec = refuse_ceiling_below_random(
        "learner_or_observability_ceiling", _AUTOPSY_LEARNERS, _AUTOPSY_RANDOM, rung="D3")
    assert label == REQUEUE_LABEL
    assert rec is not None and rec["refused"] is True
    assert rec["original_label"] == "learner_or_observability_ceiling"
    assert rec["downgraded_to"] == REQUEUE_LABEL
    got = {d["arm"] for d in rec["sub_random_arms"]}
    assert got == set(_AUTOPSY_LEARNERS), "both sub-random learners must be named"


def test_g2_ceiling_route_stands_when_every_learner_at_or_above_random():
    """A ceiling verdict with all learners at/above the random anchor is NOT refused: that is a
    verdict the below-random lint has nothing to say about."""
    label, rec = refuse_ceiling_below_random(
        "ree_substrate_ceiling", {"ree": 1.5, "ppo": 2.0}, _AUTOPSY_RANDOM)
    assert label == "ree_substrate_ceiling" and rec is None


def test_g3_equal_to_anchor_is_not_below():
    """Strict `<`: a learner exactly AT the anchor is not below it (asymptote-from-below is the
    ceiling signature; equality is the boundary, not a violation)."""
    label, rec = refuse_ceiling_below_random(
        "substrate_ceiling", {"ree": _AUTOPSY_RANDOM}, _AUTOPSY_RANDOM)
    assert label == "substrate_ceiling" and rec is None


@pytest.mark.parametrize("label", ["substrate_not_ready_requeue", "env_difficulty_recoverable",
                                   "does_not_support", None])
def test_g4_non_ceiling_labels_never_refused(label):
    """Only ceiling-class routes are governed; a requeue/recoverable label with a below-random
    learner is left untouched (a below-floor requeue is already the safe non-verdict)."""
    out_label, rec = refuse_ceiling_below_random(label, {"ree": 0.01}, _AUTOPSY_RANDOM)
    assert out_label == label and rec is None


def test_g5_none_scores_and_none_anchor_are_skipped_not_refused():
    """A refused/unmeasured cell (None) can neither support nor refute a verdict, and a None
    anchor cannot be compared against -- neither triggers a refusal."""
    # one learner unmeasured (None), the other above the anchor -> stands
    l1, r1 = refuse_ceiling_below_random("learner_ceiling", {"a": None, "b": 2.0}, _AUTOPSY_RANDOM)
    assert l1 == "learner_ceiling" and r1 is None
    # anchor None -> cannot compare -> stands even with a tiny learner score
    l2, r2 = refuse_ceiling_below_random("substrate_ceiling", {"a": 0.01}, None)
    assert l2 == "substrate_ceiling" and r2 is None
    # empty learner set -> nothing measurable -> stands
    assert sub_random_learner_arms({}, _AUTOPSY_RANDOM) == []


def test_g6_strict_mode_raises_with_the_offending_numbers():
    """strict=True is the hard-stop path for a caller that wants an exception, and the message
    must carry the anchor and the offending arm so a reader can re-derive the refusal."""
    with pytest.raises(CeilingBelowRandomAnchorError) as exc:
        refuse_ceiling_below_random(
            "substrate_ceiling", {"ppo_ree_latent": 0.233}, _AUTOPSY_RANDOM, strict=True, rung="D3")
    msg = str(exc.value)
    assert "0.933" in msg and "ppo_ree_latent" in msg


def test_g7_label_set_and_membership_helpers_agree():
    assert is_ceiling_route("learner_or_observability_ceiling") is True
    assert is_ceiling_route("substrate_not_ready_requeue") is False
    assert "ree_substrate_ceiling" in CEILING_ROUTE_LABELS
    # a record is only produced for a genuine ceiling+below-random pairing
    assert ceiling_route_refusal("env_difficulty_recoverable", {"a": 0.0}, 1.0) is None


# ----------------------------------------------------- reference consumer + the standing lint
def test_x1_v3_exq_734_wires_the_runtime_guard():
    """v3_exq_734 is the reference consumer: it MUST import and call refuse_ceiling_below_random,
    else the autopsy's canonical ceiling-routing driver stops consuming its own numbers."""
    src = (EXPERIMENTS_DIR / "v3_exq_734_env_difficulty_competence_recovery_sweep.py").read_text()
    assert "refuse_ceiling_below_random" in src
    assert "anchor_floor_guard" in src


def test_x2_734_scope_label_is_gating_not_detection_only():
    """Item-1 finish: 734 genuinely refuses the arm, so its manifest scope must not still
    self-describe as detection_only."""
    src = (EXPERIMENTS_DIR / "v3_exq_734_env_difficulty_competence_recovery_sweep.py").read_text()
    assert '"scope": "gating"' in src
    assert '"scope": "detection_only"' not in src


def _lint(tmp_path, body: str):
    f = tmp_path / "drv.py"
    f.write_text(body, encoding="utf-8")
    return V.ceiling_route_anchor_floor_lint(f)


_UNGUARDED = '''
EXPERIMENT_PURPOSE = "diagnostic"

def run(arm_summaries):
    build_report(arm_summaries, floor="random_walk", ceiling="greedy_oracle")
    if cond:
        label = "learner_or_observability_ceiling"
    else:
        label = "substrate_not_ready_requeue"
    return label

if __name__ == "__main__":
    run({})
'''


def test_l1_lint_fires_on_emitted_ceiling_route_without_guard(tmp_path):
    warn = _lint(tmp_path, _UNGUARDED)
    assert warn is not None
    assert "learner_or_observability_ceiling" in warn
    assert "refuse_ceiling_below_random" in warn


def test_l2_lint_silent_when_guard_is_wired(tmp_path):
    guarded = _UNGUARDED.replace(
        "def run(arm_summaries):",
        "from experiments._lib.anchor_floor_guard import refuse_ceiling_below_random\n"
        "def run(arm_summaries):",
        1,
    )
    assert _lint(tmp_path, guarded) is None


def test_l3_lint_silent_on_prose_only_mention(tmp_path):
    """The label appears only inside a description string, and the EMITTED route is a requeue --
    the precision that keeps the lint off the ~30 behavioural drivers that mention ceilings."""
    prose = '''
EXPERIMENT_PURPOSE = "diagnostic"

def run(arm_summaries):
    build_report(arm_summaries, floor="random_walk", ceiling="greedy_oracle")
    grid = {"env_ok": "if PPO clears but ree cannot -> a substrate_ceiling reading"}
    label = "substrate_not_ready_requeue"
    return label

if __name__ == "__main__":
    run({})
'''
    assert _lint(tmp_path, prose) is None


def test_l4_lint_silent_without_random_walk_floor(tmp_path):
    """No random_walk floor anchor -> nothing to compare a learner against -> no obligation."""
    no_floor = _UNGUARDED.replace('floor="random_walk"', 'floor="some_other_baseline"')
    assert _lint(tmp_path, no_floor) is None


def test_l5_exempt_marker_silences(tmp_path):
    exempt = _UNGUARDED.replace(
        'EXPERIMENT_PURPOSE = "diagnostic"',
        'EXPERIMENT_PURPOSE = "diagnostic"\n'
        'CEILING_ANCHOR_FLOOR_EXEMPT = "readout-side control unaffected by the floor"',
        1,
    )
    assert _lint(tmp_path, exempt) is None


def test_l6_lint_silent_on_the_current_corpus():
    """The standing lint must add ZERO noise today: 734 wires the guard, and no other corpus
    script emits a ceiling route with a random_walk floor. A future regression is what it is
    for. (If this fails, a driver emitted a ceiling route without the guard -- wire it or
    CEILING_ANCHOR_FLOOR_EXEMPT it; do not weaken this test.)"""
    offenders = []
    for f in sorted(EXPERIMENTS_DIR.glob("v3_exq_*.py")):
        if V.ceiling_route_anchor_floor_lint(f):
            offenders.append(f.name)
    assert offenders == [], f"unexpected ceiling-route-anchor-floor warnings: {offenders}"


# --------------------------------------------------------------------------------- wiring
def test_l7_lint_is_registered_in_check_names():
    """An unregistered lint is dead code: main() gates every lint on `selected`."""
    assert "ceiling_route_anchor_floor" in V.CHECK_NAMES


def test_l8_lint_is_reachable_through_main(tmp_path, capsys):
    """Executes the real CLI over a synthetic carrier. Registration in CHECK_NAMES is necessary
    but not sufficient -- the main-loop branch, the summary count, and the report block are
    separate edits and any one can be missed. Also pins WARN-only (rc==0 even under --paths)."""
    p = tmp_path / "drv.py"
    p.write_text(_UNGUARDED, encoding="utf-8")
    argv = sys.argv
    sys.argv = ["validate_experiments.py", "--checks", "ceiling_route_anchor_floor",
                "--paths", str(p)]
    try:
        rc = V.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "CEILING-ROUTE-ANCHOR-FLOOR WARNINGS" in out, out
    assert rc == 0, "WARN-only: must never harden, even under --paths"
