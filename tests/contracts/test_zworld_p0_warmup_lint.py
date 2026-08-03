"""Contracts for the `zworld_p0_warmup` lint (SD-070, confirmed twice).

Surfaces under test:
  (1) validate_experiments.zworld_p0_warmup_lint -- flags a driver that calls
      `experiments._lib.allon_training._train_all_on_agent` but never passes
      `zworld_p0_episodes` with a plausibly nonzero value anywhere in the file, while its
      acceptance criteria plausibly read a z_world-dependent competence/survival/foraging
      metric.
  (2) validate_experiments.py --checks zworld_p0_warmup -- the selector, and the invariant
      that this gate is WARN-ONLY IN BOTH MODES (never hardens under --paths, never affects
      the exit code even under --strict).
  (3) The corpus fire count, pinned against the two REAL confirmed carriers, with a
      non-vacuity guard that names them rather than trusting the number alone.

WHY THIS GATE EXISTS. `_train_all_on_agent`'s P0a stage -- the SD-070 z_world-encoder
warmup -- is opt-in via `zworld_p0_episodes: int = 0`. With the default, `run_zworld_p0`
never runs, `split_encoder.world_encoder` is never stepped, and z_world stays a frozen
random projection for the whole run. No error, no warning: the agent trains and every
downstream competence/survival/foraging metric is computed exactly as if the encoder had
been trained, just against noise.

CONFIRMED TWICE, independently, on two different drivers, both still live in this corpus:
  V3-EXQ-728 (original run) -- "3/3 seeds failed"; fixed by adding zworld_p0_episodes=60.
    The CURRENT `v3_exq_728_trained_allon_capability_point.py` on disk is the POST-FIX
    version (threads `zworld_p0_episodes` through by name) and must NOT fire -- it is a
    negative control, not (any longer) a positive one.
  V3-EXQ-875 (MECH-471, 2026-08-03) -- all three `_train_all_on_agent` call sites omitted
    the kwarg; a ~20.5h wall-clock run self-routed `substrate_not_ready_requeue` with zero
    usable evidence. `REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-875_2026-08-03.md`,
    "Root cause". The corrected re-run, `v3_exq_875a_mech471_competence_provenance.py`, adds
    `zworld_p0_episodes=60` to its first (acquisition) call and is the POSITIVE control's own
    paired negative control -- same driver family, same acceptance criteria, fixed shape.

A THIRD corpus file, `v3_exq_882_mech472_context_memorization_generalization.py`, independently
carries the identical defect (a single `_train_all_on_agent` call, no `zworld_p0_episodes`
anywhere, criteria built on `survival_horizon`) -- found by this gate, not the autopsy that
motivated it. It is left in the corpus (not fixed as part of this change; the task here is the
lint, not a driver retro-fix) and is exactly what the pinned corpus count below is naming.

SCOPE, and why file-wide rather than per-call-site or per-training-arm. The precise rule is
"the FIRST (acquisition) call per shared agent needs the warmup; a later call training the
SAME agent on a second competence correctly passes zworld_p0_episodes=0" (see 875a's own
`_run_cell`: first call `zworld_p0_episodes=zworld_p0`, the next two omit it entirely and are
correct to do so). Distinguishing "first call for this agent" from "later call for the same
agent" statically needs data-flow tracking of the `agent` receiver across calls, which this
scan does not attempt. Instead it fires only when NO call site in the file passes
`zworld_p0_episodes` positively ANYWHERE -- mirroring `_dacc_weight_set_positively_anywhere`'s
file-wide escape. This is deliberately UNDER-firing: a driver that wires the kwarg into a
non-first call only (an authoring mistake in its own right) would incorrectly escape here.
Accepted, like every other WARN-only gate in this family: a false WARN costs a reader a
minute; both confirmed incidents (kwarg absent from EVERY call site) are caught either way.

SCOPE, second axis: only fires when the file plausibly reports a z_world-dependent
competence/survival/foraging metric (a lowercase substring match on "survival", "foraging",
or "competence" -- capability_eval's own DV vocabulary plus the word stems both confirmed
carriers' criteria are built from). This keeps the gate from firing on a hypothetical caller
that trains the agent for a purpose the encoder state does not affect.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import validate_experiments as V  # noqa: E402

POSITIVE_SPECIMEN = "v3_exq_875_mech471_competence_provenance.py"
NEGATIVE_SPECIMEN = "v3_exq_875a_mech471_competence_provenance.py"
SECOND_POSITIVE_SPECIMEN = "v3_exq_882_mech472_context_memorization_generalization.py"


def _run(*args):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "validate_experiments.py"), *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT))


def _lint_src(src: str):
    """Lint a synthetic script written into experiments/ (so relative scoping holds)."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(textwrap.dedent(src))
        name = f.name
    try:
        return V.zworld_p0_warmup_lint(Path(name))
    finally:
        Path(name).unlink()


# The target shape, reduced to its skeleton from the 875 lineage: a driver that calls the
# shared training helper without ever threading zworld_p0_episodes, and reports a
# survival-competence metric as its DV.
_DEFECTIVE = '''
"""A driver that trains a survival competence without the SD-070 z_world warmup."""
from experiments._lib.allon_training import _train_all_on_agent


def _run_cell(agent, env, seed):
    result = _train_all_on_agent(
        agent, env, seed, p0_episodes=50, p1_episodes=90,
        steps_per_episode=150, rung_id="acquire", total_denominator=140,
    )
    return {"survival_horizon": result["survival_horizon"]}
'''


# ---- (1) the defect fires -----------------------------------------------------------

def test_zpw_fires_on_the_canonical_shape():
    assert _lint_src(_DEFECTIVE) is not None


def test_zpw_fires_via_the_attribute_call_form():
    """The real corpus imports the module and calls `x734._train_all_on_agent(...)` --
    `_call_name` must resolve the attribute form to the same bare name."""
    src = _DEFECTIVE.replace(
        "from experiments._lib.allon_training import _train_all_on_agent",
        "from experiments._lib import allon_training as x734")
    src = src.replace("result = _train_all_on_agent(", "result = x734._train_all_on_agent(")
    assert _lint_src(src) is not None


def test_zpw_names_the_call_sites_and_the_matched_tokens():
    msg = _lint_src(_DEFECTIVE)
    assert msg is not None
    assert "_train_all_on_agent" in msg
    assert "zworld_p0_episodes" in msg
    assert "survival" in msg
    # and it must point at the fix, not merely the defect
    assert "zworld_p0_episodes=<n>" in msg
    assert "V3-EXQ-728" in msg and "V3-EXQ-875" in msg


def test_zpw_message_is_ascii_only():
    """CLAUDE.md: anything reaching stdout is cp1252-safe."""
    msg = _lint_src(_DEFECTIVE)
    assert msg is not None
    assert all(ord(c) < 128 for c in msg), "non-ASCII in a printed lint message"


def test_zpw_fires_on_multiple_call_sites_all_omitting_the_kwarg():
    """The real 875 shape: three call sites, none of them wired."""
    src = _DEFECTIVE.replace(
        '    return {"survival_horizon": result["survival_horizon"]}',
        "    _train_all_on_agent(\n"
        "        agent, env, seed, p0_episodes=0, p1_episodes=90,\n"
        '        steps_per_episode=150, rung_id="second", total_denominator=140,\n'
        "    )\n"
        '    return {"survival_horizon": result["survival_horizon"]}')
    msg = _lint_src(src)
    assert msg is not None
    assert msg.count("line ") >= 2


# ---- (2) THE ESCAPE: a positive kwarg anywhere in the file silences it ---------------

def test_zpw_a_literal_positive_kwarg_on_the_call_is_silent():
    src = _DEFECTIVE.replace(
        "        steps_per_episode=150, rung_id=\"acquire\", total_denominator=140,\n",
        "        steps_per_episode=150, rung_id=\"acquire\", total_denominator=140,\n"
        "        zworld_p0_episodes=60, zworld_p0_env=env,\n")
    assert _lint_src(src) is None


def test_zpw_a_name_forwarded_kwarg_is_silent():
    """728/728b/734's own wrapper shape: the driver defines ITS OWN `zworld_p0_episodes`
    parameter and forwards it by name. A Name value is not statically known, so this must
    be treated as set-and-positive (the false-negative-safe direction) -- exactly the
    `_dacc_weight_is_positive` precedent for the analogous case."""
    src = _DEFECTIVE.replace(
        "def _run_cell(agent, env, seed):",
        "def _run_cell(agent, env, seed, zworld_p0_episodes=0):")
    src = src.replace(
        "        steps_per_episode=150, rung_id=\"acquire\", total_denominator=140,\n",
        "        steps_per_episode=150, rung_id=\"acquire\", total_denominator=140,\n"
        "        zworld_p0_episodes=zworld_p0_episodes,\n")
    assert _lint_src(src) is None


def test_zpw_an_explicit_literal_zero_kwarg_still_fires():
    """`zworld_p0_episodes=0` written out is exactly the inert default, spelled out --
    it must NOT clear the finding."""
    src = _DEFECTIVE.replace(
        "        steps_per_episode=150, rung_id=\"acquire\", total_denominator=140,\n",
        "        steps_per_episode=150, rung_id=\"acquire\", total_denominator=140,\n"
        "        zworld_p0_episodes=0,\n")
    assert _lint_src(src) is not None


def test_zpw_a_positive_kwarg_on_a_LATER_call_still_silences_the_earlier_one():
    """Documents the deliberate under-fire bias: the file-wide escape does not
    distinguish which call site carries the positive kwarg."""
    src = _DEFECTIVE.replace(
        '    return {"survival_horizon": result["survival_horizon"]}',
        "    _train_all_on_agent(\n"
        "        agent, env, seed, p0_episodes=0, p1_episodes=90,\n"
        '        steps_per_episode=150, rung_id="second", total_denominator=140,\n'
        "        zworld_p0_episodes=60, zworld_p0_env=env,\n"
        "    )\n"
        '    return {"survival_horizon": result["survival_horizon"]}')
    assert _lint_src(src) is None


# ---- (3) the remaining preconditions --------------------------------------------------

def test_zpw_no_train_all_on_agent_call_is_silent():
    src = _DEFECTIVE.replace(
        "from experiments._lib.allon_training import _train_all_on_agent\n", "")
    src = src.replace(
        "    result = _train_all_on_agent(\n"
        "        agent, env, seed, p0_episodes=50, p1_episodes=90,\n"
        "        steps_per_episode=150, rung_id=\"acquire\", total_denominator=140,\n"
        "    )\n",
        "    result = {\"survival_horizon\": 12.0}\n")
    assert _lint_src(src) is None


def test_zpw_no_competence_survival_foraging_token_is_silent():
    """The gate requires the file to plausibly read a z_world-dependent metric -- a
    driver whose criteria mention none of the three tokens must stay quiet even though
    it calls _train_all_on_agent with no warmup."""
    src = _DEFECTIVE.replace('"""A driver that trains a survival competence without the '
                             'SD-070 z_world warmup."""',
                             '"""A driver that trains something unrelated."""')
    src = src.replace('return {"survival_horizon": result["survival_horizon"]}',
                      'return {"n_ticks": result["n_ticks"]}')
    assert _lint_src(src) is None


def test_zpw_syntax_error_is_silent_not_fatal():
    assert _lint_src("def broken(:\n    pass\n") is None


def test_zpw_explicit_opt_out_is_honoured():
    src = _DEFECTIVE.replace(
        '"""A driver that trains a survival competence without the SD-070 z_world warmup."""',
        'ZWORLD_P0_WARMUP_EXEMPT = "deliberately frozen-encoder ablation"')
    assert _lint_src(src) is None


# ---- (4) the confirmed carriers, as a differential against the real drivers -----------

def test_confirmed_875_positive_specimen_fires():
    """V3-EXQ-875 (MECH-471) -- the incident this gate was written for. All three call
    sites omit zworld_p0_episodes; must fire."""
    p = EXPERIMENTS_DIR / POSITIVE_SPECIMEN
    if not p.exists():
        pytest.skip(f"{POSITIVE_SPECIMEN} not present in this checkout")
    msg = V.zworld_p0_warmup_lint(p)
    assert msg is not None, (
        "V3-EXQ-875 omits zworld_p0_episodes on all three _train_all_on_agent call sites "
        "(lines 270/279/288) -- the lint must fire on it.")


def test_confirmed_875a_negative_specimen_is_quiet():
    """V3-EXQ-875a -- the corrected re-run. Its FIRST call passes
    zworld_p0_episodes=zworld_p0; the other two correctly omit it (same shared agent,
    already-trained encoder). Must NOT fire -- this is the reference fixed shape."""
    p = EXPERIMENTS_DIR / NEGATIVE_SPECIMEN
    if not p.exists():
        pytest.skip(f"{NEGATIVE_SPECIMEN} not present in this checkout")
    assert V.zworld_p0_warmup_lint(p) is None, (
        "V3-EXQ-875a wires zworld_p0_episodes=zworld_p0 into its first (acquisition) call "
        "-- the file-wide escape must clear it.")


def test_confirmed_728_postfix_driver_is_quiet():
    """V3-EXQ-728's ORIGINAL run had this exact defect ('3/3 seeds failed'); the driver
    on disk today is the post-fix version (zworld_p0_episodes threaded through by name
    from its own wrapper parameter). Must NOT fire -- it is a negative control now."""
    p = EXPERIMENTS_DIR / "v3_exq_728_trained_allon_capability_point.py"
    if not p.exists():
        pytest.skip("V3-EXQ-728 not present in this checkout")
    assert V.zworld_p0_warmup_lint(p) is None


# ---- (5) selector + WARN-only invariant ------------------------------------------------

def test_zpw_is_registered():
    assert "zworld_p0_warmup" in V.CHECK_NAMES


def test_zpw_is_warn_only_under_strict_and_paths():
    """INVARIANT: never blocks, like every other lint in this family."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(textwrap.dedent(_DEFECTIVE))
        name = f.name
    try:
        r = _run("--checks", "zworld_p0_warmup", "--quiet", "--strict", "--paths", name)
        assert r.returncode == 0, r.stdout[-2000:]
        assert "ZWORLD_P0-WARMUP" in r.stdout
    finally:
        Path(name).unlink()


def test_zpw_is_selectable_and_does_not_drag_in_other_checks():
    r = _run("--checks", "zworld_p0_warmup", "--quiet")
    assert r.returncode == 0
    assert "zworld_p0-warmup-warning(s)" in r.stdout
    assert "0 inert-salience-dacc_bias-warning(s)" in r.stdout
    assert "0 dacc-_last_bundle-warning(s)" in r.stdout


# ---- (6) the corpus pin ----------------------------------------------------------------

# Pinned 2026-08-03, at the commit that introduced this gate. TWO real carriers:
#   v3_exq_875_mech471_competence_provenance.py  -- the incident this gate was written
#     for (already superseded by 875a; kept on disk, record-frozen, per the "do not
#     retro-edit a landed driver" rule every sibling in this file also follows).
#   v3_exq_882_mech472_context_memorization_generalization.py -- an INDEPENDENT carrier
#     of the identical defect, found BY THIS GATE rather than by an autopsy. Left
#     unfixed here on purpose: fixing a driver is out of scope for a lint-authoring
#     change, and the task this gate was commissioned under is the lint itself.
#
# THE PIN'S JOB IS MOVEMENT DETECTION. A RISE means a NEW driver carries the defect: fix
# that driver (pass zworld_p0_episodes on its first call, or add
# ZWORLD_P0_WARMUP_EXEMPT if the frozen encoder is deliberate) rather than re-pinning. A
# FALL is expected only when a carrier is deliberately repaired (an 875a-shaped fix) --
# and should be accompanied by a reason, not just a smaller number.
_PINNED_CORPUS_FIRE_COUNT = 2

# Below this, the pin could pass while measuring a corpus far smaller than the real one.
# The same `> 500` floor the sibling gates and test_corpus_scan_sharing.py use.
_MIN_CORPUS_FILES_FOR_A_MEANINGFUL_PIN = 500


def test_zpw_corpus_fire_rate_is_pinned(corpus_scan):
    """Consumes the SHARED corpus walk (`tests/contracts/conftest.py`) rather than
    enumerating `experiments/` itself -- the standing pattern conftest's module docstring
    lays down for a new corpus-wide lint.
    """
    assert corpus_scan.n_glob_files > _MIN_CORPUS_FILES_FOR_A_MEANINGFUL_PIN, (
        f"corpus walk covered only {corpus_scan.n_glob_files} v3_exq_* drivers, below the "
        f"{_MIN_CORPUS_FILES_FOR_A_MEANINGFUL_PIN} floor -- the fire-count pin would be "
        f"measuring a truncated corpus. Fix the walk (tests/contracts/conftest.py) rather "
        f"than lowering this floor.")
    fired = corpus_scan["zworld_p0_warmup_lint"]
    assert len(fired) == _PINNED_CORPUS_FIRE_COUNT, (
        f"zworld_p0-warmup fire count moved: {len(fired)} vs pinned "
        f"{_PINNED_CORPUS_FIRE_COUNT}. If a NEW driver is in this list, fix the driver "
        f"(pass zworld_p0_episodes on its first call, or add ZWORLD_P0_WARMUP_EXEMPT) "
        f"rather than re-pinning. Fired: {sorted(p.name for p in fired)}")


def test_zpw_the_pin_is_not_vacuous_both_specimens_actually_fire(corpus_scan):
    """SECOND non-vacuity guard, and the one that would survive a rule that quietly
    stopped detecting anything. A count of 2 could in principle be reached by a different
    pair of files; the gate exists because of these two specific drivers, so both are
    named here.

    If this fails while the count above passes, the rule has drifted off its own
    specimens -- do not re-pin, fix the rule.
    """
    fired = {p.name for p in corpus_scan["zworld_p0_warmup_lint"]}
    assert POSITIVE_SPECIMEN in fired, (
        f"{POSITIVE_SPECIMEN} no longer fires -- it is the canonical carrier this gate "
        f"was written for (three _train_all_on_agent call sites, lines 270/279/288, none "
        f"passing zworld_p0_episodes). Fired instead: {sorted(fired)}")
    assert SECOND_POSITIVE_SPECIMEN in fired, (
        f"{SECOND_POSITIVE_SPECIMEN} no longer fires -- it is the second, independently-"
        f"found carrier (single _train_all_on_agent call, criteria built on "
        f"survival_horizon). Fired instead: {sorted(fired)}")
    assert fired == {POSITIVE_SPECIMEN, SECOND_POSITIVE_SPECIMEN}, (
        f"an unexpected THIRD driver fired: "
        f"{sorted(fired - {POSITIVE_SPECIMEN, SECOND_POSITIVE_SPECIMEN})}. That is a NEW "
        f"carrier of the defect -- fix the driver, do not re-pin.")
