"""Contracts for the `sd056_training_without_rollout_clamp` lint.

Surfaces under test:
  (1) validate_experiments.sd056_training_without_rollout_clamp_lint -- flags a driver
      that calls `<e2>.world_forward_contrastive_loss(...)` (SD-056 action-conditional
      divergence contrastive training) with no reachable
      `e2_rollout_output_norm_clamp_enabled=True` setting, in the driver itself or in a
      LOCAL config-builder helper it imports and calls one level deep.
  (2) validate_experiments.py --checks sd056_training_without_rollout_clamp -- the
      selector, and the invariant that this gate is WARN-ONLY IN BOTH MODES (never
      hardens under --paths, never affects the exit code even under --strict).
  (3) The corpus fire count, pinned against the real confirmed carrier
      (v3_exq_936_mech439_f_variance_share_under_f_demotion.py) with a non-vacuity
      guard, plus a real negative-control specimen that arms the clamp.

WHY THIS GATE EXISTS. E2.rollout_with_world's per-step ||z_world|| clamp
(ree_core/predictors/e2_fast.py:694-747) is the fix V3-EXQ-617 validated (PASS,
2026-05-31) for an unbounded E2 imagination rollout under sustained SD-056 contrastive
training. It ships DEFAULT OFF (`e2_rollout_output_norm_clamp_enabled: bool = False`)
and is silently omitted rather than deliberately declined on most of the corpus.
Confirmed twice on the identical code path: V3-EXQ-569e (2026-05-31, the incident the
clamp was built for -- rollout magnitudes 1e16-1e18) and V3-EXQ-936 (2026-08-17,
back-solved ||dz_world|| ~1.42e18, f_variance_share saturated to 1.0 in all 8 cells,
MECH-439 neither supported nor weakened -- failure_autopsy_V3-EXQ-936_2026-08-18.json).
This lint is a WARN-only advisory over the residual gap; it deliberately does NOT flip
the config default (a bit-identity-breaking change out of scope here -- see the
SD-056 substrate_queue.json amend_history Step 8 gate).
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

POSITIVE_SPECIMEN = "v3_exq_936_mech439_f_variance_share_under_f_demotion.py"
NEGATIVE_SPECIMEN = "v3_exq_689i_mech448_f_eligibility_demotion_falsifier_repair.py"


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
        return V.sd056_training_without_rollout_clamp_lint(Path(name))
    finally:
        Path(name).unlink()


# The target shape, reduced to its skeleton from the 936 lineage: a driver that calls
# the SD-056 contrastive loss without ever arming the rollout-stability clamp.
_DEFECTIVE = '''
"""A driver that runs SD-056 contrastive training without the rollout clamp."""


def _train_step(agent, z_world, z_world_next, actions):
    loss = agent.e2.world_forward_contrastive_loss(
        z_world, z_world_next, actions, temperature=0.1,
    )
    return loss
'''


# ---- (1) the defect fires -----------------------------------------------------------

def test_sd056_fires_on_the_canonical_shape():
    assert _lint_src(_DEFECTIVE) is not None


def test_sd056_names_the_call_and_the_fix():
    msg = _lint_src(_DEFECTIVE)
    assert msg is not None
    assert "world_forward_contrastive_loss" in msg
    assert "e2_rollout_output_norm_clamp_enabled" in msg
    assert "V3-EXQ-569e" in msg and "V3-EXQ-936" in msg


def test_sd056_message_is_ascii_only():
    """CLAUDE.md: anything reaching stdout is cp1252-safe."""
    msg = _lint_src(_DEFECTIVE)
    assert msg is not None
    assert all(ord(c) < 128 for c in msg), "non-ASCII in a printed lint message"


# ---- (2) THE ESCAPE: the clamp set truthy anywhere in the file is silent ------------

def test_sd056_a_direct_keyword_true_is_silent():
    src = _DEFECTIVE.replace(
        '    return loss\n',
        '    return loss\n\n\n'
        'def _build_agent_config():\n'
        '    return dict(e2_rollout_output_norm_clamp_enabled=True)\n')
    assert _lint_src(src) is None


def test_sd056_an_explicit_false_still_fires():
    """Setting it to a literal False is exactly the inert default, spelled out -- it
    must NOT clear the finding (mirrors _sets_knob_truthy's documented false-literal
    exclusion, shared with dead_z_goal_stream_lint)."""
    src = _DEFECTIVE.replace(
        '    return loss\n',
        '    return loss\n\n\n'
        'def _build_agent_config():\n'
        '    return dict(e2_rollout_output_norm_clamp_enabled=False)\n')
    assert _lint_src(src) is not None


def test_sd056_an_attribute_assignment_is_silent():
    src = _DEFECTIVE.replace(
        '    return loss\n',
        '    return loss\n\n\n'
        'def _configure(config):\n'
        '    config.e2.e2_rollout_output_norm_clamp_enabled = True\n')
    assert _lint_src(src) is None


def test_sd056_a_docstring_mention_alone_does_not_silence():
    """A name-scan would fire on the many scripts that only DISCUSS the flag in prose
    while never setting it -- AST-based `_sets_knob_truthy` must not be fooled."""
    src = _DEFECTIVE.replace(
        '"""A driver that runs SD-056 contrastive training without the rollout clamp."""',
        '"""Discusses e2_rollout_output_norm_clamp_enabled but never sets it."""')
    assert _lint_src(src) is not None


# ---- (3) the remaining preconditions --------------------------------------------------

def test_sd056_no_contrastive_call_is_silent():
    src = _DEFECTIVE.replace(
        "    loss = agent.e2.world_forward_contrastive_loss(\n"
        "        z_world, z_world_next, actions, temperature=0.1,\n"
        "    )\n"
        "    return loss\n",
        "    return 0.0\n")
    assert _lint_src(src) is None


def test_sd056_exempt_marker_is_honoured():
    src = _DEFECTIVE.replace(
        '"""A driver that runs SD-056 contrastive training without the rollout clamp."""',
        'SD056_ROLLOUT_CLAMP_EXEMPT = "deliberately measuring the unclamped path"')
    assert _lint_src(src) is None


def test_sd056_syntax_error_is_silent_not_fatal():
    assert _lint_src("def broken(:\n    pass\n") is None


# ---- (4) the confirmed carrier + a real negative control ------------------------------

def test_confirmed_936_positive_specimen_fires():
    """V3-EXQ-936 -- the incident this gate was written for. Calls
    world_forward_contrastive_loss with no reachable clamp setting; must fire."""
    p = EXPERIMENTS_DIR / POSITIVE_SPECIMEN
    if not p.exists():
        pytest.skip(f"{POSITIVE_SPECIMEN} not present in this checkout")
    msg = V.sd056_training_without_rollout_clamp_lint(p)
    assert msg is not None, (
        "V3-EXQ-936 never sets e2_rollout_output_norm_clamp_enabled -- the lint must "
        "fire on it (failure_autopsy_V3-EXQ-936_2026-08-18.json).")


def test_confirmed_689i_negative_specimen_is_quiet():
    """V3-EXQ-689i -- a real corpus driver that arms the clamp
    (e2_rollout_output_norm_clamp_enabled=True) alongside its own contrastive call.
    Must NOT fire -- this is a genuine armed carrier, not an omission."""
    p = EXPERIMENTS_DIR / NEGATIVE_SPECIMEN
    if not p.exists():
        pytest.skip(f"{NEGATIVE_SPECIMEN} not present in this checkout")
    assert V.sd056_training_without_rollout_clamp_lint(p) is None, (
        "V3-EXQ-689i sets e2_rollout_output_norm_clamp_enabled=True -- must not fire.")


# ---- (5) selector + WARN-only invariant ------------------------------------------------

def test_sd056_is_registered():
    assert "sd056_training_without_rollout_clamp" in V.CHECK_NAMES


def test_sd056_is_warn_only_under_strict_and_paths():
    """INVARIANT: never blocks, like every other lint in this family."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(textwrap.dedent(_DEFECTIVE))
        name = f.name
    try:
        r = _run("--checks", "sd056_training_without_rollout_clamp", "--quiet",
                  "--strict", "--paths", name)
        assert r.returncode == 0, r.stdout[-2000:]
        assert "SD056-TRAINING-WITHOUT-ROLLOUT-CLAMP" in r.stdout
    finally:
        Path(name).unlink()


def test_sd056_is_selectable_and_does_not_drag_in_other_checks():
    r = _run("--checks", "sd056_training_without_rollout_clamp", "--quiet")
    assert r.returncode == 0
    assert "sd056-training-without-rollout-clamp-warning(s)" in r.stdout
    assert "0 dead-z_goal-stream-warning(s)" in r.stdout
    assert "0 multi_arm-default_off_flags-collapse-warning(s)" in r.stdout


# ---- (6) the corpus pin ----------------------------------------------------------------

# Pinned 2026-08-21, at the commit that introduced this gate. 18 corpus drivers call
# world_forward_contrastive_loss with no reachable clamp setting -- see
# failure_autopsy_V3-EXQ-936_2026-08-18.json's implementation_hint (measured "20 of
# 117" there; this repo's corpus at pin time yields 18, six of which -- v3_exq_569a-e
# and v3_exq_613 -- legitimately predate the 2026-05-31 clamp amend and are not
# omissions).
#
# THE PIN'S JOB IS MOVEMENT DETECTION. A RISE means a NEW driver carries the omission:
# fix that driver (pass e2_rollout_output_norm_clamp_enabled=True) or add
# SD056_ROLLOUT_CLAMP_EXEMPT if the unclamped path is deliberate, rather than
# re-pinning. Do NOT retro-edit a LANDED driver whose run is complete.
_PINNED_CORPUS_FIRE_COUNT = 18

# Below this, the pin could pass while measuring a corpus far smaller than the real
# one. The same `> 500` floor the sibling gates and test_corpus_scan_sharing.py use.
_MIN_CORPUS_FILES_FOR_A_MEANINGFUL_PIN = 500


def test_sd056_corpus_fire_rate_is_pinned(corpus_scan):
    """Consumes the SHARED corpus walk (`tests/contracts/conftest.py`) rather than
    enumerating `experiments/` itself -- the standing pattern conftest's module
    docstring lays down for a new corpus-wide lint.
    """
    assert corpus_scan.n_glob_files > _MIN_CORPUS_FILES_FOR_A_MEANINGFUL_PIN, (
        f"corpus walk covered only {corpus_scan.n_glob_files} v3_exq_* drivers, below "
        f"the {_MIN_CORPUS_FILES_FOR_A_MEANINGFUL_PIN} floor -- the fire-count pin "
        f"would be measuring a truncated corpus. Fix the walk (tests/contracts/"
        f"conftest.py) rather than lowering this floor.")
    fired = corpus_scan["sd056_training_without_rollout_clamp_lint"]
    assert len(fired) == _PINNED_CORPUS_FIRE_COUNT, (
        f"sd056-training-without-rollout-clamp fire count moved: {len(fired)} vs "
        f"pinned {_PINNED_CORPUS_FIRE_COUNT}. If a NEW driver is in this list, fix the "
        f"driver (pass e2_rollout_output_norm_clamp_enabled=True) or add "
        f"SD056_ROLLOUT_CLAMP_EXEMPT rather than re-pinning. "
        f"Fired: {sorted(p.name for p in fired)}")


def test_sd056_the_pin_is_not_vacuous_the_confirmed_carrier_fires(corpus_scan):
    """Non-vacuity guard: the pinned count must include the specific incident driver
    this gate was written for, not just any 18 files.
    """
    fired = {p.name for p in corpus_scan["sd056_training_without_rollout_clamp_lint"]}
    assert POSITIVE_SPECIMEN in fired, (
        f"{POSITIVE_SPECIMEN} no longer fires -- it is the canonical carrier this gate "
        f"was written for. Fired instead: {sorted(fired)}")
