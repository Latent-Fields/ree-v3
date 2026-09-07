"""Contract for `contextmemory_write_enablement` -- a bank readout with no write-mode choice.

THE DECISION THIS ENFORCES (user, 2026-09-06; recorded verbatim in the
`decision_2026_09_06` field of substrate entry
`contextmemory-write-path-addressing-degeneracy` in
REE_assembly/evidence/planning/substrate_queue.json). ContextMemory.write()'s
hard-argmin addressing has a deterministic single-slot fixed point under a
low-variance query stream. Until the CONTENT half validates, a driver whose DV
reads bank occupancy, slot content, or a sleep/consolidation contrast on the bank
must set `E1Config.contextmemory_write_selection='refractory'` with
`contextmemory_write_refractory_k=2`. `usage_balancing` is not the interim choice;
`gumbel_learned` is reserved for content-discrimination work and only with
`contextmemory_write_addressing_loss_weight > 0`. THE LIBRARY DEFAULT STAYS
`argmin` -- which is exactly why enforcement is a driver-side WARN, never an
ERROR and never a default change.

WHY A LINT: V3-EXQ-994 hit the 1-slot bank again AFTER the default-off fix
landed. A default-off knob nobody knows about is a knob, not a fix.

THE DELIBERATE NARROWING, pinned here so a later "tightening" is a decision and
not an accident. The brief asked to fire when a driver does not SET the flag to
refractory/gumbel_learned, and named V3-EXQ-943 and V3-EXQ-436g as required
NEGATIVES. Those cannot both hold: 436g deliberately KEEPS `argmin` and says so,
and 943 SWEEPS the mode across arms including argmin. So the firing condition is
AWARENESS -- a bank readout with no mention of the flag anywhere -- which is the
994 shape exactly, and leaves a documented deliberate choice silent. The one
compliance clause kept is the gumbel/loss-weight pair, which is unambiguous.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

import validate_experiments as V  # noqa: E402

EXPERIMENTS_DIR = REPO_ROOT / "experiments"

# The brief's own named cases. These are the ground truth the narrowing above was
# derived from, so they are asserted by NAME rather than by a count.
POSITIVE = "v3_exq_994_claim_probe_ext_007_consolidation_retention.py"
NEGATIVES = (
    "v3_exq_943_contextmemory_write_selection_validation.py",   # sweeps the mode
    "v3_exq_436g_sd017_mech166_bias_writesel_ceiling_retest.py",  # keeps argmin, says so
)
# Found by the same corpus scan; genuine carriers of the shape the decision names.
OTHER_CARRIERS = (
    "v3_exq_436e_sd017_mech166_occupied_slot_retest.py",
    "v3_exq_436f_sd017_mech166_sd016_armed_retest.py",
)


def _lint_src(src: str):
    """Lint a synthetic script from a TEMP DIR -- never experiments/.

    Writing a probe into the shared checkout is the cross-session contamination
    that made this corpus's lint gate bypassable with --no-verify on 2026-09-07.
    Nothing in this lint is path-relative.
    """
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "v3_exq_000_probe.py"
        p.write_text(src, encoding="utf-8")
        return V.contextmemory_write_enablement_lint(p)


# --------------------------------------------------------------------------- #
# (1) the incident shape, and what silences it
# --------------------------------------------------------------------------- #

_READS_BANK = '''
"""A driver whose DV reads the bank."""


def measure(agent):
    return {"n_occupied_slots": agent.e1.context_memory.n_occupied_slots}
'''


def test_fires_on_a_bank_readout_with_no_write_mode_choice():
    out = _lint_src(_READS_BANK)
    assert out is not None
    assert "n_occupied_slots" in out
    assert "contextmemory_write_selection" in out
    assert "refractory" in out


def test_silent_once_the_choice_is_made():
    src = _READS_BANK + '\nCFG = {"contextmemory_write_selection": "refractory",\n' \
                        '       "contextmemory_write_refractory_k": 2}\n'
    assert _lint_src(src) is None


def test_silent_on_a_deliberate_argmin_that_says_so():
    """The 436g shape. A documented deliberate choice is a choice; this lint
    enforces that the question was ASKED, not which answer was given -- the
    value is routinely arm-swept or set through a helper a static scan cannot
    resolve."""
    src = _READS_BANK + '\n# contextmemory_write_selection is deliberately LEFT at "argmin" here.\n'
    assert _lint_src(src) is None


def test_silent_on_a_driver_that_never_touches_the_bank():
    assert _lint_src('def measure(a):\n    return {"reward": a.reward}\n') is None


def test_the_exemption_marker_silences_it():
    src = _READS_BANK + '\nCONTEXTMEMORY_WRITE_ENABLEMENT_EXEMPT = "DV is write-independent"\n'
    assert _lint_src(src) is None


@pytest.mark.parametrize("token", [
    "n_occupied_slots", "n_encode_written_slots", "occupied_slots", "slot_cosine",
])
def test_each_readout_token_triggers(token):
    assert _lint_src(f'def m(a):\n    return a.{token}\n') is not None


def test_the_bare_class_name_is_not_a_trigger():
    """Measured 2026-09-07: matching `ContextMemory` anywhere selects 96 drivers
    and would warn on 83, nearly all docstring mentions in unrelated history.
    The four readout tokens select 11. This pins the narrow trigger."""
    assert _lint_src('"""Uses the ContextMemory bank somewhere."""\nX = 1\n') is None


# --------------------------------------------------------------------------- #
# (2) the gumbel / addressing-loss clause
# --------------------------------------------------------------------------- #
# ZERO corpus carriers today, deliberately pinned by fixtures instead: a clause
# whose only evidence is "nothing fires" has not been tested.

def test_gumbel_without_an_addressing_objective_warns():
    src = _READS_BANK + '\nCFG = {"contextmemory_write_selection": "gumbel_learned"}\n'
    out = _lint_src(src)
    assert out is not None
    assert "contextmemory_write_addressing_loss_weight" in out
    assert "UNTRAINED selection" in out


def test_gumbel_with_an_addressing_objective_is_silent():
    src = _READS_BANK + ('\nCFG = {"contextmemory_write_selection": "gumbel_learned",\n'
                         '       "contextmemory_write_addressing_loss_weight": 0.1}\n')
    assert _lint_src(src) is None


def test_a_prose_mention_of_gumbel_is_not_an_election():
    """The false positive this clause was narrowed to avoid: v3_exq_436g, a
    required NEGATIVE, whose only two occurrences are a docstring listing the
    modes -- "(argmin / refractory / gumbel_learned / BIAS-adjusted argmin)"."""
    src = _READS_BANK + (
        '\n# Modes available: argmin / refractory / gumbel_learned / BIAS-adjusted argmin.\n'
        '# contextmemory_write_selection stays at the default here.\n')
    assert _lint_src(src) is None


def test_the_election_regex_accepts_kwarg_and_dict_spellings():
    for form in ('contextmemory_write_selection="gumbel_learned"',
                 "contextmemory_write_selection = 'gumbel_learned'",
                 '"contextmemory_write_selection": "gumbel_learned"'):
        assert V._CTXMEM_GUMBEL_ELECTION_RE.search(form), form


# --------------------------------------------------------------------------- #
# (3) the corpus, by NAME -- the brief's own positive and negatives
# --------------------------------------------------------------------------- #

def test_the_incident_driver_warns():
    p = EXPERIMENTS_DIR / POSITIVE
    if not p.exists():
        pytest.skip("994 not present")
    out = V.contextmemory_write_enablement_lint(p)
    assert out is not None, "V3-EXQ-994 is the motivating instance and must warn"
    assert "n_encode_written_slots" in out


@pytest.mark.parametrize("name", NEGATIVES)
def test_the_named_negatives_stay_silent(name):
    """943 sweeps the mode across arms; 436g deliberately keeps argmin. Both were
    named as negatives in the brief, and both are why this lint asserts awareness
    rather than compliance."""
    p = EXPERIMENTS_DIR / name
    if not p.exists():
        pytest.skip(f"{name} not present")
    assert V.contextmemory_write_enablement_lint(p) is None


@pytest.mark.parametrize("name", OTHER_CARRIERS)
def test_the_other_measured_carriers_warn(name):
    """Landed drivers with an occupancy DV and no write-mode choice. Reported,
    NOT retro-edited -- their runs are complete."""
    p = EXPERIMENTS_DIR / name
    if not p.exists():
        pytest.skip(f"{name} not present")
    assert V.contextmemory_write_enablement_lint(p) is not None


def test_the_corpus_carrier_set_is_exactly_the_measured_three():
    """Non-vacuity plus a drift alarm. Measured 2026-09-07 over the corpus: 11
    drivers carry a readout token, 8 already mention the flag, 3 warn. A change
    here means the detector's shape moved -- investigate before re-pinning."""
    files = sorted(EXPERIMENTS_DIR.glob("*.py"))
    if len(files) < 100:
        pytest.skip("corpus not present")
    fired = {p.name for p in files
             if V.contextmemory_write_enablement_lint(p) is not None}
    assert fired == {POSITIVE, *OTHER_CARRIERS}, sorted(fired)


# --------------------------------------------------------------------------- #
# (4) the selector, and NEVER-ERROR
# --------------------------------------------------------------------------- #

def _run(*args):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "validate_experiments.py"), *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT))


def test_check_is_selectable_and_reports_its_own_counter():
    p = EXPERIMENTS_DIR / POSITIVE
    if not p.exists():
        pytest.skip("994 not present")
    r = _run("--checks", "contextmemory_write_enablement", "--quiet",
             "--paths", f"experiments/{POSITIVE}")
    assert r.returncode == 0
    assert "contextmemory-write-enablement-warning(s)" in r.stdout


def test_it_never_errors_in_either_mode():
    """The decision says WARN-only in terms, and the three carriers are landed
    drivers whose runs are complete."""
    p = EXPERIMENTS_DIR / POSITIVE
    if not p.exists():
        pytest.skip("994 not present")
    for extra in ([], ["--strict"]):
        r = _run("--checks", "contextmemory_write_enablement", "--quiet", *extra,
                 "--paths", f"experiments/{POSITIVE}")
        assert r.returncode == 0, r.stdout[-2000:]


def test_the_warning_text_is_ascii_only():
    """It reaches stdout (CLAUDE.md ASCII-Only rule)."""
    out = _lint_src(_READS_BANK)
    assert out and all(ord(ch) < 128 for ch in out)


def test_an_unreadable_file_is_not_a_finding():
    assert V.contextmemory_write_enablement_lint(
        EXPERIMENTS_DIR / "does_not_exist_probe.py") is None
