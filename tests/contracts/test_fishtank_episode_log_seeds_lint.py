"""Contracts for `validate_experiments.fishtank_episode_log_seeds_lint`.

THE DEFECT THIS GUARDS (corpus-wide instrument defect, found 2026-08-11).

`REE_assembly/fishtank_viz.html`'s `loadData()` hard-requires `data.seeds` -- a
non-empty list, each entry read as at least `{seed, episodes}` (an optional `arm`
string is used for seed-button labelling) -- and shows "Episode log has no seed
data." for anything else, silently, with no error from the writer.
`v3_exq_913_developmental_ecology_fishtank.py` wrote its episode_log dict with the
top-level key `"runs"` instead, breaking the picker for all 3 of that driver's
logged runs undetected by `validate_experiments.py --strict` or the driver's own
`--dry-run` smoke -- neither checks this specific driver-JSON-vs-viewer-JS
contract. Fixed ree-v3 828bd8b8c9.

WHY A LINT AND NOT JUST THE FIX. Writing this lint found a SECOND, independent
instance the same day: the V3-EXQ-483 lineage (483/483a/483b) writes `"arms"`
instead of `"seeds"` -- see `test_landed_carrier_drivers_still_fire` below. Two
different wrong key names on two unrelated lineages means the failure mode is
"forgot the contract", not "typo'd one string", so the lint checks for the
PRESENCE of `"seeds"` rather than the absence of any one wrong name.

V3-EXQ-483 LINEAGE FIXED 2026-08-11 (chip-20260811-fishtank-483-arms-key).
The 3 driver sources now write `"seeds"` (matching the V3-EXQ-913 precedent,
ree-v3 828bd8b8c9), so `test_landed_carrier_drivers_still_fire` below now pins
the OPPOSITE of its original name: that the set is EMPTY, not that it still
holds the 3 names. This is a deliberate, recorded decision -- not a silent
regression of the pin -- because the driver-source edit does not retro-edit
what any completed run scientifically emitted: it only changes what a FUTURE
re-run (or a future copy-paste of these scripts as a template) would write.
The 4 already-landed sibling `*_episode_log.json` evidence files (one each for
483/483a, two for 483b) were separately regenerated in place -- a pure
`"arms"` -> `"seeds"` key rename, verified as a 1-line diff per file with no
other content touched, and confirmed to render correctly in
`fishtank_viz.html` -- mirroring `REE_assembly` a2758c41d0 for V3-EXQ-913.
Neither of these experiments was re-run; no manifest, metric, or
`evidence_direction` was touched.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import validate_experiments as V  # noqa: E402

EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def _write(tmp_path: Path, body: str, name: str = "v3_exq_999_probe.py") -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


# ---- (1) the firing forms -------------------------------------------------------------

def test_fires_on_runs_key(tmp_path):
    """The exact V3-EXQ-913 shape."""
    p = _write(tmp_path, '''
        def build():
            episode_log = {
                "experiment_type": "x",
                "runs": [{"seed": 0, "episodes": []}],
            }
            return episode_log
    ''')
    issue = V.fishtank_episode_log_seeds_lint(p)
    assert issue is not None
    assert "'seeds'" in issue
    assert "line" in issue


def test_fires_on_arms_key(tmp_path):
    """The exact V3-EXQ-483 lineage shape."""
    p = _write(tmp_path, '''
        def build():
            episode_log = {
                "experiment_type": "x",
                "arms": [{"arm_id": 0, "seed": 0, "episodes": []}],
            }
            return episode_log
    ''')
    issue = V.fishtank_episode_log_seeds_lint(p)
    assert issue is not None
    assert "'arms'" in issue, f"message must name the found key(s): {issue}"


def test_reports_every_site_not_just_the_first(tmp_path):
    """Two constructions with the wrong key must both surface."""
    p = _write(tmp_path, '''
        def a():
            episode_log = {"runs": []}
            return episode_log

        def b():
            episode_log = {"arms": []}
            return episode_log
    ''')
    issue = V.fishtank_episode_log_seeds_lint(p)
    assert issue is not None
    assert "line 3" in issue and "line 7" in issue, issue


# ---- (2) the correct spelling, and non-carriers, must NOT fire ------------------------

def test_correct_seeds_key_does_not_fire(tmp_path):
    p = _write(tmp_path, '''
        def build():
            episode_log = {
                "experiment_type": "x",
                "seeds": [{"seed": 0, "episodes": []}],
            }
            return episode_log
    ''')
    assert V.fishtank_episode_log_seeds_lint(p) is None


def test_seeds_key_alongside_other_keys_does_not_fire(tmp_path):
    """The dict may carry any number of other keys -- only 'seeds' is required."""
    p = _write(tmp_path, '''
        def build():
            episode_log = {
                "experiment_type": "x",
                "env_config": {},
                "phase": "eval",
                "toroidal": False,
                "seeds": [{"seed": 0, "arm": "A", "episodes": []}],
            }
            return episode_log
    ''')
    assert V.fishtank_episode_log_seeds_lint(p) is None


def test_no_episode_log_assignment_does_not_fire(tmp_path):
    """A script that never builds an episode_log dict is not a fishtank-feed
    driver at all -- the lint must not invent a finding for it."""
    p = _write(tmp_path, '''
        def run():
            return {"status": "PASS"}
    ''')
    assert V.fishtank_episode_log_seeds_lint(p) is None


def test_episode_log_passthrough_does_not_fire(tmp_path):
    """`episode_log = result.pop("episode_log", None)` (the real 223/483 read-side
    shape) is not a dict LITERAL construction -- the lint must only look at the
    literal that built the dict in the first place, not every rebinding of the
    name."""
    p = _write(tmp_path, '''
        def build(result):
            episode_log = result.pop("episode_log", None)
            return episode_log
    ''')
    assert V.fishtank_episode_log_seeds_lint(p) is None


def test_unrelated_variable_name_does_not_fire(tmp_path):
    """Only the exact name `episode_log` is in scope -- a differently-named dict
    missing 'seeds' is not this contract's business."""
    p = _write(tmp_path, '''
        def build():
            some_other_log = {"runs": []}
            return some_other_log
    ''')
    assert V.fishtank_episode_log_seeds_lint(p) is None


def test_prose_mention_does_not_fire(tmp_path):
    """Docstrings and comments are invisible to the AST -- deliberately."""
    p = _write(tmp_path, '''
        """This driver used to write episode_log = {"runs": [...]}. Fixed."""
        # historical: episode_log["runs"] was wrong, should be "seeds"

        def build():
            episode_log = {"seeds": []}
            return episode_log
    ''')
    assert V.fishtank_episode_log_seeds_lint(p) is None


# ---- (3) robustness ---------------------------------------------------------------------

def test_exempt_marker_suppresses(tmp_path):
    p = _write(tmp_path, '''
        FISHTANK_EPISODE_LOG_SEEDS_EXEMPT = "local diagnostic dict, not fishtank-viz feed"

        def build():
            episode_log = {"runs": []}
            return episode_log
    ''')
    assert V.fishtank_episode_log_seeds_lint(p) is None


def test_syntax_error_returns_none_not_raise(tmp_path):
    p = _write(tmp_path, "def broken(:\n")
    assert V.fishtank_episode_log_seeds_lint(p) is None


def test_missing_file_returns_none_not_raise(tmp_path):
    assert V.fishtank_episode_log_seeds_lint(tmp_path / "does_not_exist.py") is None


def test_dict_with_no_string_literal_keys_reports_gracefully(tmp_path):
    """A dict built entirely from **-unpacked / computed keys has no string-literal
    keys `_dict_str_keys` can see -- the lint must fire (it cannot prove 'seeds' is
    present) without crashing on an empty key set."""
    p = _write(tmp_path, '''
        def build(extra):
            episode_log = {**extra}
            return episode_log
    ''')
    issue = V.fishtank_episode_log_seeds_lint(p)
    assert issue is not None
    assert "no string-literal keys" in issue


# ---- (4) corpus calibration ------------------------------------------------------------

def test_landed_carrier_drivers_still_fire(corpus_scan):
    """RENAMED IN PLACE, 2026-08-11 (chip-20260811-fishtank-483-arms-key) -- kept
    the original name so its git history stays attached, but the assertion is now
    the opposite of what the name says: the 3 V3-EXQ-483-lineage drivers were
    FIXED (source now writes "seeds"), so this pins the set as EMPTY, not as the
    3 names it originally pinned.

    This is the recorded governance decision the docstring above and the
    module-level docstring's "V3-EXQ-483 LINEAGE FIXED" note both point to: the
    driver-source edit does not retro-edit what any completed run scientifically
    emitted (nothing was re-run; no manifest/metric/evidence_direction changed),
    it only changes what a FUTURE re-run or copy-paste-as-template would write --
    matching the V3-EXQ-913 precedent (ree-v3 828bd8b8c9) exactly. The 4 already
    -landed sibling episode_log.json evidence files were separately regenerated
    in place (pure key rename, verified rendering in fishtank_viz.html).

    A NEW driver firing here (this set becoming non-empty again) means the
    "arms" defect was copied forward into a new script; fix that one the same way.
    """
    expected = set()
    actual = {p.name for p in corpus_scan["fishtank_episode_log_seeds_lint"]}
    assert actual == expected, (
        f"carrier set changed: newly firing {sorted(actual - expected)}, "
        f"no longer firing {sorted(expected - actual)}. A driver newly firing here "
        f"means the 'arms' defect was copied forward into a new script -- fix it "
        f"the same way as V3-EXQ-483/913 (see this test's docstring).")


def test_v3_exq_913_stays_fixed(corpus_scan):
    """The motivating incident. Pinned by name so a regression on the specific
    driver that started this lint is never silent."""
    fired = {p.name for p in corpus_scan["fishtank_episode_log_seeds_lint"]}
    assert "v3_exq_913_developmental_ecology_fishtank.py" not in fired


def test_known_conforming_fishtank_family_drivers_are_clean(corpus_scan):
    """A representative sample of the real fishtank-feed corpus that DOES use
    'seeds' correctly -- guards against an over-broad rewrite of the lint firing on
    conforming drivers."""
    conforming = {
        "v3_exq_223_minimal_vertebrate_ablation.py",
        "v3_exq_223a_toroidal_minimal_vertebrate.py",
        "v3_exq_471_best_agent_fishtank_showcase.py",
        "v3_exq_495_mech163_planned_system_gate.py",
        "v3_exq_524_reef_fishtank_showcase.py",
        "v3_exq_664_affective_fishtank_showcase.py",
        "v3_exq_906_full_stack_observational_fishtank.py",
        "v3_exq_909_sleep_dv_fishtank_multifiring.py",
        "v3_exq_911_ecology_enrichment_fishtank.py",
        "v3_exq_912_uncensored_survival_fishtank.py",
        "v3_exq_916_relief_safety_fishtank_showcase.py",
    }
    for name in conforming:
        assert (EXPERIMENTS_DIR / name).exists(), f"missing {name}"
    fired = {p.name for p in corpus_scan["fishtank_episode_log_seeds_lint"]}
    assert not (conforming & fired), f"regressed: {sorted(conforming & fired)}"


# ---- (5) wiring --------------------------------------------------------------------------

def test_lint_is_registered_in_check_names():
    """An unregistered lint is dead code: `main()` gates every lint on `selected`."""
    assert "fishtank_episode_log_seeds" in V.CHECK_NAMES


def test_lint_is_reachable_through_main(tmp_path, capsys):
    """Executes the real CLI over a synthetic carrier. Registration in CHECK_NAMES is
    necessary but not sufficient -- the main-loop branch and the report block are
    separate edits and any one of them can be missed."""
    p = _write(tmp_path, '''
        def build():
            episode_log = {"runs": []}
            return episode_log
    ''')
    argv = sys.argv
    sys.argv = ["validate_experiments.py", "--checks", "fishtank_episode_log_seeds",
                "--paths", str(p)]
    try:
        rc = V.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "FISHTANK-EPISODE_LOG-SEEDS WARNINGS" in out, out
    assert rc == 0, "WARN-only: must never harden, even under --paths"
