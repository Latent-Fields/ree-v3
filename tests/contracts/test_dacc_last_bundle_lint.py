"""Contracts for `validate_experiments.dacc_last_bundle_lint`.

THE DEFECT THIS GUARDS (corpus-wide instrument defect, found 2026-07-29).

Experiment drivers and two shared SD-054 stage helpers read the dACC (MECH-260)
bundle as `getattr(dacc, "_last_bundle", None)`. No object in the substrate defines
`_last_bundle` -- `ree_core/cingulate/dacc.py` has none. The bundle lives on the
AGENT as `agent._dacc_last_bundle` (written `ree_core/agent.py:6148`, canonical read
`:10340`), carrying the per-candidate `suppression` [K] tensor built at
`dacc.py:401` and packed at `:430`.

Because `getattr` was given a default, the miss was SILENT: the read returned None
every tick, so `dacc_max_suppression` / `dacc_pe` / `dacc_bias_nonzero` were pinned
to 0.0 BY CONSTRUCTION -- structurally indistinguishable in a landed manifest from a
measured zero. V3-EXQ-687 self-routed `substrate_not_ready_requeue` on exactly that
zero.

WHY A LINT AND NOT JUST THE FIX. Nothing audited this class. The wrong spelling
reads as ordinary defensive code, and `getattr(x, "name", None)` is the single most
common idiom in these drivers, so review does not catch it. The lint is the durable
half of the repair.

WHAT IS DELIBERATELY *NOT* TESTED HERE: that the 15 landed carrier drivers get
fixed. They are record-frozen -- a completed run's pre-registered emission is not
rewritten -- so the corpus pin below asserts they STILL FIRE. That count going DOWN
without a governance decision is itself a finding.
"""

from __future__ import annotations

import ast
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import validate_experiments as V  # noqa: E402

EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def _write(tmp_path: Path, body: str, name: str = "v3_exq_999_probe.py") -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


# ---- (1) the two firing forms --------------------------------------------------------

def test_fires_on_getattr_form(tmp_path):
    """The swallowing form -- the one every real carrier used."""
    p = _write(tmp_path, '''
        def diag(agent):
            dacc = getattr(agent, "dacc", None)
            bundle = getattr(dacc, "_last_bundle", None)
            return bundle
    ''')
    issue = V.dacc_last_bundle_lint(p)
    assert issue is not None
    assert "_dacc_last_bundle" in issue, "message must name the CORRECT attribute"
    assert "line 4" in issue, f"message must locate the site: {issue}"


def test_fires_on_direct_attribute_form(tmp_path):
    """The raising form. Rarer, but it is the same wrong name."""
    p = _write(tmp_path, '''
        def diag(agent):
            return agent.dacc._last_bundle
    ''')
    issue = V.dacc_last_bundle_lint(p)
    assert issue is not None
    assert "_last_bundle" in issue


def test_reports_every_site_not_just_the_first(tmp_path):
    """A driver with two carrier reads must surface both -- fixing one and shipping is
    the obvious partial-repair failure, and a first-hit-only message invites it."""
    p = _write(tmp_path, '''
        def a(agent):
            return getattr(agent.dacc, "_last_bundle", None)

        def b(agent):
            return getattr(agent.dacc, "_last_bundle", None)
    ''')
    issue = V.dacc_last_bundle_lint(p)
    assert issue is not None
    assert "line 3" in issue and "line 6" in issue, issue


# ---- (2) the correct spelling must NOT fire ------------------------------------------

def test_correct_agent_spelling_does_not_fire(tmp_path):
    """THE substring trap. `_dacc_last_bundle` CONTAINS `_last_bundle`, so a naive
    `"_last_bundle" in src` test fires on every correctly-repaired site -- turning the
    lint into noise precisely where the fix landed. Exact attribute-name equality is
    what makes the lint usable, so it is pinned."""
    p = _write(tmp_path, '''
        def diag(agent):
            bundle = getattr(agent, "_dacc_last_bundle", None)
            direct = agent._dacc_last_bundle
            return bundle, direct
    ''')
    assert V.dacc_last_bundle_lint(p) is None


def test_naive_substring_check_would_have_fired(tmp_path):
    """Negative control for the test above: proves the trap is REAL and not a
    hypothetical. If this ever stops holding, the test above has gone vacuous."""
    src = 'bundle = getattr(agent, "_dacc_last_bundle", None)\n'
    p = _write(tmp_path, src)
    assert "_last_bundle" in src, "the substring collision is the whole hazard"
    assert V.dacc_last_bundle_lint(p) is None, "AST equality must not inherit it"


def test_prose_mention_does_not_fire(tmp_path):
    """Docstrings and comments are invisible to the AST -- deliberately. Several real
    drivers (490h, 490i, 490j, 687a) and validate_experiments.py itself only DESCRIBE
    the defect; a source-text scan would fire on the documentation of the bug."""
    p = _write(tmp_path, '''
        """This driver notes that dacc._last_bundle does not exist."""
        # historical: we used to read dacc._last_bundle here
        MSG = "the dacc._last_bundle path was wrong"

        def diag(agent):
            return getattr(agent, "_dacc_last_bundle", None)
    ''')
    assert V.dacc_last_bundle_lint(p) is None


def test_unrelated_last_bundle_free_file_does_not_fire(tmp_path):
    p = _write(tmp_path, '''
        def diag(agent):
            return getattr(agent, "dacc", None)
    ''')
    assert V.dacc_last_bundle_lint(p) is None


# ---- (3) robustness ------------------------------------------------------------------

def test_exempt_marker_suppresses(tmp_path):
    p = _write(tmp_path, '''
        DACC_LAST_BUNDLE_EXEMPT = "some future object really defines this"

        def diag(obj):
            return getattr(obj, "_last_bundle", None)
    ''')
    assert V.dacc_last_bundle_lint(p) is None


def test_syntax_error_returns_none_not_raise(tmp_path):
    """check_script already reports unreadable/malformed files; a lint that raised
    here would abort the whole corpus sweep."""
    p = _write(tmp_path, "def broken(:\n")
    assert V.dacc_last_bundle_lint(p) is None


def test_missing_file_returns_none_not_raise(tmp_path):
    assert V.dacc_last_bundle_lint(tmp_path / "does_not_exist.py") is None


def test_exotic_receiver_does_not_raise(tmp_path):
    """`_unparse_recv` is message-cosmetic only and must never be able to fail the
    sweep on an expression it merely wanted to quote."""
    p = _write(tmp_path, '''
        def diag(things):
            return getattr(things[0]["k"](1).x, "_last_bundle", None)
    ''')
    issue = V.dacc_last_bundle_lint(p)
    assert issue is not None


def test_getattr_without_default_still_fires(tmp_path):
    """Two-arg getattr raises rather than swallowing, but the NAME is equally wrong."""
    p = _write(tmp_path, '''
        def diag(agent):
            return getattr(agent.dacc, "_last_bundle")
    ''')
    assert V.dacc_last_bundle_lint(p) is not None


# ---- (4) the message must carry the second-order trap --------------------------------

def test_message_warns_about_the_tensor_or_landmine(tmp_path):
    """Repairing the attribute ALONE is not safe at several real carrier sites.

    They pair it with `bundle.get("mode_ev") or bundle.get("harm_interaction")`.
    `mode_ev` is a [K] tensor, so `or` calls `__bool__` and raises "Boolean value of
    Tensor with more than one value is ambiguous" -- and in the two SD-054 stage
    helpers that expression sat OUTSIDE the enclosing try, so the attribute fix on its
    own converts a silent zero into a CRASH. The lint cannot detect that statically,
    so the warning has to ride in the message or the next person repeats it."""
    p = _write(tmp_path, '''
        def diag(agent):
            return getattr(agent.dacc, "_last_bundle", None)
    ''')
    issue = V.dacc_last_bundle_lint(p)
    assert issue is not None
    assert "mode_ev" in issue and "or" in issue, issue
    assert "687a" in issue, "must point at the reference implementation"


def test_tensor_or_really_raises():
    """Negative control for the test above -- pins that the hazard it warns about is
    real in this torch build, so the warning cannot quietly become folklore."""
    torch = pytest.importorskip("torch")
    bundle = {"mode_ev": torch.tensor([1.0, 2.0, 3.0]),
              "harm_interaction": torch.tensor([0.1])}
    with pytest.raises(RuntimeError, match="ambiguous"):
        _ = bundle.get("mode_ev") or bundle.get("harm_interaction")


# ---- (5) corpus calibration ----------------------------------------------------------

def test_repaired_shared_helpers_are_clean():
    """The three shared helpers repaired on 2026-07-29. These are the MULTIPLIER --
    `_metrics.py` is the canonical extractor module and the two SD-054 stage helpers
    back the 603/620/621/626/687 families -- so a regression here re-zeroes the readout
    for every future run of every consumer, not just one driver."""
    for rel in ("_metrics.py",
                "scaffolded_sd054_onboarding.py",
                "goal_stream_stages_sd054.py"):
        p = EXPERIMENTS_DIR / rel
        assert p.exists(), f"missing {rel}"
        assert V.dacc_last_bundle_lint(p) is None, \
            f"{rel} has regressed to the wrong dACC bundle attribute"


def test_metrics_extractor_uses_the_real_agent_attribute_names():
    """`_metrics.extract_dacc_score_bias` had BOTH its substrate attribute names wrong
    (`_last_dacc_score_bias` and `dacc._last_bundle`; the real names are
    `agent._dacc_last_bias` and `agent._dacc_last_bundle`), so it returned None
    unconditionally and `extract_dacc_diagnostics` emitted its zero-filled dict on every
    tick. Asserted against ree_core rather than by name-matching, so the pin tracks the
    substrate if it is ever renamed."""
    src = (EXPERIMENTS_DIR / "_metrics.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "extract_dacc_score_bias")
    names = {a.args[1].value for a in ast.walk(fn)
             if isinstance(a, ast.Call) and isinstance(a.func, ast.Name)
             and a.func.id == "getattr" and len(a.args) >= 2
             and isinstance(a.args[1], ast.Constant)}
    assert "_dacc_last_bundle" in names
    assert "_dacc_last_bias" in names
    assert "_last_dacc_score_bias" not in names, "the non-existent cache name is back"

    agent_src = (REPO_ROOT / "ree_core" / "agent.py").read_text(encoding="utf-8")
    for attr in ("_dacc_last_bundle", "_dacc_last_bias"):
        assert f"self.{attr}" in agent_src, \
            f"{attr} no longer exists on the agent -- re-derive the extractor"


def test_dacc_module_defines_no_last_bundle():
    """The root fact the whole lint rests on. If the dACC module ever grows a real
    `_last_bundle`, this lint's premise changes and it must be re-derived rather than
    left firing on what would then be correct code."""
    dacc_src = (REPO_ROOT / "ree_core" / "cingulate" / "dacc.py").read_text(encoding="utf-8")
    tree = ast.parse(dacc_src)
    assigned = {t.attr for n in ast.walk(tree) if isinstance(n, ast.Assign)
                for t in n.targets
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                and t.value.id == "self"}
    assigned |= {n.target.attr for n in ast.walk(tree) if isinstance(n, ast.AnnAssign)
                 and isinstance(n.target, ast.Attribute)
                 and isinstance(n.target.value, ast.Name) and n.target.value.id == "self"}
    assert "_last_bundle" not in assigned


def test_landed_carrier_drivers_still_fire():
    """The 15 landed carrier drivers are RECORD-FROZEN and deliberately NOT repaired.

    Pinning the exact set (not just a count) does two things: it stops a well-meaning
    session from retro-editing a completed run's pre-registered emission, and it keeps
    the governance record of which manifests carry a structurally-zeroed dACC readout
    checkable from the code rather than from a prose note that will drift.
    """
    expected = {
        "v3_exq_490_mech269b_vs_rollout_gating.py",
        "v3_exq_490b_mech269b_vs_rollout_gating.py",
        "v3_exq_490c_mech269b_with_liking_bridge.py",
        "v3_exq_490e_mech295_seeding_strengthening.py",
        "v3_exq_490f_mech295_seeding_strengthening.py",
        "v3_exq_620_sd037_axis_a_phase1_consumer_input_distributions.py",
        "v3_exq_620b_sd037_axis_a_phase1_consumer_input_distributions_stream_on.py",
        "v3_exq_621_scaffolded_sd054_onboarding_substrate_readiness.py",
        "v3_exq_621a_scaffolded_sd054_onboarding_substrate_readiness.py",
        "v3_exq_625_sd037_axis_b_phase1b_consumer_input_distributions_sustained_threat.py",
        "v3_exq_625b_sd037_axis_b_phase1b_consumer_input_distributions_sustained_threat.py",
        "v3_exq_625c_sd037_axis_b_phase1b_dynamic_crossings_mech341.py",
        "v3_exq_626_goal_pipeline_developmental_window_diagnostic.py",
        "v3_exq_626a_goal_pipeline_developmental_window_diagnostic.py",
        "v3_exq_687_q045_mech313_mech260_4arm_tonic_noise_ablation.py",
    }
    actual = {p.name for p in sorted(EXPERIMENTS_DIR.glob("v3_exq_*.py"))
              if V.dacc_last_bundle_lint(p)}
    assert actual == expected, (
        f"carrier set changed: newly firing {sorted(actual - expected)}, "
        f"no longer firing {sorted(expected - actual)}. A driver that STOPPED firing "
        f"was retro-edited -- a completed run's emission must not be rewritten. A NEW "
        f"driver firing means the defect was copied forward; fix that one.")


def test_only_known_helper_carrier_remains():
    """Non-driver carriers under experiments/. `_lib/goal_pipeline_tier1.py` was the
    last one, DEFERRED on 2026-07-29 because another session (dazzling-taussig-f58f4c)
    held the claim on it with uncommitted work in the shared checkout, and repaired
    2026-07-31 (chip-20260729-dacc-tier1-carrier) -- it is live _lib code consumed by
    483d/483e/490j, so unlike the frozen drivers it SHOULD be repaired. Tightened to
    the empty set now that it has landed; if something NEW appears here it means a
    shared helper has regressed, which is the multiplier case and matters more than a
    single driver.
    """
    drivers = set(EXPERIMENTS_DIR.glob("v3_exq_*.py"))
    actual = {str(p.relative_to(EXPERIMENTS_DIR))
              for p in sorted(EXPERIMENTS_DIR.rglob("*.py"))
              if p not in drivers and V.dacc_last_bundle_lint(p)}
    assert actual == set(), (
        f"helper carrier set changed: {sorted(actual)}. A shared helper has regressed "
        f"to the wrong dACC bundle attribute -- that is the multiplier case and "
        f"matters more than a single driver.")


# ---- (5b) the repair actually reads data ---------------------------------------------
# Name-matching (above) proves the spelling is right; these prove the extractor RETURNS
# something. Stubs rather than a real REEAgent deliberately: the point is the read path,
# and a full agent construction would make this a slow integration test for no extra
# coverage of the thing that broke.

class _StubDACC:
    """Stands in for the dACC module. Note it defines NO `_last_bundle` -- that absence
    is the defect, so the stub must reproduce it or the test proves nothing."""


class _StubAgent:
    def __init__(self, bundle=None, bias=None, adapter=None):
        self.dacc = _StubDACC()
        if bundle is not None:
            self._dacc_last_bundle = bundle
        if bias is not None:
            self._dacc_last_bias = bias
        self.dacc_adapter = adapter


def test_extractor_returns_the_cached_bias_when_present():
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(EXPERIMENTS_DIR))
    try:
        import _metrics
    finally:
        sys.path.pop(0)
    bias = torch.tensor([0.5, -0.25, 0.75])
    agent = _StubAgent(bias=bias)
    got = _metrics.extract_dacc_score_bias(agent)
    assert got is not None, "the cache fast path is still reading a non-existent attribute"
    assert torch.equal(got, bias)


def test_extractor_recomputes_from_the_agent_bundle_when_cache_is_cold():
    """The fallback path -- the one that was reading `dacc._last_bundle`."""
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(EXPERIMENTS_DIR))
    try:
        import _metrics
    finally:
        sys.path.pop(0)
    seen = {}

    class _Adapter:
        def forward(self, bundle):
            seen["bundle"] = bundle
            return torch.tensor([1.0, 2.0])

    bundle = {"suppression": torch.tensor([1.0, 0.0, 0.0]), "pe": 0.3}
    agent = _StubAgent(bundle=bundle, adapter=_Adapter())
    got = _metrics.extract_dacc_score_bias(agent)
    assert got is not None, "the bundle fallback is still reading dacc._last_bundle"
    assert seen["bundle"] is bundle, "the adapter must receive the AGENT's bundle"


def test_diagnostics_are_nonzero_for_a_live_bias():
    """The end-to-end consequence. Before the repair this returned the zero-filled dict
    unconditionally -- which is exactly what a landed manifest recorded as a measurement."""
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(EXPERIMENTS_DIR))
    try:
        import _metrics
    finally:
        sys.path.pop(0)
    agent = _StubAgent(bias=torch.tensor([0.5, -0.25, 0.75]))
    diag = _metrics.extract_dacc_diagnostics(agent)
    assert diag["dacc_score_bias_nonzero"] == 1
    assert diag["dacc_score_bias_max_abs"] == pytest.approx(0.75)
    assert diag["dacc_score_bias_norm"] > 0.0


def test_diagnostics_still_zero_when_dacc_is_genuinely_absent():
    """The true-negative. A structural zero and a real 'dACC is off' zero must stay
    distinguishable by CONFIG, not by both being unconditionally zero."""
    pytest.importorskip("torch")
    sys.path.insert(0, str(EXPERIMENTS_DIR))
    try:
        import _metrics
    finally:
        sys.path.pop(0)

    class _NoDACC:
        dacc = None

    diag = _metrics.extract_dacc_diagnostics(_NoDACC())
    assert diag["dacc_score_bias_nonzero"] == 0
    assert diag["dacc_score_bias_norm"] == 0.0


# ---- (6) wiring ----------------------------------------------------------------------

def test_lint_is_registered_in_check_names():
    """An unregistered lint is dead code: `main()` gates every lint on `selected`."""
    assert "dacc_last_bundle" in V.CHECK_NAMES


def test_lint_is_reachable_through_main(tmp_path, capsys):
    """Executes the real CLI over a synthetic carrier. Registration in CHECK_NAMES is
    necessary but not sufficient -- the main-loop branch and the report block are three
    separate edits and any one of them can be missed."""
    p = _write(tmp_path, '''
        def diag(agent):
            return getattr(agent.dacc, "_last_bundle", None)
    ''')
    argv = sys.argv
    sys.argv = ["validate_experiments.py", "--checks", "dacc_last_bundle", "--paths", str(p)]
    try:
        rc = V.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "DACC-_LAST_BUNDLE WARNINGS" in out, out
    assert rc == 0, "WARN-only: must never harden, even under --paths"
