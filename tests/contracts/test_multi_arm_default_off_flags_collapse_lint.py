"""Contracts for `validate_experiments.multi_arm_default_off_flags_collapse_lint`.

THE DEFECT THIS GUARDS (confirmed 2026-08-16, wanting-authority cluster autopsy).

`experiments/_lib/manifest_core.py::enabled_default_off_flags_for_agents` pools multiple
agents' default-off REEConfig knobs with a plain `dict.update()` per agent, so on a
genuine multi-arm sweep of a default-off knob, only the LAST agent constructed survives in
the manifest's top-level `enabled_default_off_flags` block. That function's own docstring
calls this out as "a known, stated simplification ... not a guarantee of per-arm
attribution" -- but a driver relying on it can still silently misrepresent which arm's
config the manifest records.

V3-EXQ-931 is the confirmed carrier: `ARM_WEIGHTS: Dict[str, float] = {"ARM_W0": 0.0,
"ARM_W05": 0.5, "ARM_W50": 50.0, "ARM_W500": 500.0, "ARM_W5000": 5000.0}`, and the manifest
recorded `hippocampal.wanting_weight: 5000.0` (the last-constructed arm, the positive
control) even though the run's own `OPERATING_ARM` is `ARM_W05` (0.5). The driver's own
comment at the agent-construction site names the hazard and falls into it anyway.

CALIBRATION HISTORY (see REE_assembly/evidence/planning/
nonproduction_config_drift_audit_2026-08-18.md for the full numbers). Two broader framings
of "non-production config drift" were tried first and both failed calibration hard --
77% and 72% fire rates over the corpus, respectively -- before this narrower,
software-defect-scoped signal was accepted at exactly ONE fire.

WHAT THIS DELIBERATELY DOES NOT CATCH. It does not detect "is this driver testing a
non-production configuration" in general -- see the two rejected framings above. It fires
only on the specific shape: a default-off knob swept from a module-level >=2-distinct-value
collection via subscript, fed into a chokepoint-writer call whose `agent=` is structurally a
multi-agent expression, with no explicit `enabled_default_off_flags=` override. A single-arm
driver, a driver that resolves its per-arm value some other way (a dict comprehension, a
helper function), or a driver that already overrides `enabled_default_off_flags=` explicitly
is not this lint's business.
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

CARRIER = "v3_exq_931_cem_wanting_weight_selection_authority.py"


def _write(tmp_path: Path, body: str, name: str = "v3_exq_999_probe.py") -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


# ---- (1) the firing forms ---------------------------------------------------------------

def test_fires_on_the_real_carrier_shape(tmp_path):
    """The exact v3_exq_931 shape: a module-level per-arm dict, `wanting_weight=` set from
    a subscript into it, and `agent=list(...)` with no explicit override."""
    p = _write(tmp_path, '''
        from experiments.pack_writer import write_flat_manifest

        ARM_WEIGHTS = {"ARM_W0": 0.0, "ARM_W05": 0.5, "ARM_W50": 50.0,
                       "ARM_W500": 500.0, "ARM_W5000": 5000.0}
        ARMS = list(ARM_WEIGHTS)

        def build_agent(arm, cfg_cls, agent_cls):
            cfg = cfg_cls(wanting_weight=ARM_WEIGHTS[arm])
            return agent_cls(cfg)

        def main(cfg_cls, agent_cls, out_dir):
            arm_agents = {arm: build_agent(arm, cfg_cls, agent_cls) for arm in ARMS}
            write_flat_manifest(
                {"run_id": "v3_exq_931_probe_v3"},
                out_dir,
                agent=list(arm_agents.values()),
            )
    ''')
    issue = V.multi_arm_default_off_flags_collapse_lint(p)
    assert issue is not None
    assert "wanting_weight" in issue, issue
    assert "MULTI-ARM enabled_default_off_flags COLLAPSE RISK" in issue, issue


def test_fires_on_a_list_literal_agent_arg(tmp_path):
    # Fixture flag is `use_cem_modulatory_authority`, not `mode_partitioned_cem` --
    # the latter's real config.py default flipped False -> True 2026-08-26
    # (SD-MECH267-CEM-SELECTION-FIX), so `_default_off_knob_names()` (which parses
    # the REAL config.py) no longer recognises it as default-off and this lint
    # would stop firing on it. Any still-default-off bool works here; the lint's
    # own logic under test does not care which one.
    p = _write(tmp_path, '''
        from experiments.pack_writer import write_pack

        CAPS = [True, False, True]

        def build(i, cfg_cls):
            return cfg_cls(use_cem_modulatory_authority=CAPS[i])

        def main(cfg_cls, a0, a1, a2, out_dir):
            write_pack({"run_id": "x_v3"}, out_dir, agent=[a0, a1, a2])
    ''')
    assert V.multi_arm_default_off_flags_collapse_lint(p) is not None


def test_fires_on_a_tuple_agent_arg(tmp_path):
    p = _write(tmp_path, '''
        from experiments.pack_writer import write_flat_manifest

        DRIVES = (False, True)

        def build(i, cfg_cls):
            return cfg_cls(use_external_task_drive=DRIVES[i])

        def main(cfg_cls, a0, a1, out_dir):
            write_flat_manifest({"run_id": "x_v3"}, out_dir, agent=(a0, a1))
    ''')
    assert V.multi_arm_default_off_flags_collapse_lint(p) is not None


def test_fires_on_a_listcomp_agent_arg(tmp_path):
    p = _write(tmp_path, '''
        from experiments.pack_writer import write_flat_manifest

        CAPS = {"a": True, "b": False}

        def build(arm, cfg_cls):
            return cfg_cls(use_cem_modulatory_authority=CAPS[arm])

        def main(cfg_cls, out_dir):
            agents = {arm: build(arm, cfg_cls) for arm in CAPS}
            write_flat_manifest({"run_id": "x_v3"}, out_dir,
                                 agent=[agents[a] for a in CAPS])
    ''')
    assert V.multi_arm_default_off_flags_collapse_lint(p) is not None


def test_fires_on_attribute_assignment_form(tmp_path):
    """`cfg.wanting_weight = ARM_WEIGHTS[arm]` (attribute assign) rather than a keyword
    argument -- the sibling form `_sets_knob_truthy`-style helpers already cover."""
    p = _write(tmp_path, '''
        from experiments.pack_writer import write_flat_manifest

        ARM_WEIGHTS = {"ARM_W0": 0.0, "ARM_W05": 0.5, "ARM_W50": 50.0}

        def build(arm, cfg):
            cfg.wanting_weight = ARM_WEIGHTS[arm]
            return cfg

        def main(cfg_cls, agent_cls, out_dir):
            agents = [agent_cls(build(a, cfg_cls())) for a in ARM_WEIGHTS]
            write_flat_manifest({"run_id": "x_v3"}, out_dir, agent=list(agents))
    ''')
    assert V.multi_arm_default_off_flags_collapse_lint(p) is not None


# ---- (2) negative controls -----------------------------------------------------------

def test_single_agent_name_does_not_fire(tmp_path):
    """A bare `agent=rep_agent` cannot collapse anything -- there is nothing to pool."""
    p = _write(tmp_path, '''
        from experiments.pack_writer import write_flat_manifest

        ARM_WEIGHTS = {"ARM_W0": 0.0, "ARM_W05": 0.5, "ARM_W50": 50.0}

        def build(arm, cfg_cls):
            return cfg_cls(wanting_weight=ARM_WEIGHTS[arm])

        def main(cfg_cls, agent_cls, out_dir):
            rep_agent = agent_cls(build("ARM_W05", cfg_cls))
            write_flat_manifest({"run_id": "x_v3"}, out_dir, agent=rep_agent)
    ''')
    assert V.multi_arm_default_off_flags_collapse_lint(p) is None


def test_explicit_enabled_default_off_flags_override_does_not_fire(tmp_path):
    """The prescribed fix, verbatim: pass enabled_default_off_flags= explicitly."""
    p = _write(tmp_path, '''
        from experiments.pack_writer import write_flat_manifest

        ARM_WEIGHTS = {"ARM_W0": 0.0, "ARM_W05": 0.5, "ARM_W50": 50.0}

        def build(arm, cfg_cls):
            return cfg_cls(wanting_weight=ARM_WEIGHTS[arm])

        def main(cfg_cls, agent_cls, out_dir):
            agents = [agent_cls(build(a, cfg_cls)) for a in ARM_WEIGHTS]
            write_flat_manifest(
                {"run_id": "x_v3"}, out_dir, agent=list(agents),
                enabled_default_off_flags={"hippocampal.wanting_weight": 0.5},
            )
    ''')
    assert V.multi_arm_default_off_flags_collapse_lint(p) is None


def test_no_writer_call_does_not_fire(tmp_path):
    """Sweeps the knob but never writes a manifest at all -- not this lint's business,
    and also exercises the cheap source pre-filter."""
    p = _write(tmp_path, '''
        ARM_WEIGHTS = {"ARM_W0": 0.0, "ARM_W05": 0.5, "ARM_W50": 50.0}

        def build(arm, cfg_cls):
            return cfg_cls(wanting_weight=ARM_WEIGHTS[arm])
    ''')
    assert V.multi_arm_default_off_flags_collapse_lint(p) is None


def test_single_value_collection_is_not_a_sweep(tmp_path):
    """A dict/list with only ONE distinct value is not a genuine per-arm sweep --
    could be an incidental single-entry lookup table."""
    p = _write(tmp_path, '''
        from experiments.pack_writer import write_flat_manifest

        ARM_WEIGHTS = {"ARM_W05": 0.5}

        def build(arm, cfg_cls):
            return cfg_cls(wanting_weight=ARM_WEIGHTS[arm])

        def main(cfg_cls, agent_cls, out_dir):
            agents = [agent_cls(build(a, cfg_cls)) for a in ARM_WEIGHTS]
            write_flat_manifest({"run_id": "x_v3"}, out_dir, agent=list(agents))
    ''')
    assert V.multi_arm_default_off_flags_collapse_lint(p) is None


def test_duplicate_valued_collection_is_not_a_sweep(tmp_path):
    """Same literal value repeated is not a genuine sweep -- `set(vals)` collapses to 1."""
    p = _write(tmp_path, '''
        from experiments.pack_writer import write_flat_manifest

        ARM_WEIGHTS = {"ARM_W0": 0.0, "ARM_W0_DUP": 0.0}

        def build(arm, cfg_cls):
            return cfg_cls(wanting_weight=ARM_WEIGHTS[arm])

        def main(cfg_cls, agent_cls, out_dir):
            agents = [agent_cls(build(a, cfg_cls)) for a in ARM_WEIGHTS]
            write_flat_manifest({"run_id": "x_v3"}, out_dir, agent=list(agents))
    ''')
    assert V.multi_arm_default_off_flags_collapse_lint(p) is None


def test_non_default_off_field_name_does_not_fire(tmp_path):
    """`totally_fake_knob` is not a REEConfig dataclass field at all -- must not fire on
    an arbitrary subscripted keyword argument that happens to look like the shape."""
    p = _write(tmp_path, '''
        from experiments.pack_writer import write_flat_manifest

        VALUES = {"a": 1, "b": 2, "c": 3}

        def build(k, cfg_cls):
            return cfg_cls(totally_fake_knob_xyz=VALUES[k])

        def main(cfg_cls, agent_cls, out_dir):
            agents = [agent_cls(build(k, cfg_cls)) for k in VALUES]
            write_flat_manifest({"run_id": "x_v3"}, out_dir, agent=list(agents))
    ''')
    assert V.multi_arm_default_off_flags_collapse_lint(p) is None


def test_literal_value_not_subscript_does_not_fire(tmp_path):
    """Setting the knob to a plain literal (not sourced from a swept collection) is
    ordinary single-value configuration, not a sweep."""
    p = _write(tmp_path, '''
        from experiments.pack_writer import write_flat_manifest

        def build(cfg_cls):
            return cfg_cls(wanting_weight=0.5)

        def main(cfg_cls, agent_cls, out_dir):
            agents = [agent_cls(build(cfg_cls)) for _ in range(3)]
            write_flat_manifest({"run_id": "x_v3"}, out_dir, agent=list(agents))
    ''')
    assert V.multi_arm_default_off_flags_collapse_lint(p) is None


def test_exempt_marker_suppresses(tmp_path):
    p = _write(tmp_path, '''
        from experiments.pack_writer import write_flat_manifest

        MULTI_ARM_DEFAULT_OFF_FLAGS_EXEMPT = "last arm is the one this manifest reports"

        ARM_WEIGHTS = {"ARM_W0": 0.0, "ARM_W05": 0.5, "ARM_W50": 50.0}

        def build(arm, cfg_cls):
            return cfg_cls(wanting_weight=ARM_WEIGHTS[arm])

        def main(cfg_cls, agent_cls, out_dir):
            agents = [agent_cls(build(a, cfg_cls)) for a in ARM_WEIGHTS]
            write_flat_manifest({"run_id": "x_v3"}, out_dir, agent=list(agents))
    ''')
    assert V.multi_arm_default_off_flags_collapse_lint(p) is None


def test_syntax_error_returns_none_not_raise(tmp_path):
    p = _write(tmp_path,
               'from experiments.pack_writer import write_flat_manifest\ndef broken(:\n')
    assert V.multi_arm_default_off_flags_collapse_lint(p) is None


def test_missing_file_returns_none_not_raise(tmp_path):
    assert V.multi_arm_default_off_flags_collapse_lint(tmp_path / "nope.py") is None


def test_message_is_ascii_only(tmp_path):
    """CLAUDE.md: anything reaching stdout must be ASCII (cp1252 mojibake on
    Windows terminals)."""
    p = _write(tmp_path, '''
        from experiments.pack_writer import write_flat_manifest

        ARM_WEIGHTS = {"ARM_W0": 0.0, "ARM_W05": 0.5, "ARM_W50": 50.0}

        def build(arm, cfg_cls):
            return cfg_cls(wanting_weight=ARM_WEIGHTS[arm])

        def main(cfg_cls, agent_cls, out_dir):
            agents = [agent_cls(build(a, cfg_cls)) for a in ARM_WEIGHTS]
            write_flat_manifest({"run_id": "x_v3"}, out_dir, agent=list(agents))
    ''')
    issue = V.multi_arm_default_off_flags_collapse_lint(p)
    assert issue is not None
    issue.encode("ascii")


# ---- (3) corpus calibration ------------------------------------------------------------

def test_carrier_set_is_exactly_the_known_carrier(corpus_scan):
    """Pinned at ONE fire across the whole corpus, by name -- see the module docstring
    for the full calibration history (two rejected broader framings at 77%/72%)."""
    expected = {CARRIER}
    actual = {p.name for p in corpus_scan["multi_arm_default_off_flags_collapse_lint"]}
    assert actual == expected, (
        f"carrier set changed: newly firing {sorted(actual - expected)}, "
        f"no longer firing {sorted(expected - actual)}")


def test_the_carrier_driver_exists(corpus_scan):
    """Guards the pin above against going vacuously green on a renamed file."""
    assert (EXPERIMENTS_DIR / CARRIER).exists(), f"missing {CARRIER}"


# ---- (4) default-off knob parse ---------------------------------------------------------

def test_default_off_knob_names_includes_the_carrier_knob():
    """wanting_weight (HippocampalConfig, config.py) must resolve as default-off -- this
    is the parse the whole lint's knob-name gate depends on."""
    assert "wanting_weight" in V._default_off_knob_names()


def test_default_off_knob_names_excludes_a_nonzero_default_field():
    """A field whose coded default is truthy/non-zero must NOT appear -- otherwise the
    lint would fire on ordinary configuration of an already-on knob.
    `external_task_drive_commit_weight` defaults 1.0 (config.py)."""
    assert "external_task_drive_commit_weight" not in V._default_off_knob_names()


def test_default_off_knob_names_is_memoised():
    """Same object identity on a second call -- the module-level cache, not a re-parse
    of config.py every invocation."""
    first = V._default_off_knob_names()
    second = V._default_off_knob_names()
    assert first is second


# ---- (5) wiring -------------------------------------------------------------------------

def test_lint_is_registered_in_check_names():
    """An unregistered lint is dead code: `main()` gates every lint on `selected`."""
    assert "multi_arm_default_off_flags_collapse" in V.CHECK_NAMES


def test_lint_is_reachable_through_main(tmp_path, capsys):
    """Executes the real CLI over a synthetic carrier. Registration in CHECK_NAMES is
    necessary but not sufficient -- the main-loop branch and the report block are
    separate edits and any one of them can be missed."""
    p = _write(tmp_path, '''
        from experiments.pack_writer import write_flat_manifest

        ARM_WEIGHTS = {"ARM_W0": 0.0, "ARM_W05": 0.5, "ARM_W50": 50.0}

        def build(arm, cfg_cls):
            return cfg_cls(wanting_weight=ARM_WEIGHTS[arm])

        def main(cfg_cls, agent_cls, out_dir):
            agents = [agent_cls(build(a, cfg_cls)) for a in ARM_WEIGHTS]
            write_flat_manifest({"run_id": "x_v3"}, out_dir, agent=list(agents))
    ''')
    argv = sys.argv
    sys.argv = ["validate_experiments.py", "--checks", "multi_arm_default_off_flags_collapse",
                "--paths", str(p)]
    try:
        rc = V.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "MULTI_ARM-DEFAULT_OFF_FLAGS-COLLAPSE WARNINGS" in out, out
    assert rc == 0, "WARN-only: must never harden, even under --paths"


def test_lint_never_hardens_under_strict(tmp_path):
    """--strict must not turn this into a hard failure, per the WARN-only posture stated
    in the lint's own docstring."""
    p = _write(tmp_path, '''
        from experiments.pack_writer import write_flat_manifest

        ARM_WEIGHTS = {"ARM_W0": 0.0, "ARM_W05": 0.5, "ARM_W50": 50.0}

        def build(arm, cfg_cls):
            return cfg_cls(wanting_weight=ARM_WEIGHTS[arm])

        def main(cfg_cls, agent_cls, out_dir):
            agents = [agent_cls(build(a, cfg_cls)) for a in ARM_WEIGHTS]
            write_flat_manifest({"run_id": "x_v3"}, out_dir, agent=list(agents))
    ''')
    argv = sys.argv
    sys.argv = ["validate_experiments.py", "--strict", "--checks",
                "multi_arm_default_off_flags_collapse", "--paths", str(p)]
    try:
        rc = V.main()
    finally:
        sys.argv = argv
    assert rc == 0, "WARN-only: --strict must not harden this lint"
