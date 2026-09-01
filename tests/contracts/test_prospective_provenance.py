#!/usr/bin/env python3
"""Prospective provenance at the central chokepoint.

Closes two nodes of REE_assembly/evidence/planning/
substrate_stability_and_drift_detection_plan.md:

  substrate-commit-coverage   -- a manifest may no longer be SILENT about
                                 substrate_commit: it carries the commit, or the
                                 machine-readable reason it could not be taken.
  P1c-prospective-recording   -- enabled_default_off_flags no longer depends on a
                                 driver remembering `agent=`; every REEConfig
                                 built in the process is observed automatically.

DESIGN NOTE ON THE FIXTURES. The manifest_core half is tested with plain stdlib
dataclasses, matching that module's own stdlib-only posture (no torch import
needed for most of this file). The registry half needs the REAL REEConfig,
because the whole claim under test is that `ree_core/utils/config.py`'s
`__post_init__` and `manifest_core`'s reader agree on a `sys` attribute name that
neither imports from the other -- a fixture would test the fixture, not the seam.
Those tests are marked by importing ree_core at call time.

Roughly half of these are NEGATIVE CONTROLS: precedence must not invert, the
never-measured / measured-empty distinction must not collapse, a stock config
must not be reported as having enabled anything, and reading the registry must
not grow it.
"""
from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments._lib import manifest_core as mc  # noqa: E402


@dataclass
class _Nested:
    off_flag: bool = False
    off_count: int = 0
    on_flag: bool = True


@dataclass
class _Cfg:
    nested: _Nested = field(default_factory=_Nested)
    top_off: bool = False
    top_on: bool = True
    scale: float = 1.0


class _Agent:
    def __init__(self, config):
        self.config = config


def _registry():
    return getattr(sys, mc._OBSERVED_CONFIG_ATTR, None)


class _RegistrySandbox:
    """Swap in a private registry so a test can never disturb the real one."""
    def __enter__(self):
        self._prev = getattr(sys, mc._OBSERVED_CONFIG_ATTR, None)
        setattr(sys, mc._OBSERVED_CONFIG_ATTR,
                {"configs": [], "overflow": 0, "suppressed": False})
        return getattr(sys, mc._OBSERVED_CONFIG_ATTR)

    def __exit__(self, *exc):
        if self._prev is None:
            try:
                delattr(sys, mc._OBSERVED_CONFIG_ATTR)
            except AttributeError:
                pass
        else:
            setattr(sys, mc._OBSERVED_CONFIG_ATTR, self._prev)
        return False


# ---------------------------------------------------------------------------
# substrate_commit: the absence is self-describing
# ---------------------------------------------------------------------------

class SubstrateCommitCoverageTests(unittest.TestCase):

    def test_reason_is_no_git_checkout_outside_a_repo(self):
        """The staged remote_pytest tree: git works, the directory is not a repo.

        This is the environment that made hard-enforcing substrate_commit break
        10/84 tests in 2026-08, and it must still be accepted -- with a reason.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            out = mc.substrate_commit_unavailable_reason(repo_root=td)
        self.assertEqual(out["reason"], "no_git_checkout")
        self.assertIn("checked_at_utc", out)
        self.assertTrue(out["checked_at_utc"].endswith("Z"))

    def test_reason_distinguishes_a_missing_path_from_a_missing_git(self):
        """A path that does not exist must report as itself.

        Every probe runs with cwd=root and fails identically when the directory is
        absent, so without an explicit check the reason would read
        `git_unavailable` -- a small but real lie, and the whole value of this
        block is that the recorded explanation is true.
        """
        out = mc.substrate_commit_unavailable_reason(repo_root="/no/such/path/at/all")
        self.assertEqual(out["reason"], "repo_root_missing")

    def test_reason_is_not_fabricated_when_head_resolves(self):
        """NEGATIVE CONTROL: inside a real repo the reason must be `unknown`.

        An honest 'unknown' beats a plausible-but-wrong explanation -- the same
        rule substrate_commit() applies when it reports dirty=None rather than
        claiming a cleanliness it did not verify.

        SKIPPED on a git-less tree, and the skip is the point rather than a
        weakening: scripts/remote_pytest.sh rsyncs its staged tree WITHOUT `.git/`
        on purpose (CLAUDE.md, "Running the test suite"), so on the hub this
        repo_root correctly reports `no_git_checkout`. Asserting `unknown`
        unconditionally made this test pass on the laptop and fail on the box the
        suite actually runs on -- caught by the first full remote run, and the
        same environment split that `test_write_flat_manifest_passes_when_stamped_
        from_real_repo` already guards for.
        """
        if not (REPO / ".git").exists():
            self.skipTest("no real git checkout (staged remote_pytest tree)")
        out = mc.substrate_commit_unavailable_reason(repo_root=REPO)
        self.assertEqual(out["reason"], "unknown")

    def test_bare_omission_is_a_hard_gate_failure(self):
        """Neither field present -> write_flat_manifest must refuse."""
        m = {"recording_schema": "rec/v1", "substrate_hash": "h",
             "machine": "m", "machine_class": "c"}
        self.assertIn("substrate_commit", mc.missing_mandatory_core_fields(m))

    def test_explained_absence_passes_the_gate(self):
        m = {"recording_schema": "rec/v1", "substrate_hash": "h",
             "machine": "m", "machine_class": "c",
             "substrate_commit_unavailable": {"reason": "no_git_checkout"}}
        self.assertEqual(mc.missing_mandatory_core_fields(m), [])

    def test_present_commit_passes_the_gate(self):
        m = {"recording_schema": "rec/v1", "substrate_hash": "h",
             "machine": "m", "machine_class": "c",
             "substrate_commit": {"commit": "abc123"}}
        self.assertEqual(mc.missing_mandatory_core_fields(m), [])

    def test_stamp_fills_one_or_the_other_never_neither(self):
        """The invariant, stated as a test: after a stamp, silence is impossible."""
        m = {"run_id": "x_v3"}
        mc.stamp_recording_core(m, script_path=Path(__file__))
        self.assertTrue(
            m.get("substrate_commit") or m.get("substrate_commit_unavailable"),
            "a stamped manifest must carry the commit or the reason it could not",
        )

    def test_stamp_in_a_gitless_tree_records_the_reason(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            m = {"run_id": "x_v3"}
            mc.stamp_recording_core(m, script_path=Path(__file__), repo_root=td)
        self.assertIsNone(m.get("substrate_commit"))
        self.assertEqual(
            (m.get("substrate_commit_unavailable") or {}).get("reason"),
            "no_git_checkout")
        self.assertEqual(mc.missing_mandatory_core_fields(
            {**m, "recording_schema": "rec/v1", "substrate_hash": "h",
             "machine": "m", "machine_class": "c"}), [])

    def test_an_author_set_commit_is_never_clobbered(self):
        """NEGATIVE CONTROL: fill-only semantics still hold for both fields."""
        m = {"run_id": "x_v3", "substrate_commit": {"commit": "deliberate"}}
        mc.stamp_recording_core(m, script_path=Path(__file__))
        self.assertEqual(m["substrate_commit"]["commit"], "deliberate")
        self.assertIsNone(m.get("substrate_commit_unavailable"))


# ---------------------------------------------------------------------------
# enabled_default_off_flags: automatic at the chokepoint
# ---------------------------------------------------------------------------

class ObservedConfigFlagTests(unittest.TestCase):

    def test_absent_registry_returns_none_not_empty(self):
        """NEGATIVE CONTROL: never-measured must stay distinct from measured-empty."""
        prev = getattr(sys, mc._OBSERVED_CONFIG_ATTR, None)
        try:
            if prev is not None:
                delattr(sys, mc._OBSERVED_CONFIG_ATTR)
            self.assertIsNone(mc.enabled_default_off_flags_from_observed())
        finally:
            if prev is not None:
                setattr(sys, mc._OBSERVED_CONFIG_ATTR, prev)

    def test_empty_registry_returns_none(self):
        with _RegistrySandbox():
            self.assertIsNone(mc.enabled_default_off_flags_from_observed())

    def test_observed_config_with_nothing_enabled_returns_empty_dict(self):
        """Measured-and-all-default is a POSITIVE statement, recorded as {}."""
        with _RegistrySandbox() as reg:
            reg["configs"].append(_Cfg())
            self.assertEqual(mc.enabled_default_off_flags_from_observed(), {})

    def test_observed_config_reports_top_level_and_nested_flags(self):
        with _RegistrySandbox() as reg:
            c = _Cfg()
            c.top_off = True
            c.nested.off_flag = True
            c.nested.off_count = 3
            reg["configs"].append(c)
            self.assertEqual(
                mc.enabled_default_off_flags_from_observed(),
                {"top_off": True, "nested.off_flag": True, "nested.off_count": 3},
            )

    def test_default_on_fields_are_never_reported(self):
        """NEGATIVE CONTROL: only DEFAULT-OFF knobs that moved are 'enabled'."""
        with _RegistrySandbox() as reg:
            c = _Cfg()
            c.top_on = False      # default-ON turned off -- not a default-off knob
            c.scale = 2.0         # default 1.0 -- not default-off
            reg["configs"].append(c)
            self.assertEqual(mc.enabled_default_off_flags_from_observed(), {})

    def test_union_across_configs_later_wins(self):
        with _RegistrySandbox() as reg:
            a = _Cfg(); a.top_off = True
            b = _Cfg(); b.nested.off_count = 7
            reg["configs"].extend([a, b])
            self.assertEqual(
                mc.enabled_default_off_flags_from_observed(),
                {"top_off": True, "nested.off_count": 7},
            )

    def test_reading_the_registry_does_not_grow_it(self):
        """Stock instances built for comparison must be suppressed.

        Without the suppression this grows by one entry per top-level call and
        eventually evicts real configs at the cap -- silently under-reporting.
        """
        with _RegistrySandbox() as reg:
            reg["configs"].append(_Cfg())
            before = len(reg["configs"])
            for _ in range(20):
                mc.enabled_default_off_flags_from_observed()
            self.assertEqual(len(reg["configs"]), before)
            self.assertFalse(reg["suppressed"], "suppression must be restored")

    def test_suppression_is_restored_even_if_comparison_raises(self):
        """The finally: clause. Without it, one failed stock build would leave the
        registry permanently suppressed and silently stop recording every config
        for the rest of the process."""
        @dataclass
        class _Bad:
            x: bool = False

        with _RegistrySandbox() as reg:
            bad = _Bad()
            orig_init = _Bad.__init__

            def _raise(self, *a, **k):
                raise RuntimeError("stock construction failed")

            _Bad.__init__ = _raise
            try:
                with self.assertRaises(RuntimeError):
                    mc.enabled_default_off_flags(bad)
            finally:
                _Bad.__init__ = orig_init
            self.assertFalse(reg["suppressed"],
                             "suppression must be restored on the exception path")


class StampSourcePrecedenceTests(unittest.TestCase):

    def test_agent_wins_over_the_observed_registry(self):
        """NEGATIVE CONTROL: the automatic path must not displace the precise one."""
        with _RegistrySandbox() as reg:
            observed = _Cfg(); observed.top_off = True
            reg["configs"].append(observed)
            agent_cfg = _Cfg(); agent_cfg.nested.off_flag = True
            m = {"run_id": "x_v3"}
            mc.stamp_recording_core(m, agent=_Agent(agent_cfg),
                                    script_path=Path(__file__))
        self.assertEqual(m["enabled_default_off_flags_source"], "agent")
        self.assertEqual(m["enabled_default_off_flags"], {"nested.off_flag": True})

    def test_registry_is_used_when_no_agent_is_passed(self):
        """THE POINT: coverage without the driver doing anything."""
        with _RegistrySandbox() as reg:
            c = _Cfg(); c.top_off = True
            reg["configs"].append(c)
            m = {"run_id": "x_v3"}
            mc.stamp_recording_core(m, script_path=Path(__file__))
        self.assertEqual(m["enabled_default_off_flags_source"],
                         "process_observed_config")
        self.assertEqual(m["enabled_default_off_flags"], {"top_off": True})

    def test_field_is_omitted_when_nothing_was_observed_at_all(self):
        """NEGATIVE CONTROL: never fabricate a measurement."""
        with _RegistrySandbox():
            m = {"run_id": "x_v3"}
            mc.stamp_recording_core(m, script_path=Path(__file__))
        self.assertNotIn("enabled_default_off_flags", m)
        self.assertNotIn("enabled_default_off_flags_source", m)

    def test_author_set_flags_are_never_overwritten(self):
        with _RegistrySandbox() as reg:
            reg["configs"].append(_Cfg())
            m = {"run_id": "x_v3", "enabled_default_off_flags": {"mine": True}}
            mc.stamp_recording_core(m, script_path=Path(__file__))
        self.assertEqual(m["enabled_default_off_flags"], {"mine": True})
        self.assertNotIn("enabled_default_off_flags_source", m)

    def test_truncation_is_recorded_not_silent(self):
        with _RegistrySandbox() as reg:
            c = _Cfg(); c.top_off = True
            reg["configs"].append(c)
            reg["overflow"] = 5
            m = {"run_id": "x_v3"}
            mc.stamp_recording_core(m, script_path=Path(__file__))
        self.assertEqual(m["enabled_default_off_flags_truncated"], 5)

    def test_no_truncation_marker_on_the_agent_path(self):
        """NEGATIVE CONTROL: overflow describes the registry, not the agent path."""
        with _RegistrySandbox() as reg:
            reg["overflow"] = 5
            m = {"run_id": "x_v3"}
            mc.stamp_recording_core(m, agent=_Agent(_Cfg()),
                                    script_path=Path(__file__))
        self.assertNotIn("enabled_default_off_flags_truncated", m)


class RealREEConfigSeamTests(unittest.TestCase):
    """The seam itself: ree_core writes the registry, manifest_core reads it,
    and neither imports the other. A fixture cannot test this."""

    def test_building_a_real_reeconfig_registers_it(self):
        from ree_core.utils.config import REEConfig
        with _RegistrySandbox() as reg:
            cfg = REEConfig()
            self.assertEqual(len(reg["configs"]), 1)
            self.assertIs(reg["configs"][0], cfg)

    def test_a_real_enabled_flag_reaches_the_manifest_with_no_agent(self):
        """End-to-end, on the exact flag the drift plan reported as invisible.

        `goal.use_hierarchical_goal_credit` lives in GoalConfig (ree_core/goal.py),
        which is why the plan's AST-based analyses could not see it. Runtime
        introspection of the live dataclass tree has no such blind spot.
        """
        from ree_core.utils.config import REEConfig
        with _RegistrySandbox():
            cfg = REEConfig()
            cfg.goal.use_hierarchical_goal_credit = True
            cfg.action_loop_gate_enabled = True
            m = {"run_id": "x_v3"}
            mc.stamp_recording_core(m, script_path=Path(__file__))
        self.assertEqual(m["enabled_default_off_flags_source"],
                         "process_observed_config")
        self.assertEqual(
            m["enabled_default_off_flags"],
            {"goal.use_hierarchical_goal_credit": True,
             "action_loop_gate_enabled": True},
        )

    def test_a_stock_reeconfig_reports_nothing_enabled(self):
        """NEGATIVE CONTROL, and the one that would catch a broken default walk.

        If this ever returns a non-empty dict, `enabled_default_off_flags` has
        started reporting coded defaults as enablements and EVERY manifest is
        wrong -- which is worse than the coverage gap this feature closes.
        """
        from ree_core.utils.config import REEConfig
        with _RegistrySandbox() as reg:
            reg["configs"].append(REEConfig())
            self.assertEqual(mc.enabled_default_off_flags_from_observed(), {})

    def test_registration_observes_the_RESOLVED_config(self):
        """Master-flag resolvers run before registration, so the recorded flags
        are the ones the run actually uses, not the literal the caller passed."""
        from ree_core.utils.config import REEConfig
        with _RegistrySandbox():
            REEConfig(use_mech307_conjunction=True)
            flags = mc.enabled_default_off_flags_from_observed()
        self.assertTrue(flags.get("use_mech307_conjunction"))
        for sub in ("use_mech307_split_surprise",
                    "use_mech307_schema_multichannel",
                    "use_mech307_predicted_location_write"):
            self.assertTrue(flags.get(sub), f"{sub} should be recorded as resolved-on")

    def test_registration_never_raises_on_a_broken_registry(self):
        """A provenance side effect must not break config construction."""
        from ree_core.utils.config import REEConfig
        prev = getattr(sys, mc._OBSERVED_CONFIG_ATTR, None)
        try:
            setattr(sys, mc._OBSERVED_CONFIG_ATTR, "not-a-dict")
            REEConfig()  # must not raise
        finally:
            if prev is None:
                try:
                    delattr(sys, mc._OBSERVED_CONFIG_ATTR)
                except AttributeError:
                    pass
            else:
                setattr(sys, mc._OBSERVED_CONFIG_ATTR, prev)

    def test_cap_bounds_the_registry_and_counts_the_overflow(self):
        from ree_core.utils import config as cfgmod
        with _RegistrySandbox() as reg:
            cap = cfgmod._OBSERVED_CONFIG_CAP
            for _ in range(cap + 7):
                cfgmod.REEConfig()
            self.assertEqual(len(reg["configs"]), cap)
            self.assertEqual(reg["overflow"], 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
