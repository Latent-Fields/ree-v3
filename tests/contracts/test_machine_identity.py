"""Contract: machine identity is blind to macOS LocalHostName suffix drift for
the laptop, and is NEVER blind to it for the numbered cloud fleet.

Regression for the 2026-08-15 finding. macOS re-suffixed the Mac's LocalHostName
(`HostName` was unset, so `socket.gethostname()` follows it), taking the box from
`DLAPTOP-4.local` to `DLAPTOP-5.local`. Two consequences, both silent:

  1. The laptop's identity SPLIT. Measured in evidence/experiments/ on that date:
     99 manifests `"machine": "DLAPTOP-4.local"` vs 38 `"DLAPTOP-5.local"`, one
     physical machine, earliest drifted manifest 2026-07-25 -- three weeks
     undetected.
  2. `--laptop-yield-to-cloud` auto-arming compared RAW `socket.gethostname()`
     against `LAPTOP_AUTO_YIELD_HOSTNAME` and so went permanently False,
     disarming cloud-yield with no warning. `test_auto_yield_rearms_*` below is
     the direct regression for that.

ROUGHLY HALF OF THESE TESTS ARE NEGATIVE CONTROLS, deliberately. The tempting
implementation is "strip any trailing -<digits>", which would collapse
ree-cloud-1..5 and ree-worker-1..4 into single identities -- one shared heartbeat
file for the whole fleet and every claim colliding. The fleet tests below are
what stop a later session widening the predicate into that outage; if you relax
`SUFFIX_BLIND_BASES`, they are what should fail.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import machine_identity as mi  # noqa: E402
import experiment_runner as er  # noqa: E402


# --------------------------------------------------------------------------
# Suffix blindness -- the laptop
# --------------------------------------------------------------------------

def test_drifted_laptop_name_canonicalises_to_the_computer_name():
    # `DLAPTOP` is `scutil --get ComputerName` -- what the machine calls itself.
    assert mi.canonical_machine_name("DLAPTOP-5.local") == "DLAPTOP"


def test_the_other_historical_suffix_also_canonicalises_forward():
    # `-4` has no more claim to being the real name than `-5`; both are
    # LocalHostName accidents and both alias forward. This is what keeps the 567
    # existing evidence rows resolving to this machine with no rewrite.
    assert mi.canonical_machine_name("DLAPTOP-4.local") == "DLAPTOP"


def test_bare_computer_name_is_already_canonical():
    assert mi.canonical_machine_name("DLAPTOP") == "DLAPTOP"


def test_suffix_without_local_tld_canonicalises():
    assert mi.canonical_machine_name("DLAPTOP-5") == "DLAPTOP"


def test_canonicalisation_is_case_insensitive():
    assert mi.canonical_machine_name("dlaptop-5.local") == "DLAPTOP"


def test_canonicalisation_is_idempotent():
    once = mi.canonical_machine_name("DLAPTOP-5.local")
    assert mi.canonical_machine_name(once) == once


def test_drifted_and_undrifted_are_the_same_machine():
    assert mi.same_machine("DLAPTOP-4.local", "DLAPTOP-5.local")
    assert mi.same_machine("DLAPTOP-5.local", "DLAPTOP")


# --------------------------------------------------------------------------
# NEGATIVE CONTROLS -- the numbered fleet must never collapse
# --------------------------------------------------------------------------

def test_cloud_fleet_names_pass_through_unchanged():
    for n in range(1, 6):
        name = f"ree-cloud-{n}"
        assert mi.canonical_machine_name(name) == name


def test_worker_fleet_names_pass_through_unchanged():
    for n in range(1, 5):
        name = f"ree-worker-{n}"
        assert mi.canonical_machine_name(name) == name


def test_distinct_cloud_workers_are_distinct_machines():
    names = [f"ree-cloud-{n}" for n in range(1, 6)]
    canon = [mi.canonical_machine_name(n) for n in names]
    assert len(set(canon)) == len(names), "cloud fleet collapsed into one identity"
    assert not mi.same_machine("ree-cloud-1", "ree-cloud-2")


def test_hub_dual_naming_is_deliberately_NOT_handled_here():
    # The hub really is one box answering to both, but that accommodation lives
    # in runner_remote_control._PHASE3_HUB_HOSTNAMES and is scoped to one gate.
    # Folding it in here would silently rename the hub's heartbeat file. Keep the
    # two mechanisms separate.
    assert not mi.same_machine("ree-cloud-1", "ree-worker-1")


def test_unrelated_hosts_pass_through_unchanged():
    for name in ("Daniel-PC", "EWIN-PC", "ree-cloud-1", "buildbox-7"):
        assert mi.canonical_machine_name(name) == name


def test_unknown_numbered_host_is_not_collapsed():
    # Only bases in SUFFIX_BLIND_BASES collapse. A machine nobody has declared
    # drift-prone keeps its number.
    assert mi.canonical_machine_name("buildbox-7") == "buildbox-7"
    assert not mi.same_machine("buildbox-7", "buildbox-8")


# --------------------------------------------------------------------------
# Empty / unknown identity must not satisfy a host-gated branch
# --------------------------------------------------------------------------

def test_empty_input_yields_empty_string():
    assert mi.canonical_machine_name(None) == ""
    assert mi.canonical_machine_name("") == ""
    assert mi.canonical_machine_name("   ") == ""


def test_unknown_identity_is_never_the_same_machine():
    assert not mi.same_machine(None, None)
    assert not mi.same_machine("", "")
    assert not mi.same_machine(None, "DLAPTOP-4.local")


# --------------------------------------------------------------------------
# Wiring -- the call sites that actually decide behaviour
# --------------------------------------------------------------------------

def test_get_machine_name_canonicalises_the_override():
    # A stale `--machine DLAPTOP-5.local` in someone's shell history or a
    # systemd drop-in must still resolve to the one identity.
    assert er._get_machine_name("DLAPTOP-5.local") == "DLAPTOP"
    assert er._get_machine_name("DLAPTOP-4.local") == "DLAPTOP"


def test_get_machine_name_leaves_fleet_override_alone():
    # The hub's `--machine ree-cloud-1` is load-bearing; it must survive verbatim.
    assert er._get_machine_name("ree-cloud-1") == "ree-cloud-1"


def test_affinity_matches_across_the_drift():
    item = {"machine_affinity": "DLAPTOP-4.local"}
    assert er._affinity_matches(item, "DLAPTOP-5.local")


def test_affinity_still_rejects_a_different_cloud_worker():
    item = {"machine_affinity": "ree-cloud-1"}
    assert not er._affinity_matches(item, "ree-cloud-2")


def test_affinity_any_still_matches_everything():
    for affinity in ("any", None, ""):
        assert er._affinity_matches({"machine_affinity": affinity}, "ree-cloud-3")


def test_auto_yield_rearms_on_the_drifted_laptop_name():
    # THE live regression: this was False for three weeks, silently disabling
    # cloud-yield on the laptop.
    assert mi.same_machine("DLAPTOP-5.local", er.LAPTOP_AUTO_YIELD_HOSTNAME)


def test_auto_yield_does_not_arm_on_a_cloud_worker():
    assert not mi.same_machine("ree-cloud-2", er.LAPTOP_AUTO_YIELD_HOSTNAME)
