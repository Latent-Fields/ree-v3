"""Contracts for the torch component of machine_class() (plan sec 7b/9, added 2026-07-19).

THE HAZARD THIS CLOSES. machine_class() used to be "{system}-{arch}-py{major}.{minor}".
Upgrading torch on the cloud fleet leaves python at 3.10, so the tag stayed BYTE-IDENTICAL
across the upgrade while float behaviour underneath changed. All 1170 banked linux
fingerprints would have remained matchable and a post-upgrade consumer would have compared
new-torch treatment arms against old-torch baselines -- no cache miss, no warning. That is
the false HIT the design exists to prevent (plan sec 2: a false hit corrupts a conclusion,
a false miss only wastes compute).

Invariants asserted:
  (1) The torch version is IN the tag.
  (2) The tag DISCRIMINATES on torch -- two otherwise-identical hosts differing only in
      torch get different classes, and therefore different fingerprints.
  (3) A torchless host gets its own reserved class, never silently joining a torch-bearing
      one.
  (4) torch_version is recorded on the emitted payload as observability (the field whose
      absence made the 2026-07-19 cut unmigratable).
  (5) IMPORTABILITY: importing arm_fingerprint must NOT import torch. manifest_core.py
      documents this module as safe to import without torch/ree_core; the torch lookup is
      lazy + memoised to preserve that.

Design plan: REE_assembly/evidence/planning/arm_reuse_fingerprint_plan.md sec 7b/9.
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
for _p in (str(REPO_ROOT), str(REPO_ROOT / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _lib import arm_fingerprint as A  # noqa: E402


def _fp(**kw):
    base = dict(config_slice={"lr": 0.01}, seed=42, rng_fully_reset=True)
    base.update(kw)
    return A.compute_arm_fingerprint(**base)


def _with_torch_tag(tag, fn):
    """Run fn() with the memoised torch tag forced to `tag`, then restore."""
    saved = A._TORCH_TAG
    try:
        A._TORCH_TAG = tag
        return fn()
    finally:
        A._TORCH_TAG = saved


def test_torch_version_is_in_the_tag():
    """(1) The tag names the torch build, not just OS/arch/python."""
    tag = _with_torch_tag("2.5.1+cu121", A.machine_class)
    assert tag.endswith("-torch2.5.1+cu121"), tag
    assert "py{}.{}".format(*sys.version_info[:2]) in tag


def test_tag_discriminates_on_torch_version():
    """(2) THE REGRESSION GUARD. Same OS/arch/python, different torch -> different class.

    This is the exact scenario that was silent before: py3.10 either side of a fleet
    upgrade. If this ever fails, a torch upgrade is once again invisible to the cache.
    """
    old = _with_torch_tag("2.5.1+cu121", A.machine_class)
    new = _with_torch_tag("2.10.0+cu124", A.machine_class)
    assert old != new

    # ...and the discrimination must propagate to the fingerprint itself, which is what
    # the reuse lookup actually keys on.
    fp_old = _with_torch_tag("2.5.1+cu121", lambda: _fp()["arm_fingerprint"])
    fp_new = _with_torch_tag("2.10.0+cu124", lambda: _fp()["arm_fingerprint"])
    assert fp_old != fp_new


def test_cuda_build_change_discriminates():
    """(2b) The local version segment counts too -- a CUDA-build swap is a new class.
    Over-inclusion is the safe direction (false miss only)."""
    cpu = _with_torch_tag("2.5.1", A.machine_class)
    cu = _with_torch_tag("2.5.1+cu121", A.machine_class)
    assert cpu != cu


def test_torchless_host_gets_reserved_class():
    """(3) No torch -> a reserved token that cannot collide with any real version."""
    tag = _with_torch_tag(A.TORCH_ABSENT_TAG, A.machine_class)
    assert tag.endswith("-torch" + A.TORCH_ABSENT_TAG)
    assert tag != _with_torch_tag("2.5.1+cu121", A.machine_class)


def test_payload_records_torch_version_separately():
    """(4) Observability field, so a future miss is triageable rather than mysterious."""
    payload = _with_torch_tag("2.5.1+cu121", _fp)
    assert payload["torch_version"] == "2.5.1+cu121"
    assert payload["machine_class"].endswith("-torch2.5.1+cu121")


def test_importing_arm_fingerprint_does_not_import_torch():
    """(5) Module import stays stdlib-only; only CALLING machine_class() resolves torch.

    Runs in a subprocess because torch is already resident in this test process.
    """
    code = (
        "import sys; sys.path.insert(0, %r);"
        "import arm_fingerprint as A;"
        "assert 'torch' not in sys.modules, 'import pulled torch in';"
        "A.machine_class();"
        "print('ok')" % str(REPO_ROOT / "experiments" / "_lib")
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout
