"""Run an experiment driver against a HISTORICAL `ree_core` substrate commit.

WHY THIS EXISTS
===============
`/failure-autopsy` sometimes routes a fan-out leg whose whole hypothesis IS the
substrate delta between two already-executed runs ("did the collapse come from
the code changing, or from the intervention?"). The fleet runner always executes
the driver against the CURRENT checkout, so there is no other way to put that
question to an experiment. Canonical first consumer: the GOV-FANOUT-1 portfolio
for `inv050_mech180_861e_producer_vs_intervention_isolation`
(`failure_autopsy_V3-EXQ-861e_2026-08-21`), whose H3 leg reads verbatim
"Pin 861c substrate (f810969) and rerun the 861e protocol".

WHAT IT DOES
============
`pin_ree_core(ref)` extracts ONLY `ree_core/` at `ref` (via `git archive`, a
READ-ONLY plumbing command -- never `worktree add`, which mutates `.git` and can
race the runner's own `git pull --rebase`) into a per-ref cache directory, then
puts that directory FIRST on `sys.path`. `import ree_core` then resolves to the
pinned tree; `experiments.*`, `experiment_protocol` and `pack_writer` continue to
resolve from the live checkout, because the extraction deliberately contains NO
`experiments/` directory for them to shadow.

MUST BE CALLED BEFORE THE FIRST `import ree_core` IN THE DRIVER. A driver that
imports `ree_core` first gets the live substrate and the pin silently does
nothing -- which is exactly the verdict-aliasing failure a pinned leg exists to
avoid. `verify_pin()` therefore HARD-FAILS rather than warning; see below.

VERIFICATION IS NOT OPTIONAL
===========================
Two independent checks, both fatal:

  (1) STRUCTURAL -- `ree_core.__file__` must live under the pin directory.
  (2) BEHAVIOURAL -- a caller-supplied source marker (a symbol that exists on
      exactly one side of the pinned/live boundary) must be present-or-absent as
      declared. A structural check alone cannot catch a stale cache directory
      holding the WRONG ref's content; the marker can.

A failed check raises `SubstratePinError`, which propagates as a non-zero exit
and is classified ERROR by the runner. That is the correct outcome: a leg that
cannot prove which substrate it ran is worse than a leg that did not run.

FINGERPRINT / PROVENANCE CONTRACT
=================================
A pinned cell's `arm_fingerprint` MUST be computed with
`repo_root=<pin dir>` and `substrate_scope=("ree_core/**/*.py",)` so the recorded
`substrate_hash` describes the code that ACTUALLY EXECUTED, not the checkout.
Recording the checkout's hash for a pinned run would be false provenance.

That scope is deliberately NARROWER than `_SUBSTRATE_GLOBS` (it omits
`experiments/_harness.py`, `_metrics.py`, `_lib/**`, which really are read from
the LIVE checkout), so it is NOT a conservative superset and the cell is NOT
reuse-safe. `pin_fingerprint_kwargs()` therefore always ships
`extra_ineligible_reasons=["substrate_pinned_to_historical_commit"]`. Never
strip it: a pinned cell must never be minted as a reusable baseline, because a
future consumer running on trunk would silently inherit historical substrate.

ASCII-only output (CLAUDE.md). No network. Never writes inside the repo.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

__all__ = [
    "SubstratePinError",
    "pin_ree_core",
    "verify_pin",
    "pin_fingerprint_kwargs",
    "pin_manifest_block",
]

REPO_ROOT = Path(__file__).resolve().parents[2]   # ree-v3/
PIN_INELIGIBLE_REASON = "substrate_pinned_to_historical_commit"
PIN_SUBSTRATE_SCOPE: Sequence[str] = ("ree_core/**/*.py",)


class SubstratePinError(RuntimeError):
    """The pin could not be established or could not be proven. Always fatal."""


def _git(args, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True, text=True, timeout=180,
    )


def _cache_root() -> Path:
    """Per-user scratch, OUTSIDE the repo (writing inside it dirties the tree
    and wedges every phase3 writer -- CLAUDE.md 'Hub telemetry git path')."""
    env = os.environ.get("REE_SUBSTRATE_PIN_CACHE")
    if env:
        return Path(env)
    return Path(tempfile.gettempdir()) / "ree_substrate_pins"


def pin_ree_core(ref: str, *, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Materialise `ree_core/` at `ref` and put it first on sys.path.

    Returns a provenance dict (also the input to `pin_manifest_block`). Raises
    `SubstratePinError` on any failure -- there is no degraded mode.
    """
    repo = Path(repo_root or REPO_ROOT).resolve()
    rp = _git(["rev-parse", "--verify", f"{ref}^{{commit}}"], repo)
    if rp.returncode != 0:
        raise SubstratePinError(
            f"cannot resolve substrate pin ref {ref!r} in {repo}: "
            f"{(rp.stderr or '').strip()}"
        )
    sha = rp.stdout.strip()

    head = _git(["rev-parse", "HEAD"], repo)
    head_sha = head.stdout.strip() if head.returncode == 0 else "unknown"

    dest = _cache_root() / sha
    marker = dest / ".ree_pin_complete"
    if not marker.exists():
        # Rebuild from scratch: a partial extraction left by an earlier crash
        # would otherwise look usable and silently under-populate the tree.
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        # Stage INSIDE the cache root so the final os.replace() is same-filesystem
        # (a cross-device rename raises OSError, not a fallback).
        dest.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="ree_pin_staging_", dir=str(dest.parent)))
        try:
            ar = subprocess.run(
                ["git", "-C", str(repo), "archive", "--format=tar", sha, "ree_core"],
                capture_output=True, timeout=600,
            )
            if ar.returncode != 0:
                raise SubstratePinError(
                    f"git archive {sha} ree_core failed: "
                    f"{(ar.stderr or b'').decode('ascii', 'replace').strip()}"
                )
            tr = subprocess.run(["tar", "-x", "-C", str(staging)],
                                input=ar.stdout, capture_output=True, timeout=600)
            if tr.returncode != 0:
                raise SubstratePinError(
                    "tar extract of pinned ree_core failed: "
                    f"{(tr.stderr or b'').decode('ascii', 'replace').strip()}"
                )
            if not (staging / "ree_core" / "__init__.py").is_file():
                raise SubstratePinError(
                    f"pinned tree at {sha} has no ree_core/__init__.py")
            (staging / ".ree_pin_complete").write_text(sha + "\n")
            os.replace(str(staging), str(dest))   # atomic within one filesystem
            staging = None
        finally:
            if staging is not None:
                shutil.rmtree(str(staging), ignore_errors=True)

    if not (dest / "ree_core" / "__init__.py").is_file():
        raise SubstratePinError(f"pin cache {dest} is not a usable ree_core tree")

    # There must be NO experiments/ here -- see the module docstring: the live
    # harness must keep resolving from the checkout.
    if (dest / "experiments").exists():
        raise SubstratePinError(
            f"pin cache {dest} unexpectedly contains experiments/; it would "
            "shadow the live recording harness")

    if "ree_core" in sys.modules:
        raise SubstratePinError(
            "ree_core was already imported before pin_ree_core() ran -- the pin "
            "would be a silent no-op. Move the pin call above every ree_core "
            "import in the driver.")

    sys.path.insert(0, str(dest))
    return {
        "mode": "git_archive_ree_core_only",
        "requested_ref": ref,
        "resolved_sha": sha,
        "pin_dir": str(dest),
        "checkout_head_sha": head_sha,
        "substrate_scope": list(PIN_SUBSTRATE_SCOPE),
        "reuse_ineligible_reason": PIN_INELIGIBLE_REASON,
        "verified": False,
    }


def verify_pin(pin: Dict[str, Any], *, marker_module: str, marker_attr: str,
               marker_expected_present: bool) -> Dict[str, Any]:
    """Prove the pin took. Mutates and returns `pin`. Fatal on mismatch.

    `marker_*` names a symbol whose presence DIFFERS between the pinned ref and
    the live checkout, so it discriminates a real pin from a no-op or a stale
    cache. Pick it from the actual diff, not by guessing.
    """
    import importlib
    import ree_core   # noqa: F401  -- resolves through the pinned sys.path entry

    pin_dir = str(Path(pin["pin_dir"]).resolve())
    actual = str(Path(ree_core.__file__).resolve())
    if not actual.startswith(pin_dir + os.sep):
        raise SubstratePinError(
            f"STRUCTURAL pin check FAILED: ree_core resolved to {actual}, "
            f"expected a path under {pin_dir}")

    mod = importlib.import_module(marker_module)
    mod_file = str(Path(mod.__file__).resolve())
    if not mod_file.startswith(pin_dir + os.sep):
        raise SubstratePinError(
            f"STRUCTURAL pin check FAILED: {marker_module} resolved to "
            f"{mod_file}, expected a path under {pin_dir}")
    present = hasattr(mod, marker_attr)
    if present is not bool(marker_expected_present):
        raise SubstratePinError(
            f"BEHAVIOURAL pin check FAILED: {marker_module}.{marker_attr} "
            f"present={present}, expected present={bool(marker_expected_present)} "
            f"for ref {pin['requested_ref']} ({pin['resolved_sha'][:10]}). The "
            "pin directory may hold the wrong ref's content.")

    pin["verified"] = True
    pin["marker"] = {
        "module": marker_module,
        "attr": marker_attr,
        "expected_present": bool(marker_expected_present),
        "observed_present": bool(present),
    }
    pin["ree_core_file"] = actual
    return pin


def pin_fingerprint_kwargs(pin: Dict[str, Any],
                           extra_reasons: Optional[Sequence[str]] = None
                           ) -> Dict[str, Any]:
    """kwargs for `arm_cell` / `compute_arm_fingerprint` on a pinned cell.

    Always carries the reuse-ineligibility reason -- see the module docstring.
    """
    reasons = [PIN_INELIGIBLE_REASON, *(extra_reasons or ())]
    return {
        "repo_root": Path(pin["pin_dir"]),
        "substrate_scope": tuple(PIN_SUBSTRATE_SCOPE),
        "extra_ineligible_reasons": reasons,
    }


def pin_manifest_block(pin: Dict[str, Any]) -> Dict[str, Any]:
    """The provenance block to record verbatim in the manifest."""
    block = dict(pin)
    block["note"] = (
        "ree_core/ was executed from the pinned ref, NOT from the checkout at "
        "checkout_head_sha. experiments/_lib/**, experiment_protocol and "
        "pack_writer came from the LIVE checkout. Per-cell substrate_hash is "
        "scoped to ree_core/**/*.py under pin_dir, so it describes the code "
        "that actually ran. All pinned cells are reuse-INELIGIBLE."
    )
    return block
