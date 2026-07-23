"""Contracts pinning that CROSS-DRIVER arm reuse cannot cache-HIT on stale training code.

THE DEFECT THESE DEFEND AGAINST (found 2026-07-22, closed the same day).
`arm_fingerprint.compute_substrate_hash` hashes the content of every file matching
`_SUBSTRATE_GLOBS` -- `ree_core/**`, `experiments/_harness.py`, `experiments/_metrics.py`,
`experiments/_lib/**`. A file under `experiments/` but OUTSIDE `_lib` enters a cell's
substrate_hash by exactly one route: as the calling script's own path, folded in by
`include_driver_script_in_hash`.

But cross-driver arm REUSE *requires* `include_driver_script_in_hash=False` on both the mint
and the consumer -- with it True, two distinct drivers can never hash-match, so nothing is
ever reusable across them. Two such callers exist:
    experiments/v3_exq_742_mech457_actor_critic_onoff.py            (consumer)
    experiments/v3_exq_742m_mech457_bias_head_baseline_mint.py      (mint)
and both executed `_train_all_on_agent`, which used to be defined in the x734 DRIVER file and
called helpers defined in the x724 DRIVER file. Neither file was in any glob, and both were
excluded as own-script by that very flag. Net effect: an edit to the training recipe -- a
learning rate, the e2 contrastive step, either REINFORCE loss -- did not change those cells'
substrate_hash, so a banked arm computed by the OLD code could cache-HIT a consumer running
the NEW code, silently comparing a treatment and a control trained by different code.

`arm_fingerprint`'s own docstring states the governing asymmetry: OVER-inclusion causes false
MISSES only (cheap wasted compute); UNDER-inclusion is a false HIT that corrupts a
conclusion. This was under-inclusion, i.e. the unsafe direction.

THE REMEDY IS STRUCTURAL. The shared computation moved into
`experiments/_lib/allon_training.py`, where the pre-existing `experiments/_lib/**/*.py` glob
covers it and every future caller inherits the coverage. The alternative -- passing an
explicit `scope=` naming the driver -- is per-caller and easy to forget, which is precisely
the failure mode being closed.

WHY A CONTRACT AND NOT JUST THE MOVE. Nothing at runtime distinguishes a correct cache hit
from a stale one: the numbers look ordinary either way. If a future edit moves shared
training code back into a driver file, or has the shared module start delegating to one, the
hole reopens silently. These checks make that visible at commit time.

ASCII-only (repo rule).
"""

import inspect
from pathlib import Path

import pytest

import experiments._lib.allon_training as allon
import experiments._lib.arm_fingerprint as afp
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734

REPO_ROOT = Path(afp.__file__).resolve().parents[2]

# Every driver known to pass include_driver_script_in_hash=False, i.e. every driver whose
# cells rely on the globs ALONE to notice a substrate change. Its own source is excluded by
# construction, so anything it executes must live in a hashed tree.
DRIVER_SCRIPT_EXCLUDED_CALLERS = (
    "experiments/v3_exq_742_mech457_actor_critic_onoff.py",
    "experiments/v3_exq_742m_mech457_bias_head_baseline_mint.py",
)


def _substrate_files() -> frozenset:
    """The exact repo-relative file set `compute_substrate_hash` hashes by default.

    Enumerated with `Path.glob` against `_SUBSTRATE_GLOBS`, the same call that function makes
    -- NOT `fnmatch`, which does not treat `**` as path-recursive and would silently report
    `experiments/_lib/allon_training.py` as uncovered.
    """
    return frozenset(
        str(f.relative_to(REPO_ROOT))
        for g in afp._SUBSTRATE_GLOBS
        for f in REPO_ROOT.glob(g)
        if f.is_file()
    )


def _matches_a_substrate_glob(rel: str) -> bool:
    return rel in _substrate_files()


# --------------------------------------------------------------------------- C1
def test_c1_shared_training_module_is_inside_the_substrate_globs():
    """The shared A0 training toolkit must sit in a tree `compute_substrate_hash` hashes.

    This is the whole fix in one assertion. If someone moves `allon_training.py` back under
    `experiments/` proper, or renames it outside the `_lib` tree, the reuse-eligible cells
    stop noticing training-code edits and this fails.
    """
    rel = str(Path(allon.__file__).resolve().relative_to(REPO_ROOT))
    assert _matches_a_substrate_glob(rel), (
        f"{rel} matches none of _SUBSTRATE_GLOBS {afp._SUBSTRATE_GLOBS}; the training code it "
        "holds would be invisible to substrate_hash for every cell built with "
        "include_driver_script_in_hash=False, which is the false-cache-HIT defect"
    )


# --------------------------------------------------------------------------- C2
def test_c2_the_shared_training_path_executes_no_driver_file_code():
    """Nothing the shared module reaches may be DEFINED in a driver file.

    Living in `_lib` is necessary but not sufficient: a `_lib` module that imports its
    learning rates or its e2 training step back out of `v3_exq_724...` would hash its own
    bytes and still execute unhashed code. That is the exact shape of the original defect one
    level of indirection down, so it is pinned separately.
    """
    offenders = []
    for name, obj in vars(allon).items():
        if name.startswith("__"):
            continue
        mod = getattr(obj, "__module__", None)
        if not isinstance(mod, str):
            continue
        if not mod.startswith("experiments."):
            continue  # ree_core / stdlib / third-party are hashed or irrelevant
        try:
            src_rel = str(Path(inspect.getfile(obj)).resolve().relative_to(REPO_ROOT))
        except (TypeError, OSError, ValueError):
            continue
        if not _matches_a_substrate_glob(src_rel):
            offenders.append((name, src_rel))
    assert not offenders, (
        "experiments/_lib/allon_training.py reaches code defined outside the substrate globs: "
        f"{offenders}. Those bytes are executed by reuse-eligible cells but do not enter "
        "substrate_hash, so an edit to them cannot invalidate a banked arm."
    )


# --------------------------------------------------------------------------- C3
def test_c3_x734_re_exports_rather_than_redefines_the_shared_recipe():
    """`x734._train_all_on_agent` must BE the shared function, not a second copy.

    Five call sites reach the recipe through this attribute path (v3_exq_737, v3_exq_742,
    v3_exq_808, `_lib/mech457_fanout`, `_lib/baselines/exq742_mech457_bias_head_baseline`).
    If x734 ever grows its own definition again, all five silently revert to executing
    unhashed driver-file code while still importing from a module named x734.
    """
    # Compared by defining module, not by object identity: pytest's import mode can load the
    # library module under two names in one session, which makes `is` spuriously false while
    # the property under test -- x734 not defining its own copy -- still holds.
    assert x734._train_all_on_agent.__module__ == "experiments._lib.allon_training", (
        "x734._train_all_on_agent is defined in "
        f"{x734._train_all_on_agent.__module__}, not the shared library module"
    )
    assert "def _train_all_on_agent(" not in Path(x734.__file__).read_text(encoding="utf-8"), (
        "x734 has grown its own `_train_all_on_agent` definition again; its bytes are in no "
        "substrate glob, so cells using include_driver_script_in_hash=False would stop "
        "noticing edits to the training recipe"
    )


# --------------------------------------------------------------------------- C4
@pytest.mark.parametrize("rel", DRIVER_SCRIPT_EXCLUDED_CALLERS)
def test_c4_reuse_callers_still_declare_the_flag_that_makes_this_load_bearing(rel):
    """Pins WHY C1-C3 matter: these callers exclude their own script from the hash.

    If a caller flips back to the default True, cross-driver reuse silently stops working
    (every lookup misses) -- a different bug, but one worth surfacing here, because it also
    removes the justification a reader would otherwise use to relax C1-C3.
    """
    src = (REPO_ROOT / rel).read_text(encoding="utf-8")
    assert "include_driver_script_in_hash=False" in src, (
        f"{rel} no longer passes include_driver_script_in_hash=False; cross-driver arm reuse "
        "with its counterpart cannot hash-match, so no lookup will ever hit"
    )


# --------------------------------------------------------------------------- C5
def test_c5_the_lib_glob_actually_covers_a_new_lib_file(tmp_path):
    """Behavioural anchor on the glob itself, run on a synthetic tree.

    C1 compares a path against `_SUBSTRATE_GLOBS` textually. This confirms the globs are also
    applied that way by `compute_substrate_hash`, so C1 cannot pass against a hashing routine
    that has stopped honouring the `experiments/_lib/**` entry. Deliberately synthetic: it
    must not hash the real repo, which other sessions are concurrently editing.
    """
    lib = tmp_path / "experiments" / "_lib"
    lib.mkdir(parents=True)
    target = lib / "allon_training.py"
    target.write_text("LR = 1e-3\n", encoding="utf-8")
    before = afp.compute_substrate_hash(repo_root=tmp_path)
    assert before["n_files"] >= 1, "the _lib glob matched nothing in the synthetic tree"

    target.write_text("LR = 2e-3\n", encoding="utf-8")
    after = afp.compute_substrate_hash(repo_root=tmp_path)
    assert after["substrate_hash"] != before["substrate_hash"], (
        "editing a file under experiments/_lib/ did not change the substrate hash; the "
        "structural fix for the false-cache-HIT defect rests on it doing so"
    )


# --------------------------------------------------------------------------- C6
def test_c6_source_is_ascii():
    for mod in (allon,):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        bad = [(i, ch) for i, ch in enumerate(src) if ord(ch) > 127]
        assert not bad, f"non-ASCII in {mod.__name__}: {bad[:5]}"
