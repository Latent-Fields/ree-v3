"""Contracts for the SD-081 spearman-guard-shape lint.

Surfaces under test:
  (1) validate_experiments.spearman_guard_shape_lint -- flags a hand-rolled rank
      correlation that computes ORDINAL ranks (np.argsort(np.argsort(...)) or
      sorted(range(...), key=...)) and guards degeneracy on the variance of the RANK
      vector rather than the INPUT vector, i.e. the "19th copy" of the defect.
  (2) validate_experiments.py --checks spearman_guard_shape -- the selector, and the
      invariant that this gate is WARN-ONLY IN BOTH MODES (never hardens under --paths,
      never affects the exit code even under --strict).

WHY THIS GATE EXISTS. Prior to SD-081, 18 experiment scripts each carried a `_spearman*`
copy that guarded on `np.std(np.argsort(np.argsort(a))) == 0`. Double-argsort of a
CONSTANT input returns a permutation of 0..K-1 whose std is large (9.23 at K=32), not 0 --
so the guard NEVER fires, and Spearman is computed against an arbitrary stable-sort
tie-break ordering (deterministic noise; confirmed |rho| up to 0.74 on genuinely constant
vectors). The 18 copies were migrated to the canonical helper
experiments/_lib/stats.spearman (input-guarded + average-ranks ties), pinned by
tests/contracts/test_spearman_input_guard.py. This lint closes the SHAPE going forward.

THE DISCRIMINATOR against the 16 already-safe average-rank helpers in the corpus is that
they AVERAGE-RANK ties (a constant input -> all-equal ranks -> genuine 0 rank-variance),
which fixes the same bug structurally. The lint must NOT flag them -- see the safe cases
and the real-corpus witnesses below. Full autopsy:
REE_assembly/evidence/planning/failure_autopsy_sd081-spearman-degenerate-dv_2026-07-27.md.

ASCII-only. Run: pytest tests/contracts/test_spearman_guard_shape_lint.py -q
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import validate_experiments as V  # noqa: E402

EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def _run(*args):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "validate_experiments.py"), *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT))


def _lint_src(src: str):
    """Lint a synthetic script written into experiments/ (so path scoping holds)."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(src)
        name = f.name
    try:
        return V.spearman_guard_shape_lint(Path(name))
    finally:
        os.unlink(name)


# ---- (1) the defective shapes (the three that existed in the corpus) -----------------

# 543 family / most copies: double-argsort + `rx.std() < 1e-9` guard on the RANK vector.
_DEFECTIVE_DOUBLE_ARGSORT = '''
import numpy as np
from typing import List
def _spearman_rho(x: List[float], y: List[float]) -> float:
    if len(x) < 4 or len(y) < 4:
        return 0.0
    rx = np.argsort(np.argsort(np.asarray(x, dtype=np.float64)))
    ry = np.argsort(np.argsort(np.asarray(y, dtype=np.float64)))
    if rx.std() < 1e-9 or ry.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])
'''

# 786 shape: double-argsort + `float(np.std(ra)) == 0.0` (numeric-cast-wrapped std).
_DEFECTIVE_DOUBLE_ARGSORT_EQ0 = '''
import numpy as np
from typing import List, Optional
def _spearman(a: List[float], b: List[float]) -> Optional[float]:
    n = len(a)
    if n < 2 or len(b) != n:
        return None
    ra = np.argsort(np.argsort(np.asarray(a, dtype=float))).astype(float)
    rb = np.argsort(np.argsort(np.asarray(b, dtype=float))).astype(float)
    if float(np.std(ra)) == 0.0 or float(np.std(rb)) == 0.0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])
'''

# 207/208/210 shape: sorted(range,key=) ORDINAL rank (rank_val+1, no tie averaging) +
# a manual Pearson denominator guard `den_a < 1e-12` on the RANK vector.
_DEFECTIVE_SORTED_RANGE = '''
import math
from typing import List
def _spearman_r(a: List[float], b: List[float]) -> float:
    n = len(a)
    if n < 3:
        return 0.0
    def _rank(lst: List[float]) -> List[float]:
        sorted_idx = sorted(range(n), key=lambda i: lst[i])
        ranks = [0.0] * n
        for rank_val, idx in enumerate(sorted_idx):
            ranks[idx] = float(rank_val + 1)
        return ranks
    ra = _rank(a); rb = _rank(b)
    mean_ra = sum(ra) / n; mean_rb = sum(rb) / n
    num = sum((ra[i] - mean_ra) * (rb[i] - mean_rb) for i in range(n))
    den_a = math.sqrt(sum((ra[i] - mean_ra) ** 2 for i in range(n)))
    den_b = math.sqrt(sum((rb[i] - mean_rb) ** 2 for i in range(n)))
    if den_a < 1e-12 or den_b < 1e-12:
        return 0.0
    return num / (den_a * den_b)
'''


def test_flags_double_argsort_rank_std_guard():
    out = _lint_src(_DEFECTIVE_DOUBLE_ARGSORT)
    assert out is not None, "double-argsort + rank-std guard must be flagged"
    assert "_spearman_rho" in out, out
    assert "RANK vector rather than the INPUT vector" in out, out


def test_flags_double_argsort_std_eq_zero_guard():
    """The `float(np.std(ra)) == 0.0` (786) spelling -- the numeric cast must not hide
    the std call from the guard-subject search."""
    out = _lint_src(_DEFECTIVE_DOUBLE_ARGSORT_EQ0)
    assert out is not None and "_spearman" in out, out


def test_flags_sorted_range_ordinal_denominator_guard():
    """The sorted(range,key=) ordinal shape (207 family) with a manual `den < 1e-12`
    guard on the ranks. The ordinal construct is in the NESTED _rank; ast.walk over the
    outer function sees it."""
    out = _lint_src(_DEFECTIVE_SORTED_RANGE)
    assert out is not None and "_spearman_r" in out, out


# ---- (2) the safe average-rank helpers must NOT be flagged ---------------------------

# The 785/743 shape: sorted(range,key=) for the sort ORDER, then a tie-run equality check
# `v[order[j+1]] == v[order[i]]` + midrank average. Same manual-denominator guard as the
# defective 207 shape -- so the ONLY thing that saves it is the tie-averaging.
_SAFE_AVG_SORTED_RANGE = '''
import math
from typing import List
def _spearman_rho(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    def _rank(v: List[float]) -> List[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = _rank(x), _rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx <= 0.0 or dy <= 0.0:
        return 0.0
    return float(num / (dx * dy))
'''

# The 818 shape: single argsort + arange + np.mean tie averaging, guarded on the RANK
# vector (`np.std(rx) < 1e-12`) -- safe because the ranks are average-ranked. It uses
# neither double-argsort nor sorted(range,key=), so it never reaches the correlation gate.
_SAFE_AVG_ARGSORT_MEAN = '''
import numpy as np
from typing import List
def _spearman(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 2 or len(y) != n:
        return 0.0
    def _rank(v: List[float]) -> np.ndarray:
        a = np.asarray(v, dtype=float)
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(n, dtype=float)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and a[order[j + 1]] == a[order[i]]:
                j += 1
            if j > i:
                ranks[order[i:j + 1]] = np.mean(ranks[order[i:j + 1]])
            i = j + 1
        return ranks
    rx, ry = _rank(x), _rank(y)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])
'''

# The migrated shape: a thin wrapper over the canonical helper -- no ordinal construct.
_SAFE_CANONICAL_IMPORT = '''
from typing import List
from experiments._lib.stats import spearman as _spearman_canonical
def _spearman_rho(x: List[float], y: List[float]) -> float:
    if len(x) < 4 or len(y) < 4:
        return 0.0
    rho = _spearman_canonical(x, y)
    return 0.0 if rho is None else rho
'''

# An input-guarded ordinal helper: uses double-argsort but guards the INPUT vector
# (`len(set(x)) < 2`). The SD-081 defect (constant input past the guard) is closed, so
# it must NOT fire -- this is the "rather than the INPUT vector" clause.
_SAFE_INPUT_GUARDED_ORDINAL = '''
import numpy as np
from typing import List, Optional
def _spearman(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 2 or len(y) != len(x):
        return None
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None
    rx = np.argsort(np.argsort(np.asarray(x, dtype=float)))
    ry = np.argsort(np.argsort(np.asarray(y, dtype=float)))
    return float(np.corrcoef(rx, ry)[0, 1])
'''

# sorted(range,key=) used for TOP-K selection -- no correlation, must not fire.
_SAFE_TOPK = '''
from typing import List
def _top_k(scores: List[float], k: int) -> List[int]:
    idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return idx[:k]
'''


def test_safe_average_rank_sorted_range_not_flagged():
    """THE load-bearing negative: the 16-safe-helper representative. Same ordinal
    sort-order construct AND same denominator guard as the defective 207 shape -- saved
    only by average-ranking ties."""
    assert _lint_src(_SAFE_AVG_SORTED_RANGE) is None


def test_safe_average_rank_argsort_mean_not_flagged():
    assert _lint_src(_SAFE_AVG_ARGSORT_MEAN) is None


def test_safe_canonical_import_wrapper_not_flagged():
    assert _lint_src(_SAFE_CANONICAL_IMPORT) is None


def test_safe_input_guarded_ordinal_not_flagged():
    """An ordinal helper that guards the INPUT vector has closed the SD-081 defect."""
    assert _lint_src(_SAFE_INPUT_GUARDED_ORDINAL) is None


def test_safe_topk_sorted_range_not_flagged():
    """sorted(range,key=) is a general idiom; only a rank CORRELATION is in scope."""
    assert _lint_src(_SAFE_TOPK) is None


# ---- (3) exemption + robustness -----------------------------------------------------

def test_explicit_opt_out_is_honoured():
    src = ('SPEARMAN_GUARD_SHAPE_EXEMPT = "diagnostic probe of the collapse"\n'
           + _DEFECTIVE_DOUBLE_ARGSORT)
    assert _lint_src(src) is None


def test_per_function_scoping_ignores_unrelated_eq_subscript():
    """The averaging discriminator (an Eq over subscripts) is evaluated PER FUNCTION.
    An unrelated `cfg[a] == cfg[b]` in a DIFFERENT function must not mask the defect."""
    src = _DEFECTIVE_DOUBLE_ARGSORT + '''
def _unrelated(cfg, a, b):
    if cfg[a] == cfg[b]:
        return 1
    return 0
'''
    out = _lint_src(src)
    assert out is not None and "_spearman_rho" in out, out


def test_message_names_the_canonical_helper():
    out = _lint_src(_DEFECTIVE_DOUBLE_ARGSORT)
    assert "from experiments._lib.stats import spearman" in out, out
    assert "SPEARMAN_GUARD_SHAPE_EXEMPT" in out, out


# ---- (4) selector + WARN-only invariant ---------------------------------------------

def test_selector_runs_only_this_check():
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(_DEFECTIVE_DOUBLE_ARGSORT)
        name = f.name
    try:
        r = _run("--checks", "spearman_guard_shape", "--quiet", "--paths", name)
        assert r.returncode == 0, r.stdout[-2000:]
        assert "spearman-guard-shape-warning(s)" in r.stdout, r.stdout[-2000:]
        assert "SPEARMAN-GUARD-SHAPE WARNINGS" in r.stdout, r.stdout[-2000:]
    finally:
        os.unlink(name)


def test_is_warn_only_under_strict_and_paths():
    """INVARIANT: never blocks, even under --strict --paths (like the other advisory
    *-warning gates). A fire is a triage signal, not a commit gate."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(_DEFECTIVE_DOUBLE_ARGSORT)
        name = f.name
    try:
        r = _run("--checks", "spearman_guard_shape", "--quiet", "--strict",
                 "--paths", name)
        assert r.returncode == 0, r.stdout[-2000:]
        assert "SPEARMAN-GUARD-SHAPE WARNINGS" in r.stdout, r.stdout[-2000:]
    finally:
        os.unlink(name)


def test_check_is_registered_in_check_names():
    assert "spearman_guard_shape" in V.CHECK_NAMES


# ---- (5) real-corpus witnesses ------------------------------------------------------

def test_lib_helper_is_excluded():
    """The canonical helper lives under experiments/_lib/ and is excluded by contract."""
    lib = EXPERIMENTS_DIR / "_lib" / "stats.py"
    if lib.exists():
        assert V.spearman_guard_shape_lint(lib) is None


def test_real_safe_average_rank_helpers_not_flagged():
    """Named real safe average-rank helpers must stay clean -- otherwise the WARN is
    unactionable. These use sorted(range,key=) + tie-averaging (the safe 16)."""
    for fname in (
        "v3_exq_785_mech463_arousal_variance_amplifier_decomp.py",
        "v3_exq_743_inv089_harm_evaluator_z_harm_bounded.py",
        "v3_exq_818_arc016_eval_derived_noise_precision_sweep.py",
    ):
        real = EXPERIMENTS_DIR / fname
        if real.exists():
            assert V.spearman_guard_shape_lint(real) is None, fname


def test_corpus_has_zero_fires():
    """All 18 defective copies were migrated to the canonical helper, so the live corpus
    must carry ZERO of this shape. A rise means a new hand-rolled defective helper landed
    (the thing this gate exists to catch) -- or a real false positive to triage."""
    fires = [p.name for p in sorted(EXPERIMENTS_DIR.glob("v3_exq_*.py"))
             if V.spearman_guard_shape_lint(p) is not None]
    assert fires == [], f"unexpected spearman-guard-shape fires: {fires}"
