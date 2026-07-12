"""Shared conservatism guards for author-declared substrate scopes (plan sec 11).

Design ref: REE_assembly/evidence/planning/arm_reuse_fingerprint_plan.md sec 11
("Dependency-scoped substrate hashing").

WHY THIS MODULE EXISTS
----------------------
`arm_fingerprint.compute_substrate_hash(scope=...)` lets a cell hash ONLY a declared
subset of the substrate instead of the whole `ree_core/**` + `experiments/_lib/**`
trees, turning the majority of unrelated `ree_core` churn from a (cheap but real)
cache-bust into a legitimate hit. That narrowing is sound ONLY under the plan's
governing asymmetry (sec 2): a false cache MISS wastes compute, a false cache HIT
corrupts a scientific conclusion. So a declared scope is safe **iff it is a provable
SUPERSET of every file the cell can execute or read a constant from** -- an
UNDER-approximation (omitting an executed/read file) is a false-HIT bug.

This module holds the machinery that lets an experiment PROVE its declared scope is
such an over-approximation, so any experiment (not just the maturation-curriculum
prefix cache that prototyped it) can run the same two guards before trusting a scope:

  guard 1 (call-trace, `traced_execution_files` / `verify_scope_conservatism` with
           `run_once`): run a real (cheap) cell under `sys.settrace` and assert EVERY
           repo file whose code executed is inside the declared scope. Closes the
           code-execution channel.
  guard 2 (static AST, `verify_scope_static` / `static_data_closure`): assert the
           declared scope is a FIXPOINT of the data-closure operator -- i.e. it also
           hashes every module a scope file value-imports a module-level CONSTANT
           from (following re-exports to the leaf). Class/function imports of
           NON-executed modules are EXCLUDED (guard 1 proves their bodies never run,
           so they cannot change a deterministic result). Closes the data-read
           channel. Cheap (no cell execution) -> safe to run in CI + opt-in at key
           time.

A refactor that moves executed code out of a declared file, or adds a constant
value-import from outside the scope, trips a guard LOUDLY rather than silently
under-approximating.

SCOPE FORMAT
------------
A `scope` is a sequence of repo-root-relative globs (relative to the ree-v3 repo
root). Exact one-file paths are valid single-match globs (as the maturation legs use);
wildcard globs (e.g. `ree_core/latent/**/*.py`) are expanded against the tree. This
matches `compute_substrate_hash`'s own glob semantics, so the set the guards reason
about is exactly the set that gets hashed.

ASCII-only output (repo rule). Stdlib only (ast + pathlib + sys) -- importable without
ree_core / numpy / torch, so the static guard 2 runs anywhere.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

# Repo root: this file is ree-v3/experiments/_lib/substrate_scope_guard.py, so
# parents[2] == ree-v3. Matches arm_fingerprint._REPO_ROOT and the maturation module.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Repo-module prefixes recognised as in-tree (everything else is stdlib / 3rd-party
# and never enters a data-closure). Mirrors compute_substrate_hash's substrate trees.
_REPO_PREFIXES: Tuple[str, ...] = ("ree_core", "experiments", "_harness", "_lib", "_metrics")

# Memoised across calls -- pure functions of (dirty-tree) file content. Keeps guard 2
# from re-parsing large modules (e.g. agent.py) on every fixpoint step / call.
_MOD_RELPATH_CACHE: Dict[str, Optional[str]] = {}
_TREE_CACHE: Dict[str, ast.AST] = {}
_IMPORTS_CACHE: Dict[str, List[Tuple[str, Optional[str]]]] = {}

_GLOB_METACHARS = ("*", "?", "[")


def _is_glob(g: str) -> bool:
    return any(c in g for c in _GLOB_METACHARS)


def expand_scope(scope: Sequence[str]) -> List[str]:
    """Expand a scope (repo-relative globs) to the sorted set of concrete .py files it
    matches, relative to the repo root. Exact paths that exist map to themselves; a
    wildcard contributes each of its matches. Order-stable (sorted)."""
    files: Set[str] = set()
    for g in scope:
        for p in _REPO_ROOT.glob(g):
            if p.is_file():
                try:
                    files.add(str(p.relative_to(_REPO_ROOT)))
                except ValueError:
                    pass
    return sorted(files)


def _mod_to_relpath(mod: str) -> Optional[str]:
    """Resolve a dotted repo module to its repo-relative .py path (or None). Memoised."""
    if mod in _MOD_RELPATH_CACHE:
        return _MOD_RELPATH_CACHE[mod]
    rel: Optional[str] = None
    if _is_repo_module(mod):
        dotted = "/".join(mod.split("."))
        for cand in (dotted + ".py", dotted + "/__init__.py",
                     "experiments/" + dotted + ".py", "experiments/" + dotted + "/__init__.py"):
            if (_REPO_ROOT / cand).is_file():
                rel = cand
                break
    _MOD_RELPATH_CACHE[mod] = rel
    return rel


def _is_repo_module(mod: str) -> bool:
    return (mod.startswith("ree_core") or mod.startswith("experiments")
            or mod == "_harness" or mod.startswith("_lib") or mod.startswith("_metrics"))


def _tree(rel: str) -> ast.AST:
    if rel not in _TREE_CACHE:
        _TREE_CACHE[rel] = ast.parse((_REPO_ROOT / rel).read_text())
    return _TREE_CACHE[rel]


def _file_imports(rel: str) -> List[Tuple[str, Optional[str]]]:
    """All repo-module imports in `rel` (top-level AND function-local) as (module, name)
    pairs; `name=None` marks a bare `import mod` / `from pkg import submod` / star. Walked
    ONCE per file and cached -- this is the expensive step for large modules."""
    if rel in _IMPORTS_CACHE:
        return _IMPORTS_CACHE[rel]
    recs: List[Tuple[str, Optional[str]]] = []
    for node in ast.walk(_tree(rel)):
        if isinstance(node, ast.Import):
            for a in node.names:
                if _mod_to_relpath(a.name):
                    recs.append((a.name, None))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if not _is_repo_module(node.module):
                continue
            for a in node.names:
                if a.name == "*":
                    recs.append((node.module, None))
                elif _mod_to_relpath(node.module + "." + a.name):
                    recs.append((node.module + "." + a.name, None))  # submodule import
                else:
                    recs.append((node.module, a.name))
    _IMPORTS_CACHE[rel] = recs
    return recs


def _name_kind(rel: str, name: str):
    """Classify a top-level `name` in file `rel`:
      ('code',)                 -- ClassDef/FunctionDef (safe: trace proves uncalled)
      ('data',)                 -- module-level assignment (a data value-import channel)
      ('reexport', mod, name)   -- re-imported from another module (follow to leaf)
      ('unknown',)              -- not found at top level -> treated as data (conservative)
    """
    for node in _tree(rel).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                return ("code",)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for a in node.names:
                if (a.asname or a.name) == name:
                    return ("reexport", node.module, a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if (a.asname or a.name.split(".")[0]) == name:
                    return ("reexport", a.name, None)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            tgts = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in tgts:
                if isinstance(t, ast.Name) and t.id == name:
                    return ("data",)
    return ("unknown",)


def _leaf_is_data(mod: Optional[str], name: Optional[str], chain: List[str]) -> Tuple[bool, List[str]]:
    """Follow a `from mod import name` re-export chain to its leaf; classify the leaf.

    Returns (is_data, chain_of_relpaths). A bare module (`name is None`) is treated as a
    data channel (attribute reads cannot be ruled out statically).
    """
    rel = _mod_to_relpath(mod) if mod else None
    if rel is None:
        return (False, chain)  # not a repo module (stdlib / 3rd-party)
    chain = chain + [rel]
    if name is None:
        return (True, chain)
    k = _name_kind(rel, name)
    if k[0] == "code":
        return (False, chain)
    if k[0] == "reexport":
        return _leaf_is_data(k[1], k[2], chain)
    return (True, chain)  # data or unknown


def static_data_closure(files: Sequence[str]) -> Set[str]:
    """The transitive DATA-closure of `files`: `files` plus every repo module a scope
    file value-imports a module-level CONSTANT from (following re-exports to the leaf).
    Class/function imports of other modules are NOT added (safe -- see guard model).
    `files` must be concrete repo-relative .py paths (use `expand_scope` first for globs).
    Worklist fixpoint: each file's imports are extracted once (cached), each file scanned
    once."""
    out: Set[str] = set(files)
    worklist: List[str] = list(out)
    while worklist:
        rel = worklist.pop()
        for mod, name in _file_imports(rel):
            is_d, chain = _leaf_is_data(mod, name, [])
            if is_d:
                for c in chain:
                    if c not in out:
                        out.add(c)
                        worklist.append(c)
    return out


def verify_scope_static(scope: Sequence[str], *, label: str = "scope") -> Set[str]:
    """Guard 2 (cheap, static AST): every declared EXACT scope file exists, and the
    (expanded) scope is a FIXPOINT of the data-closure operator (hashes every module a
    scope file reads a constant from). Returns the expanded concrete file set. Raises
    AssertionError loudly on any violation. Does NOT run the experiment -- see
    `verify_scope_conservatism` for the call-trace guard (guard 1)."""
    exact_missing = [g for g in scope if not _is_glob(g) and not (_REPO_ROOT / g).is_file()]
    assert not exact_missing, (
        "%s: declared files do not exist (refactor drift?): %s. The scope must be "
        "corrected before it can be trusted (plan sec 11)." % (label, exact_missing))
    files = expand_scope(scope)
    closure = static_data_closure(files)
    escaped = sorted(closure - set(files))
    assert not escaped, (
        "%s is NOT data-closed -- a scope file value-imports a module-level constant from "
        "these UNSCOPED repo modules: %s. This is a false-HIT hazard (plan sec 2/11): add "
        "them to the declared scope or the key can serve a stale prefix when they change."
        % (label, escaped))
    return set(files)


def traced_execution_files(run_once: Callable[[], Any]) -> Set[str]:
    """Run `run_once()` under a Python call-trace; return the set of repo-relative .py
    files whose code executed. Used by guard 1."""
    executed: Set[str] = set()
    root_str = str(_REPO_ROOT) + os.sep

    def _tr(frame, event, arg):
        fn = frame.f_code.co_filename
        if fn.startswith(root_str):
            try:
                executed.add(str(Path(fn).resolve().relative_to(_REPO_ROOT)))
            except ValueError:
                pass
        return None

    old = sys.gettrace()
    sys.settrace(_tr)
    try:
        run_once()
    finally:
        sys.settrace(old)
    return executed


def verify_scope_conservatism(
    scope: Sequence[str],
    run_once: Optional[Callable[[], Any]] = None,
    *,
    label: str = "scope",
) -> Dict[str, Any]:
    """Full conservatism check for a declared substrate scope (plan sec 11).

    guard 2 (always): static data-closure fixpoint + existence (via `verify_scope_static`).
    guard 1 (only if `run_once` is given): execute a real (cheap) cell under a call-trace
    and assert EVERY executed repo file is in the declared scope -- i.e. the scope is a
    superset of what actually runs. `run_once` must invoke ONE representative cell whose
    substrate footprint the scope is meant to cover.

    Returns a small report dict; raises AssertionError on any violation. This is the
    mechanism that makes author-declared narrowing safe (false-miss-only). It is the SAME
    check the maturation-curriculum prototype proved out; any experiment can call it before
    trusting a scope it passes to `compute_arm_fingerprint(substrate_scope=...)`."""
    declared_files = verify_scope_static(scope, label=label)
    report: Dict[str, Any] = {
        "label": label,
        "n_declared_globs": len(scope),
        "n_declared_files": len(declared_files),
        "static_guard": "ok",
    }
    if run_once is not None:
        executed = traced_execution_files(run_once)
        escaped = sorted(executed - declared_files)
        assert not escaped, (
            "%s: these repo files EXECUTED during a real run but are NOT in the declared "
            "scope: %s. The scope under-approximates -- a false-HIT hazard (plan sec 2/11). "
            "Add them to the declared scope." % (label, escaped))
        report["n_executed"] = len(executed)
        report["trace_guard"] = "ok"
    return report


__all__ = [
    "expand_scope",
    "static_data_closure",
    "verify_scope_static",
    "traced_execution_files",
    "verify_scope_conservatism",
]
