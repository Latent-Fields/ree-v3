"""Mechanical migrator: route an experiment script's hand-rolled flat-manifest
tail through experiments.pack_writer.write_flat_manifest.

CONSERVATIVE BY DESIGN. It only rewrites a script whose manifest-write tail
matches the CANONICAL shape exactly:

    [out_dir.mkdir(...)]                                  # optional
    out_path = out_dir / f"{manifest['run_id']}.json"    # or _dry_ first
    if <DRY>:                                             # optional
        out_path = out_dir / f"_dry_{manifest['run_id']}.json"
    with open(out_path, "w") as <FH>:
        json.dump(manifest, <FH>, indent=2[, sort_keys=True])

Anything else is REPORTED as unmatched with a reason and left untouched (hand-
migrate later). No logic change: write_flat_manifest recomputes the exact same
out_path (<out_dir>/<run_id>.json, _dry_ prefix under dry_run) and preserves the
manifest field names verbatim, additionally stamping the always-record core.

Preconditions a script must meet to be auto-migrated:
  * the manifest local is named `manifest`;
  * an `out_dir` variable is assigned somewhere above the tail (write_flat_manifest
    needs it);
  * run_id is accessed as manifest['run_id'] (single OR double quotes);
  * the dry-run filename prefix is `_dry_` (the plumbing SKIP/rescan expectation);
  * a module-level SEEDS list is discoverable (else seeds=None -> a recording WARN,
    still allowed but flagged).

Usage:
  python3 migrate_manifest_writers.py --report  <files...>
  python3 migrate_manifest_writers.py --apply   <files...>
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

ALREADY = re.compile(r"write_flat_manifest|ExperimentPackWriter|\.write_pack\(")
EXEMPT = re.compile(r"MANIFEST_WRITER_EXEMPT")

# The json.dump(<var>, <fh>, indent=2[, sort_keys=True][, default=str]) line and the
# out_path.write_text(json.dumps(<var>, indent=2)[+ "\n"][, encoding=...]) idiom are
# built per-var. The var is USUALLY `manifest`, but the early-era corpus names the flat
# manifest dict `result`/`output`/`pack`/etc; detect_manifest_var proves such a var IS a
# flat manifest (dict literal carrying run_id + architecture_epoch + a status key) before
# it is routed, so a non-`manifest` var is accepted ONLY on that AST proof.
# default=str is preserved by threading json_default=str into write_flat_manifest.
def dump_re(var):
    return re.compile(
        r"^(?P<indent>\s*)json\.dump\(\s*" + re.escape(var) + r"\s*,\s*(?P<fh>\w+)\s*,\s*indent=2"
        r"(?P<extra>(?:\s*,\s*(?:sort_keys=True|default=str))*)\s*\)\s*$"
    )


def writetext_re(var):
    return re.compile(
        r'^(?P<indent>\s*)(?P<path>\w+)\.write_text\(\s*json\.dumps\(\s*' + re.escape(var) + r'\s*,\s*indent=2'
        r'(?P<extra>(?:\s*,\s*(?:sort_keys=True|default=str))*)\s*\)(?:\s*\+\s*["\']\\n["\'])?(?:\s*,\s*encoding=["\'][\w-]+["\'])?\s*\)\s*$'
    )


DUMP_RE = dump_re("manifest")
WRITETEXT_RE = writetext_re("manifest")
# with-open spelling variants (batch 4). The mode arg may carry a trailing
# `, encoding="utf-8"`, and the block may use the bound-path method form
# `<path>.open("w", ...)` instead of the `open(<path>, "w", ...)` builtin. Both are
# byte-identical for the migration: write_flat_manifest recomputes the same path
# and always writes utf-8 with a trailing newline, so the original encoding= arg is
# irrelevant to output bytes (every matched dump is ensure_ascii JSON -- dump_re /
# writetext_re never accept ensure_ascii=False). `_ENC` is the optional encoding tail.
_ENC = r'(?:\s*,\s*encoding=["\'][\w-]+["\'])?'
WITH_RE = re.compile(r'^(?P<indent>\s*)with\s+open\(\s*(?P<path>\w+)\s*,\s*["\']w["\']' + _ENC + r'\s*\)\s*as\s+(?P<fh>\w+)\s*:\s*$')
WITH_METHOD_RE = re.compile(r'^(?P<indent>\s*)with\s+(?P<path>\w+)\.open\(\s*["\']w["\']' + _ENC + r'\s*\)\s*as\s+(?P<fh>\w+)\s*:\s*$')


def detect_manifest_var(src: str):
    """Return the variable name holding the FLAT manifest dict to route, or None.

    Fast path: if the script writes a var literally named `manifest` via json.dump /
    write_text, route that (the batch-1/2 behaviour -- unchanged).

    Otherwise (early-era corpus) find a non-`manifest` var that is PROVABLY the flat
    manifest: assigned a dict LITERAL whose keys include `run_id`, `architecture_epoch`
    AND a status key (`status`|`outcome`|`overall_outcome`) -- the fields sync_v3_results
    + write_flat_manifest's identity/validity invariants require. To avoid ambiguity the
    var must be UNIQUE: exactly one such proven var is also written via json.dump/
    write_text. More than one candidate -> None (leave unmatched; hand-migrate)."""
    if "json.dump(manifest" in src or re.search(r"\.write_text\(\s*json\.dumps\(\s*manifest\b", src):
        return "manifest"
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    STATUS = {"status", "outcome", "overall_outcome"}
    proven = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            keys = {k.value for k in node.value.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            if {"run_id", "architecture_epoch"} <= keys and keys & STATUS:
                proven.add(node.targets[0].id)
    written = {v for v in proven
               if re.search(r"json\.dump\(\s*" + re.escape(v) + r"\s*,", src)
               or re.search(r"\.write_text\(\s*json\.dumps\(\s*" + re.escape(v) + r"\b", src)}
    if len(written) == 1:
        return next(iter(written))
    return None
OUTPATH_RE = re.compile(
    r"""^(?P<indent>\s*)(?P<var>\w+)\s*=\s*out_dir\s*/\s*f["']\{?_dry_\}?"""
)
# canonical primary + dry lines. run_id resolved either as <mvar>['run_id']
# (the batch-1 canonical form -- `manifest['run_id']`, generalized in batch 4 to the
# detected mvar, e.g. `pack['run_id']`/`output['run_id']`) OR as a bare local `run_id`
# variable. The bare form is accepted ONLY when `runid_is_manifest_runid` proves the
# manifest sets `"run_id": run_id`, so <dir> / f"{run_id}.json" is byte-identical to
# what write_flat_manifest recomputes (<dir> / f"{<mvar>['run_id']}.json"). The
# subscript form <mvar>['run_id'] needs no such proof -- write_flat_manifest reads the
# SAME <mvar>['run_id'], so the recomputed filename is identical by construction.
def _runid_pat(mvar: str) -> str:
    return r"""(?:""" + re.escape(mvar) + r"""\[["']run_id["']\]|run_id)"""


# The LHS path var and the dir var are CAPTURED (not hardcoded out_path/out_dir) so
# a script that names them differently -- e.g. `out_file = out_dir / f"{run_id}.json"`
# or `manifest_path = evidence_dir / f"{manifest['run_id']}.json"` -- migrates too,
# PROVIDED (a) the with-open/write_text path var equals this LHS var, and (b) the dir
# var is assigned above the tail. Both guarantee the path write_flat_manifest recomputes
# (Path(<dir>) / f"{manifest['run_id']}.json") is byte-location-identical to the original
# write target. Two path idioms are accepted, both byte-location-identical on POSIX (the
# only platform the fleet runs): the pathlib `<dir> / f"{run_id}.json"` form AND the
# `os.path.join(<dir>, f"{run_id}.json")` form (batch 4 -- write_flat_manifest does
# Path(out_dir) / f"{run_id}.json", identical to os.path.join(<dir>, ...) under os.sep=="/").
# For mvar=="manifest" + the slash form these compile IDENTICALLY to the frozen
# origin/main PRIMARY_RE/DRY_REASSIGN_RE, so existing batch output is bit-for-bit
# unchanged (validated against the frozen migrator); the os.path.join alternative only
# ever fires when the slash form fails.
def primary_res(mvar: str):
    rid = _runid_pat(mvar)
    slash = re.compile(
        r"""^(?P<indent>\s*)(?P<var>\w+)\s*=\s*(?P<dir>\w+)\s*/\s*f["']\{""" + rid + r"""\}\.json["']\s*$"""
    )
    osjoin = re.compile(
        r"""^(?P<indent>\s*)(?P<var>\w+)\s*=\s*os\.path\.join\(\s*(?P<dir>\w+)\s*,\s*f["']\{""" + rid + r"""\}\.json["']\s*\)\s*$"""
    )
    return slash, osjoin


def dry_res(mvar: str):
    rid = _runid_pat(mvar)
    slash = re.compile(
        r"""^(?P<indent>\s*)(?P<var>\w+)\s*=\s*(?P<dir>\w+)\s*/\s*f["']_dry_\{""" + rid + r"""\}\.json["']\s*$"""
    )
    osjoin = re.compile(
        r"""^(?P<indent>\s*)(?P<var>\w+)\s*=\s*os\.path\.join\(\s*(?P<dir>\w+)\s*,\s*f["']_dry_\{""" + rid + r"""\}\.json["']\s*\)\s*$"""
    )
    return slash, osjoin
# `"run_id": run_id` (or single-quoted) in the manifest dict literal proves the
# local run_id var IS the manifest run_id.
RUNID_FIELD_RE = re.compile(r"""["']run_id["']\s*:\s*run_id\b""")


def runid_is_manifest_runid(src: str) -> bool:
    """True if the manifest dict sets its run_id from the local `run_id` var, so
    that a bare-`run_id` out_path is byte-identical to the manifest['run_id'] one."""
    return bool(RUNID_FIELD_RE.search(src))
DRY_IF_RE = re.compile(r"^(?P<indent>\s*)if\s+(?P<cond>[^:]+):\s*$")
MKDIR_RE = re.compile(r"^\s*out_dir\.mkdir\([^)]*\)\s*$")


def find_seeds_symbol(src: str) -> str | None:
    for m in re.finditer(r"^(SEEDS)\b\s*[:=]", src, re.M):
        return "SEEDS"
    return None


def has_dir_assignment(lines: list[str], upto: int, dirvar: str) -> bool:
    pat = re.compile(r"^\s*" + re.escape(dirvar) + r"\s*=")
    for i in range(upto):
        if pat.match(lines[i]):
            return True
    return False


def _fn_params(fn) -> set:
    a = fn.args
    names = set()
    for grp in (a.posonlyargs, a.args, a.kwonlyargs):
        names.update(x.arg for x in grp)
    if a.vararg:
        names.add(a.vararg.arg)
    if a.kwarg:
        names.add(a.kwarg.arg)
    return names


def dir_bound_as_param(src: str, dirvar: str, ref_lineno: int) -> bool:
    """BATCH 6: True iff `dirvar` is a PARAMETER of the function that lexically
    encloses the write (the innermost def whose body spans ref_lineno) AND is NOT
    rebound anywhere in that function before ref_lineno.

    This generalizes the dir-binding requirement to the early-era `emit_manifest(...,
    out_dir: Path, ...)` shape (v3_exq_621/622/626/636/637 ...), where `out_dir` is a
    formal parameter, not an `out_dir = ...` statement, so has_dir_assignment (a
    line-scan for `<dir> =`) misses it -- and a whole-file scan would be UNSOUND
    (these scripts ALSO assign a *different*, main()-local `out_dir = Path(args.output_dir)`
    BELOW the write in another scope). Byte-safety: write_flat_manifest recomputes
    Path(<dir>) / f"{<mvar>['run_id']}.json"; when <dir> is the enclosing function's
    param and is never rebound before the write, that param value is exactly the
    dir the original `out_path = <dir> / f"{run_id}.json"` used -- byte-location-
    identical. CONSERVATIVE: any Store of dirvar before ref_lineno (even in a nested
    scope) -> refuse (a false refuse is safe); a syntactically un-parseable file ->
    refuse. Fires ONLY when has_dir_assignment already returned False, so the
    canonical batch-1..5 output (dir assigned as a statement) is byte-unchanged."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    # innermost enclosing function of ref_lineno
    encl = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", None) or node.lineno
            if node.lineno < ref_lineno <= end:
                if encl is None or node.lineno > encl.lineno:
                    encl = node
    if encl is None or dirvar not in _fn_params(encl):
        return False
    # refuse if dirvar is rebound (Store) before the write within the enclosing fn
    for node in ast.walk(encl):
        if isinstance(node, ast.Name) and node.id == dirvar and isinstance(node.ctx, ast.Store):
            if node.lineno < ref_lineno:
                return False
    return True


def config_expr(src: str, mvar: str = "manifest") -> str:
    has_cfg = re.search(r'^\s{6,}["\']config["\']\s*:', src, re.M)
    has_cfgsum = re.search(r'["\']config_summary["\']\s*:', src)
    if has_cfg and not has_cfgsum:
        return f'{mvar}.get("config")'
    if has_cfgsum:
        return f'{mvar}.get("config") or {mvar}.get("config_summary")'
    return f'{mvar}.get("config")'


# ---------------------------------------------------------------------------
# BATCH 5: non-canonical filename proof.
#
# The batches above route ONLY the canonical `<dir>/f"{run_id}.json"` (slash or
# os.path.join) path. A residual class writes a NON-canonical filename while
# detect_manifest_var still succeeds: a hardcoded literal (`"exq_051b_v3.json"`),
# a `"%s.json" % run_id` idiom, or a `f"{TYPE}_{ts}_v3.json"` form where the
# filename happens to already equal the run_id. Routing these is byte-location-safe
# IFF the filename string provably equals `f"{<mvar>['run_id']}.json"` -- because
# write_flat_manifest recomputes `Path(out_dir)/f"{run_id}.json"`, and a mismatch
# would RENAME the flat file (the whole reason the coordinator/indexer/explorer key
# on `<run_id>.json`). The overwhelming early-era `f"{TYPE}_{ts}.json"` form has
# run_id = `f"{TYPE}_{ts}_v3"` (filename missing the `_v3`) or inlines a SECOND
# `datetime.now()` read -- both must be REFUSED, so the proof is strict.
#
# Proof model (sound, conservative): reduce both the filename expr and the manifest
# run_id value expr to a "template" -- a list of atoms, each a literal chunk
# ('L', s) or a single-static-assignment variable leaf ('N', name). f-string
# interpolations must be BARE NAMES (an inline Call like datetime.now() -> refuse:
# recomputed / non-deterministic); only `%s`-style %-formatting and `+` string
# concat are accepted. Every ('N', name) leaf must have exactly ONE binding site and
# that binding must not be inside a loop, so the run_id and the filename read the
# SAME runtime value (even a once-computed `ts = datetime.now()...`). PROVE:
#   normalize(template(filename)) == normalize(template(run_id) + [('L', '.json')]).
# When proven, the ORIGINAL path assignment is LEFT in place (write-only-rewrite,
# like the non-adjacent batch-3 path) and write_flat_manifest is called with the
# out_dir sub-expression taken verbatim from the original assignment.
# ---------------------------------------------------------------------------
def _ncf_template(node):
    """A string-valued expr -> [('L',s)|('N',name)] atoms, or None if not provably a
    deterministic concatenation of literals + single-assignment variable values."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [("L", node.value)]
    if isinstance(node, ast.Name):
        return [("N", node.id)]
    if isinstance(node, ast.JoinedStr):
        atoms = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                atoms.append(("L", v.value))
            elif isinstance(v, ast.FormattedValue):
                # {x!r}/{x:spec} transform the string; an inline non-Name value
                # (Call/Attribute/Subscript) is recomputed/non-deterministic.
                if v.conversion != -1 or v.format_spec is not None:
                    return None
                if isinstance(v.value, ast.Name):
                    atoms.append(("N", v.value.id))
                else:
                    return None
            else:
                return None
        return atoms
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _ncf_template(node.left)
        right = _ncf_template(node.right)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        fmt = node.left
        if not (isinstance(fmt, ast.Constant) and isinstance(fmt.value, str)):
            return None
        s = fmt.value
        if any(sp != "%s" for sp in re.findall(r"%.", s)):
            return None  # any conversion other than %s (incl. %% or %d) -> refuse
        parts = s.split("%s")
        args = node.right.elts if isinstance(node.right, ast.Tuple) else [node.right]
        if len(parts) - 1 != len(args):
            return None
        atoms = []
        for i, p in enumerate(parts):
            if p:
                atoms.append(("L", p))
            if i < len(args):
                if isinstance(args[i], ast.Name):
                    atoms.append(("N", args[i].id))
                else:
                    return None
        return atoms
    return None


def _ncf_normalize(atoms):
    """Drop empty literals, merge adjacent literal chunks (so ['a','b'] == ['ab'])."""
    if atoms is None:
        return None
    out = []
    for a in atoms:
        if a[0] == "L" and a[1] == "":
            continue
        if out and out[-1][0] == "L" and a[0] == "L":
            out[-1] = ("L", out[-1][1] + a[1])
        else:
            out.append(a)
    return out


def _ncf_binding_map(tree):
    """name -> list of (binding_node, in_loop_bool). Covers assign/ann/aug/walrus/for/
    with-as targets and function args (a function arg is one binding per call, fixed
    within the call, so both uses read the same value)."""
    parent = {}
    for p in ast.walk(tree):
        for c in ast.iter_child_nodes(p):
            parent[c] = p

    def in_loop(n):
        cur = parent.get(n)
        while cur is not None:
            if isinstance(cur, (ast.For, ast.AsyncFor, ast.While, ast.comprehension,
                                ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                return True
            cur = parent.get(cur)
        return False

    sites = {}
    for n in ast.walk(tree):
        binds = []
        if isinstance(n, ast.Assign):
            for t in n.targets:
                binds += [x for x in ast.walk(t)
                          if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)]
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None:
            binds.append(n.target)
        elif isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
            binds.append(n.target)
        elif isinstance(n, ast.NamedExpr) and isinstance(n.target, ast.Name):
            binds.append(n.target)
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            binds += [x for x in ast.walk(n.target)
                      if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)]
        elif isinstance(n, ast.withitem) and n.optional_vars is not None:
            binds += [x for x in ast.walk(n.optional_vars)
                      if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)]
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = n.args
            allargs = (list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs)
                       + ([a.vararg] if a.vararg else []) + ([a.kwarg] if a.kwarg else []))
            for arg in allargs:
                sites.setdefault(arg.arg, []).append((arg, False))
        for b in binds:
            sites.setdefault(b.id, []).append((b, in_loop(b)))
    return sites


def _ncf_ssa_ok(name, sites):
    lst = sites.get(name, [])
    return len(lst) == 1 and not lst[0][1]


def _ncf_split_path(node):
    """A path-building expr -> (dir_node, filename_node). Handles `<dir>/<fn>` (the
    OUTERMOST `/`, so `a/b/c/"fn"` -> dir=`a/b/c`, fn=`"fn"`) and
    `os.path.join(<dir>, <fn>)`. Else (None, node)."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return node.left, node.right
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "join" and len(node.args) == 2):
        return node.args[0], node.args[1]
    return None, node


def _ncf_runid_value_node(tree, mvar):
    """The AST value node bound to the 'run_id' key of the mvar dict LITERAL (the value
    write_flat_manifest will key the filename on). None if absent/ambiguous."""
    found = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == mvar):
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and k.value == "run_id":
                    found.append(v)
    return found[0] if len(found) == 1 else None


def noncanonical_filename_proof(src: str, mvar: str, pathvar: str, write_lineno: int):
    """Prove the NON-canonical filename in the nearest `pathvar = <path>` assignment
    above line ``write_lineno`` equals f"{<mvar>['run_id']}.json". Returns
    (dir_src, "ok") on proof, else (None, reason). ``dir_src`` is the exact source text
    of the out_dir sub-expression (quote style preserved) for the write_flat_manifest call."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return (None, "parse-error")
    rid = _ncf_runid_value_node(tree, mvar)
    if rid is None:
        return (None, f"no run_id value in {mvar} dict literal")
    # nearest `pathvar = <path>` assignment above the write whose RHS looks path-like
    cand = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == pathvar and node.lineno < write_lineno):
            d, _fn = _ncf_split_path(node.value)
            if d is not None and (cand is None or node.lineno > cand.lineno):
                cand = node
    if cand is None:
        return (None, f"no dir/filename path assignment to {pathvar} above the write")
    dir_node, fn_node = _ncf_split_path(cand.value)
    t_fn = _ncf_normalize(_ncf_template(fn_node))
    t_rid = _ncf_normalize(_ncf_template(rid))
    if t_fn is None:
        return (None, f"filename not templatable: {ast.dump(fn_node)[:60]}")
    if t_rid is None:
        return (None, f"run_id not templatable: {ast.dump(rid)[:60]}")
    if t_fn != _ncf_normalize(t_rid + [("L", ".json")]):
        return (None, f"filename {t_fn} != run_id+'.json' {t_rid}")
    sites = _ncf_binding_map(tree)
    names = {a[1] for a in t_fn if a[0] == "N"} | {a[1] for a in t_rid if a[0] == "N"}
    bad = sorted(n for n in names if not _ncf_ssa_ok(n, sites))
    if bad:
        return (None, f"non-single-assignment leaf(s): {bad}")
    dir_src = ast.get_source_segment(src, dir_node)
    if dir_src is None or "\n" in dir_src:
        return (None, "dir sub-expression spans multiple lines / unavailable")
    return (dir_src, "ok")


def migrate_one(path: Path):
    """Return (status, detail, new_src_or_None)."""
    src = path.read_text(encoding="utf-8")
    if EXEMPT.search(src):
        return ("skip", "MANIFEST_WRITER_EXEMPT", None)
    if ALREADY.search(src):
        return ("skip", "already-routed", None)
    # Which var holds the flat manifest dict? `manifest` (fast path) OR a proven
    # non-`manifest` early-era var (result/output/pack/...). None -> not migratable here.
    mvar = detect_manifest_var(src)
    if mvar is None:
        return ("unmatched", "no json.dump(manifest tail", None)
    MVAR_DUMP_RE = dump_re(mvar)
    MVAR_WRITETEXT_RE = writetext_re(mvar)
    has_jsondump = f"json.dump({mvar}" in src
    has_writetext = bool(re.search(r"\.write_text\(\s*json\.dumps\(\s*" + re.escape(mvar) + r"\b", src))
    if not has_jsondump and not has_writetext:
        return ("unmatched", "no json.dump(<mvar> tail", None)

    # Multiple write sites for the SAME mvar (e.g. a branchy `if dry: {..write..} else:
    # {..write..}` shape like v3_exq_354) cannot be routed by a single write_flat_manifest
    # call -- the migrator rewrites only the FIRST write, which would leave the SECOND as a
    # raw json.dump AND mark the script "already-routed" on the next pass. The canonical
    # dry-run idiom the migrator DOES support is single-write-with-reassign (`out_path =
    # ...; if dry: out_path = _dry_...; with open(out_path): dump`). So: if mvar is written
    # via json.dump/write_text more than once, refuse (hand-migrate). Every batch-1/2/3
    # canonical script is single-write, so this never regresses them (backward-compat check).
    n_write_sites = (len(re.findall(r"json\.dump\(\s*" + re.escape(mvar) + r"\s*,", src))
                     + len(re.findall(r"\.write_text\(\s*json\.dumps\(\s*" + re.escape(mvar) + r"\b", src)))
    if n_write_sites > 1:
        return ("unmatched", f"multiple ({n_write_sites}) {mvar} write sites (branchy); hand-migrate", None)

    # Per-mvar canonical path matchers (slash + os.path.join alternatives). For
    # mvar=="manifest" + the slash form these are byte-identical to the frozen
    # PRIMARY_RE/DRY_REASSIGN_RE; the os.path.join alternative only fires when slash fails.
    PRIMARY_SLASH_RE, PRIMARY_OSJOIN_RE = primary_res(mvar)
    DRY_SLASH_RE, DRY_OSJOIN_RE = dry_res(mvar)

    def match_primary(line: str):
        return PRIMARY_SLASH_RE.match(line) or PRIMARY_OSJOIN_RE.match(line)

    def match_dry(line: str):
        return DRY_SLASH_RE.match(line) or DRY_OSJOIN_RE.match(line)

    lines = src.splitlines()
    # Locate the manifest write statement. Two accepted idioms:
    #   A) with open(out_path, "w") as fh:   (2 lines)
    #          json.dump(<mvar>, fh, indent=2[, sort_keys=True])
    #   B) out_path.write_text(json.dumps(<mvar>, indent=2)[ + "\n"][, encoding=...])  (1 line)
    # write_top/write_bot bracket the write statement (inclusive) for replacement.
    write_top = write_bot = None
    has_default_str = False
    pathvar = None  # the file path var in the write statement; must == the primary LHS var
    write_indent = ""  # indent of the write statement (used for the non-adjacent replacement)
    for i, ln in enumerate(lines):
        wt = MVAR_WRITETEXT_RE.match(ln)
        if wt:
            pathvar = wt.group("path")
            write_indent = wt.group("indent")
            has_default_str = "default=str" in (wt.group("extra") or "")
            write_top = write_bot = i
            break
    if write_top is None:
        if not has_jsondump:
            return ("unmatched", "write_text(json.dumps(<mvar> present but not the canonical form", None)
        dump_idx = None
        for i, ln in enumerate(lines):
            if MVAR_DUMP_RE.match(ln):
                dump_idx = i
                break
        if dump_idx is None:
            return ("unmatched", "json.dump(<mvar> present but not the canonical indent=2 form", None)
        dm = MVAR_DUMP_RE.match(lines[dump_idx])
        # the preceding line must be `with open(<path>, "w") as <fh>:` and the handle
        # must match the json.dump handle (a name mismatch = a broken/unexpected script).
        # The path var may be named anything (out_path / out_file / manifest_path / ...);
        # its identity as the flat manifest target is proven by the primary-assignment
        # + pathvar==primary-var guard below, NOT by the name.
        if dump_idx == 0:
            return ("unmatched", "dump at file start", None)
        wm = WITH_RE.match(lines[dump_idx - 1]) or WITH_METHOD_RE.match(lines[dump_idx - 1])
        if not wm:
            return ("unmatched", "no canonical `with open(<path>, \"w\") as fh:` above dump", None)
        if wm.group("fh") != dm.group("fh"):
            return ("unmatched", "with-open handle != json.dump handle", None)
        pathvar = wm.group("path")
        write_indent = wm.group("indent")
        has_default_str = "default=str" in (dm.group("extra") or "")
        write_top = dump_idx - 1
        write_bot = dump_idx
        # The with-block may carry a trailing `<fh>.write("\n")` after the dump (some
        # scripts add the newline by hand). write_flat_manifest ALREADY appends a
        # trailing "\n", so absorbing that line is byte-identical output. Absorb any
        # such trailing newline-writes; if the block holds ANY other statement, the
        # write does more than emit the manifest -> refuse (leave unmatched).
        fh = dm.group("fh")
        fwrite_nl = re.compile(r'^\s*' + re.escape(fh) + r'\.write\(\s*["\']\\n["\']\s*\)\s*$')
        wlen = len(write_indent)
        b = write_bot + 1
        while b < len(lines):
            ln = lines[b]
            if ln.strip() == "":
                b += 1
                continue  # intra-block blank -- keep scanning
            if len(ln) - len(ln.lstrip()) <= wlen:
                break  # dedented out of the with-block
            if not fwrite_nl.match(ln):
                return ("unmatched", "with-block has a statement after json.dump other than <fh>.write(newline)", None)
            write_bot = b
            b += 1

    # walk upward past optional blank lines to find the out_path assignment(s)
    j = write_top - 1
    while j >= 0 and lines[j].strip() == "":
        j -= 1
    # optional dry reassignment block: `if <cond>:` \n `    out_path = ..._dry_...`
    dry_cond = None
    # pattern A: primary then if/dry
    # Identify contiguous assignment region
    block_top = None
    dry_m = None
    non_adjacent = False
    noncanon = False        # batch 5: non-canonical filename proven == f"{run_id}.json"
    noncanon_dir = None     # the out_dir sub-expression source for the wfm call
    dir_via_param = False   # batch 6: dirvar bound as an enclosing-function parameter
    # Case: lines[j] is the dry reassignment inside an if
    if j >= 1 and match_dry(lines[j]) and DRY_IF_RE.match(lines[j - 1]):
        dry_m = match_dry(lines[j])
        dry_cond = DRY_IF_RE.match(lines[j - 1]).group("cond").strip()
        k = j - 2
        while k >= 0 and lines[k].strip() == "":
            k -= 1
        if k >= 0 and match_primary(lines[k]):
            block_top = k
        else:
            return ("unmatched", "dry-reassign present but no canonical primary out_path above", None)
    elif j >= 0 and match_primary(lines[j]):
        block_top = j
    else:
        # Non-adjacent: the path var is assigned canonically but NOT immediately above
        # the write (intervening manifest-building code, e.g. v3_exq_620). Find the
        # NEAREST assignment to pathvar above the write. Because it is the nearest, the
        # var is provably NOT reassigned between it and the write -- so the recomputed
        # path is byte-location-identical. We then rewrite ONLY the write statement
        # (leaving the assignment + intervening code untouched, since that code may read
        # the path); the wfm return reassigns pathvar to the same value.
        assign_re = re.compile(r"^\s*" + re.escape(pathvar) + r"\s*=(?!=)")
        a = write_top - 1
        near = None
        while a >= 0:
            if assign_re.match(lines[a]):
                near = a
                break
            a -= 1
        if near is None:
            return ("unmatched", "no path assignment above tail", None)
        if not match_primary(lines[near]):
            # BATCH 5: the path var IS assigned above but the filename is NON-canonical
            # (a literal, a "%s.json" % run_id idiom, or a f"{TYPE}_{ts}_v3.json" that
            # already equals run_id). Attempt an AST proof that the filename string ==
            # f"{<mvar>['run_id']}.json"; on proof, route via write-only-rewrite with the
            # ORIGINAL out_dir sub-expression. On any doubt, refuse (leave unmatched).
            dir_src, why = noncanonical_filename_proof(src, mvar, pathvar, write_top + 1)
            if dir_src is None:
                return ("unmatched", f"non-canonical filename not provably == run_id ({why})", None)
            noncanon = True
            noncanon_dir = dir_src
        block_top = near
        non_adjacent = True

    if noncanon:
        # Proof already established: filename == f"{<mvar>['run_id']}.json" AND the path
        # assignment sets `pathvar`; the dir sub-expression is taken verbatim from that
        # assignment (still present after the write-only-rewrite, so its free names are
        # in scope). No primary_m / dirvar-var / dry-var checks apply here.
        primary_m = None
        dirvar = noncanon_dir
    else:
        primary_m = match_primary(lines[block_top])
        dirvar = primary_m.group("dir")
        # The write statement's path var MUST be the var this primary assignment sets --
        # else the with-open/write_text is targeting a DIFFERENT file (e.g. a pack-style
        # runs/<id>/manifest.json or a per-run sibling) and routing it would relocate the
        # write. This is the identity proof for the generalized (non-out_path) path var.
        if primary_m.group("var") != pathvar:
            return ("unmatched", f"write path var {pathvar!r} != primary assignment var {primary_m.group('var')!r}", None)
        # A dry reassignment (if present) must set the SAME path var from the SAME dir var,
        # so the recomputed _dry_ path is byte-location-identical too.
        if dry_m is not None and (dry_m.group("var") != pathvar or dry_m.group("dir") != dirvar):
            return ("unmatched", "dry-reassign var/dir mismatch vs primary", None)

    if non_adjacent:
        # Rewrite ONLY the write statement; leave the assignment + intervening code and
        # any mkdir in place. Indent = the write statement's own indent.
        top = write_top
        indent = write_indent
    else:
        # optional mkdir line just above block_top (write_flat_manifest mkdirs itself)
        mkdir_re = re.compile(r"^\s*" + re.escape(dirvar) + r"\.mkdir\([^)]*\)\s*$")
        top = block_top
        p = top - 1
        while p >= 0 and lines[p].strip() == "":
            p -= 1
        if p >= 0 and mkdir_re.match(lines[p]):
            top = p  # absorb mkdir
        # Statement-level indent = the indent of the out_path assignment being
        # replaced (NOT the json.dump body indent, which is one level deeper and
        # would produce an IndentationError).
        indent = primary_m.group("indent")

    # For a canonical `<dir>/f"{run_id}.json"` path, `dirvar` is a plain Name that must be
    # assigned above, and the run_id in the path must provably equal <mvar>['run_id'].
    # For the batch-5 non-canonical path both are already established by the proof:
    # `dirvar` is the ORIGINAL dir sub-expression (still evaluated by the untouched path
    # assignment a line above the wfm call), and the filename==f"{run_id}.json" proof
    # replaces the bare-run_id check. So skip both var-only guards under noncanon.
    if not noncanon:
        # dirvar must be bound above the write: either an `<dir> = ...` statement
        # (has_dir_assignment, the canonical batch-1..5 path) OR a formal parameter of
        # the enclosing function, never rebound before the write (batch 6 -- the early-era
        # `emit_manifest(..., out_dir, ...)` shape). block_top+1 is the 1-indexed line of
        # the primary path assignment, where dirvar is consumed.
        if not has_dir_assignment(lines, top, dirvar):
            if dir_bound_as_param(src, dirvar, block_top + 1):
                dir_via_param = True
            else:
                return ("unmatched", f"no {dirvar} assignment above tail", None)
        # Bare-`run_id` out_path is only safe when it provably equals <mvar>['run_id']. The
        # subscript form <mvar>['run_id'] in the path needs no proof -- write_flat_manifest
        # reads the SAME <mvar>['run_id'], so the recomputed filename is identical.
        if (mvar + "[") not in lines[block_top] and not runid_is_manifest_runid(src):
            return ("unmatched", f"bare run_id out_path not provably == {mvar}['run_id']", None)
    dry_run_arg = dry_cond if dry_cond is not None else "False"
    seeds_sym = find_seeds_symbol(src)
    seeds_arg = seeds_sym if seeds_sym else "None"
    cfg = config_expr(src, mvar)

    replacement = [
        f"{indent}{pathvar} = write_flat_manifest(",
        f"{indent}    {mvar},",
        f"{indent}    {dirvar},",
        f"{indent}    dry_run={dry_run_arg},",
        f"{indent}    config={cfg},",
        f"{indent}    seeds={seeds_arg},",
        f"{indent}    script_path=Path(__file__),",
    ]
    if has_default_str:
        replacement.append(f"{indent}    json_default=str,")
    replacement.append(f"{indent})")
    new_lines = lines[:top] + replacement + lines[write_bot + 1:]

    # insert import if absent
    new_src = "\n".join(new_lines)
    if src.endswith("\n"):
        new_src += "\n"
    if "from experiments.pack_writer import write_flat_manifest" not in new_src:
        new_src = insert_import(new_src)
    # The emitted call passes `script_path=Path(__file__)`, so `Path` must be bound.
    # pathlib-idiom scripts (the `<dir> / f"{run_id}.json"` slash form) already import
    # it (they divide Path objects); but the os.path.join family imports only `os`, so
    # a bare `Path` would NameError at runtime. Ensure the import when it is absent --
    # a no-op (byte-identical) for every already-Path-bound script, so backward-compat
    # against the frozen migrator is preserved (all 25 canonical are slash-form).
    new_src = ensure_path_import(new_src)

    flags = []
    if noncanon:
        flags.append(f"non-canonical-filename(dir={noncanon_dir})")
    if mvar != "manifest":
        flags.append(f"non-manifest-var({mvar})")
    if non_adjacent:
        flags.append("non-adjacent(write-only-rewrite)")
    if dir_via_param:
        flags.append(f"dir-bound-as-param({dirvar})")
    if seeds_sym is None:
        flags.append("SEEDS-not-found(seeds=None)")
    if dry_cond is None:
        flags.append("no-dry-branch(dry_run=False)")
    if has_default_str:
        flags.append("json_default=str")
    return ("migrate", ";".join(flags) or "ok", new_src)


def insert_import(src: str, imp: str = "from experiments.pack_writer import write_flat_manifest  # noqa: E402") -> str:
    """Insert `imp` AFTER the last complete top-level import statement, correctly
    stepping over multi-line parenthesized imports (inserting inside a
    `from x import (` continuation would be a SyntaxError). Defaults to the
    write_flat_manifest import (batch-1 behaviour, unchanged)."""
    lines = src.splitlines()
    n = len(lines)
    last_import_end = None
    i = 0
    while i < n:
        ln = lines[i]
        if re.match(r"^(import |from )", ln):
            depth = ln.count("(") - ln.count(")")
            j = i
            while depth > 0 and j + 1 < n:
                j += 1
                depth += lines[j].count("(") - lines[j].count(")")
            last_import_end = j
            i = j + 1
        else:
            i += 1
    anchor = last_import_end if last_import_end is not None else 0
    lines.insert(anchor + 1, imp)
    out = "\n".join(lines)
    if src.endswith("\n"):
        out += "\n"
    return out


def _path_is_bound(src: str) -> bool:
    """True if the bare name ``Path`` is bound at module scope -- via
    ``from pathlib import Path`` (any spelling, incl. multi-name / aliased) or a
    top-level ``Path = ...`` assignment. A bare ``import pathlib`` does NOT bind
    ``Path`` (only ``pathlib.Path``), so it returns False. On a parse failure we
    return True (never touch a script we cannot parse)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "pathlib":
            for a in node.names:
                if (a.asname or a.name) == "Path":
                    return True
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "Path":
                    return True
    return False


def ensure_path_import(src: str) -> str:
    """Insert ``from pathlib import Path`` (after the last top-level import) when the
    bare name ``Path`` is not already bound -- required because the migrated call uses
    ``script_path=Path(__file__)``. No-op when ``Path`` is already in scope."""
    if _path_is_bound(src):
        return src
    return insert_import(src, "from pathlib import Path  # noqa: E402")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    counts = {"migrate": 0, "skip": 0, "unmatched": 0}
    for f in args.files:
        p = Path(f)
        try:
            status, detail, new_src = migrate_one(p)
        except Exception as exc:  # never crash the batch on one file
            print(f"[ERROR ] {p.name}: {exc}")
            counts["unmatched"] += 1
            continue
        counts[status] = counts.get(status, 0) + 1
        tag = {"migrate": "MIGRATE", "skip": "skip   ", "unmatched": "UNMATCH"}[status]
        print(f"[{tag}] {p.name}: {detail}")
        if args.apply and status == "migrate" and new_src is not None:
            p.write_text(new_src, encoding="utf-8")
    print(f"\n=== {counts['migrate']} migrate, {counts['skip']} skip, {counts['unmatched']} unmatched "
          f"(of {len(args.files)}) ===")


if __name__ == "__main__":
    main()
