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
        r'(?:\s*,\s*sort_keys=True)?\s*\)(?:\s*\+\s*["\']\\n["\'])?(?:\s*,\s*encoding=["\']utf-8["\'])?\s*\)\s*$'
    )


DUMP_RE = dump_re("manifest")
WRITETEXT_RE = writetext_re("manifest")
WITH_RE = re.compile(r'^(?P<indent>\s*)with\s+open\(\s*(?P<path>\w+)\s*,\s*["\']w["\']\s*\)\s*as\s+(?P<fh>\w+)\s*:\s*$')


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
# canonical primary + dry lines. run_id resolved either as manifest['run_id']
# (the batch-1 canonical form) OR as a bare local `run_id` variable. The bare
# form is accepted ONLY when `runid_is_manifest_runid` proves the manifest sets
# `"run_id": run_id`, so out_dir / f"{run_id}.json" is byte-identical to what
# write_flat_manifest recomputes (out_dir / f"{manifest['run_id']}.json").
_RUNID = r"""(?:manifest\[["']run_id["']\]|run_id)"""
# The LHS path var and the dir var are CAPTURED (not hardcoded out_path/out_dir) so
# a script that names them differently -- e.g. `out_file = out_dir / f"{run_id}.json"`
# or `manifest_path = evidence_dir / f"{manifest['run_id']}.json"` -- migrates too,
# PROVIDED (a) the with-open/write_text path var equals this LHS var, and (b) the dir
# var is assigned above the tail. Both guarantee the path write_flat_manifest recomputes
# (Path(<dir>) / f"{manifest['run_id']}.json") is byte-location-identical to the original
# write target. out_path/out_dir remain a matched instance -> existing batch output is
# bit-for-bit unchanged (validated against the frozen origin/main migrator).
PRIMARY_RE = re.compile(
    r"""^(?P<indent>\s*)(?P<var>\w+)\s*=\s*(?P<dir>\w+)\s*/\s*f["']\{""" + _RUNID + r"""\}\.json["']\s*$"""
)
DRY_REASSIGN_RE = re.compile(
    r"""^(?P<indent>\s*)(?P<var>\w+)\s*=\s*(?P<dir>\w+)\s*/\s*f["']_dry_\{""" + _RUNID + r"""\}\.json["']\s*$"""
)
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


def config_expr(src: str, mvar: str = "manifest") -> str:
    has_cfg = re.search(r'^\s{6,}["\']config["\']\s*:', src, re.M)
    has_cfgsum = re.search(r'["\']config_summary["\']\s*:', src)
    if has_cfg and not has_cfgsum:
        return f'{mvar}.get("config")'
    if has_cfgsum:
        return f'{mvar}.get("config") or {mvar}.get("config_summary")'
    return f'{mvar}.get("config")'


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
        wm = WITH_RE.match(lines[dump_idx - 1])
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
    # Case: lines[j] is the dry reassignment inside an if
    if j >= 1 and DRY_REASSIGN_RE.match(lines[j]) and DRY_IF_RE.match(lines[j - 1]):
        dry_m = DRY_REASSIGN_RE.match(lines[j])
        dry_cond = DRY_IF_RE.match(lines[j - 1]).group("cond").strip()
        k = j - 2
        while k >= 0 and lines[k].strip() == "":
            k -= 1
        if k >= 0 and PRIMARY_RE.match(lines[k]):
            block_top = k
        else:
            return ("unmatched", "dry-reassign present but no canonical primary out_path above", None)
    elif j >= 0 and PRIMARY_RE.match(lines[j]):
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
        if near is None or not PRIMARY_RE.match(lines[near]):
            return ("unmatched", "no canonical `out_path = out_dir / f\"{manifest['run_id']}.json\"` (adjacent or nearest)", None)
        block_top = near
        non_adjacent = True

    primary_m = PRIMARY_RE.match(lines[block_top])
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

    if not has_dir_assignment(lines, top, dirvar):
        return ("unmatched", f"no {dirvar} assignment above tail", None)

    # Bare-`run_id` out_path is only safe when it provably equals manifest['run_id'].
    if "manifest[" not in lines[block_top] and not runid_is_manifest_runid(src):
        return ("unmatched", "bare run_id out_path not provably == manifest['run_id']", None)
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

    flags = []
    if mvar != "manifest":
        flags.append(f"non-manifest-var({mvar})")
    if non_adjacent:
        flags.append("non-adjacent(write-only-rewrite)")
    if seeds_sym is None:
        flags.append("SEEDS-not-found(seeds=None)")
    if dry_cond is None:
        flags.append("no-dry-branch(dry_run=False)")
    if has_default_str:
        flags.append("json_default=str")
    return ("migrate", ";".join(flags) or "ok", new_src)


def insert_import(src: str) -> str:
    """Insert the write_flat_manifest import AFTER the last complete top-level
    import statement, correctly stepping over multi-line parenthesized imports
    (inserting inside a `from x import (` continuation would be a SyntaxError)."""
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
    imp = "from experiments.pack_writer import write_flat_manifest  # noqa: E402"
    lines.insert(anchor + 1, imp)
    out = "\n".join(lines)
    if src.endswith("\n"):
        out += "\n"
    return out


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
