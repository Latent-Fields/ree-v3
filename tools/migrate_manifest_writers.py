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
import re
import sys
from pathlib import Path

ALREADY = re.compile(r"write_flat_manifest|ExperimentPackWriter|\.write_pack\(")
EXEMPT = re.compile(r"MANIFEST_WRITER_EXEMPT")

# The json.dump(manifest, <fh>, indent=2[, sort_keys=True][, default=str]) line.
# default=str is preserved by threading json_default=str into write_flat_manifest.
DUMP_RE = re.compile(
    r"^(?P<indent>\s*)json\.dump\(\s*manifest\s*,\s*(?P<fh>\w+)\s*,\s*indent=2"
    r"(?P<extra>(?:\s*,\s*(?:sort_keys=True|default=str))*)\s*\)\s*$"
)
WITH_RE = re.compile(r'^(?P<indent>\s*)with\s+open\(\s*(?P<path>\w+)\s*,\s*["\']w["\']\s*\)\s*as\s+(?P<fh>\w+)\s*:\s*$')
# Early-era single-line write idiom: out_path.write_text(json.dumps(manifest, indent=2)
# [+ "\n"][, encoding="utf-8"]). Only the `manifest` var is accepted -- the same
# identity guarantee as the json.dump(manifest tail. `\\n` matches a literal \n.
WRITETEXT_RE = re.compile(
    r'^(?P<indent>\s*)(?P<path>\w+)\.write_text\(\s*json\.dumps\(\s*manifest\s*,\s*indent=2'
    r'(?:\s*,\s*sort_keys=True)?\s*\)(?:\s*\+\s*["\']\\n["\'])?(?:\s*,\s*encoding=["\']utf-8["\'])?\s*\)\s*$'
)
OUTPATH_RE = re.compile(
    r"""^(?P<indent>\s*)(?P<var>\w+)\s*=\s*out_dir\s*/\s*f["']\{?_dry_\}?"""
)
# canonical primary + dry lines. run_id resolved either as manifest['run_id']
# (the batch-1 canonical form) OR as a bare local `run_id` variable. The bare
# form is accepted ONLY when `runid_is_manifest_runid` proves the manifest sets
# `"run_id": run_id`, so out_dir / f"{run_id}.json" is byte-identical to what
# write_flat_manifest recomputes (out_dir / f"{manifest['run_id']}.json").
_RUNID = r"""(?:manifest\[["']run_id["']\]|run_id)"""
PRIMARY_RE = re.compile(
    r"""^(?P<indent>\s*)out_path\s*=\s*out_dir\s*/\s*f["']\{""" + _RUNID + r"""\}\.json["']\s*$"""
)
DRY_REASSIGN_RE = re.compile(
    r"""^(?P<indent>\s*)out_path\s*=\s*out_dir\s*/\s*f["']_dry_\{""" + _RUNID + r"""\}\.json["']\s*$"""
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


def has_out_dir_assignment(lines: list[str], upto: int) -> bool:
    for i in range(upto):
        if re.match(r"^\s*out_dir\s*=", lines[i]):
            return True
    return False


def config_expr(src: str) -> str:
    has_cfg = re.search(r'^\s{6,}["\']config["\']\s*:', src, re.M)
    has_cfgsum = re.search(r'["\']config_summary["\']\s*:', src)
    if has_cfg and not has_cfgsum:
        return 'manifest.get("config")'
    if has_cfgsum:
        return 'manifest.get("config") or manifest.get("config_summary")'
    return 'manifest.get("config")'


def migrate_one(path: Path):
    """Return (status, detail, new_src_or_None)."""
    src = path.read_text(encoding="utf-8")
    if EXEMPT.search(src):
        return ("skip", "MANIFEST_WRITER_EXEMPT", None)
    if ALREADY.search(src):
        return ("skip", "already-routed", None)
    has_jsondump = "json.dump(manifest" in src
    has_writetext = bool(re.search(r"\.write_text\(\s*json\.dumps\(\s*manifest\b", src))
    if not has_jsondump and not has_writetext:
        return ("unmatched", "no json.dump(manifest tail", None)

    lines = src.splitlines()
    # Locate the manifest write statement. Two accepted idioms:
    #   A) with open(out_path, "w") as fh:   (2 lines)
    #          json.dump(manifest, fh, indent=2[, sort_keys=True])
    #   B) out_path.write_text(json.dumps(manifest, indent=2)[ + "\n"][, encoding=...])  (1 line)
    # write_top/write_bot bracket the write statement (inclusive) for replacement.
    write_top = write_bot = None
    has_default_str = False
    for i, ln in enumerate(lines):
        wt = WRITETEXT_RE.match(ln)
        if wt and wt.group("path") == "out_path":
            write_top = write_bot = i
            break
    if write_top is None:
        if not has_jsondump:
            return ("unmatched", "write_text(json.dumps(manifest present but not the canonical form", None)
        dump_idx = None
        for i, ln in enumerate(lines):
            if DUMP_RE.match(ln):
                dump_idx = i
                break
        if dump_idx is None:
            return ("unmatched", "json.dump(manifest present but not the canonical indent=2 form", None)
        dm = DUMP_RE.match(lines[dump_idx])
        # the preceding line must be `with open(out_path, "w") as <fh>:`
        if dump_idx == 0:
            return ("unmatched", "dump at file start", None)
        wm = WITH_RE.match(lines[dump_idx - 1])
        if not wm:
            return ("unmatched", "no canonical `with open(out_path, \"w\") as fh:` above dump", None)
        if wm.group("path") != "out_path" or wm.group("fh") != dm.group("fh"):
            return ("unmatched", "with-open path/handle not out_path/<fh>", None)
        has_default_str = "default=str" in (dm.group("extra") or "")
        write_top = dump_idx - 1
        write_bot = dump_idx

    # walk upward past optional blank lines to find the out_path assignment(s)
    j = write_top - 1
    while j >= 0 and lines[j].strip() == "":
        j -= 1
    # optional dry reassignment block: `if <cond>:` \n `    out_path = ..._dry_...`
    dry_cond = None
    # pattern A: primary then if/dry
    # Identify contiguous assignment region
    block_top = None
    # Case: lines[j] is the dry reassignment inside an if
    if j >= 1 and DRY_REASSIGN_RE.match(lines[j]) and DRY_IF_RE.match(lines[j - 1]):
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
        return ("unmatched", "no canonical `out_path = out_dir / f\"{manifest['run_id']}.json\"`", None)

    # optional mkdir line just above block_top
    top = block_top
    p = top - 1
    while p >= 0 and lines[p].strip() == "":
        p -= 1
    if p >= 0 and MKDIR_RE.match(lines[p]):
        top = p  # absorb mkdir (write_flat_manifest mkdirs itself)

    if not has_out_dir_assignment(lines, top):
        return ("unmatched", "no out_dir assignment above tail", None)

    # Bare-`run_id` out_path is only safe when it provably equals manifest['run_id'].
    if "manifest[" not in lines[block_top] and not runid_is_manifest_runid(src):
        return ("unmatched", "bare run_id out_path not provably == manifest['run_id']", None)

    # Statement-level indent = the indent of the out_path assignment being
    # replaced (NOT the json.dump body indent, which is one level deeper and
    # would produce an IndentationError).
    indent = PRIMARY_RE.match(lines[block_top]).group("indent")
    dry_run_arg = dry_cond if dry_cond is not None else "False"
    seeds_sym = find_seeds_symbol(src)
    seeds_arg = seeds_sym if seeds_sym else "None"
    cfg = config_expr(src)

    replacement = [
        f"{indent}out_path = write_flat_manifest(",
        f"{indent}    manifest,",
        f"{indent}    out_dir,",
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
