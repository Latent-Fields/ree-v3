"""Companion migrator pass: retrofit the always-record-core `elapsed_seconds`
field onto experiment scripts already routed through
`experiments.pack_writer.write_flat_manifest` (batch 1 + batch 2 of the
pack_writer single-writer migration) but lacking a run timer.

Background: `write_flat_manifest` accepts an `elapsed_seconds=` kwarg, but the
batch migrator (`migrate_manifest_writers.py`) could not inject a timer -- it
needs a `_run_started` timestamp captured far from the write tail (at main
entry), which is per-script-variable. So batch-migrated manifests gain 6/7
always-core but show a lone advisory `elapsed_seconds` gap in
`validate_recording` (non-blocking; the hard `manifest_writer_lint` is
unaffected). See pack_writer_single_writer_migration_plan.md sec 7.4.

This pass, mirroring the F pilot (734/735/736/737) hand retrofit, does two
edits ONLY when BOTH preconditions hold:
  (a) `datetime` AND `timezone` are both imported unaliased from the `datetime`
      module (so `datetime.now(timezone.utc)` resolves), with no shadowing bare
      `import datetime`; and
  (b) there is exactly ONE unambiguous `args = <parser>.parse_args()` anchor,
      it is a DIRECT statement of its enclosing function body, and the (single)
      `write_flat_manifest(...)` call is in that SAME function, after the anchor,
      and does not already pass `elapsed_seconds`.

Edits (byte-shape identical to the pilot):
  1. insert `_run_started = datetime.now(timezone.utc)` immediately after the
     `args = <parser>.parse_args()` line (same indentation); and
  2. thread `elapsed_seconds=(datetime.now(timezone.utc) - _run_started).total_seconds(),`
     as a new kwarg line just before the `write_flat_manifest(...)` closing paren.

Where the anchor is ambiguous or the imports are absent/shadowed, the file is
REPORTED unmatched and left untouched -- the advisory gap remains (by design).

Usage:
  python3 tools/retrofit_elapsed_seconds.py --report  <files...>
  python3 tools/retrofit_elapsed_seconds.py --apply   <files...>
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path


def _parent_map(tree: ast.AST) -> dict:
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _enclosing_funcdef(node, parents) -> ast.AST | None:
    """Nearest ancestor FunctionDef/AsyncFunctionDef, or None (module scope)."""
    cur = parents.get(node)
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return cur
        cur = parents.get(cur)
    return None


def datetime_timezone_importable(tree: ast.AST) -> bool:
    """True iff both `datetime` and `timezone` are bound unaliased via
    `from datetime import ...` and no bare `import datetime` shadows `datetime`."""
    have_datetime = False
    have_timezone = False
    shadow_import_datetime = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "datetime":
            for alias in node.names:
                # unaliased binding only (asname None means bound as its own name)
                if alias.asname is None and alias.name == "datetime":
                    have_datetime = True
                if alias.asname is None and alias.name == "timezone":
                    have_timezone = True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "datetime" and (alias.asname is None or alias.asname == "datetime"):
                    # `import datetime` (or `import datetime as datetime`) binds the
                    # MODULE to the name `datetime`, shadowing the class binding and
                    # breaking `datetime.now(...)`.
                    shadow_import_datetime = True
    return have_datetime and have_timezone and not shadow_import_datetime


def find_parse_args_anchors(tree: ast.AST, parents: dict):
    """All `args = <expr>.parse_args(...)` Assign nodes that are DIRECT
    statements of their enclosing function body (not nested in if/for/try/with)."""
    anchors = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if node.targets[0].id != "args":
            continue
        val = node.value
        if not (isinstance(val, ast.Call) and isinstance(val.func, ast.Attribute)
                and val.func.attr == "parse_args"):
            continue
        func = _enclosing_funcdef(node, parents)
        # must be a direct statement of func.body (guarantees _run_started, inserted
        # right after, sits at the function-body level and is visible everywhere later)
        body = func.body if func is not None else getattr(tree, "body", [])
        if node in body:
            anchors.append((node, func))
    return anchors


def find_write_calls(tree: ast.AST):
    """All write_flat_manifest(...) Call nodes."""
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "write_flat_manifest":
            calls.append(node)
    return calls


def call_has_elapsed(call: ast.Call) -> bool:
    """True if the call already supplies a wallclock source. `started_at` counts:
    stamp_recording_core derives elapsed_seconds from started_at (perf_counter delta)
    when elapsed_seconds is absent, so such a call already closes the advisory gap."""
    return any(kw.arg in ("elapsed_seconds", "started_at") for kw in call.keywords)


def _line_indent(line: str) -> str:
    return line[:len(line) - len(line.lstrip())]


def retrofit_one(path: Path):
    """Return (status, detail, new_src_or_None)."""
    src = path.read_text(encoding="utf-8")
    if "write_flat_manifest(" not in src:
        return ("unmatched", "no write_flat_manifest call", None)
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return ("unmatched", f"parse error: {exc}", None)

    parents = _parent_map(tree)
    write_calls = find_write_calls(tree)
    if len(write_calls) != 1:
        return ("unmatched", f"{len(write_calls)} write_flat_manifest calls (want exactly 1)", None)
    call = write_calls[0]
    if call_has_elapsed(call):
        return ("skip", "already has elapsed_seconds/started_at", None)

    if not datetime_timezone_importable(tree):
        return ("unmatched", "datetime/timezone not both importable unaliased (or shadowed)", None)

    anchors = find_parse_args_anchors(tree, parents)
    if len(anchors) != 1:
        return ("unmatched", f"{len(anchors)} unambiguous parse_args anchors (want exactly 1)", None)
    anchor_node, anchor_func = anchors[0]

    call_func = _enclosing_funcdef(call, parents)
    if call_func is not anchor_func:
        return ("unmatched", "write_flat_manifest call not in the parse_args function scope", None)
    # call must come AFTER the anchor lexically
    if call.lineno <= anchor_node.end_lineno:
        return ("unmatched", "write_flat_manifest call precedes the parse_args anchor", None)

    if "_run_started" in src:
        # unexpected for the batch; refuse rather than risk a double-define
        return ("unmatched", "_run_started already present", None)

    lines = src.splitlines()
    keepends_nl = src.endswith("\n")

    # --- edit 2 (do the LATER line first so edit-1's insertion doesn't shift it) ---
    close_line_idx = call.end_lineno - 1  # 0-based line holding the ')'
    open_line_idx = call.func.lineno - 1
    # indentation for the new kwarg: match the last existing arg/kw line; fall back
    # to the call's own indent + 4.
    arg_nodes = list(call.args) + [kw.value for kw in call.keywords]
    if arg_nodes:
        last_arg = max(arg_nodes, key=lambda n: (n.end_lineno, n.end_col_offset))
        kw_indent = _line_indent(lines[last_arg.lineno - 1])
    else:
        kw_indent = _line_indent(lines[open_line_idx]) + "    "

    elapsed_line = (f"{kw_indent}elapsed_seconds="
                    f"(datetime.now(timezone.utc) - _run_started).total_seconds(),")

    if open_line_idx == close_line_idx:
        # single-line call: insert the kwarg before the final ')'
        ln = lines[close_line_idx]
        rp = ln.rstrip().rfind(")")
        head = ln[:rp].rstrip()
        sep = "" if head.endswith(",") or head.endswith("(") else ","
        insert = f" elapsed_seconds=(datetime.now(timezone.utc) - _run_started).total_seconds()"
        lines[close_line_idx] = f"{head}{sep}{insert}" + ln[rp:]
    else:
        # multi-line: ensure the last kwarg line ends with a comma, then insert the
        # new kwarg line just before the ')' line.
        prev_idx = close_line_idx - 1
        if lines[prev_idx].rstrip() and not lines[prev_idx].rstrip().endswith(","):
            lines[prev_idx] = lines[prev_idx].rstrip() + ","
        lines.insert(close_line_idx, elapsed_line)

    # --- edit 1: insert `_run_started = ...` after the parse_args line ---
    pa_idx = anchor_node.end_lineno - 1  # 0-based last line of the assign
    pa_indent = _line_indent(lines[pa_idx])
    lines.insert(pa_idx + 1, f"{pa_indent}_run_started = datetime.now(timezone.utc)")

    new_src = "\n".join(lines)
    if keepends_nl:
        new_src += "\n"
    return ("retrofit", "ok", new_src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    counts = {"retrofit": 0, "skip": 0, "unmatched": 0}
    for f in args.files:
        p = Path(f)
        try:
            status, detail, new_src = retrofit_one(p)
        except Exception as exc:  # never crash the batch on one file
            print(f"[ERROR ] {p.name}: {exc}")
            counts["unmatched"] += 1
            continue
        counts[status] = counts.get(status, 0) + 1
        tag = {"retrofit": "RETROFIT", "skip": "skip    ", "unmatched": "UNMATCH "}[status]
        print(f"[{tag}] {p.name}: {detail}")
        if args.apply and status == "retrofit" and new_src is not None:
            p.write_text(new_src, encoding="utf-8")
    print(f"\n=== {counts['retrofit']} retrofit, {counts['skip']} skip, {counts['unmatched']} unmatched "
          f"(of {len(args.files)}) ===")


if __name__ == "__main__":
    main()
