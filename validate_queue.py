#!/usr/bin/env python3
"""
validate_queue.py -- Validate experiment_queue.json against the expected schema.

Called automatically by experiment_runner.py at startup.
Also run manually after editing the queue:
    /opt/local/bin/python3 validate_queue.py

Exit codes:
    0  -- queue is valid
    1  -- one or more validation errors (details printed to stderr)
    2  -- queue file missing or unparseable JSON
"""

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

QUEUE_FILE = Path(__file__).resolve().parent / "experiment_queue.json"

# ------------------------------------------------------------------
# Valid values for enum fields
# ------------------------------------------------------------------
VALID_STATUSES = {"pending", "claimed", "failed", "suspended"}
VALID_AFFINITIES = {"any", "DLAPTOP-4.local", "Daniel-PC", "EWIN-PC", "ree-cloud-1", "ree-cloud-2", "ree-cloud-3", "ree-cloud-4"}

# queue_id must match: V<gen>-EXQ-<digits>[optional letter][optional -<letter>]
# OR onboarding smoke tests: V3-ONBOARD-smoke-<machine-name>
# Examples: V3-EXQ-047, V3-EXQ-047j, V3-EXQ-001-a, V3-ONBOARD-smoke-EWIN-PC,
#           V4-EXQ-001 (first V4 experiment, self_model_v4:SELF-4, 2026-06-17)
# The generation prefix is V<digits> so the V4/V5 experiment namespaces
# (V4-EXQ-NNN parallel to V3-EXQ-NNN) validate -- precedent set by V4-EXQ-001.
RE_QUEUE_ID = re.compile(r"^V\d+-EXQ-\d+[a-z]?(-[a-z])?$|^V3-ONBOARD-smoke-.+$")

# Contract with experiment_runner.py RE_SAVED_TO (line 73).
# If a script writes a manifest under evidence/experiments, it MUST print
# 'Result written to: <path>' on stdout so the runner captures output_file.
# Without this, runner_status.output_file stays empty and generate_pending_review
# cannot derive the on-disk dir_name -- the manifest exists but appears undiscussed.
# Historical incident: V3-EXQ-325b/325c (2026-04-18/19) used 'Results -> {path}'.
RE_SAVED_TO_IN_SCRIPT = re.compile(r"Result (?:pack )?written to")

# Regression guard against the 2026-05-29 emit_outcome copy-paste bug.
# emit_outcome() in experiment_protocol.py accepts only: outcome, manifest_path,
# run_id, queue_id, exit_reason, extra, signal_dir. Six scripts copy-pasted
# extra kwargs (experiment_type, claim_ids, evidence_direction, results,
# metrics, experiment_purpose, architecture_epoch) that the function has
# never accepted -- the call crashes with TypeError AFTER the manifest is
# written, so the runner classifies ERROR and the Phase 3 writer never
# ingests the result. Canonical incident: V3-EXQ-610a 2026-05-29T22:44Z
# on ree-cloud-3 (manifest preserved as .bak.20260530, rescued via the
# 2026-05-30 sweep). Pattern is permissive of valid kwargs but flags the
# disallowed ones.
RE_EMIT_OUTCOME_CALL = re.compile(r"emit_outcome\s*\([^)]*\)", re.DOTALL)
EMIT_OUTCOME_DISALLOWED_KWARGS = (
    "experiment_type",
    "claim_ids",
    "evidence_direction",
    "results",
    "metrics",
    "experiment_purpose",
    "architecture_epoch",
)

# ------------------------------------------------------------------
# Pre-registration feasibility: share-decomposition non-triviality gate
# ------------------------------------------------------------------
# A gate expressed over a quantity the design ALSO pre-registers a value for is
# checkable before any compute is spent. This check applies that principle to the
# one shape where the arithmetic is unambiguous: a SHARE DECOMPOSITION.
#
# Canonical incident -- V3-EXQ-785 (MECH-463 arousal variance-amplifier decomp,
# 2026-07-19, autopsy REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-785_2026-07-19.md
# section 2a). The script committed, in its own config:
#     REGIMES[1]["expected_incumbent_share"] = 1.043
# while gating that same regime on precondition
#     n_components_with_nontrivial_share  ( >= 2 components holding |share| > 0.01 )
# The decomposition is a FULL covariance attribution, so its shares sum to exactly
# 1.0 by construction. An incumbent share at or above unity therefore leaves <= 0
# for every other component combined, and the ">= 2 non-trivial" gate cannot be
# satisfied. The regime could not pass its own gate, and the number proving it was
# sitting in the config. Realised cost: 460s of compute, one vacated arm, and a
# GREEN arm's well-powered finding buried under a whole-run "substrate not ready"
# label until an autopsy recovered it.
#
# TRIGGER (deliberately narrow -- both conditions must hold in one script):
#   (a) a non-triviality COUNT gate over a shares mapping: a counting comprehension
#       of the form  sum(1 for v in <...share...> if abs(v) > <floor>)  . Such a gate
#       is only meaningful when it demands >= 2 components (a count gate of >= 1 is
#       vacuous), so the required-count threshold is NOT parsed -- see below.
#   (b) a PRE-REGISTERED share literal >= 1.0 under a dict key naming it as expected
#       / pre-registered / predicted.
#
# WHY >= 1.0 AND NOT THE TIGHTER INEQUALITY. The general non-negativity condition is
# infeasible iff  (1 - S) < floor * (K - 1)  -- which for S=0.995, floor=0.01 would
# also fire. That is deliberately NOT implemented. Real covariance attributions do
# produce small negative components (785 measured f = -0.0013), so a near-unity
# pre-registration is a margin judgement, not an arithmetic impossibility. Firing on
# judgement calls would get this check routed around, which is worse than not having
# it. S >= 1.0 leaves literally nothing for the other components and needs no
# tolerance argument. Keep it that way.
#
# Scoped to 'share' mappings on purpose: sum-to-one is what makes >= 1.0 fatal.
# An unrelated counting comprehension over a non-share mapping must not trip this.
RE_PREREG_SHARE_KEY = re.compile(
    r"(?:expect|prereg|pre_reg|pre_registered|predicted|declared)\w*share"
    r"|share\w*(?:expect|prereg|pre_reg|predicted|declared)",
    re.IGNORECASE,
)

# Names/descriptions identifying the precondition, used only to make the error
# message point at the actual gate. Absence downgrades to a line number.
RE_NONTRIVIAL_SHARE_LABEL = re.compile(
    r"non[_\-\s]?trivial.*share|share.*non[_\-\s]?trivial", re.IGNORECASE
)


def _module_numeric_constants(tree) -> "dict[str, float]":
    """Module-level NAME = <number> bindings, for resolving a gate floor that is
    factored into a named constant rather than inlined.

    V3-EXQ-785 inlined its floor (`abs(v) > 0.01`); its successor 785a factored it
    into NONTRIVIAL_SHARE_FLOOR. Without this resolution the check would silently
    stop applying to the successor lineage -- i.e. exactly the scripts most likely
    to inherit the defect. Only simple module-level literals are resolved; anything
    computed is left unresolved and the gate is skipped (fail-soft, never a guess).
    """
    out: "dict[str, float]" = {}
    for node in getattr(tree, "body", []):
        if not isinstance(node, ast.Assign):
            continue
        val = node.value
        if not (isinstance(val, ast.Constant)
                and isinstance(val.value, (int, float))
                and not isinstance(val.value, bool)):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                out[tgt.id] = float(val.value)
    return out


def _share_nontriviality_gate(tree) -> "tuple[float, int] | None":
    """Locate a share-decomposition non-triviality COUNT gate.

    Matches a counting comprehension  sum(1 for v in <iter> if abs(v) > <floor>)
    where <iter> mentions 'share'. Returns (floor, lineno) for the first match,
    or None. Structural (AST) rather than name-based so it does not depend on the
    785 script's particular identifiers.
    """
    consts = _module_numeric_constants(tree)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "sum"
                and len(node.args) == 1
                and isinstance(node.args[0], ast.GeneratorExp)):
            continue
        gen = node.args[0]
        # elt must be the literal 1 (a COUNT, not a magnitude sum).
        if not (isinstance(gen.elt, ast.Constant)
                and isinstance(gen.elt.value, int)
                and not isinstance(gen.elt.value, bool)
                and gen.elt.value == 1):
            continue
        if not gen.generators:
            continue
        comp = gen.generators[0]
        try:
            iter_src = ast.unparse(comp.iter)
        except Exception:
            continue
        if "share" not in iter_src.lower():
            continue
        # condition must be an absolute-value floor test: abs(x) > <number>
        for cond in comp.ifs:
            if not (isinstance(cond, ast.Compare)
                    and len(cond.ops) == 1
                    and isinstance(cond.ops[0], ast.Gt)
                    and isinstance(cond.left, ast.Call)
                    and isinstance(cond.left.func, ast.Name)
                    and cond.left.func.id == "abs"):
                continue
            rhs = cond.comparators[0]
            if isinstance(rhs, ast.Constant) and isinstance(rhs.value, (int, float)) \
                    and not isinstance(rhs.value, bool):
                return (float(rhs.value), getattr(node, "lineno", 0))
            # Floor factored into a module-level named constant (785a pattern).
            if isinstance(rhs, ast.Name) and rhs.id in consts:
                return (consts[rhs.id], getattr(node, "lineno", 0))
    return None


def _preregistered_shares_at_or_above_unity(tree) -> "list[tuple[str, float, int]]":
    """Pre-registered share literals >= 1.0 in dict literals.

    Returns a list of (key, value, lineno). Only plain numeric Constants are read;
    a computed expression is not a pre-registration this check can reason about.
    """
    out: "list[tuple[str, float, int]]" = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, val in zip(node.keys, node.values):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                continue
            if not RE_PREREG_SHARE_KEY.search(key.value):
                continue
            if not (isinstance(val, ast.Constant)
                    and isinstance(val.value, (int, float))
                    and not isinstance(val.value, bool)):
                continue
            if float(val.value) >= 1.0:
                out.append((key.value, float(val.value),
                            getattr(val, "lineno", getattr(node, "lineno", 0))))
    return out


def prereg_share_feasibility_lint(source: str) -> list[str]:
    """Reject a pre-registered incumbent share that its own non-triviality gate
    makes unreachable. Returns a list of message bodies (empty == clean).

    Fail-soft: an unparseable script yields no findings (other checks report it).
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []

    gate = _share_nontriviality_gate(tree)
    if gate is None:
        return []
    floor, gate_line = gate

    offenders = _preregistered_shares_at_or_above_unity(tree)
    if not offenders:
        return []

    # Name the precondition if the script labels it, else point at the line.
    # Identifier-like strings ONLY (the precondition's `name` field, e.g.
    # "n_components_with_nontrivial_share"). Prose is deliberately excluded: a
    # docstring describing the gate matches the same regex, and quoting it back as
    # if it were the precondition name sends the author to the wrong place.
    label = ""
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        cand = node.value.strip()
        if len(cand.split()) != 1 or len(cand) > 80:
            continue
        if RE_NONTRIVIAL_SHARE_LABEL.search(cand):
            label = cand
            break
    gate_ref = f"'{label}'" if label else f"the gate at line {gate_line}"

    findings = []
    for key, value, lineno in offenders:
        findings.append(
            f"pre-registered '{key}' = {value} (line {lineno}) is unreachable under "
            f"its own precondition {gate_ref} (line {gate_line}), which requires at "
            f"least 2 components holding |share| > {floor}. Decomposition shares sum "
            f"to 1.0 by construction, so an incumbent share at or above unity leaves "
            f"<= 0 for all other components combined and none can clear the floor -- "
            f"the gate is unsatisfiable before the run starts. Either re-check the "
            f"pre-registered value, or condition the gate on the regimes where it is "
            f"meaningful (the script's own P1 note gives the pattern). Canonical "
            f"incident: V3-EXQ-785 2026-07-19, 460s burned on an un-passable gate "
            f"(failure_autopsy_V3-EXQ-785_2026-07-19.md section 2a)."
        )
    return findings


# ------------------------------------------------------------------
# Re-derive brake backstop (MOVE-3, assembly_vs_closure_plan.md)
# ------------------------------------------------------------------
# WARN-ONLY code gate hardening the /failure-autopsy Step 7 + /queue-experiment
# Step 2.5b "re-derive brake" so a hand-edited queue append cannot bypass it.
# The skill-doc brake stops the 7-12x-lettered-re-run-circling-one-substrate-ceiling
# pathology: on the Nth (default 2) substrate_ceiling/non_contributory autopsy for the
# same claim, /queue-experiment refuses a new same-granularity test of that claim
# unless the named upstream substrate now shows IMPLEMENTED/VALIDATED in
# ree-v3/CLAUDE.md. This validator re-applies the EXACT counting logic at
# queue-validate time (PreToolUse commit hook + runner startup) and WARNs (never
# blocks) so a queue entry that the skill would have braked is surfaced even when the
# skill was bypassed. Warn-only by design (user-confirmed 2026-06-22), mirroring
# validate_claims.py's warn-only enum checks; elevate to ERROR once it stabilises.
RE_DERIVE_BRAKE_THRESHOLD = 2

# Matches the trailing _YYYY-MM-DD date in a failure_autopsy_<slug>_YYYY-MM-DD.json
# filename, used to pick the MOST RECENT counted autopsy (the one whose
# recommended_substrate_queue_entry names the current upstream substrate).
RE_AUTOPSY_DATE = re.compile(r"_(\d{4}-\d{2}-\d{2})\.json$")

# ------------------------------------------------------------------
# Field specs: (field_name, required, expected_type_or_None_for_any)
# ------------------------------------------------------------------
TOP_LEVEL_REQUIRED = [
    ("schema_version", True, str),
    ("calibration", True, dict),
    ("items", True, list),
]

ITEM_REQUIRED = [
    ("queue_id", True, str),
    ("script", True, str),
    ("priority", True, int),
    ("machine_affinity", True, str),
    ("status", True, str),
    ("estimated_minutes", True, (int, float)),
]

ITEM_OPTIONAL = [
    ("note", False, str),
    ("title", False, str),
    ("backlog_id", False, str),
    ("claim_id", False, str),
    ("claim_ids", False, list),
    ("supersedes", False, str),
    ("claimed_by", False, (dict, type(None))),
    ("machine_affinity_note", False, str),
    ("force_rerun", False, bool),
    ("experiment_type", False, str),
    ("checkpoint_resumable", False, bool),
    ("checkpoint_experiment_type", False, str),
    ("checkpoint_path", False, str),
    ("suspended_at", False, str),
]


# ------------------------------------------------------------------
# Per-machine runner_status scan (silent re-queue guard)
# ------------------------------------------------------------------
# Historical incidents (canonical: EXQ-126 on 2026-04-20/21) showed that a
# previously-run queue_id can be re-added to the queue and silently re-executed
# when its original completion record is not present in the local per-machine
# status file -- e.g. the completion was recorded under a prior hostname
# (Mac -> DLAPTOP-4.local), or on a different machine whose status file is
# offline. The runner only checks the local per-machine file + any peer files
# it can see at startup. If none of those contain the queue_id, dedup silently
# passes and the experiment runs again.
#
# This guard scans every per-machine runner_status file in REE_assembly and
# raises a validation error on any queue_id that already has a completion
# record, unless the queue item carries force_rerun: true. New letter/number
# suffix IDs (EXQ-126a, EXQ-127) are the normal path; force_rerun is the
# explicit escape hatch for the rare case where re-using the same ID is
# intentional (e.g. the prior record is from a superseded contamination epoch).

_REE_ASSEMBLY_STATUS_DIR_CANDIDATES = [
    QUEUE_FILE.parent.parent / "REE_assembly" / "evidence" / "experiments" / "runner_status",
    Path.home() / "REE_Working" / "REE_assembly" / "evidence" / "experiments" / "runner_status",
]


def _find_status_dir() -> Path | None:
    for cand in _REE_ASSEMBLY_STATUS_DIR_CANDIDATES:
        if cand.is_dir():
            return cand
    return None


def _scan_completed_queue_ids() -> dict[str, list[tuple[str, str, str]]]:
    """Scan per-machine runner_status files for completed queue_ids.

    Returns a dict mapping queue_id -> list of (machine_file, result, completed_at).
    Returns an empty dict (fail-soft) if the status dir is missing or unreadable.
    """
    status_dir = _find_status_dir()
    if status_dir is None:
        return {}
    out: dict[str, list[tuple[str, str, str]]] = {}
    for f in sorted(status_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for entry in data.get("completed", []) or []:
            qid = entry.get("queue_id", "")
            if not qid:
                continue
            out.setdefault(qid, []).append(
                (f.name, entry.get("result", "?"), entry.get("completed_at", ""))
            )
    return out


# ------------------------------------------------------------------
# Re-derive brake helpers (MOVE-3 backstop)
# ------------------------------------------------------------------
_REE_ASSEMBLY_PLANNING_DIR_CANDIDATES = [
    QUEUE_FILE.parent.parent / "REE_assembly" / "evidence" / "planning",
    Path.home() / "REE_Working" / "REE_assembly" / "evidence" / "planning",
]

_REE_V3_CLAUDE_MD_CANDIDATES = [
    QUEUE_FILE.parent / "CLAUDE.md",
    Path.home() / "REE_Working" / "ree-v3" / "CLAUDE.md",
]


def _find_planning_dir() -> "Path | None":
    for cand in _REE_ASSEMBLY_PLANNING_DIR_CANDIDATES:
        if cand.is_dir():
            return cand
    return None


def _read_ree_v3_claude_md() -> str:
    """Read ree-v3/CLAUDE.md text; '' (fail-soft) if it cannot be read."""
    for cand in _REE_V3_CLAUDE_MD_CANDIDATES:
        try:
            if cand.is_file():
                return cand.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
    return ""


def _scan_substrate_ceiling_autopsies() -> "dict[str, list[tuple[str, str, dict]]]":
    """Scan failure_autopsy_*.json for substrate_ceiling / non_contributory readings.

    Returns claim_id -> list of (autopsy_filename, date_str, matching_target_dict),
    one entry per (file, claim) using the first matching target in that file --
    EXACTLY the count the /queue-experiment Step 2.5b + /failure-autopsy Step 7
    snippet produces (it appends the filename once per claim and ``break``s at the
    first matching target). Fail-soft to {} if the planning dir is missing.
    """
    planning = _find_planning_dir()
    if planning is None:
        return {}
    out: "dict[str, list[tuple[str, str, dict]]]" = {}
    for f in sorted(planning.glob("failure_autopsy_*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        m = RE_AUTOPSY_DATE.search(f.name)
        date_str = m.group(1) if m else ""
        # First matching target per claim in this file (mirrors the skill's break).
        first_match: "dict[str, dict]" = {}
        for t in data.get("targets", []) or []:
            if not isinstance(t, dict):
                continue
            cat = str(t.get("recommended_epistemic_category") or "")
            direction = str(t.get("recommended_evidence_direction") or "")
            if not ("substrate_ceiling" in cat or "non_contributory" in direction):
                continue
            for claim in t.get("claim_ids", []) or []:
                if isinstance(claim, str) and claim not in first_match:
                    first_match[claim] = t
        for claim, t in first_match.items():
            out.setdefault(claim, []).append((f.name, date_str, t))
    return out


def _upstream_substrate_from_target(target: dict) -> str:
    """The named upstream substrate the brake routes to, per the skill's lookup order:
    recommended_substrate_queue_entry.target_sd_id / .sd_id_suggested, then
    re_derive_brake.upstream_substrate. '' when none is recorded."""
    rsqe = target.get("recommended_substrate_queue_entry") or {}
    if isinstance(rsqe, dict):
        for key in ("target_sd_id", "sd_id_suggested"):
            val = rsqe.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    rdb = target.get("re_derive_brake") or {}
    if isinstance(rdb, dict):
        val = rdb.get("upstream_substrate")
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _substrate_is_built(substrate_id: str, claude_md_text: str) -> bool:
    """True iff `substrate_id` appears on a SINGLE ree-v3/CLAUDE.md line that also
    carries IMPLEMENTED or VALIDATED -- the brake-release condition.

    Same-line (not windowed) is deliberate: CLAUDE.md status declarations put the id
    and its IMPLEMENTED/VALIDATED token on one physical line ('- SD-058 ... -- IMPLEMENTED'),
    while a windowed match would falsely clear e.g. 'f_dominance_conversion_ceiling'
    off a NEARBY but unrelated 'natural_commit_occupancy_release -- IMPLEMENTED' header.
    The id is matched as a standalone token so 'MECH-448' does not match 'MECH-4480'."""
    if not substrate_id or not claude_md_text:
        return False
    id_pat = re.compile(r"(?<![\w-])" + re.escape(substrate_id) + r"(?![\w-])")
    for line in claude_md_text.splitlines():
        if ("IMPLEMENTED" in line or "VALIDATED" in line) and id_pat.search(line):
            return True
    return False


def _type_name(t) -> str:
    if isinstance(t, tuple):
        return "/".join(x.__name__ for x in t if x is not type(None))
    return t.__name__


def _is_tracked(repo_root: Path, rel_path: str) -> bool:
    # Returns True iff `rel_path` is tracked in the git index at repo_root.
    # If git is unavailable (no .git, no git binary), we fail-open: return
    # True so non-git checkouts (e.g. fresh tarball extractions in CI sanity
    # checks) do not spuriously fail validation. The real production path
    # always has git.
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", rel_path],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return True  # fail-open when git is unavailable
    return result.returncode == 0


def _validate_run_axis(prefix: str, field_name: str, value, elem_type) -> list[str]:
    """Validate seeds/conditions fields which may be counts or explicit lists."""
    errors: list[str] = []
    if isinstance(value, bool):
        return [f"{prefix}: '{field_name}' must be int or list, got bool"]
    if isinstance(value, int):
        if value <= 0:
            errors.append(f"{prefix}: '{field_name}' must be > 0, got {value}")
        return errors
    if isinstance(value, list):
        if not value:
            errors.append(f"{prefix}: '{field_name}' list must not be empty")
            return errors
        for sub_idx, sub_val in enumerate(value):
            if isinstance(sub_val, bool) or not isinstance(sub_val, elem_type):
                errors.append(
                    f"{prefix}: '{field_name}[{sub_idx}]' must be {elem_type.__name__}, "
                    f"got {type(sub_val).__name__}"
                )
                continue
            if elem_type is int and sub_val <= 0:
                errors.append(
                    f"{prefix}: '{field_name}[{sub_idx}]' must be > 0, got {sub_val}"
                )
            if elem_type is str and not sub_val.strip():
                errors.append(
                    f"{prefix}: '{field_name}[{sub_idx}]' must not be empty"
                )
        return errors
    return [f"{prefix}: '{field_name}' must be int or list, got {type(value).__name__}"]


# Non-blocking advisories collected by the most recent validate() call.
# Kept separate from the returned error list so callers (the runner, the
# preflight test) keep their exit-code contract -- warnings never block a
# commit or a runner start. main() prints them; validate() resets + fills it.
_LAST_WARNINGS: list[str] = []


def validate(queue_path: Path = QUEUE_FILE) -> list[str]:
    """
    Validate the queue file.  Returns a list of error strings.
    Empty list means the queue is valid.

    Non-fatal advisories (e.g. an item carrying no claim tag) are appended to
    the module-level ``_LAST_WARNINGS`` list rather than the returned errors,
    so they surface to a human without failing the commit hook.
    """
    errors: list[str] = []
    _LAST_WARNINGS.clear()

    # --- 1. Parse JSON ---
    try:
        raw = queue_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        errors.append(f"Queue file not found: {queue_path}")
        return errors

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        errors.append(f"JSON parse error: {exc}")
        return errors

    if not isinstance(data, dict):
        errors.append("Top-level value must be a JSON object.")
        return errors

    # --- 2. Top-level fields ---
    for fname, required, ftype in TOP_LEVEL_REQUIRED:
        val = data.get(fname)
        if val is None:
            if required:
                errors.append(f"Missing required top-level field: '{fname}'")
        elif not isinstance(val, ftype):
            errors.append(
                f"Top-level '{fname}' must be {_type_name(ftype)}, "
                f"got {type(val).__name__}"
            )

    if data.get("schema_version") not in (None, "v1"):
        errors.append(
            f"Unknown schema_version '{data['schema_version']}' -- expected 'v1'"
        )

    items = data.get("items")
    if not isinstance(items, list):
        # Already reported above; bail to avoid cascading noise
        return errors

    # --- 3. Per-item validation ---
    seen_ids: dict[str, int] = {}
    completed_scan = _scan_completed_queue_ids()

    # Re-derive brake backstop (MOVE-3): scan substrate_ceiling/non_contributory
    # autopsies once, but only when at least one item carries a claim tag (avoids the
    # ~150-file scan on a claimless queue). Read ree-v3/CLAUDE.md once for the
    # brake-release (substrate IMPLEMENTED/VALIDATED) check. Both fail-soft to empty.
    _any_claim_tagged = any(
        isinstance(it, dict)
        and (
            (isinstance(it.get("claim_id"), str) and it.get("claim_id").strip())
            or (isinstance(it.get("claim_ids"), list) and len(it.get("claim_ids")) > 0)
        )
        for it in items
    )
    _brake_autopsies = _scan_substrate_ceiling_autopsies() if _any_claim_tagged else {}
    _brake_claude_md = _read_ree_v3_claude_md() if _brake_autopsies else ""

    for idx, item in enumerate(items):
        prefix = f"items[{idx}]"

        if not isinstance(item, dict):
            errors.append(f"{prefix}: each item must be a JSON object")
            continue

        queue_id = item.get("queue_id", f"<unknown at index {idx}>")
        prefix = f"items[{idx}] ({queue_id})"

        # Required fields
        for fname, required, ftype in ITEM_REQUIRED:
            val = item.get(fname)
            if val is None:
                if required:
                    errors.append(f"{prefix}: missing required field '{fname}'")
            elif not isinstance(val, ftype):
                errors.append(
                    f"{prefix}: '{fname}' must be {_type_name(ftype)}, "
                    f"got {type(val).__name__}"
                )

        # Optional fields -- type-check if present
        for fname, _required, ftype in ITEM_OPTIONAL:
            val = item.get(fname)
            if val is not None and not isinstance(val, ftype):
                errors.append(
                    f"{prefix}: '{fname}' must be {_type_name(ftype)}, "
                    f"got {type(val).__name__}"
                )

        # queue_id format
        if isinstance(queue_id, str):
            if not RE_QUEUE_ID.match(queue_id):
                errors.append(
                    f"{prefix}: queue_id '{queue_id}' does not match expected pattern "
                    f"V3-EXQ-<digits>[letter][-letter] "
                    f"(e.g. V3-EXQ-047, V3-EXQ-047j, V3-EXQ-001-a)"
                )
            # Duplicate check
            if queue_id in seen_ids:
                errors.append(
                    f"{prefix}: duplicate queue_id '{queue_id}' "
                    f"(first seen at index {seen_ids[queue_id]})"
                )
            else:
                seen_ids[queue_id] = idx

        # machine_affinity enum
        affinity = item.get("machine_affinity")
        if isinstance(affinity, str) and affinity not in VALID_AFFINITIES:
            errors.append(
                f"{prefix}: machine_affinity '{affinity}' is not a recognised value "
                f"({', '.join(sorted(VALID_AFFINITIES))})"
            )

        # status enum
        status = item.get("status")
        if isinstance(status, str) and status not in VALID_STATUSES:
            errors.append(
                f"{prefix}: status '{status}' is not a recognised value "
                f"({', '.join(sorted(VALID_STATUSES))})"
            )

        # estimated_minutes must be > 0
        est = item.get("estimated_minutes")
        if isinstance(est, (int, float)) and est <= 0:
            errors.append(
                f"{prefix}: estimated_minutes must be > 0, got {est}"
            )

        # claim-tag presence (WARN, non-blocking). Every queue item should declare
        # either claim_id / a non-empty claim_ids (the claim(s) it tests) OR an
        # explicit claim_ids: [] marking an intentional claimless diagnostic.
        # Items declaring NEITHER evade per-claim governance trend tracking -- the
        # 39-untagged-ERROR class flagged in insights_report.md 2026-06-06. An
        # explicit claim_ids: [] silences this (use it for substrate-readiness /
        # ablation runs that legitimately map to no claims.yaml id).
        _cid = item.get("claim_id")
        _has_claim_id = isinstance(_cid, str) and _cid.strip()
        _cids = item.get("claim_ids")
        _has_claim_ids = isinstance(_cids, list) and len(_cids) > 0
        _declares_claimless = "claim_ids" in item and isinstance(_cids, list)
        if not _has_claim_id and not _has_claim_ids and not _declares_claimless:
            _LAST_WARNINGS.append(
                f"{prefix}: no claim tag -- set claim_id / claim_ids to the "
                f"claim(s) under test, or claim_ids: [] to mark an intentional "
                f"claimless diagnostic."
            )

        # Re-derive brake backstop (MOVE-3, WARN-only). For every claim this item
        # tags, count the prior substrate_ceiling / non_contributory autopsies; if a
        # claim has >= RE_DERIVE_BRAKE_THRESHOLD and the named upstream substrate is
        # not yet IMPLEMENTED/VALIDATED in ree-v3/CLAUDE.md, this is the same-claim
        # re-test the /queue-experiment Step 2.5b brake refuses. Warn (do not block) --
        # a redesign of a DIFFERENT mechanism, a commitment-free read, or a diagnostic
        # is exempt and cannot be told apart from the queue entry alone. An item whose
        # note documents a brake clearance is skipped (the skill records the release).
        if _brake_autopsies:
            _note = item.get("note")
            _note_clears = isinstance(_note, str) and "re-derive brake" in _note.lower()
            if not _note_clears:
                _claims_to_check: list[str] = []
                if _has_claim_id:
                    _claims_to_check.append(_cid.strip())
                if isinstance(_cids, list):
                    _claims_to_check.extend(
                        c.strip() for c in _cids if isinstance(c, str) and c.strip()
                    )
                for _claim in dict.fromkeys(_claims_to_check):  # dedup, preserve order
                    _counted = _brake_autopsies.get(_claim, [])
                    if len(_counted) < RE_DERIVE_BRAKE_THRESHOLD:
                        continue
                    # Most recent counted autopsy by filename date (fallback: last).
                    _recent = max(_counted, key=lambda e: e[1])
                    _upstream = _upstream_substrate_from_target(_recent[2])
                    if _upstream and _substrate_is_built(_upstream, _brake_claude_md):
                        continue  # brake released -- substrate now built
                    _slugs = ", ".join(
                        e[0].replace(".json", "") for e in sorted(_counted, key=lambda e: e[1])[-3:]
                    )
                    _sub_txt = (
                        f"the named upstream substrate '{_upstream}' is not yet "
                        f"IMPLEMENTED/VALIDATED in ree-v3/CLAUDE.md"
                        if _upstream
                        else "no upstream substrate is named in the latest counted autopsy"
                    )
                    _LAST_WARNINGS.append(
                        f"{prefix}: re-derive brake -- claim '{_claim}' has "
                        f"{len(_counted)} substrate_ceiling/non_contributory autopsies "
                        f"on record (recent: {_slugs}) and {_sub_txt}. A same-granularity "
                        f"lettered re-test re-derives the ceiling; build the upstream "
                        f"substrate via /implement-substrate first (see /queue-experiment "
                        f"Step 2.5b). EXEMPT (verify, then ignore): a redesign of a "
                        f"different mechanism, a commitment-free read, or a diagnostic."
                    )

        if "seeds" in item:
            errors.extend(_validate_run_axis(prefix, "seeds", item["seeds"], int))
        if "conditions" in item:
            errors.extend(_validate_run_axis(prefix, "conditions", item["conditions"], str))

        # script field -- require the file to be both on disk AND tracked in git.
        # An untracked-but-on-disk script passes Path.exists() in the producer's
        # checkout but fails on every consumer that pulls the queue (the 2026-05-27
        # ree-cloud-1/2/3 fleet wedge: V3-EXQ-610 queue entry committed without its
        # script via a parallel-session in-file edit; producer's validate passed
        # because the untracked script existed on disk locally; every cloud worker
        # crashed at startup). git ls-files is the authoritative check.
        script_val = item.get("script")
        if isinstance(script_val, str):
            script_path = queue_path.parent / script_val
            if not script_path.exists():
                errors.append(
                    f"{prefix}: script file not found on disk: {script_val}"
                )
            elif not _is_tracked(queue_path.parent, script_val):
                errors.append(
                    f"{prefix}: script file exists on disk but is not tracked in "
                    f"git (untracked or ignored): {script_val}. Run `git add "
                    f"{script_val}` in the same commit as the queue entry to "
                    f"prevent pulling consumers from crashing at validate startup."
                )
            else:
                # Manifest-writing scripts must print the runner-matching save line.
                try:
                    source = script_path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    source = ""
                writes_manifest = (
                    "json.dump(" in source and "evidence/experiments" in source
                )
                if writes_manifest and not RE_SAVED_TO_IN_SCRIPT.search(source):
                    errors.append(
                        f"{prefix}: script {script_val} writes a JSON manifest "
                        f"under evidence/experiments but does not print "
                        f"'Result written to: <path>' -- experiment_runner.py "
                        f"RE_SAVED_TO will not capture output_file. "
                        f"Add: print(f\"Result written to: {{out_path}}\", flush=True)"
                    )

                # Regression guard: emit_outcome() must not be called with
                # disallowed kwargs (see RE_EMIT_OUTCOME_CALL above).
                for call_match in RE_EMIT_OUTCOME_CALL.finditer(source):
                    call_text = call_match.group(0)
                    for bad_kwarg in EMIT_OUTCOME_DISALLOWED_KWARGS:
                        if re.search(rf"\b{bad_kwarg}\s*=", call_text):
                            errors.append(
                                f"{prefix}: script {script_val} calls "
                                f"emit_outcome({bad_kwarg}=...) but the "
                                f"function never accepted this kwarg. "
                                f"emit_outcome's signature is "
                                f"(outcome, manifest_path, *, run_id, "
                                f"queue_id, exit_reason, extra, signal_dir). "
                                f"Drop the disallowed kwarg(s); the call "
                                f"will crash AFTER the manifest is written "
                                f"and the runner will mis-classify ERROR "
                                f"(canonical incident: V3-EXQ-610a 2026-05-29)."
                            )
                            break  # one error per call site is enough

                # Pre-registration feasibility: a pre-registered share >= 1.0
                # cannot coexist with a ">= 2 non-trivial components" gate over a
                # sum-to-one decomposition (V3-EXQ-785, 2026-07-19).
                for finding in prereg_share_feasibility_lint(source):
                    errors.append(f"{prefix}: script {script_val} {finding}")

        # Silent re-queue guard: queue_id must not already have a completion
        # record in any per-machine runner_status file, unless force_rerun=true.
        if isinstance(queue_id, str) and queue_id in completed_scan:
            if item.get("force_rerun") is not True:
                records = completed_scan[queue_id]
                rec_strs = "; ".join(
                    f"{mfile} ({result} at {cat})" for mfile, result, cat in records
                )
                errors.append(
                    f"{prefix}: queue_id already has a completion record in "
                    f"{rec_strs}. The runner WILL silently skip or re-run under a "
                    f"lost-completion edge case. Use a new letter/number suffix "
                    f"(EXQ-126a, EXQ-127, ...), or set 'force_rerun': true to "
                    f"intentionally re-run under the same ID."
                )

        # claimed_by structure check
        claimed_by = item.get("claimed_by")
        if isinstance(claimed_by, dict):
            for sub in ("machine", "claimed_at"):
                if sub not in claimed_by:
                    errors.append(
                        f"{prefix}: claimed_by missing sub-field '{sub}'"
                    )

    return errors


def main() -> int:
    errors = validate()
    if _LAST_WARNINGS:
        print(f"Queue advisories -- {len(_LAST_WARNINGS)} warning(s):", file=sys.stderr)
        for w in _LAST_WARNINGS:
            print(f"  WARN: {w}", file=sys.stderr)
    if errors:
        print(f"Queue validation FAILED -- {len(errors)} error(s):", file=sys.stderr)
        for e in errors:
            print(f"  ERROR: {e}", file=sys.stderr)
        return 1
    print(f"Queue OK -- {QUEUE_FILE.name} is valid.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
