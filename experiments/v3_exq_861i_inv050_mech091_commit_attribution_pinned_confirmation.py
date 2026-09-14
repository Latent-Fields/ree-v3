"""
V3-EXQ-861i -- pinned commit-attribution confirmation for the INV-050 ree_core
bisect: does ree-v3 commit 6293b23 (MECH-091 phase_reset() triggers) carry the
WHOLE V3-EXQ-861e seed-271 ARM_3_HIGH_ON MEL collapse under the FULL 861e
protocol, as the wake-only desk bisect predicts?

SLEEP DRIVER: manual-cycle-loop (force_cycle() called once per cycle in a
dedicated MEAS_CYCLES wake-sleep loop) -- unchanged from 861/861c/861e/861g.

RED-TEAM (fable, Step 4.5): CONTESTED -> 2 findings fixed (label/reading only); see RED_TEAM_VERDICT.

================================================================================
WHERE THIS COMES FROM
================================================================================
REE_assembly/evidence/planning/inv050_reecore_bisect_20260911.md (origin/master
8e1216d2c3) replayed 861e's ARM_3_HIGH_ON agent+env as a SHORT WAKE-ONLY rollout
against every ree_core commit in f810969..17befb8c and compared the per-step
MEL producer stream bitwise. One clean divergence point on every probe (3 seeds,
2 rollout lengths): 6293b23 ("MECH-091: wire task-completion and
commitment-boundary-crossing triggers into phase_reset()"). Its parent is
5f64a53f. The standing candidate 76cbf84 is excluded four independent ways and
is NOT re-opened here.

The bisect itself says what it does NOT establish: it never ran training, the
10 calibration draws, or sleep. Those code paths are exactly where the other
ten commits in the range could still act. So "6293b23 is the mover" is a
wake-path localisation, and attributing the recorded 0.884 factor to it wants
the full protocol at the commit boundary. That is this run.

================================================================================
DESIGN -- four pinned cells, one process per pin
================================================================================
Every cell is 861e's decisive cell, byte-identical cell logic (sliced verbatim
from 861g, itself byte-identical to 861e): seed 271, ARM_3_HIGH_ON,
CALIB_DRAWS=10, legacy argmin ContextMemory write path, no reseed.

  cell id              ree_core pinned to     role
  ANCHOR_PRE_f810969   f810969 (861c/861g)    in-run anchor: pre-range endpoint
  PIN_A_5f64a53f       5f64a53f (6293b23^)    the commit boundary, before
  PIN_B_6293b23        6293b23                the commit boundary, after
  ANCHOR_POST_17befb8c 17befb8c (861e)        in-run anchor: post-range endpoint

WHY FOUR AND NOT TWO. experiments/_lib/substrate_pin.py pins ONLY ree_core/.
experiments/_lib/**, the harness and the recording code come from the LIVE
checkout, which has moved since 861e/861g ran. A two-cell A-vs-B run could not
tell "6293b23 does not reproduce the recorded 0.884" apart from "the live
harness no longer reproduces 861e at all". The anchors make every comparison
IN-RUN (same process tree, same box, same live harness), so the attribution
never leans on a cross-run number. Their agreement with the RECORDED 861g/861e
cells is still measured, as a provenance qualifier on the label.

WHY ONE SUBPROCESS PER PIN. substrate_pin.pin_ree_core() refuses to run if
ree_core is already imported (the pin would silently no-op). A process can hold
exactly one pinned ree_core, so the parent process never imports ree_core; it
re-invokes this same file with --child-cell --pin-cell-id <id> per pin, the
child pins + verifies + runs one cell under arm_cell() and writes its row to a
scratch JSON outside the repo, and the parent aggregates and writes ONE manifest.
Child stdout/stderr are inherited, so the runner sees every [train]/Seed/verdict
line. A non-zero child exit raises in the parent (ERROR, never a partial
manifest).

PIN VERIFICATION -- three independent checks per cell, all fatal:
  (1) structural: ree_core.__file__ under the pin dir (substrate_pin.verify_pin);
  (2) behavioural markers, a chain that separates all four refs:
        ref        MultiArchive  agent.py phase_reset() lines  authority_spread_ratio
        f810969    absent        1                              absent
        5f64a53f   present       1                              absent
        6293b23    present       6                              absent
        17befb8c   present       6                              present
  (3) full content: every ree_core/**/*.py blob in the pin dir hashed with git's
      blob format and compared to `git ls-tree -r <sha> ree_core`, both
      directions (no missing, no extra, no altered file).
  A cell that cannot prove its substrate raises SubstratePinError -> ERROR.

================================================================================
DECISIVE READOUT AND PRE-REGISTERED CRITERIA
================================================================================
Per cell: mean_mel (mean measured MEL across the 6 sleep cycles) and
mel_reference (that cell's own 10-draw calibrated reference). Their ratio,
clamped, is mean_duration_factor -- the 0.884 figure.

D(x, y) := max(reldiff(mean_mel_x, mean_mel_y), reldiff(mel_reference_x,
mel_reference_y)), reldiff(a, b) = |a - b| / max(|a|, |b|, 1e-30).
Comparing both components rather than the clamped factor means a coincidental
factor match cannot pass.

  P1 (readiness precondition, load-bearing for every branch):
     D(ANCHOR_PRE, ANCHOR_POST) >= ANCHOR_DELTA_MIN (0.01). The effect to be
     attributed must exist IN THIS RUN. Recorded value between the 861g and 861e
     cells is 0.1115, ~11x the floor. Unmet -> uninformative, not a verdict.
  C1 (load-bearing): D(PIN_A, ANCHOR_PRE)  <= EQUIV_REL_TOL (1e-4)
  C2 (load-bearing): D(PIN_B, ANCHOR_POST) <= EQUIV_REL_TOL (1e-4)
  combination: PASS iff P1 AND C1 AND C2.
  Qualifiers (recorded, not load-bearing):
  C3: D(PIN_A, PIN_B) -- how far apart the boundary cells are.
  C4: anchors reproduce the RECORDED cells: D(ANCHOR_PRE, 861g n10 cell) and
      D(ANCHOR_POST, 861e cell) both <= EQUIV_REL_TOL.
  C5: grading split: factor(PIN_A) > 1.0 and factor(PIN_B) < 1.0. Recorded only;
      861g's pre-range factor was 1.0070, too thin a margin to gate on.

WHY 1e-4. Within one box and one torch build the prediction is BITWISE equality
(the bisect found every other commit bitwise inert on the wake path, and 861h
reproduced 861e's decisive mean_mel to all 17 digits on a different cloud box).
The only cross-machine-class difference ever measured on this cell is Mac vs
linux at 1.3e-7 relative (861f unreseeded). 1e-4 sits far above both and ~1000x
below the 0.1115 effect, so the choice of tolerance cannot decide a verdict.
Bitwise equality is recorded separately (repr of the float).

================================================================================
VERDICT GRID (DECLARED NULLS)
================================================================================
  P1 unmet                     -> uninformative_anchor_delta_absent_in_run (FAIL).
                                  The live harness no longer produces the
                                  pre/post difference at all; nothing to attribute.
  C1 and C2                    -> mover_6293b23_confirmed_full_protocol (PASS),
                                  suffixed _reproduces_recorded or
                                  _recorded_not_reproduced by C4.
  C3 <= tol (A == B)           -> not_confirmed_6293b23_inert_under_full_protocol
                                  (FAIL). The wake-path localisation does NOT carry:
                                  the collapse comes from training/calibration/
                                  sleep code in another commit.
  C1 only (B != POST)          -> boundary_confirmed_later_commits_<kind> (FAIL).
                                  6293b23 moves the readout, but one or more of
                                  68173a7..0911574 ALSO perturbs the full-protocol
                                  path. <kind> = sub_effect_perturbation when
                                  D(B, POST) < ANCHOR_DELTA_MIN (a perturbation far
                                  below the effect -- NOT a share of the 0.884), else
                                  share_effect (0.884 is not 6293b23's alone).
                                  Read distances.b_post against distances.pre_post.
  C2 only (A != PRE)           -> boundary_confirmed_earlier_commits_<kind> (FAIL).
                                  Same, for 6f46a70/bbc69c4/93d5d98, on D(A, PRE).
  neither, A != B              -> multiple_commits_perturb_full_protocol (FAIL).

WHAT A PASS MEANS -- two tiers, by C4 (red-team F1):
  PASS with C4 met (label suffix _reproduces_recorded): the anchors reproduce the
  recorded 861g/861e cells, so the RECORDED seed-271 movement between 861g and
  861e is carried entirely by 6293b23.
  PASS with C4 unmet (suffix _recorded_not_reproduced): 6293b23 carries the whole
  IN-RUN anchor difference, but this run's anchors are a different realisation
  from the recorded ones (first suspect: machine_class / torch / live harness).
  That supports attribution of THIS run's gap only, NOT of the recorded 0.884.
  manifest readout.outcome_pass_qualified_by_c4 carries the distinction as a
  scalar, and interpretation.reading states it in words.
In the C4-met tier the movement is carried
entirely by 6293b23 -- consistent with the author-documented mechanism (a
legitimate E3 tick-cadence change that shifts the fixed-seed rollout RNG draw
sequence). It suggests, and does NOT establish, that the collapse is a
seed-realisation shift rather than a regression of the MEL mechanism.

WHAT A PASS DOES NOT MEAN: nothing about INV-050 or MECH-180 as claims. It says
which code change moved one seed's number; it is silent on whether MEL couples
to sleep. Neither claim's status, confidence or v3_pending may move on this run.

WHAT A NULL DOES NOT MEAN: a FAIL is not evidence against INV-050/MECH-180 and
not a substrate ceiling; it re-opens the localisation question inside the
full-protocol code paths.

================================================================================
EXPERIMENT_PURPOSE = "diagnostic"; directions pinned
================================================================================
Instrument/provenance diagnostic for the INV-050 / MECH-180 lineage it
adjudicates. evidence_direction = non_contributory and
evidence_direction_per_claim = {"INV-050": "unknown", "MECH-180": "unknown"}
REGARDLESS of outcome, as for 861f/861g/861h.

================================================================================
QUEUE-EXPERIMENT GATE DISPOSITIONS (recorded at authoring, 2026-09-14)
================================================================================
Step 2.4 GOV-REUSE-1: decisive readout = seed-271 ARM_3_HIGH_ON mean_mel and
mel_reference at pins 5f64a53f and 6293b23 under the full 861e protocol. No
recorded manifest ran either commit (861c/861g: f810969; 861e/861h/861f:
17befb8c; the bisect: wake-only). Not recoverable -> run. The two anchor cells
ARE recorded (861g, 861e) but are re-run deliberately as in-run controls for
live-harness drift (see WHY FOUR).

Step 2.5b re-derive brake: the literal run-keyed counter reads INV-050 = 7,
MECH-180 = 5. Released under the skill's own "Not braked" clause: this is a
diagnostic that discriminates WHY a reading moved (which commit), not a
lettered re-derive of either claim at the same granularity -- the same release
861f/861g/861h carried, and it tests no claim hypothesis at all.

Step 2.5c substrate-path overlap: the only CORRUPTING entry this driver's code
path overlaps is contextmemory-write-path-addressing-degeneracy
(ree_core/predictors/e1_deep.py::ContextMemory.write). Disposition: not
blocking. (a) Every cell executes HISTORICAL pinned code; the object of study is
the number that code produced, defect included, so the gate's premise (a known
defect would corrupt a claim reading) does not apply to a provenance
attribution. (b) V3-EXQ-861h measured this exact defect on this exact cell as
NOT load-bearing for measured MEL: the refractory write path reproduced 861e's
mean_mel to within 5e-9 relative. (c) The repair (76cbf84/692f852) is
default-off and this driver passes no ContextMemory write knob, so the path is
identical across all four pins. Other open entries (MECH-320 tonic_vigor,
sd_blocked_agency, SD-MECH303, etc.) are default-off or absent in these refs.

Step 2.6 ethics: all-false / allow (V3, SENT-0).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from collections import deque
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# NOTE: nothing imported at module level may import ree_core. The child process
# pins ree_core BEFORE its first import (substrate_pin refuses otherwise); the
# parent never imports ree_core at all. Verified at authoring: these modules
# leave sys.modules free of ree_core.
from experiments._lib.substrate_pin import (            # noqa: E402
    SubstratePinError, pin_ree_core, verify_pin, pin_fingerprint_kwargs,
    pin_manifest_block,
)
from experiments._lib.arm_fingerprint import arm_cell   # noqa: E402  (lazy torch/_harness)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiment_protocol import emit_outcome            # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_TYPE = "v3_exq_861i_inv050_mech091_commit_attribution_pinned_confirmation"
QUEUE_ID = "V3-EXQ-861i"
CLAIM_IDS = ["INV-050", "MECH-180"]
EXPERIMENT_PURPOSE = "diagnostic"
SUPERSEDES = None
COMPARES_AGAINST_RUN_ID = (
    "v3_exq_861e_inv050_mech180_calibration_power_raised_replication"
    "_20260820T214522Z_v3"
)
SOURCE_BISECT_DOC = (
    "REE_assembly/evidence/planning/inv050_reecore_bisect_20260911.md "
    "(origin/master 8e1216d2c3)"
)
SOURCE_CHIP_REF = "chip-20260911-inv050-pinned-861e-confirmation-cell"

RED_TEAM_VERDICT = (
    "red-team (fable): CONTESTED -> 2 findings, both FIXED, 0 dismissed. "
    "F1 PASS with C4 unmet read as attribution of the RECORDED 0.884 -> PASS "
    "meaning two-tiered by C4 (interpretation.reading, "
    "readout.outcome_pass_qualified_by_c4). F2 the also_perturb FAIL labels "
    "did not separate a sub-effect perturbation from a share of the effect -> "
    "labels split sub_effect_perturbation vs share_effect at ANCHOR_DELTA_MIN. "
    "Causal chain unchanged, so no re-spawn."
)

DECISIVE_SEED = 271
DECISIVE_ARM_ID = "ARM_3_HIGH_ON"

REF_F810969 = "f810969089fa8193959f49072e8aa1c2de0cb193"
REF_PIN_A = "5f64a53f14bb67c54d4c3239807fadb490f3d36c"
REF_PIN_B = "6293b2395248524f243364b49ea0ce52298f4200"
REF_17BEFB8C = "17befb8c46f0b7352f74a6b6e3ee4fc9715878fc"

# Marker chain (see docstring). Values computed with git show at authoring time.
PIN_CELLS: List[Dict[str, Any]] = [
    {"cell_id": "ANCHOR_PRE_f810969", "role": "anchor_pre", "ref": REF_F810969,
     "attr_markers": [
         ("ree_core.preservation.archive", "MultiArchive", False),
         ("ree_core.predictors.e3_selector", "authority_spread_ratio", False)],
     "agent_phase_reset_lines": 1},
    {"cell_id": "PIN_A_5f64a53f", "role": "pin_a", "ref": REF_PIN_A,
     "attr_markers": [
         ("ree_core.preservation.archive", "MultiArchive", True),
         ("ree_core.predictors.e3_selector", "authority_spread_ratio", False)],
     "agent_phase_reset_lines": 1},
    {"cell_id": "PIN_B_6293b23", "role": "pin_b", "ref": REF_PIN_B,
     "attr_markers": [
         ("ree_core.preservation.archive", "MultiArchive", True),
         ("ree_core.predictors.e3_selector", "authority_spread_ratio", False)],
     "agent_phase_reset_lines": 6},
    {"cell_id": "ANCHOR_POST_17befb8c", "role": "anchor_post", "ref": REF_17BEFB8C,
     "attr_markers": [
         ("ree_core.preservation.archive", "MultiArchive", True),
         ("ree_core.predictors.e3_selector", "authority_spread_ratio", True)],
     "agent_phase_reset_lines": 6},
]
PIN_BY_ID = {c["cell_id"]: c for c in PIN_CELLS}
CELL_BY_ROLE = {c["role"]: c["cell_id"] for c in PIN_CELLS}

# Recorded decisive cells (seed 271, ARM_3_HIGH_ON, CALIB_DRAWS=10, argmin, no
# reseed), copied verbatim from the manifests at authoring time.
RECORDED_ANCHORS = {
    "anchor_pre": {
        "source": "V3-EXQ-861g variant n10",
        "run_id": ("v3_exq_861g_inv050_mech180_h3_substrate_pin_f810969"
                   "_20260822T175951Z_v3"),
        "machine": "ree-cloud-2",
        "machine_class": "linux-x86_64-py3.10-torch2.12.0+cpu",
        "mean_mel": 2.586514469180167e-05,
        "mel_reference": 2.5685158609726182e-05,
        "mean_duration_factor": 1.0070073961703054,
    },
    "anchor_post": {
        "source": "V3-EXQ-861e (bitwise-reproduced by V3-EXQ-861h argmin_legacy on ree-cloud-4)",
        "run_id": COMPARES_AGAINST_RUN_ID,
        "machine": "ree-worker-1",
        "machine_class": "linux-x86_64-py3.10-torch2.12.0+cpu",
        "mean_mel": 2.2980323189226863e-05,
        "mel_reference": 2.5982329782371255e-05,
        "mean_duration_factor": 0.8844596840125855,
    },
}

EQUIV_REL_TOL = 1e-4
ANCHOR_DELTA_MIN = 0.01
FACTOR_GRADED_FLOOR = 1.0
CRITERION_ACHIEVABLE_RANGE_EXEMPT = (
    "P1 is itself the realised-range (headroom) check: it measures the in-run "
    "anchor cells' distance D against the 0.01 floor and routes the run "
    "uninformative when the range is absent. C1/C2 are equivalence bounds "
    "(D <= 1e-4) that bitwise equality reaches by construction."
)


def _reldiff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    a = float(a)
    b = float(b)
    if not (math.isfinite(a) and math.isfinite(b)):
        return None
    den = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / den


def _D(x: Optional[Dict[str, Any]], y: Optional[Dict[str, Any]]) -> Optional[float]:
    """Component-wise max relative difference over (mean_mel, mel_reference)."""
    if not x or not y:
        return None
    parts = [_reldiff(x.get("mean_mel"), y.get("mean_mel")),
             _reldiff(x.get("mel_reference"), y.get("mel_reference"))]
    if any(p is None for p in parts):
        return None
    return float(max(parts))


# -- child-side substrate import (ONLY after the pin is proven) ---------------
def _import_pinned_substrate() -> None:
    import numpy as _np
    import torch as _torch
    import torch.nn.functional as _F
    from ree_core.agent import REEAgent as _REEAgent
    from ree_core.environment.causal_grid_world import CausalGridWorldV2 as _Env
    from ree_core.utils.config import REEConfig as _REEConfig
    globals().update(np=_np, torch=_torch, F=_F, REEAgent=_REEAgent,
                     CausalGridWorldV2=_Env, REEConfig=_REEConfig)


def _git_blob_sha1(data: bytes) -> str:
    h = hashlib.sha1()
    h.update(b"blob " + str(len(data)).encode("ascii") + b"\0")
    h.update(data)
    return h.hexdigest()


def _verify_pinned_tree(pin: Dict[str, Any]) -> Dict[str, Any]:
    """Every ree_core/**/*.py in the pin dir must equal git's blob at the ref,
    both directions. Fatal on any mismatch."""
    sha = pin["resolved_sha"]
    r = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-tree", "-r", sha, "ree_core"],
                       capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise SubstratePinError(f"git ls-tree {sha} ree_core failed: {r.stderr.strip()}")
    expected: Dict[str, str] = {}
    for line in r.stdout.splitlines():
        meta, path = line.split("\t", 1)
        parts = meta.split()
        if len(parts) == 3 and parts[1] == "blob" and path.endswith(".py"):
            expected[path] = parts[2]
    pin_dir = Path(pin["pin_dir"])
    actual_paths = sorted(str(p.relative_to(pin_dir)).replace(os.sep, "/")
                          for p in (pin_dir / "ree_core").rglob("*.py"))
    missing = sorted(set(expected) - set(actual_paths))
    extra = sorted(set(actual_paths) - set(expected))
    altered = sorted(p for p in expected if p in set(actual_paths)
                     and _git_blob_sha1((pin_dir / p).read_bytes()) != expected[p])
    if missing or extra or altered:
        raise SubstratePinError(
            f"CONTENT pin check FAILED for {sha[:10]}: missing={missing[:5]} "
            f"extra={extra[:5]} altered={altered[:5]}")
    digest = hashlib.sha256("\n".join(f"{p} {expected[p]}" for p in sorted(expected))
                            .encode("ascii")).hexdigest()
    return {"n_py_files": len(expected), "tree_blob_digest_sha256": digest,
            "missing": 0, "extra": 0, "altered": 0}


def _verify_phase_reset_marker(pin: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    text = (Path(pin["pin_dir"]) / "ree_core" / "agent.py").read_text(encoding="utf-8")
    n = sum(1 for line in text.splitlines() if "phase_reset()" in line)
    if n != int(spec["agent_phase_reset_lines"]):
        raise SubstratePinError(
            f"SOURCE marker FAILED for {spec['cell_id']}: agent.py phase_reset() "
            f"lines={n}, expected {spec['agent_phase_reset_lines']}")
    return {"agent_py_phase_reset_lines": n, "expected": spec["agent_phase_reset_lines"]}


# ============================================================================
# BEGIN verbatim slice from V3-EXQ-861g (constants) -- do not edit by hand
# ============================================================================
TOUCHED_SLOT_L2_EPS = 1e-6

# MECH-122 content-packaging: OFF in this run (861c/861b's value). See module
# docstring "SUBSTRATE-PATH OVERLAP GATE" for the empirical confirmation that
# this keeps the corrupting-flagged relative_novelty() consumption dead code.
USE_MECH122_SPINDLE_CONTENT_SELECTION = False
MECH122_SPINDLE_SELECTION_GAIN = 1.0   # inert while the flag is False

# z_goal_enabled=True inherited verbatim for architecture parity; see
# DEAD_Z_GOAL_STREAM_EXEMPT below (unchanged reasoning from 861c).
DEAD_Z_GOAL_STREAM_EXEMPT = (
    "inherited verbatim from V3-EXQ-718a/798a/845/861/861b/861c for "
    "architecture parity; wiring update_z_goal would activate the E3 goal "
    "term, E1 conditioning, and the SD-024 benefit-attractor producer, "
    "breaking the single-variable comparison against V3-EXQ-861c that this "
    "replication depends on. Knob is arm-symmetric (identical in every arm)."
)

# -- Design parameters (IDENTICAL to V3-EXQ-861c except the CALIBRATION block) -
CONV_EPISODES = 60
STEPS_PER_EPISODE = 90
PROBE_BATTERY_SIZE = 64

# -- CALIBRATION POWER RAISE (the substantive change from V3-EXQ-861c) -------
# 861c used CALIB_DRAWS=5 (rel_sd 0.152-0.283), which left seed 271 failing C2
# by 0.5% purely on calibration sample size (confirmed
# failure_autopsy_861c-861d-mech180-cluster_2026-08-16 section 2c). Raising to
# 10 is the exact n that autopsy's own projection shows flips seed 271's
# margin to PASS, using SEM ~ rel_sd/sqrt(n_draws) (confirmed to hold in
# 861c's own manifest). CALIB_EPISODES_PER_DRAW unchanged at 6 (episodes per
# draw is not the lever the autopsy identified; more draws, not longer draws).
CALIB_DRAWS = 10               # independent repeated calibration draws (861c: 5)
CALIB_EPISODES_PER_DRAW = 6    # stable-base wake episodes per draw (unchanged)
CALIB_EPISODES = CALIB_DRAWS * CALIB_EPISODES_PER_DRAW   # total: 60 (861c: 30)
K_CALIB_MARGIN = 2.0           # unchanged from 861c

# -- NEW: calibration-precision readiness precondition (R3) -------------------
# See module docstring "THE FIX" item (2) for the full justification. At
# CALIB_DRAWS=10, the confirmed 861c rel_sd range (0.152-0.283) projects
# calib_rel_sd_of_mean in [0.048, 0.090]; 0.15 sits ~67% above the worst
# projected value. Pre-registered, NOT derived from this run's own stats.
MAX_CALIB_REL_SD_OF_MEAN = 0.15

MEAS_CYCLES = 6
WAKE_EPISODES_PER_CYCLE = 2
EPISODES_PER_RUN = (CONV_EPISODES + CALIB_EPISODES
                    + MEAS_CYCLES * WAKE_EPISODES_PER_CYCLE)

SWS_CONSOLIDATION_STEPS = 5
REM_ATTRIBUTION_STEPS = 10

MEL_GAIN = 1.0
FACTOR_MIN = 0.5
FACTOR_MAX = 3.0
MEL_RELATIVE_FLOOR = 1e-6

SD056_WEIGHT = 0.05
E2_LR = 1e-3
CONTRASTIVE_BATCH_K = 8
MIN_BUFFER_BEFORE_TRAIN = 16
MIN_CLASSES_FOR_TRAIN = 2
MAX_GRAD_NORM = 1.0
TRANSITION_BUFFER_MAX = 256

# -- Thresholds (pre-registered constants, NOT derived from run stats). ------
# Byte-identical to V3-EXQ-845/861/861a/861b/861c EXCEPT MAX_CALIB_REL_SD_OF_MEAN
# (new). Deliberately NOT re-tuned otherwise: a replication that moved its
# thresholds would not be a replication.
MIN_REL_CONV_DROP = 0.10
SEED_PASS_FRAC = 2.0 / 3.0
MIN_MEL_SPREAD = 0.15
MIN_REL_DV_SPREAD = 0.15
MONO_TOL = 0.05
PINNED_ABS_VAR_ATOL = 1e-6

ENV_BASE: Dict[str, Any] = dict(
    size=12,
    num_hazards=4,
    num_resources=5,
    hazard_harm=0.05,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=10,
    use_proxy_fields=True,
)

STABLE_DRIFT = dict(env_drift_interval=999, env_drift_prob=0.0)
WORLD_RULE_SHIFT_DEPTH = 2

ARMS: List[Dict[str, Any]] = [
    {"arm_id": "ARM_0_NONE_ON",  "level": 0, "interval": 0,  "mel_on": True},
    {"arm_id": "ARM_1_LOW_ON",   "level": 1, "interval": 60, "mel_on": True},
    {"arm_id": "ARM_2_MED_ON",   "level": 2, "interval": 25, "mel_on": True},
    {"arm_id": "ARM_3_HIGH_ON",  "level": 3, "interval": 10, "mel_on": True},
    {"arm_id": "ARM_4_HIGH_OFF", "level": 3, "interval": 10, "mel_on": False},
]
ON_ECO_ARMS = ["ARM_0_NONE_ON", "ARM_1_LOW_ON", "ARM_2_MED_ON", "ARM_3_HIGH_ON"]


# END verbatim slice (constants)


# BEGIN verbatim slice from V3-EXQ-861g (cell helpers)
def _make_env(seed: int, interval: int) -> CausalGridWorldV2:
    kw = dict(ENV_BASE)
    kw.update(STABLE_DRIFT)
    kw.update(
        world_rule_shift_enabled=(interval > 0),
        world_rule_shift_interval=interval,
        world_rule_shift_depth=WORLD_RULE_SHIFT_DEPTH if interval > 0 else 0,
    )
    return CausalGridWorldV2(seed=seed, **kw)


def _make_agent(env: CausalGridWorldV2, mel_on: bool, mel_reference: float) -> REEAgent:
    """UNCHANGED from 861c -- byte-identical config."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        use_harm_stream=True,
        z_harm_dim=32,
        use_affective_harm_stream=True,
        z_harm_a_dim=16,
        harm_history_len=10,
        z_goal_enabled=True,
        goal_weight=0.5,
        drive_weight=2.0,
        e1_goal_conditioned=True,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        benefit_eval_enabled=True,
        benefit_weight=1.0,
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=SD056_WEIGHT,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=2.0,
        surprise_gated_replay=True,
        use_sleep_loop=True,
        sleep_loop_episodes_K=10**9,
        sws_enabled=True,
        sws_consolidation_steps=SWS_CONSOLIDATION_STEPS,
        use_mech122_spindle_content_selection=USE_MECH122_SPINDLE_CONTENT_SELECTION,
        mech122_spindle_selection_gain=MECH122_SPINDLE_SELECTION_GAIN,
        rem_enabled=True,
        rem_attribution_steps=REM_ATTRIBUTION_STEPS,
        use_mel_consumer=bool(mel_on),
        mel_gain=MEL_GAIN,
        mel_reference=float(mel_reference),
        mel_reference_mode="fixed",
        mel_duration_factor_min=FACTOR_MIN,
        mel_duration_factor_max=FACTOR_MAX,
        mel_relative_floor=MEL_RELATIVE_FLOOR,
        mel_scale_sws=True,
        mel_scale_rem=True,
        use_mel_entry=False,
    )
    return REEAgent(cfg)


def _obs(d: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    h = d.get(key)
    if h is None:
        return None
    return h.float().unsqueeze(0) if h.dim() == 1 else h.float()


def _sense_latent(agent: REEAgent, obs_dict: Dict[str, Any]):
    body = obs_dict["body_state"].float()
    world = obs_dict["world_state"].float()
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    return agent.sense(
        obs_body=body, obs_world=world,
        obs_harm=_obs(obs_dict, "harm_obs"),
        obs_harm_a=_obs(obs_dict, "harm_obs_a"),
        obs_harm_history=_obs(obs_dict, "harm_history"),
    )


def _sample_class_diverse_batch(
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    k: int, rng: random.Random,
) -> Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    if len(buffer) < MIN_BUFFER_BEFORE_TRAIN:
        return None
    pool = list(buffer)
    rng.shuffle(pool)
    seen: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for tup in pool:
        cls = int(tup[1].argmax().item())
        if cls not in seen:
            seen[cls] = tup
        if len(seen) >= k:
            break
    if len(seen) < MIN_CLASSES_FOR_TRAIN:
        return None
    samples = list(seen.values())
    picked = {id(s) for s in samples}
    for tup in pool:
        if len(samples) >= k:
            break
        if id(tup) in picked:
            continue
        samples.append(tup)
        picked.add(id(tup))
    return samples


def _e2_train_step(
    agent: REEAgent,
    buffer: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    optimiser: torch.optim.Optimizer, rng: random.Random,
) -> Optional[float]:
    batch = _sample_class_diverse_batch(buffer, CONTRASTIVE_BATCH_K, rng)
    if batch is None:
        return None
    z0_K = torch.stack([t[0] for t in batch]).to(agent.device)
    actions_K = torch.stack([t[1] for t in batch]).to(agent.device)
    z1_K = torch.stack([t[2] for t in batch]).to(agent.device)
    optimiser.zero_grad(set_to_none=True)
    z1_pred = agent.e2.world_forward(z0_K, actions_K)
    recon = F.mse_loss(z1_pred, z1_K)
    recon_val = float(recon.detach().item())
    if not math.isfinite(recon_val):
        return recon_val
    recon.backward()
    torch.nn.utils.clip_grad_norm_(agent.e2.parameters(), max_norm=MAX_GRAD_NORM)
    optimiser.step()
    return recon_val


def _waking_step(
    agent: REEAgent, env: CausalGridWorldV2, obs_dict: Dict[str, Any],
    train: bool, buffer: Optional[Deque],
    e2_opt: Optional[torch.optim.Optimizer], sample_rng: Optional[random.Random],
    pending_capture_ref: List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
) -> Tuple[Dict[str, Any], bool]:
    latent = _sense_latent(agent, obs_dict)

    if train and buffer is not None:
        pend = pending_capture_ref[0]
        if pend is not None:
            z0_prev, a_prev = pend
            z1_obs = latent.z_world.detach().reshape(-1).clone()
            if (torch.isfinite(z0_prev).all() and torch.isfinite(a_prev).all()
                    and torch.isfinite(z1_obs).all()):
                buffer.append((z0_prev, a_prev, z1_obs))
            pending_capture_ref[0] = None

    ticks = agent.clock.advance()
    wdim = latent.z_world.shape[-1]
    e1_prior = (agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device))
    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
    action = agent.select_action(candidates, ticks)

    if action is None:
        idx = int(np.random.randint(0, env.action_dim))
        action = torch.zeros(1, env.action_dim, device=agent.device)
        action[0, idx] = 1.0
        agent._last_action = action
    if not torch.isfinite(action).all():
        return obs_dict, True

    if train and buffer is not None and torch.isfinite(latent.z_world).all():
        pending_capture_ref[0] = (
            latent.z_world.detach().reshape(-1).clone(),
            action.detach().reshape(-1).clone(),
        )
        if e2_opt is not None and sample_rng is not None:
            _e2_train_step(agent, buffer, e2_opt, sample_rng)

    _, harm_signal, done, info, next_obs_dict = env.step(action)
    with torch.no_grad():
        agent.update_residue(
            harm_signal=float(harm_signal), world_delta=None,
            hypothesis_tag=False, owned=True,
        )
    return next_obs_dict, bool(done)


def _run_wake_window(
    agent: REEAgent, env: CausalGridWorldV2, n_episodes: int, steps: int,
    train: bool, buffer: Optional[Deque],
    e2_opt: Optional[torch.optim.Optimizer], sample_rng: Optional[random.Random],
    ep_offset: int, arm_id: str, seed: int,
) -> None:
    pending_capture_ref: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None]
    for ep in range(n_episodes):
        glob_ep = ep_offset + ep
        if (glob_ep % 10 == 0) or (glob_ep == EPISODES_PER_RUN - 1):
            print(f"  [train] {arm_id} seed={seed} ep {glob_ep+1}/{EPISODES_PER_RUN}",
                  flush=True)
        _, obs_dict = env.reset()
        agent.reset()
        agent.e1.reset_hidden_state()
        pending_capture_ref[0] = None
        for _step in range(steps):
            obs_dict, done = _waking_step(
                agent, env, obs_dict, train, buffer, e2_opt, sample_rng,
                pending_capture_ref,
            )
            if done:
                break


def _sample_probe_battery(
    agent: REEAgent, seed: int, n_transitions: int, steps: int,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    env = _make_env(seed, interval=0)
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    act_rng = random.Random(seed + 9973)
    battery: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    prev: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    guard = 0
    max_guard = max(steps, 1) * 8
    while len(battery) < n_transitions and guard < max_guard:
        guard += 1
        latent = _sense_latent(agent, obs_dict)
        if not torch.isfinite(latent.z_world).all():
            break
        z_now = latent.z_world.detach().reshape(1, -1).clone()
        if prev is not None:
            z0, a = prev
            battery.append((z0, a, z_now))
        idx = act_rng.randrange(env.action_dim)
        action = torch.zeros(1, env.action_dim, device=agent.device)
        action[0, idx] = 1.0
        _, _, done, _, obs_dict = env.step(action)
        prev = (z_now, action)
        if done:
            _, obs_dict = env.reset()
            agent.reset()
            agent.e1.reset_hidden_state()
            prev = None
    return battery


def _frozen_probe_pe(
    agent: REEAgent, battery: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> float:
    if not battery:
        return 0.0
    errs: List[float] = []
    with torch.no_grad():
        for z0, a, z1 in battery:
            pred = agent.e2.world_forward(z0.to(agent.device), a.to(agent.device))
            err = float((pred - z1.to(agent.device)).pow(2).mean().item())
            if math.isfinite(err):
                errs.append(err)
    return float(np.mean(errs)) if errs else 0.0


def _touched_slot_diversity(
    mem_before: torch.Tensor, mem_after: torch.Tensor, eps: float = TOUCHED_SLOT_L2_EPS,
) -> Tuple[float, int, bool]:
    """Byte-identical to 861/861a/861c."""
    diffs = (mem_after - mem_before).norm(dim=-1)
    touched_mask = diffs > eps
    n_touched = int(touched_mask.sum().item())
    if n_touched < 2:
        return 0.0, n_touched, True
    touched = mem_after[touched_mask]
    norms = touched.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    normed = touched / norms
    sim_mat = torch.mm(normed, normed.t())
    mask = torch.eye(n_touched, device=sim_mat.device, dtype=torch.bool)
    off_diag = sim_mat[~mask]
    diversity = float((1.0 - off_diag).mean().item())
    return diversity, n_touched, False


# END verbatim slice (cell helpers)


_ZG = ZGoalStreamAccumulator()
_LAST_AGENT: List[Optional[Any]] = [None]


# BEGIN verbatim slice from V3-EXQ-861g (_run_cell; only the `variant`
# default and the boundary print label differ)
def _run_cell(seed: int, arm: Dict[str, Any], steps: int, conv_eps: int,
              meas_cycles: int, calib_draws: int,
              calib_eps_per_draw: int, *,
              variant: str = "",
              reseed_before_measurement: bool = False,
              write_selection: str = "argmin") -> Dict[str, Any]:
    """One (seed, arm, variant) cell.

    Cell logic is byte-identical to V3-EXQ-861e except for the ONE knob this
    leg varies, which is carried by the keyword-only arguments above and
    recorded on the returned row as `variant`. Every cell still resets all RNG
    at entry, so cells are independent of each other and of their order --
    which is what makes the in-run control set a valid same-machine,
    same-substrate replica rather than a sequence effect."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    arm_id = arm["arm_id"]
    mel_on = bool(arm["mel_on"])
    print(f"Seed {seed} Condition {arm_id}@{variant}", flush=True)

    stable_env = _make_env(seed, interval=0)
    agent = _make_agent(stable_env, mel_on=mel_on, mel_reference=0.0)
    battery = _sample_probe_battery(agent, seed, PROBE_BATTERY_SIZE, steps)
    probe_pe_init = _frozen_probe_pe(agent, battery)

    buffer: Deque = deque(maxlen=TRANSITION_BUFFER_MAX)
    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_LR)
    sample_rng = random.Random(seed + 4242)

    _run_wake_window(
        agent, stable_env, conv_eps, steps, train=True, buffer=buffer,
        e2_opt=e2_opt, sample_rng=sample_rng, ep_offset=0, arm_id=arm_id, seed=seed,
    )

    probe_pe_final = _frozen_probe_pe(agent, battery)
    conv_rel_drop = (((probe_pe_init - probe_pe_final) / probe_pe_init)
                     if probe_pe_init > 1e-12 else 0.0)

    # -- Reference calibration (V3-EXQ-861c methodology, CALIB_DRAWS raised
    # to 10 in this run -- see module docstring "THE FIX" item (1)) --------
    calib_draws_mel: List[float] = []
    calib_ep_cursor = conv_eps
    if agent.mel_consumer is not None:
        for _draw in range(calib_draws):
            agent.mel_consumer.reset()
            _run_wake_window(
                agent, stable_env, calib_eps_per_draw, steps, train=False,
                buffer=None, e2_opt=None, sample_rng=None,
                ep_offset=calib_ep_cursor, arm_id=arm_id, seed=seed,
            )
            calib_ep_cursor += calib_eps_per_draw
            draw_mel = float(agent.mel_consumer.current_mel())
            if math.isfinite(draw_mel) and draw_mel > 0.0:
                calib_draws_mel.append(draw_mel)
    else:
        _run_wake_window(
            agent, stable_env, calib_draws * calib_eps_per_draw, steps,
            train=False, buffer=None, e2_opt=None, sample_rng=None,
            ep_offset=calib_ep_cursor, arm_id=arm_id, seed=seed,
        )
        calib_ep_cursor += calib_draws * calib_eps_per_draw

    n_calib_valid = len(calib_draws_mel)
    if mel_on and agent.mel_consumer is not None:
        if calib_draws_mel:
            base_ref = float(np.mean(calib_draws_mel))
            calib_sd = float(np.std(calib_draws_mel)) if n_calib_valid > 1 else 0.0
        else:
            base_ref = float(probe_pe_final)
            calib_sd = 0.0
        if not (base_ref > 0.0):
            base_ref = float(probe_pe_final)
            calib_sd = 0.0
        calib_rel_sd = (calib_sd / base_ref) if base_ref > 0.0 else 0.0
        calib_rel_sd_of_mean = calib_rel_sd / math.sqrt(max(1, n_calib_valid))
        agent.config.mel_reference = base_ref
        agent.mel_consumer.config.mel_reference = base_ref
        agent.mel_consumer.reset()
    else:
        base_ref = float(probe_pe_final)
        calib_sd = 0.0
        calib_rel_sd = 0.0
        calib_rel_sd_of_mean = 0.0

    meas_env = _make_env(seed, arm["interval"])
    cum_sws = 0.0
    cum_rem = 0.0
    per_cycle_sws: List[float] = []
    per_cycle_rem: List[float] = []
    per_cycle_diversity_legacy: List[float] = []
    per_cycle_new_diversity: List[float] = []
    per_cycle_n_touched: List[int] = []
    n_cycles_insufficient_touched = 0
    factors: List[float] = []
    mels: List[float] = []
    per_cycle_spindle_selection_applied: List[float] = []
    per_cycle_spindle_selection_mean_weight: List[float] = []
    ep_off = calib_ep_cursor
    for _cyc in range(meas_cycles):
        _run_wake_window(
            agent, meas_env, WAKE_EPISODES_PER_CYCLE, steps, train=False,
            buffer=None, e2_opt=None, sample_rng=None, ep_offset=ep_off,
            arm_id=arm_id, seed=seed,
        )
        ep_off += WAKE_EPISODES_PER_CYCLE

        mem_before = agent.e1.context_memory.memory.detach().clone()
        m = agent.sleep_loop.force_cycle(agent)
        mem_after = agent.e1.context_memory.memory.detach().clone()
        new_div, n_touched, insufficient = _touched_slot_diversity(mem_before, mem_after)
        if insufficient:
            n_cycles_insufficient_touched += 1

        sws = float(m.get("sws_n_writes", 0.0))
        rem = float(m.get("rem_n_rollouts", 0.0))
        diversity_legacy = float(m.get("sws_slot_diversity", 0.0))
        cum_sws += sws
        cum_rem += rem
        per_cycle_sws.append(sws)
        per_cycle_rem.append(rem)
        per_cycle_diversity_legacy.append(diversity_legacy)
        per_cycle_new_diversity.append(new_div)
        per_cycle_n_touched.append(n_touched)
        per_cycle_spindle_selection_applied.append(
            float(m.get("sws_spindle_selection_applied", 0.0))
        )
        per_cycle_spindle_selection_mean_weight.append(
            float(m.get("sws_spindle_selection_mean_weight", 0.0))
        )
        if mel_on:
            factors.append(float(m.get("mel_duration_factor", 1.0)))
            mels.append(float(m.get("mel_mean", 0.0)))

    valid_new_div = [v for v, n in zip(per_cycle_new_diversity, per_cycle_n_touched)
                     if n >= 2]
    mean_new_diversity = float(np.mean(valid_new_div)) if valid_new_div else 0.0
    mean_diversity_legacy = (float(np.mean(per_cycle_diversity_legacy))
                             if per_cycle_diversity_legacy else 0.0)
    sws_count_var = float(np.var(per_cycle_sws))
    rem_count_var = float(np.var(per_cycle_rem))
    new_diversity_var = float(np.var(valid_new_div)) if len(valid_new_div) > 1 else 0.0
    diversity_var_legacy = float(np.var(per_cycle_diversity_legacy))
    mean_factor = float(np.mean(factors)) if factors else 1.0
    mean_mel = float(np.mean(mels)) if mels else 0.0
    mean_spindle_selection_applied = float(np.mean(per_cycle_spindle_selection_applied))
    mean_spindle_selection_weight = float(np.mean(per_cycle_spindle_selection_mean_weight))

    _ZG.observe(agent)
    _LAST_AGENT[0] = agent

    print(f"    {arm_id} seed={seed}: conv_drop={conv_rel_drop:.3f} "
          f"ref={base_ref:.3e} (calib n={n_calib_valid} rel_sd={calib_rel_sd:.3f} "
          f"rel_sd_of_mean={calib_rel_sd_of_mean:.3f}) "
          f"mel={mean_mel:.3e} factor={mean_factor:.3f} "
          f"cum_sws={cum_sws:.0f} cum_rem={cum_rem:.0f} "
          f"[descoped dv3: mean_new_div={mean_new_diversity:.4f} "
          f"legacy={mean_diversity_legacy:.4f} "
          f"n_touched_mean={np.mean(per_cycle_n_touched):.1f} "
          f"insufficient_cycles={n_cycles_insufficient_touched}/{meas_cycles}] "
          f"spindle_sel_applied={mean_spindle_selection_applied:.2f} "
          f"spindle_sel_weight={mean_spindle_selection_weight:.4f}",
          flush=True)
    print(f"verdict: {'PASS' if conv_rel_drop >= MIN_REL_CONV_DROP else 'FAIL'}",
          flush=True)

    return {
        "arm_id": arm_id,
        "variant": variant,
        "reseed_before_measurement": bool(reseed_before_measurement),
        "contextmemory_write_selection": str(write_selection),
        "calib_draws_this_cell": int(calib_draws),
        "level": arm["level"],
        "mel_on": mel_on,
        "world_rule_shift_interval": arm["interval"],
        "seed": seed,
        "conv_rel_drop": conv_rel_drop,
        "probe_pe_init": probe_pe_init,
        "probe_pe_final": probe_pe_final,
        "mel_reference": base_ref,
        "mel_reference_calib_draws": list(calib_draws_mel),
        "mel_reference_calib_n_valid": n_calib_valid,
        "mel_reference_calib_sd": calib_sd,
        "mel_reference_calib_rel_sd": calib_rel_sd,
        "mel_reference_calib_rel_sd_of_mean": calib_rel_sd_of_mean,
        "mean_mel": mean_mel,
        "mean_duration_factor": mean_factor,
        "cumulative_sws_writes": cum_sws,
        "cumulative_rem_rollouts": cum_rem,
        "mean_sws_new_slot_diversity": mean_new_diversity,
        "per_cycle_new_diversity": per_cycle_new_diversity,
        "per_cycle_n_touched_slots": per_cycle_n_touched,
        "n_cycles_insufficient_touched_slots": n_cycles_insufficient_touched,
        "new_diversity_variance": new_diversity_var,
        "mean_sws_slot_diversity_wholebank_legacy": mean_diversity_legacy,
        "per_cycle_diversity_wholebank_legacy": per_cycle_diversity_legacy,
        "diversity_variance_wholebank_legacy": diversity_var_legacy,
        "per_cycle_sws": per_cycle_sws,
        "per_cycle_rem": per_cycle_rem,
        "per_cycle_mel": mels,
        "per_cycle_factor": factors,
        "sws_count_variance": sws_count_var,
        "rem_count_variance": rem_count_var,
        "meas_cycles": meas_cycles,
        "mean_spindle_selection_applied": mean_spindle_selection_applied,
        "mean_spindle_selection_weight": mean_spindle_selection_weight,
        "per_cycle_spindle_selection_applied": per_cycle_spindle_selection_applied,
        "per_cycle_spindle_selection_mean_weight": per_cycle_spindle_selection_mean_weight,
    }


# END verbatim slice (_run_cell)


# ============================================================================
# NEW in V3-EXQ-861i: per-pin child cell + parent aggregation
# ============================================================================
DECISIVE_ARM = next(a for a in ARMS if a["arm_id"] == DECISIVE_ARM_ID)


def _params(dry_run: bool) -> Dict[str, int]:
    if dry_run:
        # Tiny, but still multi-draw calibration (>1 draw so calib_sd is a real
        # std) and >1 measurement cycle so mean_mel is a real mean.
        return dict(steps=12, conv_eps=4, meas_cycles=3, calib_draws=2,
                    calib_eps_per_draw=2)
    return dict(steps=STEPS_PER_EPISODE, conv_eps=CONV_EPISODES,
                meas_cycles=MEAS_CYCLES, calib_draws=CALIB_DRAWS,
                calib_eps_per_draw=CALIB_EPISODES_PER_DRAW)


def _cell_config(spec: Dict[str, Any], p: Dict[str, int]) -> Dict[str, Any]:
    return {
        "env_base": ENV_BASE,
        "arm": DECISIVE_ARM,
        "world_rule_shift_depth": WORLD_RULE_SHIFT_DEPTH,
        "conv_episodes": p["conv_eps"],
        "calib_draws": p["calib_draws"],
        "calib_episodes_per_draw": p["calib_eps_per_draw"],
        "k_calib_margin": K_CALIB_MARGIN,
        "meas_cycles": p["meas_cycles"],
        "steps_per_episode": p["steps"],
        "sws_steps": SWS_CONSOLIDATION_STEPS,
        "rem_steps": REM_ATTRIBUTION_STEPS,
        "mel_gain": MEL_GAIN,
        "factor_min": FACTOR_MIN,
        "factor_max": FACTOR_MAX,
        "mel_relative_floor": MEL_RELATIVE_FLOOR,
        "touched_slot_l2_eps": TOUCHED_SLOT_L2_EPS,
        "use_mech122_spindle_content_selection": USE_MECH122_SPINDLE_CONTENT_SELECTION,
        "mech122_spindle_selection_gain": MECH122_SPINDLE_SELECTION_GAIN,
        "contextmemory_write_selection": "argmin_default_no_knob_passed",
        "reseed_before_measurement": False,
        "pin_cell_id": spec["cell_id"],
        "substrate_pin_ref": spec["ref"],
    }


def _child_main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child-cell", action="store_true", required=True)
    ap.add_argument("--pin-cell-id", required=True, choices=sorted(PIN_BY_ID))
    ap.add_argument("--cell-out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    spec = PIN_BY_ID[a.pin_cell_id]

    # -- PIN first, before ANY ree_core import in this process ----------------
    pin = pin_ree_core(spec["ref"])
    marker_records = []
    for mod, attr, present in spec["attr_markers"]:
        verify_pin(pin, marker_module=mod, marker_attr=attr,
                   marker_expected_present=present)
        marker_records.append(dict(pin["marker"]))
    tree = _verify_pinned_tree(pin)
    src_marker = _verify_phase_reset_marker(pin, spec)
    print(f"substrate_pin: cell={spec['cell_id']} ref={spec['ref'][:10]} "
          f"verified={pin['verified']} files={tree['n_py_files']} "
          f"phase_reset_lines={src_marker['agent_py_phase_reset_lines']}", flush=True)

    _import_pinned_substrate()
    from experiments._lib.manifest_core import enabled_default_off_flags_for_agents

    p = _params(bool(a.dry_run))
    cfg = _cell_config(spec, p)
    with arm_cell(DECISIVE_SEED, config_slice=cfg, script_path=Path(__file__),
                  **pin_fingerprint_kwargs(pin)) as cell:
        row = _run_cell(DECISIVE_SEED, DECISIVE_ARM, p["steps"], p["conv_eps"],
                        p["meas_cycles"], p["calib_draws"], p["calib_eps_per_draw"],
                        variant=spec["cell_id"])
        cell.stamp(row)
    row["pin_cell_id"] = spec["cell_id"]
    row["pin_role"] = spec["role"]
    row["substrate_pin"] = pin_manifest_block(pin)
    row["pin_verification"] = {
        "attr_markers": marker_records,
        "tree_content": tree,
        "source_marker": src_marker,
    }
    row["mean_mel_repr"] = repr(float(row["mean_mel"]))
    row["mel_reference_repr"] = repr(float(row["mel_reference"]))
    payload = {
        "row": row,
        "z_goal_stream_stats": _ZG.stats(),
        "enabled_default_off_flags": enabled_default_off_flags_for_agents(_LAST_AGENT[0]),
    }
    Path(a.cell_out).write_text(json.dumps(payload, default=float))
    return 0


def _run_child(spec: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="ree_861i_cell_") as td:
        out = Path(td) / "cell.json"
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child-cell",
               "--pin-cell-id", spec["cell_id"], "--cell-out", str(out)]
        if dry_run:
            cmd.append("--dry-run")
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            raise RuntimeError(f"child cell {spec['cell_id']} exited {rc}; no manifest "
                               "is written for a run with a failed pinned cell")
        return json.loads(out.read_text())


def _cell_summary(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pin_cell_id": row["pin_cell_id"], "role": row["pin_role"],
        "ref": row["substrate_pin"]["resolved_sha"],
        "mean_mel": row["mean_mel"], "mean_mel_repr": row["mean_mel_repr"],
        "mel_reference": row["mel_reference"],
        "mel_reference_repr": row["mel_reference_repr"],
        "mean_duration_factor": row["mean_duration_factor"],
        "high_graded": bool(float(row["mean_duration_factor"]) > FACTOR_GRADED_FLOOR),
        "conv_rel_drop": row.get("conv_rel_drop"),
        "n_cycles_insufficient_touched_slots": row.get("n_cycles_insufficient_touched_slots"),
        "per_cycle_mel": row.get("per_cycle_mel"),
        "cumulative_sws_writes": row.get("cumulative_sws_writes"),
        "cumulative_rem_rollouts": row.get("cumulative_rem_rollouts"),
    }


def _bitwise(x: Dict[str, Any], y: Dict[str, Any]) -> bool:
    return (x["mean_mel_repr"] == y["mean_mel_repr"]
            and x["mel_reference_repr"] == y["mel_reference_repr"])


def _analyse(rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    pre = rows[CELL_BY_ROLE["anchor_pre"]]
    a = rows[CELL_BY_ROLE["pin_a"]]
    b = rows[CELL_BY_ROLE["pin_b"]]
    post = rows[CELL_BY_ROLE["anchor_post"]]

    d_pre_post = _D(pre, post)
    d_a_pre = _D(a, pre)
    d_b_post = _D(b, post)
    d_a_b = _D(a, b)
    d_pre_rec = _D(pre, RECORDED_ANCHORS["anchor_pre"])
    d_post_rec = _D(post, RECORDED_ANCHORS["anchor_post"])

    p1 = d_pre_post is not None and d_pre_post >= ANCHOR_DELTA_MIN
    c1 = d_a_pre is not None and d_a_pre <= EQUIV_REL_TOL
    c2 = d_b_post is not None and d_b_post <= EQUIV_REL_TOL
    c3_equal = d_a_b is not None and d_a_b <= EQUIV_REL_TOL
    c4 = (d_pre_rec is not None and d_post_rec is not None
          and max(d_pre_rec, d_post_rec) <= EQUIV_REL_TOL)
    fa = float(a["mean_duration_factor"])
    fb = float(b["mean_duration_factor"])
    c5 = fa > FACTOR_GRADED_FLOOR and fb < FACTOR_GRADED_FLOOR

    if not p1:
        label, outcome = "uninformative_anchor_delta_absent_in_run", "FAIL"
    elif c1 and c2:
        label = ("mover_6293b23_confirmed_full_protocol_"
                 + ("reproduces_recorded" if c4 else "recorded_not_reproduced"))
        outcome = "PASS"
    elif c3_equal:
        label, outcome = "not_confirmed_6293b23_inert_under_full_protocol", "FAIL"
    elif c1:
        # B != POST. Separate a sub-effect perturbation (below the anchor-effect
        # floor) from one that takes a share of the effect (red-team F2).
        sub = d_b_post is not None and d_b_post < ANCHOR_DELTA_MIN
        label = ("boundary_confirmed_later_commits_sub_effect_perturbation" if sub
                 else "boundary_confirmed_later_commits_share_effect")
        outcome = "FAIL"
    elif c2:
        sub = d_a_pre is not None and d_a_pre < ANCHOR_DELTA_MIN
        label = ("boundary_confirmed_earlier_commits_sub_effect_perturbation" if sub
                 else "boundary_confirmed_earlier_commits_share_effect")
        outcome = "FAIL"
    else:
        label, outcome = "multiple_commits_perturb_full_protocol", "FAIL"

    preconditions = [
        {"name": "anchor_pre_post_difference_present_in_run", "kind": "readiness",
         "description": ("D(ANCHOR_PRE_f810969, ANCHOR_POST_17befb8c) must clear "
                         "ANCHOR_DELTA_MIN: the effect to attribute has to exist in "
                         "this run before C1/C2 can mean anything. Same statistic D "
                         "the load-bearing criteria route on."),
         "measured": d_pre_post, "threshold": ANCHOR_DELTA_MIN, "direction": "lower",
         "control": ("positive control = the two range endpoints, recorded apart at "
                     "D=0.1115 (861g n10 cell vs 861e cell); floor pre-registered "
                     "~11x below that, not fitted to this run"),
         "certifies": "C1 and C2 (both compare against these anchors)",
         "met": bool(p1)},
        {"name": "all_four_pins_verified_structural_marker_content", "kind": "provenance",
         "description": ("every child cell passed verify_pin (structural + attr "
                         "markers), the agent.py phase_reset() source marker, and the "
                         "full ree_core/**/*.py blob comparison against git; a failure "
                         "raises in the child and no manifest is written"),
         "met": all(bool(r["substrate_pin"].get("verified")) for r in rows.values())},
    ]
    criteria = [
        {"name": "C1_pin_a_5f64a53f_equivalent_to_anchor_pre_f810969",
         "load_bearing": True, "measured": d_a_pre, "threshold": EQUIV_REL_TOL,
         "comparator": "<=", "passed": bool(c1),
         "bitwise_equal": _bitwise(a, pre)},
        {"name": "C2_pin_b_6293b23_equivalent_to_anchor_post_17befb8c",
         "load_bearing": True, "measured": d_b_post, "threshold": EQUIV_REL_TOL,
         "comparator": "<=", "passed": bool(c2),
         "bitwise_equal": _bitwise(b, post)},
        {"name": "C3_pin_a_vs_pin_b_distance",
         "load_bearing": False, "measured": d_a_b, "threshold": EQUIV_REL_TOL,
         "comparator": "<=", "passed": bool(c3_equal),
         "note": "passed == the two boundary cells are EQUIVALENT (6293b23 inert)"},
        {"name": "C4_anchors_reproduce_recorded_861g_861e_cells",
         "load_bearing": False,
         "measured": (max(d_pre_rec, d_post_rec)
                      if d_pre_rec is not None and d_post_rec is not None else None),
         "threshold": EQUIV_REL_TOL, "comparator": "<=", "passed": bool(c4),
         "measured_pre_vs_861g": d_pre_rec, "measured_post_vs_861e": d_post_rec},
        {"name": "C5_grading_split_pin_a_high_pin_b_low",
         "load_bearing": False, "measured_factor_pin_a": fa,
         "measured_factor_pin_b": fb, "threshold": FACTOR_GRADED_FLOOR,
         "passed": bool(c5)},
    ]
    if outcome == "PASS" and c4:
        reading = ("6293b23 carries the whole in-run f810969->17befb8c difference AND the "
                   "anchors reproduce the recorded 861g/861e cells: the recorded 0.884 "
                   "movement is attributable to 6293b23.")
    elif outcome == "PASS":
        reading = ("6293b23 carries the whole IN-RUN anchor difference, but the anchors do "
                   "NOT reproduce the recorded 861g/861e cells (first suspect: a "
                   "machine_class / torch / live-harness change). This supports "
                   "attribution of THIS run's anchor gap only, NOT of the recorded 0.884.")
    else:
        reading = f"not a confirmation; see label {label} and distances."
    return {
        "label": label, "outcome": outcome, "reading": reading,
        "p1": p1, "c1": c1, "c2": c2, "c3_equal": c3_equal, "c4": c4, "c5": c5,
        "distances": {"pre_post": d_pre_post, "a_pre": d_a_pre, "b_post": d_b_post,
                      "a_b": d_a_b, "pre_vs_recorded_861g": d_pre_rec,
                      "post_vs_recorded_861e": d_post_rec},
        "bitwise": {"a_eq_pre": _bitwise(a, pre), "b_eq_post": _bitwise(b, post),
                    "a_eq_b": _bitwise(a, b), "pre_eq_post": _bitwise(pre, post)},
        "factors": {cid: float(r["mean_duration_factor"]) for cid, r in rows.items()},
        "preconditions": preconditions, "criteria": criteria,
    }


def _flat_scalar(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return v
    return None


def run_experiment(dry_run: bool) -> Dict[str, Any]:
    rows: Dict[str, Dict[str, Any]] = {}
    flags_per_pin: Dict[str, Any] = {}
    for spec in PIN_CELLS:
        payload = _run_child(spec, dry_run)
        row = payload["row"]
        rows[spec["cell_id"]] = row
        _ZG.observe_stats(payload.get("z_goal_stream_stats"))
        flags_per_pin[spec["cell_id"]] = payload.get("enabled_default_off_flags")
    an = _analyse(rows)
    return {"rows": rows, "analysis": an, "flags_per_pin": flags_per_pin}


def write_manifest(result: Dict[str, Any], *, dry_run: bool,
                   started_at: float) -> str:
    from experiments.pack_writer import write_flat_manifest

    ts = dt.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    an = result["analysis"]
    rows = result["rows"]
    arm_results = [rows[c["cell_id"]] for c in PIN_CELLS]
    p = _params(dry_run)
    full_config = {
        "env_base": ENV_BASE, "arm": DECISIVE_ARM, "seed": DECISIVE_SEED,
        "world_rule_shift_depth": WORLD_RULE_SHIFT_DEPTH,
        "params": p, "calib_draws_production": CALIB_DRAWS,
        "pin_cells": [{"cell_id": c["cell_id"], "role": c["role"], "ref": c["ref"],
                       "attr_markers": c["attr_markers"],
                       "agent_phase_reset_lines": c["agent_phase_reset_lines"]}
                      for c in PIN_CELLS],
        "equiv_rel_tol": EQUIV_REL_TOL, "anchor_delta_min": ANCHOR_DELTA_MIN,
        "factor_graded_floor": FACTOR_GRADED_FLOOR,
        "use_mech122_spindle_content_selection": USE_MECH122_SPINDLE_CONTENT_SELECTION,
    }
    readout: Dict[str, Any] = {}
    for k, v in an["distances"].items():
        fv = _flat_scalar(v)
        if fv is not None:
            readout[f"D_{k}"] = fv
    for cid, f in an["factors"].items():
        readout[f"factor_{cid}"] = f
    for c in PIN_CELLS:
        r = rows[c["cell_id"]]
        readout[f"mean_mel_{c['cell_id']}"] = float(r["mean_mel"])
        readout[f"mel_reference_{c['cell_id']}"] = float(r["mel_reference"])
    for k in ("p1", "c1", "c2", "c3_equal", "c4", "c5"):
        readout[k] = 1 if an[k] else 0
    for k, v in an["bitwise"].items():
        readout[f"bitwise_{k}"] = 1 if v else 0
    readout["outcome_pass"] = 1 if an["outcome"] == "PASS" else 0
    readout["outcome_pass_qualified_by_c4"] = 1 if (an["outcome"] == "PASS" and an["c4"]) else 0

    flags_union: Dict[str, Any] = {}
    for v in result["flags_per_pin"].values():
        if isinstance(v, dict):
            flags_union.update(v)

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes": SUPERSEDES,
        "compares_against_run_id": COMPARES_AGAINST_RUN_ID,
        "source_bisect_doc": SOURCE_BISECT_DOC,
        "source_chip_ref": SOURCE_CHIP_REF,
        "red_team_verdict": RED_TEAM_VERDICT,
        "sleep_driver_pattern": ("manual-cycle-loop (force_cycle() once per cycle in a "
                                 "MEAS_CYCLES wake-sleep loop; unchanged from 861e/861g)"),
        "timestamp_utc": ts,
        "seeds": [DECISIVE_SEED],
        "dry_run": bool(dry_run),
        "outcome": an["outcome"],
        "evidence_direction": "non_contributory",
        "evidence_direction_per_claim": {"INV-050": "unknown", "MECH-180": "unknown"},
        "evidence_direction_note": (
            "diagnostic provenance attribution (which ree_core commit moved one seed's "
            "readout); tests no claim hypothesis; directions pinned regardless of outcome"),
        "interpretation": {
            "label": an["label"],
            "reading": an["reading"],
            "preconditions": an["preconditions"],
            "criteria_non_degenerate": {
                "C1": bool(an["p1"]), "C2": bool(an["p1"]), "C3": bool(an["p1"]),
                "C4": True, "C5": bool(an["p1"]),
            },
        },
        "criteria": an["criteria"],
        "combination_rule": ("PASS iff P1 (anchor difference present in run) AND C1 "
                             "(pin A == anchor pre) AND C2 (pin B == anchor post), "
                             "equivalence by D <= EQUIV_REL_TOL; C3-C5 are recorded "
                             "qualifiers that select the FAIL label or suffix the PASS"),
        "readout": readout,
        "attribution": {
            "distances": an["distances"], "bitwise": an["bitwise"],
            "factors": an["factors"],
            "cells": {cid: _cell_summary(r) for cid, r in rows.items()},
            "recorded_anchors": RECORDED_ANCHORS,
        },
        "substrate_pins": {cid: r["substrate_pin"] for cid, r in rows.items()},
        "pin_verification": {cid: r["pin_verification"] for cid, r in rows.items()},
        "substrate_hash_note": (
            "arm_results carry FOUR distinct per-cell substrate_hash values by design "
            "(one per pinned ree_core commit, scoped to ree_core/**/*.py under each pin "
            "dir). Any hoisted top-level substrate_hash is the first cell's only; read "
            "substrate_pins for provenance. All cells are reuse-INELIGIBLE."),
        "enabled_default_off_flags": flags_union,
        "enabled_default_off_flags_per_pin": result["flags_per_pin"],
        "arm_results": arm_results,
        "thresholds": {"EQUIV_REL_TOL": EQUIV_REL_TOL,
                       "ANCHOR_DELTA_MIN": ANCHOR_DELTA_MIN,
                       "FACTOR_GRADED_FLOOR": FACTOR_GRADED_FLOOR},
        "custom_information": {
            "dead_z_goal_stream_exempt": DEAD_Z_GOAL_STREAM_EXEMPT,
            "gate_dispositions": ("see module docstring QUEUE-EXPERIMENT GATE "
                                  "DISPOSITIONS: 2.4 not recoverable; 2.5b brake "
                                  "released (diagnostic, not a re-derive); 2.5c "
                                  "contextmemory corrupting overlap not blocking "
                                  "(historical pinned code; 861h not load-bearing)"),
        },
    }
    out_path = write_flat_manifest(
        manifest,
        None,
        dry_run=dry_run,
        config=full_config,
        seeds=[DECISIVE_SEED],
        script_path=Path(__file__),
        z_goal_stream_stats=_ZG.stats(),
        started_at=started_at,
    )
    return str(out_path)


def main() -> Tuple[Dict[str, Any], str, bool]:
    t0 = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="4 pinned cells at tiny scale (smoke)")
    args = ap.parse_args()
    result = run_experiment(bool(args.dry_run))
    out_path = write_manifest(result, dry_run=bool(args.dry_run), started_at=t0)
    an = result["analysis"]
    print(f"outcome: {an['outcome']}", flush=True)
    print(f"label: {an['label']}", flush=True)
    print(f"reading: {an['reading']}", flush=True)
    print(f"distances: {an['distances']}", flush=True)
    print(f"bitwise: {an['bitwise']}", flush=True)
    print(f"factors: {an['factors']}", flush=True)
    print(f"manifest: {out_path}", flush=True)
    return result, out_path, bool(args.dry_run)


if __name__ == "__main__":
    if "--child-cell" in sys.argv[1:]:
        sys.exit(_child_main(sys.argv[1:]))
    _result, _out_path, _dry_run = main()
    _outcome_raw = str(_result["analysis"]["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        dry_run=_dry_run,
    )
