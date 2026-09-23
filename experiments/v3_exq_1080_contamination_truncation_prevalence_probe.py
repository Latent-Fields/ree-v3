"""V3-EXQ-1080 -- contamination-footgun PREVALENCE probe (GFLAG-0304 option B, user-ruled
2026-09-23). Diagnostic, claim-free.

QUESTION. The corpus audit `REE_assembly/evidence/planning/corpus_audit_contamination_footgun_
20260916.md` found 92 CausalGridWorld runs (60 chip-scored) configured `num_hazards=0` with no
contamination opt-out, so a self-poisoning agent could have truncated its own measurement window
(the SD-094 footgun: `contamination_spread` defaults to 0.5 on EVERY entered cell regardless of
`num_hazards`; four entries mark a cell contaminated and each re-entry drains 0.4 health). 13
indexer-scored claims rest >= 50% on such runs. Truncation was actually MEASURED only once
(V3-EXQ-940, C1, 11% margin). This probe asks, on four representative exposed runs: how often
does the footgun actually fire, and does it change the run's OWN verdict?

DESIGN. For each target, run the target driver's OWN grid loop twice, at the target's original
full-scale configuration, under a class-level instrument on `CausalGridWorld`:
  ARM_STOCK   the target exactly as written (contamination at stock defaults) -- the
              reproduction arm; this is what measures prevalence.
  ARM_OPTOUT  identical, except every env constructed during the run gets
              `hazard_free_contamination_gate=True`. That gate zeroes contamination_spread ONLY
              when num_hazards == 0 (causal_grid_world.py SD-094 note), so a target's hazarded
              envs (888's threat context, 939a's HAZARD arms) are untouched -- the manipulation
              is scoped to exactly the footgun.
The two arms are PAIRED: every target cell enters through its own `arm_cell(...)`, which resets
all RNG, so STOCK and OPTOUT start each cell from bit-identical state and differ only in the gate.

Targets (claims directly re-measured; family = claims whose exposed runs share the target's env
constructor, to which the result transfers by construction rather than by measurement):
  V3-EXQ-888  MECH-074, MECH-074a, MECH-074b. Family: MECH-074d (894/894a/894b/894c build the
              identical THREAT/NEUTRAL_ENV_KWARGS pair). Neutral context = size 10, 0 hazards,
              60-step episodes, break-on-done. Original: PASS / supports.
  V3-EXQ-669c MECH-329, MECH-189. Family: SD-077 (1040 uses the identical nursery constructor).
              Nursery = CausalGridWorldV2(size=8, num_hazards=0, num_resources=6), 100-step
              episodes, break-on-done. Original: FAIL / mixed.
  V3-EXQ-904  ARC-070. Family: SD-079 (807/823 use the identical size-8 V2 constructor, 32-step
              episodes vs 904's 24). Break-on-done. Original: PASS / supports.
  V3-EXQ-939a MECH-303. Mixed SAFE (0 hazards) / HAZARD (8) contexts, size 10, random-policy
              exposure walk that RESETS on death and continues (so a death is a harm event +
              layout reset inside a nominally safe context, not a lost window). Original:
              PASS / supports.
NOT covered by any target or family, stated rather than extrapolated silently: MECH-106
(V3-EXQ-231a), INV-054 (V3-EXQ-278/435), MECH-427 (V3-EXQ-883, size-10 40-step V2 env).

INSTRUMENT. `CausalGridWorld.__init__/step/reset` are wrapped at class level (CausalGridWorldV2
is a factory over that class, so every driver is covered). Per env instance: num_hazards at
construction (hazard-free = 0), effective contamination_spread, gate-applied flag. Per episode:
length (`info["episode_steps"]`), cause (`info["done_cause"]`: health_depleted / step_limit), or
`driver_budget` when the driver itself reset/abandoned the episode before the env ended it;
contaminated-cell contacts (`transition_type == "agent_caused_hazard"`, which in a hazard-free
env can only be a contaminated cell); minimum health. Target stdout is captured (so the runner
parses only this driver's own progress lines) and its tail is kept in the manifest.

FIDELITY CAVEAT (load-bearing for interpretation). This re-runs the targets on the CURRENT
substrate, not the substrate they originally ran on. ARM_STOCK therefore measures what the
original CONFIGURATION does today; `stock_reproduces_original` records whether ARM_STOCK's verdict
matches the historical manifest. A mismatch weakens the transfer to the historical run but not
the within-run STOCK-vs-OPTOUT comparison.

PRE-REGISTERED (constants below, fixed before any real run; amended once, pre-run, by the
Step 4.5 red-team -- see RED-TEAM DISPOSITIONS at the end of this docstring):
  DV-WINDOW UNIT. The death fraction is taken over the window that carries each target's DV:
      break-on-done drivers (669c, 904, 888): unit = one hazard-free EPISODE; it "dies" if it
          ended health_depleted (each such death IS a truncated window).
      939a (reset-and-continue): unit = one hazard-free DV WALK -- the P1 exposure walk or the
          P2 test walk, identified by phase-tagging those two driver functions at run time; it
          "dies" if it contains >= 1 health_depleted segment (an intrusion of harm and a layout
          reset into a nominally SAFE context). 939a's P0 warmup deaths are reported separately
          (they are a different mechanism -- encoder-buffer truncation -- and are 97% of its
          hazard-free episodes, so pooling them would mismeasure the DV window).
  MATERIAL_DEATH_FRAC = 0.10 -- a target is MATERIALLY TRUNCATED when >= 10% of its ARM_STOCK
      DV-window units die.
  VERDICT CHANGE -- the target's own outcome (PASS/FAIL), overall evidence_direction, ANY
      per-claim direction, or its interpretation label differs between ARM_STOCK and ARM_OPTOUT.
  TARGET DEGENERACY -- a target verdict is degenerate when its direction is unknown /
      non_contributory or its label is a readiness route (not_ready / requeue / starved). Both
      arms degenerate -> cannot_determine (the target measured nothing; it must NOT count as
      robust). Exactly one arm degenerate -> that IS a gate-caused change (sensitive).
  Per-target class:
      clean                          dv_death_frac < 0.10, verdict unchanged
      truncated_verdict_robust       dv_death_frac >= 0.10, verdict unchanged
      verdict_sensitive_truncation   verdict changed, dv_death_frac >= 0.10
      verdict_sensitive_observation  verdict changed, dv_death_frac < 0.10, >= 1 STOCK contact
      verdict_sensitive_contact_free verdict changed with ZERO contaminated contacts in STOCK:
                                     the gate also zeroes the contamination_view observation
                                     channel from step 1, so this flip is a knife-edge verdict
                                     moved by an observation change, not by poisoning. It still
                                     counts as sensitive (a gated re-run would change the
                                     verdict) but is labelled so it is not read as truncation.
      cannot_determine               a per-target precondition failed, or both arms degenerate
                                     (excluded; never vacates another target -- V3-EXQ-785)
  Run-level routing on n_sensitive (verdict_sensitive_*) and n_truncated:
      not ready                      -> substrate_not_ready_requeue
      n_sensitive 0, n_truncated 0   -> contamination_prevalence_low_no_reruns_owed
      n_sensitive 0, n_truncated >=1 -> contamination_truncation_present_verdicts_robust_no_reruns_owed
      n_sensitive 1                  -> contamination_isolated_sensitivity_reruns_owed
      n_sensitive >= 2               -> contamination_prevalence_high_all_reruns_owed
  What "no re-runs owed" covers: ONLY the determinable targets' DIRECT claims. Family claims are
      never cleared (transfer is of the manipulation's reach, not of verdict sensitivity: 894*
      add a P0 warmup 888 lacks; 807/823 run 32-step episodes vs 904's 24); their caveat notes
      stand. Undetermined targets' claims move to claims_not_covered. A determinable target
      whose STOCK verdict does not reproduce its historical manifest is listed under
      historical_verdict_not_reproduced (a substrate-drift finding for /governance, not a
      contamination re-run).
  Readiness: positive control met AND >= MIN_DETERMINABLE_TARGETS (3) determinable targets.
  Every target and the control must record exactly its INTENDED number of DV units in both
      arms (a death ends a unit, it never adds one), so a silently dropped episode fails loud.
  outcome PASS iff ready and n_sensitive == 0; FAIL otherwise.

POSITIVE CONTROL (P0, same statistic as the load-bearing criterion). A random walker on 669c's
nursery geometry, 20 episodes x 100 steps, measured before the targets: at stock contamination
death_frac must be >= 0.5 (measured 0.95 on 2026-09-23 while designing this probe), and with the
gate it must be 0. That proves the instrument can see a truncation and that the gate removes it.

DV-SYMMETRY (Step 3.5). DV = fraction of hazard-free episodes whose cause is health_depleted; its
symmetry group is permutation of episodes. The manipulation (zeroing contamination deposition)
is not a permutation of episodes -- it changes which episodes die -- so it is not invariant.
The verdict DV is each target's own categorical verdict; the gate is not invariant under it
either (it can change the target's inputs).

red-team (fable): CONTESTED, 8 findings, all FIXED pre-run (none dismissed):
  F1 degenerate-in-both-arms target scored robust -> cannot_determine + label in verdict key;
  F2 label ignored truncation -> truncation_present label; F3 undetermined targets' claims
  reported covered -> moved to claims_not_covered; F4 stock_reproduces_original unrouted ->
  historical_verdict_not_reproduced list; F5 939a death_frac was 97% P0 warmup -> DV-walk unit
  via phase tags, warmup reported separately; F6 family claims inherited the target's
  disposition -> never cleared, caveat stands; F7 contact-free flip -> own class; F8 realized-n
  readiness -> intended-count + arm-equality preconditions. Low note (C_PREV measured/passed
  inconsistent when not ready) fixed. Low note "gated control rail cannot fail" dismissed: it
  can fail if the gate does not bind on the control env (spread != 0), which the target-level
  optout_hf_spread_zero checks do not cover.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import random
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.environment import causal_grid_world as _cgw  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorld, CausalGridWorldV2  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.episode_termination import (  # noqa: E402
    EpisodeTerminationAccumulator, stats_from_episodes,
)

EXPERIMENT_TYPE = "v3_exq_1080_contamination_truncation_prevalence_probe"
QUEUE_ID = "V3-EXQ-1080"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS: List[str] = []   # claim-free: this measures the evidence base, it tests no claim
AUDITED_CLAIM_IDS = [
    "ARC-070", "INV-054", "MECH-074", "MECH-074a", "MECH-074b", "MECH-074d", "MECH-106",
    "MECH-189", "MECH-303", "MECH-329", "MECH-427", "SD-077", "SD-079",
]
SOURCE_FLAG = "GFLAG-0304"
# The control predicate IS the statistic (fraction of hazard-free episodes ending
# health_depleted), not a narrower hand-written signature of it, and both rails were
# measured reachable at design time (stock 0.95, gated 0.00; 20x100 random walk,
# 2026-09-23), so it is reachable by construction.
ANCHOR_REACHABILITY_EXEMPT = ("control predicate is the death-fraction statistic itself; "
                              "both rails measured reachable (0.95 stock / 0.00 gated)")
SOURCE_AUDIT = "REE_assembly/evidence/planning/corpus_audit_contamination_footgun_20260916.md"

# ---- pre-registered constants (never derived from this run's own statistics) ----
MATERIAL_DEATH_FRAC = 0.10
MIN_DETERMINABLE_TARGETS = 3
CONTROL_EPISODES = 20
CONTROL_STEPS = 100
CONTROL_STOCK_DEATH_FLOOR = 0.5
CONTROL_GATED_DEATH_CEIL = 0.0

ARM_STOCK = "ARM_STOCK"
ARM_OPTOUT = "ARM_OPTOUT"
ARMS = (ARM_STOCK, ARM_OPTOUT)

PROGRESS_DENOM = 100            # [train] ep N/100 per (target, arm) run == queue episodes_per_run

UNCOVERED_CLAIMS = {
    "MECH-106": "V3-EXQ-231a (size-8 mixed 0/2/5-hazard probe envs)",
    "INV-054": "V3-EXQ-278/435 (300+300-episode depression/recovery, LOW_HARM phase)",
    "MECH-427": "V3-EXQ-883 (size-10 V2 env, 40-step episodes)",
}


# ======================================================================================
# Class-level instrument on CausalGridWorld
# ======================================================================================
class _ProbeState:
    def __init__(self) -> None:
        self.cell: Optional[str] = None          # "<target>::<arm>" or "control::<arm>"
        self.force_gate: bool = False
        self.episodes: Dict[str, List[Dict[str, Any]]] = {}
        self.census: Dict[str, Dict[str, int]] = {}
        # STRONG refs, held for one (target, arm) run: a target drops its env when its cell
        # function returns, and a weak ref would let the last in-flight episode of every
        # cell vanish unrecorded (found in the dry-run smoke: 9 of 12 669c episodes).
        self.live: List[Any] = []
        self.post_done_steps: Dict[str, int] = {}
        self.phase: str = "all"                  # set by _phased() wrappers (939a only)
        self.env_seq: int = 0


_STATE = _ProbeState()
_ORIG_INIT = CausalGridWorld.__init__
_ORIG_STEP = CausalGridWorld.step
_ORIG_RESET = CausalGridWorld.reset


def _record_episode(env: Any, steps: int, cause: str) -> None:
    cell = getattr(env, "_probe_cell", None)
    if cell is None:
        return
    _STATE.episodes.setdefault(cell, []).append({
        "hf": bool(env._probe_hf),
        "steps": int(steps),
        "cause": str(cause or "") or "driver_budget",
        "contacts": int(env._probe_contacts),
        "min_health": float(env._probe_min_health),
        "env_id": int(env._probe_env_id),
        "phase": str(env._probe_ep_phase or _STATE.phase),
    })
    env._probe_open = False
    env._probe_closed = True


def _begin_episode(env: Any) -> None:
    env._probe_open = False
    env._probe_closed = False
    env._probe_contacts = 0
    env._probe_min_health = 1.0
    env._probe_ep_phase = None


def _probe_init(self, *args, **kwargs):
    if _STATE.force_gate:
        kwargs["hazard_free_contamination_gate"] = True
    _ORIG_INIT(self, *args, **kwargs)
    self._probe_cell = _STATE.cell
    _STATE.env_seq += 1
    self._probe_env_id = _STATE.env_seq
    self._probe_hf = int(getattr(self, "num_hazards", -1)) == 0
    _begin_episode(self)
    if _STATE.cell is not None:
        sig = json.dumps({
            "num_hazards": int(getattr(self, "num_hazards", -1)),
            "size": int(getattr(self, "size", -1)),
            "toroidal": bool(getattr(self, "toroidal", False)),
            "contamination_spread": float(getattr(self, "contamination_spread", -1.0)),
            "gate_applied": bool(getattr(self, "_contamination_gate_applied", False)),
            "max_episode_steps": int(getattr(self, "max_episode_steps", -1)),
        }, sort_keys=True)
        c = _STATE.census.setdefault(_STATE.cell, {})
        c[sig] = c.get(sig, 0) + 1
        _STATE.live.append(self)


def _probe_step(self, action):
    out = _ORIG_STEP(self, action)
    if getattr(self, "_probe_cell", None) is None:
        return out
    done, info = out[2], out[3]
    if self._probe_closed:
        # The driver kept stepping a finished episode without a reset. Counted, not scored.
        _STATE.post_done_steps[self._probe_cell] = (
            _STATE.post_done_steps.get(self._probe_cell, 0) + 1)
        return out
    if not self._probe_open:
        self._probe_ep_phase = _STATE.phase
    self._probe_open = True
    if info.get("transition_type") == "agent_caused_hazard":
        self._probe_contacts += 1
    try:
        self._probe_min_health = min(self._probe_min_health, float(info.get("health", 1.0)))
    except Exception:
        pass
    if done:
        _record_episode(self, info.get("episode_steps", self.steps), info.get("done_cause", ""))
    return out


def _probe_reset(self, *args, **kwargs):
    if getattr(self, "_probe_cell", None) is not None and getattr(self, "_probe_open", False):
        _record_episode(self, int(getattr(self, "steps", 0)), "")
    out = _ORIG_RESET(self, *args, **kwargs)
    if getattr(self, "_probe_cell", None) is not None:
        _begin_episode(self)
    return out


def _install_instrument() -> None:
    CausalGridWorld.__init__ = _probe_init
    CausalGridWorld.step = _probe_step
    CausalGridWorld.reset = _probe_reset


def _flush_live() -> None:
    """Close every episode still in flight at the end of a (target, arm) run."""
    for env in list(_STATE.live):
        if getattr(env, "_probe_open", False):
            _record_episode(env, int(getattr(env, "steps", 0)), "")
    _STATE.live = []


@contextlib.contextmanager
def _cell(cell: str, force_gate: bool):
    _STATE.cell = cell
    _STATE.force_gate = bool(force_gate)
    _STATE.episodes.setdefault(cell, [])
    try:
        yield
    finally:
        _flush_live()
        _STATE.cell = None
        _STATE.force_gate = False


# ======================================================================================
# Target stdout capture: the runner must parse only THIS driver's progress lines
# ======================================================================================
class _TargetStdout(io.TextIOBase):
    def __init__(self, real, label: str, n_cells: int) -> None:
        self._real = real
        self._label = label
        self._n_cells = max(1, int(n_cells))
        self._buf = ""
        self.tail: "deque[str]" = deque(maxlen=60)
        self.cells_done = 0
        self._last_pct = -1

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._line(line)
        return len(s)

    def _line(self, line: str) -> None:
        self.tail.append(line.encode("ascii", "replace").decode("ascii")[:300])
        if line.strip().startswith("verdict:"):
            self.cells_done += 1
            pct = min(PROGRESS_DENOM - 1, int(PROGRESS_DENOM * self.cells_done / self._n_cells))
            if pct != self._last_pct:
                self._last_pct = pct
                self._real.write(
                    f"  [train] probe {self._label} ep {pct}/{PROGRESS_DENOM} "
                    f"target_cells={self.cells_done}/{self._n_cells}\n")
                self._real.flush()

    def flush(self) -> None:
        self._real.flush()


# ======================================================================================
# Target adapters -- each runs the target's OWN grid loop and returns its OWN verdict
# ======================================================================================
def _scalars(d: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, bool):
                out[k] = int(v)
            elif isinstance(v, (int, float)) and np.isfinite(float(v)):
                out[k] = v
    return out


def _phased(fn, name: str):
    """Tag every episode that OPENS inside `fn` with phase `name`."""
    def wrapper(*a, **k):
        prev = _STATE.phase
        _STATE.phase = name
        try:
            return fn(*a, **k)
        finally:
            _STATE.phase = prev
    return wrapper


def _mod(name: str):
    return importlib.import_module(f"experiments.{name}")


def _run_669c(dry: bool) -> Dict[str, Any]:
    m = _mod("v3_exq_669c_mech329_wanting_first_goal_seeding")
    n_child, n_wean, steps, seeds = (2, 2, 20, [42]) if dry else (8, 6, 100, [42, 43, 44])
    r = m.run_experiment(n_child, n_wean, steps, seeds, dry)
    return {
        "outcome": str(r["outcome"]).upper(),
        "evidence_direction": r.get("evidence_direction"),
        "evidence_direction_per_claim": r.get("evidence_direction_per_claim") or {},
        "label": (r.get("interpretation") or {}).get("label"),
        "target_metrics": _scalars(r.get("metrics")),
    }


def _run_904(dry: bool) -> Dict[str, Any]:
    m = _mod("v3_exq_904_arc070_decomposition_trigger_selectivity")
    m.SEEDS = [11] if dry else [11, 23, 47, 71]
    m.N_EPISODES = 6 if dry else 20
    r = m.run_experiment()
    return {
        "outcome": str(r["outcome"]).upper(),
        "evidence_direction": r.get("evidence_direction"),
        "evidence_direction_per_claim": r.get("evidence_direction_per_claim") or {},
        "label": (r.get("interpretation") or {}).get("label"),
        "target_metrics": _scalars(r.get("metrics")),
    }


def _run_888(dry: bool) -> Dict[str, Any]:
    m = _mod("v3_exq_888_mech074_readwrite_head_route_dissociation")
    seeds = m.SEEDS[:1] if dry else list(m.SEEDS)
    m.assert_no_structurally_unsatisfiable_gate(
        m.PRECONDITION_SPECS, [m._arm_ctx(a) for a in m.ARMS])
    rows = []
    for seed in seeds:           # main()'s own loop: _run_cell wraps itself in arm_cell
        for arm in m.ARMS:
            rows.append(m._run_cell(seed, arm, dry=dry))
    ev = m._evaluate(rows)
    aor: Dict[str, List[float]] = {}
    for r in rows:
        aor.setdefault(r["arm"], []).append(float(r["arousal_over_representation_z"]))
    metrics = _scalars(ev)
    for a, vals in aor.items():
        metrics[f"aor_z_mean_{a}"] = float(np.mean(vals))
    return {
        "outcome": str(ev["outcome"]).upper(),
        "evidence_direction": ev.get("evidence_direction"),
        "evidence_direction_per_claim": ev.get("evidence_direction_per_claim") or {},
        "label": ev.get("interpretation_label"),
        "target_metrics": metrics,
    }


def _run_939a(dry: bool) -> Dict[str, Any]:
    m = _mod("v3_exq_939a_mech303_proximity_gated_contextual_safety_vigilance_release")
    if dry:
        m._DRY_RUN = True
        m.EXPOSURE_STEPS, m.N_TEST_TRIALS, m.P0_EPISODES, m.MIN_VALID_SEEDS = 30, 6, 3, 1
    else:
        m._DRY_RUN = False
        m.EXPOSURE_STEPS, m.N_TEST_TRIALS, m.P0_EPISODES, m.MIN_VALID_SEEDS = 240, 90, 60, 4
    m.TOTAL_DENOM = m.P0_EPISODES + m.EXPOSURE_STEPS
    seeds = list(range(2 if dry else m.N_SEEDS))
    random.seed(m.SEED_BASE)            # build_and_run()'s own pre-run seeding
    torch.manual_seed(m.SEED_BASE)
    np.random.seed(m.SEED_BASE)
    orig = (m._p0_warmup, m._expose, m._test_release)
    m._p0_warmup = _phased(orig[0], "warmup")
    m._expose = _phased(orig[1], "exposure")
    m._test_release = _phased(orig[2], "test")
    try:
        man, _overall = m.run_experiment(seeds)
    finally:
        m._p0_warmup, m._expose, m._test_release = orig
    return {
        "outcome": str(man["outcome"]).upper(),
        "evidence_direction": man.get("evidence_direction"),
        "evidence_direction_per_claim": man.get("evidence_direction_per_claim") or {},
        "label": (man.get("interpretation") or {}).get("label"),
        "target_metrics": _scalars(man.get("metrics") or man.get("readout")),
    }


TARGETS: List[Dict[str, Any]] = [
    {"key": "669c", "queue_id": "V3-EXQ-669c", "run": _run_669c,
     "script": "experiments/v3_exq_669c_mech329_wanting_first_goal_seeding.py",
     "direct_claims": ["MECH-329", "MECH-189"], "family_claims": ["SD-077"],
     "original": {"outcome": "FAIL", "evidence_direction": "mixed",
                  "run_id": "v3_exq_669c_mech329_wanting_first_goal_seeding_20260722T214724Z_v3"},
     "hf_step_budget": {True: 20, False: 100}, "cells": {True: 3, False: 9},
     "unit": "episode", "dv_phases": None,
     "intended_units": {True: 3 * 4, False: 9 * 14}},     # cells x (n_child + n_wean)
    {"key": "939a", "queue_id": "V3-EXQ-939a", "run": _run_939a,
     "script": "experiments/v3_exq_939a_mech303_proximity_gated_contextual_safety_vigilance_release.py",
     "direct_claims": ["MECH-303"], "family_claims": [],
     "original": {"outcome": "PASS", "evidence_direction": "supports",
                  "run_id": "v3_exq_939a_mech303_proximity_gated_contextual_safety_vigilance_release_20260821T235047Z_v3"},
     "hf_step_budget": {True: None, False: None}, "cells": {True: 8, False: 24},
     "unit": "walk", "dv_phases": ("exposure", "test"),
     "intended_units": {True: 2 * 2 * 2, False: 6 * 2 * 2}},  # seeds x SAFE arms x (P1+P2)
    {"key": "904", "queue_id": "V3-EXQ-904", "run": _run_904,
     "script": "experiments/v3_exq_904_arc070_decomposition_trigger_selectivity.py",
     "direct_claims": ["ARC-070"], "family_claims": ["SD-079"],
     "original": {"outcome": "PASS", "evidence_direction": "supports",
                  "run_id": "v3_exq_904_arc070_decomposition_trigger_selectivity_20260808T201150Z_v3"},
     "hf_step_budget": {True: 24, False: 24}, "cells": {True: 4, False: 16},
     "unit": "episode", "dv_phases": None,
     "intended_units": {True: 4 * 6, False: 16 * 20}},    # cells x N_EPISODES
    {"key": "888", "queue_id": "V3-EXQ-888", "run": _run_888,
     "script": "experiments/v3_exq_888_mech074_readwrite_head_route_dissociation.py",
     "direct_claims": ["MECH-074", "MECH-074a", "MECH-074b"], "family_claims": ["MECH-074d"],
     "original": {"outcome": "PASS", "evidence_direction": "supports",
                  "run_id": "v3_exq_888_mech074_readwrite_head_route_dissociation_20260804T075257Z_v3"},
     "hf_step_budget": {True: 20, False: 60}, "cells": {True: 4, False: 12},
     "unit": "episode", "dv_phases": None,
     "intended_units": {True: 4 * 4, False: 12 * 20}},    # cells x neutral half of collect eps
]


# ======================================================================================
# Summaries
# ======================================================================================
def _dv_units(hf: List[Dict[str, Any]], unit: str, dv_phases: Any) -> List[bool]:
    """One bool per DV-window unit: did it die (health_depleted)?"""
    if unit == "episode":
        return [e["cause"] == "health_depleted" for e in hf]
    walks: Dict[int, bool] = {}
    for e in hf:
        if e["phase"] in (dv_phases or ()):
            walks[e["env_id"]] = walks.get(e["env_id"], False) or e["cause"] == "health_depleted"
    return [walks[k] for k in sorted(walks)]


def _episode_summary(eps: List[Dict[str, Any]], budget: Optional[int],
                     unit: str = "episode", dv_phases: Any = None) -> Dict[str, Any]:
    hf = [e for e in eps if e["hf"]]
    units = _dv_units(hf, unit, dv_phases)
    phases = sorted({e["phase"] for e in hf})
    by_phase = {}
    for ph in phases:
        pe = [e for e in hf if e["phase"] == ph]
        by_phase[ph] = {"n_episodes": len(pe),
                        "n_deaths": sum(1 for e in pe if e["cause"] == "health_depleted"),
                        "steps_total": sum(e["steps"] for e in pe),
                        "contacts_total": sum(e["contacts"] for e in pe)}
    hz = [e for e in eps if not e["hf"]]
    n = len(hf)
    deaths = sum(1 for e in hf if e["cause"] == "health_depleted")
    hf_steps = sum(e["steps"] for e in hf)
    contacts = sum(e["contacts"] for e in hf)
    out: Dict[str, Any] = {
        "n_hf_episodes": n,
        "n_hf_deaths": deaths,
        "hf_death_frac": (deaths / n) if n else None,
        "hf_steps_total": hf_steps,
        "hf_contacts_total": contacts,
        "hf_contacts_per_1k_steps": (1000.0 * contacts / hf_steps) if hf_steps else None,
        "hf_min_health_min": min((e["min_health"] for e in hf), default=None),
        "hf_episode_termination": stats_from_episodes(
            [(e["steps"], "" if e["cause"] == "driver_budget" else e["cause"]) for e in hf],
            budget),
        "n_hazarded_episodes": len(hz),
        "hazarded_death_frac": (
            sum(1 for e in hz if e["cause"] == "health_depleted") / len(hz)) if hz else None,
        "hf_step_budget": budget,
        "dv_unit": unit,
        "dv_phases": list(dv_phases) if dv_phases else None,
        "n_dv_units": len(units),
        "n_dv_units_died": sum(1 for u in units if u),
        "dv_death_frac": (sum(1 for u in units if u) / len(units)) if units else None,
        "hf_by_phase": by_phase,
    }
    if budget:
        out["hf_window_loss"] = (
            1.0 - sum(min(e["steps"], budget) for e in hf) / float(n * budget)) if n else None
        out["hf_window_loss_applicable"] = True
    else:
        out["hf_window_loss"] = None
        out["hf_window_loss_applicable"] = False   # reset-and-continue driver (939a)
    return out


def _census_spreads(census: Dict[str, int], hf: bool) -> List[float]:
    vals = []
    for sig in census:
        d = json.loads(sig)
        if (d["num_hazards"] == 0) == hf:
            vals.append(float(d["contamination_spread"]))
    return vals


def _verdict_key(v: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "outcome": v.get("outcome"),
        "evidence_direction": v.get("evidence_direction"),
        "evidence_direction_per_claim": dict(sorted(
            (v.get("evidence_direction_per_claim") or {}).items())),
        "label": v.get("label"),   # red-team F1: a label-only change is a verdict change
    }


def _positive_control() -> Dict[str, Any]:
    """Random walker on 669c's nursery geometry. Same statistic as the load-bearing criterion
    (fraction of hazard-free episodes ending health_depleted)."""
    res: Dict[str, Any] = {}
    for arm, gate in ((ARM_STOCK, False), (ARM_OPTOUT, True)):
        cell = f"control::{arm}"
        with _cell(cell, force_gate=gate):
            rng = random.Random(20260923)
            for ep in range(CONTROL_EPISODES):
                env = CausalGridWorldV2(size=8, num_hazards=0, num_resources=6,
                                        use_proxy_fields=True, seed=ep)
                env.reset()
                for _ in range(CONTROL_STEPS):
                    _, _, done, _, _ = env.step(rng.randint(0, 3))
                    if done:
                        break
        res[arm] = _episode_summary(_STATE.episodes[cell], CONTROL_STEPS, "episode", None)
    return res


def _target_hash(rel: str) -> str:
    p = Path(__file__).resolve().parents[1] / rel
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ======================================================================================
# Main
# ======================================================================================
def run(dry: bool) -> Dict[str, Any]:
    _install_instrument()
    assert _cgw.CausalGridWorld.step is _probe_step, "instrument not installed"

    print("[P0] positive control: random walker, 669c nursery geometry", flush=True)
    control = _positive_control()
    c_stock = control[ARM_STOCK]["dv_death_frac"]
    c_gated = control[ARM_OPTOUT]["dv_death_frac"]
    print(f"[P0] control death_frac stock={c_stock} gated={c_gated}", flush=True)

    cells: List[Dict[str, Any]] = []
    for t in TARGETS:
        per_arm: Dict[str, Any] = {}
        for arm in ARMS:
            label = f"{t['key']}__{arm}"
            print(f"Seed 0 Condition {label}", flush=True)
            cell = f"{t['key']}::{arm}"
            t0 = time.perf_counter()
            sink = _TargetStdout(sys.stdout, label, t["cells"][dry])
            with _cell(cell, force_gate=(arm == ARM_OPTOUT)):
                with contextlib.redirect_stdout(sink):
                    verdict = t["run"](dry)
            elapsed = time.perf_counter() - t0
            summ = _episode_summary(_STATE.episodes[cell], t["hf_step_budget"][dry],
                                    t["unit"], t["dv_phases"])
            # z_goal liveness: the agents are built INSIDE the target drivers, so the probe
            # records each target's OWN accumulator (cumulative over the arms run so far),
            # not a top-level block it has no agent handle for.
            zg = getattr(sys.modules.get(f"experiments.{Path(t['script']).stem}"), "_ZG", None)
            try:
                zg_stats = zg.stats() if zg is not None else None
            except Exception as exc:   # recording nicety -- never kill a multi-hour run for it
                zg_stats = {"_error": str(exc)[:200]}
            per_arm[arm] = {
                "verdict": verdict,
                "episodes": summ,
                "env_census": _STATE.census.get(cell, {}),
                "post_done_steps": _STATE.post_done_steps.get(cell, 0),
                "elapsed_seconds": elapsed,
                "target_stdout_tail": list(sink.tail)[-25:],
                "z_goal_stream_cumulative": zg_stats,
            }
            print(f"  [train] probe {label} ep {PROGRESS_DENOM}/{PROGRESS_DENOM} "
                  f"hf_eps={summ['n_hf_episodes']} deaths={summ['n_hf_deaths']} "
                  f"dv_units={summ['n_dv_units']} dv_died={summ['n_dv_units_died']} "
                  f"outcome={verdict['outcome']} dir={verdict['evidence_direction']} "
                  f"({elapsed:.0f}s)", flush=True)
            print(f"verdict: {verdict['outcome']}", flush=True)
        cells.append(_classify(t, per_arm, t["intended_units"][dry]))
    return _assemble(cells, control, dry)


def _met(p: Dict[str, Any]) -> bool:
    m, th = p["measured"], p["threshold"]
    if m is None:
        return False
    return bool(m <= th) if p["direction"] == "upper" else bool(m >= th)


_DEGENERATE_DIRECTIONS = ("unknown", "non_contributory")
_DEGENERATE_LABEL_MARKERS = ("not_ready", "requeue", "starved")


def _degenerate(v: Dict[str, Any]) -> bool:
    """Red-team F1: the target's OWN readiness route fired -- it measured nothing."""
    if str(v.get("evidence_direction") or "") in _DEGENERATE_DIRECTIONS:
        return True
    label = str(v.get("label") or "")
    return any(mk in label for mk in _DEGENERATE_LABEL_MARKERS)


def _classify(t: Dict[str, Any], per_arm: Dict[str, Any], intended: int) -> Dict[str, Any]:
    s, o = per_arm[ARM_STOCK], per_arm[ARM_OPTOUT]
    k = t["key"]
    opt_hf_spreads = _census_spreads(o["env_census"], hf=True)
    stock_hz = sorted(set(_census_spreads(s["env_census"], hf=False)))
    opt_hz = sorted(set(_census_spreads(o["env_census"], hf=False)))
    n_s, n_o = s["episodes"]["n_dv_units"], o["episodes"]["n_dv_units"]
    pre = [
        {"name": f"{k}::optout_hf_spread_zero",
         "description": "ARM_OPTOUT: every hazard-free env had contamination_spread 0 (gate took)",
         "measured": max(opt_hf_spreads) if opt_hf_spreads else None,
         "threshold": 0.0, "direction": "upper", "kind": "manipulation"},
        {"name": f"{k}::optout_hf_contacts_zero",
         "description": "ARM_OPTOUT: zero contaminated-cell contacts in hazard-free envs",
         "measured": float(o["episodes"]["hf_contacts_total"]),
         "threshold": 0.0, "direction": "upper", "kind": "manipulation"},
        # Red-team F8: denominated on the INTENDED n, and equal across arms (a death ends a DV
        # unit, it never adds one), so a silently dropped episode fails loud.
        {"name": f"{k}::dv_units_match_intended",
         "description": ("Both arms recorded exactly the intended number of DV-window units "
                         f"({intended}); |n_stock - intended| + |n_optout - intended|"),
         "measured": float(abs(n_s - intended) + abs(n_o - intended)),
         "threshold": 0.0, "direction": "upper", "kind": "readiness",
         "control": "intended count derived from the target's own loop bounds"},
    ]
    if stock_hz or opt_hz:
        pre.append({
            "name": f"{k}::hazarded_env_spread_untouched",
            "description": "Gate left num_hazards>0 envs at their stock contamination_spread",
            "measured": float(0.0 if stock_hz == opt_hz else 1.0),
            "threshold": 0.0, "direction": "upper", "kind": "manipulation"})
    for p in pre:
        p["met"] = _met(p)
    instrument_ok = all(p["met"] for p in pre)

    deg_s, deg_o = _degenerate(s["verdict"]), _degenerate(o["verdict"])
    both_degenerate = deg_s and deg_o
    determinable = instrument_ok and not both_degenerate

    death = s["episodes"]["dv_death_frac"]
    stock_contacts = int(s["episodes"]["hf_contacts_total"])
    changed = _verdict_key(s["verdict"]) != _verdict_key(o["verdict"])
    material = (death is not None) and death >= MATERIAL_DEATH_FRAC
    if not determinable:
        cls = "cannot_determine"
    elif changed:
        if material:
            cls = "verdict_sensitive_truncation"
        elif stock_contacts == 0:
            cls = "verdict_sensitive_contact_free"
        else:
            cls = "verdict_sensitive_observation"
    else:
        cls = "truncated_verdict_robust" if material else "clean"
    orig = t["original"]
    stock_repro = (s["verdict"]["outcome"] == orig["outcome"]
                   and s["verdict"]["evidence_direction"] == orig["evidence_direction"])
    print(f"[classify] {k}: {cls} (stock dv_death_frac={death}, verdict_changed={changed}, "
          f"degenerate stock/optout={deg_s}/{deg_o}, "
          f"stock_reproduces_original={stock_repro})", flush=True)
    return {
        "target": k, "queue_id": t["queue_id"], "script": t["script"],
        "script_sha256": _target_hash(t["script"]),
        "direct_claims": t["direct_claims"], "family_claims": t["family_claims"],
        "original": orig, "per_arm": per_arm, "preconditions": pre,
        "instrument_ok": instrument_ok,
        "stock_verdict_degenerate": deg_s, "optout_verdict_degenerate": deg_o,
        "determinable": determinable, "classification": cls,
        "dv_unit": t["unit"], "intended_dv_units": intended,
        "stock_dv_death_frac": death,
        "optout_dv_death_frac": o["episodes"]["dv_death_frac"],
        "stock_hf_death_frac_all_phases": s["episodes"]["hf_death_frac"],
        "stock_hf_window_loss": s["episodes"]["hf_window_loss"],
        "stock_hf_contacts_total": stock_contacts,
        "verdict_changed": changed, "materially_truncated": material,
        "stock_reproduces_original": stock_repro,
        "stock_verdict": _verdict_key(s["verdict"]), "optout_verdict": _verdict_key(o["verdict"]),
    }


def _assemble(cells: List[Dict[str, Any]], control: Dict[str, Any], dry: bool) -> Dict[str, Any]:
    c_stock = control[ARM_STOCK]["dv_death_frac"]
    c_gated = control[ARM_OPTOUT]["dv_death_frac"]
    c_n = [control[a]["n_dv_units"] for a in ARMS]
    control_pre = [
        {"name": "control::stock_death_frac_supra_floor",
         "description": "Random walker on the 669c nursery dies of contamination at stock settings",
         "measured": c_stock, "threshold": CONTROL_STOCK_DEATH_FLOOR, "direction": "lower",
         "kind": "readiness",
         "control": "random walk, 20x100 steps, size-8 0-hazard V2 env; 0.95 measured at design time"},
        {"name": "control::gated_death_frac_zero",
         "description": "The same walker with hazard_free_contamination_gate=True never dies",
         "measured": c_gated, "threshold": CONTROL_GATED_DEATH_CEIL, "direction": "upper",
         "kind": "readiness", "control": "same walker, gate on"},
        {"name": "control::episodes_match_intended",
         "description": f"Both control arms recorded exactly {CONTROL_EPISODES} episodes",
         "measured": float(sum(abs(n - CONTROL_EPISODES) for n in c_n)),
         "threshold": 0.0, "direction": "upper", "kind": "readiness",
         "control": "loop bound CONTROL_EPISODES"},
    ]
    for p in control_pre:
        p["met"] = _met(p)
    control_ok = all(p["met"] for p in control_pre)

    det = [c for c in cells if c["determinable"]]
    undet = [c for c in cells if not c["determinable"]]
    sens = [c for c in det if c["classification"].startswith("verdict_sensitive")]
    trunc = [c for c in det if c["materially_truncated"]]
    ready = control_ok and len(det) >= MIN_DETERMINABLE_TARGETS
    n_sens = len(sens)

    family_unmeasured = sorted({x for c in det for x in c["family_claims"]})
    if not ready:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        reruns: List[str] = []
    elif n_sens == 0:
        label = ("contamination_prevalence_low_no_reruns_owed" if not trunc else
                 "contamination_truncation_present_verdicts_robust_no_reruns_owed")
        outcome = "PASS"
        reruns = []
    elif n_sens == 1:
        label = "contamination_isolated_sensitivity_reruns_owed"
        outcome = "FAIL"
        reruns = sorted(set(sens[0]["direct_claims"]))
    else:
        label = "contamination_prevalence_high_all_reruns_owed"
        outcome = "FAIL"
        reruns = list(AUDITED_CLAIM_IDS)

    # Red-team F3/F6: only determinable targets' DIRECT claims can be cleared.
    measured_direct = sorted({x for c in det for x in c["direct_claims"]})
    cleared_direct = ([] if (not ready or n_sens >= 2) else
                      sorted({x for c in det if c not in sens for x in c["direct_claims"]}))
    not_covered = dict(UNCOVERED_CLAIMS)
    for c in undet:
        for x in c["direct_claims"] + c["family_claims"]:
            not_covered[x] = f"{c['queue_id']} could not be determined ({c['classification']})"
    family_recommended = (sorted({x for c in sens for x in c["family_claims"]})
                          if ready and n_sens == 1 else [])
    # Red-team F4: a STOCK that does not reproduce history is a substrate-drift finding.
    not_reproduced = [{"queue_id": c["queue_id"], "claims": c["direct_claims"],
                       "original": {"outcome": c["original"]["outcome"],
                                    "evidence_direction": c["original"]["evidence_direction"]},
                       "stock_today": {"outcome": c["stock_verdict"]["outcome"],
                                       "evidence_direction": c["stock_verdict"]["evidence_direction"]}}
                      for c in det if not c["stock_reproduces_original"]]

    per_target_gate = {c["target"]: {"green": c["determinable"],
                                    "instrument_ok": c["instrument_ok"],
                                    "both_verdicts_degenerate": (c["stock_verdict_degenerate"]
                                                                 and c["optout_verdict_degenerate"]),
                                    "failed": [p["name"] for p in c["preconditions"] if not p["met"]]}
                       for c in cells}
    adjudication_pre = list(control_pre) + [p for c in det for p in c["preconditions"]]
    all_pre = list(control_pre) + [p for c in cells for p in c["preconditions"]]

    c_prev = {
        "name": "C_PREV_no_verdict_sensitive_target",
        "load_bearing": True,
        "threshold": 0.0, "comparator": "<=",
        "passed": bool(ready and n_sens == 0),
        "description": "No determinable target's own verdict changes when the footgun is gated",
    }
    if ready:
        c_prev["measured"] = float(n_sens)
    else:   # no measured value -> nothing can recompute it as met while passed is False
        c_prev["threshold_not_applicable"] = "run not ready; C_PREV not evaluated"
    criteria = [c_prev]
    for c in cells:
        crit = {
            "name": f"{c['target']}::stock_dv_death_frac_below_material",
            "load_bearing": False,
            "threshold": MATERIAL_DEATH_FRAC, "comparator": "<",
            "passed": (not c["materially_truncated"]),
        }
        if c["stock_dv_death_frac"] is not None:
            crit["measured"] = c["stock_dv_death_frac"]
        criteria.append(crit)
        criteria.append({
            "name": f"{c['target']}::verdict_unchanged_stock_vs_optout",
            "load_bearing": False, "passed": (not c["verdict_changed"]),
            "threshold_not_applicable": "categorical equality of the target's own verdict tuple",
        })
    non_degen = {"C_PREV_no_verdict_sensitive_target": bool(len(det) >= MIN_DETERMINABLE_TARGETS)}
    for c in cells:
        non_degen[f"{c['target']}::stock_dv_death_frac_below_material"] = c["determinable"]
        non_degen[f"{c['target']}::verdict_unchanged_stock_vs_optout"] = c["determinable"]

    readout: Dict[str, Any] = {
        "ready": int(ready), "control_ok": int(control_ok),
        "n_targets_determinable": len(det), "n_verdict_sensitive": n_sens,
        "n_materially_truncated": len(trunc),
        "n_historical_not_reproduced": len(not_reproduced),
        "material_death_frac_threshold": MATERIAL_DEATH_FRAC,
    }
    if c_stock is not None:
        readout["control_stock_dv_death_frac"] = c_stock
    if c_gated is not None:
        readout["control_gated_dv_death_frac"] = c_gated
    for c in cells:
        k = c["target"]
        wp = c["per_arm"][ARM_STOCK]["episodes"]["hf_by_phase"].get("warmup")
        warm = (wp["n_deaths"] / wp["n_episodes"]) if wp and wp["n_episodes"] else None
        for name, val in (("stock_dv_death_frac", c["stock_dv_death_frac"]),
                          ("optout_dv_death_frac", c["optout_dv_death_frac"]),
                          ("stock_hf_death_frac_all_phases", c["stock_hf_death_frac_all_phases"]),
                          ("stock_warmup_death_frac", warm),
                          ("stock_hf_window_loss", c["stock_hf_window_loss"]),
                          ("stock_hf_contacts_per_1k_steps",
                           c["per_arm"][ARM_STOCK]["episodes"]["hf_contacts_per_1k_steps"])):
            if val is not None and np.isfinite(float(val)):
                readout[f"t{k}_{name}"] = float(val)
        readout[f"t{k}_determinable"] = int(c["determinable"])
        readout[f"t{k}_verdict_changed"] = int(c["verdict_changed"])
        readout[f"t{k}_materially_truncated"] = int(c["materially_truncated"])
        readout[f"t{k}_stock_reproduces_original"] = int(c["stock_reproduces_original"])
        readout[f"t{k}_stock_outcome_pass"] = int(c["stock_verdict"]["outcome"] == "PASS")
        readout[f"t{k}_optout_outcome_pass"] = int(c["optout_verdict"]["outcome"] == "PASS")

    all_term = EpisodeTerminationAccumulator(steps_configured=None)
    for c in cells:
        for e in _STATE.episodes.get(f"{c['target']}::{ARM_STOCK}", []):
            if e["hf"]:
                all_term.record(e["steps"], "" if e["cause"] == "driver_budget" else e["cause"])

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "audited_claim_ids": AUDITED_CLAIM_IDS,
        "source_flag": SOURCE_FLAG,
        "source_audit": SOURCE_AUDIT,
        "outcome": outcome,
        "evidence_direction": "non_contributory",
        "timestamp_utc": ts,
        "dry_run": bool(dry),
        "readout": readout,
        "criteria": criteria,
        "combination_rule": ("PASS iff ready (positive control met AND >= "
                             f"{MIN_DETERMINABLE_TARGETS} determinable targets) AND "
                             "C_PREV (zero verdict-sensitive targets). Per-target "
                             "dv_death_frac / verdict criteria are reported, not combined."),
        "per_arm_gate": per_target_gate,
        "interpretation": {
            "label": label,
            "preconditions": adjudication_pre,
            "criteria_non_degenerate": non_degen,
            "reruns_owed_for_claims": reruns,
            "family_reruns_recommended_unmeasured": family_recommended,
            "direct_claims_measured": measured_direct,
            "direct_claims_cleared_no_rerun_owed": cleared_direct,
            "family_claims_not_cleared_caveat_stands": family_unmeasured,
            "claims_not_covered": not_covered,
            "historical_verdict_not_reproduced": not_reproduced,
            "fidelity_caveat": ("targets re-run on the CURRENT substrate; a determinable target "
                                "whose STOCK verdict differs from its historical manifest is "
                                "listed in historical_verdict_not_reproduced for /governance"),
        },
        "preconditions_all": all_pre,
        "non_degenerate": bool(ready),
        "degeneracy_reason": (None if ready else
                              "positive control failed or fewer than "
                              f"{MIN_DETERMINABLE_TARGETS} determinable targets"),
        "positive_control": control,
        "probe_cells": cells,
        "episode_termination_stock_hf_all_targets": all_term.stats(),
        "custom_information": {
            "instrument": "class-level wrap of CausalGridWorld.__init__/step/reset",
            "gate": "hazard_free_contamination_gate=True forced on every env in ARM_OPTOUT",
            "pairing": "each target cell enters arm_cell (full RNG reset); arms differ only in the gate",
            "z_goal_stream": ("recorded per target in probe_cells[].per_arm[].z_goal_stream_cumulative "
                              "(each target's own accumulator, cumulative across its arms); no "
                              "top-level block because the probe holds no agent handle"),
            "phase_tags": "939a only: _p0_warmup / _expose / _test_release wrapped as warmup / exposure / test",
        },
    }


def main() -> Any:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()
    manifest = run(args.dry_run)
    config = {
        "targets": [
            {**{k: v for k, v in t.items()
                if k not in ("run", "hf_step_budget", "cells", "intended_units")},
             "hf_step_budget": t["hf_step_budget"][args.dry_run],
             "intended_dv_units": t["intended_units"][args.dry_run],
             "n_target_cells": t["cells"][args.dry_run]}
            for t in TARGETS
        ],
        "arms": list(ARMS),
        "material_death_frac": MATERIAL_DEATH_FRAC,
        "min_determinable_targets": MIN_DETERMINABLE_TARGETS,
        "control": {"episodes": CONTROL_EPISODES, "steps": CONTROL_STEPS,
                    "stock_floor": CONTROL_STOCK_DEATH_FLOOR,
                    "gated_ceiling": CONTROL_GATED_DEATH_CEIL},
    }
    out_path = write_flat_manifest(
        manifest,
        Path(__file__).resolve().parents[2] / "REE_assembly" / "evidence" / "experiments",
        dry_run=args.dry_run,
        config=config,
        seeds=[0],
        script_path=Path(__file__),
        started_at=t0,
        episode_termination=manifest["episode_termination_stock_hf_all_targets"],
    )
    r = manifest["readout"]
    print(f"outcome: {manifest['outcome']}", flush=True)
    print(f"label: {manifest['interpretation']['label']}", flush=True)
    print(f"determinable={r['n_targets_determinable']} sensitive={r['n_verdict_sensitive']} "
          f"truncated={r['n_materially_truncated']} control_ok={r['control_ok']}", flush=True)
    for c in manifest["probe_cells"]:
        print(f"  {c['target']}: {c['classification']} stock_dv_death={c['stock_dv_death_frac']} "
              f"optout_dv_death={c['optout_dv_death_frac']} "
              f"stock={c['stock_verdict']['outcome']}/{c['stock_verdict']['evidence_direction']} "
              f"optout={c['optout_verdict']['outcome']}/{c['optout_verdict']['evidence_direction']} "
              f"repro_orig={c['stock_reproduces_original']}", flush=True)
    print(f"wrote: {out_path}", flush=True)
    return manifest, out_path, args.dry_run


if __name__ == "__main__":
    _manifest, _out_path, _dry_run = main()
    _outcome_raw = str(_manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(_out_path),
        dry_run=_dry_run,
    )
