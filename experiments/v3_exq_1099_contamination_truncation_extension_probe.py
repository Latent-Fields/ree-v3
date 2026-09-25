"""V3-EXQ-1099 -- contamination-footgun truncation EXTENSION probe. Diagnostic, claim-free.

*** NOT QUEUED. DO NOT ADD A QUEUE ENTRY FOR THIS SCRIPT WITHOUT READING THE FILE BELOW. ***
The Step 4.5 adversarial design review (Fable, 2026-09-25) returned BLOCKING, and the two decisive
findings were re-verified from source by the authoring session. In short: V3-EXQ-883's contamination
exposure lands entirely on its NO_ATTAINMENT arm, whose DV (`parent_goal_norm_final`) is exactly
0.0 in BOTH arms by construction, so both of its pre-registered criteria are undiscriminating while
the taxonomy would label the result `truncated_verdict_robust` -- i.e. a PASSED sensitivity test --
and clear MECH-427 on it. Separately, V3-EXQ-435's baseline `evidence_direction` in the landed
manifest is a hand-applied 2026-04-22 reclassification rather than the driver's own output, so
`stock_reproduces_original` is False for 435 by construction. Resolving either changes what gets
measured, which is a user decision, so the design was left as authored and the refusal recorded:
  REE_assembly/evidence/planning/v3_exq_1099_contamination_extension_redteam_blocking_staged_20260925.md

Same-pattern extension of V3-EXQ-1080, chipped by /governance 2026-09-24 once it ratified
`failure_autopsy_V3-EXQ-1080_2026-09-24` (section 7, "Optional follow-on"). 1080 cleared the
contamination caveat for 7 DIRECT claims by re-running 4 exposed targets with and without
`hazard_free_contamination_gate`; three claims it explicitly did NOT cover stay undecided.
This probe covers those three.

QUESTION. `contamination_spread` defaults to 0.5 on EVERY entered cell regardless of
`num_hazards` (SD-094 footgun, causal_grid_world.py docstring L70-86), so a nominally
hazard-free measurement window can be truncated by the agent poisoning its own cells. For
INV-054, MECH-427 and MECH-106: how often does the footgun actually fire in the window that
carries each claim's DV, and does gating it change that run's OWN verdict?

DESIGN. Identical to 1080's: for each target, run the target driver's OWN loop twice at its
original full-scale configuration, under a class-level instrument on `CausalGridWorld`:
  ARM_STOCK   the target exactly as written -- the reproduction arm; this measures prevalence.
  ARM_OPTOUT  identical, except every env constructed during the run gets
              `hazard_free_contamination_gate=True`. That zeroes contamination_spread ONLY
              when num_hazards == 0, so 278/435's 3-hazard LONG_HORIZON phase and 231a's
              2-hazard standard / 5-hazard hard envs are untouched -- the manipulation is
              scoped to exactly the footgun. Verified at runtime, this session:
              V2 hf stock spread=0.5 applied=False / V2 hf gated spread=0.0 applied=True /
              V2 3-hazard gated spread=0.5 applied=False / V1 883-shape gated applied=True.
The arms are PAIRED: every cell resets all RNG on entry, so STOCK and OPTOUT start from
bit-identical state and differ only in the gate.

TARGETS (all four named by autopsy section 7; the "optional" 894* / 807-823 family extras are
deliberately NOT included -- see SCOPE below):
  V3-EXQ-278  INV-054. Phase 2 LOW_HARM (num_hazards=0, 300 eps x 150 steps x 3 seeds) is the
              recovery window the DV (recovery_latency) is read in. Historical: FAIL /
              does_not_support. The autopsy's named highest-risk target: long episodes, negative
              verdict, hazard-free DV window.
  V3-EXQ-435  INV-054. Same LOW_HARM phase 2 (300 eps x 150 steps x 3 seeds); DV is sustained
              recovery_onset. Historical: FAIL / non_contributory -- which is itself the verdict
              a truncated measurement window produces, so this target is the sharpest test in
              the set. Expected a priori to be DEGENERATE in ARM_STOCK (non_contributory is in
              _DEGENERATE_DIRECTIONS); under the classifier below, exactly-one-arm-degenerate
              counts as a gate-caused CHANGE, so "gating rescues 435 from non_contributory" is
              a live, falsifiable outcome rather than a hole in the design.
  V3-EXQ-883  MECH-427. size-10 V1 env, num_hazards=0, num_resources=0, subgoal_mode, 40 steps.
              This is V3-EXQ-884's constructor, which the causal_grid_world docstring records
              dying at 32/19/90 of 400 configured steps from self-poisoning -- so 40 steps is
              close to the measured kill horizon, not comfortably under it. Historical:
              PASS / supports.
  V3-EXQ-231a MECH-106. The hazard-free env is `_make_env_easy` (num_hazards=0, 40 eps x 200
              steps x 5 seeds), used ONLY for the POSITIVE_HISTORY phase -- i.e. exactly the
              phase that builds the positive valence whose asymmetry against NEGATIVE_HISTORY
              is MECH-106's DV. Probes run in the 2-hazard standard env and are untouched.
              Historical: PASS / supports.

PREMISE CORRECTION (audited against the landed manifests this session, before any code). The
dispatching brief described all three targets as carrying "the negative (does_not_support)
results". Measured: 278 does_not_support, 435 non_contributory, 883 PASS/supports,
231a PASS/supports. Only ONE of the four is does_not_support. The probe's question is unchanged
(a PASS can be gate-sensitive in either direction, and the classifier tests verdict EQUALITY,
not verdict sign), but no reader should take "these were negative results" from this run.

SCOPE -- the optional family extras are DROPPED, deliberately. Autopsy section 7 offered "one
894* target and one 807/823 target" to close the MECH-074d / SD-079 family caveats but named no
selection criterion, and each resolves ambiguously: 894* is four distinct landed runs
(894/894a/894b/894c) and 807 vs 823 are two different SD-079 questions
(centered_goal_anchor_match vs ghost_goal_retrieval_consumer). Picking one would be new design
judgement beyond what governance ratified, so this probe takes the pre-flight's stated safe
option: score only the three uncovered claims. MECH-074d / SD-077 / SD-079's family caveat
STANDS, untouched, exactly as 1080 left it.

THREE IMPROVEMENTS over 1080, each from its confirmed autopsy or the dispatch pre-flight:
  (1) ACTION-CLASS DIVERSITY READOUT, non-gating (pre-flight NAMED CHANGE 2, on open
      GFLAG-0487/0489). GFLAG-0487 reports the V3 action loop collapsing to an 88-99%
      single-action share via untrained compression/readout sites. All three target claims are
      BEHAVIOURAL, so a clean-contamination result on a monostrategy-collapsed agent would not
      be dispositive for the claim. Every `env.step(action)` is counted at the env boundary, per
      cell; `modal_action_share >= MONOSTRATEGY_MODAL_SHARE` (0.88, the conservative end of
      GFLAG-0487's measured range) sets `monostrategy_suspect` on that target. It is REPORT-ONLY:
      it enters no criterion, no precondition, no readiness gate and cannot move `outcome`. For
      883 it is recorded but flagged `action_diversity_interpretable: false` -- 883 drives a
      SCRIPTED action sequence (`_scripted_action`), so a high modal share there is the design,
      not collapse. The full per-cell action histogram is recorded so a reader can recompute the
      share against its own denominator rather than trusting this one number.
      The class is read from the env's OWN resolution (`CausalGridWorld._last_action`), never
      re-derived from the driver's `step()` argument: the four targets pass three different
      types (883 an int, 278/435 a one-hot tensor, 231a a [1, N] continuous candidate vector),
      and this probe's first draft DID re-derive it and reported 231a as using 45 distinct
      action classes at a 0.068 modal share -- a broken instrument reading as a clean result,
      caught by the dry-run smoke. `action_resolution` is three-valued
      (ready / unresolved / not_measured) and `monostrategy_suspect` is None -- CANNOT
      DETERMINE, never "no collapse" -- whenever it is not `ready`.
  (2) GATE IN THE CONFIG SLICE (autopsy lesson 3a). 1080 injected the gate through the class
      wrap only, so STOCK and OPTOUT cells emitted IDENTICAL arm fingerprints -- harmless while
      arm reuse is emit-only, a false cache hit if it ever becomes consuming. Of these four
      targets only 883 uses `arm_cell`, so its module-level `arm_cell` is wrapped to inject
      `hazard_free_contamination_gate` into the slice. The other three emit no arm fingerprint
      at all, so there is nothing to distinguish -- stated rather than silently skipped.
  (3) GENERAL METRICS ADAPTER (autopsy lesson 3, "read 939a-style criteria[].gap margins").
      1080's adapter read only `man["metrics"]` / `man["readout"]` and captured `{}` for 939a,
      losing its margins. `_target_metrics()` here reads `metrics`, `readout`, `aggregates`,
      `summary_metrics`, `criteria[].gap`/`measured`/`threshold`, and `per_seed_*` scalars. None
      of THESE four targets is 939a-shaped, so this buys robustness, not a specific rescue; it
      is recorded that way.

INSTRUMENT. `CausalGridWorld.__init__/step/reset` wrapped at class level (CausalGridWorldV2 is a
factory over that class, so every driver is covered). Per env: num_hazards at construction,
effective contamination_spread, gate-applied flag. Per episode: length, cause
(`info["done_cause"]`: health_depleted / step_limit, or `driver_budget` when the driver reset or
abandoned before the env ended it), contaminated-cell contacts
(`transition_type == "agent_caused_hazard"`, which in a hazard-free env can only be a
contaminated cell), min health, and the action histogram. All four info keys confirmed present
at runtime this session.

FIDELITY CAVEAT (load-bearing for interpretation, unchanged from 1080). This re-runs the targets
on the CURRENT substrate, not the one they originally ran on -- 278/435/231a are April-2026 runs.
ARM_STOCK therefore measures what the original CONFIGURATION does today, and
`stock_reproduces_original` records whether it matches the historical manifest. A mismatch is a
substrate-drift finding for /governance (routed to `historical_verdict_not_reproduced`), NOT a
contamination result, and it weakens transfer to the historical run without touching the
within-run STOCK-vs-OPTOUT comparison.

PRE-REGISTERED (constants below, fixed before any real run):
  DV-WINDOW UNIT = one HAZARD-FREE EPISODE for every target (all four are break-on-done
      drivers; none is a 939a-style reset-and-continue walk, so 1080's "walk" unit is not
      needed). A unit "dies" if it ended health_depleted -- each such death IS a truncated
      window.
  MATERIAL_DEATH_FRAC = 0.10 -- a target is MATERIALLY TRUNCATED when >= 10% of its ARM_STOCK
      DV units die. Same threshold as 1080, so the two runs' prevalence figures are comparable.
  VERDICT CHANGE -- the target's own outcome (PASS/FAIL), overall evidence_direction, ANY
      per-claim direction, or its interpretation label differs between the arms.
      NOTE, stated because it bounds what this run can detect: 278, 435 and 231a emit neither
      `evidence_direction_per_claim` nor an `interpretation.label`, so for those three the
      verdict tuple is effectively (outcome, evidence_direction) -- a COARSER instrument than
      1080 had on 669c/888/904/939a. 883 emits per-claim. This is a property of the historical
      drivers, not a choice here, and it is why a verdict-unchanged reading on 278/435/231a is
      weaker evidence than the same reading on 883.
  TARGET DEGENERACY -- a verdict is degenerate when its direction is unknown /
      non_contributory or its label is a readiness route. Both arms degenerate ->
      cannot_determine (it measured nothing; must NOT count as robust). Exactly one arm
      degenerate -> that IS a gate-caused change (sensitive).
  Per-target class: clean / truncated_verdict_robust / verdict_sensitive_truncation /
      verdict_sensitive_observation / verdict_sensitive_contact_free / cannot_determine --
      definitions as in V3-EXQ-1080 (a contact-free flip is labelled separately because the gate
      also zeroes the `contamination_view` observation channel, so such a flip is moved by an
      observation change, not by poisoning).
  Run-level routing on n_sensitive / n_truncated, as 1080:
      not ready                      -> substrate_not_ready_requeue
      n_sensitive 0, n_truncated 0   -> contamination_prevalence_low_no_reruns_owed
      n_sensitive 0, n_truncated >=1 -> contamination_truncation_present_verdicts_robust_no_reruns_owed
      n_sensitive 1                  -> contamination_isolated_sensitivity_reruns_owed
      n_sensitive >= 2               -> contamination_prevalence_high_all_reruns_owed
  READINESS = positive control met AND all MIN_DETERMINABLE_CLAIMS (3) of the mandatory claims
      have at least one DETERMINABLE target. 1080 counted determinable TARGETS, which is the
      same thing when each target carries its own claim -- it is not here, because INV-054 is
      carried by TWO targets (278 and 435) and 435 is expected a priori to be degenerate in
      both arms. Counting targets would let one foreseeably-degenerate target vacate the
      well-powered findings of the other three, which is exactly the whole-run precondition AND
      that V3-EXQ-785 established as a defect. Counting claims is the faithful translation of
      1080's bar to this target set, and it still FAILS whenever any mandatory claim goes
      unmeasured -- it is not a relaxation into unfalsifiability.
      Belt and braces for the same hazard: `interpretation.per_claim_disposition` is populated
      from each claim's OWN determinable targets and is reported WHATEVER the run-level
      readiness verdict, so one target's degeneracy can never erase another's measured result.
      It is a record, not a route: run-level `outcome`, `label` and `reruns_owed_for_claims`
      stay exactly on 1080's pre-registered routing.
  What "no re-runs owed" covers: ONLY the determinable targets' DIRECT claims. INV-054 is
      covered by TWO targets (278 and 435); it is cleared only if every determinable one of
      them is non-sensitive, and if BOTH are undeterminable it moves to claims_not_covered.
  Every target and the control must record exactly its INTENDED number of DV units in BOTH
      arms, so a silently dropped episode fails loud. The intended counts are the drivers' own
      unconditional loop bounds, read this session: 278 phase-2 `for ep in range(phase2_eps)`
      with no break (3 x 300); 435 the same (3 x 300; its phase-2 `break`s are in the inner
      recent-above and onset-search loops, not the episode loop); 231a POSITIVE_HISTORY
      `for ep in range(n_pos)` (5 x 40); 883 one episode per (seed, arm) cell (3 x 2 x 1).
  outcome PASS iff ready and n_sensitive == 0; FAIL otherwise.

POSITIVE CONTROL (P0, same statistic as the load-bearing criterion). A random walker on
278/435's EXACT LOW_HARM geometry -- the highest-risk target's own DV window -- 20 episodes x
150 steps. At stock contamination the death fraction must be >= 0.5; with the gate it must be 0.
MEASURED while designing this probe (2026-09-25, this box): stock 1.000 (20/20 health_depleted),
gated 0.000 (20/20 ran the full 150 steps). Both rails are reachable by measurement, not by
assumption, and the separation is total. That measurement is itself the first finding: a random
policy dies of self-poisoning in EVERY episode of the window in which INV-054's recovery latency
is read.

DV-SYMMETRY (Step 3.5). DV = fraction of hazard-free episodes ending health_depleted; its
symmetry group is permutation of episodes. Zeroing contamination deposition is not a permutation
of episodes -- it changes WHICH episodes die -- so the DV is not invariant under it. The verdict
DV is each target's own categorical verdict; the gate is not invariant under that either, since
it changes the target's inputs. The action-diversity readout is a report, not a DV, and is
excluded from this declaration on purpose.

red-team (fable, 2026-09-25): BLOCKING -- see the NOT QUEUED banner at the top of this
docstring and the staged findings file it names. Findings F1 (435 baseline is a reclassification)
and F2 (883's exposed arm has a structurally pinned DV) were both re-verified from source by the
authoring session; F4/F5 (278's DV at its floor, and the gate inert in the phase-1 env that
actually sets INV-054's verdict) were measured by the reviewer and not independently re-measured.
No criterion was relaxed and no target dropped in response -- every available fix changes what gets
measured, so the design is recorded as-authored pending the user's decision.
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

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment import causal_grid_world as _cgw  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorld, CausalGridWorldV2  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.episode_termination import (  # noqa: E402
    EpisodeTerminationAccumulator, stats_from_episodes,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_1099_contamination_truncation_extension_probe"
QUEUE_ID = "V3-EXQ-1099"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS: List[str] = []   # claim-free: this audits the evidence base, it tests no claim
AUDITED_CLAIM_IDS = ["INV-054", "MECH-106", "MECH-427"]
SOURCE_FLAG = "GFLAG-0304"
SOURCE_AUTOPSY = "REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-1080_2026-09-24.md"
SOURCE_AUDIT = "REE_assembly/evidence/planning/corpus_audit_contamination_footgun_20260916.md"
PREDECESSOR_RUN = "v3_exq_1080_contamination_truncation_prevalence_probe_20260924T000105Z_v3"
# The control predicate IS the statistic the load-bearing criterion routes on (fraction of
# hazard-free episodes ending health_depleted), and BOTH rails were measured reachable on
# 278/435's own LOW_HARM geometry on 2026-09-25 (stock 1.000, gated 0.000; 20 x 150 random
# walk), so it is reachable by measurement rather than by construction.
ANCHOR_REACHABILITY_EXEMPT = ("control predicate is the death-fraction statistic itself; both "
                              "rails measured reachable on the target geometry "
                              "(1.000 stock / 0.000 gated, 2026-09-25)")

# ---- pre-registered constants (never derived from this run's own statistics) ----
MATERIAL_DEATH_FRAC = 0.10
# Readiness counts mandatory CLAIMS covered by >= 1 determinable target, not raw targets --
# INV-054 is carried by two targets and 435 is expected degenerate. See the READINESS note in
# the module docstring for why this is 1080's bar translated, not relaxed.
MIN_DETERMINABLE_CLAIMS = 3
CONTROL_EPISODES = 20
CONTROL_STEPS = 150
CONTROL_STOCK_DEATH_FLOOR = 0.5
CONTROL_GATED_DEATH_CEIL = 0.0
# Report-only. Lower end of GFLAG-0487's measured 88-99% single-action-share range.
MONOSTRATEGY_MODAL_SHARE = 0.88

ARM_STOCK = "ARM_STOCK"
ARM_OPTOUT = "ARM_OPTOUT"
ARMS = (ARM_STOCK, ARM_OPTOUT)

PROGRESS_DENOM = 100            # [train] ep N/100 per (target, arm) run == queue episodes_per_run

# Claims this probe does NOT reach, stated rather than extrapolated silently.
UNCOVERED_CLAIMS = {
    "MECH-074d": "V3-EXQ-894/894a/894b/894c -- family caveat left standing (see SCOPE)",
    "SD-077": "V3-EXQ-1040 -- family caveat left standing (1080)",
    "SD-079": "V3-EXQ-807/823 (32-step episodes) -- family caveat left standing (see SCOPE)",
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
        self.actions: Dict[str, Dict[str, int]] = {}   # cell -> {action_str: count}
        # STRONG refs, held for one (target, arm) run: a target drops its env when its cell
        # function returns, and a weak ref would let the last in-flight episode of every cell
        # vanish unrecorded (1080 found this in its dry-run smoke: 9 of 12 669c episodes).
        self.live: List[Any] = []
        self.post_done_steps: Dict[str, int] = {}
        self.phase: str = "all"
        self.env_seq: int = 0
        # Every REEAgent built inside a target driver, per cell. The probe holds no agent
        # handle of its own (each target constructs its agents internally), so a class-level
        # wrap on REEAgent.__init__ is the only way to record z_goal liveness for all four
        # targets rather than only for the one that happens to export an accumulator (883).
        # V3-EXQ-1080 recorded just that one; this closes the gap.
        self.agents: Dict[str, List[Any]] = {}


_STATE = _ProbeState()
_ZG = ZGoalStreamAccumulator()
_ORIG_INIT = CausalGridWorld.__init__
_ORIG_STEP = CausalGridWorld.step
_ORIG_RESET = CausalGridWorld.reset
_ORIG_AGENT_INIT = REEAgent.__init__


def _probe_agent_init(self, *args, **kwargs):
    _ORIG_AGENT_INIT(self, *args, **kwargs)
    if _STATE.cell is not None:
        # Held, not observed, until the cell finishes: ZGoalStreamAccumulator.observe reads
        # the counters AT CALL TIME, and at construction they are all still zero (the one site
        # its docstring warns about).
        _STATE.agents.setdefault(_STATE.cell, []).append(self)


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
    cell = getattr(self, "_probe_cell", None)
    out = _ORIG_STEP(self, action)
    if cell is not None and getattr(self, "_probe_hf", False):
        # NAMED CHANGE 2: action-class histogram over hazard-free ticks (the DV window),
        # counted at the env boundary so it is driver-agnostic. REPORT-ONLY.
        #
        # Read the env's OWN resolved action (`_last_action`), set by CausalGridWorld.step as
        # `action.argmax().item() if action.dim() > 0 else action.item()` then `% action_dim`.
        # Do NOT re-derive the class from the `action` ARGUMENT: the four targets pass three
        # different types -- 883 an int, 278/435 a one-hot tensor, 231a a [1, N] continuous
        # candidate vector -- and the first draft of this probe did re-derive it, fell through
        # to `str(action)[:16]`, and reported 231a as using 45 distinct "action classes" with a
        # 0.068 modal share. That is a broken instrument reading as a clean result, which is
        # exactly the reading the readout exists to prevent, so it is read from the env now.
        # `_ticks_unresolved` keeps "could not resolve" structurally distinct from "diverse".
        la = getattr(self, "_last_action", None)
        h = _STATE.actions.setdefault(cell, {})
        if isinstance(la, (int, np.integer)):
            key = str(int(la))
        else:
            key = "_unresolved"
        h[key] = h.get(key, 0) + 1
    if cell is None:
        return out
    done, info = out[2], out[3]
    if self._probe_closed:
        # The driver kept stepping a finished episode without a reset. Counted, not scored.
        _STATE.post_done_steps[cell] = _STATE.post_done_steps.get(cell, 0) + 1
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
    REEAgent.__init__ = _probe_agent_init


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
    _STATE.actions.setdefault(cell, {})
    try:
        yield
    finally:
        _flush_live()
        for agent in _STATE.agents.get(cell, ()):   # AFTER stepping -- see _probe_agent_init
            try:
                _ZG.observe(agent)
            except Exception:
                pass          # recording nicety -- never kill a multi-hour run for it
        _STATE.agents[cell] = []                   # drop the strong refs
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
        # Targets differ in what they print per unit of progress; count any of the three
        # per-seed/per-phase completion markers the four drivers actually emit.
        s = line.strip()
        if (s.startswith("verdict:") or "[phase2 done]" in s
                or s.startswith("--- seed=") or "seed=" in s and "] POSITIVE_HISTORY" in s):
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
# Metrics adapter (IMPROVEMENT 3) -- read every shape the corpus's drivers emit
# ======================================================================================
def _scalars(d: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, bool):
                out[f"{prefix}{k}"] = int(v)
            elif isinstance(v, (int, float)) and np.isfinite(float(v)):
                out[f"{prefix}{k}"] = float(v)
    return out


def _target_metrics(obj: Any) -> Dict[str, Any]:
    """Harvest every scalar readout the target emitted, across the shapes the corpus uses.

    1080's adapter read only `metrics` / `readout` and therefore captured {} for a driver whose
    DV margins live in `criteria[].gap` (autopsy lesson 3). This reads the flat blocks, the
    per-criterion numbers, and per-seed scalar lists. None of THIS probe's four targets is
    939a-shaped, so the criteria branch is robustness rather than a specific rescue -- recorded
    that way so a later reader does not read a populated block as proof the branch fired.
    """
    if not isinstance(obj, dict):
        return {}
    out: Dict[str, Any] = {}
    for block in ("metrics", "readout", "aggregates", "summary_metrics"):
        out.update(_scalars(obj.get(block)))
    crit = obj.get("criteria")
    if isinstance(crit, list):
        for i, c in enumerate(crit):
            if not isinstance(c, dict):
                continue
            name = str(c.get("name") or f"c{i}")
            for field in ("gap", "measured", "threshold", "value", "margin"):
                v = c.get(field)
                if isinstance(v, bool):
                    out[f"crit_{name}_{field}"] = int(v)
                elif isinstance(v, (int, float)) and np.isfinite(float(v)):
                    out[f"crit_{name}_{field}"] = float(v)
    for key, val in obj.items():
        if not (isinstance(key, str) and key.startswith("per_seed")):
            continue
        if isinstance(val, list):
            nums = [float(x) for x in val
                    if isinstance(x, (int, float)) and not isinstance(x, bool)
                    and np.isfinite(float(x))]
            if nums:
                out[f"{key}_mean"] = float(np.mean(nums))
                out[f"{key}_min"] = float(min(nums))
                out[f"{key}_max"] = float(max(nums))
        elif isinstance(val, dict):
            out.update(_scalars(val, prefix=f"{key}_"))
    return out


def _mod(name: str):
    return importlib.import_module(f"experiments.{name}")


def _gate_tagged_arm_cell(orig, gate: bool):
    """IMPROVEMENT 2 -- put the gate in the config slice so the arms' fingerprints differ.

    1080 injected the gate only through the class wrap, so STOCK and OPTOUT emitted identical
    arm fingerprints (autopsy lesson 3a: harmless while arm reuse is emit-only, a false cache
    hit the moment it consumes). Only 883 of this probe's four targets uses arm_cell; the other
    three emit no fingerprint at all, so for them there is nothing to distinguish.
    """
    def wrapper(seed, *a, **kw):
        cs = dict(kw.pop("config_slice", None) or {})
        cs["hazard_free_contamination_gate"] = bool(gate)
        cs["probe_queue_id"] = QUEUE_ID
        return orig(seed, *a, config_slice=cs, **kw)
    return wrapper


# ======================================================================================
# Target adapters -- each runs the target's OWN loop and returns its OWN verdict
# ======================================================================================
def _verdict_from(outcome: Any, direction: Any, per_claim: Any, label: Any,
                  metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "outcome": str(outcome).upper() if outcome is not None else None,
        "evidence_direction": direction,
        "evidence_direction_per_claim": per_claim or {},
        "label": label,
        "target_metrics": metrics,
    }


def _run_278(dry: bool) -> Dict[str, Any]:
    m = _mod("v3_exq_278_inv054_depression_recovery_phase_transition")
    phase1, phase2, seeds = (5, 5, [42]) if dry else (m.PHASE1_MAX_EPS, m.PHASE2_EPS, list(m.SEEDS))
    rows = [m._run_seed(s, phase1, phase2) for s in seeds]
    outcome, direction = m._aggregate(rows)
    metrics: Dict[str, Any] = {
        "seeds_passing": float(sum(1 for r in rows if r["seed_passed"])),
        "depression_established": float(sum(1 for r in rows if r["depression_established"])),
        "recovery_latency_mean": float(np.mean([r["recovery_latency"] for r in rows])),
        "recovery_latency_min": float(min(r["recovery_latency"] for r in rows)),
        "recovery_latency_max": float(max(r["recovery_latency"] for r in rows)),
        "phase1_final_z_goal_norm_mean": float(
            np.mean([r["phase1_final_z_goal_norm"] for r in rows])),
    }
    for r in rows:
        metrics[f"seed{r['seed']}_recovery_latency"] = float(r["recovery_latency"])
    return _verdict_from(outcome, direction, {}, None, metrics)


def _run_435(dry: bool) -> Dict[str, Any]:
    m = _mod("v3_exq_435_inv054_phase_transition_sustained_recovery")
    phase1, phase2, seeds = (5, 5, [42]) if dry else (m.PHASE1_MAX_EPS, m.PHASE2_EPS, list(m.SEEDS))
    total = phase1 + phase2
    rows = [m._run_seed(s, phase1, phase2, total) for s in seeds]
    outcome, direction = m._aggregate(rows)
    metrics: Dict[str, Any] = {
        "seeds_passing": float(sum(1 for r in rows if r.get("seed_passed"))),
        "depression_established": float(sum(1 for r in rows if r.get("depression_established"))),
        "recovery_onset_mean": float(np.mean([r["recovery_onset"] for r in rows])),
        "recovery_onset_min": float(min(r["recovery_onset"] for r in rows)),
        "recovery_onset_max": float(max(r["recovery_onset"] for r in rows)),
        "phase1_eps_run_mean": float(np.mean([r["phase1_eps_run"] for r in rows])),
    }
    for r in rows:
        metrics[f"seed{r['seed']}_recovery_onset"] = float(r["recovery_onset"])
    return _verdict_from(outcome, direction, {}, None, metrics)


def _run_883(dry: bool) -> Dict[str, Any]:
    m = _mod("v3_exq_883_mech427_cross_level_subgoal_credit")
    orig = m.arm_cell
    m.arm_cell = _gate_tagged_arm_cell(orig, _STATE.force_gate)
    try:
        res, _zg = m.run(dry_run=dry)
    finally:
        m.arm_cell = orig
    v = _verdict_from(res.get("outcome") or res.get("status"),
                      res.get("evidence_direction"),
                      res.get("evidence_direction_per_claim"),
                      None, _target_metrics(res))
    # IMPROVEMENT 2 made AUDITABLE rather than asserted: keep the per-cell fingerprints so a
    # reader can confirm the STOCK and OPTOUT cells really do hash differently once the gate is
    # in the config slice (1080's did not -- autopsy lesson 3a).
    # The hash lives under the key `arm_fingerprint` INSIDE the `arm_fingerprint` payload
    # (arm_fingerprint.py:722), not under `fingerprint` -- the first draft of this audit read
    # the wrong key, reported every hash as null, and so reported IMPROVEMENT 2 as not working.
    # Reading the right key is the whole point of auditing the improvement instead of asserting
    # it (CLAUDE.md: a guard that supplies the thing it asserts is not a guard).
    v["arm_fingerprints"] = [
        {"arm": r.get("arm"), "seed": r.get("seed"),
         "fingerprint": (r.get("arm_fingerprint") or {}).get("arm_fingerprint"),
         "substrate_hash": (r.get("arm_fingerprint") or {}).get("substrate_hash"),
         "config_slice_declared": (r.get("arm_fingerprint") or {}).get("config_slice_declared")}
        for r in (res.get("arm_results") or []) if isinstance(r, dict)]
    return v


def _run_231a(dry: bool) -> Dict[str, Any]:
    m = _mod("v3_exq_231a_mech106_bg_hysteresis_redesign")
    res = m.run(dry_run=dry)
    return _verdict_from(res.get("status"), res.get("evidence_direction"), {}, None,
                         _target_metrics(res))


TARGETS: List[Dict[str, Any]] = [
    {"key": "278", "queue_id": "V3-EXQ-278", "run": _run_278,
     "script": "experiments/v3_exq_278_inv054_depression_recovery_phase_transition.py",
     "direct_claims": ["INV-054"], "family_claims": [],
     "original": {"outcome": "FAIL", "evidence_direction": "does_not_support",
                  "run_id": "v3_exq_278_inv054_depression_recovery_phase_transition_1775764609_v3"},
     "hf_step_budget": {True: 150, False: 150},          # LOW_HARM STEPS_PER_EP
     "cells": {True: 1, False: 3},                       # seeds
     "action_diversity_interpretable": True,
     # phase-2 LOW_HARM (num_hazards=0) episodes: seeds x phase2_eps, loop bound unconditional
     "intended_units": {True: 1 * 5, False: 3 * 300}},
    {"key": "435", "queue_id": "V3-EXQ-435", "run": _run_435,
     "script": "experiments/v3_exq_435_inv054_phase_transition_sustained_recovery.py",
     "direct_claims": ["INV-054"], "family_claims": [],
     "original": {"outcome": "FAIL", "evidence_direction": "non_contributory",
                  "run_id": "v3_exq_435_inv054_phase_transition_sustained_recovery_1776660122_v3"},
     "hf_step_budget": {True: 150, False: 150},
     "cells": {True: 1, False: 3},
     "action_diversity_interpretable": True,
     "intended_units": {True: 1 * 5, False: 3 * 300}},
    {"key": "883", "queue_id": "V3-EXQ-883", "run": _run_883,
     "script": "experiments/v3_exq_883_mech427_cross_level_subgoal_credit.py",
     "direct_claims": ["MECH-427"], "family_claims": [],
     "original": {"outcome": "PASS", "evidence_direction": "supports",
                  "run_id": "v3_exq_883_mech427_cross_level_subgoal_credit_20260803T022051Z_v3"},
     "hf_step_budget": {True: 15, False: 40},            # N_STEPS_DRY / N_STEPS
     "cells": {True: 6, False: 6},                       # seeds x arms
     # 883 drives a SCRIPTED action sequence, so a high modal share is the design, not collapse.
     "action_diversity_interpretable": False,
     "intended_units": {True: 3 * 2 * 1, False: 3 * 2 * 1}},
    {"key": "231a", "queue_id": "V3-EXQ-231a", "run": _run_231a,
     "script": "experiments/v3_exq_231a_mech106_bg_hysteresis_redesign.py",
     "direct_claims": ["MECH-106"], "family_claims": [],
     "original": {"outcome": "PASS", "evidence_direction": "supports",
                  "run_id": "v3_exq_231a_mech106_bg_hysteresis_redesign_20260404T231335Z_v3"},
     "hf_step_budget": {True: 20, False: 200},           # STEPS_PER_EP
     "cells": {True: 2, False: 5},                       # seeds
     "action_diversity_interpretable": True,
     # POSITIVE_HISTORY runs the only num_hazards=0 env (`_make_env_easy`): seeds x n_pos
     "intended_units": {True: 2 * 5, False: 5 * 40}},
]


# ======================================================================================
# Summaries
# ======================================================================================
def _action_diversity(cell: str, interpretable: bool) -> Dict[str, Any]:
    """REPORT-ONLY (NAMED CHANGE 2). Modal action-class share over hazard-free ticks.

    The full histogram is emitted so a reader can recompute the share against its own
    denominator rather than trusting this one number (CLAUDE.md: a negative instrument must
    print its denominator). `modal_action_share=None` means NOT MEASURED (no hazard-free ticks
    recorded), never "no collapse".
    """
    hist = dict(_STATE.actions.get(cell) or {})
    total = sum(hist.values())
    n_unresolved = int(hist.get("_unresolved", 0))
    resolved = {k: v for k, v in hist.items() if k != "_unresolved"}
    n_resolved = sum(resolved.values())
    share = (max(resolved.values()) / n_resolved) if n_resolved else None
    # Three-valued by construction, so a BROKEN resolution can never read as "diverse"
    # (CLAUDE.md, negative instruments): ready / unresolved / not_measured.
    resolution = ("not_measured" if total == 0 else
                  "unresolved" if n_unresolved else "ready")
    return {
        "action_histogram": resolved,
        "n_action_classes_used": len(resolved),
        "n_ticks": total,
        "n_ticks_resolved": n_resolved,
        "n_ticks_unresolved": n_unresolved,
        "action_resolution": resolution,
        "modal_action_share": share,
        "monostrategy_threshold": MONOSTRATEGY_MODAL_SHARE,
        # None means CANNOT DETERMINE (not measured, unresolved ticks, or a scripted policy) --
        # it never means "no collapse".
        "monostrategy_suspect": (None if (share is None or not interpretable
                                          or resolution != "ready")
                                 else bool(share >= MONOSTRATEGY_MODAL_SHARE)),
        "interpretable": bool(interpretable),
        "interpretable_note": (
            "scripted action sequence -- a high modal share is the design, not monostrategy "
            "collapse" if not interpretable else
            "actions are policy-selected; a high modal share is readable as collapse "
            "(GFLAG-0487)"),
        "gating": "report-only; enters no criterion, precondition or readiness gate",
        "source": ("env-resolved action class (CausalGridWorld._last_action), not the driver's "
                   "step() argument -- the four targets pass int / one-hot / continuous vector"),
    }


def _episode_summary(eps: List[Dict[str, Any]], budget: Optional[int]) -> Dict[str, Any]:
    """Every target here is a break-on-done driver, so one DV unit == one hazard-free episode."""
    hf = [e for e in eps if e["hf"]]
    hz = [e for e in eps if not e["hf"]]
    phases = sorted({e["phase"] for e in hf})
    by_phase = {}
    for ph in phases:
        pe = [e for e in hf if e["phase"] == ph]
        by_phase[ph] = {"n_episodes": len(pe),
                        "n_deaths": sum(1 for e in pe if e["cause"] == "health_depleted"),
                        "steps_total": sum(e["steps"] for e in pe),
                        "contacts_total": sum(e["contacts"] for e in pe)}
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
        "dv_unit": "episode",
        "n_dv_units": n,
        "n_dv_units_died": deaths,
        "dv_death_frac": (deaths / n) if n else None,
        "hf_by_phase": by_phase,
    }
    if budget:
        out["hf_window_loss"] = (
            1.0 - sum(min(e["steps"], budget) for e in hf) / float(n * budget)) if n else None
        out["hf_window_loss_applicable"] = True
    else:
        out["hf_window_loss"] = None
        out["hf_window_loss_applicable"] = False
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
        "label": v.get("label"),
    }


def _positive_control() -> Dict[str, Any]:
    """Random walker on 278/435's EXACT LOW_HARM geometry -- the highest-risk target's own DV
    window. Same statistic as the load-bearing criterion. Measured 2026-09-25: stock 1.000,
    gated 0.000."""
    res: Dict[str, Any] = {}
    for arm, gate in ((ARM_STOCK, False), (ARM_OPTOUT, True)):
        cell = f"control::{arm}"
        with _cell(cell, force_gate=gate):
            rng = random.Random(20260925)
            for ep in range(CONTROL_EPISODES):
                env = CausalGridWorldV2(
                    seed=ep + 1000, size=8, num_resources=1, num_hazards=0,
                    use_proxy_fields=True, resource_respawn_on_consume=True, hazard_harm=0.02,
                    proximity_harm_scale=0.3, proximity_benefit_scale=0.18,
                    proximity_approach_threshold=0.15, hazard_field_decay=0.5,
                    energy_decay=0.005, env_drift_interval=999, env_drift_prob=0.0)
                env.reset()
                for _ in range(CONTROL_STEPS):
                    _, _, done, _, _ = env.step(rng.randint(0, 3))
                    if done:
                        break
        res[arm] = _episode_summary(_STATE.episodes[cell], CONTROL_STEPS)
        res[arm]["action_diversity"] = _action_diversity(cell, interpretable=True)
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

    print("[P0] positive control: random walker, 278/435 LOW_HARM geometry", flush=True)
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
            summ = _episode_summary(_STATE.episodes[cell], t["hf_step_budget"][dry])
            # z_goal liveness: the agents are built INSIDE the target drivers, so record each
            # target's OWN accumulator where it has one (only 883 does), not a top-level block
            # this probe has no agent handle for.
            zg = getattr(sys.modules.get(f"experiments.{Path(t['script']).stem}"), "_ZG", None)
            try:
                zg_stats = zg.stats() if zg is not None else None
            except Exception as exc:   # recording nicety -- never kill a multi-hour run for it
                zg_stats = {"_error": str(exc)[:200]}
            per_arm[arm] = {
                "verdict": verdict,
                "episodes": summ,
                "action_diversity": _action_diversity(
                    cell, bool(t["action_diversity_interpretable"])),
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
    """The target's OWN readiness route fired -- it measured nothing (1080 red-team F1)."""
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
        # Denominated on the INTENDED n and equal across arms (a death ENDS a DV unit, it never
        # adds one), so a silently dropped episode fails loud rather than shrinking the
        # denominator it is measured against (1080 red-team F8).
        {"name": f"{k}::dv_units_match_intended",
         "description": ("Both arms recorded exactly the intended number of DV-window units "
                         f"({intended}); |n_stock - intended| + |n_optout - intended|"),
         "measured": float(abs(n_s - intended) + abs(n_o - intended)),
         "threshold": 0.0, "direction": "upper", "kind": "readiness",
         "control": "intended count derived from the target's own unconditional loop bounds"},
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
    mono = {a: per_arm[a]["action_diversity"]["monostrategy_suspect"] for a in ARMS}
    print(f"[classify] {k}: {cls} (stock dv_death_frac={death}, verdict_changed={changed}, "
          f"degenerate stock/optout={deg_s}/{deg_o}, "
          f"stock_reproduces_original={stock_repro}, monostrategy_suspect={mono})", flush=True)
    return {
        "target": k, "queue_id": t["queue_id"], "script": t["script"],
        "script_sha256": _target_hash(t["script"]),
        "direct_claims": t["direct_claims"], "family_claims": t["family_claims"],
        "original": orig, "per_arm": per_arm, "preconditions": pre,
        "instrument_ok": instrument_ok,
        "stock_verdict_degenerate": deg_s, "optout_verdict_degenerate": deg_o,
        "determinable": determinable, "classification": cls,
        "dv_unit": "episode", "intended_dv_units": intended,
        "stock_dv_death_frac": death,
        "optout_dv_death_frac": o["episodes"]["dv_death_frac"],
        "stock_hf_window_loss": s["episodes"]["hf_window_loss"],
        "stock_hf_contacts_total": stock_contacts,
        "verdict_changed": changed, "materially_truncated": material,
        "stock_reproduces_original": stock_repro,
        "stock_verdict": _verdict_key(s["verdict"]), "optout_verdict": _verdict_key(o["verdict"]),
        # REPORT-ONLY, per NAMED CHANGE 2. A True here means this target's result is confounded
        # by monostrategy collapse INDEPENDENTLY of the contamination manipulation, so a clean
        # -contamination reading is not dispositive for its claim.
        "monostrategy_suspect_by_arm": mono,
        "action_diversity_interpretable": bool(t["action_diversity_interpretable"]),
    }


def _assemble(cells: List[Dict[str, Any]], control: Dict[str, Any], dry: bool) -> Dict[str, Any]:
    c_stock = control[ARM_STOCK]["dv_death_frac"]
    c_gated = control[ARM_OPTOUT]["dv_death_frac"]
    c_n = [control[a]["n_dv_units"] for a in ARMS]
    control_pre = [
        {"name": "control::stock_death_frac_supra_floor",
         "description": ("Random walker on 278/435's LOW_HARM geometry dies of contamination at "
                         "stock settings"),
         "measured": c_stock, "threshold": CONTROL_STOCK_DEATH_FLOOR, "direction": "lower",
         "kind": "readiness",
         "control": ("random walk, 20x150 steps, size-8 0-hazard V2 LOW_HARM env; 1.000 "
                     "measured 2026-09-25 at design time")},
        {"name": "control::gated_death_frac_zero",
         "description": "The same walker with hazard_free_contamination_gate=True never dies",
         "measured": c_gated, "threshold": CONTROL_GATED_DEATH_CEIL, "direction": "upper",
         "kind": "readiness", "control": "same walker, gate on; 0.000 measured 2026-09-25"},
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
    claims_determinable = {claim: [c["queue_id"] for c in det if claim in c["direct_claims"]]
                           for claim in AUDITED_CLAIM_IDS}
    n_claims_covered = sum(1 for v in claims_determinable.values() if v)
    ready = control_ok and n_claims_covered >= MIN_DETERMINABLE_CLAIMS
    n_sens = len(sens)

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

    # Only determinable targets' DIRECT claims can be cleared (1080 red-team F3/F6). INV-054 is
    # carried by TWO targets, so it clears only if EVERY determinable one of them is
    # non-sensitive -- a per-claim fold, not a per-target one.
    measured_direct = sorted({x for c in det for x in c["direct_claims"]})
    sens_claims = {x for c in sens for x in c["direct_claims"]}
    cleared_direct = ([] if (not ready or n_sens >= 2) else
                      sorted({x for x in measured_direct if x not in sens_claims}))
    not_covered = dict(UNCOVERED_CLAIMS)
    for claim in AUDITED_CLAIM_IDS:
        if claim not in measured_direct:
            owners = [c["queue_id"] for c in cells if claim in c["direct_claims"]]
            not_covered[claim] = (f"no determinable target ({', '.join(owners) or 'none'}); "
                                  "undecided")
    for c in undet:
        for x in c["family_claims"]:
            not_covered[x] = f"{c['queue_id']} could not be determined ({c['classification']})"
    family_recommended = (sorted({x for c in sens for x in c["family_claims"]})
                          if ready and n_sens == 1 else [])
    family_unmeasured = sorted({x for c in det for x in c["family_claims"]})
    # A STOCK that does not reproduce history is a substrate-drift finding, not a contamination
    # result (1080 red-team F4). These three targets are April/August 2026 runs, so this list is
    # more likely to be non-empty here than it was for 1080.
    not_reproduced = [{"queue_id": c["queue_id"], "claims": c["direct_claims"],
                       "original": {"outcome": c["original"]["outcome"],
                                    "evidence_direction": c["original"]["evidence_direction"]},
                       "stock_today": {"outcome": c["stock_verdict"]["outcome"],
                                       "evidence_direction": c["stock_verdict"]["evidence_direction"]}}
                      for c in det if not c["stock_reproduces_original"]]
    # Populated from each claim's OWN determinable targets, REGARDLESS of run-level readiness
    # (V3-EXQ-785: a whole-run AND must not vacate a clean target's finding). A record for
    # /governance, not a route -- outcome/label/reruns_owed_for_claims are unchanged by it.
    per_claim_disposition: Dict[str, Any] = {}
    for claim in AUDITED_CLAIM_IDS:
        own = [c for c in cells if claim in c["direct_claims"]]
        own_det = [c for c in own if c["determinable"]]
        own_sens = [c for c in own_det if c["classification"].startswith("verdict_sensitive")]
        if not own_det:
            disp = "not_covered_undecided"
        elif own_sens:
            disp = "verdict_sensitive_rerun_owed"
        elif any(c["materially_truncated"] for c in own_det):
            disp = "truncation_present_verdict_robust_no_rerun_owed"
        else:
            disp = "not_materially_exposed_no_rerun_owed"
        per_claim_disposition[claim] = {
            "disposition": disp,
            "targets": {c["queue_id"]: c["classification"] for c in own},
            "determinable_targets": [c["queue_id"] for c in own_det],
            "verdict_sensitive_targets": [c["queue_id"] for c in own_sens],
            "max_stock_dv_death_frac": max(
                (c["stock_dv_death_frac"] for c in own_det
                 if c["stock_dv_death_frac"] is not None), default=None),
            "monostrategy_confounded": bool(any(
                c["action_diversity_interpretable"]
                and any(c["monostrategy_suspect_by_arm"].get(a) for a in ARMS) for c in own_det)),
            "historical_verdict_reproduced": [
                c["queue_id"] for c in own_det if c["stock_reproduces_original"]],
        }

    monostrategy_confounded = sorted(
        c["queue_id"] for c in cells
        if c["action_diversity_interpretable"] and any(
            c["monostrategy_suspect_by_arm"].get(a) for a in ARMS))

    per_target_gate = {c["target"]: {"green": c["determinable"],
                                    "instrument_ok": c["instrument_ok"],
                                    "both_verdicts_degenerate": (c["stock_verdict_degenerate"]
                                                                 and c["optout_verdict_degenerate"]),
                                    "failed": [p["name"] for p in c["preconditions"]
                                               if not p["met"]]}
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
    non_degen = {"C_PREV_no_verdict_sensitive_target":
                 bool(n_claims_covered >= MIN_DETERMINABLE_CLAIMS)}
    for c in cells:
        non_degen[f"{c['target']}::stock_dv_death_frac_below_material"] = c["determinable"]
        non_degen[f"{c['target']}::verdict_unchanged_stock_vs_optout"] = c["determinable"]

    readout: Dict[str, Any] = {
        "ready": int(ready), "control_ok": int(control_ok),
        "n_targets_determinable": len(det),
        "n_mandatory_claims_covered": n_claims_covered,
        "n_verdict_sensitive": n_sens,
        "n_materially_truncated": len(trunc),
        "n_historical_not_reproduced": len(not_reproduced),
        "n_monostrategy_confounded": len(monostrategy_confounded),
        "material_death_frac_threshold": MATERIAL_DEATH_FRAC,
        "monostrategy_modal_share_threshold": MONOSTRATEGY_MODAL_SHARE,
    }
    if c_stock is not None:
        readout["control_stock_dv_death_frac"] = float(c_stock)
    if c_gated is not None:
        readout["control_gated_dv_death_frac"] = float(c_gated)
    for c in cells:
        k = c["target"]
        for name, val in (("stock_dv_death_frac", c["stock_dv_death_frac"]),
                          ("optout_dv_death_frac", c["optout_dv_death_frac"]),
                          ("stock_hf_window_loss", c["stock_hf_window_loss"]),
                          ("stock_hf_contacts_per_1k_steps",
                           c["per_arm"][ARM_STOCK]["episodes"]["hf_contacts_per_1k_steps"]),
                          ("stock_modal_action_share",
                           c["per_arm"][ARM_STOCK]["action_diversity"]["modal_action_share"]),
                          ("optout_modal_action_share",
                           c["per_arm"][ARM_OPTOUT]["action_diversity"]["modal_action_share"])):
            if val is not None and np.isfinite(float(val)):
                readout[f"t{k}_{name}"] = float(val)
        readout[f"t{k}_determinable"] = int(c["determinable"])
        readout[f"t{k}_verdict_changed"] = int(c["verdict_changed"])
        readout[f"t{k}_materially_truncated"] = int(c["materially_truncated"])
        readout[f"t{k}_stock_reproduces_original"] = int(c["stock_reproduces_original"])
        readout[f"t{k}_stock_outcome_pass"] = int(c["stock_verdict"]["outcome"] == "PASS")
        readout[f"t{k}_optout_outcome_pass"] = int(c["optout_verdict"]["outcome"] == "PASS")
        readout[f"t{k}_stock_dv_units"] = float(c["per_arm"][ARM_STOCK]["episodes"]["n_dv_units"])
        readout[f"t{k}_optout_dv_units"] = float(c["per_arm"][ARM_OPTOUT]["episodes"]["n_dv_units"])

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
        "source_autopsy": SOURCE_AUTOPSY,
        "source_audit": SOURCE_AUDIT,
        "predecessor_run_id": PREDECESSOR_RUN,
        "outcome": outcome,
        "evidence_direction": "non_contributory",
        "timestamp_utc": ts,
        "dry_run": bool(dry),
        "readout": readout,
        "criteria": criteria,
        "combination_rule": ("PASS iff ready (positive control met AND all "
                             f"{MIN_DETERMINABLE_CLAIMS} mandatory claims covered by >= 1 "
                             "determinable target) AND C_PREV "
                             "(zero verdict-sensitive targets). Per-target dv_death_frac / "
                             "verdict criteria are reported, not combined. The action-diversity "
                             "readout is REPORT-ONLY and enters no criterion."),
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
            "per_claim_disposition": per_claim_disposition,
            "claims_determinable_targets": claims_determinable,
            "per_claim_disposition_note": (
                "populated from each claim's OWN determinable targets whatever the run-level "
                "readiness verdict, so one target's degeneracy cannot vacate another's "
                "measured result (V3-EXQ-785). A RECORD, not a route: outcome, label and "
                "reruns_owed_for_claims follow 1080's pre-registered routing unchanged."),
            "monostrategy_confounded_targets": monostrategy_confounded,
            "monostrategy_caveat": (
                "REPORT-ONLY (GFLAG-0487/0489). A target listed in "
                "monostrategy_confounded_targets had >= "
                f"{MONOSTRATEGY_MODAL_SHARE} of its hazard-free ticks in one action class, so "
                "its result is confounded by monostrategy collapse INDEPENDENTLY of the "
                "contamination gate: a clean-contamination reading there must NOT be taken as "
                "dispositive for the claim. This never changes outcome or any criterion. "
                "V3-EXQ-883 is excluded by construction (scripted actions)."),
            "fidelity_caveat": ("targets re-run on the CURRENT substrate (278/435/231a are "
                                "April-2026 runs); a determinable target whose STOCK verdict "
                                "differs from its historical manifest is listed in "
                                "historical_verdict_not_reproduced for /governance as a "
                                "substrate-drift finding, not a contamination result"),
            "verdict_tuple_coarseness_caveat": (
                "278, 435 and 231a emit no evidence_direction_per_claim and no interpretation "
                "label, so their verdict tuple is effectively (outcome, evidence_direction) -- "
                "coarser than the four-field tuple 1080 compared. A verdict-unchanged reading "
                "on those three is therefore weaker evidence than the same reading on 883."),
        },
        "preconditions_all": all_pre,
        "non_degenerate": bool(ready),
        "degeneracy_reason": (None if ready else
                              "positive control failed, or fewer than "
                              f"{MIN_DETERMINABLE_CLAIMS} of the mandatory claims "
                              f"{AUDITED_CLAIM_IDS} had a determinable target "
                              f"(covered: {n_claims_covered})"),
        "positive_control": control,
        "probe_cells": cells,
        "episode_termination_stock_hf_all_targets": all_term.stats(),
        "custom_information": {
            "instrument": "class-level wrap of CausalGridWorld.__init__/step/reset",
            "gate": "hazard_free_contamination_gate=True forced on every env in ARM_OPTOUT",
            "gate_in_config_slice": ("883 only (the sole target using arm_cell); its module-level "
                                     "arm_cell is wrapped to inject the gate into the slice so "
                                     "the arms' fingerprints differ (1080 autopsy lesson 3a). "
                                     "278/435/231a emit no arm fingerprint at all."),
            "action_diversity": ("modal action-class share over hazard-free ticks, counted at "
                                 "the env boundary; REPORT-ONLY, full histogram recorded"),
            "metrics_adapter": ("reads metrics/readout/aggregates/summary_metrics, "
                                "criteria[].gap|measured|threshold and per_seed_* scalars "
                                "(1080 autopsy lesson 3). No target here is 939a-shaped, so "
                                "the criteria branch is robustness, not a specific rescue."),
            "family_extras_dropped": ("autopsy section 7's optional 894* and 807/823 targets are "
                                      "deliberately excluded -- the section names no selection "
                                      "criterion and each resolves ambiguously; their family "
                                      "caveat stands untouched"),
            "brief_premise_corrected": ("the dispatch brief described all three target claims as "
                                        "does_not_support results; measured from the landed "
                                        "manifests, only 278 is (435 non_contributory, 883 and "
                                        "231a PASS/supports)"),
        },
        "ethics_preflight": {
            "involves_negative_valence": False,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
            "note": ("SENT-0. Re-executes four already-landed V3 grid-world runs at their own "
                     "configuration; introduces no new manipulation of any kind."),
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
        "min_determinable_claims": MIN_DETERMINABLE_CLAIMS,
        "audited_claim_ids": AUDITED_CLAIM_IDS,
        "monostrategy_modal_share": MONOSTRATEGY_MODAL_SHARE,
        "control": {"episodes": CONTROL_EPISODES, "steps": CONTROL_STEPS,
                    "stock_floor": CONTROL_STOCK_DEATH_FLOOR,
                    "gated_ceiling": CONTROL_GATED_DEATH_CEIL,
                    "geometry": "278/435 LOW_HARM (size 8, 1 resource, 0 hazards)"},
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
        z_goal_stream_stats=_ZG.stats(),
    )
    r = manifest["readout"]
    print(f"outcome: {manifest['outcome']}", flush=True)
    print(f"label: {manifest['interpretation']['label']}", flush=True)
    print(f"determinable={r['n_targets_determinable']} "
          f"claims_covered={r['n_mandatory_claims_covered']}/{MIN_DETERMINABLE_CLAIMS} "
          f"sensitive={r['n_verdict_sensitive']} "
          f"truncated={r['n_materially_truncated']} control_ok={r['control_ok']} "
          f"monostrategy_confounded={r['n_monostrategy_confounded']}", flush=True)
    for c in manifest["probe_cells"]:
        print(f"  {c['target']}: {c['classification']} stock_dv_death={c['stock_dv_death_frac']} "
              f"optout_dv_death={c['optout_dv_death_frac']} "
              f"stock={c['stock_verdict']['outcome']}/{c['stock_verdict']['evidence_direction']} "
              f"optout={c['optout_verdict']['outcome']}/{c['optout_verdict']['evidence_direction']} "
              f"repro_orig={c['stock_reproduces_original']} "
              f"mono={c['monostrategy_suspect_by_arm']}", flush=True)
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
