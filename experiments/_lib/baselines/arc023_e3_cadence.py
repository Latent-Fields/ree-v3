"""Canonical config + recording instrument for the ARC-023 E3-cadence lineage
(V3-EXQ-1098 -> ...).

WHAT THIS MODULE IS
-------------------
ARC-023's `what_would_answer` (REE_assembly/docs/claims/claims.yaml, TIGHTENED
2026-09-24 under GFLAG-0443 option A) makes a specific set of REQUIRED RECORDING
fields a PRECONDITION -- "a run without it is inadmissible":

    per-step `_current_e3_steps` and `|z_beta|`; realized E3 update invocations
    (the generate_trajectories regeneration, not the clock flag); `phase_reset()`
    counts BY TRIGGER (harm, commit-entry, completion, NCL re-assert).

`CadenceRecorder` below is the instrument that emits exactly those, and
`resolve_trigger_sites()` is the FOUR-way trigger classifier the claim requires.

WHY A LOCAL CLASSIFIER, AND NOT `mech091_phase_reset.resolve_trigger_sites()`
----------------------------------------------------------------------------
`experiments/_lib/baselines/mech091_phase_reset.py` already resolves
`clock.phase_reset()` call sites from `ree_core/agent.py`'s own source, and
V3-EXQ-944/944a/944b use it. But its `TRIGGER_CLASSES` is a THREE-way tuple
`("completion", "harm", "commit_entry")`, and it classifies by the MECH-091
comment immediately above each call site -- and the NCL-reassert site carries the
SAME comment string ("MECH-091: commitment-boundary crossing (entry) is salient.")
as the two ordinary commit-entry sites. So it silently folds NCL re-assert into
`commit_entry` and CANNOT produce the 4-way split ARC-023 now requires.

MEASURED on ree-v3 origin/main (2026-09-25, this lineage's pre-authoring probe;
REE_assembly evidence/planning/arc023_recording_design_preflight_staged_20260925.md):

    resolve_trigger_sites() [mech091] -> {6821: completion, 7849: commit_entry,
                                          10449: commit_entry, 10649: commit_entry,
                                          11564: harm}
                                                 ^^^^ 7849 IS the NCL-reassert site

This module therefore adds one discriminator the comment string cannot supply:
`_ncl_hold_reassert_count` appears in the lines preceding agent.py:7849 and in
NONE of the other four sites. It is deliberately implemented HERE rather than by
editing `mech091_phase_reset.py`: that module is inside the arm-fingerprint
substrate glob (`experiments/_lib/**/*.py`), so editing it would flip the
substrate hash and REFUSE V3-EXQ-944/944a/944b's baseline reuse for no benefit to
either lineage.

THE ONSET-GATE SPLIT IS LOAD-BEARING, NOT COSMETIC
--------------------------------------------------
ARC-023's FALSIFYING signature counts only "clock-driven plus ONSET-GATED reset
triggers", and says reset ticks from a trigger with NO onset gate are "reported
separately; an excess attributable only to them is PARTIAL and routes to MECH-091
(onset gate), not a falsification of ARC-023". Read off the five call sites'
enclosing predicates in `ree_core/agent.py`:

    harm         (11564) -- fires on EVERY owned `harm_signal < 0` step. NO onset gate.
    commit_entry (10449) -- gated on `not self.beta_gate.is_elevated` (admission site).
    commit_entry (10649) -- gated on `_entering_commitment = not is_elevated` (legacy site).
    completion   (6821)  -- gated on `released` (a genuine hippocampal-completion release).
    ncl_reassert (7849)  -- gated on `not self.beta_gate.is_elevated` inside the hold.

So `UNGATED_CLASSES = ("harm",)`, which is exactly the claim's own parenthetical
("the MECH-091 harm trigger as of 2026-09-24").

`clock.advance()` collapses several requests into ONE tick (`_pending_phase_reset`
is an idempotent flag), so a reset-driven tick is attributed by the SET of classes
that requested it since the previous advance: a tick with at least one gated
requester counts toward the falsifying share, a harm-only tick is reported
separately. That is the conservative direction for FALSIFYING.

TWO RECORDED STRUCTURAL LIMITATIONS (negative-instrument discipline)
-------------------------------------------------------------------
Two of the four required classes cannot fire at default config, for reasons that
are ARITHMETIC rather than "not trained". `trigger_reachability()` returns a
THREE-VALUED status per class so a zero count can never be read as "measured zero"
(CLAUDE.md "Negative instruments"):

  (1) `completion` -- `HippocampalModule.compute_completion_signal` is
      `sigmoid(-best_score * 0.5)`, whose achievable range on non-negative residue
      is (0, 0.5]; `BetaGate.receive_hippocampal_completion` releases only at
      `>= completion_release_threshold`, default 0.75 (beta_gate.py:33). So the
      release -- and therefore the phase_reset at agent.py:6821 -- is UNREACHABLE
      AS AN IDENTITY at default config, regardless of training and regardless of
      `beta_gate_bistable`. Registered as substrate_queue entry
      `residue-completion-signal-threshold-unreachable`; the inverted-range
      docstring was corrected 2026-09-22 under GFLAG-0344. The recorder measures
      the observed completion-signal MAX against that threshold so the claim is
      re-derivable from the manifest rather than taken on trust here.

  (2) `ncl_reassert` -- requires `_ncl_hold_active`, which needs
      `use_natural_commit_latch_hold`. Measured 0 in the 2026-09-25 probe EVEN WITH
      that knob ON, consistent with `ree_core/utils/config.py`'s own recorded note
      that the latch-hold "NEVER armed (ncl_hold_reassert_total=0)".

So the realised split is TWO live classes (harm, commit-entry) of four. That is a
property of the substrate, not of this driver, and it is recorded as such.

WHY THIS ENV / SCHEDULE
-----------------------
ENV_KWARGS and `alpha_world=0.9` are V3-EXQ-942's, deliberately: ARC-023's numeric
tolerances (+/-15% relative, >= 0.08 absolute, <= 0.10) were calibrated with 942's
measured E3 shares (0.153 / 0.169 / 0.394) in view. `num_hazards > 0` is
load-bearing -- the harm trigger is the highest-volume reset source, so a
hazard-free grid leaves the E3 tick near-periodic however the loop is driven.

`beta_gate_bistable = True` is a RATIFIED choice, not this module's default
preference -- see BISTABLE_DECISION below.

ASCII-only output (repo rule).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

import ree_core.agent as _agent_mod  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402


# --- Canonical constants -----------------------------------------------------

HARM_HISTORY_LEN = 10

ENV_KWARGS: Dict[str, Any] = dict(
    size=10, num_hazards=1, num_resources=5, harm_history_len=HARM_HISTORY_LEN,
)

ALPHA_WORLD = 0.9          # SD-008: z_world fidelity (942's value)

# RATIFIED 2026-09-25 by orchestrate-20260924-1707 under the user's standing
# delegation (rec-20260924-fb429c72), answering decision chip
# chip-20260925-arc023-trigger-reachability-config with option (2).
#
# It is NOT a from_dims kwarg -- `REEConfig.from_dims` silently swallows unknown
# kwargs (memory reference-reeconfig-from-dims-silent-kwargs), so passing it there
# would leave it False with no error. Set on cfg.heartbeat and ASSERTED in
# build_agent().
#
# CORRECTION to the ratifying reason, recorded rather than papered over
# (CLAUDE.md "Working a chip or brief: audit its premises"): the decision's
# rationale was that option (1) "makes completion structurally unreachable
# (code-gated)" while (2) makes it reachable. The code gate at agent.py:6821 is
# real, but removing it does NOT make completion reachable -- the 0.75 release
# threshold is unreachable as an identity (see limitation (1) in the module
# docstring), so completion fires under NEITHER option. The decision's OUTCOME is
# unaffected because its other two reasons stand and are the operative ones:
# (2) matches the canonical MECH-091 lineage and V3-EXQ-944/944a/944b, and it
# routes commit-entry through the CURRENT readiness-admission site (agent.py:10449)
# rather than the legacy elevate site (10649).
BETA_GATE_BISTABLE = True
BISTABLE_DECISION = "chip-20260925-arc023-trigger-reachability-config option (2)"

WARMUP_EPISODES = 40       # P0: 942's warmup, ONCE per seed
EVAL_EPISODES = 30         # measurement; 942 used 25 -- raised for sample headroom
STEPS_PER_EPISODE = 150
TOTAL_PROGRESS_EPISODES = WARMUP_EPISODES + EVAL_EPISODES   # runner denominator

SCHEDULE: Dict[str, int] = {
    "warmup_episodes": WARMUP_EPISODES,
    "eval_episodes": EVAL_EPISODES,
    "steps_per_episode": STEPS_PER_EPISODE,
}

# --- Trigger-site resolution (runtime, from agent.py's own source) -----------

TRIGGER_CLASSES: Tuple[str, ...] = ("harm", "commit_entry", "completion", "ncl_reassert")

# Exactly the classes whose call site has NO onset gate -- see the module
# docstring's per-site predicate reading. ARC-023's FALSIFYING leg excludes these.
UNGATED_CLASSES: Tuple[str, ...] = ("harm",)

_NCL_MARKER = "_ncl_hold_reassert_count"
_NCL_CONTEXT_LINES = 14


def _agent_source_lines() -> List[str]:
    return Path(_agent_mod.__file__).read_text(encoding="utf-8").splitlines()


def classify_site(lineno: int, src: Optional[List[str]] = None) -> str:
    """Classify ONE `clock.phase_reset()` call site into a 4-way MECH-091 class.

    Resolved from the live source so attribution survives line drift. The
    comment-based branches match `mech091_phase_reset.resolve_trigger_sites()`
    exactly; the `ncl_reassert` branch is this lineage's addition and is checked
    FIRST because the NCL site's comment is identical to commit-entry's.
    """
    src = _agent_source_lines() if src is None else src
    lo = max(0, lineno - 1 - _NCL_CONTEXT_LINES)
    ctx = " ".join(src[lo:lineno - 1])
    if _NCL_MARKER in ctx:
        return "ncl_reassert"
    low = ctx.lower()
    if "task completion is salient" in low:
        return "completion"
    if "commitment-boundary crossing" in low:
        return "commit_entry"
    if "harm is salient" in low:
        return "harm"
    return "unknown"


def resolve_trigger_sites() -> Dict[int, str]:
    """Map `ree_core/agent.py` line number -> 4-way MECH-091 trigger class.

    `unknown` means a `phase_reset()` call exists that this lineage cannot
    attribute -- surfaced, never silently bucketed.
    """
    src = _agent_source_lines()
    sites: Dict[int, str] = {}
    for lineno, line in enumerate(src, start=1):
        if line.strip().startswith("#"):
            continue
        if "self.clock.phase_reset()" not in line:
            continue
        sites[lineno] = classify_site(lineno, src)
    return sites


def assert_trigger_wiring(sites: Optional[Dict[int, str]] = None) -> Dict[str, int]:
    """Fail loudly if the substrate no longer wires all FOUR trigger classes.

    A WIRING assertion, not a firing assertion: it proves the call sites exist in
    `ree_core/agent.py`. Whether each site FIRES is a separate measured quantity --
    see `CadenceRecorder.summary()["requests_by_class"]` and
    `trigger_reachability()`. This is what stops a substrate edit that REMOVES a
    trigger from silently under-recording the claim's required split.
    """
    sites = resolve_trigger_sites() if sites is None else sites
    counts = {c: sum(1 for v in sites.values() if v == c) for c in TRIGGER_CLASSES}
    missing = [c for c, n in counts.items() if n == 0]
    if missing:
        raise AssertionError(
            "ARC-023 trigger wiring regression: no clock.phase_reset() call site "
            "found for %s in %s. All four MECH-091 classes are REQUIRED RECORDING "
            "for this claim." % (", ".join(missing), _agent_mod.__file__)
        )
    unknown = sorted(ln for ln, v in sites.items() if v == "unknown")
    if unknown:
        print("  [warn] unattributable clock.phase_reset() call site(s) at agent.py "
              "line(s) %s -- counted under 'unknown'" % unknown, flush=True)
    return counts


def trigger_reachability(agent: REEAgent) -> Dict[str, Dict[str, Any]]:
    """THREE-VALUED structural reachability per trigger class, read off config.

    `status` is one of:
      "live"                     -- nothing in config prevents this class firing
      "structurally_unreachable" -- a config gate or an arithmetic identity blocks it

    This exists so a ZERO count in the manifest can never be read as "measured
    zero" when it is in fact "could not fire" (CLAUDE.md "Negative instruments":
    an explicit cannot-determine CATEGORY in the data model, not a print
    statement). Each entry carries its own `reason` and `source`.
    """
    hb = agent.config.heartbeat
    bistable = bool(getattr(hb, "beta_gate_bistable", False))
    ncl_knob = bool(getattr(agent.config, "use_natural_commit_latch_hold", False))
    thr = float(getattr(agent.beta_gate, "_completion_release_threshold", 0.75))
    out: Dict[str, Dict[str, Any]] = {
        "harm": {
            "status": "live",
            "onset_gated": False,
            "reason": "fires on every owned harm_signal < 0 step; no onset gate",
            "source": "ree_core/agent.py update_residue",
        },
        "commit_entry": {
            "status": "live",
            "onset_gated": True,
            "reason": ("admission site active when beta_gate_bistable is True, "
                       "legacy elevate site otherwise; both gated on a genuine "
                       "not-elevated -> elevated transition"),
            "source": "ree_core/agent.py select_action",
        },
        "completion": {
            # Reachability here is NOT about training. See module docstring (1).
            "status": "structurally_unreachable",
            "onset_gated": True,
            "completion_release_threshold": thr,
            "completion_signal_max_possible": 0.5,
            "reason": (
                "compute_completion_signal = sigmoid(-best_score*0.5) has achievable "
                "range (0, 0.5] on non-negative residue, but release requires "
                ">= completion_release_threshold (%.2f), so the phase_reset is "
                "unreachable as an identity at default config -- independent of "
                "training and of beta_gate_bistable (%s)" % (thr, bistable)
            ),
            "source": ("ree_core/hippocampal/module.py compute_completion_signal + "
                       "ree_core/heartbeat/beta_gate.py receive_hippocampal_completion; "
                       "substrate_queue residue-completion-signal-threshold-unreachable; "
                       "GFLAG-0344"),
        },
        "ncl_reassert": {
            "status": "live" if ncl_knob else "structurally_unreachable",
            "onset_gated": True,
            "use_natural_commit_latch_hold": ncl_knob,
            "reason": (
                "requires _ncl_hold_active, which requires use_natural_commit_latch_hold "
                "(currently %s); measured 0 in the 2026-09-25 probe even with the knob ON, "
                "consistent with config.py's recorded note that the latch-hold NEVER armed "
                "(ncl_hold_reassert_total=0)" % ncl_knob
            ),
            "source": "ree_core/agent.py select_action; ree_core/utils/config.py",
        },
    }
    if thr > 0.5:
        out["completion"]["threshold_exceeds_achievable_max"] = True
    return out


# --- Build helpers -----------------------------------------------------------

def config_slice() -> Dict[str, Any]:
    """Fingerprint/provenance slice: ONLY what this lineage's computation reads.

    Never the acceptance thresholds -- those are the driver's pre-registered
    constants and must not enter a reuse key.
    """
    return {
        "env": dict(ENV_KWARGS),
        "schedule": dict(SCHEDULE),
        "alpha_world": ALPHA_WORLD,
        "beta_gate_bistable": BETA_GATE_BISTABLE,
        "harm_history_len": HARM_HISTORY_LEN,
    }


def build_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def build_agent(env: CausalGridWorldV2) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=ALPHA_WORLD,
        harm_history_len=HARM_HISTORY_LEN,
    )
    # NOT a from_dims kwarg -- see BETA_GATE_BISTABLE's comment.
    cfg.heartbeat.beta_gate_bistable = BETA_GATE_BISTABLE
    agent = REEAgent(cfg)
    assert agent.config.heartbeat.beta_gate_bistable is BETA_GATE_BISTABLE, (
        "beta_gate_bistable did not reach agent.config.heartbeat -- REEConfig "
        "wiring regression; the ratified trigger configuration would be silently wrong."
    )
    # The claim's own NON-DEGENERACY PRECONDITION: distinct configured periods.
    hb = agent.config.heartbeat
    assert len({hb.e1_steps_per_tick, hb.e2_steps_per_tick, hb.e3_steps_per_tick}) == 3, (
        "ARC-023 non-degeneracy precondition violated: E1/E2/E3 are not configured "
        "at three DISTINCT periods (%s/%s/%s) -- the test would be vacuous."
        % (hb.e1_steps_per_tick, hb.e2_steps_per_tick, hb.e3_steps_per_tick)
    )
    return agent


# --- The recording instrument ------------------------------------------------

class CadenceRecorder:
    """Record-only instrumentation for ARC-023's REQUIRED RECORDING fields.

    RECORD-ONLY IS THE POINT: every wrapper records and then calls straight
    through, so the agent's behaviour is bit-identical to an uninstrumented run.
    This is NOT the MECH-091 lineage's `install_reset_policy`, which ABLATES
    (suppresses or defers) resets to form arms -- ARC-023's falsifier is a
    single-condition measurement of production behaviour, so there is nothing to
    ablate and an ablation would measure a different claim.

    What it emits, mapped to the claim's REQUIRED RECORDING:
      per-step `_current_e3_steps`         -> k_per_step (read BEFORE each advance)
      per-step `|z_beta|`                  -> z_beta_norm_per_step (the value the
                                              clock itself consumed)
      realized E3 update invocations       -> realized_e3_invocations (wraps
                                              `agent._e3_tick`, the
                                              generate_trajectories regeneration,
                                              NOT the clock flag)
      phase_reset() counts BY TRIGGER      -> requests_by_class / requests_by_site

    WHY `_current_e3_steps` IS READ BEFORE `advance()`: `StepHarness.step` calls
    `clock.advance()` (step 3) BEFORE `agent._e1_tick()`, and it is `_e1_tick` that
    calls `clock.update_e3_rate_from_beta`. So the period in force when advance()
    evaluates its threshold is the one set on the PREVIOUS step. Reading it after
    advance would record the wrong K and bias leg (i)'s expected count.

    CLOCK-DRIVEN vs RESET-DRIVEN: `advance()` returns a single `e3_tick` flag and
    does not say which branch fired, so the recorder reads
    `clock._pending_phase_reset` immediately BEFORE advance -- True there and
    `e3_tick` True means the reset branch fired, otherwise the phase-step branch did.
    """

    def __init__(self, agent: REEAgent, sites: Optional[Dict[int, str]] = None) -> None:
        self.agent = agent
        self.clock = agent.clock
        self.sites = resolve_trigger_sites() if sites is None else sites
        self._src = _agent_source_lines()

        self.k_per_step: List[int] = []
        self.z_beta_norm_per_step: List[float] = []
        self.clock_driven = 0
        self.reset_driven = 0
        self.reset_driven_gated = 0        # >= 1 onset-gated requester in the window
        self.reset_driven_ungated_only = 0  # harm-only -> the PARTIAL attribution
        self.realized_e3_invocations = 0
        self.e3_invocations_without_tick = 0   # cache-miss regenerations
        self.requests_by_class: Dict[str, int] = {c: 0 for c in TRIGGER_CLASSES}
        self.requests_by_class["unknown"] = 0
        self.requests_by_site: Dict[str, int] = {}
        self.steps = 0
        self.phase_step_zeroed_by_reset = 0   # leg (i) interference diagnostic
        self.completion_signal_max = 0.0
        self.completion_signal_observations = 0

        self._window: Dict[str, int] = {}     # classes requesting since last advance
        self._last_tick_kind: Optional[str] = None
        self._installed: List[Tuple[Any, str, Any]] = []

    # -- install / restore ---------------------------------------------------

    def install(self) -> "CadenceRecorder":
        clock, agent = self.clock, self.agent

        real_reset = clock.phase_reset
        def phase_reset_recording() -> None:
            lineno = sys._getframe(1).f_lineno
            cls = self.sites.get(lineno) or classify_site(lineno, self._src)
            if cls not in self.requests_by_class:
                self.requests_by_class[cls] = 0
            self.requests_by_class[cls] += 1
            key = "%s@%d" % (cls, lineno)
            self.requests_by_site[key] = self.requests_by_site.get(key, 0) + 1
            self._window[cls] = self._window.get(cls, 0) + 1
            real_reset()
        self._swap(clock, "phase_reset", phase_reset_recording, real_reset)

        real_advance = clock.advance
        def advance_recording() -> dict:
            k_before = int(clock._current_e3_steps)
            pending_before = bool(clock._pending_phase_reset)
            ticks = real_advance()
            self.k_per_step.append(k_before)
            self.steps += 1
            if ticks.get("e3_tick", False):
                if pending_before:
                    self.reset_driven += 1
                    self._last_tick_kind = "reset"
                    self.phase_step_zeroed_by_reset += 1
                    gated = [c for c in self._window if c not in UNGATED_CLASSES]
                    if gated:
                        self.reset_driven_gated += 1
                    else:
                        self.reset_driven_ungated_only += 1
                else:
                    self.clock_driven += 1
                    self._last_tick_kind = "clock"
            else:
                self._last_tick_kind = None
            self._window = {}
            return ticks
        self._swap(clock, "advance", advance_recording, real_advance)

        real_rate = clock.update_e3_rate_from_beta
        def rate_recording(z_beta: torch.Tensor) -> None:
            # The |z_beta| the clock ITSELF consumed this step -- not a re-read.
            self.z_beta_norm_per_step.append(
                float(z_beta.detach().norm(dim=-1).mean().item())
            )
            real_rate(z_beta)
        self._swap(clock, "update_e3_rate_from_beta", rate_recording, real_rate)

        real_e3 = agent._e3_tick
        def e3_recording(*a: Any, **kw: Any) -> Any:
            self.realized_e3_invocations += 1
            if self._last_tick_kind is None:
                self.e3_invocations_without_tick += 1
            return real_e3(*a, **kw)
        self._swap(agent, "_e3_tick", e3_recording, real_e3)

        hip = getattr(agent, "hippocampal", None)
        if hip is not None and hasattr(hip, "compute_completion_signal"):
            real_cs = hip.compute_completion_signal
            def cs_recording(trajectories: Any) -> float:
                val = real_cs(trajectories)
                self.completion_signal_observations += 1
                if float(val) > self.completion_signal_max:
                    self.completion_signal_max = float(val)
                return val
            self._swap(hip, "compute_completion_signal", cs_recording, real_cs)
        return self

    def _swap(self, obj: Any, name: str, new: Any, old: Any) -> None:
        setattr(obj, name, new)
        self._installed.append((obj, name, old))

    def restore(self) -> None:
        for obj, name, old in reversed(self._installed):
            setattr(obj, name, old)
        self._installed = []

    def __enter__(self) -> "CadenceRecorder":
        return self.install()

    def __exit__(self, *exc: Any) -> None:
        self.restore()

    # -- readout -------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """The per-seed row. Shares are per RECORDED STEP, never per nominal step."""
        n = max(1, self.steps)
        e2_steps = int(self.agent.config.heartbeat.e2_steps_per_tick)
        e2_share = 1.0 / float(e2_steps)
        expected = sum(1.0 / float(k) for k in self.k_per_step if k > 0)
        ks = sorted(set(self.k_per_step))
        zb = self.z_beta_norm_per_step
        share_realized = self.realized_e3_invocations / float(n)
        return {
            "steps_recorded": self.steps,
            # --- REQUIRED RECORDING: per-step period + arousal ---------------
            "current_e3_steps_distinct": ks,
            "current_e3_steps_n_distinct": len(ks),
            "current_e3_steps_min": (min(ks) if ks else None),
            "current_e3_steps_max": (max(ks) if ks else None),
            "current_e3_steps_mean": (sum(self.k_per_step) / float(n)) if self.k_per_step else None,
            "z_beta_norm_min": (min(zb) if zb else None),
            "z_beta_norm_max": (max(zb) if zb else None),
            "z_beta_norm_mean": (sum(zb) / len(zb)) if zb else None,
            "z_beta_norm_spread": ((max(zb) - min(zb)) if zb else None),
            "z_beta_observations": len(zb),
            # --- REQUIRED RECORDING: realized E3 updates ---------------------
            "realized_e3_invocations": self.realized_e3_invocations,
            "e3_invocations_without_clock_tick": self.e3_invocations_without_tick,
            # --- REQUIRED RECORDING: phase_reset counts BY TRIGGER -----------
            "requests_by_class": dict(self.requests_by_class),
            "requests_by_site": dict(self.requests_by_site),
            # --- tick decomposition -----------------------------------------
            "clock_driven_ticks": self.clock_driven,
            "reset_driven_ticks": self.reset_driven,
            "reset_driven_ticks_with_gated_requester": self.reset_driven_gated,
            "reset_driven_ticks_ungated_only": self.reset_driven_ungated_only,
            "phase_step_zeroed_by_reset": self.phase_step_zeroed_by_reset,
            # --- the quantities the criteria read ---------------------------
            "expected_clock_driven": expected,
            "clock_tracking_ratio": (self.clock_driven / expected) if expected > 0 else None,
            "e3_share_realized": share_realized,
            "e3_share_flag": (self.clock_driven + self.reset_driven) / float(n),
            "clock_driven_share": self.clock_driven / float(n),
            "reset_driven_share": self.reset_driven / float(n),
            "e3_share_gated": (self.clock_driven + self.reset_driven_gated) / float(n),
            "e2_configured_share": e2_share,
            "e3_share_gap_below_e2": e2_share - share_realized,
            # --- completion reachability, measured --------------------------
            "completion_signal_max": self.completion_signal_max,
            "completion_signal_observations": self.completion_signal_observations,
        }
