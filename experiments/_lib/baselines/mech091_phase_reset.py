"""Canonical baseline arm + shared cell runner for the MECH-091 salient-event
phase-reset lineage (V3-EXQ-944 -> ...).

Arm-reuse Phase 0 (instrument-only). Design plan:
REE_assembly/evidence/planning/arm_reuse_fingerprint_plan.md (sections 2, 7b, 9.7).

WHAT THIS MODULE IS
-------------------
The single source of truth for (a) the env/schedule/config the MECH-091
lineage runs, (b) the NO_RESET baseline arm's own switch, and (c) the shared
per-cell loop every arm runs. Both the baseline and the treatment arms build
from here so a later sibling's baseline cell fingerprints identically by
construction rather than by luck, and so the treatment arms cannot silently
drift away from the baseline's loop (there is only one loop).

`experiments/_lib/**/*.py` is inside the arm-fingerprint substrate glob, so any
edit here correctly flips the substrate hash and REFUSES a stale reuse.

WHY THIS ENV, AND WHY IT IS LOAD-BEARING
----------------------------------------
ENV_KWARGS / SCHEDULE are taken deliberately from
`experiments/_lib/baselines/arc071_chunking.py` -- the lineage whose own probe
(REE_assembly/evidence/planning/diagnostic_arc071_e3_reselection_probe_2026-08-01.md,
a real 53,063-step rollout) MEASURED MECH-091's `clock.phase_reset()` as the
DOMINANT driver of E3 tick cadence. `num_hazards > 0` is load-bearing for the
same reason it is there: until 2026-08-17 `harm_signal < 0` was the sole trigger
for `phase_reset()`, so a hazard-free grid leaves the E3 tick perfectly periodic
however the loop is driven.

THE THREE MECH-091 SALIENT-EVENT TRIGGERS
-----------------------------------------
MECH-091 names three: task completion, unexpected harm, commitment-boundary
crossing. All three were wired into `clock.phase_reset()` by ree-v3 origin/main
6293b2395248 (2026-08-17), which is what un-parked this claim. `TRIGGER_SITES`
below is resolved AT RUNTIME from `ree_core/agent.py`'s own source rather than
from a hardcoded line table, so (a) attribution survives ordinary line drift and
(b) a substrate edit that REMOVES a trigger is caught by
`assert_trigger_wiring()` instead of silently under-testing the claim.

ARM MECHANICS -- A DRIVER-SIDE ABLATION, NOT A SUBSTRATE CHANGE
---------------------------------------------------------------
There is no REEConfig knob that disables `phase_reset()`; the wiring is
unconditional (gated only on the genuine-transition checks at each call site).
`install_reset_policy()` therefore replaces the bound `phase_reset` method on
the AGENT INSTANCE's clock. Every request is recorded whatever the arm does with
it, so the three arms share an identical salient-event ledger and differ only in
which requests are EXECUTED and when:

  ALIGNED       execute at the event                (production behaviour)
  RATE_MATCHED  suppress at the event; re-issue at event_step + U{K..3K}
                (same number of extra E3 ticks, decoupled from the event)
  NO_RESET      never execute                       (what_would_answer's
                                                     literal no-reset control)

RATE_MATCHED exists because ALIGNED does not only re-align the E3 window, it
also RAISES the E3 tick rate (measured 0.19-0.30 vs 0.11 for NO_RESET in the
2026-08-22 pre-queue probe). Without it, an ALIGNED-vs-NO_RESET contrast cannot
separate "replans more often" from "replans in phase with the event", and the
latter is the whole of MECH-091.

ASCII-only output (repo rule).
"""

from __future__ import annotations

import random
import sys
from bisect import bisect_right
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
from experiments._harness import StepHarness  # noqa: E402


# --- Canonical constants -----------------------------------------------------

ENV_KWARGS: Dict[str, Any] = {
    "size": 8,
    "num_hazards": 2,
    "num_resources": 6,
    "use_proxy_fields": True,
}

WARMUP_EPISODES = 60      # P0: prediction-loss warmup, ONCE per seed, shared by arms
MEASURE_EPISODES = 30     # P2: frozen-weight measurement, per arm
STEPS_PER_EPISODE = 72

SCHEDULE: Dict[str, int] = {
    "warmup_episodes": WARMUP_EPISODES,
    "measure_episodes": MEASURE_EPISODES,
    "steps_per_episode": STEPS_PER_EPISODE,
}

ALPHA_WORLD = 0.9   # SD-008: z_world fidelity
ALPHA_SELF = 0.3

# beta_gate_bistable is what makes the ARC-028 / MECH-105 hippocampal-completion
# release path (and therefore MECH-091's completion trigger, agent.py inside
# _e3_tick) reachable at all, and it routes commitment-entry through the
# readiness-conjunction admission site rather than the legacy elevate site.
# It is NOT a from_dims kwarg -- see set_bistable() below.
BETA_GATE_BISTABLE = True

ARM_ALIGNED = "ALIGNED"
ARM_RATE_MATCHED = "RATE_MATCHED"
ARM_NO_RESET = "NO_RESET"
ARM_NAMES: Tuple[str, ...] = (ARM_ALIGNED, ARM_RATE_MATCHED, ARM_NO_RESET)

# RATE_MATCHED decoupling delay, in units of the E3 cadence K. The lower bound
# is K so the re-issued reset can never land inside the cycle that contains the
# event, which is exactly the alignment ALIGNED provides. The upper bound is 2K
# rather than 3K to limit END-OF-EPISODE TRUNCATION: a reset scheduled past the
# episode's step budget would simply never fire, and RATE_MATCHED would then
# under-execute and silently drift back toward NO_RESET. Measured in the
# 2026-08-22 pre-queue probe at 3K: 65-84 percent of requests executed. The
# clamp in install_reset_policy() closes the remainder.
DECOUPLE_MIN_K = 1
DECOUPLE_MAX_K = 2


def shared_config_kwargs() -> Dict[str, Any]:
    """Config every arm is built with. The arms differ ONLY in the driver-side
    reset policy, so there are no per-arm REEConfig switches in this lineage."""
    return {
        "alpha_world": ALPHA_WORLD,
        "alpha_self": ALPHA_SELF,
        "self_dim": 32,
        "world_dim": 32,
    }


def arm_flags(arm: str) -> Dict[str, Any]:
    """The arm's own (driver-side) switch. Declared so it enters the
    fingerprint slice: two arms at one seed must not hash identically."""
    if arm not in ARM_NAMES:
        raise ValueError("unknown arm: %s" % arm)
    return {"reset_policy": arm}


def baseline_arm_flags() -> Dict[str, Any]:
    """The NO_RESET control -- this lineage's baseline arm."""
    return arm_flags(ARM_NO_RESET)


def arm_config_slice(arm: str) -> Dict[str, Any]:
    """Fingerprint slice for any cell of this lineage.

    Declares ONLY what the cell's computation reads: env, schedule, the
    substrate-operating config every arm runs, the bistable switch, and the
    arm's own reset policy. Never the acceptance thresholds.
    """
    slice_: Dict[str, Any] = {
        "env": dict(ENV_KWARGS),
        "schedule": dict(SCHEDULE),
        "beta_gate_bistable": BETA_GATE_BISTABLE,
        "decouple_k_range": [DECOUPLE_MIN_K, DECOUPLE_MAX_K],
    }
    slice_.update(shared_config_kwargs())
    slice_.update(arm_flags(arm))
    return slice_


def baseline_config_slice() -> Dict[str, Any]:
    """The fingerprint slice for the NO_RESET baseline cell."""
    return arm_config_slice(ARM_NO_RESET)


# --- Trigger-site resolution (runtime, from agent.py's own source) -----------

def resolve_trigger_sites() -> Dict[int, str]:
    """Map `ree_core/agent.py` line number -> MECH-091 salient-event class.

    Resolved from the live source so attribution survives line drift; the
    classifier reads the MECH-091 comment the wiring commit placed above each
    call site. `unknown` means a phase_reset() call exists that this lineage
    cannot attribute -- surfaced, never silently bucketed.
    """
    src = Path(_agent_mod.__file__).read_text(encoding="utf-8").splitlines()
    sites: Dict[int, str] = {}
    for lineno, line in enumerate(src, start=1):
        if line.strip().startswith("#"):
            continue
        if "self.clock.phase_reset()" not in line:
            continue
        ctx = " ".join(src[max(0, lineno - 9):lineno - 1]).lower()
        if "task completion is salient" in ctx:
            cls = "completion"
        elif "commitment-boundary crossing" in ctx:
            cls = "commit_entry"
        elif "harm is salient" in ctx:
            cls = "harm"
        else:
            cls = "unknown"
        sites[lineno] = cls
    return sites


TRIGGER_CLASSES: Tuple[str, ...] = ("completion", "harm", "commit_entry")


def assert_trigger_wiring(sites: Optional[Dict[int, str]] = None) -> Dict[str, int]:
    """Fail loudly if the substrate no longer wires all three MECH-091 triggers.

    This is a WIRING assertion, not a firing assertion: it proves the call sites
    exist in `ree_core/agent.py` (ree-v3 6293b2395248 wired the two that were
    missing). Whether each site actually FIRES in a rollout is a separate,
    measured quantity -- see `cell_result["trigger_coverage"]`.
    """
    sites = resolve_trigger_sites() if sites is None else sites
    counts = {c: sum(1 for v in sites.values() if v == c) for c in TRIGGER_CLASSES}
    missing = [c for c, n in counts.items() if n == 0]
    if missing:
        raise AssertionError(
            "MECH-091 trigger wiring regression: no clock.phase_reset() call site "
            "found for %s in %s. The 2026-08-17 wiring (ree-v3 6293b2395248) is "
            "the premise of this lineage." % (", ".join(missing), _agent_mod.__file__)
        )
    unknown = sorted(ln for ln, v in sites.items() if v == "unknown")
    if unknown:
        print("  [warn] unattributable clock.phase_reset() call site(s) at agent.py "
              "line(s) %s -- counted under 'unknown'" % unknown, flush=True)
    return counts


# --- Build helpers -----------------------------------------------------------

def build_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def build_agent(env: CausalGridWorldV2) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        **shared_config_kwargs(),
    )
    # NOT a from_dims kwarg. `REEConfig.from_dims` silently swallows unknown
    # kwargs (memory reference-reeconfig-from-dims-silent-kwargs), so passing
    # beta_gate_bistable=True above would leave it False with no error --
    # confirmed by direct probe 2026-08-22, which is how the first version of
    # this lineage's pre-queue probe measured a dead completion path.
    cfg.heartbeat.beta_gate_bistable = BETA_GATE_BISTABLE
    agent = REEAgent(cfg)
    assert agent.config.heartbeat.beta_gate_bistable is True, (
        "beta_gate_bistable did not reach agent.config.heartbeat -- REEConfig "
        "wiring regression; the MECH-091 completion trigger would be unreachable."
    )
    return agent


# --- Reset policy (the arm manipulation) ------------------------------------

class ResetLedger:
    """Records every phase_reset() REQUEST, whatever the arm does with it."""

    def __init__(self) -> None:
        self.requests: List[Tuple[int, int, str]] = []   # (episode, step, class)
        self.by_class: Dict[str, int] = {}
        self.by_site: Dict[str, int] = {}
        self.executed: int = 0
        self.episode: int = 0
        # V3-EXQ-944a: a RATE_MATCHED request whose decoupled slot could not be
        # placed inside the episode is DROPPED, not executed. 944 had no counter
        # for this, so a cell that silently degraded toward NO_RESET was
        # invisible in the manifest and surfaced only as a downstream rate-match
        # failure attributed to the substrate. Counted here so the shortfall is
        # readable directly off the cell row.
        self.dropped: int = 0

    def note(self, step: int, cls: str, lineno: int) -> None:
        self.requests.append((self.episode, step, cls))
        self.by_class[cls] = self.by_class.get(cls, 0) + 1
        key = "%s@%d" % (cls, lineno)
        self.by_site[key] = self.by_site.get(key, 0) + 1


def install_reset_policy(agent: REEAgent, arm: str, ledger: ResetLedger,
                         sites: Dict[int, str], rng: random.Random,
                         episode_step_budget: Optional[int] = None):
    """Replace the instance's bound clock.phase_reset with the arm's policy.

    Returns (restore_fn, pending_list). `pending_list` holds the RATE_MATCHED
    arm's deferred reset steps; the caller must drain it once per env step
    BEFORE `harness.step(...)` (see `_drain_pending`).
    """
    clock = agent.clock
    real = clock.phase_reset
    pending: List[int] = []
    k = int(clock.e3_steps_per_tick)

    def patched() -> None:
        lineno = sys._getframe(1).f_lineno
        ledger.note(clock._global_step, sites.get(lineno, "unknown"), lineno)
        if arm == ARM_ALIGNED:
            ledger.executed += 1
            real()
        elif arm == ARM_RATE_MATCHED:
            due = clock._global_step + rng.randint(DECOUPLE_MIN_K * k,
                                                   DECOUPLE_MAX_K * k)
            # V3-EXQ-944a FIX 1 -- OFF-BY-ONE. `drain()` runs BEFORE
            # `harness.step()`, so across a `for _ in range(steps)` loop the
            # LARGEST `clock._global_step` a drain ever observes is
            # `steps - 1`, never `steps`. 944 clamped to `episode_step_budget`
            # (= steps) exactly, so every clamped reset was scheduled one step
            # beyond the last drain that could see it and could NEVER fire --
            # reintroducing, inside the clamp, the very truncation the clamp was
            # added to remove.
            last_drainable = None
            if episode_step_budget is not None:
                last_drainable = int(episode_step_budget) - 1
                due = min(due, last_drainable)
            # V3-EXQ-944a FIX 2 -- DISTINCT SLOTS. `real()` only sets the
            # idempotent `_pending_phase_reset` flag, so two resets landing on
            # the SAME step yield ONE extra E3 tick, not two. 944's clamp
            # collapsed every late request onto one step, so a burst of events
            # near the episode tail produced a single tick while the ledger
            # counted them all as pending -- RATE_MATCHED's COUNT contract broke
            # exactly where the event rate was highest. Give each reset its own
            # step; if none is free inside the episode, DROP it and say so
            # rather than silently stacking.
            while due in pending:
                due += 1
            if last_drainable is not None and due > last_drainable:
                ledger.dropped += 1
            else:
                pending.append(due)
        # ARM_NO_RESET: request recorded, never executed.

    clock.phase_reset = patched

    def restore() -> None:
        clock.phase_reset = real

    def drain() -> None:
        """Fire any RATE_MATCHED reset whose scheduled step has arrived."""
        if arm != ARM_RATE_MATCHED or not pending:
            return
        due = min(pending)
        if due <= clock._global_step:
            pending.remove(due)
            ledger.executed += 1
            real()

    return restore, drain, pending


# --- The shared cell loop ----------------------------------------------------

def warmup(agent: REEAgent, env: CausalGridWorldV2, seed: int,
           episodes: int, steps: int, progress_every: int = 10) -> List[int]:
    """P0: prediction-loss warmup. No downstream head is trained on a latent in
    this lineage, so the phased-training protocol reduces to P0 -> P2 (frozen).
    """
    import torch.optim as optim
    opt = optim.Adam(agent.parameters(), lr=1e-3)
    harness = StepHarness(agent, env, train_mode=True, seed=seed)
    lens: List[int] = []
    for ep in range(episodes):
        _, obs = env.reset()
        agent.reset()
        harness.reset()
        n = 0
        for _ in range(steps):
            result = harness.step(obs)
            n += 1
            loss = agent.compute_prediction_loss() + agent.compute_e2_loss()
            if loss.requires_grad:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt.step()
            obs = result.next_obs_dict
            if result.done:
                break
        lens.append(n)
        if progress_every and (ep + 1) % progress_every == 0:
            # Deliberately NOT the runner's "ep N/M" shape: the runner takes its
            # denominator from that pattern and this loop is per-SEED, not per
            # seed x arm, so a matching print would corrupt episodes_per_run.
            print("  [warmup] seed=%d done=%d total=%d" % (seed, ep + 1, episodes),
                  flush=True)
    return lens


def measure_cell(agent: REEAgent, env: CausalGridWorldV2, arm: str, seed: int,
                 episodes: int, steps: int, sites: Dict[int, str],
                 zg: Any = None, progress_every: int = 10) -> Dict[str, Any]:
    """P2: frozen-weight measurement of one (arm, seed) cell.

    Returns the cell row, including the two DVs `what_would_answer` names:
    `straddle_frac` (an E3 cycle spans the salient event = a partial-integration
    artefact) and `mean_integration_lag` (env steps from the event to the next
    E3 planning window).
    """
    ledger = ResetLedger()
    rng = random.Random(seed * 7919 + 13)
    restore, drain, _pending = install_reset_policy(
        agent, arm, ledger, sites, rng, episode_step_budget=steps)
    harness = StepHarness(agent, env, train_mode=False, seed=seed)

    k = int(agent.clock.e3_steps_per_tick)
    ticks: Dict[int, List[int]] = {}
    harm_events: Dict[int, List[Tuple[int, float]]] = {}
    ep_lens: List[int] = []
    completion_signals: List[float] = []
    total_steps = 0
    elevated_steps = 0

    try:
        for ep in range(episodes):
            ledger.episode = ep
            _pending.clear()
            _, obs = env.reset()
            agent.reset()
            harness.reset()
            ticks[ep] = []
            harm_events[ep] = []
            n = 0
            for _ in range(steps):
                drain()
                result = harness.step(obs)
                n += 1
                total_steps += 1
                gstep = agent.clock._global_step
                if result.ticks.get("e3_tick"):
                    ticks[ep].append(gstep)
                harm = float(result.harm_signal)
                if harm < 0:
                    harm_events[ep].append((gstep, abs(harm)))
                cs = getattr(agent.beta_gate, "_last_completion_signal", None)
                if cs is not None:
                    completion_signals.append(float(cs))
                if agent.beta_gate.is_elevated:
                    elevated_steps += 1
                obs = result.next_obs_dict
                if result.done:
                    break
            ep_lens.append(n)
            if zg is not None:
                zg.observe(agent)
            if progress_every and (ep + 1) % progress_every == 0:
                print("  [train] %s seed=%d ep %d/%d" % (arm, seed, ep + 1, episodes),
                      flush=True)
    finally:
        restore()

    lags: List[int] = []
    post_event_harm: List[float] = []
    for ep, step, _cls in ledger.requests:
        tick_list = ticks.get(ep, [])
        idx = bisect_right(tick_list, step)
        if idx < len(tick_list):
            lags.append(tick_list[idx] - step)
        window = harm_events.get(ep, [])
        post_event_harm.append(
            sum(mag for (g, mag) in window if step < g <= step + k)
        )

    n_ticks = sum(len(v) for v in ticks.values())
    thr = getattr(agent.beta_gate, "_completion_release_threshold", None)
    coverage = {c: ledger.by_class.get(c, 0) for c in TRIGGER_CLASSES}
    coverage["unknown"] = ledger.by_class.get("unknown", 0)

    return {
        "arm": arm,
        "seed": seed,
        "e3_steps_per_tick": k,
        "episodes": episodes,
        "episode_lengths": ep_lens,
        "mean_episode_length": (sum(ep_lens) / len(ep_lens)) if ep_lens else 0.0,
        "total_steps": total_steps,
        "n_salient_events": len(ledger.requests),
        "trigger_coverage": coverage,
        "trigger_sites_fired": dict(ledger.by_site),
        "n_resets_executed": ledger.executed,
        # V3-EXQ-944a: execution accounting, so a RATE_MATCHED cell that
        # degraded toward NO_RESET is readable directly instead of being
        # inferred from a downstream rate deviation.
        "n_resets_requested": len(ledger.requests),
        "n_resets_dropped_unfired": ledger.dropped,
        "reset_execution_frac": (
            ledger.executed / len(ledger.requests)
        ) if ledger.requests else None,
        "n_e3_ticks": n_ticks,
        "e3_tick_rate": (n_ticks / total_steps) if total_steps else 0.0,
        "beta_elevated_frac": (elevated_steps / total_steps) if total_steps else 0.0,
        "completion_release_threshold": thr,
        "completion_signal_max": max(completion_signals) if completion_signals else None,
        "completion_signal_mean": (
            sum(completion_signals) / len(completion_signals)
        ) if completion_signals else None,
        "n_events_with_following_tick": len(lags),
        "mean_integration_lag": (sum(lags) / len(lags)) if lags else None,
        "straddle_frac": (
            sum(1 for l in lags if l > 1) / len(lags)
        ) if lags else None,
        "post_event_harm_mean": (
            sum(post_event_harm) / len(post_event_harm)
        ) if post_event_harm else None,
    }
