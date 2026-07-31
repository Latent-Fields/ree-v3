"""
q081_pair_reach_check -- cheap empirical pre-flight for Q-081's PAIR-SPECIFIC
reach requirement (substrate_queue.json sd_id=Q081-REACH-CHECK-PAIR-SPECIFIC).

WHY THIS EXISTS
----------------
Three consecutive Q-081 landmark-removal runs (V3-EXQ-824, 824a, 838) produced a
bit-identical RV(z_world, operating_mode) between the INTACT arm and every
manipulation arm, at every seed, despite `assert_behavioural_reach()` (this
module's sibling in `q081_landmark_removal.py`) reporting MET each time from
824a onward. The 2026-07-29 failure autopsy of 838 diagnosed why:
`assert_behavioural_reach` is a BLANKET FLAG CHECK -- it confirms that
`use_anchor_sets` and/or `use_per_region_vs` are configured True, which is
necessary but NOT sufficient. 824a proved `use_anchor_sets=True` gives reach to
z_world (via MECH-269 write_anchor) but not to operating_mode. 838 then also
enabled `use_per_region_vs=True` (the 824a autopsy's own recommended fix) and
STILL got a bit-identical result: that flag's reach path
(`apply_invalidation_broadcasts_to_regions -> vs_rollout_gate -> E3 ->
operating_mode`, agent.py:4686-4692) does not, empirically, carry variance to
operating_mode either.

The lesson both autopsies converge on: a config-level flag check cannot tell
you whether a SPECIFIC manipulation reaches a SPECIFIC measured pair -- only an
EMPIRICAL check on that pair can. This module is that check, built cheap enough
to run before every future full Q-081 recording run rather than after it.

THE MECHANISM THIS PROBE EXPLOITS
-----------------------------------
`SalienceCoordinator.tick()` (ree_core/cingulate/salience_coordinator.py:375)
computes `operating_mode` as a DETERMINISTIC softmax over logits built purely
from `self._input_signals` (a plain float dict) plus static config. Source
trace of every write site (ree_core/agent.py, SD-032b/c/d/035/037 injections +
the mode-governance-engagement block) gives the complete, closed set of named
keys that can ever appear in that dict -- `SALIENCE_SIGNAL_NAMES` below. This
gives a much cheaper equivalent test than waiting for the RV statistic on the
softmax OUTPUT across a full multi-seed, multi-episode recording run:

    operating_mode is a pure function of _input_signals (given static config).
    => If _input_signals never differs between two arms across a run,
       operating_mode CANNOT differ either -- no further computation needed.
    => If it DOES differ at some tick, operating_mode MAY differ from that
       tick onward (softmax over different logits) -- reach is confirmed.

So this probe inspects the PRECURSOR dict directly, at every tick, over a
SHORT untrained rollout, instead of computing an RV coefficient over a full
20-episode-warmup + 20-episode-recording x 5-seed x 4-arm pipeline (as 838 did)
and finding out only at the end that nothing moved. It also NAMES which
signal (if any) moved and at which tick -- strictly more actionable than
"rv_primary was bit-identical".

WHY UNTRAINED IS ENOUGH, WITH A CAVEAT
----------------------------------------
The question is WIRING reach (does any causal path exist from the boundary-
manipulation to a named salience signal), not learned behaviour. An untrained
agent's z_world still changes with actions, anchor writes still fire off live
(if randomly-initialised) latents, and `_input_signals` is populated by the
same code path regardless of whether E1/E2 have been trained. That premise
holds -- but empirically (measured while building this module, not assumed),
the FIRST lever in the causal chain -- the hippocampal event segmenter's
boundary trigger, a PE z-score threshold on a sliding window -- fires very
SPARSELY on an untrained, randomly-initialised network, with large seed-to-
seed variance (0 boundaries in 1200 untrained ticks at one random
initialisation, 47 at another, otherwise identical run). An untrained rollout
that happens to draw zero boundary events has NOTHING for the manipulation to
scramble, and any "no reach" verdict from it is VACUOUS rather than
informative -- see the NON-DEGENERACY GUARD in
`run_pair_specific_reach_probe`, which checks for this directly rather than
letting it pass silently. The practical consequence: `steps_per_episode`
needs to be generous enough (default 400, not a token handful) for at least
one boundary event to be likely, and even so a given run may need re-trying
at a different seed. This is still far cheaper than full training (no
backprop, no optimizer step, forward-pass rollout only), just not as tiny as
"a token handful of steps" might suggest before you have actually run it.

THIS PROBE DOES NOT ADJUDICATE Q-081
--------------------------------------
It answers one narrow, PRIOR question: "would a full run on this exact
manipulation and this exact config even have a chance of producing a
non-bit-identical result?" A future full driver should call
`assert_pair_specific_reach(..., strict=True)` (or
`run_pair_specific_reach_probe(...)`) BEFORE committing to the full pipeline.
If it raises, the full run would (per the confirmed 824a/838 mechanism) again
be non_contributory -- do not run it; either find a different manipulation
lever with confirmed reach to one of the named signals, or reframe the
measured pair onto a confirmed-reachable one (H2, per the
Q081-REACH-CHECK-PAIR-SPECIFIC substrate_queue.json entry).

VALIDATE-THE-NULL-BEFORE-USE (Q-081's own standing convention -- see
q081_surrogate.py and q081_landmark_removal.py). The comparison logic here is
tested in tests/contracts/test_q081_pair_reach_check.py against BOTH a
known-divergent pair of traces (proves the detector fires) and a known-
identical pair (proves it does not false-positive), so a persistently
bit-identical probe result cannot be mistaken for a broken detector.

Design record: this module implements the `implementation_hint` of the
Q081-REACH-CHECK-PAIR-SPECIFIC entry in
REE_assembly/evidence/planning/substrate_queue.json ("build a cheap low-episode
pre-flight reach probe that asserts at least one arm moves rv_primary on the
SPECIFIC measured pair, and gate any full run on it").

NO ree_core CHANGE. Like q081_landmark_removal.py and q081_surrogate.py, this
is an EXPERIMENT-LAYER diagnostic: it reads `agent.salience._input_signals`
(already a plain, stable dict attribute; see salience_coordinator.py:305-308,
:330-336) non-destructively and never mutates agent state or config. No new
REEConfig knob, no backward-compatibility surface.

ASCII-only output (repo rule). Stdlib only at module scope -- torch and
ree_core are imported lazily inside `run_pair_specific_reach_probe`, so this
module stays cheap to import for tooling and for the (torch-free) unit tests
of the comparison logic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "SALIENCE_SIGNAL_NAMES",
    "snapshot_salience_signals",
    "pair_specific_reach_report",
    "assert_pair_specific_reach",
    "run_pair_specific_reach_probe",
]


# Every named signal that can move SalienceCoordinator.tick()'s softmax logits
# or salience_aggregate. Source trace (ree_core/cingulate/salience_coordinator.py
# + every ree_core/agent.py call site that writes into it):
#
#   dacc_pe, dacc_foraging,       tick() kwarg dacc_bundle (SD-032b dACC bundle),
#     dacc_difficulty             agent.py ~6174/6262 (self._dacc_last_bundle)
#   drive_level                   tick() kwarg drive_level / per_axis_drive,
#                                  agent.py ~6172 (_effective_drive_level, SD-032e)
#   is_offline                    tick() kwarg is_offline, agent.py ~6173
#                                  (self.e1._offline_mode)
#   aic_salience                  update_signal, agent.py ~6178 (SD-032c, AIC)
#   pcc_stability                 update_signal, agent.py ~6195 (SD-032d, PCC)
#   cea_mode_prior, cea_fast_prime update_signal, agent.py ~6208/6212
#                                  (SD-035 / MECH-046 / MECH-074c, CeA)
#   override_signal                update_signal, agent.py ~6220 (SD-037 broadcast
#                                  override)
#   external_task_drive             update_signal, agent.py ~6261
#                                  (mode-governance-engagement, 2026-06-13)
#   pacc_autonomic                  REGISTERED (affinity_weights default,
#                                  salience_coordinator.py:177) but NO
#                                  `update_signal("pacc_autonomic", ...)` call
#                                  site exists anywhere in agent.py yet -- an
#                                  SD-032e stub, permanently 0.0 today. Listed
#                                  for completeness; cannot show reach until
#                                  SD-032e lands a writer.
#
# This is the CLOSED set as of this writing: SalienceCoordinator._input_signals
# (salience_coordinator.py:305-308) is seeded ONLY from config.affinity_weights
# / config.salience_weights keys, and update_signal() (only caller sites
# listed above) is the only way a value is ever WRITTEN. operating_mode is a
# pure function of exactly these values plus static config -- nothing else can
# move it. NOTE this constant is DOCUMENTATION/reference only -- the actual
# comparison in `pair_specific_reach_report` iterates over whatever keys are
# really present in each snapshot, so a future signal added without updating
# this list still gets checked; only the module docstring/error message would
# be stale. use_per_region_vs / use_anchor_sets / use_invalidation_trigger (the
# REACH_CONSUMERS flags checked by assert_behavioural_reach) are NOT in this
# list: they gate whether the boundary-event manipulation can reach anywhere
# in agent state at all, but per V3-EXQ-824a/838 their confirmed reach targets
# (z_world via write_anchor; vs_rollout_gate -> E3 candidate scoring) do not,
# empirically, write into any of the names below. That is precisely the gap
# this probe exists to catch cheaply instead of via a full recording run.
SALIENCE_SIGNAL_NAMES = frozenset({
    "dacc_pe",
    "dacc_foraging",
    "dacc_difficulty",
    "drive_level",
    "is_offline",
    "aic_salience",
    "pcc_stability",
    "cea_mode_prior",
    "cea_fast_prime",
    "override_signal",
    "external_task_drive",
    "pacc_autonomic",
})


def snapshot_salience_signals(agent: Any) -> Optional[Dict[str, float]]:
    """Read-only copy of `agent.salience._input_signals` at this tick.

    Returns None if the agent has no live SalienceCoordinator
    (`use_salience_coordinator=False`, or `agent.salience` is otherwise None).
    A probe against such an agent is meaningless -- there is no operating_mode
    to reach -- and callers must treat that as a distinct outcome from "reach
    not found" (see `run_pair_specific_reach_probe`, which asserts the
    coordinator is live before collecting any trace).

    Never mutates agent state: `getattr` reads only, and the returned dict is
    a fresh copy (mutating it cannot affect the agent).
    """
    sal = getattr(agent, "salience", None)
    if sal is None:
        return None
    raw = getattr(sal, "_input_signals", None)
    if raw is None:
        return None
    return {str(k): float(v) for k, v in raw.items()}


def pair_specific_reach_report(
    intact_trace: Sequence[Optional[Dict[str, float]]],
    manipulated_trace: Sequence[Optional[Dict[str, float]]],
    tol: float = 0.0,
) -> Dict[str, Any]:
    """Diff two per-tick salience-signal traces from an INTACT vs a manipulated arm.

    Pure function -- no agent, no torch, no I/O. `tol` is an ABSOLUTE
    per-signal tolerance; the default 0.0 means "any float64 divergence
    counts", matching the standing Q-081 convention of testing exact float
    equality across independently-simulated arms (q081_landmark_removal.py /
    V3-EXQ-824a/838: "exact float64 equality across independently-simulated
    seeds is the confirmed signature of the manipulation never reaching the
    measured pair, not landmark-invariance").

    Traces must be the same length (same number of ticks recorded); a length
    mismatch is a caller bug (mismatched episode/step counts between arms), so
    this raises rather than silently truncating. `run_pair_specific_reach_probe`
    truncates to a common length itself, with the drop recorded, before calling
    this function -- so a raise here means a DIRECT caller passed mismatched
    traces, not that the rollout ended early.

    A step where either arm's snapshot is None (coordinator not live that
    tick -- should not happen once collection has started, but tolerated
    defensively) is skipped rather than treated as a divergence.
    """
    if len(intact_trace) != len(manipulated_trace):
        raise ValueError(
            f"trace length mismatch: intact={len(intact_trace)} "
            f"manipulated={len(manipulated_trace)}. Both arms must run the "
            "identical number of ticks for a tick-aligned comparison; "
            "truncate to a common length before calling this function."
        )
    n = len(intact_trace)
    checked_names: set = set()
    divergent_first_tick: Dict[str, int] = {}
    for t in range(n):
        a = intact_trace[t]
        b = manipulated_trace[t]
        if a is None or b is None:
            continue
        checked_names.update(a.keys())
        checked_names.update(b.keys())
        for name in set(a.keys()) | set(b.keys()):
            if name in divergent_first_tick:
                continue
            va = a.get(name, 0.0)
            vb = b.get(name, 0.0)
            if abs(va - vb) > tol:
                divergent_first_tick[name] = t
    has_reach = bool(divergent_first_tick)
    first_tick = min(divergent_first_tick.values()) if has_reach else None
    return {
        "n_ticks_compared": n,
        "checked_signal_names": sorted(checked_names),
        "divergent_signals": dict(sorted(divergent_first_tick.items())),
        "has_pair_specific_reach": has_reach,
        "first_divergent_tick": first_tick,
        "tol": tol,
    }


def assert_pair_specific_reach(
    intact_trace: Sequence[Optional[Dict[str, float]]],
    manipulated_trace: Sequence[Optional[Dict[str, float]]],
    tol: float = 0.0,
    strict: bool = True,
) -> Dict[str, Any]:
    """Compute the reach report and, if strict, raise when reach is absent.

    This is the GATE a future full Q-081 driver should call BEFORE committing
    to a full multi-seed / multi-episode recording run: if it raises, a full
    run would (per the confirmed V3-EXQ-824a/838 mechanism) again produce a
    bit-identical, non_contributory RV(z_world, operating_mode) result -- do
    not run it. Self-route substrate_not_ready_requeue instead, or reframe the
    measured pair onto a confirmed-reachable one (H2, per the
    Q081-REACH-CHECK-PAIR-SPECIFIC substrate_queue.json implementation_hint).
    """
    report = pair_specific_reach_report(intact_trace, manipulated_trace, tol=tol)
    if strict and not report["has_pair_specific_reach"]:
        raise RuntimeError(
            "Q-081 pair-specific reach check FAILED: no named salience input "
            "signal (checked: " + ", ".join(sorted(SALIENCE_SIGNAL_NAMES)) + ") "
            f"ever diverged between the INTACT and manipulated arm across "
            f"{report['n_ticks_compared']} compared ticks. operating_mode is a "
            "deterministic function of exactly these signals (plus static "
            "config), so operating_mode CANNOT differ either -- a full "
            "recording run would reproduce V3-EXQ-824a/838's bit-identical, "
            "non_contributory result. Do not run the full experiment. Either "
            "(a) find a different manipulation lever with a confirmed causal "
            "path to one of these signals, or (b) reframe the measured pair "
            "onto a confirmed-reachable one (H2)."
        )
    return report


def run_pair_specific_reach_probe(
    n_episodes: int = 3,
    steps_per_episode: int = 400,
    seed: int = 0,
    env_size: int = 6,
    strict: bool = True,
    tol: float = 0.0,
    min_boundary_events: int = 1,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Cheap end-to-end pre-flight: build a tiny, UNTRAINED agent+env pair (the
    exact V3-EXQ-838 reach configuration: `use_anchor_sets=True`,
    `use_per_region_vs=True`, `q081_profile_kwargs()`), run INTACT then the
    pre-registered PRIMARY manipulation (`PRIMARY_MODE` = "iei_permute") over a
    SHORT rollout, and diff the per-tick salience-signal traces.

    Deliberately UNTRAINED (no P0 warmup) -- see the module docstring's
    "WHY UNTRAINED IS ENOUGH, WITH A CAVEAT" section. Still far cheaper than
    V3-EXQ-838's full pipeline (20 warmup + 20 recording episodes x 200 steps
    x 5 seeds x 4 arms, WITH backprop training) even at these defaults (3
    episodes x 400 steps, single seed, single manipulation arm, forward-pass
    only, no gradient step).

    NON-DEGENERACY GUARD (load-bearing -- see the module docstring; measured
    empirically, not assumed): the hierarchical event segmenter's boundary
    trigger is a PE z-score threshold on a sliding window, and on an UNTRAINED,
    randomly-initialised network this fires SPARSELY and with high
    seed-to-seed variance (measured: 0 boundaries in 1200 untrained ticks at
    one seed, 47 at another otherwise-identical seed). A rollout with ZERO
    true boundary events is one where the manipulation had NOTHING to
    scramble -- a "no reach" verdict from such a run is VACUOUS, not
    informative, exactly the failure mode MECH-466's own non-degeneracy guard
    exists to catch ("a segmenter emitting boundaries ... never, makes the
    comparison vacuous"). This function therefore checks the INTACT arm's
    `n_boundaries_true_total` (via `LandmarkScrambler.preservation_report()`)
    against `min_boundary_events` and reports `is_degenerate=True` (raising
    when strict) rather than silently returning a `has_pair_specific_reach`
    verdict that was never actually tested. A degenerate result should be
    retried with a larger `steps_per_episode`/`n_episodes` or a different
    `seed`, not read as "no reach".

    `assert_behavioural_reach(agent, strict=True)` is called first, as a
    cheaper-still precondition: an agent without a REACH_CONSUMERS flag live
    cannot possibly show pair-specific reach either, and that failure mode is
    detectable from config alone, with no rollout at all.

    `mode` selects the manipulation lever tested against INTACT, from
    `q081_landmark_removal.MODES` minus `"off"` (default: `None`, which resolves
    to `PRIMARY_MODE` = `"iei_permute"`, the pre-registered primary tested by
    V3-EXQ-838). Added 2026-07-31 for V3-EXQ-848's untried-lever reach scan
    (GOV-FANOUT-1 H1 leg: "reach scan across untried levers") -- the module's
    original single-lever form is preserved exactly when `mode` is omitted, so
    this is purely additive. `mode="jitter"` in particular has never been run
    against this pair at any level (RV or precursor): 824/824a/838 covered
    off/iei_permute/circular_shift/suppress only.

    Returns the same report shape as `assert_pair_specific_reach`, plus
    `behavioural_reach_precondition`, `is_degenerate`, `n_boundaries_true_total`,
    `dropped_ticks`, `n_episodes`, `steps_per_episode` and `seed`. Raises (from
    `assert_behavioural_reach`, the non-degeneracy guard, or
    `assert_pair_specific_reach`) when `strict=True` and a precondition, the
    non-degeneracy guard, or the reach check itself fails.

    MATCHED-ARM CONSTRUCTION (load-bearing, not a style choice): both arms run
    from a `copy.deepcopy()` of the SAME freshly-constructed agent, and each
    gets its own env built with the SAME seed (byte-identical starting RNG).
    This is the exact discipline V3-EXQ-838's `_run_arm_cell` uses (`agent =
    copy.deepcopy(agent_p0)`, `env = make_env(seed)` -- "fresh env,
    byte-identical starting RNG per arm"). Building two agents independently
    via two separate `REEAgent(cfg)` calls would let each draw different
    values from the evolving global torch RNG at weight-initialisation time,
    so ANY signal depending on initial weights (in practice: everything) would
    differ between "arms" for a reason that has nothing to do with the
    landmark manipulation -- a false positive this module exists to avoid
    manufacturing, not merely reproduce elsewhere.

    EPISODE-ALIGNED COMPARISON (load-bearing): because the manipulation can
    change action selection, the two arms' env trajectories can genuinely
    diverge and end an episode at different step counts (the same closed-loop
    point q081_landmark_removal.py's module docstring makes about
    `input_statistics_divergence`). Ticks are therefore compared by
    `(episode_index, step_index_within_episode)`, not by flat position in the
    trace list -- a flat-position comparison would silently compare one arm's
    episode 1 against the other arm's still-episode-0 tick once lengths
    diverge. `dropped_ticks` reports how many recorded ticks on each side had
    no matching (episode, step) key on the other side.
    """
    import copy
    import sys
    from pathlib import Path

    _repo_root = Path(__file__).resolve().parents[2]  # .../ree-v3
    _experiments_dir = _repo_root / "experiments"
    for _p in (str(_repo_root), str(_experiments_dir)):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    from ree_core.agent import REEAgent  # noqa: E402
    from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
    from ree_core.utils.config import REEConfig  # noqa: E402

    from _harness import StepHarness  # type: ignore  # noqa: E402

    from .q081_landmark_removal import (  # noqa: E402
        LandmarkRemovalConfig,
        LandmarkScrambler,
        MODES,
        PRIMARY_MODE,
        assert_behavioural_reach,
    )
    from .q081_profile import q081_profile_kwargs  # noqa: E402
    from .arm_fingerprint import reset_all_rng  # noqa: E402

    resolved_mode = PRIMARY_MODE if mode is None else mode
    if resolved_mode not in MODES or resolved_mode == "off":
        raise ValueError(
            f"mode={resolved_mode!r} is not a valid manipulation lever; must be "
            f"one of {sorted(set(MODES) - {'off'})}."
        )

    WORLD_DIM = SELF_DIM = 32
    ALPHA_WORLD = 0.9  # SD-008: z_world-fidelity-dependent experiments need >= 0.9

    def _make_env() -> Any:
        return CausalGridWorldV2(size=env_size, seed=seed)

    def _make_agent_template() -> Any:
        template_env = _make_env()
        flags = q081_profile_kwargs()
        flags["use_sleep_loop"] = False  # scope: waking streams only, matches 838
        cfg = REEConfig.from_dims(
            body_obs_dim=template_env.body_obs_dim,
            world_obs_dim=template_env.world_obs_dim,
            action_dim=template_env.action_dim,
            alpha_world=ALPHA_WORLD,
            world_dim=WORLD_DIM,
            self_dim=SELF_DIM,
            **flags,
        )
        # The exact V3-EXQ-838 reach configuration (824a's use_anchor_sets fix
        # PLUS 838's use_per_region_vs fix) -- both REACH_CONSUMERS flags live.
        cfg.hippocampal.use_anchor_sets = True
        cfg.hippocampal.use_per_region_vs = True
        return REEAgent(cfg)

    def _collect_trace(
        mode: str,
        agent: Any,
        donor_lookup: Optional[Dict[Tuple[int, int], Any]] = None,
    ) -> Tuple[
        Dict[Tuple[int, int], Optional[Dict[str, float]]],
        Dict[Tuple[int, int], Any],
        Dict[str, Any],
        Dict[str, Any],
    ]:
        # Reset EVERY RNG a rollout can touch (stdlib random, numpy, torch
        # +cuda, the harness module-level fallback) to the SAME seed before
        # each arm runs -- exactly V3-EXQ-838's `arm_cell` / `reset_all_rng`
        # discipline (arm_fingerprint.py). Without this, the two `_collect_trace`
        # calls run back-to-back against a single evolving global torch RNG
        # stream, so any torch-level randomness inside the agent's forward pass
        # (E3 candidate sampling, dropout, etc.) draws a DIFFERENT sequence for
        # the second arm than the first purely from call ORDER -- a confound
        # indistinguishable from genuine manipulation reach. reset_all_rng makes
        # each arm a pure function of `seed`, so the ONLY intended difference
        # between arms is the LandmarkScrambler's own manipulation.
        reset_all_rng(seed)
        env = _make_env()
        agent.eval()
        reach_precondition = assert_behavioural_reach(agent, strict=True)
        scrambler = LandmarkScrambler(LandmarkRemovalConfig(mode=mode, seed=seed))
        scrambler.attach(agent)
        harness = StepHarness(agent, env, train_mode=False, seed=seed)
        trace: Dict[Tuple[int, int], Optional[Dict[str, float]]] = {}
        for ep in range(n_episodes):
            _, obs_dict = env.reset()
            agent.reset()
            harness.reset()
            donor = donor_lookup.get((seed, ep)) if donor_lookup is not None else None
            scrambler.begin_episode(ep, n_steps=steps_per_episode, seed=seed, donor=donor)
            for step_idx in range(steps_per_episode):
                result = harness.step(obs_dict)
                trace[(ep, step_idx)] = snapshot_salience_signals(agent)
                obs_dict = result.next_obs_dict
                if result.done:
                    break
            scrambler.end_episode()
        donor_index = scrambler.donor_index()
        preservation = scrambler.preservation_report()
        scrambler.detach()
        return trace, donor_index, reach_precondition, preservation

    agent_template = _make_agent_template()
    # hippocampal._rng defaults to the `random` MODULE itself (module.py:156),
    # which copy.deepcopy cannot pickle. seed_replay_rng() installs a
    # picklable, per-seed random.Random in its place -- the same fix
    # V3-EXQ-838's run_seed() applies before its own deepcopy, for the same
    # reason (and it makes the hippocampal replay RNG reproducible and
    # identical across both arms as a side benefit).
    agent_template.hippocampal.seed_replay_rng(seed)
    agent_intact = copy.deepcopy(agent_template)
    agent_manipulated = copy.deepcopy(agent_template)

    intact_trace, donor_index, reach_precondition, intact_preservation = _collect_trace(
        "off", agent_intact
    )
    manipulated_trace, _, _, _manip_preservation = _collect_trace(
        resolved_mode, agent_manipulated, donor_lookup=donor_index
    )

    # NON-DEGENERACY GUARD -- see the module docstring "WHY UNTRAINED IS
    # ENOUGH, WITH A CAVEAT". A rollout where the INTACT arm never fired a
    # true boundary event gave the manipulation nothing to scramble; any
    # "no reach" verdict from it is vacuous, not informative.
    n_boundaries_true_total = int(
        intact_preservation.get("n_boundaries_true_total", 0)
    )
    is_degenerate = n_boundaries_true_total < min_boundary_events
    if is_degenerate and strict:
        raise RuntimeError(
            "Q-081 pair-specific reach probe is DEGENERATE, not a 'no reach' "
            f"result: the INTACT arm fired only {n_boundaries_true_total} true "
            f"boundary event(s) across {n_episodes} episodes x "
            f"{steps_per_episode} steps (need >= {min_boundary_events}). An "
            "untrained, randomly-initialised network's event segmenter fires "
            "sparsely and with high seed-to-seed variance, so a run with no "
            "boundaries gave the manipulation nothing to scramble -- this is "
            "not evidence of absent reach. Retry with a larger "
            "steps_per_episode/n_episodes or a different seed; do not read "
            "this as 'no pair-specific reach'."
        )

    common_keys = sorted(set(intact_trace) & set(manipulated_trace))
    aligned_intact = [intact_trace[k] for k in common_keys]
    aligned_manipulated = [manipulated_trace[k] for k in common_keys]
    dropped = {
        "intact": len(intact_trace) - len(common_keys),
        "manipulated": len(manipulated_trace) - len(common_keys),
    }

    report = assert_pair_specific_reach(
        aligned_intact, aligned_manipulated, tol=tol, strict=(strict and not is_degenerate)
    )
    report["behavioural_reach_precondition"] = reach_precondition
    report["is_degenerate"] = is_degenerate
    report["n_boundaries_true_total"] = n_boundaries_true_total
    report["min_boundary_events"] = min_boundary_events
    report["dropped_ticks"] = dropped
    report["n_episodes"] = n_episodes
    report["steps_per_episode"] = steps_per_episode
    report["seed"] = seed
    report["manipulation_mode"] = resolved_mode
    return report


if __name__ == "__main__":
    import json

    print(json.dumps(run_pair_specific_reach_probe(strict=False), indent=2, default=str))
