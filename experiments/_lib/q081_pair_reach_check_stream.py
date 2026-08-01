"""
q081_pair_reach_check_stream -- cheap empirical pre-flight for Q-081's
REFRAMED measured pair: RV(z_world, z_goal) [primary] / RV(z_world, z_harm_a)
[fallback]. Parallel build to `q081_pair_reach_check.py`
(Q081-REACH-CHECK-PAIR-SPECIFIC), which is hardcoded to
`agent.salience._input_signals` / operating_mode and does not generalise to
these streams as-is (see that module's own docstring; this file is the
generalisation it names as future work).

WHY THIS EXISTS
----------------
2026-08-01 /claim-synthesis (REE_assembly/evidence/planning/
claim_synthesis_Q-081_2026-08-01.md) retired RV(z_world, operating_mode) as
Q-081's operationalization: four structurally different manipulation families
(V3-EXQ-824/824a/838/849) found zero reach from any config lever to that pair,
and `salience_coordinator.py` traces to a closed named-scalar precursor set
with no wired edge from z_world or any landmark/boundary-derived quantity --
no lever will ever reach it. Reading `ree_core/agent.py` found two GENUINELY
different, already-coded, single-hop paths from the SAME landmark-manipulation
lever (`per_stream_vs` / `per_region_vs` staleness, via `use_anchor_sets` /
`use_per_region_vs`) into cross-stream targets:

  1. `vs_rollout_gate.gate_stream("z_goal", goal_state.z_goal, per_stream_vs,
     side="e1", ...)` (agent.py ~4895), gated on `goal.e1_goal_conditioned`
     (default True) and `goal_state.is_active()`. z_goal is a genuinely
     different subsystem (goal stream, not world-model stream) -- PRIMARY.
  2. `vs_rollout_gate.gate_stream("z_harm_a", agent._harm_a_prev,
     per_stream_vs, side="e2", ...)` (agent.py ~8332), gated on
     `agent.e2_harm_a is not None`. z_harm_a is also a genuinely different
     subsystem (affective/harm stream) -- FALLBACK, used only if the z_goal
     probe also comes back zero-reach.

Per the same discipline V3-EXQ-849 established for the retired pair: build a
cheap pre-flight reach probe for the SPECIFIC new pair BEFORE any full
multi-seed recording run, and gate the full run on it.

THE MECHANISM THIS PROBE EXPLOITS
-----------------------------------
Both target streams reach their consumer (E1 for z_goal, E2_harm_a's forward
model for z_harm_a) through `VsRolloutGate.gate_stream(...)`
(ree_core/regulators/vs_rollout_gate.py), a DETERMINISTIC function of
(stream_name, current_value, per_stream_vs, side, per_stream_staleness) plus
the gate's own snapshot cache. The landmark manipulation moves
`per_stream_vs` (and, when `use_staleness_lookup` is wired, the staleness
cache); if that movement never crosses the gate's per-stream threshold for
this stream, the gate's OUTPUT is bit-identical to the un-gated input on every
tick regardless of what the raw stream itself does. So this probe snapshots
BOTH the raw (pre-gate) stream value and the gate's actual output (the
gated/post-gate value E1 or E2_harm_a consumes) at every tick, over a SHORT
untrained rollout -- cheaper than waiting for an RV statistic on a downstream
readout across a full multi-seed pipeline, and strictly more informative for
a null (rules out reach at the wiring level, not just in one summary
statistic).

`_gate_value` (the gate's private, side-effect-free comparison) is called
directly rather than the public `gate()` / `gate_stream()` wrappers, which
also bump per-tick diagnostic counters (`_last_held_e1` / `_held_count_e1`)
as a side effect of being called from the real forward pass. This probe
constructs its own throwaway agents anyway (discarded after the probe), so
that side effect is functionally inert either way -- but reading `_gate_value`
keeps this module's stated guarantee ("reads agent state, never mutates it")
literally true rather than merely harmless-in-practice, matching
`q081_pair_reach_check.py`'s own framing of `agent.salience._input_signals`
as a plain, non-destructive read.

TWO INDEPENDENT NON-DEGENERACY GUARDS (this module's departure from the
operating_mode probe -- read before trusting a "no reach" verdict)
-------------------------------------------------------------------------
`q081_pair_reach_check.py` has ONE non-degeneracy guard: the untrained event
segmenter's boundary trigger fires sparsely, so a rollout with zero true
boundary events gives the manipulation nothing to scramble. That guard
applies here unchanged (same segmenter, same manipulation lever) -- but
BOTH target streams carry a SECOND, independent activation precondition the
operating_mode probe never had to consider, because `_input_signals` is
populated unconditionally every tick regardless of manipulation:

  - z_goal: `GoalState.z_goal` is a "slow-decay attractor" (goal.py) that is
    only `is_active()` once `agent.update_z_goal(benefit_exposure=...)` has
    fired at least once with `benefit_exposure > goal.benefit_threshold`
    (default 0.1). Before that, `_z_goal_input` is `None` every tick (agent.py
    ~4913: gated on `goal_state.is_active()`) and the E1 gate_stream call
    never runs at all -- there is no z_goal signal for the manipulation to
    reach, whether or not any boundary event fired. `StepHarness.step()`
    (experiments/_harness.py) already drives `update_z_goal` every tick from
    `obs_dict["benefit_exposure"]` (the canonical wiring this module reuses
    unmodified), so activation depends on the ENVIRONMENT producing a benefit
    pulse during the rollout, not on any code path this module owns.
  - z_harm_a: `agent._harm_a_prev` is set from `new_latent.z_harm_a` inside
    `sense()` (agent.py ~4544) whenever `use_affective_harm_stream=True` and
    `obs_harm_a` is supplied -- `StepHarness` supplies it via
    `sense_kwargs_from_obs()` every tick once the flag is on, so unlike
    z_goal this is NOT gated on a sparse environmental event and should
    populate from the agent's very first tick. The E2 gate_stream call
    additionally requires `agent.e2_harm_a is not None`
    (`use_e2_harm_a=True`, a config flag `q081_profile_kwargs()` does NOT
    set -- this module sets it explicitly for the z_harm_a target only) and
    `action is not None` (true from the first select_action() call onward).

A rollout where the target stream never activates gives a "no reach" verdict
that is vacuous for the SAME reason a zero-boundary rollout is: the
manipulation had nothing to act on. `run_pair_specific_stream_reach_probe`
checks this directly (`n_active_ticks_intact < min_active_ticks`) alongside
the inherited boundary-event guard, and reports `is_degenerate=True` (raising
when strict) with the applicable reason named, rather than silently reading
an all-None trace as "no reach".

NON-ADJUDICATION, SAME AS THE PARENT MODULE
----------------------------------------------
This probe answers one narrow, PRIOR question per target stream: "would a
full run on this exact manipulation and config even have a chance of
producing a non-bit-identical RV(z_world, <stream>) result?" It does not
adjudicate Q-081 Outcome A vs B. A future full driver should call
`assert_pair_specific_stream_reach(..., strict=True)` (or
`run_pair_specific_stream_reach_probe(..., strict=True)`) before committing to
a full pipeline; if it raises, do not run the full experiment -- either try a
different manipulation lever, or (for z_goal specifically) fall back to
z_harm_a; if BOTH targets come back zero-reach, this is out of scope for a
config-level probe (see the claim-synthesis doc's own escalation note: a
direct write-path substrate build is the next step, not another probe).

VALIDATE-THE-NULL-BEFORE-USE (Q-081's own standing convention). The
comparison logic (`pair_specific_reach_report`, imported from
`q081_pair_reach_check.py` rather than re-implemented -- it is a pure,
target-agnostic diff over named-float snapshot dicts) is already tested
there against both a known-divergent and a known-identical trace pair.
`tests/contracts/test_q081_pair_reach_check_stream.py` adds the
stream-specific coverage this module needs: the raw/gated snapshot reader,
both non-degeneracy guards, and a live end-to-end smoke per target stream.

Design record: REE_assembly/evidence/planning/claim_synthesis_Q-081_2026-08-01.md
section 3-5; REE_assembly/docs/claims/claims.yaml Q-081 "MEASURED-PAIR REFRAME"
note.

NO ree_core CHANGE. Like `q081_pair_reach_check.py`, this is an
EXPERIMENT-LAYER diagnostic: it reads agent/gate attributes non-destructively
(via the gate's pure `_gate_value` helper, never its mutating public
wrappers) and never mutates agent state or config beyond the throwaway
agent's own construction-time flags. No new REEConfig knob.

ASCII-only output (repo rule). Stdlib only at module scope -- torch and
ree_core are imported lazily inside `run_pair_specific_stream_reach_probe`,
matching the parent module's import discipline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from experiments._lib.q081_pair_reach_check import pair_specific_reach_report

__all__ = [
    "TARGET_STREAMS",
    "snapshot_stream_signal",
    "assert_pair_specific_stream_reach",
    "run_pair_specific_stream_reach_probe",
]


# Per-target-stream wiring. `side` and `gate_config_attr` mirror the exact
# agent.py call site named in the module docstring above. `extra_agent_flags`
# are config flags this module must set that q081_profile_kwargs() does not
# already cover (z_goal needs none extra; z_harm_a needs use_e2_harm_a, which
# is not part of the shared Q-081 recording profile -- see q081_profile.py's
# own Q081_FLAGS, which does not list it).
TARGET_STREAMS: Dict[str, Dict[str, Any]] = {
    "z_goal": {
        "side": "e1",
        "extra_agent_flags": {},
    },
    "z_harm_a": {
        "side": "e2",
        "extra_agent_flags": {"use_e2_harm_a": True},
    },
}


def _read_raw_stream(agent: Any, target: str) -> Optional[Any]:
    """Read the pre-gate value of `target` off `agent`, or None if its
    activation precondition is not yet met on this tick (see module
    docstring "TWO INDEPENDENT NON-DEGENERACY GUARDS"). Mirrors the exact
    precondition each agent.py call site checks before ever calling
    gate_stream, so a None here means "the real forward pass would not have
    consumed a value for this stream on this tick either" -- not a probe
    artefact.
    """
    if target == "z_goal":
        goal_state = getattr(agent, "goal_state", None)
        goal_cfg = getattr(agent.config, "goal", None)
        if goal_state is None or goal_cfg is None:
            return None
        if not getattr(goal_cfg, "e1_goal_conditioned", False):
            return None
        if not goal_state.is_active():
            return None
        return goal_state.z_goal
    if target == "z_harm_a":
        if getattr(agent, "e2_harm_a", None) is None:
            return None
        return getattr(agent, "_harm_a_prev", None)
    raise ValueError(f"target={target!r} not in TARGET_STREAMS: {sorted(TARGET_STREAMS)}")


def snapshot_stream_signal(agent: Any, target: str) -> Optional[Dict[str, float]]:
    """Read-only snapshot of `target`'s raw (pre-gate) and gated (post-gate,
    as actually consumed by E1/E2_harm_a) values at this tick, as a flat dict
    of per-dimension named floats -- the same shape
    `snapshot_salience_signals` (q081_pair_reach_check.py) returns for
    operating_mode's precursor dict, so `pair_specific_reach_report` diffs it
    unchanged.

    Returns None if `target`'s activation precondition is not met this tick
    (see `_read_raw_stream`) -- the tick contributes nothing to either arm's
    trace, exactly like a tick where `agent.salience` is None in the parent
    module. Never mutates agent state: reads `VsRolloutGate._gate_value`
    (pure -- see module docstring) rather than the public `gate_stream()`,
    and never calls `update_snapshots`.
    """
    spec = TARGET_STREAMS.get(target)
    if spec is None:
        raise ValueError(f"target={target!r} not in TARGET_STREAMS: {sorted(TARGET_STREAMS)}")
    raw_value = _read_raw_stream(agent, target)
    if raw_value is None:
        return None
    gate = getattr(agent, "vs_rollout_gate", None)
    if gate is not None:
        per_stream_vs = agent.hippocampal.per_stream_vs
        staleness = getattr(agent, "_vs_gate_staleness_cache", None) or None
        gated_value, _held = gate._gate_value(
            target, raw_value, per_stream_vs, spec["side"],
            per_stream_staleness=staleness,
        )
    else:
        gated_value = raw_value
    raw_flat = raw_value.detach().flatten().tolist()
    gated_flat = gated_value.detach().flatten().tolist()
    out: Dict[str, float] = {}
    for i, v in enumerate(raw_flat):
        out[f"{target}_raw_{i}"] = float(v)
    for i, v in enumerate(gated_flat):
        out[f"{target}_gated_{i}"] = float(v)
    return out


def assert_pair_specific_stream_reach(
    target: str,
    intact_trace: Sequence[Optional[Dict[str, float]]],
    manipulated_trace: Sequence[Optional[Dict[str, float]]],
    tol: float = 0.0,
    strict: bool = True,
) -> Dict[str, Any]:
    """Compute the reach report for `target` and, if strict, raise when reach
    is absent. The gate BEFORE committing to a full multi-seed recording run
    on RV(z_world, `target`): if it raises, that full run would (per the same
    mechanism V3-EXQ-824a/838 confirmed for operating_mode) reproduce a
    bit-identical, non_contributory result -- do not run it. For
    target="z_goal", fall back to target="z_harm_a" instead; if both raise,
    this is out of scope for a config-lever probe (see module docstring).
    """
    report = pair_specific_reach_report(intact_trace, manipulated_trace, tol=tol)
    if strict and not report["has_pair_specific_reach"]:
        raise RuntimeError(
            f"Q-081 pair-specific reach check FAILED for target={target!r}: "
            "no raw or gated dimension of this stream ever diverged between "
            f"the INTACT and manipulated arm across {report['n_ticks_compared']} "
            "compared ticks. A full recording run on RV(z_world, "
            f"{target}) would reproduce this bit-identical result. Do not run "
            "it. Either (a) find a different manipulation lever with a "
            f"confirmed causal path to {target}, or (b) if target='z_goal', "
            "fall back to target='z_harm_a' (or vice versa); if both are "
            "zero-reach, escalate per the claim-synthesis doc rather than "
            "queuing another probe."
        )
    return report


def run_pair_specific_stream_reach_probe(
    target: str = "z_goal",
    n_episodes: int = 3,
    steps_per_episode: int = 400,
    seed: int = 0,
    env_size: int = 6,
    strict: bool = True,
    tol: float = 0.0,
    min_boundary_events: int = 1,
    min_active_ticks: int = 1,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Cheap end-to-end pre-flight for RV(z_world, `target`): build a tiny,
    UNTRAINED agent+env pair at the exact V3-EXQ-838 reach configuration
    (`use_anchor_sets=True`, `use_per_region_vs=True`, `q081_profile_kwargs()`,
    plus `target`'s own `extra_agent_flags`), run INTACT then the
    pre-registered manipulation (default `PRIMARY_MODE` = "iei_permute") over
    a short rollout, and diff `target`'s raw+gated per-tick trace.

    MATCHED-ARM CONSTRUCTION, RNG discipline, and EPISODE-ALIGNED COMPARISON
    are identical to `q081_pair_reach_check.run_pair_specific_reach_probe` --
    see that function's docstring for the full rationale (deepcopy of one
    seeded template, reset_all_rng(seed) before each arm and before the
    template is built, tick keyed by (episode_index, step_index_within_episode)
    rather than flat position).

    TWO NON-DEGENERACY GUARDS (see module docstring): (1) inherited from the
    parent module -- the INTACT arm must fire at least `min_boundary_events`
    true boundary events, or the manipulation had nothing to scramble; (2)
    new to this module -- `target` must be active (non-None snapshot) on at
    least `min_active_ticks` of the INTACT arm's ticks, or the stream was
    never computed/gated at all this rollout. Either failure sets
    `is_degenerate=True` (raising when `strict`) with `degeneracy_reason`
    naming which guard failed; a degenerate result must be retried at a
    larger scale or different seed, never read as "no reach".

    Returns the same report shape as `assert_pair_specific_stream_reach`,
    plus `target`, `behavioural_reach_precondition`, `is_degenerate`,
    `degeneracy_reason`, `n_boundaries_true_total`, `n_active_ticks_intact`,
    `min_boundary_events`, `min_active_ticks`, `dropped_ticks`, `n_episodes`,
    `steps_per_episode`, `seed`, `manipulation_mode`.
    """
    import copy
    import sys
    from pathlib import Path

    if target not in TARGET_STREAMS:
        raise ValueError(f"target={target!r} not in TARGET_STREAMS: {sorted(TARGET_STREAMS)}")

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
        flags["use_sleep_loop"] = False  # scope: waking streams only, matches 824/824a/838/849
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
        # PLUS 838's use_per_region_vs fix) -- both REACH_CONSUMERS flags live,
        # so the manipulation reaches per_stream_vs the same way it did for
        # the retired (z_world, operating_mode) pair.
        cfg.hippocampal.use_anchor_sets = True
        cfg.hippocampal.use_per_region_vs = True
        for _flag_name, _flag_value in TARGET_STREAMS[target]["extra_agent_flags"].items():
            setattr(cfg, _flag_name, _flag_value)
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
        # Full RNG reset before each arm -- see q081_pair_reach_check.py's
        # identical block for why this is load-bearing, not a style choice.
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
                trace[(ep, step_idx)] = snapshot_stream_signal(agent, target)
                obs_dict = result.next_obs_dict
                if result.done:
                    break
            scrambler.end_episode()
        donor_index = scrambler.donor_index()
        preservation = scrambler.preservation_report()
        scrambler.detach()
        return trace, donor_index, reach_precondition, preservation

    # reset_all_rng BEFORE the template is built -- see
    # q081_pair_reach_check.py's identical fix (2026-08-01) and its
    # regression test for why agent construction must also be a pure
    # function of `seed`, not just the arm-vs-arm comparison.
    reset_all_rng(seed)
    agent_template = _make_agent_template()
    agent_template.hippocampal.seed_replay_rng(seed)
    agent_intact = copy.deepcopy(agent_template)
    agent_manipulated = copy.deepcopy(agent_template)

    intact_trace, donor_index, reach_precondition, intact_preservation = _collect_trace(
        "off", agent_intact
    )
    manipulated_trace, _, _, _manip_preservation = _collect_trace(
        resolved_mode, agent_manipulated, donor_lookup=donor_index
    )

    # GUARD 1 (inherited): the manipulation must have had real boundary
    # events to scramble.
    n_boundaries_true_total = int(
        intact_preservation.get("n_boundaries_true_total", 0)
    )
    boundary_degenerate = n_boundaries_true_total < min_boundary_events

    # GUARD 2 (new to this module): `target` must have been active
    # (non-None snapshot) on the INTACT arm for at least min_active_ticks.
    n_active_ticks_intact = sum(1 for v in intact_trace.values() if v is not None)
    activity_degenerate = n_active_ticks_intact < min_active_ticks

    is_degenerate = boundary_degenerate or activity_degenerate
    degeneracy_reason: Optional[str] = None
    if is_degenerate:
        reasons = []
        if boundary_degenerate:
            reasons.append(
                f"the INTACT arm fired only {n_boundaries_true_total} true "
                f"boundary event(s) across {n_episodes} episode(s) x "
                f"{steps_per_episode} step(s) (need >= {min_boundary_events}) "
                "-- an untrained, randomly-initialised network's event "
                "segmenter fires sparsely and with high seed-to-seed variance, "
                "so this run gave the manipulation nothing to scramble."
            )
        if activity_degenerate:
            reasons.append(
                f"target={target!r} was active (non-None snapshot) on only "
                f"{n_active_ticks_intact} of {len(intact_trace)} INTACT-arm "
                f"ticks (need >= {min_active_ticks}) -- its own activation "
                "precondition was rarely or never met this rollout, so there "
                "was little or nothing for the manipulation to reach even if "
                "a wired path exists."
            )
        degeneracy_reason = (
            "Q-081 pair-specific stream reach probe is DEGENERATE, not a 'no "
            "reach' result: " + " ".join(reasons) + " Retry with a larger "
            "steps_per_episode/n_episodes or a different seed; do not read "
            "this as 'no pair-specific reach'."
        )
        if strict:
            raise RuntimeError(degeneracy_reason)

    common_keys = sorted(set(intact_trace) & set(manipulated_trace))
    aligned_intact = [intact_trace[k] for k in common_keys]
    aligned_manipulated = [manipulated_trace[k] for k in common_keys]
    dropped = {
        "intact": len(intact_trace) - len(common_keys),
        "manipulated": len(manipulated_trace) - len(common_keys),
    }

    report = assert_pair_specific_stream_reach(
        target, aligned_intact, aligned_manipulated, tol=tol,
        strict=(strict and not is_degenerate),
    )
    report["target"] = target
    report["behavioural_reach_precondition"] = reach_precondition
    report["is_degenerate"] = is_degenerate
    report["degeneracy_reason"] = degeneracy_reason
    report["n_boundaries_true_total"] = n_boundaries_true_total
    report["n_active_ticks_intact"] = n_active_ticks_intact
    report["min_boundary_events"] = min_boundary_events
    report["min_active_ticks"] = min_active_ticks
    report["dropped_ticks"] = dropped
    report["n_episodes"] = n_episodes
    report["steps_per_episode"] = steps_per_episode
    report["seed"] = seed
    report["manipulation_mode"] = resolved_mode
    return report


if __name__ == "__main__":
    import json

    print(json.dumps(
        run_pair_specific_stream_reach_probe(target="z_goal", strict=False),
        indent=2, default=str,
    ))
