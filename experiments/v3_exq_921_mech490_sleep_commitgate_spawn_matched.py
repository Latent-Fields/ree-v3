"""
V3-EXQ-921 -- MECH-490 Sleep Commit-Gate Persistence: Spawn-Matched + Commit-Variance
Instrumented Sleep Ablation (V3-EXQ-913 successor)

Claims: MECH-490

EXPERIMENT_PURPOSE = "evidence"

SLEEP DRIVER: two-arm ablation, unchanged from V3-EXQ-913. ARM_WITH_SLEEP = K=10
multi-fire (SleepLoopManager, sleep_loop_episodes_K=10, fires roughly twice across this
run's 24-segment life). ARM_NO_SLEEP = K=never (use_sleep_loop=False -- SleepLoopManager
is never constructed, agent.sleep_loop stays None, _segment_boundary_consolidate()'s own
guard makes it a correct automatic no-op).

WHY THIS RUN. Routing source: chip-20260811-sleep-commitgate-exq-successor, from
sleep_transition_investigation_906_lineage_2026-08-10.md Section 13 (13a-13d) and
candidate claim MECH-490 (claims.yaml, registered 2026-08-11). That reanalysis re-scored
V3-EXQ-913's own `sleep_ablation_comparison` block (never previously scored against any
claim -- claim_ids=[] at run time) and found:
  (a) a weakly-suggestive, statistically underpowered signal (n=4 usable matched windows
      across 2 seeds, best sign-test p=0.25 for tortuosity);
  (b) a genuine, previously-unflagged confound -- agent spawn position differs between the
      WITH_SLEEP and NO_SLEEP arms at EVERY one of the 5 matched segment-index comparisons
      (5-11 cell Manhattan distances on a 12x12 board), even though hazard/resource LAYOUT
      was held constant via env.reset_to(). env.reset_to()'s own contract only guarantees
      layout continuity; spawn point was independently re-rolled per arm's own RNG stream
      each segment (V3-EXQ-913's _continuity_reset(), consuming env._rng, whose state at a
      boundary depends on that arm's whole prior stochastic history -- which differs
      between arms from segment 1 onward).

MECH-490's own `what_would_answer` (claims.yaml) states this experiment's own
non-degeneracy PRECONDITION explicitly: (a) spawn position matched across arms at each
compared segment index, and (b) E3Selector commit-variance/commit-gate-engagement logged
per matched window, so a behavioural signature can be tied to the mechanism rather than
left an unexplained correlation. This script builds exactly those two fixes on top of
V3-EXQ-913's own driver (reused, not rewritten -- see imports), plus scales seeds from
913's 2 to 6 (>=5, satisfying the sign-test sample-size floor 13b names as the actual
ceiling on 913's own read).

OUT OF SCOPE (Section 13d items 4-5, explicitly deferred to a LATER successor gated on
THIS run producing a real, well-powered signal): the transfer test to a structurally
different ecology, and the blinded-umpire cross-environment discriminability classifier.
Not built here.

SPAWN-MATCHING DESIGN (the actual fix). V3-EXQ-913's `_continuity_reset()` draws its
safe-spawn candidate from `env._rng.integers(...)` -- consuming that ARM's own live RNG
stream, which has already diverged between WITH_SLEEP and NO_SLEEP by segment 1 (different
agent action sequences, different hazard-drift draws, different sleep-consolidation calls
consuming RNG differently). This script replaces that with `_spawn_order_for_segment(seed,
segment_index, board_size)`: a candidate ORDER drawn from a private RNG stream keyed ONLY
on (seed, segment_index) -- never on env._rng, never on arm identity. Calling it for the
WITH_SLEEP arm now, or the NO_SLEEP arm later (the two arms still run sequentially, not in
lockstep -- see `run_seed_arm`/`run()` below, unchanged from 913's per-(seed,arm)
structure), returns the IDENTICAL candidate order either way. Each arm still safety-filters
that shared order against its OWN live occupied/hazard-field state (hazard positions can
drift slightly within a segment even under layout continuity -- an always-on, pre-existing
dynamic per 913's own docstring, orthogonal to the layout-continuity confound), so the two
arms are offered the identical candidate list but are not FORCIBLY identical -- this is
disclosed, not hidden: every matched comparison logs `spawn_position_matched` and
`spawn_manhattan_distance` computed from the ACTUAL chosen spawn in each arm, and the
non-degeneracy precondition below requires a >=80% match rate across all comparisons before
the run counts as informative for MECH-490 at all.

COMMIT-VARIANCE INSTRUMENTATION (the second fix). `E3TrajectorySelector.get_commitment_state()`
(e3_selector.py:3990, an existing public method already called elsewhere in agent.py, e.g.
agent.py:5390) returns `{precision, running_variance, commit_threshold, committed_now,
is_committed}` at any instant -- `is_committed` reflects a genuine `_committed_trajectory`
being walked (executing a pre-planned action sequence without per-tick CEM/softmax
re-scoring), which is exactly the mechanism MECH-490 hypothesises. This script reads it
once per ENV STEP (not once per E3 tick) and records the resulting sequence per matched
window. This is DELIBERATELY NOT the "diagnostics-latch re-read" sample-size-inflation
hazard the E3 selection-diagnostics literature warns about (`last_score_diagnostics` et al,
which are populated ONLY inside select() and therefore stale/latched on non-E3-tick steps):
`running_variance` / `is_committed` are persistent STATE on the selector, not a per-call
diagnostic snapshot, and PERSISTING across non-selection ticks is the literal definition of
"still committed" -- reading it every env step measures the agent's ACTUAL commitment status
at every timestep, which is the correct granularity to correlate against the trajectory
metrics (themselves computed at env-step granularity). No probe added to e3_selector.py or
agent.py; this is a read-only instrumentation call from the driver, using an existing public
API.

ACTION-SEQUENCE METRICS (reimplemented, not previously committed). `_action_sequence_stats`
computes reversal rate and action-run length directly from the per-step ACTION LABEL
(canonical inverse pairing 0<->1, 2<->3; action 4=stay has no inverse) -- distinct from
V3-EXQ-913's existing `_trajectory_organization_stats`, which measures SPATIAL turning angle
and tortuosity (conflates a 90-degree turn with a full reversal). Reimplemented from
sleep_transition_investigation_906_lineage_2026-08-10.md Section 13b's method description
(that document's own script was not committed). NOTE: `world_rule_shift_enabled=True`
(250-tick interval, inherited unchanged from V3-EXQ-913/906b) periodically permutes the
live action-ID -> spatial-direction map, so these action-ID-level metrics measure
POLICY-OUTPUT-SEQUENCE structure (does the agent keep re-emitting the same/inverse discrete
action label), not necessarily literal spatial backtracking, for any window straddling a
rule-shift.

GOV-REUSE-1 (Step 2.4). Decisive readout: a matched (spawn AND layout) WITH_SLEEP-vs-
NO_SLEEP delta in action-run coherence (mean_run_length, reversal_rate, turning entropy,
tortuosity) that COVARIES with commit-gate engagement rate, across >=5 seeds. Searched
V3-EXQ-913's own manifest (the only run with a sleep_ablation_comparison block) -- it
carries neither the spawn-matching fix nor any commit-gate instrumentation, so this
readout is not recorded there and not derivable by reanalysis (a genuinely different
substrate_hash / manipulation, not merely a re-read of existing fields). Not recoverable
-> a run is required.

Re-derive brake (2.5b): MECH-490 was registered 2026-08-11 (today) with zero prior
`/failure-autopsy` targets (grepped `failure_autopsy_*.json` for "MECH-490": 0 hits) --
count=0 < threshold 2, does not fire; this is the FIRST experiment against this claim.
Substrate-path overlap gate (2.5c): checked substrate_queue.json for open `corrupting`/
`degrading` entries touching e3_selector.py, causal_grid_world.py, or the
906b/665/664/scaffolded_sd054/913 driver modules this script imports -- none found.

Step 2.5a empirical probe (this session, throwaway script, not committed): instantiated a
fresh (untrained) REEAgent + eval env via the same `_build_env`/`_make_config` helpers this
script imports, stepped it 20 ticks, and called `agent.e3.get_commitment_state()` each
tick. Confirmed the method is reachable and returns the documented dict shape
({precision, running_variance, commit_threshold, committed_now, is_committed}) with no
error. `running_variance` was flat at `config.precision_init` (0.5) and `is_committed` was
False throughout -- EXPECTED and not a wiring-gap finding: `_running_variance` only moves
via `update_running_variance()` calls tied to trained prediction-error feedback, and this
probe used a fresh, UNTRAINED agent (no curriculum). The real driver below trains through
the full curriculum (`_run_curriculum`, unchanged from 913) before eval, exactly as 913
does for the same substrate -- commit-gate engagement varying post-training is already the
well-established basis of 913's own `sleep_cycles_fired`/commit-gate machinery, not a new
premise introduced here.

Ethics preflight (Step 2.6, descriptive, non-enforced): involves_negative_valence=false,
involves_suffering_like_state=false, involves_self_model=false,
involves_inescapability_or_helplessness=false, involves_offline_replay_over_harm=false,
involves_social_mind_or_language=false, involves_human_data_or_clinical_context=false,
decision: allow. Unchanged from V3-EXQ-913 -- same substrate, same 24-segment life length,
no new welfare-relevant mechanism introduced (only read-only instrumentation added).

Output:
  evidence/experiments/v3_exq_921_mech490_sleep_commitgate_spawn_matched/
    v3_exq_921_mech490_sleep_commitgate_spawn_matched_<ts>.json               (manifest)
    v3_exq_921_mech490_sleep_commitgate_spawn_matched_<ts>_episode_log.json   (fishtank feed)

Estimated runtime: 6 seeds x 2 arms = 12 (seed, arm) combinations, each TRAIN_TOTAL_EPS
curriculum training episodes + 24 segments x up to 500 eval steps. V3-EXQ-913 (4 cells)
estimated ~150 min; this run is 3x the cells -> order of magnitude ~450 min. machine_affinity
any (cloud), priority accordingly moderate (see queue entry note).
"""

import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from experiments.scaffolded_sd054_onboarding import ScaffoldedSD054OnboardingScheduler, _build_env
from experiments.v3_exq_665_curriculum_affective_fishtank_showcase import (
    _make_scaffold_cfg,
    _run_curriculum,
)
from experiments.v3_exq_664_affective_fishtank_showcase import (
    _read_affect,
    _classify_mode,
    _get_reef_cells,
    _obs_harm,
    _obs_harm_a,
    _obs_harm_history,
    _action_to_onehot,
)
from experiments.v3_exq_906b_full_stack_observational_fishtank import (
    _make_config,
    _safe_reset,
    _segment_boundary_consolidate,
    _env_config_snapshot,
    TRAIN_TOTAL_EPS,
    CORE_CHANNELS,
    STD_FLOOR,
)
from experiments.v3_exq_913_developmental_ecology_fishtank import (
    ARM_WITH_SLEEP,
    ARM_NO_SLEEP,
    ARMS,
    DEV_NUM_RESOURCES,
    LOCAL_WINDOW_STEPS,
    SAFE_SPAWN_ATTEMPTS,
    _build_dev_eval_env,
    _make_config_for_arm,
    _trajectory_organization_stats,
    _within_life_development,
    _lifetime_affective_occupancy,
    _zone_habitat_stats,
)
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._metrics import check_degeneracy


EXPERIMENT_TYPE    = "v3_exq_921_mech490_sleep_commitgate_spawn_matched"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS: List[str] = ["MECH-490"]

# ---- unchanged from V3-EXQ-913 (module docstring "WHY THIS RUN")
EVAL_EPISODES = 24
EVAL_STEPS    = 500
SEEDS_DEFAULT = [0, 1, 2, 3, 4, 5]   # 913 used 2; MECH-490's forward plan item 3 asks
                                     # for >=5-8 -- 6 is the chosen midpoint.

# Pre-registered thresholds (Step 3 "thresholds defined in the script, not inferred
# post-hoc"). MECH-490's own non-degeneracy precondition, operationalized:
SPAWN_MATCH_RATE_FLOOR = 0.8   # >=80% of matched comparisons must share the identical
                                # chosen spawn cell, or the run does not count as
                                # informative for MECH-490 at all (non_degenerate=False).
SIGN_TEST_ALPHA = 0.05         # MECH-490 confirming-signature threshold.
MIN_SEEDS_FOR_CONFIRM = 5      # MECH-490 confirming-signature sample-size floor.
COVARY_R_FLOOR = 0.3           # |Pearson r| floor for "covaries" in the confirming
                                # signature's part (ii); exploratory given small n, stated
                                # as such in the manifest, never treated as its own p<0.05
                                # test (MECH-490's what_would_answer names no such test).


def _spawn_order_for_segment(seed: int, segment_index: int, board_size: int) -> List[Tuple[int, int]]:
    """MECH-490 spawn-confound fix. Deterministic, ARM-INDEPENDENT candidate spawn order
    for (seed, segment_index): a private RNG stream keyed ONLY on (seed, segment_index),
    never on env._rng (whose state at a boundary depends on that arm's whole prior
    stochastic history and therefore differs between arms -- V3-EXQ-913's confound,
    sleep_transition_investigation_906_lineage_2026-08-10.md Section 13b). Calling this
    for either arm, in either execution order, returns the identical candidate order."""
    rng = np.random.default_rng([int(seed), int(segment_index), 0x4D454348])
    interior = [(i, j) for i in range(1, board_size - 1) for j in range(1, board_size - 1)]
    order = rng.permutation(len(interior))
    return [interior[int(k)] for k in order]


def _matched_continuity_reset(
    env: CausalGridWorldV2,
    hazard_positions: List[Tuple[int, int]],
    resource_positions: List[Tuple[int, int]],
    segment_index: int,
    seed: int,
    max_attempts: int = SAFE_SPAWN_ATTEMPTS,
) -> Tuple[Any, Dict, int, bool, Tuple[int, int]]:
    """LAYOUT CONTINUITY (V3-EXQ-913's env.reset_to() mechanics, unchanged) + SPAWN
    MATCHING (this successor's fix): the candidate order comes from
    _spawn_order_for_segment(seed, segment_index, ...) -- shared across both arms --
    instead of env._rng. Each arm still safety-filters the shared order against its OWN
    live occupied/hazard-field state (hazard positions can drift slightly by segment > 1,
    an always-on pre-existing dynamic per 913's own docstring, orthogonal to layout
    continuity), so the two arms are OFFERED the identical candidate list but are not
    forcibly identical -- callers must check the actually-chosen cell (5th return value)
    to know whether the two arms actually matched at this boundary.
    Returns (flat_obs, obs_dict, attempts, exhausted, chosen)."""
    occupied = {tuple(int(c) for c in h) for h in hazard_positions} | \
               {tuple(int(c) for c in r) for r in resource_positions}
    order = _spawn_order_for_segment(seed, segment_index, env.size)
    candidates = [c for c in order if c not in occupied]
    if not candidates:
        candidates = [(1, 1)]

    idx = 0
    chosen = candidates[idx]
    flat_obs, obs_dict = env.reset_to(chosen, hazard_positions, resource_positions)
    attempts = 1
    while (attempts < max_attempts
           and float(env.hazard_field[chosen[0], chosen[1]]) >= env.proximity_approach_threshold
           and idx + 1 < len(candidates)):
        idx += 1
        chosen = candidates[idx]
        flat_obs, obs_dict = env.reset_to(chosen, hazard_positions, resource_positions)
        attempts += 1
    exhausted = bool(
        float(env.hazard_field[chosen[0], chosen[1]]) >= env.proximity_approach_threshold
    )
    return flat_obs, obs_dict, attempts, exhausted, chosen


# Canonical action-ID inverse pairing (module docstring "ACTION-SEQUENCE METRICS").
# Action 4 = stay, has no inverse.
_ACTION_INVERSE = {0: 1, 1: 0, 2: 3, 3: 2}


def _action_sequence_stats(steps: List[Dict[str, Any]], window: Optional[int] = None) -> Dict[str, Any]:
    """Action-LABEL-level reversal rate / run length -- distinct from
    _trajectory_organization_stats's SPATIAL turning-angle/tortuosity measures (module
    docstring "ACTION-SEQUENCE METRICS"). Reimplemented from
    sleep_transition_investigation_906_lineage_2026-08-10.md Section 13b's method
    description; that document's own script was not committed."""
    seq = steps[:window] if window is not None else steps
    actions = [int(s["action"]) for s in seq]
    n = len(actions)
    out: Dict[str, Any] = {"n_actions": n}
    if n < 2:
        return out
    n_reversals = 0
    n_transitions = 0
    run_lengths: List[int] = []
    run_len = 1
    for i in range(1, n):
        prev_a, cur_a = actions[i - 1], actions[i]
        n_transitions += 1
        if _ACTION_INVERSE.get(prev_a) == cur_a:
            n_reversals += 1
        if cur_a == prev_a:
            run_len += 1
        else:
            run_lengths.append(run_len)
            run_len = 1
    run_lengths.append(run_len)
    out.update({
        "reversal_rate": (n_reversals / n_transitions) if n_transitions else None,
        "n_reversals": n_reversals,
        "n_transitions": n_transitions,
        "mean_run_length": float(np.mean(run_lengths)) if run_lengths else None,
        "max_run_length": int(max(run_lengths)) if run_lengths else None,
        "n_action_runs": len(run_lengths),
    })
    return out


def _commit_gate_stats(commit_states: List[Dict[str, Any]]) -> Dict[str, Any]:
    """MECH-490 mechanistic instrumentation summary for one window (module docstring
    "COMMIT-VARIANCE INSTRUMENTATION"). `commit_states` is the per-env-step sequence of
    agent.e3.get_commitment_state() reads already collected by the caller."""
    n = len(commit_states)
    out: Dict[str, Any] = {"n_ticks": n}
    if n == 0:
        return out
    rv = [float(s["running_variance"]) for s in commit_states]
    committed_flags = [bool(s["is_committed"]) for s in commit_states]
    out.update({
        "mean_running_variance": float(np.mean(rv)),
        "commit_gate_engagement_rate": float(np.mean(committed_flags)),
        "n_committed_ticks": int(sum(committed_flags)),
    })
    return out


def _continuous_life_run(
    agent: REEAgent, env: CausalGridWorldV2, num_episodes: int,
    steps_per_episode: int, seed: int, arm: str,
) -> Dict[str, Any]:
    """LAYOUT-CONTINUOUS multi-segment observation of the SAME agent, adapted from
    V3-EXQ-913's _continuous_life_run -- boundary logic replaced with
    _matched_continuity_reset() (module docstring "SPAWN-MATCHING DESIGN"), plus per-tick
    commit-gate state collection and action-sequence-stats computation (module docstring
    "COMMIT-VARIANCE INSTRUMENTATION" / "ACTION-SEQUENCE METRICS")."""
    device     = agent.device
    action_dim = env.action_dim
    episodes_log: List[Dict] = []
    chan_vals: Dict[str, List[float]] = {
        k: [] for k in ["z_harm_s", "z_harm_un", "z_harm_a", "drive", "z_goal",
                        "vigor", "override", "z_block", "excite", "dread"]
    }
    freeze_fires = 0
    block_steps  = 0
    sleep_cycles_fired = 0
    total_spawn_retries = 0
    spawn_exhausted_segments = 0
    n_continuity_resets = 0
    canonical_hazard_positions: List[Tuple[int, int]] = []
    prev_cycle_history_len = len(getattr(getattr(agent, "sleep_loop", None), "_cycle_history", []) or [])

    if getattr(agent, "pag_freeze_gate", None) is not None:
        try:
            agent.pag_freeze_gate.config.duration_input_threshold = 1e9
        except Exception:
            pass

    agent.eval()

    z_world_prev = None
    action_prev  = None
    z_self_prev  = None
    prev_surprise_write_count = int(getattr(agent, "_surprise_write_count", 0))

    for ep_idx in range(num_episodes):
        sleep_cycle_detail = None
        if ep_idx == 0:
            flat_obs, obs_dict, spawn_attempts, spawn_exhausted = _safe_reset(env)
            spawn_pos = (int(env.agent_x), int(env.agent_y))
            agent.reset()
            sleep_fired_this_boundary = False
            canonical_hazard_positions = [tuple(int(c) for c in h) for h in env.hazards]
        else:
            hazard_positions   = [tuple(int(c) for c in h) for h in env.hazards]
            resource_positions = [tuple(int(c) for c in r) for r in env.resources]
            flat_obs, obs_dict, spawn_attempts, spawn_exhausted, spawn_pos = _matched_continuity_reset(
                env, hazard_positions, resource_positions, segment_index=ep_idx, seed=seed
            )
            n_continuity_resets += 1
            cycle_history_before = len(getattr(getattr(agent, "sleep_loop", None), "_cycle_history", []) or [])
            _segment_boundary_consolidate(agent)
            cycle_history_after_list = getattr(getattr(agent, "sleep_loop", None), "_cycle_history", []) or []
            sleep_fired_this_boundary = len(cycle_history_after_list) > cycle_history_before
            if sleep_fired_this_boundary:
                sleep_cycle_detail = dict(cycle_history_after_list[-1])
        total_spawn_retries += (spawn_attempts - 1)
        if spawn_exhausted:
            spawn_exhausted_segments += 1

        cycle_history = getattr(getattr(agent, "sleep_loop", None), "_cycle_history", []) or []
        if len(cycle_history) > prev_cycle_history_len:
            sleep_cycles_fired += len(cycle_history) - prev_cycle_history_len
        prev_cycle_history_len = len(cycle_history)

        ep_steps: List[Dict] = []
        commit_states_this_segment: List[Dict[str, Any]] = []
        current_hazards   = [list(h) for h in env.hazards]
        current_resources = [list(r) for r in env.resources]
        reef_cells     = _get_reef_cells(env)
        reef_cells_set = getattr(env, "_reef_cells", set())
        zone_map = getattr(env, "_zone_map", None)
        prev_in_reef   = False
        segment_done_cause = ""

        for step_idx in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_h     = _obs_harm(obs_dict)
            obs_h_a   = _obs_harm_a(obs_dict)
            obs_h_h   = _obs_harm_history(obs_dict)
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world,
                                     obs_harm=obs_h, obs_harm_a=obs_h_a, obs_harm_history=obs_h_h)
                if z_self_prev is not None and action_prev is not None:
                    agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())
                ticks    = agent.clock.advance()
                e1_prior = (agent._e1_tick(latent) if ticks.get("e1_tick", False)
                            else torch.zeros(1, agent.config.latent.world_dim, device=device))
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                drive_level      = REEAgent.compute_drive_level(obs_body)
                benefit_exposure = max(0.0, float(obs_dict.get("benefit_exposure", 0.0)))
                agent.update_z_goal(benefit_exposure=benefit_exposure, drive_level=drive_level)
                action = agent.select_action(candidates, ticks, temperature=1.0)
                if action is None:
                    action = _action_to_onehot(random.randint(0, action_dim - 1), action_dim, device)
                    agent._last_action = action
                # MECH-490 commit-gate instrumentation (module docstring
                # "COMMIT-VARIANCE INSTRUMENTATION"): a per-env-step STATE read, not a
                # per-select() diagnostics-latch re-read -- see that section for why the
                # two are not the same hazard.
                commit_states_this_segment.append(dict(agent.e3.get_commitment_state()))

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            with torch.no_grad():
                residue_metrics = agent.update_residue(float(harm_signal))
            residue_surprise_write_fired = (
                int(getattr(agent, "_surprise_write_count", 0)) > prev_surprise_write_count
            )
            prev_surprise_write_count = int(getattr(agent, "_surprise_write_count", 0))

            if info.get("env_drift_occurred", False):
                current_hazards   = [list(h) for h in env.hazards]
                current_resources = [list(r) for r in env.resources]

            agent_pos = (int(env.agent_x), int(env.agent_y))
            in_reef   = agent_pos in reef_cells_set
            blocked   = bool(info.get("action_blocked_this_step", False))
            done_cause = str(info.get("done_cause", "") or "")
            if blocked:
                block_steps += 1
            if done_cause:
                segment_done_cause = done_cause

            affect = _read_affect(agent, latent, obs_body)
            if affect["freeze"]:
                freeze_fires += 1
            for k, lst in chan_vals.items():
                v = affect.get(k)
                if isinstance(v, (int, float)) and v is not None:
                    lst.append(float(v))

            z_harm_s = affect["z_harm_s"] if affect["z_harm_s"] is not None else 0.0
            world_change_norm = (float((latent.z_world - z_world_prev).norm().item())
                                 if z_world_prev is not None else 0.0)
            mode = _classify_mode(z_harm_s, world_change_norm, float(harm_signal),
                                  in_reef, affect["freeze"], affect["z_block"])
            if blocked:
                step_transition = "action_blocked"
            elif in_reef and not prev_in_reef:
                step_transition = "reef_entry"
            elif not in_reef and prev_in_reef:
                step_transition = "reef_exit"
            else:
                step_transition = info.get("transition_type", "none")

            zone_at_agent = (int(zone_map[agent_pos[0], agent_pos[1]])
                             if zone_map is not None else -1)
            hazard_field_at_agent = (float(env.hazard_field[agent_pos[0], agent_pos[1]])
                                     if getattr(env, "use_proxy_fields", False) else None)
            resource_field_at_agent = (float(env.resource_field[agent_pos[0], agent_pos[1]])
                                       if getattr(env, "use_proxy_fields", False) else None)
            nearest_hazard_dist = (
                min(abs(agent_pos[0] - h[0]) + abs(agent_pos[1] - h[1]) for h in current_hazards)
                if current_hazards else None
            )

            ep_steps.append({
                "t": step_idx, "pos": list(agent_pos),
                "action": int(action.argmax(dim=-1).item()),
                "harm_signal": float(harm_signal),
                "z_harm_norm": z_harm_s,
                "z_harm_s": affect["z_harm_s"], "z_harm_un": affect["z_harm_un"],
                "z_harm_a": affect["z_harm_a"],
                "drive": affect["drive"], "z_goal": affect["z_goal"],
                "vigor": affect["vigor"], "override": affect["override"],
                "z_block": affect["z_block"], "freeze": affect["freeze"],
                "excite": affect["excite"], "dread": affect["dread"],
                "surprise": affect["surprise"],
                "mode": mode, "transition_type": step_transition,
                "health": float(info.get("health", 1.0)),
                "energy": float(info.get("energy", 1.0)),
                "harm_event": float(harm_signal) < 0,
                "n_cands": len(candidates),
                "in_reef": in_reef,
                "action_blocked": blocked,
                "done_cause": done_cause,
                "zone_at_agent": zone_at_agent,
                "hazard_field_at_agent": hazard_field_at_agent,
                "resource_field_at_agent": resource_field_at_agent,
                "nearest_hazard_dist": nearest_hazard_dist,
                "n_resources_remaining": len(current_resources),
                "residue_surprise": (residue_metrics.get("mech205_surprise")
                                     if isinstance(residue_metrics, dict) else None),
                "residue_write_fired": bool(residue_surprise_write_fired),
            })

            prev_in_reef = in_reef
            z_self_prev  = latent.z_self.detach()
            z_world_prev = latent.z_world.detach()
            action_prev  = action.detach()
            if done:
                break

        traj_local = _trajectory_organization_stats(ep_steps, current_hazards, window=LOCAL_WINDOW_STEPS)
        traj_full  = _trajectory_organization_stats(ep_steps, current_hazards, window=None)
        act_local  = _action_sequence_stats(ep_steps, window=LOCAL_WINDOW_STEPS)
        act_full   = _action_sequence_stats(ep_steps, window=None)
        cg_local   = _commit_gate_stats(commit_states_this_segment[:LOCAL_WINDOW_STEPS])
        cg_full    = _commit_gate_stats(commit_states_this_segment)
        zone_counts: Dict[str, int] = {}
        for s in ep_steps:
            z = s["zone_at_agent"]
            zone_counts[str(z)] = zone_counts.get(str(z), 0) + 1

        episodes_log.append({
            "ep": ep_idx,
            "initial_hazards":   [list(h) for h in env.hazards],
            "initial_resources": [list(r) for r in env.resources],
            "reef_cells": reef_cells, "steps": ep_steps,
            "done_cause": segment_done_cause,
            "realized_steps": len(ep_steps),
            "sleep_cycle_fired_before_this_segment": bool(sleep_fired_this_boundary),
            "sleep_cycle_detail": sleep_cycle_detail,
            "spawn_safe_attempts": int(spawn_attempts),
            "spawn_safe_exhausted": bool(spawn_exhausted),
            "spawn_pos": list(spawn_pos),
            "layout_continuity_reset": bool(ep_idx > 0),
            "hazard_positions_match_canonical": bool(
                {tuple(h) for h in env.hazards} == set(canonical_hazard_positions)
            ) if ep_idx > 0 else True,
            "zone_step_counts": zone_counts,
            "trajectory_local_window": traj_local,
            "trajectory_full_segment": traj_full,
            "action_sequence_local_window": act_local,
            "action_sequence_full_segment": act_full,
            "commit_gate_local_window": cg_local,
            "commit_gate_full_segment": cg_full,
        })
        print(f"  [eval] seed={seed} arm={arm} ep {ep_idx+1}/{num_episodes} steps={len(ep_steps)} "
              f"done_cause={segment_done_cause or '(ran to end-of-loop)'} "
              f"sleep_fired={sleep_fired_this_boundary} "
              f"spawn_attempts={spawn_attempts} spawn_pos={spawn_pos}", flush=True)

    chan_std  = {k: (float(np.std(v)) if len(v) >= 2 else 0.0) for k, v in chan_vals.items()}
    chan_mean = {k: (float(np.mean(v)) if v else 0.0) for k, v in chan_vals.items()}
    return {
        "episodes": episodes_log, "chan_std": chan_std, "chan_mean": chan_mean,
        "freeze_fires": freeze_fires, "block_steps": block_steps,
        "sleep_cycles_fired": sleep_cycles_fired,
        "eval_steps": int(sum(len(e["steps"]) for e in episodes_log)),
        "total_spawn_retries": total_spawn_retries,
        "spawn_exhausted_segments": spawn_exhausted_segments,
        "n_continuity_resets": n_continuity_resets,
        "canonical_hazard_positions": canonical_hazard_positions,
    }


def _sign_test_p(deltas: List[Optional[float]]) -> Dict[str, Any]:
    """Exact two-sided sign test on delta signs (ties/None excluded), no external deps.
    Positive delta convention throughout this script's callers: positive = WITH_SLEEP
    more coherent than the matched NO_SLEEP control."""
    nz = [d for d in deltas if d is not None and abs(d) > 1e-12]
    n = len(nz)
    if n == 0:
        return {"n": 0, "n_pos": 0, "n_neg": 0, "p_value": None}
    n_pos = sum(1 for d in nz if d > 0)
    n_neg = n - n_pos
    k = min(n_pos, n_neg)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) * 2 / (2 ** n)
    return {"n": n, "n_pos": n_pos, "n_neg": n_neg, "p_value": float(min(1.0, p))}


def _pearson_r(xs: List[Optional[float]], ys: List[Optional[float]]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xa = np.array([p[0] for p in pairs], dtype=float)
    ya = np.array([p[1] for p in pairs], dtype=float)
    if np.std(xa) < 1e-12 or np.std(ya) < 1e-12:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def _sleep_ablation_comparison(
    with_sleep_episodes: List[Dict[str, Any]],
    no_sleep_episodes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """MECH-490 core readout: for every segment boundary where WITH_SLEEP fired a sleep
    cycle, compare against the matched NO_SLEEP segment (same index -- the "equivalent
    elapsed time, no sleep" control MECH-490's own falsifying signature names, already
    inherent in this two-arm design). Extended from V3-EXQ-913's version with
    spawn-match diagnostics, action-sequence deltas, and commit-gate-engagement deltas."""
    fired_indices = [i for i, seg in enumerate(with_sleep_episodes)
                     if seg.get("sleep_cycle_fired_before_this_segment")]
    comparisons = []
    for i in fired_indices:
        if i >= len(no_sleep_episodes):
            continue
        ws, ns = with_sleep_episodes[i], no_sleep_episodes[i]
        ws_traj, ns_traj = ws["trajectory_local_window"], ns["trajectory_local_window"]
        ws_act,  ns_act  = ws["action_sequence_local_window"], ns["action_sequence_local_window"]
        ws_cg,   ns_cg   = ws["commit_gate_local_window"], ns["commit_gate_local_window"]
        ws_spawn = tuple(int(c) for c in (ws.get("spawn_pos") or (-1, -1)))
        ns_spawn = tuple(int(c) for c in (ns.get("spawn_pos") or (-1, -1)))
        spawn_matched = bool(ws_spawn == ns_spawn)
        spawn_manhattan = abs(ws_spawn[0] - ns_spawn[0]) + abs(ws_spawn[1] - ns_spawn[1])

        def _delta(key_ws, key_ns, negate=False):
            a, b = key_ws, key_ns
            if a is None or b is None:
                return None
            d = float(a) - float(b)
            return -d if negate else d

        comparisons.append({
            "segment_index": i,
            "spawn_position_matched": spawn_matched,
            "spawn_manhattan_distance": spawn_manhattan,
            "with_sleep_spawn": list(ws_spawn), "no_sleep_spawn": list(ns_spawn),
            "with_sleep": ws_traj, "no_sleep_matched": ns_traj,
            "with_sleep_action_sequence": ws_act, "no_sleep_action_sequence_matched": ns_act,
            "with_sleep_commit_gate": ws_cg, "no_sleep_commit_gate_matched": ns_cg,
            # Positive-coherence-lift sign convention (module docstring "COVARIES..."):
            # a negated delta means "less turning/tortuosity/reversal under sleep", i.e.
            # MORE coherent, matching mean_run_length_delta's natural positive-is-more sign.
            "turning_entropy_delta": _delta(ws_traj.get("turning_angle_entropy_bits"),
                                            ns_traj.get("turning_angle_entropy_bits"), negate=True),
            "tortuosity_delta": _delta(ws_traj.get("tortuosity"), ns_traj.get("tortuosity"), negate=True),
            "mean_run_length_delta": _delta(ws_act.get("mean_run_length"), ns_act.get("mean_run_length")),
            "reversal_rate_delta": _delta(ws_act.get("reversal_rate"), ns_act.get("reversal_rate"), negate=True),
            "commit_gate_engagement_rate_delta": _delta(
                ws_cg.get("commit_gate_engagement_rate"), ns_cg.get("commit_gate_engagement_rate")),
            "mean_running_variance_with_sleep": ws_cg.get("mean_running_variance"),
            "mean_running_variance_no_sleep": ns_cg.get("mean_running_variance"),
        })
    return {
        "n_sleep_firings": len(fired_indices),
        "fired_segment_indices": fired_indices,
        "matched_comparisons": comparisons,
    }


def run_seed_arm(seed: int, arm: str, dry_run: bool = False) -> Dict[str, Any]:
    sleep_enabled = (arm == ARM_WITH_SLEEP)
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    device = torch.device("cpu")
    total_eps = (2 + 2 + 5 + 5 + 5) if dry_run else TRAIN_TOTAL_EPS

    print(f"\nSeed {seed} Condition {arm}", flush=True)
    scaffold_cfg = _make_scaffold_cfg(dry_run)
    scaffold_cfg.scaffold_p2_num_resources = DEV_NUM_RESOURCES
    probe_env = _build_env(scaffold_cfg, "p2")
    probe_env.reset()
    cfg = _make_config_for_arm(probe_env, sleep_enabled)
    agent = REEAgent(cfg).to(device)
    scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)
    print(f"[EXQ-921] seed={seed} arm={arm} world_obs_dim={probe_env.world_obs_dim}"
          f" body_obs_dim={probe_env.body_obs_dim} use_sleep_loop={sleep_enabled}", flush=True)

    diag = _run_curriculum(agent, scheduler, device, seed, total_eps)

    eval_eps   = 3 if dry_run else EVAL_EPISODES
    eval_steps = 30 if dry_run else EVAL_STEPS
    eval_env = _build_dev_eval_env(scaffold_cfg, seed=seed)
    env_config_snapshot = _env_config_snapshot(eval_env)
    ree = _continuous_life_run(agent, eval_env, eval_eps, eval_steps, seed, arm)

    episodes_log = ree["episodes"]
    within_life = _within_life_development(episodes_log)
    occupancy = _lifetime_affective_occupancy(episodes_log)
    zone_stats = _zone_habitat_stats(episodes_log)

    resource_exhausted_segments = sum(
        1 for seg in episodes_log
        if seg.get("steps") and all(s.get("n_resources_remaining", 1) == 0 for s in seg["steps"])
    )

    print(f"[EXQ-921] seed={seed} arm={arm} channel std: "
          + "  ".join(f"{k}={ree['chan_std'][k]:.4f}" for k in
                      ["z_harm_a", "z_harm_un", "drive", "z_goal", "vigor", "z_block", "excite", "dread"]),
          flush=True)
    print(f"[EXQ-921] seed={seed} arm={arm} segments={len(episodes_log)} "
          f"sleep_cycles_fired={ree['sleep_cycles_fired']} "
          f"continuity_resets={ree['n_continuity_resets']}/{eval_eps - 1}", flush=True)

    seed_core_ok = all(ree["chan_std"].get(k, 0.0) > STD_FLOOR for k in CORE_CHANNELS)
    harm_trained = (diag["p0_harm_train_steps"] + diag["hazard_harm_train_steps"]) > 0
    layout_continuity_confirmed = bool(ree["n_continuity_resets"] == eval_eps - 1)
    seed_pass = bool(seed_core_ok and harm_trained and layout_continuity_confirmed)
    print(f"verdict: {'PASS' if seed_pass else 'FAIL'} seed={seed} arm={arm} "
          f"core_ok={seed_core_ok} harm_trained={harm_trained} "
          f"layout_continuity_confirmed={layout_continuity_confirmed}", flush=True)

    rf_final = getattr(agent, "residue_field", None)
    residue_stats_final = {}
    if rf_final is not None:
        with torch.no_grad():
            stats = rf_final.get_statistics()
            residue_stats_final = {
                "total_residue": float(stats["total_residue"]),
                "num_harm_events": int(stats["num_harm_events"]),
                "active_centers": int(stats["active_centers"]),
                "mean_weight": float(stats["mean_weight"]),
            }

    return {
        "seed": seed, "arm": arm, "diag": diag, "chan_std": ree["chan_std"], "chan_mean": ree["chan_mean"],
        "freeze_fires": ree["freeze_fires"], "block_steps": ree["block_steps"],
        "sleep_cycles_fired": ree["sleep_cycles_fired"],
        "eval_steps": ree["eval_steps"], "z_goal_eval_mean": ree["chan_mean"].get("z_goal", 0.0),
        "harm_trained": harm_trained,
        "layout_continuity_confirmed": layout_continuity_confirmed,
        "n_continuity_resets": ree["n_continuity_resets"],
        "episodes_full": episodes_log,
        "agent": agent, "env_config": env_config_snapshot,
        "residue_stats_final": residue_stats_final,
        "total_spawn_retries": ree["total_spawn_retries"],
        "spawn_exhausted_segments": ree["spawn_exhausted_segments"],
        "within_life": within_life, "occupancy": occupancy, "zone_stats": zone_stats,
        "resource_exhausted_segments": resource_exhausted_segments,
        "seed_pass": seed_pass,
    }


def run(seeds=None, dry_run: bool = False) -> dict:
    if seeds is None:
        seeds = SEEDS_DEFAULT
    print(f"[V3-EXQ-921] MECH-490 Sleep Commit-Gate Persistence -- Spawn-Matched + "
          f"Commit-Variance Instrumented\n"
          f"  Seeds: {seeds}  Arms: {ARMS}  curriculum: Stage-0/0b/P0/Stage-H/P1 + harm-pathway training\n"
          f"  Train eps/(seed,arm): {TRAIN_TOTAL_EPS}  Eval: {EVAL_EPISODES} segments x up to {EVAL_STEPS} steps\n"
          f"  Output: REE_assembly/evidence/experiments/{EXPERIMENT_TYPE}/", flush=True)

    results = [run_seed_arm(s, a, dry_run=dry_run) for s in seeds for a in ARMS]
    agents = [r.pop("agent") for r in results]
    by_key = {(r["seed"], r["arm"]): r for r in results}

    chan_keys = list(results[0]["chan_std"].keys())
    chan_max_std = {k: max(r["chan_std"].get(k, 0.0) for r in results) for k in chan_keys}
    chan_nondegen = {k: bool(chan_max_std[k] > STD_FLOOR) for k in chan_keys}
    total_harm_steps = sum(r["diag"]["p0_harm_train_steps"] + r["diag"]["hazard_harm_train_steps"]
                           for r in results)
    total_block = sum(r["block_steps"] for r in results)
    total_sleep_cycles_with = sum(r["sleep_cycles_fired"] for r in results if r["arm"] == ARM_WITH_SLEEP)
    total_sleep_cycles_no   = sum(r["sleep_cycles_fired"] for r in results if r["arm"] == ARM_NO_SLEEP)
    total_freeze = sum(r["freeze_fires"] for r in results)
    total_steps = sum(r["eval_steps"] for r in results)
    total_spawn_retries = sum(r["total_spawn_retries"] for r in results)
    total_spawn_exhausted = sum(r["spawn_exhausted_segments"] for r in results)
    z_goal_activated = any(r["z_goal_eval_mean"] > 1e-3 for r in results)

    core_ok = all(chan_nondegen.get(k, False) for k in CORE_CHANNELS)
    harm_trained = total_harm_steps > 0
    layout_continuity_confirmed = all(r["layout_continuity_confirmed"] for r in results)
    sleep_ablation_engaged = bool(total_sleep_cycles_with > 0 and total_sleep_cycles_no == 0)
    freeze_not_locked = (total_freeze == 0) or (total_freeze < total_steps)
    all_seeds_arms_pass = all(r["seed_pass"] for r in results)

    # Sleep ablation comparison, per seed.
    sleep_ablation_per_seed: Dict[str, Any] = {}
    for s in seeds:
        ws = by_key.get((s, ARM_WITH_SLEEP))
        ns = by_key.get((s, ARM_NO_SLEEP))
        if ws is None or ns is None:
            continue
        sleep_ablation_per_seed[f"seed{s}"] = _sleep_ablation_comparison(
            ws["episodes_full"], ns["episodes_full"]
        )

    # Pool matched comparisons across ALL seeds (module docstring "MIN_SEEDS_FOR_CONFIRM").
    pooled: List[Dict[str, Any]] = []
    seeds_with_comparisons: set = set()
    for skey, block in sleep_ablation_per_seed.items():
        for c in block.get("matched_comparisons", []):
            pooled.append(c)
            seeds_with_comparisons.add(skey)
    n_pooled = len(pooled)
    n_spawn_matched = sum(1 for c in pooled if c.get("spawn_position_matched"))
    spawn_match_rate = (n_spawn_matched / n_pooled) if n_pooled else 0.0
    commit_gate_instrumented = any(
        (c.get("with_sleep_commit_gate") or {}).get("n_ticks", 0) > 0
        and (c.get("no_sleep_commit_gate_matched") or {}).get("n_ticks", 0) > 0
        for c in pooled
    )

    primary_deltas    = [c.get("mean_run_length_delta") for c in pooled]
    reversal_deltas    = [c.get("reversal_rate_delta") for c in pooled]
    turning_deltas     = [c.get("turning_entropy_delta") for c in pooled]
    tortuosity_deltas  = [c.get("tortuosity_delta") for c in pooled]
    engagement_deltas  = [c.get("commit_gate_engagement_rate_delta") for c in pooled]

    sign_test_mean_run_length = _sign_test_p(primary_deltas)
    sign_test_reversal_rate   = _sign_test_p(reversal_deltas)
    sign_test_turning_entropy = _sign_test_p(turning_deltas)
    sign_test_tortuosity      = _sign_test_p(tortuosity_deltas)
    commit_gate_covary_r      = _pearson_r(primary_deltas, engagement_deltas)

    # MECH-490 non-degeneracy precondition (what_would_answer, verbatim structure):
    # spawn matched AND commit-variance instrumented AND >=1 usable comparison.
    non_degenerate = bool(
        n_pooled >= 1
        and spawn_match_rate >= SPAWN_MATCH_RATE_FLOOR
        and commit_gate_instrumented
    )
    degeneracy_reason = "" if non_degenerate else (
        f"MECH-490 non-degeneracy precondition unmet: "
        f"n_matched_comparisons={n_pooled}, "
        f"spawn_match_rate={spawn_match_rate:.3f} (floor {SPAWN_MATCH_RATE_FLOOR}), "
        f"commit_gate_instrumented={commit_gate_instrumented}"
    )

    # MECH-490 confirming/falsifying signature (what_would_answer, this run's testable
    # subset -- the transfer-test falsifying alternative is explicitly out of scope, see
    # module docstring "OUT OF SCOPE"; the "matched no-sleep control" falsifying
    # alternative is already inherent in this two-arm design's own NO_SLEEP arm).
    coherence_lift_significant = bool(
        non_degenerate
        and sign_test_mean_run_length["p_value"] is not None
        and sign_test_mean_run_length["p_value"] < SIGN_TEST_ALPHA
        and len(seeds_with_comparisons) >= MIN_SEEDS_FOR_CONFIRM
    )
    covaries_with_commit_gate = bool(
        commit_gate_covary_r is not None and abs(commit_gate_covary_r) >= COVARY_R_FLOOR
        and commit_gate_covary_r > 0
    )

    if not non_degenerate:
        evidence_direction = "non_contributory"
    elif coherence_lift_significant and covaries_with_commit_gate:
        evidence_direction = "supports"
    elif coherence_lift_significant and not covaries_with_commit_gate:
        evidence_direction = "weakens"
    else:
        evidence_direction = "non_contributory"

    degeneracy_block = check_degeneracy({
        "mean_run_length_delta_pooled": primary_deltas,
        "commit_gate_engagement_rate_delta_pooled": engagement_deltas,
    })

    passed = bool(core_ok and harm_trained and layout_continuity_confirmed and sleep_ablation_engaged)
    outcome = "PASS" if passed else "FAIL"

    within_life_per_seed_arm = {f"seed{r['seed']}_{r['arm']}": r["within_life"] for r in results}
    lifetime_affective_occupancy = {f"seed{r['seed']}_{r['arm']}": r["occupancy"] for r in results}
    zone_habitat_per_seed_arm = {f"seed{r['seed']}_{r['arm']}": r["zone_stats"] for r in results}

    metrics: Dict[str, Any] = {
        "n_seeds": float(len(seeds)),
        "n_arms": float(len(ARMS)),
        "total_harm_pathway_train_steps": float(total_harm_steps),
        "total_block_steps": float(total_block),
        "total_sleep_cycles_fired_with_sleep_arm": float(total_sleep_cycles_with),
        "total_sleep_cycles_fired_no_sleep_arm": float(total_sleep_cycles_no),
        "total_freeze_fires": float(total_freeze),
        "total_eval_steps": float(total_steps),
        "z_goal_activated_at_eval": 1.0 if z_goal_activated else 0.0,
        "total_spawn_safe_retries": float(total_spawn_retries),
        "total_spawn_safe_exhausted_segments": float(total_spawn_exhausted),
        "total_resource_exhausted_segments": float(sum(r["resource_exhausted_segments"] for r in results)),
        "n_sleep_firing_matched_comparisons": float(n_pooled),
        "n_seeds_with_comparisons": float(len(seeds_with_comparisons)),
        "spawn_match_rate": float(spawn_match_rate),
        "commit_gate_instrumented": 1.0 if commit_gate_instrumented else 0.0,
        "sign_test_mean_run_length_p": sign_test_mean_run_length["p_value"],
        "sign_test_reversal_rate_p": sign_test_reversal_rate["p_value"],
        "sign_test_turning_entropy_p": sign_test_turning_entropy["p_value"],
        "sign_test_tortuosity_p": sign_test_tortuosity["p_value"],
        "commit_gate_covary_r": commit_gate_covary_r,
        "coherence_lift_significant": 1.0 if coherence_lift_significant else 0.0,
        "covaries_with_commit_gate": 1.0 if covaries_with_commit_gate else 0.0,
    }
    for r in results:
        key = f"seed{r['seed']}_{r['arm']}"
        metrics[f"{key}_layout_continuity_confirmed"] = 1.0 if r["layout_continuity_confirmed"] else 0.0
        metrics[f"{key}_n_continuity_resets"] = float(r["n_continuity_resets"])
        metrics[f"{key}_sleep_cycles_fired"] = float(r["sleep_cycles_fired"])
        metrics[f"{key}_z_goal_eval_mean"] = float(r["z_goal_eval_mean"])
        metrics[f"{key}_resource_exhausted_segments"] = float(r["resource_exhausted_segments"])
        rstats = r.get("residue_stats_final") or {}
        metrics[f"{key}_residue_total_residue_final"] = float(rstats.get("total_residue", 0.0))
    for k in chan_keys:
        metrics[f"chan_max_std_{k}"] = float(chan_max_std[k])
        metrics[f"chan_mean_{k}"] = float(np.mean([r["chan_mean"].get(k, 0.0) for r in results]))

    interpretation = {
        "label": ("mech490_evidence_run" if non_degenerate else "mech490_precondition_unmet"),
        "combination_rule": (
            "evidence_direction = 'non_contributory' if non_degenerate is False, else "
            "'supports' if (mean_run_length sign-test p<0.05 across >=5 seeds) AND "
            "(commit_gate_engagement_rate_delta Pearson r >= +0.3 with mean_run_length_delta), "
            "else 'weakens' if the coherence lift is significant but does NOT covary with "
            "commit-gate engagement, else 'non_contributory' (lift itself not significant -- "
            "underpowered/null, informative neither way)."
        ),
        "criteria": [
            {"name": "core_channels_non_degenerate", "load_bearing": True, "passed": core_ok},
            {"name": "harm_pathway_trained", "load_bearing": True, "passed": harm_trained},
            {"name": "layout_continuity_confirmed", "load_bearing": True, "passed": layout_continuity_confirmed},
            {"name": "sleep_ablation_engaged", "load_bearing": True, "passed": sleep_ablation_engaged},
            {"name": "spawn_match_rate_met_precondition", "load_bearing": True,
             "passed": bool(spawn_match_rate >= SPAWN_MATCH_RATE_FLOOR)},
            {"name": "commit_gate_instrumented", "load_bearing": True, "passed": commit_gate_instrumented},
            {"name": "freeze_not_locked", "load_bearing": False, "passed": freeze_not_locked},
            {"name": "all_seed_arm_verdicts_pass", "load_bearing": False, "passed": all_seeds_arms_pass},
            *[{"name": f"channel_{k}", "load_bearing": False, "passed": bool(chan_nondegen.get(k, False))}
              for k in chan_keys],
        ],
        "note": (
            f"outcome PASS/FAIL is a MECHANICAL run-health verdict (core channels vary, "
            f"harm-pathway trained, layout continuity + sleep ablation engaged) and is "
            f"INDEPENDENT of evidence_direction, which is the scientific MECH-490 verdict. "
            f"{n_pooled} matched WITH_SLEEP-vs-NO_SLEEP comparisons across "
            f"{len(seeds_with_comparisons)} seeds; spawn_match_rate={spawn_match_rate:.3f}; "
            f"claim_ids={CLAIM_IDS}; evidence_direction={evidence_direction}."
        ),
    }

    summary_markdown = f"""# V3-EXQ-921 -- MECH-490 Sleep Commit-Gate Persistence (V3-EXQ-913 successor)

**Status:** {outcome} (mechanical run-health verdict)
**MECH-490 evidence_direction:** {evidence_direction}
**Purpose:** spawn-matched + commit-variance-instrumented sleep-vs-no-sleep ablation,
testing whether post-sleep action-sequence coherence gains (if real) covary with E3
commit-gate engagement, per MECH-490's confirming/falsifying signature.

- harm-pathway train steps (total, all seed/arm): {total_harm_steps}
- layout continuity confirmed (all seed/arm): {layout_continuity_confirmed}
- sleep cycles fired -- WITH_SLEEP arm: {total_sleep_cycles_with}  NO_SLEEP arm: {total_sleep_cycles_no}
- matched sleep-firing-vs-no-sleep-control comparisons obtained: {n_pooled} across {len(seeds_with_comparisons)} seeds
- spawn_match_rate: {spawn_match_rate:.3f} (floor {SPAWN_MATCH_RATE_FLOOR}) -- non_degenerate={non_degenerate}
- commit_gate_instrumented: {commit_gate_instrumented}
- sign-test p-values: mean_run_length={sign_test_mean_run_length['p_value']}  reversal_rate={sign_test_reversal_rate['p_value']}  turning_entropy={sign_test_turning_entropy['p_value']}  tortuosity={sign_test_tortuosity['p_value']}
- commit-gate covariance (Pearson r, mean_run_length_delta vs commit_gate_engagement_rate_delta): {commit_gate_covary_r}
- coherence_lift_significant: {coherence_lift_significant}   covaries_with_commit_gate: {covaries_with_commit_gate}
- eval steps (total): {total_steps}  across {EVAL_EPISODES} segments/(seed,arm) x up to {EVAL_STEPS} steps
- events: block={total_block}  freeze fires (motor-override relaxed): {total_freeze}
- safe-spawn / continuity-spawn retries (total): {total_spawn_retries}  (segments exhausted: {total_spawn_exhausted})

## Eval channel mean / max-std
{chr(10).join(f'- {k}: mean={metrics.get("chan_mean_"+k,0.0):.4f} max_std={chan_max_std[k]:.5f} ({"varies" if chan_nondegen[k] else "FLAT"})' for k in chan_keys)}

## Sleep-vs-no-sleep matched comparisons (per seed)
{chr(10).join(f'- seed{s}: n_sleep_firings={v.get("n_sleep_firings")} fired_at_segments={v.get("fired_segment_indices")}' for s, v in sleep_ablation_per_seed.items())}

## Lifetime affective occupancy (per seed/arm, non-gating, SENT-2 hygiene)
{chr(10).join(f'- {key}: frac_dread_above_p75={v.get("frac_steps_dread_above_p75")} frac_z_harm_a_above_p75={v.get("frac_steps_z_harm_a_above_p75")} frac_harm_event={v.get("frac_harm_event")} frac_in_reef={v.get("frac_in_reef")}' for key, v in lifetime_affective_occupancy.items())}

## For a future reader (or `/failure-autopsy`) on THIS run

If `spawn_match_rate` is below {SPAWN_MATCH_RATE_FLOOR}, the shared-candidate-order fix
still left too many boundaries with genuinely divergent hazard/resource occupancy between
arms (hazard drift can differ enough by segment>1 to exclude the shared first choice in
one arm but not the other) -- a successor could canonicalize the occupied-set reference
(e.g. always filter against the ep0 canonical hazard/resource set rather than each arm's
live drifted set) to raise the match rate further. If `n_sleep_firing_matched_comparisons`
is small despite 6 seeds, the K=10 cadence did not fire enough within 24 segments for some
seeds -- lower `sleep_loop_episodes_K` or raise `EVAL_EPISODES` in a successor. This run
does NOT attempt the transfer test or blinded umpire (Section 13d items 4-5) -- those are
gated on THIS run producing `coherence_lift_significant=True`.
"""

    first_env_config = results[0].get("env_config", {}) if results else {}
    episode_log = {
        "experiment_type": EXPERIMENT_TYPE,
        "phase": "mech490_sleep_commitgate_spawn_matched",
        "toroidal": bool(first_env_config.get("toroidal", False)),
        "env_config": first_env_config,
        "seeds": [{"seed": r["seed"], "arm": r["arm"], "episodes": r.get("episodes_full", [])}
                  for r in results],
    }

    result = {
        "status": outcome, "outcome": outcome, "metrics": metrics,
        "summary_markdown": summary_markdown, "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE, "evidence_direction": evidence_direction,
        "experiment_type": EXPERIMENT_TYPE, "interpretation": interpretation,
        "episode_log": episode_log, "agents": agents,
        "within_life_development": within_life_per_seed_arm,
        "sleep_ablation_comparison": sleep_ablation_per_seed,
        "zone_habitat_stats": zone_habitat_per_seed_arm,
        "lifetime_affective_occupancy": lifetime_affective_occupancy,
        "config": first_env_config,
        "sleep_driver_pattern": "K=10 multi-fire (SleepLoopManager) vs K=never, two-arm ablation",
        "supersedes_note": (
            "extends V3-EXQ-913's sleep_ablation_comparison block (spawn-matched, "
            "commit-variance instrumented, 6 seeds vs 913's 2) -- does not supersede "
            "913's OTHER findings (layout continuity, microhabitat probabilistic cue), "
            "which are unchanged/inherited here, not this run's test target."
        ),
    }
    result.update(degeneracy_block)
    return result


if __name__ == "__main__":
    import argparse
    import json
    import time
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result = run(seeds=args.seeds, dry_run=args.dry_run)
    agents = result.pop("agents", [])

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"]      = ts
    result["timestamp_utc"]      = ts
    result["run_id"]             = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["experiment_purpose"] = EXPERIMENT_PURPOSE
    result["claim_ids"]          = CLAIM_IDS

    out_dir = (Path(__file__).resolve().parents[2]
               / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE)
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_log = result.pop("episode_log", None)
    if episode_log is not None:
        episode_log["run_id"] = result["run_id"]
        log_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}_episode_log.json"
        log_path.write_text(json.dumps(episode_log, indent=2) + "\n", encoding="utf-8")
        print(f"Episode log written to: {log_path}", flush=True)
        result["companion_files"] = [f"{EXPERIMENT_TYPE}/{log_path.name}"]

    out_path = write_flat_manifest(
        result,
        out_dir.parent,
        dry_run=args.dry_run,
        config=result.get("config"),
        seeds=args.seeds,
        script_path=Path(__file__),
        started_at=t0,
        agent=(agents[0] if len(agents) == 1 else agents) if agents else None,
    )
    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    print(f"final_outcome: {result['outcome']}", flush=True)
    print(f"evidence_direction: {result['evidence_direction']}", flush=True)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    emit_outcome(outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
                 manifest_path=out_path,
                 dry_run=bool(args.dry_run))
