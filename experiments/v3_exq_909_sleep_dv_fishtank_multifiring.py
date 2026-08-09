"""
V3-EXQ-909 -- Sleep-Refinement DV Probe on the Fishtank Substrate (multi-firing, multi-seed)

Claims: None (diagnostic discrimination; does not weight governance)

EXPERIMENT_PURPOSE = "diagnostic"

SLEEP DRIVER: manual-cycle-loop (run_sleep_cycle() called once per cycle in a dedicated
N_CYCLES wake-sleep-test loop) -- for readiness probes and discriminative experiments. This
run calls SleepLoopManager.force_cycle() once at EVERY eval segment boundary (bypassing the
sleep_loop_episodes_K=10 cadence entirely, which would otherwise only fire ~once across a
short eval window -- exactly what happened in V3-EXQ-906b), guaranteeing one sleep-cycle
firing per boundary across EVAL_EPISODES-1 boundaries x len(seeds) seeds.

WHY THIS RUN. Routing source:
REE_assembly/evidence/planning/observational_review_V3-EXQ-906b_2026-08-09.md Section 13-C
(track C), Section 11a, Section 12d. V3-EXQ-906b (full-stack all-ON observational fishtank)
fired sleep exactly once across its whole 8-segment eval window. An n=1 ad-hoc read of that
single firing (review Section 12d) found sws_n_writes=5.0 but replay_diversity_index=-1.0
(the phase-manager's own sentinel for zero SWS replay draws) and sws_slot_diversity=0.0021
(near-zero), despite genuine waking behavioural diversity being independently confirmed on
this substrate (review Section 1: 6 active modes, action entropy 1.88-2.10 bits/segment).
This is consistent with sleep_substrate_plan.md GAP-2 being blocked on CONVERSION (diffuse
repertoire diversity does not convert into sleep-replay/slot diversity) rather than on
REPERTOIRE (having no waking variation to refine at all -- the explanation for two EARLIER,
different-substrate nulls, V3-EXQ-418l/436a). But n=1 is explicitly flagged in the review as
insufficient ("a proper probe across multiple firings/seeds is still needed... report
sws_slot_diversity / replay_diversity_index as first-class manifest metrics"). This run is
that probe.

WHAT THIS RUN IS NOT: a claim test, a fix for GAP-2, or a change to the fishtank's
waking-diversity substrate. It is a discriminating diagnostic between two hypotheses about
WHY GAP-2 is blocked. Either outcome (null or non-null) is informative and is reported as
such -- this is not a foregone conclusion to rubber-stamp.

CONFIG DELTA VS V3-EXQ-906b, AND WHY IT IS NOT A SUBSTRATE-READINESS VIOLATION OF THE "DON'T
FIX THE THING UNDER TEST" CONSTRAINT (empirically verified before writing this script, per
this skill's Step 2.5a "empirical confirmation" requirement -- see the throwaway probe this
verification is based on, not committed):

  Building V3-EXQ-906b's EXACT config and instantiating a REEAgent from it shows
  `agent.sleep_loop.draws_per_cycle == 0` and `agent.sleep_replay_sampler is None`,
  DETERMINISTICALLY, regardless of seed, training, or waking behaviour. Tracing why
  (ree_core/agent.py:2688-2698): `draws_per_cycle` is set from
  `config.mech285_draws_per_cycle` (default 50) ONLY IF `self.sleep_replay_sampler is not
  None`; the sampler is constructed (agent.py:2558-2568) only when BOTH
  `config.use_mech285_sampler` (default False) is True AND `hippocampal.anchor_set` is not
  None, and `anchor_set` itself is built (hippocampal/module.py:291-299) only when
  `config.hippocampal.use_anchor_sets` (default False) is True. V3-EXQ-906b's `_make_config`
  sets none of these three flags, so ALL THREE preconditions are unmet and
  `sws_routed_draws` in `phase_manager._run_cycle` is permanently empty
  (`range(self.draws_per_cycle)` with `draws_per_cycle=0`) -- `replay_diversity_index`
  (`_n_draws=0` -> the -1.0 sentinel, `phase_manager.py:526-531`) is THEREFORE A STRUCTURAL
  ARTIFACT OF THIS CONFIG, not a measurement of anything, on every firing, on every seed,
  regardless of how much waking diversity exists to convert. Running this probe unmodified
  would not test the conversion-vs-repertoire question at all -- it would just re-confirm
  that the replay-draw mechanism was never turned on, which the review's own Section 6
  ("freeze zero-fires is a config artifact, not REE and not incapacity") already establishes
  as exactly the failure mode to distinguish from a genuine substrate finding.

  Additionally confirmed empirically (same throwaway probe): with `use_anchor_sets=True` and
  `use_mech285_sampler=True` alone, `agent.sleep_loop.routing_gate` is STILL None, and
  `_run_cycle`'s draw-recording line (`if self.routing_gate is not None: ...
  sws_routed_draws.append(routed)`, phase_manager.py:270-275) is unconditionally gated on it
  -- so a drawn anchor is silently discarded even once the sampler itself is live, and
  replay_diversity_index remains the -1.0 sentinel. A drawn anchor only reaches
  `sws_routed_draws` once `config.use_mech272_routing` (default False, constructs
  `sleep_routing_gate`, agent.py:2578-2588) is ALSO set. With all three flags
  (`use_anchor_sets`, `use_mech285_sampler`, `use_mech272_routing`) added on top of 906b's
  unmodified config, the same throwaway probe (120 waking steps, untrained agent, seed 0)
  produced `replay_diversity_index=0.02` -- a genuine, non-sentinel reading -- confirming the
  fix makes the DV reachable.

  This is a targeted fix to the SLEEP-REPLAY MECHANISM ITSELF (three independent gates that
  each default OFF and each of which 906b left off), not a change to the waking-diversity
  fishtank substrate (environment, policy stack, affect modules -- all identical to 906b,
  see EVAL_ENV_EXTRA_KWARGS and _make_config, both imported unchanged). Confirmed no
  waking-side consumer is affected: `anchor_set` has exactly two OTHER consumers besides the
  sleep-replay sampler (`ghost_goal_bank.py`, gated on the separate flag
  `use_mech292_ghost_bank`; and `claustrum/coalition_templates.py`, referenced only in
  commentary, not wired into `REEAgent.__init__`) -- neither flag is set here, so
  `GhostGoalBank` is never constructed (agent.py:333-354, its own construction is gated on
  `use_mech292_ghost_bank`) and MECH-293 ghost-probe waking-phase consumption
  (`use_mech293_ghost_probes`, also unset) never fires. `sleep_routing_gate` and
  `sleep_replay_sampler` are referenced ONLY inside `phase_manager._run_cycle` (sleep-cycle
  internals) -- confirmed by grep across ree_core/ -- so waking-phase action selection,
  affect computation, and environment dynamics are bit-identical to 906b's. `sws_slot_diversity`
  itself does NOT require this fix (it is driven by a separate ContextMemory-prototype-write
  pathway inside `run_sws_schema_pass`, already live and non-degenerate in 906b's own run at
  0.0021); only `replay_diversity_index` was structurally dead without it.

  `use_mech275_aggregator` (Phase D) and `use_mech273_self_model` (Phase E) are deliberately
  LEFT OFF, matching 906b -- they are not prerequisites for sws_slot_diversity or
  replay_diversity_index and adding them would widen the config delta beyond what this
  probe's decisive readouts need.

WHAT THIS RUN DOES, CONCRETELY:
  1. Same curriculum training as 906b (Stage-0/0b/P0/Stage-H/P1, TRAIN_TOTAL_EPS=220
     episodes) -- unchanged, guarantees the same trained substrate.
  2. Same eval env as 906b (proximity-radius fix, safe spawn, scheduled injections) --
     unchanged, EVAL_ENV_EXTRA_KWARGS imported verbatim from 906b's module.
  3. EVAL_EPISODES raised from 906b's 8 to 16 segments/seed (more boundaries -> more
     opportunities to fire), and seeds raised from 906b's single seed 0 to
     DEFAULT_SEEDS=[0, 1, 2] (three seeds; seed 44 excluded per the documented reef-config
     instability precedent, CLAUDE.md "Training protocol" checklist item).
  4. At every segment boundary (after the first, which still gets a full agent.reset()):
     `agent._flush_exploration_episode()` (906b's own consolidation housekeeping, unrelated
     to sleep) followed by `agent.sleep_loop.force_cycle(agent)` -- a direct call to the
     existing "Diagnostic / experiment hook: run a sleep cycle immediately, regardless of
     the K-episode counter" (phase_manager.py:200-206 docstring), guaranteeing exactly one
     firing per boundary. `notify_episode_end` (the K=10-cadence path) is NOT called during
     eval in this script -- force_cycle alone drives every eval-phase firing, deterministically.
  5. For each firing, records (as a FIRST-CLASS entry in `sleep_firing_records[]`, not
     buried in a nested per-cycle log): the JUST-COMPLETED segment's waking-diversity read
     (mode-entropy in bits, unique grid cells visited, realized step count) alongside that
     firing's sws_n_writes / sws_slot_diversity / replay_diversity_index / rem_n_rollouts /
     rem_mean_harm_terrain / rem_terrain_variance / rem_n_reverse / rem_wanting_spread_n_steps
     / post_sleep_z_goal_before / post_sleep_z_goal_after / post_sleep_z_goal_retention.
  6. Aggregates the resulting sleep_firing_records[] across all firings and all seeds into
     manifest-top-level distribution statistics (min/mean/median/max of sws_slot_diversity
     and replay_diversity_index; the fraction of firings at the replay_diversity_index=-1.0
     sentinel; the fraction of firings clearing a pre-registered non-null epsilon on either
     DV; and the Pearson correlation between waking mode-entropy and each sleep DV, pooled
     across all firings/seeds).
  7. Self-routes to substrate_not_ready_requeue (never a claim-style verdict) if the
     measurement apparatus itself did not engage -- see preconditions below -- and otherwise
     discriminates between "sleep_dv_conversion_blocked" (the pre-registered null: sleep DVs
     stay at/near their floor across firings despite confirmed waking diversity) and
     "sleep_dv_nonnull_detected" (a non-trivial fraction of firings show real diversity/
     replay engagement -- requires further characterization, does not itself resolve GAP-2).

Output:
  evidence/experiments/v3_exq_909_sleep_dv_fishtank_multifiring/
    v3_exq_909_sleep_dv_fishtank_multifiring_<ts>.json               (manifest)
    v3_exq_909_sleep_dv_fishtank_multifiring_<ts>_episode_log.json   (per-step trace, compact)
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
    _make_config as _make_906b_config,
    _build_eval_env,
    _safe_reset,
    _env_config_snapshot,
    EVAL_ENV_EXTRA_KWARGS,  # noqa: F401 -- re-exported for anyone diffing config deltas
    TRAIN_TOTAL_EPS,
)
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE    = "v3_exq_909_sleep_dv_fishtank_multifiring"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS: List[str] = []

# validate_experiments.py anchor-reachability advisory: this script's one anchor-kind
# precondition (replay_draw_mechanism_structurally_reachable) is a direct integer
# comparison on agent.sleep_loop.draws_per_cycle, not a hand-written predicate scored
# against a known-degenerate reference sample (the SD-068/V3-EXQ-778d failure mode the
# guard exists for). draws_per_cycle is deterministically set to
# config.mech285_draws_per_cycle (default 50) whenever use_anchor_sets +
# use_mech285_sampler are both True (agent.py:2688-2698) -- there is no narrower-than-
# degeneracy scoring function here for a reachability probe to guard against; it is
# reachable by construction given the three config flags this script sets, and was
# empirically confirmed reachable (measured=50.0 on every seed) both by a standalone
# throwaway probe before this script was written and by this script's own dry-run smoke
# test (see module docstring "CONFIG DELTA VS V3-EXQ-906b").
ANCHOR_REACHABILITY_EXEMPT = (
    "structural integer comparison on agent.sleep_loop.draws_per_cycle, not a scored "
    "predicate against a reference sample; reachable by construction given "
    "use_anchor_sets=True + use_mech285_sampler=True (agent.py:2688-2698), confirmed "
    "empirically pre-authoring and in this script's own smoke test"
)

# ---- eval: same per-segment step budget as 906b; more segments per seed so the
# sleep-loop force_cycle() calls at each boundary yield enough firings for the
# multi-firing / multi-seed test the review's Section 13-C track C calls for.
EVAL_EPISODES  = 16
EVAL_STEPS     = 500
DEFAULT_SEEDS  = [0, 1, 2]  # seed 44 excluded -- documented reef-config instability (CLAUDE.md)

# ---- Pre-registered thresholds (NOT derived from this run's own statistics). ----
# Readiness floors -- verify the measurement apparatus (waking diversity existed to convert,
# the SWS write path and the REM rollout path both actually ran, the replay-draw mechanism
# was structurally reachable, and enough firings were collected) before trusting any
# discrimination read off the data.
WAKING_ENTROPY_FLOOR_BITS   = 0.5   # well below 906b's own measured 1.88-2.10 bits/segment
SWS_WRITES_FLOOR            = 1.0   # mean sws_n_writes across firings must clear this
REM_ROLLOUTS_FLOOR          = 1.0   # mean rem_n_rollouts across firings must clear this
DRAWS_PER_CYCLE_FLOOR       = 1.0   # agent.sleep_loop.draws_per_cycle (structural, not a mean)
MIN_FIRINGS_FLOOR           = 10.0  # review's own "a proper probe across multiple firings" bar
# Discrimination epsilons -- a firing counts as "non-null" if EITHER DV clears its epsilon.
SWS_SLOT_DIVERSITY_EPS       = 0.01
REPLAY_DIVERSITY_EPS         = 0.01  # strictly positive and not the -1.0 zero-draws sentinel
NONNULL_FRAC_THRESHOLD       = 0.10  # >=10% of firings must be non-null to call it "detected"

SLEEP_METRIC_KEYS = (
    "sws_n_writes", "sws_slot_diversity", "replay_diversity_index",
    "rem_n_rollouts", "rem_mean_harm_terrain", "rem_terrain_variance",
    "rem_n_reverse", "rem_wanting_spread_n_steps",
    "post_sleep_z_goal_before", "post_sleep_z_goal_after", "post_sleep_z_goal_retention",
)


def _make_config(env) -> "REEConfig":
    """906b's exact all-ON stack PLUS the three sleep-replay-reachability flags (see module
    docstring "CONFIG DELTA VS V3-EXQ-906b"). use_anchor_sets / use_mech285_sampler /
    use_mech272_routing are each independently gated OFF by default and each of them is a
    hard precondition for replay_diversity_index to ever be anything other than the -1.0
    zero-draws sentinel -- confirmed empirically before writing this script. Nothing about
    the waking-diversity substrate (env, policy, affect stack) changes."""
    cfg = _make_906b_config(env)
    cfg.hippocampal.use_anchor_sets = True   # MECH-269 Phase 2 (ii): builds hippocampal.anchor_set
    cfg.use_mech285_sampler = True           # Phase B: constructs sleep_replay_sampler
    cfg.use_mech272_routing = True           # Phase C: constructs sleep_routing_gate (draw recording)
    return cfg


def _segment_waking_diversity(ep_steps: List[Dict]) -> Dict[str, Any]:
    """Shannon entropy (bits) of the per-step `mode` classification, plus unique grid cells
    visited, over one just-completed segment. Same statistic family the observational review
    (Section 1) used to characterise 906b's waking behavioural diversity."""
    if not ep_steps:
        return {"mode_entropy_bits": 0.0, "unique_cells": 0, "n_steps": 0}
    counts: Dict[str, int] = {}
    for s in ep_steps:
        m = s["mode"]
        counts[m] = counts.get(m, 0) + 1
    n = len(ep_steps)
    entropy = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            entropy -= p * math.log2(p)
    unique_cells = len({tuple(s["pos"]) for s in ep_steps})
    return {"mode_entropy_bits": float(entropy), "unique_cells": int(unique_cells), "n_steps": int(n)}


def _pearson_r(xs: List[float], ys: List[float]) -> Optional[float]:
    """Pearson correlation, guarded against zero-variance degeneracy (returns None rather
    than a NaN/inf when either series is constant)."""
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 1e-12 or syy <= 1e-12:
        return None
    return float(sxy / math.sqrt(sxx * syy))


def _observational_run_with_forced_sleep(
    agent: REEAgent, env: CausalGridWorldV2, num_episodes: int,
    steps_per_episode: int, seed: int,
) -> Dict[str, Any]:
    """Same continuity-trajectory eval loop as 906b's `_observational_run`, but every
    segment boundary (after the first) forces a sleep cycle via `force_cycle()` instead of
    the K-episode `notify_episode_end()` cadence, and records a first-class firing entry
    per boundary combining the just-completed segment's waking-diversity read with that
    firing's sleep-cycle metrics."""
    device     = agent.device
    action_dim = env.action_dim
    episodes_log: List[Dict] = []
    firing_records: List[Dict] = []

    if getattr(agent, "pag_freeze_gate", None) is not None:
        try:
            agent.pag_freeze_gate.config.duration_input_threshold = 1e9
        except Exception:
            pass

    agent.eval()

    z_world_prev = None
    action_prev  = None
    z_self_prev  = None
    prev_ep_steps: List[Dict] = []

    for ep_idx in range(num_episodes):
        flat_obs, obs_dict, spawn_attempts, spawn_exhausted = _safe_reset(env)

        if ep_idx == 0:
            agent.reset()
        else:
            agent._flush_exploration_episode()
            waking = _segment_waking_diversity(prev_ep_steps)
            sleep_metrics = agent.sleep_loop.force_cycle(agent)
            record: Dict[str, Any] = {
                "seed": seed,
                "boundary_index": ep_idx - 1,
                "waking_mode_entropy_bits_prior_segment": waking["mode_entropy_bits"],
                "waking_unique_cells_prior_segment": waking["unique_cells"],
                "waking_steps_prior_segment": waking["n_steps"],
            }
            for k in SLEEP_METRIC_KEYS:
                record[k] = float(sleep_metrics.get(k, 0.0))
            firing_records.append(record)

        ep_steps: List[Dict] = []
        reef_cells_set = getattr(env, "_reef_cells", set())
        prev_in_reef = False

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

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            with torch.no_grad():
                agent.update_residue(float(harm_signal))

            agent_pos = (int(env.agent_x), int(env.agent_y))
            in_reef   = agent_pos in reef_cells_set
            affect    = _read_affect(agent, latent, obs_body)
            z_harm_s  = affect["z_harm_s"] if affect["z_harm_s"] is not None else 0.0
            world_change_norm = (float((latent.z_world - z_world_prev).norm().item())
                                 if z_world_prev is not None else 0.0)
            mode = _classify_mode(z_harm_s, world_change_norm, float(harm_signal),
                                  in_reef, affect["freeze"], affect["z_block"])

            ep_steps.append({"t": step_idx, "pos": list(agent_pos), "mode": mode})

            prev_in_reef = in_reef
            z_self_prev  = latent.z_self.detach()
            z_world_prev = latent.z_world.detach()
            action_prev  = action.detach()
            if done:
                break

        episodes_log.append({
            "ep": ep_idx, "realized_steps": len(ep_steps),
            "spawn_safe_attempts": int(spawn_attempts), "spawn_safe_exhausted": bool(spawn_exhausted),
        })
        prev_ep_steps = ep_steps
        print(f"  [eval] seed={seed} ep {ep_idx+1}/{num_episodes} steps={len(ep_steps)} "
              f"firings_so_far={len(firing_records)}", flush=True)

    return {"episodes": episodes_log, "firing_records": firing_records,
            "eval_steps": int(sum(e["realized_steps"] for e in episodes_log))}


def run_seed(seed: int, dry_run: bool = False) -> Dict[str, Any]:
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    device = torch.device("cpu")
    total_eps = (2 + 2 + 5 + 5 + 5) if dry_run else TRAIN_TOTAL_EPS

    print(f"\nSeed {seed} Condition sleep_dv_multifiring", flush=True)
    scaffold_cfg = _make_scaffold_cfg(dry_run)
    probe_env = _build_env(scaffold_cfg, "p2")
    probe_env.reset()
    agent = REEAgent(_make_config(probe_env)).to(device)
    scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)
    print(f"[EXQ-909] seed={seed} world_obs_dim={probe_env.world_obs_dim}"
          f" body_obs_dim={probe_env.body_obs_dim} draws_per_cycle={agent.sleep_loop.draws_per_cycle}"
          f" routing_gate_live={agent.sleep_loop.routing_gate is not None}", flush=True)

    diag = _run_curriculum(agent, scheduler, device, seed, total_eps)

    eval_eps   = 3 if dry_run else EVAL_EPISODES
    eval_steps = 30 if dry_run else EVAL_STEPS
    eval_env = _build_eval_env(scaffold_cfg, seed=seed)
    env_config_snapshot = _env_config_snapshot(eval_env)
    ree = _observational_run_with_forced_sleep(agent, eval_env, eval_eps, eval_steps, seed)

    n_firings = len(ree["firing_records"])
    print(f"[EXQ-909] seed={seed} firings={n_firings} eval_steps={ree['eval_steps']}", flush=True)

    seed_pass = bool(n_firings >= (2 if dry_run else max(1, EVAL_EPISODES - 1)) and agent.sleep_loop.draws_per_cycle > 0)
    print(f"verdict: {'PASS' if seed_pass else 'FAIL'} seed={seed} n_firings={n_firings}", flush=True)

    return {
        "seed": seed, "diag": diag, "firing_records": ree["firing_records"],
        "eval_steps": ree["eval_steps"], "episodes": ree["episodes"],
        "draws_per_cycle": int(agent.sleep_loop.draws_per_cycle),
        "routing_gate_live": bool(agent.sleep_loop.routing_gate is not None),
        "env_config": env_config_snapshot, "agent": agent,
    }


def run(seeds=None, dry_run: bool = False) -> dict:
    if seeds is None:
        seeds = DEFAULT_SEEDS
    print(f"[V3-EXQ-909] Sleep-Refinement DV Probe (multi-firing, multi-seed)\n"
          f"  Seeds: {seeds}  Eval: {EVAL_EPISODES} segments x up to {EVAL_STEPS} steps/seed"
          f" (force_cycle() at every boundary)\n"
          f"  Output: REE_assembly/evidence/experiments/{EXPERIMENT_TYPE}/", flush=True)

    seed_results = [run_seed(s, dry_run=dry_run) for s in seeds]
    agents = [r.pop("agent") for r in seed_results]

    all_firings: List[Dict] = []
    for r in seed_results:
        all_firings.extend(r["firing_records"])
    n_firings = len(all_firings)

    draws_per_cycle_min = min((r["draws_per_cycle"] for r in seed_results), default=0)
    routing_gate_all_live = all(r["routing_gate_live"] for r in seed_results)

    mean_sws_writes = (sum(f["sws_n_writes"] for f in all_firings) / n_firings) if n_firings else 0.0
    mean_rem_rollouts = (sum(f["rem_n_rollouts"] for f in all_firings) / n_firings) if n_firings else 0.0
    mean_waking_entropy = (sum(f["waking_mode_entropy_bits_prior_segment"] for f in all_firings) / n_firings) if n_firings else 0.0

    waking_diversity_present   = bool(mean_waking_entropy >= WAKING_ENTROPY_FLOOR_BITS)
    sws_write_engaged          = bool(mean_sws_writes >= SWS_WRITES_FLOOR)
    rem_rollout_engaged        = bool(mean_rem_rollouts >= REM_ROLLOUTS_FLOOR)
    draws_per_cycle_reachable  = bool(draws_per_cycle_min >= DRAWS_PER_CYCLE_FLOOR)
    sufficient_firings         = bool(n_firings >= MIN_FIRINGS_FLOOR)

    apparatus_ready = bool(waking_diversity_present and sws_write_engaged and rem_rollout_engaged
                           and draws_per_cycle_reachable and sufficient_firings and routing_gate_all_live)

    sws_slot_div_vals    = [f["sws_slot_diversity"] for f in all_firings]
    replay_div_vals      = [f["replay_diversity_index"] for f in all_firings]
    waking_entropy_vals  = [f["waking_mode_entropy_bits_prior_segment"] for f in all_firings]
    waking_cells_vals    = [f["waking_unique_cells_prior_segment"] for f in all_firings]

    def _dist(vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {"min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
        s = sorted(vals)
        n = len(s)
        median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0
        return {"min": float(s[0]), "mean": float(sum(s) / n), "median": float(median), "max": float(s[-1])}

    sws_slot_div_dist = _dist(sws_slot_div_vals)
    replay_div_dist   = _dist(replay_div_vals)

    n_sentinel = sum(1 for v in replay_div_vals if v <= -1.0 + 1e-9)
    frac_sentinel = (n_sentinel / n_firings) if n_firings else 0.0

    def _is_nonnull(f: Dict) -> bool:
        return bool(f["sws_slot_diversity"] > SWS_SLOT_DIVERSITY_EPS
                     or (f["replay_diversity_index"] > REPLAY_DIVERSITY_EPS))

    n_nonnull = sum(1 for f in all_firings if _is_nonnull(f))
    frac_nonnull = (n_nonnull / n_firings) if n_firings else 0.0

    r_entropy_vs_slot   = _pearson_r(waking_entropy_vals, sws_slot_div_vals)
    r_entropy_vs_replay = _pearson_r(waking_entropy_vals, replay_div_vals)
    r_cells_vs_slot      = _pearson_r([float(c) for c in waking_cells_vals], sws_slot_div_vals)
    r_cells_vs_replay    = _pearson_r([float(c) for c in waking_cells_vals], replay_div_vals)

    if not apparatus_ready:
        label = "substrate_not_ready_requeue"
        passed = False
    elif frac_nonnull >= NONNULL_FRAC_THRESHOLD:
        label = "sleep_dv_nonnull_detected"
        passed = True
    else:
        label = "sleep_dv_conversion_blocked"
        passed = True
    outcome = "PASS" if passed else "FAIL"

    metrics: Dict[str, Any] = {
        "n_seeds": float(len(seeds)),
        "n_firings_total": float(n_firings),
        "mean_waking_mode_entropy_bits": float(mean_waking_entropy),
        "mean_sws_n_writes": float(mean_sws_writes),
        "mean_rem_n_rollouts": float(mean_rem_rollouts),
        "draws_per_cycle_min_across_seeds": float(draws_per_cycle_min),
        "routing_gate_live_all_seeds": 1.0 if routing_gate_all_live else 0.0,
        "sws_slot_diversity_min": sws_slot_div_dist["min"],
        "sws_slot_diversity_mean": sws_slot_div_dist["mean"],
        "sws_slot_diversity_median": sws_slot_div_dist["median"],
        "sws_slot_diversity_max": sws_slot_div_dist["max"],
        "replay_diversity_index_min": replay_div_dist["min"],
        "replay_diversity_index_mean": replay_div_dist["mean"],
        "replay_diversity_index_median": replay_div_dist["median"],
        "replay_diversity_index_max": replay_div_dist["max"],
        "frac_firings_replay_diversity_sentinel": float(frac_sentinel),
        "frac_firings_nonnull": float(frac_nonnull),
        "r_waking_entropy_vs_sws_slot_diversity": r_entropy_vs_slot if r_entropy_vs_slot is not None else 0.0,
        "r_waking_entropy_vs_sws_slot_diversity_defined": 1.0 if r_entropy_vs_slot is not None else 0.0,
        "r_waking_entropy_vs_replay_diversity_index": r_entropy_vs_replay if r_entropy_vs_replay is not None else 0.0,
        "r_waking_entropy_vs_replay_diversity_index_defined": 1.0 if r_entropy_vs_replay is not None else 0.0,
        "r_waking_cells_vs_sws_slot_diversity": r_cells_vs_slot if r_cells_vs_slot is not None else 0.0,
        "r_waking_cells_vs_sws_slot_diversity_defined": 1.0 if r_cells_vs_slot is not None else 0.0,
        "r_waking_cells_vs_replay_diversity_index": r_cells_vs_replay if r_cells_vs_replay is not None else 0.0,
        "r_waking_cells_vs_replay_diversity_index_defined": 1.0 if r_cells_vs_replay is not None else 0.0,
    }
    for r in seed_results:
        s = r["seed"]
        metrics[f"seed{s}_n_firings"] = float(len(r["firing_records"]))
        metrics[f"seed{s}_draws_per_cycle"] = float(r["draws_per_cycle"])

    interpretation = {
        "label": label,
        "preconditions": [
            {"name": "waking_diversity_present",
             "description": "mean waking mode-entropy across all just-completed segments clears "
                             "a generous floor -- confirms there is diversity for sleep to have "
                             "converted, well below 906b's own measured 1.88-2.10 bits/segment",
             "measured": float(mean_waking_entropy), "threshold": WAKING_ENTROPY_FLOOR_BITS,
             "direction": "lower", "met": waking_diversity_present},
            {"name": "sws_write_mechanism_engaged",
             "description": "mean sws_n_writes across firings -- confirms the SWS schema-pass "
                             "write path actually ran, not just that the cycle nominally fired",
             "measured": float(mean_sws_writes), "threshold": SWS_WRITES_FLOOR,
             "direction": "lower", "met": sws_write_engaged},
            {"name": "rem_rollout_mechanism_engaged",
             "description": "mean rem_n_rollouts across firings -- confirms REM terrain rollouts "
                             "actually ran",
             "measured": float(mean_rem_rollouts), "threshold": REM_ROLLOUTS_FLOOR,
             "direction": "lower", "met": rem_rollout_engaged},
            {"name": "replay_draw_mechanism_structurally_reachable",
             "description": "agent.sleep_loop.draws_per_cycle (structural, set once at agent "
                             "construction) -- on 906b's UNMODIFIED config this is 0 on every "
                             "seed, making replay_diversity_index permanently the -1.0 zero-draws "
                             "sentinel regardless of waking diversity (confirmed empirically "
                             "before writing this script -- see module docstring). This precondition "
                             "verifies THIS run's config fix actually took effect.",
             "measured": float(draws_per_cycle_min), "threshold": DRAWS_PER_CYCLE_FLOOR,
             "direction": "lower", "met": draws_per_cycle_reachable, "control": "structural, all seeds"},
            {"name": "sufficient_firing_sample",
             "description": "total sleep-cycle firings across all seeds -- the review's own bar "
                             "for what an n=1 ad-hoc read cannot establish",
             "measured": float(n_firings), "threshold": MIN_FIRINGS_FLOOR,
             "direction": "lower", "met": sufficient_firings},
        ],
        "criteria_non_degenerate": {
            "waking_diversity_present": waking_diversity_present,
            "sws_write_mechanism_engaged": sws_write_engaged,
            "rem_rollout_mechanism_engaged": rem_rollout_engaged,
            "replay_draw_mechanism_structurally_reachable": draws_per_cycle_reachable,
            "sufficient_firing_sample": sufficient_firings,
            "routing_gate_live_all_seeds": routing_gate_all_live,
        },
        "criteria": [
            {"name": "waking_diversity_present", "load_bearing": True, "passed": waking_diversity_present},
            {"name": "sws_write_mechanism_engaged", "load_bearing": True, "passed": sws_write_engaged},
            {"name": "rem_rollout_mechanism_engaged", "load_bearing": True, "passed": rem_rollout_engaged},
            {"name": "replay_draw_mechanism_structurally_reachable", "load_bearing": True,
             "passed": draws_per_cycle_reachable},
            {"name": "sufficient_firing_sample", "load_bearing": True, "passed": sufficient_firings},
            {"name": "routing_gate_live_all_seeds", "load_bearing": True, "passed": routing_gate_all_live},
        ],
        "note": (
            f"Discrimination rule (pre-registered): a firing is 'non-null' if "
            f"sws_slot_diversity > {SWS_SLOT_DIVERSITY_EPS} OR replay_diversity_index > "
            f"{REPLAY_DIVERSITY_EPS} (i.e. real, non-sentinel positive value). "
            f"frac_nonnull={frac_nonnull:.3f} across {n_firings} firings; label='sleep_dv_nonnull_detected' "
            f"iff frac_nonnull >= {NONNULL_FRAC_THRESHOLD}, else 'sleep_dv_conversion_blocked'. "
            "Diagnostic discrimination for sleep_substrate_plan.md GAP-2 (does NOT itself unblock "
            "GAP-2 either way) -- claim_ids=[]; does not weight governance. See module docstring "
            "for the config delta vs V3-EXQ-906b and why it does not touch the waking-diversity "
            "substrate under test."
        ),
    }

    summary_markdown = f"""# V3-EXQ-909 -- Sleep-Refinement DV Probe (multi-firing, multi-seed)

**Status:** {outcome} -- label: `{label}`
**Purpose:** diagnostic discrimination, successor probe to V3-EXQ-906b (observational review
Section 13-C track C). Does the sleep-refinement DV (SWS slot diversity / REM replay
diversity) register a non-null waking->sleep difference on the repertoire-diverse-but-
non-converting fishtank substrate?

- seeds: {seeds}
- total sleep-cycle firings: {n_firings} (target >= {int(MIN_FIRINGS_FLOOR)})
- draws_per_cycle (structural, min across seeds): {draws_per_cycle_min}
- mean waking mode-entropy across just-completed segments: {mean_waking_entropy:.4f} bits
- sws_slot_diversity: min={sws_slot_div_dist['min']:.4f} mean={sws_slot_div_dist['mean']:.4f} median={sws_slot_div_dist['median']:.4f} max={sws_slot_div_dist['max']:.4f}
- replay_diversity_index: min={replay_div_dist['min']:.4f} mean={replay_div_dist['mean']:.4f} median={replay_div_dist['median']:.4f} max={replay_div_dist['max']:.4f}
- fraction of firings at the -1.0 zero-draws sentinel: {frac_sentinel:.3f}
- fraction of firings classified non-null (pre-registered epsilons): {frac_nonnull:.3f}
- r(waking mode-entropy, sws_slot_diversity): {r_entropy_vs_slot}
- r(waking mode-entropy, replay_diversity_index): {r_entropy_vs_replay}

See `interpretation.note` for the pre-registered discrimination rule and `sleep_firing_records`
in the episode-log companion file for the full per-firing table.
"""

    first_env_config = seed_results[0].get("env_config", {}) if seed_results else {}
    episode_log = {
        "experiment_type": EXPERIMENT_TYPE,
        "phase": "sleep_dv_multifiring_probe",
        "env_config": first_env_config,
        "sleep_firing_records": all_firings,
        "seeds": [{"seed": r["seed"], "episodes": r.get("episodes", [])} for r in seed_results],
    }

    return {
        "status": outcome, "outcome": outcome, "metrics": metrics,
        "summary_markdown": summary_markdown, "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE, "evidence_direction": "non_contributory",
        "experiment_type": EXPERIMENT_TYPE, "interpretation": interpretation,
        "episode_log": episode_log, "config": first_env_config, "agents": agents,
    }


if __name__ == "__main__":
    import argparse
    import json
    import time
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
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
        # See V3-EXQ-906b module docstring point 6: companion path must be declared relative
        # to write_flat_manifest's out_dir (out_dir.parent below), not to out_dir itself.
        result["companion_files"] = [f"{EXPERIMENT_TYPE}/{log_path.name}"]

    out_path = write_flat_manifest(
        result,
        out_dir.parent,
        dry_run=args.dry_run,
        config=result.get("config"),
        seeds=(args.seeds if args.seeds is not None else DEFAULT_SEEDS),
        script_path=Path(__file__),
        started_at=t0,
        agent=(agents[0] if len(agents) == 1 else agents) if agents else None,
    )
    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    print(f"final_outcome: {result['outcome']}", flush=True)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    emit_outcome(outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
                 manifest_path=out_path,
                 dry_run=bool(args.dry_run))
