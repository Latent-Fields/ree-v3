"""
V3-EXQ-906c -- Full-Stack Observational Fishtank: Appetitive-Sequence + Coupling
Instrumentation (Section 9 item 1 of the 906b observational review)

Claims: None (diagnostic showcase; does not weight governance)

EXPERIMENT_PURPOSE = "diagnostic"

SLEEP DRIVER: manual-multi (unchanged from 906b -- agent.sleep_loop.notify_episode_end()
called directly every segment boundary via _segment_boundary_consolidate(), imported from
906b. Still fires on the K=10 cadence; sleep_loop_episodes_K=10.)

WHY THIS RUN. Routing source: Section 9 item 1 of
REE_assembly/evidence/planning/observational_review_V3-EXQ-906b_2026-08-09.md ("V3-EXQ-906c
-- appetitive-sequence + coupling instrumentation"). Section 4 of that review computed five
affect->behaviour/event Pearson correlations POST-HOC by re-reading 906b's raw episode log
in a throwaway script (dread->harm, z_goal->approach, z_goal->benefit, dread<->z_harm_a,
excite<->benefit -- all near-zero, the review's central novel finding: affect telemetry
varies but does not predict subsequent action). Section 12b did the same for a sixth
relationship (residue_surprise spike -> mode-change / movement, a naive contingency-table
read). None of these six numbers exist in any manifest today (confirmed by grepping
evidence/experiments/ for their field names before writing this script -- GOV-REUSE-1). This
run's only job is to make them FIRST-CLASS, pre-registered, in-run manifest fields on the
SAME ecology 906b already validated survivable, so a future reader gets them for free instead
of re-deriving them from a raw episode log every time.

WHAT CHANGED VS 906b, AND WHAT DID NOT.

  1. ECOLOGY UNCHANGED. This driver imports `_make_config`, `_build_eval_env`,
     `_env_config_snapshot`, `_observational_run`, `EVAL_EPISODES`, `EVAL_STEPS`,
     `CORE_CHANNELS`, `STD_FLOOR`, `TRAIN_TOTAL_EPS` directly from
     v3_exq_906b_full_stack_observational_fishtank -- same curriculum, same eval-env
     proximity-radius/safe-spawn fixes, same all-ON module stack, same PASS/FAIL gates
     (core_channels_non_degenerate, harm_pathway_trained, ecology_survivable, all
     load-bearing, thresholds unchanged). This is deliberately a thin driver, not a
     re-derivation of 906b's ~700-line eval loop -- see that module's own docstring for the
     full proximity-radius-fix rationale, unchanged here.

  2. PART (a) OF THE SECTION 9 DESIGN WAS ALREADY LANDED BY A SIBLING CHIP BEFORE THIS
     SCRIPT WAS WRITTEN (chip-20260809-906b-surprise-telemetry, landed
     2026-08-09T19:00:45Z, REE_assembly master befb35db27 + ree-v3 main
     730281cd4b+bd90e8ddbe). `_read_affect` (v3_exq_664_affective_fishtank_showcase.py:245)
     now reads residue-map indices 0/1/3 (VALENCE_WANTING/LIKING/SURPRISE) alongside the
     pre-existing 4/5 (excite/dread), and 906b's own per-step `ep_steps` dict (inherited
     unchanged here via `_observational_run`) already carries `residue_wanting`, `liking`,
     and `surprise` per step. This script does NOT re-touch `_read_affect` or
     `_observational_run` -- it only adds a MANIFEST-LEVEL aggregation those per-step values
     were still missing (mean/std across all seeds/episodes/steps), computed independently
     by scanning `ree["episodes"]`'s already-collected per-step dicts (`_extra_channel_stats`
     below). `_observational_run`'s own `chan_vals`/`chan_std`/`chan_mean` machinery is left
     untouched -- widening its hardcoded channel list would require copying that whole
     ~150-line function; scanning the already-returned episode data afterward is equivalent
     and far cheaper.

  3. PART (b), THE NEW WORK: SIX AFFECT->BEHAVIOUR COUPLING METRICS AS FIRST-CLASS MANIFEST
     `metrics` FIELDS (`_compute_coupling_metrics` below), computed AFTER `_observational_run`
     returns, pooling (x, y) pairs across ALL seeds/episodes, WITHIN-EPISODE ONLY (no
     cross-episode-boundary lag -- matching the review's own Section 4/12b methodology, which
     explicitly notes "over all 8 segments (per-segment, so no boundary artefacts)"):
       - coupling_dread_t_to_harm_t1t3_{r,n}        -- Pearson r, dread[t] vs any harm_event
         (harm_signal<0) in t+1..t+3 (window truncated at episode end).
       - coupling_zgoal_t_to_approach_t1_{r,n}       -- Pearson r, z_goal[t] vs mode[t+1]=="approach".
       - coupling_zgoal_t_to_benefit_t1t3_{r,n}      -- Pearson r, z_goal[t] vs any harm_signal>0
         in t+1..t+3.
       - coupling_dread_zharma_contemporaneous_{r,n} -- Pearson r, dread[t] vs z_harm_a[t], same t.
       - coupling_excite_benefit_contemporaneous_{r,n} -- Pearson r, excite[t] vs harm_signal[t]
         (continuous, signed; "benefit signal" = raw harm_signal per the review's own Section
         2b operational definition), same t. Plus `coupling_excite_channel_reliable=0.0` --
         see point 4 below.
       - coupling_surprise_spike_{p90,p95,p99}_*     -- within-episode contingency proportions
         (P(mode-change @ t+1 | spike) vs P(.. | no spike), and same for P(moved @ t+1 | ..)),
         using the review's own Section 12b empirical thresholds (0.040/0.084/0.233) as
         starting-point anchors on `residue_surprise` (see point 5) plus n_spike/n_no_spike and
         the threshold value itself.
     All Pearson computations degrade gracefully to r=0.0 on a zero-variance or n<2 input
     rather than raising (see `_pearson_r`) -- this is a diagnostic showcase, not a claim test,
     so a degenerate coupling on a short/edge-case run should read as "uninformative", not
     crash the manifest write.

  4. PART (c), THE HOLD-vs-QUEUE JUDGEMENT CALL. `SD-RESIDUE-VALENCE-BOUND` (the 906a-autopsy-
     routed, unclamped-accumulator fix for `RBFLayer.update_valence()` -- see 906b's own
     module docstring point 7 and Section 3d/4 of the review) had NOT landed in
     REE_assembly/docs/claims/claims.yaml as of this script's authoring (grepped empty at
     authoring time). Decision: queue now rather than hold on an unrelated /governance
     dependency with unknown ETA -- excite-based coupling is only ONE of six new metrics, and
     rather than silently trust a contaminated channel this script computes it anyway (for
     continuity with the review's own Section 4 figure) but flags it explicitly:
     `coupling_excite_channel_reliable=0.0` in `metrics`, plus prose in both
     `interpretation.preconditions[]` and `summary_markdown`. Re-run this driver once
     SD-RESIDUE-VALENCE-BOUND lands if a trustworthy excite reading is needed.

  5. residue_surprise VS surprise -- TWO DISTINCT CHANNELS, NOT A TYPO. The surprise-spike
     couplings in point 3 read `residue_surprise` (`residue_metrics.get("mech205_surprise")`,
     already logged per-step by 906b), which is the WRITE-TIME raw prediction-error magnitude
     computed every step in `agent.update_residue()` (ree_core/agent.py ~9617-9673,
     `surprise = max(0.0, pe_mag - self._pe_ema)`) -- this is the exact field the review's
     Section 12b analysed and the field its p90/p95/p99 thresholds (0.040/0.084/0.233) were
     computed against. It is DIFFERENT from the newly-surfaced `affect["surprise"]`
     (VALENCE_SURPRISE read back from `agent.residue_field.evaluate_valence(z_world)`), which
     is a READ-time, spatially-accumulated/decayed value at the agent's current position, only
     updated on a gated write. Using the wrong field with these thresholds would silently
     produce nonsense proportions -- confirmed by reading ree_core/agent.py before relying on
     this (per this skill's "re-verify substrate API correctness even if copied" rule).

  6. NO `supersedes`. This run does not correct or invalidate 906b's science (the ecology,
     config, and PASS/FAIL gates are byte-identical) -- it is purely additive instrumentation
     on the same design. 906b's own PASS result stands unchanged.

WHAT THIS RUN IS NOT: a claim test, a statistically powered multi-seed study, or a
substrate-readiness diagnostic for any single mechanism. Single seed by default (--seeds),
unchanged in kind from 906b.

Output:
  evidence/experiments/v3_exq_906c_full_stack_observational_fishtank/
    v3_exq_906c_full_stack_observational_fishtank_<ts>.json               (manifest)
    v3_exq_906c_full_stack_observational_fishtank_<ts>_episode_log.json   (fishtank feed)

Estimated runtime: unchanged in kind from 906b (same curriculum, same eval segment/step
budget) -- see the queue entry note.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np

from ree_core.agent import REEAgent
from experiments.scaffolded_sd054_onboarding import ScaffoldedSD054OnboardingScheduler, _build_env
from experiments.v3_exq_665_curriculum_affective_fishtank_showcase import (
    _make_scaffold_cfg,
    _run_curriculum,
)
from experiments.v3_exq_906b_full_stack_observational_fishtank import (
    _make_config,
    _build_eval_env,
    _env_config_snapshot,
    _observational_run,
    TRAIN_TOTAL_EPS,
    EVAL_EPISODES,
    EVAL_STEPS,
    CORE_CHANNELS,
    STD_FLOOR,
)
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE    = "v3_exq_906c_full_stack_observational_fishtank"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS: List[str] = []
# No `supersedes` -- see module docstring point 6.

# ---- Section 12b empirical anchors (residue_surprise distribution, 906b's own 3909-step
# run): p90=0.040, p95=0.084, p99=0.233. Starting points, not re-derived from THIS run --
# this driver's job is to make the coupling machinery first-class, not to re-fit thresholds.
SURPRISE_SPIKE_THRESHOLDS: Dict[str, float] = {"p90": 0.040, "p95": 0.084, "p99": 0.233}

# Manifest-level aggregation for the three channels _read_affect now surfaces per-step
# (module docstring point 2) but 906b's own chan_vals/chan_std/chan_mean never summarised.
EXTRA_CHANNELS: List[str] = ["residue_wanting", "liking", "surprise"]


# ---------------------------------------------------------------------------
# Coupling-metrics helpers (module docstring point 3)
# ---------------------------------------------------------------------------

def _pearson_r(xs: List[float], ys: List[float]) -> Tuple[float, int]:
    """Pooled Pearson r over paired (x, y) lists. Degrades to (0.0, n) on n<2 or
    zero-variance input rather than raising or returning NaN -- a diagnostic showcase must
    not crash the manifest write on an edge-case-short run."""
    n = len(xs)
    if n < 2:
        return 0.0, n
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return 0.0, n
    r = float(np.corrcoef(x, y)[0, 1])
    return (0.0 if np.isnan(r) else r), n


def _lagged_pairs(
    all_episode_steps: List[List[Dict]],
    x_key: str,
    y_positive_fn: Callable[[Dict], bool],
    horizon: int,
) -> Tuple[List[float], List[float]]:
    """Pool (x_t, y_t) pairs across all episodes: x_t = steps[t][x_key]; y_t = 1.0 if
    y_positive_fn(steps[k]) is True for any k in t+1..min(t+horizon, n-1), else 0.0.
    WITHIN-EPISODE ONLY (no cross-boundary lag -- matches Section 4/12b methodology).
    Skips t whose x value is None/missing, and the last step of an episode (no t+1)."""
    xs: List[float] = []
    ys: List[float] = []
    for steps in all_episode_steps:
        n = len(steps)
        for t in range(n - 1):
            xv = steps[t].get(x_key)
            if xv is None:
                continue
            window_end = min(t + horizon, n - 1)
            hit = any(y_positive_fn(steps[k]) for k in range(t + 1, window_end + 1))
            xs.append(float(xv))
            ys.append(1.0 if hit else 0.0)
    return xs, ys


def _contemporaneous_pairs(
    all_episode_steps: List[List[Dict]], x_key: str, y_key: str
) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for steps in all_episode_steps:
        for s in steps:
            xv, yv = s.get(x_key), s.get(y_key)
            if xv is None or yv is None:
                continue
            xs.append(float(xv))
            ys.append(float(yv))
    return xs, ys


def _spike_coupling(
    all_episode_steps: List[List[Dict]], spike_key: str, threshold: float
) -> Dict[str, float]:
    """WITHIN-EPISODE ONLY (module docstring point 3): for each t with t+1 in the same
    episode, classify spike := steps[t][spike_key] >= threshold, then compare P(mode-change
    @ t+1 | spike) vs P(.. | no spike) and P(moved @ t+1 | spike) vs P(.. | no spike).
    mode-change := mode[t+1] != mode[t]; moved := pos[t+1] != pos[t] (list compare)."""
    spike_mode_hits = spike_mode_total = 0
    nospike_mode_hits = nospike_mode_total = 0
    spike_moved_hits = spike_moved_total = 0
    nospike_moved_hits = nospike_moved_total = 0
    for steps in all_episode_steps:
        n = len(steps)
        for t in range(n - 1):
            sv = steps[t].get(spike_key)
            if sv is None:
                continue
            is_spike = float(sv) >= threshold
            mode_t, mode_t1 = steps[t].get("mode"), steps[t + 1].get("mode")
            pos_t, pos_t1 = steps[t].get("pos"), steps[t + 1].get("pos")
            mode_changed = bool(mode_t is not None and mode_t1 is not None and mode_t != mode_t1)
            moved = bool(pos_t is not None and pos_t1 is not None and list(pos_t) != list(pos_t1))
            if is_spike:
                spike_mode_total += 1
                spike_mode_hits += int(mode_changed)
                spike_moved_total += 1
                spike_moved_hits += int(moved)
            else:
                nospike_mode_total += 1
                nospike_mode_hits += int(mode_changed)
                nospike_moved_total += 1
                nospike_moved_hits += int(moved)

    def _rate(hits: int, total: int) -> float:
        return float(hits) / total if total > 0 else 0.0

    return {
        "n_spike": float(spike_mode_total),
        "n_no_spike": float(nospike_mode_total),
        "p_mode_change_given_spike": _rate(spike_mode_hits, spike_mode_total),
        "p_mode_change_given_no_spike": _rate(nospike_mode_hits, nospike_mode_total),
        "p_moved_given_spike": _rate(spike_moved_hits, spike_moved_total),
        "p_moved_given_no_spike": _rate(nospike_moved_hits, nospike_moved_total),
    }


def _compute_coupling_metrics(all_episode_steps: List[List[Dict]]) -> Dict[str, float]:
    """The six Section 4/12b couplings, as first-class manifest metrics (module docstring
    point 3). Pools across every seed's every episode's steps, passed in flattened."""
    metrics: Dict[str, float] = {}

    xs, ys = _lagged_pairs(
        all_episode_steps, "dread", lambda s: bool(s.get("harm_event")), horizon=3
    )
    r, n = _pearson_r(xs, ys)
    metrics["coupling_dread_t_to_harm_t1t3_r"] = r
    metrics["coupling_dread_t_to_harm_t1t3_n"] = float(n)

    xs, ys = _lagged_pairs(
        all_episode_steps, "z_goal", lambda s: s.get("mode") == "approach", horizon=1
    )
    r, n = _pearson_r(xs, ys)
    metrics["coupling_zgoal_t_to_approach_t1_r"] = r
    metrics["coupling_zgoal_t_to_approach_t1_n"] = float(n)

    xs, ys = _lagged_pairs(
        all_episode_steps,
        "z_goal",
        lambda s: (s.get("harm_signal") is not None and float(s["harm_signal"]) > 0.0),
        horizon=3,
    )
    r, n = _pearson_r(xs, ys)
    metrics["coupling_zgoal_t_to_benefit_t1t3_r"] = r
    metrics["coupling_zgoal_t_to_benefit_t1t3_n"] = float(n)

    xs, ys = _contemporaneous_pairs(all_episode_steps, "dread", "z_harm_a")
    r, n = _pearson_r(xs, ys)
    metrics["coupling_dread_zharma_contemporaneous_r"] = r
    metrics["coupling_dread_zharma_contemporaneous_n"] = float(n)

    # Module docstring point 4: excite is contaminated by the unbounded
    # RBFLayer.update_valence accumulator (SD-RESIDUE-VALENCE-BOUND, not yet landed as of
    # this run -- see interpretation.preconditions below). Computed for continuity with the
    # review's own Section 4 figure, but flagged unreliable via the sibling metric below.
    xs, ys = _contemporaneous_pairs(all_episode_steps, "excite", "harm_signal")
    r, n = _pearson_r(xs, ys)
    metrics["coupling_excite_benefit_contemporaneous_r"] = r
    metrics["coupling_excite_benefit_contemporaneous_n"] = float(n)
    metrics["coupling_excite_channel_reliable"] = 0.0

    for label, threshold in SURPRISE_SPIKE_THRESHOLDS.items():
        sc = _spike_coupling(all_episode_steps, "residue_surprise", threshold)
        for k, v in sc.items():
            metrics[f"coupling_surprise_spike_{label}_{k}"] = v
        metrics[f"coupling_surprise_spike_{label}_threshold"] = float(threshold)

    return metrics


def _extra_channel_stats(all_episode_steps: List[List[Dict]]) -> Dict[str, float]:
    """Manifest-level mean/std for residue_wanting/liking/surprise, pooled across all
    seeds/episodes/steps (module docstring point 2) -- 906b's own chan_vals machinery never
    summarised these three, only excite/dread/z_goal/etc."""
    vals: Dict[str, List[float]] = {k: [] for k in EXTRA_CHANNELS}
    for steps in all_episode_steps:
        for s in steps:
            for k in EXTRA_CHANNELS:
                v = s.get(k)
                if isinstance(v, (int, float)):
                    vals[k].append(float(v))
    out: Dict[str, float] = {}
    for k, lst in vals.items():
        out[f"{k}_mean"] = float(np.mean(lst)) if lst else 0.0
        out[f"{k}_std"] = float(np.std(lst)) if len(lst) >= 2 else 0.0
        out[f"{k}_n"] = float(len(lst))
    return out


def run_seed(seed: int, dry_run: bool = False) -> Dict[str, Any]:
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    device = torch.device("cpu")
    total_eps = (2 + 2 + 5 + 5 + 5) if dry_run else TRAIN_TOTAL_EPS

    print(f"\nSeed {seed} Condition full_stack_observational_showcase", flush=True)
    scaffold_cfg = _make_scaffold_cfg(dry_run)
    probe_env = _build_env(scaffold_cfg, "p2")
    probe_env.reset()

    agent = REEAgent(_make_config(probe_env)).to(device)
    scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)
    print(f"[EXQ-906c] seed={seed} world_obs_dim={probe_env.world_obs_dim}"
          f" body_obs_dim={probe_env.body_obs_dim} full-stack curriculum ON", flush=True)

    diag = _run_curriculum(agent, scheduler, device, seed, total_eps)

    eval_eps   = 2 if dry_run else EVAL_EPISODES
    eval_steps = 30 if dry_run else EVAL_STEPS
    eval_env = _build_eval_env(scaffold_cfg, seed=seed)
    env_config_snapshot = _env_config_snapshot(eval_env)
    ree = _observational_run(agent, eval_env, eval_eps, eval_steps, seed)

    print(f"[EXQ-906c] seed={seed} channel std: "
          + "  ".join(f"{k}={ree['chan_std'][k]:.4f}" for k in
                      ["z_harm_a", "z_harm_un", "drive", "z_goal", "vigor", "z_block", "excite", "dread"]),
          flush=True)
    print(f"[EXQ-906c] seed={seed} events: block={ree['block_steps']} "
          f"limb_damage={ree['limb_damage_events']} external_hazard={ree['external_hazard_events']} "
          f"world_rule_shift={ree['world_rule_shift_events']} sleep_cycles={ree['sleep_cycles_fired']} "
          f"health_deaths={ree['health_depleted_terminations']} step_cap_ends={ree['step_cap_terminations']} "
          f"spawn_retries={ree['total_spawn_retries']} spawn_exhausted={ree['spawn_exhausted_segments']}",
          flush=True)

    seed_core_ok = all(ree["chan_std"].get(k, 0.0) > STD_FLOOR for k in CORE_CHANNELS)
    harm_trained = (diag["p0_harm_train_steps"] + diag["hazard_harm_train_steps"]) > 0
    seed_pass = bool(seed_core_ok and harm_trained)
    print(f"verdict: {'PASS' if seed_pass else 'FAIL'} seed={seed} "
          f"core_ok={seed_core_ok} harm_trained={harm_trained}", flush=True)

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
                "surprise_write_count_cumulative": int(getattr(agent, "_surprise_write_count", 0)),
            }

    return {
        "seed": seed, "diag": diag, "chan_std": ree["chan_std"], "chan_mean": ree["chan_mean"],
        "freeze_fires": ree["freeze_fires"], "block_steps": ree["block_steps"],
        "limb_damage_events": ree["limb_damage_events"],
        "external_hazard_events": ree["external_hazard_events"],
        "world_rule_shift_events": ree["world_rule_shift_events"],
        "sleep_cycles_fired": ree["sleep_cycles_fired"],
        "health_depleted_terminations": ree["health_depleted_terminations"],
        "step_cap_terminations": ree["step_cap_terminations"],
        "eval_steps": ree["eval_steps"], "z_goal_eval_mean": ree["chan_mean"].get("z_goal", 0.0),
        "harm_trained": harm_trained, "episodes": ree["episodes"], "agent": agent,
        "env_config": env_config_snapshot,
        "residue_stats_final": residue_stats_final,
        "total_spawn_retries": ree["total_spawn_retries"],
        "spawn_exhausted_segments": ree["spawn_exhausted_segments"],
    }


def run(seeds=None, dry_run: bool = False) -> dict:
    if seeds is None:
        seeds = [0]
    print(f"[V3-EXQ-906c] Full-Stack Observational Fishtank: Appetitive-Sequence + Coupling "
          f"Instrumentation\n"
          f"  Seeds: {seeds}  curriculum: Stage-0/0b/P0/Stage-H/P1 + harm-pathway training "
          f"(unchanged from 906b)\n"
          f"  Train eps/seed: {TRAIN_TOTAL_EPS}  Eval: {EVAL_EPISODES} segments x up to "
          f"{EVAL_STEPS} steps (same ecology as 906b, unchanged)\n"
          f"  New: 3-channel manifest aggregation (residue_wanting/liking/surprise) + "
          f"6 affect->behaviour coupling metrics\n"
          f"  Output: REE_assembly/evidence/experiments/{EXPERIMENT_TYPE}/", flush=True)

    seed_results = [run_seed(s, dry_run=dry_run) for s in seeds]
    agents = [r.pop("agent") for r in seed_results]

    chan_keys = list(seed_results[0]["chan_std"].keys())
    chan_max_std = {k: max(r["chan_std"].get(k, 0.0) for r in seed_results) for k in chan_keys}
    chan_nondegen = {k: bool(chan_max_std[k] > STD_FLOOR) for k in chan_keys}
    total_harm_steps = sum(r["diag"]["p0_harm_train_steps"] + r["diag"]["hazard_harm_train_steps"]
                           for r in seed_results)
    total_block = sum(r["block_steps"] for r in seed_results)
    total_limb_damage = sum(r["limb_damage_events"] for r in seed_results)
    total_external_hazard = sum(r["external_hazard_events"] for r in seed_results)
    total_world_rule_shift = sum(r["world_rule_shift_events"] for r in seed_results)
    total_sleep_cycles = sum(r["sleep_cycles_fired"] for r in seed_results)
    total_health_deaths = sum(r["health_depleted_terminations"] for r in seed_results)
    total_step_cap_ends = sum(r["step_cap_terminations"] for r in seed_results)
    total_freeze = sum(r["freeze_fires"] for r in seed_results)
    total_steps = sum(r["eval_steps"] for r in seed_results)
    total_spawn_retries = sum(r["total_spawn_retries"] for r in seed_results)
    total_spawn_exhausted = sum(r["spawn_exhausted_segments"] for r in seed_results)
    z_goal_activated = any(r["z_goal_eval_mean"] > 1e-3 for r in seed_results)

    core_ok = all(chan_nondegen.get(k, False) for k in CORE_CHANNELS)
    harm_trained = total_harm_steps > 0
    freeze_not_locked = (total_freeze == 0) or (total_freeze < total_steps)
    # Unchanged from 906b: same pre-registered floor, testing the same ecology-survivability
    # question. See v3_exq_906b's own docstring for the 4x-906-baseline derivation.
    mean_realized_segment_steps = (total_steps / max(1, sum(len(r["episodes"]) for r in seed_results)))
    ecology_survivable = bool(mean_realized_segment_steps >= 4.0 * (447.0 / 30.0))

    # ---- module docstring points 2/3: new instrumentation, computed once over ALL
    # seeds'/episodes' steps, pooled. ----
    all_episode_steps: List[List[Dict]] = [
        ep["steps"] for r in seed_results for ep in r.get("episodes", []) if ep.get("steps")
    ]
    coupling_metrics = _compute_coupling_metrics(all_episode_steps)
    extra_channel_stats = _extra_channel_stats(all_episode_steps)
    coupling_metrics_computed = bool(
        coupling_metrics.get("coupling_dread_zharma_contemporaneous_n", 0.0) > 0.0
    )

    passed = bool(core_ok and harm_trained and ecology_survivable)
    outcome = "PASS" if passed else "FAIL"

    metrics: Dict[str, Any] = {"n_seeds": float(len(seeds)),
                               "total_harm_pathway_train_steps": float(total_harm_steps),
                               "total_block_steps": float(total_block),
                               "total_limb_damage_events": float(total_limb_damage),
                               "total_external_hazard_events": float(total_external_hazard),
                               "total_world_rule_shift_events": float(total_world_rule_shift),
                               "total_sleep_cycles_fired": float(total_sleep_cycles),
                               "total_health_depleted_terminations": float(total_health_deaths),
                               "total_step_cap_terminations": float(total_step_cap_ends),
                               "total_freeze_fires": float(total_freeze),
                               "total_eval_steps": float(total_steps),
                               "mean_realized_segment_steps": float(mean_realized_segment_steps),
                               "z_goal_activated_at_eval": 1.0 if z_goal_activated else 0.0,
                               "total_spawn_safe_retries": float(total_spawn_retries),
                               "total_spawn_safe_exhausted_segments": float(total_spawn_exhausted)}
    metrics.update(coupling_metrics)
    metrics.update(extra_channel_stats)
    for r in seed_results:
        s = r["seed"]
        metrics[f"seed{s}_stage0_z_goal_peak"] = float(r["diag"]["stage0_z_goal_peak"])
        metrics[f"seed{s}_hazard_survival_gate"] = 1.0 if r["diag"]["hazard_survival_gate"] else 0.0
        metrics[f"seed{s}_hazard_harm_eval_range"] = float(r["diag"]["hazard_harm_eval_range"])
        metrics[f"seed{s}_z_goal_eval_mean"] = float(r["z_goal_eval_mean"])
        rstats = r.get("residue_stats_final") or {}
        metrics[f"seed{s}_residue_total_residue_final"] = float(rstats.get("total_residue", 0.0))
        metrics[f"seed{s}_residue_active_centers_final"] = float(rstats.get("active_centers", 0))
        metrics[f"seed{s}_residue_surprise_write_count_final"] = float(
            rstats.get("surprise_write_count_cumulative", 0))
    for k in chan_keys:
        metrics[f"chan_max_std_{k}"] = float(chan_max_std[k])
        metrics[f"chan_mean_{k}"] = float(np.mean([r["chan_mean"].get(k, 0.0) for r in seed_results]))

    interpretation = {
        "label": "coupling_instrumentation_live" if passed
                 else "coupling_instrumentation_degenerate",
        "preconditions": [
            {"name": "harm_pathway_trained", "description": "harm-pathway co-training ran >=1 optimizer step",
             "measured": float(total_harm_steps), "threshold": 1.0, "direction": "lower",
             "met": bool(harm_trained)},
            {"name": "ecology_survivable",
             "description": "same eval-ecology precondition as 906b (unchanged config): "
                             "segments should run well past 906's ~14.9-step/segment "
                             "early-death signature (>=4x floor)",
             "measured": float(mean_realized_segment_steps),
             "threshold": float(4.0 * (447.0 / 30.0)), "direction": "lower",
             "met": bool(ecology_survivable)},
            {"name": "excite_channel_contaminated",
             "description": "SD-RESIDUE-VALENCE-BOUND (unclamped RBFLayer.update_valence "
                             "accumulator, 906a-autopsy-routed) had NOT landed in claims.yaml "
                             "as of this run -- coupling_excite_benefit_contemporaneous_r is "
                             "computed for continuity with the review's Section 4 figure but "
                             "must NOT be read as a trustworthy appetitive-anticipation "
                             "readout. Re-run once the fix lands.",
             "measured": 0.0, "threshold": 1.0, "direction": "lower",
             "met": False, "control": "governance-state grep at authoring time, not a "
                                       "measured runtime quantity"},
        ],
        "criteria_non_degenerate": {
            **{f"channel_{k}": chan_nondegen.get(k, False) for k in chan_keys},
            "harm_pathway_trained": harm_trained,
            "freeze_not_permanently_locked": freeze_not_locked,
            "ecology_survivable": ecology_survivable,
            "coupling_metrics_computed": coupling_metrics_computed,
        },
        "criteria": [
            {"name": "core_channels_non_degenerate", "load_bearing": True, "passed": core_ok},
            {"name": "harm_pathway_trained", "load_bearing": True, "passed": harm_trained},
            {"name": "ecology_survivable", "load_bearing": True, "passed": ecology_survivable},
            {"name": "freeze_not_locked", "load_bearing": False, "passed": freeze_not_locked},
            {"name": "coupling_metrics_computed", "load_bearing": False,
             "passed": coupling_metrics_computed},
            *[{"name": f"channel_{k}", "load_bearing": False, "passed": bool(chan_nondegen.get(k, False))}
              for k in chan_keys],
        ],
        "note": ("Appetitive-sequence + coupling-instrumentation iteration of the 906 "
                 "lineage (Section 9 item 1 of the 906b observational review), same ecology "
                 "as 906b (no supersedes -- purely additive instrumentation). PASS gates are "
                 "unchanged from 906b: harm-pathway training ran AND core affect channels "
                 "vary AND the eval ecology is survivable. `coupling_metrics_computed` is a "
                 "non-load-bearing smoke check that the new instrumentation actually produced "
                 "data. claim_ids=[]; does not weight governance."),
    }

    summary_markdown = f"""# V3-EXQ-906c -- Full-Stack Observational Fishtank: Appetitive-Sequence + Coupling Instrumentation

**Status:** {outcome} (diagnostic telemetry showcase -- not scored against any claim)
**Purpose:** Section 9 item 1 of the 906b observational review
(observational_review_V3-EXQ-906b_2026-08-09.md). Same ecology as 906b, unchanged (no
`supersedes`) -- this run adds two things: (1) manifest-level aggregation for the
residue_wanting/liking/surprise channels `_read_affect` now surfaces per-step (landed by a
sibling chip before this script was authored), and (2) six affect->behaviour lagged/
contemporaneous coupling metrics as first-class manifest fields, the same ones the review
computed post-hoc by re-reading 906b's raw episode log.

- harm-pathway train steps (total): {total_harm_steps}
- z_goal activated at eval: {z_goal_activated}
- eval steps (total): {total_steps}  across {EVAL_EPISODES} segments x up to {EVAL_STEPS} steps/seed
  (mean realized segment length: {mean_realized_segment_steps:.1f} steps -- unchanged gate from 906b)
- segment endings: health_depleted={total_health_deaths} step_cap={total_step_cap_ends}
- sleep cycles fired: {total_sleep_cycles}
- freeze fires (eval, motor-override relaxed): {total_freeze}

## New: appetitive-channel aggregation
- residue_wanting: mean={extra_channel_stats.get('residue_wanting_mean',0.0):.4f} std={extra_channel_stats.get('residue_wanting_std',0.0):.4f} (n={int(extra_channel_stats.get('residue_wanting_n',0))})
- liking: mean={extra_channel_stats.get('liking_mean',0.0):.4f} std={extra_channel_stats.get('liking_std',0.0):.4f} (n={int(extra_channel_stats.get('liking_n',0))})
- surprise (VALENCE_SURPRISE read-back): mean={extra_channel_stats.get('surprise_mean',0.0):.4f} std={extra_channel_stats.get('surprise_std',0.0):.4f} (n={int(extra_channel_stats.get('surprise_n',0))})

## New: affect->behaviour coupling metrics (Section 4/12b, now first-class)
- dread(t) -> harm in t+1..t+3: r={coupling_metrics.get('coupling_dread_t_to_harm_t1t3_r',0.0):.4f} (n={int(coupling_metrics.get('coupling_dread_t_to_harm_t1t3_n',0))})
- z_goal(t) -> approach at t+1: r={coupling_metrics.get('coupling_zgoal_t_to_approach_t1_r',0.0):.4f} (n={int(coupling_metrics.get('coupling_zgoal_t_to_approach_t1_n',0))})
- z_goal(t) -> benefit in t+1..t+3: r={coupling_metrics.get('coupling_zgoal_t_to_benefit_t1t3_r',0.0):.4f} (n={int(coupling_metrics.get('coupling_zgoal_t_to_benefit_t1t3_n',0))})
- dread <-> z_harm_a (contemporaneous): r={coupling_metrics.get('coupling_dread_zharma_contemporaneous_r',0.0):.4f} (n={int(coupling_metrics.get('coupling_dread_zharma_contemporaneous_n',0))})
- excite <-> benefit signal (contemporaneous, **UNRELIABLE -- see interpretation.preconditions, SD-RESIDUE-VALENCE-BOUND not yet landed**): r={coupling_metrics.get('coupling_excite_benefit_contemporaneous_r',0.0):.4f} (n={int(coupling_metrics.get('coupling_excite_benefit_contemporaneous_n',0))})
- surprise-spike (p90={SURPRISE_SPIKE_THRESHOLDS['p90']}) -> mode-change @ t+1: {coupling_metrics.get('coupling_surprise_spike_p90_p_mode_change_given_spike',0.0)*100:.1f}% (spike, n={int(coupling_metrics.get('coupling_surprise_spike_p90_n_spike',0))}) vs {coupling_metrics.get('coupling_surprise_spike_p90_p_mode_change_given_no_spike',0.0)*100:.1f}% (no spike, n={int(coupling_metrics.get('coupling_surprise_spike_p90_n_no_spike',0))})
- surprise-spike (p90) -> moved @ t+1: {coupling_metrics.get('coupling_surprise_spike_p90_p_moved_given_spike',0.0)*100:.1f}% (spike) vs {coupling_metrics.get('coupling_surprise_spike_p90_p_moved_given_no_spike',0.0)*100:.1f}% (no spike)

The `_episode_log.json` companion feeds fishtank_viz.html via /api/fishtank/logs, same as
906b -- unchanged episode_log schema, now also carrying residue_wanting/liking/surprise per
step (inherited from the sibling telemetry chip, not from this driver).
"""

    first_env_config = seed_results[0].get("env_config", {}) if seed_results else {}
    episode_log = {
        "experiment_type": EXPERIMENT_TYPE,
        "phase": "full_stack_observational_showcase",
        "toroidal": bool(first_env_config.get("toroidal", False)),
        "env_config": first_env_config,
        "seeds": [{"seed": r["seed"], "episodes": r.get("episodes", [])} for r in seed_results],
    }

    return {
        "status": outcome, "outcome": outcome, "metrics": metrics,
        "summary_markdown": summary_markdown, "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE, "evidence_direction": "non_contributory",
        "experiment_type": EXPERIMENT_TYPE, "interpretation": interpretation,
        "episode_log": episode_log, "agents": agents,
        "config": first_env_config,
    }


if __name__ == "__main__":
    import argparse
    import json
    import time
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
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
        # Same companion-path fix 906b applied (module docstring point 6 there): declared
        # path must be relative to write_flat_manifest's out_dir (out_dir.parent below).
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

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    emit_outcome(outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
                 manifest_path=out_path,
                 dry_run=bool(args.dry_run))
