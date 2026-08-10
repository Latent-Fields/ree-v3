"""
V3-EXQ-912 -- Uncensored Survival-to-Death Fishtank Successor (large-n death-time
characterization on the 906b/906c ecology tier)

Claims: None (diagnostic showcase; does not weight governance)

EXPERIMENT_PURPOSE = "diagnostic"

SLEEP DRIVER: manual-multi (unchanged, inherited unmodified from
v3_exq_906b_full_stack_observational_fishtank -- agent.sleep_loop.notify_episode_end() is
called by _observational_run()'s own _segment_boundary_consolidate() at every segment
boundary. Still fires on the K=10 cadence, sleep_loop_episodes_K=10. This run's much larger
segment count (60/seed vs 906b/906c/911's 8) means the cadence now fires ~6x per seed instead
of ~0-1x -- richer incidental sleep-cycle sampling, not this run's target DV.)

WHY THIS RUN. Routing source: chip-20260810-fishtank-uncensored-survival (spawned from
REE_assembly/evidence/planning/organism_lifespan_development_review_906_lineage_2026-08-10.md
Section 2 "Survivability and censoring" + Section 10 item 1). That review established, from
metrics.json/manifest.json/summary.md across the full lettered lineage, that right-censoring
has gone from heavy to TOTAL: 906a 0%, 906b 75%, 906c 87.5%, 911 100% (0/8 segments died).
Mean segment length has stopped measuring survival at all -- every run since 906b measures "we
stopped watching at 500 steps," not "the organism could not continue." The review's own
proposed successor (Section 10 item 1) asks for an uncensored design and explicitly requires
declaring, in this docstring, which of two death-unit interpretations is intended:
  (a) body-respawn-on-health_depleted IS the "death" unit -- report inter-respawn survival
      times, "which the current substrate already supports without any code change beyond
      removing/raising the cap and running longer" (the review's own words), or
  (b) add an actual terminal-death branch for a dedicated long-run variant.
This run picks a variant of (a) -- see "SUBSTRATE READINESS FINDING" below for why it is a
variant, not the literal design the review described, and why (b) is out of scope for
/queue-experiment regardless.

SUBSTRATE READINESS FINDING (Step 2.5a empirical probe -- the premise both the review and the
chip's own prompt state does NOT hold, verified against source before writing any of the rest
of this script). CausalGridWorldV2.step()'s per-episode step cap is a HARDCODED LITERAL, not a
configurable constructor parameter:
    causal_grid_world.py:3067   _step_cap_reached = self.steps >= 500
Grepped the whole class for `max_episode_steps` / `episode_step_cap` / any constructor kwarg
that could parameterize this -- none exists. `self.steps` resets to 0 only inside `env.reset()`
(causal_grid_world.py:1620,1975), which _observational_run() calls exactly once per segment
boundary. So a segment CANNOT exceed 500 steps regardless of the DRIVER's own
`steps_per_episode` argument to _observational_run(): once `env.step()` reports
`done_cause="step_limit"` at self.steps==500, the driver's `if done: break` fires immediately.
906b/906c/911's own EVAL_STEPS=500 was chosen to MATCH this substrate constant (906b's own
docstring footnote: "EVAL_STEPS=500 matches CausalGridWorldV2's own hard per-episode step
cap"), not because the driver enforces an independent, raisable ceiling. Concretely, this means
option (a) as the review LITERALLY describes it -- "remove/raise the cap and run longer" per
segment -- is NOT achievable via driver configuration alone; it requires parameterizing the
`500` literal in causal_grid_world.py, a substrate change. Option (b) (terminal-death branch)
is already acknowledged by the review/chip as needing new code. So NEITHER of the chip's two
literal options is buildable from /queue-experiment alone. Per this skill's Step 2.5a ("if the
probe contradicts the doc... do NOT write the script assuming the false premise... route to
/implement-substrate for the actual gap"), this substrate gap -- parameterize
CausalGridWorldV2's per-episode step cap (e.g. a `max_episode_steps` constructor kwarg,
default 500, replacing the literal at :3067, fully backward-compatible) -- is named precisely
here and chipped separately for /implement-substrate (see session-land closing note), NOT
built inside this driver.

THIS RUN'S ACTUAL DESIGN, given the constraint above (a genuinely different, but still
scientifically useful, driver-only-achievable variant of option (a)). Rather than lengthening
individual segments (impossible without the substrate change above), this run scales up the
NUMBER of segments per continuous life: EVAL_EPISODES=60 (vs 906b/906c/911's 8, a ~7.5x
increase), seeds=[0,1]. Every segment still independently terminates at genuine
`health_depleted` OR the substrate's fixed 500-step ceiling, exactly as before -- what changes
is the SAMPLE SIZE of that per-segment outcome, converting the prior single-run point estimate
(906b: 2/8 died -> mean_realized_segment_steps=488.6; 906c: 1/8; 911: 0/8) into a proper
survival-analysis characterization: the full list of UNCENSORED death times (segments that
ended `health_depleted`, i.e. occurred strictly before the 500-step ceiling and are therefore
genuine, non-censored observations) alongside the count and step-length of CENSORED segments
(hit the 500-step ceiling without dying), with `pct_right_censored` reported directly instead
of folded silently into a possibly-misleading mean. This does not eliminate the substrate's
per-segment 500-step ceiling -- an unremovable constant without the follow-on substrate change
-- but it converts n=8 (mostly-censored, 0-2 genuine events) into n=120 (60 segments x 2 seeds),
which should yield a real, usable empirical distribution of within-cap death times if this
ecology tier's death rate is anywhere near 906b's/906c's own observed 12.5-25%.

ECOLOGY CHOICE -- DELIBERATELY 906b's, NOT 911's (a design decision, not an oversight). This
driver reuses `_make_config`, `_env_config_snapshot`, `_observational_run`, and
`EVAL_ENV_EXTRA_KWARGS` UNMODIFIED from v3_exq_906b_full_stack_observational_fishtank, and
does NOT layer in V3-EXQ-911's additional resource-field retune
(`resource_field_decay=3.0`/`proximity_benefit_scale=0.01`). Reasoning: 911's own scored run
produced ZERO health_depleted terminations across its 8 segments (100% right-censored) --
running a 60-segment-per-seed characterization on 911's ecology risks observing zero or very
few genuine deaths, defeating this run's entire purpose (characterizing WHEN death happens, not
confirming it does not). 906b (2/8 genuine deaths) and 906c (1/8, a same-substrate replicate)
are the two configs in the observed lineage that still reliably produce `health_depleted`
terminations within the 500-step ceiling, making 906b's environment tuning the correct base
for a run whose primary DV is the death-time distribution itself, even though it is one
lettered iteration behind 911 on the (orthogonal) food-seeking-realism axis.

WELFARE INSTRUMENTATION (reef_ecology_strategy_affective_occupancy_review_2026-08-10.md
Section 9, "Long-life welfare instrumentation as experimental hygiene"). That review states:
"if a future successor gives REE a substantially longer or richer continuous life, add
lifetime affective-occupancy reporting... as a standard reported statistic." This run's 60
segments/seed is ~7.5x 906b/906c/911's 8 -- the review's own trigger condition -- so
`lifetime_affective_occupancy` (fraction of lived steps with dread / z_harm_a above this run's
own within-run 75th percentile, harm-event rate, in_reef fraction) is computed per seed and
reported as a NON-GATING descriptive statistic, per SENT-2 (governance.welfare.welfare_budget,
candidate, binds_at_version v4) hygiene. No inference of sentience, suffering, or welfare is
made; SENT-0 boundary applies exactly as it does to every other V3 diagnostic (see the ethics
preflight block at the end of this docstring).

CUMULATIVE STEP COUNTER (closes a small, explicitly-flagged gap). The organism-lifespan
review's Section 1 flagged: "no manifest or episode-log field stores a cumulative/monotonic
step counter across the whole run -- every segment's per-step `t` restarts at 0... a minor
telemetry gap, cheap to fix." This run stamps `cumulative_step_start`/`cumulative_step_end`
onto every segment record before writing (post-processing _observational_run()'s unmodified
return value -- no change to that function), and reports `cumulative_lived_steps` per seed at
the top level, directly serving this run's own "report cumulative lived steps" requirement
(review Section 10 item 1) while also closing the flagged gap for any future within-life
analysis.

EPISODE-LOG SIZE MANAGEMENT (a deliberate storage trade-off, stated explicitly rather than
left implicit). 906b's own committed episode_log for 8 segments (~3909 steps) is 9.5MB;
911's is 10.5MB (~1.19MB/segment at full per-step richness). Writing full per-step records for
all 120 segments (60/seed x 2 seeds) this run collects would produce a companion file on the
order of 140MB -- disproportionate to commit for a run whose primary DV (per-segment
done_cause + realized_steps) needs only a few hundred bytes per segment, not the full ~30-field
per-step array. So the WRITTEN companion `episode_log.json` keeps FULL per-step records only
for: (i) the first FULL_LOG_FIRST_K=3 segments per seed (visual-QA / fishtank_viz continuity,
matching what a human scrubbing the viz from the start of a life would want), and (ii) EVERY
segment that ended `health_depleted` (the actual event of scientific interest -- a future
reader inspecting what happened in the steps immediately before a genuine death needs the full
trace, not a summary). All other (step-limit-censored, non-early) segments are stored as a
SLIM summary (done_cause, realized_steps, cumulative markers, spawn/sleep telemetry, and
per-segment MEAN of the core affect channels -- computed from the full per-step array before
it is dropped). This thinning is applied ONLY to the file written to disk; every statistic
reported in the manifest (survival distribution, welfare occupancy, channel non-degeneracy) is
computed from the FULL, unthinned in-memory episodes_log _observational_run() returns, so
thinning cannot bias any reported number -- it only bounds what is persisted to the companion
file.

GOV-REUSE-1 (Step 2.4). Decisive readout: the empirical distribution of within-500-step-cap
`health_depleted` survival times at this ecology tier, at n far larger than 8. Searched
existing manifests: every 906-lineage run (906, 906a, 906b, 906c, 911) records at most 8
segments and reports only a single mean_realized_segment_steps figure per run, never a
distribution; no manifest anywhere records more than 1-2 genuine health_depleted events. Not
recoverable by reanalysis -- a run is required. (Matches the organism-lifespan review's own
Section 10 item 1 duplication check: "no existing queue entry, planning note, or claim
proposes this.")

Re-derive brake (2.5b): claim_ids=[] -- not applicable, no claim to brake-check.
Substrate-path overlap gate (2.5c): checked substrate_queue.json for open `corrupting` entries
touching causal_grid_world.py or the modules this driver imports -- none found (several
IMPLEMENTED entries reference causal_grid_world.py; none open/corrupting).

Ethics preflight (Step 2.6, descriptive, non-enforced): involves_negative_valence=false,
involves_suffering_like_state=false, involves_self_model=false,
involves_inescapability_or_helplessness=false, involves_offline_replay_over_harm=false,
involves_social_mind_or_language=false, involves_human_data_or_clinical_context=false,
decision=allow. V3 has no live self-model in E3, no autobiographical memory, no social mind,
no language -- unchanged by this run's larger segment count, which only lengthens EXPOSURE
within the same pre-ethical instrumentation, not the architecture. The welfare-occupancy
reporting above operationalizes SENT-2 hygiene given that longer exposure; it does not change
this preflight's allow decision.

WHAT THIS RUN IS NOT: a claim test, a terminal-death mechanic, or a design that removes the
substrate's per-segment 500-step ceiling (that requires the chipped /implement-substrate
follow-on named above). It is a large-n characterization of the death-time distribution
achievable within that ceiling, on the most recent ecology tier that reliably produces genuine
deaths.

Output:
  evidence/experiments/v3_exq_912_uncensored_survival_fishtank/
    v3_exq_912_uncensored_survival_fishtank_<ts>.json               (manifest)
    v3_exq_912_uncensored_survival_fishtank_<ts>_episode_log.json   (fishtank feed, thinned
                                                                      per "EPISODE-LOG SIZE
                                                                      MANAGEMENT" above)

Estimated runtime: see queue entry note (data-dependent -- bounded by the aggregate step
budget of up to 60*500=30,000 eval steps/seed if every segment reached the cap, but a 906b-like
~12.5-25% death rate makes the realistic total substantially lower).
"""

import random
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Any

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
from experiments.v3_exq_906b_full_stack_observational_fishtank import (
    _make_config,
    _build_eval_env,
    _env_config_snapshot,
    _observational_run,
    TRAIN_TOTAL_EPS,
    CORE_CHANNELS,
    STD_FLOOR,
)
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE    = "v3_exq_912_uncensored_survival_fishtank"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS: List[str] = []

# ---- large-n survival characterization: 60 segments/seed vs 906b/906c/911's 8. Per-segment
# step cap (500) is NOT raised -- see module docstring "SUBSTRATE READINESS FINDING": it is a
# CausalGridWorldV2-internal hardcoded literal, not a driver-configurable parameter.
EVAL_EPISODES = 60
EVAL_STEPS    = 500

# Pre-registered (Step 3 "pre-registered thresholds must be defined in the script, not
# inferred post-hoc"): minimum total genuine (uncensored) health_depleted events across ALL
# seeds combined for this run to have obtained a usable empirical death-time distribution.
# Chosen well above the 0-2 events any single prior 906-lineage run observed, but comfortably
# below the ~15-30 expected if this ecology's death rate is anywhere near 906b's 25% (2/8) or
# 906c's 12.5% (1/8) over 60*2=120 segments.
MIN_UNCENSORED_DEATHS_TOTAL = 10

# EPISODE-LOG SIZE MANAGEMENT (module docstring): keep full per-step records for the first K
# segments per seed (viz continuity) plus every genuinely-died segment; thin the rest.
FULL_LOG_FIRST_K = 3

# Lifetime affective-occupancy percentile (reef_ecology_strategy review Section 9).
OCCUPANCY_PERCENTILE = 75


def _thin_segment_for_storage(seg: Dict[str, Any], keep_full: bool) -> Dict[str, Any]:
    """Module docstring 'EPISODE-LOG SIZE MANAGEMENT'. Returns `seg` unchanged when
    `keep_full`; otherwise drops the (large) per-step `steps` list and replaces it with a
    small per-segment summary computed from it. Never mutates `seg` -- callers still need the
    original, full-richness dict for in-memory statistics computed BEFORE this is called."""
    if keep_full:
        return seg
    steps = seg.get("steps") or []
    summary = dict(seg)
    summary.pop("steps", None)
    summary["steps_stored"] = "thinned"
    if steps:
        for ch in ("dread", "z_harm_a", "z_harm_s", "drive", "z_goal", "vigor"):
            vals = [s.get(ch) for s in steps if isinstance(s.get(ch), (int, float))]
            summary[f"mean_{ch}"] = float(np.mean(vals)) if vals else None
        summary["frac_harm_event"] = float(np.mean([bool(s.get("harm_event")) for s in steps]))
        summary["frac_in_reef"] = float(np.mean([bool(s.get("in_reef")) for s in steps]))
    return summary


def _add_cumulative_markers(episodes_log: List[Dict[str, Any]]) -> int:
    """Organism-lifespan review Section 1 gap fix: stamp a monotonic step counter across
    segment boundaries directly onto each (already-built) segment record, in place. Returns
    the total cumulative step count (== the seed's total lived steps)."""
    cum = 0
    for seg in episodes_log:
        realized = int(seg.get("realized_steps", len(seg.get("steps", []))))
        seg["cumulative_step_start"] = cum
        seg["cumulative_step_end"] = cum + realized
        cum += realized
    return cum


def _survival_stats(episodes_log: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The primary DV of this run: the per-segment done_cause/realized_steps outcome,
    partitioned into genuine (uncensored, health_depleted) vs right-censored (step_limit)
    observations, per the module docstring's survival-analysis framing."""
    uncensored = [int(s["realized_steps"]) for s in episodes_log if s.get("done_cause") == "health_depleted"]
    censored   = [int(s["realized_steps"]) for s in episodes_log if s.get("done_cause") == "step_limit"]
    incomplete = [int(s["realized_steps"]) for s in episodes_log
                  if s.get("done_cause") not in ("health_depleted", "step_limit")]
    n_total = len(episodes_log)
    n_uncensored = len(uncensored)
    n_censored = len(censored)
    stats: Dict[str, Any] = {
        "n_segments": n_total,
        "n_uncensored_deaths": n_uncensored,
        "n_censored_step_limit": n_censored,
        "n_incomplete_other": len(incomplete),
        "pct_right_censored": float(n_censored / n_total) if n_total else 0.0,
        "uncensored_survival_times": uncensored,
        "uncensored_min": float(min(uncensored)) if uncensored else None,
        "uncensored_max": float(max(uncensored)) if uncensored else None,
        "uncensored_mean": float(statistics.fmean(uncensored)) if uncensored else None,
        "uncensored_median": float(statistics.median(uncensored)) if uncensored else None,
    }
    return stats


def _lifetime_affective_occupancy(episodes_log: List[Dict[str, Any]]) -> Dict[str, Any]:
    """reef_ecology_strategy_affective_occupancy_review_2026-08-10.md Section 9: descriptive,
    non-gating, within-run-percentile occupancy stats over the WHOLE continuous life (all
    segments concatenated, in order). No inference of welfare/sentience -- SENT-0 boundary."""
    all_dread: List[float] = []
    all_z_harm_a: List[float] = []
    all_harm_event: List[bool] = []
    all_in_reef: List[bool] = []
    for seg in episodes_log:
        for s in seg.get("steps", []) or []:
            if isinstance(s.get("dread"), (int, float)):
                all_dread.append(float(s["dread"]))
            if isinstance(s.get("z_harm_a"), (int, float)):
                all_z_harm_a.append(float(s["z_harm_a"]))
            all_harm_event.append(bool(s.get("harm_event")))
            all_in_reef.append(bool(s.get("in_reef")))
    n = len(all_harm_event)
    out: Dict[str, Any] = {"n_lived_steps_measured": n}
    if all_dread:
        p = float(np.percentile(all_dread, OCCUPANCY_PERCENTILE))
        out["dread_p75_threshold"] = p
        out["frac_steps_dread_above_p75"] = float(np.mean([v > p for v in all_dread]))
    if all_z_harm_a:
        p = float(np.percentile(all_z_harm_a, OCCUPANCY_PERCENTILE))
        out["z_harm_a_p75_threshold"] = p
        out["frac_steps_z_harm_a_above_p75"] = float(np.mean([v > p for v in all_z_harm_a]))
    if n:
        out["frac_harm_event"] = float(np.mean(all_harm_event))
        out["frac_in_reef"] = float(np.mean(all_in_reef))
    return out


def run_seed(seed: int, dry_run: bool = False) -> Dict[str, Any]:
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    device = torch.device("cpu")
    total_eps = (2 + 2 + 5 + 5 + 5) if dry_run else TRAIN_TOTAL_EPS

    print(f"\nSeed {seed} Condition uncensored_survival_characterization", flush=True)
    scaffold_cfg = _make_scaffold_cfg(dry_run)
    probe_env = _build_env(scaffold_cfg, "p2")
    probe_env.reset()
    agent = REEAgent(_make_config(probe_env)).to(device)
    scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)
    print(f"[EXQ-912] seed={seed} world_obs_dim={probe_env.world_obs_dim}"
          f" body_obs_dim={probe_env.body_obs_dim} full-stack curriculum ON (906b ecology)", flush=True)

    diag = _run_curriculum(agent, scheduler, device, seed, total_eps)

    eval_eps   = 3 if dry_run else EVAL_EPISODES
    eval_steps = 30 if dry_run else EVAL_STEPS
    eval_env = _build_eval_env(scaffold_cfg, seed=seed)
    env_config_snapshot = _env_config_snapshot(eval_env)
    ree = _observational_run(agent, eval_env, eval_eps, eval_steps, seed)

    episodes_log = ree["episodes"]
    cumulative_lived_steps = _add_cumulative_markers(episodes_log)
    survival = _survival_stats(episodes_log)
    occupancy = _lifetime_affective_occupancy(episodes_log)

    print(f"[EXQ-912] seed={seed} channel std: "
          + "  ".join(f"{k}={ree['chan_std'][k]:.4f}" for k in
                      ["z_harm_a", "z_harm_un", "drive", "z_goal", "vigor", "z_block", "excite", "dread"]),
          flush=True)
    print(f"[EXQ-912] seed={seed} segments={survival['n_segments']} "
          f"uncensored_deaths={survival['n_uncensored_deaths']} "
          f"censored={survival['n_censored_step_limit']} "
          f"pct_right_censored={survival['pct_right_censored']:.3f} "
          f"cumulative_lived_steps={cumulative_lived_steps}", flush=True)

    seed_core_ok = all(ree["chan_std"].get(k, 0.0) > STD_FLOOR for k in CORE_CHANNELS)
    harm_trained = (diag["p0_harm_train_steps"] + diag["hazard_harm_train_steps"]) > 0
    segments_varied = len(set(int(s["realized_steps"]) for s in episodes_log)) > 1
    seed_pass = bool(seed_core_ok and harm_trained)
    print(f"verdict: {'PASS' if seed_pass else 'FAIL'} seed={seed} "
          f"core_ok={seed_core_ok} harm_trained={harm_trained}", flush=True)

    # Episode-log thinning for the WRITTEN companion file only (module docstring "EPISODE-LOG
    # SIZE MANAGEMENT"). Statistics above were already computed from the full, unthinned data.
    stored_episodes = []
    for i, seg in enumerate(episodes_log):
        keep_full = (i < FULL_LOG_FIRST_K) or (seg.get("done_cause") == "health_depleted")
        stored_episodes.append(_thin_segment_for_storage(seg, keep_full))
    n_full_kept = sum(1 for s in stored_episodes if s.get("steps_stored") != "thinned")

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
        "seed": seed, "diag": diag, "chan_std": ree["chan_std"], "chan_mean": ree["chan_mean"],
        "freeze_fires": ree["freeze_fires"], "block_steps": ree["block_steps"],
        "limb_damage_events": ree["limb_damage_events"],
        "external_hazard_events": ree["external_hazard_events"],
        "world_rule_shift_events": ree["world_rule_shift_events"],
        "sleep_cycles_fired": ree["sleep_cycles_fired"],
        "eval_steps": ree["eval_steps"], "z_goal_eval_mean": ree["chan_mean"].get("z_goal", 0.0),
        "harm_trained": harm_trained, "segments_varied": segments_varied,
        "episodes_full": episodes_log, "episodes_stored": stored_episodes,
        "n_full_kept": n_full_kept,
        "agent": agent, "env_config": env_config_snapshot,
        "residue_stats_final": residue_stats_final,
        "total_spawn_retries": ree["total_spawn_retries"],
        "spawn_exhausted_segments": ree["spawn_exhausted_segments"],
        "survival": survival, "occupancy": occupancy,
        "cumulative_lived_steps": cumulative_lived_steps,
    }


def run(seeds=None, dry_run: bool = False) -> dict:
    if seeds is None:
        seeds = [0]
    print(f"[V3-EXQ-912] Uncensored Survival-to-Death Fishtank Successor\n"
          f"  Seeds: {seeds}  curriculum: Stage-0/0b/P0/Stage-H/P1 + harm-pathway training\n"
          f"  Train eps/seed: {TRAIN_TOTAL_EPS}  Eval: {EVAL_EPISODES} segments x up to {EVAL_STEPS} "
          f"steps (906b ecology, per-segment cap is a substrate constant -- see module docstring)\n"
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
    total_freeze = sum(r["freeze_fires"] for r in seed_results)
    total_steps = sum(r["eval_steps"] for r in seed_results)
    total_spawn_retries = sum(r["total_spawn_retries"] for r in seed_results)
    total_spawn_exhausted = sum(r["spawn_exhausted_segments"] for r in seed_results)
    total_cumulative_lived_steps = sum(r["cumulative_lived_steps"] for r in seed_results)
    z_goal_activated = any(r["z_goal_eval_mean"] > 1e-3 for r in seed_results)

    n_uncensored_total = sum(r["survival"]["n_uncensored_deaths"] for r in seed_results)
    n_censored_total = sum(r["survival"]["n_censored_step_limit"] for r in seed_results)
    n_segments_total = sum(r["survival"]["n_segments"] for r in seed_results)
    pooled_uncensored_times: List[int] = []
    for r in seed_results:
        pooled_uncensored_times.extend(int(v) for v in r["survival"]["uncensored_survival_times"])
    pooled_pct_censored = float(n_censored_total / n_segments_total) if n_segments_total else 0.0

    core_ok = all(chan_nondegen.get(k, False) for k in CORE_CHANNELS)
    harm_trained = total_harm_steps > 0
    segments_varied = any(r["segments_varied"] for r in seed_results)
    freeze_not_locked = (total_freeze == 0) or (total_freeze < total_steps)
    sufficient_uncensored_deaths = bool(n_uncensored_total >= MIN_UNCENSORED_DEATHS_TOTAL)
    passed = bool(core_ok and harm_trained and sufficient_uncensored_deaths)
    outcome = "PASS" if passed else "FAIL"

    metrics: Dict[str, Any] = {
        "n_seeds": float(len(seeds)),
        "total_harm_pathway_train_steps": float(total_harm_steps),
        "total_block_steps": float(total_block),
        "total_limb_damage_events": float(total_limb_damage),
        "total_external_hazard_events": float(total_external_hazard),
        "total_world_rule_shift_events": float(total_world_rule_shift),
        "total_sleep_cycles_fired": float(total_sleep_cycles),
        "total_freeze_fires": float(total_freeze),
        "total_eval_steps": float(total_steps),
        "total_cumulative_lived_steps": float(total_cumulative_lived_steps),
        "z_goal_activated_at_eval": 1.0 if z_goal_activated else 0.0,
        "total_spawn_safe_retries": float(total_spawn_retries),
        "total_spawn_safe_exhausted_segments": float(total_spawn_exhausted),
        # --- primary DV: pooled (both seeds) uncensored survival-time distribution ---
        "n_segments_total": float(n_segments_total),
        "n_uncensored_deaths_total": float(n_uncensored_total),
        "n_censored_step_limit_total": float(n_censored_total),
        "pct_right_censored_pooled": pooled_pct_censored,
        "uncensored_survival_min": float(min(pooled_uncensored_times)) if pooled_uncensored_times else None,
        "uncensored_survival_max": float(max(pooled_uncensored_times)) if pooled_uncensored_times else None,
        "uncensored_survival_mean": (float(statistics.fmean(pooled_uncensored_times))
                                      if pooled_uncensored_times else None),
        "uncensored_survival_median": (float(statistics.median(pooled_uncensored_times))
                                        if pooled_uncensored_times else None),
    }
    for r in seed_results:
        s = r["seed"]
        metrics[f"seed{s}_stage0_z_goal_peak"] = float(r["diag"]["stage0_z_goal_peak"])
        metrics[f"seed{s}_hazard_survival_gate"] = 1.0 if r["diag"]["hazard_survival_gate"] else 0.0
        metrics[f"seed{s}_hazard_harm_eval_range"] = float(r["diag"]["hazard_harm_eval_range"])
        metrics[f"seed{s}_z_goal_eval_mean"] = float(r["z_goal_eval_mean"])
        metrics[f"seed{s}_n_uncensored_deaths"] = float(r["survival"]["n_uncensored_deaths"])
        metrics[f"seed{s}_n_censored_step_limit"] = float(r["survival"]["n_censored_step_limit"])
        metrics[f"seed{s}_pct_right_censored"] = float(r["survival"]["pct_right_censored"])
        metrics[f"seed{s}_cumulative_lived_steps"] = float(r["cumulative_lived_steps"])
        metrics[f"seed{s}_n_full_log_segments_kept"] = float(r["n_full_kept"])
        rstats = r.get("residue_stats_final") or {}
        metrics[f"seed{s}_residue_total_residue_final"] = float(rstats.get("total_residue", 0.0))
        metrics[f"seed{s}_residue_active_centers_final"] = float(rstats.get("active_centers", 0))
    for k in chan_keys:
        metrics[f"chan_max_std_{k}"] = float(chan_max_std[k])
        metrics[f"chan_mean_{k}"] = float(np.mean([r["chan_mean"].get(k, 0.0) for r in seed_results]))

    lifetime_affective_occupancy = {f"seed{r['seed']}": r["occupancy"] for r in seed_results}

    interpretation = {
        "label": "uncensored_survival_distribution_obtained" if sufficient_uncensored_deaths
                 else "uncensored_survival_still_censoring_dominated",
        "preconditions": [
            {"name": "harm_pathway_trained", "description": "harm-pathway co-training ran >=1 optimizer step",
             "measured": float(total_harm_steps), "threshold": 1.0, "direction": "lower",
             "met": bool(harm_trained)},
            {"name": "all_segments_completed",
             "description": "every seed's driver loop reached the configured EVAL_EPISODES "
                             "segment count (no early crash/truncation silently shrinking n)",
             "measured": float(n_segments_total),
             "threshold": float(EVAL_EPISODES * len(seeds)), "direction": "lower",
             "met": bool(n_segments_total >= EVAL_EPISODES * len(seeds))},
        ],
        "criteria_non_degenerate": {
            **{f"channel_{k}": chan_nondegen.get(k, False) for k in chan_keys},
            "harm_pathway_trained": harm_trained,
            "freeze_not_permanently_locked": freeze_not_locked,
            "segments_actually_varied_duration": segments_varied,
            "sufficient_uncensored_deaths": sufficient_uncensored_deaths,
        },
        "criteria": [
            {"name": "core_channels_non_degenerate", "load_bearing": True, "passed": core_ok},
            {"name": "harm_pathway_trained", "load_bearing": True, "passed": harm_trained},
            {"name": "sufficient_uncensored_deaths", "load_bearing": True,
             "passed": sufficient_uncensored_deaths},
            {"name": "freeze_not_locked", "load_bearing": False, "passed": freeze_not_locked},
            {"name": "segments_varied_duration", "load_bearing": False, "passed": segments_varied},
            *[{"name": f"channel_{k}", "load_bearing": False, "passed": bool(chan_nondegen.get(k, False))}
              for k in chan_keys],
        ],
        "note": (f"Large-n (60 segments/seed x {len(seeds)} seeds = {n_segments_total} total) "
                 f"characterization of the within-500-step-cap health_depleted death-time "
                 f"distribution on the 906b ecology tier. PASS = harm-pathway training ran AND "
                 f"the core affect channels vary AND >= {MIN_UNCENSORED_DEATHS_TOTAL} genuine "
                 f"(uncensored) death events were observed across all seeds combined "
                 f"(observed: {n_uncensored_total}). Even on a FAIL/still-censoring-dominated "
                 f"outcome, the pooled pct_right_censored ({pooled_pct_censored:.3f}) and "
                 f"n_uncensored_deaths_total are the load-bearing scientific readouts -- this "
                 f"is a characterization run, not a hypothesis test. See module docstring "
                 f"'SUBSTRATE READINESS FINDING' for why the per-segment 500-step ceiling "
                 f"itself is NOT removed by this run (a substrate change, chipped separately "
                 f"for /implement-substrate) and 'ECOLOGY CHOICE' for why this deliberately "
                 f"reuses 906b's environment tuning rather than 911's. claim_ids=[]; does not "
                 f"weight governance."),
    }

    summary_markdown = f"""# V3-EXQ-912 -- Uncensored Survival-to-Death Fishtank Successor

**Status:** {outcome} (diagnostic characterization run -- not scored against any claim)
**Purpose:** large-n (n={n_segments_total} segments across {len(seeds)} seeds) characterization
of the health_depleted death-time distribution on the 906b ecology tier, replacing the prior
906b/906c/911 lineage's single-run, mostly-censored n=8 point estimates (75%/87.5%/100%
right-censored respectively) with a proper survival-analysis treatment. See module docstring
for the substrate-readiness finding (the per-segment 500-step cap is a hardcoded substrate
constant, not a driver-configurable parameter) that shaped this run's actual design, and for
why 906b's ecology tuning was deliberately reused instead of 911's (which produced ZERO deaths
in its own 8-segment eval).

- harm-pathway train steps (total): {total_harm_steps}
- z_goal activated at eval: {z_goal_activated}
- eval steps (total): {total_steps}  across {EVAL_EPISODES} segments/seed x up to {EVAL_STEPS} steps
- **segments: {n_segments_total} total -- uncensored (genuine) deaths: {n_uncensored_total} -- censored (hit 500-step cap): {n_censored_total} -- pct_right_censored: {pooled_pct_censored:.3f}**
  (compare 906b 75%, 906c 87.5%, 911 100%)
- uncensored survival times (steps, pooled across seeds): min={metrics['uncensored_survival_min']} median={metrics['uncensored_survival_median']} mean={metrics['uncensored_survival_mean']} max={metrics['uncensored_survival_max']}
- cumulative lived steps (both seeds, sum of realized_steps across all segments): {total_cumulative_lived_steps}
- events: block={total_block} limb_damage={total_limb_damage} external_hazard={total_external_hazard} world_rule_shift={total_world_rule_shift}
- sleep cycles fired: {total_sleep_cycles}
- freeze fires (eval, motor-override relaxed): {total_freeze}
- safe-spawn retries (total): {total_spawn_retries}  (segments exhausted: {total_spawn_exhausted})

## Eval channel mean / max-std
{chr(10).join(f'- {k}: mean={metrics.get("chan_mean_"+k,0.0):.4f} max_std={chan_max_std[k]:.5f} ({"varies" if chan_nondegen[k] else "FLAT"})' for k in chan_keys)}

## Lifetime affective occupancy (per seed, non-gating, SENT-2 hygiene -- see module docstring)
{chr(10).join(f'- seed {r["seed"]}: n_measured={r["occupancy"].get("n_lived_steps_measured")} frac_dread_above_p75={r["occupancy"].get("frac_steps_dread_above_p75")} frac_z_harm_a_above_p75={r["occupancy"].get("frac_steps_z_harm_a_above_p75")} frac_harm_event={r["occupancy"].get("frac_harm_event")} frac_in_reef={r["occupancy"].get("frac_in_reef")}' for r in seed_results)}

The `_episode_log.json` companion is THINNED per the module docstring ("EPISODE-LOG SIZE
MANAGEMENT") -- full per-step records are kept only for the first {FULL_LOG_FIRST_K}
segments/seed and every genuinely-died segment; other censored segments are stored as a
per-segment summary. Every statistic in THIS manifest is computed from the full, unthinned
in-memory data, so thinning does not bias any reported number.

## For a future reader (or `/failure-autopsy`) on THIS run

If `n_uncensored_deaths_total` is near zero despite reusing 906b's (not 911's) ecology, the
906b ecology tier's death rate may itself have drifted, or this run's larger n surfaced a
genuine within-life adaptation effect (fewer deaths later in a life than earlier -- see the
organism-lifespan review Section 7's flagged, currently-unresolved within-life-development
question; this run's per-segment `cumulative_step_start`/`cumulative_step_end` markers make
that a checkable follow-on without re-running anything). If `pct_right_censored_pooled` is
still high, the next lever is the substrate follow-on named in this docstring
(parameterize CausalGridWorldV2's per-episode step cap), not another lettered increase to
EVAL_EPISODES, which cannot remove the 500-step ceiling.
"""

    first_env_config = seed_results[0].get("env_config", {}) if seed_results else {}
    episode_log = {
        "experiment_type": EXPERIMENT_TYPE,
        "phase": "uncensored_survival_characterization",
        "toroidal": bool(first_env_config.get("toroidal", False)),
        "env_config": first_env_config,
        "seeds": [{"seed": r["seed"], "episodes": r.get("episodes_stored", [])} for r in seed_results],
    }

    return {
        "status": outcome, "outcome": outcome, "metrics": metrics,
        "summary_markdown": summary_markdown, "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE, "evidence_direction": "non_contributory",
        "experiment_type": EXPERIMENT_TYPE, "interpretation": interpretation,
        "episode_log": episode_log, "agents": agents,
        "lifetime_affective_occupancy": lifetime_affective_occupancy,
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
