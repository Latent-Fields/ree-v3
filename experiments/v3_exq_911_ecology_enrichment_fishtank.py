"""
V3-EXQ-911 -- Full-Stack Observational Fishtank: Ecology Enrichment for Discrete-Resource
Acquisition (Section 9 item 3 of the 906b observational review)

Claims: None (diagnostic showcase; does not weight governance)

EXPERIMENT_PURPOSE = "diagnostic"

SLEEP DRIVER: manual-multi (unchanged from 906b -- agent.sleep_loop.notify_episode_end()
called directly every segment boundary via _segment_boundary_consolidate(), imported from
906b. Still fires on the K=10 cadence; sleep_loop_episodes_K=10.)

WHY THIS RUN. Routing source: Section 9 item 3 of
REE_assembly/evidence/planning/observational_review_V3-EXQ-906b_2026-08-09.md ("Ecology
enrichment for food-seeking (design note, lower priority)"). Section 2b of that review found
906b's positive-signal picture is NOT discrete-resource acquisition: mean Manhattan distance
to the nearest mapped resource was 6.23 across all 3909 eval steps and 6.02 on steps that
received positive `harm_signal` (i.e. the agent was no closer to a resource when it received
benefit than at baseline). Only 11 genuine `resource` (consummatory) transitions occurred in
the whole run, against 1384 `benefit_approach` transitions -- "food-seeking" in 906b is
overwhelmingly the agent exploiting an ambient proximity/reef gradient, not navigating to and
consuming discrete resources. This is EXPLICITLY LOWER PRIORITY than tracks A/C/906c (review
Section 9) -- queued but not treated as urgent.

ROOT CAUSE, CONFIRMED AGAINST SOURCE BEFORE DESIGNING A FIX (same "re-verify substrate API
correctness even if copied" discipline 906b/906c already document). `_compute_proximity_fields()`
(causal_grid_world.py ~4367-4426) computes `resource_field` as a SUM of
`1/(1+dist*resource_field_decay)` over every one of `num_resources=5` sources -- structurally
identical to the hazard_field summed-field design 906b already root-caused and fixed for
hazards. `resource_field_decay` (env constructor default 0.5) is a SEPARATE config field from
`hazard_field_decay` (confirmed by grep -- two distinct constructor kwargs, two distinct
`self.*` attributes) and was left UNTOUCHED by 906b/906c, which only touched the hazard side.
The `benefit_approach` transition (causal_grid_world.py ~2470-2473) fires whenever
`resource_field[agent_pos] >= proximity_approach_threshold` -- the SAME shared threshold 906b
tuned to 0.4 for hazard "smell before harm" legibility. A standalone Monte-Carlo probe of the
exact field formula (500 random 5-resource layouts on the same 12x12 interior grid 906b used
for its own hazard probe) confirms the mechanism directly: at the UNTOUCHED
`resource_field_decay=0.5` / `threshold=0.4`, P(field active | any distance to nearest
resource, including >=3 cells) = 1.000 -- the resource-benefit field is active on effectively
the WHOLE grid regardless of proximity, exactly 906a's pre-fix hazard signature and exactly
what Section 2b's ~6-cell "no closer on benefit steps" finding predicts. This driver's fix
mirrors 906b's own hazard methodology exactly (empirical grid-search over decay, not hand
arithmetic -- see 906b's module docstring point 1 for why the hand-arithmetic version there was
wrong): decay=3.0 was chosen from a sweep over {1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.5} at the
UNCHANGED threshold=0.4 (resource_field_decay is independent of hazard_field_decay, so this
does NOT touch 906b's already-validated hazard-survivability tuning), measuring
P(active | d=1)=0.918, P(active | d=2)=0.261, P(active | d>=3)=0.006, mean active grid fraction
0.264 -- closely matching 906b's own hazard curve (0.969 / 0.265 / 0.006 for 4 hazards at
decay=2.5), i.e. a genuine "smell before benefit" radius rather than a grid-wide field.

WHAT CHANGED VS 906b/906c, AND WHAT DID NOT.

  1. ECOLOGY MOSTLY UNCHANGED. Imports `_make_config`, `_env_config_snapshot`,
     `_observational_run`, `_segment_boundary_consolidate` is reached via `_observational_run`
     internally (unchanged), `TRAIN_TOTAL_EPS`, `EVAL_EPISODES`, `EVAL_STEPS`, `CORE_CHANNELS`,
     `STD_FLOOR`, `EVAL_ENV_EXTRA_KWARGS` directly from
     v3_exq_906b_full_stack_observational_fishtank -- same curriculum, same all-ON module
     stack, same hazard-side proximity-radius/safe-spawn fixes 906b validated, same
     `ecology_survivable` gate (unchanged threshold -- this run must not regress
     survivability while fixing the resource-field confound). This is a thin driver, not a
     re-derivation of the ~700-line eval loop -- see 906b's own module docstring for the full
     hazard-side rationale, unchanged here.

  2. THE ECOLOGY-ENRICHMENT CHANGE ITSELF (Section 9 item 3's two named levers, BOTH applied --
     `_build_eval_env` below overrides three env attributes on top of 906b's unchanged
     `EVAL_ENV_EXTRA_KWARGS` dict, applied post-construction exactly the way 906b/906c already
     apply their own overrides -- confirmed safe post-hoc by reading
     `_compute_proximity_fields()` (called fresh after every placement/drift/consumption event,
     never baked into `__init__`) and by a standalone smoke probe at authoring time
     (env.reset() + a forced `_consume_resource_at` call) that resource count is genuinely
     held stable across a consumption event under these overrides:
       - `resource_field_decay=3.0` (up from the untouched default 0.5) -- the root-cause fix,
         see above. `proximity_approach_threshold` is left at 906b's 0.4 UNCHANGED (it is
         shared with the hazard branch; changing it here would re-open 906b's hazard tuning,
         which this run has no mandate to touch).
       - `proximity_benefit_scale=0.01` (down from the untouched default 0.03) -- the review's
         second named lever ("reduce proximity_benefit_scale relative to the discrete-resource
         reward"), a further, independent gentling of the ambient benefit magnitude, mirroring
         906b's own halving-style gentling of `proximity_harm_scale` (0.05->0.02) on the hazard
         side. `resource_benefit` (discrete consummatory reward, unchanged default 0.3) is
         untouched by this driver -- already ~10x the untouched ambient scale, so the ratio is
         widened further, not created from nothing.
       - `resource_respawn_on_consume=True` (up from the untouched default False) -- the
         review's first named lever, so consumption events can recur through a segment instead
         of the resource pool depleting after 906b's fixed-pool 11-events-in-3909-steps. Uses
         the existing SD-012 `_respawn_resource()` mechanism (spawns one new resource at a
         random empty interior cell immediately after consumption) -- no new substrate code,
         confirmed already GA per `ree-v3/CLAUDE.md`.

  3. NEW INSTRUMENTATION: the confound this run exists to fix is measured directly, not just
     asserted fixed (`_resource_distance_metrics` below). Pools (agent_pos, nearest-resource
     Manhattan distance) pairs across every seed/episode/step from the already-collected
     per-step `pos` / `resources` fields (`_observational_run` inherited unchanged from 906b),
     split by `transition_type`:
       - `mean_dist_to_resource_all_steps` -- the same "all steps" baseline Section 2b computed
         (906b's own value: 6.23).
       - `mean_dist_to_resource_benefit_approach_steps` -- mean distance specifically on steps
         where the diffuse proximity-gradient benefit fired (`transition_type=="benefit_approach"`).
         This is the DIRECT falsifier for the fix: Section 2b's finding was that this number
         (6.02, using "any positive harm_signal" as its benefit proxy) was barely different from
         the all-steps baseline -- i.e. the field was not spatially organised. A successful fix
         should push this number substantially below the all-steps baseline (the field is now
         only active close to a resource, so a step where it fires should, on average, be a
         step where the agent genuinely is closer than baseline).
       - `n_benefit_approach_steps` -- sample-floor precondition (see below): with a genuinely
         narrowed field, `benefit_approach` should fire LESS often in absolute terms than
         906b's grid-wide 1384/3909, so the comparison needs enough samples to be meaningful,
         not just directionally suggestive.
       - `n_resource_consume_events` -- reported (NOT load-bearing; see criteria below) count of
         genuine `transition_type=="resource"` consummatory events, for direct comparison
         against 906b's fixed-pool baseline of 11/3909. NOT gated on a specific floor: how much
         a still-imperfect trained policy actually benefits from a larger consumption budget is
         an open behavioural-competency question this driver does not force an answer to (that
         is squarely Section 4's decoupling finding, addressed by other tracks, not this one) --
         this run's job is only to make repeated consumption POSSIBLE and to make the diffuse
         field spatially honest, not to prove the agent exploits either.
       - `mean_dist_to_resource_at_consume_events` -- reported for completeness only. By
         construction this is ~0 at every consumption event (the agent is standing on the
         resource cell when it consumes it), so it is NOT informative about navigation quality
         and is explicitly NOT used in any criterion -- recorded so a future reader does not
         mistake its trivial near-zero value for evidence of anything.

  4. NO `supersedes`. This run does not correct or invalidate 906b's or 906c's science (their
     ecology, hazard-side config, and PASS/FAIL gates are unchanged here on the hazard side) --
     it is a deliberately DIFFERENT ecology on the resource side, purpose-built to answer a
     narrower question (Section 9 item 3) that 906b/906c did not ask.

WHAT THIS RUN IS NOT: a claim test, a statistically powered multi-seed study, a substrate
build, or a demonstration that the trained agent's food-seeking COMPETENCE has improved (that
is Section 4's decoupling / MECH-439 conversion-ceiling territory, addressed by other tracks).
This run only tests whether the ENVIRONMENT-LEVEL confound Section 2b identified is reduced,
so a FUTURE food-seeking metric on this ecology would measure navigation-to-resource rather
than gradient-sitting. Single seed by default (--seeds), unchanged in kind from 906b/906c.

Output:
  evidence/experiments/v3_exq_911_ecology_enrichment_fishtank/
    v3_exq_911_ecology_enrichment_fishtank_<ts>.json               (manifest)
    v3_exq_911_ecology_enrichment_fishtank_<ts>_episode_log.json   (fishtank feed)

Estimated runtime: unchanged in kind from 906b/906c (same curriculum, same eval segment/step
budget) -- see the queue entry note. LOWER PRIORITY than tracks A/C/906c per the review.
"""

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
from experiments.v3_exq_906b_full_stack_observational_fishtank import (
    _make_config,
    _env_config_snapshot,
    _observational_run,
    EVAL_ENV_EXTRA_KWARGS as _906B_EVAL_ENV_EXTRA_KWARGS,
    TRAIN_TOTAL_EPS,
    EVAL_EPISODES,
    EVAL_STEPS,
    CORE_CHANNELS,
    STD_FLOOR,
)
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE    = "v3_exq_911_ecology_enrichment_fishtank"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS: List[str] = []
# No `supersedes` -- see module docstring point 4.

# ---- ecology-enrichment overrides (module docstring point 2), layered on top of 906b's
# unchanged hazard-side EVAL_ENV_EXTRA_KWARGS. Only the resource side changes.
ECOLOGY_ENRICHMENT_OVERRIDES: Dict[str, Any] = dict(
    resource_field_decay=3.0,          # untouched default 0.5 -- see module docstring root cause
    proximity_benefit_scale=0.01,      # untouched default 0.03 -- sharper ambient-vs-discrete contrast
    resource_respawn_on_consume=True,  # untouched default False -- SD-012, already-GA mechanism
)
EVAL_ENV_EXTRA_KWARGS: Dict[str, Any] = dict(_906B_EVAL_ENV_EXTRA_KWARGS)
EVAL_ENV_EXTRA_KWARGS.update(ECOLOGY_ENRICHMENT_OVERRIDES)

# Section 2b's own baseline numbers (906b's actual run), cited for the summary markdown only
# -- not re-derived here, not used in any pre-registered threshold below.
BASELINE_906B_MEAN_DIST_ALL_STEPS = 6.23
BASELINE_906B_MEAN_DIST_BENEFIT_STEPS = 6.02
BASELINE_906B_N_RESOURCE_CONSUME_EVENTS = 11

# Pre-registered sample floor for the confound-reduction criterion (module docstring point 3):
# with a genuinely narrowed field, benefit_approach should fire in absolute terms LESS often
# than 906b's grid-wide 1384/3909 -- this floor guards against the comparison being computed on
# too few samples to be meaningful.
MIN_BENEFIT_APPROACH_SAMPLES = 20.0
# Pre-registered threshold for the confound-reduction criterion: mean distance on
# benefit_approach steps must sit substantially below the 906b/906c all-steps baseline
# (~6.2), not merely "less than baseline" (which a tiny, noisy sample could satisfy by chance).
CONFOUND_REDUCED_MAX_MEAN_DIST = 3.0


def _build_eval_env(scaffold_cfg, seed: int):
    """Same construction path as 906b's own `_build_eval_env`, but applying the ecology-
    enrichment overrides (module docstring point 2) in place of 906b's hazard-only
    EVAL_ENV_EXTRA_KWARGS. `resource_field_decay` / `proximity_benefit_scale` /
    `resource_respawn_on_consume` are all read fresh at consumption/drift time
    (`_compute_proximity_fields()` / the resource-consumption branch), never baked into
    `__init__` -- confirmed by reading the source and by a standalone smoke probe at
    authoring time before relying on this, per this skill's "re-verify substrate API
    correctness even if copied" rule."""
    env = _build_env(scaffold_cfg, "p2", seed=seed)
    for k, v in EVAL_ENV_EXTRA_KWARGS.items():
        setattr(env, k, v)
    return env


# ---------------------------------------------------------------------------
# Confound-reduction instrumentation (module docstring point 3)
# ---------------------------------------------------------------------------

def _nearest_resource_dist(pos: List[int], resources: List[List[int]]) -> Optional[float]:
    if not pos or not resources:
        return None
    return float(min(abs(pos[0] - r[0]) + abs(pos[1] - r[1]) for r in resources))


def _resource_distance_metrics(all_episode_steps: List[List[Dict]]) -> Dict[str, float]:
    """Pools (agent_pos, nearest-resource distance) across every seed/episode/step, split by
    transition_type -- the direct falsifier for whether the ecology-enrichment fix actually
    made the diffuse proximity-benefit field spatially honest (module docstring point 3)."""
    all_dists: List[float] = []
    benefit_dists: List[float] = []
    consume_dists: List[float] = []
    for steps in all_episode_steps:
        for s in steps:
            d = _nearest_resource_dist(s.get("pos"), s.get("resources") or [])
            if d is None:
                continue
            all_dists.append(d)
            ttype = s.get("transition_type")
            if ttype == "benefit_approach":
                benefit_dists.append(d)
            elif ttype == "resource":
                consume_dists.append(d)

    def _mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    return {
        "mean_dist_to_resource_all_steps": _mean(all_dists),
        "n_all_steps_with_resource_data": float(len(all_dists)),
        "mean_dist_to_resource_benefit_approach_steps": _mean(benefit_dists),
        "n_benefit_approach_steps": float(len(benefit_dists)),
        # Reported only -- trivially ~0 by construction, NOT used in any criterion (module
        # docstring point 3).
        "mean_dist_to_resource_at_consume_events": _mean(consume_dists),
        "n_resource_consume_events": float(len(consume_dists)),
    }


def run_seed(seed: int, dry_run: bool = False) -> Dict[str, Any]:
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    device = torch.device("cpu")
    total_eps = (2 + 2 + 5 + 5 + 5) if dry_run else TRAIN_TOTAL_EPS

    print(f"\nSeed {seed} Condition ecology_enrichment_fishtank", flush=True)
    scaffold_cfg = _make_scaffold_cfg(dry_run)
    probe_env = _build_env(scaffold_cfg, "p2")
    probe_env.reset()

    agent = REEAgent(_make_config(probe_env)).to(device)
    scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)
    print(f"[EXQ-911] seed={seed} world_obs_dim={probe_env.world_obs_dim}"
          f" body_obs_dim={probe_env.body_obs_dim} full-stack curriculum ON", flush=True)

    diag = _run_curriculum(agent, scheduler, device, seed, total_eps)

    eval_eps   = 2 if dry_run else EVAL_EPISODES
    eval_steps = 30 if dry_run else EVAL_STEPS
    eval_env = _build_eval_env(scaffold_cfg, seed=seed)
    env_config_snapshot = _env_config_snapshot(eval_env)
    ree = _observational_run(agent, eval_env, eval_eps, eval_steps, seed)

    print(f"[EXQ-911] seed={seed} channel std: "
          + "  ".join(f"{k}={ree['chan_std'][k]:.4f}" for k in
                      ["z_harm_a", "z_harm_un", "drive", "z_goal", "vigor", "z_block", "excite", "dread"]),
          flush=True)
    print(f"[EXQ-911] seed={seed} events: block={ree['block_steps']} "
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
    print(f"[V3-EXQ-911] Ecology Enrichment for Discrete-Resource Acquisition\n"
          f"  Seeds: {seeds}  curriculum: Stage-0/0b/P0/Stage-H/P1 + harm-pathway training "
          f"(unchanged from 906b)\n"
          f"  Train eps/seed: {TRAIN_TOTAL_EPS}  Eval: {EVAL_EPISODES} segments x up to "
          f"{EVAL_STEPS} steps (hazard side unchanged from 906b; resource side re-tuned)\n"
          f"  New: resource_field_decay=3.0 (was 0.5), proximity_benefit_scale=0.01 (was 0.03), "
          f"resource_respawn_on_consume=True (was False)\n"
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
    # Unchanged from 906b/906c: same pre-registered floor, testing the same
    # ecology-survivability question -- this run's resource-side re-tuning must not regress it.
    mean_realized_segment_steps = (total_steps / max(1, sum(len(r["episodes"]) for r in seed_results)))
    ecology_survivable = bool(mean_realized_segment_steps >= 4.0 * (447.0 / 30.0))

    # ---- module docstring point 3: the confound-reduction instrumentation, pooled over ALL
    # seeds'/episodes' steps. ----
    all_episode_steps: List[List[Dict]] = [
        ep["steps"] for r in seed_results for ep in r.get("episodes", []) if ep.get("steps")
    ]
    dist_metrics = _resource_distance_metrics(all_episode_steps)
    sample_floor_met = bool(dist_metrics["n_benefit_approach_steps"] >= MIN_BENEFIT_APPROACH_SAMPLES)
    confound_reduced = bool(
        sample_floor_met
        and dist_metrics["mean_dist_to_resource_benefit_approach_steps"] <= CONFOUND_REDUCED_MAX_MEAN_DIST
    )

    passed = bool(core_ok and harm_trained and ecology_survivable and confound_reduced)
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
                               "total_spawn_safe_exhausted_segments": float(total_spawn_exhausted),
                               "baseline_906b_mean_dist_all_steps": float(BASELINE_906B_MEAN_DIST_ALL_STEPS),
                               "baseline_906b_mean_dist_benefit_steps": float(BASELINE_906B_MEAN_DIST_BENEFIT_STEPS),
                               "baseline_906b_n_resource_consume_events": float(BASELINE_906B_N_RESOURCE_CONSUME_EVENTS)}
    metrics.update(dist_metrics)
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
        "label": "ecology_enrichment_confound_reduced" if passed
                 else "ecology_enrichment_confound_not_reduced",
        "preconditions": [
            {"name": "harm_pathway_trained", "description": "harm-pathway co-training ran >=1 optimizer step",
             "measured": float(total_harm_steps), "threshold": 1.0, "direction": "lower",
             "met": bool(harm_trained)},
            {"name": "ecology_survivable",
             "description": "same eval-ecology precondition as 906b/906c (hazard side "
                             "unchanged): segments should run well past 906's ~14.9-step/"
                             "segment early-death signature (>=4x floor)",
             "measured": float(mean_realized_segment_steps),
             "threshold": float(4.0 * (447.0 / 30.0)), "direction": "lower",
             "met": bool(ecology_survivable)},
            {"name": "benefit_approach_sample_floor",
             "description": "enough benefit_approach steps exist to make the confound-"
                             "reduction comparison meaningful rather than noise -- a "
                             "genuinely narrowed field should fire less often in absolute "
                             "terms than 906b's grid-wide 1384/3909, so this floor guards "
                             "against an uninterpretably small sample",
             "measured": float(dist_metrics["n_benefit_approach_steps"]),
             "threshold": float(MIN_BENEFIT_APPROACH_SAMPLES), "direction": "lower",
             "met": bool(sample_floor_met)},
        ],
        "criteria_non_degenerate": {
            **{f"channel_{k}": chan_nondegen.get(k, False) for k in chan_keys},
            "harm_pathway_trained": harm_trained,
            "freeze_not_permanently_locked": freeze_not_locked,
            "ecology_survivable": ecology_survivable,
            "confound_reduced": sample_floor_met,
        },
        "criteria": [
            {"name": "core_channels_non_degenerate", "load_bearing": True, "passed": core_ok},
            {"name": "harm_pathway_trained", "load_bearing": True, "passed": harm_trained},
            {"name": "ecology_survivable", "load_bearing": True, "passed": ecology_survivable},
            {"name": "benefit_approach_confound_reduced", "load_bearing": True,
             "passed": confound_reduced},
            {"name": "freeze_not_locked", "load_bearing": False, "passed": freeze_not_locked},
            *[{"name": f"channel_{k}", "load_bearing": False, "passed": bool(chan_nondegen.get(k, False))}
              for k in chan_keys],
        ],
        "note": ("Ecology-enrichment iteration of the 906 lineage (Section 9 item 3 of the "
                 "906b observational review, explicitly LOWER PRIORITY than tracks A/C/906c). "
                 "Hazard side unchanged from 906b -- resource_field_decay/"
                 "proximity_benefit_scale/resource_respawn_on_consume re-tuned so the diffuse "
                 "proximity-benefit field becomes spatially honest (a genuine 'smell before "
                 "benefit' radius, mirroring 906b's own hazard-field fix methodology) instead "
                 "of grid-wide. PASS = harm-pathway trained AND core affect channels vary AND "
                 "the ecology remains survivable AND the benefit_approach-step mean distance "
                 "to the nearest resource has genuinely dropped (<=" + str(CONFOUND_REDUCED_MAX_MEAN_DIST) +
                 " cells, vs 906b's ~6.0-6.2 baseline) on a non-degenerate sample. This run "
                 "does NOT test or claim improved food-seeking COMPETENCE -- only that the "
                 "environment-level confound Section 2b identified is reduced, so a future "
                 "food-seeking metric on this ecology would measure navigation-to-resource "
                 "rather than gradient-sitting. claim_ids=[]; does not weight governance."),
    }

    summary_markdown = f"""# V3-EXQ-911 -- Ecology Enrichment for Discrete-Resource Acquisition

**Status:** {outcome} (diagnostic telemetry showcase -- not scored against any claim)
**Purpose:** Section 9 item 3 of the 906b observational review
(observational_review_V3-EXQ-906b_2026-08-09.md), explicitly LOWER PRIORITY than tracks
A/C/906c. Section 2b of that review found 906b's "food-seeking" was overwhelmingly ambient
proximity/reef-gradient exploitation, not discrete-resource navigation (mean distance to
nearest resource ~6.2 cells regardless of whether the agent was receiving benefit). This run
re-tunes the resource side of the ecology (hazard side unchanged from 906b) so a future
food-seeking metric on this ecology would measure real navigation instead of gradient-sitting.

## What changed vs 906b (resource side only)
- `resource_field_decay`: 0.5 -> 3.0 (root-cause fix -- see module docstring for the
  Monte-Carlo grid-search that chose this value, mirroring 906b's own hazard-field methodology)
- `proximity_benefit_scale`: 0.03 -> 0.01 (sharper ambient-vs-discrete contrast)
- `resource_respawn_on_consume`: False -> True (SD-012, already-GA mechanism -- lets
  consumption recur through a segment instead of the resource pool depleting)

## Confound-reduction result (the direct falsifier for this run's purpose)
- mean distance to nearest resource, ALL steps: {dist_metrics['mean_dist_to_resource_all_steps']:.2f}
  (906b baseline: {BASELINE_906B_MEAN_DIST_ALL_STEPS})
- mean distance to nearest resource, `benefit_approach` steps: {dist_metrics['mean_dist_to_resource_benefit_approach_steps']:.2f}
  (906b baseline: {BASELINE_906B_MEAN_DIST_BENEFIT_STEPS}; n={int(dist_metrics['n_benefit_approach_steps'])}
  vs sample floor {int(MIN_BENEFIT_APPROACH_SAMPLES)}; PASS threshold <= {CONFOUND_REDUCED_MAX_MEAN_DIST})
- genuine `resource` consumption events: {int(dist_metrics['n_resource_consume_events'])}
  (906b baseline, fixed non-respawning pool: {BASELINE_906B_N_RESOURCE_CONSUME_EVENTS}; reported
  only, NOT load-bearing -- see module docstring point 3)
- mean distance at consume events (trivially ~0 by construction, reported only):
  {dist_metrics['mean_dist_to_resource_at_consume_events']:.2f}

## Ecology-survivability sanity check (unchanged gate from 906b/906c)
- harm-pathway train steps (total): {total_harm_steps}
- eval steps (total): {total_steps}  across {EVAL_EPISODES} segments x up to {EVAL_STEPS} steps/seed
  (mean realized segment length: {mean_realized_segment_steps:.1f} steps -- unchanged gate from 906b)
- segment endings: health_depleted={total_health_deaths} step_cap={total_step_cap_ends}
- sleep cycles fired: {total_sleep_cycles}
- freeze fires (eval, motor-override relaxed): {total_freeze}

The `_episode_log.json` companion feeds fishtank_viz.html via /api/fishtank/logs, same schema
as 906b/906c.
"""

    first_env_config = seed_results[0].get("env_config", {}) if seed_results else {}
    episode_log = {
        "experiment_type": EXPERIMENT_TYPE,
        "phase": "ecology_enrichment_fishtank",
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
        "sleep_driver_pattern": ("manual-multi (agent.sleep_loop.notify_episode_end() called "
                                  "directly every segment boundary via "
                                  "_segment_boundary_consolidate(), inherited unchanged from "
                                  "906b; K=10 cadence)"),
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
        # Same companion-path fix 906b/906c apply (906b module docstring point 6): declared
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
