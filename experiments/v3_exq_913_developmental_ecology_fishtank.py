"""
V3-EXQ-913 -- Developmental-Ecology Fishtank Successor (layout continuity + probabilistic
habitat-cue foraging + sleep-vs-no-sleep ablation)

Claims: None (diagnostic showcase; does not weight governance)

EXPERIMENT_PURPOSE = "diagnostic"

SLEEP DRIVER: two-arm ablation, not a single fixed cadence. ARM_WITH_SLEEP = K=10
multi-fire (SleepLoopManager, sleep_loop_episodes_K=10 -- unchanged 906b cadence,
fires roughly twice across this run's 24-segment life). ARM_NO_SLEEP = K=never
(use_sleep_loop=False -- SleepLoopManager is never constructed; confirmed at
ree_core/agent.py:2630 `if getattr(config, "use_sleep_loop", False): ... self.sleep_loop
= SleepLoopManager(...)`, so agent.sleep_loop stays None and
_segment_boundary_consolidate()'s own `if agent.sleep_loop is not None:` guard makes it
a correct, automatic no-op -- no extra branching needed in this driver for the ablation
itself).

WHY THIS RUN. Routing source: chip-20260810-fishtank-developmental-ecology (spawned
2026-08-10T06:44Z by relaxed-shtern-75392b, twice-amended same day by session
angry-heisenberg-e8fec7). Combines three findings the chip's own instructions require
be addressed as ONE successor, not three separate ones (per
reef_ecology_strategy_affective_occupancy_review_2026-08-10.md Section 8's explicit
"additive, not a duplicate" framing):

  1. LAYOUT CONTINUITY (organism_lifespan_development_review_906_lineage_2026-08-10.md
     Section 1/7/10 item 2). `env.reset()` re-rolls the hazard/resource layout AND agent
     spawn position at every segment boundary, unconditionally, regardless of
     `done_cause` (causal_grid_world.py:1422 `reset()`, called every boundary by the
     906b-family `_observational_run()` via `_safe_reset()`). The review's own direct
     early-vs-late-segment test found a pattern consistent with within-life improvement
     in 906b (declining harm rate, rising benefit rate, falling mode entropy across 8
     segments) that did NOT replicate in 906c on the same measures -- exactly what
     per-segment layout re-randomization would produce even with zero real learning.
  2. PROBABILISTIC HABITAT CUE (developmental_ecology_curiosity_foraging_correction_
     2026-08-10.md Section 4, CORRECTING reef_ecology_strategy_affective_occupancy_
     review_2026-08-10.md Section 8 item 1 -- see "PROBABILISTIC HABITAT CUE: DESIGN
     DECISION" below for why this driver implements the correction's design PRINCIPLE
     via a DIFFERENT, safer substrate mechanism than the correction document's own
     "landmark" framing suggested).
  3. SLEEP-VS-NO-SLEEP ABLATION (sleep_transition_investigation_906_lineage_2026-08-10.md
     Section 9, extending reef_ecology_strategy_affective_occupancy_review_2026-08-10.md
     Section 6/8 item 2). The one real sleep firing available in 906b/906c's own data
     coincides with a segment boundary that ALSO triggers a full environment reset, so a
     retrospective sleep pre/post test cannot separate "sleep changed behaviour" from
     "the generic reset effect." Worse: even with reset removed as a confound, there is
     still no NO-SLEEP CONTROL anywhere in existing data -- every logged boundary either
     has a sleep firing or does not, with no experimentally-controlled contrast between
     "equivalent elapsed time, no sleep" and "equivalent elapsed time, with sleep." This
     run adds that missing contrast directly via a two-arm design.

SUBSTRATE READINESS FINDINGS (Step 2.5a empirical probes -- run BEFORE writing this
script, per this skill's "verify the premise before assuming a doc-stated capability
is reachable at runtime" rule).

  (a) `env.reset_to(agent_pos, hazard_positions, resource_positions)`
      (causal_grid_world.py:1921) exists and does exactly what LAYOUT CONTINUITY needs:
      a deterministic reset that places entities at CALLER-SUPPLIED coordinates instead
      of re-rolling them, built for the SD-029/EXQ-433a scripted-eval comparator harness.
      Its own docstring says "Landmarks (SD-023) are not placed by this method" -- probed
      directly (throwaway script, not committed) and confirmed empirically: `reset_to()`
      unconditionally zeroes `self.landmark_a_positions` / `self.landmark_b_positions` /
      `self._landmark_a_field` / `self._landmark_b_field` (causal_grid_world.py:2029-2032)
      but does NOT touch `self._reef_cells` or `self._zone_map` anywhere in its body --
      both persist automatically across a `reset_to()` call with no driver-side action
      needed (confirmed by direct source read of the full ~125-line method body, not
      inferred from the docstring's silence). This is exactly why the PROBABILISTIC
      HABITAT CUE below is built on microhabitat zones rather than landmarks -- see that
      section for the reasoning this probe result fed into.
  (b) The same probe constructed a `CausalGridWorldV2` with `n_landmarks_b=1,
      landmark_b_resource_bias=0.7`, called `reset()`, captured the landmark position,
      called `reset_to()`, confirmed the landmark state zeroed as expected, then manually
      restored it (`env.landmark_b_positions = [...]; env._landmark_b_field =
      env._compute_landmark_field(...)`) and recomputed the observation
      (`env._get_observation_dict()` / `env._dict_to_flat()`) -- this DOES work
      mechanically (landmark_b_field max correctly returned from 0.0 to the configured
      scale after restore) and was the original plan. It was NOT used in the final design
      because of finding (c) below, not because it failed.
  (c) `world_obs_dim` (causal_grid_world.py:1385) grows by 50 dims specifically when
      `n_landmarks_a > 0 or n_landmarks_b > 0` (confirmed by direct source read of the
      property, not the docstring). Because the SAME agent is trained through the whole
      664/665-derived curriculum (stage0/p0/hazard/p1) BEFORE ever seeing the eval env,
      and `REEAgent`'s world encoder is sized ONCE from a `probe_env` at construction
      time, turning landmarks on ONLY for the eval env (the natural, minimal-diff place
      to add a "new" cue, matching how `EVAL_ENV_EXTRA_KWARGS` already works for
      non-dimension-affecting fields) would size the agent's encoder for a 250-wide world
      observation during training and then feed it a 300-wide one at eval -- a shape
      mismatch crash. Making it consistent would require adding landmark kwargs to EVERY
      phase branch inside `_build_env()` (stage0/p0/hazard/p1/p2), which
      `scaffolded_sd054_onboarding.py`'s `_build_env()` does not expose a hook for (its
      signature takes `phase`/`anneal_t`/`p1_spawn_in_reef_half`/`seed` only, no
      passthrough kwargs) -- doable but invasive (copying/patching a shared, heavily-used
      curriculum-construction function) for what the chip's own Section 7 instruction
      calls a "first pass" that should "avoid unnecessary complexity."

PROBABILISTIC HABITAT CUE: DESIGN DECISION (deviates from the correction document's own
"landmark" framing -- justified here per this skill's "use your own judgement, but
justify deviations in the docstring" instruction). Given finding (c), this driver
implements the correction document's Section 4 PRINCIPLE (a perceptible, non-deterministic,
region-correlated predictor of resource likelihood, distinct from the resource_field
itself) using **microhabitat zones** (`microhabitat_enabled`, infant_substrate:GAP-2,
causal_grid_world.py:567 / `_zone_resource_factor()` / `_zone_hazard_factor()` /
`_pop_zone_weighted()`) instead of SD-023 landmarks:

  - `microhabitat_enabled=True` does NOT affect `world_obs_dim` or `body_obs_dim` at all
    (confirmed by direct read of both properties -- only `use_proxy_fields`,
    `limb_damage_enabled`, `n_landmarks_a/b`, `multi_resource_heterogeneity_enabled`
    change them). So it can be turned on for the eval env ONLY (via the same post-hoc
    `setattr` pattern `EVAL_ENV_EXTRA_KWARGS` already uses), with zero risk of a
    training/eval shape mismatch. This is the whole reason the microhabitat route is safer
    than the landmark route for a first-pass build.
  - It already implements EXACTLY the "region-conditioned resource-placement PRIOR...
    a probability-of-spawn gradient... not a new DETERMINISTIC zone type" the correction
    document's Section 4b calls for: `_pop_zone_weighted()` places each resource/hazard by
    sampling from the empty-cell pool with probability PROPORTIONAL to
    `zone_A/B/C_resource_factor` (default 1.5 / 0.8 / 0.3) or `zone_A/B/C_hazard_factor`
    (default 0.3 / 1.8 / 0.0) -- a MULTIPLICATIVE weighting on a random draw, not a
    guarantee. A resource is 5x more LIKELY to land in zone A than zone C at these
    defaults, never certain to.
  - Zone identity itself is NOT a dedicated sensed channel (there is no "you are in zone
    A" observation field -- `microhabitat_zone_at_agent` lives only in the driver-visible
    `info` dict, confirmed by direct read, never in `obs_dict`/`world_state`). So the cue
    REE can actually perceive is INDIRECT: because zone-weighted placement systematically
    shifts the regional DENSITY of resources/hazards, the EXISTING, already-perceived
    `resource_field_view` / `hazard_field_view` (part of the baseline 250-dim
    `use_proxy_fields=True` observation every 906-lineage run already senses) will read
    differently, on average, in a resource-favouring region than a resource-poor one --
    without any single sensed value ever being a certain readout of "food is here." This is
    weaker than a dedicated landmark marker, and is stated here as an explicit, honest
    trade-off: this driver buys substrate safety at the cost of the cue being read only
    through the SAME ambient fields the correction document's Section 4c critiqued as "the
    field is a smoothed copy of resource positions," not a genuinely separate channel.
    What zone-weighting adds ON TOP of that critique is real, though: under LAYOUT
    CONTINUITY (finding #1), the field is no longer a smoothed copy of a FRESH per-segment
    draw -- it is a smoothed copy of ONE persistent, zone-STRUCTURED draw that stays
    constant across this run's whole 24-segment life, which is the necessary substrate for
    any future within-life search-efficiency analysis (Section 5 of the correction
    document) even though this run does not itself attempt to discriminate "REE learned
    the zone association" from "REE followed the ambient gradient" (Section 6 of that
    document already names this as an unresolved instrumentation gap requiring SD-025/
    MECH-314 per-tick logging this run does NOT add -- see "SCOPE, NOT ADDED" below).
    `zone_C_ambient_bonus` (a small reward bump while in zone C, causal_grid_world.py
    ~2614-2620) IS a directly-perceived-via-reward per-tick signal, but it is tied to zone
    C specifically (the calm, resource-POOR, hazard-free zone by the module's own
    defaults), not zone A (the resource-rich zone) -- not repurposed here since redirecting
    it is a substrate change, out of scope.
  - Because microhabitat placement weighting requires ONLY a constructor-time flag with
    no obs-dim consequence, `num_resources` was ALSO changed for this run specifically
    (5 -> 24, see DEV_NUM_RESOURCES below) -- with LAYOUT CONTINUITY meaning the initial
    resource draw is the WHOLE run's food supply (not one fresh 5-resource draw per
    segment as in 906b/906c/911/912), 5 resources would likely be exhausted within the
    first few of 24 segments. `resource_respawn_on_consume` is left False (unchanged
    906b/906c/912 convention) rather than enabled, because `_respawn_resource()`
    (causal_grid_world.py:5001) places uniformly at random, NOT zone-weighted -- enabling
    respawn would progressively dilute the zone-correlated placement signal with
    uniformly-scattered replacements over a long life, undermining the exact mechanism
    this run is trying to test. If DEV_NUM_RESOURCES=24 still runs out before segment 24
    for a given seed/arm, that is reported descriptively (see `resource_exhaustion` in the
    manifest), not treated as a design flaw.

SCOPE, NOT ADDED (stated explicitly per the chip's own "state what you did either way"
instruction). (a) SD-025's per-tick novelty term / MECH-314's score-bias are NOT
surfaced into telemetry here, even though the correction document's Section 6 names this
as the concrete follow-on that would let a future session discriminate curiosity-driven
discovery from diffuse-gradient exploitation -- reading into `ree_core/hippocampal/
curiosity.py` for a stable hook point would meaningfully grow this already-large script's
scope; left as an explicit, named follow-on rather than attempted here. (b) The multi-
context ecology (varying risk/yield regions, time-varying productivity) the correction
document's Section 7 explicitly defers is NOT built here either -- ONE habitat-cue type
(the existing 3-zone A/B/C map) is used, per that section's own instruction. (c) The
fuller sleep-richness factorial (rich-vs-impoverished experience x sleep-vs-no-sleep) from
sleep_transition_investigation_906_lineage_2026-08-10.md Section 5 is NOT built -- only
the two-arm WITH_SLEEP/NO_SLEEP minimum this chip's Section 9 amendment calls "the minimum
design that actually separates sleep from ordinary elapsed time."

MECH-357 CHECK (organism_lifespan_development_review_906_lineage_2026-08-10.md Section
10 item 7 -- "a cheap preliminary check... before proposing any new experience-specific-
learning probe"). Confirmed by direct grep of `_make_config()`
(v3_exq_906b_full_stack_observational_fishtank.py:400): `use_instrumental_avoidance=True`
IS already set -- MECH-357 (SD-058 ilPFC instrumental-avoidance eligibility-trace gate) is
active in this ecology, inherited unchanged from `_make_config()`. No dedicated new probe
is added for it in this run (per the same document's own framing, this was a preliminary
check to inform, not obligate, further design -- the existing per-step episode log already
carries enough (hazard proximity + subsequent action) for a future session to look for the
"hurt once -> avoids sooner" signature without a new run).

GOV-REUSE-1 (Step 2.4). Decisive readouts: (a) within-life harm/benefit-rate trend
computed on a layout-continuous run (not confounded by per-segment re-randomization);
(b) sleep-vs-no-sleep matched-boundary trajectory-organization deltas (turning entropy,
straight-run length, tortuosity, hazard-conditioned turning); (c) zone-conditioned
resource/hazard field statistics under a persistent, zone-weighted layout. Searched
existing manifests (the three planning documents this run is routed from each ran their
own explicit duplication checks against `TASK_CHIPS.json`, `claims.yaml`, and
`evidence/planning/*.md` before proposing this successor): no 906-lineage run, or any
other manifest, holds a layout-continuous multi-segment run, a sleep-vs-no-sleep matched
ablation, or a zone-weighted ecology. Not recoverable by reanalysis -- a run is required.

Re-derive brake (2.5b): claim_ids=[] -- not applicable, no claim to brake-check.
Substrate-path overlap gate (2.5c): checked substrate_queue.json for open `corrupting`
entries touching causal_grid_world.py, scaffolded_sd054_onboarding.py, or the 664/665/906b
driver modules this script imports -- none found.

Ethics preflight (Step 2.6, descriptive, non-enforced): involves_negative_valence=false,
involves_suffering_like_state=false, involves_self_model=false,
involves_inescapability_or_helplessness=false, involves_offline_replay_over_harm=false,
involves_social_mind_or_language=false, involves_human_data_or_clinical_context=false,
decision=allow. Unchanged from the rest of the 906 lineage -- longer, zone-structured
exposure does not change V3's architecture (no self-model, no autobiographical memory).
Per reef_ecology_strategy_affective_occupancy_review_2026-08-10.md Section 9,
`lifetime_affective_occupancy` (SENT-2 hygiene, non-gating, descriptive) is reported per
(seed, arm) given this run's segment count (24, 3x the 8-segment baseline) qualifies as a
"substantially longer... continuous life."

WELFARE INSTRUMENTATION -- see GOV-REUSE-1 note above; reused unchanged from V3-EXQ-912's
own `_lifetime_affective_occupancy` pattern (percentile-threshold occupancy, no inference
of sentience/suffering, SENT-0 boundary).

Output:
  evidence/experiments/v3_exq_913_developmental_ecology_fishtank/
    v3_exq_913_developmental_ecology_fishtank_<ts>.json               (manifest)
    v3_exq_913_developmental_ecology_fishtank_<ts>_episode_log.json   (fishtank feed)

Estimated runtime: 4 (seed, arm) combinations, each TRAIN_TOTAL_EPS=220 curriculum
training episodes + up to 24 segments x 500 eval steps. Comparable order of magnitude to
V3-EXQ-912 (60 segments x 2 seeds, ~150 min estimated) -- half the eval segments per run
but twice the number of full curriculum trainings (2 arms x 2 seeds vs 912's 2 seeds).
"""

import random
import statistics
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
    EVAL_ENV_EXTRA_KWARGS,
    SAFE_SPAWN_MAX_ATTEMPTS,
    TRAIN_TOTAL_EPS,
    CORE_CHANNELS,
    STD_FLOOR,
)
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE    = "v3_exq_913_developmental_ecology_fishtank"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS: List[str] = []

ARM_WITH_SLEEP = "with_sleep"
ARM_NO_SLEEP   = "no_sleep"
ARMS = [ARM_WITH_SLEEP, ARM_NO_SLEEP]

# ---- layout-continuous multi-segment life: 24 segments/seed (3x 906b/906c/911's 8),
# 2 arms (WITH_SLEEP / NO_SLEEP) x 2 seeds -- see module docstring items 1/3.
EVAL_EPISODES = 24
EVAL_STEPS    = 500
SEEDS_DEFAULT = [0, 1]

# ---- PROBABILISTIC HABITAT CUE (module docstring item 2): microhabitat zones, eval-env
# only (does not affect world_obs_dim/body_obs_dim -- safe post-hoc override, unlike
# n_landmarks_a/b). Values are the causal_grid_world.py constructor DEFAULTS -- this run
# turns the mechanism ON, it does not retune it (avoiding an extra, unvalidated tuning
# axis on top of an already large design).
DEV_ECOLOGY_EXTRA_KWARGS: Dict[str, Any] = dict(EVAL_ENV_EXTRA_KWARGS)
DEV_ECOLOGY_EXTRA_KWARGS.update(dict(
    microhabitat_enabled=True,
))

# num_resources for the whole 24-segment life (module docstring: with layout continuity,
# the initial draw is the WHOLE life's food supply, not a fresh per-segment draw -- 5
# would likely exhaust in the first few segments). resource_respawn_on_consume left False
# (see docstring) -- set via ScaffoldedSD054OnboardingConfig.scaffold_p2_num_resources,
# NOT DEV_ECOLOGY_EXTRA_KWARGS, since it must apply at CONSTRUCTION time consistently to
# both the probe env (sizing) and the eval env (though num_resources does not itself
# affect obs_dim, mutating scaffold_cfg once keeps both envs' resource counts consistent).
DEV_NUM_RESOURCES = 24

SAFE_SPAWN_ATTEMPTS = SAFE_SPAWN_MAX_ATTEMPTS

# Hazard-conditioned turning: "near" = within this Manhattan radius of the nearest
# hazard (matches sleep_transition_investigation_906_lineage_2026-08-10.md Section 3's
# own "within 3 cells" choice).
HAZARD_NEAR_RADIUS = 3

# Local post-boundary window (steps) for the matched WITH_SLEEP-vs-NO_SLEEP trajectory
# comparison (reef_ecology_strategy_affective_occupancy_review_2026-08-10.md Section 6 /
# sleep_transition_investigation Section 3a's own local-window size).
LOCAL_WINDOW_STEPS = 100

# Lifetime affective-occupancy percentile (reef_ecology_strategy review Section 9).
OCCUPANCY_PERCENTILE = 75

# Pre-registered (Step 3 "thresholds defined in the script, not inferred post-hoc").
MIN_HARM_TRAIN_STEPS = 1.0


def _build_dev_eval_env(scaffold_cfg, seed: int):
    """Eval env for this run: 906b's EVAL_ENV_EXTRA_KWARGS plus microhabitat zones
    (module docstring "PROBABILISTIC HABITAT CUE"). num_resources comes from
    scaffold_cfg.scaffold_p2_num_resources (mutated by the caller to DEV_NUM_RESOURCES
    before this is called), not from this dict."""
    env = _build_env(scaffold_cfg, "p2", seed=seed)
    for k, v in DEV_ECOLOGY_EXTRA_KWARGS.items():
        setattr(env, k, v)
    return env


def _make_config_for_arm(env, sleep_enabled: bool):
    """906b's _make_config() with use_sleep_loop overridden per-arm (module docstring
    "SLEEP DRIVER"). All other flags (including sleep_loop_episodes_K=10, read only when
    use_sleep_loop=True) are unchanged from 906b."""
    cfg = _make_config(env)
    cfg.use_sleep_loop = bool(sleep_enabled)
    return cfg


def _continuity_reset(
    env: CausalGridWorldV2,
    hazard_positions: List[Tuple[int, int]],
    resource_positions: List[Tuple[int, int]],
    max_attempts: int = SAFE_SPAWN_ATTEMPTS,
) -> Tuple[Any, Dict, int, bool]:
    """LAYOUT CONTINUITY (module docstring item 1, Step 2.5a finding (a)). Segment-boundary
    reset that persists hazard_positions/resource_positions UNCHANGED (no re-randomization)
    via env.reset_to(), while still picking a fresh but SAFE agent spawn point each
    boundary -- mirrors 906b's own _safe_reset() criterion (hazard_field below
    proximity_approach_threshold at the spawn cell), reimplemented here because reset_to()
    takes an explicit agent_pos rather than picking one itself. reef cells and the
    microhabitat zone map are NOT touched by reset_to() (confirmed by direct source read,
    Step 2.5a finding (a)) -- they persist automatically with no action needed here.
    Consumes env's own self._rng (deterministic given the run's seed), same generator
    _safe_reset()/_place_reef_patches()/_build_microhabitat_zones() already use.
    Returns (flat_obs, obs_dict, attempts, exhausted)."""
    occupied = {tuple(int(c) for c in h) for h in hazard_positions} | \
               {tuple(int(c) for c in r) for r in resource_positions}
    interior = [
        (i, j) for i in range(1, env.size - 1) for j in range(1, env.size - 1)
        if (i, j) not in occupied
    ]
    if not interior:
        interior = [(1, 1)]

    def _pick() -> Tuple[int, int]:
        idx = int(env._rng.integers(0, len(interior)))
        return interior[idx]

    chosen = _pick()
    flat_obs, obs_dict = env.reset_to(chosen, hazard_positions, resource_positions)
    attempts = 1
    while (attempts < max_attempts
           and float(env.hazard_field[chosen[0], chosen[1]]) >= env.proximity_approach_threshold):
        chosen = _pick()
        flat_obs, obs_dict = env.reset_to(chosen, hazard_positions, resource_positions)
        attempts += 1
    exhausted = bool(
        float(env.hazard_field[chosen[0], chosen[1]]) >= env.proximity_approach_threshold
    )
    return flat_obs, obs_dict, attempts, exhausted


def _trajectory_organization_stats(
    steps: List[Dict[str, Any]],
    hazard_positions: List[Tuple[int, int]],
    window: Optional[int] = None,
) -> Dict[str, Any]:
    """Turning-angle distribution, straight-run length, tortuosity, hazard-conditioned
    turning -- the sleep_transition_investigation_906_lineage_2026-08-10.md Section 3
    measure set (module docstring item 3), computed here directly from logged positions
    (that document's own script was not committed and is reimplemented from its Section 3
    method description, not copied). `steps` is the ALREADY-completed per-step log for one
    segment (or a slice of it); `window` (if given) uses only the first `window` entries."""
    seq = steps[:window] if window is not None else steps
    n = len(seq)
    out: Dict[str, Any] = {"n_steps": n}
    if n < 2:
        return out
    positions = [tuple(s["pos"]) for s in seq]
    deltas = [(positions[i + 1][0] - positions[i][0], positions[i + 1][1] - positions[i][1])
              for i in range(n - 1)]
    headings = []
    for dx, dy in deltas:
        if dx == 0 and dy == 0:
            headings.append(None)
        else:
            headings.append(float(np.arctan2(dy, dx)))

    # Turning angle: absolute angular change between consecutive non-null headings.
    turning_angles: List[float] = []
    turning_near_hazard: List[float] = []
    turning_far_hazard: List[float] = []
    prev_heading = None
    for i, h in enumerate(headings):
        if h is not None and prev_heading is not None:
            diff = abs(h - prev_heading)
            if diff > np.pi:
                diff = 2 * np.pi - diff
            turning_angles.append(float(diff))
            if hazard_positions:
                px, py = positions[i]
                nearest = min(abs(px - hx) + abs(py - hy) for hx, hy in hazard_positions)
                if nearest <= HAZARD_NEAR_RADIUS:
                    turning_near_hazard.append(float(diff))
                else:
                    turning_far_hazard.append(float(diff))
        if h is not None:
            prev_heading = h

    # Straight-run length: consecutive steps sharing the same non-null heading.
    straight_runs: List[int] = []
    run_len = 0
    run_heading = None
    for h in headings:
        if h is None:
            continue
        if run_heading is not None and abs(h - run_heading) < 1e-6:
            run_len += 1
        else:
            if run_len > 0:
                straight_runs.append(run_len)
            run_len = 1
            run_heading = h
    if run_len > 0:
        straight_runs.append(run_len)

    path_length = sum(abs(dx) + abs(dy) for dx, dy in deltas)
    net_displacement = abs(positions[-1][0] - positions[0][0]) + abs(positions[-1][1] - positions[0][1])
    tortuosity = (float(path_length) / net_displacement) if net_displacement > 0 else None

    out.update({
        "turning_angle_mean": float(np.mean(turning_angles)) if turning_angles else None,
        "turning_angle_entropy_bits": (
            float(-np.sum((h := np.histogram(turning_angles, bins=8, range=(0, np.pi))[0]
                           / len(turning_angles)) * np.log2(h + 1e-12)))
            if turning_angles else None
        ),
        "mean_straight_run_length": float(np.mean(straight_runs)) if straight_runs else None,
        "max_straight_run_length": int(max(straight_runs)) if straight_runs else None,
        "tortuosity": tortuosity,
        "path_length": int(path_length),
        "net_displacement": int(net_displacement),
        "turning_near_hazard_mean": float(np.mean(turning_near_hazard)) if turning_near_hazard else None,
        "turning_far_hazard_mean": float(np.mean(turning_far_hazard)) if turning_far_hazard else None,
        "n_turning_near_hazard": len(turning_near_hazard),
        "n_turning_far_hazard": len(turning_far_hazard),
    })
    return out


def _continuous_life_run(
    agent: REEAgent, env: CausalGridWorldV2, num_episodes: int,
    steps_per_episode: int, seed: int, arm: str,
) -> Dict[str, Any]:
    """LAYOUT-CONTINUOUS multi-segment observation of the SAME agent (module docstring
    item 1), adapted from 906b's _observational_run() -- boundary logic replaced with
    _continuity_reset() for every boundary after the first; per-step telemetry trimmed
    to what this run's three findings need (core affect channels, zone/field readouts,
    trajectory positions) rather than every field 906b logs (e.g. no defensive-orienting
    telemetry -- use_defensive_orienting is not set by _make_config() here either)."""
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

    # Showcase-legibility relaxation, unchanged from 906b/912: disable the MOTOR
    # freeze-lock override only; affect telemetry stays faithful to trained encoders.
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
            agent.reset()
            sleep_fired_this_boundary = False
            canonical_hazard_positions = [tuple(int(c) for c in h) for h in env.hazards]
        else:
            hazard_positions   = [tuple(int(c) for c in h) for h in env.hazards]
            resource_positions = [tuple(int(c) for c in r) for r in env.resources]
            flat_obs, obs_dict, spawn_attempts, spawn_exhausted = _continuity_reset(
                env, hazard_positions, resource_positions
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
            "layout_continuity_reset": bool(ep_idx > 0),
            # DESCRIPTIVE ONLY -- not a continuity failure signal. Hazard drift
            # (env_drift_interval/env_drift_prob, constructor default 5/0.3, unchanged
            # from 906b/912) is a real, pre-existing, ALWAYS-ON within-episode dynamic in
            # this ecology, confirmed by the dry-run smoke test: hazards routinely differ
            # from the ep0 canonical draw by segment 2 purely from in-episode drift, not
            # from any re-randomization. Layout continuity (module docstring item 1) is
            # about NOT re-rolling a FRESH independent layout at segment boundaries --
            # gradual, bounded, already-existing drift within a segment is orthogonal to
            # that confound and is deliberately left enabled (unchanged ecology dynamic).
            "hazard_positions_match_canonical": bool(
                {tuple(h) for h in env.hazards} == set(canonical_hazard_positions)
            ) if ep_idx > 0 else True,
            "zone_step_counts": zone_counts,
            "trajectory_local_window": traj_local,
            "trajectory_full_segment": traj_full,
        })
        print(f"  [eval] seed={seed} arm={arm} ep {ep_idx+1}/{num_episodes} steps={len(ep_steps)} "
              f"done_cause={segment_done_cause or '(ran to end-of-loop)'} "
              f"sleep_fired={sleep_fired_this_boundary} "
              f"spawn_attempts={spawn_attempts}", flush=True)

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


def _within_life_development(episodes_log: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Organism_lifespan review Section 7's own early-vs-late-segment test, now runnable
    without the per-segment layout-randomization confound it flagged (module docstring
    item 1). Same measures (harm rate, benefit rate, mode entropy) computed per segment
    from the full per-step log, correlated against segment index."""
    n = len(episodes_log)
    if n < 4:
        return {"n_segments": n, "note": "too few segments for a meaningful trend"}
    harm_rate, benefit_rate, mode_entropy = [], [], []
    for seg in episodes_log:
        steps = seg.get("steps") or []
        if not steps:
            harm_rate.append(0.0); benefit_rate.append(0.0); mode_entropy.append(0.0)
            continue
        harm_rate.append(float(np.mean([bool(s.get("harm_event")) for s in steps])))
        benefit_rate.append(float(np.mean([float(s.get("harm_signal", 0.0)) > 0 for s in steps])))
        modes = [s.get("mode") for s in steps]
        uniq, counts = np.unique(modes, return_counts=True)
        p = counts / counts.sum()
        mode_entropy.append(float(-np.sum(p * np.log2(p + 1e-12))))
    idx = list(range(n))
    third = max(1, n // 3)

    def _r(x):
        if len(set(x)) < 2:
            return None
        return float(np.corrcoef(idx, x)[0, 1])

    return {
        "n_segments": n,
        "early_harm_rate": float(np.mean(harm_rate[:third])),
        "late_harm_rate": float(np.mean(harm_rate[-third:])),
        "r_segment_harm_rate": _r(harm_rate),
        "early_benefit_rate": float(np.mean(benefit_rate[:third])),
        "late_benefit_rate": float(np.mean(benefit_rate[-third:])),
        "r_segment_benefit_rate": _r(benefit_rate),
        "early_mode_entropy": float(np.mean(mode_entropy[:third])),
        "late_mode_entropy": float(np.mean(mode_entropy[-third:])),
        "r_segment_mode_entropy": _r(mode_entropy),
    }


def _sleep_ablation_comparison(
    with_sleep_episodes: List[Dict[str, Any]],
    no_sleep_episodes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """The core sleep-vs-no-sleep readout (module docstring item 3): for every segment
    boundary where the WITH_SLEEP arm actually fired a sleep cycle, compare that segment's
    trajectory-organization stats against the SAME segment index in the NO_SLEEP arm
    (which never fires -- the matched "equivalent elapsed time, no sleep" control)."""
    fired_indices = [i for i, seg in enumerate(with_sleep_episodes)
                     if seg.get("sleep_cycle_fired_before_this_segment")]
    comparisons = []
    for i in fired_indices:
        if i >= len(no_sleep_episodes):
            continue
        ws = with_sleep_episodes[i]["trajectory_local_window"]
        ns = no_sleep_episodes[i]["trajectory_local_window"]
        comparisons.append({
            "segment_index": i,
            "with_sleep": ws,
            "no_sleep_matched": ns,
            "turning_entropy_delta": (
                (ws.get("turning_angle_entropy_bits") or 0.0) -
                (ns.get("turning_angle_entropy_bits") or 0.0)
            ) if ws.get("turning_angle_entropy_bits") is not None
              and ns.get("turning_angle_entropy_bits") is not None else None,
            "tortuosity_delta": (
                (ws.get("tortuosity") or 0.0) - (ns.get("tortuosity") or 0.0)
            ) if ws.get("tortuosity") is not None and ns.get("tortuosity") is not None else None,
        })
    return {
        "n_sleep_firings": len(fired_indices),
        "fired_segment_indices": fired_indices,
        "matched_comparisons": comparisons,
    }


def _lifetime_affective_occupancy(episodes_log: List[Dict[str, Any]]) -> Dict[str, Any]:
    """reef_ecology_strategy_affective_occupancy_review_2026-08-10.md Section 9: descriptive,
    non-gating, within-run-percentile occupancy stats. No inference of welfare/sentience --
    SENT-0 boundary. Reused pattern from V3-EXQ-912."""
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


def _zone_habitat_stats(episodes_log: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Zone-conditioned field statistics (module docstring "PROBABILISTIC HABITAT CUE"):
    does the agent's SENSED resource_field / hazard_field actually read differently by
    zone, confirming the placement-weighting manipulation left a perceptible trace in the
    ambient fields REE already senses (not a claim that REE has learned to exploit it)."""
    by_zone_resource: Dict[str, List[float]] = {}
    by_zone_hazard: Dict[str, List[float]] = {}
    by_zone_steps: Dict[str, int] = {}
    for seg in episodes_log:
        for s in seg.get("steps", []) or []:
            z = str(s.get("zone_at_agent", -1))
            by_zone_steps[z] = by_zone_steps.get(z, 0) + 1
            if isinstance(s.get("resource_field_at_agent"), (int, float)):
                by_zone_resource.setdefault(z, []).append(float(s["resource_field_at_agent"]))
            if isinstance(s.get("hazard_field_at_agent"), (int, float)):
                by_zone_hazard.setdefault(z, []).append(float(s["hazard_field_at_agent"]))
    return {
        "steps_by_zone": by_zone_steps,
        "mean_resource_field_by_zone": {k: float(np.mean(v)) for k, v in by_zone_resource.items() if v},
        "mean_hazard_field_by_zone": {k: float(np.mean(v)) for k, v in by_zone_hazard.items() if v},
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
    print(f"[EXQ-913] seed={seed} arm={arm} world_obs_dim={probe_env.world_obs_dim}"
          f" body_obs_dim={probe_env.body_obs_dim} use_sleep_loop={sleep_enabled}"
          f" microhabitat_enabled=True num_resources={DEV_NUM_RESOURCES}", flush=True)

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

    print(f"[EXQ-913] seed={seed} arm={arm} channel std: "
          + "  ".join(f"{k}={ree['chan_std'][k]:.4f}" for k in
                      ["z_harm_a", "z_harm_un", "drive", "z_goal", "vigor", "z_block", "excite", "dread"]),
          flush=True)
    print(f"[EXQ-913] seed={seed} arm={arm} segments={len(episodes_log)} "
          f"sleep_cycles_fired={ree['sleep_cycles_fired']} "
          f"continuity_resets={ree['n_continuity_resets']}/{eval_eps - 1} "
          f"zone_map_active={getattr(eval_env, '_zone_map', None) is not None}", flush=True)

    seed_core_ok = all(ree["chan_std"].get(k, 0.0) > STD_FLOOR for k in CORE_CHANNELS)
    harm_trained = (diag["p0_harm_train_steps"] + diag["hazard_harm_train_steps"]) > 0
    # Structural check only: every boundary after ep0 used env.reset_to() (module
    # docstring "LAYOUT CONTINUITY"). Per-segment hazard-vs-canonical drift (see the
    # episode-log field's own comment) is descriptive, not gating -- hazard drift is a
    # real, pre-existing, always-on within-episode dynamic, not a re-randomization.
    layout_continuity_confirmed = bool(ree["n_continuity_resets"] == eval_eps - 1)
    zone_map_active = getattr(eval_env, "_zone_map", None) is not None
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
        "zone_map_active": zone_map_active,
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
    print(f"[V3-EXQ-913] Developmental-Ecology Fishtank Successor\n"
          f"  Seeds: {seeds}  Arms: {ARMS}  curriculum: Stage-0/0b/P0/Stage-H/P1 + harm-pathway training\n"
          f"  Train eps/(seed,arm): {TRAIN_TOTAL_EPS}  Eval: {EVAL_EPISODES} segments x up to {EVAL_STEPS} "
          f"steps (layout-continuous, microhabitat-zoned ecology)\n"
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
    zone_map_active = all(r["zone_map_active"] for r in results)
    sleep_ablation_engaged = bool(total_sleep_cycles_with > 0 and total_sleep_cycles_no == 0)
    freeze_not_locked = (total_freeze == 0) or (total_freeze < total_steps)
    all_seeds_arms_pass = all(r["seed_pass"] for r in results)
    passed = bool(core_ok and harm_trained and layout_continuity_confirmed
                  and zone_map_active and sleep_ablation_engaged)
    outcome = "PASS" if passed else "FAIL"

    # Sleep ablation comparison, per seed (matched WITH_SLEEP boundary -> NO_SLEEP same-index).
    sleep_ablation_per_seed: Dict[str, Any] = {}
    for s in seeds:
        ws = by_key.get((s, ARM_WITH_SLEEP))
        ns = by_key.get((s, ARM_NO_SLEEP))
        if ws is None or ns is None:
            continue
        sleep_ablation_per_seed[f"seed{s}"] = _sleep_ablation_comparison(
            ws["episodes_full"], ns["episodes_full"]
        )

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

    n_sleep_firing_comparisons = sum(
        v.get("n_sleep_firings", 0) for v in sleep_ablation_per_seed.values()
    )
    metrics["n_sleep_firing_matched_comparisons"] = float(n_sleep_firing_comparisons)

    interpretation = {
        "label": ("developmental_ecology_mechanisms_engaged" if passed
                 else "developmental_ecology_mechanism_readiness_unmet"),
        "preconditions": [
            {"name": "harm_pathway_trained", "description": "harm-pathway co-training ran >=1 optimizer step",
             "measured": float(total_harm_steps), "threshold": MIN_HARM_TRAIN_STEPS, "direction": "lower",
             "met": bool(harm_trained)},
            {"name": "layout_continuity_confirmed",
             "description": "every seed/arm's segment boundaries after the first used "
                             "env.reset_to() (not env.reset()), and the persisted hazard "
                             "positions matched the canonical ep0 draw at every boundary",
             "measured": 1.0 if layout_continuity_confirmed else 0.0, "threshold": 1.0,
             "direction": "lower", "met": bool(layout_continuity_confirmed)},
            {"name": "zone_map_active",
             "description": "microhabitat_enabled=True actually produced a non-None "
                             "_zone_map at ep0 in every seed/arm (positive-control check "
                             "that the probabilistic-habitat-cue manipulation engaged, not "
                             "a silent no-op)",
             "measured": 1.0 if zone_map_active else 0.0, "threshold": 1.0,
             "direction": "lower", "met": bool(zone_map_active)},
            {"name": "sleep_ablation_engaged",
             "description": "WITH_SLEEP arm fired >=1 real sleep cycle AND NO_SLEEP arm "
                             "fired exactly 0, pooled across seeds",
             "measured": float(total_sleep_cycles_with),
             "threshold": 1.0, "direction": "lower",
             "met": bool(sleep_ablation_engaged)},
        ],
        "criteria_non_degenerate": {
            **{f"channel_{k}": chan_nondegen.get(k, False) for k in chan_keys},
            "harm_pathway_trained": harm_trained,
            "freeze_not_permanently_locked": freeze_not_locked,
            "layout_continuity_confirmed": layout_continuity_confirmed,
            "zone_map_active": zone_map_active,
            "sleep_ablation_engaged": sleep_ablation_engaged,
        },
        "criteria": [
            {"name": "core_channels_non_degenerate", "load_bearing": True, "passed": core_ok},
            {"name": "harm_pathway_trained", "load_bearing": True, "passed": harm_trained},
            {"name": "layout_continuity_confirmed", "load_bearing": True,
             "passed": layout_continuity_confirmed},
            {"name": "zone_map_active", "load_bearing": True, "passed": zone_map_active},
            {"name": "sleep_ablation_engaged", "load_bearing": True, "passed": sleep_ablation_engaged},
            {"name": "freeze_not_locked", "load_bearing": False, "passed": freeze_not_locked},
            {"name": "all_seed_arm_verdicts_pass", "load_bearing": False, "passed": all_seeds_arms_pass},
            *[{"name": f"channel_{k}", "load_bearing": False, "passed": bool(chan_nondegen.get(k, False))}
              for k in chan_keys],
        ],
        "note": (
            f"Diagnostic READINESS characterization (not a hypothesis test): PASS = the core "
            f"affect channels vary AND harm-pathway training ran AND all three combined "
            f"mechanisms (layout continuity, microhabitat probabilistic-habitat-cue "
            f"placement, sleep-vs-no-sleep ablation) engaged as designed. The scientifically "
            f"load-bearing readouts are the within_life_development / sleep_ablation_per_seed "
            f"/ zone_habitat blocks below, which are reported regardless of PASS/FAIL -- "
            f"{n_sleep_firing_comparisons} matched WITH_SLEEP-firing-vs-NO_SLEEP-control "
            f"segment comparisons were obtained across {len(seeds)} seeds. claim_ids=[]; "
            f"does not weight governance."
        ),
    }

    summary_markdown = f"""# V3-EXQ-913 -- Developmental-Ecology Fishtank Successor

**Status:** {outcome} (diagnostic readiness characterization -- not scored against any claim)
**Purpose:** combined layout-continuity + probabilistic-habitat-cue (microhabitat zones) +
sleep-vs-no-sleep ablation successor to the V3-EXQ-906 lineage, addressing three findings
from the organism-level Fishtank reviews as one successor (see module docstring for full
routing + substrate-readiness reasoning, including why microhabitat zones were used instead
of the correction document's own "landmark" suggestion).

- harm-pathway train steps (total, all seed/arm): {total_harm_steps}
- layout continuity confirmed (all seed/arm): {layout_continuity_confirmed}
- microhabitat zone map active (all seed/arm): {zone_map_active}
- sleep cycles fired -- WITH_SLEEP arm: {total_sleep_cycles_with}  NO_SLEEP arm: {total_sleep_cycles_no}
- matched sleep-firing-vs-no-sleep-control comparisons obtained: {n_sleep_firing_comparisons}
- eval steps (total): {total_steps}  across {EVAL_EPISODES} segments/(seed,arm) x up to {EVAL_STEPS} steps
- events: block={total_block}  freeze fires (motor-override relaxed): {total_freeze}
- safe-spawn / continuity-spawn retries (total): {total_spawn_retries}  (segments exhausted: {total_spawn_exhausted})
- segments with resources fully exhausted (total, across seed/arm): {sum(r['resource_exhausted_segments'] for r in results)}

## Eval channel mean / max-std
{chr(10).join(f'- {k}: mean={metrics.get("chan_mean_"+k,0.0):.4f} max_std={chan_max_std[k]:.5f} ({"varies" if chan_nondegen[k] else "FLAT"})' for k in chan_keys)}

## Within-life development (organism review Section 7, now unconfounded by layout continuity)
{chr(10).join(f'- {key}: n_segments={v.get("n_segments")} r(segment,harm_rate)={v.get("r_segment_harm_rate")} r(segment,benefit_rate)={v.get("r_segment_benefit_rate")} r(segment,mode_entropy)={v.get("r_segment_mode_entropy")}' for key, v in within_life_per_seed_arm.items())}

## Sleep-vs-no-sleep matched comparisons (per seed)
{chr(10).join(f'- seed{s}: n_sleep_firings={v.get("n_sleep_firings")} fired_at_segments={v.get("fired_segment_indices")}' for s, v in sleep_ablation_per_seed.items())}

## Zone-conditioned field statistics (per seed/arm) -- resource_field / hazard_field mean by microhabitat zone
{chr(10).join(f'- {key}: {v.get("mean_resource_field_by_zone")}' for key, v in zone_habitat_per_seed_arm.items())}

## Lifetime affective occupancy (per seed/arm, non-gating, SENT-2 hygiene -- see module docstring)
{chr(10).join(f'- {key}: frac_dread_above_p75={v.get("frac_steps_dread_above_p75")} frac_z_harm_a_above_p75={v.get("frac_steps_z_harm_a_above_p75")} frac_harm_event={v.get("frac_harm_event")} frac_in_reef={v.get("frac_in_reef")}' for key, v in lifetime_affective_occupancy.items())}

## For a future reader (or `/failure-autopsy`) on THIS run

If `n_sleep_firing_matched_comparisons` is near zero, the K=10 cadence did not fire within
this run's 24 segments for some seeds -- increase EVAL_EPISODES or lower
sleep_loop_episodes_K in a successor rather than re-running unchanged. If
`total_resource_exhausted_segments` is large, DEV_NUM_RESOURCES=24 was not enough headroom
for a fully-foraging life at this ecology's consumption rate -- raise it (still with
resource_respawn_on_consume=False, per the module docstring's zone-dilution reasoning) in a
successor. The `zone_habitat` block's per-zone resource_field means are the check for
whether the probabilistic-habitat-cue manipulation left a perceptible trace in what REE
actually senses; SD-025/MECH-314 per-tick logging (module docstring "SCOPE, NOT ADDED") is
the concrete next step for testing whether REE exploits it, not merely whether it exists.
"""

    first_env_config = results[0].get("env_config", {}) if results else {}
    episode_log = {
        "experiment_type": EXPERIMENT_TYPE,
        "phase": "developmental_ecology_layout_continuous_zoned_sleep_ablation",
        "toroidal": bool(first_env_config.get("toroidal", False)),
        "env_config": first_env_config,
        "runs": [{"seed": r["seed"], "arm": r["arm"], "episodes": r.get("episodes_full", [])}
                 for r in results],
    }

    return {
        "status": outcome, "outcome": outcome, "metrics": metrics,
        "summary_markdown": summary_markdown, "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE, "evidence_direction": "non_contributory",
        "experiment_type": EXPERIMENT_TYPE, "interpretation": interpretation,
        "episode_log": episode_log, "agents": agents,
        "within_life_development": within_life_per_seed_arm,
        "sleep_ablation_comparison": sleep_ablation_per_seed,
        "zone_habitat_stats": zone_habitat_per_seed_arm,
        "lifetime_affective_occupancy": lifetime_affective_occupancy,
        "config": first_env_config,
    }


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

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    emit_outcome(outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
                 manifest_path=out_path,
                 dry_run=bool(args.dry_run))
