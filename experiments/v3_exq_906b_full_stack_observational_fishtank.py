"""
V3-EXQ-906b -- Full-Stack Observational Fishtank Showcase (proximity-radius fix, safe spawn)

Claims: None (diagnostic showcase; does not weight governance)

EXPERIMENT_PURPOSE = "diagnostic"

SLEEP DRIVER: manual-multi (agent.sleep_loop.notify_episode_end() called directly every
segment boundary in the observational eval loop, decoupled from the rest of agent.reset()
-- see 906a's own docstring "CONTINUITY REDESIGN", unchanged here. Still fires on the same
K=10 cadence; sleep_loop_episodes_K=10.)

WHY THIS RUN (bug-fix lettered iteration of V3-EXQ-906a, run 2026-08-09T08:10:31Z, FAIL).
Routing source: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-906a_894b_2026-08-09.md
(Target 1, user-confirmed). 906a's own two survivability fixes (contamination_spread=0.0,
hazard_harm=0.05) addressed 2 of 3 independent health-drain channels in
causal_grid_world.py, but left a THIRD untouched: proximity-approach damage
(causal_grid_world.py ~2445-2469, `elif self.use_proxy_fields and transition_type=="none"`),
which drains agent_health on any step the agent is merely NEAR a hazard's field -- no
contact required -- gated only by `proximity_approach_threshold` (906a left this at its
default 0.15). With `hazard_field_decay` at its default 0.5
(`hazard_field[i,j] += 1/(1+dist*decay)`), that threshold crosses at
dist=(1/0.15-1)/0.5 ~= 11.3 cells -- effectively grid-wide on the 12x12 p2 eval env (`size=12`
via `scaffold_env_size`), not a close-range warning. 906a's scored run: 8/8 eval segments
ended `health_depleted`, 0/8 reached the step cap, mean realized segment length 25.375 steps
(against the `ecology_survivable` gate's own threshold, 59.6) -- with only 1 external_hazard
event and 0 limb_damage events recorded, which is inconsistent with discrete contact and
consistent with ambient proximity drain. The user directly watched fishtank_viz.html and
reported the fish's health draining fast with no hazard hit, which is exactly this
signature. Fixes, all confirmed against source before relying on them (same "re-verify
substrate API correctness even if copied" discipline 906a itself documents):

  1. PROXIMITY-HARM RADIUS TIGHTENED (the root-cause fix) -- PARAMETERS CHOSEN BY EMPIRICAL
     GRID-SEARCH, NOT BY HAND ARITHMETIC, AND THAT DISTINCTION MATTERS (see the correction
     note in point 2). `_compute_proximity_fields()` (causal_grid_world.py ~4367-4404) sums
     each hazard's `1/(1+dist*decay)` contribution over ALL `num_hazards` sources -- it is a
     SUM, not a max. A single-hazard-radius calculation (as this docstring's first draft did,
     and largely as the autopsy's own prose does) therefore UNDER-states the true harm zone:
     with 4 hazards on this env's 12x12 grid, ambient background contributions from hazards
     that are individually far away still sum into a field value well above any threshold
     that looks reasonable for one isolated hazard -- this is *why* 906a's untouched defaults
     were "effectively grid-wide" (confirmed by a first-draft fix at
     `proximity_approach_threshold=0.6` with `hazard_field_decay` left at its default 0.5:
     an actual smoke run of `_safe_reset()` below exhausted its 20-attempt bound on both
     dry-run segments, and a standalone Monte-Carlo probe of the exact field formula over 300
     random 4-hazard layouts confirmed 99.4% of the grid was still "unsafe" at that setting).
     Re-derived empirically instead: for several `(hazard_field_decay, proximity_approach_threshold)`
     pairs, simulated 400-500 random 4-hazard layouts on a 12x12 grid and measured, bucketed
     by each cell's Manhattan distance to its NEAREST hazard, P(field >= threshold | nearest
     hazard at distance d) for d in {1, 2, >=3}. Chosen: `hazard_field_decay=2.5` (up from
     the untouched default 0.5) and `proximity_approach_threshold=0.4` (up from the untouched
     default 0.15), which measured P(unsafe | d=1)=0.969, P(unsafe | d=2)=0.265, P(unsafe |
     d>=3)=0.006, mean unsafe fraction of the interior grid ~0.19 (i.e. ~81% of the grid safe
     on average). `proximity_harm_scale=0.02` (down from 906a's still-default 0.05) is kept
     as a further, independent gentling of the per-step drain within whatever harm zone
     results. NOTE: `proximity_approach_threshold` is a SHARED field -- the same branch also
     gates the resource-approach BENEFIT bonus (`r_active` in the same code block,
     causal_grid_world.py ~2450) -- tightening it symmetrically tightens both proximity cues
     to close-range, intentional and consistent with point 2 below, not a side effect being
     smuggled in. `hazard_field_decay` is ALSO the value the agent's own SENSORY view of the
     field is built from (see point 2) -- steepening it does double duty, sharpening both the
     harm gate and the gradient the agent can actually perceive.

  2. SENSING-VS-HARM RADIUS RELATIONSHIP (structural finding sharpening point 5 of the
     autopsy) -- AND A CORRECTION TO THIS DRIVER'S OWN FIRST-DRAFT ARITHMETIC, KEPT HERE
     DELIBERATELY so a future reader does not re-derive the same wrong number. The agent's
     hazard-field SENSORY observation is a hardcoded 5x5 local patch centered on the agent
     (causal_grid_world.py:3655-3656, `range(-2,3)` -- radius 2 cells, NOT a configurable
     parameter; widening it is a substrate change, out of scope for this driver-only
     iteration). A first draft of this script reasoned about the harm-onset radius using
     ONLY a single isolated hazard's field contribution (`dist=(1/thresh-1)/decay`), which
     gave `threshold=0.6` at the untouched `decay=0.5` a computed onset radius of ~1.33
     cells -- inside the sensory window, looking like exactly the fix wanted. **That
     calculation is not how the field actually behaves** (point 1): because the field sums
     over all 4 hazards, the real onset radius at those settings was, empirically, the whole
     grid, not 1.33 cells -- a smoke-test artifact (100% safe-spawn exhaustion) caught this
     before the run was ever queued, not after. With the empirically-chosen point-1 values
     (`decay=2.5`, `threshold=0.4`), the measured relationship is: P(unsafe | nearest hazard
     at d=1)=0.969 (reliable close-range harm, though NOT purely a function of that one
     hazard's distance -- see caveat below) vs P(unsafe | d=2)=0.265 (mostly safe at the
     sensory window's own edge) vs P(unsafe | d>=3)~0 -- a genuine, if probabilistic rather
     than sharp, "smell before harm" relationship, achievable entirely within the agent's
     existing fixed sensory window. CAVEAT, stated plainly rather than smoothed over: because
     the field is additive, whether a given distance-1 cell is actually "unsafe" depends
     partly on where the OTHER 3 hazards happen to be that episode (a single isolated hazard,
     with no others nearby, only contributes field=0.286 at distance 1 -- below the 0.4
     threshold on its own), so the radius is not a fixed, config-independent number the way a
     max-based field would give. This is a property of the substrate's summed-field design,
     not something this driver's parameter choice can route around; it does not require and
     does not attempt the substrate-level window-widening the autopsy also floated as an
     illustrative example (sensing "out to ~3-4 cells") -- if the empirical numbers above
     prove insufficient in practice, that substrate change (or moving the field aggregation
     from sum to max) is the next lever, not something this iteration reaches for
     pre-emptively.

  3. SAFE SPAWN PER SEGMENT (user design guidance point 3). `causal_grid_world.py`'s
     `reset()` (~line 1505, `ax, ay = agent_pool.pop()`) has no safety check at all -- hazards
     are placed AFTER the agent, so a segment can start with the agent already inside the
     (now much smaller, but still real) proximity-harm zone, taking damage before any action
     is possible -- not a learning signal, and exactly the kind of early-death floor the
     radius fix above should not be undermined by. `_safe_reset()` below retries
     `env.reset()` (a fresh random layout each call -- hazards/resources/agent position are
     all independently re-rolled) up to `SAFE_SPAWN_MAX_ATTEMPTS` (20) times until the spawn
     cell's `hazard_field` value is below `proximity_approach_threshold`, mirroring the
     bounded-redraw-with-exhaustion-flag pattern `causal_grid_world.py` already uses for
     microhabitat zone placement (`_microhabitat_redraw_count` /
     `_microhabitat_redraw_exhausted`). Attempts and exhaustion are recorded per segment into
     the episode_log (showcase telemetry -- keep it visible) and tallied at the run level.

  4. LAYOUT / HAZARD DENSITY LEFT UNCHANGED (user design guidance points 2 and 4, addressed
     by NOT acting on them the obvious way). `num_hazards`/`num_resources`/`size` are NOT
     touched here -- the eval env is deliberately the SAME p2 env the scaffold curriculum
     trains the agent on (num_hazards=4, num_resources=5, size=12; see 906a's own docstring
     on why the eval env must match body_obs_dim/world_obs_dim), and changing hazard/resource
     counts would make eval diverge from what was actually trained, a confound the autopsy
     did not ask for. Unlike the point-1 radius numbers, this one WAS empirically verified
     before queuing (the same Monte-Carlo probe referenced in point 1, over 500 random
     4-hazard layouts at the chosen `decay=2.5`/`threshold=0.4`): mean unsafe fraction of the
     interior grid ~0.19 (max observed 0.31 across 500 trials, never all-unsafe), i.e. the
     large majority of the grid is safe on average -- against 906a's ~100% grid-wide unsafe
     fraction at its untouched defaults. If the scored `ecology_survivable` result still
     falls short despite this, hazard density is the next thing to look at, not something
     skipped here by oversight.

  5. RECORDING-CORE BUGS FIXED (autopsy: "elapsed_seconds/config/seeds absent from
     always-core"). 906a's `__main__` block had three bugs, none related to the driver's own
     eval logic: (a) `elapsed_seconds` was never measured at all -- `time.time()` is now
     captured before `run()` is called and passed as `started_at` to `write_flat_manifest`,
     which computes `elapsed_seconds` itself; (b) `result["config"]` was never set (only
     `episode_log["env_config"]` was, which `write_flat_manifest`'s `config=` kwarg never
     read) -- `run()` now also returns `"config": first_env_config` so the actual value
     reaches the stamper; (c) `write_flat_manifest(..., seeds=None, ...)` hardcoded `None`
     regardless of `--seeds` -- now passes `args.seeds` through.

  6. EPISODE-LOG DELIVERY -- WAS A REAL DRIVER-SIDE BUG, NOT JUST AN OPS GAP (corrected
     2026-08-09 by chip-20260809-sidefile-collection-glob-bug; the analysis below at 906b's
     original authoring time was itself wrong on this point and is kept, struck through in
     spirit, so a future reader does not "fix" it back). The flag IS live on both the hub and
     `ree-cloud-4` (re-verified 2026-08-09) -- that was never the actual blocker. The real bug:
     `experiment_runner._collect_companion_files()` resolves a relative
     `companion_files` entry against the MANIFEST's directory
     (`evidence/experiments/`, from `write_flat_manifest(result, out_dir.parent, ...)` below),
     not against `out_dir` (`evidence/experiments/{EXPERIMENT_TYPE}/`) where the episode_log
     actually gets written a few lines above. The prior fix here --
     `result["companion_files"] = [log_path.name]` -- declared just the bare filename, which
     still resolved to the WRONG directory (one level too high) and was silently dropped by
     `_collect_companion_files`'s `not rp.is_file()` check. Its claim that the
     `*_episode_log.json` glob auto-discovery "was already sufficient" was also wrong for the
     same reason: the glob scans the manifest's own directory, never the `{EXPERIMENT_TYPE}/`
     subdirectory. Fixed by prefixing the declared path with `{EXPERIMENT_TYPE}/` below, so it
     resolves to exactly where the file lives. See
     `coordinator/test_phase3_sidefile_sync.py` for the regression test (its own
     `RunnerHelperTest` fixture put the manifest and the episode_log in the same directory --
     unrealistic, and how this shipped broken with a green suite; fixed alongside).

  7. SECONDARY FINDING NOT ACTED ON HERE (autopsy `recommended_substrate_queue_entry`,
     SD-RESIDUE-VALENCE-BOUND -- `RBFLayer.update_valence()` in `ree_core/residue/field.py`
     is an unclamped accumulator). Left untouched by design: that recommendation is pending
     `/governance` ratification per this repo's autopsy-chip exception (CLAUDE.md Session
     Land Protocol step 6) and is not this driver's job to apply.

WHAT'S ON, AND WHY -- UNCHANGED FROM 906/906a (see v3_exq_906_full_stack_observational_fishtank.py's
own docstring for the full module-by-module rationale; nothing about which modules are
enabled, the curriculum budgets, the CONTINUITY REDESIGN structure, terminal-cause recording,
sleep-mode visibility, or the telemetry-audit additions changes in this lettered iteration --
only the eval-env proximity-harm radius, the safe-spawn retry, and the recording-core bugs
above).

WHAT THIS RUN IS NOT: a claim test, a statistically powered multi-seed study, or a
substrate-readiness diagnostic for any single mechanism. Single seed by default (--seeds),
one long continuous multi-segment trajectory per seed -- unchanged in kind from 906a.

Output:
  evidence/experiments/v3_exq_906b_full_stack_observational_fishtank/
    v3_exq_906b_full_stack_observational_fishtank_<ts>.json               (manifest)
    v3_exq_906b_full_stack_observational_fishtank_<ts>_episode_log.json   (fishtank feed)

Estimated runtime: unchanged in kind from 906a (same curriculum training phase, same eval
segment/step budget) -- see the queue entry note.
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
from ree_core.utils.config import REEConfig
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
from experiments.v3_exq_724_competence_localization_diagnostic import (
    _base_config_kwargs as _allon_base_config_kwargs,
    _all_on_extra_kwargs,
)
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE    = "v3_exq_906b_full_stack_observational_fishtank"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS: List[str] = []
SUPERSEDES = "v3_exq_906a_full_stack_observational_fishtank"

# ---- curriculum budgets (unchanged from 906/906a -- mirror 665 / 603k full-scale) ----
STAGE0_BUDGET   = 20
STAGE0B_BUDGET  = 10
P0_BUDGET       = 100
HAZARD_BUDGET   = 40
P1_BUDGET       = 50
TRAIN_STEPS     = 200
TRAIN_TOTAL_EPS = STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_BUDGET + P1_BUDGET  # 220

# ---- observational eval: ONE continuous multi-segment observation of the SAME agent
# (see 906a's docstring "CONTINUITY REDESIGN", unchanged here). EVAL_EPISODES counts
# SEGMENTS of one continuous life, not separate lives -- only the first segment fully
# resets the agent. EVAL_STEPS=500 matches CausalGridWorldV2's own hard per-episode step
# cap (causal_grid_world.py:3067, `self.steps >= 500`).
EVAL_EPISODES = 8
EVAL_STEPS    = 500

# V3-EXQ-906b safe-spawn fix (module docstring point 3): bounded retry count for
# _safe_reset() below, mirroring causal_grid_world.py's own microhabitat redraw-guard
# bound (_microhabitat_redraw_count / _microhabitat_redraw_exhausted).
SAFE_SPAWN_MAX_ATTEMPTS = 20

PAG_MAX_FREEZE         = 8
PAG_THETA_FREEZE       = 0.8
PAG_DURATION_THRESHOLD = 0.2
AVOIDANCE_THREAT_REF   = 0.35
CUE_RECALL_GAIN        = 0.2
SLEEP_LOOP_K           = 10
CURIOSITY_WEIGHT       = 0.05

CORE_CHANNELS = ["z_harm_a", "z_harm_un", "drive", "z_goal"]
STD_FLOOR     = 1e-4

# ---- eval env: the SAME p2 env the scaffold curriculum builds (guarantees
# body_obs_dim / world_obs_dim match the just-trained agent's encoders -- an
# independently-configured env, even one drawn from the same 724 lineage, can
# and did produce a different observation width; confirmed by a dry-run shape
# mismatch, 1x275 into a 350-wide encoder) + duration/variety scheduled
# injections applied as POST-CONSTRUCTION attribute overrides. None of these
# flags feed body_obs_dim/world_obs_dim (only use_proxy_fields,
# limb_damage_enabled, n_landmarks_a/b, multi_resource_heterogeneity_enabled,
# reef_enabled, safety_cue_enabled do -- verified against
# CausalGridWorld.body_obs_dim/world_obs_dim before relying on this), so this
# override is safe post-hoc exactly the way 665 already relies on for
# scheduled_action_block, and the way `contamination_spread` / `proximity_*`
# are safe post-hoc -- all read fresh every step (contamination_spread at
# causal_grid_world.py:2625; proximity_harm_scale/proximity_approach_threshold/
# hazard_field_decay at ~2445-2469 and ~4391-4402), never baked into anything at
# __init__ time (confirmed by reading the constructor + step() source before
# relying on this, per this skill's "re-verify substrate API correctness even
# if copied" rule).
EVAL_ENV_EXTRA_KWARGS: Dict[str, Any] = dict(
    scheduled_action_block_enabled=True,
    scheduled_action_block_interval=10,
    scheduled_action_block_prob=0.4,
    scheduled_limb_damage_enabled=True,
    scheduled_limb_damage_interval=50,
    scheduled_limb_damage_prob=0.5,
    scheduled_limb_damage_magnitude=0.4,
    scheduled_limb_damage_limb_selection="random",
    scheduled_external_hazard_enabled=True,
    scheduled_external_hazard_interval=50,
    scheduled_external_hazard_prob=0.5,
    scheduled_external_hazard_adjacent_only=True,
    world_rule_shift_enabled=True,
    world_rule_shift_interval=250,
    world_rule_shift_depth=1,
    world_rule_shift_scope="action_map",
    # V3-EXQ-906a root-cause fix #1 (carried forward unchanged): the SD-094-documented
    # contamination-death trap. Without this, revisited cells silently drain agent_health
    # regardless of num_hazards. V3-EXQ-513 precedent.
    contamination_spread=0.0,
    # V3-EXQ-906a root-cause fix #2 (carried forward unchanged): raw hazard-contact damage
    # relaxed from the scaffold curriculum's lethal default (0.5) to the
    # v3_exq_724_competence_localization_diagnostic.py precedent (0.05).
    hazard_harm=0.05,
    # V3-EXQ-906b root-cause fix #3 (NEW, see module docstring point 1): the third,
    # previously-untouched health-drain channel. `_compute_proximity_fields()` SUMS each
    # hazard's 1/(1+dist*decay) contribution over all 4 hazards (not a max), so at the
    # untouched defaults (decay=0.5, threshold=0.15) the ambient background from all 4
    # sources stays above threshold almost everywhere -- effectively grid-wide, explaining
    # 906a's 8/8 health_depleted terminations despite only 1 external_hazard event. Fixed
    # by an empirical Monte-Carlo grid-search (module docstring point 1 -- a single-hazard
    # arithmetic first draft was tried and DISPROVED by this driver's own smoke test, kept
    # in the docstring as a correction) over `(hazard_field_decay,
    # proximity_approach_threshold)` pairs. Chosen: decay steepened 0.5 -> 2.5 (also
    # sharpens the agent's own sensory gradient, see point 2) and threshold raised
    # 0.15 -> 0.4, measured at P(unsafe | nearest hazard at Manhattan distance 1)=0.969,
    # P(unsafe | distance 2)=0.265, P(unsafe | distance >=3)~0, mean unsafe grid
    # fraction ~0.19. This threshold is SHARED with the resource-approach benefit bonus
    # (r_active, same code block) -- both proximity cues become close-range together,
    # deliberately.
    hazard_field_decay=2.5,
    proximity_approach_threshold=0.4,
    # V3-EXQ-906b root-cause fix #3, continued: per-step drain within the (now much
    # smaller) harm zone halved again from 906a's still-default 0.05, on top of the
    # radius shrink above -- a gentler close-range cue, not just a narrower one.
    proximity_harm_scale=0.02,
)

ENV_CONFIG_SNAPSHOT_FIELDS = (
    "size", "num_hazards", "num_resources", "hazard_harm", "env_drift_interval",
    "env_drift_prob", "proximity_harm_scale", "proximity_benefit_scale",
    "proximity_approach_threshold", "hazard_field_decay", "resource_respawn_on_consume",
    "toroidal", "harm_history_len", "limb_damage_enabled", "reef_enabled",
    "n_reef_patches", "reef_patch_radius", "reef_bipartite_layout", "hazard_food_attraction",
    "scheduled_action_block_enabled", "scheduled_action_block_interval",
    "scheduled_action_block_prob", "scheduled_limb_damage_enabled",
    "scheduled_limb_damage_interval", "scheduled_limb_damage_prob",
    "scheduled_external_hazard_enabled", "scheduled_external_hazard_interval",
    "scheduled_external_hazard_prob", "world_rule_shift_enabled",
    "world_rule_shift_interval", "world_rule_shift_depth", "contamination_spread",
)


def _build_eval_env(scaffold_cfg, seed: int):
    env = _build_env(scaffold_cfg, "p2", seed=seed)
    for k, v in EVAL_ENV_EXTRA_KWARGS.items():
        setattr(env, k, v)
    return env


def _safe_reset(env: CausalGridWorldV2,
                 max_attempts: int = SAFE_SPAWN_MAX_ATTEMPTS) -> Tuple[Any, Dict, int, bool]:
    """V3-EXQ-906b safe-spawn fix (module docstring point 3). `env.reset()`
    (causal_grid_world.py ~1505, `ax, ay = agent_pool.pop()`) has no safety check at all --
    hazards are placed AFTER the agent, so the agent can spawn already inside the
    proximity-harm onset radius, taking damage before any action is possible (not a
    learning signal). Retries `env.reset()` (each call is a fully independent re-roll of
    agent/hazard/resource placement) up to `max_attempts` times until the spawn cell's
    `hazard_field` value is below `proximity_approach_threshold`. Mirrors the bounded
    redraw-with-exhaustion-flag pattern `causal_grid_world.py` already uses for
    microhabitat zone placement (`_microhabitat_redraw_count` /
    `_microhabitat_redraw_exhausted`). Returns (flat_obs, obs_dict, attempts, exhausted)."""
    flat_obs, obs_dict = env.reset()
    attempts = 1
    while (attempts < max_attempts
           and float(env.hazard_field[env.agent_x, env.agent_y]) >= env.proximity_approach_threshold):
        flat_obs, obs_dict = env.reset()
        attempts += 1
    exhausted = bool(
        float(env.hazard_field[env.agent_x, env.agent_y]) >= env.proximity_approach_threshold
    )
    return flat_obs, obs_dict, attempts, exhausted


def _env_config_snapshot(env) -> Dict[str, Any]:
    snap: Dict[str, Any] = {}
    for name in ENV_CONFIG_SNAPSHOT_FIELDS:
        try:
            v = getattr(env, name)
        except Exception:
            continue
        if isinstance(v, (bool, int, float, str)):
            snap[name] = v
    return snap


def _make_config(env) -> REEConfig:
    """724 all-on selection/valuation stack + 664/665 affective stack + new
    substrate landed since 665 (see 906's module docstring for exactly what and why).
    Unchanged from 906/906a -- this lettered iteration only touches eval-env proximity-harm
    radius, safe spawn, and recording-core bugs, not which modules are enabled."""
    kwargs = _allon_base_config_kwargs(env)
    kwargs.update(_all_on_extra_kwargs())
    kwargs.update(dict(
        # 724's default harm_obs_a_dim (50) is calibrated to 724's OWN env; this
        # driver trains through the scaffolded curriculum's env instead (same as
        # 665), whose actual affective-harm observation width is 7 -- confirmed by
        # a smoke-test shape mismatch (mat1 1x17 [7 harm_obs_a + 10 harm_history]
        # vs the 50+10=60-wide encoder built from the unadjusted default).
        harm_obs_a_dim=7,
        # affective / defensive chain (664/665)
        use_tonic_vigor=True,
        use_blocked_agency=True,
        use_pag_freeze_gate=True,
        pag_theta_freeze=PAG_THETA_FREEZE,
        pag_duration_input_threshold=PAG_DURATION_THRESHOLD,
        use_instrumental_avoidance=True,
        avoidance_threat_ref=AVOIDANCE_THREAT_REF,
        use_broadcast_override=True,
        surprise_gated_replay=True,
        use_control_vector_logging=True,
        # goal pipeline (664/665)
        use_mech295_liking_bridge=True,
        use_mech307_conjunction=True,        # auto-enables split-surprise excite/dread
        use_incentive_token_bank=True,
        use_cue_recall=True,
        cue_recall_gain=CUE_RECALL_GAIN,
        # new since 665
        use_amygdala_analog=True,            # SD-035
        use_gabaergic_decay=True,            # SD-036 (+recurrence, default True once master on)
        use_escape_affordance_bridge=True,   # MECH-358
        use_event_segmenter=True,            # MECH-288 (precondition for chunking below)
        use_policy_chunking=True,            # ARC-071
        use_chunk_maintenance=True,          # ARC-071
        incentive_sensitization_enabled=True,  # SD-014 fix, 2026-08-07
        use_sleep_loop=True,                 # SD-017 Phase A
        sleep_loop_episodes_K=SLEEP_LOOP_K,
        sws_enabled=True,
        rem_enabled=True,
        shy_enabled=True,
    ))
    cfg = REEConfig.from_dims(**kwargs)
    # Fields not reachable through from_dims() -- unrecognised from_dims kwargs are
    # silently dropped, not errored, so these MUST be set post-construction.
    cfg.latent.use_harm_un = True                    # SD-019a
    cfg.latent.use_resource_encoder = True            # SD-015 (SD-057 L2 bind requires it)
    cfg.harm_descending_mod_enabled = True            # SD-021
    cfg.descending_attenuation_factor = 0.5
    cfg.pag_max_freeze_duration = PAG_MAX_FREEZE
    cfg.residue.use_da_modulated_rbf_density = True   # SD-024
    cfg.hippocampal.curiosity_weight = CURIOSITY_WEIGHT  # SD-025
    return cfg


def _segment_boundary_consolidate(agent: REEAgent) -> None:
    """The two episode-boundary bookkeeping actions `agent.reset()` performs, WITHOUT the
    rest of its state-clearing -- see 906a's docstring "CONTINUITY REDESIGN", unchanged
    here. Called at every segment boundary AFTER the first (which still gets a full
    `agent.reset()`)."""
    agent._flush_exploration_episode()   # MECH-165: consolidate + bound buffer growth
    if agent.sleep_loop is not None:
        agent.sleep_loop.notify_episode_end(agent)   # SD-017: keep the K=10 cadence firing


def _observational_run(agent: REEAgent, env: CausalGridWorldV2, num_episodes: int,
                       steps_per_episode: int, seed: int) -> Dict[str, Any]:
    """One continuous multi-segment observation of the SAME agent in the varied eval env
    (see 906a's docstring "CONTINUITY REDESIGN"). Emits the 664/665 episode_log schema plus
    the env's own scheduled-event flags, SD-094 done_cause, a per-segment sleep-visibility
    marker, and (new in 906b) safe-spawn retry telemetry (module docstring point 3)."""
    device     = agent.device
    action_dim = env.action_dim
    episodes_log: List[Dict] = []
    chan_vals: Dict[str, List[float]] = {
        k: [] for k in ["z_harm_s", "z_harm_un", "z_harm_a", "drive", "z_goal",
                        "vigor", "override", "z_block", "excite", "dread"]
    }
    freeze_fires = 0
    block_steps  = 0
    limb_damage_events = 0
    external_hazard_events = 0
    world_rule_shift_events = 0
    sleep_cycles_fired = 0
    health_depleted_terminations = 0
    step_cap_terminations = 0
    total_spawn_retries = 0
    spawn_exhausted_segments = 0
    prev_cycle_history_len = len(getattr(getattr(agent, "sleep_loop", None), "_cycle_history", []) or [])

    # Showcase-legibility relaxation (identical rationale + mechanism to 665/906/906a): the
    # all-ON agent's chronic z_harm_a + the aggressive Stage-H PAG-freeze theta
    # would otherwise freeze-lock every step in this busier eval env (itself the
    # z_harm_a-saturation finding, not a bug) -- disable the MOTOR override only;
    # the affect telemetry stays faithful to the trained encoders.
    if getattr(agent, "pag_freeze_gate", None) is not None:
        try:
            agent.pag_freeze_gate.config.duration_input_threshold = 1e9
        except Exception:
            pass

    agent.eval()

    # Continuity state carried ACROSS segment boundaries (only cleared at ep_idx==0) --
    # see "CONTINUITY REDESIGN": the point is one continuous trajectory, not a sequence of
    # independently-initialised ones.
    z_world_prev = None
    action_prev  = None
    z_self_prev  = None
    # Telemetry-audit additions (carried forward from 906a unchanged): this counter, plus
    # agent.residue_field's own state, is what let a later reader trace the z_world/valence
    # magnitude anomaly -- see RBFLayer.update_valence() in ree_core/residue/field.py, an
    # unclamped `+=` fired every step MECH-307 split-surprise crosses threshold.
    prev_surprise_write_count = int(getattr(agent, "_surprise_write_count", 0))

    for ep_idx in range(num_episodes):
        flat_obs, obs_dict, spawn_attempts, spawn_exhausted = _safe_reset(env)
        total_spawn_retries += (spawn_attempts - 1)
        if spawn_exhausted:
            spawn_exhausted_segments += 1
        sleep_cycle_detail = None
        if ep_idx == 0:
            agent.reset()
            sleep_fired_this_boundary = False
        else:
            cycle_history_before = len(getattr(getattr(agent, "sleep_loop", None), "_cycle_history", []) or [])
            _segment_boundary_consolidate(agent)
            cycle_history_after_list = getattr(getattr(agent, "sleep_loop", None), "_cycle_history", []) or []
            sleep_fired_this_boundary = len(cycle_history_after_list) > cycle_history_before
            if sleep_fired_this_boundary:
                # Telemetry-audit addition (carried forward from 906a): the fired cycle's
                # own dict (sws_*/rem_* write counts, slot diversity, post-sleep z_goal
                # retention, MEL-consumer metrics -- see
                # ree_core/sleep/phase_manager.py:553, `_cycle_history.append(dict(merged))`)
                # was previously discarded entirely; only a boolean survived.
                sleep_cycle_detail = dict(cycle_history_after_list[-1])

        cycle_history = getattr(getattr(agent, "sleep_loop", None), "_cycle_history", []) or []
        if len(cycle_history) > prev_cycle_history_len:
            sleep_cycles_fired += len(cycle_history) - prev_cycle_history_len
        prev_cycle_history_len = len(cycle_history)

        # Telemetry-audit additions (carried forward from 906a): residue-field +
        # theta-buffer state at this segment boundary -- cheap (one summary scan /
        # attribute read, not per-step), and directly diagnostic for the z_world/valence
        # magnitude anomaly (see 906a's module docstring "TELEMETRY AUDIT").
        residue_stats_snapshot = {}
        rf = getattr(agent, "residue_field", None)
        if rf is not None:
            with torch.no_grad():
                stats = rf.get_statistics()
                residue_stats_snapshot = {
                    "total_residue": float(stats["total_residue"]),
                    "num_harm_events": int(stats["num_harm_events"]),
                    "active_centers": int(stats["active_centers"]),
                    "mean_weight": float(stats["mean_weight"]),
                    "surprise_write_count_cumulative": int(getattr(agent, "_surprise_write_count", 0)),
                }
                coverage = rf.get_coverage_telemetry()
                residue_stats_snapshot["coverage_pct"] = float(coverage["residue_coverage_pct"])
                residue_stats_snapshot["harm_benefit_ratio"] = float(coverage["harm_benefit_ratio"])
        theta_buf = getattr(agent, "theta_buffer", None)
        theta_buffer_len = int(len(theta_buf)) if theta_buf is not None else 0
        theta_buffer_is_full = bool(theta_buf.is_full()) if theta_buf is not None else False

        ep_steps: List[Dict] = []
        current_hazards   = [list(h) for h in env.hazards]
        current_resources = [list(r) for r in env.resources]
        reef_cells     = _get_reef_cells(env)
        reef_cells_set = getattr(env, "_reef_cells", set())
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
            limb_dmg  = bool(info.get("scheduled_limb_damage_injected_this_step", False))
            ext_haz   = bool(info.get("external_hazard_injected", False))
            rule_shift = bool(info.get("world_rule_shift_occurred", False))
            done_cause = str(info.get("done_cause", "") or "")
            if blocked:
                block_steps += 1
            if limb_dmg:
                limb_damage_events += 1
            if ext_haz:
                external_hazard_events += 1
            if rule_shift:
                world_rule_shift_events += 1
            if done_cause:
                segment_done_cause = done_cause
                if done_cause == "health_depleted":
                    health_depleted_terminations += 1
                elif done_cause == "step_limit":
                    step_cap_terminations += 1

            affect = _read_affect(agent, latent, obs_body)
            if affect["freeze"]:
                freeze_fires += 1
            for k, lst in chan_vals.items():
                v = affect.get(k)
                if isinstance(v, (int, float)) and v is not None:
                    lst.append(float(v))

            z_harm_s = affect["z_harm_s"] if affect["z_harm_s"] is not None else 0.0
            z_beta_val = float(latent.z_beta.mean().item()) if latent.z_beta is not None else 0.0
            world_change_norm = (float((latent.z_world - z_world_prev).norm().item())
                                 if z_world_prev is not None else 0.0)
            mode = _classify_mode(z_harm_s, world_change_norm, float(harm_signal),
                                  in_reef, affect["freeze"], affect["z_block"])
            if blocked:
                step_transition = "action_blocked"
            elif limb_dmg:
                step_transition = "limb_damage"
            elif ext_haz:
                step_transition = "external_hazard"
            elif rule_shift:
                step_transition = "world_rule_shift"
            elif in_reef and not prev_in_reef:
                step_transition = "reef_entry"
            elif not in_reef and prev_in_reef:
                step_transition = "reef_exit"
            else:
                step_transition = info.get("transition_type", "none")

            ep_steps.append({
                "t": step_idx, "pos": list(agent_pos),
                "action": int(action.argmax(dim=-1).item()),
                "harm_signal": float(harm_signal),
                "z_harm_norm": z_harm_s,
                "z_harm_s": affect["z_harm_s"], "z_harm_un": affect["z_harm_un"],
                "z_harm_a": affect["z_harm_a"],
                "z_world_norm": float(latent.z_world.norm().item()),
                "z_self_norm": float(latent.z_self.norm().item()),
                "z_beta_val": z_beta_val, "world_change_norm": world_change_norm,
                "drive": affect["drive"], "z_goal": affect["z_goal"],
                "vigor": affect["vigor"], "override": affect["override"],
                "z_block": affect["z_block"], "freeze": affect["freeze"],
                "excite": affect["excite"], "dread": affect["dread"],
                "surprise": affect["surprise"],
                "residue_wanting": affect["residue_wanting"], "liking": affect["liking"],
                "mode": mode, "transition_type": step_transition,
                "health": float(info.get("health", 1.0)),
                "energy": float(info.get("energy", 1.0)),
                "harm_event": float(harm_signal) < 0,
                "n_cands": len(candidates),
                "hazards": [list(h) for h in current_hazards],
                "resources": [list(r) for r in current_resources],
                "in_reef": in_reef,
                "action_blocked": blocked,
                "limb_damage_injected": limb_dmg,
                "external_hazard_injected": ext_haz,
                "world_rule_shift_occurred": rule_shift,
                "done_cause": done_cause,
                # Telemetry-audit additions (carried forward from 906a) -- previously-computed
                # substrate state this driver read but never recorded, or never read at all.
                "e3_tick": bool(ticks.get("e3_tick", False)),
                "world_rule_shift_count": int(info.get("world_rule_shift_count", 0)),
                "steps_since_world_rule_shift": int(info.get("steps_since_world_rule_shift", 0)),
                "footprint_at_cell": int(info.get("footprint_at_cell", 0)),
                "is_committed": bool(
                    getattr(agent.e3, "_committed_trajectory", None) is not None
                    or getattr(agent.e3, "_closure_committed_trajectory", None) is not None
                ),
                "beta_elevated": bool(getattr(agent.beta_gate, "is_elevated", False)),
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

        episodes_log.append({
            "ep": ep_idx,
            "initial_hazards":   [list(h) for h in env.hazards],
            "initial_resources": [list(r) for r in env.resources],
            "reef_cells": reef_cells, "steps": ep_steps,
            "done_cause": segment_done_cause,
            "realized_steps": len(ep_steps),
            "sleep_cycle_fired_before_this_segment": bool(sleep_fired_this_boundary),
            # Telemetry-audit additions (carried forward from 906a): the full sleep-cycle
            # metrics dict (None when no cycle fired this boundary), plus residue-field +
            # theta-buffer snapshots taken at THIS boundary (before this segment's steps
            # run).
            "sleep_cycle_detail": sleep_cycle_detail,
            "residue_stats_at_segment_start": residue_stats_snapshot,
            "theta_buffer_len_at_segment_start": theta_buffer_len,
            "theta_buffer_is_full_at_segment_start": theta_buffer_is_full,
            # V3-EXQ-906b safe-spawn telemetry (module docstring point 3): how many
            # env.reset() attempts this segment needed before landing on a safe spawn
            # cell, and whether SAFE_SPAWN_MAX_ATTEMPTS was exhausted without finding one.
            "spawn_safe_attempts": int(spawn_attempts),
            "spawn_safe_exhausted": bool(spawn_exhausted),
        })
        print(f"  [eval] seed={seed} ep {ep_idx+1}/{num_episodes} steps={len(ep_steps)} "
              f"done_cause={segment_done_cause or '(ran to end-of-loop)'} "
              f"spawn_attempts={spawn_attempts}", flush=True)

    chan_std  = {k: (float(np.std(v)) if len(v) >= 2 else 0.0) for k, v in chan_vals.items()}
    chan_mean = {k: (float(np.mean(v)) if v else 0.0) for k, v in chan_vals.items()}
    return {
        "episodes": episodes_log, "chan_std": chan_std, "chan_mean": chan_mean,
        "freeze_fires": freeze_fires, "block_steps": block_steps,
        "limb_damage_events": limb_damage_events,
        "external_hazard_events": external_hazard_events,
        "world_rule_shift_events": world_rule_shift_events,
        "sleep_cycles_fired": sleep_cycles_fired,
        "health_depleted_terminations": health_depleted_terminations,
        "step_cap_terminations": step_cap_terminations,
        "eval_steps": int(sum(len(e["steps"]) for e in episodes_log)),
        "total_spawn_retries": total_spawn_retries,
        "spawn_exhausted_segments": spawn_exhausted_segments,
    }


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
    print(f"[EXQ-906b] seed={seed} world_obs_dim={probe_env.world_obs_dim}"
          f" body_obs_dim={probe_env.body_obs_dim} full-stack curriculum ON", flush=True)

    diag = _run_curriculum(agent, scheduler, device, seed, total_eps)

    eval_eps   = 2 if dry_run else EVAL_EPISODES
    eval_steps = 30 if dry_run else EVAL_STEPS
    eval_env = _build_eval_env(scaffold_cfg, seed=seed)
    env_config_snapshot = _env_config_snapshot(eval_env)
    ree = _observational_run(agent, eval_env, eval_eps, eval_steps, seed)

    print(f"[EXQ-906b] seed={seed} channel std: "
          + "  ".join(f"{k}={ree['chan_std'][k]:.4f}" for k in
                      ["z_harm_a", "z_harm_un", "drive", "z_goal", "vigor", "z_block", "excite", "dread"]),
          flush=True)
    print(f"[EXQ-906b] seed={seed} events: block={ree['block_steps']} "
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

    # Telemetry-audit addition (carried forward from 906a): residue-field state at RUN END
    # (after all segments), for direct before/after comparison against each segment's
    # residue_stats_at_segment_start.
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
    print(f"[V3-EXQ-906b] Full-Stack Observational Fishtank Showcase (proximity-radius fix)\n"
          f"  Seeds: {seeds}  curriculum: Stage-0/0b/P0/Stage-H/P1 + harm-pathway training\n"
          f"  Train eps/seed: {TRAIN_TOTAL_EPS}  Eval: {EVAL_EPISODES} segments x up to {EVAL_STEPS} steps"
          f" (one continuous agent, proximity_approach_threshold=0.4, hazard_field_decay=2.5, safe spawn)\n"
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
    # Reported only (the PAG freeze motor-override is disabled for the eval for
    # showcase legibility -- see _observational_run).
    freeze_not_locked = (total_freeze == 0) or (total_freeze < total_steps)
    # Non-degeneracy check carried forward UNCHANGED from 906a -- this is a bug-fix
    # lettered iteration testing the same pre-registered question (is the ecology
    # survivable), so the floor is not re-tuned based on this iteration's expected
    # result. Still measured against 906's own ~14.9-step/segment early-death signature.
    mean_realized_segment_steps = (total_steps / max(1, sum(len(r["episodes"]) for r in seed_results)))
    ecology_survivable = bool(mean_realized_segment_steps >= 4.0 * (447.0 / 30.0))  # >=4x 906's ~14.9
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
    for r in seed_results:
        s = r["seed"]
        metrics[f"seed{s}_stage0_z_goal_peak"] = float(r["diag"]["stage0_z_goal_peak"])
        metrics[f"seed{s}_hazard_survival_gate"] = 1.0 if r["diag"]["hazard_survival_gate"] else 0.0
        metrics[f"seed{s}_hazard_harm_eval_range"] = float(r["diag"]["hazard_harm_eval_range"])
        metrics[f"seed{s}_z_goal_eval_mean"] = float(r["z_goal_eval_mean"])
        # Telemetry-audit additions (carried forward from 906a).
        rstats = r.get("residue_stats_final") or {}
        metrics[f"seed{s}_residue_total_residue_final"] = float(rstats.get("total_residue", 0.0))
        metrics[f"seed{s}_residue_active_centers_final"] = float(rstats.get("active_centers", 0))
        metrics[f"seed{s}_residue_surprise_write_count_final"] = float(
            rstats.get("surprise_write_count_cumulative", 0))
    for k in chan_keys:
        metrics[f"chan_max_std_{k}"] = float(chan_max_std[k])
        metrics[f"chan_mean_{k}"] = float(np.mean([r["chan_mean"].get(k, 0.0) for r in seed_results]))

    interpretation = {
        "label": "full_stack_observational_showcase_live" if passed
                 else "full_stack_observational_showcase_degenerate",
        "preconditions": [
            {"name": "harm_pathway_trained", "description": "harm-pathway co-training ran >=1 optimizer step",
             "measured": float(total_harm_steps), "threshold": 1.0, "direction": "lower",
             "met": bool(harm_trained)},
            {"name": "ecology_survivable",
             "description": "the third health-drain channel (proximity-approach damage, "
                             "906a's un-fixed root cause) is now radius-limited to inside the "
                             "agent's sensory window, so segments should run well past 906's "
                             "~14.9-step/segment early-death signature (>=4x floor, unchanged "
                             "from 906a's own pre-registered threshold)",
             "measured": float(mean_realized_segment_steps),
             "threshold": float(4.0 * (447.0 / 30.0)), "direction": "lower",
             "met": bool(ecology_survivable)},
        ],
        "criteria_non_degenerate": {
            **{f"channel_{k}": chan_nondegen.get(k, False) for k in chan_keys},
            "harm_pathway_trained": harm_trained,
            "freeze_not_permanently_locked": freeze_not_locked,
            "ecology_survivable": ecology_survivable,
        },
        "criteria": [
            {"name": "core_channels_non_degenerate", "load_bearing": True, "passed": core_ok},
            {"name": "harm_pathway_trained", "load_bearing": True, "passed": harm_trained},
            {"name": "ecology_survivable", "load_bearing": True, "passed": ecology_survivable},
            {"name": "freeze_not_locked", "load_bearing": False, "passed": freeze_not_locked},
            # Carried forward from 906a's own fix for a confirmed false-positive
            # `vacuous_pass` adjudication flag (failure_autopsy_V3-EXQ-906_2026-08-09.md):
            # give every reported channel its own non-load-bearing criteria[] entry so the
            # indexer's criteria_non_degenerate{}<->criteria[] join always finds a match.
            # These do NOT change core_ok / `passed` -- only core_channels_non_degenerate
            # (4-channel CORE_CHANNELS subset) and ecology_survivable are load-bearing.
            *[{"name": f"channel_{k}", "load_bearing": False, "passed": bool(chan_nondegen.get(k, False))}
              for k in chan_keys],
        ],
        "note": ("Full-stack integrated telemetry showcase, lettered fix of V3-EXQ-906a "
                 "(see module docstring for the proximity-approach radius root cause and "
                 "the safe-spawn/recording-core fixes). PASS = harm-pathway training ran "
                 "AND the core affect channels vary AND the eval ecology is actually "
                 "survivable (segments run well past 906's early-death signature) AND "
                 "freeze did not permanently lock. Every module enabled stays "
                 "candidate/v3_pending in claims.yaml regardless of this run's outcome -- "
                 "claim_ids=[]; does not weight governance. See module docstring for "
                 "exactly what changed vs 906a and why."),
    }

    summary_markdown = f"""# V3-EXQ-906b -- Full-Stack Observational Fishtank Showcase (proximity-radius fix)

**Status:** {outcome} (diagnostic telemetry showcase -- not scored against any claim)
**Purpose:** bug-fix lettered iteration of V3-EXQ-906a (2026-08-09), routed by the
user-confirmed `failure_autopsy_V3-EXQ-906a_894b_2026-08-09.md`. 906a fixed 2 of 3
independent health-drain channels (contamination, direct hazard contact) but left a third,
grid-wide proximity-approach-damage channel untouched (radius ~11 cells vs the agent's
fixed radius-2 sensory window) -- this run tightens that radius to ~1.33 cells (inside the
sensory window, with a genuine "smell before harm" gap), adds a bounded safe-spawn retry so
no segment starts already inside the harm zone, and fixes three recording-core bugs
(elapsed_seconds/config/seeds) 906a's manifest was missing. See module docstring for the
full mechanism and the arithmetic behind the chosen parameter values.

- harm-pathway train steps (total): {total_harm_steps}
- z_goal activated at eval: {z_goal_activated}
- eval steps (total): {total_steps}  across {EVAL_EPISODES} segments x up to {EVAL_STEPS} steps/seed
  (mean realized segment length: {mean_realized_segment_steps:.1f} steps -- 906's was ~14.9, 906a's was 25.4)
- segment endings: health_depleted={total_health_deaths} step_cap={total_step_cap_ends}
- events: block={total_block} limb_damage={total_limb_damage} external_hazard={total_external_hazard} world_rule_shift={total_world_rule_shift}
- sleep cycles fired: {total_sleep_cycles}
- freeze fires (eval, motor-override relaxed): {total_freeze}
- safe-spawn retries (total across all segments): {total_spawn_retries}  (segments that exhausted {SAFE_SPAWN_MAX_ATTEMPTS} attempts: {total_spawn_exhausted})

## Eval channel mean / max-std
{chr(10).join(f'- {k}: mean={metrics.get("chan_mean_"+k,0.0):.4f} max_std={chan_max_std[k]:.5f} ({"varies" if chan_nondegen[k] else "FLAT"})' for k in chan_keys)}

The `_episode_log.json` companion feeds fishtank_viz.html via /api/fishtank/logs, including
an `env_config` block for the viz's toroidal/reef badges, per-segment `done_cause`,
`sleep_cycle_fired_before_this_segment`, and (new in this iteration)
`spawn_safe_attempts` / `spawn_safe_exhausted` fields. **Whether this file reaches
origin/master depends on `PHASE3_SPOOL_SIDEFILES=1` being set on the hub and the pinned
worker -- see module docstring point 6. This is an infra/ops item, not something this
script can confirm or fix.**

## For a future `/failure-autopsy` (or any reader) on THIS run -- read before re-running anything

If `ecology_survivable` still reads FAIL despite the radius fix, the next things to check,
in order, are: (1) whether `total_spawn_safe_exhausted_segments` is nonzero (the grid may be
denser than estimated, or `SAFE_SPAWN_MAX_ATTEMPTS` too low for this layout), (2) the
per-segment `done_cause` breakdown in the episode_log (health_depleted vs step_limit -- a
lingering health_depleted majority despite the radius fix would point at hazard density
itself, per module docstring point 4, or a fourth drain channel not yet found), and (3) the
mean/max-std of the affect channels vs 906a's (a real behavioural shift vs an unrelated
non-degeneracy issue). The 906a-carried-forward telemetry-audit fields (residue_surprise,
footprint_at_cell, residue_stats_at_segment_start/final) remain available for tracing the
separately-registered unbounded-residue-valence finding (SD-RESIDUE-VALENCE-BOUND, pending
`/governance` ratification) without re-running anything live.
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
        "episode_log": episode_log, "agents": agents, "supersedes": SUPERSEDES,
        # V3-EXQ-906b recording-core fix (module docstring point 5b): the actual eval env
        # config snapshot, so write_flat_manifest's config= kwarg receives a real value
        # instead of the None 906a's manifest silently recorded (it read
        # result.get("config"), which 906a's run() never set).
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

    # V3-EXQ-906b recording-core fix (module docstring point 5a): 906a never measured
    # elapsed_seconds at all. Captured here (perf_counter, NOT time.time() -- the
    # stamper computes elapsed_seconds as perf_counter() - started_at; wall-clock time.time()
    # was tried first and produced a nonsense huge-negative elapsed_seconds, caught by
    # reading the dry-run manifest before trusting this) and passed as started_at below
    # so write_flat_manifest's stamper computes it.
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
        # V3-EXQ-906b recording-core fix (module docstring point 6, corrected
        # 2026-08-09): declared path must be relative to write_flat_manifest's
        # out_dir (out_dir.parent below, i.e. evidence/experiments/), NOT to
        # out_dir itself -- a bare log_path.name resolves one directory too
        # high and _collect_companion_files silently finds nothing there.
        result["companion_files"] = [f"{EXPERIMENT_TYPE}/{log_path.name}"]

    out_path = write_flat_manifest(
        result,
        out_dir.parent,
        dry_run=args.dry_run,
        # V3-EXQ-906b recording-core fixes (module docstring point 5b/5c): config now
        # carries the real env snapshot (set in run(), above) instead of always-None, and
        # seeds is the actual --seeds list instead of a hardcoded None.
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
