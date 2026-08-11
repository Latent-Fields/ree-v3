"""
V3-EXQ-920 -- Uncensored Survival-to-Death Fishtank, TRUE Single-Life Design (906b
ecology tier, max_episode_steps-enabled)

Claims: None (diagnostic characterization; does not weight governance)

EXPERIMENT_PURPOSE = "diagnostic"

SLEEP DRIVER: K=never-within-eval (deliberate design consequence, not a bug -- see
"SLEEP-CADENCE DESIGN NOTE" below). TRAINING keeps 906b's unchanged manual-multi
cadence (sleep_loop_episodes_K=10, fired via _run_curriculum's own boundary calls,
untouched by this script). The EVAL phase reuses _observational_run() UNCHANGED with
num_episodes=1: _observational_run's own boundary logic calls agent.reset() at
ep_idx==0 and agent.sleep_loop.notify_episode_end() (via
_segment_boundary_consolidate) only at every NON-ZERO ep_idx boundary. With exactly
one episode, ep_idx never exceeds 0, so notify_episode_end() is never called during
the observed life, regardless of how long that life runs. No sleep cycle can fire
within this run's eval window.

WHY THIS RUN (routing + predecessor). Chip chip-20260810-fishtank-uncensored-survival-v2,
spawned from REE_assembly/evidence/planning/organism_lifespan_development_review_906_lineage_2026-08-10.md
Section 10 item 1, itself following on from the substrate change this run consumes:
ree-v3 9d3d148ff8 (chip-20260810-fishtank-max-episode-steps, landed 2026-08-10) added a
`max_episode_steps: int = 500` constructor kwarg to CausalGridWorldV2
(ree_core/environment/causal_grid_world.py), replacing the previously-hardcoded
`self.steps >= 500` literal `_step_cap_reached` check with `self.steps >=
self.max_episode_steps`. That substrate change is registered
IMPLEMENTED/closed/ready in substrate_queue.json as SD-FISHTANK-MAX-EPISODE-STEPS
(implementation_note dated 2026-08-10), whose own `validation_experiment` field
explicitly defers "the actual uncensored-single-life survival experiment that
consumes this kwarg" to a separate /queue-experiment follow-on -- this run.

PREDECESSOR -- V3-EXQ-912 (ree-v3 a3d36d4878, ran 2026-08-10, outcome FAIL /
"uncensored_survival_still_censoring_dominated") is NOT superseded by this run in the
claims.yaml sense (claim_ids=[] on both; nothing to supersede), but IS the workaround
this run replaces methodologically. At authoring time, CausalGridWorldV2 had no
max_episode_steps kwarg, so 912's own "SUBSTRATE READINESS FINDING" (its module
docstring) correctly found that neither of the review's two literal design options --
(a) raise the per-segment cap and run longer, or (b) add a terminal-death branch --
was buildable from /queue-experiment alone, and instead scaled the NUMBER of 500-step
segments per seed (60 vs the 906b/906c/911 lineage's 8) to accumulate exposure. That
workaround is now unnecessary: the substrate gap it named is closed, and THIS run
implements design (a) literally, as the review originally described it -- "the
current substrate already supports this without further change beyond
removing/raising the cap and running longer" is now TRUE, not aspirational.

DESIGN-DECISION DECLARATION (review Section 10 item 1's mandatory explicit choice
between (a) body-respawn-as-death-unit and (b) a new terminal-death branch). This run
picks (a), and -- unlike 912, which could only approximate (a) via many short
segments -- implements it literally: EVAL_EPISODES=1 per seed. Because
_observational_run() only calls env.reset() at a segment boundary, and there is
exactly one segment, NO body-respawn occurs anywhere within the observed life -- the
single episode runs, unbroken, from spawn to either genuine `health_depleted` or the
`max_episode_steps` ceiling. This closes the residual imprecision in labelling 912's
enumeration-of-many-short-segments a variant of (a): here there is no "inter-respawn"
interval to report, because there is no respawn during the eval window at all. Option
(b) (a new terminal-death branch, e.g. ending the whole run rather than the
substrate's env.reset()-driven revival) remains out of scope for /queue-experiment --
it would be a further substrate change, chippable separately if this run's result
motivates it (see "FOR A FUTURE READER" below).

STEP-BUDGET CALIBRATION (empirical, from V3-EXQ-912's own seed-0 data -- the only
completed uncensored-survival run on this ecology tier). 912 observed 4 genuine
health_depleted terminations across 60 x 500-step segments (n_uncensored=4,
n_censored=56, pct_right_censored=93.3%), giving a naive per-500-step hazard estimate
p~=4/60~=6.7%, with observed death times (uncensored, within the artificial 500-step
window) ranging 261-487 steps (mean 382, median 390). Treating this as approximately
memoryless for calibration purposes only (a simplification -- the review's own Section
7 leaves within-life competence change an open, unresolved question; this run's own
result is itself the correct test of memorylessness, not an assumption to import from
912), P(right-censored at N steps) ~= (1-p)^(N/500). EVAL_STEPS=20000 (40x the legacy
500-step segment) gives P(censored)~=(0.933)^40~=6.2% per seed -- i.e. an expected
~94% chance of observing a genuine single-life death per seed, comfortably satisfying
the review's "cap far beyond anything currently observed" requirement (the longest
genuine death 912 observed was 487 steps; 20000 is ~41x that). SEEDS=8 gives an
expected ~7.5 genuine uncensored deaths under this naive rate (pre-registered PASS
floor set below that, at 4, since this design trades segment count for exposure depth
per seed and the naive rate is explicitly not trusted as ground truth).

SUBSTRATE READINESS CHECK (Step 2.5/2.5a, this run). Read
ree_core/environment/causal_grid_world.py directly (not just substrate_queue.json's
status field): `max_episode_steps: int = 500` constructor kwarg exists (__init__,
~line 147), stored as `self.max_episode_steps` (~line 805), and
`_step_cap_reached = self.steps >= self.max_episode_steps` (~line 3092) reads it
fresh every step -- exactly like the other EVAL_ENV_EXTRA_KWARGS retunes
(contamination_spread, hazard_field_decay, etc.) that 906b already applies as
POST-CONSTRUCTION setattr overrides (906b module docstring lines ~260-272: "None of
these flags feed body_obs_dim/world_obs_dim... this override is safe post-hoc...
all read fresh every step"). Empirically probed before writing this script (scratch,
not committed): constructed a bare CausalGridWorldV2, confirmed default
max_episode_steps=500; setattr'd it to 1200; confirmed the override value survives
env.reset() unchanged (reset() never touches self.max_episode_steps); stepped a
random-action policy and confirmed env.step()'s 5-tuple unpacking
(flat_obs, harm_signal, done, info, obs_dict) with a real info["done_cause"] field
(the random policy died genuinely of health_depleted at step 39, well before either
the old 500 cap or the new 1200 override -- consistent with an untrained policy, not
evidence either way about the cap itself, but the setattr-persists-through-reset
behaviour was confirmed directly, which is what this run's design depends on).
Because max_episode_steps is a post-construction-settable attribute like the other
EVAL_ENV_EXTRA_KWARGS entries, this run reuses 906b's `_build_env` (via
scaffolded_sd054_onboarding) and applies its OWN extended kwargs dict (906b's
unmodified + max_episode_steps=EVAL_STEPS) with the identical setattr loop pattern --
no substrate-side code change needed beyond the already-landed kwarg.

ECOLOGY CHOICE -- 906b's, NOT 911's (same reasoning as 912, restated because it is
still the governing choice here). This driver reuses `_make_config`,
`_env_config_snapshot`, `_observational_run`, `TRAIN_TOTAL_EPS`, `CORE_CHANNELS`,
`STD_FLOOR` UNMODIFIED from v3_exq_906b_full_stack_observational_fishtank, and builds
its OWN eval env kwargs as 906b's EVAL_ENV_EXTRA_KWARGS plus max_episode_steps (NOT
911's additional resource_field_decay=3.0/proximity_benefit_scale=0.01 retune): 911's
own scored 8-segment run produced ZERO health_depleted terminations, which would make
a long single-life run on that ecology tier likely to censor at whatever ceiling is
chosen, defeating this run's purpose of characterizing WHEN death happens.

WELFARE INSTRUMENTATION (reef_ecology_strategy_affective_occupancy_review_2026-08-10.md
Section 9, "Long-life welfare instrumentation as experimental hygiene" -- same
citation and mechanism as 912). That review's trigger condition ("a future successor
gives REE a substantially longer or richer continuous life") is met even more sharply
here than by 912: a single seed's continuous life is now up to EVAL_STEPS=20000 steps
in ONE unbroken segment (vs 912's 60 separate 500-step segments totalling a similar
raw step count but never a single continuous exposure longer than 500 steps).
`lifetime_affective_occupancy` (fraction of the life's steps with dread / z_harm_a
above this run's own within-life 75th percentile, harm-event rate, in_reef fraction)
is computed per seed and reported as a NON-GATING descriptive statistic, per SENT-2
(governance.welfare.welfare_budget, candidate, binds_at_version v4) hygiene. No
inference of sentience, suffering, or welfare is made; SENT-0 boundary applies
exactly as it does to every other V3 diagnostic (see ethics preflight block below).

SLEEP-CADENCE DESIGN NOTE (a genuine finding this run's design surfaces, not
something it resolves). Because sleep only fires at inter-segment boundaries in the
current substrate (agent.sleep_loop.notify_episode_end() is called exclusively from
_segment_boundary_consolidate(), itself called only for ep_idx>0), a TRUE single
unbroken life -- exactly the design the organism-lifespan review asked for -- can
NEVER sleep during that life, no matter how long it runs. Every prior 906-lineage
run's incidental sleep-cycle sampling (906b/906c/911: 0-1 boundaries/seed at 8
segments; 912: ~6 boundaries/seed at 60 segments) was an artifact of the very
segment-chunking this run's design deliberately removes. This is worth naming
explicitly for a future reader: if REE's substrate is meant to sleep as an emergent
consequence of lived experience (time- or fatigue-triggered) rather than only at the
experimenter's artificial recording-chunk boundaries, that is a SEPARATE substrate
question this run's result cannot speak to and does not attempt to -- flagged as a
FOR-A-FUTURE-READER item below, not built here.

CUMULATIVE STEP COUNTER. Organism-lifespan review Section 1's flagged gap ("no
manifest field stores a cumulative/monotonic step counter across the whole run") does
not apply in the same way here as it did to 912's multi-segment design -- with
EVAL_EPISODES=1, `cumulative_lived_steps == realized_steps` trivially, since there is
only one segment. Still stamped (`cumulative_step_start`=0,
`cumulative_step_end`=realized_steps) on each seed's life record for downstream
tooling consistency with the 906-lineage schema.

EPISODE-LOG SIZE MANAGEMENT (a deliberate storage trade-off, stated explicitly).
906b's committed 8-segment/3909-step episode_log is 9.5MB (~2.4KB/step at full
per-step richness). At up to 8 seeds x 20000 steps = 160000 steps worst-case, an
unthinned log would be on the order of 390MB -- disproportionate to commit. Full
per-step records are kept only for: (i) the first FULL_LOG_FIRST_K_SEEDS=2 seeds
(visual-QA / fishtank_viz continuity, matching 912's "first K segments" convention
adapted to "first K seeds" since there is only one segment per seed here), and (ii)
EVERY seed whose life ended `health_depleted` (the event of scientific interest --
inspecting the steps immediately before a genuine death needs the full trace). All
other (right-censored, non-early-seed) lives are stored as a SLIM per-life summary
(done_cause, realized_steps, cumulative markers, per-channel MEAN computed from the
full per-step array before it is dropped) -- exactly 912's `_thin_segment_for_storage`
logic, reused verbatim (renamed for clarity) since the per-life dict shape is
identical to 912's per-segment dict shape. Every statistic reported in the manifest
is computed from the FULL, unthinned in-memory per-seed life BEFORE thinning is
applied, so thinning cannot bias any reported number.

GOV-REUSE-1 (Step 2.4). Decisive readout: the empirical distribution of TRUE
single-continuous-life survival times (uncensored health_depleted step counts, or
right-censored at EVAL_STEPS) on this ecology tier, with no segment-boundary
respawns anywhere in the observed window. Searched existing manifests: every
906-lineage run (906, 906a, 906b, 906c, 911) records at most 8 segments of <=500
steps each; V3-EXQ-912 records 60x500-step segments/seed but still never a single
continuous exposure longer than 500 steps -- no manifest anywhere records a
continuous single-life trajectory beyond the substrate's former 500-step ceiling.
Not recoverable by reanalysis (the underlying substrate limitation that made every
prior run's longest continuous segment exactly 500 steps did not exist as
configurable data to reanalyze) -- a run is required.

Re-derive brake (2.5b): claim_ids=[] -- not applicable, no claim to brake-check.
Substrate-path overlap gate (2.5c): checked substrate_queue.json for open
`corrupting` entries touching causal_grid_world.py or this driver's other imports
(scaffolded_sd054_onboarding, v3_exq_665, v3_exq_906b) -- the only unset-severity
open entry found (mech203-valence-pool-admissibility) touches
ree_core/agent.py::update_harm_salience/update_benefit_salience and
ree_core/neuromodulation/serotonin.py, neither of which this driver's inherited
_observational_run step loop calls (it calls agent.update_z_goal, not
update_benefit_salience/update_harm_salience) -- no overlap, and severity is unset
(not corrupting) regardless.

Ethics preflight (Step 2.6, descriptive, non-enforced): involves_negative_valence=false,
involves_suffering_like_state=false, involves_self_model=false,
involves_inescapability_or_helplessness=false, involves_offline_replay_over_harm=false,
involves_social_mind_or_language=false, involves_human_data_or_clinical_context=false,
decision=allow. V3 has no live self-model in E3, no autobiographical memory, no social
mind, no language -- unchanged by this run's much longer single-episode exposure,
which lengthens EXPOSURE within the same pre-ethical instrumentation, not the
architecture. The welfare-occupancy reporting above operationalizes SENT-2 hygiene
given that longer exposure; it does not change this preflight's allow decision.

WHAT THIS RUN IS NOT: a claim test, a terminal-death mechanic (option (b) above), or
evidence about whether sleep SHOULD be experience-triggered rather than
boundary-triggered (see "SLEEP-CADENCE DESIGN NOTE"). It is the first TRUE
single-continuous-life uncensored survival characterization this substrate can
produce, on the ecology tier most likely to yield genuine deaths.

FOR A FUTURE READER (or /failure-autopsy) ON THIS RUN. If `n_uncensored_deaths_total`
is near zero despite reusing 906b's (not 911's) ecology and the 40x step-budget
calibration above, either this ecology tier's death rate has drifted since 912, or
within-life competence genuinely improves with sustained exposure (the review's own
Section 7 flagged-but-unresolved question) enough to suppress death well past 500
steps -- this run's own uncensored death-time distribution, if any deaths occur, is
now the direct within-life-competence-vs-time data the review's Section 7 lacked (no
segment-boundary confound, since there are no segment boundaries within a life). If
`pct_right_censored_pooled` stays high, the next lever is a substantially larger
EVAL_STEPS ceiling (cheap: it is now a driver-level config choice, not a substrate
gap) -- NOT a return to segment-count scaling. The "SLEEP-CADENCE DESIGN NOTE" above
is a separate, un-built follow-on: whether sleep should trigger on elapsed
time/fatigue within a single life, independent of segment boundaries, is a substrate
design question this run surfaces but does not answer.

Output:
  evidence/experiments/v3_exq_920_uncensored_survival_single_life_fishtank/
    v3_exq_920_uncensored_survival_single_life_fishtank_<ts>.json               (manifest)
    v3_exq_920_uncensored_survival_single_life_fishtank_<ts>_episode_log.json   (fishtank
                                                                                  feed, thinned
                                                                                  per "EPISODE-LOG
                                                                                  SIZE MANAGEMENT")

Estimated runtime: calibrated from V3-EXQ-911's real elapsed_seconds (1006s for one
seed, 220-episode curriculum + 4000 eval steps, all censored -- no early exits) and
V3-EXQ-912's (2514s for one seed, 220-episode curriculum + 29529 realized eval steps)
via a linear fit (fixed curriculum cost ~=12.9 min + ~=0.000983 min/eval-step). Per
seed worst case (full EVAL_STEPS=20000 realized, i.e. right-censored):
~=12.9 + 0.000983*20000 ~= 32.6 min. 8 seeds worst-case total ~= 261 min; realistically
lower since seeds that die early realize fewer than 20000 steps.
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
    _env_config_snapshot,
    _observational_run,
    EVAL_ENV_EXTRA_KWARGS as _906B_EVAL_ENV_EXTRA_KWARGS,
    TRAIN_TOTAL_EPS,
    CORE_CHANNELS,
    STD_FLOOR,
)
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE    = "v3_exq_920_uncensored_survival_single_life_fishtank"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS: List[str] = []

# ---- TRUE single-continuous-life design: EVAL_EPISODES=1 (no segment boundary, no
# body-respawn anywhere in the observed window). See module docstring "STEP-BUDGET
# CALIBRATION" for the derivation of EVAL_STEPS=20000 from V3-EXQ-912's own seed-0
# hazard-rate data.
EVAL_EPISODES = 1
EVAL_STEPS    = 20000

# max_episode_steps is applied to the environment as a POST-CONSTRUCTION setattr
# override, identically to every other EVAL_ENV_EXTRA_KWARGS entry (906b module
# docstring lines ~260-272) -- set EQUAL to EVAL_STEPS so the environment's own
# internal step-limit termination (done_cause="step_limit") coincides exactly with
# _observational_run's own `for step_idx in range(steps_per_episode)` loop bound,
# giving a clean, unambiguous done_cause on every life (never an artificial
# driver-side truncation with done_cause="").
EVAL_ENV_EXTRA_KWARGS: Dict[str, Any] = dict(_906B_EVAL_ENV_EXTRA_KWARGS)
EVAL_ENV_EXTRA_KWARGS["max_episode_steps"] = EVAL_STEPS

# Pre-registered (Step 3): minimum total genuine (uncensored) health_depleted events
# across ALL seeds combined for this run to have obtained a usable empirical
# single-life death-time distribution. Set well below the ~7.5 events the naive
# memoryless extrapolation from V3-EXQ-912 predicts (module docstring "STEP-BUDGET
# CALIBRATION"), since that extrapolation is explicitly not trusted as ground truth
# and each of this run's 8 draws is far more expensive than one of 912's 120
# short-segment draws.
MIN_UNCENSORED_DEATHS_TOTAL = 4

# EPISODE-LOG SIZE MANAGEMENT (module docstring): keep full per-step records for the
# first K seeds (viz continuity) plus every seed whose life genuinely ended
# health_depleted; thin the rest.
FULL_LOG_FIRST_K_SEEDS = 2

# Lifetime affective-occupancy percentile (reef_ecology_strategy review Section 9).
OCCUPANCY_PERCENTILE = 75


def _build_eval_env_uncensored(scaffold_cfg, seed: int):
    """Local variant of 906b's `_build_eval_env`: identical `_build_env(..., "p2",
    seed=seed)` construction + post-construction setattr loop, but applies THIS
    module's own EVAL_ENV_EXTRA_KWARGS (906b's unmodified retune plus
    max_episode_steps=EVAL_STEPS) rather than importing 906b's `_build_eval_env`
    directly, since that function is hard-bound to 906b's own module-level dict."""
    env = _build_env(scaffold_cfg, "p2", seed=seed)
    for k, v in EVAL_ENV_EXTRA_KWARGS.items():
        setattr(env, k, v)
    return env


def _thin_life_for_storage(life: Dict[str, Any], keep_full: bool) -> Dict[str, Any]:
    """V3-EXQ-912's `_thin_segment_for_storage`, reused verbatim (renamed for this
    run's per-life framing -- the per-life dict shape produced by
    _observational_run() with num_episodes=1 is identical to 912's per-segment dict
    shape, since it is the same function). Returns `life` unchanged when `keep_full`;
    otherwise drops the (large) per-step `steps` list and replaces it with a small
    per-life summary computed from it. Never mutates `life` -- callers still need the
    original, full-richness dict for in-memory statistics computed BEFORE this is
    called."""
    if keep_full:
        return life
    steps = life.get("steps") or []
    summary = dict(life)
    summary.pop("steps", None)
    summary["steps_stored"] = "thinned"
    if steps:
        for ch in ("dread", "z_harm_a", "z_harm_s", "drive", "z_goal", "vigor"):
            vals = [s.get(ch) for s in steps if isinstance(s.get(ch), (int, float))]
            summary[f"mean_{ch}"] = float(np.mean(vals)) if vals else None
        summary["frac_harm_event"] = float(np.mean([bool(s.get("harm_event")) for s in steps]))
        summary["frac_in_reef"] = float(np.mean([bool(s.get("in_reef")) for s in steps]))
    return summary


def _lifetime_affective_occupancy(life: Dict[str, Any]) -> Dict[str, Any]:
    """reef_ecology_strategy_affective_occupancy_review_2026-08-10.md Section 9:
    descriptive, non-gating, within-life-percentile occupancy stats over the SINGLE
    continuous life this seed produced. No inference of welfare/sentience -- SENT-0
    boundary."""
    all_dread: List[float] = []
    all_z_harm_a: List[float] = []
    all_harm_event: List[bool] = []
    all_in_reef: List[bool] = []
    for s in life.get("steps", []) or []:
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

    print(f"\nSeed {seed} Condition single_life_uncensored_survival", flush=True)
    scaffold_cfg = _make_scaffold_cfg(dry_run)
    probe_env = _build_env(scaffold_cfg, "p2")
    probe_env.reset()
    agent = REEAgent(_make_config(probe_env)).to(device)
    scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)
    print(f"[EXQ-920] seed={seed} world_obs_dim={probe_env.world_obs_dim}"
          f" body_obs_dim={probe_env.body_obs_dim} full-stack curriculum ON (906b ecology,"
          f" max_episode_steps={EVAL_STEPS})", flush=True)

    diag = _run_curriculum(agent, scheduler, device, seed, total_eps)

    eval_steps = 30 if dry_run else EVAL_STEPS
    eval_env = _build_eval_env_uncensored(scaffold_cfg, seed=seed)
    env_config_snapshot = _env_config_snapshot(eval_env)
    ree = _observational_run(agent, eval_env, EVAL_EPISODES, eval_steps, seed)

    life = ree["episodes"][0]
    life["cumulative_step_start"] = 0
    life["cumulative_step_end"] = int(life["realized_steps"])
    cumulative_lived_steps = int(life["realized_steps"])
    done_cause = life.get("done_cause") or ""
    occupancy = _lifetime_affective_occupancy(life)

    print(f"[EXQ-920] seed={seed} channel std: "
          + "  ".join(f"{k}={ree['chan_std'][k]:.4f}" for k in
                      ["z_harm_a", "z_harm_un", "drive", "z_goal", "vigor", "z_block", "excite", "dread"]),
          flush=True)
    print(f"[EXQ-920] seed={seed} realized_steps={life['realized_steps']} "
          f"done_cause={done_cause or '(incomplete)'} cumulative_lived_steps={cumulative_lived_steps}",
          flush=True)

    seed_core_ok = all(ree["chan_std"].get(k, 0.0) > STD_FLOOR for k in CORE_CHANNELS)
    harm_trained = (diag["p0_harm_train_steps"] + diag["hazard_harm_train_steps"]) > 0
    seed_pass = bool(seed_core_ok and harm_trained)
    print(f"verdict: {'PASS' if seed_pass else 'FAIL'} seed={seed} "
          f"core_ok={seed_core_ok} harm_trained={harm_trained}", flush=True)

    # Episode-log thinning for the WRITTEN companion file only (module docstring
    # "EPISODE-LOG SIZE MANAGEMENT"). Statistics above were already computed from
    # the full, unthinned data.
    keep_full = (seed < FULL_LOG_FIRST_K_SEEDS) or (done_cause == "health_depleted")
    life_stored = _thin_life_for_storage(life, keep_full)

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
        "harm_trained": harm_trained,
        "life_full": life, "life_stored": life_stored,
        "kept_full": bool(keep_full),
        "agent": agent, "env_config": env_config_snapshot,
        "residue_stats_final": residue_stats_final,
        "total_spawn_retries": ree["total_spawn_retries"],
        "spawn_exhausted_segments": ree["spawn_exhausted_segments"],
        "occupancy": occupancy,
        "cumulative_lived_steps": cumulative_lived_steps,
        "realized_steps": int(life["realized_steps"]),
        "done_cause": done_cause,
    }


def run(seeds=None, dry_run: bool = False) -> dict:
    if seeds is None:
        seeds = [0]
    print(f"[V3-EXQ-920] Uncensored Survival-to-Death Fishtank, TRUE Single-Life Design\n"
          f"  Seeds: {seeds}  curriculum: Stage-0/0b/P0/Stage-H/P1 + harm-pathway training\n"
          f"  Train eps/seed: {TRAIN_TOTAL_EPS}  Eval: 1 continuous life x up to {EVAL_STEPS} "
          f"steps/seed (906b ecology, max_episode_steps={EVAL_STEPS} via SD-FISHTANK-MAX-EPISODE-STEPS)\n"
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

    # ---- pooled single-life survival stats (one observation per seed) ----
    uncensored_times = [r["realized_steps"] for r in seed_results if r["done_cause"] == "health_depleted"]
    censored_times    = [r["realized_steps"] for r in seed_results if r["done_cause"] == "step_limit"]
    incomplete_times  = [r["realized_steps"] for r in seed_results
                          if r["done_cause"] not in ("health_depleted", "step_limit")]
    n_seeds_total = len(seed_results)
    n_uncensored_total = len(uncensored_times)
    n_censored_total = len(censored_times)
    pooled_pct_censored = float(n_censored_total / n_seeds_total) if n_seeds_total else 0.0

    core_ok = all(chan_nondegen.get(k, False) for k in CORE_CHANNELS)
    harm_trained = total_harm_steps > 0
    freeze_not_locked = (total_freeze == 0) or (total_freeze < total_steps)
    sufficient_uncensored_deaths = bool(n_uncensored_total >= MIN_UNCENSORED_DEATHS_TOTAL)
    all_seeds_completed = bool(n_seeds_total == len(seeds))
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
        # --- primary DV: pooled single-life survival-time distribution (one draw/seed) ---
        "n_seeds_total": float(n_seeds_total),
        "n_uncensored_deaths_total": float(n_uncensored_total),
        "n_censored_step_limit_total": float(n_censored_total),
        "n_incomplete_other_total": float(len(incomplete_times)),
        "pct_right_censored_pooled": pooled_pct_censored,
        "uncensored_survival_min": float(min(uncensored_times)) if uncensored_times else None,
        "uncensored_survival_max": float(max(uncensored_times)) if uncensored_times else None,
        "uncensored_survival_mean": (float(statistics.fmean(uncensored_times))
                                      if uncensored_times else None),
        "uncensored_survival_median": (float(statistics.median(uncensored_times))
                                        if uncensored_times else None),
        "censored_survival_min": float(min(censored_times)) if censored_times else None,
        "censored_survival_max": float(max(censored_times)) if censored_times else None,
    }
    for r in seed_results:
        s = r["seed"]
        metrics[f"seed{s}_stage0_z_goal_peak"] = float(r["diag"]["stage0_z_goal_peak"])
        metrics[f"seed{s}_hazard_survival_gate"] = 1.0 if r["diag"]["hazard_survival_gate"] else 0.0
        metrics[f"seed{s}_hazard_harm_eval_range"] = float(r["diag"]["hazard_harm_eval_range"])
        metrics[f"seed{s}_z_goal_eval_mean"] = float(r["z_goal_eval_mean"])
        metrics[f"seed{s}_realized_steps"] = float(r["realized_steps"])
        metrics[f"seed{s}_done_cause_is_health_depleted"] = 1.0 if r["done_cause"] == "health_depleted" else 0.0
        metrics[f"seed{s}_cumulative_lived_steps"] = float(r["cumulative_lived_steps"])
        metrics[f"seed{s}_life_kept_full_in_log"] = 1.0 if r["kept_full"] else 0.0
        rstats = r.get("residue_stats_final") or {}
        metrics[f"seed{s}_residue_total_residue_final"] = float(rstats.get("total_residue", 0.0))
        metrics[f"seed{s}_residue_active_centers_final"] = float(rstats.get("active_centers", 0))
    for k in chan_keys:
        metrics[f"chan_max_std_{k}"] = float(chan_max_std[k])
        metrics[f"chan_mean_{k}"] = float(np.mean([r["chan_mean"].get(k, 0.0) for r in seed_results]))

    lifetime_affective_occupancy = {f"seed{r['seed']}": r["occupancy"] for r in seed_results}

    interpretation = {
        "label": "single_life_uncensored_survival_distribution_obtained" if sufficient_uncensored_deaths
                 else "single_life_uncensored_survival_still_censoring_dominated",
        "preconditions": [
            {"name": "harm_pathway_trained", "description": "harm-pathway co-training ran >=1 optimizer step",
             "measured": float(total_harm_steps), "threshold": 1.0, "direction": "lower",
             "met": bool(harm_trained)},
            {"name": "all_seeds_completed",
             "description": "every requested seed produced exactly one completed single-life "
                             "observation (no early crash/truncation silently shrinking n)",
             "measured": float(n_seeds_total), "threshold": float(len(seeds)), "direction": "lower",
             "met": all_seeds_completed},
        ],
        "criteria_non_degenerate": {
            **{f"channel_{k}": chan_nondegen.get(k, False) for k in chan_keys},
            "harm_pathway_trained": harm_trained,
            "freeze_not_permanently_locked": freeze_not_locked,
            "sufficient_uncensored_deaths": sufficient_uncensored_deaths,
        },
        "criteria": [
            {"name": "core_channels_non_degenerate", "load_bearing": True, "passed": core_ok},
            {"name": "harm_pathway_trained", "load_bearing": True, "passed": harm_trained},
            {"name": "sufficient_uncensored_deaths", "load_bearing": True,
             "passed": sufficient_uncensored_deaths},
            {"name": "freeze_not_locked", "load_bearing": False, "passed": freeze_not_locked},
            {"name": "all_seeds_completed", "load_bearing": False, "passed": all_seeds_completed},
            *[{"name": f"channel_{k}", "load_bearing": False, "passed": bool(chan_nondegen.get(k, False))}
              for k in chan_keys],
        ],
        "note": (f"TRUE single-continuous-life ({len(seeds)} independent lives, one per seed, "
                 f"each up to {EVAL_STEPS} steps with NO segment-boundary body-respawn anywhere "
                 f"in the observed window) characterization of the health_depleted death-time "
                 f"distribution on the 906b ecology tier, superseding V3-EXQ-912's "
                 f"segment-count-scaling workaround now that CausalGridWorldV2's per-episode "
                 f"step cap is a driver-configurable max_episode_steps kwarg "
                 f"(SD-FISHTANK-MAX-EPISODE-STEPS, ree-v3 9d3d148ff8). PASS = harm-pathway "
                 f"training ran AND the core affect channels vary AND >= "
                 f"{MIN_UNCENSORED_DEATHS_TOTAL} genuine (uncensored) single-life deaths were "
                 f"observed across all seeds combined (observed: {n_uncensored_total} of "
                 f"{n_seeds_total}). Even on a FAIL/still-censoring-dominated outcome, "
                 f"pct_right_censored_pooled ({pooled_pct_censored:.3f}) and "
                 f"n_uncensored_deaths_total are the load-bearing scientific readouts -- this "
                 f"is a characterization run, not a hypothesis test. See module docstring "
                 f"'SLEEP-CADENCE DESIGN NOTE' for why zero within-eval sleep cycles is the "
                 f"EXPECTED reading here, not a defect. claim_ids=[]; does not weight "
                 f"governance."),
    }

    summary_markdown = f"""# V3-EXQ-920 -- Uncensored Survival-to-Death Fishtank, TRUE Single-Life Design

**Status:** {outcome} (diagnostic characterization run -- not scored against any claim)
**Purpose:** TRUE single-continuous-life ({n_seeds_total} independent lives, one per seed,
each up to {EVAL_STEPS} steps with NO segment-boundary respawn) characterization of the
health_depleted death-time distribution on the 906b ecology tier. Supersedes V3-EXQ-912's
segment-count-scaling workaround: that run's own "SUBSTRATE READINESS FINDING" section
named the per-episode step-cap literal as a substrate gap; this run consumes the now-landed
`max_episode_steps` kwarg (SD-FISHTANK-MAX-EPISODE-STEPS, ree-v3 9d3d148ff8) to implement
option (a) from organism_lifespan_development_review_906_lineage_2026-08-10.md Section 10
item 1 literally, as originally described, rather than as an approximation.

- harm-pathway train steps (total): {total_harm_steps}
- z_goal activated at eval: {z_goal_activated}
- eval steps (total): {total_steps}  across {n_seeds_total} single-life continuous segments
- **lives: {n_seeds_total} total -- uncensored (genuine) deaths: {n_uncensored_total} -- censored (hit {EVAL_STEPS}-step ceiling): {n_censored_total} -- pct_right_censored: {pooled_pct_censored:.3f}**
  (compare V3-EXQ-912's segment-count-scaling design: 93.3% right-censored at n=60 segments/seed x 1 seed)
- uncensored survival times (steps, one per genuinely-died seed): min={metrics['uncensored_survival_min']} median={metrics['uncensored_survival_median']} mean={metrics['uncensored_survival_mean']} max={metrics['uncensored_survival_max']}
- cumulative lived steps (sum of realized_steps across all seeds' single lives): {total_cumulative_lived_steps}
- events: block={total_block} limb_damage={total_limb_damage} external_hazard={total_external_hazard} world_rule_shift={total_world_rule_shift}
- sleep cycles fired (during eval): {total_sleep_cycles} (EXPECTED 0 -- see module docstring "SLEEP-CADENCE DESIGN NOTE": a single-episode eval has no non-zero segment boundary to trigger sleep_loop.notify_episode_end())
- freeze fires (eval, motor-override relaxed): {total_freeze}
- safe-spawn retries (total, at each life's single spawn): {total_spawn_retries}  (lives exhausted: {total_spawn_exhausted})

## Eval channel mean / max-std
{chr(10).join(f'- {k}: mean={metrics.get("chan_mean_"+k,0.0):.4f} max_std={chan_max_std[k]:.5f} ({"varies" if chan_nondegen[k] else "FLAT"})' for k in chan_keys)}

## Per-seed single-life outcome
{chr(10).join(f'- seed {r["seed"]}: realized_steps={r["realized_steps"]} done_cause={r["done_cause"] or "(incomplete)"} kept_full_in_log={r["kept_full"]}' for r in seed_results)}

## Lifetime affective occupancy (per seed, non-gating, SENT-2 hygiene -- see module docstring)
{chr(10).join(f'- seed {r["seed"]}: n_measured={r["occupancy"].get("n_lived_steps_measured")} frac_dread_above_p75={r["occupancy"].get("frac_steps_dread_above_p75")} frac_z_harm_a_above_p75={r["occupancy"].get("frac_steps_z_harm_a_above_p75")} frac_harm_event={r["occupancy"].get("frac_harm_event")} frac_in_reef={r["occupancy"].get("frac_in_reef")}' for r in seed_results)}

The `_episode_log.json` companion is THINNED per the module docstring ("EPISODE-LOG SIZE
MANAGEMENT") -- full per-step records are kept only for the first {FULL_LOG_FIRST_K_SEEDS}
seeds and every seed whose life genuinely ended health_depleted; other censored lives are
stored as a per-life summary. Every statistic in THIS manifest is computed from the full,
unthinned in-memory data, so thinning does not bias any reported number.

## For a future reader (or `/failure-autopsy`) on THIS run

If `n_uncensored_deaths_total` is near zero despite the 40x step-budget calibration in the
module docstring, either this ecology tier's death rate has drifted since V3-EXQ-912, or
within-life competence genuinely improves with sustained exposure enough to suppress death
well past 500 steps (organism_lifespan_development_review_906_lineage_2026-08-10.md Section
7's own flagged-but-unresolved question) -- this run's own death-time distribution, where
deaths occur, is now direct within-life-competence-vs-time evidence with no segment-boundary
confound. If `pct_right_censored_pooled` stays high, the next lever is a larger EVAL_STEPS
ceiling (a driver-level config change, not a substrate gap) -- not a return to segment-count
scaling. Separately, this run's design surfaces (does not resolve) whether REE's sleep
substrate should trigger on elapsed time/fatigue within an unbroken life rather than only at
experimenter-imposed recording-chunk boundaries -- see module docstring "SLEEP-CADENCE DESIGN
NOTE".
"""

    first_env_config = seed_results[0].get("env_config", {}) if seed_results else {}
    episode_log = {
        "experiment_type": EXPERIMENT_TYPE,
        "phase": "single_life_uncensored_survival_characterization",
        "toroidal": bool(first_env_config.get("toroidal", False)),
        "env_config": first_env_config,
        "seeds": [{"seed": r["seed"], "life": r.get("life_stored", {})} for r in seed_results],
    }

    return {
        "status": outcome, "outcome": outcome, "metrics": metrics,
        "summary_markdown": summary_markdown, "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE, "evidence_direction": "non_contributory",
        "experiment_type": EXPERIMENT_TYPE, "interpretation": interpretation,
        "episode_log": episode_log, "agents": agents,
        "lifetime_affective_occupancy": lifetime_affective_occupancy,
        "config": first_env_config,
        "supersedes": None,
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
