"""V3-EXQ-952: SD-075 phasic-warmup-rescue DIAGNOSTIC (probe-gated, NOT a MECH-063 retest).

Purpose: evidence (tests MECH-063 sub-claim ii directly). -- NO, see EXPERIMENT_PURPOSE below;
this line intentionally left absent from the docstring template because this script is a
DIAGNOSTIC. It does not test MECH-063/SD-069 directly; it answers the readiness question that
gates the NEXT retest of that claim pair.

WHY THIS EXISTS INSTEAD OF A "V3-EXQ-779c" (read this before touching the 779 lineage again).

The obvious next move after V3-EXQ-779b (self-routed sample_starvation_requeue, 15/20 cells
capped at MAX_ENV_STEPS_PER_CELL=2400 before reaching their TARGET_SELECTS/TARGET_EVENT_TICKS
floors) looks like "raise the step cap and re-run" -- SUPERSEDED and REFUSED. The confirmed
`failure_autopsy_V3-EXQ-779b_2026-07-19.json` (targets[0], MECH-063) ran the re-derive brake
BEFORE that plan could be queued and REFUSED it:

    re_derive_brake.fired = true, count_including_this = 3 (prior substrate_ceiling autopsies:
    MECH-063-777-779-cluster, MECH-063-777a-779a-cluster), refused_requeue = true,
    refusal_statement: "A V3-EXQ-779c is REFUSED. No further lettered iteration of the
    tonic/phasic dissociation probe against the current phasic regulator -- the binding
    constraint is episode-length/EMA-cadence and no probe-side parameter reaches it. Build
    sd_phasic_ema_episode_continuity first."

The autopsy's own root-cause finding, confirmed against the 779b manifest: seed 23 / T0P1's
event-tick count did NOT move between 779a (6 events / 835 env steps) and 779b (6 events /
2400 env steps, 2.9x more exposure). `PhasicSurpriseBurst.reset()` clears the surprise-EMA at
every episode boundary (PHASIC_EMA_DECAY=0.1 -> ~10-tick time constant), and seed 23's episodes
average 6.9 steps -- so the EMA baseline never converges within one episode, REGARDLESS of how
many total env steps the run is given. A wider step cap cannot fix an EMA that is reset before
it ever settles. (Independently reproduced in this session, 2026-08-28: an untrained agent
under the OLD "reset" continuity mode gets 0/230 events window on a 60-step read; see the run
log for the smaller probe below.)

The named upstream substrate (`sd_phasic_ema_episode_continuity`, SD-075) IS NOW BUILT
(ree-v3 4a5139838b, 2026-07-19; `REE_assembly/evidence/planning/substrate_queue.json` entry
status=implemented) -- so per this repo's re-derive-brake protocol (CLAUDE.md Step 2.5b of
/queue-experiment: "substrate now IMPLEMENTED -> brake released -> retest is meaningful"), a
NEW-NUMBER retest against the upgraded substrate is no longer blocked outright. BUT the SD-075
implementation note itself records an unresolved, EXPLICITLY DEFERRED design question:

    "with the gate ON n_events_converged is 0 in BOTH modes [reset AND carry] because every
    event on an untrained agent lands in the first 30 ticks. The legs therefore DISAGREE about
    this cell and the build deliberately does not choose -- the autopsy target sentence admits
    both readings and choosing is a retest-design decision. ... RETEST BRAKE: MECH-063 (ii)
    stays pending_retest_after_substrate; NO V3-EXQ-779c may be queued."

In words: SD-075 gives the regulator a "carry" continuity mode (EMA baseline survives episode
boundaries) plus a convergence gate (an event only counts once the carried baseline has settled
past `phasic_burst_warmup_ticks`, DERIVE=30 lifetime ticks) -- but SD-075's OWN live smoke used
an UNTRAINED agent, and found that EVERY phasic event an untrained agent produces lands inside
the first 30 lifetime ticks, i.e. before the convergence gate ever opens. So "carry" mode alone
does not yet clear MIN_EVENT_TICKS on the CONVERGED count for an untrained agent -- the same
failure shape as "reset" mode, for a different underlying reason. The build's own author left
an UNTESTED hypothesis on the table (`experiments/_scratch/sd075_warmup_rescue_spike.py`, never
queued or run to completion): a TRAINED agent's surprise stream may keep producing genuine
relative excesses PAST tick 30, because an untrained forward model's prediction-error stream
is dominated by transient early-lifetime noise that a trained model's is not.

THIS SCRIPT answers that one question -- and ONLY that question. It is the missing fact the
substrate note names, not a MECH-063 retest, and it is scoped to be genuinely `complex
(probe-gated)` per CLAUDE.md's work-graph vocabulary: a puzzle with known rules (does warmup
training let n_events_converged clear MIN_EVENT_TICKS on the worst-case seed?), not a
mystery and not a redesign. It does not resolve the "retest-design decision" the SD-075 note
defers to governance (which continuity mode / how much warmup / which seeds the EVENTUAL
MECH-063(ii) retest should use) -- it produces the evidence that decision needs.

DESIGN. Adapted from the scratch spike (byte-identical env/phasic constants, so the read is
denominated in the same units as the 779b failure record and the SD-075 live smoke). For each
(seed, warmup_episodes) cell:
  1. Build a fresh env + REEConfig (phasic_burst_baseline_continuity="carry",
     phasic_burst_warmup_ticks=-1 [DERIVE=30] -- the SD-075 fix, held fixed across the whole
     grid so the swept variable is ONLY the agent's training exposure) + REEAgent.
  2. WARM the agent via `experiments._lib.probe_warmup.warm_agent` (SD-074's own recipe;
     `warmup_episodes` IS the swept parameter -- 0 = untrained control, matching SD-075's own
     live smoke exactly; WARMUP_EPISODES_TRAINED = a materially larger training exposure).
  3. Reinstall a FRESH `PhasicSurpriseBurst` regulator after warmup (zero lifetime-tick
     counters) so the READ phase's convergence-gate accounting is not contaminated by ticks the
     warmup phase already burned -- this is the one place a cache-hit vs cache-miss warmup path
     could otherwise desynchronise the read's tick-zero point; see `_fresh_regulator`.
  4. READ rollout (train_mode=False, eval): step until READ_MAX_ENV_STEPS or READ_MAX_EPISODES,
     recording burst_level every tick and the regulator's own `get_state()` at the end
     (n_events_converged, n_converged_ticks, n_events_prewarmup, lifetime_ticks).

READOUT. `n_events_converged` per cell -- the SAME quantity 779b's R1 precondition
(phasic_fires_real_events) reads, now denominated post-convergence-gate. MIN_EVENT_TICKS = 10,
UNCHANGED from 779b (no threshold is being retuned here; this probe asks whether the SAME bar
becomes reachable, not whether the bar should move).

INTERPRETATION (self-routed; this is a diagnostic, not a claim verdict):
  precondition R0 "regulator_fires_at_all": max burst_level across the WHOLE grid clears
      EVENT_LEVEL_FLOOR. A positive control -- if this fails, nothing below is informative
      (the regulator itself is inert under this config, a capability failure, not a warmup
      question). Self-routes `substrate_not_ready_requeue` if unmet.
  criterion "phasic_warmup_rescues_event_convergence": min-across-WARMED-seeds of
      n_events_converged >= MIN_EVENT_TICKS on the seed set tested (SEEDS). load_bearing=true.
      criteria_non_degenerate: the warmed condition's readout differs from the untrained
      control's (not both pinned at the same value -- the comparison this probe exists to make
      would otherwise be vacuous).
  label:
    "phasic_warmup_rescue_confirmed"    -- criterion holds: SD-075 carry mode + agent training
        together clear MIN_EVENT_TICKS on the converged count. A NEW-NUMBER MECH-063(ii) retest
        using carry mode + this warmup exposure is now well-motivated (still a governance design
        decision to actually queue it -- this probe supplies the evidence, not the decision).
    "phasic_warmup_rescue_insufficient" -- criterion fails on every warmed seed tested: training
        exposure alone does not rescue the convergence gate at this budget. Route to further
        /implement-substrate work on SD-075 (larger warmup budget, or the "reset" vs "carry"
        choice needs a different resolution than training) rather than another lettered probe.

SUBSTRATE-PATH OVERLAP GATE (skill step 2.5c). Two open `corrupting` substrate_queue entries
name files this driver's REEAgent imports at module level (`ree_core/agent.py`,
`ree_core/predictors/e1_deep.py`), both `status: implemented_pending_validation`:
  `mode-governance-engagement` (SalienceCoordinator.tick / external-task mode-switching box
    clamp) is not in this driver's causal path -- no mode-governance / external-task-switching
    knob is enabled anywhere in `_mk_config`; the grid is a single foraging env with no mode
    axis. Not exercised.
  `contextmemory-write-path-addressing-degeneracy` (ContextMemory.write() hard-argmin lock,
    e1_deep.py) IS potentially in path via E1's forward-model training during warmup, but
    applies IDENTICALLY across every cell in this grid (all seeds, both warmup conditions) and
    does not interact with the swept factor (warmup_episodes) or the readout (n_events_converged,
    which reads instantaneous prediction-error magnitude, not which memory slot was addressed).
    At worst it biases absolute PE levels toward the null (phasic_warmup_rescue_insufficient),
    never toward a false-positive rescue -- same reasoning as the 947/949 driver family's
    identical overlap note.
Two `degrading`-severity entries (`mech357-freeze-incompatible-pressure-mechanism`,
`SD-MECH303-THRESHOLD-SOURCING`) also name `ree_core/environment/causal_grid_world.py`; both
concern hazard-pursuit-pressure and contextual-safety-threshold subsystems this driver's config
does not touch beyond defaults (`num_hazards=2`, no safety-threshold override) -- noted per the
skill's degrading-severity handling, not blocking.

CLAIM_IDS = [] (diagnostic; excluded from MECH-063/SD-069 confidence scoring by
EXPERIMENT_PURPOSE="diagnostic"). This experiment does not test MECH-063 or SD-069 -- it tests
substrate READINESS for a future test of them, so tagging either claim here would be exactly
the claim-tagging-integrity violation GFLAG-0077 flags elsewhere in this lineage (779/779a/779b
mis-tagging their OWN direct evidence). See `interpretation.summary` for the explicit statement
that a FAIL/insufficient result here carries NO evidence against MECH-063 or SD-069 -- it is a
readiness measurement, not a claim test.

MECH-094: trains a downstream forward-model/action-value landscape via probe_warmup's
warmup_train (P0-style de-saturation only, matching SD-074's own protocol), never trains any
head on z_world/z_harm output -- no phased-training applicability. The READ phase uses
train_mode=False (eval only, no gradient step, no memory write) -- N/A for phased training.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

_THIS = Path(__file__).resolve()
_REE_V3 = _THIS.parent.parent
if str(_REE_V3) not in sys.path:
    sys.path.insert(0, str(_REE_V3))

from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.regulators.phasic_surprise_burst import (  # noqa: E402
    PhasicSurpriseBurst,
    PhasicSurpriseBurstConfig,
)
from experiments._harness import StepHarness  # noqa: E402
from experiments._lib.probe_warmup import WarmupRecipe, warm_agent  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_952_sd075_phasic_warmup_rescue_diagnostic"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS: List[str] = []
# regulator_fires_at_all reuses the exact burst_level computation already validated live in
# 779/779a/779b and this session's own untrained-agent probe (burst_level_max measured 1.00);
# it is not a new or retuned gate, so reachability is by construction, not by a threshold this
# script owns.
ANCHOR_REACHABILITY_EXEMPT = (
    "regulator_fires_at_all reuses PhasicSurpriseBurst.burst_level, already confirmed >0.05 "
    "(measured 1.00) in 779/779a/779b and this diagnostic's own untrained-agent probe run "
    "2026-08-28; not a new or retuned gate."
)

# ---- Pre-registered constants ----
SEEDS = [11, 23, 29]                    # 779b's mild-select / event-starved / severe-select
                                         # seeds -- one from each starvation category.
WARMUP_CONDITIONS = [0, 40]             # 0 = untrained control (matches SD-075 live smoke
                                         # exactly); 40 = trained (the untested leg).
STEPS_PER_EPISODE = 300                 # unchanged from 779/779a/779b/the scratch spike.

READ_MAX_ENV_STEPS = 600                # unchanged from the scratch spike defaults.
READ_MAX_EPISODES = 200

MIN_EVENT_TICKS = 10                    # 779b R1 bar -- NOT retuned here.
EVENT_LEVEL_FLOOR = 0.05                # unchanged from 779b.

ENV_SIZE = 8
ENV_HAZARDS = 2
ENV_RESOURCES = 3
ENV_DRIFT_SOURCES = 3
ENV_DRIFT_POLICY = "random_walk"

# PHASIC axis -- byte-identical to 779b except the two SD-075 fields below.
PHASIC_SOURCE = "instantaneous_pe"
PHASIC_TRIGGER_RATIO = 1.2
PHASIC_EMA_DECAY = 0.1
PHASIC_TEMP_DELTA = -0.5
PHASIC_DECAY = 0.5
PHASIC_TRIGGER_FLOOR = 1e-6
PHASIC_MIN_T = 0.1
# SD-075 fix, held FIXED across the whole grid (the swept variable is warmup only).
PHASIC_BASELINE_CONTINUITY = "carry"
PHASIC_WARMUP_TICKS = -1                 # DERIVE = ceil(3 / PHASIC_EMA_DECAY) = 30

PROGRESS_EVERY_ENV_STEPS = 100

_ZG = ZGoalStreamAccumulator()


def _mk_env() -> CausalGridWorldV2:
    return CausalGridWorldV2(
        size=ENV_SIZE,
        num_hazards=ENV_HAZARDS,
        num_resources=ENV_RESOURCES,
        background_drift_enabled=True,
        n_drift_sources=ENV_DRIFT_SOURCES,
        drift_policy=ENV_DRIFT_POLICY,
    )


def _mk_config(env: CausalGridWorldV2) -> REEConfig:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
    )
    cfg.use_control_vector_logging = True
    cfg.hippocampal.use_action_class_scaffold_candidates = True
    cfg.use_noise_floor = False
    cfg.use_phasic_burst = True
    cfg.phasic_burst_signal_source = PHASIC_SOURCE
    cfg.phasic_burst_trigger_ratio = PHASIC_TRIGGER_RATIO
    cfg.phasic_burst_surprise_ema_decay = PHASIC_EMA_DECAY
    cfg.phasic_burst_temp_delta = PHASIC_TEMP_DELTA
    cfg.phasic_burst_decay = PHASIC_DECAY
    cfg.phasic_burst_trigger_floor = PHASIC_TRIGGER_FLOOR
    cfg.phasic_burst_min_temperature = PHASIC_MIN_T
    cfg.phasic_burst_baseline_continuity = PHASIC_BASELINE_CONTINUITY
    cfg.phasic_burst_warmup_ticks = PHASIC_WARMUP_TICKS
    return cfg


def _config_slice(warmup_episodes: int) -> Dict[str, Any]:
    """Fingerprint config slice: env + phasic config + the swept warmup exposure."""
    return {
        "env_size": ENV_SIZE,
        "env_hazards": ENV_HAZARDS,
        "env_resources": ENV_RESOURCES,
        "env_drift_sources": ENV_DRIFT_SOURCES,
        "env_drift_policy": ENV_DRIFT_POLICY,
        "phasic_burst_signal_source": PHASIC_SOURCE,
        "phasic_burst_trigger_ratio": PHASIC_TRIGGER_RATIO,
        "phasic_burst_surprise_ema_decay": PHASIC_EMA_DECAY,
        "phasic_burst_temp_delta": PHASIC_TEMP_DELTA,
        "phasic_burst_decay": PHASIC_DECAY,
        "phasic_burst_trigger_floor": PHASIC_TRIGGER_FLOOR,
        "phasic_burst_min_temperature": PHASIC_MIN_T,
        "phasic_burst_baseline_continuity": PHASIC_BASELINE_CONTINUITY,
        "phasic_burst_warmup_ticks": PHASIC_WARMUP_TICKS,
        "warmup_episodes": warmup_episodes,
        "steps_per_episode": STEPS_PER_EPISODE,
        "read_max_env_steps": READ_MAX_ENV_STEPS,
        "read_max_episodes": READ_MAX_EPISODES,
    }


def _fresh_regulator(agent: REEAgent) -> None:
    """Reinstall a zero-lifetime regulator so the READ phase's convergence-gate
    accounting starts at tick zero regardless of the warmup path taken (cache-hit
    vs cache-miss warmup consume different numbers of regulator ticks otherwise)."""
    agent.phasic_burst = PhasicSurpriseBurst(
        config=PhasicSurpriseBurstConfig(
            enabled=True,
            surprise_ema_decay=PHASIC_EMA_DECAY,
            trigger_ratio=PHASIC_TRIGGER_RATIO,
            trigger_floor=PHASIC_TRIGGER_FLOOR,
            temp_delta=PHASIC_TEMP_DELTA,
            decay=PHASIC_DECAY,
            min_temperature=PHASIC_MIN_T,
            baseline_continuity=PHASIC_BASELINE_CONTINUITY,
            warmup_ticks=PHASIC_WARMUP_TICKS,
        )
    )


def _run_cell(
    seed: int, warmup_episodes: int, cell_idx: int, n_cells: int,
) -> Dict[str, Any]:
    print("Seed %d Condition warmup%d" % (seed, warmup_episodes), flush=True)
    with arm_cell(
        seed,
        config_slice=_config_slice(warmup_episodes),
        script_path=_THIS,
        config_slice_declared=True,
    ) as cell:
        env = _mk_env()
        cfg = _mk_config(env)
        agent = REEAgent(cfg)

        recipe = WarmupRecipe(num_episodes=int(warmup_episodes),
                               steps_per_episode=STEPS_PER_EPISODE)
        env_kwargs = {
            "size": ENV_SIZE, "num_hazards": ENV_HAZARDS,
            "num_resources": ENV_RESOURCES, "background_drift_enabled": True,
            "n_drift_sources": ENV_DRIFT_SOURCES, "drift_policy": ENV_DRIFT_POLICY,
        }
        print("  [warmup] seed=%d warmup_episodes=%d starting" % (seed, warmup_episodes),
              flush=True)
        warm_out = warm_agent(
            agent, env, seed=seed, recipe=recipe, env_kwargs=env_kwargs,
            label="v3_exq_952 cell%d/%d" % (cell_idx + 1, n_cells), measure=False,
        )
        _fresh_regulator(agent)

        harness = StepHarness(agent, env, train_mode=False, seed=seed)
        burst_max = 0.0
        n_event_window_ticks = 0
        ep_lengths: List[int] = []
        steps = 0
        agent.eval()
        with torch.no_grad():
            for ep in range(READ_MAX_EPISODES):
                _flat, obs_dict = env.reset()
                agent.reset()
                harness.reset()
                ep_len = 0
                for _ in range(STEPS_PER_EPISODE):
                    r = harness.step(obs_dict)
                    obs_dict = r.next_obs_dict
                    steps += 1
                    ep_len += 1
                    lvl = float(agent.phasic_burst.burst_level)
                    burst_max = max(burst_max, lvl)
                    if lvl >= EVENT_LEVEL_FLOOR:
                        n_event_window_ticks += 1
                    if (steps == 1) or (steps % PROGRESS_EVERY_ENV_STEPS == 0):
                        print(
                            "  [train] warmup%d seed=%d "
                            "ep %d/%d env-steps (episode %d) event_ticks=%d"
                            % (warmup_episodes, seed, steps, READ_MAX_ENV_STEPS,
                               ep + 1, n_event_window_ticks),
                            flush=True,
                        )
                    if r.done or steps >= READ_MAX_ENV_STEPS:
                        break
                ep_lengths.append(ep_len)
                if steps >= READ_MAX_ENV_STEPS:
                    break

        _ZG.observe(agent)
        st = agent.phasic_burst.get_state()
        row: Dict[str, Any] = {
            "seed": seed,
            "warmup_episodes": warmup_episodes,
            "warmup_cache_hit": bool(warm_out.cache_hit),
            "n_read_env_steps": steps,
            "n_read_episodes": len(ep_lengths),
            "mean_episode_len": (statistics.fmean(ep_lengths) if ep_lengths else 0.0),
            "burst_level_max": float(burst_max),
            "n_event_window_ticks": int(n_event_window_ticks),
            "lifetime_ticks": int(st["lifetime_ticks"]),
            "lifetime_episodes": int(st["lifetime_episodes"]),
            "warmup_ticks_resolved": int(st["warmup_ticks"]),
            "n_events_converged": int(st["n_events_converged"]),
            "n_converged_ticks": int(st["n_converged_ticks"]),
            "n_events_prewarmup": int(st["n_events_prewarmup"]),
            "baseline_continuity": str(st["baseline_continuity"]),
            "meets_min_event_ticks_converged": bool(
                int(st["n_events_converged"]) >= MIN_EVENT_TICKS),
        }
        cell.stamp(row)
    print("verdict: %s" % ("PASS" if row["meets_min_event_ticks_converged"] else "FAIL"),
          flush=True)
    return row


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    global SEEDS, WARMUP_CONDITIONS, READ_MAX_ENV_STEPS, READ_MAX_EPISODES
    seeds = SEEDS[:1] if dry_run else SEEDS
    warmups = [0, 2] if dry_run else WARMUP_CONDITIONS
    read_max_env_steps = 40 if dry_run else READ_MAX_ENV_STEPS
    read_max_episodes = 5 if dry_run else READ_MAX_EPISODES
    if dry_run:
        READ_MAX_ENV_STEPS = read_max_env_steps
        READ_MAX_EPISODES = read_max_episodes

    cells = [(w, s) for w in warmups for s in seeds]
    rows: List[Dict[str, Any]] = []
    for idx, (w, s) in enumerate(cells):
        rows.append(_run_cell(s, w, idx, len(cells)))

    # ---- Readiness precondition R0: the regulator fires at all (positive control) ----
    max_burst = max((r["burst_level_max"] for r in rows), default=0.0)
    r0_fires = bool(max_burst > EVENT_LEVEL_FLOOR)

    # ---- Load-bearing criterion: warmup rescues convergence-gated event counting ----
    warmed_rows = [r for r in rows if r["warmup_episodes"] == max(WARMUP_CONDITIONS)]
    control_rows = [r for r in rows if r["warmup_episodes"] == 0]
    min_warmed = min((r["n_events_converged"] for r in warmed_rows), default=0)
    min_control = min((r["n_events_converged"] for r in control_rows), default=0)
    max_control = max((r["n_events_converged"] for r in control_rows), default=0)
    worst_warmed_cell = min(warmed_rows, key=lambda r: r["n_events_converged"]) \
        if warmed_rows else None

    rescue_confirmed = bool(r0_fires and min_warmed >= MIN_EVENT_TICKS)
    non_degenerate = bool(
        warmed_rows and control_rows and
        (min_warmed != max_control or min_warmed != min_control)
    )

    if not r0_fires:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        degeneracy_reason = "R0 unmet: regulator never fired (burst_level_max <= floor) " \
                             "across the whole grid -- capability failure, not a warmup question."
        non_degenerate = False
    elif rescue_confirmed:
        outcome = "PASS"
        label = "phasic_warmup_rescue_confirmed"
        degeneracy_reason = None
    else:
        outcome = "FAIL"
        label = "phasic_warmup_rescue_insufficient"
        degeneracy_reason = None

    interpretation: Dict[str, Any] = {
        "label": label,
        "preconditions": [
            {
                "name": "regulator_fires_at_all",
                "kind": "capability",
                "control": "max burst_level_max across the whole grid (positive control)",
                "measured": max_burst,
                "threshold": EVENT_LEVEL_FLOOR,
                "direction": "lower",
                "met": bool(r0_fires),
            },
        ],
        "criteria": [
            {
                "name": "phasic_warmup_rescues_event_convergence",
                "load_bearing": True,
                "passed": bool(rescue_confirmed),
                "measured_min_warmed_n_events_converged": min_warmed,
                "threshold": MIN_EVENT_TICKS,
                "control_n_events_converged_range": [min_control, max_control],
                "offending_cell": (
                    {"seed": worst_warmed_cell["seed"],
                     "warmup_episodes": worst_warmed_cell["warmup_episodes"],
                     "n_events_converged": worst_warmed_cell["n_events_converged"]}
                    if worst_warmed_cell else None
                ),
            },
        ],
        "criteria_non_degenerate": {
            "phasic_warmup_rescues_event_convergence": non_degenerate,
        },
        "summary": (
            "DIAGNOSTIC (not a MECH-063/SD-069 claim test). Answers the retest-design "
            "question SD-075's own implementation note left open: does AGENT TRAINING let "
            "the SD-075 carry-continuity convergence gate clear MIN_EVENT_TICKS=10 on "
            "n_events_converged, where an untrained agent (SD-075's own live smoke, and this "
            "probe's warmup=0 control) gets 0 in both continuity modes? "
            "phasic_warmup_rescue_confirmed = yes (a new-number MECH-063(ii) retest using "
            "carry mode + this warmup exposure is now evidence-supported, still a governance "
            "decision to queue it -- see the module docstring). "
            "phasic_warmup_rescue_insufficient = no (training exposure alone does not rescue "
            "the gate at this budget -- route to further /implement-substrate work on SD-075, "
            "not another lettered probe of the 779 lineage, which the confirmed "
            "failure_autopsy_V3-EXQ-779b_2026-07-19 re-derive brake already REFUSES). "
            "Neither outcome is evidence for or against MECH-063 or SD-069 -- CLAIM_IDS is "
            "deliberately empty; this measures substrate readiness, not the claim."
        ),
    }

    ethics_preflight = {
        "involves_negative_valence": False,
        "involves_suffering_like_state": False,
        "involves_self_model": False,
        "involves_inescapability_or_helplessness": False,
        "involves_offline_replay_over_harm": False,
        "involves_social_mind_or_language": False,
        "involves_human_data_or_clinical_context": False,
        "decision": "allow",
    }

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest: Dict[str, Any] = {
        "schema_version": "v1",
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "outcome": outcome,
        "evidence_direction": "non_contributory",
        "dry_run": dry_run,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "interpretation": interpretation,
        "ethics_preflight": ethics_preflight,
        "arm_results": rows,
        "seeds": seeds,
        "notes": (
            "Answers the SD-075 (sd_phasic_ema_episode_continuity) retest-design question "
            "left open in REE_assembly/evidence/planning/substrate_queue.json "
            "(sd_phasic_ema_episode_continuity implementation_note): whether agent training "
            "lets the carry-continuity convergence gate clear MIN_EVENT_TICKS. Supersedes "
            "no prior run (new probe); explicitly NOT a V3-EXQ-779c, which the confirmed "
            "failure_autopsy_V3-EXQ-779b_2026-07-19 re-derive brake refuses."
        ),
    }
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result = run_experiment(dry_run=args.dry_run)
    elapsed = time.perf_counter() - t0

    out_path = write_flat_manifest(
        result,
        dry_run=args.dry_run,
        config={
            "phasic_burst_baseline_continuity": PHASIC_BASELINE_CONTINUITY,
            "phasic_burst_warmup_ticks": PHASIC_WARMUP_TICKS,
            "warmup_conditions": WARMUP_CONDITIONS,
            "read_max_env_steps": READ_MAX_ENV_STEPS,
            "read_max_episodes": READ_MAX_EPISODES,
            "min_event_ticks": MIN_EVENT_TICKS,
        },
        seeds=result["seeds"],
        script_path=_THIS,
        elapsed_seconds=elapsed,
        z_goal_stream_stats=_ZG.stats(),
    )
    print(f"outcome: {result['outcome']}")
    print(f"label: {result['interpretation']['label']}")
    print(f"manifest: {out_path}")

    emit_outcome(
        outcome=result["outcome"] if result["outcome"] in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
