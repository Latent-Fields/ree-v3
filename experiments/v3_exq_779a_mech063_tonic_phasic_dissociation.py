"""V3-EXQ-779a: MECH-063 sub-claim (ii) tonic-vs-phasic behavioural DISSOCIATION.

Claim:    MECH-063 (control plane retains orthogonal tonic/phasic axes rather than
          collapsing into one scalar) -- SUB-CLAIM (ii): each control axis carries
          BOTH a slow TONIC baseline AND a fast PHASIC event-burst as independent,
          independently-toggleable degrees of freedom on a comparable readout.
Purpose:  evidence (tests MECH-063 sub-claim ii directly).
Supersedes: V3-EXQ-779 (same scientific question; implementation fix only).

WHY 779a (adjudicated by failure_autopsy_MECH-063-777-779-cluster_2026-07-18):
V3-EXQ-779 FAILed with evidence_direction non_contributory (CORRECT) but self-routed
substrate_not_ready_requeue (a MISLABEL). The substrate was NOT the problem:
  - the arms DID set the sharp source (config.phasic_burst.signal_source =
    "instantaneous_pe"), and
  - bursts FIRED in every PHASIC-ON cell (burst_level_max = 1.0; n_event_ticks
    19/15/18/12/6/5/19/17/22/22 across the ten P1 cells).
The failed precondition phasic_fires_real_events (measured 5, threshold 10) was a MIN
ACROSS CELLS sourced ENTIRELY from ONE cell -- seed 23 / T1P1, which ran 19 env steps of
a 900-step budget. sample_sufficiency (19 vs 20) had the same single origin. Root cause:
SAMPLE STARVATION. Seeds 11/17/23 ran at 2-28% of the fixed 3 x 300 episode budget while
29/37 ran at 100% (untrained agent in a hazard-terminating env; `if r.done: break` with no
continuation). The same seed-survival ladder reproduced in V3-EXQ-777, so it is a property
of the env x seed, not of this probe. mean_dR_phasic (-0.083) was computed over ~5 event
ticks and carried no weight.

THE THREE FIXES IN 779a (nothing else about the design changes):

 1. SAMPLE-DRIVEN STOPPING (replaces the fixed 3 x 300 episode budget). Each cell
    auto-resets across episodes and runs until it reaches its sample TARGETS --
    TARGET_SELECTS fresh E3 selections for every cell, plus TARGET_EVENT_TICKS
    event-window ticks and TARGET_QUIESCENT_TICKS quiescent ticks for PHASIC-ON cells --
    bounded by MAX_ENV_STEPS_PER_CELL / MAX_EPISODES_PER_CELL so runtime stays finite.
    Targets are set well ABOVE the readiness floors (MIN_*), so a cell that stops on a
    cap still normally clears readiness, and the transient estimate is no longer computed
    over ~5 ticks. Realised counts (n_env_steps, n_episodes, episode_lengths,
    rollout_stop_reason) are recorded per cell, via the shared
    experiments/_lib/sample_driven_rollout.py helper (autopsy routing (c)) rather than a
    bespoke loop, so the whole 2x2 telemetry-probe family gets the same fix. The readiness FLOORS are unchanged from 779 --
    same pre-registered bar, now guaranteed by construction rather than by seed luck.

 2. DISTINCT SAMPLE-STARVATION SELF-ROUTE. When the unmet precondition is a SAMPLE COUNT
    (phasic_fires_real_events / both_partitions_populated / sample_sufficiency) rather
    than a substrate-CAPABILITY check (tonic_axis_live / baseline_entropy_headroom), the
    label is `sample_starvation_requeue`, NOT `substrate_not_ready_requeue`. Every
    min-across-cells precondition also reports its OFFENDING CELL (seed + arm + that
    cell's realised step/episode counts), so a reader sees on sight which cell vetoed
    readiness and why. 779's label sent readers hunting for a missing SD-069 capability
    that was present and firing; this is the single highest-value fix.

 3. SURPRISE-EMA TELEMETRY (checkability for the drift-desensitisation hypothesis). The
    SD-069 event test is purely RELATIVE: s_t >= trigger_ratio * max(ema_baseline,
    trigger_floor). Under background_drift_enabled the chronic drift elevates the surprise
    EMA baseline, which can desensitise a purely relative trigger (trigger_floor is an
    absolute floor on the BASELINE, guarding a ~0 stream -- it is NOT an absolute-level
    trigger). An absolute-level companion trigger would be a ree_core substrate change and
    is deliberately OUT OF SCOPE here; instead every cell records the surprise EMA and the
    raw per-tick surprise alongside burst_level (means, medians, series summaries, and
    surprise_over_ema_mean), so the desensitisation reading is checkable straight from the
    manifest and can be adjudicated before any substrate work is proposed.

DESIGN (unchanged from 779): 2x2 factorial telemetry probe, NO gradient training, NO
synthetic signals.
  Factor TONIC  (T) = use_noise_floor  in {OFF, ON}  (MECH-313 SUSTAINED every-tick
                      temperature lift; NF_ALPHA / NF_MIN_T)
  Factor PHASIC (P) = use_phasic_burst in {OFF, ON}  (SD-069 EVENT-LOCKED transient
                      temperature delta; PHASIC-ON sets signal_source="instantaneous_pe")
  4 arms x SEEDS seeds. Shared across ALL arms: use_control_vector_logging=True
  (read-only telemetry, bit-identical), hippocampal.use_action_class_scaffold_candidates
  =True, and a VOLATILITY-ENABLED env (CausalGridWorldV2 background_drift_enabled=True,
  n_drift_sources=3, drift_policy=random_walk) so genuine per-tick PE spikes exist for
  the phasic detector to fire on.

READOUT (per fresh E3 selection; candidates captured via a read-only wrapper on
agent.generate_trajectories aligned with agent.e3.last_precommit_probs):
  E = normalised Shannon entropy of last_precommit_probs (nats / ln K); softmax spread.
Each tick is classified by the phasic burst envelope (agent._last_control_vector
["phasic_burst"]["burst_level"]):
  EVENT-WINDOW tick : burst_level  > EVENT_LEVEL_FLOOR (the phasic delta is behaviourally
                      active this tick -- default temp_delta NEGATIVE = sharpening).
  QUIESCENT   tick : burst_level <= EVENT_LEVEL_FLOOR (no active phasic delta).
PHASIC-OFF arms have NO event-window ticks (phasic_burst is None -> burst_level == 0 every
tick), so their transient is 0 by construction.

Per-arm readouts:
  S = E_quiescent_mean            -- the SUSTAINED baseline entropy (tonic lever moves this).
  R = E_event_mean - E_quiescent_mean  -- the EVENT-LOCKED TRANSIENT (phasic lever moves
      this; 0 for PHASIC-OFF arms, negative for PHASIC-ON since phasic sharpens).

AGGREGATION (per seed): the two main effects on each readout (averaged over the other
factor):
  dS_tonic  = mean_T1(S) - mean_T0(S)   (noise_floor ON-OFF on the sustained baseline)
  dS_phasic = mean_P1(S) - mean_P0(S)   (phasic ON-OFF on the sustained baseline -> ~0)
  dR_tonic  = mean_T1(R) - mean_T0(R)   (noise_floor ON-OFF on the transient -> ~0)
  dR_phasic = mean_P1(R) - mean_P0(R)   (phasic ON-OFF on the transient -> non-zero)

DOUBLE DISSOCIATION (pre-registered; supports MECH-063 sub-claim ii):
  C1 (TONIC owns the SUSTAINED baseline): |dS_tonic| >= SUSTAINED_MARGIN
       AND |dS_tonic| >= DOMINANCE_K * |dR_tonic|   (tonic moves sustained, not transient).
  C2 (PHASIC owns the EVENT-LOCKED TRANSIENT): |dR_phasic| >= TRANSIENT_MARGIN
       AND |dR_phasic| >= DOMINANCE_K * |dS_phasic|  (phasic moves transient, not baseline).
  Load-bearing verdict = C1 AND C2 on >= MIN_SEEDS seeds, robust (mean margin exceeds its
  own cross-seed SD). Honest note: the dS_phasic~0 leg is PARTLY structural (on quiescent
  ticks the PHASIC-ON effective temperature equals the PHASIC-OFF one by construction), so
  the genuinely falsifiable, load-bearing content is (a) tonic actually lifts the sustained
  baseline (dS_tonic non-trivial), (b) phasic actually produces an event-locked entropy
  transient (dR_phasic non-trivial), and (c) tonic does NOT fake a transient (dR_tonic~0).

P0 READINESS (self-routes a REQUEUE label -- NEVER a claim verdict):
  SAMPLE-kind preconditions (unmet -> `sample_starvation_requeue`):
    R1 phasic fires  : PHASIC-ON cells have >= MIN_EVENT_TICKS event-window ticks summed
                       across episodes (the regulator n_events counter resets per episode
                       via agent.reset -> phasic_burst.reset, so tick counts are
                       accumulated during the rollout, NOT read from get_state at the end).
    R2 both partitions: PHASIC-ON cells also have >= MIN_QUIESCENT_TICKS quiescent ticks
                       (so R is computable).
    R4 samples       : every cell has >= MIN_SELECTS fresh E3 selections.
  CAPABILITY-kind preconditions (unmet -> `substrate_not_ready_requeue`):
    R3 tonic live    : TONIC-ON cells mean noise_floor_temp_lift >= TEMP_LIFT_FLOOR.
    R5 headroom      : T0P0 baseline entropy in (E_SAT_LOW, E_SAT_HIGH) so both a tonic
                       lift (up) and a phasic sharpening (down) have room to move.
Below any precondition -> outcome FAIL, evidence_direction non_contributory,
non_degenerate False. Capability failures take label precedence over sample failures (a
missing capability is the more consequential finding); every min-across-cells precondition
carries `offending_cell` regardless.

VERDICT (pre-registered constants; not derived post-hoc):
  readiness unmet                    -> FAIL / non_contributory (requeue).
  readiness met AND C1 AND C2 robust -> PASS / supports (tonic and phasic are independent
     tonic-baseline vs phasic-transient degrees of freedom on one readout -- MECH-063 ii).
  readiness met AND (C1 AND C2) unmet -> FAIL / weakens (the two levers do not dissociate
     into a sustained-baseline vs event-transient split on a comparable readout).

MECH-094: trains nothing, writes no memory, no replay -- N/A (waking select only).
"""

from __future__ import annotations

import argparse
import math
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
from experiments._harness import StepHarness  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.sample_driven_rollout import (  # noqa: E402
    SELF_ROUTE_SAMPLE_STARVATION,
    RolloutBudget,
    RolloutOutcome,
    TickContext,
    run_cell_until_samples,
    starvation_selfroute,
)
from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_779a_mech063_tonic_phasic_dissociation"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS: List[str] = ["MECH-063"]
SUPERSEDES = "v3_exq_779_mech063_tonic_phasic_dissociation"

# ---- Pre-registered constants (fixed before the run; not derived post-hoc) ----
SEEDS = [11, 17, 23, 29, 37]
MIN_SEEDS = 4                 # criteria must hold on >= this many of the 5 seeds

# ---- Sample-driven stopping (779a fix 1; replaces 779's fixed 3 x 300 budget) ----
# TARGETS: a cell keeps auto-resetting across episodes until it reaches these.
TARGET_SELECTS = 800          # fresh E3 selections, EVERY cell (equalises the S estimate
                              # across PHASIC-ON and PHASIC-OFF arms)
TARGET_EVENT_TICKS = 30       # event-window ticks, PHASIC-ON cells only
TARGET_QUIESCENT_TICKS = 200  # quiescent ticks, PHASIC-ON cells only
# CAPS: bound runtime. A cell that stops on a cap records rollout_stop_reason and is
# still scored -- the readiness FLOORS below are what decide contributory-vs-requeue.
MAX_ENV_STEPS_PER_CELL = 2400
MAX_EPISODES_PER_CELL = 120
# Per-episode step cap, unchanged from 779, so the episode structure (and therefore the
# phasic EMA's per-episode reset cadence, via agent.reset -> phasic_burst.reset) matches
# the superseded design. 779a changes HOW MANY episodes run, not what an episode is.
STEPS_PER_EPISODE = 300
EPISODES_PER_RUN = MAX_EPISODES_PER_CELL  # denominator for [train] progress prints

ENV_SIZE = 8
ENV_HAZARDS = 2
ENV_RESOURCES = 3
# Volatility so genuine per-tick PE spikes exist for the phasic detector.
ENV_DRIFT_SOURCES = 3
ENV_DRIFT_POLICY = "random_walk"

# TONIC axis (MECH-313 noise_floor). Base E3 temperature is 1.0 (StepHarness passes
# temperature=1.0); effective = max(1.0 + alpha, min_T) = 2.0 -> a sustained +1.0 lift.
NF_ALPHA = 1.0
NF_MIN_T = 2.0

# PHASIC axis (SD-069 phasic_surprise_burst). Sharp source; trigger unchanged from 779
# (the ratio trigger is UNDER TEST for drift-desensitisation, so it is NOT re-tuned here --
# see fix 3: the surprise-EMA telemetry makes desensitisation checkable from the manifest).
PHASIC_SOURCE = "instantaneous_pe"
PHASIC_TRIGGER_RATIO = 1.2
PHASIC_EMA_DECAY = 0.1
PHASIC_TEMP_DELTA = -0.5        # NEGATIVE = phasic sharpening (LC-NE phasic gain increase)
PHASIC_DECAY = 0.5
PHASIC_TRIGGER_FLOOR = 1e-6
PHASIC_MIN_T = 0.1
# Burst-level above which the phasic temperature delta is behaviourally active
# (scopes the event window; below this the decaying envelope is treated as quiescent).
EVENT_LEVEL_FLOOR = 0.05

# Readiness thresholds (UNCHANGED from 779 -- same pre-registered bar).
MIN_SELECTS = 20              # R4: fresh E3 selections per cell
MIN_EVENT_TICKS = 10          # R1: event-window ticks in PHASIC-ON cells
MIN_QUIESCENT_TICKS = 10      # R2: quiescent ticks in PHASIC-ON cells
TEMP_LIFT_FLOOR = 0.5         # R3: noise_floor_temp_lift in TONIC-ON cells
E_SAT_LOW = 0.02              # R5: baseline entropy floor (headroom to sharpen down)
E_SAT_HIGH = 0.98             # R5: baseline entropy ceiling (headroom to lift up)

# Verdict thresholds (normalised entropy is in [0, 1]).
SUSTAINED_MARGIN = 0.05      # C1: min |tonic effect on sustained baseline entropy|
TRANSIENT_MARGIN = 0.02      # C2: min |phasic effect on event-locked transient|
DOMINANCE_K = 2.0            # each axis moves its OWN readout >= K x the cross-readout

# Invariant of 779a fix 1: the stopping TARGETS must dominate the readiness FLOORS, so a
# cell that stops on "floors_met" clears readiness by construction. (Only --dry-run
# shrinks the targets below the floors; that is why a dry-run legitimately self-routes
# sample_starvation_requeue.)
assert TARGET_SELECTS >= MIN_SELECTS
assert TARGET_EVENT_TICKS >= MIN_EVENT_TICKS
assert TARGET_QUIESCENT_TICKS >= MIN_QUIESCENT_TICKS

ARMS = ["T0P0", "T1P0", "T0P1", "T1P1"]  # (noise_floor, phasic_burst)
_ARM_FLAGS = {
    "T0P0": (False, False),
    "T1P0": (True, False),
    "T0P1": (False, True),
    "T1P1": (True, True),
}

# Self-route labels. The sample-starvation label is the shared-helper constant, so the
# whole 2x2 telemetry-probe family emits one vocabulary.
LABEL_SAMPLE_STARVED = SELF_ROUTE_SAMPLE_STARVATION
LABEL_SUBSTRATE_NOT_READY = "substrate_not_ready_requeue"


def _mk_env() -> CausalGridWorldV2:
    return CausalGridWorldV2(
        size=ENV_SIZE,
        num_hazards=ENV_HAZARDS,
        num_resources=ENV_RESOURCES,
        background_drift_enabled=True,
        n_drift_sources=ENV_DRIFT_SOURCES,
        drift_policy=ENV_DRIFT_POLICY,
    )


def _build_config(arm: str, env: CausalGridWorldV2) -> REEConfig:
    use_nf, use_pb = _ARM_FLAGS[arm]
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
    )
    # Shared substrate operating settings (identical across all four arms).
    cfg.use_control_vector_logging = True
    cfg.hippocampal.use_action_class_scaffold_candidates = True
    # TONIC axis.
    cfg.use_noise_floor = use_nf
    if use_nf:
        cfg.noise_floor_alpha = NF_ALPHA
        cfg.noise_floor_min_temperature = NF_MIN_T
    # PHASIC axis.
    cfg.use_phasic_burst = use_pb
    if use_pb:
        cfg.phasic_burst_signal_source = PHASIC_SOURCE
        cfg.phasic_burst_trigger_ratio = PHASIC_TRIGGER_RATIO
        cfg.phasic_burst_surprise_ema_decay = PHASIC_EMA_DECAY
        cfg.phasic_burst_temp_delta = PHASIC_TEMP_DELTA
        cfg.phasic_burst_decay = PHASIC_DECAY
        cfg.phasic_burst_trigger_floor = PHASIC_TRIGGER_FLOOR
        cfg.phasic_burst_min_temperature = PHASIC_MIN_T
    return cfg


def _config_slice(arm: str) -> Dict[str, Any]:
    """Fingerprint config slice: env + shared operating settings + this arm's
    control flags + the sample-driven stopping rule. Declares only what the
    cell's build+collect path reads."""
    use_nf, use_pb = _ARM_FLAGS[arm]
    sl: Dict[str, Any] = {
        "env_size": ENV_SIZE,
        "env_hazards": ENV_HAZARDS,
        "env_resources": ENV_RESOURCES,
        "env_drift_sources": ENV_DRIFT_SOURCES,
        "env_drift_policy": ENV_DRIFT_POLICY,
        "stopping_rule": "sample_driven_v1",
        "target_selects": TARGET_SELECTS,
        "target_event_ticks": TARGET_EVENT_TICKS,
        "target_quiescent_ticks": TARGET_QUIESCENT_TICKS,
        "max_env_steps_per_cell": MAX_ENV_STEPS_PER_CELL,
        "max_episodes_per_cell": MAX_EPISODES_PER_CELL,
        "steps_per_episode": STEPS_PER_EPISODE,
        "use_control_vector_logging": True,
        "use_action_class_scaffold_candidates": True,
        "use_noise_floor": use_nf,
        "use_phasic_burst": use_pb,
    }
    if use_nf:
        sl.update(noise_floor_alpha=NF_ALPHA, noise_floor_min_temperature=NF_MIN_T)
    if use_pb:
        sl.update(
            phasic_burst_signal_source=PHASIC_SOURCE,
            phasic_burst_trigger_ratio=PHASIC_TRIGGER_RATIO,
            phasic_burst_surprise_ema_decay=PHASIC_EMA_DECAY,
            phasic_burst_temp_delta=PHASIC_TEMP_DELTA,
            phasic_burst_decay=PHASIC_DECAY,
            phasic_burst_trigger_floor=PHASIC_TRIGGER_FLOOR,
            phasic_burst_min_temperature=PHASIC_MIN_T,
            event_level_floor=EVENT_LEVEL_FLOOR,
        )
    return sl


def _norm_entropy(probs: torch.Tensor) -> Optional[float]:
    """Normalised Shannon entropy (nats / ln K) of a candidate softmax."""
    p = probs.detach().reshape(-1).float()
    k = int(p.numel())
    if k < 2:
        return None
    p = p[p > 0]
    h = float(-(p * p.log()).sum().item())
    return h / math.log(k)


def _mean(xs: List[float]) -> float:
    return float(statistics.fmean(xs)) if xs else 0.0


def _median(xs: List[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def _budget(arm: str) -> RolloutBudget:
    """Sample-driven stopping rule (779a fix 1), denominated in the SAME units as
    the readiness gates.

    EVERY cell must reach TARGET_SELECTS (so the sustained-baseline estimate S is
    equally powered in PHASIC-ON and PHASIC-OFF arms, which is what the dS_tonic
    contrast compares). PHASIC-ON cells must ALSO populate both tick partitions, so
    the event-locked transient R is estimated from a real sample rather than the
    handful of ticks a short-lived seed happened to yield.

    PHASIC-OFF arms carry NO event/quiescent floors: burst_level is 0 every tick by
    construction there, so an event floor could never be met and would burn the whole
    step cap while reporting false starvation (the helper's documented trap).
    """
    floors: Dict[str, int] = {"selections": TARGET_SELECTS}
    if _ARM_FLAGS[arm][1]:  # PHASIC-ON
        floors["event_ticks"] = TARGET_EVENT_TICKS
        floors["quiescent_ticks"] = TARGET_QUIESCENT_TICKS
    return RolloutBudget(
        sample_floors=floors,
        max_env_steps=MAX_ENV_STEPS_PER_CELL,
        steps_per_episode=STEPS_PER_EPISODE,
        max_episodes=MAX_EPISODES_PER_CELL,
    )


def _run_cell(arm: str, seed: int) -> Tuple[Dict[str, Any], RolloutOutcome]:
    """One (arm, seed) cell: sample-driven telemetry rollout, no gradient training.

    Auto-resets across episodes until the cell's sample targets are met or a cap is
    hit. This replaces 779's fixed episode budget, under which a hazard-terminating
    env gave a 40x spread in per-cell yield across one seed set and let a single
    19-step cell veto the whole run's readiness.
    """
    with arm_cell(
        seed,
        config_slice=_config_slice(arm),
        script_path=_THIS,
        config_slice_declared=True,
        include_driver_script_in_hash=False,  # cross-driver reusable mint
    ) as cell:
        env = _mk_env()
        cfg = _build_config(arm, env)
        agent = REEAgent(cfg)

        # Read-only wrapper to capture the candidate list E3 selects over, so a
        # fresh-selection is detected against agent.e3.last_precommit_probs.
        captured: Dict[str, Any] = {"cands": None}
        _orig_gen = agent.generate_trajectories

        def _gen_capture(*a: Any, **k: Any) -> Any:
            cands = _orig_gen(*a, **k)
            captured["cands"] = cands
            return cands

        agent.generate_trajectories = _gen_capture  # type: ignore[assignment]

        harness = StepHarness(agent, env, train_mode=True, seed=seed)

        e_event: List[float] = []       # entropy on event-window ticks
        e_quiescent: List[float] = []   # entropy on quiescent ticks
        templift_vals: List[float] = []
        burst_levels: List[float] = []
        # 779a fix 3: surprise-EMA telemetry (drift-desensitisation checkability).
        surprise_vals: List[float] = []
        surprise_ema_vals: List[float] = []
        surprise_over_ema: List[float] = []
        episode_lengths: List[int] = []

        def _observe(ctx: TickContext) -> Optional[Dict[str, int]]:
            """Accumulate this cell's readouts; return per-readout increments.

            The shared helper owns stopping and counting; this callback owns all
            data accumulation (entropy partitions, telemetry series).
            """
            if ctx.step_in_episode == 0:
                episode_lengths.append(0)
                # Runner progress line. Denominator is the loop bound
                # MAX_EPISODES_PER_CELL (== EPISODES_PER_RUN in the queue entry).
                print(
                    f"  [train] {arm} seed={seed} "
                    f"ep {ctx.episode_index + 1}/{EPISODES_PER_RUN} "
                    f"env_steps={ctx.n_env_steps}/{MAX_ENV_STEPS_PER_CELL} "
                    f"e3_selects={len(e_event) + len(e_quiescent)}/{TARGET_SELECTS} "
                    f"event_ticks={len(e_event)} quiescent_ticks={len(e_quiescent)}",
                    flush=True,
                )
            episode_lengths[-1] += 1

            probs = ctx.probs
            if not (ctx.fresh and probs is not None and int(probs.numel()) >= 2):
                return None
            ent = _norm_entropy(probs)
            if ent is None:
                return None

            cv = agent._last_control_vector or {}
            pb = cv.get("phasic_burst", {}) or {}
            gv = cv.get("G_vigor", {}) or {}
            blevel = float(pb.get("burst_level", 0.0))
            burst_levels.append(blevel)
            templift_vals.append(float(gv.get("noise_floor_temp_lift", 0.0)))
            # Read the regulator's own EMA state (read-only; the control vector
            # does not carry it). PHASIC-OFF cells have no regulator.
            reg = getattr(agent, "phasic_burst", None)
            if reg is not None:
                st = reg.get_state()
                s_t = float(st.get("last_surprise", 0.0))
                s_ema = float(st.get("surprise_ema", 0.0))
                surprise_vals.append(s_t)
                surprise_ema_vals.append(s_ema)
                eff = max(s_ema, PHASIC_TRIGGER_FLOOR)
                surprise_over_ema.append(s_t / eff if eff > 0.0 else 0.0)

            if blevel > EVENT_LEVEL_FLOOR:
                e_event.append(ent)
                return {"selections": 1, "event_ticks": 1}
            e_quiescent.append(ent)
            return {"selections": 1, "quiescent_ticks": 1}

        outcome: RolloutOutcome = run_cell_until_samples(
            env=env,
            agent=agent,
            harness=harness,
            budget=_budget(arm),
            observe=_observe,
            progress_label=f"{arm} seed={seed}",
        )
        n_selects = len(e_event) + len(e_quiescent)

        s_quiescent = _mean(e_quiescent)     # SUSTAINED baseline
        s_event = _mean(e_event)
        # EVENT-LOCKED TRANSIENT: 0 when there are no event-window ticks
        # (PHASIC-OFF arms, by construction).
        transient = (s_event - s_quiescent) if e_event else 0.0

        row: Dict[str, Any] = {
            "arm": arm,
            "seed": seed,
            "use_noise_floor": _ARM_FLAGS[arm][0],
            "use_phasic_burst": _ARM_FLAGS[arm][1],
            # Realised sampling (779a fix 1: recorded, not assumed). n_env_steps /
            # n_episodes / rollout_stop_reason / rollout_floors_met /
            # rollout_sample_floors / rollout_sample_counts come from the shared
            # helper via as_manifest_fields() below.
            "episode_lengths": episode_lengths,
            "n_e3_selects": n_selects,
            "n_event_ticks": len(e_event),
            "n_quiescent_ticks": len(e_quiescent),
            "S_sustained_entropy": s_quiescent,
            "E_event_entropy": s_event,
            "R_transient": transient,
            "noise_floor_temp_lift_mean": _mean(templift_vals),
            "burst_level_mean": _mean(burst_levels),
            "burst_level_max": float(max(burst_levels)) if burst_levels else 0.0,
            # 779a fix 3: burst level recorded AGAINST the surprise EMA it is
            # triggered from, so drift-desensitisation of the purely relative
            # trigger is checkable straight from the manifest.
            "surprise_mean": _mean(surprise_vals),
            "surprise_median": _median(surprise_vals),
            "surprise_ema_mean": _mean(surprise_ema_vals),
            "surprise_ema_median": _median(surprise_ema_vals),
            "surprise_over_ema_mean": _mean(surprise_over_ema),
            "surprise_over_ema_median": _median(surprise_over_ema),
            "trigger_ratio": PHASIC_TRIGGER_RATIO,
            "event_rate": (len(e_event) / n_selects) if n_selects else 0.0,
            # Per-tick series retained for generous recording / reanalysis.
            "E_event_series": e_event,
            "E_quiescent_series": e_quiescent,
            "surprise_series": surprise_vals,
            "surprise_ema_series": surprise_ema_vals,
        }
        row.update(outcome.as_manifest_fields())
        cell.stamp(row)
    return row, outcome


# ----------------------------------------------------------------------
# Aggregation: 2x2 double dissociation of the tonic/phasic control axes.
# ----------------------------------------------------------------------
def _pooled_std(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    return float(statistics.pstdev(vals))


def _seed_effects(rows_by_arm: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Main effects of TONIC and PHASIC on the sustained baseline (S) and the
    event-locked transient (R)."""
    def _S(arm: str) -> float:
        return float(rows_by_arm[arm]["S_sustained_entropy"])

    def _R(arm: str) -> float:
        return float(rows_by_arm[arm]["R_transient"])

    # TONIC = noise_floor ON - OFF, averaged over the PHASIC factor.
    dS_tonic = ((_S("T1P0") + _S("T1P1")) / 2.0) - ((_S("T0P0") + _S("T0P1")) / 2.0)
    dR_tonic = ((_R("T1P0") + _R("T1P1")) / 2.0) - ((_R("T0P0") + _R("T0P1")) / 2.0)
    # PHASIC = phasic ON - OFF, averaged over the TONIC factor.
    dS_phasic = ((_S("T0P1") + _S("T1P1")) / 2.0) - ((_S("T0P0") + _S("T1P0")) / 2.0)
    dR_phasic = ((_R("T0P1") + _R("T1P1")) / 2.0) - ((_R("T0P0") + _R("T1P0")) / 2.0)

    # C1: tonic owns the sustained baseline.
    c1 = (abs(dS_tonic) >= SUSTAINED_MARGIN) and (
        abs(dS_tonic) >= DOMINANCE_K * abs(dR_tonic)
    )
    # C2: phasic owns the event-locked transient.
    c2 = (abs(dR_phasic) >= TRANSIENT_MARGIN) and (
        abs(dR_phasic) >= DOMINANCE_K * abs(dS_phasic)
    )
    return {
        "dS_tonic": dS_tonic,
        "dR_tonic": dR_tonic,
        "dS_phasic": dS_phasic,
        "dR_phasic": dR_phasic,
        "C1_tonic_owns_sustained": bool(c1),
        "C2_phasic_owns_transient": bool(c2),
        "dissociation": bool(c1 and c2),
    }


def _worst_cell(
    rows: List[Dict[str, Any]], key: str
) -> Tuple[float, Optional[Dict[str, Any]]]:
    """Min-across-cells value PLUS the offending cell that produced it.

    779a fix 2: 779 reported `phasic_fires_real_events: measured 5, threshold 10` with
    no indication that the 5 came from ONE 19-step cell (seed 23 / T1P1). Reporting the
    offending cell alongside the min makes a starved-cell veto legible on sight.
    """
    if not rows:
        return 0.0, None
    worst = min(rows, key=lambda r: r[key])
    return float(worst[key]), {
        "seed": worst["seed"],
        "arm": worst["arm"],
        "value": worst[key],
        "n_env_steps": worst["n_env_steps"],
        "n_episodes": worst["n_episodes"],
        "stop_reason": worst["rollout_stop_reason"],
        "sample_floors_met": worst["rollout_floors_met"],
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    global TARGET_SELECTS, TARGET_EVENT_TICKS, TARGET_QUIESCENT_TICKS
    global MAX_ENV_STEPS_PER_CELL, MAX_EPISODES_PER_CELL, EPISODES_PER_RUN
    global STEPS_PER_EPISODE
    seeds = SEEDS[:2] if dry_run else SEEDS
    if dry_run:
        TARGET_SELECTS = 40
        TARGET_EVENT_TICKS = 3
        TARGET_QUIESCENT_TICKS = 10
        MAX_ENV_STEPS_PER_CELL = 120
        MAX_EPISODES_PER_CELL = 6
        STEPS_PER_EPISODE = 40
        EPISODES_PER_RUN = MAX_EPISODES_PER_CELL

    rows: List[Dict[str, Any]] = []
    rollout_cells: List[Dict[str, Any]] = []
    for seed in seeds:
        for arm in ARMS:
            print(f"Seed {seed} Condition {arm}", flush=True)
            row, rollout = _run_cell(arm, seed)
            rows.append(row)
            rollout_cells.append({"arm": arm, "seed": seed, "outcome": rollout})
            print(f"verdict: {'PASS' if row['n_e3_selects'] > 0 else 'FAIL'}", flush=True)

    # Per-seed double-dissociation analysis.
    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        by_arm = {r["arm"]: r for r in rows if r["seed"] == seed}
        if len(by_arm) != len(ARMS):
            continue
        eff = _seed_effects(by_arm)
        eff["seed"] = seed
        per_seed.append(eff)

    # ---- Readiness preconditions (self-route requeue if unmet) ----
    p1_rows = [r for r in rows if r["use_phasic_burst"]]   # PHASIC-ON arms
    t1_rows = [r for r in rows if r["use_noise_floor"]]    # TONIC-ON arms
    baseline_rows = [r for r in rows if r["arm"] == "T0P0"]

    r1_val, r1_cell = _worst_cell(p1_rows, "n_event_ticks")
    r2_val, r2_cell = _worst_cell(p1_rows, "n_quiescent_ticks")
    r4_val, r4_cell = _worst_cell(rows, "n_e3_selects")

    r1_phasic_fires = bool(p1_rows) and r1_val >= MIN_EVENT_TICKS
    r2_both_partitions = bool(p1_rows) and r2_val >= MIN_QUIESCENT_TICKS
    r3_tonic_live = bool(t1_rows) and all(
        r["noise_floor_temp_lift_mean"] >= TEMP_LIFT_FLOOR for r in t1_rows
    )
    r4_samples = bool(rows) and r4_val >= MIN_SELECTS
    r5_headroom = bool(baseline_rows) and all(
        E_SAT_LOW < r["S_sustained_entropy"] < E_SAT_HIGH for r in baseline_rows
    )
    # 779a fix 2: separate SAMPLE-kind from CAPABILITY-kind readiness.
    sample_unmet = [
        name for name, ok in (
            ("phasic_fires_real_events", r1_phasic_fires),
            ("both_partitions_populated", r2_both_partitions),
            ("sample_sufficiency", r4_samples),
        ) if not ok
    ]
    capability_unmet = [
        name for name, ok in (
            ("tonic_axis_live", r3_tonic_live),
            ("baseline_entropy_headroom", r5_headroom),
        ) if not ok
    ]
    readiness_met = not sample_unmet and not capability_unmet

    # ---- Load-bearing criterion: double dissociation (C1 AND C2) ----
    diss = [a["dissociation"] for a in per_seed]
    seeds_diss = sum(1 for d in diss if d)
    diss_seed_count = seeds_diss >= min(MIN_SEEDS, len(seeds))
    # Robustness: mean tonic-sustained and phasic-transient effects exceed their
    # own cross-seed SD (effect is not noise).
    dS_tonic_all = [a["dS_tonic"] for a in per_seed]
    dR_phasic_all = [a["dR_phasic"] for a in per_seed]
    mean_dS_tonic = statistics.fmean(dS_tonic_all) if dS_tonic_all else 0.0
    mean_dR_phasic = statistics.fmean(dR_phasic_all) if dR_phasic_all else 0.0
    robust = (
        abs(mean_dS_tonic) - _pooled_std(dS_tonic_all) > 0.0
        and abs(mean_dR_phasic) - _pooled_std(dR_phasic_all) > 0.0
    )
    dissociation = bool(diss_seed_count and robust)

    non_degenerate = bool(readiness_met and len(per_seed) >= 1)
    degeneracy_reason = None

    # Shared-helper shortfall record: which cells missed their stopping TARGETS and by
    # how much. Informational when readiness (the lower MIN_* floors) is still met; it
    # becomes the self-route evidence when a sample-kind precondition is unmet.
    sampling_shortfall = starvation_selfroute(rollout_cells)

    if not readiness_met:
        outcome = "FAIL"
        direction = "non_contributory"
        # Capability failures take label precedence: a genuinely missing/inactive
        # regulator is the more consequential finding. Sample-count failures get
        # their own label so a reader is NOT sent hunting for absent substrate
        # (the exact mislabel V3-EXQ-779 committed).
        if capability_unmet:
            label = LABEL_SUBSTRATE_NOT_READY
            degeneracy_reason = (
                "substrate capability precondition unmet: "
                + ", ".join(capability_unmet)
            )
        else:
            label = LABEL_SAMPLE_STARVED
            offender = r1_cell if not r1_phasic_fires else (
                r2_cell if not r2_both_partitions else r4_cell
            )
            degeneracy_reason = (
                "sample-count precondition unmet (substrate present and active): "
                + ", ".join(sample_unmet)
                + (f"; offending cell seed={offender['seed']} arm={offender['arm']} "
                   f"n_env_steps={offender['n_env_steps']} "
                   f"stop_reason={offender['stop_reason']}" if offender else "")
            )
        non_degenerate = False
    elif dissociation:
        outcome = "PASS"
        direction = "supports"
        label = "tonic_phasic_double_dissociation"
    else:
        outcome = "FAIL"
        direction = "weakens"
        label = "tonic_phasic_no_dissociation"

    interpretation = {
        "label": label,
        "sampling_shortfall": sampling_shortfall,
        "readiness_unmet_sample_kind": sample_unmet,
        "readiness_unmet_capability_kind": capability_unmet,
        "preconditions": [
            {"name": "phasic_fires_real_events",
             "kind": "sample",
             "control": "PHASIC-ON cells: event-window ticks summed across episodes "
                        "(min across cells)",
             "measured": r1_val,
             "threshold": MIN_EVENT_TICKS, "met": bool(r1_phasic_fires),
             "offending_cell": r1_cell},
            {"name": "both_partitions_populated",
             "kind": "sample",
             "control": "PHASIC-ON cells: quiescent ticks (so the transient is "
                        "computable; min across cells)",
             "measured": r2_val,
             "threshold": MIN_QUIESCENT_TICKS, "met": bool(r2_both_partitions),
             "offending_cell": r2_cell},
            {"name": "tonic_axis_live",
             "kind": "capability",
             "control": "TONIC-ON cells: noise_floor_temp_lift",
             "measured": (statistics.fmean([r["noise_floor_temp_lift_mean"] for r in t1_rows]) if t1_rows else 0.0),
             "threshold": TEMP_LIFT_FLOOR, "met": bool(r3_tonic_live)},
            {"name": "sample_sufficiency",
             "kind": "sample",
             "control": "min fresh E3 selections over cells",
             "measured": r4_val,
             "threshold": MIN_SELECTS, "met": bool(r4_samples),
             "offending_cell": r4_cell},
            {"name": "baseline_entropy_headroom",
             "kind": "capability",
             "control": "T0P0 sustained entropy strictly inside (E_SAT_LOW, E_SAT_HIGH)",
             "measured": (statistics.fmean([r["S_sustained_entropy"] for r in baseline_rows]) if baseline_rows else 0.0),
             "threshold": E_SAT_HIGH, "direction": "upper", "met": bool(r5_headroom)},
        ],
        "criteria": [
            {"name": "double_dissociation_C1_and_C2", "load_bearing": True, "passed": bool(dissociation)},
        ],
        "criteria_non_degenerate": {
            # Non-degenerate iff the load-bearing effects carry real cross-seed
            # signal (not all-pinned): tonic-sustained OR phasic-transient effect
            # has non-zero spread across seeds, on >= MIN_SEEDS seeds.
            "double_dissociation_C1_and_C2": bool(
                len(per_seed) >= min(MIN_SEEDS, len(seeds))
                and (_pooled_std(dS_tonic_all) > 0.0 or _pooled_std(dR_phasic_all) > 0.0)
            ),
        },
        "summary": (
            "MECH-063 sub-claim (ii) tonic-vs-phasic dissociation on the E3 softmax "
            "temperature. The TONIC lever (MECH-313 noise_floor) moves the SUSTAINED "
            "(quiescent-tick) baseline entropy; the PHASIC lever (SD-069 phasic_surprise_"
            "burst, sharp instantaneous_pe source) moves an EVENT-LOCKED entropy TRANSIENT; "
            "cross-legs ~0. PASS = both hold (independent tonic-baseline vs phasic-transient "
            "degrees of freedom on one readout); FAIL/weakens = they do not dissociate; "
            "FAIL/non_contributory = the run could not test the claim -- labelled "
            "sample_starvation_requeue when a SAMPLE COUNT was short (substrate present "
            "and firing; offending cell named) and substrate_not_ready_requeue only when a "
            "substrate CAPABILITY check failed. Supersedes V3-EXQ-779, whose fixed episode "
            "budget let a single 19-step cell veto readiness under the latter label."
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
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "supersedes": SUPERSEDES,
        "outcome": outcome,
        "evidence_direction": direction,
        "dry_run": bool(dry_run),
        "non_degenerate": non_degenerate,
        "interpretation": interpretation,
        "ethics_preflight": ethics_preflight,
        "acceptance": {
            "readiness_met": readiness_met,
            "double_dissociation": dissociation,
            "diss_seed_count": seeds_diss,
            "min_seeds": min(MIN_SEEDS, len(seeds)),
            "mean_dS_tonic": mean_dS_tonic,
            "mean_dR_phasic": mean_dR_phasic,
            "sd_dS_tonic": _pooled_std(dS_tonic_all),
            "sd_dR_phasic": _pooled_std(dR_phasic_all),
            "robust": robust,
            "SUSTAINED_MARGIN": SUSTAINED_MARGIN,
            "TRANSIENT_MARGIN": TRANSIENT_MARGIN,
            "DOMINANCE_K": DOMINANCE_K,
        },
        # 779a: realised sampling summary, so a reader can see at a glance whether the
        # sample-driven stopping rule actually equalised yield across cells.
        "sampling_summary": {
            "stopping_rule": "sample_driven_v1",
            "target_selects": TARGET_SELECTS,
            "target_event_ticks": TARGET_EVENT_TICKS,
            "target_quiescent_ticks": TARGET_QUIESCENT_TICKS,
            "max_env_steps_per_cell": MAX_ENV_STEPS_PER_CELL,
            "max_episodes_per_cell": MAX_EPISODES_PER_CELL,
            "cells_floors_met": sum(1 for r in rows if r["rollout_floors_met"]),
            "n_cells": len(rows),
            "stop_reasons": {
                reason: sum(1 for r in rows if r["rollout_stop_reason"] == reason)
                for reason in sorted({r["rollout_stop_reason"] for r in rows})
            },
            "min_n_e3_selects": (min(r["n_e3_selects"] for r in rows) if rows else 0),
            "max_n_e3_selects": (max(r["n_e3_selects"] for r in rows) if rows else 0),
            "min_n_event_ticks_phasic_on": (
                min(r["n_event_ticks"] for r in p1_rows) if p1_rows else 0
            ),
            "total_env_steps": sum(r["n_env_steps"] for r in rows),
        },
        "per_seed": per_seed,
        "arm_results": rows,
        "notes": (
            "MECH-063 sub-claim (ii) tonic/phasic split behavioural dissociation. "
            "SUPERSEDES V3-EXQ-779 (same scientific question; implementation fix per "
            "failure_autopsy_MECH-063-777-779-cluster_2026-07-18). Three fixes: (1) "
            "sample-driven stopping -- each cell auto-resets across episodes until it "
            "reaches TARGET_SELECTS (all cells) plus TARGET_EVENT_TICKS / "
            "TARGET_QUIESCENT_TICKS (PHASIC-ON), capped by MAX_ENV_STEPS_PER_CELL, "
            "replacing 779's fixed 3 x 300 episode budget under which 3 of 5 seeds ran at "
            "2-28% of budget; (2) a distinct sample_starvation_requeue self-route with the "
            "OFFENDING CELL named, so a short-sample veto is no longer mislabelled as a "
            "missing SD-069 capability (779 reported measured 5 vs threshold 10 sourced "
            "entirely from one 19-step cell while burst_level_max was 1.0 in every "
            "PHASIC-ON cell); (3) surprise-EMA telemetry recorded against burst_level so "
            "drift-desensitisation of the purely relative trigger_ratio is checkable from "
            "the manifest (an absolute-level companion trigger would be a ree_core change "
            "and is out of scope here). TONIC = MECH-313 noise_floor (sustained "
            "temperature lift); PHASIC = SD-069 phasic_surprise_burst with the sharp "
            "instantaneous_pe source. Complements V3-EXQ-777/777a (sub-claim i). "
            "GOV-REUSE-1: the decisive readout (event-window-partitioned entropy transient "
            "under the sharp phasic source, at an adequate per-cell sample) exists in NO "
            "recorded manifest -- 779 is the only run of this family and its own cells are "
            "the starved data this run replaces -> run. Re-derive brake: 1 "
            "non_contributory autopsy on MECH-063 (this cluster), below the threshold of "
            "2 -> not braked; the autopsy explicitly permits this re-queue because it "
            "fixes the sampling model rather than nudging a threshold."
        ),
    }
    if non_degenerate is False and degeneracy_reason:
        manifest["degeneracy_reason"] = degeneracy_reason

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t_start = time.perf_counter()
    manifest = run_experiment(dry_run=args.dry_run)

    out_dir = _REE_V3.parent / "REE_assembly" / "evidence" / "experiments"
    full_config = {
        "seeds": SEEDS,
        "stopping_rule": "sample_driven_v1",
        "target_selects": TARGET_SELECTS,
        "target_event_ticks": TARGET_EVENT_TICKS,
        "target_quiescent_ticks": TARGET_QUIESCENT_TICKS,
        "max_env_steps_per_cell": MAX_ENV_STEPS_PER_CELL,
        "max_episodes_per_cell": MAX_EPISODES_PER_CELL,
        "steps_per_episode": STEPS_PER_EPISODE,
        "env": {
            "size": ENV_SIZE, "num_hazards": ENV_HAZARDS, "num_resources": ENV_RESOURCES,
            "background_drift_enabled": True, "n_drift_sources": ENV_DRIFT_SOURCES,
            "drift_policy": ENV_DRIFT_POLICY,
        },
        "tonic_noise_floor": {"alpha": NF_ALPHA, "min_temperature": NF_MIN_T},
        "phasic_burst": {
            "signal_source": PHASIC_SOURCE, "trigger_ratio": PHASIC_TRIGGER_RATIO,
            "surprise_ema_decay": PHASIC_EMA_DECAY, "temp_delta": PHASIC_TEMP_DELTA,
            "decay": PHASIC_DECAY, "trigger_floor": PHASIC_TRIGGER_FLOOR,
            "min_temperature": PHASIC_MIN_T, "event_level_floor": EVENT_LEVEL_FLOOR,
        },
        "thresholds": {
            "MIN_SEEDS": MIN_SEEDS, "MIN_SELECTS": MIN_SELECTS,
            "MIN_EVENT_TICKS": MIN_EVENT_TICKS, "MIN_QUIESCENT_TICKS": MIN_QUIESCENT_TICKS,
            "TEMP_LIFT_FLOOR": TEMP_LIFT_FLOOR, "E_SAT_LOW": E_SAT_LOW, "E_SAT_HIGH": E_SAT_HIGH,
            "SUSTAINED_MARGIN": SUSTAINED_MARGIN, "TRANSIENT_MARGIN": TRANSIENT_MARGIN,
            "DOMINANCE_K": DOMINANCE_K,
        },
    }

    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=args.dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=_THIS,
        started_at=t_start,
    )

    # NOTE: use a non-"verdict:" prefix here -- the runner counts "verdict:" lines
    # as completed runs, and the per-cell prints already emit exactly
    # seeds x conditions of them.
    print(f"FINAL_OUTCOME: {manifest['outcome']}", flush=True)
    print(f"  label: {manifest['interpretation']['label']}", flush=True)
    print(f"  readiness_met: {manifest['acceptance']['readiness_met']}", flush=True)
    _samp = manifest["sampling_summary"]
    print(f"  cells_floors_met: {_samp['cells_floors_met']}/{_samp['n_cells']} "
          f"stop_reasons={_samp['stop_reasons']}", flush=True)
    print(f"  min_selects={_samp['min_n_e3_selects']} "
          f"min_event_ticks_P1={_samp['min_n_event_ticks_phasic_on']}", flush=True)
    print(f"  mean_dS_tonic: {manifest['acceptance']['mean_dS_tonic']:.3f} "
          f"(margin {SUSTAINED_MARGIN})", flush=True)
    print(f"  mean_dR_phasic: {manifest['acceptance']['mean_dR_phasic']:.3f} "
          f"(margin {TRANSIENT_MARGIN})", flush=True)
    print(f"Result written to: {out_path}", flush=True)

    emit_outcome(
        outcome=manifest["outcome"] if manifest["outcome"] in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(out_path),
        dry_run=args.dry_run,
    )
