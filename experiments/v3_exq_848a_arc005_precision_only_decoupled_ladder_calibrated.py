"""V3-EXQ-848a: ARC-005 -- does the CALIBRATED dACC goal-readout channel route precision?

Successor to V3-EXQ-848 (bug-fix / calibration lettered iteration, same design, same
pre-registered criterion). claim_ids = [ARC-005] ONLY.

=== WHY THIS SUCCESSOR EXISTS (848's silent-zero gap + the landed calibration fix) ===
848 laddered channels 1 (5-HT) and 2 (phasic-burst temperature) while holding channels 3/4
fixed at L0, and set every prerequisite for channel 1's NEW dACC-consume pathway
(use_mech_consume=True, cfg.goal.z_goal_enabled=True, candidate_summary_source=
"e2_world_forward") -- EXCEPT `dacc_goal_readout_weight`, which 848's `_agent_kwargs()`
never set at all, so it silently defaulted to 0.0 (`DACCConfig.dacc_goal_readout_weight:
float = 0.0`). The dACC goal-readout term added to `score_bias` in
`DACCtoE3Adapter.forward()` is gated by `grw != 0.0 and gr is not None`
(ree_core/cingulate/dacc.py) -- with `grw==0.0` the term contributed EXACTLY ZERO to every
candidate's score on every tick of 848's run, regardless of how informative
`candidate_goal_proximity` actually was. 848 landed FAIL/mixed (4/10 units satisfied,
rho up to 0.866 on some units) -- a real, non-null signal, but attributable entirely to
channel 1's PRE-EXISTING serotonin-mediated pathway (tonic_5ht_baseline ->
z_goal_seeding_gain + wanting_floor, unrelated to dACC-consume) and to channel 2, NOT to
the new dACC goal-readout mechanism this successor exists to test. 848's own docstring
attributed the (never-actually-tested) dACC-consume null to "dacc_adapter's response to a
genuine post-fix goal_proximity signal is still ~1e-8 in scale" -- that characterisation
came from a SEPARATE ad-hoc instrumentation probe run during 848's authoring session
(chip-20260731-arc005-802-precision-anomaly), not from 848's own queued driver, whose
`dacc_goal_readout_weight` omission means the pathway never fired at all in the run that
actually produced 848's manifest.

A follow-up direct probe (instrumenting `REEAgent.select_action` on 848's own driver
config, session IGW-20260801-199 / arc005_dacc_adapter_goal_proximity_training) found TWO
independent, non-training gaps, both fixed on `main` in commit `4b79d18d44`
(SD-057 L7 amend, ree-v3/CLAUDE.md):
  (1) `dacc_goal_readout_weight` must be set explicitly -- confirmed above. V3-EXQ-637's
      phase-2-on cell used `dacc_goal_readout_weight=0.5`; this successor reuses that
      precedent value.
  (2) Raw `candidate_goal_proximity` is NOT floor-pinned or degenerate (measured range
      ~0.3-0.8 across candidates, genuine non-degenerate per-candidate-set spread
      ~0.003-0.03) -- but that achieved spread is small relative to the nominal [0,1]
      range because `GoalState.goal_proximity()`'s MSE-sum distance is dominated by
      ||z_world||^2 (candidate-summary operating norm ~0.5-1.5) rather than genuine
      goal-relative displacement (z_goal's operating norm ~0.08 when active) -- a
      units/calibration mismatch, NOT an untrained or broken consumer (`DACCtoE3Adapter`
      has no `nn.Parameter` anywhere; "training" was a category error in the original
      substrate_queue hint). `DACCConfig.dacc_goal_readout_normalize` (new, default False,
      bit-identical when off) rescales `candidate_goal_proximity` to the candidate SET's
      own [0,1] range via per-candidate-set min-max BEFORE the `dacc_goal_readout_weight`
      multiply, so what reaches `score_bias` is the set's relative proximity spread
      (what actually matters for influencing an argmin) rather than its calibration-
      diluted absolute magnitude.

=== WHAT THIS EXPERIMENT IS, AND IS NOT ===
This is the FIRST run in which the dACC goal-readout channel is genuinely LIVE and
CALIBRATED -- every other design element (channel-1/2-only ladder, channels 3/4 fixed at
L0, content sets, arena, schedule, seeds, the pre-registered C_PRECISION_MONOTONICITY
criterion and its thresholds) is UNCHANGED from 848, per the "bug fix / minor tweak to an
existing experiment" convention (same scientific question -- does the control plane route
precision, decoupled from occupancy -- with a corrected, previously-inert configuration).
The ONLY diffs vs 848's `_agent_kwargs()` are the two new flags:
    dacc_goal_readout_weight=0.5           (V3-EXQ-637 precedent value; 848 omitted this)
    dacc_goal_readout_normalize=True        (new flag; landed this session, default off)
TRACK B (channel 2 under an uncommitted-only regime) and TRACK C (retraining dacc_adapter
-- since it has no parameters, this was always a category error, now corrected in the
commit message itself) remain out of scope, exactly as in 848.

The mode-occupancy side of ARC-005 does NOT need re-testing here either: 802's C1/C3
results were clean and strong and are not in question; GAP-B (V3-EXQ-846) already covers
per-channel occupancy attribution. This experiment is PRECISION-ONLY, exactly like 848.

=== DESIGN: SINGLE-FACTOR 3-LEVEL LADDER ON CHANNELS 1+2 ONLY (unchanged from 848) ===
Factor   channel level (1+2 only)   L0 (substrate defaults) / L1 / L2, laddered together
Held fixed EVERY cell               channels 3+4 at their L0 substrate-default values
Content  set A / set B (independent units for the monotonicity criterion, not a
                          dissociation factor)

  cells: L0_A L1_A L2_A L0_B L1_B L2_B  (6 cells x 5 seeds = 30 cells; same as 848/802)

THE TWO CHANNELS UNDER TEST (channels 3+4 explicitly excluded, see SCOPE):
  1. 5-HT rigidity     serotonin.tonic_5ht_baseline        0.50 -> 1.00 (unchanged from 848)
  2. phasic gain       phasic_burst_temp_delta            -0.10 -> -1.00 (unchanged from 848)
CALIBRATION FIX (new relative to 848; the only design delta):
  dacc_goal_readout_weight=0.5, dacc_goal_readout_normalize=True
  (use_mech_consume=True, cfg.goal.z_goal_enabled=True,
   candidate_summary_source="e2_world_forward" are unchanged from 848 -- already correct)

DV: E3 precision readout -- e3.current_precision = 1/(running_variance + 1e-6), recorded as
log10 precision, via the CANONICAL update_residue() / update_z_goal() path -- identical to
848 (see 848's own docstring for why this path is canonical vs 802's manual recomputation).

=== DV-SYMMETRY INVARIANCE DECLARATION (mandatory, per arm) ===
  channel 1 (5-HT / dACC goal-readout): reaches dACC's score_bias via
    candidate_goal_proximity, which is now BOTH non-zero-weighted (0.5, vs 848's silent
    0.0) AND per-candidate-set-normalised (removing the units/calibration dilution). This
    is NOT a broadcast constant and NOT a monotone reparameterisation of anything
    downstream of it (the normalisation is itself the per-candidate-set signal, not a
    global rescale applied after selection) -- so it is NOT invariant under the precision
    DV in principle, and this run is a genuine test of whether the calibrated pathway
    produces a detectable monotonicity signal, not a pre-registered-null confirmation.
    Channel 1 ALSO still carries its pre-existing serotonin-mediated pathway (unchanged
    from 848), so a positive result cannot by itself attribute causation to the
    goal-readout channel specifically vs. the serotonin channel -- see PRE-REGISTERED
    CRITERION below for why this is acceptable given the criterion this experiment
    reuses unchanged from 848.
  channel 2 (phasic gain): a genuine E3 SOFTMAX TEMPERATURE -- a monotone
    reparameterisation, and therefore provably argmax-invariant under DETERMINISTIC
    (committed) selection, which this experiment runs under (identical to 848, ~97%
    commitment rate measured on this config family). NOT invariant in principle under an
    uncommitted/stochastic selection regime (TRACK B, not built here, same as 848). Its
    contribution to any observed rho here is expected to remain null for the same
    architectural reason as 848 -- unchanged by this successor's calibration fix, which
    touches channel 1 only.
  channels 3/4: excluded from this experiment entirely (SCOPE) -- never varied, so no
    DV-symmetry claim is made about them here; see 802/GAP-B for their (occupancy)
    authority.

=== PRE-REGISTERED CRITERION (constant below; nothing derived from the run; UNCHANGED from
848 -- this successor's purpose is to correct the CONFIGURATION, not the criterion) ===
Per (content, seed) unit, over the L0<L1<L2 ladder of channels 1+2:
  C_PRECISION_MONOTONICITY (load-bearing)
    |Spearman rho of log10-precision vs level| >= 0.60   [UNSIGNED -- channels 1 and 2 touch
      precision with opposite-signed intuitions, and channel 1 itself now carries two
      distinct sub-pathways (serotonin + calibrated dACC goal-readout) with no shared sign
      convention pre-registered between them; same UNSIGNED convention as 802's C2 and
      848's C_PRECISION_MONOTONICITY]
    satisfied in >= 7 of the 10 (content x seed) units

  PASS  = C_PRECISION_MONOTONICITY                -> evidence_direction supports
  FAIL, all 10 units rho in [-0.1, 0.1] (near-zero, not just sub-threshold)
                                                    -> evidence_direction non_contributory
    (would indicate the calibrated dACC goal-readout channel STILL fails to produce a
     detectable monotonicity signal even once genuinely live and calibrated -- an
     informative negative about this specific pathway, distinct from 848's confounded
     silent-zero non-result; NOT evidence the wiring is broken given the DV-symmetry
     declaration for channel 2 above and the pre-existing weak/mixed serotonin signal 848
     already recorded)
  FAIL, otherwise (some real but sub-threshold trend) -> evidence_direction mixed

This experiment does NOT attempt to isolate the calibrated dACC goal-readout channel's
marginal contribution from channel 1's pre-existing serotonin-mediated contribution (that
would require a further decoupled design -- e.g. ladder channel 1's dACC-consume flags
alone while holding serotonin fixed -- explicitly OUT OF SCOPE here, noted as candidate
follow-on in the queue entry). This run answers the narrower, pre-registered question 848
was designed to answer but could not, due to the omitted weight: does the decoupled
channel-1/2 ladder, with the dACC goal-readout pathway now genuinely engaged and
calibrated, produce a detectable precision-monotonicity signal under normal (committed)
operation?

=== NON-DEGENERACY (else substrate_not_ready_requeue, NOT a verdict; UNCHANGED from 848) ===
Per-cell gates via experiments/_lib/precondition_gate.py:
  P1 channel_state_delta_vs_L0 > 0.05  applies_to PERTURBED CELLS ONLY (channels 1+2 moved
       vs the same-content L0 cell). Verifies the settings TOOK EFFECT.
  P2 precision_cross_seed_sd > 1e-6    all cells. A precision readout permanently floor-
       pinned across DIFFERENT SEEDS would be a substrate-readiness failure distinct from
       -- and prior to -- the monotonicity question this experiment asks.
  P3 n_salience_ticks >= 150           all cells. The coordinator must have actually ticked
       enough for a genuine precision trajectory to accumulate.

BUILD-TIME GUARD (new in this successor, not a per-cell precondition -- a static config
invariant, guarding against a repeat of 848's exact failure mode): `_build()` asserts the
constructed `REEConfig` actually carries `dacc_goal_readout_weight == 0.5` and
`dacc_goal_readout_normalize is True` before any cell runs, raising immediately (ERROR,
not a silent pass-through) if either flag failed to thread through `REEConfig.from_dims`.
This is what 848 lacked -- an omitted kwarg silently defaulted to 0.0 with no error at any
layer.

=== SCOPE (unchanged from 848) ===
Channels 3 and 4 are DELIBERATELY held fixed at L0 in every cell and are not tested here.
TRACK B (channel 2 under uncommitted selection) and a channel-1-sub-pathway-isolating
follow-up (dACC goal-readout alone vs serotonin alone) are NOT built in this experiment.

=== NO GRADIENT TRAINING ===
Nothing is trained; no head reads a latent under a loss (DACCtoE3Adapter has no
nn.Parameter). The agent is driven in eval() exactly as the cells are constructed. A
WARMUP phase precedes measurement, unchanged from 848.

=== SAMPLE-SIZE INTEGRITY (unchanged from 848) ===
`current_mode`/`current_precision` are agent STATE, held between coordinator ticks; a
per-env-step read is a time-fraction/EMA-continuation, not pseudo-replication. The number
of genuine coordinator ticks is counted (`n_salience_ticks`) and gated (P3) rather than
inferred from the step count. UNIT OF ANALYSIS for the criterion is the SEED (n=5) x
CONTENT (n=2).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorld  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.baselines import exq802_arc005_control_plane as BASE  # noqa: E402

# ------------------------------------------------------------------ #
# Identity                                                            #
# ------------------------------------------------------------------ #
EXPERIMENT_TYPE = "v3_exq_848a_arc005_precision_only_decoupled_ladder_calibrated"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS: List[str] = ["ARC-005"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
BACKLOG_ID = None
SUPERSEDES_NOTE = (
    "Bug-fix / calibration lettered iteration of V3-EXQ-848 (same design, same "
    "pre-registered C_PRECISION_MONOTONICITY criterion). 848's _agent_kwargs() never set "
    "dacc_goal_readout_weight, so it silently defaulted to 0.0 and the dACC goal-readout "
    "term contributed EXACTLY ZERO to score_bias throughout 848's run -- the pathway 848 "
    "was designed to test was never actually engaged (see module docstring). This "
    "successor sets dacc_goal_readout_weight=0.5 (V3-EXQ-637 precedent) and "
    "dacc_goal_readout_normalize=True (new flag, ree-v3 commit 4b79d18d44, "
    "IGW-20260801-199), which corrects both the omitted weight and the units/calibration "
    "mismatch that same commit diagnosed in raw candidate_goal_proximity. 848's own "
    "manifest evidence (4/10 units satisfied, rho up to 0.866) is NOT invalidated by this "
    "fix -- it reflects channel 1's pre-existing serotonin-mediated pathway plus channel "
    "2, both unchanged here -- so 848 is cited as supersedes but its result should be "
    "read alongside this one, not discarded; governance review should judge whether 848's "
    "evidence_direction=mixed reading still holds once this run's calibrated-channel "
    "result is available."
)

OUT_DIR = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

# ------------------------------------------------------------------ #
# Pre-registered constants (NOT derived from the run's own statistics) #
# ------------------------------------------------------------------ #
SEEDS = [0, 1, 2, 3, 4]
CONTENTS = ["A", "B"]
LEVELS = BASE.CHANNEL_LEVELS                    # [0.0, 0.5, 1.0]
WARMUP_TICKS = BASE.WARMUP_TICKS                # 200
MEASURE_TICKS = BASE.MEASURE_TICKS              # 1800
TOTAL_TICKS = BASE.TOTAL_TICKS                  # 2000

DRY_WARMUP_TICKS = 10
DRY_MEASURE_TICKS = 40
DRY_SEEDS = [0, 1]

# --- calibration fix (the only design delta vs 848) ---
DACC_GOAL_READOUT_WEIGHT = 0.5     # V3-EXQ-637 phase2_on precedent value
DACC_GOAL_READOUT_NORMALIZE = True  # ree-v3 commit 4b79d18d44 (IGW-20260801-199)

# --- precision monotonicity criterion (unchanged from 848) ---
C_RHO_ABS_FLOOR = 0.60            # UNSIGNED
C_MIN_UNITS = 7                   # of 10 (content x seed)
C_NULL_BAND = 0.10                # |rho| <= this in ALL 10 units -> non_contributory, not mixed

# --- non-degeneracy floors (unchanged from 848) ---
P1_CHANNEL_DELTA_FLOOR = 0.05
P2_PRECISION_SD_FLOOR = 1e-6
P3_SALIENCE_TICK_FLOOR = 150.0
DRY_P3_SALIENCE_TICK_FLOOR = 5.0  # smoke only -- criteria are not scored on a smoke


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #
def _spearman_rho(x: List[float], y: List[float]) -> float:
    """Spearman rank correlation. 0.0 when undefined (n < 3 or a flat side)."""
    n = len(x)
    if n < 3:
        return 0.0

    def _rank(v: List[float]) -> List[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = _rank(x), _rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx <= 0.0 or dy <= 0.0:
        return 0.0
    return float(num / (dx * dy))


def _arm_id(level: float, content: str) -> str:
    return BASE.arm_id(level, content)


def _precision_channel_settings(level: float) -> Dict[str, Any]:
    """Channels 1+2 laddered by `level`; channels 3+4 held at L0 (0.0) in EVERY cell.

    Unchanged from 848 -- the decoupling design is not what this successor corrects.
    """
    laddered = BASE.channel_settings(level)
    fixed = BASE.channel_settings(0.0)
    return {
        "serotonin_seeding_gain": laddered["serotonin_seeding_gain"],
        "serotonin_wanting_floor": laddered["serotonin_wanting_floor"],
        "phasic_burst_temp_delta": laddered["phasic_burst_temp_delta"],
        # channels 3+4 -- FIXED at L0 regardless of `level`. SCOPE (see docstring).
        "salience_external_task_bias": fixed["salience_external_task_bias"],
        "pcc_stability_baseline": fixed["pcc_stability_baseline"],
    }


def _agent_kwargs(level: float) -> Dict[str, Any]:
    """REEConfig.from_dims kwargs for a channel level (obs dims added by caller).

    Identical to 848's _agent_kwargs() EXCEPT the two calibration-fix flags
    (dacc_goal_readout_weight, dacc_goal_readout_normalize) -- see module docstring
    "WHY THIS SUCCESSOR EXISTS". Everything else, including the channel-1 dACC-consume
    prerequisites 848 already set correctly (use_mech_consume, z_goal_enabled,
    candidate_summary_source), is unchanged.
    """
    ch = _precision_channel_settings(level)
    return dict(
        alpha_world=0.9,
        # --- channel 1: 5-HT (live at every level) ---
        tonic_5ht_enabled=True,
        # --- channel 2: phasic surprise burst (live at every level) ---
        use_phasic_burst=True,
        phasic_burst_temp_delta=ch["phasic_burst_temp_delta"],
        phasic_burst_signal_source="instantaneous_pe",
        phasic_burst_baseline_continuity="carry",
        # --- channels 3+4: salience coordinator + PCC-analog, FIXED at L0 ---
        use_salience_coordinator=True,
        use_dacc=True,
        dacc_weight=0.5,
        dacc_foraging_weight=0.5,
        use_aic_analog=True,
        salience_external_task_bias=ch["salience_external_task_bias"],
        use_pcc_analog=True,
        pcc_stability_baseline=ch["pcc_stability_baseline"],
        salience_use_stability_temperature=True,
        salience_temperature_mu_alpha=1.0,
        # --- substrate operating config, identical in every cell ---
        use_harm_stream=True,
        use_affective_harm_stream=True,
        use_support_preserving_cem=True,
        support_preserving_min_first_action_classes=2,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=4.0,
        # --- unchanged from 848: the channel-1 dACC-consume prerequisites ---
        use_mech_consume=True,
        z_goal_enabled=True,
        candidate_summary_source="e2_world_forward",
        # --- NEW vs 848: the calibration fix itself (848 omitted the weight) ---
        dacc_goal_readout_weight=DACC_GOAL_READOUT_WEIGHT,
        dacc_goal_readout_normalize=DACC_GOAL_READOUT_NORMALIZE,
    )


def _build(seed: int, level: float, content: str):
    """Construct env + agent for one cell, THROUGH the canonical baseline module
    for env/content (unchanged from 848/802) with the calibration-fixed agent config
    above.

    BUILD-TIME GUARD: asserts the constructed REEConfig actually carries the two
    calibration flags this successor exists to test, so an omission (848's exact
    failure mode -- a kwarg silently defaulting to 0.0/False with no error anywhere)
    raises immediately here instead of silently producing another null-by-construction
    run.
    """
    env_kw = BASE.content_env_kwargs(content, seed)
    env = CausalGridWorld(**env_kw)
    _obs, obs_dict = env.reset()

    kw: Dict[str, Any] = dict(_agent_kwargs(level))
    kw.update(
        body_obs_dim=obs_dict["body_state"].shape[-1],
        world_obs_dim=obs_dict["world_state"].shape[-1],
        action_dim=env.action_dim,
    )
    cfg = REEConfig.from_dims(**kw)
    ch = _precision_channel_settings(level)
    cfg.serotonin.gain_min = cfg.serotonin.gain_max = float(ch["serotonin_seeding_gain"])
    cfg.serotonin.floor_min = cfg.serotonin.floor_max = float(
        ch["serotonin_wanting_floor"]
    )

    assert cfg.dacc_goal_readout_weight == DACC_GOAL_READOUT_WEIGHT, (
        f"BUILD-TIME GUARD FAILED: cfg.dacc_goal_readout_weight="
        f"{cfg.dacc_goal_readout_weight!r}, expected {DACC_GOAL_READOUT_WEIGHT!r} -- "
        f"this is exactly 848's silent-zero failure mode; the calibration fix this "
        f"successor exists to test did not thread through REEConfig.from_dims."
    )
    assert cfg.dacc_goal_readout_normalize is DACC_GOAL_READOUT_NORMALIZE, (
        f"BUILD-TIME GUARD FAILED: cfg.dacc_goal_readout_normalize="
        f"{cfg.dacc_goal_readout_normalize!r}, expected {DACC_GOAL_READOUT_NORMALIZE!r}."
    )

    agent = REEAgent(cfg)
    agent.eval()

    assert agent.dacc_adapter is not None, (
        "BUILD-TIME GUARD FAILED: agent.dacc_adapter is None -- use_dacc must be True "
        "for the goal-readout term to have any bias to attach to."
    )
    assert agent.dacc_adapter.config.dacc_goal_readout_weight == DACC_GOAL_READOUT_WEIGHT, (
        "BUILD-TIME GUARD FAILED: the constructed DACCtoE3Adapter's own config does not "
        "carry dacc_goal_readout_weight -- REEAgent's DACCConfig construction site did "
        "not propagate cfg.dacc_goal_readout_weight."
    )

    return agent, env, obs_dict


def _run_cell(
    seed: int, level: float, content: str, n_warmup: int, n_measure: int,
    zg_acc: ZGoalStreamAccumulator,
) -> Dict[str, Any]:
    """Drive one (seed, level, content) cell and return its readouts.

    Unchanged from 848: uses the CANONICAL update_z_goal() / update_residue() calls.
    """
    arm = _arm_id(level, content)
    agent, env, obs_dict = _build(seed, level, content)
    n_ticks = n_warmup + n_measure

    print(f"Seed {seed} Condition {arm}", flush=True)

    log_precisions: List[float] = []
    seeding_gains: List[float] = []
    tonic_5ht: List[float] = []
    burst_levels: List[float] = []
    n_salience_ticks = 0
    n_measured_steps = 0
    n_dacc_bias_calls_start = int(agent.dacc_adapter._n_bias_calls)
    prev_sal_tick: Any = None
    last_reward = 0.0

    for tick in range(n_ticks):
        measuring = tick >= n_warmup
        with torch.no_grad():
            latent = agent.sense(
                obs_dict["body_state"].unsqueeze(0),
                obs_dict["world_state"].unsqueeze(0),
                obs_harm=obs_dict.get("harm_obs"),
                obs_harm_a=obs_dict.get("harm_obs_a"),
                obs_harm_history=obs_dict.get("harm_history"),
            )
            ticks_d = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent)
                if ticks_d["e1_tick"]
                else torch.zeros(1, agent.config.latent.world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks_d)
            # Canonical z_goal writer, BEFORE select_action (StepHarness order).
            drive_level = REEAgent.compute_drive_level(obs_dict["body_state"].unsqueeze(0))
            agent.update_z_goal(
                benefit_exposure=max(0.0, float(last_reward)), drive_level=drive_level
            )
            action = agent.select_action(candidates, ticks_d, 1.0)
        agent._step_count += 1

        cur_sal_tick = getattr(agent, "_salience_last_tick", None)
        if cur_sal_tick is not None and cur_sal_tick is not prev_sal_tick:
            n_salience_ticks += 1
        prev_sal_tick = cur_sal_tick

        act_idx = (
            int(action.argmax().item()) if isinstance(action, torch.Tensor) else int(action)
        )
        _obs, reward, done, _info, obs_dict = env.step(act_idx % env.action_dim)
        last_reward = float(reward)

        # Keep the 5-HT channel LIVE (as 848/802 did).
        agent.serotonin_step(max(0.0, float(reward)))

        # Canonical precision-update path (unchanged from 848).
        with torch.no_grad():
            agent.sense(
                obs_dict["body_state"].unsqueeze(0),
                obs_dict["world_state"].unsqueeze(0),
                obs_harm=obs_dict.get("harm_obs"),
                obs_harm_a=obs_dict.get("harm_obs_a"),
                obs_harm_history=obs_dict.get("harm_history"),
            )
            agent.update_residue(
                harm_signal=float(reward), world_delta=None,
                hypothesis_tag=False, owned=True,
            )

        if measuring:
            n_measured_steps += 1
            log_precisions.append(math.log10(max(agent.e3.current_precision, 1e-12)))
            seeding_gains.append(float(agent.serotonin.current_seeding_gain()))
            tonic_5ht.append(float(agent.serotonin.tonic_5ht))
            burst_levels.append(
                float(agent.phasic_burst.burst_level)
                if agent.phasic_burst is not None else 0.0
            )

        if done:
            _obs, obs_dict = env.reset()

        if (tick + 1) % 250 == 0 or tick == n_ticks - 1:
            print(
                f"  [train] arc005precisioncalib {arm} seed={seed} ep {tick + 1}/{n_ticks} "
                f"sal_ticks={n_salience_ticks} steps={n_measured_steps}",
                flush=True,
            )

    zg_acc.observe(agent)
    n_dacc_bias_calls = int(agent.dacc_adapter._n_bias_calls) - n_dacc_bias_calls_start

    row: Dict[str, Any] = {
        "arm_id": arm,
        "channel_level": float(level),
        "content": content,
        "seed": int(seed),
        "n_ticks": n_ticks,
        "n_warmup_ticks": n_warmup,
        "n_measured_steps": n_measured_steps,
        "n_salience_ticks": n_salience_ticks,
        "n_dacc_bias_calls": n_dacc_bias_calls,
        "log10_precision_mean": round(float(np.mean(log_precisions)), 8)
        if log_precisions else 0.0,
        "log10_precision_sd": round(float(np.std(log_precisions)), 8)
        if len(log_precisions) > 1 else 0.0,
        "realised_channel_state": {
            "seeding_gain_mean": round(float(np.mean(seeding_gains)), 8)
            if seeding_gains else 0.0,
            "tonic_5ht_mean": round(float(np.mean(tonic_5ht)), 8) if tonic_5ht else 0.0,
            "burst_level_mean": round(float(np.mean(burst_levels)), 8)
            if burst_levels else 0.0,
            "phasic_temp_delta_cfg": float(
                agent.phasic_burst.config.temp_delta
                if agent.phasic_burst is not None else 0.0
            ),
            # channels 3+4 are FIXED -- recorded for audit, not part of any criterion.
            "salience_external_task_bias_cfg_fixed_at_L0": float(
                agent.salience.config.external_task_bias
                if agent.salience is not None else 0.0
            ),
            # calibration-fix audit trail -- recorded for audit, not part of any criterion
            # (both are constant across all cells by design; see the build-time guard).
            "dacc_goal_readout_weight_cfg": float(
                agent.dacc_adapter.config.dacc_goal_readout_weight
            ),
            "dacc_goal_readout_normalize_cfg": bool(
                agent.dacc_adapter.config.dacc_goal_readout_normalize
            ),
        },
        "goal_state_active_at_end": bool(
            agent.goal_state is not None and agent.goal_state.is_active()
        ),
    }
    return row


# ------------------------------------------------------------------ #
# Precondition specs (regime-conditioned, unchanged from 848)         #
# ------------------------------------------------------------------ #
def _specs(salience_tick_floor: float) -> List[PreconditionSpec]:
    return [
        PreconditionSpec(
            name="channel_state_delta_vs_L0",
            description=(
                "normalised L1 distance of the cell's REALISED channel-1/2 state from "
                "the same-content L0 cell's"
            ),
            control="same-content level-0 cell",
            threshold=P1_CHANNEL_DELTA_FLOOR,
            applies_to=lambda ctx: float(ctx["channel_level"]) != LEVELS[0],
            applies_note=(
                "perturbed cells only -- an L0 cell compared with itself is 0 by "
                "construction, not by substrate failure"
            ),
        ),
        PreconditionSpec(
            name="precision_cross_seed_sd",
            description="cross-seed SD of the cell's log10-precision mean, at a fixed level",
            control="five independently seeded cells of the same level+content",
            threshold=P2_PRECISION_SD_FLOOR,
        ),
        PreconditionSpec(
            name="n_salience_ticks",
            description="genuine SalienceCoordinator ticks",
            control="coordinator ticking on the live selection path",
            threshold=salience_tick_floor,
        ),
    ]


def _cell_contexts() -> List[Dict[str, Any]]:
    out = []
    for content in CONTENTS:
        for level in LEVELS:
            out.append(
                {
                    "id": _arm_id(level, content),
                    "channel_level": float(level),
                    "content": content,
                }
            )
    return out


def _channel_state_vector(row: Dict[str, Any]) -> np.ndarray:
    s = row["realised_channel_state"]
    return np.array(
        [
            (float(s["seeding_gain_mean"]) - 0.90) / 0.60,
            (abs(float(s["phasic_temp_delta_cfg"])) - 0.10) / 0.90,
        ],
        dtype=float,
    )


# ------------------------------------------------------------------ #
# Analysis (unchanged from 848)                                       #
# ------------------------------------------------------------------ #
def _analyse(rows: List[Dict[str, Any]], seeds: List[int]) -> Dict[str, Any]:
    by: Dict[Tuple[float, str, int], Dict[str, Any]] = {
        (r["channel_level"], r["content"], r["seed"]): r for r in rows
    }

    units: List[Dict[str, Any]] = []
    for c in CONTENTS:
        for s in seeds:
            try:
                trip = [by[(lv, c, s)] for lv in LEVELS]
            except KeyError:
                continue
            prec = [float(r["log10_precision_mean"]) for r in trip]
            rho = _spearman_rho(list(LEVELS), prec)
            ok = abs(rho) >= C_RHO_ABS_FLOOR
            units.append(
                {"content": c, "seed": s, "rho_log10_precision": round(rho, 4),
                 "satisfied": bool(ok)}
            )
    n_satisfied = sum(1 for u in units if u["satisfied"])
    criterion_pass = n_satisfied >= C_MIN_UNITS
    all_near_zero = bool(units) and all(
        abs(u["rho_log10_precision"]) <= C_NULL_BAND for u in units
    )

    return {
        "units": units,
        "n_satisfied": n_satisfied,
        "n_units": len(units),
        "criterion_pass": criterion_pass,
        "all_near_zero_null": all_near_zero,
    }


# ------------------------------------------------------------------ #
# Driver                                                              #
# ------------------------------------------------------------------ #
def run_experiment(dry_run: bool) -> Dict[str, Any]:
    seeds = DRY_SEEDS if dry_run else SEEDS
    n_warmup = DRY_WARMUP_TICKS if dry_run else WARMUP_TICKS
    n_measure = DRY_MEASURE_TICKS if dry_run else MEASURE_TICKS
    tick_floor = DRY_P3_SALIENCE_TICK_FLOOR if dry_run else P3_SALIENCE_TICK_FLOOR

    specs = _specs(tick_floor)
    contexts = _cell_contexts()
    assert_no_structurally_unsatisfiable_gate(specs, contexts)

    zg_acc = ZGoalStreamAccumulator()
    rows: List[Dict[str, Any]] = []
    total_dacc_bias_calls = 0
    for content in CONTENTS:
        for level in LEVELS:
            for seed in seeds:
                slice_ = {
                    "lineage": "exq848_arc005_precision_only",
                    "arm_id": _arm_id(level, content),
                    "channel_level": float(level),
                    "content": content,
                    "env_kwargs": BASE.content_env_kwargs(content, seed),
                    "agent_kwargs": _agent_kwargs(level),
                    "schedule": {
                        "warmup_ticks": WARMUP_TICKS, "measure_ticks": MEASURE_TICKS,
                        "total_ticks": TOTAL_TICKS,
                    },
                }
                with arm_cell(
                    seed,
                    config_slice=slice_,
                    script_path=Path(__file__),
                    config_slice_declared=True,
                    include_driver_script_in_hash=True,
                ) as cell:
                    row = _run_cell(seed, level, content, n_warmup, n_measure, zg_acc)
                    cell.stamp(row)
                rows.append(row)
                total_dacc_bias_calls += int(row["n_dacc_bias_calls"])
                print(
                    f"verdict: {'PASS' if row['n_measured_steps'] > 0 else 'FAIL'}",
                    flush=True,
                )

    # Smoke-scale engagement check: the calibrated dACC goal-readout channel must
    # actually be invoked at least once per cell (adapter forward() is called every
    # tick regardless of dacc_goal_readout_weight, since use_dacc=True/dacc_weight=0.5
    # are already live -- so this specifically catches the adapter never being
    # constructed/called at all, not the narrower weight-omission bug the build-time
    # guard in _build() already catches at construction time).
    if total_dacc_bias_calls == 0:
        raise RuntimeError(
            "ENGAGEMENT CHECK FAILED: agent.dacc_adapter._n_bias_calls never incremented "
            "across any cell -- the dACC adapter this experiment exists to test was never "
            "invoked. This is a decisive-readout engagement failure, not a scientific "
            "result; do not interpret any rho computed alongside this condition."
        )

    by_cell: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_cell.setdefault(r["arm_id"], []).append(r)

    l0_state: Dict[str, np.ndarray] = {}
    for content in CONTENTS:
        cell0 = _arm_id(LEVELS[0], content)
        vecs = [_channel_state_vector(r) for r in by_cell.get(cell0, [])]
        l0_state[content] = (
            np.mean(np.stack(vecs), axis=0) if vecs else np.zeros(2, dtype=float)
        )

    cell_gates = []
    for ctx in contexts:
        cell_rows = by_cell.get(ctx["id"], [])
        if not cell_rows:
            continue
        prec_means = [float(r["log10_precision_mean"]) for r in cell_rows]
        prec_sd = float(statistics.stdev(prec_means)) if len(prec_means) > 1 else 0.0
        ticks_worst = min(float(r["n_salience_ticks"]) for r in cell_rows)
        vec = np.mean(np.stack([_channel_state_vector(r) for r in cell_rows]), axis=0)
        chan_delta = float(np.mean(np.abs(vec - l0_state[ctx["content"]])))
        cell_gates.append(
            evaluate_arm_gate(
                ctx["id"], ctx, specs,
                measured={
                    "channel_state_delta_vs_L0": chan_delta,
                    "precision_cross_seed_sd": prec_sd,
                    "n_salience_ticks": ticks_worst,
                },
            )
        )
    gate = aggregate_arm_gates(cell_gates)
    green = set(gate["green_arms"])
    scorable = set(c["id"] for c in contexts).issubset(green)

    analysis = _analyse(rows, seeds)

    if not scorable:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        evidence_direction = "non_contributory"
        non_degenerate = False
        degeneracy_reason = (
            f"cell(s) failed non-degeneracy gate (green: {sorted(green)} of "
            f"{sorted(c['id'] for c in contexts)})"
        )
        overall_pass = False
    else:
        overall_pass = bool(analysis["criterion_pass"])
        outcome = "PASS" if overall_pass else "FAIL"
        if overall_pass:
            label = "control_plane_routes_precision_decoupled_calibrated"
            evidence_direction = "supports"
        elif analysis["all_near_zero_null"]:
            label = "precision_channel_authority_null_even_calibrated"
            evidence_direction = "non_contributory"
        else:
            label = "precision_channel_authority_weak_calibrated"
            evidence_direction = "mixed"
        non_degenerate = True
        degeneracy_reason = ""

    criteria = [
        {"name": "C_precision_monotonicity", "load_bearing": True,
         "passed": bool(analysis["criterion_pass"])},
    ]
    criteria_nd = {"C_precision_monotonicity": scorable}

    manifest: Dict[str, Any] = {
        "run_id": None,  # filled by __main__
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "backlog_id": BACKLOG_ID,
        "supersedes_note": SUPERSEDES_NOTE,
        "outcome": outcome,
        "overall_pass": overall_pass,
        "evidence_direction": evidence_direction,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "criteria": criteria,
        "analysis": analysis,
        "per_arm_gate": gate["per_arm_gate"],
        "diagnostics": {
            "n_distinct_argmax_modes_across_design": None,  # not tested here -- see SCOPE
            "total_dacc_bias_calls": total_dacc_bias_calls,
        },
        "scorable": scorable,
        "arm_results": rows,
        "interpretation": {
            "label": label,
            "preconditions": gate["adjudication_preconditions"],
            "preconditions_scope_note": gate["per_arm_gate"]["preconditions_scope_note"],
            "criteria_non_degenerate": criteria_nd,
        },
        "pre_registered_thresholds": {
            "c_rho_abs_floor": C_RHO_ABS_FLOOR,
            "c_min_units": C_MIN_UNITS,
            "c_null_band": C_NULL_BAND,
            "p1_channel_delta_floor": P1_CHANNEL_DELTA_FLOOR,
            "p2_precision_sd_floor": P2_PRECISION_SD_FLOOR,
            "p3_salience_tick_floor": tick_floor,
            "dacc_goal_readout_weight": DACC_GOAL_READOUT_WEIGHT,
            "dacc_goal_readout_normalize": DACC_GOAL_READOUT_NORMALIZE,
        },
        "custom_information": {
            "channel_ladder": {str(lv): _precision_channel_settings(lv) for lv in LEVELS},
            "content_sets": BASE.CONTENT_SETS,
            "arena": BASE.ARENA,
            "predecessor_run_id": "v3_exq_848_arc005_precision_only_decoupled_ladder",
            "predecessor_bug": (
                "848's _agent_kwargs() never set dacc_goal_readout_weight, so it "
                "defaulted to 0.0 and the dACC goal-readout term contributed exactly "
                "zero to score_bias throughout 848's run -- the pathway 848 intended to "
                "test was never engaged. Fixed here (see module docstring)."
            ),
            "dv_symmetry_declaration": (
                "channel 1 (5-HT / dACC goal-readout): now genuinely weighted (0.5, vs "
                "848's silent 0.0) and per-candidate-set normalised -- not a broadcast "
                "constant, not a monotone reparameterisation applied after selection, so "
                "NOT invariant under the precision DV in principle. This run is a genuine "
                "test of the calibrated pathway, not a pre-registered-null confirmation. "
                "Channel 1 also still carries its pre-existing serotonin-mediated "
                "pathway (unchanged from 848), so a positive result cannot be attributed "
                "to the goal-readout channel alone -- see module docstring. channel 2 "
                "(phasic gain) is a genuine E3 softmax TEMPERATURE, a monotone "
                "reparameterisation and therefore provably argmax-invariant under the "
                "DETERMINISTIC (committed) selection this experiment runs under "
                "(~97% commitment rate measured on this config family, unchanged from "
                "848) -- a null contribution from channel 2 specifically is the "
                "architecturally expected reading for this regime, unaffected by this "
                "successor's calibration fix. Channels 3/4 are held fixed at L0 "
                "throughout and make no DV-symmetry claim here -- see SCOPE."
            ),
            "scope_note": (
                "Channels 3 (mode prior) and 4 (pcc_stability) are DELIBERATELY held "
                "fixed at their L0 values in every cell and excluded from this "
                "experiment's criterion a priori, unchanged from 848 -- see 802/GAP-B "
                "for their occupancy authority. TRACK B (channel 2 under uncommitted "
                "selection) and a follow-up isolating the calibrated dACC goal-readout "
                "channel's marginal contribution from channel 1's pre-existing "
                "serotonin-mediated pathway are explicitly OUT OF SCOPE here."
            ),
            "supersedes_848": SUPERSEDES_NOTE,
            "expected_outcome": (
                "Not pre-determined for channel 1 (this is the first genuine test of the "
                "calibrated dACC goal-readout pathway). Channel 2's contribution remains "
                "architecturally expected to stay null under committed selection, "
                "unchanged from 848. A PASS newly supports ARC-005 via the calibrated "
                "channel-1 pathway (jointly with its pre-existing serotonin component); "
                "an all-near-zero FAIL indicates the calibrated pathway still fails to "
                "produce a detectable signal even once genuinely engaged -- an "
                "informative negative, distinct from 848's confounded silent-zero "
                "non-result; a sub-threshold-but-nonzero FAIL is 'mixed', same "
                "convention as 848."
            ),
        },
        "ethics_preflight": {
            "involves_negative_valence": False,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
        },
    }
    return manifest, zg_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    from datetime import datetime

    manifest, zg_acc = run_experiment(dry_run=args.dry_run)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest["timestamp_utc"] = ts
    manifest["dry_run"] = bool(args.dry_run)

    full_config = {
        "arena": BASE.ARENA,
        "content_sets": BASE.CONTENT_SETS,
        "channel_levels": LEVELS,
        "channel_ladder": {str(lv): _precision_channel_settings(lv) for lv in LEVELS},
        "agent_kwargs_by_level": {str(lv): _agent_kwargs(lv) for lv in LEVELS},
        "schedule": {
            "warmup_ticks": DRY_WARMUP_TICKS if args.dry_run else WARMUP_TICKS,
            "measure_ticks": DRY_MEASURE_TICKS if args.dry_run else MEASURE_TICKS,
        },
        "pre_registered_thresholds": manifest["pre_registered_thresholds"],
    }
    seeds_used = DRY_SEEDS if args.dry_run else SEEDS

    stamp_recording_core(
        manifest,
        config=full_config,
        seeds=seeds_used,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=zg_acc.stats(),
    )

    out_path = write_flat_manifest(
        manifest, OUT_DIR, dry_run=args.dry_run, stamp=False
    )

    print(json.dumps({
        "run_id": manifest["run_id"],
        "outcome": manifest["outcome"],
        "evidence_direction": manifest["evidence_direction"],
        "non_degenerate": manifest["non_degenerate"],
        "n_satisfied": manifest["analysis"]["n_satisfied"],
        "n_units": manifest["analysis"]["n_units"],
        "all_near_zero_null": manifest["analysis"]["all_near_zero_null"],
        "label": manifest["interpretation"]["label"],
        "total_dacc_bias_calls": manifest["diagnostics"]["total_dacc_bias_calls"],
        "manifest": str(out_path),
    }, indent=2), flush=True)

    _outcome_raw = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
